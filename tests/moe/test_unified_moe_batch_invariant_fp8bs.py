# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The DeepSeekFp8 x DeepSeekFp8 pair of the batch-invariant fp8-block MoE backend of the unified MoE layer
(``BatchInvariantFp8BlockConfig`` under ``QuantConfig(DeepSeekFp8, DeepSeekFp8)``, the backend grouping the fp8 formats
with block scales under one key as TRTLLM-Gen's ``TrtllmFp8BlockConfig`` does): the DeepSeek-style classes
``moe.e4m3bs.<act>``, one per typed activation of the layer and its scalars, on the surface of the
TRTLLM-Gen block-fp8 backend under ``QuantConfig(DeepSeekFp8, DeepSeekFp8)`` (e4m3 activations with fp32 block scales per
1x128 block, the weights with fp32 block scales per 128x128 block, every routing input mode).

The class has an exact CPU model piece by piece (the routing, both quantizations by the ``fp8bs.quant`` class, the dense
block-scale class's fold per expert -- the device's block partials promoted in exact fp32 arithmetic --, the activation
chain, GEMM2's conversion, the slot-order combine), so the tests are bit comparisons: the layer's output against the class
composed from its own pieces for several activations, the fused and the standalone requantization giving one output,
run-to-run determinism including CUDA-graph replay, batch / permutation / sub-batch invariance, agreement of every tactic the
autotuner may pick (both requantization spellings), the three routing input modes giving the same bits, the largest expert
count, alternating inputs under programmatic dependent launch, the error contract, and a plausibility observation against
FlashInfer's torch reference of the DeepSeek block-fp8 MoE. The device-free tests pin the registration, the fingerprint
dimensions, the quantizer's vectorised and exact models, the weight view (FlashInfer's DeepSeek view taken as given) and the
dispatcher's membership in the enumerated space at the nominal parts.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer import batch_invariant as bi
from flashinfer.fused_moe import (
    GELU,
    BackendOptions,
    BatchInvariantFp8BlockConfig,
    ExecutionConfig,
    ExpertConfig,
    GeGLU,
    GeGLUTanh,
    Identity,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    PowLU,
    QuantConfig,
    QuantFormat,
    ReLU,
    ReLU2,
    RoutingConfig,
    RoutingInputMode,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
    TrtllmFp8BlockConfig,
)
from flashinfer.fused_moe.api import ALL_BACKEND_CONFIGS, _DEFAULT_BACKEND
from flashinfer.fused_moe.batch_invariant.runner import BatchInvariantMoeRunner
from flashinfer.fused_moe.batch_invariant.runner_fp8_block import (
    BatchInvariantFp8BlockRunner,
)
from flashinfer.fused_moe.batch_invariant.runner_fp8bs import (
    MOE_FP8BS_ARCHS,
    BatchInvariantDeepSeekFp8Runner,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.tllm_enums import RoutingMethodType

_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), MOE_FP8BS_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)

_SWIGLU = SwiGLU()
QUANT = QuantConfig(weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8)
#: The activations the suites cover (the bf16 backend's list): every ``ActivationConfig`` class of the layer with its defaults,
#: and the parameterised ones once more with scalars that move every branch.
ACTIVATIONS = [
    SwiGLU(),
    SwiGLU(alpha=1.702, beta=1.0, limit=7.0),
    SwiGLUStep(),
    SwiGLUStep(limit=3.0),
    GeGLU(),
    GeGLUTanh(),
    SiTU(),
    SiTU(gate_scale=2.0, linear_scale=None, clamp_limit=5.0),
    PowLU(),
    PowLU(m=1.5, limit=4.0),
    ReLU2(),
    Identity(),
    GELU(),
    ReLU(),
    SiLU(),
]
ACTIVATION_IDS = [
    "swiglu",
    "swiglu_oai",
    "swiglu_step",
    "swiglu_step_limit3",
    "geglu",
    "geglu_tanh",
    "situ",
    "situ_clamped",
    "powlu",
    "powlu_limit4",
    "relu2",
    "identity",
    "gelu",
    "relu",
    "silu",
]
BLOCK = 128


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(
        {1: torch.int8, 2: torch.int16, 4: torch.int32}[t.element_size()]
    )


def _problem(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    seed: int,
    gated: bool = True,
    adversarial: bool = False,
):
    """bf16 activations, logits and canonical weights (``w1 [E, 2I, H]`` rows ``[up, gate]``; ``[E, I, H]`` ungated).
    Adversarial: wide per-row and per-block exponent spreads (the block scales span the fp32 range the recipe uses),
    zero blocks (the clamped scale) and a saturating block."""
    g = torch.Generator().manual_seed(seed)
    if adversarial:
        x = torch.randn(T, H, generator=g) * torch.exp2(
            torch.randint(-8, 6, (T, 1), generator=g).float()
        )
        x = x * torch.exp2(
            torch.randint(-6, 6, (1, H // BLOCK), generator=g).float()
        ).repeat_interleave(BLOCK, 1)
        x[:, 5::97] = 0.0
        if T > 2:
            x[2, :BLOCK] = 0.0
        logits = (torch.randn(T, E, generator=g) * 2).float()
        logits[0, : min(k, E)] = 1.0
        span = torch.exp2(torch.randint(-3, 3, (E, 1, 1), generator=g).float())
    else:
        x = torch.randn(T, H, generator=g) * 0.5
        logits = (torch.randn(T, E, generator=g) * 2).float()
        span = 1.0
    x = x.to(torch.bfloat16)
    n1 = 2 * I if gated else I
    w1 = (torch.randn(E, n1, H, generator=g) * 0.05 * span).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, generator=g) * 0.05 * span).to(torch.bfloat16)
    return x.cuda(), logits.cuda(), w1.cuda(), w2.cuda()


def _config(
    k: int,
    E: int,
    I: int,
    act=_SWIGLU,
    method=RoutingMethodType.Renormalize,
    tune_max: int = 4096,
):
    return MoEConfig(
        routing=RoutingConfig(num_experts=E, top_k=k, method=method),
        quant=QUANT,
        experts=ExpertConfig(intermediate_size=I),
        activation=act,
        backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=tune_max),
        finalize=MoEFinalizeConfig(do_finalize=True),
    )


def _layer(*args, device="cuda", **kw):
    return MoELayer(_config(*args, **kw), device=torch.device(device))


def _weights(w1: torch.Tensor, w2: torch.Tensor, act=_SWIGLU) -> MoEWeightPack:
    pack = MoEWeightPack({})
    view = BatchInvariantFp8BlockConfig.prepare_weights(
        w1,
        w2,
        quant=QUANT,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        activation=act,
    )
    pack.prepare_for(BatchInvariantFp8BlockRunner.backend_key, view)
    return pack


def _xq(x: torch.Tensor):
    xq, sx = BatchInvariantFp8BlockConfig.prepare_activations(x, quant=QUANT)
    assert (
        xq.dtype == torch.float8_e4m3fn
        and sx.dtype == torch.float32
        and tuple(sx.shape) == (x.shape[1] // BLOCK, x.shape[0])
    )
    return xq, sx


def _logits_pack(xq, sx, logits: torch.Tensor, bias=None) -> MoEActivationPack:
    return MoEActivationPack(
        xq,
        sx,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
        routing_bias=bias,
    )


def _routed_pack(
    xq, sx, ids: torch.Tensor, weights: torch.Tensor, packed: bool = False
) -> MoEActivationPack:
    if packed:
        return MoEActivationPack(
            xq,
            sx,
            topk_ids=ids.contiguous(),
            topk_weights=weights.float().contiguous(),
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
        )
    return MoEActivationPack(
        xq,
        sx,
        topk_ids=ids.contiguous(),
        topk_weights=weights.contiguous(),
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )


# ----------------------------------------------------------------------------- device-free


def test_backend_is_registered_and_never_a_default_candidate():
    """The config is a known backend of the layer, mapped to the fp8-block runner, absent from the default candidates; the
    runner serves both block-scaled fp8 pairs (the DeepSeek pair by this pair's runner), declares the three routing modes
    and every activation class of the layer (the bf16 runner's); a pair it does not serve is not admitted."""
    assert BatchInvariantFp8BlockConfig in ALL_BACKEND_CONFIGS
    assert (
        _BACKEND_RUNNERS[BatchInvariantFp8BlockConfig] is BatchInvariantFp8BlockRunner
    )
    assert not any(
        isinstance(c, BatchInvariantFp8BlockConfig) for c in _DEFAULT_BACKEND.candidates
    )
    assert [
        a
        for a in (89, 90, 100, 103, 107, 110, 120, 121)
        if BatchInvariantFp8BlockConfig.supported(a)
    ] == [100, 103, 107]
    assert issubclass(BatchInvariantDeepSeekFp8Runner, BatchInvariantFp8BlockRunner)
    assert (
        BatchInvariantFp8BlockRunner.backend_key
        == "batch_invariant_fp8_block"
        != BatchInvariantMoeRunner.backend_key
    )
    assert BatchInvariantFp8BlockRunner.supported_quant_variants == (
        (QuantFormat.MXFP8, QuantFormat.MXFP8),
        (QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8),
    )
    assert BatchInvariantDeepSeekFp8Runner.supported_quant_variants == (QUANT.pair,)
    assert (
        BatchInvariantFp8BlockRunner.implementation(QUANT.pair)
        is BatchInvariantDeepSeekFp8Runner
    )
    assert BatchInvariantFp8BlockRunner.supports_quant(QUANT)
    assert not BatchInvariantFp8BlockRunner.supports_quant(QuantConfig())
    assert set(BatchInvariantFp8BlockRunner.supported_routing_modes) == {
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    }
    assert (
        set(BatchInvariantFp8BlockRunner.supported_activation_classes)
        == set(BatchInvariantMoeRunner.supported_activation_classes)
        == {
            SwiGLU,
            SwiGLUStep,
            GeGLU,
            GeGLUTanh,
            SiTU,
            PowLU,
            ReLU2,
            Identity,
            GELU,
            ReLU,
            SiLU,
        }
    )
    assert repr(BatchInvariantFp8BlockConfig()) == "BatchInvariantFp8BlockConfig()"
    assert (
        "fused_moe_fp8bs" in bi.CLASSES
        and bi.class_name("fused_moe_fp8bs") == "moe.e4m3bs"
    )


def test_fingerprint_dimensions_and_classes():
    """``fingerprint("fused_moe_fp8bs", ...)`` takes the geometry, the routing method and the activation; it differs from the bf16
    class's at the same dimensions; every listed activation is its own class; the routing method is a dimension; the block
    scales' values are not; the quantizer class's fingerprint has no dimension."""
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant import spec_fp8bs as SB

    dims = dict(H=2048, I=768, top_k=8, num_experts=128)
    fp = bi.fingerprint("fused_moe_fp8bs", **dims)
    assert fp.startswith("moe-") and len(fp) == 20
    assert fp != bi.fingerprint("fused_moe", **dims)
    assert (
        bi.fingerprint("fused_moe_fp8bs", method="Renormalize", act="swiglu", **dims)
        == fp
    )
    assert bi.fingerprint("fused_moe_fp8bs", act=SwiGLU(), **dims) == fp
    assert (
        bi.fingerprint("fused_moe_fp8bs", method=RoutingMethodType.Sigmoid, **dims)
        != fp
    )
    assert (
        bi.fingerprint("fused_moe_fp8bs", H=2048, I=768, top_k=8, num_experts=64) != fp
    )
    assert (
        bi.fingerprint("fused_moe_fp8bs", H=2048, I=640, top_k=8, num_experts=128) != fp
    )
    fps = {bi.fingerprint("fused_moe_fp8bs", act=a, **dims) for a in ACTIVATIONS}
    assert len(fps) == len(ACTIVATIONS) and fp in fps
    assert bi.fingerprint("fused_moe_fp8bs", act="relu2", **dims) == bi.fingerprint(
        "fused_moe_fp8bs", act=ReLU2(), **dims
    )
    with pytest.raises(ValueError):
        bi.fingerprint("fused_moe_fp8bs", method=RoutingMethodType.Llama4, **dims)
    with pytest.raises(TypeError):
        bi.fingerprint("fused_moe_fp8bs", block=128, **dims)
    for cfg in ACTIVATIONS:
        a = S.activation_of(cfg)
        assert bi.fingerprint(
            "fused_moe_fp8bs", act=cfg, **dims
        ) == SB.spec_fingerprint_fp8bs(
            2048, 768, 8, 128, S.SOFTMAX_RENORM, a.act, a.params, a.gated
        )
    s = SB.Fp8BsMoESpec(256, 128, 2, 8)
    assert (
        s.name == "moe.e4m3bs.swiglu"
        and SB.Fp8BsMoESpec(256, 128, 2, 8, act="relu2", gated=False).name
        == "moe.e4m3bs.relu2"
    )
    from flashinfer.fused_moe.batch_invariant import mterm

    assert mterm.sexpr(s.gemm1()) == mterm.sexpr(
        S.MoESpec(256, 128, 2, 8, alg=s.alg).gemm1()
    )  # the bf16 chain, no scale symbol
    assert mterm.sexpr(s.gemm2()) == "(cvt bf16 (acc2 0 0))"
    q = SB.quant_fingerprint()
    assert q.startswith("fp8bsq-") and len(q) == 23 and q == SB.quant_fingerprint()


def _adversarial_rows(M: int, K: int, seed: int) -> torch.Tensor:
    """bf16 rows for the quantizer: wide exponent spreads inside one block (the block scale saturates the small values to zero),
    blocks at the e4m3 maximum, signed zeros, subnormals, a zero block, special values."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, K, generator=g) * torch.exp2(
        torch.randint(-20, 20, (M, K), generator=g).float()
    )
    x[:, ::37] *= 2.0**30
    x[:, 5::64] = 0.0
    x[:, 6::64] = -0.0
    x[:, 7::64] = torch.rand(M, len(range(7, K, 64)), generator=g) * 2.0**-130
    if M > 1:
        x[1, :BLOCK] = 0.0
    x = x.to(torch.bfloat16)
    x[0, :16] = torch.tensor(
        [
            0.0,
            -0.0,
            448.0,
            -448.0,
            500.0,
            -1e30,
            2.0**-10,
            2.0**-12,
            1e-38,
            0.17,
            0.1875,
            0.203125,
            0.21875,
            0.1953125,
            1.0,
            2.0,
        ]
    ).to(torch.bfloat16)
    return x


def test_quantizer_vectorised_model_equals_exact_model_and_weight_reference():
    """The class's vectorised CPU reference (``reference_fp8bs``) is the exact scalar model block for block on adversarial rows
    (both dtypes, both scale layouts); the weight reference is the row chain per 128x128 block; FlashInfer's DeepSeek recipe
    gives the same scales and codes within an e4m3 rounding (it divides, the class multiplies by the reciprocal)."""
    from flashinfer.batch_invariant.fp_interface import exact as fp32
    from flashinfer.fused_moe.batch_invariant import reference_fp8bs as RB
    from flashinfer.fused_moe.prepare import _deepseek_fp8_quantize_activations

    for dtype in (torch.bfloat16, torch.float16):
        x = _adversarial_rows(6, 512, 3).to(dtype)
        if dtype == torch.float16:
            x = x.float().clamp(-60000.0, 60000.0).to(dtype)
        codes_k, s_k = RB.quantize_rows_reference(x, "K")
        codes_mn, s_mn = RB.quantize_rows_reference(x, "MN")
        assert (
            torch.equal(codes_k, codes_mn)
            and torch.equal(s_k, s_mn.t())
            and s_k.shape == (6, 4)
        )
        for r in range(6):
            for b in range(4):
                s_e, _, codes_e = RB.exact_block(
                    x[r, b * BLOCK : (b + 1) * BLOCK].float().tolist()
                )
                assert s_e == s_k[r, b].item(), (dtype, r, b)
                assert codes_e == codes_k[r, b * BLOCK : (b + 1) * BLOCK].tolist(), (
                    dtype,
                    r,
                    b,
                )
        assert (
            s_k[1, 0].item() == fp32.to_f32(1e-12) and (codes_k[1, :BLOCK] == 0).all()
        )  # the zero block
        assert codes_k.abs().max().item() <= 448.0
        by, _ = RB.quantize_rows_bytes(x, "K")
        assert torch.equal(by, RB.e4m3_bytes(codes_k)) and by.dtype == torch.uint8
    x = (torch.randn(64, 512, generator=torch.Generator().manual_seed(7)) * 0.5).to(
        torch.bfloat16
    )
    q_fi, s_fi = _deepseek_fp8_quantize_activations(x)
    codes, s = RB.quantize_rows_reference(x, "MN")
    assert torch.equal(bits(s), bits(s_fi.cpu())), (
        "the same block scales as FlashInfer's recipe"
    )
    off = (codes != q_fi.cpu().float()).float().mean().item()
    assert off < 0.01, (
        off
    )  # a few elements per block round differently under the division
    w = _adversarial_rows(256, 384, 9)
    codes, s = RB.quantize_weight_reference(w)
    assert codes.shape == (256, 384) and s.shape == (2, 3)
    wf = w.float()
    for nb in range(2):
        for kb in range(3):
            blk = wf[nb * 128 : (nb + 1) * 128, kb * 128 : (kb + 1) * 128]
            s_e = fp32.fmaxc(
                fp32.div(blk.abs().max().item(), 448.0), fp32.to_f32(1e-12)
            )
            inv_e = fp32.div(1.0, s_e)
            assert s_e == s[nb, kb].item(), (nb, kb)
            for i, j in [(0, 0), (127, 127), (5, 37), (64, 1)]:
                assert codes[nb * 128 + i, kb * 128 + j].item() == fp32.to_fp8(
                    fp32.mul(blk[i, j].item(), inv_e), "e4m3"
                ), (nb, kb, i, j)


def test_weight_view_and_the_trtllm_view_taken_as_given():
    """``prepare_weights`` quantizes per expert per 128x128 block by the class's reference, interleaves the gated weight's
    quantized rows and keeps the canonical rows' scales; the ungated weight is taken as given; the acceptance (H, I multiples
    of 128) raises the mode's error; ``from_trtllm_view`` consumes FlashInfer's DeepSeek weight view as given."""
    from flashinfer.fused_moe.batch_invariant import reference_fp8bs as RB
    from flashinfer.fused_moe.batch_invariant.api import interleave_fc1
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        check_shape_fp8bs,
    )
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    g = torch.Generator().manual_seed(3)
    E, I, H = 3, 128, 256
    w1 = (
        torch.randn(E, 2 * I, H, generator=g)
        * torch.exp2(torch.randint(-6, 7, (E, 2 * I, H), generator=g).float())
    ).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, generator=g) * 0.05).to(torch.bfloat16)
    w1[1, :128, :128] = 0.0
    view = BatchInvariantFp8BlockConfig.prepare_weights(
        w1, w2, quant=QUANT, activation=SwiGLU()
    )
    W = MoEWeightsE4M3BS.from_view(view)
    q1 = torch.stack(
        [RB.e4m3_bytes(RB.quantize_weight_reference(w1[e])[0]) for e in range(E)]
    )
    s1 = torch.stack([RB.quantize_weight_reference(w1[e])[1] for e in range(E)])
    assert torch.equal(
        W.w1.view(torch.uint8).cpu(), interleave_fc1(q1)
    ) and torch.equal(bits(W.w1_sf.cpu()), bits(s1))
    for e in range(E):
        rq, rs = RB.quantize_weight_reference(w2[e])
        assert torch.equal(
            W.w2[e].view(torch.uint8).cpu(), RB.e4m3_bytes(rq)
        ) and torch.equal(bits(W.w2_sf[e].cpu()), bits(rs)), e
    assert (
        W.gated
        and W.act == "silu"
        and W.params == ()
        and (W.num_experts, W.hidden_size, W.intermediate_size) == (E, H, I)
    )
    Wu = MoEWeightsE4M3BS.from_view(
        BatchInvariantFp8BlockConfig.prepare_weights(
            w1[:, :I].contiguous(), w2, quant=QUANT, activation=ReLU2()
        )
    )
    assert not Wu.gated and Wu.act == "relu2" and Wu.w1.shape == (E, I, H)
    for e in range(E):
        rq, rs = RB.quantize_weight_reference(w1[e, :I])
        assert torch.equal(
            Wu.w1[e].view(torch.uint8).cpu(), RB.e4m3_bytes(rq)
        ) and torch.equal(bits(Wu.w1_sf[e].cpu()), bits(rs)), e
    a = activation_of(SwiGLU(alpha=1.702, beta=1.0, limit=7.0))
    Wp = MoEWeightsE4M3BS.from_view(
        BatchInvariantFp8BlockConfig.prepare_weights(
            w1, w2, quant=QUANT, activation=SwiGLU(alpha=1.702, beta=1.0, limit=7.0)
        )
    )
    assert (Wp.act, Wp.params) == (a.act, a.params)
    # FlashInfer's DeepSeek weight view: the canonical rows' block quantization taken as given, interleaved here
    fi_view = TrtllmFp8BlockConfig.prepare_weights(
        w1,
        w2,
        quant=QUANT,
        num_local_experts=E,
        hidden_size=H,
        intermediate_size=I,
        activation=SwiGLU(),
        device=torch.device("cpu"),
    )
    Wg = MoEWeightsE4M3BS.from_view(
        BatchInvariantFp8BlockConfig.from_trtllm_view(
            fi_view, quant=QUANT, activation=SwiGLU()
        )
    )
    assert torch.equal(
        Wg.w1.view(torch.uint8),
        interleave_fc1(fi_view["gemm1_weights"].view(torch.uint8)),
    )
    assert torch.equal(
        bits(Wg.w1_sf), bits(fi_view["gemm1_weights_scale"].float())
    ) and torch.equal(
        Wg.w2.view(torch.uint8), fi_view["gemm2_weights"].view(torch.uint8)
    )
    assert torch.equal(bits(W.w1_sf.cpu()), bits(Wg.w1_sf.cpu())), (
        "the same block scales as FlashInfer's recipe"
    )
    for bad in [
        (w1[:, :, :192].contiguous(), w2[:, :192].contiguous()),
        (w1[:, :192].contiguous(), w2[:, :, :96].contiguous()),
    ]:
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            MoEWeightsE4M3BS.prepare(*bad)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        check_shape_fp8bs(256, 128, 1025)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        check_shape_fp8bs(256, 128, 8, 9)
    check_shape_fp8bs(7168, 2048, 1024, 16)


@pytest.mark.parametrize(
    "family,sm_count",
    [("sm_107", 212), ("sm_107", 204), ("sm_100", 148), ("sm_103", 152)],
)
def test_dispatcher_picks_are_members_of_the_space(family, sm_count):
    """The dispatcher's pick for every bench geometry and token count lies in the enumerated grouped block-scale space of its
    GEMM (the fused requantization's chain in its whole-block subspace), for the gated and the ungated classes, at the nominal
    parts; the ranked list is short, distinct and inside the space; the requantization rule is a function of the rows per expert."""
    from flashinfer.fused_moe.batch_invariant.blocks.fp8bs_compile import GEMM2_CHAIN
    from flashinfer.fused_moe.batch_invariant.schedule import rows_per_expert
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        FUSED_QUANT_MIN_ROWS,
        all_valid_configs_fp8bs,
        all_valid_configs_fp8bs_swapped,
        default_configs_fp8bs,
        fused_quant_default,
        gemm1_chain,
        grouped_knobs_fp8bs,
        rank_grouped_fp8bs,
        swapped_knobs_fp8bs,
    )

    assert len(grouped_knobs_fp8bs()) == 54
    # the swapped decode space: 64 / 128 weight rows x 8 / 16 / 32 / 64 tokens, 1-CTA; the bf16 / f32 chains only; disjoint
    sw = swapped_knobs_fp8bs()
    assert len(sw) == 8 and all(
        k.swap_ab and k.tile_m in (64, 128) and k.tile_n in (8, 16, 32, 64) for k in sw
    )
    assert not set(sw) & set(grouped_knobs_fp8bs())
    geoms = [
        (2048, 768, 128, 8),
        (4096, 14336, 8, 2),
        (7168, 2048, 256, 8),
        (5120, 8192, 16, 1),
        (512, 256, 64, 6),
        (1024, 512, 64, 8),
    ]
    for H, I, E, k in geoms:
        s1 = set(all_valid_configs_fp8bs(2 * I, H, E, gemm1_chain("silu")))
        s1q = set(all_valid_configs_fp8bs(2 * I, H, E, gemm1_chain("silu", fused=True)))
        s1u = set(all_valid_configs_fp8bs(I, H, E, gemm1_chain("relu2", (), False)))
        s1uq = set(
            all_valid_configs_fp8bs(I, H, E, gemm1_chain("relu2", (), False, True))
        )
        s2 = set(all_valid_configs_fp8bs(H, I, E, GEMM2_CHAIN))
        assert (
            len(s1) == len(s2) == 54
            and len(s1q) == 18
            and len(s1uq) == 36
            and s1q <= s1
            and s1uq <= s1u
        )
        w1 = set(all_valid_configs_fp8bs_swapped(2 * I, H, E, gemm1_chain("silu")))
        w1u = set(
            all_valid_configs_fp8bs_swapped(I, H, E, gemm1_chain("relu2", (), False))
        )
        w2 = set(all_valid_configs_fp8bs_swapped(H, I, E, GEMM2_CHAIN))
        assert len(w1) == len(w1u) == len(w2) == 8
        assert not all_valid_configs_fp8bs_swapped(
            2 * I, H, E, gemm1_chain("silu", fused=True)
        )
        for T in (1, 7, 64, 300, 800, 4096, 16384):
            for fused in (False, True):
                c1, c2 = default_configs_fp8bs(
                    T,
                    k,
                    H,
                    I,
                    E,
                    sm_count,
                    family,
                    "silu",
                    True,
                    (),
                    fused,
                    nominal=True,
                )
                assert c1 in (s1q if fused else s1 | w1) and c2 in s2 | w2, (
                    family,
                    T,
                    H,
                    I,
                    E,
                    k,
                    fused,
                    c1.describe(),
                    c2.describe(),
                )
                assert not (
                    fused and c1.swap_ab
                )  # the fused requantization is the padded body's spelling
                c1u, _ = default_configs_fp8bs(
                    T,
                    k,
                    H,
                    I,
                    E,
                    sm_count,
                    family,
                    "relu2",
                    False,
                    (),
                    fused,
                    nominal=True,
                )
                assert c1u in (s1uq if fused else s1u | w1u), (
                    family,
                    T,
                    H,
                    I,
                    E,
                    k,
                    fused,
                    c1u.describe(),
                )
            ranked = rank_grouped_fp8bs(
                rows_per_expert(T, k, E),
                H,
                I,
                E,
                GEMM2_CHAIN,
                sm_count,
                family,
                3,
                nominal=True,
            )
            assert (
                1 <= len(ranked) <= 3
                and len(set(ranked)) == len(ranked)
                and all(c in s2 | w2 for c in ranked)
            )
            assert fused_quant_default(T, k, E) == (
                rows_per_expert(T, k, E) >= FUSED_QUANT_MIN_ROWS
            )
    # the acceptance: whole scale blocks of every operand
    assert (
        all_valid_configs_fp8bs(2 * 760, 2048, 128, gemm1_chain("silu")) == []
    )  # I % 128
    assert all_valid_configs_fp8bs(2048, 760, 128, GEMM2_CHAIN) == []  # K % 128
    assert all_valid_configs_fp8bs(2048 + 64, 768, 128, GEMM2_CHAIN) == []  # N % 128


def test_unsupported_device_raises_before_any_kernel():
    """Constructing the backend's runner for the DeepSeek pair gives this pair's runner; on a CPU device it fails its support
    check with the mode's error naming the class. The pair's runner admits its own pair only, and ``from_trtllm_view`` takes
    the DeepSeek pair only (TRTLLM-Gen's MXFP8 view is shuffled)."""
    runner = BatchInvariantFp8BlockRunner(_config(8, 64, 256), torch.device("cpu"))
    assert type(runner) is BatchInvariantDeepSeekFp8Runner
    with pytest.raises(bi.BatchInvariantUnsupportedError) as excinfo:
        runner._check_support()
    assert "moe.e4m3bs" in str(excinfo.value) and "cpu" in str(excinfo.value)
    mxfp8 = QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8)
    cfg = MoEConfig(
        routing=RoutingConfig(num_experts=64, top_k=8),
        quant=mxfp8,
        experts=ExpertConfig(intermediate_size=256),
        backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
    )
    with pytest.raises(NotImplementedError):
        BatchInvariantDeepSeekFp8Runner(cfg, torch.device("cpu"))._check_support()
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="DeepSeekFp8"):
        BatchInvariantFp8BlockConfig.from_trtllm_view({}, quant=mxfp8)


# ----------------------------------------------------------------------------- GPU


def _reference_route(logits: torch.Tensor, k: int, method):
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S

    return R.route_cpu(logits.cpu(), k, S.METHOD_OF_FLASHINFER[method])


def _exact_acc(
    a: torch.Tensor,
    sa_mn: torch.Tensor,
    w: torch.Tensor,
    w_sf: torch.Tensor,
    counts,
    canon_rows,
    sel,
) -> torch.Tensor:
    """The dense block-scale class, exactly, on every selected expert's rows of the dense packing: the device block partials
    (the per-tensor fp8 class at K = 128 with an fp32 output: ``tests/gemm/test_batch_invariant_gemm_fp8_blockscale.py``)
    promoted in exact fp32 arithmetic with the row's and the weight block's scales. ``canon_rows``: the canonical weight row
    of every output column (the gated weight's interleave), or None."""
    from tests.gemm.test_batch_invariant_gemm_fp8_blockscale import (
        block_partials,
        exact_promotion,
    )

    Tk = a.shape[0]
    N = w.shape[1]
    acc = torch.zeros(Tk, N, dtype=torch.float32)
    sa = sa_mn.t().cpu()
    off = 0
    for e, c in enumerate(counts):
        if c and e in sel:
            rows = a[off : off + c].contiguous()
            if (
                canon_rows is not None
            ):  # the exact fold on the canonical rows, its columns permuted to the interleaved order
                w_canon = torch.empty_like(w[e])
                w_canon[canon_rows] = w[e]
                acc[off : off + c] = exact_promotion(
                    block_partials(rows, w_canon), sa[off : off + c], w_sf[e]
                )[:, canon_rows]
            else:
                acc[off : off + c] = exact_promotion(
                    block_partials(rows, w[e]), sa[off : off + c], w_sf[e]
                )
        off += c
    return acc


def _selected(counts, limit: int = 32):
    active = [e for e, c in enumerate(counts) if c]
    if len(active) <= limit:
        return set(active)
    step = -(-len(active) // (limit - 4))
    return set(active[::step]) | {
        active[0],
        active[-1],
        max(active, key=lambda e: counts[e]),
        min(active, key=lambda e: counts[e]),
    }


def _ladder(xq, sx, ids, weights, view, act, out=None):
    """The class composed from its own pieces, bit for bit: the dense packing, the exact block-scale fold per (selected) expert,
    the CPU chain, the quantizer's reference on the device's bf16 intermediate, the exact fold of GEMM2, the CPU conversion and
    combine; the op's intermediates in both requantization spellings against it. Returns the reference output rows checked."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import reference_fp8bs as RB
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        fused_moe_fp8bs,
    )

    W = MoEWeightsE4M3BS.from_view(view)
    T, H = xq.shape
    k = ids.shape[1]
    I = W.intermediate_size
    N1 = W.w1.shape[1]
    ref_out, im = fused_moe_fp8bs(
        xq, sx, ids, weights, W, return_intermediates=True, fuse_quant=False
    )
    counts = im["counts"].cpu().tolist()
    assert (
        counts
        == torch.bincount(
            ids.cpu().reshape(-1).long(), minlength=W.num_experts
        ).tolist()
    )
    Tk = T * k
    dest = im["dest"].cpu().reshape(-1).long()
    tok = torch.arange(Tk) // k
    assert torch.equal(
        im["x_perm"].view(torch.uint8).cpu()[dest], xq.view(torch.uint8).cpu()[tok]
    )
    assert torch.equal(bits(im["sx_perm"].cpu()[:, dest]), bits(sx.cpu()[:, tok]))
    assert torch.equal(im["src"].cpu().long()[dest], torch.arange(Tk))
    sel = _selected(counts)
    mask = torch.zeros(Tk, dtype=torch.bool)
    off = 0
    for e, c in enumerate(counts):
        if e in sel:
            mask[off : off + c] = True
        off += c
    acc1 = _exact_acc(
        im["x_perm"],
        im["sx_perm"],
        W.w1,
        W.w1_sf,
        counts,
        RB.orig_rows(N1) if W.gated else None,
        sel,
    )
    a_ref = RB.chain_cpu(acc1, act)
    a = im["a"].cpu()
    assert a.shape == (Tk, I) and torch.equal(bits(a[mask]), bits(a_ref[mask])), (
        "gemm1",
        (a[mask] != a_ref[mask]).sum().item(),
    )
    aq_ref, sa_ref = RB.quantize_rows_bytes(a, "MN")
    assert torch.equal(im["aq"].view(torch.uint8).cpu(), aq_ref) and torch.equal(
        bits(im["sa"].cpu()), bits(sa_ref)
    ), "the requantization"
    acc2 = _exact_acc(im["aq"], im["sa"], W.w2, W.w2_sf, counts, None, sel)
    y_ref = R.TorchF32.cvt_out(acc2, "bf16")
    y = im["y"].cpu()
    assert torch.equal(bits(y[mask]), bits(y_ref[mask])), (
        "gemm2",
        (y[mask] != y_ref[mask]).sum().item(),
    )
    assert torch.equal(
        bits(ref_out.cpu()), bits(R.combine_cpu(im["y"], im["dest"], weights))
    )
    outq, imq = fused_moe_fp8bs(
        xq, sx, ids, weights, W, return_intermediates=True, fuse_quant=True
    )
    assert imq["a"] is None and imq["fused"]
    assert torch.equal(imq["aq"].view(torch.uint8).cpu(), aq_ref) and torch.equal(
        bits(imq["sa"].cpu()), bits(sa_ref)
    ), "the fused requantization"
    assert torch.equal(bits(imq["y"].cpu()), bits(y)) and torch.equal(
        bits(outq), bits(ref_out)
    )
    if out is not None:
        assert torch.equal(bits(out), bits(ref_out))
    return ref_out


@needs_device
@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_layer_matches_the_class_composition(act):
    """The layer (from logits) == the class's own routing on the CPU and the oracle ladder of the class (the exact block-scale
    fold per expert, the CPU chain, the quantizer's reference, GEMM2, the combine) bit for bit, for every activation; the fused
    and the standalone requantization give the same bytes and the same output."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    T, k, H, I, E = 300, 6, 512, 256, 64
    chain = activation_of(act)
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=1, gated=chain.gated, adversarial=True
    )
    xq, sx = _xq(x)
    W = _weights(w1, w2, act)
    layer = _layer(k, E, I, act)
    out = layer(_logits_pack(xq, sx, logits), W)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    _ladder(
        xq,
        sx,
        ids.cuda(),
        weights.cuda(),
        W.get_view(BatchInvariantFp8BlockRunner.backend_key),
        chain,
        out,
    )


@needs_device
def test_input_quantization_is_the_class_on_the_device():
    """``prepare_activations`` runs the ``fp8bs.quant`` kernel: codes and block scales == the class's reference bit for bit on
    adversarial rows, in both scale layouts and every warps-per-row placement; R1 / R2 at op level; the layer takes the pair in
    either scale layout."""
    from flashinfer.fused_moe.batch_invariant import reference_fp8bs as RB
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        fp8_block_quantize,
        quant_warps_per_row,
    )

    for M, K in [(1, 128), (37, 256), (300, 2048), (1000, 512), (129, 384)]:
        x = _adversarial_rows(M, K, M + K).cuda()
        rq, rs = RB.quantize_rows_bytes(x, "MN")
        for lay in ("MN", "K"):
            for wpr in (1, 2, 4, 8):
                if wpr > max(1, (K // BLOCK + 1) // 2):
                    continue
                q, s = fp8_block_quantize(x, lay, wpr=wpr)
                s_mn = s if lay == "MN" else s.t()
                assert torch.equal(q.view(torch.uint8).cpu(), rq) and torch.equal(
                    bits(s_mn.cpu().contiguous()), bits(rs)
                ), (M, K, lay, wpr)
        assert 1 <= quant_warps_per_row(M, K, 212) <= 8
        q, s = BatchInvariantFp8BlockConfig.prepare_activations(x, quant=QUANT)
        assert torch.equal(q.view(torch.uint8).cpu(), rq) and torch.equal(
            bits(s.cpu()), bits(rs)
        )
    x = _adversarial_rows(200, 1024, 3).cuda()
    q, s = fp8_block_quantize(x)
    perm = torch.randperm(200, generator=torch.Generator().manual_seed(1)).cuda()
    qp, sp = fp8_block_quantize(x[perm].contiguous())
    assert torch.equal(qp.view(torch.uint8), q.view(torch.uint8)[perm]) and torch.equal(
        bits(sp), bits(s[:, perm])
    )
    for n in (1, 7, 64):
        qs, ss = fp8_block_quantize(x[:n].contiguous())
        assert torch.equal(
            qs.view(torch.uint8), q.view(torch.uint8)[:n]
        ) and torch.equal(bits(ss), bits(s[:, :n]))
    T, k, H, I, E = 100, 4, 256, 128, 16
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=23, adversarial=True)
    xq, sx = _xq(x)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    ref = bits(layer(_logits_pack(xq, sx, logits), W))
    assert torch.equal(
        ref, bits(layer(_logits_pack(xq, sx.t().contiguous(), logits), W))
    )
    # FlashInfer's own DeepSeek pair (its torch recipe divides by the scale): the class starts at the bytes and the scales it is handed
    q_fi, s_fi = TrtllmFp8BlockConfig.prepare_activations(x, quant=QUANT)
    out_fi = layer(_logits_pack(q_fi, s_fi, logits), W)
    assert out_fi.shape == (T, H) and torch.isfinite(out_fi.float()).all()


@needs_device
def test_the_largest_expert_count_and_slot_count_route_and_run():
    """1024 experts and 16 slots: the class's routing from the logits equals the CPU model and the layer equals the class
    composition."""
    from flashinfer.fused_moe.batch_invariant.api import route
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    for T, k, H, I, E in ((96, 8, 256, 128, 1024), (40, 16, 256, 128, 64)):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=9, adversarial=True)
        xq, sx = _xq(x)
        ids, wts = route(logits, k)
        ref_ids, ref_w = _reference_route(logits, k, RoutingMethodType.Renormalize)
        assert torch.equal(ids.cpu(), ref_ids) and torch.equal(
            bits(wts.cpu()), bits(ref_w)
        )
        W = _weights(w1, w2)
        out = _layer(k, E, I)(_logits_pack(xq, sx, logits), W)
        _ladder(
            xq,
            sx,
            ids,
            wts,
            W.get_view(BatchInvariantFp8BlockRunner.backend_key),
            activation_of(_SWIGLU),
            out,
        )


@needs_device
def test_three_routing_modes_agree_and_runs_are_deterministic():
    """From logits, pre-routed unpacked (fp32 and bf16 weights) and pre-routed packed give the bits of the class on their inputs;
    repeated runs and a CUDA-graph replay reproduce them."""
    T, k, H, I, E = 257, 8, 1024, 512, 128
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=2, adversarial=True)
    xq, sx = _xq(x)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    out_l = layer(_logits_pack(xq, sx, logits), W)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    out_u = layer(_routed_pack(xq, sx, ids.cuda(), weights.cuda()), W)
    assert torch.equal(bits(out_l), bits(out_u))
    w16 = weights.to(torch.bfloat16)
    out_u16 = layer(_routed_pack(xq, sx, ids.cuda(), w16.cuda()), W)
    out_p = layer(_routed_pack(xq, sx, ids.cuda(), w16.cuda(), packed=True), W)
    assert torch.equal(bits(out_u16), bits(out_p))
    assert torch.equal(
        bits(out_u16),
        bits(layer(_routed_pack(xq, sx, ids.cuda(), w16.float().cuda()), W)),
    )
    for _ in range(3):
        assert torch.equal(bits(layer(_logits_pack(xq, sx, logits), W)), bits(out_l))
    stream = torch.cuda.Stream()
    pack = _logits_pack(xq, sx, logits)
    with torch.cuda.stream(stream):
        layer(pack, W)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out_g = layer(pack, W)
    torch.cuda.synchronize()
    for _ in range(2):
        out_g.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out_g), bits(out_l))


@needs_device
def test_batch_permutation_and_subbatch_invariance():
    """A token's bits do not depend on the batch size, its position, its neighbours or the expert loads the other tokens create
    (the requantization spelling the rows-per-expert rule picks changes with the load; the bits do not)."""
    T, k, H, I, E = 500, 8, 512, 256, 64
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=3, adversarial=True)
    xq, sx = _xq(x)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    full = bits(layer(_logits_pack(xq, sx, logits), W))
    for n in (1, 2, 7, 64, 200, 333):
        assert torch.equal(
            bits(layer(_logits_pack(xq[:n], sx[:, :n].contiguous(), logits[:n]), W)),
            full[:n],
        ), n
    perm = torch.randperm(T, generator=torch.Generator().manual_seed(4)).cuda()
    assert torch.equal(
        bits(
            layer(
                _logits_pack(
                    xq[perm].contiguous(),
                    sx[:, perm].contiguous(),
                    logits[perm].contiguous(),
                ),
                W,
            )
        ),
        full[perm],
    )
    for n in (1, 2, 7, 64, 200):
        idx = torch.randperm(T, generator=torch.Generator().manual_seed(n))[:n].cuda()
        assert torch.equal(
            bits(
                layer(
                    _logits_pack(
                        xq[idx].contiguous(),
                        sx[:, idx].contiguous(),
                        logits[idx].contiguous(),
                    ),
                    W,
                )
            ),
            full[idx],
        ), n
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda()
    base = bits(layer(_routed_pack(xq, sx, ids, weights), W))
    g = torch.Generator().manual_seed(5)
    xq_extra, sx_extra = _xq(
        (torch.randn(556, H, generator=g) * 0.5).to(torch.bfloat16).cuda()
    )
    w_extra = torch.rand(556, k, generator=g).cuda()
    for t in (0, T // 2, T - 1):
        hot = ids[t : t + 1].expand(556, k).contiguous()
        out = layer(
            _routed_pack(
                torch.cat([xq[t : t + 1], xq_extra]),
                torch.cat([sx[:, t : t + 1], sx_extra], 1).contiguous(),
                torch.cat([ids[t : t + 1], hot]),
                torch.cat([weights[t : t + 1], w_extra]),
            ),
            W,
        )
        assert torch.equal(bits(out)[:1], base[t : t + 1]), ("hot group", t)
        mine = set(ids[t].tolist())
        others = (
            torch.tensor(
                [e for e in range(E) if e not in mine][:k],
                dtype=torch.int32,
                device="cuda",
            )
            .view(1, k)
            .expand(556, k)
            .contiguous()
        )
        out = layer(
            _routed_pack(
                torch.cat([xq[t : t + 1], xq_extra]),
                torch.cat([sx[:, t : t + 1], sx_extra], 1).contiguous(),
                torch.cat([ids[t : t + 1], others]),
                torch.cat([weights[t : t + 1], w_extra]),
            ),
            W,
        )
        assert torch.equal(bits(out)[:1], base[t : t + 1]), ("unrelated hot experts", t)


@needs_device
def test_every_tactic_computes_the_class_and_autotuning_keeps_the_bits():
    """Every tactic the autotuner may pick (the dispatcher's ranked record pairs in both requantization spellings) gives the same
    bits; every record of the fast grouped space in both spellings too; tuning the layer profiles them and the output before,
    during and after tuning is one output."""
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        fused_moe_fp8bs,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.fp8bs_compile import GEMM2_CHAIN
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        all_valid_configs_fp8bs,
        all_valid_configs_fp8bs_swapped,
        gemm1_chain,
    )

    T, k, H, I, E = 300, 6, 1024, 512, 64
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=6, adversarial=True)
    xq, sx = _xq(x)
    layer = _layer(k, E, I)
    W = _weights(w1, w2)
    runner = layer.runners[0]
    assert isinstance(runner, BatchInvariantDeepSeekFp8Runner)
    pack = _logits_pack(xq, sx, logits)
    ref = bits(layer(pack, W))
    inputs = runner.pack_inputs(pack, W)
    tactics = runner.get_valid_tactics(inputs, None)
    assert (
        2 <= len(tactics) <= 18
        and len(set(tactics)) == len(tactics)
        and {t[-1] for t in tactics} == {0, 1}
        and all(len(t) == 15 for t in tactics)
    )
    for t in tactics:
        assert torch.equal(
            bits(runner.forward(runner.pack_inputs(pack, W), tactic=t)), ref
        ), t
    failed_before = (
        len(AutoTuner.get().stats.failed_tactics)
        if hasattr(AutoTuner.get(), "stats")
        else 0
    )
    with autotune(True):
        assert torch.equal(bits(layer(pack, W)), ref)
    assert torch.equal(bits(layer(pack, W)), ref)
    if hasattr(AutoTuner.get(), "stats"):
        assert len(AutoTuner.get().stats.failed_tactics) == failed_before
    # the fast cluster subset of the whole grouped space, both GEMMs, both spellings
    Wd = MoEWeightsE4M3BS.from_view(W.get_view(runner.backend_key))
    ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, wts = ids.cuda(), wts.cuda()
    clusters = ((1, 1), (1, 2), (2, 1))
    s1 = all_valid_configs_fp8bs(2 * I, H, E, gemm1_chain("silu"), clusters)
    s1q = all_valid_configs_fp8bs(
        2 * I, H, E, gemm1_chain("silu", fused=True), clusters
    )
    s2 = all_valid_configs_fp8bs(H, I, E, GEMM2_CHAIN, clusters)
    assert (
        len(s1) == len(s2) == 24 and len(s1q) == 8
    )  # 1x1 and 1x2: 6 records each; 2x1: the 1-CTA and the 2-CTA records
    bad = [
        c.describe()
        for c in s1
        if not torch.equal(
            bits(
                fused_moe_fp8bs(
                    xq, sx, ids, wts, Wd, config1=c, config2=s2[0], fuse_quant=False
                )
            ),
            ref,
        )
    ]
    bad += [
        "fused:" + c.describe()
        for c in s1q
        if not torch.equal(
            bits(
                fused_moe_fp8bs(
                    xq, sx, ids, wts, Wd, config1=c, config2=s2[0], fuse_quant=True
                )
            ),
            ref,
        )
    ]
    bad += [
        "gemm2:" + c.describe()
        for c in s2
        if not torch.equal(
            bits(
                fused_moe_fp8bs(
                    xq, sx, ids, wts, Wd, config1=s1[0], config2=c, fuse_quant=False
                )
            ),
            ref,
        )
    ]
    # the swapped decode records (the dispatcher's at a few rows per expert; every one a member of the class): both GEMMs
    sw1 = all_valid_configs_fp8bs_swapped(2 * I, H, E, gemm1_chain("silu"))
    sw2 = all_valid_configs_fp8bs_swapped(H, I, E, GEMM2_CHAIN)
    assert len(sw1) == len(sw2) == 8
    bad += [
        "swapped:" + c.describe()
        for c in sw1
        if not torch.equal(
            bits(
                fused_moe_fp8bs(
                    xq, sx, ids, wts, Wd, config1=c, config2=s2[0], fuse_quant=False
                )
            ),
            ref,
        )
    ]
    bad += [
        "swapped gemm2:" + c.describe()
        for c in sw2
        if not torch.equal(
            bits(
                fused_moe_fp8bs(
                    xq, sx, ids, wts, Wd, config1=s1[0], config2=c, fuse_quant=False
                )
            ),
            ref,
        )
    ]
    assert not bad, bad[:8]


@needs_device
@pytest.mark.parametrize("act", [SwiGLU(), ReLU2()], ids=["swiglu", "relu2"])
def test_swapped_decode_body_equals_the_padded_body(act):
    """At a few rows per active expert the dispatcher picks the swapped decode body for both GEMMs (the structural rule); the layer's
    output equals the padded body's bit for bit, every swapped record of both GEMMs agrees, the first tactic is the swapped pair
    and the autotuner keeps the bits."""
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        fused_moe_fp8bs,
    )
    from flashinfer.fused_moe.batch_invariant.blocks.fp8bs_compile import GEMM2_CHAIN
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        all_valid_configs_fp8bs_swapped,
        default_configs_fp8bs,
        gemm1_chain,
    )

    T, k, H, I, E = 8, 4, 1024, 512, 64
    gated = not isinstance(act, ReLU2)
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=9, adversarial=True)
    if not gated:
        w1 = w1[:, :I].contiguous()
    xq, sx = _xq(x)
    layer = _layer(k, E, I, act=act)
    W = _weights(w1, w2, act=act)
    runner = layer.runners[0]
    pack = _logits_pack(xq, sx, logits)
    got = bits(layer(pack, W))  # the launch binds the hidden size the dispatcher reads
    c1, c2, fused = runner._default(T)
    assert c1.swap_ab and c2.swap_ab and not fused, (
        c1.describe(),
        c2.describe(),
        fused,
    )
    Wd = MoEWeightsE4M3BS.from_view(W.get_view(runner.backend_key))
    ids, wts = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, wts = ids.cuda(), wts.cuda()
    sm, fam = runner._part if runner._part is not None else (None, None)
    name = "silu" if gated else "relu2"
    p1, p2 = default_configs_fp8bs(
        T, k, H, I, E, sm, fam, name, gated, (), False, swap=False
    )
    assert not p1.swap_ab and not p2.swap_ab
    ref = bits(
        fused_moe_fp8bs(xq, sx, ids, wts, Wd, config1=p1, config2=p2, fuse_quant=False)
    )
    assert torch.equal(got, ref), (c1.describe(), c2.describe())
    n1 = 2 * I if gated else I
    ch1 = gemm1_chain(name, (), gated)
    for c in all_valid_configs_fp8bs_swapped(n1, H, E, ch1):
        out = fused_moe_fp8bs(
            xq, sx, ids, wts, Wd, config1=c, config2=p2, fuse_quant=False
        )
        assert torch.equal(bits(out), ref), ("gemm1", c.describe())
    for c in all_valid_configs_fp8bs_swapped(H, I, E, GEMM2_CHAIN):
        out = fused_moe_fp8bs(
            xq, sx, ids, wts, Wd, config1=p1, config2=c, fuse_quant=False
        )
        assert torch.equal(bits(out), ref), ("gemm2", c.describe())
    tactics = runner.get_valid_tactics(runner.pack_inputs(pack, W), None)
    # both records swapped (field 6 of each 7-field record), the standalone spelling last
    assert tactics[0][6] == 1 and tactics[0][13] == 1 and tactics[0][14] == 0, tactics[
        0
    ]
    with autotune(True):
        assert torch.equal(bits(layer(pack, W)), ref)
    assert torch.equal(bits(layer(pack, W)), ref)


@needs_device
def test_routing_methods_from_logits_match_the_cpu_model():
    """Every routing method the class implements, from logits, selects and weights as the exact CPU model; the layer's bits follow."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import spec as S

    T, k, H, I, E = 64, 4, 512, 256, 32
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=7, adversarial=True)
    xq, sx = _xq(x)
    bias = (
        (torch.randn(E, generator=torch.Generator().manual_seed(8)) * 0.1)
        .float()
        .cuda()
    )
    W = _weights(w1, w2)
    for method in (
        RoutingMethodType.Default,
        RoutingMethodType.Renormalize,
        RoutingMethodType.Sigmoid,
        RoutingMethodType.SigmoidRenorm,
        RoutingMethodType.DeepSeekV3,
    ):
        rm = S.METHOD_OF_FLASHINFER[method]
        kwargs = {"routed_scaling_factor": 1.75} if rm.scaled else {}
        layer = MoELayer(
            MoEConfig(
                routing=RoutingConfig(num_experts=E, top_k=k, method=method, **kwargs),
                quant=QUANT,
                experts=ExpertConfig(intermediate_size=I),
                activation=SwiGLU(),
                backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
                execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=T),
            ),
            device=torch.device("cuda"),
        )
        b = bias if rm.bias else None
        out = layer(_logits_pack(xq, sx, logits, b), W)
        ex, w = R.route_cpu(
            logits.cpu(),
            k,
            rm,
            b.cpu() if b is not None else None,
            1.75 if rm.scaled else None,
        )
        ref = layer(_routed_pack(xq, sx, ex.cuda(), w.cuda()), W)
        assert torch.equal(bits(out), bits(ref)), method


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    """The error contract: a configuration outside the classes raises the mode's error at build or at the first call (top_k /
    num_experts outside the bounds, fused shared experts, an expert-parallel shard, group-limited routing, no finalize, an
    intermediate size that is not a multiple of 128, a hidden size that is not, bf16 activations handed to the fp8 layer, no
    activation scale tensor, a scale tensor of the wrong shape, weights prepared for another activation, an unimplemented routing
    method from logits); a zero-token call returns an empty output."""
    k, H, I, E = 6, 512, 256, 64
    errors = (bi.BatchInvariantUnsupportedError, RuntimeError, NotImplementedError)
    for cfg_kw, what in (
        (dict(k=17), "top_k"),
        (dict(E=1025), "num_experts"),
        (dict(I=320), "intermediate_size"),
    ):
        with pytest.raises(errors) as excinfo:
            _layer(cfg_kw.get("k", k), cfg_kw.get("E", E), cfg_kw.get("I", I))
        assert what in str(excinfo.value) or "moe.e4m3bs" in str(excinfo.value)
    routing = RoutingConfig(num_experts=E, top_k=k)
    experts = ExpertConfig(intermediate_size=I)
    for kw, what in (
        (
            dict(
                routing=RoutingConfig(
                    num_experts=E, top_k=k, method=RoutingMethodType.DeepSeekV3
                ),
                experts=ExpertConfig(intermediate_size=I, num_fused_shared_experts=1),
            ),
            "fused shared experts",
        ),
        (
            dict(
                experts=ExpertConfig(
                    intermediate_size=I,
                    local_expert_offset=E // 2,
                    local_num_experts=E // 2,
                )
            ),
            "parallel",
        ),
        (
            dict(
                routing=RoutingConfig(num_experts=E, top_k=k, n_group=2, topk_group=1)
            ),
            "n_group",
        ),
        (dict(finalize=MoEFinalizeConfig(do_finalize=False)), "do_finalize"),
    ):
        with pytest.raises(errors) as excinfo:
            MoELayer(
                MoEConfig(
                    routing=kw.get("routing", routing),
                    quant=QUANT,
                    experts=kw.get("experts", experts),
                    backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
                    finalize=kw.get("finalize", MoEFinalizeConfig(do_finalize=True)),
                ),
                device=torch.device("cuda"),
            )
        assert what in str(excinfo.value) or "moe.e4m3bs" in str(excinfo.value), what
    # the bf16 and the per-tensor fp8 quantization pairs never reach this runner
    for quant in (
        QuantConfig(),
        QuantConfig(
            weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
        ),
    ):
        with pytest.raises(errors):
            MoELayer(
                MoEConfig(
                    routing=routing,
                    quant=quant,
                    experts=experts,
                    backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
                ),
                device=torch.device("cuda"),
            )
    layer = _layer(k, E, I)
    x, logits, w1, w2 = _problem(8, k, H, I, E, seed=9)
    xq, sx = _xq(x)
    W = _weights(w1, w2)
    with pytest.raises(errors):
        layer(
            _logits_pack(x, None, logits), W
        )  # bf16 activations: no runner accepts the pack
    with pytest.raises(errors):
        layer(_logits_pack(xq, None, logits), W)  # no block scales
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        layer(
            _logits_pack(xq, sx[:1].contiguous(), logits), W
        )  # a scale tensor of the wrong shape
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        layer(
            _logits_pack(xq, sx, logits), _weights(w1[:, :I].contiguous(), w2, ReLU2())
        )  # the ungated weights with the gated class
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        layer(
            _logits_pack(xq, sx, logits), _weights(w1, w2, GeGLU())
        )  # another activation's chain
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        _layer(k, E, I)(
            _logits_pack(xq[:, :384].contiguous(), sx[:3].contiguous(), logits), W
        )  # H = 384 does not match the weights
    llama = MoELayer(
        MoEConfig(
            routing=RoutingConfig(
                num_experts=E, top_k=1, method=RoutingMethodType.Llama4
            ),
            quant=QUANT,
            experts=experts,
            backend=BackendOptions((BatchInvariantFp8BlockConfig(),)),
        ),
        device=torch.device("cuda"),
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        llama(_logits_pack(xq, sx, logits), W)
    ids = torch.zeros(8, 1, dtype=torch.int32, device="cuda")
    assert llama(
        _routed_pack(xq, sx, ids, torch.ones(8, 1, device="cuda")), W
    ).shape == (8, H)
    out = layer(_logits_pack(xq[:0], sx[:, :0].contiguous(), logits[:0]), W)
    assert out.shape == (0, H) and out.dtype == torch.bfloat16


@needs_device
def test_alternating_inputs_under_programmatic_dependent_launch():
    """Two problems run alternately through the op, back to back without host syncs (the kernels' PDL chains overlap and every
    reused intermediate holds the other call's bytes), many times, in both requantization spellings; every output equals the
    problem's isolated result bit for bit."""
    from flashinfer.fused_moe.batch_invariant import api
    from flashinfer.fused_moe.batch_invariant import spec as S
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        fused_moe_fp8bs,
    )

    T, k, H, I, E = 64, 8, 256, 128, 256
    probs = []
    for seed, fused in ((3, False), (11, True)):
        x, logits, w1, w2 = _problem(T, k, H, I, E, seed=seed, adversarial=True)
        W = MoEWeightsE4M3BS.prepare(w1, w2)
        xq, sx = _xq(x)
        ids, wts = api.route(logits, k, S.SOFTMAX_RENORM)
        ref = bits(fused_moe_fp8bs(xq, sx, ids, wts, W, fuse_quant=fused))
        torch.cuda.synchronize()
        probs.append((xq, sx, ids, wts, W, fused, ref))
    outs = []
    for _ in range(20):
        for xq, sx, ids, wts, W, fused, _r in probs:
            outs.append(fused_moe_fp8bs(xq, sx, ids, wts, W, fuse_quant=fused))
    torch.cuda.synchronize()
    for i, out in enumerate(outs):
        assert torch.equal(bits(out), probs[i % 2][6]), i


def _flashinfer_block_fp8_reference(
    x, w1, w2, ids, weights, act, activation_fn=None
) -> torch.Tensor:
    """FlashInfer's torch reference of the DeepSeek block-fp8 MoE: the form of ``tests/moe/test_unified_moe_fp8.py``'s
    ``_block_fp8_reference`` (the reference its TRTLLM-Gen backend is checked against): the activations quantized per 1x128
    block and the weights per 128x128 block by FlashInfer's own recipe (``prepare.py``: ``(amax / 448).clamp(1e-12)``, the
    division, the clamp) and dequantized to fp32; fp32 GEMM1; the typed activation in fp32 (``tests.moe.utils``'s formulas, the
    layer's own); the intermediate requantized through bf16 the same way; fp32 GEMM2; the routing weights narrowed to bf16;
    the fp32 routed sum. Independent of the class (its exact model is the promise, tested bit for bit above).
    ``activation_fn`` replaces the activation (the negative control of the tolerance)."""
    from flashinfer.fused_moe.prepare import (
        _deepseek_fp8_quantize_activations,
        _deepseek_fp8_quantize_weights,
    )
    from tests.moe.utils import compute_reference_activation, powlu_ref

    def deq_act(t):
        q, s = _deepseek_fp8_quantize_activations(t.to(torch.bfloat16))
        return q.float() * s.transpose(0, 1).repeat_interleave(BLOCK, dim=1)

    def deq_w(w):
        q, s = _deepseek_fp8_quantize_weights(w)
        return q.float() * s.repeat_interleave(BLOCK, dim=1).repeat_interleave(
            BLOCK, dim=2
        )

    weights = weights.to(torch.bfloat16).float()
    x_deq = deq_act(x)
    w1_deq, w2_deq = deq_w(w1), deq_w(w2)
    I = w2.shape[2]
    if activation_fn is None:

        def activation_fn(gemm1):
            if isinstance(act, PowLU):
                up, gate = gemm1.split(I, dim=-1)
                g = powlu_ref(gate, act.m)
                if act.limit is not None:
                    g = g.clamp_max(act.limit)
                    up = up.clamp(-act.limit, act.limit)
                return g * up
            return compute_reference_activation(gemm1, act, I, out_dtype=torch.float32)

    out = torch.zeros_like(x_deq)
    for e in range(w1.shape[0]):
        tok, slot = torch.where(ids == e)
        if tok.numel() == 0:
            continue
        inter = activation_fn(x_deq[tok] @ w1_deq[e].t())
        y = deq_act(inter) @ w2_deq[e].t()
        out[tok] += weights[tok, slot, None] * y
    return out


def _wrong_activation(act, I: int):
    """A genuinely different formula over the same weights and scales (FlashInfer's negative control: the tolerance must reject
    it): the up branch alone for a gated activation, plain ReLU for an ungated one (the identity for ReLU itself)."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    if activation_of(act).gated:
        return lambda gemm1: gemm1[:, :I]
    if isinstance(act, ReLU):
        return lambda gemm1: gemm1
    return torch.relu


@needs_device
@pytest.mark.parametrize("act", ACTIVATIONS, ids=ACTIVATION_IDS)
def test_plausible_against_flashinfers_block_fp8_torch_reference(act):
    """An observation on moderate inputs under FlashInfer's own discipline for this quantization pair
    (``tests/moe/test_unified_moe_fp8.py``: ``check_accuracy`` at atol 0.05 / rtol 0.3 on 99 % of the outputs against the torch
    reference above, and the negative control that the tolerance rejects a wrong activation formula). Not a promise of the
    class (the reference and the class round at different points; the exact model is the promise, tested bit for bit above),
    and never a comparison with another backend's output."""
    from flashinfer.fused_moe.batch_invariant.spec import activation_of
    from tests.moe.trtllm_gen_fused_moe_utils import check_accuracy

    T, k, H, I, E = 128, 4, 512, 256, 32
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=11, gated=activation_of(act).gated)
    xq, sx = _xq(x)
    out = _layer(k, E, I, act)(
        _logits_pack(xq, sx, logits), _weights(w1, w2, act)
    ).float()
    vals, ids = torch.topk(
        logits, k, dim=-1
    )  # the layer's Renormalize routing: top-k, softmax over the selected
    weights = torch.softmax(vals, dim=-1)
    ref = _flashinfer_block_fp8_reference(x, w1, w2, ids, weights, act)
    check_accuracy(out, ref, atol=0.05, rtol=0.3, percent=0.99)
    wrong = _flashinfer_block_fp8_reference(
        x, w1, w2, ids, weights, act, activation_fn=_wrong_activation(act, I)
    )
    with pytest.raises(Exception, match="Mismatch percentage"):
        check_accuracy(out, wrong, atol=0.05, rtol=0.3, percent=0.99)


# ----------------------------------------------------------------------------- the 32-wide classes (deepseek_fp8_block_size = 32)

BLOCK32 = 32


def _weights32(w1: torch.Tensor, w2: torch.Tensor, act=_SWIGLU) -> MoEWeightPack:
    pack = MoEWeightPack({})
    view = BatchInvariantFp8BlockConfig.prepare_weights(
        w1,
        w2,
        quant=QUANT,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        activation=act,
        deepseek_fp8_block_size=BLOCK32,
    )
    pack.prepare_for(BatchInvariantFp8BlockRunner.backend_key, view)
    return pack


def _xq32(x: torch.Tensor):
    xq, sx = BatchInvariantFp8BlockConfig.prepare_activations(
        x, quant=QUANT, deepseek_fp8_block_size=BLOCK32
    )
    assert tuple(sx.shape) == (x.shape[1] // BLOCK32, x.shape[0])
    return xq, sx


def _block_partials32(a: torch.Tensor, w: torch.Tensor, block: int) -> torch.Tensor:
    """``P[kb, m, n]`` of the per-tensor fp8 class's fold of each ``block``-wide K slice from +0 (the dense kernel's zero-filled tail
    steps are identities: at 32 the slice is one instruction step), through ``mm_fp8_batch_invariant`` at the pinned record."""
    from tests.gemm.test_batch_invariant_gemm_fp8_blockscale import PARTIAL_RECORD

    from flashinfer.gemm.batch_invariant.api import mm_fp8_batch_invariant

    M, K = a.shape
    N = w.shape[0]
    N16 = -(-N // 16) * 16
    if N16 != N:
        w = torch.cat([w, torch.zeros(N16 - N, K, dtype=w.dtype, device=w.device)], 0)
    one = torch.ones(1, dtype=torch.float32, device=a.device)
    P = torch.empty(K // block, M, N16, dtype=torch.float32, device=a.device)
    for kb in range(K // block):
        mm_fp8_batch_invariant(
            a[:, kb * block : (kb + 1) * block].contiguous(),
            w[:, kb * block : (kb + 1) * block].contiguous().t(),
            one,
            one,
            out=P[kb],
            out_dtype=torch.float32,
            config=PARTIAL_RECORD,
            chunks=1,
        )
    return P[:, :, :N] if N16 != N else P


def _exact_promotion32(P, sa, sb, block: int):
    """The class's fp32 fold over the block partials with ``block``-column weight scale blocks (the 128-wide helper with the block as
    a parameter)."""
    from flashinfer.batch_invariant.fp_interface import exact

    P, sa, sb = P.cpu(), sa.cpu().float(), sb.cpu().float()
    KB, M, N = P.shape
    col_blk = torch.arange(N) // block
    acc = torch.zeros(M, N, dtype=torch.float32)
    tiny = 2 * 2.0**-126
    for kb in range(KB):
        scale = (sa[:, kb].double()[:, None] * sb[col_blk, kb].double()[None, :]).to(
            torch.float32
        )
        t = scale.double() * P[kb].double()
        acc64 = acc.double()
        s = t + acc64
        bb = s - t
        err = (t - (s - bb)) + (acc64 - bb)
        out = s.to(torch.float32)
        midpoint = (s.view(torch.int64) & ((1 << 29) - 1)) == (1 << 28)
        risky = (err != 0) & torch.isfinite(s) & (midpoint | (s.abs() < tiny))
        for m, n in risky.nonzero(as_tuple=False).tolist():
            out[m, n] = exact.fma(
                float(scale[m, n]), float(P[kb, m, n]), float(acc[m, n])
            )
        acc = out
    return acc


def _exact_acc32(a, sa_mn, w, w_sf, counts, canon_rows, sel, block: int):
    """The 32-wide core, exactly, on every selected expert's rows of the dense packing."""
    Tk = a.shape[0]
    N = w.shape[1]
    acc = torch.zeros(Tk, N, dtype=torch.float32)
    sa = sa_mn.t().cpu()
    off = 0
    for e, c in enumerate(counts):
        if c and e in sel:
            rows = a[off : off + c].contiguous()
            if canon_rows is not None:
                w_canon = torch.empty_like(w[e])
                w_canon[canon_rows] = w[e]
                acc[off : off + c] = _exact_promotion32(
                    _block_partials32(rows, w_canon, block),
                    sa[off : off + c],
                    w_sf[e],
                    block,
                )[:, canon_rows]
            else:
                acc[off : off + c] = _exact_promotion32(
                    _block_partials32(rows, w[e], block),
                    sa[off : off + c],
                    w_sf[e],
                    block,
                )
        off += c
    return acc


def test_block32_is_its_own_class_with_the_same_chains_and_spaces():
    """``deepseek_fp8_block_size`` is a dimension of the class: the fingerprint, the class name (``moe.e4m3bs32.<act>``), the GEMM
    core (``gemm.e4m3.kblock32``: one step per block, a different term from the 128-wide core at the same K) and the quantizer's
    instance differ at 32 while the 128-wide class's values are unchanged; the chains are the same; the record spaces and the
    acceptance read the block; the dispatcher's picks at 32 are members; the registry refuses another block."""
    from flashinfer.fused_moe.batch_invariant import mterm
    from flashinfer.fused_moe.batch_invariant import spec_fp8bs as SB
    from flashinfer.fused_moe.batch_invariant.blocks.fp8bs_compile import (
        GEMM2_CHAIN,
        can_implement_fp8bs,
        gemm1_chain,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        all_valid_configs_fp8bs,
        all_valid_configs_fp8bs_swapped,
        default_configs_fp8bs,
    )
    from flashinfer.gemm.batch_invariant.spec import GemmSpec

    dims = dict(H=2048, I=768, top_k=8, num_experts=128)
    fp, fp32 = (
        bi.fingerprint("fused_moe_fp8bs", **dims),
        bi.fingerprint("fused_moe_fp8bs", deepseek_fp8_block_size=32, **dims),
    )
    assert fp != fp32 and fp32.startswith("moe-")
    assert bi.fingerprint("fused_moe_fp8bs", deepseek_fp8_block_size=128, **dims) == fp
    with pytest.raises(ValueError):
        bi.fingerprint("fused_moe_fp8bs", deepseek_fp8_block_size=64, **dims)
    assert (
        SB.quant_fingerprint(32) != SB.quant_fingerprint() == SB.quant_fingerprint(128)
    )
    assert SB.gemm_core_fingerprint(256, "f32", 32).startswith(
        "gemm-e4m3-kblock32-"
    ) and SB.gemm_core_fingerprint(256, "f32", 128).startswith("gemm-e4m3-kblock128-")
    s32 = SB.Fp8BsMoESpec(256, 128, 2, 8, block=32)
    assert (
        s32.name == "moe.e4m3bs32.swiglu"
        and SB.Fp8BsMoESpec(256, 128, 2, 8, act="relu2", gated=False, block=32).name
        == "moe.e4m3bs32.relu2"
    )
    s128 = SB.Fp8BsMoESpec(256, 128, 2, 8, alg=s32.alg)
    assert mterm.sexpr(s32.gemm1()) == mterm.sexpr(s128.gemm1()) and mterm.sexpr(
        s32.gemm2()
    ) == mterm.sexpr(s128.gemm2())
    from flashinfer.batch_invariant.fp_interface.term import TermAlgebra, sexpr

    alg = TermAlgebra()
    core32 = sexpr(SB.reduction_kblock(32, 0, 0, 32).canonical(alg))
    # one instruction step per block, promoted once; the 128-wide core at K = 128 is four steps promoted once
    assert (
        core32
        == "(fma (mma32 (const 0.0) (ab (a 0 0) (b 0 0))) (mul (sfa_blk 0 0) (sfb_blk 0 0)) (const 0.0))"
    )
    core128 = sexpr(SB.reduction_kblock(128, 0, 0, 128).canonical(alg))
    assert (
        core128.startswith("(fma (mma32 (mma32 (mma32 (mma32 (const 0.0)")
        and core128.count("(mul (sfa_blk") == 1
    )
    assert (
        sexpr(SB.reduction_kblock(128, 0, 0, 32).canonical(alg)).count("(mul (sfa_blk")
        == 4
    )
    assert "(sfb_blk 1 0)" in sexpr(SB.reduction_kblock(32, 3, 33, 32).canonical(alg))
    assert sexpr(SB.reduction_kblock(128, 0, 0, 32).canonical(alg)) != sexpr(
        GemmSpec(128, "f32", in_fmt="e4m3_bs").acc(0, 0)
    )
    # the spaces and the acceptance at 32: the same records, whole 32-row blocks (the halves of the interleaved weight each), no
    # fused requantization
    assert (
        len(all_valid_configs_fp8bs(1280, 2560, 513, gemm1_chain("silu"), block=32))
        == 54
    )
    assert (
        len(all_valid_configs_fp8bs_swapped(1280, 2560, 513, gemm1_chain("silu"), 32))
        == 8
    )
    assert (
        all_valid_configs_fp8bs(
            1280, 2560, 513, gemm1_chain("silu", fused=True), block=32
        )
        == []
    )
    assert (
        len(
            all_valid_configs_fp8bs(1280 + 64, 2560, 513, gemm1_chain("silu"), block=32)
        )
        == 54
    )
    assert (
        all_valid_configs_fp8bs(1280 + 32, 2560, 513, gemm1_chain("silu"), block=32)
        == []
    )
    assert (
        all_valid_configs_fp8bs(1280, 2560 + 64, 513, gemm1_chain("silu"), block=32)
        == []
    )
    for T in (1, 7, 64, 300, 4096):
        for family, sm in (("sm_107", 212), ("sm_100", 148)):
            for act, gated in (("silu", True), ("relu2", False)):
                c1, c2 = default_configs_fp8bs(
                    T,
                    8,
                    2048,
                    768,
                    128,
                    sm,
                    family,
                    act,
                    gated,
                    (),
                    False,
                    nominal=True,
                    block=32,
                )
                assert can_implement_fp8bs(
                    c1,
                    2 * 768 if gated else 768,
                    2048,
                    128,
                    gemm1_chain(act, (), gated),
                    None,
                    32,
                ) and can_implement_fp8bs(c2, 2048, 768, 128, GEMM2_CHAIN, None, 32)
                if T <= 7:
                    assert c1.swap_ab and c2.swap_ab


def test_block32_weight_view_and_quantizer_reference():
    """``prepare_weights(..., deepseek_fp8_block_size=32)`` quantizes per 32x32 block by the class's reference at 32, writes the block
    into the view (read back by the runner), and the activation preparation at 32 is the quantizer's reference; a view of another
    block is told apart by its scales' shape; the MXFP8 pair refuses the parameter."""
    from flashinfer.fused_moe.batch_invariant import reference_fp8bs as RB
    from flashinfer.fused_moe.batch_invariant.api import interleave_fc1
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import MoEWeightsE4M3BS

    g = torch.Generator().manual_seed(5)
    E, I, H = 3, 128, 256
    w1 = (
        torch.randn(E, 2 * I, H, generator=g)
        * torch.exp2(torch.randint(-6, 7, (E, 2 * I, H), generator=g).float())
    ).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, generator=g) * 0.05).to(torch.bfloat16)
    w1[1, 32:64, :32] = 0.0
    view = BatchInvariantFp8BlockConfig.prepare_weights(
        w1, w2, quant=QUANT, activation=SwiGLU(), deepseek_fp8_block_size=32
    )
    assert view["deepseek_fp8_block_size"] == 32
    W = MoEWeightsE4M3BS.from_view(view)
    assert W.block == 32 and W.in_fmt == "e4m3_bs32"
    assert tuple(W.w1_sf.shape) == (E, 2 * I // 32, H // 32) and tuple(
        W.w2_sf.shape
    ) == (
        E,
        H // 32,
        I // 32,
    )
    q1 = torch.stack(
        [RB.e4m3_bytes(RB.quantize_weight_reference(w1[e], 32)[0]) for e in range(E)]
    )
    s1 = torch.stack([RB.quantize_weight_reference(w1[e], 32)[1] for e in range(E)])
    assert torch.equal(
        W.w1.view(torch.uint8).cpu(), interleave_fc1(q1)
    ) and torch.equal(bits(W.w1_sf.cpu()), bits(s1))
    for e in range(E):
        rq, rs = RB.quantize_weight_reference(w2[e], 32)
        assert torch.equal(
            W.w2[e].view(torch.uint8).cpu(), RB.e4m3_bytes(rq)
        ) and torch.equal(bits(W.w2_sf[e].cpu()), bits(rs)), e
    # the block read off the scales' shape when the view lacks the key (a caller's quantization)
    view2 = dict(view)
    del view2["deepseek_fp8_block_size"]
    assert MoEWeightsE4M3BS.from_view(view2).block == 32
    assert MoEWeightsE4M3BS.block_of(W.w1, W.w1_sf) == 32
    with pytest.raises(ValueError):
        BatchInvariantFp8BlockConfig.prepare_weights(
            w1,
            w2,
            quant=QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8),
            activation=SwiGLU(),
            deepseek_fp8_block_size=32,
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        MoEWeightsE4M3BS.prepare(w1, w2, block=64)
    # the quantizer at 32 on the CPU reference: the vectorised model is the exact model block for block
    x = _adversarial_rows(4, 128, seed=11)
    codes, s = RB.quantize_rows_reference(x, "K", 32)
    assert tuple(s.shape) == (4, 4)
    for r in range(4):
        for b in range(4):
            s_e, _inv, codes_e = RB.exact_block(
                x[r, b * 32 : (b + 1) * 32].float().tolist()
            )
            assert (
                codes_e == codes[r, b * 32 : (b + 1) * 32].tolist()
                and s_e == s[r, b].item()
            ), (r, b)


@needs_device
@pytest.mark.parametrize("act", [SwiGLU(), ReLU2()], ids=["swiglu", "relu2"])
def test_block32_layer_matches_the_class_composition(act):
    """The 32-wide class end to end: the layer (from logits) == the class composed from its own pieces -- the quantizer at 32 on the
    device, the exact 32-wide core per (selected) expert (one instruction step per block, the exact fp32 promotion with the 32-row
    weight blocks), the CPU chain, the quantizer's reference on the device's bf16 intermediate at 32, GEMM2, the combine; every
    record of both spaces gives the same bits; the activation scales of the wrong block are refused."""
    from flashinfer.fused_moe.batch_invariant import reference as R
    from flashinfer.fused_moe.batch_invariant import reference_fp8bs as RB
    from flashinfer.fused_moe.batch_invariant.api_fp8bs import (
        MoEWeightsE4M3BS,
        fused_moe_fp8bs,
    )
    from flashinfer.fused_moe.batch_invariant.schedule_fp8bs import (
        all_valid_configs_fp8bs,
        all_valid_configs_fp8bs_swapped,
    )
    from flashinfer.fused_moe.batch_invariant.spec import activation_of

    clusters = ((1, 1), (1, 2), (2, 1))
    T, k, H, I, E = 300, 6, 512, 256, 64
    chain = activation_of(act)
    x, logits, w1, w2 = _problem(
        T, k, H, I, E, seed=5, gated=chain.gated, adversarial=True
    )
    xq, sx = _xq32(x)
    rq, rs = RB.quantize_rows_bytes(x, "MN", 32)
    assert torch.equal(xq.view(torch.uint8).cpu(), rq) and torch.equal(
        bits(sx.cpu()), bits(rs)
    )
    Wp = _weights32(w1, w2, act)
    layer = _layer(k, E, I, act)
    out = layer(_logits_pack(xq, sx, logits), Wp)
    ids, weights = _reference_route(logits, k, RoutingMethodType.Renormalize)
    ids, weights = ids.cuda(), weights.cuda()
    W = MoEWeightsE4M3BS.from_view(
        Wp.get_view(BatchInvariantFp8BlockRunner.backend_key)
    )
    assert W.block == 32
    ref_out, im = fused_moe_fp8bs(xq, sx, ids, weights, W, return_intermediates=True)
    assert not im["fused"] and im["a"] is not None
    assert torch.equal(bits(out), bits(ref_out))
    counts = im["counts"].cpu().tolist()
    Tk = T * k
    assert tuple(im["sx_perm"].shape) == (H // 32, Tk) and tuple(im["sa"].shape) == (
        I // 32,
        Tk,
    )
    sel = _selected(counts, 16)
    mask = torch.zeros(Tk, dtype=torch.bool)
    off = 0
    for e, c in enumerate(counts):
        if e in sel:
            mask[off : off + c] = True
        off += c
    N1 = W.w1.shape[1]
    acc1 = _exact_acc32(
        im["x_perm"],
        im["sx_perm"],
        W.w1,
        W.w1_sf,
        counts,
        RB.orig_rows(N1) if W.gated else None,
        sel,
        32,
    )
    a_ref = RB.chain_cpu(acc1, chain)
    a = im["a"].cpu()
    assert torch.equal(bits(a[mask]), bits(a_ref[mask])), (
        "gemm1",
        (a[mask] != a_ref[mask]).sum().item(),
    )
    aq_ref, sa_ref = RB.quantize_rows_bytes(a, "MN", 32)
    assert torch.equal(im["aq"].view(torch.uint8).cpu(), aq_ref) and torch.equal(
        bits(im["sa"].cpu()), bits(sa_ref)
    )
    acc2 = _exact_acc32(im["aq"], im["sa"], W.w2, W.w2_sf, counts, None, sel, 32)
    y = im["y"].cpu()
    assert torch.equal(bits(y[mask]), bits(R.TorchF32.cvt_out(acc2, "bf16")[mask])), (
        "gemm2"
    )
    assert torch.equal(
        bits(ref_out.cpu()), bits(R.combine_cpu(im["y"], im["dest"], weights))
    )
    # every record of both spaces at the class (the fast clusters) gives the same bits
    ch1 = W.chain1(False)
    space1 = all_valid_configs_fp8bs(
        N1, H, E, ch1, clusters, block=32
    ) + all_valid_configs_fp8bs_swapped(N1, H, E, ch1, 32)
    space2 = all_valid_configs_fp8bs(
        H, I, E, block=32, clusters=clusters
    ) + all_valid_configs_fp8bs_swapped(H, I, E, block=32)
    assert len(space1) >= 18 + 8 and len(space2) >= 18 + 8
    ref = bits(ref_out.cpu())
    for c in space1:
        assert torch.equal(
            bits(
                fused_moe_fp8bs(
                    xq, sx, ids, weights, W, config1=c, config2=space2[0]
                ).cpu()
            ),
            ref,
        ), c.describe()
    for c in space2:
        assert torch.equal(
            bits(
                fused_moe_fp8bs(
                    xq, sx, ids, weights, W, config1=space1[0], config2=c
                ).cpu()
            ),
            ref,
        ), c.describe()
    # the fused requantization is not a spelling of this class; the 128-wide activation scales are refused with these weights
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        fused_moe_fp8bs(xq, sx, ids, weights, W, fuse_quant=True)
    xq128, sx128 = _xq(x)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        layer(_logits_pack(xq128, sx128, logits), Wp)


@needs_device
def test_block32_tactics_and_graph_replay_keep_the_bits():
    """Every tactic the autotuner may pick at the 32-wide class computes the class; the autotuned layer and a CUDA-graph replay give
    the same bits; the 128-wide and the 32-wide layers are different classes on the same weights' values (their outputs differ)."""
    from flashinfer.autotuner import autotune

    T, k, H, I, E = 64, 4, 512, 256, 32
    x, logits, w1, w2 = _problem(T, k, H, I, E, seed=9, adversarial=True)
    xq, sx = _xq32(x)
    W = _weights32(w1, w2)
    layer = _layer(k, E, I)
    pack = _logits_pack(xq, sx, logits)
    ref = bits(layer(pack, W))
    runner = layer.runners[0]
    inputs = runner.pack_inputs(pack, W)
    tactics = runner.get_valid_tactics(inputs, None)
    assert tactics and all(len(t) == 15 and t[14] == 0 for t in tactics)
    for t in tactics:
        assert torch.equal(
            bits(runner.forward(runner.pack_inputs(pack, W), tactic=t)), ref
        )
    with autotune(True):
        assert torch.equal(bits(layer(pack, W)), ref)
    assert torch.equal(bits(layer(pack, W)), ref)
    s = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out_g = layer(pack, W)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out_g), ref)
    xq128, sx128 = _xq(x)
    out128 = _layer(k, E, I)(_logits_pack(xq128, sx128, logits), _weights(w1, w2))
    assert not torch.equal(bits(out128), ref)
