# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The block-scale fp8 MoE classes ``moe.e4m3bs.<act>`` (DeepSeek-style fp8): the quantizer entry points, the weight view, the
grouped block-scale GEMMs (placements of the dense class ``gemm.e4m3.kblock128``) and the fused FFN on pre-routed tokens.

The routing, the deterministic permutation and the slot-order combine are the bf16 classes' (:mod:`.api`). The operands follow
FlashInfer's DeepSeek conventions: e4m3 activations with one fp32 scale per 1x128 block (``hidden_states_scale [H / 128, T]``,
the layout :meth:`TrtllmFp8BlockConfig.prepare_activations` hands around; the ``[T, H / 128]`` layout is taken too), e4m3
experts with one fp32 scale per 128x128 block (``[E, N / 128, K / 128]``). Both quantizations are the ``fp8bs.quant`` class
(:mod:`.spec_fp8bs`); the class takes a caller's quantization as given (it starts at the bytes and the scales).

Pipeline: the dense permutation plan -> the e4m3 row gather and the block-scale gather (``sx_perm [H / 128, Tk]``: the MN-major
layout the grouped GEMM's row-scale ring reads) -> grouped block-scale GEMM1 (the dense class per expert: the per-block fold
promoted with the token row's and the weight block's scales; the bf16 class's activation chain, one bf16 rounding) -> the
intermediate's requantization per 1x128 block by the ``fp8bs.quant`` class, either fused into GEMM1's epilogue (the wide-tile
records) or as the standalone pass on the materialised bf16 intermediate -- two placements of one term, the same bytes ->
grouped block-scale GEMM2 (bf16 out) -> the bf16 class's slot-order combine. Every output bit is a function of the token's own
e4m3 row and block scales, its ``k`` experts and routing weights, the expert bytes and block scales: batch size and
composition, row position, expert load, the grouped records and the requantization spelling never enter the dataflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from ...batch_invariant.fp_interface import exact as fp32
from ...batch_invariant.support import unsupported
from ...gemm.batch_invariant.api import persistent_clusters, sm_count_of
from ...gemm.batch_invariant.generator import Knobs
from ...gemm.batch_invariant.spec import BLOCK_K, SCALE_BLOCK_N
from . import spec as S
from .api import interleave_fc1, permute_plan
from .blocks.fp8bs_compile import GEMM2_CHAIN, OUT_FMT_QUANT, Fp8BsChain, gemm1_chain
from .blocks.scheduler import MAX_E
from .reference import TorchF32
from .spec_fp8bs import BLOCK as QBLOCK
from .spec_fp8bs import CLASS, E4M3_MAX, block_chain, quant_pre

ALIGN_E4M3 = 16  # e4m3 elements per 16-byte row: the TMA alignment of every operand
TAIL_ROWS = 256  # rows past the live rows of every packed buffer: the largest cluster tile's TMA box stays in bounds
_OUT_DTYPE = {"bf16": torch.bfloat16, "f32": torch.float32}
_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "f16"}
SCALE_LAYOUTS = ("MN", "K")
#: Below this many warps per SM the quantizer's launch is latency-bound: more warps share a row (empirical; a placement).
ROW_WARPS_PER_SM_SMALL = 16
#: The route kernel's top-k bound; the class's bound on the slots.
K_MAX = 16


# ----------------------------------------------------------------------------- the quantizer
def quant_warps_per_row(M: int, K: int, sms: Optional[int] = None) -> int:
    """The quantizer's placement at (rows, K): warps per row, a power of two of at most 8 and at most the row's block pairs,
    doubled while the launch holds fewer than ``ROW_WARPS_PER_SM_SMALL`` warps per SM (a placement: the block trees never move)."""
    pairs = max(1, (K // QBLOCK + 1) // 2)
    cap = 1
    while cap * 2 <= min(8, pairs):
        cap *= 2
    sms = sms or 1
    w = 1
    while w * 2 <= cap and M * w * 2 <= ROW_WARPS_PER_SM_SMALL * sms:
        w *= 2
    return w


def fp8_block_quantize(
    x: torch.Tensor,
    scale_layout: str = "MN",
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    wpr: Optional[int] = None,
    enable_pdl: bool = True,
    trigger: str = "none",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(q [M, K] float8_e4m3fn, s fp32)`` of ``x [M, K]`` (bf16 / f16, 16-byte-aligned rows, ``K`` a whole number of 128-element
    blocks): per block ``s = fmaxc(div(amax |x|, 448), 1e-12)``, ``q = cvt.rn.satfinite.e4m3(mul(x, div(1, s)))``. ``s`` is
    ``[K / 128, M]`` (``"MN"``, FlashInfer's DeepSeek activation layout) or ``[M, K / 128]`` (``"K"``). ``out = (q, s)`` writes
    into the caller's tensors (``s`` in the requested layout, any strides). Bitwise identical for every ``wpr``
    (:func:`quant_warps_per_row`), every batch composition and across runs."""
    from .blocks.fp8bs_compile import get_compiled_fp8bs_quant

    if x.dim() != 2 or x.dtype not in _DTYPE_STR:
        raise unsupported(
            "fp8bs.quant",
            f"input {tuple(x.shape)} of {x.dtype}",
            "a 2-D bf16 / f16 tensor",
        )
    M, K = x.shape
    if K <= 0 or K % QBLOCK:
        raise unsupported(
            "fp8bs.quant", f"K = {K}", f"a whole number of {QBLOCK}-element blocks"
        )
    if x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16:
        raise unsupported(
            "fp8bs.quant",
            f"rows with strides {tuple(x.stride())}",
            "16-byte-aligned contiguous rows",
        )
    if scale_layout not in SCALE_LAYOUTS:
        raise unsupported(
            "fp8bs.quant", f"scale_layout {scale_layout!r}", '"MN" or "K"'
        )
    KB = K // QBLOCK
    want = (KB, M) if scale_layout == "MN" else (M, KB)
    if out is None:
        q = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
        s = torch.empty(want, dtype=torch.float32, device=x.device)
    else:
        q, s = out
        if (
            tuple(q.shape) != (M, K)
            or q.dtype != torch.float8_e4m3fn
            or q.stride(1) != 1
            or q.stride(0) % 16
        ):
            raise ValueError(
                "fp8_block_quantize: out[0] must be a 16-byte-aligned row-major e4m3 [M, K]"
            )
        if s.dtype != torch.float32 or tuple(s.shape) != want:
            raise ValueError(
                f"fp8_block_quantize: out[1] must be an fp32 {want} for scale_layout={scale_layout!r}"
            )
    if M == 0:
        return q, s
    s_view = (
        s.t() if scale_layout == "MN" else s
    )  # the kernel addresses (M, KB) through the tensor's strides
    w = wpr if wpr is not None else quant_warps_per_row(M, K, sm_count_of(x.device))
    get_compiled_fp8bs_quant(_DTYPE_STR[x.dtype], w, trigger, enable_pdl)(
        x, q.view(torch.uint8), s_view, M, K
    )
    return q, s


def _on_device(dev: torch.device):
    """``TorchF32`` with its constants on ``dev``: a tensor divided by a CPU scalar is multiplied by the scalar's reciprocal by
    torch's CUDA kernel (two roundings); a 0-d device tensor divisor takes the IEEE division."""

    class OnDevice(TorchF32):
        @staticmethod
        def const(v):
            return torch.tensor(fp32.to_f32(v), dtype=torch.float32, device=dev)

    return OnDevice


def quantize_weight_blocks(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The class per 128x128 block of ``w [N, K]`` (bf16): ``(q [N, K] float8_e4m3fn, s [N / 128, K / 128] fp32)`` as torch ops
    on the device, each a single IEEE rounding (``amax`` exact, ``div`` / ``mul`` correctly rounded, the saturating RNE
    conversion of the exact model); the rounded values are e4m3 values, so the final cast is exact. The same chain as the
    activations' blocks."""
    if w.dim() != 2 or w.dtype != torch.bfloat16:
        raise unsupported(
            "fp8bs.quant", f"weight {tuple(w.shape)} of {w.dtype}", "a 2-D bf16 weight"
        )
    N, K = w.shape
    if N % QBLOCK or K % QBLOCK or N == 0 or K == 0:
        raise unsupported(
            "fp8bs.quant", f"weight ({N}, {K})", f"whole {QBLOCK}x{QBLOCK} blocks"
        )
    blocks = (
        w.to(torch.float32)
        .view(N // QBLOCK, QBLOCK, K // QBLOCK, QBLOCK)
        .permute(0, 2, 1, 3)
    )
    amax = blocks.abs().amax(dim=(-1, -2), keepdim=True)
    s, inv = block_chain(_on_device(w.device), amax)
    # the conversion: torch's RNE cast after the clamp to the e4m3 range (the clamp makes it the saturating conversion on
    # finite values)
    y = quant_pre(TorchF32, blocks, inv).clamp(-float(E4M3_MAX), float(E4M3_MAX))
    q = y.permute(0, 2, 1, 3).reshape(N, K).to(torch.float8_e4m3fn)
    return q, s[..., 0, 0].contiguous()


# ----------------------------------------------------------------------------- the weights
def check_shape_fp8bs(H: int, I: int, E: int, k: Optional[int] = None) -> None:
    """The class's shape rule: ``H`` and ``I`` positive multiples of 128 (whole 128-element scale blocks of the quantized
    activations (K = H), of the quantized intermediate (K = I) and whole 128-row scale blocks of both expert weights, the
    gated weight's up and gate halves each; the 16-byte e4m3 rows and the gate / up pairing are implied), ``1 <= E <= MAX_E``
    (the group tables), ``1 <= k <= min(E, 16)``. Everything else about a shape is placement; the class has no split variant."""
    if H <= 0 or I <= 0 or H % QBLOCK or I % QBLOCK:
        raise unsupported(
            CLASS,
            f"hidden size {H}, intermediate size {I}",
            f"positive multiples of {QBLOCK} (whole scale blocks of every operand)",
        )
    if not 1 <= E <= MAX_E:
        raise unsupported(CLASS, f"num_experts {E}", f"1 <= num_experts <= {MAX_E}")
    if k is not None and not 1 <= k <= min(E, K_MAX):
        raise unsupported(
            CLASS, f"top_k {k}", f"1 <= top_k <= min(num_experts, {K_MAX})"
        )


@dataclass
class MoEWeightsE4M3BS:
    """The block-scale experts as the kernels consume them: ``w1 [E, N1, H]`` e4m3 (``N1 = 2I`` gate / up interleaved in groups of 8
    rows for the gated chains, ``I`` as given for the ungated ones), ``w1_sf [E, N1 / 128, H / 128]`` fp32 -- the scales of the
    ORIGINAL rows' 128x128 blocks (the interleave permutes the quantized rows only; the kernel reads an interleaved column's
    scale from its original block), ``w2 [E, H, I]`` e4m3, ``w2_sf [E, H / 128, I / 128]``; ``act`` / ``params`` name the GEMM1 chain."""

    w1: torch.Tensor
    w1_sf: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    gated: bool = True
    act: str = "silu"
    params: Tuple[Optional[float], ...] = ()

    @staticmethod
    def prepare(
        fc1: torch.Tensor,
        fc2: torch.Tensor,
        gated: bool = True,
        act: Optional[str] = None,
        params: Sequence[Optional[float]] = (),
        quantized: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
    ) -> "MoEWeightsE4M3BS":
        """Quantize bf16 experts (``fc1 [E, 2I, H]`` rows ``[up, gate]``, or ``[E, I, H]`` (``gated=False``), ``fc2 [E, H, I]``) per
        128x128 block with the ``fp8bs.quant`` chain (:func:`quantize_weight_blocks`, one expert at a time), then interleave the
        gated weight's quantized rows. ``quantized = (w1_q, w1_sf, w2_q, w2_sf)``: the caller's block quantization of the canonical
        (non-interleaved) weights, e4m3 ``[E, N1, H]`` with fp32 ``[E, N1 / 128, H / 128]`` scales, taken as given (FlashInfer's
        DeepSeek weight view: ``gemm1_weights`` / ``gemm1_weights_scale`` / ``gemm2_weights`` / ``gemm2_weights_scale``)."""
        act = act if act is not None else ("silu" if gated else "relu2")
        if act not in (S.ACTS if gated else S.UNGATED_ACTS):
            raise ValueError(f"no {'gated' if gated else 'ungated'} chain {act!r}")
        key: Tuple[Optional[float], ...] = S.params_key(params)
        if len(key) != (len(S.ACT_PARAMS.get(act, ())) if gated else 0):
            raise ValueError(
                f"chain {act!r} takes parameters {S.ACT_PARAMS.get(act, ())}, got {key!r}"
            )
        if quantized is None:
            if (
                fc1.dtype != torch.bfloat16
                or fc2.dtype != torch.bfloat16
                or fc1.dim() != 3
                or fc2.dim() != 3
            ):
                raise TypeError(
                    f"{CLASS}: the classes take bf16 expert weights fc1 [E, N1, H], fc2 [E, H, I]"
                )
            E, N1, H = fc1.shape
        else:
            w1q, w1s, w2q, w2s = quantized
            if (
                w1q.dtype != torch.float8_e4m3fn
                or w2q.dtype != torch.float8_e4m3fn
                or w1q.dim() != 3
                or w2q.dim() != 3
            ):
                raise TypeError(
                    f"{CLASS}: the quantized experts are e4m3 [E, N1, H] and [E, H, I]"
                )
            E, N1, H = w1q.shape
        I = N1 // 2 if gated else N1
        check_shape_fp8bs(H, I, E)
        if quantized is not None:
            if tuple(w2q.shape) != (E, H, I):
                raise ValueError(
                    f"w2 {tuple(w2q.shape)} does not match w1 {tuple(w1q.shape)} (gated={gated})"
                )
            if (
                tuple(w1s.shape) != (E, N1 // QBLOCK, H // QBLOCK)
                or tuple(w2s.shape) != (E, H // QBLOCK, I // QBLOCK)
                or w1s.dtype != torch.float32
                or w2s.dtype != torch.float32
            ):
                raise ValueError(
                    f"{CLASS}: the block scales are fp32 [E, N / 128, K / 128] of each weight"
                )
            w1q, w1s, w2q, w2s = (
                w1q.contiguous(),
                w1s.contiguous(),
                w2q.contiguous(),
                w2s.contiguous(),
            )
        else:
            if tuple(fc2.shape) != (E, H, I):
                raise ValueError(
                    f"fc2 {tuple(fc2.shape)} does not match fc1 {tuple(fc1.shape)} (gated={gated})"
                )
            w1q = torch.empty(E, N1, H, dtype=torch.float8_e4m3fn, device=fc1.device)
            w1s = torch.empty(
                E, N1 // QBLOCK, H // QBLOCK, dtype=torch.float32, device=fc1.device
            )
            w2q = torch.empty(E, H, I, dtype=torch.float8_e4m3fn, device=fc1.device)
            w2s = torch.empty(
                E, H // QBLOCK, I // QBLOCK, dtype=torch.float32, device=fc1.device
            )
            for e in range(E):  # one expert at a time: bounded fp32 temporaries
                w1q[e], w1s[e] = quantize_weight_blocks(fc1[e])
                w2q[e], w2s[e] = quantize_weight_blocks(fc2[e])
        if gated:
            w1q = interleave_fc1(
                w1q
            )  # a row permutation of the quantized rows: no arithmetic
        return MoEWeightsE4M3BS(w1q.contiguous(), w1s, w2q, w2s, gated, act, key)

    @property
    def num_experts(self) -> int:
        return self.w1.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.w1.shape[2]

    @property
    def intermediate_size(self) -> int:
        return self.w1.shape[1] // 2 if self.gated else self.w1.shape[1]

    def chain1(self, fused: bool) -> Fp8BsChain:
        """GEMM1's chain: the activation's bf16 output, or its fused requantization."""
        return gemm1_chain(self.act, self.params, self.gated, fused)

    def view(self) -> Dict[str, Any]:
        """The weight view a :class:`flashinfer.fused_moe.MoEWeightPack` stores for this backend."""
        return {
            "w1": self.w1,
            "w1_sf": self.w1_sf,
            "w2": self.w2,
            "w2_sf": self.w2_sf,
            "gated": self.gated,
            "act": self.act,
            "params": self.params,
        }

    @staticmethod
    def from_view(view: Dict[str, Any]) -> "MoEWeightsE4M3BS":
        gated = bool(view.get("gated", True))
        return MoEWeightsE4M3BS(
            view["w1"],
            view["w1_sf"],
            view["w2"],
            view["w2_sf"],
            gated,
            str(view.get("act", "silu" if gated else "relu2")),
            tuple(view.get("params", ())),
        )


# ----------------------------------------------------------------------------- the grouped GEMM
def scale_mn_view(s: torch.Tensor, KB: int, rows: int, name: str) -> torch.Tensor:
    """A block-scale tensor of ``rows`` rows as the kernel's ``(rows, KB)`` view with its own strides: the MN layout ``[KB, rows]``
    (FlashInfer's DeepSeek activation layout, the layout the block-scale GEMM's row-scale ring reads) or the K layout
    ``[rows, KB]`` (accepted when the shapes tell them apart)."""
    if not isinstance(s, torch.Tensor) or s.dtype != torch.float32:
        raise unsupported(
            CLASS,
            f"{name} of {type(s).__name__ if not isinstance(s, torch.Tensor) else s.dtype}",
            f"an fp32 {name}",
        )
    if tuple(s.shape) == (KB, rows):
        return s.t()
    if tuple(s.shape) == (rows, KB):
        return s
    raise unsupported(
        CLASS,
        f"{name} of shape {tuple(s.shape)}",
        f"[{KB}, {rows}] (one fp32 scale per 1x{QBLOCK} block) or [{rows}, {KB}]",
    )


def grouped_gemm_fp8bs(
    a: torch.Tensor,
    sfa: torch.Tensor,
    w: torch.Tensor,
    w_sf: torch.Tensor,
    counts: torch.Tensor,
    config: Knobs,
    chain: Fp8BsChain = GEMM2_CHAIN,
    out: Optional[torch.Tensor] = None,
    sfo: Optional[torch.Tensor] = None,
    interleaved: Optional[bool] = None,
):
    """``out[p] = chain( fold_kb fma(mul(sfa[p, kb], w_sf[e(p), blk(n), kb]), P_kb, acc) )`` on the dense expert segments
    ``counts``: ``a [Tk, K]`` e4m3, ``sfa [K / 128, Tk]`` fp32 (MN; or ``[Tk, K / 128]``), ``w [E, N, K]`` e4m3, ``w_sf [E, N /
    128, K / 128]`` fp32. ``chain``: the bf16 / f32 chains (one tensor returned) or the fused requantization (``out_fmt =
    "e4m3bs"``: ``(codes [Tk, Nout] float8_e4m3fn, scales [Nout / 128, Tk] fp32)``). ``interleaved``: the weight's rows are gate /
    up interleaved (default: the chain's pairing; the f32 oracle output of GEMM1 reads the interleaved weight with the plain
    chain). The persistent grid is a launch argument."""
    from .blocks.fp8bs_compile import (
        can_implement_grouped_fp8bs,
        get_compiled_grouped_fp8bs,
    )

    if (
        a.dtype != torch.float8_e4m3fn
        or w.dtype != torch.float8_e4m3fn
        or a.dim() != 2
        or w.dim() != 3
    ):
        raise TypeError("grouped_gemm_fp8bs takes e4m3 a [Tk, K] and w [E, N, K]")
    Tk, K = a.shape
    E, N, K2 = w.shape
    if (
        K != K2
        or a.stride(1) != 1
        or not w.is_contiguous()
        or a.stride(0) % ALIGN_E4M3
        or K % BLOCK_K
    ):
        raise ValueError(
            "grouped_gemm_fp8bs: K-contiguous, 16-byte-aligned operands of one K, a whole number of scale blocks "
            f"(a {tuple(a.shape)} strides {tuple(a.stride())}, w {tuple(w.shape)})"
        )
    if (
        counts.dtype != torch.int32
        or tuple(counts.shape) != (E,)
        or not counts.is_contiguous()
    ):
        raise ValueError("grouped_gemm_fp8bs: counts must be a contiguous int32 [E]")
    KB, NB = K // BLOCK_K, N // SCALE_BLOCK_N
    if (
        tuple(w_sf.shape) != (E, NB, KB)
        or w_sf.dtype != torch.float32
        or not w_sf.is_contiguous()
    ):
        raise ValueError(
            f"grouped_gemm_fp8bs: w_sf must be a contiguous fp32 {(E, NB, KB)}, got {tuple(w_sf.shape)}"
        )
    if not can_implement_grouped_fp8bs(config, N, K, E, chain, interleaved):
        raise ValueError(
            f"grouped_gemm_fp8bs: {config.describe()} does not implement (N={N}, K={K}, E={E}, {chain.describe()}, interleaved={interleaved})"
        )
    sfa_v = scale_mn_view(sfa, KB, Tk, "sfa")
    n_out = N // 2 if chain.pair else N
    quant = chain.out_fmt == OUT_FMT_QUANT
    if quant:
        if out is None:
            out = torch.empty(Tk, n_out, dtype=torch.float8_e4m3fn, device=a.device)
        if sfo is None:
            sfo = torch.empty(n_out // QBLOCK, Tk, dtype=torch.float32, device=a.device)
        if (
            out.dtype != torch.float8_e4m3fn
            or tuple(out.shape) != (Tk, n_out)
            or out.stride(1) != 1
            or out.stride(0) % 16
        ):
            raise ValueError(
                "grouped_gemm_fp8bs: out must be a 16-byte-aligned row-major e4m3 [Tk, n_out]"
            )
        if (
            tuple(sfo.shape) != (n_out // QBLOCK, Tk)
            or sfo.dtype != torch.float32
            or not sfo.is_contiguous()
        ):
            raise ValueError(
                f"grouped_gemm_fp8bs: sfo must be a contiguous fp32 [{n_out // QBLOCK}, {Tk}]"
            )
        out_k, sfo_k = out.view(torch.uint8), sfo
    else:
        out_dtype = _OUT_DTYPE[chain.out_fmt]
        if out is None:
            out = torch.empty(Tk, n_out, dtype=out_dtype, device=a.device)
        if (
            out.dtype != out_dtype
            or tuple(out.shape) != (Tk, n_out)
            or out.stride(1) != 1
            or out.stride(0) % 8
        ):
            raise ValueError(
                "grouped_gemm_fp8bs: out must be a 16-byte-aligned row-major [Tk, n_out] of the chain's dtype"
            )
        out_k, sfo_k = (
            out,
            w_sf[0, :1, :1].reshape(1, 1),
        )  # the scale output is unread: any fp32 (1, 1) tensor
    if Tk:
        get_compiled_grouped_fp8bs(config, chain, interleaved)(
            a.unsqueeze(-1),
            sfa_v,
            w.permute(1, 2, 0),
            w_sf,
            out_k,
            sfo_k,
            counts,
            persistent_clusters(a.device, config),
        )
    return (out, sfo) if quant else out


# ----------------------------------------------------------------------------- the op
def fused_moe_fp8bs(
    xq: torch.Tensor,
    sx: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w: MoEWeightsE4M3BS,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
    fuse_quant: Optional[bool] = None,
    return_intermediates: bool = False,
    plan: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
):
    """The block-scale fp8 MoE FFN for pre-routed tokens: ``xq [T, H]`` e4m3 with ``sx`` its block scales (``[H / 128, T]``, or
    ``[T, H / 128]``), ``token_selected_experts [T, k]`` int32, ``token_final_scales [T, k]`` fp32; ``out[t] = sum_r w[t, r] * y_r``
    with the class's chains and rounding (:mod:`.spec_fp8bs`), the ``k`` slots folded in slot order in fp32 and converted once.
    ``config1`` / ``config2``: the grouped records (any record of the space gives the same bits; the dispatcher's pick by
    default); ``fuse_quant``: the intermediate's requantization in GEMM1's epilogue (True) or by the standalone pass (False):
    a placement, the same bytes either way (:func:`.schedule_fp8bs.fused_quant_default` by default). ``plan``:
    ``route_plan``'s ``(counts, dest, src)``."""
    from .blocks.fp8bs_compile import get_compiled_gather_scales
    from .combine import get_compiled_combine, warps_per_token
    from .permute import get_compiled_gather, warps_per_row
    from .schedule_fp8bs import default_configs_fp8bs, fused_quant_default

    if (
        xq.dtype != torch.float8_e4m3fn
        or xq.dim() != 2
        or xq.stride(1) != 1
        or xq.stride(0) % ALIGN_E4M3
    ):
        raise TypeError(
            "fused_moe_fp8bs: xq must be a row-major, 16-byte-aligned e4m3 [T, H]"
        )
    T, H = xq.shape
    if w.hidden_size != H:
        raise ValueError(
            f"fused_moe_fp8bs: xq has H = {H}, the weights H = {w.hidden_size}"
        )
    k = token_selected_experts.shape[1]
    if (
        tuple(token_selected_experts.shape) != (T, k)
        or token_selected_experts.dtype != torch.int32
    ):
        raise TypeError(
            "fused_moe_fp8bs: token_selected_experts must be an int32 [T, k]"
        )
    if (
        tuple(token_final_scales.shape) != (T, k)
        or token_final_scales.dtype != torch.float32
    ):
        raise TypeError("fused_moe_fp8bs: token_final_scales must be a float32 [T, k]")
    experts = token_selected_experts.contiguous()
    scales = token_final_scales.contiguous()
    E, I = w.num_experts, w.intermediate_size
    check_shape_fp8bs(H, I, E, k)
    dev = xq.device
    KB = H // QBLOCK
    sx_kt = scale_mn_view(
        sx, KB, T, "hidden_states_scale"
    ).t()  # as [KB, T] (a view; the gather reads its strides)
    if out is None:
        out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    if (
        tuple(out.shape) != (T, H)
        or out.dtype != torch.bfloat16
        or out.stride(1) != 1
        or out.stride(0) % 8
    ):
        raise ValueError(
            "fused_moe_fp8bs: out must be a row-major, 16-byte-aligned bf16 [T, H]"
        )
    if fuse_quant is None:
        fuse_quant = fused_quant_default(T, k, E) if config1 is None else False
    chain1 = w.chain1(fuse_quant)
    if config1 is None or config2 is None:
        from ...gemm.batch_invariant.api import part_of

        sm_count, family = part_of(dev)
        c1, c2 = default_configs_fp8bs(
            T, k, H, I, E, sm_count, family, w.act, w.gated, w.params, fuse_quant
        )
        config1, config2 = config1 or c1, config2 or c2
    counts, dest, src = (
        plan if plan is not None else permute_plan(experts, E)
    )  # ``plan``: ``route_plan``'s outputs
    Tk = T * k
    # the packed row buffers (and their scale columns) carry TAIL_ROWS past Tk so every tile's TMA box stays in bounds (an
    # out-of-bounds box is served far more slowly than the rows it zero-fills); the tail rows are never stored or read back
    Tk_a = Tk + TAIL_ROWS if Tk else 0
    x_perm = torch.empty(Tk_a, H, dtype=torch.float8_e4m3fn, device=dev)
    sx_perm = torch.empty(KB, Tk_a, dtype=torch.float32, device=dev)
    if Tk:
        get_compiled_gather(k, "e4m3", warps_per_row(Tk, H // 16, sm_count_of(dev)))(
            xq, src, x_perm[:Tk], Tk, H
        )  # the live rows: the kernel's row count is the index tensor's
        get_compiled_gather_scales(k)(sx_kt, src.reshape(-1), sx_perm, Tk, KB)
    a = None
    if fuse_quant:
        aq, sa = grouped_gemm_fp8bs(
            x_perm, sx_perm, w.w1, w.w1_sf, counts, config1, chain1
        )  # [Tk_a, I] e4m3 + [I / 128, Tk_a]
    else:
        a = grouped_gemm_fp8bs(
            x_perm, sx_perm, w.w1, w.w1_sf, counts, config1, chain1
        )  # [Tk_a, I] bf16
        aq = torch.empty(Tk_a, I, dtype=torch.float8_e4m3fn, device=dev)
        sa = torch.empty(I // QBLOCK, Tk_a, dtype=torch.float32, device=dev)
        if Tk:
            fp8_block_quantize(
                a[:Tk], "MN", out=(aq[:Tk], sa[:, :Tk])
            )  # the live rows only
    y = grouped_gemm_fp8bs(
        aq, sa, w.w2, w.w2_sf, counts, config2, GEMM2_CHAIN
    )  # [Tk_a, H] bf16
    if T:
        get_compiled_combine(
            k, "bf16", "bf16", warps_per_token(T, H, sm_count_of(dev))
        )(y[:Tk], dest, scales, out, T, H)
    if return_intermediates:
        # the live rows under the common keys; the tail-carrying buffers the kernels consume under ``*_buf`` (a kernel timed
        # on a live-row view reintroduces the out-of-bounds boxes)
        return out, dict(
            counts=counts,
            dest=dest,
            src=src,
            x_perm=x_perm[:Tk],
            sx_perm=sx_perm[:, :Tk],
            a=None if a is None else a[:Tk],
            aq=aq[:Tk],
            sa=sa[:, :Tk],
            y=y[:Tk],
            x_perm_buf=x_perm,
            sx_perm_buf=sx_perm,
            a_buf=a,
            aq_buf=aq,
            sa_buf=sa,
            y_buf=y,
            config1=config1,
            config2=config2,
            fused=fuse_quant,
            chain1=chain1,
        )
    return out


__all__ = [
    "ALIGN_E4M3",
    "CLASS",
    "K_MAX",
    "ROW_WARPS_PER_SM_SMALL",
    "SCALE_LAYOUTS",
    "TAIL_ROWS",
    "MoEWeightsE4M3BS",
    "check_shape_fp8bs",
    "fp8_block_quantize",
    "fused_moe_fp8bs",
    "grouped_gemm_fp8bs",
    "quant_warps_per_row",
    "quantize_weight_blocks",
    "scale_mn_view",
]
