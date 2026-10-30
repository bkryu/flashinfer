"""The batch-invariant class ``mxfp8.quant`` behind ``flashinfer.mxfp8_quantize(..., backend="auto", batch_invariant=True)``.

Quick tier (every CI job):

* device-free: the exact CPU model (vectorised reference == the symbolic spec evaluated with Fractions), the fingerprint
  pin, the engine's plans on every configuration, the closed-form dispatcher's membership over a (M, K, dtype, SM count)
  grid incl. the SM counts 148 / 152 / 204 / 224, the compile key never containing M, the cache closure, the Lean
  generated-cases conformance (regenerate + byte-compare, no toolchain), the backend rule and the fingerprint API;
* GPU (skipped off SM100 / SM103 / SM107): GPU == exact CPU model on adversarial rows and the quantization edge families
  in the three layouts, FlashInfer's output contract (shapes, dtype, alignment padding, >2-D and strided inputs),
  determinism incl. CUDA-graph replay and PDL, batch / permutation / sub-batch invariance, configuration agreement over
  the dispatcher's picks for every SKU, bitwise identity with FlashInfer's ``cute-dsl`` backend outside the documented
  tiny-block corner (and the corner itself pinned), the backend-rule errors, the unsupported-raises rule.

Nightly tier (``FLASHINFER_BI_NIGHTLY=1`` or ``-m nightly``): the whole enumerated configuration space per K / dtype /
layout, the reference against the exact spec under every recipe variant, TE's bf16 production path identity, the
exhaustive UE8M0 device sweep over all 2^32 bit patterns.

Every assertion is bitwise; a tolerance is never used to pass a test.
"""

import inspect
import math
import os
import re
import struct
import time

import pytest
import torch

import flashinfer
from flashinfer import SfLayout
from flashinfer.batch_invariant import (
    CLASSES,
    SUPPORTED_ARCHS,
    BatchInvariantUnsupportedError,
    class_name,
    fingerprint,
    is_supported_device,
    resolve_backend,
)
from flashinfer.batch_invariant import canon, fp32, jit, lean_cases, term
from flashinfer.batch_invariant.kernel_names import (
    MX_QUANT_MODULE,
    mx_quant_kernel_name,
)
from flashinfer.batch_invariant.quant import layout as L
from flashinfer.batch_invariant.quant import mx_reference as R
from flashinfer.batch_invariant.quant import mx_schedule as C
from flashinfer.batch_invariant.quant import mx_spec as S
from flashinfer.batch_invariant.quant.mx_api import (
    mx_quantize,
    mxfp8_quantize_batch_invariant,
)
from flashinfer.batch_invariant.testing import adversarial_rows

FMT = S.MXFP8
FMTS = [S.MXFP8, S.MXFP4]
RULES = [
    S.DEFAULT_RULE,
    S.OCP_FLOOR_RULE,
    S.FI_CLAMP_RULE,
    S.TE_FMA_RULE,
    S.ScaleRule(norm="div"),
    S.ScaleRule(round="rn"),
    S.ScaleRule(norm="pow2", round="rp"),
]
DT = [torch.bfloat16, torch.float16, torch.float32]
EB = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}
DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
SM_COUNTS = [148, 152, 204, 224]
GRID_M = [1, 2, 7, 16, 37, 256, 1000, 4096, 8192, 16384, 65536]
GRID_K = [2048, 4096, 7168, 8192, 16384]
KS_CPU = [32, 64, 96, 256, 1024, 2048, 4096, 7168, 8192, 16384]
# GPU quick tier: the K that exercise one block, a partial scale-column tile, a 2-block bucket, the Kbp < 4 padding
# columns, the bench grid's common sizes and the largest bucket
KS_GPU = [32, 96, 1024, 2048, 4096, 7168]
MS_GPU = [1, 37, 130]
KS_SPACE = [32, 96, 1024, 4096, 16384]  # nightly: whole-space agreement
KS_SPACE_OTHER = [64, 2048]
FI_LAYOUTS = {
    "128x4": SfLayout.layout_128x4,
    "8x4": SfLayout.layout_8x4,
    "linear": SfLayout.layout_linear,
}


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _bits(x):
    return struct.pack("<f", x)


def _same(a, b):
    return all(torch.equal(p, q) for p, q in zip(a, b, strict=True))


# =============================================================================== device-free: the spec and the engine
def test_fingerprint_is_pinned():
    """The class fingerprint is the contract: TE's chain (RN 1/448 constant, RCEIL, the exact inverse at every byte,
    the bare saturating e4m3 conversion, -0 keeps its sign). A change here is a release-note item."""
    assert (
        S.spec_fingerprint(S.MXFP8)
        == S.FINGERPRINT_MXFP8
        == "mxquant-mxfp8-9018bd1627b3e72a"
    )
    assert (
        S.spec_fingerprint(S.MXFP4)
        == S.FINGERPRINT_MXFP4
        == "mxquant-mxfp4-af5e163620dac20c"
    )
    assert (
        S.DEFAULT_RULE
        == S.TE_RULE
        == S.ScaleRule("rcp", "rp", "exact", "satfinite", "keep")
    )
    assert (
        fingerprint("mxfp8_quantize") == S.FINGERPRINT_MXFP8
        and class_name("mxfp8_quantize") == "mxfp8.quant"
    )
    assert set(CLASSES) == {"mxfp8_quantize"}
    with pytest.raises(ValueError, match="no batch-invariant class"):
        fingerprint("rmsnorm")
    with pytest.raises(TypeError, match="takes the fixed dimensions"):
        fingerprint("mxfp8_quantize", K=4096)
    # every recipe variant is another class; the formats differ; K, the input dtype and the layout do not enter
    fps = {S.spec_fingerprint(fmt, r) for fmt in FMTS for r in RULES}
    assert len(fps) == len(FMTS) * len(RULES)
    a = term.TermAlgebra()
    assert term.sexpr(S.MXQuantSpec(32, alg=a).block(0, 0)["sf"]) == term.sexpr(
        S.MXQuantSpec(4096, alg=a).block(7, 0)["sf"]
    )


def test_term_structure():
    """The spec's terms have the documented structure: a radix-32 fmax tree over fabs leaves, mul by the RN constant,
    the UE8M0 ceil conversion, div(1, sf) and per element cvt_e4m3(mul(x, inv))."""
    K = 128
    for fmt in FMTS:
        s = S.MXQuantSpec(K, fmt)
        blk = s.block(0, 1)
        amax = blk["amax"]
        leaves, stack = set(), [amax]
        while stack:
            t = stack.pop()
            if t.op == "fabs":
                assert t.args[0].op == "x"
                leaves.add(t.args[0].args[1])
            else:
                assert t.op == "fmaxc"
                stack.extend(t.args)
        assert leaves == set(range(32, 64))
        sf = blk["sf"]
        assert sf.op == "cvt_out" and sf.args[1] == "ue8m0_rp"
        m = sf.args[0]
        assert m.op == "mul" and amax in m.args
        c = [a for a in m.args if a is not amax][0]
        assert c.op == "const" and c.args[0] == fp32.to_f32(1.0 / fmt.elem_max)
        inv = blk["inv"]
        assert (
            inv.op == "div"
            and inv.args[0].op == "const"
            and inv.args[0].args[0] == 1.0
            and inv.args[1] is sf
        )
        code = blk["codes"][5]
        assert (
            code.op == "cvt_out"
            and code.args[1] == fmt.elem_fmt
            and code.args[0].op == "mul"
            and inv in code.args[0].args
        )
        assert any(a.op == "x" and a.args[1] == 37 for a in code.args[0].args)
    assert (
        S.MXFP8.c_rcp == _f32(1 / 448)
        and struct.unpack("<I", struct.pack("<f", S.MXFP8.c_rcp))[0] == 0x3B124925
    )


@pytest.mark.parametrize("eb", [2, 4])
@pytest.mark.parametrize("K", KS_CPU)
def test_every_config_plan_verifies_and_block_terms_are_canonical(K, eb):
    """Every configuration of the enumerated space is a placement of the canonical reduction: the engine's trace of every
    simulated thread equals the canonical term, and the block states the kernel reads before the ownership digit are the
    canonical block terms (R3 at the model level)."""
    specs = {}
    n = 0
    for sw in L.LAYOUTS:
        for cfg in C.all_valid_configs(K, eb, sw):
            Kp = cfg.bucket(K) * 32
            if Kp not in specs:
                spec = S.MXQuantSpec(Kp)
                specs[Kp] = (
                    spec,
                    spec.row_reduction(0),
                    {b: spec.block_amax(0, b) for b in range(spec.nb)},
                )
            spec, red, blocks = specs[Kp]
            p = canon.plan(red, cfg.placement(Kp))
            assert canon.verify(p, spec.alg) is red.canonical(spec.alg), cfg.describe()
            states = canon.trace(p, spec.alg, stop_before="blk")
            for (cta, w, lane), st in states.items():
                coords = {"lane": lane, "warp": w, "cta": cta, "grid": 0}
                for key, val in st.items():
                    b = p.digit_value("blk", coords, key)
                    assert val is blocks.get(b), (cfg.describe(), b)
            n += 1
    assert n >= 20


@pytest.mark.parametrize("V", [32, 16, 8])
def test_block_amax_placement_is_canonical(V):
    alg = term.TermAlgebra()
    red = S.block_reduction(lambda fp, e: alg.x(0, e))
    p = canon.plan(red, S.block_amax_placement(V))
    assert canon.verify(p, alg) is red.canonical(alg)
    assert p.units["lane"] == 32 // V and p.units["warp"] == 1
    for lane in range(32 // V):
        coords = {"lane": lane, "warp": 0, "cta": 0, "grid": 0}
        owned = sorted(p.digit_value("elem", coords, k) for k in p.reg_keys())
        assert owned == list(range(lane * V, (lane + 1) * V))


@pytest.mark.parametrize("K", [32, 256, 4096, 7168])
def test_ownership_is_aligned(K):
    for cfg in C.all_valid_configs(K, 2) + C.all_valid_configs(K, 4):
        if cfg.mode == "tile":
            continue
        red = S.row_reduction(K, lambda fp, i: i)
        p = canon.plan(red, cfg.placement(K))
        for w in range(cfg.warps_per_row):
            for lane in range(cfg.lanes_per_row):
                coords = {"lane": lane, "warp": w, "cta": 0, "grid": 0}
                blocks = sorted({p.digit_value("blk", coords, k) for k in p.reg_keys()})
                assert blocks == sorted(cfg.owned_blocks(K, w, lane)), (
                    cfg.describe(),
                    w,
                    lane,
                )
                parts = {
                    p.digit_value("elem", coords, k) // cfg.elems_per_thread
                    for k in p.reg_keys()
                }
                assert parts == {cfg.owned_part(lane)}, (cfg.describe(), lane)


def test_engine_rejects_invalid_placements_configs_and_mutations():
    red = S.row_reduction(4096, lambda fp, i: i)
    for f in [
        {"elem": [(16, "reg")], "blk": [(128, "reg")]},
        {"elem": [(8, "reg"), (4, "lane")], "blk": [(16, "lane"), (8, "reg")]},
        {"elem": [(32, "reg")], "blk": [(2, "grid"), (64, "reg")]},
    ]:
        with pytest.raises((AssertionError, ValueError)):
            canon.plan(red, canon.Placement.make(f))
    for bad in [
        C.MXQuantConfig(3, 1),
        C.MXQuantConfig(1, 1, 16),
        C.MXQuantConfig(16, 2, 32),
        C.MXQuantConfig(1, 32, 32),
        C.MXQuantConfig(1, 128, 32, mode="tile"),
        C.MXQuantConfig(32, 64, 32, mode="tile"),
        C.MXQuantConfig(32, 128, 32, mode="tile", tile_warps=3),
    ]:
        with pytest.raises(ValueError):
            bad.validate(
                16384 if bad.threads_per_row == 1 and bad.rows_per_cta == 32 else 256
            )
    with pytest.raises(ValueError):
        C.MXQuantConfig(32, 128, 32, mode="tile").validate(4096, 2, swizzled=False)
    with pytest.raises(ValueError):
        C.MXQuantConfig(32, 4, 32).validate(16384, 4)
    spec = S.MXQuantSpec(4096)
    q = canon.plan(
        spec.row_reduction(0), C.MXQuantConfig(128, 1, 8, "lane").placement(4096)
    )
    canon.verify(q, spec.alg)
    lane_steps = [s for s in q.steps if s.level == "lane"]
    q.steps = [s for s in q.steps if s.level != "lane"] + lane_steps[::-1]
    with pytest.raises(AssertionError):
        canon.verify(q, spec.alg)


# =============================================================================== device-free: the exact CPU model
def test_ue8m0_exact_primitive():
    for rnd in fp32.UE8M0_ROUNDINGS:
        assert (
            fp32.to_ue8m0(0.0, rnd) == 2.0**-127
            and fp32.to_ue8m0(-0.0, rnd) == 2.0**-127
        )
        assert fp32.to_ue8m0(-5.0, rnd) == fp32.to_ue8m0(5.0, rnd)
        assert fp32.to_ue8m0(float("inf"), rnd) == 2.0**127 and math.isnan(
            fp32.to_ue8m0(float("nan"), rnd)
        )
        assert (
            fp32.to_ue8m0(2.0**-130, rnd) == 2.0**-127
            and fp32.to_ue8m0(2.0**-127, rnd) == 2.0**-127
        )
        for k in range(-126, 128):
            assert fp32.to_ue8m0(2.0**k, rnd) == 2.0**k, (rnd, k)
        assert fp32.ue8m0_byte(fp32.to_ue8m0(1.0, rnd)) == 127
    assert (
        fp32.to_ue8m0(1.5, "rp") == 2.0
        and fp32.to_ue8m0(1.5, "rz") == 1.0
        and fp32.to_ue8m0(1.5, "rn") == 2.0
    )
    assert (
        fp32.to_ue8m0(_f32(1.0 + 2**-23), "rp") == 2.0
        and fp32.to_ue8m0(_f32(1.0 + 2**-23), "rz") == 1.0
    )
    assert (
        fp32.to_ue8m0(2.0**127 * 1.5, "rp") == 2.0**127
        and fp32.to_ue8m0(fp32.F32_MAX, "rz") == 2.0**127
    )
    for b in range(255):
        assert fp32.ue8m0_byte(fp32.ue8m0_value(b)) == b
    assert fp32.ue8m0_byte(float("nan")) == 0xFF and _bits(
        fp32.ue8m0_value(0)
    ) == _bits(2.0**-127)
    # the boundary: amax = 448 * 2^k lands on byte 127 + k under the RN reciprocal (448 * fp32(1/448) rounds to 1.0)
    for k in (-10, 0, 7):
        v = fp32.mul(448.0 * 2.0**k, S.MXFP8.c_rcp)
        assert v == 2.0**k and fp32.ue8m0_byte(fp32.to_ue8m0(v, "rp")) == 127 + k
        v2 = fp32.mul(_f32(448.0 * 2.0**k * (1 + 2**-22)), S.MXFP8.c_rcp)
        assert fp32.ue8m0_byte(fp32.to_ue8m0(v2, "rp")) == 128 + k


def test_vectorised_ue8m0_matches_exact():
    import random

    rng = random.Random(0)
    vals = []
    for k in range(-149, 128):
        m = 2.0**k
        bits = struct.unpack("<I", struct.pack("<f", m))[0]
        for d in (-2, -1, 0, 1, 2):
            vals.append(
                struct.unpack("<f", struct.pack("<I", (bits + d) & 0xFFFFFFFF))[0]
            )
        if k < 127:
            bits = struct.unpack("<I", struct.pack("<f", 1.5 * m))[0]
            for d in (-1, 0, 1):
                vals.append(struct.unpack("<f", struct.pack("<I", bits + d))[0])
    vals += [_f32(rng.uniform(0, 1000)) for _ in range(5000)] + [
        _f32(rng.uniform(-1, 1) * 2.0 ** rng.randint(-140, 127)) for _ in range(5000)
    ]
    vals += [
        0.0,
        -0.0,
        float("inf"),
        -float("inf"),
        float("nan"),
        fp32.F32_MAX,
        -fp32.F32_MAX,
        2.0**-127,
        2.0**-149,
    ]
    t = torch.tensor(vals, dtype=torch.float32)
    for rnd in fp32.UE8M0_ROUNDINGS:
        got = R.ue8m0_round(t, rnd).tolist()
        exp = [fp32.to_ue8m0(v, rnd) for v in vals]
        bad = [
            (v, g, e)
            for v, g, e in zip(vals, got, exp, strict=True)
            if _bits(g) != _bits(e) and not (math.isnan(g) and math.isnan(e))
        ]
        assert not bad, (rnd, bad[:5])
        assert R.ue8m0_bytes(torch.tensor(exp)).tolist() == [
            fp32.ue8m0_byte(e) for e in exp
        ]


def _edge_rows(M, K, dtype, fmt=FMT, seed=0, device="cpu"):
    """Adversarial rows (wide exponents, zeros of both signs, subnormals, outliers) plus the recipe's corners: an all-zero
    row, an all-zero block, a block of -0, a tiny block (UE8M0 byte 0), infinities, the dtype's maximum, the
    ELEM_MAX x 2^k boundaries."""
    x = adversarial_rows(M, K, dtype, seed=seed)
    if M >= 8:
        x[1] = 0
        x[2, :32] = 0
        x[3, :32] = -0.0
        x[4] = (
            torch.rand(K, generator=torch.Generator().manual_seed(seed + 1)) * 2 - 1
        ).to(dtype) * 2.0**-125
        x[5, 0] = float("inf")
        x[6, min(33, K - 1)] = -float("inf")
        x[7] = torch.finfo(dtype).max
        if M >= 10:
            x[8, :32] = torch.tensor([fmt.elem_max] + [0.0] * 31).to(dtype)
            x[9, :32] = torch.tensor([fmt.elem_max * 1.75 * 2.0**-3] + [1e-3] * 31).to(
                dtype
            )
    return x.to(device)


@pytest.mark.parametrize("dtype", DT)
def test_vectorised_reference_equals_exact_spec(dtype):
    """The whole-tensor torch reference (what the GPU is compared with) equals the symbolic spec evaluated with the exact
    Fractions backend, bit for bit, on the edge families, in the three layouts; the recorded corners hold."""
    K = 96
    x = _edge_rows(10, K, dtype, seed=K)
    for sw in L.LAYOUTS:
        packed, sf, sf_rows = R.quantize_mx_reference(x, FMT, S.DEFAULT_RULE, sw)
        assert sf.numel() == L.sf_numel(10, K, 32, sw)
        assert torch.equal(R.logical_sf(sf, 10, K, sw), sf_rows)
        ep, esf = R.exact_rows(x, FMT, range(10))
        assert torch.equal(packed, ep) and torch.equal(sf_rows, esf), (dtype, sw)
    sfr = sf_rows
    vals = R.unpack_elems(packed, FMT)
    assert (
        int(sfr[1].max()) == 0
        and int(sfr[2, 0]) == 0
        and int(sfr[3, 0]) == 0
        and int(sfr[5, 0]) == 254
        and int(sfr[8, 0]) == 127
    )
    assert bool(torch.signbit(vals[3, :32]).all()) and bool(
        (vals[1] == 0).all()
    )  # -0 keeps its sign (code 0x80)
    assert vals[5, 0] == 448.0 and vals[6, 33] == -448.0 and int(sfr[4].max()) == 0
    if (
        dtype != torch.float16
    ):  # f16 has no values below 2^-24: its "tiny" row is exactly zero
        assert bool(
            (vals[4] != 0).any()
        )  # the tiny block keeps its values under the exact 2^127 inverse


@pytest.mark.nightly
@pytest.mark.parametrize("rule", RULES[1:], ids=lambda r: r.name)
@pytest.mark.parametrize("fmt", FMTS, ids=lambda f: f.name)
@pytest.mark.parametrize("dtype", DT)
def test_vectorised_reference_equals_exact_spec_every_rule(fmt, rule, dtype):
    K = 96
    x = _edge_rows(10, K, dtype, fmt=fmt, seed=K + fmt.elem_bits)
    for sw in L.LAYOUTS:
        packed, sf, sf_rows = R.quantize_mx_reference(x, fmt, rule, sw)
        ep, esf = R.exact_rows(x, fmt, range(10), rule)
        assert torch.equal(packed, ep) and torch.equal(sf_rows, esf), (
            fmt.name,
            rule.name,
            dtype,
            sw,
        )


def test_layout_formulas_roundtrip_and_match_flashinfer():
    """The three store placements round-trip and agree with FlashInfer's own swizzle formulas (tests/utils_fp8.py)."""
    from tests.utils_fp8 import _swizzle_mxfp8_scales

    M, K = 300, 2016
    nb = L.num_blocks(K, 32)
    sf = torch.randint(0, 255, (M, nb), dtype=torch.uint8)
    for name, lay in FI_LAYOUTS.items():
        ours = L.swizzle_sf(sf, M, K, 32, name)
        assert torch.equal(ours, _swizzle_mxfp8_scales(sf, lay)), name
        assert torch.equal(L.unswizzle_sf(ours, M, K, 32, name), sf)
        assert ours.numel() == L.sf_numel(M, K, 32, name)
    pc = L.padded_sf_cols(K, 32)
    for r, c in [(0, 0), (1, 0), (31, 3), (32, 0), (127, 62), (128, 0), (299, 61)]:
        assert int(
            L.swizzle_sf(sf, M, K, 32, "128x4")[L.sf_offset_128x4(r, c, pc)]
        ) == int(sf[r, c])
        assert int(L.swizzle_sf(sf, M, K, 32, "8x4")[L.sf_offset_8x4(r, c, pc)]) == int(
            sf[r, c]
        )
    assert L.sf_offset_8x4(9, 5, 8) == 101 and L.sf_offset_8x4(0, 4, 8) == 32


@pytest.mark.nightly
def test_te_bf16_production_path_is_bit_equivalent():
    """TransformerEngine's production bf16 kernel multiplies in bf16 (``mul.rn.bf16x2`` by the bf16 power-of-two inverse,
    then ``cvt.rn.satfinite.e4m3x2.bf16x2``): exhaustively over every finite bf16 value x every UE8M0 byte it equals the
    class's fp32 chain bit for bit."""
    from flashinfer.batch_invariant.quant.reference import e4m3_round

    bits16 = (
        torch.arange(0, 1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    )
    x = bits16.float()
    x = x[torch.isfinite(x)]
    bad = 0
    for e8 in range(255):
        inv = torch.tensor(math.ldexp(1.0, 127 - e8), dtype=torch.float32)
        assert inv.to(torch.bfloat16).float().item() == inv.item()
        f32_path = e4m3_round(x * inv)
        bf16_path = e4m3_round((x * inv).to(torch.bfloat16).float())
        bad += int((f32_path.view(torch.int32) != bf16_path.view(torch.int32)).sum())
    assert bad == 0, bad


# =============================================================================== device-free: the dispatcher and the compile key
@pytest.mark.parametrize("K", KS_CPU)
def test_dispatcher_returns_members_for_every_sku(K):
    """Every pick of the closed-form dispatcher over the (M, dtype, layout, SM count) grid is a member of the enumerated
    space, so the pick never changes bits (R3); the space is non-empty and duplicate-free."""
    for eb in (2, 4):
        for sw in L.LAYOUTS:
            space = C.all_valid_configs(K, eb, sw)
            assert len(space) >= 8 and len(set(space)) == len(space)
            assert (any(c.mode == "tile" for c in space)) == (sw == "128x4")
            rows = [c for c in space if c.mode == "row"]
            assert len({c.placement(K) for c in rows}) == len(rows)
            for M in GRID_M:
                for sm in SM_COUNTS:
                    cfg = C.default_config(M, K, eb, sm, sw)
                    assert cfg in space, (K, eb, sw, M, sm, cfg)


def test_dispatcher_is_closed_form_and_the_compile_key_has_no_m():
    """No measured table; the compiled specialization's name carries the config, the bucket, the dtype, the format, the
    recipe, the layout and PDL -- never M; the dispatcher's reach over the grid is a small finite set."""
    from flashinfer.batch_invariant.quant.mx_kernel import _compile_bucket

    assert not any(n.startswith("_DISPATCH") for n in dir(C)), (
        "a measured dispatch table crept in"
    )
    assert "M" not in inspect.signature(_compile_bucket).parameters
    assert "M" not in inspect.signature(mx_quant_kernel_name).parameters
    reach = {}
    for eb, dt in ((2, "bf16"), (4, "f32")):
        for sw in L.LAYOUTS:
            names = set()
            for K in GRID_K:
                for M in GRID_M + [3, 64, 128, 512, 2048, 32768]:
                    for sm in SM_COUNTS:
                        cfg = C.default_config(M, K, eb, sm, sw)
                        name = mx_quant_kernel_name(
                            FMT.name,
                            S.DEFAULT_RULE.name,
                            dt,
                            cfg.bucket(K),
                            cfg.describe(),
                            sw,
                            False,
                        )
                        assert not re.search(r"_M\d|_rows\d|_m\d", name), name
                        names.add(name)
            reach[(dt, sw)] = len(names)
    assert all(v <= 40 for v in reach.values()), reach
    assert (
        C.default_config(16384, 2048, 2, 204).mode == "tile"
        and C.default_config(4096, 2048, 2, 204).mode == "row"
    )
    assert C.default_config(16384, 2048, 2, 204, swizzled="linear").mode == "row"
    assert C.default_config(16384, 2048, 2, 204, swizzled="8x4").mode == "row"


def test_dispatcher_is_monotone_in_the_sm_count():
    """More SMs never buy a pick with fewer CTAs; the tile tier never switches on with more SMs."""
    counts = [148, 152, 204, 224, 296]

    def ctas(cfg, M, K):
        if cfg.mode == "tile":
            return -(-M // 128) * -(-C.num_blocks(K) // cfg.tile_blocks)
        return -(-M // cfg.rows_per_cta)

    for K in GRID_K + [96, 1024]:
        for eb in (2, 4):
            for sw in (True, False):
                space = set(C.all_valid_configs(K, eb, sw))
                for M in GRID_M + [512, 2048]:
                    picks = [C.default_config(M, K, eb, sm, sw) for sm in counts]
                    assert all(p in space for p in picks)
                    n = [ctas(p, M, K) for p in picks]
                    assert all(a <= b for a, b in zip(n, n[1:], strict=False)), (
                        K,
                        eb,
                        sw,
                        M,
                    )
                    tile = [p.mode == "tile" for p in picks]
                    assert all(
                        not (not a and b) for a, b in zip(tile, tile[1:], strict=False)
                    )


def test_cache_closure_and_name_mirror(monkeypatch):
    """The compiled-kernel cache is keyed on the module's source closure: the kernel, the sweep, the blocks, the spec, the
    schedule, the layout, the engine core and the JIT glue; the API, the precompiler, the references and the tests are
    outside (an edit there keeps the kernels valid). The name mirror equals what the kernel module passes to the cache."""
    s = set(jit.relative_sources(MX_QUANT_MODULE))
    assert {
        "quant/mx_kernel.py",
        "quant/mx_sweep.py",
        "quant/blocks.py",
        "quant/mx_spec.py",
        "quant/mx_schedule.py",
        "quant/layout.py",
        "canon.py",
        "term.py",
        "fp32.py",
        "cute_fp.py",
        "cvt_fp8.py",
        "dtypes.py",
        "jit.py",
        "kernel_names.py",
        "__init__.py",
        "quant/__init__.py",
    } == s, s
    for outside in (
        "quant/mx_api.py",
        "quant/mx_precompile.py",
        "quant/mx_reference.py",
        "quant/reference.py",
        "support.py",
        "testing.py",
        "lean_cases.py",
    ):
        assert outside not in s
    assert len(jit.source_sha256(MX_QUANT_MODULE)) == 64
    from flashinfer.batch_invariant.quant import mx_kernel as mk

    seen = []
    monkeypatch.setattr(
        jit,
        "build_and_load",
        lambda module, kernel, fn: seen.append((module, kernel)) or object(),
    )
    mk._compile_bucket.cache_clear()
    cfg = C.MXQuantConfig(64, 2, 8, "reg")
    tile = C.MXQuantConfig(32, 128, 32, mode="tile", tile_warps=16, tile_bpt=2)
    mk.get_compiled_kernel("f32", 2048, cfg, S.MXFP8, S.DEFAULT_RULE, False, True)
    mk.get_compiled_kernel("bf16", 4096, tile, S.MXFP4, S.OCP_FLOOR_RULE, True, False)
    assert seen == [
        (
            MX_QUANT_MODULE,
            mx_quant_kernel_name(
                "mxfp8",
                S.DEFAULT_RULE.name,
                "f32",
                C.padded_blocks(2048),
                cfg.describe(),
                False,
                True,
            ),
        ),
        (
            MX_QUANT_MODULE,
            mx_quant_kernel_name(
                "mxfp4", S.OCP_FLOOR_RULE.name, "bf16", 64, tile.describe(), True, False
            ),
        ),
    ]
    assert seen[0][1] == "mxfp8_rcprp_f32_Kbp64_T64_R2_V8_RM_lin_pdl1"
    mk._compile_bucket.cache_clear()


def test_backend_rule_and_fingerprint_api():
    """The backend rule: batch_invariant=True only through backend="auto"; "auto" without the flag is the API's default."""
    assert (
        resolve_backend("mxfp8_quantize", "auto", True, default="cuda")
        == "batch-invariant"
    )
    assert resolve_backend("mxfp8_quantize", "auto", False, default="cuda") == "cuda"
    assert (
        resolve_backend("mxfp8_quantize", "cute-dsl", False, default="cuda")
        == "cute-dsl"
    )
    for b in ("cuda", "cute-dsl", "batch-invariant", "trtllm"):
        with pytest.raises(ValueError, match="reachable only through backend='auto'"):
            resolve_backend("mxfp8_quantize", b, True, default="cuda")
    assert set(SUPPORTED_ARCHS) == {(10, 0), (10, 3), (10, 7)}
    sig = inspect.signature(flashinfer.mxfp8_quantize)
    p = sig.parameters["batch_invariant"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False
    assert sig.parameters["backend"].default == "cuda"


def test_lean_generated_cases_are_current():
    """The committed ``lean/RLKernels/Cases.lean`` is byte for byte what the Python spec / engine emit today (no Lean
    toolchain needed): a change to the spec, the engine or a placement must regenerate it, and ``lake build`` (the Lean
    tier) then re-checks the Lean model against the new terms."""
    with open(lean_cases.CASES_PATH) as f:
        committed = f.read()
    assert committed == lean_cases.render(), (
        "regenerate with: python -m flashinfer.batch_invariant.lean_cases"
    )
    assert (
        len(lean_cases.mxquant_cases()) == 3
        and len(lean_cases.mxquant_out_cases()) == 12
    )
    assert "sorry" not in committed


# =============================================================================== GPU tiers
def _gpu_ok():
    return torch.cuda.is_available() and is_supported_device(
        torch.device("cuda", torch.cuda.current_device())
    )


needs_gpu = pytest.mark.skipif(
    not _gpu_ok(), reason="needs an SM100 / SM103 / SM107 device with the CuTe DSL"
)


@pytest.fixture(scope="module")
def precompiled():
    """Compile every specialization the quick GPU tier touches, in parallel, device-free, before the first test."""
    if not _gpu_ok():
        pytest.skip("needs an SM100 / SM103 / SM107 device with the CuTe DSL")
    if os.environ.get("FLASHINFER_BI_NO_PRECOMPILE") == "1":
        return
    from flashinfer.batch_invariant.quant.mx_precompile import (
        dispatch_specs,
        precompile_mx_quant,
    )

    sm = torch.cuda.get_device_properties(0).multi_processor_count
    specs = dispatch_specs(
        [FMT],
        ("bf16", "f16", "f32"),
        KS_GPU + [128],
        L.LAYOUTS,
        sm_counts=tuple(sorted(set(SM_COUNTS + [sm]))),
    )
    specs.append(
        ("bf16", 1024, C.default_config(37, 1024, 2, sm, True), FMT, True, True)
    )  # PDL
    t0 = time.time()
    failures = precompile_mx_quant(specs)
    print(
        f"[batch_invariant] {len(specs)} specializations ready in {time.time() - t0:.0f} s",
        flush=True,
    )
    assert not failures, failures[:5]


def _api(x, layout, **kw):
    """``flashinfer.mxfp8_quantize`` through the public API with the flag; the outputs as uint8 on the CPU."""
    q, sf = flashinfer.mxfp8_quantize(
        x,
        sf_swizzle_layout=FI_LAYOUTS[layout],
        backend="auto",
        batch_invariant=True,
        **kw,
    )
    assert q.dtype == torch.float8_e4m3fn and sf.dtype == torch.uint8
    return q.view(torch.uint8).cpu(), sf.cpu()


def _fi(x, layout, backend="cute-dsl"):
    q, sf = flashinfer.mxfp8_quantize(
        x, sf_swizzle_layout=FI_LAYOUTS[layout], backend=backend, enable_pdl=False
    )
    return q.view(torch.uint8).cpu(), sf.view(torch.uint8).cpu().reshape(-1)


def _no_tiny_blocks(x):
    """Zero every block whose amax is below ``448 x 2^-127`` (where FlashInfer's zero inverse differs from TE's)."""
    M, K = x.shape
    xb = x.float().view(M, K // 32, 32)
    tiny = (xb.abs().amax(-1) < 448.0 * 2.0**-120).unsqueeze(-1)
    return torch.where(tiny, torch.zeros_like(xb), xb).view(M, K).to(x.dtype)


@needs_gpu
@pytest.mark.parametrize("layout", L.LAYOUTS)
@pytest.mark.parametrize("K", KS_GPU)
@pytest.mark.parametrize("dtype", DT)
def test_gpu_equals_exact_cpu_model(precompiled, dtype, K, layout):
    """The API's bits == the exact CPU model (codes, scale bytes incl. padding) on adversarial rows and the edge families, at
    odd and tile-crossing row counts; a few rows also against the Fractions evaluation of the symbolic spec."""
    for M in MS_GPU:
        x = _edge_rows(M, K, dtype, seed=M + K, device="cuda")
        codes, sf = _api(x, layout)
        rc, rsf, _ = R.quantize_mx_reference(x, FMT, S.DEFAULT_RULE, layout)
        assert codes.shape == (M, K) and sf.numel() == L.sf_numel(M, K, 32, layout)
        assert torch.equal(codes, rc), (
            f"codes differ from the spec (M={M}, K={K}, {dtype}, {layout})"
        )
        assert torch.equal(sf, rsf), (
            f"scales differ from the spec (M={M}, K={K}, {dtype}, {layout})"
        )
    rows = [0, 1, 4, 36] if K <= 4096 else [0, 4]
    x = _edge_rows(37, K, dtype, seed=7, device="cuda")
    codes, sf = _api(x, layout)
    ec, esf = R.exact_rows(x, FMT, rows)
    assert torch.equal(codes[rows], ec) and torch.equal(
        R.logical_sf(sf, 37, K, layout)[rows], esf
    )


@needs_gpu
def test_flashinfer_output_contract(precompiled):
    """FlashInfer's contract exactly: float8_e4m3fn codes of the input's shape, flat uint8 scales of the layout's size with
    zero padding, ``alignment`` padding K with zero columns (same shapes and bytes as the cute-dsl backend), >2-D inputs,
    row-strided views and under-aligned storage offsets, ``enable_pdl`` None / True / False, ``is_sf_swizzled_layout``."""
    x = _edge_rows(37, 96, torch.bfloat16, device="cuda")
    q, sf = flashinfer.mxfp8_quantize(x, backend="auto", batch_invariant=True)
    assert (
        q.dtype == torch.float8_e4m3fn
        and q.shape == (37, 96)
        and sf.shape == (128 * 4,)
        and sf.dtype == torch.uint8
    )
    logical = L.unswizzle_sf_128x4(sf.cpu(), 128, 96, 32)
    assert int(logical[37:].sum()) == 0 and all(
        int(sf[L.sf_offset_128x4(r, 3, 4)]) == 0 for r in range(128)
    )
    q8, sf8 = flashinfer.mxfp8_quantize(
        x, sf_swizzle_layout=SfLayout.layout_8x4, backend="auto", batch_invariant=True
    )
    assert sf8.shape == (40 * 4,) and torch.equal(
        q8.view(torch.uint8), q.view(torch.uint8)
    )
    ql, sfl = flashinfer.mxfp8_quantize(
        x, is_sf_swizzled_layout=False, backend="auto", batch_invariant=True
    )
    assert sfl.shape == (37 * 3,) and torch.equal(
        ql.view(torch.uint8), q.view(torch.uint8)
    )
    # alignment: K = 96 padded to 128 with zero columns, as FlashInfer's backends do (bitwise the same bytes)
    for layout in L.LAYOUTS:
        qa, sfa = flashinfer.mxfp8_quantize(
            x,
            alignment=128,
            sf_swizzle_layout=FI_LAYOUTS[layout],
            backend="auto",
            batch_invariant=True,
        )
        qf, sff = flashinfer.mxfp8_quantize(
            x,
            alignment=128,
            sf_swizzle_layout=FI_LAYOUTS[layout],
            backend="cute-dsl",
            enable_pdl=False,
        )
        assert qa.shape == qf.shape == (37, 128) and sfa.shape == sff.shape
        assert torch.equal(qa.view(torch.uint8), qf.view(torch.uint8)) and torch.equal(
            sfa, sff
        ), layout
        assert int(qa.view(torch.uint8)[:, 96:].sum()) == 0
    # >2-D input: rows are the flattened leading dimensions
    x3 = _edge_rows(36, 96, torch.bfloat16, seed=3, device="cuda").view(3, 12, 96)
    q3, s3 = flashinfer.mxfp8_quantize(x3, backend="auto", batch_invariant=True)
    q2, s2 = flashinfer.mxfp8_quantize(
        x3.view(36, 96), backend="auto", batch_invariant=True
    )
    assert (
        q3.shape == (3, 12, 96)
        and torch.equal(q3.view(36, 96).view(torch.uint8), q2.view(torch.uint8))
        and torch.equal(s3, s2)
    )
    # a row-strided view (no copy) and an under-aligned storage offset (an aligned copy): the same bits as the contiguous input
    wide = _edge_rows(37, 192, torch.bfloat16, seed=5, device="cuda")
    view = wide[:, :96]
    assert not view.is_contiguous()
    assert _same(_api(view, "128x4"), _api(view.contiguous(), "128x4"))
    pool = torch.empty(4 + 37 * 96, dtype=torch.bfloat16, device="cuda")
    pool[4:].copy_(x.reshape(-1))
    off = pool[4:].view(37, 96)
    assert off.data_ptr() % 16 != 0
    assert _same(_api(off, "128x4"), _api(x, "128x4"))
    # PDL: the same bits with PDL on, off and auto
    ref = _api(x, "128x4")
    assert _same(_api(x, "128x4", enable_pdl=True), ref) and _same(
        _api(x, "128x4", enable_pdl=False), ref
    )


@needs_gpu
def test_determinism_and_graph_replay(precompiled):
    x = _edge_rows(37, 1024, torch.bfloat16, device="cuda")
    ref = _api(x, "128x4")
    for _ in range(5):
        assert _same(_api(x, "128x4"), ref)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        outs = flashinfer.mxfp8_quantize(x, backend="auto", batch_invariant=True)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            outs = flashinfer.mxfp8_quantize(x, backend="auto", batch_invariant=True)
    for _ in range(2):
        outs[0].view(torch.uint8).zero_()
        outs[1].zero_()
        g.replay()
        torch.cuda.synchronize()
        assert _same((outs[0].view(torch.uint8).cpu(), outs[1].cpu()), ref)


@needs_gpu
def test_batch_permutation_subbatch_invariance(precompiled):
    """R2 and R4 within the class: a row's bits in a batch of 256, alone, embedded in other rows, permuted and in sub-batches
    across the dispatcher's padding buckets; the three layouts carry the same bytes."""
    K = 1024
    xb = _edge_rows(256, K, torch.bfloat16, seed=1, device="cuda")
    full_c, full_s = _api(xb, "128x4")
    sf_full = L.unswizzle_sf_128x4(full_s, 256, K, 32)

    def logical(outs, M):
        return outs[0], L.unswizzle_sf_128x4(outs[1], M, K, 32)

    for i in (0, 17, 130, 255):
        one = logical(_api(xb[i : i + 1].clone(), "128x4"), 1)
        assert torch.equal(one[0][0], full_c[i]) and torch.equal(one[1][0], sf_full[i])
    xo = adversarial_rows(13, K, torch.bfloat16, seed=5).cuda()
    xo[4] = xb[17]
    o = logical(_api(xo, "128x4"), 13)
    assert torch.equal(o[0][4], full_c[17]) and torch.equal(o[1][4], sf_full[17])
    perm = torch.randperm(256, generator=torch.Generator().manual_seed(2)).cuda()
    p = logical(_api(xb[perm].contiguous(), "128x4"), 256)
    assert torch.equal(p[0], full_c[perm.cpu()]) and torch.equal(
        p[1], sf_full[perm.cpu()]
    )
    for k in (1, 3, 16, 100, 129):
        idx = torch.randperm(256, generator=torch.Generator().manual_seed(k))[:k].cuda()
        s = logical(_api(xb[idx].contiguous(), "128x4"), k)
        assert torch.equal(s[0], full_c[idx.cpu()]) and torch.equal(
            s[1], sf_full[idx.cpu()]
        )
    lin_c, lin_s = _api(xb, "linear")
    assert torch.equal(lin_c, full_c) and torch.equal(lin_s.view(256, K // 32), sf_full)
    c8, s8 = _api(xb, "8x4")
    assert torch.equal(c8, full_c) and torch.equal(
        L.unswizzle_sf(s8, 256, K, 32, "8x4"), sf_full
    )
    c8o, s8o = _api(xb[:37].clone(), "8x4")
    assert torch.equal(L.unswizzle_sf(s8o, 37, K, 32, "8x4"), sf_full[:37]) and int(
        s8o.sum()
    ) == int(sf_full[:37].sum())
    # fp32 == the exact widening of the 16-bit input: one class per format
    for dtype in (torch.bfloat16, torch.float16):
        x = _edge_rows(130, 2048, dtype, seed=3, device="cuda")
        for layout in L.LAYOUTS:
            assert _same(_api(x.float(), layout), _api(x, layout)), (dtype, layout)


@needs_gpu
def test_config_agreement_over_the_dispatcher_picks(precompiled):
    """R3 over the picks the dispatcher makes for the four SKUs at every quick-tier shape: the same bits whatever
    configuration was picked (the whole enumerated space is the nightly tier)."""
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    n = 0
    for K in KS_GPU:
        for layout in L.LAYOUTS:
            for dtype in (torch.bfloat16, torch.float32):
                x = _edge_rows(
                    2048 if K <= 1024 else 300, K, dtype, seed=K, device="cuda"
                )
                ref = _api(x, layout)
                picks = {
                    C.default_config(x.shape[0], K, EB[dtype], s, layout)
                    for s in SM_COUNTS + [sm]
                }
                for cfg in picks:
                    codes, sf = mx_quantize(x, FMT, layout, config=cfg)
                    assert _same((codes.cpu(), sf.cpu()), ref), (
                        cfg.describe(),
                        K,
                        dtype,
                        layout,
                    )
                    n += 1
                for s in SM_COUNTS:
                    codes, sf = mxfp8_quantize_batch_invariant(
                        x, FI_LAYOUTS[layout], sm_count=s
                    )
                    assert _same((codes.view(torch.uint8).cpu(), sf.cpu()), ref), (
                        s,
                        K,
                        dtype,
                        layout,
                    )
    assert n >= 30


@needs_gpu
@pytest.mark.parametrize("layout", L.LAYOUTS)
def test_fi_cute_dsl_is_bitwise_identical(precompiled, layout):
    """FlashInfer's own ``cute-dsl`` backend computes the class on inputs without tiny blocks (its inverse scale is 0 at
    UE8M0 byte 0 where the class keeps the exact 2^127: the one documented difference, pinned in the next test): codes and
    scale bytes equal in every layout, bf16 / f16 / fp32."""
    for dtype in DT:
        for M, K in [(37, 256), (130, 1024), (3, 4096), (8, 96), (200, 7168)]:
            x = _no_tiny_blocks(adversarial_rows(M, K, dtype, seed=M).cuda())
            ours = _api(x, layout)
            theirs = _fi(x, layout, "cute-dsl")
            assert ours[1].numel() == theirs[1].numel()
            assert torch.equal(ours[0], theirs[0]), (
                f"codes differ from FlashInfer cute-dsl (M={M}, K={K}, {dtype}, {layout})"
            )
            assert torch.equal(ours[1], theirs[1]), (
                f"scales differ from FlashInfer cute-dsl (M={M}, K={K}, {dtype}, {layout})"
            )
            if dtype != torch.float32:  # the cuda backend refuses fp32
                theirs = _fi(x, layout, "cuda")
                assert torch.equal(ours[0], theirs[0]) and torch.equal(
                    ours[1], theirs[1]
                ), f"vs FlashInfer cuda (M={M}, K={K}, {dtype}, {layout})"


@needs_gpu
def test_tiny_block_corner_vs_flashinfer_is_pinned(precompiled):
    """The documented difference: for a block whose amax is below ``448 x 2^-127`` (UE8M0 byte 0) FlashInfer's inverse scale
    is 0 (every code 0), the class's is the exact 2^127 (the values survive). The scale bytes agree; every differing code
    sits in such a block; nothing else differs. And a -inf element: -448 in the class, +448 in FlashInfer."""
    M, K = 64, 1024
    x = adversarial_rows(M, K, torch.bfloat16, seed=11).cuda()
    xb = x.float().view(M, K // 32, 32)
    tiny = torch.zeros(M, K // 32, dtype=torch.bool, device="cuda")
    tiny[::7, ::5] = True
    xb = torch.where(
        tiny.unsqueeze(-1), (xb.sign() + (xb == 0).float()) * 2.0**-126, xb
    )
    x = xb.view(M, K).to(torch.bfloat16)
    ours, theirs = _api(x, "128x4"), _fi(x, "128x4", "cute-dsl")
    assert torch.equal(ours[1], theirs[1]), "scale bytes differ"
    assert bool((L.unswizzle_sf_128x4(ours[1], M, K, 32)[tiny.cpu()] == 0).all())
    ov, tv = (
        R.unpack_elems(ours[0], FMT).view(M, K // 32, 32),
        R.unpack_elems(theirs[0], FMT).view(M, K // 32, 32),
    )
    diff = (ov != tv).any(-1)
    assert bool((tv[tiny.cpu()] == 0).all()) and bool((ov[tiny.cpu()] != 0).any())
    assert bool((diff <= tiny.cpu()).all()), "a code differs outside a tiny block"
    y = torch.zeros(8, 64, dtype=torch.bfloat16, device="cuda")
    y[0, 0], y[0, 1] = -float("inf"), 3.0
    oy, ty = _api(y, "linear"), _fi(y, "linear", "cute-dsl")
    assert torch.equal(oy[1], ty[1]) and int(oy[1][0]) == 254
    assert (
        R.unpack_elems(oy[0], FMT)[0, 0] == -448.0
        and R.unpack_elems(ty[0], FMT)[0, 0] == 448.0
    )


@needs_gpu
def test_mm_mxfp8_consumes_the_output(precompiled):
    """FlashInfer's ``mm_mxfp8`` on the class's (codes, scales) == the same GEMM on FlashInfer's own quantization of the
    same inputs (the quantizers agree bit for bit on inputs without tiny blocks)."""
    m, n, k = 300, 1024, 2048
    a = _no_tiny_blocks(adversarial_rows(m, k, torch.bfloat16, seed=1).cuda())
    b = _no_tiny_blocks(adversarial_rows(n, k, torch.bfloat16, seed=2).cuda())
    aq, asf = flashinfer.mxfp8_quantize(a, backend="auto", batch_invariant=True)
    bq, bsf = flashinfer.mxfp8_quantize(b, backend="auto", batch_invariant=True)
    fa, fasf = flashinfer.mxfp8_quantize(a, backend="cute-dsl", enable_pdl=False)
    fb, fbsf = flashinfer.mxfp8_quantize(b, backend="cute-dsl", enable_pdl=False)
    assert torch.equal(aq.view(torch.uint8), fa.view(torch.uint8)) and torch.equal(
        asf, fasf
    )
    assert torch.equal(bq.view(torch.uint8), fb.view(torch.uint8)) and torch.equal(
        bsf, fbsf
    )
    try:
        ours = flashinfer.mm_mxfp8(aq, bq.t(), asf, bsf, out_dtype=torch.bfloat16)
        theirs = flashinfer.mm_mxfp8(fa, fb.t(), fasf, fbsf, out_dtype=torch.bfloat16)
    except Exception as e:  # noqa: BLE001 -- no MXFP8 GEMM backend for this device / environment
        pytest.skip(f"mm_mxfp8 unavailable here: {type(e).__name__}: {str(e)[:120]}")
    assert torch.equal(ours, theirs)


@needs_gpu
def test_backend_rule_through_the_api(precompiled):
    """batch_invariant=True is reachable only through backend="auto"; "auto" without the flag is the "cuda" default."""
    x = _edge_rows(37, 1024, torch.bfloat16, device="cuda")
    for backend in ("cuda", "cute-dsl"):
        with pytest.raises(ValueError, match="reachable only through backend='auto'"):
            flashinfer.mxfp8_quantize(x, backend=backend, batch_invariant=True)
    auto = flashinfer.mxfp8_quantize(x, backend="auto")
    cuda = flashinfer.mxfp8_quantize(x, backend="cuda")
    assert (
        auto[0].shape == cuda[0].shape
        and torch.equal(auto[0].view(torch.uint8), cuda[0].view(torch.uint8))
        and torch.equal(auto[1], cuda[1])
    )
    with pytest.raises(AssertionError, match="backend must be"):
        flashinfer.mxfp8_quantize(x, backend="rlk")


@needs_gpu
def test_unsupported_raises_never_falls_back(precompiled, monkeypatch):
    """Every input outside the class raises BatchInvariantUnsupportedError naming the class and the limit: dtype, alignment,
    a CPU tensor, an unsupported architecture (the capability probe is monkeypatched), a missing CuTe DSL."""
    x = _edge_rows(37, 1024, torch.bfloat16, device="cuda")
    with pytest.raises(BatchInvariantUnsupportedError, match="mxfp8.quant.*dtype"):
        flashinfer.mxfp8_quantize(
            x.to(torch.float8_e4m3fn), backend="auto", batch_invariant=True
        )
    with pytest.raises(BatchInvariantUnsupportedError, match="alignment"):
        flashinfer.mxfp8_quantize(x, alignment=48, backend="auto", batch_invariant=True)
    with pytest.raises(BatchInvariantUnsupportedError, match="CUDA device"):
        flashinfer.mxfp8_quantize(x.cpu(), backend="auto", batch_invariant=True)
    import flashinfer.utils as fu

    monkeypatch.setattr(fu, "get_compute_capability", lambda device: (9, 0))
    with pytest.raises(
        BatchInvariantUnsupportedError,
        match="compute capability 9.0.*SM100 / SM103 / SM107",
    ):
        flashinfer.mxfp8_quantize(x, backend="auto", batch_invariant=True)
    monkeypatch.undo()
    import flashinfer.cute_dsl.availability as av

    monkeypatch.setattr(av, "is_cute_dsl_available", lambda: False)
    with pytest.raises(BatchInvariantUnsupportedError, match="CuTe DSL"):
        flashinfer.mxfp8_quantize(x, backend="auto", batch_invariant=True)
    monkeypatch.undo()
    assert isinstance(BatchInvariantUnsupportedError("x"), ValueError)


@needs_gpu
def test_ue8m0_device_conversion_matches_model_sample(precompiled):
    """The three UE8M0 device conversions == the CPU model on a 2^24-pattern sample (every pattern is the nightly tier)."""
    from flashinfer.batch_invariant.quant.mx_sweep import get_ue8m0_sweep

    bad = _ue8m0_sweep(
        get_ue8m0_sweep(), [0, 0x3F000000, 0x7F000000, 0x80000000, 0xFF000000], 1 << 22
    )
    assert bad == [0, 0, 0], bad


def _ue8m0_sweep(k, starts, n):
    bad = [0, 0, 0]
    for start in starts:
        bits = torch.arange(start, start + n, dtype=torch.int64, device="cuda")
        x = bits.to(torch.int32).view(torch.float32).contiguous()
        b = torch.empty(3, n, dtype=torch.uint8, device="cuda")
        v = torch.empty(3, n, dtype=torch.float32, device="cuda")
        k(x, b, v, n)
        xc, bc, vc = (
            x.cpu(),
            b.cpu(),
            v.cpu(),
        )  # the vectorised model is a CPU model (CUDA frexp differs on UE8M0 subnormals)
        for r, rnd in enumerate(fp32.UE8M0_ROUNDINGS):
            rv = R.ue8m0_round(xc, rnd)
            rb = R.ue8m0_bytes(rv)
            nan = torch.isnan(xc)
            bad[r] += int((bc[r] != rb).sum()) + int(
                ((vc[r].view(torch.int32) != rv.view(torch.int32)) & ~nan).sum()
            )
    return bad


# =============================================================================== nightly GPU
@needs_gpu
@pytest.mark.nightly
@pytest.mark.parametrize("K", KS_SPACE)
def test_nightly_config_agreement_whole_space(K):
    """R3 over the WHOLE enumerated space (bf16, 128x4) at K: every configuration gives the bits of the dispatcher's pick."""
    from flashinfer.batch_invariant.quant.mx_precompile import (
        precompile_mx_quant,
        space_specs,
    )

    assert not precompile_mx_quant(space_specs([FMT], ("bf16",), [K], ("128x4",)))
    x = _edge_rows(37, K, torch.bfloat16, seed=K, device="cuda")
    ref = _api(x, "128x4")
    n = 0
    for cfg in C.all_valid_configs(K, 2, "128x4"):
        codes, sf = mx_quantize(x, FMT, "128x4", config=cfg)
        assert _same((codes.cpu(), sf.cpu()), ref), (
            f"config {cfg.describe()} differs (K={K})"
        )
        n += 1
    assert n >= 8


@needs_gpu
@pytest.mark.nightly
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("K", KS_SPACE_OTHER)
def test_nightly_config_agreement_other_dtypes_and_layouts(K, dtype):
    from flashinfer.batch_invariant.quant.mx_precompile import (
        precompile_mx_quant,
        space_specs,
    )

    assert not precompile_mx_quant(
        space_specs([FMT], (DTYPE_STR[dtype],), [K], L.LAYOUTS)
    )
    x = _edge_rows(37, K, dtype, seed=K, device="cuda")
    for layout in L.LAYOUTS:
        ref = _api(x, layout)
        for cfg in C.all_valid_configs(K, EB[dtype], layout):
            codes, sf = mx_quantize(x, FMT, layout, config=cfg)
            assert _same((codes.cpu(), sf.cpu()), ref), (
                cfg.describe(),
                K,
                dtype,
                layout,
            )


@needs_gpu
@pytest.mark.nightly
def test_nightly_ue8m0_device_conversions_exhaustive():
    """``fp32.to_ue8m0`` (through the vectorised model, itself checked against the exact one on the CPU) == the device
    conversions for the three roundings over all 2^32 binary32 bit patterns."""
    from flashinfer.batch_invariant.quant.mx_sweep import get_ue8m0_sweep

    n = 1 << 24
    t0 = time.time()
    bad = _ue8m0_sweep(get_ue8m0_sweep(), range(0, 1 << 32, n), n)
    torch.cuda.synchronize()
    print(
        f"ue8m0 sweep: 2^32 inputs in {time.time() - t0:.1f} s; mismatches rp / rz / rn = {bad}"
    )
    assert bad == [0, 0, 0]
