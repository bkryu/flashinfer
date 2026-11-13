# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The support class ``support.row.v1``: ``sampling_support`` and its internal renormalisation entries (the operators
``top_k_renorm_probs`` / ``top_p_renorm_probs`` / ``top_k_mask_logits`` carry no ``batch_invariant`` flag in this
release; the class is the selection stage for the truncated samplers).

Device-free: the pinned fingerprints; the exact model's declared properties (top-k an upward-closed cut with ties, top-p
reaching its fixed-point target minimally, the two orders, min-p on the row maximum, the exact fixed point, the
renormalisation term); dispatcher membership over the grid and four SM counts; the error contract; the PTX pins of
three kernels (only the declared forms). On a device of the class's architectures (SM100 / SM103 / SM107): the mask, the
effective threshold, the kept count, the kept mass and the row sum equal the exact model bit for bit for probability
and logits inputs, every filter set and both orders, across the configuration space (single-CTA, cluster, staged) and
the dispatcher's picks; the special parameter values and ties; batch invariance (a row alone, a permutation, a
sub-batch); the renormalised probabilities and masked logits bit for bit and the internal entries equal to the class's
entry; SM-count independence where the picks differ; determinism, graph replay and PDL; views and empty batches."""

import os

import pytest
import torch

from flashinfer import batch_invariant as bi
from flashinfer.batch_invariant.sampling import classes as CL
from flashinfer.batch_invariant.sampling import schedule as C
from flashinfer.batch_invariant.sampling import sterm as term
from flashinfer.batch_invariant.sampling import support_reference as SR
from flashinfer.batch_invariant.sampling import support_schedule as DS
from flashinfer.batch_invariant.sampling import support_spec as SS
from flashinfer.batch_invariant.sampling.schedule import device_sm_count
from flashinfer.batch_invariant.sampling.support_api import (
    SUPPORT_ARCHS,
    sampling_support,
    top_k_mask_logits_batch_invariant,
    top_k_renorm_probs_batch_invariant,
    top_p_renorm_probs_batch_invariant,
)
from flashinfer.batch_invariant.sampling.support_spec import Filters

from .batch_invariant_sampling_common import (
    DS as _DS,
    PDL as _PDL,
    SM_COUNTS,
    bad_ptx_ops,
    compile_device_free,
    device_marker,
    device_supported,
    device_tests_selected,
    logits,
    precompile,
)

DT = [torch.float32, torch.bfloat16]
#: One block, a partial block (an odd vocabulary), 16 blocks, an odd long one, a long one.
NS = [64, 1000, 4096, 4097, 32768]
N_BIG = 152064
SETS = {
    "k": Filters(top_k=1),
    "p": Filters(top_p=1.0),
    "m": Filters(min_p=0.0),
    "kp": Filters(top_k=1, top_p=1.0),
    "kpj": Filters(top_k=1, top_p=1.0, order="joint"),
    "kpm": Filters(top_k=1, top_p=1.0, min_p=0.0),
    "kpmj": Filters(top_k=1, top_p=1.0, min_p=0.0, order="joint"),
}
EF_NS = (1000, 32768)  # the placement test's vocabularies
EF_M = 24
EF_SETS = ("k", "p", "kp", "kpj", "kpm")
SETS_BY_N = {
    64: ("k", "p", "kpm"),
    1000: ("kp", "kpj", "m"),
    4096: ("kpmj", "k"),
    4097: ("kp", "p"),
    32768: ("kpm", "kpj"),
}
#: ``(N, M)`` at which the SM counts split the dispatcher's pick (512 threads below the device's row count, 256 at it).
SM_SPLIT_ROWS = ((152064, 180),)
_supported = device_supported(SUPPORT_ARCHS)
needs_device = device_marker(SUPPORT_ARCHS)


# ------------------------------------------------------------------------------------------------ helpers
def _pow2_divisor(N, cap):
    v = cap
    while v > 1 and N % v:
        v //= 2
    return v


def _configs(N, eb, k=4):
    """Members with a share of at most 256 elements per thread whose staged tile fits the budget (the long single-CTA
    members are never dispatcher picks and compile for many minutes): a single-CTA, a cluster, a staged and a wide
    member, and a re-read one."""
    cfgs = [
        c
        for c in C.all_valid_configs(N, eb)
        if c.elems_per_thread(N) <= 256
        and c.rows_per_cta == 1
        and c.blocks_per_cta(N) * C.BLOCK * eb <= C.SMEM_BUDGET
    ]
    picks = (
        cfgs[:1]
        + [c for c in cfgs if c.cluster > 1][:1]
        + [c for c in cfgs if c.stage == "smem"][:1]
        + [c for c in cfgs if c.threads_per_row >= 512 and c.cluster == 1][:1]
        + [c for c in cfgs if c.stage == "reread"][:1]
    )
    return list(dict.fromkeys(picks))[:k]


def weights(M, N, dtype, seed=0):
    """Weights with zeros, ties at the top, a near-uniform row, an all-zero row and a one-hot row."""
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(M, N, generator=g) ** 3
    w[:, ::7] = 0.0
    w[0, 1] = w[0, 2] = w[0].max() * 1.5
    if M > 3:
        w[3] = 1.0 / N
        w[3, 5] = 2.0 / N
    if M > 5:
        w[5] = 0.0
    if M > 6:
        w[6] = 0.0
        w[6, N - 1] = 0.25
    return w.to(dtype).cuda()


def tied_logits(M, N, dtype, seed=0):
    """The adversarial logits rows with a three-way tie at the top of row 1."""
    x = logits(M, N, dtype, seed=seed).float()
    x[1, :3] = x[1].max() + 1.0
    return x.to(dtype).cuda()


def params(fset, M, N, seed=0):
    """Per-row parameters for the filter set, spanning the special values (``k`` of 0 / N / beyond, ``p`` of 0 / 1 /
    0.999, ``min_p`` of 0 / 1); returns the keyword arguments and the per-row ``Filters``."""
    g = torch.Generator().manual_seed(seed)
    ks = torch.randint(1, max(2, N // 2), (M,), generator=g)
    ks[0] = 1
    if M > 1:
        ks[1] = 0
    if M > 2:
        ks[2] = N
    if M > 3:
        ks[3] = N + 7
    ps = torch.rand(M, generator=g)
    if M > 1:
        ps[1] = 0.0
    if M > 2:
        ps[2] = 1.0
    if M > 4:
        ps[4] = 0.999
    ms = torch.rand(M, generator=g) * 0.5
    if M > 2:
        ms[2] = 0.0
    if M > 3:
        ms[3] = 1.0
    f = SETS[fset]
    kw = {}
    if f.top_k is not None:
        kw["top_k"] = ks.to(torch.int32).cuda()
    if f.top_p is not None:
        kw["top_p"] = ps.cuda()
    if f.min_p is not None:
        kw["min_p"] = ms.cuda()
    kw["order"] = f.order
    filters = [
        Filters(
            top_k=int(ks[i]) if f.top_k is not None else None,
            top_p=float(ps[i]) if f.top_p is not None else None,
            min_p=float(ms[i]) if f.min_p is not None else None,
            order=f.order,
        )
        for i in range(M)
    ]
    return kw, filters


def _eq_nan(a, b):
    """Bitwise equality with NaN == NaN (a kept set of zero mass renormalises to NaN: the division by +0)."""
    return torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(
        torch.nan_to_num(a.float(), 0.0).view(torch.int32),
        torch.nan_to_num(b.float(), 0.0).view(torch.int32),
    )


def _same(r, ref, from_logits):
    assert torch.equal(r.mask.cpu(), ref["mask"]), "mask"
    assert torch.equal(
        r.threshold.cpu().view(torch.int32), ref["threshold"].view(torch.int32)
    ), "threshold"
    assert torch.equal(r.count.cpu(), ref["count"]), "count"
    assert torch.equal(r.mass.cpu().view(torch.int32), ref["mass"].view(torch.int32)), (
        "mass"
    )
    assert torch.equal(r.sum.cpu().view(torch.int32), ref["S"].view(torch.int32)), "sum"
    if from_logits:
        a, b = r.logprobs.cpu(), ref["logprobs"]
        assert torch.equal(torch.isnan(a), torch.isnan(b)), "logprobs (kept set)"
        assert torch.equal(
            torch.nan_to_num(a, 0.0).view(torch.int32),
            torch.nan_to_num(b, 0.0).view(torch.int32),
        ), "logprobs"


def _row(N=1000, seed=0):
    """A CPU weight row with two tied maxima, a third value and zeros every 97th element (the model's property tests)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(N, generator=g).to(torch.float32) * 0.01
    w[10] = 0.3
    w[20] = 0.3
    w[30] = 0.2
    w[::97] = 0.0
    return w


# ------------------------------------------------------------------------------------------------ device-free
def test_fingerprints_are_pinned():
    """The receipts: the class's fingerprints of the three renormalisation forms at N = 1000 in bf16 (the filter set,
    the selection rules, the vocabulary's sum tree, the weight form and the output format are hashed); the dimensions
    move them. The class has no registry entry of its own (the operators' flag is deferred): the truncated samplers'
    entries name it."""
    f = SS.spec_fingerprint
    assert f(Filters(top_k=1), 1000, False, "bf16") == "support-row-4a26c16577a204b6"
    assert f(Filters(top_p=1.0), 1000, False, "bf16") == "support-row-475956ca58f276bb"
    assert f(Filters(top_k=1), 1000, True, "bf16") == "support-row-5b80a40dba511dd7"
    assert CL._support_fingerprint(1000, torch.bfloat16, Filters(top_k=1), False) == f(
        Filters(top_k=1), 1000, False, "bf16"
    )
    for op in ("top_k_renorm_probs", "top_p_renorm_probs", "top_k_mask_logits"):
        assert op not in bi.CLASSES
    assert f(Filters(top_k=1), 1000) == f(
        Filters(top_k=7), 1000
    )  # the parameter values are inputs, not dimensions
    assert f(Filters(top_k=1), 1000) != f(Filters(top_p=0.5), 1000)
    assert f(Filters(top_k=1, top_p=0.5), 1000) != f(
        Filters(top_k=1, top_p=0.5, order="joint"), 1000
    )
    assert f(Filters(top_k=1), 1000) != f(Filters(top_k=1), 1001)
    assert f(Filters(top_k=1), 1000, False) != f(Filters(top_k=1), 1000, True)
    assert f(Filters(top_k=1), 1000, False, "f32") != f(
        Filters(top_k=1), 1000, False, "bf16"
    )
    assert "ties-kept" in SS.SELECT_RULE and "trunc(p*2^44)" in SS.SELECT_RULE
    with pytest.raises(ValueError):
        CL._support_fingerprint(1000, "f64", Filters(top_k=1), False)


@pytest.mark.parametrize("k", [0, 1, 2, 3, 17, 500, 999, 1000, 5000])
def test_top_k_is_an_upward_closed_cut_with_ties(k):
    w = _row()
    r = SR.support_row(w, w.numel(), Filters(top_k=k))
    thr = r["threshold"]
    assert bool(((w >= thr) == r["mask"]).all())
    if k <= 0:
        assert r["count"] == 0
    elif k >= w.numel():
        assert r["count"] == w.numel()
    else:
        # at least k kept, and dropping the threshold value would leave fewer than k
        assert r["count"] >= k and int((w > thr).sum()) < k
    assert (
        SR.support_row(w, w.numel(), Filters(top_k=1))["count"] == 2
    )  # the tied maxima


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 0.999, 1.0])
def test_top_p_reaches_the_target_mass_minimally(p):
    w = _row()
    N = w.numel()
    r = SR.support_row(w, N, Filters(top_p=p))
    thr = r["threshold"]
    assert bool(((w >= thr) == r["mask"]).all())
    scale = SS.fixed_scale(r["S"])
    q = torch.tensor(
        [SS.fixed_weight(float(x), scale) for x in w.tolist()], dtype=torch.int64
    )
    target = SS.fixed_target(p)
    kept_mass = int(q[r["mask"]].sum())
    assert kept_mass >= target or r["count"] == int(
        (w > 0).sum()
    )  # reached, or everything positive is kept
    if p >= 1.0:
        assert r["count"] == int((w > 0).sum())  # the filter off: every positive weight
    if r["count"] and thr < float("inf") and target > 0:
        assert int(q[w > thr].sum()) < target  # the next cut up would miss the target
    assert not bool(r["mask"][w == 0].any())  # zero weights are never kept


def test_orders_differ_as_declared():
    w = _row()
    N = w.numel()
    a = SR.support_row(w, N, Filters(top_k=50, top_p=0.5))
    b = SR.support_row(w, N, Filters(top_k=50, top_p=0.5, order="joint"))
    assert (
        a["count"] == 3 and b["count"] == 50
    )  # top-p on the kept 50 vs the intersection
    assert bool(
        (a["mask"] & ~b["mask"]).sum() == 0
    )  # top_k_first's set is within the joint set here
    for r in (
        a,
        b,
        SR.support_row(w, N, Filters(top_k=50, top_p=0.5, min_p=0.1, order="joint")),
    ):
        assert bool(
            ((w >= r["threshold"]) == r["mask"]).all()
        )  # the effective threshold cuts the mask


def test_min_p_uses_the_row_maximum():
    from flashinfer.batch_invariant.fp_interface import exact as fp32

    w = _row()
    r = SR.support_row(w, w.numel(), Filters(min_p=0.5))
    assert r["threshold"] == fp32.mul(0.3, fp32.to_f32(0.5)) and r["count"] == 3
    r = SR.support_row(
        w, w.numel(), Filters(min_p=0.5), w_max=1.0
    )  # logits: e_max = 1 by the shift
    assert r["threshold"] == 0.5 and r["count"] == 0


def test_fixed_point_mass_is_exact_and_declared():
    from flashinfer.batch_invariant.fp_interface import exact as fp32

    S = 5.7959
    scale = SS.fixed_scale(S)
    assert scale == fp32.div(float(1 << SS.FIX_BITS), S)
    assert (
        SS.fixed_weight(0.3, scale) == int(fp32.mul(0.3, scale))
        and SS.fixed_weight(0.0, scale) == 0
    )
    assert SS.fixed_target(1.0) == SS.UNREACHABLE
    assert SS.fixed_target(0.5) == 1 << (SS.FIX_BITS - 1) and SS.fixed_target(0.0) == 0


def test_model_rows_and_renormalisation():
    """The model over rows with per-row filters; the renormalised probabilities are ``lse.row``'s prob chain over the
    kept mass (``cvt(mul(w, div(1, mass)))``, +0 elsewhere, in the input dtype); the term is the spec's."""
    from flashinfer.batch_invariant.fp_interface import exact as fp32

    w = torch.stack([_row(300, 1), _row(300, 2)])
    r = SR.support_reference(w, [Filters(top_k=5), Filters(top_p=0.3)])
    assert r["mask"].shape == (2, 300) and r["count"].tolist()[0] >= 5
    wb = w.to(torch.bfloat16)
    r = SR.support_reference(wb, [Filters(top_k=5), Filters(top_p=0.3)])
    assert r["probs"].dtype == torch.bfloat16
    for i in range(2):
        inv = fp32.div(1.0, float(r["mass"][i]))
        want = torch.where(
            r["mask"][i],
            torch.mul(wb[i].float(), torch.tensor(inv, dtype=torch.float32)),
            torch.zeros(300),
        ).to(torch.bfloat16)
        assert torch.equal(r["probs"][i], want)
        assert (
            abs(float(r["probs"][i].float().sum()) - 1.0) < 2e-2
        )  # bf16 outputs: renormalised within rounding
    sx = term.sexpr(SS.renorm_term(40, [True] * 40, 3))
    assert sx.startswith("(mul (div (const 1.0) (add") and sx.endswith(
        "(x 3))"
    )  # mul(x_3, div(1, mass tree))
    assert term.sexpr(SS.mass_term(40, [True] * 40)) == term.sexpr(
        __import__(
            "flashinfer.batch_invariant.sampling.draw_spec", fromlist=["DrawSpec"]
        )
        .DrawSpec(40)
        .total(0)
    )  # keeping everything is the draw's total (Lean: massTree_all)


def test_dispatcher_picks_are_members_of_the_space():
    """R3 membership: every pick of the support's closed-form dispatcher, over the grid, both element widths and the
    SM counts of the SM100 / SM103 / SM107 parts, is a member of the enumerated space at one row per CTA whose staged
    share fits the budget; the picks split by the SM count only through the device's row count."""
    for N in (8, 64, 1000, 4097, 32000, 32768, 128256, 152064, 65537, 262145):
        for eb in (2, 4):
            members = set(C.all_valid_configs(N, eb))
            for M in (1, 4, 37, 256, 4096):
                for sm in SM_COUNTS:
                    cfg = DS.support_config(N, M, eb, sm)
                    assert cfg in members and cfg.rows_per_cta == 1, (
                        N,
                        eb,
                        M,
                        sm,
                        cfg.describe(),
                    )
                    assert cfg.blocks_per_cta(N) * C.BLOCK * eb <= C.SMEM_BUDGET, (
                        N,
                        eb,
                        M,
                        sm,
                        cfg.describe(),
                    )
                    assert cfg.vec_elems <= C.max_vec_elems(N, eb)
    for N, M in SM_SPLIT_ROWS:
        assert len({DS.support_config(N, M, 2, sm) for sm in SM_COUNTS}) >= 2


def test_unsupported_inputs_raise_before_any_kernel():
    x = torch.rand(4, 64, dtype=torch.bfloat16)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="support.row"):
        top_k_renorm_probs_batch_invariant(x, 5)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="support.row"):
        top_p_renorm_probs_batch_invariant(x, 0.5)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="support.row"):
        top_k_mask_logits_batch_invariant(x, 5)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="support.row"):
        sampling_support(x.double(), top_k=5)
    assert (
        SS.Filters(top_k=3).name == "k"
        and SS.Filters(top_p=0.5, min_p=0.1, order="joint").name == "pm_joint"
    )
    with pytest.raises(ValueError):
        Filters(top_k=1, order="first")


IR_CASES = [
    (
        "f32",
        4096,
        (128, 1, 4),
        False,
        "kp",
        False,
        "probs",
    ),  # probabilities, single CTA, top-k then top-p, renormalised
    (
        "bf16",
        152064,
        (128, 1, 8, "regs", 8),
        True,
        "kpm",
        True,
        "",
    ),  # logits, cluster, per-row T, the log-probabilities
    (
        "bf16",
        32768,
        (64, 1, 8, "regs", 4),
        True,
        "k",
        False,
        "masked",
    ),  # logits, cluster, top-k, the masked logits
]


@pytest.mark.parametrize("dtype,N,cfg,from_logits,fset,trow,out_mode", IR_CASES)
def test_support_ir_uses_only_declared_forms(
    tmp_path, dtype, N, cfg, from_logits, fset, trow, out_mode
):
    """Compile one kernel per code path with no GPU visible and check the PTX: `.rn` arithmetic only, `ex2.approx` iff
    logits and `lg2.approx` iff the log-probabilities, the declared `cvt.rzi.s64.f32` iff top-p, the divisions declared
    (`div(2^44, S)` for top-p, `div(1, mass)` for the renormalised probabilities, the per-row temperature's), the
    shared-memory integer adds of the histograms, the cluster forms only in cluster members, no fast-math anywhere."""
    pytest.importorskip("cutlass")
    f = SETS[fset]
    eb = 4 if dtype == "f32" else 2
    code = (
        "from flashinfer.batch_invariant.sampling.schedule import LSEConfig\n"
        "from flashinfer.batch_invariant.sampling.support_kernel import get_compiled_kernel\n"
        "from flashinfer.batch_invariant.sampling.support_spec import Filters\n"
        f"get_compiled_kernel({dtype!r}, {N}, LSEConfig{cfg}, {from_logits}, Filters({f.top_k!r}, {f.top_p!r}, {f.min_p!r}, {f.order!r}), "
        f"{trow}, {not out_mode}, {from_logits and not out_mode}, {_pow2_divisor(N, 8)}, {_pow2_divisor(N, 4)}, False, {out_mode!r}, "
        f"{_pow2_divisor(N, min(8, 16 // eb))})\n"
    )
    _, ptx = compile_device_free(tmp_path, code)
    bad = bad_ptx_ops(ptx)
    assert not bad, f"PTX contains non-admitted floating-point forms: {bad}"
    assert "ftz" not in ptx.replace("ex2.approx.ftz.f32", "").replace(
        "lg2.approx.ftz.f32", ""
    )
    assert ("ex2.approx.ftz.f32" in ptx) == from_logits
    assert ("lg2.approx.ftz.f32" in ptx) == (from_logits and not out_mode)
    assert ("cvt.rzi.s64.f32" in ptx) == (f.top_p is not None)
    divs = (
        (1 if trow else 0)
        + (1 if f.top_p is not None else 0)
        + (1 if out_mode == "probs" else 0)
    )
    assert ptx.count("div.rn.f32") == divs, (ptx.count("div.rn.f32"), divs)
    assert "red.shared.add.u32" in ptx
    Cc = cfg[4] if len(cfg) > 4 else 1
    assert ("barrier.cluster" in ptx) == (Cc > 1) and (
        "ld.shared::cluster.b32" in ptx
    ) == (Cc > 1)


# ------------------------------------------------------------------------------------------------ the precompile
_WORKER_CODE = r"""
import ast, sys
from flashinfer.batch_invariant.sampling.schedule import LSEConfig
from flashinfer.batch_invariant.sampling.support_kernel import get_compiled_kernel
from flashinfer.batch_invariant.sampling.support_spec import Filters
bad = []
for s in ast.literal_eval(sys.argv[1]):
    try:
        ds, N, cfg, from_logits, fset, trow, wmask, wlp, mvec, lvec, pdl, out_mode, ovec, *rest = s
        get_compiled_kernel(ds, N, LSEConfig(*cfg), from_logits, Filters(**dict(fset)), trow, wmask, wlp, mvec, lvec, pdl, out_mode, ovec, *rest)
    except Exception as e:  # noqa: BLE001
        bad.append((s, repr(e)[:300]))
print("BAD", repr(bad))
"""


def _cfg_tuple(c):
    return (c.threads_per_row, c.rows_per_cta, c.vec_elems, c.stage, c.cluster)


def _fset(f: Filters):
    # a tuple of items (the specs are de-duplicated by hashing; the worker rebuilds the Filters)
    return (
        ("top_k", f.top_k),
        ("top_p", f.top_p),
        ("min_p", f.min_p),
        ("order", f.order),
    )


def _spec(ds, N, cfg, from_logits, fset, trow, out_mode="", early_finish="mass"):
    eb = 4 if ds == "f32" else 2
    return (
        ds,
        N,
        _cfg_tuple(cfg),
        from_logits,
        _fset(SETS[fset]),
        trow,
        not out_mode,
        from_logits and not out_mode,
        _pow2_divisor(N, 8),
        _pow2_divisor(N, 4),
        _PDL,
        out_mode,
        _pow2_divisor(N, min(8, 16 // eb)),
        early_finish,
    )


def _ef_configs(N, eb, sm):
    """The placement test's members: the dispatcher's pick at its row count and the first cluster member."""
    picks = [DS.support_config(N, EF_M, eb, sm)]
    cl = [
        c
        for c in _configs(N, eb)
        if c.cluster > 1 and c.describe() != picks[0].describe()
    ]
    return picks + cl[:1]


def _EF_SPECS(sm):
    for ds in ("f32", "bf16"):
        eb = 4 if ds == "f32" else 2
        for N in EF_NS:
            for c in _ef_configs(N, eb, sm):
                for from_logits in (False, True):
                    for fset in EF_SETS:
                        yield ds, N, c, from_logits, fset, ""


def _suite_specs(sm):
    """Every kernel the device tests below may ask for."""
    specs = []
    for dtype in DT:
        ds, eb = _DS[dtype], dtype.itemsize
        for N in NS:
            for fset in SETS_BY_N[N]:
                for c in _configs(N, eb):
                    specs.append(_spec(ds, N, c, False, fset, False))
                    specs.append(_spec(ds, N, c, True, fset, True))
                d12 = DS.support_config(N, 12, eb, sm)
                specs.append(_spec(ds, N, d12, False, fset, False))
                specs.append(_spec(ds, N, d12, True, fset, True))
    for fset in ("kpm", "p", "k", "m"):
        specs.append(
            _spec("f32", 1000, DS.support_config(1000, 10, 4, sm), False, fset, False)
        )
        for M in (64, 1):
            specs.append(
                _spec(
                    "f32", 4096, DS.support_config(4096, M, 4, sm), False, fset, False
                )
            )
    for c in _configs(N_BIG, 2, 2):
        specs.append(_spec("bf16", N_BIG, c, True, "kpm", False))
    specs.append(
        _spec("bf16", N_BIG, DS.support_config(N_BIG, 2, 2, sm), True, "kpm", False)
    )
    for ds in ("f32", "bf16"):
        eb = 4 if ds == "f32" else 2
        for N in (1000, 32768):
            for fset in ("k", "p", "kpm"):
                for c in _configs(N, eb, 2) + [DS.support_config(N, 12, eb, sm)]:
                    specs.append(_spec(ds, N, c, False, fset, False, "probs"))
                d = DS.support_config(N, 12, eb, sm)
                specs.append(_spec(ds, N, d, True, fset, True, "probs"))
                specs.append(_spec(ds, N, d, True, fset, True, "masked"))
                specs.append(_spec(ds, N, d, True, fset, False, "masked"))
                specs.append(_spec(ds, N, d, False, fset, False, "probs"))
    # the SM-count test's picks, the determinism test's pdl-off kernel, the views / empty-batch shapes
    for N, M in SM_SPLIT_ROWS:
        for smc in SM_COUNTS:
            specs.append(
                _spec(
                    "bf16",
                    N,
                    DS.support_config(N, M, 2, smc),
                    False,
                    "kp",
                    False,
                    "probs",
                )
            )
    d = DS.support_config(4096, 5, 2, sm)
    s = list(_spec("bf16", 4096, d, False, "kp", False, "probs"))
    s[10] = False
    specs.append(tuple(s))
    specs.append(
        _spec(
            "bf16", 4096, DS.support_config(4096, 6, 2, sm), False, "k", False, "probs"
        )
    )
    specs.append(
        _spec(
            "bf16", 4096, DS.support_config(4096, 6, 2, sm), True, "k", False, "masked"
        )
    )
    for ds, N, c, from_logits, fset, out_mode in _EF_SPECS(sm):
        for ef in ("all", "mass", ""):
            specs.append(_spec(ds, N, c, from_logits, fset, False, out_mode, ef))
    return list(dict.fromkeys(specs))


@pytest.fixture(scope="session", autouse=True)
def _precompile_support_kernels(request):
    """Precompile the device tests' kernels on device-free workers once per session: a no-op without a supported
    device, under ``BI_SUPPORT_NO_PRECOMPILE=1``, and when no device test is selected."""
    if (
        not _supported
        or os.environ.get("BI_SUPPORT_NO_PRECOMPILE") == "1"
        or not device_tests_selected(request.session, needs_device.kwargs["reason"])
    ):
        yield
        return
    precompile(
        _suite_specs(device_sm_count(torch.device("cuda"))),
        _WORKER_CODE,
        "batch_invariant_sampling_support",
        "[batch_invariant.sampling.support]",
    )
    yield


# ------------------------------------------------------------------------------------------------ device
@needs_device
@pytest.mark.parametrize("dtype", DT, ids=["f32", "bf16"])
@pytest.mark.parametrize("N", NS)
def test_probs_match_exact_model_across_configs(dtype, N):
    M = 12
    w = weights(M, N, dtype)
    for fset in SETS_BY_N[N]:
        kw, filters = params(fset, M, N)
        ref = SR.support_reference(w, filters)
        for cfg in _configs(N, dtype.itemsize) + [None]:
            r = sampling_support(w, config=cfg, enable_pdl=_PDL, **kw)
            try:
                _same(r, ref, False)
            except AssertionError as e:
                raise AssertionError(
                    f"{fset} {cfg.describe() if cfg else 'default'}: {e}"
                ) from None


@needs_device
@pytest.mark.parametrize("dtype", DT, ids=["f32", "bf16"])
@pytest.mark.parametrize("N", NS)
def test_logits_match_exact_model_and_logprobs(dtype, N):
    M = 12
    x = tied_logits(M, N, dtype)
    temps = torch.rand(M, device="cuda") + 0.5
    for fset in SETS_BY_N[N]:
        kw, filters = params(fset, M, N, seed=1)
        ref = SR.support_reference(x, filters, from_logits=True, temperature=temps)
        for cfg in _configs(N, dtype.itemsize) + [None]:
            r = sampling_support(
                x,
                from_logits=True,
                temperature=temps,
                return_logprobs=True,
                config=cfg,
                enable_pdl=_PDL,
                **kw,
            )
            try:
                _same(r, ref, True)
            except AssertionError as e:
                raise AssertionError(
                    f"{fset} {cfg.describe() if cfg else 'default'}: {e}"
                ) from None


@needs_device
def test_special_values_and_ties():
    N, M = 1000, 10
    w = weights(M, N, torch.float32)
    # top-k: nothing / everything / beyond; ties at the cut are kept
    ks = [1, 0, N, N + 7, 2, 3, 1, 500, 999, 1000]
    r = sampling_support(w, top_k=torch.tensor(ks, dtype=torch.int32, device="cuda"))
    assert (
        r.count[0].item() == 2
        and r.count[1].item() == 0
        and r.count[2].item() == N
        and r.count[3].item() == N
    )
    assert r.threshold[1].item() == float("inf") and r.threshold[2].item() == 0.0
    assert (
        r.count[5].item() == N and r.threshold[5].item() == 0.0
    )  # the zero row: its k-th largest weight is 0
    assert r.count[6].item() == 1 and r.threshold[6].item() == 0.25  # the one-hot row
    _same(r, SR.support_reference(w, [Filters(top_k=k) for k in ks]), False)
    # top-p: p = 0 keeps the top group, p = 1 every positive weight; the zero row keeps nothing
    ps = [0.0, 1.0, 0.999, 0.5, 0.0, 1.0, 0.3, 0.9, 0.99, 0.1]
    r = sampling_support(w, top_p=torch.tensor(ps, device="cuda"))
    assert (
        r.count[0].item() == 2
        and r.count[1].item() == int((w[1] > 0).sum())
        and r.count[5].item() == 0
    )
    assert r.threshold[5].item() == float("inf")
    _same(r, SR.support_reference(w, [Filters(top_p=p) for p in ps]), False)
    # min-p: 0 keeps everything (zeros too), 1 the maxima
    ms = [1.0, 0.0, 0.5, 0.1, 1.0, 0.0, 1.0, 0.2, 0.3, 0.4]
    r = sampling_support(w, min_p=torch.tensor(ms, device="cuda"))
    assert (
        r.count[0].item() == 2
        and r.count[1].item() == N
        and r.count[5].item() == N
        and r.count[6].item() == 1
    )
    _same(r, SR.support_reference(w, [Filters(min_p=m) for m in ms]), False)
    # the mask is the effective threshold's cut; scalars broadcast; int64 parameters are accepted
    r = sampling_support(w, top_k=50, top_p=0.9, min_p=0.05)
    assert torch.equal(r.mask, w >= r.threshold[:, None])
    assert bool((r.count == r.mask.sum(1)).all())
    r64 = sampling_support(
        w,
        top_k=torch.full((M,), 50, dtype=torch.int64, device="cuda"),
        top_p=0.9,
        min_p=0.05,
    )
    assert torch.equal(r64.mask, r.mask)
    with pytest.raises(ValueError, match="at least one"):
        sampling_support(w)
    with pytest.raises(ValueError, match="shape"):
        sampling_support(w, top_k=torch.ones(3, dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="finite"):
        sampling_support(w, top_p=float("nan"))
    with pytest.raises(ValueError):
        sampling_support(w, top_k=5, return_masked_logits=True)
    with pytest.raises(ValueError):
        sampling_support(w, top_k=5, return_logprobs=True)


@needs_device
def test_batch_invariance():
    """R2: a row alone, at its slot in a permuted batch and in a sub-batch gives the same mask, mass and threshold."""
    N, M = 4096, 64
    w = weights(M, N, torch.float32, seed=4)
    kw, _ = params("kpm", M, N, seed=2)
    full = sampling_support(w, **kw)
    for i in (0, 17, 63):
        one = sampling_support(
            w[i : i + 1].clone(),
            top_k=kw["top_k"][i : i + 1],
            top_p=kw["top_p"][i : i + 1],
            min_p=kw["min_p"][i : i + 1],
        )
        assert torch.equal(one.mask, full.mask[i : i + 1])
        assert torch.equal(one.mass, full.mass[i : i + 1]) and torch.equal(
            one.threshold, full.threshold[i : i + 1]
        )
    perm = torch.randperm(M, generator=torch.Generator().manual_seed(1)).cuda()
    permuted = sampling_support(
        w[perm].contiguous(),
        top_k=kw["top_k"][perm],
        top_p=kw["top_p"][perm],
        min_p=kw["min_p"][perm],
    )
    assert torch.equal(permuted.mask, full.mask[perm]) and torch.equal(
        permuted.mass, full.mass[perm]
    )
    for k in (1, 3, 16):
        idx = torch.randperm(M, generator=torch.Generator().manual_seed(k))[:k].cuda()
        sub = sampling_support(
            w[idx].contiguous(),
            top_k=kw["top_k"][idx],
            top_p=kw["top_p"][idx],
            min_p=kw["min_p"][idx],
        )
        assert torch.equal(sub.mask, full.mask[idx]) and torch.equal(
            sub.threshold, full.threshold[idx]
        ), k


@needs_device
def test_big_vocabulary():
    x = tied_logits(2, N_BIG, torch.bfloat16)
    kw, filters = params("kpm", 2, N_BIG, seed=3)
    ref = SR.support_reference(x, filters, from_logits=True)
    for cfg in _configs(N_BIG, 2, 2) + [None]:
        r = sampling_support(
            x, from_logits=True, return_logprobs=True, config=cfg, **kw
        )
        _same(r, ref, True)


@needs_device
@pytest.mark.parametrize("dtype", DT, ids=["f32", "bf16"])
@pytest.mark.parametrize("N", [1000, 32768])
def test_renormalised_outputs_and_internal_entries(dtype, N):
    """The renormalised probabilities (``lse.row``'s prob chain over the kept mass, in the input dtype, +0 outside the
    kept set) and the masked logits are the model's bit for bit, for probabilities and logits, across configs; the
    internal renormalisation entries return the class's tensors (with scalar and per-row parameters)."""
    M = 12
    w = weights(M, N, dtype)
    for fset in ("k", "p", "kpm"):
        kw, filters = params(fset, M, N, seed=2)
        ref = SR.support_reference(w, filters)
        for cfg in _configs(N, dtype.itemsize, 2) + [None]:
            r = sampling_support(
                w, return_mask=False, return_probs=True, config=cfg, **kw
            )
            assert r.probs.dtype == dtype and _eq_nan(r.probs.cpu(), ref["probs"]), (
                f"{fset} {cfg.describe() if cfg else 'default'}"
            )
        x = tied_logits(M, N, dtype, seed=4)
        temps = torch.rand(M, device="cuda") + 0.5
        refl = SR.support_reference(x, filters, from_logits=True, temperature=temps)
        r = sampling_support(
            x,
            from_logits=True,
            temperature=temps,
            return_mask=False,
            return_probs=True,
            **kw,
        )
        assert _eq_nan(r.probs.cpu(), refl["probs"]), f"{fset} logits probs"
        r = sampling_support(
            x,
            from_logits=True,
            temperature=temps,
            return_mask=False,
            return_masked_logits=True,
            **kw,
        )
        assert torch.equal(r.masked_logits.cpu(), refl["masked_logits"]), (
            f"{fset} masked logits"
        )
    kw, _ = params("k", M, N, seed=2)
    want = sampling_support(w, top_k=kw["top_k"], return_probs=True).probs
    assert _eq_nan(top_k_renorm_probs_batch_invariant(w, kw["top_k"]), want)
    assert _eq_nan(
        top_k_renorm_probs_batch_invariant(w, 7),
        sampling_support(w, top_k=7, return_probs=True).probs,
    )
    kw, _ = params("p", M, N, seed=2)
    want = sampling_support(w, top_p=kw["top_p"], return_probs=True).probs
    assert _eq_nan(top_p_renorm_probs_batch_invariant(w, kw["top_p"]), want)
    assert _eq_nan(
        top_p_renorm_probs_batch_invariant(w, 0.9),
        sampling_support(w, top_p=0.9, return_probs=True).probs,
    )
    x = tied_logits(M, N, dtype, seed=4)
    kw, _ = params("k", M, N, seed=2)
    want = sampling_support(
        x, top_k=kw["top_k"], from_logits=True, return_masked_logits=True
    ).masked_logits
    assert torch.equal(top_k_mask_logits_batch_invariant(x, kw["top_k"]), want)
    # the renormalised rows sum to one within the output rounding; the masked logits keep the kept values
    pr = top_k_renorm_probs_batch_invariant(w[3:4], 20).float()
    assert abs(float(pr.sum()) - 1.0) < (2e-2 if dtype == torch.bfloat16 else 1e-4)
    ml = top_k_mask_logits_batch_invariant(x, 5)
    keep = torch.isfinite(ml)
    assert torch.equal(ml[keep], x[keep]) and bool((keep.sum(1) >= 5).all())


@needs_device
def test_early_finish_is_a_placement():
    """The early finish -- after two passes, the compaction of the selected bin's candidates and the exact rank selection
    over them in place of the last two passes when they fit the list -- and the four-pass form give the same bits on
    every output: rows whose cuts fall in small bins (random logits: the early finish runs), in large tie groups (eight
    levels; two levels with 100 at the top: the four-pass tail runs) and with half the weights zero; top-k / top-p / both
    orders / with min-p, per-row parameters, probabilities and logits, both dtypes, the dispatcher's pick and a cluster
    member. (The model-equality tests above run the default placement; this one pins the other against it.)"""
    sm = device_sm_count(torch.device("cuda"))
    g = torch.Generator().manual_seed(7)
    M = EF_M
    fields = (
        "threshold",
        "count",
        "mass",
        "sum",
        "mask",
        "logprobs",
        "probs",
        "masked_logits",
    )

    def bits(v):
        if v is None:
            return None
        if v.dtype in (torch.bfloat16, torch.float16):
            return v.view(torch.int16)
        return v.view(torch.int32) if v.dtype == torch.float32 else v

    def same(a, b):
        return all(
            (getattr(a, f) is None) == (getattr(b, f) is None)
            and (
                getattr(a, f) is None
                or torch.equal(bits(getattr(a, f)), bits(getattr(b, f)))
            )
            for f in fields
        )

    n_checked = 0
    for dtype in DT:
        eb = dtype.itemsize
        for N in EF_NS:
            rows = [
                torch.randn(M, N, generator=g) * 2,
                torch.randint(0, 8, (M, N), generator=g).float() * 1.5,
            ]
            two = torch.zeros(M, N)
            two.scatter_(1, torch.rand(M, N, generator=g).argsort(dim=-1)[:, :100], 6.0)
            rows.append(two)
            sparse = torch.randn(M, N, generator=g) * 2
            sparse[torch.rand(M, N, generator=g) < 0.5] = float("-inf")
            rows.append(sparse)
            ks = torch.randint(1, 200, (M,), generator=g).to(torch.int32).cuda()
            ps = torch.rand(M, generator=g).cuda()
            cases = [
                dict(top_k=1),
                dict(top_k=50),
                dict(top_p=0.0),
                dict(top_p=0.9),
                dict(top_k=50, top_p=0.9),
                dict(top_k=50, top_p=0.9, order="joint"),
                dict(top_k=50, top_p=0.9, min_p=0.05),
                dict(top_k=ks, top_p=ps),
            ]
            for lg_cpu in rows:
                lg = lg_cpu.to(dtype).cuda()
                probs = torch.nan_to_num(torch.softmax(lg.float(), dim=-1), nan=0.0).to(
                    dtype
                )
                for cfg in _ef_configs(N, eb, sm):
                    for from_logits in (False, True):
                        x = lg if from_logits else probs
                        for case in cases:
                            kw = dict(
                                case,
                                from_logits=from_logits,
                                return_logprobs=from_logits,
                                config=cfg,
                            )
                            a = sampling_support(x, early_finish="all", **kw)
                            b = sampling_support(x, early_finish="", **kw)
                            c = sampling_support(
                                x, **kw
                            )  # the default policy (the mass selection only)
                            d = sampling_support(
                                x, count_list=False, **kw
                            )  # the count selection's passes over the words
                            e = sampling_support(
                                x, count_list=True, **kw
                            )  # the list mode on every kernel (the rule picks it for some)
                            assert (
                                same(a, b) and same(c, b) and same(d, b) and same(e, b)
                            ), (
                                dtype,
                                N,
                                cfg.describe(),
                                from_logits,
                                {
                                    k: v
                                    for k, v in case.items()
                                    if not torch.is_tensor(v)
                                },
                            )
                            n_checked += 1
    assert n_checked == len(DT) * len(EF_NS) * 4 * 2 * 2 * 8


@needs_device
def test_sm_count_independence():
    """The dispatcher's pick depends on the SM count (at the rows of ``SM_SPLIT_ROWS`` the 148 / 152-SM parts take 256
    threads and the 204 / 224-SM parts 512: asserted to differ) and the bits do not."""
    for N, M in SM_SPLIT_ROWS:
        picks = {DS.support_config(N, M, 2, sm) for sm in SM_COUNTS}
        assert len(picks) >= 2, picks
        w = weights(M, N, torch.bfloat16, seed=3)
        kw, _ = params("kp", M, N, seed=5)
        ref = sampling_support(w, return_mask=False, return_probs=True, **kw)
        for sm in SM_COUNTS:
            r = sampling_support(
                w, return_mask=False, return_probs=True, sm_count=sm, **kw
            )
            assert _eq_nan(r.probs, ref.probs) and torch.equal(
                r.threshold, ref.threshold
            ), sm


@needs_device
def test_determinism_graph_replay_and_pdl():
    """R1: the same call twice, under CUDA-graph replay and with PDL off gives the same bits."""
    N, M = 4096, 5
    w = weights(M, N, torch.bfloat16, seed=6)
    kw, _ = params("kp", M, N, seed=7)
    a = sampling_support(w, return_probs=True, **kw)
    b = sampling_support(w, return_probs=True, **kw)
    assert (
        _eq_nan(a.probs, b.probs)
        and torch.equal(a.mask, b.mask)
        and torch.equal(a.mass, b.mass)
    )
    c = sampling_support(w, return_probs=True, enable_pdl=False, **kw)
    assert _eq_nan(a.probs, c.probs) and torch.equal(a.mask, c.mask)
    out = torch.empty_like(a.probs)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        sampling_support(
            w, return_probs=True, **kw
        )  # warm the kernel on the capture stream
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out.copy_(sampling_support(w, return_probs=True, **kw).probs)
    for _ in range(3):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert _eq_nan(out, a.probs)


@needs_device
def test_shapes_views_and_empty_batches():
    """3-D inputs give outputs of the leading shape; a strided view is accepted (copied first); M = 0 launches
    nothing; the 2-D entries match their row-wise results."""
    N = 4096
    w = weights(6, N, torch.bfloat16, seed=9)
    kw, _ = params("k", 6, N, seed=1)
    flat = sampling_support(w, return_probs=True, **kw)
    r3 = sampling_support(
        w.view(2, 3, N), top_k=kw["top_k"].view(2, 3), return_probs=True
    )
    assert (
        r3.threshold.shape == (2, 3)
        and r3.mask.shape == (2, 3, N)
        and r3.probs.shape == (2, 3, N)
    )
    assert torch.equal(r3.mask.reshape(6, N), flat.mask) and _eq_nan(
        r3.probs.reshape(6, N), flat.probs
    )
    assert torch.equal(r3.count.reshape(6), flat.count)
    wide = torch.zeros(6, 2 * N, dtype=torch.bfloat16, device="cuda")
    wide[:, :N] = w
    rv = sampling_support(wide[:, :N], return_probs=True, **kw)
    assert torch.equal(rv.mask, flat.mask) and _eq_nan(rv.probs, flat.probs)
    x = tied_logits(6, N, torch.bfloat16, seed=2)
    ml = top_k_mask_logits_batch_invariant(x.view(2, 3, N), 5)
    assert ml.shape == (2, 3, N) and torch.equal(
        ml.view(6, N), top_k_mask_logits_batch_invariant(x, 5)
    )
    e = sampling_support(w[:0], top_k=5, return_probs=True)
    assert (
        e.threshold.shape == (0,) and e.mask.shape == (0, N) and e.probs.shape == (0, N)
    )
    assert top_k_renorm_probs_batch_invariant(w[:0], 5).shape == (0, N)
