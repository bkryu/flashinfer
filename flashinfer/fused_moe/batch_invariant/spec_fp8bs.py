# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The block-scale fp8 MoE classes ``moe.e4m3bs.<act>`` (DeepSeek-style fp8: e4m3 experts with one fp32 scale per 128x128
block, e4m3 activations with one fp32 scale per 1x128 block) and the quantizer class ``fp8bs.quant`` they are built on.

The routing, the activations (:func:`.spec.glu_elem` / :func:`.spec.act_elem`: every typed activation of the layer) and
the combine are the bf16 classes' (:mod:`.spec`). The GEMM folds are the dense block-scale class's
(:mod:`flashinfer.gemm.batch_invariant.spec`: ``gemm.e4m3.kblock128``, the per-tensor fp8 ``mma32`` step folded per
128-element K block and promoted in block order as ``fma.rn(mul.rn(s_a, s_b), P_kb, acc)``) at ``K = H`` and ``K = I``; the
grouped GEMM is a placement of that class. The block scales are the fold's own operands, so GEMM1 is the bf16 class's chain
on the promoted accumulators as they stand (no alpha, no per-token scale, no static scale) and GEMM2 the plain ``cvt_bf16``.

The quantizer class ``fp8bs.quant`` (the activations per token, the intermediate per expanded row, the expert weights per
128x128 block at preparation) is the canon engine's block digit and three chains, declared once here and evaluated by every
placement (the symbolic terms, the exact and the vectorised CPU references, the standalone kernel, the fused GEMM1 epilogue):

* ``amax = fmaxc-tree(fabs(x_e))``, ``e`` in 0..127: one ``pairwise`` digit of radix 128 (:func:`block_reduction`), placed
  ``[(V, reg), (128 / V, lane)]`` by whoever computes it (:func:`block_amax_placement`);
* ``s = fmaxc(div(amax, 448), fp32(1e-12))`` (:func:`scale_term`): the fp32 dequantization scale the GEMM class reads (the
  floor gives a zero block a finite scale: FlashInfer's recipe ``(amax / 448).clamp(1e-12)``);
* ``inv = div(1, s)`` (:func:`inv_term`): the correctly rounded reciprocal, formed once per block;
* ``q = cvt.rn.satfinite.e4m3(mul(x, inv))`` (:func:`quant_elem`): one multiply and the declared saturating conversion per
  element (FlashInfer's torch recipe divides by ``s``; the class multiplies by the reciprocal: one term in every placement).

Every S-expression this module builds is the reference model's (the Lean cases compare them textually:
``lean/BIKernels/FP8BlockQuant.lean`` for the quantizer, ``MoEFp8Bs.lean`` for the class).
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional, Sequence

from ...batch_invariant.canon import Digit, Placement, Reduction
from ...batch_invariant.fp_interface import exact as fp32
from ...batch_invariant.quantization.qterm import Term as QTerm
from ...batch_invariant.quantization.qterm import TermAlgebra as QTermAlgebra
from ...batch_invariant.quantization.qterm import sexpr as qsexpr
from ...gemm.batch_invariant.spec import spec_fingerprint as gemm_fingerprint
from .mterm import Term, TermAlgebra, fingerprint
from .spec import ACTS, SOFTMAX_RENORM, UNGATED_ACTS, MoESpec, RoutingMethod

CLASS = "moe.e4m3bs"
IN_FMT = "e4m3_bs"
#: Elements per fp32 scale: the class constant (FlashInfer's ``scale_granularity_mnk = (1, 128, 128)``).
BLOCK = 128
#: The largest finite e4m3 magnitude: the scale maps the block amax onto it.
E4M3_MAX = 448.0
#: The floor of a block's scale (a zero block quantizes to zeros under a finite scale).
SCALE_EPS = fp32.to_f32(1e-12)
AMAX_IDENTITY_NOTE = (
    "fmaxc(v, +0) == v for every v >= +0; every leaf is fabs(x) >= +0 and never -0"
)
PLACEMENT_VS = (128, 64, 32, 16, 8, 4)


# ----------------------------------------------------------------------------- the reduction
def leaf_value(fp: Any, x: Any) -> Any:
    """The reduction leaf: ``fabs(x)`` of the (exactly widened) input element."""
    return fp.fabs(x)


def elem_digit() -> Digit:
    return Digit(
        "elem",
        BLOCK,
        "pairwise",
        combine=lambda fp, a, b: fp.fmaxc(a, b),
        identity_note=AMAX_IDENTITY_NOTE,
    )


def block_reduction(leaf_x: Callable) -> Reduction:
    """``amax = fmaxc-tree over fabs(x_e)``, ``e`` in 0..127; ``leaf_x(fp, e)`` is element ``e`` of the block as an f32 term /
    value (the caller fixes the row and the block)."""
    return Reduction(
        "fp8bs.block_amax",
        (elem_digit(),),
        lambda fp, idx: leaf_value(fp, leaf_x(fp, idx["elem"])),
    )


def block_amax_placement(V: int) -> Placement:
    """The placement of the ``elem`` digit a thread holding ``V`` consecutive elements of a block uses: ``[(V, reg),
    (128 / V, lane)]`` (the ``128 / V`` lanes of a block are adjacent lanes; the block amax is finished by ``log2(128 / V)``
    xor shuffles). ``V`` in ``PLACEMENT_VS``: a whole block per thread (the fused epilogue's half-block form is the register
    tree over 64 leaves, whose combine with the other half's is the block's tree) down to four elements per lane."""
    if V not in PLACEMENT_VS:
        raise ValueError(f"fp8bs.quant: no placement with {V} elements per lane")
    return Placement.make({"elem": [(V, "reg"), (BLOCK // V, "lane")]})


# ----------------------------------------------------------------------------- the chains
def scale_term(fp: Any, amax: Any) -> Any:
    """``s = fmaxc(div(amax, 448), 1e-12)``: the block's fp32 dequantization scale (the ``max.f32`` the kernels issue is
    commutative; the canonical S-expression sorts its operands)."""
    return fp.fmaxc(fp.div(amax, fp.const(E4M3_MAX)), fp.const(SCALE_EPS))


def inv_term(fp: Any, s: Any) -> Any:
    """``inv = div(1, s)``: the correctly rounded reciprocal of the block scale, formed once per block."""
    return fp.div(fp.const(1.0), s)


def quant_pre(fp: Any, x: Any, inv: Any) -> Any:
    """``mul(x, inv)``: the value the e4m3 conversion rounds (the fp32 part of :func:`quant_elem`)."""
    return fp.mul(x, inv)


def quant_elem(fp: Any, x: Any, inv: Any) -> Any:
    """``q = cvt.rn.satfinite.e4m3(mul(x, inv))`` as the element value (the caller packs the code)."""
    return fp.cvt_out(quant_pre(fp, x, inv), "e4m3")


def block_chain(fp: Any, amax: Any):
    """``(s, inv)`` of a block from its amax."""
    s = scale_term(fp, amax)
    return s, inv_term(fp, s)


# ----------------------------------------------------------------------------- the quantizer class
class BSQuantSpec:
    """Symbolic (term) view of one row of the class ``fp8bs.quant``: ``x(row, i)`` is the input element widened to f32 (exact
    for bf16 / f16, the identity for fp32: one class per input dtype family, not per dtype). A 128x128 weight block is the
    same chain on a block whose 16384 leaves are read row-major (the max of non-negative values is exact in every order,
    so the 2-D block's tree is the row-major pairwise tree by declaration)."""

    def __init__(self, K: int, alg: Optional[QTermAlgebra] = None):
        if K % BLOCK:
            raise ValueError(
                f"fp8bs.quant: K = {K} is not a whole number of {BLOCK}-element blocks"
            )
        self.K = K
        self.nb = K // BLOCK
        self.alg = alg or QTermAlgebra()

    def leaf(self, row: int, i: int) -> QTerm:
        return self.alg.x(row, i)

    def block_reduction(self, row: int, b: int) -> Reduction:
        return block_reduction(lambda fp, e: self.leaf(row, BLOCK * b + e))

    def block_amax(self, row: int, b: int) -> QTerm:
        return self.block_reduction(row, b).canonical(self.alg)

    def block(self, row: int, b: int) -> Dict[str, Any]:
        a = self.alg
        amax = self.block_amax(row, b)
        s, inv = block_chain(a, amax)
        codes = [
            quant_elem(a, self.leaf(row, BLOCK * b + e), inv) for e in range(BLOCK)
        ]
        return {"amax": amax, "s": s, "inv": inv, "codes": codes}

    def outputs(self, row: int) -> Dict[str, List[QTerm]]:
        """``{"codes": [K element-value terms], "s": [nb fp32 scale terms]}``."""
        codes: List[QTerm] = []
        ss: List[QTerm] = []
        for b in range(self.nb):
            blk = self.block(row, b)
            codes += blk["codes"]
            ss.append(blk["s"])
        return {"codes": codes, "s": ss}


def quant_fingerprint() -> str:
    """Identifier of the numerical specification of the class ``fp8bs.quant``: the structural hash of block 0's scale term and
    its first and last code terms. Independent of K, of the input dtype and of the scale-factor layout (a block's bits depend on
    its 128 inputs alone; the layout -- the ``K`` or the ``MN`` arrangement of the fp32 scales -- is a placement of the store)."""
    s = BSQuantSpec(BLOCK)
    blk = s.block(0, 0)
    parts = [
        "fp8bs.quant",
        str(BLOCK),
        qsexpr(blk["s"]),
        qsexpr(blk["codes"][0]),
        qsexpr(blk["codes"][BLOCK - 1]),
    ]
    return "fp8bsq-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- the MoE class
class Fp8BsMoESpec(MoESpec):
    """Symbolic view of the class ``moe.e4m3bs.<act>[H, I, k, E, method]`` for one token: the routing and the combine of
    :class:`.spec.MoESpec`, GEMM1 the bf16 class's chain on the promoted accumulators (the block scales are inside the fold:
    no scale symbol enters the chain), GEMM2 the plain conversion."""

    def __init__(
        self,
        H: int,
        I: int,
        k: int,
        E: int,
        method: RoutingMethod = SOFTMAX_RENORM,
        act: str = "silu",
        params: Sequence[Optional[float]] = (),
        gated: Optional[bool] = None,
        alg: Optional[TermAlgebra] = None,
    ):
        super().__init__(H, I, k, E, method, "bf16", alg, act, params, gated)

    def gemm2(self, r: int = 0, h: int = 0) -> Term:
        """GEMM2 of the block-scale class: ``cvt_bf16(acc2)`` of the promoted accumulator."""
        return self.alg.cvt_out(self.alg.leaf("acc2", r, h), "bf16")

    @property
    def name(self) -> str:
        return "moe.e4m3bs." + (
            "swiglu"
            if self.gated and self.act == "silu" and not self.params
            else self.act
        )


def spec_fingerprint_fp8bs(
    H: int,
    I: int,
    k: int,
    E: int,
    method: RoutingMethod = SOFTMAX_RENORM,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
    gated: Optional[bool] = None,
) -> str:
    """Identifier of the class: the routing weights of slot 0 and slot k-1, the GEMM1 chain (its activation, gating and
    binary32 parameters), the block-scale GEMM classes at ``K = H`` and ``K = I``, the ``fp8bs.quant`` class on the activations,
    the intermediate and the expert weights (one class, three placements), GEMM2's chain and the combine. The scales' values,
    like the operands, are inputs."""
    s = Fp8BsMoESpec(H, I, k, E, method, act, params, gated)
    q = quant_fingerprint()
    parts = [
        CLASS,
        f"H={H}",
        f"I={I}",
        f"k={k}",
        f"E={E}",
        method.name,
        f"act={'glu' if s.gated else 'act'}:{act}",
        f"params={s.params!r}",
        fingerprint(s.weight(0)),
        fingerprint(s.weight(k - 1)),
        fingerprint(s.gemm1()),
        gemm_fingerprint(H, "f32", 1, IN_FMT),
        gemm_fingerprint(I, "f32", 1, IN_FMT),
        q,
        q,
        q,
        fingerprint(s.gemm2()),
        fingerprint(s.combine()),
    ]
    return "moe-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


__all__ = [
    "ACTS",
    "AMAX_IDENTITY_NOTE",
    "BLOCK",
    "BSQuantSpec",
    "CLASS",
    "E4M3_MAX",
    "Fp8BsMoESpec",
    "IN_FMT",
    "PLACEMENT_VS",
    "SCALE_EPS",
    "UNGATED_ACTS",
    "block_amax_placement",
    "block_chain",
    "block_reduction",
    "elem_digit",
    "inv_term",
    "leaf_value",
    "quant_elem",
    "quant_fingerprint",
    "quant_pre",
    "scale_term",
    "spec_fingerprint_fp8bs",
]
