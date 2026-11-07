# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Canonical dataflow of the NVFP4 MoE classes ``moe.nvfp4.<act>`` (W4A4: NVFP4 experts with one per-tensor global scale per
expert and GEMM, per-token NVFP4 activations), on the bf16 classes' routing, permutation and combine (:mod:`.spec`).

Per token ``t``, slot ``r`` with expert ``e`` and the class's per-token quantizer ``Q`` (``nvfp4.quant.pertoken``, the
TransformerEngine recipe of :mod:`flashinfer.batch_invariant.quantization`):

* ``(x4, xsf, pts1) = Q(x_t)``: the input row quantized per token (the row's global scale derived from its amax; ``pts1``
  the returned per-token scale, a multiplier);
* ``g_raw, u_raw``: the raw fp32 accumulators of the dense NVFP4 class on the expert's GEMM1 weight (``gemm.nvfp4.kfold`` /
  ``kfold128``: the block scales applied inside the instruction, no alpha);
* ``s1 = mul(alpha1[e], pts1)`` (:func:`w4a4_row_scale`): the expert's alpha (``1 / global_scale``, as given, the multiplier
  of ``mm_fp4``) times the token's per-token scale, one ``mul.rn`` per (row, expert);
* ``a = cvt_bf16(glu_pre(mul(g_raw, s1), mul(u_raw, s1)))`` (:func:`glu_scaled_elem`): the gated chain of the bf16 class on
  the scaled accumulators, or ``cvt_bf16(act(mul(acc, s1)))`` (:func:`act_scaled_elem`) for the ungated chains;
* ``(a4, asf, pts2) = Q(a)``: the materialised bf16 intermediate quantized per token by the same class (two passes: the
  per-token rule's row amax spans the whole intermediate width, which no GEMM1 tile sees);
* ``y = cvt_bf16(mul(acc2, mul(alpha2[e], pts2)))`` (:func:`scaled_out_elem`) on GEMM2's raw accumulators;
* the bf16 class's slot-order combine.

The W4A4 class therefore *contains* the per-token quantization class at row lengths ``H`` and ``I`` (its fingerprints are
part of this class's) and the dense NVFP4 GEMM classes of the two expert weight shapes (the instruction class per weight
shape by the dense rule, :func:`flashinfer.gemm.batch_invariant.schedule.fp4_class_for`: a function of the shape and the
architecture family only, never of the batch). Every S-expression here is the Lean model's (``MoENvfp4.lean``).
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Sequence, Tuple

from ...gemm.batch_invariant.schedule import fp4_class_for
from ...gemm.batch_invariant.spec import NVFP4_FMTS
from ...gemm.batch_invariant.spec import spec_fingerprint as gemm_fingerprint
from . import spec as S
from .mterm import Term, TermAlgebra, fingerprint

CLASS = "moe.nvfp4"
#: TransformerEngine's inverse base scale of the per-token NVFP4 recipe (``1 / (e4m3 max * e2m1 max)``): the per-token
#: quantizer's input for the activations and the intermediate.
TE_GS_INV = 1.0 / (448.0 * 6.0)
#: ``448 * 6``: the per-tensor global scale of a weight is this constant over the weight's amax (FlashInfer's convention).
E4M3_MAX, E2M1_MAX = 448.0, 6.0


def w4a4_row_scale(fp: Any, alpha_e: Any, pts: Any) -> Any:
    """``s = mul(alpha_e, pts_row)``: the expert's alpha (the caller's ``1 / global_scale[e]``, as given, like ``mm_fp4``'s)
    times the row's per-token scale; one ``mul.rn`` per (row, expert), then one ``mul.rn`` per accumulator element."""
    return fp.mul(alpha_e, pts)


def glu_scaled_elem(
    fp: Any,
    g: Any,
    u: Any,
    s: Any,
    act: str = "silu",
    out_fmt: str = "bf16",
    params: Sequence[Optional[float]] = (),
) -> Any:
    """GEMM1 of the gated W4A4 classes: ``cvt_bf16(glu_pre(mul(g, s), mul(u, s)))`` from the two raw accumulators (the
    bf16 class's gated chain ``act`` with its parameters on the scaled pair)."""
    return S.glu_elem(fp, fp.mul(g, s), fp.mul(u, s), act, out_fmt, params)


def act_scaled_elem(
    fp: Any,
    acc: Any,
    s: Any,
    act: str = "relu2",
    out_fmt: str = "bf16",
    params: Sequence[Optional[float]] = (),
) -> Any:
    """GEMM1 of the ungated W4A4 classes: ``cvt_bf16(act(mul(acc, s)))`` from the single raw accumulator (the row scale
    first, then the bf16 class's ungated chain)."""
    return S.act_elem(fp, fp.mul(acc, s), act, out_fmt, params)


def scaled_out_elem(fp: Any, acc: Any, s: Any, out_fmt: str = "bf16") -> Any:
    """GEMM2 of the W4A4 classes: ``cvt_bf16(mul(acc, s))`` with ``s`` the row scale of the intermediate row."""
    return fp.cvt_out(fp.mul(acc, s), out_fmt)


def expert_classes(
    H: int, I: int, family: Optional[str], gated: bool = True
) -> Tuple[str, str]:
    """The NVFP4 instruction class of each expert weight by the dense rule: GEMM1 on ``(2I, H)`` (``(I, H)`` ungated), GEMM2
    on ``(H, I)``; a function of the weight shapes and the architecture family only."""
    return fp4_class_for(2 * I if gated else I, H, family), fp4_class_for(H, I, family)


class MoENvfp4Spec(S.MoESpec):
    """Symbolic view of ``moe.nvfp4.<act>[H, I, k, E, method]`` for one token: the routing and the combine of
    :class:`.spec.MoESpec`, the GEMM1 chain on the raw accumulators with the row scale, GEMM2's scaled chain; the two
    per-token quantizations and the GEMM folds are opaque members named by their own fingerprints."""

    def __init__(
        self,
        H: int,
        I: int,
        k: int,
        E: int,
        method: S.RoutingMethod = S.SOFTMAX_RENORM,
        act: str = "silu",
        params: Sequence[Optional[float]] = (),
        gated: Optional[bool] = None,
        fp4_class: Tuple[str, str] = ("nvfp4", "nvfp4"),
        alg: Optional[TermAlgebra] = None,
    ):
        super().__init__(H, I, k, E, method, "bf16", alg, act, params, gated)
        assert all(c in NVFP4_FMTS for c in fp4_class), fp4_class
        self.fp4_class = fp4_class

    def alpha(self, which: int, r: int = 0) -> Term:
        """The expert's alpha of GEMM ``which`` (1 / 2) for slot ``r``: a per-expert leaf."""
        return self.alg.leaf(f"alpha{which}", r)

    def pts(self, which: int, r: int = 0) -> Term:
        """The per-token scale of the GEMM's A row (GEMM1: the token's; GEMM2: the intermediate row of slot ``r``)."""
        return self.alg.leaf("pts2", r) if which == 2 else self.alg.leaf("pts1")

    def row_scale(self, which: int, r: int = 0) -> Term:
        return w4a4_row_scale(self.alg, self.alpha(which, r), self.pts(which, r))

    def gemm1(self, r: int = 0, j: int = 0) -> Term:
        """The GEMM1 epilogue output element (bf16; the per-token class then quantizes the row)."""
        s = self.row_scale(1, r)
        if not self.gated:
            return act_scaled_elem(
                self.alg, self.acc("u", r, j), s, self.act, "bf16", self.params
            )
        return glu_scaled_elem(
            self.alg,
            self.acc("g", r, j),
            self.acc("u", r, j),
            s,
            self.act,
            "bf16",
            self.params,
        )

    def gemm2(self, r: int = 0, h: int = 0) -> Term:
        return scaled_out_elem(
            self.alg, self.alg.leaf("acc2", r, h), self.row_scale(2, r), "bf16"
        )

    @property
    def name(self) -> str:
        return f"{CLASS}.{self.act}"


def spec_fingerprint_nvfp4(
    H: int,
    I: int,
    k: int,
    E: int,
    method: S.RoutingMethod = S.SOFTMAX_RENORM,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
    gated: Optional[bool] = None,
    fp4_class: Tuple[str, str] = ("nvfp4", "nvfp4"),
) -> str:
    """Identifier of the class: the routing weights of slots 0 and k-1, the GEMM1 chain, the dense NVFP4 GEMM classes of
    both expert weights (named per GEMM: the instruction class is a dimension), the per-token quantization class at row
    lengths ``H`` and ``I``, GEMM2's chain and the combine. The activation and its binary32 parameters name the class."""
    # absolute on purpose: a fingerprint-only dependency, outside the kernel cache closures
    from flashinfer.batch_invariant.quantization.nvfp4_spec import PER_TOKEN
    from flashinfer.batch_invariant.quantization.nvfp4_spec import (
        spec_fingerprint as quant_fingerprint,
    )

    s = MoENvfp4Spec(H, I, k, E, method, act, params, gated, fp4_class)
    parts = [
        s.name,
        f"H={H}",
        f"I={I}",
        f"k={k}",
        f"E={E}",
        method.name,
        f"act={'glu' if s.gated else 'act'}:{act}",
        f"params={s.params!r}",
        *fp4_class,
        fingerprint(s.weight(0)),
        fingerprint(s.weight(k - 1)),
        fingerprint(s.gemm1()),
        gemm_fingerprint(H, "f32", 1, fp4_class[0]),
        gemm_fingerprint(I, "f32", 1, fp4_class[1]),
        quant_fingerprint(H, PER_TOKEN),
        quant_fingerprint(I, PER_TOKEN),
        fingerprint(s.gemm2()),
        fingerprint(s.combine()),
    ]
    return "moe-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
