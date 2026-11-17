# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Probe variants of the softmax term (``AttnConfig.lazy`` / ``AttnConfig.exp16``): class CHANGES, not placements, kept behind the
probe opt-in (``engine.CLASS_PROBE_ENV``) so the enumerated configuration space and the published class are untouched. Both bodies
(the row body and the Swaps-AB body) take their tile step from here, so every body applies the same fp32 operations in the same
order: the variants keep R1-R4 only because the per-row decision and the per-key exponential are functions of the row's own values.

* lazy rescale (``lazy = L > 0``, in log2 units): the running maximum moves to the tile maximum only when the alpha argument
  ``add(mul(m, c), mul(m', nc))`` -- the spec's own term -- is below ``-L``; otherwise the row keeps ``m``, ``alpha = 1`` (the
  correction skips its O pass) and the exponentials use the stale maximum, so ``P`` reaches at most ``2^L``.
* fp16 exponentials (``exp16``): ``p = widen(ex2_f16(cvt_f16(x)))`` for ``x = fma(s, c, mul(m', nc))``, two keys per ``ex2.approx.f16x2``;
  the widening is exact, the row sum tree and the bf16 ``P`` word are formed from the widened values exactly as in the fp32 class.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ....batch_invariant.fp_interface.cute import CuteF32

_NEG_INF = float("-inf")


@cute.jit
def tile_max_step(m, m_t, c: Float32, nc: Float32, lazy: cutlass.Constexpr):
    """The per-row statistics of one tile from the running maximum ``m`` and the tile maximum ``m_t`` (``-inf`` = no visible key):
    returns ``(m_new, alpha, nms)`` with ``nms = mul(m_used, nc)`` the exponentials' offset. ``lazy = 0`` is the exact class
    (``spec.alpha_term``); ``lazy = L`` keeps ``m`` while the alpha argument is at least ``-L``. Trace-time ``lazy``."""
    neg_inf = Float32(_NEG_INF)
    null = m_t == neg_inf
    m_cand = CuteF32.fmaxc(m, m_t)
    m_base = m_cand
    if null:
        m_base = Float32(0.0)
    x = CuteF32.add(CuteF32.mul(m, c), CuteF32.mul(m_base, nc))  # the alpha argument
    alpha = CuteF32.ex2(x)
    if null:
        alpha = Float32(1.0)
    m_new = m_cand
    if cutlass.const_expr(lazy > 0):
        keep = x >= Float32(-float(lazy))  # -inf on a chunk's first tile: never kept
        if keep:
            m_new = m
            m_base = m
            alpha = Float32(1.0)
    nms = CuteF32.mul(m_base, nc)
    return m_new, alpha, nms


@dsl_user_op
def exp16_pair(x0: Float32, x1: Float32, *, loc=None, ip=None):
    """``(p0, p1) = widen(ex2.approx.f16x2(cvt.rn.f16x2.f32(x0, x1)))``: two exponentials in one MUFU instruction, widened back to
    binary32 exactly (``add.f32x2.f16x2.f32x2`` with ``+0``)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Float32(x0).ir_value(loc=loc, ip=ip), Float32(x1).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 h, e, z;
            .reg .b64 zz, w;
            cvt.rn.f16x2.f32 h, $3, $2;
            ex2.approx.f16x2 e, h;
            mov.b32 z, 0;
            mov.b64 zz, {z, z};
            add.f32x2.f16x2.f32x2 w, e, zz;
            mov.b64 {$0, $1}, w;
        }
        """,
        "=f,=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def exp16_pair_word(x0: Float32, x1: Float32, *, loc=None, ip=None):
    """``exp16_pair`` plus the bf16 pair ``cvt.rn.bf16x2.f32(p1, p0)`` as one 32-bit word (``p0`` in the low half: the P store's
    element order). Returns ``(p0, p1, word)`` with the word as the bits of a ``Float32`` register (the P fragment's element type)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32()]),
        [Float32(x0).ir_value(loc=loc, ip=ip), Float32(x1).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 h, e, z, pw;
            .reg .b64 zz, w;
            cvt.rn.f16x2.f32 h, $4, $3;
            ex2.approx.f16x2 e, h;
            mov.b32 z, 0;
            mov.b64 zz, {z, z};
            add.f32x2.f16x2.f32x2 w, e, zz;
            mov.b64 {$0, $1}, w;
            cvt.rn.bf16x2.f32 pw, $1, $0;
            mov.b32 $2, pw;
        }
        """,
        "=f,=f,=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [2], loc=loc, ip=ip)),
    )
