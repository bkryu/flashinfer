"""The only floating-point primitives a batch-invariant CuTe-DSL kernel may use.

Each one lowers to an NVVM intrinsic with an explicit ``.rn`` rounding mode
(``add.rn.f32`` etc.). Unlike ``a + b`` on ``Float32`` (which lowers to ``arith``
ops that the NVPTX backend is free to contract into FMAs), these are opaque to the
compiler: no contraction, no reassociation, no flush-to-zero.

The class mirrors the interface of :class:`flashinfer.batch_invariant.fp.term.TermAlgebra` (symbolic) and the
functions in :mod:`flashinfer.batch_invariant.fp.exact` (exact CPU), so schedule code can be written once
against an ``fp`` object and run under all three backends.
"""

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Float32
from cutlass._mlir.dialects import llvm
from cutlass.cute.math import RoundingMode
from cutlass.cutlass_dsl import T, dsl_user_op
import cutlass
from cutlass import Float32, Uint16

RN = RoundingMode.NEAREST_EVEN


class CuteF32:
    @staticmethod
    def const(value: float):
        from . import exact

        return Float32(exact.to_f32(value))

    @staticmethod
    def add(a, b):
        return cute.math.add(a, b, rounding=RN)

    @staticmethod
    def mul(a, b):
        return cute.math.mul(a, b, rounding=RN)

    @staticmethod
    def fma(a, b, c):
        return cute.math.fma(a, b, c, rounding=RN)

    @staticmethod
    def div(a, b):
        return cute.math.div(a, b, rounding=RN)


    @staticmethod
    def fmax(a, b):
        """``nvvm.fmax`` (PTX ``max.f32``, FMNMX): exact, no rounding; maximumNumber NaN rule; on
        sm_107 ``max(+0, -0) = max(-0, +0) = +0`` (bitwise commutative on non-NaN inputs)."""
        return cute.arch.fmax(a, b)

    @staticmethod
    def fmaxc(a, b):
        """``max.f32`` as a commutative join: the same instruction as ``fmax`` (the symbolic algebra normalises the
        operand order; the CPU model uses the measured zero rule)."""
        return cute.arch.fmax(a, b)



    # -- exact (rounding-free) primitives ---------------------------------------

    @staticmethod
    def fmin(a, b):
        """``min.f32`` (no ftz, no NaN propagation flag): exact."""
        return cute.arch.fmin(a, b)

    @staticmethod
    def cvt_out(a, fmt_name: str):
        """``cvt.rn.<fmt>.f32`` (``arith.truncf``, RNE); the identity for ``f32``."""

        return a if fmt_name == "f32" else a.to(get_cutlass_dtype(fmt_name))

    @staticmethod
    def fabs(a):
        """``abs.f32`` (``nvvm.fabs`` without ftz): clears the sign bit, exact."""
        from cutlass._mlir.dialects import nvvm

        return Float32(nvvm.fabs(Float32(a).ir_value(), ftz=False))

    # ``fmaxc`` is the commutative max above (same ``max.f32``); the quantizer's amax trees use it on
    # ``fabs(x) >= +0`` operands.


# --- declared conversion primitives and the dtype table shared by the kernels ---------------------------------------

# The declared fp8 conversion primitive shared by the kernels that quantize to e4m3 / e5m2: ``cvt.rn.satfinite.{e4m3,e5m2}x2.f32``
# (RNE, saturating to the largest finite value; the exact CPU model is :func:`flashinfer.batch_invariant.fp.exact.to_fp8`).
@dsl_user_op
def cvt_f32x2_to_fp8x2(
    a: Float32, b: Float32, kind: cutlass.Constexpr[str], *, loc=None, ip=None
) -> Uint16:
    """``cvt.rn.satfinite.{e4m3,e5m2}x2.f32``: a -> low byte, b -> high byte."""
    res = llvm.inline_asm(
        T.i16(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        f"cvt.rn.satfinite.{kind}x2.f32 $0, $2, $1;",
        "=h,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Uint16(res)


# Name <-> CuTe dtype table shared by the kernels (kept out of the API modules so that compiled-kernel cache keys do not depend
# on them).
def get_cutlass_dtype(name: str):
    import cutlass

    return {
        "bf16": cutlass.BFloat16,
        "f16": cutlass.Float16,
        "f32": cutlass.Float32,
        "e4m3": cutlass.Float8E4M3FN,
        "e5m2": cutlass.Float8E5M2,
    }[name]
