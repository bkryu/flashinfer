"""Exact IEEE-754 binary32 / bfloat16 / binary16 arithmetic on the CPU.

Values are Python floats restricted to the target format (every binary32 and every
bf16 / f16 value is exactly representable as a double). Every op computes the exact
rational result with ``fractions.Fraction`` and rounds once, round-to-nearest-even,
including subnormals, overflow to infinity, and the IEEE sign-of-zero rules. This is
the reference backend used to evaluate a canonical spec bit-exactly.
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction

INF = float("inf")
NAN = float("nan")

# (precision bits incl. hidden bit, min normal exponent, max exponent)
F32 = (24, -126, 127)
BF16 = (8, -126, 127)
F16 = (11, -14, 15)
# fp8 (saturating-finite conversion): (precision, emin, emax, max finite)
E4M3FN = (4, -6, 8, 448.0)
E5M2 = (3, -14, 15, 57344.0)


def _is_neg(x: float) -> bool:
    return math.copysign(1.0, x) < 0


def round_to_format(fr: Fraction, fmt=F32, *, zero_sign_neg: bool = False) -> float:
    """Round an exact rational to the given binary format (RNE)."""
    p, emin, emax = fmt
    if fr == 0:
        return -0.0 if zero_sign_neg else 0.0
    sign = -1.0 if fr < 0 else 1.0
    a = -fr if fr < 0 else fr
    # exponent e with 2^e <= a < 2^(e+1)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    elif Fraction(2) ** (e + 1) <= a:
        e += 1
    e = max(e, emin)
    q = Fraction(2) ** (e - p + 1)
    n = a / q
    r = n.numerator // n.denominator
    rem = n - r
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (r & 1)):
        r += 1
    val = r * q
    if val >= Fraction(2) ** (emax + 1):
        return sign * INF
    return sign * float(val)  # exact: val is representable in binary64


def _finite(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


def add(a: float, b: float, fmt=F32) -> float:
    if not _finite(a, b):
        return a + b
    fr = Fraction(a) + Fraction(b)
    if fr == 0:
        # exact zero: -0 only if both operands are -0 (RNE rules)
        neg = a == 0 and b == 0 and _is_neg(a) and _is_neg(b)
        return round_to_format(fr, fmt, zero_sign_neg=neg)
    return round_to_format(fr, fmt)


def mul(a: float, b: float, fmt=F32) -> float:
    if not _finite(a, b):
        return a * b
    fr = Fraction(a) * Fraction(b)
    return round_to_format(fr, fmt, zero_sign_neg=(_is_neg(a) != _is_neg(b)))


def fma(a: float, b: float, c: float, fmt=F32) -> float:
    if not _finite(a, b, c):
        return a * b + c
    prod = Fraction(a) * Fraction(b)
    fr = prod + Fraction(c)
    if fr == 0:
        prod_neg = _is_neg(a) != _is_neg(b)
        if prod == 0 and c == 0:
            neg = prod_neg and _is_neg(c)
        else:
            neg = False  # exact cancellation -> +0 under RNE
        return round_to_format(fr, fmt, zero_sign_neg=neg)
    return round_to_format(fr, fmt)


def div(a: float, b: float, fmt=F32) -> float:
    if not _finite(a, b) or b == 0:
        if b == 0 and a != 0 and math.isfinite(a):
            return math.copysign(INF, a) * math.copysign(1.0, b)
        if a == 0 and b == 0:
            return NAN
        return a / b
    fr = Fraction(a) / Fraction(b)
    return round_to_format(fr, fmt, zero_sign_neg=(_is_neg(a) != _is_neg(b)))


def fmaxc(a: float, b: float) -> float:
    """PTX ``max.f32`` (NVVM ``fmax``, IEEE maximumNumber) as a commutative join: a NaN operand yields the other operand;
    zeros of opposite sign yield ``+0`` whatever the operand order (so ``fmax`` is bitwise
    commutative on non-NaN inputs and a join of the total order ``-inf < ... < -0 < +0 < ... < inf``);
    measured on SM107."""
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    if a == b:
        if a == 0.0:
            return -0.0 if (_is_neg(a) and _is_neg(b)) else 0.0
        return a
    return a if a > b else b


def to_format(x: float, fmt) -> float:
    """Round a binary32 value to bf16 / f16 (RNE). Identity for F32."""
    if fmt == F32 or not math.isfinite(x):
        return x
    if x == 0:
        return x
    return round_to_format(Fraction(x), fmt)


def to_fp8(x: float, kind: str) -> float:
    """``cvt.rn.satfinite.{e4m3,e5m2}x2.f32`` semantics: RNE, then saturate to the
    largest finite value (no infinities). NaN stays NaN."""
    p, emin, emax, fmax = {"e4m3": E4M3FN, "e5m2": E5M2}[kind]
    if math.isnan(x):
        return x
    if x == 0 or math.isinf(x):
        return math.copysign(0.0, x) if x == 0 else math.copysign(fmax, x)
    v = round_to_format(Fraction(x), (p, emin, emax))
    if math.isinf(v) or abs(v) > fmax:
        v = math.copysign(fmax, v)
    return v


def fmin(a: float, b: float) -> float:
    """``min.f32`` on non-NaN operands. Equal values (incl. ``+0`` / ``-0``) return ``a``; the specs
    only clamp against non-zero constants, so that case never reaches the bits."""
    return a if a <= b else b


# ----------------------------------------------------------------------------- block-scaled quantization primitives
F32_MAX = 3.4028234663852886e38  # FLT_MAX, 0x7F7FFFFF


def fabs(a: float) -> float:
    """``abs.f32``: clears the sign bit, exact (``fabs(-0) = +0``; NaN stays NaN)."""
    return abs(a)


def _e4m3_table():
    out = []
    for b in range(256):
        sign, exp, mant = b >> 7, (b >> 3) & 0xF, b & 0x7
        if exp == 15 and mant == 7:
            v = NAN
        elif exp == 0:
            v = mant * 2.0**-9
        else:
            v = (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
        out.append(-v if sign else v)
    return tuple(out)


E4M3_VALUES = (
    _e4m3_table()
)  # byte -> value (e4m3fn: 0x7F / 0xFF are NaN, no infinities)


def e4m3_bits(v: float) -> int:
    """The byte of an e4m3fn value (``v`` must be exactly representable, e.g. the result of
    ``to_fp8(x, "e4m3")``)."""
    if math.isnan(v):
        return 0x7F
    neg = math.copysign(1.0, v) < 0
    a = abs(v)
    for b in range(127):
        if E4M3_VALUES[b] == a:
            return b | (0x80 if neg else 0)
    raise ValueError(f"{v!r} is not an e4m3 value")


# ----------------------------------------------------------------------------- MX (UE8M0) primitives
UE8M0_BIAS = 127
UE8M0_MAX_BYTE = 254  # the largest finite byte (2^127); 255 is the UE8M0 NaN
UE8M0_MIN_VALUE = 2.0**-127  # byte 0: a binary32 subnormal (0x00400000)
UE8M0_MAX_VALUE = 2.0**127  # byte 254
UE8M0_NAN_BYTE = 0xFF
UE8M0_ROUNDINGS = ("rp", "rz", "rn")


def to_ue8m0(x: float, rnd: str = "rp") -> float:
    """``cvt.<rnd>.satfinite.ue8m0x2.f32`` on one value, as the VALUE ``2^(byte - 127)`` of the resulting UE8M0 byte
    (every such value is a binary32: ``2^-127`` is the subnormal ``0x00400000``). ``rp``: the least power of two
    ``>= x`` (the hardware mode TE and FlashInfer use, "RCEIL"); ``rz``: the greatest power of two ``<= x``; ``rn``:
    the nearest power of two in linear space, ties up (no hardware mode: an exact software rule). Every mode converts the
    MAGNITUDE of ``x`` (the hardware ignores the sign: ``-2.0`` gives byte 128; never reached on the classes' domain, whose
    amax is ``fabs(x) >= +0``; found by the exhaustive device sweep), maps both zeros and ``0 < |x| <= 2^-127`` to
    byte 0 (the format has nothing smaller), ``|x| >= 2^127`` and infinities to byte 254 (satfinite), NaN to NaN (byte 255;
    out of scope). The device conversions are verified against this function over every binary32 bit pattern
    (the nightly UE8M0 sweep of ``tests/batch_invariant``)."""
    if rnd not in UE8M0_ROUNDINGS:
        raise ValueError(rnd)
    if math.isnan(x):
        return NAN
    x = abs(x)
    if x == 0:
        return UE8M0_MIN_VALUE
    if math.isinf(x):
        return UE8M0_MAX_VALUE
    fr = Fraction(x)
    e = fr.numerator.bit_length() - fr.denominator.bit_length()
    if Fraction(2) ** e > fr:
        e -= 1
    elif Fraction(2) ** (e + 1) <= fr:
        e += 1
    lo = Fraction(2) ** e  # 2^e <= x < 2^(e+1)
    if rnd == "rz":
        k = e
    elif rnd == "rp":
        k = e if fr == lo else e + 1
    else:
        k = e if fr < lo * 3 / 2 else e + 1  # nearest in linear space, ties up
    k = max(-UE8M0_BIAS, min(127, k))
    return math.ldexp(1.0, k)


def ue8m0_byte(v: float) -> int:
    """The byte of a UE8M0 value (a power of two in ``[2^-127, 2^127]``, as :func:`to_ue8m0` returns); NaN -> 0xFF."""
    if math.isnan(v):
        return UE8M0_NAN_BYTE
    m, e = math.frexp(v)  # v = m * 2^e with m in [0.5, 1)
    if m != 0.5 or not (-UE8M0_BIAS <= e - 1 <= 127):
        raise ValueError(f"{v!r} is not a UE8M0 value")
    return e - 1 + UE8M0_BIAS


def ue8m0_value(byte: int) -> float:
    """The value of a UE8M0 byte: ``2^(byte - 127)``; 0xFF is NaN."""
    if byte == UE8M0_NAN_BYTE:
        return NAN
    return math.ldexp(1.0, byte - UE8M0_BIAS)


CANONICAL_NAN_F32 = 0x7FFFFFFF  # what sm_100-class FP32 ops produce for any NaN result
CANONICAL_NAN_16 = 0x7FFF  # bf16 / f16 canonical NaN after cvt.rn from an f32 NaN


def to_f32(x: float) -> float:
    """The Python float ``x`` (a double) rounded to binary32 with RNE; inf/nan pass through.
    Every constant that enters a kernel or a spec goes through this, so the CPU reference,
    the symbolic spec and the device constant are the same binary32 value."""
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return x
    return round_to_format(
        Fraction(x), F32, zero_sign_neg=(x == 0.0 and math.copysign(1.0, x) < 0)
    )


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bf16_bits(x: float) -> int:
    return f32_bits(x) >> 16


def f16_bits(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]
