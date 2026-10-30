"""CPU backends of the quantization specs: the exact scalar backend and the vectorised torch backend.

* ``ExactF32``: every op through :mod:`flashinfer.batch_invariant.fp32` (Fractions): the ground truth, scalar, slow.
* ``TorchF32``: torch float32 CPU tensors (``+ - * /`` and ``min`` / ``max`` / ``abs`` are single IEEE
  roundings or exact; no flush-to-zero), vectorised; the e4m3 and e2m1 conversions are implemented here
  (``e4m3_round``, ``e2m1_round``) and validated against ``fp32.to_fp8`` / ``fp32.to_e2m1`` by
  the device-free quick tier.

Both implement the ``fp`` interface the specs are written against, so the whole-tensor references
(:mod:`flashinfer.batch_invariant.quant.mx_reference`) evaluate the *same* spec functions the kernel executes
(rule I8 of the contract); the exact backend evaluates the symbolic spec for a few rows and is compared bit for
bit with the vectorised one.
"""

from __future__ import annotations

import torch

from .. import fp32


class ExactF32:
    """Scalar exact backend over Python floats restricted to binary32."""

    @staticmethod
    def const(v):
        return fp32.to_f32(v)

    add = staticmethod(fp32.add)
    mul = staticmethod(fp32.mul)
    div = staticmethod(fp32.div)
    fma = staticmethod(fp32.fma)
    fmin = staticmethod(fp32.fmin)
    fmax = staticmethod(fp32.fmax)
    fmaxc = staticmethod(
        fp32.fmaxc
    )  # the SM107 max.f32 zero rule (shared with the attention classes)
    fabs = staticmethod(fp32.fabs)
    nzor = staticmethod(fp32.nzor)

    @staticmethod
    def cvt_out(a, fmt):
        if fmt == "e4m3":
            return fp32.to_fp8(a, "e4m3")
        if fmt == "e2m1":
            return fp32.to_e2m1(a)
        if fmt.startswith(
            "ue8m0_"
        ):  # the MX classes' scale conversion (flashinfer/batch_invariant/quant/mx_spec.py)
            return fp32.to_ue8m0(a, fmt[6:])
        return fp32.to_format(
            a, {"bf16": fp32.BF16, "f16": fp32.F16, "f32": fp32.F32}[fmt]
        )


# ----------------------------------------------------------------------------- vectorised conversions
_E2M1_MAGS = torch.tensor(fp32.E2M1_MAGNITUDES, dtype=torch.float32)
_E2M1_MIDS = torch.tensor(fp32._E2M1_MIDPOINTS, dtype=torch.float32)
_E4M3_POS = torch.tensor(
    fp32.E4M3_VALUES[:127], dtype=torch.float32
)  # bytes 0..126, ascending


def e2m1_round(a: torch.Tensor) -> torch.Tensor:
    """Vectorised :func:`flashinfer.batch_invariant.fp32.to_e2m1` (values, not codes): RNE to the e2m1 magnitudes with ties to
    the even code, saturation beyond 6, sign kept; NaN gives +6.0 as the device does."""
    a = a.to(torch.float32)
    m = a.abs()
    mids = _E2M1_MIDS.to(a.device)
    gt = (m.unsqueeze(-1) > mids).sum(-1)  # midpoints strictly below |a|
    eq = m.unsqueeze(-1) == mids  # at most one tie
    even_up = torch.tensor([(k + 1) % 2 == 0 for k in range(7)], device=a.device)
    code = gt + (eq & even_up).sum(-1)
    code = torch.where(m >= fp32.E2M1_MAX, torch.full_like(code, 7), code)
    v = _E2M1_MAGS.to(a.device)[code.clamp(0, 7)]
    v = torch.where(torch.signbit(a), -v, v)
    return torch.where(torch.isnan(a), torch.full_like(v, fp32.E2M1_MAX), v)


def e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """4-bit codes (uint8) of e2m1 values (sign in bit 3)."""
    mags = _E2M1_MAGS.to(v.device)
    k = torch.searchsorted(mags, v.abs().contiguous()).clamp(max=7)
    assert bool((mags[k] == v.abs()).all()), "not e2m1 values"
    return k.to(torch.uint8) | torch.where(torch.signbit(v), 8, 0).to(torch.uint8)


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """(…, K) codes -> (…, K/2) bytes; element 2j in the low nibble."""
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    lo, hi = packed & 0xF, packed >> 4
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def e4m3_round(a: torch.Tensor) -> torch.Tensor:
    """Vectorised ``cvt.rn.satfinite.e4m3x2.f32`` as values: RNE to e4m3fn (3 mantissa bits, subnormals in
    steps of 2^-9), magnitudes beyond 448 saturate to 448, sign kept, NaN stays NaN."""
    a = a.to(torch.float32)
    m = a.abs().clamp(max=448.0)  # satfinite (also maps +inf to 448)
    small = m < 2.0**-6
    sub = torch.round(m * 2.0**9) * 2.0**-9  # torch.round is half-to-even
    mant, exp = torch.frexp(
        torch.where(small, torch.ones_like(m), m)
    )  # m = mant * 2^exp, mant in [0.5, 1)
    ulp = torch.ldexp(torch.ones_like(m), exp - 4)  # 2^(e-3) with e = exp - 1
    nrm = torch.round(m / ulp) * ulp
    v = torch.where(small, sub, nrm)
    v = torch.where(torch.signbit(a), -v, v)
    return torch.where(torch.isnan(a), a, v)


def e4m3_bytes(v: torch.Tensor) -> torch.Tensor:
    """Bytes (uint8) of e4m3fn values (exactly representable, as :func:`e4m3_round` returns)."""
    tab = _E4M3_POS.to(v.device)
    k = torch.searchsorted(tab, v.abs().contiguous()).clamp(max=126)
    ok = (tab[k] == v.abs()) | torch.isnan(v)
    assert bool(ok.all()), "not e4m3 values"
    b = k.to(torch.uint8) | torch.where(torch.signbit(v), 0x80, 0).to(torch.uint8)
    return torch.where(
        torch.isnan(v), torch.tensor(0x7F, dtype=torch.uint8, device=v.device), b
    )


class TorchF32:
    """Vectorised backend on float32 CPU (or CUDA) tensors with broadcasting. Scalars are 0-d tensors."""

    @staticmethod
    def const(v):
        return torch.tensor(fp32.to_f32(v), dtype=torch.float32)

    @staticmethod
    def add(a, b):
        return torch.add(a, b)

    @staticmethod
    def mul(a, b):
        return torch.mul(a, b)

    @staticmethod
    def div(a, b):
        return torch.div(a, b)

    @staticmethod
    def fma(a, b, c):
        """``fma.rn.f32`` for the MX classes' ``zero = "positive"`` rule, ``fma(x, inv, +0)``: the product of two binary32
        values is exact in binary64 (48 significant bits) and adding ``+0`` is exact (``-0 + +0 = +0``), so one rounding to
        binary32 remains -- the fused semantics. (A general ``c`` could double-round; the specs only use ``c = +0``.)"""
        c = torch.as_tensor(c, dtype=torch.float32, device=a.device)
        assert bool((c == 0).all()), "TorchF32.fma is exact only for c = +-0"
        return (
            a.double()
            * torch.as_tensor(b, dtype=torch.float32, device=a.device).double()
            + c.double()
        ).float()

    @staticmethod
    def fmin(a, b):
        return torch.minimum(
            a, torch.as_tensor(b, dtype=torch.float32, device=a.device)
        )

    @staticmethod
    def fmax(a, b):
        return torch.maximum(
            a, torch.as_tensor(b, dtype=torch.float32, device=a.device)
        )

    fmaxc = fmax

    @staticmethod
    def fabs(a):
        return torch.abs(a)

    @staticmethod
    def nzor(a, b):
        return torch.where(
            a != 0, a, torch.as_tensor(b, dtype=torch.float32, device=a.device)
        )

    @staticmethod
    def cvt_out(a, fmt):
        if fmt == "e4m3":
            return e4m3_round(a)
        if fmt == "e2m1":
            return e2m1_round(a)
        if fmt.startswith("ue8m0_"):
            from .mx_reference import ue8m0_round

            return ue8m0_round(a, fmt[6:])
        return a.to(
            {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[fmt]
        )
