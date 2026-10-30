"""Adversarial inputs for the bitwise test tiers.

Benign Gaussian inputs let reduction-order bugs hide (most reorderings agree to the
last bit on them). These rows mix exponents spanning ~2^40, cancellation pairs,
exact and negative zeros, and subnormals so that almost any reassociation flips bits.
"""

from __future__ import annotations

import torch

from . import fp32


def adversarial_rows(
    M: int, H: int, dtype: torch.dtype, seed: int = 0, device="cpu"
) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    exp_span = 20.0 if dtype != torch.float16 else 6.0
    e = (torch.rand(M, H, generator=g) * 2 - 1) * exp_span
    mant = torch.rand(M, H, generator=g) + 1.0
    sign = torch.where(torch.rand(M, H, generator=g) < 0.5, -1.0, 1.0)
    x = sign * mant * torch.exp2(e)
    # cancellation pairs: x[2j+1] ~= -x[2j] with a tiny perturbation
    pairs = torch.rand(M, H // 2, generator=g) < 0.15
    idx = torch.arange(0, H - 1, 2)
    x[:, idx + 1] = torch.where(
        pairs,
        -x[:, idx] * (1 + 1e-3 * torch.randn(M, H // 2, generator=g)),
        x[:, idx + 1],
    )
    # a few huge outliers (kept finite in the target dtype), zeros, negative zeros, subnormals
    r = torch.rand(M, H, generator=g)
    outlier = 2.0**12 if dtype != torch.float16 else 2.0**4
    x = torch.where(r < 0.01, x * outlier, x)
    x = torch.where((r >= 0.01) & (r < 0.02), torch.zeros_like(x), x)
    x = torch.where((r >= 0.02) & (r < 0.03), -torch.zeros_like(x), x)
    x = torch.where(
        (r >= 0.03) & (r < 0.035), x * 2.0**-130, x
    )  # subnormal in f32/bf16
    x = x.to(dtype)
    assert torch.isfinite(x).all(), (
        "adversarial rows must be finite (NaN payloads are not canonical)"
    )
    return x.to(device)


def adversarial_weight(
    H: int, dtype: torch.dtype, seed: int = 1, device="cpu"
) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.rand(H, generator=g) * 2 - 1) * torch.exp2(
        (torch.rand(H, generator=g) * 2 - 1) * 4
    )
    w[::97] = 0.0
    w[1::89] = -0.0
    return w.to(dtype).to(device)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(
        {1: torch.int8, 2: torch.int16, 4: torch.int32}[t.element_size()]
    )


_CANON_NAN = {
    torch.float32: (torch.int32, fp32.CANONICAL_NAN_F32),
    torch.bfloat16: (torch.int16, fp32.CANONICAL_NAN_16),
    torch.float16: (torch.int16, fp32.CANONICAL_NAN_16),
    torch.float8_e4m3fn: (torch.int8, 0x7F),
    torch.float8_e5m2: (torch.int8, 0x7F),
}


def canonical_nan(
    t: torch.Tensor, nan_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """The spec does not define NaN payloads or signs; the device produces the canonical quiet
    NaN of each format (0x7fffffff, 0x7fff, 0x7f). Rewrite every NaN of the CPU reference to
    that pattern so a bitwise comparison is meaningful (F06). ``nan_mask`` may be supplied
    (e.g. from the float64 values) for dtypes without ``isnan`` support."""
    it, pat = _CANON_NAN[t.dtype]
    mask = torch.isnan(t) if nan_mask is None else nan_mask
    if not bool(mask.any()):
        return t
    b = t.contiguous().view(it).clone()
    b[mask] = pat
    return b.view(t.dtype)
