"""Tests for flashinfer.mm_fp4_w4a16 (Option B contract).

The kernel takes pre-prepared B from ``flashinfer.prepare_fp4_w4a16_weight``,
not raw packed FP4.  Tests validate both backends (cute-dsl + torch) against
a naive full-precision reference.
"""

import pytest
import torch

from flashinfer import (
    mm_fp4_w4a16,
    prepare_fp4_w4a16_weight,
)


E2M1_TO_FLOAT32 = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _make_raw_packed_weight(k, n, device, seed=0):
    """Generate random FP4 codes and the standard (K // 2, N) uint8 pack."""
    g = torch.Generator(device=device).manual_seed(seed)
    codes = torch.randint(0, 16, (k, n), dtype=torch.uint8, device=device, generator=g)
    packed = ((codes[1::2, :] << 4) | codes[0::2, :]).contiguous()
    return codes, packed


def _dequant_reference(codes, b_descale, alpha, block_size):
    """Full-precision reference: dequant FP4 -> fp32 -> matmul."""
    lut = torch.tensor(E2M1_TO_FLOAT32, dtype=torch.float32, device=codes.device)
    weight = lut[codes.long()]
    weight = weight * b_descale.to(torch.float32).repeat_interleave(block_size, dim=0)
    if alpha is not None:
        weight = weight * alpha.to(torch.float32).reshape(())
    return weight


# K must be a multiple of 16 (Marlin K-tile); N must be a multiple of 64
# (kernel tile_N).  These constraints come from the kernel partition.
_VALID_SHAPES = [(1, 64, 64), (4, 64, 64), (17, 64, 128), (64, 128, 256)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m,n,k", _VALID_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("alpha_kind", ["none", "scalar"])
@pytest.mark.parametrize("backend", ["torch", "cute-dsl"])
def test_mm_fp4_w4a16_matches_reference(m, n, k, dtype, alpha_kind, backend):
    device = "cuda"
    block_size = 16
    a = torch.randn((m, k), device=device, dtype=dtype)
    codes, raw_packed = _make_raw_packed_weight(k, n, device)
    b_descale = (
        torch.rand((k // block_size, n), device=device, dtype=torch.float32) * 0.5
        + 0.25
    ).to(torch.float8_e4m3fn)

    if alpha_kind == "none":
        alpha = None
    else:
        alpha = torch.tensor([0.5], device=device, dtype=torch.float32)

    # New contract: caller prepares once.
    b_prepared = prepare_fp4_w4a16_weight(raw_packed)

    out = mm_fp4_w4a16(
        a,
        b_prepared,
        b_descale,
        alpha,
        out_dtype=dtype,
        block_size=block_size,
        backend=backend,
    )
    ref_weight = _dequant_reference(codes, b_descale, alpha, block_size)
    ref = torch.matmul(a.to(torch.float32), ref_weight).to(dtype)

    # cute-dsl path goes through fp32 inside the MMA accumulator but
    # casts intermediate scale multiplies to bf16/fp16 -- tolerate one
    # ULP of 16-bit slack.
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend", ["torch", "cute-dsl"])
def test_mm_fp4_w4a16_out_parameter(backend):
    device = "cuda"
    m, n, k = 4, 64, 64
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    codes, raw_packed = _make_raw_packed_weight(k, n, device)
    b_descale = torch.ones(
        (k // 16, n), device=device, dtype=torch.float8_e4m3fn
    )
    b_prepared = prepare_fp4_w4a16_weight(raw_packed)
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)

    result = mm_fp4_w4a16(
        a,
        b_prepared,
        b_descale,
        out_dtype=torch.bfloat16,
        out=out,
        backend=backend,
    )
    ref = torch.matmul(
        a.to(torch.float32), _dequant_reference(codes, b_descale, None, 16)
    ).to(torch.bfloat16)

    assert result is out
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mm_fp4_w4a16_rejects_raw_packed_input():
    """The new contract requires pre-prepared (K // 16, N * 2) int32.

    Passing the legacy (K // 2, N) uint8 layout must raise a clear error
    that points users at ``prepare_fp4_w4a16_weight``.
    """
    device = "cuda"
    a = torch.randn((4, 64), device=device, dtype=torch.bfloat16)
    raw = torch.zeros((32, 64), device=device, dtype=torch.uint8)  # legacy shape
    b_descale = torch.zeros(
        (4, 64), device=device, dtype=torch.float8_e4m3fn
    )

    with pytest.raises(ValueError, match=r"prepare_fp4_w4a16_weight"):
        mm_fp4_w4a16(a, raw, b_descale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mm_fp4_w4a16_rejects_n_not_multiple_of_64():
    """N must be a multiple of 64 (kernel tile_N)."""
    device = "cuda"
    a = torch.randn((4, 64), device=device, dtype=torch.bfloat16)
    # Build a valid prepared shape but with N=32 -- impossible because
    # prepare_fp4_w4a16_weight rejects N % 64 != 0, so we synthesise a
    # malformed int32 tensor manually.
    bad_b = torch.zeros((4, 64), device=device, dtype=torch.int32)  # implies N=32
    b_descale = torch.zeros((4, 32), device=device, dtype=torch.float8_e4m3fn)

    with pytest.raises(ValueError, match=r"N must be a multiple of 64"):
        mm_fp4_w4a16(a, bad_b, b_descale)
