"""Tests for flashinfer.mm_fp4_w4a16_native (mm_fp4-layout contract).

The native variant accepts the same B + SF format as flashinfer.mm_fp4
(no host-side Marlin repack).  Tests validate against a pure-PyTorch
reference that inverts the 128x4 swizzle and dequantizes manually.
"""

import pytest
import torch

import flashinfer
from flashinfer.utils import is_sm100a_supported


_E2M1_TO_FLOAT32 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _unswizzle_sf_128x4(
    sf_swizzled: torch.Tensor, n: int, k_sf: int
) -> torch.Tensor:
    """Reverse the 128x4 SF swizzle.

    Returns an `(n, k_sf)` uint8 tensor in linear (n outer, k_sf inner)
    order.  Mirrors ``unswizzle_sf_128x4`` from
    ``scratch_w4a16_mmfp4_reference.py``.
    """
    if (n % 128) != 0 or (k_sf % 4) != 0:
        raise NotImplementedError(
            "_unswizzle_sf_128x4 requires N % 128 == 0 and K_sf % 4 == 0"
        )
    sf_flat = sf_swizzled.contiguous().view(-1)
    num_sf_tiles_k = k_sf // 4

    m_idx = torch.arange(n, device=sf_swizzled.device)
    k_idx = torch.arange(k_sf, device=sf_swizzled.device)
    m_grid, k_grid = torch.meshgrid(m_idx, k_idx, indexing="ij")

    dst = (
        ((m_grid // 128) * num_sf_tiles_k + k_grid // 4) * 512
        + (m_grid % 32) * 16
        + ((m_grid % 128) // 32) * 4
        + (k_grid % 4)
    )
    return sf_flat[dst]


def _dequant_b_to_float(
    b_fp4: torch.Tensor,  # (N, K/2) uint8
    b_sf_swizzled: torch.Tensor,  # 1-D or 2-D 128x4-swizzled FP8 SF
) -> torch.Tensor:
    """Per-element dequant: ``value = code * sf_byte_as_fp8`` (no global)."""
    device = b_fp4.device
    n, k_half = b_fp4.shape
    k = k_half * 2

    low = (b_fp4 & 0x0F).to(torch.long)
    high = (b_fp4 >> 4).to(torch.long)
    lut = _E2M1_TO_FLOAT32.to(device)
    codes = torch.empty((n, k), dtype=torch.float32, device=device)
    codes[:, 0::2] = lut[low]
    codes[:, 1::2] = lut[high]

    b_sf_un = _unswizzle_sf_128x4(b_sf_swizzled, n, k // 16)
    sf_f32 = b_sf_un.view(torch.float8_e4m3fn).float()  # (n, k/16)
    sf_broadcast = sf_f32.repeat_interleave(16, dim=-1)  # (n, k)
    return codes * sf_broadcast


def _torch_reference(
    a: torch.Tensor,
    b_fp4: torch.Tensor,
    b_sf_swizzled: torch.Tensor,
    alpha: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    b_dequant = _dequant_b_to_float(b_fp4, b_sf_swizzled)
    out_f32 = a.float() @ b_dequant.T
    return (out_f32 * alpha).to(out_dtype)


_NATIVE_VALID_SHAPES = [
    # (M, N, K) -- N % 128 == 0, K % 64 == 0
    (1, 128, 128),
    (4, 256, 256),
    (16, 256, 512),
    (32, 128, 128),
    (64, 512, 256),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    not is_sm100a_supported(torch.device("cuda")),
    reason="mm_fp4_w4a16_native currently requires SM100+ (Blackwell)",
)
@pytest.mark.parametrize("m,n,k", _NATIVE_VALID_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("alpha_kind", ["none", "scalar"])
def test_mm_fp4_w4a16_native_matches_reference(m, n, k, dtype, alpha_kind):
    device = "cuda"
    torch.manual_seed(m * 100 + n + k)
    a = torch.randn((m, k), device=device, dtype=dtype) * 0.5
    w = torch.randn((n, k), device=device, dtype=dtype) * 0.1
    g_b = (448 * 6) / w.float().abs().nan_to_num().max()

    w_fp4, w_inv_s = flashinfer.nvfp4_quantize(
        w, g_b, sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False, backend="cute-dsl",
    )

    if alpha_kind == "none":
        alpha = None
        alpha_val = 1.0
    else:
        # For nvfp4-quantized B with no A quantization, alpha = 1 / g_b
        # recovers the original A @ B^T.
        alpha_val = float(1.0 / g_b.item())
        alpha = torch.tensor([alpha_val], device=device, dtype=torch.float32)

    out = flashinfer.mm_fp4_w4a16_native(
        a, w_fp4, w_inv_s, alpha=alpha, out_dtype=dtype,
    )

    ref = _torch_reference(a, w_fp4, w_inv_s, alpha_val, dtype)

    # bf16 accumulator-cast tolerance: rtol ~ 1e-2.  Increase if M is
    # tiny (relative diff blows up when ref ~ 0).
    d = (out.float() - ref.float()).abs()
    norm_ratio = d.mean().item() / (ref.float().abs().mean().item() + 1e-6)
    assert norm_ratio < 0.05, (
        f"output diverges from torch reference: norm_ratio={norm_ratio:.4f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    not is_sm100a_supported(torch.device("cuda")),
    reason="mm_fp4_w4a16_native currently requires SM100+ (Blackwell)",
)
def test_mm_fp4_w4a16_native_out_parameter():
    device = "cuda"
    m, n, k = 4, 128, 128
    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16) * 0.5
    w = torch.randn((n, k), device=device, dtype=torch.bfloat16) * 0.1
    g_b = (448 * 6) / w.float().abs().nan_to_num().max()
    w_fp4, w_inv_s = flashinfer.nvfp4_quantize(
        w, g_b, sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False, backend="cute-dsl",
    )

    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    result = flashinfer.mm_fp4_w4a16_native(
        a, w_fp4, w_inv_s,
        alpha=torch.tensor([1.0], device=device, dtype=torch.float32),
        out_dtype=torch.bfloat16,
        out=out,
    )
    assert result is out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mm_fp4_w4a16_native_rejects_int32_b():
    """Passing Marlin-format int32 B must raise a clear error."""
    device = "cuda"
    a = torch.randn((4, 128), device=device, dtype=torch.bfloat16)
    marlin_b = torch.zeros((128 // 16, 128 * 2), device=device, dtype=torch.int32)
    b_sf = torch.zeros(128 * 8, device=device, dtype=torch.uint8)

    with pytest.raises(ValueError, match=r"b must be uint8"):
        flashinfer.mm_fp4_w4a16_native(a, marlin_b, b_sf)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mm_fp4_w4a16_native_rejects_n_not_multiple_of_128():
    """N must be a multiple of 128 for the 128x4 swizzle to align."""
    device = "cuda"
    a = torch.randn((4, 128), device=device, dtype=torch.bfloat16)
    # N=64 -- fails the N % 128 check.
    b = torch.zeros((64, 64), device=device, dtype=torch.uint8)
    b_sf = torch.zeros(128 * 8, device=device, dtype=torch.uint8)
    with pytest.raises(ValueError, match=r"N must be a multiple of 128"):
        flashinfer.mm_fp4_w4a16_native(a, b, b_sf)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mm_fp4_w4a16_native_rejects_k_not_multiple_of_64():
    """K must be a multiple of 64 (= 4 * GROUP_SIZE)."""
    device = "cuda"
    a = torch.randn((4, 32), device=device, dtype=torch.bfloat16)
    b = torch.zeros((128, 16), device=device, dtype=torch.uint8)
    b_sf = torch.zeros(128 * 2, device=device, dtype=torch.uint8)
    with pytest.raises(ValueError, match=r"K must be a multiple of 64"):
        flashinfer.mm_fp4_w4a16_native(a, b, b_sf)
