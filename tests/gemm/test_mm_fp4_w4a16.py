import pytest
import torch

from flashinfer import mm_fp4_w4a16


E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _make_packed_weight(k, n, device):
    codes = torch.randint(0, 16, (k, n), dtype=torch.uint8, device=device)
    packed = (codes[1::2, :] << 4) | codes[0::2, :]
    return codes, packed.contiguous()


def _dequant_reference(codes, b_descale, alpha, block_size):
    lut = torch.tensor(E2M1_TO_FLOAT32, dtype=torch.float32, device=codes.device)
    weight = lut[codes.long()]
    weight = weight * b_descale.to(torch.float32).repeat_interleave(block_size, dim=0)
    if alpha is not None:
        if alpha.numel() == 1:
            weight = weight * alpha.to(torch.float32).reshape(())
        else:
            weight = weight * alpha.to(torch.float32).reshape(1, -1)
    return weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m,n,k", [(1, 32, 64), (17, 48, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("alpha_kind", ["none", "scalar", "per_column"])
def test_mm_fp4_w4a16_matches_reference(m, n, k, dtype, alpha_kind):
    device = "cuda"
    block_size = 16
    a = torch.randn((m, k), device=device, dtype=dtype)
    codes, b = _make_packed_weight(k, n, device)
    b_descale = (
        torch.rand((k // block_size, n), device=device, dtype=torch.float32) * 0.5
        + 0.25
    ).to(torch.float8_e4m3fn)

    if alpha_kind == "none":
        alpha = None
    elif alpha_kind == "scalar":
        alpha = torch.tensor([0.5], device=device, dtype=torch.float32)
    else:
        alpha = torch.linspace(0.25, 1.0, n, device=device, dtype=torch.float32)

    out = mm_fp4_w4a16(a, b, b_descale, alpha, out_dtype=dtype, block_size=block_size)
    ref_weight = _dequant_reference(codes, b_descale, alpha, block_size)
    ref = torch.matmul(a.to(torch.float32), ref_weight).to(dtype)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mm_fp4_w4a16_out_parameter():
    device = "cuda"
    m, n, k = 4, 16, 32
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    codes, b = _make_packed_weight(k, n, device)
    b_descale = torch.ones(
        (k // 16, n), device=device, dtype=torch.float8_e4m3fn
    )
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)

    result = mm_fp4_w4a16(a, b, b_descale, out_dtype=torch.bfloat16, out=out)
    ref = torch.matmul(
        a.to(torch.float32), _dequant_reference(codes, b_descale, None, 16)
    ).to(torch.bfloat16)

    assert result is out
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mm_fp4_w4a16_rejects_non_kn_packed_weight():
    device = "cuda"
    a = torch.randn((2, 32), device=device, dtype=torch.bfloat16)
    b = torch.empty((8, 16), device=device, dtype=torch.uint8)
    b_descale = torch.empty((2, 16), device=device, dtype=torch.float8_e4m3fn)

    with pytest.raises(ValueError, match=r"\[K // 2, N\]"):
        mm_fp4_w4a16(a, b, b_descale)

