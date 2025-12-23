# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for Fused RMSNorm + FP4 Quantization using CuTe-DSL backend.
"""

import pytest
import torch

from flashinfer.cute_dsl.utils import is_cute_dsl_available
from tests.test_helpers.utils_fp4 import cast_from_fp4


def get_cc():
    """Get CUDA compute capability."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def llama_rms_norm(x, w, eps=1e-6):
    """Reference RMSNorm implementation (LLaMA style)."""
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


def dequantize_fp4_output(
    y_fp4: torch.Tensor,
    block_scale: torch.Tensor,
    block_size: int,
    global_scale: torch.Tensor | None = None,
):
    """
    Dequantize packed FP4 tensor using the associated block scales.

    Handles both 2D inputs shaped [B, H/2] and 3D inputs shaped [B, S, H/2].

    If global_scale is provided, the result is divided by global_scale to reverse
    the global scaling applied during quantization:
        original_value = (fp4_value * block_scale) / global_scale
    """
    y_fp4_float = cast_from_fp4(y_fp4)
    if y_fp4_float.dim() == 2:
        b, hidden_size = y_fp4_float.shape
        assert hidden_size % block_size == 0
        y_fp4_float = y_fp4_float.view(b, hidden_size // block_size, block_size)
        # Handle different scale dtype (E4M3 vs UE8M0)
        if block_scale.dtype == torch.uint8:
            # UE8M0: scale = 2^(ue8m0 - 127)
            scales = torch.pow(2.0, block_scale.int() - 127).unsqueeze(-1)
        else:
            scales = block_scale.float().unsqueeze(-1)
        result = (y_fp4_float * scales).reshape(b, hidden_size)
        if global_scale is not None:
            result = result / global_scale.item()
        return result
    elif y_fp4_float.dim() == 3:
        b, s, hidden_size = y_fp4_float.shape
        assert hidden_size % block_size == 0
        y_fp4_float = y_fp4_float.view(b, s, hidden_size // block_size, block_size)
        if block_scale.dtype == torch.uint8:
            scales = torch.pow(2.0, block_scale.int() - 127).unsqueeze(-1)
        else:
            scales = block_scale.float().unsqueeze(-1)
        result = (y_fp4_float * scales).reshape(b, s, hidden_size)
        if global_scale is not None:
            result = result / global_scale.item()
        return result
    else:
        raise ValueError(f"Unsupported FP4 output rank: {y_fp4_float.dim()}")


def requires_cute_dsl():
    """Check if CuTe-DSL is available."""
    return is_cute_dsl_available()


def requires_hopper_or_later():
    """Check if running on Hopper (SM90+) or later GPU."""
    return get_cc() >= 90


def requires_blackwell():
    """Check if running on Blackwell GPU."""
    return get_cc() >= 100


# Skip conditions
cute_dsl_available = pytest.mark.skipif(
    not requires_cute_dsl(), reason="CuTe-DSL not available"
)

hopper_required = pytest.mark.skipif(
    not requires_hopper_or_later(),
    reason="CuTe-DSL kernel requires Hopper (SM90+) or later GPU",
)

blackwell_required = pytest.mark.skipif(
    not requires_blackwell(),
    reason="FP4 quantization requires Blackwell GPU (SM100+)",
)


@cute_dsl_available
@blackwell_required
class TestRMSNormNVFP4QuantCuteDSL:
    """Tests for CuTe-DSL RMSNorm + NVFP4 Quantization (block_size=16, E4M3 scales, global_scale)."""

    @pytest.mark.parametrize(
        "batch_size", [1, 4, 16, 32, 7, 13, 33, 100, 128, 8192, 16384]
    )
    @pytest.mark.parametrize(
        "hidden_size", [64, 128, 256, 512, 1024, 1536, 2048, 4096, 8192]
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("eps", [1e-5, 1e-6])
    def test_rmsnorm_nvfp4quant_2d(self, batch_size, hidden_size, dtype, eps):
        """Test fused RMSNorm + NVFP4 quantization with 2D input."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_nvfp4quant

        torch.manual_seed(42)
        block_size = 16  # Fixed for NVFP4

        # Create input tensors
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Allocate output tensors
        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Run fused kernel
        _, _, global_scale = rmsnorm_nvfp4quant(x, weight, y_fp4, block_scale, eps=eps)

        # Verify output shapes
        assert y_fp4.shape == (batch_size, hidden_size // 2)
        assert block_scale.shape == (batch_size, hidden_size // block_size)
        assert global_scale.shape == (1,)

        # Verify output dtypes
        assert y_fp4.dtype == torch.uint8
        assert block_scale.dtype == torch.float8_e4m3fn
        assert global_scale.dtype == torch.float32

        # Reference computation
        ref_rmsnorm = llama_rms_norm(x, weight, eps=eps)

        # Verify global_scale is computed correctly: FP8_MAX * FP4_MAX / max_abs
        expected_global_scale = 448.0 * 6.0 / ref_rmsnorm.abs().max().item()
        assert global_scale.item() > 0, "global_scale should be positive"
        torch.testing.assert_close(
            global_scale,
            torch.tensor([expected_global_scale], dtype=torch.float32, device="cuda"),
            rtol=0.01,
            atol=1.0,
        )

        # Dequantize FP4 output for value-level comparison
        # Note: global_scale must be applied during dequantization to reverse the scaling
        # Tolerance based on separate FP4 roundtrip test (rtol=0.3, atol=0.5)
        y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size, global_scale)
        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.5,
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 3, 7, 128])
    @pytest.mark.parametrize("seq_len", [16, 64, 128, 37, 99])
    @pytest.mark.parametrize("hidden_size", [128, 256, 1536, 4096, 8192])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rmsnorm_nvfp4quant_3d(self, batch_size, seq_len, hidden_size, dtype):
        """Test fused RMSNorm + NVFP4 quantization with 3D input."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_nvfp4quant

        torch.manual_seed(42)
        block_size = 16  # Fixed for NVFP4
        eps = 1e-5

        # Create input tensors
        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Allocate output tensors
        y_fp4 = torch.empty(
            batch_size, seq_len, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale = torch.empty(
            batch_size,
            seq_len,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Run fused kernel
        _, _, global_scale = rmsnorm_nvfp4quant(x, weight, y_fp4, block_scale, eps=eps)

        # Verify output shapes
        assert y_fp4.shape == (batch_size, seq_len, hidden_size // 2)
        assert block_scale.shape == (batch_size, seq_len, hidden_size // block_size)
        assert global_scale.shape == (1,)

        # Verify output dtypes
        assert y_fp4.dtype == torch.uint8
        assert block_scale.dtype == torch.float8_e4m3fn
        assert global_scale.dtype == torch.float32

        # Reference computation
        ref_rmsnorm = llama_rms_norm(x, weight, eps=eps)

        # Verify global_scale is computed correctly: FP8_MAX * FP4_MAX / max_abs
        expected_global_scale = 448.0 * 6.0 / ref_rmsnorm.abs().max().item()
        torch.testing.assert_close(
            global_scale,
            torch.tensor([expected_global_scale], dtype=torch.float32, device="cuda"),
            rtol=0.01,
            atol=1.0,
        )

        # Dequantize FP4 output for value-level comparison
        # Note: global_scale must be applied during dequantization to reverse the scaling
        y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size, global_scale)
        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.5,
        )

    @pytest.mark.parametrize(
        "batch_size,hidden_size",
        [
            (512, 4096),
            (1024, 4096),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_large_batch(self, batch_size, hidden_size, dtype):
        """Test with large batch sizes."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_nvfp4quant

        torch.manual_seed(42)
        block_size = 16  # Fixed for NVFP4
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Should complete without error
        _, _, global_scale = rmsnorm_nvfp4quant(x, weight, y_fp4, block_scale, eps=eps)

        # Reference computation (sample first 10 rows for speed)
        ref_rmsnorm = llama_rms_norm(x[:10], weight, eps=eps)
        y_dequant = dequantize_fp4_output(
            y_fp4[:10], block_scale[:10], block_size, global_scale
        )

        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.5,
        )


@cute_dsl_available
@blackwell_required
class TestRMSNormMXFP4QuantCuteDSL:
    """Tests for MXFP4 format (block_size=32, UE8M0 scales, no global_scale)."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 7, 25, 128, 8192])
    @pytest.mark.parametrize("hidden_size", [128, 256, 512, 1536, 2048, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mxfp4_basic(self, batch_size, hidden_size, dtype):
        """Test MXFP4 format (block_size=32, UE8M0 scales)."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_mxfp4quant

        torch.manual_seed(42)
        block_size = 32  # Fixed for MXFP4
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        # UE8M0 scale factors are returned as uint8
        block_scale = torch.empty(
            batch_size, hidden_size // block_size, device="cuda", dtype=torch.uint8
        )

        # MXFP4 doesn't use global_scale
        y_fp4_out, block_scale_out = rmsnorm_mxfp4quant(
            x,
            weight,
            y_fp4,
            block_scale,
            eps=eps,
        )

        # Verify output shapes
        assert y_fp4_out.shape == (batch_size, hidden_size // 2)
        assert block_scale_out.shape == (batch_size, hidden_size // block_size)
        assert y_fp4_out.dtype == torch.uint8
        assert block_scale_out.dtype == torch.uint8

        # Reference computation
        ref_rmsnorm = llama_rms_norm(x, weight, eps=eps)

        # Dequantize FP4 output (no global_scale for MXFP4)
        # MXFP4 uses power-of-2 scales which can introduce more quantization error
        y_dequant = dequantize_fp4_output(y_fp4_out, block_scale_out, block_size)
        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.4,
            atol=0.7,
        )


@cute_dsl_available
@blackwell_required
class TestNVFP4VsReference:
    """Tests comparing CuTe-DSL NVFP4 fused kernel output against reference RMSNorm."""

    @pytest.mark.parametrize("batch_size", [4, 16, 128, 512, 8192])
    @pytest.mark.parametrize("hidden_size", [256, 512, 1024, 1536, 4096, 8192])
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_nvfp4_vs_reference(self, batch_size, hidden_size, dtype):
        """
        Compare CuTe-DSL NVFP4 fused output with reference RMSNorm.

        We compare the dequantized output against the reference RMSNorm,
        rather than comparing bitwise with separate fp4_quantize (which uses
        different scaling approaches).
        """
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_nvfp4quant
        from flashinfer.norm import rmsnorm

        torch.manual_seed(42)
        block_size = 16  # Fixed for NVFP4
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Fused CuTe-DSL kernel
        y_fp4_fused = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale_fused = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        _, _, global_scale = rmsnorm_nvfp4quant(
            x, weight, y_fp4_fused, block_scale_fused, eps=eps
        )

        # Reference: RMSNorm only
        y_ref = rmsnorm(x, weight, eps=eps)

        # Verify output shapes
        assert y_fp4_fused.shape == (batch_size, hidden_size // 2)
        assert block_scale_fused.shape == (batch_size, hidden_size // block_size)
        assert global_scale.shape == (1,)
        assert global_scale.dtype == torch.float32

        # Dequantize fused output and compare to reference
        y_fused_dequant = dequantize_fp4_output(
            y_fp4_fused, block_scale_fused, block_size, global_scale
        )

        # Value-level comparison against reference RMSNorm output
        torch.testing.assert_close(
            y_fused_dequant,
            y_ref.float(),
            rtol=0.3,
            atol=0.5,
        )


@cute_dsl_available
@blackwell_required
class TestLargeHiddenSize:
    """Tests for large hidden sizes (16K, 32K) that use cluster synchronization.

    These hidden sizes trigger the cluster sync code path in the CuTe-DSL kernel.
    Uses fewer batch sizes to keep test time reasonable, and samples rows for
    value comparison since full dequantization is slow for large tensors.
    """

    @pytest.mark.parametrize("batch_size", [1, 16, 128, 1024])
    @pytest.mark.parametrize("hidden_size", [16384, 32768])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_hidden_nvfp4(self, batch_size, hidden_size, dtype):
        """Test NVFP4 format with large hidden sizes (cluster sync path)."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_nvfp4quant

        torch.manual_seed(42)
        block_size = 16  # Fixed for NVFP4
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Run kernel
        _, _, global_scale = rmsnorm_nvfp4quant(x, weight, y_fp4, block_scale, eps=eps)

        # Verify output shapes
        assert y_fp4.shape == (batch_size, hidden_size // 2)
        assert block_scale.shape == (batch_size, hidden_size // block_size)
        assert global_scale.shape == (1,)
        assert y_fp4.dtype == torch.uint8
        assert block_scale.dtype == torch.float8_e4m3fn
        assert global_scale.dtype == torch.float32

        # Sample first few rows for value comparison (full dequant is slow)
        num_check = min(10, batch_size)
        ref_rmsnorm = llama_rms_norm(x[:num_check], weight, eps=eps)
        y_dequant = dequantize_fp4_output(
            y_fp4[:num_check], block_scale[:num_check], block_size, global_scale
        )

        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.5,
        )

    @pytest.mark.parametrize("batch_size", [1, 16, 128, 1024])
    @pytest.mark.parametrize("hidden_size", [16384, 32768])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_hidden_mxfp4(self, batch_size, hidden_size, dtype):
        """Test MXFP4 format with large hidden sizes (cluster sync path)."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_mxfp4quant

        torch.manual_seed(42)
        block_size = 32  # Fixed for MXFP4
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y_fp4 = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale = torch.empty(
            batch_size, hidden_size // block_size, device="cuda", dtype=torch.uint8
        )

        # Run kernel (MXFP4 doesn't use global_scale)
        y_fp4_out, block_scale_out = rmsnorm_mxfp4quant(
            x,
            weight,
            y_fp4,
            block_scale,
            eps=eps,
        )

        # Verify output shapes
        assert y_fp4_out.shape == (batch_size, hidden_size // 2)
        assert block_scale_out.shape == (batch_size, hidden_size // block_size)
        assert y_fp4_out.dtype == torch.uint8
        assert block_scale_out.dtype == torch.uint8

        # Sample first few rows for value comparison (full dequant is slow)
        num_check = min(10, batch_size)
        ref_rmsnorm = llama_rms_norm(x[:num_check], weight, eps=eps)
        y_dequant = dequantize_fp4_output(
            y_fp4_out[:num_check], block_scale_out[:num_check], block_size
        )

        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.7,
        )


def unswizzle_sf(
    sf: torch.Tensor, row: int, col: int, scaling_vector_size: int = 16
) -> torch.Tensor:
    """
    Unswizzle scale factors from 128x4 tile swizzled layout to row-major layout.

    The swizzle pattern uses 128x4 tiles where scales are arranged as:
    [m_tile][k_tile][outer_m (32)][inner_m (4)][inner_k (4)]

    Parameters
    ----------
    sf : torch.Tensor
        Swizzled scale factor tensor.
    row : int
        Number of rows (batch_size).
    col : int
        Number of columns (hidden_size).
    scaling_vector_size : int
        Block size for quantization (16 for NVFP4, 32 for MXFP4).

    Returns
    -------
    torch.Tensor
        Unswizzled scale factors in row-major layout, shape (row, col // scaling_vector_size).
    """
    factor = scaling_vector_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    # SF layout [num_m_tiles, num_k_tiles, 32 (m_tile column major), 4 (m_tile column major), 4(k_tile)]
    sf_reshaped = sf.view(num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzle = sf_reshaped.transpose(1, 3)
    sf_unswizzle = sf_unswizzle.reshape(num_m_tiles * 32 * 4, num_k_tiles * 4)
    sf_unswizzle_sliced = sf_unswizzle[:row, : (col // scaling_vector_size)]
    return sf_unswizzle_sliced.contiguous()


@pytest.mark.skipif(not is_cute_dsl_available(), reason="CuTe-DSL not available")
@pytest.mark.skipif(get_cc() < 100, reason="Requires SM100+")
class TestSwizzledScaleFactors:
    """Tests for swizzled scale factor output layout."""

    @pytest.mark.parametrize("batch_size", [1, 16, 128, 256])
    @pytest.mark.parametrize("hidden_size", [512, 1024, 2048, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_nvfp4_swizzled_vs_unswizzled(self, batch_size, hidden_size, dtype):
        """
        Test that swizzled output, when unswizzled, matches the non-swizzled output.
        Uses NVFP4 format (block_size=16, E4M3 scales).
        """
        from flashinfer.cute_dsl import rmsnorm_nvfp4quant

        block_size = 16  # Fixed for NVFP4
        torch.manual_seed(42)

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Non-swizzled output
        y_fp4_ref = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale_ref = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        # Swizzled output - allocate padded buffer for swizzle
        factor = block_size * 4
        num_m_tiles = (batch_size + 128 - 1) // 128
        num_k_tiles = (hidden_size + factor - 1) // factor
        swizzled_size = num_m_tiles * num_k_tiles * 32 * 4 * 4  # 128x4 tile pattern
        y_fp4_swizzled = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale_swizzled = torch.empty(
            swizzled_size, device="cuda", dtype=torch.float8_e4m3fn
        )

        # Run kernels
        _, _, global_scale_ref = rmsnorm_nvfp4quant(
            x,
            weight,
            y_fp4_ref,
            block_scale_ref,
            is_sf_swizzled_layout=False,
        )
        _, _, global_scale_swizzled = rmsnorm_nvfp4quant(
            x,
            weight,
            y_fp4_swizzled,
            block_scale_swizzled,
            is_sf_swizzled_layout=True,
        )

        # Unswizzle and compare
        block_scale_unswizzled = unswizzle_sf(
            block_scale_swizzled.view(torch.uint8), batch_size, hidden_size, block_size
        ).view(torch.float8_e4m3fn)

        # FP4 values should be identical
        torch.testing.assert_close(y_fp4_swizzled, y_fp4_ref)

        # Scale factors should match after unswizzling
        torch.testing.assert_close(
            block_scale_unswizzled.view(torch.uint8), block_scale_ref.view(torch.uint8)
        )

        # Global scales should be identical
        torch.testing.assert_close(global_scale_ref, global_scale_swizzled)

    @pytest.mark.parametrize("batch_size", [1, 16, 128, 256])
    @pytest.mark.parametrize("hidden_size", [512, 1024, 2048, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mxfp4_swizzled_vs_unswizzled(self, batch_size, hidden_size, dtype):
        """
        Test that swizzled output, when unswizzled, matches the non-swizzled output.
        Uses MXFP4 format (block_size=32, UE8M0 scales).
        """
        from flashinfer.cute_dsl import rmsnorm_mxfp4quant

        block_size = 32  # Fixed for MXFP4
        torch.manual_seed(42)

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Non-swizzled output
        y_fp4_ref = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale_ref = torch.empty(
            batch_size, hidden_size // block_size, device="cuda", dtype=torch.uint8
        )

        # Swizzled output - allocate padded buffer for swizzle
        factor = block_size * 4
        num_m_tiles = (batch_size + 128 - 1) // 128
        num_k_tiles = (hidden_size + factor - 1) // factor
        swizzled_size = num_m_tiles * num_k_tiles * 32 * 4 * 4  # 128x4 tile pattern
        y_fp4_swizzled = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale_swizzled = torch.empty(
            swizzled_size, device="cuda", dtype=torch.uint8
        )

        # Run kernels (MXFP4 doesn't use global_scale)
        y_fp4_ref_out, block_scale_ref_out = rmsnorm_mxfp4quant(
            x,
            weight,
            y_fp4_ref,
            block_scale_ref,
            is_sf_swizzled_layout=False,
        )
        y_fp4_swizzled_out, block_scale_swizzled_out = rmsnorm_mxfp4quant(
            x,
            weight,
            y_fp4_swizzled,
            block_scale_swizzled,
            is_sf_swizzled_layout=True,
        )

        # Unswizzle and compare
        block_scale_unswizzled = unswizzle_sf(
            block_scale_swizzled_out, batch_size, hidden_size, block_size
        )

        # FP4 values should be identical
        torch.testing.assert_close(y_fp4_swizzled_out, y_fp4_ref_out)

        # Scale factors should match after unswizzling
        torch.testing.assert_close(block_scale_unswizzled, block_scale_ref_out)


@cute_dsl_available
@blackwell_required
class TestAutoAllocation:
    """Tests for automatic output tensor allocation when y_fp4, block_scale, and global_scale are None."""

    @pytest.mark.parametrize("batch_size", [1, 16, 128])
    @pytest.mark.parametrize("hidden_size", [256, 1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_auto_allocation_2d_nvfp4(self, batch_size, hidden_size, dtype):
        """Test auto-allocation with 2D input and NVFP4 format."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_nvfp4quant

        torch.manual_seed(42)
        block_size = 16  # Fixed for NVFP4
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Call without providing y_fp4, block_scale, and global_scale
        y_fp4, block_scale, global_scale = rmsnorm_nvfp4quant(x, weight, eps=eps)

        # Verify output shapes
        assert y_fp4.shape == (batch_size, hidden_size // 2)
        assert block_scale.shape == (batch_size, hidden_size // block_size)
        assert global_scale.shape == (1,)

        # Verify output dtypes
        assert y_fp4.dtype == torch.uint8
        assert block_scale.dtype == torch.float8_e4m3fn
        assert global_scale.dtype == torch.float32

        # Reference computation
        ref_rmsnorm = llama_rms_norm(x, weight, eps=eps)

        # Verify global_scale is computed correctly: FP8_MAX * FP4_MAX / max_abs
        expected_global_scale = 448.0 * 6.0 / ref_rmsnorm.abs().max().item()
        assert global_scale.item() > 0, "global_scale should be positive"
        torch.testing.assert_close(
            global_scale,
            torch.tensor([expected_global_scale], dtype=torch.float32, device="cuda"),
            rtol=0.01,
            atol=1.0,
        )

        # Dequantize and verify values
        y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size, global_scale)
        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.5,
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("seq_len", [16, 64])
    @pytest.mark.parametrize("hidden_size", [256, 1024])
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_auto_allocation_3d_nvfp4(self, batch_size, seq_len, hidden_size, dtype):
        """Test auto-allocation with 3D input and NVFP4 format."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_nvfp4quant

        torch.manual_seed(42)
        block_size = 16  # Fixed for NVFP4
        eps = 1e-6

        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Call without providing y_fp4, block_scale, and global_scale
        y_fp4, block_scale, global_scale = rmsnorm_nvfp4quant(x, weight, eps=eps)

        # Verify output shapes
        assert y_fp4.shape == (batch_size, seq_len, hidden_size // 2)
        assert block_scale.shape == (batch_size, seq_len, hidden_size // block_size)
        assert global_scale.shape == (1,)

        # Verify output dtypes
        assert y_fp4.dtype == torch.uint8
        assert block_scale.dtype == torch.float8_e4m3fn
        assert global_scale.dtype == torch.float32

        # Reference computation
        ref_rmsnorm = llama_rms_norm(x, weight, eps=eps)

        # Verify global_scale is computed correctly: FP8_MAX * FP4_MAX / max_abs
        expected_global_scale = 448.0 * 6.0 / ref_rmsnorm.abs().max().item()
        torch.testing.assert_close(
            global_scale,
            torch.tensor([expected_global_scale], dtype=torch.float32, device="cuda"),
            rtol=0.01,
            atol=1.0,
        )

        # Dequantize and verify values
        y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size, global_scale)
        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.5,
        )

    @pytest.mark.parametrize("batch_size", [1, 16, 128])
    @pytest.mark.parametrize("hidden_size", [256, 1024])
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_auto_allocation_mxfp4(self, batch_size, hidden_size, dtype):
        """Test auto-allocation with MXFP4 format (block_size=32, UE8M0 scales)."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_mxfp4quant

        torch.manual_seed(42)
        block_size = 32  # Fixed for MXFP4
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Call without providing y_fp4 and block_scale (MXFP4 doesn't use global_scale)
        y_fp4, block_scale = rmsnorm_mxfp4quant(x, weight, eps=eps)

        # Verify output shapes
        assert y_fp4.shape == (batch_size, hidden_size // 2)
        assert block_scale.shape == (batch_size, hidden_size // block_size)

        # Verify output dtypes
        assert y_fp4.dtype == torch.uint8
        assert block_scale.dtype == torch.uint8  # UE8M0 uses uint8

        # Reference computation
        ref_rmsnorm = llama_rms_norm(x, weight, eps=eps)

        # Dequantize and verify values (no global_scale for MXFP4)
        y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size)
        torch.testing.assert_close(
            y_dequant,
            ref_rmsnorm.float(),
            rtol=0.3,
            atol=0.7,
        )

    @pytest.mark.parametrize("batch_size", [16, 128])
    @pytest.mark.parametrize("hidden_size", [512, 1024])
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_auto_allocation_swizzled_nvfp4(self, batch_size, hidden_size, dtype):
        """Test auto-allocation with swizzled scale factor layout (NVFP4)."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_nvfp4quant

        torch.manual_seed(42)
        block_size = 16  # Fixed for NVFP4
        eps = 1e-6

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Call without providing y_fp4, block_scale, and global_scale, with swizzled layout
        y_fp4, block_scale, global_scale = rmsnorm_nvfp4quant(
            x, weight, eps=eps, is_sf_swizzled_layout=True
        )

        # Verify output shapes
        assert y_fp4.shape == (batch_size, hidden_size // 2)
        # Swizzled layout has different shape
        factor = block_size * 4
        num_m_tiles = (batch_size + 127) // 128
        num_k_tiles = (hidden_size + factor - 1) // factor
        expected_swizzled_size = num_m_tiles * num_k_tiles * 32 * 4 * 4
        assert block_scale.shape == (expected_swizzled_size,)
        assert global_scale.shape == (1,)

        # Verify output dtypes
        assert y_fp4.dtype == torch.uint8
        assert block_scale.dtype == torch.float8_e4m3fn
        assert global_scale.dtype == torch.float32

        # Unswizzle and compare with non-swizzled version
        y_fp4_ref = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale_ref = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        _, _, global_scale_ref = rmsnorm_nvfp4quant(
            x, weight, y_fp4_ref, block_scale_ref, eps=eps
        )

        # FP4 values should be identical
        torch.testing.assert_close(y_fp4, y_fp4_ref)

        # Unswizzle and compare scales
        block_scale_unswizzled = unswizzle_sf(
            block_scale.view(torch.uint8), batch_size, hidden_size, block_size
        ).view(torch.float8_e4m3fn)
        torch.testing.assert_close(
            block_scale_unswizzled.view(torch.uint8), block_scale_ref.view(torch.uint8)
        )

        # Global scales should be identical
        torch.testing.assert_close(global_scale, global_scale_ref)

    @pytest.mark.parametrize("batch_size", [16, 128])
    @pytest.mark.parametrize("hidden_size", [512, 1024])
    def test_auto_allocation_matches_preallocated_nvfp4(self, batch_size, hidden_size):
        """Test that auto-allocation produces same results as pre-allocated tensors (NVFP4)."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_nvfp4quant

        torch.manual_seed(42)
        block_size = 16  # Fixed for NVFP4
        eps = 1e-6
        dtype = torch.float16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Pre-allocated version
        y_fp4_pre = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale_pre = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        global_scale_pre = torch.empty(1, dtype=torch.float32, device="cuda")
        _, _, _ = rmsnorm_nvfp4quant(
            x, weight, y_fp4_pre, block_scale_pre, global_scale_pre, eps=eps
        )

        # Auto-allocated version
        y_fp4_auto, block_scale_auto, global_scale_auto = rmsnorm_nvfp4quant(
            x, weight, eps=eps
        )

        # Results should be identical
        torch.testing.assert_close(y_fp4_auto, y_fp4_pre)
        torch.testing.assert_close(
            block_scale_auto.view(torch.uint8), block_scale_pre.view(torch.uint8)
        )
        torch.testing.assert_close(global_scale_auto, global_scale_pre)

    @pytest.mark.parametrize("batch_size", [16, 128])
    @pytest.mark.parametrize("hidden_size", [512, 1024])
    def test_auto_allocation_matches_preallocated_mxfp4(self, batch_size, hidden_size):
        """Test that auto-allocation produces same results as pre-allocated tensors (MXFP4)."""
        from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_mxfp4quant

        torch.manual_seed(42)
        block_size = 32  # Fixed for MXFP4
        eps = 1e-6
        dtype = torch.float16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Pre-allocated version
        y_fp4_pre = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale_pre = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.uint8,
        )
        _, _ = rmsnorm_mxfp4quant(x, weight, y_fp4_pre, block_scale_pre, eps=eps)

        # Auto-allocated version
        y_fp4_auto, block_scale_auto = rmsnorm_mxfp4quant(x, weight, eps=eps)

        # Results should be identical
        torch.testing.assert_close(y_fp4_auto, y_fp4_pre)
        torch.testing.assert_close(block_scale_auto, block_scale_pre)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
