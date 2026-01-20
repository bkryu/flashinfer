"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for Fused Add + RMSNorm CuTe-DSL kernel.
"""

import pytest
import torch


def rmsnorm_reference(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float
) -> torch.Tensor:
    """Reference RMSNorm implementation in PyTorch."""
    # Compute RMS
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)

    # Apply weight if provided
    if weight is not None:
        x_normed = x_normed * weight

    return x_normed


def fused_add_rmsnorm_reference(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation for fused add + rmsnorm.

    Returns:
        (output, updated_residual) where:
        - updated_residual = residual + input
        - output = RMSNorm(updated_residual) * weight
    """
    updated_residual = residual + input
    output = rmsnorm_reference(updated_residual, weight, eps)
    return output, updated_residual


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestAddRMSNorm:
    """Tests for the fused Add + RMSNorm kernel."""

    @pytest.mark.parametrize("M", [1, 4, 32, 128])
    @pytest.mark.parametrize("N", [64, 128, 512, 1024, 2048, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_correctness_with_weight(self, cuda_device, M, N, dtype):
        """Test numerical correctness with weight tensor."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        torch.manual_seed(42)
        eps = 1e-6

        # Create input tensors
        input_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        residual_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        weight = torch.randn(N, device=cuda_device, dtype=dtype)

        # Reference computation (in FP32 for accuracy)
        output_ref, residual_ref = fused_add_rmsnorm_reference(
            input_orig.float(), residual_orig.float(), weight.float(), eps
        )
        output_ref = output_ref.to(dtype)
        residual_ref = residual_ref.to(dtype)

        # CuTe-DSL kernel (modifies in-place)
        input_test = input_orig.clone()
        residual_test = residual_orig.clone()
        add_rmsnorm(input_test, residual_test, weight, eps)

        # Check residual (updated_residual = residual + input)
        torch.testing.assert_close(
            residual_test,
            residual_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Residual mismatch for M={M}, N={N}, dtype={dtype}",
        )

        # Check output (normalized)
        torch.testing.assert_close(
            input_test,
            output_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Output mismatch for M={M}, N={N}, dtype={dtype}",
        )

    @pytest.mark.parametrize("M", [1, 32, 128])
    @pytest.mark.parametrize("N", [512, 2048, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_correctness_without_weight(self, cuda_device, M, N, dtype):
        """Test numerical correctness without weight tensor."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        torch.manual_seed(42)
        eps = 1e-6

        # Create input tensors
        input_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        residual_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)

        # Reference computation (in FP32 for accuracy)
        output_ref, residual_ref = fused_add_rmsnorm_reference(
            input_orig.float(), residual_orig.float(), None, eps
        )
        output_ref = output_ref.to(dtype)
        residual_ref = residual_ref.to(dtype)

        # CuTe-DSL kernel (modifies in-place)
        input_test = input_orig.clone()
        residual_test = residual_orig.clone()
        add_rmsnorm(input_test, residual_test, None, eps)

        # Check residual
        torch.testing.assert_close(
            residual_test,
            residual_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Residual mismatch for M={M}, N={N}, dtype={dtype}",
        )

        # Check output
        torch.testing.assert_close(
            input_test,
            output_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Output mismatch for M={M}, N={N}, dtype={dtype}",
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_3d_input(self, cuda_device, dtype):
        """Test with 3D input shape (batch, seq_len, hidden)."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        torch.manual_seed(42)
        B, S, N = 2, 64, 1024
        eps = 1e-6

        # Create input tensors
        input_orig = torch.randn(B, S, N, device=cuda_device, dtype=dtype)
        residual_orig = torch.randn(B, S, N, device=cuda_device, dtype=dtype)
        weight = torch.randn(N, device=cuda_device, dtype=dtype)

        # Reference computation
        input_flat = input_orig.view(B * S, N)
        residual_flat = residual_orig.view(B * S, N)
        output_ref, residual_ref = fused_add_rmsnorm_reference(
            input_flat.float(), residual_flat.float(), weight.float(), eps
        )
        output_ref = output_ref.view(B, S, N).to(dtype)
        residual_ref = residual_ref.view(B, S, N).to(dtype)

        # CuTe-DSL kernel
        input_test = input_orig.clone()
        residual_test = residual_orig.clone()
        add_rmsnorm(input_test, residual_test, weight, eps)

        # Check results
        torch.testing.assert_close(
            residual_test,
            residual_ref,
            atol=1e-2,
            rtol=1e-2,
            msg="Residual mismatch for 3D input",
        )
        torch.testing.assert_close(
            input_test,
            output_ref,
            atol=1e-2,
            rtol=1e-2,
            msg="Output mismatch for 3D input",
        )

    def test_fp32_dtype(self, cuda_device):
        """Test with float32 dtype."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        torch.manual_seed(42)
        M, N = 32, 1024
        eps = 1e-6
        dtype = torch.float32

        # Create input tensors
        input_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        residual_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        weight = torch.randn(N, device=cuda_device, dtype=dtype)

        # Reference computation
        output_ref, residual_ref = fused_add_rmsnorm_reference(
            input_orig, residual_orig, weight, eps
        )

        # CuTe-DSL kernel
        input_test = input_orig.clone()
        residual_test = residual_orig.clone()
        add_rmsnorm(input_test, residual_test, weight, eps)

        # Check results (tighter tolerance for FP32)
        torch.testing.assert_close(
            residual_test,
            residual_ref,
            atol=1e-5,
            rtol=1e-5,
            msg="Residual mismatch for FP32",
        )
        torch.testing.assert_close(
            input_test,
            output_ref,
            atol=1e-5,
            rtol=1e-5,
            msg="Output mismatch for FP32",
        )


class TestAddRMSNormEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.parametrize(
        "N", [8, 16, 32, 48, 96, 104, 1000]
    )  # Must be divisible by 8 for FP16/BF16
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_non_power_of_two_n(self, cuda_device, N, dtype):
        """Test with non-power-of-2 hidden dimensions (but aligned to vec_size)."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        torch.manual_seed(42)
        M = 32
        eps = 1e-6

        # Create input tensors
        input_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        residual_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        weight = torch.randn(N, device=cuda_device, dtype=dtype)

        # Reference computation
        output_ref, residual_ref = fused_add_rmsnorm_reference(
            input_orig.float(), residual_orig.float(), weight.float(), eps
        )
        output_ref = output_ref.to(dtype)
        residual_ref = residual_ref.to(dtype)

        # CuTe-DSL kernel
        input_test = input_orig.clone()
        residual_test = residual_orig.clone()
        add_rmsnorm(input_test, residual_test, weight, eps)

        # Check results
        torch.testing.assert_close(
            residual_test,
            residual_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Residual mismatch for N={N}",
        )
        torch.testing.assert_close(
            input_test,
            output_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Output mismatch for N={N}",
        )

    @pytest.mark.parametrize("M", [1, 2, 3, 7, 15, 33, 127, 1024, 4096])
    def test_various_m_sizes(self, cuda_device, M):
        """Test with various batch sizes including non-power-of-2."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        torch.manual_seed(42)
        N = 1024
        dtype = torch.float16
        eps = 1e-6

        # Create input tensors
        input_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        residual_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        weight = torch.randn(N, device=cuda_device, dtype=dtype)

        # Reference computation
        output_ref, residual_ref = fused_add_rmsnorm_reference(
            input_orig.float(), residual_orig.float(), weight.float(), eps
        )
        output_ref = output_ref.to(dtype)
        residual_ref = residual_ref.to(dtype)

        # CuTe-DSL kernel
        input_test = input_orig.clone()
        residual_test = residual_orig.clone()
        add_rmsnorm(input_test, residual_test, weight, eps)

        # Check results
        torch.testing.assert_close(
            residual_test,
            residual_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Residual mismatch for M={M}",
        )
        torch.testing.assert_close(
            input_test,
            output_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Output mismatch for M={M}",
        )

    def test_minimum_n_error(self, cuda_device):
        """Test that N < 8 raises ValueError."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        M, N = 32, 4  # N=4 is below minimum
        dtype = torch.float16

        input_tensor = torch.randn(M, N, device=cuda_device, dtype=dtype)
        residual = torch.randn(M, N, device=cuda_device, dtype=dtype)

        with pytest.raises(ValueError, match="too small"):
            add_rmsnorm(input_tensor, residual)

    def test_alignment_error(self, cuda_device):
        """Test that misaligned N raises ValueError."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        M = 32

        # N=100 is not divisible by 8 (vec_size for FP16/BF16)
        N_misaligned_fp16 = 100
        input_fp16 = torch.randn(
            M, N_misaligned_fp16, device=cuda_device, dtype=torch.float16
        )
        residual_fp16 = torch.randn(
            M, N_misaligned_fp16, device=cuda_device, dtype=torch.float16
        )

        with pytest.raises(ValueError, match="divisible by"):
            add_rmsnorm(input_fp16, residual_fp16)

        # N=102 is not divisible by 4 (vec_size for FP32)
        # Note: N=100 IS divisible by 4, so use 102 instead
        N_misaligned_fp32 = 102
        input_fp32 = torch.randn(
            M, N_misaligned_fp32, device=cuda_device, dtype=torch.float32
        )
        residual_fp32 = torch.randn(
            M, N_misaligned_fp32, device=cuda_device, dtype=torch.float32
        )

        with pytest.raises(ValueError, match="divisible by"):
            add_rmsnorm(input_fp32, residual_fp32)

    def test_non_contiguous_input_error(self, cuda_device):
        """Test that non-contiguous input raises ValueError."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        M, N = 32, 1024
        dtype = torch.float16

        # Create non-contiguous tensor by transposing
        input_tensor = torch.randn(N, M, device=cuda_device, dtype=dtype).T
        residual = torch.randn(M, N, device=cuda_device, dtype=dtype)

        assert not input_tensor.is_contiguous()

        with pytest.raises(ValueError, match="contiguous"):
            add_rmsnorm(input_tensor, residual)

    def test_shape_mismatch_error(self, cuda_device):
        """Test that shape mismatch raises ValueError."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        dtype = torch.float16

        input_tensor = torch.randn(32, 1024, device=cuda_device, dtype=dtype)
        residual = torch.randn(64, 1024, device=cuda_device, dtype=dtype)  # Different M

        with pytest.raises(ValueError, match="shape"):
            add_rmsnorm(input_tensor, residual)

    def test_dtype_mismatch_error(self, cuda_device):
        """Test that dtype mismatch raises TypeError."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        M, N = 32, 1024

        input_tensor = torch.randn(M, N, device=cuda_device, dtype=torch.float16)
        residual = torch.randn(M, N, device=cuda_device, dtype=torch.bfloat16)

        with pytest.raises(TypeError, match="dtype"):
            add_rmsnorm(input_tensor, residual)

    def test_invalid_dtype_error(self, cuda_device):
        """Test that unsupported dtype raises TypeError."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        M, N = 32, 1024

        input_tensor = torch.randn(M, N, device=cuda_device, dtype=torch.float64)
        residual = torch.randn(M, N, device=cuda_device, dtype=torch.float64)

        with pytest.raises(TypeError, match="float16, bfloat16, or float32"):
            add_rmsnorm(input_tensor, residual)

    def test_1d_input_error(self, cuda_device):
        """Test that 1D input raises ValueError."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        input_tensor = torch.randn(1024, device=cuda_device, dtype=torch.float16)
        residual = torch.randn(1024, device=cuda_device, dtype=torch.float16)

        with pytest.raises(ValueError, match="2D.*3D"):
            add_rmsnorm(input_tensor, residual)

    def test_weight_shape_mismatch_error(self, cuda_device):
        """Test that weight shape mismatch raises ValueError."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        M, N = 32, 1024
        dtype = torch.float16

        input_tensor = torch.randn(M, N, device=cuda_device, dtype=dtype)
        residual = torch.randn(M, N, device=cuda_device, dtype=dtype)
        weight = torch.randn(512, device=cuda_device, dtype=dtype)  # Wrong size

        with pytest.raises(ValueError, match="Weight.*shape"):
            add_rmsnorm(input_tensor, residual, weight)


class TestAddRMSNormCluster:
    """Tests for cluster support (SM90+, large N)."""

    @pytest.mark.parametrize("M", [4, 32])
    @pytest.mark.parametrize(
        "N", [16384, 20480, 32768]
    )  # Large N triggers cluster mode
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_n_cluster(self, cuda_device, M, N, dtype):
        """Test numerical correctness with large N (cluster mode on SM90+)."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        torch.manual_seed(42)
        eps = 1e-6

        # Create input tensors
        input_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        residual_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        weight = torch.randn(N, device=cuda_device, dtype=dtype)

        # Reference computation (in FP32 for accuracy)
        output_ref, residual_ref = fused_add_rmsnorm_reference(
            input_orig.float(), residual_orig.float(), weight.float(), eps
        )
        output_ref = output_ref.to(dtype)
        residual_ref = residual_ref.to(dtype)

        # CuTe-DSL kernel (modifies in-place)
        input_test = input_orig.clone()
        residual_test = residual_orig.clone()
        add_rmsnorm(input_test, residual_test, weight, eps)

        # Check residual
        torch.testing.assert_close(
            residual_test,
            residual_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Residual mismatch for cluster mode M={M}, N={N}, dtype={dtype}",
        )

        # Check output
        torch.testing.assert_close(
            input_test,
            output_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Output mismatch for cluster mode M={M}, N={N}, dtype={dtype}",
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_pdl_parameter(self, cuda_device, dtype):
        """Test that enable_pdl parameter works without errors."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
        except ImportError:
            pytest.skip("CuTe-DSL not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        torch.manual_seed(42)
        M, N = 32, 2048
        eps = 1e-6

        # Create input tensors
        input_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        residual_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        weight = torch.randn(N, device=cuda_device, dtype=dtype)

        # Reference computation
        output_ref, residual_ref = fused_add_rmsnorm_reference(
            input_orig.float(), residual_orig.float(), weight.float(), eps
        )
        output_ref = output_ref.to(dtype)
        residual_ref = residual_ref.to(dtype)

        # Test with enable_pdl=False
        input_test = input_orig.clone()
        residual_test = residual_orig.clone()
        add_rmsnorm(input_test, residual_test, weight, eps, enable_pdl=False)

        torch.testing.assert_close(
            residual_test,
            residual_ref,
            atol=1e-2,
            rtol=1e-2,
            msg="Residual mismatch with enable_pdl=False",
        )
        torch.testing.assert_close(
            input_test,
            output_ref,
            atol=1e-2,
            rtol=1e-2,
            msg="Output mismatch with enable_pdl=False",
        )

        # Test with enable_pdl=True (only meaningful on SM90+, but should work on any GPU)
        input_test2 = input_orig.clone()
        residual_test2 = residual_orig.clone()
        add_rmsnorm(input_test2, residual_test2, weight, eps, enable_pdl=True)

        torch.testing.assert_close(
            residual_test2,
            residual_ref,
            atol=1e-2,
            rtol=1e-2,
            msg="Residual mismatch with enable_pdl=True",
        )
        torch.testing.assert_close(
            input_test2,
            output_ref,
            atol=1e-2,
            rtol=1e-2,
            msg="Output mismatch with enable_pdl=True",
        )


class TestAddRMSNormComparison:
    """Compare CuTe-DSL add_rmsnorm against existing FlashInfer fused_add_rmsnorm."""

    @pytest.mark.parametrize("M", [32, 128])
    @pytest.mark.parametrize("N", [1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_against_flashinfer_cuda(self, cuda_device, M, N, dtype):
        """Compare against FlashInfer's CUDA fused_add_rmsnorm."""
        try:
            from flashinfer.cute_dsl import add_rmsnorm, is_cute_dsl_available
            from flashinfer import fused_add_rmsnorm as flashinfer_fused_add_rmsnorm
        except ImportError:
            pytest.skip("Required modules not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        torch.manual_seed(42)
        eps = 1e-6

        # Create input tensors
        input_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        residual_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        weight = torch.randn(N, device=cuda_device, dtype=dtype)

        # FlashInfer CUDA kernel
        input_fi = input_orig.clone()
        residual_fi = residual_orig.clone()
        flashinfer_fused_add_rmsnorm(input_fi, residual_fi, weight, eps)

        # CuTe-DSL kernel
        input_cute = input_orig.clone()
        residual_cute = residual_orig.clone()
        add_rmsnorm(input_cute, residual_cute, weight, eps)

        # Compare results
        torch.testing.assert_close(
            residual_cute,
            residual_fi,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Residual mismatch vs FlashInfer CUDA for M={M}, N={N}, dtype={dtype}",
        )
        torch.testing.assert_close(
            input_cute,
            input_fi,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Output mismatch vs FlashInfer CUDA for M={M}, N={N}, dtype={dtype}",
        )


class TestFusedAddRmsnormUnifiedAPI:
    """Test the unified fused_add_rmsnorm API with backend='cute-dsl'."""

    @pytest.mark.parametrize("M", [32, 128])
    @pytest.mark.parametrize("N", [1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_unified_api_cute_dsl_backend(self, cuda_device, M, N, dtype):
        """Test flashinfer.fused_add_rmsnorm with backend='cute-dsl'."""
        try:
            from flashinfer import fused_add_rmsnorm
            from flashinfer.cute_dsl import is_cute_dsl_available
        except ImportError:
            pytest.skip("Required modules not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        torch.manual_seed(42)
        eps = 1e-6

        # Create input tensors
        input_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        residual_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        weight = torch.randn(N, device=cuda_device, dtype=dtype)

        # Reference computation
        output_ref, residual_ref = fused_add_rmsnorm_reference(
            input_orig.float(), residual_orig.float(), weight.float(), eps
        )
        output_ref = output_ref.to(dtype)
        residual_ref = residual_ref.to(dtype)

        # Test with backend='cute-dsl'
        input_test = input_orig.clone()
        residual_test = residual_orig.clone()
        fused_add_rmsnorm(input_test, residual_test, weight, eps, backend="cute-dsl")

        # Check results
        torch.testing.assert_close(
            residual_test,
            residual_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Residual mismatch for unified API M={M}, N={N}, dtype={dtype}",
        )
        torch.testing.assert_close(
            input_test,
            output_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Output mismatch for unified API M={M}, N={N}, dtype={dtype}",
        )

    @pytest.mark.parametrize("M", [32, 128])
    @pytest.mark.parametrize("N", [1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_unified_api_cuda_vs_cute_dsl(self, cuda_device, M, N, dtype):
        """Compare unified API with backend='cuda' vs backend='cute-dsl'."""
        try:
            from flashinfer import fused_add_rmsnorm
            from flashinfer.cute_dsl import is_cute_dsl_available
        except ImportError:
            pytest.skip("Required modules not available")

        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        torch.manual_seed(42)
        eps = 1e-6

        # Create input tensors
        input_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        residual_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        weight = torch.randn(N, device=cuda_device, dtype=dtype)

        # Test with backend='cuda'
        input_cuda = input_orig.clone()
        residual_cuda = residual_orig.clone()
        fused_add_rmsnorm(input_cuda, residual_cuda, weight, eps, backend="cuda")

        # Test with backend='cute-dsl'
        input_cute = input_orig.clone()
        residual_cute = residual_orig.clone()
        fused_add_rmsnorm(input_cute, residual_cute, weight, eps, backend="cute-dsl")

        # Compare results between backends
        torch.testing.assert_close(
            residual_cute,
            residual_cuda,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Residual mismatch cuda vs cute-dsl for M={M}, N={N}, dtype={dtype}",
        )
        torch.testing.assert_close(
            input_cute,
            input_cuda,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Output mismatch cuda vs cute-dsl for M={M}, N={N}, dtype={dtype}",
        )

    def test_unified_api_invalid_backend(self, cuda_device):
        """Test that invalid backend raises ValueError."""
        try:
            from flashinfer import fused_add_rmsnorm
        except ImportError:
            pytest.skip("Required modules not available")

        M, N = 32, 1024
        dtype = torch.float16
        eps = 1e-6

        input_tensor = torch.randn(M, N, device=cuda_device, dtype=dtype)
        residual = torch.randn(M, N, device=cuda_device, dtype=dtype)
        weight = torch.randn(N, device=cuda_device, dtype=dtype)

        with pytest.raises(ValueError, match="Unknown backend"):
            fused_add_rmsnorm(input_tensor, residual, weight, eps, backend="invalid")

    def test_unified_api_default_backend(self, cuda_device):
        """Test that default backend (no backend arg) uses CUDA and is backwards compatible."""
        try:
            from flashinfer import fused_add_rmsnorm
        except ImportError:
            pytest.skip("Required modules not available")

        torch.manual_seed(42)
        M, N = 32, 1024
        dtype = torch.float16
        eps = 1e-6

        # Create input tensors
        input_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        residual_orig = torch.randn(M, N, device=cuda_device, dtype=dtype)
        weight = torch.randn(N, device=cuda_device, dtype=dtype)

        # Reference computation
        output_ref, residual_ref = fused_add_rmsnorm_reference(
            input_orig.float(), residual_orig.float(), weight.float(), eps
        )
        output_ref = output_ref.to(dtype)
        residual_ref = residual_ref.to(dtype)

        # Test with no backend argument (should default to CUDA)
        input_default = input_orig.clone()
        residual_default = residual_orig.clone()
        fused_add_rmsnorm(input_default, residual_default, weight, eps)

        # Test with explicit backend='cuda'
        input_cuda = input_orig.clone()
        residual_cuda = residual_orig.clone()
        fused_add_rmsnorm(input_cuda, residual_cuda, weight, eps, backend="cuda")

        # Results should be identical
        torch.testing.assert_close(
            input_default,
            input_cuda,
            atol=0,
            rtol=0,
            msg="Default backend should be identical to explicit backend='cuda'",
        )
        torch.testing.assert_close(
            residual_default,
            residual_cuda,
            atol=0,
            rtol=0,
            msg="Default backend residual should be identical to explicit backend='cuda'",
        )

        # Both should match reference
        torch.testing.assert_close(
            input_default,
            output_ref,
            atol=1e-2,
            rtol=1e-2,
            msg="Default backend output mismatch with reference",
        )

    @pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
    def test_unified_api_3d_input_parity(self, cuda_device, backend):
        """Test that 3D input works with both backends (API parity)."""
        try:
            from flashinfer import fused_add_rmsnorm
            from flashinfer.cute_dsl import is_cute_dsl_available
        except ImportError:
            pytest.skip("Required modules not available")

        if backend == "cute-dsl" and not is_cute_dsl_available():
            pytest.skip("CuTe-DSL not available")

        torch.manual_seed(42)
        B, S, N = 2, 64, 1024
        dtype = torch.float16
        eps = 1e-6

        # Create 3D input tensors
        input_orig = torch.randn(B, S, N, device=cuda_device, dtype=dtype)
        residual_orig = torch.randn(B, S, N, device=cuda_device, dtype=dtype)
        weight = torch.randn(N, device=cuda_device, dtype=dtype)

        # Reference computation (flatten to 2D for reference)
        output_ref, residual_ref = fused_add_rmsnorm_reference(
            input_orig.view(B * S, N).float(),
            residual_orig.view(B * S, N).float(),
            weight.float(),
            eps,
        )
        output_ref = output_ref.view(B, S, N).to(dtype)
        residual_ref = residual_ref.view(B, S, N).to(dtype)

        # Test with 3D input
        input_test = input_orig.clone()
        residual_test = residual_orig.clone()
        fused_add_rmsnorm(input_test, residual_test, weight, eps, backend=backend)

        # Verify shape is preserved
        assert input_test.shape == (B, S, N), f"Shape changed: {input_test.shape}"
        assert residual_test.shape == (B, S, N), (
            f"Residual shape changed: {residual_test.shape}"
        )

        # Check results
        torch.testing.assert_close(
            residual_test,
            residual_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Residual mismatch for 3D input with backend={backend}",
        )
        torch.testing.assert_close(
            input_test,
            output_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Output mismatch for 3D input with backend={backend}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
