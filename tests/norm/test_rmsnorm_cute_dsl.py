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
Unit tests for RMSNorm using CuTe-DSL backend.

This tests the standalone RMSNorm implementation in flashinfer.cute_dsl.rmsnorm
which uses cluster-based reduction for large hidden dimensions on SM90+.
"""

import pytest
import torch

from flashinfer.cute_dsl.utils import is_cute_dsl_available


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


def llama_rms_norm_no_weight(x, eps=1e-6):
    """Reference RMSNorm implementation without weight."""
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype)
    return x


# Skip all tests if CuTe-DSL is not available
pytestmark = pytest.mark.skipif(
    not is_cute_dsl_available(),
    reason="CuTe-DSL is not available",
)


@pytest.fixture(autouse=True)
def skip_if_not_sm90():
    """Skip tests if CUDA compute capability < 90 (Hopper/Blackwell required)."""
    if get_cc() < 90:
        pytest.skip("CuTe-DSL RMSNorm requires SM90+ (Hopper/Blackwell)")


class TestRMSNormCuteDSL:
    """Test suite for CuTe-DSL RMSNorm implementation."""

    @pytest.mark.parametrize("batch_size", [1, 19, 99, 128])
    @pytest.mark.parametrize("hidden_size", [128, 1024, 4096, 8192])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rmsnorm_basic(self, batch_size, hidden_size, dtype):
        """Test basic RMSNorm functionality."""
        from flashinfer.cute_dsl import rmsnorm

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y = rmsnorm(x, w)
        y_ref = llama_rms_norm(x, w)

        torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("batch_size", [1, 32, 128])
    @pytest.mark.parametrize("hidden_size", [128, 1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rmsnorm_no_weight(self, batch_size, hidden_size, dtype):
        """Test RMSNorm without weight."""
        from flashinfer.cute_dsl import rmsnorm

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)

        y = rmsnorm(x, weight=None)
        y_ref = llama_rms_norm_no_weight(x)

        torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("batch_size", [1, 32])
    @pytest.mark.parametrize("hidden_size", [128, 1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rmsnorm_with_output(self, batch_size, hidden_size, dtype):
        """Test RMSNorm with pre-allocated output tensor."""
        from flashinfer.cute_dsl import rmsnorm

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)
        out = torch.empty_like(x)

        result = rmsnorm(x, w, output=out)
        y_ref = llama_rms_norm(x, w)

        # Verify output is same object
        assert result is out
        torch.testing.assert_close(out, y_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("batch_size", [2, 4])
    @pytest.mark.parametrize("seq_len", [16, 64])
    @pytest.mark.parametrize("hidden_size", [128, 1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rmsnorm_3d_input(self, batch_size, seq_len, hidden_size, dtype):
        """Test RMSNorm with 3D input (batch, seq_len, hidden)."""
        from flashinfer.cute_dsl import rmsnorm

        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y = rmsnorm(x, w)
        y_ref = llama_rms_norm(x, w)

        assert y.shape == x.shape
        torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("hidden_size", [16384, 32768])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rmsnorm_large_hidden(self, hidden_size, dtype):
        """Test RMSNorm with large hidden dimensions (triggers cluster reduction)."""
        from flashinfer.cute_dsl import rmsnorm

        batch_size = 32
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y = rmsnorm(x, w)
        y_ref = llama_rms_norm(x, w)

        torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-4])
    def test_rmsnorm_different_eps(self, eps):
        """Test RMSNorm with different epsilon values."""
        from flashinfer.cute_dsl import rmsnorm

        batch_size = 32
        hidden_size = 1024
        dtype = torch.float16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y = rmsnorm(x, w, eps=eps)
        y_ref = llama_rms_norm(x, w, eps=eps)

        torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rmsnorm_numerical_stability(self, dtype):
        """Test RMSNorm numerical stability with edge case inputs."""
        from flashinfer.cute_dsl import rmsnorm

        batch_size = 32
        hidden_size = 1024

        # Test with very small values
        x_small = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype) * 1e-4
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y_small = rmsnorm(x_small, w)
        y_small_ref = llama_rms_norm(x_small, w)
        torch.testing.assert_close(y_small, y_small_ref, rtol=1e-2, atol=1e-2)

        # Test with large values
        x_large = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype) * 100
        y_large = rmsnorm(x_large, w)
        y_large_ref = llama_rms_norm(x_large, w)
        torch.testing.assert_close(y_large, y_large_ref, rtol=1e-2, atol=1e-2)

    def test_rmsnorm_dtype_mismatch_error(self):
        """Test that dtype mismatch raises an error."""
        from flashinfer.cute_dsl import rmsnorm

        batch_size = 32
        hidden_size = 1024

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
        w = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)

        with pytest.raises(AssertionError, match="dtype"):
            rmsnorm(x, w)

    def test_rmsnorm_weight_shape_error(self):
        """Test that weight shape mismatch raises an error."""
        from flashinfer.cute_dsl import rmsnorm

        batch_size = 32
        hidden_size = 1024

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
        w = torch.randn(hidden_size // 2, device="cuda", dtype=torch.float16)

        with pytest.raises(AssertionError, match="shape"):
            rmsnorm(x, w)


class TestRMSNormCuteDSLCluster:
    """Test suite specifically for cluster reduction scenarios."""

    @pytest.mark.parametrize(
        "hidden_size,expected_cluster",
        [
            (8192, 1),      # N <= 16K: cluster_n = 1
            (16384, 1),     # N <= 16K: cluster_n = 1
            (20480, 2),     # 16K < N <= 32K: cluster_n = 2
            (32768, 2),     # 16K < N <= 32K: cluster_n = 2
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cluster_size_selection(self, hidden_size, expected_cluster, dtype):
        """Test that cluster size is selected correctly based on hidden dimension."""
        from flashinfer.cute_dsl.rmsnorm import RMSNormKernel

        kernel = RMSNormKernel(
            dtype={"float16": __import__("cutlass").Float16, "bfloat16": __import__("cutlass").BFloat16}[
                "float16" if dtype == torch.float16 else "bfloat16"
            ],
            N=hidden_size,
            has_weight=True,
        )

        assert kernel.cluster_n == expected_cluster, (
            f"Expected cluster_n={expected_cluster} for N={hidden_size}, "
            f"got cluster_n={kernel.cluster_n}"
        )

    @pytest.mark.parametrize("hidden_size", [16384, 32768, 65536])
    def test_rmsnorm_cluster_correctness(self, hidden_size):
        """Test correctness across different cluster configurations."""
        from flashinfer.cute_dsl import rmsnorm

        batch_size = 64
        dtype = torch.bfloat16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y = rmsnorm(x, w)
        y_ref = llama_rms_norm(x, w)

        # Use slightly looser tolerance for very large hidden dims
        torch.testing.assert_close(y, y_ref, rtol=2e-2, atol=2e-2)


class TestRMSNormCuteDSLIntegration:
    """Integration tests for RMSNorm CuTe-DSL."""

    def test_rmsnorm_multiple_calls(self):
        """Test that multiple calls with same parameters use cached kernel."""
        from flashinfer.cute_dsl import rmsnorm

        batch_size = 32
        hidden_size = 1024
        dtype = torch.float16

        x1 = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        x2 = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # First call compiles the kernel
        y1 = rmsnorm(x1, w)
        # Second call should use cached kernel
        y2 = rmsnorm(x2, w)

        y1_ref = llama_rms_norm(x1, w)
        y2_ref = llama_rms_norm(x2, w)

        torch.testing.assert_close(y1, y1_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(y2, y2_ref, rtol=1e-2, atol=1e-2)

    def test_rmsnorm_different_batch_sizes(self):
        """Test that RMSNorm works with different batch sizes (dynamic M)."""
        from flashinfer.cute_dsl import rmsnorm

        hidden_size = 1024
        dtype = torch.float16
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        for batch_size in [1, 16, 64, 128, 256]:
            x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
            y = rmsnorm(x, w)
            y_ref = llama_rms_norm(x, w)
            torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    def test_rmsnorm_contiguous_vs_noncontiguous(self):
        """Test RMSNorm with contiguous and non-contiguous inputs."""
        from flashinfer.cute_dsl import rmsnorm

        batch_size = 32
        hidden_size = 1024
        dtype = torch.float16

        # Contiguous input
        x_contig = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y_contig = rmsnorm(x_contig, w)
        y_ref = llama_rms_norm(x_contig, w)
        torch.testing.assert_close(y_contig, y_ref, rtol=1e-2, atol=1e-2)

        # Non-contiguous input (strided)
        x_noncontig = torch.randn(batch_size, hidden_size * 2, device="cuda", dtype=dtype)
        x_noncontig = x_noncontig[:, :hidden_size]
        assert not x_noncontig.is_contiguous()

        # The kernel should handle this by making it contiguous internally
        y_noncontig = rmsnorm(x_noncontig, w)
        y_ref_noncontig = llama_rms_norm(x_noncontig, w)
        torch.testing.assert_close(y_noncontig, y_ref_noncontig, rtol=1e-2, atol=1e-2)


class TestRMSNormCuteDSLPDL:
    """Test suite for Programmatic Dependent Launch (PDL) support."""

    @pytest.mark.parametrize("use_pdl", [True, False, None])
    @pytest.mark.parametrize("hidden_size", [1024, 4096])
    def test_pdl_correctness(self, use_pdl, hidden_size):
        """Test that PDL produces correct results."""
        from flashinfer.cute_dsl import rmsnorm

        batch_size = 32
        dtype = torch.float16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y = rmsnorm(x, w, use_pdl=use_pdl)
        y_ref = llama_rms_norm(x, w)

        torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    def test_pdl_with_and_without_produces_same_result(self):
        """Test that PDL and non-PDL produce identical results."""
        from flashinfer.cute_dsl import rmsnorm

        batch_size = 64
        hidden_size = 4096
        dtype = torch.float16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y_pdl = rmsnorm(x, w, use_pdl=True)
        y_no_pdl = rmsnorm(x, w, use_pdl=False)

        torch.testing.assert_close(y_pdl, y_no_pdl, rtol=0, atol=0)

    def test_pdl_multiple_consecutive_calls(self):
        """Test PDL with multiple consecutive kernel calls (the main use case)."""
        from flashinfer.cute_dsl import rmsnorm

        batch_size = 32
        hidden_size = 4096
        dtype = torch.float16

        # Create weights for two RMSNorm layers
        w1 = torch.randn(hidden_size, device="cuda", dtype=dtype)
        w2 = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Input
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)

        # Chain two RMSNorm calls with PDL enabled
        # This is the use case where PDL provides benefit
        y1 = rmsnorm(x, w1, use_pdl=True)
        y2 = rmsnorm(y1, w2, use_pdl=True)

        # Reference
        y1_ref = llama_rms_norm(x, w1)
        y2_ref = llama_rms_norm(y1_ref, w2)

        torch.testing.assert_close(y2, y2_ref, rtol=1e-2, atol=1e-2)

    def test_pdl_large_hidden_with_cluster(self):
        """Test PDL with large hidden dimensions that trigger cluster reduction."""
        from flashinfer.cute_dsl import rmsnorm

        batch_size = 32
        hidden_size = 32768  # Large enough to trigger cluster_n > 1
        dtype = torch.bfloat16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y = rmsnorm(x, w, use_pdl=True)
        y_ref = llama_rms_norm(x, w)

        torch.testing.assert_close(y, y_ref, rtol=2e-2, atol=2e-2)


class TestRMSNormUnifiedAPI:
    """Test the unified flashinfer.norm.rmsnorm API with backend parameter."""

    @pytest.mark.parametrize("batch_size", [1, 32, 128])
    @pytest.mark.parametrize("hidden_size", [128, 1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_unified_api_with_cute_dsl_backend(self, batch_size, hidden_size, dtype):
        """Test flashinfer.norm.rmsnorm with backend='cute-dsl'."""
        import flashinfer

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y = flashinfer.norm.rmsnorm(x, w, backend="cute-dsl")
        y_ref = llama_rms_norm(x, w)

        torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("batch_size", [1, 32])
    @pytest.mark.parametrize("hidden_size", [128, 1024])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_unified_api_with_output(self, batch_size, hidden_size, dtype):
        """Test flashinfer.norm.rmsnorm with pre-allocated output and cute-dsl backend."""
        import flashinfer

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)
        out = torch.empty_like(x)

        result = flashinfer.norm.rmsnorm(x, w, out=out, backend="cute-dsl")
        y_ref = llama_rms_norm(x, w)

        assert result is out
        torch.testing.assert_close(out, y_ref, rtol=1e-2, atol=1e-2)

    def test_unified_api_matches_direct_import(self):
        """Test that unified API produces same results as direct import."""
        import flashinfer
        from flashinfer.cute_dsl import rmsnorm as cute_dsl_rmsnorm

        batch_size = 32
        hidden_size = 1024
        dtype = torch.float16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Via unified API
        y_unified = flashinfer.norm.rmsnorm(x, w, backend="cute-dsl")
        # Via direct import
        y_direct = cute_dsl_rmsnorm(x, w)

        torch.testing.assert_close(y_unified, y_direct, rtol=0, atol=0)

    @pytest.mark.parametrize("enable_pdl", [True, False, None])
    def test_unified_api_with_pdl(self, enable_pdl):
        """Test that enable_pdl works correctly for cute-dsl backend."""
        import flashinfer

        batch_size = 32
        hidden_size = 1024
        dtype = torch.float16

        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y = flashinfer.norm.rmsnorm(x, w, enable_pdl=enable_pdl, backend="cute-dsl")
        y_ref = llama_rms_norm(x, w)

        torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    def test_unified_api_3d_input(self):
        """Test flashinfer.norm.rmsnorm with 3D input and cute-dsl backend."""
        import flashinfer

        batch_size = 2
        seq_len = 64
        hidden_size = 1024
        dtype = torch.float16

        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        w = torch.randn(hidden_size, device="cuda", dtype=dtype)

        y = flashinfer.norm.rmsnorm(x, w, backend="cute-dsl")
        y_ref = llama_rms_norm(x, w)

        assert y.shape == x.shape
        torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

