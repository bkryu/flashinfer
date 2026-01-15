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
Unit tests for Bitonic Top-K + Softmax using CuTe-DSL backend.

These tests verify the bitonic_topk_softmax kernel against PyTorch's
torch.topk and torch.softmax reference implementations.
"""

import pytest
import torch

from flashinfer.cute_dsl.utils import is_cute_dsl_available


def get_cc():
    """Get CUDA compute capability."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def requires_cute_dsl():
    """Check if CuTe-DSL is available."""
    return is_cute_dsl_available()


def requires_hopper_or_later():
    """Check if running on Hopper (SM90+) or later GPU."""
    return get_cc() >= 90


# Skip conditions
cute_dsl_available = pytest.mark.skipif(
    not requires_cute_dsl(), reason="CuTe-DSL not available"
)

hopper_required = pytest.mark.skipif(
    not requires_hopper_or_later(),
    reason="CuTe-DSL kernel requires Hopper (SM90+) or later GPU",
)


def reference_topk(x: torch.Tensor, k: int, largest: bool = True, sorted: bool = True):
    """Reference top-k using PyTorch."""
    return torch.topk(x, k, dim=-1, largest=largest, sorted=sorted)


def reference_topk_softmax(
    x: torch.Tensor, k: int, largest: bool = True, sorted: bool = True
):
    """Reference top-k with softmax using PyTorch."""
    values, indices = torch.topk(x, k, dim=-1, largest=largest, sorted=sorted)
    # Apply softmax to the selected values
    probs = torch.softmax(values.float(), dim=-1).to(x.dtype)
    return probs, indices


@cute_dsl_available
@hopper_required
class TestBitonicTopKBasic:
    """Basic tests for bitonic top-k selection."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 32, 128])
    @pytest.mark.parametrize("N", [64, 128, 256, 512, 1024, 2048, 4096])
    @pytest.mark.parametrize("k", [1, 2, 4, 8, 16, 32, 64])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_topk_values_and_indices(self, batch_size, N, k, dtype):
        """Test top-k selection returns correct values and indices."""
        if k > N:
            pytest.skip(f"k={k} > N={N}, skipping")

        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)

        x = torch.randn(batch_size, N, device="cuda", dtype=dtype)

        # Run CuTe-DSL kernel
        values, indices = bitonic_topk_softmax(x, k, softmax=False, largest=True)

        # Reference
        ref_values, ref_indices = reference_topk(x, k, largest=True)

        # Verify shapes
        assert values.shape == (batch_size, k)
        assert indices.shape == (batch_size, k)

        # Verify dtypes
        assert values.dtype == dtype
        assert indices.dtype == torch.int32

        # Verify values match (with tolerance for float precision)
        torch.testing.assert_close(
            values, ref_values, rtol=1e-3 if dtype == torch.float32 else 1e-2, atol=1e-3
        )

        # Verify we selected the same elements (indices may differ if ties exist)
        # Check by gathering values from original tensor
        gathered = torch.gather(x, 1, indices.long())
        ref_gathered = torch.gather(x, 1, ref_indices.long())
        torch.testing.assert_close(
            gathered, ref_gathered, rtol=1e-3 if dtype == torch.float32 else 1e-2, atol=1e-3
        )

    @pytest.mark.parametrize("batch_size", [1, 16, 64])
    @pytest.mark.parametrize("N", [128, 512, 1024])
    @pytest.mark.parametrize("k", [2, 8, 32])
    def test_topk_smallest(self, batch_size, N, k):
        """Test bottom-k (smallest) selection."""
        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)
        dtype = torch.float32

        x = torch.randn(batch_size, N, device="cuda", dtype=dtype)

        # Run CuTe-DSL kernel with largest=False
        values, indices = bitonic_topk_softmax(x, k, softmax=False, largest=False)

        # Reference
        ref_values, ref_indices = reference_topk(x, k, largest=False)

        # Verify values match
        torch.testing.assert_close(values, ref_values, rtol=1e-3, atol=1e-3)

        # Verify gathered values match
        gathered = torch.gather(x, 1, indices.long())
        ref_gathered = torch.gather(x, 1, ref_indices.long())
        torch.testing.assert_close(gathered, ref_gathered, rtol=1e-3, atol=1e-3)


@cute_dsl_available
@hopper_required
class TestBitonicTopKSoftmax:
    """Tests for top-k with fused softmax."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
    @pytest.mark.parametrize("N", [128, 256, 512, 1024])
    @pytest.mark.parametrize("k", [2, 4, 8, 16, 32])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_topk_softmax_probs(self, batch_size, N, k, dtype):
        """Test that softmax probabilities sum to 1."""
        if k > N:
            pytest.skip(f"k={k} > N={N}, skipping")

        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)

        x = torch.randn(batch_size, N, device="cuda", dtype=dtype)

        # Run CuTe-DSL kernel with softmax
        probs, indices = bitonic_topk_softmax(x, k, softmax=True, largest=True)

        # Verify shapes
        assert probs.shape == (batch_size, k)
        assert indices.shape == (batch_size, k)

        # Verify probabilities sum to 1
        prob_sums = probs.sum(dim=-1)
        torch.testing.assert_close(
            prob_sums,
            torch.ones(batch_size, device="cuda", dtype=dtype),
            rtol=1e-2,
            atol=1e-2,
        )

        # Verify probabilities are non-negative
        assert (probs >= 0).all(), "Probabilities should be non-negative"

        # Verify probabilities are <= 1
        assert (probs <= 1).all(), "Probabilities should be <= 1"

    @pytest.mark.parametrize("batch_size", [1, 16, 64])
    @pytest.mark.parametrize("N", [256, 512, 1024])
    @pytest.mark.parametrize("k", [4, 8, 16])
    def test_topk_softmax_values(self, batch_size, N, k):
        """Test that softmax values match reference."""
        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)
        dtype = torch.float32

        x = torch.randn(batch_size, N, device="cuda", dtype=dtype)

        # Run CuTe-DSL kernel with softmax
        probs, indices = bitonic_topk_softmax(x, k, softmax=True, largest=True)

        # Reference: top-k then softmax
        ref_probs, ref_indices = reference_topk_softmax(x, k, largest=True)

        # Verify probabilities match (with tolerance for fast-math approximations)
        # The kernel uses exp2 + rcp_approx which introduces small errors
        torch.testing.assert_close(probs, ref_probs, rtol=0.05, atol=0.01)

    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("N", [256, 512])
    @pytest.mark.parametrize("k", [4, 8])
    def test_topk_softmax_smallest(self, batch_size, N, k):
        """Test bottom-k with softmax."""
        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)
        dtype = torch.float32

        x = torch.randn(batch_size, N, device="cuda", dtype=dtype)

        # Run CuTe-DSL kernel with largest=False and softmax
        probs, indices = bitonic_topk_softmax(
            x, k, softmax=True, largest=False
        )

        # Verify probabilities sum to 1
        prob_sums = probs.sum(dim=-1)
        torch.testing.assert_close(
            prob_sums,
            torch.ones(batch_size, device="cuda", dtype=dtype),
            rtol=1e-2,
            atol=1e-2,
        )

        # Reference: bottom-k then softmax on negated values
        # (softmax should still work on the smallest values)
        ref_probs, ref_indices = reference_topk_softmax(x, k, largest=False)

        # Verify probabilities match
        torch.testing.assert_close(probs, ref_probs, rtol=0.05, atol=0.01)


@cute_dsl_available
@hopper_required
class TestBitonicTopKEdgeCases:
    """Edge cases and boundary condition tests."""

    def test_k_equals_n(self):
        """Test when k equals N (select all elements)."""
        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)
        batch_size = 4
        N = 64  # Power of 2 for simplicity
        k = 64

        x = torch.randn(batch_size, N, device="cuda", dtype=torch.float32)

        values, indices = bitonic_topk_softmax(x, k, softmax=False, largest=True)

        assert values.shape == (batch_size, k)
        assert indices.shape == (batch_size, k)

        # All elements should be selected
        # Check that the values contain all elements (sorted)
        ref_values, _ = torch.sort(x, dim=-1, descending=True)
        torch.testing.assert_close(values, ref_values, rtol=1e-3, atol=1e-3)

    def test_k_equals_1(self):
        """Test when k equals 1 (select max/min only)."""
        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)
        batch_size = 16
        N = 256
        k = 1

        x = torch.randn(batch_size, N, device="cuda", dtype=torch.float32)

        # Test largest
        values, indices = bitonic_topk_softmax(x, k, softmax=False, largest=True)

        assert values.shape == (batch_size, k)
        ref_max = x.max(dim=-1, keepdim=True).values
        torch.testing.assert_close(values, ref_max, rtol=1e-3, atol=1e-3)

        # Test smallest
        values, indices = bitonic_topk_softmax(x, k, softmax=False, largest=False)
        ref_min = x.min(dim=-1, keepdim=True).values
        torch.testing.assert_close(values, ref_min, rtol=1e-3, atol=1e-3)

    def test_1d_input(self):
        """Test with 1D input tensor."""
        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)
        N = 256
        k = 8

        x = torch.randn(N, device="cuda", dtype=torch.float32)

        values, indices = bitonic_topk_softmax(x, k, softmax=False, largest=True)

        # Should return 1D output
        assert values.shape == (k,)
        assert indices.shape == (k,)

        # Reference
        ref_values, ref_indices = torch.topk(x, k, largest=True)
        torch.testing.assert_close(values, ref_values, rtol=1e-3, atol=1e-3)

    def test_3d_input(self):
        """Test with 3D input tensor (batch, seq, dim)."""
        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)
        batch = 2
        seq = 4
        N = 256
        k = 8

        x = torch.randn(batch, seq, N, device="cuda", dtype=torch.float32)

        values, indices = bitonic_topk_softmax(x, k, softmax=False, largest=True)

        # Should return 3D output
        assert values.shape == (batch, seq, k)
        assert indices.shape == (batch, seq, k)

        # Reference: reshape and compare
        x_flat = x.reshape(-1, N)
        ref_values, _ = torch.topk(x_flat, k, largest=True)
        ref_values = ref_values.reshape(batch, seq, k)
        torch.testing.assert_close(values, ref_values, rtol=1e-3, atol=1e-3)

    def test_non_power_of_2_n(self):
        """Test with N that is not a power of 2."""
        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)
        batch_size = 8
        N = 300  # Not a power of 2
        k = 8

        x = torch.randn(batch_size, N, device="cuda", dtype=torch.float32)

        values, indices = bitonic_topk_softmax(x, k, softmax=False, largest=True)

        # Verify shapes
        assert values.shape == (batch_size, k)
        assert indices.shape == (batch_size, k)

        # Verify values match reference
        ref_values, ref_indices = torch.topk(x, k, largest=True)
        torch.testing.assert_close(values, ref_values, rtol=1e-3, atol=1e-3)

        # Verify indices are valid (within bounds)
        assert (indices >= 0).all()
        assert (indices < N).all()

    def test_non_power_of_2_k(self):
        """Test with k that is not a power of 2."""
        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)
        batch_size = 8
        N = 512
        k = 10  # Not a power of 2

        x = torch.randn(batch_size, N, device="cuda", dtype=torch.float32)

        values, indices = bitonic_topk_softmax(x, k, softmax=False, largest=True)

        # Verify shapes
        assert values.shape == (batch_size, k)
        assert indices.shape == (batch_size, k)

        # Verify values match reference
        ref_values, ref_indices = torch.topk(x, k, largest=True)
        torch.testing.assert_close(values, ref_values, rtol=1e-3, atol=1e-3)


@cute_dsl_available
@hopper_required
class TestBitonicTopKSorted:
    """Tests for sorted vs unsorted output."""

    @pytest.mark.parametrize("batch_size", [4, 16])
    @pytest.mark.parametrize("N", [256, 512])
    @pytest.mark.parametrize("k", [8, 16])
    def test_sorted_output(self, batch_size, N, k):
        """Test that sorted=True returns sorted values."""
        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)

        x = torch.randn(batch_size, N, device="cuda", dtype=torch.float32)

        values, indices = bitonic_topk_softmax(
            x, k, softmax=False, largest=True, sorted=True
        )

        # Verify values are sorted (descending for largest=True)
        for i in range(batch_size):
            for j in range(k - 1):
                assert values[i, j] >= values[i, j + 1], (
                    f"Values not sorted at batch {i}, position {j}"
                )

    @pytest.mark.parametrize("batch_size", [4, 16])
    @pytest.mark.parametrize("N", [256, 512])
    @pytest.mark.parametrize("k", [8, 16])
    def test_sorted_smallest(self, batch_size, N, k):
        """Test that sorted=True with largest=False returns ascending sorted values."""
        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)

        x = torch.randn(batch_size, N, device="cuda", dtype=torch.float32)

        values, indices = bitonic_topk_softmax(
            x, k, softmax=False, largest=False, sorted=True
        )

        # Verify values are sorted (ascending for largest=False)
        for i in range(batch_size):
            for j in range(k - 1):
                assert values[i, j] <= values[i, j + 1], (
                    f"Values not sorted at batch {i}, position {j}"
                )


@cute_dsl_available
@hopper_required
class TestBitonicTopKNumericalStability:
    """Tests for numerical stability."""

    def test_large_values(self):
        """Test with large input values."""
        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)
        batch_size = 8
        N = 256
        k = 8

        # Create tensor with large values
        x = torch.randn(batch_size, N, device="cuda", dtype=torch.float32) * 1000

        values, indices = bitonic_topk_softmax(x, k, softmax=False, largest=True)

        ref_values, _ = torch.topk(x, k, largest=True)
        torch.testing.assert_close(values, ref_values, rtol=1e-3, atol=1.0)

    def test_small_values(self):
        """Test with small input values."""
        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)
        batch_size = 8
        N = 256
        k = 8

        # Create tensor with small values
        x = torch.randn(batch_size, N, device="cuda", dtype=torch.float32) * 1e-5

        values, indices = bitonic_topk_softmax(x, k, softmax=False, largest=True)

        ref_values, _ = torch.topk(x, k, largest=True)
        torch.testing.assert_close(values, ref_values, rtol=1e-2, atol=1e-6)

    def test_softmax_large_values(self):
        """Test softmax stability with large input values."""
        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)
        batch_size = 8
        N = 256
        k = 8

        # Create tensor with large values that could cause overflow
        x = torch.randn(batch_size, N, device="cuda", dtype=torch.float32) * 100

        probs, indices = bitonic_topk_softmax(x, k, softmax=True, largest=True)

        # Verify no NaN or Inf
        assert not torch.isnan(probs).any(), "Softmax produced NaN"
        assert not torch.isinf(probs).any(), "Softmax produced Inf"

        # Verify probabilities still sum to 1
        prob_sums = probs.sum(dim=-1)
        torch.testing.assert_close(
            prob_sums,
            torch.ones(batch_size, device="cuda", dtype=torch.float32),
            rtol=0.05,
            atol=0.05,
        )


@cute_dsl_available
@hopper_required
class TestBitonicTopKMoEUseCase:
    """Tests simulating MoE (Mixture of Experts) routing use case."""

    @pytest.mark.parametrize("num_tokens", [64, 256, 1024, 4096])
    @pytest.mark.parametrize("num_experts", [8, 16, 32, 64])
    @pytest.mark.parametrize("top_k_experts", [2, 4, 8])
    def test_moe_routing(self, num_tokens, num_experts, top_k_experts):
        """Test MoE routing: select top-k experts per token with softmax."""
        if top_k_experts > num_experts:
            pytest.skip(f"top_k={top_k_experts} > num_experts={num_experts}")

        from flashinfer.cute_dsl import bitonic_topk_softmax

        torch.manual_seed(42)

        # Router logits: (num_tokens, num_experts)
        router_logits = torch.randn(
            num_tokens, num_experts, device="cuda", dtype=torch.float32
        )

        # Run top-k softmax (this is what MoE routing does)
        expert_probs, expert_indices = bitonic_topk_softmax(
            router_logits, top_k_experts, softmax=True, largest=True
        )

        # Verify shapes
        assert expert_probs.shape == (num_tokens, top_k_experts)
        assert expert_indices.shape == (num_tokens, top_k_experts)

        # Verify indices are valid expert IDs
        assert (expert_indices >= 0).all()
        assert (expert_indices < num_experts).all()

        # Verify probabilities sum to 1 (required for weighted combination)
        prob_sums = expert_probs.sum(dim=-1)
        torch.testing.assert_close(
            prob_sums,
            torch.ones(num_tokens, device="cuda", dtype=torch.float32),
            rtol=0.02,
            atol=0.02,
        )

        # Verify probabilities are non-negative
        assert (expert_probs >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

