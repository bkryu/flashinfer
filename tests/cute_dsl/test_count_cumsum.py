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
Unit tests for Count-Cumsum CuTe-DSL kernel.

These tests verify the count_cumsum kernel against PyTorch reference
implementations for histogram counting and cumulative sum operations.
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


def reference_count(x: torch.Tensor, E: int) -> torch.Tensor:
    """Reference histogram count using PyTorch bincount."""
    # bincount requires int64 input and returns int64
    counts = torch.bincount(x.long(), minlength=E)
    return counts[:E].to(torch.int32)


def reference_cumsum(counts: torch.Tensor) -> torch.Tensor:
    """Reference cumulative sum using PyTorch."""
    return torch.cumsum(counts, dim=0).to(torch.int32)


def reference_count_cumsum(
    x: torch.Tensor, E: int, do_cumsum: bool = True
):
    """Reference count + cumsum using PyTorch."""
    count = reference_count(x, E)
    if do_cumsum:
        cumsum = reference_cumsum(count)
        return count, cumsum
    return count


@cute_dsl_available
@hopper_required
class TestCountCumsumBasic:
    """Basic tests for count_cumsum kernel."""

    @pytest.mark.parametrize("N", [100, 1000, 4096, 10000, 50000])
    @pytest.mark.parametrize("E", [4, 8, 16, 64, 256])
    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_count_only(self, N, E, dtype):
        """Test histogram counting without cumsum."""
        from flashinfer.cute_dsl import count_cumsum

        torch.manual_seed(42)

        # Generate random expert indices
        x = torch.randint(0, E, (N,), device="cuda", dtype=dtype)

        # Run CuTe-DSL kernel
        count = count_cumsum(x, E, do_cumsum=False)

        # Reference
        ref_count = reference_count(x, E)

        # Verify shape and dtype
        assert count.shape == (E,)
        assert count.dtype == torch.int32

        # Verify correctness
        assert torch.equal(count, ref_count), f"Count mismatch!\nGot: {count}\nExpected: {ref_count}"

    @pytest.mark.parametrize("N", [100, 1000, 4096, 10000, 50000])
    @pytest.mark.parametrize("E", [4, 8, 16, 64, 256])
    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_count_and_cumsum(self, N, E, dtype):
        """Test histogram counting with cumsum."""
        from flashinfer.cute_dsl import count_cumsum

        torch.manual_seed(42)

        # Generate random expert indices
        x = torch.randint(0, E, (N,), device="cuda", dtype=dtype)

        # Run CuTe-DSL kernel
        count, cumsum = count_cumsum(x, E, do_cumsum=True)

        # Reference
        ref_count, ref_cumsum = reference_count_cumsum(x, E, do_cumsum=True)

        # Verify shapes and dtypes
        assert count.shape == (E,)
        assert cumsum.shape == (E,)
        assert count.dtype == torch.int32
        assert cumsum.dtype == torch.int32

        # Verify correctness
        assert torch.equal(count, ref_count), f"Count mismatch!\nGot: {count}\nExpected: {ref_count}"
        assert torch.equal(cumsum, ref_cumsum), f"Cumsum mismatch!\nGot: {cumsum}\nExpected: {ref_cumsum}"


@cute_dsl_available
@hopper_required
class TestCountCumsumMoEUseCase:
    """Tests simulating MoE routing use cases."""

    @pytest.mark.parametrize("num_tokens", [128, 512, 2048, 8192])
    @pytest.mark.parametrize("num_experts", [8, 16, 32, 64])
    @pytest.mark.parametrize("topk", [1, 2, 4, 8])
    def test_moe_routing_histogram(self, num_tokens, num_experts, topk):
        """Test count_cumsum in MoE routing scenario."""
        from flashinfer.cute_dsl import count_cumsum

        torch.manual_seed(42)

        # Simulate router output: each token selects topk experts
        # Total assignments = num_tokens * topk
        N = num_tokens * topk
        E = num_experts

        # Skip if E not divisible by 4 (kernel constraint)
        if E % 4 != 0:
            pytest.skip(f"E={E} not divisible by 4")

        # Generate expert assignments (flattened)
        expert_indices = torch.randint(0, E, (N,), device="cuda", dtype=torch.int32)

        # Run kernel
        count, cumsum = count_cumsum(expert_indices, E, do_cumsum=True)

        # Reference
        ref_count, ref_cumsum = reference_count_cumsum(expert_indices, E)

        # Verify
        assert torch.equal(count, ref_count)
        assert torch.equal(cumsum, ref_cumsum)

        # Additional MoE-specific checks
        # Total count should equal N
        assert count.sum().item() == N
        # Cumsum[-1] should equal N
        assert cumsum[-1].item() == N


@cute_dsl_available
@hopper_required
class TestCountCumsumEdgeCases:
    """Edge case tests for count_cumsum kernel."""

    def test_all_same_expert(self):
        """Test when all tokens go to the same expert."""
        from flashinfer.cute_dsl import count_cumsum

        N = 1000
        E = 8

        # All tokens assigned to expert 0
        x = torch.zeros(N, device="cuda", dtype=torch.int32)

        count, cumsum = count_cumsum(x, E, do_cumsum=True)

        # Expected: count[0] = N, all others = 0
        expected_count = torch.zeros(E, dtype=torch.int32, device="cuda")
        expected_count[0] = N
        expected_cumsum = torch.full((E,), N, dtype=torch.int32, device="cuda")

        assert torch.equal(count, expected_count)
        assert torch.equal(cumsum, expected_cumsum)

    def test_uniform_distribution(self):
        """Test with uniform expert distribution."""
        from flashinfer.cute_dsl import count_cumsum

        E = 8
        tokens_per_expert = 100
        N = E * tokens_per_expert

        # Create uniform distribution
        x = torch.arange(E, device="cuda", dtype=torch.int32).repeat(tokens_per_expert)

        count, cumsum = count_cumsum(x, E, do_cumsum=True)

        # Expected: each expert gets exactly tokens_per_expert
        expected_count = torch.full((E,), tokens_per_expert, dtype=torch.int32, device="cuda")
        expected_cumsum = torch.arange(1, E + 1, device="cuda", dtype=torch.int32) * tokens_per_expert

        assert torch.equal(count, expected_count)
        assert torch.equal(cumsum, expected_cumsum)

    def test_empty_experts(self):
        """Test when some experts receive no tokens."""
        from flashinfer.cute_dsl import count_cumsum

        N = 100
        E = 8

        # Only use even-numbered experts
        x = torch.randint(0, E // 2, (N,), device="cuda", dtype=torch.int32) * 2

        count, cumsum = count_cumsum(x, E, do_cumsum=True)

        # Reference
        ref_count, ref_cumsum = reference_count_cumsum(x, E)

        assert torch.equal(count, ref_count)
        assert torch.equal(cumsum, ref_cumsum)

        # Odd experts should have 0 count
        for i in range(1, E, 2):
            assert count[i].item() == 0

    def test_small_n(self):
        """Test with very small N (single block)."""
        from flashinfer.cute_dsl import count_cumsum

        N = 10
        E = 4

        x = torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 1, 0], device="cuda", dtype=torch.int32)

        count, cumsum = count_cumsum(x, E, do_cumsum=True)

        # Expected counts: [4, 3, 2, 1]
        expected_count = torch.tensor([4, 3, 2, 1], dtype=torch.int32, device="cuda")
        expected_cumsum = torch.tensor([4, 7, 9, 10], dtype=torch.int32, device="cuda")

        assert torch.equal(count, expected_count)
        assert torch.equal(cumsum, expected_cumsum)

    def test_large_e(self):
        """Test with large number of experts."""
        from flashinfer.cute_dsl import count_cumsum

        N = 10000
        E = 1024  # Large but reasonable

        torch.manual_seed(42)
        x = torch.randint(0, E, (N,), device="cuda", dtype=torch.int32)

        count, cumsum = count_cumsum(x, E, do_cumsum=True)

        # Reference
        ref_count, ref_cumsum = reference_count_cumsum(x, E)

        assert torch.equal(count, ref_count)
        assert torch.equal(cumsum, ref_cumsum)


@cute_dsl_available
@hopper_required
class TestCountCumsumMultiBlock:
    """Tests specifically for multi-block kernel behavior."""

    @pytest.mark.parametrize("N", [4097, 8192, 16384, 50000, 100000])
    def test_large_n_multi_block(self, N):
        """Test with N > CHUNK_SIZE to trigger multiple blocks."""
        from flashinfer.cute_dsl import count_cumsum

        E = 64

        torch.manual_seed(42)
        x = torch.randint(0, E, (N,), device="cuda", dtype=torch.int32)

        count, cumsum = count_cumsum(x, E, do_cumsum=True)

        # Reference
        ref_count, ref_cumsum = reference_count_cumsum(x, E)

        assert torch.equal(count, ref_count)
        assert torch.equal(cumsum, ref_cumsum)

    def test_exactly_one_block(self):
        """Test with N exactly equal to CHUNK_SIZE."""
        from flashinfer.cute_dsl import count_cumsum
        from flashinfer.cute_dsl.count_cumsum import CHUNK_SIZE

        N = CHUNK_SIZE
        E = 32

        torch.manual_seed(42)
        x = torch.randint(0, E, (N,), device="cuda", dtype=torch.int32)

        count, cumsum = count_cumsum(x, E, do_cumsum=True)

        # Reference
        ref_count, ref_cumsum = reference_count_cumsum(x, E)

        assert torch.equal(count, ref_count)
        assert torch.equal(cumsum, ref_cumsum)

    def test_multiple_blocks_boundary(self):
        """Test with N just over CHUNK_SIZE boundary."""
        from flashinfer.cute_dsl import count_cumsum
        from flashinfer.cute_dsl.count_cumsum import CHUNK_SIZE

        N = CHUNK_SIZE + 1
        E = 16

        torch.manual_seed(42)
        x = torch.randint(0, E, (N,), device="cuda", dtype=torch.int32)

        count, cumsum = count_cumsum(x, E, do_cumsum=True)

        # Reference
        ref_count, ref_cumsum = reference_count_cumsum(x, E)

        assert torch.equal(count, ref_count)
        assert torch.equal(cumsum, ref_cumsum)


@cute_dsl_available
@hopper_required
class TestCountCumsumDtypes:
    """Test different input dtypes."""

    def test_int32_input(self):
        """Test with int32 input."""
        from flashinfer.cute_dsl import count_cumsum

        N = 1000
        E = 16

        x = torch.randint(0, E, (N,), device="cuda", dtype=torch.int32)
        count, cumsum = count_cumsum(x, E, do_cumsum=True)

        ref_count, ref_cumsum = reference_count_cumsum(x, E)
        assert torch.equal(count, ref_count)
        assert torch.equal(cumsum, ref_cumsum)

    def test_int64_input(self):
        """Test with int64 input."""
        from flashinfer.cute_dsl import count_cumsum

        N = 1000
        E = 16

        x = torch.randint(0, E, (N,), device="cuda", dtype=torch.int64)
        count, cumsum = count_cumsum(x, E, do_cumsum=True)

        ref_count, ref_cumsum = reference_count_cumsum(x, E)
        assert torch.equal(count, ref_count)
        assert torch.equal(cumsum, ref_cumsum)


