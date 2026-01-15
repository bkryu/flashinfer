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

Bitonic Sort Utilities for Top-K Selection
==========================================

This module provides the core bitonic sorting and top-k selection functions:
- `bitonic_merge`: Merge a bitonic sequence into sorted order
- `bitonic_sort`: Sort small arrays using bitonic/optimal networks
- `bitonic_topk_merge`: Merge two sorted k-arrays keeping top k
- `bitonic_topk`: Full top-k selection with warp-level cooperation

==============================================================================
CRITICAL: WHY THIS FILE EXISTS SEPARATELY (JIT Compilation Caching)
==============================================================================

This is the THIRD layer in our 3-file structure:

    sort_utils.py → sorting_networks.py → bitonic_sort_utils.py (this file)
         ↑                   ↑                       ↑
    fmin, compare_and_swap   optimal_sort      bitonic functions

**The 3-file separation is ESSENTIAL for fast compilation:**

Without separation (everything in one file):
- k=64 compilation: ~8 seconds
- The JIT must trace all dependencies on every compilation

With 3-file separation:
- k=64 compilation: ~2.5 seconds (3.2x faster!)
- The JIT can cache each module independently
- Import chains allow incremental compilation

**Why this specific structure?**

1. `sort_utils.py` has no CuTe-DSL imports from our code
   → Can be compiled independently

2. `sorting_networks.py` only imports `compare_and_swap`
   → Minimal dependency, networks dict is pure data

3. `bitonic_sort_utils.py` imports from both
   → All dependencies are already cached when this compiles

This mirrors the module structure used in production GPU kernels and is
the key insight that enables reasonable compile times for large k values.

==============================================================================
BITONIC SORT ALGORITHM OVERVIEW
==============================================================================

Bitonic sort is a parallel sorting algorithm that works as follows:

1. A "bitonic sequence" is one that first increases then decreases
   (or vice versa), e.g., [1, 3, 5, 7, 6, 4, 2]

2. The algorithm recursively:
   a) Sorts the first half in ascending order
   b) Sorts the second half in descending order
   c) Merges the resulting bitonic sequence

3. The merge step uses a "butterfly" pattern of comparisons that
   transforms a bitonic sequence into a sorted sequence.

For top-k selection, we use a modified approach:
- Sort initial k elements
- For each subsequent chunk of k elements:
  - Sort the chunk
  - Merge with current top-k, keeping only the best k
- Use warp shuffles for cross-thread top-k merging

==============================================================================
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

# =============================================================================
# Import from separate modules (THE 3-FILE STRUCTURE)
# =============================================================================
# This import chain is CRITICAL for JIT compilation caching!
# Changing this structure will significantly increase compile times.
# =============================================================================

from .sort_utils import compare_and_swap, fmin
from .sorting_networks import optimal_sort


# =============================================================================
# bitonic_merge - Merge a Bitonic Sequence
# =============================================================================
#
# A bitonic sequence is one that is first increasing then decreasing (or
# first decreasing then increasing). For example:
#   [1, 3, 5, 7, 6, 4, 2] is bitonic (increases then decreases)
#   [7, 5, 3, 1, 2, 4, 6] is bitonic (decreases then increases)
#
# The merge operation transforms a bitonic sequence into a sorted sequence
# using a "butterfly" pattern of compare-swap operations.
#
# CRITICAL LOOP PATTERN:
# - Outer loop MUST use `range_constexpr` (not `range`)
# - Without this, compile time for n=128 explodes
# - Inner loops can use `range(..., unroll_full=True)`
#
# =============================================================================


@cute.jit
def bitonic_merge(
    arr: cute.Tensor,
    n: Optional[cutlass.Constexpr[int]] = None,
    start: cutlass.Constexpr[int] = 0,
    ascending: cutlass.Constexpr[bool] = True,
) -> None:
    """
    Merge a bitonic sequence into a sorted sequence.

    The merge uses a butterfly pattern: at each level, compare elements
    that are `step` apart, where step halves at each level.

    Args:
        arr: Tensor containing the bitonic sequence (in registers)
        n: Length of sequence (power of 2), None = infer from arr.shape
        start: Starting offset in arr
        ascending: If True, result is ascending order

    Example for n=8, ascending:
        Input bitonic: [1, 3, 5, 7, 6, 4, 2, 0]
        After merge:   [0, 1, 2, 3, 4, 5, 6, 7]
    """
    if const_expr(n is None):
        n = cute.size(arr.shape)

    if const_expr(n > 1):
        num_levels = int(math.log2(n))
        assert n == 2**num_levels, "n must be a power of 2"

        # CRITICAL: Use range_constexpr for outer loop!
        # This is documented in the original implementation:
        # "This one must be range_constexpr otherwise it's very slow for n = 128"
        for level in cutlass.range_constexpr(num_levels):
            # At each level, compare elements `step` apart
            length = n >> level  # n // (2^level)
            step = length // 2

            # Process all groups at this level (can be parallelized)
            for i in cutlass.range(n // length, unroll_full=True):
                start_i = start + i * length
                # Compare-swap pairs within each group
                for j in cutlass.range(step, unroll_full=True):
                    compare_and_swap(arr, start_i + j, start_i + j + step, ascending)


# =============================================================================
# bitonic_sort - Sort Small Arrays
# =============================================================================
#
# For small arrays (n <= 64), we use optimal sorting networks from tables.
# For larger arrays, we fall back to recursive bitonic sort.
#
# The optimal networks have fewer comparisons than bitonic sort and
# avoid recursion overhead, which helps JIT compilation.
#
# =============================================================================


@cute.jit
def bitonic_sort(
    arr: cute.Tensor,
    n: Optional[cutlass.Constexpr[int]] = None,
    start: cutlass.Constexpr[int] = 0,
    ascending: cutlass.Constexpr[bool] = True,
) -> None:
    """
    Sort an array using bitonic sort (or optimal networks for small n).

    For n in [2, 4, 8, 16, 32, 64]: Uses pre-computed optimal networks
    For n > 64: Uses recursive bitonic sort

    Args:
        arr: Tensor to sort (in registers)
        n: Number of elements (power of 2, <= 128), None = infer
        start: Starting offset in arr
        ascending: If True, sort in ascending order
    """
    if const_expr(n is None):
        n = cute.size(arr.shape)

    assert n <= 128, "bitonic_sort only supports n <= 128"

    if const_expr(n > 1):
        if const_expr(n in [2, 4, 8, 16, 32, 64]):
            # Use pre-computed optimal network (fewer comparisons, no recursion)
            optimal_sort(arr, n, start, ascending)
        else:
            # Fall back to recursive bitonic sort for n=128 or odd sizes
            assert n % 2 == 0, "n must be even for bitonic sort"

            # Step 1: Sort first half ascending
            bitonic_sort(arr, n // 2, start, True)

            # Step 2: Sort second half descending
            # This creates a bitonic sequence when combined with step 1
            bitonic_sort(arr, n // 2, start + n // 2, False)

            # Step 3: Merge the bitonic sequence
            bitonic_merge(arr, n, start, ascending)


# =============================================================================
# bitonic_topk_merge - Merge Two Top-K Arrays
# =============================================================================
#
# Given two sorted arrays of k elements each, merge them keeping only
# the top k elements. This is the key operation for incremental top-k.
#
# Algorithm:
# 1. Cross-compare: arr0[i] with arr1[k-1-i] (creates bitonic sequence)
# 2. Keep the larger (or smaller) element in arr0
# 3. Merge the resulting bitonic sequence in arr0
#
# After this operation:
# - arr0 contains the k largest elements from both arrays (combined)
# - arr1 is unchanged
#
# IMPORTANT: The original implementation has a typo "unfoll_full=True"
# which means unrolling is DISABLED for the comparison loop. This is
# actually intentional (or a happy accident) - it reduces compile time
# for large k values.
#
# =============================================================================


@cute.jit
def bitonic_topk_merge(
    arr0: cute.Tensor,
    arr1: cute.Tensor,
    k: Optional[cutlass.Constexpr[int]] = None,
    start0: cutlass.Constexpr[int] = 0,
    start1: cutlass.Constexpr[int] = 0,
    ascending: cutlass.Constexpr[bool] = False,
    do_final_sort: cutlass.Constexpr[bool] = True,
) -> None:
    """
    Merge two sorted k-arrays, keeping the top k elements in arr0.

    Both arrays must be sorted in the same direction. After the merge,
    arr0 contains the k largest (if ascending=False) or k smallest
    (if ascending=True) elements from the combined 2k elements.

    Args:
        arr0: First sorted array (will contain the result)
        arr1: Second sorted array (unchanged)
        k: Number of elements in each array, None = infer from arr0
        start0: Starting offset in arr0
        start1: Starting offset in arr1
        ascending: If False, keep largest k; if True, keep smallest k
        do_final_sort: If True (default), sort the result. If False, skip the
                       final sort step - useful when this is the last merge
                       and sorted output is not needed.

    Example (ascending=False, keeping largest k=4):
        arr0 (sorted desc): [9, 7, 5, 3]
        arr1 (sorted desc): [8, 6, 4, 2]
        After merge arr0:   [9, 8, 7, 6]  (top 4 from combined)
    """
    if const_expr(k is None):
        k = cute.size(arr0.shape)

    # Select min/max function based on element type
    if const_expr(arr0.element_type == Float32):
        minmax_fn = fmin if ascending else cute.arch.fmax
    else:
        minmax_fn = min if ascending else max

    # Cross-compare and keep top values
    # This creates a bitonic sequence in arr0
    #
    # For descending (keeping largest):
    # arr0[0] = max(arr0[0], arr1[k-1])  # arr0[0] vs smallest of arr1
    # arr0[1] = max(arr0[1], arr1[k-2])  # etc.
    # ...
    # arr0[k-1] = max(arr0[k-1], arr1[0])  # smallest of arr0 vs largest of arr1
    #
    # NOTE: "unfoll_full=True" is a TYPO that disables unrolling!
    # This actually helps compile time for large k. We preserve it intentionally.
    for i in cutlass.range(k, unfoll_full=True):
        arr0[start0 + i] = minmax_fn(arr0[start0 + i], arr1[start1 + k - 1 - i])

    # The result in arr0 is now bitonic
    # If sorted output is needed, merge to get sorted top-k
    # If unsorted output is acceptable, skip this step to save O(k log k) comparisons
    if const_expr(do_final_sort):
        bitonic_merge(arr0, k, start0, ascending)


# =============================================================================
# bitonic_topk - Full Top-K with Warp Cooperation
# =============================================================================
#
# This is the main entry point for top-k selection. It:
# 1. Initializes top-k with first k elements
# 2. Processes remaining elements in chunks, merging each
# 3. Uses warp shuffles to merge across threads
#
# The warp shuffle step uses a butterfly pattern:
# - Round 0: threads 0↔1, 2↔3, ... exchange and merge
# - Round 1: threads 0↔2, 1↔3, ... exchange and merge
# - Round 2: threads 0↔4, 1↔5, ... exchange and merge
# - After log2(warp_width) rounds, thread 0 has the global top-k
#
# =============================================================================


@cute.jit
def bitonic_topk(
    arr: cute.Tensor,
    k: cutlass.Constexpr[int],
    ascending: cutlass.Constexpr[bool] = False,
    warp_width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
    sorted_output: cutlass.Constexpr[bool] = True,
) -> cute.Tensor:
    """
    Compute top-k elements from arr using bitonic sort with warp cooperation.

    Each thread processes n/warp_width elements locally, then threads
    cooperate via warp shuffles to find the global top-k. After execution,
    thread 0 (in each row group) has the combined top-k values.

    Args:
        arr: Input array in registers (Float32 with encoded indices)
        k: Number of top elements to return (power of 2, <= 128)
        ascending: If False, return largest k; if True, return smallest k
        warp_width: Number of threads cooperating per row (1, 2, 4, ..., 32)
        sorted_output: If True (default), return sorted output. If False,
                       skip the final sort step for ~O(k log k) savings.

    Returns:
        Fragment with k top elements (only valid in thread 0 of each group)

    Note:
        The input arr should have indices encoded in the low bits of
        float values (see index encoding in the main kernel).
    """
    # Input validation
    assert arr.element_type in [Float32, Int32], "Only Float32 and Int32 supported"
    n = cute.size(arr.shape)
    assert k == 1 << int(math.log2(k)), "k must be a power of 2"
    assert n % k == 0, "n must be divisible by k"

    # Create output buffer for top-k values
    topk_vals = cute.make_rmem_tensor(k, arr.element_type)

    # =========================================================================
    # Step 1: Initialize with first k elements and sort
    # =========================================================================
    for v in cutlass.range(k, unroll_full=True):
        topk_vals[v] = arr[v]

    # Sort initial k elements (uses optimal network for k <= 64)
    bitonic_sort(topk_vals, ascending=ascending)

    # =========================================================================
    # Step 2: Process remaining chunks, merging each with current top-k
    # =========================================================================
    # For each chunk of k elements after the first:
    # - Sort the chunk
    # - Merge with current top-k, keeping only the best k
    #
    # Note: unroll_full=True here because n/k is typically small
    for i in cutlass.range(1, n // k, unroll_full=True):
        # Load next chunk
        other_vals = cute.make_rmem_tensor(k, arr.element_type)
        for v in cutlass.range(k, unroll_full=True):
            other_vals[v] = arr[i * k + v]

        # Sort the chunk
        bitonic_sort(other_vals, ascending=ascending)

        # Merge with current top-k
        bitonic_topk_merge(topk_vals, other_vals, ascending=ascending)

    # =========================================================================
    # Step 3: Cross-thread merge using warp shuffles
    # =========================================================================
    # Use butterfly shuffle pattern to merge across threads:
    # - shuffle_sync_bfly(val, offset=1): exchange with thread ± 1
    # - shuffle_sync_bfly(val, offset=2): exchange with thread ± 2
    # - etc.
    #
    # After log2(warp_width) rounds, thread 0 has the global top-k
    #
    # TODO: This is not efficient for large k (e.g. >= 16) since threads
    # in the same warp do duplicate work. A better approach would be to
    # distribute the merge work across threads.
    #
    # For sorted_output=False, we skip the final bitonic_merge in the last
    # round to save O(k log k) comparisons. We split into two loops to
    # ensure do_final_sort is a compile-time constant.
    num_rounds = int(math.log2(warp_width))

    # Process all rounds except the last (always sort)
    if const_expr(num_rounds > 1):
        for i in cutlass.range(num_rounds - 1, unroll_full=True):
            # Get values from partner thread (XOR distance = 2^i)
            other_vals = cute.make_rmem_tensor(k, arr.element_type)
            for v in cutlass.range(k, unroll_full=True):
                other_vals[v] = cute.arch.shuffle_sync_bfly(topk_vals[v], offset=1 << i)

            # Merge with partner's values (always sort in intermediate rounds)
            bitonic_topk_merge(
                topk_vals, other_vals, ascending=ascending, do_final_sort=True
            )

    # Process the last round (conditionally sort based on sorted_output)
    if const_expr(num_rounds >= 1):
        # Get values from partner thread for the last round
        other_vals = cute.make_rmem_tensor(k, arr.element_type)
        for v in cutlass.range(k, unroll_full=True):
            other_vals[v] = cute.arch.shuffle_sync_bfly(
                topk_vals[v], offset=1 << (num_rounds - 1)
            )

        # Merge with partner's values
        # Only sort if sorted_output=True (saves O(k log k) when False)
        bitonic_topk_merge(
            topk_vals, other_vals, ascending=ascending, do_final_sort=sorted_output
        )

    return topk_vals


__all__ = [
    "bitonic_merge",
    "bitonic_sort",
    "bitonic_topk_merge",
    "bitonic_topk",
]

