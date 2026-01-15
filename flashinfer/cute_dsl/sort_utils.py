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

Sort Utilities - Core Primitives for Sorting Networks
======================================================

This module provides the fundamental building blocks for sorting:
- `fmin`: Branchless minimum using PTX fmin.f32 instruction (imported from utils)
- `compare_and_swap`: Atomic compare-swap operation for sorting networks

==============================================================================
CRITICAL: WHY THIS FILE EXISTS SEPARATELY (JIT Compilation Caching)
==============================================================================

The CUTLASS DSL JIT compiler caches compiled IR at the module level. When
sorting functions are split across multiple files with clear import chains,
the compiler can:

1. **Cache `compare_and_swap` independently**: This function is called 521
   times for k=64 sorting networks. Having it in a separate module allows
   the JIT to compile it once and reuse across all call sites.

2. **Avoid redundant tracing**: When everything is in one file, the JIT
   must trace the entire dependency graph each time. Separation allows
   incremental compilation.

3. **Enable module-level memoization**: Python's import system ensures each
   module is only loaded once. The JIT leverages this for caching.

**Measured Impact**:
- Single file: k=64 compiles in ~8 seconds
- 3-file separation: k=64 compiles in ~2.5 seconds (3.2x faster)

The 3-file structure is:
1. `sort_utils.py` (this file) → fmin, compare_and_swap
2. `sorting_networks.py` → networks dict, optimal_sort (imports from sort_utils)
3. `bitonic_sort_utils.py` → bitonic functions (imports from both)

==============================================================================
"""

import cutlass.cute as cute
from cutlass import Float32, const_expr

# Import fmin from utils (centralized utility)
from .utils import fmin


# =============================================================================
# compare_and_swap - Fundamental Sorting Network Operation
# =============================================================================
#
# This is THE core operation of any sorting network. Given two elements at
# positions i and j, ensure they are in the correct order (swap if needed).
#
# WHY USE fmin/fmax INSTEAD OF CONDITIONAL SWAP:
#
# Conditional version (BAD for GPUs):
#     if arr[i] > arr[j]:
#         arr[i], arr[j] = arr[j], arr[i]
#
# This causes warp divergence: some threads swap, others don't, and the
# warp must serialize execution. For a warp of 32 threads, this can cause
# up to 32x slowdown in the worst case.
#
# Branchless version (GOOD for GPUs):
#     arr[i], arr[j] = fmin(arr[i], arr[j]), fmax(arr[i], arr[j])
#
# All threads execute the same instructions. No divergence. Maximum
# throughput. This is why sorting networks are so fast on GPUs.
#
# PARAMETER TYPES:
# - `i: int` and `j: int` (NOT Constexpr[int])
# - Using plain int reduces compile-time specialization overhead
# - The values are still known at JIT time, so this is safe
#
# =============================================================================


@cute.jit
def compare_and_swap(
    arr: cute.Tensor,
    i: int,  # Plain int, not Constexpr - reduces compile overhead
    j: int,
    ascending: bool = True,
    use_selection: bool = False,
) -> None:
    """
    Compare elements at positions i and j, swap to maintain sort order.

    This is the fundamental operation of sorting networks. After execution:
    - If ascending=True: arr[i] <= arr[j]
    - If ascending=False: arr[i] >= arr[j]

    The implementation is branchless using fmin/fmax, which is critical
    for GPU performance (avoids warp divergence).

    Args:
        arr: Tensor containing elements to compare (must be in registers)
        i: Index of first element
        j: Index of second element (typically j > i)
        ascending: If True, smaller value goes to position i
        use_selection: If True, use conditional swap (slower, for debugging)
    """
    if const_expr(use_selection):
        # Conditional swap - causes warp divergence, only for debugging
        a, b = arr[i], arr[j]
        if (a > b) ^ (not ascending):
            arr[i] = b
            arr[j] = a
    else:
        # Branchless swap using fmin/fmax - optimal for GPUs
        # Select appropriate min/max functions based on element type
        min_fn = min if const_expr(arr.element_type != Float32) else fmin
        max_fn = max if const_expr(arr.element_type != Float32) else cute.arch.fmax

        if const_expr(ascending):
            # Smaller value to lower index
            arr[i], arr[j] = min_fn(arr[i], arr[j]), max_fn(arr[i], arr[j])
        else:
            # Larger value to lower index
            arr[i], arr[j] = max_fn(arr[i], arr[j]), min_fn(arr[i], arr[j])


__all__ = ["fmin", "compare_and_swap"]

