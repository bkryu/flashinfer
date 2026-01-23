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

Block Prefix Sum Kernel for CuTe-DSL
====================================

Helper functions for warp-level and block-level prefix sum operations,
used by the filtered top-k kernel.
"""

import cutlass
import cutlass.cute as cute


@cute.jit
def warp_scan(
    val: cutlass.Int32, tidx, lane_id, num_threads_per_warp: cutlass.Constexpr
):
    """Warp-level inclusive prefix sum using shuffle instructions.

    Args:
        val: Input value for this thread
        tidx: Thread index within the block
        lane_id: Lane index within the warp (0-31)
        num_threads_per_warp: Number of threads participating in the scan

    Returns:
        Inclusive prefix sum for this thread's lane
    """
    mask_val = cutlass.const_expr(
        cutlass.Int32(((1 << num_threads_per_warp) - 1) & 0xFFFFFFFF)
    )
    mask_and_clamp_val = cutlass.const_expr(cutlass.Int32(0))
    iteration = cute.arch.log2_of_pow2_int(cutlass.Int32(num_threads_per_warp))

    for i in cutlass.range(iteration, unroll_full=True):
        offset = cutlass.Int32(1) << i
        other = cute.arch.shuffle_sync_up(
            val, cutlass.Int32(offset), mask=mask_val, mask_and_clamp=mask_and_clamp_val
        )
        if lane_id >= offset:
            val = val + other
        else:
            val = val

    return val


@cute.jit
def block_prefix_sum_kernel(
    val: cutlass.Int32,
    warp_sums: cute.Tensor,
    tidx,
    num_threads,
    num_warps,
    barrier_id=1,
    need_total_sum=False,
):
    """Block-level inclusive prefix sum using warp-level scans.

    This implements a two-level reduction:
    1. Warp-level prefix sum using shuffle
    2. Cross-warp prefix sum using shared memory

    Args:
        val: Input value for this thread
        warp_sums: Shared memory tensor for cross-warp communication
        tidx: Thread index within the block
        num_threads: Number of threads participating
        num_warps: Number of warps participating
        barrier_id: Barrier ID for synchronization
        need_total_sum: Whether to compute and return total sum

    Returns:
        Tuple of (prefix_sum, total_sum) where total_sum is 0 if not requested
    """
    # Thread and warp id
    warp_id = tidx // cute.Int32(32)
    lane_id = tidx % cute.Int32(32)

    # Currently, we only support num_warps > 1
    assert num_threads % 32 == 0, (
        "num_threads must be divisible by 32, but got {}".format(num_threads)
    )
    assert num_warps > 1, "num_warps must be > 1, but got {}".format(num_warps)

    # Step 1: Warp-level prefix sum using shuffle
    val = warp_scan(val, tidx, lane_id, num_threads_per_warp=32)

    # Step 2: Store warp prefix sums
    if lane_id == 31:  # Last thread in warp stores warp sum
        warp_sums[warp_id] = val
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)

    # Step 3: Prefix sum across warps
    if warp_id == 0:
        if lane_id < num_warps:
            warp_val = warp_sums[lane_id]
            # Call warp-level prefix sum
            warp_val = warp_scan(
                warp_val, tidx, lane_id, num_threads_per_warp=num_warps
            )
            warp_sums[lane_id] = warp_val
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)

    # Step 4: Add warp-level prefix
    if warp_id > 0:
        val = val + warp_sums[warp_id - 1]

    # Step 5: Get total sum if need_total_sum is True
    total_sum = 0
    if need_total_sum:
        if tidx == num_threads - 1:
            # Note: warp_id must be >= 1
            warp_sums[warp_id - 1] = val
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)
        total_sum = warp_sums[num_warps - 1]

    return val, total_sum

