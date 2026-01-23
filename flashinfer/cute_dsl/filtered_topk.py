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

Filtered Top-K using CuTe-DSL
=============================

High-performance radix-based filter top-k algorithm for the NVIDIA Blackwell SM100
architecture using CuTe DSL.

The radix-based filter top-k algorithm has two phases:
1. Coarse filter: Build histogram using 8-bit radix, find threshold bin
2. Fine-grained filter: Multiple rounds of refinement to find exact top-k

Supported data types:
- Float32
- Float16
- BFloat16

Constraints:
- num_cols >= 256 and must be divisible by vec_size (256 bits / element_bits)
- top_k <= 2048
- Input tensor must be 32-byte aligned (for 256-bit vectorized loads)
- Input tensor must be row-major (contiguous on columns)
"""

import functools
from typing import Callable, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass._mlir.dialects import llvm

from cutlass.cute.arch.nvvm_wrappers import atomic_add as atomicAdd

from ..api_logging import flashinfer_api
from .block_scan import block_prefix_sum_kernel


# =============================================================================
# Helper Functions
# =============================================================================


def half_as_ushort(half_val):
    """Interpret FP16 value as uint16 bit pattern."""
    return llvm.bitcast(cutlass.Uint16.mlir_type, half_val.ir_value())


def float_as_uint32(float_val):
    """Interpret FP32 value as uint32 bit pattern."""
    return llvm.bitcast(cutlass.Uint32.mlir_type, float_val.ir_value())


# =============================================================================
# Filtered Top-K Kernel Class
# =============================================================================


class FilteredTopKKernel:
    """Radix-based filtered top-k kernel using CuTe-DSL.

    This implements a histogram-based radix filter algorithm that is O(n) in
    vocabulary size, making it faster than heap-based O(n log k) methods for
    large vocabularies.

    Args:
        dtype: Data type (Float32, Float16, BFloat16)
        num_cols: Number of columns (vocabulary size)
        top_k: Number of top elements to select
        num_copy_bits: Bits per vectorized copy (default 256)
        return_val: Whether to return values (default True)
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        num_cols: int,
        top_k: int,
        num_copy_bits: int = 256,
        return_val: bool = True,
    ):
        self.dtype = dtype
        self.num_cols = num_cols
        self.top_k = top_k
        self.num_copy_bits = num_copy_bits

        self.filtered_topk_max_k = 2048
        self.filtered_topk_smem_input_size = 16 * 1024
        # 8 bits for radix-based filter
        self.radix = 256

        self.return_val = return_val

        self.vec_size = num_copy_bits // dtype.width
        assert self.num_cols >= 256, f"num_cols must be >= 256, but got {self.num_cols}"
        if cutlass.const_expr(
            dtype not in [cutlass.Float32, cute.BFloat16, cutlass.Float16]
        ):
            raise ValueError(f"Unsupported dtype: {dtype}")

        # Determine thread block size based on problem size
        if self.num_cols >= self.vec_size * 1024:
            self.num_threads_per_cta = 1024
        else:
            if cutlass.const_expr(dtype == cutlass.Float32):
                if cutlass.const_expr(self.num_cols > 2048 and self.num_cols < 8192):
                    self.num_threads_per_cta = 512
                else:
                    self.num_threads_per_cta = 256
            else:
                if cutlass.const_expr(self.num_cols > 4096 and self.num_cols < 16384):
                    self.num_threads_per_cta = 512
                else:
                    self.num_threads_per_cta = 256

        assert self.num_cols % self.vec_size == 0, (
            f"num_cols must be divisible by vec_size, but got {self.num_cols} and {self.vec_size}"
        )

        # Radix-based filter parameters
        if cutlass.const_expr(dtype == cutlass.Float32):
            self.ordered_type = cute.Uint32
            self.first_refine_shift = 24
            self.num_refine_rounds = 4
        elif cutlass.const_expr(dtype in [cutlass.Float16, cute.BFloat16]):
            self.ordered_type = cute.Uint16
            self.first_refine_shift = 0
            self.num_refine_rounds = 1

    @cute.jit
    def to_coarse_key(self, x):
        """Convert to coarse 8-bit key for histogram."""
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            # Convert to FP16 and extract high 8 bits
            h = x.to(cutlass.Float16)
            bits = half_as_ushort(h)

            key = cutlass.Uint16(0)

            # Extract the sign bit and convert to ordered representation
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)

            # High 8 bits
            return cute.Uint8((key >> 8) & 0xFF)
        else:
            # For half/bfloat16, extract high 8 bits directly
            if cutlass.const_expr(self.dtype == cutlass.Float16):
                bits = half_as_ushort(x)
            else:  # BFloat16
                bits = half_as_ushort(x)

            key = cute.Uint16(0)
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
            # High 8 bits
            return cute.Uint8((key >> 8) & 0xFF)

    @cute.jit
    def to_ordered(self, x):
        """Convert to ordered integer for comparison."""
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            bits = float_as_uint32(x)

            key = cutlass.Uint32(0)
            if bits & 0x80000000:
                key = cutlass.Uint32(bits)
            else:
                key = (bits ^ cutlass.Uint32(0xFFFFFFFF)) & cutlass.Uint32(0x7FFFFFFF)
            return cute.Uint32(key)
        else:
            if cutlass.const_expr(self.dtype == cutlass.Float16):
                bits = half_as_ushort(x)
            else:  # BFloat16
                bits = half_as_ushort(x)

            key = cute.Uint16(0)
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cute.Uint16(0xFFFF)) & cute.Uint16(0x7FFF)
            return cute.Uint16(key)

    @cute.jit
    def prefix_sum(self, tidx, s_histogram, s_warp_sums, num_warps):
        """Compute prefix sum on histogram."""
        if cutlass.const_expr(self.radix < self.num_threads_per_cta):
            if tidx < cutlass.Int32(self.radix):
                val = s_histogram[tidx]
                val, total_sum = block_prefix_sum_kernel(
                    val, s_warp_sums, tidx, self.radix, num_warps
                )
                s_histogram[tidx] = val
        elif cutlass.const_expr(self.radix == self.num_threads_per_cta):
            val = s_histogram[tidx]
            val, total_sum = block_prefix_sum_kernel(
                val, s_warp_sums, tidx, self.radix, num_warps
            )
            s_histogram[tidx] = val
        else:
            assert self.radix % self.num_threads_per_cta == 0
            previous_sum = 0
            val = cutlass.Int32(0)
            total_sum = cutlass.Int32(0)
            for i in range(tidx, self.radix, self.num_threads_per_cta):
                val = s_histogram[i]
                val, total_sum = block_prefix_sum_kernel(
                    val,
                    s_warp_sums,
                    tidx,
                    self.num_threads_per_cta,
                    num_warps,
                    need_total_sum=True,
                )
                s_histogram[i] = val + previous_sum
                previous_sum = previous_sum + total_sum

    @cute.jit
    def prefix_sum_and_find_threshold_coarse(
        self,
        tidx,
        s_histogram,
        s_warp_sums,
        num_warps,
        s_threshold_bin_id,
        s_num_input,
        s_counter,
        s_last_remain,
        topk_remaining,
        s_num_input_idx=0,
    ):
        """Prefix sum and find threshold bin for coarse filter."""
        if cutlass.const_expr(self.radix <= self.num_threads_per_cta):
            previous = 0
            if tidx < cutlass.Int32(self.radix):
                val = s_histogram[tidx]
                val, total_sum = block_prefix_sum_kernel(
                    val, s_warp_sums, tidx, self.radix, num_warps, barrier_id=1
                )
                s_histogram[tidx] = val
                # Sync among self.radix threads
                cute.arch.barrier(barrier_id=1, number_of_threads=self.radix)

                if tidx > 0:
                    previous = s_histogram[tidx - 1]
                if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                    s_threshold_bin_id[0] = tidx
                    s_num_input[s_num_input_idx] = 0
                    s_counter[0] = 0
            # Sync among all threads in a CTA
            cute.arch.barrier()
        else:
            assert self.radix % self.num_threads_per_cta == 0
            previous_sum = 0
            val = cutlass.Int32(0)
            total_sum = cutlass.Int32(0)
            for i in range(tidx, self.radix, self.num_threads_per_cta):
                val = s_histogram[i]
                val, total_sum = block_prefix_sum_kernel(
                    val,
                    s_warp_sums,
                    tidx,
                    self.num_threads_per_cta,
                    num_warps,
                    barrier_id=2,
                    need_total_sum=True,
                )
                s_histogram[i] = val + previous_sum
                previous_sum = previous_sum + total_sum
            # Sync among all threads in a CTA
            cute.arch.barrier()

            previous = 0
            run_loop = True
            if tidx > 0:
                previous = s_histogram[tidx - 1]
            if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                s_threshold_bin_id[0] = tidx
                s_num_input[s_num_input_idx] = 0
                s_counter[0] = 0
                run_loop = False

            if run_loop:
                run_next_loop = True
                for i in range(
                    tidx + self.num_threads_per_cta,
                    self.radix,
                    self.num_threads_per_cta,
                ):
                    if run_next_loop:
                        previous = s_histogram[i - 1]
                        if (
                            previous <= topk_remaining
                            and s_histogram[i] > topk_remaining
                        ):
                            s_threshold_bin_id[0] = i
                            s_num_input[s_num_input_idx] = 0
                            s_counter[0] = 0
                            run_next_loop = False
            # Sync among all threads in a CTA
            cute.arch.barrier()

    @cute.jit
    def prefix_sum_and_find_threshold_fine_grained(
        self,
        tidx,
        s_histogram,
        s_warp_sums,
        num_warps,
        s_threshold_bin_id,
        s_num_input,
        s_counter,
        s_last_remain,
        topk_remaining,
        s_num_input_idx=0,
    ):
        """Prefix sum and find threshold bin for fine-grained filter."""
        if cutlass.const_expr(self.radix <= self.num_threads_per_cta):
            previous = 0
            if tidx < cutlass.Int32(self.radix):
                val = s_histogram[tidx]
                val, total_sum = block_prefix_sum_kernel(
                    val, s_warp_sums, tidx, self.radix, num_warps, barrier_id=1
                )
                s_histogram[tidx] = val
                # Sync
                cute.arch.barrier(barrier_id=1, number_of_threads=self.radix)

                if tidx > 0:
                    previous = s_histogram[tidx - 1]
                if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                    s_threshold_bin_id[0] = tidx
                    s_num_input[s_num_input_idx] = 0
                    # The difference between coarse and fine-grained
                    s_last_remain[0] = topk_remaining - s_histogram[tidx - 1]
            cute.arch.barrier()
        else:
            assert self.radix % self.num_threads_per_cta == 0
            previous_sum = 0
            val = cutlass.Int32(0)
            total_sum = cutlass.Int32(0)
            for i in range(tidx, self.radix, self.num_threads_per_cta):
                val = s_histogram[i]
                val, total_sum = block_prefix_sum_kernel(
                    val,
                    s_warp_sums,
                    tidx,
                    self.num_threads_per_cta,
                    num_warps,
                    barrier_id=2,
                    need_total_sum=True,
                )
                s_histogram[i] = val + previous_sum
                previous_sum = previous_sum + total_sum
            # Sync among all threads in a CTA
            cute.arch.barrier()

            previous = 0
            run_loop = True
            if tidx > 0:
                previous = s_histogram[tidx - 1]
            if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                s_threshold_bin_id[0] = tidx
                s_num_input[s_num_input_idx] = 0
                # The difference between coarse and fine-grained
                s_last_remain[0] = topk_remaining - s_histogram[tidx - 1]
                run_loop = False
            if run_loop:
                run_next_loop = True
                for i in range(
                    tidx + self.num_threads_per_cta,
                    self.radix,
                    self.num_threads_per_cta,
                ):
                    if run_next_loop:
                        previous = s_histogram[i - 1]
                        if (
                            previous <= topk_remaining
                            and s_histogram[i] > topk_remaining
                        ):
                            s_threshold_bin_id[0] = i
                            s_num_input[s_num_input_idx] = 0
                            # The difference between coarse and fine-grained
                            s_last_remain[0] = topk_remaining - s_histogram[i - 1]
                            run_next_loop = False
            # Sync among all threads in a CTA
            cute.arch.barrier()

    @cute.kernel
    def filtered_topk_kernel(
        self,
        input: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
        tiler_mn: cute.Shape,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
    ):
        """CuTe DSL implementation of TopK kernel based on radix-based filter algorithm."""
        # Thread and block indexing
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        length = self.num_cols
        score = input[bidx, None]
        dst = output_indices[bidx, None]
        if cutlass.const_expr(self.return_val):
            dst_values = output_values[bidx, None]

        shape = input.shape
        idX = cute.make_identity_tensor(shape)

        # Slice for CTAs
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, None)) for mT in (input, idX)]
        self.num_sub_tiles = gX.shape[2]

        thr_copy = tiled_copy.get_slice(tidx)

        tXgX = thr_copy.partition_S(gX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None, None]
        tXrX = cute.make_fragment_like(tXgX[None, None, None, 0])

        is_even_N = cutlass.const_expr(shape[1] % tiler_mn[1] == 0)
        tXpX = (
            None
            if is_even_N
            else self.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        )

        # Trivial case: length <= top_k
        if cutlass.const_expr(length <= self.top_k):
            for i in range(tidx, self.top_k, self.num_threads_per_cta):
                if i < length:
                    dst[i] = i
                    if cutlass.const_expr(self.return_val):
                        dst_values[i] = score[i]
                else:
                    dst[i] = -1
                    if cutlass.const_expr(self.return_val):
                        dst_values[i] = self.dtype(0.0)
        else:
            # Shared memory allocation
            smem = utils.SmemAllocator()
            s_histogram_buf_layout = cute.make_ordered_layout(
                (self.radix + 128), order=(0)
            )
            s_histogram = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=s_histogram_buf_layout,
                byte_alignment=128,
            )
            s_counter = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1), order=(0)),
                byte_alignment=128,
            )
            s_threshold_bin_id = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1), order=(0)),
                byte_alignment=128,
            )
            s_num_input = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((2,), order=(0)),
                byte_alignment=128,
            )
            s_indices = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout(
                    (self.filtered_topk_max_k,), order=(0)
                ),
                byte_alignment=128,
            )
            s_input_idx = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout(
                    (2, self.filtered_topk_smem_input_size), order=(1, 0)
                ),
                byte_alignment=128,
            )
            s_last_remain = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1), order=(0)),
                byte_alignment=128,
            )
            num_warps = cutlass.const_expr(
                min(self.radix, self.num_threads_per_cta) // cutlass.Int32(32)
            )
            s_warp_sums = smem.allocate_tensor(
                element_type=cute.Int32,
                layout=cute.make_ordered_layout((num_warps,), order=(0,)),
                byte_alignment=128,
            )

            topk_remaining = cutlass.Int32(self.top_k)

            val_one = cutlass.Int32(1)
            val_one_negative = cutlass.Int32(-1)

            # Stage 1: Coarse histogram
            if tidx < self.radix + 1:
                s_histogram[tidx] = 0
            cute.arch.barrier()

            # 1.1 Build histogram with vectorized loads
            vec_size = self.vec_size

            for tile_idx in range(self.num_sub_tiles):
                if cutlass.const_expr(tXpX is not None):
                    cute.copy(
                        copy_atom,
                        tXgX[None, None, None, tile_idx],
                        tXrX,
                        pred=tXpX[None, None, None, tile_idx],
                    )
                    self._fill_oob(
                        tXrX,
                        tXpX[None, None, None, tile_idx],
                        -tXrX.element_type.inf,
                    )
                else:
                    cute.copy(
                        copy_atom, tXgX[None, None, None, tile_idx], tXrX, pred=tXpX
                    )

                for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                    bin_val = self.to_coarse_key(tXrX[i])
                    atomicAdd(
                        s_histogram.iterator + cutlass.Int32(bin_val),
                        val_one,
                        scope="cta",
                    )
            cute.arch.barrier()

            # 1.2 and 1.3 Suffix sum to find threshold and find threshold bin
            self.prefix_sum_and_find_threshold_coarse(
                tidx,
                s_histogram,
                s_warp_sums,
                num_warps,
                s_threshold_bin_id,
                s_num_input,
                s_counter,
                s_last_remain,
                topk_remaining,
                s_num_input_idx=0,
            )

            threshold_bin = cutlass.Int32(s_threshold_bin_id[0])
            if threshold_bin > 0:
                topk_remaining -= cutlass.Int32(s_histogram[threshold_bin - 1])

            # 1.4 Collect indices
            if topk_remaining == cutlass.Int32(0):
                # Collect indices where bin > threshold
                for tile_idx in range(self.num_sub_tiles):
                    if cutlass.const_expr(tXpX is not None):
                        cute.copy(
                            copy_atom,
                            tXgX[None, None, None, tile_idx],
                            tXrX,
                            pred=tXpX[None, None, None, tile_idx],
                        )
                        self._fill_oob(
                            tXrX,
                            tXpX[None, None, None, tile_idx],
                            -tXrX.element_type.inf,
                        )
                    else:
                        cute.copy(
                            copy_atom,
                            tXgX[None, None, None, tile_idx],
                            tXrX,
                            pred=tXpX,
                        )

                    for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                        cur_tXcX = tXcX[None, None, None, tile_idx]
                        bin_val = self.to_coarse_key(tXrX[i])
                        if bin_val < threshold_bin:
                            pos = atomicAdd(s_counter.iterator, val_one, scope="cta")
                            idx = cutlass.Int32(
                                cur_tXcX[i // vec_size][1] + i % vec_size
                            )
                            s_indices[pos] = idx
                cute.arch.barrier()

            else:
                # Reset histogram for refinement
                if tidx < self.radix + 1:
                    s_histogram[tidx] = 0
                cute.arch.barrier()

                # Filter and build refinement histogram
                for tile_idx in range(self.num_sub_tiles):
                    if cutlass.const_expr(tXpX is not None):
                        cute.copy(
                            copy_atom,
                            tXgX[None, None, None, tile_idx],
                            tXrX,
                            pred=tXpX[None, None, None, tile_idx],
                        )
                        self._fill_oob(
                            tXrX,
                            tXpX[None, None, None, tile_idx],
                            -tXrX.element_type.inf,
                        )
                    else:
                        cute.copy(
                            copy_atom,
                            tXgX[None, None, None, tile_idx],
                            tXrX,
                            pred=tXpX,
                        )

                    for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                        raw_input = tXrX[i]
                        bin_val = self.to_coarse_key(raw_input)
                        cur_tXcX = tXcX[None, None, None, tile_idx]
                        idx = cutlass.Int32(cur_tXcX[i // vec_size][1] + i % vec_size)
                        if bin_val < threshold_bin:
                            pos = atomicAdd(s_counter.iterator, val_one, scope="cta")
                            s_indices[pos] = idx
                        elif bin_val == threshold_bin:
                            pos = atomicAdd(s_num_input.iterator, val_one, scope="cta")
                            if pos < self.filtered_topk_smem_input_size:
                                s_input_idx[0, pos] = idx
                                ordered = self.to_ordered(raw_input)
                                sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                                atomicAdd(
                                    s_histogram.iterator + cutlass.Int32(sub_bin),
                                    val_one,
                                    scope="cta",
                                )
                cute.arch.barrier()

                # Phase 2: Refinement rounds
                run_next_round = True
                for round in range(self.num_refine_rounds):
                    if run_next_round:
                        r_idx = round % 2
                        num_input = min(
                            s_num_input[r_idx], self.filtered_topk_smem_input_size
                        )

                        self.prefix_sum_and_find_threshold_fine_grained(
                            tidx,
                            s_histogram,
                            s_warp_sums,
                            num_warps,
                            s_threshold_bin_id,
                            s_num_input,
                            s_counter,
                            s_last_remain,
                            topk_remaining,
                            s_num_input_idx=r_idx ^ 1,
                        )

                        threshold = cutlass.Int32(s_threshold_bin_id[0])
                        if threshold > 0:
                            topk_remaining -= cutlass.Int32(s_histogram[threshold - 1])
                        offset = self.first_refine_shift - round * 8
                        is_last_round = round == self.num_refine_rounds - 1

                        if topk_remaining == cutlass.Int32(0):
                            for i in range(tidx, num_input, self.num_threads_per_cta):
                                idx = s_input_idx[r_idx, i]
                                bin_val = (self.to_ordered(score[idx]) >> offset) & 0xFF
                                if bin_val < threshold:
                                    pos = atomicAdd(
                                        s_counter.iterator, val_one, scope="cta"
                                    )
                                    s_indices[pos] = idx
                            cute.arch.barrier()
                            run_next_round = False
                        else:
                            # Reset histogram
                            if tidx < self.radix + 1:
                                s_histogram[tidx] = 0
                            cute.arch.barrier()

                            for i in range(tidx, num_input, self.num_threads_per_cta):
                                idx = s_input_idx[r_idx, i]
                                raw_input = score[idx]
                                bin_val = (self.to_ordered(raw_input) >> offset) & 0xFF
                                if bin_val < threshold:
                                    pos = atomicAdd(
                                        s_counter.iterator, val_one, scope="cta"
                                    )
                                    s_indices[pos] = idx
                                elif bin_val == threshold:
                                    if is_last_round:
                                        cur_pos = atomicAdd(
                                            s_last_remain.iterator,
                                            val_one_negative,
                                            scope="cta",
                                        )
                                        if cur_pos > 0:
                                            s_indices[self.top_k - cur_pos] = idx
                                    else:
                                        cur_pos = atomicAdd(
                                            s_num_input.iterator + (r_idx ^ 1),
                                            val_one,
                                            scope="cta",
                                        )
                                        if cur_pos < self.filtered_topk_smem_input_size:
                                            s_input_idx[r_idx ^ 1, cur_pos] = idx
                                            bin32 = self.to_ordered(raw_input)
                                            sub_bin = (bin32 >> (offset - 8)) & 0xFF
                                            atomicAdd(
                                                s_histogram.iterator
                                                + cutlass.Int32(sub_bin),
                                                val_one,
                                                scope="cta",
                                            )
                            cute.arch.barrier()

            # Phase 3: Output phase
            vecsize_out = cutlass.const_expr(
                min(
                    self.top_k,
                    cute.ceil_div(self.top_k, self.num_threads_per_cta),
                    self.num_copy_bits // self.dtype.width,
                    2,
                )
            )
            assert self.top_k % vecsize_out == 0

            nvec_per_thread = cutlass.const_expr(
                cute.ceil_div(self.top_k, vecsize_out * self.num_threads_per_cta)
            )
            topk_vals = cute.make_fragment((vecsize_out, nvec_per_thread), self.dtype)
            topk_indices = cute.make_fragment(
                (vecsize_out, nvec_per_thread), cutlass.Int32
            )

            stride = self.num_threads_per_cta * vecsize_out
            for i in cutlass.range(nvec_per_thread, unroll_full=True):
                idx = i * stride + tidx % self.num_threads_per_cta * vecsize_out
                if idx < self.top_k:
                    for v in cutlass.range(vecsize_out, unroll_full=True):
                        index = s_indices[idx + v]
                        topk_indices[v, i] = index
                        if cutlass.const_expr(self.return_val):
                            topk_vals[v, i] = score[index]
            # [atom, rest_vec]
            mIndices_store = cute.tiled_divide(dst, (vecsize_out,))
            if cutlass.const_expr(self.return_val):
                mValues_store = cute.tiled_divide(dst_values, (vecsize_out,))
            # i represents the index of the vector in the output
            for i in cutlass.range(cute.size(topk_vals.shape, [1]), unroll_full=True):
                col = i * self.num_threads_per_cta + tidx % self.num_threads_per_cta
                if col < self.top_k // vecsize_out:
                    cute.autovec_copy(topk_indices[None, i], mIndices_store[None, col])
                    if cutlass.const_expr(self.return_val):
                        cute.autovec_copy(topk_vals[None, i], mValues_store[None, col])

    def _get_tiled_copy(self):
        """Get tiled copy configuration for vectorized loads."""
        threads_per_row = self.num_threads_per_cta
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            threshold_elems = 32768
        elif cutlass.const_expr(
            self.dtype == cutlass.Float16 or self.dtype == cutlass.BFloat16
        ):
            threshold_elems = 65536
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        vecs_per_thread = cute.ceil_div(
            min(self.num_cols, threshold_elems) // self.vec_size,
            threads_per_row,
        )
        tiler_mn = (
            1,
            self.vec_size * vecs_per_thread * threads_per_row,
        )

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=self.num_copy_bits,
        )

        thr_layout = cute.make_ordered_layout(
            (1, threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, self.vec_size))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        return copy_atom, tiled_copy, tiler_mn

    @cute.jit
    def predicate_k(self, tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
        """Create predicate tensor for boundary checking."""
        tApA = cute.make_fragment(
            cute.make_layout(
                (
                    cute.size(tAcA, mode=[0, 1]),
                    cute.size(tAcA, mode=[1]),
                    cute.size(tAcA, mode=[2]),
                    cute.size(tAcA, mode=[3]),
                ),
                stride=(cute.size(tAcA, mode=[2]), 0, 1, cute.size(tAcA, mode=[2])),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tApA.shape[0]):
            for rest_k in cutlass.range_constexpr(tApA.shape[2]):
                for rest_t in cutlass.range_constexpr(tApA.shape[3]):
                    tApA[rest_v, 0, rest_k, rest_t] = cute.elem_less(
                        tAcA[(0, rest_v), 0, rest_k, rest_t][1], limit
                    )
        return tApA

    @cute.jit
    def _fill_oob(
        self, tXrX: cute.Tensor, tXpX: cute.Tensor, fill_value: cute.Numeric
    ) -> None:
        """Fill out-of-bounds values in register tensor."""
        tXrX_fill = cute.make_fragment_like(tXrX[(None, 0), None, 0])
        tXrX_fill.fill(fill_value)
        for rest_v in cutlass.range_constexpr(tXrX.shape[0][1]):
            for rest_k in cutlass.range_constexpr(tXrX.shape[2]):
                if cutlass.const_expr(tXpX is not None):
                    if not tXpX[0, rest_v, rest_k]:
                        cute.autovec_copy(tXrX_fill, tXrX[(None, rest_v), None, rest_k])

    @cute.jit
    def __call__(
        self, input_values, output_indices, output_values, stream: cuda.CUstream
    ):
        """Host function for the filtered topk kernel."""
        num_rows = input_values.shape[0]
        # Each CTA processes one row of input
        blocks = (num_rows, 1, 1)
        copy_atom, tiled_copy, tiler_mn = self._get_tiled_copy()
        self.filtered_topk_kernel(
            input_values, output_indices, output_values, tiler_mn, copy_atom, tiled_copy
        ).launch(
            grid=blocks,
            block=(tiled_copy.size, 1, 1),
            stream=stream,
        )
        return


# =============================================================================
# Compiled Kernel Cache and Public API
# =============================================================================

# Cache for compiled kernels
_compiled_kernel_cache = {}


@functools.cache
def _get_compiled_kernel(
    num_cols: int,
    top_k: int,
    dtype_str: str,
) -> Callable:
    """Get a compiled kernel closure that takes torch.Tensor directly.

    Uses TVM-FFI for efficient tensor passing without manual pointer construction.
    """
    # Map dtype string to cutlass type
    dtype_map = {
        "float32": cutlass.Float32,
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
    }
    dtype = dtype_map[dtype_str]

    # Create kernel instance
    kernel_obj = FilteredTopKKernel(
        dtype=dtype,
        num_cols=num_cols,
        top_k=top_k,
        num_copy_bits=256,
        return_val=True,
    )

    # Use symbolic size for dynamic batch dimension
    sym_n = cute.sym_int()

    # Create fake tensors for compilation with TVM-FFI
    input_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_n, num_cols), stride_order=(1, 0), assumed_align=32
    )
    output_indices_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_n, top_k),
        stride_order=(1, 0),
    )
    output_values_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_n, top_k),
        stride_order=(1, 0),
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Compile with TVM-FFI enabled
    compiled_kernel = cute.compile(
        kernel_obj,
        input_fake,
        output_indices_fake,
        output_values_fake,
        fake_stream,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        input_values: torch.Tensor,
        output_indices: torch.Tensor,
        output_values: torch.Tensor,
    ) -> None:
        """Runtime API that passes torch tensors directly via TVM-FFI."""
        nonlocal compiled_kernel
        compiled_kernel(
            input_values,
            output_indices,
            output_values,
        )

    return tensor_api


def can_use_filtered_topk(num_cols: int, top_k: int) -> bool:
    """Check if filtered top-k can be used for the given parameters.

    Args:
        num_cols: Number of columns (vocabulary size)
        top_k: Number of top elements to select

    Returns:
        True if filtered top-k can be used, False otherwise
    """
    # Check basic constraints
    if num_cols < 256:
        return False
    if top_k > 2048:
        return False

    # Check alignment (256-bit vectorized loads require proper alignment)
    # For float32: vec_size = 256/32 = 8
    # For float16/bfloat16: vec_size = 256/16 = 16
    return True


@flashinfer_api
def filtered_topk(
    input: torch.Tensor,
    k: int,
    sorted: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filtered top-k selection using CuTe-DSL.

    This function selects the top-k largest elements from each row of the input
    tensor using a radix-based filter algorithm implemented in CuTe-DSL.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape ``(batch_size, num_cols)`` containing the values to select from.
        Supported dtypes: ``float32``, ``float16``, ``bfloat16``.
        Requires: num_cols >= 256.
    k : int
        Number of top elements to select from each row.
        Requires: k <= 2048.
    sorted : bool, optional
        If True, the returned top-k elements will be sorted in descending order.
        Default is False (unsorted, which is faster).

    Returns
    -------
    values : torch.Tensor
        Tensor of shape ``(batch_size, k)`` containing the top-k values.
        Same dtype as input.
    indices : torch.Tensor
        Tensor of shape ``(batch_size, k)`` with int64 dtype containing the
        indices of the top-k elements.

    Raises
    ------
    ValueError
        If num_cols < 256 or k > 2048.
    """
    batch_size = input.size(0)
    num_cols = input.size(1)
    device = input.device

    # Validate constraints
    if num_cols < 256:
        raise ValueError(
            f"filtered_topk requires num_cols >= 256, but got {num_cols}. "
            "Use backend='cuda' for smaller inputs."
        )
    if k > 2048:
        raise ValueError(
            f"filtered_topk requires k <= 2048, but got {k}. "
            "Use backend='cuda' for larger k values."
        )

    # Get dtype string
    dtype_map = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    if input.dtype not in dtype_map:
        raise ValueError(
            f"Unsupported dtype {input.dtype}. "
            "Supported dtypes: float32, float16, bfloat16."
        )
    dtype_str = dtype_map[input.dtype]

    # Allocate output tensors
    output_values = torch.empty(batch_size, k, dtype=input.dtype, device=device)
    output_indices = torch.empty(batch_size, k, dtype=torch.int32, device=device)

    # Get compiled kernel
    tensor_api = _get_compiled_kernel(num_cols, k, dtype_str)

    # Run kernel
    tensor_api(
        input.contiguous(),
        output_indices,
        output_values,
    )

    # Convert indices to int64 for compatibility
    indices = output_indices.long()

    if sorted:
        # Sort within each row by value (descending)
        sorted_values, sort_indices = torch.sort(output_values, dim=-1, descending=True)
        sorted_indices = torch.gather(indices, dim=-1, index=sort_indices)
        return sorted_values, sorted_indices

    return output_values, indices

