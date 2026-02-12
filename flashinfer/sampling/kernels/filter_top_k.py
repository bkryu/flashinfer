# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
High-performance radix-based filtered top-k kernel in CuTe DSL.

Ported from ``filter_top_k_padded.py`` for integration into FlashInfer.

Supported data types: Float32, Float16, BFloat16.

Constraints:
* ``num_cols >= vec_size`` (input is padded to be vec_size-aligned).
* ``top_k <= 2048`` and ``top_k`` is even.
* Input tensor is row-major (data contiguous on the N dimension).
* Candidates can be stored in SMEM or spill to GMEM.
* ``num_cols`` is static at compile time (but dynamic at the Python API level
  via ``@functools.cache``).
"""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.utils.distributed import atomicAdd

from .block_scan import block_prefix_sum_kernel, fence_acq_rel_cta


# ---------------------------------------------------------------------------
# Helper: bit-level conversions
# ---------------------------------------------------------------------------


def half_as_ushort(half_val):
    """Interpret FP16 value as uint16 bit pattern"""
    return llvm.bitcast(cutlass.Uint16.mlir_type, half_val.ir_value())


def float_as_uint32(float_val):
    """Interpret FP32 value as uint32 bit pattern"""
    return llvm.bitcast(cutlass.Uint32.mlir_type, float_val.ir_value())


# ---------------------------------------------------------------------------
# Kernel class
# ---------------------------------------------------------------------------


class FilteredTopKKernel:
    """CuTe DSL radix-based filtered top-k kernel.

    Parameters
    ----------
    dtype : cutlass.Numeric
        Element data type (Float32, Float16, BFloat16).
    num_cols : int
        Number of columns (vocabulary size) -- static at compile time.
    top_k : int
        Number of top elements to select (must be even, <= 2048).
    num_copy_bits : int
        Bits per vectorised load (default 256).
    is_prefill : bool
        Whether the kernel is in the prefill regime (more rows, smaller
        SMEM input buffer per row).
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        num_cols: int,
        top_k: int,
        num_copy_bits: int = 256,
        is_prefill: bool = True,
    ):
        self.dtype = dtype
        self.num_cols = num_cols
        self.top_k = top_k
        self.num_copy_bits = num_copy_bits
        self.is_prefill = is_prefill

        # Always return values (FlashInfer API contract)
        self.return_val = True

        self.filtered_topk_max_k = 2048
        # 8 bits for radix-based filter.
        self.radix = 256

        if cutlass.const_expr(self.dtype == cutlass.Float32):
            self.num_buffer_smem_input_idx = 2
        else:
            self.num_buffer_smem_input_idx = 1

        # 65536 is the max index value for uint16.
        if cutlass.const_expr(self.num_cols <= 65536):
            self.index_type = cutlass.Uint16
            if cutlass.const_expr(self.num_buffer_smem_input_idx == 2):
                self.max_smem_input_size = 32 * 1024
            else:
                self.max_smem_input_size = 64 * 1024
        else:
            self.index_type = cutlass.Uint32
            if cutlass.const_expr(self.num_buffer_smem_input_idx == 2):
                self.max_smem_input_size = 16 * 1024
            else:
                self.max_smem_input_size = 32 * 1024

        if cutlass.const_expr(self.is_prefill):
            if self.num_cols >= 262144:
                self.filtered_topk_smem_input_size = 4096
            elif self.num_cols >= 131072:
                self.filtered_topk_smem_input_size = 3072
            elif self.num_cols >= 65536:
                self.filtered_topk_smem_input_size = 2048
            elif self.num_cols >= 16384:
                self.filtered_topk_smem_input_size = 1024
            elif self.num_cols >= 8192:
                self.filtered_topk_smem_input_size = 512
            else:
                self.filtered_topk_smem_input_size = 256
        else:
            self.filtered_topk_smem_input_size = min(
                self.max_smem_input_size, self.num_cols
            )

        if cutlass.const_expr(self.num_cols > self.filtered_topk_smem_input_size):
            self.enable_gmem_store = True
        else:
            self.enable_gmem_store = False

        self.vec_size = num_copy_bits // dtype.width
        assert self.num_cols >= self.vec_size, (
            f"num_cols must be >= vec_size, but got {self.num_cols} and {self.vec_size}"
        )
        if cutlass.const_expr(
            dtype not in [cutlass.Float32, cute.BFloat16, cutlass.Float16]
        ):
            raise ValueError(f"Unsupported dtype: {dtype}")

        if cutlass.const_expr(self.is_prefill):
            self.num_threads_per_cta = 512
        else:
            if cutlass.const_expr(dtype == cutlass.Float32):
                if self.num_cols >= self.vec_size * 1024:
                    self.num_threads_per_cta = 1024
                else:
                    if cutlass.const_expr(
                        self.num_cols > 2048 and self.num_cols < self.vec_size * 1024
                    ):
                        self.num_threads_per_cta = 512
                    else:
                        self.num_threads_per_cta = 256
            else:
                if self.num_cols >= 43008:
                    self.num_threads_per_cta = 1024
                else:
                    if cutlass.const_expr(
                        self.num_cols > 4096 and self.num_cols < 43008
                    ):
                        self.num_threads_per_cta = 512
                    else:
                        self.num_threads_per_cta = 256

        # radix-based filter parameters.
        if cutlass.const_expr(dtype == cutlass.Float32):
            self.ordered_type = cute.Uint32
            self.first_refine_shift = 24
            self.num_refine_rounds = 4
        elif cutlass.const_expr(dtype in [cutlass.Float16, cute.BFloat16]):
            self.ordered_type = cute.Uint16
            self.first_refine_shift = 0
            self.num_refine_rounds = 1

    # ------------------------------------------------------------------
    # Bit-conversion helpers
    # ------------------------------------------------------------------

    @cute.jit
    def to_coarse_key(self, x):
        """Convert to coarse 8-bit key for histogram"""
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            h = x.to(cutlass.Float16)
            bits = half_as_ushort(h)
            key = cutlass.Uint16(0)
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
            return cute.Uint8((key >> 8) & 0xFF)
        else:
            if cutlass.const_expr(self.dtype == cutlass.Float16):
                bits = half_as_ushort(x)
            else:  # BFloat16
                bits = half_as_ushort(x)
            key = cute.Uint16(0)
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
            return cute.Uint8((key >> 8) & 0xFF)

    @cute.jit
    def to_ordered(self, x):
        """Convert to ordered integer for comparison"""
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

    # ------------------------------------------------------------------
    # Prefix-sum helpers
    # ------------------------------------------------------------------

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
        g_num_input,
        s_num_input_idx=0,
    ):
        if cutlass.const_expr(self.radix <= self.num_threads_per_cta):
            previous = 0
            if tidx < cutlass.Int32(self.radix):
                val = s_histogram[tidx]
                val, total_sum = block_prefix_sum_kernel(
                    val, s_warp_sums, tidx, self.radix, num_warps, barrier_id=1
                )
                s_histogram[tidx] = val
                cute.arch.barrier(barrier_id=1, number_of_threads=self.radix)

                if tidx > 0:
                    previous = s_histogram[tidx - 1]
                if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                    s_threshold_bin_id[0] = tidx
                    s_num_input[s_num_input_idx] = 0
                    if cutlass.const_expr(self.enable_gmem_store):
                        g_num_input[s_num_input_idx] = 0
                    s_counter[0] = 0
            cute.arch.barrier()
        else:
            assert self.radix % self.num_threads_per_cta == 0
            previous_sum = 0
            val = 0
            total_sum = 0
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
            cute.arch.barrier()

            previous = 0
            run_loop = True
            if tidx > 0:
                previous = s_histogram[tidx - 1]
            if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                s_threshold_bin_id[0] = tidx
                s_num_input[s_num_input_idx] = 0
                if cutlass.const_expr(self.enable_gmem_store):
                    g_num_input[s_num_input_idx] = 0
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
                            if cutlass.const_expr(self.enable_gmem_store):
                                g_num_input[s_num_input_idx] = 0
                            s_counter[0] = 0
                            run_next_loop = False
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
        g_num_input,
        s_num_input_idx=0,
    ):
        if cutlass.const_expr(self.radix <= self.num_threads_per_cta):
            previous = 0
            if tidx < cutlass.Int32(self.radix):
                val = s_histogram[tidx]
                val, total_sum = block_prefix_sum_kernel(
                    val, s_warp_sums, tidx, self.radix, num_warps, barrier_id=1
                )
                s_histogram[tidx] = val
                cute.arch.barrier(barrier_id=1, number_of_threads=self.radix)

                if tidx > 0:
                    previous = s_histogram[tidx - 1]
                if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                    s_threshold_bin_id[0] = tidx
                    s_num_input[s_num_input_idx] = 0
                    if cutlass.const_expr(self.enable_gmem_store):
                        g_num_input[s_num_input_idx] = 0
                    s_last_remain[0] = topk_remaining - previous
            cute.arch.barrier()
        else:
            assert self.radix % self.num_threads_per_cta == 0
            previous_sum = 0
            val = 0
            total_sum = 0
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
            cute.arch.barrier()

            previous = 0
            run_loop = True
            if tidx > 0:
                previous = s_histogram[tidx - 1]
            if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                s_threshold_bin_id[0] = tidx
                s_num_input[s_num_input_idx] = 0
                if cutlass.const_expr(self.enable_gmem_store):
                    g_num_input[s_num_input_idx] = 0
                s_last_remain[0] = topk_remaining - previous
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
                            if cutlass.const_expr(self.enable_gmem_store):
                                g_num_input[s_num_input_idx] = 0
                            s_last_remain[0] = topk_remaining - previous
                            run_next_loop = False
            cute.arch.barrier()

    # ------------------------------------------------------------------
    # Main kernel
    # ------------------------------------------------------------------

    @cute.kernel
    def filtered_topk_kernel(
        self,
        input: cute.Tensor,
        extra_buffer: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
        tiler_mn: cute.Shape,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        aligned_size: cutlass.Constexpr,
        left_size: cutlass.Constexpr,
    ):
        """CuTe DSL implementation of TopK kernel based on radix-based filter algorithm."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        length = self.num_cols
        score = input[bidx, None]
        dst = output_indices[bidx, None]
        if cutlass.const_expr(self.enable_gmem_store):
            buffer = extra_buffer[bidx, None, None]
        dst_values = output_values[bidx, None]

        has_left_part = cutlass.const_expr(self.num_cols % self.vec_size != 0)
        shape = input.shape

        idX = cute.make_identity_tensor((shape[0], aligned_size))
        input_tensor = cute.make_tensor(
            input.iterator,
            cute.make_layout((shape[0], aligned_size), stride=input.stride),
        )

        gX, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, None)) for mT in (input_tensor, idX)
        ]

        self.num_sub_tiles = gX.shape[2]

        thr_copy = tiled_copy.get_slice(tidx)

        tXgX = thr_copy.partition_S(gX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None, None]
        tXrX = cute.make_fragment_like(tXgX[None, None, None, 0])

        is_even_N = cutlass.const_expr(self.num_cols % tiler_mn[1] == 0)
        tXpX = (
            None
            if is_even_N
            else self.predicate_k(thr_copy.partition_S(cX), limit=aligned_size)
        )

        # Trivial case: length <= top_k
        if cutlass.const_expr(length <= self.top_k):
            for i in range(tidx, self.top_k, self.num_threads_per_cta):
                if i < length:
                    dst[i] = i
                    dst_values[i] = score[i]
                else:
                    dst[i] = -1
                    dst_values[i] = self.dtype(0.0)
        else:
            # Shared memory allocation
            smem = utils.SmemAllocator()
            s_histogram_buf_layout = cute.make_ordered_layout(
                (self.radix + 1), order=(0)
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
            if cutlass.const_expr(self.enable_gmem_store):
                g_num_input = smem.allocate_tensor(
                    element_type=cutlass.Int32,
                    layout=cute.make_ordered_layout((2), order=(0)),
                    byte_alignment=128,
                )
            else:
                g_num_input = None
            s_indices = smem.allocate_tensor(
                element_type=self.index_type,
                layout=cute.make_ordered_layout((self.filtered_topk_max_k,), order=(0)),
                byte_alignment=128,
            )
            s_input_idx = smem.allocate_tensor(
                element_type=self.index_type,
                layout=cute.make_ordered_layout(
                    (
                        self.num_buffer_smem_input_idx,
                        self.filtered_topk_smem_input_size,
                    ),
                    order=(1, 0),
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

            topk_remaining = self.top_k

            val_one = cutlass.Int32(1)
            val_one_negative = cutlass.Int32(-1)

            # Stage 1: Coarse histogram.
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
                    )

            # for left part (left_size)
            if cutlass.const_expr(has_left_part):
                for j in range(tidx, left_size, self.num_threads_per_cta):
                    col_idx = cutlass.Int32(aligned_size + j)
                    raw = score[col_idx]
                    bin_val = self.to_coarse_key(raw)
                    atomicAdd(
                        s_histogram.iterator + cutlass.Int32(bin_val),
                        val_one,
                    )

            cute.arch.barrier()

            # 1.2 and 1.3  Suffix sum to find threshold and find threshold bin
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
                g_num_input,
                s_num_input_idx=0,
            )

            threshold_bin = s_threshold_bin_id[0]
            if threshold_bin > 0:
                topk_remaining -= s_histogram[threshold_bin - 1]

            # 1.4 Collect indices
            if topk_remaining == 0:
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
                            copy_atom, tXgX[None, None, None, tile_idx], tXrX, pred=tXpX
                        )

                    for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                        cur_tXcX = tXcX[None, None, None, tile_idx]
                        bin_val = self.to_coarse_key(tXrX[i])
                        if bin_val < threshold_bin:
                            pos = atomicAdd(s_counter.iterator, val_one)
                            idx = self.index_type(
                                cur_tXcX[i // vec_size][1] + i % vec_size
                            )
                            s_indices[pos] = idx

                # for left part (left_size)
                if cutlass.const_expr(has_left_part):
                    for j in range(tidx, left_size, self.num_threads_per_cta):
                        col_idx = cutlass.Int32(aligned_size + j)
                        raw = score[col_idx]
                        bin_val = self.to_coarse_key(raw)
                        if bin_val < threshold_bin:
                            pos = atomicAdd(s_counter.iterator, val_one)
                            idx = self.index_type(col_idx)
                            s_indices[pos] = idx

                cute.arch.barrier()

            else:
                # Reset histogram for refinement
                cute.arch.barrier()
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
                            copy_atom, tXgX[None, None, None, tile_idx], tXrX, pred=tXpX
                        )

                    for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                        raw_input = tXrX[i]
                        bin_val = self.to_coarse_key(raw_input)
                        cur_tXcX = tXcX[None, None, None, tile_idx]
                        idx = self.index_type(cur_tXcX[i // vec_size][1] + i % vec_size)
                        if bin_val < threshold_bin:
                            pos = atomicAdd(s_counter.iterator, val_one)
                            s_indices[pos] = idx
                        elif bin_val == threshold_bin:
                            pos = atomicAdd(s_num_input.iterator, val_one)
                            if cutlass.const_expr(self.enable_gmem_store):
                                if pos < self.filtered_topk_smem_input_size:
                                    s_input_idx[0, pos] = idx
                                else:
                                    buffer_pos = atomicAdd(
                                        g_num_input.iterator,
                                        val_one,
                                    )
                                    buffer[0, buffer_pos] = cutlass.Int32(idx)
                            else:
                                if pos < self.filtered_topk_smem_input_size:
                                    s_input_idx[0, pos] = idx
                            ordered = self.to_ordered(raw_input)
                            sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                            atomicAdd(
                                s_histogram.iterator + cutlass.Int32(sub_bin),
                                val_one,
                            )

                # for left part
                if cutlass.const_expr(has_left_part):
                    for j in range(tidx, left_size, self.num_threads_per_cta):
                        col_idx = cutlass.Int32(aligned_size + j)
                        raw = score[col_idx]
                        bin_val = self.to_coarse_key(raw)
                        if bin_val < threshold_bin:
                            pos = atomicAdd(s_counter.iterator, val_one)
                            idx = self.index_type(col_idx)
                            s_indices[pos] = idx
                        elif bin_val == threshold_bin:
                            pos = atomicAdd(
                                s_num_input.iterator + cutlass.Int32(0),
                                val_one,
                            )
                            if cutlass.const_expr(self.enable_gmem_store):
                                if pos < self.filtered_topk_smem_input_size:
                                    s_input_idx[0, pos] = self.index_type(col_idx)
                                else:
                                    buffer_pos = atomicAdd(
                                        g_num_input.iterator,
                                        val_one,
                                    )
                                    buffer[0, buffer_pos] = cutlass.Int32(col_idx)
                            else:
                                if pos < self.filtered_topk_smem_input_size:
                                    s_input_idx[0, pos] = self.index_type(col_idx)
                            ordered = self.to_ordered(raw)
                            sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                            atomicAdd(
                                s_histogram.iterator + cutlass.Int32(sub_bin),
                                val_one,
                            )
                fence_acq_rel_cta()
                cute.arch.barrier()

                # Phase 2: Refinement rounds
                run_next_round = True
                for round in range(self.num_refine_rounds):
                    if run_next_round:
                        r_idx = round % 2
                        num_input = min(
                            s_num_input[r_idx], self.filtered_topk_smem_input_size
                        )
                        if cutlass.const_expr(self.enable_gmem_store):
                            cur_g_num_input = g_num_input[r_idx]

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
                            g_num_input,
                            s_num_input_idx=r_idx ^ 1,
                        )

                        threshold = s_threshold_bin_id[0]
                        if threshold > 0:
                            topk_remaining -= s_histogram[threshold - 1]
                        offset = self.first_refine_shift - round * 8
                        is_last_round = round == self.num_refine_rounds - 1

                        if topk_remaining == 0:
                            for i in range(tidx, num_input, self.num_threads_per_cta):
                                idx = s_input_idx[r_idx, i]
                                idx_int32 = cutlass.Int32(cutlass.Uint32(idx))
                                bin_val = (
                                    self.to_ordered(score[idx_int32]) >> offset
                                ) & 0xFF
                                if bin_val < threshold:
                                    pos = atomicAdd(s_counter.iterator, val_one)
                                    s_indices[pos] = idx
                            if cutlass.const_expr(self.enable_gmem_store):
                                for i in range(
                                    tidx,
                                    cur_g_num_input,
                                    self.num_threads_per_cta,
                                ):
                                    idx = buffer[r_idx, i]
                                    bin_val = (
                                        self.to_ordered(score[idx]) >> offset
                                    ) & 0xFF
                                    if bin_val < threshold:
                                        pos = atomicAdd(
                                            s_counter.iterator,
                                            val_one,
                                        )
                                        s_indices[pos] = self.index_type(idx)
                            cute.arch.barrier()
                            run_next_round = False
                        else:
                            # Reset histogram
                            cute.arch.barrier()
                            if tidx < self.radix + 1:
                                s_histogram[tidx] = 0
                            cute.arch.barrier()

                            for i in range(tidx, num_input, self.num_threads_per_cta):
                                idx = s_input_idx[r_idx, i]
                                idx_int32 = cutlass.Int32(cutlass.Uint32(idx))
                                raw_input = score[idx_int32]
                                bin_val = (self.to_ordered(raw_input) >> offset) & 0xFF
                                if bin_val < threshold:
                                    pos = atomicAdd(s_counter.iterator, val_one)
                                    s_indices[pos] = idx
                                elif bin_val == threshold:
                                    if is_last_round:
                                        cur_pos = atomicAdd(
                                            s_last_remain.iterator,
                                            val_one_negative,
                                        )
                                        if cur_pos > 0:
                                            s_indices[self.top_k - cur_pos] = idx
                                    else:
                                        cur_pos = atomicAdd(
                                            s_num_input.iterator + (r_idx ^ 1),
                                            val_one,
                                        )
                                        if cur_pos < self.filtered_topk_smem_input_size:
                                            s_input_idx[r_idx ^ 1, cur_pos] = idx
                                            bin32 = self.to_ordered(raw_input)
                                            sub_bin = (bin32 >> (offset - 8)) & 0xFF
                                            atomicAdd(
                                                s_histogram.iterator
                                                + cutlass.Int32(sub_bin),
                                                val_one,
                                            )
                            if cutlass.const_expr(self.enable_gmem_store):
                                cute.arch.barrier()
                                for i in range(
                                    tidx, cur_g_num_input, self.num_threads_per_cta
                                ):
                                    idx_int32 = buffer[r_idx, i]
                                    raw_input = score[idx_int32]
                                    idx = self.index_type(idx_int32)
                                    bin_val = (
                                        self.to_ordered(raw_input) >> offset
                                    ) & 0xFF
                                    if bin_val < threshold:
                                        pos = atomicAdd(
                                            s_counter.iterator,
                                            val_one,
                                        )
                                        s_indices[pos] = idx
                                    elif bin_val == threshold:
                                        if is_last_round:
                                            cur_pos = atomicAdd(
                                                s_last_remain.iterator,
                                                val_one_negative,
                                            )
                                            if cur_pos > 0:
                                                s_indices[self.top_k - cur_pos] = idx
                                        else:
                                            cur_pos = atomicAdd(
                                                s_num_input.iterator + (r_idx ^ 1),
                                                val_one,
                                            )
                                            if cutlass.const_expr(
                                                self.enable_gmem_store
                                            ):
                                                if (
                                                    cur_pos
                                                    < self.filtered_topk_smem_input_size
                                                ):
                                                    s_input_idx[r_idx ^ 1, cur_pos] = (
                                                        idx
                                                    )
                                                else:
                                                    buffer_pos = atomicAdd(
                                                        g_num_input.iterator
                                                        + (r_idx ^ 1),
                                                        val_one,
                                                    )
                                                    buffer[r_idx ^ 1, buffer_pos] = (
                                                        idx_int32
                                                    )
                                            else:
                                                if (
                                                    cur_pos
                                                    < self.filtered_topk_smem_input_size
                                                ):
                                                    s_input_idx[r_idx ^ 1, cur_pos] = (
                                                        idx
                                                    )
                                            bin32 = self.to_ordered(raw_input)
                                            sub_bin = (bin32 >> (offset - 8)) & 0xFF
                                            atomicAdd(
                                                s_histogram.iterator
                                                + cutlass.Int32(sub_bin),
                                                val_one,
                                            )

                            fence_acq_rel_cta()
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
                        index = cutlass.Int32(cutlass.Uint32(index))
                        topk_indices[v, i] = index
                        topk_vals[v, i] = score[index]
            # [atom, rest_vec]
            mIndices_store = cute.tiled_divide(dst, (vecsize_out,))
            mValues_store = cute.tiled_divide(dst_values, (vecsize_out,))
            for i in cutlass.range(cute.size(topk_vals.shape, [1]), unroll_full=True):
                col = i * self.num_threads_per_cta + tidx % self.num_threads_per_cta
                if col < self.top_k // vecsize_out:
                    cute.autovec_copy(topk_indices[None, i], mIndices_store[None, col])
                    cute.autovec_copy(topk_vals[None, i], mValues_store[None, col])

    # ------------------------------------------------------------------
    # Tiled copy setup
    # ------------------------------------------------------------------

    def _get_tiled_copy(self):
        threads_per_row = self.num_threads_per_cta
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            if cutlass.const_expr(self.is_prefill):
                threshold_elems = 4096
            else:
                threshold_elems = 32768
        elif cutlass.const_expr(
            self.dtype == cutlass.Float16 or self.dtype == cutlass.BFloat16
        ):
            if cutlass.const_expr(self.is_prefill):
                threshold_elems = 4096
            else:
                threshold_elems = 65536
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        # vectorized load part
        aligned_size = (self.num_cols // self.vec_size) * self.vec_size
        vecs_per_thread = cute.ceil_div(
            cute.ceil_div(min(aligned_size, threshold_elems), self.vec_size),
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

        # left scalar load part
        left_size = self.num_cols - aligned_size
        return (
            copy_atom,
            tiled_copy,
            tiler_mn,
            aligned_size,
            left_size,
        )

    @cute.jit
    def predicate_k(self, tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
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

    # ------------------------------------------------------------------
    # Host-side entry (__call__)
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(self, input_values, buffer, output_indices, output_values, stream):
        """Host function for the filtered topk kernel"""
        num_rows = input_values.shape[0]
        blocks = (num_rows, 1, 1)
        (
            copy_atom,
            tiled_copy,
            tiler_mn,
            aligned_size,
            left_size,
        ) = self._get_tiled_copy()
        self.filtered_topk_kernel(
            input_values,
            buffer,
            output_indices,
            output_values,
            tiler_mn,
            copy_atom,
            tiled_copy,
            aligned_size,
            left_size,
        ).launch(
            grid=blocks,
            block=(tiled_copy.size, 1, 1),
            stream=stream,
        )
        return


# ==========================================================================
# TVM-FFI compilation wrapper (follows rmsnorm_fp4quant.py pattern)
# ==========================================================================

_TORCH_TO_CUTLASS = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


@functools.cache
def _get_compiled_kernel(
    torch_dtype: torch.dtype,
    num_cols: int,
    top_k: int,
    is_prefill: bool,
    num_copy_bits: int = 256,
):
    """Return a compiled ``tensor_api`` closure that accepts ``torch.Tensor``
    arguments directly (via TVM-FFI).

    The result is cached by ``(torch_dtype, num_cols, top_k, is_prefill, num_copy_bits)``.
    """
    cutlass_dtype = _TORCH_TO_CUTLASS[torch_dtype]

    if cutlass_dtype == cutlass.Float32:
        buffer_numbers = 2
    else:
        buffer_numbers = 1

    # Pad num_cols to vec_size alignment for vectorised copy.
    vec_size = num_copy_bits // cutlass_dtype.width
    num_cols_padded = (num_cols + vec_size - 1) // vec_size * vec_size

    kernel_obj = FilteredTopKKernel(
        dtype=cutlass_dtype,
        num_cols=num_cols,
        top_k=top_k,
        num_copy_bits=num_copy_bits,
        is_prefill=is_prefill,
    )

    # Symbolic batch dimension
    sym_n = cute.sym_int()

    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (sym_n, num_cols_padded),
        stride_order=(1, 0),
        assumed_align=32,
    )
    buffer_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_n, buffer_numbers, num_cols_padded),
        stride_order=(2, 1, 0),
        assumed_align=32,
    )
    output_indices_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_n, top_k),
        stride_order=(1, 0),
    )
    output_values_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (sym_n, top_k),
        stride_order=(1, 0),
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        input_fake,
        buffer_fake,
        output_indices_fake,
        output_values_fake,
        stream=stream_fake,
        options="--enable-tvm-ffi",
    )

    # ---- tensor_api closure (matches rmsnorm_fp4quant pattern) ----
    def tensor_api(
        input_values: torch.Tensor,
        buffer_tensor: torch.Tensor,
        output_indices: torch.Tensor,
        output_values: torch.Tensor,
    ) -> None:
        """Call the compiled kernel, passing torch.Tensors via TVM-FFI."""
        nonlocal compiled_kernel
        compiled_kernel(
            input_values,
            buffer_tensor,
            output_indices,
            output_values,
        )

    return tensor_api, buffer_numbers, num_cols_padded


# ==========================================================================
# Public entry-point
# ==========================================================================


def cute_dsl_top_k(
    input: torch.Tensor,
    k: int,
    num_copy_bits: int = 256,
) -> tuple:
    """Run the CuTe DSL radix-based filtered top-k kernel.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape ``(num_rows, num_cols)``.  Supported dtypes:
        ``float32``, ``float16``, ``bfloat16``.
    k : int
        Number of top elements to select per row (must be even, <= 2048).
    num_copy_bits : int
        Bits per vectorised load (default 256).

    Returns
    -------
    output_values : torch.Tensor
        Top-k values, shape ``(num_rows, k)``, same dtype as *input*.
    output_indices : torch.Tensor
        Top-k column indices, shape ``(num_rows, k)``, dtype ``int32``.
    """
    assert input.ndim == 2, f"Expected 2-D input, got {input.ndim}-D"
    assert k % 2 == 0, f"top_k must be even, got {k}"
    assert k <= 2048, f"top_k must be <= 2048, got {k}"

    num_rows, num_cols = input.shape
    is_prefill = num_rows > 148

    tensor_api, buffer_numbers, num_cols_padded = _get_compiled_kernel(
        input.dtype, num_cols, k, is_prefill, num_copy_bits
    )

    # Pad input if needed so that vectorised copy is aligned.
    if num_cols_padded != num_cols:
        padded = torch.full(
            (num_rows, num_cols_padded),
            float("-inf"),
            dtype=input.dtype,
            device=input.device,
        )
        padded[:, :num_cols] = input
        input_padded = padded
    else:
        input_padded = input

    # Allocate output & scratch buffers
    output_indices = torch.empty(num_rows, k, dtype=torch.int32, device=input.device)
    output_values = torch.empty(num_rows, k, dtype=input.dtype, device=input.device)
    buffer_tensor = torch.zeros(
        num_rows,
        buffer_numbers,
        num_cols_padded,
        dtype=torch.int32,
        device=input.device,
    )

    tensor_api(input_padded, buffer_tensor, output_indices, output_values)

    return output_values, output_indices
