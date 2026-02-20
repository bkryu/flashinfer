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
High-performance radix-based filtered top-k kernel in CuTe DSL
with **dynamic column count**.

Supported data types: Float32, Float16, BFloat16.

Constraints:
* ``num_cols`` can be any value -- no padding required.  The kernel reads
  the actual column count from ``input.shape[1]`` at runtime and handles
  alignment internally.
* ``top_k <= 2048`` and ``top_k`` is even.
* Input tensor is row-major (data contiguous on the N dimension).
* Candidates can be stored in SMEM or spill to GMEM.
* ``max_num_cols`` (a power-of-2 ceiling of the actual ``num_cols``) is
  used at compile time for SMEM/thread tuning; a single compiled kernel
  serves all ``num_cols <= max_num_cols``.
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
    """CuTe DSL radix-based filtered top-k kernel with dynamic column count.

    Parameters
    ----------
    dtype : cutlass.Numeric
        Element data type (Float32, Float16, BFloat16).
    max_num_cols : int
        Upper-bound on the number of columns (vocabulary size).  Used at
        compile time for SMEM sizing and thread-count tuning.  The actual
        column count is read from ``input.shape[1]`` at runtime.
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
        max_num_cols: int,
        top_k: int,
        num_copy_bits: int = 256,
        is_prefill: bool = True,
        return_val: bool = True,
    ):
        self.dtype = dtype
        self.max_num_cols = max_num_cols
        self.top_k = top_k
        self.num_copy_bits = num_copy_bits
        self.is_prefill = is_prefill
        self.return_val = return_val

        self.filtered_topk_max_k = 2048
        # 8 bits for radix-based filter.
        self.radix = 256

        if cutlass.const_expr(self.dtype == cutlass.Float32):
            self.num_buffer_smem_input_idx = 2
        else:
            self.num_buffer_smem_input_idx = 1

        # 65536 is the max index value for uint16.
        if cutlass.const_expr(self.max_num_cols <= 65536):
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
            if self.max_num_cols >= 262144:
                self.filtered_topk_smem_input_size = 4096
            elif self.max_num_cols >= 131072:
                self.filtered_topk_smem_input_size = 3072
            elif self.max_num_cols >= 65536:
                self.filtered_topk_smem_input_size = 2048
            elif self.max_num_cols >= 16384:
                self.filtered_topk_smem_input_size = 1024
            elif self.max_num_cols >= 8192:
                self.filtered_topk_smem_input_size = 512
            else:
                self.filtered_topk_smem_input_size = 256
        else:
            self.filtered_topk_smem_input_size = min(
                self.max_smem_input_size, self.max_num_cols
            )

        if cutlass.const_expr(self.max_num_cols > self.filtered_topk_smem_input_size):
            self.enable_gmem_store = True
        else:
            self.enable_gmem_store = False

        self.vec_size = num_copy_bits // dtype.width
        if cutlass.const_expr(
            dtype not in [cutlass.Float32, cute.BFloat16, cutlass.Float16]
        ):
            raise ValueError(f"Unsupported dtype: {dtype}")

        if cutlass.const_expr(self.is_prefill):
            if cutlass.const_expr(dtype == cutlass.Float32):
                if self.max_num_cols >= 8192:
                    self.num_threads_per_cta = 512
                else:
                    self.num_threads_per_cta = 256
            else:
                if self.max_num_cols >= self.vec_size * 512:
                    self.num_threads_per_cta = 512
                else:
                    self.num_threads_per_cta = 256
        else:
            if cutlass.const_expr(dtype == cutlass.Float32):
                if self.max_num_cols >= self.vec_size * 1024:
                    self.num_threads_per_cta = 1024
                else:
                    if self.max_num_cols > 2048:
                        self.num_threads_per_cta = 512
                    else:
                        self.num_threads_per_cta = 256
            else:
                if self.max_num_cols >= self.vec_size * 1024:
                    self.num_threads_per_cta = 1024
                else:
                    if cutlass.const_expr(self.max_num_cols > 4096):
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
    # Main kernel (dynamic column count -- reads length from input.shape)
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
    ):
        """CuTe DSL implementation of TopK kernel based on radix-based filter algorithm.

        The actual column count (``length``) is read from ``input.shape[1]``
        at runtime.  Alignment is handled via prologue/epilogue scalar loads.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # Runtime column count
        length = input.shape[1]

        score = input[bidx, None]
        dst = output_indices[bidx, None]
        if cutlass.const_expr(self.enable_gmem_store):
            buffer = extra_buffer[bidx, None, None]
        if cutlass.const_expr(self.return_val):
            dst_values = output_values[bidx, None]

        # --- Runtime alignment handling ---
        row_start = 0
        row_ptr = score.iterator + row_start
        row_addr_u64 = row_ptr.toint()

        align_bytes = self.num_copy_bits // 8  # 256/8 = 32 bytes
        elem_bytes = self.dtype.width // 8

        misalign = row_addr_u64 % align_bytes
        fix_bytes = cutlass.Int64(0)
        if misalign != 0:
            fix_bytes = align_bytes - misalign

        prologue_elems = cutlass.Int32(fix_bytes // elem_bytes)

        remaining = length - prologue_elems
        aligned_size = (remaining // self.vec_size) * self.vec_size
        left_size = remaining - aligned_size

        vec_start = row_start + prologue_elems
        left_start = vec_start + aligned_size

        shape = input.shape

        idX = cute.make_identity_tensor((shape[0], aligned_size))
        input_ptr = input.iterator + vec_start
        input_addr_u64 = input_ptr.toint()
        input_ptr_aligned = cute.make_ptr(
            self.dtype, input_addr_u64, assumed_align=align_bytes
        )

        input_tensor = cute.make_tensor(
            input_ptr_aligned,
            cute.make_layout((shape[0], aligned_size), stride=input.stride),
        )

        # Slice for CTAs
        gX, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, None)) for mT in (input_tensor, idX)
        ]
        self.num_sub_tiles = gX.shape[2]
        thr_copy = tiled_copy.get_slice(tidx)

        tXgX = thr_copy.partition_S(gX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None, None]
        tXrX = cute.make_fragment_like(tXgX[None, None, None, 0])

        tXcX_tile = thr_copy.partition_S(cX)

        # Trivial case: length <= top_k
        if length <= self.top_k:
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
                tXpX_tile = self.predicate_tile(
                    tXcX_tile[None, None, None, tile_idx],
                    cutlass.Int32(aligned_size),
                )
                cute.copy(
                    copy_atom,
                    tXgX[None, None, None, tile_idx],
                    tXrX,
                    pred=tXpX_tile[None, None, None],
                )
                self._fill_oob(
                    tXrX,
                    tXpX_tile[None, None, None],
                    -tXrX.element_type.inf,
                )

                for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                    bin_val = self.to_coarse_key(tXrX[i])
                    atomicAdd(
                        s_histogram.iterator + cutlass.Int32(bin_val),
                        val_one,
                    )

            # Prologue scalar loads (for initial misalignment)
            for j in range(tidx, prologue_elems, self.num_threads_per_cta):
                col_idx = cutlass.Int32(j)
                raw = score[col_idx]
                bin_val = self.to_coarse_key(raw)
                atomicAdd(
                    s_histogram.iterator + cutlass.Int32(bin_val),
                    val_one,
                )

            # Epilogue scalar loads (leftover after vectorized part)
            for j in range(tidx, left_size, self.num_threads_per_cta):
                col_idx = cutlass.Int32(left_start + j)
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
                    tXpX_tile = self.predicate_tile(
                        tXcX_tile[None, None, None, tile_idx],
                        cutlass.Int32(aligned_size),
                    )
                    cute.copy(
                        copy_atom,
                        tXgX[None, None, None, tile_idx],
                        tXrX,
                        pred=tXpX_tile[None, None, None],
                    )
                    self._fill_oob(
                        tXrX,
                        tXpX_tile[None, None, None],
                        -tXrX.element_type.inf,
                    )

                    for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                        cur_tXcX = tXcX[None, None, None, tile_idx]
                        bin_val = self.to_coarse_key(tXrX[i])
                        if bin_val < threshold_bin:
                            pos = atomicAdd(s_counter.iterator, val_one)
                            idx = self.index_type(
                                cur_tXcX[i // vec_size][1] + i % vec_size
                            )
                            s_indices[pos] = idx + self.index_type(vec_start)

                # Prologue scalar load indices
                for j in range(tidx, prologue_elems, self.num_threads_per_cta):
                    col_idx = cutlass.Int32(j)
                    raw = score[col_idx]
                    bin_val = self.to_coarse_key(raw)
                    if bin_val < threshold_bin:
                        pos = atomicAdd(s_counter.iterator, val_one)
                        idx = self.index_type(col_idx)
                        s_indices[pos] = idx

                # Epilogue scalar load indices
                for j in range(tidx, left_size, self.num_threads_per_cta):
                    col_idx = cutlass.Int32(left_start + j)
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
                    tXpX_tile = self.predicate_tile(
                        tXcX_tile[None, None, None, tile_idx],
                        cutlass.Int32(aligned_size),
                    )
                    cute.copy(
                        copy_atom,
                        tXgX[None, None, None, tile_idx],
                        tXrX,
                        pred=tXpX_tile[None, None, None],
                    )
                    self._fill_oob(
                        tXrX,
                        tXpX_tile[None, None, None],
                        -tXrX.element_type.inf,
                    )

                    for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                        raw_input = tXrX[i]
                        bin_val = self.to_coarse_key(raw_input)
                        cur_tXcX = tXcX[None, None, None, tile_idx]
                        idx = self.index_type(
                            cur_tXcX[i // vec_size][1] + i % vec_size + vec_start
                        )
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

                # Prologue scalar refinement
                for j in range(tidx, prologue_elems, self.num_threads_per_cta):
                    col_idx = cutlass.Int32(j)
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

                # Epilogue scalar refinement
                for j in range(tidx, left_size, self.num_threads_per_cta):
                    col_idx = cutlass.Int32(left_start + j)
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
                                idx = self.index_type(idx)
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
                        if cutlass.const_expr(self.return_val):
                            topk_vals[v, i] = score[index]
            # [atom, rest_vec]
            mIndices_store = cute.tiled_divide(dst, (vecsize_out,))
            if cutlass.const_expr(self.return_val):
                mValues_store = cute.tiled_divide(dst_values, (vecsize_out,))
            for i in cutlass.range(cute.size(topk_vals.shape, [1]), unroll_full=True):
                col = i * self.num_threads_per_cta + tidx % self.num_threads_per_cta
                if col < self.top_k // vecsize_out:
                    cute.autovec_copy(topk_indices[None, i], mIndices_store[None, col])
                    if cutlass.const_expr(self.return_val):
                        cute.autovec_copy(topk_vals[None, i], mValues_store[None, col])

    # ------------------------------------------------------------------
    # Tiled copy setup
    # ------------------------------------------------------------------

    def _get_tiled_copy(self):
        threads_per_row = self.num_threads_per_cta
        tiler_mn = (
            1,
            self.vec_size * threads_per_row,
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

        return (
            copy_atom,
            tiled_copy,
            tiler_mn,
        )

    @cute.jit
    def predicate_tile(self, tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
        tApA = cute.make_fragment(
            cute.make_layout(
                (
                    cute.size(tAcA, mode=[0, 1]),
                    cute.size(tAcA, mode=[1]),
                    cute.size(tAcA, mode=[2]),
                ),
                stride=(cute.size(tAcA, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tApA.shape[0]):
            for rest_k in cutlass.range_constexpr(tApA.shape[2]):
                tApA[rest_v, 0, rest_k] = cute.elem_less(
                    tAcA[(0, rest_v), 0, rest_k][1], limit
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
        ) = self._get_tiled_copy()
        self.filtered_topk_kernel(
            input_values,
            buffer,
            output_indices,
            output_values,
            tiler_mn,
            copy_atom,
            tiled_copy,
        ).launch(
            grid=blocks,
            block=(tiled_copy.size, 1, 1),
            stream=stream,
        )
        return


# ==========================================================================
# TVM-FFI compilation wrapper
# ==========================================================================

_TORCH_TO_CUTLASS = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _bucket_num_cols(num_cols: int) -> int:
    """Round *num_cols* up to the next power of 2 for kernel caching.

    This keeps the number of unique compiled kernels small (one per
    power-of-2 bucket) while the kernel itself reads the actual column
    count from ``input.shape[1]`` at runtime.
    """
    if num_cols <= 256:
        return 256
    return 1 << (num_cols - 1).bit_length()


@functools.cache
def _get_compiled_kernel(
    torch_dtype: torch.dtype,
    max_num_cols: int,
    top_k: int,
    is_prefill: bool,
    return_val: bool = True,
    num_copy_bits: int = 256,
):
    """Return a compiled ``tensor_api`` closure that accepts ``torch.Tensor``
    arguments directly (via TVM-FFI).

    The result is cached by ``(torch_dtype, max_num_cols, top_k, is_prefill,
    return_val, num_copy_bits)``.  Because *max_num_cols* is a power-of-2
    bucket, a single compiled kernel serves all ``num_cols <= max_num_cols``.
    """
    cutlass_dtype = _TORCH_TO_CUTLASS[torch_dtype]

    if cutlass_dtype == cutlass.Float32:
        buffer_numbers = 2
    else:
        buffer_numbers = 1

    kernel_obj = FilteredTopKKernel(
        dtype=cutlass_dtype,
        max_num_cols=max_num_cols,
        top_k=top_k,
        num_copy_bits=num_copy_bits,
        is_prefill=is_prefill,
        return_val=return_val,
    )

    # Symbolic dimensions -- each independent axis gets its own symbol so
    # that TVM-FFI does not enforce equality between unrelated shapes.
    sym_n = cute.sym_int()  # batch / num_rows
    sym_buf = cute.sym_int()  # buffer_numbers (1 or 2)
    sym_cols = cute.sym_int(divisibility=32)  # column dimension (dynamic)

    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (sym_n, sym_cols),
        stride_order=(1, 0),
        assumed_align=32,
    )
    buffer_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_n, sym_buf, sym_cols),
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

    return tensor_api, buffer_numbers


# ==========================================================================
# Public entry-point
# ==========================================================================


def cute_dsl_top_k(
    input: torch.Tensor,
    k: int,
    num_copy_bits: int = 256,
    return_values: bool = True,
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
    return_values : bool
        If True, also gather top-k values; if False, skip value gathering
        for reduced memory bandwidth.

    Returns
    -------
    output_values : torch.Tensor or None
        Top-k values, shape ``(num_rows, k)``, same dtype as *input*.
        ``None`` when ``return_values=False``.
    output_indices : torch.Tensor
        Top-k column indices, shape ``(num_rows, k)``, dtype ``int32``.
    """
    assert input.ndim == 2, f"Expected 2-D input, got {input.ndim}-D"
    assert k % 2 == 0, f"top_k must be even, got {k}"
    assert k <= 2048, f"top_k must be <= 2048, got {k}"

    num_rows, num_cols = input.shape
    is_prefill = num_rows > 148

    max_num_cols = _bucket_num_cols(num_cols)

    tensor_api, buffer_numbers = _get_compiled_kernel(
        input.dtype, max_num_cols, k, is_prefill, return_values, num_copy_bits
    )

    output_indices = torch.empty(num_rows, k, dtype=torch.int32, device=input.device)
    # Always allocate output_values: the compiled kernel expects the tensor
    # argument even when return_val=False (it simply won't write to it).
    output_values = torch.empty(num_rows, k, dtype=input.dtype, device=input.device)
    buffer_tensor = torch.empty(
        num_rows,
        buffer_numbers,
        num_cols,
        dtype=torch.int32,
        device=input.device,
    )

    tensor_api(input, buffer_tensor, output_indices, output_values)

    return (output_values if return_values else None), output_indices
