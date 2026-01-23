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

Top-K Selection using CuTe DSL
==============================

This module provides a CuTe DSL implementation of radix-based top-K selection.

Key Features:
- Native BF16 processing with only 2 radix rounds (vs 4 for F32)
- F32 support with 4 radix rounds
- Both single-CTA and multi-CTA implementations for large N

Based on the radix select algorithm from cute-dsl-zoo.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm, nvvm
from cutlass._mlir.dialects.nvvm import AtomicOpKind, MemOrderKind, MemScopeKind
from cutlass.cutlass_dsl import T, dsl_user_op

from .utils import make_ptr

# =============================================================================
# Constants
# =============================================================================

RADIX_BITS = 8
RADIX = 256

# BF16: 2 rounds (16-bit)
NUM_ROUNDS_BF16 = 2

# F32: 4 rounds (32-bit)
NUM_ROUNDS_F32 = 4

# Single-CTA threshold
MAX_N_SINGLE_CTA = 16384

# Multi-CTA parameters
CHUNK_SIZE_DEFAULT = 8192
MIN_N_MULTI_CTA = 16384

# RadixRowState layout
HISTOGRAM_OFFSET = 0
ARRIVAL_COUNTER_OFFSET = 768
OUTPUT_COUNTER_OFFSET = 769
STATE_SIZE_INT32 = 770


# =============================================================================
# BF16 Ordered Conversion (Native 16-bit in 32-bit container)
# =============================================================================


@dsl_user_op
def bf16_to_ordered16(bf16_bits: Int32, *, loc=None, ip=None) -> Int32:
    """
    Convert BF16 bits to ordered 16-bit representation.

    Input: BF16 bits in low 16 bits of Int32
    Output: Ordered value in low 16 bits of Int32

    BF16 format: [S:1][Exp:8][Mantissa:7] = 16 bits
    Ordered: flip all bits if negative, flip sign bit if positive
    """
    result = llvm.inline_asm(
        T.i32(),
        [bf16_bits.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p_neg;
            .reg .u32 val, mask, sign_bit;
            and.b32 val, $1, 0xFFFF;
            and.b32 sign_bit, val, 0x8000;
            setp.ne.u32 p_neg, sign_bit, 0;
            selp.u32 mask, 0xFFFF, 0x8000, p_neg;
            xor.b32 $0, val, mask;
            and.b32 $0, $0, 0xFFFF;
        }
        """,
        "=r,r",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def ordered16_to_bf16_bits(ordered_val: Int32, *, loc=None, ip=None) -> Int32:
    """
    Convert ordered 16-bit back to BF16 bits.

    Input: Ordered value in low 16 bits of Int32
    Output: BF16 bits in low 16 bits of Int32

    Inverse of bf16_to_ordered16:
    - If bit 15 set (was positive): flip sign bit only
    - If bit 15 clear (was negative): flip all bits
    """
    result = llvm.inline_asm(
        T.i32(),
        [ordered_val.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p_was_pos;
            .reg .u32 val, mask, sign_bit;
            and.b32 val, $1, 0xFFFF;
            and.b32 sign_bit, val, 0x8000;
            setp.ne.u32 p_was_pos, sign_bit, 0;
            selp.u32 mask, 0x8000, 0xFFFF, p_was_pos;
            xor.b32 $0, val, mask;
            and.b32 $0, $0, 0xFFFF;
        }
        """,
        "=r,r",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def load_bf16_as_u16(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """Load BF16 value as uint16 in low bits of Int32."""
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.i32(),
        [ptr_int],
        """
        {
            .reg .u16 tmp;
            ld.global.u16 tmp, [$1];
            cvt.u32.u16 $0, tmp;
        }
        """,
        "=r,l",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def store_bf16_from_u16(ptr: cute.Pointer, val: Int32, *, loc=None, ip=None) -> None:
    """Store low 16 bits of Int32 as BF16."""
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    val_ir = val.ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [ptr_int, val_ir],
        """
        {
            .reg .u16 tmp;
            cvt.u16.u32 tmp, $1;
            st.global.u16 [$0], tmp;
        }
        """,
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_i32_global(ptr: cute.Pointer, val: Int32, *, loc=None, ip=None) -> None:
    """Store Int32 to global memory."""
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    val_ir = val.ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [ptr_int, val_ir],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


# =============================================================================
# F32 Ordered Conversion
# =============================================================================


@dsl_user_op
def float32_to_ordered(val_bits: Int32, *, loc=None, ip=None) -> Int32:
    """Convert float32 bits to ordered int32 representation."""
    result = llvm.inline_asm(
        T.i32(),
        [val_bits.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p_neg;
            .reg .s32 mask, result;
            setp.lt.s32 p_neg, $1, 0;
            selp.s32 mask, -1, 0x80000000, p_neg;
            xor.b32 $0, $1, mask;
        }
        """,
        "=r,r",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def ordered_to_float32(ordered_val: Int32, *, loc=None, ip=None) -> Float32:
    """Convert ordered int32 back to float32."""
    result = llvm.inline_asm(
        T.f32(),
        [ordered_val.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p_was_neg;
            .reg .s32 mask, bits;
            setp.ge.s32 p_was_neg, $1, 0;
            selp.s32 mask, -1, 0x80000000, p_was_neg;
            xor.b32 bits, $1, mask;
            mov.b32 $0, bits;
        }
        """,
        "=f,r",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Float32(result)


@dsl_user_op
def _float32_to_bits(val: Float32, *, loc=None, ip=None) -> Int32:
    """Reinterpret float32 as int32 bits."""
    result = llvm.inline_asm(
        T.i32(),
        [val.ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=r,f",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


# =============================================================================
# Unsigned Comparisons
# =============================================================================


@dsl_user_op
def unsigned_gt_32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """Unsigned greater-than comparison."""
    result = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p;
            setp.hi.u32 p, $1, $2;
            selp.u32 $0, 1, 0, p;
        }
        """,
        "=r,r,r",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def unsigned_ge_16(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """Unsigned greater-than-or-equal for 16-bit values in Int32."""
    result = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p;
            .reg .u32 a16, b16;
            and.b32 a16, $1, 0xFFFF;
            and.b32 b16, $2, 0xFFFF;
            setp.hs.u32 p, a16, b16;
            selp.u32 $0, 1, 0, p;
        }
        """,
        "=r,r,r",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def unsigned_gt_16(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """Unsigned greater-than for 16-bit values in Int32."""
    result = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p;
            .reg .u32 a16, b16;
            and.b32 a16, $1, 0xFFFF;
            and.b32 b16, $2, 0xFFFF;
            setp.hi.u32 p, a16, b16;
            selp.u32 $0, 1, 0, p;
        }
        """,
        "=r,r,r",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


# =============================================================================
# Atomic Operations
# =============================================================================


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: Int32, *, loc=None, ip=None) -> cute.Pointer:
    """Get pointer to element at 1D coordinate in tensor."""
    return x.iterator + coord


@dsl_user_op
def atomicAdd_smem(dst_ptr: cute.Pointer, val: Int32, *, loc=None, ip=None) -> Int32:
    """Atomic add to shared memory."""
    return nvvm.atomicrmw(
        T.i32(),
        AtomicOpKind.ADD,
        dst_ptr.llvm_ptr,
        val.ir_value(loc=loc, ip=ip),
        mem_order=MemOrderKind.RELAXED,
        syncscope=MemScopeKind.CTA,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomicAdd_gmem(dst_ptr: cute.Pointer, val: Int32, *, loc=None, ip=None) -> Int32:
    """Atomic add to global memory."""
    return nvvm.atomicrmw(
        T.i32(),
        AtomicOpKind.ADD,
        dst_ptr.llvm_ptr,
        val.ir_value(loc=loc, ip=ip),
        mem_order=MemOrderKind.RELAXED,
        syncscope=MemScopeKind.GPU,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_acquire_gpu(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """Load from global memory with acquire semantics."""
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.i32(),
        [ptr_int],
        "ld.global.acquire.gpu.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def red_release_add_gpu(ptr: cute.Pointer, val: Int32, *, loc=None, ip=None) -> None:
    """Atomic add with release semantics."""
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    val_ir = val.ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [],
        "fence.acq_rel.gpu;",
        "",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    llvm.inline_asm(
        None,
        [ptr_int, val_ir],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


# =============================================================================
# Helper JIT Functions
# =============================================================================


@cute.jit
def clear_histogram_block(
    tidx: Int32, histogram: cute.Tensor, num_threads: cutlass.Constexpr[int]
):
    """Clear histogram using all threads in block."""
    if tidx < 256:
        histogram[tidx] = Int32(0)


@cute.jit
def parallel_suffix_sum_inplace(tidx: Int32, data: cute.Tensor):
    """Compute suffix sum in-place."""
    for stride_exp in cutlass.range_constexpr(8):
        stride = 1 << stride_exp
        val = Int32(0)
        if tidx < 256:
            val = data[tidx]
            next_idx = tidx + stride
            if next_idx < 256:
                val = val + data[next_idx]
        cute.arch.barrier()
        if tidx < 256:
            data[tidx] = val
        cute.arch.barrier()


@cute.jit
def find_threshold_bucket_local(
    tidx: Int32,
    suffix_sum: cute.Tensor,
    remaining_k: Int32,
    shared_scalars: cute.Tensor,
):
    """Find threshold bucket containing the k-th largest element."""
    if tidx == 0:
        shared_scalars[0] = Int32(0)
        shared_scalars[1] = remaining_k
    cute.arch.barrier()

    if tidx < 256:
        count_ge = suffix_sum[tidx]
        count_gt = Int32(0)
        if tidx + 1 < 256:
            count_gt = suffix_sum[tidx + 1]

        is_threshold = (count_ge >= remaining_k) & (count_gt < remaining_k)
        if is_threshold:
            shared_scalars[0] = tidx
            shared_scalars[1] = remaining_k - count_gt
    cute.arch.barrier()


# =============================================================================
# Native BF16 Single-CTA Kernel
# =============================================================================


class NativeBF16SingleCTA:
    """
    Single-CTA native BF16 radix select with:
    - 16-bit ordered representation (in 32-bit containers)
    - Only 2 radix rounds
    - Direct value and index output (UNSORTED)
    """

    NUM_THREADS: cutlass.Constexpr[int] = 256

    def __init__(self, N: int, k: int):
        self.N = N
        self.k = k
        self.elements_per_thread = (N + 255) // 256

    def _smem_size_in_bytes(self) -> int:
        # histogram[256] + suffix_sum[256] + scalars[8] + ordered_data[N] (32-bit each)
        return 256 * 4 + 256 * 4 + 8 * 4 + self.N * 4 + 128

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        values_ptr: cute.Pointer,
        indices_ptr: cute.Pointer,
        M: Int32,
        stream,
    ):
        # Create tensor view for indices (Int32, easy to handle)
        mIndices = cute.make_tensor(
            indices_ptr,
            cute.make_layout((M, self.k), stride=(self.k, 1)),
        )

        self.kernel(input_ptr, values_ptr, mIndices, M).launch(
            grid=[M, 1, 1],
            block=[self.NUM_THREADS, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_ptr: cute.Pointer,
        values_ptr: cute.Pointer,
        mIndices: cute.Tensor,
        M: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row_idx, _, _ = cute.arch.block_idx()

        NUM_THREADS = self.NUM_THREADS
        N = self.N
        k = self.k

        # Get row view for indices
        row_Indices = mIndices[row_idx, None]

        # Shared Memory
        smem = cutlass.utils.SmemAllocator()

        histogram = smem.allocate_tensor(
            Int32, cute.make_layout((256,), stride=(1,)), byte_alignment=16
        )
        suffix_sum = smem.allocate_tensor(
            Int32, cute.make_layout((256,), stride=(1,)), byte_alignment=16
        )
        shared_scalars = smem.allocate_tensor(
            Int32, cute.make_layout((8,), stride=(1,)), byte_alignment=16
        )
        # Store ordered 16-bit values in 32-bit for easier manipulation
        shared_ordered = smem.allocate_tensor(
            Int32, cute.make_layout((N,), stride=(1,)), byte_alignment=16
        )

        # Calculate row offsets (BF16 = 2 bytes per element)
        row_input_offset = row_idx * N * 2  # bytes
        row_values_offset = row_idx * k * 2  # bytes

        # Phase 1: Load BF16, convert to ordered 16-bit
        for i in range(self.elements_per_thread):
            idx = tidx + i * NUM_THREADS
            if idx < N:
                # Load BF16 as uint16
                elem_ptr = input_ptr + (row_input_offset + idx * 2)
                bf16_bits = load_bf16_as_u16(elem_ptr)
                ordered = bf16_to_ordered16(bf16_bits)
                shared_ordered[idx] = ordered

        cute.arch.barrier()

        # Initialize scalars
        if tidx == 0:
            shared_scalars[0] = Int32(0)
            shared_scalars[1] = Int32(k)
            shared_scalars[2] = Int32(0)
            shared_scalars[3] = Int32(0)
        cute.arch.barrier()

        # ROUND 0: bits 15-8 (MSB - sign + exponent)
        clear_histogram_block(tidx, histogram, NUM_THREADS)
        cute.arch.barrier()

        for i in range(self.elements_per_thread):
            idx = tidx + i * NUM_THREADS
            if idx < N:
                ordered_val = shared_ordered[idx]
                bucket = (ordered_val >> Int32(8)) & Int32(0xFF)
                ptr = elem_pointer(histogram, bucket)
                atomicAdd_smem(ptr, Int32(1))

        cute.arch.barrier()

        if tidx < 256:
            suffix_sum[tidx] = histogram[tidx]
        cute.arch.barrier()

        parallel_suffix_sum_inplace(tidx, suffix_sum)

        remaining_k_r0 = shared_scalars[1]
        find_threshold_bucket_local(tidx, suffix_sum, remaining_k_r0, shared_scalars)

        if tidx == 0:
            shared_scalars[2] = shared_scalars[0]
        cute.arch.barrier()

        # ROUND 1: bits 7-0 (LSB - mantissa)
        clear_histogram_block(tidx, histogram, NUM_THREADS)
        cute.arch.barrier()

        prefix_r1 = shared_scalars[2]

        for i in range(self.elements_per_thread):
            idx = tidx + i * NUM_THREADS
            if idx < N:
                ordered_val = shared_ordered[idx]
                msb = (ordered_val >> Int32(8)) & Int32(0xFF)
                if msb == prefix_r1:
                    bucket = ordered_val & Int32(0xFF)
                    ptr = elem_pointer(histogram, bucket)
                    atomicAdd_smem(ptr, Int32(1))

        cute.arch.barrier()

        if tidx < 256:
            suffix_sum[tidx] = histogram[tidx]
        cute.arch.barrier()

        parallel_suffix_sum_inplace(tidx, suffix_sum)

        remaining_k_r1 = shared_scalars[1]
        find_threshold_bucket_local(tidx, suffix_sum, remaining_k_r1, shared_scalars)

        if tidx == 0:
            shared_scalars[2] = (prefix_r1 << Int32(8)) | shared_scalars[0]
        cute.arch.barrier()

        # Phase 3: Collection
        radix_threshold = shared_scalars[2]

        if tidx == 0:
            shared_scalars[3] = Int32(0)
        cute.arch.barrier()

        # Collect > threshold
        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            if local_idx < N:
                ordered_val = shared_ordered[local_idx]
                if unsigned_gt_16(ordered_val, radix_threshold) != Int32(0):
                    ptr = elem_pointer(shared_scalars, Int32(3))
                    pos = atomicAdd_smem(ptr, Int32(1))
                    if pos < k:
                        bf16_bits = ordered16_to_bf16_bits(ordered_val)
                        val_offset = row_values_offset + pos * Int32(2)
                        val_ptr = values_ptr + val_offset
                        store_bf16_from_u16(val_ptr, bf16_bits)
                        row_Indices[pos] = local_idx

        cute.arch.barrier()

        # Collect == threshold
        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            if local_idx < N:
                ordered_val = shared_ordered[local_idx]
                # Compare low 16 bits
                ord_masked = ordered_val & Int32(0xFFFF)
                thresh_masked = radix_threshold & Int32(0xFFFF)
                if ord_masked == thresh_masked:
                    ptr = elem_pointer(shared_scalars, Int32(3))
                    pos = atomicAdd_smem(ptr, Int32(1))
                    if pos < k:
                        bf16_bits = ordered16_to_bf16_bits(ordered_val)
                        val_offset = row_values_offset + pos * Int32(2)
                        val_ptr = values_ptr + val_offset
                        store_bf16_from_u16(val_ptr, bf16_bits)
                        row_Indices[pos] = local_idx

        cute.arch.barrier()


# =============================================================================
# Native BF16 Multi-CTA Kernel
# =============================================================================


class NativeBF16MultiCTA:
    """
    Multi-CTA native BF16 radix select with:
    - 16-bit ordered representation (in 32-bit containers)
    - Only 2 radix rounds
    - Inter-CTA coordination via global state
    """

    NUM_THREADS: cutlass.Constexpr[int] = 256

    def __init__(self, N: int, k: int, chunk_size: int, num_ctas_per_row: int):
        self.N = N
        self.k = k
        self.chunk_size = chunk_size
        self.num_ctas_per_row = num_ctas_per_row
        self.elements_per_thread = (chunk_size + 255) // 256

    def _smem_size_in_bytes(self) -> int:
        return 256 * 4 + 256 * 4 + 8 * 4 + self.chunk_size * 4 + 64

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        values_ptr: cute.Pointer,
        indices_ptr: cute.Pointer,
        state_ptr: cute.Pointer,
        M: Int32,
        stream,
    ):
        mState = cute.make_tensor(
            state_ptr,
            cute.make_layout((M, STATE_SIZE_INT32), stride=(STATE_SIZE_INT32, 1)),
        )
        mIndices = cute.make_tensor(
            indices_ptr,
            cute.make_layout((M, self.k), stride=(self.k, 1)),
        )

        total_ctas = M * self.num_ctas_per_row

        self.kernel(input_ptr, values_ptr, mIndices, mState, M).launch(
            grid=[total_ctas, 1, 1],
            block=[self.NUM_THREADS, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_ptr: cute.Pointer,
        values_ptr: cute.Pointer,
        mIndices: cute.Tensor,
        mState: cute.Tensor,
        M: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        global_cta_id, _, _ = cute.arch.block_idx()

        NUM_THREADS = self.NUM_THREADS
        N = self.N
        k = self.k
        chunk_size = self.chunk_size
        num_ctas_per_row = self.num_ctas_per_row

        row_idx = global_cta_id // num_ctas_per_row
        cta_in_row = global_cta_id % num_ctas_per_row

        row_State = mState[row_idx, None]
        row_Indices = mIndices[row_idx, None]

        chunk_start = cta_in_row * chunk_size
        actual_chunk_size = chunk_size
        if chunk_start + chunk_size > N:
            actual_chunk_size = N - chunk_start
            if actual_chunk_size < 0:
                actual_chunk_size = 0

        # Calculate row offsets (for BF16 values only - indices use tensor view)
        row_input_offset = row_idx * N * 2  # BF16 = 2 bytes
        row_values_offset = row_idx * k * 2

        # Shared Memory
        smem = cutlass.utils.SmemAllocator()

        local_histogram = smem.allocate_tensor(
            Int32, cute.make_layout((256,), stride=(1,)), byte_alignment=16
        )
        suffix_sum = smem.allocate_tensor(
            Int32, cute.make_layout((256,), stride=(1,)), byte_alignment=16
        )
        shared_scalars = smem.allocate_tensor(
            Int32, cute.make_layout((8,), stride=(1,)), byte_alignment=16
        )
        shared_ordered = smem.allocate_tensor(
            Int32, cute.make_layout((chunk_size,), stride=(1,)), byte_alignment=16
        )

        # Load chunk with BF16 to ordered conversion
        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            global_idx = chunk_start + local_idx
            if local_idx < actual_chunk_size and global_idx < N:
                elem_ptr = input_ptr + (row_input_offset + global_idx * 2)
                bf16_bits = load_bf16_as_u16(elem_ptr)
                ordered = bf16_to_ordered16(bf16_bits)
                shared_ordered[local_idx] = ordered

        cute.arch.barrier()

        # Initialize: CTA 0 clears global state
        if cta_in_row == 0:
            for buf in cutlass.range_constexpr(3):
                if tidx < 256:
                    row_State[HISTOGRAM_OFFSET + buf * 256 + tidx] = Int32(0)
            cute.arch.barrier()
            if tidx == 0:
                row_State[ARRIVAL_COUNTER_OFFSET] = Int32(0)
                row_State[OUTPUT_COUNTER_OFFSET] = Int32(0)
                arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
                red_release_add_gpu(arrival_ptr, Int32(1))
        else:
            cute.arch.barrier()

        # Wait for initialization
        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            current = ld_acquire_gpu(arrival_ptr)
            while current < Int32(1):
                current = ld_acquire_gpu(arrival_ptr)
        cute.arch.barrier()

        if tidx == 0:
            shared_scalars[0] = Int32(0)
            shared_scalars[1] = Int32(k)
            shared_scalars[2] = Int32(0)
            shared_scalars[3] = Int32(0)
        cute.arch.barrier()

        barrier_phase = Int32(1)

        # ROUND 0: bits 15-8
        clear_histogram_block(tidx, local_histogram, NUM_THREADS)
        cute.arch.barrier()

        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            if local_idx < actual_chunk_size:
                ordered_val = shared_ordered[local_idx]
                bucket = (ordered_val >> Int32(8)) & Int32(0xFF)
                ptr = elem_pointer(local_histogram, bucket)
                atomicAdd_smem(ptr, Int32(1))

        cute.arch.barrier()

        hist_base = HISTOGRAM_OFFSET + 0 * 256
        if tidx < 256:
            local_count = local_histogram[tidx]
            if local_count > Int32(0):
                global_hist_ptr = elem_pointer(row_State, Int32(hist_base + tidx))
                atomicAdd_gmem(global_hist_ptr, local_count)

        if cta_in_row == 0 and tidx < 256:
            row_State[HISTOGRAM_OFFSET + 1 * 256 + tidx] = Int32(0)

        cute.arch.barrier()

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            red_release_add_gpu(arrival_ptr, Int32(1))

        target = Int32(1) + barrier_phase * num_ctas_per_row
        barrier_phase = barrier_phase + Int32(1)

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            current = ld_acquire_gpu(arrival_ptr)
            while current < target:
                current = ld_acquire_gpu(arrival_ptr)
        cute.arch.barrier()

        if tidx < 256:
            suffix_sum[tidx] = row_State[hist_base + tidx]
        cute.arch.barrier()

        parallel_suffix_sum_inplace(tidx, suffix_sum)

        remaining_k_r0 = shared_scalars[1]
        find_threshold_bucket_local(tidx, suffix_sum, remaining_k_r0, shared_scalars)

        if tidx == 0:
            shared_scalars[2] = shared_scalars[0]
        cute.arch.barrier()

        # ROUND 1: bits 7-0
        clear_histogram_block(tidx, local_histogram, NUM_THREADS)
        cute.arch.barrier()

        prefix_r1 = shared_scalars[2]

        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            if local_idx < actual_chunk_size:
                ordered_val = shared_ordered[local_idx]
                msb = (ordered_val >> Int32(8)) & Int32(0xFF)
                if msb == prefix_r1:
                    bucket = ordered_val & Int32(0xFF)
                    ptr = elem_pointer(local_histogram, bucket)
                    atomicAdd_smem(ptr, Int32(1))

        cute.arch.barrier()

        hist_base = HISTOGRAM_OFFSET + 1 * 256
        if tidx < 256:
            local_count = local_histogram[tidx]
            if local_count > Int32(0):
                global_hist_ptr = elem_pointer(row_State, Int32(hist_base + tidx))
                atomicAdd_gmem(global_hist_ptr, local_count)

        cute.arch.barrier()

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            red_release_add_gpu(arrival_ptr, Int32(1))

        target = Int32(1) + barrier_phase * num_ctas_per_row
        barrier_phase = barrier_phase + Int32(1)

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            current = ld_acquire_gpu(arrival_ptr)
            while current < target:
                current = ld_acquire_gpu(arrival_ptr)
        cute.arch.barrier()

        if tidx < 256:
            suffix_sum[tidx] = row_State[hist_base + tidx]
        cute.arch.barrier()

        parallel_suffix_sum_inplace(tidx, suffix_sum)
        remaining_k_r1 = shared_scalars[1]
        find_threshold_bucket_local(tidx, suffix_sum, remaining_k_r1, shared_scalars)

        if tidx == 0:
            shared_scalars[2] = (prefix_r1 << Int32(8)) | shared_scalars[0]
        cute.arch.barrier()

        # COLLECTION
        radix_threshold = shared_scalars[2]

        if cta_in_row == 0 and tidx == 0:
            row_State[OUTPUT_COUNTER_OFFSET] = Int32(0)

        cute.arch.barrier()

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            red_release_add_gpu(arrival_ptr, Int32(1))

        target = Int32(1) + barrier_phase * num_ctas_per_row
        barrier_phase = barrier_phase + Int32(1)

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            current = ld_acquire_gpu(arrival_ptr)
            while current < target:
                current = ld_acquire_gpu(arrival_ptr)
        cute.arch.barrier()

        # Count local > threshold
        if tidx == 0:
            shared_scalars[3] = Int32(0)
        cute.arch.barrier()

        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            if local_idx < actual_chunk_size:
                ordered_val = shared_ordered[local_idx]
                if unsigned_gt_16(ordered_val, radix_threshold) != Int32(0):
                    ptr = elem_pointer(shared_scalars, Int32(3))
                    atomicAdd_smem(ptr, Int32(1))

        cute.arch.barrier()

        local_gt_count = shared_scalars[3]

        if tidx == 0 and local_gt_count > Int32(0):
            output_ptr = elem_pointer(row_State, Int32(OUTPUT_COUNTER_OFFSET))
            global_base = atomicAdd_gmem(output_ptr, local_gt_count)
            shared_scalars[3] = global_base

        cute.arch.barrier()
        global_base_gt = shared_scalars[3]

        if tidx == 0:
            shared_scalars[0] = Int32(0)
        cute.arch.barrier()

        # Write > threshold
        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            global_idx = chunk_start + local_idx
            if local_idx < actual_chunk_size and global_idx < N:
                ordered_val = shared_ordered[local_idx]
                if unsigned_gt_16(ordered_val, radix_threshold) != Int32(0):
                    ptr = elem_pointer(shared_scalars, Int32(0))
                    local_offset = atomicAdd_smem(ptr, Int32(1))
                    out_pos = global_base_gt + local_offset
                    if out_pos < k:
                        bf16_bits = ordered16_to_bf16_bits(ordered_val)
                        val_offset = row_values_offset + out_pos * Int32(2)
                        val_ptr = values_ptr + val_offset
                        store_bf16_from_u16(val_ptr, bf16_bits)
                        row_Indices[out_pos] = global_idx

        cute.arch.barrier()

        # Barrier before == collection
        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            red_release_add_gpu(arrival_ptr, Int32(1))

        target = Int32(1) + barrier_phase * num_ctas_per_row
        barrier_phase = barrier_phase + Int32(1)

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            current = ld_acquire_gpu(arrival_ptr)
            while current < target:
                current = ld_acquire_gpu(arrival_ptr)
        cute.arch.barrier()

        # Write == threshold
        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            global_idx = chunk_start + local_idx
            if local_idx < actual_chunk_size and global_idx < N:
                ordered_val = shared_ordered[local_idx]
                ord_masked = ordered_val & Int32(0xFFFF)
                thresh_masked = radix_threshold & Int32(0xFFFF)
                if ord_masked == thresh_masked:
                    output_ptr = elem_pointer(row_State, Int32(OUTPUT_COUNTER_OFFSET))
                    out_pos = atomicAdd_gmem(output_ptr, Int32(1))
                    if out_pos < k:
                        bf16_bits = ordered16_to_bf16_bits(ordered_val)
                        val_offset = row_values_offset + out_pos * Int32(2)
                        val_ptr = values_ptr + val_offset
                        store_bf16_from_u16(val_ptr, bf16_bits)
                        row_Indices[out_pos] = global_idx

        cute.arch.barrier()


# =============================================================================
# F32 Single-CTA Fused Kernel
# =============================================================================


class FusedSingleCTA:
    """
    Single-CTA fused radix select kernel for F32 with:
    - Fused float_to_ordered on load
    - Direct value and index output (UNSORTED)
    """

    NUM_THREADS: cutlass.Constexpr[int] = 256

    def __init__(self, N: int, k: int):
        self.N = N
        self.k = k
        self.elements_per_thread = (N + 255) // 256

    def _smem_size_in_bytes(self) -> int:
        return 256 * 4 + 256 * 4 + 4 * 4 + self.N * 4 + 128

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        values_ptr: cute.Pointer,
        indices_ptr: cute.Pointer,
        M: Int32,
        stream,
    ):
        mInput = cute.make_tensor(
            input_ptr,
            cute.make_layout((M, self.N), stride=(self.N, 1)),
        )
        mValues = cute.make_tensor(
            values_ptr,
            cute.make_layout((M, self.k), stride=(self.k, 1)),
        )
        mIndices = cute.make_tensor(
            indices_ptr,
            cute.make_layout((M, self.k), stride=(self.k, 1)),
        )

        self.kernel(mInput, mValues, mIndices).launch(
            grid=[M, 1, 1],
            block=[self.NUM_THREADS, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mInput: cute.Tensor, mValues: cute.Tensor, mIndices: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        row_idx, _, _ = cute.arch.block_idx()

        NUM_THREADS = self.NUM_THREADS
        N = self.N
        k = self.k

        row_Input = mInput[row_idx, None]
        row_Values = mValues[row_idx, None]
        row_Indices = mIndices[row_idx, None]

        # Shared Memory
        smem = cutlass.utils.SmemAllocator()

        histogram = smem.allocate_tensor(
            Int32, cute.make_layout((256,), stride=(1,)), byte_alignment=16
        )
        suffix_sum = smem.allocate_tensor(
            Int32, cute.make_layout((256,), stride=(1,)), byte_alignment=16
        )
        shared_scalars = smem.allocate_tensor(
            Int32, cute.make_layout((4,), stride=(1,)), byte_alignment=16
        )
        shared_ordered = smem.allocate_tensor(
            Int32, cute.make_layout((N,), stride=(1,)), byte_alignment=16
        )

        # Phase 1: Load and Fuse float_to_ordered
        for i in range(self.elements_per_thread):
            idx = tidx + i * NUM_THREADS
            if idx < N:
                float_val = row_Input[idx]
                bits = _float32_to_bits(float_val)
                ordered = float32_to_ordered(bits)
                shared_ordered[idx] = ordered

        cute.arch.barrier()

        # Initialize scalars
        if tidx == 0:
            shared_scalars[0] = Int32(0)
            shared_scalars[1] = Int32(k)
            shared_scalars[2] = Int32(0)
            shared_scalars[3] = Int32(0)
        cute.arch.barrier()

        # Phase 2: Radix Select (4 rounds)
        # ROUND 0: bits 31-24
        clear_histogram_block(tidx, histogram, NUM_THREADS)
        cute.arch.barrier()

        for i in range(self.elements_per_thread):
            idx = tidx + i * NUM_THREADS
            if idx < N:
                ordered_val = shared_ordered[idx]
                bucket = (ordered_val >> Int32(24)) & Int32(0xFF)
                ptr = elem_pointer(histogram, bucket)
                atomicAdd_smem(ptr, Int32(1))

        cute.arch.barrier()

        if tidx < 256:
            suffix_sum[tidx] = histogram[tidx]
        cute.arch.barrier()

        parallel_suffix_sum_inplace(tidx, suffix_sum)

        remaining_k_r0 = shared_scalars[1]
        find_threshold_bucket_local(tidx, suffix_sum, remaining_k_r0, shared_scalars)

        if tidx == 0:
            shared_scalars[2] = shared_scalars[0]
        cute.arch.barrier()

        # ROUND 1: bits 23-16
        clear_histogram_block(tidx, histogram, NUM_THREADS)
        cute.arch.barrier()

        prefix_r1 = shared_scalars[2]

        for i in range(self.elements_per_thread):
            idx = tidx + i * NUM_THREADS
            if idx < N:
                ordered_val = shared_ordered[idx]
                msb = (ordered_val >> Int32(24)) & Int32(0xFF)
                if msb == prefix_r1:
                    bucket = (ordered_val >> Int32(16)) & Int32(0xFF)
                    ptr = elem_pointer(histogram, bucket)
                    atomicAdd_smem(ptr, Int32(1))

        cute.arch.barrier()

        if tidx < 256:
            suffix_sum[tidx] = histogram[tidx]
        cute.arch.barrier()

        parallel_suffix_sum_inplace(tidx, suffix_sum)

        remaining_k_r1 = shared_scalars[1]
        find_threshold_bucket_local(tidx, suffix_sum, remaining_k_r1, shared_scalars)

        if tidx == 0:
            shared_scalars[2] = (prefix_r1 << Int32(8)) | shared_scalars[0]
        cute.arch.barrier()

        # ROUND 2: bits 15-8
        clear_histogram_block(tidx, histogram, NUM_THREADS)
        cute.arch.barrier()

        prefix_r2 = shared_scalars[2]

        for i in range(self.elements_per_thread):
            idx = tidx + i * NUM_THREADS
            if idx < N:
                ordered_val = shared_ordered[idx]
                top2 = (ordered_val >> Int32(16)) & Int32(0xFFFF)
                if top2 == prefix_r2:
                    bucket = (ordered_val >> Int32(8)) & Int32(0xFF)
                    ptr = elem_pointer(histogram, bucket)
                    atomicAdd_smem(ptr, Int32(1))

        cute.arch.barrier()

        if tidx < 256:
            suffix_sum[tidx] = histogram[tidx]
        cute.arch.barrier()

        parallel_suffix_sum_inplace(tidx, suffix_sum)

        remaining_k_r2 = shared_scalars[1]
        find_threshold_bucket_local(tidx, suffix_sum, remaining_k_r2, shared_scalars)

        if tidx == 0:
            shared_scalars[2] = (prefix_r2 << Int32(8)) | shared_scalars[0]
        cute.arch.barrier()

        # ROUND 3: bits 7-0
        clear_histogram_block(tidx, histogram, NUM_THREADS)
        cute.arch.barrier()

        prefix_r3 = shared_scalars[2]

        for i in range(self.elements_per_thread):
            idx = tidx + i * NUM_THREADS
            if idx < N:
                ordered_val = shared_ordered[idx]
                top3 = (ordered_val >> Int32(8)) & Int32(0xFFFFFF)
                if top3 == prefix_r3:
                    bucket = ordered_val & Int32(0xFF)
                    ptr = elem_pointer(histogram, bucket)
                    atomicAdd_smem(ptr, Int32(1))

        cute.arch.barrier()

        if tidx < 256:
            suffix_sum[tidx] = histogram[tidx]
        cute.arch.barrier()

        parallel_suffix_sum_inplace(tidx, suffix_sum)

        remaining_k_r3 = shared_scalars[1]
        find_threshold_bucket_local(tidx, suffix_sum, remaining_k_r3, shared_scalars)

        if tidx == 0:
            shared_scalars[2] = (prefix_r3 << Int32(8)) | shared_scalars[0]
        cute.arch.barrier()

        # Phase 3: Collection with Direct Output
        threshold = shared_scalars[2]

        if tidx == 0:
            shared_scalars[3] = Int32(0)
        cute.arch.barrier()

        # Collect > threshold elements
        for i in range(self.elements_per_thread):
            idx = tidx + i * NUM_THREADS
            if idx < N:
                ordered_val = shared_ordered[idx]
                if unsigned_gt_32(ordered_val, threshold) != Int32(0):
                    ptr = elem_pointer(shared_scalars, Int32(3))
                    pos = atomicAdd_smem(ptr, Int32(1))
                    if pos < k:
                        float_val = ordered_to_float32(ordered_val)
                        row_Values[pos] = float_val
                        row_Indices[pos] = idx

        cute.arch.barrier()

        # Collect == threshold elements
        for i in range(self.elements_per_thread):
            idx = tidx + i * NUM_THREADS
            if idx < N:
                ordered_val = shared_ordered[idx]
                if ordered_val == threshold:
                    ptr = elem_pointer(shared_scalars, Int32(3))
                    pos = atomicAdd_smem(ptr, Int32(1))
                    if pos < k:
                        float_val = ordered_to_float32(ordered_val)
                        row_Values[pos] = float_val
                        row_Indices[pos] = idx

        cute.arch.barrier()


# =============================================================================
# F32 Multi-CTA Fused Kernel
# =============================================================================


class FusedMultiCTA:
    """
    Multi-CTA fused radix select kernel for F32 with:
    - Fused float_to_ordered on load
    - Direct value and index output (UNSORTED)
    - Inter-CTA coordination via global state
    """

    NUM_THREADS: cutlass.Constexpr[int] = 256

    def __init__(self, N: int, k: int, chunk_size: int, num_ctas_per_row: int):
        self.N = N
        self.k = k
        self.chunk_size = chunk_size
        self.num_ctas_per_row = num_ctas_per_row
        self.elements_per_thread = (chunk_size + 255) // 256

    def _smem_size_in_bytes(self) -> int:
        return 256 * 4 + 256 * 4 + 4 * 4 + self.chunk_size * 4 + 64

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        values_ptr: cute.Pointer,
        indices_ptr: cute.Pointer,
        state_ptr: cute.Pointer,
        M: Int32,
        stream,
    ):
        mInput = cute.make_tensor(
            input_ptr,
            cute.make_layout((M, self.N), stride=(self.N, 1)),
        )
        mValues = cute.make_tensor(
            values_ptr,
            cute.make_layout((M, self.k), stride=(self.k, 1)),
        )
        mIndices = cute.make_tensor(
            indices_ptr,
            cute.make_layout((M, self.k), stride=(self.k, 1)),
        )
        mState = cute.make_tensor(
            state_ptr,
            cute.make_layout((M, STATE_SIZE_INT32), stride=(STATE_SIZE_INT32, 1)),
        )

        total_ctas = M * self.num_ctas_per_row

        self.kernel(mInput, mValues, mIndices, mState).launch(
            grid=[total_ctas, 1, 1],
            block=[self.NUM_THREADS, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        mState: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        global_cta_id, _, _ = cute.arch.block_idx()

        NUM_THREADS = self.NUM_THREADS
        N = self.N
        k = self.k
        chunk_size = self.chunk_size
        num_ctas_per_row = self.num_ctas_per_row

        row_idx = global_cta_id // num_ctas_per_row
        cta_in_row = global_cta_id % num_ctas_per_row

        row_Input = mInput[row_idx, None]
        row_Values = mValues[row_idx, None]
        row_Indices = mIndices[row_idx, None]
        row_State = mState[row_idx, None]

        chunk_start = cta_in_row * chunk_size
        actual_chunk_size = chunk_size
        if chunk_start + chunk_size > N:
            actual_chunk_size = N - chunk_start
            if actual_chunk_size < 0:
                actual_chunk_size = 0

        # Shared Memory
        smem = cutlass.utils.SmemAllocator()

        local_histogram = smem.allocate_tensor(
            Int32, cute.make_layout((256,), stride=(1,)), byte_alignment=16
        )
        suffix_sum = smem.allocate_tensor(
            Int32, cute.make_layout((256,), stride=(1,)), byte_alignment=16
        )
        shared_scalars = smem.allocate_tensor(
            Int32, cute.make_layout((4,), stride=(1,)), byte_alignment=16
        )
        shared_ordered = smem.allocate_tensor(
            Int32, cute.make_layout((chunk_size,), stride=(1,)), byte_alignment=16
        )

        # Load chunk with fused float_to_ordered
        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            global_idx = chunk_start + local_idx
            if local_idx < actual_chunk_size and global_idx < N:
                float_val = row_Input[global_idx]
                bits = _float32_to_bits(float_val)
                ordered = float32_to_ordered(bits)
                shared_ordered[local_idx] = ordered

        cute.arch.barrier()

        # Initialize: CTA 0 clears global state
        if cta_in_row == 0:
            for buf in cutlass.range_constexpr(3):
                if tidx < 256:
                    row_State[HISTOGRAM_OFFSET + buf * 256 + tidx] = Int32(0)

            cute.arch.barrier()

            if tidx == 0:
                row_State[ARRIVAL_COUNTER_OFFSET] = Int32(0)
                row_State[OUTPUT_COUNTER_OFFSET] = Int32(0)
                arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
                red_release_add_gpu(arrival_ptr, Int32(1))
        else:
            cute.arch.barrier()

        # Wait for initialization
        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            current = ld_acquire_gpu(arrival_ptr)
            while current < Int32(1):
                current = ld_acquire_gpu(arrival_ptr)
        cute.arch.barrier()

        # Initialize local scalars
        if tidx == 0:
            shared_scalars[0] = Int32(0)
            shared_scalars[1] = Int32(k)
            shared_scalars[2] = Int32(0)
            shared_scalars[3] = Int32(0)
        cute.arch.barrier()

        barrier_phase = Int32(1)

        # ROUND 0
        clear_histogram_block(tidx, local_histogram, NUM_THREADS)
        cute.arch.barrier()

        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            if local_idx < actual_chunk_size:
                ordered_val = shared_ordered[local_idx]
                bucket = (ordered_val >> Int32(24)) & Int32(0xFF)
                ptr = elem_pointer(local_histogram, bucket)
                atomicAdd_smem(ptr, Int32(1))

        cute.arch.barrier()

        hist_base = HISTOGRAM_OFFSET + 0 * 256
        if tidx < 256:
            local_count = local_histogram[tidx]
            if local_count > Int32(0):
                global_hist_ptr = elem_pointer(row_State, Int32(hist_base + tidx))
                atomicAdd_gmem(global_hist_ptr, local_count)

        if cta_in_row == 0 and tidx < 256:
            row_State[HISTOGRAM_OFFSET + 1 * 256 + tidx] = Int32(0)

        cute.arch.barrier()

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            red_release_add_gpu(arrival_ptr, Int32(1))

        target = Int32(1) + barrier_phase * num_ctas_per_row
        barrier_phase = barrier_phase + Int32(1)

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            current = ld_acquire_gpu(arrival_ptr)
            while current < target:
                current = ld_acquire_gpu(arrival_ptr)
        cute.arch.barrier()

        if tidx < 256:
            suffix_sum[tidx] = row_State[hist_base + tidx]
        cute.arch.barrier()

        parallel_suffix_sum_inplace(tidx, suffix_sum)

        remaining_k_r0 = shared_scalars[1]
        find_threshold_bucket_local(tidx, suffix_sum, remaining_k_r0, shared_scalars)

        if tidx == 0:
            shared_scalars[2] = shared_scalars[0]
        cute.arch.barrier()

        # ROUND 1
        clear_histogram_block(tidx, local_histogram, NUM_THREADS)
        cute.arch.barrier()

        prefix_r1 = shared_scalars[2]

        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            if local_idx < actual_chunk_size:
                ordered_val = shared_ordered[local_idx]
                msb = (ordered_val >> Int32(24)) & Int32(0xFF)
                if msb == prefix_r1:
                    bucket = (ordered_val >> Int32(16)) & Int32(0xFF)
                    ptr = elem_pointer(local_histogram, bucket)
                    atomicAdd_smem(ptr, Int32(1))

        cute.arch.barrier()

        hist_base = HISTOGRAM_OFFSET + 1 * 256
        if tidx < 256:
            local_count = local_histogram[tidx]
            if local_count > Int32(0):
                global_hist_ptr = elem_pointer(row_State, Int32(hist_base + tidx))
                atomicAdd_gmem(global_hist_ptr, local_count)

        if cta_in_row == 0 and tidx < 256:
            row_State[HISTOGRAM_OFFSET + 2 * 256 + tidx] = Int32(0)

        cute.arch.barrier()

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            red_release_add_gpu(arrival_ptr, Int32(1))

        target = Int32(1) + barrier_phase * num_ctas_per_row
        barrier_phase = barrier_phase + Int32(1)

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            current = ld_acquire_gpu(arrival_ptr)
            while current < target:
                current = ld_acquire_gpu(arrival_ptr)
        cute.arch.barrier()

        if tidx < 256:
            suffix_sum[tidx] = row_State[hist_base + tidx]
        cute.arch.barrier()

        parallel_suffix_sum_inplace(tidx, suffix_sum)
        remaining_k_r1 = shared_scalars[1]
        find_threshold_bucket_local(tidx, suffix_sum, remaining_k_r1, shared_scalars)

        if tidx == 0:
            shared_scalars[2] = (prefix_r1 << Int32(8)) | shared_scalars[0]
        cute.arch.barrier()

        # ROUND 2
        clear_histogram_block(tidx, local_histogram, NUM_THREADS)
        cute.arch.barrier()

        prefix_r2 = shared_scalars[2]

        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            if local_idx < actual_chunk_size:
                ordered_val = shared_ordered[local_idx]
                top2 = (ordered_val >> Int32(16)) & Int32(0xFFFF)
                if top2 == prefix_r2:
                    bucket = (ordered_val >> Int32(8)) & Int32(0xFF)
                    ptr = elem_pointer(local_histogram, bucket)
                    atomicAdd_smem(ptr, Int32(1))

        cute.arch.barrier()

        hist_base = HISTOGRAM_OFFSET + 2 * 256
        if tidx < 256:
            local_count = local_histogram[tidx]
            if local_count > Int32(0):
                global_hist_ptr = elem_pointer(row_State, Int32(hist_base + tidx))
                atomicAdd_gmem(global_hist_ptr, local_count)

        if cta_in_row == 0 and tidx < 256:
            row_State[HISTOGRAM_OFFSET + 0 * 256 + tidx] = Int32(0)

        cute.arch.barrier()

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            red_release_add_gpu(arrival_ptr, Int32(1))

        target = Int32(1) + barrier_phase * num_ctas_per_row
        barrier_phase = barrier_phase + Int32(1)

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            current = ld_acquire_gpu(arrival_ptr)
            while current < target:
                current = ld_acquire_gpu(arrival_ptr)
        cute.arch.barrier()

        if tidx < 256:
            suffix_sum[tidx] = row_State[hist_base + tidx]
        cute.arch.barrier()

        parallel_suffix_sum_inplace(tidx, suffix_sum)
        remaining_k_r2 = shared_scalars[1]
        find_threshold_bucket_local(tidx, suffix_sum, remaining_k_r2, shared_scalars)

        if tidx == 0:
            shared_scalars[2] = (prefix_r2 << Int32(8)) | shared_scalars[0]
        cute.arch.barrier()

        # ROUND 3
        clear_histogram_block(tidx, local_histogram, NUM_THREADS)
        cute.arch.barrier()

        prefix_r3 = shared_scalars[2]

        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            if local_idx < actual_chunk_size:
                ordered_val = shared_ordered[local_idx]
                top3 = (ordered_val >> Int32(8)) & Int32(0xFFFFFF)
                if top3 == prefix_r3:
                    bucket = ordered_val & Int32(0xFF)
                    ptr = elem_pointer(local_histogram, bucket)
                    atomicAdd_smem(ptr, Int32(1))

        cute.arch.barrier()

        hist_base = HISTOGRAM_OFFSET + 0 * 256
        if tidx < 256:
            local_count = local_histogram[tidx]
            if local_count > Int32(0):
                global_hist_ptr = elem_pointer(row_State, Int32(hist_base + tidx))
                atomicAdd_gmem(global_hist_ptr, local_count)

        cute.arch.barrier()

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            red_release_add_gpu(arrival_ptr, Int32(1))

        target = Int32(1) + barrier_phase * num_ctas_per_row
        barrier_phase = barrier_phase + Int32(1)

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            current = ld_acquire_gpu(arrival_ptr)
            while current < target:
                current = ld_acquire_gpu(arrival_ptr)
        cute.arch.barrier()

        if tidx < 256:
            suffix_sum[tidx] = row_State[hist_base + tidx]
        cute.arch.barrier()

        parallel_suffix_sum_inplace(tidx, suffix_sum)
        remaining_k_r3 = shared_scalars[1]
        find_threshold_bucket_local(tidx, suffix_sum, remaining_k_r3, shared_scalars)

        if tidx == 0:
            shared_scalars[2] = (prefix_r3 << Int32(8)) | shared_scalars[0]
        cute.arch.barrier()

        # COLLECTION with direct output
        threshold = shared_scalars[2]

        if cta_in_row == 0 and tidx == 0:
            row_State[OUTPUT_COUNTER_OFFSET] = Int32(0)

        cute.arch.barrier()

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            red_release_add_gpu(arrival_ptr, Int32(1))

        target = Int32(1) + barrier_phase * num_ctas_per_row
        barrier_phase = barrier_phase + Int32(1)

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            current = ld_acquire_gpu(arrival_ptr)
            while current < target:
                current = ld_acquire_gpu(arrival_ptr)
        cute.arch.barrier()

        # Count local > threshold
        if tidx == 0:
            shared_scalars[3] = Int32(0)
        cute.arch.barrier()

        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            if local_idx < actual_chunk_size:
                ordered_val = shared_ordered[local_idx]
                if unsigned_gt_32(ordered_val, threshold) != Int32(0):
                    ptr = elem_pointer(shared_scalars, Int32(3))
                    atomicAdd_smem(ptr, Int32(1))

        cute.arch.barrier()

        local_gt_count = shared_scalars[3]

        # Reserve global positions
        if tidx == 0 and local_gt_count > Int32(0):
            output_ptr = elem_pointer(row_State, Int32(OUTPUT_COUNTER_OFFSET))
            global_base = atomicAdd_gmem(output_ptr, local_gt_count)
            shared_scalars[3] = global_base

        cute.arch.barrier()
        global_base_gt = shared_scalars[3]

        if tidx == 0:
            shared_scalars[0] = Int32(0)
        cute.arch.barrier()

        # Write > threshold elements with direct output
        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            global_idx = chunk_start + local_idx
            if local_idx < actual_chunk_size and global_idx < N:
                ordered_val = shared_ordered[local_idx]
                if unsigned_gt_32(ordered_val, threshold) != Int32(0):
                    ptr = elem_pointer(shared_scalars, Int32(0))
                    local_offset = atomicAdd_smem(ptr, Int32(1))
                    out_pos = global_base_gt + local_offset
                    if out_pos < k:
                        float_val = ordered_to_float32(ordered_val)
                        row_Values[out_pos] = float_val
                        row_Indices[out_pos] = global_idx

        cute.arch.barrier()

        # Barrier before == collection
        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            red_release_add_gpu(arrival_ptr, Int32(1))

        target = Int32(1) + barrier_phase * num_ctas_per_row
        barrier_phase = barrier_phase + Int32(1)

        if tidx == 0:
            arrival_ptr = elem_pointer(row_State, Int32(ARRIVAL_COUNTER_OFFSET))
            current = ld_acquire_gpu(arrival_ptr)
            while current < target:
                current = ld_acquire_gpu(arrival_ptr)
        cute.arch.barrier()

        # Write == threshold elements
        for i in range(self.elements_per_thread):
            local_idx = tidx + i * NUM_THREADS
            global_idx = chunk_start + local_idx
            if local_idx < actual_chunk_size and global_idx < N:
                ordered_val = shared_ordered[local_idx]
                if ordered_val == threshold:
                    output_ptr = elem_pointer(row_State, Int32(OUTPUT_COUNTER_OFFSET))
                    out_pos = atomicAdd_gmem(output_ptr, Int32(1))
                    if out_pos < k:
                        float_val = ordered_to_float32(ordered_val)
                        row_Values[out_pos] = float_val
                        row_Indices[out_pos] = global_idx

        cute.arch.barrier()


# =============================================================================
# F32 Compilation Cache
# =============================================================================


@functools.cache
def _get_f32_single_cta_kernel(N: int, k: int):
    """Get or compile F32 single-CTA kernel."""
    kernel_instance = FusedSingleCTA(N, k)

    compiled = cute.compile(
        kernel_instance,
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int32(1),
        cutlass_torch.current_stream(),
    )

    def tensor_api(
        x: torch.Tensor, values_out: torch.Tensor, indices_out: torch.Tensor, M: int
    ) -> None:
        compiled(
            make_ptr(cutlass.Float32, x.data_ptr()),
            make_ptr(cutlass.Float32, values_out.data_ptr()),
            make_ptr(cutlass.Int32, indices_out.data_ptr()),
            Int32(M),
            cutlass_torch.current_stream(),
        )

    return tensor_api


@functools.cache
def _get_f32_multi_cta_kernel(N: int, k: int, chunk_size: int, num_ctas_per_row: int):
    """Get or compile F32 multi-CTA kernel."""
    kernel_instance = FusedMultiCTA(N, k, chunk_size, num_ctas_per_row)

    compiled = cute.compile(
        kernel_instance,
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int32(1),
        cutlass_torch.current_stream(),
    )

    def tensor_api(
        x: torch.Tensor,
        values_out: torch.Tensor,
        indices_out: torch.Tensor,
        state: torch.Tensor,
        M: int,
    ) -> None:
        compiled(
            make_ptr(cutlass.Float32, x.data_ptr()),
            make_ptr(cutlass.Float32, values_out.data_ptr()),
            make_ptr(cutlass.Int32, indices_out.data_ptr()),
            make_ptr(cutlass.Int32, state.data_ptr()),
            Int32(M),
            cutlass_torch.current_stream(),
        )

    return tensor_api


# =============================================================================
# BF16 Compilation Cache
# =============================================================================


@functools.cache
def _get_bf16_single_cta_kernel(N: int, k: int):
    """Get or compile native BF16 single-CTA kernel."""
    kernel_instance = NativeBF16SingleCTA(N, k)

    compiled = cute.compile(
        kernel_instance,
        make_ptr(cutlass.Int8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int32(1),
        cutlass_torch.current_stream(),
    )

    def tensor_api(
        x: torch.Tensor, values_out: torch.Tensor, indices_out: torch.Tensor, M: int
    ) -> None:
        compiled(
            make_ptr(cutlass.Int8, x.data_ptr()),
            make_ptr(cutlass.Int8, values_out.data_ptr()),
            make_ptr(cutlass.Int32, indices_out.data_ptr()),
            Int32(M),
            cutlass_torch.current_stream(),
        )

    return tensor_api


@functools.cache
def _get_bf16_multi_cta_kernel(N: int, k: int, chunk_size: int, num_ctas_per_row: int):
    """Get or compile native BF16 multi-CTA kernel."""
    kernel_instance = NativeBF16MultiCTA(N, k, chunk_size, num_ctas_per_row)

    compiled = cute.compile(
        kernel_instance,
        make_ptr(cutlass.Int8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int32(1),
        cutlass_torch.current_stream(),
    )

    def tensor_api(
        x: torch.Tensor,
        values_out: torch.Tensor,
        indices_out: torch.Tensor,
        state: torch.Tensor,
        M: int,
    ) -> None:
        compiled(
            make_ptr(cutlass.Int8, x.data_ptr()),
            make_ptr(cutlass.Int8, values_out.data_ptr()),
            make_ptr(cutlass.Int32, indices_out.data_ptr()),
            make_ptr(cutlass.Int32, state.data_ptr()),
            Int32(M),
            cutlass_torch.current_stream(),
        )

    return tensor_api


# =============================================================================
# Low-Overhead API (internal)
# =============================================================================


def _top_k_cute_dsl_impl(
    x: torch.Tensor,
    values_out: torch.Tensor,
    indices_out: torch.Tensor,
    k: int,
    sorted: bool = True,
    state: Optional[torch.Tensor] = None,
) -> None:
    """
    Low-overhead API for CuTe DSL top-k selection.

    Routes to:
    - Native BF16 kernel (2 rounds) for bfloat16 input
    - F32 kernel (4 rounds) for float32 input

    Args:
        x: Input tensor [M, N] with dtype float32 or bfloat16
        values_out: PRE-ALLOCATED output values [M, k] with same dtype as x
        indices_out: PRE-ALLOCATED output indices [M, k] with dtype int32
        k: Number of top elements
        sorted: Whether to return sorted output (uses torch.sort post-kernel)
        state: PRE-ALLOCATED state tensor for multi-CTA [M, STATE_SIZE_INT32] int32
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"

    M, N = x.shape
    device = x.device

    assert 0 < k <= N, f"k must be in (0, N], got k={k}, N={N}"
    assert values_out.shape == (M, k)
    assert indices_out.shape == (M, k)
    assert indices_out.dtype == torch.int32

    x = x.contiguous()

    if x.dtype == torch.bfloat16:
        # Native BF16 path - 2 radix rounds
        if N <= MAX_N_SINGLE_CTA:
            kernel = _get_bf16_single_cta_kernel(N, k)
            kernel(x, values_out, indices_out, M)
        else:
            chunk_size = CHUNK_SIZE_DEFAULT
            num_ctas_per_row = (N + chunk_size - 1) // chunk_size

            if state is None:
                state = torch.zeros(M, STATE_SIZE_INT32, dtype=torch.int32, device=device)
            else:
                state.zero_()

            kernel = _get_bf16_multi_cta_kernel(N, k, chunk_size, num_ctas_per_row)
            kernel(x, values_out, indices_out, state, M)
    elif x.dtype == torch.float32:
        # F32 path - 4 radix rounds
        if N <= MAX_N_SINGLE_CTA:
            kernel = _get_f32_single_cta_kernel(N, k)
            kernel(x, values_out, indices_out, M)
        else:
            chunk_size = CHUNK_SIZE_DEFAULT
            num_ctas_per_row = (N + chunk_size - 1) // chunk_size

            if state is None:
                state = torch.zeros(M, STATE_SIZE_INT32, dtype=torch.int32, device=device)
            else:
                state.zero_()

            kernel = _get_f32_multi_cta_kernel(N, k, chunk_size, num_ctas_per_row)
            kernel(x, values_out, indices_out, state, M)
    else:
        raise ValueError(
            f"CuTe DSL top_k backend only supports float32 and bfloat16, got {x.dtype}"
        )

    # Post-kernel sorting
    if sorted:
        sorted_values, sort_indices = torch.sort(values_out, dim=-1, descending=True)
        sorted_indices = torch.gather(indices_out, dim=-1, index=sort_indices)
        values_out.copy_(sorted_values)
        indices_out.copy_(sorted_indices)


# =============================================================================
# Public API
# =============================================================================


def topk_cute_dsl(
    x: torch.Tensor,
    k: int,
    sorted: bool = False,
    state_buffer: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CuTe DSL implementation of radix-based top-K selection.

    This function selects the top-k largest elements from each row of the input
    tensor using a CuTe DSL radix-based selection algorithm.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape ``(batch_size, d)`` containing the values to select from.
        Supported dtypes: ``float32``, ``bfloat16``.
        Note: ``float16`` is NOT supported - use the CUDA backend instead.
    k : int
        Number of top elements to select from each row.
    sorted : bool, optional
        If True, the returned top-k elements will be sorted in descending order.
        Default is False (unsorted, which is faster).
    state_buffer : torch.Tensor, optional
        Pre-allocated state buffer for multi-CTA path. If None, will be allocated
        internally when needed (for N > 16384).

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
        If input dtype is float16 (not supported by CuTe DSL backend).

    Note
    ----
    - For BF16 inputs, uses native 16-bit processing with only 2 radix rounds
      (faster than F32 which requires 4 rounds).
    - For large N (> 16384), uses multi-CTA path with inter-CTA coordination.
    """
    is_1d = x.dim() == 1
    if is_1d:
        x = x.unsqueeze(0)

    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"
    assert x.is_cuda, "Input must be on CUDA"

    if x.dtype == torch.float16:
        raise ValueError(
            "CuTe DSL top_k backend does not support float16. "
            "Use backend='cuda' for float16 inputs."
        )

    if x.dtype not in [torch.float32, torch.bfloat16]:
        raise ValueError(
            f"CuTe DSL top_k backend only supports float32 and bfloat16, got {x.dtype}"
        )

    M, N = x.shape
    device = x.device
    dtype = x.dtype

    assert 0 < k <= N, f"k must be in (0, N], got k={k}, N={N}"

    values_out = torch.empty(M, k, dtype=dtype, device=device)
    indices_out = torch.empty(M, k, dtype=torch.int32, device=device)

    _top_k_cute_dsl_impl(x, values_out, indices_out, k, sorted, state_buffer)

    # Convert to int64 for PyTorch compatibility
    indices = indices_out.long()

    if is_1d:
        values_out = values_out.squeeze(0)
        indices = indices.squeeze(0)

    return values_out, indices


__all__ = [
    "topk_cute_dsl",
    # Kernel classes (for advanced users)
    "NativeBF16SingleCTA",
    "NativeBF16MultiCTA",
    "FusedSingleCTA",
    "FusedMultiCTA",
    # Constants
    "STATE_SIZE_INT32",
    "MIN_N_MULTI_CTA",
    "MAX_N_SINGLE_CTA",
]

