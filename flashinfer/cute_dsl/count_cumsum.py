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

Count-Cumsum CuTe-DSL Kernel for MoE Routing
=============================================

This module provides a multi-block GPU kernel for computing histogram counts
and cumulative sums, essential for Mixture of Experts (MoE) token routing.

Key Features
------------
1. **Multi-block parallelism**: Scales with input size N
2. **Per-block shared memory histogram**: Reduces global atomic contention
3. **Global atomic aggregation**: Combines local histograms efficiently
4. **"Last block wins" cumsum**: No cooperative groups needed
5. **TVM-FFI enabled**: Efficient tensor passing without pointer construction

Algorithm
---------
Phase 1 (All blocks in parallel):
1. Each block processes a chunk of N/num_blocks elements
2. Build local histogram in shared memory (atomicAdd_smem)
3. Sync threads within block
4. Atomic add local histogram to global count (atomicAdd_gmem)
5. Atomic increment "done" counter

Phase 2 (Last block only):
6. The block that sees done_counter == num_blocks - 1 is "last"
7. Last block performs sequential cumsum over global count
8. Write cumsum to output

Use Case
--------
In MoE layers, after the router selects top-k experts for each token,
we need to:
1. Count how many tokens go to each expert (histogram)
2. Compute prefix sums to determine where each expert's tokens start (cumsum)

This kernel fuses both operations for efficiency.

Constraints
-----------
- E: 4 to 50,000 (must be divisible by 4)
- N: any size (scales with multiple blocks)
- dtype: int32 or int64 input
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch

import cutlass
from cutlass import Int32, Int64
from cutlass import cute

# For atomic operations
from cutlass._mlir.dialects import nvvm
from cutlass._mlir.dialects.nvvm import AtomicOpKind, MemOrderKind, MemScopeKind
from cutlass.cutlass_dsl import T, dsl_user_op

from ..api_logging import flashinfer_api


# =============================================================================
# Atomic Operations
# =============================================================================


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: Int32, *, loc=None, ip=None) -> cute.Pointer:
    """Get pointer to element at 1D coordinate in tensor."""
    return x.iterator + coord


@dsl_user_op
def atomicAdd_smem(dst_ptr: cute.Pointer, val: Int32, *, loc=None, ip=None) -> Int32:
    """
    Atomic add to shared memory with CTA scope.
    Returns the old value before the add.
    """
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
    """
    Atomic add to global memory with GPU scope.
    Returns the old value before the add.

    Used for:
    1. Aggregating local histograms to global count
    2. Incrementing "done" counter for last-block detection
    """
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


# =============================================================================
# Kernel Configuration
# =============================================================================

# Threads per block - 256 is a good balance
BLOCK_SIZE = 256

# Elements per block - chunk size for each block to process
# Larger = fewer blocks, less atomic contention
# Smaller = more parallelism
CHUNK_SIZE = 4096


# =============================================================================
# Helper Functions for TVM-FFI
# =============================================================================


def _make_fake_tensor(dtype, shape, divisibility=1):
    """Create a fake tensor for CuTe-DSL compilation with TVM-FFI."""
    if dtype is None:
        return None
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=(*[cute.sym_int64(divisibility=divisibility)] * (len(shape) - 1), 1),
        assumed_align=divisibility * dtype.width // 8,
    )


# =============================================================================
# CuTe-DSL Kernel Class
# =============================================================================


class CountCumsumKernel:
    """
    Multi-block CuTe-DSL kernel for count_cumsum.

    Key features:
    - Multiple blocks for parallel processing of large N
    - Per-block shared memory histogram
    - Global atomic aggregation
    - "Last block wins" pattern for cumsum (no cooperative groups)
    - TVM-FFI enabled for efficient tensor passing
    """

    def __init__(self, E: int, N: int, do_cumsum: bool, input_dtype: torch.dtype):
        """
        Initialize kernel configuration.

        Parameters
        ----------
        E : int
            Number of histogram bins (experts)
        N : int
            Number of input elements
        do_cumsum : bool
            Whether to compute cumulative sum
        input_dtype : torch.dtype
            Input tensor dtype (int32 or int64)
        """
        self.E = E
        self.N = N
        self.do_cumsum = do_cumsum
        self.input_dtype = input_dtype

        # Calculate number of blocks needed
        # Each block processes CHUNK_SIZE elements
        self.num_blocks = (N + CHUNK_SIZE - 1) // CHUNK_SIZE

        # Validate constraints
        assert 4 <= E <= 50000, f"E must be in [4, 50000], got {E}"
        assert E % 4 == 0, f"E must be divisible by 4, got {E}"

    def _shared_mem_bytes(self) -> int:
        """
        Calculate shared memory requirement in bytes.

        Shared memory layout:
        - [0, E): Local histogram (E * 4 bytes)
        """
        return self.E * 4

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,   # Input indices tensor (N,)
        mCount: cute.Tensor,   # Output count tensor (E,) - must be pre-zeroed!
        mCumsum: cute.Tensor,  # Output cumsum tensor (E,)
        mDone: cute.Tensor,    # Done counter tensor (1,) - must be pre-zeroed!
        stream,
    ):
        """
        JIT launcher for multi-block count_cumsum kernel.

        IMPORTANT: mCount and mDone must be zeroed tensors!
        The kernel uses atomic adds, so non-zero initial values will corrupt results.

        This method launches multiple blocks, each processing CHUNK_SIZE elements.
        """
        # Calculate shared memory size
        smem_bytes = self._shared_mem_bytes()

        # Launch kernel with multiple blocks
        self.kernel(mInput, mCount, mCumsum, mDone).launch(
            grid=(self.num_blocks, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,  # Input indices tensor (N,)
        mCount: cute.Tensor,  # Global count tensor (E,) - pre-zeroed
        mCumsum: cute.Tensor,  # Output cumsum tensor (E,)
        mDone: cute.Tensor,  # Done counter tensor (1,) - pre-zeroed
    ):
        """
        Device kernel for multi-block count_cumsum.

        Each block:
        1. Processes a chunk of input (block_start to block_end)
        2. Builds local histogram in shared memory
        3. Atomically adds to global count
        4. Increments done counter
        5. If last block, computes cumsum
        """
        # =========================================================================
        # Step 0: Get thread/block info and config
        # =========================================================================
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]

        E = self.E
        N = self.N
        do_cumsum = self.do_cumsum
        num_blocks = self.num_blocks

        # =========================================================================
        # Step 1: Calculate this block's input range
        # =========================================================================
        # Each block processes CHUNK_SIZE elements
        block_start = bidx * CHUNK_SIZE
        block_end = block_start + CHUNK_SIZE
        # Clamp to N
        if block_end > N:
            block_end = N

        # =========================================================================
        # Step 2: Allocate and initialize shared memory histogram
        # =========================================================================
        smem_ptr = cute.arch.get_dyn_smem(Int32)
        smem_layout = cute.make_layout((E,))
        smem_hist = cute.make_tensor(smem_ptr, smem_layout)

        # Initialize local histogram to zeros
        # Each thread initializes E/BLOCK_SIZE bins with striding
        for i in cutlass.range(0, E, BLOCK_SIZE, unroll_count=4):
            idx = i + tidx
            if idx < E:
                smem_hist[idx] = Int32(0)

        cute.arch.sync_threads()

        # =========================================================================
        # Step 3: Build local histogram with atomic increments
        # =========================================================================
        # Each thread processes (block_end - block_start) / BLOCK_SIZE elements
        # Using striding pattern for coalesced memory access
        for i in cutlass.range(block_start, block_end, BLOCK_SIZE, unroll_count=4):
            idx = i + tidx
            if idx < block_end:
                # Load input element (expert index)
                expert_idx = mInput[idx]

                # Atomically increment local histogram bin
                bin_ptr = elem_pointer(smem_hist, Int32(expert_idx))
                atomicAdd_smem(bin_ptr, Int32(1))

        cute.arch.sync_threads()

        # =========================================================================
        # Step 4: Atomic add local histogram to global count
        # =========================================================================
        # Each thread adds its portion of the local histogram to global
        for i in cutlass.range(0, E, BLOCK_SIZE, unroll_count=4):
            idx = i + tidx
            if idx < E:
                local_count = smem_hist[idx]
                # Only add if non-zero (small optimization)
                if local_count > 0:
                    global_ptr = elem_pointer(mCount, Int32(idx))
                    atomicAdd_gmem(global_ptr, local_count)

        # Ensure all threads have finished their atomic adds
        cute.arch.sync_threads()

        # =========================================================================
        # Step 5: Thread 0 increments done counter, checks if last block
        # =========================================================================
        # "Last block wins" pattern:
        # - Each block's thread 0 atomically increments done counter
        # - The block that gets (num_blocks - 1) is the last to finish
        # - That block performs the cumsum

        if cutlass.const_expr(do_cumsum):
            if tidx == 0:
                # Atomic increment returns OLD value
                done_ptr = elem_pointer(mDone, Int32(0))
                old_done = atomicAdd_gmem(done_ptr, Int32(1))

                # Check if we're the last block
                # If old_done == num_blocks - 1, all other blocks have finished
                if old_done == num_blocks - 1:
                    # =========================================================
                    # Step 6: Last block computes cumsum (sequential)
                    # =========================================================
                    # At this point, all blocks have written to mCount
                    # We can safely read mCount and compute cumsum
                    #
                    # Note: This is O(E) sequential, but only runs once.
                    # For production, could use parallel prefix sum.

                    running_sum = Int32(0)
                    for j in cutlass.range(E, unroll_count=8):
                        count_val = mCount[j]
                        running_sum = running_sum + count_val
                        mCumsum[j] = running_sum


# =============================================================================
# Kernel Cache and Compilation (TVM-FFI enabled)
# =============================================================================

_kernel_cache: dict = {}


def _get_compiled_kernel(E: int, N: int, do_cumsum: bool, input_dtype: torch.dtype):
    """
    Get or compile a kernel for the given configuration.

    Uses TVM-FFI for efficient tensor passing without manual pointer construction.
    """
    key = (E, N, do_cumsum, input_dtype)
    if key not in _kernel_cache:
        # Create kernel object
        kernel_obj = CountCumsumKernel(E, N, do_cumsum, input_dtype)

        # Determine CUTLASS dtype for input
        if input_dtype == torch.int32:
            cutlass_dtype = cutlass.Int32
        else:
            cutlass_dtype = cutlass.Int64

        # Calculate divisibility for alignment
        div_input = math.gcd(128 // cutlass_dtype.width, N)
        div_count = math.gcd(128 // Int32.width, E)
        div_done = 1  # Single element

        # Create fake tensors for TVM-FFI compilation
        input_fake = _make_fake_tensor(cutlass_dtype, (N,), div_input)
        count_fake = _make_fake_tensor(Int32, (E,), div_count)
        cumsum_fake = _make_fake_tensor(Int32, (E,), div_count)
        done_fake = _make_fake_tensor(Int32, (1,), div_done)

        # Compile with TVM-FFI enabled
        compiled_fn = cute.compile(
            kernel_obj,
            input_fake,
            count_fake,
            cumsum_fake,
            done_fake,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

        def tensor_api(
            x: torch.Tensor,
            count_out: torch.Tensor,
            cumsum_out: torch.Tensor,
            done: torch.Tensor,
        ) -> None:
            """Runtime API that passes torch tensors directly via TVM-FFI."""
            compiled_fn(x, count_out, cumsum_out, done)

        _kernel_cache[key] = tensor_api

    return _kernel_cache[key]


# =============================================================================
# Public API
# =============================================================================


@flashinfer_api
def count_cumsum(
    x: torch.Tensor,
    E: int,
    do_cumsum: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute histogram and optional cumulative sum using CuTe-DSL kernel.

    This is optimized for MoE routing where we need to count how many tokens
    go to each expert and compute prefix sums for token organization.

    Parameters
    ----------
    x : torch.Tensor
        1D tensor of expert indices. Shape: ``(N,)``.
        Supported dtypes: ``int32`` (torch.int), ``int64`` (torch.long).
        Must be on CUDA device. Values must be in range ``[0, E)``.
    E : int
        Number of experts (histogram bins).
        Must be in range ``[4, 50000]`` and divisible by 4.
    do_cumsum : bool, optional
        If ``True`` (default), also compute cumulative sum of counts.

    Returns
    -------
    count : torch.Tensor
        Histogram of expert assignments. Shape: ``(E,)``, dtype: ``int32``.
        ``count[i]`` = number of tokens assigned to expert ``i``.
    cumsum : torch.Tensor (only if ``do_cumsum=True``)
        Cumulative sum of count. Shape: ``(E,)``, dtype: ``int32``.
        ``cumsum[i]`` = sum of ``count[0:i+1]``.

    Examples
    --------
    Basic histogram:

    >>> x = torch.tensor([0, 1, 0, 2, 1, 0], device='cuda', dtype=torch.int)
    >>> count = count_cumsum(x, E=4, do_cumsum=False)
    >>> count  # tensor([3, 2, 1, 0], dtype=torch.int32)

    Histogram with cumsum (for MoE routing):

    >>> x = torch.tensor([0, 1, 0, 2, 1, 0], device='cuda', dtype=torch.int)
    >>> count, cumsum = count_cumsum(x, E=4, do_cumsum=True)
    >>> count   # tensor([3, 2, 1, 0], dtype=torch.int32)
    >>> cumsum  # tensor([3, 5, 6, 6], dtype=torch.int32)

    The cumsum tells us:
    - Expert 0's tokens end at index 3 (has 3 tokens)
    - Expert 1's tokens end at index 5 (has 2 tokens)
    - Expert 2's tokens end at index 6 (has 1 token)
    - Expert 3 has no tokens

    Notes
    -----
    - Uses shared memory histogram per block to reduce atomic contention
    - "Last block wins" pattern for cumsum computation
    - Output count tensor is automatically zeroed before kernel execution
    - Uses TVM-FFI for efficient tensor passing without pointer construction
    """
    # =========================================================================
    # Input Validation
    # =========================================================================
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.dim() == 1, f"Input must be 1D, got {x.dim()}D"
    assert x.dtype in [torch.int32, torch.int64], (
        f"Input dtype must be int32 or int64, got {x.dtype}"
    )
    assert 4 <= E <= 50000, f"E must be in [4, 50000], got {E}"
    assert E % 4 == 0, f"E must be divisible by 4, got {E}"

    N = x.numel()

    # Ensure input is contiguous
    x = x.contiguous()

    # =========================================================================
    # Allocate Output Tensors
    # =========================================================================
    # IMPORTANT: count_output must be zeroed since we use atomic adds!
    count_output = torch.zeros(E, dtype=torch.int32, device=x.device)
    cumsum_output = (
        torch.empty(E, dtype=torch.int32, device=x.device) if do_cumsum else
        torch.empty(E, dtype=torch.int32, device=x.device)  # Dummy for API
    )

    # Done counter for "last block wins" pattern - must be zeroed!
    done_counter = torch.zeros(1, dtype=torch.int32, device=x.device)

    # =========================================================================
    # Get Compiled Kernel and Execute
    # =========================================================================
    tensor_api = _get_compiled_kernel(E, N, do_cumsum, x.dtype)

    # Call kernel with tensors directly (TVM-FFI)
    tensor_api(x, count_output, cumsum_output, done_counter)

    # =========================================================================
    # Return Results
    # =========================================================================
    if do_cumsum:
        return count_output, cumsum_output
    else:
        return count_output


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "count_cumsum",
    "CountCumsumKernel",
    "BLOCK_SIZE",
    "CHUNK_SIZE",
]
