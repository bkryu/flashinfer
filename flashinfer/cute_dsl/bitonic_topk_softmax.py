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

Bitonic Top-K + Softmax using CuTe-DSL
======================================

High-performance fused kernel for top-k selection with optional softmax.
Implements optimized bitonic sort-based top-k with warp-level cooperation.

Key Optimizations:
1. "Free max" when sorted=True: max = topk_vals[0], no reduction needed
2. Fast math: exp2(x * log2(e)) with fastmath=True instead of exp(x)
3. Fast reciprocal: rcp_approx(sum) instead of division

Constraints:
- N can be any value 1 <= N <= 4096 (internally padded to power of 2)
- k can be any value 1 <= k <= 128 (internally rounded to power of 2)
- k <= N
"""

from __future__ import annotations

import functools
import math
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, const_expr

from ..api_logging import flashinfer_api

# Import sorting functions from separate module
from .bitonic_sort_utils import bitonic_topk

# =============================================================================
# Torch to CuTe dtype mapping
# =============================================================================

_torch_to_cutlass_dtype = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

# =============================================================================
# Constants
# =============================================================================

MAX_K = 128  # Maximum k we support (after rounding to power of 2)
MAX_N = 4096  # Maximum N we support (power of 2)

# Precompute log2(e) for fast exp: exp(x) = exp2(x * log2(e))
LOG2_E = math.log2(math.e)  # ≈ 1.4427

# =============================================================================
# Helper Functions
# =============================================================================


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


# =============================================================================
# Top-K + Softmax Kernel Class
# =============================================================================


class BitonicTopKSoftmaxKernel:
    """
    Warp-level optimized bitonic top-k kernel with fused softmax.

    This is the fully optimized version with:
    - "Free max" when sorted (topk_vals[0] is the max)
    - Fast exp: exp2(x * log2(e)) with fastmath
    - Fast reciprocal: rcp_approx instead of division
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        k: int,
        k_actual: int = None,
        softmax: bool = False,
        sorted_output: bool = True,
        negate_for_softmax: bool = False,
    ):
        """
        Initialize the top-k + softmax kernel.

        Args:
            dtype: Data type (Float16, BFloat16, Float32)
            N: Row width (must be power of 2)
            k: Number of top elements (must be power of 2, <= 128)
            k_actual: Actual k requested (may be less than k due to rounding)
            softmax: If True, apply softmax to selected values
            sorted_output: If True, return sorted output. When True with softmax,
                          enables "free max" optimization.
            negate_for_softmax: If True, negate values before softmax
        """
        self.dtype = dtype
        self.N = N
        self.k = k
        self.k_actual = k_actual if k_actual is not None else k
        self.softmax = softmax
        self.sorted_output = sorted_output
        self.negate_for_softmax = negate_for_softmax

        # Compute vector size for loads (128-bit = 16 bytes)
        self.vecsize = 128 // dtype.width

        # Validate constraints
        assert _is_power_of_2(N), f"N={N} must be power of 2"
        assert _is_power_of_2(k), f"k={k} must be power of 2"
        assert k <= 128, f"k={k} exceeds maximum 128"
        assert N <= MAX_N, f"N={N} exceeds maximum {MAX_N}"

    def _threads_per_row(self) -> int:
        """Compute optimal number of threads per row."""
        N = self.N
        k = self.k
        num_threads = max(min(N // k, 32, N // 64), 1)
        return num_threads

    def _num_threads(self) -> int:
        """Total threads per block."""
        return 128 if self.N <= 16384 else 256

    def _get_tiled_copy(self):
        """Create TiledCopy for vectorized loads."""
        N = self.N
        vecsize = self.vecsize
        num_threads = self._num_threads()
        threads_per_row = self._threads_per_row()
        cols_per_block = num_threads // threads_per_row
        num_blocks_N = cute.ceil_div(min(N, 16384) // vecsize, threads_per_row)
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)

        # Store compile-time constants in self for kernel access
        self.tiler_mn = tiler_mn
        self.is_even_N = (N == tiler_mn[1])
        self.threads_per_row = threads_per_row

        num_copy_bits = vecsize * self.dtype.width
        copy_op = cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(
            copy_op, self.dtype, num_bits_per_copy=num_copy_bits
        )

        thr_layout = cute.make_ordered_layout(
            (num_threads // threads_per_row, threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, vecsize))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        return tiled_copy

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        stream,
    ):
        """Launch the top-k + softmax kernel."""
        tiled_copy = self._get_tiled_copy()
        num_threads = tiled_copy.size
        tiler_mn = self.tiler_mn  # Access compile-time constant

        self.kernel(
            mX, mValues, mIndices, tiled_copy
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        tiled_copy: cute.TiledCopy,
    ):
        """
        Top-k + Optimized Softmax kernel implementation.

        Algorithm:
        1. Load data with vectorized copy into registers
        2. Convert to Float32 for processing
        3. Encode column indices into low bits of float
        4. Run bitonic top-k with warp shuffle merge
        5. Distribute results across threads
        6. Decode indices from low bits
        7. [OPTIMIZED] Apply softmax if requested:
           a. Negate if needed (for largest=False)
           b. Mask OOB elements with -inf
           c. Get max (free if sorted, else reduction)
           d. Compute exp2(x * log2(e) - max * log2(e)) with fastmath
           e. Sum via warp reduction
           f. Multiply by rcp_approx(sum)
        8. Convert and write output
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        M = shape[0]
        N = self.N
        k = self.k
        threads_per_row = self.threads_per_row  # Compile-time constant

        # Create identity tensor for coordinate tracking
        idX = cute.make_identity_tensor(shape)

        # Get this CTA's tile - use self.tiler_mn for compile-time constant
        tiler_mn = self.tiler_mn
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, 0)) for mT in (mX, idX)]

        # Partition for this thread
        thr_copy = tiled_copy.get_slice(tidx)
        tXgX = thr_copy.partition_S(gX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_rmem_tensor_like(tXgX)

        # Handle boundary
        row = tXcX[0][0]
        is_even_N = self.is_even_N  # Compile-time constant

        # Load data
        if row < M:
            cute.copy(tiled_copy, tXgX, tXrX)

        # Convert to Float32 for processing
        tXrX_f32 = cute.make_rmem_tensor(tXrX.shape, Float32)
        tXrX_f32.store(tXrX.load().to(Float32))

        # =================================================================
        # Encode column indices into low bits of float values
        # =================================================================
        log_N = int(math.log2(N))
        idx_mask = (1 << log_N) - 1
        vecsize = const_expr(cute.size(tv_layout.shape[1]))

        tXrX_i32 = cute.recast_tensor(tXrX_f32, Int32)

        for i in cutlass.range(cute.size(tXrX_i32), unroll_full=True):
            col_idx = Int32(tXcX[i // vecsize][1] + i % vecsize)
            encoded_idx = ~col_idx if tXrX_f32[i] >= Float32(0.0) else col_idx
            encoded_idx = encoded_idx & idx_mask
            tXrX_i32[i] = (tXrX_i32[i] & ~idx_mask) | encoded_idx

        # Fill OOB values with -inf
        if const_expr(not is_even_N):
            for i in cutlass.range(cute.size(tXrX_f32), unroll_full=True):
                col_idx = tXcX[i // vecsize][1] + i % vecsize
                if col_idx >= shape[1]:
                    tXrX_f32[i] = -Float32.inf

        # =================================================================
        # Run bitonic top-k with warp shuffle merge
        # =================================================================
        topk_vals = bitonic_topk(
            tXrX_f32,
            k,
            ascending=False,
            warp_width=threads_per_row,
            sorted_output=self.sorted_output,
        )

        # =================================================================
        # Distribute results across threads for vectorized output
        # =================================================================
        vecsize_out = const_expr(
            min(k, vecsize, 128 // mIndices.element_type.width)
        )
        nvec_per_thread = const_expr(cute.ceil_div(k, vecsize_out * threads_per_row))

        mask = cute.arch.WARP_SIZE - threads_per_row
        mask_and_clamp = mask << 8 | (cute.arch.WARP_SIZE - 1)

        topk_vals_split = cute.make_rmem_tensor((vecsize_out, nvec_per_thread), Float32)

        for i in cutlass.range(cute.ceil_div(k, vecsize_out), unroll_full=True):
            should_receive = tidx % threads_per_row == i % threads_per_row
            for v in cutlass.range(vecsize_out, unroll_full=True):
                if const_expr(threads_per_row > 1):
                    if i * vecsize_out + v < k:
                        val = cute.arch.shuffle_sync(
                            topk_vals[i * vecsize_out + v],
                            offset=0,
                            mask_and_clamp=mask_and_clamp,
                        )
                        if should_receive:
                            topk_vals_split[v, i // threads_per_row] = val
                else:
                    topk_vals_split[v, i // threads_per_row] = topk_vals[
                        i * vecsize_out + v
                    ]

        # =================================================================
        # Decode indices from low bits
        # =================================================================
        topk_vals_i32 = cute.recast_tensor(topk_vals_split, Int32)
        topk_indices = cute.make_rmem_tensor(topk_vals_i32.shape, Int32)

        for i in cutlass.range(cute.size(topk_vals_i32), unroll_full=True):
            encoded_idx = topk_vals_i32[i] & idx_mask
            topk_vals_i32[i] = topk_vals_i32[i] & ~idx_mask
            col_idx = ~encoded_idx if topk_vals_split[i] >= Float32(0.0) else encoded_idx
            topk_indices[i] = Int32(col_idx & idx_mask)

        # =================================================================
        # [OPTIMIZED] Apply softmax if requested
        #
        # This uses all Quack optimizations:
        # 1. "Free max" when sorted - topk_vals[0] is the max
        # 2. Fast exp: exp2(x * log2(e)) with fastmath=True
        # 3. Fast reciprocal: rcp_approx instead of division
        # =================================================================
        if const_expr(self.softmax):
            # ---------------------------------------------------------
            # Step 1: Negate values if needed (for largest=False)
            # ---------------------------------------------------------
            if const_expr(self.negate_for_softmax):
                for i in cutlass.range(cute.size(topk_vals_split), unroll_full=True):
                    topk_vals_split[i] = -topk_vals_split[i]

            # ---------------------------------------------------------
            # Step 2: Mask OOB elements with -inf
            # ---------------------------------------------------------
            k_actual = self.k_actual

            for i in cutlass.range(
                cute.size(topk_vals_split.shape, [1]), unroll_full=True
            ):
                col = i * threads_per_row + tidx % threads_per_row
                for v in cutlass.range(vecsize_out, unroll_full=True):
                    elem_idx = col * vecsize_out + v
                    if col >= k // vecsize_out or elem_idx >= k_actual:
                        topk_vals_split[v, i] = -Float32.inf

            # ---------------------------------------------------------
            # Step 3: Get max value for numerical stability
            #
            # OPTIMIZATION: "Free max" when sorted=True
            #
            # When sorted, topk_vals[0] IS the maximum (largest element).
            # Just broadcast it from thread 0 - no reduction needed!
            #
            # When not sorted (sorted_output=False), fall back to
            # warp_reduction_max.
            # ---------------------------------------------------------
            if const_expr(self.sorted_output):
                # FREE MAX: topk_vals[0] is the max, just broadcast
                max_val = cute.arch.shuffle_sync(
                    topk_vals[0], offset=0, mask_and_clamp=mask_and_clamp
                )
            else:
                # FALLBACK: sorted=False, need full reduction
                local_max = topk_vals_split.load().reduce(
                    cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0
                )
                max_val = cute.arch.warp_reduction_max(
                    local_max,
                    threads_in_group=threads_per_row,
                )

            # ---------------------------------------------------------
            # Step 4: Compute exp2(x * log2(e) - max * log2(e))
            #
            # OPTIMIZATION: Fast exp using exp2
            #
            # exp(x) = exp2(x * log2(e))
            # exp(x - max) = exp2((x - max) * log2(e))
            #              = exp2(x * log2(e) - max * log2(e))
            #
            # With fastmath=True, compiler uses faster approximations.
            # ---------------------------------------------------------
            log2_e = LOG2_E
            max_scaled = max_val * log2_e

            exp_vals = cute.math.exp2(
                topk_vals_split.load() * log2_e - max_scaled, fastmath=True
            )

            # ---------------------------------------------------------
            # Step 5: Sum all exp values (warp reduction)
            # ---------------------------------------------------------
            local_sum = exp_vals.reduce(
                cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
            )

            sum_exp = cute.arch.warp_reduction_sum(
                local_sum,
                threads_in_group=threads_per_row,
            )

            # ---------------------------------------------------------
            # Step 6: Multiply by rcp_approx(sum)
            #
            # OPTIMIZATION: Fast reciprocal
            #
            # Instead of: probs = exp_vals / sum_exp
            # Use:        probs = exp_vals * rcp_approx(sum_exp)
            #
            # rcp_approx uses the GPU's fast reciprocal instruction.
            # Slight precision loss but much faster.
            # ---------------------------------------------------------
            topk_vals_split.store(exp_vals * cute.arch.rcp_approx(sum_exp))

        # =================================================================
        # Convert and write output
        # =================================================================
        topk_vals_out = cute.make_rmem_tensor_like(topk_vals_split, mValues.element_type)
        topk_vals_out.store(topk_vals_split.load().to(mValues.element_type))

        if tiler_mn[0] == 0 or row < M:
            mValues_store = cute.tiled_divide(mValues[row, None], (vecsize_out,))
            mIndices_store = cute.tiled_divide(mIndices[row, None], (vecsize_out,))

            for i in cutlass.range(
                cute.size(topk_vals_out.shape, [1]), unroll_full=True
            ):
                col = i * threads_per_row + tidx % threads_per_row
                if col < k // vecsize_out:
                    cute.autovec_copy(topk_vals_out[None, i], mValues_store[None, col])
                    cute.autovec_copy(topk_indices[None, i], mIndices_store[None, col])


# =============================================================================
# Compilation and Caching
# =============================================================================


def _make_fake_tensor(dtype, shape, divisibility=1):
    """Create a fake tensor for CuTe-DSL compilation."""
    if dtype is None:
        return None
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=(*[cute.sym_int64(divisibility=divisibility)] * (len(shape) - 1), 1),
        assumed_align=divisibility * dtype.width // 8,
    )


# Compilation cache
_compile_cache = {}


def _get_compiled_kernel(
    dtype,
    N: int,
    k_pow2: int,
    k_actual: int = None,
    softmax: bool = False,
    sorted_output: bool = True,
    negate_for_softmax: bool = False,
):
    """Get or compile the top-k + softmax kernel for given parameters."""
    if k_actual is None:
        k_actual = k_pow2
    compile_key = (dtype, N, k_pow2, k_actual, softmax, sorted_output, negate_for_softmax)
    if compile_key in _compile_cache:
        return _compile_cache[compile_key]

    kernel_op = BitonicTopKSoftmaxKernel(
        dtype,
        N,
        k_pow2,
        k_actual=k_actual,
        softmax=softmax,
        sorted_output=sorted_output,
        negate_for_softmax=negate_for_softmax,
    )

    batch_sym = cute.sym_int()
    div_input = math.gcd(128 // dtype.width, N)
    div_output = math.gcd(128 // dtype.width, k_pow2)
    div_indices = math.gcd(128 // Int32.width, k_pow2)

    x_fake = _make_fake_tensor(dtype, (batch_sym, N), div_input)
    values_fake = _make_fake_tensor(dtype, (batch_sym, k_pow2), div_output)
    indices_fake = _make_fake_tensor(Int32, (batch_sym, k_pow2), div_indices)

    compiled = cute.compile(
        kernel_op,
        x_fake,
        values_fake,
        indices_fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        x: torch.Tensor,
        values_out: torch.Tensor,
        indices_out: torch.Tensor,
    ) -> None:
        """Runtime API that passes torch tensors directly via TVM-FFI."""
        compiled(x, values_out, indices_out)

    _compile_cache[compile_key] = tensor_api
    return tensor_api


# =============================================================================
# Main API Function
# =============================================================================


@flashinfer_api
def bitonic_topk_softmax(
    x: torch.Tensor,
    k: int,
    softmax: bool = False,
    largest: bool = True,
    sorted: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bitonic top-k selection with optional fused softmax using CuTe-DSL.

    High-performance GPU kernel for selecting the top-k elements from the
    last dimension of a tensor, with optional softmax normalization over
    the selected elements.

    Key Optimizations:
    - "Free max" when sorted: topk_vals[0] is the max
    - Fast exp: exp2(x * log2(e)) with fastmath
    - Fast reciprocal: rcp_approx instead of division

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape ``[..., N]`` where N is the last dimension.
        N can be any value from 1 to 4096.
        Supported dtypes: ``float32``, ``float16``, ``bfloat16``.
        Must be on CUDA device.
    k : int
        Number of top (or bottom) elements to return.
        Can be any value from 1 to 128.
    softmax : bool, optional
        If ``True``, apply softmax to the selected top-k values. Default is ``False``.
    largest : bool, optional
        If ``True`` (default), return k largest elements.
        If ``False``, return k smallest elements.
    sorted : bool, optional
        Whether to return sorted output. Default is ``True``.
        When ``True`` with softmax, enables "free max" optimization.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple of ``(values, indices)``:

        - ``values``: If softmax=False: Top (or bottom) k values, shape ``[..., k]``.
          If softmax=True: Softmax probabilities over the k values, shape ``[..., k]``.
          Same dtype as input.
        - ``indices``: Indices of selected values in original tensor, shape ``[..., k]``.
          dtype is ``int32`` for performance.

    Examples
    --------
    Basic top-k selection:

    >>> x = torch.randn(4, 256, device='cuda')
    >>> values, indices = bitonic_topk_softmax(x, k=10)
    >>> values.shape
    torch.Size([4, 10])

    Top-k with softmax (for MoE routing):

    >>> x = torch.randn(4, 256, device='cuda')
    >>> probs, indices = bitonic_topk_softmax(x, k=10, softmax=True)
    >>> probs.sum(dim=-1)  # tensor([1., 1., 1., 1.])

    Bottom-k selection:

    >>> values, indices = bitonic_topk_softmax(x, k=10, largest=False)

    Notes
    -----
    - Internally rounds k up to the nearest power of 2 for bitonic sort efficiency.
    - Internally pads N to the nearest power of 2 if not already a power of 2.
    - Results are sliced back to the requested k after computation.
    """
    # =================================================================
    # Input validation
    # =================================================================
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.dtype in [torch.float32, torch.float16, torch.bfloat16], (
        f"Unsupported dtype: {x.dtype}"
    )

    # =================================================================
    # Handle arbitrary tensor shapes (3D+)
    # =================================================================
    original_shape = x.shape
    N_original = original_shape[-1]

    if x.dim() == 1:
        x = x.unsqueeze(0)
        batch_shape = ()
    else:
        batch_shape = original_shape[:-1]
        x = x.reshape(-1, N_original)

    M = x.shape[0]

    # =================================================================
    # Validate and compute k_pow2
    # =================================================================
    assert k > 0, f"k must be positive, got {k}"
    assert k <= N_original, f"k={k} must be <= N={N_original}"
    assert k <= MAX_K, f"k={k} exceeds maximum {MAX_K}"

    k_pow2 = _next_power_of_2(k)
    k_pow2 = min(k_pow2, MAX_K)

    # =================================================================
    # Handle non-power-of-2 N by padding
    # =================================================================
    N_pow2 = _next_power_of_2(N_original)
    needs_padding = not _is_power_of_2(N_original)

    assert N_pow2 <= MAX_N, (
        f"N={N_original} (padded to {N_pow2}) exceeds maximum {MAX_N}"
    )

    # =================================================================
    # Handle largest=False by negating values
    # =================================================================
    device = x.device
    dtype = x.dtype
    x = x.contiguous()

    if not largest:
        x = -x

    # =================================================================
    # Pad input if N is not power of 2
    # =================================================================
    if needs_padding:
        if dtype == torch.float32:
            pad_value = -1e30
        else:
            pad_value = -1e4

        x_padded = torch.full((M, N_pow2), pad_value, dtype=dtype, device=device)
        x_padded[:, :N_original] = x
        x = x_padded

    # =================================================================
    # Allocate output tensors
    # =================================================================
    values_out = torch.empty(M, k_pow2, dtype=dtype, device=device)
    indices_out = torch.empty(M, k_pow2, dtype=torch.int32, device=device)

    # =================================================================
    # Run kernel with OPTIMIZED fused softmax
    # =================================================================
    cutlass_dtype = _torch_to_cutlass_dtype[dtype]
    need_slicing = k < k_pow2
    effective_sorted = sorted or need_slicing
    negate_for_softmax = (not largest) and softmax

    tensor_api = _get_compiled_kernel(
        cutlass_dtype,
        N_pow2,
        k_pow2,
        k_actual=k,
        softmax=softmax,
        sorted_output=effective_sorted,
        negate_for_softmax=negate_for_softmax,
    )
    tensor_api(x, values_out, indices_out)

    # =================================================================
    # Post-process
    # =================================================================
    if k < k_pow2:
        values_out = values_out[:, :k].contiguous()
        indices_out = indices_out[:, :k].contiguous()

    if not largest and not softmax:
        values_out = -values_out

    # =================================================================
    # Reshape output to match input batch dimensions
    # =================================================================
    if len(batch_shape) == 0:
        values_out = values_out.squeeze(0)
        indices_out = indices_out.squeeze(0)
    elif len(batch_shape) > 1:
        output_shape = batch_shape + (k,)
        values_out = values_out.reshape(output_shape)
        indices_out = indices_out.reshape(output_shape)

    return values_out, indices_out


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "bitonic_topk_softmax",
    "BitonicTopKSoftmaxKernel",
    "MAX_K",
    "MAX_N",
]

