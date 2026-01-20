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

Fused Add + RMSNorm: Element-wise Addition followed by RMS Normalization
========================================================================

High-performance fused kernel for element-wise addition followed by RMS
normalization using CuTe DSL. This fusion saves memory bandwidth by:
1. Loading input and residual tensors once
2. Computing the sum in registers
3. Writing back both the updated residual and normalized output

Operation:
    1. residual = residual + input
    2. output = (residual / sqrt(mean(residual²) + eps)) * weight

Both input (receives normalized output) and residual are modified in-place.

Iteration 1: Basic single-CTA kernel for correctness.
Iteration 2: Optimized memory access - eliminated redundant shared memory
             reload by keeping fused values in registers across reduction.
Iteration 3: Cluster support (SM90+) - multiple CTAs cooperate on large N
             using distributed shared memory and mbarrier synchronization.
Iteration 4: Edge case handling - improved input validation, clearer error
             messages, auto-PDL on SM90+, non-power-of-2 N support.
"""

import functools
from typing import Callable

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Boolean, Float32, Int32, Int64

from ..api_logging import flashinfer_api
from ..utils import device_support_pdl
from .reduce import row_reduce
from .rmsnorm import get_sm_version


# =============================================================================
# Constants
# =============================================================================

# Minimum hidden size supported (must be >= vec_size for smallest dtype)
# FP32: 128 bits / 32 bits = 4 elements minimum
# FP16/BF16: 128 bits / 16 bits = 8 elements minimum
MIN_HIDDEN_SIZE = 8

# Vector sizes for 128-bit loads (for alignment validation)
VEC_SIZE_FP16 = 8  # 128 bits / 16 bits
VEC_SIZE_FP32 = 4  # 128 bits / 32 bits


# =============================================================================
# Predicate Utility
# =============================================================================


@cute.jit
def predicate_k(tXcX: cute.Tensor, limit: int) -> cute.Tensor:
    """Create predicate tensor for bounds checking."""
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            tXpX[rest_v, 0, rest_k] = cute.elem_less(
                tXcX[(0, rest_v), 0, rest_k][1], limit
            )
    return tXpX


# =============================================================================
# Fused Add + RMSNorm Kernel Class
# =============================================================================


COPY_BITS = 128  # 128-bit vectorized loads


class AddRMSNormKernel:
    """
    Fused Add + RMSNorm kernel with cluster support (SM90+).

    Computes:
        1. residual = residual + input
        2. output = (residual / sqrt(mean(residual²) + eps)) * weight

    Features:
    - Keeps fused values in registers across reduction (no redundant reload)
    - Cluster-based reduction for large N (SM90+)
    - Multiple CTAs cooperate via mbarrier for distributed reduction
    - 128-bit vectorized loads for optimal memory throughput
    - PDL (Programmatic Dependent Launch) support on SM90+

    Supported configurations:
    - Hidden size N >= 8 (MIN_HIDDEN_SIZE)
    - Any M (batch size)
    - Dtypes: float16, bfloat16, float32

    Example:
        >>> kernel = AddRMSNormKernel(cutlass.Float16, N=4096)
        >>> compiled = cute.compile(kernel, ...)
        >>> compiled(input, residual, weight, M, eps)
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        N: int,
        has_weight: bool = True,
        sm_version: int | None = None,
    ):
        self.dtype = dtype
        self.N = N
        self.has_weight = has_weight
        self.sm_version = sm_version if sm_version is not None else get_sm_version()

        # Vector size for 128-bit loads
        self.vec_size = COPY_BITS // dtype.width

        # Compute cluster size (SM90+ only)
        self.cluster_n = self._compute_cluster_n(N, dtype, self.sm_version)

        # N per CTA for cluster case
        self.N_per_cta = N // self.cluster_n

        # Thread configuration
        self.threads_per_row = self._compute_threads_per_row(self.N_per_cta)
        self.num_threads = self._compute_num_threads(self.N_per_cta)

        # Derived values
        self.num_vec_blocks = max(
            1,
            (self.N_per_cta // self.vec_size + self.threads_per_row - 1)
            // self.threads_per_row,
        )
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

    @staticmethod
    def _compute_cluster_n(N: int, dtype: cutlass.Numeric, sm_version: int) -> int:
        """Compute optimal cluster size based on N and architecture."""
        if sm_version < 90:
            return 1

        if dtype.width == 16:  # FP16/BF16
            if N <= 16 * 1024:
                return 1
            elif N <= 32 * 1024:
                return 2
            elif N <= 64 * 1024:
                return 4
            elif N <= 128 * 1024:
                return 8
            else:
                return 16
        else:  # FP32
            if N <= 32 * 1024:
                return 1
            elif N <= 64 * 1024:
                return 2
            elif N <= 128 * 1024:
                return 4
            elif N <= 256 * 1024:
                return 8
            else:
                return 16

    @staticmethod
    def _compute_threads_per_row(N_per_cta: int) -> int:
        """Compute optimal threads per row based on N per CTA."""
        if N_per_cta <= 64:
            return 8
        elif N_per_cta <= 128:
            return 16
        elif N_per_cta <= 3072:
            return 32
        elif N_per_cta <= 6144:
            return 64
        elif N_per_cta <= 16384:
            return 128
        else:
            return 256

    @staticmethod
    def _compute_num_threads(N_per_cta: int) -> int:
        """Compute total threads per block."""
        return 128 if N_per_cta <= 16384 else 256

    @staticmethod
    def _make_tv_layout(
        threads_per_row: int,
        rows_per_block: int,
        vec_size: int,
        num_vec_blocks: int,
    ) -> tuple:
        """Create Thread-Value layout for coalesced vectorized memory access."""
        shape = (
            (threads_per_row, rows_per_block),
            (vec_size, num_vec_blocks),
        )
        stride = (
            (vec_size * rows_per_block, 1),
            (rows_per_block, rows_per_block * vec_size * threads_per_row),
        )
        return shape, stride

    def _smem_size_in_bytes(self) -> int:
        """Calculate shared memory requirement in bytes.

        For fused add+rmsnorm, we need shared memory for:
        - Input tile (rows_per_block x cols_per_tile)
        - Residual tile (rows_per_block x cols_per_tile)
        - Reduction buffer (rows_per_block x warps_per_row x cluster_n)
        - Mbarrier (8 bytes, only for cluster_n > 1)
        """
        # Two input tiles (input + residual)
        tile_bytes = self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
        input_tile_bytes = tile_bytes
        residual_tile_bytes = tile_bytes

        # Reduction buffer (includes cluster dimension)
        reduction_bytes = (
            self.rows_per_block * self.warps_per_row * self.cluster_n * 4
        )  # Float32

        # Mbarrier for cluster synchronization
        mbar_bytes = 8 if self.cluster_n > 1 else 0

        return input_tile_bytes + residual_tile_bytes + reduction_bytes + mbar_bytes

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # Input tensor, will receive normalized output
        mR: cute.Tensor,  # Residual tensor, will be updated in-place
        mW: cute.Tensor | None,  # Weight tensor
        M: Int32,
        eps: Float32,
        stream,
        use_pdl: cutlass.Constexpr[bool] = False,
    ):
        """Host function to launch the fused Add+RMSNorm kernel.

        Args:
            mX: Input tensor, shape (M, N), row-major. Will be overwritten with
                normalized output.
            mR: Residual tensor, shape (M, N), row-major. Will be updated to
                residual + input.
            mW: Weight tensor, shape (N,), or None if no weight.
            M: Number of rows (batch dimension).
            eps: Epsilon for numerical stability.
            stream: CUDA stream.
            use_pdl: Whether to enable Programmatic Dependent Launch (SM90+).
                Must be a compile-time constant (Constexpr).
        """
        # Create TV layout
        tv_shape, tv_stride = self._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        # Launch with cluster support
        self.kernel(mX, mR, mW, eps, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), self.cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1]
            if cutlass.const_expr(self.cluster_n > 1)
            else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,  # Input tensor
        mR: cute.Tensor,  # Residual tensor
        mW: cute.Tensor | None,  # Weight tensor
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        """Device kernel implementing fused Add + RMSNorm with cluster support."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: Wait for previous kernel to complete (no-op if PDL disabled at launch)
        if cutlass.const_expr(self.sm_version >= 90):
            cute.arch.griddepcontrol_wait()

        # Get cluster coordinate (y-dimension of grid)
        if cutlass.const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        M = mX.shape[0]
        threads_per_row = tv_layout.shape[0][0]
        warps_per_row = max(threads_per_row // 32, 1)
        rows_per_block = tiler_mn[0]

        # =====================================================================
        # Allocate shared memory
        # =====================================================================
        smem = utils.SmemAllocator()

        # Shared memory for input tile
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        # Shared memory for residual tile
        sR = smem.allocate_tensor(
            mR.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        # Reduction buffer and mbarrier allocation depend on cluster size
        if cutlass.const_expr(self.cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            # Hierarchical buffer: (rows_per_block, (warps_per_row, cluster_n))
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, self.cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(Int64, num_elems=1)

        # =====================================================================
        # Initialize cluster (SM90+ with cluster_n > 1)
        # =====================================================================
        if cutlass.const_expr(self.cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # =====================================================================
        # Create identity tensor and partition
        # =====================================================================
        idX = cute.make_identity_tensor(mX.shape)

        # Global memory tiles (partitioned by cluster_y for cluster case)
        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))  # Input tile
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))  # Residual tile
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))  # Coordinate tile

        # Weight tensor (expand to 2D for tiling)
        if cutlass.const_expr(self.has_weight and mW is not None):
            mW_expanded_layout = cute.prepend(
                mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
            gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # =====================================================================
        # Create TiledCopy operations
        # =====================================================================
        copy_atom_load_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=COPY_BITS,
        )

        copy_atom_load_W = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=COPY_BITS,
        )

        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=COPY_BITS,
        )

        # Tiled copies
        tiled_copy_load = cute.make_tiled_copy(copy_atom_load_async, tv_layout, tiler_mn)
        tiled_copy_W = cute.make_tiled_copy(copy_atom_load_W, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        # Thread slices
        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # Partition input tensor
        tXgX = thr_copy_X.partition_S(gX)  # Global input
        tXsX = thr_copy_X.partition_D(sX)  # Shared input
        tXcX = thr_copy_X.partition_S(cX)  # Coordinates

        # Partition residual tensor (use same thread copy pattern)
        tRgR = thr_copy_X.partition_S(gR)  # Global residual
        tRsR = thr_copy_X.partition_D(sR)  # Shared residual

        # Partition output tensors
        tXgO = thr_copy_O.partition_D(gX)  # Output goes back to input tensor
        tRgO = thr_copy_O.partition_D(gR)  # Updated residual

        # Register fragments
        tXrX = cute.make_fragment_like(tXgX)  # Input registers
        tRrR = cute.make_fragment_like(tRgR)  # Residual registers
        tXrO = cute.make_fragment_like(tXgO)  # Output registers (normalized)
        tRrO = cute.make_fragment_like(tRgO)  # Output registers (fused residual)

        # Weight registers
        if cutlass.const_expr(self.has_weight and mW is not None):
            tWgW = thr_copy_W.partition_S(gW)
            tWrW = cute.make_fragment_like(tWgW)
            tXrW = thr_copy_X.retile(tWrW)

        # =====================================================================
        # Bounds checking
        # =====================================================================
        tXpX = predicate_k(tXcX, limit=self.N)

        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # =====================================================================
        # Async copy global → shared (both input and residual)
        # =====================================================================
        if row_in_bounds:
            cute.copy(copy_atom_load_async, tXgX, tXsX, pred=tXpX)  # Input
            cute.copy(copy_atom_load_async, tRgR, tRsR, pred=tXpX)  # Residual

        cute.arch.cp_async_commit_group()

        # Load weight while waiting
        if cutlass.const_expr(self.has_weight and mW is not None):
            tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=self.N)
            cute.copy(copy_atom_load_W, tWgW, tWrW, pred=tWpW)

        cute.arch.cp_async_wait_group(0)

        # =====================================================================
        # Load from shared memory to registers
        # =====================================================================
        cute.autovec_copy(tXsX, tXrX)  # Input
        cute.autovec_copy(tRsR, tRrR)  # Residual

        x = tXrX.load().to(Float32)  # Input in FP32
        r = tRrR.load().to(Float32)  # Residual in FP32

        # Compute fused = input + residual (kept in registers across reduction)
        fused = x + r

        # =====================================================================
        # Reduction: Compute sum of squares with cluster support
        # =====================================================================
        fused_sq = fused * fused
        sum_sq = row_reduce(
            fused_sq,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer,
            mbar_ptr,
            self.cluster_n,
            Float32(0.0),
        )

        # rstd = 1 / sqrt(mean(fused²) + eps)
        mean_sq = sum_sq / self.N
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        # Sync after reduction
        if cutlass.const_expr(self.cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # =====================================================================
        # Normalize and write outputs (reuse fused from registers - no reload!)
        # =====================================================================
        # Compute normalized output: y = fused * rstd * weight
        y = fused * rstd

        # Apply weight if present
        if cutlass.const_expr(self.has_weight and mW is not None):
            w = tXrW.load().to(Float32)
            y = y * w

        # Store outputs to register fragments
        tRrO.store(fused.to(self.dtype))  # Store fused to residual tensor
        tXrO.store(y.to(self.dtype))  # Store normalized to input tensor

        # Write from registers to global memory
        if row_in_bounds:
            cute.copy(copy_atom_store, tRrO, tRgO, pred=tXpX)  # Write residual
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)  # Write output

        # PDL: Signal that dependent kernels can start (no-op if PDL disabled at launch)
        if cutlass.const_expr(self.sm_version >= 90):
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# Kernel Compilation and Caching
# =============================================================================


@functools.cache
def _get_compiled_kernel(
    hidden_size: int,
    has_weight: bool,
    sm_version: int,
    dtype_str: str,  # "float16" or "bfloat16" - hashable key
    use_pdl: bool = False,  # PDL must be a compile-time constant
) -> Callable:
    """
    Get a compiled fused Add+RMSNorm kernel closure.

    Uses TVM-FFI for efficient tensor passing without manual pointer construction.

    Note: use_pdl is a compile-time constant in CuTe DSL, so separate kernel
    binaries are compiled for PDL on/off. The cache handles this automatically.
    """
    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float32": cutlass.Float32,
    }
    dtype = dtype_map[dtype_str]

    # Create kernel instance
    kernel_obj = AddRMSNormKernel(
        dtype=dtype,
        N=hidden_size,
        has_weight=has_weight,
        sm_version=sm_version,
    )

    # Use symbolic size for dynamic M dimension
    sym_m = cute.sym_int()

    # Create fake tensors for compilation with TVM-FFI
    # Use stride_order=(1, 0) for row-major layout
    x_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, hidden_size), stride_order=(1, 0), assumed_align=128
    )
    r_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, hidden_size), stride_order=(1, 0), assumed_align=128
    )

    if has_weight:
        w_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (hidden_size,), assumed_align=128
        )
    else:
        w_fake = None

    # Create fake stream that uses environment stream at runtime
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Compile with TVM-FFI enabled
    # use_pdl must be a Python bool (compile-time constant for CuTe DSL)
    compiled_kernel = cute.compile(
        kernel_obj,
        x_fake,
        r_fake,
        w_fake,
        Int32(1),  # Dummy M
        Float32(1e-6),  # Dummy eps
        stream_fake,
        use_pdl,  # Compile-time constant
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        x: torch.Tensor,
        r: torch.Tensor,
        w: torch.Tensor | None,
        M: int,
        eps: float,
    ) -> None:
        """Runtime API that passes torch tensors directly via TVM-FFI."""
        nonlocal compiled_kernel
        compiled_kernel(x, r, w, Int32(M), Float32(eps))

    return tensor_api


# =============================================================================
# Public API
# =============================================================================


@flashinfer_api
def add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
    enable_pdl: bool | None = None,
) -> None:
    """
    Fused Add + RMSNorm using CuTe-DSL.

    Computes:
        1. ``residual = residual + input``
        2. ``input = (residual / sqrt(mean(residual²) + eps)) * weight``

    Both tensors are modified in-place.

    This implementation fuses element-wise addition with RMS normalization to
    reduce memory bandwidth by loading both tensors once and writing both
    outputs in a single kernel.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, shape ``(M, N)`` or ``(batch, seq_len, hidden_size)``.
        Must be ``torch.float16``, ``torch.bfloat16``, or ``torch.float32``.
        Must be contiguous. Will be overwritten with the normalized output.
    residual : torch.Tensor
        Residual tensor, same shape and dtype as input. Must be contiguous.
        Will be updated in-place to ``residual + input``.
    weight : torch.Tensor, optional
        Weight tensor for scaling, shape ``(N,)`` or ``(hidden_size,)``.
        Must have the same dtype as input and be contiguous.
        If ``None``, no weight is applied.
    eps : float
        Epsilon for numerical stability. Default is ``1e-6``.
    enable_pdl : bool, optional
        Enable `Programmatic Dependent Launch (PDL)
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_.
        PDL allows overlapping execution of consecutive kernels in the same stream.
        If ``None``, PDL is automatically enabled on supported hardware (SM90+).
        Default is ``None``.

    Returns
    -------
    None
        Both ``input`` and ``residual`` are modified in-place.

    Raises
    ------
    TypeError
        If dtype is not float16, bfloat16, or float32, or if dtypes don't match.
    ValueError
        If hidden dimension N is too small (< 8 elements), not properly aligned
        (must be divisible by 8 for FP16/BF16, 4 for FP32), shapes don't match,
        tensors are not contiguous, or devices don't match.

    Notes
    -----
    - This kernel requires CuTe-DSL to be available.
    - Supports cluster-based reduction for large hidden dimensions (N > 16K) on SM90+.
    - Fused values are kept in registers across reduction (no redundant reload).
    - The kernel uses 128-bit vectorized loads for optimal memory throughput.
    - PDL enables overlapping of consecutive kernels in the same stream,
      which can improve performance in kernel-launch-bound scenarios.

    Examples
    --------
    >>> import torch
    >>> from flashinfer.cute_dsl import add_rmsnorm
    >>> x = torch.randn(128, 4096, device="cuda", dtype=torch.float16)
    >>> residual = torch.randn(128, 4096, device="cuda", dtype=torch.float16)
    >>> w = torch.randn(4096, device="cuda", dtype=torch.float16)
    >>> add_rmsnorm(x, residual, w)
    >>> # Now: residual contains original_residual + original_input
    >>> #      x contains RMSNorm(residual) * w

    >>> # Without weight
    >>> add_rmsnorm(x, residual)

    >>> # With PDL enabled (SM90+ only)
    >>> add_rmsnorm(x, residual, w, enable_pdl=True)
    """
    # =========================================================================
    # Input validation
    # =========================================================================

    # Validate tensor dimensions
    if input.dim() not in (2, 3):
        raise ValueError(
            f"Input must be 2D (M, N) or 3D (batch, seq_len, hidden), got {input.dim()}D"
        )

    # Handle 2D vs 3D input
    is_3d = input.dim() == 3
    if is_3d:
        B, S, N = input.shape
        M = B * S
    else:
        M, N = input.shape

    # Validate minimum hidden size
    if N < MIN_HIDDEN_SIZE:
        raise ValueError(
            f"Hidden dimension N={N} is too small. Minimum supported is {MIN_HIDDEN_SIZE}."
        )

    # Validate alignment for 128-bit vectorized loads
    # FP16/BF16 require N divisible by 8, FP32 requires N divisible by 4
    vec_size = VEC_SIZE_FP16 if input.dtype in (torch.float16, torch.bfloat16) else VEC_SIZE_FP32
    if N % vec_size != 0:
        raise ValueError(
            f"Hidden dimension N={N} must be divisible by {vec_size} for {input.dtype} "
            f"(required for 128-bit vectorized memory access). "
            f"Consider padding to N={((N + vec_size - 1) // vec_size) * vec_size}."
        )

    # Validate dtype
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"Input must be float16, bfloat16, or float32, got {input.dtype}"
        )

    # Validate shapes match
    if input.shape != residual.shape:
        raise ValueError(
            f"Input shape {input.shape} must match residual shape {residual.shape}"
        )

    # Validate dtypes match
    if input.dtype != residual.dtype:
        raise TypeError(
            f"Input dtype {input.dtype} must match residual dtype {residual.dtype}"
        )

    # Validate devices match
    if input.device != residual.device:
        raise ValueError(
            f"Input device {input.device} must match residual device {residual.device}"
        )

    # Validate contiguity
    if not input.is_contiguous():
        raise ValueError(
            "Input tensor must be contiguous. Use input.contiguous() to fix."
        )
    if not residual.is_contiguous():
        raise ValueError(
            "Residual tensor must be contiguous. Use residual.contiguous() to fix."
        )

    # Get 2D views for kernel
    if is_3d:
        input_2d = input.view(M, N)
        residual_2d = residual.view(M, N)
    else:
        input_2d = input
        residual_2d = residual

    # Validate weight if provided
    has_weight = weight is not None
    if has_weight:
        if weight.dtype != input.dtype:
            raise TypeError(
                f"Weight dtype {weight.dtype} must match input dtype {input.dtype}"
            )
        if weight.shape != (N,):
            raise ValueError(
                f"Weight shape {weight.shape} must be ({N},)"
            )
        if weight.device != input.device:
            raise ValueError(
                f"Weight device {weight.device} must match input device {input.device}"
            )
        if not weight.is_contiguous():
            raise ValueError(
                "Weight tensor must be contiguous. Use weight.contiguous() to fix."
            )

    # =========================================================================
    # Kernel dispatch
    # =========================================================================

    # Get dtype string for cache key
    dtype_str = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[input.dtype]
    sm_version = get_sm_version(input.device)

    # Determine PDL setting: auto-enable on SM90+ if not specified
    if enable_pdl is None:
        use_pdl = device_support_pdl(input.device)
    else:
        use_pdl = bool(enable_pdl) and sm_version >= 90

    # Get compiled kernel and run
    tensor_api = _get_compiled_kernel(N, has_weight, sm_version, dtype_str, use_pdl)
    tensor_api(input_2d, residual_2d, weight, M, eps)


__all__ = [
    "add_rmsnorm",
    "AddRMSNormKernel",
]

