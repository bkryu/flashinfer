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

RMSNorm: Root Mean Square Layer Normalization using CuTe-DSL
=============================================================

High-performance RMSNorm implementation using CuTe DSL with cluster-based
reduction for large hidden dimensions. Supports SM90+ (Hopper/Blackwell).

RMSNorm computes: y = x / sqrt(mean(x²) + eps) * weight

Key Features:
-------------
1. CLUSTER SYNCHRONIZATION (SM90+)
   - Multiple CTAs cooperate to process large N dimensions
   - Each CTA handles N/cluster_n elements, then reduces across the cluster
   - Uses mbarrier for efficient cross-CTA synchronization

2. ARCHITECTURE-SPECIFIC TUNING
   - SM80 (Ampere): Single-CTA execution (cluster_n=1)
   - SM90 (Hopper): Cluster support enabled for large N
   - SM100+ (Blackwell): Same as SM90

3. VECTORIZED MEMORY ACCESS
   - 128-bit vectorized loads/stores for optimal memory throughput
   - TiledCopy abstraction for organized gmem↔smem↔rmem transfers

Cluster Size Selection (FP16/BF16):
-----------------------------------
- N <= 16K: cluster_n = 1 (single CTA)
- N <= 32K: cluster_n = 2
- N <= 64K: cluster_n = 4
- N <= 128K: cluster_n = 8
- Larger: cluster_n = 16
"""

import functools
from typing import Callable, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Boolean, Float32, Int32, Int64

from ..api_logging import flashinfer_api
from ..utils import device_support_pdl
from .reduce import row_reduce


# =============================================================================
# Architecture Detection
# =============================================================================


@functools.lru_cache(maxsize=16)
def get_sm_version(device: int | torch.device | str | None = None) -> int:
    """Get the SM version of a CUDA device.

    Args:
        device: CUDA device to query. Can be an int (device index), torch.device,
            device string (e.g., 'cuda:0'), or None to use current device.

    Returns:
        SM version as an integer (e.g., 100 for SM100).
    """
    if not torch.cuda.is_available():
        return 80
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


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
# RMSNorm Kernel Class
# =============================================================================


COPY_BITS = 128  # 128-bit vectorized loads


class RMSNormKernel:
    """
    RMSNorm kernel with cluster synchronization for large N.

    Features:
    - Cluster-based reduction for large N (SM90+)
    - Multiple CTAs cooperate via mbarrier
    - Single reduction (sum of squares) with cluster-level aggregation

    Example:
        >>> kernel = RMSNormKernel(cutlass.Float16, N=4096)
        >>> compiled = cute.compile(kernel, ...)
        >>> compiled(x, w, o, M, eps)
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
        """Calculate shared memory requirement in bytes."""
        tile_bytes = self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
        reduction_bytes = self.rows_per_block * self.warps_per_row * self.cluster_n * 4
        mbar_bytes = 8 if self.cluster_n > 1 else 0
        return tile_bytes + reduction_bytes + mbar_bytes

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mO: cute.Tensor,
        M: Int32,
        eps: Float32,
        stream,
        use_pdl: cutlass.Constexpr[bool] = False,
    ):
        """Host function to launch the RMSNorm kernel.

        Args:
            mX: Input tensor, shape (M, N), row-major
            mW: Weight tensor, shape (N,), or None if no weight
            mO: Output tensor, shape (M, N), row-major
            M: Number of rows (batch dimension)
            eps: Epsilon for numerical stability
            stream: CUDA stream
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

        # Launch with cluster support and optional PDL
        self.kernel(mX, mW, mO, eps, tv_layout, tiler_mn).launch(
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
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mO: cute.Tensor,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        """Device kernel implementing RMSNorm with cluster support and PDL."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: Wait for previous kernel to complete (no-op if PDL disabled at launch)
        # This ensures input data from previous kernel is visible
        if cutlass.const_expr(self.sm_version >= 90):
            cute.arch.griddepcontrol_wait()

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

        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        if cutlass.const_expr(self.cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, self.cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(Int64, num_elems=1)

        # =====================================================================
        # Initialize cluster
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

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gO = cute.local_tile(mO, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

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
            mO.element_type,
            num_bits_per_copy=COPY_BITS,
        )

        tiled_copy_load = cute.make_tiled_copy(copy_atom_load_async, tv_layout, tiler_mn)
        tiled_copy_W = cute.make_tiled_copy(copy_atom_load_W, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # Partition tensors
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_O.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)

        # Register fragments
        tXrX = cute.make_fragment_like(tXgX)
        tXrO = cute.make_fragment_like(tXgO)

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
        # Async copy global → shared
        # =====================================================================
        if row_in_bounds:
            cute.copy(copy_atom_load_async, tXgX, tXsX, pred=tXpX)

        cute.arch.cp_async_commit_group()

        # Load weight while waiting
        if cutlass.const_expr(self.has_weight and mW is not None):
            tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=self.N)
            cute.copy(copy_atom_load_W, tWgW, tWrW, pred=tWpW)

        cute.arch.cp_async_wait_group(0)

        # =====================================================================
        # Pass 1: Compute sum of squares with cluster reduction
        # =====================================================================
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        x_sq = x * x
        sum_sq = row_reduce(
            x_sq,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer,
            mbar_ptr,
            self.cluster_n,
            Float32(0.0),
        )

        # rstd = 1 / sqrt(mean(x²) + eps)
        mean_sq = sum_sq / self.N
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        # Sync after reduction
        if cutlass.const_expr(self.cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # =====================================================================
        # Pass 2: Normalize and output
        # =====================================================================
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        y = x * rstd

        # Apply weight if present
        if cutlass.const_expr(self.has_weight and mW is not None):
            w = tXrW.load().to(Float32)
            y = y * w

        # Store to global memory
        tXrO.store(y.to(self.dtype))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        # PDL: Signal that dependent kernels can start (no-op if PDL disabled at launch)
        # This allows the next kernel to begin loading independent data
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
    Get a compiled RMSNorm kernel closure that takes torch.Tensor directly.

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
    kernel_obj = RMSNormKernel(
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
    o_fake = cute.runtime.make_fake_compact_tensor(
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
        w_fake,
        o_fake,
        Int32(1),  # Dummy M
        Float32(1e-6),  # Dummy eps
        stream_fake,
        use_pdl,  # Compile-time constant
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        x: torch.Tensor,
        w: torch.Tensor | None,
        o: torch.Tensor,
        M: int,
        eps: float,
    ) -> None:
        """Runtime API that passes torch tensors directly via TVM-FFI."""
        nonlocal compiled_kernel
        compiled_kernel(x, w, o, Int32(M), Float32(eps))

    return tensor_api


# =============================================================================
# Public API
# =============================================================================


@flashinfer_api
def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    eps: float = 1e-6,
    use_pdl: bool | None = None,
) -> torch.Tensor:
    """
    RMS Normalization using CuTe-DSL.

    Computes: ``y = x / sqrt(mean(x²) + eps) * weight`` (if weight is provided)

    This implementation uses cluster-based reduction for large hidden dimensions
    on SM90+ architectures (Hopper/Blackwell), automatically falling back to
    single-CTA execution on older architectures.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, shape ``(M, N)`` or ``(batch, seq_len, hidden_size)``.
        Must be ``torch.float16``, ``torch.bfloat16``, or ``torch.float32``.
    weight : torch.Tensor, optional
        Weight tensor for scaling, shape ``(N,)`` or ``(hidden_size,)``.
        Must have the same dtype as input. If ``None``, no weight is applied.
    output : torch.Tensor, optional
        Output tensor. If ``None``, will be allocated automatically.
        Must have the same shape and dtype as input.
    eps : float
        Epsilon for numerical stability. Default is ``1e-6``.
    use_pdl : bool, optional
        Whether to enable `Programmatic Dependent Launch (PDL)
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_.
        PDL allows overlapping execution of consecutive kernels in the same stream.
        If ``None``, PDL is automatically enabled on supported hardware (SM90+).
        Default is ``None``.

    Returns
    -------
    torch.Tensor
        Normalized output tensor, same shape and dtype as input.

    Notes
    -----
    - This kernel requires CuTe-DSL to be available.
    - For best performance on large hidden dimensions (N > 16K), use on
      SM90+ (Hopper/Blackwell) to benefit from cluster-based reduction.
    - The kernel uses 128-bit vectorized loads for optimal memory throughput.
    - PDL enables overlapping of consecutive kernels in the same stream,
      which can improve performance in kernel-launch-bound scenarios.

    Examples
    --------
    >>> import torch
    >>> from flashinfer.cute_dsl import rmsnorm
    >>> x = torch.randn(128, 4096, device="cuda", dtype=torch.float16)
    >>> w = torch.randn(4096, device="cuda", dtype=torch.float16)
    >>> y = rmsnorm(x, w)

    >>> # Without weight
    >>> y = rmsnorm(x)

    >>> # With 3D input (batch, seq_len, hidden)
    >>> x = torch.randn(2, 64, 4096, device="cuda", dtype=torch.bfloat16)
    >>> w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    >>> y = rmsnorm(x, w)

    >>> # Explicitly enable PDL
    >>> y = rmsnorm(x, w, use_pdl=True)
    """
    # Handle 2D vs 3D input
    is_3d = input.dim() == 3
    if is_3d:
        B, S, N = input.shape
        input_2d = input.view(B * S, N).contiguous()
        M = B * S
    else:
        M, N = input.shape
        input_2d = input.contiguous()

    # Validate dtype
    assert input.dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"Input must be float16, bfloat16, or float32, got {input.dtype}"
    )

    has_weight = weight is not None
    if has_weight:
        assert weight.dtype == input.dtype, (
            f"Weight dtype {weight.dtype} must match input dtype {input.dtype}"
        )
        assert weight.shape == (N,), (
            f"Weight shape {weight.shape} must be ({N},)"
        )

    # Allocate output if needed
    if output is None:
        output = torch.empty_like(input)
    else:
        assert output.shape == input.shape, (
            f"Output shape {output.shape} must match input shape {input.shape}"
        )
        assert output.dtype == input.dtype, (
            f"Output dtype {output.dtype} must match input dtype {input.dtype}"
        )

    output_2d = output.view(M, N) if is_3d else output

    # Get dtype string for cache key
    dtype_str = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[input.dtype]
    sm_version = get_sm_version(input.device)

    # Determine PDL setting: auto-enable on SM90+ if not specified
    if use_pdl is None:
        use_pdl = device_support_pdl(input.device)

    # Get compiled kernel and run
    # Note: use_pdl is part of cache key since CuTe DSL requires it at compile time
    tensor_api = _get_compiled_kernel(N, has_weight, sm_version, dtype_str, use_pdl)
    tensor_api(input_2d, weight, output_2d, M, eps)

    return output


__all__ = [
    "rmsnorm",
    "RMSNormKernel",
    "get_sm_version",
]

