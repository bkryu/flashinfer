from ..api_logging import flashinfer_api
from flashinfer.jit import (
    gen_dsv3_router_gemm_module,
    gen_tinygemm2_module,
    gen_tinygemm_fp8_module,
)
import functools
from types import SimpleNamespace
from typing import Optional
import torch
from flashinfer.utils import (
    register_custom_op,
    supported_compute_capability,
    backend_requirement,
)


def _mm_M1_16_K7168_shape_checks(
    mat_a, mat_b, out, launch_with_pdl, expected_num_experts, expected_out_dtype
):
    # Dimension checks
    if mat_a.dim() != 2:
        raise ValueError("mat_a must be a 2D tensor")
    if mat_b.dim() != 2:
        raise ValueError("mat_b must be a 2D tensor")
    if out.dim() != 2:
        raise ValueError("out must be a 2D tensor")

    # Stride checks (check these before dimension checks to give better error messages)
    if mat_a.stride(1) != 1:
        raise ValueError("mat_a must be row-major")
    if out.stride(1) != 1:
        raise ValueError("out must be row-major")
    if mat_b.stride(0) != 1:
        raise ValueError("mat_b must be column-major")

    if mat_a.shape[1] != mat_b.shape[0]:
        raise ValueError("mat_a.shape[1] must be equal to mat_b.shape[0]")
    if out.shape[0] != mat_a.shape[0]:
        raise ValueError("out.shape[0] must be equal to mat_a.shape[0]")
    if out.shape[1] != mat_b.shape[1]:
        raise ValueError("out.shape[1] must be equal to mat_b.shape[1]")

    # Problem size checks
    expected_hidden_dim = 7168
    min_tokens = 1
    max_tokens = 16
    if mat_a.shape[0] < min_tokens or mat_a.shape[0] > max_tokens:
        raise ValueError(
            f"mat_a.shape[0] (num_tokens) must be between {min_tokens} and {max_tokens}"
        )
    if mat_a.shape[1] != expected_hidden_dim:
        raise ValueError(
            f"mat_a.shape[1] (hidden_dim) must be equal to {expected_hidden_dim}"
        )
    if mat_b.shape[1] != expected_num_experts:
        raise ValueError(
            f"mat_b.shape[1] (num_experts) must be equal to {expected_num_experts}"
        )

    # Data type checks
    if mat_a.dtype != torch.bfloat16:
        raise ValueError("mat_a must be a bfloat16 tensor")
    if mat_b.dtype != torch.bfloat16:
        raise ValueError("mat_b must be a bfloat16 tensor")
    if out.dtype != expected_out_dtype:
        raise ValueError(f"out must be a {expected_out_dtype} tensor")

    return True


# TODO: other compute capabilities may be supported but are untested
@supported_compute_capability([100, 103])
def _mm_M1_16_K7168_N256_shape_checks(mat_a, mat_b, out, launch_with_pdl):
    return _mm_M1_16_K7168_shape_checks(
        mat_a,
        mat_b,
        out,
        launch_with_pdl,
        expected_num_experts=256,
        expected_out_dtype=torch.float32,
    )


# TODO: other compute capabilities may be supported but are untested
@supported_compute_capability([100, 103])
def _mm_M1_16_K7168_N128_shape_checks(mat_a, mat_b, out, launch_with_pdl):
    return _mm_M1_16_K7168_shape_checks(
        mat_a,
        mat_b,
        out,
        launch_with_pdl,
        expected_num_experts=128,
        expected_out_dtype=torch.bfloat16,
    )


@functools.cache
def get_dsv3_router_gemm_module():
    module = gen_dsv3_router_gemm_module().build_and_load()

    @register_custom_op(
        "flashinfer::ml3_router_gemm_op",
        mutates_args=["out"],
    )
    def mm_M1_16_K7168_N128(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = False,
    ) -> None:
        module.ml3_router_gemm_op(mat_a, mat_b, out, launch_with_pdl)

    @register_custom_op(
        "flashinfer::dsv3_router_gemm_op",
        mutates_args=["out"],
    )
    def mm_M1_16_K7168_N256(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = False,
    ) -> None:
        module.dsv3_router_gemm_op(mat_a, mat_b, out, launch_with_pdl)

    return SimpleNamespace(
        mm_M1_16_K7168_N128=mm_M1_16_K7168_N128,
        mm_M1_16_K7168_N256=mm_M1_16_K7168_N256,
    )


@backend_requirement({}, common_check=_mm_M1_16_K7168_N128_shape_checks)
@flashinfer_api
def mm_M1_16_K7168_N128(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = False,
) -> None:
    """Optimized GEMM for the router operation in Mistral Large 3.

    This function performs a highly optimized matrix multiplication specifically tailored
    for the expert routing GEMM in Mistral Large 3's Mixture of Experts (MoE) architecture.
    It computes out = mat_a @ mat_b where mat_a contains token embeddings and mat_b
    contains expert routing weights.

    The implementation is optimized for the specific problem dimensions used in Mistral Large 3:
    - Hidden dimension (K): 7168
    - Number of experts (N): 128
    - Number of tokens (M): 1-16

    Args:
        mat_a (torch.Tensor): Input token embeddings of shape (M, K) where M is the number
            of tokens (1-16) and K is the hidden dimension (7168). Must be bfloat16,
            row-major (contiguous).
        mat_b (torch.Tensor): Expert routing weights of shape (K, N) where K is the hidden
            dimension (7168) and N is the number of experts (128). Must be bfloat16,
            column-major (transposed layout).
        out (torch.Tensor): Pre-allocated output tensor of shape (M, N) containing the
            routing scores. Must be bfloat16, row-major (contiguous). This tensor is
            mutated in-place.
        launch_with_pdl (bool, optional): Whether to launch the kernel using Persistent
            Device-side Launch. Defaults to False.

    Returns:
        None: The result is written directly to the `out` tensor.

    Raises:
        ValueError: If tensor dimensions, strides, or data types do not match the
            expected Mistral Large 3 router configuration.

    Note:
        This kernel is specialized for compute capability 10.0 (Blackwell architecture).
        The specific problem size optimization makes this significantly faster than
        general-purpose GEMM implementations for the router operation.
    """
    get_dsv3_router_gemm_module().mm_M1_16_K7168_N128(
        mat_a, mat_b, out, launch_with_pdl
    )


@backend_requirement({}, common_check=_mm_M1_16_K7168_N256_shape_checks)
@flashinfer_api
def mm_M1_16_K7168_N256(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = False,
) -> None:
    """Optimized GEMM for the router operation in DeepSeek-V3.

    This function performs a highly optimized matrix multiplication specifically tailored
    for the expert routing GEMM in DeepSeek-V3's Mixture of Experts (MoE) architecture.
    It computes out = mat_a @ mat_b where mat_a contains token embeddings and mat_b
    contains expert routing weights.

    The implementation is optimized for the specific problem dimensions used in DeepSeek-V3:
    - Hidden dimension (K): 7168
    - Number of experts (N): 256
    - Number of tokens (M): 1-16

    Args:
        mat_a (torch.Tensor): Input token embeddings of shape (M, K) where M is the number
            of tokens (1-16) and K is the hidden dimension (7168). Must be bfloat16,
            row-major (contiguous).
        mat_b (torch.Tensor): Expert routing weights of shape (K, N) where K is the hidden
            dimension (7168) and N is the number of experts (256). Must be bfloat16,
            column-major (transposed layout).
        out (torch.Tensor): Pre-allocated output tensor of shape (M, N) containing the
            routing scores. Must be float32, row-major (contiguous). This tensor is
            mutated in-place.
        launch_with_pdl (bool, optional): Whether to launch the kernel using Persistent
            Device-side Launch. Defaults to False.

    Returns:
        None: The result is written directly to the `out` tensor.

    Raises:
        ValueError: If tensor dimensions, strides, or data types do not match the
            expected DeepSeek-V3 router configuration.

    Note:
        This kernel is specialized for compute capability 10.0 (Blackwell architecture).
        The specific problem size optimization makes this significantly faster than
        general-purpose GEMM implementations for the router operation.
    """
    get_dsv3_router_gemm_module().mm_M1_16_K7168_N256(
        mat_a, mat_b, out, launch_with_pdl
    )


# ============================================================================
# tinygemm2: SM90+ BF16 small GEMM with bias (from TensorRT-LLM)
# Computes: output = input @ weight.T + bias  (equivalent to F.linear)
# ============================================================================


@supported_compute_capability([90, 100, 103, 110, 120, 121])
def _tinygemm_bf16_shape_checks(input, weight, out, bias, use_pdl):
    if input.dim() != 2:
        raise ValueError("input must be a 2D tensor")
    if weight.dim() != 2:
        raise ValueError("weight must be a 2D tensor")
    if out.dim() != 2:
        raise ValueError("out must be a 2D tensor")

    if not input.is_contiguous():
        raise ValueError("input must be contiguous (row-major)")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous (row-major)")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous (row-major)")

    if input.shape[1] != weight.shape[1]:
        raise ValueError(
            f"input.shape[1] ({input.shape[1]}) must equal weight.shape[1] ({weight.shape[1]})"
        )
    if out.shape[0] != input.shape[0]:
        raise ValueError(
            f"out.shape[0] ({out.shape[0]}) must equal input.shape[0] ({input.shape[0]})"
        )
    if out.shape[1] != weight.shape[0]:
        raise ValueError(
            f"out.shape[1] ({out.shape[1]}) must equal weight.shape[0] ({weight.shape[0]})"
        )
    output_features = weight.shape[0]

    if output_features % 16 != 0:
        raise ValueError(
            f"output_features ({output_features}) must be a multiple of 16 (tile alignment)"
        )

    if input.dtype != torch.bfloat16:
        raise ValueError("input must be bfloat16")
    if weight.dtype != torch.bfloat16:
        raise ValueError("weight must be bfloat16")
    if out.dtype != torch.bfloat16:
        raise ValueError("out must be bfloat16")

    if bias is not None:
        if bias.dim() != 1:
            raise ValueError("bias must be a 1D tensor")
        if bias.shape[0] != weight.shape[0]:
            raise ValueError(
                f"bias.shape[0] ({bias.shape[0]}) must equal weight.shape[0] ({weight.shape[0]})"
            )
        if bias.dtype != torch.bfloat16:
            raise ValueError("bias must be bfloat16")
        if not bias.is_contiguous():
            raise ValueError("bias must be contiguous")

    return True


@functools.cache
def get_tinygemm2_module():
    module = gen_tinygemm2_module().build_and_load()

    @register_custom_op(
        "flashinfer::tinygemm2_op",
        mutates_args=["out"],
    )
    def tinygemm2_op_impl(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        use_pdl: bool = False,
    ) -> None:
        module.tinygemm2_op(input, weight, bias, out, use_pdl)

    @register_custom_op(
        "flashinfer::tinygemm2_nobias_op",
        mutates_args=["out"],
    )
    def tinygemm2_nobias_op_impl(
        input: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        use_pdl: bool = False,
    ) -> None:
        module.tinygemm2_nobias_op(input, weight, out, use_pdl)

    return SimpleNamespace(
        tinygemm2_op=tinygemm2_op_impl,
        tinygemm2_nobias_op=tinygemm2_nobias_op_impl,
    )


@backend_requirement({}, common_check=_tinygemm_bf16_shape_checks)
@flashinfer_api
def tinygemm_bf16(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_pdl: bool = False,
) -> None:
    """SM90+ optimized small GEMM: out = input @ weight.T + bias (equivalent to F.linear).

    A latency-optimized, warp-specialized GEMM designed for tiny batch sizes (ideally
    1-8 rows, where a single TILE_N=8 tile covers the entire batch dimension) using
    Ampere-style HMMA instructions. Uses TMA for async bulk data loads and
    mma.sync.aligned.m16n8k16 tensor core instructions with BF16 input/weight/bias/output
    and FP32 internal accumulation. The warp-specialized design (384 threads: 4 compute +
    8 DMA warps) with 16 pipeline stages and 4x stage unroll trades off peak throughput
    in favor of minimal latency.

    From TensorRT-LLM tinygemm2 kernel.

    Args:
        input: Input activations of shape (batch_size, input_features). Must be
            bfloat16, contiguous. input_features must be a multiple of 64.
        weight: Weight matrix of shape (output_features, input_features). Must be
            bfloat16, contiguous (row-major). output_features must be a multiple of 16.
        out: Pre-allocated output tensor of shape (batch_size, output_features).
            Must be bfloat16, contiguous. Mutated in-place.
        bias: Optional bias vector of shape (output_features,). Must be bfloat16,
            contiguous. If None, zero bias is used.
        use_pdl: Enable Programmatic Dependent Launch (stream serialization).
            When True, the kernel uses cudaGridDependencySynchronize() to overlap
            DMA with the preceding kernel's compute. Only enable when ALL preceding
            stream operations also use PDL, otherwise the kernel hangs. Defaults
            to False.

    Raises:
        ValueError: If tensor dimensions, dtypes, or alignment constraints are violated.

    Note:
        This kernel requires SM90+ (Hopper or newer).
    """
    if bias is None:
        get_tinygemm2_module().tinygemm2_nobias_op(input, weight, out, use_pdl)
    else:
        get_tinygemm2_module().tinygemm2_op(input, weight, bias, out, use_pdl)


@supported_compute_capability([100, 103, 110, 120, 121])
def _tinygemm_fp8_shape_checks(A, B, A_scale, B_scale, dtype, bias=None, use_pdl=False):
    if A.dim() != 2:
        raise ValueError("A must be a 2D tensor")
    if B.dim() != 2:
        raise ValueError("B must be a 2D tensor")

    if not A.is_contiguous():
        raise ValueError("A must be contiguous (row-major)")
    if B.stride(0) != 1 or B.stride(1) != B.shape[0]:
        raise ValueError("B must be tightly packed column-major with shape (k, n)")

    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"A.shape[1] ({A.shape[1]}) must equal B.shape[0] ({B.shape[0]})"
        )

    if A.shape[1] % 128 != 0:
        raise ValueError(
            f"A.shape[1] ({A.shape[1]}) must be a multiple of 128 (tile alignment)"
        )
    if B.shape[1] % 16 != 0:
        raise ValueError(
            f"B.shape[1] ({B.shape[1]}) must be a multiple of 16 (tile alignment)"
        )

    if A.dtype != torch.float8_e4m3fn:
        raise ValueError("A must be float8_e4m3fn")
    if B.dtype != torch.float8_e4m3fn:
        raise ValueError("B must be float8_e4m3fn")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("dtype must be torch.float16 or torch.bfloat16")

    for name, scale in (("A_scale", A_scale), ("B_scale", B_scale)):
        if not isinstance(scale, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if scale.device != A.device:
            raise ValueError(f"{name} must be on the same device as A")
        if scale.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if scale.numel() != 1:
            raise ValueError(f"{name} must contain exactly one element")
        if not scale.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if bias is not None:
        if bias.dim() != 1:
            raise ValueError("bias must be a 1D tensor")
        if bias.shape[0] != B.shape[1]:
            raise ValueError(
                f"bias.shape[0] ({bias.shape[0]}) must equal B.shape[1] ({B.shape[1]})"
            )
        if bias.dtype != dtype:
            raise ValueError(f"bias must match output dtype ({dtype})")
        if bias.device != A.device:
            raise ValueError("bias must be on the same device as A")
        if not bias.is_contiguous():
            raise ValueError("bias must be contiguous")

    return True


@functools.cache
def get_tinygemm_fp8_module():
    module = gen_tinygemm_fp8_module().build_and_load()

    @register_custom_op(
        "flashinfer::tinygemm_fp8_op",
        mutates_args=["out"],
    )
    def tinygemm_fp8_op_impl(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        use_pdl: bool = False,
    ) -> None:
        module.tinygemm_fp8_op(A, B, A_scale, B_scale, bias, out, use_pdl)

    @register_custom_op(
        "flashinfer::tinygemm_fp8_nobias_op",
        mutates_args=["out"],
    )
    def tinygemm_fp8_nobias_op_impl(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        out: torch.Tensor,
        use_pdl: bool = False,
    ) -> None:
        module.tinygemm_fp8_nobias_op(A, B, A_scale, B_scale, out, use_pdl)

    return SimpleNamespace(
        tinygemm_fp8_op=tinygemm_fp8_op_impl,
        tinygemm_fp8_nobias_op=tinygemm_fp8_nobias_op_impl,
    )


@backend_requirement({}, common_check=_tinygemm_fp8_shape_checks)
@flashinfer_api
def tinygemm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
    use_pdl: bool = False,
) -> torch.Tensor:
    r"""Blackwell+ latency-optimized FP8 GEMM with optional bias.

    This kernel is the FP8 counterpart to ``tinygemm_bf16``. It computes
    ``out = (A @ B) * (A_scale * B_scale) + bias`` using e4m3 inputs and FP32
    accumulation, then converts to the requested output dtype.

    Parameters
    ----------
    A: torch.Tensor
        Row-major input tensor of shape ``(m, k)``, dtype ``torch.float8_e4m3fn``.

    B: torch.Tensor
        Column-major weight tensor of shape ``(k, n)``, dtype
        ``torch.float8_e4m3fn``. A convenient way to build this layout is
        ``weight_fp8.t()`` from a row-major ``(n, k)`` weight tensor.

    A_scale: torch.Tensor
        Scalar float32 tensor containing the dequant scale for ``A``.

    B_scale: torch.Tensor
        Scalar float32 tensor containing the dequant scale for ``B``.

    dtype: torch.dtype
        Output dtype. Must be ``torch.float16`` or ``torch.bfloat16``.

    bias: Optional[torch.Tensor]
        Optional bias vector of shape ``(n,)`` with dtype matching ``dtype``.

    use_pdl: bool
        Enable Programmatic Dependent Launch. As with ``tinygemm_bf16``, only
        enable this when preceding work on the same stream is also PDL-aware.

    Returns
    -------
    torch.Tensor
        Output tensor of shape ``(m, n)`` with dtype ``dtype``.
    """
    out = torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=dtype)
    if bias is None:
        get_tinygemm_fp8_module().tinygemm_fp8_nobias_op(
            A, B, A_scale, B_scale, out, use_pdl
        )
    else:
        get_tinygemm_fp8_module().tinygemm_fp8_op(
            A, B, A_scale, B_scale, bias, out, use_pdl
        )
    return out
