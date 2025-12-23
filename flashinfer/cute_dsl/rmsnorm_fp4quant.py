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

Fused RMSNorm + FP4 Quantization using CuTe-DSL
================================================

High-performance fused kernel for RMS normalization followed by FP4 quantization.
This is an alternative backend to cuDNN, using CuTe-DSL for maximum flexibility
and performance on SM100+ architectures.

Supports both NVFP4 and MXFP4 quantization formats.
"""

import functools
import math
import operator
from typing import Callable, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64, Uint8
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from .utils import make_ptr
from ..api_logging import flashinfer_api


# =============================================================================
# Constants
# =============================================================================

FLOAT4_E2M1_MAX = 6.0  # Maximum value representable in FP4 E2M1
FLOAT8_E4M3_MAX = 448.0  # Maximum value representable in FP8 E4M3
SF_VEC_SIZE = 16  # Elements per scale factor block
COPY_BITS = 128  # 128-bit vectorized loads


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
# PTX Intrinsics - Cluster Operations
# =============================================================================


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    """Map smem pointer to address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store Float32 value to shared memory on a remote CTA in the cluster."""
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];",
        "r,f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:
    """Get pointer to element at coordinate in tensor."""
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


# =============================================================================
# PTX Intrinsics - 128-bit Vectorized Global Loads
# =============================================================================


@dsl_user_op
def ld_global_v4_u32(
    base_ptr: Int64, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load 128 bits (4 x uint32) from global memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    v0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    v1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    v2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    v3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)

    return Uint32(v0), Uint32(v1), Uint32(v2), Uint32(v3)


@dsl_user_op
def st_global_u64(base_ptr: Int64, value: Uint64, *, loc=None, ip=None):
    """Store 64 bits to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint64(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_global_f32(base_ptr: Int64, value: Float32, *, loc=None, ip=None):
    """Store 32-bit float to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def ld_global_f32(base_ptr: Int64, *, loc=None, ip=None) -> Float32:
    """Load 32-bit float from global memory."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def st_global_v4_u32(
    base_ptr: Int64,
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    *,
    loc=None,
    ip=None,
):
    """Store 128 bits (4 x uint32) to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
            Uint32(v2).ir_value(loc=loc, ip=ip),
            Uint32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def fabs_f32(val: Float32, *, loc=None, ip=None) -> Float32:
    """Compute absolute value of float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmax_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute max of two float32 values."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def float2_to_half2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert two float32 values to packed half2 (uint32)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "cvt.rn.f16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def float2_to_bfloat2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert two float32 values to packed bfloat16x2 (uint32)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def atomic_max_global_u32(
    base_ptr: Int64, value: Uint32, *, loc=None, ip=None
) -> Uint32:
    """Atomically update global memory with max(current, value) for uint32.

    Uses atom.global.max.u32 which is supported on all architectures.
    For positive floats, we use float-as-uint trick since IEEE754 preserves order.
    Returns the old value (which we ignore).
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(base_ptr).ir_value(loc=loc, ip=ip),
                Uint32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.max.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def float_as_uint(value: Float32, *, loc=None, ip=None) -> Uint32:
    """Reinterpret float32 bits as uint32.

    For positive floats, IEEE754 representation preserves ordering when
    interpreted as unsigned integers, so atomicMax(uint32) can be used
    to find max of positive floats.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def uint_as_float(value: Uint32, *, loc=None, ip=None) -> Float32:
    """Reinterpret uint32 bits as float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(value).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def griddepcontrol_launch_dependents(*, loc=None, ip=None):
    """Signal dependent kernels to launch (PDL).

    Issuing this instruction hints a dependent kernel to launch earlier.
    Used for overlapping kernel launches with computation.
    """
    llvm.inline_asm(
        None,
        [],
        "griddepcontrol.launch_dependents;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def griddepcontrol_wait(*, loc=None, ip=None):
    """Wait for the previous kernel's grid to finish (PDL).

    This instruction blocks until the previous grid has completed
    and all memory operations are flushed.
    """
    llvm.inline_asm(
        None,
        [],
        "griddepcontrol.wait;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def get_ptr_as_int64(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None) -> Int64:
    """Get the memory address of tensor[offset] as Int64."""
    elem_ptr = tensor.iterator + Int32(offset)
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int64(ptr_int)


@dsl_user_op
def rcp_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """Fast reciprocal using PTX rcp.approx.ftz.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmin_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute min of two float32 values using PTX min.f32 (branchless clamping)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# Half2 SIMD Intrinsics
# =============================================================================


@dsl_user_op
def half2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Multiply two Half2 values element-wise: (a.x*b.x, a.y*b.y)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "mul.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Multiply two BFloat2 values element-wise: (a.x*b.x, a.y*b.y)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "mul.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# Half2 SIMD Max-Abs Intrinsics
# =============================================================================


@dsl_user_op
def habs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """Half2 absolute value - clears sign bits of both fp16 values."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0x7FFF7FFF;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def hmax2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Half2 max - element-wise max of 2 fp16 pairs."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def hmax_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract max of 2 fp16 values in half2 as float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 h0, h1;
                .reg .f32 f0, f1;
                mov.b32 {h0, h1}, $1;
                cvt.f32.f16 f0, h0;
                cvt.f32.f16 f1, h1;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_habs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """BFloat16x2 absolute value - clears sign bits of both bf16 values."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0x7FFF7FFF;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_hmax2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """BFloat16x2 max - element-wise max of 2 bf16 pairs."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_hmax_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract max of 2 bf16 values in bfloat2 as float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 f0, f1;
                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 f0, lo;
                mov.b32 f1, hi;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# Half2 to Float2 Scaled Conversion
# =============================================================================


@dsl_user_op
def half2_to_float2_scaled(
    h2: Uint32, scale: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Convert half2 to float2 AND multiply by scale."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(h2).ir_value(loc=loc, ip=ip), Float32(scale).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 h0, h1;
            .reg .f32 f0, f1;
            mov.b32 {h0, h1}, $2;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            mul.f32 $0, f0, $3;
            mul.f32 $1, f1, $3;
        }
        """,
        "=f,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)

    return Float32(f0), Float32(f1)


@dsl_user_op
def bfloat2_to_float2_scaled(
    bf2: Uint32, scale: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Convert bfloat16x2 to float2 AND multiply by scale."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(bf2).ir_value(loc=loc, ip=ip), Float32(scale).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            .reg .f32 f0, f1;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 f0, lo;
            mov.b32 f1, hi;
            mul.f32 $0, f0, $3;
            mul.f32 $1, f1, $3;
        }
        """,
        "=f,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)

    return Float32(f0), Float32(f1)


# =============================================================================
# FP8 E4M3 Intrinsics
# =============================================================================


@dsl_user_op
def cvt_f32_to_e4m3(a: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert float32 to E4M3 using native cvt.rn.satfinite.e4m3x2.f32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .f32 zero;
                mov.f32 zero, 0f00000000;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fp8_e4m3_to_f32_and_rcp(fp8_val: Uint32, *, loc=None, ip=None) -> Float32:
    """Convert FP8 E4M3 to float32 AND compute reciprocal."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(fp8_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero;
                .reg .u32 exp_u, mant_u;
                .reg .s32 exp_s;
                .reg .f32 exp_f, mant_f, fp8_float, result;

                setp.eq.u32 p_zero, $1, 0;
                and.b32 mant_u, $1, 7;
                shr.b32 exp_u, $1, 3;
                and.b32 exp_u, exp_u, 15;
                sub.s32 exp_s, exp_u, 7;
                cvt.rn.f32.s32 exp_f, exp_s;
                ex2.approx.f32 exp_f, exp_f;
                cvt.rn.f32.u32 mant_f, mant_u;
                fma.rn.f32 mant_f, mant_f, 0f3E000000, 0f3F800000;
                mul.f32 fp8_float, exp_f, mant_f;
                rcp.approx.ftz.f32 result, fp8_float;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# UE8M0 Intrinsics (for MXFP4)
# =============================================================================


@dsl_user_op
def cvt_f32_to_ue8m0(max_val: Float32, *, loc=None, ip=None) -> Uint32:
    """
    Convert float32 max value to UE8M0 scale factor.

    UE8M0 is unsigned 8-bit exponent-only format:
    - value = 2^(ue8m0 - 127)
    - ue8m0 = ceil(log2(max_val)) + 127

    Uses lg2.approx.f32 for fast log2 approximation.
    Uses cvt.rpi (round towards positive infinity, i.e., ceiling).
    Returns value clamped to [0, 255].
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(max_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero, p_neg, p_ovf;
                .reg .f32 log2_val;
                .reg .s32 exp_int, result;

                // Check for zero/negative
                setp.le.f32 p_zero, $1, 0f00000000;

                // Compute ceil(log2(max_val)) using cvt.rpi (round towards +inf)
                lg2.approx.f32 log2_val, $1;
                cvt.rpi.s32.f32 exp_int, log2_val;

                // Add bias and clamp to [0, 255]
                add.s32 result, exp_int, 127;
                setp.lt.s32 p_neg, result, 0;
                setp.gt.s32 p_ovf, result, 255;
                selp.s32 result, 0, result, p_neg;
                selp.s32 result, 255, result, p_ovf;
                selp.s32 $0, 0, result, p_zero;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ue8m0_to_output_scale(ue8m0_val: Uint32, *, loc=None, ip=None) -> Float32:
    """
    Convert UE8M0 to output_scale for MXFP4 quantization.

    UE8M0 value = 2^(ue8m0 - 127)
    Returns 1 / 2^(ue8m0 - 127) = 2^(127 - ue8m0)
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(ue8m0_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero;
                .reg .s32 neg_exp;
                .reg .f32 neg_exp_f, result;

                // Check for zero
                setp.eq.u32 p_zero, $1, 0;

                // Compute 2^(127 - ue8m0) = 1 / 2^(ue8m0 - 127)
                sub.s32 neg_exp, 127, $1;
                cvt.rn.f32.s32 neg_exp_f, neg_exp;
                ex2.approx.f32 result, neg_exp_f;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# E2M1 Conversion
# =============================================================================


@dsl_user_op
def cvt_e2m1x8_f32(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Convert eight float32 values to eight E2M1 (4-bit) values packed into uint32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v0).ir_value(loc=loc, ip=ip),
                Float32(v1).ir_value(loc=loc, ip=ip),
                Float32(v2).ir_value(loc=loc, ip=ip),
                Float32(v3).ir_value(loc=loc, ip=ip),
                Float32(v4).ir_value(loc=loc, ip=ip),
                Float32(v5).ir_value(loc=loc, ip=ip),
                Float32(v6).ir_value(loc=loc, ip=ip),
                Float32(v7).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 byte0, byte1, byte2, byte3;
                cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;
                mov.b32 $0, {byte0, byte1, byte2, byte3};
            }
            """,
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# Warp, Block, and Cluster Reduction Utilities
# =============================================================================


@cute.jit
def warp_reduce(val, op, width: cutlass.Constexpr[int] = 32):
    """Reduce across threads in a warp using butterfly shuffle."""
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
        return val


@cute.jit
def block_reduce(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: Float32,
) -> Float32:
    """Block reduction across multiple warps using shared memory."""
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()

    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return warp_reduce(block_reduce_val, op)


@cute.jit
def cluster_reduce(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: cute.Pointer,
    cluster_n: cutlass.Constexpr[int],
    init_val: Float32,
) -> Float32:
    """Cluster reduction across multiple CTAs using mbarrier."""
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()

    rows_per_block = reduction_buffer.shape[0]
    warps_per_row = reduction_buffer.shape[1][0]

    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    # Warp 0 sets up mbarrier transaction count
    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            expected_bytes = num_warps * cluster_n * 4
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, expected_bytes)

    # Each lane < cluster_n writes to a different CTA's shared memory
    if lane_idx < cluster_n:
        store_shared_remote(
            val,
            elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )

    # Wait for all cluster writes
    cute.arch.mbarrier_wait(mbar_ptr, phase=0)

    # Reduce across all values
    num_total = warps_per_row * cluster_n
    num_iter = cute.ceil_div(num_total, 32)

    block_reduce_val = init_val
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * 32
        if idx < num_total:
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])

    return warp_reduce(block_reduce_val, op)


@cute.jit
def row_reduce(
    x: cute.TensorSSA,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
    mbar_ptr,
    cluster_n: cutlass.Constexpr[int],
    init_val: Float32,
):
    """Row reduction with optional cluster support."""
    local_val = x.reduce(op, init_val=init_val, reduction_profile=0)

    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax,
    }[op]
    warp_width = min(threads_per_row, 32)
    warp_val = warp_reduce(local_val, warp_op, width=warp_width)

    warps_per_row = max(threads_per_row // 32, 1)

    if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
        if cutlass.const_expr(cluster_n == 1):
            return block_reduce(warp_val, warp_op, reduction_buffer, init_val)
        else:
            return cluster_reduce(
                warp_val, warp_op, reduction_buffer, mbar_ptr, cluster_n, init_val
            )
    else:
        return warp_val


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
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            tXpX[rest_v, 0, rest_k] = cute.elem_less(
                tXcX[(0, rest_v), 0, rest_k][1], limit
            )
    return tXpX


# =============================================================================
# CuTe-DSL Kernel Class
# =============================================================================


class RMSNormFP4QuantKernel:
    """
    Fused RMSNorm + FP4 Quantization Kernel.

    Key optimizations:
    1. Half2/BFloat2 SIMD for max-abs computation
    2. Branchless scale clamping via fmin_f32
    3. Cluster synchronization for large H dimensions
    4. Direct 128-bit vectorized global loads
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        block_size: int,
        output_swizzled: bool,
        is_fp16: bool,
        sm_version: int | None = None,
        scale_format: str | None = None,  # "e4m3" or "ue8m0", None = auto
    ):
        self.dtype = dtype
        self.H = H
        self.block_size = block_size
        self.output_swizzled = output_swizzled
        self.is_fp16 = is_fp16
        self.sm_version = sm_version if sm_version is not None else get_sm_version()

        # Auto-select scale format based on block_size if not specified
        if scale_format is None:
            self.scale_format = "ue8m0" if block_size == 32 else "e4m3"
        else:
            self.scale_format = scale_format

        # Validate configuration
        assert block_size in (16, 32), f"block_size must be 16 or 32, got {block_size}"
        assert self.scale_format in ("e4m3", "ue8m0"), (
            "scale_format must be 'e4m3' or 'ue8m0'"
        )

        # Compute cluster size
        self.cluster_n = self._compute_cluster_n(H, dtype, self.sm_version)

        # H per CTA for cluster case
        self.H_per_cta = H // self.cluster_n

        # Compute thread configuration
        self.threads_per_row = self._compute_threads_per_row(self.H_per_cta)
        self.num_threads = self._compute_num_threads(self.H_per_cta)
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

        # Vectorization parameters
        elem_bytes = dtype.width // 8
        self.vec_size = COPY_BITS // 8 // elem_bytes
        self.num_vec_blocks = max(
            1,
            (self.H_per_cta // self.vec_size + self.threads_per_row - 1)
            // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

        # Scale factor block count (full H, not per CTA)
        self.num_sf_blocks_per_row = H // block_size

        # Swizzle parameters
        if output_swizzled:
            num_col_vecs = H // block_size
            self.num_k_tiles = (num_col_vecs + 3) // 4
            self.k_tile_stride = 512

    @staticmethod
    def _compute_cluster_n(H: int, dtype: cutlass.Numeric, sm_version: int) -> int:
        """Compute optimal cluster size based on H and device shared memory.

        Dynamically determines the minimum cluster_n that fits within the
        device's shared memory limit, making it compatible with different
        GPU architectures (e.g., SM100 with 228KB vs SM120 with 128KB).
        """
        if sm_version < 90:
            return 1

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        max_smem_bytes = props.shared_memory_per_block_optin
        elem_size = dtype.width // 8

        for cluster_n in [1, 2, 4, 8, 16]:
            if H % cluster_n != 0:
                continue
            smem_needed = RMSNormFP4QuantKernel._estimate_smem_bytes(
                H, cluster_n, elem_size
            )
            if smem_needed <= max_smem_bytes:
                return cluster_n

        return 16

    @staticmethod
    def _compute_threads_per_row(H_per_cta: int) -> int:
        """Compute optimal threads per row based on H per CTA."""
        if H_per_cta <= 64:
            return 8
        elif H_per_cta <= 128:
            return 16
        elif H_per_cta <= 3072:
            return 32
        elif H_per_cta <= 6144:
            return 64
        elif H_per_cta <= 16384:
            return 128
        else:
            return 256

    @staticmethod
    def _compute_num_threads(H_per_cta: int) -> int:
        """Compute total threads per block based on H per CTA."""
        return 128 if H_per_cta <= 16384 else 256

    @staticmethod
    def _estimate_smem_bytes(H: int, cluster_n: int, elem_size: int) -> int:
        """Estimate shared memory bytes needed for given configuration.

        This is used to dynamically determine cluster_n based on device
        shared memory limits.
        """
        H_per_cta = H // cluster_n
        threads_per_row = RMSNormFP4QuantKernel._compute_threads_per_row(H_per_cta)
        num_threads = RMSNormFP4QuantKernel._compute_num_threads(H_per_cta)
        rows_per_block = num_threads // threads_per_row
        warps_per_row = max(threads_per_row // 32, 1)

        vec_size = COPY_BITS // 8 // elem_size
        num_vec_blocks = max(
            1, (H_per_cta // vec_size + threads_per_row - 1) // threads_per_row
        )
        cols_per_tile = vec_size * num_vec_blocks * threads_per_row

        tile_bytes = rows_per_block * cols_per_tile * elem_size

        if cluster_n == 1:
            # 2 tiles: sX, sW + reduction buffer
            return 2 * tile_bytes + rows_per_block * warps_per_row * 4
        else:
            # 1 tile: sX + larger reduction buffer + mbarrier
            return tile_bytes + rows_per_block * warps_per_row * cluster_n * 4 + 8

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
        """Calculate shared memory requirement."""
        # Input tile
        tile_bytes = self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)

        # Reduction buffer (with cluster support)
        if self.cluster_n == 1:
            reduction_bytes = self.rows_per_block * self.warps_per_row * 4
        else:
            reduction_bytes = (
                self.rows_per_block * self.warps_per_row * self.cluster_n * 4
            )

        # mbarrier (for cluster mode)
        mbar_bytes = 8 if self.cluster_n > 1 else 0

        return tile_bytes + reduction_bytes + mbar_bytes

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        s_ptr: cute.Pointer,
        M: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        """Host function to launch the kernel.

        Takes raw pointers and batch size M, creates tensors internally.
        This avoids the overhead of from_dlpack() at runtime.
        """
        H = self.H

        # Create tensors from pointers with known layouts
        # Layout: (M, H) with row-major order (stride H for rows, 1 for columns)
        mX = cute.make_tensor(
            x_ptr,
            layout=cute.make_ordered_layout((M, H), order=(1, 0)),
        )
        mW = cute.make_tensor(
            w_ptr,
            layout=cute.make_layout((H,)),
        )
        mY = cute.make_tensor(
            y_ptr,
            layout=cute.make_ordered_layout((M, H // 2), order=(1, 0)),
        )

        # Create mS tensor with appropriate layout based on swizzle mode
        if cutlass.const_expr(self.output_swizzled):
            # For swizzled output, use 1D layout
            # The swizzle writes use flat offsets: mS[swizzled_offset]
            # We compute the swizzled offset in the kernel, so just use a 1D layout
            # with stride 1 to treat the pointer as a flat array
            num_m_tiles = (M + Int32(127)) // Int32(128)
            swizzled_size = num_m_tiles * Int32(self.num_k_tiles * self.k_tile_stride)
            mS = cute.make_tensor(
                s_ptr,
                layout=cute.make_layout((swizzled_size,), stride=(Int32(1),)),
            )
        else:
            # For non-swizzled output, use 2D row-major layout
            mS = cute.make_tensor(
                s_ptr,
                layout=cute.make_ordered_layout(
                    (M, self.num_sf_blocks_per_row), order=(1, 0)
                ),
            )

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
        self.kernel(mX, mW, mY, mS, M, eps, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), self.cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1]
            if cutlass.const_expr(self.cluster_n > 1)
            else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        M: Int32,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        """Device kernel with cluster synchronization for large H."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        H = self.H
        block_size = self.block_size
        num_sf_blocks_per_row = self.num_sf_blocks_per_row
        is_fp16 = self.is_fp16
        cluster_n = self.cluster_n

        # Get cluster position
        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        threads_per_row = tv_layout.shape[0][0]
        warps_per_row = max(threads_per_row // 32, 1)
        rows_per_block = tiler_mn[0]

        # Thread position within row
        lane_in_row = tidx % threads_per_row
        row_in_block = tidx // threads_per_row

        # Precompute
        fp4_max_rcp = rcp_approx_ftz(Float32(FLOAT4_E2M1_MAX))

        # ==================================================================
        # Allocate shared memory
        # ==================================================================
        smem = cutlass.utils.SmemAllocator()

        # Shared memory tile for input (for sum-of-squares)
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        # Reduction buffer (with cluster support)
        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(Int64, num_elems=1)

        # ==================================================================
        # Initialize cluster
        # ==================================================================
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # ==================================================================
        # Create identity tensor for coordinate tracking
        # ==================================================================
        idX = cute.make_identity_tensor(mX.shape)

        # Slice for this block's rows (and cluster CTA's slice of columns)
        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        # Expand weight to 2D for tiled copy
        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        _ = cute.local_tile(
            mW_2d, tiler_mn, (0, cluster_y)
        )  # gW unused but call needed

        # ==================================================================
        # Create TiledCopy for sum-of-squares phase
        # ==================================================================
        copy_atom_load_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=COPY_BITS,
        )

        tiled_copy_load = cute.make_tiled_copy(
            copy_atom_load_async, tv_layout, tiler_mn
        )
        thr_copy_X = tiled_copy_load.get_slice(tidx)

        # Partition tensors
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXcX = thr_copy_X.partition_S(cX)

        # Register fragments
        tXrX = cute.make_fragment_like(tXgX)

        # ==================================================================
        # Bounds checking
        # ==================================================================
        tXpX = predicate_k(tXcX, limit=H)

        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ==================================================================
        # Phase 1: Async copy global → shared (for sum-of-squares)
        # ==================================================================
        if row_in_bounds:
            cute.copy(copy_atom_load_async, tXgX, tXsX, pred=tXpX)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        # ==================================================================
        # Phase 2: Compute sum of squares with cluster reduction
        # ==================================================================
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        # Sum of squares
        x_sq = x * x
        sum_sq = row_reduce(
            x_sq,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer,
            mbar_ptr,
            cluster_n,
            Float32(0.0),
        )

        # rstd = 1 / sqrt(mean(x²) + eps)
        mean_sq = sum_sq / H  # Use full H, not H_per_cta
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        # Sync after reduction
        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ==================================================================
        # Phase 3: RMSNorm + Quantize with Vectorized Global Loads
        # ==================================================================
        # Get actual row index
        actual_row_idx = bidx * rows_per_block + row_in_block

        if actual_row_idx < M:
            # Process SF blocks assigned to this thread
            num_sf_per_thread = (
                num_sf_blocks_per_row + threads_per_row - 1
            ) // threads_per_row

            for sf_iter in range(num_sf_per_thread):
                sf_idx = lane_in_row + sf_iter * threads_per_row

                if sf_idx < num_sf_blocks_per_row:
                    block_start = sf_idx * block_size

                    # =======================================================
                    # Load elements from GLOBAL MEMORY as Half2
                    # =======================================================

                    if cutlass.const_expr(block_size == 16):
                        # Get pointers for X and W
                        x_ptr0 = get_ptr_as_int64(mX, actual_row_idx * H + block_start)
                        x_ptr1 = get_ptr_as_int64(
                            mX, actual_row_idx * H + block_start + Int32(8)
                        )

                        w_ptr0 = get_ptr_as_int64(mW, block_start)
                        w_ptr1 = get_ptr_as_int64(mW, block_start + Int32(8))

                        # Load 16 elements of X as 8 Half2 pairs
                        x0, x1, x2, x3 = ld_global_v4_u32(x_ptr0)
                        x4, x5, x6, x7 = ld_global_v4_u32(x_ptr1)

                        # Load 16 elements of W as 8 Half2 pairs
                        w0, w1, w2, w3 = ld_global_v4_u32(w_ptr0)
                        w4, w5, w6, w7 = ld_global_v4_u32(w_ptr1)

                        # Multiply x * w in Half2/BFloat2
                        if cutlass.const_expr(is_fp16):
                            xw0 = half2_mul(x0, w0)
                            xw1 = half2_mul(x1, w1)
                            xw2 = half2_mul(x2, w2)
                            xw3 = half2_mul(x3, w3)
                            xw4 = half2_mul(x4, w4)
                            xw5 = half2_mul(x5, w5)
                            xw6 = half2_mul(x6, w6)
                            xw7 = half2_mul(x7, w7)

                            # Half2 SIMD max-abs tree reduction
                            abs0_h2 = habs2(xw0)
                            abs1_h2 = habs2(xw1)
                            abs2_h2 = habs2(xw2)
                            abs3_h2 = habs2(xw3)
                            abs4_h2 = habs2(xw4)
                            abs5_h2 = habs2(xw5)
                            abs6_h2 = habs2(xw6)
                            abs7_h2 = habs2(xw7)

                            max01_h2 = hmax2(abs0_h2, abs1_h2)
                            max23_h2 = hmax2(abs2_h2, abs3_h2)
                            max45_h2 = hmax2(abs4_h2, abs5_h2)
                            max67_h2 = hmax2(abs6_h2, abs7_h2)
                            max0123_h2 = hmax2(max01_h2, max23_h2)
                            max4567_h2 = hmax2(max45_h2, max67_h2)
                            max_xw_h2 = hmax2(max0123_h2, max4567_h2)

                            max_xw = hmax_to_f32(max_xw_h2)
                            max_abs = max_xw * rstd

                            # Convert to Float32 for quantization
                            y0, y1 = half2_to_float2_scaled(xw0, rstd)
                            y2, y3 = half2_to_float2_scaled(xw1, rstd)
                            y4, y5 = half2_to_float2_scaled(xw2, rstd)
                            y6, y7 = half2_to_float2_scaled(xw3, rstd)
                            y8, y9 = half2_to_float2_scaled(xw4, rstd)
                            y10, y11 = half2_to_float2_scaled(xw5, rstd)
                            y12, y13 = half2_to_float2_scaled(xw6, rstd)
                            y14, y15 = half2_to_float2_scaled(xw7, rstd)
                        else:
                            xw0 = bfloat2_mul(x0, w0)
                            xw1 = bfloat2_mul(x1, w1)
                            xw2 = bfloat2_mul(x2, w2)
                            xw3 = bfloat2_mul(x3, w3)
                            xw4 = bfloat2_mul(x4, w4)
                            xw5 = bfloat2_mul(x5, w5)
                            xw6 = bfloat2_mul(x6, w6)
                            xw7 = bfloat2_mul(x7, w7)

                            # BFloat2 SIMD max-abs tree reduction
                            abs0_h2 = bfloat2_habs2(xw0)
                            abs1_h2 = bfloat2_habs2(xw1)
                            abs2_h2 = bfloat2_habs2(xw2)
                            abs3_h2 = bfloat2_habs2(xw3)
                            abs4_h2 = bfloat2_habs2(xw4)
                            abs5_h2 = bfloat2_habs2(xw5)
                            abs6_h2 = bfloat2_habs2(xw6)
                            abs7_h2 = bfloat2_habs2(xw7)

                            max01_h2 = bfloat2_hmax2(abs0_h2, abs1_h2)
                            max23_h2 = bfloat2_hmax2(abs2_h2, abs3_h2)
                            max45_h2 = bfloat2_hmax2(abs4_h2, abs5_h2)
                            max67_h2 = bfloat2_hmax2(abs6_h2, abs7_h2)
                            max0123_h2 = bfloat2_hmax2(max01_h2, max23_h2)
                            max4567_h2 = bfloat2_hmax2(max45_h2, max67_h2)
                            max_xw_h2 = bfloat2_hmax2(max0123_h2, max4567_h2)

                            max_xw = bfloat2_hmax_to_f32(max_xw_h2)
                            max_abs = max_xw * rstd

                            # Convert to Float32 for quantization
                            y0, y1 = bfloat2_to_float2_scaled(xw0, rstd)
                            y2, y3 = bfloat2_to_float2_scaled(xw1, rstd)
                            y4, y5 = bfloat2_to_float2_scaled(xw2, rstd)
                            y6, y7 = bfloat2_to_float2_scaled(xw3, rstd)
                            y8, y9 = bfloat2_to_float2_scaled(xw4, rstd)
                            y10, y11 = bfloat2_to_float2_scaled(xw5, rstd)
                            y12, y13 = bfloat2_to_float2_scaled(xw6, rstd)
                            y14, y15 = bfloat2_to_float2_scaled(xw7, rstd)

                        # =======================================================
                        # Compute scale factor (FP8 E4M3) - Branchless clamping
                        # =======================================================
                        scale_float = max_abs * fp4_max_rcp
                        scale_float = fmin_f32(scale_float, Float32(FLOAT8_E4M3_MAX))

                        scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
                        scale_fp8 = Uint8(scale_fp8_u32 & Uint32(0xFF))

                        inv_scale = fp8_e4m3_to_f32_and_rcp(scale_fp8_u32)

                        # =======================================================
                        # Store scale factor
                        # =======================================================
                        if cutlass.const_expr(self.output_swizzled):
                            inner_k_idx = sf_idx % Int32(4)
                            inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                            outer_m_idx = actual_row_idx % Int32(32)
                            k_tile_idx = sf_idx // Int32(4)
                            m_tile_idx = actual_row_idx // Int32(128)

                            m_tile_stride = self.num_k_tiles * self.k_tile_stride
                            swizzled_offset = (
                                m_tile_idx * m_tile_stride
                                + k_tile_idx * self.k_tile_stride
                                + outer_m_idx * Int32(16)
                                + inner_m_idx * Int32(4)
                                + inner_k_idx
                            )
                            mS[swizzled_offset] = scale_fp8
                        else:
                            mS[actual_row_idx, sf_idx] = scale_fp8

                        # =======================================================
                        # Quantize and pack FP4 values
                        # =======================================================
                        q0 = y0 * inv_scale
                        q1 = y1 * inv_scale
                        q2 = y2 * inv_scale
                        q3 = y3 * inv_scale
                        q4 = y4 * inv_scale
                        q5 = y5 * inv_scale
                        q6 = y6 * inv_scale
                        q7 = y7 * inv_scale
                        q8 = y8 * inv_scale
                        q9 = y9 * inv_scale
                        q10 = y10 * inv_scale
                        q11 = y11 * inv_scale
                        q12 = y12 * inv_scale
                        q13 = y13 * inv_scale
                        q14 = y14 * inv_scale
                        q15 = y15 * inv_scale

                        packed_lo = cvt_e2m1x8_f32(q0, q1, q2, q3, q4, q5, q6, q7)
                        packed_hi = cvt_e2m1x8_f32(q8, q9, q10, q11, q12, q13, q14, q15)

                        packed64 = (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)

                        out_offset = block_start // 2
                        out_ptr = get_ptr_as_int64(
                            mY, actual_row_idx * (H // 2) + out_offset
                        )
                        st_global_u64(out_ptr, packed64)

                    else:
                        # block_size == 32: Process in 2 chunks of 16
                        x_ptr0_c0 = get_ptr_as_int64(
                            mX, actual_row_idx * H + block_start
                        )
                        x_ptr1_c0 = get_ptr_as_int64(
                            mX, actual_row_idx * H + block_start + Int32(8)
                        )
                        w_ptr0_c0 = get_ptr_as_int64(mW, block_start)
                        w_ptr1_c0 = get_ptr_as_int64(mW, block_start + Int32(8))

                        x_ptr0_c1 = get_ptr_as_int64(
                            mX, actual_row_idx * H + block_start + Int32(16)
                        )
                        x_ptr1_c1 = get_ptr_as_int64(
                            mX, actual_row_idx * H + block_start + Int32(24)
                        )
                        w_ptr0_c1 = get_ptr_as_int64(mW, block_start + Int32(16))
                        w_ptr1_c1 = get_ptr_as_int64(mW, block_start + Int32(24))

                        x0_c0, x1_c0, x2_c0, x3_c0 = ld_global_v4_u32(x_ptr0_c0)
                        x4_c0, x5_c0, x6_c0, x7_c0 = ld_global_v4_u32(x_ptr1_c0)
                        w0_c0, w1_c0, w2_c0, w3_c0 = ld_global_v4_u32(w_ptr0_c0)
                        w4_c0, w5_c0, w6_c0, w7_c0 = ld_global_v4_u32(w_ptr1_c0)

                        x0_c1, x1_c1, x2_c1, x3_c1 = ld_global_v4_u32(x_ptr0_c1)
                        x4_c1, x5_c1, x6_c1, x7_c1 = ld_global_v4_u32(x_ptr1_c1)
                        w0_c1, w1_c1, w2_c1, w3_c1 = ld_global_v4_u32(w_ptr0_c1)
                        w4_c1, w5_c1, w6_c1, w7_c1 = ld_global_v4_u32(w_ptr1_c1)

                        if cutlass.const_expr(is_fp16):
                            xw0_c0 = half2_mul(x0_c0, w0_c0)
                            xw1_c0 = half2_mul(x1_c0, w1_c0)
                            xw2_c0 = half2_mul(x2_c0, w2_c0)
                            xw3_c0 = half2_mul(x3_c0, w3_c0)
                            xw4_c0 = half2_mul(x4_c0, w4_c0)
                            xw5_c0 = half2_mul(x5_c0, w5_c0)
                            xw6_c0 = half2_mul(x6_c0, w6_c0)
                            xw7_c0 = half2_mul(x7_c0, w7_c0)

                            xw0_c1 = half2_mul(x0_c1, w0_c1)
                            xw1_c1 = half2_mul(x1_c1, w1_c1)
                            xw2_c1 = half2_mul(x2_c1, w2_c1)
                            xw3_c1 = half2_mul(x3_c1, w3_c1)
                            xw4_c1 = half2_mul(x4_c1, w4_c1)
                            xw5_c1 = half2_mul(x5_c1, w5_c1)
                            xw6_c1 = half2_mul(x6_c1, w6_c1)
                            xw7_c1 = half2_mul(x7_c1, w7_c1)

                            abs0_c0_h2 = habs2(xw0_c0)
                            abs1_c0_h2 = habs2(xw1_c0)
                            abs2_c0_h2 = habs2(xw2_c0)
                            abs3_c0_h2 = habs2(xw3_c0)
                            abs4_c0_h2 = habs2(xw4_c0)
                            abs5_c0_h2 = habs2(xw5_c0)
                            abs6_c0_h2 = habs2(xw6_c0)
                            abs7_c0_h2 = habs2(xw7_c0)

                            abs0_c1_h2 = habs2(xw0_c1)
                            abs1_c1_h2 = habs2(xw1_c1)
                            abs2_c1_h2 = habs2(xw2_c1)
                            abs3_c1_h2 = habs2(xw3_c1)
                            abs4_c1_h2 = habs2(xw4_c1)
                            abs5_c1_h2 = habs2(xw5_c1)
                            abs6_c1_h2 = habs2(xw6_c1)
                            abs7_c1_h2 = habs2(xw7_c1)

                            max01_c0_h2 = hmax2(abs0_c0_h2, abs1_c0_h2)
                            max23_c0_h2 = hmax2(abs2_c0_h2, abs3_c0_h2)
                            max45_c0_h2 = hmax2(abs4_c0_h2, abs5_c0_h2)
                            max67_c0_h2 = hmax2(abs6_c0_h2, abs7_c0_h2)
                            max0123_c0_h2 = hmax2(max01_c0_h2, max23_c0_h2)
                            max4567_c0_h2 = hmax2(max45_c0_h2, max67_c0_h2)
                            max_c0_h2 = hmax2(max0123_c0_h2, max4567_c0_h2)

                            max01_c1_h2 = hmax2(abs0_c1_h2, abs1_c1_h2)
                            max23_c1_h2 = hmax2(abs2_c1_h2, abs3_c1_h2)
                            max45_c1_h2 = hmax2(abs4_c1_h2, abs5_c1_h2)
                            max67_c1_h2 = hmax2(abs6_c1_h2, abs7_c1_h2)
                            max0123_c1_h2 = hmax2(max01_c1_h2, max23_c1_h2)
                            max4567_c1_h2 = hmax2(max45_c1_h2, max67_c1_h2)
                            max_c1_h2 = hmax2(max0123_c1_h2, max4567_c1_h2)

                            max_xw_h2 = hmax2(max_c0_h2, max_c1_h2)
                            max_xw = hmax_to_f32(max_xw_h2)
                            max_abs = max_xw * rstd

                            y0_c0, y1_c0 = half2_to_float2_scaled(xw0_c0, rstd)
                            y2_c0, y3_c0 = half2_to_float2_scaled(xw1_c0, rstd)
                            y4_c0, y5_c0 = half2_to_float2_scaled(xw2_c0, rstd)
                            y6_c0, y7_c0 = half2_to_float2_scaled(xw3_c0, rstd)
                            y8_c0, y9_c0 = half2_to_float2_scaled(xw4_c0, rstd)
                            y10_c0, y11_c0 = half2_to_float2_scaled(xw5_c0, rstd)
                            y12_c0, y13_c0 = half2_to_float2_scaled(xw6_c0, rstd)
                            y14_c0, y15_c0 = half2_to_float2_scaled(xw7_c0, rstd)

                            y0_c1, y1_c1 = half2_to_float2_scaled(xw0_c1, rstd)
                            y2_c1, y3_c1 = half2_to_float2_scaled(xw1_c1, rstd)
                            y4_c1, y5_c1 = half2_to_float2_scaled(xw2_c1, rstd)
                            y6_c1, y7_c1 = half2_to_float2_scaled(xw3_c1, rstd)
                            y8_c1, y9_c1 = half2_to_float2_scaled(xw4_c1, rstd)
                            y10_c1, y11_c1 = half2_to_float2_scaled(xw5_c1, rstd)
                            y12_c1, y13_c1 = half2_to_float2_scaled(xw6_c1, rstd)
                            y14_c1, y15_c1 = half2_to_float2_scaled(xw7_c1, rstd)
                        else:
                            xw0_c0 = bfloat2_mul(x0_c0, w0_c0)
                            xw1_c0 = bfloat2_mul(x1_c0, w1_c0)
                            xw2_c0 = bfloat2_mul(x2_c0, w2_c0)
                            xw3_c0 = bfloat2_mul(x3_c0, w3_c0)
                            xw4_c0 = bfloat2_mul(x4_c0, w4_c0)
                            xw5_c0 = bfloat2_mul(x5_c0, w5_c0)
                            xw6_c0 = bfloat2_mul(x6_c0, w6_c0)
                            xw7_c0 = bfloat2_mul(x7_c0, w7_c0)

                            xw0_c1 = bfloat2_mul(x0_c1, w0_c1)
                            xw1_c1 = bfloat2_mul(x1_c1, w1_c1)
                            xw2_c1 = bfloat2_mul(x2_c1, w2_c1)
                            xw3_c1 = bfloat2_mul(x3_c1, w3_c1)
                            xw4_c1 = bfloat2_mul(x4_c1, w4_c1)
                            xw5_c1 = bfloat2_mul(x5_c1, w5_c1)
                            xw6_c1 = bfloat2_mul(x6_c1, w6_c1)
                            xw7_c1 = bfloat2_mul(x7_c1, w7_c1)

                            abs0_c0_h2 = bfloat2_habs2(xw0_c0)
                            abs1_c0_h2 = bfloat2_habs2(xw1_c0)
                            abs2_c0_h2 = bfloat2_habs2(xw2_c0)
                            abs3_c0_h2 = bfloat2_habs2(xw3_c0)
                            abs4_c0_h2 = bfloat2_habs2(xw4_c0)
                            abs5_c0_h2 = bfloat2_habs2(xw5_c0)
                            abs6_c0_h2 = bfloat2_habs2(xw6_c0)
                            abs7_c0_h2 = bfloat2_habs2(xw7_c0)

                            abs0_c1_h2 = bfloat2_habs2(xw0_c1)
                            abs1_c1_h2 = bfloat2_habs2(xw1_c1)
                            abs2_c1_h2 = bfloat2_habs2(xw2_c1)
                            abs3_c1_h2 = bfloat2_habs2(xw3_c1)
                            abs4_c1_h2 = bfloat2_habs2(xw4_c1)
                            abs5_c1_h2 = bfloat2_habs2(xw5_c1)
                            abs6_c1_h2 = bfloat2_habs2(xw6_c1)
                            abs7_c1_h2 = bfloat2_habs2(xw7_c1)

                            max01_c0_h2 = bfloat2_hmax2(abs0_c0_h2, abs1_c0_h2)
                            max23_c0_h2 = bfloat2_hmax2(abs2_c0_h2, abs3_c0_h2)
                            max45_c0_h2 = bfloat2_hmax2(abs4_c0_h2, abs5_c0_h2)
                            max67_c0_h2 = bfloat2_hmax2(abs6_c0_h2, abs7_c0_h2)
                            max0123_c0_h2 = bfloat2_hmax2(max01_c0_h2, max23_c0_h2)
                            max4567_c0_h2 = bfloat2_hmax2(max45_c0_h2, max67_c0_h2)
                            max_c0_h2 = bfloat2_hmax2(max0123_c0_h2, max4567_c0_h2)

                            max01_c1_h2 = bfloat2_hmax2(abs0_c1_h2, abs1_c1_h2)
                            max23_c1_h2 = bfloat2_hmax2(abs2_c1_h2, abs3_c1_h2)
                            max45_c1_h2 = bfloat2_hmax2(abs4_c1_h2, abs5_c1_h2)
                            max67_c1_h2 = bfloat2_hmax2(abs6_c1_h2, abs7_c1_h2)
                            max0123_c1_h2 = bfloat2_hmax2(max01_c1_h2, max23_c1_h2)
                            max4567_c1_h2 = bfloat2_hmax2(max45_c1_h2, max67_c1_h2)
                            max_c1_h2 = bfloat2_hmax2(max0123_c1_h2, max4567_c1_h2)

                            max_xw_h2 = bfloat2_hmax2(max_c0_h2, max_c1_h2)
                            max_xw = bfloat2_hmax_to_f32(max_xw_h2)
                            max_abs = max_xw * rstd

                            y0_c0, y1_c0 = bfloat2_to_float2_scaled(xw0_c0, rstd)
                            y2_c0, y3_c0 = bfloat2_to_float2_scaled(xw1_c0, rstd)
                            y4_c0, y5_c0 = bfloat2_to_float2_scaled(xw2_c0, rstd)
                            y6_c0, y7_c0 = bfloat2_to_float2_scaled(xw3_c0, rstd)
                            y8_c0, y9_c0 = bfloat2_to_float2_scaled(xw4_c0, rstd)
                            y10_c0, y11_c0 = bfloat2_to_float2_scaled(xw5_c0, rstd)
                            y12_c0, y13_c0 = bfloat2_to_float2_scaled(xw6_c0, rstd)
                            y14_c0, y15_c0 = bfloat2_to_float2_scaled(xw7_c0, rstd)

                            y0_c1, y1_c1 = bfloat2_to_float2_scaled(xw0_c1, rstd)
                            y2_c1, y3_c1 = bfloat2_to_float2_scaled(xw1_c1, rstd)
                            y4_c1, y5_c1 = bfloat2_to_float2_scaled(xw2_c1, rstd)
                            y6_c1, y7_c1 = bfloat2_to_float2_scaled(xw3_c1, rstd)
                            y8_c1, y9_c1 = bfloat2_to_float2_scaled(xw4_c1, rstd)
                            y10_c1, y11_c1 = bfloat2_to_float2_scaled(xw5_c1, rstd)
                            y12_c1, y13_c1 = bfloat2_to_float2_scaled(xw6_c1, rstd)
                            y14_c1, y15_c1 = bfloat2_to_float2_scaled(xw7_c1, rstd)

                        # Compute scale factor (E4M3 or UE8M0 based on scale_format)
                        if cutlass.const_expr(self.scale_format == "ue8m0"):
                            scale_float = max_abs * fp4_max_rcp
                            scale_ue8m0 = cvt_f32_to_ue8m0(scale_float)
                            scale_u8 = Uint8(scale_ue8m0 & Uint32(0xFF))
                            inv_scale = ue8m0_to_output_scale(scale_ue8m0)
                        else:
                            scale_float = max_abs * fp4_max_rcp
                            scale_float = fmin_f32(
                                scale_float, Float32(FLOAT8_E4M3_MAX)
                            )
                            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
                            scale_u8 = Uint8(scale_fp8_u32 & Uint32(0xFF))
                            inv_scale = fp8_e4m3_to_f32_and_rcp(scale_fp8_u32)

                        if cutlass.const_expr(self.output_swizzled):
                            inner_k_idx = sf_idx % Int32(4)
                            inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                            outer_m_idx = actual_row_idx % Int32(32)
                            k_tile_idx = sf_idx // Int32(4)
                            m_tile_idx = actual_row_idx // Int32(128)

                            m_tile_stride = self.num_k_tiles * self.k_tile_stride
                            swizzled_offset = (
                                m_tile_idx * m_tile_stride
                                + k_tile_idx * self.k_tile_stride
                                + outer_m_idx * Int32(16)
                                + inner_m_idx * Int32(4)
                                + inner_k_idx
                            )
                            mS[swizzled_offset] = scale_u8
                        else:
                            mS[actual_row_idx, sf_idx] = scale_u8

                        # Quantize and store chunk 0
                        q0 = y0_c0 * inv_scale
                        q1 = y1_c0 * inv_scale
                        q2 = y2_c0 * inv_scale
                        q3 = y3_c0 * inv_scale
                        q4 = y4_c0 * inv_scale
                        q5 = y5_c0 * inv_scale
                        q6 = y6_c0 * inv_scale
                        q7 = y7_c0 * inv_scale
                        q8 = y8_c0 * inv_scale
                        q9 = y9_c0 * inv_scale
                        q10 = y10_c0 * inv_scale
                        q11 = y11_c0 * inv_scale
                        q12 = y12_c0 * inv_scale
                        q13 = y13_c0 * inv_scale
                        q14 = y14_c0 * inv_scale
                        q15 = y15_c0 * inv_scale

                        packed_lo = cvt_e2m1x8_f32(q0, q1, q2, q3, q4, q5, q6, q7)
                        packed_hi = cvt_e2m1x8_f32(q8, q9, q10, q11, q12, q13, q14, q15)
                        packed64 = (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)
                        out_offset = block_start // 2
                        out_ptr = get_ptr_as_int64(
                            mY, actual_row_idx * (H // 2) + out_offset
                        )
                        st_global_u64(out_ptr, packed64)

                        # Quantize and store chunk 1
                        q0 = y0_c1 * inv_scale
                        q1 = y1_c1 * inv_scale
                        q2 = y2_c1 * inv_scale
                        q3 = y3_c1 * inv_scale
                        q4 = y4_c1 * inv_scale
                        q5 = y5_c1 * inv_scale
                        q6 = y6_c1 * inv_scale
                        q7 = y7_c1 * inv_scale
                        q8 = y8_c1 * inv_scale
                        q9 = y9_c1 * inv_scale
                        q10 = y10_c1 * inv_scale
                        q11 = y11_c1 * inv_scale
                        q12 = y12_c1 * inv_scale
                        q13 = y13_c1 * inv_scale
                        q14 = y14_c1 * inv_scale
                        q15 = y15_c1 * inv_scale

                        packed_lo = cvt_e2m1x8_f32(q0, q1, q2, q3, q4, q5, q6, q7)
                        packed_hi = cvt_e2m1x8_f32(q8, q9, q10, q11, q12, q13, q14, q15)
                        packed64 = (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)
                        out_offset = (block_start + 16) // 2
                        out_ptr = get_ptr_as_int64(
                            mY, actual_row_idx * (H // 2) + out_offset
                        )
                        st_global_u64(out_ptr, packed64)


# =============================================================================
# Kernel 1: RMSNorm + Global Max (for two-kernel approach with global_scale)
# =============================================================================


class RMSNormGlobalMaxKernel:
    """
    Kernel 1 of two-kernel approach: RMSNorm + Find Global Max.

    This kernel:
    1. Computes RMSNorm: y = x * rstd * w
    2. Stores y to intermediate buffer
    3. Atomically updates global_max with max(|y|)
    4. Signals dependent kernel via griddepcontrol.launch_dependents()
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        is_fp16: bool,
        sm_version: int | None = None,
    ):
        self.dtype = dtype
        self.H = H
        self.is_fp16 = is_fp16
        self.sm_version = sm_version if sm_version is not None else get_sm_version()

        # For simplicity, use cluster_n=1 for this kernel
        # The main overhead is memory bandwidth anyway
        self.cluster_n = 1
        self.H_per_cta = H

        # Compute thread configuration
        self.threads_per_row = self._compute_threads_per_row(self.H_per_cta)
        self.num_threads = self._compute_num_threads(self.H_per_cta)
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

        # Vectorization parameters
        elem_bytes = dtype.width // 8
        self.vec_size = COPY_BITS // 8 // elem_bytes
        self.num_vec_blocks = max(
            1,
            (self.H_per_cta // self.vec_size + self.threads_per_row - 1)
            // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

    @staticmethod
    def _compute_threads_per_row(H_per_cta: int) -> int:
        """Compute optimal threads per row based on H per CTA."""
        if H_per_cta <= 64:
            return 8
        elif H_per_cta <= 128:
            return 16
        elif H_per_cta <= 3072:
            return 32
        elif H_per_cta <= 6144:
            return 64
        elif H_per_cta <= 16384:
            return 128
        else:
            return 256

    @staticmethod
    def _compute_num_threads(H_per_cta: int) -> int:
        """Compute total threads per block based on H per CTA."""
        return 128 if H_per_cta <= 16384 else 256

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
        """Calculate shared memory requirement."""
        # Input tile for X
        tile_bytes = self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
        # Reduction buffer
        reduction_bytes = self.rows_per_block * self.warps_per_row * 4
        return tile_bytes + reduction_bytes

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        y_ptr: cute.Pointer,  # Output: RMSNorm result (same dtype as input)
        global_max_ptr: cute.Pointer,  # Output: global max (float32)
        M: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        """Host function to launch the kernel."""
        H = self.H

        mX = cute.make_tensor(
            x_ptr,
            layout=cute.make_ordered_layout((M, H), order=(1, 0)),
        )
        mW = cute.make_tensor(
            w_ptr,
            layout=cute.make_layout((H,)),
        )
        # Y has same layout as X (stores RMSNorm output in input dtype)
        mY = cute.make_tensor(
            y_ptr,
            layout=cute.make_ordered_layout((M, H), order=(1, 0)),
        )
        mGlobalMax = cute.make_tensor(
            global_max_ptr,
            layout=cute.make_layout((1,)),
        )

        self.kernel(mX, mW, mY, mGlobalMax, M, eps).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mGlobalMax: cute.Tensor,
        M: Int32,
        eps: Float32,
    ):
        """Device kernel: RMSNorm + store + global max."""
        bidx = cute.arch.block_idx()[0]
        tidx = cute.arch.thread_idx()[0]
        H = self.H
        is_fp16 = self.is_fp16
        rows_per_block = self.rows_per_block
        cols_per_tile = self.cols_per_tile
        threads_per_row = self.threads_per_row
        warps_per_row = self.warps_per_row
        vec_size = self.vec_size
        num_vec_blocks = self.num_vec_blocks

        lane_in_row = tidx % threads_per_row
        row_in_block = tidx // threads_per_row

        # Create layouts using compile-time constants
        tv_shape = (
            (threads_per_row, rows_per_block),
            (vec_size, num_vec_blocks),
        )
        tv_stride = (
            (vec_size * rows_per_block, 1),
            (rows_per_block, rows_per_block * vec_size * threads_per_row),
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (rows_per_block, cols_per_tile)

        # Allocate shared memory using static shapes
        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout((rows_per_block, cols_per_tile), order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, warps_per_row)),
            byte_alignment=4,
        )

        # Create identity tensor for coordinate tracking
        idX = cute.make_identity_tensor(mX.shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        # Create TiledCopy for async load
        copy_atom_load_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=COPY_BITS,
        )
        tiled_copy_load = cute.make_tiled_copy(
            copy_atom_load_async, tv_layout, tiler_mn
        )
        thr_copy_X = tiled_copy_load.get_slice(tidx)

        # Partition tensors
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXcX = thr_copy_X.partition_S(cX)

        # Register fragments
        tXrX = cute.make_fragment_like(tXgX)

        # Create predicate tensor
        tXpX = predicate_k(tXcX, limit=H)

        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # Phase 1: Load X to shared memory
        if row_in_bounds:
            cute.copy(copy_atom_load_async, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        # Phase 2: Compute sum of squares
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce(
            x_sq,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer,
            None,  # No mbarrier for cluster_n=1
            1,  # cluster_n
            Float32(0.0),
        )

        mean_sq = sum_sq / H
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        cute.arch.barrier()

        # Phase 3: Compute RMSNorm and store, tracking max_abs
        actual_row_idx = bidx * rows_per_block + row_in_block

        if actual_row_idx < M:
            # Get pointer to global_max for atomic updates
            global_max_addr = get_ptr_as_int64(mGlobalMax, Int32(0))

            # Track thread-local max
            thread_max_abs = Float32(0.0)

            # Process elements in blocks of vec_size (8 elements for FP16)
            num_vec_per_thread = (H + threads_per_row * self.vec_size - 1) // (
                threads_per_row * self.vec_size
            )

            for vec_iter in range(num_vec_per_thread):
                vec_idx = lane_in_row + vec_iter * threads_per_row
                col_start = vec_idx * self.vec_size

                if col_start < H:
                    # Load X and W vectors
                    x_ptr = get_ptr_as_int64(mX, actual_row_idx * H + col_start)
                    w_ptr_addr = get_ptr_as_int64(mW, col_start)

                    x0, x1, x2, x3 = ld_global_v4_u32(x_ptr)

                    w0, w1, w2, w3 = ld_global_v4_u32(w_ptr_addr)

                    # Compute y = x * rstd * w
                    if cutlass.const_expr(is_fp16):
                        # Half2 operations
                        y0_h2 = half2_mul(x0, w0)
                        y1_h2 = half2_mul(x1, w1)
                        y2_h2 = half2_mul(x2, w2)
                        y3_h2 = half2_mul(x3, w3)

                        # Scale by rstd and convert to float
                        y0, y1 = half2_to_float2_scaled(y0_h2, rstd)
                        y2, y3 = half2_to_float2_scaled(y1_h2, rstd)
                        y4, y5 = half2_to_float2_scaled(y2_h2, rstd)
                        y6, y7 = half2_to_float2_scaled(y3_h2, rstd)

                        # Track max abs
                        abs0 = fabs_f32(y0)
                        abs1 = fabs_f32(y1)
                        abs2 = fabs_f32(y2)
                        abs3 = fabs_f32(y3)
                        abs4 = fabs_f32(y4)
                        abs5 = fabs_f32(y5)
                        abs6 = fabs_f32(y6)
                        abs7 = fabs_f32(y7)

                        local_max = fmax_f32(abs0, abs1)
                        local_max = fmax_f32(local_max, abs2)
                        local_max = fmax_f32(local_max, abs3)
                        local_max = fmax_f32(local_max, abs4)
                        local_max = fmax_f32(local_max, abs5)
                        local_max = fmax_f32(local_max, abs6)
                        local_max = fmax_f32(local_max, abs7)
                        thread_max_abs = fmax_f32(thread_max_abs, local_max)

                        # Convert back to half and store
                        y_h2_0 = float2_to_half2(y0, y1)
                        y_h2_1 = float2_to_half2(y2, y3)
                        y_h2_2 = float2_to_half2(y4, y5)
                        y_h2_3 = float2_to_half2(y6, y7)

                        y_ptr = get_ptr_as_int64(mY, actual_row_idx * H + col_start)
                        st_global_v4_u32(y_ptr, y_h2_0, y_h2_1, y_h2_2, y_h2_3)
                    else:
                        # BFloat16 operations
                        y0_h2 = bfloat2_mul(x0, w0)
                        y1_h2 = bfloat2_mul(x1, w1)
                        y2_h2 = bfloat2_mul(x2, w2)
                        y3_h2 = bfloat2_mul(x3, w3)

                        y0, y1 = bfloat2_to_float2_scaled(y0_h2, rstd)
                        y2, y3 = bfloat2_to_float2_scaled(y1_h2, rstd)
                        y4, y5 = bfloat2_to_float2_scaled(y2_h2, rstd)
                        y6, y7 = bfloat2_to_float2_scaled(y3_h2, rstd)

                        abs0 = fabs_f32(y0)
                        abs1 = fabs_f32(y1)
                        abs2 = fabs_f32(y2)
                        abs3 = fabs_f32(y3)
                        abs4 = fabs_f32(y4)
                        abs5 = fabs_f32(y5)
                        abs6 = fabs_f32(y6)
                        abs7 = fabs_f32(y7)

                        local_max = fmax_f32(abs0, abs1)
                        local_max = fmax_f32(local_max, abs2)
                        local_max = fmax_f32(local_max, abs3)
                        local_max = fmax_f32(local_max, abs4)
                        local_max = fmax_f32(local_max, abs5)
                        local_max = fmax_f32(local_max, abs6)
                        local_max = fmax_f32(local_max, abs7)
                        thread_max_abs = fmax_f32(thread_max_abs, local_max)

                        y_h2_0 = float2_to_bfloat2(y0, y1)
                        y_h2_1 = float2_to_bfloat2(y2, y3)
                        y_h2_2 = float2_to_bfloat2(y4, y5)
                        y_h2_3 = float2_to_bfloat2(y6, y7)

                        y_ptr = get_ptr_as_int64(mY, actual_row_idx * H + col_start)
                        st_global_v4_u32(y_ptr, y_h2_0, y_h2_1, y_h2_2, y_h2_3)

            # Atomic update global max using uint32 reinterpretation
            # For positive floats, IEEE754 preserves ordering when treated as uint
            thread_max_abs_u32 = float_as_uint(thread_max_abs)
            _ = atomic_max_global_u32(global_max_addr, thread_max_abs_u32)

        # Signal dependent kernel
        cute.arch.barrier()
        if tidx == 0:
            griddepcontrol_launch_dependents()


# =============================================================================
# Kernel 2: Global Scale + Quantize (for two-kernel approach with global_scale)
# =============================================================================


class GlobalScaleQuantizeKernel:
    """
    Kernel 2 of two-kernel approach: Apply Global Scale + Quantize.

    This kernel:
    1. Waits for Kernel 1 via griddepcontrol.wait()
    2. Computes global_scale = FP8_MAX * FP4_MAX / global_max
    3. Loads RMSNorm output from buffer
    4. Applies global_scale and computes per-block scales
    5. Quantizes to FP4 and stores
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        block_size: int,
        output_swizzled: bool,
        is_fp16: bool,
        sm_version: int | None = None,
        scale_format: str | None = None,
    ):
        self.dtype = dtype
        self.H = H
        self.block_size = block_size
        self.output_swizzled = output_swizzled
        self.is_fp16 = is_fp16
        self.sm_version = sm_version if sm_version is not None else get_sm_version()

        if scale_format is None:
            self.scale_format = "ue8m0" if block_size == 32 else "e4m3"
        else:
            self.scale_format = scale_format

        assert block_size in (16, 32), f"block_size must be 16 or 32, got {block_size}"

        # Use simple configuration for this kernel
        self.cluster_n = 1
        self.threads_per_row = 32
        self.num_threads = 128
        self.rows_per_block = self.num_threads // self.threads_per_row

        self.num_sf_blocks_per_row = H // block_size

        if output_swizzled:
            num_col_vecs = H // block_size
            self.num_k_tiles = (num_col_vecs + 3) // 4
            self.k_tile_stride = 512

    def _smem_size_in_bytes(self) -> int:
        """Minimal shared memory for reduction."""
        return 128  # Small buffer for reductions if needed

    @cute.jit
    def __call__(
        self,
        y_in_ptr: cute.Pointer,  # Input: RMSNorm output (input dtype)
        global_max_ptr: cute.Pointer,  # Input: global max (float32)
        global_scale_ptr: cute.Pointer,  # Output: global scale (float32)
        y_fp4_ptr: cute.Pointer,  # Output: quantized FP4
        scale_ptr: cute.Pointer,  # Output: block scales
        M: Int32,
        stream: cuda.CUstream,
    ):
        """Host function to launch the kernel."""
        H = self.H

        mYIn = cute.make_tensor(
            y_in_ptr,
            layout=cute.make_ordered_layout((M, H), order=(1, 0)),
        )
        mGlobalMax = cute.make_tensor(
            global_max_ptr,
            layout=cute.make_layout((1,)),
        )
        mGlobalScale = cute.make_tensor(
            global_scale_ptr,
            layout=cute.make_layout((1,)),
        )
        mYFP4 = cute.make_tensor(
            y_fp4_ptr,
            layout=cute.make_ordered_layout((M, H // 2), order=(1, 0)),
        )

        if cutlass.const_expr(self.output_swizzled):
            num_m_tiles = (M + Int32(127)) // Int32(128)
            swizzled_size = num_m_tiles * Int32(self.num_k_tiles * self.k_tile_stride)
            mS = cute.make_tensor(
                scale_ptr,
                layout=cute.make_layout((swizzled_size,), stride=(Int32(1),)),
            )
        else:
            mS = cute.make_tensor(
                scale_ptr,
                layout=cute.make_ordered_layout(
                    (M, self.num_sf_blocks_per_row), order=(1, 0)
                ),
            )

        self.kernel(mYIn, mGlobalMax, mGlobalScale, mYFP4, mS, M).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mYIn: cute.Tensor,
        mGlobalMax: cute.Tensor,
        mGlobalScale: cute.Tensor,
        mYFP4: cute.Tensor,
        mS: cute.Tensor,
        M: Int32,
    ):
        """Device kernel: Apply global scale + quantize."""
        bidx = cute.arch.block_idx()[0]
        tidx = cute.arch.thread_idx()[0]
        H = self.H
        block_size = self.block_size
        num_sf_blocks_per_row = self.num_sf_blocks_per_row
        is_fp16 = self.is_fp16
        rows_per_block = self.rows_per_block
        threads_per_row = self.threads_per_row

        lane_in_row = tidx % threads_per_row
        row_in_block = tidx // threads_per_row

        # Wait for Kernel 1 to complete
        griddepcontrol_wait()

        # Thread 0 computes and stores global_scale
        global_max_addr = get_ptr_as_int64(mGlobalMax, Int32(0))
        global_scale_addr = get_ptr_as_int64(mGlobalScale, Int32(0))

        # Read global_max (stored as uint32 for atomic max compatibility)
        # Convert back to float - this works because we used float-as-uint for positive values
        global_max_val = ld_global_f32(global_max_addr)

        # Compute global_scale = FP8_MAX * FP4_MAX / global_max
        # Clamp to avoid division by zero
        fp8_fp4_product = Float32(FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
        global_max_clamped = fmax_f32(global_max_val, Float32(1e-12))
        global_scale = fp8_fp4_product * rcp_approx_ftz(global_max_clamped)

        # Store global_scale (only thread 0)
        if tidx == 0:
            st_global_f32(global_scale_addr, global_scale)

        # Precompute for quantization
        fp4_max_rcp = rcp_approx_ftz(Float32(FLOAT4_E2M1_MAX))

        actual_row_idx = bidx * rows_per_block + row_in_block

        if actual_row_idx < M:
            # Process scale factor blocks assigned to this thread
            num_sf_per_thread = (
                num_sf_blocks_per_row + threads_per_row - 1
            ) // threads_per_row

            for sf_iter in range(num_sf_per_thread):
                sf_idx = lane_in_row + sf_iter * threads_per_row

                if sf_idx < num_sf_blocks_per_row:
                    block_start = sf_idx * block_size

                    if cutlass.const_expr(block_size == 16):
                        # Load 16 elements from Y buffer
                        y_ptr0 = get_ptr_as_int64(
                            mYIn, actual_row_idx * H + block_start
                        )
                        y_ptr1 = get_ptr_as_int64(
                            mYIn, actual_row_idx * H + block_start + Int32(8)
                        )

                        y0, y1, y2, y3 = ld_global_v4_u32(y_ptr0)
                        y4, y5, y6, y7 = ld_global_v4_u32(y_ptr1)

                        # Convert to float and apply global_scale
                        if cutlass.const_expr(is_fp16):
                            v0, v1 = half2_to_float2_scaled(y0, global_scale)
                            v2, v3 = half2_to_float2_scaled(y1, global_scale)
                            v4, v5 = half2_to_float2_scaled(y2, global_scale)
                            v6, v7 = half2_to_float2_scaled(y3, global_scale)
                            v8, v9 = half2_to_float2_scaled(y4, global_scale)
                            v10, v11 = half2_to_float2_scaled(y5, global_scale)
                            v12, v13 = half2_to_float2_scaled(y6, global_scale)
                            v14, v15 = half2_to_float2_scaled(y7, global_scale)

                            # Max-abs for per-block scale
                            abs0_h2 = habs2(y0)
                            abs1_h2 = habs2(y1)
                            abs2_h2 = habs2(y2)
                            abs3_h2 = habs2(y3)
                            abs4_h2 = habs2(y4)
                            abs5_h2 = habs2(y5)
                            abs6_h2 = habs2(y6)
                            abs7_h2 = habs2(y7)

                            max01_h2 = hmax2(abs0_h2, abs1_h2)
                            max23_h2 = hmax2(abs2_h2, abs3_h2)
                            max45_h2 = hmax2(abs4_h2, abs5_h2)
                            max67_h2 = hmax2(abs6_h2, abs7_h2)
                            max0123_h2 = hmax2(max01_h2, max23_h2)
                            max4567_h2 = hmax2(max45_h2, max67_h2)
                            max_h2 = hmax2(max0123_h2, max4567_h2)
                            max_abs_block = hmax_to_f32(max_h2) * global_scale
                        else:
                            v0, v1 = bfloat2_to_float2_scaled(y0, global_scale)
                            v2, v3 = bfloat2_to_float2_scaled(y1, global_scale)
                            v4, v5 = bfloat2_to_float2_scaled(y2, global_scale)
                            v6, v7 = bfloat2_to_float2_scaled(y3, global_scale)
                            v8, v9 = bfloat2_to_float2_scaled(y4, global_scale)
                            v10, v11 = bfloat2_to_float2_scaled(y5, global_scale)
                            v12, v13 = bfloat2_to_float2_scaled(y6, global_scale)
                            v14, v15 = bfloat2_to_float2_scaled(y7, global_scale)

                            abs0_h2 = bfloat2_habs2(y0)
                            abs1_h2 = bfloat2_habs2(y1)
                            abs2_h2 = bfloat2_habs2(y2)
                            abs3_h2 = bfloat2_habs2(y3)
                            abs4_h2 = bfloat2_habs2(y4)
                            abs5_h2 = bfloat2_habs2(y5)
                            abs6_h2 = bfloat2_habs2(y6)
                            abs7_h2 = bfloat2_habs2(y7)

                            max01_h2 = bfloat2_hmax2(abs0_h2, abs1_h2)
                            max23_h2 = bfloat2_hmax2(abs2_h2, abs3_h2)
                            max45_h2 = bfloat2_hmax2(abs4_h2, abs5_h2)
                            max67_h2 = bfloat2_hmax2(abs6_h2, abs7_h2)
                            max0123_h2 = bfloat2_hmax2(max01_h2, max23_h2)
                            max4567_h2 = bfloat2_hmax2(max45_h2, max67_h2)
                            max_h2 = bfloat2_hmax2(max0123_h2, max4567_h2)
                            max_abs_block = bfloat2_hmax_to_f32(max_h2) * global_scale

                        # Compute per-block scale
                        scale_float = max_abs_block * fp4_max_rcp
                        scale_float = fmin_f32(scale_float, Float32(FLOAT8_E4M3_MAX))

                        scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
                        scale_fp8 = Uint8(scale_fp8_u32 & Uint32(0xFF))
                        inv_scale = fp8_e4m3_to_f32_and_rcp(scale_fp8_u32)

                        # Store scale
                        if cutlass.const_expr(self.output_swizzled):
                            inner_k_idx = sf_idx % Int32(4)
                            inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                            outer_m_idx = actual_row_idx % Int32(32)
                            k_tile_idx = sf_idx // Int32(4)
                            m_tile_idx = actual_row_idx // Int32(128)
                            m_tile_stride = self.num_k_tiles * self.k_tile_stride
                            swizzled_offset = (
                                m_tile_idx * m_tile_stride
                                + k_tile_idx * self.k_tile_stride
                                + outer_m_idx * Int32(16)
                                + inner_m_idx * Int32(4)
                                + inner_k_idx
                            )
                            mS[swizzled_offset] = scale_fp8
                        else:
                            mS[actual_row_idx, sf_idx] = scale_fp8

                        # Quantize
                        q0 = v0 * inv_scale
                        q1 = v1 * inv_scale
                        q2 = v2 * inv_scale
                        q3 = v3 * inv_scale
                        q4 = v4 * inv_scale
                        q5 = v5 * inv_scale
                        q6 = v6 * inv_scale
                        q7 = v7 * inv_scale
                        q8 = v8 * inv_scale
                        q9 = v9 * inv_scale
                        q10 = v10 * inv_scale
                        q11 = v11 * inv_scale
                        q12 = v12 * inv_scale
                        q13 = v13 * inv_scale
                        q14 = v14 * inv_scale
                        q15 = v15 * inv_scale

                        packed_lo = cvt_e2m1x8_f32(q0, q1, q2, q3, q4, q5, q6, q7)
                        packed_hi = cvt_e2m1x8_f32(q8, q9, q10, q11, q12, q13, q14, q15)
                        packed64 = (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)

                        out_offset = block_start // 2
                        out_ptr = get_ptr_as_int64(
                            mYFP4, actual_row_idx * (H // 2) + out_offset
                        )
                        st_global_u64(out_ptr, packed64)

                    else:
                        # block_size == 32: Process in 2 chunks
                        # Similar to block_size==16 but with 32 elements
                        # For brevity, handle this case with UE8M0 scale format
                        y_ptr0_c0 = get_ptr_as_int64(
                            mYIn, actual_row_idx * H + block_start
                        )
                        y_ptr1_c0 = get_ptr_as_int64(
                            mYIn, actual_row_idx * H + block_start + Int32(8)
                        )
                        y_ptr0_c1 = get_ptr_as_int64(
                            mYIn, actual_row_idx * H + block_start + Int32(16)
                        )
                        y_ptr1_c1 = get_ptr_as_int64(
                            mYIn, actual_row_idx * H + block_start + Int32(24)
                        )

                        y0_c0, y1_c0, y2_c0, y3_c0 = ld_global_v4_u32(y_ptr0_c0)
                        y4_c0, y5_c0, y6_c0, y7_c0 = ld_global_v4_u32(y_ptr1_c0)
                        y0_c1, y1_c1, y2_c1, y3_c1 = ld_global_v4_u32(y_ptr0_c1)
                        y4_c1, y5_c1, y6_c1, y7_c1 = ld_global_v4_u32(y_ptr1_c1)

                        if cutlass.const_expr(is_fp16):
                            # Chunk 0
                            v0_c0, v1_c0 = half2_to_float2_scaled(y0_c0, global_scale)
                            v2_c0, v3_c0 = half2_to_float2_scaled(y1_c0, global_scale)
                            v4_c0, v5_c0 = half2_to_float2_scaled(y2_c0, global_scale)
                            v6_c0, v7_c0 = half2_to_float2_scaled(y3_c0, global_scale)
                            v8_c0, v9_c0 = half2_to_float2_scaled(y4_c0, global_scale)
                            v10_c0, v11_c0 = half2_to_float2_scaled(y5_c0, global_scale)
                            v12_c0, v13_c0 = half2_to_float2_scaled(y6_c0, global_scale)
                            v14_c0, v15_c0 = half2_to_float2_scaled(y7_c0, global_scale)

                            # Chunk 1
                            v0_c1, v1_c1 = half2_to_float2_scaled(y0_c1, global_scale)
                            v2_c1, v3_c1 = half2_to_float2_scaled(y1_c1, global_scale)
                            v4_c1, v5_c1 = half2_to_float2_scaled(y2_c1, global_scale)
                            v6_c1, v7_c1 = half2_to_float2_scaled(y3_c1, global_scale)
                            v8_c1, v9_c1 = half2_to_float2_scaled(y4_c1, global_scale)
                            v10_c1, v11_c1 = half2_to_float2_scaled(y5_c1, global_scale)
                            v12_c1, v13_c1 = half2_to_float2_scaled(y6_c1, global_scale)
                            v14_c1, v15_c1 = half2_to_float2_scaled(y7_c1, global_scale)

                            # Max-abs across both chunks
                            abs0_c0 = habs2(y0_c0)
                            abs1_c0 = habs2(y1_c0)
                            abs2_c0 = habs2(y2_c0)
                            abs3_c0 = habs2(y3_c0)
                            abs4_c0 = habs2(y4_c0)
                            abs5_c0 = habs2(y5_c0)
                            abs6_c0 = habs2(y6_c0)
                            abs7_c0 = habs2(y7_c0)

                            abs0_c1 = habs2(y0_c1)
                            abs1_c1 = habs2(y1_c1)
                            abs2_c1 = habs2(y2_c1)
                            abs3_c1 = habs2(y3_c1)
                            abs4_c1 = habs2(y4_c1)
                            abs5_c1 = habs2(y5_c1)
                            abs6_c1 = habs2(y6_c1)
                            abs7_c1 = habs2(y7_c1)

                            max_c0 = hmax2(
                                hmax2(
                                    hmax2(
                                        hmax2(abs0_c0, abs1_c0), hmax2(abs2_c0, abs3_c0)
                                    ),
                                    hmax2(
                                        hmax2(abs4_c0, abs5_c0), hmax2(abs6_c0, abs7_c0)
                                    ),
                                ),
                                hmax2(
                                    hmax2(
                                        hmax2(abs0_c1, abs1_c1), hmax2(abs2_c1, abs3_c1)
                                    ),
                                    hmax2(
                                        hmax2(abs4_c1, abs5_c1), hmax2(abs6_c1, abs7_c1)
                                    ),
                                ),
                            )
                            max_abs_block = hmax_to_f32(max_c0) * global_scale
                        else:
                            # BFloat16
                            v0_c0, v1_c0 = bfloat2_to_float2_scaled(y0_c0, global_scale)
                            v2_c0, v3_c0 = bfloat2_to_float2_scaled(y1_c0, global_scale)
                            v4_c0, v5_c0 = bfloat2_to_float2_scaled(y2_c0, global_scale)
                            v6_c0, v7_c0 = bfloat2_to_float2_scaled(y3_c0, global_scale)
                            v8_c0, v9_c0 = bfloat2_to_float2_scaled(y4_c0, global_scale)
                            v10_c0, v11_c0 = bfloat2_to_float2_scaled(
                                y5_c0, global_scale
                            )
                            v12_c0, v13_c0 = bfloat2_to_float2_scaled(
                                y6_c0, global_scale
                            )
                            v14_c0, v15_c0 = bfloat2_to_float2_scaled(
                                y7_c0, global_scale
                            )

                            v0_c1, v1_c1 = bfloat2_to_float2_scaled(y0_c1, global_scale)
                            v2_c1, v3_c1 = bfloat2_to_float2_scaled(y1_c1, global_scale)
                            v4_c1, v5_c1 = bfloat2_to_float2_scaled(y2_c1, global_scale)
                            v6_c1, v7_c1 = bfloat2_to_float2_scaled(y3_c1, global_scale)
                            v8_c1, v9_c1 = bfloat2_to_float2_scaled(y4_c1, global_scale)
                            v10_c1, v11_c1 = bfloat2_to_float2_scaled(
                                y5_c1, global_scale
                            )
                            v12_c1, v13_c1 = bfloat2_to_float2_scaled(
                                y6_c1, global_scale
                            )
                            v14_c1, v15_c1 = bfloat2_to_float2_scaled(
                                y7_c1, global_scale
                            )

                            abs0_c0 = bfloat2_habs2(y0_c0)
                            abs1_c0 = bfloat2_habs2(y1_c0)
                            abs2_c0 = bfloat2_habs2(y2_c0)
                            abs3_c0 = bfloat2_habs2(y3_c0)
                            abs4_c0 = bfloat2_habs2(y4_c0)
                            abs5_c0 = bfloat2_habs2(y5_c0)
                            abs6_c0 = bfloat2_habs2(y6_c0)
                            abs7_c0 = bfloat2_habs2(y7_c0)

                            abs0_c1 = bfloat2_habs2(y0_c1)
                            abs1_c1 = bfloat2_habs2(y1_c1)
                            abs2_c1 = bfloat2_habs2(y2_c1)
                            abs3_c1 = bfloat2_habs2(y3_c1)
                            abs4_c1 = bfloat2_habs2(y4_c1)
                            abs5_c1 = bfloat2_habs2(y5_c1)
                            abs6_c1 = bfloat2_habs2(y6_c1)
                            abs7_c1 = bfloat2_habs2(y7_c1)

                            max_c0 = bfloat2_hmax2(
                                bfloat2_hmax2(
                                    bfloat2_hmax2(
                                        bfloat2_hmax2(abs0_c0, abs1_c0),
                                        bfloat2_hmax2(abs2_c0, abs3_c0),
                                    ),
                                    bfloat2_hmax2(
                                        bfloat2_hmax2(abs4_c0, abs5_c0),
                                        bfloat2_hmax2(abs6_c0, abs7_c0),
                                    ),
                                ),
                                bfloat2_hmax2(
                                    bfloat2_hmax2(
                                        bfloat2_hmax2(abs0_c1, abs1_c1),
                                        bfloat2_hmax2(abs2_c1, abs3_c1),
                                    ),
                                    bfloat2_hmax2(
                                        bfloat2_hmax2(abs4_c1, abs5_c1),
                                        bfloat2_hmax2(abs6_c1, abs7_c1),
                                    ),
                                ),
                            )
                            max_abs_block = bfloat2_hmax_to_f32(max_c0) * global_scale

                        # Compute UE8M0 scale
                        scale_float = max_abs_block * fp4_max_rcp
                        scale_ue8m0 = cvt_f32_to_ue8m0(scale_float)
                        scale_exp = Uint8(scale_ue8m0 & Uint32(0xFF))

                        # Store scale
                        if cutlass.const_expr(self.output_swizzled):
                            inner_k_idx = sf_idx % Int32(4)
                            inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                            outer_m_idx = actual_row_idx % Int32(32)
                            k_tile_idx = sf_idx // Int32(4)
                            m_tile_idx = actual_row_idx // Int32(128)
                            m_tile_stride = self.num_k_tiles * self.k_tile_stride
                            swizzled_offset = (
                                m_tile_idx * m_tile_stride
                                + k_tile_idx * self.k_tile_stride
                                + outer_m_idx * Int32(16)
                                + inner_m_idx * Int32(4)
                                + inner_k_idx
                            )
                            mS[swizzled_offset] = scale_exp
                        else:
                            mS[actual_row_idx, sf_idx] = scale_exp

                        inv_scale = ue8m0_to_output_scale(scale_ue8m0)

                        # Quantize chunk 0
                        q0 = v0_c0 * inv_scale
                        q1 = v1_c0 * inv_scale
                        q2 = v2_c0 * inv_scale
                        q3 = v3_c0 * inv_scale
                        q4 = v4_c0 * inv_scale
                        q5 = v5_c0 * inv_scale
                        q6 = v6_c0 * inv_scale
                        q7 = v7_c0 * inv_scale
                        q8 = v8_c0 * inv_scale
                        q9 = v9_c0 * inv_scale
                        q10 = v10_c0 * inv_scale
                        q11 = v11_c0 * inv_scale
                        q12 = v12_c0 * inv_scale
                        q13 = v13_c0 * inv_scale
                        q14 = v14_c0 * inv_scale
                        q15 = v15_c0 * inv_scale

                        packed_lo = cvt_e2m1x8_f32(q0, q1, q2, q3, q4, q5, q6, q7)
                        packed_hi = cvt_e2m1x8_f32(q8, q9, q10, q11, q12, q13, q14, q15)
                        packed64 = (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)
                        out_offset = block_start // 2
                        out_ptr = get_ptr_as_int64(
                            mYFP4, actual_row_idx * (H // 2) + out_offset
                        )
                        st_global_u64(out_ptr, packed64)

                        # Quantize chunk 1
                        q0 = v0_c1 * inv_scale
                        q1 = v1_c1 * inv_scale
                        q2 = v2_c1 * inv_scale
                        q3 = v3_c1 * inv_scale
                        q4 = v4_c1 * inv_scale
                        q5 = v5_c1 * inv_scale
                        q6 = v6_c1 * inv_scale
                        q7 = v7_c1 * inv_scale
                        q8 = v8_c1 * inv_scale
                        q9 = v9_c1 * inv_scale
                        q10 = v10_c1 * inv_scale
                        q11 = v11_c1 * inv_scale
                        q12 = v12_c1 * inv_scale
                        q13 = v13_c1 * inv_scale
                        q14 = v14_c1 * inv_scale
                        q15 = v15_c1 * inv_scale

                        packed_lo = cvt_e2m1x8_f32(q0, q1, q2, q3, q4, q5, q6, q7)
                        packed_hi = cvt_e2m1x8_f32(q8, q9, q10, q11, q12, q13, q14, q15)
                        packed64 = (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)
                        out_offset = (block_start + 16) // 2
                        out_ptr = get_ptr_as_int64(
                            mYFP4, actual_row_idx * (H // 2) + out_offset
                        )
                        st_global_u64(out_ptr, packed64)


# =============================================================================
# PyTorch API Functions - Streamlined with Pointer-based Compilation
# =============================================================================


@functools.cache
def _get_compiled_kernel(
    hidden_size: int,
    block_size: int,
    is_fp16: bool,
    sm_version: int,
    scale_format: str,
    is_sf_swizzled_layout: bool,
) -> Callable:
    """
    Get a compiled kernel closure that takes torch.Tensor directly.

    """
    cutlass_dtype = cutlass.Float16 if is_fp16 else cutlass.BFloat16

    def get_cute_pointers(tensors):
        """Convert torch tensors to cute pointers, or create dummy pointers for compilation."""
        if tensors is None:
            # Dummy pointers for compilation - just need alignment
            return [
                make_ptr(
                    cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
                ),  # x
                make_ptr(
                    cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
                ),  # w
                make_ptr(
                    cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
                ),  # y
                make_ptr(
                    cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
                ),  # s
            ]
        x, w, y, s = tensors
        return [
            make_ptr(
                cutlass_dtype, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass_dtype, w.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Uint8, y.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Uint8, s.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
        ]

    # Create kernel instance
    kernel_obj = RMSNormFP4QuantKernel(
        dtype=cutlass_dtype,
        H=hidden_size,
        block_size=block_size,
        output_swizzled=is_sf_swizzled_layout,
        is_fp16=is_fp16,
        sm_version=sm_version,
        scale_format=scale_format,
    )

    # Compile with dummy pointers
    compiled_kernel = cute.compile(
        kernel_obj,
        *get_cute_pointers(None),
        Int32(1),  # Dummy M
        Float32(1e-6),  # Dummy eps
        cutlass_torch.current_stream(),
    )

    def tensor_api(
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
        s: torch.Tensor,
        M: int,
        eps: float,
    ) -> None:
        """Runtime API that converts tensors to pointers and calls the kernel."""
        nonlocal compiled_kernel
        compiled_kernel(
            *get_cute_pointers([x, w, y, s]),
            Int32(M),
            Float32(eps),
            cutlass_torch.current_stream(),
        )

    return tensor_api


@functools.cache
def _get_compiled_kernels_with_global_scale(
    hidden_size: int,
    block_size: int,
    is_fp16: bool,
    sm_version: int,
    scale_format: str,
    is_sf_swizzled_layout: bool,
) -> Tuple[Callable, Callable]:
    """
    Get compiled kernel closures for two-kernel approach with global_scale.

    Returns a tuple of (kernel1_api, kernel2_api):
    - kernel1: RMSNorm + find global max
    - kernel2: Apply global scale + quantize
    """
    cutlass_dtype = cutlass.Float16 if is_fp16 else cutlass.BFloat16

    # Kernel 1: RMSNorm + Global Max
    def get_kernel1_pointers(tensors):
        if tensors is None:
            return [
                make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
            ]
        x, w, y, global_max = tensors
        return [
            make_ptr(
                cutlass_dtype, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass_dtype, w.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass_dtype, y.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Float32,
                global_max.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
        ]

    kernel1_obj = RMSNormGlobalMaxKernel(
        dtype=cutlass_dtype,
        H=hidden_size,
        is_fp16=is_fp16,
        sm_version=sm_version,
    )

    compiled_kernel1 = cute.compile(
        kernel1_obj,
        *get_kernel1_pointers(None),
        Int32(1),
        Float32(1e-6),
        cutlass_torch.current_stream(),
    )

    def kernel1_api(
        x: torch.Tensor,
        w: torch.Tensor,
        y_buffer: torch.Tensor,
        global_max: torch.Tensor,
        M: int,
        eps: float,
    ) -> None:
        compiled_kernel1(
            *get_kernel1_pointers([x, w, y_buffer, global_max]),
            Int32(M),
            Float32(eps),
            cutlass_torch.current_stream(),
        )

    # Kernel 2: Global Scale + Quantize
    def get_kernel2_pointers(tensors):
        if tensors is None:
            return [
                make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
                make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
                make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
            ]
        y_in, global_max, global_scale, y_fp4, scale = tensors
        return [
            make_ptr(
                cutlass_dtype, y_in.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Float32,
                global_max.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                cutlass.Float32,
                global_scale.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                cutlass.Uint8,
                y_fp4.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Uint8,
                scale.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
        ]

    kernel2_obj = GlobalScaleQuantizeKernel(
        dtype=cutlass_dtype,
        H=hidden_size,
        block_size=block_size,
        output_swizzled=is_sf_swizzled_layout,
        is_fp16=is_fp16,
        sm_version=sm_version,
        scale_format=scale_format,
    )

    compiled_kernel2 = cute.compile(
        kernel2_obj,
        *get_kernel2_pointers(None),
        Int32(1),
        cutlass_torch.current_stream(),
    )

    def kernel2_api(
        y_buffer: torch.Tensor,
        global_max: torch.Tensor,
        global_scale: torch.Tensor,
        y_fp4: torch.Tensor,
        block_scale: torch.Tensor,
        M: int,
    ) -> None:
        compiled_kernel2(
            *get_kernel2_pointers(
                [y_buffer, global_max, global_scale, y_fp4, block_scale]
            ),
            Int32(M),
            cutlass_torch.current_stream(),
        )

    return kernel1_api, kernel2_api


@flashinfer_api
def rmsnorm_mxfp4quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    y_fp4: torch.Tensor | None = None,
    block_scale: torch.Tensor | None = None,
    eps: float = 1e-6,
    is_sf_swizzled_layout: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused RMS normalization with MXFP4 quantization using CuTe-DSL.

    Computes: ``y = RMSNorm(input) * weight``, then quantizes ``y`` to FP4
    with per-block UE8M0 (power-of-2) scale factors.

    This uses a single-kernel approach optimized for MXFP4 format which uses
    block_size=32 and UE8M0 scale factors. No global scale is needed for MXFP4.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, shape ``(batch_size, hidden_size)`` or ``(batch_size, seq_len, hidden_size)``.
        Must be ``torch.float16`` or ``torch.bfloat16``.
        ``hidden_size`` must be divisible by 32.
    weight : torch.Tensor
        Weight tensor for RMSNorm, shape ``(hidden_size,)``.
        Must have the same dtype as input.
    y_fp4 : torch.Tensor, optional
        Output tensor for quantized values in FP4_E2M1 format, packed as uint8.
        Two FP4 values are packed into each uint8 byte.
        Shape must be ``(batch_size, hidden_size // 2)`` or matching 3D input.
        If ``None``, will be allocated automatically.
    block_scale : torch.Tensor, optional
        Output tensor for per-block UE8M0 scale factors.

        - If ``is_sf_swizzled_layout=False`` (default): row-major layout with shape
          ``(batch_size, hidden_size // 32)`` or matching 3D input.
        - If ``is_sf_swizzled_layout=True``: swizzled layout for efficient tensor core
          access.

        Dtype should be ``torch.uint8`` for UE8M0 format.
        If ``None``, will be allocated automatically.
    eps : float
        Epsilon for numerical stability in RMSNorm. Default is ``1e-6``.
    is_sf_swizzled_layout : bool
        If ``True``, output scale factors in swizzled layout optimized for
        tensor core GEMM operations. Default is ``False`` (row-major layout).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple of ``(y_fp4, block_scale)``:

        - ``y_fp4``: Quantized FP4 values packed as uint8.
        - ``block_scale``: Per-block UE8M0 scale factors as uint8.

    Notes
    -----
    - Requires SM100+ (Blackwell) for FP4 quantization PTX intrinsics.
    - Uses MXFP4 format: block_size=32, UE8M0 (power-of-2) scale factors.
    - FP4 E2M1 format has a max representable value of 6.0.
    - Single-kernel approach for optimal performance.
    """
    block_size = 32  # Fixed for MXFP4
    scale_format = "ue8m0"

    # Handle 2D vs 3D input
    is_3d = input.dim() == 3
    if is_3d:
        B, S, H = input.shape
        input_2d = input.view(B * S, H).contiguous()
    else:
        input_2d = input

    batch_size, hidden_size = input_2d.shape
    dtype = input.dtype

    assert hidden_size % block_size == 0, (
        "hidden_size must be divisible by 32 for MXFP4"
    )
    assert hidden_size >= 64, "hidden_size must be >= 64"

    # Determine data type
    is_fp16 = dtype == torch.float16
    sm_version = get_sm_version(input.device)

    # Allocate output tensors if not provided
    if y_fp4 is None:
        if is_3d:
            y_fp4 = torch.empty(
                (B, S, hidden_size // 2), dtype=torch.uint8, device=input.device
            )
        else:
            y_fp4 = torch.empty(
                (batch_size, hidden_size // 2), dtype=torch.uint8, device=input.device
            )

    if block_scale is None:
        num_sf_blocks_per_row = hidden_size // block_size

        if is_sf_swizzled_layout:
            num_m_tiles = (batch_size + 127) // 128
            num_k_tiles = (num_sf_blocks_per_row + 3) // 4
            k_tile_stride = 512
            swizzled_size = num_m_tiles * num_k_tiles * k_tile_stride
            block_scale = torch.empty(
                (swizzled_size,), dtype=torch.uint8, device=input.device
            )
        else:
            if is_3d:
                block_scale = torch.empty(
                    (B, S, num_sf_blocks_per_row),
                    dtype=torch.uint8,
                    device=input.device,
                )
            else:
                block_scale = torch.empty(
                    (batch_size, num_sf_blocks_per_row),
                    dtype=torch.uint8,
                    device=input.device,
                )

    # Get 2D views for kernel
    if is_3d:
        y_fp4_2d = y_fp4.view(B * S, -1)
        block_scale_2d = (
            block_scale.view(B * S, -1) if not is_sf_swizzled_layout else block_scale
        )
    else:
        y_fp4_2d = y_fp4
        block_scale_2d = block_scale

    # Get compiled kernel (original 1-kernel approach)
    kernel_api = _get_compiled_kernel(
        hidden_size,
        block_size,
        is_fp16,
        sm_version,
        scale_format,
        is_sf_swizzled_layout,
    )

    # Launch kernel
    kernel_api(
        input_2d.contiguous(),
        weight.contiguous(),
        y_fp4_2d,
        block_scale_2d.view(torch.uint8),
        batch_size,
        eps,
    )

    return y_fp4, block_scale


@flashinfer_api
def rmsnorm_nvfp4quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    y_fp4: torch.Tensor | None = None,
    block_scale: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
    eps: float = 1e-6,
    is_sf_swizzled_layout: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused RMS normalization with NVFP4 quantization using CuTe-DSL.

    Computes: ``y = RMSNorm(input) * weight``, applies global scaling,
    then quantizes ``y`` to FP4 with per-block E4M3 scale factors.

    This uses a two-kernel approach with PDL (Programmatic Dependent Launch):
    1. Pass 1: Compute RMSNorm, find global max across all elements
    2. Pass 2: Apply global scale, compute per-block scales, quantize to FP4

    The global scale lifts the per-block E4M3 scales into their optimal dynamic range.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, shape ``(batch_size, hidden_size)`` or ``(batch_size, seq_len, hidden_size)``.
        Must be ``torch.float16`` or ``torch.bfloat16``.
        ``hidden_size`` must be divisible by 16.
    weight : torch.Tensor
        Weight tensor for RMSNorm, shape ``(hidden_size,)``.
        Must have the same dtype as input.
    y_fp4 : torch.Tensor, optional
        Output tensor for quantized values in FP4_E2M1 format, packed as uint8.
        Two FP4 values are packed into each uint8 byte.
        Shape must be ``(batch_size, hidden_size // 2)`` or matching 3D input.
        If ``None``, will be allocated automatically.
    block_scale : torch.Tensor, optional
        Output tensor for per-block E4M3 scale factors.

        - If ``is_sf_swizzled_layout=False`` (default): row-major layout with shape
          ``(batch_size, hidden_size // 16)`` or matching 3D input.
        - If ``is_sf_swizzled_layout=True``: swizzled layout for efficient tensor core
          access.

        Dtype should be ``torch.float8_e4m3fn`` for E4M3 format.
        If ``None``, will be allocated automatically.
    global_scale : torch.Tensor, optional
        Output tensor for the global scale factor, shape ``(1,)`` with dtype ``torch.float32``.
        The global scale is computed as ``FP8_MAX * FP4_MAX / max_abs(rmsnorm_output)``.
        If ``None``, will be allocated automatically.
    eps : float
        Epsilon for numerical stability in RMSNorm. Default is ``1e-6``.
    is_sf_swizzled_layout : bool
        If ``True``, output scale factors in swizzled layout optimized for
        tensor core GEMM operations. Default is ``False`` (row-major layout).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple of ``(y_fp4, block_scale, global_scale)``:

        - ``y_fp4``: Quantized FP4 values packed as uint8.
        - ``block_scale``: Per-block E4M3 scale factors.
        - ``global_scale``: Global scale factor as a scalar tensor of shape ``(1,)``.

    Notes
    -----
    - Requires SM100+ (Blackwell) for FP4 quantization PTX intrinsics.
    - Uses NVFP4 format: block_size=16, E4M3 scale factors (max value 448.0).
    - FP4 E2M1 format has a max representable value of 6.0.
    - Uses PDL for efficient kernel pipelining between the two passes.
    - Global scale = FP8_MAX * FP4_MAX / max_abs = 448.0 * 6.0 / max_abs
    """
    block_size = 16  # Fixed for NVFP4
    scale_format = "e4m3"

    # Handle 2D vs 3D input
    is_3d = input.dim() == 3
    if is_3d:
        B, S, H = input.shape
        input_2d = input.view(B * S, H).contiguous()
    else:
        input_2d = input

    batch_size, hidden_size = input_2d.shape
    dtype = input.dtype

    assert hidden_size % block_size == 0, (
        "hidden_size must be divisible by 16 for NVFP4"
    )
    assert hidden_size >= 64, "hidden_size must be >= 64"

    # Determine data type
    is_fp16 = dtype == torch.float16
    sm_version = get_sm_version(input.device)

    # Allocate output tensors if not provided
    if y_fp4 is None:
        if is_3d:
            y_fp4 = torch.empty(
                (B, S, hidden_size // 2), dtype=torch.uint8, device=input.device
            )
        else:
            y_fp4 = torch.empty(
                (batch_size, hidden_size // 2), dtype=torch.uint8, device=input.device
            )

    if block_scale is None:
        num_sf_blocks_per_row = hidden_size // block_size

        if is_sf_swizzled_layout:
            num_m_tiles = (batch_size + 127) // 128
            num_k_tiles = (num_sf_blocks_per_row + 3) // 4
            k_tile_stride = 512
            swizzled_size = num_m_tiles * num_k_tiles * k_tile_stride
            block_scale = torch.empty(
                (swizzled_size,), dtype=torch.float8_e4m3fn, device=input.device
            )
        else:
            if is_3d:
                block_scale = torch.empty(
                    (B, S, num_sf_blocks_per_row),
                    dtype=torch.float8_e4m3fn,
                    device=input.device,
                )
            else:
                block_scale = torch.empty(
                    (batch_size, num_sf_blocks_per_row),
                    dtype=torch.float8_e4m3fn,
                    device=input.device,
                )

    if global_scale is None:
        global_scale = torch.empty(1, dtype=torch.float32, device=input.device)

    # Get 2D views for kernel
    if is_3d:
        y_fp4_2d = y_fp4.view(B * S, -1)
        block_scale_2d = (
            block_scale.view(B * S, -1) if not is_sf_swizzled_layout else block_scale
        )
    else:
        y_fp4_2d = y_fp4
        block_scale_2d = block_scale

    # Allocate intermediate buffer for RMSNorm output (same dtype as input)
    y_buffer = torch.empty_like(input_2d)

    # Allocate global_max (initialized to 0 for atomic max)
    global_max = torch.zeros(1, dtype=torch.float32, device=input.device)

    # Get compiled kernels (2-kernel approach with global scale)
    kernel1_api, kernel2_api = _get_compiled_kernels_with_global_scale(
        hidden_size,
        block_size,
        is_fp16,
        sm_version,
        scale_format,
        is_sf_swizzled_layout,
    )

    # Launch kernel 1: RMSNorm + find global max
    kernel1_api(
        input_2d.contiguous(),
        weight.contiguous(),
        y_buffer,
        global_max,
        batch_size,
        eps,
    )

    # Launch kernel 2: Apply global scale + quantize
    kernel2_api(
        y_buffer,
        global_max,
        global_scale,
        y_fp4_2d,
        block_scale_2d.view(torch.uint8),
        batch_size,
    )

    return y_fp4, block_scale, global_scale


# Keep rmsnorm_fp4quant as a convenience wrapper that routes to the appropriate implementation
@flashinfer_api
def rmsnorm_fp4quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    y_fp4: torch.Tensor | None = None,
    block_scale: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
    eps: float = 1e-6,
    block_size: int = 16,
    scale_format: str | None = None,
    is_sf_swizzled_layout: bool = False,
) -> (
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]
):
    """
    Fused RMS normalization with FP4 quantization using CuTe-DSL.

    This is a convenience wrapper that routes to the appropriate implementation:
    - ``block_size=16`` (NVFP4): Uses :func:`rmsnorm_nvfp4quant` with global scale
    - ``block_size=32`` (MXFP4): Uses :func:`rmsnorm_mxfp4quant` without global scale

    For new code, prefer using the specific APIs directly:
    - :func:`rmsnorm_nvfp4quant` for NVFP4 format
    - :func:`rmsnorm_mxfp4quant` for MXFP4 format

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, shape ``(batch_size, hidden_size)`` or ``(batch_size, seq_len, hidden_size)``.
    weight : torch.Tensor
        Weight tensor for RMSNorm, shape ``(hidden_size,)``.
    y_fp4 : torch.Tensor, optional
        Output tensor for quantized FP4 values. If ``None``, allocated automatically.
    block_scale : torch.Tensor, optional
        Output tensor for per-block scale factors. If ``None``, allocated automatically.
    global_scale : torch.Tensor, optional
        Output tensor for global scale (NVFP4 only). If ``None``, allocated automatically.
    eps : float
        Epsilon for numerical stability in RMSNorm. Default is ``1e-6``.
    block_size : int
        Quantization block size: ``16`` for NVFP4, ``32`` for MXFP4. Default is ``16``.
    scale_format : str, optional
        Scale factor format (auto-selected based on block_size if not specified).
    is_sf_swizzled_layout : bool
        Whether to use swizzled scale factor layout. Default is ``False``.

    Returns
    -------
    Tuple
        For NVFP4 (block_size=16): ``(y_fp4, block_scale, global_scale)``
        For MXFP4 (block_size=32): ``(y_fp4, block_scale)``
    """
    if block_size == 32:
        # MXFP4 - use 1-kernel approach, no global scale
        y_fp4_out, block_scale_out = rmsnorm_mxfp4quant(
            input, weight, y_fp4, block_scale, eps, is_sf_swizzled_layout
        )
        # For backwards compatibility, return a dummy global_scale if caller expects it
        if global_scale is not None:
            global_scale.fill_(1.0)
            return y_fp4_out, block_scale_out, global_scale
        return y_fp4_out, block_scale_out
    else:
        # NVFP4 - use 2-kernel approach with global scale
        return rmsnorm_nvfp4quant(
            input, weight, y_fp4, block_scale, global_scale, eps, is_sf_swizzled_layout
        )


__all__ = [
    "RMSNormFP4QuantKernel",
    "RMSNormGlobalMaxKernel",
    "GlobalScaleQuantizeKernel",
    "get_sm_version",
    "rmsnorm_fp4quant",
    "rmsnorm_mxfp4quant",
    "rmsnorm_nvfp4quant",
]
