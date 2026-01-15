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
"""

import ctypes
import functools
import importlib.util
from typing import Optional, Union

import cutlass
import cutlass._mlir.dialects.cute as _cute_ir
import cutlass.cute as cute
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.typing import AddressSpace, Numeric, Pointer, Type
from cutlass.cutlass_dsl import T, dsl_user_op


def is_cute_dsl_available() -> bool:
    return (
        importlib.util.find_spec("cutlass") is not None
        and importlib.util.find_spec("cutlass.cute") is not None
    )


def get_cutlass_dtype(dtype: str) -> cutlass.dtype:
    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float32": cutlass.Float32,
        "float8_e5m2": cutlass.Float8E5M2,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
        "float8_e8m0fnu": cutlass.Float8E8M0FNU,
        "float4_e2m1fn": cutlass.Float4E2M1FN,
    }
    return dtype_map[dtype]


def cutlass_to_torch_dtype(cutlass_dtype):
    """
    Return the corresponding torch.dtype per the given DSL type
    """
    torch_dtype = getattr(torch, cutlass_dtype.__name__.lower(), None)

    torch_type_map = {
        cutlass.TFloat32: torch.float32,
        cutlass.Float32: torch.float32,
        cutlass.Float16: torch.float16,
        cutlass.BFloat16: torch.bfloat16,
        cutlass.Float8E5M2: torch.float8_e5m2,
        cutlass.Float8E4M3FN: torch.float8_e4m3fn,
        cutlass.Float8E4M3B11FNUZ: torch.float8_e4m3fnuz,
    }
    if torch_dtype is None:
        torch_dtype = torch_type_map.get(cutlass_dtype)

    if torch_dtype is None:
        raise TypeError(f"{cutlass_dtype} is not supported by torch")
    return torch_dtype


@functools.cache
def get_num_sm(device: torch.device) -> int:
    # get the compute capability of the device, which would be cached
    return torch.cuda.get_device_properties(device).multi_processor_count


# WAR for CuTeDSL make_ptr implementation for flashinfer
class _Pointer(Pointer):
    """Runtime representation of a pointer that can inter-operate with
    various data structures, including numpy arrays and device memory.

    :param pointer: The pointer to the data
    :type pointer: int or pointer-like object
    :param dtype: Data type of the elements pointed to
    :type dtype: Type
    :param mem_space: Memory space where the pointer resides, defaults generic
    :type mem_space: _cute_ir.AddressSpace, optional
    :param assumed_align: Alignment of input pointer in bytes, defaults None
    :type assumed_align: int, optional

    :ivar _pointer: The underlying pointer
    :ivar _dtype: Data type of the elements
    :ivar _addr_space: Memory space of the pointer
    :ivar _assumed_align: Alignment of the pointer in bytes
    :ivar _desc: C-type descriptor for the pointer
    :ivar _c_pointer: C-compatible pointer representation
    """

    def __init__(
        self,
        pointer,
        dtype,
        mem_space: _cute_ir.AddressSpace = _cute_ir.AddressSpace.generic,
        assumed_align=None,
    ):
        self._pointer = pointer
        self._dtype = dtype
        self._addr_space = mem_space

        if assumed_align is None:
            self._assumed_align = dtype.width // 8
        else:
            self._assumed_align = assumed_align

        self._desc = None
        self._c_pointer = None
        assert int(self._pointer) % self._assumed_align == 0, (
            f"pointer must be {self._assumed_align} bytes aligned"
        )

    def size_in_bytes(self) -> int:
        return ctypes.sizeof(ctypes.c_void_p(int(self._pointer)))

    def __get_mlir_types__(self):
        return [self.mlir_type]

    def __c_pointers__(self):
        if self._c_pointer is None:
            self._desc = ctypes.c_void_p(int(self._pointer))
            self._c_pointer = ctypes.addressof(self._desc)
        return [self._c_pointer]

    def __new_from_mlir_values__(self, values):
        assert len(values) == 1
        return values[0]

    # Move mlir Type out of __init__ to decouple with mlir Context
    @property
    def mlir_type(self) -> ir.Type:
        return _cute_ir.PtrType.get(
            self._dtype.mlir_type, self._addr_space, self._assumed_align
        )

    @property
    def dtype(self) -> Type[Numeric]:
        return self._dtype

    @property
    def memspace(self):
        return self._addr_space

    def align(self, min_align: int, *, loc=None, ip=None) -> Pointer:
        raise NotImplementedError("align is not supported in runtime")

    def verify(self, expected_py_type):
        # if expected_py_type is Pointer:
        #     return True
        # elif isinstance(expected_py_type, ir.Value) and expected_py_type.ty is Pointer:
        #     return True
        if expected_py_type is Pointer or (
            isinstance(expected_py_type, ir.Value) and expected_py_type.ty is Pointer
        ):
            return True

        return False

    def __str__(self) -> str:
        return f"Ptr<0x{int(self._pointer):016x}@{self._addr_space}>"

    def __repr__(self):
        return self.__str__()


def make_ptr(
    dtype: Type[Numeric],
    value: Union[int, ctypes._Pointer],
    mem_space: AddressSpace = AddressSpace.generic,
    assumed_align=None,
) -> Pointer:
    """Create a pointer from a memory address

    :param dtype: Data type of the pointer elements
    :type dtype: Type[Numeric]
    :param value: Memory address as integer or ctypes pointer
    :type value: Union[int, ctypes._Pointer]
    :param mem_space: Memory address space, defaults to AddressSpace.generic
    :type mem_space: AddressSpace, optional
    :param assumed_align: Alignment in bytes, defaults to None
    :type assumed_align: int, optional
    :return: A pointer object
    :rtype: Pointer

    .. code-block:: python

        import numpy as np
        import ctypes

        from cutlass import Float32
        from cutlass.cute.runtime import make_ptr

        # Create a numpy array
        a = np.random.randn(16, 32).astype(np.float32)

        # Get pointer address as integer
        ptr_address = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Create pointer from address
        y = make_ptr(cutlass.Float32, ptr_address)
    """
    # check if value is int or ctypes.POINTER
    if isinstance(value, int):
        address_value = value
    elif isinstance(value, ctypes._Pointer):
        # get address value
        address_value = ctypes.cast(value, ctypes.c_void_p).value
        assert address_value is not None, "Pointer address is None"
    else:
        raise TypeError(
            f"Expect int or ctypes.POINTER for value but got {type(value)=}"
        )

    return _Pointer(address_value, dtype, mem_space, assumed_align=assumed_align)


# =============================================================================
# Device Info Utilities (from cute-dsl-zoo)
# =============================================================================


@functools.lru_cache(maxsize=16)
def get_sm_version(device: Optional[Union[int, torch.device, str]] = None) -> int:
    """Get the SM (compute capability) version of a CUDA device.

    Parameters
    ----------
    device : int, torch.device, str, or None
        Device to query. If None, uses current device.

    Returns
    -------
    int
        SM version as integer (e.g., 80 for SM80/Ampere, 90 for SM90/Hopper,
        100 for SM100/Blackwell).

    Example
    -------
    >>> sm = get_sm_version()
    >>> if sm >= 100:
    ...     print("Running on Blackwell!")
    """
    if not torch.cuda.is_available():
        return 80  # Default fallback
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


def get_l2_cache_size(device: Optional[Union[int, torch.device]] = None) -> int:
    """Get L2 cache size in bytes for a CUDA device.

    Parameters
    ----------
    device : int or torch.device, optional
        CUDA device. Default is current device.

    Returns
    -------
    int
        L2 cache size in bytes.

    Example
    -------
    >>> l2_size = get_l2_cache_size()
    >>> print(f"L2 cache: {l2_size / 1024 / 1024:.1f} MB")
    """
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.L2_cache_size


def get_shared_memory_per_block(
    device: Optional[Union[int, torch.device]] = None,
) -> int:
    """Get maximum shared memory per block (with optin) for a CUDA device.

    Parameters
    ----------
    device : int or torch.device, optional
        CUDA device. Default is current device.

    Returns
    -------
    int
        Maximum shared memory per block in bytes.

    Example
    -------
    >>> smem = get_shared_memory_per_block()
    >>> print(f"Max shared memory: {smem / 1024:.0f} KB")
    """
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    # Use shared_memory_per_block_optin if available (newer GPUs)
    return getattr(
        props,
        "shared_memory_per_block_optin",
        getattr(props, "shared_memory_per_block", 49152),
    )


# =============================================================================
# Math Utilities (from TRT-LLM)
# =============================================================================


def is_power_of_2(x: int) -> bool:
    """Check if a number is a power of 2.

    Parameters
    ----------
    x : int
        The number to check.

    Returns
    -------
    bool
        True if x is a positive power of 2.
    """
    return x > 0 and (x & (x - 1)) == 0


@dsl_user_op
def fmin(
    a: Union[float, cutlass.Float32],
    b: Union[float, cutlass.Float32],
    *,
    nan: bool = False,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    """Compute the minimum of two float32 values using PTX fmin.

    Parameters
    ----------
    a : float or cutlass.Float32
        First operand.
    b : float or cutlass.Float32
        Second operand.
    nan : bool, optional
        If True, propagate NaN values. Default is False.

    Returns
    -------
    cutlass.Float32
        The minimum value.
    """
    return cutlass.Float32(
        nvvm.fmin(
            T.f32(),
            cutlass.Float32(a).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            nan=nan,
            loc=loc,
            ip=ip,
        )
    )


def sigmoid_f32(
    a: Union[float, cutlass.Float32], fastmath: bool = False
) -> Union[float, cutlass.Float32]:
    """Compute the sigmoid of the input value.

    sigmoid(x) = 1 / (1 + exp(-x))

    Parameters
    ----------
    a : float or cutlass.Float32
        Input value.
    fastmath : bool, optional
        Use fast math approximations. Default is False.

    Returns
    -------
    float or cutlass.Float32
        The sigmoid of the input.
    """
    return cute.arch.rcp_approx(1.0 + cute.math.exp(-a, fastmath=fastmath))


def silu_f32(
    a: Union[float, cutlass.Float32], fastmath: bool = False
) -> Union[float, cutlass.Float32]:
    """Compute the SiLU (Sigmoid Linear Unit) activation.

    silu(x) = x * sigmoid(x)

    Parameters
    ----------
    a : float or cutlass.Float32
        Input value.
    fastmath : bool, optional
        Use fast math approximations. Default is False.

    Returns
    -------
    float or cutlass.Float32
        The SiLU of the input.
    """
    return a * sigmoid_f32(a, fastmath=fastmath)


# =============================================================================
# Atomic Operations (from TRT-LLM)
# =============================================================================


@dsl_user_op
def vectorized_atomic_add_bf16x8(
    rOut_epi_packed, scatter_out_offset, loc=None, ip=None
):
    """Perform a vectorized atomic add of 8 bf16 values (packed as 4 bf16x2).

    This uses PTX red.global.v4.bf16x2.add instruction for efficient
    scatter-accumulate operations in MoE kernels.

    Parameters
    ----------
    rOut_epi_packed : tensor
        Register tensor with 4 packed bf16x2 values.
    scatter_out_offset : pointer
        Output pointer with scatter offset applied.
    """
    llvm.inline_asm(
        None,
        [
            scatter_out_offset.iterator.llvm_ptr,
            llvm.bitcast(T.i32(), rOut_epi_packed[0, None].load().ir_value()),
            llvm.bitcast(T.i32(), rOut_epi_packed[1, None].load().ir_value()),
            llvm.bitcast(T.i32(), rOut_epi_packed[2, None].load().ir_value()),
            llvm.bitcast(T.i32(), rOut_epi_packed[3, None].load().ir_value()),
        ],
        "red.global.v4.bf16x2.add.noftz [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def vectorized_atomic_add_fp32x2(
    rOut_epi_packed, scatter_out_offset, loc=None, ip=None
):
    """Perform a vectorized atomic add of 2 fp32 values.

    This uses PTX red.global.v2.f32.add instruction.

    Parameters
    ----------
    rOut_epi_packed : tensor
        Register tensor with 2 fp32 values.
    scatter_out_offset : pointer
        Output pointer with scatter offset applied.
    """
    llvm.inline_asm(
        None,
        [
            scatter_out_offset.iterator.llvm_ptr,
            rOut_epi_packed[0].ir_value(),
            rOut_epi_packed[1].ir_value(),
        ],
        "red.global.v2.f32.add [$0], {$1, $2};",
        "l,f,f",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_add_func(rOut_epi_packed, scatter_out_offset, loc=None, ip=None):
    """Perform a scalar atomic add (supports fp32 and bf16).

    Parameters
    ----------
    rOut_epi_packed : value
        Scalar value to add.
    scatter_out_offset : pointer
        Output pointer with scatter offset applied.
    """
    if cutlass.const_expr(rOut_epi_packed.dtype == cutlass.Float32):
        llvm.inline_asm(
            None,
            [
                scatter_out_offset.iterator.llvm_ptr,
                rOut_epi_packed.ir_value(),
            ],
            "red.global.add.f32 [$0], $1;",
            "l,f",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )
    elif cutlass.const_expr(rOut_epi_packed.dtype == cutlass.BFloat16):
        llvm.inline_asm(
            None,
            [
                scatter_out_offset.iterator.llvm_ptr,
                llvm.bitcast(T.i16(), rOut_epi_packed.ir_value()),
            ],
            "red.add.noftz.bf16 [$0], $1;",
            "l,h",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )


# =============================================================================
# Grid Dependency Control (from TRT-LLM, for Blackwell/SM100)
# =============================================================================


@dsl_user_op
def griddepcontrol_wait(*, loc=None, ip=None) -> None:
    """Wait for the previous kernel's grid to complete.

    This instruction is used to wait for the previous kernel's grid ending
    (all blocks of the previous kernel have finished and memflushed), i.e.,
    the instruction after this instruction will not be issued until the previous
    grid has finished.

    Note: This is a Blackwell (SM100) specific instruction.
    """
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="griddepcontrol.wait;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def griddepcontrol_launch_dependents(*, loc=None, ip=None) -> None:
    """Hint to launch dependent kernels earlier.

    Issuing the launch_dependents instruction hints a dependent kernel to
    launch earlier. launch_dependents doesn't impact the functionality but
    the performance: Launching a dependent kernel too early can compete
    with current kernels, while launching too late can lead to a long latency.

    Note: This is a Blackwell (SM100) specific instruction.
    """
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="griddepcontrol.launch_dependents;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
