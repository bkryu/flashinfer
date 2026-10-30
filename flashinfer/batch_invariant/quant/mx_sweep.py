"""Device conversion sweep for the MX classes: apply ``blocks.cvt_f32_to_ue8m0`` (all three roundings) to arbitrary
f32 inputs so the CPU model ``fp32.to_ue8m0`` can be compared with the hardware over all 2^32 bit patterns
(tests/test_mx_quant_gpu.py). Test utilities, not part of the op."""

from __future__ import annotations

import functools

import cutlass.cute as cute
from cutlass import Float32, Int32, Uint8

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .. import jit
from ..kernel_names import MX_QUANT_MODULE
from .blocks import cvt_f32_to_ue8m0

THREADS = 256


class UE8M0Sweep:
    """``bytes[r][i]`` = the UE8M0 byte of ``inp[i]`` under rounding ``r`` (0 = rp, 1 = rz, 2 = rn); ``vals[r][i]`` = its widened value."""

    @cute.jit
    def __call__(
        self,
        mIn: cute.Tensor,
        mBytes: cute.Tensor,
        mVals: cute.Tensor,
        N: Int32,
        stream,
    ):
        self.kernel(mIn, mBytes, mVals, N).launch(
            grid=[cute.ceil_div(N, THREADS), 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self, mIn: cute.Tensor, mBytes: cute.Tensor, mVals: cute.Tensor, N: Int32
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * THREADS + tidx
        if i < N:
            x = mIn[i]
            for r, rnd in enumerate(("rp", "rz", "rn")):
                byte, val = cvt_f32_to_ue8m0(x, rnd)
                mBytes[r, i] = byte.to(Uint8)
                mVals[r, i] = val


@functools.cache
def get_ue8m0_sweep():
    def compile_fn():
        n = cute.sym_int(32)
        inp = cute.runtime.make_fake_tensor(Float32, (n,), (1,), assumed_align=16)
        b = cute.runtime.make_fake_tensor(
            Uint8,
            (3, cute.sym_int(32)),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        v = cute.runtime.make_fake_tensor(
            Float32,
            (3, cute.sym_int(32)),
            (cute.sym_int64(divisibility=4), 1),
            assumed_align=16,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            UE8M0Sweep(), inp, b, v, Int32(1), stream, options="--enable-tvm-ffi"
        )

    return jit.build_and_load(MX_QUANT_MODULE, "sweep_ue8m0", compile_fn)


@dsl_user_op
def _rcp_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """FlashInfer's ``rcp.approx.ftz.f32`` (test probe only: rule I1 forbids it in our kernels)."""
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


class RcpProbe:
    """``out[i] = rcp.approx.ftz.f32(inp[i])``: what FlashInfer's MX quantizers multiply the amax by (``rcp(448)``, ``rcp(6)``)."""

    @cute.jit
    def __call__(self, mIn: cute.Tensor, mOut: cute.Tensor, N: Int32, stream):
        self.kernel(mIn, mOut, N).launch(
            grid=[cute.ceil_div(N, THREADS), 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mIn: cute.Tensor, mOut: cute.Tensor, N: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * THREADS + tidx
        if i < N:
            mOut[i] = _rcp_approx_ftz(mIn[i])


@functools.cache
def get_rcp_probe():
    def compile_fn():
        n = cute.sym_int(32)
        inp = cute.runtime.make_fake_tensor(Float32, (n,), (1,), assumed_align=16)
        out = cute.runtime.make_fake_tensor(
            Float32, (cute.sym_int(32),), (1,), assumed_align=16
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            RcpProbe(), inp, out, Int32(1), stream, options="--enable-tvm-ffi"
        )

    return jit.build_and_load(MX_QUANT_MODULE, "probe_rcp_approx", compile_fn)
