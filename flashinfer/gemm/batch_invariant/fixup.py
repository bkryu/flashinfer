# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Split-K fixup: ``out[m, n] = cvt_rn( ((+0 + W[m,n,0]) + W[m,n,1]) + ... )`` over the C fp32 chunk partials, in
chunk order (the ``chunk`` fold digit of the chunked specification); with a bias, ``cvt_rn(add_rn(fold, bias[n]))``
(the biased chain: the bias enters once, here, after the whole chunk fold; the partials are bias-free). All math
through ``CuteF32``; vectorised over n with 16-byte loads; columns predicated; the grid's y extent is capped
(``MAX_GRID_Y``) and each block strides over the rows, so any M launches."""

from __future__ import annotations

import functools
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from ...batch_invariant import jit
from ...batch_invariant.fp_interface.cute import CuteF32, get_cutlass_dtype
from .blocks.compile import MODULE

VEC = 4  # fp32 per 16-byte load
THREADS = 256
MAX_GRID_Y = 65535  # the hardware's grid.y limit; rows beyond it are strided over


class SplitKFixup:
    def __init__(self, C: int, out_dtype, bias: bool = False):
        self.C, self.out_dtype, self.bias = C, out_dtype, bias

    @cute.jit
    def __call__(
        self, mW: cute.Tensor, mY: cute.Tensor, M: Int32, N: Int32, rows: Int32, stream
    ):
        """``rows = min(M, MAX_GRID_Y)`` is the grid's y extent (chosen by the caller); block ``by`` handles rows ``by, by + rows, ...``."""
        n_vec = cute.ceil_div(N, VEC)
        self.kernel(mW, mY, None, M, N, n_vec, rows).launch(
            grid=[cute.ceil_div(n_vec, THREADS), rows, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mBias: Optional[cute.Tensor],
        M: Int32,
        N: Int32,
        n_vec: Int32,
        rows: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        v = bx * THREADS + tidx
        if v < n_vec:
            n0 = v * VEC
            for m in cutlass.range(by, M, rows, unroll=1):
                for j in cutlass.range_constexpr(VEC):
                    n = n0 + j
                    if n < N:
                        acc = CuteF32.const(0.0)
                        for c in cutlass.range_constexpr(self.C):
                            acc = CuteF32.add(acc, mW[m, n, c])
                        if cutlass.const_expr(self.bias):
                            acc = CuteF32.add(
                                acc, CuteF32.widen(mBias[n])
                            )  # the chain's bias step, once, after the chunk fold
                        mY[m, n] = acc.to(mY.element_type)


class SplitKFixupBias(SplitKFixup):
    """The fixup of the biased split class: the (N,) bf16 bias is one more operand."""

    def __init__(self, C: int, out_dtype):
        super().__init__(C, out_dtype, bias=True)

    @cute.jit
    def __call__(
        self,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mBias: cute.Tensor,
        M: Int32,
        N: Int32,
        rows: Int32,
        stream,
    ):
        n_vec = cute.ceil_div(N, VEC)
        self.kernel(mW, mY, mBias, M, N, n_vec, rows).launch(
            grid=[cute.ceil_div(n_vec, THREADS), rows, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )


@functools.cache
def get_fixup(C: int, out_dtype_name: str = "bf16", bias: bool = False):
    """One fixup kernel per chunk count (and per chain: with or without the bias); M and N are runtime arguments."""
    out_dt = get_cutlass_dtype(out_dtype_name)
    name = jit.sanitize_symbol_name(
        f"splitk_fixup_C{C}_{out_dtype_name}" + ("_bias" if bias else "")
    )

    def compile_fn():
        sym_m, sym_n = cute.sym_int(64), cute.sym_int(32, divisibility=4)
        w_fake = cute.runtime.make_fake_tensor(
            Float32,
            (sym_m, sym_n, C),
            (cute.sym_int64(divisibility=4), 1, cute.sym_int64(divisibility=4)),
            assumed_align=16,
        )
        y_fake = cute.runtime.make_fake_tensor(
            out_dt,
            (sym_m, sym_n),
            (cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if bias:
            bias_fake = cute.runtime.make_fake_tensor(
                get_cutlass_dtype("bf16"), (sym_n,), (1,), assumed_align=16
            )
            return cute.compile(
                SplitKFixupBias(C, out_dt),
                w_fake,
                y_fake,
                bias_fake,
                Int32(1),
                Int32(1),
                Int32(1),
                stream_fake,
                options="--enable-tvm-ffi",
            )
        return cute.compile(
            SplitKFixup(C, out_dt),
            w_fake,
            y_fake,
            Int32(1),
            Int32(1),
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, name, compile_fn)
