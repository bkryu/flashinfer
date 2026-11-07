# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 class's scatter (plumbing, no floating point): the quantized activation rows and their UE8M0 block scales moved
into the padded group layout the grouped block-scaled GEMM reads.

``x_perm[dest[i], :] = xq[i // k, :]`` for every expanded row ``i`` of the ``T * k`` (token, slot) pairs: one warp per live
row, nothing written to the padding rows of the 128-row atoms (their bytes stay unspecified: the MMA is row-local and no tile
stores a padding row, so they never reach a stored value); the scale bytes of token ``t`` are read from the **linear** layout
the class takes its activations in (``xsf [T, K / 32]``: the MXFP8 MoE activation pack) and written to the **128x4
swizzled** layout of the ``(Tk_pad, K)`` operand (byte ``(r // 128) * 128 * pcols + (j // 4) * 512 + (r % 32) * 16 +
((r % 128) // 32) * 4 + j % 4``), the padding columns ``j >= K / 32`` of a live row zero. 16-byte vectors for the values
(lanes stride over the row), one lane per scale byte. Every copy is an unpredicated ``cute.copy`` under a plain branch (a
predicated copy lowers to element-wise stores), launched with programmatic dependent launch.
"""

from __future__ import annotations

import functools
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8

from ...batch_invariant import jit

MODULE = "batch_invariant_moe_mxfp8"
WARPS = 4
LANES = 32
VEC_BYTES = 16


@cute.jit
def _sf_byte_offset(row: Int32, col: Int32, pcols: Int32) -> Int64:
    """Byte offset of scale column ``col`` of ``row`` in the 128x4 swizzled layout of a buffer with ``pcols`` scale columns."""
    r = row.to(Int64)
    r128 = (r % 128).to(Int32)
    inner = (col // 4) * Int32(512) + (r128 % 32) * 16 + (r128 // 32) * 4 + (col % 4)
    return (r // 128) * (Int32(128) * pcols).to(Int64) + inner.to(Int64)


class ScatterMxKernel:
    """``kernel(xq [T, B] u8, xsf [T * nb] u8 (linear), dest [Tk] i32, x_perm [Tk_pad, B] u8, sf_perm [sf_numel(Tk_pad, K)]
    u8, Tk, B, nb, pcols)``: ``B`` bytes per row (``B % 16 == 0``), ``nb = K / 32`` scale bytes per row, ``pcols`` the
    padded scale columns of the destination (a multiple of 4); ``dest[i]`` the padded row of expanded row ``i`` (token
    ``i // k``). One warp per live row; the padding rows of the 128-row atoms are never written (unspecified bytes the
    row-local MMA reads into rows no tile stores)."""

    def __init__(self, k: int):
        self.k = k

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mSf: cute.Tensor,
        mDest: cute.Tensor,
        mXp: cute.Tensor,
        mSfp: cute.Tensor,
        Tk: Int32,
        B: Int32,
        nb: Int32,
        pcols: Int32,
        stream,
    ):
        self.kernel(mX, mSf, mDest, mXp, mSfp, Tk, B, nb, pcols).launch(
            grid=[cute.ceil_div(Tk, WARPS), 1, 1],
            block=[WARPS * LANES, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mSf: cute.Tensor,
        mDest: cute.Tensor,
        mXp: cute.Tensor,
        mSfp: cute.Tensor,
        Tk: Int32,
        B: Int32,
        nb: Int32,
        pcols: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        cute.arch.griddepcontrol_wait()
        i_raw = bidx * WARPS + tidx // LANES
        valid = i_raw < Tk
        i = cutlass.min(i_raw, Tk - Int32(1))
        p = mDest[i]
        t = i // Int32(self.k)
        # the values: 16-byte vectors, lanes stride over the row
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Uint8, num_bits_per_copy=8 * VEC_BYTES
        )
        n_vec = B // Int32(VEC_BYTES)
        frag = cute.make_rmem_tensor((VEC_BYTES,), Uint8)
        row_x = mX[(t, None)]
        row_p = mXp[(p, None)]
        n_mine = (n_vec - lane + Int32(LANES - 1)) // Int32(LANES)
        for jj in cutlass.range(0, n_mine, 1):
            j = jj * LANES + lane
            cute.copy(atom, cute.local_tile(row_x, (VEC_BYTES,), (j,)), frag)
            if valid:
                cute.copy(atom, frag, cute.local_tile(row_p, (VEC_BYTES,), (j,)))
        # the scale bytes: linear source (row t, column j) -> the swizzled destination (row p, column j); padding columns zero
        n_sf = (pcols - lane + Int32(LANES - 1)) // Int32(LANES)
        for jj in cutlass.range(0, n_sf, 1):
            j = jj * LANES + lane
            v = Uint8(0)
            if j < nb:
                v = mSf[t.to(Int64) * nb.to(Int64) + j.to(Int64)]
            if valid:
                mSfp[_sf_byte_offset(p, j, pcols)] = v
        cute.arch.griddepcontrol_launch_dependents()


def scatter_mx_kernel_name(k: int) -> str:
    return jit.sanitize_symbol_name(f"moe_scatter_mx_k{k}")


@functools.cache
def get_compiled_scatter_mx(k: int) -> Any:
    """``kernel(xq [T, B] u8, xsf [n_src] u8, src [Tk_pad] i32, x_perm [Tk_pad, B] u8, sf_perm [n_dst] u8, Tk_pad, B, nb, pcols)``."""

    def compile_fn():
        sym_t, sym_n, sym_p, sym_b = (
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32, divisibility=16),
        )
        sym_s, sym_sp = cute.sym_int(32), cute.sym_int(32, divisibility=512)
        x_fake = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_t, sym_b),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        sf_fake = cute.runtime.make_fake_tensor(Uint8, (sym_s,), (1,), assumed_align=16)
        dest_fake = cute.runtime.make_fake_tensor(
            Int32, (sym_p,), (1,), assumed_align=4
        )
        xp_fake = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_n, sym_b),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        sfp_fake = cute.runtime.make_fake_tensor(
            Uint8, (sym_sp,), (1,), assumed_align=16
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            ScatterMxKernel(k),
            x_fake,
            sf_fake,
            dest_fake,
            xp_fake,
            sfp_fake,
            Int32(128),
            Int32(16),
            Int32(1),
            Int32(4),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, scatter_mx_kernel_name(k), compile_fn)
