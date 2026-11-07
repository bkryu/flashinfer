# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 MoE classes' row moves (cache module ``batch_invariant_moe_nvfp4``): the scatter of the per-token quantized
activations into the grouped layout (the packed fp4 rows to their dense destinations, the 128x4 swizzled scale factors and
the per-token scales to the padded positions), and the re-tiling of the intermediate's scale factors and row scales from the
dense rows to the padded positions. Byte moves only: no arithmetic, a placement of the class's bytes.

The scale-factor buffers are FlashInfer's 128x4 swizzled layout (:mod:`flashinfer.batch_invariant.quantization.layout`):
the four scales of columns ``4g .. 4g + 3`` of a row are contiguous at byte ``g * 512 + (row % 32) * 16 + ((row % 128) //
32) * 4 + (row // 128) * 128 * pcols``, so a row's scales move as 4-byte groups.
"""

from __future__ import annotations

import functools
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint32

from ...batch_invariant import jit
from .permute import LANES, WARPS

MODULE = "batch_invariant_moe_nvfp4"
SF_COL_TILE = 4  # scale columns per contiguous 4-byte group of the 128x4 layout
# blocks past which the scatter kernels' warps stride over the rows: the block launch rate, not the bytes, bounds a
# one-row-per-warp grid at the large batches
SCATTER_GRID_CAP = 4096


def _sf_group_offset(row: Int32, g: Int32, pcols: Int32) -> Int64:
    """Byte offset of the first scale of column group ``g`` of ``row`` in the 128x4 swizzled layout."""
    r = row.to(Int64)
    r128 = (r % 128).to(Int32)
    inner = g * Int32(512) + (r128 % 32) * 16 + (r128 // 32) * 4
    return (r // 128) * (Int32(128) * pcols).to(Int64) + inner.to(Int64)


class ScatterFp4Kernel:
    """The packed fp4 row of token ``t = i // k`` to row ``dest[i]`` (``B`` bytes), its swizzled scale-factor groups and its
    per-token scale to row ``dest_sf[i]`` (the padded position; the same tensor as ``dest`` for the padded packing). ``wpr``
    warps per expanded row (a placement, as the row gather: ``warps_per_row``), the row groups of a capped grid striding over
    the rows (``SCATTER_GRID_CAP``: a block per row would be bound by the block launch rate at the large batches); a warp's
    lanes stride over its 32-chunk slabs of the 16-byte row chunks, then over its slabs of the 4-byte scale groups; lane 0 of
    the row's first warp moves the per-token scale."""

    def __init__(self, k: int, wpr: int = 1):
        assert wpr >= 1 and wpr & (wpr - 1) == 0, wpr
        self.k, self.wpr = k, wpr

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mSf: cute.Tensor,
        mPts: cute.Tensor,
        mDest: cute.Tensor,
        mDestSf: cute.Tensor,
        mXp: cute.Tensor,
        mSfp: cute.Tensor,
        mPtsp: cute.Tensor,
        Tk: Int32,
        B: Int32,
        pcols: Int32,
        stream,
    ):
        self.kernel(
            mX, mSf, mPts, mDest, mDestSf, mXp, mSfp, mPtsp, Tk, B, pcols
        ).launch(
            grid=[scatter_grid(Tk * Int32(self.wpr)), 1, 1],
            block=[WARPS * LANES, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mSf: cute.Tensor,
        mPts: cute.Tensor,
        mDest: cute.Tensor,
        mDestSf: cute.Tensor,
        mXp: cute.Tensor,
        mSfp: cute.Tensor,
        mPtsp: cute.Tensor,
        Tk: Int32,
        B: Int32,
        pcols: Int32,
    ):
        wpr = Int32(self.wpr)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = cute.arch.lane_idx()
        cute.arch.griddepcontrol_wait()
        g = bidx * WARPS + tidx // LANES
        w = g // wpr  # this warp's row group
        part = g % wpr  # its slab phase within the row
        stride = (gdim * WARPS) // wpr  # row groups per sweep
        # this group's rows: w, w + stride, ...
        n_rows = cutlass.max((Tk - w + stride - Int32(1)) // stride, Int32(0))
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Uint8, num_bits_per_copy=128
        )
        n_vec = B // Int32(16)
        step = wpr * Int32(LANES)  # chunk j = (jj * wpr + part) * LANES + lane
        n_mine = (n_vec - lane - part * Int32(LANES) + step - Int32(1)) // step
        frag = cute.make_rmem_tensor((16,), Uint8)
        src32 = cute.make_tensor(
            cute.recast_ptr(mSf.iterator, dtype=Uint32),
            cute.make_layout((cute.size(mSf) // 4,)),
        )
        dst32 = cute.make_tensor(
            cute.recast_ptr(mSfp.iterator, dtype=Uint32),
            cute.make_layout((cute.size(mSfp) // 4,)),
        )
        n_groups = pcols // Int32(SF_COL_TILE)
        n_sf = (n_groups - lane - part * Int32(LANES) + step - Int32(1)) // step
        if w < stride:
            for r in cutlass.range(0, n_rows, 1):
                i = w + r * stride
                t = i // Int32(self.k)
                p = mDest[i]
                ps = mDestSf[i]
                if (lane == 0) & (part == 0):
                    mPtsp[ps] = mPts[t]
                row_x = mX[(t, None)]
                row_p = mXp[(p, None)]
                for jj in cutlass.range(0, n_mine, 1):
                    j = (jj * wpr + part) * Int32(LANES) + lane
                    cute.copy(atom, cute.local_tile(row_x, (16,), (j,)), frag)
                    cute.copy(atom, frag, cute.local_tile(row_p, (16,), (j,)))
                for jj in cutlass.range(0, n_sf, 1):
                    gg = (jj * wpr + part) * Int32(LANES) + lane
                    dst32[_sf_group_offset(ps, gg, pcols) // 4] = src32[
                        _sf_group_offset(t, gg, pcols) // 4
                    ]
        cute.arch.griddepcontrol_launch_dependents()


class ScatterSfKernel:
    """The scale factors of row ``t = i // k`` moved to row ``dest[i]`` (4-byte groups of the 128x4 layout; the source and
    the destination with the same ``pcols``), and ``pts_perm[dest[i]] = pts[t]``: the re-tiling of the intermediate's scale
    factors and row scales from the dense rows to the padded positions (``k = 1``, ``dest`` the padded position of every
    dense row). One lane per 4-byte group, one warp per row, the warps of a capped grid striding over the rows."""

    def __init__(self, k: int):
        self.k = k

    @cute.jit
    def __call__(
        self,
        mSf: cute.Tensor,
        mPts: cute.Tensor,
        mDest: cute.Tensor,
        mSfp: cute.Tensor,
        mPtsp: cute.Tensor,
        Tk: Int32,
        pcols: Int32,
        stream,
    ):
        self.kernel(mSf, mPts, mDest, mSfp, mPtsp, Tk, pcols).launch(
            grid=[scatter_grid(Tk), 1, 1],
            block=[WARPS * LANES, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mSf: cute.Tensor,
        mPts: cute.Tensor,
        mDest: cute.Tensor,
        mSfp: cute.Tensor,
        mPtsp: cute.Tensor,
        Tk: Int32,
        pcols: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = cute.arch.lane_idx()
        cute.arch.griddepcontrol_wait()
        w = bidx * WARPS + tidx // LANES
        stride = gdim * WARPS
        n_rows = cutlass.max((Tk - w + stride - Int32(1)) // stride, Int32(0))
        src32 = cute.make_tensor(
            cute.recast_ptr(mSf.iterator, dtype=Uint32),
            cute.make_layout((cute.size(mSf) // 4,)),
        )
        dst32 = cute.make_tensor(
            cute.recast_ptr(mSfp.iterator, dtype=Uint32),
            cute.make_layout((cute.size(mSfp) // 4,)),
        )
        n_groups = pcols // Int32(SF_COL_TILE)
        n_mine = (n_groups - lane + Int32(LANES - 1)) // Int32(LANES)
        for r in cutlass.range(0, n_rows, 1):
            i = w + r * stride
            t = i // Int32(self.k)
            p = mDest[i]
            if lane == 0:
                mPtsp[p] = mPts[t]
            for jj in cutlass.range(0, n_mine, 1):
                g = jj * LANES + lane
                dst32[_sf_group_offset(p, g, pcols) // 4] = src32[
                    _sf_group_offset(t, g, pcols) // 4
                ]
        cute.arch.griddepcontrol_launch_dependents()


def scatter_grid(Tk: Int32) -> Int32:
    """Blocks of a scatter launch: one warp per row up to ``SCATTER_GRID_CAP`` blocks, beyond which the warps stride over
    the rows (a placement: the same bytes move whatever the grid)."""
    return cutlass.min(cute.ceil_div(Tk, WARPS), Int32(SCATTER_GRID_CAP))


@functools.cache
def get_compiled_scatter_fp4(k: int, wpr: int = 1) -> Any:
    """``kernel(x4 [T, B] u8, sf [n_src] u8, pts [T] f32, dest [T k] i32, dest_sf [T k] i32, x_perm [rows, B] u8, sf_perm
    [n_dst] u8, pts_perm [rows_sf] f32, Tk, B, pcols)``, ``B % 16 == 0``; ``wpr`` warps per row (``warps_per_row``)."""

    def compile_fn():
        sym_t, sym_n, sym_p, sym_b = (
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32, divisibility=16),
        )
        sym_s, sym_sp, sym_pp = (
            cute.sym_int(32, divisibility=512),
            cute.sym_int(32, divisibility=512),
            cute.sym_int(32),
        )
        x_fake = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_t, sym_b),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        sf_fake = cute.runtime.make_fake_tensor(Uint8, (sym_s,), (1,), assumed_align=16)
        pts_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_t,), (1,), assumed_align=4
        )
        d_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        ds_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        xp_fake = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_p, sym_b),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        sfp_fake = cute.runtime.make_fake_tensor(
            Uint8, (sym_sp,), (1,), assumed_align=16
        )
        ptsp_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_pp,), (1,), assumed_align=4
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            ScatterFp4Kernel(k, wpr),
            x_fake,
            sf_fake,
            pts_fake,
            d_fake,
            ds_fake,
            xp_fake,
            sfp_fake,
            ptsp_fake,
            Int32(1),
            Int32(16),
            Int32(4),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(
        MODULE,
        jit.sanitize_symbol_name(
            f"moe_scatter_fp4_k{k}" + (f"_w{wpr}" if wpr > 1 else "")
        ),
        compile_fn,
    )


@functools.cache
def get_compiled_scatter_sf(k: int) -> Any:
    """``kernel(sf [n_src] u8, pts [T] f32, dest [T k] i32, sf_perm [n_dst] u8, pts_perm [rows_sf] f32, Tk, pcols)``."""

    def compile_fn():
        sym_s, sym_t, sym_n, sym_sp, sym_p = (
            cute.sym_int(32, divisibility=512),
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32, divisibility=512),
            cute.sym_int(32),
        )
        sf_fake = cute.runtime.make_fake_tensor(Uint8, (sym_s,), (1,), assumed_align=16)
        pts_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_t,), (1,), assumed_align=4
        )
        d_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        sfp_fake = cute.runtime.make_fake_tensor(
            Uint8, (sym_sp,), (1,), assumed_align=16
        )
        ptsp_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_p,), (1,), assumed_align=4
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            ScatterSfKernel(k),
            sf_fake,
            pts_fake,
            d_fake,
            sfp_fake,
            ptsp_fake,
            Int32(1),
            Int32(4),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(
        MODULE, jit.sanitize_symbol_name(f"moe_scatter_sf_k{k}"), compile_fn
    )
