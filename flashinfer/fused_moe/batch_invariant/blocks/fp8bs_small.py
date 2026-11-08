# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The block-scale fp8 MoE class's small kernels: the standalone ``fp8bs.quant`` kernel (rows of 16-bit values to e4m3 codes and
one fp32 scale per 128-element block, the scales in the ``MN`` layout ``(K / 128, M)`` the block-scale GEMM's row-scale ring
reads or in the ``K`` layout ``(M, K / 128)``: a placement of the store, one kernel, the tensor's own strides) and the gather
of the activations' block scales to the expanded rows (``sf_perm[kb, p] = sf[kb, src[p] // k]``: a copy, no arithmetic).

Quantizer placement: a warp pass covers two blocks (sixteen lanes per block, eight consecutive elements per lane: one 16-byte
load, one 8-byte store); ``wpr`` warps share a row and take its block pairs round-robin (a placement: every block's amax is
the plan's tree ``[(8, reg), (16, lane)]`` through ``canon.init_state`` / ``canon.run`` whichever warp holds it). ``K`` and
the row strides are runtime arguments; a kernel is compiled per (input dtype, warps per row, PDL trigger placement). Nothing
here decides a numeric order: the reduction is the engine's plan, the chains are :mod:`..spec_fp8bs`'s, the conversion the
declared ``cvt.rn.satfinite.e4m3x2.f32``.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint8, Uint32

from ....batch_invariant import canon
from ....batch_invariant.quantization.blocks import LanePeers, pack_block_codes
from ....batch_invariant.quantization.cute_ext import CuteF32
from .. import pdl
from ..spec_fp8bs import (
    BLOCK,
    block_amax_placement,
    block_chain,
    block_reduction,
    leaf_value,
    quant_pre,
)

WARPS = 8  # warps per CTA
LANES = 32
V = 8  # elements per lane: one 16-byte load of 16-bit values, one block per sixteen lanes
LANES_PER_BLOCK = BLOCK // V
TRIGGERS = ("none", "late", "early")
SCATTER_THREADS = 256
SCALE_BLOCKS_PER_THREAD = (
    8  # block columns per thread of the scale gather, their loads issued together
)


class _Loader:
    """``canon.init_state`` leaf loader: the spec's leaf (``fabs`` of the widened element) for the register key."""

    def __init__(self, frag):
        self.frag = frag

    def __call__(self, key):
        return leaf_value(CuteF32, self.frag[key[0]].to(Float32))


class BSQuantKernel:
    """``codes[m, k] = cvt_e4m3(mul(x[m, k], inv(m, k // 128)))``, ``s[m, k // 128]`` for rows of 16-bit values."""

    def __init__(self, dtype: cutlass.Numeric, wpr: int, trigger: str = "none"):
        assert dtype.width == 16, dtype
        assert wpr >= 1 and wpr & (wpr - 1) == 0 and wpr <= 8, wpr
        assert trigger in TRIGGERS, trigger
        self.dtype, self.wpr, self.trigger = dtype, wpr, trigger
        self.red = block_reduction(leaf_x=None)
        self.plan = canon.plan(self.red, block_amax_placement(V))
        assert self.plan.units["lane"] == LANES_PER_BLOCK and self.plan.n_reg == 1

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        M: Int32,
        K: Int32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel(mX, mY, mS, M, K, enable_pdl).launch(
            grid=[cute.ceil_div(M * Int32(self.wpr), WARPS), 1, 1],
            block=[WARPS * LANES, 1, 1],
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        M: Int32,
        K: Int32,
        enable_pdl: cutlass.Constexpr[bool],
    ):
        fp = CuteF32
        wpr = self.wpr
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        if cutlass.const_expr(enable_pdl):
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.trigger == "early"):
                cute.arch.griddepcontrol_launch_dependents()
        g = bidx * WARPS + tidx // LANES
        row_raw = g // Int32(wpr)
        part = g % Int32(wpr)
        valid = row_raw < M
        row = cutlass.min(row_raw, M - Int32(1))
        half = lane // Int32(
            LANES_PER_BLOCK
        )  # which of the pass's two blocks this lane holds
        sub = lane % Int32(
            LANES_PER_BLOCK
        )  # the lane's eight elements inside the block
        nb = K // Int32(BLOCK)
        n_pairs = (nb + Int32(1)) // Int32(2)
        n_mine = (n_pairs - part + Int32(wpr - 1)) // Int32(wpr)
        atom_x = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=V * mX.element_type.width,
        )
        atom_y = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Uint8, num_bits_per_copy=8 * V
        )
        row_x = mX[(row, None)]  # 1-D row views (a (1, V) tile mis-vectorises)
        row_y = mY[(row, None)]
        frag = cute.make_rmem_tensor((V,), mX.element_type)
        fy_words = cute.make_rmem_tensor(cute.make_layout((V // 4,)), Uint32)
        fy = cute.make_tensor(
            cute.recast_ptr(fy_words.iterator, dtype=Uint8), cute.make_layout((V,))
        )
        for jj in cutlass.range(0, n_mine, 1):
            blk = (jj * Int32(wpr) + part) * Int32(2) + half
            live = valid & (blk < nb)
            blk_c = cutlass.min(blk, nb - Int32(1))
            chunk = blk_c * Int32(LANES_PER_BLOCK) + sub
            cute.copy(
                atom_x, cute.local_tile(row_x, (V,), (chunk,)), frag
            )  # the lane's eight elements (clamped: a dead lane's bytes are never stored)
            # ===== the block amax: the engine's plan (the register tree fused with the load, then the xor butterfly) =====
            state, start = canon.init_state(self.plan, fp, _Loader(frag))
            amax = canon.run(self.plan, state, start, LanePeers(), fp)
            # ===== the chains: the scale, its reciprocal, the element conversion =====
            s, inv = block_chain(fp, amax)
            ys = [quant_pre(fp, frag[v].to(Float32), inv) for v in range(V)]
            for w, word in enumerate(pack_block_codes(fp, ys, "e4m3")):
                fy_words[w] = word
            if live:
                cute.copy(atom_y, fy, cute.local_tile(row_y, (V,), (chunk,)))
            if live & (sub == Int32(0)):
                mS[(row, blk_c)] = s
        if cutlass.const_expr(enable_pdl and self.trigger == "late"):
            cute.arch.griddepcontrol_launch_dependents()


class GatherScalesKernel:
    """``sf_perm[kb, p] = sf[kb, src[p] // k]`` for ``p < Tk``, ``kb < KB``; ``sf`` is read through its own strides (the ``(KB, T)``
    view of either scale layout), ``sf_perm`` is the contiguous ``(KB, Tk)`` MN-major tensor. Thread = expanded row, so a
    warp's stores of one block column are one contiguous line (the reads gather one 4-byte value per token); a thread handles
    ``SCALE_BLOCKS_PER_THREAD`` block columns with their loads in flight together and the grid's second dimension covers the
    rest (the launch is latency-bound: a serial walk over the block columns cost one memory round trip each)."""

    def __init__(self, k: int):
        self.k = k

    @cute.jit
    def __call__(
        self,
        mS: cute.Tensor,
        mSrc: cute.Tensor,
        mSp: cute.Tensor,
        Tk: Int32,
        KB: Int32,
        stream,
    ):
        self.kernel(mS, mSrc, mSp, Tk, KB).launch(
            grid=[
                cute.ceil_div(Tk, SCATTER_THREADS),
                cute.ceil_div(KB, SCALE_BLOCKS_PER_THREAD),
                1,
            ],
            block=[SCATTER_THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self, mS: cute.Tensor, mSrc: cute.Tensor, mSp: cute.Tensor, Tk: Int32, KB: Int32
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        p_raw = bidx * SCATTER_THREADS + tidx
        valid = p_raw < Tk
        p = cutlass.min(p_raw, Tk - Int32(1))
        t = mSrc[p] // Int32(self.k)
        kb0 = bidy * SCALE_BLOCKS_PER_THREAD
        vals = []
        for j in cutlass.range_constexpr(SCALE_BLOCKS_PER_THREAD):
            # the thread's block columns, their loads all in flight
            vals.append(mS[(cutlass.min(kb0 + j, KB - Int32(1)), t)])
        for j in cutlass.range_constexpr(SCALE_BLOCKS_PER_THREAD):
            kb = kb0 + j
            if valid:
                if kb < KB:
                    mSp[(kb, p)] = vals[j]
        pdl.trigger_one_wave()
