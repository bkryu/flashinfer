# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Group-masked epilogue of the NVFP4 MoE classes: the bf16 grouped epilogue (:mod:`.epilogue`: TMEM -> registers, the
chain, the staged vector stores) with the W4A4 row scale.

:class:`W4A4Chain` is the bf16 class's :class:`.epilogue.GroupedChain` with the scale source ``w4a4`` (``s = mul(alpha[e],
pts[row])``: the expert's alpha as given times the row's per-token scale, formed once per (row, tile) and applied to every
accumulator of the row before the chain, :mod:`..spec_nvfp4`) or ``none`` (the unscaled oracle chains: the raw fold in
fp32 / bf16, what the tests compare with the dense class). The chains are :func:`..spec_nvfp4.glu_scaled_elem` on the
interleaved (gate, up) pair, :func:`..spec_nvfp4.act_scaled_elem` on one accumulator and :func:`..spec_nvfp4.scaled_out_elem`
(GEMM2), every op in fp32 through ``MoeF32``.

Two packings of the A rows reach the epilogue (:mod:`.kernel_nvfp4`): the padded one (the group's rows start at a 128-row
scale-factor atom; ``pts`` indexed by the output row) and the dense one (``dense_a``: the rows back to back, the row scales
at the padded positions ``row + off_pad[e] - off[e]``, read through the two group-offset tables). Either way a row's chain
is computed by one thread from the same accumulator value in the same fp32 sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Boolean, Int32, Int64, Uint8, Uint32
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils.blackwell_helpers import get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ....batch_invariant.quantization.cvt_fp8 import (
    cvt_f32_to_e4m3,
    cvt_f32x8_to_e2m1x8,
)
from ....gemm.batch_invariant.blocks.operands import ACC_DTYPE
from ..spec_nvfp4 import (
    NVFP4_BLOCK,
    NVFP4_C_SIXTH,
    act_scaled_elem,
    act_scaled_pre,
    glu_scaled_elem,
    glu_scaled_pre,
    nvfp4_block_amax,
    scaled_out_elem,
    w4a4_row_scale,
)
from .epilogue import (
    BAR_EPI_STAGE,
    GROUP,
    GroupedChain,
    MoeF32,
    _chain_elem,
    fragment_offsets,
    store_plan,
)

SCALES = ("none", "w4a4", "w4a4s")  # w4a4s: the static class's GEMM2 scale mul(alpha[e], inv2[e]), once per tile
#: The chain's output formats: the bf16 class's, the f32 oracle, and ``nvfp4`` -- the static class's GEMM1, whose epilogue
#: quantizes the fp32 chain value per 16-column block (codes and scale bytes out).
OUT_FMTS_NVFP4 = ("bf16", "f32", "nvfp4")


@dataclass(frozen=True)
class W4A4Chain(GroupedChain):
    """A grouped chain with the NVFP4 classes' scale source: ``w4a4`` (the row scale ``mul(alpha[e], pts[row])`` before the
    chain), ``w4a4s`` (the static class's GEMM2: ``mul(alpha[e], inv2[e])`` once per tile) or ``none`` (the unscaled oracle
    chain); the output format ``nvfp4`` is the static class's GEMM1 (the fused block quantization). ``describe()`` keys the
    kernel cache; the scaled chains carry their tag so they never share an object with the bf16 class's chains."""

    scale: str = "w4a4"

    def __post_init__(self):
        assert self.scale in SCALES, self
        if self.out_fmt == "nvfp4":
            # the static class's GEMM1: validated as the bf16 chain of the same kind, then the output format replaced
            object.__setattr__(self, "out_fmt", "bf16")
            super().__post_init__()
            object.__setattr__(self, "out_fmt", "nvfp4")
            assert self.kind in ("glu", "act") and self.scale == "w4a4", self
        else:
            super().__post_init__()
        if self.scale == "w4a4s":
            assert self.kind == "plain" and self.out_fmt == "bf16", self

    @property
    def static_quant(self) -> bool:
        """The static class's GEMM1: the fused block quantization of the fp32 chain value."""
        return self.out_fmt == "nvfp4"

    @property
    def needs_aux(self) -> bool:
        """The per-expert static scales the chain reads (``gs2[e]`` for the fused quantization, ``inv2[e]`` for GEMM2)."""
        return self.out_fmt == "nvfp4" or self.scale == "w4a4s"

    def describe(self) -> str:
        base = super().describe()  # the output format names the static class's GEMM1 (`..._nvfp4`)
        return base if self.scale == "none" else f"{self.scale}_{base}"


def _scaled_elem(chain: W4A4Chain, acc, acc_gate, s):
    """One output element of the W4A4 chain: the gated term of the scaled (gate, up) pair, the ungated activation of the
    scaled accumulator, or GEMM2's ``cvt(mul(acc, s))``; the unscaled chains are the bf16 class's; the static class's GEMM1
    returns the fp32 value before the block quantization (the caller forms the blocks)."""
    fp = MoeF32
    if chain.scale == "none":
        return _chain_elem(chain, acc, acc_gate)
    if chain.out_fmt == "nvfp4":
        if chain.pair:
            return glu_scaled_pre(fp, acc_gate, acc, s, chain.act, chain.params)
        return act_scaled_pre(fp, acc, s, chain.act, chain.params)
    if chain.pair:
        return glu_scaled_elem(
            fp, acc_gate, acc, s, chain.act, chain.out_fmt, chain.params
        )
    if chain.kind == "act":
        return act_scaled_elem(fp, acc, s, chain.act, chain.out_fmt, chain.params)
    return scaled_out_elem(fp, acc, s, chain.out_fmt)


def static_block_plan(offs: List[Tuple[int, int]], pair: bool) -> List[List[int]]:
    """The static class's blocks of one thread's fragment: for every (row, 16-output block) the 16 registers in output-column
    order (the gated chain's up positions ``16q + r`` -> output ``8q + r``; the ungated chain's every column). Raises when the
    fragment does not hold whole blocks (such a record is not in the static class's space)."""
    key = (lambda n: (n // (2 * GROUP)) * GROUP + n % GROUP) if pair else (lambda n: n)
    groups: dict = {}
    for v, (m, n) in enumerate(offs):
        if pair and n % (2 * GROUP) >= GROUP:
            continue
        groups.setdefault((m, key(n) // NVFP4_BLOCK), []).append((key(n) % NVFP4_BLOCK, v))
    blocks = []
    for (m, b), lst in sorted(groups.items()):
        if sorted(pos for pos, _ in lst) != list(range(NVFP4_BLOCK)):
            raise ValueError(
                f"the fragment holds a partial 16-output block at row {m}, block {b}: {offs}"
            )
        blocks.append([v for _, v in sorted(lst)])
    return blocks


@cute.jit
def sf_byte_offset(row: Int32, col: Int32, pcols: Int32) -> Int64:
    """The byte of block column ``col`` of row ``row`` in the 128x4 swizzled scale-factor layout (``sf_offset_128x4`` on device
    values)."""
    r = row.to(Int64)
    r128 = (r % 128).to(Int32)
    inner = (
        (col % Int32(4))
        + (col // Int32(4)) * Int32(512)
        + (r128 % 32) * 16
        + (r128 // 32) * 4
    )
    return (r // 128) * (Int32(128) * pcols).to(Int64) + inner.to(Int64)


@cute.jit
def static_block_quant(vals, gs6, inv_gs):
    """One block of the static class's GEMM1 output from its 16 fp32 values: ``(sf_byte, w0, w1)`` -- the e4m3 scale byte
    (``cvt(mul(amax, mul(gs, 1/6)))``, ``gs6 = mul(gs, 1/6)`` formed once per tile) and the 16 e2m1 codes as two little-endian
    words (``cvt(mul(v, inv))``, ``inv = fmin(div(1, mul(sf_d, div(1, gs))), FLT_MAX)``, ``inv_gs = div(1, gs)`` formed once per
    tile): the per-tensor quantizer's block chain, every op in fp32 through ``MoeF32``."""
    fp = MoeF32
    amax = nvfp4_block_amax(fp, vals)
    sf_byte, sf_d = cvt_f32_to_e4m3(fp.mul(amax, gs6))
    inv = fp.fmin(
        fp.div(fp.const(1.0), fp.mul(sf_d, inv_gs)), fp.const(3.4028234663852886e38)
    )
    ys = [fp.mul(v, inv) for v in vals]
    return sf_byte, cvt_f32x8_to_e2m1x8(*ys[:8]), cvt_f32x8_to_e2m1x8(*ys[8:])


@cute.jit
def grouped_w4a4_epilogue_loop(
    st,
    chain: cutlass.Constexpr,
    tidx: Int32,
    tile_sched,
    tCtAcc_base: cute.Tensor,
    thr_mma,
    mOut: cute.Tensor,
    Tk: Int32,
    N: Int32,
    epi_tile: cute.Tile,
    acc_pipeline,
    mma_m_in_cluster,
    mAlpha: cute.Tensor,
    mPts: cute.Tensor,
    epi_warpgroups: cutlass.Constexpr,
    s_off,
    s_off_pad,
    sC: cute.Tensor,
    cta_v: Int32,
    mOutSf: cute.Tensor = None,
    mAux: cute.Tensor = None,
):
    """``mAlpha``: the per-expert alpha (as given); ``mPts``: the per-row scale of the A operand at the padded positions;
    ``s_off`` / ``s_off_pad``: the dense and the padded group offsets (``dense_a``: the row scale of output row ``p`` of
    expert ``e`` sits at ``p + off_pad[e] - off[e]``; the padded packing passes ``s_off_pad = None``). ``sC``: the staging
    buffers (one epilogue subtile per warpgroup); ``cta_v``: this CTA's index inside its MMA (0, or 0 / 1 for the 2-CTA
    instructions), which with ``mma_m_in_cluster`` locates its rows in the cluster tile. The 16-bit outputs take the staged
    row-coalesced store, the f32 oracle output the direct vector stores (as :func:`.epilogue.grouped_epilogue_loop`). The
    static class's GEMM1 (``chain.out_fmt == "nvfp4"``): ``mOut`` is the packed-code buffer ``[rows, I/2]`` (uint8), ``mOutSf`` the
    flat 128x4 scale-factor buffer at the (padded) row positions, ``mAux[e]`` the expert's static scale ``gs2``; the thread forms
    every 16-output block of its row from its own registers (:func:`static_block_plan`) and stores 8 code bytes and 1 scale byte
    per block. GEMM2 of that class (``chain.scale == "w4a4s"``): ``s = mul(alpha[e], mAux[e])``, ``mAux[e] = inv2[e]``, once per
    tile."""
    pair = chain.pair
    scaled = chain.scale != "none"
    static_q = chain.out_fmt == "nvfp4"
    out_dt = mOut.element_type
    wg = tidx // 128
    tidx = tidx % 128
    warp_row0 = Int32(32) * ((tidx // Int32(32)) % Int32(4))
    vec_elems = 16 // (out_dt.width // 8)
    ld = cute.assume(mOut.layout.stride[0], divby=vec_elems)
    tiler_mn = cute.slice_(st.mma_tiler, (None, None, 0))
    if cutlass.const_expr(pair):
        col_shape = (GROUP, 2, N // (2 * GROUP))
        mC = cute.make_tensor(
            mOut.iterator,
            cute.make_layout((Tk, col_shape, 1), stride=(ld, (1, 0, GROUP), 0)),
        )
    else:
        col_shape = N
        mC = cute.make_tensor(
            mOut.iterator, cute.make_layout((Tk, N, 1), stride=(ld, 1, 0))
        )
    idC = cute.make_identity_tensor((Tk, col_shape, 1))
    cC = cute.local_tile(idC, tiler_mn, (None, None, None))
    tCcC = transform_partitioned_tensor_layout(thr_mma.partition_C(cC))
    idF = cute.make_identity_tensor((Tk, N, 1))
    cF = cute.local_tile(idF, tiler_mn, (None, None, None))
    tCcF = transform_partitioned_tensor_layout(thr_mma.partition_C(cF))
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    copy_atom_t2r = get_tmem_load_op(
        st.cta_tile_shape_mnk,
        st.c_layout,
        st.c_dtype,
        ACC_DTYPE,
        epi_tile,
        st.use_2cta_instrs,
    )
    tAcc_epi = cute.flat_divide(tCtAcc, epi_tile)
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
    )
    thr_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc_base = thr_t2r.partition_S(tAcc_epi)
    cC_epi = cute.flat_divide(tCcC, epi_tile)
    cF_epi = cute.flat_divide(tCcF, epi_tile)
    frag = thr_t2r.partition_D(cF_epi)[(None, None, None, 0, 0, 0, 0, 0)]
    offs = fragment_offsets(frag.layout)
    ve, partner, is_out = store_plan(offs, pair, 16 // (out_dt.width // 8))
    nv = len(offs)
    rows_in_frag = sorted({m for m, _ in offs})
    tTR_rAcc = cute.make_rmem_tensor(frag.shape, ACC_DTYPE)
    tTR_rC = cute.make_rmem_tensor(frag.shape, out_dt)
    tTR_rC.fill(0)
    atom_r2g = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), out_dt, num_bits_per_copy=ve * out_dt.width
    )
    tiled_copy_r2g = cute.make_tiled_copy_D(atom_r2g, tiled_copy_t2r)
    thr_r2g = tiled_copy_r2g.get_slice(tidx)
    tRG_rC = tiled_copy_r2g.retile(tTR_rC)
    tRG_cC_all = thr_r2g.partition_D(cC_epi)
    nvec = cute.size(tRG_rC.shape[0][1])
    n1, n2 = cute.size(tRG_rC.shape[1]), cute.size(tRG_rC.shape[2])
    staged = sC is not None and out_dt.width == 16
    tn = st.mma_tiler[1]
    cta_m = st.cta_tile_shape_mnk[0]
    epi_m = epi_n = CH = RS = SM = SN = 1
    N_out = N
    if cutlass.const_expr(staged):
        epi_m, epi_n = cute.size(epi_tile[0]), cute.size(epi_tile[1])
        n_lay = epi_tile[1]
        nested_n = cute.rank(n_lay) == 2
        P = cute.size(n_lay.shape[0]) if nested_n else epi_n
        NP = cute.size(n_lay.shape[1]) if nested_n else 1
        S = int(n_lay.stride[1]) if nested_n else 0
        step = 2 * GROUP if pair else GROUP
        CH = P // step
        RS = 128 // CH
        assert (
            P % step == 0
            and 128 % CH == 0
            and epi_m % RS == 0
            and epi_m == cta_m
            and epi_n == P * NP
        ), (epi_tile, step, cta_m)
        N_out = N // 2 if pair else N
        tiled_copy_r2s = cute.make_tiled_copy_D(atom_r2g, tiled_copy_t2r)
        thr_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        sC_wg = sC[(None, None, wg)]
        atom_s2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), out_dt, num_bits_per_copy=GROUP * out_dt.width
        )
        tiled_s2g = cute.make_tiled_copy_tv(
            atom_s2g,
            cute.make_layout((RS, CH), stride=(CH, 1)),
            cute.make_layout((1, GROUP)),
        )
        thr_s2g = tiled_s2g.get_slice(tidx)
        sel = cute.make_layout((epi_m, (GROUP, CH)), stride=(1, (epi_m, step * epi_m)))
        tS_sC = [
            thr_s2g.partition_S(
                cute.composition(cute.local_tile(sC_wg, (epi_m, P), (0, pc)), sel)
            )
            for pc in range(NP)
        ]
        cO = cute.make_identity_tensor((epi_m, GROUP * CH))
        tS_cO = thr_s2g.partition_D(cO)
        SM, SN = cute.size(tS_cO.shape[1]), cute.size(tS_cO.shape[2])
        mOutFlat = cute.make_tensor(
            mOut.iterator, cute.make_layout((Tk, N_out), stride=(ld, 1))
        )
    blocks = []
    mOut32 = None
    n_blocks = pcols = Int32(0)
    tTR_cF_all = thr_t2r.partition_D(cF_epi)  # the flat (row, accumulator column) coordinates
    if cutlass.const_expr(static_q):
        assert len(rows_in_frag) == 1 and not staged, (
            "the static class's fused quantization takes the 128-row layouts (thread = row), unstaged"
        )
        blocks = static_block_plan(offs, pair)
        N_out = N // 2 if pair else N
        n_blocks = N_out // Int32(NVFP4_BLOCK)  # scale blocks per output row
        pcols = ((n_blocks + Int32(3)) // Int32(4)) * Int32(4)  # scale columns of the 128x4 buffer
        ld32 = cute.assume(ld // Int32(4), divby=4)  # the code row in 32-bit words (16-byte rows)
        mOut32 = cute.make_tensor(
            cute.recast_ptr(mOut.iterator, dtype=Uint32),
            cute.make_layout((Tk, n_blocks * Int32(2)), stride=(ld32, 1)),
        )
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, st.num_acc_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        row0, row_end, n_tile, e = (
            work_tile.row0,
            work_tile.row_end,
            work_tile.n_tile,
            work_tile.e,
        )
        pts_shift = Int32(0)
        if cutlass.const_expr(s_off_pad is not None):
            pts_shift = (
                s_off_pad[e] - s_off[e]
            )  # dense A: the row scales sit at the padded positions
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        rows_left = row_end - row0
        alpha_e = None
        gs6 = inv_gs = None
        if cutlass.const_expr(chain.scale == "w4a4s"):
            alpha_e = w4a4_row_scale(MoeF32, mAlpha[e], mAux[e])  # mul(alpha2[e], inv2[e]): the static class's GEMM2 scale
        elif cutlass.const_expr(scaled):
            alpha_e = mAlpha[e]  # the tile's expert alpha, as given
            if cutlass.const_expr(static_q):
                # the expert's static intermediate scale: mul(gs, 1/6) and div(1, gs) once per tile (the block chain's own ops)
                gs_e = mAux[e]
                gs6 = MoeF32.mul(gs_e, MoeF32.const(NVFP4_C_SIXTH))
                inv_gs = MoeF32.div(MoeF32.const(1.0), gs_e)
        gC = cute.local_tile(
            cute.domain_offset((row0, 0, 0), mC), tiler_mn, (None, None, None)
        )
        tCgC = transform_partitioned_tensor_layout(thr_mma.partition_C(gC))
        gC_epi = cute.flat_divide(tCgC, epi_tile)
        tRG_gC = thr_r2g.partition_D(gC_epi)[
            (None, None, None, None, None, mma_m_in_cluster, n_tile, 0)
        ]
        tRG_cC = tRG_cC_all[(None, None, None, None, None, mma_m_in_cluster, n_tile, 0)]
        tRG_gC = cute.group_modes(tRG_gC, 3, cute.rank(tRG_gC))
        tRG_cC = cute.group_modes(tRG_cC, 3, cute.rank(tRG_cC))
        if cutlass.const_expr(static_q):
            tTR_cF_tile = tTR_cF_all[
                (None, None, None, None, None, mma_m_in_cluster, n_tile, 0)
            ]
            tTR_cF_tile = cute.group_modes(tTR_cF_tile, 3, cute.rank(tTR_cF_tile))
        cta_row_rel = Int32(0)
        if cutlass.const_expr(staged):
            cta_row_rel = mma_m_in_cluster * Int32(st.mma_tiler[0]) + cta_v * Int32(
                cta_m
            )
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_state.index)]
        acc_pipeline.consumer_wait(acc_state)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        # a warp whose 32 rows all lie past the group's real end has nothing to store (warp-uniform; 128-row CTA tiles)
        warp_active = Boolean(True)
        if cutlass.const_expr(
            len(rows_in_frag) == 1 and st.cta_tile_shape_mnk[0] == 128
        ):
            warp_active = rows_left > warp_row0
        for s in range(wg, subtile_cnt, epi_warpgroups):
            if warp_active:
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
                cC_sub = tRG_cC[(None, None, None, s)]
                # the row scale once per distinct row of the fragment: s = mul(alpha_e, pts[row]); a tile past the padded end
                # reads a valid row and stores nothing (the static class's GEMM2: the per-tile scale for every row)
                row_scale = {}
                if cutlass.const_expr(chain.scale == "w4a4s"):
                    for m in rows_in_frag:
                        row_scale[m] = alpha_e
                elif cutlass.const_expr(scaled):
                    crd0 = cC_sub[((0, 0), 0, 0)]
                    for m in rows_in_frag:
                        p = row0 + pts_shift + (crd0[0] - Int32(offs[0][0]) + Int32(m))
                        p = cutlass.min(
                            cutlass.max(p, Int32(0)), mPts.shape[0] - Int32(1)
                        )
                        row_scale[m] = w4a4_row_scale(MoeF32, alpha_e, mPts[p])
                if cutlass.const_expr(static_q):
                    # the static class: the fp32 chain values, then every 16-output block of the row quantized from this thread's
                    # registers; 8 code bytes to the packed-code row, the scale byte to the (padded) row's 128x4 position
                    crdF0 = tTR_cF_tile[(None, None, None, s)][(0, 0, 0)]
                    pre = [None] * nv
                    for v in cutlass.range_constexpr(nv):
                        if cutlass.const_expr(is_out[v]):
                            pre[v] = _scaled_elem(
                                chain,
                                tTR_rAcc[v],
                                tTR_rAcc[partner[v]] if cutlass.const_expr(pair) else None,
                                row_scale[offs[v][0]],
                            )
                    for blk in blocks:
                        rrow = crdF0[0] - Int32(offs[0][0]) + Int32(offs[blk[0]][0])
                        acc_col = crdF0[1] - Int32(offs[0][1]) + Int32(offs[blk[0]][1])
                        out_col = (
                            (acc_col // Int32(2 * GROUP)) * Int32(GROUP) + acc_col % Int32(GROUP)
                            if cutlass.const_expr(pair)
                            else acc_col
                        )
                        bcol = out_col // Int32(NVFP4_BLOCK)
                        if (rrow < rows_left) & (bcol < n_blocks):
                            sf_byte, w0, w1 = static_block_quant(
                                [pre[v] for v in blk], gs6, inv_gs
                            )
                            prow = row0 + rrow
                            mOut32[(prow, Int32(2) * bcol)] = w0
                            mOut32[(prow, Int32(2) * bcol + Int32(1))] = w1
                            mOutSf[sf_byte_offset(prow + pts_shift, bcol, pcols)] = sf_byte.to(Uint8)
                else:
                    for v in cutlass.range_constexpr(nv):
                        if cutlass.const_expr(is_out[v]):
                            sv = (
                                row_scale[offs[v][0]]
                                if cutlass.const_expr(scaled)
                                else None
                            )
                            tTR_rC[v] = _scaled_elem(
                                chain,
                                tTR_rAcc[v],
                                tTR_rAcc[partner[v]] if cutlass.const_expr(pair) else None,
                                sv,
                            )
                if cutlass.const_expr(staged):
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, wg)])
                elif cutlass.const_expr(not static_q):
                    gC_sub = tRG_gC[(None, None, None, s)]
                    for j in cutlass.range_constexpr(nvec):
                        for mi in cutlass.range_constexpr(n1):
                            for ni in cutlass.range_constexpr(n2):
                                if cutlass.const_expr(is_out[j * ve]):
                                    crd = cC_sub[((0, j), mi, ni)]
                                    if cute.elem_less(crd, (rows_left, col_shape, 1)):
                                        cute.copy(
                                            atom_r2g,
                                            tRG_rC[((None, j), mi, ni)],
                                            gC_sub[((None, j), mi, ni)],
                                        )
            if cutlass.const_expr(staged):
                cute.arch.barrier(
                    barrier_id=Int32(BAR_EPI_STAGE) + wg, number_of_threads=128
                )
                for pc in cutlass.range_constexpr(NP):
                    acc_col0 = n_tile * Int32(tn) + s * Int32(P) + Int32(pc * S)
                    out_col0 = (
                        acc_col0 // Int32(2) if cutlass.const_expr(pair) else acc_col0
                    )
                    out_col0 = cute.assume(out_col0, divby=GROUP)
                    gO = cute.local_tile(
                        cute.domain_offset((row0 + cta_row_rel, out_col0), mOutFlat),
                        (epi_m, GROUP * CH),
                        (0, 0),
                    )
                    tS_gO = thr_s2g.partition_D(gO)
                    for i in cutlass.range_constexpr(SM):
                        for j in cutlass.range_constexpr(SN):
                            crd = tS_cO[(0, i, j)]
                            if (cta_row_rel + crd[0] < rows_left) & (
                                out_col0 + crd[1] < N_out
                            ):
                                cute.copy(
                                    atom_s2g,
                                    tS_sC[pc][(None, i, j)],
                                    tS_gO[(None, i, j)],
                                )
                cute.arch.barrier(
                    barrier_id=Int32(BAR_EPI_STAGE) + wg, number_of_threads=128
                )
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_state)
        acc_state.advance()
