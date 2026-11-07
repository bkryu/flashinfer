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

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Boolean, Int32
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils.blackwell_helpers import get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ....gemm.batch_invariant.blocks.operands import ACC_DTYPE
from ..spec_nvfp4 import (
    act_scaled_elem,
    glu_scaled_elem,
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

SCALES = ("none", "w4a4")


@dataclass(frozen=True)
class W4A4Chain(GroupedChain):
    """A grouped chain with the NVFP4 class's scale source: ``w4a4`` (the row scale ``mul(alpha[e], pts[row])`` before
    the chain) or ``none`` (the unscaled oracle chain). ``describe()`` keys the kernel cache; the scaled chains carry the
    ``w4a4`` tag so they never share an object with the bf16 class's chains."""

    scale: str = "w4a4"

    def __post_init__(self):
        super().__post_init__()
        assert self.scale in SCALES, self

    def describe(self) -> str:
        base = super().describe()
        return base if self.scale == "none" else f"w4a4_{base}"


def _scaled_elem(chain: W4A4Chain, acc, acc_gate, s):
    """One output element of the W4A4 chain: the gated term of the scaled (gate, up) pair, the ungated activation of the
    scaled accumulator, or GEMM2's ``cvt(mul(acc, s))``; the unscaled chains are the bf16 class's."""
    fp = MoeF32
    if chain.scale == "none":
        return _chain_elem(chain, acc, acc_gate)
    if chain.pair:
        return glu_scaled_elem(
            fp, acc_gate, acc, s, chain.act, chain.out_fmt, chain.params
        )
    if chain.kind == "act":
        return act_scaled_elem(fp, acc, s, chain.act, chain.out_fmt, chain.params)
    return scaled_out_elem(fp, acc, s, chain.out_fmt)


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
):
    """``mAlpha``: the per-expert alpha (as given); ``mPts``: the per-row scale of the A operand at the padded positions;
    ``s_off`` / ``s_off_pad``: the dense and the padded group offsets (``dense_a``: the row scale of output row ``p`` of
    expert ``e`` sits at ``p + off_pad[e] - off[e]``; the padded packing passes ``s_off_pad = None``). ``sC``: the staging
    buffers (one epilogue subtile per warpgroup); ``cta_v``: this CTA's index inside its MMA (0, or 0 / 1 for the 2-CTA
    instructions), which with ``mma_m_in_cluster`` locates its rows in the cluster tile. The 16-bit outputs take the staged
    row-coalesced store, the f32 oracle output the direct vector stores (as :func:`.epilogue.grouped_epilogue_loop`)."""
    pair = chain.pair
    scaled = chain.scale != "none"
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
        if cutlass.const_expr(scaled):
            alpha_e = mAlpha[e]  # the tile's expert alpha, as given
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
                # reads a valid row and stores nothing
                row_scale = {}
                if cutlass.const_expr(scaled):
                    crd0 = cC_sub[((0, 0), 0, 0)]
                    for m in rows_in_frag:
                        p = row0 + pts_shift + (crd0[0] - Int32(offs[0][0]) + Int32(m))
                        p = cutlass.min(
                            cutlass.max(p, Int32(0)), mPts.shape[0] - Int32(1)
                        )
                        row_scale[m] = w4a4_row_scale(MoeF32, alpha_e, mPts[p])
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
                else:
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
