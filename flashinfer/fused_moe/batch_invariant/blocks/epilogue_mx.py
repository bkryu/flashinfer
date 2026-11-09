# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 class's fused requantization: the grouped GEMM1 epilogue that writes the ``mxfp8.quant`` class's codes and block
scales of the bf16-rounded chain output instead of the bf16 output itself (the fused requantization).

The term is the standalone quantizer's, applied to the same values: the chain's output rounded to bf16 (the class's
intermediate), widened exactly to fp32; the block amax of 32 consecutive output columns by the quantizer's canonical pairwise
``fmaxc`` tree over ``fabs`` (``mx_spec.row_reduction``'s ``elem`` digit: one thread holds the whole block, so the tree is
in-thread); the scale chain (``scale_norm`` -> ``cvt_f32_to_ue8m0`` -> ``inv_scale``) and the element chain (``quant_pre`` ->
``saturate`` -> the declared e4m3 conversion, ``pack_block_codes``) of ``batch_invariant.quantization.mx_spec`` / ``.blocks``. A placement
of the standalone pass: the same bytes on the live rows (asserted bit for bit by the suites), no bf16 intermediate in memory.

The placement: the 128-row TMEM load hands each epilogue thread one row of the tile (thread = TMEM lane = tile row) and the
subtile's accumulator columns; the gated chain's (gate, up) pairs are interleaved by ``GROUP`` so a block of 32 outputs is
the output of ``32 // outs_per_subtile`` consecutive subtiles of the same thread (two 32-column subtiles for the gated chain,
one for the ungated). The subtiles go round-robin to the epilogue warpgroups as in the bf16 epilogue, so every warpgroup
works on the 128-wide tiles too; the two subtiles of a gated block land in a pair of warpgroups, which exchange their
half-block amax through shared memory (one named barrier per block; the canonical tree over 32 leaves is the combine of the
two halves' trees, and the max of non-negative values is exact in every order) and each quantizes its 16 outputs under the
shared scale. Codes: one 16-byte vector per thread per subtile into ``out [Tk, N_out]`` (uint8 e4m3 bytes); the scale: one
byte into the 128x4 swizzled layout of ``(Tk, N_out)`` (``sf_offset_128x4_dev``), stored by the block's first half. Rows
past the group's real end and blocks past ``N_out`` are not stored.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils.blackwell_helpers import get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ....batch_invariant.canon import _pairwise
from ....batch_invariant.quantization.blocks import (
    cvt_f32_to_ue8m0,
    pack_block_codes,
    sf_offset_128x4_dev,
)
from ....batch_invariant.quantization.cute_ext import CuteF32
from ....batch_invariant.quantization.layout import SF_COL_TILE
from ....batch_invariant.quantization.mx_spec import (
    BLOCK,
    DEFAULT_RULE,
    MXFP8,
    elem_digit,
    inv_scale,
    leaf_value,
    quant_pre,
    saturate,
    scale_norm,
)
from ....gemm.batch_invariant.blocks.operands import ACC_DTYPE
from .epilogue import GROUP, GroupedChain, _chain_elem, fragment_offsets, store_plan

OUT_FMT = "mxfp8"  # the GroupedChain out_fmt of this epilogue: e4m3 codes + UE8M0 block scales of the bf16-rounded chain
VEC_BYTES = 16
XCH_BUFFERS = 2  # the half-block amax exchange of the gated chain, double-buffered by block parity (one barrier per block)
BAR_EPI_XCH = 8  # named barrier ids BAR_EPI_XCH + warpgroup pair (the staged store uses BAR_EPI_STAGE + warpgroup)
BAR_EPI_JOIN = (
    3  # every epilogue warpgroup once, after its last TMEM load, before the TMEM free
)


def xch_layout(epi_warpgroups: int):
    """The shared-memory exchange of the gated chain's half-block amax: fp32 per (buffer, epilogue warpgroup, tile row)."""
    return cute.make_layout((XCH_BUFFERS, epi_warpgroups, 128))


def xch_bytes(chain: GroupedChain, epi_warpgroups: int) -> int:
    return (
        4 * XCH_BUFFERS * epi_warpgroups * 128
        if chain.out_fmt == OUT_FMT and chain.pair
        else 0
    )


def bf16_twin(chain: GroupedChain) -> GroupedChain:
    """The bf16 chain whose rounded output this epilogue quantizes."""
    assert chain.out_fmt == OUT_FMT, chain
    return GroupedChain(chain.kind, chain.act, "bf16", chain.params)


def block_amax(fp, ys):
    """The quantizer's block reduction on one thread's values: ``fabs`` leaves, the ``elem`` digit's pairwise ``fmaxc`` tree
    (the canonical order of ``mx_spec.row_reduction``, which every placement of the standalone kernel realises); on a half
    block (16 values) the subtree whose combine with the other half's is the block's tree."""
    d = elem_digit()
    return _pairwise([leaf_value(fp, y) for y in ys], lambda a, b: d.combine(fp, a, b))


@cute.jit
def grouped_epilogue_quant_loop(
    st,
    chain: cutlass.Constexpr,
    tidx: Int32,
    tile_sched,
    tCtAcc_base: cute.Tensor,
    thr_mma,
    mOut: cute.Tensor,
    mSf: cute.Tensor,
    Tk: Int32,
    N: Int32,
    epi_tile: cute.Tile,
    acc_pipeline,
    mma_m_in_cluster,
    epi_warpgroups: cutlass.Constexpr = 1,
    sX: cute.Tensor = None,
    s_off=None,
    s_off_pad=None,
    cta_v: Int32 = None,
):
    """``mOut [Tk, N_out]`` uint8 (the e4m3 codes), ``mSf`` the flat uint8 scale bytes in the 128x4 layout of ``(Tk, N_out)``
    (of the padded rows: with the dense packing, ``s_off`` / ``s_off_pad`` give each group's shift from its dense to its padded
    position and the scale byte of row ``r`` lands at padded row ``r + shift``); ``sX`` the exchange buffers (``xch_layout``, fp32)
    of the gated chain (None for the ungated). 128-row CTA tiles only (thread = tile row); ``cta_v``: this CTA's index inside its
    MMA (0 / 1 for the 2-CTA instructions)."""
    pair = chain.pair
    twin = bf16_twin(chain)
    fp = CuteF32
    rule, fmt = DEFAULT_RULE, MXFP8
    wg = tidx // 128  # the epilogue warpgroup (warp-uniform)
    tidx = tidx % 128
    warp_row0 = Int32(32) * (
        (tidx // Int32(32)) % Int32(4)
    )  # the first tile row of this warp's TMEM lanes
    tiler_mn = cute.slice_(st.mma_tiler, (None, None, 0))
    N_out = N // Int32(2) if cutlass.const_expr(pair) else N
    nb = N_out // Int32(BLOCK)  # scale columns of a row
    pcols = ((nb + Int32(SF_COL_TILE - 1)) // Int32(SF_COL_TILE)) * Int32(SF_COL_TILE)
    # the flat accumulator column space: per-register (row, column) offsets of the fragment, the chain's pairing
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
    tAcc_epi = cute.flat_divide(tCtAcc, epi_tile)  # (EPI_M, EPI_N, SUB_M, SUB_N, STAGE)
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
    )
    thr_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc_base = thr_t2r.partition_S(
        tAcc_epi
    )  # (T2R, T2R_M, T2R_N, SUB_M, SUB_N, STAGE)
    cF_epi = cute.flat_divide(
        tCcF, epi_tile
    )  # (EPI_M, EPI_N, SUB_M, SUB_N, RestM, RestN, RestL)
    tTR_cF_all = thr_t2r.partition_D(cF_epi)  # the registers' coordinates
    frag = tTR_cF_all[(None, None, None, 0, 0, 0, 0, 0)]
    offs = fragment_offsets(frag.layout)
    _, partner_reg, is_out = store_plan(offs, pair, 16)
    nv = len(offs)
    rows_in_frag = sorted({m for m, _ in offs})
    assert len(rows_in_frag) == 1 and st.cta_tile_shape_mnk[0] == 128, (
        "the fused requantization takes the 128-row layouts",
        offs,
    )
    P = cute.size(epi_tile[1])  # accumulator columns per subtile
    V = P // 2 if pair else P  # this thread's outputs per subtile
    assert BLOCK % V == 0, (BLOCK, V)
    SB = BLOCK // V  # subtiles per output block (2 gated, 1 ungated)
    assert epi_warpgroups % SB == 0, (
        epi_warpgroups,
        SB,
    )  # a block's subtiles land in SB consecutive warpgroups
    assert SB == 1 or sX is not None
    # register -> output column within the subtile (a comprehension: the DSL's preprocessor turns a plain `if` into a region)
    out_col = {
        v: ((n // (2 * GROUP)) * GROUP + n % GROUP if pair else n)
        for v, (m, n) in enumerate(offs)
        if is_out[v]
    }
    assert sorted(out_col.values()) == list(range(V)), out_col
    tTR_rAcc = cute.make_rmem_tensor(frag.shape, ACC_DTYPE)
    ys_r = cute.make_rmem_tensor(
        cute.make_layout((V,)), Float32
    )  # the subtile's bf16-rounded chain values, widened
    fy_words = cute.make_rmem_tensor(
        cute.make_layout((V // 4,)), Uint32
    )  # the subtile's code bytes as words
    fy = cute.make_tensor(
        cute.recast_ptr(fy_words.iterator, dtype=Uint8), cute.make_layout((V,))
    )
    atom16 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), Uint8, num_bits_per_copy=8 * VEC_BYTES
    )
    tn = st.mma_tiler[1]
    blocks_per_tile = tn // (BLOCK * (2 if pair else 1))
    half = wg % Int32(SB)  # this warpgroup's half of its blocks (0 ungated)
    pair_id = wg // Int32(SB)
    partner = wg ^ Int32(1) if cutlass.const_expr(SB == 2) else wg
    parity = Int32(0)
    d = elem_digit()
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
        sf_shift = Int32(0)
        if cutlass.const_expr(s_off_pad is not None):
            sf_shift = (
                s_off_pad[e] - s_off[e]
            )  # dense A: the group's padded (atom-aligned) position
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        rows_left = row_end - row0
        tTR_cF = tTR_cF_all[(None, None, None, None, None, mma_m_in_cluster, n_tile, 0)]
        tTR_cF = cute.group_modes(tTR_cF, 3, cute.rank(tTR_cF))
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_state.index)]
        acc_pipeline.consumer_wait(acc_state)  # the fold of this tile is complete
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        # a warp whose 32 rows all lie past the group's real end skips its subtiles (warp-uniform); the warp's rows are
        # counted from the cluster tile's first row, as the store predicate is (the peer CTA of a 2-CTA pair and the later
        # CTAs of a cluster along M sit past the first 128 rows: the bf16 epilogue's rule)
        cta_row_rel = Int32(0)
        if cutlass.const_expr(cta_v is not None):
            cta_row_rel = mma_m_in_cluster * Int32(st.mma_tiler[0]) + cta_v * Int32(
                st.cta_tile_shape_mnk[0]
            )
        warp_active = rows_left > cta_row_rel + warp_row0
        for s in range(
            wg, subtile_cnt, epi_warpgroups
        ):  # this warpgroup's subtiles (half blocks when gated)
            blk = n_tile * Int32(blocks_per_tile) + s // Int32(SB)
            crd0 = tTR_cF[(None, None, None, s)][
                0
            ]  # register 0's (row, column) of the subtile
            row_rel = crd0[0] - Int32(
                offs[0][0]
            )  # the thread's row inside the cluster tile
            amax = Float32(0.0)
            if warp_active:
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
                # the chain, rounded to bf16 (the class's intermediate), widened exactly: the subtile's outputs in column order
                for v in cutlass.range_constexpr(nv):
                    if cutlass.const_expr(is_out[v]):
                        g = (
                            tTR_rAcc[partner_reg[v]]
                            if cutlass.const_expr(pair)
                            else None
                        )
                        yb = _chain_elem(twin, tTR_rAcc[v], g)
                        ys_r[out_col[v]] = yb.to(Float32)
                amax = block_amax(fp, [ys_r[j] for j in range(V)])
                if cutlass.const_expr(SB == 2):
                    sX[(parity, wg, tidx)] = amax
            if cutlass.const_expr(SB == 2):
                cute.arch.barrier(
                    barrier_id=Int32(BAR_EPI_XCH) + pair_id, number_of_threads=128 * SB
                )  # the pair's halves are in
                if warp_active:
                    amax = d.combine(
                        fp, amax, sX[(parity, partner, tidx)]
                    )  # the block's tree: the combine of the two halves'
                parity = Int32(1) - parity
            if warp_active:
                # the quantizer class's term on the block (the scale from the block amax, the codes of this subtile's outputs)
                sf_byte, sf_d = cvt_f32_to_ue8m0(
                    scale_norm(fp, amax, fmt, rule), rule.round
                )
                inv = inv_scale(fp, sf_d, rule)
                yq = [
                    saturate(fp, quant_pre(fp, ys_r[j], inv, rule), fmt, rule)
                    for j in range(V)
                ]
                for w, word in enumerate(pack_block_codes(fp, yq, fmt.elem_fmt)):
                    fy_words[w] = word
                stored = (row_rel < rows_left) & (blk * Int32(BLOCK) < N_out)
                if stored:
                    row_g = row0 + row_rel
                    dst = mOut[(row_g, None)]
                    for h in cutlass.range_constexpr(V // VEC_BYTES):
                        cute.copy(
                            atom16,
                            cute.local_tile(fy, (VEC_BYTES,), (h,)),
                            cute.local_tile(
                                dst,
                                (VEC_BYTES,),
                                (
                                    blk * Int32(BLOCK // VEC_BYTES)
                                    + half * Int32(V // VEC_BYTES)
                                    + h,
                                ),
                            ),
                        )
                if stored & (half == Int32(0)):
                    mSf[
                        sf_offset_128x4_dev(
                            (row0 + row_rel + sf_shift).to(Int64), blk, pcols
                        )
                    ] = sf_byte.to(Uint8)
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(
                acc_state
            )  # TMEM buffer free for the next fold
        acc_state.advance()
