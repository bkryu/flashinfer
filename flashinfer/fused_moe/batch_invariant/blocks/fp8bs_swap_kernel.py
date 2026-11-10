# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The swapped decode body of the block-scale fp8 MoE class ``moe.e4m3bs.<act>``: ``C^T`` per expert -- the expert's weight rows on
the MMA's M side (a 64- or 128-row tile streamed once by TMA with the expert as the batch coordinate) and the group's tokens on the N
side as a narrow token tile (8 / 16 / 32 / 64 rows of the dense packing, a TMA box from the group's first row; rows past the group's
end are finite, computed and never stored), the block-scale fold promoted per 128-element K block exactly as the padded body
(:mod:`.fp8bs_kernel`) promotes it, the transposed group-masked store of the swapped bodies of the sibling classes.

The same class: the fp8 step is symmetric in its operands (the dense block-scale class's swapped records were verified bit for bit
against the exact model on both parts; the MX classes' swapped records are the same hardware hypothesis), so every output element's
block partial ``P_kb`` is the dense fp8 ``mma32`` fold of the block from +0, and the promotion ``acc = fma.rn(mul.rn(s_a[tok, kb],
s_b[blk(n), kb]), P_kb, acc)`` runs in block order from +0 in one epilogue thread's registers -- here the thread owns one or two
weight rows and the tile's token columns (the TMEM load's layout: the 128-row tile one row and consecutive columns, the 64-row tile two
rows 8 apart and every fourth column), so it forms one product per register and block (``mul.rn.f32x2`` packed, each lane the scalar
op) and folds each register's partial with ``fma.rn.f32x2``; the chain (one ``cvt.rn``) and the store are per element. The gated
chain's gate row sits 8 rows below its up row (the thread's own fragment or lane ``+ 8``), and the gate row's scale is its own
block's (the interleaved tile holds one up block and one gate block, as the padded body's tile does). Everything else is placement:
which CTA owns which (expert, token tile, weight tile), the ring depths, the store pattern.

Operands: ``a (R, K, 1)`` K-major e4m3 (the dense packing's token rows; ``R >= Tk + token tile`` keeps the last tiles' boxes in
bounds, else they zero-fill), ``sfa (R, K / 128)`` fp32 with any strides (the MN layout), ``w (N, K, E)`` K-major e4m3, ``sfw (E, N /
128, K / 128)`` fp32, ``out (Tk', Nout)`` row-major bf16 / f32 (``Tk' >= Tk``), ``counts (E,)``. The fused requantization is not a
spelling of this body (the dispatcher's row threshold keeps the standalone pass at the token counts this body serves). The records
are the MoE class's own (``swap_ab`` with a 64- / 128-row tile and a narrow token tile): the dense block-scale GEMM's knob rule does
not admit them, and its body never sees them.
"""

from __future__ import annotations

import dataclasses
from typing import List

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.tensor_utils import LayoutEnum
from cutlass.utils.blackwell_helpers import get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ....batch_invariant.quantization.cute_ext import CuteF32
from ....gemm.batch_invariant.blocks.blockscale_fp8 import (
    IN_FMT,
    _elem,
    _vec4,
    acc_stages_for,
)
from ....gemm.batch_invariant.blocks.operands import (
    ACC_DTYPE,
    BAR_TMEM_ALLOC,
    IN_DT,
    StaticPlan,
)
from ....gemm.batch_invariant.generator import Knobs, MainloopBlock
from ....gemm.batch_invariant.spec import BLOCK_K
from .. import pdl
from ..spec_fp8bs import BLOCK as QBLOCK
from ..spec_fp8bs import check_block
from .epilogue import BAR_EPI_JOIN, GROUP, fragment_offsets
from .fp8bs_kernel import (
    NUM_SCALE_STAGES_OF,
    Fp8BsChain,
    _chain_elem_bs,
    blocks_per_stage,
    n_scale_slots,
    slot_blocks,
)
from .scheduler import MAX_E, TABLE_SMEM_BYTES, GroupScheduler, build_group_tables

EPILOGUE_WARPS = (
    0,
    1,
    2,
    3,
)  # one warpgroup: the weight rows of the tile, one per thread (the 64-row tile: two column halves)
MMA_WARP = 4
TMA_WARP = 5  # the operands and the scale ring
THREADS = 32 * (TMA_WARP + 1)
WEIGHT_TILES = (
    64,
    128,
)  # the MMA's M: the weight rows per CTA tile (prims-TS's DeepSeek body runs 64)
TOKEN_TILES = (8, 16, 32, 64)  # the MMA's N: the tokens per tile


def swap_knobs_fp8bs(
    weight_tile: int = 128, token_tile: int = 8, raster: str = "m"
) -> Knobs:
    """The block-scale class's swapped decode record: ``weight_tile`` weight rows on the MMA's M, ``token_tile`` tokens on its N, 1-CTA."""
    assert weight_tile in WEIGHT_TILES and token_tile in TOKEN_TILES, (
        weight_tile,
        token_tile,
    )
    return Knobs(
        (weight_tile, token_tile), (1, 1), False, True, raster, 1, swap_ab=True
    )


def swapped_knobs_fp8bs() -> List[Knobs]:
    """The swapped decode space of the block-scale class: one record per (weight tile, token tile)."""
    return [swap_knobs_fp8bs(wt, tn) for wt in WEIGHT_TILES for tn in TOKEN_TILES]


def is_swapped_record_fp8bs(knobs: Knobs) -> bool:
    """A record the swapped block-scale body takes: the operands swapped, a 64- / 128-row weight tile, a narrow token tile, 1-CTA, no
    cluster, swizzle 1, the format's stage, no split."""
    return (
        knobs.swap_ab
        and knobs.tile_m in WEIGHT_TILES
        and knobs.tile_n in TOKEN_TILES
        and knobs.cluster_shape_mn == (1, 1)
        and not knobs.use_2cta_instrs
        and knobs.swizzle_size == 1
        and knobs.cluster_k == 1
        and not knobs.cta_chunks
        and knobs.use_tma_store
        and dataclasses.replace(knobs, swap_ab=False).well_formed()
    )


def swap_scale_ring_bytes(
    knobs: Knobs, block: int = QBLOCK, interleaved: bool = True
) -> int:
    """Smem of the scale ring: per stage the tile's token scales and the weight tile's slot scales, ``blocks_per_stage`` of each,
    fp32."""
    spb = blocks_per_stage(block)
    return (
        NUM_SCALE_STAGES_OF[block]
        * (knobs.tile_n + n_scale_slots(knobs.tile_m, block, interleaved))
        * spb
        * (ACC_DTYPE.width // 8)
    )


class SwappedBlockScaleFp8Gemm:
    """``out[p, n] = chain( fold_kb fma(mul(sfa[p, kb], sfw[e(p), blk(n), kb]), P_kb(p, n), acc) )`` for every row ``p`` of every group,
    the weights as the MMA's A operand; ``block`` the class (128 / 32: the 32-wide class runs four blocks per smem stage, one
    instruction step each, as on the padded body)."""

    def __init__(
        self,
        knobs: Knobs,
        chain: Fp8BsChain,
        interleaved: bool = None,
        block: int = QBLOCK,
    ):
        assert isinstance(chain, Fp8BsChain) and chain.out_fmt in ("bf16", "f32"), chain
        assert is_swapped_record_fp8bs(knobs), knobs
        self.interleaved = chain.pair if interleaved is None else bool(interleaved)
        assert self.interleaved or not chain.pair, (
            "the gated chain pairs the interleaved weight's columns"
        )
        self.knobs, self.chain = knobs, chain
        self.weight_tile, self.token_tile = knobs.tile_m, knobs.tile_n
        self.block = check_block(block)
        self.spb = blocks_per_stage(block)
        self.n_slots = n_scale_slots(self.weight_tile, block, self.interleaved)
        self.n_scale_stages = NUM_SCALE_STAGES_OF[block]

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        sfa: cute.Tensor,
        w: cute.Tensor,
        sfw: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        assert a.element_type is IN_DT[IN_FMT] and w.element_type is IN_DT[IN_FMT], (
            a.element_type,
            w.element_type,
        )
        assert sfa.element_type is ACC_DTYPE and sfw.element_type is ACC_DTYPE
        # the weights are the plan's A (M = the weight rows), the tokens its B (N = the token tile); the plan sizes the tiles and rings
        # from the record's dense twin (the swap is this body's)
        st = StaticPlan(
            dataclasses.replace(self.knobs, swap_ab=False),
            w.element_type,
            a.element_type,
            ACC_DTYPE,
            LayoutEnum.from_tensor(w).mma_major_mode(),
            LayoutEnum.from_tensor(a).mma_major_mode(),
            LayoutEnum.ROW_MAJOR,
            c_in_smem=False,
            extra_smem_bytes=TABLE_SMEM_BYTES,
            acc_stages=acc_stages_for(self.knobs),
            reserve_smem_bytes=swap_scale_ring_bytes(
                self.knobs, self.block, self.interleaved
            ),
        )
        self.st = st
        self.mainloop = MainloopBlock(IN_FMT)
        self.lowering = self.mainloop.lowering()
        assert (
            st.mma_tiler[2] == self.mainloop.k_block == BLOCK_K == self.spb * self.block
        ), (st.mma_tiler, self.mainloop.k_block)  # one stage = spb scale blocks
        assert (
            st.cta_tile_shape_mnk[0] == self.weight_tile
            and st.cta_tile_shape_mnk[1] == self.token_tile
        ), st.cta_tile_shape_mnk
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        a_one = cute.select(
            st.a_smem_layout_staged,
            mode=list(range(cute.rank(st.a_smem_layout_staged) - 1)),
        )
        atom_w, tensor_w = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w, a_one, st.mma_tiler, st.tiled_mma, st.cluster_layout_vmnk.shape
        )
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        b_one = cute.select(
            st.b_smem_layout_staged,
            mode=list(range(cute.rank(st.b_smem_layout_staged) - 1)),
        )
        atom_tok, tensor_tok = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, a, b_one, st.mma_tiler, st.tiled_mma, st.cluster_layout_vmnk.shape
        )
        self.num_tma_load_bytes = cute.size_in_bytes(
            st.a_dtype, a_one
        ) + cute.size_in_bytes(st.b_dtype, b_one)
        self.kernel(
            st.tiled_mma,
            atom_w,
            tensor_w,
            atom_tok,
            tensor_tok,
            sfa,
            sfw,
            out,
            counts,
            st.cluster_layout_vmnk,
            st.a_smem_layout_staged,
            st.b_smem_layout_staged,
            st.epi_tile,
        ).launch(
            grid=(1, 1, max_active_clusters),
            block=[THREADS, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_w: cute.CopyAtom,
        mW_mkl: cute.Tensor,
        tma_atom_tok: cute.CopyAtom,
        mTok_nkl: cute.Tensor,
        mSFA: cute.Tensor,
        mSFW: cute.Tensor,
        mOut: cute.Tensor,
        counts: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
    ):
        st = self.st
        tn = self.token_tile
        wt = self.weight_tile
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_tok)
        # programmatic dependent launch: the group tables (the plan: complete before this grid launched; no kernel of the chain triggers
        # before its own wait) and the weight loads do not wait; the TMA warp waits before its first token box and token scale
        # (the predecessor's output), so the weight ring fills during the launch (the swapped NVFP4 body's placement)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()
        E = counts.shape[0]
        N = mW_mkl.shape[0]
        Tk = mOut.shape[0]
        R = mTok_nkl.shape[0]

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_acc_stage * 2]
            scale_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.n_scale_stages * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            group_off: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
            group_pre: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]

        spb, NS, n_stages = self.spb, self.n_slots, self.n_scale_stages

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=st.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Warp),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        ).make_participants()
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=st.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(EPILOGUE_WARPS)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        scale_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.scale_mbar_ptr.data_ptr(),
            num_stages=n_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * len(EPILOGUE_WARPS)
            ),
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=BAR_TMEM_ALLOC, num_threads=32 * (len(EPILOGUE_WARPS) + 1)
        )
        epi_join = pipeline.NamedBarrier(
            barrier_id=BAR_EPI_JOIN, num_threads=32 * len(EPILOGUE_WARPS)
        )
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=EPILOGUE_WARPS[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        s_off = storage.group_off.get_tensor(cute.make_layout((MAX_E + 1,)))
        s_pre = storage.group_pre.get_tensor(cute.make_layout((MAX_E + 1,)))
        if warp_idx == EPILOGUE_WARPS[0]:
            build_group_tables(
                counts, E, tn, s_off, s_pre, 1
            )  # the dense rows: the token tiles of a group
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sW = smem.allocate_tensor(
            element_type=st.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sTok = smem.allocate_tensor(
            element_type=st.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # the scale ring: a stage holds the tile's token scales and the weight tile's slot scales of the stage's `spb` blocks (a
        # token's or a slot's `spb` scales contiguous)
        sSFA = smem.allocate_tensor(
            element_type=ACC_DTYPE,
            layout=cute.make_layout((tn, spb, n_stages), stride=(spb, 1, tn * spb)),
            byte_alignment=16,
        )  # the token scales
        sSFB = smem.allocate_tensor(
            element_type=ACC_DTYPE,
            layout=cute.make_layout((NS, spb, n_stages), stride=(spb, 1, NS * spb)),
            byte_alignment=16,
        )  # the weight slot scales
        gW_mkl = cute.local_tile(
            mW_mkl, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )  # (bM, bK, RestM, RestK, RestL)
        k_tile_cnt = cute.size(gW_mkl, mode=[3])  # = K / BLOCK_K: the block count
        thr_mma = tiled_mma.get_slice(0)
        tCgW = thr_mma.partition_A(gW_mkl)
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sW, 0, 3),
            cute.group_modes(tCgW, 0, 3),
        )
        tCrW = tiled_mma.make_fragment_A(sW)
        tCrTok = tiled_mma.make_fragment_B(sTok)
        acc_shape = tiled_mma.partition_shape_C(st.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, st.num_acc_stage)
        )
        pipeline_init_wait(cluster_shape_mn=(1, 1))
        cute.arch.barrier()  # the group tables are complete

        raster_m = self.knobs.raster_along == "m"
        # the scheduler's "cluster rows" are the token tile, its "N tiles" the weight tiles
        if warp_idx == TMA_WARP:
            tile_sched = GroupScheduler.create(
                s_off, s_pre, counts, E, N, tn, wt, 1, raster_m, 0, 0
            )
            swap_bs_producer_loop(
                st,
                k_tile_cnt,
                tile_sched,
                ab_producer,
                tma_atom_w,
                tma_atom_tok,
                tWgW,
                tWsW,
                mTok_nkl,
                sTok,
                thr_mma,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                scale_pipeline,
                mSFA,
                mSFW,
                sSFA,
                sSFB,
                R,
                self.interleaved,
                self.block,
            )
        if warp_idx == MMA_WARP:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = GroupScheduler.create(
                s_off, s_pre, counts, E, N, tn, wt, 1, raster_m, 0, 0
            )
            swap_bs_mma_loop(
                self.lowering,
                spb,
                st.num_acc_stage,
                k_tile_cnt,
                tile_sched,
                tiled_mma,
                ab_consumer,
                acc_pipeline,
                tCrW,
                tCrTok,
                tCtAcc_base,
            )
        if warp_idx < len(EPILOGUE_WARPS):
            tmem.allocate(st.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = GroupScheduler.create(
                s_off, s_pre, counts, E, N, tn, wt, 1, raster_m, 0, 0
            )
            swap_bs_epilogue_loop(
                st,
                self.chain,
                self.interleaved,
                tidx,
                tile_sched,
                tCtAcc_base,
                thr_mma,
                mOut,
                Tk,
                N,
                epi_tile,
                acc_pipeline,
                scale_pipeline,
                sSFA,
                sSFB,
                k_tile_cnt,
                self.block,
            )
            tmem.relinquish_alloc_permit()
            # the allocator warp alone frees the columns: the epilogue warpgroup's warps join here after their last TMEM loads
            epi_join.arrive_and_wait()
            tmem.free(tmem_ptr)


@cute.jit
def swap_bs_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    ab_producer,
    tma_atom_w: cute.CopyAtom,
    tma_atom_tok: cute.CopyAtom,
    tWgW: cute.Tensor,
    tWsW: cute.Tensor,
    mTok_nkl: cute.Tensor,
    sTok: cute.Tensor,
    thr_mma,
    b_cta_coord,
    b_cta_layout,
    scale_pipeline,
    mSFA: cute.Tensor,
    mSFW: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    R: Int32,
    interleaved: cutlass.Constexpr,
    block: cutlass.Constexpr,
):
    """The TMA producer: per tile the expert's weight rows ``n_tile`` (the expert as the batch coordinate) and the token box re-addressed
    at the group's dense row ``row0`` (``token_tile`` rows: the rows past the group's end are computed and never stored, the rows past the
    buffer zero-filled); k-tiles (smem stages of ``spb`` scale blocks) ascending, one ring slot holds both. Next to each stage the warp's
    lanes ``cp.async`` the tile's token scales of the stage's blocks (lane ``j < token_tile``: ``mSFA[(row0 + j, kb)]``, the row clamped
    to the buffer) and lanes ``t < NS`` the weight tile's slot scales (:func:`.fp8bs_kernel.slot_blocks`: the interleaved weight's up
    and gate blocks, the plain weight's blocks; the 32-wide class's four blocks of a slot in one 16-byte copy) into the scale ring. The
    weights are inputs of the whole chain, the token rows and their scales the predecessor's output: the first tile's weight stages are
    issued before the programmatic-dependent-launch wait and their token boxes right after it. Order and staging only."""
    lane = cute.arch.lane_idx()
    NB = mSFW.shape[1]
    tn = st.cta_tile_shape_mnk[1]
    wt = st.cta_tile_shape_mnk[0]
    rows_per_lane = (tn + 31) // 32
    spb = cute.size(sSFA, mode=[1])
    NS = cute.size(sSFB, mode=[0])
    n_stages = cute.size(sSFA, mode=[2])
    atom_g2s = cute.make_copy_atom(
        cpasync.CopyG2SOp(), ACC_DTYPE, num_bits_per_copy=ACC_DTYPE.width
    )
    atom_g2s4 = cute.make_copy_atom(
        cpasync.CopyG2SOp(), ACC_DTYPE, num_bits_per_copy=4 * ACC_DTYPE.width
    )
    scale_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, n_stages
    )
    work_tile = tile_sched.initial_work_tile_info()
    n_pre = cutlass.min(Int32(st.num_ab_stage), k_tile_cnt)
    if not work_tile.is_valid_tile:
        n_pre = Int32(0)  # a CTA without tiles issues nothing
    tWgW_slice = tWgW[(None, work_tile.n_tile, None, work_tile.e)]
    ab_producer.reset()
    tok_producer = (
        ab_producer.clone()
    )  # the same stages, walked again for the token boxes
    for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
        handle = ab_producer.acquire_and_advance()
        cute.copy(
            tma_atom_w,
            tWgW_slice[(None, handle.count)],
            tWsW[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
        )
    cute.arch.griddepcontrol_wait()  # the token rows and their scales are the predecessor's
    k_start = n_pre
    first = Int32(1)
    while work_tile.is_valid_tile:
        e = work_tile.e
        row0 = work_tile.row0
        n0 = work_tile.n_tile * Int32(wt)
        tWgW_slice = tWgW[(None, work_tile.n_tile, None, e)]
        real_tok = cute.domain_offset((row0, 0, 0), mTok_nkl)
        gTok = cute.local_tile(
            real_tok, cute.slice_(st.mma_tiler, (0, None, None)), (None, None, None)
        )
        tCgTok = thr_mma.partition_B(gTok)
        tTsTok, tTgTok = cpasync.tma_partition(
            tma_atom_tok,
            b_cta_coord,
            b_cta_layout,
            cute.group_modes(sTok, 0, 3),
            cute.group_modes(tCgTok, 0, 3),
        )
        tTgTok_slice = tTgTok[(None, 0, None, 0)]
        nbs = slot_blocks(n0, NB, NS, block, interleaved)
        if first == 1:
            # the first tile's prefetched weight stages: their token boxes and scales
            for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
                handle = tok_producer.current_handle()
                cute.copy(
                    tma_atom_tok,
                    tTgTok_slice[(None, handle.count)],
                    tTsTok[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                tok_producer.advance()
                scale_pipeline.producer_acquire(scale_state)
                swap_bs_stage_scales(
                    lane,
                    handle.count,
                    scale_state.index,
                    row0,
                    R,
                    e,
                    nbs,
                    mSFA,
                    mSFW,
                    sSFA,
                    sSFB,
                    atom_g2s,
                    atom_g2s4,
                    tn,
                    rows_per_lane,
                    spb,
                    NS,
                )
                scale_pipeline.producer_commit(scale_state)
                scale_state.advance()
        first = Int32(0)
        peek_empty = cutlass.Boolean(1)
        if k_start < k_tile_cnt:
            peek_empty = ab_producer.try_acquire()
        for _k_tile in cutlass.range(k_start, k_tile_cnt, 1, unroll=1):
            handle = ab_producer.acquire_and_advance(peek_empty)
            cute.copy(
                tma_atom_w,
                tWgW_slice[(None, handle.count)],
                tWsW[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            cute.copy(
                tma_atom_tok,
                tTgTok_slice[(None, handle.count)],
                tTsTok[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            scale_pipeline.producer_acquire(
                scale_state
            )  # ring stage free (the promotion released it)
            swap_bs_stage_scales(
                lane,
                handle.count,
                scale_state.index,
                row0,
                R,
                e,
                nbs,
                mSFA,
                mSFW,
                sSFA,
                sSFB,
                atom_g2s,
                atom_g2s4,
                tn,
                rows_per_lane,
                spb,
                NS,
            )
            scale_pipeline.producer_commit(
                scale_state
            )  # arrives when this lane's copies have landed
            scale_state.advance()
            peek_empty = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_empty = ab_producer.try_acquire()
        ab_producer.reset()  # the next tile's k-tiles count from 0
        k_start = Int32(0)
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    pdl.trigger_late()
    ab_producer.tail()
    scale_pipeline.producer_tail(scale_state)


@cute.jit
def swap_bs_stage_scales(
    lane,
    kt,
    sidx,
    row0: Int32,
    R: Int32,
    e,
    nbs,
    mSFA: cute.Tensor,
    mSFW: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    atom_g2s,
    atom_g2s4,
    tn: cutlass.Constexpr,
    rows_per_lane: cutlass.Constexpr,
    spb: cutlass.Constexpr,
    NS: cutlass.Constexpr,
):
    """One ring stage's copies for smem stage ``kt`` (scale blocks ``kt * spb ..``): the tile's token scales per block (lanes ``j <
    tn``, the row clamped to the buffer) and the weight tile's slot scales (lanes ``t < NS``; the four blocks of a slot one 16-byte
    copy at ``spb = 4``: the weight scales are K-contiguous and ``K / 32`` a multiple of four)."""
    kb0 = kt * Int32(spb)
    for r in cutlass.range_constexpr(rows_per_lane):
        j = r * 32 + lane
        if j < tn:
            row = cutlass.min(row0 + j, R - Int32(1))
            for jb in cutlass.range_constexpr(spb):
                cute.copy(
                    atom_g2s,
                    _elem(mSFA, (row, kb0 + Int32(jb))),
                    _elem(sSFA, (j, jb, sidx)),
                )
    for t in cutlass.range_constexpr(NS):
        if lane == t:
            if cutlass.const_expr(spb == 4):
                cute.copy(
                    atom_g2s4, _vec4(mSFW, (e, nbs[t], kb0)), _vec4(sSFB, (t, 0, sidx))
                )
            else:
                cute.copy(
                    atom_g2s, _elem(mSFW, (e, nbs[t], kb0)), _elem(sSFB, (t, 0, sidx))
                )


@cute.jit
def swap_bs_mma_loop(
    lowering,
    spb: cutlass.Constexpr,
    num_acc_stage: cutlass.Constexpr,
    k_tile_cnt: Int32,
    tile_sched,
    tiled_mma: cute.TiledMma,
    ab_consumer,
    acc_pipeline,
    tCrW: cute.Tensor,
    tCrTok: cute.Tensor,
    tCtAcc_base: cute.Tensor,
):
    """Per tile, per k-tile (stages ascending), per scale block of the stage (``spb``, ascending): the block's steps from a reset
    accumulator into the next accumulator stage, committed to the epilogue -- the dense block-scale body's mainloop
    (``blockscale_mma_loop``) with the weights as the A operand; the 32-wide class commits every instruction step."""
    assert cute.size(tCrW, mode=[2]) == lowering.inner and lowering.inner % spb == 0, (
        "one smem stage must hold exactly the lowering's inner steps"
    )
    steps = lowering.inner // spb  # instruction steps per scale block
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, num_acc_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        ab_consumer.reset()
        peek_full = ab_consumer.try_wait()
        for _kt in range(k_tile_cnt):
            handle = ab_consumer.wait_and_advance(peek_full)
            for jb in cutlass.range_constexpr(
                spb
            ):  # the stage's blocks ascending, each from +0 into its own stage
                acc_pipeline.producer_acquire(
                    acc_state
                )  # the epilogue has drained this TMEM stage
                tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
                for j in cutlass.range_constexpr(
                    steps
                ):  # the block's steps ascending, the first from +0
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, j > 0)
                    crd = (None, None, jb * steps + j, handle.index)
                    cute.gemm(tiled_mma, tCtAcc, tCrW[crd], tCrTok[crd], tCtAcc)
                acc_pipeline.producer_commit(
                    acc_state
                )  # the block's partial is complete: hand to the promotion
                acc_state.advance()
            handle.release()
            peek_full = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_full = ab_consumer.try_wait()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    acc_pipeline.producer_tail(acc_state)


@cute.jit
def swap_bs_epilogue_loop(
    st,
    chain: cutlass.Constexpr,
    interleaved: cutlass.Constexpr,
    tidx: Int32,
    tile_sched,
    tCtAcc_base: cute.Tensor,
    thr_mma,
    mOut: cute.Tensor,
    Tk: Int32,
    N: Int32,
    epi_tile: cute.Tile,
    acc_pipeline,
    scale_pipeline,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    k_tile_cnt: Int32,
    block: cutlass.Constexpr = QBLOCK,
):
    """The promotion and the transposed group-masked store: the accumulator tile is (weight rows, tokens). The TMEM load's layout gives
    each thread a static set of (weight row, token column) offsets from a per-thread base (the 128-row tile: one row, consecutive
    columns; the 64-row tile: two rows 8 apart, every fourth column). Per block the thread forms ``mul.rn(s_a[tok j, kb], s_b[blk(row),
    kb])`` per register (packed in pairs: each lane the scalar op; the weight block of a register's row is a slot fixed per thread -- the
    row's ``block``-row block of the tile, the interleaved tile's gate rows reading the gate half's slots) and folds the block partial
    with ``fma.rn.f32x2`` into its register running sums (the 32-wide class: four blocks per smem stage, one ring stage). After the
    last block the chain -- gated: an up row's gate row (8 rows below) is in the thread's own fragment or at lane ``+ 8`` -- and the
    store: token ``j`` of the tile is expanded row ``row0 + j``, stored iff ``j < rows_left``; the up rows store output column ``(n //
    16) * 8 + n % 8`` (gated), every row column ``n`` (plain)."""
    pair = chain.pair
    wt = st.cta_tile_shape_mnk[0]
    spb = cute.size(sSFA, mode=[1])
    NS = cute.size(sSFB, mode=[0])
    n_stages = cute.size(sSFA, mode=[2])
    tiler_mn = cute.slice_(st.mma_tiler, (None, None, 0))
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    copy_atom_t2r = get_tmem_load_op(
        st.cta_tile_shape_mnk, st.c_layout, st.c_dtype, ACC_DTYPE, epi_tile, False
    )
    tAcc_epi = cute.flat_divide(tCtAcc, epi_tile)  # (EPI_M, EPI_N, SUB_M, SUB_N, STAGE)
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
    )
    thr_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc_base = thr_t2r.partition_S(
        tAcc_epi
    )  # (T2R, T2R_M, T2R_N, SUB_M, SUB_N, STAGE)
    idF = cute.make_identity_tensor(
        (st.cta_tile_shape_mnk[0], st.cta_tile_shape_mnk[1], 1)
    )  # the tile's own (weight row, token) coordinates
    cF = cute.local_tile(idF, tiler_mn, (None, None, None))
    tCcF = transform_partitioned_tensor_layout(thr_mma.partition_C(cF))
    cF_epi = cute.flat_divide(
        tCcF, epi_tile
    )  # (EPI_M, EPI_N, SUB_M, SUB_N, RestM, RestN, RestL)
    tTR_cF = thr_t2r.partition_D(cF_epi)  # (T2R, T2R_M, T2R_N, SUB_M, SUB_N, 1, 1, 1)
    frag = tTR_cF[(None, None, None, 0, 0, 0, 0, 0)]
    offs = fragment_offsets(
        frag.layout
    )  # the registers' static (row, column) offsets from the thread's base
    nv = len(offs)
    assert nv % 2 == 0, offs
    index = {o: v for v, o in enumerate(offs)}
    # the gate row of an up register (8 rows below it): the thread's own register when the layout holds that row, else lane + 8
    partner = [index.get((m + GROUP, n)) for m, n in offs]
    sub_m, sub_n = cute.size(tAcc_epi, mode=[2]), cute.size(tAcc_epi, mode=[3])
    assert sub_m == 1, "the epilogue subtiles run along the token columns only"
    NSUB = sub_n
    VW = NSUB * nv
    tTR_cF = cute.group_modes(tTR_cF, 3, cute.rank(tTR_cF))
    crd0 = tTR_cF[(None, None, None, 0)][(0, 0, 0)]
    base_row = crd0[0] - Int32(offs[0][0])  # this thread's base weight row in the tile
    col_base = [
        tTR_cF[(None, None, None, s)][(0, 0, 0)][1] - Int32(offs[0][1])
        for s in range(NSUB)
    ]  # its base token column per subtile
    rows = sorted({m for m, _ in offs})
    buf = cute.make_rmem_tensor(frag.shape, ACC_DTYPE)
    tTR_rSum = cute.make_rmem_tensor((VW,), ACC_DTYPE)
    # the weight rows of the thread's registers: gate rows of the interleaved tile read the gate half's slots; the slot of a row is
    # its `block`-row block of the tile (`slot_blocks`: the interleaved tile's up rows (n // 16) * 8 + n % 8 index the up half's slots)
    gate_row = {
        m: ((base_row + Int32(m)) % Int32(2 * GROUP)) >= Int32(GROUP) for m in rows
    }
    slot_row = {}
    for m in rows:
        wrow = base_row + Int32(m)
        if cutlass.const_expr(interleaved):
            u = (wrow // Int32(2 * GROUP)) * Int32(GROUP) + wrow % Int32(
                GROUP
            )  # the row's original row within the tile's half
            slot = (u // Int32(block)) if (wt // 2) >= block else Int32(0)
            if gate_row[m]:
                slot = slot + Int32(
                    NS // 2
                )  # the gate half's slots follow the up half's
            slot_row[m] = slot
        else:
            slot_row[m] = (wrow // Int32(block)) if wt >= block else Int32(0)
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, st.num_acc_stage
    )
    scale_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, n_stages
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        row0, row_end, n_tile = work_tile.row0, work_tile.row_end, work_tile.n_tile
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        rows_left = cutlass.min(row_end, Tk) - row0  # valid tokens of this tile
        for i in cutlass.range_constexpr(VW):
            tTR_rSum[i] = CuteF32.const(0.0)  # acc_0 = +0
        for _kt in range(
            k_tile_cnt
        ):  # stages ascending, their blocks ascending: the block fold digit
            scale_pipeline.consumer_wait(
                scale_state
            )  # the stage's blocks' scales are in the ring
            sidx = scale_state.index
            for jb in cutlass.range_constexpr(spb):
                sb = {}
                for mi in cutlass.range_constexpr(len(rows)):
                    m = rows[mi]
                    sb[m] = sSFB[(slot_row[m], jb, sidx)]
                scs = []
                for s in cutlass.range_constexpr(NSUB):
                    for v in cutlass.range_constexpr(0, nv, 2):
                        a0 = sSFA[(col_base[s] + Int32(offs[v][1]), jb, sidx)]
                        a1 = sSFA[(col_base[s] + Int32(offs[v + 1][1]), jb, sidx)]
                        p0, p1 = CuteF32.mul2(
                            (a0, a1), (sb[offs[v][0]], sb[offs[v + 1][0]])
                        )  # mul.rn(s_a[tok, kb], s_b[blk(row), kb]) per register
                        scs.append(p0)
                        scs.append(p1)
                if cutlass.const_expr(jb == spb - 1):
                    scale_pipeline.consumer_release(scale_state)
                    scale_state.advance()
                acc_pipeline.consumer_wait(acc_state)  # the block's fold is complete
                tTR_tAcc = cute.group_modes(
                    tTR_tAcc_base[(None, None, None, None, None, acc_state.index)], 3, 5
                )
                for s in cutlass.range_constexpr(NSUB):
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], buf)
                    for v in cutlass.range_constexpr(0, nv, 2):
                        # acc = fma(scale, P_kb, acc) for two registers per fma.rn.f32x2 (each lane the same correctly rounded fma)
                        r0, r1 = CuteF32.fma2(
                            (scs[s * nv + v], scs[s * nv + v + 1]),
                            (buf[v], buf[v + 1]),
                            (tTR_rSum[s * nv + v], tTR_rSum[s * nv + v + 1]),
                        )
                        tTR_rSum[s * nv + v] = r0
                        tTR_rSum[s * nv + v + 1] = r1
                cute.arch.fence_view_async_tmem_load()  # this warp's reads of the partial are complete
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(
                        acc_state
                    )  # one arrive per warp: the stage is free once all arrived
                acc_state.advance()
        # ===== the chain and the transposed store =====
        # the gate rows' sums for the up rows (gated): the thread's own register 8 rows below, or lane + 8 (rows 16q + 8 + r sit 8 lanes
        # below rows 16q + r in the 128-row layout); the shuffles are warp-uniform, before any branch
        gate = [None] * VW
        if cutlass.const_expr(pair):
            for s in cutlass.range_constexpr(NSUB):
                for v in cutlass.range_constexpr(nv):
                    if cutlass.const_expr(partner[v] is not None):
                        gate[s * nv + v] = tTR_rSum[s * nv + partner[v]]
                    else:
                        gate[s * nv + v] = cute.arch.shuffle_sync_down(
                            tTR_rSum[s * nv + v], offset=GROUP
                        )
        for mi in cutlass.range_constexpr(len(rows)):
            m = rows[mi]
            wrow = base_row + Int32(m)  # this register row's weight row in the tile
            n_glob = n_tile * Int32(wt) + wrow
            if cutlass.const_expr(pair):
                out_col = (
                    n_tile * Int32(wt // 2)
                    + (wrow // Int32(2 * GROUP)) * Int32(GROUP)
                    + wrow % Int32(GROUP)
                )
                store_ok = (~gate_row[m]) & (n_glob < N)
            else:
                out_col = n_glob
                store_ok = n_glob < N
            for s in cutlass.range_constexpr(NSUB):
                for v in cutlass.range_constexpr(nv):
                    if cutlass.const_expr(offs[v][0] == m):
                        i = s * nv + v
                        j = col_base[s] + Int32(offs[v][1])
                        if (j < rows_left) & store_ok:
                            val = _chain_elem_bs(
                                chain,
                                tTR_rSum[i],
                                gate[i] if cutlass.const_expr(pair) else None,
                            )
                            mOut[(row0 + j, out_col)] = val
