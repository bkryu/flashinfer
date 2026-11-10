# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The swapped decode body of the NVFP4 MoE classes: ``C^T`` per expert, with the expert's weight rows on the MMA's M side
(a 128-row tile and its 128x4 scale-factor atom streamed by TMA with the expert as the batch coordinate) and the group's
tokens on the N side as a **narrow token tile** (8, 16 or 32 rows: the dense 4-bit class's swapped decode records). The
token tile is the dense packing's rows (:mod:`.kernel_nvfp4`, ``dense_a``: the groups' quantized rows back to back, their
scale factors and per-token scales at the 128-row padded positions), so

* the token operand is a TMA box of ``token_tile`` rows starting at the group's dense row (rows past the group's end belong
  to the next group or to the buffer's tail: finite values, computed and never stored);
* the token tile's scale factors are gathered by one LDGSTS warp from the padded positions into rows ``0 .. token_tile - 1``
  of the stage's SFB atom (the narrow swapped records read the atom from its first rows: no sub-tile addressing), rows past
  the group's end skipped;
* the transposed group-masked epilogue (:func:`swapped_w4a4_epilogue_loop`) stores token ``j`` of the tile as output row
  ``row0 + j`` with the W4A4 per-token scale per column: ``s_j = mul(alpha_e, pts[pad(row0 + j)])``.

Per tile the operand bytes are the weight tile plus ``token_tile`` rows of tokens, instead of the padded body's 128-row token
atom: at decode (a few rows per active expert) the GEMM traffic becomes the weights read once.

The same class: the block-scaled step is symmetric in its operands (the dense classes' swapped records are bit-identical to
the unswapped ones over the whole space), the step sequence per output element is the dense class's (k-tiles ascending, the
inner steps, the scale factors applied inside the instruction), the gated pairing moves to lanes ``l`` / ``l + 8`` of one
warp (rows ``16q + r`` (up) and ``16q + 8 + r`` (gate) of the interleaved weight are TMEM lanes ``l`` and ``l + 8``: a
``shfl.sync.down 8`` of the gate accumulators). Per-expert agreement with ``mm_fp4`` on the group's rows is the test, at
every token tile, with groups spanning several tiles.

Format-generic over the plan registry (``PLANS``: the plan class carries the operand / scale formats, the scale block and
the instruction), as the padded body.

Operands: ``a (R, K/2, 1)`` packed fp4 as uint8 (the dense rows; ``R >= Tk + token_tile``: the last tiles' boxes stay in
bounds), ``sfa`` the flat 128x4 swizzled scale factors at the padded positions, ``w (N, K/2, E)``, ``sfw`` (``E`` consecutive
128x4 buffers), ``out (Tk', Nout)`` (``Tk' >= Tk``), ``counts (E,)``, ``alpha (E,)``, ``pts`` (the per-token scales at the padded
positions, ``Tk_pad`` entries at least).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as bs_layout
from cutlass import Boolean, Int32, Int64, Uint8, Uint32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineOp,
    agent_sync,
    pipeline_init_arrive,
    pipeline_init_wait,
)
from cutlass.pipeline.helpers import MbarrierArray
from cutlass.utils.blackwell_helpers import get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ....batch_invariant.quantization.cvt_fp8 import (
    cvt_f32_to_e4m3,
    cvt_f32x8_to_e2m1x8,
)
from ....gemm.batch_invariant.blocks.blockscaled import (
    PLANS,
    SF_ATOM_BYTES,
    fp4_view,
    s2t_copy_and_partition,
)
from ....gemm.batch_invariant.blocks.operands import ACC_DTYPE, BAR_TMEM_ALLOC
from ....gemm.batch_invariant.generator import TILES_N_NARROW, Knobs
from ....gemm.batch_invariant.spec import NVFP4_FMTS, SF_ATOM_KBLOCKS
from ..spec_nvfp4 import NVFP4_BLOCK, NVFP4_C_SIXTH, w4a4_row_scale
from .epilogue import BAR_EPI_JOIN, GROUP, MoeF32, fragment_offsets
from .epilogue_nvfp4 import W4A4Chain, _scaled_elem, sf_byte_offset
from .kernel_nvfp4 import GROUP_ALIGN, grouped_plan, sf_view_experts
from .scheduler import (
    MAX_E,
    GroupScheduler,
    build_group_tables,
    build_group_tables_dual,
)

EPILOGUE_WARPS = (
    0,
    1,
    2,
    3,
)  # one warpgroup: the 128 weight rows of the tile, one per thread
SF_WARP = 4  # the LDGSTS warp: the token tile's scale-factor rows (one lane per token)
MMA_WARP = 5
TMA_WARP = 6
THREADS = 32 * (TMA_WARP + 1)
WEIGHT_TILE = 128  # weight rows per tile (the MMA's M: the instruction's 1-CTA M and one scale-factor atom)
TOKEN_TILES = TILES_N_NARROW  # the swapped records' token tiles (8, 16, 32): one scale-factor atom read from its first rows
SF_GROUP = (
    4  # scale-factor bytes per 4-byte cp.async (one 4-block column group of the atom)
)
# TMA L2 cache-policy words (CUTLASS's CacheHintSm90): a weight tile read by one CTA only is evicted first, the token box
# every weight tile of the group re-reads is kept
L2_EVICT_NORMAL, L2_EVICT_FIRST, L2_EVICT_LAST = (
    0x1000000000000000,
    0x12F0000000000000,
    0x14F0000000000000,
)
#: Placement variants of the body (A/B records; the kernel name carries them): ``legacy_tables`` = the epilogue's allocator
#: warp scans the group tables twice before the TMEM allocation; ``l2`` = the weight / token TMA cache hints. The default is
#: the one-pass scan on the scale-factor warp with the TMEM allocation issued first.
VARIANTS = ("legacy_tables", "l2")


def variant_tokens(variant: str) -> tuple:
    toks = tuple(t for t in variant.split("+") if t)
    assert all(t in VARIANTS for t in toks), (variant, VARIANTS)
    return toks


def is_swapped_record(knobs: Knobs) -> bool:
    """A record the swapped body takes: the dense class's swapped 1-CTA record at a narrow token tile, no cluster, swizzle 1."""
    return (
        knobs.swap_ab
        and knobs.tile_m == WEIGHT_TILE
        and knobs.tile_n in TOKEN_TILES
        and knobs.cluster_shape_mn == (1, 1)
        and not knobs.use_2cta_instrs
        and knobs.swizzle_size == 1
        and knobs.cluster_k == 1
        and not knobs.cta_chunks
    )


@dataclass(frozen=True)
class PipelineCpAsyncUmma(PipelineAsync):
    """``cp.async`` producers (the scale-factor warp: each thread's arrival is its ``cp.async.mbarrier.arrive.noinc``, so the
    full barrier flips when every thread's copies have landed) and the UMMA consumer (its release is the ``tcgen05`` commit of
    the MMA that read the stage). The DSL's own ``PipelineCpAsync`` has an async-thread consumer; this is its UMMA-consumer
    sibling (the 1-CTA form: no peer mask)."""

    cta_group: tcgen05.CtaGroup = tcgen05.CtaGroup.ONE

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        barrier_storage: cute.Pointer,
        defer_sync: bool = False,
    ):
        producer = (PipelineOp.AsyncLoad, producer_group)
        consumer = (PipelineOp.TCGen05Mma, consumer_group)
        sync_object_full = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer
        )
        sync_object_empty = MbarrierArray(
            barrier_storage=barrier_storage.align(min_align=8) + num_stages,
            num_stages=num_stages,
            agent=consumer,
        )
        if not defer_sync:
            agent_sync(Agent.ThreadBlock)
        return PipelineCpAsyncUmma(
            sync_object_full,
            sync_object_empty,
            num_stages,
            None,
            None,
            tcgen05.CtaGroup.ONE,
        )

    def consumer_release(self, state):
        self.sync_object_empty.arrive(state.index, self.consumer_mask, self.cta_group)


class SwappedBlockScaledGemm:
    """``out[p, n] = chain(sum_k MMA(W[n, e(p)], SFW[n, e(p)], A[p], SFA[pad(p)]))`` for every dense row ``p`` of every group."""

    def __init__(
        self, knobs: Knobs, chain: W4A4Chain, in_fmt: str = "nvfp4", variant: str = ""
    ):
        assert isinstance(chain, W4A4Chain) and chain.out_fmt in (
            "bf16",
            "f32",
            "nvfp4",
        ), chain
        assert in_fmt in NVFP4_FMTS and in_fmt in PLANS, in_fmt
        assert is_swapped_record(knobs), (
            "the swapped block-scaled body takes the narrow swapped records (128 weight rows x 8 / 16 / 32 tokens, 1-CTA)",
            knobs,
        )
        self.knobs, self.chain, self.in_fmt = knobs, chain, in_fmt
        toks = variant_tokens(variant)
        self.legacy_tables, self.l2_hints = "legacy_tables" in toks, "l2" in toks
        self.plan_cls = PLANS[in_fmt]
        self.token_tile = knobs.tile_n

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        sfa: cute.Tensor,
        w: cute.Tensor,
        sfw: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        alpha: cute.Tensor,
        pts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._launch(
            a, sfa, w, sfw, out, counts, alpha, pts, max_active_clusters, stream
        )

    @cute.jit
    def call_static(
        self,
        a: cute.Tensor,
        sfa: cute.Tensor,
        w: cute.Tensor,
        sfw: cute.Tensor,
        out: cute.Tensor,
        out_sf: cute.Tensor,
        counts: cute.Tensor,
        alpha: cute.Tensor,
        pts: cute.Tensor,
        aux: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The static class's chains (as ``GroupedBlockScaledGemm.call_static``): ``out`` the packed codes, ``out_sf`` the flat 128x4
        scale bytes at the padded positions, ``aux[e]`` the expert's ``gs2`` (GEMM1) or ``inv2`` (GEMM2, ``out_sf`` unused)."""
        self._launch(
            a,
            sfa,
            w,
            sfw,
            out,
            counts,
            alpha,
            pts,
            max_active_clusters,
            stream,
            out_sf,
            aux,
        )

    def _launch(
        self,
        a,
        sfa,
        w,
        sfw,
        out,
        counts,
        alpha,
        pts,
        max_active_clusters,
        stream,
        out_sf=None,
        aux=None,
    ):
        plan_cls = self.plan_cls
        a4 = fp4_view(a)  # (R, K, 1): the dense token rows, the MMA's B operand
        w4 = fp4_view(w)  # (N, K, E): the MMA's A operand (M = N, K, L = E)
        sfw_t = sf_view_experts(
            sfw, w4.shape[0], w4.shape[1], w4.shape[2], plan_cls.SF_VEC, plan_cls.SF_DT
        )
        tables = 4 * (MAX_E + 1) * 4  # the dense and the padded group tables
        st = grouped_plan(
            plan_cls, self.knobs, lambda st: tables, plan_cls.UMMA_SMEM_ADDR_MAX
        )
        self.st = st
        self.block = st.block
        assert self.block.in_fmt == self.in_fmt, (self.block.in_fmt, self.in_fmt)
        self.lowering = self.block.lowering()
        assert st.mma_tiler[2] == self.block.k_block, (st.mma_tiler, self.block.k_block)
        assert (
            st.cta_tile_shape_mnk[0] == WEIGHT_TILE
            and st.cta_tile_shape_mnk[1] == self.token_tile
        ), st.cta_tile_shape_mnk
        # the weights and their scale factors on the TMA as the A / SFA operands (the expert as the batch coordinate)
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        a_one = cute.slice_(st.a_smem_layout_staged, (None, None, None, 0))
        atom_w, tensor_w = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w4, a_one, st.mma_tiler, st.tiled_mma, st.cluster_layout_vmnk.shape
        )
        sfa_one = cute.slice_(st.sfa_smem_layout_staged, (None, None, None, 0))
        atom_sfw, tensor_sfw = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            sfw_t,
            sfa_one,
            st.mma_tiler,
            st.tiled_mma,
            st.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        # the token rows on the TMA as the B operand: a box of token_tile rows x one k-tile, re-addressed per tile at the
        # group's dense row
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        b_one = cute.slice_(st.b_smem_layout_staged, (None, None, None, 0))
        atom_tok, tensor_tok = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, a4, b_one, st.mma_tiler, st.tiled_mma, st.cluster_layout_vmnk.shape
        )
        self.num_tma_load_bytes = (
            cute.size_in_bytes(st.a_dtype, a_one)
            + cute.size_in_bytes(st.sf_dtype, sfa_one)
            + cute.size_in_bytes(st.b_dtype, b_one)
        )
        self.sf_stage_bytes = cute.size_in_bytes(
            st.sf_dtype, cute.slice_(st.sfb_smem_layout_staged, (None, None, None, 0))
        )
        self.kernel(
            st.tiled_mma,
            atom_w,
            tensor_w,
            atom_sfw,
            tensor_sfw,
            atom_tok,
            tensor_tok,
            sfa,
            out,
            counts,
            alpha,
            pts,
            out_sf,
            aux,
            st.cluster_layout_vmnk,
            st.a_smem_layout_staged,
            st.b_smem_layout_staged,
            st.sfa_smem_layout_staged,
            st.sfb_smem_layout_staged,
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
        tma_atom_sfw: cute.CopyAtom,
        mSFW_mkl: cute.Tensor,
        tma_atom_tok: cute.CopyAtom,
        mTok_nkl: cute.Tensor,
        mSFA_flat: cute.Tensor,
        mOut: cute.Tensor,
        counts: cute.Tensor,
        mAlpha: cute.Tensor,
        mPts: cute.Tensor,
        mOutSf: Optional[cute.Tensor],
        mAux: Optional[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
    ):
        st = self.st
        tn = self.token_tile
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_sfw)
            cpasync.prefetch_descriptor(tma_atom_tok)
        # programmatic dependent launch: the wait for the predecessor grid sits in each warp before its first read of the
        # predecessor's output (the token rows, their scale factors, the per-token scales) -- the weight loads and the group
        # tables (the plan: complete before this grid launched; no kernel of the chain triggers before its own wait) do not
        # wait, so the weight ring fills during the launch; no explicit trigger -- the dependents launch at this grid's
        # completion (triggered at the top, their resident waiting CTAs cost this one-CTA-per-SM kernel its bandwidth on the
        # 148-SM part; triggered after the CTA's last tile's loads they still cost the op)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()
        E = counts.shape[0]
        N = mW_mkl.shape[0]
        K = mTok_nkl.shape[1]
        Tk = mOut.shape[0]
        tile_k = st.cta_tile_shape_mnk[2]
        kblocks_per_tile = tile_k // st.sf_vec
        sf_groups = kblocks_per_tile // SF_GROUP
        pcols = (
            (K // st.sf_vec + SF_ATOM_KBLOCKS - 1) // SF_ATOM_KBLOCKS
        ) * SF_ATOM_KBLOCKS  # scale-factor columns of the padded buffer

        @cute.struct
        class SharedStorage:
            w_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_ab_stage * 2]
            sf_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            group_off: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
            group_pre: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
            group_off_pad: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
            group_pre_pad: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # the TMA pipeline: the weight tile, its scale-factor atom and the token box of one k-tile per stage
        tma_producer, tma_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.w_full_mbar_ptr.data_ptr(),
            num_stages=st.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Warp),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        ).make_participants()
        # the cp.async pipeline: the token tile's scale-factor rows (one warp of producers)
        sf_pipeline = PipelineCpAsyncUmma.create(
            barrier_storage=storage.sf_full_mbar_ptr.data_ptr(),
            num_stages=st.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            defer_sync=True,
        )
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
            arch=st.TMEM_ALLOC_ARCH,
        )
        s_off = storage.group_off.get_tensor(cute.make_layout((MAX_E + 1,)))
        s_pre = storage.group_pre.get_tensor(cute.make_layout((MAX_E + 1,)))
        s_off_pad = storage.group_off_pad.get_tensor(cute.make_layout((MAX_E + 1,)))
        s_pre_pad = storage.group_pre_pad.get_tensor(cute.make_layout((MAX_E + 1,)))
        if cutlass.const_expr(self.legacy_tables):
            if warp_idx == EPILOGUE_WARPS[0]:
                build_group_tables(
                    counts, E, tn, s_off, s_pre, 1
                )  # the dense rows: the token tiles of a group
                build_group_tables(
                    counts, E, tn, s_off_pad, s_pre_pad, GROUP_ALIGN
                )  # the padded positions of the scale factors and row scales
        else:
            # the three tables in one scan on the scale-factor warp (idle until its first tile) while the allocator warp issues
            # the TMEM allocation: the two longest legs of the prologue overlap instead of running back to back
            if warp_idx == SF_WARP:
                build_group_tables_dual(
                    counts, E, tn, s_off, s_pre, s_off_pad, GROUP_ALIGN
                )
            if warp_idx < len(EPILOGUE_WARPS):
                tmem.allocate(st.num_tmem_alloc_cols)
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
        sSFW = smem.allocate_tensor(
            element_type=st.sf_dtype, layout=sfa_smem_layout_staged, byte_alignment=128
        )
        sSFA = smem.allocate_tensor(
            element_type=st.sf_dtype, layout=sfb_smem_layout_staged, byte_alignment=128
        )
        gW_mkl = cute.local_tile(
            mW_mkl, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFW_mkl = cute.local_tile(
            mSFW_mkl, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )
        k_tile_cnt = cute.size(gW_mkl, mode=[3])
        thr_mma = tiled_mma.get_slice(0)
        tCgW = thr_mma.partition_A(gW_mkl)
        tCgSFW = thr_mma.partition_A(gSFW_mkl)
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
        tWsSFW, tWgSFW = cpasync.tma_partition(
            tma_atom_sfw,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFW, 0, 3),
            cute.group_modes(tCgSFW, 0, 3),
        )
        tWsSFW = cute.filter_zeros(tWsSFW)
        tWgSFW = cute.filter_zeros(tWgSFW)
        tCrW = tiled_mma.make_fragment_A(sW)
        tCrTok = tiled_mma.make_fragment_B(sTok)
        acc_shape = tiled_mma.partition_shape_C(st.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, st.num_acc_stage)
        )
        tCtSFW_layout = bs_layout.make_tmem_layout_sfa(
            tiled_mma,
            st.mma_tiler,
            st.sf_vec,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFA_layout = bs_layout.make_tmem_layout_sfb(
            tiled_mma,
            st.mma_tiler,
            st.sf_vec,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        pipeline_init_wait(cluster_shape_mn=(1, 1))
        cute.arch.barrier()  # the group tables are complete

        raster_m = self.knobs.raster_along == "m"
        # the scheduler's "cluster rows" are the token tile, its "N tiles" the weight tiles
        if warp_idx == TMA_WARP:
            tile_sched = GroupScheduler.create(
                s_off, s_pre, counts, E, N, tn, WEIGHT_TILE, 1, raster_m, 0, 0
            )
            swap_tma_producer_loop(
                st,
                k_tile_cnt,
                tile_sched,
                tma_producer,
                tma_atom_w,
                tma_atom_sfw,
                tma_atom_tok,
                tWgW,
                tWgSFW,
                tWsW,
                tWsSFW,
                mTok_nkl,
                sTok,
                thr_mma,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                self.l2_hints,
            )
        if warp_idx == SF_WARP:
            tile_sched = GroupScheduler.create(
                s_off, s_pre, counts, E, N, tn, WEIGHT_TILE, 1, raster_m, 0, 0
            )
            cute.arch.griddepcontrol_wait()  # the token scale factors are the predecessor's
            swap_sf_producer_loop(
                st,
                k_tile_cnt,
                tile_sched,
                sf_pipeline,
                mSFA_flat,
                sSFA,
                tidx - 32 * SF_WARP,
                Tk,
                K,
                pcols,
                s_off,
                s_off_pad,
                tn,
                sf_groups,
                self.sf_stage_bytes,
            )
        if warp_idx == MMA_WARP:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtSFW = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + st.num_acc_tmem_cols, dtype=st.sf_dtype),
                tCtSFW_layout,
            )
            tCtSFA = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + st.num_acc_tmem_cols + st.num_sfa_tmem_cols,
                    dtype=st.sf_dtype,
                ),
                tCtSFA_layout,
            )
            tile_sched = GroupScheduler.create(
                s_off, s_pre, counts, E, N, tn, WEIGHT_TILE, 1, raster_m, 0, 0
            )
            swap_mma_loop(
                self.lowering,
                st,
                k_tile_cnt,
                tile_sched,
                tiled_mma,
                tma_consumer,
                sf_pipeline,
                acc_pipeline,
                tCrW,
                tCrTok,
                tCtAcc_base,
                tCtSFW,
                tCtSFA,
                sSFW,
                sSFA,
            )
        if warp_idx < len(EPILOGUE_WARPS):
            if cutlass.const_expr(self.legacy_tables):
                tmem.allocate(st.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = GroupScheduler.create(
                s_off, s_pre, counts, E, N, tn, WEIGHT_TILE, 1, raster_m, 0, 0
            )
            cute.arch.griddepcontrol_wait()  # the per-token scales are the predecessor's
            swapped_w4a4_epilogue_loop(
                st,
                self.chain,
                tidx,
                tile_sched,
                tCtAcc_base,
                thr_mma,
                mOut,
                Tk,
                N,
                epi_tile,
                acc_pipeline,
                mAlpha,
                mPts,
                s_off,
                s_off_pad,
                mOutSf,
                mAux,
            )
            tmem.relinquish_alloc_permit()
            # the allocator warp alone frees the columns: the epilogue warpgroup's warps join here after their last TMEM loads
            # (the allocation issued in the prologue's own region: the column count is given explicitly)
            epi_join.arrive_and_wait()
            if cutlass.const_expr(self.legacy_tables):
                tmem.free(tmem_ptr)
            else:
                tmem.free(tmem_ptr, st.num_tmem_alloc_cols)


@cute.jit
def _weight_hint(tile_sched, e: Int32, l2_hints: cutlass.Constexpr):
    """The weight tile's cache policy for expert ``e``: evict-first when the group spans one token tile (this CTA is its only
    reader), normal otherwise."""
    hint = Int64(L2_EVICT_NORMAL)
    if cutlass.const_expr(l2_hints):
        ee = cutlass.min(cutlass.max(e, Int32(0)), Int32(MAX_E - 1))
        single = (tile_sched.s_pre[ee + Int32(1)] - tile_sched.s_pre[ee]) == Int32(1)
        if single:
            hint = Int64(L2_EVICT_FIRST)
    return hint


@cute.jit
def _copy_hinted(atom, src, dst, bar, hint: Int64, l2_hints: cutlass.Constexpr):
    if cutlass.const_expr(l2_hints):
        cute.copy(atom, src, dst, tma_bar_ptr=bar, cache_policy=hint)
    else:
        cute.copy(atom, src, dst, tma_bar_ptr=bar)


@cute.jit
def swap_tma_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    tma_producer,
    tma_atom_w: cute.CopyAtom,
    tma_atom_sfw: cute.CopyAtom,
    tma_atom_tok: cute.CopyAtom,
    tWgW: cute.Tensor,
    tWgSFW: cute.Tensor,
    tWsW: cute.Tensor,
    tWsSFW: cute.Tensor,
    mTok_nkl: cute.Tensor,
    sTok: cute.Tensor,
    thr_mma,
    b_cta_coord,
    b_cta_layout,
    l2_hints: cutlass.Constexpr = False,
):
    """The TMA producer: per tile the expert's 128 weight rows ``n_tile`` and their scale-factor atom (the expert as the batch
    coordinate) and the token box re-addressed at the group's dense row ``row0`` (``token_tile`` rows: the rows past the group's
    end are computed and never stored, the rows past the buffer are zero-filled); k-tiles ascending, one ring slot holds the
    three. The weights are inputs of the whole chain, the token rows the predecessor's output: the first tile's weight stages
    (as many as the ring holds) are issued before the programmatic-dependent-launch wait and their token boxes right after it,
    so the ring fills while the predecessor drains; the stage's transaction barrier completes when all three copies have landed,
    whatever their order. ``l2_hints``: the weight tile and its scale factors carry the evict-first cache policy when the group
    spans one token tile (no other CTA reads them) and the normal one otherwise, the token box the evict-last policy (every
    weight tile of the group reads it); the hints steer the L2's replacement, never the bytes."""

    work_tile = tile_sched.initial_work_tile_info()
    n_pre = cutlass.min(Int32(st.num_ab_stage), k_tile_cnt)
    w_hint = _weight_hint(tile_sched, work_tile.e, l2_hints)
    if not work_tile.is_valid_tile:
        n_pre = Int32(0)  # a CTA without tiles issues nothing
    tWgW_slice = tWgW[(None, work_tile.n_tile, None, work_tile.e)]
    tWgSFW_slice = tWgSFW[(None, work_tile.n_tile, None, work_tile.e)]
    real_tok = cute.domain_offset((work_tile.row0, 0, 0), mTok_nkl)
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
    tma_producer.reset()
    tok_producer = (
        tma_producer.clone()
    )  # the same stages, walked again for the token boxes
    for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
        handle = tma_producer.acquire_and_advance()
        _copy_hinted(
            tma_atom_w,
            tWgW_slice[(None, handle.count)],
            tWsW[(None, handle.index)],
            handle.barrier,
            w_hint,
            l2_hints,
        )
        _copy_hinted(
            tma_atom_sfw,
            tWgSFW_slice[(None, handle.count)],
            tWsSFW[(None, handle.index)],
            handle.barrier,
            w_hint,
            l2_hints,
        )
    cute.arch.griddepcontrol_wait()  # the token rows are the predecessor's
    for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
        handle = tok_producer.current_handle()
        _copy_hinted(
            tma_atom_tok,
            tTgTok_slice[(None, handle.count)],
            tTsTok[(None, handle.index)],
            handle.barrier,
            Int64(L2_EVICT_LAST),
            l2_hints,
        )
        tok_producer.advance()
    k_start = n_pre
    while work_tile.is_valid_tile:
        tWgW_slice = tWgW[(None, work_tile.n_tile, None, work_tile.e)]
        tWgSFW_slice = tWgSFW[(None, work_tile.n_tile, None, work_tile.e)]
        real_tok = cute.domain_offset((work_tile.row0, 0, 0), mTok_nkl)
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
        w_hint = _weight_hint(tile_sched, work_tile.e, l2_hints)
        peek_empty = cutlass.Boolean(1)
        if k_start < k_tile_cnt:
            peek_empty = tma_producer.try_acquire()
        for _k_tile in cutlass.range(k_start, k_tile_cnt, 1, unroll=1):
            handle = tma_producer.acquire_and_advance(peek_empty)
            _copy_hinted(
                tma_atom_w,
                tWgW_slice[(None, handle.count)],
                tWsW[(None, handle.index)],
                handle.barrier,
                w_hint,
                l2_hints,
            )
            _copy_hinted(
                tma_atom_sfw,
                tWgSFW_slice[(None, handle.count)],
                tWsSFW[(None, handle.index)],
                handle.barrier,
                w_hint,
                l2_hints,
            )
            _copy_hinted(
                tma_atom_tok,
                tTgTok_slice[(None, handle.count)],
                tTsTok[(None, handle.index)],
                handle.barrier,
                Int64(L2_EVICT_LAST),
                l2_hints,
            )
            peek_empty = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_empty = tma_producer.try_acquire()
        tma_producer.reset()  # the next tile's k-tiles count from 0
        k_start = Int32(0)
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    tma_producer.tail()


@cute.jit
def gather_sf_row_padded(
    sf_atom,
    mSFA_flat: cute.Tensor,
    sSFA: cute.Tensor,
    r: Int32,
    tk: Int32,
    ok: Boolean,
    kt: Int32,
    stage: Int32,
    n_blocks: Int32,
    pcols: Int32,
    sf_stage_bytes: cutlass.Constexpr,
    sf_groups: cutlass.Constexpr,
    pred1,
):
    """The scale factors of tile row ``r`` (padded position ``tk``) for k-tile ``kt``: its ``sf_groups`` 4-byte block groups from
    the 128x4 swizzled buffer into the stage's atom by 4-byte ``cp.async``; a block group past the operand's ``n_blocks`` (the tail
    of a partial k-tile) is the same ``cp.async`` with a false predicate, a zero-fill (ignore-src) tracked by the stage's cp.async
    arrive -- never a generic store, whose visibility to the MMA's async-proxy reads that arrive does not order; a row that is not
    ``ok`` copies nothing (only the owning lane may touch a row's slot). The byte positions are the layout's: row ``r`` at
    ``(r % 32) * 16 + (r // 32) * 4``, block group ``g`` at ``g * 512``, atoms at ``128 * pcols``."""
    g_src_row = (
        (tk // Int32(128)) * (Int32(128) * pcols)
        + (tk % Int32(32)) * Int32(16)
        + ((tk % Int32(128)) // Int32(32)) * Int32(4)
    )
    g_dst_row = (
        stage * Int32(sf_stage_bytes)
        + (r % Int32(32)) * Int32(16)
        + (r // Int32(32)) * Int32(4)
    )
    for g in cutlass.range_constexpr(sf_groups):
        blk0 = (kt * Int32(sf_groups) + Int32(g)) * Int32(SF_GROUP)
        src_off = g_src_row + (kt * Int32(sf_groups) + Int32(g)) * Int32(SF_ATOM_BYTES)
        src_off = cute.assume(src_off, divby=4)
        dst_off = g_dst_row + Int32(g * SF_ATOM_BYTES)
        dst_off = cute.assume(dst_off, divby=4)
        src = cute.make_tensor(
            mSFA_flat.iterator + src_off, cute.make_layout((SF_GROUP,))
        )
        dst = cute.make_tensor(
            cute.recast_ptr(sSFA.iterator, dtype=Uint8) + dst_off,
            cute.make_layout((SF_GROUP,)),
        )
        pred1[0] = ok & (blk0 < n_blocks)
        if ok:
            cute.copy_atom_call(sf_atom, src, dst, pred=pred1)


@cute.jit
def swap_sf_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    sf_pipeline,
    mSFA_flat: cute.Tensor,
    sSFA: cute.Tensor,
    lane: Int32,
    Tk: Int32,
    K: Int32,
    pcols: Int32,
    s_off: cute.Tensor,
    s_off_pad: cute.Tensor,
    tn: cutlass.Constexpr,
    sf_groups: cutlass.Constexpr,
    sf_stage_bytes: cutlass.Constexpr,
):
    """The LDGSTS producer of the token tile's scale factors (one warp): lane ``j < tn`` owns tile row ``j`` = dense row
    ``row0 + j``, whose scale factors sit at the padded position ``row0 + j + (off_pad[e] - off[e])``; rows past the group's
    real end are skipped (their accumulator columns are never stored). Every lane arrives on the stage's barrier (the pipeline
    counts the warp)."""
    sf_atom = cute.make_copy_atom(cpasync.CopyG2SOp(), Uint8, num_bits_per_copy=32)
    pred1 = cute.make_rmem_tensor(cute.make_layout((1,)), Boolean)
    n_blocks = K // Int32(st.sf_vec)
    state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, st.num_ab_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        row0 = work_tile.row0
        row_end = cutlass.min(work_tile.row_end, Tk)
        e = work_tile.e
        shift = s_off_pad[e] - s_off[e]
        p = row0 + lane
        ok = (lane < Int32(tn)) & (p < row_end)
        tk = cutlass.max(p + shift, Int32(0))
        state.reset_count()
        peek = Boolean(1)
        if state.count < k_tile_cnt:
            peek = sf_pipeline.producer_try_acquire(state)
        for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
            sf_pipeline.producer_acquire(state, peek)
            gather_sf_row_padded(
                sf_atom,
                mSFA_flat,
                sSFA,
                lane,
                tk,
                ok,
                state.count,
                state.index,
                n_blocks,
                pcols,
                sf_stage_bytes,
                sf_groups,
                pred1,
            )
            sf_pipeline.producer_commit(state)
            state.advance()
            peek = Boolean(1)
            if state.count < k_tile_cnt:
                peek = sf_pipeline.producer_try_acquire(state)
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    sf_pipeline.producer_tail(state)


@cute.jit
def swap_mma_loop(
    lowering,
    st,
    k_tile_cnt: Int32,
    tile_sched,
    tiled_mma: cute.TiledMma,
    tma_consumer,
    sf_pipeline,
    acc_pipeline,
    tCrW: cute.Tensor,
    tCrTok: cute.Tensor,
    tCtAcc_base: cute.Tensor,
    tCtSFW: cute.Tensor,
    tCtSFA: cute.Tensor,
    sSFW: cute.Tensor,
    sSFA: cute.Tensor,
):
    """The block-scaled MMA loop with the weights, their scale factors and the token rows behind the TMA pipeline and the token
    scale factors behind the LDGSTS pipeline: both waited per k-tile, the per-block ``tcgen05.cp`` of both scale-factor sets,
    the lowering's inner steps with the SFA / SFB TMEM slices (the narrow token tile reads the SFB atom from its first rows: no
    window shift), ``ACCUMULATE`` off on the fold's first step only, both stages released by the commit."""
    assert cute.size(tCrW, mode=[2]) == lowering.inner, (
        "one smem stage must hold exactly the lowering's inner steps"
    )
    tiled_copy_sfw, tCsSFW_s2t, tCtSFW_s2t = s2t_copy_and_partition(
        sSFW, tCtSFW, st.cta_group, st.S2T_BROADCAST
    )
    tiled_copy_sfa, tCsSFA_s2t, tCtSFA_s2t = s2t_copy_and_partition(
        sSFA, tCtSFA, st.cta_group, st.S2T_BROADCAST
    )
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, st.num_acc_stage
    )
    sf_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, st.num_ab_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
        tma_consumer.reset()
        sf_state.reset_count()
        peek_w = tma_consumer.try_wait()
        acc_pipeline.producer_acquire(acc_state)
        for k_tile in range(k_tile_cnt):
            handle = tma_consumer.wait_and_advance(peek_w)
            sf_pipeline.consumer_wait(sf_state)
            w_stage = (None, None, None, None, handle.index)
            sf_stage = (None, None, None, None, sf_state.index)
            cute.copy(tiled_copy_sfw, tCsSFW_s2t[w_stage], tCtSFW_s2t)
            cute.copy(tiled_copy_sfa, tCsSFA_s2t[sf_stage], tCtSFA_s2t)
            for j in lowering.inner_steps():
                tiled_mma.set(tcgen05.Field.ACCUMULATE, lowering.accumulate(k_tile, j))
                tiled_mma.set(tcgen05.Field.SFA, tCtSFW[(None, None, j)].iterator)
                tiled_mma.set(tcgen05.Field.SFB, tCtSFA[(None, None, j)].iterator)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrW[(None, None, j, handle.index)],
                    tCrTok[(None, None, j, handle.index)],
                    tCtAcc,
                )
            handle.release()
            sf_pipeline.consumer_release(sf_state)
            sf_state.advance()
            peek_w = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_w = tma_consumer.try_wait()
        acc_pipeline.producer_commit(acc_state)
        acc_state.advance()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    acc_pipeline.producer_tail(acc_state)


@cute.jit
def _col_scale(
    alpha_e,
    mPts: cute.Tensor,
    row0: Int32,
    j: Int32,
    rows_left: Int32,
    shift: Int32,
):
    """``mul(alpha_e, pts[pad(row0 + j)])`` for token column ``j`` of the tile: the row scale of dense row ``row0 + j`` sits at the
    padded position ``+ shift`` (the group's padded offset less its dense one); columns past the valid tokens read a valid row
    and store nothing."""
    p = row0 + cutlass.min(j, cutlass.max(rows_left - Int32(1), Int32(0)))
    p = cutlass.max(p, Int32(0)) + shift
    p = cutlass.min(cutlass.max(p, Int32(0)), mPts.shape[0] - Int32(1))
    return w4a4_row_scale(MoeF32, alpha_e, mPts[p])


@cute.jit
def swapped_w4a4_epilogue_loop(
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
    mAlpha: cute.Tensor,
    mPts: cute.Tensor,
    s_off: cute.Tensor,
    s_off_pad: cute.Tensor,
    mOutSf: cute.Tensor = None,
    mAux: cute.Tensor = None,
):
    """The transposed group-masked epilogue: the accumulator tile is (weight rows, tokens); the 128-row TMEM layout gives
    thread ``t`` weight row ``t`` of the tile and its token columns (the record's N tile). Gated chain: the gate accumulators
    come from lane ``+8`` (``shfl.sync.down``); the up lanes (row % 16 < 8) compute the chain and store output column
    ``(n // 16) * 8 + n % 8``; plain and ungated chains: every lane stores column ``n``. Token ``j`` of the tile is dense row
    ``row0 + j``, stored iff ``< row_end``. The W4A4 per-token scale is per column, ``s_j = mul(alpha_e, pts[row0 + j +
    off_pad[e] - off[e]])`` (the row scales at the padded positions); the unscaled oracle chains take the accumulators as is.
    Every output element's chain is computed by one thread from the same accumulator value in the same fp32 sequence as the
    padded body's epilogue. The static class's GEMM1 (``out_fmt == "nvfp4"``) quantizes every live token column's 16-output
    block across the warp's output lanes (a block = the up lanes of one warp for the gated chain, 16 consecutive lanes for the
    ungated one): the block amax by a butterfly (exact in every association), the e4m3 scale and the encode scale on every lane,
    the codes paired into bytes by the even lanes (``mOut`` the packed-code buffer ``[rows, I/2]`` uint8, ``mOutSf`` the flat
    128x4 scale bytes at the padded positions, ``mAux[e] = gs2[e]``); its GEMM2 (``scale == "w4a4s"``): ``s = mul(alpha[e],
    mAux[e])`` once per tile."""
    pair = chain.pair
    scaled = chain.scale != "none"
    static_q = chain.out_fmt == "nvfp4"
    n_blocks = pcols = Int32(0)
    if cutlass.const_expr(static_q):
        N_out = N // Int32(2) if cutlass.const_expr(pair) else N
        n_blocks = N_out // Int32(NVFP4_BLOCK)
        pcols = ((n_blocks + Int32(3)) // Int32(4)) * Int32(4)
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
    cF_epi = cute.flat_divide(tCcF, epi_tile)
    tTR_cF = thr_t2r.partition_D(cF_epi)
    frag = tTR_cF[(None, None, None, 0, 0, 0, 0, 0)]
    offs = fragment_offsets(frag.layout)
    nv = len(offs)
    assert len({m for m, _ in offs}) == 1, (
        "the 128-row TMEM layout: one weight row per thread"
    )
    cols = [
        n - offs[0][1] for _, n in offs
    ]  # the register's token column inside the subtile
    tTR_rAcc = cute.make_rmem_tensor(frag.shape, ACC_DTYPE)
    tTR_cF = cute.group_modes(tTR_cF, 3, cute.rank(tTR_cF))
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
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        rows_left = cutlass.min(row_end, Tk) - row0  # valid tokens of this tile
        shift = s_off_pad[e] - s_off[e]  # the row scales at the padded positions
        alpha_e = None
        gs6 = inv_gs = None
        if cutlass.const_expr(chain.scale == "w4a4s"):
            alpha_e = w4a4_row_scale(
                MoeF32, mAlpha[e], mAux[e]
            )  # mul(alpha2[e], inv2[e]), once per tile
        elif cutlass.const_expr(scaled):
            alpha_e = mAlpha[e]
            if cutlass.const_expr(static_q):
                gs_e = mAux[e]  # the expert's static intermediate scale
                gs6 = MoeF32.mul(gs_e, MoeF32.const(NVFP4_C_SIXTH))
                inv_gs = MoeF32.div(MoeF32.const(1.0), gs_e)
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_state.index)]
        acc_pipeline.consumer_wait(acc_state)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for s in cutlass.range_constexpr(subtile_cnt):
            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
            crd0 = tTR_cF[(None, None, None, s)][(0, 0, 0)]
            wrow = crd0[0]  # this thread's weight row in the tile
            col0 = crd0[1] - Int32(offs[0][1])  # the subtile's first token column
            n_glob = n_tile * Int32(WEIGHT_TILE) + wrow
            # the gate accumulators of the up rows: lane + 8 (rows 16q + 8 + r sit 8 lanes below rows 16q + r)
            gate = None
            if cutlass.const_expr(pair):
                gate = [
                    cute.arch.shuffle_sync_down(tTR_rAcc[v], offset=8)
                    for v in range(nv)
                ]
                is_up = (wrow % Int32(2 * GROUP)) < Int32(GROUP)
                out_col = (
                    n_tile * Int32(WEIGHT_TILE // 2)
                    + (wrow // Int32(2 * GROUP)) * Int32(GROUP)
                    + wrow % Int32(GROUP)
                )
                store_ok = is_up & (n_glob < N)
            else:
                out_col = n_glob
                store_ok = n_glob < N
            # the chain of a token column is computed only where the column is a live token of the group (a warp-uniform
            # branch: every lane of the warp holds the same column), so a tile with few live tokens pays few chains
            if cutlass.const_expr(static_q):
                # the static class's GEMM1: per live token column, the block of this warp's output lanes
                if cutlass.const_expr(pair):
                    bcol = n_tile * Int32(
                        WEIGHT_TILE // (2 * NVFP4_BLOCK)
                    ) + wrow // Int32(2 * NVFP4_BLOCK)  # 16 outputs = 32 weight rows
                    even = (wrow % Int32(GROUP)) % Int32(2) == Int32(0)
                    sf_lane = (wrow % Int32(32)) == Int32(0)
                else:
                    bcol = n_tile * Int32(WEIGHT_TILE // NVFP4_BLOCK) + wrow // Int32(
                        NVFP4_BLOCK
                    )
                    even = (wrow % Int32(2)) == Int32(0)
                    sf_lane = (wrow % Int32(NVFP4_BLOCK)) == Int32(0)
                byte_col = out_col // Int32(2)
                for v in cutlass.range_constexpr(nv):
                    j = col0 + Int32(cols[v])
                    if j < rows_left:
                        sv = _col_scale(alpha_e, mPts, row0, j, rows_left, shift)
                        val = _scaled_elem(
                            chain,
                            tTR_rAcc[v],
                            gate[v] if cutlass.const_expr(pair) else None,
                            sv,
                        )
                        amax = MoeF32.fabs(val)
                        for off in (
                            (1, 2, 4, 16) if cutlass.const_expr(pair) else (1, 2, 4, 8)
                        ):
                            amax = MoeF32.fmaxc(
                                amax, cute.arch.shuffle_sync_bfly(amax, offset=off)
                            )
                        sf_byte, sf_d = cvt_f32_to_e4m3(MoeF32.mul(amax, gs6))
                        inv_b = MoeF32.fmin(
                            MoeF32.div(MoeF32.const(1.0), MoeF32.mul(sf_d, inv_gs)),
                            MoeF32.const(3.4028234663852886e38),
                        )
                        y = MoeF32.mul(val, inv_b)
                        y_odd = cute.arch.shuffle_sync_down(
                            y, offset=1
                        )  # the neighbouring output column's value
                        word = cvt_f32x8_to_e2m1x8(
                            y, y_odd, y, y, y, y, y, y
                        )  # byte 0 = (this column, the next)
                        prow = row0 + j
                        if store_ok & even & (bcol < n_blocks):
                            mOut[(prow, byte_col)] = (word & Uint32(0xFF)).to(Uint8)
                        if sf_lane & (bcol < n_blocks):
                            mOutSf[sf_byte_offset(prow + shift, bcol, pcols)] = (
                                sf_byte.to(Uint8)
                            )
            else:
                for v in cutlass.range_constexpr(nv):
                    j = col0 + Int32(cols[v])
                    if j < rows_left:
                        sv = None
                        if cutlass.const_expr(chain.scale == "w4a4s"):
                            sv = alpha_e
                        elif cutlass.const_expr(scaled):
                            sv = _col_scale(alpha_e, mPts, row0, j, rows_left, shift)
                        val = _scaled_elem(
                            chain,
                            tTR_rAcc[v],
                            gate[v] if cutlass.const_expr(pair) else None,
                            sv,
                        )
                        if store_ok:
                            mOut[(row0 + j, out_col)] = val
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_state)
        acc_state.advance()
