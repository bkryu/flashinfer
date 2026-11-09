# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The grouped block-scaled (NVFP4) GEMM kernel body: the dense block-scaled blocks of the 4-bit GEMM classes
(:mod:`flashinfer.gemm.batch_invariant.blocks.blockscaled`: the plan with the K = 64 or the sm_107 K = 128 instruction,
the SFA / SFB TMA, smem and TMEM layouts, ``blockscaled_mma_loop`` issuing ``MainloopBlock.lowering()`` with the
per-stage ``tcgen05.cp`` of the scale factors) under the MoE group scheduler (:mod:`.scheduler`), with per-tile TMA
coordinates (A and SFA re-partitioned from the group-offset view; B and SFB indexed by the expert as the batch coordinate)
and the group-masked epilogue with the W4A4 chains (:mod:`.epilogue_nvfp4`).

Operands: A ``(rows, K/2, 1)`` packed fp4 as uint8 (the permuted activations, K contiguous), SFA the flat uint8
scale-factor buffer of the ``(rows_sf, K)`` operand in the 128x4 swizzled layout, B ``(N, K/2, E)`` packed fp4 (the stacked
expert weights, ``l`` = expert), SFB ``E`` consecutive 128x4 buffers of ``(N, K)``, out ``(rows, Nout)`` bf16 / f32
row-major, ``counts (E,)`` int32, ``alpha (E,)`` fp32 (the experts' alphas as given), ``pts (rows_sf,)`` fp32 (the per-token
scales of the A rows). **A group's scale factors start at a 128-row atom** (``GROUP_ALIGN``): the swizzled scale layout
is affine only inside an atom, so a TMA box of SFA must start at an atom boundary. The padded packing pads the rows
themselves (the group's rows are ``[off_e, off_e + counts[e])`` with ``off`` the exclusive prefix of the counts rounded up to
the atom; the padding rows, zero-filled by the caller, are computed and never stored); the dense packing (``dense_a``) keeps
the rows back to back and places the scale factors and the row scales at the padded positions (a second offset table), so
the output is dense and the stale rows of a partial tile (the next group's rows) are never stored. Both are placements of
the same per-row fold.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as bs_layout
from cutlass import Int32
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.tensor_utils import LayoutEnum

from ....gemm.batch_invariant.blocks.blockscaled import (
    A_RING_MAX_BYTES,
    PLANS,
    SF_ATOM_BYTES,
    blockscaled_mma_loop,
    fp4_view,
    sf_view,
)
from ....gemm.batch_invariant.blocks.operands import (
    ACC_DTYPE,
    BAR_TMEM_ALLOC,
    MBAR_RESERVE_BYTES,
    NUM_C_STAGE_MIN,
)
from ....gemm.batch_invariant.generator import Knobs
from ....gemm.batch_invariant.spec import NVFP4_FMTS, SF_ATOM_KBLOCKS, SF_ATOM_ROWS
from .epilogue import staged_store, staging_bytes, staging_layout
from .epilogue_nvfp4 import W4A4Chain, grouped_w4a4_epilogue_loop
from .scheduler import MAX_E, GroupScheduler, build_group_tables, warp_roles

GROUP_ALIGN = SF_ATOM_ROWS  # the group alignment of the scale factors (rows): one scale-factor atom
#: Epilogue warpgroups of the NVFP4 grouped kernel: the 4-bit MMA leaves the declared fp32 chain as the tile's longest leg,
#: so four warpgroups split a tile's epilogue subtiles (the bf16 kernel takes two). A placement.
EPI_WARPGROUPS_NVFP4 = 4
EPILOGUE_WARPS, MMA_WARP, TMA_WARP, THREADS = warp_roles(EPI_WARPGROUPS_NVFP4)


def sf_view_experts(sf: cute.Tensor, rows, k, E, vec: int, sf_dtype):
    """The ``E`` consecutive 128x4 swizzled scale-factor buffers of ``(rows, k)`` operands as the ``(rows, k, E)`` scale tensor
    the block-scaled TMA reads: tiles of 128 rows x 4 K-blocks (512 B), K-tiles consecutive, row-tiles strided by the
    K-tiles, experts strided by one whole buffer (``sf_bytes(rows, k)``)."""
    mt = cute.ceil_div(rows, SF_ATOM_ROWS)
    kt = cute.ceil_div(k, vec * SF_ATOM_KBLOCKS)
    ptr = cute.recast_ptr(sf.iterator, dtype=sf_dtype)
    layout = cute.make_layout(
        (((32, 4), mt), ((vec, SF_ATOM_KBLOCKS), kt), E),
        stride=(
            ((16, 4), SF_ATOM_BYTES * kt),
            ((0, 1), SF_ATOM_BYTES),
            SF_ATOM_BYTES * kt * mt,
        ),
    )
    return cute.make_tensor(ptr, layout)


class GroupedBlockScaledTma:
    """TMA atoms and coordinate tensors for A, SFA (re-partitioned per tile in the producer), B and SFB (the expert as
    the batch coordinate); the transaction bytes of one stage of all four operands. The dense ``BlockScaledTmaSetup``
    without the output store atom (the grouped epilogue stores from registers)."""

    def __init__(
        self, st, a: cute.Tensor, b: cute.Tensor, sfa: cute.Tensor, sfb: cute.Tensor
    ):
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        a_one = cute.slice_(st.a_smem_layout_staged, (None, None, None, 0))
        self.atom_a, self.tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_one, st.mma_tiler, st.tiled_mma, st.cluster_layout_vmnk.shape
        )
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        b_one = cute.slice_(st.b_smem_layout_staged, (None, None, None, 0))
        self.atom_b, self.tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_one, st.mma_tiler, st.tiled_mma, st.cluster_layout_vmnk.shape
        )
        sfa_op = utils.sm100.cluster_shape_to_tma_atom_A(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        sfa_one = cute.slice_(st.sfa_smem_layout_staged, (None, None, None, 0))
        self.atom_sfa, self.tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa,
            sfa_one,
            st.mma_tiler,
            st.tiled_mma,
            st.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_op = utils.sm100.cluster_shape_to_tma_atom_SFB(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        sfb_one = cute.slice_(st.sfb_smem_layout_staged, (None, None, None, 0))
        self.atom_sfb, tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb,
            sfb_one,
            st.mma_tiler_sfb,
            st.tiled_mma_sfb,
            st.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        if st.cta_tile_shape_mnk[1] == 192:
            # a 192-wide N tile spans one and a half scale-factor atoms: N tile n (a 256-row box) starts at atom 3(n//2) + n%2
            # (the dense block's re-addressing)
            x = tensor_sfb.stride[0][1]
            y = cute.ceil_div(tensor_sfb.shape[0][1], 4)
            new_shape = (
                (tensor_sfb.shape[0][0], ((2, 2), y)),
                tensor_sfb.shape[1],
                tensor_sfb.shape[2],
            )
            new_stride = (
                (tensor_sfb.stride[0][0], ((x, x), 3 * x)),
                tensor_sfb.stride[1],
                tensor_sfb.stride[2],
            )
            tensor_sfb = cute.make_tensor(
                tensor_sfb.iterator, cute.make_layout(new_shape, stride=new_stride)
            )
        self.tensor_sfb = tensor_sfb
        self.tx_bytes = (
            cute.size_in_bytes(st.a_dtype, a_one)
            + cute.size_in_bytes(st.b_dtype, b_one)
            + cute.size_in_bytes(st.sf_dtype, sfa_one)
            + cute.size_in_bytes(st.sf_dtype, sfb_one)
        ) * st.atom_thr_size


def grouped_plan(plan_cls, knobs: Knobs, own_bytes):
    """The block-scaled plan with the grouped kernel's own stage rule: the operand ring (A, B, SFA, SFB per stage) fills the
    CTA's shared memory after the barrier reserve and the kernel's own buffers (``own_bytes(st)``: the group tables, the
    epilogue staging buffers), under the dense plan's A-ring cap. The dense plan's output stages are not reserved (the
    grouped epilogue stores from its own staging buffers). A placement: the ring depth moves no bits."""

    class _Plan(plan_cls):
        def _stages(self):
            a_one = utils.sm100.make_smem_layout_a(
                self.tiled_mma, self.mma_tiler, self.a_dtype, 1
            )
            b_one = utils.sm100.make_smem_layout_b(
                self.tiled_mma, self.mma_tiler, self.b_dtype, 1
            )
            sfa_one = bs_layout.make_smem_layout_sfa(
                self.tiled_mma, self.mma_tiler, self.sf_vec, 1
            )
            sfb_one = bs_layout.make_smem_layout_sfb(
                self.tiled_mma, self.mma_tiler, self.sf_vec, 1
            )
            a_bytes = cute.size_in_bytes(self.a_dtype, a_one)
            ab_bytes = (
                a_bytes
                + cute.size_in_bytes(self.b_dtype, b_one)
                + cute.size_in_bytes(self.sf_dtype, sfa_one)
                + cute.size_in_bytes(self.sf_dtype, sfb_one)
            )
            num_ab = (
                self.smem_capacity - MBAR_RESERVE_BYTES - own_bytes(self)
            ) // ab_bytes
            num_ab = min(num_ab, (A_RING_MAX_BYTES - 1) // a_bytes)
            assert num_ab >= 2, (
                "the grouped kernel's operand ring does not fit the shared memory",
                self.knobs,
                num_ab,
            )
            return num_ab, NUM_C_STAGE_MIN

    return _Plan(knobs, ACC_DTYPE, LayoutEnum.ROW_MAJOR)


class GroupedBlockScaledGemm:
    """``out[p, n] = chain(sum_k MMA(A[p], SFA[p], B[n, e(p)], SFB[n, e(p)]))`` for every row ``p`` of every group, with the
    W4A4 chains (``W4A4Chain.scale == "w4a4"``: ``s = mul(alpha[e], pts[p])``) or the unscaled oracle chains. The persistent
    grid (in clusters) is a dynamic launch argument: which cluster computes a tile, never the bits."""

    def __init__(
        self,
        knobs: Knobs,
        chain: W4A4Chain,
        in_fmt: str = "nvfp4",
        dense_a: bool = False,
    ):
        """``dense_a``: the A rows densely packed (group offsets at align 1), the scale factors and the row scales at the
        padded positions (``pts`` has the padded row count, the scale-factor buffer the padded rows), the output ``Tk`` dense
        rows. Default: the padded packing (the rows of every group start at a multiple of ``GROUP_ALIGN``)."""
        assert isinstance(chain, W4A4Chain) and chain.out_fmt in ("bf16", "f32"), chain
        assert in_fmt in NVFP4_FMTS, in_fmt
        self.knobs, self.chain, self.in_fmt, self.dense_a = (
            knobs,
            chain,
            in_fmt,
            dense_a,
        )
        self.plan_cls = PLANS[in_fmt]
        self.cluster_shape_mn = knobs.cluster_shape_mn

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        alpha: cute.Tensor,
        pts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        plan_cls = self.plan_cls
        a4 = fp4_view(a)
        b4 = fp4_view(b)
        sfa_t = sf_view(
            sfa, pts.shape[0], a4.shape[1], 1, plan_cls.SF_VEC, plan_cls.SF_DT
        )  # the scale rows: padded
        sfb_t = sf_view_experts(
            sfb, b4.shape[0], b4.shape[1], b4.shape[2], plan_cls.SF_VEC, plan_cls.SF_DT
        )
        self.staged = staged_store(self.chain.out_fmt)
        tables = (4 if self.dense_a else 2) * (MAX_E + 1) * 4
        st = grouped_plan(
            plan_cls,
            self.knobs,
            lambda st: (
                tables
                + (
                    staging_bytes(out.element_type, st.epi_tile, EPI_WARPGROUPS_NVFP4)
                    if self.staged
                    else 0
                )
            ),
        )
        self.st = st
        self.c_smem_layout_staged = (
            staging_layout(out.element_type, st.epi_tile, EPI_WARPGROUPS_NVFP4)
            if self.staged
            else st.a_smem_layout_staged
        )
        self.block = st.block
        assert self.block.in_fmt == self.in_fmt, (self.block.in_fmt, self.in_fmt)
        self.lowering = self.block.lowering()
        assert st.mma_tiler[2] == self.block.k_block, (st.mma_tiler, self.block.k_block)
        assert st.cluster_k == 1 and not self.knobs.swap_ab and not st.breuse, (
            "the grouped body takes the dense records"
        )
        tma = GroupedBlockScaledTma(st, a4, b4, sfa_t, sfb_t)
        self.num_tma_load_bytes = tma.tx_bytes
        cm, cn = self.cluster_shape_mn
        grid = (cm, cn, max_active_clusters)
        self.kernel(
            st.tiled_mma,
            st.tiled_mma_sfb,
            tma.atom_a,
            tma.tensor_a,
            tma.atom_b,
            tma.tensor_b,
            tma.atom_sfa,
            tma.tensor_sfa,
            tma.atom_sfb,
            tma.tensor_sfb,
            out,
            counts,
            alpha,
            pts,
            st.cluster_layout_vmnk,
            st.cluster_layout_sfb_vmnk,
            st.a_smem_layout_staged,
            st.b_smem_layout_staged,
            st.sfa_smem_layout_staged,
            st.sfb_smem_layout_staged,
            st.epi_tile,
            self.c_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[THREADS, 1, 1],
            cluster=(cm, cn, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        mOut: cute.Tensor,
        counts: cute.Tensor,
        mAlpha: cute.Tensor,
        mPts: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        c_smem_layout_staged,
    ):
        st = self.st
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
        # programmatic dependent launch: wait for the predecessor grid before any dependent read; no explicit trigger -- the
        # dependents launch at this grid's completion, which measured faster on the op than a trigger at the top (the dependents'
        # CTAs resident on the SMs while the walk still needs them: the swapped body at half speed on SM100) and than one issued
        # after the CTA's last tile's loads
        cute.arch.griddepcontrol_wait()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        use_2cta_instrs = atom_thr_size == 2
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % atom_thr_size
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()
        E = counts.shape[0]
        N = mB_nkl.shape[0]
        Tk = mA_mkl.shape[0]

        # --- scaffolding: barriers, pipelines, TMEM allocator (as the dense block-scaled body), the group tables
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            group_off: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
            group_pre: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]

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
                pipeline.Agent.Thread,
                len(EPILOGUE_WARPS) * (2 if use_2cta_instrs else 1),
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=BAR_TMEM_ALLOC, num_threads=32 * (len(EPILOGUE_WARPS) + 1)
        )
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=EPILOGUE_WARPS[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=st.TMEM_ALLOC_ARCH,
        )
        s_off = storage.group_off.get_tensor(cute.make_layout((MAX_E + 1,)))
        s_pre = storage.group_pre.get_tensor(cute.make_layout((MAX_E + 1,)))
        s_off_pad = None
        if cutlass.const_expr(self.dense_a):
            # the padded (atom-aligned) group offsets: the scale-factor and row-scale positions of the dense rows
            s_off_pad = smem.allocate_tensor(
                Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
            )
            s_pre_pad = smem.allocate_tensor(
                Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
            )
        cluster_rows = st.cta_tile_shape_mnk[0] * self.cluster_shape_mn[0]
        if warp_idx == EPILOGUE_WARPS[0]:
            if cutlass.const_expr(self.dense_a):
                build_group_tables(counts, E, cluster_rows, s_off, s_pre, 1)
                build_group_tables(
                    counts, E, cluster_rows, s_off_pad, s_pre_pad, GROUP_ALIGN
                )
            else:
                build_group_tables(counts, E, cluster_rows, s_off, s_pre, GROUP_ALIGN)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # --- operand staging: smem rings (A, B swizzled; SFA, SFB plain), multicast masks, the B / SFB partitions
        sA = smem.allocate_tensor(
            element_type=st.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=st.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=st.sf_dtype, layout=sfa_smem_layout_staged, byte_alignment=128
        )
        sSFB = smem.allocate_tensor(
            element_type=st.sf_dtype, layout=sfb_smem_layout_staged, byte_alignment=128
        )
        sC = None
        if cutlass.const_expr(self.staged):  # one output subtile per epilogue warpgroup
            if cutlass.const_expr(
                isinstance(c_smem_layout_staged, cute.ComposedLayout)
            ):
                sC = smem.allocate_tensor(
                    element_type=mOut.element_type,
                    layout=c_smem_layout_staged.outer,
                    byte_alignment=128,
                    swizzle=c_smem_layout_staged.inner,
                )
            else:
                sC = smem.allocate_tensor(
                    element_type=mOut.element_type,
                    layout=c_smem_layout_staged,
                    byte_alignment=128,
                )
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(st.is_a_mcast or st.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(st.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(st.mma_tiler_sfb, (0, None, None)), (None, None, None)
        )
        k_tile_cnt = cute.size(gB_nkl, mode=[3])
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(st.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, st.num_acc_stage)
        )
        tCtSFA_layout = bs_layout.make_tmem_layout_sfa(
            tiled_mma,
            st.mma_tiler,
            st.sf_vec,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFB_layout = bs_layout.make_tmem_layout_sfb(
            tiled_mma,
            st.mma_tiler,
            st.sf_vec,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        mma_m_in_cluster = block_in_cluster_coord_vmnk[1]
        cta_n_in_cluster = block_in_cluster_coord_vmnk[2]
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        cute.arch.barrier()  # the group tables are complete

        raster_m = self.knobs.raster_along == "m"
        tn, cn = st.mma_tiler[1], self.cluster_shape_mn[1]
        # --- roles
        if warp_idx == TMA_WARP:
            tile_sched = GroupScheduler.create(
                s_off,
                s_pre,
                counts,
                E,
                N,
                cluster_rows,
                tn,
                cn,
                raster_m,
                mma_m_in_cluster,
                cta_n_in_cluster,
            )
            grouped_blockscaled_producer_loop(
                st,
                k_tile_cnt,
                tile_sched,
                ab_producer,
                tma_atom_a,
                tma_atom_b,
                tma_atom_sfa,
                tma_atom_sfb,
                mA_mkl,
                mSFA_mkl,
                tBgB,
                tBgSFB,
                sA,
                sSFA,
                tBsB,
                tBsSFB,
                thr_mma,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                mma_m_in_cluster,
                a_full_mcast_mask,
                b_full_mcast_mask,
                sfb_full_mcast_mask,
                s_off,
                s_off_pad,
            )
        if warp_idx == MMA_WARP:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtSFA = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + st.num_acc_tmem_cols, dtype=st.sf_dtype),
                tCtSFA_layout,
            )
            tCtSFB = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + st.num_acc_tmem_cols + st.num_sfa_tmem_cols,
                    dtype=st.sf_dtype,
                ),
                tCtSFB_layout,
            )
            tile_sched = GroupScheduler.create(
                s_off,
                s_pre,
                counts,
                E,
                N,
                cluster_rows,
                tn,
                cn,
                raster_m,
                mma_m_in_cluster,
                cta_n_in_cluster,
            )
            blockscaled_mma_loop(
                self.lowering,
                st,
                k_tile_cnt,
                is_leader_cta,
                tile_sched,
                tiled_mma,
                ab_consumer,
                acc_pipeline,
                tCrA,
                tCrB,
                tCtAcc_base,
                tmem_ptr,
                tCtSFA,
                tCtSFB,
                sSFA,
                sSFB,
            )
        if warp_idx < MMA_WARP:
            tmem.allocate(st.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = GroupScheduler.create(
                s_off,
                s_pre,
                counts,
                E,
                N,
                cluster_rows,
                tn,
                cn,
                raster_m,
                mma_m_in_cluster,
                cta_n_in_cluster,
            )
            grouped_w4a4_epilogue_loop(
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
                mma_m_in_cluster,
                mAlpha,
                mPts,
                EPI_WARPGROUPS_NVFP4,
                s_off,
                s_off_pad,
                sC,
                mma_tile_coord_v,
            )
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


@cute.jit
def grouped_blockscaled_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    ab_producer,
    tma_atom_a: cute.CopyAtom,
    tma_atom_b: cute.CopyAtom,
    tma_atom_sfa: cute.CopyAtom,
    tma_atom_sfb: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    mSFA_mkl: cute.Tensor,
    tBgB: cute.Tensor,
    tBgSFB: cute.Tensor,
    sA: cute.Tensor,
    sSFA: cute.Tensor,
    tBsB: cute.Tensor,
    tBsSFB: cute.Tensor,
    thr_mma,
    a_cta_coord,
    a_cta_layout,
    mma_m_in_cluster,
    a_mcast_mask,
    b_mcast_mask,
    sfb_mcast_mask,
    s_off=None,
    s_off_pad=None,
):
    """The block-scaled producer with per-tile coordinates: A and SFA are re-partitioned from the group-offset view (the
    scale-factor row a multiple of the atom, so the swizzled SFA box is addressed exactly; the dense packing shifts it by
    ``off_pad[e] - off[e]``), B and SFB are the dense partitions indexed by the expert as the batch coordinate. k-tiles
    ascending, one ring slot holds A, B, SFA and SFB of a k-tile. A 64-wide N tile reads half of a 128-row SFB atom; the
    192-wide tile's addressing lives in the SFB coordinate tensor."""
    tn = st.cta_tile_shape_mnk[1]
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        real_a = cute.domain_offset((work_tile.row0, 0, 0), mA_mkl)
        gA = cute.local_tile(
            real_a, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )
        tCgA = thr_mma.partition_A(gA)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            a_cta_coord,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tAgA_slice = tAgA[(None, mma_m_in_cluster, None, 0)]
        row0_sf = work_tile.row0
        if cutlass.const_expr(s_off_pad is not None):
            row0_sf = work_tile.row0 + (s_off_pad[work_tile.e] - s_off[work_tile.e])
        real_sfa = cute.domain_offset((row0_sf, 0, 0), mSFA_mkl)
        gSFA = cute.local_tile(
            real_sfa, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )
        tCgSFA = thr_mma.partition_A(gSFA)
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            a_cta_coord,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tAgSFA_slice = tAgSFA[(None, mma_m_in_cluster, None, 0)]
        n_tile, e = work_tile.n_tile, work_tile.e
        tBgB_slice = tBgB[(None, n_tile, None, e)]
        sfb_n = n_tile
        if cutlass.const_expr(tn < 128):
            sfb_n = n_tile // (SF_ATOM_ROWS // tn)  # 64-wide: half an atom
        tBgSFB_slice = tBgSFB[(None, sfb_n, None, e)]
        ab_producer.reset()
        peek_empty = ab_producer.try_acquire()
        for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
            handle = ab_producer.acquire_and_advance(peek_empty)
            cute.copy(
                tma_atom_a,
                tAgA_slice[(None, handle.count)],
                tAsA[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=a_mcast_mask,
            )
            cute.copy(
                tma_atom_b,
                tBgB_slice[(None, handle.count)],
                tBsB[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=b_mcast_mask,
            )
            cute.copy(
                tma_atom_sfa,
                tAgSFA_slice[(None, handle.count)],
                tAsSFA[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=a_mcast_mask,
            )
            cute.copy(
                tma_atom_sfb,
                tBgSFB_slice[(None, handle.count)],
                tBsSFB[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=sfb_mcast_mask,
            )
            peek_empty = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_empty = ab_producer.try_acquire()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    ab_producer.tail()
