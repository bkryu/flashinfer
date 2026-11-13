# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The grouped block-scaled GEMM kernel body of the MXFP8 MoE classes: the block-scaled dense blocks
(``gemm/batch_invariant/blocks/blockscaled.py``: the MX plan of the part -- the SM100 / SM103 K = 32 step or the SM107
K = 64 step -- with its SFA / SFB TMA, smem and TMEM layouts, and ``blockscaled_mma_loop`` issuing the dense class's
instruction sequence with the per-stage ``tcgen05.cp`` of the scale factors) under the MoE group scheduler, with per-tile
TMA coordinates (A and SFA re-partitioned from the group-offset view; B and SFB indexed by the expert as the batch
coordinate) and the group-masked epilogue with the bf16 classes' chains (the UE8M0 block scales are operands of the
instruction, so the epilogue sees dequantised accumulators and applies no scale).

Operands: A ``(Tk_pad, K, 1)`` e4m3 K-major (the permuted activations), SFA the flat uint8 UE8M0 buffer of the
``(Tk_pad, K)`` operand in the 128x4 swizzled layout, B ``(N, K, E)`` e4m3 K-major (the stacked expert weights, ``l`` =
the expert), SFB ``E`` consecutive 128x4 buffers of ``(N, K)``, out ``(Tk_pad, Nout)`` row-major (bf16 / f16 / f32),
``counts (E,)`` int32. **Group offsets are padded to 128 rows** (one scale-factor atom): the swizzled scale layout is
affine only inside an atom, so a TMA box of SFA must start at an atom boundary; the real rows of a group are ``[off_e,
off_e + counts[e])`` and the padding rows (whatever bytes they hold) are computed and never stored.
"""

from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as bs_layout
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.tensor_utils import LayoutEnum

from ....gemm.batch_invariant.blocks.blockscaled import (
    PLANS,
    SF_ATOM_BYTES,
    BlockScaledPlan,
    blockscaled_mma_loop,
    sf_view,
)
from ....gemm.batch_invariant.blocks.operands import (
    ACC_DTYPE,
    BAR_TMEM_ALLOC,
    MBAR_RESERVE_BYTES,
)
from ....gemm.batch_invariant.generator import Knobs
from ....batch_invariant.quantization.layout import SF_COL_TILE
from ....gemm.batch_invariant.spec import SF_ATOM_KBLOCKS, SF_ATOM_ROWS
from .epilogue import (
    GroupedChain,
    grouped_epilogue_loop,
    staged_store,
    staging_bytes,
    staging_layout,
)
from .scheduler import (
    MAX_E,
    GroupScheduler,
    build_group_tables,
    build_group_tables_dual,
    warp_roles,
)

#: The MX classes the grouped body serves: the SM100 / SM103 instruction (K = 32) and the SM107 one (K = 64).
MX_FMTS = ("mxfp8", "mxfp8_k64")
#: The padded group alignment (rows): one 128-row scale-factor atom.
GROUP_ALIGN = SF_ATOM_ROWS
#: Epilogue warpgroups of the block-scaled grouped kernel (the chains are the epilogue leg once the stores are vectors).
from .epilogue_mx import (  # noqa: E402
    BAR_EPI_JOIN,
    grouped_epilogue_quant_loop,
    xch_bytes,
    xch_layout,
)

EPI_WARPGROUPS_MX = 4
EPILOGUE_WARPS, MMA_WARP, TMA_WARP, THREADS = warp_roles(EPI_WARPGROUPS_MX)
TABLE_BYTES = (MAX_E + 1) * 4  # one group table


def sf_view_experts(sf: cute.Tensor, rows, k, E, vec: int, sf_dtype):
    """The ``E`` consecutive 128x4 swizzled scale-factor buffers of ``(rows, k)`` operands as the ``(rows, k, E)`` scale
    tensor the block-scaled TMA reads (one UE8M0 per ``vec`` elements): tiles of 128 rows x 4 K-blocks (512 B), K-tiles
    consecutive, row-tiles strided by the K-tiles, experts strided by one whole buffer."""
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
        self,
        st: BlockScaledPlan,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
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
            # a 192-wide N tile spans one and a half 128-row scale-factor atoms: N tile n (a 256-row box) starts at atom
            # 3(n//2) + n%2 (the dense block's re-addressing)
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


def ring_bytes(st) -> int:
    """Shared memory of the operand rings of a block-scaled plan (A, B, SFA, SFB, all stages)."""
    return (
        cute.size_in_bytes(st.a_dtype, st.a_smem_layout_staged)
        + cute.size_in_bytes(st.b_dtype, st.b_smem_layout_staged)
        + cute.size_in_bytes(st.sf_dtype, st.sfa_smem_layout_staged)
        + cute.size_in_bytes(st.sf_dtype, st.sfb_smem_layout_staged)
    )


def umma_ring_bytes(st) -> int:
    """Shared memory of the M-side operand ring a UMMA reads (A, all stages: the first ring behind the barriers and the kernel's
    own buffers in every grouped body), the ring the plans' address bound applies to."""
    return cute.size_in_bytes(st.a_dtype, st.a_smem_layout_staged)


def plan_with_room(
    plan_cls,
    knobs: Knobs,
    own_bytes,
    umma_addr_max: Optional[int] = None,
    k_blocks: Optional[int] = None,
):
    """The block-scaled plan whose operand ring leaves room for the grouped kernel's own shared memory (the group tables,
    the epilogue staging buffers): built once; if the ring plus ``own_bytes(st)`` plus the barrier reserve overflows the
    CTA's shared memory, rebuilt with the budget shortened by the overflow (one stage fewer). ``umma_addr_max``: the
    M-side operand ring a UMMA reads (A, first behind the barriers and the kernel's own buffers) ends below this many bytes of
    the CTA's shared memory -- the plan's ``UMMA_SMEM_ADDR_MAX``, the bound above which the M-side operand ring (A) reads
    wrong data on the sm_100 instruction plans of this part; the B and scale-factor rings above it compute the class (the
    long-K test over every record is the guard). ``k_blocks``: the plan's K steps per stage (the swapped body's two-atom
    stage; default the format's). A placement: the ring depth moves no bits."""
    st = plan_cls(knobs, ACC_DTYPE, LayoutEnum.ROW_MAJOR, k_blocks=k_blocks)
    over = ring_bytes(st) + own_bytes(st) + MBAR_RESERVE_BYTES - st.smem_capacity
    if umma_addr_max is not None:
        over = max(
            over,
            umma_ring_bytes(st) + own_bytes(st) + MBAR_RESERVE_BYTES - umma_addr_max,
        )
    if over <= 0:
        return st
    cut = over
    for _ in range(
        64
    ):  # the plan's own leftover may absorb part of the cut: shorten until the ring fits

        class _Plan(plan_cls):  # type: ignore[valid-type, misc]
            def _stages(self):
                self.smem_capacity -= cut
                try:
                    return super()._stages()
                finally:
                    self.smem_capacity += cut

        st2 = _Plan(knobs, ACC_DTYPE, LayoutEnum.ROW_MAJOR, k_blocks=k_blocks)
        fits = (
            ring_bytes(st2) + own_bytes(st2) + MBAR_RESERVE_BYTES <= st2.smem_capacity
        )
        if umma_addr_max is not None:
            fits = fits and (
                umma_ring_bytes(st2) + own_bytes(st2) + MBAR_RESERVE_BYTES
                <= umma_addr_max
            )
        if fits:
            return st2
        cut += 1024
    raise AssertionError(
        ("the grouped kernel's shared memory does not fit", knobs, over)
    )


class GroupedBlockScaledGemmMX:
    """``out[p, n] = chain(sum_k MMA(A[p], SFA[p], B[n, e(p)], SFB[n, e(p)]))`` for every row ``p`` of every group: the
    dense MX class's fold per row (the block scales inside the step), the bf16 classes' chain on the accumulators. The
    persistent grid (``max_active_clusters``, in clusters) is a dynamic launch argument: which cluster computes a tile,
    never the bits."""

    def __init__(
        self,
        knobs: Knobs,
        chain: GroupedChain,
        in_fmt: str = "mxfp8",
        dense_a: bool = False,
    ):
        """``dense_a``: the A rows densely packed (group offsets at align 1; the TMA box of a tile starts at the group's
        row), the scale factors at the padded positions (one 128x4 atom per (expert, 128-row tile): the swapped decode
        body's packing, :mod:`.swap_mxfp8`), the output's rows the A rows; the stale rows of a partial tile (the next
        group's rows) are never stored. Default: the padded packing."""
        assert isinstance(chain, GroupedChain), chain
        assert in_fmt in MX_FMTS, in_fmt
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
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._launch(a, b, sfa, sfb, out, counts, max_active_clusters, stream)

    @cute.jit
    def call_quant(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        out: cute.Tensor,
        out_sf: cute.Tensor,
        counts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The fused-requantization chains (``out_fmt == "mxfp8"``): ``out`` the e4m3 codes (uint8 ``[Tk_pad, N_out]``),
        ``out_sf`` the flat uint8 UE8M0 bytes in the 128x4 layout of ``(Tk_pad, N_out)``."""
        self._launch(a, b, sfa, sfb, out, counts, max_active_clusters, stream, out_sf)

    # host side (trace time)
    def _launch(
        self, a, b, sfa, sfb, out, counts, max_active_clusters, stream, out_sf=None
    ):
        pc = self.plan_cls
        assert a.element_type is pc.A_DT and b.element_type is pc.B_DT, (
            a.element_type,
            b.element_type,
        )
        sf_rows = a.shape[0]  # the padded packing: A's rows
        if self.dense_a:
            # the dense packing: the scale rows (the padded positions plus the tail) from the buffer itself
            sf_rows = sfa.shape[0] // (
                cute.ceil_div(a.shape[1] // pc.SF_VEC, SF_COL_TILE) * SF_COL_TILE
            )
        sfa_t = sf_view(sfa, sf_rows, a.shape[1], 1, pc.SF_VEC, pc.SF_DT)
        sfb_t = sf_view_experts(
            sfb, b.shape[0], b.shape[1], b.shape[2], pc.SF_VEC, pc.SF_DT
        )
        # the accumulator-side "C" of the plan is fp32 row-major: it only sizes the epilogue tile; the ring leaves room
        # for the group tables and the epilogue staging buffers
        self.staged = staged_store(self.chain.out_fmt)
        tables = (3 if self.dense_a else 2) * TABLE_BYTES
        st = plan_with_room(
            self.plan_cls,
            self.knobs,
            lambda st: (
                tables
                + (
                    staging_bytes(out.element_type, st.epi_tile, EPI_WARPGROUPS_MX)
                    if self.staged
                    else 0
                )
                + xch_bytes(self.chain, EPI_WARPGROUPS_MX)
            ),
            self.plan_cls.UMMA_SMEM_ADDR_MAX,
        )
        self.st = st
        self.c_smem_layout_staged = (
            staging_layout(out.element_type, st.epi_tile, EPI_WARPGROUPS_MX)
            if self.staged
            else st.a_smem_layout_staged
        )
        self.block = st.block
        assert self.block.in_fmt == self.in_fmt, (self.block.in_fmt, self.in_fmt)
        self.lowering = self.block.lowering()
        assert st.mma_tiler[2] == self.block.k_block, (st.mma_tiler, self.block.k_block)
        assert st.cluster_k == 1 and not self.knobs.swap_ab and not st.breuse, (
            "the grouped body takes the dense (unswapped, unsplit, no B-reuse) records"
        )
        tma = GroupedBlockScaledTma(st, a, b, sfa_t, sfb_t)
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
            out_sf,
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

    # device side
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
        mOutSf: Optional[cute.Tensor],
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

        # --- scaffolding: barriers, pipelines, TMEM allocator (as the block-scaled dense body), the group tables
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
        epi_join = pipeline.NamedBarrier(
            barrier_id=BAR_EPI_JOIN, num_threads=32 * len(EPILOGUE_WARPS)
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
            # the padded (atom-aligned) group offsets: the scale-factor positions of the dense rows
            s_off_pad = smem.allocate_tensor(
                Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
            )
        cluster_rows = st.cta_tile_shape_mnk[0] * self.cluster_shape_mn[0]
        # programmatic dependent launch: the prologue above (descriptor prefetch, barrier and pipeline setup, the TMEM
        # allocator) reads nothing of the predecessor's, so it overlaps the predecessor's tail; every thread waits here,
        # before the first dependent read (counts, then the operands and the scales), and lets the successor start its
        # own prologue at once (this grid is one persistent wave)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if warp_idx == EPILOGUE_WARPS[0]:
            if cutlass.const_expr(self.dense_a):
                # one pass for both tables: the dense group offsets (the A rows, the output) and the padded
                # scale-factor positions
                build_group_tables_dual(
                    counts, E, cluster_rows, s_off, s_pre, s_off_pad, GROUP_ALIGN
                )
            else:
                build_group_tables(
                    counts, E, cluster_rows, s_off, s_pre, GROUP_ALIGN
                )  # padded group offsets (atom-aligned)
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
        sX = None
        if cutlass.const_expr(xch_bytes(self.chain, EPI_WARPGROUPS_MX) > 0):
            # the fused requantization's half-block amax exchange
            sX = smem.allocate_tensor(
                element_type=Float32,
                layout=xch_layout(EPI_WARPGROUPS_MX),
                byte_alignment=16,
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
            mSFB_nkl,
            cute.slice_(st.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
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
        # the TMEM layouts of one stage of scale factors, built inside the kernel (a layout built at plan time belongs to
        # the host region)
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
            if cutlass.const_expr(self.chain.out_fmt == "mxfp8"):
                # the MXFP8 class's fused requantization: the codes and the 128x4 block scales of the bf16-rounded chain
                grouped_epilogue_quant_loop(
                    st,
                    self.chain,
                    tidx,
                    tile_sched,
                    tCtAcc_base,
                    thr_mma,
                    mOut,
                    mOutSf,
                    Tk,
                    N,
                    epi_tile,
                    acc_pipeline,
                    mma_m_in_cluster,
                    EPI_WARPGROUPS_MX,
                    sX,
                    s_off,
                    s_off_pad,
                    cta_v=mma_tile_coord_v,
                )
            else:
                grouped_epilogue_loop(
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
                    EPI_WARPGROUPS_MX,
                    sC=sC,
                    cta_v=mma_tile_coord_v,
                )
            tmem.relinquish_alloc_permit()
            # the allocator warp alone frees the columns: every epilogue warpgroup joins here after its last TMEM load (the
            # staged store's barriers join one group only)
            epi_join.arrive_and_wait()
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
    """The block-scaled producer with per-tile coordinates: A and SFA are re-partitioned from the group-offset view
    (``row0`` a multiple of 128: a whole scale-factor atom, so the swizzled SFA box is addressed exactly; the dense
    packing's SFA box at the group's padded position ``row0 + off_pad[e] - off[e]``), B and SFB are the dense partitions
    indexed by the expert as the batch coordinate. k-tiles ascending, one ring slot holds A, B, SFA and SFB of a k-tile.
    A 64-wide N tile reads half of a 128-row SFB atom (tile n -> atom n // 2); the 192-wide tile's addressing lives in
    the SFB coordinate tensor."""
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
            row0_sf = work_tile.row0 + (
                s_off_pad[work_tile.e] - s_off[work_tile.e]
            )  # dense A: the tile's padded (atom-aligned) position
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
