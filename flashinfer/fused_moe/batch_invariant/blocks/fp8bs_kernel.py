# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The grouped block-scale fp8 GEMM: the dense block-scale body (:mod:`flashinfer.gemm.batch_invariant.blocks.blockscale_fp8`,
class ``gemm.e4m3.kblock128``) under the MoE group scheduler. The class ``moe.e4m3bs.<act>`` is a placement of the dense class
per expert: every expanded row's output element is ``fold_kb fma(mul(s_a[row, kb], s_b[blk(n), kb]), P_kb, acc)`` -- the dense
fp8 ``mma32`` fold of each 128-element K block from +0, promoted in block order by the two promotion warpgroups -- whatever
expert group, tile, cluster or grid computes it.

Operands: A ``(Tk, K, 1)`` K-major e4m3 (the permuted activations), SFA ``(Tk, K / 128)`` fp32 with any strides (the API hands
the MN-major layout: a block's 128 row scales contiguous), B ``(N, K, E)`` K-major e4m3 (the stacked expert weights), SFB
``(E, N / 128, K / 128)`` fp32 (the per-expert 128x128 block scales), ``counts (E,)`` int32 (the dense packing: group ``e``
holds rows ``[off_e, off_e + counts[e])``), out ``(Tk, Nout)`` row-major. Three outputs, one promotion: the bf16 / f32 chain
output (``Nout = N / 2`` for the gated chain, ``N`` otherwise; the group-masked staged store of the bf16 class, the direct
store for the f32 oracle output), or the fused requantization (``out_fmt = "e4m3bs"``): the chain's bf16-rounded value,
widened, quantized per 128-column block by the ``fp8bs.quant`` class (:mod:`..spec_fp8bs`: the block amax tree, ``s =
fmaxc(div(amax, 448), 1e-12)``, ``inv = div(1, s)``, ``cvt.rn.satfinite.e4m3(mul(y, inv))``) with the e4m3 codes to ``out``
(uint8) and the fp32 scales to ``sfo (Nout / 128, Tk)`` (the MN layout GEMM2's ring reads). A block of 128 outputs lies inside
one tile (the fused chain's acceptance: the tile's outputs are whole blocks); its two 64-output halves land in the two
promotion warpgroups, which exchange their half-block amax through shared memory (one named barrier per tile; the canonical
tree over 128 leaves is the combine of the two halves' trees), or one warpgroup holds a whole block (the ungated chain on
the 256-wide tile).

The 32-wide class ``moe.e4m3bs32.<act>`` (``block = 32``, the dense core ``gemm.e4m3.kblock32``: :mod:`..spec_fp8bs`) runs on
the same body with the block as a parameter: a 128-element smem stage holds ``spb = 4`` blocks, the MMA warp commits every
instruction step to its own accumulator stage (:func:`grouped_blockscale_mma_loop`), the scale ring stage carries the stage's
four block scales per row and per weight slot (a tile spans ``tile_n / 32`` weight scale blocks: one scale product per 32
columns per thread and block), and the promotion drains the four partials in block order -- four times the 128-wide class's
TMEM reads and fmas per K, the recipe's cost. The fused requantization is the 128-wide class's alone (the 32-wide class
requantizes by the standalone pass).

The gated chain's weight is gate / up interleaved in groups of 8 rows (:func:`..api.interleave_fc1`) AFTER its quantization:
the scale of an interleaved column is its original row's 128-row block, so a tile holds one up block and one gate block
(``tile_n <= 256``, ``I % 128 == 0``), the scale ring stage carries both, and the promotion multiplies each register by the
block its column belongs to (a static choice per register: the fold is the spec's per element). The scale ring, the
accumulator ring, the per-block promotion and the ``fma.rn.f32x2`` packing are the dense body's; the group scheduler, the
group tables and the row predicates are the grouped kernels' (:mod:`.kernel`, :mod:`.epilogue`). Every placement here moves
no arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32, Uint8, Uint32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.tensor_utils import LayoutEnum
from cutlass.utils.blackwell_helpers import get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ....batch_invariant.canon import _pairwise
from ....batch_invariant.fp_interface.exact import f32_bits
from ....batch_invariant.quantization.blocks import pack_block_codes
from ....batch_invariant.quantization.cute_ext import CuteF32
from ....gemm.batch_invariant.blocks.blockscale_fp8 import (
    IN_FMT,
    MMA_WARP_BS,
    NUM_PROMO_GROUPS,
    NUM_PROMO_WARPS,
    NUM_SCALE_STAGES,
    PROMO_BATCH,
    PROMO_REGS,
    PROMO_WARPS,
    THREADS_BS,
    TMA_WARP_BS,
    UNIFORM_REGS,
    _elem,
    _vec4,
    acc_stages_for,
)
from ....gemm.batch_invariant.blocks.mainloop import mma_fold
from ....gemm.batch_invariant.blocks.operands import (
    ACC_DTYPE,
    BAR_TMEM_ALLOC,
    IN_DT,
    StaticPlan,
    TmaSetup,
)
from ....gemm.batch_invariant.generator import Knobs, MainloopBlock
from ....gemm.batch_invariant.spec import BLOCK_K
from .. import pdl
from ..spec import ACT_PARAMS, ACTS, UNGATED_ACTS, act_elem, glu_elem
from ..spec_fp8bs import BLOCK as QBLOCK
from ..spec_fp8bs import block_chain, check_block, elem_digit, leaf_value, quant_pre
from .epilogue import (
    BAR_EPI_STAGE,
    GROUP,
    MoeF32,
    fragment_offsets,
    staging_bytes,
    staging_layout,
    store_plan,
)
from .scheduler import MAX_E, TABLE_SMEM_BYTES, GroupScheduler, build_group_tables

#: The chain out_fmt of the fused requantization: e4m3 codes + fp32 block scales of the bf16-rounded chain.
OUT_FMT_QUANT = "e4m3bs"
OUT_FMTS = ("bf16", "f32", OUT_FMT_QUANT)
KINDS = ("plain", "glu", "act")
#: The grouped space's N tiles: two promotion warpgroups each own a contiguous column half (the narrow decode tiles have one subtile).
TILES_N_GROUPED = (64, 128, 256)
XCH_BUFFERS = 2  # the half-block amax exchange, double-buffered by tile parity (one barrier per tile)
BAR_XCH = 8  # the exchange's named barrier (both promotion warpgroups)
BAR_PROMO_JOIN = (
    BAR_XCH + 1
)  # both promotion warpgroups once, after their last TMEM loads, before the TMEM free
VEC_BYTES = 16
#: The scale ring's depth in smem stages per class: the 128-wide class's (one block per stage) and the 32-wide class's (four
#: blocks per stage: a stage holds four times the scales, so fewer stages hold as many blocks ahead).
NUM_SCALE_STAGES_OF = {128: NUM_SCALE_STAGES, 32: NUM_SCALE_STAGES // 2}


def blocks_per_stage(block: int) -> int:
    """Scale blocks per 128-element smem stage: 1 for the 128-wide class, 4 for the 32-wide one."""
    assert BLOCK_K % check_block(block) == 0, block
    return BLOCK_K // block


def n_scale_slots(cols: int, block: int, interleaved: bool) -> int:
    """The weight scale blocks a span of ``cols`` interleaved-or-plain weight columns reads: plain -- the span's ``block``-column
    blocks (at least one); interleaved -- its up rows' blocks and its gate rows' blocks (the span's ``cols / 2`` up rows: at least
    one block each half). The ring stage holds one slot per block (slot ``t``: :func:`slot_blocks`); the two halves of the
    interleaved weight occupy the slots' first and second half."""
    return 2 * max(1, (cols // 2) // block) if interleaved else max(1, cols // block)


def slot_blocks(n0, NB, NS: int, block: int, interleaved: bool):
    """The weight scale block (clamped to the tensor: a tile past N reads a finite scale and multiplies a zero-filled partial) of
    every slot of a tile whose first interleaved-or-plain column is ``n0``: plain -- ``n0 / block + t``; interleaved -- the up
    blocks of the tile's up rows ``(n0 / 2) / block + t`` for the first half of the slots, the gate blocks ``NB / 2 + (n0 / 2) /
    block + t`` for the second."""
    out = []
    for t in range(NS):
        if interleaved:
            half, tt = NS // 2, t % (NS // 2)
            base = (n0 // Int32(2)) // Int32(block) + Int32(tt)
            if t >= half:
                base = base + NB // Int32(2)
        else:
            base = n0 // Int32(block) + Int32(t)
        out.append(cutlass.min(base, NB - Int32(1)))
    return out


def group_slot_table(
    per_group: int, V: int, P: int, cols_g: int, NS: int, block: int, interleaved: bool
):
    """The weight scale slot of every register of a promotion group (static per register; the group's slot base is a runtime
    offset that enters addresses only): ``(n_slots_g, g_mult, slot_of[k][e])`` -- the group's slots per ring stage, the group
    index's multiplier and the local slot of register ``e`` of the group's subtile ``k`` (``P`` accumulator columns per subtile).
    Plain: the register's ``block``-column block of the group's ``cols_g`` columns (one slot where they lie in one block); gated:
    the block of the register's original up row (the up columns 16q .. 16q + 7 of the tile) in the slots' first half, its gate
    column's block in the second (:func:`n_scale_slots`, :func:`slot_blocks`)."""
    slot_of = []
    if interleaved:
        half_cols = cols_g // 2  # the group's up (= gate) rows
        n_slots_g = 2 * max(1, half_cols // block)
        g_mult = (n_slots_g // 2) if half_cols >= block else 0
        for k in range(per_group):
            row = []
            for e in range(V):
                u = (
                    k * (P // 2) + (e // (2 * GROUP)) * GROUP + e % GROUP
                )  # the register's original row within the group's half
                t = (u // block) if half_cols >= block else 0
                if (e % (2 * GROUP)) >= GROUP:
                    t += n_slots_g // 2  # a gate column: the gate half's slots
                row.append(t)
            slot_of.append(row)
        assert g_mult * (NUM_PROMO_GROUPS - 1) + n_slots_g // 2 <= NS // 2, (
            g_mult,
            n_slots_g,
            NS,
        )
    else:
        n_slots_g = max(1, cols_g // block)
        g_mult = n_slots_g if cols_g >= block else 0
        for k in range(per_group):
            slot_of.append(
                [((k * P + e) // block) if cols_g >= block else 0 for e in range(V)]
            )
        assert g_mult * (NUM_PROMO_GROUPS - 1) + n_slots_g <= NS, (
            g_mult,
            n_slots_g,
            NS,
        )
    assert all(
        slot_of[k][e] == slot_of[k][e + 1]
        for k in range(per_group)
        for e in range(0, V, 2)
    ), "a packed pair straddles two weight blocks"
    return n_slots_g, g_mult, slot_of


def group_slot_indices(gsel, g_mult: int, n_slots_g: int, NS: int, interleaved: bool):
    """The ring-stage slot of every local slot of promotion group ``gsel`` (a runtime index): the group's base ``gsel * g_mult``
    into the up (or plain) slots, then into the gate slots at ``NS / 2``."""
    out = []
    for t in range(n_slots_g):
        if interleaved and t >= n_slots_g // 2:
            out.append(
                Int32(NS // 2) + gsel * Int32(g_mult) + Int32(t - n_slots_g // 2)
            )
        else:
            out.append(gsel * Int32(g_mult) + Int32(t))
    return out


def grouped_scale_ring_bytes(knobs: Knobs, block: int, interleaved: bool) -> int:
    """Smem of the scale ring: per stage the CTA tile's row scales and the tile's weight slot scales, ``blocks_per_stage`` of
    each, fp32."""
    spb = blocks_per_stage(block)
    return (
        NUM_SCALE_STAGES_OF[block]
        * (knobs.cta_tile_mn[0] + n_scale_slots(knobs.tile_n, block, interleaved))
        * spb
        * (ACC_DTYPE.width // 8)
    )


@dataclass(frozen=True)
class Fp8BsChain:
    """The epilogue chain of a grouped block-scale GEMM: ``kind`` (``plain``, the gated ``glu`` on an accumulator pair, or the
    ungated ``act`` on one accumulator), the activation, the output format (``bf16``: the chain's bf16 output; ``f32``: the
    promoted fold as it stands, the oracle output; ``e4m3bs``: the fused requantization of the bf16-rounded chain) and the
    activation's parameters (binary32 values, ``None`` for an absent clamp / scale: ``spec.ACT_PARAMS``). ``describe()`` names
    it, parameters included (their bit patterns), so every parameter set is its own compiled kernel."""

    kind: str = "plain"
    act: str = "silu"
    out_fmt: str = "bf16"
    params: Tuple[Optional[float], ...] = ()

    def __post_init__(self):
        assert self.kind in KINDS and self.out_fmt in OUT_FMTS, self
        assert self.act in (UNGATED_ACTS if self.kind == "act" else ACTS), self
        if self.kind == "plain" and self.out_fmt == OUT_FMT_QUANT:
            raise AssertionError("the plain chain has no requantization")
        if self.kind in ("glu", "act") and self.out_fmt == "f32":
            raise AssertionError("the activation chains round to bf16 once")
        expected = len(ACT_PARAMS.get(self.act, ())) if self.kind == "glu" else 0
        assert len(self.params) == expected, (self, expected)

    @property
    def pair(self) -> bool:
        return self.kind == "glu"

    def describe(self) -> str:
        suffix = "".join(
            "_pn" if v is None else f"_p{f32_bits(v):08x}" for v in self.params
        )
        return f"{self.kind}_{self.act}_{self.out_fmt}{suffix}"


GEMM2_CHAIN = Fp8BsChain(
    "plain", "silu", "bf16"
)  # GEMM2: the plain bf16 conversion (the block scales are in the fold)
F32_CHAIN = Fp8BsChain(
    "plain", "silu", "f32"
)  # the oracle output: the promoted fp32 fold as it stands


def xch_floats() -> int:
    return XCH_BUFFERS * NUM_PROMO_GROUPS * 128


def quant_twin(chain: Fp8BsChain) -> Fp8BsChain:
    """The bf16 chain whose rounded output the fused requantization quantizes."""
    assert chain.out_fmt == OUT_FMT_QUANT, chain
    return Fp8BsChain(chain.kind, chain.act, "bf16", chain.params)


def fused_quant_ok(chain: Fp8BsChain, tile_n: int) -> bool:
    """Whether the fused requantization fits the record: the tile's outputs (half its accumulator columns for the gated chain)
    are whole 128-column blocks, so every block's amax is formed inside one tile (by one warpgroup or the pair's exchange)."""
    outs = tile_n // 2 if chain.pair else tile_n
    return chain.out_fmt == OUT_FMT_QUANT and outs % QBLOCK == 0 and outs >= QBLOCK


def _chain_elem_bs(chain: Fp8BsChain, acc, acc_gate):
    """One output element of a bf16 / f32 chain: the gated term of the (gate, up) accumulator pair, the ungated activation of
    one accumulator, or the plain conversion; every op in fp32 through ``MoeF32`` (the bf16 classes' chains)."""
    fp = MoeF32
    if chain.pair:
        return glu_elem(fp, acc_gate, acc, chain.act, chain.out_fmt, chain.params)
    if chain.kind == "act":
        return act_elem(fp, acc, chain.act, chain.out_fmt, chain.params)
    return fp.cvt_out(acc, chain.out_fmt)


class GroupedBlockScaleFp8Gemm:
    """``out[p, n] = chain( fold_kb fma(mul(sfa[p, kb], sfb[e(p), blk(n), kb]), P_kb(p, n), acc) )`` for every row ``p`` of every group."""

    def __init__(
        self,
        knobs: Knobs,
        chain: Fp8BsChain,
        interleaved: Optional[bool] = None,
        block: int = QBLOCK,
    ):
        """``interleaved``: the weight's rows are gate / up interleaved (the gated chains' weight; default ``chain.pair``) -- a
        property of the operand's layout that decides which weight scale block a column reads (the f32 oracle output of GEMM1
        reads the interleaved weight with the plain chain). ``block``: the class -- 128 (``moe.e4m3bs``) or 32 (``moe.e4m3bs32``)."""
        assert isinstance(chain, Fp8BsChain), chain
        assert (
            knobs.well_formed(IN_FMT)
            and knobs.swizzle_size == 1
            and knobs.tile_n in TILES_N_GROUPED
        ), knobs
        self.block = check_block(block)
        self.spb = blocks_per_stage(block)
        if chain.out_fmt == OUT_FMT_QUANT:
            assert block == QBLOCK and fused_quant_ok(chain, knobs.tile_n), (
                chain,
                knobs.tile_n,
                block,
            )
        self.interleaved = chain.pair if interleaved is None else bool(interleaved)
        assert self.interleaved or not chain.pair, (
            "the gated chain pairs the interleaved weight's columns"
        )
        self.knobs, self.chain = knobs, chain
        self.cluster_shape_mn = knobs.cluster_shape_mn
        self.quant = chain.out_fmt == OUT_FMT_QUANT
        self.staged = chain.out_fmt == "bf16"
        self.twin = quant_twin(chain) if self.quant else chain
        self.n_slots = n_scale_slots(knobs.tile_n, block, self.interleaved)
        self.n_scale_stages = NUM_SCALE_STAGES_OF[block]

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        sfa: cute.Tensor,
        b: cute.Tensor,
        sfb: cute.Tensor,
        out: cute.Tensor,
        sfo: cute.Tensor,
        counts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        assert a.element_type is IN_DT[IN_FMT] and b.element_type is IN_DT[IN_FMT], (
            a.element_type,
            b.element_type,
        )
        assert (
            sfa.element_type is ACC_DTYPE
            and sfb.element_type is ACC_DTYPE
            and sfo.element_type is ACC_DTYPE
        )
        tables = TABLE_SMEM_BYTES + 4 * xch_floats()

        def plan(extra):
            return StaticPlan(
                self.knobs,
                a.element_type,
                b.element_type,
                ACC_DTYPE,
                LayoutEnum.from_tensor(a).mma_major_mode(),
                LayoutEnum.from_tensor(b).mma_major_mode(),
                LayoutEnum.ROW_MAJOR,
                c_in_smem=False,
                extra_smem_bytes=tables + extra,
                in_fmt=IN_FMT,
                acc_stages=acc_stages_for(self.knobs),
                reserve_smem_bytes=grouped_scale_ring_bytes(
                    self.knobs, self.block, self.interleaved
                ),
            )

        st = plan(0)
        if cutlass.const_expr(self.staged):
            st = plan(staging_bytes(out.element_type, st.epi_tile, NUM_PROMO_GROUPS))
        self.st = st
        self.c_smem_layout_staged = (
            staging_layout(out.element_type, st.epi_tile, NUM_PROMO_GROUPS)
            if self.staged
            else None
        )
        self.mainloop = MainloopBlock(IN_FMT)
        self.lowering = self.mainloop.lowering()
        assert (
            st.mma_tiler[2] == self.mainloop.k_block == BLOCK_K == self.spb * self.block
        ), (st.mma_tiler, self.mainloop.k_block)  # one stage = spb scale blocks
        tma = TmaSetup(st, a, b, None)
        self.num_tma_load_bytes = tma.tx_bytes
        cm, cn = self.cluster_shape_mn
        grid = (cm, cn, max_active_clusters)
        self.kernel(
            st.tiled_mma,
            tma.atom_a,
            tma.tensor_a,
            tma.atom_b,
            tma.tensor_b,
            sfa,
            sfb,
            out,
            sfo,
            counts,
            st.cluster_layout_vmnk,
            st.a_smem_layout_staged,
            st.b_smem_layout_staged,
            st.epi_tile,
            self.c_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[THREADS_BS, 1, 1],
            cluster=(cm, cn, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mSFA: cute.Tensor,
        mSFB: cute.Tensor,
        mOut: cute.Tensor,
        mSfo: cute.Tensor,
        counts: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        c_smem_layout_staged,
    ):
        st = self.st
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == TMA_WARP_BS:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
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
        tidx, _, _ = cute.arch.thread_idx()
        E = counts.shape[0]
        N = mB_nkl.shape[0]
        Tk = mOut.shape[0]
        tile_m = st.cta_tile_shape_mnk[0]

        # --- scaffolding: barriers, pipelines, TMEM allocator, the group tables, the exchange buffer
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_acc_stage * 2]
            scale_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.n_scale_stages * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            group_off: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
            group_pre: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
            xch: cute.struct.MemRange[cutlass.Float32, xch_floats()]

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
                pipeline.Agent.Thread, len(PROMO_WARPS) * (2 if use_2cta_instrs else 1)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        scale_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.scale_mbar_ptr.data_ptr(),
            num_stages=n_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * len(PROMO_WARPS)
            ),
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=BAR_TMEM_ALLOC, num_threads=32 * (len(PROMO_WARPS) + 1)
        )
        promo_join = pipeline.NamedBarrier(
            barrier_id=BAR_PROMO_JOIN, num_threads=32 * NUM_PROMO_WARPS
        )
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=PROMO_WARPS[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        s_off = storage.group_off.get_tensor(cute.make_layout((MAX_E + 1,)))
        s_pre = storage.group_pre.get_tensor(cute.make_layout((MAX_E + 1,)))
        sX = storage.xch.get_tensor(
            cute.make_layout((XCH_BUFFERS, NUM_PROMO_GROUPS, 128))
        )
        cluster_rows = tile_m * self.cluster_shape_mn[0]
        # programmatic dependent launch: the prologue above depends on no predecessor; the wait sits before the first dependent
        # read (counts: the group tables, then the operands and scales); no trigger by default (pdl.py)
        cute.arch.griddepcontrol_wait()
        pdl.trigger_early()
        if warp_idx == PROMO_WARPS[0]:
            build_group_tables(counts, E, cluster_rows, s_off, s_pre)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # --- operand staging: smem rings, the scale ring, multicast masks, partitions
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
        # the scale ring: a stage holds the CTA tile's row scales and the tile's weight slot scales of the stage's `spb` blocks (a
        # row's or a slot's `spb` scales contiguous: one 16-byte copy of a slot's four at the 32-wide class)
        sSFA = smem.allocate_tensor(
            element_type=ACC_DTYPE,
            layout=cute.make_layout(
                (tile_m, spb, n_stages), stride=(spb, 1, tile_m * spb)
            ),
            byte_alignment=16,
        )
        sSFB = smem.allocate_tensor(
            element_type=ACC_DTYPE,
            layout=cute.make_layout((NS, spb, n_stages), stride=(spb, 1, NS * spb)),
            byte_alignment=16,
        )
        sC = None
        if cutlass.const_expr(self.staged):
            sC = smem.allocate_tensor(
                element_type=mOut.element_type,
                layout=c_smem_layout_staged,
                byte_alignment=128,
            )
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(st.is_a_mcast or st.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(st.mma_tiler, (0, None, None)), (None, None, None)
        )  # (bN, bK, RestN, RestK, RestL)
        k_tile_cnt = cute.size(gB_nkl, mode=[3])  # = K / BLOCK_K: the block count
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgB = thr_mma.partition_B(gB_nkl)
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
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(st.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, st.num_acc_stage)
        )
        mma_m_in_cluster = block_in_cluster_coord_vmnk[1]
        cta_n_in_cluster = block_in_cluster_coord_vmnk[2]
        # this CTA's first row inside the cluster tile: the M coordinate's block of the MMA tile's rows, then the pair member's half
        cta_row_rel = mma_m_in_cluster * Int32(
            st.mma_tiler[0]
        ) + mma_tile_coord_v * Int32(tile_m)
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        cute.arch.barrier()  # the group tables are complete

        raster_m = self.knobs.raster_along == "m"
        tn, cn = st.mma_tiler[1], self.cluster_shape_mn[1]
        # --- roles
        if warp_idx == TMA_WARP_BS:
            cute.arch.setmaxregister_decrease(UNIFORM_REGS)
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
            grouped_blockscale_producer_loop(
                st,
                k_tile_cnt,
                tile_sched,
                ab_producer,
                tma_atom_a,
                tma_atom_b,
                mA_mkl,
                tBgB,
                sA,
                tBsB,
                thr_mma,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                mma_m_in_cluster,
                a_full_mcast_mask,
                b_full_mcast_mask,
                scale_pipeline,
                mSFA,
                mSFB,
                sSFA,
                sSFB,
                cta_row_rel,
                Tk,
                self.interleaved,
                self.block,
            )
        if warp_idx == MMA_WARP_BS:
            cute.arch.setmaxregister_decrease(UNIFORM_REGS)
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
            grouped_blockscale_mma_loop(
                self.lowering,
                spb,
                st.num_acc_stage,
                k_tile_cnt,
                is_leader_cta,
                tile_sched,
                tiled_mma,
                ab_consumer,
                acc_pipeline,
                tCrA,
                tCrB,
                tCtAcc_base,
            )
        if warp_idx < NUM_PROMO_WARPS:
            cute.arch.setmaxregister_increase(PROMO_REGS)
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
            grouped_blockscale_epilogue_loop(
                st,
                self.chain,
                self.twin,
                self.interleaved,
                tidx,
                warp_idx,
                tile_sched,
                tCtAcc_base,
                thr_mma,
                mOut,
                mSfo,
                Tk,
                N,
                epi_tile,
                acc_pipeline,
                scale_pipeline,
                sSFA,
                sSFB,
                k_tile_cnt,
                mma_m_in_cluster,
                cta_row_rel,
                sC,
                sX,
                self.block,
            )
            tmem.relinquish_alloc_permit()
            # the allocator warp alone frees the columns: both promotion warpgroups join here after their last TMEM loads (the
            # store barriers join one group only)
            promo_join.arrive_and_wait()
            tmem.free(tmem_ptr)
        if warp_idx > TMA_WARP_BS:
            cute.arch.setmaxregister_decrease(
                UNIFORM_REGS
            )  # the warps padding the last warpgroup: no role


@cute.jit
def grouped_blockscale_mma_loop(
    lowering,
    spb: cutlass.Constexpr,
    num_acc_stage: cutlass.Constexpr,
    k_tile_cnt: Int32,
    is_leader_cta,
    tile_sched,
    tiled_mma: cute.TiledMma,
    ab_consumer,
    acc_pipeline,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCtAcc_base: cute.Tensor,
):
    """Per tile, per k-tile (stages ascending), per scale block of the stage (``spb`` of them, ascending): the block's steps from
    a reset accumulator into the next accumulator stage, committed to the epilogue -- the dense block-scale body's mainloop
    (``blockscale_mma_loop``: one block per stage, ``mma_fold`` once per k-tile) for the 128-wide class, and one instruction
    step per block for the 32-wide class (the stage released after its last block's steps are issued). Every block's partial is
    the spec's ``P_kb``."""
    assert cute.size(tCrA, mode=[2]) == lowering.inner and lowering.inner % spb == 0, (
        "one smem stage holds exactly the lowering's inner steps"
    )
    steps = lowering.inner // spb  # instruction steps per scale block
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, num_acc_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        if cutlass.const_expr(spb == 1):
            for _kb in range(
                k_tile_cnt
            ):  # blocks ascending, each a fold from +0 into the next stage
                ab_consumer, acc_state = mma_fold(
                    lowering,
                    1,
                    is_leader_cta,
                    tiled_mma,
                    ab_consumer,
                    acc_pipeline,
                    acc_state,
                    tCrA,
                    tCrB,
                    tCtAcc_base,
                )
        else:
            ab_consumer.reset()
            peek_full = cutlass.Boolean(1)
            if is_leader_cta:
                peek_full = ab_consumer.try_wait()
            for _kt in range(k_tile_cnt):  # stages ascending
                if is_leader_cta:
                    handle = ab_consumer.wait_and_advance(peek_full)
                    for jb in cutlass.range_constexpr(
                        spb
                    ):  # the stage's blocks ascending, each from +0 into its own stage
                        acc_pipeline.producer_acquire(
                            acc_state
                        )  # the epilogue has drained this TMEM buffer
                        tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
                        for j in cutlass.range_constexpr(steps):
                            tiled_mma.set(
                                tcgen05.Field.ACCUMULATE, j > 0
                            )  # off on the block's first step only
                            crd = (None, None, jb * steps + j, handle.index)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[crd], tCrB[crd], tCtAcc)
                        acc_pipeline.producer_commit(
                            acc_state
                        )  # the block's partial is complete: hand to the promotion
                        acc_state.advance()
                    handle.release()
                    peek_full = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_full = ab_consumer.try_wait()
                else:
                    acc_state.advance(spb)
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    acc_pipeline.producer_tail(acc_state)


@cute.jit
def grouped_blockscale_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    ab_producer,
    tma_atom_a: cute.CopyAtom,
    tma_atom_b: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tBgB: cute.Tensor,
    sA: cute.Tensor,
    tBsB: cute.Tensor,
    thr_mma,
    a_cta_coord,
    a_cta_layout,
    mma_m_in_cluster,
    a_mcast_mask,
    b_mcast_mask,
    scale_pipeline,
    mSFA: cute.Tensor,
    mSFB: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    cta_row_rel: Int32,
    Tk: Int32,
    interleaved: cutlass.Constexpr,
    block: cutlass.Constexpr,
):
    """The grouped TMA producer (A re-partitioned from the group-offset view, B indexed by the expert) with the dense body's
    scale ring: per stage (its ``spb`` blocks) the warp's 32 lanes ``cp.async`` this CTA's 128 row scales of each block
    (``mSFA[(row, kb)]``, rows clamped to the last expanded row: a row past the end multiplies a partial that is never stored)
    and lanes ``t < NS`` the tile's weight slot scales (:func:`slot_blocks`: the gated chain's up and gate blocks of the
    interleaved weight, or the plain tile's blocks; the 32-wide class's four blocks of a slot in one 16-byte copy -- the weight
    scales are K-contiguous and ``K / 32`` a multiple of four) into the next ring stage. Order and staging only."""
    lane = cute.arch.lane_idx()
    NB = mSFB.shape[1]
    tile_n = st.mma_tiler[1]
    rows_per_lane = cute.size(sSFA, mode=[0]) // 32
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
    while work_tile.is_valid_tile:
        real_a = cute.domain_offset((work_tile.row0, 0, 0), mA_mkl)
        gA = cute.local_tile(
            real_a, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )  # (bM, bK, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            a_cta_coord,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tAgA_slice = tAgA[(None, mma_m_in_cluster, None, 0)]
        tBgB_slice = tBgB[(None, work_tile.n_tile, None, work_tile.e)]
        e = work_tile.e
        row_base = work_tile.row0 + cta_row_rel
        n0 = work_tile.n_tile * Int32(tile_n)
        nbs = slot_blocks(n0, NB, NS, block, interleaved)
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
            scale_pipeline.producer_acquire(
                scale_state
            )  # ring stage free (the promotion released it)
            kb0 = handle.count * Int32(spb)  # the stage's first scale block
            for r in cutlass.range_constexpr(rows_per_lane):
                i = r * 32 + lane
                row = cutlass.min(row_base + i, Tk - Int32(1))
                for j in cutlass.range_constexpr(spb):
                    cute.copy(
                        atom_g2s,
                        _elem(mSFA, (row, kb0 + Int32(j))),
                        _elem(sSFA, (i, j, scale_state.index)),
                    )
            for t in cutlass.range_constexpr(NS):
                if lane == t:
                    if cutlass.const_expr(spb == 4):
                        cute.copy(
                            atom_g2s4,
                            _vec4(mSFB, (e, nbs[t], kb0)),
                            _vec4(sSFB, (t, 0, scale_state.index)),
                        )
                    else:
                        cute.copy(
                            atom_g2s,
                            _elem(mSFB, (e, nbs[t], kb0)),
                            _elem(sSFB, (t, 0, scale_state.index)),
                        )
            scale_pipeline.producer_commit(
                scale_state
            )  # arrives when this lane's copies have landed
            scale_state.advance()
            peek_empty = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_empty = ab_producer.try_acquire()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    pdl.trigger_late()
    ab_producer.tail()
    scale_pipeline.producer_tail(scale_state)


def block_combine(a, b):
    """The ``elem`` digit's combine (``fmaxc``) of two sub-tree amaxes, through ``CuteF32``."""
    return elem_digit().combine(CuteF32, a, b)


def block_amax_tree(fp, ys):
    """The quantizer's block reduction on one thread's consecutive output values: ``fabs`` leaves, the ``elem`` digit's pairwise
    ``fmaxc`` tree (the canonical order of ``spec_fp8bs.block_reduction``); on a half block the subtree whose combine with the
    other half's is the block's tree."""
    d = elem_digit()
    return _pairwise([leaf_value(fp, y) for y in ys], lambda a, b: d.combine(fp, a, b))


@cute.jit
def grouped_blockscale_epilogue_loop(
    st,
    chain: cutlass.Constexpr,
    twin: cutlass.Constexpr,
    interleaved: cutlass.Constexpr,
    tidx: Int32,
    warp_idx: Int32,
    tile_sched,
    tCtAcc_base: cute.Tensor,
    thr_mma,
    mOut: cute.Tensor,
    mSfo: cute.Tensor,
    Tk: Int32,
    N: Int32,
    epi_tile: cute.Tile,
    acc_pipeline,
    scale_pipeline,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    k_tile_cnt: Int32,
    mma_m_in_cluster,
    cta_row_rel: Int32,
    sC: cute.Tensor = None,
    sX: cute.Tensor = None,
    block: cutlass.Constexpr = QBLOCK,
):
    """The promotion (the dense body's: two warpgroups, each a contiguous column half of the tile, the block partials folded
    with packed fmas into a register running sum, one accumulator stage released per block, one scale ring stage per smem
    stage) and the group-masked store of the chain's output: the staged row-coalesced store (bf16), the direct vector store
    (the f32 oracle output) or the fused requantization (e4m3 codes and fp32 block scales, module docstring). Every register
    reads the scale of its own weight block (a static slot per register: the gated chain's up or gate block, the 32-wide
    class's ``block``-column block of the group's columns)."""
    pair = chain.pair
    quant = chain.out_fmt == OUT_FMT_QUANT
    out_dt = mOut.element_type
    g = warp_idx // 4  # this warp's promotion group
    tidx_g = tidx % 128
    warp_row0 = Int32(32) * (
        (tidx_g // Int32(32)) % Int32(4)
    )  # the first tile row of this warp's TMEM lanes
    tiler_mn = cute.slice_(st.mma_tiler, (None, None, 0))
    tn = st.mma_tiler[1]
    cta_m = st.cta_tile_shape_mnk[0]
    N_out = N // Int32(2) if cutlass.const_expr(pair) else N
    # the accumulator's flat column space: per-register (row, column) offsets, the chain's pairing
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
    thr_t2r = tiled_copy_t2r.get_slice(tidx_g)
    tTR_tAcc_base = thr_t2r.partition_S(
        tAcc_epi
    )  # (T2R, T2R_M, T2R_N, SUB_M, SUB_N, STAGE)
    cF_epi = cute.flat_divide(tCcF, epi_tile)
    tTR_cF_all = thr_t2r.partition_D(cF_epi)  # the registers' coordinates
    frag = tTR_cF_all[(None, None, None, 0, 0, 0, 0, 0)]
    offs = fragment_offsets(frag.layout)
    V = len(offs)
    sub_m, sub_n = cute.size(tAcc_epi, mode=[2]), cute.size(tAcc_epi, mode=[3])
    assert sub_m == 1 and cta_m == 128, (
        "the epilogue subtiles run along N only: the 128-row CTA tile is one subtile row"
    )
    assert len({m for m, _ in offs}) == 1, (
        "the fragment spans more than one output row",
        offs,
    )
    P = cute.size(epi_tile[1])  # accumulator columns per subtile
    assert sorted(n for _, n in offs) == list(range(P)) and all(
        offs[e][1] == e for e in range(V)
    ), offs
    subtile_cnt = sub_n
    groups = NUM_PROMO_GROUPS if subtile_cnt >= NUM_PROMO_GROUPS else 1
    per_group = subtile_cnt // groups
    assert per_group * groups == subtile_cnt and groups == NUM_PROMO_GROUPS, (
        subtile_cnt,
        groups,
    )
    cols_g = per_group * P  # this group's columns
    assert cols_g % 2 == 0, cols_g
    assert per_group % PROMO_BATCH == 0 or per_group < PROMO_BATCH, (
        per_group,
        PROMO_BATCH,
    )
    batch = min(per_group, PROMO_BATCH)
    VW = per_group * V
    spb = cute.size(sSFA, mode=[1])  # scale blocks per smem stage (1 / 4)
    NS = cute.size(
        sSFB, mode=[0]
    )  # weight scale slots per ring stage (the tile's blocks)
    n_stages = cute.size(sSFA, mode=[2])
    # the weight scale slot of every register (static per register; the group's slot base is a runtime offset that enters
    # addresses only): plain -- the register's `block`-column block of the group's columns; gated -- the block of the register's
    # original up row in the slots' first half, its gate column's block in the second (`group_slot_table`)
    assert P % (2 * GROUP) == 0, P
    n_slots_g, g_mult, slot_of = group_slot_table(
        per_group, V, P, cols_g, NS, block, interleaved
    )
    bufs = [cute.make_rmem_tensor(frag.shape, ACC_DTYPE) for _ in range(batch)]
    tTR_rSum = cute.make_rmem_tensor((VW,), ACC_DTYPE)
    # the chain's output positions: the up positions of the pair (half the registers) or every register; the outputs of a
    # subtile in column order (`out_col`), the group's outputs in order over its subtiles
    _, partner, is_out = store_plan(offs, pair, 16)
    out_col = {
        v: ((n // (2 * GROUP)) * GROUP + n % GROUP if pair else n)
        for v, (m, n) in enumerate(offs)
        if is_out[v]
    }
    V_out = len(out_col)
    OUT_G = per_group * V_out  # this group's output columns
    # the stores
    tTR_rC = cute.make_rmem_tensor(frag.shape, out_dt)
    tTR_rC.fill(0)
    staged = chain.out_fmt == "bf16"
    direct = chain.out_fmt == "f32"
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, st.num_acc_stage
    )
    scale_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, n_stages
    )
    gsel = g
    slot_idx = group_slot_indices(
        gsel, g_mult, n_slots_g, NS, interleaved
    )  # the group's slots in a ring stage
    # -- the staged bf16 store (the bf16 class's: this group's subtiles through its staging buffer, whole lines per warp instruction)
    epi_m = epi_n = CH = RS = SM = SN = NP = 1
    S_ = 0
    if cutlass.const_expr(staged):
        epi_m, epi_n = cute.size(epi_tile[0]), cute.size(epi_tile[1])
        n_lay = epi_tile[1]
        nested_n = cute.rank(n_lay) == 2
        Pp = cute.size(n_lay.shape[0]) if nested_n else epi_n
        NP = cute.size(n_lay.shape[1]) if nested_n else 1
        S_ = int(n_lay.stride[1]) if nested_n else 0
        step = 2 * GROUP if pair else GROUP
        CH = Pp // step
        RS = 128 // CH
        assert (
            Pp % step == 0
            and 128 % CH == 0
            and epi_m % RS == 0
            and epi_m == cta_m
            and Pp * NP == epi_n
        ), (
            epi_tile,
            step,
            cta_m,
        )
        atom_r2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            out_dt,
            num_bits_per_copy=store_plan(offs, pair, 16 // (out_dt.width // 8))[0]
            * out_dt.width,
        )  # 16-byte vectors: the staging rows' skew
        tiled_copy_r2s = cute.make_tiled_copy_D(atom_r2g, tiled_copy_t2r)
        thr_r2s = tiled_copy_r2s.get_slice(tidx_g)
        tRS_sC = thr_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        sC_g = sC[(None, None, gsel)]
        atom_s2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), out_dt, num_bits_per_copy=GROUP * out_dt.width
        )
        tiled_s2g = cute.make_tiled_copy_tv(
            atom_s2g,
            cute.make_layout((RS, CH), stride=(CH, 1)),
            cute.make_layout((1, GROUP)),
        )
        thr_s2g = tiled_s2g.get_slice(tidx_g)
        selL = cute.make_layout((epi_m, (GROUP, CH)), stride=(1, (epi_m, step * epi_m)))
        tS_sC = [
            thr_s2g.partition_S(
                cute.composition(cute.local_tile(sC_g, (epi_m, Pp), (0, pc)), selL)
            )
            for pc in range(NP)
        ]
        cO = cute.make_identity_tensor((epi_m, GROUP * CH))
        tS_cO = thr_s2g.partition_D(cO)
        SM, SN = cute.size(tS_cO.shape[1]), cute.size(tS_cO.shape[2])
        ld = cute.assume(mOut.layout.stride[0], divby=8)
        mOutFlat = cute.make_tensor(
            mOut.iterator, cute.make_layout((Tk, N_out), stride=(ld, 1))
        )
    if cutlass.const_expr(direct):
        ve_d = store_plan(offs, pair, 4)[0]
        atom_r2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), out_dt, num_bits_per_copy=ve_d * out_dt.width
        )
        tiled_copy_r2g = cute.make_tiled_copy_D(atom_r2g, tiled_copy_t2r)
        thr_r2g = tiled_copy_r2g.get_slice(tidx_g)
        tRG_rC = tiled_copy_r2g.retile(tTR_rC)
        nvec = cute.size(tRG_rC.shape[0][1])
        n1, n2 = cute.size(tRG_rC.shape[1]), cute.size(tRG_rC.shape[2])
        ldd = cute.assume(mOut.layout.stride[0], divby=4)
        if cutlass.const_expr(pair):
            col_shape = (GROUP, 2, N // (2 * GROUP))
            mC = cute.make_tensor(
                mOut.iterator,
                cute.make_layout((Tk, col_shape, 1), stride=(ldd, (1, 0, GROUP), 0)),
            )
        else:
            col_shape = N
            mC = cute.make_tensor(
                mOut.iterator, cute.make_layout((Tk, N, 1), stride=(ldd, 1, 0))
            )
        idC = cute.make_identity_tensor((Tk, col_shape, 1))
        cC = cute.local_tile(idC, tiler_mn, (None, None, None))
        tCcC = transform_partitioned_tensor_layout(thr_mma.partition_C(cC))
        cC_epi = cute.flat_divide(tCcC, epi_tile)
        tRG_cC_all = thr_r2g.partition_D(cC_epi)
    # -- the fused requantization: the group's outputs, bf16-rounded and widened, the block amax (a half or a whole block)
    SB = 1
    blocks_per_tile = 1
    ys_r = fy_words = fy = None
    if cutlass.const_expr(quant):
        tile_outs = tn // 2 if pair else tn
        assert (
            tile_outs % QBLOCK == 0 and OUT_G % VEC_BYTES == 0 and QBLOCK % OUT_G == 0
        ), (tile_outs, OUT_G)
        SB = QBLOCK // OUT_G  # groups per output block (2: the halves exchange)
        blocks_per_tile = tile_outs // QBLOCK
        assert (
            SB in (1, 2)
            and blocks_per_tile * QBLOCK == tile_outs
            and (SB == 1 or blocks_per_tile == 1)
        )
        ys_r = cute.make_rmem_tensor(
            cute.make_layout((OUT_G,)), Float32
        )  # the group's bf16-rounded outputs, widened
        fy_words = cute.make_rmem_tensor(
            cute.make_layout((OUT_G // 4,)), Uint32
        )  # their e4m3 codes as words
        fy = cute.make_tensor(
            cute.recast_ptr(fy_words.iterator, dtype=Uint8), cute.make_layout((OUT_G,))
        )
    atom16 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), Uint8, num_bits_per_copy=8 * VEC_BYTES
    )
    parity = Int32(0)
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        row0, row_end, n_tile = work_tile.row0, work_tile.row_end, work_tile.n_tile
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        rows_left = row_end - row0
        tTR_cF = tTR_cF_all[(None, None, None, None, None, mma_m_in_cluster, n_tile, 0)]
        tTR_cF = cute.group_modes(tTR_cF, 3, cute.rank(tTR_cF))
        crd0 = tTR_cF[(None, None, None, 0)][
            0
        ]  # register 0's (row, column) of the tile
        row_rel = crd0[0] - Int32(
            offs[0][0]
        )  # this thread's row inside the cluster tile
        irow = (
            row_rel - cta_row_rel
        )  # its row inside the CTA tile: the ring slot of its A scale
        # a warp whose 32 rows all lie past the group's end skips its TMEM loads and fmas (warp-uniform; the stage releases and
        # the barriers are kept); the pair tile's rows are counted from the cluster tile's first row
        warp_active = rows_left > cta_row_rel + warp_row0
        lane_live = row_rel < rows_left  # this lane's row is inside the group
        for i in cutlass.range_constexpr(VW):
            tTR_rSum[i] = CuteF32.const(0.0)  # acc_0 = +0
        for _kt in range(
            k_tile_cnt
        ):  # stages ascending, their blocks ascending: the block fold digit
            scale_pipeline.consumer_wait(
                scale_state
            )  # the stage's blocks' scales are in the ring
            sidx = scale_state.index
            scs = []  # [block of the stage][group slot]: mul.rn(s_a[m, kb], s_b[blk, kb]), once per slot and block
            for jb in cutlass.range_constexpr(spb):
                sa = sSFA[(irow, jb, sidx)]
                scs.append(
                    [
                        CuteF32.mul(sa, sSFB[(slot_idx[t], jb, sidx)])
                        for t in range(n_slots_g)
                    ]
                )
            scale_pipeline.consumer_release(scale_state)
            scale_state.advance()
            for jb in cutlass.range_constexpr(spb):
                acc_pipeline.consumer_wait(acc_state)  # the block's fold is complete
                if warp_active:
                    tTR_tAcc = cute.group_modes(
                        tTR_tAcc_base[(None, None, None, None, None, acc_state.index)],
                        3,
                        5,
                    )
                    for b0 in cutlass.range_constexpr(
                        0, per_group, batch
                    ):  # the group's subtiles in batches: a batch's loads first
                        for j in cutlass.range_constexpr(batch):
                            cute.copy(
                                tiled_copy_t2r,
                                tTR_tAcc[(None, None, None, gsel * per_group + b0 + j)],
                                bufs[j],
                            )
                        for j in cutlass.range_constexpr(batch):
                            k = b0 + j
                            for e in cutlass.range_constexpr(0, V, 2):
                                # acc = fma(scale, P_kb, acc) for two elements per fma.rn.f32x2 (each lane the same correctly
                                # rounded fma) with the scale of the register's weight block (a static slot per register)
                                sc = scs[jb][slot_of[k][e]]
                                r0, r1 = CuteF32.fma2(
                                    (sc, sc),
                                    (bufs[j][e], bufs[j][e + 1]),
                                    (tTR_rSum[k * V + e], tTR_rSum[k * V + e + 1]),
                                )
                                tTR_rSum[k * V + e] = r0
                                tTR_rSum[k * V + e + 1] = r1
                cute.arch.fence_view_async_tmem_load()  # this warp's reads of the partial are complete
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(
                        acc_state
                    )  # one arrive per warp: the stage is free once all arrived
                acc_state.advance()
        # ===== the chain and the store of this group's half of the tile's fold =====
        if cutlass.const_expr(quant):
            # the fused requantization: the bf16-rounded chain values of the group's outputs in column order, then the block amax
            amax = Float32(0.0)
            if warp_active:
                for k in cutlass.range_constexpr(per_group):
                    for v in cutlass.range_constexpr(V):
                        if cutlass.const_expr(is_out[v]):
                            if cutlass.const_expr(pair):
                                yb = _chain_elem_bs(
                                    twin,
                                    tTR_rSum[k * V + v],
                                    tTR_rSum[k * V + partner[v]],
                                )
                            else:
                                yb = _chain_elem_bs(twin, tTR_rSum[k * V + v], None)
                            ys_r[k * V_out + out_col[v]] = yb.to(Float32)
                amax = block_amax_tree(CuteF32, [ys_r[j] for j in range(OUT_G)])
                if cutlass.const_expr(SB == 2):
                    sX[(parity, gsel, tidx_g)] = amax
            if cutlass.const_expr(SB == 2):
                cute.arch.barrier(
                    barrier_id=Int32(BAR_XCH), number_of_threads=128 * NUM_PROMO_GROUPS
                )  # both halves' amaxes are in
                if warp_active:
                    other = sX[(parity, Int32(1) - gsel, tidx_g)]
                    lo = amax  # the block's tree: combine(first half, second half)
                    hi = other
                    if gsel == 1:
                        lo = other
                        hi = amax
                    amax = block_combine(lo, hi)
                parity = Int32(1) - parity
            if warp_active:
                s_blk, inv = block_chain(CuteF32, amax)
                yq = [quant_pre(CuteF32, ys_r[j], inv) for j in range(OUT_G)]
                for w, word in enumerate(pack_block_codes(CuteF32, yq, "e4m3")):
                    fy_words[w] = word
                blk = n_tile * Int32(blocks_per_tile) + (
                    gsel if cutlass.const_expr(SB == 1) else Int32(0)
                )
                half = gsel if cutlass.const_expr(SB == 2) else Int32(0)
                # rows past the group's real end and blocks past the output's columns (a tile wider than N, or an N-cluster's
                # tiles beyond the last one: their B boxes were zero-filled) are never stored
                stored = (row_rel < rows_left) & (blk * Int32(QBLOCK) < N_out)
                row_g = row0 + row_rel
                if stored:
                    dst = mOut[(row_g, None)]
                    for h in cutlass.range_constexpr(OUT_G // VEC_BYTES):
                        cute.copy(
                            atom16,
                            cute.local_tile(fy, (VEC_BYTES,), (h,)),
                            cute.local_tile(
                                dst,
                                (VEC_BYTES,),
                                (
                                    blk * Int32(QBLOCK // VEC_BYTES)
                                    + half * Int32(OUT_G // VEC_BYTES)
                                    + h,
                                ),
                            ),
                        )
                if stored & (half == Int32(0)):
                    mSfo[(blk, row_g)] = s_blk
        else:
            # a dead lane's accumulators (a row past the group's end: a zero-filled TMA row) replaced by a benign constant, so
            # the chain below runs unconditionally with its exact divisions off the hardware's slow path; dead rows are never
            # stored. The fused requantization's path (whole tiles by its row rule) keeps its accumulators as they are: the
            # select there cost its prefill
            for i in cutlass.range_constexpr(VW):
                tTR_rSum[i] = Float32(
                    cutlass.select_(lane_live, tTR_rSum[i], CuteF32.const(1.0))
                )
            for k in cutlass.range_constexpr(per_group):
                s = gsel * per_group + k  # the group's subtile index within the tile
                if warp_active:
                    for v in cutlass.range_constexpr(V):
                        if cutlass.const_expr(is_out[v]):
                            if cutlass.const_expr(pair):
                                tTR_rC[v] = _chain_elem_bs(
                                    chain,
                                    tTR_rSum[k * V + v],
                                    tTR_rSum[k * V + partner[v]],
                                )
                            else:
                                tTR_rC[v] = _chain_elem_bs(
                                    chain, tTR_rSum[k * V + v], None
                                )
                    if cutlass.const_expr(staged):
                        cute.copy(
                            tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, gsel)]
                        )
                    if cutlass.const_expr(direct):
                        tRG_cC = tRG_cC_all[
                            (None, None, None, None, None, mma_m_in_cluster, n_tile, 0)
                        ]
                        tRG_cC = cute.group_modes(tRG_cC, 3, cute.rank(tRG_cC))
                        gC = cute.local_tile(
                            cute.domain_offset((row0, 0, 0), mC),
                            tiler_mn,
                            (None, None, None),
                        )
                        tCgC = transform_partitioned_tensor_layout(
                            thr_mma.partition_C(gC)
                        )
                        gC_epi = cute.flat_divide(tCgC, epi_tile)
                        tRG_gC = thr_r2g.partition_D(gC_epi)[
                            (None, None, None, None, None, mma_m_in_cluster, n_tile, 0)
                        ]
                        tRG_gC = cute.group_modes(tRG_gC, 3, cute.rank(tRG_gC))
                        cC_sub = tRG_cC[(None, None, None, s)]
                        gC_sub = tRG_gC[(None, None, None, s)]
                        for jv in cutlass.range_constexpr(nvec):
                            for mi in cutlass.range_constexpr(n1):
                                for ni in cutlass.range_constexpr(n2):
                                    if cutlass.const_expr(is_out[jv * ve_d]):
                                        crd = cC_sub[((0, jv), mi, ni)]
                                        if cute.elem_less(
                                            crd, (rows_left, col_shape, 1)
                                        ):
                                            cute.copy(
                                                atom_r2g,
                                                tRG_rC[((None, jv), mi, ni)],
                                                gC_sub[((None, jv), mi, ni)],
                                            )
                if cutlass.const_expr(staged):
                    # the buffer -> global by the whole warpgroup: CH consecutive lanes write the CH 16-byte chunks of one output row
                    cute.arch.barrier(
                        barrier_id=Int32(BAR_EPI_STAGE) + gsel, number_of_threads=128
                    )
                    for pc in cutlass.range_constexpr(NP):
                        acc_col0 = n_tile * Int32(tn) + s * Int32(Pp) + Int32(pc * S_)
                        out_col0 = (
                            acc_col0 // Int32(2)
                            if cutlass.const_expr(pair)
                            else acc_col0
                        )
                        out_col0 = cute.assume(out_col0, divby=GROUP)
                        gO = cute.local_tile(
                            cute.domain_offset(
                                (row0 + cta_row_rel, out_col0), mOutFlat
                            ),
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
                        barrier_id=Int32(BAR_EPI_STAGE) + gsel, number_of_threads=128
                    )  # the buffer is free again
