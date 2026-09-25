# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# Ported from LightLM's cutlass_dsl megakernel study (docs/megakernel-plan.md there) for the FlashInfer experimental
# track: the Qwen3.8-27B one-launch decoder on SM120.
"""Megakernel study, GDN mixer block (docs/megakernel-plan.md §7): ONE persistent
CuTe DSL kernel runs a Qwen3.8-27B linear-attention block at decode:

  phase A1  in_proj_qkvz tiles (fp8 W8A16 swap-AB, the D-phase mainloop): the norm is
            absorbed (input xw = x * w_norm from the previous epilogue, 1/rms per token
            from the [NT, M] statistic partials applied as a column scale), per-shard
            alpha (qkv | z), direct bf16 stores into mz[N1, M]. Each tile publishes
            its own flag.
  phase B   one item per (token m, key-head kh) — a DATAFLOW dependency, not a grid
            barrier: thread 0 spins (acquire) on the flags of the 4 + 4*rep in_proj
            tiles that item reads, then 128 threads (one channel / one state column
            each) run: the decode conv step for q, k and the rep value heads (this item
            is the unique owner of those channels, so the tap shift is race-free); the
            b|a projections as 128-thread dot products against the bf16 in_proj_ba rows
            (the tiny bf16 GEMM absorbed); beta/g gates with the HF rounding; l2-norms;
            the delta-rule update on the fp32 pool state (thread t owns column t of
            the 128x128 state); the z-gated RMSNorm epilogue -> attn_out[M, vd]; then
            one heads-done increment.
  phase C   out_proj split-K items (fp8 mainloop again): the producer pre-issues the
            weight stages, spins on heads-done == TB, fence.proxy.async.global, then
            issues the activation stages; last-arriver reduce + residual + xw_out +
            sq_out (the next norm's inputs). The final reducer resets every counter.

Work item w: [0, TA) in_proj tile w; [TA, TA+TB) core item (m = i // Hk, kh = i % Hk);
[TA+TB, total) out_proj item (n_tile = j // S, split = j % S). CTA b owns b, b+G, ...
so a CTA never waits before finishing its own producers' work. Counters int32:
[0, TA) tile flags, [TA] heads-done, [TA+1] C-done, [TA+2, TA+2+NT2) split arrivals.
Assumes Dk == Dv == 128 (the 27B); rep = Hv // Hk value heads per key head.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync

from .gemm_w8a16_sm120 import W8A16SwapSm120


class GdnMegaSm120(W8A16SwapSm120):
    def __init__(
        self,
        acc_dtype,
        tile_shape_mnk,
        hk,
        hv,
        k1,
        nsplit=2,
        eps=1e-6,
        occupancy=1,
        prefetch_across_barrier=True,
        ablate=(),
        chunk_rows=32,
        head_groups=False,
        nbuf=2,
        state_alias=True,
        state_bulk=True,
        bulk_parts=4,
        pdl=0,
    ):
        super().__init__(acc_dtype, tile_shape_mnk, 2, (4, 1, 1), occupancy, nsh=2)
        if tile_shape_mnk[1] not in (8, 16) or tile_shape_mnk[0] != 64:
            raise ValueError("gdn mega assumes the (64, 8|16, TILE_K) tile")
        if hv % hk:
            raise ValueError("Hv must be a multiple of Hk")
        self.hk, self.hv, self.rep = int(hk), int(hv), int(hv) // int(hk)
        self.dk = self.dv = 128
        # hidden size as a constexpr: the b|a dot products give each of the 128 core
        # threads a CONTIGUOUS K1/128 slice (16 B vector copies, fully unrolled)
        self.k1 = int(k1)
        if self.k1 % (128 * 8):
            raise ValueError(
                "hidden size must be a multiple of 1024 (128 threads x 8-element vectors)"
            )
        self.kchunk = self.k1 // 128
        self.nsplit = int(nsplit)
        self.eps = float(eps)
        self.prefetch_across_barrier = bool(prefetch_across_barrier)
        self.pdl = int(
            pdl
        )  # 0 off; 1 = griddepcontrol.wait at entry + launch attribute
        # timing attribution arms (WRONG results): "wait" skips the dataflow spins, "core"
        # skips all core math (items publish immediately), "state" skips the two state passes
        self.ablate = frozenset(ablate)
        # state staging chunk (rows of the 128x128 fp32 state per smem pass) and the GEMM
        # stage cap that fits next to it in SM120's 101,376 B: (32 rows, 3 stages) = 88 KB,
        # (64 rows, 2 stages) = 80 KB — fewer barriers per head, shallower GEMM pipeline
        self.chunk_rows = int(chunk_rows)
        self.nbuf = int(nbuf)
        if self.chunk_rows not in (16, 32) or self.nbuf not in (2, 3, 4):
            raise ValueError("chunk_rows must be 16 or 32, nbuf 2..4")
        # nbuf chunk buffers (cp.async, prefetch distance nbuf-1) next to the GEMM stages in
        # 101,376 B: buffers = nbuf x rows x 512 B; the GEMM keeps the stages that still fit
        buf_bytes = self.nbuf * self.chunk_rows * 512
        # state_alias: the head's WHOLE 128x128 fp32 state (64 KB) is staged in the GEMM stage
        # smem instead (the stages are idle during head items: the producer only feeds A/C
        # items, and out_proj TMA cannot start before every head is done) -> one cp.async
        # burst, one wait, no per-chunk barrier chain; the GEMM keeps all of its stages
        self.state_alias = bool(state_alias)
        # state_bulk (needs state_alias): the 64 KB state moves with ONE descriptor-free TMA bulk
        # copy each way (cp.async.bulk + mbarrier in, bulk_group out) instead of 4096 per-thread
        # cp.async / st.global instructions — per-SM LSU miss handling capped the cp.async paths
        # 0/False: none, 1/True: bulk load + bulk store, 2: bulk load only, 3: bulk store only (bisect aid)
        self.state_bulk = int(state_bulk) if self.state_alias else 0
        self.bulk_load = self.state_bulk in (1, 2)
        self.bulk_store = self.state_bulk in (1, 3)
        self.bulk_parts = int(bulk_parts)  # bulk load issued as this many equal pieces
        if self.state_alias:
            buf_bytes = 0
        self.stage_cap = max(1, (101376 - 2048 - buf_bytes) // (24 * 1024))
        if self.stage_cap < 2:
            raise ValueError("state buffers leave fewer than 2 GEMM stages")
        # head_groups: out_proj K-split s waits only for the heads in its K range (starts earlier,
        # overlaps the head items); False: wait for every head (measured: the overlap HURTS)
        self.head_groups = bool(head_groups)
        if self.nsplit < 1 or self.hv % self.nsplit:
            raise ValueError(
                "Hv must be divisible by the out_proj split count (per-split heads-done groups)"
            )

    def _setup_attributes(self):
        super()._setup_attributes()
        if self.ab_stage > self.stage_cap:
            self.ab_stage = self.stage_cap
            self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
                self.a_layout, self.a_tile_mnk, self.a_dtype, self.ab_stage
            )
            self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
                self.b_layout, self.tile_shape_mnk, self.b_dtype, self.ab_stage
            )
        if self.state_alias:
            stage_bytes = self.ab_stage * (
                cute.size_in_bytes(
                    self.a_dtype,
                    cute.slice_(self.a_smem_layout_staged, (None, None, 0)),
                )
                + cute.size_in_bytes(
                    self.b_dtype,
                    cute.slice_(self.b_smem_layout_staged, (None, None, 0)),
                )
            )
            if stage_bytes < 128 * 128 * 4:
                raise ValueError(
                    f"GEMM stage smem ({stage_bytes} B) cannot hold a 64 KB head state"
                )

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        a1: cute.Tensor,
        b1: cute.Tensor,
        c1: cute.Tensor,  # in_proj container (N1, K1/2, 1); xw_in (M, K1, 1); dummy (N1, M, 1)
        mz: cute.Tensor,  # (N1, M, 1) bf16 in_proj output
        alpha_a: cute.Tensor,
        rs: cute.Tensor,  # [2] (qkv | z) fp8 dequant scales; [NT, M] sum-of-squares partials
        convw: cute.Tensor,
        conv_pool: cute.Tensor,
        ssm_pool: cute.Tensor,
        slots: cute.Tensor,  # (C, 4) bf16; (S, C, 3) bf16; (S, Hv, 128, 128) f32; [M] i32
        wba: cute.Tensor,
        alog: cute.Tensor,
        dtb: cute.Tensor,
        normw: cute.Tensor,  # (2Hv, K1) bf16; [Hv] f32; [Hv] f32; [128] bf16
        attn: cute.Tensor,  # (vd, M, 1) bf16 gated output = phase C's activation
        qk: cute.Tensor,  # fp32 [M*Hk*256] l2-normalized q|k per (token, key head) from phase B0
        a2: cute.Tensor,
        b2: cute.Tensor,
        ws: cute.Tensor,  # out_proj container (N2, K2/2/S, S); attn view (M, K2/S, S); ws (N2, M, S)
        resid: cute.Tensor,
        alpha_b: cute.Tensor,
        wn: cute.Tensor,
        xw_out: cute.Tensor,
        sq: cute.Tensor,
        cnt: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype, self.b_dtype, self.c_dtype = (
            a1.element_type,
            b1.element_type,
            c1.element_type,
        )
        self.a_layout = utils.LayoutEnum.from_tensor(a1)
        self.b_layout = utils.LayoutEnum.from_tensor(b1)
        self.c_layout = utils.LayoutEnum.from_tensor(c1)
        self._setup_attributes()
        tma_a1, mA1 = self._make_tma_atoms_and_tensors(
            a1, self.a_smem_layout_staged, (self.a_tile_mnk[0], self.a_tile_mnk[2]), 1
        )
        tma_b1, mB1 = self._make_tma_atoms_and_tensors(
            b1,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_a2, mA2 = self._make_tma_atoms_and_tensors(
            a2, self.a_smem_layout_staged, (self.a_tile_mnk[0], self.a_tile_mnk[2]), 1
        )
        tma_b2, mB2 = self._make_tma_atoms_and_tensors(
            b2,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        ta = c1.shape[0] // self.tile_shape_mnk[0]
        m_tok = b1.shape[0]
        tb0 = m_tok * self.hk
        tb = m_tok * self.hv
        nt2 = ws.shape[0] // self.tile_shape_mnk[0]
        tc = nt2 * self.nsplit
        k_cnt_a = b1.shape[1] // self.tile_shape_mnk[2]
        k_cnt_c = b2.shape[1] // self.tile_shape_mnk[2]
        kd = self.hk * self.dk
        vd = self.hv * self.dv
        grid = (
            cutlass.min(cutlass.Int32(max_active_clusters), ta + tb0 + tb + tc),
            1,
            1,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            epi_flag: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 8], 16]
            sq_part: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 64], 16]
            core_q: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 128], 16]
            core_k: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 128], 16]
            core_red: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 8], 16]
            state_bar: cute.struct.MemRange[
                cutlass.Int64, 1
            ]  # bulk-copy completion mbarrier (state_bulk)
            core_state: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    32 if self.state_alias else self.nbuf * self.chunk_rows * 128,
                ],
                128,
            ]  # nbuf row chunks (chunked mode)

        self.shared_storage = SharedStorage
        self.kernel(
            tma_a1,
            mA1,
            tma_b1,
            mB1,
            tma_a2,
            mA2,
            tma_b2,
            mB2,
            b1,
            mz,
            alpha_a,
            rs,
            convw,
            conv_pool,
            ssm_pool,
            slots,
            wba,
            alog,
            dtb,
            normw,
            attn,
            qk,
            ws,
            resid,
            alpha_b,
            wn,
            xw_out,
            sq,
            cnt,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            ta,
            tb0,
            tb,
            tc,
            nt2,
            k_cnt_a,
            k_cnt_c,
            kd,
            vd,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            use_pdl=bool(self.pdl),
        )

    # ------------------------------------------------------------------ device helpers
    @cute.jit
    def _block_sum(self, val, sRed, warp_idx, lane):
        """Sum of `val` over the 128 MMA threads, fixed order (deterministic)."""
        v = val
        for off in [16, 8, 4, 2, 1]:
            v = v + cute.arch.shuffle_sync_bfly(v, offset=off)
        if lane == 0:
            sRed[warp_idx] = v
        self.epilog_sync_barrier.arrive_and_wait()
        tot = sRed[0] + sRed[1] + sRed[2] + sRed[3]
        self.epilog_sync_barrier.arrive_and_wait()  # sRed reusable
        return tot

    @cute.jit
    def _wait_flag(self, mCnt, idx):
        """Spin (acquire) until counter idx is set; called by ONE thread per flag."""
        seen = cute.arch.atomic_add(
            mCnt.iterator + idx, cutlass.Int32(0), sem="acquire", scope="gpu"
        )
        while seen < 1:
            seen = cute.arch.atomic_add(
                mCnt.iterator + idx, cutlass.Int32(0), sem="acquire", scope="gpu"
            )

    @cute.jit
    def _conv_channel(self, mMz, mConvW, mConv, slot, c, m):
        """Decode conv step for channel c of token m (taps [s0,s1,s2] + new x; W=4):
        returns silu(sum) and shifts the pool taps (this item owns the channel)."""
        x_raw = mMz[(c, m, 0)]
        s0 = mConv[(slot, c, 0)]
        s1 = mConv[(slot, c, 1)]
        s2 = mConv[(slot, c, 2)]
        acc = (
            x_raw.to(cutlass.Float32) * mConvW[(c, 3)].to(cutlass.Float32)
            + s0.to(cutlass.Float32) * mConvW[(c, 0)].to(cutlass.Float32)
            + s1.to(cutlass.Float32) * mConvW[(c, 1)].to(cutlass.Float32)
            + s2.to(cutlass.Float32) * mConvW[(c, 2)].to(cutlass.Float32)
        )
        mConv[(slot, c, 0)] = s1
        mConv[(slot, c, 1)] = s2
        mConv[(slot, c, 2)] = x_raw
        return acc / (cutlass.Float32(1.0) + cute.math.exp(-acc))

    @cute.jit
    def _emit_sq(
        self, rfin, tCcC, sSq, mSq, n_tile, n_cols, tidx, lane, warp_idx, mma_m
    ):
        mma_nn = cute.size(rfin) // (4 * mma_m)
        for cc in cutlass.range_constexpr(2):
            for n in cutlass.range_constexpr(2):
                ss = cutlass.Float32(0.0)
                if n < mma_nn:
                    for rh in cutlass.range_constexpr(2):
                        for m in cutlass.range_constexpr(mma_m):
                            idx = cc + 2 * rh + 4 * (m + mma_m * n)
                            ss = ss + rfin[idx] * rfin[idx]
                    for off in [4, 8, 16]:
                        ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=off)
                if lane < 4:
                    sSq[(warp_idx, lane, cc, n)] = ss
        self.epilog_sync_barrier.arrive_and_wait()
        if tidx < 16:
            tok = tidx
            if tok < n_cols:
                tot = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(4):
                    tot = tot + sSq[(w, (tok % 8) // 2, tok % 2, tok // 8)]
                mSq[(n_tile, tok)] = tot
        self.epilog_sync_barrier.arrive_and_wait()

    # ------------------------------------------------------------------ device
    @cute.kernel
    def kernel(
        self,
        tma_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_b1: cute.CopyAtom,
        mB1: cute.Tensor,
        tma_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        tma_b2: cute.CopyAtom,
        mB2: cute.Tensor,
        mXwIn: cute.Tensor,  # real-pointer xw_in (M, K1, 1): the b|a dots read it (TMA tensors are coordinate-only)
        mMz: cute.Tensor,
        mAlphaA: cute.Tensor,
        mRs: cute.Tensor,
        mConvW: cute.Tensor,
        mConv: cute.Tensor,
        mSsm: cute.Tensor,
        mSlots: cute.Tensor,
        mWba: cute.Tensor,
        mAlog: cute.Tensor,
        mDtb: cute.Tensor,
        mNormW: cute.Tensor,
        mAttn: cute.Tensor,
        mQK: cute.Tensor,
        mWs: cute.Tensor,
        mRes: cute.Tensor,
        mAlphaB: cute.Tensor,
        mWn: cute.Tensor,
        mXw: cute.Tensor,
        mSq: cute.Tensor,
        mCnt: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        TA: cutlass.Int32,
        TB0: cutlass.Int32,
        TB: cutlass.Int32,
        TC: cutlass.Int32,
        NT2: cutlass.Int32,
        k_cnt_a: cutlass.Int32,
        k_cnt_c: cutlass.Int32,
        KD: cutlass.Int32,
        VD: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tidx % 32
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_a1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_b1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_a2)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_b2)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        state_bar = storage.state_bar.data_ptr()
        if cutlass.const_expr(self.state_bulk):
            if tidx == 0:
                cute.arch.mbarrier_init(state_bar, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mma_warps
            ),
            tx_count=tma_copy_bytes,
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sFlag = storage.epi_flag.get_tensor(cute.make_layout((8,)))
        sSq = storage.sq_part.get_tensor(
            cute.make_layout((4, 4, 2, 2), stride=(16, 4, 2, 1))
        )
        sQ = storage.core_q.get_tensor(cute.make_layout((128,)))
        sK = storage.core_k.get_tensor(cute.make_layout((128,)))
        sRed = storage.core_red.get_tensor(cute.make_layout((8,)))
        sSt = storage.core_state.get_tensor(
            cute.make_layout(
                (self.nbuf, self.chunk_rows, 128),
                stride=(self.chunk_rows * 128, 128, 1),
            )
        )  # (buf, row, col)
        # state_alias: the whole head state over the (contiguous) sA|sB stage buffers
        sStA = cute.make_tensor(
            cute.recast_ptr(storage.sA.data_ptr(), dtype=cutlass.Float32),
            cute.make_layout((128, 128), stride=(128, 1)),
        )
        bulk_g2s = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), cutlass.Float32)
        bulk_s2g = cute.make_copy_atom(cpasync.CopyBulkS2GOp(), cutlass.Float32)

        gA1 = cute.local_tile(
            mA1, cute.slice_(self.a_tile_mnk, (None, 0, None)), (None, None, None)
        )
        gB1 = cute.local_tile(
            mB1, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None)
        )
        gA2 = cute.local_tile(
            mA2, cute.slice_(self.a_tile_mnk, (None, 0, None)), (None, None, None)
        )
        gB2 = cute.local_tile(
            mB2, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(tidx)
        one = cute.make_layout(1)
        tAsA, tAgA1 = cute.nvgpu.cpasync.tma_partition(
            tma_a1, 0, one, cute.group_modes(sA, 0, 2), cute.group_modes(gA1, 0, 2)
        )
        tBsB, tBgB1 = cute.nvgpu.cpasync.tma_partition(
            tma_b1, 0, one, cute.group_modes(sB, 0, 2), cute.group_modes(gB1, 0, 2)
        )
        _, tAgA2 = cute.nvgpu.cpasync.tma_partition(
            tma_a2, 0, one, cute.group_modes(sA, 0, 2), cute.group_modes(gA2, 0, 2)
        )
        _, tBgB2 = cute.nvgpu.cpasync.tma_partition(
            tma_b2, 0, one, cute.group_modes(sB, 0, 2), cute.group_modes(gB2, 0, 2)
        )

        tCsA = thr_mma.partition_A(sA)
        perm_k = cute.make_layout(
            ((2, 4, 2), self.tile_shape_mnk[2] // 16), stride=((1, 4, 2), 16)
        )
        sB_perm = cute.composition(
            sB,
            (
                cute.make_layout(self.tile_shape_mnk[1]),
                perm_k,
                cute.make_layout(self.ab_stage),
            ),
        )
        tCsB = thr_mma.partition_B(sB_perm)
        tCrA_c = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        mma_m = cute.size(tCrA_c, mode=[1])
        tCrA_log = cute.make_rmem_tensor(
            cute.make_layout(((2, 2, 2), mma_m, 2)), self.b_dtype
        )
        tCcC = thr_mma.partition_C(
            cute.make_identity_tensor((self.tile_shape_mnk[0], self.tile_shape_mnk[1]))
        )
        accumulators = cute.make_rmem_tensor(tCcC.shape[:3], self.acc_dtype)

        pipeline.sync(barrier_id=1)
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        total = TA + TB0 + TB + TC
        W = 2 * KD + VD  # conv channels = qkv width; z follows at column W
        HD = TA  # heads-done per out_proj K-split group: [TA, TA + S)
        CDONE = TA + self.nsplit
        SPLIT0 = TA + self.nsplit + 1  # split-K arrivals per out_proj tile
        B0_FLAGS = SPLIT0 + NT2  # per-(token, key head) q|k-ready flags
        HEADS_PER_GROUP = self.hv // self.nsplit

        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            if cutlass.const_expr(self.pdl):
                # PDL (kernels/pdl.py mode 1 for the megakernels): everything this kernel reads
                # except weights was produced by the previous launch of the step; wait once at
                # entry (after the smem/pipeline/descriptor prologue, which is what overlaps),
                # then let the next kernel schedule early
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            num_kc_blocks = cute.size(tCrA_c, mode=[2])
            atom_ldsm_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_ldsm_A, tiled_mma)
            thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
            tCsA_copy_view = thr_copy_A.partition_S(sA)
            tCrA_copy_view = thr_copy_A.retile(tCrA_c)
            tCrA_log32 = cute.recast_tensor(tCrA_log, cutlass.Int32)
            conv16 = cute.make_rmem_tensor((16,), self.b_dtype)
            conv32 = cute.recast_tensor(conv16, cutlass.Int32)
            rres = cute.make_rmem_tensor((cute.size(accumulators),), cutlass.Float32)
            rfin = cute.make_rmem_tensor((cute.size(accumulators),), cutlass.Float32)
            rinv_c = cute.make_rmem_tensor(
                (2, cute.size(accumulators, mode=[2])), cutlass.Float32
            )
            scol = cute.make_rmem_tensor(
                (128,), cutlass.Float32
            )  # this thread's state column
            n_tok = mMz.shape[1]
            n_rows2 = mRes.shape[0]
            k_total = cutlass.Float32(mB1.shape[1])
            nt_rs = mRs.shape[0]
            alpha_c = mAlphaB[0]
            qscale = cutlass.Float32(0.08838834764831845)  # 128^-0.5

            work = cutlass.Int32(bidx)
            st_phase = cutlass.Int32(0)  # state mbarrier parity (state_bulk)
            while work < total:
                is_a = work < TA
                is_b0 = (work >= TA) and (work < TA + TB0)
                is_b = (work >= TA + TB0) and (work < TA + TB0 + TB)
                is_core = is_b0 or is_b
                # ---- GEMM phases share one mainloop
                if not is_core:
                    n_tile = cutlass.Int32(work)
                    split = cutlass.Int32(0)
                    kcnt = cutlass.Int32(k_cnt_a)
                    if not is_a:
                        jt = work - TA - TB0 - TB
                        n_tile = jt // self.nsplit
                        split = jt % self.nsplit
                        kcnt = k_cnt_c
                    row0 = n_tile * self.tile_shape_mnk[0]
                    accumulators.fill(0.0)
                    if is_a:
                        for cc in cutlass.range_constexpr(2):
                            for n in cutlass.range_constexpr(
                                cute.size(accumulators, mode=[2])
                            ):
                                tok = tCcC[((cc, 0), 0, n)][1]
                                ssc = cutlass.Float32(0.0)
                                rinv_c[(cc, n)] = cutlass.Float32(0.0)
                                if tok < n_tok:
                                    for j in range(0, nt_rs, 1):
                                        ssc = ssc + mRs[(j, tok)]
                                    rinv_c[(cc, n)] = cute.math.rsqrt(
                                        ssc / k_total + cutlass.Float32(self.eps)
                                    )
                    else:
                        for i in cutlass.range_constexpr(cute.size(accumulators)):
                            crd = tCcC[i]
                            r = row0 + crd[0]
                            col = crd[1]
                            rres[i] = cutlass.Float32(0.0)
                            if r < n_rows2 and col < n_tok:
                                rres[i] = mRes[(r, col, 0)].to(cutlass.Float32)
                    mainloop_consumer_state.reset_count()
                    for _k_tile in range(0, kcnt, 1, unroll=1):
                        mainloop_pipeline.consumer_wait(mainloop_consumer_state)
                        stage = mainloop_consumer_state.index
                        tCsA_p = tCsA_copy_view[None, None, None, stage]
                        tCsB_p = tCsB[None, None, None, stage]
                        for kc in cutlass.range_constexpr(num_kc_blocks):
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, kc],
                                tCrA_copy_view[None, None, kc],
                            )
                            cute.autovec_copy(
                                tCsB_p[None, None, 2 * kc], tCrB[None, None, 2 * kc]
                            )
                            cute.autovec_copy(
                                tCsB_p[None, None, 2 * kc + 1],
                                tCrB[None, None, 2 * kc + 1],
                            )
                            for m in cutlass.range_constexpr(mma_m):
                                v8 = tCrA_c[(None, m, kc)].load()
                                vf8 = v8.bitcast(cutlass.Float8E4M3FN)
                                conv16.store(vf8.to(self.b_dtype))
                                for khc in cutlass.range_constexpr(2):
                                    for rh in cutlass.range_constexpr(2):
                                        for e in cutlass.range_constexpr(2):
                                            tCrA_log32[((0, rh, e), m, khc)] = conv32[
                                                e + 2 * rh + 4 * khc
                                            ]
                            for half in cutlass.range_constexpr(2):
                                cute.gemm(
                                    tiled_mma,
                                    accumulators,
                                    tCrA_log[None, None, half],
                                    tCrB[None, None, 2 * kc + half],
                                    accumulators,
                                )
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()
                    if is_a:
                        # in_proj epilogue: shard alpha (qkv rows < W, z rows >= W) x 1/rms column scale
                        alpha_v = mAlphaA[0]
                        if row0 >= W:
                            alpha_v = mAlphaA[1]
                        for cc in cutlass.range_constexpr(2):
                            for rh in cutlass.range_constexpr(2):
                                for m in cutlass.range_constexpr(mma_m):
                                    for n in cutlass.range_constexpr(
                                        cute.size(accumulators, mode=[2])
                                    ):
                                        crd = tCcC[((cc, rh), m, n)]
                                        r = row0 + crd[0]
                                        col = crd[1]
                                        if col < n_tok:
                                            v = (
                                                accumulators[((cc, rh), m, n)]
                                                * alpha_v
                                                * rinv_c[(cc, n)]
                                            )
                                            mMz[(r, col, 0)] = v.to(mMz.element_type)
                        cute.arch.fence_acq_rel_gpu()
                        self.epilog_sync_barrier.arrive_and_wait()
                        if tidx == 0:
                            cute.arch.atomic_add(
                                mCnt.iterator + n_tile,
                                cutlass.Int32(1),
                                sem="acq_rel",
                                scope="gpu",
                            )
                    else:
                        # out_proj: split-K partial, last-arriver reduce, residual, xw_out, sq_out
                        for i in cutlass.range_constexpr(cute.size(accumulators)):
                            crd = tCcC[i]
                            r = row0 + crd[0]
                            col = crd[1]
                            if r < n_rows2 and col < n_tok:
                                mWs[(r, col, split)] = accumulators[i] * alpha_c
                        cute.arch.fence_acq_rel_gpu()
                        self.epilog_sync_barrier.arrive_and_wait()
                        if tidx == 0:
                            old = cute.arch.atomic_add(
                                mCnt.iterator + (SPLIT0 + n_tile),
                                cutlass.Int32(1),
                                sem="acq_rel",
                                scope="gpu",
                            )
                            sFlag[0] = old
                        self.epilog_sync_barrier.arrive_and_wait()
                        if sFlag[0] == self.nsplit - 1:
                            cute.arch.fence_acq_rel_gpu()
                            for i in cutlass.range_constexpr(cute.size(accumulators)):
                                crd = tCcC[i]
                                r = row0 + crd[0]
                                col = crd[1]
                                rfin[i] = cutlass.Float32(0.0)
                                if r < n_rows2 and col < n_tok:
                                    acc_total = rres[i]
                                    for sp in cutlass.range_constexpr(self.nsplit):
                                        acc_total = acc_total + mWs[(r, col, sp)]
                                    mRes[(r, col, 0)] = acc_total.to(mRes.element_type)
                                    mXw[(r, col, 0)] = (
                                        acc_total * mWn[(r,)].to(cutlass.Float32)
                                    ).to(mXw.element_type)
                                    rfin[i] = acc_total
                            self._emit_sq(
                                rfin,
                                tCcC,
                                sSq,
                                mSq,
                                n_tile,
                                n_tok,
                                tidx,
                                lane,
                                warp_idx,
                                mma_m,
                            )
                            self.epilog_sync_barrier.arrive_and_wait()
                            if tidx == 0:
                                mCnt[SPLIT0 + n_tile] = cutlass.Int32(0)
                                done = cute.arch.atomic_add(
                                    mCnt.iterator + CDONE,
                                    cutlass.Int32(1),
                                    sem="acq_rel",
                                    scope="gpu",
                                )
                                if (
                                    done == NT2 - 1
                                ):  # whole block done: re-arm every counter
                                    for i in cutlass.range_constexpr(self.nsplit):
                                        mCnt[HD + i] = cutlass.Int32(0)
                                    mCnt[CDONE] = cutlass.Int32(0)
                                    for i in range(0, TA, 1):
                                        mCnt[i] = cutlass.Int32(0)
                                    for i in range(0, TB0, 1):
                                        mCnt[B0_FLAGS + i] = cutlass.Int32(0)
                elif is_b0:
                    # ---- phase B0: (token m, key head kh): q|k conv + shift + l2-norm, once per key head
                    it = work - TA
                    m = it // self.hk
                    kh = it % self.hk
                    slot = mSlots[m]
                    q0 = kh * self.dk
                    kk0 = KD + kh * self.dk
                    if cutlass.const_expr(
                        "wait" not in self.ablate
                    ):  # q, k tiles: one polling thread per flag
                        if tidx < 4:
                            base = q0
                            if tidx >= 2:
                                base = kk0
                            self._wait_flag(
                                mCnt, base // self.tile_shape_mnk[0] + tidx % 2
                            )
                    self.epilog_sync_barrier.arrive_and_wait()
                    if cutlass.const_expr("core" not in self.ablate):
                        yq = self._conv_channel(mMz, mConvW, mConv, slot, q0 + tidx, m)
                        yk = self._conv_channel(mMz, mConvW, mConv, slot, kk0 + tidx, m)
                        ssq = self._block_sum(yq * yq, sRed, warp_idx, lane)
                        ssk = self._block_sum(yk * yk, sRed, warp_idx, lane)
                        mQK[(it * 256 + tidx,)] = (
                            yq * cute.math.rsqrt(ssq + cutlass.Float32(1e-6)) * qscale
                        )
                        mQK[(it * 256 + 128 + tidx,)] = yk * cute.math.rsqrt(
                            ssk + cutlass.Float32(1e-6)
                        )
                    cute.arch.fence_acq_rel_gpu()
                    self.epilog_sync_barrier.arrive_and_wait()
                    if tidx == 0:
                        cute.arch.atomic_add(
                            mCnt.iterator + (B0_FLAGS + it),
                            cutlass.Int32(1),
                            sem="acq_rel",
                            scope="gpu",
                        )
                else:
                    # ---- phase B: (token m, value head h): v conv, b|a dots, gates, delta rule, gated norm
                    it = work - TA - TB0
                    m = it // self.hv
                    h = it % self.hv
                    kh = h // self.rep
                    slot = mSlots[m]
                    vc0 = 2 * KD + h * self.dv
                    zc0 = W + h * self.dv
                    if cutlass.const_expr(
                        "wait" not in self.ablate
                    ):  # v tiles + this key head's q|k flag, in parallel
                        if tidx < 2:
                            self._wait_flag(mCnt, vc0 // self.tile_shape_mnk[0] + tidx)
                        if tidx == 2:
                            self._wait_flag(mCnt, B0_FLAGS + m * self.hk + kh)
                    self.epilog_sync_barrier.arrive_and_wait()
                    if cutlass.const_expr("core" not in self.ablate):
                        b0 = (m * self.hk + kh) * 256
                        sQ[tidx] = mQK[(b0 + tidx,)]
                        sK[tidx] = mQK[(b0 + 128 + tidx,)]
                        ssm_ = cutlass.Float32(0.0)
                        for j in range(0, nt_rs, 1):
                            ssm_ = ssm_ + mRs[(j, m)]
                        rinv_m = cute.math.rsqrt(
                            ssm_ / k_total + cutlass.Float32(self.eps)
                        )
                        yv = self._conv_channel(mMz, mConvW, mConv, slot, vc0 + tidx, m)
                        # b | a projections: contiguous K1/128 slices, 16 B vector copies
                        gx = cute.make_tensor(
                            mXwIn.iterator
                            + cute.assume(m * self.k1 + tidx * self.kchunk, divby=8),
                            cute.make_layout((self.kchunk,)),
                        )
                        gwb = cute.make_tensor(
                            mWba.iterator
                            + cute.assume(h * self.k1 + tidx * self.kchunk, divby=8),
                            cute.make_layout((self.kchunk,)),
                        )
                        gwa = cute.make_tensor(
                            mWba.iterator
                            + cute.assume(
                                (self.hv + h) * self.k1 + tidx * self.kchunk, divby=8
                            ),
                            cute.make_layout((self.kchunk,)),
                        )
                        fx = cute.make_fragment_like(gx)
                        fwb = cute.make_fragment_like(gwb)
                        fwa = cute.make_fragment_like(gwa)
                        cute.autovec_copy(gx, fx)
                        cute.autovec_copy(gwb, fwb)
                        cute.autovec_copy(gwa, fwa)
                        pb = cutlass.Float32(0.0)
                        pa = cutlass.Float32(0.0)
                        for j in cutlass.range_constexpr(self.kchunk):
                            xn = fx[j].to(cutlass.Float32) * rinv_m
                            pb = pb + xn * fwb[j].to(cutlass.Float32)
                            pa = pa + xn * fwa[j].to(cutlass.Float32)
                        bsum = self._block_sum(pb, sRed, warp_idx, lane)
                        asum = self._block_sum(pa, sRed, warp_idx, lane)
                        b_bf = bsum.to(cutlass.BFloat16).to(cutlass.Float32)
                        a_bf = asum.to(cutlass.BFloat16).to(cutlass.Float32)
                        beta = (
                            (
                                cutlass.Float32(1.0)
                                / (cutlass.Float32(1.0) + cute.math.exp(-b_bf))
                            )
                            .to(cutlass.BFloat16)
                            .to(cutlass.Float32)
                        )
                        xg = a_bf + mDtb[(h,)]
                        spl = cute.math.log(cutlass.Float32(1.0) + cute.math.exp(xg))
                        if xg > cutlass.Float32(20.0):
                            spl = xg
                        g = -cute.math.exp(mAlog[(h,)]) * spl
                        eg = cute.math.exp(g)
                        # delta rule on column t; the head's state streams through smem in 32-row
                        # chunks: thread t copies a quarter row (8 x 16 B) per chunk, then reads its
                        # column out of smem (a warp reads 32 consecutive columns: conflict-free)
                        kvm = cutlass.Float32(0.0)
                        ot = cutlass.Float32(0.0)
                        if cutlass.const_expr("state" not in self.ablate):
                            head_base = (slot * self.hv + h) * 16384
                            if cutlass.const_expr(self.state_alias):
                                arow = tidx // 8
                                aseg = (tidx % 8) * 16
                                if cutlass.const_expr(self.bulk_load):
                                    # 64 KB of descriptor-free bulk copies issued by ONE thread (the DSL does
                                    # not elect a lane for CopyBulkG2SOp: every lane would issue -> 32x the
                                    # tx bytes -> mbarrier fault), completion on the state mbarrier
                                    if tidx == 0:
                                        cute.arch.mbarrier_arrive_and_expect_tx(
                                            state_bar, 65536
                                        )
                                        for q in cutlass.range_constexpr(
                                            self.bulk_parts
                                        ):
                                            qn = 16384 // self.bulk_parts
                                            gStF = cute.make_tensor(
                                                mSsm.iterator + (head_base + q * qn),
                                                cute.make_layout((qn,)),
                                            )
                                            sStF = cute.make_tensor(
                                                sStA.iterator + (q * qn),
                                                cute.make_layout((qn,)),
                                            )
                                            cute.copy(
                                                bulk_g2s, gStF, sStF, mbar_ptr=state_bar
                                            )
                                    cute.arch.mbarrier_wait(state_bar, st_phase)
                                    st_phase = st_phase ^ 1
                                else:
                                    # whole state -> stage smem in one cp.async burst (thread: 8 rows x 64 B, a
                                    # warp covers 4 full 512 B rows per instruction), one wait, one barrier
                                    for c in cutlass.range_constexpr(8):
                                        for v in cutlass.range_constexpr(4):
                                            cute.arch.cp_async_shared_global(
                                                sStA.iterator
                                                + (
                                                    (16 * c + arow) * 128 + aseg + 4 * v
                                                ),
                                                mSsm.iterator
                                                + (
                                                    head_base
                                                    + (16 * c + arow) * 128
                                                    + aseg
                                                    + 4 * v
                                                ),
                                                16,
                                                "cg",
                                            )
                                    cute.arch.cp_async_commit_group()
                                    cute.arch.cp_async_wait_group(0)
                                    self.epilog_sync_barrier.arrive_and_wait()
                                # pass 1 (scale + k.S) and pass 2 (update + q.S) on this thread's column: no
                                # cross-thread hazard between them (column-private smem)
                                # pass 1 (k.S) and pass 2 (update + q.S) read the column out of smem: a DYNAMIC
                                # loop over 16-row chunks (unroll=1) bounds the loads in flight — fully unrolled,
                                # ptxas hoisted all 128 state + k loads ahead of the FMA chain and spilled them
                                # (4 independent partial sums per pass: one 128-long FMA chain was latency-bound)
                                kv1 = cutlass.Float32(0.0)
                                kv2 = cutlass.Float32(0.0)
                                kv3 = cutlass.Float32(0.0)
                                for c8 in range(0, 8, 1, unroll=1):
                                    for i in cutlass.range_constexpr(4):
                                        r0 = c8 * 16 + 4 * i
                                        kvm = kvm + sStA[(r0, tidx)] * sK[r0]
                                        kv1 = kv1 + sStA[(r0 + 1, tidx)] * sK[r0 + 1]
                                        kv2 = kv2 + sStA[(r0 + 2, tidx)] * sK[r0 + 2]
                                        kv3 = kv3 + sStA[(r0 + 3, tidx)] * sK[r0 + 3]
                                kvm = ((kvm + kv1) + (kv2 + kv3)) * eg
                                delta = (yv - kvm) * beta
                                ot1 = cutlass.Float32(0.0)
                                ot2 = cutlass.Float32(0.0)
                                ot3 = cutlass.Float32(0.0)
                                for c8 in range(0, 8, 1, unroll=1):
                                    for i in cutlass.range_constexpr(4):
                                        r0 = c8 * 16 + 4 * i
                                        s0 = sStA[(r0, tidx)] * eg + sK[r0] * delta
                                        s1 = (
                                            sStA[(r0 + 1, tidx)] * eg
                                            + sK[r0 + 1] * delta
                                        )
                                        s2 = (
                                            sStA[(r0 + 2, tidx)] * eg
                                            + sK[r0 + 2] * delta
                                        )
                                        s3 = (
                                            sStA[(r0 + 3, tidx)] * eg
                                            + sK[r0 + 3] * delta
                                        )
                                        sStA[(r0, tidx)] = s0
                                        sStA[(r0 + 1, tidx)] = s1
                                        sStA[(r0 + 2, tidx)] = s2
                                        sStA[(r0 + 3, tidx)] = s3
                                        ot = ot + s0 * sQ[r0]
                                        ot1 = ot1 + s1 * sQ[r0 + 1]
                                        ot2 = ot2 + s2 * sQ[r0 + 2]
                                        ot3 = ot3 + s3 * sQ[r0 + 3]
                                ot = (ot + ot1) + (ot2 + ot3)
                                if cutlass.const_expr(self.bulk_store):
                                    # generic smem writes -> async proxy, then one bulk store; wait for its smem
                                    # READS before anyone (next item's load, out_proj TMA) reuses the buffers
                                    cute.arch.fence_view_async_shared()
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if tidx == 0:  # one issuing thread (see the load)
                                        gStF2 = cute.make_tensor(
                                            mSsm.iterator + head_base,
                                            cute.make_layout((16384,)),
                                        )
                                        sStF2 = cute.make_tensor(
                                            sStA.iterator, cute.make_layout((16384,))
                                        )
                                        cute.copy(bulk_s2g, sStF2, gStF2)
                                        cute.arch.cp_async_bulk_commit_group()
                                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                                    self.epilog_sync_barrier.arrive_and_wait()
                                else:
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    for c in cutlass.range_constexpr(8):
                                        sO = cute.make_tensor(
                                            sStA.iterator
                                            + ((16 * c + arow) * 128 + aseg),
                                            cute.make_layout((16,)),
                                        )
                                        gO = cute.make_tensor(
                                            mSsm.iterator
                                            + (
                                                head_base + (16 * c + arow) * 128 + aseg
                                            ),
                                            cute.make_layout((16,)),
                                        )
                                        cute.autovec_copy(sO, gO)
                                    # generic-proxy smem traffic must be fenced before the producer's next TMA
                                    # lands in these stage buffers (out_proj items)
                                    cute.arch.fence_view_async_shared()
                                    self.epilog_sync_barrier.arrive_and_wait()
                            else:
                                CR = self.chunk_rows
                                TPR = 128 // CR  # threads per row of the chunk
                                SEG = (
                                    128 // TPR
                                )  # elements each thread copies per chunk
                                crow = tidx // TPR
                                cseg = (tidx % TPR) * SEG
                                NCH = 128 // CR
                                NV = SEG // 4  # 16 B cp.async per thread per chunk
                                NB = self.nbuf
                                # pass 1 (scale + k.S): chunks stream through NB smem buffers with register-free
                                # cp.async at prefetch distance NB-1; one barrier per chunk (it both publishes
                                # chunk c and frees the buffer chunk c+NB-1 is about to be issued into)
                                for c0 in cutlass.range_constexpr(NB - 1):
                                    if c0 < NCH:
                                        for v in cutlass.range_constexpr(NV):
                                            cute.arch.cp_async_shared_global(
                                                sSt.iterator
                                                + (
                                                    c0 * CR * 128
                                                    + crow * 128
                                                    + cseg
                                                    + 4 * v
                                                ),
                                                mSsm.iterator
                                                + (
                                                    head_base
                                                    + (CR * c0 + crow) * 128
                                                    + cseg
                                                    + 4 * v
                                                ),
                                                16,
                                                "cg",
                                            )
                                        cute.arch.cp_async_commit_group()
                                for c in cutlass.range_constexpr(NCH):
                                    # chunk c must have landed: allow the (NB-2) later groups to stay pending
                                    if c + NB - 2 < NCH:
                                        cute.arch.cp_async_wait_group(NB - 2)
                                    else:
                                        cute.arch.cp_async_wait_group(0)
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if c + NB - 1 < NCH:
                                        cn = c + NB - 1
                                        for v in cutlass.range_constexpr(NV):
                                            cute.arch.cp_async_shared_global(
                                                sSt.iterator
                                                + (
                                                    (cn % NB) * CR * 128
                                                    + crow * 128
                                                    + cseg
                                                    + 4 * v
                                                ),
                                                mSsm.iterator
                                                + (
                                                    head_base
                                                    + (CR * cn + crow) * 128
                                                    + cseg
                                                    + 4 * v
                                                ),
                                                16,
                                                "cg",
                                            )
                                        cute.arch.cp_async_commit_group()
                                    for i in cutlass.range_constexpr(CR):
                                        sv = sSt[(c % NB, i, tidx)] * eg
                                        scol[CR * c + i] = sv
                                        kvm = kvm + sv * sK[CR * c + i]
                                self.epilog_sync_barrier.arrive_and_wait()  # everyone done reading before pass 2 overwrites
                                delta = (yv - kvm) * beta
                                # pass 2 (update + q.S + write-back): alternate buffers; global stores are
                                # fire-and-forget, the barrier only protects the smem buffer reuse
                                for c in cutlass.range_constexpr(NCH):
                                    for i in cutlass.range_constexpr(CR):
                                        sv = scol[CR * c + i] + sK[CR * c + i] * delta
                                        sSt[(c % NB, i, tidx)] = sv
                                        ot = ot + sv * sQ[CR * c + i]
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    sO = cute.make_tensor(
                                        sSt.iterator
                                        + ((c % NB) * CR * 128 + crow * 128 + cseg),
                                        cute.make_layout((SEG,)),
                                    )
                                    gO = cute.make_tensor(
                                        mSsm.iterator
                                        + (head_base + (CR * c + crow) * 128 + cseg),
                                        cute.make_layout((SEG,)),
                                    )
                                    cute.autovec_copy(sO, gO)
                                self.epilog_sync_barrier.arrive_and_wait()  # both buffers free for the next head
                        else:
                            ot = (yv - kvm) * beta
                        # the z (gate) tiles are needed only now: wait for them here so the state work
                        # overlaps the in_proj second wave that produces them
                        if cutlass.const_expr("wait" not in self.ablate):
                            if tidx < 2:
                                self._wait_flag(
                                    mCnt, zc0 // self.tile_shape_mnk[0] + tidx
                                )
                        self.epilog_sync_barrier.arrive_and_wait()
                        zv = mMz[(zc0 + tidx, m, 0)].to(cutlass.Float32)
                        xf = ot.to(cutlass.BFloat16).to(cutlass.Float32)
                        ssx = self._block_sum(xf * xf, sRed, warp_idx, lane)
                        rstd = cute.math.rsqrt(
                            ssx / cutlass.Float32(128.0) + cutlass.Float32(self.eps)
                        )
                        y_bf = (
                            (xf * rstd * mNormW[(tidx,)].to(cutlass.Float32))
                            .to(cutlass.BFloat16)
                            .to(cutlass.Float32)
                        )
                        sil = zv / (cutlass.Float32(1.0) + cute.math.exp(-zv))
                        mAttn[(h * self.dv + tidx, m, 0)] = (y_bf * sil).to(
                            mAttn.element_type
                        )
                    cute.arch.fence_acq_rel_gpu()
                    self.epilog_sync_barrier.arrive_and_wait()
                    if (
                        tidx == 0
                    ):  # this head's out_proj K-split group (all heads in group 0 when head_groups is off)
                        grp = cutlass.Int32(0)
                        if cutlass.const_expr(self.head_groups):
                            grp = h // HEADS_PER_GROUP
                        cute.arch.atomic_add(
                            mCnt.iterator + (HD + grp),
                            cutlass.Int32(1),
                            sem="acq_rel",
                            scope="gpu",
                        )
                work = work + gdim

        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()  # the activation (B) TMAs read the previous kernel's output
                cute.arch.griddepcontrol_launch_dependents()
            work = cutlass.Int32(bidx)
            while work < total:
                is_a = work < TA
                is_core = (work >= TA) and (work < TA + TB0 + TB)
                if is_a:
                    tAg = tAgA1[(None, work, None, 0)]
                    tBg = tBgB1[(None, 0, None, 0)]
                    mainloop_producer_state.reset_count()
                    for _k_tile in range(0, k_cnt_a, 1, unroll=1):
                        mainloop_pipeline.producer_acquire(mainloop_producer_state)
                        bar = mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        )
                        cnt_ = mainloop_producer_state.count
                        idx = mainloop_producer_state.index
                        cute.copy(
                            tma_a1,
                            tAg[(None, cnt_)],
                            tAsA[(None, idx)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_b1,
                            tBg[(None, cnt_)],
                            tBsB[(None, idx)],
                            tma_bar_ptr=bar,
                        )
                        mainloop_pipeline.producer_commit(mainloop_producer_state)
                        mainloop_producer_state.advance()
                elif not is_core:
                    jt = work - TA - TB0 - TB
                    n_tile = jt // self.nsplit
                    split = jt % self.nsplit
                    tAg2 = tAgA2[(None, n_tile, None, split)]
                    tBg2 = tBgB2[(None, 0, None, split)]
                    mainloop_producer_state.reset_count()
                    need = cutlass.Int32(TB)
                    grp = cutlass.Int32(0)
                    if cutlass.const_expr(self.head_groups):
                        need = (
                            TB // self.nsplit
                        )  # heads (x tokens) in this split's K range
                        grp = split
                    seen0 = cute.arch.atomic_add(
                        mCnt.iterator + (HD + grp),
                        cutlass.Int32(0),
                        sem="acquire",
                        scope="gpu",
                    )
                    while seen0 < need:
                        seen0 = cute.arch.atomic_add(
                            mCnt.iterator + (HD + grp),
                            cutlass.Int32(0),
                            sem="acquire",
                            scope="gpu",
                        )
                    cute.arch.fence_proxy("async.global")
                    for _k_tile in range(0, k_cnt_c, 1, unroll=1):
                        mainloop_pipeline.producer_acquire(mainloop_producer_state)
                        bar = mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        )
                        cnt_ = mainloop_producer_state.count
                        idx = mainloop_producer_state.index
                        cute.copy(
                            tma_a2,
                            tAg2[(None, cnt_)],
                            tAsA[(None, idx)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_b2,
                            tBg2[(None, cnt_)],
                            tBsB[(None, idx)],
                            tma_bar_ptr=bar,
                        )
                        mainloop_pipeline.producer_commit(mainloop_producer_state)
                        mainloop_producer_state.advance()
                work = work + gdim
            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return
