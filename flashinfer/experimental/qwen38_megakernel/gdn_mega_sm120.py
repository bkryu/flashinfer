# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# Qwen3.8-27B one-launch decoder on SM120 (FlashInfer experimental track).
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

Only the constructor, the attribute setup and the jit helpers (`_block_sum`, `_wait_flag`, `_conv_channel`,
`_emit_sq`) are used here (as the base of the one-launch decoder); the standalone one-layer GDN launch
(`__call__` / `kernel`) was removed from this copy.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils

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
