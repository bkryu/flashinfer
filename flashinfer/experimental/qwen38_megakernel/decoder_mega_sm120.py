# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# Ported from LightLM's cutlass_dsl megakernel study (docs/megakernel-plan.md there) for the FlashInfer experimental
# track: the Qwen3.8-27B one-launch decoder on SM120.
"""Megakernel study phase 6 (docs/megakernel-plan.md §13): the WHOLE Qwen3.8-27B decoder as ONE
launch. The any-layer kernel (§12) inside an outer loop over layers:

  for L in 0..NL-1:
      per-layer state from three device tables (Int64 pointers, Int32 counts/kind/bounds, fp32
      alphas); the layer's weight TMA descriptors are REWRITTEN in a per-CTA global workspace
      (base / shape / strides; box, swizzle and dtype are the atoms') by the producer warp,
      release+acquire fenced, and the weight copies read them through `tma_desc_ptr`;
      the layer's work list runs exactly as in the any-layer kernel (persistent striding,
      flags, split-K reducers, CTA phase switch fp8 -> NVFP4) over the layer's own counter
      region (L * CS);
      layer L+1 starts (both warp roles) once a single monotonic layer-done counter has
      received all NT2 down reducers of layer L — that spin is also what makes the descriptor
      rewrite safe (every TMA load of layer L has been consumed by then).

Fixed across layers: the activation scratch (xw ping-pong, rs / sq stats, mz, attn, act, ws),
resid, the attention metadata and the KV split count. Per layer: kind, N1, K2 (out-projection
K), item counts, alpha boundaries, tile-order rotation, heads-done count, conv width, pool /
cache / weight pointers, norm weights, alphas.
"""

from __future__ import annotations


import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import AddressSpace
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync

from .gdn_mega_sm120 import GdnMegaSm120

LOG2E = 1.4426950408889634

# ---- table columns (mirrored by decoder_fused.py)
(
    IT_KIND,
    IT_N1,
    IT_K2,
    IT_TA,
    IT_TB0,
    IT_TB,
    IT_KCC,
    IT_B1,
    IT_B2,
    IT_B3,
    IT_KT0,
    IT_HDN,
    IT_W,
    IT_TB0I,
    IT_TBKV,
    IT_TF,
) = range(16)
NI = 16
(
    PT_WIN,
    PT_WOUT,
    PT_WGU,
    PT_SGU,
    PT_WDN,
    PT_SDN,
    PT_WBA,
    PT_CONVW,
    PT_ALOG,
    PT_DTB,
    PT_NORMW,
    PT_WQ,
    PT_WK,
    PT_KC,
    PT_VC,
    PT_CONV,
    PT_SSM,
    PT_WNMID,
    PT_WNOUT,
    PT_STX,
    PT_STY,
    PT_STA,
    PT_STB,
    PT_WFC,
    PT_WPRE,
    PT_WNIN,
) = range(26)
NP = 28
FOLD_CNT = 64  # §19.7: per-layer counter slots reserved above the layer's own for the fold's fc / final-norm counters (+ NT2)
NF = 8  # alpha_in[4], alpha_out, alpha_g, alpha_u, alpha_dn
NTMAP = 7  # a1, a2, b2, a3, s3, a4, s4
BIG = 1 << 30  # flat scratch / cache layouts (indexing never consults the bound)


class DecoderMegaSm120(GdnMegaSm120):
    def __init__(
        self,
        acc_dtype,
        tile_shape_mnk,
        hk,
        hv,
        k1,
        hq,
        hkv,
        d,
        rot_dim,
        page,
        inter,
        nsplit=2,
        eps=1e-6,
        occupancy=1,
        pdl=0,
        prefetch=True,
        l2_prefetch=False,
        xb_prefetch=True,
        profile=False,
        mlp_stages=4,
        mma_attn=True,
        dual_acc=True,
        kring3=False,
        tail_split=False,
        spec_rows=1,
        wtype="fp8fp4",
        w16_stages=4,
        drafter_fold=False,
    ):
        super().__init__(
            acc_dtype,
            tile_shape_mnk,
            hk,
            hv,
            k1,
            nsplit=nsplit,
            eps=eps,
            occupancy=occupancy,
            state_alias=True,
            state_bulk=True,
            pdl=pdl,
        )
        if tile_shape_mnk[2] % 256:
            raise ValueError("TILE_K must be a multiple of 256 (fp4 scale tile rows)")
        self.l2_prefetch = bool(
            l2_prefetch
        )  # lever 1: L2 prefetch of the next item's weights before each dependency spin
        self.xb_prefetch = bool(
            xb_prefetch
        )  # lever 5: next layer's first in-projection weight stages issued across the layer boundary
        self.profile = bool(
            profile
        )  # §15 timeline: per-item / per-wait globaltimer stamps into a per-CTA record buffer
        self.mlp_stage = int(
            mlp_stages
        )  # §15 lever A: the NVFP4 ring is deeper than the fp8 ring (17 KB stages vs 24 KB)
        self.mma_attn = bool(
            mma_attn
        )  # §16 item 3b: tensor-core attention page loop (cp.async-staged K/V pages in the ring smem)
        self.dual_acc = bool(
            dual_acc
        )  # §17 item 1: two dequant/accumulator chains per warp in the NVFP4 mainloop (ILP)
        self.kring3 = bool(
            kring3
        )  # §17 item 5: three K page buffers (K two pages ahead), Q fragments in registers
        self.nkb = 3 if self.kring3 else 2
        self.tail_split = bool(
            tail_split
        )  # §17 item 2: in-projection tail tiles as two K halves + last-arriver reduce
        self.R = int(
            spec_rows
        )  # §18: rows per sequence (1 = plain decode; R > 1 = the MTP verify form, pools read-only)
        self.w16 = (
            wtype == "bf16"
        )  # §19: unquantized (bf16) weights everywhere — the MTP drafter layer; all phases on the fp8 ring
        self.w16_stages = int(
            w16_stages
        )  # §19: fp8-ring depth of the bf16 kind (the fp4 ring's smem is free)
        self.fold = bool(
            drafter_fold
        )  # §19.7: fc + pre-fc norms in front of layer 0, final norm behind the last layer
        if self.fold and (not self.w16 or self.nsplit != 2):
            raise ValueError(
                "drafter_fold needs the bf16 weight kind with 2 splits (one per pre-fc norm half)"
            )
        self.hq, self.hkv, self.d = int(hq), int(hkv), int(d)
        if (
            self.hq % self.hkv
            or self.d != 256
            or int(page) != 32
            or self.hq // self.hkv > 8
        ):
            raise ValueError("attention core v1: head_dim 256, page 32, GQA group <= 8")
        self.group = self.hq // self.hkv
        self.rot = int(rot_dim)
        self.page = int(page)
        self.inter = int(inter)
        self.a4_tile_mnk = (
            tile_shape_mnk[0],
            tile_shape_mnk[1],
            tile_shape_mnk[2] // 4,
        )
        self.s_tile = (tile_shape_mnk[0], tile_shape_mnk[2] // 16)
        self.prefetch = bool(prefetch)

    def _setup_attributes(self):
        super()._setup_attributes()
        tn = self.tile_shape_mnk[1]
        if tn != 16:
            # §17 item 1b: narrow activation tile (M <= 8): a single MMA n-tile; B stages carry half the bytes
            op = cute.nvgpu.warp.MmaF16BF16Op(
                self.b_dtype, self.acc_dtype, self.mma_inst_mnk
            )
            self.tiled_mma = cute.make_tiled_mma(
                op,
                cute.make_layout(self.atom_layout),
                permutation_mnk=(
                    self.atom_layout[0] * self.mma_inst_mnk[0],
                    tn,
                    self.mma_inst_mnk[2],
                ),
            )
        cap = (
            self.w16_stages if self.w16 else 3
        )  # the megakernel's own smem residents cap the fp8 ring (the base sizes it from the raw budget)
        if self.ab_stage != cap:
            self.ab_stage = cap
            self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
                self.a_layout, self.a_tile_mnk, self.a_dtype, self.ab_stage
            )
            self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
                self.b_layout, self.tile_shape_mnk, self.b_dtype, self.ab_stage
            )
        if self.w16:
            # §19 bf16 kind: the MLP ring is the fp8 ring's smem (same stage layouts, same depth); separate barriers / states
            self.mlp_stage = self.ab_stage
            self.a4_smem_layout_staged = self.a_smem_layout_staged
            self.b4_smem_layout_staged = self.b_smem_layout_staged
        else:
            self.a4_smem_layout_staged = sm90_utils.make_smem_layout_a(
                self.a_layout, self.a4_tile_mnk, self.a_dtype, self.mlp_stage
            )
            self.b4_smem_layout_staged = sm90_utils.make_smem_layout_b(
                self.b_layout, self.tile_shape_mnk, self.b_dtype, self.mlp_stage
            )
        tks = self.s_tile[1]
        self.s_smem_layout_staged = cute.make_layout(
            (self.s_tile[0], tks, self.mlp_stage), stride=(tks, 1, self.s_tile[0] * tks)
        )
        if cute.cosize(self.a4_smem_layout_staged) > cute.cosize(
            self.a_smem_layout_staged
        ):
            raise ValueError("fp4 A stages do not fit under the fp8 A stages")
        self.sb_elems = max(
            cute.cosize(self.b_smem_layout_staged),
            cute.cosize(self.b4_smem_layout_staged),
        )
        ring_bytes = (
            cute.cosize(self.a_smem_layout_staged) * self.a_dtype.width // 8
            + self.sb_elems * self.b_dtype.width // 8
        )
        ring_min = ((80 if self.kring3 else 72) * 1024) if self.mma_attn else 0
        if self.R > 1 and self.mma_attn and not self.kring3:
            ss_b = cute.cosize(self.s_smem_layout_staged) * self.s_dtype.width // 8
            ring_min = max(
                ring_min, 65536 + self.R * self.group * self.d * 2 - ss_b
            )  # the R*G-row Q tile after the K/V buffers
        if self.w16 and ring_bytes < ring_min:
            # the attention core's smem need does not depend on the stage count: pad the B region
            self.sb_elems += (ring_min - ring_bytes + self.b_dtype.width // 8 - 1) // (
                self.b_dtype.width // 8
            )
            ring_bytes = (
                self.sb_elems * self.b_dtype.width // 8
                + cute.cosize(self.a_smem_layout_staged) * self.a_dtype.width // 8
            )
        if self.R > 1:
            if not self.mma_attn or self.kring3:
                raise ValueError(
                    "spec_rows > 1 needs the tensor-core attention core with the 2-buffer ring"
                )
            nq_bytes = self.R * self.group * self.d * 2  # R*G rows of bf16 Q
            ss_bytes = cute.cosize(self.s_smem_layout_staged) * self.s_dtype.width // 8
            if 65536 + nq_bytes > ring_bytes + ss_bytes:
                raise ValueError(
                    f"spec_rows={self.R}: the {self.R * self.group}-row Q tile does not fit after the K/V buffers"
                )
        if self.mma_attn and ring_bytes < (80 if self.kring3 else 72) * 1024:
            raise ValueError(
                f"mma_attn needs 72 KB of ring smem (K/V double buffer + Q), have {ring_bytes}"
            )

    @cute.jit
    def _prec(self, mProf, mPcnt, blk, code, t0, t1, work):
        """One timeline record for (cta, role) block `blk`; the slot comes from a per-block atomic counter
        (loop-carried slot counters lost inner increments across the layer loop)."""
        slot = cute.arch.atomic_add(
            mPcnt.iterator + blk, cutlass.Int32(1), sem="relaxed", scope="gpu"
        )
        o = (blk * 4096 + slot) * 4
        mProf[(o,)] = cutlass.Int64(code)
        mProf[(o + 1,)] = t0
        mProf[(o + 2,)] = t1
        mProf[(o + 3,)] = cutlass.Int64(work)

    @cute.jit
    def _pf(self, base, nbytes, lane):
        """Warp-cooperative L2 prefetch of [base, base + nbytes) in 4 KB bulk pieces (lever 1, §14):
        issued by the producer warp before a dependency spin so HBM streams the next item's weights
        while the grid drains; nbytes is a multiple of 16."""
        n = (nbytes + 4095) // 4096
        for i in range(lane, n, 32):
            off = i * 4096
            sz = cutlass.min(cutlass.Int32(4096), nbytes - off)
            addr = base + cutlass.Int64(off)
            llvm.inline_asm(
                None,
                [addr.ir_value(), sz.ir_value()],
                "cp.async.bulk.prefetch.L2.global [$0], $1;",
                "l,r",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )

    @cute.jit
    def _pf_rows(self, base, row0, stride, col_off, seg, lane):
        """L2 prefetch of a 64-row x seg-byte block: rows row0.., byte stride `stride`, column offset col_off."""
        for r in range(lane, 64, 32):
            a0 = (
                base
                + cutlass.Int64(row0 + r) * cutlass.Int64(stride)
                + cutlass.Int64(col_off)
            )
            n = (seg + 4095) // 4096
            for i in range(0, n, 1):
                off = i * 4096
                sz = cutlass.min(cutlass.Int32(4096), seg - off)
                addr = a0 + cutlass.Int64(off)
                llvm.inline_asm(
                    None,
                    [addr.ir_value(), sz.ir_value()],
                    "cp.async.bulk.prefetch.L2.global [$0], $1;",
                    "l,r",
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )

    @cute.jit
    def _warp_sum(self, v):
        for off in [16, 8, 4, 2, 1]:
            v = v + cute.arch.shuffle_sync_bfly(v, offset=off)
        return v

    @cute.jit
    def _warp_max(self, v):
        for off in [16, 8, 4, 2, 1]:
            o = cute.arch.shuffle_sync_bfly(v, offset=off)
            if o > v:
                v = o
        return v

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        # TMA templates (layouts + atoms; the per-layer descriptors are rewritten in the kernel)
        a1: cute.Tensor,
        b1: cute.Tensor,
        c1: cute.Tensor,  # in-proj container (N1, K1/2, 1) / xw0 view (M, K1, 1) / dummy (N1, M, 1)
        a2: cute.Tensor,
        b2: cute.Tensor,
        ws: cute.Tensor,  # out-proj container (H, K2/2/S, S) / attn view (M, K2/S, S) / ws (H, M, S)
        wsa: cute.Tensor,  # in-proj tail half-partials (N1_max, M, 2) fp32
        a3: cute.Tensor,
        s3: cute.Tensor,
        b3: cute.Tensor,
        c3: cute.Tensor,  # gate_up (2I, K1/4, 1), scales, xw1 view, dummy (2I, M, 1)
        a4: cute.Tensor,
        s4: cute.Tensor,
        b4: cute.Tensor,  # down (H, I/4/S, S), scales, act view (M, I/S, S)
        fa: cute.Tensor,
        fb: cute.Tensor,
        out: cute.Tensor,  # §19.7 fold: fc weight (H, 2H/S, S) bf16, cat(e|h) doubled view, out (H, M, 1)
        # shared activations / scratch
        xw0: cute.Tensor,
        xw1: cute.Tensor,  # (H, M, 1) generic views of the ping-pong buffers (xw0 = layer input & output, xw1 = mid)
        sq0: cute.Tensor,
        sq1: cute.Tensor,  # (H/64, M) fp32 stats (sq0 = layer input & output, sq1 = mid)
        mz: cute.Tensor,
        attn: cute.Tensor,  # flat bf16 scratch, sized for the largest N1 / K2
        resid: cute.Tensor,
        act: cute.Tensor,  # (H, M, 1), (I, M, 1)
        qk: cute.Tensor,
        part_out: cute.Tensor,
        part_lse: cute.Tensor,
        qbuf: cute.Tensor,  # core staging
        rope: cute.Tensor,
        pos: cute.Tensor,
        slot: cute.Tensor,
        seqlen: cute.Tensor,
        bt: cute.Tensor,
        slots: cute.Tensor,
        cnt: cute.Tensor,
        tmaps: cute.Tensor,  # counters (NL*CS + 32), tensormap workspace (num_ctas, NTMAP, 16) Int64
        ptab: cute.Tensor,
        itab: cute.Tensor,
        ftab: cute.Tensor,  # (NL, NP) Int64, (NL, NI) Int32, (NL, NF) Float32
        prof: cute.Tensor,  # Int64 flat: [(cta * 2 + role) * NSLOT + slot] * 4 + {code, t0, t1, work}; 1 element when profile is off
        pcnt: cute.Tensor,  # Int32 [n_ctas * 2] record counters
        nl: cutlass.Int32,
        cs: cutlass.Int32,
        kv_splits: cutlass.Int32,
        bt_stride: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype, self.b_dtype, self.c_dtype = (
            a1.element_type,
            b1.element_type,
            c1.element_type,
        )
        self.s_dtype = s3.element_type
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
        if cutlass.const_expr(self.w16):
            tma_a3, mA3 = self._make_tma_atoms_and_tensors(
                a3,
                self.a_smem_layout_staged,
                (self.a_tile_mnk[0], self.a_tile_mnk[2]),
                1,
            )
            tma_a4, mA4 = self._make_tma_atoms_and_tensors(
                a4,
                self.a_smem_layout_staged,
                (self.a_tile_mnk[0], self.a_tile_mnk[2]),
                1,
            )
            tma_b3, mB3 = self._make_tma_atoms_and_tensors(
                b3,
                self.b_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
                1,
            )
            tma_b4, mB4 = self._make_tma_atoms_and_tensors(
                b4,
                self.b_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
                1,
            )
        else:
            tma_a3, mA3 = self._make_tma_atoms_and_tensors(
                a3,
                self.a4_smem_layout_staged,
                (self.a4_tile_mnk[0], self.a4_tile_mnk[2]),
                1,
            )
            tma_a4, mA4 = self._make_tma_atoms_and_tensors(
                a4,
                self.a4_smem_layout_staged,
                (self.a4_tile_mnk[0], self.a4_tile_mnk[2]),
                1,
            )
            tma_b3, mB3 = self._make_tma_atoms_and_tensors(
                b3,
                self.b4_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
                1,
            )
            tma_b4, mB4 = self._make_tma_atoms_and_tensors(
                b4,
                self.b4_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
                1,
            )
        tma_s3, mS3 = self._make_tma_atoms_and_tensors(
            s3, self.s_smem_layout_staged, self.s_tile, 1
        )
        tma_s4, mS4 = self._make_tma_atoms_and_tensors(
            s4, self.s_smem_layout_staged, self.s_tile, 1
        )
        tma_fa, mFA = self._make_tma_atoms_and_tensors(
            fa, self.a_smem_layout_staged, (self.a_tile_mnk[0], self.a_tile_mnk[2]), 1
        )
        tma_fb, mFB = self._make_tma_atoms_and_tensors(
            fb,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        k_cnt_f = fb.shape[2] // self.nsplit if self.fold else 0
        nt2 = ws.shape[0] // self.tile_shape_mnk[0]
        tg = c3.shape[0] // self.tile_shape_mnk[0]
        if cutlass.const_expr(self.w16):
            # doubled B views (n_tok, 256, K/128): the tile mode counts 128-wide k tiles; split phases pack split*k + k
            k_cnt_a = b1.shape[2]
            k_cnt_g = b3.shape[2]
            k_cnt_d = b4.shape[2] // self.nsplit
        else:
            k_cnt_a = b1.shape[1] // self.tile_shape_mnk[2]
            k_cnt_g = b3.shape[1] // self.tile_shape_mnk[2]
            k_cnt_d = b4.shape[1] // self.tile_shape_mnk[2]
        kd = self.hk * self.dk
        vd = self.hv * self.dv
        grid = (
            cutlass.Int32(max_active_clusters),
            1,
            1,
        )  # every CTA owns a descriptor set; layers stride the same grid
        G = self.group
        QCORE = max(128, G * 256)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            mlp_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mlp_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, self.sb_elems],
                self.buffer_align_bytes,
            ]
            sS: cute.struct.Align[
                cute.struct.MemRange[
                    self.s_dtype, cute.cosize(self.s_smem_layout_staged)
                ],
                128,
            ]
            epi_flag: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 8], 16]
            sq_part: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 64], 16]
            rinv_c: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 32], 16
            ]  # §16: per-CTA 1/rms cache (2 stats x 16 tokens)
            core_q: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, QCORE], 16]
            core_k: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 128], 16]
            core_part: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 4 * 32 * 8], 16
            ]
            core_p: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 32 * 8], 16]
            core_stat: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 4 * 8], 16
            ]
            core_red: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 4 * 8], 16
            ]
            core_tmp: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 256], 16]
            state_bar: cute.struct.MemRange[cutlass.Int64, 1]

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
            tma_a3,
            mA3,
            tma_s3,
            mS3,
            tma_b3,
            mB3,
            tma_a4,
            mA4,
            tma_s4,
            mS4,
            tma_b4,
            mB4,
            tma_fa,
            mFA,
            tma_fb,
            mFB,
            xw0,
            xw1,
            sq0,
            sq1,
            mz,
            attn,
            resid,
            act,
            out,
            qk,
            part_out,
            part_lse,
            qbuf,
            rope,
            pos,
            slot,
            seqlen,
            bt,
            slots,
            ws,
            wsa,
            cnt,
            tmaps,
            ptab,
            itab,
            ftab,
            prof,
            pcnt,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.a4_smem_layout_staged,
            self.s_smem_layout_staged,
            self.b4_smem_layout_staged,
            nl,
            cs,
            nt2,
            tg,
            k_cnt_a,
            k_cnt_g,
            k_cnt_d,
            k_cnt_f,
            kd,
            vd,
            kv_splits,
            bt_stride,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            use_pdl=bool(self.pdl),
        )

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
        tma_a3: cute.CopyAtom,
        mA3: cute.Tensor,
        tma_s3: cute.CopyAtom,
        mS3: cute.Tensor,
        tma_b3: cute.CopyAtom,
        mB3: cute.Tensor,
        tma_a4: cute.CopyAtom,
        mA4: cute.Tensor,
        tma_s4: cute.CopyAtom,
        mS4: cute.Tensor,
        tma_b4: cute.CopyAtom,
        mB4: cute.Tensor,
        tma_fa: cute.CopyAtom,
        mFA: cute.Tensor,
        tma_fb: cute.CopyAtom,
        mFB: cute.Tensor,
        mXw0: cute.Tensor,
        mXw1: cute.Tensor,
        mSq0: cute.Tensor,
        mSq1: cute.Tensor,
        mMzF: cute.Tensor,
        mAttnF: cute.Tensor,
        mRes: cute.Tensor,
        mAct: cute.Tensor,
        mOut: cute.Tensor,
        mQK: cute.Tensor,
        mPO: cute.Tensor,
        mPL: cute.Tensor,
        mQB: cute.Tensor,
        mRope: cute.Tensor,
        mPos: cute.Tensor,
        mSlot: cute.Tensor,
        mSeq: cute.Tensor,
        mBT: cute.Tensor,
        mSlots: cute.Tensor,
        mWs: cute.Tensor,
        mWsA: cute.Tensor,
        mCnt: cute.Tensor,
        mTmaps: cute.Tensor,
        mPtab: cute.Tensor,
        mItab: cute.Tensor,
        mFtab: cute.Tensor,
        mProf: cute.Tensor,
        mPcnt: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        a4_smem_layout_staged: cute.ComposedLayout,
        s_smem_layout_staged: cute.Layout,
        b4_smem_layout_staged: cute.ComposedLayout,
        NL: cutlass.Int32,
        CS: cutlass.Int32,
        NT2: cutlass.Int32,
        TG: cutlass.Int32,
        k_cnt_a: cutlass.Int32,
        k_cnt_g: cutlass.Int32,
        k_cnt_d: cutlass.Int32,
        k_cnt_f: cutlass.Int32,
        KD: cutlass.Int32,
        VD: cutlass.Int32,
        KV_SPLITS: cutlass.Int32,
        BT_STRIDE: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tidx % 32
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_b1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_b3)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_b4)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        a4_smem_layout = cute.slice_(a4_smem_layout_staged, (None, None, 0))
        s_smem_layout = cute.slice_(s_smem_layout_staged, (None, None, 0))
        tma_bytes_fp8 = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        b4_smem_layout = cute.slice_(b4_smem_layout_staged, (None, None, 0))
        tma_bytes_fp4 = (
            cute.size_in_bytes(self.a_dtype, a4_smem_layout)
            + cute.size_in_bytes(self.s_dtype, s_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b4_smem_layout)
        )
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        state_bar = storage.state_bar.data_ptr()
        if tidx == 0:
            cute.arch.mbarrier_init(state_bar, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        mix_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mma_warps
            ),
            tx_count=tma_bytes_fp8,
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        mlp_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.mlp_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mma_warps
            ),
            tx_count=tma_bytes_fp8 if self.w16 else tma_bytes_fp4,
            barrier_storage=storage.mlp_pipeline_array_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sA4 = storage.sA.get_tensor(
            a4_smem_layout_staged.outer, swizzle=a4_smem_layout_staged.inner
        )
        sB4 = storage.sB.get_tensor(
            b4_smem_layout_staged.outer, swizzle=b4_smem_layout_staged.inner
        )
        sS = storage.sS.get_tensor(s_smem_layout_staged)
        sFlag = storage.epi_flag.get_tensor(cute.make_layout((8,)))
        sSq = storage.sq_part.get_tensor(
            cute.make_layout((4, 4, 2, 2), stride=(16, 4, 2, 1))
        )
        sRinv = storage.rinv_c.get_tensor(cute.make_layout((2, 16), stride=(16, 1)))
        G = self.group
        D = self.d
        sQg = storage.core_q.get_tensor(cute.make_layout((128,)))
        sK = storage.core_k.get_tensor(cute.make_layout((128,)))
        sQ = cute.make_tensor(
            storage.core_q.data_ptr(), cute.make_layout((G, D), stride=(D, 1))
        )
        sPart = storage.core_part.get_tensor(
            cute.make_layout((4, 32, 8), stride=(256, 8, 1))
        )
        sP = storage.core_p.get_tensor(cute.make_layout((32, 8), stride=(8, 1)))
        sStat = storage.core_stat.get_tensor(cute.make_layout((4, 8), stride=(8, 1)))
        sRed = storage.core_red.get_tensor(cute.make_layout((4, 8), stride=(8, 1)))
        sRed1 = storage.core_red.get_tensor(cute.make_layout((8,)))
        sTmp = storage.core_tmp.get_tensor(cute.make_layout((256,)))
        sStA = cute.make_tensor(
            cute.recast_ptr(storage.sA.data_ptr(), dtype=cutlass.Float32),
            cute.make_layout((128, 128), stride=(128, 1)),
        )
        # §16 item 3b: attention-core smem, aliasing the TMA ring like the GDN state does (core_bar guards the producer):
        #   [0, 32 KB) K pages x2, [32, 64 KB) V pages x2, [64, 72 KB) Q bf16 16 x 256, [72, 73 KB) P bf16 16 x 32.
        #   512-B rows are swizzled 16-B chunk ^= row % 8 (byte Swizzle<3,4,5>) so ldmatrix / cp.async stay conflict-free.
        p_ring = cute.recast_ptr(storage.sA.data_ptr(), dtype=self.b_dtype)
        sw_kv = cute.make_swizzle(3, 4, 5)
        # §17 item 5: K x3 [0, 48 KB), V x2 [48, 80 KB); Q is staged through K buffer 2 into registers; P lives in core_p
        sKp = cute.make_tensor(
            cute.recast_ptr(p_ring, swizzle_=sw_kv, dtype=self.b_dtype),
            cute.make_layout((self.page, D, self.nkb), stride=(D, 1, self.page * D)),
        )
        sV = cute.make_tensor(
            cute.recast_ptr(
                p_ring + self.nkb * self.page * D, swizzle_=sw_kv, dtype=self.b_dtype
            ),
            cute.make_layout((D, self.page, 2), stride=(1, D, self.page * D)),
        )
        # Q: with 3 K buffers it is staged through K buffer 2 into registers; with 2 it lives at 64 KB and is re-read per page
        sQb = cute.make_tensor(
            cute.recast_ptr(
                p_ring + (2 if self.kring3 else 4) * self.page * D,
                swizzle_=sw_kv,
                dtype=self.b_dtype,
            ),
            cute.make_layout((16, D), stride=(D, 1)),
        )
        sPb = cute.make_tensor(
            cute.recast_ptr(
                storage.core_p.data_ptr(),
                swizzle_=cute.make_swizzle(2, 4, 3),
                dtype=self.b_dtype,
            ),
            cute.make_layout((16, self.page), stride=(self.page, 1)),
        )
        sSc = cute.make_tensor(
            sPart.iterator, cute.make_layout((16, self.page), stride=(self.page, 1))
        )
        bulk_g2s = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), cutlass.Float32)
        bulk_s2g = cute.make_copy_atom(cpasync.CopyBulkS2GOp(), cutlass.Float32)
        phase_bar = pipeline.NamedBarrier(
            barrier_id=3, num_threads=self.threads_per_cta
        )
        core_bar = pipeline.NamedBarrier(
            barrier_id=4, num_threads=self.threads_per_cta
        )  # ring-alias guard (§15)

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
        gFA = cute.local_tile(
            mFA, cute.slice_(self.a_tile_mnk, (None, 0, None)), (None, None, None)
        )
        gFB = cute.local_tile(
            mFB, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None)
        )
        gA3 = cute.local_tile(
            mA3,
            cute.slice_(
                self.a_tile_mnk if self.w16 else self.a4_tile_mnk, (None, 0, None)
            ),
            (None, None, None),
        )
        gS3 = cute.local_tile(mS3, self.s_tile, (None, None, None))
        gB3 = cute.local_tile(
            mB3, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None)
        )
        gA4 = cute.local_tile(
            mA4,
            cute.slice_(
                self.a_tile_mnk if self.w16 else self.a4_tile_mnk, (None, 0, None)
            ),
            (None, None, None),
        )
        gS4 = cute.local_tile(mS4, self.s_tile, (None, None, None))
        gB4 = cute.local_tile(
            mB4, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None)
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
        _, tAgFA = cute.nvgpu.cpasync.tma_partition(
            tma_fa, 0, one, cute.group_modes(sA, 0, 2), cute.group_modes(gFA, 0, 2)
        )
        _, tBgFB = cute.nvgpu.cpasync.tma_partition(
            tma_fb, 0, one, cute.group_modes(sB, 0, 2), cute.group_modes(gFB, 0, 2)
        )
        tAsA4, tAgA3 = cute.nvgpu.cpasync.tma_partition(
            tma_a3, 0, one, cute.group_modes(sA4, 0, 2), cute.group_modes(gA3, 0, 2)
        )
        tSsS, tSgS3 = cute.nvgpu.cpasync.tma_partition(
            tma_s3, 0, one, cute.group_modes(sS, 0, 2), cute.group_modes(gS3, 0, 2)
        )
        tBsB4, tBgB3 = cute.nvgpu.cpasync.tma_partition(
            tma_b3, 0, one, cute.group_modes(sB4, 0, 2), cute.group_modes(gB3, 0, 2)
        )
        _, tAgA4 = cute.nvgpu.cpasync.tma_partition(
            tma_a4, 0, one, cute.group_modes(sA4, 0, 2), cute.group_modes(gA4, 0, 2)
        )
        _, tSgS4 = cute.nvgpu.cpasync.tma_partition(
            tma_s4, 0, one, cute.group_modes(sS, 0, 2), cute.group_modes(gS4, 0, 2)
        )
        _, tBgB4 = cute.nvgpu.cpasync.tma_partition(
            tma_b4, 0, one, cute.group_modes(sB4, 0, 2), cute.group_modes(gB4, 0, 2)
        )

        tCsA = thr_mma.partition_A(sA)
        perm_k8 = cute.make_layout(
            ((2, 4, 2), self.tile_shape_mnk[2] // 16), stride=((1, 4, 2), 16)
        )
        sB_perm8 = cute.composition(
            sB,
            (
                cute.make_layout(self.tile_shape_mnk[1]),
                perm_k8,
                cute.make_layout(self.ab_stage),
            ),
        )
        tCsB8 = thr_mma.partition_B(sB_perm8)
        tCrA_c = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB8 = tiled_mma.make_fragment_B(tCsB8[None, None, None, 0])
        mma_m = cute.size(tCrA_c, mode=[1])
        tCrA_log8 = cute.make_rmem_tensor(
            cute.make_layout(((2, 2, 2), mma_m, 2)), self.b_dtype
        )
        tCsA4 = thr_mma.partition_A(sA4)
        perm_k4 = cute.make_layout(
            ((2, 4, 2, 2, 2), self.tile_shape_mnk[2] // 64),
            stride=((1, 8, 2, 4, 32), 64),
        )
        sB_perm4 = cute.composition(
            sB4,
            (
                cute.make_layout(self.tile_shape_mnk[1]),
                perm_k4,
                cute.make_layout(self.mlp_stage),
            ),
        )
        tCsB4 = thr_mma.partition_B(sB_perm4)
        # §19 bf16 kind: B fragments in the canonical m16n8k16 k order (the perm_k8 / perm_k4 shuffles pair with the fp8 / fp4
        # A-conversion orders; a plain ldmatrix A fragment needs the unpermuted B)
        tCsB16 = thr_mma.partition_B(sB)
        tCsB16_4 = thr_mma.partition_B(sB4)
        tCcB = thr_mma.partition_B(
            cute.make_identity_tensor((self.tile_shape_mnk[1], self.tile_shape_mnk[2]))
        )  # (token, k) of each B fragment element
        tCrA4_c = tiled_mma.make_fragment_A(tCsA4[None, None, None, 0])
        tCrB4 = tiled_mma.make_fragment_B(tCsB4[None, None, None, 0])
        tCrA_log4 = cute.make_rmem_tensor(
            cute.make_layout(((2, 2, 2), mma_m, 4)), self.b_dtype
        )
        tCcC = thr_mma.partition_C(
            cute.make_identity_tensor((self.tile_shape_mnk[0], self.tile_shape_mnk[1]))
        )
        accumulators = cute.make_rmem_tensor(tCcC.shape[:3], self.acc_dtype)

        # ---- per-CTA TMA descriptor workspace (rewritten per layer by the producer warp)
        tmm = utils.TensorMapManager(utils.TensorMapUpdateMode.GMEM, 128)
        tm_a1 = tmm.get_tensormap_ptr(mTmaps[(bidx, 0, None)].iterator)
        tm_a2 = tmm.get_tensormap_ptr(mTmaps[(bidx, 1, None)].iterator)
        tm_b2 = tmm.get_tensormap_ptr(mTmaps[(bidx, 2, None)].iterator)
        tm_a3 = tmm.get_tensormap_ptr(mTmaps[(bidx, 3, None)].iterator)
        tm_s3 = tmm.get_tensormap_ptr(mTmaps[(bidx, 4, None)].iterator)
        tm_a4 = tmm.get_tensormap_ptr(mTmaps[(bidx, 5, None)].iterator)
        tm_s4 = tmm.get_tensormap_ptr(mTmaps[(bidx, 6, None)].iterator)
        dp_a1 = tmm.get_tensormap_ptr(tm_a1, AddressSpace.generic)
        dp_a2 = tmm.get_tensormap_ptr(tm_a2, AddressSpace.generic)
        dp_b2 = tmm.get_tensormap_ptr(tm_b2, AddressSpace.generic)
        dp_a3 = tmm.get_tensormap_ptr(tm_a3, AddressSpace.generic)
        dp_s3 = tmm.get_tensormap_ptr(tm_s3, AddressSpace.generic)
        dp_a4 = tmm.get_tensormap_ptr(tm_a4, AddressSpace.generic)
        dp_s4 = tmm.get_tensormap_ptr(tm_s4, AddressSpace.generic)

        pipeline.sync(barrier_id=1)
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        mlp_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mlp_stage
        )
        mlp_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mlp_stage
        )
        LDONE = (
            NL * CS
        )  # monotonic layer-done counter (NT2 arrivals per layer), own line after the layer regions
        n_tok = mB1.shape[0]
        K1 = self.k1
        I_ = self.inter
        S_K = self.nsplit
        QD = self.hq * D
        KVD = self.hkv * D
        S = KV_SPLITS
        PAGE = self.page
        ROT = self.rot
        HALF = ROT // 2
        qk_scale = cutlass.Float32(LOG2E / (float(D) ** 0.5))
        k_total = cutlass.Float32(K1)
        nt_rs = mSq0.shape[0]
        nt_rs2 = mSq1.shape[0]
        n_rows2 = mRes.shape[0]
        qscale = cutlass.Float32(0.08838834764831845)  # 128^-0.5 (GDN q)
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)
        KDA = (
            1 if self.w16 else 2
        )  # §19: bf16 weights address their real K; fp8 rows pack 2 per bf16 slot ...
        KDG = 1 if self.w16 else 4  # ... and NVFP4 rows 4 per slot
        KB = 2 if self.w16 else 1  # weight bytes per K element (L2 prefetch geometry)

        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            num_kc_blocks = cute.size(tCrA_c, mode=[2])
            num_kc_blocks4 = cute.size(tCrA4_c, mode=[2])
            atom_ldsm_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_ldsm_A, tiled_mma)
            thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
            tCsA_copy_view = thr_copy_A.partition_S(sA)
            tCrA_copy_view = thr_copy_A.retile(tCrA_c)
            tCsA4_copy_view = thr_copy_A.partition_S(sA4)
            tCrA4_copy_view = thr_copy_A.retile(tCrA4_c)
            tCrA_log32 = cute.recast_tensor(tCrA_log8, cutlass.Int32)
            tCrA_log32_4 = cute.recast_tensor(tCrA_log4, cutlass.Int32)
            if cutlass.const_expr(self.mma_attn):
                # QK^T: 4 warps along the keys (8 each), rows = heads; PV: 4 warps along the dims (16 each per 64-wide N tile)
                op_at = cute.nvgpu.warp.MmaF16BF16Op(
                    self.b_dtype, cutlass.Float32, (16, 8, 16)
                )
                mma_qk = cute.make_tiled_mma(op_at, cute.make_layout((1, 4, 1)))
                mma_pv = cute.make_tiled_mma(
                    op_at, cute.make_layout((1, 4, 1)), permutation_mnk=(16, 64, 16)
                )
                thr_qk = mma_qk.get_slice(tidx)
                thr_pv = mma_pv.get_slice(tidx)
                ldsm_x4 = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), self.b_dtype
                )
                ldsm_x2 = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2), self.b_dtype
                )
                ldsm_x4t = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), self.b_dtype
                )
                tcpy_q = cute.make_tiled_copy_A(ldsm_x4, mma_qk)
                tcpy_k = cute.make_tiled_copy_B(ldsm_x2, mma_qk)
                tcpy_p = cute.make_tiled_copy_A(ldsm_x4, mma_pv)
                tcpy_v = cute.make_tiled_copy_B(ldsm_x4t, mma_pv)
                tCsQ = thr_qk.partition_A(sQb)
                tCrQ = mma_qk.make_fragment_A(tCsQ)
                tQs = tcpy_q.get_slice(tidx).partition_S(sQb)
                tQr = tcpy_q.get_slice(tidx).retile(tCrQ)
                tCsK = thr_qk.partition_B(sKp)
                tCrK = mma_qk.make_fragment_B(tCsK[None, None, None, 0])
                tKs = tcpy_k.get_slice(tidx).partition_S(sKp)
                tKr = tcpy_k.get_slice(tidx).retile(tCrK)
                tCsP = thr_pv.partition_A(sPb)
                tCrP = mma_pv.make_fragment_A(tCsP)
                tPs = tcpy_p.get_slice(tidx).partition_S(sPb)
                tPr = tcpy_p.get_slice(tidx).retile(tCrP)
                tCsV = thr_pv.partition_B(sV)
                tCrV = mma_pv.make_fragment_B(tCsV[None, None, None, 0])
                tVs = tcpy_v.get_slice(tidx).partition_S(sV)
                tVr = tcpy_v.get_slice(tidx).retile(tCrV)
                tCcS = thr_qk.partition_C(cute.make_identity_tensor((16, PAGE)))
                acc_s = cute.make_rmem_tensor(tCcS.shape, cutlass.Float32)
                tCcO = thr_pv.partition_C(cute.make_identity_tensor((16, D)))
                acc_pv = cute.make_rmem_tensor(tCcO.shape, cutlass.Float32)
                if cutlass.const_expr(self.R > 1):
                    # §18b: Q tile of R*G rows (row = r*G + h) at ring + 64 KB, processed as MTQ m16 tiles per page
                    MTQ = (self.R * self.group + 15) // 16
                    sQw = cute.make_tensor(
                        cute.recast_ptr(
                            p_ring + 4 * self.page * D,
                            swizzle_=sw_kv,
                            dtype=self.b_dtype,
                        ),
                        cute.make_layout((16 * MTQ, D), stride=(D, 1)),
                    )
                    sStw = cute.make_tensor(
                        sTmp.iterator, cute.make_layout((3, 32), stride=(32, 1))
                    )  # per-row m / l / alpha
                    tQs_w = []
                    tQr_w = []
                    tCrQ_w = []
                    acc_pv_w = []
                    for mt in cutlass.range_constexpr(
                        MTQ
                    ):  # Python-unrolled: builds the per-m-tile partition lists
                        sQ_mt = cute.local_tile(sQw, (16, D), (mt, 0))
                        tCsQ_mt = thr_qk.partition_A(sQ_mt)
                        tCrQ_mt = mma_qk.make_fragment_A(tCsQ_mt)
                        tCrQ_w.append(tCrQ_mt)
                        tQs_w.append(tcpy_q.get_slice(tidx).partition_S(sQ_mt))
                        tQr_w.append(tcpy_q.get_slice(tidx).retile(tCrQ_mt))
                        acc_pv_w.append(
                            cute.make_rmem_tensor(tCcO.shape, cutlass.Float32)
                        )
                # cp.async page staging: 128 threads x 16 B; K tile (4 keys x 256 dims) per pass, V tile (256 dims x 4 keys)
                atom_cp = cute.make_copy_atom(
                    cpasync.CopyG2SOp(), self.b_dtype, num_bits_per_copy=128
                )
                tcp_k = cute.make_tiled_copy_tv(
                    atom_cp,
                    cute.make_layout((4, 32), stride=(32, 1)),
                    cute.make_layout((1, 8)),
                )
                tcp_v = cute.make_tiled_copy_tv(
                    atom_cp,
                    cute.make_layout((32, 4), stride=(1, 32)),
                    cute.make_layout((8, 1)),
                )
                thr_cpk = tcp_k.get_slice(tidx)
                thr_cpv = tcp_v.get_slice(tidx)
                tKsD = thr_cpk.partition_D(sKp)
                tVsD = thr_cpv.partition_D(sV)
                gK_lay = cute.make_layout((PAGE, D), stride=(self.hkv * D, 1))
                gV_lay = cute.make_layout((D, PAGE), stride=(1, self.hkv * D))
            conv16 = cute.make_rmem_tensor((16,), self.b_dtype)
            conv32 = cute.recast_tensor(conv16, cutlass.Int32)
            conv16f = cute.make_rmem_tensor((8, 4), cutlass.Float16)
            deq16 = cute.make_rmem_tensor((8, 4), cutlass.Float16)
            deqbf = cute.make_rmem_tensor((32,), self.b_dtype)
            deq32 = cute.recast_tensor(deqbf, cutlass.Int32)
            if cutlass.const_expr(self.dual_acc):
                # second chain (odd k-blocks): own conversion registers, own A fragment, own accumulator
                conv16f_b = cute.make_rmem_tensor((8, 4), cutlass.Float16)
                deq16_b = cute.make_rmem_tensor((8, 4), cutlass.Float16)
                deqbf_b = cute.make_rmem_tensor((32,), self.b_dtype)
                deq32_b = cute.recast_tensor(deqbf_b, cutlass.Int32)
                tCrA_log4_b = cute.make_rmem_tensor(
                    cute.make_layout(((2, 2, 2), mma_m, 4)), self.b_dtype
                )
                tCrA_log32_4_b = cute.recast_tensor(tCrA_log4_b, cutlass.Int32)
                accumulators_b = cute.make_rmem_tensor(tCcC.shape[:3], self.acc_dtype)
            jhalf = (tidx % 4) // 2
            rres = cute.make_rmem_tensor((cute.size(accumulators),), cutlass.Float32)
            rfin = cute.make_rmem_tensor((cute.size(accumulators),), cutlass.Float32)
            rinv_c = cute.make_rmem_tensor(
                (2, cute.size(accumulators, mode=[2])), cutlass.Float32
            )
            acc_o = cute.make_rmem_tensor((8, 2), cutlass.Float32)
            if cutlass.const_expr(not self.mma_attn):
                acc_v = cute.make_rmem_tensor((8, 8), cutlass.Float32)
                kfrag = cute.make_rmem_tensor((64,), self.b_dtype)
                kfrag_n = cute.make_rmem_tensor((64,), self.b_dtype)
                kf32 = cute.make_rmem_tensor((64,), cutlass.Float32)
                vfrag = cute.make_rmem_tensor((8, 8), self.b_dtype)
                q4 = cute.make_rmem_tensor((4,), cutlass.Float32)
            rinv_l0 = cutlass.Int32(
                -1
            )  # layer whose input-stat 1/rms is cached in sRinv row 0
            rinv_l1 = cutlass.Int32(-1)  # ... mid-stat 1/rms in row 1
            om = cute.make_rmem_tensor(
                (8,), cutlass.Float32
            )  # merge: 8 output dims per lane
            pv8 = cute.make_rmem_tensor((8,), cutlass.Float32)
            st_phase = cutlass.Int32(0)
            pblk = bidx * 2
            FSPLIT0 = (
                CS - FOLD_CNT - NT2
            )  # §19.7 fold counters, top of every layer region: per-tile fc arrivals ...
            FDONE = (
                CS - FOLD_CNT + 8
            )  # ... fc tiles done (layer 0), final-norm arrivals / departures (last layer)
            FNORM = CS - FOLD_CNT + 16
            FNORM2 = CS - FOLD_CNT + 24
            ssq0 = cutlass.Float32(
                0.0
            )  # §19.7 fc phase: per-thread sum of squares of its token's k's (n-tile 0 / 1)
            ssq1 = cutlass.Float32(0.0)
            # names the fc phase shares with the layer loop, declared on every path (the DSL joins the const_expr paths by type)
            work = cutlass.Int32(bidx)
            n_tile = cutlass.Int32(0)
            split = cutlass.Int32(0)
            row0 = cutlass.Int32(0)
            stage = cutlass.Int32(0)
            r = cutlass.Int32(0)
            col = cutlass.Int32(0)
            old = cutlass.Int32(0)
            tokf = cutlass.Int32(0)
            kbase = cutlass.Int32(0)
            acc_total = cutlass.Float32(0.0)
            vf = cutlass.Float32(0.0)
            xvf = cutlass.Float32(0.0)
            wvf = cutlass.Float32(0.0)
            tf0 = cutlass.Int64(0)
            if cutlass.const_expr(self.fold):
                # ============================ §19.7 fc phase: x = Wfc . cat(norm_e(e), norm_h(h)) as ONE split-K GEMM ============================
                # B = cat(e | h) x w_pre (absorbed norms: split s streams K half s, sums x^2 per token on the fly and scales its
                # partial by that half's 1/rms); the last arriver per tile sums the halves -> resid, xw0 = x * wn_in, sq0 partial
                TF = mItab[(0, IT_TF)]
                mCnt0 = cute.make_tensor(mCnt.iterator, cute.make_layout((BIG,)))
                mWpre = cute.make_tensor(
                    cute.make_ptr(
                        self.b_dtype,
                        mPtab[(0, PT_WPRE)],
                        AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((2 * K1,)),
                )
                mWnIn = cute.make_tensor(
                    cute.make_ptr(
                        self.b_dtype,
                        mPtab[(0, PT_WNIN)],
                        AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((K1,)),
                )
                work = cutlass.Int32(bidx)
                while work < TF:
                    tf0 = cutlass.Int64(0)
                    if cutlass.const_expr(self.profile):
                        tf0 = cute.arch.globaltimer()
                    n_tile = work // self.nsplit
                    split = work % self.nsplit
                    row0 = n_tile * self.tile_shape_mnk[0]
                    kbase = (
                        split * K1
                    )  # this half's offset in w_pre (K half = H with 2 splits)
                    accumulators.fill(0.0)
                    ssq0 = cutlass.Float32(0.0)
                    ssq1 = cutlass.Float32(0.0)
                    consumer_state.reset_count()
                    for k_tile in range(0, k_cnt_f, 1, unroll=1):
                        mix_pipeline.consumer_wait(consumer_state)
                        stage = consumer_state.index
                        tCsA_f = tCsA_copy_view[None, None, None, stage]
                        for kc_ in cutlass.range_constexpr(num_kc_blocks):
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_f[None, None, kc_],
                                tCrA_copy_view[None, None, kc_],
                            )
                            cute.autovec_copy(
                                tCsB16[None, None, kc_, stage], tCrB8[None, None, kc_]
                            )
                            for nn in cutlass.range_constexpr(
                                cute.size(tCrB8, mode=[1])
                            ):
                                for e_ in cutlass.range_constexpr(
                                    cute.size(tCrB8, mode=[0])
                                ):
                                    crdf = tCcB[(e_, nn, kc_)]
                                    xvf = tCrB8[(e_, nn, kc_)].to(cutlass.Float32)
                                    if cutlass.const_expr(nn == 0):
                                        ssq0 = ssq0 + xvf * xvf
                                    else:
                                        ssq1 = ssq1 + xvf * xvf
                                    wvf = mWpre[(kbase + k_tile * 128 + crdf[1],)].to(
                                        cutlass.Float32
                                    )
                                    tCrB8[(e_, nn, kc_)] = (xvf * wvf).to(self.b_dtype)
                            cute.gemm(
                                tiled_mma,
                                accumulators,
                                tCrA_c[None, None, kc_],
                                tCrB8[None, None, kc_],
                                accumulators,
                            )
                        mix_pipeline.consumer_release(consumer_state)
                        consumer_state.advance()
                    # 1/rms of this half per token (the 4 lanes of a token hold disjoint k's; every warp sees the same B)
                    for nn in cutlass.range_constexpr(cute.size(tCrB8, mode=[1])):
                        vf = ssq0
                        if cutlass.const_expr(nn == 1):
                            vf = ssq1
                        vf = vf + cute.arch.shuffle_sync_bfly(vf, offset=1)
                        vf = vf + cute.arch.shuffle_sync_bfly(vf, offset=2)
                        if warp_idx == 0:
                            if lane % 4 == 0:
                                tokf = tCcB[(0, nn, 0)][0]
                                if tokf < n_tok:
                                    sRinv[(1, tokf)] = cute.math.rsqrt(
                                        vf / k_total + cutlass.Float32(self.eps)
                                    )
                    self.epilog_sync_barrier.arrive_and_wait()
                    for i in cutlass.range_constexpr(cute.size(accumulators)):
                        crdf = tCcC[i]
                        r = row0 + crdf[0]
                        col = crdf[1]
                        if r < n_rows2 and col < n_tok:
                            mWs[(r, col, split)] = accumulators[i] * sRinv[(1, col)]
                    cute.arch.fence_acq_rel_gpu()
                    self.epilog_sync_barrier.arrive_and_wait()
                    if tidx == 0:
                        old = cute.arch.atomic_add(
                            mCnt0.iterator + (FSPLIT0 + n_tile),
                            cutlass.Int32(1),
                            sem="acq_rel",
                            scope="gpu",
                        )
                        sFlag[0] = old
                    self.epilog_sync_barrier.arrive_and_wait()
                    if sFlag[0] == self.nsplit - 1:
                        cute.arch.fence_acq_rel_gpu()
                        for i in cutlass.range_constexpr(cute.size(accumulators)):
                            crdf = tCcC[i]
                            r = row0 + crdf[0]
                            col = crdf[1]
                            rfin[i] = cutlass.Float32(0.0)
                            if r < n_rows2 and col < n_tok:
                                acc_total = mWs[(r, col, 0)]
                                for sp in cutlass.range_constexpr(self.nsplit - 1):
                                    acc_total = acc_total + mWs[(r, col, sp + 1)]
                                mRes[(r, col, 0)] = acc_total.to(mRes.element_type)
                                mXw0[(r, col, 0)] = (
                                    acc_total * mWnIn[(r,)].to(cutlass.Float32)
                                ).to(mXw0.element_type)
                                rfin[i] = acc_total
                        self._emit_sq(
                            rfin,
                            tCcC,
                            sSq,
                            mSq0,
                            n_tile,
                            n_tok,
                            tidx,
                            lane,
                            warp_idx,
                            mma_m,
                        )
                        self.epilog_sync_barrier.arrive_and_wait()
                        if tidx == 0:
                            mCnt0[FSPLIT0 + n_tile] = cutlass.Int32(0)
                            cute.arch.atomic_add(
                                mCnt0.iterator + FDONE,
                                cutlass.Int32(1),
                                sem="acq_rel",
                                scope="gpu",
                            )
                    if cutlass.const_expr(self.profile):
                        if tidx == 0:
                            self._prec(
                                mProf,
                                mPcnt,
                                pblk,
                                10,
                                tf0,
                                cute.arch.globaltimer(),
                                work,
                            )
                    work = work + gdim

            for L in range(0, NL, 1):
                ta_l = mItab[(L, IT_TA)]
                tail_l = cutlass.Int32(0)
                if cutlass.const_expr(self.tail_split):
                    if ta_l > gdim:
                        if (k_cnt_a % 2) == 0:
                            tail_l = ta_l % gdim
                total = (
                    ta_l
                    + tail_l
                    + mItab[(L, IT_TB0I)]
                    + mItab[(L, IT_TB)]
                    + 2 * NT2 * self.nsplit
                    + TG
                )
                # ---- layer boundary: layer L-1's down reducers are all done (xw0 / sq0 / resid complete)
                if L > 0:
                    tw0 = cutlass.Int64(0)
                    if cutlass.const_expr(self.profile):
                        tw0 = cute.arch.globaltimer()
                    if tidx == 0:
                        need = L * NT2
                        seen = cute.arch.atomic_add(
                            mCnt.iterator + LDONE,
                            cutlass.Int32(0),
                            sem="acquire",
                            scope="gpu",
                        )
                        while seen < need:
                            seen = cute.arch.atomic_add(
                                mCnt.iterator + LDONE,
                                cutlass.Int32(0),
                                sem="acquire",
                                scope="gpu",
                            )
                    self.epilog_sync_barrier.arrive_and_wait()
                    if cutlass.const_expr(self.profile):
                        if tidx == 0:
                            self._prec(
                                mProf,
                                mPcnt,
                                pblk,
                                L * 32 + 6,
                                tw0,
                                cute.arch.globaltimer(),
                                cutlass.Int32(0),
                            )
                if cutlass.const_expr(self.fold):
                    if (
                        L == 0
                    ):  # §19.7: layer 0's resid / xw0 / sq0 come from the fc reducers (every item of layer 0 may read them)
                        if tidx == 0:
                            seen_f = cute.arch.atomic_add(
                                mCnt.iterator + FDONE,
                                cutlass.Int32(0),
                                sem="acquire",
                                scope="gpu",
                            )
                            while seen_f < NT2:
                                seen_f = cute.arch.atomic_add(
                                    mCnt.iterator + FDONE,
                                    cutlass.Int32(0),
                                    sem="acquire",
                                    scope="gpu",
                                )
                        self.epilog_sync_barrier.arrive_and_wait()
                        cute.arch.fence_acq_rel_gpu()

                work = cutlass.Int32(bidx)
                switched = cutlass.Boolean(False)
                while work < total:
                    ti0 = cutlass.Int64(0)
                    if cutlass.const_expr(self.profile):
                        ti0 = cute.arch.globaltimer()
                    KIND = mItab[(L, IT_KIND)]
                    N1 = mItab[(L, IT_N1)]
                    K2 = mItab[(L, IT_K2)]
                    TA = mItab[(L, IT_TA)]
                    TAIL = cutlass.Int32(
                        0
                    )  # §17 item 2: tail tiles split in two K halves (TA > grid, even k-tile count)
                    if cutlass.const_expr(self.tail_split):
                        if gdim < TA:
                            if (k_cnt_a % 2) == 0:
                                TAIL = TA % gdim
                    TAI = (
                        TA + TAIL
                    )  # in-projection ITEM count (TA stays the tile / flag count)
                    TB0 = mItab[(L, IT_TB0)]
                    TB0I = mItab[(L, IT_TB0I)]
                    TBKV = mItab[
                        (L, IT_TBKV)
                    ]  # §18: kv-write items (attention) = sequences x kv heads
                    TB = mItab[(L, IT_TB)]
                    KT0 = mItab[(L, IT_KT0)]
                    W = mItab[(L, IT_W)]
                    mMz = cute.make_tensor(
                        mMzF.iterator,
                        cute.make_layout((N1, n_tok, c1), stride=(c1, N1, c0)),
                    )
                    mAttn = cute.make_tensor(
                        mAttnF.iterator,
                        cute.make_layout((K2, n_tok, c1), stride=(c1, K2, c0)),
                    )
                    if cutlass.const_expr(self.R > 1):
                        # §18: per-layer stash rows (m = R*i + r) of conv inputs / outputs [rows, W] and gate inputs [rows, Hv]
                        mStX = cute.make_tensor(
                            cute.make_ptr(
                                self.b_dtype,
                                mPtab[(L, PT_STX)],
                                AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout((BIG,)),
                        )
                        mStY = cute.make_tensor(
                            cute.make_ptr(
                                self.b_dtype,
                                mPtab[(L, PT_STY)],
                                AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout((BIG,)),
                        )
                        mStA = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float32,
                                mPtab[(L, PT_STA)],
                                AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout((BIG,)),
                        )
                        mStB = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float32,
                                mPtab[(L, PT_STB)],
                                AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout((BIG,)),
                        )
                    mCntL = cute.make_tensor(
                        mCnt.iterator + L * CS, cute.make_layout((BIG,))
                    )
                    mRs = mSq0
                    mSqMid = mSq1
                    mSqOut = mSq0
                    mXwIn = mXw0
                    mXwMid = mXw1
                    mXwOut = mXw0
                    TC = NT2 * self.nsplit
                    TD = TC
                    TGDN = TAI + TB0I + TB + TC
                    # counter layout inside the layer region (each kind's own)
                    HD = TA  # lever 3 (§14): one heads-done counter per v head (GDN) / kv head (attention)
                    NH = cutlass.Int32(self.hv)
                    if KIND == 1:
                        NH = cutlass.Int32(self.hkv)
                    CDONE = TA + NH
                    SPLIT0 = TA + NH + 1
                    B0_FLAGS = SPLIT0 + NT2
                    GDONE = B0_FLAGS + TB0 + 32
                    KVW = SPLIT0 + NT2
                    MERGE = KVW + TBKV
                    QW = MERGE + TB0  # §16 item 3a: q-prep done flags (attention)
                    if KIND == 1:
                        GDONE = QW + TB0 + 32
                    DSPLIT0 = GDONE + 32
                    GACT = (
                        DSPLIT0 + NT2 + 32
                    )  # S counters: gate_up tiles done per down split
                    DDONE = GACT + 16
                    TAILC = (
                        DDONE + 16
                    )  # §17 item 2: per-tile arrival counters of the tail halves
                    if work >= TGDN:
                        # ============================ MLP block (NVFP4 mainloop) ============================
                        if not switched:
                            tg0 = cutlass.Int64(0)
                            if cutlass.const_expr(self.profile):
                                tg0 = cute.arch.globaltimer()
                            phase_bar.arrive_and_wait()
                            if tidx == 0:
                                self._wait_flag(mCntL, GDONE)
                            self.epilog_sync_barrier.arrive_and_wait()
                            if cutlass.const_expr(self.profile):
                                if tidx == 0:
                                    self._prec(
                                        mProf,
                                        mPcnt,
                                        pblk,
                                        L * 32 + 7,
                                        tg0,
                                        cute.arch.globaltimer(),
                                        work,
                                    )
                            switched = cutlass.Boolean(True)
                        is_g = work < TGDN + TG
                        n_tile = work - TGDN
                        split = cutlass.Int32(0)
                        kcnt = cutlass.Int32(k_cnt_g)
                        if not is_g:
                            jt = work - TGDN - TG
                            n_tile = jt // self.nsplit
                            split = jt % self.nsplit
                            kcnt = k_cnt_d
                        row0 = n_tile * self.tile_shape_mnk[0]
                        accumulators.fill(0.0)
                        if cutlass.const_expr(self.dual_acc):
                            accumulators_b.fill(0.0)
                        if is_g:
                            if rinv_l1 != L:
                                # 1/rms per token: one column per thread (16 tokens x 8-fold redundancy) with the same
                                # sequential row sum as before (bitwise), cached in smem for this CTA's other gate_up tiles
                                tk = tidx % 16
                                if tk < n_tok:
                                    ssc = cutlass.Float32(0.0)
                                    for j in range(0, nt_rs2, 1, unroll=8):
                                        ssc = ssc + mSqMid[(j, tk)]
                                    sRinv[(1, tk)] = cute.math.rsqrt(
                                        ssc / k_total + cutlass.Float32(self.eps)
                                    )
                                self.epilog_sync_barrier.arrive_and_wait()
                                rinv_l1 = L
                            for cc in cutlass.range_constexpr(2):
                                for n in cutlass.range_constexpr(
                                    cute.size(accumulators, mode=[2])
                                ):
                                    tok = tCcC[((cc, 0), 0, n)][1]
                                    rinv_c[(cc, n)] = cutlass.Float32(0.0)
                                    if tok < n_tok:
                                        rinv_c[(cc, n)] = sRinv[(1, tok)]
                        else:
                            for i in cutlass.range_constexpr(cute.size(accumulators)):
                                crd = tCcC[i]
                                r = row0 + crd[0]
                                col = crd[1]
                                rres[i] = cutlass.Float32(0.0)
                                if r < n_rows2 and col < n_tok:
                                    rres[i] = mRes[(r, col, 0)].to(cutlass.Float32)
                        if cutlass.const_expr(self.profile):
                            if tidx == 0:
                                self._prec(
                                    mProf,
                                    mPcnt,
                                    pblk,
                                    L * 32 + 19,
                                    ti0,
                                    cute.arch.globaltimer(),
                                    work,
                                )
                        mlp_consumer_state.reset_count()
                        tw_sum = cutlass.Int64(0)
                        tc_sum = cutlass.Int64(0)
                        tst = cutlass.Int64(0)
                        for _k_tile in range(0, kcnt, 1, unroll=1):
                            if cutlass.const_expr(self.profile):
                                tst = cute.arch.globaltimer()
                            mlp_pipeline.consumer_wait(mlp_consumer_state)
                            if cutlass.const_expr(self.profile):
                                tw1 = cute.arch.globaltimer()
                                tw_sum = tw_sum + (tw1 - tst)
                                tst = tw1
                            stage = mlp_consumer_state.index
                            if cutlass.const_expr(self.w16):
                                # §19: bf16 gate_up / down on stages laid out like the fp8 ring (A ldmatrix view, canonical B), direct MMA
                                tCsA_p = tCsA4_copy_view[None, None, None, stage]
                                for kc_ in cutlass.range_constexpr(num_kc_blocks):
                                    cute.copy(
                                        smem_tiled_copy_A,
                                        tCsA_p[None, None, kc_],
                                        tCrA_copy_view[None, None, kc_],
                                    )
                                    cute.autovec_copy(
                                        tCsB16_4[None, None, kc_, stage],
                                        tCrB8[None, None, kc_],
                                    )
                                    cute.gemm(
                                        tiled_mma,
                                        accumulators,
                                        tCrA_c[None, None, kc_],
                                        tCrB8[None, None, kc_],
                                        accumulators,
                                    )
                            else:
                                tCsA_p = tCsA4_copy_view[None, None, None, stage]
                                tCsB_p = tCsB4[None, None, None, stage]
                                sS_p = sS[None, None, stage]
                                for kc_ in cutlass.range_constexpr(num_kc_blocks4):
                                    cute.copy(
                                        smem_tiled_copy_A,
                                        tCsA_p[None, None, kc_],
                                        tCrA4_copy_view[None, None, kc_],
                                    )
                                    for Lq in cutlass.range_constexpr(4):
                                        cute.autovec_copy(
                                            tCsB_p[None, None, 4 * kc_ + Lq],
                                            tCrB4[None, None, 4 * kc_ + Lq],
                                        )
                                    # §17 item 1: odd k-blocks run on a second, independent register chain and accumulator
                                    c16 = conv16f
                                    d16 = deq16
                                    d32 = deq32
                                    lg32 = tCrA_log32_4
                                    lg = tCrA_log4
                                    accs = accumulators
                                    if cutlass.const_expr(
                                        self.dual_acc and kc_ % 2 == 1
                                    ):
                                        c16 = conv16f_b
                                        d16 = deq16_b
                                        d32 = deq32_b
                                        lg32 = tCrA_log32_4_b
                                        lg = tCrA_log4_b
                                        accs = accumulators_b
                                    for m in cutlass.range_constexpr(mma_m):
                                        v8 = tCrA4_c[(None, m, kc_)].load()
                                        vf4 = v8.bitcast(cutlass.Float4E2M1FN)
                                        c16.store(vf4.to(cutlass.Float16))
                                        for l1 in cutlass.range_constexpr(2):
                                            for rh in cutlass.range_constexpr(2):
                                                row = tCcC[((0, rh), m, 0)][0]
                                                sv = sS_p[
                                                    (row, 4 * kc_ + 2 * l1 + jhalf)
                                                ].to(cutlass.Float16)
                                                ch = rh + 2 * l1
                                                d16[(None, ch)].store(
                                                    c16[(None, ch)].load() * sv
                                                )
                                        d16b = d16.load().to(self.b_dtype)
                                        if cutlass.const_expr(
                                            self.dual_acc and kc_ % 2 == 1
                                        ):
                                            deqbf_b.store(d16b)
                                        else:
                                            deqbf.store(d16b)
                                        for l1 in cutlass.range_constexpr(2):
                                            for l0 in cutlass.range_constexpr(2):
                                                for rh in cutlass.range_constexpr(2):
                                                    for kh_ in cutlass.range_constexpr(
                                                        2
                                                    ):
                                                        lg32[
                                                            (
                                                                (0, rh, kh_),
                                                                m,
                                                                l0 + 2 * l1,
                                                            )
                                                        ] = d32[
                                                            kh_
                                                            + 2 * l0
                                                            + 4 * rh
                                                            + 8 * l1
                                                        ]
                                    for Lq in cutlass.range_constexpr(4):
                                        cute.gemm(
                                            tiled_mma,
                                            accs,
                                            lg[None, None, Lq],
                                            tCrB4[None, None, 4 * kc_ + Lq],
                                            accs,
                                        )
                            mlp_pipeline.consumer_release(mlp_consumer_state)
                            mlp_consumer_state.advance()
                            if cutlass.const_expr(self.profile):
                                tc_sum = tc_sum + (cute.arch.globaltimer() - tst)
                        if cutlass.const_expr(self.profile):
                            if tidx == 0:
                                tnow = cute.arch.globaltimer()
                                self._prec(
                                    mProf,
                                    mPcnt,
                                    pblk,
                                    L * 32 + 8,
                                    tnow - tw_sum,
                                    tnow,
                                    work,
                                )  # mlp.wait (as a span of its length)
                                self._prec(
                                    mProf,
                                    mPcnt,
                                    pblk,
                                    L * 32 + 9,
                                    tnow - tc_sum,
                                    tnow,
                                    work,
                                )  # mlp.compute
                        if cutlass.const_expr(self.dual_acc):
                            for i in cutlass.range_constexpr(cute.size(accumulators)):
                                accumulators[i] = accumulators[i] + accumulators_b[i]
                        if is_g:
                            for cc in cutlass.range_constexpr(2):
                                for m in cutlass.range_constexpr(mma_m):
                                    for n in cutlass.range_constexpr(
                                        cute.size(accumulators, mode=[2])
                                    ):
                                        crd = tCcC[((cc, 0), m, n)]
                                        r = row0 + crd[0]
                                        col = crd[1]
                                        gg = (
                                            accumulators[((cc, 0), m, n)]
                                            * mFtab[(L, 5)]
                                            * rinv_c[(cc, n)]
                                        )
                                        uu = (
                                            accumulators[((cc, 1), m, n)]
                                            * mFtab[(L, 6)]
                                            * rinv_c[(cc, n)]
                                        )
                                        y = (
                                            gg
                                            / (
                                                cutlass.Float32(1.0)
                                                + cute.math.exp(-gg)
                                            )
                                            * uu
                                        )
                                        orow = (r // 16) * 8 + (r % 16)
                                        if col < n_tok:
                                            mAct[(orow, col, 0)] = y.to(
                                                mAct.element_type
                                            )
                            cute.arch.fence_acq_rel_gpu()
                            self.epilog_sync_barrier.arrive_and_wait()
                            if tidx == 0:
                                cute.arch.atomic_add(
                                    mCntL.iterator
                                    + (GACT + n_tile // (TG // self.nsplit)),
                                    cutlass.Int32(1),
                                    sem="acq_rel",
                                    scope="gpu",
                                )
                        else:
                            for i in cutlass.range_constexpr(cute.size(accumulators)):
                                crd = tCcC[i]
                                r = row0 + crd[0]
                                col = crd[1]
                                if r < n_rows2 and col < n_tok:
                                    mWs[(r, col, split)] = (
                                        accumulators[i] * mFtab[(L, 7)]
                                    )
                            cute.arch.fence_acq_rel_gpu()
                            self.epilog_sync_barrier.arrive_and_wait()
                            if tidx == 0:
                                old = cute.arch.atomic_add(
                                    mCntL.iterator + (DSPLIT0 + n_tile),
                                    cutlass.Int32(1),
                                    sem="acq_rel",
                                    scope="gpu",
                                )
                                sFlag[0] = old
                            self.epilog_sync_barrier.arrive_and_wait()
                            if sFlag[0] == self.nsplit - 1:
                                cute.arch.fence_acq_rel_gpu()
                                mWnOut = cute.make_tensor(
                                    cute.make_ptr(
                                        self.b_dtype,
                                        mPtab[(L, PT_WNOUT)],
                                        AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout((K1,)),
                                )
                                for i in cutlass.range_constexpr(
                                    cute.size(accumulators)
                                ):
                                    crd = tCcC[i]
                                    r = row0 + crd[0]
                                    col = crd[1]
                                    rfin[i] = cutlass.Float32(0.0)
                                    if r < n_rows2 and col < n_tok:
                                        acc_total = rres[i]
                                        for sp in cutlass.range_constexpr(self.nsplit):
                                            acc_total = acc_total + mWs[(r, col, sp)]
                                        mRes[(r, col, 0)] = acc_total.to(
                                            mRes.element_type
                                        )
                                        mXwOut[(r, col, 0)] = (
                                            acc_total * mWnOut[(r,)].to(cutlass.Float32)
                                        ).to(mXwOut.element_type)
                                        rfin[i] = acc_total
                                self._emit_sq(
                                    rfin,
                                    tCcC,
                                    sSq,
                                    mSqOut,
                                    n_tile,
                                    n_tok,
                                    tidx,
                                    lane,
                                    warp_idx,
                                    mma_m,
                                )
                                self.epilog_sync_barrier.arrive_and_wait()
                                if tidx == 0:
                                    mCntL[DSPLIT0 + n_tile] = cutlass.Int32(0)
                                    # layer-done arrival BEFORE the per-layer done count (the last DDONE arriver of the
                                    # last layer then knows every LDONE arrival landed and re-arms it)
                                    cute.arch.atomic_add(
                                        mCnt.iterator + LDONE,
                                        cutlass.Int32(1),
                                        sem="acq_rel",
                                        scope="gpu",
                                    )
                                    done = cute.arch.atomic_add(
                                        mCntL.iterator + DDONE,
                                        cutlass.Int32(1),
                                        sem="acq_rel",
                                        scope="gpu",
                                    )
                                    if done == NT2 - 1:
                                        for i in cutlass.range_constexpr(self.nsplit):
                                            mCntL[GACT + i] = cutlass.Int32(0)
                                        mCntL[DDONE] = cutlass.Int32(0)
                                        mCntL[GDONE] = cutlass.Int32(0)
                                        if L == NL - 1:
                                            mCnt[LDONE] = cutlass.Int32(0)
                                if cutlass.const_expr(self.fold):
                                    if L == NL - 1:
                                        # §19.7 final norm: once every tile's row statistic has landed, normalize this tile's columns
                                        # of the residual (the down items are at most one per CTA: the host checks NT2 * S <= grid)
                                        tn0 = cutlass.Int64(0)
                                        if cutlass.const_expr(self.profile):
                                            tn0 = cute.arch.globaltimer()
                                        if tidx == 0:
                                            cute.arch.atomic_add(
                                                mCntL.iterator + FNORM,
                                                cutlass.Int32(1),
                                                sem="acq_rel",
                                                scope="gpu",
                                            )
                                            seen_n = cute.arch.atomic_add(
                                                mCntL.iterator + FNORM,
                                                cutlass.Int32(0),
                                                sem="acquire",
                                                scope="gpu",
                                            )
                                            while seen_n < NT2:
                                                seen_n = cute.arch.atomic_add(
                                                    mCntL.iterator + FNORM,
                                                    cutlass.Int32(0),
                                                    sem="acquire",
                                                    scope="gpu",
                                                )
                                        self.epilog_sync_barrier.arrive_and_wait()
                                        cute.arch.fence_acq_rel_gpu()
                                        tk = tidx % 16
                                        if tk < n_tok:
                                            ssc = cutlass.Float32(0.0)
                                            for j in range(0, nt_rs, 1, unroll=8):
                                                ssc = ssc + mSqOut[(j, tk)]
                                            sRinv[(0, tk)] = cute.math.rsqrt(
                                                ssc / k_total
                                                + cutlass.Float32(self.eps)
                                            )
                                        self.epilog_sync_barrier.arrive_and_wait()
                                        for i in cutlass.range_constexpr(
                                            cute.size(accumulators)
                                        ):
                                            crd = tCcC[i]
                                            r = row0 + crd[0]
                                            col = crd[1]
                                            if r < n_rows2 and col < n_tok:
                                                xb = (
                                                    rfin[i]
                                                    .to(self.b_dtype)
                                                    .to(cutlass.Float32)
                                                )  # the bf16 residual the stock norm reads
                                                mOut[(r, col, 0)] = (
                                                    (xb * sRinv[(0, col)])
                                                    * mWnOut[(r,)].to(cutlass.Float32)
                                                ).to(mOut.element_type)
                                        cute.arch.fence_acq_rel_gpu()
                                        self.epilog_sync_barrier.arrive_and_wait()
                                        if tidx == 0:
                                            d2 = cute.arch.atomic_add(
                                                mCntL.iterator + FNORM2,
                                                cutlass.Int32(1),
                                                sem="acq_rel",
                                                scope="gpu",
                                            )
                                            if (
                                                d2 == NT2 - 1
                                            ):  # every tile is normalized: re-arm the fold counters (fc's FDONE lives in region 0)
                                                mCntL[FNORM] = cutlass.Int32(0)
                                                mCntL[FNORM2] = cutlass.Int32(0)
                                                mCnt[FDONE] = cutlass.Int32(0)
                                        rinv_l0 = cutlass.Int32(
                                            -1
                                        )  # sRinv row 0 was overwritten (all threads)
                                        if cutlass.const_expr(self.profile):
                                            if tidx == 0:
                                                self._prec(
                                                    mProf,
                                                    mPcnt,
                                                    pblk,
                                                    L * 32 + 11,
                                                    tn0,
                                                    cute.arch.globaltimer(),
                                                    work,
                                                )
                    else:
                        is_a = work < TAI
                        is_b0 = (work >= TAI) and (work < TAI + TB0I)
                        is_b = (work >= TAI + TB0I) and (work < TAI + TB0I + TB)
                        is_core = is_b0 or is_b
                        if not is_core:
                            # ============================ mixer GEMM phases (fp8): shared by both kinds ============================
                            n_tile = (work + KT0) % TA
                            split = cutlass.Int32(0)
                            kcnt = cutlass.Int32(k_cnt_a)
                            a_half = cutlass.Int32(
                                -1
                            )  # §17 item 2: -1 = whole tile, 0/1 = K half of a tail tile
                            if is_a:
                                if work >= TA - TAIL:
                                    ja = work - (TA - TAIL)
                                    n_tile = (TA - TAIL + ja // 2 + KT0) % TA
                                    a_half = ja % 2
                                    kcnt = k_cnt_a // 2
                            if not is_a:
                                jt = work - TAI - TB0I - TB
                                n_tile = jt // self.nsplit
                                split = jt % self.nsplit
                                kcnt = mItab[(L, IT_KCC)]
                            row0 = n_tile * self.tile_shape_mnk[0]
                            accumulators.fill(0.0)
                            if is_a:
                                if rinv_l0 != L:
                                    tk = tidx % 16
                                    if tk < n_tok:
                                        ssc = cutlass.Float32(0.0)
                                        for j in range(0, nt_rs, 1, unroll=8):
                                            ssc = ssc + mRs[(j, tk)]
                                        sRinv[(0, tk)] = cute.math.rsqrt(
                                            ssc / k_total + cutlass.Float32(self.eps)
                                        )
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    rinv_l0 = L
                                for cc in cutlass.range_constexpr(2):
                                    for n in cutlass.range_constexpr(
                                        cute.size(accumulators, mode=[2])
                                    ):
                                        tok = tCcC[((cc, 0), 0, n)][1]
                                        rinv_c[(cc, n)] = cutlass.Float32(0.0)
                                        if tok < n_tok:
                                            rinv_c[(cc, n)] = sRinv[(0, tok)]
                            else:
                                for i in cutlass.range_constexpr(
                                    cute.size(accumulators)
                                ):
                                    crd = tCcC[i]
                                    r = row0 + crd[0]
                                    col = crd[1]
                                    rres[i] = cutlass.Float32(0.0)
                                    if r < n_rows2 and col < n_tok:
                                        rres[i] = mRes[(r, col, 0)].to(cutlass.Float32)
                            if cutlass.const_expr(self.profile):
                                if tidx == 0:
                                    self._prec(
                                        mProf,
                                        mPcnt,
                                        pblk,
                                        L * 32 + 27,
                                        ti0,
                                        cute.arch.globaltimer(),
                                        work,
                                    )
                            consumer_state.reset_count()
                            for _k_tile in range(0, kcnt, 1, unroll=1):
                                mix_pipeline.consumer_wait(consumer_state)
                                stage = consumer_state.index
                                tCsA_p = tCsA_copy_view[None, None, None, stage]
                                tCsB_p = tCsB8[None, None, None, stage]
                                for kc_ in cutlass.range_constexpr(num_kc_blocks):
                                    cute.copy(
                                        smem_tiled_copy_A,
                                        tCsA_p[None, None, kc_],
                                        tCrA_copy_view[None, None, kc_],
                                    )
                                    if cutlass.const_expr(self.w16):
                                        # §19: bf16 weights — the ldmatrix fragment IS the MMA A fragment (16 K per block)
                                        cute.autovec_copy(
                                            tCsB16[None, None, kc_, stage],
                                            tCrB8[None, None, kc_],
                                        )
                                        cute.gemm(
                                            tiled_mma,
                                            accumulators,
                                            tCrA_c[None, None, kc_],
                                            tCrB8[None, None, kc_],
                                            accumulators,
                                        )
                                    else:
                                        cute.autovec_copy(
                                            tCsB_p[None, None, 2 * kc_],
                                            tCrB8[None, None, 2 * kc_],
                                        )
                                        cute.autovec_copy(
                                            tCsB_p[None, None, 2 * kc_ + 1],
                                            tCrB8[None, None, 2 * kc_ + 1],
                                        )
                                        for m in cutlass.range_constexpr(mma_m):
                                            v8 = tCrA_c[(None, m, kc_)].load()
                                            vf8 = v8.bitcast(cutlass.Float8E4M3FN)
                                            conv16.store(vf8.to(self.b_dtype))
                                            for khc in cutlass.range_constexpr(2):
                                                for rh in cutlass.range_constexpr(2):
                                                    for e in cutlass.range_constexpr(2):
                                                        tCrA_log32[
                                                            ((0, rh, e), m, khc)
                                                        ] = conv32[e + 2 * rh + 4 * khc]
                                        for half in cutlass.range_constexpr(2):
                                            cute.gemm(
                                                tiled_mma,
                                                accumulators,
                                                tCrA_log8[None, None, half],
                                                tCrB8[None, None, 2 * kc_ + half],
                                                accumulators,
                                            )
                                mix_pipeline.consumer_release(consumer_state)
                                consumer_state.advance()
                            if is_a:
                                do_epi = cutlass.Boolean(True)
                                if a_half >= 0:
                                    # §17 item 2: publish this K half; the last arriver sums both halves and owns the epilogue
                                    for i in cutlass.range_constexpr(
                                        cute.size(accumulators)
                                    ):
                                        crd = tCcC[i]
                                        r = row0 + crd[0]
                                        col = crd[1]
                                        if col < n_tok:
                                            mWsA[(r, col, a_half)] = accumulators[i]
                                    cute.arch.fence_acq_rel_gpu()
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if tidx == 0:
                                        old = cute.arch.atomic_add(
                                            mCntL.iterator + (TAILC + n_tile),
                                            cutlass.Int32(1),
                                            sem="acq_rel",
                                            scope="gpu",
                                        )
                                        sFlag[0] = old
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if sFlag[0] == 1:
                                        cute.arch.fence_acq_rel_gpu()
                                        for i in cutlass.range_constexpr(
                                            cute.size(accumulators)
                                        ):
                                            crd = tCcC[i]
                                            r = row0 + crd[0]
                                            col = crd[1]
                                            if col < n_tok:
                                                accumulators[i] = (
                                                    mWsA[(r, col, 0)]
                                                    + mWsA[(r, col, 1)]
                                                )
                                        if tidx == 0:
                                            mCntL[TAILC + n_tile] = cutlass.Int32(0)
                                    else:
                                        do_epi = cutlass.Boolean(False)
                                if do_epi:
                                    alpha_v = mFtab[(L, 0)]
                                    if row0 >= mItab[(L, IT_B1)]:
                                        alpha_v = mFtab[(L, 1)]
                                    if row0 >= mItab[(L, IT_B2)]:
                                        alpha_v = mFtab[(L, 2)]
                                    if row0 >= mItab[(L, IT_B3)]:
                                        alpha_v = mFtab[(L, 3)]
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
                                                            accumulators[
                                                                ((cc, rh), m, n)
                                                            ]
                                                            * alpha_v
                                                            * rinv_c[(cc, n)]
                                                        )
                                                        mMz[(r, col, 0)] = v.to(
                                                            mMz.element_type
                                                        )
                                    cute.arch.fence_acq_rel_gpu()
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if tidx == 0:
                                        cute.arch.atomic_add(
                                            mCntL.iterator + n_tile,
                                            cutlass.Int32(1),
                                            sem="acq_rel",
                                            scope="gpu",
                                        )
                            else:
                                for i in cutlass.range_constexpr(
                                    cute.size(accumulators)
                                ):
                                    crd = tCcC[i]
                                    r = row0 + crd[0]
                                    col = crd[1]
                                    if r < n_rows2 and col < n_tok:
                                        mWs[(r, col, split)] = (
                                            accumulators[i] * mFtab[(L, 4)]
                                        )
                                cute.arch.fence_acq_rel_gpu()
                                self.epilog_sync_barrier.arrive_and_wait()
                                if tidx == 0:
                                    old = cute.arch.atomic_add(
                                        mCntL.iterator + (SPLIT0 + n_tile),
                                        cutlass.Int32(1),
                                        sem="acq_rel",
                                        scope="gpu",
                                    )
                                    sFlag[0] = old
                                self.epilog_sync_barrier.arrive_and_wait()
                                if sFlag[0] == self.nsplit - 1:
                                    cute.arch.fence_acq_rel_gpu()
                                    mWnMid = cute.make_tensor(
                                        cute.make_ptr(
                                            self.b_dtype,
                                            mPtab[(L, PT_WNMID)],
                                            AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        cute.make_layout((K1,)),
                                    )
                                    for i in cutlass.range_constexpr(
                                        cute.size(accumulators)
                                    ):
                                        crd = tCcC[i]
                                        r = row0 + crd[0]
                                        col = crd[1]
                                        rfin[i] = cutlass.Float32(0.0)
                                        if r < n_rows2 and col < n_tok:
                                            acc_total = rres[i]
                                            for sp in cutlass.range_constexpr(
                                                self.nsplit
                                            ):
                                                acc_total = (
                                                    acc_total + mWs[(r, col, sp)]
                                                )
                                            mRes[(r, col, 0)] = acc_total.to(
                                                mRes.element_type
                                            )
                                            mXwMid[(r, col, 0)] = (
                                                acc_total
                                                * mWnMid[(r,)].to(cutlass.Float32)
                                            ).to(mXwMid.element_type)
                                            rfin[i] = acc_total
                                    self._emit_sq(
                                        rfin,
                                        tCcC,
                                        sSq,
                                        mSqMid,
                                        n_tile,
                                        n_tok,
                                        tidx,
                                        lane,
                                        warp_idx,
                                        mma_m,
                                    )
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if tidx == 0:
                                        mCntL[SPLIT0 + n_tile] = cutlass.Int32(0)
                                        done = cute.arch.atomic_add(
                                            mCntL.iterator + CDONE,
                                            cutlass.Int32(1),
                                            sem="acq_rel",
                                            scope="gpu",
                                        )
                                        if done == NT2 - 1:
                                            mCntL[CDONE] = cutlass.Int32(0)
                                            for i in range(0, TA, 1):
                                                mCntL[i] = cutlass.Int32(0)
                                            if KIND == 0:
                                                for i in cutlass.range_constexpr(
                                                    self.hv
                                                ):
                                                    mCntL[HD + i] = cutlass.Int32(0)
                                                for i in range(0, TB0, 1):
                                                    mCntL[B0_FLAGS + i] = cutlass.Int32(
                                                        0
                                                    )
                                            else:
                                                for i in cutlass.range_constexpr(
                                                    self.hkv
                                                ):
                                                    mCntL[HD + i] = cutlass.Int32(0)
                                                for i in range(0, TBKV, 1):
                                                    mCntL[KVW + i] = cutlass.Int32(0)
                                                for i in range(0, TB0, 1):
                                                    mCntL[MERGE + i] = cutlass.Int32(0)
                                                    mCntL[QW + i] = cutlass.Int32(0)
                                            cute.arch.fence_acq_rel_gpu()
                                            cute.arch.atomic_add(
                                                mCntL.iterator + GDONE,
                                                cutlass.Int32(1),
                                                sem="acq_rel",
                                                scope="gpu",
                                            )
                        elif KIND == 0:
                            # ============================ GDN core ============================
                            mConvW = cute.make_tensor(
                                cute.make_ptr(
                                    self.b_dtype,
                                    mPtab[(L, PT_CONVW)],
                                    AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((W, 4), stride=(4, 1)),
                            )
                            mConv = cute.make_tensor(
                                cute.make_ptr(
                                    self.b_dtype,
                                    mPtab[(L, PT_CONV)],
                                    AddressSpace.gmem,
                                    assumed_align=2,
                                ),
                                cute.make_layout((BIG, W, 3), stride=(3 * W, 3, 1)),
                            )
                            if is_b0:
                                if cutlass.const_expr(self.R > 1):
                                    # §18 verify form: q|k conv of the R rows of one (sequence, key head) on register tap windows
                                    it = work - TAI
                                    sq_ = it // self.hk
                                    kh = it % self.hk
                                    slot = mSlots[sq_]
                                    q0 = kh * self.dk
                                    kk0 = KD + kh * self.dk
                                    if tidx < 4:
                                        base = q0
                                        if tidx >= 2:
                                            base = kk0
                                        self._wait_flag(
                                            mCntL,
                                            base // self.tile_shape_mnk[0] + tidx % 2,
                                        )
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    cq = q0 + tidx
                                    ck = kk0 + tidx
                                    tq0 = mConv[(slot, cq, 0)].to(cutlass.Float32)
                                    tq1 = mConv[(slot, cq, 1)].to(cutlass.Float32)
                                    tq2 = mConv[(slot, cq, 2)].to(cutlass.Float32)
                                    tk0 = mConv[(slot, ck, 0)].to(cutlass.Float32)
                                    tk1 = mConv[(slot, ck, 1)].to(cutlass.Float32)
                                    tk2 = mConv[(slot, ck, 2)].to(cutlass.Float32)
                                    wq0 = mConvW[(cq, 0)].to(cutlass.Float32)
                                    wq1 = mConvW[(cq, 1)].to(cutlass.Float32)
                                    wq2 = mConvW[(cq, 2)].to(cutlass.Float32)
                                    wq3 = mConvW[(cq, 3)].to(cutlass.Float32)
                                    wk0 = mConvW[(ck, 0)].to(cutlass.Float32)
                                    wk1 = mConvW[(ck, 1)].to(cutlass.Float32)
                                    wk2 = mConvW[(ck, 2)].to(cutlass.Float32)
                                    wk3 = mConvW[(ck, 3)].to(cutlass.Float32)
                                    for r_ in cutlass.range_constexpr(self.R):
                                        m = sq_ * self.R + r_
                                        xq = mMz[(cq, m, 0)]
                                        xk = mMz[(ck, m, 0)]
                                        aq = (
                                            xq.to(cutlass.Float32) * wq3
                                            + tq0 * wq0
                                            + tq1 * wq1
                                            + tq2 * wq2
                                        )
                                        ak = (
                                            xk.to(cutlass.Float32) * wk3
                                            + tk0 * wk0
                                            + tk1 * wk1
                                            + tk2 * wk2
                                        )
                                        tq0 = tq1
                                        tq1 = tq2
                                        tq2 = xq.to(cutlass.Float32)
                                        tk0 = tk1
                                        tk1 = tk2
                                        tk2 = xk.to(cutlass.Float32)
                                        yq = aq / (
                                            cutlass.Float32(1.0) + cute.math.exp(-aq)
                                        )
                                        yk = ak / (
                                            cutlass.Float32(1.0) + cute.math.exp(-ak)
                                        )
                                        mStX[(m * W + cq,)] = xq
                                        mStX[(m * W + ck,)] = xk
                                        mStY[(m * W + cq,)] = yq.to(mStY.element_type)
                                        mStY[(m * W + ck,)] = yk.to(mStY.element_type)
                                        ssq = self._block_sum(
                                            yq * yq, sRed1, warp_idx, lane
                                        )
                                        ssk = self._block_sum(
                                            yk * yk, sRed1, warp_idx, lane
                                        )
                                        mQK[((m * self.hk + kh) * 256 + tidx,)] = (
                                            yq
                                            * cute.math.rsqrt(
                                                ssq + cutlass.Float32(1e-6)
                                            )
                                            * qscale
                                        )
                                        mQK[
                                            ((m * self.hk + kh) * 256 + 128 + tidx,)
                                        ] = yk * cute.math.rsqrt(
                                            ssk + cutlass.Float32(1e-6)
                                        )
                                    cute.arch.fence_acq_rel_gpu()
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if tidx == 0:
                                        cute.arch.atomic_add(
                                            mCntL.iterator + (B0_FLAGS + it),
                                            cutlass.Int32(1),
                                            sem="acq_rel",
                                            scope="gpu",
                                        )
                                else:
                                    it = work - TAI
                                    m = it // self.hk
                                    kh = it % self.hk
                                    slot = mSlots[m]
                                    q0 = kh * self.dk
                                    kk0 = KD + kh * self.dk
                                    if tidx < 4:
                                        base = q0
                                        if tidx >= 2:
                                            base = kk0
                                        self._wait_flag(
                                            mCntL,
                                            base // self.tile_shape_mnk[0] + tidx % 2,
                                        )
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    yq = self._conv_channel(
                                        mMz, mConvW, mConv, slot, q0 + tidx, m
                                    )
                                    yk = self._conv_channel(
                                        mMz, mConvW, mConv, slot, kk0 + tidx, m
                                    )
                                    ssq = self._block_sum(
                                        yq * yq, sRed1, warp_idx, lane
                                    )
                                    ssk = self._block_sum(
                                        yk * yk, sRed1, warp_idx, lane
                                    )
                                    mQK[(it * 256 + tidx,)] = (
                                        yq
                                        * cute.math.rsqrt(ssq + cutlass.Float32(1e-6))
                                        * qscale
                                    )
                                    mQK[(it * 256 + 128 + tidx,)] = (
                                        yk
                                        * cute.math.rsqrt(ssk + cutlass.Float32(1e-6))
                                    )
                                    cute.arch.fence_acq_rel_gpu()
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if tidx == 0:
                                        cute.arch.atomic_add(
                                            mCntL.iterator + (B0_FLAGS + it),
                                            cutlass.Int32(1),
                                            sem="acq_rel",
                                            scope="gpu",
                                        )
                            else:
                                if cutlass.const_expr(self.R > 1):
                                    # §18 verify form: the R rows of one (sequence, value head) through the delta rule in order,
                                    # on the smem-resident state loaded once; the pool is NOT written back (apply_accept replays
                                    # the accepted prefix from the stash); conv taps live in registers
                                    it = work - TAI - TB0
                                    sq_ = it // self.hv
                                    h = it % self.hv
                                    kh = h // self.rep
                                    slot = mSlots[sq_]
                                    mSsm = cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.Float32,
                                            mPtab[(L, PT_SSM)],
                                            AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        cute.make_layout((BIG,)),
                                    )
                                    mWba = cute.make_tensor(
                                        cute.make_ptr(
                                            self.b_dtype,
                                            mPtab[(L, PT_WBA)],
                                            AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        cute.make_layout((BIG,)),
                                    )
                                    mAlog = cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.Float32,
                                            mPtab[(L, PT_ALOG)],
                                            AddressSpace.gmem,
                                            assumed_align=4,
                                        ),
                                        cute.make_layout((self.hv,)),
                                    )
                                    mDtb = cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.Float32,
                                            mPtab[(L, PT_DTB)],
                                            AddressSpace.gmem,
                                            assumed_align=4,
                                        ),
                                        cute.make_layout((self.hv,)),
                                    )
                                    mNormW = cute.make_tensor(
                                        cute.make_ptr(
                                            self.b_dtype,
                                            mPtab[(L, PT_NORMW)],
                                            AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        cute.make_layout((128,)),
                                    )
                                    vc0 = 2 * KD + h * self.dv
                                    zc0 = W + h * self.dv
                                    head_base = (slot * self.hv + h) * 16384
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
                                    gwb = cute.make_tensor(
                                        mWba.iterator
                                        + cute.assume(
                                            h * self.k1 + tidx * self.kchunk, divby=8
                                        ),
                                        cute.make_layout((self.kchunk,)),
                                    )
                                    gwa = cute.make_tensor(
                                        mWba.iterator
                                        + cute.assume(
                                            (self.hv + h) * self.k1
                                            + tidx * self.kchunk,
                                            divby=8,
                                        ),
                                        cute.make_layout((self.kchunk,)),
                                    )
                                    fwb = cute.make_fragment_like(gwb)
                                    fwa = cute.make_fragment_like(gwa)
                                    cute.autovec_copy(gwb, fwb)
                                    cute.autovec_copy(gwa, fwa)
                                    dtb_h = mDtb[(h,)]
                                    alog_h = mAlog[(h,)]
                                    cv = vc0 + tidx
                                    tv0 = mConv[(slot, cv, 0)].to(cutlass.Float32)
                                    tv1 = mConv[(slot, cv, 1)].to(cutlass.Float32)
                                    tv2 = mConv[(slot, cv, 2)].to(cutlass.Float32)
                                    wv0 = mConvW[(cv, 0)].to(cutlass.Float32)
                                    wv1 = mConvW[(cv, 1)].to(cutlass.Float32)
                                    wv2 = mConvW[(cv, 2)].to(cutlass.Float32)
                                    wv3 = mConvW[(cv, 3)].to(cutlass.Float32)
                                    if tidx < 2:
                                        self._wait_flag(
                                            mCntL, vc0 // self.tile_shape_mnk[0] + tidx
                                        )
                                    if tidx == 2:
                                        self._wait_flag(
                                            mCntL, B0_FLAGS + sq_ * self.hk + kh
                                        )
                                    if tidx >= 3:
                                        if tidx < 5:
                                            self._wait_flag(
                                                mCntL,
                                                zc0 // self.tile_shape_mnk[0]
                                                + tidx
                                                - 3,
                                            )
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    cute.arch.mbarrier_wait(state_bar, st_phase)
                                    st_phase = st_phase ^ 1
                                    # ---- §19.10 WY form: per-row light work first (conv-v, stash, gates), rows' k / q into smem
                                    RR = self.R
                                    sKq = cute.make_tensor(
                                        cute.recast_ptr(
                                            p_ring + 4 * self.page * D,
                                            dtype=cutlass.Float32,
                                        ),
                                        cute.make_layout(
                                            (2, RR, 128), stride=(RR * 128, 128, 1)
                                        ),
                                    )  # ring + 64 KB (the Q slot)
                                    vv = cute.make_rmem_tensor((RR,), cutlass.Float32)
                                    beta_r = cute.make_rmem_tensor(
                                        (RR,), cutlass.Float32
                                    )
                                    gc_r = cute.make_rmem_tensor((RR,), cutlass.Float32)
                                    gcum = cutlass.Float32(0.0)
                                    for r_ in cutlass.range_constexpr(RR):
                                        m = sq_ * RR + r_
                                        gx = cute.make_tensor(
                                            mXwIn.iterator
                                            + cute.assume(
                                                m * self.k1 + tidx * self.kchunk,
                                                divby=8,
                                            ),
                                            cute.make_layout((self.kchunk,)),
                                        )
                                        fx = cute.make_fragment_like(gx)
                                        cute.autovec_copy(gx, fx)
                                        ssm_ = cutlass.Float32(0.0)
                                        for j in range(0, nt_rs, 1, unroll=8):
                                            ssm_ = ssm_ + mRs[(j, m)]
                                        rinv_m = cute.math.rsqrt(
                                            ssm_ / k_total + cutlass.Float32(self.eps)
                                        )
                                        b0 = (m * self.hk + kh) * 256
                                        sKq[(1, r_, tidx)] = mQK[(b0 + tidx,)]
                                        sKq[(0, r_, tidx)] = mQK[(b0 + 128 + tidx,)]
                                        xv = mMz[(cv, m, 0)]
                                        av = (
                                            xv.to(cutlass.Float32) * wv3
                                            + tv0 * wv0
                                            + tv1 * wv1
                                            + tv2 * wv2
                                        )
                                        tv0 = tv1
                                        tv1 = tv2
                                        tv2 = xv.to(cutlass.Float32)
                                        yv = av / (
                                            cutlass.Float32(1.0) + cute.math.exp(-av)
                                        )
                                        vv[r_] = yv
                                        mStX[(m * W + cv,)] = xv
                                        mStY[(m * W + cv,)] = yv.to(mStY.element_type)
                                        pb = cutlass.Float32(0.0)
                                        pa = cutlass.Float32(0.0)
                                        for j in cutlass.range_constexpr(self.kchunk):
                                            xn = fx[j].to(cutlass.Float32) * rinv_m
                                            pb = pb + xn * fwb[j].to(cutlass.Float32)
                                            pa = pa + xn * fwa[j].to(cutlass.Float32)
                                        bsum = self._block_sum(
                                            pb, sRed1, warp_idx, lane
                                        )
                                        asum = self._block_sum(
                                            pa, sRed1, warp_idx, lane
                                        )
                                        b_bf = bsum.to(cutlass.BFloat16).to(
                                            cutlass.Float32
                                        )
                                        a_bf = asum.to(cutlass.BFloat16).to(
                                            cutlass.Float32
                                        )
                                        if tidx == 0:
                                            mStA[(m * self.hv + h,)] = a_bf
                                            mStB[(m * self.hv + h,)] = b_bf
                                        beta_r[r_] = (
                                            (
                                                cutlass.Float32(1.0)
                                                / (
                                                    cutlass.Float32(1.0)
                                                    + cute.math.exp(-b_bf)
                                                )
                                            )
                                            .to(cutlass.BFloat16)
                                            .to(cutlass.Float32)
                                        )
                                        xg = a_bf + dtb_h
                                        spl = cute.math.log(
                                            cutlass.Float32(1.0) + cute.math.exp(xg)
                                        )
                                        if xg > cutlass.Float32(20.0):
                                            spl = xg
                                        gcum = gcum - cute.math.exp(alog_h) * spl
                                        gc_r[r_] = gcum
                                    self.epilog_sync_barrier.arrive_and_wait()  # k / q rows visible
                                    # ---- one fused pass over the frozen state: this thread's value column against every row's k and q
                                    kvp = cute.make_rmem_tensor((RR,), cutlass.Float32)
                                    oqp = cute.make_rmem_tensor((RR,), cutlass.Float32)
                                    kvp.fill(0.0)
                                    oqp.fill(0.0)
                                    for c8 in range(0, 32, 1, unroll=1):
                                        for i in cutlass.range_constexpr(4):
                                            r0 = c8 * 4 + i
                                            s0 = sStA[(r0, tidx)]
                                            for r_ in cutlass.range_constexpr(RR):
                                                kvp[r_] = (
                                                    kvp[r_] + s0 * sKq[(0, r_, r0)]
                                                )
                                                oqp[r_] = (
                                                    oqp[r_] + s0 * sKq[(1, r_, r0)]
                                                )
                                    # ---- R x R factors: (k_t.k_i) i < t and (q_t.k_i) i <= t, partials over this thread's dk, one batched block reduction
                                    NPAIR = RR * (RR - 1) // 2 + RR * (RR + 1) // 2
                                    dpp = cute.make_rmem_tensor(
                                        (NPAIR,), cutlass.Float32
                                    )
                                    e_ = 0
                                    for t_ in cutlass.range_constexpr(RR):
                                        for i_ in cutlass.range_constexpr(t_):
                                            dpp[e_] = (
                                                sKq[(0, t_, tidx)] * sKq[(0, i_, tidx)]
                                            )
                                            e_ += 1
                                    for t_ in cutlass.range_constexpr(RR):
                                        for i_ in cutlass.range_constexpr(t_ + 1):
                                            dpp[e_] = (
                                                sKq[(1, t_, tidx)] * sKq[(0, i_, tidx)]
                                            )
                                            e_ += 1
                                    for off in [16, 8, 4, 2, 1]:
                                        for e_ in cutlass.range_constexpr(NPAIR):
                                            dpp[e_] = dpp[
                                                e_
                                            ] + cute.arch.shuffle_sync_bfly(
                                                dpp[e_], offset=off
                                            )
                                    sPair = cute.make_tensor(
                                        sTmp.iterator,
                                        cute.make_layout((4, NPAIR), stride=(NPAIR, 1)),
                                    )  # core_tmp scratch: 4 warps x pairs
                                    if lane == 0:
                                        for e_ in cutlass.range_constexpr(NPAIR):
                                            sPair[(warp_idx, e_)] = dpp[e_]
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    for e_ in cutlass.range_constexpr(NPAIR):
                                        dpp[e_] = (sPair[(0, e_)] + sPair[(1, e_)]) + (
                                            sPair[(2, e_)] + sPair[(3, e_)]
                                        )
                                    # ---- the chain (this thread's value column) and the rows' outputs
                                    uu = cute.make_rmem_tensor((RR,), cutlass.Float32)
                                    e_ = 0
                                    eq_ = RR * (RR - 1) // 2
                                    for t_ in cutlass.range_constexpr(RR):
                                        m = sq_ * RR + t_
                                        egt = cute.math.exp(gc_r[t_])
                                        kvm = egt * kvp[t_]
                                        for i_ in cutlass.range_constexpr(t_):
                                            kvm = (
                                                kvm
                                                + cute.math.exp(gc_r[t_] - gc_r[i_])
                                                * dpp[e_]
                                                * uu[i_]
                                            )
                                            e_ += 1
                                        uu[t_] = beta_r[t_] * (vv[t_] - kvm)
                                        ot = egt * oqp[t_]
                                        for i_ in cutlass.range_constexpr(t_ + 1):
                                            ot = (
                                                ot
                                                + cute.math.exp(gc_r[t_] - gc_r[i_])
                                                * dpp[eq_]
                                                * uu[i_]
                                            )
                                            eq_ += 1
                                        zv = mMz[(zc0 + tidx, m, 0)].to(cutlass.Float32)
                                        xf = ot.to(cutlass.BFloat16).to(cutlass.Float32)
                                        ssx = self._block_sum(
                                            xf * xf, sRed1, warp_idx, lane
                                        )
                                        rstd = cute.math.rsqrt(
                                            ssx / cutlass.Float32(128.0)
                                            + cutlass.Float32(self.eps)
                                        )
                                        y_bf = (
                                            (
                                                xf
                                                * rstd
                                                * mNormW[(tidx,)].to(cutlass.Float32)
                                            )
                                            .to(cutlass.BFloat16)
                                            .to(cutlass.Float32)
                                        )
                                        sil = zv / (
                                            cutlass.Float32(1.0) + cute.math.exp(-zv)
                                        )
                                        mAttn[(h * self.dv + tidx, m, 0)] = (
                                            y_bf * sil
                                        ).to(mAttn.element_type)
                                    cute.arch.fence_view_async_shared()
                                    cute.arch.fence_acq_rel_gpu()
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if tidx == 0:
                                        cute.arch.atomic_add(
                                            mCntL.iterator + (HD + h),
                                            cutlass.Int32(1),
                                            sem="acq_rel",
                                            scope="gpu",
                                        )
                                    core_bar.arrive()  # ring smem is free again for this CTA's producer
                                else:
                                    it = work - TAI - TB0
                                    m = it // self.hv
                                    h = it % self.hv
                                    kh = h // self.rep
                                    slot = mSlots[m]
                                    mSsm = cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.Float32,
                                            mPtab[(L, PT_SSM)],
                                            AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        cute.make_layout((BIG,)),
                                    )
                                    mWba = cute.make_tensor(
                                        cute.make_ptr(
                                            self.b_dtype,
                                            mPtab[(L, PT_WBA)],
                                            AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        cute.make_layout((BIG,)),
                                    )
                                    mAlog = cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.Float32,
                                            mPtab[(L, PT_ALOG)],
                                            AddressSpace.gmem,
                                            assumed_align=4,
                                        ),
                                        cute.make_layout((self.hv,)),
                                    )
                                    mDtb = cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.Float32,
                                            mPtab[(L, PT_DTB)],
                                            AddressSpace.gmem,
                                            assumed_align=4,
                                        ),
                                        cute.make_layout((self.hv,)),
                                    )
                                    mNormW = cute.make_tensor(
                                        cute.make_ptr(
                                            self.b_dtype,
                                            mPtab[(L, PT_NORMW)],
                                            AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        cute.make_layout((128,)),
                                    )
                                    vc0 = 2 * KD + h * self.dv
                                    zc0 = W + h * self.dv
                                    ts0 = cutlass.Int64(0)
                                    if cutlass.const_expr(self.profile):
                                        ts0 = cute.arch.globaltimer()
                                    # independent of the in-projection flags: state load, gate-projection fragments, per-head
                                    # scalars — all issued before the flag spin so their latency overlaps the wait (§15)
                                    head_base = (slot * self.hv + h) * 16384
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
                                    gx = cute.make_tensor(
                                        mXwIn.iterator
                                        + cute.assume(
                                            m * self.k1 + tidx * self.kchunk, divby=8
                                        ),
                                        cute.make_layout((self.kchunk,)),
                                    )
                                    gwb = cute.make_tensor(
                                        mWba.iterator
                                        + cute.assume(
                                            h * self.k1 + tidx * self.kchunk, divby=8
                                        ),
                                        cute.make_layout((self.kchunk,)),
                                    )
                                    gwa = cute.make_tensor(
                                        mWba.iterator
                                        + cute.assume(
                                            (self.hv + h) * self.k1
                                            + tidx * self.kchunk,
                                            divby=8,
                                        ),
                                        cute.make_layout((self.kchunk,)),
                                    )
                                    fx = cute.make_fragment_like(gx)
                                    fwb = cute.make_fragment_like(gwb)
                                    fwa = cute.make_fragment_like(gwa)
                                    cute.autovec_copy(gx, fx)
                                    cute.autovec_copy(gwb, fwb)
                                    cute.autovec_copy(gwa, fwa)
                                    dtb_h = mDtb[(h,)]
                                    alog_h = mAlog[(h,)]
                                    ssm_ = cutlass.Float32(0.0)
                                    for j in range(0, nt_rs, 1, unroll=8):
                                        ssm_ = ssm_ + mRs[(j, m)]
                                    rinv_m = cute.math.rsqrt(
                                        ssm_ / k_total + cutlass.Float32(self.eps)
                                    )
                                    if tidx < 2:
                                        self._wait_flag(
                                            mCntL, vc0 // self.tile_shape_mnk[0] + tidx
                                        )
                                    if tidx == 2:
                                        self._wait_flag(
                                            mCntL, B0_FLAGS + m * self.hk + kh
                                        )
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if cutlass.const_expr(self.profile):
                                        ts1 = cute.arch.globaltimer()
                                        if tidx == 0:
                                            self._prec(
                                                mProf,
                                                mPcnt,
                                                pblk,
                                                L * 32 + 20,
                                                ts0,
                                                ts1,
                                                work,
                                            )
                                        ts0 = ts1
                                    b0 = (m * self.hk + kh) * 256
                                    sQg[tidx] = mQK[(b0 + tidx,)]
                                    sK[tidx] = mQK[(b0 + 128 + tidx,)]
                                    yv = self._conv_channel(
                                        mMz, mConvW, mConv, slot, vc0 + tidx, m
                                    )
                                    pb = cutlass.Float32(0.0)
                                    pa = cutlass.Float32(0.0)
                                    for j in cutlass.range_constexpr(self.kchunk):
                                        xn = fx[j].to(cutlass.Float32) * rinv_m
                                        pb = pb + xn * fwb[j].to(cutlass.Float32)
                                        pa = pa + xn * fwa[j].to(cutlass.Float32)
                                    bsum = self._block_sum(pb, sRed1, warp_idx, lane)
                                    asum = self._block_sum(pa, sRed1, warp_idx, lane)
                                    b_bf = bsum.to(cutlass.BFloat16).to(cutlass.Float32)
                                    a_bf = asum.to(cutlass.BFloat16).to(cutlass.Float32)
                                    beta = (
                                        (
                                            cutlass.Float32(1.0)
                                            / (
                                                cutlass.Float32(1.0)
                                                + cute.math.exp(-b_bf)
                                            )
                                        )
                                        .to(cutlass.BFloat16)
                                        .to(cutlass.Float32)
                                    )
                                    xg = a_bf + dtb_h
                                    spl = cute.math.log(
                                        cutlass.Float32(1.0) + cute.math.exp(xg)
                                    )
                                    if xg > cutlass.Float32(20.0):
                                        spl = xg
                                    g = -cute.math.exp(alog_h) * spl
                                    eg = cute.math.exp(g)
                                    kvm = cutlass.Float32(0.0)
                                    ot = cutlass.Float32(0.0)
                                    if cutlass.const_expr(self.profile):
                                        ts1 = cute.arch.globaltimer()
                                        if tidx == 0:
                                            self._prec(
                                                mProf,
                                                mPcnt,
                                                pblk,
                                                L * 32 + 21,
                                                ts0,
                                                ts1,
                                                work,
                                            )
                                        ts0 = ts1
                                    cute.arch.mbarrier_wait(state_bar, st_phase)
                                    st_phase = st_phase ^ 1
                                    if cutlass.const_expr(self.profile):
                                        ts1 = cute.arch.globaltimer()
                                        if tidx == 0:
                                            self._prec(
                                                mProf,
                                                mPcnt,
                                                pblk,
                                                L * 32 + 22,
                                                ts0,
                                                ts1,
                                                work,
                                            )
                                        ts0 = ts1
                                    kv1 = cutlass.Float32(0.0)
                                    kv2 = cutlass.Float32(0.0)
                                    kv3 = cutlass.Float32(0.0)
                                    for c8 in range(0, 8, 1, unroll=1):
                                        for i in cutlass.range_constexpr(4):
                                            r0 = c8 * 16 + 4 * i
                                            kvm = kvm + sStA[(r0, tidx)] * sK[r0]
                                            kv1 = (
                                                kv1 + sStA[(r0 + 1, tidx)] * sK[r0 + 1]
                                            )
                                            kv2 = (
                                                kv2 + sStA[(r0 + 2, tidx)] * sK[r0 + 2]
                                            )
                                            kv3 = (
                                                kv3 + sStA[(r0 + 3, tidx)] * sK[r0 + 3]
                                            )
                                    kvm = ((kvm + kv1) + (kv2 + kv3)) * eg
                                    delta = (yv - kvm) * beta
                                    if cutlass.const_expr(self.profile):
                                        ts1 = cute.arch.globaltimer()
                                        if tidx == 0:
                                            self._prec(
                                                mProf,
                                                mPcnt,
                                                pblk,
                                                L * 32 + 23,
                                                ts0,
                                                ts1,
                                                work,
                                            )
                                        ts0 = ts1
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
                                            ot = ot + s0 * sQg[r0]
                                            ot1 = ot1 + s1 * sQg[r0 + 1]
                                            ot2 = ot2 + s2 * sQg[r0 + 2]
                                            ot3 = ot3 + s3 * sQg[r0 + 3]
                                    ot = (ot + ot1) + (ot2 + ot3)
                                    cute.arch.fence_view_async_shared()
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if tidx == 0:
                                        gStF2 = cute.make_tensor(
                                            mSsm.iterator + head_base,
                                            cute.make_layout((16384,)),
                                        )
                                        sStF2 = cute.make_tensor(
                                            sStA.iterator, cute.make_layout((16384,))
                                        )
                                        cute.copy(bulk_s2g, sStF2, gStF2)
                                        cute.arch.cp_async_bulk_commit_group()
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if cutlass.const_expr(self.profile):
                                        ts1 = cute.arch.globaltimer()
                                        if tidx == 0:
                                            self._prec(
                                                mProf,
                                                mPcnt,
                                                pblk,
                                                L * 32 + 24,
                                                ts0,
                                                ts1,
                                                work,
                                            )
                                        ts0 = ts1
                                    if tidx < 2:
                                        self._wait_flag(
                                            mCntL, zc0 // self.tile_shape_mnk[0] + tidx
                                        )
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if cutlass.const_expr(self.profile):
                                        ts1 = cute.arch.globaltimer()
                                        if tidx == 0:
                                            self._prec(
                                                mProf,
                                                mPcnt,
                                                pblk,
                                                L * 32 + 25,
                                                ts0,
                                                ts1,
                                                work,
                                            )
                                        ts0 = ts1
                                    zv = mMz[(zc0 + tidx, m, 0)].to(cutlass.Float32)
                                    xf = ot.to(cutlass.BFloat16).to(cutlass.Float32)
                                    ssx = self._block_sum(
                                        xf * xf, sRed1, warp_idx, lane
                                    )
                                    rstd = cute.math.rsqrt(
                                        ssx / cutlass.Float32(128.0)
                                        + cutlass.Float32(self.eps)
                                    )
                                    y_bf = (
                                        (
                                            xf
                                            * rstd
                                            * mNormW[(tidx,)].to(cutlass.Float32)
                                        )
                                        .to(cutlass.BFloat16)
                                        .to(cutlass.Float32)
                                    )
                                    sil = zv / (
                                        cutlass.Float32(1.0) + cute.math.exp(-zv)
                                    )
                                    mAttn[(h * self.dv + tidx, m, 0)] = (y_bf * sil).to(
                                        mAttn.element_type
                                    )
                                    cute.arch.fence_acq_rel_gpu()
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if cutlass.const_expr(self.profile):
                                        if tidx == 0:
                                            self._prec(
                                                mProf,
                                                mPcnt,
                                                pblk,
                                                L * 32 + 26,
                                                ts0,
                                                cute.arch.globaltimer(),
                                                work,
                                            )
                                    if tidx == 0:
                                        cute.arch.cp_async_bulk_wait_group(
                                            0, read=True
                                        )  # the state store has left smem (deferred from the store)
                                        cute.arch.atomic_add(
                                            mCntL.iterator + (HD + h),
                                            cutlass.Int32(1),
                                            sem="acq_rel",
                                            scope="gpu",
                                        )
                                    core_bar.arrive()  # ring smem is free again for this CTA's producer
                        else:
                            # ============================ attention core ============================
                            mKC = cute.make_tensor(
                                cute.make_ptr(
                                    self.b_dtype,
                                    mPtab[(L, PT_KC)],
                                    AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((BIG,)),
                            )
                            mVC = cute.make_tensor(
                                cute.make_ptr(
                                    self.b_dtype,
                                    mPtab[(L, PT_VC)],
                                    AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((BIG,)),
                            )
                            if is_b0:
                                it0 = work - TAI
                                is_qp = it0 < TB0  # q-prep item (else kv-write item)
                                it = it0
                                if not is_qp:
                                    it = it0 - TB0
                                m = it // self.hkv
                                kh = it % self.hkv
                                krow0 = 2 * QD + kh * D
                                vrow0 = 2 * QD + KVD + kh * D
                                qrow0 = kh * G * D
                                NQT = G * D // 64
                                if is_qp:
                                    if tidx < NQT:
                                        self._wait_flag(mCntL, qrow0 // 64 + tidx)
                                else:
                                    if tidx < 8:
                                        base = krow0
                                        if tidx >= 4:
                                            base = vrow0
                                        self._wait_flag(mCntL, base // 64 + tidx % 4)
                                self.epilog_sync_barrier.arrive_and_wait()
                                mWq = cute.make_tensor(
                                    cute.make_ptr(
                                        self.b_dtype,
                                        mPtab[(L, PT_WQ)],
                                        AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout((D,)),
                                )
                                mWk = cute.make_tensor(
                                    cute.make_ptr(
                                        self.b_dtype,
                                        mPtab[(L, PT_WK)],
                                        AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout((D,)),
                                )
                                if is_qp:
                                    for h in cutlass.range_constexpr(G):
                                        q0 = mMz[(qrow0 + h * D + 2 * tidx, m, 0)].to(
                                            cutlass.Float32
                                        )
                                        q1 = mMz[
                                            (qrow0 + h * D + 2 * tidx + 1, m, 0)
                                        ].to(cutlass.Float32)
                                        ss = self._warp_sum(q0 * q0 + q1 * q1)
                                        if lane == 0:
                                            sRed[(warp_idx, h)] = ss
                                        acc_o[(h, 0)] = q0
                                        acc_o[(h, 1)] = q1
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    for h in cutlass.range_constexpr(G):
                                        rstd = cute.math.rsqrt(
                                            (
                                                sRed[(0, h)]
                                                + sRed[(1, h)]
                                                + sRed[(2, h)]
                                                + sRed[(3, h)]
                                            )
                                            / cutlass.Float32(float(D))
                                            + cutlass.Float32(self.eps)
                                        )
                                        sQ[(h, 2 * tidx)] = (
                                            (
                                                acc_o[(h, 0)]
                                                * rstd
                                                * mWq[(2 * tidx,)].to(cutlass.Float32)
                                            )
                                            .to(cutlass.BFloat16)
                                            .to(cutlass.Float32)
                                        )
                                        sQ[(h, 2 * tidx + 1)] = (
                                            (
                                                acc_o[(h, 1)]
                                                * rstd
                                                * mWq[(2 * tidx + 1,)].to(
                                                    cutlass.Float32
                                                )
                                            )
                                            .to(cutlass.BFloat16)
                                            .to(cutlass.Float32)
                                        )
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    pq = mPos[m]
                                    qb0 = it * G * D
                                    for h in cutlass.range_constexpr(G):
                                        for e in cutlass.range_constexpr(2):
                                            dd = 2 * tidx + e
                                            val = sQ[(h, dd)]
                                            if dd < HALF:
                                                val = (
                                                    (
                                                        val * mRope[(pq, dd)]
                                                        - sQ[(h, dd + HALF)]
                                                        * mRope[(pq, HALF + dd)]
                                                    )
                                                    .to(cutlass.BFloat16)
                                                    .to(cutlass.Float32)
                                                )
                                            elif dd < ROT:
                                                val = (
                                                    (
                                                        val * mRope[(pq, dd - HALF)]
                                                        + sQ[(h, dd - HALF)]
                                                        * mRope[(pq, dd)]
                                                    )
                                                    .to(cutlass.BFloat16)
                                                    .to(cutlass.Float32)
                                                )
                                            mQB[(qb0 + h * D + dd,)] = val
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    cute.arch.fence_acq_rel_gpu()
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if tidx == 0:
                                        cute.arch.atomic_add(
                                            mCntL.iterator + (QW + it),
                                            cutlass.Int32(1),
                                            sem="acq_rel",
                                            scope="gpu",
                                        )
                                else:
                                    for r_ in cutlass.range_constexpr(
                                        self.R
                                    ):  # §18: the R rows of sequence it // hkv, in order
                                        m = (it // self.hkv) * self.R + r_
                                        k0 = mMz[(krow0 + 2 * tidx, m, 0)].to(
                                            cutlass.Float32
                                        )
                                        k1 = mMz[(krow0 + 2 * tidx + 1, m, 0)].to(
                                            cutlass.Float32
                                        )
                                        ss = self._warp_sum(k0 * k0 + k1 * k1)
                                        if lane == 0:
                                            sRed[(warp_idx, 0)] = ss
                                        self.epilog_sync_barrier.arrive_and_wait()
                                        rstd = cute.math.rsqrt(
                                            (
                                                sRed[(0, 0)]
                                                + sRed[(1, 0)]
                                                + sRed[(2, 0)]
                                                + sRed[(3, 0)]
                                            )
                                            / cutlass.Float32(float(D))
                                            + cutlass.Float32(self.eps)
                                        )
                                        kn0 = (
                                            (
                                                k0
                                                * rstd
                                                * mWk[(2 * tidx,)].to(cutlass.Float32)
                                            )
                                            .to(cutlass.BFloat16)
                                            .to(cutlass.Float32)
                                        )
                                        kn1 = (
                                            (
                                                k1
                                                * rstd
                                                * mWk[(2 * tidx + 1,)].to(
                                                    cutlass.Float32
                                                )
                                            )
                                            .to(cutlass.BFloat16)
                                            .to(cutlass.Float32)
                                        )
                                        sTmp[2 * tidx] = kn0
                                        sTmp[2 * tidx + 1] = kn1
                                        self.epilog_sync_barrier.arrive_and_wait()
                                        p = mPos[m]
                                        o0 = kn0
                                        o1 = kn1
                                        for e in cutlass.range_constexpr(2):
                                            dd = 2 * tidx + e
                                            val = kn0
                                            if e == 1:
                                                val = kn1
                                            if dd < HALF:
                                                cs_ = mRope[(p, dd)]
                                                sn = mRope[(p, HALF + dd)]
                                                val = val * cs_ - sTmp[dd + HALF] * sn
                                            elif dd < ROT:
                                                cs_ = mRope[(p, dd - HALF)]
                                                sn = mRope[(p, HALF + dd - HALF)]
                                                val = val * cs_ + sTmp[dd - HALF] * sn
                                            if e == 0:
                                                o0 = val
                                            else:
                                                o1 = val
                                        sl = mSlot[m]
                                        if sl >= 0:
                                            cbase = (sl * self.hkv + kh) * D
                                            mKC[(cbase + 2 * tidx,)] = o0.to(
                                                mKC.element_type
                                            )
                                            mKC[(cbase + 2 * tidx + 1,)] = o1.to(
                                                mKC.element_type
                                            )
                                            mVC[(cbase + 2 * tidx,)] = mMz[
                                                (vrow0 + 2 * tidx, m, 0)
                                            ]
                                            mVC[(cbase + 2 * tidx + 1,)] = mMz[
                                                (vrow0 + 2 * tidx + 1, m, 0)
                                            ]
                                    cute.arch.fence_acq_rel_gpu()
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if tidx == 0:
                                        cute.arch.atomic_add(
                                            mCntL.iterator + (KVW + it),
                                            cutlass.Int32(1),
                                            sem="acq_rel",
                                            scope="gpu",
                                        )
                            else:
                                if cutlass.const_expr(self.R > 1):
                                    # §18b verify form: one item per (sequence, kv head, split) scores the R rows together
                                    it = work - TAI - TB0I
                                    s_id = it % S
                                    ik = it // S
                                    sq_ = ik // self.hkv
                                    kh = ik % self.hkv
                                    qrow0 = kh * G * D
                                    grow0 = QD + qrow0
                                    m0 = sq_ * self.R
                                    NQ = self.R * G
                                    seq_len = mSeq[sq_]
                                    lps = (
                                        (seq_len + S * PAGE - 1) // (S * PAGE)
                                    ) * PAGE
                                    start_n = lps * s_id
                                    end_n = start_n + lps
                                    if end_n > seq_len:
                                        end_n = seq_len
                                    if tidx < self.R:
                                        self._wait_flag(
                                            mCntL, QW + (m0 + tidx) * self.hkv + kh
                                        )  # q of every row prepared
                                    if tidx == 16:
                                        if (
                                            end_n > seq_len - self.R
                                        ):  # the split holding the new rows needs their K/V written
                                            if start_n < seq_len:
                                                self._wait_flag(
                                                    mCntL, KVW + sq_ * self.hkv + kh
                                                )
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    for j in cutlass.range_constexpr(
                                        (NQ * 8 + 127) // 128
                                    ):  # 32-column chunks of the NQ rows
                                        cch = tidx + 128 * j
                                        if cch < NQ * 8:
                                            qr = cch // 8
                                            qc0 = (cch % 8) * 32
                                            qb0 = (
                                                (m0 + qr // G) * self.hkv + kh
                                            ) * G * D + (qr % G) * D
                                            for e in cutlass.range_constexpr(32):
                                                sQw[(qr, qc0 + e)] = mQB[
                                                    (qb0 + qc0 + e,)
                                                ].to(cutlass.BFloat16)
                                    if tidx < 32:
                                        sStw[(0, tidx)] = cutlass.Float32(-1.0e30)
                                        sStw[(1, tidx)] = cutlass.Float32(0.0)
                                        sStw[(2, tidx)] = cutlass.Float32(1.0)
                                    for mt in cutlass.range_constexpr(MTQ):
                                        acc_pv_w[mt].fill(0.0)
                                    npages = cutlass.Int32(0)
                                    if end_n > start_n:
                                        npages = (end_n - start_n + PAGE - 1) // PAGE
                                    if npages > 0:
                                        page_id0 = mBT[
                                            (sq_ * BT_STRIDE + start_n // PAGE,)
                                        ]
                                        kb0 = cute.assume(
                                            (page_id0 * PAGE * self.hkv + kh) * D,
                                            divby=256,
                                        )
                                        gK0 = cute.make_tensor(
                                            mKC.iterator + kb0, gK_lay
                                        )
                                        gV0 = cute.make_tensor(
                                            mVC.iterator + kb0, gV_lay
                                        )
                                        cute.copy(
                                            tcp_k,
                                            thr_cpk.partition_S(gK0),
                                            tKsD[(None, None, None, 0)],
                                        )
                                        cute.copy(
                                            tcp_v,
                                            thr_cpv.partition_S(gV0),
                                            tVsD[(None, None, None, 0)],
                                        )
                                        cute.arch.cp_async_commit_group()
                                    for pg in range(0, npages, 1):
                                        buf = pg % 2
                                        n0 = start_n + pg * PAGE
                                        cute.arch.cp_async_wait_group(0)
                                        self.epilog_sync_barrier.arrive_and_wait()
                                        if pg + 1 < npages:
                                            page_idn = mBT[
                                                (sq_ * BT_STRIDE + (n0 + PAGE) // PAGE,)
                                            ]
                                            kbn = cute.assume(
                                                (page_idn * PAGE * self.hkv + kh) * D,
                                                divby=256,
                                            )
                                            gKn = cute.make_tensor(
                                                mKC.iterator + kbn, gK_lay
                                            )
                                            gVn = cute.make_tensor(
                                                mVC.iterator + kbn, gV_lay
                                            )
                                            cute.copy(
                                                tcp_k,
                                                thr_cpk.partition_S(gKn),
                                                tKsD[(None, None, None, 1 - buf)],
                                            )
                                            cute.copy(
                                                tcp_v,
                                                thr_cpv.partition_S(gVn),
                                                tVsD[(None, None, None, 1 - buf)],
                                            )
                                            cute.arch.cp_async_commit_group()
                                        tKs_b = tKs[(None, None, None, buf)]
                                        tVs_b = tVs[(None, None, None, buf)]
                                        for mt in cutlass.range_constexpr(MTQ):
                                            acc_s.fill(0.0)
                                            for kk in cutlass.range_constexpr(16):
                                                cute.copy(
                                                    tcpy_q,
                                                    tQs_w[mt][(None, None, kk)],
                                                    tQr_w[mt][(None, None, kk)],
                                                )
                                                cute.copy(
                                                    tcpy_k,
                                                    tKs_b[(None, None, kk)],
                                                    tKr[(None, None, kk)],
                                                )
                                                cute.gemm(
                                                    mma_qk,
                                                    acc_s,
                                                    tCrQ_w[mt][(None, None, kk)],
                                                    tCrK[(None, None, kk)],
                                                    acc_s,
                                                )
                                            for i in cutlass.range_constexpr(
                                                cute.size(acc_s)
                                            ):
                                                crd = tCcS[i]
                                                sSc[(crd[0], crd[1])] = acc_s[i]
                                            self.epilog_sync_barrier.arrive_and_wait()
                                            for rep in cutlass.range_constexpr(4):
                                                lr = (
                                                    warp_idx + 4 * rep
                                                )  # row within the m-tile (lane = key)
                                                gr = (
                                                    mt * 16 + lr
                                                )  # row of the wide tile = r*G + h
                                                if gr < NQ:
                                                    end_r = (
                                                        seq_len - (self.R - 1 - gr // G)
                                                    )  # causal: row r sees keys < seq_len-(R-1-r)
                                                    sc = sSc[(lr, lane)] * qk_scale
                                                    if n0 + lane >= end_r:
                                                        sc = cutlass.Float32(-1.0e30)
                                                    mx = self._warp_max(sc)
                                                    m_old = sStw[(0, gr)]
                                                    m_new = m_old
                                                    if mx > m_new:
                                                        m_new = mx
                                                    pv = cute.math.exp2(sc - m_new)
                                                    if sc <= cutlass.Float32(-1.0e29):
                                                        pv = cutlass.Float32(0.0)
                                                    lsum = self._warp_sum(pv)
                                                    alpha = cute.math.exp2(
                                                        m_old - m_new
                                                    )
                                                    sPb[(lr, lane)] = pv.to(
                                                        cutlass.BFloat16
                                                    )
                                                    if lane == 0:
                                                        sStw[(0, gr)] = m_new
                                                        sStw[(1, gr)] = (
                                                            sStw[(1, gr)] * alpha + lsum
                                                        )
                                                        sStw[(2, gr)] = alpha
                                                else:
                                                    sPb[(lr, lane)] = cutlass.BFloat16(
                                                        0.0
                                                    )
                                            self.epilog_sync_barrier.arrive_and_wait()
                                            for i in cutlass.range_constexpr(
                                                cute.size(acc_pv)
                                            ):
                                                gr2 = mt * 16 + tCcO[i][0]
                                                if gr2 < NQ:
                                                    acc_pv_w[mt][i] = (
                                                        acc_pv_w[mt][i] * sStw[(2, gr2)]
                                                    )
                                            for kk in cutlass.range_constexpr(2):
                                                cute.copy(
                                                    tcpy_p,
                                                    tPs[(None, None, kk)],
                                                    tPr[(None, None, kk)],
                                                )
                                                cute.copy(
                                                    tcpy_v,
                                                    tVs_b[(None, None, kk)],
                                                    tVr[(None, None, kk)],
                                                )
                                                cute.gemm(
                                                    mma_pv,
                                                    acc_pv_w[mt],
                                                    tCrP[(None, None, kk)],
                                                    tCrV[(None, None, kk)],
                                                    acc_pv_w[mt],
                                                )
                                    self.epilog_sync_barrier.arrive_and_wait()  # every warp is done with the ring (P / V / Q)
                                    core_bar.arrive()  # ring smem is free again for this CTA's producer
                                    for mt in cutlass.range_constexpr(MTQ):
                                        for i in cutlass.range_constexpr(
                                            cute.size(acc_pv)
                                        ):
                                            crd = tCcO[i]
                                            gr3 = mt * 16 + crd[0]
                                            if gr3 < NQ:
                                                mrow = m0 + gr3 // G
                                                pbase_r = (
                                                    (mrow * self.hkv + kh) * S + s_id
                                                ) * G
                                                l_o = sStw[(1, gr3)]
                                                inv_o = cutlass.Float32(0.0)
                                                if l_o > cutlass.Float32(0.0):
                                                    inv_o = cutlass.Float32(1.0) / l_o
                                                mPO[
                                                    ((pbase_r + gr3 % G) * D + crd[1],)
                                                ] = acc_pv_w[mt][i] * inv_o
                                    if tidx < NQ:
                                        mrow = m0 + tidx // G
                                        pbase_r = (
                                            (mrow * self.hkv + kh) * S + s_id
                                        ) * G
                                        l_h = sStw[(1, tidx)]
                                        lse = cutlass.Float32(-1.0e30)
                                        if l_h > cutlass.Float32(0.0):
                                            lse = sStw[(0, tidx)] + cute.math.log2(l_h)
                                        mPL[(pbase_r + tidx % G,)] = lse
                                    cute.arch.fence_acq_rel_gpu()
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if tidx < self.R:
                                        old = cute.arch.atomic_add(
                                            mCntL.iterator
                                            + (MERGE + (m0 + tidx) * self.hkv + kh),
                                            cutlass.Int32(1),
                                            sem="acq_rel",
                                            scope="gpu",
                                        )
                                        sFlag[tidx] = old
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    for r_ in cutlass.range_constexpr(self.R):
                                        if sFlag[r_] == S - 1:
                                            # last arriver for token m0 + r_: merge its S partials (same code as the plain item)
                                            m = m0 + r_
                                            mk = m * self.hkv + kh
                                            cute.arch.fence_acq_rel_gpu()
                                            NGT = G * D // 64
                                            if tidx < NGT:
                                                self._wait_flag(
                                                    mCntL, grow0 // 64 + tidx
                                                )
                                            self.epilog_sync_barrier.arrive_and_wait()
                                            mbase = mk * S * G
                                            for rep_ in cutlass.range_constexpr(
                                                (G + 3) // 4
                                            ):
                                                hh = warp_idx + 4 * rep_
                                                if hh < G:
                                                    lmax = cutlass.Float32(-1.0e30)
                                                    for s2 in range(0, S, 1):
                                                        lv = mPL[(mbase + s2 * G + hh,)]
                                                        if lv > lmax:
                                                            lmax = lv
                                                    wsum = cutlass.Float32(0.0)
                                                    for j in cutlass.range_constexpr(8):
                                                        om[j] = cutlass.Float32(0.0)
                                                    for s2 in range(0, S, 1, unroll=4):
                                                        lv = mPL[(mbase + s2 * G + hh,)]
                                                        gpo = cute.make_tensor(
                                                            mPO.iterator
                                                            + cute.assume(
                                                                (mbase + s2 * G + hh)
                                                                * D
                                                                + lane * 8,
                                                                divby=8,
                                                            ),
                                                            cute.make_layout((8,)),
                                                        )
                                                        cute.autovec_copy(gpo, pv8)
                                                        wgt = cutlass.Float32(0.0)
                                                        if lv > cutlass.Float32(
                                                            -1.0e29
                                                        ):
                                                            wgt = cute.math.exp2(
                                                                lv - lmax
                                                            )
                                                        wsum = wsum + wgt
                                                        for (
                                                            j
                                                        ) in cutlass.range_constexpr(8):
                                                            om[j] = om[j] + wgt * pv8[j]
                                                    for j in cutlass.range_constexpr(8):
                                                        dd = lane * 8 + j
                                                        oo = (
                                                            (om[j] / wsum)
                                                            .to(cutlass.BFloat16)
                                                            .to(cutlass.Float32)
                                                        )
                                                        gg = mMz[
                                                            (grow0 + hh * D + dd, m, 0)
                                                        ].to(cutlass.Float32)
                                                        mAttn[
                                                            (qrow0 + hh * D + dd, m, 0)
                                                        ] = (
                                                            oo
                                                            / (
                                                                cutlass.Float32(1.0)
                                                                + cute.math.exp(-gg)
                                                            )
                                                        ).to(mAttn.element_type)
                                            cute.arch.fence_acq_rel_gpu()
                                            self.epilog_sync_barrier.arrive_and_wait()
                                            if tidx == 0:
                                                cute.arch.atomic_add(
                                                    mCntL.iterator + (HD + kh),
                                                    cutlass.Int32(1),
                                                    sem="acq_rel",
                                                    scope="gpu",
                                                )
                                else:
                                    it = (
                                        work - TAI - TB0I
                                    )  # after the q-prep AND kv-write items
                                    s_id = it % S
                                    mk = it // S
                                    m = mk // self.hkv
                                    kh = mk % self.hkv
                                    qrow0 = kh * G * D
                                    grow0 = QD + qrow0
                                    tu0 = cutlass.Int64(0)
                                    if cutlass.const_expr(self.profile):
                                        tu0 = cute.arch.globaltimer()
                                    sq_ = (
                                        m // self.R
                                    )  # §18: sequence of row m (== m when R == 1)
                                    rr_ = m % self.R
                                    seq_len = mSeq[sq_]
                                    lps = (
                                        (seq_len + S * PAGE - 1) // (S * PAGE)
                                    ) * PAGE
                                    start_n = lps * s_id
                                    end_n = start_n + lps
                                    if end_n > seq_len:
                                        end_n = seq_len
                                    end_eff = end_n  # §18: row rr_ of the R verify rows sees keys [0, seq_len - (R-1-rr_))
                                    if end_eff > seq_len - (self.R - 1 - rr_):
                                        end_eff = seq_len - (self.R - 1 - rr_)
                                    if tidx == 0:
                                        self._wait_flag(
                                            mCntL, QW + mk
                                        )  # q of this (token, kv head) is prepared
                                    if tidx == 1:
                                        if (
                                            end_n > seq_len - self.R
                                        ):  # only the split holding the new row(s) needs their K/V written
                                            if start_n < seq_len:
                                                self._wait_flag(
                                                    mCntL, KVW + sq_ * self.hkv + kh
                                                )
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    pbase = (mk * S + s_id) * G
                                    if cutlass.const_expr(self.mma_attn):
                                        # ---- §16 item 3b: tensor-core page loop. Q (bf16-rounded, as in mQB) rows >= G are zero, P rows
                                        #      >= G stay zero, so heads are MMA rows; K/V pages stream through the ring via cp.async.
                                        qb0 = mk * G * D
                                        for j in cutlass.range_constexpr(4):
                                            pidx = tidx * 4 + j
                                            prow = pidx // PAGE
                                            if prow >= G:
                                                sPb[(prow, pidx % PAGE)] = (
                                                    cutlass.BFloat16(0.0)
                                                )
                                        qrow = tidx // 8
                                        for j in cutlass.range_constexpr(32):
                                            qcol = (tidx % 8) * 32 + j
                                            qv = cutlass.Float32(0.0)
                                            if qrow < G:
                                                qv = mQB[(qb0 + qrow * D + qcol,)]
                                            sQb[(qrow, qcol)] = qv.to(cutlass.BFloat16)
                                        if tidx < 8:
                                            sStat[(0, tidx)] = cutlass.Float32(-1.0e30)
                                            sStat[(1, tidx)] = cutlass.Float32(0.0)
                                        acc_pv.fill(0.0)
                                        if cutlass.const_expr(self.kring3):
                                            self.epilog_sync_barrier.arrive_and_wait()  # Q staged (in K buffer 2's smem)
                                            for kk in cutlass.range_constexpr(16):
                                                cute.copy(
                                                    tcpy_q,
                                                    tQs[(None, None, kk)],
                                                    tQr[(None, None, kk)],
                                                )
                                        npages = cutlass.Int32(0)
                                        if end_eff > start_n:
                                            npages = (
                                                end_eff - start_n + PAGE - 1
                                            ) // PAGE
                                        if cutlass.const_expr(self.profile):
                                            tu1 = cute.arch.globaltimer()
                                            if tidx == 0:
                                                self._prec(
                                                    mProf,
                                                    mPcnt,
                                                    pblk,
                                                    L * 32 + 28,
                                                    tu0,
                                                    tu1,
                                                    work,
                                                )
                                            tu0 = tu1
                                        if npages > 0:
                                            page_id0 = mBT[
                                                (sq_ * BT_STRIDE + start_n // PAGE,)
                                            ]
                                            kb0 = cute.assume(
                                                (page_id0 * PAGE * self.hkv + kh) * D,
                                                divby=256,
                                            )
                                            gK0 = cute.make_tensor(
                                                mKC.iterator + kb0, gK_lay
                                            )
                                            gV0 = cute.make_tensor(
                                                mVC.iterator + kb0, gV_lay
                                            )
                                            cute.copy(
                                                tcp_k,
                                                thr_cpk.partition_S(gK0),
                                                tKsD[(None, None, None, 0)],
                                            )
                                            cute.copy(
                                                tcp_v,
                                                thr_cpv.partition_S(gV0),
                                                tVsD[(None, None, None, 0)],
                                            )
                                            cute.arch.cp_async_commit_group()
                                        if (
                                            cutlass.const_expr(self.kring3)
                                            and npages > 1
                                        ):  # groups: [K0 V0] [V1] [K1]  (K2 follows at iteration 0, after the Q fragments are in registers)
                                            page_id1 = mBT[
                                                (sq_ * BT_STRIDE + start_n // PAGE + 1,)
                                            ]
                                            kb1 = cute.assume(
                                                (page_id1 * PAGE * self.hkv + kh) * D,
                                                divby=256,
                                            )
                                            gV1 = cute.make_tensor(
                                                mVC.iterator + kb1, gV_lay
                                            )
                                            cute.copy(
                                                tcp_v,
                                                thr_cpv.partition_S(gV1),
                                                tVsD[(None, None, None, 1)],
                                            )
                                            cute.arch.cp_async_commit_group()
                                            gK1 = cute.make_tensor(
                                                mKC.iterator + kb1, gK_lay
                                            )
                                            cute.copy(
                                                tcp_k,
                                                thr_cpk.partition_S(gK1),
                                                tKsD[(None, None, None, 1)],
                                            )
                                            cute.arch.cp_async_commit_group()
                                        for pg in range(0, npages, 1):
                                            kbuf = pg % self.nkb
                                            vbuf = pg % 2
                                            n0 = start_n + pg * PAGE
                                            if cutlass.const_expr(self.kring3):
                                                # pending groups at this point: [.. K(pg)] [V(pg)] [K(pg+1)] (the last only if it exists;
                                                # at pg == 0 the order is [K0 V0] [V1] [K1])
                                                if pg == 0:
                                                    if npages > 1:
                                                        cute.arch.cp_async_wait_group(2)
                                                    else:
                                                        cute.arch.cp_async_wait_group(0)
                                                else:
                                                    if pg + 1 < npages:
                                                        cute.arch.cp_async_wait_group(1)
                                                    else:
                                                        cute.arch.cp_async_wait_group(0)
                                            else:
                                                cute.arch.cp_async_wait_group(0)
                                            self.epilog_sync_barrier.arrive_and_wait()  # page pg landed for every thread; page pg-1 fully consumed
                                            if cutlass.const_expr(self.kring3):
                                                if pg >= 1:
                                                    if (
                                                        pg + 1 < npages
                                                    ):  # V one page ahead (its buffer held V(pg-1))
                                                        page_idn = mBT[
                                                            (
                                                                sq_ * BT_STRIDE
                                                                + (n0 + PAGE) // PAGE,
                                                            )
                                                        ]
                                                        kbn = cute.assume(
                                                            (
                                                                page_idn
                                                                * PAGE
                                                                * self.hkv
                                                                + kh
                                                            )
                                                            * D,
                                                            divby=256,
                                                        )
                                                        gVn = cute.make_tensor(
                                                            mVC.iterator + kbn, gV_lay
                                                        )
                                                        cute.copy(
                                                            tcp_v,
                                                            thr_cpv.partition_S(gVn),
                                                            tVsD[
                                                                (
                                                                    None,
                                                                    None,
                                                                    None,
                                                                    1 - vbuf,
                                                                )
                                                            ],
                                                        )
                                                        cute.arch.cp_async_commit_group()
                                                if (
                                                    pg + 2 < npages
                                                ):  # K two pages ahead (its buffer held K(pg-1))
                                                    page_id2 = mBT[
                                                        (
                                                            sq_ * BT_STRIDE
                                                            + (n0 + 2 * PAGE) // PAGE,
                                                        )
                                                    ]
                                                    kb2 = cute.assume(
                                                        (
                                                            page_id2 * PAGE * self.hkv
                                                            + kh
                                                        )
                                                        * D,
                                                        divby=256,
                                                    )
                                                    gK2 = cute.make_tensor(
                                                        mKC.iterator + kb2, gK_lay
                                                    )
                                                    cute.copy(
                                                        tcp_k,
                                                        thr_cpk.partition_S(gK2),
                                                        tKsD[
                                                            (
                                                                None,
                                                                None,
                                                                None,
                                                                (pg + 2) % 3,
                                                            )
                                                        ],
                                                    )
                                                    cute.arch.cp_async_commit_group()
                                            else:
                                                if (
                                                    pg + 1 < npages
                                                ):  # 2 buffers: page pg+1 goes out once page pg-1's buffer is free
                                                    page_idn = mBT[
                                                        (
                                                            sq_ * BT_STRIDE
                                                            + (n0 + PAGE) // PAGE,
                                                        )
                                                    ]
                                                    kbn = cute.assume(
                                                        (
                                                            page_idn * PAGE * self.hkv
                                                            + kh
                                                        )
                                                        * D,
                                                        divby=256,
                                                    )
                                                    gKn = cute.make_tensor(
                                                        mKC.iterator + kbn, gK_lay
                                                    )
                                                    gVn = cute.make_tensor(
                                                        mVC.iterator + kbn, gV_lay
                                                    )
                                                    cute.copy(
                                                        tcp_k,
                                                        thr_cpk.partition_S(gKn),
                                                        tKsD[
                                                            (None, None, None, 1 - vbuf)
                                                        ],
                                                    )
                                                    cute.copy(
                                                        tcp_v,
                                                        thr_cpv.partition_S(gVn),
                                                        tVsD[
                                                            (None, None, None, 1 - vbuf)
                                                        ],
                                                    )
                                                    cute.arch.cp_async_commit_group()
                                            # S = Q K^T (16 heads x 32 keys), 16 k-steps of 16 dims (Q fragments in registers with kring3)
                                            acc_s.fill(0.0)
                                            tKs_b = tKs[(None, None, None, kbuf)]
                                            for kk in cutlass.range_constexpr(16):
                                                if cutlass.const_expr(not self.kring3):
                                                    cute.copy(
                                                        tcpy_q,
                                                        tQs[(None, None, kk)],
                                                        tQr[(None, None, kk)],
                                                    )
                                                cute.copy(
                                                    tcpy_k,
                                                    tKs_b[(None, None, kk)],
                                                    tKr[(None, None, kk)],
                                                )
                                                cute.gemm(
                                                    mma_qk,
                                                    acc_s,
                                                    tCrQ[(None, None, kk)],
                                                    tCrK[(None, None, kk)],
                                                    acc_s,
                                                )
                                            for i in cutlass.range_constexpr(
                                                cute.size(acc_s)
                                            ):
                                                crd = tCcS[i]
                                                sSc[(crd[0], crd[1])] = acc_s[i]
                                            self.epilog_sync_barrier.arrive_and_wait()
                                            for rep in cutlass.range_constexpr(2):
                                                hh = warp_idx + 4 * rep
                                                if hh < G:
                                                    sc = sSc[(hh, lane)] * qk_scale
                                                    if n0 + lane >= end_eff:
                                                        sc = cutlass.Float32(-1.0e30)
                                                    mx = self._warp_max(sc)
                                                    m_old = sStat[(0, hh)]
                                                    m_new = m_old
                                                    if mx > m_new:
                                                        m_new = mx
                                                    pv = cute.math.exp2(sc - m_new)
                                                    if sc <= cutlass.Float32(-1.0e29):
                                                        pv = cutlass.Float32(0.0)
                                                    lsum = self._warp_sum(pv)
                                                    alpha = cute.math.exp2(
                                                        m_old - m_new
                                                    )
                                                    sPb[(hh, lane)] = pv.to(
                                                        cutlass.BFloat16
                                                    )
                                                    if lane == 0:
                                                        sStat[(0, hh)] = m_new
                                                        sStat[(1, hh)] = (
                                                            sStat[(1, hh)] * alpha
                                                            + lsum
                                                        )
                                                        sStat[(2, hh)] = alpha
                                            self.epilog_sync_barrier.arrive_and_wait()
                                            for i in cutlass.range_constexpr(
                                                cute.size(acc_pv)
                                            ):
                                                orow = tCcO[i][0]
                                                if orow < G:
                                                    acc_pv[i] = (
                                                        acc_pv[i] * sStat[(2, orow)]
                                                    )
                                            # O += P V (16 heads x 256 dims), 2 k-steps of 16 keys; V fragments via ldmatrix.trans
                                            tVs_b = tVs[(None, None, None, vbuf)]
                                            for kk in cutlass.range_constexpr(2):
                                                cute.copy(
                                                    tcpy_p,
                                                    tPs[(None, None, kk)],
                                                    tPr[(None, None, kk)],
                                                )
                                                cute.copy(
                                                    tcpy_v,
                                                    tVs_b[(None, None, kk)],
                                                    tVr[(None, None, kk)],
                                                )
                                                cute.gemm(
                                                    mma_pv,
                                                    acc_pv,
                                                    tCrP[(None, None, kk)],
                                                    tCrV[(None, None, kk)],
                                                    acc_pv,
                                                )
                                            # (no loop-end barrier: the next page's post-wait barrier orders the buffer reuse)
                                        self.epilog_sync_barrier.arrive_and_wait()  # every warp is done with the ring (P / V / Q)
                                        core_bar.arrive()  # ring smem is free again for this CTA's producer
                                        if cutlass.const_expr(self.profile):
                                            tu1 = cute.arch.globaltimer()
                                            if tidx == 0:
                                                self._prec(
                                                    mProf,
                                                    mPcnt,
                                                    pblk,
                                                    L * 32 + 29,
                                                    tu0,
                                                    tu1,
                                                    work,
                                                )
                                            tu0 = tu1
                                        for i in cutlass.range_constexpr(
                                            cute.size(acc_pv)
                                        ):
                                            crd = tCcO[i]
                                            orow = crd[0]
                                            if orow < G:
                                                l_o = sStat[(1, orow)]
                                                inv_o = cutlass.Float32(0.0)
                                                if l_o > cutlass.Float32(0.0):
                                                    inv_o = cutlass.Float32(1.0) / l_o
                                                mPO[((pbase + orow) * D + crd[1],)] = (
                                                    acc_pv[i] * inv_o
                                                )
                                    else:
                                        qb0 = mk * G * D
                                        for h in cutlass.range_constexpr(G):
                                            sQ[(h, 2 * tidx)] = mQB[
                                                (qb0 + h * D + 2 * tidx,)
                                            ]
                                            sQ[(h, 2 * tidx + 1)] = mQB[
                                                (qb0 + h * D + 2 * tidx + 1,)
                                            ]
                                            acc_o[(h, 0)] = cutlass.Float32(0.0)
                                            acc_o[(h, 1)] = cutlass.Float32(0.0)
                                        if tidx < 8:
                                            sStat[(0, tidx)] = cutlass.Float32(-1.0e30)
                                            sStat[(1, tidx)] = cutlass.Float32(0.0)
                                        self.epilog_sync_barrier.arrive_and_wait()
                                        key_id = tidx % 32
                                        quarter = tidx // 32
                                        for h in cutlass.range_constexpr(G):
                                            for j in cutlass.range_constexpr(8):
                                                acc_v[(h, j)] = cutlass.Float32(0.0)
                                        npages = cutlass.Int32(0)
                                        if end_eff > start_n:
                                            npages = (
                                                end_eff - start_n + PAGE - 1
                                            ) // PAGE
                                        if cutlass.const_expr(self.profile):
                                            tu1 = cute.arch.globaltimer()
                                            if tidx == 0:
                                                self._prec(
                                                    mProf,
                                                    mPcnt,
                                                    pblk,
                                                    L * 32 + 28,
                                                    tu0,
                                                    tu1,
                                                    work,
                                                )
                                            tu0 = tu1
                                        if npages > 0:
                                            page_id0 = mBT[
                                                (sq_ * BT_STRIDE + start_n // PAGE,)
                                            ]
                                            kbase0 = cute.assume(
                                                (
                                                    (page_id0 * PAGE + key_id)
                                                    * self.hkv
                                                    + kh
                                                )
                                                * D
                                                + quarter * 64,
                                                divby=64,
                                            )
                                            gk0 = cute.make_tensor(
                                                mKC.iterator + kbase0,
                                                cute.make_layout((64,)),
                                            )
                                            cute.autovec_copy(gk0, kfrag)
                                        for pg in range(0, npages, 1):
                                            n0 = start_n + pg * PAGE
                                            page_id = mBT[
                                                (sq_ * BT_STRIDE + n0 // PAGE,)
                                            ]
                                            for kk in cutlass.range_constexpr(8):
                                                vbase = cute.assume(
                                                    (
                                                        (
                                                            page_id * PAGE
                                                            + warp_idx * 8
                                                            + kk
                                                        )
                                                        * self.hkv
                                                        + kh
                                                    )
                                                    * D
                                                    + lane * 8,
                                                    divby=8,
                                                )
                                                gv = cute.make_tensor(
                                                    mVC.iterator + vbase,
                                                    cute.make_layout((8,)),
                                                )
                                                cute.autovec_copy(gv, vfrag[(kk, None)])
                                            for j in cutlass.range_constexpr(64):
                                                kf32[j] = kfrag[j].to(cutlass.Float32)
                                            for h in cutlass.range_constexpr(G):
                                                d0 = cutlass.Float32(0.0)
                                                d1 = cutlass.Float32(0.0)
                                                d2 = cutlass.Float32(0.0)
                                                d3 = cutlass.Float32(0.0)
                                                for jj in cutlass.range_constexpr(16):
                                                    sq4 = cute.make_tensor(
                                                        sQ.iterator
                                                        + (
                                                            h * D
                                                            + quarter * 64
                                                            + 4 * jj
                                                        ),
                                                        cute.make_layout((4,)),
                                                    )
                                                    cute.autovec_copy(sq4, q4)
                                                    d0 = d0 + kf32[4 * jj] * q4[0]
                                                    d1 = d1 + kf32[4 * jj + 1] * q4[1]
                                                    d2 = d2 + kf32[4 * jj + 2] * q4[2]
                                                    d3 = d3 + kf32[4 * jj + 3] * q4[3]
                                                sPart[(quarter, key_id, h)] = (
                                                    d0 + d1
                                                ) + (d2 + d3)
                                            if pg + 1 < npages:
                                                page_idn = mBT[
                                                    (
                                                        sq_ * BT_STRIDE
                                                        + (n0 + PAGE) // PAGE,
                                                    )
                                                ]
                                                kbasen = cute.assume(
                                                    (
                                                        (page_idn * PAGE + key_id)
                                                        * self.hkv
                                                        + kh
                                                    )
                                                    * D
                                                    + quarter * 64,
                                                    divby=64,
                                                )
                                                gkn = cute.make_tensor(
                                                    mKC.iterator + kbasen,
                                                    cute.make_layout((64,)),
                                                )
                                                cute.autovec_copy(gkn, kfrag_n)
                                            self.epilog_sync_barrier.arrive_and_wait()
                                            for rep in cutlass.range_constexpr(2):
                                                hh = warp_idx + 4 * rep
                                                if hh < G:
                                                    sc = (
                                                        sPart[(0, lane, hh)]
                                                        + sPart[(1, lane, hh)]
                                                        + sPart[(2, lane, hh)]
                                                        + sPart[(3, lane, hh)]
                                                    ) * qk_scale
                                                    if n0 + lane >= end_eff:
                                                        sc = cutlass.Float32(-1.0e30)
                                                    mx = self._warp_max(sc)
                                                    m_old = sStat[(0, hh)]
                                                    m_new = m_old
                                                    if mx > m_new:
                                                        m_new = mx
                                                    pv = cute.math.exp2(sc - m_new)
                                                    if sc <= cutlass.Float32(-1.0e29):
                                                        pv = cutlass.Float32(0.0)
                                                    lsum = self._warp_sum(pv)
                                                    alpha = cute.math.exp2(
                                                        m_old - m_new
                                                    )
                                                    sP[(lane, hh)] = pv
                                                    if lane == 0:
                                                        sStat[(0, hh)] = m_new
                                                        sStat[(1, hh)] = (
                                                            sStat[(1, hh)] * alpha
                                                            + lsum
                                                        )
                                                        sStat[(2, hh)] = alpha
                                            self.epilog_sync_barrier.arrive_and_wait()
                                            for h in cutlass.range_constexpr(G):
                                                al = sStat[(2, h)]
                                                for j in cutlass.range_constexpr(8):
                                                    acc_v[(h, j)] = acc_v[(h, j)] * al
                                            for kk in cutlass.range_constexpr(8):
                                                for h in cutlass.range_constexpr(G):
                                                    pk = sP[(warp_idx * 8 + kk, h)]
                                                    for j in cutlass.range_constexpr(8):
                                                        acc_v[(h, j)] = acc_v[
                                                            (h, j)
                                                        ] + pk * vfrag[(kk, j)].to(
                                                            cutlass.Float32
                                                        )
                                            if pg + 1 < npages:
                                                for j in cutlass.range_constexpr(64):
                                                    kfrag[j] = kfrag_n[j]
                                            # (no loop-end barrier: the next page's post-score barrier orders every hazard, §14 lever 4)
                                        if cutlass.const_expr(self.profile):
                                            tu1 = cute.arch.globaltimer()
                                            if tidx == 0:
                                                self._prec(
                                                    mProf,
                                                    mPcnt,
                                                    pblk,
                                                    L * 32 + 29,
                                                    tu0,
                                                    tu1,
                                                    work,
                                                )
                                            tu0 = tu1
                                        sRedV = cute.make_tensor(
                                            sPart.iterator,
                                            cute.make_layout((4, 256), stride=(256, 1)),
                                        )
                                        for h in cutlass.range_constexpr(G):
                                            for j in cutlass.range_constexpr(8):
                                                sRedV[(warp_idx, lane * 8 + j)] = acc_v[
                                                    (h, j)
                                                ]
                                            self.epilog_sync_barrier.arrive_and_wait()
                                            acc_o[(h, 0)] = (
                                                sRedV[(0, 2 * tidx)]
                                                + sRedV[(1, 2 * tidx)]
                                                + sRedV[(2, 2 * tidx)]
                                                + sRedV[(3, 2 * tidx)]
                                            )
                                            acc_o[(h, 1)] = (
                                                sRedV[(0, 2 * tidx + 1)]
                                                + sRedV[(1, 2 * tidx + 1)]
                                                + sRedV[(2, 2 * tidx + 1)]
                                                + sRedV[(3, 2 * tidx + 1)]
                                            )
                                            self.epilog_sync_barrier.arrive_and_wait()
                                        for h in cutlass.range_constexpr(G):
                                            l_h = sStat[(1, h)]
                                            inv = cutlass.Float32(0.0)
                                            if l_h > cutlass.Float32(0.0):
                                                inv = cutlass.Float32(1.0) / l_h
                                            mPO[((pbase + h) * D + 2 * tidx,)] = (
                                                acc_o[(h, 0)] * inv
                                            )
                                            mPO[((pbase + h) * D + 2 * tidx + 1,)] = (
                                                acc_o[(h, 1)] * inv
                                            )
                                    l_h = cutlass.Float32(0.0)
                                    lse = cutlass.Float32(-1.0e30)
                                    if tidx < G:
                                        l_h = sStat[(1, tidx)]
                                        lse = cutlass.Float32(-1.0e30)
                                        if l_h > cutlass.Float32(0.0):
                                            lse = sStat[(0, tidx)] + cute.math.log2(l_h)
                                        mPL[(pbase + tidx,)] = lse
                                    cute.arch.fence_acq_rel_gpu()
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if tidx == 0:
                                        old = cute.arch.atomic_add(
                                            mCntL.iterator + (MERGE + mk),
                                            cutlass.Int32(1),
                                            sem="acq_rel",
                                            scope="gpu",
                                        )
                                        sFlag[0] = old
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if cutlass.const_expr(self.profile):
                                        tu1 = cute.arch.globaltimer()
                                        if tidx == 0:
                                            self._prec(
                                                mProf,
                                                mPcnt,
                                                pblk,
                                                L * 32 + 30,
                                                tu0,
                                                tu1,
                                                work,
                                            )
                                        tu0 = tu1
                                    if sFlag[0] == S - 1:
                                        cute.arch.fence_acq_rel_gpu()
                                        NGT = G * D // 64
                                        if tidx < NGT:
                                            self._wait_flag(mCntL, grow0 // 64 + tidx)
                                        self.epilog_sync_barrier.arrive_and_wait()
                                        mbase = mk * S * G
                                        # §15 lever C: heads spread across the 4 warps (lane = 8 dims of one head); per-element
                                        # arithmetic and split order unchanged (bitwise with the per-layer kernels)
                                        for rep_ in cutlass.range_constexpr(
                                            (G + 3) // 4
                                        ):
                                            hh = warp_idx + 4 * rep_
                                            if hh < G:
                                                lmax = cutlass.Float32(-1.0e30)
                                                for s2 in range(0, S, 1):
                                                    lv = mPL[(mbase + s2 * G + hh,)]
                                                    if lv > lmax:
                                                        lmax = lv
                                                wsum = cutlass.Float32(0.0)
                                                for j in cutlass.range_constexpr(8):
                                                    om[j] = cutlass.Float32(0.0)
                                                # §17 item 4: no branch around the loads -> the unrolled iterations' loads issue together
                                                # (an empty split's partial is exact zeros and its weight 0: om + 0*0 == om bitwise)
                                                for s2 in range(0, S, 1, unroll=4):
                                                    lv = mPL[(mbase + s2 * G + hh,)]
                                                    gpo = cute.make_tensor(
                                                        mPO.iterator
                                                        + cute.assume(
                                                            (mbase + s2 * G + hh) * D
                                                            + lane * 8,
                                                            divby=8,
                                                        ),
                                                        cute.make_layout((8,)),
                                                    )
                                                    cute.autovec_copy(gpo, pv8)
                                                    wgt = cutlass.Float32(0.0)
                                                    if lv > cutlass.Float32(-1.0e29):
                                                        wgt = cute.math.exp2(lv - lmax)
                                                    wsum = wsum + wgt
                                                    for j in cutlass.range_constexpr(8):
                                                        om[j] = om[j] + wgt * pv8[j]
                                                for j in cutlass.range_constexpr(8):
                                                    dd = lane * 8 + j
                                                    oo = (
                                                        (om[j] / wsum)
                                                        .to(cutlass.BFloat16)
                                                        .to(cutlass.Float32)
                                                    )
                                                    gg = mMz[
                                                        (grow0 + hh * D + dd, m, 0)
                                                    ].to(cutlass.Float32)
                                                    mAttn[
                                                        (qrow0 + hh * D + dd, m, 0)
                                                    ] = (
                                                        oo
                                                        / (
                                                            cutlass.Float32(1.0)
                                                            + cute.math.exp(-gg)
                                                        )
                                                    ).to(mAttn.element_type)
                                        cute.arch.fence_acq_rel_gpu()
                                        self.epilog_sync_barrier.arrive_and_wait()
                                        if cutlass.const_expr(self.profile):
                                            if tidx == 0:
                                                self._prec(
                                                    mProf,
                                                    mPcnt,
                                                    pblk,
                                                    L * 32 + 31,
                                                    tu0,
                                                    cute.arch.globaltimer(),
                                                    work,
                                                )
                                        if tidx == 0:
                                            cute.arch.atomic_add(
                                                mCntL.iterator + (HD + kh),
                                                cutlass.Int32(1),
                                                sem="acq_rel",
                                                scope="gpu",
                                            )
                    if cutlass.const_expr(self.profile):
                        ph = cutlass.Int32(0)
                        if work >= TAI:
                            ph = cutlass.Int32(1)
                        if work >= TAI + TB0I:
                            ph = cutlass.Int32(2)
                        if work >= TAI + TB0I + TB:
                            ph = cutlass.Int32(3)
                        if work >= TGDN:
                            ph = cutlass.Int32(4)
                        if work >= TGDN + TG:
                            ph = cutlass.Int32(5)
                        if tidx == 0:
                            self._prec(
                                mProf,
                                mPcnt,
                                pblk,
                                L * 32 + ph,
                                ti0,
                                cute.arch.globaltimer(),
                                work,
                            )
                    work = work + gdim
                if cutlass.const_expr(self.xb_prefetch):
                    phase_bar.arrive_and_wait()  # end of layer L: both warp roles are done with every stage (lever 5)

        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            # this CTA's descriptor set: box / swizzle / dtype from the atoms, base / shape / strides per layer
            tmm.init_tensormap_from_atom(tma_a1, tm_a1, self.num_mma_warps)
            tmm.init_tensormap_from_atom(tma_a2, tm_a2, self.num_mma_warps)
            tmm.init_tensormap_from_atom(tma_b2, tm_b2, self.num_mma_warps)
            tmm.init_tensormap_from_atom(tma_a3, tm_a3, self.num_mma_warps)
            tmm.init_tensormap_from_atom(tma_s3, tm_s3, self.num_mma_warps)
            tmm.init_tensormap_from_atom(tma_a4, tm_a4, self.num_mma_warps)
            tmm.init_tensormap_from_atom(tma_s4, tm_s4, self.num_mma_warps)
            tmm.fence_tensormap_initialization()
            N1x = mItab[(0, IT_N1)]
            K2x = mItab[(0, IT_K2)]
            k2h = K2x // KDA
            k2s = K2x // S_K
            k2hs = k2h // S_K
            gA1L = cute.make_tensor(
                cute.make_ptr(
                    self.a_dtype,
                    mPtab[(0, PT_WIN)],
                    AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout((N1x, K1 // KDA, c1), stride=(K1 // KDA, 1, c0)),
            )
            gA2L = cute.make_tensor(
                cute.make_ptr(
                    self.a_dtype,
                    mPtab[(0, PT_WOUT)],
                    AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout((K1, k2hs, S_K), stride=(k2h, 1, k2hs)),
            )
            if cutlass.const_expr(
                self.w16
            ):  # doubled activation view: 256-wide boxes over 128-K tiles, tile = split*kcount + k
                gB2L = cute.make_tensor(
                    mAttnF.iterator,
                    cute.make_layout((n_tok, 256, K2x // 128), stride=(K2x, 1, 128)),
                )
            else:
                gB2L = cute.make_tensor(
                    mAttnF.iterator,
                    cute.make_layout((n_tok, k2s, S_K), stride=(K2x, 1, k2s)),
                )
            tmm.update_tensormap(
                (gA1L, gA2L, gB2L),
                (tma_a1, tma_a2, tma_b2),
                (tm_a1, tm_a2, tm_b2),
                self.num_mma_warps,
                (None, None, None),
            )
            tmm.fence_tensormap_update(tm_a1)
            tmm.fence_tensormap_update(tm_a2)
            tmm.fence_tensormap_update(tm_b2)
            gA3L = cute.make_tensor(
                cute.make_ptr(
                    self.a_dtype,
                    mPtab[(0, PT_WGU)],
                    AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout((2 * I_, K1 // KDG, c1), stride=(K1 // KDG, 1, c0)),
            )
            gS3L = cute.make_tensor(
                cute.make_ptr(
                    self.s_dtype,
                    mPtab[(0, PT_SGU)],
                    AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout((2 * I_, K1 // 16, c1), stride=(K1 // 16, 1, c0)),
            )
            gA4L = cute.make_tensor(
                cute.make_ptr(
                    self.a_dtype,
                    mPtab[(0, PT_WDN)],
                    AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(
                    (K1, I_ // KDG // S_K, S_K), stride=(I_ // KDG, 1, I_ // KDG // S_K)
                ),
            )
            gS4L = cute.make_tensor(
                cute.make_ptr(
                    self.s_dtype,
                    mPtab[(0, PT_SDN)],
                    AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(
                    (K1, I_ // 16 // S_K, S_K), stride=(I_ // 16, 1, I_ // 16 // S_K)
                ),
            )
            tmm.update_tensormap(
                (gA3L, gS3L, gA4L, gS4L),
                (tma_a3, tma_s3, tma_a4, tma_s4),
                (tm_a3, tm_s3, tm_a4, tm_s4),
                self.num_mma_warps,
                (None, None, None, None),
            )
            tmm.fence_tensormap_update(tm_a3)
            tmm.fence_tensormap_update(tm_s3)
            tmm.fence_tensormap_update(tm_a4)
            tmm.fence_tensormap_update(tm_s4)
            fp8_ready = cutlass.Boolean(True)
            qblk = bidx * 2 + 1
            FDONE = CS - FOLD_CNT + 8
            work = cutlass.Int32(bidx)
            n_tile = cutlass.Int32(0)
            split = cutlass.Int32(0)
            kbf = cutlass.Int32(0)
            cnt_ = cutlass.Int32(0)
            idx = cutlass.Int32(0)
            bar = mix_pipeline.producer_get_barrier(producer_state)
            if cutlass.const_expr(self.fold):
                # §19.7 fc phase loads: weight tile (n_tile, split) x the cat(e|h) half; both descriptors are static
                TF = mItab[(0, IT_TF)]
                while work < TF:
                    n_tile = work // self.nsplit
                    split = work % self.nsplit
                    tAgF = tAgFA[(None, n_tile, None, split)]
                    tBgF = tBgFB[(None, 0, 0, None)]
                    kbf = split * k_cnt_f
                    producer_state.reset_count()
                    for _k_tile in range(0, k_cnt_f, 1, unroll=1):
                        mix_pipeline.producer_acquire(producer_state)
                        bar = mix_pipeline.producer_get_barrier(producer_state)
                        cnt_ = producer_state.count
                        idx = producer_state.index
                        cute.copy(
                            tma_fa,
                            tAgF[(None, cnt_)],
                            tAsA[(None, idx)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_fb,
                            tBgF[(None, kbf + cnt_)],
                            tBsB[(None, idx)],
                            tma_bar_ptr=bar,
                        )
                        mix_pipeline.producer_commit(producer_state)
                        producer_state.advance()
                    work = work + gdim
            xb_n = cutlass.Int32(
                0
            )  # in-projection weight stages of the next layer's first tile already in the ring
            xb_state = producer_state.clone()
            for L in range(0, NL, 1):
                # layer boundary: every TMA load of layer L-1 has been consumed (its down reducers are done),
                # so the descriptors can be rewritten; also the data dependency for this layer's activations
                if L > 0:
                    if cutlass.const_expr(self.l2_prefetch):
                        TA_n = mItab[(L, IT_TA)]
                        if bidx < TA_n:
                            nt_first = (bidx + mItab[(L, IT_KT0)]) % TA_n
                            self._pf_rows(
                                mPtab[(L, PT_WIN)],
                                nt_first * 64,
                                K1 * KB,
                                0,
                                256 * self.ab_stage,
                                lane,
                            )
                    tp0 = cutlass.Int64(0)
                    if cutlass.const_expr(self.profile):
                        tp0 = cute.arch.globaltimer()
                    if lane == 0:
                        need = L * NT2
                        seen = cute.arch.atomic_add(
                            mCnt.iterator + LDONE,
                            cutlass.Int32(0),
                            sem="acquire",
                            scope="gpu",
                        )
                        while seen < need:
                            seen = cute.arch.atomic_add(
                                mCnt.iterator + LDONE,
                                cutlass.Int32(0),
                                sem="acquire",
                                scope="gpu",
                            )
                    cute.arch.sync_warp()
                    cute.arch.fence_proxy("async.global")
                    if cutlass.const_expr(self.profile):
                        if lane == 0:
                            self._prec(
                                mProf,
                                mPcnt,
                                qblk,
                                L * 32 + 16,
                                tp0,
                                cute.arch.globaltimer(),
                                cutlass.Int32(0),
                            )
                if cutlass.const_expr(self.fold):
                    if (
                        L == 0
                    ):  # §19.7: layer 0's activations (xw0) come from the fc reducers
                        if lane == 0:
                            seen_f = cute.arch.atomic_add(
                                mCnt.iterator + FDONE,
                                cutlass.Int32(0),
                                sem="acquire",
                                scope="gpu",
                            )
                            while seen_f < NT2:
                                seen_f = cute.arch.atomic_add(
                                    mCnt.iterator + FDONE,
                                    cutlass.Int32(0),
                                    sem="acquire",
                                    scope="gpu",
                                )
                        cute.arch.sync_warp()
                        cute.arch.fence_proxy("async.global")
                if L > 0:
                    if not fp8_ready:  # this CTA had no MLP item in layer L-1 (no phase switch): fp8 descriptors now
                        N1x = mItab[(L, IT_N1)]
                        K2x = mItab[(L, IT_K2)]
                        k2h = K2x // KDA
                        k2s = K2x // S_K
                        k2hs = k2h // S_K
                        gA1L = cute.make_tensor(
                            cute.make_ptr(
                                self.a_dtype,
                                mPtab[(L, PT_WIN)],
                                AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(
                                (N1x, K1 // KDA, c1), stride=(K1 // KDA, 1, c0)
                            ),
                        )
                        gA2L = cute.make_tensor(
                            cute.make_ptr(
                                self.a_dtype,
                                mPtab[(L, PT_WOUT)],
                                AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout((K1, k2hs, S_K), stride=(k2h, 1, k2hs)),
                        )
                        if cutlass.const_expr(
                            self.w16
                        ):  # doubled activation view: 256-wide boxes over 128-K tiles, tile = split*kcount + k
                            gB2L = cute.make_tensor(
                                mAttnF.iterator,
                                cute.make_layout(
                                    (n_tok, 256, K2x // 128), stride=(K2x, 1, 128)
                                ),
                            )
                        else:
                            gB2L = cute.make_tensor(
                                mAttnF.iterator,
                                cute.make_layout(
                                    (n_tok, k2s, S_K), stride=(K2x, 1, k2s)
                                ),
                            )
                        tmm.update_tensormap(
                            (gA1L, gA2L, gB2L),
                            (tma_a1, tma_a2, tma_b2),
                            (tm_a1, tm_a2, tm_b2),
                            self.num_mma_warps,
                            (None, None, None),
                        )
                        tmm.fence_tensormap_update(tm_a1)
                        tmm.fence_tensormap_update(tm_a2)
                        tmm.fence_tensormap_update(tm_b2)
                    # NVFP4 descriptors: every MLP load of layer L-1 is consumed once LDONE says so (needed only at
                    # this layer's phase switch, so their latency is off the critical path)
                    gA3L = cute.make_tensor(
                        cute.make_ptr(
                            self.a_dtype,
                            mPtab[(L, PT_WGU)],
                            AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(
                            (2 * I_, K1 // KDG, c1), stride=(K1 // KDG, 1, c0)
                        ),
                    )
                    gS3L = cute.make_tensor(
                        cute.make_ptr(
                            self.s_dtype,
                            mPtab[(L, PT_SGU)],
                            AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(
                            (2 * I_, K1 // 16, c1), stride=(K1 // 16, 1, c0)
                        ),
                    )
                    gA4L = cute.make_tensor(
                        cute.make_ptr(
                            self.a_dtype,
                            mPtab[(L, PT_WDN)],
                            AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(
                            (K1, I_ // KDG // S_K, S_K),
                            stride=(I_ // KDG, 1, I_ // KDG // S_K),
                        ),
                    )
                    gS4L = cute.make_tensor(
                        cute.make_ptr(
                            self.s_dtype,
                            mPtab[(L, PT_SDN)],
                            AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(
                            (K1, I_ // 16 // S_K, S_K),
                            stride=(I_ // 16, 1, I_ // 16 // S_K),
                        ),
                    )
                    tmm.update_tensormap(
                        (gA3L, gS3L, gA4L, gS4L),
                        (tma_a3, tma_s3, tma_a4, tma_s4),
                        (tm_a3, tm_s3, tm_a4, tm_s4),
                        self.num_mma_warps,
                        (None, None, None, None),
                    )
                    tmm.fence_tensormap_update(tm_a3)
                    tmm.fence_tensormap_update(tm_s3)
                    tmm.fence_tensormap_update(tm_a4)
                    tmm.fence_tensormap_update(tm_s4)
                fp8_ready = cutlass.Boolean(False)
                ta_l = mItab[(L, IT_TA)]
                tail_l = cutlass.Int32(0)
                if cutlass.const_expr(self.tail_split):
                    if ta_l > gdim:
                        if (k_cnt_a % 2) == 0:
                            tail_l = ta_l % gdim
                total = (
                    ta_l
                    + tail_l
                    + mItab[(L, IT_TB0I)]
                    + mItab[(L, IT_TB)]
                    + 2 * NT2 * self.nsplit
                    + TG
                )

                work = cutlass.Int32(bidx)
                switched = cutlass.Boolean(False)
                act_mask = cutlass.Int32(0)
                prev_head = cutlass.Boolean(False)
                while work < total:
                    TA = mItab[(L, IT_TA)]
                    TAIL = cutlass.Int32(
                        0
                    )  # §17 item 2: tail tiles split in two K halves (TA > grid, even k-tile count)
                    if cutlass.const_expr(self.tail_split):
                        if gdim < TA:
                            if (k_cnt_a % 2) == 0:
                                TAIL = TA % gdim
                    TAI = (
                        TA + TAIL
                    )  # in-projection ITEM count (TA stays the tile / flag count)
                    TB0 = mItab[(L, IT_TB0)]
                    TB0I = mItab[(L, IT_TB0I)]
                    TBKV = mItab[
                        (L, IT_TBKV)
                    ]  # §18: kv-write items (attention) = sequences x kv heads
                    TB = mItab[(L, IT_TB)]
                    k_cnt_c = mItab[(L, IT_KCC)]
                    KT0 = mItab[(L, IT_KT0)]
                    HD_NEED = mItab[(L, IT_HDN)]
                    TC = NT2 * self.nsplit
                    TGDN = TAI + TB0I + TB + TC
                    HD = TA
                    NH = cutlass.Int32(self.hv)
                    GDONE_G = TA + self.hv + 1 + NT2 + TB0 + 32
                    GDONE_A = TA + self.hkv + 1 + NT2 + TBKV + 2 * TB0 + 32
                    GDONE = GDONE_G
                    if mItab[(L, IT_KIND)] == 1:
                        GDONE = GDONE_A
                        NH = cutlass.Int32(self.hkv)
                    DSPLIT0 = GDONE + 32
                    GACT = DSPLIT0 + NT2 + 32
                    mCntL = cute.make_tensor(
                        mCnt.iterator + L * CS, cute.make_layout((BIG,))
                    )
                    if prev_head:
                        core_bar.arrive_and_wait()
                        prev_head = cutlass.Boolean(False)
                    if work >= TGDN:
                        is_g = work < TGDN + TG
                        issued = cutlass.Boolean(False)
                        if not switched:
                            phase_bar.arrive_and_wait()
                            if L + 1 < NL:
                                N1x = mItab[(L + 1, IT_N1)]
                                K2x = mItab[(L + 1, IT_K2)]
                                k2h = K2x // KDA
                                k2s = K2x // S_K
                                k2hs = k2h // S_K
                                gA1L = cute.make_tensor(
                                    cute.make_ptr(
                                        self.a_dtype,
                                        mPtab[(L + 1, PT_WIN)],
                                        AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout(
                                        (N1x, K1 // KDA, c1), stride=(K1 // KDA, 1, c0)
                                    ),
                                )
                                gA2L = cute.make_tensor(
                                    cute.make_ptr(
                                        self.a_dtype,
                                        mPtab[(L + 1, PT_WOUT)],
                                        AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout(
                                        (K1, k2hs, S_K), stride=(k2h, 1, k2hs)
                                    ),
                                )
                                if cutlass.const_expr(
                                    self.w16
                                ):  # doubled activation view: 256-wide boxes over 128-K tiles, tile = split*kcount + k
                                    gB2L = cute.make_tensor(
                                        mAttnF.iterator,
                                        cute.make_layout(
                                            (n_tok, 256, K2x // 128),
                                            stride=(K2x, 1, 128),
                                        ),
                                    )
                                else:
                                    gB2L = cute.make_tensor(
                                        mAttnF.iterator,
                                        cute.make_layout(
                                            (n_tok, k2s, S_K), stride=(K2x, 1, k2s)
                                        ),
                                    )
                                tmm.update_tensormap(
                                    (gA1L, gA2L, gB2L),
                                    (tma_a1, tma_a2, tma_b2),
                                    (tm_a1, tm_a2, tm_b2),
                                    self.num_mma_warps,
                                    (None, None, None),
                                )
                                tmm.fence_tensormap_update(tm_a1)
                                tmm.fence_tensormap_update(tm_a2)
                                tmm.fence_tensormap_update(tm_b2)
                                fp8_ready = cutlass.Boolean(True)
                            mlp_producer_state.reset_count()
                            n_tile0 = work - TGDN
                            npre = cutlass.Int32(0)
                            pre_state = mlp_producer_state.clone()
                            if cutlass.const_expr(self.prefetch):
                                if is_g:
                                    npre = cutlass.min(
                                        cutlass.Int32(self.mlp_stage), k_cnt_g
                                    )
                                    tAg3p = tAgA3[(None, n_tile0, None, 0)]
                                    tSg3p = tSgS3[(None, n_tile0, None, 0)]
                                    for _k_tile in range(0, npre, 1, unroll=1):
                                        mlp_pipeline.producer_acquire(
                                            mlp_producer_state
                                        )
                                        bar = mlp_pipeline.producer_get_barrier(
                                            mlp_producer_state
                                        )
                                        cnt_ = mlp_producer_state.count
                                        idx = mlp_producer_state.index
                                        cute.copy(
                                            tma_a3,
                                            tAg3p[(None, cnt_)],
                                            tAsA4[(None, idx)],
                                            tma_bar_ptr=bar,
                                            tma_desc_ptr=dp_a3,
                                        )
                                        if cutlass.const_expr(not self.w16):
                                            cute.copy(
                                                tma_s3,
                                                tSg3p[(None, cnt_)],
                                                tSsS[(None, idx)],
                                                tma_bar_ptr=bar,
                                                tma_desc_ptr=dp_s3,
                                            )
                                        mlp_producer_state.advance()
                            tq0 = cutlass.Int64(0)
                            if cutlass.const_expr(self.profile):
                                tq0 = cute.arch.globaltimer()
                            if lane == 0:
                                seen0 = cute.arch.atomic_add(
                                    mCntL.iterator + GDONE,
                                    cutlass.Int32(0),
                                    sem="acquire",
                                    scope="gpu",
                                )
                                while seen0 < 1:
                                    seen0 = cute.arch.atomic_add(
                                        mCntL.iterator + GDONE,
                                        cutlass.Int32(0),
                                        sem="acquire",
                                        scope="gpu",
                                    )
                            cute.arch.sync_warp()
                            cute.arch.fence_proxy("async.global")
                            if cutlass.const_expr(self.profile):
                                if lane == 0:
                                    self._prec(
                                        mProf,
                                        mPcnt,
                                        qblk,
                                        L * 32 + 17,
                                        tq0,
                                        cute.arch.globaltimer(),
                                        work,
                                    )
                            if cutlass.const_expr(self.prefetch):
                                if is_g:
                                    tBg3p = (
                                        tBgB3[(None, 0, 0, None)]
                                        if self.w16
                                        else tBgB3[(None, 0, None, 0)]
                                    )
                                    for _k_tile in range(0, npre, 1, unroll=1):
                                        bar = mlp_pipeline.producer_get_barrier(
                                            pre_state
                                        )
                                        cnt_ = pre_state.count
                                        idx = pre_state.index
                                        cute.copy(
                                            tma_b3,
                                            tBg3p[(None, cnt_)],
                                            tBsB4[(None, idx)],
                                            tma_bar_ptr=bar,
                                        )
                                        mlp_pipeline.producer_commit(pre_state)
                                        pre_state.advance()
                                    tAg3p2 = tAgA3[(None, n_tile0, None, 0)]
                                    tSg3p2 = tSgS3[(None, n_tile0, None, 0)]
                                    for _k_tile in range(npre, k_cnt_g, 1, unroll=1):
                                        mlp_pipeline.producer_acquire(
                                            mlp_producer_state
                                        )
                                        bar = mlp_pipeline.producer_get_barrier(
                                            mlp_producer_state
                                        )
                                        cnt_ = mlp_producer_state.count
                                        idx = mlp_producer_state.index
                                        cute.copy(
                                            tma_a3,
                                            tAg3p2[(None, cnt_)],
                                            tAsA4[(None, idx)],
                                            tma_bar_ptr=bar,
                                            tma_desc_ptr=dp_a3,
                                        )
                                        if cutlass.const_expr(not self.w16):
                                            cute.copy(
                                                tma_s3,
                                                tSg3p2[(None, cnt_)],
                                                tSsS[(None, idx)],
                                                tma_bar_ptr=bar,
                                                tma_desc_ptr=dp_s3,
                                            )
                                        cute.copy(
                                            tma_b3,
                                            tBg3p[(None, cnt_)],
                                            tBsB4[(None, idx)],
                                            tma_bar_ptr=bar,
                                        )
                                        mlp_pipeline.producer_commit(mlp_producer_state)
                                        mlp_producer_state.advance()
                            switched = cutlass.Boolean(True)
                            if cutlass.const_expr(self.prefetch):
                                if is_g:
                                    issued = cutlass.Boolean(True)
                        if is_g:
                            if not issued:
                                n_tile = work - TGDN
                                tAg = tAgA3[(None, n_tile, None, 0)]
                                tSg = tSgS3[(None, n_tile, None, 0)]
                                tBg = (
                                    tBgB3[(None, 0, 0, None)]
                                    if self.w16
                                    else tBgB3[(None, 0, None, 0)]
                                )
                                mlp_producer_state.reset_count()
                                for _k_tile in range(0, k_cnt_g, 1, unroll=1):
                                    mlp_pipeline.producer_acquire(mlp_producer_state)
                                    bar = mlp_pipeline.producer_get_barrier(
                                        mlp_producer_state
                                    )
                                    cnt_ = mlp_producer_state.count
                                    idx = mlp_producer_state.index
                                    cute.copy(
                                        tma_a3,
                                        tAg[(None, cnt_)],
                                        tAsA4[(None, idx)],
                                        tma_bar_ptr=bar,
                                        tma_desc_ptr=dp_a3,
                                    )
                                    if cutlass.const_expr(not self.w16):
                                        cute.copy(
                                            tma_s3,
                                            tSg[(None, cnt_)],
                                            tSsS[(None, idx)],
                                            tma_bar_ptr=bar,
                                            tma_desc_ptr=dp_s3,
                                        )
                                    cute.copy(
                                        tma_b3,
                                        tBg[(None, cnt_)],
                                        tBsB4[(None, idx)],
                                        tma_bar_ptr=bar,
                                    )
                                    mlp_pipeline.producer_commit(mlp_producer_state)
                                    mlp_producer_state.advance()
                        else:
                            jt = work - TGDN - TG
                            n_tile = jt // self.nsplit
                            split = jt % self.nsplit
                            tAg = tAgA4[(None, n_tile, None, split)]
                            tSg = tSgS4[(None, n_tile, None, split)]
                            tBg = (
                                tBgB4[(None, 0, 0, None)]
                                if self.w16
                                else tBgB4[(None, 0, None, split)]
                            )
                            kb4 = (
                                split * k_cnt_d if self.w16 else cutlass.Int32(0)
                            )  # doubled view: tile = split*kcount + k
                            mlp_producer_state.reset_count()
                            k_from4 = cutlass.Int32(0)
                            pre4 = mlp_producer_state.clone()
                            if ((act_mask >> split) & 1) == 0:
                                # weight + scale stages of this item go out BEFORE the gate_up-done spin (§15 pre-issue)
                                npre4 = cutlass.min(
                                    cutlass.Int32(self.mlp_stage), k_cnt_d
                                )
                                for _k_tile in range(0, npre4, 1, unroll=1):
                                    mlp_pipeline.producer_acquire(mlp_producer_state)
                                    bar = mlp_pipeline.producer_get_barrier(
                                        mlp_producer_state
                                    )
                                    cnt_ = mlp_producer_state.count
                                    idx = mlp_producer_state.index
                                    cute.copy(
                                        tma_a4,
                                        tAg[(None, cnt_)],
                                        tAsA4[(None, idx)],
                                        tma_bar_ptr=bar,
                                        tma_desc_ptr=dp_a4,
                                    )
                                    if cutlass.const_expr(not self.w16):
                                        cute.copy(
                                            tma_s4,
                                            tSg[(None, cnt_)],
                                            tSsS[(None, idx)],
                                            tma_bar_ptr=bar,
                                            tma_desc_ptr=dp_s4,
                                        )
                                    mlp_producer_state.advance()
                                if cutlass.const_expr(self.l2_prefetch):
                                    if cutlass.const_expr(self.w16):
                                        self._pf_rows(
                                            mPtab[(L, PT_WDN)],
                                            n_tile * 64,
                                            2 * I_,
                                            split * (2 * I_ // S_K),
                                            256 * self.ab_stage,
                                            lane,
                                        )
                                    else:
                                        self._pf_rows(
                                            mPtab[(L, PT_WDN)],
                                            n_tile * 64,
                                            I_ // 2,
                                            split * (I_ // 2 // S_K),
                                            128 * self.mlp_stage,
                                            lane,
                                        )
                                        self._pf_rows(
                                            mPtab[(L, PT_SDN)],
                                            n_tile * 64,
                                            I_ // 16,
                                            split * (I_ // 16 // S_K),
                                            16 * self.mlp_stage,
                                            lane,
                                        )
                                ta0 = cutlass.Int64(0)
                                if cutlass.const_expr(self.profile):
                                    ta0 = cute.arch.globaltimer()
                                if lane == 0:
                                    need_g = TG // self.nsplit
                                    seen = cute.arch.atomic_add(
                                        mCntL.iterator + (GACT + split),
                                        cutlass.Int32(0),
                                        sem="acquire",
                                        scope="gpu",
                                    )
                                    while seen < need_g:
                                        seen = cute.arch.atomic_add(
                                            mCntL.iterator + (GACT + split),
                                            cutlass.Int32(0),
                                            sem="acquire",
                                            scope="gpu",
                                        )
                                cute.arch.sync_warp()
                                cute.arch.fence_proxy("async.global")
                                if cutlass.const_expr(self.profile):
                                    if lane == 0:
                                        self._prec(
                                            mProf,
                                            mPcnt,
                                            qblk,
                                            L * 32 + 14,
                                            ta0,
                                            cute.arch.globaltimer(),
                                            work,
                                        )
                                act_mask = act_mask | (cutlass.Int32(1) << split)
                                for _k_tile in range(0, npre4, 1, unroll=1):
                                    bar = mlp_pipeline.producer_get_barrier(pre4)
                                    cute.copy(
                                        tma_b4,
                                        tBg[(None, kb4 + pre4.count)],
                                        tBsB4[(None, pre4.index)],
                                        tma_bar_ptr=bar,
                                    )
                                    mlp_pipeline.producer_commit(pre4)
                                    pre4.advance()
                                k_from4 = npre4
                            for _k_tile in range(k_from4, k_cnt_d, 1, unroll=1):
                                mlp_pipeline.producer_acquire(mlp_producer_state)
                                bar = mlp_pipeline.producer_get_barrier(
                                    mlp_producer_state
                                )
                                cnt_ = mlp_producer_state.count
                                idx = mlp_producer_state.index
                                cute.copy(
                                    tma_a4,
                                    tAg[(None, cnt_)],
                                    tAsA4[(None, idx)],
                                    tma_bar_ptr=bar,
                                    tma_desc_ptr=dp_a4,
                                )
                                if cutlass.const_expr(not self.w16):
                                    cute.copy(
                                        tma_s4,
                                        tSg[(None, cnt_)],
                                        tSsS[(None, idx)],
                                        tma_bar_ptr=bar,
                                        tma_desc_ptr=dp_s4,
                                    )
                                cute.copy(
                                    tma_b4,
                                    tBg[(None, kb4 + cnt_)],
                                    tBsB4[(None, idx)],
                                    tma_bar_ptr=bar,
                                )
                                mlp_pipeline.producer_commit(mlp_producer_state)
                                mlp_producer_state.advance()
                    else:
                        is_a = work < TAI
                        is_core = (work >= TAI) and (work < TAI + TB0I + TB)
                        if is_core:
                            if work >= TAI + TB0I:
                                if cutlass.const_expr(self.mma_attn):
                                    prev_head = cutlass.Boolean(
                                        True
                                    )  # GDN head / attention split item: consumers arrive on core_bar
                                else:
                                    if mItab[(L, IT_KIND)] == 0:
                                        prev_head = cutlass.Boolean(
                                            True
                                        )  # a GDN head item: consumers will arrive on core_bar
                        if is_a:
                            n_tile_a = (work + KT0) % TA
                            k_off_a = cutlass.Int32(0)
                            k_end_a = cutlass.Int32(k_cnt_a)
                            if (
                                work >= TA - TAIL
                            ):  # §17 item 2: one K half of a tail tile
                                ja = work - (TA - TAIL)
                                n_tile_a = (TA - TAIL + ja // 2 + KT0) % TA
                                k_off_a = (ja % 2) * (k_cnt_a // 2)
                                k_end_a = k_cnt_a // 2
                            tAg = tAgA1[(None, n_tile_a, None, 0)]
                            tBg = (
                                tBgB1[(None, 0, 0, None)]
                                if self.w16
                                else tBgB1[(None, 0, None, 0)]
                            )
                            k_from = cutlass.Int32(0)
                            if (
                                xb_n > 0
                            ):  # weight stages issued across the boundary: add the activation copies, commit
                                for _k_tile in range(0, xb_n, 1, unroll=1):
                                    bar = mix_pipeline.producer_get_barrier(xb_state)
                                    cute.copy(
                                        tma_b1,
                                        tBg[(None, xb_state.count)],
                                        tBsB[(None, xb_state.index)],
                                        tma_bar_ptr=bar,
                                    )
                                    mix_pipeline.producer_commit(xb_state)
                                    xb_state.advance()
                                k_from = xb_n
                                xb_n = cutlass.Int32(0)
                            else:
                                producer_state.reset_count()
                            for _k_tile in range(k_from, k_end_a, 1, unroll=1):
                                mix_pipeline.producer_acquire(producer_state)
                                bar = mix_pipeline.producer_get_barrier(producer_state)
                                cnt_ = producer_state.count
                                idx = producer_state.index
                                cute.copy(
                                    tma_a1,
                                    tAg[(None, cnt_ + k_off_a)],
                                    tAsA[(None, idx)],
                                    tma_bar_ptr=bar,
                                    tma_desc_ptr=dp_a1,
                                )
                                cute.copy(
                                    tma_b1,
                                    tBg[(None, cnt_ + k_off_a)],
                                    tBsB[(None, idx)],
                                    tma_bar_ptr=bar,
                                )
                                mix_pipeline.producer_commit(producer_state)
                                producer_state.advance()
                        elif not is_core:
                            jt = work - TAI - TB0I - TB
                            n_tile = jt // self.nsplit
                            split = jt % self.nsplit
                            tAg2 = tAgA2[(None, n_tile, None, split)]
                            tBg2 = (
                                tBgB2[(None, 0, 0, None)]
                                if self.w16
                                else tBgB2[(None, 0, None, split)]
                            )
                            kb2 = split * k_cnt_c if self.w16 else cutlass.Int32(0)
                            producer_state.reset_count()
                            pre2 = producer_state.clone()
                            npre2 = cutlass.min(cutlass.Int32(self.ab_stage), k_cnt_c)
                            for _k_tile in range(
                                0, npre2, 1, unroll=1
                            ):  # weight stages BEFORE the heads-done spin (§15 pre-issue)
                                mix_pipeline.producer_acquire(producer_state)
                                bar = mix_pipeline.producer_get_barrier(producer_state)
                                cute.copy(
                                    tma_a2,
                                    tAg2[(None, producer_state.count)],
                                    tAsA[(None, producer_state.index)],
                                    tma_bar_ptr=bar,
                                    tma_desc_ptr=dp_a2,
                                )
                                producer_state.advance()
                            if cutlass.const_expr(self.l2_prefetch):
                                K2p = mItab[(L, IT_K2)]
                                self._pf_rows(
                                    mPtab[(L, PT_WOUT)],
                                    n_tile * 64,
                                    K2p * KB,
                                    split * (K2p // S_K) * KB,
                                    256 * self.ab_stage,
                                    lane,
                                )
                            th0 = cutlass.Int64(0)
                            if cutlass.const_expr(self.profile):
                                th0 = cute.arch.globaltimer()
                            if lane == 0:
                                h_lo = (split * NH) // self.nsplit
                                h_hi = (
                                    (split + 1) * NH + self.nsplit - 1
                                ) // self.nsplit
                                for hh in range(h_lo, h_hi, 1):
                                    seen0 = cute.arch.atomic_add(
                                        mCntL.iterator + (HD + hh),
                                        cutlass.Int32(0),
                                        sem="acquire",
                                        scope="gpu",
                                    )
                                    while (
                                        seen0 < HD_NEED
                                    ):  # per-head arrivals: tokens (plain / attention) or sequences (GDN verify)
                                        seen0 = cute.arch.atomic_add(
                                            mCntL.iterator + (HD + hh),
                                            cutlass.Int32(0),
                                            sem="acquire",
                                            scope="gpu",
                                        )
                            cute.arch.sync_warp()
                            cute.arch.fence_proxy("async.global")
                            if cutlass.const_expr(self.profile):
                                if lane == 0:
                                    self._prec(
                                        mProf,
                                        mPcnt,
                                        qblk,
                                        L * 32 + 13,
                                        th0,
                                        cute.arch.globaltimer(),
                                        work,
                                    )
                            for _k_tile in range(0, npre2, 1, unroll=1):
                                bar = mix_pipeline.producer_get_barrier(pre2)
                                cute.copy(
                                    tma_b2,
                                    tBg2[(None, kb2 + pre2.count)],
                                    tBsB[(None, pre2.index)],
                                    tma_bar_ptr=bar,
                                    tma_desc_ptr=dp_b2,
                                )
                                mix_pipeline.producer_commit(pre2)
                                pre2.advance()
                            for _k_tile in range(npre2, k_cnt_c, 1, unroll=1):
                                mix_pipeline.producer_acquire(producer_state)
                                bar = mix_pipeline.producer_get_barrier(producer_state)
                                cnt_ = producer_state.count
                                idx = producer_state.index
                                cute.copy(
                                    tma_a2,
                                    tAg2[(None, cnt_)],
                                    tAsA[(None, idx)],
                                    tma_bar_ptr=bar,
                                    tma_desc_ptr=dp_a2,
                                )
                                cute.copy(
                                    tma_b2,
                                    tBg2[(None, kb2 + cnt_)],
                                    tBsB[(None, idx)],
                                    tma_bar_ptr=bar,
                                    tma_desc_ptr=dp_b2,
                                )
                                mix_pipeline.producer_commit(producer_state)
                                producer_state.advance()
                    work = work + gdim
                if prev_head:
                    core_bar.arrive_and_wait()
                    prev_head = cutlass.Boolean(False)
                if cutlass.const_expr(self.xb_prefetch):
                    phase_bar.arrive_and_wait()  # consumers are done with every stage of layer L
                    if L + 1 < NL:
                        TA_x = mItab[(L + 1, IT_TA)]
                        if bidx < TA_x:
                            if not fp8_ready:
                                N1x = mItab[(L + 1, IT_N1)]
                                K2x = mItab[(L + 1, IT_K2)]
                                k2h = K2x // KDA
                                k2s = K2x // S_K
                                k2hs = k2h // S_K
                                gA1L = cute.make_tensor(
                                    cute.make_ptr(
                                        self.a_dtype,
                                        mPtab[(L + 1, PT_WIN)],
                                        AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout(
                                        (N1x, K1 // KDA, c1), stride=(K1 // KDA, 1, c0)
                                    ),
                                )
                                gA2L = cute.make_tensor(
                                    cute.make_ptr(
                                        self.a_dtype,
                                        mPtab[(L + 1, PT_WOUT)],
                                        AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout(
                                        (K1, k2hs, S_K), stride=(k2h, 1, k2hs)
                                    ),
                                )
                                if cutlass.const_expr(
                                    self.w16
                                ):  # doubled activation view: 256-wide boxes over 128-K tiles, tile = split*kcount + k
                                    gB2L = cute.make_tensor(
                                        mAttnF.iterator,
                                        cute.make_layout(
                                            (n_tok, 256, K2x // 128),
                                            stride=(K2x, 1, 128),
                                        ),
                                    )
                                else:
                                    gB2L = cute.make_tensor(
                                        mAttnF.iterator,
                                        cute.make_layout(
                                            (n_tok, k2s, S_K), stride=(K2x, 1, k2s)
                                        ),
                                    )
                                tmm.update_tensormap(
                                    (gA1L, gA2L, gB2L),
                                    (tma_a1, tma_a2, tma_b2),
                                    (tm_a1, tm_a2, tm_b2),
                                    self.num_mma_warps,
                                    (None, None, None),
                                )
                                tmm.fence_tensormap_update(tm_a1)
                                tmm.fence_tensormap_update(tm_a2)
                                tmm.fence_tensormap_update(tm_b2)
                                fp8_ready = cutlass.Boolean(True)
                            nt_x = (bidx + mItab[(L + 1, IT_KT0)]) % TA_x
                            tAgx = tAgA1[(None, nt_x, None, 0)]
                            producer_state.reset_count()
                            xb_state = producer_state.clone()
                            xb_n = cutlass.min(cutlass.Int32(self.ab_stage), k_cnt_a)
                            for _k_tile in range(0, xb_n, 1, unroll=1):
                                mix_pipeline.producer_acquire(producer_state)
                                bar = mix_pipeline.producer_get_barrier(producer_state)
                                cute.copy(
                                    tma_a1,
                                    tAgx[(None, producer_state.count)],
                                    tAsA[(None, producer_state.index)],
                                    tma_bar_ptr=bar,
                                    tma_desc_ptr=dp_a1,
                                )
                                producer_state.advance()
            mix_pipeline.producer_tail(producer_state)
            mlp_pipeline.producer_tail(mlp_producer_state)
        return
