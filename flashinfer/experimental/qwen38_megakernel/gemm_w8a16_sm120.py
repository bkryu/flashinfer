# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# Qwen3.8-27B one-launch decoder on SM120 (FlashInfer experimental track).
"""cutlass_dsl decode W8A16 GEMM for SM120 (Phase C.4): fp8 [N,K] weight
streamed as the swap-AB MMA A operand, dequantized to bf16 in registers.

Problem: y[M,N] = x[M,K](bf16) @ dequant(W)^T, W fp8 e4m3 stored
CHECKPOINT-NATIVE [N,K] (K contiguous, bryu ruling 2026-09-08), one fp32
scale per fused shard applied in the epilogue. Decode M <= 16.

Structure (subclass of the B.1 kernel `Sm120DenseGemm`, swap-AB form):
  C^T[N,M] = W[N,K] @ x^T[K,M]
  A operand = W presented as a 16-BIT CONTAINER [N, K/2] (each bf16 slot
             holds two fp8 bytes), so the unchanged bf16 machinery (TMA,
             128B-swizzled smem, ldmatrix.x4) moves the fp8 bytes;
  B operand = x [M,K] bf16, natural order in gmem/smem (TMA);
  C         = out^T [N,M] via the TMA store (m-major).

The container trick, no weight repack: the bf16 m16n8k16 A fragment gives
lane t (j' = t%4), for row r, container columns 2j'+e (e = 0,1) and 2j'+8+e,
i.e. fp8 bytes at k = 4j'+2e+{0,1} and 16+4j'+2e+{0,1}. Converting one
16-bit container element to bf16x2 therefore yields the MMA A register for
the k PAIR (4j'+2e, 4j'+2e+1). Feed that as the A register of k-slot
(kh = e) and the MMA's k-slot order inside a k16 block becomes
    slot (e_log, kh)  <->  memory k = 4j' + 2kh + e_log
for BOTH operands. The A side is free (that IS what ldmatrix hands us); the
B side needs the activation fragment for slot (e_log, kh) to hold
x[4j' + 2kh + e_log] — 4 consecutive bf16 per lane, i.e. one 64-bit smem
load instead of ldmatrix. We express it as a K-PERMUTED VIEW of the
activation smem tile (`cute.composition` with the layout
((2,4,2), K/16):((1,4,2), 16) on the K mode) partitioned by the same
tiled MMA, copied with `cute.autovec_copy`. One container k-block
(32 fp8) = two logical k16 blocks; the weight stays exactly as shipped.
Register work per A fragment: fp8x2 -> bf16x2 conversion plus a
compile-time swap of two 32-bit registers (row-half / k-half order).

Bytes per stage: A = TILE_N*TILE_K fp8, B = 16*TILE_K bf16 (tiny), so the
stage budget is ~2x the bf16 swap-AB kernel's at the same tile.

Only the constructor and the attribute setup are used here (as a base of the one-launch decoder); the
standalone swap-AB GEMM launch (`__call__` / `kernel`) was removed from this copy.
"""

from __future__ import annotations

import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils

from .gemm_sm120 import Sm120DenseGemm

MAX_SHARDS = 8


class W8A16SwapSm120(Sm120DenseGemm):
    """Swap-AB fp8-weight / bf16-activation GEMM, see module docstring.

    tile_shape_mnk is (TILE_N, 16, TILE_K) in LOGICAL (fp8) K units.
    """

    def __init__(
        self,
        acc_dtype,
        tile_shape_mnk,
        epi_stage=2,
        atom_layout=(4, 1, 1),
        occupancy=1,
        nsh=1,
        pdl=0,
    ):
        if tile_shape_mnk[2] % 32:
            raise ValueError(
                "TILE_K must be a multiple of 32 (one container k-block = 32 fp8)"
            )
        super().__init__(
            acc_dtype, tile_shape_mnk, epi_stage, atom_layout, occupancy, pdl=pdl
        )
        self.nsh = nsh
        # container tile for the A operand: half the K extent in bf16 slots
        self.a_tile_mnk = (tile_shape_mnk[0], tile_shape_mnk[1], tile_shape_mnk[2] // 2)

    # ------------------------------------------------------------------ setup
    def _setup_attributes(self):
        self.mma_inst_mnk = (16, 8, 16)
        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.b_dtype, self.acc_dtype, self.mma_inst_mnk
        )
        tC = cute.make_layout(self.atom_layout)
        permutation_mnk = (
            self.atom_layout[0] * self.mma_inst_mnk[0],
            self.atom_layout[1] * self.mma_inst_mnk[1] * 2,
            self.atom_layout[2] * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.num_mcast_ctas_a = 1
        self.num_mcast_ctas_b = 1
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=False
        )
        # stage budget with the fp8-sized A tile
        c_bytes = (
            cute.size(self.epi_tile) * self.c_dtype.width // 8 * self.epi_stage_cfg
        )
        a_bytes = (
            self.a_tile_mnk[0] * self.a_tile_mnk[2] * 2
        )  # bf16 container = fp8 bytes
        b_bytes = self.tile_shape_mnk[1] * self.tile_shape_mnk[2] * 2
        self.ab_stage = (
            (self.smem_capacity - self.occupancy * 1024) // self.occupancy
            - 1024
            - c_bytes
        ) // (a_bytes + b_bytes)
        self.epi_stage = self.epi_stage_cfg
        if self.ab_stage < 1:
            raise ValueError("config does not fit SM120 SMEM")
        self.ab_stage = min(
            self.ab_stage, 8
        )  # deeper buys nothing at decode; keeps mbarrier count sane
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            self.a_layout, self.a_tile_mnk, self.a_dtype, self.ab_stage
        )
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.b_layout, self.tile_shape_mnk, self.b_dtype, self.ab_stage
        )
        self.epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.epi_stage
        )

    # ------------------------------------------------------------------ host

    # ------------------------------------------------------------------ device
