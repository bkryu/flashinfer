# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""W4A16 dense GEMM for Blackwell (SM100/103/120/121) -- mm_fp4 layout.

Sibling of ``dense_gemm_w4a16_blackwell.py`` (the Marlin-repack variant)
that consumes the SAME input format as ``flashinfer.gemm.mm_fp4`` -- no
host-side weight repack:

  - B:    ``(N, K/2)`` uint8 packed FP4.  Byte at offset n*K/2+k_half
          carries FP4 code K=2*k_half (low nibble) + K=2*k_half+1
          (high nibble).  K is stride-1; N stride = K/2.
  - B_sf: 1-D byte buffer holding the 128x4-swizzled FP8-E4M3 SF.
          Address per (n, k_sf) is::
              ((n/128) * (K_sf/4) + k_sf/4) * 512
              + (n%32)*16 + ((n%128)/32)*4 + (k_sf%4)
  - alpha: ``(1,) float32`` scalar.

Architecture matches the sibling Marlin kernel exactly: per-thread,
per-K-block, in-register FP4 decode -> bf16 -> write straight to
``tCrB``.  No bf16 SMEM materialization (this is what kept the b12x
variant pinned at ~70 us on B200).

Only two things change vs the Marlin sibling:

  1. B TMA loads raw uint8 ``(tile_N, tile_K/2)`` bytes into sB_raw
     instead of Marlin-tied int32.  ldmatrix is NOT used on B; the
     dequant function reads raw bytes per-thread from sB_raw.
  2. SF is gmem-direct (not TMA'd to SMEM); each thread computes its
     SF address via the 128x4 swizzle formula and issues ld.global.nc.u8.
     (The Marlin kernel TMA's a linear SF tile; we can't reuse that
     here because the on-disk SF is already swizzled.)
"""

from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Float32, Int32, Int64, Uint32

from ...cute_dsl.fp4_common import (
    cvt_e4m3_to_f32_via_f16,
    f16x2_to_f32x2,
    fp4_decode_4bytes,
    fp4_decode_2,
    get_smem_ptr_as_int32,
    half2_mul,
    ld_shared_v2_u32,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass._mlir.extras import types as T
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def cvt_bf16x2_to_f16x2_via_f32(
    packed_bf16x2: Uint32, *, loc=None, ip=None
) -> Uint32:
    """Packed bf16x2 (u32) -> f16x2 (u32) via f32 intermediate.

    Single inline-asm block: mov.b32 unpack + 2x cvt.f32.bf16 + 1x
    cvt.rn.f16x2.f32 packed.  Total 4 PTX instructions for the pair.

    The direct `cvt.rn.f16x2.bf16x2` (a single PTX instr) is NOT
    supported by ptxas on sm_100/sm_120 with CUDA 13.1 -- confirmed
    via scratch_cvt_bf16x2_to_f16x2_probe.py.  When the toolchain
    adds support we can swap this body for the single-instruction
    direct cvt.

    Compares to cute's per-element scalar `.to(Float16)` lowering:
    that emits ~2 scalar instructions per element (cvt.f32.bf16 +
    cvt.rn.f16.f32), or 4 instr per pair -- same count, but ours
    keeps the second cvt packed."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(packed_bf16x2).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 b_lo, b_hi;
                .reg .f32 f_lo, f_hi;
                mov.b32 {b_lo, b_hi}, $1;
                cvt.f32.bf16 f_lo, b_lo;
                cvt.f32.bf16 f_hi, b_hi;
                cvt.rn.f16x2.f32 $0, f_hi, f_lo;
            }
            """,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f16x2_unpack(
    packed_h2: Uint32, *, loc=None, ip=None
) -> Tuple["cutlass.Float16", "cutlass.Float16"]:
    """Unpack f16x2 (u32) into (f16_lo, f16_hi).  Free at HW level --
    just a register-rename (mov.b32 {h_lo, h_hi}, packed).  Used by the
    fp16-MMA path to bypass the f16->f32->bf16 cvt chain that the
    default bf16-MMA path needs after hmul2."""
    from cutlass import Float16
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f16, f16)>"),
        [Uint32(packed_h2).ir_value(loc=loc, ip=ip)],
        "mov.b32 {$0, $1}, $2;",
        "=h,=h,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float16(llvm.extractvalue(T.f16(), res, [0], loc=loc, ip=ip)),
        Float16(llvm.extractvalue(T.f16(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def f16x2_pack_broadcast(v: Float32, *, loc=None, ip=None) -> Uint32:
    """Broadcast a single fp32 value to both lanes of an fp16x2 register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(v).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 h;
                cvt.rn.f16.f32 h, $1;
                mov.b32 $0, {h, h};
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )



# mm_fp4 input constants.
_FP4_PER_BYTE: cutlass.Constexpr = 2     # two FP4 codes packed per uint8
_TILE_N_MULT: cutlass.Constexpr = 64     # tile_N must be multiple of 64
_TILE_K_MULT: cutlass.Constexpr = 16     # = GROUP_SIZE (one SF byte per K-block)

# 128x4 SF swizzle constants.
_SF_BLOCK_BYTES: cutlass.Constexpr = 512  # bytes per (128 N x 4 K_sf) block
_SF_BLOCK_N: cutlass.Constexpr = 128
_SF_BLOCK_KSF: cutlass.Constexpr = 4


def _sb_raw_n_stride_bytes(tile_k_half: int) -> int:
    """Per-row SMEM stride for sB_raw to suppress bank conflicts.

    The dequant function reads 1 byte per thread per ld.shared.u8, where 8
    threads in a warp share the same K coordinate and read 8 distinct N rows
    (lane // 4 ∈ [0, 8)).  SMEM has 32 banks × 4 bytes wide, so the bank
    each row hits is ``(n * row_stride // 4) % 32``.  For the natural compact
    stride of 64 bytes (tile_K=128), 8 rows hit only 2 banks (4-way conflict
    — ncu reports ~60% wavefront-replay rate).

    Padding the row stride to a value that spreads the 8 rows across more
    banks reduces conflict-replay cost.  Blackwell TMA's bulk-copy atom
    requires 32-byte alignment on per-row destination addresses, so the
    stride must be a multiple of 32.  Among 32-aligned strides >= tile_K/2:

      stride  banks_seen_by_8_rows  conflict_factor
        64    {0,16}                4-way (current)
        96    {0,24,16,8}           2-way   ← best 32-aligned option
       128    {0}                   8-way   (catastrophic)
       160    {0,8,16,24}           2-way   (more SMEM, no improvement vs 96)
       192    {0,16}                4-way   (same as 64)

    For tile_K=128 we pick 96 (32-aligned, 2-way conflict, +50% SMEM vs 64).
    For other tile_K we use the natural compact stride.

    DISABLED 2026-05-20: stride 80 and 96 both crash with cudaErrorMisalignedAddress
    despite satisfying 16/32-byte alignment.  TMA in cute-DSL appears to require
    a tightly-packed destination layout (or there is another SMEM access path
    that requires the natural compact stride).  Need to investigate further --
    likely needs a different layout strategy (XOR swizzle or b12x-style decode-
    to-bf16-SMEM) to break the bank conflicts.
    """
    return tile_k_half


# 1-byte gmem load through non-coherent cache (for per-thread SF reads).
@dsl_user_op
def _ld_global_nc_u8(base_ptr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u8 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# 1-byte SMEM load.
@dsl_user_op
def _ld_shared_u8(smem_addr: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.u8 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class BlackwellDenseGemmW4A16MmFp4Kernel:
    """Warp-MMA dense GEMM for Blackwell, FP4-weight A bf16/fp16 input.

    A: (M, K, L) bf16/fp16.
    B: (K // 16, N * 2, L) int32 -- Marlin-packed FP4 (see prepare).
    B_sf: (K // 16, N, L) uint8 -- FP8-E4M3 per-group scales.
    alpha: (1,) fp32 -- global scalar scale.
    C: (M, N, L) bf16/fp16; fp32 accumulator cast at write.

    Uses ``MmaF16BF16Op`` (16x8x16) + ``LdMatrix8x8x16bOp.x4`` for A and
    in-register FP4 decode for B.  Producer/consumer pipeline split with
    a single DMA warp doing TMA loads of A and B_marlin.  Group size is
    hardcoded to 16 (= MMA K-block size).
    """

    GROUP_SIZE: cutlass.Constexpr = 16

    # Ablation bitflags -- compile-time toggles that omit one or more
    # components from the inner loop while keeping the producer-consumer
    # pipeline barriers intact (so the kernel doesn't deadlock).  Output
    # is *incorrect* under any non-zero mode -- this is for perf
    # decomposition only.
    ABL_NONE: cutlass.Constexpr = 0
    ABL_SKIP_DEQUANT: cutlass.Constexpr = 1  # tCrB left at uninitialized values
    ABL_SKIP_MMA: cutlass.Constexpr = 2      # cute.gemm calls skipped
    ABL_SKIP_LDMATRIX_A: cutlass.Constexpr = 4  # ldmatrix sA -> tCrA skipped
    ABL_SKIP_EPILOGUE: cutlass.Constexpr = 8  # accumulator -> smem -> TMA store skipped
    ABL_SKIP_PRODUCER_TAIL: cutlass.Constexpr = 16  # skip producer_tail drain
    ABL_SKIP_FENCE: cutlass.Constexpr = 32  # skip fence_proxy in epilogue
    ABL_SKIP_TMA_STORE: cutlass.Constexpr = 64  # skip TMA store call
    ABL_SKIP_R2S: cutlass.Constexpr = 128  # skip the StMatrix R2S copy

    # Fine-grained dequant ablations (HMUL2 path only).  Each replaces a
    # stage's outputs with a runtime sentinel (Uint32(tidx)) to prevent
    # constant-folding while skipping that stage's compute.  Output is
    # incorrect under any of these -- timing only.
    ABL_DQ_NO_SMEM: cutlass.Constexpr = 256    # skip sB_raw + sB_sf loads
    ABL_DQ_NO_DECODE: cutlass.Constexpr = 512  # skip fp4_decode_2 + cvt_e4m3
    ABL_DQ_NO_CVT_CHAIN: cutlass.Constexpr = 1024  # skip hmul2 + f16->f32->bf16

    # Dequant primitive selection (constexpr-friendly).
    #   0 = CVT_F32 (baseline, retained for ablation):
    #       cvt.rn.f16x2.e2m1x2 + per-element fp32 scale multiply, ~6 ops/pair.
    #   1 = HMUL2_F16 (default, ~2% faster):
    #       cvt.rn.f16x2.e2m1x2 + hmul2 in fp16 + cvt fp16x2->fp32x2->bf16
    #       at write time.  Saves the 16 fp32 muls vs 8 hmul2s per K-block.
    #
    # Phase B experiment (see scratch_w4a16_dequant_sweep.py): we also tested
    # vLLM Marlin's AND/OR/SHIFT bit-extract straight to bf16x2 + folded
    # alpha*scale*2^126 hmul2 (csrc/quantization/marlin/dequant.h:434).  The
    # algebra wins on op count (~66 vs ~72 ops/K-block) but the natural
    # output pairing (same mma_i, varying nn) collides with the MMA register
    # layout (varying mma_i, same nn).  The required bf16x2 unpack + non-
    # adjacent tCrB writes regressed by 5 us, so we kept HMUL2_F16.
    DEQUANT_MODE_CVT_F32: cutlass.Constexpr = 0
    DEQUANT_MODE_HMUL2_F16: cutlass.Constexpr = 1

    # Reserved for a future sB_raw load-width experiment (see memory note
    # project_w4a16_mmfp4_kernel for prior v2/v4.u32 attempts that didn't
    # pay back -- the bottleneck is layout-bound, not width-bound).
    VECTORIZED_SMEM_B_DEFAULT: cutlass.Constexpr = 1

    # Phase C: K-block dequant pipeline depth (constexpr).
    #   0 = no prefetch -- dequant(k) then gemm(k) in the same iteration.
    #       Used as a baseline to measure whether prefetch is helping.
    #   1 = current default -- dequant(k+1) while gemm(k) within the same
    #       iteration; relies on the compiler to interleave the two streams.
    PIPELINE_DEPTH_DEFAULT: cutlass.Constexpr = 1

    def __init__(
        self,
        acc_dtype,
        tile_shape_mnk,
        epi_stage: int = 4,
        ablation_mode: int = 0,
        dequant_mode: int = 1,
        pipeline_depth: int = 1,
        # Currently a no-op (the dequant function always uses ld.shared.u8).
        # v2.u32 and v4.u32 widening were both tried and neither paid off
        # -- see memory note project_w4a16_mmfp4_kernel.  Kept as a
        # placeholder for future width-related experiments.
        vectorized_smem_b: int = 1,
        atom_layout: Tuple[int, int, int] = (2, 2, 1),
        epi_tile_override: Optional[Tuple[int, int]] = None,
        sb_raw_n_stride_bytes: Optional[int] = None,
        # 0 = bf16 MMA (default; A and B both bf16, matching the bf16 A
        # input).  1 = fp16 MMA: drops the f16->f32->bf16 cvt chain on B
        # (dequant writes fp16 directly), at the cost of an in-register
        # bf16->fp16 conversion on A after ldmatrix.  Net hoped-for win
        # is ~4-5us on the B-side cvt chain minus ~1-2us A conversion.
        # Accumulator stays fp32 either way.
        use_fp16_mma: int = 0,
    ):
        """W4A16 kernel.

        Args:
            acc_dtype: accumulator dtype (always Float32 for this kernel).
            tile_shape_mnk: CTA tile shape.
            epi_stage: TMA-store pipeline depth.  Default 4 balances
                cross-tile overlap (epi_stage > 1 lets next-tile compute
                overlap with this-tile store) against SMEM available for
                ab_stage (deeper ab_stage hides TMA-load latency, which
                dominates the small-M shape's stalls).  At M=4 with the
                default 64-CTA grid on 148 SMs, each SM gets at most
                one tile, so epi_stage > 1 doesn't help much locally;
                we keep 2-4 to preserve persistent-scheduler benefits
                at larger M.
        """
        self.acc_dtype = acc_dtype
        self.cluster_shape_mnk = (1, 1, 1)
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.epi_stage_target = int(epi_stage)
        if self.epi_stage_target < 1:
            raise ValueError(f"epi_stage must be >= 1 (got {epi_stage})")
        # Constexpr-friendly ablation bitflags (see class docstring).
        self.ablation_mode = int(ablation_mode)
        self.dequant_mode = int(dequant_mode)
        self.pipeline_depth = int(pipeline_depth)
        self.vectorized_smem_b = int(vectorized_smem_b)
        # Optional override for the epilogue tile shape.  Smaller epi_tile_m
        # enables fine-grained skipping of OOB epilogue iterations when
        # m_actual < tile_M.  Must be a multiple of the MMA-atom shape
        # (e.g., for our m16n8k16 atom: epi_tile_m must be a multiple of 16,
        # epi_tile_n must be a multiple of 16).  None = sm90 default
        # heuristic (min(64, tile_M), 32).
        self.epi_tile_override = (
            tuple(epi_tile_override) if epi_tile_override is not None else None
        )
        # Optional override for the sB_raw N-row SMEM stride.  None = pick via
        # _sb_raw_n_stride_bytes() to avoid bank conflicts (default).  Pass an
        # explicit value (e.g., tile_K // 2 for natural compact stride) to A/B
        # test against the padded version.
        self.sb_raw_n_stride_bytes_override = (
            int(sb_raw_n_stride_bytes) if sb_raw_n_stride_bytes is not None else None
        )
        self.use_fp16_mma = int(use_fp16_mma)
        self.tiled_mma = None
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False

        if self.tile_shape_mnk[1] % _TILE_N_MULT != 0:
            raise ValueError(
                f"W4A16 requires tile_N % {_TILE_N_MULT} == 0 "
                f"(got tile_N={self.tile_shape_mnk[1]})"
            )
        if self.tile_shape_mnk[2] % _TILE_K_MULT != 0:
            raise ValueError(
                f"W4A16 requires tile_K % {_TILE_K_MULT} == 0 "
                f"(got tile_K={self.tile_shape_mnk[2]})"
            )

        self.occupancy = 1
        # 2x2 atom layout: 4 MMA warps arranged as 2 M-warps x 2 N-warps.
        # Matches the upstream Blackwell-Geforce reference; the partition
        # diagnostic (see scratch_partition_mapping.py) shows mma_n=4
        # (n_inner=2 via *2 trick x n_outer=2), giving 16 fp16 per thread
        # per K-block = 2 int32 of FP4 to decode + scale.
        self.atom_layout = tuple(atom_layout)
        if self.atom_layout not in ((2, 2, 1), (1, 2, 1)):
            raise ValueError(
                f"Unsupported atom_layout {self.atom_layout!r}; "
                "expected (2,2,1) or (1,2,1)"
            )
        self.num_mma_warps = (
            self.atom_layout[0] * self.atom_layout[1] * self.atom_layout[2]
        )
        self.num_dma_warps = 1
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + self.num_dma_warps
        ) * self.num_threads_per_warp
        # SM100/103 expose >= SM120 SMEM/CTA; using sm_120 as the cap means
        # one binary works across all four Blackwell targets without
        # over-allocating on the smaller consumer chips.
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        self.mma_inst_mnk = (16, 8, 16)
        # MmaF16BF16Op accepts ab_dtype in {Float16, BFloat16}.  We pick
        # via b_compute_dtype so use_fp16_mma=1 swings the whole MMA to
        # fp16 (both A and B fragments will be fp16-typed).
        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.b_compute_dtype,
            self.acc_dtype,
            self.mma_inst_mnk,
        )
        tC = cute.make_layout(self.atom_layout)
        permutation_mnk = (
            self.atom_layout[0] * self.mma_inst_mnk[0],
            # *2 trick: each warp covers two atom-N tiles in one ldmatrix.x4
            self.atom_layout[1] * self.mma_inst_mnk[1] * 2,
            self.atom_layout[2] * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(
            op,
            tC,
            permutation_mnk=permutation_mnk,
        )

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk,
            self.c_dtype,
            is_cooperative=False,
            epi_tile_override=self.epi_tile_override,
        )

        # B-side smem is Marlin-packed int32 (4 bytes per logical FP4
        # pair).  Stage budget uses int32 B + uint8 scales; bf16 phantom
        # layout is only used by partition_B/make_fragment_B.
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
            self.GROUP_SIZE,
            self.epi_stage_target,
        )

        if self.ab_stage == 0:
            raise RuntimeError(
                "ab_stage == 0: not enough shared memory for this tile shape "
                f"({self.tile_shape_mnk}) at occupancy {self.occupancy}."
            )

        (
            self.a_smem_layout_staged,
            self.b_raw_smem_layout_staged,
            self.b_sf_smem_layout_staged,
            self.b_bf16_logical_layout,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_compute_dtype,
            self.b_layout_compute,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
            self.GROUP_SIZE,
            self.sb_raw_n_stride_bytes_override,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,             # (N, K/2) uint8 raw FP4
        b_sf: cute.Tensor,          # 1-D byte buffer, 128x4-swizzled FP8 SF
        c: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        from cutlass import Float16
        self.a_dtype = a.element_type
        # MMA operand dtype: bf16 by default (matches A), or fp16 when
        # use_fp16_mma=1 (lets us skip the f16->f32->bf16 cvt chain on B).
        # When fp16 MMA, A must be converted from bf16 to fp16 in-register
        # after ldmatrix; see the mainloop's tCrA_bf16_staging path.
        if cutlass.const_expr(self.use_fp16_mma == 1):
            self.b_compute_dtype = Float16
        else:
            self.b_compute_dtype = a.element_type
        self.c_dtype = c.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        # B's compute (= post-dequant) view is N-major (matches the
        # phantom bf16 layout used by partition_B / make_fragment_B).
        self.b_layout_compute = utils.LayoutEnum.ROW_MAJOR
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype.width != 16):
            raise TypeError(f"a_dtype must be 16-bit (bf16/fp16), got {self.a_dtype}")
        if cutlass.const_expr(self.a_dtype != self.c_dtype):
            raise TypeError(
                f"a_dtype and c_dtype must match, got {self.a_dtype} vs {self.c_dtype}"
            )

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )

        # B TMA: tile is (tile_N, tile_K/2) uint8 -- raw FP4 bytes,
        # K stride-1 in the source tensor.
        b_tma_tile = (
            self.tile_shape_mnk[1],
            self.tile_shape_mnk[2] // _FP4_PER_BYTE,
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_raw_smem_layout_staged,
            b_tma_tile,
            1,
        )

        # SF TMA: re-view the SF byte buffer as (N_blocks, K_sf_blocks,
        # 512) where each row is one 128x4-swizzled SF block.  Tile shape
        # (1, num_sf_blocks_per_k_tile, 512) loads the swizzle blocks
        # this CTA needs for one K-tile.  cluster_shape (1,1,1) means
        # no multicast; CTAs sharing an N-block load the same bytes.
        num_sf_blocks_per_k_tile = self.tile_shape_mnk[2] // (
            self.GROUP_SIZE * _SF_BLOCK_KSF
        )
        b_sf_tma_tile = (1, num_sf_blocks_per_k_tile, _SF_BLOCK_BYTES)
        # SF SMEM layout is rank 4 ((1, num, 512, stage)).  Use a custom
        # slice over the stage axis instead of the rank-3 helper.
        b_sf_smem_layout = cute.slice_(
            self.b_sf_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_b_sf, tma_tensor_b_sf = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            b_sf,
            b_sf_smem_layout,
            b_sf_tma_tile,
            num_multicast=1,
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
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
            sB_raw: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8, cute.cosize(self.b_raw_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB_sf: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8, cute.cosize(self.b_sf_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_b_sf,
            tma_tensor_b_sf,
            tma_atom_c,
            tma_tensor_c,
            alpha,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_raw_smem_layout_staged,
            self.b_sf_smem_layout_staged,
            self.b_bf16_logical_layout,
            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_kn: cute.Tensor,
        tma_atom_b_sf: cute.CopyAtom,
        mB_sf_nkblock: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mAlpha: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_raw_smem_layout_staged: cute.Layout,
        b_sf_smem_layout_staged: cute.Layout,
        b_bf16_logical_layout: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors from warp 0.
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b_sf)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_raw_smem_layout = cute.slice_(b_raw_smem_layout_staged, (None, None, 0))
        b_sf_smem_layout = cute.slice_(b_sf_smem_layout_staged, (None, None, None, 0))
        # Pipeline transaction count: A + raw FP4 B + SF bytes.
        tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(cutlass.Uint8, b_raw_smem_layout)
            + cute.size_in_bytes(cutlass.Uint8, b_sf_smem_layout)
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * self.num_mma_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # sB_raw is plain (non-swizzled) uint8 (tile_N, tile_K/2, stage).
        sB_raw = storage.sB_raw.get_tensor(b_raw_smem_layout_staged)
        # sB_sf is plain uint8 (num_sf_blocks_per_k_tile, 512, stage).
        sB_sf = storage.sB_sf.get_tensor(b_sf_smem_layout_staged)
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # B raw: (N, K/2, L) uint8; tile = (tile_N, tile_K/2).
        b_tile_shape = (
            self.tile_shape_mnk[1],
            self.tile_shape_mnk[2] // _FP4_PER_BYTE,
        )
        gB_kn = cute.local_tile(
            mB_kn,
            b_tile_shape,
            (None, None, None),
        )
        # SF view as (N_blocks, K_sf_blocks, 512); tile = (1, num_sf_blocks_per_k_tile, 512).
        num_sf_blocks_per_k_tile_c = cutlass.const_expr(
            self.tile_shape_mnk[2] // (self.GROUP_SIZE * _SF_BLOCK_KSF)
        )
        b_sf_tile_shape = (1, num_sf_blocks_per_k_tile_c, _SF_BLOCK_BYTES)
        gB_sf = cute.local_tile(
            mB_sf_nkblock,
            b_sf_tile_shape,
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        thr_mma = tiled_mma.get_slice(tidx)

        # TMA partition for A: (m, k) -> per-CTA partition.
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA partition for B raw FP4: (tile_N, tile_K/2) uint8.
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tB_s, tB_g = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_raw, 0, 2),
            cute.group_modes(gB_kn, 0, 2),
        )

        # TMA partition for SF: (1, num_sf_blocks_per_k_tile, 512) uint8.
        # Both sB_sf and gB_sf have the 3 tile-inner modes grouped.
        tBsf_s, tBsf_g = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b_sf,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_sf, 0, 3),
            cute.group_modes(gB_sf, 0, 3),
        )

        # B partition uses a phantom bf16 layout on top of sB_raw (uint8)
        # storage (recast pointer + bf16 logical layout).  This gives
        # ``partition_B`` / ``make_fragment_B`` the right fragment shape;
        # the data is never read through this view -- we always decode
        # FP4 ourselves in _dequant_b_to_register.
        sB_phantom = cute.make_tensor(
            cute.recast_ptr(sB_raw.iterator, dtype=self.b_compute_dtype),
            b_bf16_logical_layout,
        )
        tCsA = thr_mma.partition_A(sA)
        tCsB_phantom = thr_mma.partition_B(sB_phantom)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB_phantom[None, None, None, 0])

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            pipeline.sync(barrier_id=1)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # MMA warp group: warps [0, num_mma_warps) compute.
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # ldmatrix is only for A.  B is filled in-register from sB_raw
            # via FP4 decode + per-group scale.
            #
            # use_fp16_mma=1: tCrA is fp16-typed (matches the fp16 MMA),
            # but sA is bf16 in SMEM.  ldmatrix into a bf16 staging
            # fragment, then per-K-block convert bf16 -> fp16 in-register
            # before MMA reads tCrA.  See _ldmatrix_a in the inner loop.
            from cutlass import BFloat16, Float16
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            if cutlass.const_expr(self.use_fp16_mma == 1):
                tCrA_bf16 = cute.make_fragment_like(tCrA, BFloat16)
                tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA_bf16)
            else:
                # Placeholder so the (dead) cvt branch at call sites can
                # name-lookup tCrA_bf16 even when bf16-MMA is active.
                tCrA_bf16 = tCrA
                tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)

            alpha_val = Float32(mAlpha[Int32(0)])

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                accumulators.fill(0.0)
                # CTA's N offset within its 128-N SF block.  For tile_N
                # >= 128 this is always 0; for tile_N < 128 it cycles
                # 0, tile_N, 2*tile_N, ... up to 128-tile_N.
                cta_n_within_block = (
                    tile_coord_mnl[1] * Int32(self.tile_shape_mnk[1])
                ) & Int32(_SF_BLOCK_N - 1)

                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]

                # Prologue: prefetch tCrA[0] and tCrB[0] only when running with
                # pipeline_depth >= 1.  With depth=0 the dequant for block k
                # happens inside the same iteration as gemm(k) -- no prefetch.
                if cutlass.const_expr(self.pipeline_depth >= 1):
                    if cutlass.const_expr(
                        (self.ablation_mode & self.ABL_SKIP_LDMATRIX_A) == 0
                    ):
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, 0],
                            tCrA_copy_view[None, None, 0],
                        )
                        if cutlass.const_expr(self.use_fp16_mma == 1):
                            self._cvt_a_bf16_to_fp16_one_k_block(
                                tCrA, tCrA_bf16, 0
                            )
                    self._dequant_b_to_register(
                        sB_raw,
                        sB_sf,
                        tCrB,
                        alpha_val,
                        tidx,
                        mainloop_consumer_state.index,
                        0,
                        cta_n_within_block,
                    )

                for k_tile in range(0, k_tile_cnt - 1, 1, unroll=1):
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )

                        if cutlass.const_expr(self.pipeline_depth == 0):
                            # No-prefetch: dequant(k) just before gemm(k),
                            # all on the CURRENT K-tile's stage.  Release +
                            # advance + wait happens AFTER gemm of the last
                            # block so we don't drop SMEM under our own read.
                            if cutlass.const_expr(
                                (self.ablation_mode & self.ABL_SKIP_LDMATRIX_A) == 0
                            ):
                                cute.copy(
                                    smem_tiled_copy_A,
                                    tCsA_p[None, None, k_block_idx],
                                    tCrA_copy_view[None, None, k_block_idx],
                                )
                                if cutlass.const_expr(self.use_fp16_mma == 1):
                                    self._cvt_a_bf16_to_fp16_one_k_block(
                                        tCrA, tCrA_bf16, k_block_idx
                                    )
                            self._dequant_b_to_register(
                                sB_raw,
                                sB_sf,
                                tCrB,
                                alpha_val,
                                tidx,
                                mainloop_consumer_state.index,
                                k_block_idx,
                                cta_n_within_block,
                            )
                            if cutlass.const_expr(
                                (self.ablation_mode & self.ABL_SKIP_MMA) == 0
                            ):
                                cute.gemm(
                                    tiled_mma,
                                    accumulators,
                                    tCrA[None, None, k_block_idx],
                                    tCrB[None, None, k_block_idx],
                                    accumulators,
                                )
                            if k_block_idx == num_k_blocks - 1:
                                mainloop_pipeline.consumer_release(
                                    mainloop_consumer_state
                                )
                                mainloop_consumer_state.advance()
                                peek_ab_full_status = cutlass.Boolean(1)
                                peek_ab_full_status = (
                                    mainloop_pipeline.consumer_try_wait(
                                        mainloop_consumer_state
                                    )
                                )
                                tCsA_p = tCsA_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                mainloop_pipeline.consumer_wait(
                                    mainloop_consumer_state, peek_ab_full_status
                                )
                        else:
                            # 1-stage prefetch: dequant(k+1) while gemm(k).
                            if k_block_idx == num_k_blocks - 1:
                                mainloop_pipeline.consumer_release(
                                    mainloop_consumer_state
                                )
                                mainloop_consumer_state.advance()

                                peek_ab_full_status = cutlass.Boolean(1)
                                peek_ab_full_status = (
                                    mainloop_pipeline.consumer_try_wait(
                                        mainloop_consumer_state
                                    )
                                )

                                tCsA_p = tCsA_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                mainloop_pipeline.consumer_wait(
                                    mainloop_consumer_state, peek_ab_full_status
                                )

                            if cutlass.const_expr(
                                (self.ablation_mode & self.ABL_SKIP_LDMATRIX_A) == 0
                            ):
                                cute.copy(
                                    smem_tiled_copy_A,
                                    tCsA_p[None, None, k_block_next],
                                    tCrA_copy_view[None, None, k_block_next],
                                )
                                if cutlass.const_expr(self.use_fp16_mma == 1):
                                    self._cvt_a_bf16_to_fp16_one_k_block(
                                        tCrA, tCrA_bf16, k_block_next
                                    )
                            self._dequant_b_to_register(
                                sB_raw,
                                sB_sf,
                                tCrB,
                                alpha_val,
                                tidx,
                                mainloop_consumer_state.index,
                                k_block_next,
                                cta_n_within_block,
                            )
                            if cutlass.const_expr(
                                (self.ablation_mode & self.ABL_SKIP_MMA) == 0
                            ):
                                cute.gemm(
                                    tiled_mma,
                                    accumulators,
                                    tCrA[None, None, k_block_idx],
                                    tCrB[None, None, k_block_idx],
                                    accumulators,
                                )
                # Hoist out last k_tile (no further loads after the last k_block)
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )

                    if cutlass.const_expr(self.pipeline_depth == 0):
                        # No-prefetch path for last K-tile.  Release happens
                        # AFTER gemm of the last block (kernel exits then).
                        if cutlass.const_expr(
                            (self.ablation_mode & self.ABL_SKIP_LDMATRIX_A) == 0
                        ):
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, k_block_idx],
                                tCrA_copy_view[None, None, k_block_idx],
                            )
                            if cutlass.const_expr(self.use_fp16_mma == 1):
                                self._cvt_a_bf16_to_fp16_one_k_block(
                                    tCrA, tCrA_bf16, k_block_idx
                                )
                        self._dequant_b_to_register(
                            sB_raw,
                            sB_sf,
                            tCrB,
                            alpha_val,
                            tidx,
                            mainloop_consumer_state.index,
                            k_block_idx,
                            cta_n_within_block,
                        )
                        if cutlass.const_expr(
                            (self.ablation_mode & self.ABL_SKIP_MMA) == 0
                        ):
                            cute.gemm(
                                tiled_mma,
                                accumulators,
                                tCrA[None, None, k_block_idx],
                                tCrB[None, None, k_block_idx],
                                accumulators,
                            )
                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()
                    else:
                        # 1-stage prefetch path for last K-tile.
                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                        if k_block_next > 0:
                            if cutlass.const_expr(
                                (self.ablation_mode & self.ABL_SKIP_LDMATRIX_A) == 0
                            ):
                                cute.copy(
                                    smem_tiled_copy_A,
                                    tCsA_p[None, None, k_block_next],
                                    tCrA_copy_view[None, None, k_block_next],
                                )
                                if cutlass.const_expr(self.use_fp16_mma == 1):
                                    self._cvt_a_bf16_to_fp16_one_k_block(
                                        tCrA, tCrA_bf16, k_block_next
                                    )
                            self._dequant_b_to_register(
                                sB_raw,
                                sB_sf,
                                tCrB,
                                alpha_val,
                                tidx,
                                mainloop_consumer_state.index,
                                k_block_next,
                                cta_n_within_block,
                            )
                        if cutlass.const_expr(
                            (self.ablation_mode & self.ABL_SKIP_MMA) == 0
                        ):
                            cute.gemm(
                                tiled_mma,
                                accumulators,
                                tCrA[None, None, k_block_idx],
                                tCrB[None, None, k_block_idx],
                                accumulators,
                            )

                # Epilogue: accumulator -> smem -> gmem via R2S (StMatrix.x4)
                # + TMA bulk store.  Skipped under ABL_SKIP_EPILOGUE.
                if cutlass.const_expr(
                    (self.ablation_mode & self.ABL_SKIP_EPILOGUE) == 0
                ):
                    copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                        self.c_layout,
                        elem_ty_d=self.c_dtype,
                        elem_ty_acc=self.acc_dtype,
                    )

                    copy_atom_C = cute.make_copy_atom(
                        cute.nvgpu.warp.StMatrix8x8x16bOp(
                            self.c_layout.is_m_major_c(),
                            4,
                        ),
                        self.c_dtype,
                    )

                    tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(
                        copy_atom_C, tiled_mma
                    )

                    tiled_copy_r2s = cute.make_tiled_copy_S(
                        copy_atom_r2s,
                        tiled_copy_C_Atom,
                    )

                    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                    tRS_sD = thr_copy_r2s.partition_D(sC)
                    tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                    rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                    tRS_rD_layout = cute.make_layout(rD_shape[:3])
                    tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
                    size_tRS_rD = cute.size(tRS_rD)

                    sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                    tcgc_for_tma_partition = cute.zipped_divide(
                        gC_mnl_slice, self.epi_tile
                    )

                    bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        sepi_for_tma_partition,
                        tcgc_for_tma_partition,
                    )

                    # b12x-style epilogue: iterate (epi_m, epi_n) explicitly
                    # and use (mma_m, mma_n) mode indexing into tRS_rAcc so
                    # the loop works for any epi_tile_m / epi_tile_n.  Also
                    # supports OOB-iteration skipping when m_actual < tile_M.
                    epi_rest_m = cute.size(tcgc_for_tma_partition, mode=[1, 0])
                    epi_rest_n = cute.size(tcgc_for_tma_partition, mode=[1, 1])
                    epi_tile_m = self.epi_tile[0]
                    epi_tile_n = self.epi_tile[1]
                    # mma_tile_{m,n} = per-mma-atom (M,N) size.  tRS_rAcc has
                    # shape (atom_v, mma_m, mma_n); modes 1, 2 give the atom
                    # counts in M, N.
                    mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rAcc, mode=[1])
                    mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rAcc, mode=[2])
                    MmaMPerEpiM = epi_tile_m // mma_tile_m
                    MmaNPerEpiN = epi_tile_n // mma_tile_n

                    tma_store_producer_group = pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        self.num_mma_warps * self.num_threads_per_warp,
                    )
                    tma_store_pipeline = pipeline.PipelineTmaStore.create(
                        num_stages=self.epi_stage,
                        producer_group=tma_store_producer_group,
                    )

                    # Skip OOB epilogue iterations when actual M < tile_M.
                    # TMA OOB clipping already protects correctness for
                    # the non-skipped iterations; this avoids the work for
                    # tiles that would be fully clipped anyway.
                    m_actual = cute.size(mC_mnl, mode=[0])
                    cta_m_offset = (
                        tile_coord_mnl[0] * Int32(self.tile_shape_mnk[0])
                    )

                    # kept_count cycles in lockstep with the TMA-store pipeline.
                    # epi_buffer = kept_count % num_stages stays in sync with
                    # producer_commit/acquire calls, regardless of how many
                    # iterations are skipped.
                    kept_count = 0
                    for epi_n in cutlass.range_constexpr(epi_rest_n):
                        for epi_m in cutlass.range_constexpr(epi_rest_m):
                            epi_m_global_start = (
                                cta_m_offset + Int32(epi_m * epi_tile_m)
                            )
                            if epi_m_global_start < m_actual:
                                # Copy this epi-tile's slice of acc -> tRS_rD
                                # using b12x-style (mma_m, mma_n) indexing.
                                for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                                    for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                                        mma_n = epi_n * MmaNPerEpiN + mma_n_in_epi
                                        mma_m = epi_m * MmaMPerEpiM + mma_m_in_epi
                                        tRS_rD_slice = tRS_rD[
                                            (None, mma_m_in_epi, mma_n_in_epi)
                                        ]
                                        tRS_rAcc_slice = tRS_rAcc[
                                            (None, mma_m, mma_n)
                                        ]
                                        for elem_idx in cutlass.range_constexpr(
                                            cute.size(tRS_rD_slice)
                                        ):
                                            tRS_rD_slice[elem_idx] = (
                                                tRS_rAcc_slice[elem_idx]
                                            )

                                tRS_rD_out = cute.make_rmem_tensor(
                                    tRS_rD_layout.shape, self.c_dtype
                                )
                                acc_vec = tRS_rD.load()
                                tRS_rD_out.store(acc_vec.to(self.c_dtype))

                                epi_buffer = kept_count % cute.size(
                                    tRS_sD, mode=[3]
                                )
                                if cutlass.const_expr(
                                    (self.ablation_mode & self.ABL_SKIP_R2S) == 0
                                ):
                                    cute.copy(
                                        tiled_copy_r2s,
                                        tRS_rD_out,
                                        tRS_sD[(None, None, None, epi_buffer)],
                                    )
                                if cutlass.const_expr(
                                    (self.ablation_mode & self.ABL_SKIP_FENCE) == 0
                                ):
                                    cute.arch.fence_proxy(
                                        "async.shared", space="cta"
                                    )
                                self.epilog_sync_barrier.arrive_and_wait()

                                if warp_idx == 0:
                                    if cutlass.const_expr(
                                        (self.ablation_mode & self.ABL_SKIP_TMA_STORE) == 0
                                    ):
                                        cute.copy(
                                            tma_atom_c,
                                            bSG_sD[(None, epi_buffer)],
                                            bSG_gD[(None, (epi_m, epi_n))],
                                        )
                                        tma_store_pipeline.producer_commit()
                                        tma_store_pipeline.producer_acquire()
                                kept_count = kept_count + 1

                    if cutlass.const_expr(
                        (self.ablation_mode & self.ABL_SKIP_PRODUCER_TAIL) == 0
                    ):
                        tma_store_pipeline.producer_tail()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
        # Single DMA warp: TMA-loads A + raw FP4 B into the same stage
        # barrier per K-tile.  SF is gmem-direct (no producer step).
        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                # B tensor outer modes are (N_tiles, K_tiles, L); fix N+L,
                # iterate K.
                tB_g_kn = tB_g[
                    (None, tile_coord_mnl[1], None, tile_coord_mnl[2])
                ]
                # SF tensor outer modes are (N_blocks, K_sf_blocks_outer,
                # 1) where N_blocks = N/128 (128 N per swizzle block).
                # Each CTA's tile_N <= 128 maps to one N_block; for
                # tile_N < 128 multiple CTAs share the same block.
                n_block_per_cta = cutlass.const_expr(
                    self.tile_shape_mnk[1] // _SF_BLOCK_N
                )
                if cutlass.const_expr(n_block_per_cta == 0):
                    # tile_N < 128: divide n_tile by (128 / tile_N).
                    ctas_per_n_block = cutlass.const_expr(
                        _SF_BLOCK_N // self.tile_shape_mnk[1]
                    )
                    n_block_idx = tile_coord_mnl[1] // Int32(ctas_per_n_block)
                else:
                    # tile_N >= 128: n_block_idx = n_tile * n_block_per_cta.
                    n_block_idx = tile_coord_mnl[1] * Int32(n_block_per_cta)
                tBsf_g_kn = tBsf_g[
                    (None, n_block_idx, None, Int32(0))
                ]
                mainloop_producer_state.reset_count()
                for k_tile in range(0, k_tile_cnt, 1, unroll=1):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    barrier_ptr = mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    )

                    tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=barrier_ptr,
                        mcast_mask=a_mcast_mask,
                    )

                    tB_g_k = tB_g_kn[(None, mainloop_producer_state.count)]
                    tB_s_pipe = tB_s[(None, mainloop_producer_state.index)]
                    cute.copy(
                        tma_atom_b,
                        tB_g_k,
                        tB_s_pipe,
                        tma_bar_ptr=barrier_ptr,
                        mcast_mask=b_mcast_mask,
                    )

                    tBsf_g_k = tBsf_g_kn[
                        (None, mainloop_producer_state.count)
                    ]
                    tBsf_s_pipe = tBsf_s[
                        (None, mainloop_producer_state.index)
                    ]
                    cute.copy(
                        tma_atom_b_sf,
                        tBsf_g_k,
                        tBsf_s_pipe,
                        tma_bar_ptr=barrier_ptr,
                        mcast_mask=b_mcast_mask,
                    )

                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        epi_tile: Tuple[int, int],
        c_dtype: Type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
        group_size: int = 16,
        epi_stage: int = 4,
    ) -> Tuple[int, int]:
        """Stage budget accounting for A + B (Marlin int32) + B_sf (uint8).

        Per (16K x 64N) Marlin block: 128 int32 = 512 bytes.
        Per K-tile we hold:
          - sA: tile_M * tile_K * sizeof(a_dtype) bytes
          - sB_raw: tile_N * tile_K / 2 bytes (raw uint8 FP4)
          - sB_sf: num_sf_blocks_per_k_tile * 512 bytes (128x4-swizzled
            SF blocks TMA-loaded for this CTA's N range and this K-tile's
            K_sf range)
        """
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        a_bytes_per_stage = cute.size(a_shape) * a_dtype.width // 8

        # B raw FP4: tile_N rows x (tile_K / 2) uint8 bytes.
        b_raw_bytes_per_stage = tile_shape_mnk[1] * (
            tile_shape_mnk[2] // _FP4_PER_BYTE
        )

        # SF: num_sf_blocks_per_k_tile * 512 bytes.
        num_sf_blocks_per_k_tile = tile_shape_mnk[2] // (
            group_size * _SF_BLOCK_KSF
        )
        b_sf_bytes_per_stage = num_sf_blocks_per_k_tile * _SF_BLOCK_BYTES

        ab_bytes_per_stage = (
            a_bytes_per_stage + b_raw_bytes_per_stage + b_sf_bytes_per_stage
        )
        mbar_helpers_bytes = 1024

        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
        a_dtype: Type[cutlass.Numeric],
        a_layout: cute.Layout,
        b_compute_dtype: Type[cutlass.Numeric],
        b_layout_compute: cute.Layout,
        ab_stage: int,
        c_dtype: Type[cutlass.Numeric],
        c_layout: cute.Layout,
        epi_stage: int,
        group_size: int,
        sb_raw_n_stride_bytes_override: Optional[int] = None,
    ):
        """Returns (sA, sB_raw, sB_sf, b_bf16_phantom, sC) layouts.

        ``sB_raw_layout`` is a plain (non-swizzled) staged uint8 layout
        ``(tile_N, tile_K // 2, ab_stage)`` matching the (N, K/2) K-major
        gmem layout of the user's FP4 weight tensor.

        ``sB_sf_layout`` is a plain staged uint8 layout
        ``(num_sf_blocks_per_k_tile, 512, ab_stage)`` -- TMA-loads the
        128x4-swizzled SF blocks needed by this CTA for each K-tile.
        Each 512-byte block holds (128 N x 4 K_sf) entries in their
        swizzled in-block byte order; the dequant function reads
        per-thread using the same swizzle formula but with the SMEM
        block index in place of the gmem (n/128, k_sf/4) coordinate.

        ``b_bf16_logical_layout`` is the phantom bf16 layout used only by
        ``partition_B`` / ``make_fragment_B`` for fragment-shape
        determination -- there is no real bf16 SMEM allocation.
        """
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )

        # sB_raw: (tile_N) rows x (tile_K // 2) uint8 cols x stage.
        # K is stride-1 (innermost), matching the (N, K/2) gmem layout the
        # user provides.  N-row stride is padded above the natural
        # (tile_K // 2) bytes to break SMEM bank-conflict aliasing for the
        # dequant function's per-thread byte reads -- see the
        # _sb_raw_n_stride_bytes() docstring for the bank math.
        #
        # Per row we only fill (tile_K // 2) bytes from gmem via TMA; the
        # extra (stride - tile_K // 2) bytes at the end of each N row are
        # untouched padding that the consumer never reads.
        #
        # Earlier attempts: K_SW32 / K_SW64 cute XOR swizzles -- both TMA-
        # compatible for u8, but per-access swizzle ALU overhead (XOR/AND/
        # shift) on every ld.shared.u8 exceeds the bank-conflict savings.
        # XOR swizzle only pays off paired with ldmatrix (HW resolves the
        # swizzle as part of the warp-wide load); for raw u8 reads it's
        # net-negative.  See memory note project_w4a16_mmfp4_kernel.
        if sb_raw_n_stride_bytes_override is not None:
            n_stride_bytes = sb_raw_n_stride_bytes_override
        else:
            n_stride_bytes = _sb_raw_n_stride_bytes(
                tile_shape_mnk[2] // _FP4_PER_BYTE
            )
        b_raw_smem_layout_staged = cute.make_layout(
            (
                tile_shape_mnk[1],
                tile_shape_mnk[2] // _FP4_PER_BYTE,
                ab_stage,
            ),
            stride=(
                n_stride_bytes,
                1,
                n_stride_bytes * tile_shape_mnk[1],
            ),
        )

        # sB_sf: (num_sf_blocks_per_k_tile=tile_K/64) rows x 512 uint8
        # cols x stage.  Each row is one 128x4-swizzled SF block.
        num_sf_blocks_per_k_tile = tile_shape_mnk[2] // (
            group_size * _SF_BLOCK_KSF
        )
        if num_sf_blocks_per_k_tile == 0:
            raise ValueError(
                f"tile_K={tile_shape_mnk[2]} must be a multiple of "
                f"{group_size * _SF_BLOCK_KSF} (= group_size * 4) for the "
                f"SF SMEM tile to align with the 128x4 swizzle."
            )
        # Rank 4 = tile_rank (3) + stage axis, to match the (1, num, 512)
        # TMA tile rank.  The leading 1-dim is a no-op for indexing but
        # required to match the SF gmem view's (N_blocks, K_sf_blocks,
        # 512) shape.
        b_sf_smem_layout_staged = cute.make_ordered_layout(
            (
                1,
                num_sf_blocks_per_k_tile,
                _SF_BLOCK_BYTES,
                ab_stage,
            ),
            order=(2, 1, 0, 3),
        )

        # bf16 phantom layout for partition_B / make_fragment_B.
        b_bf16_logical_layout = sm90_utils.make_smem_layout_b(
            b_layout_compute,
            tile_shape_mnk,
            b_compute_dtype,
            ab_stage,
        )

        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            epi_stage,
        )
        return (
            a_smem_layout_staged,
            b_raw_smem_layout_staged,
            b_sf_smem_layout_staged,
            b_bf16_logical_layout,
            epi_smem_layout_staged,
        )

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: Tuple[int, int, int],
        max_active_clusters: cutlass.Constexpr,
    ):
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: Tuple[int, int],
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )
        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: Tuple[int, int],
        mcast_dim: int,
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    @cute.jit
    def _cvt_a_bf16_to_fp16_one_k_block(
        self,
        tCrA_dst,
        tCrA_bf16_src,
        k_block: cutlass.Constexpr,
    ):
        """In-register bf16 -> fp16 cvt for one K-block of A.

        Recasts both fragments to Uint32 (each u32 packs 2 16-bit elems)
        and applies `cvt_bf16x2_to_f16x2_via_f32` per pair.  The packed
        narrowing cvt (`cvt.rn.f16x2.f32`) combines what would be two
        scalar `cvt.rn.f16.f32` instructions if we used cute's default
        `.to(Float16)` lowering.

        We can't use bf16/f16 typed inline-asm constraints directly on
        sm_100a (NVVM rejects =h with bf16/f16 LLVM struct returns -- see
        scratch_cvt_bf16x2_to_f16x2_probe.py), so we keep everything in
        Uint32 pair representation.
        """
        bf_u32 = cute.recast_tensor(
            tCrA_bf16_src[None, None, k_block], Uint32
        )
        fp_u32 = cute.recast_tensor(
            tCrA_dst[None, None, k_block], Uint32
        )
        n_pairs = cute.size(bf_u32)
        for i in cutlass.range_constexpr(n_pairs):
            fp_u32[i] = cvt_bf16x2_to_f16x2_via_f32(Uint32(bf_u32[i]))

    @cute.jit
    def _swizzled_e4m3_offset(
        self,
        n: Int32,         # global N coordinate
        k_sf: Int32,      # global K_sf coordinate (= K // sf_vec_size)
        sf_cols: Int32,   # padded ceil(K/16, 4) -- col stride within an N-block
    ) -> Int64:
        """128x4 mm_fp4 SF swizzle byte offset.

        Matches mm_fp4's ``get_sf_out_offset_128x4`` and b12x's
        ``swizzle_block_scale`` (Reshape-then-permute formula)::

            offset = ((n/128) * (K_sf/4) + k_sf/4) * 512
                   + (n%32)*16 + ((n%128)/32)*4 + (k_sf%4)
        """
        row_rb = n >> Int32(7)                   # n / 128
        mode_a = (n >> Int32(5)) & Int32(3)      # (n % 128) / 32
        mode_32 = n & Int32(31)                  # n % 32
        cb_idx = k_sf >> Int32(2)                # k_sf / 4
        mode_c = k_sf & Int32(3)                 # k_sf % 4
        return (
            Int64(row_rb) * Int64(sf_cols * Int32(128))
            + Int64(cb_idx) * Int64(512)
            + Int64(mode_32) * Int64(16)
            + Int64(mode_a) * Int64(4)
            + Int64(mode_c)
        )

    @cute.jit
    def _dequant_b_to_register(
        self,
        sB_raw: cute.Tensor,       # (tile_N, tile_K/2, ab_stage) uint8 raw FP4
        sB_sf: cute.Tensor,        # (num_sf_blocks_per_k_tile, 512, ab_stage) uint8 swizzle blocks
        tCrB: cute.Tensor,
        alpha_val: Float32,
        tidx: Int32,
        stage_idx: Int32,
        k_block_idx: cutlass.Constexpr,
        cta_n_within_block: Int32,
    ):
        """Decode 8 packed FP4 bytes per thread per K-block into 16
        fp16 fragment slots, write to tCrB in-register.

        sB_raw byte addressing matches the (N, K/2) K-major gmem layout;
        per K-block kb, per N position, we read 2 bytes at K-pair offsets
        {kb*8 + tc_row/2, kb*8 + tc_row/2 + 4}.

        sB_sf holds N TMA-loaded 512-byte 128x4 swizzle blocks for this
        K-tile.  Each block packs (128 N x 4 K_sf) entries in the same
        internal byte order as gmem.  For thread (lane, warp) at K-block
        kb, the SF byte for (n_in_tile, k_sf=kb) is at SMEM offset::

            block_idx = kb // 4                # 0..num_sf_blocks-1
            n_in_block = cta_n_within_block + n_in_tile  # in [0, 128)
            intra = (n_in_block % 32)*16
                  + ((n_in_block % 128)/32)*4
                  + (kb % 4)
            sB_sf[block_idx, intra, stage_idx]

        ``cta_n_within_block`` is the CTA's N offset relative to its
        128-N SF block start (= (n_tile_idx * tile_N) % 128).  For tile_N
        >= 128 this is 0; for tile_N = 64 it alternates 0 / 64.

        Ablation: ``ABL_SKIP_DEQUANT`` no-ops the whole function.
        """
        if cutlass.const_expr(
            (self.ablation_mode & self.ABL_SKIP_DEQUANT) != 0
        ):
            return

        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        n_warp_idx = warp // Int32(self.atom_layout[0])
        tc_row = (lane % Int32(4)) * Int32(2)       # {0, 2, 4, 6}
        tc_col = lane // Int32(4)                   # [0, 8)
        base_n_in_tile = n_warp_idx * Int32(8) + tc_col

        # K-pair byte offsets within sB_raw row.  k_block_idx is constexpr.
        kb_base = Int32(k_block_idx) * Int32(self.GROUP_SIZE // _FP4_PER_BYTE)
        k_pair_lo = kb_base + (tc_row // Int32(2))
        k_pair_hi = k_pair_lo + Int32(4)

        # 8 SMEM byte loads -- 4 N positions x 2 K-pair offsets.
        # Documented bottleneck: 4-way bank conflict (8 N rows at stride-64
        # alias to 2 banks; ~60% wavefront-replay per ncu).  Instruction-
        # width tricks (v2.u32, v4.u32) don't help -- conflict is layout-
        # bound, not width-bound.  See memory note project_w4a16_mmfp4_kernel.
        n0 = base_n_in_tile + Int32(0)
        n1 = base_n_in_tile + Int32(16)
        n2 = base_n_in_tile + Int32(32)
        n3 = base_n_in_tile + Int32(48)

        # Runtime sentinel for fine-grained dequant ablations.  Must vary
        # per use (otherwise the compiler CSEs the downstream chain since
        # all sentinels share the same value) -- we XOR tidx with a
        # per-variable constant.  Cheap runtime expression that defeats CSE.
        def _sentinel(salt: int) -> "Uint32":
            return Uint32(tidx ^ Int32(salt))

        if cutlass.const_expr(
            (self.ablation_mode & self.ABL_DQ_NO_SMEM) != 0
        ):
            b00 = _sentinel(0x01); b01 = _sentinel(0x02)
            b10 = _sentinel(0x03); b11 = _sentinel(0x04)
            b20 = _sentinel(0x05); b21 = _sentinel(0x06)
            b30 = _sentinel(0x07); b31 = _sentinel(0x08)
        else:
            b00 = Uint32(sB_raw[n0, k_pair_lo, stage_idx])
            b01 = Uint32(sB_raw[n0, k_pair_hi, stage_idx])
            b10 = Uint32(sB_raw[n1, k_pair_lo, stage_idx])
            b11 = Uint32(sB_raw[n1, k_pair_hi, stage_idx])
            b20 = Uint32(sB_raw[n2, k_pair_lo, stage_idx])
            b21 = Uint32(sB_raw[n2, k_pair_hi, stage_idx])
            b30 = Uint32(sB_raw[n3, k_pair_lo, stage_idx])
            b31 = Uint32(sB_raw[n3, k_pair_hi, stage_idx])

        # SF SMEM reads.  block_idx and (kb%4) are constexpr.
        sf_block_idx = cutlass.const_expr(k_block_idx // _SF_BLOCK_KSF)
        kb_mod4 = cutlass.const_expr(k_block_idx % _SF_BLOCK_KSF)

        # Per-thread N position within the 128-N SF block.
        nb0 = cta_n_within_block + n0
        nb1 = cta_n_within_block + n1
        nb2 = cta_n_within_block + n2
        nb3 = cta_n_within_block + n3

        def _sf_intra(nb: Int32) -> Int32:
            return (
                (nb & Int32(31)) * Int32(16)
                + ((nb & Int32(127)) >> Int32(5)) * Int32(4)
                + Int32(kb_mod4)
            )

        if cutlass.const_expr(
            (self.ablation_mode & self.ABL_DQ_NO_SMEM) != 0
        ):
            sf_byte_0 = _sentinel(0x11); sf_byte_1 = _sentinel(0x12)
            sf_byte_2 = _sentinel(0x13); sf_byte_3 = _sentinel(0x14)
        else:
            sf_byte_0 = Uint32(sB_sf[0, sf_block_idx, _sf_intra(nb0), stage_idx])
            sf_byte_1 = Uint32(sB_sf[0, sf_block_idx, _sf_intra(nb1), stage_idx])
            sf_byte_2 = Uint32(sB_sf[0, sf_block_idx, _sf_intra(nb2), stage_idx])
            sf_byte_3 = Uint32(sB_sf[0, sf_block_idx, _sf_intra(nb3), stage_idx])

        # Decode each byte -> one fp16x2 = (K=tc_row, K=tc_row+1) for *_lo
        # bytes; (K=tc_row+8, K=tc_row+9) for *_hi bytes.  ABL_DQ_NO_DECODE
        # skips the cvt.rn.f16x2.e2m1x2 by passing the source byte through
        # as if it were already an f16x2 -- downstream hmul2 + cvt chain
        # still execute on real runtime values.
        if cutlass.const_expr(
            (self.ablation_mode & self.ABL_DQ_NO_DECODE) != 0
        ):
            h0_a = b00; h0_b = b01; h0_c = b10; h0_d = b11
            h1_a = b20; h1_b = b21; h1_c = b30; h1_d = b31
        else:
            h0_a = fp4_decode_2(b00)
            h0_b = fp4_decode_2(b01)
            h0_c = fp4_decode_2(b10)
            h0_d = fp4_decode_2(b11)
            h1_a = fp4_decode_2(b20)
            h1_b = fp4_decode_2(b21)
            h1_c = fp4_decode_2(b30)
            h1_d = fp4_decode_2(b31)

        # ============================================================
        # MODE 1 (default): cvt + hmul2 in fp16 + cvt to bf16 via fp32.
        # ============================================================
        if cutlass.const_expr(self.dequant_mode == self.DEQUANT_MODE_HMUL2_F16):
            # ABL_DQ_NO_DECODE also skips cvt_e4m3 + f16x2_pack_broadcast --
            # we use sf_byte_* directly as the f16x2-packed scale.
            if cutlass.const_expr(
                (self.ablation_mode & self.ABL_DQ_NO_DECODE) != 0
            ):
                sc_n0 = sf_byte_0
                sc_n1 = sf_byte_1
                sc_n2 = sf_byte_2
                sc_n3 = sf_byte_3
            else:
                scale_n0_f32 = cvt_e4m3_to_f32_via_f16(sf_byte_0) * alpha_val
                scale_n1_f32 = cvt_e4m3_to_f32_via_f16(sf_byte_1) * alpha_val
                scale_n2_f32 = cvt_e4m3_to_f32_via_f16(sf_byte_2) * alpha_val
                scale_n3_f32 = cvt_e4m3_to_f32_via_f16(sf_byte_3) * alpha_val
                sc_n0 = f16x2_pack_broadcast(scale_n0_f32)
                sc_n1 = f16x2_pack_broadcast(scale_n1_f32)
                sc_n2 = f16x2_pack_broadcast(scale_n2_f32)
                sc_n3 = f16x2_pack_broadcast(scale_n3_f32)

            # ABL_DQ_NO_CVT_CHAIN skips half2_mul + f16x2_to_f32x2 + .to(bf16)
            # by writing a sentinel bf16 directly.  Lets us isolate the cost
            # of the multiply + double-cvt sequence (called out in the
            # optimization log as ~5-6 us on the older Marlin sibling).
            if cutlass.const_expr(
                (self.ablation_mode & self.ABL_DQ_NO_CVT_CHAIN) != 0
            ):
                # Vary the bf16 sentinel per write to defeat CSE.
                tidx_f = Float32(tidx)
                for mma_i in cutlass.range_constexpr(4):
                    for nn in cutlass.range_constexpr(4):
                        salt = Float32(mma_i * 4 + nn)
                        tCrB[mma_i, nn, k_block_idx] = (
                            tidx_f + salt
                        ).to(self.b_compute_dtype)
                return

            if cutlass.const_expr(self.use_fp16_mma == 1):
                # fp16-MMA path: scaled_h2 (f16x2) -> 2 fp16 -> tCrB.
                # Saves the f16->f32->bf16 cvt chain (5 ops/pair -> 1 op/pair).
                def _write_hmul2(h2, scale_h2, mma_i_low, nn):
                    scaled_h2 = half2_mul(h2, scale_h2)
                    f_lo, f_hi = f16x2_unpack(scaled_h2)
                    tCrB[mma_i_low, nn, k_block_idx] = f_lo
                    tCrB[mma_i_low + 1, nn, k_block_idx] = f_hi
            else:
                def _write_hmul2(h2, scale_h2, mma_i_low, nn):
                    scaled_h2 = half2_mul(h2, scale_h2)
                    f_lo, f_hi = f16x2_to_f32x2(scaled_h2)
                    tCrB[mma_i_low, nn, k_block_idx] = f_lo.to(
                        self.b_compute_dtype
                    )
                    tCrB[mma_i_low + 1, nn, k_block_idx] = f_hi.to(
                        self.b_compute_dtype
                    )

            _write_hmul2(h0_a, sc_n0, 0, 0)
            _write_hmul2(h0_b, sc_n0, 2, 0)
            _write_hmul2(h0_c, sc_n1, 0, 1)
            _write_hmul2(h0_d, sc_n1, 2, 1)
            _write_hmul2(h1_a, sc_n2, 0, 2)
            _write_hmul2(h1_b, sc_n2, 2, 2)
            _write_hmul2(h1_c, sc_n3, 0, 3)
            _write_hmul2(h1_d, sc_n3, 2, 3)
            return

        # ============================================================
        # MODE 0 (baseline): per-element fp32 multiply.  Uses the same
        # h0_*/h1_* fp16x2 values from the byte-decode above.
        # ============================================================
        scale_n0 = cvt_e4m3_to_f32_via_f16(sf_byte_0) * alpha_val
        scale_n1 = cvt_e4m3_to_f32_via_f16(sf_byte_1) * alpha_val
        scale_n2 = cvt_e4m3_to_f32_via_f16(sf_byte_2) * alpha_val
        scale_n3 = cvt_e4m3_to_f32_via_f16(sf_byte_3) * alpha_val

        def _write_pair(h2, scale, mma_i_low, nn):
            f_lo, f_hi = f16x2_to_f32x2(h2)
            tCrB[mma_i_low, nn, k_block_idx] = (f_lo * scale).to(self.b_compute_dtype)
            tCrB[mma_i_low + 1, nn, k_block_idx] = (f_hi * scale).to(self.b_compute_dtype)

        _write_pair(h0_a, scale_n0, 0, 0)
        _write_pair(h0_b, scale_n0, 2, 0)
        _write_pair(h0_c, scale_n1, 0, 1)
        _write_pair(h0_d, scale_n1, 2, 1)
        _write_pair(h1_a, scale_n2, 0, 2)
        _write_pair(h1_b, scale_n2, 2, 2)
        _write_pair(h1_c, scale_n3, 0, 3)
        _write_pair(h1_d, scale_n3, 2, 3)

    # ------------------------------------------------------------------
    # TVM-FFI entry: takes (m, k) A, (k//16, n*2) B (Marlin int32),
    # (k//16, n) B_sf (E4M3), (m, n) C, (1,) alpha, with explicit batch L.
    # Re-wraps each input with explicit (m, k, l) / (k_tiles, n2, l) /
    # (k_sf, n, l) / (m, n, l) layouts the kernel mainloop expects.
    # ------------------------------------------------------------------
    @cute.jit
    def wrapper(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mB_sf: cute.Tensor,
        mC: cute.Tensor,
        mAlpha: cute.Tensor,
        l: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        current_stream,
    ):
        """W4A16 mm_fp4-layout wrapper for the FlashInfer compile interface.

        Args:
            mA:     (m, k) input tensor A, bf16 or fp16.
            mB:     (n, k // 2) uint8 packed FP4 (= mat2_fp4).
            mB_sf:  1-D byte buffer of the 128x4-swizzled FP8-E4M3 SF
                    tensor (= mat2_inv_s.view(-1) from nvfp4_quantize
                    with sfLayout=layout_128x4).
            mC:     (m, n) output tensor C, bf16 or fp16.
            mAlpha: (1,) fp32 scalar (= 1 / global_sf_B for nvfp4).
        """
        m = cute.size(mA, mode=[0])
        k = cute.size(mA, mode=[1])
        n = cute.size(mC, mode=[1])
        k_half = k // _FP4_PER_BYTE
        # SF gmem layout is (n_blocks, k_sf_blocks, 512) of uint8, where
        # each 512-byte block is one 128x4 swizzle block.  n_blocks =
        # ceil(N, 128) / 128 and k_sf_blocks = ceil(K/16, 4) / 4.  For
        # supported shapes (N%128==0, K%64==0) this is exact division.
        n_blocks = n // _SF_BLOCK_N
        k_sf_blocks = k // (self.GROUP_SIZE * _SF_BLOCK_KSF)

        a_tensor = cute.make_tensor(
            mA.iterator,
            layout=cute.make_ordered_layout((m, k, l), order=(1, 0, 2)),
        )
        # B underlying storage is (N, K/2, L) row-major (K/2 stride-1).
        b_tensor = cute.make_tensor(
            mB.iterator,
            layout=cute.make_ordered_layout((n, k_half, l), order=(1, 0, 2)),
        )
        # SF: re-view as (n_blocks, k_sf_blocks, 512) so TMA can chunk
        # by swizzle block.  Each block is one 128x4-swizzled chunk; the
        # dequant function reads bytes per-thread using the same intra-
        # block swizzle formula.
        b_sf_tensor = cute.make_tensor(
            mB_sf.iterator,
            layout=cute.make_ordered_layout(
                (n_blocks, k_sf_blocks, _SF_BLOCK_BYTES), order=(2, 1, 0)
            ),
        )
        c_tensor = cute.make_tensor(
            mC.iterator,
            layout=cute.make_ordered_layout((m, n, l), order=(1, 0, 2)),
        )

        self(
            a_tensor,
            b_tensor,
            b_sf_tensor,
            c_tensor,
            mAlpha,
            max_active_clusters,
            current_stream,
        )
