# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# Ported from LightLM's cutlass_dsl megakernel study (docs/megakernel-plan.md there) for the FlashInfer experimental
# track: the Qwen3.8-27B one-launch decoder on SM120.
"""cutlass_dsl decode W8A16 GEMM for SM120 (Phase C.4): fp8 [N,K] weight
streamed as the swap-AB MMA A operand, dequantized to bf16 in registers.

Problem: y[M,N] = x[M,K](bf16) @ dequant(W)^T, W fp8 e4m3 stored
CHECKPOINT-NATIVE [N,K] (K contiguous, bryu ruling 2026-09-08), one fp32
scale per fused shard applied in the epilogue. Decode M <= 16.

Structure (subclass of the B.1 kernel `LtlmSm120Gemm`, swap-AB form):
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
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils

from .gemm_sm120 import LtlmSm120Gemm

MAX_SHARDS = 8


class W8A16SwapSm120(LtlmSm120Gemm):
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
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,  # container view of the permuted weight: (N, K/2, 1) k-major bf16
        b: cute.Tensor,  # activations (M, K, 1) k-major bf16
        c: cute.Tensor,  # out^T (N, M, 1) m-major
        alpha: cute.Tensor,  # fp32 [nsh] per-shard weight scales
        bounds: cute.Tensor,  # int32 [MAX_SHARDS] interior row boundaries (padded)
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype, self.b_dtype, self.c_dtype = (
            a.element_type,
            b.element_type,
            c.element_type,
        )
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self._setup_attributes()
        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a, self.a_smem_layout_staged, (self.a_tile_mnk[0], self.a_tile_mnk[2]), 1
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c, self.epi_smem_layout_staged, self.epi_tile
        )
        tile_sched_params, grid = self._compute_grid(
            c, self.tile_shape_mnk, max_active_clusters
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
            tma_atom_c,
            tma_tensor_c,
            alpha,
            bounds,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
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
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        alpha: cute.Tensor,
        bounds: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
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
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        # A tiles in CONTAINER units, B/C in logical units
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.a_tile_mnk, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        thr_mma = tiled_mma.get_slice(tidx)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        tCsA = thr_mma.partition_A(
            sA
        )  # container fragments: ((2,2,2), MMA_M, KC_BLOCKS, STAGES)
        # K-permuted view of the activation tile: MMA k-slot (e, j', kh) -> memory
        # k = e + 4j' + 2kh (see module docstring); N and STAGE modes identity
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
        tCsB = thr_mma.partition_B(
            sB_perm
        )  # ((2,2), MMA_N, K_BLOCKS, STAGES), permuted k
        tCrA_c = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        # logical A fragments for the two k16 blocks of one container block
        tCrA_log = cute.make_rmem_tensor(
            cute.make_layout(((2, 2, 2), cute.size(tCrA_c, mode=[1]), 2)), self.b_dtype
        )

        tCgC = thr_mma.partition_C(gC_mnl)
        accumulators = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)

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

        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()  # PDL: before alpha/bounds reads and every C store
            num_kc_blocks = cute.size(tCrA_c, mode=[2])
            mma_m = cute.size(tCrA_c, mode=[1])

            atom_ldsm_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_ldsm_A, tiled_mma)
            thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
            tCsA_copy_view = thr_copy_A.partition_S(sA)
            tCrA_copy_view = thr_copy_A.retile(tCrA_c)

            # 32-bit views for the container -> logical register shuffle
            tCrA_log32 = cute.recast_tensor(
                tCrA_log, cutlass.Int32
            )  # ((1,2,2), MMA_M, 2): (., rh, kh) 32-bit regs
            conv16 = cute.make_rmem_tensor((16,), self.b_dtype)
            conv32 = cute.recast_tensor(conv16, cutlass.Int32)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                accumulators.fill(0.0)
                # per-tile shard scale: rows of this tile all lie in one shard
                row0 = tile_coord_mnl[0] * self.tile_shape_mnk[0]
                sh = cutlass.Int32(0)
                for j in cutlass.range_constexpr(self.nsh - 1):
                    if row0 >= bounds[j]:
                        sh = sh + 1
                alpha_val = alpha[sh]

                mainloop_consumer_state.reset_count()
                for _k_tile in range(0, k_tile_cnt, 1, unroll=1):
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
                        # activation fragments: 4 consecutive bf16 per lane from the permuted view
                        cute.autovec_copy(
                            tCsB_p[None, None, 2 * kc], tCrB[None, None, 2 * kc]
                        )
                        cute.autovec_copy(
                            tCsB_p[None, None, 2 * kc + 1], tCrB[None, None, 2 * kc + 1]
                        )
                        # dequant: container fragment (m, kc) -> logical fragments (m, 0/1)
                        for m in cutlass.range_constexpr(mma_m):
                            v8 = tCrA_c[
                                (None, m, kc)
                            ].load()  # 8 x bf16 container slots
                            vf8 = v8.bitcast(
                                cutlass.Float8E4M3FN
                            )  # 16 x fp8, order e_log + 2e + 4rh + 8khc
                            conv16.store(
                                vf8.to(self.b_dtype)
                            )  # 16 x bf16 in that order
                            # conv32[e + 2rh + 4khc] -> log32[(rh, kh=e), m, L=khc]
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
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_launch_dependents()  # PDL: successor may schedule

                # ---------------- epilogue (B.1's, with the shard scale) ----------------
                copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                    self.c_layout, elem_ty_d=self.c_dtype, elem_ty_acc=self.acc_dtype
                )
                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
                    self.c_dtype,
                )
                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s, tiled_copy_C_Atom
                )
                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)
                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
                size_tRS_rD = cute.size(tRS_rD)
                sepi = cute.group_modes(sC, 0, 2)
                tcgc = cute.zipped_divide(gC_mnl_slice, self.epi_tile)
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c, 0, cute.make_layout(1), sepi, tcgc
                )
                epi_tile_num = cute.size(tcgc, mode=[1])
                epi_tile_shape = tcgc.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(1, epi_tile_shape[0])
                )
                tma_store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.epi_stage,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        self.num_mma_warps * self.num_threads_per_warp,
                    ),
                )
                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
                    tRS_rD_out = cute.make_rmem_tensor(
                        tRS_rD_layout.shape, self.c_dtype
                    )
                    acc_vec = tRS_rD.load() * alpha_val
                    tRS_rD_out.store(acc_vec.to(self.c_dtype))
                    epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    if warp_idx == 0:
                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                tma_store_pipeline.producer_tail()
        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            if cutlass.const_expr(self.pdl == 1):
                cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.pdl == 2):
                # PDL mode 2: the first tile's first stages of A (fp8 weight
                # container, static) go out BEFORE the wait; B (activations)
                # after it, on the same per-stage full barriers
                if not work_tile.is_valid_tile:
                    cute.arch.griddepcontrol_wait()
                tc0 = work_tile.tile_idx
                tAgA0 = tAgA[(None, tc0[0], None, tc0[2])]
                tBgB0 = tBgB[(None, tc0[1], None, tc0[2])]
                mainloop_producer_state.reset_count()
                npre = cutlass.min(cutlass.Int32(self.ab_stage), k_tile_cnt)
                for _k_tile in range(0, npre, 1, unroll=1):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    bar = mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    )
                    cute.copy(
                        tma_atom_a,
                        tAgA0[(None, mainloop_producer_state.count)],
                        tAsA[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=bar,
                    )
                    mainloop_producer_state.advance()
                cute.arch.griddepcontrol_wait()
                pre_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                for _k_tile in range(0, npre, 1, unroll=1):
                    bar = mainloop_pipeline.producer_get_barrier(pre_state)
                    cute.copy(
                        tma_atom_b,
                        tBgB0[(None, pre_state.count)],
                        tBsB[(None, pre_state.index)],
                        tma_bar_ptr=bar,
                    )
                    mainloop_pipeline.producer_commit(pre_state)
                    pre_state.advance()
                for _k_tile in range(npre, k_tile_cnt, 1, unroll=1):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    bar = mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    )
                    cute.copy(
                        tma_atom_a,
                        tAgA0[(None, mainloop_producer_state.count)],
                        tAsA[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=bar,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB0[(None, mainloop_producer_state.count)],
                        tBsB[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=bar,
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                mainloop_producer_state.reset_count()
                for _k_tile in range(0, k_tile_cnt, 1, unroll=1):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    bar = mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    )
                    cute.copy(
                        tma_atom_a,
                        tAgA_mkl[(None, mainloop_producer_state.count)],
                        tAsA[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=bar,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_nkl[(None, mainloop_producer_state.count)],
                        tBsB[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=bar,
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return
