# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# Qwen3.8-27B one-launch decoder on SM120 (FlashInfer experimental track).
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

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

"""SM120 dense GEMM kernel (bf16/fp16, fp32 acc).

Derived from NVIDIA CUTLASS `examples/python/CuTeDSL/cute/blackwell_geforce/
kernel/dense_gemm/dense_gemm.py` (BSD-3-Clause, header above), which is the
verified SM120 recipe: TMA-fed `PipelineTmaAsync` mainloop, one DMA warp +
`atom_layout` MMA warps on `mma.sync.m16n8k16` (`cute.nvgpu.warp.
MmaF16BF16Op`), ldmatrix operand fetch, `stmatrix` + TMA-store epilogue,
static persistent tile scheduler, `setmaxregister` 40/232 split.

Changes vs the upstream example:
  * `epi_stage`, `atom_layout`, `occupancy` are constructor parameters
    (the example fixes 8 / (2,2,1) / 1). The example's epilogue staging
    (8 x 64x32 subtiles = the WHOLE C tile, 32 KB at 128x128) is what
    starves the A/B ring to 2 stages.
  * Stage-safe TMA-store epilogue: the example only had warp 0 wait on
    the store pipeline before a staging buffer is reused, so with fewer
    staging buffers than C subtiles the other MMA warps overwrote a
    buffer still being read by TMA (measured: wrong results for every
    epi_stage < subtiles). All MMA warps now rendezvous on the named
    barrier after warp 0's `producer_acquire()`.
  * Host CLI / reference-check harness removed; the host drives it from
    `cutlass_dsl/gemm.py` (dynamic layouts, compile cache, raced configs).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils


class Sm120DenseGemm:
    def __init__(
        self,
        acc_dtype,
        tile_shape_mnk,
        epi_stage: int = 2,
        atom_layout=(2, 2, 1),
        occupancy: int = 1,
        pdl: int = 0,
        pdl_static: str = "b",
    ):
        self.acc_dtype = acc_dtype
        # PDL (kernels/pdl.py): 0 off; 1 wait at entry; 2 the first stages of
        # the STATIC operand (weights: A under swap-AB, B otherwise) are
        # TMA'd before the wait, the other operand after it
        self.pdl = int(pdl)
        self.pdl_static = pdl_static
        self.epi_stage_cfg = int(epi_stage)
        self.cluster_shape_mnk = (1, 1, 1)
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.tiled_mma = None
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False

        self.occupancy = int(occupancy)
        # TODO: remove this hard code for user input ?
        self.atom_layout = tuple(atom_layout)
        self.num_mma_warps = (
            self.atom_layout[0] * self.atom_layout[1] * self.atom_layout[2]
        )
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + 1  # 1 warp for DMA
        ) * self.num_threads_per_warp
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
        # TODO: remove this hard code for user input ?
        self.mma_inst_mnk = (16, 8, 16)
        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.a_dtype,
            self.acc_dtype,
            self.mma_inst_mnk,
        )
        tC = cute.make_layout(self.atom_layout)
        permutation_mnk = (
            self.atom_layout[0] * self.mma_inst_mnk[0],
            # TODO: to leverage ldmatrix.x4, when self.atom_layout[1] is 1, mma tile is ((8x16)x2)
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
            self.tile_shape_mnk, self.c_dtype, is_cooperative=False
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
            self.epi_stage_cfg,
        )
        if self.ab_stage < 1:
            raise ValueError(
                f"config does not fit SM120 SMEM: tile {self.tile_shape_mnk}, "
                f"epi_stage {self.epi_stage_cfg}, occupancy {self.occupancy}"
            )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(
            self.a_dtype.width == 16 and self.a_dtype != self.b_dtype
        ):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16 or float8")
        if cutlass.const_expr(self.b_dtype.width != 16 and self.b_dtype.width != 8):
            raise TypeError("b_dtype should be float16 or float8")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
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

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.cta_layout_mnk,
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
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_c: TMA copy atom for C tensor
        :type tma_atom_c: cute.CopyAtom
        :param mC_mnl: Output tensor C
        :type mC_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        """

        # ///////////////////////////////////////////////////////////////////////////////
        #  Get cta/warp/thread idx
        # ///////////////////////////////////////////////////////////////////////////////
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        # bidx, bidy, bidz = cute.arch.block_idx()
        # bdimx, bdimy, bdimz = cute.arch.grid_dim()

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get mcast mask
        # ///////////////////////////////////////////////////////////////////////////////
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )

        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        # Each warp will constribute to the arrive count with the number of mcast size
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

        #  Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # ///////////////////////////////////////////////////////////////////////////////
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition global tensor for TiledMMA_A/B/C
        # //////////////////////////////////////////////////////////////////////////////
        thr_mma = tiled_mma.get_slice(tidx)

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for TMA load A/B
        # //////////////////////////////////////////////////////////////////////////////
        #  TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        #  Make frangments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # cluster wait for barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            pipeline.sync(barrier_id=1)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Create the tile scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # Create the pipeline states for producer and consumer
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # MMA warp group
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()  # PDL: before every C store

            num_k_blocks = cute.size(tCrA, mode=[2])

            # ///////////////////////////////////////////////////////////////////////////////
            # Copy Atom A/B retiling for TMA load A/B
            # ///////////////////////////////////////////////////////////////////////////////
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)

            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)

            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                # Clear the accumulator
                accumulators.fill(0.0)

                # /////////////////////////////////////////////////////////////////////////////
                #  Pipelined MAINLOOP
                # /////////////////////////////////////////////////////////////////////////////

                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                #  Wait for TMA copies to complete
                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                # tCsA_p: (MMA, (4, MMA_M / 4), MMA_K), tCsA_p: (MMA, (4, MMA_N / 4), MMA_K)
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

                for _k_tile in range(0, k_tile_cnt - 1, 1, unroll=1):
                    # unroll the loop
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )

                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )

                            # tCsA_p: (MMA, (4, MMA_M / 4), MMA_K), tCsA_p: (MMA, (4, MMA_N / 4), MMA_K)
                            tCsA_p = tCsA_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsB_p = tCsB_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )

                        # Copy data from smem to tCrA/tCrB for the next k_block
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                        # Gemm of the current k_block
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[None, None, k_block_idx],
                            tCrB[None, None, k_block_idx],
                            accumulators,
                        )
                # end of for loop
                # Hoist out last k_tile
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )

                    if k_block_idx == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                    # Gemm of the current k_block
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        tCrA[None, None, k_block_idx],
                        tCrB[None, None, k_block_idx],
                        accumulators,
                    )

                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_launch_dependents()  # PDL: successor may schedule

                # /////////////////////////////////////////////////////////////////////////////
                #  EPILOG
                # /////////////////////////////////////////////////////////////////////////////

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

                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s,
                    tiled_copy_C_Atom,
                )

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                # (R2S, R2S_M, R2S_N, PIPE_D)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                # (R2S, R2S_M, R2S_N)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                # Allocate D registers.
                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
                size_tRS_rD = cute.size(tRS_rD)

                sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                tcgc_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)

                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    sepi_for_tma_partition,
                    tcgc_for_tma_partition,
                )

                epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])
                epi_tile_shape = tcgc_for_tma_partition.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(1, epi_tile_shape[0])
                )

                # Initialize tma store pipeline
                tma_store_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.num_mma_warps * self.num_threads_per_warp,
                )
                tma_store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.epi_stage,
                    producer_group=tma_store_producer_group,
                )

                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    # Copy from accumulators to D registers
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                    # Type conversion
                    tRS_rD_out = cute.make_rmem_tensor(
                        tRS_rD_layout.shape, self.c_dtype
                    )
                    acc_vec = tRS_rD.load()
                    tRS_rD_out.store(acc_vec.to(self.c_dtype))

                    # Register to shared memory
                    epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )

                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    # barrier for sync
                    self.epilog_sync_barrier.arrive_and_wait()

                    # Get the global memory coordinate for the current epi tile.
                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    # Copy from shared memory to global memory
                    if warp_idx == 0:
                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()
                    # with epi_stage < number of C subtiles the
                    # buffer written next iteration is one TMA is still
                    # reading unless EVERY MMA warp waits for warp 0's
                    # acquire above (the example let them run ahead)
                    self.epilog_sync_barrier.arrive_and_wait()

                # Advance to the next work tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                tma_store_pipeline.producer_tail()
                # End of for k_tile loop
            # End of while loop
        # End of MMA warp group
        # Start of DMA warp group
        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            if cutlass.const_expr(self.pdl == 1):
                cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.pdl == 2):
                # PDL mode 2: static operand's first stages before the wait
                if not work_tile.is_valid_tile:
                    cute.arch.griddepcontrol_wait()
                tc0 = work_tile.tile_idx
                tAgA0 = tAgA[(None, tc0[0], None, tc0[2])]
                tBgB0 = tBgB[(None, tc0[1], None, tc0[2])]
                mainloop_producer_state.reset_count()
                npre = cutlass.min(cutlass.Int32(self.ab_stage), k_tile_cnt)
                for _k_tile in range(0, npre, 1, unroll=1):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    bar0 = mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    )
                    if cutlass.const_expr(self.pdl_static == "a"):
                        cute.copy(
                            tma_atom_a,
                            tAgA0[(None, mainloop_producer_state.count)],
                            tAsA[(None, mainloop_producer_state.index)],
                            tma_bar_ptr=bar0,
                            mcast_mask=a_mcast_mask,
                        )
                    else:
                        cute.copy(
                            tma_atom_b,
                            tBgB0[(None, mainloop_producer_state.count)],
                            tBsB[(None, mainloop_producer_state.index)],
                            tma_bar_ptr=bar0,
                            mcast_mask=b_mcast_mask,
                        )
                    mainloop_producer_state.advance()
                cute.arch.griddepcontrol_wait()
                pre_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                for _k_tile in range(0, npre, 1, unroll=1):
                    bar1 = mainloop_pipeline.producer_get_barrier(pre_state)
                    if cutlass.const_expr(self.pdl_static == "a"):
                        cute.copy(
                            tma_atom_b,
                            tBgB0[(None, pre_state.count)],
                            tBsB[(None, pre_state.index)],
                            tma_bar_ptr=bar1,
                            mcast_mask=b_mcast_mask,
                        )
                    else:
                        cute.copy(
                            tma_atom_a,
                            tAgA0[(None, pre_state.count)],
                            tAsA[(None, pre_state.index)],
                            tma_bar_ptr=bar1,
                            mcast_mask=a_mcast_mask,
                        )
                    mainloop_pipeline.producer_commit(pre_state)
                    pre_state.advance()
                for _k_tile in range(npre, k_tile_cnt, 1, unroll=1):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    bar2 = mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    )
                    cute.copy(
                        tma_atom_a,
                        tAgA0[(None, mainloop_producer_state.count)],
                        tAsA[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=bar2,
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB0[(None, mainloop_producer_state.count)],
                        tBsB[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=bar2,
                        mcast_mask=b_mcast_mask,
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
                    # /////////////////////////////////////////////////////////////////////////////
                    #  Wait for A/B buffers to be empty before loading into them
                    #  Also sets the transaction barrier for the A/B buffers
                    # /////////////////////////////////////////////////////////////////////////////
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

                    # /////////////////////////////////////////////////////////////////////////////
                    #  Slice to global/shared memref to current k_tile
                    # /////////////////////////////////////////////////////////////////////////////
                    tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                    tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                    tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                    # /////////////////////////////////////////////////////////////////////////////
                    #  TMA load A/B
                    # /////////////////////////////////////////////////////////////////////////////
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask=b_mcast_mask,
                    )
                    # Mainloop pipeline's producer commit is a NOP
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # end of while loop

            # Wait A/B buffer empty
            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: tuple[int, int],
        c_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
        epi_stage: int = 8,
    ) -> tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """
        # epi_stage from the config; the example fixed 8
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
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
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: cute.Layout,
        b_dtype: type[cutlass.Numeric],
        b_layout: cute.Layout,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: cute.Layout,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout for matrix A
        :type a_layout: Layout
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout for matrix B
        :type b_layout: Layout
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param c_dtype: Data type for output matrix C
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: leading dimension of the output matrix C
        :type c_layout: Layout
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )

        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout,
            tile_shape_mnk,
            b_dtype,
            ab_stage,
        )

        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            epi_stage,
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[int, int, int]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]
        """

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
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for C tensor storage.

        :param tensor_c: Output tensor C
        :type tensor_c: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
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
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
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
