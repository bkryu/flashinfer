/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cuda/barrier>
#include <cuda/std/utility>

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_fp8.h"
#include "cuda_pipeline.h"
#include "tvm_ffi_utils.h"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__device__ void ldmatrix4_u32(uint32_t rv[4], uint32_t smem_ptr) {
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(rv[0]), "=r"(rv[1]), "=r"(rv[2]), "=r"(rv[3])
               : "r"(smem_ptr));
}

template <typename OutType>
__device__ __forceinline__ float out_to_float(OutType value);

template <>
__device__ __forceinline__ float out_to_float<half>(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float out_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename OutType>
__device__ __forceinline__ OutType float_to_out(float value);

template <>
__device__ __forceinline__ half float_to_out<half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_out<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

__device__ void HMMA_1616K32(float d[8], uint32_t const a[4], uint32_t const b[4], float c[8]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),
        "f"(c[2]), "f"(c[3]));
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2]), "r"(b[3]), "f"(c[4]), "f"(c[5]),
        "f"(c[6]), "f"(c[7]));
}

__device__ void bar_wait(uint32_t bar_ptr, int phase) {
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(bar_ptr),
      "r"(phase));
}

__device__ bool bar_try_wait(uint32_t bar_ptr, int phase) {
  uint32_t success;
  asm volatile(
      "{\n\t"
      ".reg .pred P1; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P1; \n\t"
      "}"
      : "=r"(success)
      : "r"(bar_ptr), "r"(phase));
  return success;
}

__device__ uint32_t elect_one_sync() {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "     elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "     mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}
#endif

template <typename OutType, int WARP_TILE_M, int TILE_M, int TILE_N, int TILE_K, int STAGES,
          int STAGE_UNROLL, bool USE_PDL = false, bool HAS_BIAS = true>
__global__ __launch_bounds__(384, 1) void tinygemm_fp8_kernel(
    OutType* output, const __nv_fp8_e4m3* weights, const __nv_fp8_e4m3* activations,
    const float* a_scale, const float* b_scale, const OutType* bias, int M, int N, int K,
    const __grid_constant__ CUtensorMap weight_map,
    const __grid_constant__ CUtensorMap activation_map) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  extern __shared__ __align__(128) char smem[];

  __nv_fp8_e4m3* sh_weights = reinterpret_cast<__nv_fp8_e4m3*>(&smem[0]);
  __nv_fp8_e4m3* sh_activations = reinterpret_cast<__nv_fp8_e4m3*>(
      &smem[STAGES * STAGE_UNROLL * TILE_M * TILE_K * sizeof(__nv_fp8_e4m3)]);

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar_wt_ready[STAGES];
  __shared__ barrier bar_act_ready[STAGES];
  __shared__ barrier bar_data_consumed[STAGES];

  __shared__ float4 reduction_buffer[256];
  __shared__ OutType sh_bias[TILE_M];
  __shared__ float sh_scale;

  if (threadIdx.x == 0) {
    for (int i = 0; i < STAGES; i++) {
      init(&bar_wt_ready[i], 1);
      init(&bar_act_ready[i], 1);
      init(&bar_data_consumed[i], 32);
    }
    sh_scale = a_scale[0] * b_scale[0];
    ptx::fence_proxy_async(ptx::space_shared);
    asm volatile("prefetch.tensormap [%0];"
                 :
                 : "l"(reinterpret_cast<uint64_t>(&weight_map))
                 : "memory");
    asm volatile("prefetch.tensormap [%0];"
                 :
                 : "l"(reinterpret_cast<uint64_t>(&activation_map))
                 : "memory");
  }
  __syncthreads();

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int phase = 0;

  int mib = blockIdx.x * TILE_M;
  int ni = blockIdx.y * TILE_N;

  float accum[8];
  for (int i = 0; i < 8; i++) accum[i] = 0.f;

  int const K_LOOPS_DMA = (K + 4 * TILE_K * STAGE_UNROLL - 1) / (4 * (TILE_K * STAGE_UNROLL));
  int const K_LOOPS_COMPUTE = K_LOOPS_DMA;

  if (warp_id >= 4) {
    if (elect_one_sync()) {
      int stage = warp_id % 4;
      bool weight_warp = warp_id < 8;
      if constexpr (USE_PDL) {
        if (!weight_warp) {
          cudaGridDependencySynchronize();
          cudaTriggerProgrammaticLaunchCompletion();
        }
      }

      for (int ki = 0; ki < K_LOOPS_DMA; ki++) {
        int k = (ki * 4 + (warp_id % 4)) * TILE_K * STAGE_UNROLL;

        uint64_t desc_ptr_wt = reinterpret_cast<uint64_t>(&weight_map);
        uint64_t desc_ptr_act = reinterpret_cast<uint64_t>(&activation_map);

        uint32_t bar_ptr_wt = __cvta_generic_to_shared(&bar_wt_ready[stage]);
        uint32_t bar_ptr_act = __cvta_generic_to_shared(&bar_act_ready[stage]);
        int bytes_wt = TILE_M * TILE_K * sizeof(__nv_fp8_e4m3);
        int bytes_act = TILE_N * TILE_K * sizeof(__nv_fp8_e4m3);

        bar_wait(__cvta_generic_to_shared(&bar_data_consumed[stage]), phase ^ 1);

        if (weight_warp)
          asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                       :
                       : "r"(bar_ptr_wt), "r"(STAGE_UNROLL * bytes_wt));
        if (!weight_warp)
          asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                       :
                       : "r"(bar_ptr_act), "r"(STAGE_UNROLL * bytes_act));

        for (int i = 0; i < STAGE_UNROLL; i++) {
          uint32_t smem_ptr_wt =
              __cvta_generic_to_shared(&sh_weights[(stage * STAGE_UNROLL + i) * TILE_M * TILE_K]);
          uint32_t crd0 = k + i * TILE_K;
          uint32_t crd1 = mib;
          if (weight_warp)
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], "
                "[%1, {%3,%4}], [%2];"
                :
                : "r"(smem_ptr_wt), "l"(desc_ptr_wt), "r"(bar_ptr_wt), "r"(crd0), "r"(crd1)
                : "memory");

          uint32_t smem_ptr_act = __cvta_generic_to_shared(
              &sh_activations[(stage * STAGE_UNROLL + i) * TILE_N * TILE_K]);
          crd0 = k + i * TILE_K;
          crd1 = ni;
          if (!weight_warp)
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], "
                "[%1, {%3,%4}], [%2];"
                :
                : "r"(smem_ptr_act), "l"(desc_ptr_act), "r"(bar_ptr_act), "r"(crd0), "r"(crd1)
                : "memory");
        }

        stage += 4;
        if (stage >= STAGES) {
          stage = warp_id % 4;
          phase ^= 1;
        }
      }

      for (int i = 0; i < (STAGES / 4) - 1; i++) {
        bar_wait(__cvta_generic_to_shared(&bar_data_consumed[stage]), phase ^ 1);
        stage += 4;
        if (stage >= STAGES) {
          stage = warp_id % 4;
          phase ^= 1;
        }
      }
    }
  } else {
    if constexpr (HAS_BIAS) {
      if (threadIdx.x < TILE_M) sh_bias[threadIdx.x] = bias[mib + threadIdx.x];
    }

    int stage = warp_id;
    int phase = 0;

    int rA = lane_id % 16;
    int cA = lane_id / 16;
    int rB = (lane_id / 16) * 8 + lane_id % 8;
    int cB = (lane_id % 16) / 8;

    int row_offset_wt = (reinterpret_cast<uintptr_t>(sh_weights) / 128) % 8;
    int row_offset_act = (reinterpret_cast<uintptr_t>(sh_activations) / 128) % 8;

    uint32_t bar_ptr_wt = __cvta_generic_to_shared(&bar_wt_ready[stage]);
    uint32_t bar_ptr_act = __cvta_generic_to_shared(&bar_act_ready[stage]);

    bool weight_ready = bar_try_wait(bar_ptr_wt, phase);
    bool act_ready = bar_try_wait(bar_ptr_act, phase);

#pragma unroll 2
    for (int ki = 0; ki < K_LOOPS_COMPUTE; ki++) {
      int next_stage = stage + 4;
      int next_phase = phase;
      if (next_stage >= STAGES) {
        next_stage = warp_id;
        next_phase ^= 1;
      }

      while (!weight_ready || !act_ready) {
        weight_ready = bar_try_wait(bar_ptr_wt, phase);
        act_ready = bar_try_wait(bar_ptr_act, phase);
      }

      if (ki + 1 < K_LOOPS_COMPUTE) {
        weight_ready =
            bar_try_wait(__cvta_generic_to_shared(&bar_wt_ready[next_stage]), next_phase);
        act_ready = bar_try_wait(__cvta_generic_to_shared(&bar_act_ready[next_stage]), next_phase);
      }

#pragma unroll
      for (int su = 0; su < STAGE_UNROLL; su++) {
        __nv_fp8_e4m3* ptr_weights = &sh_weights[(stage * STAGE_UNROLL + su) * TILE_M * TILE_K];
        __nv_fp8_e4m3* ptr_act = &sh_activations[(stage * STAGE_UNROLL + su) * TILE_N * TILE_K];

#pragma unroll
        for (int kii = 0; kii < TILE_K / 32; kii++) {
          uint32_t a[4];
          uint32_t b[4];

          int col = 2 * kii + cA;
          int col_sw = ((rA + row_offset_wt) % 8) ^ col;
          ldmatrix4_u32(a, __cvta_generic_to_shared(&ptr_weights[rA * TILE_K + col_sw * 16]));

          col = 2 * kii + cB;
          col_sw = ((rB + row_offset_act) % 8) ^ col;
          ldmatrix4_u32(b, __cvta_generic_to_shared(&ptr_act[rB * TILE_K + col_sw * 16]));

          HMMA_1616K32(accum, a, b, accum);
        }
      }

      uint32_t bar_c = __cvta_generic_to_shared(&bar_data_consumed[stage]);
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(bar_c));

      stage = next_stage;
      phase = next_phase;
    }

    reduction_buffer[threadIdx.x] = make_float4(accum[0], accum[1], accum[2], accum[3]);
    reduction_buffer[128 + threadIdx.x] = make_float4(accum[4], accum[5], accum[6], accum[7]);
  }

  __syncthreads();

  if (warp_id == 0) {
    int tm = mib + lane_id / 4;
    int tn0 = ni + 2 * (lane_id % 4);
    int tn1 = ni + 8 + 2 * (lane_id % 4);

    float4 accum_lo_1 = reduction_buffer[32 + threadIdx.x];
    float4 accum_lo_2 = reduction_buffer[64 + threadIdx.x];
    float4 accum_lo_3 = reduction_buffer[96 + threadIdx.x];
    float4 accum_hi_1 = reduction_buffer[160 + threadIdx.x];
    float4 accum_hi_2 = reduction_buffer[192 + threadIdx.x];
    float4 accum_hi_3 = reduction_buffer[224 + threadIdx.x];

    accum[0] += accum_lo_1.x + accum_lo_2.x + accum_lo_3.x;
    accum[1] += accum_lo_1.y + accum_lo_2.y + accum_lo_3.y;
    accum[2] += accum_lo_1.z + accum_lo_2.z + accum_lo_3.z;
    accum[3] += accum_lo_1.w + accum_lo_2.w + accum_lo_3.w;
    accum[4] += accum_hi_1.x + accum_hi_2.x + accum_hi_3.x;
    accum[5] += accum_hi_1.y + accum_hi_2.y + accum_hi_3.y;
    accum[6] += accum_hi_1.z + accum_hi_2.z + accum_hi_3.z;
    accum[7] += accum_hi_1.w + accum_hi_2.w + accum_hi_3.w;

    float bias_lo = 0.f;
    float bias_hi = 0.f;
    if constexpr (HAS_BIAS) {
      bias_lo = out_to_float(sh_bias[tm - mib]);
      bias_hi = out_to_float(sh_bias[tm + 8 - mib]);
    }
    float scale = sh_scale;

    if (tn0 < N && tm < M) output[tn0 * M + tm] = float_to_out<OutType>(accum[0] * scale + bias_lo);
    if (tn0 + 1 < N && tm < M)
      output[(tn0 + 1) * M + tm] = float_to_out<OutType>(accum[1] * scale + bias_lo);
    if (tn0 < N && tm + 8 < M)
      output[tn0 * M + tm + 8] = float_to_out<OutType>(accum[2] * scale + bias_hi);
    if (tn0 + 1 < N && tm + 8 < M)
      output[(tn0 + 1) * M + tm + 8] = float_to_out<OutType>(accum[3] * scale + bias_hi);

    if (tn1 < N && tm < M) output[tn1 * M + tm] = float_to_out<OutType>(accum[4] * scale + bias_lo);
    if (tn1 + 1 < N && tm < M)
      output[(tn1 + 1) * M + tm] = float_to_out<OutType>(accum[5] * scale + bias_lo);
    if (tn1 < N && tm + 8 < M)
      output[tn1 * M + tm + 8] = float_to_out<OutType>(accum[6] * scale + bias_hi);
    if (tn1 + 1 < N && tm + 8 < M)
      output[(tn1 + 1) * M + tm + 8] = float_to_out<OutType>(accum[7] * scale + bias_hi);
  }
#endif
}

namespace flashinfer {
namespace tinygemm_fp8 {

static int get_max_dynamic_smem() {
  static int cached = -1;
  if (cached >= 0) return cached;
  int device;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&cached, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  return cached;
}

template <typename OutType, int STAGES, bool HAS_BIAS>
void launch_tinygemm_fp8_impl(const __nv_fp8_e4m3* gA, const __nv_fp8_e4m3* gB, OutType* gC,
                              const float* a_scale, const float* b_scale, const OutType* bias,
                              int batch_size, int output_features, int input_features,
                              cudaStream_t stream, bool use_pdl) {
  static int const WARP_TILE_M = 16;
  static int const TILE_M = WARP_TILE_M;
  static int const TILE_N = 16;
  static int const TILE_K = 128;
  static int const STAGE_UNROLL = 4;

  CUtensorMap weight_map{};
  CUtensorMap activation_map{};

  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {(uint64_t)input_features, (uint64_t)output_features};
  uint64_t stride[rank - 1] = {input_features * sizeof(__nv_fp8_e4m3)};
  uint32_t box_size[rank] = {TILE_K, TILE_M};
  uint32_t elem_stride[rank] = {1, 1};

  CUresult res =
      cuTensorMapEncodeTiled(&weight_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, rank,
                             const_cast<__nv_fp8_e4m3*>(gB), size, stride, box_size, elem_stride,
                             CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                             CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                             CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(res == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for weight_map with error code " << res;

  size[1] = batch_size;
  box_size[1] = TILE_N;
  res = cuTensorMapEncodeTiled(&activation_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
                               rank, const_cast<__nv_fp8_e4m3*>(gA), size, stride, box_size,
                               elem_stride, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                               CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                               CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(res == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for activation_map with error code " << res;

  int smem_size =
      STAGES * STAGE_UNROLL *
      (TILE_M * TILE_K * sizeof(__nv_fp8_e4m3) + TILE_N * TILE_K * sizeof(__nv_fp8_e4m3));

  int tiles_m = (output_features + TILE_M - 1) / TILE_M;
  int tiles_n = (batch_size + TILE_N - 1) / TILE_N;

  dim3 grid(tiles_m, tiles_n);
  dim3 block(384);

  if (use_pdl) {
    auto status =
        cudaFuncSetAttribute(tinygemm_fp8_kernel<OutType, WARP_TILE_M, TILE_M, TILE_N, TILE_K,
                                                 STAGES, STAGE_UNROLL, true, HAS_BIAS>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "cudaFuncSetAttribute failed: " << cudaGetErrorString(status);

    cudaLaunchConfig_t config;
    cudaLaunchAttribute attrs[1];
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    status = cudaLaunchKernelEx(&config,
                                &tinygemm_fp8_kernel<OutType, WARP_TILE_M, TILE_M, TILE_N, TILE_K,
                                                     STAGES, STAGE_UNROLL, true, HAS_BIAS>,
                                gC, gB, gA, a_scale, b_scale, bias, output_features, batch_size,
                                input_features, weight_map, activation_map);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "cudaLaunchKernelEx failed: " << cudaGetErrorString(status);
  } else {
    auto status =
        cudaFuncSetAttribute(tinygemm_fp8_kernel<OutType, WARP_TILE_M, TILE_M, TILE_N, TILE_K,
                                                 STAGES, STAGE_UNROLL, false, HAS_BIAS>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "cudaFuncSetAttribute failed: " << cudaGetErrorString(status);

    tinygemm_fp8_kernel<OutType, WARP_TILE_M, TILE_M, TILE_N, TILE_K, STAGES, STAGE_UNROLL, false,
                        HAS_BIAS><<<grid, block, smem_size, stream>>>(
        gC, gB, gA, a_scale, b_scale, bias, output_features, batch_size, input_features, weight_map,
        activation_map);
    status = cudaGetLastError();
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "tinygemm_fp8_kernel launch failed: " << cudaGetErrorString(status);
  }
}

template <typename OutType, bool HAS_BIAS>
void launch_tinygemm_fp8(const __nv_fp8_e4m3* gA, const __nv_fp8_e4m3* gB, OutType* gC,
                         const float* a_scale, const float* b_scale, const OutType* bias,
                         int batch_size, int output_features, int input_features,
                         cudaStream_t stream, bool use_pdl) {
  static constexpr int SMEM_PER_STAGE_GROUP = 4 * (16 * 128 + 16 * 128);  // 16384 bytes

  int max_smem = get_max_dynamic_smem();
  int max_stages = max_smem / SMEM_PER_STAGE_GROUP;

  if (max_stages >= 16) {
    launch_tinygemm_fp8_impl<OutType, 16, HAS_BIAS>(gA, gB, gC, a_scale, b_scale, bias, batch_size,
                                                    output_features, input_features, stream,
                                                    use_pdl);
  } else if (max_stages >= 12) {
    launch_tinygemm_fp8_impl<OutType, 12, HAS_BIAS>(gA, gB, gC, a_scale, b_scale, bias, batch_size,
                                                    output_features, input_features, stream,
                                                    use_pdl);
  } else if (max_stages >= 8) {
    launch_tinygemm_fp8_impl<OutType, 8, HAS_BIAS>(gA, gB, gC, a_scale, b_scale, bias, batch_size,
                                                   output_features, input_features, stream,
                                                   use_pdl);
  } else if (max_stages >= 4) {
    launch_tinygemm_fp8_impl<OutType, 4, HAS_BIAS>(gA, gB, gC, a_scale, b_scale, bias, batch_size,
                                                   output_features, input_features, stream,
                                                   use_pdl);
  } else {
    TVM_FFI_ICHECK(false) << "Device has insufficient shared memory for tinygemm_fp8 kernel ("
                          << max_smem << " bytes available, minimum " << 4 * SMEM_PER_STAGE_GROUP
                          << " bytes required)";
  }
}

void tinygemm_fp8_op(TensorView input, TensorView weight, TensorView a_scale, TensorView b_scale,
                     TensorView bias, TensorView output, bool use_pdl) {
  auto stream = get_stream(input.device());

  int batch_size = input.shape()[0];
  int input_features = input.shape()[1];
  int output_features = weight.shape()[1];

  switch (encode_dlpack_dtype(output.dtype())) {
    case float16_code:
      launch_tinygemm_fp8<half, true>(reinterpret_cast<const __nv_fp8_e4m3*>(input.data_ptr()),
                                      reinterpret_cast<const __nv_fp8_e4m3*>(weight.data_ptr()),
                                      reinterpret_cast<half*>(output.data_ptr()),
                                      reinterpret_cast<const float*>(a_scale.data_ptr()),
                                      reinterpret_cast<const float*>(b_scale.data_ptr()),
                                      reinterpret_cast<const half*>(bias.data_ptr()), batch_size,
                                      output_features, input_features, stream, use_pdl);
      break;
    case bfloat16_code:
      launch_tinygemm_fp8<__nv_bfloat16, true>(
          reinterpret_cast<const __nv_fp8_e4m3*>(input.data_ptr()),
          reinterpret_cast<const __nv_fp8_e4m3*>(weight.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
          reinterpret_cast<const float*>(a_scale.data_ptr()),
          reinterpret_cast<const float*>(b_scale.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()), batch_size, output_features,
          input_features, stream, use_pdl);
      break;
    default:
      TVM_FFI_LOG_AND_THROW(ValueError) << "tinygemm_fp8 output dtype must be fp16 or bfloat16";
  }
}

void tinygemm_fp8_nobias_op(TensorView input, TensorView weight, TensorView a_scale,
                            TensorView b_scale, TensorView output, bool use_pdl) {
  auto stream = get_stream(input.device());

  int batch_size = input.shape()[0];
  int input_features = input.shape()[1];
  int output_features = weight.shape()[1];

  switch (encode_dlpack_dtype(output.dtype())) {
    case float16_code:
      launch_tinygemm_fp8<half, false>(reinterpret_cast<const __nv_fp8_e4m3*>(input.data_ptr()),
                                       reinterpret_cast<const __nv_fp8_e4m3*>(weight.data_ptr()),
                                       reinterpret_cast<half*>(output.data_ptr()),
                                       reinterpret_cast<const float*>(a_scale.data_ptr()),
                                       reinterpret_cast<const float*>(b_scale.data_ptr()),
                                       static_cast<const half*>(nullptr), batch_size,
                                       output_features, input_features, stream, use_pdl);
      break;
    case bfloat16_code:
      launch_tinygemm_fp8<__nv_bfloat16, false>(
          reinterpret_cast<const __nv_fp8_e4m3*>(input.data_ptr()),
          reinterpret_cast<const __nv_fp8_e4m3*>(weight.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
          reinterpret_cast<const float*>(a_scale.data_ptr()),
          reinterpret_cast<const float*>(b_scale.data_ptr()),
          static_cast<const __nv_bfloat16*>(nullptr), batch_size, output_features, input_features,
          stream, use_pdl);
      break;
    default:
      TVM_FFI_LOG_AND_THROW(ValueError) << "tinygemm_fp8 output dtype must be fp16 or bfloat16";
  }
}

}  // namespace tinygemm_fp8
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tinygemm_fp8_op, flashinfer::tinygemm_fp8::tinygemm_fp8_op);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tinygemm_fp8_nobias_op,
                              flashinfer::tinygemm_fp8::tinygemm_fp8_nobias_op);
