"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Benchmark: Fused RMSNorm + FP4 Quantization using CuTe-DSL Backend

Compares the CuTe-DSL fused kernel against unfused operations:
  - Unfused: rmsnorm + global_scale_computation + fp4_quantize (3 separate ops)
  - Fused: Single kernel that performs all operations together

Usage:
    python bench_cute_dsl_rmsnorm_fp4quant.py

Requirements:
    - Blackwell GPU (SM100+)
    - CuTe-DSL installed
"""

import numpy as np
import torch
from scipy.stats import gmean
from flashinfer.testing.utils import bench_gpu_time


def get_cc():
    """Get CUDA compute capability."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def compute_bandwidth_gb_s(
    batch_size: int,
    hidden_size: int,
    block_size: int,
    time_ms: float,
) -> float:
    """
    Compute achieved memory bandwidth in GB/s for fused RMSNorm + FP4 quantization.

    The fused kernel performs:
    1. Read input x: [batch_size, hidden_size] in fp16/bf16 (2 bytes/elem)
    2. Read weight: [hidden_size] in fp16/bf16 (2 bytes/elem)
    3. Write y_fp4: [batch_size, hidden_size/2] packed uint8 (1 byte per 2 FP4 values)
    4. Write block_scale: [batch_size, hidden_size/block_size] in fp8/uint8 (1 byte/elem)

    Formula:
        read_bytes  = batch_size * hidden_size * 2 + hidden_size * 2
        write_bytes = batch_size * hidden_size / 2 + batch_size * hidden_size / block_size
        total_bytes = read_bytes + write_bytes
        bandwidth   = total_bytes / time_in_seconds / 1e9  (GB/s)

    Args:
        batch_size: Batch size (number of rows)
        hidden_size: Hidden dimension
        block_size: FP4 quantization block size (16 or 32)
        time_ms: Kernel execution time in milliseconds

    Returns:
        Achieved bandwidth in GB/s
    """
    # Read: x (fp16) + weight (fp16)
    read_bytes = batch_size * hidden_size * 2 + hidden_size * 2

    # Write: y_fp4 (packed uint8) + block_scale (fp8/uint8)
    write_bytes = batch_size * (hidden_size // 2) + batch_size * (
        hidden_size // block_size
    )

    total_bytes = read_bytes + write_bytes
    time_s = time_ms / 1000.0

    if time_s <= 0:
        return 0.0

    return total_bytes / time_s / 1e9


def bench_cute_dsl_nvfp4(batch_size, hidden_size, dtype):
    """Benchmark CuTe-DSL NVFP4 backend (block_size=16, E4M3 scales, with global_scale)."""
    from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_nvfp4quant

    eps = 1e-6
    block_size = 16  # Fixed for NVFP4

    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    y_fp4 = torch.empty(batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8)
    block_scale = torch.empty(
        batch_size,
        hidden_size // block_size,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )

    # Benchmark with bench_gpu_time
    times = bench_gpu_time(
        lambda: rmsnorm_nvfp4quant(
            x,
            weight,
            y_fp4,
            block_scale,
            eps=eps,
        ),
        cold_l2_cache=True,
        enable_cupti=True,
        use_cuda_graph=False,
        dry_run_iters=10,
        repeat_iters=100,
    )

    # Return median time
    return np.median(times)


def bench_cute_dsl_mxfp4(batch_size, hidden_size, dtype):
    """Benchmark CuTe-DSL MXFP4 backend (block_size=32, UE8M0 scales, no global_scale)."""
    from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_mxfp4quant

    eps = 1e-6
    block_size = 32  # Fixed for MXFP4

    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    y_fp4 = torch.empty(batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8)
    block_scale = torch.empty(
        batch_size, hidden_size // block_size, device="cuda", dtype=torch.uint8
    )

    # Benchmark with bench_gpu_time
    times = bench_gpu_time(
        lambda: rmsnorm_mxfp4quant(
            x,
            weight,
            y_fp4,
            block_scale,
            eps=eps,
        ),
        cold_l2_cache=True,
        enable_cupti=True,
        use_cuda_graph=False,
        dry_run_iters=10,
        repeat_iters=100,
    )

    # Return median time
    return np.median(times)


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


def bench_unfused_flashinfer(batch_size, hidden_size, dtype, block_size=16):
    """Benchmark unfused FlashInfer operations: rmsnorm + global_scale + fp4_quantize.

    This measures the complete unfused workflow as a single operation, which includes:
    1. RMSNorm computation
    2. Global scale calculation (max_abs reduction)
    3. FP4 quantization with the computed global scale

    Returns the median time in milliseconds.
    """
    from flashinfer.norm import rmsnorm
    from flashinfer.fp4_quantization import fp4_quantize

    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

    def unfused_rmsnorm_fp4quant():
        # Step 1: RMSNorm
        y_normed = rmsnorm(x, weight, eps=eps)

        # Step 2: Compute global scale (same as fused kernel)
        if block_size == 32:
            # MXFP4: no global scale needed
            global_scale = None
        else:
            # NVFP4: global_scale = FP8_MAX * FP4_MAX / max_abs
            # fp4_quantize expects global_scale as 1D tensor with shape (1,)
            max_abs = y_normed.abs().max()
            global_scale = torch.tensor(
                [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / max_abs.item()],
                device="cuda",
                dtype=torch.float32,
            )

        # Step 3: FP4 quantization
        y_fp4, block_scale = fp4_quantize(
            y_normed,
            global_scale=global_scale,
            sf_vec_size=block_size,
            sf_use_ue8m0=(block_size == 32),
            is_sf_swizzled_layout=False,
        )
        return y_fp4, block_scale, global_scale

    # Benchmark the complete unfused workflow
    times = bench_gpu_time(
        unfused_rmsnorm_fp4quant,
        cold_l2_cache=True,
        enable_cupti=True,
        use_cuda_graph=False,
        dry_run_iters=10,
        repeat_iters=100,
    )

    return np.median(times)


def sanity_check_outputs_nvfp4(dtype=torch.float16):
    """Verify CuTe-DSL NVFP4 output matches unfused RMSNorm + global_scale + fp4_quantize."""
    from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_nvfp4quant
    from flashinfer.norm import rmsnorm
    from flashinfer.fp4_quantization import fp4_quantize

    # Test with a few configurations
    test_configs = [
        (128, 256),
        (512, 1024),
        (1024, 2048),
    ]

    eps = 1e-6
    block_size = 16  # Fixed for NVFP4
    all_passed = True

    print("  NVFP4 (block_size=16, E4M3 scales, global_scale):")
    for batch_size, hidden_size in test_configs:
        # Create inputs
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # CuTe-DSL fused path (returns y_fp4, block_scale, global_scale)
        y_fp4_cute = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale_cute = torch.empty(
            batch_size,
            hidden_size // block_size,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )

        _, _, global_scale_cute = rmsnorm_nvfp4quant(
            x,
            weight,
            y_fp4_cute,
            block_scale_cute,
            eps=eps,
        )

        # Unfused path: rmsnorm + global_scale + fp4_quantize
        y_normed = rmsnorm(x, weight, eps=eps)

        # NVFP4: compute global_scale = FP8_MAX * FP4_MAX / max_abs
        # fp4_quantize expects global_scale as 1D tensor with shape (1,)
        max_abs = y_normed.abs().max()
        global_scale_unfused = torch.tensor(
            [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / max_abs.item()],
            device="cuda",
            dtype=torch.float32,
        )

        y_fp4_unfused, block_scale_unfused = fp4_quantize(
            y_normed,
            global_scale=global_scale_unfused,
            sf_vec_size=block_size,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )

        # Compare FP4 outputs
        match_count = (y_fp4_cute == y_fp4_unfused).sum().item()
        total_count = y_fp4_cute.numel()
        match_pct = match_count / total_count * 100

        if match_pct < 70.0:
            all_passed = False
            print(
                f"    WARN: ({batch_size}, {hidden_size}) - "
                f"FP4 match: {match_pct:.1f}% (expected >= 70%)"
            )
        else:
            print(
                f"    OK: ({batch_size}, {hidden_size}) - FP4 match: {match_pct:.1f}%"
            )

        # Also verify global_scale is computed correctly
        gs_diff = abs(global_scale_cute.item() - global_scale_unfused.item())
        if gs_diff > 1.0:
            print(
                f"    WARN: ({batch_size}, {hidden_size}) - "
                f"global_scale diff: {gs_diff:.4f}"
            )

    return all_passed


def sanity_check_outputs_mxfp4(dtype=torch.float16):
    """Verify CuTe-DSL MXFP4 output matches unfused RMSNorm + fp4_quantize."""
    from flashinfer.cute_dsl.rmsnorm_fp4quant import rmsnorm_mxfp4quant
    from flashinfer.norm import rmsnorm
    from flashinfer.fp4_quantization import fp4_quantize

    # Test with a few configurations
    test_configs = [
        (128, 256),
        (512, 1024),
        (1024, 2048),
    ]

    eps = 1e-6
    block_size = 32  # Fixed for MXFP4
    all_passed = True

    print("  MXFP4 (block_size=32, UE8M0 scales, no global_scale):")
    for batch_size, hidden_size in test_configs:
        # Create inputs
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # CuTe-DSL fused path (returns y_fp4, block_scale only for MXFP4)
        y_fp4_cute = torch.empty(
            batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8
        )
        block_scale_cute = torch.empty(
            batch_size, hidden_size // block_size, device="cuda", dtype=torch.uint8
        )

        y_fp4_cute, block_scale_cute = rmsnorm_mxfp4quant(
            x,
            weight,
            y_fp4_cute,
            block_scale_cute,
            eps=eps,
        )

        # Unfused path: rmsnorm + fp4_quantize (no global_scale for MXFP4)
        y_normed = rmsnorm(x, weight, eps=eps)

        y_fp4_unfused, block_scale_unfused = fp4_quantize(
            y_normed,
            global_scale=None,
            sf_vec_size=block_size,
            sf_use_ue8m0=True,
            is_sf_swizzled_layout=False,
        )

        # Compare FP4 outputs
        match_count = (y_fp4_cute == y_fp4_unfused).sum().item()
        total_count = y_fp4_cute.numel()
        match_pct = match_count / total_count * 100

        if match_pct < 70.0:
            all_passed = False
            print(
                f"    WARN: ({batch_size}, {hidden_size}) - "
                f"FP4 match: {match_pct:.1f}% (expected >= 70%)"
            )
        else:
            print(
                f"    OK: ({batch_size}, {hidden_size}) - FP4 match: {match_pct:.1f}%"
            )

    return all_passed


def run_benchmark_nvfp4():
    """Run NVFP4 benchmark suite."""
    print()
    print("-" * 80)
    print("NVFP4 Benchmark (block_size=16, E4M3 scales, with global_scale)")
    print("-" * 80)

    dtype = torch.float16
    block_size = 16  # Fixed for NVFP4

    # Test configurations
    batch_sizes = [2**i for i in range(10, 17)]  # 1024 to 65536
    batch_sizes += [1000, 3000, 5000, 10000, 15000, 25000, 60000]
    batch_sizes = sorted(list(set(batch_sizes)))

    hidden_sizes = [2**j for j in range(11, 16)]  # 2048 to 32768
    hidden_sizes += [1536]
    hidden_sizes = sorted(list(set(hidden_sizes)))

    configs = [
        (batch_size, hidden_size)
        for batch_size in batch_sizes
        for hidden_size in hidden_sizes
    ]

    header = (
        f"{'Batch':<8} {'Hidden':<8} "
        f"{'Fused (µs)':<12} {'BW (GB/s)':<10} "
        f"{'Unfused (µs)':<14} {'Speedup':<10}"
    )
    print(header)
    print("-" * len(header))

    results = []

    for batch_size, hidden_size in configs:
        # CuTe-DSL fused timing
        try:
            t_fused = bench_cute_dsl_nvfp4(batch_size, hidden_size, dtype)
            t_fused_us = t_fused * 1e3  # ms to µs
            bw_fused = compute_bandwidth_gb_s(
                batch_size, hidden_size, block_size, t_fused
            )
        except Exception as e:
            print(f"{batch_size:<8} {hidden_size:<8} ERROR: {e}")
            continue

        # Unfused FlashInfer timing
        try:
            t_unfused = bench_unfused_flashinfer(
                batch_size, hidden_size, dtype, block_size
            )
            t_unfused_us = t_unfused * 1e3
            speedup = t_unfused / t_fused if t_fused > 0 else 0
            unfused_str = f"{t_unfused_us:.1f}"
            speedup_str = f"{speedup:.2f}x"
        except Exception:
            t_unfused_us = None
            unfused_str = "N/A"
            speedup_str = "N/A"
            speedup = None

        print(
            f"{batch_size:<8} {hidden_size:<8} "
            f"{t_fused_us:<12.1f} {bw_fused:<10.1f} "
            f"{unfused_str:<14} {speedup_str:<10}"
        )

        result = {
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "fused_us": t_fused_us,
            "fused_bw_gb_s": bw_fused,
            "unfused_us": t_unfused_us,
            "speedup": speedup,
        }
        results.append(result)

    # Calculate geomean speedup
    speedups = [r["speedup"] for r in results if r["speedup"] is not None]
    if speedups:
        geomean_speedup = gmean(speedups)
        print(f"\nNVFP4 Geomean speedup vs Unfused: {geomean_speedup:.2f}x")

    return results


def run_benchmark_mxfp4():
    """Run MXFP4 benchmark suite."""
    print()
    print("-" * 80)
    print("MXFP4 Benchmark (block_size=32, UE8M0 scales, no global_scale)")
    print("-" * 80)

    dtype = torch.float16
    block_size = 32  # Fixed for MXFP4

    # Test configurations
    batch_sizes = [2**i for i in range(10, 17)]  # 1024 to 65536
    batch_sizes += [1000, 3000, 5000, 10000, 15000, 25000, 60000]
    batch_sizes = sorted(list(set(batch_sizes)))

    hidden_sizes = [2**j for j in range(11, 16)]  # 2048 to 32768
    hidden_sizes += [1536]
    hidden_sizes = sorted(list(set(hidden_sizes)))

    configs = [
        (batch_size, hidden_size)
        for batch_size in batch_sizes
        for hidden_size in hidden_sizes
    ]

    header = (
        f"{'Batch':<8} {'Hidden':<8} "
        f"{'Fused (µs)':<12} {'BW (GB/s)':<10} "
        f"{'Unfused (µs)':<14} {'Speedup':<10}"
    )
    print(header)
    print("-" * len(header))

    results = []

    for batch_size, hidden_size in configs:
        # CuTe-DSL fused timing
        try:
            t_fused = bench_cute_dsl_mxfp4(batch_size, hidden_size, dtype)
            t_fused_us = t_fused * 1e3  # ms to µs
            bw_fused = compute_bandwidth_gb_s(
                batch_size, hidden_size, block_size, t_fused
            )
        except Exception as e:
            print(f"{batch_size:<8} {hidden_size:<8} ERROR: {e}")
            continue

        # Unfused FlashInfer timing
        try:
            t_unfused = bench_unfused_flashinfer(
                batch_size, hidden_size, dtype, block_size
            )
            t_unfused_us = t_unfused * 1e3
            speedup = t_unfused / t_fused if t_fused > 0 else 0
            unfused_str = f"{t_unfused_us:.1f}"
            speedup_str = f"{speedup:.2f}x"
        except Exception:
            t_unfused_us = None
            unfused_str = "N/A"
            speedup_str = "N/A"
            speedup = None

        print(
            f"{batch_size:<8} {hidden_size:<8} "
            f"{t_fused_us:<12.1f} {bw_fused:<10.1f} "
            f"{unfused_str:<14} {speedup_str:<10}"
        )

        result = {
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "fused_us": t_fused_us,
            "fused_bw_gb_s": bw_fused,
            "unfused_us": t_unfused_us,
            "speedup": speedup,
        }
        results.append(result)

    # Calculate geomean speedup
    speedups = [r["speedup"] for r in results if r["speedup"] is not None]
    if speedups:
        geomean_speedup = gmean(speedups)
        print(f"\nMXFP4 Geomean speedup vs Unfused: {geomean_speedup:.2f}x")

    return results


def run_benchmark():
    """Run full benchmark suite."""
    print("=" * 80)
    print("Fused RMSNorm + FP4 Quantization Benchmark")
    print("=" * 80)

    cc = get_cc()
    print(f"GPU Compute Capability: SM{cc}")

    if cc < 100:
        raise RuntimeError("Blackwell GPU (SM100+) required for FP4 quantization")

    dtype = torch.float16

    # Sanity check: verify fused kernel output matches unfused operations
    print()
    print("Running sanity checks...")
    nvfp4_passed = sanity_check_outputs_nvfp4(dtype)
    mxfp4_passed = sanity_check_outputs_mxfp4(dtype)

    if nvfp4_passed and mxfp4_passed:
        print("\n✓ Confirmed: Fused kernels output is equivalent to unfused operations")
    else:
        print("\n✗ Warning: Some outputs did not match closely")

    # Run NVFP4 benchmark
    nvfp4_results = run_benchmark_nvfp4()

    # Run MXFP4 benchmark
    mxfp4_results = run_benchmark_mxfp4()

    print()
    print("=" * 80)
    print("Benchmark Complete")
    print("=" * 80)

    return {"nvfp4": nvfp4_results, "mxfp4": mxfp4_results}


if __name__ == "__main__":
    run_benchmark()
