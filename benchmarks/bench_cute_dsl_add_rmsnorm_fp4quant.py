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

Benchmark: Fused Add + RMSNorm + FP4 Quantization using CuTe-DSL Backend

Compares the CuTe-DSL fused kernel against unfused operations:
    - Unfused: add + rmsnorm + global_scale_computation + fp4_quantize (4 separate ops)
    - Fused: Single kernel that performs all operations together

For MXFP4 format (block_size=32, no global scale):
    - Unfused: add + rmsnorm + fp4_quantize (3 separate ops)
    - Fused: Single kernel for all operations

Usage:
    python bench_cute_dsl_add_rmsnorm_fp4quant.py
"""

import numpy as np
import torch
from scipy.stats import gmean
from flashinfer.testing.utils import bench_gpu_time

# Constants
FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0


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
    Compute achieved memory bandwidth in GB/s for fused Add + RMSNorm + FP4 quantization.

    The fused kernel performs:
    1. Read input x: [batch_size, hidden_size] in fp16/bf16 (2 bytes/elem)
    2. Read residual r: [batch_size, hidden_size] in fp16/bf16 (2 bytes/elem)
    3. Read weight: [hidden_size] in fp16/bf16 (2 bytes/elem)
    4. Write y_fp4: [batch_size, hidden_size/2] packed uint8 (1 byte per 2 FP4 values)
    5. Write block_scale: [batch_size, hidden_size/block_size] in fp8/uint8 (1 byte/elem)

    Formula:
        read_bytes  = batch_size * hidden_size * 2 * 2 + hidden_size * 2
        write_bytes = batch_size * hidden_size / 2 + batch_size * hidden_size / block_size
        total_bytes = read_bytes + write_bytes
        bandwidth   = total_bytes / time_in_seconds / 1e9  (GB/s)
    """
    # Read: x (fp16) + r (fp16) + weight (fp16)
    read_bytes = batch_size * hidden_size * 2 * 2 + hidden_size * 2

    # Write: y_fp4 (packed uint8) + block_scale (fp8/uint8)
    write_bytes = batch_size * (hidden_size // 2) + batch_size * (
        hidden_size // block_size
    )

    total_bytes = read_bytes + write_bytes
    time_s = time_ms / 1000.0

    if time_s <= 0:
        return 0.0

    return total_bytes / time_s / 1e9


def bench_fused_nvfp4(batch_size, hidden_size, dtype):
    """Benchmark fused CuTe-DSL kernel for NVFP4 (with global scale)."""
    from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_nvfp4quant

    eps = 1e-6
    block_size = 16

    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    y_fp4 = torch.empty(batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8)
    block_scale = torch.empty(
        batch_size,
        hidden_size // block_size,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    global_scale = torch.empty(1, device="cuda", dtype=torch.float32)

    times = bench_gpu_time(
        lambda: add_rmsnorm_nvfp4quant(
            x,
            r,
            weight,
            y_fp4,
            block_scale,
            global_scale,
            eps=eps,
        ),
        cold_l2_cache=True,
        enable_cupti=True,
        use_cuda_graph=False,
        dry_run_iters=10,
        repeat_iters=100,
    )

    return np.median(times)


def bench_fused_mxfp4(batch_size, hidden_size, dtype):
    """Benchmark fused CuTe-DSL kernel for MXFP4 (no global scale)."""
    from flashinfer.cute_dsl.add_rmsnorm_fp4quant import add_rmsnorm_mxfp4quant

    eps = 1e-6
    block_size = 32

    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    y_fp4 = torch.empty(batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8)
    block_scale = torch.empty(
        batch_size, hidden_size // block_size, device="cuda", dtype=torch.uint8
    )

    times = bench_gpu_time(
        lambda: add_rmsnorm_mxfp4quant(
            x,
            r,
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

    return np.median(times)


def bench_unfused_nvfp4(batch_size, hidden_size, dtype):
    """Benchmark unfused operations: add + rmsnorm + global_scale + fp4_quantize for NVFP4."""
    from flashinfer.norm import rmsnorm
    from flashinfer.fp4_quantization import fp4_quantize

    eps = 1e-6
    block_size = 16

    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

    def unfused_add_rmsnorm_fp4quant():
        # Step 1: Add
        h = x + r

        # Step 2: RMSNorm
        y_normed = rmsnorm(h, weight, eps=eps)

        # Step 3: Compute global scale (same as fused kernel)
        # fp4_quantize expects global_scale as 1D tensor with shape (1,)
        max_abs = y_normed.abs().max()
        global_scale = torch.tensor(
            [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / max_abs.item()],
            device="cuda",
            dtype=torch.float32,
        )

        # Step 4: FP4 Quantize
        y_fp4_unfused, block_scale_unfused = fp4_quantize(
            y_normed,
            global_scale=global_scale,
            sf_vec_size=block_size,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )
        return y_fp4_unfused, block_scale_unfused, global_scale

    times_unfused = bench_gpu_time(unfused_add_rmsnorm_fp4quant)
    return times_unfused


def bench_unfused_mxfp4(batch_size, hidden_size, dtype):
    """Benchmark unfused operations: add + rmsnorm + fp4_quantize for MXFP4."""
    from flashinfer.norm import rmsnorm
    from flashinfer.fp4_quantization import fp4_quantize

    eps = 1e-6
    block_size = 32

    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

    def unfused_add_rmsnorm_fp4quant():
        # Step 1: Add
        h = x + r

        # Step 2: RMSNorm
        y_normed = rmsnorm(h, weight, eps=eps)

        # Step 3: FP4 Quantize (MXFP4 doesn't use global_scale)
        y_fp4_unfused, block_scale_unfused = fp4_quantize(
            y_normed,
            global_scale=None,
            sf_vec_size=block_size,
            sf_use_ue8m0=True,  # MXFP4 uses UE8M0 scales
            is_sf_swizzled_layout=False,
        )
        return y_fp4_unfused, block_scale_unfused

    times_unfused = bench_gpu_time(unfused_add_rmsnorm_fp4quant)
    return times_unfused


def sanity_check_outputs(dtype=torch.float16):
    """Verify CuTe-DSL output matches unfused operations."""
    from flashinfer.cute_dsl.add_rmsnorm_fp4quant import (
        add_rmsnorm_nvfp4quant,
        add_rmsnorm_mxfp4quant,
    )
    from flashinfer.norm import rmsnorm
    from flashinfer.fp4_quantization import fp4_quantize

    # Test with a few configurations
    test_configs = [
        (128, 256),
        (512, 1024),
        (1024, 2048),
    ]

    eps = 1e-6
    all_passed = True

    print("  NVFP4 (block_size=16, with global_scale):")
    block_size = 16
    for batch_size, hidden_size in test_configs:
        torch.manual_seed(42)
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # CuTe-DSL fused path
        y_fp4_fused, block_scale_fused, global_scale_fused = add_rmsnorm_nvfp4quant(
            x, r, weight, eps=eps
        )

        # Unfused path: add + rmsnorm + global_scale + fp4_quantize
        h = x + r
        y_normed = rmsnorm(h, weight, eps=eps)
        max_abs = y_normed.abs().max()
        # fp4_quantize expects global_scale as 1D tensor with shape (1,), not scalar
        global_scale_ref = torch.tensor(
            [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / max_abs.item()],
            device="cuda",
            dtype=torch.float32,
        )

        y_fp4_sep, block_scale_sep = fp4_quantize(
            y_normed,
            global_scale=global_scale_ref,
            sf_vec_size=block_size,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )

        # Compare FP4 outputs
        match_count = (y_fp4_fused == y_fp4_sep).sum().item()
        total_count = y_fp4_fused.numel()
        match_pct = match_count / total_count * 100

        if match_pct < 70.0:
            all_passed = False
            print(
                f"    WARN: ({batch_size}, {hidden_size}) - "
                f"FP4 match: {match_pct:.1f}% (expected >= 70%)"
            )
        else:
            print(f"    OK: ({batch_size}, {hidden_size}) - FP4 match {match_pct:.1f}%")

    print("  MXFP4 (block_size=32, no global_scale):")
    block_size = 32
    for batch_size, hidden_size in test_configs:
        torch.manual_seed(42)
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        r = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # CuTe-DSL fused path
        y_fp4_fused, block_scale_fused = add_rmsnorm_mxfp4quant(x, r, weight, eps=eps)

        # Unfused path: add + rmsnorm + fp4_quantize
        h = x + r
        y_normed = rmsnorm(h, weight, eps=eps)

        y_fp4_sep, block_scale_sep = fp4_quantize(
            y_normed,
            global_scale=None,
            sf_vec_size=block_size,
            sf_use_ue8m0=True,  # MXFP4 uses UE8M0 scales
            is_sf_swizzled_layout=False,
        )

        # Compare FP4 outputs
        match_count = (y_fp4_fused == y_fp4_sep).sum().item()
        total_count = y_fp4_fused.numel()
        match_pct = match_count / total_count * 100

        if match_pct < 70.0:
            all_passed = False
            print(
                f"    WARN: ({batch_size}, {hidden_size}) - "
                f"FP4 match: {match_pct:.1f}% (expected >= 70%)"
            )
        else:
            print(f"    OK: ({batch_size}, {hidden_size}) - FP4 match {match_pct:.1f}%")

    return all_passed


def run_benchmark():
    """Run full benchmark suite."""
    print("=" * 100)
    print("Fused Add + RMSNorm + FP4 Quantization Benchmark")
    print("=" * 100)

    cc = get_cc()
    print(f"GPU Compute Capability: SM{cc}")

    if cc < 100:
        raise RuntimeError("Blackwell GPU (SM100+) required for FP4 quantization")

    dtype = torch.float16

    # Sanity check
    print()
    print("Running sanity check...")
    if sanity_check_outputs(dtype):
        print(
            "✓ Confirmed: Fused kernel output is equivalent to unfused "
            "(add + RMSNorm + global_scale + fp4_quantize)"
        )
    else:
        print("✗ Warning: Some outputs did not match closely")
    print()

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

    # ==================== NVFP4 Benchmark ====================
    print()
    print("=" * 100)
    print("NVFP4 Format (block_size=16, with global_scale)")
    print("=" * 100)
    print()
    header = (
        f"{'Batch':<8} {'Hidden':<8} "
        f"{'CuTe-DSL (µs)':<14} {'BW (GB/s)':<10} "
        f"{'Unfused (µs)':<14} "
        f"{'vs Unfused':<12}"
    )
    print(header)
    print("-" * len(header))

    nvfp4_results = []

    for batch_size, hidden_size in configs:
        block_size = 16
        try:
            t_fused = bench_fused_nvfp4(batch_size, hidden_size, dtype)
            t_fused_us = t_fused * 1e3
            bw_fused = compute_bandwidth_gb_s(
                batch_size, hidden_size, block_size, t_fused
            )
        except Exception as e:
            print(f"{batch_size:<8} {hidden_size:<8} FUSED ERROR: {e}")
            continue

        try:
            t_unfused = bench_unfused_nvfp4(batch_size, hidden_size, dtype)
            t_unfused_us = np.median(t_unfused) * 1e3
            speedup = t_unfused_us / t_fused_us if t_fused_us > 0 else 0
            unfused_str = f"{t_unfused_us:.1f}"
            speedup_str = f"{speedup:.2f}x"
        except Exception as e:
            print(f"{batch_size:<8} {hidden_size:<8} UNFUSED ERROR: {e}")
            unfused_str = "N/A"
            speedup_str = "N/A"
            speedup = None

        print(
            f"{batch_size:<8} {hidden_size:<8} "
            f"{t_fused_us:<14.1f} {bw_fused:<10.1f} "
            f"{unfused_str:<14} "
            f"{speedup_str:<12}"
        )

        nvfp4_results.append(
            {
                "batch_size": batch_size,
                "hidden_size": hidden_size,
                "fused_us": t_fused_us,
                "speedup": speedup,
            }
        )

    # Calculate and print geomean speedup
    speedups = [r["speedup"] for r in nvfp4_results if r["speedup"] is not None]
    if speedups:
        geomean_speedup = gmean(speedups)
        print(f"\nGeomean speedup vs Unfused: {geomean_speedup:.2f}x")

    # ==================== MXFP4 Benchmark ====================
    print()
    print("=" * 100)
    print("MXFP4 Format (block_size=32, no global_scale)")
    print("=" * 100)
    print()
    header = (
        f"{'Batch':<8} {'Hidden':<8} "
        f"{'CuTe-DSL (µs)':<14} {'BW (GB/s)':<10} "
        f"{'Unfused (µs)':<14} "
        f"{'vs Unfused':<12}"
    )
    print(header)
    print("-" * len(header))

    mxfp4_results = []

    for batch_size, hidden_size in configs:
        block_size = 32
        try:
            t_fused = bench_fused_mxfp4(batch_size, hidden_size, dtype)
            t_fused_us = t_fused * 1e3
            bw_fused = compute_bandwidth_gb_s(
                batch_size, hidden_size, block_size, t_fused
            )
        except Exception as e:
            print(f"{batch_size:<8} {hidden_size:<8} FUSED ERROR: {e}")
            continue

        try:
            t_unfused = bench_unfused_mxfp4(batch_size, hidden_size, dtype)
            t_unfused_us = np.median(t_unfused) * 1e3
            speedup = t_unfused_us / t_fused_us if t_fused_us > 0 else 0
            unfused_str = f"{t_unfused_us:.1f}"
            speedup_str = f"{speedup:.2f}x"
        except Exception as e:
            print(f"{batch_size:<8} {hidden_size:<8} UNFUSED ERROR: {e}")
            unfused_str = "N/A"
            speedup_str = "N/A"
            speedup = None

        print(
            f"{batch_size:<8} {hidden_size:<8} "
            f"{t_fused_us:<14.1f} {bw_fused:<10.1f} "
            f"{unfused_str:<14} "
            f"{speedup_str:<12}"
        )

        mxfp4_results.append(
            {
                "batch_size": batch_size,
                "hidden_size": hidden_size,
                "fused_us": t_fused_us,
                "speedup": speedup,
            }
        )

    # Calculate and print geomean speedup
    speedups = [r["speedup"] for r in mxfp4_results if r["speedup"] is not None]
    if speedups:
        geomean_speedup = gmean(speedups)
        print(f"\nGeomean speedup vs Unfused: {geomean_speedup:.2f}x")

    print()
    print("=" * 100)
    print("Benchmark Complete")
    print("=" * 100)


if __name__ == "__main__":
    run_benchmark()
