"""
Head-to-head top-k benchmark comparing backends.

Backends measured:
  - torch.topk
  - flashinfer.top_k with backend="cuda"  (FI CUDA)
  - flashinfer.top_k with backend="cute-dsl" (FI CuTe DSL)

Usage:
  python benchmarks/bench_topk_backend_comparison.py
  python benchmarks/bench_topk_backend_comparison.py --dtype bf16
  python benchmarks/bench_topk_backend_comparison.py --distribution uniform
  python benchmarks/bench_topk_backend_comparison.py --return-values
"""

import argparse
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


# ---------------------------------------------------------------------------
# Input distribution helpers
# ---------------------------------------------------------------------------


class Distribution(Enum):
    normal = 1
    uniform = 2
    radix_adversarial_20bit = 3


def create_radix_adversarial_20bit_logits(
    num_rows: int, max_len: int, dtype: torch.dtype, device: str
) -> torch.Tensor:
    """Float32 values whose top 20 bits are identical (worst-case for radix sort)."""
    M = 20
    num_low = 32 - M

    base_pattern = 0x3F800000  # float32(1.0)
    high_mask = (0xFFFFFFFF << num_low) & 0xFFFFFFFF
    high_part = base_pattern & high_mask

    low_max = 1 << num_low
    low_bits = torch.randint(0, low_max, (num_rows, max_len), dtype=torch.int64)
    low_bits = (low_bits & (low_max - 1)).to(torch.uint32)

    combined = torch.tensor(high_part, dtype=torch.uint32, device="cpu") | low_bits
    return combined.view(torch.float32).to(dtype=dtype, device=device)


def create_logits(
    num_rows: int,
    num_cols: int,
    dtype: torch.dtype,
    distribution: Distribution,
) -> torch.Tensor:
    if distribution == Distribution.normal:
        return torch.randn(num_rows, num_cols, dtype=dtype, device="cuda")
    elif distribution == Distribution.uniform:
        return torch.rand(num_rows, num_cols, dtype=dtype, device="cuda")
    elif distribution == Distribution.radix_adversarial_20bit:
        return create_radix_adversarial_20bit_logits(num_rows, num_cols, dtype, "cuda")
    raise ValueError(f"Unknown distribution: {distribution}")


# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    Rows: int
    Cols: int
    K: int
    dtype: torch.dtype = torch.float32
    name: str = ""


def bench_single_config(
    config: BenchConfig,
    dry_run_iters: int = 5,
    repeat_iters: int = 100,
    distribution: Distribution = Distribution.normal,
    indices_only: bool = False,
    enable_cupti: bool = True,
):
    torch.manual_seed(1111)
    torch.cuda.manual_seed(1111)

    batch_size = config.Rows
    num_cols = config.Cols
    k = config.K
    dtype = config.dtype

    results = {"config": config}

    logits = create_logits(batch_size, num_cols, dtype, distribution)

    # -- torch.topk -----------------------------------------------------
    measurements = bench_gpu_time(
        lambda: torch.topk(logits, k, dim=-1),
        enable_cupti=enable_cupti,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
    )
    results["torch_us"] = np.median(measurements) * 1e3

    # -- FI CUDA (public API, auto algorithm selection) ------------------
    measurements = bench_gpu_time(
        lambda: flashinfer.top_k(logits, k),
        enable_cupti=enable_cupti,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
    )
    results["fi_cuda_us"] = np.median(measurements) * 1e3

    # -- FI CuTe DSL (requires k <= 2048 and k even) --------------------
    if k <= 2048 and k % 2 == 0:
        try:
            measurements = bench_gpu_time(
                lambda: flashinfer.top_k(
                    logits,
                    k,
                    backend="cute-dsl",
                    return_values=not indices_only,
                ),
                enable_cupti=enable_cupti,
                dry_run_iters=dry_run_iters,
                repeat_iters=repeat_iters,
            )
            results["fi_cute_dsl_us"] = np.median(measurements) * 1e3
        except Exception as e:
            results["fi_cute_dsl_error"] = str(e)

    return results


# ---------------------------------------------------------------------------
# Pretty-print comparison table (streaming: header once, row per config)
# ---------------------------------------------------------------------------

_COL_W = 140


def _fmt_us(val):
    return f"{val:>10.2f}us" if not np.isnan(val) else f"{'N/A':>12}"


def print_table_header(title: str, indices_only: bool = True):
    print("\n" + "=" * _COL_W)
    print(title)
    if indices_only:
        print("  (FI CuTe DSL: indices-only; FI CUDA / torch.topk: values+indices)")
    else:
        print("  (All backends: values+indices)")
    print("=" * _COL_W)

    header = (
        f"{'Config':<25} | {'Torch':>12} | {'FI CUDA':>12} | {'FI CuTe DSL':>12} "
        f"| {'Best':>12} | {'Torch/Best':>10} | {'CUDA/DSL':>10}"
    )
    print(header)
    print("-" * _COL_W)


def print_result_row(res: dict):
    cfg = res["config"]
    config_str = cfg.name if cfg.name else f"{cfg.Rows}-{cfg.Cols}-{cfg.K}"

    torch_us = res.get("torch_us", float("nan"))
    fi_cuda_us = res.get("fi_cuda_us", float("nan"))
    fi_dsl_us = res.get("fi_cute_dsl_us", float("nan"))

    times = {}
    for name, val in [
        ("Torch", torch_us),
        ("FI CUDA", fi_cuda_us),
        ("FI CuTe DSL", fi_dsl_us),
    ]:
        if not np.isnan(val):
            times[name] = val

    if times:
        best_name = min(times, key=times.get)
        best_time = times[best_name]
    else:
        best_name = "N/A"
        best_time = float("nan")

    torch_over_best = torch_us / best_time if best_time > 0 else float("nan")
    cuda_over_dsl = fi_cuda_us / fi_dsl_us if fi_dsl_us > 0 else float("nan")

    print(
        f"{config_str:<25} | {_fmt_us(torch_us)} | {_fmt_us(fi_cuda_us)} | {_fmt_us(fi_dsl_us)} "
        f"| {best_name:>12} | {torch_over_best:>8.2f}x | {cuda_over_dsl:>8.2f}x",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Pre-built config sweeps
# ---------------------------------------------------------------------------


def make_k2048_configs(dtype: torch.dtype):
    """Large sweep matching the upstream benchmark: K=2048, varying Rows/Cols."""
    rows_list = [1, 16, 128, 256, 512, 1024, 2048]
    cols_list = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    configs = []
    for r in rows_list:
        for c in cols_list:
            configs.append(
                BenchConfig(Rows=r, Cols=c, K=2048, dtype=dtype, name=f"{r}-{c}-2048")
            )
    return configs


def make_quick_configs(dtype: torch.dtype):
    """Smaller subset for quick sanity checks."""
    return [
        BenchConfig(Rows=1, Cols=65536, K=2048, dtype=dtype, name="1-65536-2048"),
        BenchConfig(Rows=16, Cols=65536, K=2048, dtype=dtype, name="16-65536-2048"),
        BenchConfig(Rows=128, Cols=131072, K=2048, dtype=dtype, name="128-131072-2048"),
        BenchConfig(Rows=256, Cols=131072, K=2048, dtype=dtype, name="256-131072-2048"),
        BenchConfig(Rows=512, Cols=131072, K=2048, dtype=dtype, name="512-131072-2048"),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_dtype(s: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }[s.lower()]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        description="Head-to-head top-k benchmark (no TRTLLM dependency)"
    )
    parser.add_argument(
        "--dtype", type=str, default="fp32", help="fp32 | fp16 | bf16 (default: fp32)"
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="normal",
        choices=["normal", "uniform", "radix_adversarial_20bit"],
        help="Input data distribution (default: normal)",
    )
    parser.add_argument(
        "--return-values",
        action="store_true",
        help="Benchmark FI CuTe DSL with return_values=True (default: indices-only)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small subset of configs for a quick sanity check",
    )
    parser.add_argument(
        "--dry-run-iters", type=int, default=5, help="Warmup iterations (default: 5)"
    )
    parser.add_argument(
        "--repeat-iters", type=int, default=100, help="Timed iterations (default: 100)"
    )
    parser.add_argument(
        "--no-cupti",
        action="store_true",
        help="Disable CUPTI (use CUDA event timing instead)",
    )
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    distribution = Distribution[args.distribution]
    enable_cupti = not args.no_cupti

    indices_only = not args.return_values
    configs = make_quick_configs(dtype) if args.quick else make_k2048_configs(dtype)

    title = (
        f"Top-K Benchmark (K=2048), dtype={args.dtype}, "
        f"distribution={args.distribution}"
    )
    print_table_header(title, indices_only=indices_only)

    for cfg in configs:
        try:
            res = bench_single_config(
                cfg,
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.repeat_iters,
                distribution=distribution,
                indices_only=indices_only,
                enable_cupti=enable_cupti,
            )
            print_result_row(res)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                config_str = cfg.name or f"{cfg.Rows}-{cfg.Cols}-{cfg.K}"
                print(f"{config_str:<25} | {'OOM':>12}", flush=True)
                torch.cuda.empty_cache()
            else:
                raise


if __name__ == "__main__":
    main()
