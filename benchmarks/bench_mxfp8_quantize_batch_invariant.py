"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Benchmark: MXFP8 quantization, the batch-invariant class vs the CuTe-DSL backend

Times ``flashinfer.mxfp8_quantize(x, backend="auto", batch_invariant=True)`` (the class ``mxfp8.quant`` of
``flashinfer.batch_invariant``) against ``flashinfer.mxfp8_quantize(x, backend="cute-dsl")`` over the M x K grid
of ``bench_nvfp4_quantize_backend_comparison.py`` (22 x 15 = 330 points), for the three scale-factor layouts and
bf16 / f16 / fp32 inputs, with FlashInfer's timing harness exactly as that benchmark calls it: CUPTI kernel time,
5 warm-up and 30 measured iterations, the L2 flushed before every iteration, no CUDA graph, the median.

Every point is checked for correctness before it is timed: the two backends must agree bit for bit on the random
inputs of the grid (they differ only on blocks whose amax is below 448 * 2**-127, which torch.randn never produces,
and on infinite inputs). Each point is banked to the CSV as soon as it is measured, so an interrupted run resumes
where it stopped; ``--budget-s`` stops a run cleanly before a session cap.

Outputs, per (layout, dtype), under ``--out-dir`` (default ``benchmarks/results/batch_invariant/<device>/``):

* ``mxfp8_<layout>_<dtype>.csv``: one row per point (both times, achieved TB/s, the speedup, the dispatcher's
  configuration, the correctness check);
* ``mxfp8_<layout>_<dtype>_speedup.png``: the speedup heatmap (CuTe-DSL time / batch-invariant time; > 1 means the
  batch-invariant kernel is faster; the number in every cell);
* ``mxfp8_<layout>_<dtype>_tbps.png``: the achieved TB/s of the batch-invariant kernel (bytes read + written);
* ``mxfp8_<layout>_<dtype>_summary.json``: per-regime geomeans (decode M <= 16, mid 32 <= M <= 1024, prefill
  M >= 1536), the share of cells at >= 0.95 and the worst cells.

Usage:
    python bench_mxfp8_quantize_batch_invariant.py --layout swizzled_128x4 --dtype bf16
    python bench_mxfp8_quantize_batch_invariant.py --layout linear --dtype f32          # fp32: both paths accept it
    python bench_mxfp8_quantize_batch_invariant.py --layout swizzled_8x4 --dtype f16 --plot-only

Requirements:
    - an SM100 / SM103 / SM107 GPU (both kernels are CuTe-DSL kernels)
    - cupti-python >= 13 (the CUPTI timer; the harness falls back to CUDA events with a warning otherwise)
    - matplotlib for the heatmaps
"""

import argparse
import csv
import json
import math
import os
import time
import warnings

import numpy as np
import torch

from flashinfer import SfLayout

M_VALUES = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    32768,
]
K_VALUES = [
    128,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    5120,
    6144,
    8192,
    12288,
    16384,
]
LAYOUTS = {
    "swizzled_128x4": SfLayout.layout_128x4,
    "swizzled_8x4": SfLayout.layout_8x4,
    "linear": SfLayout.layout_linear,
}
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
REGIMES = (
    ("decode (M <= 16)", lambda m: m <= 16),
    ("mid (32 <= M <= 1024)", lambda m: 32 <= m <= 1024),
    ("prefill (M >= 1536)", lambda m: m >= 1536),
)
FIELDS = [
    "M",
    "K",
    "bi_ms",
    "cute_ms",
    "bi_tbps",
    "cute_tbps",
    "speedup",
    "bi_config",
    "match",
    "bytes",
    "device",
    "arch",
    "sm_count",
    "layout",
    "dtype",
]
DEVICE_NAMES = {
    (10, 0): "B200 (SM100)",
    (10, 3): "B300 (SM103)",
    (10, 7): "Rubin NVL8 (SM107)",
}


def device_name() -> str:
    cc = torch.cuda.get_device_capability()
    return DEVICE_NAMES.get(cc, f"SM{cc[0]}{cc[1]}")


def arch_tag() -> str:
    cc = torch.cuda.get_device_capability()
    return f"sm{cc[0]}{cc[1]}"


def time_ms(fn) -> float:
    """FlashInfer's timing call, as ``bench_nvfp4_quantize_backend_comparison.py`` issues it; the median in ms."""
    from flashinfer.testing.utils import bench_gpu_time

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r"(?s).*Falling back to CUDA events.*",
            category=UserWarning,
        )
        times = bench_gpu_time(
            fn=fn,
            enable_cupti=True,
            dry_run_iters=5,
            repeat_iters=30,
            cold_l2_cache=True,
            use_cuda_graph=False,
        )
    return float(np.median(times))


def problem_bytes(M: int, K: int, elem_bytes: int) -> int:
    """Input read + e4m3 codes written + one UE8M0 byte per 32-element block (the unpadded count)."""
    return M * K * elem_bytes + M * K + M * K // 32


def make_point(layout: str, dtype: torch.dtype, M: int, K: int, seed: int):
    """``(run_bi, run_cute, bi_config, check)`` for one grid point; identical inputs for both backends."""
    import flashinfer
    from flashinfer.batch_invariant.quant import mx_schedule

    torch.manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    sf_layout = LAYOUTS[layout]
    sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    cfg = mx_schedule.default_config(
        M,
        K,
        dtype.itemsize,
        sm,
        {"swizzled_128x4": "128x4", "swizzled_8x4": "8x4", "linear": "linear"}[layout],
    )

    def run_bi():
        flashinfer.mxfp8_quantize(
            x,
            sf_swizzle_layout=sf_layout,
            backend="auto",
            batch_invariant=True,
            enable_pdl=False,
        )

    def run_cute():
        flashinfer.mxfp8_quantize(
            x, sf_swizzle_layout=sf_layout, backend="cute-dsl", enable_pdl=False
        )

    def check() -> str:
        q_bi, s_bi = flashinfer.mxfp8_quantize(
            x,
            sf_swizzle_layout=sf_layout,
            backend="auto",
            batch_invariant=True,
            enable_pdl=False,
        )
        q_fi, s_fi = flashinfer.mxfp8_quantize(
            x, sf_swizzle_layout=sf_layout, backend="cute-dsl", enable_pdl=False
        )
        same_q = q_bi.shape == q_fi.shape and bool(
            torch.equal(q_bi.view(torch.uint8), q_fi.view(torch.uint8))
        )
        same_s = s_bi.shape == s_fi.shape and bool(torch.equal(s_bi, s_fi))
        return (
            "bitwise"
            if same_q and same_s
            else f"codes {'=' if same_q else '!='} scales {'=' if same_s else '!='}"
        )

    return run_bi, run_cute, cfg.describe(), check


def load_banked(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                done[(int(r["M"]), int(r["K"]))] = r
    return done


def geomean(xs):
    xs = [x for x in xs if x == x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def heatmap(ms, ks, grid, title, cbar, path, center=None, fmt_cell="{:.2f}"):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping heatmap generation")
        return
    fig, ax = plt.subplots(figsize=(13, 11))
    if center is not None:
        lo = min(0.5, float(np.nanmin(grid))) if np.isfinite(np.nanmin(grid)) else 0.5
        hi = max(2.0, float(np.nanmax(grid))) if np.isfinite(np.nanmax(grid)) else 2.0
        im = ax.imshow(
            grid,
            cmap="RdYlGn",
            norm=mcolors.TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi),
            aspect="auto",
        )
    else:
        im = ax.imshow(grid, cmap="viridis", aspect="auto")
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cb.ax.set_ylabel(cbar, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(ks)))
    ax.set_yticks(np.arange(len(ms)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_yticklabels([str(m) for m in ms])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    vmax = float(np.nanmax(grid)) if np.isfinite(np.nanmax(grid)) else 1.0
    for i in range(len(ms)):
        for j in range(len(ks)):
            v = grid[i, j]
            if not np.isnan(v):
                dark = (center is not None and (v < 0.7 or v > 1.5)) or (
                    center is None and v < 0.4 * vmax
                )
                ax.text(
                    j,
                    i,
                    fmt_cell.format(v),
                    ha="center",
                    va="center",
                    color="white" if dark else "black",
                    fontsize=7,
                )
    ax.set_xlabel("K (columns)")
    ax.set_ylabel("M (rows)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {path}")


def plot_and_summarize(path: str, layout: str, dtype_name: str, device: str) -> dict:
    done = load_banked(path)
    ms = [m for m in M_VALUES if any(k == m for k, _ in done)]
    ks = [k for k in K_VALUES if any(kk == k for _, kk in done)]
    if not ms or not ks:
        print("nothing banked yet")
        return {}
    sp = np.full((len(ms), len(ks)), np.nan)
    tb = np.full((len(ms), len(ks)), np.nan)
    for i, m in enumerate(ms):
        for j, k in enumerate(ks):
            r = done.get((m, k))
            if r is None:
                continue
            sp[i, j] = (
                float(r["speedup"]) if r["speedup"] not in ("", "nan") else float("nan")
            )
            tb[i, j] = (
                float(r["bi_tbps"]) if r["bi_tbps"] not in ("", "nan") else float("nan")
            )
    base = path[:-4]
    label = f"MXFP8 {layout} {dtype_name} on {device}"
    heatmap(
        ms,
        ks,
        sp,
        f"{label}: batch-invariant class vs FlashInfer CuTe-DSL backend\n(CuTe-DSL time / batch-invariant time; > 1 = batch-invariant faster; CUPTI kernel time, cold L2, median of 30)",
        "speedup",
        base + "_speedup.png",
        center=1.0,
    )
    heatmap(
        ms,
        ks,
        tb,
        f"{label}: achieved TB/s of the batch-invariant class (input + codes + scales)\nCUPTI kernel time, cold L2, median of 30",
        "TB/s",
        base + "_tbps.png",
    )
    summary = {
        "points": len(done),
        "device": device,
        "layout": layout,
        "dtype": dtype_name,
        "regimes": {},
    }
    for name, pred in REGIMES:
        rs = [r for (m, k), r in done.items() if pred(m)]
        sv = [float(r["speedup"]) for r in rs if r["speedup"] not in ("", "nan")]
        tbs = [float(r["bi_tbps"]) for r in rs if r["bi_tbps"] not in ("", "nan")]
        worst = sorted(
            (float(r["speedup"]), r) for r in rs if r["speedup"] not in ("", "nan")
        )[:3]
        summary["regimes"][name] = {
            "n": len(rs),
            "geomean": geomean(sv),
            "share_ge_0.95": (sum(x >= 0.95 for x in sv) / len(sv)) if sv else None,
            "min": min(sv) if sv else None,
            "peak_tbps": max(tbs) if tbs else None,
            "worst_cells": [
                f"M={r['M']} K={r['K']} {v:.2f} ({r['bi_config']})" for v, r in worst
            ],
        }
        reg = summary["regimes"][name]
        print(
            f"  {name}: n={len(rs)} geomean {reg['geomean']:.3f} share>=0.95 {reg['share_ge_0.95']} min {reg['min']} peak {reg['peak_tbps']} TB/s; worst {reg['worst_cells'][:1]}"
        )
    allv = [
        float(r["speedup"]) for r in done.values() if r["speedup"] not in ("", "nan")
    ]
    summary["geomean_all"] = geomean(allv)
    summary["share_ge_0.95_all"] = (
        (sum(x >= 0.95 for x in allv) / len(allv)) if allv else None
    )
    summary["mismatches"] = sorted(
        f"M={r['M']} K={r['K']}: {r['match']}"
        for r in done.values()
        if r["match"] != "bitwise"
    )
    with open(base + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"all {len(done)} points: geomean {summary['geomean_all']:.3f}, share >= 0.95 {summary['share_ge_0.95_all']}, mismatches {len(summary['mismatches'])}"
    )
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="MXFP8 quantization: the batch-invariant class vs FlashInfer's CuTe-DSL backend"
    )
    ap.add_argument("--layout", default="swizzled_128x4", choices=list(LAYOUTS))
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument(
        "--m-values", type=int, nargs="+", help="Row counts (default: the full grid)."
    )
    ap.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        help="Column counts (default: the full grid).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: benchmarks/results/batch_invariant/<arch>/).",
    )
    ap.add_argument(
        "--budget-s",
        type=float,
        default=float("inf"),
        help="Stop banking new points after this many seconds.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw the heatmaps and the summary from the banked CSV.",
    )
    args = ap.parse_args()
    dtype = DTYPES[args.dtype]
    cc = torch.cuda.get_device_capability()
    if cc not in DEVICE_NAMES:
        print(
            f"ERROR: the batch-invariant class needs SM100 / SM103 / SM107 (got SM{cc[0]}{cc[1]})"
        )
        return
    device = device_name()
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "batch_invariant",
        arch_tag(),
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"mxfp8_{args.layout}_{args.dtype}.csv")
    if args.plot_only:
        plot_and_summarize(path, args.layout, args.dtype, device)
        return
    ms = args.m_values or M_VALUES
    ks = args.k_values or K_VALUES
    done = load_banked(path)
    todo = [(m, k) for k in ks for m in ms if (m, k) not in done]
    print(
        f"MXFP8 {args.layout} {args.dtype} on {device} ({sm} SMs): {len(done)} points banked, {len(todo)} to do; CSV {path}",
        flush=True,
    )
    if todo:
        # the batch-invariant kernels the dispatcher picks over the grid, compiled device-free in parallel
        from flashinfer.batch_invariant.quant.mx_precompile import precompile_mx_quant
        from flashinfer.batch_invariant.quant.mx_schedule import default_config
        from flashinfer.batch_invariant.quant.mx_spec import MXFP8

        lay = {"swizzled_128x4": "128x4", "swizzled_8x4": "8x4", "linear": "linear"}[
            args.layout
        ]
        t0 = time.time()
        failures = precompile_mx_quant(
            [
                (
                    args.dtype,
                    k,
                    default_config(m, k, dtype.itemsize, sm, lay),
                    MXFP8,
                    lay,
                    False,
                )
                for m, k in todo
            ]
        )
        assert not failures, failures[:3]
        print(f"precompiled in {time.time() - t0:.0f} s", flush=True)
    new_file = not os.path.exists(path)
    t_start = time.time()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for m, k in todo:
            if time.time() - t_start > args.budget_s:
                print(
                    f"budget reached after {len(done)} points; resume with the same command",
                    flush=True,
                )
                break
            run_bi, run_cute, cfg, check = make_point(
                args.layout, dtype, m, k, args.seed
            )
            try:
                match = check()
            except Exception as e:  # noqa: BLE001 -- recorded in the CSV, the point is still timed
                match = f"check failed: {type(e).__name__}"
            t_bi = time_ms(run_bi)
            try:
                t_cute = time_ms(run_cute)
            except Exception as e:  # noqa: BLE001 -- a cute-dsl rejection is recorded, not substituted
                print(
                    f"  cute-dsl backend unavailable at M={m} K={k}: {type(e).__name__}: {str(e)[:100]}",
                    flush=True,
                )
                t_cute = float("nan")
            nbytes = problem_bytes(m, k, dtype.itemsize)
            rec = dict(
                M=m,
                K=k,
                bi_ms=t_bi,
                cute_ms=t_cute,
                bi_tbps=nbytes / (1e9 * t_bi),
                cute_tbps=nbytes / (1e9 * t_cute) if t_cute == t_cute else float("nan"),
                speedup=t_cute / t_bi if t_cute == t_cute else float("nan"),
                bi_config=cfg,
                match=match,
                bytes=nbytes,
                device=device,
                arch=arch_tag(),
                sm_count=sm,
                layout=args.layout,
                dtype=args.dtype,
            )
            w.writerow(rec)
            f.flush()
            done[(m, k)] = rec
            print(
                f"M={m:>6} K={k:>6}  batch-invariant {t_bi * 1e3:8.2f} us  cute-dsl {t_cute * 1e3:8.2f} us  "
                f"{rec['bi_tbps']:5.2f} TB/s  x{rec['speedup']:.2f}  {cfg}  [{match}]",
                flush=True,
            )
    plot_and_summarize(path, args.layout, args.dtype, device)


if __name__ == "__main__":
    main()
