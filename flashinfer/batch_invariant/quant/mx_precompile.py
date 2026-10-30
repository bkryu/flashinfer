"""Device-free bulk precompilation of the MX quantizer kernels into FlashInfer's CuTe-DSL cache.

    python -m flashinfer.batch_invariant.quant.mx_precompile                 # the dispatcher's kernels over the bench grid x SKUs
    python -m flashinfer.batch_invariant.quant.mx_precompile --space full --hidden 2048,4096 --dtypes bf16
    python -m flashinfer.batch_invariant.quant.mx_precompile --dry-run

Specs are ``(dtype_str, K, MXQuantConfig, MXFormat, layout, enable_pdl)``, de-duplicated by what is compiled (the Kbp
bucket). Workers are *spawned* (fresh interpreters, never forked from a process holding a CUDA context) with
``CUDA_VISIBLE_DEVICES=""`` and ``CUTE_DSL_ARCH=<arch of the current device>`` set before anything is imported, so the
DSL's target is exactly the device's and the cache key agrees with the GPU process. Each ``cute.compile`` takes 1-5 s;
the pool is capped at ``FLASHINFER_BI_POOL_WORKERS`` (default 16) workers and waits while the host's load average is
above ``FLASHINFER_BI_POOL_MAX_LOAD`` (default 100): this is NFS-bound work on a shared box.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List

KS = (
    32,
    64,
    96,
    256,
    1024,
    2048,
    4096,
    7168,
    8192,
    16384,
)  # the bench grid + odd sizes
ROWS = tuple(1 << i for i in range(15))
SM_COUNTS = (
    148,
    152,
    204,
    224,
)  # the SKUs the dispatcher is validated for: B200 (148), B300 (152), Rubin NVL8 (204), the full Rubin part (224)
DEFAULT_WORKERS = 16


def _arch_for_env() -> str:
    arch = os.environ.get("CUTE_DSL_ARCH")
    if arch:
        return arch
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


def _init_worker(arch: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUTE_DSL_ARCH"] = arch


def _wait_for_load(max_load: float, verbose: bool) -> None:
    waited = 0
    while True:
        try:
            load = os.getloadavg()[0]
        except OSError:
            return
        if load <= max_load or waited >= 3600:
            return
        if verbose and waited % 300 == 0:
            print(
                f"[batch_invariant.precompile] load average {load:.0f} > {max_load:.0f}: waiting",
                flush=True,
            )
        time.sleep(30)
        waited += 30


def run_pool(
    worker,
    items,
    workers: int | None = None,
    verbose: bool = True,
    label: str = "batch_invariant.precompile",
) -> List[str]:
    """Run ``worker(item)`` for every item in spawned device-free workers. Returns the failures as strings."""
    import multiprocessing
    import sys

    items = list(dict.fromkeys(items))
    if not items:
        return []
    cap = int(os.environ.get("FLASHINFER_BI_POOL_WORKERS", DEFAULT_WORKERS))
    workers = max(1, min(workers or cap, cap, (os.cpu_count() or 8) - 2, len(items)))
    _wait_for_load(float(os.environ.get("FLASHINFER_BI_POOL_MAX_LOAD", 100)), verbose)
    arch = _arch_for_env()
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    ctx = multiprocessing.get_context(
        "spawn" if main_file and os.path.exists(main_file) else "fork"
    )
    failures: List[str] = []
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(arch,), mp_context=ctx
    ) as ex:
        futs = {ex.submit(worker, s): s for s in items}
        for f in as_completed(futs):
            done += 1
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                failures.append(f"{futs[f]}: {e!r}")
            if verbose and (done % 50 == 0 or done == len(items)):
                print(
                    f"[{label}] {done}/{len(items)} ({len(failures)} failed)",
                    flush=True,
                )
    return failures


def _worker(spec) -> str:
    from flashinfer.batch_invariant.quant.mx_kernel import get_compiled_kernel
    from flashinfer.batch_invariant.quant.mx_spec import DEFAULT_RULE

    dtype_str, K, cfg, fmt, swizzled, pdl = spec[:6]
    get_compiled_kernel(dtype_str, K, cfg, fmt, DEFAULT_RULE, swizzled, pdl)
    return f"{fmt.name}/{dtype_str}/K{K}/{cfg.describe()}/{swizzled}"


def _dedupe(specs: Iterable) -> List:
    from .layout import layout_name

    out, seen = [], set()
    for s in specs:
        s = (s[0], s[1], s[2], s[3], layout_name(s[4]), s[5])
        key = (s[0], s[2].bucket(s[1]), s[2], s[3], s[4], s[5])
        if key not in seen:
            seen.add(key)
            out.append(tuple(s[:6]))
    return out


def dispatch_specs(
    fmts,
    dtypes=("bf16", "f16", "f32"),
    ks=None,
    layouts=("128x4", "8x4", "linear"),
    pdl=(False,),
    sm_counts=SM_COUNTS,
) -> List:
    """What the closed-form dispatcher reaches over the bench grid x every SKU's SM count."""
    from .mx_schedule import default_config

    ks = tuple(ks) if ks else KS
    eb = {"bf16": 2, "f16": 2, "f32": 4}
    out = []
    for fmt in fmts:
        for dt in dtypes:
            for K in ks:
                for sw in layouts:
                    for M in ROWS:
                        for sm in sm_counts:
                            for p in pdl:
                                out.append(
                                    (
                                        dt,
                                        K,
                                        default_config(M, K, eb[dt], sm, sw),
                                        fmt,
                                        sw,
                                        p,
                                    )
                                )
    return _dedupe(out)


def space_specs(
    fmts,
    dtypes=("bf16", "f16", "f32"),
    ks=None,
    layouts=("128x4", "8x4", "linear"),
    pdl=(False,),
) -> List:
    from .mx_schedule import all_valid_configs

    ks = tuple(ks) if ks else KS
    eb = {"bf16": 2, "f16": 2, "f32": 4}
    out = []
    for fmt in fmts:
        for dt in dtypes:
            for K in ks:
                for sw in layouts:
                    for cfg in all_valid_configs(K, eb[dt], sw):
                        for p in pdl:
                            out.append((dt, K, cfg, fmt, sw, p))
    return _dedupe(out)


def precompile_mx_quant(
    specs: Iterable, workers: int | None = None, verbose: bool = True
) -> List[str]:
    """Compile the specs that are not on disk, in spawned device-free workers. Returns the failures."""
    from .. import jit
    from ..kernel_names import MX_QUANT_MODULE, mx_quant_kernel_name
    from .mx_spec import DEFAULT_RULE

    specs = _dedupe(specs)
    if not specs or jit.cute_dsl_cache_disabled():
        return []
    todo = [
        s
        for s in specs
        if not jit.is_cached(
            MX_QUANT_MODULE,
            mx_quant_kernel_name(
                s[3].name,
                DEFAULT_RULE.name,
                s[0],
                s[2].bucket(s[1]),
                s[2].describe(),
                s[4],
                s[5],
            ),
        )
    ]
    if verbose and len(todo) < len(specs):
        print(
            f"[batch_invariant.precompile] {len(specs) - len(todo)} of {len(specs)} MX quant specializations already cached",
            flush=True,
        )
    return (
        run_pool(
            _worker, todo, workers, verbose, label="batch_invariant.precompile/mx_quant"
        )
        if todo
        else []
    )


def main(argv=None) -> int:
    import argparse

    from .mx_spec import FORMATS

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--formats", default="mxfp8", help="comma list of MX formats (mxfp8, mxfp4)"
    )
    ap.add_argument("--space", default="dispatch", choices=["dispatch", "full"])
    ap.add_argument("--dtypes", default="bf16,f16,f32")
    ap.add_argument(
        "--hidden",
        default="",
        help="row lengths K (default: the bench grid + odd sizes)",
    )
    ap.add_argument("--layouts", default="128x4,8x4,linear")
    ap.add_argument("--pdl", default="off", choices=["off", "both"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    fmts = [FORMATS[f] for f in args.formats.split(",")]
    ks = [int(k) for k in args.hidden.split(",")] if args.hidden else None
    pdl = (False, True) if args.pdl == "both" else (False,)
    fn = dispatch_specs if args.space == "dispatch" else space_specs
    specs = fn(
        fmts, tuple(args.dtypes.split(",")), ks, tuple(args.layouts.split(",")), pdl
    )
    print(
        f"[batch_invariant.precompile] space={args.space}: {len(specs)} specializations",
        flush=True,
    )
    if args.dry_run:
        return 0
    t0 = time.time()
    failures = precompile_mx_quant(specs, args.workers)
    print(
        f"[batch_invariant.precompile] {len(specs)} specializations in {time.time() - t0:.0f} s, {len(failures)} failed",
        flush=True,
    )
    for f in failures[:10]:
        print("  FAILED", f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
