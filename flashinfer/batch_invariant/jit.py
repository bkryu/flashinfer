"""Compiled kernels of the batch-invariant classes in FlashInfer's CuTe-DSL disk cache.

The glue maps the engine's cache contract onto :mod:`flashinfer.jit.cute_dsl_core`: one module directory per op
family under ``FLASHINFER_JIT_DIR`` (``<module>_<arch>_cute_dsl/``), one exported ``.o`` per kernel specialization and
one ``meta.json`` per module whose ``source_sha256`` is the hash of the **source closure** of the module's kernels: the
transitive closure of relative imports (module- and function-level) from the module's root files (:data:`MODULE_ROOTS`)
plus the ``__init__`` of every package touched. The closure of the MX quantizer is {``quant/mx_kernel.py``,
``quant/mx_sweep.py``, ``quant/blocks.py``, ``quant/mx_spec.py``, ``quant/mx_schedule.py``, ``quant/layout.py``,
``canon.py``, ``term.py``, ``fp32.py``, ``cute_fp.py``, ``cvt_fp8.py``, ``dtypes.py``, ``kernel_names.py``, this file and
the two ``__init__`` files}; the APIs, the CPU references and the precompiler are outside it, so editing them keeps the
compiled kernels valid (``tests/batch_invariant/test_mxfp8_quantize.py`` pins the closure).

Two rules of the engine's cache are kept on top of :class:`flashinfer.jit.cute_dsl_core.JitSpecCuteDsl`:

* **wipe only on definite evidence**: a module directory is deleted when its meta reads back as a *different* key or
  when the directory exists with no meta at all (a half-committed module); a meta that exists but cannot be read right
  now (another process replacing it, a transient NFS error) is retried once and is then NOT evidence -- a reader once
  wiped an 11,000-kernel module under a running GPU session because a precompile pool was committing into it;
* **compile outside the module lock**: the lock is taken only to commit the artifact (milliseconds), so the device-free
  precompile pool (:mod:`flashinfer.batch_invariant.quant.mx_precompile`) really compiles in parallel.

``FLASHINFER_CUTE_DSL_DISABLE_CACHE=1`` bypasses the cache (FlashInfer's switch).
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from ..jit.cute_dsl_core import (
    JitSpecCuteDsl,
    _read_meta,
    cute_dsl_cache_disabled,
    sanitize_symbol_name,
)
from ..jit.cute_dsl_core import _get_compile_arch as compile_arch
from ..jit import env as jit_env

logger = logging.getLogger("flashinfer.batch_invariant.jit")
_PKG_DIR = Path(__file__).resolve().parent

#: Root files of every cache module; the module's closure is the transitive relative-import closure of these files.
MODULE_ROOTS: Dict[str, Tuple[str, ...]] = {
    "batch_invariant_mx_quant": ("quant/mx_kernel.py", "quant/mx_sweep.py"),
}


def _relative_imports(path: Path) -> set:
    """Files named by the relative imports of ``path`` (module- and function-level)."""
    out: set = set()
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.level):
            continue
        base = path.parent
        for _ in range(node.level - 1):
            base = base.parent
        parts = node.module.split(".") if node.module else []
        target = base.joinpath(*parts) if parts else base
        cands = []
        if parts:
            cands += [target.with_suffix(".py"), target / "__init__.py"]
        cands += [target / f"{a.name}.py" for a in node.names]
        cands += [target / a.name / "__init__.py" for a in node.names]
        for c in cands:
            if c.exists():
                r = c.resolve()
                if r == _PKG_DIR or _PKG_DIR in r.parents:
                    out.add(r)
    return out


@functools.cache
def module_sources(module: str) -> Tuple[Path, ...]:
    """The source files a module's compiled objects depend on (see the module docstring)."""
    roots = MODULE_ROOTS.get(module)
    if roots is None:
        return tuple(sorted(_PKG_DIR.rglob("*.py")))
    seen: set = set()
    stack = [(_PKG_DIR / r).resolve() for r in roots]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack.extend(_relative_imports(p) - seen)
    for p in list(seen):
        d = p.parent
        while True:
            init = d / "__init__.py"
            if init.exists():
                seen.add(init.resolve())
            if d == _PKG_DIR or _PKG_DIR not in d.parents:
                break
            d = d.parent
    return tuple(sorted(seen))


def relative_sources(module: str) -> list:
    return [str(p.relative_to(_PKG_DIR)) for p in module_sources(module)]


@functools.cache
def source_sha256(module: str) -> str:
    sha = hashlib.sha256()
    for p in module_sources(module):
        sha.update(str(p.relative_to(_PKG_DIR)).encode())
        sha.update(p.read_bytes())
    return sha.hexdigest()


def module_is_stale(module_dir: Path, meta_path: Path, expected: dict) -> bool:
    """DEFINITE evidence that a module directory was built from another closure (see the module docstring)."""
    if not module_dir.exists():
        return False
    meta = _read_meta(meta_path)
    if meta is None:
        time.sleep(0.05)
        meta = _read_meta(meta_path)
        if meta is None and meta_path.exists():
            return False
    return meta != expected


class BatchInvariantJitSpec(JitSpecCuteDsl):
    """A :class:`JitSpecCuteDsl` keyed by the module's source closure, with the two rules of the module docstring."""

    def __init__(
        self, module_name: str, kernel_name: str, compile_fn: Callable[[], Any]
    ):
        super().__init__(
            module_name, kernel_name, compile_fn, source_sha256(module_name)
        )
        self.expected_meta["sources"] = relative_sources(module_name)

    def _wipe_if_stale(self) -> None:
        if module_is_stale(self.module_dir, self.meta_path, self.expected_meta):
            logger.info(
                f"Invalidating stale batch-invariant module {self.module_dir_name}"
            )
            shutil.rmtree(self.module_dir, ignore_errors=True)

    def build(self) -> None:
        """Compile and export under the module lock (``JitSpec.build_and_load`` holds it)."""
        self._wipe_if_stale()
        logger.info(f"Compiling batch-invariant kernel {self.name}")
        self._compiled_kernel = self.compile_fn()
        self._persist()

    def compile_and_persist(self) -> None:
        """Compile outside the module lock; commit the artifact under it."""
        from filelock import FileLock

        self._compiled_kernel = self.compile_fn()
        with FileLock(self.lock_path, thread_local=False):
            self._wipe_if_stale()
            self._persist()

    def _persist(self) -> None:
        try:
            if (
                self.object_path.exists()
            ):  # another process committed the same kernel meanwhile
                return
            self._export()
        except Exception as e:  # noqa: BLE001 -- persistence is best-effort, as in FlashInfer
            logger.warning(
                f"Failed to persist batch-invariant kernel {self.name} to {self.object_path}: {e}. "
                "The kernel will be recompiled next run."
            )

    def build_and_load(self) -> Any:
        cached = self.try_load()
        if cached is not None:
            return cached
        started = time.perf_counter()
        self.compile_and_persist()
        from ..jit.core import _record_jit_build

        _record_jit_build(self.toolchain, self.name, time.perf_counter() - started)
        return self.load()


def build_and_load(module: str, kernel: str, compile_fn: Callable[[], Any]) -> Any:
    """Return a TVM-FFI callable for ``kernel`` of ``module``, compiling only on a miss."""
    if cute_dsl_cache_disabled():
        return compile_fn()
    return BatchInvariantJitSpec(module, kernel, compile_fn).build_and_load()


def module_dir(module: str) -> Path:
    return jit_env.FLASHINFER_JIT_DIR / sanitize_symbol_name(
        f"{module}_{compile_arch()}_cute_dsl"
    )


def expected_meta(module: str) -> dict:
    from ..jit.cute_dsl_core import _get_cute_dsl_version

    return {
        "arch": compile_arch(),
        "cute_dsl_version": _get_cute_dsl_version(),
        "source_sha256": source_sha256(module),
        "sources": relative_sources(module),
    }


def is_cached(module: str, kernel: str) -> bool:
    """True iff ``kernel``'s object is on disk under a meta that matches the current key (no load)."""
    if cute_dsl_cache_disabled():
        return False
    d = module_dir(module)
    obj = d / f"{sanitize_symbol_name(kernel)}.o"
    return obj.exists() and _read_meta(d / "meta.json") == expected_meta(module)


def wipe_module(module: str) -> None:
    """Delete a module's cache directory (used when precompiled objects prove unusable)."""
    shutil.rmtree(module_dir(module), ignore_errors=True)


def read_meta(module: str) -> Optional[dict]:
    meta = _read_meta(module_dir(module) / "meta.json")
    return json.loads(json.dumps(meta)) if isinstance(meta, dict) else None
