# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# Ported from LightLM's cutlass_dsl megakernel study (docs/megakernel-plan.md there) for the FlashInfer experimental
# track: the Qwen3.8-27B one-launch decoder on SM120.

from __future__ import annotations

import os
from typing import Any, Callable

import torch

_COMPILED: dict[tuple, Any] = {}
_HW: dict[tuple, int] = {}
# TVM-FFI launch path (cute.compile[EnableTVMFFI] + from_dlpack(enable_tvm_ffi=True)): ~20 us of host per launch vs
# ~45 us through ctypes marshalling; FLASHINFER_QWEN38_TVM_FFI=0 reverts.
TVM_FFI: bool = os.environ.get("FLASHINFER_QWEN38_TVM_FFI", "1") != "0"
# Programmatic dependent launch for the decoder launches (griddepcontrol wait/launch_dependents inside the kernel).
_PDL_ENABLED: bool = os.environ.get("FLASHINFER_QWEN38_PDL", "1") != "0"


def _cutlass():
    import cutlass  # noqa: F401  (import error = the CuTe DSL is absent)
    import cutlass.cute as cute

    return cutlass, cute


def stream_handle(stream: torch.cuda.Stream | None = None):
    """torch stream -> driver CUstream for `cute.compile` / launch."""
    import cuda.bindings.driver as cuda

    s = stream if stream is not None else torch.cuda.current_stream()
    return cuda.CUstream(s.cuda_stream)


def from_dlpack(t: torch.Tensor, assumed_align: int | None = 16, **kw):
    from cutlass.cute.runtime import from_dlpack as _fd

    kw.setdefault("enable_tvm_ffi", TVM_FFI)
    return _fd(t, assumed_align=assumed_align, **kw)


def set_pdl(enabled: bool) -> None:
    global _PDL_ENABLED
    _PDL_ENABLED = bool(enabled)


def launch_mode() -> int:
    """1 = launch with the programmatic-dependent-launch attribute (the kernel waits at entry), 0 = off."""
    if not _PDL_ENABLED:
        return 0
    try:
        from ...utils import device_support_pdl

        return (
            1
            if device_support_pdl(torch.device("cuda", torch.cuda.current_device()))
            else 0
        )
    except Exception:
        return 0


def _max_active_clusters() -> int:
    """Persistent grid size: one CTA per SM (cutlass HardwareInfo), cached per device."""
    key = ("mac", torch.cuda.current_device())
    v = _HW.get(key)
    if v is None:
        import cutlass

        v = _HW[key] = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    return v


def compile_cached(
    fn: Callable, *args, stream=None, options: tuple = (), key: tuple | None = None
):
    """Memoized `cute.compile[options](fn, *args, stream)`. Callers pass `key=` (the kernel's shape-ish knobs): the
    CuTe tensors carry dynamic layouts, so their identity changes per call while the compiled kernel does not."""
    _, cute = _cutlass()
    if TVM_FFI:
        options = tuple(options) + (cute.EnableTVMFFI(True),)
    if key is None:
        key = (id(fn), tuple(repr(o) for o in options))
    c = _COMPILED.get(key)
    if c is None:
        s = stream_handle(stream)
        compiler = cute.compile[tuple(options)] if options else cute.compile
        c = _COMPILED[key] = compiler(fn, *args, s)
    return c


def reset_for_testing() -> None:
    _COMPILED.clear()
