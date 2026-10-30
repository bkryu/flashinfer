"""Name <-> CuTe dtype table shared by the kernels (kept out of the APIs so that compiled-kernel
cache keys do not depend on API modules)."""

from __future__ import annotations


def get_cutlass_dtype(name: str):
    import cutlass

    return {
        "bf16": cutlass.BFloat16,
        "f16": cutlass.Float16,
        "f32": cutlass.Float32,
        "e4m3": cutlass.Float8E4M3FN,
        "e5m2": cutlass.Float8E5M2,
    }[name]
