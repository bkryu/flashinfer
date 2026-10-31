"""Batch-invariant mode: deterministic, batch-invariant kernels behind ``batch_invariant=True``.

The package holds the engine of the batch-invariant kernels (the canonical reduction plan, the floating-point interface
in its symbolic, exact-CPU and device backends, the term language and its fingerprints), the kernel classes behind the
FlashInfer operators that accept the flag, and the Lean project that proves the engine's reduction orders. ``README.md``
next to this file is the contract: the guarantee (R1-R4), the flag semantics, the tests and how to check the proofs.
"""

from __future__ import annotations

from typing import Callable, Dict, NamedTuple

from .support import (
    ARCH_NAMES,
    SUPPORTED_ARCHS,
    BatchInvariantUnsupportedError,
    check_device,
    is_supported_device,
)


class ClassInfo(NamedTuple):
    """A consistency class behind a FlashInfer API: its name, the fixed dimensions its fingerprint takes, and the
    function computing the fingerprint from them."""

    name: str
    dims: tuple
    fingerprint: Callable[..., str]


def _mxfp8_quant_fingerprint() -> str:
    from .quantization.mx_spec import FINGERPRINT_MXFP8

    return FINGERPRINT_MXFP8


#: The FlashInfer APIs with a batch-invariant class, by API name.
CLASSES: Dict[str, ClassInfo] = {
    "mxfp8_quantize": ClassInfo("mxfp8.quant", (), _mxfp8_quant_fingerprint),
}


def fingerprint(op: str, **dims) -> str:
    """The receipt of a batch-invariant output: the fingerprint of the consistency class the FlashInfer API ``op``
    computes under ``batch_invariant=True`` for the fixed dimensions ``dims``.

    The guarantee is scoped to ``(fingerprint, FlashInfer version, architecture)``: two outputs with the same
    fingerprint, the same inputs and the same environment are the same bits, whatever the batch, the configuration
    the dispatcher picked or the API path. A fingerprint change is a release-note item.

    ``mxfp8_quantize`` takes no dimensions (a block's bits depend on its 32 inputs alone: K, the input dtype and the
    scale-factor layout do not enter the class).

    >>> fingerprint("mxfp8_quantize")
    'mxquant-mxfp8-9018bd1627b3e72a'
    """
    info = CLASSES.get(op)
    if info is None:
        raise ValueError(
            f"no batch-invariant class behind {op!r}; the APIs with one are {sorted(CLASSES)}"
        )
    unknown = sorted(set(dims) - set(info.dims))
    if unknown:
        raise TypeError(
            f"fingerprint({op!r}) takes the fixed dimensions {list(info.dims)}, not {unknown}"
        )
    missing = [d for d in info.dims if d not in dims]
    if missing:
        raise TypeError(f"fingerprint({op!r}) needs the fixed dimensions {missing}")
    return info.fingerprint(**dims)


def class_name(op: str) -> str:
    """The consistency class name behind the API ``op`` (e.g. ``"mxfp8.quant"``)."""
    info = CLASSES.get(op)
    if info is None:
        raise ValueError(
            f"no batch-invariant class behind {op!r}; the APIs with one are {sorted(CLASSES)}"
        )
    return info.name


__all__ = [
    "ARCH_NAMES",
    "CLASSES",
    "ClassInfo",
    "SUPPORTED_ARCHS",
    "BatchInvariantUnsupportedError",
    "check_device",
    "class_name",
    "fingerprint",
    "is_supported_device",
]
