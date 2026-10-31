"""The contract's gates: the supported architectures, the ``backend`` rule of the batch-invariant mode and the
"unsupported raises" rule (``README.md``, "The flag"). Light imports only: the operators import it on every call with
``batch_invariant=True``."""

from __future__ import annotations

from typing import Tuple

import torch

#: The architectures the guarantee is stated for (tcgen05 parts): SM100 (B200), SM103 (B300), SM107 (Rubin).
SUPPORTED_ARCHS: Tuple[Tuple[int, int], ...] = ((10, 0), (10, 3), (10, 7))
ARCH_NAMES = {(10, 0): "SM100", (10, 3): "SM103", (10, 7): "SM107"}


class BatchInvariantUnsupportedError(ValueError):
    """``batch_invariant=True`` was asked for an input the batch-invariant class does not cover.

    The mode never falls back to a non-invariant kernel: an unsupported shape, dtype, layout, device or architecture
    raises this error, whose message names the class and the limit."""


def unsupported(
    class_name: str, what: str, limit: str
) -> BatchInvariantUnsupportedError:
    return BatchInvariantUnsupportedError(
        f"batch_invariant=True: the class `{class_name}` does not support {what}; the limit is {limit}. "
        "The batch-invariant mode never falls back to another kernel; call without batch_invariant=True to use "
        "the regular backends."
    )


def is_supported_device(device: torch.device) -> bool:
    """Whether ``device`` is a CUDA device of a supported architecture that the installed CuTe DSL can target."""
    if device.type != "cuda":
        return False
    from ..utils import get_compute_capability

    cc = get_compute_capability(device)
    if cc not in SUPPORTED_ARCHS:
        return False
    from ..cute_dsl.availability import (
        is_cute_dsl_arch_supported,
        is_cute_dsl_available,
    )

    return is_cute_dsl_available() and is_cute_dsl_arch_supported(*cc)


def check_device(device: torch.device, class_name: str) -> Tuple[int, int]:
    """The compute capability of ``device``, or :class:`BatchInvariantUnsupportedError` naming the limit."""
    archs = " / ".join(ARCH_NAMES[a] for a in SUPPORTED_ARCHS)
    if device.type != "cuda":
        raise unsupported(
            class_name, f"device {device}", f"a CUDA device of architecture {archs}"
        )
    from ..utils import get_compute_capability

    cc = get_compute_capability(device)
    if cc not in SUPPORTED_ARCHS:
        raise unsupported(
            class_name,
            f"compute capability {cc[0]}.{cc[1]}",
            f"the architectures {archs}",
        )
    from ..cute_dsl.availability import (
        is_cute_dsl_arch_supported,
        is_cute_dsl_available,
    )

    if not is_cute_dsl_available():
        raise unsupported(
            class_name,
            "an environment without the CuTe DSL",
            "the nvidia-cutlass-dsl package being installed (the kernels are CuTe-DSL kernels)",
        )
    if not is_cute_dsl_arch_supported(*cc):
        raise unsupported(
            class_name,
            f"the installed CuTe DSL on compute capability {cc[0]}.{cc[1]}",
            "a CuTe DSL release that can target this device",
        )
    return cc
