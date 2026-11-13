# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant sampling classes: the row log-sum-exp class ``lse.row`` behind
``flashinfer.sampling.softmax(..., batch_invariant=True)`` and ``flashinfer.sampling.log_probs`` (``api``), and the
inverse-CDF draw ``sample.icdf.v1`` behind ``sampling_from_probs(..., batch_invariant=True)`` /
``sampling_from_logits(..., batch_invariant=True)`` (``draw_api``) on the request-keyed stream (``stream``), and the
support class ``support.row.v1`` (``support_api``: the kept set of a row under top-k / top-p / min-p filters; its
renormalisation entries are internal in this release) with the truncated draw over it behind
``top_k_sampling_from_probs`` / ``top_p_sampling_from_probs`` / ``min_p_sampling_from_probs`` /
``top_k_top_p_sampling_from_probs`` / ``top_k_top_p_sampling_from_logits(..., batch_invariant=True)``
(``draw_support_api``). Light imports
only: the CuTe-DSL kernel modules are imported on first use.

The kernels' compiled-object cache modules are registered here (the kernel modules are their roots: an edit to a kernel
module, or to anything it imports within the mode's directories, invalidates its objects and nothing else's)."""

from .. import jit as _jit

_jit.MODULE_ROOTS.setdefault(
    "batch_invariant_sampling_lse", ("batch_invariant/sampling/kernel.py",)
)
_jit.MODULE_ROOTS.setdefault(
    "batch_invariant_sampling_mufu", ("batch_invariant/sampling/mufu_device.py",)
)
# the draw kernel of sample.icdf.v1 and the stream's conformance kernel: their own module, so an edit to them never
# recompiles the row kernels (they import the row kernel's loader / peers plumbing, so a row-kernel edit recompiles both)
_jit.MODULE_ROOTS.setdefault(
    "batch_invariant_sampling_draw",
    (
        "batch_invariant/sampling/draw_kernel.py",
        "batch_invariant/sampling/stream_kernel.py",
    ),
)
# the support kernel of support.row.v1 (the kept set under top-k / top-p / min-p): its own module, so an edit to it never
# recompiles the row or the draw kernels (it imports the row kernel's plumbing and the draw kernel's loader, so an edit
# to either recompiles it too)
_jit.MODULE_ROOTS.setdefault(
    "batch_invariant_sampling_support",
    ("batch_invariant/sampling/support_kernel.py",),
)
# the truncated draw (the support's thresholds, then the draw over the kept weights): its own module; it imports both
# kernels' plumbing, so an edit to either of them recompiles it too, and an edit here recompiles nothing else
_jit.MODULE_ROOTS.setdefault(
    "batch_invariant_sampling_draw_support",
    ("batch_invariant/sampling/draw_support_kernel.py",),
)
