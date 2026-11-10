# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Programmatic dependent launch: where a kernel lets its dependents start. Every kernel waits for its predecessor before
its first dependent read (``griddepcontrol.wait``); the trigger (``griddepcontrol.launch_dependents``) decides when the next
kernel's CTAs may become resident. The one-wave kernels of this backend (the routing, the plan, the gather, the combine)
trigger right after their own wait (``TRIGGER = "small"``): the grouped GEMM's CTAs become resident and run their prologue
(barrier init, TMEM allocation, descriptor prefetch) while the one-wave kernel runs; the persistent GEMM bodies never trigger
(an early trigger lets the dependents' CTAs occupy SM slots during the main work; a trigger after the body's last loads still
overlaps its epilogue, which at decode is most of the kernel). The chain's rule holds under it: a kernel that triggered has
waited, so its own predecessor is complete when its dependents launch, and a dependent may read its predecessor's
predecessor before its own wait (the grouped bodies' group tables; the gather form waits first, its predecessor being the
plan). The other trigger sites mark where the other placements would go (``late``: after the CTA's last tile's loads;
``early``: right after the wait, in the persistent bodies too). The placement is baked into every compiled object and the
kernel names do not carry it, so it is a constant of the module, not a runtime or environment setting."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

#: The trigger placement of every kernel of the backend: ``none`` (the dependents launch at completion).
TRIGGER = "small"


@cute.jit
def trigger_early():
    """A persistent body's trigger right after its wait (the placement under comparison)."""
    if cutlass.const_expr(TRIGGER == "early"):
        cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def trigger_late():
    """A persistent body's trigger once the CTA's last loads are issued (the producer warp, after its tile loop)."""
    if cutlass.const_expr(TRIGGER == "late"):
        cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def trigger_after_wait():
    """A one-wave kernel's trigger right after its wait (the ``small`` placement)."""
    if cutlass.const_expr(TRIGGER == "small"):
        cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def trigger_one_wave():
    """A one-wave kernel's trigger: only the early placement triggers at all (the dependents launch at completion otherwise)."""
    if cutlass.const_expr(TRIGGER == "early"):
        cute.arch.griddepcontrol_launch_dependents()
