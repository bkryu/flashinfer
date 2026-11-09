# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Vectorised CPU references of the NVFP4 MoE classes' chains (:mod:`.spec_nvfp4` through :class:`.reference.TorchF32`),
validated against the exact scalar model (:class:`.reference.ExactF32`) by the device-free tests. The GEMM accumulators
come from the device (the opaque block-scaled steps); the per-token quantizations are the quantizer class's own reference
(:func:`flashinfer.batch_invariant.quantization.nvfp4_reference.quantize_reference`)."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from . import spec_nvfp4 as Q
from .reference import TorchF32, _f32


def _row_scale(alpha_rows: torch.Tensor, pts_rows: torch.Tensor) -> torch.Tensor:
    return Q.w4a4_row_scale(
        TorchF32, _f32(alpha_rows).view(-1, 1), _f32(pts_rows).view(-1, 1)
    )


def w4a4_gemm1_cpu(
    g: torch.Tensor,
    u: torch.Tensor,
    alpha_rows: torch.Tensor,
    pts_rows: torch.Tensor,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
) -> torch.Tensor:
    """The gated W4A4 GEMM1 chain for whole rows: ``s = mul(alpha_e(row), pts(row))``, ``cvt_bf16(glu_pre(mul(g, s), mul(u,
    s)))``; ``alpha_rows`` / ``pts_rows`` are the per-row scalars (``[rows]`` f32), ``g`` / ``u`` the ``[rows, I]`` fp32
    accumulators."""
    return Q.glu_scaled_elem(
        TorchF32,
        _f32(g),
        _f32(u),
        _row_scale(alpha_rows, pts_rows),
        act,
        "bf16",
        params,
    )


def w4a4_act_cpu(
    acc: torch.Tensor,
    alpha_rows: torch.Tensor,
    pts_rows: torch.Tensor,
    act: str = "relu2",
    params: Sequence[Optional[float]] = (),
) -> torch.Tensor:
    """The ungated W4A4 GEMM1 chain for whole rows: ``cvt_bf16(act(mul(acc, s)))``."""
    return Q.act_scaled_elem(
        TorchF32, _f32(acc), _row_scale(alpha_rows, pts_rows), act, "bf16", params
    )


def w4a4_gemm2_cpu(
    acc: torch.Tensor, alpha_rows: torch.Tensor, pts_rows: torch.Tensor
) -> torch.Tensor:
    """The W4A4 GEMM2 chain for whole rows: ``cvt_bf16(mul(acc, mul(alpha_e(row), pts(row))))``."""
    return Q.scaled_out_elem(
        TorchF32, _f32(acc), _row_scale(alpha_rows, pts_rows), "bf16"
    )
