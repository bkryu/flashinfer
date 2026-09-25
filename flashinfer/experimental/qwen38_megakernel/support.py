# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# Ported from LightLM's cutlass_dsl megakernel study (docs/megakernel-plan.md there) for the FlashInfer experimental
# track: the Qwen3.8-27B one-launch decoder on SM120.

from __future__ import annotations

import torch


def qwen38_megakernel_unsupported_reason(
    device: torch.device | str,
    *,
    hidden: int,
    inter: int,
    hk: int,
    hv: int,
    dk: int,
    dv: int,
    hq: int,
    hkv: int,
    head_dim: int,
    rot_dim: int,
    page_size: int,
    num_tokens: int,
    spec_rows: int = 1,
    splits: int = 2,
) -> str | None:
    """None when the one-launch decoder can serve this geometry on this device, else the reason (shape/dtype/CC logic only)."""
    from ...utils import get_compute_capability

    dev = torch.device(device)
    if dev.type != "cuda":
        return "cuda device required"
    major, minor = get_compute_capability(dev)
    if (major, minor) != (12, 0):
        return f"SM120 only (have sm_{major}{minor})"
    try:
        from ...cute_dsl.availability import is_cute_dsl_available

        if not is_cute_dsl_available():
            return "nvidia-cutlass-dsl is not importable"
    except Exception:  # pragma: no cover
        return "nvidia-cutlass-dsl is not importable"
    if dk != 128 or dv != 128:
        return "GDN head dims must be 128"
    if head_dim != 256 or page_size != 32:
        return "attention head_dim 256 and page size 32 only (v1)"
    if hq % hkv or hq // hkv > 8:
        return "GQA group must divide and be <= 8"
    if rot_dim <= 0 or rot_dim % 2 or rot_dim > head_dim:
        return "rotary dim must be even and <= head_dim"
    if hidden % 256 or inter % 64 or (inter // 64) % splits or hv % splits or hv % 2:
        return "hidden % 256, inter % 64, gate_up tiles / value heads divisible by the split count"
    if (hq * head_dim // 256) % 2:
        return "q rows must tile in pairs of 256"
    if num_tokens < 1 or num_tokens > 16:
        return "1..16 decode rows per launch"
    if spec_rows < 1 or num_tokens % spec_rows or spec_rows > 4:
        return "spec_rows must divide the rows and be <= 4"
    return None


def qwen38_megakernel_supported(device, **geometry) -> bool:
    return qwen38_megakernel_unsupported_reason(device, **geometry) is None
