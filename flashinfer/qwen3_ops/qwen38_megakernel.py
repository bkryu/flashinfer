# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# Ported from LightLM's cutlass_dsl megakernel study (docs/megakernel-plan.md there) for the FlashInfer experimental
# track: the Qwen3.8-27B one-launch decoder on SM120.

"""Experimental: the Qwen3.8-27B decoder (48 gated-DeltaNet + 16 full-attention layers, or any prefix / the MTP drafter
layer) as ONE persistent CuTe DSL launch per decode step on SM120 (RTX PRO 6000 / GeForce Blackwell).

Thin entry points only (validation + deferred import): the kernel, its per-layer device tables and the scratch live in
`flashinfer.experimental.qwen38_megakernel`. Layouts in this first form are the study's (see the plan docstring):
paged bf16 K / V caches [pages, 32, hkv, 256], fp32 GDN state [slots, hv, 128, 128] and conv taps [slots, conv_dim, 3]
with one state slot per sequence, weights fp8 / NVFP4 [N, K] with per-shard alphas or bf16 [N, K] (wtype="bf16").
"""

from __future__ import annotations


import torch

from ..api_logging import flashinfer_experimental_api
from ..experimental.qwen38_megakernel.support import (  # lightweight: shape / CC logic only
    qwen38_megakernel_supported,
    qwen38_megakernel_unsupported_reason,
)

__all__ = [
    "Qwen38MegakernelPlan",
    "qwen38_megakernel_decode",
    "qwen38_megakernel_prepare",
    "qwen38_megakernel_supported",
    "qwen38_megakernel_unsupported_reason",
]


class Qwen38MegakernelPlan:
    """Everything one launch shape needs, built once and reused across CUDA-graph replays: the compile-time spec, the
    device tables for (layers, rows, kv splits), the activation scratch (slack-padded for the bf16 kind) and the
    first / last norm weights. Attributes are read by `qwen38_megakernel_decode`; treat them as opaque."""

    def __init__(
        self,
        spec,
        layers: list[dict],
        num_tokens: int,
        kv_splits: int,
        wn_in: torch.Tensor,
        wn_out: torch.Tensor,
        rope_table: torch.Tensor,
        device: torch.device,
    ):
        from ..experimental.qwen38_megakernel import DecoderTables, slack_buffer

        H, I = spec.H, spec.I
        dt = torch.bfloat16
        self.spec, self.layers, self.num_tokens, self.kv_splits = (
            spec,
            layers,
            int(num_tokens),
            int(kv_splits),
        )
        self.tables = DecoderTables(spec, layers, self.num_tokens, self.kv_splits)
        self.wn_in, self.wn_out, self.rope_table = wn_in, wn_out, rope_table
        M = self.num_tokens
        self.xw = [slack_buffer((M, H), dt, device) for _ in range(2)]
        self.sq = [
            torch.zeros(H // 64, M, dtype=torch.float32, device=device)
            for _ in range(2)
        ]
        self.act = slack_buffer((M, I), dt, device)
        self.resid = torch.empty(M, H, dtype=dt, device=device)
        self.out = torch.empty(M, H, dtype=dt, device=device)
        self.xcat = (
            slack_buffer((M, 2 * H), dt, device)
            if getattr(spec, "drafter_fold", False)
            else None
        )
        self.state_slots_pad = torch.zeros(16, dtype=torch.int32, device=device)


@flashinfer_experimental_api
def qwen38_megakernel_prepare(
    spec,
    layers: list[dict],
    num_tokens: int,
    *,
    kv_splits: int | None = None,
    max_seq_len: int | None = None,
    seq_lens: torch.Tensor | None = None,
    wn_in: torch.Tensor | None = None,
    wn_out: torch.Tensor | None = None,
    rope_table: torch.Tensor | None = None,
    device: torch.device | str | None = None,
    enable_pdl: bool | None = None,
) -> Qwen38MegakernelPlan:
    """Build the plan for one launch shape.

    ``spec`` is a ``flashinfer.experimental.qwen38_megakernel.DecoderSpec`` (geometry + compile-time knobs: spec_rows =
    rows per sequence, wtype = "fp8fp4" | "bf16", drafter_fold); ``layers`` the per-layer weight / cache / pool dicts
    (``LAYER_KEYS_*``) in decoder order; ``num_tokens`` the decode rows of the launch (1..16, a multiple of spec_rows).
    ``kv_splits`` defaults to the study's heuristic from the sequence count and ``max_seq_len``. ``wn_in`` is the first
    layer's input-norm weight (the kernel absorbs that norm; not needed with drafter_fold), ``wn_out`` the final norm
    weight (returned output = rms_norm(resid) * wn_out). Prepare eagerly, then capture: replays reuse the plan's buffers.
    """
    from ..experimental.qwen38_megakernel import (
        _runtime,
        build_rope_cache,
        kv_splits_for,
    )

    if not layers:
        raise ValueError("at least one layer")
    dev = torch.device(device) if device is not None else layers[0]["w_in"].device
    reason = qwen38_megakernel_unsupported_reason(
        dev,
        hidden=spec.H,
        inter=spec.I,
        hk=spec.hk,
        hv=spec.hv,
        dk=128,
        dv=128,
        hq=spec.hq,
        hkv=spec.hkv,
        head_dim=spec.d,
        rot_dim=spec.rot,
        page_size=spec.page,
        num_tokens=num_tokens,
        spec_rows=int(spec.spec_rows),
        splits=spec.S,
    )
    if reason is not None:
        raise ValueError(f"qwen38 megakernel: unsupported: {reason}")
    if enable_pdl is not None:
        _runtime.set_pdl(bool(enable_pdl))
    R = int(spec.spec_rows)
    if kv_splits is None:
        kv_splits = kv_splits_for(spec, num_tokens // R, max_seq_len, seq_lens)
    if rope_table is None:
        rope_table = build_rope_cache(
            max(int(max_seq_len or 4096) + 16 * R, 4096), spec.rot, device=dev
        )
    if wn_in is None and not getattr(spec, "drafter_fold", False):
        wn_in = torch.ones(spec.H, dtype=torch.bfloat16, device=dev)
    if wn_out is None:
        wn_out = layers[-1]["wn_out"]
    return Qwen38MegakernelPlan(
        spec, layers, num_tokens, kv_splits, wn_in, wn_out, rope_table, dev
    )


@flashinfer_experimental_api
def qwen38_megakernel_decode(
    plan: Qwen38MegakernelPlan,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    state_slots: torch.Tensor | None = None,
    *,
    embeds: torch.Tensor | None = None,
    final_norm: bool = True,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the decoder over ``hidden`` [num_tokens, H] bf16 (the residual stream entering layer 0; with drafter_fold the
    drafter's ``hidden`` input, and ``embeds`` [num_tokens, H] its embedding input). ``positions`` / ``slot_mapping``
    int32 [num_tokens]; ``seq_lens`` int32 [num_seqs] (context length including the new rows); ``block_table`` int32
    [num_seqs, max_pages]; ``state_slots`` int32 [num_seqs] GDN pool slots (None for attention-only layer lists).
    Returns rms_norm(final residual) * wn_out as bf16 [num_tokens, H] (or the raw residual with final_norm=False).
    Every tensor must keep its address across CUDA-graph replays (the plan's own buffers do)."""
    from ..experimental.qwen38_megakernel import decoder_entry, rowstat

    spec, M, H = plan.spec, plan.num_tokens, plan.spec.H
    if (
        hidden.shape != (M, H)
        or hidden.dtype != torch.bfloat16
        or not hidden.is_contiguous()
    ):
        raise ValueError(f"hidden must be contiguous bf16 [{M}, {H}]")
    nseq = M // int(spec.spec_rows)
    slots = state_slots if state_slots is not None else plan.state_slots_pad[:nseq]
    stream = torch.cuda.current_stream()
    if plan.xcat is not None:
        if embeds is None:
            raise ValueError("drafter_fold needs embeds")
        torch.cat([embeds, hidden], dim=-1, out=plan.xcat)
        decoder_entry(
            spec,
            plan.tables,
            plan.xw[0],
            plan.sq[0],
            plan.xw[1],
            plan.sq[1],
            plan.resid,
            plan.act,
            plan.rope_table,
            positions,
            slot_mapping,
            seq_lens,
            block_table,
            slots,
            xcat=plan.xcat,
            out=plan.out,
        )(stream)
        res = plan.out
        if out is not None:
            out.copy_(res)
            res = out
        return res
    plan.resid.copy_(hidden)
    torch.mul(plan.resid, plan.wn_in, out=plan.xw[0])
    plan.sq[0].zero_()
    plan.sq[0][0:1].copy_(rowstat(plan.resid))
    decoder_entry(
        spec,
        plan.tables,
        plan.xw[0],
        plan.sq[0],
        plan.xw[1],
        plan.sq[1],
        plan.resid,
        plan.act,
        plan.rope_table,
        positions,
        slot_mapping,
        seq_lens,
        block_table,
        slots,
    )(stream)
    if not final_norm:
        return plan.resid
    xf = plan.resid.float()
    y = (
        xf
        * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + spec.eps)
        * plan.wn_out.float()
    ).to(torch.bfloat16)
    if out is not None:
        out.copy_(y)
        return out
    return y
