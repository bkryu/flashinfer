# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# Ported from LightLM's cutlass_dsl megakernel study (docs/megakernel-plan.md there) for the FlashInfer experimental
# track: the Qwen3.8-27B one-launch decoder on SM120.

from __future__ import annotations

import torch

from ._runtime import from_dlpack

MAX_M = 16  # decode rows per launch (TILE_N 8 / 16 activation tiles)
TILE = (
    64,
    16,
    256,
)  # weight-tile rows, activation-tile columns, K per stage (bf16 slots x 2 for fp8 / x 4 for NVFP4)
_DUMMY_C: dict = {}
_WS: dict = {}
_STAGE: dict = {}
_QK: dict = {}


def _cute(t3, leading_dim):
    return from_dlpack(t3, assumed_align=16).mark_layout_dynamic(
        leading_dim=leading_dim
    )


def _fp8_cute(t3_u8, leading_dim):
    import cutlass

    t = from_dlpack(t3_u8, assumed_align=16)
    t.element_type = cutlass.Float8E4M3FN
    return t.mark_layout_dynamic(leading_dim=leading_dim)


def _dummy_c(M, N, dtype, device):
    """The GEMM's TMA-store C atom / grid derive from a (N, M) tensor; nothing is stored through it."""
    key = (M, N, dtype, str(device))
    t = _DUMMY_C.get(key)
    if t is None:
        t = _DUMMY_C[key] = torch.empty(M, N, dtype=dtype, device=device)
    return t


def _workspace(device, S, M, N):
    key = (str(device), S, M, N)
    t = _WS.get(key)
    if t is None:
        t = _WS[key] = torch.empty(S, M, N, dtype=torch.float32, device=device)
    return t


def _staging(device, n_out, n_lse):
    key = (str(device), n_out, n_lse)
    t = _STAGE.get(key)
    if t is None:
        t = _STAGE[key] = (
            torch.zeros(n_out, dtype=torch.float32, device=device),
            torch.full((n_lse,), float("-inf"), dtype=torch.float32, device=device),
        )
    return t


def _qk_scratch(device, n):
    key = (str(device), n)
    t = _QK.get(key)
    if t is None:
        t = _QK[key] = torch.empty(n, dtype=torch.float32, device=device)
    return t


def _kv_splits(M, hkv, max_seq_len, page, seq_lens):
    """KV split count per (sequence, kv head) for the attention items: power of two, >= 128 items when the pages allow."""
    msl = int(max_seq_len) if max_seq_len else int(seq_lens.max().item())
    pages = max(1, (msl + page - 1) // page)
    KS = 1
    while KS < 64 and M * hkv * KS < 128:
        KS *= 2
    while KS > 1 and pages < KS:
        KS //= 2
    return KS


def interleave_gate_up(w_packed: torch.Tensor, w_scale: torch.Tensor, inter: int):
    """[gate(I) | up(I)] rows -> [g0..7, u0..7, g8..15, u8..15, ...] (8-row groups) so one 64-row weight tile holds the
    gate and up rows the SwiGLU epilogue pairs. Row permutation only; returns (w_il, s_il) contiguous."""
    N = w_packed.shape[0]
    if 2 * inter != N or inter % 8:
        raise ValueError(
            f"gate_up weight must have 2*I rows with I % 8 == 0, got N={N}, I={inter}"
        )

    def il(t):
        g = t[:inter].reshape(inter // 8, 8, *t.shape[1:])
        u = t[inter:].reshape(inter // 8, 8, *t.shape[1:])
        return torch.stack([g, u], dim=1).reshape(N, *t.shape[1:]).contiguous()

    return il(w_packed), il(w_scale)


def rowstat(x: torch.Tensor) -> torch.Tensor:
    """fp32 [1, M] sum of squares per token (row 0 of the absorbed input norm's statistic)."""
    return x.float().pow(2).sum(-1, keepdim=True).t().contiguous()


def build_rope_cache(
    max_pos: int, rot_dim: int, theta: float = 10000.0, device="cuda"
) -> torch.Tensor:
    """[max_pos, rot_dim] fp32: first half cos, second half sin (rotate-half pairs (i, i + rot_dim/2))."""
    if rot_dim % 2:
        raise ValueError(f"rot_dim must be even, got {rot_dim}")
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float64) / rot_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float64)
    freqs = torch.outer(t, inv_freq)
    return (
        torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        .to(torch.float32)
        .to(device)
        .contiguous()
    )
