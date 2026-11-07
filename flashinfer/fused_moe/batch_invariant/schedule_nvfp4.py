# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 MoE classes' grouped-GEMM dispatcher: the dense 4-bit GEMM class's ranking heuristic at the grouped launch's
shape, as the bf16 classes' (:mod:`.schedule`).

The grouped block-scaled GEMM is the dense NVFP4 class under a different scheduler (one segment of rows per expert, whole
128-row scale-factor atoms per segment), so its records are ranked by the dense dispatcher's model
(:func:`flashinfer.gemm.batch_invariant.schedule.rank_topk` at the instruction class ``in_fmt``: the part's roofline with
nominal constants, no measured tables) at the dense shape whose tiles and operand bytes are the grouped launch's: ``M`` =
the rows of one active expert rounded up to the scale-factor atom (the grouped kernel computes whole atoms per group,
whether or not their rows are real), ``N`` = the active experts' weight rows side by side (every group's tiles count, and the
weights are read once per group). The ranking is mapped onto the grouped block-scaled space (the dense records at swizzle 1:
no in-cluster split, no operand swap, no B-reuse tile) and filtered by the grouped body's acceptance. Numerics-neutral: every
record of the space is canonical (the same bits), so the dispatcher may depend on anything; the ranked list is what the
autotuner profiles under ``batch_invariant``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ...gemm.batch_invariant.generator import (
    CLUSTERS,
    TILES_M_BLOCKSCALED,
    TILES_N_BLOCKSCALED,
    Knobs,
    knob_product,
)
from ...gemm.batch_invariant.schedule import rank_topk
from .blocks.epilogue_nvfp4 import W4A4Chain
from .schedule import TOP_K, active_experts, rows_per_expert  # noqa: F401  (re-exported)

GROUP_ALIGN = (
    128  # the padded group alignment the dispatcher ranks at (one scale-factor atom)
)


def grouped_knobs_blockscaled(clusters=None) -> List[Knobs]:
    """The grouped block-scaled space: the dense records of the nvfp4 knob product at swizzle 1 (no in-cluster split, no
    operand swap, no B-reuse tile, no in-CTA chunk fold: the group scheduler has none of them), the same records for both
    instruction classes (the step is the class)."""
    return [
        k
        for k in knob_product(
            swizzles=(1,),
            clusters=CLUSTERS if clusters is None else clusters,
            in_fmt="nvfp4",
        )
        if k.cluster_k == 1 and not k.swap_ab and not k.b_reuse and not k.cta_chunks
    ]


def _grouped(cfg: Knobs) -> Knobs:
    """A dense block-scaled record as a grouped one: swizzle 1, no grid cap, the tile clamped into the grouped space (a
    swapped decode record's narrow token tile -> the narrowest grouped N tile; a B-reuse tile -> the 2-CTA tile; an
    in-cluster split or an in-CTA chunk fold -> the plain record of the same tile and cluster)."""
    tm = (
        cfg.tile_m
        if cfg.tile_m in TILES_M_BLOCKSCALED[cfg.use_2cta_instrs]
        else TILES_M_BLOCKSCALED[cfg.use_2cta_instrs][0]
    )
    tn = cfg.tile_n if cfg.tile_n in TILES_N_BLOCKSCALED else TILES_N_BLOCKSCALED[0]
    return Knobs(
        (tm, tn), cfg.cluster_shape_mn, cfg.use_2cta_instrs, True, cfg.raster_along, 1
    )


def all_valid_configs_nvfp4(
    N: int, K: int, E: int, chain: W4A4Chain, in_fmt: str = "nvfp4", clusters=None
) -> List[Knobs]:
    """The configuration space the grouped guarantee is claimed over, for an expert weight shape, a chain and an
    instruction class."""
    from .blocks.compile_nvfp4 import can_implement_grouped_nvfp4

    return [
        k
        for k in grouped_knobs_blockscaled(clusters)
        if can_implement_grouped_nvfp4(k, N, K, E, chain, in_fmt)
    ]


def padded_rows(rows: int) -> int:
    """The rows per active expert the ranking sees: whole scale-factor atoms."""
    return -(-max(1, rows) // GROUP_ALIGN) * GROUP_ALIGN


def rank_grouped_nvfp4(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: W4A4Chain,
    in_fmt: str,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    k: int = TOP_K,
    nominal: bool = False,
    n_active: Optional[int] = None,
) -> List[Knobs]:
    """The ranked grouped records for one GEMM: the dense ranking of the instruction class at ``(M = padded rows, N =
    n_active * N)`` mapped onto the grouped space, in rank order, de-duplicated, at least ``k`` records when the space has
    them (the ranking is extended by the rest of the space in its enumeration order, so the list is never empty for an
    implementable shape)."""
    from .blocks.compile_nvfp4 import can_implement_grouped_nvfp4

    out: List[Knobs] = []
    seen = set()
    ranked = rank_topk(
        padded_rows(rows),
        (n_active or E) * N,
        K,
        1,
        "k",
        sm_count,
        max(k, 10),
        family,
        in_fmt,
        nominal,
    )
    for cand in [_grouped(c) for c in ranked] + grouped_knobs_blockscaled():
        if cand in seen or not can_implement_grouped_nvfp4(
            cand, N, K, E, chain, in_fmt
        ):
            continue
        seen.add(cand)
        out.append(cand)
        if len(out) >= k:
            break
    return out


def default_configs_nvfp4(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    in_fmts: Tuple[str, str],
    chain1: W4A4Chain,
    chain2: W4A4Chain,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    nominal: bool = False,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I or I, K = H), config for GEMM2 (N = H, K = I))``: the first record of each ranked list."""
    rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
    n1 = 2 * I if chain1.pair else I
    c1 = rank_grouped_nvfp4(
        rows, n1, H, E, chain1, in_fmts[0], sm_count, family, 1, nominal, n_act
    )
    c2 = rank_grouped_nvfp4(
        rows, H, I, E, chain2, in_fmts[1], sm_count, family, 1, nominal, n_act
    )
    if not c1 or not c2:
        raise ValueError(
            f"no grouped block-scaled record implements (H={H}, I={I}, E={E}, {in_fmts})"
        )
    return c1[0], c2[0]
