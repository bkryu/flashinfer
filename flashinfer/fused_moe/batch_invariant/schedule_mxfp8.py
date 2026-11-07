# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 grouped GEMMs' dispatcher: the dense MX class's ranking heuristic at the grouped launch's shape.

As for the bf16 classes (:mod:`.schedule`), a grouped GEMM is the dense class under a different scheduler, so its records
are ranked by the dense dispatcher's model (`flashinfer.gemm.batch_invariant.schedule.rank_topk` for the MX class of the
part: the part's roofline with nominal constants, no measured tables) at the dense shape whose tiles, M tail and operand
bytes are the grouped launch's (``M`` = the rows of one active expert, ``N`` = the active experts' weight rows side by
side), filtered by the grouped block-scaled body's acceptance and the grouped MX space (the dense block-scaled records at
swizzle 1 without in-cluster split, operand swap or B-reuse tile). Numerics-neutral: every record of the space computes
the class.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ...gemm.batch_invariant.generator import (
    CLUSTERS,
    TILES_N_BLOCKSCALED,
    Knobs,
    knob_product,
)
from ...gemm.batch_invariant.schedule import rank_topk
from .blocks.epilogue import GroupedChain
from .schedule import TOP_K, active_experts, rows_per_expert

GEMM2_CHAIN = GroupedChain("plain", "silu", "bf16")


def grouped_knobs_mx(in_fmt: str = "mxfp8", clusters=None) -> List[Knobs]:
    """The grouped MX space before the body's shape acceptance: every well-formed dense record of the class at swizzle 1
    with a block-scaled tile N, no in-cluster split, no operand swap, no B-reuse tile, no in-CTA chunk fold (the group
    scheduler has none of them). One record list for both instruction classes."""
    return [
        k
        for k in knob_product(
            swizzles=(1,),
            clusters=CLUSTERS if clusters is None else clusters,
            in_fmt=in_fmt,
        )
        if k.cluster_k == 1
        and not k.swap_ab
        and not k.b_reuse
        and not k.cta_chunks
        and k.tile_n in TILES_N_BLOCKSCALED
    ]


def _grouped_mx(cfg: Knobs) -> Knobs:
    """A dense MX record as a grouped one: the instruction's M (the B-reuse tile halved), the N tile clamped into the
    block-scaled tiles (a swapped decode record's token tile has no grouped layout), swizzle 1, no grid cap, no split."""
    return Knobs(
        (cfg.inst_m, max(cfg.tile_n, TILES_N_BLOCKSCALED[0])),
        cfg.cluster_shape_mn,
        cfg.use_2cta_instrs,
        True,
        cfg.raster_along,
        1,
    )


def all_valid_configs_mx(
    N: int, K: int, E: int, chain: GroupedChain, in_fmt: str = "mxfp8", clusters=None
) -> List[Knobs]:
    """The configuration space the grouped MXFP8 guarantee is claimed over, for a weight shape, a chain and a class."""
    from .blocks.compile_mxfp8 import can_implement_grouped_mxfp8

    return [
        k
        for k in grouped_knobs_mx(in_fmt, clusters)
        if can_implement_grouped_mxfp8(k, N, K, E, chain, in_fmt)
    ]


def rank_grouped_mx(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: GroupedChain,
    in_fmt: str = "mxfp8",
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    k: int = TOP_K,
    nominal: bool = False,
    n_active: Optional[int] = None,
) -> List[Knobs]:
    """The ranked grouped records for one GEMM: the dense MX class's ranking at ``(M = rows, N = n_active * N)`` mapped
    onto the grouped space, in rank order, de-duplicated, at least ``k`` records when the space has them (the ranking is
    extended by the rest of the space in its enumeration order, so the list is never empty for an implementable shape)."""
    from .blocks.compile_mxfp8 import can_implement_grouped_mxfp8

    out: List[Knobs] = []
    seen = set()
    ranked = rank_topk(
        rows,
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
    for cand in [_grouped_mx(c) for c in ranked] + grouped_knobs_mx(in_fmt):
        if cand in seen or not can_implement_grouped_mxfp8(
            cand, N, K, E, chain, in_fmt
        ):
            continue
        seen.add(cand)
        out.append(cand)
        if len(out) >= k:
            break
    return out


def default_configs_mx(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    chain1: GroupedChain,
    in_fmt: str = "mxfp8",
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    nominal: bool = False,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I or I, K = H), config for GEMM2 (N = H, K = I))``: the first record of each ranked list."""
    rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
    n1 = 2 * I if chain1.pair else I
    c1 = rank_grouped_mx(
        rows, n1, H, E, chain1, in_fmt, sm_count, family, 1, nominal, n_act
    )
    c2 = rank_grouped_mx(
        rows, H, I, E, GEMM2_CHAIN, in_fmt, sm_count, family, 1, nominal, n_act
    )
    if not c1 or not c2:
        raise ValueError(
            f"no grouped MX record implements (H={H}, I={I}, E={E}, chain={chain1.describe()}, class={in_fmt})"
        )
    return c1[0], c2[0]
