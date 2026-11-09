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
no in-cluster split, no operand swap, no B-reuse tile), filtered by the grouped body's acceptance and by the tile-fit guard: a
record whose M tile is taller than the group's padded rows is not a candidate (the group scheduler's tiles never span groups,
so the rows past the group are padding the tile computes for nothing; the dense estimate, which prices such a tile as a dense
M tail, overrates it there). At few rows per active expert the swapped decode body (:mod:`.blocks.swap_nvfp4`: the weights on
the MMA's M side, the group's tokens a narrow N tile) leads the ranking by a structural rule -- its record is the narrowest
swapped token tile the rows per active expert fill to at most three quarters, the padded body otherwise
(:func:`token_tile_default`) -- so the GEMM traffic at decode is the weights read once instead of a 128-row token atom per
weight tile. Numerics-neutral: every record of either space is canonical (the same bits), so the dispatcher may depend on
anything; the ranked list is what the autotuner profiles under ``batch_invariant``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ...gemm.batch_invariant.generator import (
    CLUSTERS,
    TILES_M_BLOCKSCALED,
    TILES_N_BLOCKSCALED,
    TILES_N_NARROW,
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


def swapped_knobs_blockscaled() -> List[Knobs]:
    """The swapped decode space of the grouped block-scaled classes (:mod:`.blocks.swap_nvfp4`): the dense class's swapped 1-CTA
    records at the narrow token tiles (128 weight rows x 8 / 16 / 32 tokens), one per tile, no cluster, swizzle 1. Disjoint from
    the grouped space; the body reads the dense packing."""
    return [
        Knobs((128, tn), (1, 1), False, True, "m", 1, swap_ab=True)
        for tn in TILES_N_NARROW
    ]


#: The average rows per active expert fill at most three quarters of the swapped token tile (an empirical margin).
SWAP_TILE_FILL = (3, 4)
#: The tiles the rule may pick: the narrowest. The wider swapped tiles measured behind the padded body's dense records of this
#: dispatcher at the layer level (8-16 rows per active expert); they stay records of the space, selectable.
SWAP_TILES_DEFAULT = (8,)


def token_tile_default(rows: int) -> Optional[int]:
    """The swapped decode body's token tile for ``rows`` rows per active expert: the narrowest swapped tile of ``SWAP_TILES_DEFAULT``
    the average fills to at most ``SWAP_TILE_FILL`` (a quarter of the tile stays free for the load's unevenness: an expert above the
    average still fits one tile, so the weights are read once), ``None`` when none does (the padded body: a 128-row token atom holds
    the group's rows). A structural rule (the rows vs the tile), no measured table; a placement: both bodies compute the same bits."""
    num, den = SWAP_TILE_FILL
    for tn in SWAP_TILES_DEFAULT:
        if rows * den <= tn * num:
            return tn
    return None


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


def all_valid_configs_nvfp4_swapped(
    N: int, K: int, E: int, chain: W4A4Chain, in_fmt: str = "nvfp4"
) -> List[Knobs]:
    """The swapped decode records the swapped body accepts for an expert weight shape, a chain and an instruction class (the
    dense packing); the guarantee is claimed over this space and the grouped space together."""
    from .blocks.compile_nvfp4 import can_implement_swapped_nvfp4

    return [
        k
        for k in swapped_knobs_blockscaled()
        if can_implement_swapped_nvfp4(k, N, K, E, chain, in_fmt)
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
    swap: Optional[bool] = None,
) -> List[Knobs]:
    """The ranked grouped records for one GEMM: the swapped decode body's record first when the rows per active expert fit
    a narrow token tile with room to spare (:func:`token_tile_default`; ``swap`` True / False forces / forbids it -- the padded packing takes
    the grouped records only), then the dense ranking of the instruction class at ``(M = padded rows, N = n_active * N)``
    mapped onto the grouped space, in rank order, de-duplicated, the records whose M tile fits the padded rows of a group
    (a taller tile computes padding only), at least ``k`` records when the spaces have them (the ranking is extended by the
    rest of the grouped space in its enumeration order, so the list is never empty for an implementable shape: the 128-row
    tiles always fit)."""
    from .blocks.compile_nvfp4 import (
        can_implement_grouped_nvfp4,
        can_implement_swapped_nvfp4,
    )

    out: List[Knobs] = []
    seen = set()
    tn = None
    if swap is not False:
        tn = (
            token_tile_default(rows)
            if swap is None
            else (token_tile_default(rows) or TILES_N_NARROW[-1])
        )
    if tn is not None:
        sw = Knobs((128, tn), (1, 1), False, True, "m", 1, swap_ab=True)
        if can_implement_swapped_nvfp4(sw, N, K, E, chain, in_fmt):
            out.append(sw)
            seen.add(sw)
    m = padded_rows(rows)
    ranked = rank_topk(
        m,
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
        if (
            cand in seen
            or cand.tile_m > m
            or not can_implement_grouped_nvfp4(cand, N, K, E, chain, in_fmt)
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
    swap: Optional[bool] = None,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I or I, K = H), config for GEMM2 (N = H, K = I))``: the first record of each ranked list
    (``swap``: the swapped decode body by the structural rule (None), forced (True) or forbidden (False))."""
    rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
    n1 = 2 * I if chain1.pair else I
    c1 = rank_grouped_nvfp4(
        rows, n1, H, E, chain1, in_fmts[0], sm_count, family, 1, nominal, n_act, swap
    )
    c2 = rank_grouped_nvfp4(
        rows, H, I, E, chain2, in_fmts[1], sm_count, family, 1, nominal, n_act, swap
    )
    if not c1 or not c2:
        raise ValueError(
            f"no grouped block-scaled record implements (H={H}, I={I}, E={E}, {in_fmts})"
        )
    return c1[0], c2[0]
