# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The block-scale fp8 grouped GEMMs' dispatcher: the dense block-scale class's ranking heuristic at the grouped launch's shape,
and the placement rule of the intermediate's requantization.

As for the bf16 classes (:mod:`.schedule`): a grouped GEMM is the dense class under a different scheduler, so its records are
ranked by the dense dispatcher's model at the dense shape whose tiles, M tail and operand bytes are the grouped launch's
(``M`` = the rows of one active expert, ``N`` = the active experts' weight rows side by side), for the block-scale format
(``in_fmt="e4m3_bs"``: the fp8 instruction rate, the promotion hand-off chain, no 2-CTA issue gain for the lockstep pair),
with the dense dispatcher's decode guard and its few-token N-cluster rule, mapped onto the grouped space and filtered by the
grouped body's acceptance. The requantization's two spellings (GEMM1's epilogue on the whole-block tiles, or the standalone
pass on the bf16 intermediate) are placements of one term: ``fused_quant_default`` picks by the rows per active expert.
Numerics-neutral: every record of the space computes the class.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ...gemm.batch_invariant.generator import CLUSTERS, Knobs, knob_product
from .blocks.fp8bs_compile import (
    GEMM2_CHAIN,
    TILES_N_GROUPED,
    Fp8BsChain,
    can_implement_grouped_fp8bs,
    can_implement_swapped_fp8bs,
    gemm1_chain,
    swap_knobs_fp8bs,
    swapped_knobs_fp8bs,
)
from .schedule import TOP_K, active_experts, pair_term, raster_for, rows_per_expert

IN_FMT = "e4m3_bs"
#: The fused requantization (GEMM1's output blocks quantized in its epilogue, the 256-wide gated tile) from this many rows per
#: active expert; below it the standalone pass and GEMM1's free tile choice (empirical; a placement).
FUSED_QUANT_MIN_ROWS = 128


def grouped_knobs_fp8bs(clusters=None) -> List[Knobs]:
    """The grouped block-scale space: the dense block-scale class's records (128-row CTA tiles, the six clusters, both rasters)
    at swizzle 1 with a tile of two promotion warpgroups (N 64 / 128 / 256), or the subset with the given clusters."""
    return [
        k
        for k in knob_product(
            swizzles=(1,),
            clusters=CLUSTERS if clusters is None else clusters,
            in_fmt=IN_FMT,
        )
        if k.tile_n in TILES_N_GROUPED
    ]


def all_valid_configs_fp8bs(
    N: int,
    K: int,
    E: int,
    chain: Fp8BsChain = GEMM2_CHAIN,
    clusters=None,
    interleaved: Optional[bool] = None,
) -> List[Knobs]:
    """The configuration space the grouped block-scale guarantee is claimed over, for a weight shape and a chain (the fused
    requantization's chain admits the tiles whose outputs are whole 128-column blocks)."""
    return [
        k
        for k in grouped_knobs_fp8bs(clusters)
        if can_implement_grouped_fp8bs(k, N, K, E, chain, interleaved)
    ]


def all_valid_configs_fp8bs_swapped(
    N: int, K: int, E: int, chain: Fp8BsChain = GEMM2_CHAIN
) -> List[Knobs]:
    """The swapped decode records the swapped body accepts for a weight shape and a chain (the bf16 / f32 chains)."""
    return [
        k
        for k in swapped_knobs_fp8bs()
        if can_implement_swapped_fp8bs(k, N, K, E, chain)
    ]


#: The swapped decode body's structural rule (declared empirical, no measured table): the token tile is the narrowest of
#: SWAP_TOKEN_TILES the average rows per active expert fill to at most SWAP_TILE_FILL (an expert above the average still fits one
#: tile: its weights stream once); the weight tile is 64 rows while the active experts' 64-row tiles fit the part in SWAP_WAVES_64
#: waves (twice the CTAs on the weight stream: the decode GEMM is bound by its per-CTA chain, not by the HBM), 128 rows beyond.
SWAP_TOKEN_TILES = (8, 16, 32)
SWAP_TILE_FILL = (3, 4)
SWAP_WAVES_64 = 1


def swap_record_fp8bs(
    rows: int,
    N: int,
    n_active: int,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
) -> Optional[Knobs]:
    """The swapped decode record for ``rows`` rows per active expert and ``n_active`` experts of ``N`` weight rows each, or ``None``
    when the rows do not fit a narrow token tile (the padded body). A placement: both bodies compute the same bits."""
    from ...gemm.batch_invariant.heuristic import PART_TABLE

    num, den = SWAP_TILE_FILL
    tn = next((t for t in SWAP_TOKEN_TILES if rows * den <= t * num), None)
    if tn is None:
        return None
    sms = sm_count or PART_TABLE[family or "sm_107"].sm_count
    ctas64 = n_active * ((N + 63) // 64)
    wt = 64 if ctas64 <= SWAP_WAVES_64 * sms else 128
    return swap_knobs_fp8bs(wt, tn)


def rank_grouped_fp8bs(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: Fp8BsChain,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    k: int = TOP_K,
    nominal: bool = False,
    n_active: Optional[int] = None,
) -> List[Knobs]:
    """The ranked grouped records for one block-scale GEMM, as the bf16 backend's ``schedule.rank_grouped``: the grouped space
    the body accepts for the weight shape and the chain, ordered by the dense block-scale class's estimate at ``(M = rows,
    N = n_active * N, K)`` times the pair term, each tile family priced once at the enumeration's raster and the grouped
    scheduler's structural raster first among its twins; at a decode-sized ``rows`` the dense dispatcher's structural guard
    selects the candidates and its few-token rule prefers the widest N-cluster that still fits the part in one wave; each of
    the best ceil(k/2) records is followed by its raster twin, then the rank order fills the list: distinct records, at most
    ``k``, never empty for an implementable shape."""
    import dataclasses

    from ...gemm.batch_invariant.heuristic import (
        PART_TABLE,
        TOKEN_BOX_ROWS,
        capacity_for,
        decode_sane,
        estimate_cycles,
    )

    fam = family or "sm_107"
    dense_n = (n_active or E) * N
    space = [
        c
        for c in grouped_knobs_fp8bs()
        if can_implement_grouped_fp8bs(c, N, K, E, chain)
    ]
    if not space:
        return []
    guarded = [c for c in space if decode_sane(c, rows, False, True)] or space
    if rows <= TOKEN_BOX_ROWS // 4:
        # the dense block-scale dispatcher's few-token rule: the 128-row token box is at least three quarters zero-fill, a TMA
        # cost per block per CTA that the N-cluster's multicast issues once per cluster; the widest N-cluster shares it most
        # while its clusters still fit the part in one wave (every active expert's tiles count)
        cap4 = capacity_for(fam, sm_count, nominal).get(4) or max(
            1, (sm_count or PART_TABLE[fam].sm_count) // 4
        )
        wide = [
            c
            for c in guarded
            if c.cluster_n == 4
            and ((dense_n + c.tile_n - 1) // c.tile_n + 3) // 4 <= cap4
        ]
        guarded = wide or guarded

    def key(i, c):
        canon = dataclasses.replace(c, raster_along="m")
        est = estimate_cycles(
            canon, rows, dense_n, K, 1, "k", IN_FMT, fam, sm_count, nominal=nominal
        )
        return (
            est * pair_term(c, rows),
            0 if c.raster_along == raster_for(c, rows) else 1,
            i,
        )

    ranked = [c for _k, c in sorted((key(i, c), c) for i, c in enumerate(guarded))]
    members = set(space)
    out: List[Knobs] = []
    seen = set()
    for rec in ranked[: -(-k // 2)]:
        twin = dataclasses.replace(
            rec, raster_along="n" if rec.raster_along == "m" else "m"
        )
        for cand in (rec, twin):
            if cand not in seen and (cand is rec or cand in members):
                out.append(cand)
                seen.add(cand)
    for rec in ranked:
        if len(out) >= k:
            break
        if rec not in seen:
            out.append(rec)
            seen.add(rec)
    return out[:k]


def fused_quant_default(T: int, k: int, E: int) -> bool:
    """Whether the block-scale op requantizes GEMM1's output in the epilogue (the wide-tile records) or by the standalone quantizer
    pass at this (T, k, E) by default: a placement rule on the rows per active expert, both spellings the same bits."""
    return rows_per_expert(T, k, E) >= FUSED_QUANT_MIN_ROWS


def default_configs_fp8bs(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    act: str = "silu",
    gated: bool = True,
    params: Sequence[Optional[float]] = (),
    fused: bool = False,
    nominal: bool = False,
    swap: Optional[bool] = None,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I, K = H; N = I ungated), config for GEMM2 (N = H, K = I))`` of the block-scale class: the swapped
    decode record where the rows per active expert fit a narrow token tile (`swap_record_fp8bs`; ``swap`` True / False forces /
    forbids it; the fused requantization is not a spelling of the swapped body, so ``fused`` keeps the padded GEMM1 records), else
    the first record of each ranked list; ``fused`` ranks GEMM1 over the fused requantization's records."""
    rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
    n1 = 2 * I if gated else I
    ch1 = gemm1_chain(act, params, gated, fused)
    out = []
    for N, K, chain in ((n1, H, ch1), (H, I, GEMM2_CHAIN)):
        sw = None
        if swap is not False and not (fused and chain is ch1):
            sw = swap_record_fp8bs(rows, N, n_act, sm_count, family)
            if sw is None and swap:
                sw = swap_record_fp8bs(1, N, n_act, sm_count, family)
        if sw is not None and can_implement_swapped_fp8bs(sw, N, K, E, chain):
            out.append(sw)
            continue
        ranked = rank_grouped_fp8bs(
            rows, N, K, E, chain, sm_count, family, 1, nominal, n_act
        )
        if not ranked:
            raise ValueError(
                f"no grouped block-scale record implements (H={H}, I={I}, E={E}, gated={gated}, fused={fused})"
            )
        out.append(ranked[0])
    return out[0], out[1]


__all__ = [
    "FUSED_QUANT_MIN_ROWS",
    "IN_FMT",
    "SWAP_TILE_FILL",
    "SWAP_TOKEN_TILES",
    "SWAP_WAVES_64",
    "all_valid_configs_fp8bs",
    "all_valid_configs_fp8bs_swapped",
    "default_configs_fp8bs",
    "fused_quant_default",
    "gemm1_chain",
    "grouped_knobs_fp8bs",
    "rank_grouped_fp8bs",
    "swap_record_fp8bs",
    "swapped_knobs_fp8bs",
]
