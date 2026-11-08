# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 grouped GEMMs' dispatcher: the dense MX class's estimate over the grouped space at the grouped launch's shape.

As for the bf16 classes (:mod:`.schedule`), a grouped GEMM is the dense class under a different scheduler, so its records
are ranked by the dense dispatcher's model (`flashinfer.gemm.batch_invariant.heuristic.estimate_cycles` for the MX class
of the part: the part's roofline with nominal constants, no measured tables) at the dense shape whose tiles, M tail and
operand bytes are the grouped launch's (``M`` = the rows of one active expert, ``N`` = the active experts' weight rows
side by side). The records ranked are the grouped MX space itself (the dense block-scaled records at swizzle 1 without
in-cluster split, operand swap or B-reuse tile, filtered by the grouped block-scaled body's acceptance), not the dense
ranking's list mapped record by record: the dense MX class's own candidates at a decode-sized ``M`` are its swapped
decode records (the weights on the MMA's M side), a body the grouped kernel does not have, and their cluster and tile
knobs read in the grouped orientation name records the estimate itself ranks last. Two structural terms of the grouped
launch the dense shape cannot carry: the raster (`raster_for`: along N once an expert fills a 128-row block, along M
below) and the expected dummy half of the clusters that pair two 128-row tiles along M (`pair_term`: the experts' row
counts phased over the pair's extent). Numerics-neutral: every record of the space computes the class.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ...gemm.batch_invariant.generator import (
    CLUSTERS,
    TILES_N_BLOCKSCALED,
    Knobs,
    knob_product,
)
from ...gemm.batch_invariant.heuristic import decode_sane, estimate_cycles
from .blocks.epilogue import GroupedChain
from .schedule import TOP_K, active_experts, rows_per_expert

GEMM2_CHAIN = GroupedChain("plain", "silu", "bf16")
RASTER_N_ROWS = 128  # an expert spanning a whole 128-row block or more: the grouped scheduler rasters along N


def raster_for(rows: int) -> str:
    """The grouped scheduler's raster as a structural rule, not an estimate term: along N once an expert's rows fill a
    128-row block (consecutive CTAs then share the expert's A block while its weight tiles stream), along M below it
    (the weight tile is the larger operand of a few rows)."""
    return "n" if rows >= RASTER_N_ROWS else "m"


def pair_term(k: Knobs, rows: int) -> float:
    """The expected dummy half of a cluster that pairs two 128-row tiles along M (``cluster_shape_mn[0] == 2``): every
    expert's rows are padded to the record's M extent (128 x cm), and with the experts' row counts phased uniformly over
    that extent the expected padding is half an extent, so the pair's records carry (rows + 128) / (rows + 64) of the
    single-tile records' rows; the dense estimate at the average rows sees only the average's own tail."""
    cm = k.cluster_shape_mn[0]
    return (rows + 64.0 * cm) / (rows + 64.0)


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
    """The ranked grouped records for one GEMM: the grouped MX space the body accepts for the weight shape and the chain,
    ordered by the dense MX class's estimate at ``(M = rows, N = n_active * N, K)`` (``n_active`` the active experts, ``E``
    by default) times the pair term (the expected dummy half of the two-tile clusters along M), each tile family priced
    once (at the enumeration's raster) and the raster the structural rule gives the grouped scheduler (`raster_for`)
    first among its twins,
    best first, ties in the space's enumeration order; at a decode-sized ``rows`` the dense dispatcher's
    structural guard for a space without swapped records (`heuristic.decode_sane`: 1-CTA tiles in an N-cluster) selects
    the candidates, as it does for the bf16 classes. As in the dense ``rank_topk``, each of the best ceil(k/2) records is
    followed by its raster twin (the one knob the estimate does not price), then the rank order fills the list: distinct
    records, at most ``k``, never empty for an implementable shape."""
    import dataclasses

    from .blocks.compile_mxfp8 import can_implement_grouped_mxfp8

    fam = family or "sm_107"
    dense_n = (n_active or E) * N
    space = [
        c
        for c in grouped_knobs_mx(in_fmt)
        if can_implement_grouped_mxfp8(c, N, K, E, chain, in_fmt)
    ]
    guarded = [c for c in space if decode_sane(c, rows, False)]
    raster = raster_for(rows)

    def key(i, c):
        # the tile family priced once, at the enumeration's raster (the dense estimate's tile-order term is the dense
        # scheduler's, not the grouped one's), scaled by the pair term; the rule's raster first within the family, then
        # the enumeration order
        canon = dataclasses.replace(c, raster_along="m")
        est = estimate_cycles(
            canon, rows, dense_n, K, 1, "k", in_fmt, fam, sm_count, nominal=nominal
        )
        return (est * pair_term(c, rows), 0 if c.raster_along == raster else 1, i)

    ranked = [
        c for _k, c in sorted((key(i, c), c) for i, c in enumerate(guarded or space))
    ]
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
