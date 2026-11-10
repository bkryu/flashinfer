# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The mixed MXFP4 x MXFP8 grouped GEMMs' dispatcher: the dense mixed class's estimate over both record spaces at the grouped
launch's shape.

As for the bf16 classes (:mod:`.schedule`), a grouped GEMM is the dense class under a different scheduler, so its records
are ranked by the shared contract (`schedule.record_cost`: `flashinfer.gemm.batch_invariant.heuristic.estimate_cycles` for the mixed
class of the part: the part's roofline with nominal constants and the mixed classes' per-operand traffic terms -- the
e2m1 weight fetched at four bits, staged at the container's width -- no measured tables) at the dense shape whose tiles,
M tail and operand bytes are the grouped launch's (``M`` = the rows of one active expert, ``N`` = the active experts'
weight rows side by side). Two record spaces are ranked together: the grouped mixed space (the dense block-scaled records
at swizzle 1 without in-cluster split, operand swap, B-reuse tile or in-CTA chunk fold, filtered by the grouped body's
acceptance) and the swapped decode space (the dense mixed class's swapped decode records at the narrow token tiles, the
weights on the MMA's M side: :mod:`.blocks.swap_nvfp4` on the mixed plans), each priced as the dense model prices it, with
the grouped launch's own terms of the bf16 classes' ranking (:mod:`.schedule`): the pair term, the swapped records' spill
over their token tile under the load, the last tile's exposed epilogue chain (the fused requantization's element work on the
padded records) and the token box's zero-fill. Numerics-neutral: every record of either space computes the class.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ...gemm.batch_invariant.generator import (
    CLUSTERS,
    TILES_N_BLOCKSCALED,
    Knobs,
    knob_product,
)
from ...gemm.batch_invariant.heuristic import decode_sane
from .blocks.epilogue import GroupedChain
from .schedule import (
    TOP_K,
    active_experts,
    chain_nodes,
    raster_for,
    record_cost,
    rows_per_expert,
)

GEMM2_CHAIN = GroupedChain("plain", "silu", "bf16")
#: The activation operand's format for the zero-fill term: e4m3 rows (the token box's bytes).
ACT_FMT = "mxfp8"
#: Nodes of the fused requantization's per-element term on top of the bf16 chain's output (the ``mxfp8.quant`` class's element
#: chain: the block's inverse scale, the scaled value and its e4m3 conversion), the DAG count as `chain_nodes` counts the chain's.
REQUANT_NODES = 3


#: Cycles per dead token column per tile of the swapped body's fused requantization epilogue: the block amax exchange, the
#: scale and its traffic run for every column of the token tile whatever the live rows, so a wide tile at a few live rows pays
#: for its dead columns beyond what the chain's node count sees. One declared empirical constant of this class, fitted on the
#: 8 / 16 / 32-token ladders at four and twelve live rows per expert; a fully live tile, the padded body and the bf16 chains
#: carry no such term.
SWAPPED_QUANT_COLUMN_CYCLES = 150.0


#: Cycles per launch of the swapped body's own prologue on the MX classes beyond the grouped body's: its group tables (the dense
#: and the padded offsets in one fused scan) and the token scale-factor gather, paid once per CTA whatever the live rows and whatever
#: the chain (GEMM2's conversion as much as GEMM1's quantizing chain). One declared empirical constant of this class, fitted on the
#: decode cells of the campaign; a per-launch term cannot also price the swapped records' per-k-tile deficit at sixteen rows per
#: expert (the padded body's lead there is per k-tile: docs, the open follow-on), so the value is kept where no value keeps every
#: stamped cell within the regret bound. The padded body carries no such term.
SWAPPED_FIXED_CYCLES = 12000.0


def swapped_fixed_cycles(cfg: Knobs, chain: GroupedChain) -> float:
    """``SWAPPED_FIXED_CYCLES`` for a swapped record (the body's prologue, every chain); zero for the padded body."""
    return SWAPPED_FIXED_CYCLES if cfg.swap_ab else 0.0


def swapped_quant_column_cycles(
    cfg: Knobs, rows: int, n_active: int, N: int, chain: GroupedChain, sm_count: int
) -> float:
    """``SWAPPED_QUANT_COLUMN_CYCLES x the token tile's dead columns x the record's tiles per SM`` for a swapped record
    running the fused requantization chain (the dead columns: the tile less the live rows per expert; the tiles: the active
    experts' token tiles times the weight tiles, over the part's SMs); zero for a fully live tile, the padded body and the
    bf16 chains."""
    if not cfg.swap_ab or chain.out_fmt != ACT_FMT:
        return 0.0
    dead = max(0, cfg.tile_n - max(1, rows))
    tiles = (
        max(1, n_active) * -(-max(1, rows) // cfg.tile_n) * -(-N // cfg.mma_tiler_mn[0])
    )
    return SWAPPED_QUANT_COLUMN_CYCLES * dead * tiles / max(1, sm_count)


def epilogue_nodes(chain: GroupedChain) -> int:
    """The per-element epilogue work of a GEMM1 chain: the bf16 chain's nodes, plus the requantization's for the fused
    spelling (``out_fmt == "mxfp8"``); GEMM2's plain conversion is one node."""
    return chain_nodes(chain) + (REQUANT_NODES if chain.out_fmt == "mxfp8" else 0)


def grouped_knobs_mx(in_fmt: str = "mxfp4w8a", clusters=None) -> List[Knobs]:
    """The grouped mixed space before the body's shape acceptance: every well-formed dense record of the class at swizzle
    1 with a block-scaled tile N, no in-cluster split, no operand swap, no B-reuse tile, no in-CTA chunk fold (the group
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
    N: int, K: int, E: int, chain: GroupedChain, in_fmt: str = "mxfp4w8a", clusters=None
) -> List[Knobs]:
    """The configuration space the grouped mixed guarantee is claimed over, for a weight shape, a chain and a class."""
    from .blocks.compile_mxfp4w8a import can_implement_grouped_mxfp4w8a

    return [
        k
        for k in grouped_knobs_mx(in_fmt, clusters)
        if can_implement_grouped_mxfp4w8a(k, N, K, E, chain, in_fmt)
    ]


def all_valid_configs_mx_swapped(
    N: int, K: int, E: int, chain: GroupedChain, in_fmt: str = "mxfp4w8a"
) -> List[Knobs]:
    """The swapped decode records the swapped body accepts for a weight shape, a chain and a class (the bf16 / f32 chains;
    the padded packing)."""
    from .blocks.compile_mxfp4w8a import (
        can_implement_swapped_mxfp4w8a,
        swapped_knobs_mx,
    )

    return [
        k
        for k in swapped_knobs_mx(in_fmt)
        if can_implement_swapped_mxfp4w8a(k, N, K, E, chain, in_fmt)
    ]


def rank_grouped_mx(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: GroupedChain,
    in_fmt: str = "mxfp4w8a",
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    k: int = TOP_K,
    nominal: bool = False,
    n_active: Optional[int] = None,
    swap: Optional[bool] = None,
) -> List[Knobs]:
    """The ranked records for one GEMM over both record spaces: the grouped mixed records the padded body accepts for the
    weight shape and the chain (at a decode-sized ``rows`` the dense dispatcher's structural guard, `heuristic.decode_sane`:
    1-CTA tiles in an N-cluster, selects the candidates) and the swapped records the swapped body accepts for the chain
    (either spelling: the transposed quantizing epilogue; ``swap``: None ranks them with the rest, True keeps only them,
    False excludes them), ordered by the shared contract's cost (`schedule.record_cost`: the dense mixed class's estimate at ``(M = rows, N = n_active * N,
    K)`` (``n_active`` the active experts, ``E`` by default; a swapped record priced as the dense model prices a swapped decode
    record) under the MoE ranking's overrides, times the pair term and the swapped records' spill factor, plus the last tile's
    exposed epilogue chain (the chain's nodes, the fused requantization's on either body; the padded body's four epilogue
    warpgroups) and the token box's exposed zero-fill at the e4m3 row width), plus this class's per-column cost of the swapped
    body's fused requantization epilogue over the token tile (`swapped_quant_column_cycles`) and the swapped body's per-launch
    prologue cost on the MX classes (`swapped_fixed_cycles`), at three significant digits; each tile family priced once (at the enumeration's raster: the dense estimate's
    tile-order term is the dense scheduler's, not the grouped one's) and the raster the structural rule gives the grouped
    scheduler (`raster_for`) first among its twins, best first, ties in the spaces' enumeration order (the grouped records
    before the swapped ones, the narrower token tile before the wider). As in the dense ``rank_topk``, each of the best
    ceil(k/2) records is followed by its raster twin (the one knob the estimate does not price), then the rank order fills
    the list: distinct records, at most ``k``, never empty for an implementable shape."""
    import dataclasses

    from .blocks.compile_mxfp4w8a import (
        can_implement_grouped_mxfp4w8a,
        can_implement_swapped_mxfp4w8a,
        swapped_knobs_mx,
    )

    from ...gemm.batch_invariant.heuristic import PART_TABLE
    from .blocks.blockscaled_kernel import EPI_WARPGROUPS_MX

    fam = family or "sm_107"
    sms = sm_count or PART_TABLE.get(fam, PART_TABLE["sm_107"]).sm_count
    dense = (
        []
        if swap is True
        else [
            c
            for c in grouped_knobs_mx(in_fmt)
            if can_implement_grouped_mxfp4w8a(c, N, K, E, chain, in_fmt)
        ]
    )
    swapped = (
        []
        if swap is False
        else [
            c
            for c in swapped_knobs_mx(in_fmt)
            if can_implement_swapped_mxfp4w8a(c, N, K, E, chain, in_fmt)
        ]
    )
    space = dense + swapped
    guarded = [c for c in dense if decode_sane(c, rows, False)] or dense
    nodes = epilogue_nodes(chain)

    def key(i, c):
        # the shared contract's cost (`schedule.record_cost`): the dense mixed estimate at the grouped shape under the MoE
        # ranking's overrides and the launch's own terms, the fill term at the e4m3 token width and the padded body's four
        # epilogue warpgroups
        est = (
            record_cost(
                c,
                rows,
                n_active or E,
                N,
                K,
                nodes,
                chain.pair,
                fam,
                sm_count,
                in_fmt,
                nominal,
                fill_fmt=ACT_FMT,
                epi_warpgroups=EPI_WARPGROUPS_MX,
            )
            + swapped_quant_column_cycles(c, rows, n_active or E, N, chain, sms)
            + swapped_fixed_cycles(c, chain)
        )
        return (
            float(f"{est:.3g}"),
            0 if (c.swap_ab or c.raster_along == raster_for(c, rows)) else 1,
            i,
        )

    ranked = [
        c for _k, c in sorted((key(i, c), c) for i, c in enumerate(guarded + swapped))
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
    in_fmt: str = "mxfp4w8a",
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    nominal: bool = False,
    swap: Optional[bool] = None,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I or I, K = H), config for GEMM2 (N = H, K = I))``: the first record of each ranked list
    (``swap``: both record spaces ranked together (None), the swapped records only (True) or the grouped ones only (False))."""
    rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
    n1 = 2 * I if chain1.pair else I
    c1 = rank_grouped_mx(
        rows, n1, H, E, chain1, in_fmt, sm_count, family, 1, nominal, n_act, swap
    )
    c2 = rank_grouped_mx(
        rows, H, I, E, GEMM2_CHAIN, in_fmt, sm_count, family, 1, nominal, n_act, swap
    )
    if not c1 or not c2:
        raise ValueError(
            f"no grouped mixed record implements (H={H}, I={I}, E={E}, chain={chain1.describe()}, class={in_fmt})"
        )
    return c1[0], c2[0]
