# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The mixed MXFP4 x MXFP8 grouped GEMMs' dispatcher: the dense mixed class's estimate over both record spaces at the grouped
launch's shape.

As for the bf16 classes (:mod:`.schedule`), a grouped GEMM is the dense class under a different scheduler, so its records
are ranked by the shared contract (`schedule.record_cost`: `flashinfer.gemm.batch_invariant.heuristic.estimate_cycles` for
the mixed class of the part -- the part's roofline with nominal constants and the mixed classes' per-operand traffic terms,
the e2m1 weight fetched at four bits and staged at the container's width, no measured tables) at the dense shape whose tiles,
M tail and operand bytes are the grouped launch's (``M`` = the rows of one active expert, ``N`` = the active experts' weight
rows side by side). Two record spaces are ranked together: the grouped mixed space (the dense block-scaled records at swizzle
1 without in-cluster split, operand swap, B-reuse tile or in-CTA chunk fold, filtered by the unified grouped body's
acceptance at the chain's instruction class) and the swapped decode space (the dense mixed class's swapped decode records at
the narrow token tiles, the weights on the MMA's M side, with the stage width the dense class gives the part's class:
:mod:`.blocks.swap` on the twin plan), each priced as the dense model prices it, with the grouped launch's own terms of the
bf16 classes' ranking: the pair term, the swapped records' spill over their token tile under the load, the last tile's
exposed epilogue chain (the fused requantization's element work on either body) and the token box's zero-fill at the e4m3
row width, plus this class's two declared empirical constants (the swapped body's per-launch prologue, the dead token
columns of its fused requantization epilogue). Numerics-neutral: every record of either space computes the class.
"""

from __future__ import annotations

import functools
from typing import List, Optional, Tuple

from ...gemm.batch_invariant.generator import (
    CLUSTERS,
    SWAP_K_BLOCK,
    TILES_N_BLOCKSCALED,
    TILES_N_NARROW,
    Knobs,
    knob_product,
)
from ...gemm.batch_invariant.heuristic import decode_sane
from .blocks.epilogue import GroupedChain, GroupedEpilogueBlock
from .blocks.epilogue_mx import OUT_FMT, MxChain
from .blocks.operands import EPI_WARPGROUPS_BLOCKSCALED
from .schedule import (
    TOP_K,
    active_experts,
    chain_nodes,
    raster_for,
    record_cost,
    rows_per_expert,
)
from .spec_mxfp4w8a import MX_CLASSES

#: GEMM2 of the mixed classes: the dequantised accumulator converted to bf16 (the instruction class is the launch's).
GEMM2_CHAIN = MxChain("plain", "silu", "bf16", (), MX_CLASSES[0])
#: The activation operand's format for the zero-fill term: e4m3 rows (the token box's bytes).
ACT_FMT = "mxfp8"
#: Nodes of the fused requantization's per-element term on top of the bf16 chain's output (the ``mxfp8.quant`` class's element
#: chain: the block's inverse scale, the scaled value and its e4m3 conversion), the DAG count as `chain_nodes` counts the chain's.
REQUANT_NODES = 3
#: Epilogue warpgroups of the block-scaled grouped body (the exposed epilogue term's divisor).
EPI_WARPGROUPS_MX = EPI_WARPGROUPS_BLOCKSCALED
PACKINGS = ("padded", "dense")


#: Cycles per dead token column per tile of the swapped body's fused requantization epilogue: the block amax exchange, the
#: scale and its traffic run for every column of the token tile whatever the live rows, so a wide tile at a few live rows pays
#: for its dead columns beyond what the chain's node count sees. One declared empirical constant of this class; a fully live
#: tile, the grouped body and the bf16 chains carry no such term.
SWAPPED_QUANT_COLUMN_CYCLES = 150.0


#: Cycles per launch of the swapped body's own prologue on the MX classes beyond the grouped body's: its group tables (the dense
#: and the padded offsets in one fused scan) and the token scale-factor gather, paid once per CTA whatever the live rows and
#: whatever the chain (GEMM2's conversion as much as GEMM1's quantizing chain). One declared empirical constant of this class;
#: the grouped body carries no such term.
SWAPPED_FIXED_CYCLES = 12000.0


def swapped_fixed_cycles(cfg: Knobs, chain: GroupedEpilogueBlock) -> float:
    """``SWAPPED_FIXED_CYCLES`` for a swapped record (the body's prologue, every chain); zero for the grouped body."""
    return SWAPPED_FIXED_CYCLES if cfg.swap_ab else 0.0


def swapped_quant_column_cycles(
    cfg: Knobs,
    rows: int,
    n_active: int,
    N: int,
    chain: GroupedEpilogueBlock,
    sm_count: int,
) -> float:
    """``SWAPPED_QUANT_COLUMN_CYCLES x the token tile's dead columns x the record's tiles per SM`` for a swapped record
    running the fused requantization chain (the dead columns: the tile less the live rows per expert; the tiles: the active
    experts' token tiles times the weight tiles, over the part's SMs); zero for a fully live tile, the grouped body and the
    bf16 chains."""
    if not cfg.swap_ab or chain.out_fmt != OUT_FMT:
        return 0.0
    dead = max(0, cfg.tile_n - max(1, rows))
    tiles = (
        max(1, n_active) * -(-max(1, rows) // cfg.tile_n) * -(-N // cfg.mma_tiler_mn[0])
    )
    return SWAPPED_QUANT_COLUMN_CYCLES * dead * tiles / max(1, sm_count)


def epilogue_nodes(chain: GroupedEpilogueBlock) -> int:
    """The per-element epilogue work of a GEMM1 chain: the bf16 chain's nodes, plus the requantization's for the fused
    spelling (``out_fmt == "mxfp8"``); GEMM2's plain conversion is one node."""
    # the bf16 classes' chain object the MX chain wraps (its conversion is one node whatever the output format), then the
    # requantization's nodes for the fused spelling
    quant = chain.out_fmt == OUT_FMT
    bf16 = GroupedChain(
        chain.kind, chain.act, "bf16" if quant else chain.out_fmt, chain.params
    )
    return chain_nodes(bf16) + (REQUANT_NODES if quant else 0)


def as_mixed_chain(chain: GroupedEpilogueBlock, in_fmt: str) -> MxChain:
    """The MX chain of the mixed class ``in_fmt`` for a bf16 chain object or an MX chain (its class replaced)."""
    assert in_fmt in MX_CLASSES, in_fmt
    return MxChain.of(chain, in_fmt)


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


def swapped_knobs_mx(in_fmt: str = "mxfp4w8a") -> List[Knobs]:
    """The swapped decode space of the mixed class (:mod:`.blocks.swap` on the twin plan): the dense mixed class's swapped
    1-CTA records at the narrow token tiles (128 weight rows x 8 / 16 / 32 tokens) at the dense class's stage width for the
    instruction class (``SWAP_K_BLOCK``: the sm_107 class streams its e2m1 weight at a 256-element stage, a 128-byte TMA row;
    the sm_100 class the format's default). The record's k-block is part of the record (the lowering's inner radix, a
    placement of the same fold); disjoint from the grouped space (the swapped flag)."""
    assert in_fmt in MX_CLASSES, in_fmt
    return [
        Knobs(
            (128, tn),
            (1, 1),
            False,
            True,
            "m",
            1,
            swap_ab=True,
            k_block=SWAP_K_BLOCK.get(in_fmt, 0),
        )
        for tn in TILES_N_NARROW
    ]


def can_implement_grouped_mxfp4w8a(
    knobs: Knobs, N: int, K: int, E: int, chain: GroupedEpilogueBlock, in_fmt: str
) -> bool:
    """Acceptance of the grouped body for the mixed classes (`blocks.compile.can_implement_grouped_blockscaled` at the chain
    taken as an MX chain of the instruction class ``in_fmt``): a well-formed dense record of the class's space at swizzle 1
    (no in-cluster split, no operand swap, no B-reuse tile, no in-CTA chunk fold) with a block-scaled tile N, a chain with a
    bf16 / f16 / f32 output or the fused requantization, K a whole number of the class's granules (32 on the sm_107 class,
    128 on the sm_100 / sm_103 one: the TMA's unpacking format of the e2m1 weight), N a multiple of 8, 16-byte-aligned
    output rows (whole 32-element blocks for the fused requantization), whole ``2 x GROUP`` column blocks for the gated
    chain, at most ``MAX_E`` experts."""
    from .blocks.compile import can_implement_grouped_blockscaled

    if in_fmt not in MX_CLASSES or not isinstance(chain, GroupedEpilogueBlock):
        return False
    return can_implement_grouped_blockscaled(
        knobs, N, K, E, as_mixed_chain(chain, in_fmt), in_fmt
    )


def can_implement_swapped_mxfp4w8a(
    knobs: Knobs, N: int, K: int, E: int, chain: GroupedEpilogueBlock, in_fmt: str
) -> bool:
    """Acceptance of the swapped body for the mixed classes (`blocks.compile.can_implement_swapped_blockscaled` at the chain
    taken as an MX chain): a swapped record of the dense mixed space at a narrow token tile (128 weight rows x 8 / 16 / 32
    tokens, 1-CTA, no cluster, swizzle 1, a k-block the class admits) under the grouped body's operand rules. The body reads
    the dense packing: the dense rows with the scale factors at the padded positions."""
    from .blocks.compile import can_implement_swapped_blockscaled

    if in_fmt not in MX_CLASSES or not isinstance(chain, GroupedEpilogueBlock):
        return False
    return can_implement_swapped_blockscaled(
        knobs, N, K, E, as_mixed_chain(chain, in_fmt), in_fmt
    )


def can_implement_mxfp4w8a(
    knobs: Knobs, N: int, K: int, E: int, chain: GroupedEpilogueBlock, in_fmt: str
) -> bool:
    """A record either body takes for the mixed classes: the grouped body (a dense record, either packing) or the swapped body
    (a narrow swapped record); the two spaces are disjoint."""
    return can_implement_grouped_mxfp4w8a(
        knobs, N, K, E, chain, in_fmt
    ) or can_implement_swapped_mxfp4w8a(knobs, N, K, E, chain, in_fmt)


def all_valid_configs_mx(
    N: int,
    K: int,
    E: int,
    chain: GroupedEpilogueBlock,
    in_fmt: str = "mxfp4w8a",
    clusters=None,
) -> List[Knobs]:
    """The grouped record space the guarantee is claimed over, for a weight shape, a chain and a class (the swapped records
    are the second space: :func:`all_valid_configs_mx_swapped`; the guarantee covers both)."""
    return [
        k
        for k in grouped_knobs_mx(in_fmt, clusters)
        if can_implement_grouped_mxfp4w8a(k, N, K, E, chain, in_fmt)
    ]


def all_valid_configs_mx_swapped(
    N: int, K: int, E: int, chain: GroupedEpilogueBlock, in_fmt: str = "mxfp4w8a"
) -> List[Knobs]:
    """The swapped records the swapped body accepts for a weight shape, a chain and a class (the dense packing)."""
    return [
        k
        for k in swapped_knobs_mx(in_fmt)
        if can_implement_swapped_mxfp4w8a(k, N, K, E, chain, in_fmt)
    ]


def all_valid_configs_mxfp4w8a(
    N: int,
    K: int,
    E: int,
    chain: GroupedEpilogueBlock,
    in_fmt: str = "mxfp4w8a",
    clusters=None,
) -> List[Knobs]:
    """Both record spaces for a weight shape, a chain and a class: the grouped records, then the swapped ones."""
    return all_valid_configs_mx(
        N, K, E, chain, in_fmt, clusters
    ) + all_valid_configs_mx_swapped(N, K, E, chain, in_fmt)


def rank_grouped_mx(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: GroupedEpilogueBlock,
    in_fmt: str = "mxfp4w8a",
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    k: int = TOP_K,
    nominal: bool = False,
    n_active: Optional[int] = None,
    swap: Optional[bool] = None,
) -> List[Knobs]:
    """:func:`_rank_grouped_mx` memoised on its arguments (all hashable scalars; the chain a frozen dataclass, taken at
    ``in_fmt``): the runner ranks at every forward without a tactic, once per token count."""
    chain = as_mixed_chain(chain, in_fmt)
    return list(
        _rank_grouped_mx(
            rows, N, K, E, chain, in_fmt, sm_count, family, k, nominal, n_active, swap
        )
    )


@functools.lru_cache(maxsize=16384)
def _rank_grouped_mx(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: MxChain,
    in_fmt: str,
    sm_count: Optional[int],
    family: Optional[str],
    k: int,
    nominal: bool,
    n_active: Optional[int],
    swap: Optional[bool],
) -> Tuple[Knobs, ...]:
    """The ranked records for one GEMM over both record spaces: the grouped mixed records the body accepts for the weight shape
    and the chain (at a decode-sized ``rows`` the dense dispatcher's structural guard, `heuristic.decode_sane`: 1-CTA tiles
    in an N-cluster, selects the candidates) and the swapped records the swapped body accepts for the chain (``swap``: None
    ranks them with the rest, True keeps only them, False excludes them), ordered by the shared contract's cost
    (`schedule.record_cost`: the dense mixed class's estimate at ``(M = rows, N = n_active * N, K)`` -- ``n_active`` the active
    experts, ``E`` by default; a swapped record priced as the dense model prices a swapped decode record -- under the MoE
    ranking's overrides, times the pair term and the swapped records' spill factor, plus the last tile's exposed epilogue
    chain (the chain's nodes, the fused requantization's on either body; the block-scaled body's four epilogue warpgroups) and
    the token box's exposed zero-fill at the e4m3 row width), plus this class's per-column cost of the swapped body's fused
    requantization epilogue over the token tile (`swapped_quant_column_cycles`) and the swapped body's per-launch prologue
    cost on the MX classes (`swapped_fixed_cycles`), at three significant digits; each tile family priced once (at the
    enumeration's raster) and the raster the structural rule gives the grouped scheduler (`schedule.raster_for`) first among
    its twins, best first, ties in the spaces' enumeration order (the grouped records before the swapped ones, the narrower
    token tile before the wider). As in the dense ``rank_topk``, each of the best ceil(k/2) records is followed by its raster
    twin (the one knob the estimate does not price), then the rank order fills the list: distinct records, at most ``k``,
    never empty for an implementable shape."""
    import dataclasses

    from ...gemm.batch_invariant.heuristic import PART_TABLE

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
        # ranking's overrides and the launch's own terms, the fill term at the e4m3 token width and the block-scaled body's
        # four epilogue warpgroups; then this class's two terms
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
    return tuple(out[:k])


def ranked_mx(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    chain1: GroupedEpilogueBlock,
    in_fmt: str = "mxfp4w8a",
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    top_k: int = TOP_K,
    nominal: bool = False,
    swap: Optional[bool] = None,
) -> Tuple[List[Knobs], List[Knobs]]:
    """The ranked lists of both GEMMs (GEMM1 on (N = 2I or I, K = H), GEMM2 on (N = H, K = I)) at the grouped launch's shape."""
    rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
    n1 = 2 * I if chain1.pair else I
    return (
        rank_grouped_mx(
            rows,
            n1,
            H,
            E,
            chain1,
            in_fmt,
            sm_count,
            family,
            top_k,
            nominal,
            n_act,
            swap,
        ),
        rank_grouped_mx(
            rows,
            H,
            I,
            E,
            GEMM2_CHAIN,
            in_fmt,
            sm_count,
            family,
            top_k,
            nominal,
            n_act,
            swap,
        ),
    )


def default_configs_mx(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    chain1: GroupedEpilogueBlock,
    in_fmt: str = "mxfp4w8a",
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    nominal: bool = False,
    swap: Optional[bool] = None,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I or I, K = H), config for GEMM2 (N = H, K = I))``: the first record of each ranked list
    over both record spaces (``swap``: both record spaces ranked together (None), the swapped records only (True) or the
    grouped ones only (False))."""
    c1, c2 = ranked_mx(
        T, k, H, I, E, chain1, in_fmt, sm_count, family, 1, nominal, swap
    )
    if not c1 or not c2:
        raise ValueError(
            f"no grouped mixed record implements (H={H}, I={I}, E={E}, chain={chain1.describe()}, class={in_fmt})"
        )
    return c1[0], c2[0]


def packing_for(c1: Knobs, c2: Knobs) -> str:
    """The packing of the expanded rows the two records run on: the dense packing (the rows back to back, the scale factors
    at the padded positions) when either is a swapped record (its body reads nothing else; the grouped body reads it too),
    the padded packing (whole 128-row atoms per group) otherwise. A placement: the same bits."""
    return "dense" if (c1.swap_ab or c2.swap_ab) else "padded"


# the bf16 chain object of GEMM2 for callers that build their own (the MX chain is the dispatcher's form)
GEMM2_BF16_CHAIN = GroupedChain("plain", "silu", "bf16")
