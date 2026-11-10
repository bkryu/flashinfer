# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 grouped GEMMs' dispatcher: the dense MX class's estimate over both record spaces at the grouped launch's shape.

As for the bf16 classes (:mod:`.schedule`), a grouped GEMM is the dense class under a different scheduler, so its records
are ranked by the dense dispatcher's model (`flashinfer.gemm.batch_invariant.heuristic.estimate_cycles` for the MX class
of the part: the part's roofline with nominal constants, no measured tables) at the dense shape whose tiles, M tail and
operand bytes are the grouped launch's (``M`` = the rows of one active expert, ``N`` = the active experts' weight rows
side by side). Two record spaces are ranked together: the grouped MX space (the dense block-scaled records at swizzle 1
without in-cluster split, operand swap or B-reuse tile, filtered by the grouped block-scaled body's acceptance) and the
swapped records (:mod:`.blocks.swap_mxfp8`: the weights on the MMA's M side, 128 rows per CTA, the group's tokens a
narrow N tile of 8 / 16 / 32 rows on the dense packing), which the dense model prices as it prices a swapped decode
record, so the CTA count and the operand bytes per CTA of each body enter one estimate and the pick follows the shape,
never a rows-per-expert switch. The grouped launch's own terms on the estimate are the bf16 dispatcher's (`schedule`):
the raster rule and the pair term, the swapped records' spill over their token tile under the load, the last tile's
exposed epilogue chain (the MX requantization's term for the fused chain, four epilogue warpgroups) and the token box's
zero-fill at the e4m3 width. Numerics-neutral: every record of either space computes the class.
"""

from __future__ import annotations

import functools
from typing import List, Optional, Tuple

from ...gemm.batch_invariant.generator import (
    CLUSTERS,
    TILES_N_BLOCKSCALED,
    Knobs,
    knob_product,
)
from ...gemm.batch_invariant.heuristic import decode_sane, estimate_cycles
from .blocks.epilogue import GroupedChain
from .schedule import (
    TOP_K,
    active_experts,
    chain_nodes,
    exposed_epilogue_cycles,
    rows_per_expert,
    tiles_term,
    zero_fill_cycles,
)

GEMM2_CHAIN = GroupedChain("plain", "silu", "bf16")
RASTER_N_ROWS = 128  # an expert spanning a whole 128-row block or more: the grouped scheduler rasters along N
EPI_WARPGROUPS = 4  # the block-scaled grouped body's epilogue warpgroups (`blocks.blockscaled_kernel.EPI_WARPGROUPS_MX`)
PACKINGS = ("padded", "dense")


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


def swapped_knobs_mx() -> List[Knobs]:
    """The swapped record space of the MXFP8 classes (:mod:`.blocks.swap_mxfp8`): the dense MX class's swapped 1-CTA records
    at the narrow token tiles (128 weight rows x 8 / 16 / 32 tokens), one per tile. Disjoint from the grouped space (the
    swapped flag); the body reads the dense packing."""
    from .blocks.swap_mxfp8 import swapped_knobs

    return swapped_knobs()


def all_valid_configs_mx(
    N: int, K: int, E: int, chain: GroupedChain, in_fmt: str = "mxfp8", clusters=None
) -> List[Knobs]:
    """The grouped record space the guarantee is claimed over, for a weight shape, a chain and a class (the swapped records
    are the second space: :func:`all_valid_configs_mx_swapped`; the guarantee covers both)."""
    from .blocks.compile_mxfp8 import can_implement_grouped_mxfp8

    return [
        k
        for k in grouped_knobs_mx(in_fmt, clusters)
        if can_implement_grouped_mxfp8(k, N, K, E, chain, in_fmt)
    ]


def all_valid_configs_mx_swapped(
    N: int, K: int, E: int, chain: GroupedChain, in_fmt: str = "mxfp8"
) -> List[Knobs]:
    """The swapped records the swapped body accepts for a weight shape, a chain and a class (the dense packing)."""
    from .blocks.compile_mxfp8 import can_implement_swapped_mxfp8

    return [
        k
        for k in swapped_knobs_mx()
        if can_implement_swapped_mxfp8(k, N, K, E, chain, in_fmt)
    ]


def all_valid_configs_mxfp8(
    N: int, K: int, E: int, chain: GroupedChain, in_fmt: str = "mxfp8", clusters=None
) -> List[Knobs]:
    """Both record spaces for a weight shape, a chain and a class: the grouped records, then the swapped ones."""
    return all_valid_configs_mx(
        N, K, E, chain, in_fmt, clusters
    ) + all_valid_configs_mx_swapped(N, K, E, chain, in_fmt)


@functools.lru_cache(maxsize=None)
def chain_nodes_mx(chain: GroupedChain) -> float:
    """The node count of the chain's term per output element, the exposed epilogue's cost unit (:func:`.schedule.chain_nodes`):
    the bf16 chains' own; the fused requantization's (``out_fmt == "mxfp8"``) the bf16-rounded chain's plus the quantizer
    class's element chain (the inverse scale applied, the e4m3 conversion) and its share of the block's scale chain and amax
    tree (one scale per 32 outputs: the amax's ``fabs`` leaves and ``fmaxc`` combines, the scale's normalisation, UE8M0
    conversion and exact inverse)."""
    from ...batch_invariant.quantization import mx_spec
    from . import mterm as term
    from .blocks.epilogue_mx import OUT_FMT, bf16_twin

    if chain.out_fmt != OUT_FMT:
        return float(chain_nodes(chain))
    alg = term.TermAlgebra()
    fmt, rule = mx_spec.MXFP8, mx_spec.DEFAULT_RULE
    amax = alg.leaf("amax", 0)
    sf_d, inv = mx_spec.scale_chain(alg, amax, fmt, rule)
    elem = mx_spec.quant_elem(alg, alg.leaf("y", 0), alg.leaf("inv", 0), fmt, rule)
    scale_nodes = term.sexpr(inv).count("(")
    amax_nodes = (
        2 * mx_spec.BLOCK - 1
    )  # the fabs leaves and the pairwise fmaxc combines of one block
    return (
        float(chain_nodes(bf16_twin(chain)))
        + term.sexpr(elem).count("(")
        + (scale_nodes + amax_nodes) / float(mx_spec.BLOCK)
    )


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
    swap: Optional[bool] = None,
) -> List[Knobs]:
    """The ranked records for one GEMM over both record spaces: the grouped MX records the body accepts for the weight shape
    and the chain (at a decode-sized ``rows`` the dense dispatcher's structural guard, `heuristic.decode_sane`: 1-CTA tiles
    in an N-cluster, selects the candidates) and the swapped records the swapped body accepts (``swap``: None ranks them
    with the rest, True keeps only them, False excludes them), ordered by the dense MX class's estimate at ``(M = rows,
    N = n_active * N, K)`` (``n_active`` the active experts, ``E`` by default; a swapped record is priced as the dense model
    prices a swapped decode record) times the pair term (the expected dummy half of the two-tile clusters along M) and the
    swapped records' spill factor (`schedule.tiles_term`), plus the last tile's exposed epilogue chain
    (`schedule.exposed_epilogue_cycles` with the chain's term, :func:`chain_nodes_mx`, over the block-scaled body's four
    epilogue warpgroups) and the token box's zero-fill at the e4m3 width (`schedule.zero_fill_cycles`), each tile family
    priced once (at the enumeration's raster) and the raster the structural rule gives the grouped scheduler
    (`raster_for`) first among its twins, best first, ties in the spaces' enumeration order (the grouped records before the
    swapped ones, the narrower token tile first). As in the dense ``rank_topk``, each of the best ceil(k/2) records is
    followed by its raster twin (the one knob the estimate does not price), then the rank order fills the list: distinct
    records, at most ``k``, never empty for an implementable shape."""
    import dataclasses

    from .blocks.compile_mxfp8 import (
        can_implement_grouped_mxfp8,
        can_implement_swapped_mxfp8,
    )

    fam = family or "sm_107"
    dense_n = (n_active or E) * N
    dense = (
        []
        if swap is True
        else [
            c
            for c in grouped_knobs_mx(in_fmt)
            if can_implement_grouped_mxfp8(c, N, K, E, chain, in_fmt)
        ]
    )
    swapped = (
        []
        if swap is False
        else [
            c
            for c in swapped_knobs_mx()
            if can_implement_swapped_mxfp8(c, N, K, E, chain, in_fmt)
        ]
    )
    space = dense + swapped
    guarded = [c for c in dense if decode_sane(c, rows, False)] or dense
    raster = raster_for(rows)
    nodes = chain_nodes_mx(chain)

    def key(i, c):
        # the tile family priced once, at the enumeration's raster (the dense estimate's tile-order term is the dense
        # scheduler's, not the grouped one's); the grouped launch's own terms on the dense estimate: the cluster tile's
        # dummy share, the swapped records' spill under the load, the last tile's exposed epilogue chain, the token box's
        # zero-fill; three significant digits, so records the estimate cannot separate fall through to the structural
        # raster and the enumeration order
        canon = dataclasses.replace(c, raster_along="m")
        est = estimate_cycles(
            canon, rows, dense_n, K, 1, "k", in_fmt, fam, sm_count, nominal=nominal
        )
        est = (
            est * pair_term(c, rows) * tiles_term(c, rows)
            + exposed_epilogue_cycles(c, rows, nodes, epi_warpgroups=EPI_WARPGROUPS)
            + zero_fill_cycles(c, rows, K, fam, in_fmt)
        )
        return (
            float(f"{est:.3g}"),
            0 if (c.swap_ab or c.raster_along == raster) else 1,
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


def ranked_mx(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    chain1: GroupedChain,
    in_fmt: str = "mxfp8",
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
    chain1: GroupedChain,
    in_fmt: str = "mxfp8",
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    nominal: bool = False,
    swap: Optional[bool] = None,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I or I, K = H), config for GEMM2 (N = H, K = I))``: the first record of each ranked list over
    both record spaces (``swap``: None, or True / False to keep only / exclude the swapped records)."""
    c1, c2 = ranked_mx(
        T, k, H, I, E, chain1, in_fmt, sm_count, family, 1, nominal, swap
    )
    if not c1 or not c2:
        raise ValueError(
            f"no grouped MX record implements (H={H}, I={I}, E={E}, chain={chain1.describe()}, class={in_fmt})"
        )
    return c1[0], c2[0]


def packing_for(c1: Knobs, c2: Knobs) -> str:
    """The packing of the expanded rows the two records run on: the dense packing (the rows back to back, the scale factors
    at the padded positions) when either is a swapped record (its body reads nothing else; the padded body reads it too),
    the padded packing (whole 128-row atoms per group) otherwise. A placement: the same bits."""
    return "dense" if (c1.swap_ab or c2.swap_ab) else "padded"
