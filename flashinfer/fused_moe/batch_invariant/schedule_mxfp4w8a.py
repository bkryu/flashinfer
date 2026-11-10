# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The mixed MXFP4 x MXFP8 grouped GEMMs' dispatcher: the dense mixed class's estimate over both record spaces at the grouped
launch's shape.

As for the bf16 classes (:mod:`.schedule`), a grouped GEMM is the dense class under a different scheduler, so its records
are ranked by the dense dispatcher's model (`flashinfer.gemm.batch_invariant.heuristic.estimate_cycles` for the mixed
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
from ...gemm.batch_invariant.heuristic import decode_sane, estimate_cycles
from .blocks.epilogue import GroupedChain
from .schedule import (
    TOP_K,
    _EPI_WARPGROUPS,
    active_experts,
    chain_nodes,
    exposed_epilogue_cycles,
    pair_term,
    raster_for,
    rows_per_expert,
    tiles_term,
    zero_fill_cycles,
)

GEMM2_CHAIN = GroupedChain("plain", "silu", "bf16")
#: The activation operand's format for the zero-fill term: e4m3 rows (the token box's bytes).
ACT_FMT = "mxfp8"
#: Nodes of the fused requantization's per-element term on top of the bf16 chain (the ``mxfp8.quant`` class's element chain:
#: the scaled value and its e4m3 conversion), counted from its S-expression as `chain_nodes` counts the chain's.
REQUANT_NODES = 3


def epilogue_nodes(chain: GroupedChain) -> int:
    """The per-element epilogue work of a GEMM1 chain: the bf16 chain's nodes, plus the requantization's for the fused
    spelling (``out_fmt == "mxfp8"``); GEMM2's plain conversion is one node."""
    return chain_nodes(chain) + (REQUANT_NODES if chain.out_fmt == "mxfp8" else 0)


def exposed_epilogue_cycles_mx(cfg: Knobs, rows: int, nodes: int) -> float:
    """The bf16 ranking's exposed-epilogue term for this class's bodies: the swapped body as the bf16 one, the padded body's
    subtiles split between its four epilogue warpgroups (`blocks.blockscaled_kernel.EPI_WARPGROUPS_MX`) where the bf16
    body's are split between two."""
    from .blocks.blockscaled_kernel import EPI_WARPGROUPS_MX

    cycles = exposed_epilogue_cycles(cfg, rows, nodes)
    return cycles if cfg.swap_ab else cycles * _EPI_WARPGROUPS / EPI_WARPGROUPS_MX


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
    False excludes them), ordered by the dense mixed class's estimate at ``(M = rows, N = n_active * N,
    K)`` (``n_active`` the active experts, ``E`` by default; a swapped record priced as the dense model prices a swapped
    decode record) times the pair term (the expected dummy half of the two-tile clusters along M) and the swapped records'
    spill factor (`tiles_term`), plus the last tile's exposed epilogue chain (`exposed_epilogue_cycles_mx`: the chain's
    nodes, the fused requantization's on either body) and the token box's zero-fill (`zero_fill_cycles` at the e4m3
    row width), at three significant digits; each tile family priced once (at the enumeration's raster: the dense estimate's
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

    fam = family or "sm_107"
    dense_n = (n_active or E) * N
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
        canon = dataclasses.replace(c, raster_along="m")
        est = estimate_cycles(
            canon, rows, dense_n, K, 1, "k", in_fmt, fam, sm_count, nominal=nominal
        )
        est = (
            est * pair_term(c, rows) * tiles_term(c, rows)
            + exposed_epilogue_cycles_mx(c, rows, nodes)
            + zero_fill_cycles(c, rows, K, fam, ACT_FMT)
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
