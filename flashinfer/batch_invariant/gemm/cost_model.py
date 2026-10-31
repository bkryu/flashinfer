"""Physics-informed cost model of the bf16 GEMM records: the dispatcher of `mm_bf16(batch_invariant=True)`.

For one knob record at one shape the model predicts the device time (us) from the hardware's structure and a handful of fitted
constants; `rank` evaluates a whole configuration space at once (numpy, a few ms for 720 records) and `schedule.default_config`
takes the argmin. Nothing here reaches the bits: every record of the space computes the class (R3), the model only orders them.

Terms (per record; the symbols are the fields of `Arch` and `Fit`):

* **tiles and waves**: the MMA tile is `tile_m x tile_n` (two CTAs of `tile_m / 2` rows for the 2-CTA instruction); the scheduler's
  unit is a cluster tile of `cm x cn` CTAs; the persistent grid holds `P = SM // (cm cn)` clusters and a CTA runs `waves = ceil(units / P)`
  tiles in turn (the split class's workspace record has `C` units per output tile; the in-CTA record folds the `C` chunks on one CTA).
* **per k-tile time** (one 128-byte smem stage = `inner` instruction K-steps): the largest of the instruction time
  (`2 ctm ctn k / (flop_per_clk x f x mma_eff)`, with a minimum per instruction for the small tiles the issue rate bounds), the TMA
  path -- the bytes this CTA's TMA unit fetches from L2 (its multicast share: `A / cn + B / pairs_m`) over `fetch_gbps` plus the
  per-k-tile issue and mbarrier hand-shake cost (`ktile_ns`, `ktile_2cta_ns` for the pair's cross-CTA hand-shake; `cl4_ns` on top for
  clusters of four or more CTAs, whose multicast fan-in is not hidden) plus the zero-fill of the A box's rows past M (`oob_gbps`: the
  decode regime's dominant per-CTA cost, split among an N-cluster's CTAs), the smem ingress
  of the whole stage and the smem bandwidth the stage's writes share with the MMA's operand reads (`smem_gbps`: a wide 1-CTA tile is bound
  there, the 2-CTA pair of the same CTA tile is not) -- (`(A + B) / ingress`; the `n` layout's MN-major B box with rows under 128 B pays `nmajor_pen`), and the ring's
  refill under memory latency (Little's law: `latency / stages`, the stage count from the smem budget as `StaticPlan._stages` forms
  it). Ghost units: the scheduler rounds the cluster grid up to the swizzle along the raster's minor axis and a cluster spans
  `cm x cn` tiles, so one-M-tile problems run ghost CTAs on zero-filled A at full cost (the decode pathologies of the measurements).
* **epilogue**: `ctm ctn out_bytes / epi_gbps + tile_us` per tile (TMEM -> registers -> smem -> TMA store, the scheduler step); it
  overlaps the next tile's mainloop when two TMEM accumulator stages exist, and is serialised when one does (the in-CTA chunk fold's
  running sum at tile N 256 / 64-row tiles); the in-CTA fold adds one TMEM read + write pass per chunk.
* **aggregate bounds**: the L2 -> SM traffic (`units x k-tiles x (A / cn + B / pairs_m)` bytes: multicast shares the L2 reads, not the
  ingress) against `l2_tbps`, and the HBM traffic against `hbm_eff x hbm_tbps`. HBM bytes follow the persistent raster: with raster
  along M and swizzle `s` the tile sequence walks all M for `s` N-columns of clusters (a group), so the walked operand (A) is streamed
  once per group pass -- the groups a wave spans share a pass -- and every pass after the first misses the L2 for the part of
  (A + the group's B) that does not fit `l2_frac` of it (the cyclic re-use distance is the operand); the group's fixed operand (B) is read
  once per group and re-read every wave of the group only when it does not stay resident next to the wave's walked footprint
  (transposed for raster along N); plus the output, the split class's fp32 partials out and back and the fixup node's own floor.
* **the hot-line bound** of small M: the A box fetches only its valid rows (a 128-byte line per row per k-tile), and with M rows the
  machine's A requests land on M lines, i.e. M L2 home slices (`hot_gbps` each); at M = 1 every CTA's A fetch is served by one slice,
  so decode time grows with the number of tiles and the A multicast of the N-clusters is what relieves it.
* `time = launch + p-norm(fill + waves x tile time + the last epilogue, HBM, L2 (hits + `miss_cost` x misses), hot lines)` -- a soft
  maximum (`softmax_p`): bounds of similar size interfere (an HBM-heavy raster slows a compute-bound tile) -- times a cluster-coupling
  penalty per extra CTA, plus the fixup node for the workspace split.

Structural (never re-fitted): the instruction rates per clock, the tile geometry, the stage counts, the raster / footprint rule, the
TMEM stage rule. Fitted per architecture family on full-space CUPTI sweeps (every record of the space timed at every point of an
M x (N, K) grid, both weight layouts): the `Fit` constants, stored in `cost_model_fits.json` per family. `Arch` holds the part's
structural numbers (the SM count is a parameter within a family); an entry marked ``placeholder`` carries public figures only.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .generator import Knobs, inner_radix
from .spec import ELEM_BITS, MMA_K

# ----------------------------------------------------------------------------- the part


@dataclass(frozen=True)
class Arch:
    """Structural numbers of a part. ``flop_per_clk_sm`` is the dense bf16 rate of one SM's tensor core (tcgen05 kind::f16).
    ``hbm_tbps`` is the measured copy bandwidth where known (the nominal figure otherwise, marked by
    ``placeholder``). ``family`` is the architecture generation whose fitted constants (`Fit`, keyed by family in
    `cost_model_fits.json`) apply: the per-SM constants are the generation's, the SM count is a parameter of the part."""

    name: str
    sm_count: int
    clock_ghz: float  # sustained SM clock under a GEMM load
    flop_per_clk_sm: int = 8192  # dense bf16 FLOP / clk / SM
    hbm_tbps: float = 9.1  # measured 2-8 GB copy bandwidth on this part (7.0 nominal)
    l2_bytes: int = 112 << 20
    smem_bytes: int = 334848  # cutlass.memory.get_smem_capacity_in_bytes() on sm_107a: what StaticPlan budgets against
    tmem_cols: int = 512  # the dense body plans the 512-column TMEM (sm_107 has 576)
    placeholder: bool = False
    family: str = "sm_107"  # the fit's key (sm_107 / sm_100 / sm_103)


# The parts the dispatcher serves: Rubin NVL8 (sm_107, 204 SMs), Vera Rubin NVL72 (sm_107, 224),
# B200 (sm_100, 148), GB200 (sm_100, 152), B300 (sm_103, 148), GB300 (sm_103, 152). Within a family the SM count is the only
# difference the model sees (`P = SM // cluster`, the waves, the hot-line and aggregate bounds); the family carries the fitted
# per-SM constants and the part the memory-system numbers. The first entry of a family is its canonical part (the one measured).
# sm_100 numbers: 148 SMs, max SM clock 1.83 GHz (sustained ~1.75 under a GEMM load), HBM3e 8 TB/s nominal (7.4 measured-class),
# 126 MB L2, 228 KB smem (232448 B as the DSL budgets it), 512 TMEM columns; sm_103 (B300 / GB300) the same memory system at the
# same SM counts, public figures only (placeholder until measured; its fit falls back to sm_100's).
_SM100 = dict(clock_ghz=1.75, hbm_tbps=7.4, l2_bytes=126 << 20, smem_bytes=232448)
ARCHS: Dict[str, Arch] = {
    # Rubin NVL8 (SM107): 204 SMs, 112 MB L2; the 256x256 2-CTA mainloop sustains 36 TFLOP/s per pair at 16384 x 8192^2
    # (`mma_eff` x 8192 FLOP / clk / SM x 2.3 GHz)
    "sm_107": Arch("sm_107", 204, 2.3),
    # Vera Rubin NVL72 (SM107): 224 SMs, otherwise the same part (the fit's constants carry over; the SM count is a parameter)
    "sm_107_224": Arch("sm_107_224", 224, 2.3),
    "sm_100": Arch("sm_100", 148, family="sm_100", **_SM100),  # B200 (measured)
    "sm_100_152": Arch("sm_100_152", 152, family="sm_100", **_SM100),  # GB200
    "sm_103": Arch("sm_103", 148, family="sm_103", placeholder=True, **_SM100),  # B300
    "sm_103_152": Arch(
        "sm_103_152", 152, family="sm_103", placeholder=True, **_SM100
    ),  # GB300
}
FAMILY_OF_CC = {(10, 0): "sm_100", (10, 3): "sm_103", (10, 7): "sm_107"}
FIT_FALLBACK = {
    "sm_103": "sm_100",
    "sm_100": "sm_107",
}  # a family without its own fit takes its nearest measured relative's


def family_of(cc: Tuple[int, int]) -> str:
    """The architecture family of a compute capability (the fits' key); the API passes the operand device's."""
    return FAMILY_OF_CC.get(tuple(cc), "sm_107")


def arch_for(
    sm_count: Optional[int] = None,
    name: Optional[str] = None,
    family: Optional[str] = None,
) -> Arch:
    """The part the dispatcher models: by name; else the ``family`` (the operand device's, `family_of`; sm_107 when not given) at
    ``sm_count`` SMs (204 when not given). A (family, SM count) with an entry in `ARCHS` is that part; any other count is the
    family's canonical part with the SM count replaced (an emulated SKU: `name = <family>_<count>`, e.g. ``sm_107_148`` for a
    148-SM grid cap on a Rubin part) -- the measured override cells of a part never apply to it, only the model does."""
    if name:
        return ARCHS[name]
    family = family or "sm_107"
    if sm_count is None:
        sm_count = 204
    for a in ARCHS.values():
        if a.family == family and a.sm_count == sm_count:
            return a
    base = next((a for a in ARCHS.values() if a.family == family), ARCHS["sm_107"])
    return replace(base, name=f"{family}_{sm_count}", sm_count=sm_count)


# ----------------------------------------------------------------------------- the fitted constants


@dataclass(frozen=True)
class Fit:
    """The empirical constants (fitted per architecture family; the defaults are first-principles
    guesses that the entries of `cost_model_fits.json` replace). Two kinds: **per-SM physics** -- the instruction efficiency
    (`mma_eff`, `inst_min_ns`), the per-CTA TMA path (`ktile_ns`, `ktile_2cta_ns`, `cl4_ns`,
    `fetch_gbps`, `oob_gbps`, `ingress_gbps`, `nmajor_pen`), the ring's latency (`latency_us`), the epilogue (`epi_gbps`,
    `tile_us`, `fill_us`) and the cluster coupling (`cluster_pen`) -- which carry across the parts of a family (the SM count enters
    only through the structure: P, waves, aggregate demand); and **part-level** -- the aggregate L2 (`l2_tbps`, `hot_gbps`,
    `l2_frac`, `miss_cost`), the HBM efficiency (`hbm_eff`), the launch and fixup floors (`launch_us`, `fixup_us`) and the bounds'
    interference (`softmax_p`) -- which a part with another memory system re-fits."""

    mma_eff: float = 0.92  # sustained fraction of the instruction rate in a long mainloop (kind::f16)
    inst_min_ns: float = (
        8.0  # the smallest time one tcgen05.mma takes (issue-bound narrow tiles)
    )
    ktile_ns: float = 60.0  # per k-tile per CTA: TMA issue + mbarrier hand-shakes, serial with the fetch on the TMA path
    ktile_2cta_ns: float = 60.0  # the same on the 2-CTA instruction (the pair's hand-shake overlaps the MMA in steady state)
    fetch_gbps: float = 140.0  # per-SM TMA fetch from L2 (the bytes this CTA's TMA unit requests: its multicast share), GB/s
    hot_gbps: float = 120.0  # L2 bandwidth of ONE line's home slice: at small M every CTA fetches the same few A lines per k-tile
    oob_gbps: float = 70.0  # the TMA engine's zero-fill rate for out-of-bounds box rows (the A tile's rows past M; per issuing CTA)
    cl4_ns: float = 70.0  # per k-tile cost of a cluster of four or more CTAs (the multicast fan-in / slot-release hand-shake is not hidden)
    miss_cost: float = 2.0  # L2 bandwidth consumed per byte that misses to HBM (the fill and its read), on top of the hit traffic
    softmax_p: float = 3.0  # the aggregate bounds combine as a p-norm (p -> inf: a pure max; finite: HBM traffic slows a compute-bound tile)
    ingress_gbps: float = 220.0  # per-SM smem ingress (every byte that lands in this CTA's ring, multicast or not), GB/s
    smem_gbps: float = 300.0  # per-SM smem bandwidth shared by the TMA's writes and the MMA's operand reads (~128 B/clk): the 1-CTA
    # 128x256 tile reads 12 KB per instruction from ONE CTA's smem where the 2-CTA 256x256 pair reads 8 KB
    # per CTA -- measured 1.3x slower at M = 32768 with the same MMA, ingress and L2 figures (2026-10-31)
    l2_tbps: float = 28.0  # aggregate L2 -> SM bandwidth, TB/s
    hbm_eff: float = 0.85  # fraction of the copy bandwidth a streaming GEMM reaches
    latency_us: float = 1.3  # memory latency behind the ring (Little's law)
    launch_us: float = (
        3.0  # kernel floor: graph-node launch, prologue (TMEM alloc, barriers), tail
    )
    fill_us: float = 0.7  # the first stage's latency per persistent CTA
    tile_us: float = (
        0.25  # per tile: scheduler step, accumulator hand-off, store commit
    )
    epi_gbps: float = 120.0  # epilogue bandwidth per SM (TMEM -> RF -> smem -> TMA store), output bytes
    fixup_us: float = 2.5  # the split class's fixup node: launch gap + its own floor
    cluster_pen: float = 0.02  # coupling penalty per extra CTA in the cluster
    l2_frac: float = 0.8  # usable fraction of the L2 for the residency rule
    nmajor_pen: float = 1.2  # residual multiplier on the line-granular B traffic of the `n` layout when the B box row is under 128 bytes
    two_cta_gain: float = 1.0  # per-k-tile time multiplier of the 2-CTA instruction (< 1: one issue for 256 rows)


FITS_PATH = os.path.join(os.path.dirname(__file__), "cost_model_fits.json")


def _load_fits() -> Dict[str, Fit]:
    out: Dict[str, Fit] = {}
    if os.path.exists(FITS_PATH):
        with open(FITS_PATH) as f:
            for name, d in json.load(f).items():
                out[name] = Fit(
                    **{k: v for k, v in d.items() if k in Fit.__dataclass_fields__}
                )
    return out


FITS: Dict[str, Fit] = _load_fits()


def fit_for(arch: Arch) -> Fit:
    """The part's fitted constants: its family's entry of `cost_model_fits.json` (the SM count is a parameter of the structure, the
    per-SM constants carry across the family's parts), else the nearest measured family's (`FIT_FALLBACK`), else the first-principles
    defaults."""
    fam: Optional[str] = arch.family
    while fam:
        if fam in FITS:
            return FITS[fam]
        fam = FIT_FALLBACK.get(fam)
    return FITS.get("sm_107") or Fit()


# ----------------------------------------------------------------------------- the model


@functools.lru_cache(maxsize=None)
def _epi_tile(ctm: int, ctn: int, two: bool, out_bytes: int) -> Tuple[int, int]:
    """The epilogue subtile the body stores through (the library's rule, pure Python), (rows, cols)."""
    try:
        from cutlass.utils.blackwell_helpers import compute_epilogue_tile_size

        return tuple(
            int(v)
            for v in compute_epilogue_tile_size(
                ctm, ctn, two, out_bytes * 8, None, False, True
            )
        )
    except Exception:  # noqa: BLE001  (no DSL at hand)
        return (ctm, min(ctn, 32 if ctm == 128 else 64))


def knob_arrays(recs: Sequence[Knobs]) -> Dict[str, np.ndarray]:
    """The record fields the model reads, as arrays (one entry per record)."""
    return dict(
        tm=np.array([k.tile_m for k in recs], dtype=np.float64),
        tn=np.array([k.tile_n for k in recs], dtype=np.float64),
        cm=np.array([k.cluster_m for k in recs], dtype=np.float64),
        cn=np.array([k.cluster_n for k in recs], dtype=np.float64),
        two=np.array([k.use_2cta_instrs for k in recs], dtype=bool),
        rm=np.array([k.raster_along == "m" for k in recs], dtype=bool),
        sw=np.array([k.swizzle_size for k in recs], dtype=np.float64),
        cc=np.array([k.cta_chunks for k in recs], dtype=bool),
    )


def predict_arrays(
    a: Dict[str, np.ndarray],
    M,
    N,
    K,
    chunks,
    b_major: str,
    out_bytes: int,
    arch: Arch,
    fit: Fit,
    parts: bool = False,
):
    """The model on arrays: one entry per (record, shape); ``M``, ``N``, ``K``, ``chunks`` scalars or arrays matching the records."""
    M = np.asarray(M, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    chunks = np.asarray(chunks, dtype=np.float64)
    elem = ELEM_BITS / 8.0
    inst_k = MMA_K
    inner = inner_radix()
    kb = inst_k * inner  # K elements per 128-byte stage
    fpc = arch.flop_per_clk_sm
    two = a["two"]
    cta_div = np.where(two, 2.0, 1.0)
    ctm = a["tm"] / cta_div  # the CTA's rows
    ctn = a["tn"]
    cm, cn = a["cm"], a["cn"]
    cl = cm * cn
    pairs_m = (
        cm / cta_div
    )  # MMA tiles (CTA pairs) along M in the cluster: B is multicast across them
    split = chunks > 1
    ws = (
        split & ~a["cc"]
    )  # the workspace split (C units per output tile) vs the in-CTA fold
    nchunk_cta = np.where(a["cc"] & split, chunks, 1.0)
    Kc = K / chunks

    # tiles, cluster units, waves. The persistent scheduler rounds the cluster grid up to the swizzle along the raster's minor axis
    # (`PersistentTileSchedulerParams`: `round_up(shape, (1, s, 1))` for raster m, `(s, 1, 1)` for raster n), so a problem with one M
    # tile and raster n at swizzle 8 runs 8 "ghost" cluster rows whose CTAs stream zero-filled A and the real B at full cost; likewise a
    # cluster of 4 along M at one M tile runs 3 ghost CTAs. Every unit, ghost or not, costs a tile time.
    m_tiles = np.ceil(M / a["tm"])
    n_tiles = np.ceil(N / ctn)
    m_cl = np.ceil(m_tiles / pairs_m)
    n_cl = np.ceil(n_tiles / cn)
    s = a["sw"]
    rm = a["rm"]
    n_cl = np.where(rm, np.ceil(n_cl / s) * s, n_cl)
    m_cl = np.where(rm, m_cl, np.ceil(m_cl / s) * s)
    units = m_cl * n_cl * np.where(ws, chunks, 1.0)
    P = np.floor(arch.sm_count / cl)
    waves = np.ceil(units / P)
    k_tiles = np.ceil(Kc / kb) * nchunk_cta  # k-tiles per tile per CTA

    # per k-tile time
    mma_eff = fit.mma_eff
    t_inst = np.maximum(
        2.0 * ctm * ctn * inst_k / (fpc * arch.clock_ghz) / mma_eff, fit.inst_min_ns
    )  # ns
    t_mma_kt = inner * t_inst * np.where(two, fit.two_cta_gain, 1.0)
    a_stage = ctm * kb * elem
    b_stage = ctn * kb * elem / cta_div
    # the `n` layout's B box is kb K-rows of ctn x elem bytes: a row narrower than a 128-byte line still costs the line (and the TMA
    # engine a request), so the narrow tiles' B traffic is counted at line granularity times a fitted residual penalty
    b_row = ctn * elem / cta_div
    b_eff = np.where(
        (b_major == "n") & (b_row < 128), kb * 128.0 * fit.nmajor_pen, b_stage
    )
    out_b = np.where(ws, 4.0, float(out_bytes))
    ep = np.array(
        [
            _epi_tile(int(m_), int(n_), bool(t_), int(o_))
            for m_, n_, t_, o_ in zip(ctm, ctn, two, out_b, strict=False)
        ],
        dtype=np.float64,
    )
    c_stage = ep[:, 0] * ep[:, 1] * out_b
    stages = np.maximum(
        1.0, np.floor((arch.smem_bytes - 1024 - 2 * c_stage) / (a_stage + b_stage))
    )
    t_ingress_kt = (
        a_stage + b_eff
    ) / fit.ingress_gbps  # bytes / (GB/s) = ns: every byte landing in the ring
    # the smem bandwidth is shared by the TMA's writes (the stage) and the MMA's operand reads: per instruction the unit reads the CTA's
    # A rows and its share of the B columns (the pair's second CTA holds the other half), `inner` instructions per k-tile. A wide 1-CTA
    # tile (128x256: 12 KB per instruction + 48 KB of stage per k-tile) is bound here while the 2-CTA pair of the same CTA tile (8 KB +
    # 32 KB) is MMA-bound: the mechanism behind the pair's 1.3x at M = 32768 that the ingress and L2 terms alone could not order
    mma_reads = inner * inst_k * elem * (ctm + ctn / cta_div)
    t_smem_kt = (a_stage + b_eff + mma_reads) / fit.smem_gbps
    t_fetch_kt = (
        a_stage / cn + b_eff / pairs_m
    ) / fit.fetch_gbps  # the bytes this CTA's TMA unit requests (multicast shares)
    t_lat_kt = fit.latency_us * 1e3 / stages
    ktile = np.where(two, fit.ktile_2cta_ns, fit.ktile_ns)
    # the A box's rows past M are zero-filled by the issuing CTA's TMA engine at a modest rate (the decode measurements: a 64-row box with one
    # valid row costs ~8 KB of fill per k-tile; the N-cluster splits the box among its CTAs; the pair's second CTA fills its whole box)
    oob_rows = np.where(
        two,
        np.where(ctm >= M, ctm, np.maximum(0.0, 2 * ctm - M)),
        np.maximum(0.0, ctm - M),
    )
    oob_rows = np.minimum(oob_rows, ctm)
    t_oob_kt = oob_rows * 128.0 / cn / fit.oob_gbps
    # the TMA path pays its per-k-tile issue, zero-fill and hand-shake cost serially with its fetch; the MMA path hides it; a cluster of
    # four or more CTAs pays its fan-in hand-shake per k-tile on top (measured +20-50 % at the compute-bound shapes against the pair)
    t_kt = np.maximum.reduce(
        [t_mma_kt, t_fetch_kt + t_oob_kt + ktile, t_ingress_kt, t_smem_kt, t_lat_kt]
    ) + np.where(cl >= 4, fit.cl4_ns, 0.0)
    t_main = k_tiles * t_kt / 1e3  # us per tile per CTA

    # epilogue, accumulator stages, the in-CTA fold's passes
    t_epi = ctm * ctn * out_b / fit.epi_gbps / 1e3 + fit.tile_us
    one_stage = a["cc"] & split & ((3 * ctn > arch.tmem_cols) | (ctm == 64))
    t_fold = np.where(
        a["cc"] & split, nchunk_cta * ctm * ctn * 8.0 / fit.epi_gbps / 1e3, 0.0
    )
    t_tile = np.where(
        one_stage, t_main + t_epi + t_fold, np.maximum(t_main, t_epi + t_fold)
    )
    t_cta = fit.fill_us + waves * t_tile + t_epi

    # aggregate traffic: L2 -> SM (multicast shares the reads), HBM (the raster's footprint rule)
    l2_bytes = units * cl * k_tiles * (a_stage / cn + b_stage / pairs_m)
    rows_cl = a["tm"] * pairs_m
    cols_cl = ctn * cn
    A_total = M * K * elem
    B_total = N * K * elem
    l2_eff = fit.l2_frac * arch.l2_bytes
    K_unit = (
        Kc * nchunk_cta
    )  # the K extent one unit streams (all chunks for the in-CTA fold, one for the workspace split)
    maj_cl = np.where(
        rm, m_cl, n_cl
    )  # the long axis walked inside a group (all of it per group)
    min_cl = np.where(rm, n_cl, m_cl)  # the minor axis: s columns of it form a group
    walk_dim = np.where(
        rm, rows_cl, cols_cl
    )  # rows / columns of one cluster tile along each axis
    fix_dim = np.where(rm, cols_cl, rows_cl)
    G = s * maj_cl  # units per group
    n_groups = np.ceil(min_cl / s)
    waves_per_group = np.maximum(1.0, G / P)
    walked_total = np.where(
        rm, A_total, B_total
    )  # the operand walked along the major axis (A for raster m)
    fixed_total = np.where(
        rm, B_total, A_total
    )  # the operand a group holds fixed (B for raster m)
    walk_fp = (
        np.minimum(maj_cl, np.ceil(P / s)) * walk_dim * K_unit * elem
    )  # the walked operand's footprint of one wave
    group_fp = (
        np.minimum(s, min_cl) * fix_dim * K_unit * elem
    )  # one group's fixed operand
    # the walked operand is streamed once per group pass; the groups a wave spans share the pass, so passes = min(groups, waves); a
    # pass after the first misses the L2 for the part of (operand + group) that does not fit (the cyclic re-use distance is the operand)
    passes = np.minimum(n_groups, waves)
    miss_frac = np.clip(
        (walked_total + group_fp - l2_eff) / np.maximum(walked_total, 1.0), 0.0, 1.0
    )
    walked_hbm = walked_total * (1.0 + (passes - 1.0) * miss_frac)
    # a group's fixed operand is read when the group starts and stays resident across the group's waves when it fits next to the
    # wave's walked footprint; otherwise every wave of the group reads it again
    fixed_hbm = fixed_total * np.where(
        group_fp + walk_fp <= l2_eff, 1.0, waves_per_group
    )
    # the workspace split: the chunk is the scheduler's outermost (batch) coordinate, so the grid walks chunk 0's tiles, then chunk 1's
    # ... -- each chunk a GEMM of its own K slice (K_unit = Kc above: the footprint rule already prices one chunk's passes and groups),
    # and the C chunks' slices are distinct data, so the operand traffic is the per-chunk figure times C (monotone in the SM count,
    # unlike a rule over the wave's footprint across chunks, which predicted a 204-SM part slower than a 148-SM one at a split shape)
    units_c = (
        m_cl * n_cl
    )  # one chunk's units: the passes and groups of one chunk's walk
    waves_c = np.ceil(units_c / P)
    passes_c = np.minimum(n_groups, waves_c)
    walked_c = walked_total / chunks  # one chunk's slice of the walked operand
    fixed_c = fixed_total / chunks
    miss_c = np.clip(
        (walked_c + group_fp - l2_eff) / np.maximum(walked_c, 1.0), 0.0, 1.0
    )
    walked_ws = chunks * walked_c * (1.0 + (passes_c - 1.0) * miss_c)
    fixed_ws = (
        chunks
        * fixed_c
        * np.where(group_fp + walk_fp <= l2_eff, 1.0, np.maximum(1.0, G / P))
    )
    walked_hbm = np.where(ws, walked_ws, walked_hbm)
    fixed_hbm = np.where(ws, fixed_ws, fixed_hbm)
    out_hbm = M * N * out_bytes
    ws_hbm = np.where(ws, 2.0 * chunks * M * N * 4.0, 0.0)  # partials out and back
    hbm_bytes = walked_hbm + fixed_hbm + out_hbm + ws_hbm
    hbm_bw = fit.hbm_eff * arch.hbm_tbps * 1e6  # bytes / us
    t_hbm = hbm_bytes / hbm_bw
    t_l2 = (l2_bytes + fit.miss_cost * hbm_bytes) / (
        fit.l2_tbps * 1e6
    )  # misses cost the L2 their fill and their read
    # the hot-line bound of small M: every A box fetches the tile's VALID rows (one 128-byte line per row per k-tile; the zero-filled
    # rows cost nothing in L2), and with M rows the whole machine's A requests land on M lines -- M home slices -- per k-tile, so at
    # M = 1 every CTA's A fetch is served by one slice (the decode measurements: time ~ number of tiles for the narrow tiles; the A multicast
    # of the N-clusters divides the requests, the M-clusters' B multicast does not)
    a_rows_valid = np.minimum(M, ctm)
    a_req_bytes = units * (cl / cn) * k_tiles * a_rows_valid * 128.0
    hot_rate = np.minimum(
        fit.hot_gbps * a_rows_valid * np.where(ws, chunks, 1.0), fit.l2_tbps * 1e3
    )  # GB/s
    t_hot = a_req_bytes / (hot_rate * 1e3)  # us
    bounds = np.stack([t_cta, t_hbm, t_l2, t_hot])
    t_body = np.power(
        np.sum(np.power(bounds, fit.softmax_p), axis=0), 1.0 / fit.softmax_p
    )  # a soft max: close bounds interfere
    t_kernel = (fit.launch_us + t_body) * (1.0 + fit.cluster_pen * (cl - 1.0))
    t_fixup = np.where(
        ws, fit.fixup_us + (chunks * M * N * 4.0 + out_hbm) / hbm_bw, 0.0
    )
    t = t_kernel + t_fixup
    if not parts:
        return t
    return t, dict(
        units=units,
        waves=waves,
        stages=stages,
        k_tiles=k_tiles,
        t_mma_kt=t_mma_kt,
        t_ingress_kt=t_ingress_kt,
        t_smem_kt=t_smem_kt,
        t_fetch_kt=t_fetch_kt,
        t_oob_kt=t_oob_kt,
        t_lat_kt=t_lat_kt,
        t_kt=t_kt,
        t_main=t_main,
        t_epi=t_epi,
        t_cta=t_cta,
        t_hbm=t_hbm,
        t_l2=t_l2,
        t_hot=t_hot,
        hbm_bytes=hbm_bytes,
        l2_bytes=l2_bytes,
        t_fixup=t_fixup,
    )


def predict_many(
    recs: Sequence[Knobs],
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    out_bytes: int = 2,
    arch: Optional[Arch] = None,
    fit: Optional[Fit] = None,
    parts: bool = False,
):
    """Predicted device time (us) of every record of ``recs`` at (M, N, K); ``parts=True`` also returns the terms."""
    arch = arch or arch_for()
    fit = fit or fit_for(arch)
    return predict_arrays(
        knob_arrays(recs), M, N, K, chunks, b_major, out_bytes, arch, fit, parts
    )


def predict(
    k: Knobs,
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    out_bytes: int = 2,
    arch: Optional[Arch] = None,
    fit: Optional[Fit] = None,
) -> float:
    """Predicted device time (us) of one record at one shape."""
    return float(predict_many([k], M, N, K, chunks, b_major, out_bytes, arch, fit)[0])


def explain(
    k: Knobs,
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    out_bytes: int = 2,
    arch: Optional[Arch] = None,
    fit: Optional[Fit] = None,
) -> Dict[str, float]:
    """The predicted time and every term of one record (for the docs and the tests)."""
    t, p = predict_many([k], M, N, K, chunks, b_major, out_bytes, arch, fit, parts=True)
    return dict(us=float(t[0]), **{n: float(v[0]) for n, v in p.items()})


# ----------------------------------------------------------------------------- the dispatcher's entry

DECODE_M = 64  # up to this many rows the candidate set is the decode-sane subset below


def decode_sane(k: Knobs, M: int) -> bool:
    """The dispatcher's candidate guard at M <= DECODE_M (a heuristic, not a space change): the measurements found, at every decode
    point, the best record among the 1-CTA tiles of N >= 32 in an N-cluster (1x1 / 1x2 / 1x4) at swizzle 1; the 2-CTA pair with one
    valid row, the clusters along M, the 8 / 16-wide tiles and the swizzled rasters were 1.5-10x slower (the hot A lines and the
    per-tile costs, whose magnitudes the model reproduces only qualitatively). Records outside the guard stay members of the space."""
    if M > DECODE_M:
        return True
    return (
        (not k.use_2cta_instrs)
        and k.tile_n >= 32
        and k.cluster_m == 1
        and k.swizzle_size == 1
    )


@functools.lru_cache(maxsize=4096)
def rank(
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    out_bytes: int = 2,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    arch_name: Optional[str] = None,
    topk: Optional[int] = None,
) -> Tuple[Tuple[Knobs, float], ...]:
    """The records of the class's enumerated space (`generator.enumerate_space`: the knob product the body accepts for the shape and
    layout; at M <= DECODE_M the `decode_sane` subset) ordered by predicted time, best first; ``topk`` truncates. Cached per argument
    tuple (the dispatcher calls it per mm)."""
    from .generator import enumerate_space

    arch = arch_for(sm_count, arch_name, family)
    space = enumerate_space(M, N, K, chunks=chunks, b_major=b_major)
    guarded = [k for k in space if decode_sane(k, M)]
    space = guarded or space
    if not space:
        return ()  # nothing implements the shape in this layout / class: the caller decides
    t = predict_many(space, M, N, K, chunks, b_major, out_bytes, arch)
    order = np.argsort(t, kind="stable")
    out = tuple((space[i], float(t[i])) for i in order)
    return out[:topk] if topk else out


def best_record(
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    out_bytes: int = 2,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
) -> Knobs:
    """The model's pick: the argmin of `rank` (a member of the enumerated space by construction)."""
    return rank(M, N, K, chunks, b_major, out_bytes, sm_count, family)[0][0]


def fit_to_dict(fit: Fit) -> Dict[str, float]:
    return asdict(fit)
