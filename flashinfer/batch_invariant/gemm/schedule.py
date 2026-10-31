"""Dispatch of the bf16 GEMM: the chunk rule (a function of the weight shape only) and the cost-model dispatcher.

Every record of the configuration space computes the class (R3), so the choice here never reaches the bits; it only
moves time. The choice is the physics-informed cost model's argmin over the space (``cost_model.rank``), corrected by
the measured override cells of ``model_overrides.json`` where the search recorded the model's pick behind the best
record by more than the margin. ``rank_topk`` is the autotuner's tactic list.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from .generator import CLUSTERS, SWIZZLES, Knobs, enumerate_space  # noqa: F401  (re-exported)
from .spec import MMA_K

GemmConfig = Knobs


def all_valid_configs(
    M: int, N: int, K: int, chunks: int = 1, b_major: str = "n"
) -> List[GemmConfig]:
    """The configuration space for this shape and class: the generator's knob product filtered by the kernel's own
    ``can_implement``. This is what the full-space GPU test covers."""
    return enumerate_space(M, N, K, chunks=chunks, b_major=b_major)


SPLIT_MIN_CTAS = 256  # below this many N tiles at 32 wide, a K split is considered
SPLIT_MAX_CHUNKS = 128
SPLIT_MIN_KC = 256  # elements of K per chunk at least


def chunks_for(N: int, K: int) -> int:
    """Fixed-chunk split count of a weight shape. A function of (N, K) ONLY: the same bits for every M and on every
    part. Budget: the fp32 partial round trip (8 * C * M * N bytes) stays below a quarter of the A traffic
    (2 * M * K bytes), i.e. C <= K / (16 N); K / C must be a whole number of 16-wide MMA steps (so every chunk is
    exactly a ``kchunk`` fold of the specification) and >= SPLIT_MIN_KC; and no split at all when N alone gives
    enough CTAs. Mirrored by ``BIK.GEMM.chunksFor`` in Lean (checked by the generated conformance cases)."""
    if -(-N // 32) >= SPLIT_MIN_CTAS:
        return 1
    budget = K // (16 * N)
    C = 1
    while (
        min(budget, SPLIT_MAX_CHUNKS) >= C * 2
        and K % (MMA_K * C * 2) == 0
        and K // (C * 2) >= SPLIT_MIN_KC
    ):
        C *= 2
    return C


def _pow2ceil(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _rec(name: str) -> GemmConfig:
    """A record from its ``describe()`` name (``mma256x192_cl2x1_2cta_tma_rm_sw1[_cc]``): the overrides' notation."""
    parts = name.split("_")
    tm, tn = map(int, parts[0][3:].split("x"))
    cm, cn = map(int, parts[1][2:].split("x"))
    return GemmConfig(
        (tm, tn),
        (cm, cn),
        parts[2] == "2cta",
        True,
        parts[4][1],
        int(parts[5][2:]),
        0,
        "cc" in parts[6:],
    )


#: (b_major, N, K, grid M) -> record: the cells of the measured grid where the best record beat the model's pick by more than
#: the margin. A cell applies to its EXACT (N, K) pair and to the M that round up to its grid M (`_GRID_MS`); every other shape
#: is the model's. Edited only from the derivation script's output.
_GRID_MS = (1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
_OVERRIDES_PATH = os.path.join(os.path.dirname(__file__), "model_overrides.json")


def _load_overrides() -> dict:
    """``model_overrides.json``: a list of [format, b_major, N, K, M bound, record name] rows."""
    out: dict = {}
    if os.path.exists(_OVERRIDES_PATH):
        with open(_OVERRIDES_PATH) as f:
            for fmt, bl, n, k, m_max, name in json.load(f):
                if fmt == "bf16":
                    out[(bl, int(n), int(k), None if m_max is None else int(m_max))] = (
                        _rec(name)
                    )
    return out


_MODEL_OVERRIDES: dict = _load_overrides()


def rank_topk(
    M: int,
    N: int,
    K: int,
    chunks: int,
    b_major: str,
    sm_count: Optional[int] = None,
    k: int = 10,
) -> List[GemmConfig]:
    """The cost model's best ``k`` records for the shape, best first, with the measured override (if any) first. Every
    record computes the class, so an autotuner may pick among them freely without changing a single bit."""
    from .blocks.operands import can_implement
    from .cost_model import rank

    out: List[GemmConfig] = []
    grid_m = next((g for g in _GRID_MS if g >= M), None)
    if grid_m is not None:
        rec = _MODEL_OVERRIDES.get((b_major, N, K, grid_m))
        if rec is not None and can_implement(rec, M, N, K, b_major, chunks):
            out.append(rec)
    for cfg, _t in rank(M, N, K, chunks, b_major, 2, sm_count):
        if cfg not in out:
            out.append(cfg)
        if len(out) >= k:
            break
    return out


def default_config(
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    sm_count: Optional[int] = None,
) -> GemmConfig:
    """The dispatcher's pick: the measured override of the (layout, N, K) cell at this M's grid step if one was recorded,
    else the cost model's argmin over the class's space for the shape, the layout and the part (``sm_count``). May
    depend on anything: every record is canonical."""
    return rank_topk(M, N, K, chunks, b_major, sm_count, k=1)[0]
