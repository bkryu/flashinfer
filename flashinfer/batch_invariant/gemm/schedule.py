"""Dispatch of the bf16 GEMM: the chunk rule (a function of the weight shape only) and the cost-model dispatcher.

Every record of the configuration space computes the class (R3), so the choice here never reaches the bits; it only
moves time. The choice is the physics-informed cost model's argmin over the space (``cost_model.rank``), corrected by
the measured override cells (one file per measured part) where the measurements recorded the model's pick behind the
best record by more than the margin. ``rank_topk`` is the autotuner's tactic list.
"""

from __future__ import annotations

import glob
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


# The measured override cells: (part name, b_major, N, K, grid M) -> record, the cells of the measured grids where the best record
# beat the model's pick by more than the margin, confirmed by an interleaved ABBA. A cell applies to its EXACT (N, K) pair, on the
# part it was measured on (`cost_model.Arch.name`: an emulated SM count or another part gets the model alone), and to the M that
# round up to its grid step: a shape takes the cell of the first step >= its M (M = 3000 takes the 3072 cell; M = 5 the 8 cell);
# above 32768 there are no cells. One file per measured part: ``model_overrides.json`` for sm_107 (Rubin NVL8),
# ``model_overrides_<part>.json`` otherwise (``model_overrides_sm_100.json``: the B200). Rows: [format, b_major, N, K, M bound,
# record name]; edited only from the derivation of the measurements.
_GRID_MS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    32768,
)
_OVERRIDES_DIR = os.path.dirname(__file__)
_OVERRIDES_PATH = os.path.join(_OVERRIDES_DIR, "model_overrides.json")


def overrides_path(part: str) -> str:
    """The override file of a measured part (see the note on `_GRID_MS`)."""
    if part == "sm_107":
        return _OVERRIDES_PATH
    return os.path.join(_OVERRIDES_DIR, f"model_overrides_{part}.json")


def _load_overrides() -> dict:
    files = {"sm_107": _OVERRIDES_PATH}
    for path in glob.glob(os.path.join(_OVERRIDES_DIR, "model_overrides_*.json")):
        files[os.path.basename(path)[len("model_overrides_") : -len(".json")]] = path
    out: dict = {}
    for part, path in files.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for fmt, bl, n, k, m_max, name in json.load(f):
                if fmt == "bf16":
                    out[
                        (
                            part,
                            bl,
                            int(n),
                            int(k),
                            None if m_max is None else int(m_max),
                        )
                    ] = _rec(name)
    return out


_MODEL_OVERRIDES: dict = _load_overrides()


def override_record(
    M: int,
    N: int,
    K: int,
    chunks: int,
    b_major: str,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
) -> Optional[GemmConfig]:
    """The measured override of the (part, layout, N, K) cell at this M's grid step, if the measurements recorded one on this
    part and it implements the shape; else None."""
    from .blocks.operands import can_implement
    from .cost_model import arch_for

    grid_m = next((g for g in _GRID_MS if g >= M), None)
    if grid_m is None:
        return None
    rec = _MODEL_OVERRIDES.get(
        (arch_for(sm_count, family=family).name, b_major, N, K, grid_m)
    )
    if rec is not None and can_implement(rec, M, N, K, b_major, chunks):
        return rec
    return None


def rank_topk(
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    sm_count: Optional[int] = None,
    k: int = 10,
    family: Optional[str] = None,
) -> List[GemmConfig]:
    """The dispatcher's ranked list (what FlashInfer's autotuner profiles under ``batch_invariant=True``): the measured override
    of the cell first when there is one, then `cost_model.rank`'s order (the model's argmin first), distinct records, at most
    ``k``; ``default_config(...) == rank_topk(...)[0]``. Every entry is a member of the class's enumerated space, so an autotuner
    may pick among them freely without changing a single bit. ``sm_count`` and ``family`` name the part (the operand device's).
    Empty when nothing implements the shape."""
    from .cost_model import rank

    out: List[GemmConfig] = []
    ov = override_record(M, N, K, chunks, b_major, sm_count, family)
    if ov is not None:
        out.append(ov)
    for rec, _t in rank(M, N, K, chunks, b_major, 2, sm_count, family):
        if len(out) >= k:
            break
        if rec not in out:
            out.append(rec)
    return out


def default_config(
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
) -> GemmConfig:
    """The dispatcher's pick: `rank_topk(...)[0]`, the measured override of the (part, layout, N, K) cell at this M's grid step if
    one was recorded on this part, else the cost model's argmin over the class's space for the shape, the layout and the part.
    May depend on anything: every record is canonical."""
    top = rank_topk(M, N, K, chunks, b_major, sm_count, 1, family)
    if not top:
        raise ValueError(
            f"no record of the class implements {(M, N, K)} in layout {b_major} with {chunks} chunks"
        )
    return top[0]
