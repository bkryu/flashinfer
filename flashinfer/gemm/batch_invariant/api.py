# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entry of the bf16 GEMM class: ``mm_bf16_batch_invariant`` is what :func:`flashinfer.mm_bf16` runs
under ``batch_invariant=True``. C = A @ B (+ bias) in bf16 with fp32 accumulation; A is (M, K) row-major, B is (K, N)
either row-major or the transpose of a row-major (N, K) tensor (FlashInfer's documented "column-major" weight), the
optional bias is a bf16 (N,) vector added once in fp32 to the reduced accumulator before the single rounding, C is
(M, N) row-major bf16. Every unsupported input raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`;
nothing falls back."""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

from ...batch_invariant.support import check_device, unsupported
from ...jit.core import logger
from .generator import Knobs
from .schedule import chunks_for, default_config
from .spec import MMA_K

CLASS = "gemm.bf16"
#: The compute capabilities the class is stated and tested for: SM100 (B200), SM103 (B300), SM107 (Rubin).
MM_BF16_ARCHS = ((10, 0), (10, 3), (10, 7))
_ALIGN = 8  # bf16 elements per 16 bytes
_ALIGN_BYTES = 16  # the operand base alignment the kernels assume (TMA)


def _layout_of_b(b: torch.Tensor, K: int, N: int) -> str:
    if b.stride(1) == 1 and b.stride(0) == N:
        return "n"  # (K, N) row-major
    if b.stride(0) == 1 and b.stride(1) == K:
        return "k"  # the transpose of a row-major (N, K) weight
    raise unsupported(
        CLASS,
        f"B with strides {tuple(b.stride())}",
        "B (K, N) row-major or the transpose of a row-major (N, K) tensor",
    )


def sm_count_of(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def part_of(device: torch.device) -> Tuple[int, str]:
    """The part the dispatcher models for ``device``: its SM count and architecture family (`cost_model.family_of`)."""
    from ...batch_invariant.support import compute_capability
    from .cost_model import family_of

    return sm_count_of(device), family_of(compute_capability(device))


@functools.cache
def _co_resident_clusters(device_index: int, cluster_size: int, sm_count: int) -> int:
    """How many clusters of ``cluster_size`` CTAs the device can host at once for a one-CTA-per-SM body: the CuTe DSL's
    GPC-aware occupancy query (what FlashInfer's CuTe-DSL GEMMs size their persistent grids with), after a CUDA context
    on the device has been made current (the query fails without one); ``sm_count // cluster_size`` if the query fails,
    which is logged."""
    import cutlass.utils as cutlass_utils

    try:
        torch.cuda.synchronize(device_index)  # a current context on the device
        with torch.cuda.device(device_index):
            n = int(
                cutlass_utils.HardwareInfo(
                    device_id=device_index
                ).get_max_active_clusters(cluster_size)
            )
        if n <= 0:
            raise RuntimeError(f"the occupancy query returned {n}")
        return n
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "batch_invariant: the co-resident cluster query failed on device %d (%s: %s); the persistent grid falls back to "
            "SMs // cluster size",
            device_index,
            type(e).__name__,
            e,
        )
        return sm_count // cluster_size


def persistent_clusters(device: torch.device, cfg: Knobs) -> int:
    """The persistent grid in clusters for ``cfg`` on ``device``: the device's co-resident cluster count for the knobs'
    cluster shape (cached per device and cluster size), so that the compiled grid is what the device can run at once;
    a launch parameter that decides which CTA computes which tile and nothing else (the bits never depend on it). A
    grid of ``SMs // size`` would leave its surplus clusters to run as a serial tail under the static persistent
    assignment whenever a GPC cannot host a whole cluster of every SM."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _co_resident_clusters(
        index, cfg.cluster_m * cfg.cluster_n, sm_count_of(device)
    )


def prepare_operands(
    a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, str, int, int, int]:
    """The class's typed checks and operand normalisation, shared by the direct entry and the autotuner entry: device
    and architecture, dtypes, 2-D shapes with matching K, 16-byte TMA alignment of K and N, A's row placement, B's layout
    (``"n"``: (K, N) row-major; ``"k"``: the transpose of a row-major (N, K) weight), the output's dtype / shape / layout
    (allocated when absent). Returns ``(a, out, b_major, M, N, K)``; raises :class:`BatchInvariantUnsupportedError`."""
    check_device(a.device, CLASS, MM_BF16_ARCHS)
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise unsupported(CLASS, f"dtypes ({a.dtype}, {b.dtype})", "bf16 A and B")
    if a.dim() != 2 or b.dim() != 2:
        raise unsupported(
            CLASS, f"{a.dim()}-D A and {b.dim()}-D B", "2-D operands (M, K) and (K, N)"
        )
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise unsupported(CLASS, f"A (M, {K}) with B ({K2}, N)", "matching K")
    if b.device != a.device:
        raise unsupported(
            CLASS, "operands on different devices", "A and B on one device"
        )
    if K == 0 or N == 0:
        raise unsupported(
            CLASS, f"K = {K}, N = {N}", "K > 0 and N > 0 (only M may be empty)"
        )
    if K % _ALIGN or N % _ALIGN:
        raise unsupported(
            CLASS,
            f"K = {K}, N = {N}",
            f"K and N multiples of {_ALIGN} (16-byte TMA rows)",
        )
    if a.stride(1) != 1 or a.stride(0) % _ALIGN or a.data_ptr() % _ALIGN_BYTES:
        # a placement, not numerics: the kernel reads 16-byte-aligned rows from a 16-byte-aligned base (a single
        # row's row stride is immaterial)
        a = (
            a.as_strided((1, K), (K, 1), a.storage_offset())
            if (M == 1 and a.stride(1) == 1 and a.data_ptr() % _ALIGN_BYTES == 0)
            else a.clone(memory_format=torch.contiguous_format)
        )
    b_major = _layout_of_b(b, K, N)
    if b.data_ptr() % _ALIGN_BYTES:
        raise unsupported(
            CLASS,
            "B at a base address that is not 16-byte aligned",
            "a 16-byte-aligned B",
        )
    if out is None:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    else:
        if out.dtype != torch.bfloat16 or tuple(out.shape) != (M, N):
            raise unsupported(
                CLASS,
                f"out {tuple(out.shape)} {out.dtype}",
                f"a bf16 (M, N) = ({M}, {N}) output",
            )
        if out.device != a.device:
            raise unsupported(
                CLASS, "out on another device", "the output on A's device"
            )
        if out.data_ptr() % _ALIGN_BYTES:
            raise unsupported(
                CLASS,
                "out at a base address that is not 16-byte aligned",
                "a 16-byte-aligned output",
            )
        if out.stride(1) != 1 or (M > 1 and out.stride(0) != N):
            raise unsupported(
                CLASS,
                f"out with strides {tuple(out.stride())}",
                "a row-major contiguous output",
            )
        if M == 1 and out.stride(0) != N:
            out = out.as_strided(
                (1, N), (N, 1), out.storage_offset()
            )  # the same elements, an aligned row stride
    return a, out, b_major, M, N, K


def prepare_bias(
    bias: Optional[torch.Tensor], N: int, a: torch.Tensor
) -> Optional[torch.Tensor]:
    """The class's typed checks of the bias (what the operator's backends accept): a bf16 vector of ``N`` elements on
    A's device, contiguous, at a 16-byte-aligned base; ``None`` is the unbiased chain. Raises
    :class:`BatchInvariantUnsupportedError`; never copies (a bias is an operand the kernel reads as given)."""
    if bias is None:
        return None
    if bias.dtype != torch.bfloat16:
        raise unsupported(CLASS, f"a {bias.dtype} bias", "a bf16 bias")
    if bias.dim() != 1 or bias.shape[0] != N:
        raise unsupported(
            CLASS, f"a bias of shape {tuple(bias.shape)}", f"a bias of shape ({N},)"
        )
    if bias.device != a.device:
        raise unsupported(CLASS, "a bias on another device", "the bias on A's device")
    if bias.stride(0) != 1:
        raise unsupported(
            CLASS, f"a bias with stride {bias.stride(0)}", "a contiguous bias"
        )
    if bias.data_ptr() % _ALIGN_BYTES:
        raise unsupported(
            CLASS,
            "a bias at a base address that is not 16-byte aligned",
            "a 16-byte-aligned bias",
        )
    return bias


def mm_bf16_batch_invariant(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    bias: Optional[torch.Tensor] = None,
    config: Optional[Knobs] = None,
    chunks: Optional[int] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``C = A @ B`` (``+ bias`` when given) by the class ``gemm.bf16``: ``kfold`` when the weight shape's chunk rule
    gives 1, else ``kchunk[C]`` with ``C = chunks_for(N, K)``, a function of the weight shape only, so a row's bits never
    depend on the batch. ``bias`` (bf16, (N,)) selects the biased chain: ``cvt(add(fold, bias[n]))``, the add once in
    fp32 after the whole fold (in the epilogue, or in the fixup for the workspace split). ``config`` forces a
    configuration (a member of the enumerated space: the bits never depend on it); ``chunks`` forces the class's
    chunk count (tests); ``workspace`` (fp32, at least ``chunks * M * N`` elements) is the split class's partials
    buffer, otherwise allocated per call from the caching allocator (stream-ordered)."""
    a, out, b_major, M, N, K = prepare_operands(a, b, out)
    bias = prepare_bias(bias, N, a)
    if M == 0:
        return out  # no rows: nothing to launch
    C = chunks if chunks is not None else chunks_for(N, K)
    if C > 1 and K % (MMA_K * C):
        raise unsupported(
            CLASS,
            f"chunks = {C} at K = {K}",
            "each K chunk a whole number of 16-wide MMA steps",
        )
    if config is None:
        config = default_config(M, N, K, C, b_major, *part_of(a.device))
    from .blocks.compile import get_compiled_gemm
    from .fixup import MAX_GRID_Y, get_fixup

    direct = C == 1 or config.cta_chunks
    kernel = get_compiled_gemm(
        config,
        b_major,
        C,
        persistent_clusters(a.device, config),
        bias is not None and direct,
    )
    # the kernel sees 3-D (m, k, l) / (n, k, l) / (m, n, l) tensors: l = the K chunk (1 for the unsplit class and the in-CTA
    # fold's output); every stride handed over is a multiple of the declared 16-byte alignment, the size-1 l stride included
    Kc = K // C
    a3 = a.as_strided((M, Kc, C), (a.stride(0), 1, Kc))
    bt = b.t()  # (N, K) view with strides (1, N) or (K, 1)
    b3 = bt.as_strided((N, Kc, C), (bt.stride(0), bt.stride(1), Kc * bt.stride(1)))
    if direct:
        c3 = out.as_strided((M, N, 1), (out.stride(0), 1, N))
        if bias is None:
            kernel(a3, b3, c3)
        else:
            kernel(a3, b3, c3, bias)  # the biased chain in the epilogue
        return out
    # the fixed-chunk split with the partials in an fp32 workspace, folded in chunk order by the fixup kernel
    if workspace is None:
        ws = torch.empty(C, M, N, dtype=torch.float32, device=a.device)
    else:
        if (
            workspace.dtype != torch.float32
            or workspace.device != a.device
            or workspace.numel() < C * M * N
            or not workspace.is_contiguous()
            or workspace.data_ptr() % 16
        ):
            raise unsupported(
                CLASS,
                "the given workspace",
                f"a contiguous, 16-byte-aligned fp32 tensor on A's device of at least {C * M * N} elements",
            )
        ws = workspace.reshape(-1)[: C * M * N].view(C, M, N)
    ws = ws.permute(1, 2, 0)  # chunk-major storage (C, M, N) viewed as (M, N, C)
    kernel(a3, b3, ws)  # the bias-free partials
    rows = min(M, MAX_GRID_Y)
    if bias is None:
        get_fixup(C, "bf16")(ws, out, M, N, rows)
    else:
        get_fixup(C, "bf16", bias=True)(
            ws, out, bias, M, N, rows
        )  # the chunk fold, then the biased chain
    return out
