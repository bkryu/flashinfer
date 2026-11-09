# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 MoE classes ``moe.nvfp4.<act>`` (W4A4): NVFP4 expert weights with one per-tensor global scale per expert and
GEMM, per-token NVFP4 activations, on the bf16 classes' routing, permutation and combine (:mod:`.api`).

Pipeline: the per-token NVFP4 quantization of ``x`` (the class ``nvfp4.quant.pertoken`` of
:mod:`flashinfer.batch_invariant.quantization`, or the caller's quantized triple ``(x4, x_sf, x_pts)``) -> the deterministic
permutation of the expanded rows (expert order, then token-slot order) with its scale factors placed at 128-row atoms ->
the grouped block-scaled GEMM1 with the W4A4 chain (``cvt_bf16(act(mul(g, s)) * mul(u, s))``, ``s = mul(alpha1[e],
pts[row])``) -> the per-token quantization of the materialised bf16 intermediate (the same class, its second pass) -> the
grouped block-scaled GEMM2 (``cvt_bf16(mul(acc, mul(alpha2[e], pts2[row])))``) -> the slot-order combine. Every output bit
is a function of the token's own row, its ``k`` experts and routing weights and the expert weights (the specification is
:mod:`.spec_nvfp4`; the Lean statements the ``MoENvfp4`` module of the mode).

Two packings of the expanded rows are placements of the same bits (``path``): ``"dense"`` (the default: the fp4 rows back
to back, the scale factors and per-token scales at the 128-row-padded positions, a dense output; the intermediate quantized
over the live rows and its scales re-tiled to the padded positions) and ``"padded"`` (every group's rows start at a 128-row
atom; the padding rows are zero-filled, computed and never stored).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from ...batch_invariant.quantization.nvfp4_schedule import padded_sf_cols, sf_numel
from ...batch_invariant.support import BatchInvariantUnsupportedError, unsupported
from ...gemm.batch_invariant.api import part_of, persistent_clusters, sm_count_of
from ...gemm.batch_invariant.generator import Knobs
from ...gemm.batch_invariant.schedule import chunks_for
from ...gemm.batch_invariant.spec import NVFP4_FMTS
from . import spec as S
from .api import interleave_fc1, permute_plan, route  # noqa: F401  (route re-exported for the class's callers)
from .blocks.epilogue_nvfp4 import W4A4Chain
from .spec_nvfp4 import CLASS, E2M1_MAX, E4M3_MAX, TE_GS_INV, expert_classes

#: 16-byte rows of the packed fp4 operands (the TMA alignment) and whole 64-element scale-factor column groups.
ALIGN_NVFP4 = 32
MAX_TOP_K = 16
PATHS = ("dense", "padded")
#: Rows past the live rows in the dense packing's A buffers (and past the padded rows in its scale-factor buffers) so the
#: last tiles' TMA boxes stay in bounds: a box past the end is zero-filled by the engine at a cost. The tail rows are never
#: stored. A placement.
DENSE_TAIL = 256
# the intermediate's scale factors reach the padded positions by the row-mapped quantizer (one launch) or by the quantizer
# and a re-tiling move (two launches; the same bytes): the two spellings of one placement
INTERMEDIATE_RETILE = "mapped"
RETILES = ("mapped", "scatter")
# a bf16 input reaches the grouped layout by the quantizer and the fp4 scatter (two launches, the dense codes round-tripped)
# or by the quantizer storing into the grouped layout itself (one launch; the same bytes): the two spellings of one placement
INPUT_SCATTER = "fused"
INPUT_SCATTERS = ("fused", "scatter")
GROUP_ALIGN = 128  # the scale-factor atom: the group alignment of the scale factors
_OUT = {"bf16": torch.bfloat16, "f32": torch.float32}


def check_shape_nvfp4(
    H: Optional[int], I: int, E: int, k: Optional[int] = None, gated: bool = True
) -> None:
    """The complete acceptance rule of the class on ``(H, I, E, k)``: ``H`` and ``I`` positive multiples of 32 (16-byte rows
    of the packed fp4 operands and whole scale-factor column groups), ``1 <= E <= 1024`` (the routing kernel's widest bucket
    and the group tables), ``1 <= k <= min(E, 16)``, and the expert weight shapes unsplit by the dense class's chunk rule
    (``chunks_for(N1, H) == chunks_for(H, I) == 1``: the grouped classes take one fold per weight). Everything else about a
    shape is placement. ``H=None`` checks the rules the layer's configuration fixes (the hidden size enters at the first
    call, with the full rule). Raises :class:`BatchInvariantUnsupportedError`; the class never routes a shape elsewhere."""
    from .blocks.scheduler import MAX_E

    def bad(what: str, limit: str):
        raise unsupported(
            CLASS,
            f"{what} ("
            + (f"H={H}, " if H is not None else "")
            + f"I={I}, E={E}"
            + (f", k={k}" if k is not None else "")
            + ")",
            limit,
        )

    if I <= 0 or I % ALIGN_NVFP4 or (H is not None and (H <= 0 or H % ALIGN_NVFP4)):
        bad(
            f"hidden size {H} / intermediate size {I}"
            if H is not None
            else f"intermediate size {I}",
            f"positive multiples of {ALIGN_NVFP4} (16-byte rows of the quantized activations and weights)",
        )
    if not 1 <= E <= MAX_E:
        bad(f"{E} experts", f"1 <= num_experts <= {MAX_E}")
    if k is not None and not 1 <= k <= min(E, MAX_TOP_K):
        bad(f"top_k {k}", f"1 <= top_k <= min(num_experts, {MAX_TOP_K})")
    if H is None:
        return
    n1 = 2 * I if gated else I
    if chunks_for(n1, H, "nvfp4") != 1 or chunks_for(H, I, "nvfp4") != 1:
        bad(
            "the expert weight shapes",
            "unsplit expert weights (the dense class would K-split this shape)",
        )


def fp4_global_scale(w: torch.Tensor) -> torch.Tensor:
    """FlashInfer's per-tensor global scale of a weight: ``448 * 6 / amax(w)`` (a 1-element f32 tensor; a static property of
    the model, computed once at preparation)."""
    amax = (
        w.detach()
        .to(torch.float32)
        .abs()
        .amax()
        .clamp_min(torch.finfo(torch.float32).tiny)
    )
    return (E4M3_MAX * E2M1_MAX / amax).reshape(1).to(torch.float32)


@dataclass
class MoEWeightsNVFP4:
    """The W4A4 expert weights as the kernels consume them: ``w1 [E, N1, H/2]`` packed fp4 (``N1 = 2I``, gate / up
    interleaved, for a gated chain; ``I`` as given for an ungated one), ``w1_sf [E, sf_bytes(N1, H)]`` 128x4 swizzled e4m3
    scales per expert, ``w1_alpha [E]`` f32 = ``1 / global_scale`` (as given to the kernel: a multiplier), likewise ``w2 [E,
    H, I/2]``, ``w2_sf``, ``w2_alpha``; the GEMM1 chain ``act`` with its ``params``."""

    w1: torch.Tensor
    w1_sf: torch.Tensor
    w1_alpha: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_alpha: torch.Tensor
    hidden_size: int
    intermediate_size: int
    gated: bool = True
    act: str = "silu"
    params: Tuple[Optional[float], ...] = ()

    @staticmethod
    def prepare(
        fc1: torch.Tensor,
        fc2: torch.Tensor,
        interleaved: bool = False,
        gated: bool = True,
        act: Optional[str] = None,
        params: Sequence[Optional[float]] = (),
        global_scales: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> "MoEWeightsNVFP4":
        """Quantize bf16 experts (``fc1 [E, 2I, H]`` rows ``[up, gate]``, or ``[E, I, H]`` ungated; ``fc2 [E, H, I]``) with the
        batch-invariant per-tensor NVFP4 class, one global scale per expert and GEMM (``global_scales = (gs1 [E], gs2 [E])``,
        or FlashInfer's ``448 * 6 / amax``). The gate / up interleave is a row permutation applied before the quantization (a
        per-tensor global scale commutes with it). ``act`` defaults to the class's default chain of the gating."""
        from ...batch_invariant.quantization.nvfp4_api import nvfp4_quantize_rows

        act = act if act is not None else ("silu" if gated else "relu2")
        if act not in (S.ACTS if gated else S.UNGATED_ACTS):
            raise ValueError(f"no {'gated' if gated else 'ungated'} chain {act!r}")
        key: Tuple[Optional[float], ...] = S.params_key(params)
        if len(key) != len(S.ACT_PARAMS.get(act, ())):
            raise ValueError(
                f"chain {act!r} takes parameters {S.ACT_PARAMS.get(act, ())}, got {key!r}"
            )
        if (
            fc1.dtype != torch.bfloat16
            or fc2.dtype != torch.bfloat16
            or fc1.dim() != 3
            or fc2.dim() != 3
        ):
            raise TypeError(
                "the NVFP4 MoE classes take bf16 expert weights fc1 [E, N1, H], fc2 [E, H, I]"
            )
        E, N1, H = fc1.shape
        I = N1 // 2 if gated else N1
        if tuple(fc2.shape) != (E, H, I) or (gated and N1 % 2):
            raise ValueError(
                f"fc2 {tuple(fc2.shape)} does not match fc1 {tuple(fc1.shape)} (gated={gated})"
            )
        check_shape_nvfp4(H, I, E, gated=gated)
        w1i = fc1.contiguous() if (interleaved or not gated) else interleave_fc1(fc1)
        fc2 = fc2.contiguous()
        w1s, sf1s, a1s, w2s, sf2s, a2s = [], [], [], [], [], []
        for e in range(E):
            gs1 = (
                global_scales[0][e].reshape(1).float()
                if global_scales is not None
                else fp4_global_scale(w1i[e])
            )
            gs2 = (
                global_scales[1][e].reshape(1).float()
                if global_scales is not None
                else fp4_global_scale(fc2[e])
            )
            q1, sf1 = nvfp4_quantize_rows(w1i[e], gs1, "128x4")
            q2, sf2 = nvfp4_quantize_rows(fc2[e], gs2, "128x4")
            w1s.append(q1)
            sf1s.append(sf1.reshape(-1))
            a1s.append(
                torch.reciprocal(gs1.to(w1i.device))
            )  # div.rn: the alpha as mm_fp4 takes it (a multiplier)
            w2s.append(q2)
            sf2s.append(sf2.reshape(-1))
            a2s.append(torch.reciprocal(gs2.to(w1i.device)))
        return MoEWeightsNVFP4(
            torch.stack(w1s).contiguous(),
            torch.stack(sf1s).contiguous(),
            torch.cat(a1s).contiguous(),
            torch.stack(w2s).contiguous(),
            torch.stack(sf2s).contiguous(),
            torch.cat(a2s).contiguous(),
            H,
            I,
            gated,
            act,
            key,
        )

    @property
    def num_experts(self) -> int:
        return self.w1.shape[0]

    @property
    def n1(self) -> int:
        """GEMM1's weight rows: ``2I`` gated, ``I`` ungated."""
        return self.w1.shape[1]

    def classes(self, family: Optional[str]) -> Tuple[str, str]:
        """The NVFP4 instruction class of each GEMM by the dense rule on the expert weight shape and the architecture family."""
        return expert_classes(
            self.hidden_size, self.intermediate_size, family, self.gated
        )

    @property
    def chain1(self) -> W4A4Chain:
        """The GEMM1 chain of the class: the gated ``act`` on the scaled (gate, up) pair or the ungated ``act`` on the scaled
        accumulator, with the parameters; bf16 output."""
        return W4A4Chain(
            "glu" if self.gated else "act", self.act, "bf16", self.params, "w4a4"
        )

    def view(self) -> Dict[str, object]:
        """The weight view a :class:`flashinfer.fused_moe.MoEWeightPack` stores for this backend."""
        return {
            "gemm1_weights": self.w1,
            "gemm1_weights_scale": self.w1_sf,
            "gemm1_alpha": self.w1_alpha,
            "gemm2_weights": self.w2,
            "gemm2_weights_scale": self.w2_sf,
            "gemm2_alpha": self.w2_alpha,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "gated": self.gated,
            "act": self.act,
            "params": self.params,
        }

    @staticmethod
    def from_view(view: Dict[str, object]) -> "MoEWeightsNVFP4":
        gated = bool(view.get("gated", True))
        return MoEWeightsNVFP4(
            view["gemm1_weights"],  # type: ignore[arg-type]
            view["gemm1_weights_scale"],  # type: ignore[arg-type]
            view["gemm1_alpha"],  # type: ignore[arg-type]
            view["gemm2_weights"],  # type: ignore[arg-type]
            view["gemm2_weights_scale"],  # type: ignore[arg-type]
            view["gemm2_alpha"],  # type: ignore[arg-type]
            int(view["hidden_size"]),  # type: ignore[call-overload]
            int(view["intermediate_size"]),  # type: ignore[call-overload]
            gated,
            str(view.get("act", "silu" if gated else "relu2")),
            tuple(view.get("params", ())),  # type: ignore[arg-type]
        )


GEMM2_CHAIN = W4A4Chain("plain", "silu", "bf16", (), "w4a4")
F32_CHAIN = W4A4Chain(
    "plain", "silu", "f32", (), "none"
)  # the unscaled oracle: the raw fp32 fold


def grouped_gemm_nvfp4(
    a4: torch.Tensor,
    a_sf: torch.Tensor,
    a_pts: torch.Tensor,
    w4: torch.Tensor,
    w_sf: torch.Tensor,
    alpha: torch.Tensor,
    counts: torch.Tensor,
    config: Knobs,
    chain: W4A4Chain,
    in_fmt: str,
    out: Optional[torch.Tensor] = None,
    dense_a: bool = False,
) -> torch.Tensor:
    """``out[p] = chain(a4[p] @ w4[e(p)].T, alpha[e(p)], pts[p])`` on the expert segments ``counts`` (int32 ``[E]``, group
    order = expert order). Padded packing (default): ``a4 [rows, K/2]`` uint8 with every group's rows starting at a multiple
    of 128 (``rows`` a multiple of 128), ``a_sf`` its 128x4 swizzled scales, ``a_pts [rows]`` f32. Dense packing (``dense_a``):
    ``a4 [rows, K/2]`` the groups' rows back to back (any row count), ``a_sf`` and ``a_pts`` at the padded positions
    (``a_pts`` of the padded row count), ``out [rows]`` dense. ``w4 [E, N, K/2]`` uint8, ``w_sf [E, sf_bytes(N, K)]``,
    ``alpha [E]`` f32. ``chain``: the W4A4 GEMM1 / GEMM2 chains or the unscaled oracle (``scale == "none"``: ``alpha`` and
    ``pts`` ignored). The record selects the body: a dense record runs the padded body on either packing, a swapped record
    (``config.swap_ab``: 128 weight rows x 8 / 16 / 32 tokens, the weights on the MMA's M side, :mod:`.blocks.swap_nvfp4`) runs
    the swapped decode body on the dense packing (``a4`` carries at least the token tile's rows past the live rows: the op's
    ``DENSE_TAIL``). The persistent grid is a launch argument: the device's co-resident cluster capacity for the record."""
    from .blocks.compile_nvfp4 import (
        can_implement_grouped_nvfp4,
        can_implement_swapped_nvfp4,
        get_compiled_grouped_nvfp4,
        get_compiled_swapped_nvfp4,
    )

    if (
        a4.dtype != torch.uint8
        or w4.dtype != torch.uint8
        or a4.dim() != 2
        or w4.dim() != 3
    ):
        raise TypeError(
            "grouped_gemm_nvfp4 takes packed-fp4 uint8 a4 [rows, K/2] and w4 [E, N, K/2]"
        )
    rows, Kh = a4.shape
    E, N, Kh2 = w4.shape
    K = 2 * Kh
    if Kh != Kh2 or not a4.is_contiguous() or not w4.is_contiguous():
        raise ValueError("grouped_gemm_nvfp4: contiguous operands of one K")
    rows_sf = a_pts.numel()
    if dense_a:
        if rows_sf % GROUP_ALIGN or rows_sf < rows - DENSE_TAIL:
            raise ValueError(
                "grouped_gemm_nvfp4: dense A takes the scale factors and row scales at the padded positions"
            )
    elif rows % GROUP_ALIGN or rows_sf != rows:
        raise ValueError(
            "grouped_gemm_nvfp4: the padded rows are a multiple of the 128-row scale-factor atom"
        )
    if (
        counts.dtype != torch.int32
        or tuple(counts.shape) != (E,)
        or not counts.is_contiguous()
    ):
        raise ValueError("grouped_gemm_nvfp4: counts must be a contiguous int32 [E]")
    if a_sf.dtype != torch.uint8 or a_sf.numel() < sf_numel(rows_sf, K):
        raise ValueError(
            "grouped_gemm_nvfp4: a_sf must hold the 128x4 swizzled scales of the (padded) rows"
        )
    if w_sf.dtype != torch.uint8 or w_sf.numel() < E * sf_numel(N, K):
        raise ValueError(
            "grouped_gemm_nvfp4: w_sf must hold E consecutive 128x4 swizzled scale buffers of (N, K)"
        )
    if (
        alpha.dtype != torch.float32
        or alpha.numel() != E
        or a_pts.dtype != torch.float32
    ):
        raise ValueError("grouped_gemm_nvfp4: alpha [E] and pts must be float32")
    if in_fmt not in NVFP4_FMTS or chunks_for(N, K, in_fmt) != 1:
        raise ValueError(
            "grouped_gemm_nvfp4: the grouped class takes unsplit expert weight shapes"
        )
    if config.swap_ab:
        if not dense_a:
            raise ValueError(
                "grouped_gemm_nvfp4: the swapped decode body reads the dense packing (dense_a=True)"
            )
        if not can_implement_swapped_nvfp4(config, N, K, E, chain, in_fmt):
            raise ValueError(
                f"grouped_gemm_nvfp4: swapped record {config.describe()} does not implement ({N}, {K}, {E}, {chain}, {in_fmt})"
            )
    elif not can_implement_grouped_nvfp4(config, N, K, E, chain, in_fmt):
        raise ValueError(
            f"grouped_gemm_nvfp4: record {config.describe()} does not implement ({N}, {K}, {E}, {chain}, {in_fmt})"
        )
    n_out = N // 2 if chain.pair else N
    out_dtype = _OUT[chain.out_fmt]
    if out is None:
        out = torch.zeros(rows, n_out, dtype=out_dtype, device=a4.device)
    if (
        tuple(out.shape) != (rows, n_out)
        or out.dtype != out_dtype
        or out.stride(1) != 1
        or out.stride(0) % 8
    ):
        raise ValueError(
            "grouped_gemm_nvfp4: out must be a 16-byte-aligned row-major [rows, n_out] of the chain's dtype"
        )
    if rows == 0:
        return out
    if config.swap_ab:
        get_compiled_swapped_nvfp4(config, chain, in_fmt)(
            a4.unsqueeze(-1),
            a_sf.reshape(-1),
            w4.permute(1, 2, 0),
            w_sf.reshape(-1),
            out,
            counts,
            alpha.reshape(-1),
            a_pts.reshape(-1),
            persistent_clusters(a4.device, config),
        )
        return out
    kernel = get_compiled_grouped_nvfp4(config, chain, in_fmt, dense_a)
    kernel(
        a4.unsqueeze(-1),
        w4.permute(1, 2, 0),
        a_sf.reshape(-1),
        w_sf.reshape(-1),
        out,
        counts,
        alpha.reshape(-1),
        a_pts.reshape(-1),
        persistent_clusters(a4.device, config),
    )
    return out


def padded_rows_bound(T: int, k: int, E: int, align: int = GROUP_ALIGN) -> int:
    """An upper bound of the padded expanded row count ``sum_e roundup(counts[e], align)`` that does not read the device
    (CUDA-graph safe): every active expert pads at most ``align - 1`` rows."""
    n = T * k + (align - 1) * min(E, T * k)
    return -(-n // align) * align


def permute_plan_dual(
    experts: torch.Tensor, E: int, pad_align: int = GROUP_ALIGN
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(counts [E], dest [T, k], src [Tk], dest_pad [T, k], pos_pad [Tk])``: the dense plan (:func:`.api.permute_plan`)
    plus the positions with the groups padded to ``pad_align`` rows (``dest_pad[i]``) and the padded position of every dense
    row (``pos_pad[p]``): the dense packing's row map (A rows dense, scale factors and row scales at the padded positions)."""
    from .api import PLAN_MULTI_MIN_N
    from .permute import PLAN_BLOCK, get_compiled_plan, get_compiled_plan_multi

    T, k = experts.shape
    n = T * k
    dev = experts.device
    dest = torch.empty(n, dtype=torch.int32, device=dev)
    src = torch.empty(n, dtype=torch.int32, device=dev)
    dest_pad = torch.empty(n, dtype=torch.int32, device=dev)
    pos_pad = torch.empty(n, dtype=torch.int32, device=dev)
    counts = torch.empty(E, dtype=torch.int32, device=dev)
    flat = experts.reshape(-1)
    if not flat.is_contiguous() or flat.dtype != torch.int32:
        raise TypeError("permute_plan_dual: experts must be a contiguous int32 [T, k]")
    if n == 0:
        counts.zero_()
    elif n >= PLAN_MULTI_MIN_N:
        nb = (n + PLAN_BLOCK - 1) // PLAN_BLOCK
        hist_blk = torch.empty(nb * E, dtype=torch.int32, device=dev)
        get_compiled_plan_multi(1, True, pad_align)(
            flat, counts, dest, src, dest_pad, pos_pad, hist_blk, n, E, nb
        )
    else:
        get_compiled_plan(1, True, pad_align)(
            flat, counts, dest, src, dest_pad, pos_pad, n, E
        )
    return counts, dest.view(T, k), src, dest_pad.view(T, k), pos_pad


_GS_INV: Dict[
    Tuple[str, float], torch.Tensor
] = {}  # the recipe's constant per device, filled once (no fill launch per call)


def route_plan_dual(
    logits: torch.Tensor,
    top_k: int,
    method: Union[S.RoutingMethod, str] = S.SOFTMAX_RENORM,
    bias: Optional[torch.Tensor] = None,
    scaling_factor: Optional[Union[float, torch.Tensor]] = None,
    pad_align: int = GROUP_ALIGN,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """:func:`.api.route` and :func:`permute_plan_dual` in one launch for small token counts: the shared fused kernel's dual form
    (:func:`.api.route_plan` with ``dual=True``): ``(experts [T, k] int32, weights [T, k] f32, counts [E] int32, dest [T, k] int32,
    src [T k] int32, dest_pad [T, k] int32, pos_pad [T k] int32)``, the same bits and integers as the two calls."""
    from .api import route_plan

    return route_plan(
        logits, top_k, method, bias, scaling_factor, dual=True, pad_align=pad_align
    )


def _gs_inv(gs_inv: Union[float, torch.Tensor], device: torch.device) -> torch.Tensor:
    if isinstance(gs_inv, torch.Tensor):
        return gs_inv.to(device=device, dtype=torch.float32).reshape(1)
    key = (str(torch.device(device)), float(gs_inv))
    t = _GS_INV.get(key)
    if t is None:
        t = _GS_INV[key] = torch.full(
            (1,), float(gs_inv), dtype=torch.float32, device=device
        )
    return t


def quantize_activations(
    x: torch.Tensor, gs_inv: Union[float, torch.Tensor] = TE_GS_INV, out=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The class's per-token NVFP4 quantization of bf16 rows ``[M, K]``: ``(x4 [M, K/2] uint8, sf flat 128x4 uint8,
    pts [M] f32)``; ``gs_inv`` the recipe's inverse base scale. ``out = (x4, sf)`` preallocated buffers."""
    from ...batch_invariant.quantization.nvfp4_api import nvfp4_quantize_rows

    M, K = x.shape
    dev = x.device
    if M == 0:
        return (
            torch.empty(0, K // 2, dtype=torch.uint8, device=dev),
            torch.zeros(sf_numel(0, K), dtype=torch.uint8, device=dev),
            torch.empty(0, dtype=torch.float32, device=dev),
        )
    x4, sf, pts = nvfp4_quantize_rows(
        x, _gs_inv(gs_inv, dev), "128x4", per_token=True, out=out, enable_pdl=True
    )
    return x4, sf.reshape(-1), pts


def quantize_rows_mapped(
    x: torch.Tensor,
    gs_inv: Union[float, torch.Tensor],
    row_map: torch.Tensor,
    out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """The class's per-token NVFP4 quantization of bf16 rows ``[M, K]`` with the scale factors and per-row scales stored at
    row ``row_map[r]`` of the padded buffers ``out = (x4 [M, K/2], sf_pad flat 128x4, pts_pad)`` while the codes stay at row
    ``r``: the quantizer's kernel term for term (:mod:`.blocks.quant_nvfp4`), one launch in place of the quantizer and the
    scale re-tiling."""
    from ...batch_invariant.quantization.mx_api import device_sm_count
    from ...batch_invariant.quantization.nvfp4_schedule import default_config
    from .blocks.quant_nvfp4 import get_compiled_rowmap_quant

    M, K = x.shape
    x4, sf_pad, pts_pad = out
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0
    assert x4.shape == (M, K // 2) and x4.is_contiguous() and x4.dtype == torch.uint8
    assert (
        row_map.dtype == torch.int32
        and row_map.is_contiguous()
        and row_map.numel() >= M
    )
    assert sf_pad.dtype == torch.uint8 and pts_pad.dtype == torch.float32
    if M == 0:
        return
    cfg = default_config(
        K, M, x.element_size(), True, device_sm_count(x.device), "128x4"
    )
    kernel = get_compiled_rowmap_quant("bf16", K, cfg, True)
    kernel(x, x4, sf_pad.view(-1), pts_pad, _gs_inv(gs_inv, x.device), row_map, M, K)


def quantize_scatter_rows(
    x: torch.Tensor,
    gs_inv: Union[float, torch.Tensor],
    dest: torch.Tensor,
    dest_pad: torch.Tensor,
    k: int,
    out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """The class's per-token NVFP4 quantization of bf16 rows ``[T, K]`` stored into the grouped layout: token ``t``'s codes at
    the dense rows ``dest[t k + j]`` of ``out[0]`` (``[rows, K/2]``), its scale factors and per-token scale at the padded
    positions ``dest_pad[t k + j]`` of ``out[1]`` (flat 128x4) and ``out[2]``: the quantizer's kernel term for term
    (:mod:`.blocks.quant_nvfp4`), one launch in place of the quantizer and the fp4 scatter."""
    from ...batch_invariant.quantization.mx_api import device_sm_count
    from ...batch_invariant.quantization.nvfp4_schedule import default_config
    from .blocks.quant_nvfp4 import get_compiled_scatter_quant

    T, K = x.shape
    x_perm4, sf_pad, pts_pad = out
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0
    assert (
        x_perm4.dtype == torch.uint8
        and x_perm4.is_contiguous()
        and x_perm4.shape[1] == K // 2
    )
    assert (
        dest.dtype == torch.int32
        and dest_pad.dtype == torch.int32
        and dest.numel() >= T * k
    )
    assert dest.is_contiguous() and dest_pad.is_contiguous()
    assert sf_pad.dtype == torch.uint8 and pts_pad.dtype == torch.float32
    if T == 0:
        return
    cfg = default_config(
        K, T, x.element_size(), True, device_sm_count(x.device), "128x4"
    )
    kernel = get_compiled_scatter_quant("bf16", K, cfg, k, True)
    kernel(
        x,
        x_perm4,
        sf_pad.view(-1),
        pts_pad,
        _gs_inv(gs_inv, x.device),
        dest,
        dest_pad,
        T,
        K,
    )


def _part(device: torch.device) -> Tuple[int, str]:
    return part_of(device)


def fused_moe_nvfp4(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w: MoEWeightsNVFP4,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
    in_fmts: Optional[Tuple[str, str]] = None,
    path: str = "dense",
    gs_inv: Union[float, torch.Tensor] = TE_GS_INV,
    return_intermediates: bool = False,
    swap: Optional[bool] = None,
    plan: Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
):
    """The W4A4 MoE FFN for pre-routed tokens: ``x`` bf16 ``[T, H]`` (quantized here per token with the class and
    ``gs_inv``) or the quantized triple ``(x4 [T, H/2] uint8, x_sf (128x4 swizzled, uint8), x_pts [T] f32)``;
    ``token_selected_experts [T, k]`` int32, ``token_final_scales [T, k]`` f32; the output bf16 ``[T, H]`` in token order.
    ``in_fmts`` names the instruction class per GEMM (default: the dense rule on each expert weight shape and the device's
    family); ``config1`` / ``config2`` the grouped records (any record of the space gives the same bits; the dispatcher's pick
    by default); ``path`` the packing of the expanded rows (``"dense"`` / ``"padded"``: placements); ``swap`` the dense path's
    body -- None: the dispatcher's structural rule (the swapped decode body when the rows per active expert fit a narrow token
    tile), True / False: the swapped / the padded body on both GEMMs (placements: the same bits); explicit records select the
    body themselves (a swapped record needs the dense path). ``plan``: the dense packing's precomputed plan, the outputs of
    :func:`route_plan_dual` after the ids and weights (``counts, dest, src, dest_pad, pos_pad``: the same integers as
    :func:`permute_plan_dual`'s, one launch saved)."""
    from .combine import get_compiled_combine, warps_per_token
    from .permute import warps_per_row
    from .permute_nvfp4 import get_compiled_scatter_fp4, get_compiled_scatter_sf
    from .schedule_nvfp4 import default_configs_nvfp4

    if path not in PATHS:
        raise ValueError(f"fused_moe_nvfp4: path must be one of {PATHS}, not {path!r}")
    H, I, E = w.hidden_size, w.intermediate_size, w.num_experts
    if isinstance(x, torch.Tensor):
        if (
            x.dtype != torch.bfloat16
            or x.dim() != 2
            or x.stride(1) != 1
            or x.stride(0) % 8
        ):
            raise TypeError(
                "fused_moe_nvfp4: x must be a row-major, 16-byte-aligned bf16 [T, H] or a quantized triple"
            )
        if x.shape[1] != H:
            raise ValueError(
                f"fused_moe_nvfp4: x has H = {x.shape[1]}, the weights H = {H}"
            )
        T = x.shape[0]
        dev = x.device
        x_rows: Optional[torch.Tensor] = x.contiguous()
        if path == "dense" and INPUT_SCATTER == "fused":
            x4 = x_sf = x_pts = None  # quantized straight into the grouped layout below
        else:
            x4, x_sf, x_pts = quantize_activations(x, gs_inv)
    else:
        x_rows = None
        x4, x_sf, x_pts = x
        T = x4.shape[0]
        dev = x4.device
        x_sf = x_sf.reshape(-1)
        if (
            x4.dtype != torch.uint8
            or tuple(x4.shape) != (T, H // 2)
            or not x4.is_contiguous()
            or x_sf.dtype != torch.uint8
            or x_sf.numel() < sf_numel(T, H)
            or tuple(x_pts.shape) != (T,)
            or x_pts.dtype != torch.float32
        ):
            raise TypeError(
                "fused_moe_nvfp4: the quantized triple is (x4 [T, H/2] uint8, x_sf 128x4 uint8 of >= sf_numel(T, H) bytes, x_pts [T] f32)"
            )
    k = token_selected_experts.shape[1]
    check_shape_nvfp4(H, I, E, k, w.gated)
    if (
        tuple(token_selected_experts.shape) != (T, k)
        or token_selected_experts.dtype != torch.int32
    ):
        raise TypeError(
            "fused_moe_nvfp4: token_selected_experts must be an int32 [T, k]"
        )
    if (
        tuple(token_final_scales.shape) != (T, k)
        or token_final_scales.dtype != torch.float32
    ):
        raise TypeError("fused_moe_nvfp4: token_final_scales must be a float32 [T, k]")
    experts = token_selected_experts.contiguous()
    scales = token_final_scales.contiguous()
    if out is None:
        out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    if (
        tuple(out.shape) != (T, H)
        or out.dtype != torch.bfloat16
        or out.stride(1) != 1
        or out.stride(0) % 8
    ):
        raise ValueError(
            "fused_moe_nvfp4: out must be a row-major, 16-byte-aligned bf16 [T, H]"
        )
    sm_count, family = _part(dev)
    in1, in2 = in_fmts if in_fmts is not None else w.classes(family)
    chain1 = w.chain1
    if config1 is None or config2 is None:
        c1, c2 = default_configs_nvfp4(
            T,
            k,
            H,
            I,
            E,
            (in1, in2),
            chain1,
            GEMM2_CHAIN,
            sm_count,
            family,
            swap=swap if path == "dense" else False,
        )
        config1, config2 = config1 or c1, config2 or c2
    if path != "dense" and (swap or config1.swap_ab or config2.swap_ab):
        raise ValueError(
            "fused_moe_nvfp4: the swapped decode body reads the dense packing (path='dense')"
        )
    if path != "dense" and plan is not None:
        raise ValueError(
            "fused_moe_nvfp4: the precomputed plan is the dense packing's (path='dense')"
        )
    Tk = T * k
    Tk_pad = padded_rows_bound(T, k, E)
    pcols_h, pcols_i = padded_sf_cols(H), padded_sf_cols(I)
    gsi = _gs_inv(gs_inv, dev)
    if path == "dense":
        counts, dest, src, dest_pad, pos_pad = (
            plan if plan is not None else permute_plan_dual(experts, E)
        )  # ``plan``: ``route_plan_dual``'s outputs, the same integers
        # the fp4 rows back to back (a tail past Tk for the last tiles' TMA boxes); the scale factors and the row scales at
        # the padded positions (one 128x4 atom per (expert, 128-row tile); a tail past Tk_pad likewise)
        Tk_a, Tk_sf = Tk + DENSE_TAIL, Tk_pad + DENSE_TAIL
        x_perm4 = torch.empty(Tk_a, H // 2, dtype=torch.uint8, device=dev)
        x_perm_sf = torch.empty(sf_numel(Tk_sf, H), dtype=torch.uint8, device=dev)
        x_perm_pts = torch.empty(Tk_sf, dtype=torch.float32, device=dev)
        if Tk and x4 is None:
            quantize_scatter_rows(
                x_rows,
                gsi,
                dest.reshape(-1),
                dest_pad.reshape(-1),
                k,
                (x_perm4, x_perm_sf, x_perm_pts),
            )
        elif Tk:
            get_compiled_scatter_fp4(k, warps_per_row(Tk, H // 32, sm_count))(
                x4,
                x_sf,
                x_pts,
                dest.reshape(-1),
                dest_pad.reshape(-1),
                x_perm4,
                x_perm_sf,
                x_perm_pts,
                Tk,
                H // 2,
                pcols_h,
            )
        a = torch.empty(Tk_a, I, dtype=torch.bfloat16, device=dev)
        grouped_gemm_nvfp4(
            x_perm4,
            x_perm_sf,
            x_perm_pts,
            w.w1,
            w.w1_sf,
            w.w1_alpha,
            counts,
            config1,
            chain1,
            in1,
            out=a,
            dense_a=True,
        )
        # the class's second pass over the live rows: the codes stay dense; the scale factors and the row scales are re-tiled
        # to the padded positions for GEMM2's atom-aligned scale boxes (a byte move)
        a4 = torch.empty(Tk_a, I // 2, dtype=torch.uint8, device=dev)
        a_sf_pad = torch.empty(sf_numel(Tk_sf, I), dtype=torch.uint8, device=dev)
        a_pts_pad = torch.empty(Tk_sf, dtype=torch.float32, device=dev)
        a_sf = a_pts = None
        if Tk and INTERMEDIATE_RETILE == "mapped":
            quantize_rows_mapped(a[:Tk], gsi, pos_pad, (a4[:Tk], a_sf_pad, a_pts_pad))
        elif Tk:
            a_sf = torch.empty(sf_numel(Tk, I), dtype=torch.uint8, device=dev)
            _, _, a_pts = quantize_activations(a[:Tk], gsi, out=(a4[:Tk], a_sf))
            get_compiled_scatter_sf(1)(
                a_sf, a_pts, pos_pad, a_sf_pad, a_pts_pad, Tk, pcols_i
            )
        y = torch.empty(Tk_a, H, dtype=torch.bfloat16, device=dev)
        grouped_gemm_nvfp4(
            a4,
            a_sf_pad,
            a_pts_pad,
            w.w2,
            w.w2_sf,
            w.w2_alpha,
            counts,
            config2,
            GEMM2_CHAIN,
            in2,
            out=y,
            dense_a=True,
        )
        if T:
            get_compiled_combine(k, "bf16", "bf16", warps_per_token(T, H, sm_count))(
                y, dest, scales, out, T, H
            )
        if return_intermediates:
            return out, dict(
                counts=counts,
                dest=dest,
                src=src,
                dest_pad=dest_pad,
                pos_pad=pos_pad,
                x4=x4,
                x_sf=x_sf,
                x_pts=x_pts,
                x_perm4=x_perm4[:Tk],
                x_perm_sf=x_perm_sf[: sf_numel(Tk_pad, H)],
                x_perm_pts=x_perm_pts[:Tk_pad],
                a=a[:Tk],
                a4=a4[:Tk],
                a_sf=a_sf,
                a_pts=a_pts,
                a_sf_pad=a_sf_pad[: sf_numel(Tk_pad, I)],
                a_pts_pad=a_pts_pad[:Tk_pad],
                y=y[:Tk],
                config1=config1,
                config2=config2,
                in_fmts=(in1, in2),
                Tk_pad=Tk_pad,
                path=path,
            )
        return out
    # the padded packing: every group's rows start at a 128-row atom; the padding rows are zero (computed, never stored)
    from .permute import get_compiled_plan, get_compiled_plan_multi, PLAN_BLOCK
    from .api import PLAN_MULTI_MIN_N

    n = Tk
    dest = torch.empty(n, dtype=torch.int32, device=dev)
    src = torch.full((Tk_pad,), -1, dtype=torch.int32, device=dev)
    counts = torch.empty(E, dtype=torch.int32, device=dev)
    flat = experts.reshape(-1)
    if n == 0:
        counts.zero_()
    elif n >= PLAN_MULTI_MIN_N:
        nb = (n + PLAN_BLOCK - 1) // PLAN_BLOCK
        hist_blk = torch.empty(nb * E, dtype=torch.int32, device=dev)
        get_compiled_plan_multi(GROUP_ALIGN, False)(
            flat, counts, dest, src, dest, dest, hist_blk, n, E, nb
        )
    else:
        get_compiled_plan(GROUP_ALIGN, False)(flat, counts, dest, src, dest, dest, n, E)
    dest = dest.view(T, k)
    x_perm4 = torch.zeros(Tk_pad, H // 2, dtype=torch.uint8, device=dev)
    x_perm_sf = torch.zeros(sf_numel(Tk_pad, H), dtype=torch.uint8, device=dev)
    x_perm_pts = torch.zeros(Tk_pad, dtype=torch.float32, device=dev)
    if Tk:
        flat_dest = dest.reshape(-1)
        get_compiled_scatter_fp4(k, warps_per_row(Tk, H // 32, sm_count))(
            x4,
            x_sf,
            x_pts,
            flat_dest,
            flat_dest,
            x_perm4,
            x_perm_sf,
            x_perm_pts,
            Tk,
            H // 2,
            pcols_h,
        )
    a = grouped_gemm_nvfp4(
        x_perm4,
        x_perm_sf,
        x_perm_pts,
        w.w1,
        w.w1_sf,
        w.w1_alpha,
        counts,
        config1,
        chain1,
        in1,
    )
    a4, a_sf, a_pts = quantize_activations(
        a, gsi
    )  # the class, second pass (padding rows included: zeros)
    y = grouped_gemm_nvfp4(
        a4, a_sf, a_pts, w.w2, w.w2_sf, w.w2_alpha, counts, config2, GEMM2_CHAIN, in2
    )
    if T:
        get_compiled_combine(k, "bf16", "bf16", warps_per_token(T, H, sm_count))(
            y, dest, scales, out, T, H
        )
    if return_intermediates:
        return out, dict(
            counts=counts,
            dest=dest,
            src=src,
            x4=x4,
            x_sf=x_sf,
            x_pts=x_pts,
            x_perm4=x_perm4,
            x_perm_sf=x_perm_sf,
            x_perm_pts=x_perm_pts,
            a=a,
            a4=a4,
            a_sf=a_sf,
            a_pts=a_pts,
            y=y,
            config1=config1,
            config2=config2,
            in_fmts=(in1, in2),
            Tk_pad=Tk_pad,
            path=path,
        )
    return out


def moe_ffn_nvfp4(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    router_logits: torch.Tensor,
    w: MoEWeightsNVFP4,
    top_k: int,
    method: Union[S.RoutingMethod, str] = S.SOFTMAX_RENORM,
    bias: Optional[torch.Tensor] = None,
    scaling_factor: Optional[Union[float, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
) -> torch.Tensor:
    """Routing (the bf16 classes' kernel) + the W4A4 FFN."""
    experts, weights = route(router_logits, top_k, method, bias, scaling_factor)
    return fused_moe_nvfp4(x, experts, weights, w, out, config1, config2)


__all__ = [
    "ALIGN_NVFP4",
    "BatchInvariantUnsupportedError",
    "DENSE_TAIL",
    "F32_CHAIN",
    "GEMM2_CHAIN",
    "GROUP_ALIGN",
    "MAX_TOP_K",
    "MoEWeightsNVFP4",
    "PATHS",
    "TE_GS_INV",
    "check_shape_nvfp4",
    "fp4_global_scale",
    "fused_moe_nvfp4",
    "grouped_gemm_nvfp4",
    "moe_ffn_nvfp4",
    "padded_rows_bound",
    "permute_plan_dual",
    "route_plan_dual",
    "quantize_activations",
    "sm_count_of",
]
