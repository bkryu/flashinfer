# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 x MXFP8 MoE classes ``moe.mxfp8.<act>``: the expert weights and the activations quantized by the
``mxfp8.quant`` class (e4m3 values, one UE8M0 scale per 32 elements), the two GEMMs as grouped placements of the dense
MXFP8 class of the part (the block scales inside the block-scaled step), the bf16 classes' chains, routing, deterministic
permutation and slot-order combine.

``MoEWeightsMXFP8.prepare`` quantizes bf16 experts per expert into the 128x4 swizzled scale layout the instruction reads;
``quantize_activations_mxfp8`` is the class's per-token activation quantization into the layer's canonical MXFP8 pack
(e4m3 ``[T, H]`` with the token-major UE8M0 scales ``[T, H / 32]``, the pack ``TrtllmFp8BlockConfig`` and
``CutlassMxfp8Config`` prepare); ``fused_moe_mxfp8`` runs the FFN on pre-routed tokens from that pack (the caller's bytes
taken as given) and ``moe_ffn_mxfp8`` composes the routing. Every output bit is a function of the token's own quantized
row, its ``k`` experts and routing weights, and the quantized expert weights: batch size, composition, row position,
expert load and the grouped records never enter the dataflow (the specification is :mod:`.spec_mxfp8`).

The expanded rows are permuted into groups padded to whole 128-row scale-factor atoms (the swizzled scale layout is affine
only inside an atom); one scatter launch moves the e4m3 rows and re-tiles the linear scale bytes into the atoms, zero-filling
the padding rows. The GEMM outputs are allocated uninitialised: a padding row of the intermediate holds unspecified bytes,
is quantized to some codes and read by the padded tiles, and no stored value depends on it (the step is row-local, the
store predicate is the group's real end).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from ...batch_invariant.quantization.layout import padded_sf_cols as _padded_sf_cols
from ...batch_invariant.quantization.layout import sf_numel as _sf_numel
from ...batch_invariant.quantization.mx_api import mxfp8_quantize_batch_invariant
from ...batch_invariant.support import compute_capability, unsupported
from ...gemm.batch_invariant.api import persistent_clusters, sm_count_of
from ...gemm.batch_invariant.generator import Knobs
from ...gemm.batch_invariant.heuristic import family_of
from ...gemm.batch_invariant.schedule import chunks_for, mx_class_for
from . import spec as S
from .api import interleave_fc1, route
from .blocks.epilogue import GroupedChain
from .spec_mxfp8 import BLOCK, CLASS, MX_CLASSES

#: The padded group alignment (rows): one 128x4 scale-factor atom.
GROUP_ALIGN = 128
#: The routed expert bound of the routing kernel (the bf16 classes' bucket); the group tables take ``MAX_E``.
E_MAX = 1024
K_MAX = 16
_OUT = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


def mx_class_of(device: torch.device) -> str:
    """The instruction class a device runs: a function of the architecture family only (``mxfp8_k64`` on SM107,
    ``mxfp8`` on SM100 / SM103), never of the shape."""
    return mx_class_for("mxfp8", family_of(compute_capability(device)))


def sf_numel(rows: int, K: int) -> int:
    """Bytes of the 128x4 swizzled UE8M0 buffer of a ``(rows, K)`` operand (rows padded to 128, scale columns to 4)."""
    return _sf_numel(rows, K, BLOCK, "128x4")


def padded_sf_cols(K: int) -> int:
    return _padded_sf_cols(K, BLOCK, "128x4")


def padded_rows_bound(T: int, k: int, E: int, align: int = GROUP_ALIGN) -> int:
    """An upper bound of the padded expanded row count ``sum_e roundup(counts[e], align)`` that does not read the device
    (CUDA-graph safe): every active expert pads at most ``align - 1`` rows."""
    n = T * k
    if n == 0:
        return 0
    return -(-(n + (align - 1) * min(E, n)) // align) * align


def check_shape_mxfp8(H: int, I: int, E: int, k: Optional[int] = None) -> None:
    """The complete acceptance rule of the class on ``(H, I, E, k)``; anything else raises
    :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`. Everything else about a shape (the padded group
    layout, the tile scheduler, the records, the per-expert tensor maps) is a placement.

    * ``H`` and ``I`` positive multiples of 32: whole 32-element scale blocks of the quantized ``x`` (K = H), of the
      quantized intermediate (K = I) and of both expert weights, hence 16-byte e4m3 rows (the gate / up pairing needs
      ``I % 8``, implied);
    * ``1 <= E <= 1024`` (the routing kernel's buckets, the group tables), ``1 <= k <= min(E, 16)``;
    * ``chunks_for(2I, H) == chunks_for(H, I) == 1`` for the MX class: the grouped classes take unsplit expert weight shapes
      (the dense class K-splits a weight only when ``K`` is very large against ``N``)."""
    if H <= 0 or I <= 0 or H % BLOCK or I % BLOCK:
        raise unsupported(
            CLASS,
            f"hidden_size {H}, intermediate_size {I}",
            f"positive multiples of {BLOCK} (whole UE8M0 scale blocks: 16-byte e4m3 rows of the activations, the intermediate and the weights)",
        )
    if not 1 <= E <= E_MAX:
        raise unsupported(CLASS, f"num_experts {E}", f"1 <= num_experts <= {E_MAX}")
    if k is not None and not 1 <= k <= min(E, K_MAX):
        raise unsupported(
            CLASS, f"top_k {k}", f"1 <= top_k <= min(num_experts, {K_MAX})"
        )
    if chunks_for(2 * I, H, "mxfp8") != 1 or chunks_for(H, I, "mxfp8") != 1:
        raise unsupported(
            CLASS,
            f"expert weights (2I = {2 * I}, H = {H}) / (H = {H}, I = {I})",
            "weight shapes the dense MXFP8 class folds in one chunk (the grouped class takes unsplit weights)",
        )


# ----------------------------------------------------------------------------- the quantizer (the declared class)


def quantize_mxfp8_rows(
    x: torch.Tensor,
    swizzled: bool,
    enable_pdl: bool = False,
    valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The ``mxfp8.quant`` class on bf16 rows ``[M, K]``: ``(codes [M, K] float8_e4m3fn, sf)`` with ``sf`` the flat uint8
    UE8M0 bytes in the 128x4 swizzled layout (``sf_numel(M, K)``: the block-scaled GEMMs' operand) or the token-major
    ``[M, K / 32]`` bytes (the activation pack). ``valid`` (int32 ``[M]``, negative = skip): only the rows marked live are
    read and written (the padded operand's padding rows hold unspecified bytes no stored value depends on)."""
    codes, sf = mxfp8_quantize_batch_invariant(
        x, "128x4" if swizzled else "linear", BLOCK, enable_pdl, valid=valid
    )
    sf = sf.view(torch.uint8)
    return codes, sf.reshape(-1) if swizzled else sf.reshape(x.shape[0], -1)


def quantize_activations_mxfp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The class's activation quantization: ``(xq [T, H] float8_e4m3fn, xsf [T, H / 32] uint8)``, the scales token-major
    (the layer's canonical MXFP8 activation pack)."""
    if x.dim() != 2 or x.dtype != torch.bfloat16:
        raise TypeError("quantize_activations_mxfp8 takes a bf16 [T, H]")
    if x.shape[1] % BLOCK:
        raise unsupported(CLASS, f"hidden_size {x.shape[1]}", f"a multiple of {BLOCK}")
    return quantize_mxfp8_rows(x.contiguous(), swizzled=False)


# ----------------------------------------------------------------------------- weights


@dataclass
class MoEWeightsMXFP8:
    """The MXFP8 expert weights as the kernels consume them: ``w1 [E, N1, H]`` e4m3 (``N1 = 2I`` gate / up interleaved
    for a gated chain, ``I`` as given for an ungated one), ``w1_sf [E, sf_numel(N1, H)]`` uint8 (one 128x4 swizzled
    UE8M0 buffer per expert), ``w2 [E, H, I]`` e4m3, ``w2_sf [E, sf_numel(H, I)]``; the GEMM1 chain ``act`` (a gated
    chain of ``spec.ACTS`` or an ungated one of ``spec.UNGATED_ACTS``) with its ``params``."""

    w1: torch.Tensor
    w1_sf: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
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
    ) -> "MoEWeightsMXFP8":
        """Quantize bf16 experts (``fc1 [E, 2I, H]`` rows ``[up, gate]`` for a gated chain, ``[E, I, H]`` for an ungated
        one; ``fc2 [E, H, I]``) per expert with the ``mxfp8.quant`` class into the 128x4 swizzled scale layout. The gate /
        up interleave is a row permutation applied before the quantization (a row's block scales are the row's own).
        ``act`` defaults to the class's default chain of the gating (``silu`` gated, ``relu2`` ungated)."""
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
                "the MXFP8 MoE classes take bf16 expert weights fc1 [E, N1, H], fc2 [E, H, I]"
            )
        E, N1, H = fc1.shape
        I = N1 // 2 if gated else N1
        if tuple(fc2.shape) != (E, H, I):
            raise ValueError(
                f"fc2 {tuple(fc2.shape)} does not match fc1 {tuple(fc1.shape)} (gated={gated})"
            )
        check_shape_mxfp8(H, I, E)
        if not fc1.is_cuda:
            raise TypeError(
                "the MXFP8 MoE classes quantize the expert weights on a CUDA device"
            )
        w1i = fc1.contiguous() if (interleaved or not gated) else interleave_fc1(fc1)
        w1, w1_sf = _quantize_experts(w1i)
        w2, w2_sf = _quantize_experts(fc2.contiguous())
        return MoEWeightsMXFP8(w1, w1_sf, w2, w2_sf, gated, act, key)

    @property
    def num_experts(self) -> int:
        return self.w1.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.w1.shape[2]

    @property
    def intermediate_size(self) -> int:
        return self.w2.shape[2]

    @property
    def n1(self) -> int:
        return self.w1.shape[1]

    @property
    def chain1(self) -> GroupedChain:
        """The GEMM1 epilogue chain of the class: the gated ``act`` on the (gate, up) pair or the ungated ``act`` on the
        single accumulator, with the parameters (the bf16 classes' chains: no scale)."""
        return GroupedChain(
            "glu" if self.gated else "act", self.act, "bf16", self.params
        )

    def view(self) -> Dict[str, object]:
        """The weight view a :class:`flashinfer.fused_moe.MoEWeightPack` stores for this backend."""
        return {
            "w1": self.w1,
            "w1_sf": self.w1_sf,
            "w2": self.w2,
            "w2_sf": self.w2_sf,
            "gated": self.gated,
            "act": self.act,
            "params": self.params,
        }

    @staticmethod
    def from_view(view: Dict[str, object]) -> "MoEWeightsMXFP8":
        gated = bool(view.get("gated", True))
        return MoEWeightsMXFP8(
            view["w1"],  # type: ignore[arg-type]
            view["w1_sf"],  # type: ignore[arg-type]
            view["w2"],  # type: ignore[arg-type]
            view["w2_sf"],  # type: ignore[arg-type]
            gated,
            str(view.get("act", "silu" if gated else "relu2")),
            tuple(view.get("params", ())),  # type: ignore[arg-type]
        )


def _quantize_experts(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[E, N, K]`` bf16 -> (``[E, N, K]`` e4m3, ``[E, sf_numel(N, K)]`` uint8): one launch when ``N % 128 == 0`` (an
    expert's rows fill whole scale-factor atoms, so the per-expert buffers are consecutive slices of the stacked one and
    the bits are the per-expert quantization's: a row's quantization is row-local), else per expert."""
    E, N, K = w.shape
    if N % GROUP_ALIGN == 0:
        codes, sf = quantize_mxfp8_rows(w.reshape(E * N, K), swizzled=True)
        return codes.view(E, N, K), sf.view(E, sf_numel(N, K))
    outs = [quantize_mxfp8_rows(w[e], swizzled=True) for e in range(E)]
    return (
        torch.stack([c for c, _ in outs]).contiguous(),
        torch.stack([s for _, s in outs]).contiguous(),
    )


# ----------------------------------------------------------------------------- the grouped GEMM


def _bytes(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.dtype == torch.uint8:
        return t
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        return t.view(torch.uint8)
    raise TypeError(
        f"{name}: expected float8_e4m3fn / UE8M0 / uint8 bytes, got {t.dtype}"
    )


def grouped_gemm_mxfp8(
    a: torch.Tensor,
    a_sf: torch.Tensor,
    w: torch.Tensor,
    w_sf: torch.Tensor,
    counts: torch.Tensor,
    config: Knobs,
    chain: GroupedChain,
    in_fmt: Optional[str] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``out[p] = chain(a[p] @ w[e(p)].T)`` with the UE8M0 block scales applied inside the block-scaled step, on the padded
    expert segments ``counts`` (int32 ``[E]``; the rows of group ``e`` start at the exclusive prefix of the counts rounded
    up to 128): ``a [Tk_pad, K]`` e4m3, ``a_sf`` its 128x4 swizzled scales (``sf_numel(Tk_pad, K)`` bytes), ``w [E, N,
    K]`` e4m3, ``w_sf [E, sf_numel(N, K)]``. ``chain``: a bf16 class chain (no scale); ``in_fmt`` the instruction class
    (the device's by default); ``out [Tk_pad, Nout]`` of the chain's dtype (rows past a group's real end are never stored)."""
    from .blocks.compile_mxfp8 import (
        can_implement_grouped_mxfp8,
        get_compiled_grouped_mxfp8,
    )

    if (
        a.dtype != torch.float8_e4m3fn
        or w.dtype != torch.float8_e4m3fn
        or a.dim() != 2
        or w.dim() != 3
    ):
        raise TypeError("grouped_gemm_mxfp8 takes e4m3 a [Tk_pad, K] and w [E, N, K]")
    Tk, K = a.shape
    E, N, K2 = w.shape
    if K != K2 or a.stride(1) != 1 or a.stride(0) != K or not w.is_contiguous():
        raise ValueError("grouped_gemm_mxfp8: K-contiguous operands of one K")
    if Tk % GROUP_ALIGN:
        raise ValueError(
            "grouped_gemm_mxfp8: the padded rows are a multiple of the 128-row scale-factor atom"
        )
    if (
        counts.dtype != torch.int32
        or tuple(counts.shape) != (E,)
        or not counts.is_contiguous()
    ):
        raise ValueError("grouped_gemm_mxfp8: counts must be a contiguous int32 [E]")
    a_sf, w_sf = _bytes(a_sf, "a_sf").reshape(-1), _bytes(w_sf, "w_sf").reshape(-1)
    if a_sf.numel() < sf_numel(Tk, K) or w_sf.numel() < E * sf_numel(N, K):
        raise ValueError(
            "grouped_gemm_mxfp8: the scale-factor buffers are smaller than the operands' 128x4 layouts"
        )
    in_fmt = in_fmt or mx_class_of(a.device)
    if in_fmt not in MX_CLASSES:
        raise ValueError(f"grouped_gemm_mxfp8: unknown instruction class {in_fmt!r}")
    if not can_implement_grouped_mxfp8(config, N, K, E, chain, in_fmt):
        raise ValueError(
            f"grouped_gemm_mxfp8: record {config.describe()} does not implement (N={N}, K={K}, E={E}, {chain.describe()}, {in_fmt})"
        )
    n_out = N // 2 if chain.pair else N
    out_dtype = _OUT[chain.out_fmt]
    if out is None:
        out = torch.empty(Tk, n_out, dtype=out_dtype, device=a.device)
    if (
        tuple(out.shape) != (Tk, n_out)
        or out.dtype != out_dtype
        or out.stride(1) != 1
        or out.stride(0) % 8
    ):
        raise ValueError(
            "grouped_gemm_mxfp8: out must be a 16-byte-aligned row-major [Tk_pad, n_out] of the chain's dtype"
        )
    kernel = get_compiled_grouped_mxfp8(config, chain, in_fmt)
    if Tk:
        kernel(
            a.unsqueeze(-1),
            w.permute(1, 2, 0),
            a_sf,
            w_sf,
            out,
            counts,
            persistent_clusters(a.device, config),
        )
    return out


# ----------------------------------------------------------------------------- the padded permutation


def permute_plan_padded(
    experts: torch.Tensor, E: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(counts [E] int32, dest [T, k] int32, src [Tk_pad] int32)``: the bf16 classes' deterministic permutation
    (expert order, then token-slot order) with the group offsets rounded up to 128 rows; ``src`` has
    ``padded_rows_bound`` rows and ``-1`` in every padding slot."""
    from .api import PLAN_MULTI_MIN_N
    from .permute import PLAN_BLOCK, get_compiled_plan, get_compiled_plan_multi

    T, k = experts.shape
    n = T * k
    dev = experts.device
    flat = experts.reshape(-1)
    if not flat.is_contiguous() or flat.dtype != torch.int32:
        raise TypeError(
            "permute_plan_padded: experts must be a contiguous int32 [T, k]"
        )
    dest = torch.empty(n, dtype=torch.int32, device=dev)
    src = torch.full((padded_rows_bound(T, k, E),), -1, dtype=torch.int32, device=dev)
    counts = torch.empty(E, dtype=torch.int32, device=dev)
    if n == 0:
        counts.zero_()
        return counts, dest.view(T, k), src
    if n >= PLAN_MULTI_MIN_N:
        nb = (n + PLAN_BLOCK - 1) // PLAN_BLOCK
        hist_blk = torch.empty(nb * E, dtype=torch.int32, device=dev)
        get_compiled_plan_multi(GROUP_ALIGN, False, GROUP_ALIGN)(
            flat, counts, dest, src, dest, dest, hist_blk, n, E, nb
        )
    else:
        get_compiled_plan(GROUP_ALIGN, False, GROUP_ALIGN)(
            flat, counts, dest, src, dest, dest, n, E
        )
    return counts, dest.view(T, k), src


# ----------------------------------------------------------------------------- the op


def fused_moe_mxfp8(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w: MoEWeightsMXFP8,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
    in_fmt: Optional[str] = None,
    return_intermediates: bool = False,
) -> Any:
    """The MXFP8 x MXFP8 MoE FFN for pre-routed tokens: ``x`` a bf16 ``[T, H]`` (quantized here per token by the
    ``mxfp8.quant`` class) or the quantized pack ``(xq [T, H] e4m3, xsf [T, H / 32] uint8 / UE8M0)`` taken as given;
    ``token_selected_experts [T, k]`` int32, ``token_final_scales [T, k]`` fp32; the output bf16 ``[T, H]``. ``in_fmt``
    names the instruction class (the device's by default); ``config1`` / ``config2`` the grouped records (any record of
    the space gives the same bits; the dispatcher's pick by default). ``return_intermediates``: the live buffers
    (``counts``, ``dest``, ``src``, ``xq``, ``xsf``, ``x_perm`` / ``x_perm_sf``, ``a``, ``aq`` / ``a_sf``, ``y``) and
    the records used."""
    from .combine import get_compiled_combine, warps_per_token
    from .mx_permute import get_compiled_scatter_mx
    from .schedule_mxfp8 import GEMM2_CHAIN, default_configs_mx

    H, I, E = w.hidden_size, w.intermediate_size, w.num_experts
    if isinstance(x, torch.Tensor):
        if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != H:
            raise TypeError(
                f"fused_moe_mxfp8: x must be a bf16 [T, {H}] or the quantized pack (xq, xsf)"
            )
        dev, T = x.device, x.shape[0]
        if T:
            xq, xsf = quantize_activations_mxfp8(x)
        else:
            xq = torch.empty(0, H, dtype=torch.float8_e4m3fn, device=dev)
            xsf = torch.empty(0, H // BLOCK, dtype=torch.uint8, device=dev)
    else:
        xq, xsf = x
        dev, T = xq.device, xq.shape[0]
    if (
        xq.dtype != torch.float8_e4m3fn
        or tuple(xq.shape) != (T, H)
        or not xq.is_contiguous()
    ):
        raise TypeError(f"fused_moe_mxfp8: xq must be a contiguous e4m3 [T, {H}]")
    xsf8 = _bytes(xsf, "xsf")
    if tuple(xsf8.shape) != (T, H // BLOCK) or not xsf8.is_contiguous():
        raise TypeError(
            f"fused_moe_mxfp8: xsf must be the token-major [T, {H // BLOCK}] UE8M0 scales of the activation pack, got {tuple(xsf8.shape)}"
        )
    k = token_selected_experts.shape[1]
    check_shape_mxfp8(H, I, E, k)
    if (
        tuple(token_selected_experts.shape) != (T, k)
        or token_selected_experts.dtype != torch.int32
    ):
        raise TypeError(
            "fused_moe_mxfp8: token_selected_experts must be an int32 [T, k]"
        )
    if (
        tuple(token_final_scales.shape) != (T, k)
        or token_final_scales.dtype != torch.float32
    ):
        raise TypeError("fused_moe_mxfp8: token_final_scales must be a float32 [T, k]")
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
            "fused_moe_mxfp8: out must be a row-major, 16-byte-aligned bf16 [T, H]"
        )
    in_fmt = in_fmt or mx_class_of(dev)
    if in_fmt not in MX_CLASSES:
        raise ValueError(f"fused_moe_mxfp8: unknown instruction class {in_fmt!r}")
    if T == 0:  # no tokens: nothing to route, the (0, H) output as it stands
        if return_intermediates:
            return out, dict(
                counts=torch.zeros(E, dtype=torch.int32, device=dev),
                in_fmt=in_fmt,
                Tk_pad=0,
            )
        return out
    if config1 is None or config2 is None:
        sm_count, family = _part(dev)
        c1, c2 = default_configs_mx(T, k, H, I, E, w.chain1, in_fmt, sm_count, family)
        config1, config2 = config1 or c1, config2 or c2
    counts, dest, src = permute_plan_padded(experts, E)
    Tk_pad = src.shape[0]
    nb, pcols = H // BLOCK, padded_sf_cols(H)
    x_perm = torch.empty(Tk_pad, H, dtype=torch.float8_e4m3fn, device=dev)
    x_perm_sf = torch.empty(sf_numel(Tk_pad, H), dtype=torch.uint8, device=dev)
    # one launch: the e4m3 rows and the UE8M0 scales to the padded positions (the scales re-tiled into the swizzled
    # atoms), the padding rows zero-filled (values and scale bytes)
    get_compiled_scatter_mx(k)(
        xq.view(torch.uint8),
        xsf8.reshape(-1),
        src,
        x_perm.view(torch.uint8),
        x_perm_sf,
        Tk_pad,
        H,
        nb,
        pcols,
    )
    a = grouped_gemm_mxfp8(
        x_perm, x_perm_sf, w.w1, w.w1_sf, counts, config1, w.chain1, in_fmt
    )  # [Tk_pad, I] bf16
    aq, a_sf = quantize_mxfp8_rows(
        a, swizzled=True, enable_pdl=True, valid=src
    )  # the quantizer class, second pass, the live rows
    y = grouped_gemm_mxfp8(
        aq, a_sf, w.w2, w.w2_sf, counts, config2, GEMM2_CHAIN, in_fmt
    )  # [Tk_pad, H] bf16
    get_compiled_combine(k, "bf16", "bf16", warps_per_token(T, H, sm_count_of(dev)))(
        y, dest, scales, out, T, H
    )
    if return_intermediates:
        return out, dict(
            counts=counts,
            dest=dest,
            src=src,
            xq=xq,
            xsf=xsf8,
            x_perm=x_perm,
            x_perm_sf=x_perm_sf,
            a=a,
            aq=aq,
            a_sf=a_sf,
            y=y,
            config1=config1,
            config2=config2,
            in_fmt=in_fmt,
            Tk_pad=Tk_pad,
        )
    return out


def _part(device: torch.device) -> Tuple[int, str]:
    from ...gemm.batch_invariant.api import part_of

    return part_of(device)


def moe_ffn_mxfp8(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    router_logits: torch.Tensor,
    w: MoEWeightsMXFP8,
    top_k: int,
    method: Union[S.RoutingMethod, str] = S.SOFTMAX_RENORM,
    bias: Optional[torch.Tensor] = None,
    scaling_factor: Optional[Union[float, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
    in_fmt: Optional[str] = None,
) -> torch.Tensor:
    """Routing (the bf16 classes' kernel) + the MXFP8 FFN."""
    experts, weights = route(router_logits, top_k, method, bias, scaling_factor)
    return fused_moe_mxfp8(x, experts, weights, w, out, config1, config2, in_fmt)
