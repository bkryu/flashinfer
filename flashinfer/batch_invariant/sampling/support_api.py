# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entry points of the support class ``support.row.v1``.

:func:`sampling_support` is the class's entry: the kept set of every row under top-k / top-p / min-p (ties kept) with its
effective threshold, count and mass, and on request the renormalised probabilities (``lse.row``'s ``prob`` chain with the
kept mass), the masked logits or the renormalised log-probabilities. :func:`top_k_renorm_probs_batch_invariant`,
:func:`top_p_renorm_probs_batch_invariant` and :func:`top_k_mask_logits_batch_invariant` are the class's forms of
``flashinfer.sampling.top_k_renorm_probs`` / ``top_p_renorm_probs`` / ``top_k_mask_logits`` -- one call of
:func:`sampling_support` each; internal entries: the operators carry no ``batch_invariant`` flag in this release
(their ``is_deterministic=True`` kernels serve them; exposing the class there is deferred).

Every unsupported input raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`; nothing falls back.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from .api import (
    _DTYPE_STR,
    _aligned_rows,
    _dummy,
    _pdl,
    _rows,
    _temperature,
    check_input,
)
from .schedule import BLOCK, SMEM_BUDGET, LSEConfig, device_sm_count, max_vec_elems
from .support_schedule import support_config
from .support_spec import ORDERS, Filters

CLASS = "support.row.v1"
#: The compute capabilities the class is stated and tested for: SM100 (B200), SM103 (B300), SM107 (Rubin).
SUPPORT_ARCHS = ((10, 0), (10, 3), (10, 7))


class Support:
    """The per-row results of :func:`sampling_support`: ``threshold`` f32 (the effective threshold; the kept set is
    ``w >= threshold``), ``count`` int64, ``mass`` f32 (the kept weights' canonical tree), ``sum`` f32 (the row's
    canonical tree) -- each of the input's leading shape -- and, when requested, ``mask`` bool, ``logprobs`` f32 (logits
    only; NaN where not kept), ``probs`` (the renormalised probabilities in the input dtype; ``+0`` where not kept) or
    ``masked_logits`` (the input with ``-inf`` where not kept), each of the input's shape."""

    __slots__ = (
        "threshold",
        "count",
        "mass",
        "sum",
        "mask",
        "logprobs",
        "probs",
        "masked_logits",
    )

    def __init__(
        self,
        threshold,
        count,
        mass,
        sum_,
        mask=None,
        logprobs=None,
        probs=None,
        masked_logits=None,
    ):
        self.threshold, self.count, self.mass, self.sum = threshold, count, mass, sum_
        self.mask, self.logprobs, self.probs, self.masked_logits = (
            mask,
            logprobs,
            probs,
            masked_logits,
        )


def _row_param(
    value,
    M: int,
    lead: Tuple[int, ...],
    device,
    dtype: torch.dtype,
    name: str,
    lo: float = 0.0,
) -> Tuple[Optional[torch.Tensor], bool]:
    """``None`` -> ``(None, False)``; a scalar or a tensor of one value, of shape ``(M,)`` or of the input's leading
    shape -> ``((M,) tensor of dtype, True)``. The parameters are exact in their dtype (``top_k`` int32, ``top_p`` /
    ``min_p`` the binary32 values the class compares with)."""
    if value is None:
        return None, False
    if isinstance(value, torch.Tensor):
        t = value.to(device=device, dtype=dtype).reshape(-1)
        if t.numel() == 1 and M != 1:
            t = t.expand(M)
        if t.shape != (M,):
            raise ValueError(
                f"{name} must be a scalar or a tensor of one value, of shape ({M},) or of the input's leading shape "
                f"{lead}; got {tuple(value.shape)}"
            )
        return t.contiguous(), True
    v: Union[int, float]
    if dtype == torch.int32:
        v = int(value)
        if not (-(1 << 31) <= v < (1 << 31)):
            raise ValueError(f"{name}={value!r} out of the int32 range")
    else:
        v = float(value)
        if not (v >= lo) or v != v or v == float("inf"):
            raise ValueError(f"{name}={value!r} must be a finite number >= {lo}")
    return torch.full((M,), v, dtype=dtype, device=device), True


def _pow2_divisor(N: int, cap: int) -> int:
    """The largest power of two up to ``cap`` dividing ``N`` (the vector width of a row output)."""
    v = cap
    while v > 1 and N % v:
        v //= 2
    return v


def sampling_support(
    x: torch.Tensor,
    *,
    top_k: Optional[Union[int, torch.Tensor]] = None,
    top_p: Optional[Union[float, torch.Tensor]] = None,
    min_p: Optional[Union[float, torch.Tensor]] = None,
    order: str = "top_k_first",
    from_logits: bool = False,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    return_mask: bool = True,
    return_logprobs: bool = False,
    return_probs: bool = False,
    return_masked_logits: bool = False,
    enable_pdl: Optional[bool] = None,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
    early_finish: str = "mass",
) -> Support:
    """The kept set of every row of ``x`` (probabilities / unnormalised weights ``>= 0``, or logits with ``from_logits``)
    under top-k (the k-th largest weight, ties kept; ``k <= 0`` keeps nothing, ``k >= N`` everything), top-p (the
    smallest upward-closed set whose fixed-point mass reaches ``p`` of the row's -- of the top-k kept set's with
    ``order="top_k_first"``, of the whole row's with ``"joint"``; ``p >= 1`` keeps every positive weight) and min-p
    (``w >= mul(w_max, min_p)``), each a scalar or a per-row tensor. Returns :class:`Support`: the effective threshold,
    the kept count and mass, the row sum, the mask, for logits the renormalised log-probabilities of the kept tokens,
    and on request (one of) the renormalised probabilities ``cvt(mul(w_j, div(1, mass)))`` in the input dtype (``+0``
    outside the kept set; a kept set of zero mass renormalises to NaN, the declared ``div(1, +0)``) or, for logits, the
    masked logits (``-inf`` outside the kept set). ``enable_pdl=None`` follows the device; ``config`` / ``sm_count``
    force a configuration / another part's dispatcher pick (members of the enumerated space: the bits never depend on
    them). Bitwise identical for every member, every batch composition and every SM count.

    Raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for any input outside the class (dtype,
    device, architecture, vocabulary); ``early_finish`` is the selection's placement -- ``"mass"`` (the default: the
    early finish on the mass selection), ``"all"`` or ``""`` (the plain passes), bit-identical by construction and pinned
    against each other by the suite; ``ValueError`` for a malformed parameter; it never falls back."""
    from .support_kernel import get_compiled_kernel

    name = "logits" if from_logits else "probs"
    N = check_input(x, name, CLASS, SUPPORT_ARCHS)
    if order not in ORDERS:
        raise ValueError(f"order={order!r}: one of {ORDERS}")
    if top_k is None and top_p is None and min_p is None:
        raise ValueError("at least one of top_k / top_p / min_p")
    if return_logprobs and not from_logits:
        raise ValueError("log-probabilities are defined for logits (from_logits=True)")
    if return_probs and return_masked_logits:
        raise ValueError(
            "one input-dtype row output per call: return_probs or return_masked_logits"
        )
    if return_masked_logits and not from_logits:
        raise ValueError("masked logits are defined for logits (from_logits=True)")
    out_mode = "probs" if return_probs else ("masked" if return_masked_logits else "")
    if early_finish not in ("", "mass", "all"):
        raise ValueError(
            f"early_finish must be '', 'mass' or 'all', got {early_finish!r}"
        )
    try:
        x2 = _rows(x, N, name)
    except ValueError:
        x2 = x.reshape(-1, N).contiguous()  # a strided view: copied first, a placement
    M = x2.shape[0]
    lead = tuple(x.shape[:-1])
    dev = x.device
    k_t, use_k = _row_param(top_k, M, lead, dev, torch.int32, "top_k")
    p_t, use_p = _row_param(top_p, M, lead, dev, torch.float32, "top_p")
    m_t, use_m = _row_param(min_p, M, lead, dev, torch.float32, "min_p")
    filters = Filters(
        top_k=1 if use_k else None,
        top_p=1.0 if use_p else None,
        min_p=0.0 if use_m else None,
        order=order,
    )
    temp_t, c = _temperature(temperature, M, lead, dev) if from_logits else (None, 0.0)
    elem_bytes = x.element_size()
    if config is None:
        config = support_config(N, M, elem_bytes, sm_count or device_sm_count(dev))
    else:
        config.validate(N, elem_bytes)
        if config.rows_per_cta != 1:
            raise ValueError("the support kernel needs a config with one row per CTA")
        if config.blocks_per_cta(N) * BLOCK * elem_bytes > SMEM_BUDGET:
            raise ValueError(
                "the CTA's share of the row exceeds the staging budget: use a config with more CTAs per row"
            )
        if config.vec_elems > max_vec_elems(N, elem_bytes):
            raise ValueError("config vector width exceeds the row alignment")
    x2 = _aligned_rows(x2, config.vec_elems * elem_bytes)
    out_vec = _pow2_divisor(N, min(8, 16 // elem_bytes))
    mask_vec, lp_vec = (
        _pow2_divisor(N, 8),
        _pow2_divisor(N, 4),
    )  # <= V elements per lane per block
    thr = torch.empty(M, dtype=torch.float32, device=dev)
    count = torch.empty(M, dtype=torch.int64, device=dev)
    mass = torch.empty(M, dtype=torch.float32, device=dev)
    S = torch.empty(M, dtype=torch.float32, device=dev)
    mask = (
        torch.empty(M, N, dtype=torch.uint8, device=dev)
        if return_mask
        else _dummy(dev, torch.uint8).view(1, 8)
    )
    lp = (
        torch.empty(M, N, dtype=torch.float32, device=dev)
        if return_logprobs
        else _dummy(dev, torch.float32).view(1, 8)
    )
    out = (
        torch.empty(M, N, dtype=x.dtype, device=dev)
        if out_mode
        else _dummy(dev, x.dtype).view(1, 8)
    )
    if M > 0:
        kernel = get_compiled_kernel(
            _DTYPE_STR[x.dtype],
            N,
            config,
            from_logits,
            filters,
            temp_t is not None,
            return_mask,
            return_logprobs,
            mask_vec,
            lp_vec,
            _pdl(enable_pdl, dev),
            out_mode,
            out_vec,
            early_finish,
        )
        kernel(
            x2,
            k_t if use_k else _dummy(dev, torch.int32),
            p_t if use_p else _dummy(dev, torch.float32),
            m_t if use_m else _dummy(dev, torch.float32),
            temp_t if temp_t is not None else _dummy(dev, torch.float32),
            thr,
            count,
            mass,
            S,
            mask,
            lp,
            out,
            float(c),
            M,
            N,
        )
    full = (*lead, N)
    return Support(
        thr.view(lead),
        count.view(lead),
        mass.view(lead),
        S.view(lead),
        mask.view(torch.bool).view(full) if return_mask else None,
        lp.view(full) if return_logprobs else None,
        out.view(full) if return_probs else None,
        out.view(full) if return_masked_logits else None,
    )


def top_k_renorm_probs_batch_invariant(
    probs: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    *,
    enable_pdl: Optional[bool] = None,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
) -> torch.Tensor:
    """The class's ``top_k_renorm_probs`` (an internal entry; not exposed on the operator in this release): ``probs``
    renormalised over the
    top-k kept set of the class ``support.row.v1`` -- the k-th largest weight is the cut, ties at it are kept (the count
    may exceed ``k``), ``k <= 0`` keeps nothing (a row of ``+0``), ``k >= N`` everything -- as ``cvt(mul(p_j, div(1,
    mass)))`` with the kept mass the canonical tree over the kept weights (``lse.row``'s ``prob`` chain), ``+0`` outside
    the kept set, in the input dtype (bf16 / f16 / fp32). A kept set of zero mass (an all-zero row) renormalises to NaN."""
    return sampling_support(
        probs,
        top_k=top_k,
        return_mask=False,
        return_probs=True,
        enable_pdl=enable_pdl,
        config=config,
        sm_count=sm_count,
    ).probs  # type: ignore[return-value]


def top_p_renorm_probs_batch_invariant(
    probs: torch.Tensor,
    top_p: Union[float, torch.Tensor],
    *,
    enable_pdl: Optional[bool] = None,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
) -> torch.Tensor:
    """The class's ``top_p_renorm_probs`` (an internal entry; not exposed on the operator in this release): ``probs``
    renormalised over the
    top-p kept set of the class ``support.row.v1`` -- the smallest upward-closed set of weights whose declared
    fixed-point mass (``trunc(mul(p_j, div(2^44, S)))``, an exact integer sum) reaches ``trunc(mul(p, 2^44))``, ties at
    the cut kept, every positive weight when ``p >= 1`` or when the positive mass does not reach the target, zero
    weights never -- as ``cvt(mul(p_j, div(1, mass)))`` with the kept mass the canonical tree over the kept weights,
    ``+0`` outside the kept set, in the input dtype (bf16 / f16 / fp32)."""
    return sampling_support(
        probs,
        top_p=top_p,
        return_mask=False,
        return_probs=True,
        enable_pdl=enable_pdl,
        config=config,
        sm_count=sm_count,
    ).probs  # type: ignore[return-value]


def top_k_mask_logits_batch_invariant(
    logits: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    *,
    enable_pdl: Optional[bool] = None,
    config: Optional[LSEConfig] = None,
    sm_count: Optional[int] = None,
) -> torch.Tensor:
    """The class's ``top_k_mask_logits`` (an internal entry; not exposed on the operator in this release): ``logits``
    with ``-inf`` outside
    the top-k kept set of the class ``support.row.v1`` over the ``lse.row`` weights ``e_j = ex2((x_j - m) log2(e))`` of
    the row (the k-th largest weight is the cut, ties kept; the logits' own values are untouched inside the set), in the
    input dtype (bf16 / f16 / fp32)."""
    return sampling_support(
        logits,
        top_k=top_k,
        from_logits=True,
        return_mask=False,
        return_masked_logits=True,
        enable_pdl=enable_pdl,
        config=config,
        sm_count=sm_count,
    ).masked_logits  # type: ignore[return-value]
