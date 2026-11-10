# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 x MXFP8 pair of the batch-invariant fp8-block MoE backend of the unified MoE layer
(``BatchInvariantFp8BlockConfig`` under ``QuantConfig(MXFP8, MXFP8)`` -> ``BatchInvariantFp8BlockRunner`` ->
``BatchInvariantMxfp8Runner``, the pair's runner).

The runner serves the classes ``moe.mxfp8.<act>`` (MXFP8 experts and activations, every typed activation of the layer)
on SM100 / SM103 / SM107, through the dense MXFP8 class of the part (the SM107 instruction folds 64 elements per step, the
SM100 / SM103 one 32: a dimension of the specification). It takes the layer's canonical MXFP8 activation pack (e4m3
``hidden_states_q [T, H]`` with the token-major UE8M0 ``hidden_states_scale [T, H / 32]``, as ``TrtllmFp8BlockConfig`` /
``CutlassMxfp8Config`` / this config's ``prepare_activations`` produce it) and every routing input mode: ``FromLogits``
runs the class's own routing kernel, ``PackedPrecomputed`` / ``UnpackedPrecomputed`` take the caller's expert ids and
weights. The combine is the slot-order fp32 fold (``use_fused_finalize`` is ignored, ``do_finalize=True`` only), the
tactics are the ranked grouped-GEMM records of the dense MX heuristic over both record spaces -- the padded body's and the
swapped decode body's (the weights on the MMA's M side, the group's tokens a narrow tile; `schedule_mxfp8`) -- any tactic
gives the same bits, and the packing of the expanded rows follows the first picks (the dense packing when either is a
swapped record, else the padded one: a placement decided by the token count, so the routing-and-plan launch takes the
matching form). An unsupported configuration raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`
instead of falling back.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from ...batch_invariant.support import check_device, unsupported
from ..api import MoEActivationPack, MoEWeightPack, RoutingInputMode
from ..runners import _validate_logits_inputs, _validate_prerouted_inputs
from .runner import _configs_from_tactic, _tactic
from .runner_fp8_block import PAIR_MXFP8, BatchInvariantFp8BlockRunner
from .spec_mxfp8 import BLOCK, CLASS

#: The compute capabilities the classes are stated for (the dense MXFP8 GEMM class's parts).
MOE_MXFP8_ARCHS: Tuple[Tuple[int, int], ...] = ((10, 0), (10, 3), (10, 7))
#: Positions of the packed inputs: the quantized activations and their scales, the expert ids, the routing weights, the
#: two weights with their scales and the output.
_XQ, _XSF, _IDS, _W, _W1, _W1SF, _W2, _W2SF, _OUT = range(9)


class BatchInvariantMxfp8Runner(BatchInvariantFp8BlockRunner):
    """The fp8-block backend's runner for the MXFP8 x MXFP8 pair: the batch-invariant MXFP8 MoE classes."""

    supported_quant_variants = (PAIR_MXFP8,)

    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        self._in_fmt: Optional[str] = None

    # ---- support ---------------------------------------------------------------------------------------------------
    def _check_support(self) -> None:
        super()._check_support()
        from .spec import METHOD_OF_FLASHINFER

        cfg = self.config
        check_device(self.device, CLASS, MOE_MXFP8_ARCHS)
        try:
            self._activation()
        except ValueError as e:
            raise unsupported(
                CLASS,
                f"activation {cfg.activation!r}",
                "one of the layer's typed activations",
            ) from e
        experts = cfg.experts
        if experts.num_fused_shared_experts:
            raise unsupported(
                CLASS,
                "fused shared experts",
                "num_fused_shared_experts = 0 (a shared expert is one more routed slot)",
            )
        routing = cfg.routing
        if experts.intermediate_size <= 0 or experts.intermediate_size % BLOCK:
            raise unsupported(
                CLASS,
                f"intermediate_size {experts.intermediate_size}",
                f"a positive multiple of {BLOCK}",
            )
        if not 1 <= routing.top_k <= min(routing.num_experts, 16):
            raise unsupported(
                CLASS, f"top_k {routing.top_k}", "1 <= top_k <= min(num_experts, 16)"
            )
        if not 1 <= routing.num_experts <= 1024:
            raise unsupported(
                CLASS, f"num_experts {routing.num_experts}", "1 <= num_experts <= 1024"
            )
        # group-limited routing (n_group / topk_group > 1) is refused only from logits (the class has no grouped routing
        # term); with pre-routed ids the group structure is irrelevant, the output is the ungrouped configuration's
        self._method = METHOD_OF_FLASHINFER.get(routing.method)
        if not cfg.finalize.do_finalize:
            raise unsupported(
                CLASS,
                "do_finalize=False",
                "do_finalize=True (the slot-order combine is part of the class)",
            )

    def _build(self) -> None:
        super()._build()
        from .api_mxfp8 import mx_class_of

        self._in_fmt = mx_class_of(self.device)

    # ---- inputs ----------------------------------------------------------------------------------------------------
    def _accepts_shape(self, act: MoEActivationPack) -> bool:
        x = act.hidden_states_q
        return (
            x.ndim == 2
            and x.dtype == torch.float8_e4m3fn
            and x.shape[1] % BLOCK == 0
            and act.hidden_states_scale is not None
        )

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._validate_pack_contract(act)
        self._require_built()
        from .api import route, route_plan
        from .api_mxfp8 import MoEWeightsMXFP8, check_shape_mxfp8

        cfg = self.config
        x = act.hidden_states_q
        if x.dtype != torch.float8_e4m3fn or x.ndim != 2:
            raise unsupported(
                CLASS,
                f"hidden_states_q {x.dtype} of rank {x.ndim}",
                "the MXFP8 activation pack: an e4m3 [num_tokens, hidden_size] with its UE8M0 scales",
            )
        sf = act.hidden_states_scale
        if sf is None:
            raise unsupported(
                CLASS,
                "no hidden_states_scale",
                "the token-major UE8M0 scales [num_tokens, hidden_size / 32]",
            )
        x = x.contiguous()
        T, H = x.shape
        if sf.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise unsupported(
                CLASS, f"hidden_states_scale {sf.dtype}", "uint8 or UE8M0 scale bytes"
            )
        sf = (
            sf.view(torch.uint8).reshape(T, -1 if T else H // BLOCK).contiguous()
        )  # the zero-token pack has no inferable column count
        if sf.shape[1] != H // BLOCK:
            raise unsupported(
                CLASS,
                f"hidden_states_scale of {sf.shape[1]} columns",
                f"{H // BLOCK} token-major UE8M0 scales per row (the linear layout; a swizzled buffer is not this pack)",
            )
        k, E = cfg.routing.top_k, cfg.routing.num_experts
        view = weights.get_view(self.backend_key)
        w = MoEWeightsMXFP8.from_view(view)
        if (
            w.hidden_size != H
            or w.num_experts != E
            or w.intermediate_size != cfg.experts.intermediate_size
        ):
            raise unsupported(
                CLASS,
                f"weights (E {w.num_experts}, I {w.intermediate_size}, H {w.hidden_size}) for hidden {H}, E {E}, I {cfg.experts.intermediate_size}",
                "weights prepared for this configuration",
            )
        check_shape_mxfp8(H, w.intermediate_size, E, k)
        a = self._activation()
        if (w.gated, w.act, w.params) != (a.gated, a.act, a.params):
            raise unsupported(
                CLASS,
                f"{type(cfg.activation).__name__} with {'gated' if w.gated else 'ungated'} weights of chain {w.act!r}",
                "weights prepared with the layer's activation",
            )
        plan: List[torch.Tensor] = []
        mode = act.routing_input_mode
        if mode is RoutingInputMode.FromLogits:
            _validate_logits_inputs(act, T, E, type(self).__name__)
            if cfg.routing.n_group not in (
                None,
                0,
                1,
            ) or cfg.routing.topk_group not in (
                None,
                0,
                1,
            ):
                raise unsupported(
                    CLASS,
                    "group-limited routing (n_group > 1) from logits",
                    "n_group in (None, 0, 1) from logits, or pre-routed ids for grouped routing",
                )
            if self._method is None:
                raise unsupported(
                    CLASS,
                    f"routing method {cfg.routing.method!r} from logits",
                    "Default, Renormalize, RenormalizeNaive, Sigmoid, SigmoidRenorm or DeepSeekV3 with n_group <= 1",
                )
            bias = act.routing_bias
            if (bias is not None) != self._method.bias:
                raise unsupported(
                    CLASS,
                    "a routing bias" if bias is not None else "no routing bias",
                    "a bias exactly for the methods that select on sigmoid(l) + bias",
                )
            scale = cfg.routing.routed_scaling_factor if self._method.scaled else None
            if self._method.scaled and scale is None:
                scale = 1.0
            from .api_mxfp8 import GROUP_ALIGN
            from .blocks.scheduler import MAX_E
            from .route_plan import ROUTE_PLAN_MAX_SLOTS, ROUTE_PLAN_MAX_TOKENS

            if (
                0 < T <= ROUTE_PLAN_MAX_TOKENS
                and T * k <= ROUTE_PLAN_MAX_SLOTS
                and cfg.routing.num_experts <= MAX_E
            ):
                # routing and the permutation plan in one launch (a placement: the same bits and integers), in the form of
                # the packing the ranked records run on at this token count; the plan's group tables ride along (the GEMMs
                # load them in place of a per-CTA scan)
                from .api import group_tables_buffer

                tables = group_tables_buffer(
                    cfg.routing.num_experts, act.routing_logits.device
                )
                if self._packing(T, H) == "dense":
                    ids, wts, counts, dest, src, dest_pad, pos_pad = route_plan(
                        act.routing_logits,
                        k,
                        self._method,
                        bias,
                        scale,
                        dual=True,
                        pad_align=GROUP_ALIGN,
                        tables=tables,
                    )
                    plan = [counts, dest, src, dest_pad, pos_pad, tables]
                else:
                    ids, wts, counts, dest, src = route_plan(
                        act.routing_logits,
                        k,
                        self._method,
                        bias,
                        scale,
                        align=GROUP_ALIGN,
                        tables=tables,
                    )
                    plan = [counts, dest, src, tables]
            else:
                ids, wts = route(act.routing_logits, k, self._method, bias, scale)
        else:
            _validate_prerouted_inputs(
                act,
                T,
                k,
                type(self).__name__,
                allowed_weights_dtypes=(torch.bfloat16, torch.float32),
                require_contiguous=True,
            )
            ids = act.topk_ids.contiguous()
            wts = act.topk_weights
            if mode is RoutingInputMode.PackedPrecomputed:
                # the packed word carries the weight as bf16: the rounded weights are the class's inputs in this mode
                wts = wts.to(torch.bfloat16)
            wts = wts.to(torch.float32).contiguous()
        out = torch.empty(T, H, dtype=torch.bfloat16, device=x.device)
        return [x, sf, ids, wts, w.w1, w.w1_sf, w.w2, w.w2_sf, out] + plan

    # ---- tactics ---------------------------------------------------------------------------------------------------
    def _ranked(self, T: int, H: Optional[int] = None) -> Tuple[list, list]:
        """The ranked records of both GEMMs over both record spaces at ``T`` tokens (`schedule_mxfp8.ranked_mx`), trimmed to
        the packing the first picks run on: on the padded packing the swapped records cannot run, so they leave the lists."""
        from .blocks.epilogue import GroupedChain
        from .schedule import TOP_K
        from .schedule_mxfp8 import packing_for, ranked_mx

        cfg = self.config
        E, k, I = (
            cfg.routing.num_experts,
            cfg.routing.top_k,
            cfg.experts.intermediate_size,
        )
        H = H if H is not None else self._hidden
        a = self._activation()
        chain1 = GroupedChain(a.chain, a.act, "bf16", a.params)
        sm_count, family = self._part if self._part is not None else (None, None)
        in_fmt = self._in_fmt or "mxfp8"
        r1, r2 = ranked_mx(T, k, H, I, E, chain1, in_fmt, sm_count, family, TOP_K)
        if packing_for(r1[0], r2[0]) == "padded":
            r1 = [c for c in r1 if not c.swap_ab]
            r2 = [c for c in r2 if not c.swap_ab]
        return r1, r2

    def _packing(self, T: int, H: int) -> str:
        """The packing of the expanded rows at ``T`` tokens: the first picks' (`schedule_mxfp8.packing_for`)."""
        from .schedule_mxfp8 import packing_for

        r1, r2 = self._ranked(T, H)
        return packing_for(r1[0], r2[0])

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile) -> List[Any]:
        self._require_built()
        self._hidden = int(inputs[_XQ].shape[1])
        r1, r2 = self._ranked(int(inputs[_XQ].shape[0]))
        return [_tactic(c1, c2) for c1 in r1 for c2 in r2]

    def _cache_key_extras(self) -> tuple:
        return super()._cache_key_extras() + (
            "moe.mxfp8-v2",
            self._in_fmt or "",
        )  # v2: the tactic carries the swapped flag

    # ---- launch ----------------------------------------------------------------------------------------------------
    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        self._require_built()
        from .api_mxfp8 import MoEWeightsMXFP8, fused_moe_mxfp8

        xq, xsf, ids, wts, w1, w1_sf, w2, w2_sf, out = inputs[:9]
        plan = tuple(inputs[9:]) if len(inputs) >= 13 else None  # [.., tables]
        self._hidden = int(xq.shape[1])
        T = int(xq.shape[0])
        if tactic == -1 or tactic is None:
            r1, r2 = self._ranked(T)
            c1, c2 = r1[0], r2[0]
        else:
            c1, c2 = _configs_from_tactic(tuple(tactic))
        a = self._activation()
        w = MoEWeightsMXFP8(w1, w1_sf, w2, w2_sf, a.gated, a.act, a.params)
        # the packing follows the first picks at this token count (the plan, when given, was built in its form); a tactic
        # of the padded packing is a pair of dense records, one of the dense packing any pair
        fused_moe_mxfp8(
            (xq, xsf),
            ids,
            wts,
            w,
            out=out,
            config1=c1,
            config2=c2,
            in_fmt=self._in_fmt,
            plan=plan,
            path=self._packing(T, self._hidden),
        )
        return out
