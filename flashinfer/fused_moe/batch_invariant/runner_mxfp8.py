# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant MXFP8 MoE backend of the unified MoE layer (``BatchInvariantMoeMxfp8Config`` ->
``BatchInvariantMoeMxfp8Runner``).

The runner serves the classes ``moe.mxfp8.<act>`` (MXFP8 experts and activations, every typed activation of the layer)
on SM100 / SM103 / SM107, through the dense MXFP8 class of the part (the SM107 instruction folds 64 elements per step, the
SM100 / SM103 one 32: a dimension of the specification). It takes the layer's canonical MXFP8 activation pack (e4m3
``hidden_states_q [T, H]`` with the token-major UE8M0 ``hidden_states_scale [T, H / 32]``, as ``TrtllmFp8BlockConfig`` /
``CutlassMxfp8Config`` / this config's ``prepare_activations`` produce it) and every routing input mode: ``FromLogits``
runs the class's own routing kernel, ``PackedPrecomputed`` / ``UnpackedPrecomputed`` take the caller's expert ids and
weights. The combine is the slot-order fp32 fold (``use_fused_finalize`` is ignored, ``do_finalize=True`` only), the
tactics are the ranked grouped-GEMM records of the dense MX heuristic (any tactic gives the same bits), and an
unsupported configuration raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` instead of falling back.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from ...autotuner import TuningConfig
from ...batch_invariant.support import check_device, unsupported
from ..api import (
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    MoEActivationPack,
    MoEWeightPack,
    PowLU,
    QuantFormat,
    ReLU,
    ReLU2,
    RoutingInputMode,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)
from ..runners import MoERunner, _validate_logits_inputs, _validate_prerouted_inputs
from .runner import _configs_from_tactic, _tactic
from .spec_mxfp8 import BLOCK, CLASS

#: The compute capabilities the classes are stated for (the dense MXFP8 GEMM class's parts).
MOE_MXFP8_ARCHS: Tuple[Tuple[int, int], ...] = ((10, 0), (10, 3), (10, 7))
#: Positions of the packed inputs: the quantized activations and their scales, the expert ids, the routing weights, the
#: two weights with their scales and the output.
_XQ, _XSF, _IDS, _W, _W1, _W1SF, _W2, _W2SF, _OUT = range(9)


class BatchInvariantMoeMxfp8Runner(MoERunner):
    """The unified layer's runner for the batch-invariant MXFP8 MoE classes."""

    backend_key = "batch_invariant_mxfp8"
    supported_routing_modes = (
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    )
    supported_quant_variants = ((QuantFormat.MXFP8, QuantFormat.MXFP8),)
    supported_output_formats = (QuantFormat.BF16,)
    # every typed activation of the layer: the bf16 classes' chains on the dequantised accumulators (``spec.activation_of``)
    supported_activation_classes = (
        SwiGLU,
        SwiGLUStep,
        GeGLU,
        GeGLUTanh,
        SiTU,
        PowLU,
        ReLU2,
        Identity,
        GELU,
        ReLU,
        SiLU,
    )
    supports_expert_parallelism = False

    def __init__(self, config, device) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        if self.device.index is None and self.device.type == "cuda":
            self.device = torch.device("cuda", torch.cuda.current_device())
        # exact shapes: the dispatcher's pick is a function of the token count (every tactic computes the same bits)
        self.tuning_config = TuningConfig(use_cuda_graph=True)
        self._part: Optional[Tuple[int, str]] = None
        self._in_fmt: Optional[str] = None

    # ---- support ---------------------------------------------------------------------------------------------------
    def _activation(self):
        from .spec import activation_of

        return activation_of(self.config.activation)

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
        if cfg.quant.pair != (QuantFormat.MXFP8, QuantFormat.MXFP8):
            raise unsupported(
                CLASS,
                f"quant {cfg.quant!r}",
                "QuantConfig(weight=MXFP8, activation=MXFP8)",
            )
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
        from ...gemm.batch_invariant.api import part_of
        from .api_mxfp8 import mx_class_of

        self._part = part_of(self.device)
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
        from .api import route
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
        return [x, sf, ids, wts, w.w1, w.w1_sf, w.w2, w.w2_sf, out]

    # ---- tactics ---------------------------------------------------------------------------------------------------
    def _ranked(self, T: int) -> Tuple[list, list]:
        from .blocks.epilogue import GroupedChain
        from .schedule import TOP_K, active_experts, rows_per_expert
        from .schedule_mxfp8 import GEMM2_CHAIN, rank_grouped_mx

        cfg = self.config
        E, k, I = (
            cfg.routing.num_experts,
            cfg.routing.top_k,
            cfg.experts.intermediate_size,
        )
        H = self._hidden
        rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
        a = self._activation()
        chain1 = GroupedChain(a.chain, a.act, "bf16", a.params)
        n1 = 2 * I if a.gated else I
        sm_count, family = self._part if self._part is not None else (None, None)
        in_fmt = self._in_fmt or "mxfp8"
        return (
            rank_grouped_mx(
                rows, n1, H, E, chain1, in_fmt, sm_count, family, TOP_K, n_active=n_act
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
                TOP_K,
                n_active=n_act,
            ),
        )

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile) -> List[Any]:
        self._require_built()
        self._hidden = int(inputs[_XQ].shape[1])
        r1, r2 = self._ranked(int(inputs[_XQ].shape[0]))
        return [_tactic(c1, c2) for c1 in r1 for c2 in r2]

    def _cache_key_extras(self) -> tuple:
        return super()._cache_key_extras() + ("moe.mxfp8-v1", self._in_fmt or "")

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
        self._hidden = int(xq.shape[1])
        if tactic == -1 or tactic is None:
            r1, r2 = self._ranked(int(xq.shape[0]))
            c1, c2 = r1[0], r2[0]
        else:
            c1, c2 = _configs_from_tactic(tuple(tactic))
        a = self._activation()
        w = MoEWeightsMXFP8(w1, w1_sf, w2, w2_sf, a.gated, a.act, a.params)
        fused_moe_mxfp8(
            (xq, xsf), ids, wts, w, out=out, config1=c1, config2=c2, in_fmt=self._in_fmt
        )
        return out
