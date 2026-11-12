# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant fp8-block MoE backend of the unified MoE layer (``BatchInvariantFp8BlockConfig`` ->
``BatchInvariantFp8BlockRunner``): the fp8 formats with block scales under one backend key, grouped as TRTLLM-Gen groups
them (``TrtllmFp8BlockConfig`` / ``trtllm_fp8_block``). The quantization pair of the layer's ``QuantConfig`` selects the
class family and its runner: MXFP8 x MXFP8 runs the classes ``moe.mxfp8.<act>`` (:mod:`.runner_mxfp8`), DeepSeekFp8 x
DeepSeekFp8 the DeepSeek-style block-scale classes ``moe.e4m3bs.<act>`` (:mod:`.runner_fp8bs`). Every pair keeps its own
kernels, classes, fingerprints and pack conventions: the grouping is the surface alone.

The layer admits the backend for the declared pairs only (``supports_quant``); constructing the runner for another pair
raises the mode's error naming the pairs served, as the config's ``prepare_weights`` / ``prepare_activations`` do.
"""

from __future__ import annotations

from typing import Any, ClassVar, List, Optional, Tuple, Type

import torch

from ...autotuner import TuningConfig
from ...batch_invariant.support import BatchInvariantUnsupportedError
from ..api import (
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    MoEConfig,
    MoEWeightPack,
    MoEActivationPack,
    PowLU,
    QuantConfig,
    QuantFormat,
    ReLU,
    ReLU2,
    RoutingInputMode,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)
from ..runners import MoERunner

#: The quantization pairs of the fp8 formats with block scales.
PAIR_MXFP8 = (QuantFormat.MXFP8, QuantFormat.MXFP8)
PAIR_DEEPSEEK_FP8 = (QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8)


def unsupported_pair(
    quant: QuantConfig,
    *,
    served: Optional[Tuple[Tuple[QuantFormat, QuantFormat], ...]] = None,
    of: str = "the fp8-block MoE backend",
) -> BatchInvariantUnsupportedError:
    """The refusal of a quantization pair ``of`` does not serve (by default the backend, whose declared pairs the layer
    filters on before construction; a runner constructed for another pair or a ``prepare_*`` call with it raises this)."""
    if served is None:
        served = BatchInvariantFp8BlockRunner.supported_quant_variants
    names = " or ".join(
        f"QuantConfig(weight={w.name}, activation={a.name})" for w, a in served
    )
    return BatchInvariantUnsupportedError(
        f"batch_invariant=True: {of} (`{BatchInvariantFp8BlockRunner.backend_key}`) serves {names}, not "
        f"weight={quant.weight.name}, activation={quant.activation.name}. The batch-invariant mode never falls back to "
        "another kernel."
    )


class BatchInvariantFp8BlockRunner(MoERunner):
    """The unified layer's runner for the batch-invariant fp8-block MoE classes: constructing it for a served pair gives
    the pair's runner (a subclass, ``implementation``), which carries the pair's inputs, tactics and launch."""

    backend_key = "batch_invariant_fp8_block"
    supported_routing_modes = (
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    )
    supported_quant_variants: ClassVar[Tuple[Tuple[QuantFormat, QuantFormat], ...]] = (
        PAIR_MXFP8,
        PAIR_DEEPSEEK_FP8,
    )
    supported_output_formats = (QuantFormat.BF16,)
    # every typed activation of the layer, for every pair: the bf16 classes' chains on the fp32 accumulators
    # (``spec.activation_of``)
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
    # Expert-parallel shards (a local expert range with ids outside it) are the EP classes of a later PR.
    supports_expert_parallelism = False

    @classmethod
    def implementation(
        cls, pair: Tuple[QuantFormat, QuantFormat]
    ) -> Optional[Type["BatchInvariantFp8BlockRunner"]]:
        """The runner of a quantization pair; ``None`` for a pair the backend does not serve."""
        if pair == PAIR_MXFP8:
            from .runner_mxfp8 import BatchInvariantMxfp8Runner

            return BatchInvariantMxfp8Runner
        if pair == PAIR_DEEPSEEK_FP8:
            from .runner_fp8bs import BatchInvariantDeepSeekFp8Runner

            return BatchInvariantDeepSeekFp8Runner
        return None

    def __new__(
        cls, config: Optional[MoEConfig] = None, device: Any = None
    ) -> "BatchInvariantFp8BlockRunner":
        # the backend's class constructs the pair's runner; a pair without one is refused here (the layer never asks:
        # it admits the declared pairs only); without a config (the registry's contract tests) a bare instance of the class
        if cls is BatchInvariantFp8BlockRunner and config is not None:
            impl = cls.implementation(config.quant.pair)
            if impl is None:
                raise unsupported_pair(config.quant)
            cls = impl
        return super().__new__(cls)

    # the runners' lifecycle and pack-contract rules on the backend's class itself: a served pair constructs its
    # implementation runner, which overrides these; a bare instance of this class serves no pair
    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._require_built()
        self._validate_pack_contract(act)
        raise unsupported_pair(self.config.quant)

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile: Any) -> List[Any]:
        self._require_built()
        raise unsupported_pair(self.config.quant)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._require_built()
        raise unsupported_pair(self.config.quant)

    def __init__(self, config: MoEConfig, device: Any) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        if self.device.index is None and self.device.type == "cuda":
            self.device = torch.device("cuda", torch.cuda.current_device())
        # exact shapes: the dispatcher's pick is a function of the token count (every tactic computes the same bits)
        self.tuning_config = TuningConfig(use_cuda_graph=True)
        self._part: Optional[Tuple[int, str]] = None

    def _activation(self):
        """The chain of the layer's activation (``spec.ActSpec``: gating, chain name, binary32 parameters)."""
        from .spec import activation_of

        return activation_of(self.config.activation)

    def _build(self) -> None:
        from ...gemm.batch_invariant.api import part_of

        self._part = part_of(self.device)


__all__ = [
    "PAIR_DEEPSEEK_FP8",
    "PAIR_MXFP8",
    "BatchInvariantFp8BlockRunner",
    "unsupported_pair",
]
