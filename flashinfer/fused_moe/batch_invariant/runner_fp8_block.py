# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant fp8-block MoE backend of the unified MoE layer (``BatchInvariantFp8BlockConfig`` ->
``BatchInvariantFp8BlockRunner``): the fp8 formats with block scales under one backend key, grouped as TRTLLM-Gen groups
them (``TrtllmFp8BlockConfig`` / ``trtllm_fp8_block``). The quantization pair of the layer's ``QuantConfig`` selects the
class family and its runner: MXFP8 x MXFP8 runs the classes ``moe.mxfp8.<act>`` (:mod:`.runner_mxfp8`); the DeepSeekFp8 x
DeepSeekFp8 pair (the block-scale classes) arrives with the block-scale MoE PR. Every pair keeps its own kernels, classes,
fingerprints and pack conventions: the grouping is the surface alone.

The layer admits the backend for the declared pairs only (``supports_quant``); constructing the runner for another pair
raises the mode's error naming the pairs served, as the config's ``prepare_weights`` / ``prepare_activations`` do.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Type

import torch

from ...autotuner import TuningConfig
from ...batch_invariant.support import BatchInvariantUnsupportedError
from ..api import (
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    MoEConfig,
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


def unsupported_pair(quant: QuantConfig) -> BatchInvariantUnsupportedError:
    """The refusal of a quantization pair the backend does not serve (the layer filters it before construction; a runner
    constructed for it or a ``prepare_*`` call with it raises this)."""
    served = " or ".join(
        f"QuantConfig(weight={w.name}, activation={a.name})"
        for w, a in BatchInvariantFp8BlockRunner.supported_quant_variants
    )
    return BatchInvariantUnsupportedError(
        f"batch_invariant=True: the fp8-block MoE backend (`{BatchInvariantFp8BlockRunner.backend_key}`) serves {served}, "
        f"not weight={quant.weight.name}, activation={quant.activation.name}; the DeepSeekFp8 x DeepSeekFp8 pair (the "
        "block-scale classes) arrives with the block-scale MoE PR. The batch-invariant mode never falls back to another "
        "kernel."
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
    supported_quant_variants = (PAIR_MXFP8,)
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
        return None

    def __new__(cls, config: MoEConfig, device: Any) -> "BatchInvariantFp8BlockRunner":
        # the backend's class constructs the pair's runner; a pair without one is refused here (the layer never asks:
        # it admits the declared pairs only)
        if cls is BatchInvariantFp8BlockRunner:
            impl = cls.implementation(config.quant.pair)
            if impl is None:
                raise unsupported_pair(config.quant)
            cls = impl
        return super().__new__(cls)

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
