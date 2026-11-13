# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant fp4 MoE backend of the unified MoE layer (``BatchInvariantFp4Config`` -> ``BatchInvariantFp4Runner``).

One backend for the 4-bit-weight quantization pairs, grouped as ``TrtllmFp4Config`` groups them: ``QuantConfig(weight=...,
activation=...)`` selects the pair, and the runner delegates every call to the pair's implementation (``PAIRS``). The
implementations share the backend's weight-view key, so ``MoEWeightPack.prepare_for(BatchInvariantFp4Runner.backend_key, ...)``
takes the view ``BatchInvariantFp4Config.prepare_weights`` built for the configured pair. The pairs served here:

* NVFP4 x NVFP4 -- :mod:`.runner_nvfp4`, the W4A4 classes ``moe.nvfp4.<act>``;
* MXFP4 x MXFP4 -- :mod:`.runner_mxfp4`, the classes ``moe.mxfp4.<act>`` (UE8M0 block scales on both operands).

MXFP4 x MXFP8 arrives with the mixed MoE PR. A pair outside ``supported_quant_variants`` is skipped by ``MoELayer``'s
candidate filter (``supports_quant``) and refused by the runner's ``check_support`` and by the config's ``prepare_weights`` /
``prepare_activations`` with :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`, naming the served pairs.
The kernel classes keep their registry names (``fingerprint("fused_moe_nvfp4", ...)``): the backend's key names the group, not
a class.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from ...autotuner import TuningConfig
from ...batch_invariant.support import BatchInvariantUnsupportedError
from ..api import MoEActivationPack, MoEWeightPack, QuantConfig, QuantFormat
from ..runners import MoERunner
from .runner_mxfp4 import BatchInvariantMoeMxfp4Runner
from .runner_nvfp4 import BatchInvariantMoeNvfp4Runner

#: The backend's key: the name of its weight view and of its autotuner records, shared by the pair implementations.
BACKEND_KEY = "batch_invariant_fp4"
#: The quantization pairs the backend serves, each by its implementation runner (in the order they are advertised).
# the implementation runners (MoERunner subclasses constructed with (config, device); typed loosely: the base class's
# constructor takes no arguments and the delegated entries are the implementations')
PAIRS: Dict[Tuple[QuantFormat, QuantFormat], Type[Any]] = {
    (QuantFormat.NVFP4, QuantFormat.NVFP4): BatchInvariantMoeNvfp4Runner,
    (QuantFormat.MXFP4, QuantFormat.MXFP4): BatchInvariantMoeMxfp4Runner,
}
#: The pairs of the group that a later PR of the series brings (named by the refusal).
PENDING_PAIRS = "MXFP4 x MXFP8 arrives with the mixed MoE PR"


def pair_name(pair: Tuple[QuantFormat, QuantFormat]) -> str:
    return f"{pair[0].name} x {pair[1].name}"


def unsupported_pair(quant: QuantConfig) -> BatchInvariantUnsupportedError:
    """The refusal of a quantization pair the fp4 backend does not serve: it names the served pairs (and the pending ones)."""
    served = ", ".join(pair_name(p) for p in PAIRS)
    pending = f"; {PENDING_PAIRS}" if PENDING_PAIRS else ""
    return BatchInvariantUnsupportedError(
        f"batch_invariant=True: the fp4 MoE backend `{BACKEND_KEY}` (BatchInvariantFp4Config) does not serve the "
        f"quantization weight={quant.weight.name}, activation={quant.activation.name}; it serves {served}{pending}. "
        "The batch-invariant mode never falls back to another kernel; call without batch_invariant=True to use the "
        "regular backends."
    )


def _union(attr: str) -> tuple:
    return tuple(
        dict.fromkeys(v for impl in PAIRS.values() for v in getattr(impl, attr))
    )


class BatchInvariantFp4Runner(MoERunner):
    """The unified layer's runner for the batch-invariant fp4 MoE backend: one class per quantization pair, each call delegated
    to the pair's implementation runner (``PAIRS``)."""

    backend_key = BACKEND_KEY
    supported_quant_variants = tuple(PAIRS)
    supported_output_formats = (QuantFormat.BF16,)
    supported_routing_modes = _union("supported_routing_modes")
    supported_activation_classes = _union("supported_activation_classes")
    supported_activation_classes_by_quant = {
        pair: tuple(impl.supported_activation_classes) for pair, impl in PAIRS.items()
    }
    # the NVFP4 pair consumes a pre-quantized pack's per-token scale (the class's own ``pts``); the MX pair has none and
    # rejects the flag itself (``check_support`` of its implementation)
    supports_per_token_scale = any(
        impl.supports_per_token_scale for impl in PAIRS.values()
    )
    supports_expert_parallelism = False

    def __init__(self, config, device) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        if self.device.index is None and self.device.type == "cuda":
            self.device = torch.device("cuda", torch.cuda.current_device())
        impl = PAIRS.get(config.quant.pair)
        # an unserved pair is refused by check_support (the layer's filter never constructs this case)
        self._impl: Optional[Any] = (
            impl(config, self.device) if impl is not None else None
        )

    @property
    def impl(self) -> Any:
        """The configured pair's implementation runner."""
        if self._impl is None:
            raise unsupported_pair(self.config.quant)
        return self._impl

    @property
    def tuning_config(self) -> TuningConfig:
        return self.impl.tuning_config

    # ---- support ---------------------------------------------------------------------------------------------------
    def _check_support(self) -> None:
        if self._impl is None:
            raise unsupported_pair(self.config.quant)
        # the pair's own shard refusal first (the class's terms), then the layer's generic rules, then the pair's
        self._impl._check_experts()
        super()._check_support()
        self._impl.check_support()

    def _build(self) -> None:
        self.impl.build()

    # ---- inputs ----------------------------------------------------------------------------------------------------
    def _accepts_shape(self, act: MoEActivationPack) -> bool:
        return self.impl._accepts_shape(act)

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._require_built()
        self._validate_pack_contract(act)
        return self.impl.pack_inputs(act, weights)

    # ---- tactics ---------------------------------------------------------------------------------------------------
    def get_valid_tactics(self, inputs: List[torch.Tensor], profile) -> List[Any]:
        self._require_built()
        return self.impl.get_valid_tactics(inputs, profile)

    def _cache_key_extras(self) -> tuple:
        return self.impl._cache_key_extras()

    # ---- launch ----------------------------------------------------------------------------------------------------
    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        self._require_built()
        return self.impl.forward(
            inputs, tactic=tactic, do_preparation=do_preparation, **kwargs
        )
