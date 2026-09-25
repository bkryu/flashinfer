# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# Qwen3.8-27B one-launch decoder on SM120 (FlashInfer experimental track).

"""Qwen3.8-27B one-launch decoder (SM120, CuTe DSL): experimental backend package. Public entry points live in
`flashinfer.qwen3_ops.qwen38_megakernel` (thin, `@flashinfer_experimental_api`); this package holds the kernel, the
per-layer device tables and the scratch management. Import lazily."""

from .decoder_host import (  # noqa: F401
    LAYER_KEYS_ATTN,
    LAYER_KEYS_COMMON,
    LAYER_KEYS_GDN,
    PHASES,
    DecoderSpec,
    DecoderTables,
    decoder_entry,
    kv_splits_for,
    slack_buffer,
)
from ._host_utils import MAX_M, TILE, build_rope_cache, interleave_gate_up, rowstat  # noqa: F401
from .support import qwen38_megakernel_supported, qwen38_megakernel_unsupported_reason  # noqa: F401
