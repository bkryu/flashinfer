# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# Qwen3.8-27B one-launch decoder on SM120 (FlashInfer experimental track).

"""Model-specific decode operators for the Qwen3 family. Experimental: importing this package is cheap (no kernel or
JIT imports); calling the `qwen38_*` functions is the opt-in and warns once (ExperimentalWarning)."""

from .qwen38_megakernel import (  # noqa: F401
    Qwen38MegakernelPlan,
    qwen38_megakernel_decode,
    qwen38_megakernel_prepare,
    qwen38_megakernel_supported,
    qwen38_megakernel_unsupported_reason,
)

__all__ = [
    "Qwen38MegakernelPlan",
    "qwen38_megakernel_decode",
    "qwen38_megakernel_prepare",
    "qwen38_megakernel_supported",
    "qwen38_megakernel_unsupported_reason",
]
