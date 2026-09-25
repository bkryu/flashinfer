.. _apiqwen3_ops:

flashinfer.qwen3_ops
====================

.. currentmodule:: flashinfer.qwen3_ops

.. warning::

   **Experimental.** Every function on this page is an
   ``@flashinfer_experimental_api``: calling it is the opt-in (an
   ``ExperimentalWarning`` is emitted once per process) and the API, the
   tensor layouts and the ``DecoderSpec`` knobs may change or be removed
   without deprecation. See :ref:`experimental`.

Model-specific decode operators for the Qwen3 family. The first one is the
**Qwen3.8 decoder megakernel**: the whole Qwen3.5 / Qwen3.8 hybrid decoder
(48 gated-DeltaNet layers + 16 full-attention layers for
``nvidia/Qwen3.8-27B-NVFP4``), or any prefix / single layer of it, as ONE
persistent CuTe DSL launch per decode step on SM120 (RTX PRO 6000 / GeForce
Blackwell). Layers hand off through device counters and per-layer TMA
descriptor rewrites, so a decode step is one kernel instead of ~450 launches.

Supported: SM120 only, TP=1, 1..16 decode rows per launch, the checkpoint's
recipe (per-tensor fp8 projections, NVFP4 MLP) or bf16 weights (the MTP
drafter layer), GDN head dims 128, attention head dim 256, page size 32,
optional ``spec_rows=R`` verify form (R ordered rows per sequence, GDN state
read-only + per-row stash) and the drafter ``fc`` + norm fold.

Usage: build the per-layer dicts once (``LAYER_KEYS_*`` in
``flashinfer.experimental.qwen38_megakernel``), call
:func:`qwen38_megakernel_prepare` per launch shape, then
:func:`qwen38_megakernel_decode` per step — eagerly first, then inside a CUDA
graph (the plan's buffers keep their addresses).

.. autosummary::
    :toctree: ../generated

    qwen38_megakernel_supported
    qwen38_megakernel_unsupported_reason
    qwen38_megakernel_prepare
    qwen38_megakernel_decode
    Qwen38MegakernelPlan
