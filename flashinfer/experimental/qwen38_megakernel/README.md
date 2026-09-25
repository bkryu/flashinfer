# Qwen3.8 decoder megakernel (experimental, SM120)

One persistent CuTe DSL launch per decode step for the Qwen3.5 / Qwen3.8 hybrid
decoder — 48 gated-DeltaNet (GDN) layers + 16 full-attention layers for
`nvidia/Qwen3.8-27B-NVFP4` — or for any prefix / single layer of it (the MTP
drafter layer runs as the bf16 weight kind, optionally with its `fc` and norms
folded in). Public entry points: `flashinfer.qwen3_ops.qwen38_megakernel_*`
(thin, `@flashinfer_experimental_api`); this package holds the kernel, the
per-layer device tables and the scratch management.

**Status: experimental** — API and layouts may change without deprecation.
Opt-in = calling the API (an `ExperimentalWarning` is emitted once).

## What the kernel does

Per decode step (1..16 rows) every layer's chain runs inside one grid of
persistent CTAs: absorbed input RMSNorm -> fp8 W8A16 in-projection (per-tensor
alpha per shard) -> {GDN: 4-tap causal conv + SiLU, gated delta rule on fp32
state, gated RMSNorm | attention: per-head q/k RMSNorm, partial RoPE, paged KV
write, tensor-core GQA attention with the sigmoid output gate} -> fp8 W8A16
out-projection into the residual -> absorbed post-norm -> NVFP4 W4A16 gate|up
with the SwiGLU epilogue -> NVFP4 W4A16 down-projection into the residual, with
the next layer's norm weight folded into the activation the epilogue emits.
Layers hand off through device counters; the TMA descriptors are rewritten per
layer from device tables, so the layer count is a runtime value.

Measured on one RTX PRO 6000 (LightLM harness, 27B, decode): 1.33–1.36 TB/s
achieved HBM streaming (83–85 % of the 1.6 TB/s spec) at 1k–64k context, +7.5–9 %
tokens/s over the unfused kernel chain; +5 % at MTP=3 (verify form). See the
originating study for the ladder and the refuted levers.

## Layouts (v1 = the study's; see "vLLM adaptation" below)

| tensor | layout |
|---|---|
| in / out projection weights | fp8 e4m3 `[N, K]` + fp32 alphas (`alpha_in[2]` GDN = q\|k\|v rows, z rows; `[4]` attention = q, gate, k, v; `alpha_out[1]`) |
| gate\|up, down | NVFP4 packed uint8 `[N, K/2]` + e4m3 scales `[N, K/16]` (plain, unswizzled) + `alpha_gu[2]`, `alpha_dn[1]`; gate\|up rows interleaved in 8-row groups (`interleave_gate_up`) |
| bf16 kind (`wtype="bf16"`) | every weight bf16 `[N, K]`, alphas ignored (the MTP drafter) |
| GDN state | fp32 `ssm_pool [slots, HV, Dk=128, Dv=128]`, bf16 `conv_pool [slots, 2*Dk*HK + Dv*HV, 3]`; one slot per sequence (`state_slots`) |
| attention KV | bf16 `k_cache / v_cache [pages, 32, HKV, 256]`, `block_table int32 [seqs, max_pages]`, `slot_mapping int32 [tokens]` |
| verify form (`spec_rows=R`) | R ordered rows per sequence; pools read-only, per-row `stash_x/y/a/b` filled for the framework's accept step |

## Files

- `decoder_mega_sm120.py` — the kernel (`DecoderMegaSm120`), all phases.
- `gdn_mega_sm120.py`, `gemm_w8a16_sm120.py`, `gemm_sm120.py` — the GDN
  block kernel and the SM120 GEMM base classes it inherits (mainloops, TMA,
  epilogues).
- `decoder_host.py` — `DecoderSpec` (geometry + compile-time knobs),
  `DecoderTables` (device tables), `decoder_entry` (launch), `kv_splits_for`.
- `_runtime.py` — `cute.compile` memo (TVM-FFI launch path), PDL launch mode,
  persistent grid size. `_host_utils.py` — scratch caches, weight prep,
  RoPE table.
- `support.py` — lightweight support checker (shape / CC logic only).

## Tests and example

`pytest tests/experimental/test_qwen38_megakernel.py` (SM120 only; skipped
elsewhere): independent torch oracle for both layer kinds, verify form vs
sequential plain launches + stash replay, bf16 kind vs quantized kind on
exactly representable weights, drafter fold vs torch, public API + CUDA-graph
replay. Example: `examples/experimental/qwen38_megakernel_decode.py`.

## Knobs (environment)

- `FLASHINFER_QWEN38_TVM_FFI=0` — ctypes launch path instead of TVM-FFI.
- `FLASHINFER_QWEN38_PDL=0` — launch without programmatic dependent launch.

## vLLM adaptation

The v1 layouts are the study's. vLLM stores the SSM state as
`[blocks, HV, Dv, Dk]` (Dk innermost) with a padded per-block stride, the
conv state as `(K-1 [+num_spec], conv_dim)` and the FlashInfer-backend KV as
`[blocks, HKV, block, 2*head]`-style packed pages; the integration in vLLM
(`vllm/models/qwen3_5/nvidia/megakernel.py`) prepares weights once
(row permutations / dtype views only — never a re-quantization) and passes
state / cache views the kernel accepts. Layout options are added to
`DecoderSpec` as they land.
