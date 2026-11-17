<!-- SPDX-License-Identifier: Apache-2.0 -->
# Batch-invariant attention: the lazy-rescale class change (C1), measured

A design note for a decision. The batch-invariant attention class (`flashinfer/attention/batch_invariant/`) rescales the
output accumulator with `alpha = EX2(add(mul(m, c), mul(m', nc)))` on every tile whose row maximum moves. Every vendor
kernel (FA4, prims-TS, the cute-dsl FMHA, cuDNN, TRT-LLM) keeps the old maximum while the new one is within 8 binades
and lets the probabilities grow to 2^8 instead. This note states what that change is as a change of the class's term,
what it costs in accuracy, what it buys on Rubin (sm_107, bf16), and a second term change -- the row sum over the
bf16-rounded probabilities -- that recovers part of the accuracy cost. Everything here was measured with the probe knobs
on `bi/attn-next` (`AttnConfig.lazy` / `AttnConfig.exp16`, accepted only under `FLASHINFER_BI_ATTN_CLASS_PROBE=1`);
none of it changes the published class.

## 1. The term change

The exact tile step (`spec.tile_stats`) forms `m' = fmax(m, m_t)` and `alpha = EX2(x)` with
`x = add(mul(m, c), mul(m', nc))`. The lazy step keeps the same operations and adds one exact comparison on the value
`x` the class already computes:

```
m_c   = fmax(m, m_t)                      # the exact step's m'
x     = add(mul(m, c), mul(m_c, nc))       # the exact step's alpha argument (two roundings, nc = -c exactly)
keep  = x >= -L                            # L = 8 (log2 units); x = -inf on a chunk's first tile (m = -inf), never kept
m'    = m      if keep else m_c
alpha = 1.0    if keep else EX2(x)
nms   = mul(m', nc)                        # the exponentials' offset: p_k = EX2(fma(s_k, c, nms))
```

No new floating-point primitive: `>=` on an fp32 value is exact, `alpha = 1.0` is the value `EX2(+0)` would give when
`m' = m`, and everything downstream (the probabilities, the bf16 `cvt.rn`, the row-sum tree, `l = fma(l, alpha, r)`,
`mul(O, alpha)`, the chunk merge, the epilogue) is unchanged in form. The term changes because `m` is no longer the
running maximum but a function of the row's score sequence in tile order; every output bit can differ, so the class gets a
new fingerprint. Where it lives: `spec.tile_stats` (the rule), `spec.alpha_term` (unchanged, now applied under the rule),
`spec_fingerprint` / `FINGERPRINT_D{64,128,256}` (new constants), the Python term algebra's opaque `op` for the select
(`fp_interface/term.py`: `op("keep_max", x, L, m, m_c)` or a dedicated `select_ge`), and `blocks/classprobe.tile_max_step`,
which both bodies (the row body in its one-warpgroup, paired and replicated-row placements; the Swaps decode body) already
call -- the single place the rule is applied, so R3 / R4 hold by construction.

### Invariants the change relies on
* Range: with `L = 8` the reference maximum obeys `m >= max_seen - L / c` for every row at every tile (a kept step
  has `(m_t - m) c <= L`), so `p_k = EX2(c (s_k - m)) <= 2^L = 256` and `l <= 256 T` per tile: both in bf16 / fp32 range
  with the same margins the vendors rely on (FA4 asserts `2^L` below the P dtype's maximum; our P is bf16).
* Shift invariance: the chunk state `(m, l, O)` under any reference maximum `m` with `l = sum_k EX2(c (s_k - m))`,
  `O = sum_k bf16(EX2(c (s_k - m))) v_k` normalises to the same softmax; the chunk merge `merge(a, b)` and the epilogue
  `cvt(add(mul(O, div(1, l)), +0))` are already written for arbitrary `(m_a, m_b)` and need no change.

### What the Lean model has to carry
`flashinfer/batch_invariant/lean/BIKernels/Attention.lean` is a symbolic model: `ATerm` terms with `tileStep`,
`alphaTerm`, `merge`, `chunkState`, `fused`, `epilogue`, and the theorems `splitKernel_eq_fused`,
`splitKernel_assignment_irrelevant`, `prefill_decode_agree`, `determinism`, `batch_invariance`,
`permutation_invariance`, `subbatch_invariance`, `config_invariance`, `space_invariant` (all `rfl` over the same fold on
both sides), plus the separations `output_depends_on_C` / `output_depends_on_D`. C1 needs:
1. a constructor for the data-dependent select in `ATerm` (`keepMax (x L m mc : ATerm)` or a general `select`), printed
   exactly as the Python term prints it (the fingerprint is the printed term's hash);
2. `tileStep` to form `mNew := keepMax x L st.m mC` with `x := add (mul st.m c) (mul mC nc)` and `alpha := select ... 1.0 (ex2 x)`,
   `nms := mul mNew nc`;
3. nothing else: every theorem above is structural and goes through unchanged, because they compare the same fold on two
   sides (prefill == decode, split == fused, every configuration the same term). The range lemma (`m >= max - L/c`,
   `p <= 2^L`) and the shift-invariance argument are statements about VALUES, which the symbolic model does not have; they
   belong in the Python spec as checked assertions over the reference evaluator (`spec.verify_plan` / the CPU model) and in
   this note, not in Lean, unless a value semantics is added.
4. `Cases.lean` (`attnP_*` parameter sets) is unaffected; `Main.lean` runs the same statements.

### Tests that pin it
`tests/attention/test_batch_invariant_attention.py`: `test_fingerprint_is_a_function_of_the_head_dim` (new constants),
`test_matches_reference` (its tolerance: see the accuracy table), and every R1-R4 test unchanged --
`test_determinism_and_graph_replay`, `test_batch_permutation_and_subbatch_invariance`,
`test_decode_batch_invariance_and_page_table_junk`, `test_prefill_equals_decode`, `test_paged_equals_ragged`,
`test_chunked_prefill_equals_full`, `test_multi_token_decode_equals_prefill`, `test_config_agreement`,
`test_paired_d128_*`, `test_sliding_window_prefill_equals_decode`. The probe already runs the whole suite against the
variant (`FLASHINFER_BI_ATTN_CLASS_PROBE_FORCE="lazy=8"` makes every `AttnConfig` take the knob): 104 passed on
`bi/attn-next` with nothing deselected, i.e. R1-R4 hold for the lazy term in every body and placement.

## 2. What it buys (Rubin, CUPTI kernel sums, cold L2, 30 iterations, ABAB two passes; ratio exact / lazy, > 1 = faster)

| group | 1x1K 32:8 | 1x8K | 8x2K | 32x1K | 64x1K | 1x1K 8:8 | 1x8K | 8x2K | 32x1K | 64x1K |
|---|---|---|---|---|---|---|---|---|---|---|
| D = 128 paged | 1.013 | 1.055 | 1.038 | 1.014 | 1.013 | 1.028 | 1.039 | 1.027 | 0.995 | 0.983 |
| D = 128 ragged | 1.004 | 1.050 | 1.044 | 1.034 | 1.033 | 1.016 | 1.038 | 1.023 | 1.004 | 0.993 |
| D = 64 paged | 1.010 | 1.052 | 1.022 | 1.010 | 1.014 | 1.005 | 1.017 | 1.008 | 1.000 | 1.000 |
| D = 64 ragged | 1.004 | 1.031 | 1.023 | 1.021 | 1.021 | 1.001 | 1.025 | 1.010 | 1.004 | 1.004 |
| D = 256 paged | 1.015 | 0.995 | 1.003 | 1.042 | 1.048 | 1.008 | 1.002 | 1.007 | 1.018 | 1.014 |
| D = 256 ragged | 1.008 | 0.991 | 0.996 | 1.016 | 1.018 | 1.004 | 1.001 | 0.997 | 1.007 | 1.010 |

Decode (Swaps body, 32:8, B x L): D = 128 16x8K 0.997, 64x8K 0.998, 16x32K 0.999; D = 64 0.993 / 0.987 / 0.995; D = 256
0.998 / 1.000 / 1.000 -- neutral (the Swaps body is not softmax-bound). The D = 64 paged default is the paired placement at
GQA group >= 4: these cells are the shipped kernels.

Reading: +3-5.5 % on long rows at D = 64 / 128 (paged and ragged), +4-5 % on the many-request D = 256 cells, within +-1 %
elsewhere, two cells at -1.7 % (D = 128 8:8 64x1K, a pass-stable loss on short rows where the compare and the kept
branch cost more than the skipped rescales save). Geomean over the 60 prefill cells: 1.016. Timelines (probe build): the
one-warpgroup D = 128 softmax shortens 2064 -> 1861 cycles per tile because the correction's rescale stream (424 -> 100
busy cycles) stops competing for the SMSP issue slots; the chunk merges and the epilogue are untouched.

Interaction with the paired placement (section 6.6 / 6.8 of the Rubin report, and `docs/batch_invariant_attention_paired_body.md`):
on the current head C1 lifts the paired D = 128 body from 0.593-0.625 to 0.643-0.660 of the one-warpgroup time; on the
reworked paired body (W1: streamed softmax, 184 / 80 / 64 registers) from 0.711-0.782 to 0.731-0.808. No pairing cell
flips at D = 128; at D = 64 the paired default keeps its 1.05-1.09 at 8x2K / 32x1K with or without C1 (1.046 -> 1.055,
1.094 -> 1.093). C1 is independent of the pairing decision.

## 3. What it costs: accuracy (256 sampled rows per case vs an fp32 reference; frac = fraction of ALL output elements more than one bf16 ulp from the exact class)

| D | kind | case | exact max / rms | C1 (lazy 8) max / rms (x exact) / frac | rsum alone | C1 + rsum |
|---|---|---|---|---|---|---|
| 128 | paged | 1x8K 32:8 | 0.0148 / 4.63e-4 | 0.0151 / 5.43e-4 (1.17x) / 0.093 | 0.0148 / 4.62e-4 (1.00x) / 0.000 | 0.0148 / 5.17e-4 (1.12x) / 0.092 |
| 128 | paged | 8x2K 32:8 | 0.0157 / 7.06e-4 | 0.0184 / 8.13e-4 (1.15x) / 0.075 | 0.0157 / 7.02e-4 (0.99x) / 0.000 | 0.0164 / 7.59e-4 (1.08x) / 0.075 |
| 128 | paged | 64x1K 32:8 | 0.0147 / 8.45e-4 | 0.0206 / 9.37e-4 (1.11x) / 0.060 | 0.0147 / 8.40e-4 (0.99x) / 0.000 | 0.0147 / 8.86e-4 (1.05x) / 0.059 |
| 128 | paged | 1x8K 8:8 | 0.0086 / 4.55e-4 | 0.0138 / 5.41e-4 (1.19x) / 0.092 | 0.0086 / 4.53e-4 (1.00x) / 0.000 | 0.0098 / 5.11e-4 (1.12x) / 0.092 |
| 128 | paged | 8x2K 8:8 | 0.0123 / 6.97e-4 | 0.0160 / 8.00e-4 (1.15x) / 0.075 | 0.0123 / 6.94e-4 (0.99x) / 0.000 | 0.0123 / 7.48e-4 (1.07x) / 0.075 |
| 128 | paged | 64x1K 8:8 | 0.0153 / 8.59e-4 | 0.0153 / 9.73e-4 (1.13x) / 0.060 | 0.0159 / 8.54e-4 (0.99x) / 0.000 | 0.0159 / 9.07e-4 (1.05x) / 0.059 |
| 128 | ragged | 8x2K 32:8 | 0.0157 / 7.06e-4 | 0.0184 / 8.13e-4 (1.15x) / 0.075 | 0.0157 / 7.02e-4 (0.99x) / 0.000 | 0.0164 / 7.59e-4 (1.08x) / 0.075 |
| 128 | decode | 64x8K 32:8 | 0.0041 / 2.79e-4 | 0.0068 / 3.44e-4 (1.23x) / 0.102 | 0.0041 / 2.78e-4 (1.00x) / 0.000 | 0.0054 / 3.33e-4 (1.20x) / 0.101 |
| 64 | paged | 1x8K 32:8 | 0.0088 / 4.68e-4 | 0.0134 / 5.49e-4 (1.17x) / 0.092 | 0.0084 / 4.66e-4 (1.00x) / 0.000 | 0.0111 / 5.21e-4 (1.11x) / 0.092 |
| 64 | paged | 8x2K 32:8 | 0.0119 / 7.04e-4 | 0.0169 / 8.11e-4 (1.15x) / 0.076 | 0.0119 / 7.01e-4 (0.99x) / 0.000 | 0.0119 / 7.59e-4 (1.08x) / 0.075 |
| 64 | paged | 64x1K 32:8 | 0.0142 / 8.36e-4 | 0.0148 / 9.31e-4 (1.11x) / 0.060 | 0.0142 / 8.30e-4 (0.99x) / 0.000 | 0.0142 / 8.78e-4 (1.05x) / 0.059 |
| 64 | paged | 1x8K 8:8 | 0.0133 / 4.76e-4 | 0.0198 / 5.66e-4 (1.19x) / 0.093 | 0.0133 / 4.74e-4 (1.00x) / 0.000 | 0.0133 / 5.32e-4 (1.12x) / 0.092 |
| 64 | paged | 8x2K 8:8 | 0.0125 / 7.00e-4 | 0.0154 / 8.21e-4 (1.17x) / 0.076 | 0.0125 / 6.97e-4 (1.00x) / 0.000 | 0.0125 / 7.55e-4 (1.08x) / 0.075 |
| 64 | paged | 64x1K 8:8 | 0.0137 / 8.35e-4 | 0.0137 / 9.21e-4 (1.10x) / 0.060 | 0.0137 / 8.30e-4 (0.99x) / 0.000 | 0.0137 / 8.78e-4 (1.05x) / 0.059 |
| 64 | ragged | 8x2K 32:8 | 0.0119 / 7.04e-4 | 0.0169 / 8.11e-4 (1.15x) / 0.076 | 0.0119 / 7.01e-4 (0.99x) / 0.000 | 0.0119 / 7.59e-4 (1.08x) / 0.075 |
| 64 | decode | 64x8K 32:8 | 0.0062 / 2.91e-4 | 0.0069 / 3.63e-4 (1.24x) / 0.101 | 0.0062 / 2.91e-4 (1.00x) / 0.000 | 0.0062 / 3.48e-4 (1.19x) / 0.101 |
| 256 | paged | 1x8K 32:8 | 0.0157 / 4.66e-4 | 0.0157 / 5.44e-4 (1.17x) / 0.092 | 0.0156 / 4.64e-4 (1.00x) / 0.000 | 0.0157 / 5.18e-4 (1.11x) / 0.092 |
| 256 | paged | 8x2K 32:8 | 0.0160 / 7.11e-4 | 0.0165 / 8.17e-4 (1.15x) / 0.075 | 0.0160 / 7.07e-4 (0.99x) / 0.000 | 0.0160 / 7.66e-4 (1.08x) / 0.074 |
| 256 | paged | 64x1K 32:8 | 0.0152 / 8.42e-4 | 0.0253 / 9.37e-4 (1.11x) / 0.060 | 0.0152 / 8.36e-4 (0.99x) / 0.000 | 0.0162 / 8.84e-4 (1.05x) / 0.059 |
| 256 | paged | 1x8K 8:8 | 0.0148 / 4.68e-4 | 0.0148 / 5.45e-4 (1.17x) / 0.092 | 0.0148 / 4.66e-4 (1.00x) / 0.000 | 0.0148 / 5.21e-4 (1.11x) / 0.092 |
| 256 | paged | 8x2K 8:8 | 0.0124 / 6.98e-4 | 0.0225 / 8.07e-4 (1.16x) / 0.075 | 0.0124 / 6.94e-4 (0.99x) / 0.000 | 0.0124 / 7.52e-4 (1.08x) / 0.075 |
| 256 | paged | 64x1K 8:8 | 0.0155 / 8.43e-4 | 0.0202 / 9.37e-4 (1.11x) / 0.060 | 0.0158 / 8.37e-4 (0.99x) / 0.000 | 0.0158 / 8.86e-4 (1.05x) / 0.059 |
| 256 | ragged | 8x2K 32:8 | 0.0160 / 7.11e-4 | 0.0165 / 8.17e-4 (1.15x) / 0.075 | 0.0160 / 7.07e-4 (0.99x) / 0.000 | 0.0160 / 7.66e-4 (1.08x) / 0.074 |
| 256 | decode | 64x8K 32:8 | 0.0081 / 2.84e-4 | 0.0096 / 3.52e-4 (1.24x) / 0.100 | 0.0081 / 2.84e-4 (1.00x) / 0.000 | 0.0096 / 3.38e-4 (1.19x) / 0.100 |

The mechanism of C1's cost: under the exact rule the dominant key of a row has `p = EX2(0) = 1`, which bf16 holds exactly;
under the lazy rule it has `p = 2^x` with `x in (0, 8]`, a value with up to 8 significant bits below the leading one that
bf16 rounds (relative error up to 2^-9) -- and that rounding is on the LARGEST term of the PV sum while the fp32 row sum
`l` carries the unrounded value, so the normaliser no longer cancels it. Hence +11-25 % RMS, the maximum error up to
1.7x, and 6-10 % of the output elements moving by more than one bf16 ulp. FA4 and the other vendors carry exactly this.

### The rounded row sum (the second term change)
`l` accumulates `widen(cvt_bf16(p_k))` -- the probabilities the MMA consumes -- instead of `p_k`:
`r = tree_add_k(widen(cvt_rn_bf16(p_k)))`, with the widening an exact bit movement and everything else unchanged. The
output becomes `sum_k bf16(p_k) v_k / sum_k bf16(p_k)`, a convex combination with the weights the MMA actually used, so
the dominant key's rounding is cancelled by construction. Measured: alone it changes the exact class's output in no
element by more than one ulp (frac 0.000) and leaves RMS within 1 % (0.99-1.00x: the per-term roundings average out
against the fp32 sum in the exact class anyway); with C1 it returns the MAXIMUM error to the exact class's in every case
(0.0148 vs 0.0151-0.0253) and roughly halves the RMS excess (1.11-1.19x -> 1.05-1.12x prefill; decode 1.23x -> 1.19x,
where the one-token rows have few keys). Cost in this lowering: the row-sum tree now depends on the bf16 conversion of each pair, and the softmax at D = 128 is
issue-bound. Two implementations were timed (ratio exact / variant, > 1 = faster): per-element `cvt` + widen
(3-4 instructions per key): 0.73-0.93 on prefill at D = 64 / 128, 0.92-0.98 at D = 256, decode neutral; the widening by a
shift and a mask on the packed bf16x2 P word (2 integer ops per pair, the same bits): D = 128 paged 0.89-0.95, ragged
0.92-0.96, D = 64 0.94-0.96, decode neutral. With C1: 0.91-0.98 against the exact class at D = 128, 0.91-0.96 at D = 64 --
a net loss. The rounded sum is numerically the better class but costs more than C1 gains in this instruction stream.

## 4. Hazards for a careless implementation (R2 / R3 / R4)
1. The compare must be on the SAME fp32 term in every body: `x = add(mul(m, c), mul(m_c, nc))` (two roundings). FA4's
   `(m_old - m_new) * scale` (a subtraction, then a product) differs from it at the threshold's rounding boundaries; one
   body using one form and another body the other desynchronises the `m` sequences and breaks prefill == decode.
2. Per-row decisions only: any warp vote or CTA-wide choice that changes `m` (rather than merely skipping a multiply by
   exactly 1.0) makes a row's output depend on its tile-mates -- the end of batch invariance. The vote is allowed only as
   an optimisation of the skip (all lanes with `alpha == 1.0` skip; a lane with `alpha != 1.0` multiplies).
3. The chunk's first tile resets `m = -inf` exactly as the exact class, so `x = -inf` and the step is never kept there:
   chunk partials stay functions of the chunk alone (split == fused, decode == prefill).
4. RNE conversions everywhere: `P = cvt.rn.bf16(p)` in the row body's packed word and the Swaps body's per-element store;
   with the rounded row sum, the sum's leaf must be formed from the SAME bf16 word that is stored as P (not a second
   conversion of a differently rounded value).
5. The hardware row maximum (`tcgen05.ld.red`) still gives the exact tile maximum `m_t`; the rule's input is unchanged.
6. The paired body hands `m` across tiles inside one warpgroup, the Swaps body across its two warpgroups (the `hm` ring):
   the same rule in the same tile order; the probe's single helper `tile_max_step` is the way to keep it so.
7. The threshold is a class constant (`L = 8`); `L` enters the fingerprint. A different `L` is a different class.

## 5. Recommendation
C1 alone, or neither. Not C1 with the rounded row sum (in this lowering it costs 5-10 % of the softmax-bound cells, more
than C1 gains).

* Gain of C1 alone: +3-5.5 % on long-row prefill at D = 64 / 128 (paged and ragged) and +4-5 % on the many-request
  D = 256 cells, geomean 1.016 over the 60 prefill cells, decode neutral; the one-warpgroup body stays the D = 128 default
  (C1 does not reopen pairing at D = 128: section 2 and the paired-body note).
* Cost of C1 alone: RMS vs fp32 1.11-1.19x of the exact class on prefill (1.23x decode), the maximum error up to 1.7x,
  6-10 % of the elements moved by more than one bf16 ulp -- exactly what FA4 and the other vendors ship. The rounded row
  sum would bring the maximum error back to the exact class's and the RMS to 1.05-1.12x, at the speed cost above; it is
  the right companion only if a cheaper lowering appears (a packed widening with no dependency on the conversion), and it
  is NOT worth adopting alone (it changes no element by more than one ulp and costs 5-10 %).
* Price of C1: a new fingerprint for all three head dims, the spec / Lean / test changes of section 1 (one constructor,
  one rule, three constants; every theorem structural), and the hazards of section 4 reviewed in every body.
* If the accuracy budget of the class is "not worse than the exact class", neither: 1.11-1.19x RMS is a real cost for
  +1.6 % geomean (+5.5 % at best).
