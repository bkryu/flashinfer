<!-- SPDX-License-Identifier: Apache-2.0 -->
# Batch-invariant attention: the paired body's MMA commit structure at D = 128

A design note for a decision. The paired placement (`AttnConfig(q2=True)`: two Q tiles of one request and KV head per
CTA, one softmax warpgroup each, one MMA warp and one correction warpgroup shared) is the default at D = 64 from GQA group
4 and wins 5-16 % there. At D = 128 it was measured at 0.59-0.63 of the one-warpgroup body (Rubin report, sections 6.6 /
6.8) with the diagnosis "the per-tile chain is serial because P is written over its own S stage". This note instruments
that chain per role, finds a different dominant cost, measures the placement that removes it, and states what the paired
body can and cannot reach at D = 128. Everything is bits-identical to the one-warpgroup body (placements), measured on
Rubin (sm_107, bf16) with CUPTI kernel sums, cold L2, 30 iterations, two ABAB passes.

## 1. The structure (what the code does today)

TMEM at D = 128: `S_A (128) | S_B (128) | O_c,A (128) | O_c,B (128)` = 512 columns; `P_k` (bf16 pairs, 64 columns) is
written over the first half of `S_k` after the softmax has copied `S_k` out (`operands.tmem_plan_for`, `p_over_s`).
`O_m` of both tiles in smem (128 KB) next to one Q stage (2 x 32 KB) and a 3-slot K/V ring (3 x 32 KB): 309 KB of the
328 KB part. Registers 192 (softmax) / 96 (correction, 8-column pieces) / 32 (MMA, load and two empty warps: one
warpgroup, `setmaxnreg` is per warpgroup).

Per KV tile `t` the MMA warp issues, in this fixed order (`mma.mma_loop_q2`): `PV_A(t)`, `QK_A(t+1)`, `PV_B(t)`,
`QK_B(t+1)`. Because the tensor core executes a CTA's MMAs in order, `QK_k(t+1)` overwrites `P_k(t)` only after `PV_k(t)`
has read it -- the correctness argument of P-over-S -- and therefore `S_k(t+1)` cannot exist before `PV_k(t)` has been
issued, i.e. before the softmax of tile `t` has stored `P_k(t)`. The S stage itself is released by the softmax right after
its copy-out (`sh.release()` behind the TMEM-load fence, before the exponentials): there is no later signal to move
(question B2 (iv) is closed: the code already signals as early as the aliasing allows). So each warpgroup's chain is
`softmax(t) -> P(t) -> PV(t) -> QK(t+1) -> S(t+1) -> softmax(t+1)`, and the two chains share the MMA warp (fixed order),
the tensor core (in order), the correction warpgroup (A then B per tile, `correction_loop_q2`) and the four SMSPs (two
softmax warps, one correction warp and, on SMSP 0, the MMA warp each).

## 2. B1: the measured critical path on the current head (probe build: clock64 stamps per role, CTA 0, tiles 4..59)

Pairs 1x8K 32:8 (both Q tiles active): period 6177 per pair; one-warpgroup body 2562 per tile in the same build.
Softmax A / B compute 1993 / 1954 per tile and wait ~3660 for S each; the two softmaxes alternate almost perfectly (their
busy intervals overlap ~0 cycles per tile), so the period is one softmax plus one MMA segment, and the MMA segment from
"P_A(t) committed" to "S_A(t+1) full" is 3337 cycles (tile 10, `tl_seq.py` of the saved timeline):

```
   0  S_A(t) full                                      softmax A: copy-out 228, max/alpha 509, exps 858 (stats sent 1595)
1595  P stage acquire                                  +530 (the P word conversions, the acquire)
2417  P_A(t) committed
3000  MMA sees P_A(t)                                  +583  (the MMA warp was between the tile top and the V wait)
3533  MMA has O_A free                                 +533  (the correction had released O_A at 2357)
3849  PV_A(t) issued (8 MMAs)                          +316
4156  MMA at the K wait                                +307  (two commits and a loop iteration)
4486  K_{t+1} landed                                   +330
4969  MMA has the S_A stage                            +483  (the softmax had released it at 228)
5367  QK_A(t+1) issued (8 MMAs)                        +398
5754  S_A(t+1) full                                    +387  (tensor latency after the last issue)
```

Three of the segments (+583, +533, +483) are waits on barriers that had completed long before: they are the MMA warp's
own instruction stream. Its loop body is 627 SASS instructions with 32 LDL + 5 STL per pair-tile at 32 registers (the
one-warpgroup MMA loop: 334 instructions, 0 LDL; `compile_one.py` + `sass_loops.py` census), and with the 309 KB smem plan
the local spills miss L1 and pay L2 round trips. The fixed issue order adds "P_B(t) committed -> MMA sees P_B" 1488 on the
B chain. The correction is busy 54 % (rescale 869 / 778 at 8-column pieces); chunk-end tiles (`t % 4 == 3`) take 7312
against ~5900 (the smem `O_m` merge, 2500-3400 per Q tile, delays the next chunk's first PV); the V wait at the tile top
is 533 (3-slot ring). The lazy rescale (C1) on this head: period 6144 (rescale 152 / 108): the chain does not move -- the
report's 6.8 confirmed.

Single tile (an empty B): period 4882 = softmax ~2300 + the same MMA segment (~2300 without the B interleave) + waits.

## 3. B2: what was built and measured

### 3.1 Register budget and the softmax form (placements; same bits)
The MMA warp needs its two Q tiles' descriptors and eight pipeline states in registers. The budget is 65536 =
softmax x 256 + correction x 128 + other x 128, so 64 for the MMA warpgroup costs 4096 registers elsewhere. Probe-build
periods (pairs 1x8K / single tile):

| softmax / correction / MMA | softmax form | pairs | single | what moved |
|---|---|---|---|---|
| 192 / 96 / 32 (head) | packed | 6177 | 4882 | -- |
| 192 / 96 / 32 | streamed | 5809 | 4844 | |
| 176 / 96 / 64 | packed | 6402 | 5232 | the softmax spills (exps 898 -> 1924); the MMA segments shrink (P seen +378) |
| 176 / 96 / 64 | streamed | 5032 | 4415 | the softmax still spills (1470) |
| 192 / 64 / 64 | packed | 5694 | 4360 | no softmax spills, the correction at 64 holds (rescale 659) |
| 184 / 80 / 64 | packed | 5538 | 4741 | the softmax spills a little (1500) |
| 184 / 80 / 64 | streamed (W1) | 3883 | 3880 | softmax 1960 (no spills), correction busy 1102, P_A seen +282 |

W1 = the paired placement at D = 128 with the one-warpgroup body's streamed exponential form (the P words take the
scores' registers, so the live set fits 184) and 184 / 80 / 64 (`patch/w1_apply.py`; `operands.REGS_Q2_128`,
`StaticPlan.streamed`). In the probe build it removes the MMA warp's spill segments and the chunk-position profile becomes
3770 / 3837 / 3737 / 5470 (the chunk-end merge now +1700 per four tiles). Bits: `bits_q2.py` 19 cases identical to the
one-warpgroup body, the suite 104 passed on the W1 tree.

Unprobed (the number that counts; ratio one-warpgroup exact / paired, > 1 = pairing faster):

| case | head | W1 | head + C1 | W1 + C1 |
|---|---|---|---|---|
| single tile (emptyB 64x8K 8:8) | 0.608 | 0.723 | 0.659 | 0.731 |
| pairs 1x8K 32:8 | 0.593 | 0.711 | 0.643 | 0.746 |
| pairs 8x2K 32:8 | 0.625 | 0.763 | 0.660 | 0.784 |
| pairs 32x1K 32:8 | 0.622 | 0.782 | 0.642 | 0.808 |
| uniform 32 x 256q / 8K 32:8 (balanced pairs) | -- | 0.786 | -- | 0.817 |
| uniform 8 x 1Kq / 8K 32:8 | -- | 0.776 | -- | 0.813 |
| pairs 64x1K 32:8 | -- | 0.773 | -- | 0.800 |

W1 makes the paired body 1.17-1.26x faster than the head's; it stays 1.28-1.41x slower than the one-warpgroup body. The
uniform-work cases rule out load imbalance of the coarser pairs as the cause. The register split, unprobed, on the uniform
cases (32x256q / 8x1Kq / 64x1K / 1x8K): 184 / 80 / 64 = 0.786 / 0.776 / 0.773 / 0.711; 176 / 96 / 64 = 0.790 / 0.780 /
0.752 / 0.700; 192 / 64 / 64 = 0.698 / 0.728 / 0.687 / -- (the correction at 64 spills); 184 / 80 / 64 is kept. D = 64 (the W1 change is gated to D = 128):
0.965 / 0.974 / 1.046 / 1.094 on the same cases, i.e. unchanged.

### 3.2 The probe build's ratio does not transfer
In the probe build W1's pair period equals its single-tile period (3883 vs 3880: the second Q tile is free, 0.76 of the
one-warpgroup tile per tile-equivalent); a light probe with only the softmax's S-full and P-commit stamps gives the same
(pairs 3674 / 3502, single 3560, one-warpgroup 2450). Unprobed, the pair costs ~1.3x two one-warpgroup tiles. The
emptyB shape is HBM-bound (64 requests x 8 heads x 8K keys: 2 GB of K/V read once; 211 us is 10 TB/s), which inflates
every one-warpgroup probe number; the compute-bound uniform shape gives the one-warpgroup body ~1760 cycles per tile from
wall time against the probe's 2450. The paired kernel's behaviour under stamps (register allocation, the 20 predicate
tests per tile, the softmax issue slots) differs enough from the unprobed kernel that per-tile periods are a diagnosis of
WHERE the cycles go, not a prediction of the wall-time ratio. Every B-track verdict below is the unprobed number.

### 3.3 The alternatives (B2 i-iv) against the budgets
* (i) Out-of-order PV issue (poll both P barriers, issue whichever Q tile is ready; the per-tile folds unchanged, so the
  same bits): on the head the fixed order cost "P_B committed -> MMA sees P_B" 1488 and, with the MMA warp at 64
  registers, 1082; in W1 it is 282 / 358 (the MMA warp is no longer behind). Not built: with W1 the order is not where the
  cycles are.
* (ii) Per-warpgroup correction: a Q tile's O_c spans 128 TMEM lanes and warp `w` reaches lanes `32 (w % 4) ..` only, so
  each Q tile's rescale needs four warps -- two correction warpgroups (eight warps), not a 2 x 2 split. Budget:
  184 x 256 + 2 x R x 128 + 64 x 128 <= 65536 gives R <= 40, below the merge's buffers (two 8-column pieces, O_m pieces,
  coefficients); with the MMA warpgroup back at 32, R <= 56, which spills (the head's 96-register correction already
  spilled at 16-column pieces). Closed by the register file.
* (iii) P out of TMEM (smem A operand, S never aliased, QK(t+1) issued after the copy-out as at D = 64): TMEM becomes
  `S_A | S_B | O_A | O_B` with no P columns needed; smem needs 2 x 32 KB of P staging on top of 309 KB -> 342 KB > 328 KB.
  Fits only with `O_m` out of smem (the L2 scratch, 128 KB freed: then also a 5-7 slot ring and two Q stages) or a 2-slot
  ring (276 KB + 64 KB = 340 KB: no). The gain it targets -- overlapping QK(t+1) with softmax(t) -- is bounded by the same
  softmax-issue limit as pairing itself (section 4); not built in this round.
* (iv) Earlier S-full signal: the S stage is released right after the copy-out; the only ordering left is the in-order
  tensor execution of `QK_k(t+1)` behind `PV_k(t)`, which the aliasing requires. Nothing to move.

### 3.4 C1 (the lazy rescale) with W1
Probe build: pairs 4028 / single 3688 against 3883 / 3880 (the single-tile chain loses the ~200 of the exposed rescale).
Unprobed: W1 + C1 0.731-0.817 against W1 0.711-0.786 (table above): +3-5 %, the same as C1 gives the one-warpgroup body.
No pairing cell flips at D = 128 or D = 64 with C1.

## 4. Why the pair does not win at D = 128 (and what would)

The D = 128 softmax is issue-bound, not MUFU-bound: 1411 SASS instructions per tile per thread (the MUFU count is 129; the
rest is the copy-out, the maximum tree, the fmas, the conversions and the row-sum tree). With one softmax warp per SMSP the
one-warpgroup body overlaps QK(t+1) with softmax(t) (separate P columns) and runs a tile in ~1760-1930 cycles. Pairing puts
two softmax warps on every SMSP; whenever both are active each issues at half rate, so a pair-tile costs at least the sum
of the two softmaxes' issue cycles (~2 x 1400) plus whatever of the MMA segment the other softmax cannot hide. The paired
body can therefore reach at best ~0.75-0.85 of two one-warpgroup tiles even with a free MMA warp; W1 reaches 0.71-0.78 of
the one-warpgroup WALL time, i.e. 1.28-1.4x slower, with the chunk-end merge (10 %), the correction protocol and the V
wait as the identified remainders. At D = 64 the same structure wins because the one-warpgroup softmax is MUFU-bound there
(64 scores per thread but the same 129 exponentials... per 128 keys: the tile's exponentials dominate), the second softmax
warp fills the MUFU idle slots, and P has its own TMEM columns so QK(t+1) overlaps the softmax.

What would make pairing win at D = 128 is a shorter softmax instruction stream, i.e. a class change: the fp16 packed
exponentials (C2) halve the MUFU count but their current lowering adds instructions (report 6.8); a DSL-native packed
exponential (two keys per `ex2.approx.f16x2` with no inline-asm moves) and the packed `cvt`/`add` forms would cut the
1411 toward ~900 and the pair toward ~0.6 of two one-warpgroup tiles -- the "fp16-exponential lazy-rescale class" the
report's 6.6 estimated. That is a numerics decision (1.02x RMS for f16 exponentials, 6.8), not a placement.

## 5. Recommendation
* Land W1 as the paired placement's D = 128 configuration (bits identical, suite green, +17-26 % on the paired body,
  `AttnConfig(q2=True)` stays explicit, never the D = 128 default). The one-warpgroup body remains the D = 128 default.
* Do not pursue (ii) (register file) or (iv) (nothing to move); (i) only after a class change makes the chain MMA-bound;
  (iii) only together with `O_m` in the L2 scratch and only as the vehicle for a shorter softmax.
* The structural lever at D = 128 exists only together with a class change that shortens the softmax (C2 in a native
  lowering, with C1); as a placement alone the pair cannot beat the one-warpgroup body because both softmaxes share the
  issue slots. Pricing: W1 is free (a placement); the class change is the decision in `batch_invariant_attention_lazy_rescale.md`
  plus the fp16-exponential question the report left open.
* Side findings for the one-warpgroup body (placements, not built): the chunk-end TMEM-to-TMEM merge takes ~3400 cycles
  per chunk (one piece in flight at 128 registers); pipelining it as the D = 256 global merge is would save ~600 cycles per
  four tiles (~5 %); the correction's register budget at D = 128 (192 / 128) keeps the MMA warp at 32 registers with 5 LDL
  per tile -- harmless there (one Q tile), the reason the paired body needs 64.
