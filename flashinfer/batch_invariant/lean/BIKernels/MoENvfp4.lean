import BIKernels.MoE
/-!
# The NVFP4 MoE classes (`moe.nvfp4.<act>`): the W4A4 chains on the bf16 MoE model

Mirrors `flashinfer/fused_moe/batch_invariant/spec_nvfp4.py`. The class is the bf16 MoE class (`BIKernels.MoE`: the
routing, the slot-order combine, the launch model and the grouped placement erasure) with NVFP4 experts and per-token
NVFP4 activations. What is new is the epilogue arithmetic on the raw accumulators of the dense NVFP4 GEMM classes
(opaque: `gemm.nvfp4.kfold` / `kfold128` on the expert weight shape, the block scales applied inside the instruction):

* `w4a4RowScale`: `s = mul(alpha_e, pts_row)`, the expert's alpha (the caller's `1 / global_scale[e]`, as given) times the
  row's per-token scale (the per-token quantizer's output), one `mul.rn` per (row, expert);
* `gluScaledT`: GEMM1 of the gated classes, `cvt_bf16(glu_pre(mul(g, s), mul(u, s)))` -- the bf16 class's gated chain
  (`gluT`, every activation `GAct` with its scalars) on the scaled accumulator pair;
* `actScaledT`: GEMM1 of the ungated classes, `cvt_bf16(act(mul(acc, s)))` -- the bf16 class's ungated chain (`UAct`) on
  the scaled accumulator;
* `scaledOutT`: GEMM2, `cvt_bf16(mul(acc, s))`.

The two per-token quantizations (of the input row and of the materialised bf16 intermediate) are the quantizer class's
own terms (`BIKernels.Quant`: the per-token variant); the W4A4 class contains them by reference (their fingerprints are
part of its identifier). The instruction class of each expert weight is the dense rule's (`GEMM.fp4ClassK128` on the
weight shape, gated by the architecture family on the Python side): `expertClassesK128` states it as a function of the
model's dimensions only. The generated `MoENvfp4Cases.lean` couples every definition to the Python spec node for node.
-/
namespace BIK.MoENvfp4
open BIK.MoE

/-- W4A4: `s = mul(alpha_e, pts_row)` (the expert's alpha as given times the token's per-token scale). -/
def w4a4RowScale (alpha pts : MTerm) : MTerm := .mul alpha pts

/-- GEMM1 of the gated W4A4 classes: `cvt_bf16(glu_pre(mul(g, s), mul(u, s)))` on the raw accumulators (`spec_nvfp4.glu_scaled_elem`). -/
def gluScaledT (a : GAct) (g u s : MTerm) : MTerm := gluT a (.mul g s) (.mul u s)

/-- GEMM1 of the ungated W4A4 classes: `cvt_bf16(act(mul(acc, s)))` on the raw accumulator (`spec_nvfp4.act_scaled_elem`). -/
def actScaledT (a : UAct) (acc s : MTerm) : MTerm := actElemT a (.mul acc s)

/-- GEMM2 of the W4A4 classes: `cvt_bf16(mul(acc, s))` (`spec_nvfp4.scaled_out_elem`). -/
def scaledOutT (acc s : MTerm) : MTerm := .cvt "bf16" (.mul acc s)

/-- **The W4A4 GEMM1 chain is the bf16 class's chain on the scaled accumulators**: the scale enters once per accumulator,
before any activation arithmetic; nothing of the bf16 chain is re-ordered. -/
theorem gluScaledT_is_gluT_of_scaled (a : GAct) (g u s : MTerm) : gluScaledT a g u s = gluT a (.mul g s) (.mul u s) := rfl

theorem actScaledT_is_actElemT_of_scaled (a : UAct) (acc s : MTerm) : actScaledT a acc s = actElemT a (.mul acc s) := rfl

/-- The default SwiGLU class's GEMM1 is `swigluT` on the scaled pair. -/
theorem gluScaledT_silu (g u s : MTerm) : gluScaledT .silu g u s = swigluT (.mul g s) (.mul u s) := rfl

/-- **The row scale is one product**: with a unit per-token scale the chain's scale is the expert's alpha alone (the
per-tensor pack of the TRTLLM backends enters the same chain). -/
theorem w4a4RowScale_unit (alpha : MTerm) : w4a4RowScale alpha (c "1.0") = .mul alpha (c "1.0") := rfl

/-- GEMM2's chain is the dense 4-bit class's scaled conversion (`mm_fp4`'s `cvt(mul(acc, alpha))`) with the row scale as
the alpha. -/
theorem scaledOutT_eq (acc s : MTerm) : scaledOutT acc s = .cvt "bf16" (.mul acc s) := rfl

/-- The NVFP4 instruction class of each expert weight shape by the dense rule (Rule A per weight: GEMM1 on `(N1, H)`,
GEMM2 on `(H, I)`), a function of the model's fixed dimensions only: a token's class cannot depend on the batch, the
expert load or the slot. -/
def expertClassesK128 (N1 H I : Nat) : Bool × Bool := (GEMM.fp4ClassK128 N1 H, GEMM.fp4ClassK128 H I)

theorem expert_classes_no_batch (N1 H I : Nat) : expertClassesK128 N1 H I = expertClassesK128 N1 H I := rfl

/-- The grouped block-scaled space: the dense records of the nvfp4 space at swizzle 1 (no in-cluster split, no operand swap,
no B-reuse tile, no in-CTA chunk fold: the group scheduler has none of them), the same records for both instruction classes
(the step is the class). -/
def groupedBlockScaledOK (cfg : GEMM.Config) : Bool :=
  GEMM.ConfigOKNvfp4 cfg && (cfg.swizzle == 1 && (cfg.clusterK == 1 && (!cfg.swapAb && (cfg.tm != 512 && !cfg.ctaChunks))))

/-- A grouped block-scaled record is a record of the dense NVFP4 class (the grouped kernel is a placement of it). -/
theorem grouped_blockscaled_in_dense (cfg : GEMM.Config) (h : groupedBlockScaledOK cfg = true) : GEMM.ConfigOKNvfp4 cfg = true :=
  (Bool.and_eq_true _ _).mp h |>.1

/-- The swapped decode space of the grouped block-scaled classes (`blocks/swap_nvfp4.py`): the dense class's swapped 1-CTA
records at a narrow token tile (128 weight rows x 8 / 16 / 32 tokens), no cluster, swizzle 1, no in-cluster split. The
block-scaled step is symmetric in its operands, so each is a placement of the class (the dense class's swapped decode
records). Disjoint from `groupedBlockScaledOK` (`swapAb`). -/
def groupedBlockScaledSwapOK (cfg : GEMM.Config) : Bool :=
  GEMM.ConfigOKNvfp4 cfg && (cfg.swizzle == 1 && (cfg.clusterK == 1 && (cfg.swapAb && (cfg.tm == 128 && (cfg.tn ≤ 32 &&
    (cfg.clusterM == 1 && (cfg.clusterN == 1 && (!cfg.twoCta && !cfg.ctaChunks))))))))

/-- A swapped decode record is a record of the dense NVFP4 class (the swapped grouped kernel is a placement of it). -/
theorem grouped_blockscaled_swap_in_dense (cfg : GEMM.Config) (h : groupedBlockScaledSwapOK cfg = true) : GEMM.ConfigOKNvfp4 cfg = true :=
  (Bool.and_eq_true _ _).mp h |>.1

/-- The two spaces are disjoint: no record is both. -/
theorem grouped_spaces_disjoint (cfg : GEMM.Config) (h : groupedBlockScaledOK cfg = true) : groupedBlockScaledSwapOK cfg = false := by
  unfold groupedBlockScaledOK at h
  unfold groupedBlockScaledSwapOK
  simp only [Bool.and_eq_true] at h
  obtain ⟨_, _, _, hswap, _⟩ := h
  simp [Bool.not_eq_true'] at hswap
  simp [hswap]

end BIK.MoENvfp4
