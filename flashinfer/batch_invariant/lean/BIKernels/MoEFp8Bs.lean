import BIKernels.MoE
import BIKernels.FP8BlockQuant
/-!
# The block-scale fp8 MoE classes `moe.e4m3bs.<act>` (`fused_moe/batch_invariant/spec_fp8bs.py`)

DeepSeek-style fp8: e4m3 experts with one fp32 scale per 128x128 block, e4m3 activations with one fp32 scale per 1x128 block,
both by the `fp8bs.quant` class (`FP8BlockQuant.lean`). The grouped kernel is a placement of the dense block-scale class
`gemm.e4m3.kblock128` per expert (`GEMM.kblock128`: the per-block `mma32` fold promoted in block order with
`fma(mul(s_a, s_b), P_kb, acc)`), so the class has no software scale after the fold: GEMM1 is the bf16 class's chain on the
promoted accumulators, GEMM2 the plain conversion, and the intermediate's requantization is the `fp8bs.quant` class on the
bf16-rounded chain value -- fused into GEMM1's epilogue or as the standalone pass, two placements of one term.
-/
namespace BIK.MoE

/-- The grouped block-scale kernel's result for an expanded row: the dense class's block fold over `KB` blocks; the placement (the
expert segment, the load, the position in the segment, the tile and its scale ring) is erased. -/
def groupedRowBS (_grouping : Nat → Nat × Nat) (_load : Nat → Nat) (_pos : Nat → Nat)
    (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) : GEMM.GTerm :=
  GEMM.kblock128 a b KB

/-- **R4: grouped == standalone.** -/
theorem grouped_eq_dense_bs (g : Nat → Nat × Nat) (l p : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) :
    groupedRowBS g l p a b KB = GEMM.kblock128 a b KB := rfl

/-- **Expert-load, grouping and position independence (R2 b, c).** -/
theorem grouped_placement_irrelevant_bs (g₁ g₂ : Nat → Nat × Nat) (l₁ l₂ p₁ p₂ : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) :
    groupedRowBS g₁ l₁ p₁ a b KB = groupedRowBS g₂ l₂ p₂ a b KB := rfl

/-- The grouped kernel restarts the running sum per tile (the dense `tileResultKB_reset`); the promotion loop is the class
(`GEMMConfig.promoteLoop_eq_kblock128`) whatever the accumulator ring's depth. -/
theorem grouped_tile_reset_bs (cfg : GEMM.Config) (S : Nat) (prev₁ prev₂ : GEMM.GTerm) (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) :
    GEMM.tileResultKB cfg S prev₁ a b KB = GEMM.tileResultKB cfg S prev₂ a b KB := GEMM.tileResultKB_reset cfg S prev₁ prev₂ a b KB

/-- A different class from the fp8 per-tensor fold (`moe.e4m3.<act>`: `kfold32`) at the same operands. -/
theorem grouped_bs_ne_per_tensor (g : Nat → Nat × Nat) (l p : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) :
    groupedRowBS g l p a b 1 ≠ GEMM.kfold32 a b GEMM.stepsPerBlock := GEMM.kblock128_ne_kfold32 a b

/-- GEMM1 of the block-scale class: the bf16 class's gated chain on the promoted accumulators (`Fp8BsMoESpec.gemm1`). -/
def e4m3bsGemm1T (a : GAct) (g u : MTerm) : MTerm := gluT a g u

/-- The ungated chains of the class (`moe.e4m3bs.relu2` and the other ungated activations): the bf16 class's `actElemT`. -/
def e4m3bsGemm1UngatedT (a : UAct) (x : MTerm) : MTerm := actElemT a x

/-- GEMM2 of the block-scale class: the plain conversion (`Fp8BsMoESpec.gemm2`). -/
def e4m3bsGemm2T (acc : MTerm) : MTerm := .cvt "bf16" acc

theorem e4m3bs_gemm1_is_bf16_chain (a : GAct) (g u : MTerm) : e4m3bsGemm1T a g u = gluT a g u := rfl
theorem e4m3bs_gemm1_silu_is_swiglu (g u : MTerm) : e4m3bsGemm1T .silu g u = swigluT g u := rfl
theorem e4m3bs_gemm1_ungated_is_relu2 (x : MTerm) : e4m3bsGemm1UngatedT .relu2 x = relu2T x := rfl

/-- **The intermediate the requantization reads is the bf16-rounded chain value**: the `fp8bs.quant` class applied to GEMM1's
rounded output (the declared second quantization). The fused epilogue quantizes the same rounded value, so fused-vs-standalone
is a placement question (the same term on the same values), not a spec one. -/
theorem e4m3bs_requant_input_is_rounded_chain (a : GAct) (g u : MTerm) : e4m3bsGemm1T a g u = .cvt "bf16" (gluPreT a g u) := rfl

/-- The grouped block-scale space: the dense block-scale class's records at swizzle 1 (the group scheduler has no raster swizzle)
with a tile of two promotion warpgroups (N 64 / 128 / 256: the narrow decode tiles have one subtile). -/
def groupedBlockScaleFp8OK (cfg : GEMM.Config) : Bool :=
  GEMM.ConfigOKFp8BS cfg && cfg.swizzle == 1 && (cfg.tn == 64 || cfg.tn == 128 || cfg.tn == 256)

theorem groupedBlockScaleFp8OK_imp_dense (cfg : GEMM.Config) (h : groupedBlockScaleFp8OK cfg = true) : GEMM.ConfigOKFp8BS cfg = true := by
  unfold groupedBlockScaleFp8OK at h
  rw [Bool.and_eq_true, Bool.and_eq_true] at h
  exact h.1.1

/-- The swapped decode space of the block-scale MoE class (`blocks/fp8bs_swap_kernel.py`; the MoE class's own records, which the
dense block-scale GEMM's rule does not admit): the weights as the MMA's A operand on a 64- or 128-row tile, the group's tokens as its
B operand in an 8 / 16 / 32 / 64 tile, 1-CTA, no cluster, swizzle 1, no split. The fp8 step is symmetric in its operands (the dense
block-scale class's swapped records, verified bit for bit against the exact model on both parts), so each record's result is
`promoteLoop`'s -- the per-block `mma32` partial promoted in block order; `tileResultKB` does not read the orientation, and
`space_invariantKB` over the union of both spaces is the statement the suites check. -/
def groupedBlockScaleFp8SwapOK (cfg : GEMM.Config) : Bool :=
  cfg.swapAb && (cfg.tm == 64 || cfg.tm == 128) && (cfg.tn == 8 || cfg.tn == 16 || cfg.tn == 32 || cfg.tn == 64) &&
  cfg.clusterM == 1 && cfg.clusterN == 1 && !cfg.twoCta && cfg.swizzle == 1 && cfg.clusterK == 1 && !cfg.ctaChunks

/-- The two spaces are disjoint (a record is dense-derived or swapped). -/
theorem groupedBlockScaleFp8_spaces_disjoint (cfg : GEMM.Config) :
    ¬ (groupedBlockScaleFp8OK cfg = true ∧ groupedBlockScaleFp8SwapOK cfg = true) := by
  intro ⟨h1, h2⟩
  have hd := groupedBlockScaleFp8OK_imp_dense cfg h1
  unfold GEMM.ConfigOKFp8BS at hd
  unfold groupedBlockScaleFp8SwapOK at h2
  simp only [Bool.and_eq_true] at hd h2
  have hs := h2.1.1.1.1.1.1.1.1
  simp [hs] at hd

end BIK.MoE
