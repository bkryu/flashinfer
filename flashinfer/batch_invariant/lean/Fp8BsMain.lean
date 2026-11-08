import BIKernels.QuantExport
import BIKernels.MoEFp8BsCases
/-! `lake exe fp8bsconformance`: print the Lean model's terms for the generated block-scale fp8 MoE cases (the `fp8bs.quant`
class's placements and chains, the MoE class's chains). The Python test (`tests/batch_invariant/test_moe_fp8bs_lean.py`)
compares these strings with the Python spec's / engine's S-expressions. -/
open BIK

def main : IO Unit := do
  for (name, ps) in fp8bsqCases do
    IO.println s!"fp8bsq {name} spec {osexpr (FP8BQ.blockAmax fp8bsqRow 0)}"
    IO.println s!"fp8bsq {name} kernel {osexpr (execC .fmaxc (flatten ps) (FP8BQ.blockLeaf fp8bsqRow 0) (zeroCoords (flatten ps)))}"
  IO.println s!"fp8bsqout s0 {osexpr (FP8BQ.specS fp8bsqRow 0)}"
  IO.println s!"fp8bsqout s1 {osexpr (FP8BQ.specS fp8bsqRow 1)}"
  IO.println s!"fp8bsqout code0 {osexpr (FP8BQ.specCode fp8bsqRow 0)}"
  IO.println s!"fp8bsqout codeL {osexpr (FP8BQ.specCode fp8bsqRow 255)}"
  for (name, t) in moeFp8BsCases do
    IO.println s!"moe {name} {t.sexpr}"
