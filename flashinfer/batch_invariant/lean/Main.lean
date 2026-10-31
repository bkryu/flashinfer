import BIKernels.Export
import BIKernels.Cases
/-! `lake exe conformance`: print the Lean model's terms for the generated cases. The Python test
(`tests/batch_invariant/test_mxfp8_quantize.py`, the Lean tier) compares these strings with the Python engine's / spec's
S-expressions. -/
open BIK

def main : IO Unit := do
  let qrow : Nat → Term := fun i => Term.x i
  -- the MX quantization classes (MXQuant.lean): the block placements' kernel terms, the chains per format and rule
  for (name, ps) in mxqCases do
    IO.println s!"mxq {name} spec {osexpr (MXQuant.blockAmax qrow 0)}"
    IO.println s!"mxq {name} kernel {osexpr (execC .fmaxc (flatten ps) (MXQuant.blockLeaf qrow 0) (zeroCoords (flatten ps)))}"
  for (fname, rname, f, r) in mxqOutCases do
    IO.println s!"mxqout {fname} {rname} sf0 {osexpr (MXQuant.specSf f r qrow 0)}"
    IO.println s!"mxqout {fname} {rname} sf1 {osexpr (MXQuant.specSf f r qrow 1)}"
    IO.println s!"mxqout {fname} {rname} code0 {osexpr (MXQuant.specCode f r qrow 0)}"
    IO.println s!"mxqout {fname} {rname} codeL {osexpr (MXQuant.specCode f r qrow 63)}"
  for (b, name) in mxqBlockCases do
    IO.println s!"mxqblock {b} {name} {osexpr (MXQuant.blockAmax qrow b)}"
