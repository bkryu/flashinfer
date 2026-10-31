import Lean
import BIKernels
/-! `lake exe axioms`: the enforced axiom policy of the library.

Every declaration of the `BIKernels.*` modules is checked:

* `sorryAx` anywhere fails the build;
* a theorem may depend on Lean's standard axioms (`propext`, `Classical.choice`, `Quot.sound`) and on nothing else,
  except the eight named exhaustive table theorems below, which are decided by `native_decide` and therefore depend on
  `Lean.ofReduceBool` as well: the axiom that `Lean.reduceBool c = true` implies `c = true`, i.e. it trusts the
  compiled evaluator (Lean's code generator and interpreter) for the evaluation of these closed Boolean terms. The
  generated conformance cases (`BIKernels.Cases`) use kernel `decide` and need no such trust.

The claim the policy enforces is therefore "Lean's standard axioms, plus `Lean.ofReduceBool` for eight named
exhaustive table checks", never "standard axioms only". -/
open Lean

def standardAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]
def decidedAxiom : Name := ``Lean.ofReduceBool
/-- The named `native_decide` theorems: the decided layout bijections (6144 and 8320 offsets), the UE8M0 fixed points /
roundings / saturation facts, and the three generated UE8M0 tables (531 magnitudes under three roundings; `Nat.log2`
is defined by well-founded recursion, which kernel `decide` does not evaluate). -/
def decidedTheorems : List Name :=
  [``BIK.MXQuant.layout128x4_bijective, ``BIK.MXQuant.layoutLinear_bijective, ``BIK.MXQuant.ue8m0_powers_fixed,
   ``BIK.MXQuant.ue8m0_roundings, ``BIK.MXQuant.ue8m0_saturates,
   ``BIK.ue8m0_model_matches_rp, ``BIK.ue8m0_model_matches_rz, ``BIK.ue8m0_model_matches_rn]
def moduleOf (env : Environment) (n : Name) : Option Name :=
  match env.getModuleIdxFor? n with
  | some idx => env.header.moduleNames[idx.toNat]?
  | none => none

def isOurs (m : Option Name) : Bool :=
  match m with
  | some m => m.toString.startsWith "BIKernels"
  | none => false

/-- Compiler-generated auxiliaries (`_cstage1` / `_cstage2` stages carrying `lcProof` / `lcUnreachable`, `_spec_n`
specialisations, `_elambda_n` closures, the `_auxLemma.n` helpers `native_decide` emits for a decided theorem, the
`_private` prefix of derived instances) are not proofs: every user theorem they belong to is checked by its own name. -/
def isGenerated (n : Name) : Bool :=
  n.components.any fun c => c.toString.startsWith "_"

def main : IO UInt32 := do
  initSearchPath (← findSysroot)
  let env ← importModules #[{ module := `BIKernels }] {} 0
  let mut failures : Array String := #[]
  let mut checked := 0
  let mut decided := 0
  let mut sorries := 0
  for (n, _) in env.constants.toList do
    let m := moduleOf env n
    unless isOurs m do continue
    if isGenerated n then continue
    let (_, st) := ((CollectAxioms.collect n).run env).run {}
    checked := checked + 1
    if st.axioms.contains ``sorryAx then
      sorries := sorries + 1
      failures := failures.push s!"{n}: depends on sorryAx"
    let allowDecided := decidedTheorems.contains n
    if st.axioms.contains decidedAxiom then decided := decided + 1
    for a in st.axioms do
      if a == ``sorryAx then continue
      if standardAxioms.contains a then continue
      if a == decidedAxiom && allowDecided then continue
      failures := failures.push s!"{n}: axiom {a} is outside the policy (module {m.getD .anonymous})"
  IO.println s!"axioms check: {checked} user declarations of BIKernels.*, {decided} named table theorems decided by native_decide (Lean.ofReduceBool), {sorries} with sorry, {failures.size} violations"
  for f in failures do
    IO.eprintln s!"  VIOLATION {f}"
  return (if failures.isEmpty then 0 else 1)
