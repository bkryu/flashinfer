import RLKernels.Canon
import RLKernels.Semantics
import RLKernels.MXQuant
/-! Axiom audit, printed at build time: every core theorem below depends only on Lean's standard axioms (`propext`,
`Quot.sound`, `Classical.choice`) and never on `sorryAx`. The enforced form of this audit is `lake exe axioms`
(`Axioms.lean`), which walks every declaration of the library and fails the build on a `sorry` or on an axiom outside
the policy (the decided tables -- `native_decide`, axiom `Lean.ofReduceBool` -- are admitted only for the generated
conformance cases and the named table theorems of `MXQuant.lean`). -/
#print axioms RLK.tree_split
#print axioms RLK.bfly_tree
#print axioms RLK.exec_eq_tree
#print axioms RLK.config_invariance
#print axioms RLK.dispatch_irrelevant
#print axioms RLK.eval_equiv
#print axioms RLK.tree_pad_good
#print axioms RLK.exec_pad_good
#print axioms RLK.execC_eq_treeC
#print axioms RLK.config_invarianceC
#print axioms RLK.treeC_add
#print axioms RLK.execC_pad_good
#print axioms RLK.MXQuant.block_amax_invariance
#print axioms RLK.MXQuant.block_outputs_invariant
#print axioms RLK.MXQuant.sf_bits_equal
#print axioms RLK.MXQuant.blockamax_padded_bits
#print axioms RLK.MXQuant.block_locality
#print axioms RLK.MXQuant.batch_invariance
#print axioms RLK.MXQuant.permutation_invariance
#print axioms RLK.MXQuant.subbatch_invariance
#print axioms RLK.MXQuant.layout128x4_bijective
#print axioms RLK.MXQuant.ue8m0_roundings
