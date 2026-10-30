import Lake
open Lake DSL

package rlkernels

@[default_target]
lean_lib RLKernels

lean_exe conformance where
  root := `Main

lean_exe axioms where
  root := `Axioms
  supportInterpreter := true
