"""The floating-point interface of the batch-invariant kernels, in its three backends.

* :mod:`.term`: symbolic terms that record which rounded operations were applied in which association (fingerprints,
  canonical equivalence, the Lean cases);
* :mod:`.exact`: exact IEEE-754 binary32 / bfloat16 / binary16 arithmetic on the CPU (the bit-exact reference);
* :mod:`.cute`: the CuTe-DSL primitives with explicit ``.rn`` rounding that the generated kernels are allowed to use,
  together with the declared conversion primitives and the dtype table they share.

Schedule code is written once against an ``fp`` object and runs under all three.
"""
