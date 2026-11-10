# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Generate ``lean/BIKernels/MoEFp8BsCases.lean`` from the Python side: the block-scale fp8 MoE classes' conformance cases.

The committed file is what the Python spec emits (``tests/batch_invariant/test_moe_fp8bs_lean.py`` checks it byte for byte without
a toolchain, and with ``lake`` builds it and compares the Lean model's terms through ``lake exe fp8bsconformance``):

* the ``fp8bs.quant`` class (``spec_fp8bs.py`` vs ``FP8BlockQuant.lean``): every placement of the block digit
  (``block_amax_placement(V)``) with the engine's block-amax term at unit (0, 0, 0) and the spec's, the placements' validity
  and their invariance theorems instantiated, the scale and code terms of two blocks (the block index enters the leaves only),
  the half-tree split the fused requantization relies on;
* the MoE class (``spec_fp8bs.Fp8BsMoESpec`` vs ``MoEFp8Bs.lean``): GEMM1's gated and ungated chains are the bf16 classes',
  GEMM2 the plain conversion;
* the grouped block-scale space (``schedule_fp8bs.grouped_knobs_fp8bs``): every record checked against the kernel's static
  constraints (``MoE.groupedBlockScaleFp8OK``), a member of the dense block-scale space, the ``space_invariantKB`` theorem
  instantiated on it; the dispatcher's picks over the bench geometries at every nominal part, both requantization spellings,
  members of the space.

    python -m flashinfer.fused_moe.batch_invariant.lean_cases_fp8bs     # rewrite lean/BIKernels/MoEFp8BsCases.lean
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from ...batch_invariant import canon
from ...batch_invariant.quantization import qterm
from ...batch_invariant.quantization.lean_cases import _with_ref, lean_factor, lean_str
from . import mterm
from . import spec_fp8bs as B
from .lean_cases import LEAN_DIR

CASES_PATH = os.path.join(LEAN_DIR, "BIKernels", "MoEFp8BsCases.lean")
FP8BSQ_K = 256  # two blocks: the chains of block 0 and block 1, codes 0 and 255
FP8BSQ_OUT_WHICH = ("s0", "s1", "code0", "codeL")
#: The geometries the dispatcher's membership is stated over: (H, I, E, k), the bench grid and the real-model targets.
GEOMS = (
    (2048, 768, 128, 8),
    (4096, 14336, 8, 2),
    (7168, 2048, 256, 8),
    (5120, 8192, 16, 1),
    (512, 256, 64, 6),
    (1024, 512, 64, 8),
    (2560, 640, 512, 10),
)
TS = (1, 7, 64, 300, 800, 4096, 16384)
ACTS = (("silu", True), ("gelu_tanh", True), ("relu2", False))


def fp8bsq_cases() -> List[Dict]:
    """Per placement of the block digit (a whole block per thread down to four elements per lane; the standalone kernel's
    ``V = 8``, the fused epilogue's half blocks ``V = 64``): the Lean placement, the engine's block-amax term at unit (0, 0, 0)
    and the spec's."""
    out = []
    sp = B.BSQuantSpec(B.BLOCK)
    red = sp.block_reduction(0, 0)
    spec_sx = qterm.sexpr(sp.block_amax(0, 0))
    for V in B.PLACEMENT_VS:
        pl = B.block_amax_placement(V)
        ps = (
            "[["
            + ", ".join(lean_factor(f.radix, f.level) for f in pl.of("elem"))
            + "]]"
        )
        got = canon.trace(canon.plan(red, pl), sp.alg)[(0, 0, 0)]
        out.append(dict(name=f"V{V}", ps=ps, kernel=qterm.sexpr(got), spec=spec_sx))
    return out


def fp8bsq_out_cases() -> Dict[str, str]:
    """The scale terms of blocks 0 and 1 and the codes 0 and ``K - 1`` at ``K = FP8BSQ_K``, plus the two block amaxes the terms
    embed."""
    sp = B.BSQuantSpec(FP8BSQ_K)
    o = sp.outputs(0)
    return dict(
        amax0=qterm.sexpr(sp.block_amax(0, 0)),
        amax1=qterm.sexpr(sp.block_amax(0, 1)),
        s0=qterm.sexpr(o["s"][0]),
        s1=qterm.sexpr(o["s"][1]),
        code0=qterm.sexpr(o["codes"][0]),
        codeL=qterm.sexpr(o["codes"][FP8BSQ_K - 1]),
    )


def moe_cases() -> List[Tuple[str, str, str]]:
    """``(name, Python S-expression, Lean term)`` of the class's chains: the gated GEMM1 (silu), the ungated GEMM1 (relu2),
    GEMM2."""
    alg = mterm.TermAlgebra()
    s = B.Fp8BsMoESpec(256, 128, 2, 8, alg=alg)
    su = B.Fp8BsMoESpec(256, 128, 2, 8, act="relu2", gated=False, alg=alg)
    return [
        (
            "e4m3bs_gemm1_silu",
            mterm.sexpr(s.gemm1()),
            'MoE.e4m3bsGemm1T .silu (.leaf "g" [0, 0]) (.leaf "u" [0, 0])',
        ),
        (
            "e4m3bs_gemm1_relu2",
            mterm.sexpr(su.gemm1()),
            'MoE.e4m3bsGemm1UngatedT .relu2 (.leaf "u" [0, 0])',
        ),
        (
            "e4m3bs_gemm2",
            mterm.sexpr(s.gemm2()),
            'MoE.e4m3bsGemm2T (.leaf "acc2" [0, 0])',
        ),
    ]


def space():
    from .schedule_fp8bs import grouped_knobs_fp8bs

    return grouped_knobs_fp8bs()


def swap_space():
    from .schedule_fp8bs import swapped_knobs_fp8bs

    return swapped_knobs_fp8bs()


def dispatch_cases() -> Dict:
    """Every record the block-scale dispatcher returns over ``GEOMS`` x ``TS`` x every nominal part x the chains x both
    requantization spellings, with the first (T, H, I, E) that selects it."""
    from ...gemm.batch_invariant.heuristic import PARTS
    from .blocks.fp8bs_compile import (
        GEMM2_CHAIN,
        can_implement_fp8bs,
        gemm1_chain,
    )
    from .schedule_fp8bs import default_configs_fp8bs

    disp: Dict = {}
    for H, I, E, k in GEOMS:
        for T in TS:
            for family, sm in PARTS:
                for act, gated in ACTS:
                    for fused in (False, True):
                        c1, c2 = default_configs_fp8bs(
                            T,
                            k,
                            H,
                            I,
                            E,
                            sm,
                            family,
                            act,
                            gated,
                            (),
                            fused,
                            nominal=True,
                        )
                        assert can_implement_fp8bs(
                            c1,
                            2 * I if gated else I,
                            H,
                            E,
                            gemm1_chain(act, (), gated, fused),
                        )
                        assert can_implement_fp8bs(c2, H, I, E, GEMM2_CHAIN)
                        disp.setdefault(c1, (T, H, I, E))
                        disp.setdefault(c2, (T, H, I, E))
    return disp


def render() -> str:
    """The content of ``MoEFp8BsCases.lean``."""
    L = [
        "import BIKernels.Canon",
        "import BIKernels.QuantExport",
        "import BIKernels.FP8BlockQuant",
        "import BIKernels.MoE",
        "import BIKernels.MoEFp8Bs",
        "import BIKernels.Cases",
        "namespace BIK",
        "-- GENERATED by flashinfer/fused_moe/batch_invariant/lean_cases_fp8bs.py from the Python spec / engine; do not edit.",
        "set_option maxRecDepth 8192",
        "open GEMM",
        "",
        "def fp8bsqRow : Nat → Term := fun i => Term.x i",
        "",
        "/-! The fp8 block quantizer class `fp8bs.quant` (`fused_moe/batch_invariant/spec_fp8bs.py` vs `FP8BlockQuant.lean`): the block "
        "amax S-expression of blocks 0 and 1, every placement's kernel term (the engine's `execC .fmaxc` at unit (0,0,0)) and spec "
        "term with the invariance theorem instantiated, the scale and code terms, the half-tree split of the fused requantization. -/",
    ]
    bqs, bqo = fp8bsq_cases(), fp8bsq_out_cases()
    L.append(f"def fp8bsqPy_amax0 : String := {lean_str(bqo['amax0'])}")
    L.append(f"def fp8bsqPy_amax1 : String := {lean_str(bqo['amax1'])}")
    L.append(
        "theorem fp8bsq_spec_amax0 : osexpr (FP8BQ.blockAmax fp8bsqRow 0) = fp8bsqPy_amax0 := by decide +kernel"
    )
    L.append(
        "theorem fp8bsq_spec_amax1 : osexpr (FP8BQ.blockAmax fp8bsqRow 1) = fp8bsqPy_amax1 := by decide +kernel"
    )
    for i, c in enumerate(bqs):
        rhs = "fp8bsqPy_amax0" if c["kernel"] == bqo["amax0"] else lean_str(c["kernel"])
        L.append(f"def fp8bsqPl{i} : List (List Factor) := {c['ps']}  -- {c['name']}")
        L.append(
            f"theorem fp8bsq{i}_ok : PlacementOK [7] fp8bsqPl{i} := by decide +kernel"
        )
        L.append(
            f"theorem fp8bsq{i}_kernel : osexpr (execC .fmaxc (flatten fp8bsqPl{i}) (FP8BQ.blockLeaf fp8bsqRow 0) (zeroCoords (flatten fp8bsqPl{i}))) = {rhs} := by decide +kernel"
        )
        L.append(
            f"theorem fp8bsq{i}_inv : OEquiv (execC .fmaxc (flatten fp8bsqPl{i}) (FP8BQ.blockLeaf fp8bsqRow 0) (zeroCoords (flatten fp8bsqPl{i}))) (FP8BQ.blockAmax fp8bsqRow 0) :="
        )
        L.append(
            f"  FP8BQ.block_amax_invariance fp8bsqRow 0 fp8bsqPl{i} _ fp8bsq{i}_ok (coordsOK_zero _)"
        )
    L.append(
        "def fp8bsqCases : List (String × List (List Factor)) := ["
        + ", ".join(f'("{c["name"]}", fp8bsqPl{i})' for i, c in enumerate(bqs))
        + "]"
    )

    def ref(sx: str) -> str:
        return (
            _with_ref(sx, bqo["amax0"], "fp8bsqPy_amax0")
            if bqo["amax0"] in sx
            else _with_ref(sx, bqo["amax1"], "fp8bsqPy_amax1")
        )

    L.append(
        f"theorem fp8bsq_s0 : osexpr (FP8BQ.specS fp8bsqRow 0) = {ref(bqo['s0'])} := by decide +kernel"
    )
    L.append(
        f"theorem fp8bsq_s1 : osexpr (FP8BQ.specS fp8bsqRow 1) = {ref(bqo['s1'])} := by decide +kernel"
    )
    L.append(
        f"theorem fp8bsq_code0 : osexpr (FP8BQ.specCode fp8bsqRow 0) = {ref(bqo['code0'])} := by decide +kernel"
    )
    L.append(
        f"theorem fp8bsq_codeL : osexpr (FP8BQ.specCode fp8bsqRow {FP8BSQ_K - 1}) = {ref(bqo['codeL'])} := by decide +kernel"
    )
    L.append(
        "theorem fp8bsq_halves : ∀ (b : Nat), FP8BQ.blockAmax fp8bsqRow b = ocomb .fmaxc (FP8BQ.halfAmax fp8bsqRow b 0) "
        "(FP8BQ.halfAmax fp8bsqRow b 1) := fun b => FP8BQ.blockAmax_halves fp8bsqRow b"
    )
    L.append("")
    L.append(
        "/-! The MoE class (`spec_fp8bs.Fp8BsMoESpec` vs `MoEFp8Bs.lean`): GEMM1's gated and ungated chains are the bf16 classes', "
        "GEMM2 the plain conversion (`mterm.sexpr` vs `MTerm.sexpr`). -/"
    )
    mcs = moe_cases()
    for name, sx, lt in mcs:
        L.append(
            f"theorem moe_{name} : ({lt}).sexpr = {lean_str(sx)} := by decide +kernel"
        )
    L.append(
        "def moeFp8BsCases : List (String × MoE.MTerm) := ["
        + ", ".join(f'("{name}", {lt})' for name, _, lt in mcs)
        + "]"
    )
    L.append("")
    sp = space()
    L.append(
        "/-! The grouped block-scale space (`schedule_fp8bs.grouped_knobs_fp8bs`: the dense block-scale class's records at swizzle 1 "
        "with a tile of two promotion warpgroups): every record checked against the kernel's static constraints, a member of the "
        "dense block-scale space, the `space_invariantKB` theorem instantiated on it. -/"
    )
    L.append(
        "def moeE4m3BsSpace : List Config := [" + ", ".join(k.lean() for k in sp) + "]"
    )
    L.append(
        f"theorem moe_e4m3bs_space_size : moeE4m3BsSpace.length = {len(sp)} := by decide"
    )
    L.append(
        "theorem moe_e4m3bs_space_ok : moeE4m3BsSpace.all MoE.groupedBlockScaleFp8OK = true := by decide +kernel"
    )
    L.append(
        "theorem moe_e4m3bs_space_in_dense : moeE4m3BsSpace.all (fun c => gemmFp8BsSpace.any (fun s => s == c)) = true := by decide +kernel"
    )
    L.append(
        "theorem moe_e4m3bs_space_invariant : ∀ c₁ ∈ moeE4m3BsSpace, ∀ c₂ ∈ moeE4m3BsSpace, ∀ (S₁ S₂ : Nat) (p q : GTerm) "
        "(a : RowData) (b : ColData) (KB : Nat), tileResultKB c₁ S₁ p a b KB = tileResultKB c₂ S₂ q a b KB := "
        "space_invariantKB moeE4m3BsSpace"
    )
    sw = swap_space()
    L.append(
        "/-! The swapped decode space of the class (`blocks/fp8bs_swap_kernel.py`: the weights on the MMA's M in a 64- / 128-row "
        "tile, the tokens as the N tile): every record checked against the swapped body's static constraints, disjoint from the "
        "dense-derived records, the tile result invariant over the union of the two spaces (the promotion loop does not read the "
        "orientation). -/"
    )
    L.append(
        "def moeE4m3BsSwapSpace : List Config := ["
        + ", ".join(k.lean() for k in sw)
        + "]"
    )
    L.append(
        f"theorem moe_e4m3bs_swap_space_size : moeE4m3BsSwapSpace.length = {len(sw)} := by decide"
    )
    L.append(
        "theorem moe_e4m3bs_swap_space_ok : moeE4m3BsSwapSpace.all MoE.groupedBlockScaleFp8SwapOK = true := by decide +kernel"
    )
    L.append(
        "theorem moe_e4m3bs_swap_space_not_dense : moeE4m3BsSwapSpace.all (fun c => !MoE.groupedBlockScaleFp8OK c) = true := by decide +kernel"
    )
    L.append(
        "theorem moe_e4m3bs_both_spaces_invariant : ∀ c₁ ∈ moeE4m3BsSpace ++ moeE4m3BsSwapSpace, ∀ c₂ ∈ moeE4m3BsSpace ++ moeE4m3BsSwapSpace, "
        "∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (KB : Nat), tileResultKB c₁ S₁ p a b KB = tileResultKB c₂ S₂ q a b KB := "
        "space_invariantKB (moeE4m3BsSpace ++ moeE4m3BsSwapSpace)"
    )
    disp = dispatch_cases()
    L.append(
        "/-! Every knob record the block-scale MoE dispatcher returns over the bench geometries x token counts x every nominal "
        "part x the chains x both requantization spellings, with the first (T, H, I, E) that selects it: a member of one of the "
        "two spaces above (R3 membership). -/"
    )
    L.append(
        "def moeE4m3BsDispatch : List (Nat × Nat × Nat × Nat × Config) := ["
        + ", ".join(
            f"({T}, {H}, {I}, {E}, {k.lean()})" for k, (T, H, I, E) in disp.items()
        )
        + "]"
    )
    L.append(
        "theorem moe_e4m3bs_dispatch_in_space : moeE4m3BsDispatch.all (fun ⟨_, _, _, _, c⟩ => (moeE4m3BsSpace ++ moeE4m3BsSwapSpace).any (fun s => s == c)) = true := by decide +kernel"
    )
    L.append("end BIK")
    return "\n".join(L) + "\n"


def summary() -> str:
    return (
        f"fp8bs lean_cases coverage: quant placements={len(fp8bsq_cases())} quant output cases={len(FP8BSQ_OUT_WHICH)} "
        f"moe chains={len(moe_cases())} space={len(space())} dispatch records={len(dispatch_cases())}"
    )


def main(out_path: str = CASES_PATH) -> str:
    content = render()
    with open(out_path, "w") as f:
        f.write(content)
    s = summary()
    print(s)
    return s


if __name__ == "__main__":
    main()
