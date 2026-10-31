"""Generate ``lean/BIKernels/Cases.lean`` from the Python side (the level-2 coupling of the README).

For the MX quantizer the file carries

* the three placements of the block digit (``mx_spec.block_amax_placement(V)``, V = 32 / 16 / 8: every configuration of
  :mod:`quantization.mx_schedule` uses one of them), the dataflow term the *Python engine* produces for each at unit (0, 0, 0)
  (``canon.trace``, the same ``init_state`` / ``apply_step`` code path the kernel executes at trace time) and the term the
  *Python spec* produces, with Lean theorems stating that the Lean model (an independent implementation) produces the same
  canonical S-expression (by kernel ``decide``), that the placement is valid (``PlacementOK``, ``decide``) and the engine's
  invariance theorem instantiated on it;
* the block amaxes the kernel reads before the ownership digit for one row configuration per block order, a 4-lanes-per-
  block configuration and the tile placement (``canon.trace(..., stop_before="blk")``);
* per recipe rule of the mxfp8 format the scale terms of blocks 0 and 1 and the codes 0 and 63 (``MXQuantSpec.outputs``);
* the UE8M0 conversion tables: ``exact.to_ue8m0`` on binary32 magnitudes ``a * 2^e`` under the three roundings, which
  ``MXQuant.ue8m0Byte`` must reproduce (the only generated theorems decided by ``native_decide``: ``Nat.log2`` is
  defined by well-founded recursion, which kernel ``decide`` does not evaluate).

Consequently a change to ``canon.apply_step``, a placement, the spec's chain or a recipe constant that is not reflected
in the Lean model makes ``lake build`` fail; and a regenerated file that differs from the committed one fails the
device-free conformance test (``tests/batch_invariant/test_mxfp8_quantize.py``), which needs no Lean toolchain.

    python -m flashinfer.batch_invariant.lean_cases          # rewrite lean/BIKernels/Cases.lean
"""

from __future__ import annotations

import os
import random
import struct
from typing import Dict, List, Tuple

from . import canon
from .fp import exact, term
from .quantization import mx_schedule as mxsched
from .quantization import mx_spec as mxspec

LEAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lean")
CASES_PATH = os.path.join(LEAN_DIR, "BIKernels", "Cases.lean")

MXQ_K = 64  # two blocks: the chains of block 0 and block 1, codes 0 and 63
MXQ_RULES = [
    (mxspec.DEFAULT_RULE, "MXQuant.Rule.te"),
    (mxspec.OCP_FLOOR_RULE, "MXQuant.Rule.ocpFloor"),
    (mxspec.FI_CLAMP_RULE, "MXQuant.Rule.fiClamp"),
    (mxspec.TE_FMA_RULE, "MXQuant.Rule.teFma"),
    (mxspec.ScaleRule(norm="div"), "MXQuant.Rule.div"),
    (mxspec.ScaleRule(round="rn"), "MXQuant.Rule.rn"),
]
MXQ_FMTS = [(mxspec.MXFP8, "MXQuant.Fmt.mxfp8")]
MXQ_OUT_WHICH = ("sf0", "sf1", "code0", "codeL")


def lean_factor(radix: int, level: str) -> str:
    bits = radix.bit_length() - 1
    kind = {"reg": ".reg", "lane": ".lane", "warp": ".xchg", "cta": ".xchg"}[level]
    return f"⟨{bits}, {kind}⟩"


def lean_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _with_ref(sx: str, sub: str, ref: str) -> str:
    """A Python S-expression as a Lean string expression: the single occurrence of ``sub`` (a term every output embeds)
    is replaced by a reference to its ``def`` so the generated file stays bounded; the concatenation is literal."""
    if sx == sub:
        return ref
    parts = sx.split(sub)
    if len(parts) != 2:
        return lean_str(sx)
    return f"{lean_str(parts[0])} ++ {ref} ++ {lean_str(parts[1])}"


def mxquant_cases() -> List[Dict]:
    """Per placement of the block digit: the Lean placement, the engine's block-amax term at unit (0, 0, 0), the spec's."""
    out = []
    sp = mxspec.MXQuantSpec(32, mxspec.MXFP8)
    red = sp.block_reduction(0, 0)
    spec_sx = term.sexpr(sp.block_amax(0, 0))
    for V in (32, 16, 8):
        pl = mxspec.block_amax_placement(V)
        ps = (
            "[["
            + ", ".join(lean_factor(f.radix, f.level) for f in pl.of("elem"))
            + "]]"
        )
        got = canon.trace(canon.plan(red, pl), sp.alg)[(0, 0, 0)]
        out.append(dict(name=f"V{V}", ps=ps, kernel=term.sexpr(got), spec=spec_sx))
    return out


def mxquant_out_cases() -> List[Dict]:
    """Per recipe rule (mxfp8): the scale terms of blocks 0 and 1 and the codes 0 and 63 at K = 64."""
    out = []
    for fmt, lf in MXQ_FMTS:
        for rule, lr in MXQ_RULES:
            sp = mxspec.MXQuantSpec(MXQ_K, fmt, rule)
            o = sp.outputs(0)
            out.append(
                dict(
                    fmt=fmt.name,
                    rule=rule.name,
                    lean_fmt=lf,
                    lean_rule=lr,
                    amax0=term.sexpr(sp.block_amax(0, 0)),
                    amax1=term.sexpr(sp.block_amax(0, 1)),
                    sf0=term.sexpr(o["sf"][0]),
                    sf1=term.sexpr(o["sf"][1]),
                    code0=term.sexpr(o["codes"][0]),
                    codeL=term.sexpr(o["codes"][MXQ_K - 1]),
                )
            )
    return out


def mxquant_block_cases() -> List[Tuple[int, str, str]]:
    """The block amax terms the MX kernel reads before the ownership digit at K = 96 (3 of 4 padded blocks present) for
    one row configuration per block order, a 4-lanes-per-block configuration and the tile placement: ``(block, name,
    sexpr)``, compared with ``MXQuant.blockAmax``."""
    out = []
    cases = (
        (96, mxsched.MXQuantConfig(4, 32, 32, "lane")),
        (96, mxsched.MXQuantConfig(2, 64, 16, "reg")),
        (96, mxsched.MXQuantConfig(16, 4, 8, "lane")),
        (
            32 * 64,
            mxsched.MXQuantConfig(32, 128, 32, mode="tile", tile_warps=16, tile_bpt=2),
        ),
    )
    for K, cfg in cases:
        sp = mxspec.MXQuantSpec(K, mxspec.MXFP8)
        red = sp.row_reduction(0)
        p = canon.plan(red, cfg.placement(K))
        states = canon.trace(p, sp.alg, stop_before="blk")
        for (cta, w, lane), st in states.items():
            coords = {"lane": lane, "warp": w, "cta": cta, "grid": 0}
            for key, val in st.items():
                b = p.digit_value("blk", coords, key)
                if val is not None:
                    kname = "_".join("n" if v is None else str(v) for v in key)
                    out.append(
                        (b, f"{cfg.describe()}_l{lane}_k{kname}", term.sexpr(val))
                    )
    seen, uniq = set(), []
    for b, name, sx in out:
        key = (b, name.split("_l")[0])
        if key not in seen and b < 4:
            seen.add(key)
            uniq.append((b, name, sx))
    return uniq


def ue8m0_cases() -> List[Tuple[int, int, int, int]]:
    """``(rnd, a, e, byte)``: the rounding (0 = rp, 1 = rz, 2 = rn), the exact magnitude ``a * 2^e`` of a positive binary32
    and the byte ``exact.to_ue8m0`` assigns, for powers of two, their neighbours, the 1.5 x 2^k ties and neighbours, the
    subnormal and saturation ends, ELEM_MAX x 2^k boundaries and random values; Lean's ``MXQuant.ue8m0Byte`` must agree."""
    rng = random.Random(1)
    vals = []
    for k in [
        -149,
        -140,
        -130,
        -128,
        -127,
        -126,
        -125,
        -120,
        -100,
        -30,
        -10,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        10,
        30,
        100,
        120,
        125,
        126,
        127,
    ]:
        m = 2.0**k
        bits = struct.unpack("<I", struct.pack("<f", m))[0]
        for d in (-1, 0, 1):
            vals.append(struct.unpack("<f", struct.pack("<I", bits + d))[0])
        if k < 127:
            t = struct.unpack("<I", struct.pack("<f", 1.5 * m))[0]
            for d in (-1, 0, 1):
                vals.append(struct.unpack("<f", struct.pack("<I", t + d))[0])
    vals += (
        [448.0 * 2.0**k for k in (-10, 0, 7)]
        + [6.0 * 2.0**k for k in (-10, 0, 7)]
        + [exact.F32_MAX]
    )
    vals += [
        struct.unpack(
            "<f", struct.pack("<f", rng.uniform(0, 1) * 2.0 ** rng.randint(-140, 127))
        )[0]
        for _ in range(24)
    ]
    out = []
    for v in vals:
        if not (v > 0) or v == float("inf"):
            continue
        num, den = float(v).as_integer_ratio()
        e = -(den.bit_length() - 1)
        assert den == 1 << -e
        for ri, rnd in enumerate(exact.UE8M0_ROUNDINGS):
            out.append((ri, num, e, exact.ue8m0_byte(exact.to_ue8m0(v, rnd))))
    return out


def render() -> str:
    """The content of ``Cases.lean``."""
    L = [
        "import BIKernels.Canon",
        "import BIKernels.MXQuant",
        "import BIKernels.Export",
        "namespace BIK",
        "-- GENERATED by flashinfer/batch_invariant/lean_cases.py from the Python engine / specs; do not edit.",
        "set_option maxRecDepth 4096",
        "",
        "def quantRow : Nat → Term := fun i => Term.x i",
        "",
    ]
    L.append(
        "/-! The MX quantization classes (`quant/mx_spec.py` vs `MXQuant.lean`): the block amax S-expression, the three block "
        "placements' kernel terms (the engine's `execC .fmaxc` at unit (0,0,0)), the block amaxes the kernel reads before the "
        "ownership digit, the output chains per format and recipe rule, and the UE8M0 conversion table. -/"
    )
    mqs, mqo, mqb = mxquant_cases(), mxquant_out_cases(), mxquant_block_cases()
    amax_sx = mqs[0]["spec"]
    L.append(f"def mxqPy_amax0 : String := {lean_str(amax_sx)}")
    L.append(f"def mxqPy_amax1 : String := {lean_str(mqo[0]['amax1'])}")
    L.append(
        "theorem mxq_spec_amax0 : osexpr (MXQuant.blockAmax quantRow 0) = mxqPy_amax0 := by decide"
    )
    L.append(
        "theorem mxq_spec_amax1 : osexpr (MXQuant.blockAmax quantRow 1) = mxqPy_amax1 := by decide"
    )
    for i, c in enumerate(mqs):
        assert c["spec"] == amax_sx
        rhs = "mxqPy_amax0" if c["kernel"] == amax_sx else lean_str(c["kernel"])
        L.append(f"def mxqPl{i} : List (List Factor) := {c['ps']}  -- {c['name']}")
        L.append(f"theorem mxq{i}_ok : PlacementOK [5] mxqPl{i} := by decide")
        L.append(
            f"theorem mxq{i}_kernel : osexpr (execC .fmaxc (flatten mxqPl{i}) (MXQuant.blockLeaf quantRow 0) (zeroCoords (flatten mxqPl{i}))) = {rhs} := by decide"
        )
        L.append(
            f"theorem mxq{i}_inv : OEquiv (execC .fmaxc (flatten mxqPl{i}) (MXQuant.blockLeaf quantRow 0) (zeroCoords (flatten mxqPl{i}))) (MXQuant.blockAmax quantRow 0) :="
        )
        L.append(
            f"  MXQuant.block_amax_invariance quantRow 0 mxqPl{i} _ mxq{i}_ok (coordsOK_zero _)"
        )
    L.append(
        "def mxqCases : List (String × List (List Factor)) := ["
        + ", ".join(f'("{c["name"]}", mxqPl{i})' for i, c in enumerate(mqs))
        + "]"
    )

    def _mx_ref(sx, amax0, amax1):
        if amax0 in sx:
            return _with_ref(sx, amax0, "mxqPy_amax0")
        return _with_ref(sx, amax1, "mxqPy_amax1")

    for c in mqo:
        tag = f"mxq_{c['fmt']}_{c['rule'].replace('-', '_')}"
        f, r = c["lean_fmt"], c["lean_rule"]
        L.append(
            f"theorem {tag}_sf0 : osexpr (MXQuant.specSf {f} {r} quantRow 0) = {_mx_ref(c['sf0'], c['amax0'], c['amax1'])} := by decide"
        )
        L.append(
            f"theorem {tag}_sf1 : osexpr (MXQuant.specSf {f} {r} quantRow 1) = {_mx_ref(c['sf1'], c['amax0'], c['amax1'])} := by decide"
        )
        L.append(
            f"theorem {tag}_code0 : osexpr (MXQuant.specCode {f} {r} quantRow 0) = {_mx_ref(c['code0'], c['amax0'], c['amax1'])} := by decide"
        )
        L.append(
            f"theorem {tag}_codeL : osexpr (MXQuant.specCode {f} {r} quantRow {MXQ_K - 1}) = {_mx_ref(c['codeL'], c['amax0'], c['amax1'])} := by decide"
        )
    L.append(
        "def mxqOutCases : List (String × String × MXQuant.Fmt × MXQuant.Rule) := ["
        + ", ".join(
            f'("{c["fmt"]}", "{c["rule"]}", {c["lean_fmt"]}, {c["lean_rule"]})'
            for c in mqo
        )
        + "]"
    )
    for i, (b, name, sx) in enumerate(mqb):
        L.append(
            f"theorem mxqblock{i} : osexpr (MXQuant.blockAmax quantRow {b}) = {lean_str(sx)} := by decide  -- {name}"
        )
    L.append(
        "def mxqBlockCases : List (Nat × String) := ["
        + ", ".join(f'({b}, "{name}")' for b, name, _ in mqb)
        + "]"
    )
    u8 = ue8m0_cases()
    L.append(
        "/-! `(a, e, byte)` per rounding: the Python model `exact.to_ue8m0` on binary32 magnitudes `a * 2^e`; `MXQuant.ue8m0Byte` must agree. -/"
    )
    for ri, rnd in enumerate(("rp", "rz", "rn")):
        rows = [(a, e, c) for r, a, e, c in u8 if r == ri]
        L.append("set_option maxHeartbeats 2000000 in")
        L.append(
            f"def ue8m0Table_{rnd} : List (Nat × Int × Nat) := ["
            + ", ".join(f"({a}, {e}, {c})" for a, e, c in rows)
            + "]"
        )
        L.append(
            f"theorem ue8m0_model_matches_{rnd} : ue8m0Table_{rnd}.all (fun ⟨a, e, c⟩ => MXQuant.ue8m0Byte .{rnd} a e == c) = true := by native_decide"
        )
    L.append("end BIK")
    return "\n".join(L) + "\n"


def summary() -> str:
    return (
        f"lean_cases coverage: mxquant placements={len(mxquant_cases())} output cases={len(mxquant_out_cases())} "
        f"block cases={len(mxquant_block_cases())} ue8m0 cases={len(ue8m0_cases())}"
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
