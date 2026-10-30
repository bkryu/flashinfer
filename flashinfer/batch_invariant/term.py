"""Hash-consed symbolic terms over uninterpreted floating-point primitives.

A term records *which* rounded operations were applied in *which* association,
not their numeric value. Two configs are canonical-equivalent iff they build the
same term for every output element. Commutative ops normalise their operand order
(IEEE add/mul are bitwise commutative), so butterfly shuffles compare equal across
lanes. Terms can also be evaluated bit-exactly with :mod:`flashinfer.batch_invariant.fp32`.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from . import fp32


@dataclass(frozen=True, eq=False)
class Term:
    op: str
    args: Tuple[Any, ...]
    uid: int

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.op}#{self.uid}"


_UIDS = itertools.count()  # global: terms from different algebras may be mixed
_TABLE: Dict[
    Tuple[Any, ...], Term
] = {}  # global interning: structural equality == identity


class TermAlgebra:
    """Builds interned terms.

    ``drop_pos_zero_in_add`` enables the rewrite ``add(v, +0.0) -> v``; it is only
    sound where every summand is provably ``>= +0.0`` (sum-of-squares trees). Use a
    separate algebra instance for the epilogue so the rewrite cannot leak.
    """

    def __init__(self, drop_pos_zero_in_add: bool = False):
        self._table = _TABLE
        self.drop_pos_zero_in_add = drop_pos_zero_in_add

    # -- interning -------------------------------------------------------------
    def _mk(self, op: str, *args: Any) -> Term:
        key = (op,) + tuple(a.uid if isinstance(a, Term) else ("c", a) for a in args)
        t = self._table.get(key)
        if t is None:
            t = Term(op, args, next(_UIDS))
            self._table[key] = t
        return t

    @staticmethod
    def _sorted2(a: Term, b: Term):
        return (a, b) if a.uid <= b.uid else (b, a)

    # -- leaves ----------------------------------------------------------------
    def x(self, row: int, i: int) -> Term:
        return self._mk("x", row, i)

    def w(self, i: int) -> Term:
        return self._mk("w", i)

    def r(self, row: int, i: int) -> Term:
        """Residual input element (fused-add variants)."""
        return self._mk("r", row, i)

    def const(self, value: float) -> Term:
        # constants are binary32 values (F03); keyed on the bit pattern so +0.0 and -0.0 differ
        value = fp32.to_f32(value)
        return self._mk("const", value, math.copysign(1.0, value))

    def sym(self, name: str) -> Term:
        """A symbolic runtime scalar (e.g. eps); evaluation needs a binding."""
        return self._mk("sym", name)

    def is_pos_zero(self, t: Term) -> bool:
        return (
            t.op == "const" and t.args[0] == 0.0 and math.copysign(1.0, t.args[0]) > 0
        )

    # -- f32 ops (all .rn) -----------------------------------------------------
    def add(self, a: Term, b: Term) -> Term:
        if self.drop_pos_zero_in_add:
            if self.is_pos_zero(a):
                return b
            if self.is_pos_zero(b):
                return a
        a, b = self._sorted2(a, b)
        return self._mk("add", a, b)

    def mul(self, a: Term, b: Term) -> Term:
        a, b = self._sorted2(a, b)
        return self._mk("mul", a, b)

    def fma(self, a: Term, b: Term, c: Term) -> Term:
        a, b = self._sorted2(a, b)
        return self._mk("fma", a, b, c)

    def div(self, a: Term, b: Term) -> Term:
        return self._mk("div", a, b)

    def sqrt(self, a: Term) -> Term:
        return self._mk("sqrt", a)

    def fmaxc(self, a: Term, b: Term) -> Term:
        """The attention class's ``max.f32`` (NVVM ``fmax``, maximumNumber) as a *commutative join*: on
        SM107 it returns ``+0`` for both ``max(+0, -0)`` and ``max(-0, +0)`` (measured), so it is
        bitwise commutative on every non-NaN pair and the operand order is normalised like ``add``
        (the engine's lane / warp placements then verify canonical). Printed as ``(fmax a b)`` with
        the operands sorted; ``fmax`` below is the MoE family's order-preserving clamp."""
        a, b = self._sorted2(a, b)
        return self._mk("fmaxc", a, b)

    def ex2(self, a: Term) -> Term:
        """``ex2.approx.ftz.f32``: an opaque hardware primitive (no CPU model; a device helper
        supplies its values to the exact evaluator). ``ex2(+0) = ex2(-0) = 1``, ``ex2(-inf) = 0``."""
        return self._mk("ex2", a)

    def cvt_out(self, a: Term, fmt_name: str) -> Term:
        return self._mk("cvt_out", a, fmt_name)

    def op(self, name: str, *args) -> Term:
        """An opaque, non-commutative primitive (e.g. one ``tcgen05.mma`` K-step, ``ex2.approx``).
        Its internal rounding is hardware-defined; only the dataflow around it is specified."""
        return self._mk("op:" + name, *args)

    # -- exact (rounding-free) primitives, added for the MoE family -------------
    def neg(self, a: Term) -> Term:
        """Sign flip (exact)."""
        return self._mk("neg", a)

    def fmin(self, a: Term, b: Term) -> Term:
        """``min.f32`` (exact, not commutativity-normalised: the operand order decides which
        signed zero is returned; the specs only clamp against non-zero constants)."""
        return self._mk("fmin", a, b)

    def fmax(self, a: Term, b: Term) -> Term:
        return self._mk("fmax", a, b)

    def exp2i(self, k: Term) -> Term:
        """The binary32 ``2^k`` for an integer-valued ``k`` in ``[-126, 127]`` (exact; built from
        the exponent bits on the device, ``ldexp`` on the CPU)."""
        return self._mk("exp2i", k)

    def rowmax(self, *vals: Term) -> Term:
        """The maximum of ``vals`` (n-ary, exact in value for every association; the sign of a
        zero maximum is unspecified and, where the specs use it, immaterial: see
        ``flashinfer/batch_invariant/moe/spec.py``)."""
        return self._mk("rowmax", *vals)

    # -- exact primitives added for the NVFP4 quantization family ---------------
    def fabs(self, a: Term) -> Term:
        """``abs.f32`` (exact: the sign bit is cleared)."""
        return self._mk("fabs", a)

    # ``fmaxc`` (the commutative max join, shared with the attention class) is defined above; the
    # quantizer's leaves are ``fabs(x) >= +0``, where it coincides with the MoE's order-sensitive ``fmax``.

    def nzor(self, a: Term, b: Term) -> Term:
        """``a`` if ``a != 0`` else ``b`` (an exact comparison-and-select, ``setp.ne`` + ``selp``)."""
        return self._mk("nzor", a, b)

    def leaf(self, name: str, *coords) -> Term:
        """A named input leaf with integer coordinates (``leaf("l", e)``: a router logit;
        ``leaf("y", r, h)``: an expert output). Evaluation binds ``syms["leaf:" + name]`` to a
        function of the coordinates."""
        return self._mk("leaf:" + name, *coords)


def evaluate(
    t: Term,
    x_val,
    w_val,
    memo: Dict[int, float] | None = None,
    syms: Dict[str, Any] | None = None,
    r_val=None,
) -> float:
    """Bit-exact evaluation of a term with :mod:`flashinfer.batch_invariant.fp32`.

    ``x_val(row, i)`` and ``w_val(i)`` return the float32 value of the inputs;
    ``syms`` binds symbolic scalars created with :meth:`TermAlgebra.sym`.
    """
    if memo is None:
        memo = {}
    syms = syms or {}
    stack = [t]
    while stack:
        cur = stack[-1]
        if cur.uid in memo:
            stack.pop()
            continue
        deps = [a for a in cur.args if isinstance(a, Term) and a.uid not in memo]
        if deps:
            stack.extend(deps)
            continue
        stack.pop()
        op = cur.op
        if op == "x":
            v = float(x_val(cur.args[0], cur.args[1]))
        elif op == "w":
            v = float(w_val(cur.args[0]))
        elif op == "r":
            v = float(r_val(cur.args[0], cur.args[1]))
        elif op == "const":
            v = cur.args[0]
        elif op == "sym":
            v = float(syms[cur.args[0]])
        elif op == "add":
            v = fp32.add(memo[cur.args[0].uid], memo[cur.args[1].uid])
        elif op == "mul":
            v = fp32.mul(memo[cur.args[0].uid], memo[cur.args[1].uid])
        elif op == "fma":
            v = fp32.fma(
                memo[cur.args[0].uid], memo[cur.args[1].uid], memo[cur.args[2].uid]
            )
        elif op == "div":
            v = fp32.div(memo[cur.args[0].uid], memo[cur.args[1].uid])
        elif op == "sqrt":
            v = fp32.sqrt(memo[cur.args[0].uid])
        elif op == "fmaxc":
            v = fp32.fmaxc(memo[cur.args[0].uid], memo[cur.args[1].uid])
        elif op == "ex2":
            fn = (syms or {}).get("ex2")
            if fn is None:
                raise NotImplementedError(
                    "no CPU model for ex2.approx; bind syms['ex2'] to a device-produced function"
                )
            v = fn(memo[cur.args[0].uid])
        elif op == "neg":
            v = fp32.neg(memo[cur.args[0].uid])
        elif op == "fmin":
            v = fp32.fmin(memo[cur.args[0].uid], memo[cur.args[1].uid])
        elif op == "fmax":
            v = fp32.fmax(memo[cur.args[0].uid], memo[cur.args[1].uid])
        elif op == "exp2i":
            v = fp32.exp2i(memo[cur.args[0].uid])
        elif op == "rowmax":
            v = fp32.rowmax(*[memo[a.uid] for a in cur.args])
        elif op == "fabs":
            v = fp32.fabs(memo[cur.args[0].uid])
        elif op == "nzor":
            v = fp32.nzor(memo[cur.args[0].uid], memo[cur.args[1].uid])
        elif op.startswith("leaf:"):
            fn = (syms or {}).get(op)
            if fn is None:
                raise NotImplementedError(f"no binding for leaf {op!r}")
            v = float(fn(*cur.args))
        elif op.startswith("op:"):
            fn = (syms or {}).get(op)
            if fn is None:
                raise NotImplementedError(f"no CPU model for opaque primitive {op!r}")
            v = fn(*[memo[a.uid] if isinstance(a, Term) else a for a in cur.args])
        elif op == "cvt_out":
            kind = cur.args[1]
            if kind in ("e4m3", "e5m2"):
                v = fp32.to_fp8(memo[cur.args[0].uid], kind)
            elif kind == "e2m1":
                v = fp32.to_e2m1(memo[cur.args[0].uid])
            elif kind.startswith(
                "ue8m0_"
            ):  # the MX quantizer's scale conversion, "ue8m0_<rp|rz|rn>"
                v = fp32.to_ue8m0(memo[cur.args[0].uid], kind[6:])
            else:
                fmt = {"f32": fp32.F32, "bf16": fp32.BF16, "f16": fp32.F16}[kind]
                v = fp32.to_format(memo[cur.args[0].uid], fmt)
        else:  # pragma: no cover
            raise ValueError(op)
        memo[cur.uid] = v
    return memo[t.uid]


def fingerprint(t: Term) -> str:
    """Structural sha256 of the canonical S-expression of a term (op names, leaf
    coordinates, binary32 constants, association; commutative operands sorted), hence
    independent of interning ids and of the order in which terms were constructed in the
    process (F05). Two specs with equal fingerprints produce equal bits on any backend that
    satisfies the primitive assumptions."""
    import hashlib

    return hashlib.sha256(sexpr(t).encode()).hexdigest()[:16]


def size(t: Term) -> int:
    """Number of distinct nodes reachable from ``t``."""
    seen = set()
    stack = [t]
    while stack:
        c = stack.pop()
        if c.uid in seen:
            continue
        seen.add(c.uid)
        stack.extend(a for a in c.args if isinstance(a, Term))
    return len(seen)


def sexpr(t: Term) -> str:
    """Canonical S-expression of a term, with the operands of commutative nodes (add, mul,
    fma's first two) sorted by their own S-expression. Identical on the Lean side
    (``Term.sexpr``), so the two implementations can be compared textually."""
    memo: Dict[int, str] = {}
    stack = [t]
    while stack:
        cur = stack[-1]
        if cur.uid in memo:
            stack.pop()
            continue
        deps = [a for a in cur.args if isinstance(a, Term) and a.uid not in memo]
        if deps:
            stack.extend(deps)
            continue
        stack.pop()
        op = cur.op
        A = [memo[a.uid] if isinstance(a, Term) else a for a in cur.args]
        if op == "x":
            out = f"(x {A[1]})"
        elif op == "r":
            out = f"(r {A[1]})"
        elif op == "w":
            out = f"(w {A[0]})"
        elif op == "const":
            out = (
                f"(const {A[0]!r})"
                if isinstance(A[0], str)
                else f"(const {float(A[0])!r})"
            )
        elif op == "sym":
            out = f"(sym {A[0]})"
        elif op in ("add", "mul"):
            a, b = sorted(A[:2])
            out = f"({op} {a} {b})"
        elif op == "fma":
            a, b = sorted(A[:2])
            out = f"(fma {a} {b} {A[2]})"
        elif op == "div":
            out = f"(div {A[0]} {A[1]})"
        elif op == "sqrt":
            out = f"(sqrt {A[0]})"
        elif op == "fmaxc":
            a, b = sorted(A[:2])
            out = f"(fmax {a} {b})"
        elif op == "ex2":
            out = f"(ex2 {A[0]})"
        elif op in ("neg", "exp2i", "fabs"):
            out = f"({op} {A[0]})"
        elif op in ("fmin", "fmax", "nzor"):
            out = f"({op} {A[0]} {A[1]})"
        elif op == "rowmax":
            out = "(rowmax " + " ".join(A) + ")"
        elif op.startswith("leaf:"):
            out = "(" + op[5:] + "".join(f" {a}" for a in A) + ")"
        elif op == "cvt_out":
            out = f"(cvt {A[1]} {A[0]})"
        elif op == "op:mma_bf16_k16":
            out = f"(mma16 {A[0]} {A[1]})"
        elif op == "op:mma_nvfp4_k64":
            out = f"(mma64bs {A[0]} {A[1]})"
        elif op == "op:mma_nvfp4_k128":
            out = f"(mma128bs {A[0]} {A[1]})"
        elif op == "op:mma_mxfp8_k32":
            out = f"(mma32mx {A[0]} {A[1]})"  # the MX GEMM classes: UE8M0 scales per 32 elements
        elif op == "op:mma_mxfp4_k64":
            out = f"(mma64mx {A[0]} {A[1]})"
        elif op == "op:mma_mxfp8_k64":
            out = f"(mma64mx8 {A[0]} {A[1]})"  # the SM107 MX GEMM classes (K = 64 / 128 steps)
        elif op == "op:mma_mxfp4_k128":
            out = f"(mma128mx4 {A[0]} {A[1]})"

        elif op == "op:ab_kslice":
            out = f"(ab {A[0]} {A[1]})"
        elif op == "op:absf_kslice":
            out = f"(absf {A[0]} {A[1]} {A[2]} {A[3]})"
        elif op == "op:sfa_rows":
            out = f"(sfa {A[0]} {A[1]})"
        elif op == "op:sfb_cols":
            out = f"(sfb {A[0]} {A[1]})"
        elif op == "op:a_rows":
            out = f"(a {A[0]} {A[1]})"
        elif op == "op:b_cols":
            out = f"(b {A[0]} {A[1]})"
        else:
            out = "(" + op + " " + " ".join(map(str, A)) + ")"
        memo[cur.uid] = out
    return memo[t.uid]
