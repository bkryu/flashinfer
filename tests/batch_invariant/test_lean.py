"""The Lean tier (``FLASHINFER_BI_LEAN=1`` or ``-m lean``): the Lean model of the engine and of the class ``mxfp8.quant``
(``flashinfer/batch_invariant/lean/``) is an independent implementation; ``lake build`` elaborates the generated cases
(the Lean model reproduces the Python engine's / spec's S-expressions node for node, by ``native_decide``), ``lake exe
conformance`` prints the Lean terms for a textual comparison with the Python side, and ``lake exe axioms`` enforces the
axiom policy (no ``sorry``; standard axioms only, ``Lean.ofReduceBool`` for the decided tables alone).

The device-free byte-compare of the generated cases needs no toolchain and lives in ``test_mxfp8_quantize.py``.
"""

import os
import shutil
import subprocess

import pytest

from flashinfer.batch_invariant import lean_cases, term

LEAN = lean_cases.LEAN_DIR
LAKE = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
pytestmark = pytest.mark.lean


def _run(*args, timeout):
    env = dict(
        os.environ,
        ELAN_NO_OVERRIDE_NOTICE="1",
        PATH=os.path.dirname(LAKE) + os.pathsep + os.environ["PATH"],
    )
    return subprocess.run(
        [LAKE, *args],
        cwd=LEAN,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.skipif(
    not os.path.exists(LAKE), reason="Lean toolchain (elan / lake) not installed"
)
def test_lean_model_matches_python_engine_and_spec():
    with open(lean_cases.CASES_PATH) as f:
        committed = f.read()
    assert committed == lean_cases.render(), (
        "regenerate: python -m flashinfer.batch_invariant.lean_cases"
    )
    r = _run("build", timeout=3600)
    assert r.returncode == 0, (
        "Lean proofs failed on the generated cases:\n" + (r.stdout + r.stderr)[-4000:]
    )
    r = _run("build", "conformance", "axioms", timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    r = _run("exe", "conformance", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    lean = {}
    for line in r.stdout.splitlines():
        tag = line.split(" ", 1)[0]
        if tag == "mxq":
            _, name, kind, sx = line.split(" ", 3)
            lean[("mxq", name, kind)] = sx
        elif tag == "mxqout":
            _, fname, rname, which, sx = line.split(" ", 4)
            lean[("mxqout", fname, rname, which)] = sx
        elif tag == "mxqblock":
            _, b, name, sx = line.split(" ", 3)
            lean[("mxqblock", int(b), name)] = sx
        else:
            raise AssertionError(f"unexpected conformance line: {line[:80]}")
    checked = 0
    for c in lean_cases.mxquant_cases():
        assert lean[("mxq", c["name"], "spec")] == c["spec"], (
            "mx block amax spec differs"
        )
        assert lean[("mxq", c["name"], "kernel")] == c["kernel"], (
            f"mx engine differs: {c['name']}"
        )
        checked += 2
    for c in lean_cases.mxquant_out_cases():
        for which in lean_cases.MXQ_OUT_WHICH:
            assert lean[("mxqout", c["fmt"], c["rule"], which)] == c[which], (
                f"mx chain differs: {c['fmt']} {c['rule']} {which}"
            )
            checked += 1
    for b, name, sx in lean_cases.mxquant_block_cases():
        assert lean[("mxqblock", b, name)] == sx, (
            f"mx block amax differs: block {b} {name}"
        )
        checked += 1
    assert checked >= 60
    assert isinstance(
        term.sexpr, type(lean_cases.render)
    )  # the two printers are the coupled pair


@pytest.mark.skipif(
    not os.path.exists(LAKE), reason="Lean toolchain (elan / lake) not installed"
)
def test_axioms_policy():
    r = _run("exe", "axioms", timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    assert "0 with sorry, 0 violations" in r.stdout, r.stdout
