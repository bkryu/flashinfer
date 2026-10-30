"""Test tiers of the batch-invariant mode (``flashinfer/batch_invariant/README.md``, "Running the tiers").

* **quick** (every test without a tier marker): runs on every CI job. Device-free tests (exact CPU model, fingerprint
  pins, dispatcher membership, compile-key rule, Lean generated-cases conformance) run everywhere; GPU tests skip
  cleanly off SM100 / SM103 / SM107 or without the CuTe DSL.
* **nightly** (``@pytest.mark.nightly``): the full enumerated configuration spaces and the exhaustive UE8M0 device sweep.
  Selected by ``FLASHINFER_BI_NIGHTLY=1`` or by ``-m nightly``; skipped otherwise so the quick tier stays quick.
* **lean** (``@pytest.mark.lean``): ``lake build`` + ``lake exe conformance`` + ``lake exe axioms`` through the pinned
  Lean toolchain; skipped when ``lake`` is not installed (the device-free byte-compare of the generated cases needs no
  toolchain and is part of the quick tier).
"""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "nightly: the nightly tier of the batch-invariant tests (full spaces, exhaustive sweeps)",
    )
    config.addinivalue_line(
        "markers",
        "lean: runs the Lean toolchain (lake build, conformance, axioms check)",
    )


def _selected(config, marker: str) -> bool:
    expr = config.getoption("-m") or ""
    return marker in expr


def pytest_collection_modifyitems(config, items):
    nightly = os.environ.get("FLASHINFER_BI_NIGHTLY") == "1" or _selected(
        config, "nightly"
    )
    lean = os.environ.get("FLASHINFER_BI_LEAN") == "1" or _selected(config, "lean")
    for item in items:
        if "nightly" in item.keywords and not nightly:
            item.add_marker(
                pytest.mark.skip(
                    reason="nightly tier: set FLASHINFER_BI_NIGHTLY=1 or select -m nightly"
                )
            )
        if "lean" in item.keywords and not lean:
            item.add_marker(
                pytest.mark.skip(
                    reason="Lean tier: set FLASHINFER_BI_LEAN=1 or select -m lean"
                )
            )
