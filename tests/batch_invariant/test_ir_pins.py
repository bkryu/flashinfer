"""IR pins of the ``mxfp8.quant`` kernel (device-free; needs the CuTe DSL and its bundled ptxas, not a GPU).

One configuration per code path is compiled with no device visible and its MLIR is pinned: every f32 arithmetic op is an
NVVM intrinsic with explicit rounding (``mul.rn``, ``div.rn``; ``fma.rn`` only under the ``zero = "positive"`` recipe
variant), the exact primitives are ``nvvm.fmax`` (the amax tree), ``nvvm.fabs`` (the leaf) and, for the clamp variant,
``nvvm.fmin`` / ``nvvm.fmax``; the only conversions are the declared ``cvt.<rp|rz>.satfinite.ue8m0x2.f32`` (or the exact
integer ``rn`` sequence) + ``cvt.rn.bf16x2.ue8m0x2`` chain and ``cvt.rn.satfinite.e4m3x2.f32``; no ``math`` dialect, no
``arith`` float arithmetic, no fastmath, no ``rcp.approx``, no ``ftz``, no ``arith.truncf`` (no 16-bit intermediate: the
outputs are bytes); an exact ``extf`` widening for 16-bit inputs and none for fp32. The counts are checked per block and
element, so an extra or a missing rounding is caught.

The compile runs in a subprocess so ``CUTE_DSL_ARCH`` / ``CUDA_VISIBLE_DEVICES`` are set before cutlass is imported;
``FLASHINFER_BI_TEST_ARCH`` selects the target (default ``sm_100a``; ``sm_107a`` for Rubin).
"""

import glob
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _dsl_available():
    import importlib.util

    return (
        importlib.util.find_spec("cutlass") is not None
        and importlib.util.find_spec("cutlass.cute") is not None
    )


needs_dsl = pytest.mark.skipif(
    not _dsl_available(), reason="needs the CuTe DSL (nvidia-cutlass-dsl)"
)

# (dtype, K, config args, format, rule kwargs, layout): one case per code path of the mxfp8 kernel
QUICK = [
    (
        "bf16",
        4096,
        (128, 1, 32),
        "mxfp8",
        {},
        "128x4",
    ),  # one lane per block, lane-major, the 128x4 row store
    (
        "f32",
        4096,
        (128, 1, 32),
        "mxfp8",
        {},
        "128x4",
    ),  # fp32 input: 4 elements per vector, no extf
    (
        "bf16",
        1024,
        (32, 4, 16),
        "mxfp8",
        {},
        "linear",
    ),  # two lanes per block (one shuffle), the linear store
    (
        "f16",
        1024,
        (64, 2, 8, "reg"),
        "mxfp8",
        {},
        "8x4",
    ),  # four lanes per block (two shuffles), reg-major, the 8x4 store
    (
        "bf16",
        32,
        (1, 128, 32),
        "mxfp8",
        {},
        "128x4",
    ),  # a single block per row (Kbp < 4: the padding columns)
    (
        "bf16",
        4096,
        (32, 128, 32, "lane", "tile", 16, 2),
        "mxfp8",
        {},
        "128x4",
    ),  # the tile placement (smem-staged scales)
]
NIGHTLY = [
    (
        "bf16",
        2048,
        (64, 2, 32),
        "mxfp8",
        {"norm": "pow2", "round": "rz"},
        "128x4",
    ),  # OCP floor: cvt.rz
    (
        "bf16",
        2048,
        (64, 2, 32),
        "mxfp8",
        {"round": "rn", "sat": "clamp_satfinite", "zero": "positive"},
        "128x4",
    ),  # software rn, clamp, fma
    (
        "bf16",
        2048,
        (64, 2, 32),
        "mxfp8",
        {"norm": "div"},
        "linear",
    ),  # div.rn normalisation
    (
        "bf16",
        4096,
        (32, 128, 32, "lane", "tile", 16, 2),
        "mxfp4",
        {},
        "128x4",
    ),  # the e2m1 conversion (PR 2's class, same body)
]


def _compile_and_dump(tmp_path, dtype, K, cfg, fmt, rule, layout):
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        CUTE_DSL_KEEP_IR="1",
        FLASHINFER_CUTE_DSL_DISABLE_CACHE="1",
        CUTE_DSL_ARCH=os.environ.get("FLASHINFER_BI_TEST_ARCH", "sm_100a"),
        PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    code = (
        "from flashinfer.batch_invariant.quant.mx_schedule import MXQuantConfig\n"
        "from flashinfer.batch_invariant.quant.mx_kernel import get_compiled_kernel\n"
        "from flashinfer.batch_invariant.quant.mx_spec import FORMATS, ScaleRule\n"
        f"get_compiled_kernel({dtype!r}, {K}, MXQuantConfig{cfg}, FORMATS[{fmt!r}], ScaleRule(**{rule!r}), {layout!r}, False)\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    files = glob.glob(str(tmp_path / "*.mlir"))
    assert files, "no MLIR dumped (CUTE_DSL_KEEP_IR unsupported?)"
    return open(max(files, key=os.path.getmtime)).read()


def _check(ir, dtype, K, cfg, fmt, rule):
    from flashinfer.batch_invariant.quant.mx_schedule import MXQuantConfig
    from flashinfer.batch_invariant.quant.mx_spec import FORMATS, ScaleRule

    # forbidden: anything the backend may contract, reassociate, approximate or flush
    assert not re.search(
        r"arith\.(addf|mulf|divf|subf|fmaf|maxf|minf|maximumf|minimumf|maxnumf|minnumf)\b",
        ir,
    ), "arith float arithmetic present"
    assert not re.search(r"\bmath\.", ir), "math-dialect op present"
    assert (
        "fastmath<" not in ir
        and "rcp.approx" not in ir
        and "div.approx" not in ir
        and "ex2" not in ir
    )
    assert "ftz" not in ir.replace("satfinite", "").replace("ftz = false", "").replace(
        "ftz=false", ""
    )
    assert not re.search(r"arith\.truncf", ir), (
        "a 16-bit intermediate (truncf) is present"
    )
    c, f, rl = MXQuantConfig(*cfg), FORMATS[fmt], ScaleRule(**rule)
    bpt = c.blocks_per_thread(K) * (c.tile_passes if c.mode == "tile" else 1)
    V = c.elems_per_thread
    lane_steps = c.lanes_per_block.bit_length() - 1
    assert ir.count("nvvm.fabs") == V * bpt
    n_fmax_tree = (V - 1) * bpt + lane_steps * bpt
    n_clamp = V * bpt if rl.sat == "clamp_satfinite" else 0
    assert ir.count("nvvm.fmax") == n_fmax_tree + n_clamp, (
        ir.count("nvvm.fmax"),
        n_fmax_tree,
        n_clamp,
    )
    assert ir.count("nvvm.fmin") == n_clamp
    assert ir.count("nvvm.shfl.sync") == lane_steps * bpt
    n_mul = bpt * (1 if rl.norm != "div" else 0) + (bpt * V if rl.zero == "keep" else 0)
    n_div = bpt * (1 + (1 if rl.norm == "div" else 0))
    n_fma = bpt * V if rl.zero == "positive" else 0
    assert ir.count("nvvm.mul.rn") == n_mul, (ir.count("nvvm.mul.rn"), n_mul)
    assert ir.count("nvvm.div.rn") == n_div, (ir.count("nvvm.div.rn"), n_div)
    fmas = re.findall(r"nvvm\.fma\b[^\n]*", ir)
    assert len(fmas) == n_fma, (len(fmas), n_fma)
    assert all("rn" in line for line in fmas), fmas[:2]
    if rl.round in ("rp", "rz"):
        assert ir.count(f"cvt.{rl.round}.satfinite.ue8m0x2.f32") == bpt
    else:
        assert "satfinite.ue8m0x2" not in ir and ir.count("cvt.u16.u32 pair") == bpt
    assert ir.count("cvt.rn.bf16x2.ue8m0x2") == bpt
    assert ir.count(f"cvt.rn.satfinite.{f.elem_fmt}x2.f32") == bpt * V // 2
    assert (
        ir.count(
            "cvt.rn.satfinite.e2m1x2.f32"
            if f.elem_fmt == "e4m3"
            else "cvt.rn.satfinite.e4m3x2.f32"
        )
        == 0
    )
    if c.mode == "row":
        assert ir.count("nvvm.barrier") == 0
    else:
        assert ir.count("st.global.v4.u32") >= 1 and ir.count("nvvm.barrier") == 1
    if dtype == "f32":
        assert not re.search(r"arith\.extf", ir)
    else:
        assert re.search(r"arith\.extf .*: (bf16|f16) to f32", ir)


@needs_dsl
@pytest.mark.parametrize("dtype,K,cfg,fmt,rule,layout", QUICK, ids=lambda v: str(v))
def test_ir_uses_only_rn_intrinsics_and_declared_conversions(
    tmp_path, dtype, K, cfg, fmt, rule, layout
):
    _check(
        _compile_and_dump(tmp_path, dtype, K, cfg, fmt, rule, layout),
        dtype,
        K,
        cfg,
        fmt,
        rule,
    )


@needs_dsl
@pytest.mark.nightly
@pytest.mark.parametrize("dtype,K,cfg,fmt,rule,layout", NIGHTLY, ids=lambda v: str(v))
def test_nightly_ir_recipe_variants(tmp_path, dtype, K, cfg, fmt, rule, layout):
    _check(
        _compile_and_dump(tmp_path, dtype, K, cfg, fmt, rule, layout),
        dtype,
        K,
        cfg,
        fmt,
        rule,
    )
