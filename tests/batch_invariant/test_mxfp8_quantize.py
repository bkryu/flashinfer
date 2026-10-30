"""``flashinfer.mxfp8_quantize(..., backend="auto", batch_invariant=True)``: the batch-invariant class ``mxfp8.quant``.

Device-free: the backend rule, the ``fingerprint()`` receipt, the closed-form dispatcher's membership in the enumerated
space over (M, K, dtype, SM count in {148, 152, 204, 224}) and its M-independent compile key.

GPU (skipped off SM100 / SM103 / SM107 or without the CuTe DSL): bitwise equality with FlashInfer's own ``cuda`` and
``cute-dsl`` MXFP8 backends over a shape sweep in the three scale-factor layouts and bf16 / f16 / fp32 (outside the one
documented corner, which is pinned), FlashInfer's output contract (shapes, dtype, ``alignment`` padding), run-to-run
determinism incl. CUDA-graph replay and PDL, batch / permutation / sub-batch invariance across the three layouts,
agreement of every SKU's dispatcher pick, and the errors of the mode (backend rule, unsupported dtype / alignment /
device / architecture). Every assertion is bitwise; no tolerance is used.
"""

import inspect

import pytest
import torch

import flashinfer
from flashinfer import SfLayout
from flashinfer.batch_invariant import (
    SUPPORTED_ARCHS,
    BatchInvariantUnsupportedError,
    class_name,
    fingerprint,
    is_supported_device,
    resolve_backend,
)
from flashinfer.batch_invariant.quant import layout as L
from flashinfer.batch_invariant.quant import mx_schedule as C
from flashinfer.batch_invariant.quant.mx_api import mxfp8_quantize_batch_invariant
from flashinfer.batch_invariant.quant.mx_kernel import (
    MX_QUANT_MODULE,
    _compile_bucket,
    mx_quant_kernel_name,
)
from flashinfer.batch_invariant.quant.mx_spec import FINGERPRINT_MXFP8

DT = [torch.bfloat16, torch.float16, torch.float32]
EB = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}
SM_COUNTS = [148, 152, 204, 224]
GRID_M = [1, 2, 7, 16, 37, 256, 1000, 4096, 16384]
GRID_K = [32, 96, 1024, 2048, 4096, 7168, 8192, 16384]
SWEEP = [
    (1, 32),
    (3, 96),
    (37, 256),
    (130, 1024),
    (8, 2048),
    (200, 4096),
    (37, 7168),
    (5, 8192),
]
FI_LAYOUTS = {
    "128x4": SfLayout.layout_128x4,
    "8x4": SfLayout.layout_8x4,
    "linear": SfLayout.layout_linear,
}


# ---------------------------------------------------------------------------------------------------- device-free
def test_backend_rule_and_fingerprint():
    """batch_invariant=True only through backend="auto"; "auto" without the flag is the API's default; the receipt."""
    assert (
        resolve_backend("mxfp8_quantize", "auto", True, default="cuda")
        == "batch-invariant"
    )
    assert resolve_backend("mxfp8_quantize", "auto", False, default="cuda") == "cuda"
    assert (
        resolve_backend("mxfp8_quantize", "cute-dsl", False, default="cuda")
        == "cute-dsl"
    )
    for b in ("cuda", "cute-dsl", "batch-invariant"):
        with pytest.raises(ValueError, match="reachable only through backend='auto'"):
            resolve_backend("mxfp8_quantize", b, True, default="cuda")
    assert (
        fingerprint("mxfp8_quantize")
        == FINGERPRINT_MXFP8
        == "mxquant-mxfp8-9018bd1627b3e72a"
    )
    assert class_name("mxfp8_quantize") == "mxfp8.quant"
    with pytest.raises(ValueError, match="no batch-invariant class"):
        fingerprint("rmsnorm")
    with pytest.raises(TypeError):
        fingerprint("mxfp8_quantize", K=4096)
    assert set(SUPPORTED_ARCHS) == {(10, 0), (10, 3), (10, 7)}
    sig = inspect.signature(flashinfer.mxfp8_quantize)
    assert sig.parameters["batch_invariant"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["batch_invariant"].default is False
    assert sig.parameters["backend"].default == "cuda"


@pytest.mark.parametrize("K", GRID_K)
def test_dispatcher_returns_members_for_every_sku(K):
    """Every pick of the closed-form dispatcher over (M, dtype, layout, SM count) is a member of the enumerated space, so
    the pick never changes bits; the compiled specialization's name never carries M."""
    assert "M" not in inspect.signature(_compile_bucket).parameters
    for eb, dt in ((2, "bf16"), (4, "f32")):
        for sw in L.LAYOUTS:
            space = C.all_valid_configs(K, eb, sw)
            assert len(space) >= 8 and len(set(space)) == len(space)
            for M in GRID_M:
                for sm in SM_COUNTS:
                    cfg = C.default_config(M, K, eb, sm, sw)
                    assert cfg in space, (K, eb, sw, M, sm, cfg)
                    name = mx_quant_kernel_name(
                        "mxfp8", "rcprp", dt, cfg.bucket(K), cfg.describe(), sw, False
                    )
                    assert (
                        "_M" not in name
                        and str(M) not in name.split("_Kbp")[1].split("_")[1:2]
                    )
    assert MX_QUANT_MODULE == "batch_invariant_mx_quant"


# ---------------------------------------------------------------------------------------------------- GPU
def _gpu_ok():
    return torch.cuda.is_available() and is_supported_device(
        torch.device("cuda", torch.cuda.current_device())
    )


needs_gpu = pytest.mark.skipif(
    not _gpu_ok(), reason="needs an SM100 / SM103 / SM107 device with the CuTe DSL"
)


def _inputs(M, K, dtype, seed=0):
    """Rows mixing wide exponents, exact zeros of both signs, outliers and cancellation pairs; every block's amax is kept
    above 448 * 2**-127 (FlashInfer's backends zero the codes of smaller blocks, the class keeps them: pinned below)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    e = (torch.rand(M, K, generator=g) * 2 - 1) * (
        20.0 if dtype != torch.float16 else 6.0
    )
    x = (
        torch.where(torch.rand(M, K, generator=g) < 0.5, -1.0, 1.0)
        * (torch.rand(M, K, generator=g) + 1.0)
        * torch.exp2(e)
    )
    r = torch.rand(M, K, generator=g)
    x = torch.where(r < 0.01, x * (2.0**12 if dtype != torch.float16 else 2.0**4), x)
    x = torch.where((r >= 0.01) & (r < 0.02), torch.zeros_like(x), x)
    x = torch.where((r >= 0.02) & (r < 0.03), -torch.zeros_like(x), x)
    x = x.to(dtype)
    assert torch.isfinite(x).all()
    return x.cuda()


def _api(x, layout, **kw):
    q, sf = flashinfer.mxfp8_quantize(
        x,
        sf_swizzle_layout=FI_LAYOUTS[layout],
        backend="auto",
        batch_invariant=True,
        **kw,
    )
    assert q.dtype == torch.float8_e4m3fn and sf.dtype == torch.uint8 and sf.dim() == 1
    return q.view(torch.uint8).cpu(), sf.cpu()


def _fi(x, layout, backend):
    q, sf = flashinfer.mxfp8_quantize(
        x, sf_swizzle_layout=FI_LAYOUTS[layout], backend=backend, enable_pdl=False
    )
    return q.view(torch.uint8).cpu(), sf.view(torch.uint8).cpu().reshape(-1)


def _same(a, b):
    return all(torch.equal(p, q) for p, q in zip(a, b, strict=True))


@needs_gpu
@pytest.mark.parametrize("layout", L.LAYOUTS)
@pytest.mark.parametrize("dtype", DT)
def test_bitwise_equal_to_flashinfer_backends(dtype, layout):
    """Codes and scale bytes equal FlashInfer's ``cute-dsl`` backend (and its ``cuda`` backend for the 16-bit inputs it
    accepts) over the shape sweep: the same shapes, the same bytes."""
    for M, K in SWEEP:
        x = _inputs(M, K, dtype, seed=M + K)
        ours = _api(x, layout)
        for backend in ("cute-dsl",) + (("cuda",) if dtype != torch.float32 else ()):
            theirs = _fi(x, layout, backend)
            assert (
                ours[0].shape == theirs[0].shape and ours[1].shape == theirs[1].shape
            ), (backend, M, K)
            assert torch.equal(ours[0], theirs[0]), (
                f"codes differ from {backend} (M={M}, K={K}, {dtype}, {layout})"
            )
            assert torch.equal(ours[1], theirs[1]), (
                f"scales differ from {backend} (M={M}, K={K}, {dtype}, {layout})"
            )


@needs_gpu
def test_documented_corner_is_pinned():
    """The one difference from FlashInfer's backends: a block whose amax is below 448 * 2**-127 gets UE8M0 byte 0 in both,
    FlashInfer's inverse scale is then 0 (every code 0) while the class applies the exact 2**127 (the values survive); and
    a -inf element saturates to -448 in the class where FlashInfer gives +448. Nothing else differs."""
    M, K = 64, 1024
    x = _inputs(M, K, torch.bfloat16, seed=11)
    xb = x.float().view(M, K // 32, 32)
    tiny = torch.zeros(M, K // 32, dtype=torch.bool, device="cuda")
    tiny[::7, ::5] = True
    xb = torch.where(
        tiny.unsqueeze(-1), (xb.sign() + (xb == 0).float()) * 2.0**-126, xb
    )
    x = xb.view(M, K).to(torch.bfloat16)
    ours, theirs = _api(x, "linear"), _fi(x, "linear", "cuda")
    assert torch.equal(ours[1], theirs[1]) and bool(
        (ours[1].view(M, K // 32)[tiny.cpu()] == 0).all()
    )
    oc, tc = ours[0].view(M, K // 32, 32), theirs[0].view(M, K // 32, 32)
    assert bool((tc[tiny.cpu()] == 0).all()) and bool((oc[tiny.cpu()] != 0).any())
    assert bool(((oc != tc).any(-1) <= tiny.cpu()).all()), (
        "a code differs outside a tiny block"
    )
    y = torch.zeros(8, 64, dtype=torch.bfloat16, device="cuda")
    y[0, 0], y[0, 1] = -float("inf"), 3.0
    oy, ty = _api(y, "linear"), _fi(y, "linear", "cuda")
    assert torch.equal(oy[1], ty[1]) and int(oy[1][0]) == 254
    assert (
        oy[0][0, 0].view(torch.float8_e4m3fn).float() == -448.0
        and ty[0][0, 0].view(torch.float8_e4m3fn).float() == 448.0
    )


@needs_gpu
def test_output_contract():
    """FlashInfer's contract: float8_e4m3fn codes of the input's shape, flat uint8 scales of the layout's size with zero
    padding, ``alignment`` padding K with zero columns (the same bytes as the cute-dsl backend), >2-D and strided inputs,
    under-aligned storage offsets, ``is_sf_swizzled_layout`` and ``enable_pdl`` None / True / False."""
    x = _inputs(37, 96, torch.bfloat16)
    q, sf = flashinfer.mxfp8_quantize(x, backend="auto", batch_invariant=True)
    assert (
        q.dtype == torch.float8_e4m3fn
        and q.shape == (37, 96)
        and sf.shape == (128 * 4,)
    )
    logical = L.unswizzle_sf(sf.cpu(), 128, 96, 32, "128x4")
    assert int(logical[37:].sum()) == 0 and all(
        int(sf[L.sf_offset_128x4(r, 3, 4)]) == 0 for r in range(128)
    )
    q8, sf8 = flashinfer.mxfp8_quantize(
        x, sf_swizzle_layout=SfLayout.layout_8x4, backend="auto", batch_invariant=True
    )
    assert sf8.shape == (40 * 4,) and torch.equal(
        q8.view(torch.uint8), q.view(torch.uint8)
    )
    ql, sfl = flashinfer.mxfp8_quantize(
        x, is_sf_swizzled_layout=False, backend="auto", batch_invariant=True
    )
    assert sfl.shape == (37 * 3,) and torch.equal(
        ql.view(torch.uint8), q.view(torch.uint8)
    )
    for layout in L.LAYOUTS:
        qa, sfa = flashinfer.mxfp8_quantize(
            x,
            alignment=128,
            sf_swizzle_layout=FI_LAYOUTS[layout],
            backend="auto",
            batch_invariant=True,
        )
        qf, sff = flashinfer.mxfp8_quantize(
            x,
            alignment=128,
            sf_swizzle_layout=FI_LAYOUTS[layout],
            backend="cute-dsl",
            enable_pdl=False,
        )
        assert qa.shape == qf.shape == (37, 128) and sfa.shape == sff.shape
        assert torch.equal(qa.view(torch.uint8), qf.view(torch.uint8)) and torch.equal(
            sfa, sff
        ), layout
        assert int(qa.view(torch.uint8)[:, 96:].sum()) == 0
    x3 = _inputs(36, 96, torch.bfloat16, seed=3).view(3, 12, 96)
    q3, s3 = flashinfer.mxfp8_quantize(x3, backend="auto", batch_invariant=True)
    q2, s2 = flashinfer.mxfp8_quantize(
        x3.view(36, 96), backend="auto", batch_invariant=True
    )
    assert (
        q3.shape == (3, 12, 96)
        and torch.equal(q3.view(36, 96).view(torch.uint8), q2.view(torch.uint8))
        and torch.equal(s3, s2)
    )
    wide = _inputs(37, 192, torch.bfloat16, seed=5)
    view = wide[:, :96]
    assert not view.is_contiguous() and _same(
        _api(view, "128x4"), _api(view.contiguous(), "128x4")
    )
    pool = torch.empty(4 + 37 * 96, dtype=torch.bfloat16, device="cuda")
    pool[4:].copy_(x.reshape(-1))
    off = pool[4:].view(37, 96)
    assert off.data_ptr() % 16 != 0 and _same(_api(off, "128x4"), _api(x, "128x4"))
    ref = _api(x, "128x4")
    assert _same(_api(x, "128x4", enable_pdl=True), ref) and _same(
        _api(x, "128x4", enable_pdl=False), ref
    )


@needs_gpu
def test_determinism_and_graph_replay():
    x = _inputs(37, 1024, torch.bfloat16)
    ref = _api(x, "128x4")
    for _ in range(5):
        assert _same(_api(x, "128x4"), ref)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        outs = flashinfer.mxfp8_quantize(x, backend="auto", batch_invariant=True)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            outs = flashinfer.mxfp8_quantize(x, backend="auto", batch_invariant=True)
    for _ in range(2):
        outs[0].view(torch.uint8).zero_()
        outs[1].zero_()
        g.replay()
        torch.cuda.synchronize()
        assert _same((outs[0].view(torch.uint8).cpu(), outs[1].cpu()), ref)


@needs_gpu
def test_batch_permutation_subbatch_invariance():
    """A row's bits in a batch of 256, alone, embedded among other rows, permuted and in sub-batches of every size class;
    the three layouts carry the same bytes; fp32 inputs equal the exact widening of the 16-bit inputs."""
    K = 1024
    xb = _inputs(256, K, torch.bfloat16, seed=1)
    full_c, full_s = _api(xb, "128x4")
    sf_full = L.unswizzle_sf(full_s, 256, K, 32, "128x4")

    def logical(outs, M):
        return outs[0], L.unswizzle_sf(outs[1], M, K, 32, "128x4")

    for i in (0, 17, 130, 255):
        one = logical(_api(xb[i : i + 1].clone(), "128x4"), 1)
        assert torch.equal(one[0][0], full_c[i]) and torch.equal(one[1][0], sf_full[i])
    xo = _inputs(13, K, torch.bfloat16, seed=5)
    xo[4] = xb[17]
    o = logical(_api(xo, "128x4"), 13)
    assert torch.equal(o[0][4], full_c[17]) and torch.equal(o[1][4], sf_full[17])
    perm = torch.randperm(256, generator=torch.Generator().manual_seed(2)).cuda()
    p = logical(_api(xb[perm].contiguous(), "128x4"), 256)
    assert torch.equal(p[0], full_c[perm.cpu()]) and torch.equal(
        p[1], sf_full[perm.cpu()]
    )
    for k in (1, 3, 16, 100, 129):
        idx = torch.randperm(256, generator=torch.Generator().manual_seed(k))[:k].cuda()
        sub = logical(_api(xb[idx].contiguous(), "128x4"), k)
        assert torch.equal(sub[0], full_c[idx.cpu()]) and torch.equal(
            sub[1], sf_full[idx.cpu()]
        )
    lin_c, lin_s = _api(xb, "linear")
    assert torch.equal(lin_c, full_c) and torch.equal(lin_s.view(256, K // 32), sf_full)
    c8, s8 = _api(xb, "8x4")
    assert torch.equal(c8, full_c) and torch.equal(
        L.unswizzle_sf(s8, 256, K, 32, "8x4"), sf_full
    )
    for dtype in (torch.bfloat16, torch.float16):
        x = _inputs(130, 2048, dtype, seed=3)
        for layout in L.LAYOUTS:
            assert _same(_api(x.float(), layout), _api(x, layout)), (dtype, layout)


@needs_gpu
def test_dispatcher_picks_agree():
    """The configurations the dispatcher picks for the four SKUs (and this device) give the same bits at every sweep shape."""
    for layout in L.LAYOUTS:
        for dtype in (torch.bfloat16, torch.float32):
            for M, K in [(2048, 1024), (300, 4096), (37, 7168), (16384, 2048)]:
                x = _inputs(M, K, dtype, seed=K)
                ref = _api(x, layout)
                picks = {
                    C.default_config(M, K, EB[dtype], s, layout) for s in SM_COUNTS
                }
                assert len(picks) >= 1
                for s in SM_COUNTS:
                    q, sf = mxfp8_quantize_batch_invariant(
                        x, FI_LAYOUTS[layout], sm_count=s
                    )
                    assert _same((q.view(torch.uint8).cpu(), sf.cpu()), ref), (
                        s,
                        M,
                        K,
                        dtype,
                        layout,
                    )


@needs_gpu
def test_errors_never_fall_back(monkeypatch):
    """The backend rule through the API, and every input outside the class raising BatchInvariantUnsupportedError (a
    ValueError) that names the class and the limit: dtype, alignment, a CPU tensor, an unsupported architecture, a
    missing CuTe DSL."""
    x = _inputs(37, 1024, torch.bfloat16)
    for backend in ("cuda", "cute-dsl"):
        with pytest.raises(ValueError, match="reachable only through backend='auto'"):
            flashinfer.mxfp8_quantize(x, backend=backend, batch_invariant=True)
    auto, cuda = (
        flashinfer.mxfp8_quantize(x, backend="auto"),
        flashinfer.mxfp8_quantize(x, backend="cuda"),
    )
    assert torch.equal(
        auto[0].view(torch.uint8), cuda[0].view(torch.uint8)
    ) and torch.equal(auto[1], cuda[1])
    with pytest.raises(AssertionError, match="backend must be"):
        flashinfer.mxfp8_quantize(x, backend="rlk")
    assert issubclass(BatchInvariantUnsupportedError, ValueError)
    with pytest.raises(BatchInvariantUnsupportedError, match="mxfp8.quant.*dtype"):
        flashinfer.mxfp8_quantize(
            x.to(torch.float8_e4m3fn), backend="auto", batch_invariant=True
        )
    with pytest.raises(BatchInvariantUnsupportedError, match="alignment"):
        flashinfer.mxfp8_quantize(x, alignment=48, backend="auto", batch_invariant=True)
    with pytest.raises(BatchInvariantUnsupportedError, match="CUDA device"):
        flashinfer.mxfp8_quantize(x.cpu(), backend="auto", batch_invariant=True)
    import flashinfer.utils as fu

    monkeypatch.setattr(fu, "get_compute_capability", lambda device: (9, 0))
    with pytest.raises(
        BatchInvariantUnsupportedError,
        match="compute capability 9.0.*SM100 / SM103 / SM107",
    ):
        flashinfer.mxfp8_quantize(x, backend="auto", batch_invariant=True)
    monkeypatch.undo()
    import flashinfer.cute_dsl.availability as av

    monkeypatch.setattr(av, "is_cute_dsl_available", lambda: False)
    with pytest.raises(BatchInvariantUnsupportedError, match="CuTe DSL"):
        flashinfer.mxfp8_quantize(x, backend="auto", batch_invariant=True)
