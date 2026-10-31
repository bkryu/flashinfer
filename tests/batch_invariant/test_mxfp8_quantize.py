"""``flashinfer.mxfp8_quantize(..., backend="auto", batch_invariant=True)``: the batch-invariant class ``mxfp8.quant``.

GPU tests (skipped off SM100 / SM103 / SM107 or without the CuTe DSL): agreement with FlashInfer's MXFP8 reference
implementation over a shape sweep in the three scale-factor layouts and bf16 / f16 / fp32, run-to-run determinism
including CUDA-graph replay, and the batch-invariance guarantee itself: a row's bits do not depend on the batch it is
quantized in (its size, the row's position, its neighbours, the order of the rows), the three layouts carry the same
bytes, and fp32 inputs equal the exact widening of 16-bit inputs. Every assertion is bitwise.
"""

import pytest
import torch

import flashinfer
from flashinfer import SfLayout
from flashinfer.batch_invariant import is_supported_device
from flashinfer.batch_invariant.quantization import layout as L
from tests.utils_fp8 import assert_mxfp8_quantize_exact

DT = [torch.bfloat16, torch.float16, torch.float32]
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


def _gpu_ok():
    return torch.cuda.is_available() and is_supported_device(
        torch.device("cuda", torch.cuda.current_device())
    )


needs_gpu = pytest.mark.skipif(
    not _gpu_ok(), reason="needs an SM100 / SM103 / SM107 device with the CuTe DSL"
)


def _varied_inputs(M, K, dtype, seed=0):
    """Inputs that exercise the quantizer broadly rather than typically: magnitudes spread over about 40 binades (12 for
    f16) with random signs, 1 % of the elements scaled up as outliers, 1 % exact +0 and 1 % exact -0. Every value is
    finite and every 32-element block keeps an amax above 448 * 2**-127."""
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


def _quantize(x, layout, **kw):
    """The batch-invariant path through the public API; codes as uint8 and the flat scale bytes, on the host."""
    q, sf = flashinfer.mxfp8_quantize(
        x,
        sf_swizzle_layout=FI_LAYOUTS[layout],
        backend="auto",
        batch_invariant=True,
        **kw,
    )
    assert q.dtype == torch.float8_e4m3fn and sf.dtype == torch.uint8 and sf.dim() == 1
    return q.view(torch.uint8).cpu(), sf.cpu()


def _same(a, b):
    return all(torch.equal(p, q) for p, q in zip(a, b, strict=True))


@needs_gpu
@pytest.mark.parametrize("layout", L.LAYOUTS)
@pytest.mark.parametrize("dtype", DT)
def test_matches_reference(dtype, layout):
    """Codes and scale factors agree with FlashInfer's MXFP8 reference implementation (``tests/utils_fp8.py``) over the
    shape sweep, and with ``alignment`` padding."""
    for M, K in SWEEP:
        x = _varied_inputs(M, K, dtype, seed=M + K)
        q, sf = flashinfer.mxfp8_quantize(
            x,
            sf_swizzle_layout=FI_LAYOUTS[layout],
            backend="auto",
            batch_invariant=True,
        )
        assert_mxfp8_quantize_exact(x, q, sf, sf_swizzle_layout=FI_LAYOUTS[layout])
    x = _varied_inputs(37, 96, dtype, seed=1)
    q, sf = flashinfer.mxfp8_quantize(
        x,
        alignment=128,
        sf_swizzle_layout=FI_LAYOUTS[layout],
        backend="auto",
        batch_invariant=True,
    )
    assert_mxfp8_quantize_exact(
        x, q, sf, alignment=128, sf_swizzle_layout=FI_LAYOUTS[layout]
    )


@needs_gpu
def test_determinism_and_graph_replay():
    x = _varied_inputs(37, 1024, torch.bfloat16)
    ref = _quantize(x, "128x4")
    for _ in range(5):
        assert _same(_quantize(x, "128x4"), ref)
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


def _rows(x, layout, K):
    """Quantize ``x`` and return per-row views: the codes ``(M, K)`` and the scale bytes ``(M, K / 32)`` in logical order,
    whatever the storage layout."""
    codes, sf = _quantize(x.contiguous(), layout)
    return codes, L.unswizzle_sf(sf, x.shape[0], K, 32, layout)


def _assert_rows_equal(got, full, idx):
    """Row ``j`` of ``got`` equals row ``idx[j]`` of the full-batch result, codes and scales."""
    assert torch.equal(got[0], full[0][idx]) and torch.equal(got[1], full[1][idx])


@needs_gpu
@pytest.mark.parametrize("layout", L.LAYOUTS)
@pytest.mark.parametrize("dtype", DT)
@pytest.mark.parametrize("K", [32, 1024, 7168])
def test_batch_permutation_subbatch_invariance(K, dtype, layout):
    """A row's bits do not depend on the batch it is quantized in. Against the result for a batch of 300 rows:

    * every row alone (M = 1), including the first, the last and rows at tile boundaries;
    * rows embedded at arbitrary positions among unrelated rows, and the batch embedded in a much larger batch;
    * the whole batch in several random orders and reversed;
    * sub-batches of every size class (1 to 299 rows): random selections, contiguous slices (prefix, middle, suffix),
      strided selections, and selections with repeated rows.

    The batch sizes span the dispatcher's regimes, so different sub-batches run under different configurations."""
    M = 300
    g = torch.Generator().manual_seed(K)
    xb = _varied_inputs(M, K, dtype, seed=1)
    full = _rows(xb, layout, K)

    for i in (0, 1, 7, 31, 127, 128, 129, 255, 256, 299):
        _assert_rows_equal(_rows(xb[i : i + 1], layout, K), full, torch.tensor([i]))

    other = _varied_inputs(23, K, dtype, seed=5)
    for pos, i in ((0, 17), (4, 130), (22, 299)):
        xo = other.clone()
        xo[pos] = xb[i]
        got = _rows(xo, layout, K)
        _assert_rows_equal(
            (got[0][pos : pos + 1], got[1][pos : pos + 1]), full, torch.tensor([i])
        )
    big = _varied_inputs(2048, K, dtype, seed=9)
    where = torch.randperm(2048, generator=g)[:M]
    big[where.cuda()] = xb
    got = _rows(big, layout, K)
    _assert_rows_equal((got[0][where], got[1][where]), full, torch.arange(M))

    for seed in range(4):
        perm = torch.randperm(M, generator=torch.Generator().manual_seed(seed))
        _assert_rows_equal(_rows(xb[perm.cuda()], layout, K), full, perm)
    rev = torch.arange(M - 1, -1, -1)
    _assert_rows_equal(_rows(xb[rev.cuda()], layout, K), full, rev)

    for n in (1, 2, 3, 5, 8, 16, 31, 64, 100, 127, 128, 129, 200, 257, 299):
        idx = torch.randperm(M, generator=g)[:n]
        _assert_rows_equal(_rows(xb[idx.cuda()], layout, K), full, idx)
    for lo, hi in ((0, 16), (0, 129), (37, 101), (128, 256), (171, 300), (299, 300)):
        idx = torch.arange(lo, hi)
        _assert_rows_equal(_rows(xb[lo:hi], layout, K), full, idx)
    for step, start in ((2, 0), (3, 1), (7, 5), (128, 3)):
        idx = torch.arange(start, M, step)
        _assert_rows_equal(_rows(xb[idx.cuda()], layout, K), full, idx)
    idx = torch.tensor([17, 17, 130, 17, 299, 130, 0, 0])
    _assert_rows_equal(_rows(xb[idx.cuda()], layout, K), full, idx)


@needs_gpu
def test_layouts_and_widenings_agree():
    """The three scale-factor layouts carry the same bytes, and fp32 inputs give the bits of the 16-bit inputs they
    exactly widen."""
    for M, K in ((300, 1024), (37, 96), (1000, 4096)):
        xb = _varied_inputs(M, K, torch.bfloat16, seed=M)
        ref = _rows(xb, "128x4", K)
        for layout in ("8x4", "linear"):
            got = _rows(xb, layout, K)
            assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1]), layout
    for dtype in (torch.bfloat16, torch.float16):
        x = _varied_inputs(130, 2048, dtype, seed=3)
        for layout in L.LAYOUTS:
            assert _same(_quantize(x.float(), layout), _quantize(x, layout)), (
                dtype,
                layout,
            )
