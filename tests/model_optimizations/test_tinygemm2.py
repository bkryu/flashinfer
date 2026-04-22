import torch
import pytest
import torch.nn.functional as F
from flashinfer.utils import get_compute_capability
from tests.utils_fp8 import to_float8


def _skip_if_not_sm90():
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] < 9:
        pytest.skip("tinygemm2 requires SM90+")


def _skip_if_not_blackwell():
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] < 10:
        pytest.skip("tinygemm_fp8 requires Blackwell+")


# Positive tests — parameterized correctness checks
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("output_features", [32, 64, 128])
@pytest.mark.parametrize("input_features", [360, 720, 1440, 2880])
def test_tinygemm_bf16(batch_size, output_features, input_features):
    _skip_if_not_sm90()
    from flashinfer.gemm import tinygemm_bf16

    input = torch.randn(batch_size, input_features, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        output_features, input_features, device="cuda", dtype=torch.bfloat16
    )
    bias = torch.randn(output_features, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(batch_size, output_features, device="cuda", dtype=torch.bfloat16)

    tinygemm_bf16(input, weight, out, bias=bias)

    # Reference in FP32 for accuracy
    ref = F.linear(input.float(), weight.float(), bias.float()).bfloat16()

    cos_sim = F.cosine_similarity(
        ref.reshape(-1).float(), out.reshape(-1).float(), dim=0
    )
    assert cos_sim > 0.99, f"Cosine similarity {cos_sim:.6f} < 0.99"


# No-bias test — validates bias=None zero-fill path
@pytest.mark.parametrize("batch_size", [1, 4])
def test_tinygemm_bf16_no_bias(batch_size):
    _skip_if_not_sm90()
    from flashinfer.gemm import tinygemm_bf16

    input_features = 256
    output_features = 128

    input = torch.randn(batch_size, input_features, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        output_features, input_features, device="cuda", dtype=torch.bfloat16
    )
    out = torch.empty(batch_size, output_features, device="cuda", dtype=torch.bfloat16)

    tinygemm_bf16(input, weight, out)

    ref = (input.float() @ weight.float().T).bfloat16()

    cos_sim = F.cosine_similarity(
        ref.reshape(-1).float(), out.reshape(-1).float(), dim=0
    )
    assert cos_sim > 0.99, f"Cosine similarity {cos_sim:.6f} < 0.99"


# PDL tests — back-to-back launches with programmatic dependent launch
@pytest.mark.parametrize("num_launches", [2, 4, 8])
@pytest.mark.parametrize("batch_size", [1, 8, 16])
def test_tinygemm_bf16_pdl_back_to_back(num_launches, batch_size):
    """Test use_pdl=True with back-to-back launches on a clean stream.

    PDL (Programmatic Dependent Launch) enables overlapping DMA of kernel N
    with compute of kernel N-1. The first kernel's cudaGridDependencySynchronize()
    sees no previous PDL grid and returns immediately. Each subsequent kernel
    waits for the previous one's cudaTriggerProgrammaticLaunchCompletion() signal.

    We sync the device before launching to ensure no non-PDL ops are pending
    on the stream, then fire all PDL kernels back-to-back without host sync
    in between.
    """
    _skip_if_not_sm90()
    from flashinfer.gemm import tinygemm_bf16

    input_features = 2880
    output_features = 128

    # Pre-allocate everything before the PDL launch burst
    inputs = [
        torch.randn(batch_size, input_features, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_launches)
    ]
    weight = torch.randn(
        output_features, input_features, device="cuda", dtype=torch.bfloat16
    )
    bias = torch.randn(output_features, device="cuda", dtype=torch.bfloat16)
    outs = [
        torch.empty(batch_size, output_features, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_launches)
    ]

    # Ensure no pending non-PDL work on the stream
    torch.cuda.synchronize()

    # Fire all PDL kernels back-to-back (no host sync between them)
    for i in range(num_launches):
        tinygemm_bf16(inputs[i], weight, outs[i], bias=bias, use_pdl=True)

    # Sync and check correctness for every launch
    torch.cuda.synchronize()

    for i in range(num_launches):
        ref = F.linear(inputs[i].float(), weight.float(), bias.float()).bfloat16()
        cos_sim = F.cosine_similarity(
            ref.reshape(-1).float(), outs[i].reshape(-1).float(), dim=0
        )
        assert cos_sim > 0.99, (
            f"Launch {i}/{num_launches}: cosine similarity {cos_sim:.6f} < 0.99"
        )


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
@pytest.mark.parametrize("output_features", [32, 64, 128])
@pytest.mark.parametrize("input_features", [128, 256, 512, 1024])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_tinygemm_fp8(batch_size, output_features, input_features, out_dtype):
    _skip_if_not_blackwell()
    from flashinfer.gemm import tinygemm_fp8

    a_f32 = torch.randn(batch_size, input_features, device="cuda", dtype=torch.float32)
    w_nk_f32 = torch.randn(
        output_features, input_features, device="cuda", dtype=torch.float32
    )

    A, A_scale = to_float8(a_f32, dtype=torch.float8_e4m3fn)
    W_row_major, B_scale = to_float8(w_nk_f32, dtype=torch.float8_e4m3fn)
    B = W_row_major.t()
    bias = torch.randn(output_features, device="cuda", dtype=out_dtype)

    out = tinygemm_fp8(A, B, A_scale, B_scale, out_dtype, bias=bias)

    ref = (A.float() @ B.float()) * (A_scale.float() * B_scale.float())
    ref = (ref + bias.float()).to(out_dtype)

    cos_sim = F.cosine_similarity(
        ref.reshape(-1).float(), out.reshape(-1).float(), dim=0
    )
    assert cos_sim > 0.999, f"Cosine similarity {cos_sim:.6f} < 0.999"


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_tinygemm_fp8_no_bias(batch_size, out_dtype):
    _skip_if_not_blackwell()
    from flashinfer.gemm import tinygemm_fp8

    input_features = 256
    output_features = 128

    a_f32 = torch.randn(batch_size, input_features, device="cuda", dtype=torch.float32)
    w_nk_f32 = torch.randn(
        output_features, input_features, device="cuda", dtype=torch.float32
    )

    A, A_scale = to_float8(a_f32, dtype=torch.float8_e4m3fn)
    W_row_major, B_scale = to_float8(w_nk_f32, dtype=torch.float8_e4m3fn)
    B = W_row_major.t()

    out = tinygemm_fp8(A, B, A_scale, B_scale, out_dtype)
    ref = ((A.float() @ B.float()) * (A_scale.float() * B_scale.float())).to(out_dtype)

    cos_sim = F.cosine_similarity(
        ref.reshape(-1).float(), out.reshape(-1).float(), dim=0
    )
    assert cos_sim > 0.999, f"Cosine similarity {cos_sim:.6f} < 0.999"


@pytest.mark.parametrize("num_launches", [2, 4])
@pytest.mark.parametrize("batch_size", [1, 8, 16])
def test_tinygemm_fp8_pdl_back_to_back(num_launches, batch_size):
    _skip_if_not_blackwell()
    from flashinfer.gemm import tinygemm_fp8

    input_features = 512
    output_features = 128
    out_dtype = torch.bfloat16

    a_inputs = []
    a_scales = []
    for _ in range(num_launches):
        a_f32 = torch.randn(
            batch_size, input_features, device="cuda", dtype=torch.float32
        )
        a_fp8, a_scale = to_float8(a_f32, dtype=torch.float8_e4m3fn)
        a_inputs.append(a_fp8)
        a_scales.append(a_scale)

    w_nk_f32 = torch.randn(
        output_features, input_features, device="cuda", dtype=torch.float32
    )
    W_row_major, B_scale = to_float8(w_nk_f32, dtype=torch.float8_e4m3fn)
    B = W_row_major.t()
    bias = torch.randn(output_features, device="cuda", dtype=out_dtype)

    torch.cuda.synchronize()

    outs = []
    for i in range(num_launches):
        outs.append(
            tinygemm_fp8(
                a_inputs[i],
                B,
                a_scales[i],
                B_scale,
                out_dtype,
                bias=bias,
                use_pdl=True,
            )
        )

    torch.cuda.synchronize()

    for i in range(num_launches):
        ref = (a_inputs[i].float() @ B.float()) * (
            a_scales[i].float() * B_scale.float()
        )
        ref = (ref + bias.float()).to(out_dtype)
        cos_sim = F.cosine_similarity(
            ref.reshape(-1).float(), outs[i].reshape(-1).float(), dim=0
        )
        assert cos_sim > 0.999, (
            f"Launch {i}/{num_launches}: cosine similarity {cos_sim:.6f} < 0.999"
        )
