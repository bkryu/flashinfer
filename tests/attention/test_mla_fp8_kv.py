"""FP8 KV-cache support for the fa2 MLA decode kernel (issue #2745 follow-up).

The fa2 MLA kernel dequantizes an fp8 (e4m3) KV cache to the compute dtype
during the global->smem load, and the per-tensor ``kv_scale`` is folded into
``sm_scale`` (QK) and the output (PV). This test validates the fp8 path against
the bf16 reference (same kernel, bf16 KV) within fp8 quantization tolerance.
"""

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability

FP8_MAX = 448.0  # e4m3 max magnitude


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [256, 1024])
@pytest.mark.parametrize("num_heads", [16, 128])
@pytest.mark.parametrize("page_size", [32, 64])
def test_mla_fp8_kv_matches_bf16(batch_size, seq_len, num_heads, page_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # fa2 MLA is broadly supported; this test only needs the fa2 backend.
    if get_compute_capability(torch.device("cuda"))[0] < 8:
        pytest.skip("fa2 MLA requires SM80+")

    torch.manual_seed(0)
    dev = "cuda:0"
    kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim = 512, 64, 128
    sm_scale = 1.0 / ((qk_nope_head_dim + qk_rope_head_dim) ** 0.5)
    ws_ref = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=dev)
    ws_fp8 = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=dev)

    q_nope = torch.randn(
        batch_size, num_heads, kv_lora_rank, device=dev, dtype=torch.bfloat16
    )
    q_pe = torch.randn(
        batch_size, num_heads, qk_rope_head_dim, device=dev, dtype=torch.bfloat16
    )
    seq = torch.full((batch_size,), seq_len, device=dev, dtype=torch.int32)
    blocks_per_seq = (seq + page_size - 1) // page_size
    total_blocks = int(blocks_per_seq.sum())

    # Scale the magnitude so fp8 quantization is representative.
    ckv = (
        torch.randn(
            total_blocks, page_size, kv_lora_rank, device=dev, dtype=torch.bfloat16
        )
        * 0.1
    )
    kpe = (
        torch.randn(
            total_blocks, page_size, qk_rope_head_dim, device=dev, dtype=torch.bfloat16
        )
        * 0.1
    )

    qi = torch.arange(0, batch_size + 1, device=dev, dtype=torch.int32)
    ki = torch.zeros_like(qi)
    ki[1:] = torch.cumsum(blocks_per_seq, dim=0)
    kidx = torch.arange(total_blocks, device=dev, dtype=torch.int32)

    def plan_run(wrapper, ckv_in, kpe_in, kv_dtype, kv_scale):
        wrapper.plan(
            qi,
            ki,
            kidx,
            seq,
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            page_size,
            False,
            sm_scale,
            torch.bfloat16,
            kv_dtype,
        )
        return wrapper.run(
            q_nope, q_pe, ckv_in, kpe_in, return_lse=False, kv_scale=kv_scale
        )

    o_ref = plan_run(
        flashinfer.mla.BatchMLAPagedAttentionWrapper(ws_ref, backend="fa2"),
        ckv,
        kpe,
        torch.bfloat16,
        None,
    )

    # Per-tensor fp8 quantization with a single shared kv_scale (ckv and kpe).
    amax = max(ckv.abs().max().item(), kpe.abs().max().item())
    kv_scale = amax / FP8_MAX
    ckv8 = (ckv.float() / kv_scale).to(torch.float8_e4m3fn)
    kpe8 = (kpe.float() / kv_scale).to(torch.float8_e4m3fn)

    o_fp8 = plan_run(
        flashinfer.mla.BatchMLAPagedAttentionWrapper(ws_fp8, backend="fa2"),
        ckv8,
        kpe8,
        torch.float8_e4m3fn,
        kv_scale,
    )

    assert torch.isfinite(o_fp8).all()
    diff_abs = (o_fp8.float() - o_ref.float()).abs()
    diff_rel = diff_abs / (o_ref.float().abs() + 1e-3)
    pass_ratio = ((diff_abs <= 0.05) | (diff_rel <= 0.05)).float().mean().item()
    assert pass_ratio >= 0.99, (
        f"fp8 KV MLA mismatch: only {pass_ratio:.1%} of elements within tolerance "
        f"(max_abs_diff={diff_abs.max():.4f})"
    )


def test_mla_fp8_kv_requires_scale():
    """An fp8 KV cache without kv_scale must raise (avoid silent wrong results)."""
    if (
        not torch.cuda.is_available()
        or get_compute_capability(torch.device("cuda"))[0] < 8
    ):
        pytest.skip("fa2 MLA requires SM80+")
    dev = "cuda:0"
    kvr, kpe_d, ps = 512, 64, 64
    ws = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device=dev)
    q_nope = torch.randn(2, 16, kvr, device=dev, dtype=torch.bfloat16)
    q_pe = torch.randn(2, 16, kpe_d, device=dev, dtype=torch.bfloat16)
    seq = torch.full((2,), ps, device=dev, dtype=torch.int32)
    ckv8 = torch.zeros(2, ps, kvr, device=dev, dtype=torch.float8_e4m3fn)
    kpe8 = torch.zeros(2, ps, kpe_d, device=dev, dtype=torch.float8_e4m3fn)
    qi = torch.arange(0, 3, device=dev, dtype=torch.int32)
    ki = torch.tensor([0, 1, 2], device=dev, dtype=torch.int32)
    kidx = torch.arange(2, device=dev, dtype=torch.int32)
    w = flashinfer.mla.BatchMLAPagedAttentionWrapper(ws, backend="fa2")
    w.plan(
        qi,
        ki,
        kidx,
        seq,
        16,
        kvr,
        kpe_d,
        ps,
        False,
        0.07,
        torch.bfloat16,
        torch.float8_e4m3fn,
    )
    with pytest.raises(ValueError, match="kv_scale must be provided"):
        w.run(q_nope, q_pe, ckv8, kpe8, return_lse=False)
