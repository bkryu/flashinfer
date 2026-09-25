# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Qwen3.8-27B one-launch decoder (SM120, CuTe DSL): independent torch oracle + the kernel's own consistency gates.

Oracle (``ref_layer``): the checkpoint's math written from the recipe — absorbed RMSNorm, per-tensor fp8 W8A16
projections, 4-tap causal conv + SiLU, gated delta rule (FLA semantics), gated RMSNorm, per-head q/k RMSNorm +
partial rotate-half RoPE, paged GQA attention with the sigmoid output gate, NVFP4 W4A16 SwiGLU MLP. The kernel rounds
the same intermediates to bf16 (in-projection output, attention output, activation) but sums in a different order
and applies the absorbed norm as bf16(x * w) x fp32 rsqrt, so the gate is a band of a few bf16 ulps of the tensor's
scale rather than bitwise. Run-twice is bitwise. The verify form (spec_rows R) is gated against R sequential plain
launches plus a torch replay of the stashed rows; the bf16 weight kind against the quantized kind on exactly
representable weights; the drafter fold against torch.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

CUDA = "cuda"
H, I, D = 2048, 2048, 256
HK, HV, HQ, HKV = 2, 8, 6, 2
ROT, PAGE = 64, 32
KD, VD, QD, KVD = HK * 128, HV * 128, HQ * D, HKV * D
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def gate():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("the Qwen3.8 megakernel targets SM120")
    pytest.importorskip("cutlass")


@pytest.fixture(scope="module")
def mk():
    gate()
    import flashinfer.experimental.qwen38_megakernel as m

    return m


# ----------------------------------------------------------------------------------------------- synthetic layers
def _fp4(g, n, k):
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=CUDA, generator=g)
    s = (torch.rand(n, k // 16, device=CUDA, generator=g) * 2 + 0.25).to(
        torch.float8_e4m3fn
    )
    return w, s


def make_layer(mk, g, kind, M, seq_lens, npages_extra=3):
    """One layer in the kernel's dict form (+ the un-interleaved gate|up weights under ``_ref`` for the oracle)."""
    ly = {"kind": kind}
    wgu, sgu = _fp4(g, 2 * I, H)
    ly["w_gu_il"], ly["s_gu_il"] = mk.interleave_gate_up(wgu, sgu, I)
    ly["_ref"] = {"w_gu": wgu, "s_gu": sgu}
    ly["w_dn"], ly["s_dn"] = _fp4(g, H, I)
    ly["alpha_gu"] = torch.tensor([0.5, 0.125], device=CUDA)
    ly["alpha_dn"] = torch.tensor([0.02], device=CUDA)
    ly["wn_mid"] = (torch.rand(H, device=CUDA, generator=g) + 0.5).to(torch.bfloat16)
    ly["wn_out"] = (torch.rand(H, device=CUDA, generator=g) + 0.5).to(torch.bfloat16)
    ly["alpha_out"] = torch.tensor([0.01], device=CUDA)
    if kind == 0:
        n1 = 2 * KD + 2 * VD
        ly["w_in"] = (torch.randn(n1, H, device=CUDA, generator=g) * 0.05).to(
            torch.float8_e4m3fn
        )
        ly["alpha_in"] = torch.tensor([0.02, 0.03], device=CUDA)
        ly["w_out"] = (torch.randn(H, VD, device=CUDA, generator=g) * 0.05).to(
            torch.float8_e4m3fn
        )
        ly["w_ba"] = (torch.randn(2 * HV, H, device=CUDA, generator=g) * 0.02).to(
            torch.bfloat16
        )
        ly["conv_w"] = (torch.randn(2 * KD + VD, 4, device=CUDA, generator=g) * 0.3).to(
            torch.bfloat16
        )
        ly["a_log"] = torch.randn(HV, device=CUDA, generator=g) * 0.5
        ly["dt_bias"] = torch.randn(HV, device=CUDA, generator=g) * 0.5
        ly["norm_w"] = (torch.rand(128, device=CUDA, generator=g) + 0.5).to(
            torch.bfloat16
        )
        ly["conv_pool"] = (
            torch.randn(M + 1, 2 * KD + VD, 3, device=CUDA, generator=g) * 0.5
        ).to(torch.bfloat16)
        ly["ssm_pool"] = (
            torch.randn(M + 1, HV, 128, 128, device=CUDA, generator=g) * 0.1
        )
    else:
        n1 = 2 * QD + 2 * KVD
        ly["w_in"] = (torch.randn(n1, H, device=CUDA, generator=g) * 0.05).to(
            torch.float8_e4m3fn
        )
        ly["alpha_in"] = torch.tensor([0.02, 0.03, 0.025, 0.015], device=CUDA)
        ly["w_out"] = (torch.randn(H, QD, device=CUDA, generator=g) * 0.05).to(
            torch.float8_e4m3fn
        )
        ly["wq"] = (torch.rand(D, device=CUDA, generator=g) + 0.5).to(torch.bfloat16)
        ly["wk"] = (torch.rand(D, device=CUDA, generator=g) + 0.5).to(torch.bfloat16)
        npages = sum((s + PAGE - 1) // PAGE for s in seq_lens) + npages_extra
        ly["k_cache"] = torch.randn(npages, PAGE, HKV, D, device=CUDA, generator=g).to(
            torch.bfloat16
        )
        ly["v_cache"] = torch.randn(npages, PAGE, HKV, D, device=CUDA, generator=g).to(
            torch.bfloat16
        )
    return ly


STATE_KEYS = ("conv_pool", "ssm_pool", "k_cache", "v_cache")


def clone_state(layers):
    out = []
    for ly in layers:
        c = dict(ly)
        for k in STATE_KEYS:
            if k in c:
                c[k] = c[k].clone()
        out.append(c)
    return out


def state_of(layers):
    return [ly[k] for ly in layers for k in STATE_KEYS if k in ly]


def block_tables(g, nseq, seq_lens, width=64):
    npages = [(s + PAGE - 1) // PAGE for s in seq_lens]
    perm = torch.randperm(sum(npages) + 3, generator=g, device=CUDA)
    bt = torch.zeros(nseq, width, dtype=torch.int32, device=CUDA)
    used = 0
    for b in range(nseq):
        bt[b, : npages[b]] = perm[used : used + npages[b]].to(torch.int32)
        used += npages[b]
    return bt


def slot_of(bt, b, pos):
    return int(bt[b, pos // PAGE]) * PAGE + pos % PAGE


# ------------------------------------------------------------------------------------------------- torch oracle
def rms(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(
        torch.bfloat16
    )


def w8a16(x, w_fp8, alphas, bounds):
    """x bf16 [M, K] @ dequant(w)^T with a per-tensor alpha per row shard (``bounds`` = shard end rows)."""
    acc = x.float() @ w_fp8.float().t()
    alpha_rows = torch.empty(w_fp8.shape[0], device=x.device)
    start = 0
    for a, end in zip(alphas.reshape(-1).float().tolist(), bounds, strict=False):
        alpha_rows[start:end] = a
        start = end
    return (acc * alpha_rows).to(torch.bfloat16)


def unpack_e2m1(w_packed):
    lut = torch.tensor(
        _E2M1 + [-v for v in _E2M1], dtype=torch.float32, device=w_packed.device
    )
    N, Kh = w_packed.shape
    out = torch.empty(N, 2 * Kh, dtype=torch.float32, device=w_packed.device)
    out[:, 0::2] = lut[(w_packed & 0xF).long()]
    out[:, 1::2] = lut[(w_packed >> 4).long()]
    return out


def w4a16(x, w_packed, w_scale, alphas, bounds):
    N, Kh = w_packed.shape
    K = 2 * Kh
    wf = (
        unpack_e2m1(w_packed).view(N, K // 16, 16) * w_scale.float().unsqueeze(2)
    ).view(N, K)
    acc = x.float() @ wf.to(torch.bfloat16).float().t()
    alpha_rows = torch.empty(N, device=x.device)
    start = 0
    for a, end in zip(alphas.reshape(-1).float().tolist(), bounds, strict=False):
        alpha_rows[start:end] = a
        start = end
    return (acc * alpha_rows).to(torch.bfloat16)


def causal_conv1d_step(x, weight, state):
    """x bf16 [C], weight [C, 4], state [C, 3] -> (silu(conv) bf16 [C], new state [C, 3]); bf16 math like HF."""
    full = torch.cat([state.to(x.dtype), x.unsqueeze(1)], dim=1)  # [C, 4]
    y = (full.float() * weight.float()).sum(1).to(x.dtype)
    return F.silu(y.float()).to(x.dtype), full[:, 1:].contiguous()


def gated_delta_rule_step(q, k, v, a, b, a_log, dt_bias, S):
    """One token: q/k [HK, 128], v [HV, 128], a/b [HV] (bf16), S fp32 [HV, 128, 128] -> (o bf16 [HV, 128], S')."""

    def l2n(x):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    beta = torch.sigmoid(b).float()
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    rep = HV // HK
    qf = (l2n(q.float()) * 128**-0.5).repeat_interleave(rep, dim=0)
    kf = l2n(k.float()).repeat_interleave(rep, dim=0)
    S = S * g.exp().view(HV, 1, 1)
    kv_mem = (S * kf.unsqueeze(-1)).sum(dim=-2)
    delta = (v.float() - kv_mem) * beta.unsqueeze(-1)
    S = S + kf.unsqueeze(-1) * delta.unsqueeze(-2)
    o = (S * qf.unsqueeze(-1)).sum(dim=-2)
    return o.to(torch.bfloat16), S


def rms_norm_gated(x, gate, w, eps=1e-6):
    xf = x.float()
    y = (w.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps))).to(
        x.dtype
    )
    return (y * F.silu(gate.float())).to(x.dtype)


def rope_(x, positions, table):
    """x [M, heads, D] rotated in place on the first ROT dims (rotate-half pairs (i, i + ROT/2))."""
    half = ROT // 2
    cs = table[positions.long()]  # [M, ROT]
    cos, sin = cs[:, None, :half], cs[:, None, half:]
    xf = x.float()
    x1, x2 = xf[..., :half].clone(), xf[..., half:ROT].clone()
    xf[..., :half] = x1 * cos - x2 * sin
    xf[..., half:ROT] = x2 * cos + x1 * sin
    x.copy_(xf.to(x.dtype))


def ref_layer(
    ly, x, wn_in, rope_table, positions, slot_mapping, seq_lens, bt, gslots, eps=1e-6
):
    """resid bf16 [M, H] -> new resid; pools / caches in ``ly`` advanced in place. One token per sequence (R = 1)."""
    M = x.shape[0]
    h = rms(x, wn_in, eps)
    if ly["kind"] == 0:
        w = 2 * KD + VD
        mz = w8a16(h, ly["w_in"], ly["alpha_in"], (w, w + VD))
        mixed, z = mz[:, :w], mz[:, w:].view(M, HV, 128)
        ba = (h.float() @ ly["w_ba"].float().t()).to(torch.bfloat16)
        b, a = ba[:, :HV], ba[:, HV:]
        o = torch.empty(M, HV, 128, dtype=torch.bfloat16, device=CUDA)
        for i in range(M):
            s = int(gslots[i])
            y, ly["conv_pool"][s] = causal_conv1d_step(
                mixed[i], ly["conv_w"], ly["conv_pool"][s]
            )
            q, k, v = (
                y[:KD].view(HK, 128),
                y[KD : 2 * KD].view(HK, 128),
                y[2 * KD :].view(HV, 128),
            )
            o[i], S = gated_delta_rule_step(
                q, k, v, a[i], b[i], ly["a_log"], ly["dt_bias"], ly["ssm_pool"][s]
            )
            ly["ssm_pool"][s] = S
        attn = rms_norm_gated(o, z, ly["norm_w"], eps).view(M, VD)
    else:
        qgkv = w8a16(
            h, ly["w_in"], ly["alpha_in"], (QD, 2 * QD, 2 * QD + KVD, 2 * QD + 2 * KVD)
        )
        q = qgkv[:, :QD].reshape(M, HQ, D).clone()
        gate = qgkv[:, QD : 2 * QD]
        k = qgkv[:, 2 * QD : 2 * QD + KVD].reshape(M, HKV, D).clone()
        v = qgkv[:, 2 * QD + KVD :].reshape(M, HKV, D)
        q = rms(q, ly["wq"], eps)
        k = rms(k, ly["wk"], eps)
        rope_(q, positions, rope_table)
        rope_(k, positions, rope_table)
        for i in range(M):
            sidx = int(slot_mapping[i])
            ly["k_cache"][sidx // PAGE, sidx % PAGE] = k[i]
            ly["v_cache"][sidx // PAGE, sidx % PAGE] = v[i]
        out = torch.empty(M, HQ, D, dtype=torch.float32, device=CUDA)
        rep = HQ // HKV
        for i in range(M):
            L = int(seq_lens[i])
            pages = bt[i, : (L + PAGE - 1) // PAGE].long()
            K = (
                ly["k_cache"][pages]
                .reshape(-1, HKV, D)[:L]
                .float()
                .repeat_interleave(rep, dim=1)
            )  # [L, HQ, D]
            V = (
                ly["v_cache"][pages]
                .reshape(-1, HKV, D)[:L]
                .float()
                .repeat_interleave(rep, dim=1)
            )
            sc = torch.einsum("hd,lhd->hl", q[i].float(), K) * D**-0.5
            out[i] = torch.einsum("hl,lhd->hd", torch.softmax(sc, dim=-1), V)
        attn = (
            out.to(torch.bfloat16).view(M, QD).float() * torch.sigmoid(gate.float())
        ).to(torch.bfloat16)
    resid = (x.float() + w8a16(attn, ly["w_out"], ly["alpha_out"], (H,)).float()).to(
        torch.bfloat16
    )
    h2 = rms(resid, ly["wn_mid"], eps)
    gu = w4a16(h2, ly["_ref"]["w_gu"], ly["_ref"]["s_gu"], ly["alpha_gu"], (I, 2 * I))
    act = (F.silu(gu[:, :I].float()) * gu[:, I:].float()).to(torch.bfloat16)
    return (
        resid.float() + w4a16(act, ly["w_dn"], ly["s_dn"], ly["alpha_dn"], (H,)).float()
    ).to(torch.bfloat16)


def band(name, got, ref, max_tol, mean_tol):
    a_, b_ = got.float(), ref.float()
    scale = float(b_.abs().max()) + 1e-6
    err = float((a_ - b_).abs().max())
    mean = float((a_ - b_).abs().mean())
    print(
        f"{name}: max|d| {err:.4g} = {err / scale * 100:.3f}% of scale {scale:.3g}, mean {mean / scale * 100:.4f}%"
    )
    assert err <= max_tol * scale + 1e-3, (name, err, scale)
    assert mean <= mean_tol * scale + 1e-6, (name, mean, scale)


class Case:
    """Decode step geometry: ``nseq`` sequences x ``R`` ordered rows each (R = 1: plain decode)."""

    def __init__(
        self,
        g,
        nseq,
        R=1,
        committed=(100, 33, 700, 1, 260, 5, 64, 65, 9, 300, 17, 2, 128, 1000, 31, 77),
    ):
        self.nseq, self.R, self.M = nseq, R, nseq * R
        self.committed = list(committed[:nseq])
        self.seq_lens_list = [c + R for c in self.committed]
        self.bt = block_tables(g, nseq, self.seq_lens_list)
        self.gslots = torch.arange(nseq, device=CUDA, dtype=torch.int32)
        self.positions = torch.tensor(
            [c + r for c in self.committed for r in range(R)],
            dtype=torch.int32,
            device=CUDA,
        )
        self.slot_mapping = torch.tensor(
            [
                slot_of(self.bt, b, c + r)
                for b, c in enumerate(self.committed)
                for r in range(R)
            ],
            dtype=torch.int32,
            device=CUDA,
        )
        self.seq_lens = torch.tensor(self.seq_lens_list, dtype=torch.int32, device=CUDA)
        self.max_seq_len = max(self.seq_lens_list)


def run_kernel(mk, spec, layers0, case, x, wn0, KS=4, ctas=None, twice=True):
    """The one-launch decoder over cloned state; returns [resid, xw_out, sq_out, act, *state] (checked run-twice bitwise)."""
    outs = []
    for _ in range(2 if twice else 1):
        layers = clone_state(layers0)
        tables = mk.DecoderTables(spec, layers, case.M, KS)
        xw0 = mk.slack_buffer((case.M, H), torch.bfloat16, CUDA)
        torch.mul(x, wn0, out=xw0)
        sq0 = torch.zeros(H // 64, case.M, dtype=torch.float32, device=CUDA)
        sq0[0:1] = mk.rowstat(x)
        xw1 = mk.slack_buffer((case.M, H), torch.bfloat16, CUDA)
        sq1 = torch.empty(H // 64, case.M, dtype=torch.float32, device=CUDA)
        act = mk.slack_buffer((case.M, I), torch.bfloat16, CUDA)
        resid = x.clone()
        mk.decoder_entry(
            spec,
            tables,
            xw0,
            sq0,
            xw1,
            sq1,
            resid,
            act,
            mk.build_rope_cache(4096, ROT, device=CUDA),
            case.positions,
            case.slot_mapping,
            case.seq_lens,
            case.bt,
            case.gslots,
            max_ctas=ctas,
        )(torch.cuda.current_stream())
        torch.cuda.synchronize()
        outs.append([resid, xw0, sq0, act] + state_of(layers))
    if twice:
        for a_, b_ in zip(outs[0], outs[1], strict=False):
            assert torch.equal(a_, b_), "run-twice not bitwise"
    return outs[0]


def run_ref(layers0, case, x, wn0):
    layers = clone_state(layers0)
    from flashinfer.experimental.qwen38_megakernel import build_rope_cache

    table = build_rope_cache(4096, ROT, device=CUDA)
    resid = x.clone()
    wn = wn0
    for ly in layers:
        resid = ref_layer(
            ly,
            resid,
            wn,
            table,
            case.positions,
            case.slot_mapping,
            case.seq_lens,
            case.bt,
            ly.get("_gslots", case.gslots),
        )
        wn = ly["wn_out"]
    return resid, state_of(layers)


# ------------------------------------------------------------------------------------------------------- tests
@pytest.mark.parametrize(
    "ctas", [None, 24], ids=["grid", "grid24"]
)  # 24 CTAs: the in-projection has a tail (40 / 64 tiles)
@pytest.mark.parametrize(
    "M", [1, 4, 8, 16]
)  # M <= 8: the TILE_N 8 kernel; 16: TILE_N 16
@pytest.mark.parametrize(
    "kinds", [(0,), (1,), (0, 1, 0, 0)], ids=["gdn", "attn", "mixed4"]
)
def test_decoder_matches_torch_oracle(mk, kinds, M, ctas):
    g = torch.Generator(device=CUDA).manual_seed(31 + M)
    case = Case(g, M)
    layers0 = [make_layer(mk, g, k, M, case.seq_lens_list) for k in kinds]
    x = (torch.randn(M, H, device=CUDA, generator=g) * 2).to(torch.bfloat16)
    wn0 = (torch.rand(H, device=CUDA, generator=g) + 0.5).to(torch.bfloat16)
    spec = mk.DecoderSpec(HK, HV, HQ, HKV, D, ROT, PAGE, H, I, splits=2, eps=1e-6)
    got = run_kernel(mk, spec, layers0, case, x, wn0, ctas=ctas)
    ref_resid, ref_state = run_ref(layers0, case, x, wn0)
    band("resid", got[0], ref_resid, 0.03, 0.003)
    band(
        "xw_out",
        got[1],
        (ref_resid.float() * layers0[-1]["wn_out"].float()).to(torch.bfloat16),
        0.03,
        0.003,
    )
    sq_ref = ref_resid.float().pow(2).sum(-1)
    band("sq_out", got[2].sum(0), sq_ref, 0.03, 0.03)
    names = [k for ly in layers0 for k in STATE_KEYS if k in ly]
    for name, a_, b_ in zip(names, got[4:], ref_state, strict=False):
        band(name, a_, b_, 0.02, 0.002)


def test_counters_rearmed_and_deterministic(mk):
    g = torch.Generator(device=CUDA).manual_seed(5)
    case = Case(g, 4)
    layers0 = [make_layer(mk, g, k, 4, case.seq_lens_list) for k in (1, 0)]
    x = (torch.randn(4, H, device=CUDA, generator=g) * 2).to(torch.bfloat16)
    wn0 = torch.ones(H, dtype=torch.bfloat16, device=CUDA)
    spec = mk.DecoderSpec(HK, HV, HQ, HKV, D, ROT, PAGE, H, I, splits=2)
    run_kernel(mk, spec, layers0, case, x, wn0)
    from flashinfer.experimental.qwen38_megakernel import decoder_host

    for c in decoder_host._CNT.values():
        assert int(c.abs().sum()) == 0, "counters not reset"


@pytest.mark.parametrize("nseq", [1, 2, 4])
def test_verify_form_matches_sequential_plain(mk, nseq):
    """R = 4 ordered rows per sequence in ONE launch (GDN pools read-only, stash filled) vs the same tokens fed one at a
    time to the plain decoder (pools advanced in place); stash replayed in torch == the sequential pools."""
    R = 4
    kinds = (0, 1, 0, 0)
    g = torch.Generator(device=CUDA).manual_seed(41)
    case = Case(g, nseq, R, committed=(100, 33, 700, 5))
    M = case.M
    layers0 = [make_layer(mk, g, k, M, case.seq_lens_list) for k in kinds]
    for ly in layers0:
        if ly["kind"] == 0:
            W = ly["conv_w"].shape[0]
            ly["stash_x"] = torch.zeros(M, W, dtype=torch.bfloat16, device=CUDA)
            ly["stash_y"] = torch.zeros(M, W, dtype=torch.bfloat16, device=CUDA)
            ly["stash_a"] = torch.zeros(M, HV, dtype=torch.float32, device=CUDA)
            ly["stash_b"] = torch.zeros(M, HV, dtype=torch.float32, device=CUDA)
    x = (torch.randn(M, H, device=CUDA, generator=g) * 2).to(torch.bfloat16)
    wn0 = (torch.rand(H, device=CUDA, generator=g) + 0.5).to(torch.bfloat16)
    spec_v = mk.DecoderSpec(HK, HV, HQ, HKV, D, ROT, PAGE, H, I, splits=2, spec_rows=R)
    spec_p = mk.DecoderSpec(HK, HV, HQ, HKV, D, ROT, PAGE, H, I, splits=2)
    table = mk.build_rope_cache(4096, ROT, device=CUDA)

    def entry(spec, layers, xin, pos, slots, sl, act, resid):
        Mx = xin.shape[0]
        xw0 = mk.slack_buffer((Mx, H), torch.bfloat16, CUDA)
        torch.mul(xin, wn0, out=xw0)
        sq0 = torch.zeros(H // 64, Mx, dtype=torch.float32, device=CUDA)
        sq0[0:1] = mk.rowstat(xin)
        xw1 = mk.slack_buffer((Mx, H), torch.bfloat16, CUDA)
        sq1 = torch.empty(H // 64, Mx, dtype=torch.float32, device=CUDA)
        tables = mk.DecoderTables(spec, layers, Mx, 4)
        mk.decoder_entry(
            spec,
            tables,
            xw0,
            sq0,
            xw1,
            sq1,
            resid,
            act,
            table,
            pos,
            slots,
            sl,
            case.bt,
            case.gslots,
        )(torch.cuda.current_stream())
        torch.cuda.synchronize()

    layers_p = clone_state(layers0)
    resid_ref = torch.empty(M, H, dtype=torch.bfloat16, device=CUDA)
    for r in range(R):
        rows = [b * R + r for b in range(nseq)]
        xin = x[rows].contiguous()
        pos = torch.tensor(
            [c + r for c in case.committed], dtype=torch.int32, device=CUDA
        )
        slots = torch.tensor(
            [slot_of(case.bt, b, c + r) for b, c in enumerate(case.committed)],
            dtype=torch.int32,
            device=CUDA,
        )
        sl = torch.tensor(
            [c + r + 1 for c in case.committed], dtype=torch.int32, device=CUDA
        )
        resid = xin.clone()
        entry(
            spec_p,
            layers_p,
            xin,
            pos,
            slots,
            sl,
            mk.slack_buffer((nseq, I), torch.bfloat16, CUDA),
            resid,
        )
        resid_ref[rows] = resid

    layers_v = clone_state(layers0)
    pools_before = [
        (ly["conv_pool"].clone(), ly["ssm_pool"].clone())
        for ly in layers_v
        if ly["kind"] == 0
    ]
    resid_v = x.clone()
    entry(
        spec_v,
        layers_v,
        x,
        case.positions,
        case.slot_mapping,
        case.seq_lens,
        mk.slack_buffer((M, I), torch.bfloat16, CUDA),
        resid_v,
    )
    band("verify-vs-sequential resid", resid_v, resid_ref, 0.02, 0.002)
    gdn_v = [ly for ly in layers_v if ly["kind"] == 0]
    for (c0, s0), ly in zip(pools_before, gdn_v, strict=False):
        assert torch.equal(ly["conv_pool"], c0) and torch.equal(ly["ssm_pool"], s0), (
            "verify form wrote a pool"
        )
    for ly_v, ly_p in zip(
        [ly for ly in layers_v if ly["kind"] == 1],
        [ly for ly in layers_p if ly["kind"] == 1],
        strict=False,
    ):
        for b, c in enumerate(case.committed):
            for r in range(R):
                sidx = slot_of(case.bt, b, c + r)
                torch.testing.assert_close(
                    ly_v["k_cache"][sidx // PAGE, sidx % PAGE],
                    ly_p["k_cache"][sidx // PAGE, sidx % PAGE],
                    rtol=1e-2,
                    atol=1e-2,
                )
                torch.testing.assert_close(
                    ly_v["v_cache"][sidx // PAGE, sidx % PAGE],
                    ly_p["v_cache"][sidx // PAGE, sidx % PAGE],
                    rtol=1e-2,
                    atol=1e-2,
                )
    # stash replay (every row accepted): conv taps = the last 3 raw inputs; state = the delta rule over the stashed rows
    gdn_p = [ly for ly in layers_p if ly["kind"] == 0]
    for li, (lv, lp) in enumerate(zip(gdn_v, gdn_p, strict=False)):
        for b in range(nseq):
            s = int(case.gslots[b])
            rows = slice(b * R, b * R + R)
            xs = lv["stash_x"][rows]  # [R, W] raw conv inputs
            taps = torch.cat([lv["conv_pool"][s], xs.t()], dim=1)[:, -3:]
            torch.testing.assert_close(taps, lp["conv_pool"][s], rtol=2e-2, atol=1e-3)
            S = lv["ssm_pool"][s].clone()
            for r in range(R):
                y = lv["stash_y"][b * R + r]
                q, k, v = (
                    y[:KD].view(HK, 128),
                    y[KD : 2 * KD].view(HK, 128),
                    y[2 * KD :].view(HV, 128),
                )
                _, S = gated_delta_rule_step(
                    q,
                    k,
                    v,
                    lv["stash_a"][b * R + r],
                    lv["stash_b"][b * R + r],
                    lv["a_log"],
                    lv["dt_bias"],
                    S,
                )
            band(f"stash replay layer {li} seq {b}", S, lp["ssm_pool"][s], 0.02, 0.002)


# --------------------------------------------------------------------------------------- bf16 kind (MTP drafter)
def _fp4_to_bf16(w_packed, w_scale):
    N, Kh = w_packed.shape
    K = 2 * Kh
    wf = (
        unpack_e2m1(w_packed).view(N, K // 16, 16) * w_scale.float().unsqueeze(2)
    ).view(N, K)
    wb = wf.to(torch.bfloat16)
    assert torch.equal(wb.float(), wf), "e2m1 x e4m3 products must be exact in bf16"
    return wb.contiguous()


def _fold(w, alpha_rows):
    out = (w.float() * alpha_rows.float().unsqueeze(1)).to(torch.bfloat16)
    assert torch.equal(out.float(), w.float() * alpha_rows.float().unsqueeze(1)), (
        "power-of-two alphas fold exactly"
    )
    return out.contiguous()


def unit_alphas(ly):
    c = dict(ly)
    n = ly["alpha_in"].numel()
    c["alpha_in"] = torch.tensor([2.0**-5, 2.0**-5, 2.0**-6, 2.0**-6][:n], device=CUDA)
    c["alpha_out"] = torch.tensor([2.0**-6], device=CUDA)
    c["alpha_gu"] = torch.tensor([0.5, 0.125], device=CUDA)
    c["alpha_dn"] = torch.tensor([2.0**-6], device=CUDA)
    return c


def bf16_twin(ly):
    """The same layer with bf16 weights equal to the quantized kind's dequantized values x its power-of-two alphas."""
    c = dict(ly)
    n_in = ly["w_in"].shape[0]
    a_in = ly["alpha_in"].float()
    if ly["kind"] == 0:
        rows = torch.cat([a_in[0].repeat(n_in - VD), a_in[1].repeat(VD)])
    else:
        rows = torch.cat(
            [
                a_in[0].repeat(QD),
                a_in[1].repeat(QD),
                a_in[2].repeat(KVD),
                a_in[3].repeat(KVD),
            ]
        )
    c["w_in"] = _fold(ly["w_in"].to(torch.bfloat16), rows.to(CUDA))
    c["w_out"] = _fold(
        ly["w_out"].to(torch.bfloat16), ly["alpha_out"].float().repeat(H)
    )
    c["alpha_in"] = torch.ones_like(ly["alpha_in"])
    c["alpha_out"] = torch.ones_like(ly["alpha_out"])
    gu = _fp4_to_bf16(ly["w_gu_il"], ly["s_gu_il"])
    a_gu = ly["alpha_gu"].float()
    rows_gu = (
        torch.stack([a_gu[0].repeat(8), a_gu[1].repeat(8)]).reshape(-1).repeat(I // 8)
    )
    c["w_gu_il"] = _fold(gu, rows_gu)
    c["w_dn"] = _fold(
        _fp4_to_bf16(ly["w_dn"], ly["s_dn"]), ly["alpha_dn"].float().repeat(H)
    )
    c["s_gu_il"] = torch.zeros(
        64, H // 16, dtype=torch.uint8, device=CUDA
    )  # descriptor placeholders, never read
    c["s_dn"] = torch.zeros(64, I // 16, dtype=torch.uint8, device=CUDA)
    c["alpha_gu"] = torch.ones_like(ly["alpha_gu"])
    c["alpha_dn"] = torch.ones_like(ly["alpha_dn"])
    return c


@pytest.mark.parametrize("R,nseq", [(1, 1), (1, 8), (1, 16), (4, 1), (4, 4)])
@pytest.mark.parametrize("kinds", [(1,), (0, 1)], ids=["attn", "gdn+attn"])
def test_bf16_kind_matches_quantized_kind_on_exact_weights(mk, R, nseq, kinds):
    g = torch.Generator(device=CUDA).manual_seed(51)
    case = Case(g, nseq, R)
    M = case.M
    layers_q = [unit_alphas(make_layer(mk, g, k, M, case.seq_lens_list)) for k in kinds]
    if R > 1:
        for ly in layers_q:
            if ly["kind"] == 0:
                W = ly["conv_w"].shape[0]
                ly["stash_x"] = torch.zeros(M, W, dtype=torch.bfloat16, device=CUDA)
                ly["stash_y"] = torch.zeros(M, W, dtype=torch.bfloat16, device=CUDA)
                ly["stash_a"] = torch.zeros(M, HV, dtype=torch.float32, device=CUDA)
                ly["stash_b"] = torch.zeros(M, HV, dtype=torch.float32, device=CUDA)
    layers_b = [bf16_twin(ly) for ly in layers_q]
    x = (torch.randn(M, H, device=CUDA, generator=g) * 2).to(torch.bfloat16)
    wn0 = (torch.rand(H, device=CUDA, generator=g) + 0.5).to(torch.bfloat16)
    knobs = dict(
        splits=2, eps=1e-6, dual_acc=False, spec_rows=R
    )  # single accumulator chains in both arms
    ref = run_kernel(
        mk,
        mk.DecoderSpec(HK, HV, HQ, HKV, D, ROT, PAGE, H, I, **knobs),
        layers_q,
        case,
        x,
        wn0,
        twice=False,
    )
    got = run_kernel(
        mk,
        mk.DecoderSpec(HK, HV, HQ, HKV, D, ROT, PAGE, H, I, wtype="bf16", **knobs),
        layers_b,
        case,
        x,
        wn0,
    )
    names = ["resid", "xw_out", "sq_out", "act"] + [
        "state%d" % i for i in range(len(ref) - 4)
    ]
    for name, a_, b_ in zip(names, got, ref, strict=False):
        band("bf16-vs-quant " + name, a_, b_, 0.01, 1e-3)


def test_bf16_kind_rejects_buffers_without_slack(mk):
    spec_b = mk.DecoderSpec(HK, HV, HQ, HKV, D, ROT, PAGE, H, I, splits=2, wtype="bf16")
    g = torch.Generator(device=CUDA).manual_seed(7)
    ly = bf16_twin(unit_alphas(make_layer(mk, g, 1, 1, [40])))
    tables = mk.DecoderTables(spec_b, [ly], 1, 2)
    z = torch.zeros(1, H, dtype=torch.bfloat16, device=CUDA)
    sq = torch.zeros(H // 64, 1, dtype=torch.float32, device=CUDA)
    bt = torch.zeros(1, 4, dtype=torch.int32, device=CUDA)

    def i32(*v):
        return torch.tensor(v, dtype=torch.int32, device=CUDA)

    with pytest.raises(ValueError, match="slack"):
        mk.decoder_entry(
            spec_b,
            tables,
            z,
            sq,
            z.clone(),
            sq.clone(),
            z.clone(),
            torch.zeros(1, I, dtype=torch.bfloat16, device=CUDA),
            mk.build_rope_cache(64, ROT, device=CUDA),
            i32(39),
            i32(39),
            i32(40),
            bt,
            i32(0),
        )


@pytest.mark.parametrize("R,nseq", [(1, 1), (1, 8), (4, 4)])
def test_drafter_fold_matches_torch(mk, R, nseq):
    """fc + the two pre-fc norms in front of the layer and the final norm behind it, inside the kernel. Zero out-proj /
    down weights make the kernel's residual exactly the fc phase's output (the layer still runs: its out-projection
    items read the residual the fc reducers wrote — the ordering the fold's gate guards)."""
    g = torch.Generator(device=CUDA).manual_seed(61)
    case = Case(g, nseq, R, committed=(100, 33, 700, 1, 260, 5, 64, 65))
    M = case.M
    ly = bf16_twin(unit_alphas(make_layer(mk, g, 1, M, case.seq_lens_list)))
    ly["w_fc"] = (torch.randn(H, 2 * H, device=CUDA, generator=g) * 0.02).to(
        torch.bfloat16
    )
    ly["w_pre"] = (torch.rand(2 * H, device=CUDA, generator=g) + 0.5).to(torch.bfloat16)
    ly["wn_in"] = (torch.rand(H, device=CUDA, generator=g) + 0.5).to(torch.bfloat16)
    ly["w_out"] = torch.zeros_like(ly["w_out"])
    ly["w_dn"] = torch.zeros_like(ly["w_dn"])
    e = (torch.randn(M, H, device=CUDA, generator=g) * 2).to(torch.bfloat16)
    h = (torch.randn(M, H, device=CUDA, generator=g) * 3).to(torch.bfloat16)
    spec_f = mk.DecoderSpec(
        HK,
        HV,
        HQ,
        HKV,
        D,
        ROT,
        PAGE,
        H,
        I,
        splits=2,
        dual_acc=False,
        spec_rows=R,
        wtype="bf16",
        drafter_fold=True,
    )
    xin = (
        torch.cat([rms(e, ly["w_pre"][:H]), rms(h, ly["w_pre"][H:])], dim=-1).float()
        @ ly["w_fc"].float().t()
    ).to(torch.bfloat16)
    table = mk.build_rope_cache(4096, ROT, device=CUDA)

    def run_fold():
        layers = clone_state([ly])
        tables = mk.DecoderTables(spec_f, layers, M, 4)
        xw0 = mk.slack_buffer((M, H), torch.bfloat16, CUDA)
        sq0 = torch.zeros(H // 64, M, dtype=torch.float32, device=CUDA)
        xw1 = mk.slack_buffer((M, H), torch.bfloat16, CUDA)
        sq1 = torch.empty(H // 64, M, dtype=torch.float32, device=CUDA)
        act = mk.slack_buffer((M, I), torch.bfloat16, CUDA)
        xcat = mk.slack_buffer((M, 2 * H), torch.bfloat16, CUDA)
        torch.cat([e, h], dim=-1, out=xcat)
        resid = torch.zeros(M, H, dtype=torch.bfloat16, device=CUDA)
        out = torch.zeros(M, H, dtype=torch.bfloat16, device=CUDA)
        mk.decoder_entry(
            spec_f,
            tables,
            xw0,
            sq0,
            xw1,
            sq1,
            resid,
            act,
            table,
            case.positions,
            case.slot_mapping,
            case.seq_lens,
            case.bt,
            case.gslots,
            xcat=xcat,
            out=out,
        )(torch.cuda.current_stream())
        torch.cuda.synchronize()
        return [resid, out] + state_of(layers)

    got = [run_fold() for _ in range(2)]
    band("fold resid (= fc x)", got[0][0], xin, 0.01, 1e-3)
    band("fold out (final norm)", got[0][1], rms(got[0][0], ly["wn_out"]), 0.01, 1e-3)
    for a_, b_ in zip(got[0], got[1], strict=False):
        assert torch.equal(a_, b_)


# ------------------------------------------------------------------------------------------------ public API
def test_public_api_decode_and_cuda_graph(mk):
    """flashinfer.qwen3_ops: prepare once, decode eagerly, then capture the same call in a CUDA graph; replay == eager
    (bitwise: the kernel is deterministic) and both match the oracle's final-normed residual."""
    from flashinfer.qwen3_ops import qwen38_megakernel_decode, qwen38_megakernel_prepare

    g = torch.Generator(device=CUDA).manual_seed(71)
    M = 4
    case = Case(g, M)
    layers0 = [make_layer(mk, g, k, M, case.seq_lens_list) for k in (0, 1, 0)]
    x = (torch.randn(M, H, device=CUDA, generator=g) * 2).to(torch.bfloat16)
    wn0 = (torch.rand(H, device=CUDA, generator=g) + 0.5).to(torch.bfloat16)
    spec = mk.DecoderSpec(HK, HV, HQ, HKV, D, ROT, PAGE, H, I, splits=2)
    ref_resid, _ = run_ref(layers0, case, x, wn0)
    ref_out = rms(ref_resid, layers0[-1]["wn_out"])

    layers_e = clone_state(layers0)
    plan = qwen38_megakernel_prepare(
        spec, layers_e, M, max_seq_len=case.max_seq_len, wn_in=wn0
    )
    hidden = x.clone()
    out_eager = qwen38_megakernel_decode(
        plan,
        hidden,
        case.positions,
        case.slot_mapping,
        case.seq_lens,
        case.bt,
        case.gslots,
    ).clone()
    torch.cuda.synchronize()
    band("api eager vs oracle", out_eager, ref_out, 0.03, 0.003)

    layers_g = clone_state(layers0)
    plan_g = qwen38_megakernel_prepare(
        spec, layers_g, M, max_seq_len=case.max_seq_len, wn_in=wn0
    )
    out_buf = torch.empty(M, H, dtype=torch.bfloat16, device=CUDA)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        qwen38_megakernel_decode(
            plan_g,
            hidden,
            case.positions,
            case.slot_mapping,
            case.seq_lens,
            case.bt,
            case.gslots,
            out=out_buf,
        )  # warm
        torch.cuda.synchronize()
        for a_, b_ in zip(state_of(layers_g), state_of(layers_e), strict=False):
            assert torch.equal(a_, b_)
        for a_, b_ in zip(
            state_of(layers_g), state_of(layers0), strict=False
        ):  # rewind the state so the capture + replay start fresh
            a_.copy_(b_)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            qwen38_megakernel_decode(
                plan_g,
                hidden,
                case.positions,
                case.slot_mapping,
                case.seq_lens,
                case.bt,
                case.gslots,
                out=out_buf,
            )
        for a_, b_ in zip(state_of(layers_g), state_of(layers0), strict=False):
            a_.copy_(b_)
        graph.replay()
        torch.cuda.synchronize()
    assert torch.equal(out_buf, out_eager), "graph replay != eager"
    for a_, b_ in zip(state_of(layers_g), state_of(layers_e), strict=False):
        assert torch.equal(a_, b_), "graph replay state != eager state"


def test_unsupported_reasons():
    from flashinfer.qwen3_ops import qwen38_megakernel_unsupported_reason

    geo = dict(
        hidden=H,
        inter=I,
        hk=HK,
        hv=HV,
        dk=128,
        dv=128,
        hq=HQ,
        hkv=HKV,
        head_dim=D,
        rot_dim=ROT,
        page_size=PAGE,
        num_tokens=1,
    )
    assert qwen38_megakernel_unsupported_reason("cpu", **geo) == "cuda device required"
    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0):
        assert qwen38_megakernel_unsupported_reason("cuda", **geo) is None
        assert "page" in qwen38_megakernel_unsupported_reason(
            "cuda", **{**geo, "page_size": 16}
        )
        assert "1..16" in qwen38_megakernel_unsupported_reason(
            "cuda", **{**geo, "num_tokens": 17}
        )
        assert "spec_rows" in qwen38_megakernel_unsupported_reason(
            "cuda", **{**geo, "num_tokens": 4, "spec_rows": 3}
        )


# ------------------------------------------------------------------------------ framework (vLLM) layouts
# state_vk: SSM state [slot, hv, dv, dk] (dk innermost) in padded slot pages; conv_sd: conv state [slot, rows, C] (rows =
# K-1 + num_spec) in padded pages, committed taps at rows acc-1..acc+1; kv_packed: K | V as the halves of one
# [pages, page, hkv, 2D] tensor; slot_table: per-token GDN state slots [seq, R] + accepted counts, slot 0 = null.
VLLM = dict(state_vk=True, conv_sd=True, kv_packed=True, slot_table=True)
CONV_ROWS = 6  # K-1 + num_spec (num_spec = 3)
PAD = 384  # padding elements after every slot page (the framework pads mamba pages to the attention page size)


def to_vllm_layout(layers, tap_row0=None):
    """Study-layout layers -> vLLM-form state / caches (fresh storage). ``tap_row0`` [slots] int: the row where each slot's
    committed taps start (acc - 1); default 0."""
    out = []
    for ly in layers:
        c = dict(ly)
        if c["kind"] == 0:
            S_, W, _ = c["conv_pool"].shape
            buf = torch.randn(S_, CONV_ROWS * W + PAD, device=CUDA).to(
                torch.bfloat16
            )  # garbage outside the live rows
            cp = torch.as_strided(buf, (S_, CONV_ROWS, W), (CONV_ROWS * W + PAD, W, 1))
            for s in range(S_):
                r0 = 0 if tap_row0 is None else int(tap_row0[s])
                cp[s, r0 : r0 + 3] = c["conv_pool"][s].t()
            c["conv_pool"] = cp
            c["_conv_buf"] = buf
            sbuf = torch.randn(S_, HV * 16384 + PAD, device=CUDA)
            sp = torch.as_strided(
                sbuf, (S_, HV, 128, 128), (HV * 16384 + PAD, 16384, 128, 1)
            )
            sp.copy_(c["ssm_pool"].transpose(2, 3))
            c["ssm_pool"] = sp
            c["_ssm_buf"] = sbuf
        else:
            kv = torch.empty(
                *c["k_cache"].shape[:3], 2 * D, dtype=torch.bfloat16, device=CUDA
            )
            kv[..., :D] = c["k_cache"]
            kv[..., D:] = c["v_cache"]
            c["k_cache"], c["v_cache"], c["_kv"] = kv[..., :D], kv[..., D:], kv
        out.append(c)
    return out


def from_vllm_state(ly, tap_row0=0):
    """The study-layout view of a vLLM-form layer's state: (conv taps [S, W, 3] from rows tap_row0.., ssm [S, hv, dk, dv]) or (k, v)."""
    if ly["kind"] == 0:
        return ly["conv_pool"][:, tap_row0 : tap_row0 + 3].transpose(1, 2), ly[
            "ssm_pool"
        ].transpose(2, 3)
    return ly["k_cache"], ly["v_cache"]


def run_kernel_vllm(
    mk, spec, layers_v, case, x, wn0, slots_tab, num_accepted, KS=4, pad_rows=0
):
    """One launch on vLLM-form layers (state advanced in place); ``pad_rows`` trailing rows are padding (slot 0, slot_mapping
    -1, seq_len 0, block-table row 0)."""
    M = case.M + pad_rows
    tables = mk.DecoderTables(spec, layers_v, M, KS)
    xw0 = mk.slack_buffer((M, H), torch.bfloat16, CUDA)
    xin = (
        torch.cat([x, torch.zeros(pad_rows, H, dtype=torch.bfloat16, device=CUDA)])
        if pad_rows
        else x
    )
    torch.mul(xin, wn0, out=xw0)
    sq0 = torch.zeros(H // 64, M, dtype=torch.float32, device=CUDA)
    sq0[0:1] = mk.rowstat(xin)
    xw1 = mk.slack_buffer((M, H), torch.bfloat16, CUDA)
    sq1 = torch.empty(H // 64, M, dtype=torch.float32, device=CUDA)
    act = mk.slack_buffer((M, I), torch.bfloat16, CUDA)
    resid = xin.clone()
    R = int(spec.spec_rows)
    nseq = M // R
    pos = torch.cat(
        [case.positions, torch.zeros(pad_rows, dtype=torch.int32, device=CUDA)]
    )
    slot_map = torch.cat(
        [case.slot_mapping, torch.full((pad_rows,), -1, dtype=torch.int32, device=CUDA)]
    )
    seq_lens = torch.cat(
        [case.seq_lens, torch.zeros(pad_rows // R, dtype=torch.int32, device=CUDA)]
    )
    bt = torch.cat(
        [
            case.bt,
            torch.zeros(
                pad_rows // R, case.bt.shape[1], dtype=torch.int32, device=CUDA
            ),
        ]
    )
    st = torch.cat(
        [
            slots_tab,
            torch.zeros(
                *slots_tab.shape[:-2],
                pad_rows // R,
                slots_tab.shape[-1],
                dtype=torch.int32,
                device=CUDA,
            ),
        ],
        dim=-2,
    )
    acc = torch.cat(
        [num_accepted, torch.ones(pad_rows // R, dtype=torch.int32, device=CUDA)]
    )
    assert st.shape[-2] == nseq
    mk.decoder_entry(
        spec,
        tables,
        xw0,
        sq0,
        xw1,
        sq1,
        resid,
        act,
        mk.build_rope_cache(4096, ROT, device=CUDA),
        pos,
        slot_map,
        seq_lens,
        bt,
        st,
        num_accepted=acc,
    )(torch.cuda.current_stream())
    torch.cuda.synchronize()
    return resid[: case.M], xw0[: case.M]


@pytest.mark.parametrize(
    "M,pad_rows", [(1, 0), (4, 0), (16, 0), (3, 1)], ids=["m1", "m4", "m16", "m3+pad"]
)
@pytest.mark.parametrize("kinds", [(0, 1, 0, 0)], ids=["mixed4"])
def test_vllm_layouts_plain_match_torch_oracle(mk, kinds, M, pad_rows):
    """Plain decode on the framework layouts (state slots 1..M, slot 0 = null): outputs and the advanced state (viewed back
    in the study layout) match the torch oracle; padded rows touch neither slot 0's SSM state nor real rows."""
    g = torch.Generator(device=CUDA).manual_seed(131 + M)
    case = Case(g, M)
    layers0 = [make_layer(mk, g, k, M, case.seq_lens_list) for k in kinds]
    x = (torch.randn(M, H, device=CUDA, generator=g) * 2).to(torch.bfloat16)
    wn0 = (torch.rand(H, device=CUDA, generator=g) + 0.5).to(torch.bfloat16)
    # 1-based slots (vLLM's slot 0 is the null block); three slot-table groups like vLLM's GDN KV-cache groups, GDN layer j
    # reading group j % 3 (a different permutation of 1..M each)
    groups = torch.stack(
        [torch.randperm(M, generator=g, device=CUDA) + 1 for _ in range(3)]
    ).to(torch.int32)  # [3, M]
    gi = 0
    for ly in layers0:
        if ly["kind"] == 0:
            ly["slot_group"] = gi % 3
            ly["_gslots"] = groups[gi % 3]
            gi += 1
    case.gslots = groups[0]
    ref_resid, ref_state = run_ref(layers0, case, x, wn0)
    layers_v = to_vllm_layout(clone_state(layers0))
    ssm0_before = [ly["ssm_pool"][0].clone() for ly in layers_v if ly["kind"] == 0]
    spec = mk.DecoderSpec(HK, HV, HQ, HKV, D, ROT, PAGE, H, I, splits=2, **VLLM)
    resid, xw = run_kernel_vllm(
        mk,
        spec,
        layers_v,
        case,
        x,
        wn0,
        groups.view(3, M, 1),
        torch.ones(M, dtype=torch.int32, device=CUDA),
        pad_rows=pad_rows,
    )
    band("vllm-layout resid", resid, ref_resid, 0.03, 0.003)
    band(
        "vllm-layout xw_out",
        xw,
        (ref_resid.float() * layers0[-1]["wn_out"].float()).to(torch.bfloat16),
        0.03,
        0.003,
    )
    names = [k for ly in layers0 for k in STATE_KEYS if k in ly]
    got = [t for ly in layers_v for t in from_vllm_state(ly)]
    for name, a_, b_ in zip(names, got, ref_state, strict=True):
        if name == "conv_pool":
            band(
                name, a_[1:], b_[1:], 0.02, 0.002
            )  # slot 0 (null) is not compared: padded rows may shift its taps
        else:
            band(name, a_, b_, 0.02, 0.002)
    for s0, ly in zip(
        ssm0_before, [ly for ly in layers_v if ly["kind"] == 0], strict=True
    ):
        assert torch.equal(ly["ssm_pool"][0], s0), "padded row wrote the null SSM slot"


@pytest.mark.parametrize("nseq", [1, 2])
def test_vllm_per_token_slots_verify_matches_sequential(mk, nseq):
    """The verify form with vLLM per-token state slots: sequence b reads its state from slots[b, acc-1] (the accepted prefix
    of the last step) and its conv taps from window rows acc-1.., writes S_r to slots[b, r] for every row and the window
    [old(acc), old(acc+1), x_0..x_{R-1}] to rows 0..R+1 of slots[b, 0]. Reference: the study-layout plain decoder fed the
    same R tokens one at a time, each intermediate state captured."""
    R = 4
    kinds = (0, 1, 0, 0)
    g = torch.Generator(device=CUDA).manual_seed(141 + nseq)
    case = Case(g, nseq, R, committed=(100, 33))
    M = case.M
    acc = torch.tensor(
        [3, 1][:nseq], dtype=torch.int32, device=CUDA
    )  # accepted counts of the previous step (1..R)
    n_slots = nseq * R + 1
    layers0 = [
        make_layer(mk, g, k, n_slots - 1, case.seq_lens_list) for k in kinds
    ]  # pools with n_slots slots (0 = null)
    slots_tab = torch.arange(1, n_slots, dtype=torch.int32, device=CUDA).view(nseq, R)
    x = (torch.randn(M, H, device=CUDA, generator=g) * 2).to(torch.bfloat16)
    wn0 = (torch.rand(H, device=CUDA, generator=g) + 0.5).to(torch.bfloat16)
    table = mk.build_rope_cache(4096, ROT, device=CUDA)
    spec_p = mk.DecoderSpec(HK, HV, HQ, HKV, D, ROT, PAGE, H, I, splits=2)

    # ---- reference: study layout, one state slot per sequence = its "read" slot, R sequential plain launches
    read_slots = slots_tab[torch.arange(nseq, device=CUDA), acc.long() - 1]
    layers_p = clone_state(layers0)
    resid_ref = torch.empty(M, H, dtype=torch.bfloat16, device=CUDA)
    states_ref = []  # per row r: [(conv taps, ssm) of the read slots after r+1 tokens]
    for r in range(R):
        rows = [b * R + r for b in range(nseq)]
        xin = x[rows].contiguous()
        pos = torch.tensor(
            [c + r for c in case.committed], dtype=torch.int32, device=CUDA
        )
        slots = torch.tensor(
            [slot_of(case.bt, b, c + r) for b, c in enumerate(case.committed)],
            dtype=torch.int32,
            device=CUDA,
        )
        sl = torch.tensor(
            [c + r + 1 for c in case.committed], dtype=torch.int32, device=CUDA
        )
        tables = mk.DecoderTables(spec_p, layers_p, nseq, 4)
        xw0 = mk.slack_buffer((nseq, H), torch.bfloat16, CUDA)
        torch.mul(xin, wn0, out=xw0)
        sq0 = torch.zeros(H // 64, nseq, dtype=torch.float32, device=CUDA)
        sq0[0:1] = mk.rowstat(xin)
        resid = xin.clone()
        mk.decoder_entry(
            spec_p,
            tables,
            xw0,
            sq0,
            mk.slack_buffer((nseq, H), torch.bfloat16, CUDA),
            torch.empty(H // 64, nseq, dtype=torch.float32, device=CUDA),
            resid,
            mk.slack_buffer((nseq, I), torch.bfloat16, CUDA),
            table,
            pos,
            slots,
            sl,
            case.bt,
            read_slots,
        )(torch.cuda.current_stream())
        torch.cuda.synchronize()
        resid_ref[rows] = resid
        states_ref.append(
            [
                (
                    ly["conv_pool"][read_slots.long()].clone(),
                    ly["ssm_pool"][read_slots.long()].clone(),
                )
                for ly in layers_p
                if ly["kind"] == 0
            ]
        )

    # ---- the verify form on the framework layouts: taps live at rows acc-1.. of slots[b, 0]; the state at slots[b, acc-1]
    layers_v = to_vllm_layout(
        clone_state(layers0), tap_row0=torch.zeros(n_slots, dtype=torch.int64)
    )
    for lv in [
        q for q in layers_v if q["kind"] == 0
    ]:  # the conv slots' windows start as garbage
        for b in range(nseq):
            lv["conv_pool"][int(slots_tab[b, 0])] = torch.randn(
                CONV_ROWS, lv["conv_pool"].shape[2], device=CUDA
            ).to(torch.bfloat16)
    # (re)place the committed taps of the conv slot at rows acc-1..acc+1, from the original pools
    for lv, l0 in zip(
        [q for q in layers_v if q["kind"] == 0],
        [q for q in layers0 if q["kind"] == 0],
        strict=True,
    ):
        for b in range(nseq):
            s0, a_ = int(slots_tab[b, 0]), int(acc[b])
            lv["conv_pool"][s0, a_ - 1 : a_ + 2] = l0["conv_pool"][
                int(read_slots[b])
            ].t()
    spec_v = mk.DecoderSpec(
        HK, HV, HQ, HKV, D, ROT, PAGE, H, I, splits=2, spec_rows=R, **VLLM
    )
    resid_v, _ = run_kernel_vllm(mk, spec_v, layers_v, case, x, wn0, slots_tab, acc)
    band("vllm verify resid", resid_v, resid_ref, 0.02, 0.002)
    gdn_v = [q for q in layers_v if q["kind"] == 0]
    for li, lv in enumerate(gdn_v):
        for b in range(nseq):
            for r in range(R):
                got = lv["ssm_pool"][int(slots_tab[b, r])].transpose(1, 2)
                band(
                    f"S after row {r} layer {li} seq {b}",
                    got,
                    states_ref[r][li][1][b],
                    0.02,
                    0.002,
                )
            # conv window of slots[b, 0]: rows 0, 1 = the two newest committed inputs, rows 2.. = this step's raw inputs
            taps_ref = states_ref[R - 1][li][0][
                b
            ]  # [W, 3] after R tokens = [x_{R-3}, x_{R-2}, x_{R-1}]
            win = lv["conv_pool"][int(slots_tab[b, 0])]  # [rows, W]
            torch.testing.assert_close(
                win[R - 1 : R + 2].t(), taps_ref, rtol=2e-2, atol=1e-3
            )
            l0c = [q for q in layers0 if q["kind"] == 0][li]["conv_pool"][
                int(read_slots[b])
            ]  # committed taps [W, 3] = h1 h2 h3
            torch.testing.assert_close(win[0:2].t(), l0c[:, 1:3], rtol=0, atol=0)
    for lv, lp in zip(
        [q for q in layers_v if q["kind"] == 1],
        [q for q in layers_p if q["kind"] == 1],
        strict=True,
    ):
        for b, c in enumerate(case.committed):
            for r in range(R):
                sidx = slot_of(case.bt, b, c + r)
                torch.testing.assert_close(
                    lv["k_cache"][sidx // PAGE, sidx % PAGE],
                    lp["k_cache"][sidx // PAGE, sidx % PAGE],
                    rtol=1e-2,
                    atol=1e-2,
                )
                torch.testing.assert_close(
                    lv["v_cache"][sidx // PAGE, sidx % PAGE],
                    lp["v_cache"][sidx // PAGE, sidx % PAGE],
                    rtol=1e-2,
                    atol=1e-2,
                )
