# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run the experimental Qwen3.8 decoder megakernel on synthetic layers.
#
# Requires SM120 (RTX PRO 6000 / GeForce Blackwell). Builds ``--layers`` layers
# in the kernel's dict form (kinds 0 = gated DeltaNet, 1 = full attention) with
# random fp8 / NVFP4 weights at the 27B checkpoint's geometry (or a small one
# with ``--small``), prepares a plan for ``--batch`` decode rows, runs the
# decode step eagerly, captures the same step in a CUDA graph and times the
# replay.

import argparse
import time

import torch

from flashinfer.experimental.qwen38_megakernel import DecoderSpec, interleave_gate_up
from flashinfer.qwen3_ops import (
    qwen38_megakernel_decode,
    qwen38_megakernel_prepare,
    qwen38_megakernel_unsupported_reason,
)


def fp4(n, k, dev):
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
    s = (torch.rand(n, k // 16, device=dev) * 2 + 0.25).to(torch.float8_e4m3fn)
    return w, s


def make_layer(kind, geo, n_slots, n_pages, dev):
    H, I, HK, HV, HQ, HKV, D = (
        geo["H"],
        geo["I"],
        geo["hk"],
        geo["hv"],
        geo["hq"],
        geo["hkv"],
        geo["d"],
    )
    bf = torch.bfloat16
    ly = {"kind": kind}
    wgu, sgu = fp4(2 * I, H, dev)
    ly["w_gu_il"], ly["s_gu_il"] = interleave_gate_up(wgu, sgu, I)
    ly["w_dn"], ly["s_dn"] = fp4(H, I, dev)
    ly["alpha_gu"] = torch.tensor([0.5, 0.125], device=dev)
    ly["alpha_dn"] = torch.tensor([0.02], device=dev)
    ly["wn_mid"] = (torch.rand(H, device=dev) + 0.5).to(bf)
    ly["wn_out"] = (torch.rand(H, device=dev) + 0.5).to(bf)
    ly["alpha_out"] = torch.tensor([0.01], device=dev)
    if kind == 0:
        kd, vd = HK * 128, HV * 128
        ly["w_in"] = (torch.randn(2 * kd + 2 * vd, H, device=dev) * 0.05).to(
            torch.float8_e4m3fn
        )
        ly["alpha_in"] = torch.tensor([0.02, 0.03], device=dev)
        ly["w_out"] = (torch.randn(H, vd, device=dev) * 0.05).to(torch.float8_e4m3fn)
        ly["w_ba"] = (torch.randn(2 * HV, H, device=dev) * 0.02).to(bf)
        ly["conv_w"] = (torch.randn(2 * kd + vd, 4, device=dev) * 0.3).to(bf)
        ly["a_log"] = torch.randn(HV, device=dev) * 0.5
        ly["dt_bias"] = torch.randn(HV, device=dev) * 0.5
        ly["norm_w"] = (torch.rand(128, device=dev) + 0.5).to(bf)
        ly["conv_pool"] = torch.zeros(n_slots, 2 * kd + vd, 3, dtype=bf, device=dev)
        ly["ssm_pool"] = torch.zeros(
            n_slots, HV, 128, 128, dtype=torch.float32, device=dev
        )
    else:
        qd, kvd = HQ * D, HKV * D
        ly["w_in"] = (torch.randn(2 * qd + 2 * kvd, H, device=dev) * 0.05).to(
            torch.float8_e4m3fn
        )
        ly["alpha_in"] = torch.tensor([0.02, 0.03, 0.025, 0.015], device=dev)
        ly["w_out"] = (torch.randn(H, qd, device=dev) * 0.05).to(torch.float8_e4m3fn)
        ly["wq"] = (torch.rand(D, device=dev) + 0.5).to(bf)
        ly["wk"] = (torch.rand(D, device=dev) + 0.5).to(bf)
        ly["k_cache"] = torch.randn(n_pages, 32, HKV, D, device=dev).to(bf)
        ly["v_cache"] = torch.randn(n_pages, 32, HKV, D, device=dev).to(bf)
    return ly


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--layers",
        type=int,
        default=4,
        help="layer count (kinds follow Qwen3.5's 3 GDN : 1 attention pattern)",
    )
    p.add_argument(
        "--batch", type=int, default=1, help="decode rows (sequences) per step, 1..16"
    )
    p.add_argument(
        "--context", type=int, default=2048, help="context length per sequence"
    )
    p.add_argument(
        "--small",
        action="store_true",
        help="a 2048-wide geometry instead of the 27B one",
    )
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()
    dev = torch.device("cuda")
    geo = (
        dict(H=2048, I=2048, hk=2, hv=8, hq=6, hkv=2, d=256, rot=64)
        if args.small
        else dict(H=5120, I=17408, hk=16, hv=48, hq=24, hkv=4, d=256, rot=64)
    )
    PAGE = 32
    reason = qwen38_megakernel_unsupported_reason(
        dev,
        hidden=geo["H"],
        inter=geo["I"],
        hk=geo["hk"],
        hv=geo["hv"],
        dk=128,
        dv=128,
        hq=geo["hq"],
        hkv=geo["hkv"],
        head_dim=geo["d"],
        rot_dim=geo["rot"],
        page_size=PAGE,
        num_tokens=args.batch,
    )
    if reason is not None:
        raise SystemExit(f"unsupported here: {reason}")
    B, L = args.batch, args.context
    pages_per_seq = (L + PAGE - 1) // PAGE
    kinds = [1 if (j % 4 == 3) else 0 for j in range(args.layers)]
    layers = [make_layer(k, geo, B, B * pages_per_seq + 1, dev) for k in kinds]
    spec = DecoderSpec(
        geo["hk"],
        geo["hv"],
        geo["hq"],
        geo["hkv"],
        geo["d"],
        geo["rot"],
        PAGE,
        geo["H"],
        geo["I"],
        splits=2,
    )
    plan = qwen38_megakernel_prepare(spec, layers, B, max_seq_len=L)

    block_table = (
        torch.arange(B * pages_per_seq, dtype=torch.int32, device=dev) + 1
    ).view(B, pages_per_seq)
    seq_lens = torch.full((B,), L, dtype=torch.int32, device=dev)
    positions = seq_lens - 1
    slot_mapping = block_table[:, (L - 1) // PAGE] * PAGE + (L - 1) % PAGE
    state_slots = torch.arange(B, dtype=torch.int32, device=dev)
    hidden = torch.randn(B, geo["H"], device=dev).to(torch.bfloat16)

    out = qwen38_megakernel_decode(
        plan, hidden, positions, slot_mapping, seq_lens, block_table, state_slots
    )
    torch.cuda.synchronize()
    print(
        f"eager: out {tuple(out.shape)} {out.dtype}, finite={bool(torch.isfinite(out).all())}"
    )

    out_buf = torch.empty_like(out)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        qwen38_megakernel_decode(
            plan,
            hidden,
            positions,
            slot_mapping,
            seq_lens,
            block_table,
            state_slots,
            out=out_buf,
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            qwen38_megakernel_decode(
                plan,
                hidden,
                positions,
                slot_mapping,
                seq_lens,
                block_table,
                state_slots,
                out=out_buf,
            )
        graph.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            graph.replay()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.iters
    weight_bytes = sum(
        t.numel() * t.element_size()
        for ly in layers
        for k, t in ly.items()
        if isinstance(t, torch.Tensor)
        and k not in ("k_cache", "v_cache", "conv_pool", "ssm_pool")
    )
    print(
        f"graph replay: {dt * 1e3:.3f} ms per step for {args.layers} layers x {B} rows "
        f"({weight_bytes / dt / 1e12:.2f} TB/s of weights alone)"
    )


if __name__ == "__main__":
    main()
