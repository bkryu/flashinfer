# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# Ported from LightLM's cutlass_dsl megakernel study (docs/megakernel-plan.md there) for the FlashInfer experimental
# track: the Qwen3.8-27B one-launch decoder on SM120.
"""Host for the single-launch decoder kernel (decoder_mega_sm120, §13).

`DecoderSpec` holds the per-model constexpr set; `build_tables(spec, layers, M, kv_splits)` turns a
list of per-layer dicts (see LAYER_KEYS) into the three device tables the kernel reads (Int64
pointers, Int32 counts / kind / bounds, fp32 alphas) plus the counter-region stride; `decoder_entry`
launches ONE kernel for all layers. The activation contract is the layer kernels' one, folded:
xw0 / sq0 = the first layer's absorbed-norm input (x * w_norm0, row sum of squares in row 0 of sq0
with the other rows zero) and every layer's output; xw1 / sq1 the mid scratch; resid updated in
place; pools / caches updated in place."""

from __future__ import annotations

import torch

from . import _runtime as _rt
from ._runtime import from_dlpack
from ._host_utils import _staging
from ._host_utils import _kv_splits
from .decoder_mega_sm120 import (
    IT_B1,
    IT_B2,
    IT_B3,
    IT_CROWS,
    IT_CSS,
    IT_HDN,
    IT_K2,
    IT_KCC,
    IT_KIND,
    IT_KT0,
    IT_N1,
    IT_SGO,
    IT_SSS,
    IT_TA,
    IT_TB,
    IT_TB0,
    IT_TB0I,
    IT_TBKV,
    IT_W,
    NF,
    NI,
    NP,
    NTMAP,
    PT_ALOG,
    PT_CONV,
    PT_CONVW,
    PT_DTB,
    PT_KC,
    PT_NORMW,
    PT_SDN,
    PT_SGU,
    PT_SSM,
    PT_STA,
    PT_STB,
    PT_STX,
    PT_STY,
    PT_VC,
    PT_WBA,
    PT_WDN,
    PT_WGU,
    PT_WIN,
    PT_WK,
    PT_WNMID,
    PT_WNOUT,
    PT_WOUT,
    PT_WQ,
)
from ._host_utils import MAX_M, TILE, _qk_scratch
from ._host_utils import _cute, _fp8_cute
from ._host_utils import _dummy_c, _workspace
from .decoder_mega_sm120 import (
    FOLD_CNT,
    IT_TF,
    PT_WFC,
    PT_WNIN,
    PT_WPRE,
)  # §19.7 drafter fold

_KERNELS: dict = {}
_CNT: dict = {}
_TMAPS: dict = {}
_SCRATCH: dict = {}
_PROF: dict = {}
PROF_SLOTS = 4096
PHASES = {
    0: "in_proj",
    1: "core_b0",
    2: "core_b",
    3: "out_proj",
    4: "gate_up",
    5: "down",
    6: "wait_layer",
    7: "wait_gdone",
    13: "pwait_heads",
    14: "pwait_gact",
    16: "pwait_layer",
    17: "pwait_gdone",
    20: "g.flags",
    21: "g.conv_ba",
    22: "g.state_ld",
    23: "g.pass1",
    24: "g.pass2_st",
    25: "g.zwait",
    26: "g.norm_st",
    28: "a.kvw_q",
    29: "a.pages",
    30: "a.stage",
    31: "a.merge",
    19: "mlp.prologue",
    27: "fp8.prologue",
    8: "mlp.wait",
    9: "mlp.compute",
    10: "fc",
    11: "fnorm",
}


def profile_buffer(device, n_ctas):
    d = torch.device(device)
    key = (
        d.type,
        d.index if d.index is not None else torch.cuda.current_device(),
        n_ctas,
    )  # 'cuda' and 'cuda:0' are one buffer
    t = _PROF.get(key)
    if t is None:
        t = _PROF[key] = (
            torch.zeros(n_ctas * 2 * PROF_SLOTS * 4, dtype=torch.int64, device=device),
            torch.zeros(n_ctas * 2, dtype=torch.int32, device=device),
        )
    return t


def decode_profile(buf, n_ctas):
    """-> list of (cta, role, layer, phase, t0_ns, t1_ns, work) from a profile buffer (records with t1 == 0 skipped)."""
    a = buf[0].view(n_ctas, 2, PROF_SLOTS, 4).cpu()
    cnt = buf[1].view(n_ctas, 2).cpu()
    out = []
    for c in range(n_ctas):
        for r in range(2):
            rows = a[c, r]
            n = min(int(cnt[c, r]), PROF_SLOTS)
            for k in range(n):
                code, t0, t1, w = (int(v) for v in rows[k])
                out.append((c, r, code // 32, code % 32, t0, t1, w))
    return out


LAYER_KEYS_COMMON = (
    "kind",
    "w_in",
    "alpha_in",
    "w_out",
    "alpha_out",
    "w_gu_il",
    "s_gu_il",
    "alpha_gu",
    "w_dn",
    "s_dn",
    "alpha_dn",
    "wn_mid",
    "wn_out",
)
LAYER_KEYS_GDN = (
    "conv_w",
    "conv_pool",
    "ssm_pool",
    "w_ba",
    "a_log",
    "dt_bias",
    "norm_w",
)
LAYER_KEYS_ATTN = ("wq", "wk", "k_cache", "v_cache")


class DecoderSpec:
    def __init__(
        self,
        hk,
        hv,
        hq,
        hkv,
        d,
        rot,
        page,
        H,
        I,
        splits=2,
        eps=1e-6,
        prefetch=True,
        l2_prefetch=False,
        xb_prefetch=True,
        profile=False,
        mlp_stages=None,
        mma_attn=True,
        dual_acc=True,
        kring3=False,
        tail_split=None,
        tile_n=None,
        spec_rows=1,
        wtype="fp8fp4",
        w16_stages=None,
        drafter_fold=False,
        state_vk=False,
        conv_sd=False,
        kv_packed=False,
        slot_table=False,
        gate_up_split=False,
        q_gate_il=False,
    ):
        self.hk, self.hv, self.hq, self.hkv, self.d, self.rot, self.page = (
            hk,
            hv,
            hq,
            hkv,
            d,
            rot,
            page,
        )
        # ---- framework (vLLM) layouts, see `DecoderTables` for the tensor forms each one expects
        self.state_vk = bool(
            state_vk
        )  # ssm_pool [slots, hv, dv, dk] (dk innermost), any slot stride
        self.conv_sd = bool(
            conv_sd
        )  # conv_pool [slots, rows, C] (rows >= 3, row stride C), any slot stride
        self.kv_packed = bool(
            kv_packed
        )  # k_cache / v_cache = the two halves of one [pages, page, hkv, 2d] tensor
        self.slot_table = bool(
            slot_table
        )  # state_slots [seqs, >= R] + num_accepted: per-token slots (vLLM spec-decode convention)
        # w_gu_il / s_gu_il hold the stored [gate (I) | up (I)] rows (no 8-row interleave): two TMA boxes per stage
        self.gate_up_split = bool(gate_up_split)
        # attention w_in rows [q_h | gate_h] per head, then k | v (vLLM's fused QKV with the output gate); one in-proj alpha
        self.q_gate_il = bool(q_gate_il)
        if self.gate_up_split and wtype == "bf16":
            raise ValueError(
                "decoder: gate_up_split is implemented for the fp8 / NVFP4 kind"
            )
        if wtype not in ("fp8fp4", "bf16"):
            raise ValueError(
                f"decoder: wtype {wtype!r} (fp8fp4 = the 27B recipe; bf16 = unquantized weights, the MTP drafter)"
            )
        self.wtype = wtype  # §19: "bf16" = every GEMM on unquantized bf16 weights [N, K] (one fp8-ring kind; alphas unused)
        # §19: ring depth of the bf16 kind; None = 4 with TILE_N 8 (the fp4 ring's smem re-used), 3 with TILE_N 16 (smem budget)
        self.w16_stages = None if w16_stages is None else int(w16_stages)
        # §19.7: fc + the two pre-fc norms as a split-K GEMM phase in front of layer 0 and the final norm behind the last layer
        # (bf16 kind, 2 splits; layer 0 carries w_fc / w_pre / wn_in; decoder_entry takes xcat = cat(e | h) and out)
        self.drafter_fold = bool(drafter_fold)
        if self.drafter_fold and (wtype != "bf16" or splits != 2):
            raise ValueError("decoder: drafter_fold needs wtype='bf16' and splits=2")
        self.mma_attn = bool(
            mma_attn
        )  # §16 item 3b: tensor-core attention page loop (False = the FMA loop, bitwise with the layer kernels)
        self.dual_acc = bool(
            dual_acc
        )  # §17 item 1: two accumulator chains in the NVFP4 mainloop (False = the bitwise-chained order)
        self.kring3 = bool(
            kring3
        )  # §17 item 5: three K page buffers in the attention core (False = two; same arithmetic)
        self.tail_split = (
            None if tail_split is None else bool(tail_split)
        )  # §17 item 2: None = on for M >= 8 (False = bitwise-chained order)
        self.H, self.I, self.S, self.eps, self.prefetch = H, I, splits, eps, prefetch
        self.l2_prefetch = bool(l2_prefetch)
        self.xb_prefetch = bool(xb_prefetch)
        self.profile = bool(
            profile
        )  # §15: timeline records (see profile_buffer / decode_profile)
        self.mlp_stages = (
            None if mlp_stages is None else int(mlp_stages)
        )  # None: 6 with TILE_N 8, 4 with 16 (§17 item 1b)
        self.tile_n = (
            None if tile_n is None else int(tile_n)
        )  # None: 8 if M <= 8 else 16
        self.spec_rows = int(
            spec_rows
        )  # §18: rows per sequence (1 = plain decode; R = the MTP verify step, pools read-only)

    @property
    def n1_gdn(self):
        return 2 * self.hk * 128 + 2 * self.hv * 128

    @property
    def n1_attn(self):
        return 2 * self.hq * self.d + 2 * self.hkv * self.d


class DecoderTables:
    """Device tables for one (layer list, M, kv_splits); keeps the weight tensors referenced."""

    def __init__(self, spec: DecoderSpec, layers: list, M: int, kv_splits: int):
        H, I, S = spec.H, spec.I, spec.S
        nl = len(layers)
        dev = layers[0]["w_in"].device
        kd, vd = spec.hk * 128, spec.hv * 128
        qd, kvd = spec.hq * spec.d, spec.hkv * spec.d
        nt2 = H // TILE[0]
        R = int(spec.spec_rows)
        if R < 1 or M % R:
            raise ValueError(f"decoder: M={M} is not a multiple of spec_rows={R}")
        self.stash = (
            R > 1 and not spec.slot_table
        )  # the study's verify form keeps per-row stash tensors in the GDN layers
        nseq = M // R
        w16 = spec.wtype == "bf16"
        KT = (
            128 if w16 else TILE[2]
        )  # K elements per GEMM tile (bf16: the 64x128 bf16 A tile; fp8: 256 K packed 2 per slot)
        it = torch.zeros(nl, NI, dtype=torch.int64)
        pt = torch.zeros(nl, NP, dtype=torch.int64)
        ft = torch.zeros(nl, NF, dtype=torch.float32)
        cs = 0
        if (2 * I // TILE[0]) % S or spec.hv % S:
            raise ValueError(
                "decoder: gate_up tile count and Hv must be multiples of the split count (per-split flags)"
            )
        for j, ly in enumerate(layers):
            kind = int(ly["kind"])
            n1 = ly["w_in"].shape[0]
            k2 = ly["w_out"].shape[1]
            if (
                ly["w_in"].shape[1] != H
                or ly["w_out"].shape[0] != H
                or ly["w_gu_il"].shape[0] != 2 * I
                or ly["w_dn"].shape != ((H, I) if w16 else (H, I // 2))
            ):
                raise ValueError(f"decoder: layer {j} weight shapes")
            if w16:
                for key in ("w_in", "w_out", "w_gu_il", "w_dn"):
                    if ly[key].dtype != torch.bfloat16:
                        raise ValueError(f"decoder: bf16 kind needs bf16 {key}")
                if ly["w_gu_il"].shape[1] != H:
                    raise ValueError(f"decoder: layer {j} gate_up shape")
            if k2 % (S * KT):
                raise ValueError(
                    f"decoder: layer {j} out-projection K {k2} must split into {S} x {KT} tiles"
                )
            ta = n1 // TILE[0]
            a_in = ly["alpha_in"].reshape(-1).float().tolist()
            if kind == 0:
                if n1 != spec.n1_gdn or k2 != vd:
                    raise ValueError(f"decoder: GDN layer {j} shapes N1={n1} K2={k2}")
                tb0, tb, hdn, kt0, w = (
                    nseq * spec.hk,
                    nseq * spec.hv,
                    nseq,
                    0,
                    2 * kd + vd,
                )  # items per (sequence, head); hdn = arrivals per head counter
                tbkv = tb0
                if self.stash:
                    for key, col in (
                        ("stash_x", PT_STX),
                        ("stash_y", PT_STY),
                        ("stash_a", PT_STA),
                        ("stash_b", PT_STB),
                    ):
                        t = ly[key]
                        if not t.is_contiguous() or t.shape[0] < M:
                            raise ValueError(
                                f"decoder: layer {j} {key} must be contiguous with >= M rows"
                            )
                        pt[j, col] = t.data_ptr()
                    if (
                        ly["stash_x"].shape[1] != w
                        or ly["stash_a"].shape[1] != spec.hv
                        or ly["stash_a"].dtype != torch.float32
                    ):
                        raise ValueError(f"decoder: layer {j} stash shapes / dtypes")
                b1, b2, b3 = w, n1, n1
                a_in = (a_in + [0.0, 0.0])[:4]
                n_cnt = (
                    ta + spec.hv + 1 + nt2 + tb0 + 32 + 32 + nt2 + 32 + 32 + ta + 16
                )  # ... + tail-half counters (§17 item 2)
                for key, col in (
                    ("w_ba", PT_WBA),
                    ("conv_w", PT_CONVW),
                    ("a_log", PT_ALOG),
                    ("dt_bias", PT_DTB),
                    ("norm_w", PT_NORMW),
                ):
                    t = ly[key]
                    if not t.is_contiguous():
                        raise ValueError(f"decoder: layer {j} {key} must be contiguous")
                    pt[j, col] = t.data_ptr()
                cp, sp_ = ly["conv_pool"], ly["ssm_pool"]
                pt[j, PT_CONV], pt[j, PT_SSM] = cp.data_ptr(), sp_.data_ptr()
                if spec.conv_sd:
                    # [slots, rows, C]: rows >= 3 (+ R - 1 for the per-token window), row stride C, any slot stride (padded pages)
                    need = 3 + (R - 1 if spec.slot_table else 0)
                    if (
                        cp.dim() != 3
                        or cp.shape[2] != w
                        or cp.shape[1] < need
                        or cp.stride(2) != 1
                        or cp.stride(1) != w
                    ):
                        raise ValueError(
                            f"decoder: layer {j} conv_pool must be [slots, rows >= {need}, {w}] with contiguous rows"
                        )
                    it[j, IT_CSS], it[j, IT_CROWS] = cp.stride(0), cp.shape[1]
                else:
                    if cp.shape[1:] != (w, 3) or not cp.is_contiguous():
                        raise ValueError(
                            f"decoder: layer {j} conv_pool must be contiguous [slots, {w}, 3]"
                        )
                    it[j, IT_CSS], it[j, IT_CROWS] = 3 * w, 3
                if (
                    sp_.dim() != 4
                    or sp_.shape[1:] != (spec.hv, 128, 128)
                    or sp_.stride(3) != 1
                    or sp_.stride(2) != 128
                    or sp_.stride(1) != 16384
                    or sp_.stride(0) < spec.hv * 16384
                ):
                    raise ValueError(
                        f"decoder: layer {j} ssm_pool must be [slots, {spec.hv}, 128, 128] with contiguous heads"
                    )
                if not spec.state_vk and not sp_.is_contiguous():
                    raise ValueError(
                        f"decoder: layer {j} ssm_pool must be contiguous (state_vk=False)"
                    )
                it[j, IT_SSS] = sp_.stride(0)
                it[j, IT_SGO] = int(
                    ly.get("slot_group", 0)
                )  # index into the slot table's group mode (3-D slots)
                if (
                    ly["a_log"].dtype != torch.float32
                    or ly["dt_bias"].dtype != torch.float32
                    or ly["ssm_pool"].dtype != torch.float32
                ):
                    raise ValueError("decoder: a_log / dt_bias / ssm_pool must be fp32")
                if ly["conv_w"].shape != (w, 4) or ly["w_ba"].shape != (2 * spec.hv, H):
                    raise ValueError(f"decoder: layer {j} conv_w / w_ba shapes")
            else:
                if n1 != spec.n1_attn or k2 != qd:
                    raise ValueError(
                        f"decoder: attention layer {j} shapes N1={n1} K2={k2}"
                    )
                tb0, tb, hdn, kt0, w = (
                    M * spec.hkv,
                    nseq * spec.hkv * kv_splits,
                    M,
                    (2 * qd) // 64,
                    0,
                )  # q-prep per token; splits per sequence (§18b); hdn per head counter
                tbkv = nseq * spec.hkv  # kv-write items per (sequence, kv head)
                b1, b2, b3 = qd, 2 * qd, 2 * qd + kvd
                if spec.q_gate_il and (len(a_in) < 2 or a_in[0] != a_in[1]):
                    raise ValueError(
                        f"decoder: layer {j} q_gate_il needs one alpha for the interleaved q | gate rows"
                    )
                n_cnt = (
                    ta
                    + spec.hkv
                    + 1
                    + nt2
                    + tbkv
                    + 2 * tb0
                    + 32
                    + 32
                    + nt2
                    + 32
                    + 32
                    + ta
                    + 16
                )  # kv-write, merge, q-prep flags; tail-half counters
                for key, col in (("wq", PT_WQ), ("wk", PT_WK)):
                    t = ly[key]
                    if not t.is_contiguous():
                        raise ValueError(f"decoder: layer {j} {key} must be contiguous")
                    pt[j, col] = t.data_ptr()
                kc, vc = ly["k_cache"], ly["v_cache"]
                pt[j, PT_KC], pt[j, PT_VC] = kc.data_ptr(), vc.data_ptr()
                mult = 2 if spec.kv_packed else 1
                for name, t in (("k_cache", kc), ("v_cache", vc)):
                    if (
                        t.dim() != 4
                        or t.shape[1:] != (spec.page, spec.hkv, spec.d)
                        or t.dtype != torch.bfloat16
                        or t.stride(3) != 1
                        or t.stride(2) != mult * spec.d
                        or t.stride(1) != mult * spec.d * spec.hkv
                        or t.stride(0) != mult * spec.d * spec.hkv * spec.page
                    ):
                        raise ValueError(
                            f"decoder: {name} must be bf16 [pages, {spec.page}, {spec.hkv}, {spec.d}]"
                            + (
                                " as one half of a packed [pages, page, hkv, 2d] tensor"
                                if spec.kv_packed
                                else " contiguous"
                            )
                        )
                if spec.kv_packed and vc.data_ptr() - kc.data_ptr() not in (
                    2 * spec.d,
                    -2 * spec.d,
                ):
                    raise ValueError(
                        "decoder: kv_packed needs k_cache / v_cache to be the two halves of one tensor"
                    )
            for key, col in (
                ("w_in", PT_WIN),
                ("w_out", PT_WOUT),
                ("w_gu_il", PT_WGU),
                ("s_gu_il", PT_SGU),
                ("w_dn", PT_WDN),
                ("s_dn", PT_SDN),
                ("wn_mid", PT_WNMID),
                ("wn_out", PT_WNOUT),
            ):
                t = ly[key]
                if not t.is_contiguous():
                    raise ValueError(f"decoder: layer {j} {key} must be contiguous")
                pt[j, col] = t.data_ptr()
            tf = 0
            if spec.drafter_fold:
                if j != 0:
                    raise ValueError(
                        "decoder: drafter_fold is a one-layer (drafter) form"
                    )
                for key, col, shp in (
                    ("w_fc", PT_WFC, (H, 2 * H)),
                    ("w_pre", PT_WPRE, (2 * H,)),
                    ("wn_in", PT_WNIN, (H,)),
                ):
                    t = ly[key]
                    if (
                        tuple(t.shape) != shp
                        or t.dtype != torch.bfloat16
                        or not t.is_contiguous()
                    ):
                        raise ValueError(
                            f"decoder: fold tensor {key} must be contiguous bf16 {shp}"
                        )
                    pt[j, col] = t.data_ptr()
                tf = nt2 * S  # fc items: (tile, K half)
            vals = {
                IT_KIND: kind,
                IT_N1: n1,
                IT_K2: k2,
                IT_TA: ta,
                IT_TB0: tb0,
                IT_TB: tb,
                IT_KCC: k2 // S // KT,
                IT_B1: b1,
                IT_B2: b2,
                IT_B3: b3,
                IT_KT0: kt0,
                IT_HDN: hdn,
                IT_W: w,
                IT_TB0I: tb0 if kind == 0 else tb0 + tbkv,
                IT_TBKV: tbkv,
                IT_TF: tf,
            }
            for c, v in vals.items():
                it[j, c] = int(v)
            ft[j, :4] = torch.tensor(a_in)
            ft[j, 4] = float(ly["alpha_out"].reshape(-1)[0])
            ft[j, 5] = float(ly["alpha_gu"].reshape(-1)[0])
            ft[j, 6] = float(ly["alpha_gu"].reshape(-1)[1])
            ft[j, 7] = float(ly["alpha_dn"].reshape(-1)[0])
            cs = max(cs, n_cnt)
        self.cs = (
            (cs + nt2 + FOLD_CNT + 31) // 32 * 32
        )  # + the fold's counters (top of every layer region)
        self.nl = nl
        self.M = M
        self.kv_splits = kv_splits
        self.itab = it.to(torch.int32).to(dev)
        self.ptab = pt.to(dev)
        self.ftab = ft.to(dev)
        self.n1_max = max(int(ly["w_in"].shape[0]) for ly in layers)
        self.k2_max = max(int(ly["w_out"].shape[1]) for ly in layers)
        self.i_a1 = max(range(nl), key=lambda j: layers[j]["w_in"].shape[0])
        self.i_a2 = max(range(nl), key=lambda j: layers[j]["w_out"].shape[1])
        self.layers = layers  # keep the weights alive with the pointer table


def _counters(device, nl, cs):
    key = (str(device), nl, cs)
    t = _CNT.get(key)
    if t is None:
        t = _CNT[key] = torch.zeros(nl * cs + 32, dtype=torch.int32, device=device)
    return t


def _tmaps(device, n_ctas):
    key = (str(device), n_ctas)
    t = _TMAPS.get(key)
    if t is None:
        t = _TMAPS[key] = torch.zeros(
            n_ctas, NTMAP, 16, dtype=torch.int64, device=device
        )
    return t


def _scratch(device, dtype, M, n1_max, k2_max):
    key = (str(device), dtype, M, n1_max, k2_max)
    t = _SCRATCH.get(key)
    if t is None:
        # +128 elements of slack: the bf16 kind's 256-wide activation boxes over 128-K tiles read one tile past the last row
        t = _SCRATCH[key] = (
            torch.empty(M * n1_max + 128, dtype=dtype, device=device)[: M * n1_max],
            torch.empty(M * k2_max + 128, dtype=dtype, device=device)[: M * k2_max],
        )
    return t


def _has_slack(t, n_elems):
    """True when >= n_elems elements of the same dtype are readable past the end of contiguous tensor t."""
    es = t.element_size()
    return (
        t.untyped_storage().nbytes() - (t.storage_offset() + t.numel()) * es
    ) >= n_elems * es


def slack_buffer(shape, dtype, device, slack=128):
    """A contiguous tensor of `shape` followed by `slack` readable elements (the bf16 kind's activation buffers)."""
    n = 1
    for d_ in shape:
        n *= int(d_)
    return torch.empty(n + slack, dtype=dtype, device=device)[:n].view(*shape)


_W16_DUMMY: dict = {}


def _w16_scale_dummy(device, n):
    key = (str(device), n)
    t = _W16_DUMMY.get(key)
    if t is None:
        t = _W16_DUMMY[key] = torch.zeros(n, dtype=torch.uint8, device=device)
    return t


_ONES: dict = {}


def _ones_i32(device, n):
    key = (str(device), n)
    t = _ONES.get(key)
    if t is None:
        t = _ONES[key] = torch.ones(n, dtype=torch.int32, device=device)
    return t


def _i32(t):
    return from_dlpack(
        t.to(torch.int32).contiguous(), assumed_align=4
    ).mark_layout_dynamic(leading_dim=0)


def decoder_entry(
    spec: DecoderSpec,
    tables: DecoderTables,
    xw0,
    sq0,
    xw1,
    sq1,
    resid,
    act_buf,
    rope_table,
    positions,
    slot_mapping,
    seq_lens,
    block_table,
    slots,
    *,
    max_ctas=None,
    xcat=None,
    out=None,
    num_accepted=None,
):
    """ONE launch for tables.nl layers. xw0 [M,H] bf16 / sq0 [H/64,M] fp32 = first-layer input (and every
    layer's output), xw1 / sq1 mid scratch, resid [M,H] bf16 in place, act_buf [M,I] bf16 scratch.
    §19.7 drafter fold: xcat [M, 2H] bf16 = cat(embeds | hidden) (slack buffer; the kernel produces xw0 / sq0 / resid
    itself) and out [M, H] bf16 = the final-norm output.
    slot_table form: slots int32 [seqs, >= R] (per-token GDN state slots; column 0 also holds the conv state) and
    num_accepted int32 [seqs] (the previous step's accepted count, 1..R: state read from slots[:, acc - 1])."""
    from . import _runtime as pdl_mod
    from . import _runtime as pg
    from .decoder_mega_sm120 import DecoderMegaSm120

    M, H = xw0.shape
    I, S = spec.I, spec.S
    dev, dt = xw0.device, xw0.dtype
    if M != tables.M or M > MAX_M or H != spec.H:
        raise ValueError("decoder_entry: tables built for another M / H")
    for t_, shp in ((xw1, (M, H)), (resid, (M, H)), (act_buf, (M, I))):
        if t_.shape != shp or not t_.is_contiguous():
            raise ValueError(
                f"decoder_entry: bad buffer shape {tuple(t_.shape)} != {shp}"
            )
    nt2 = H // TILE[0]
    for t_ in (sq0, sq1):
        if t_.shape != (nt2, M) or t_.dtype != torch.float32:
            raise ValueError("decoder_entry: sq0 / sq1 must be fp32 [H/64, M]")
    mac = int(max_ctas) if max_ctas else pg._max_active_clusters()
    if spec.drafter_fold:
        if (
            xcat is None
            or out is None
            or tuple(xcat.shape) != (M, 2 * H)
            or tuple(out.shape) != (M, H)
            or not xcat.is_contiguous()
            or not out.is_contiguous()
            or not _has_slack(xcat, 128)
        ):
            raise ValueError(
                "decoder fold: xcat [M, 2H] bf16 with 128 elements of slack and out [M, H] are required"
            )
        if nt2 * S > mac:
            raise ValueError(
                f"decoder fold: the final norm needs at most one down item per CTA ({nt2 * S} items > {mac} CTAs)"
            )
    KS = tables.kv_splits
    G = spec.hq // spec.hkv
    D = spec.d
    ly_a1 = tables.layers[tables.i_a1]
    ly_a2 = tables.layers[tables.i_a2]
    ly0 = tables.layers[0]
    n1_max, k2_max = tables.n1_max, tables.k2_max
    mz_f, attn_f = _scratch(dev, dt, M, n1_max, k2_max)
    w16 = spec.wtype == "bf16"
    # ---- TMA templates (layouts only; the kernel rewrites base / shape / strides per layer)
    c1 = _cute(_dummy_c(M, n1_max, dt, dev).view(1, M, n1_max).permute(2, 1, 0), 0)
    ws_t = _cute(_workspace(dev, S, M, H).permute(2, 1, 0), 0)
    wsa_t = _cute(
        _workspace(dev, 2, M, n1_max).permute(2, 1, 0), 0
    )  # in-proj tail half-partials (N1_max, M, 2)
    c3 = _cute(_dummy_c(M, 2 * I, dt, dev).view(1, M, 2 * I).permute(2, 1, 0), 0)
    if spec.drafter_fold:
        fa = _cute(ly0["w_fc"].view(H, S, 2 * H // S).permute(0, 2, 1), 1)
        fb = _cute(torch.as_strided(xcat, (M, 256, 2 * H // 128), (2 * H, 1, 128)), 1)
        out_t = _cute(out.view(1, M, H).permute(2, 1, 0), 0)
    if w16:
        # §19 bf16 kind: weights [N, K] bf16 addressed at their real K; activations through "doubled" views
        # (n_tok, 256, K/128) with stride 128 on the tile mode so the 256-wide B box / smem layout of the fp8 ring
        # is reused (the second 128 columns of a box are the next tile's, never consumed) -> the buffers need 128
        # elements of readable slack after the last row (slack_buffer)
        for name, t_ in (("xw0", xw0), ("xw1", xw1), ("act_buf", act_buf)):
            if not _has_slack(t_, 128):
                raise ValueError(
                    f"decoder bf16 kind: {name} needs 128 elements of slack after its last row (use slack_buffer)"
                )
        a1 = _cute(ly_a1["w_in"].view(1, n1_max, H).permute(1, 2, 0), 1)
        b1 = _cute(torch.as_strided(xw0, (M, 256, H // 128), (H, 1, 128)), 1)
        a2 = _cute(ly_a2["w_out"].view(H, S, k2_max // S).permute(0, 2, 1), 1)
        b2 = _cute(
            torch.as_strided(attn_f, (M, 256, k2_max // 128), (k2_max, 1, 128)), 1
        )
        a3 = _cute(ly0["w_gu_il"].view(1, 2 * I, H).permute(1, 2, 0), 1)
        b3 = _cute(torch.as_strided(xw1, (M, 256, H // 128), (H, 1, 128)), 1)
        a4 = _cute(ly0["w_dn"].view(H, S, I // S).permute(0, 2, 1), 1)
        b4 = _cute(torch.as_strided(act_buf, (M, 256, I // 128), (I, 1, 128)), 1)
        # scale descriptors exist in the kind (rewritten per layer, never read): small dummies of the fp4 template shapes
        s3 = _fp8_cute(
            _w16_scale_dummy(dev, 64 * (H // 16)).view(1, 64, H // 16).permute(1, 2, 0),
            1,
        )
        s4 = _fp8_cute(
            _w16_scale_dummy(dev, 64 * (I // 16))
            .view(64, S, I // 16 // S)
            .permute(0, 2, 1),
            1,
        )
    else:
        a1 = _cute(
            ly_a1["w_in"].view(torch.bfloat16).view(1, n1_max, H // 2).permute(1, 2, 0),
            1,
        )
        b1 = _cute(xw0.view(1, M, H).permute(1, 2, 0), 1)
        a2 = _cute(
            ly_a2["w_out"]
            .view(torch.bfloat16)
            .view(H, S, k2_max // 2 // S)
            .permute(0, 2, 1),
            1,
        )
        b2 = _cute(attn_f.view(M, S, k2_max // S).permute(0, 2, 1), 1)
        a3 = _cute(
            ly0["w_gu_il"].view(torch.bfloat16).view(1, 2 * I, H // 4).permute(1, 2, 0),
            1,
        )
        s3 = _fp8_cute(
            ly0["s_gu_il"].view(torch.uint8).view(1, 2 * I, H // 16).permute(1, 2, 0), 1
        )
        b3 = _cute(xw1.view(1, M, H).permute(1, 2, 0), 1)
        a4 = _cute(
            ly0["w_dn"].view(torch.bfloat16).view(H, S, I // 4 // S).permute(0, 2, 1), 1
        )
        s4 = _fp8_cute(
            ly0["s_dn"].view(torch.uint8).view(H, S, I // 16 // S).permute(0, 2, 1), 1
        )
        b4 = _cute(act_buf.view(M, S, I // S).permute(0, 2, 1), 1)
    # ---- shared activations / scratch
    xw0_t = _cute(xw0.view(1, M, H).permute(2, 1, 0), 0)
    xw1_t = _cute(xw1.view(1, M, H).permute(2, 1, 0), 0)
    sq0_t = from_dlpack(sq0, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    sq1_t = from_dlpack(sq1, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mz_t = from_dlpack(mz_f, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    attn_t = from_dlpack(attn_f, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    res_t = _cute(resid.view(1, M, H).permute(2, 1, 0), 0)
    if not spec.drafter_fold:
        fa, fb, out_t = a1, b1, res_t  # unused by the kind: keep one kernel signature
    act_t = _cute(act_buf.view(1, M, I).permute(2, 1, 0), 0)
    qk_t = from_dlpack(
        _qk_scratch(dev, M * spec.hk * 256), assumed_align=16
    ).mark_layout_dynamic(leading_dim=0)
    po, pl = _staging(dev, M * spec.hkv * KS * G * D, M * spec.hkv * KS * G)
    qb, _ = _staging(dev, M * spec.hkv * G * D, 1)
    po_t = from_dlpack(po, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    pl_t = from_dlpack(pl, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    qb_t = from_dlpack(qb, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    rope_t = from_dlpack(rope_table.contiguous(), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    pos_t, slot_t, seq_t, bt_t = (
        _i32(positions),
        _i32(slot_mapping),
        _i32(seq_lens),
        _i32(block_table.reshape(-1)),
    )
    bt_stride = int(block_table.shape[1])
    nseq = M // int(spec.spec_rows)
    if spec.slot_table:
        if (
            slots.dim() not in (2, 3)
            or slots.shape[-2] < nseq
            or slots.shape[-1] < int(spec.spec_rows)
            or slots.stride(-1) != 1
        ):
            raise ValueError(
                f"decoder slot_table: slots must be int32 [(groups,) >= {nseq}, >= {spec.spec_rows}] with contiguous rows"
            )
        if num_accepted is None or num_accepted.shape[0] < nseq:
            raise ValueError(
                f"decoder slot_table: num_accepted int32 [>= {nseq}] is required"
            )
        if (
            slots.dim() == 3
        ):  # [groups, seqs, R]: GDN layers pick their group through `slot_group`
            slots_gstride, slots_stride = int(slots.stride(0)), int(slots.stride(1))
        else:
            slots_gstride, slots_stride = 0, int(slots.stride(0))
        slots_t = _i32(
            slots.reshape(-1)
            if slots.is_contiguous()
            else slots.contiguous().reshape(-1)
        )
        acc_t = _i32(num_accepted)
    else:
        if slots.dim() != 1:
            raise ValueError(
                "decoder: slots must be int32 [seqs] (one state slot per sequence) unless slot_table=True"
            )
        slots_t, slots_stride, slots_gstride = _i32(slots), 1, 0
        acc_t = _i32(_ones_i32(dev, nseq))
    cnt = _counters(dev, tables.nl, tables.cs)
    cnt_t = from_dlpack(cnt, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    tmaps = _tmaps(dev, mac)
    tmaps_t = from_dlpack(tmaps, assumed_align=128).mark_layout_dynamic(leading_dim=2)
    if spec.profile:
        prof, pcnt = profile_buffer(dev, mac)
        prof.zero_()
        pcnt.zero_()
    else:
        prof, pcnt = (
            torch.zeros(4, dtype=torch.int64, device=dev),
            torch.zeros(4, dtype=torch.int32, device=dev),
        )
    prof_t = from_dlpack(prof, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    pcnt_t = from_dlpack(pcnt, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    ptab_t = from_dlpack(tables.ptab, assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    itab_t = from_dlpack(tables.itab, assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    ftab_t = from_dlpack(tables.ftab, assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    pdl = 1 if pdl_mod.launch_mode() else 0
    tile_n = spec.tile_n or (8 if M <= 8 else 16)
    if tile_n < M:
        raise ValueError(f"decoder: M={M} exceeds TILE_N {tile_n}")
    mlp_stages = spec.mlp_stages or (6 if tile_n == 8 else 4)
    tail_split = (M >= 8) if spec.tail_split is None else bool(spec.tail_split)
    w16_stages = spec.w16_stages or (4 if tile_n == 8 else 3)
    key = (
        "decoder",
        str(dt),
        spec.hk,
        spec.hv,
        spec.hq,
        spec.hkv,
        spec.d,
        spec.rot,
        spec.page,
        spec.H,
        spec.I,
        S,
        float(spec.eps),
        mac,
        pdl,
        bool(spec.prefetch),
        bool(spec.l2_prefetch),
        bool(spec.xb_prefetch),
        bool(spec.profile),
        mlp_stages,
        tile_n,
        bool(spec.mma_attn),
        bool(spec.dual_acc),
        bool(spec.kring3),
        tail_split,
        int(spec.spec_rows),
        spec.wtype,
        w16_stages,
        bool(spec.drafter_fold),
        spec.state_vk,
        spec.conv_sd,
        spec.kv_packed,
        spec.slot_table,
        spec.gate_up_split,
        spec.q_gate_il,
    )
    fn = _KERNELS.get(key)
    args = (
        a1,
        b1,
        c1,
        a2,
        b2,
        ws_t,
        wsa_t,
        a3,
        s3,
        b3,
        c3,
        a4,
        s4,
        b4,
        fa,
        fb,
        out_t,
        xw0_t,
        xw1_t,
        sq0_t,
        sq1_t,
        mz_t,
        attn_t,
        res_t,
        act_t,
        qk_t,
        po_t,
        pl_t,
        qb_t,
        rope_t,
        pos_t,
        slot_t,
        seq_t,
        bt_t,
        slots_t,
        acc_t,
        cnt_t,
        tmaps_t,
        ptab_t,
        itab_t,
        ftab_t,
        prof_t,
        pcnt_t,
        int(tables.nl),
        int(tables.cs),
        int(KS),
        bt_stride,
        slots_stride,
        slots_gstride,
    )
    if fn is None:
        import cutlass

        kern = DecoderMegaSm120(
            cutlass.Float32,
            (TILE[0], tile_n, TILE[2]),
            spec.hk,
            spec.hv,
            spec.H,
            spec.hq,
            spec.hkv,
            spec.d,
            spec.rot,
            spec.page,
            spec.I,
            nsplit=S,
            eps=spec.eps,
            pdl=pdl,
            prefetch=bool(spec.prefetch),
            l2_prefetch=bool(spec.l2_prefetch),
            xb_prefetch=bool(spec.xb_prefetch),
            profile=bool(spec.profile),
            mlp_stages=mlp_stages,
            mma_attn=bool(spec.mma_attn),
            dual_acc=bool(spec.dual_acc),
            kring3=bool(spec.kring3),
            tail_split=tail_split,
            spec_rows=int(spec.spec_rows),
            wtype=spec.wtype,
            w16_stages=w16_stages,
            drafter_fold=bool(spec.drafter_fold),
            state_vk=spec.state_vk,
            conv_sd=spec.conv_sd,
            kv_packed=spec.kv_packed,
            slot_table=spec.slot_table,
            gate_up_split=spec.gate_up_split,
            q_gate_il=spec.q_gate_il,
        )
        fn = _KERNELS[key] = _rt.compile_cached(kern, *args, mac, stream=None, key=key)

    def entry(stream):
        fn(*args, _rt.stream_handle(stream))

    return entry


def kv_splits_for(spec: DecoderSpec, M: int, max_seq_len, seq_lens=None):
    return _kv_splits(M, spec.hkv, max_seq_len, spec.page, seq_lens)
