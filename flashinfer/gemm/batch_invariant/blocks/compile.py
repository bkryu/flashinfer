# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the block-built GEMM: one TVM-FFI object per (knobs, B major, chunks, persistent grid, bias) with
dynamic M, N, K; cache module ``batch_invariant_gemm_bf16`` in FlashInfer's CuTe-DSL disk cache. Operand conventions: A (M, K)
K-major, B (K, N) N-major (torch row-major) or K-major (a transposed weight), C (M, N) N-major, fp32 accumulate; the
biased body takes the (N,) bf16 bias as a fourth operand. The persistent grid (in clusters) is a compile-time constant of the
body: the caller passes the launch device's co-resident cluster count for the knobs' cluster shape
(``api.persistent_clusters``), so the compiled grid matches the device it launches on.
"""

from __future__ import annotations

import dataclasses
import functools

import cutlass
import cutlass.cute as cute

from ....batch_invariant import jit
from ..generator import Knobs
from .kernel import BiasBlockGemm, BlockGemm
from .operands import ALIGN, IN_DT, OUT_DT, can_implement  # noqa: F401  (re-exported)

MODULE = "batch_invariant_gemm_bf16"


def max_active_clusters(cfg: Knobs, sm_count: int) -> int:
    """The fallback persistent grid in clusters, one CTA per SM: what the launch device can host at most, not what it
    can host at once (a GPC holds whole clusters only, so the co-resident count is below ``SMs // size`` for clusters of
    more than one CTA; ``api.persistent_clusters`` asks the device). A launch parameter: the bits never depend on it."""
    return sm_count // (cfg.cluster_shape_mn[0] * cfg.cluster_shape_mn[1])


def kernel_name(
    cfg: Knobs, b_major: str, chunks: int, clusters: int, bias: bool = False
) -> str:
    """``clusters`` is the compiled persistent grid in clusters (the knobs' own ``grid_clusters`` when set, else the
    launch device's co-resident count), keyed once as ``_g<clusters>``."""
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    return jit.sanitize_symbol_name(
        f"bf16_b{b_major}_C{chunks}_{knobs.describe()}_g{clusters}"
        + ("_bias" if bias else "")
    )


@functools.cache
def get_compiled_gemm(
    cfg: Knobs, b_major: str, chunks: int, clusters: int, bias: bool = False
):
    """``kernel(a3, b3, c3)``: the unsplit class (``chunks == 1``, L = 1), the fixed-chunk split (batch index = K chunk,
    fp32 partials) or, with ``cfg.cta_chunks``, the in-CTA chunk fold (the same operand views, the output direct with
    L = 1). ``bias``: the biased chain, ``kernel(a3, b3, c3, bias)`` with the (N,) bf16 bias; a body that stores the
    class's output directly (the split's partial kernel never takes it: its fixup adds the bias). ``clusters`` is the
    persistent grid in clusters the body is compiled for: the launch device's co-resident cluster count for the knobs'
    cluster shape (``api.persistent_clusters``), unless the knobs force one (``cfg.grid_clusters``); it is part of the
    cache key. A placement: which CTA computes which tile, never the bits."""
    in_dt, al = IN_DT["bf16"], ALIGN["bf16"]
    in_cta = cfg.cta_chunks and chunks > 1
    direct = chunks == 1 or in_cta
    assert not bias or direct, "the split's fp32 partials are bias-free"
    out_dt = OUT_DT["bf16"] if direct else cutlass.Float32
    oal = ALIGN["bf16"] if direct else ALIGN["f32"]
    grid = cfg.grid_clusters or clusters
    assert grid > 0, (cfg, clusters)
    name = kernel_name(cfg, b_major, chunks, grid, bias)

    def compile_fn():
        sym_m, sym_n, sym_kc = (
            cute.sym_int(32),
            cute.sym_int(32, divisibility=al),
            cute.sym_int(32, divisibility=al),
        )
        L = chunks
        Lc = 1 if in_cta else L
        a_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_m, sym_kc, L),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        if b_major == "n":  # (K, N) row-major viewed as (n, kc, l)
            b_fake = cute.runtime.make_fake_tensor(
                in_dt,
                (sym_n, sym_kc, L),
                (1, cute.sym_int64(divisibility=al), cute.sym_int64(divisibility=al)),
                assumed_align=16,
            )
        else:  # (N, K) row-major, a transposed weight
            b_fake = cute.runtime.make_fake_tensor(
                in_dt,
                (sym_n, sym_kc, L),
                (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
                assumed_align=16,
            )
        c_fake = cute.runtime.make_fake_tensor(
            out_dt,
            (sym_m, sym_n, Lc),
            (cute.sym_int64(divisibility=oal), 1, cute.sym_int64(divisibility=oal)),
            assumed_align=16,
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if bias:
            bias_fake = cute.runtime.make_fake_tensor(
                in_dt, (sym_n,), (1,), assumed_align=16
            )
            return cute.compile(
                BiasBlockGemm(cfg, "bf16", chunks),
                a_fake,
                b_fake,
                c_fake,
                bias_fake,
                grid,
                stream_fake,
                options="--enable-tvm-ffi",
            )
        return cute.compile(
            BlockGemm(cfg, "bf16" if direct else "f32", chunks),
            a_fake,
            b_fake,
            c_fake,
            grid,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, name, compile_fn)
