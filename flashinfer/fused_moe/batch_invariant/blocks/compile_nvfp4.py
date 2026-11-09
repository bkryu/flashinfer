# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the grouped block-scaled GEMM of the NVFP4 MoE classes: one TVM-FFI object per (knobs, chain,
instruction class, packing) with dynamic row count, N, K and E; cache module ``batch_invariant_moe_nvfp4`` in FlashInfer's
CuTe-DSL disk cache (this file and the fp4 scatter kernels are its roots: an edit to the block-scaled body leaves the bf16
MoE module's objects valid and vice versa). The persistent grid is a dynamic ``Int32`` argument of the compiled kernel
(``GRID_PLACEHOLDER`` at compile time; the launch passes the device's co-resident cluster count), so one object serves
every grid.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from ....batch_invariant import jit
from ....gemm.batch_invariant.blocks.operands import ALIGN, OUT_DT
from ....gemm.batch_invariant.generator import Knobs
from ....gemm.batch_invariant.spec import NVFP4_FMTS, SF_ATOM_KBLOCKS, SF_ATOM_ROWS
from .epilogue import GROUP
from .epilogue_nvfp4 import W4A4Chain
from .kernel_nvfp4 import GROUP_ALIGN, GroupedBlockScaledGemm
from .scheduler import MAX_E
from .swap_nvfp4 import SwappedBlockScaledGemm, is_swapped_record

MODULE = "batch_invariant_moe_nvfp4"
GRID_PLACEHOLDER = 1  # the compile-time value of the grid argument: a dynamic Int32 of the kernel, never compiled in
SF_BUFFER_DIVISIBILITY = (
    SF_ATOM_ROWS * SF_ATOM_KBLOCKS
)  # a scale-factor buffer is a whole number of 512-byte atoms
OUT_FMTS = (
    "bf16",
    "f32",
)  # the class's output (bf16) and the fp32 oracle output of the tests


def kernel_name(
    cfg: Knobs, chain: W4A4Chain, in_fmt: str, dense_a: bool = False
) -> str:
    """The object's name: the instruction class, the packing, the chain (activation, scale, parameters) and the record
    without its ``grid_clusters``."""
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    tag = "nvfp4k128" if in_fmt == "nvfp4_k128" else "nvfp4"
    mode = "_densea" if dense_a else ""
    return jit.sanitize_symbol_name(
        f"grouped_{tag}{mode}_{chain.describe()}_{knobs.describe()}"
    )


def can_implement_grouped_nvfp4(
    knobs: Knobs, N: int, K: int, E: int, chain: W4A4Chain, in_fmt: str = "nvfp4"
) -> bool:
    """Acceptance of the grouped block-scaled body: a well-formed dense record of the block-scaled space at swizzle 1 (no
    in-cluster split, no operand swap, no B-reuse tile, no in-CTA chunk fold: the group scheduler has none of them),
    16-byte-aligned K (a whole scale block per 16 bytes), the gated chain on whole ``2 x GROUP`` column blocks, the
    output's alignment, at most ``MAX_E`` experts, a W4A4 or an unscaled oracle chain with a bf16 / f32 output."""
    if in_fmt not in NVFP4_FMTS or not isinstance(chain, W4A4Chain):
        return False
    if (
        not knobs.well_formed(in_fmt)
        or knobs.swizzle_size != 1
        or knobs.cluster_k > 1
        or knobs.swap_ab
        or knobs.b_reuse
        or knobs.cta_chunks
    ):
        return False
    if chain.out_fmt not in OUT_FMTS:
        return False
    if chain.pair and N % (2 * GROUP):
        return False
    n_out = N // 2 if chain.pair else N
    return (
        K % ALIGN[in_fmt] == 0
        and N % 8 == 0
        and n_out % ALIGN[chain.out_fmt] == 0
        and 1 <= E <= MAX_E
    )


def swapped_kernel_name(cfg: Knobs, chain: W4A4Chain, in_fmt: str) -> str:
    """The swapped decode body's object: the instruction class, the chain and the swapped record (its token tile)."""
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    tag = "nvfp4k128" if in_fmt == "nvfp4_k128" else "nvfp4"
    return jit.sanitize_symbol_name(
        f"swapped_{tag}_{chain.describe()}_{knobs.describe()}"
    )


def can_implement_swapped_nvfp4(
    knobs: Knobs, N: int, K: int, E: int, chain: W4A4Chain, in_fmt: str = "nvfp4"
) -> bool:
    """Acceptance of the swapped decode body (:mod:`.swap_nvfp4`): a swapped record of the dense block-scaled space at a narrow
    token tile (128 weight rows x 8 / 16 / 32 tokens, 1-CTA, no cluster, swizzle 1) and the grouped operand rules (a whole scale
    block per 16 bytes of K, the gated chain on whole ``2 x GROUP`` column blocks, the output's alignment, at most ``MAX_E``
    experts, a W4A4 or an unscaled oracle chain). The body reads the dense packing only."""
    if in_fmt not in NVFP4_FMTS or not isinstance(chain, W4A4Chain):
        return False
    if not is_swapped_record(knobs) or not knobs.well_formed(in_fmt):
        return False
    if chain.out_fmt not in OUT_FMTS:
        return False
    if chain.pair and N % (2 * GROUP):
        return False
    n_out = N // 2 if chain.pair else N
    return (
        K % ALIGN[in_fmt] == 0
        and N % 8 == 0
        and n_out % ALIGN[chain.out_fmt] == 0
        and 1 <= E <= MAX_E
    )


def can_implement_nvfp4(
    knobs: Knobs, N: int, K: int, E: int, chain: W4A4Chain, in_fmt: str = "nvfp4"
) -> bool:
    """A record either block-scaled grouped body takes: the padded / dense-A body (a dense record) or the swapped decode body
    (a narrow swapped record); the two spaces are disjoint."""
    return can_implement_grouped_nvfp4(
        knobs, N, K, E, chain, in_fmt
    ) or can_implement_swapped_nvfp4(knobs, N, K, E, chain, in_fmt)


def compile_swapped_nvfp4_fn(
    cfg: Knobs, chain: W4A4Chain, in_fmt: str
) -> Callable[[], Any]:
    """The zero-argument compile closure of one swapped specialisation: ``kernel(a3, sfa, w3, sfw, out2, counts, alpha, pts,
    grid)`` -- A ``(rows, K/2, 1)`` the dense rows (at least the token tile past the live rows), ``sfa`` their scale factors at
    the padded positions, W ``(N, K/2, E)``, ``sfw``, out ``(rows', Nout)``, ``pts`` at the padded positions (a multiple of 128)."""
    out_dt, oal = OUT_DT[chain.out_fmt], ALIGN[chain.out_fmt]

    def compile_fn():
        sym_r, sym_m, sym_n, sym_kh, sym_e = (
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32, divisibility=8),
            cute.sym_int(32, divisibility=16),
            cute.sym_int(32),
        )
        sym_mp = cute.sym_int(32, divisibility=GROUP_ALIGN)
        sym_no = cute.sym_int(32, divisibility=oal)
        a_fake = cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (sym_r, sym_kh, 1),
            (cute.sym_int64(divisibility=16), 1, cute.sym_int64(divisibility=16)),
            assumed_align=16,
        )
        sfa_fake = cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (cute.sym_int(32, divisibility=SF_BUFFER_DIVISIBILITY),),
            (1,),
            assumed_align=16,
        )
        w_fake = cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (sym_n, sym_kh, sym_e),
            (cute.sym_int64(divisibility=16), 1, cute.sym_int64(divisibility=16)),
            assumed_align=16,
        )
        sfw_fake = cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (cute.sym_int(32, divisibility=SF_BUFFER_DIVISIBILITY),),
            (1,),
            assumed_align=16,
        )
        o_fake = cute.runtime.make_fake_tensor(
            out_dt,
            (sym_m, sym_no),
            (cute.sym_int64(divisibility=oal), 1),
            assumed_align=16,
        )
        c_fake = cute.runtime.make_fake_tensor(Int32, (sym_e,), (1,), assumed_align=4)
        alpha_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_e,), (1,), assumed_align=4
        )
        pts_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_mp,), (1,), assumed_align=4
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        grid = GRID_PLACEHOLDER
        return cute.compile(
            SwappedBlockScaledGemm(cfg, chain, in_fmt),
            a_fake,
            sfa_fake,
            w_fake,
            sfw_fake,
            o_fake,
            c_fake,
            alpha_fake,
            pts_fake,
            grid,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return compile_fn


@functools.cache
def get_compiled_swapped_nvfp4(
    cfg: Knobs, chain: W4A4Chain, in_fmt: str = "nvfp4"
) -> Any:
    """The swapped decode body's object for a swapped record, chain and instruction class (``compile_swapped_nvfp4_fn``)."""
    assert (
        in_fmt in NVFP4_FMTS
        and isinstance(chain, W4A4Chain)
        and chain.out_fmt in OUT_FMTS
        and is_swapped_record(cfg)
    ), (in_fmt, chain, cfg.describe())
    return jit.build_and_load(
        MODULE,
        swapped_kernel_name(cfg, chain, in_fmt),
        compile_swapped_nvfp4_fn(cfg, chain, in_fmt),
    )


def compile_grouped_nvfp4_fn(
    cfg: Knobs, chain: W4A4Chain, in_fmt: str, dense_a: bool
) -> Callable[[], Any]:
    """The zero-argument compile closure of one grouped block-scaled specialisation."""
    out_dt, oal = OUT_DT[chain.out_fmt], ALIGN[chain.out_fmt]

    def compile_fn():
        sym_m = (
            cute.sym_int(32) if dense_a else cute.sym_int(32, divisibility=GROUP_ALIGN)
        )
        sym_mp = (
            cute.sym_int(32, divisibility=GROUP_ALIGN) if dense_a else sym_m
        )  # the padded rows (pts, scale factors)
        sym_n, sym_kh, sym_e = (
            cute.sym_int(32, divisibility=8),
            cute.sym_int(32, divisibility=16),
            cute.sym_int(32),
        )
        sym_no = cute.sym_int(32, divisibility=oal)
        a_fake = cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (sym_m, sym_kh, 1),
            (cute.sym_int64(divisibility=16), 1, cute.sym_int64(divisibility=16)),
            assumed_align=16,
        )
        b_fake = cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (sym_n, sym_kh, sym_e),
            (cute.sym_int64(divisibility=16), 1, cute.sym_int64(divisibility=16)),
            assumed_align=16,
        )
        sfa_fake = cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (cute.sym_int(32, divisibility=SF_BUFFER_DIVISIBILITY),),
            (1,),
            assumed_align=16,
        )
        sfb_fake = cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (cute.sym_int(32, divisibility=SF_BUFFER_DIVISIBILITY),),
            (1,),
            assumed_align=16,
        )
        o_fake = cute.runtime.make_fake_tensor(
            out_dt,
            (sym_m, sym_no),
            (cute.sym_int64(divisibility=oal), 1),
            assumed_align=16,
        )
        c_fake = cute.runtime.make_fake_tensor(Int32, (sym_e,), (1,), assumed_align=4)
        alpha_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_e,), (1,), assumed_align=4
        )
        pts_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_mp,), (1,), assumed_align=4
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        grid = GRID_PLACEHOLDER
        return cute.compile(
            GroupedBlockScaledGemm(cfg, chain, in_fmt, dense_a),
            a_fake,
            b_fake,
            sfa_fake,
            sfb_fake,
            o_fake,
            c_fake,
            alpha_fake,
            pts_fake,
            grid,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return compile_fn


@functools.cache
def get_compiled_grouped_nvfp4(
    cfg: Knobs, chain: W4A4Chain, in_fmt: str = "nvfp4", dense_a: bool = False
) -> Any:
    """``kernel(a3, b3, sfa, sfb, out2, counts, alpha, pts, grid)``: A ``(rows, K/2, 1)`` packed fp4 as uint8 (the padded
    packing: ``rows`` a multiple of 128; ``dense_a``: any row count, the scale factors and ``pts`` at the padded positions),
    B ``(N, K/2, E)`` packed fp4, SFA / SFB flat uint8 128x4 scale-factor buffers (SFB: E consecutive buffers), out ``(rows,
    Nout)`` bf16 / f32 row-major, counts ``(E,)`` int32, ``alpha (E,)`` fp32, ``pts (rows_sf,)`` fp32, ``grid`` the persistent
    grid in clusters."""
    assert (
        in_fmt in NVFP4_FMTS
        and isinstance(chain, W4A4Chain)
        and chain.out_fmt in OUT_FMTS
    ), (in_fmt, chain)
    return jit.build_and_load(
        MODULE,
        kernel_name(cfg, chain, in_fmt, dense_a),
        compile_grouped_nvfp4_fn(cfg, chain, in_fmt, dense_a),
    )
