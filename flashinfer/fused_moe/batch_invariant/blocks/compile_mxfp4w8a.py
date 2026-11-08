# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the grouped block-scaled GEMM of the mixed MXFP4-weight x MXFP8-activation MoE classes: one TVM-FFI
object per (knobs, chain, instruction class) with dynamic Tk_pad, N, K and E; cache module ``batch_invariant_moe_mxfp4w8a``
in FlashInfer's CuTe-DSL disk cache (this file is its root; the grouped block-scaled body with the mixed plans, the grouped
epilogues / scheduler and the block-scaled dense blocks are in its closure; the scatter kernel is the MXFP8 MoE module's).
The persistent grid is a dynamic ``Int32`` argument of the compiled kernel (``GRID_PLACEHOLDER`` at compile time; the
launch passes the device's co-resident cluster count), so one object serves every grid.

The e2m1 weight is passed byte-packed (uint8 ``(N, K / 2, E)``, two codes per byte, element ``2j`` in the low nibble) and
viewed as the fp4 ``(N, K, E)`` operand inside the kernel (the dense body's ``fp4_view``); on the SM100 / SM103 class the
TMA unpacks it into the instruction's 8-bit container, which needs whole 64-byte runs of the row: K a multiple of 128
there (``gemm.batch_invariant.spec.k_granule``), of 32 on the SM107 class.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from ....batch_invariant import jit
from ....gemm.batch_invariant.blocks.operands import ALIGN as _ALIGN
from ....gemm.batch_invariant.blocks.operands import IN_DT
from ....gemm.batch_invariant.blocks.operands import OUT_DT as _OUT_DT
from ....gemm.batch_invariant.generator import TILES_N_BLOCKSCALED, Knobs
from ....gemm.batch_invariant.spec import (
    SF_ATOM_KBLOCKS,
    SF_ATOM_ROWS,
    SF_VEC,
    k_granule,
)
from .blockscaled_kernel import GROUP_ALIGN, GroupedBlockScaledGemmMX
from .epilogue import GROUP, GroupedChain
from .scheduler import MAX_E

MODULE = "batch_invariant_moe_mxfp4w8a"
#: The mixed classes the grouped body serves: the SM100 / SM103 instruction (K = 32) and the SM107 one (K = 64).
MIXED_FMTS = ("mxfp4w8a", "mxfp4w8a_k64")
OUT_DT = {
    **_OUT_DT,
    "mxfp8": cutlass.Uint8,
}  # the fused requantization's codes (the scales a second output)
ALIGN = {**_ALIGN, "mxfp8": 32}  # ... whole 32-element blocks per output row
GRID_PLACEHOLDER = 1  # the compile-time value of the grid argument: a dynamic Int32 of the kernel, never compiled in
SF_BUFFER_DIVISIBILITY = (
    SF_ATOM_ROWS * SF_ATOM_KBLOCKS
)  # a scale-factor buffer is a whole number of 512-byte atoms
BLOCK = SF_VEC["mxfp4w8a"]  # elements per UE8M0 block scale


def kernel_name(cfg: Knobs, chain: GroupedChain, in_fmt: str) -> str:
    """The object's name: the instruction class, the chain (activation, output format, parameters) and the record
    without its ``grid_clusters`` (the grid is a launch argument)."""
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    return jit.sanitize_symbol_name(
        f"grouped_{in_fmt}_{chain.describe()}_{knobs.describe()}"
    )


def a_granule(in_fmt: str) -> int:
    """The K granule of the class's operand rows: whole 32-element scale blocks (16-byte e4m3 rows, 16-byte e2m1 rows per 32
    elements) on the SM107 class; whole 128-element runs on the SM100 / SM103 class (the TMA's unpacking format reads the
    e2m1 row in 64-byte groups): the dense mixed class's rule."""
    return max(BLOCK, k_granule(in_fmt))


def can_implement_grouped_mxfp4w8a(
    knobs: Knobs, N: int, K: int, E: int, chain: GroupedChain, in_fmt: str
) -> bool:
    """Acceptance of the grouped block-scaled body on the mixed MX plans: a well-formed dense record of the class's space at
    swizzle 1 (no in-cluster split, no operand swap, no B-reuse tile, no in-CTA chunk fold: the group scheduler has none of
    them) with a block-scaled tile N, a chain with a bf16 / f16 / f32 output or the fused requantization's codes, K a whole
    number of the class's granules (32 on the SM107 class, 128 on the SM100 / SM103 class), 16-byte-aligned output rows,
    whole ``2 x GROUP`` column blocks for the gated chain, at most ``MAX_E`` experts."""
    if in_fmt not in MIXED_FMTS or not isinstance(chain, GroupedChain):
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
    if knobs.tile_n not in TILES_N_BLOCKSCALED:
        return False
    if chain.out_fmt not in OUT_DT:
        return False
    if chain.pair and N % (2 * GROUP):
        return False
    n_out = N // 2 if chain.pair else N
    return (
        K % a_granule(in_fmt) == 0
        and N % 8 == 0
        and n_out % ALIGN[chain.out_fmt] == 0
        and 1 <= E <= MAX_E
    )


def compile_grouped_mxfp4w8a_fn(
    cfg: Knobs, chain: GroupedChain, in_fmt: str
) -> Callable[[], Any]:
    """The zero-argument compile closure of one grouped mixed specialisation."""
    assert in_fmt in MIXED_FMTS, in_fmt
    in_dt = IN_DT[in_fmt]  # the activation operand: e4m3
    gran = a_granule(in_fmt)
    out_dt, oal = OUT_DT[chain.out_fmt], ALIGN[chain.out_fmt]

    def compile_fn():
        sym_m = cute.sym_int(
            32, divisibility=GROUP_ALIGN
        )  # the padded rows: whole 128-row atoms
        sym_n, sym_k, sym_e = (
            cute.sym_int(32, divisibility=8),
            cute.sym_int(32, divisibility=gran),
            cute.sym_int(32),
        )
        sym_kb = cute.sym_int(
            32, divisibility=gran // 2
        )  # the weight row's bytes: two e2m1 codes per byte
        sym_no = cute.sym_int(32, divisibility=oal)
        a_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_m, sym_k, 1),
            (cute.sym_int64(divisibility=gran), 1, cute.sym_int64(divisibility=gran)),
            assumed_align=16,
        )
        b_fake = cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (sym_n, sym_kb, sym_e),
            (
                cute.sym_int64(divisibility=gran // 2),
                1,
                cute.sym_int64(divisibility=gran // 2),
            ),
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
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        grid = GRID_PLACEHOLDER  # a dynamic argument of the kernel: the launch passes the device's value
        obj = GroupedBlockScaledGemmMX(cfg, chain, in_fmt)
        if chain.out_fmt == "mxfp8":
            osf_fake = cute.runtime.make_fake_tensor(
                cutlass.Uint8,
                (cute.sym_int(32, divisibility=512),),
                (1,),
                assumed_align=16,
            )
            return cute.compile(
                obj.call_quant,
                a_fake,
                b_fake,
                sfa_fake,
                sfb_fake,
                o_fake,
                osf_fake,
                c_fake,
                grid,
                stream_fake,
                options="--enable-tvm-ffi",
            )
        return cute.compile(
            obj,
            a_fake,
            b_fake,
            sfa_fake,
            sfb_fake,
            o_fake,
            c_fake,
            grid,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return compile_fn


@functools.cache
def get_compiled_grouped_mxfp4w8a(cfg: Knobs, chain: GroupedChain, in_fmt: str) -> Any:
    """``kernel(a3, b3, sfa, sfb, out2, counts, grid)``: A ``(Tk_pad, K, 1)`` e4m3 K-major (the padded permuted
    activations: group offsets at 128-row atoms), B ``(N, K / 2, E)`` uint8 (the stacked packed-e2m1 expert weights, K
    contiguous), SFA / SFB the flat uint8 UE8M0 buffers in the 128x4 swizzled layout (SFB: ``E`` consecutive per-expert
    buffers), out ``(Tk_pad, Nout)`` row-major (``Nout = N``, or ``N / 2`` for the gated chains; the fused requantization's
    codes with the scale bytes as a second output), counts ``(E,)`` int32, ``grid`` the persistent grid in clusters.
    ``in_fmt``: the instruction class, ``mxfp4w8a`` (SM100 / SM103) or ``mxfp4w8a_k64`` (SM107)."""
    assert in_fmt in MIXED_FMTS and isinstance(chain, GroupedChain), (in_fmt, chain)
    return jit.build_and_load(
        MODULE,
        kernel_name(cfg, chain, in_fmt),
        compile_grouped_mxfp4w8a_fn(cfg, chain, in_fmt),
    )
