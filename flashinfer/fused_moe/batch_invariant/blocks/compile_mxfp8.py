# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the grouped block-scaled GEMM of the MXFP8 MoE classes: one TVM-FFI object per (knobs, chain,
instruction class) with dynamic Tk_pad, N, K and E; cache module ``batch_invariant_moe_mxfp8`` in FlashInfer's CuTe-DSL
disk cache (this file and the scatter kernel are its roots; the grouped epilogue / scheduler and the block-scaled dense
blocks are in its closure). The persistent grid is a dynamic ``Int32`` argument of the compiled kernel
(``GRID_PLACEHOLDER`` at compile time; the launch passes the device's co-resident cluster count), so one object serves
every grid.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from ....batch_invariant import jit
from ....gemm.batch_invariant.blocks.operands import ALIGN as _ALIGN
from ....gemm.batch_invariant.blocks.operands import IN_DT
from ....gemm.batch_invariant.blocks.operands import OUT_DT as _OUT_DT
from ....gemm.batch_invariant.generator import TILES_N_BLOCKSCALED, Knobs
from ....gemm.batch_invariant.spec import SF_ATOM_KBLOCKS, SF_ATOM_ROWS, SF_VEC
from ..mx_permute import get_compiled_scatter_mx  # noqa: F401  (the scatter kernel shares the module)
from .blockscaled_kernel import GROUP_ALIGN, MX_FMTS, GroupedBlockScaledGemmMX
from .epilogue import GROUP, GroupedEpilogueBlock
from .scheduler import MAX_E
from .swap_mxfp8 import (  # noqa: F401  (re-exported)
    SwappedBlockScaledGemmMX,
    is_swapped_record,
    swap_knobs,
    swapped_knobs,
)

MODULE = "batch_invariant_moe_mxfp8"
OUT_DT = {
    **_OUT_DT,
    "mxfp8": cutlass.Uint8,
}  # the fused requantization's codes (the scales a second output)
ALIGN = {**_ALIGN, "mxfp8": 32}  # ... whole 32-element blocks per output row
GRID_PLACEHOLDER = 1  # the compile-time value of the grid argument: a dynamic Int32 of the kernel, never compiled in
SF_BUFFER_DIVISIBILITY = (
    SF_ATOM_ROWS * SF_ATOM_KBLOCKS
)  # a scale-factor buffer is a whole number of 512-byte atoms
BLOCK = SF_VEC["mxfp8"]  # elements per UE8M0 block scale


def kernel_name(
    cfg: Knobs, chain: GroupedEpilogueBlock, in_fmt: str, dense_a: bool = False
) -> str:
    """The object's name: the instruction class, the chain (activation, output format, parameters) and the record
    without its ``grid_clusters`` (the grid is a launch argument); ``dense_a`` the dense packing's body."""
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    mode = "_densea" if dense_a else ""
    return jit.sanitize_symbol_name(
        f"grouped_{in_fmt}_{chain.describe()}_{knobs.describe()}{mode}"
    )


def swapped_kernel_name(
    cfg: Knobs, chain: GroupedEpilogueBlock, in_fmt: str, k_block: int = 0
) -> str:
    """The swapped decode body's object: the instruction class, the chain, the swapped record (its token tile) and the
    k-block when it is not the format's default (``_kb256``: the two-atom stage, `swap_k_block`)."""
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    return jit.sanitize_symbol_name(
        f"swapped_{in_fmt}_{chain.describe()}_{knobs.describe()}"
        + (f"_kb{k_block}" if k_block else "")
    )


#: K elements per smem stage the swapped decode body admits: the format's one 128-byte atom of K and two atoms.
SWAP_K_BLOCKS = (128, 256)


def swap_k_block(in_fmt: str, K: int) -> int:
    """The swapped decode body's k-block for a GEMM of reduction length ``K``: two atoms of K per stage when they divide
    ``K``, else the one-atom default -- a function of the fixed dims alone, so a placement of the same fold (the barrier
    round trips over K halve; the MMA sequence is the class's). The padded body and the dense class keep the default."""
    kbs = [kb for kb in SWAP_K_BLOCKS if K % kb == 0] or [SWAP_K_BLOCKS[0]]
    return max(kbs)


def swap_k_block_arg(in_fmt: str, K: int, k_block: Optional[int] = None) -> int:
    """The k-block as the kernel spells it: ``k_block`` (default `swap_k_block`) with the format's default as 0 (one object
    per k-block)."""
    kb = k_block or swap_k_block(in_fmt, K)
    return 0 if kb == SWAP_K_BLOCKS[0] else kb


def can_implement_grouped_mxfp8(
    knobs: Knobs, N: int, K: int, E: int, chain: GroupedEpilogueBlock, in_fmt: str
) -> bool:
    """Acceptance of the grouped block-scaled body on the MX plans: a well-formed dense record of the class's space at
    swizzle 1 (no in-cluster split, no operand swap, no B-reuse tile, no in-CTA chunk fold: the group scheduler has none of
    them) with a block-scaled tile N, a chain with a bf16 / f16 / f32 output, K a whole number of 32-element scale blocks
    (hence 16-byte e4m3 rows), 16-byte-aligned output rows, whole ``2 x GROUP`` column blocks for the gated chain, at most
    ``MAX_E`` experts."""
    if in_fmt not in MX_FMTS or not isinstance(chain, GroupedEpilogueBlock):
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
        K % BLOCK == 0
        and N % 8 == 0
        and n_out % ALIGN[chain.out_fmt] == 0
        and 1 <= E <= MAX_E
    )


def compile_grouped_mxfp8_fn(
    cfg: Knobs, chain: GroupedEpilogueBlock, in_fmt: str, dense_a: bool = False
) -> Callable[[], Any]:
    """The zero-argument compile closure of one grouped MXFP8 specialisation (``dense_a``: the dense packing, any A row
    count)."""
    assert in_fmt in MX_FMTS, in_fmt
    in_dt, al = IN_DT[in_fmt], ALIGN[in_fmt]
    out_dt, oal = OUT_DT[chain.out_fmt], ALIGN[chain.out_fmt]

    def compile_fn():
        sym_m = (
            cute.sym_int(32) if dense_a else cute.sym_int(32, divisibility=GROUP_ALIGN)
        )  # the padded rows: whole 128-row atoms; the dense packing: any row count
        sym_n, sym_k, sym_e = (
            cute.sym_int(32, divisibility=8),
            cute.sym_int(32, divisibility=al),
            cute.sym_int(32),
        )
        sym_no = cute.sym_int(32, divisibility=oal)
        a_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_m, sym_k, 1),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        b_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_n, sym_k, sym_e),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
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
        obj = GroupedBlockScaledGemmMX(cfg, chain, in_fmt, dense_a)
        if chain.quant:
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


def can_implement_swapped_mxfp8(
    knobs: Knobs,
    N: int,
    K: int,
    E: int,
    chain: GroupedEpilogueBlock,
    in_fmt: str,
    k_block: int = 0,
) -> bool:
    """Acceptance of the swapped decode body (:mod:`.swap_mxfp8`): a swapped record of the dense MX space at a narrow
    token tile (128 weight rows x 8 / 16 / 32 tokens, 1-CTA, no cluster, swizzle 1) and the grouped operand rules (K a
    whole number of scale blocks, N % 8, the gate / up pairing, the output alignment -- whole 32-element blocks for the
    fused requantization --, at most ``MAX_E`` experts, a bf16 / f32 chain or the fused requantization). The body reads the
    dense packing only: the dense rows with the scale factors at the padded positions."""
    if in_fmt not in MX_FMTS or not isinstance(chain, GroupedEpilogueBlock):
        return False
    if not is_swapped_record(knobs) or not knobs.well_formed(in_fmt):
        return False
    if chain.out_fmt not in OUT_DT:
        return False
    if chain.pair and N % (2 * GROUP):
        return False
    n_out = N // 2 if chain.pair else N
    if k_block and (k_block not in SWAP_K_BLOCKS or K % k_block):
        return False
    return (
        K % BLOCK == 0
        and N % 8 == 0
        and n_out % ALIGN[chain.out_fmt] == 0
        and 1 <= E <= MAX_E
    )


def can_implement_mxfp8(
    knobs: Knobs, N: int, K: int, E: int, chain: GroupedEpilogueBlock, in_fmt: str
) -> bool:
    """A record either grouped MX body takes: the padded / dense-A body (a dense record) or the swapped decode body (a
    narrow swapped record); the two spaces are disjoint."""
    return can_implement_grouped_mxfp8(
        knobs, N, K, E, chain, in_fmt
    ) or can_implement_swapped_mxfp8(knobs, N, K, E, chain, in_fmt)


def compile_swapped_mxfp8_fn(
    cfg: Knobs, chain: GroupedEpilogueBlock, in_fmt: str, k_block: int = 0
) -> Callable[[], Any]:
    """The zero-argument compile closure of one swapped MXFP8 specialisation (the dense packing; ``k_block`` as
    `swapped_kernel_name` spells it)."""
    assert in_fmt in MX_FMTS and is_swapped_record(cfg), (in_fmt, cfg.describe())
    in_dt, al = IN_DT[in_fmt], ALIGN[in_fmt]
    out_dt, oal = OUT_DT[chain.out_fmt], ALIGN[chain.out_fmt]

    def compile_fn():
        sym_r, sym_m, sym_k, sym_e = (
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32, divisibility=al),
            cute.sym_int(32),
        )
        sym_n, sym_no = (
            cute.sym_int(32, divisibility=8),
            cute.sym_int(32, divisibility=oal),
        )
        a_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_r, sym_k, 1),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        sfa_fake = cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (cute.sym_int(32, divisibility=SF_BUFFER_DIVISIBILITY),),
            (1,),
            assumed_align=16,
        )
        w_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_n, sym_k, sym_e),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
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
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        grid = GRID_PLACEHOLDER
        obj = SwappedBlockScaledGemmMX(cfg, chain, in_fmt, k_block)
        if chain.quant:
            osf_fake = cute.runtime.make_fake_tensor(
                cutlass.Uint8,
                (cute.sym_int(32, divisibility=SF_BUFFER_DIVISIBILITY),),
                (1,),
                assumed_align=16,
            )
            return cute.compile(
                obj.call_quant,
                a_fake,
                sfa_fake,
                w_fake,
                sfw_fake,
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
            sfa_fake,
            w_fake,
            sfw_fake,
            o_fake,
            c_fake,
            grid,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return compile_fn


@functools.cache
def get_compiled_swapped_mxfp8(
    cfg: Knobs, chain: GroupedEpilogueBlock, in_fmt: str, k_block: int = 0
) -> Any:
    """``kernel(a3, sfa, w3, sfw, out2[, out_sf], counts, grid)``: A ``(R, K, 1)`` the dense quantized rows (``R >= Tk +
    token tile``), ``sfa`` their flat 128x4 scale factors at the padded positions, W ``(N, K, E)``, ``sfw`` (``E``
    buffers), out ``(Tk', Nout)`` (``out_sf`` the fused requantization's flat scale bytes of the padded rows)."""
    assert in_fmt in MX_FMTS and is_swapped_record(cfg), (in_fmt, cfg.describe())
    return jit.build_and_load(
        MODULE,
        swapped_kernel_name(cfg, chain, in_fmt, k_block),
        compile_swapped_mxfp8_fn(cfg, chain, in_fmt, k_block),
    )


@functools.cache
def get_compiled_grouped_mxfp8(
    cfg: Knobs, chain: GroupedEpilogueBlock, in_fmt: str, dense_a: bool = False
) -> Any:
    """``kernel(a3, b3, sfa, sfb, out2, counts, grid)``: A ``(Tk_pad, K, 1)`` e4m3 K-major (the padded permuted
    activations: group offsets at 128-row atoms), B ``(N, K, E)`` e4m3 K-major (the stacked expert weights), SFA / SFB
    the flat uint8 UE8M0 buffers in the 128x4 swizzled layout (SFB: ``E`` consecutive per-expert buffers), out
    ``(Tk_pad, Nout)`` row-major (``Nout = N``, or ``N / 2`` for the gated chains), counts ``(E,)`` int32, ``grid`` the
    persistent grid in clusters. ``in_fmt``: the instruction class, ``mxfp8`` (SM100 / SM103) or ``mxfp8_k64`` (SM107)."""
    assert in_fmt in MX_FMTS and isinstance(chain, GroupedEpilogueBlock), (
        in_fmt,
        chain,
    )
    return jit.build_and_load(
        MODULE,
        kernel_name(cfg, chain, in_fmt, dense_a),
        compile_grouped_mxfp8_fn(cfg, chain, in_fmt, dense_a),
    )
