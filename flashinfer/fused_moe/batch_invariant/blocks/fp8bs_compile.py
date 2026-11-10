# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrappers of the block-scale fp8 MoE class's kernels: the grouped block-scale GEMM (:mod:`.fp8bs_kernel`: one TVM-FFI
object per (knobs, chain, weight interleave) with dynamic Tk, N, K and E), the standalone ``fp8bs.quant`` kernel and the
block-scale gather (:mod:`.fp8bs_small`); cache module ``batch_invariant_moe_fp8bs`` in FlashInfer's CuTe-DSL disk cache (this
file is its root: the grouped body, the dense block-scale body it composes, the quantizer's spec and the grouped blocks are its
closure; the bf16 and the dense modules' keys do not move with an edit here). The persistent grid is a dynamic argument of the
grouped kernel, as for the bf16 module.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint8

from ....batch_invariant import jit
from ....batch_invariant.fp_interface.cute import get_cutlass_dtype
from ....gemm.batch_invariant.blocks.operands import ALIGN, IN_DT, OUT_DT
from ....gemm.batch_invariant.generator import Knobs
from ....gemm.batch_invariant.spec import BLOCK_K
from ..spec import ACTS, params_key
from ..spec_fp8bs import BLOCK as QBLOCK
from ..spec_fp8bs import check_block
from .compile import GRID_PLACEHOLDER
from .epilogue import GROUP
from .fp8bs_kernel import (
    F32_CHAIN,
    GEMM2_CHAIN,
    IN_FMT,
    OUT_FMT_QUANT,
    TILES_N_GROUPED,
    Fp8BsChain,
    GroupedBlockScaleFp8Gemm,
    fused_quant_ok,
)
from .fp8bs_small import BSQuantKernel, GatherRowsScalesKernel, GatherScalesKernel
from .fp8bs_swap_kernel import (
    TOKEN_TILES,
    WEIGHT_TILES,
    SwappedBlockScaleFp8Gemm,
    is_swapped_record_fp8bs,
    swap_knobs_fp8bs,
    swapped_knobs_fp8bs,
)
from .scheduler import MAX_E

MODULE = "batch_invariant_moe_fp8bs"
#: The fused requantization's codes are stored as bytes.
OUT_DT_BS = dict(OUT_DT, e4m3bs=cutlass.Uint8)
OUT_ALIGN_BS = dict(ALIGN, e4m3bs=16)


def gemm1_chain(
    act: str = "silu", params=(), gated: bool = True, fused: bool = False
) -> Fp8BsChain:
    """GEMM1 of the block-scale class: the bf16 class's gated chain on the (gate, up) pair (``act`` in ``spec.ACTS`` with its
    scalars) or its ungated chain on the single accumulator (``spec.UNGATED_ACTS``), rounded to bf16 -- stored as bf16 (the
    standalone requantization pass follows) or requantized in the epilogue (``fused``: the e4m3 codes and the block scales of
    the ``fp8bs.quant`` class)."""
    return Fp8BsChain(
        "glu" if gated else "act",
        act,
        OUT_FMT_QUANT if fused else "bf16",
        params_key(params) if gated else (),
    )


def _tag(block: int) -> str:
    return "" if check_block(block) == QBLOCK else str(block)


def kernel_name(
    cfg: Knobs,
    chain: Fp8BsChain,
    interleaved: Optional[bool] = None,
    block: int = QBLOCK,
) -> str:
    """The grouped object's name: the chain (kind, activation, output format, parameters), the record without its
    ``grid_clusters``, and the weight interleave where it differs from the chain's own pairing (the f32 oracle on the gated
    weight); the 32-wide class's objects carry the block in the prefix (``grouped_e4m3bs32_``)."""
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    il = chain.pair if interleaved is None else interleaved
    return jit.sanitize_symbol_name(
        f"grouped_e4m3bs{_tag(block)}_{chain.describe()}_{knobs.describe()}"
        + ("_il" if il != chain.pair else "")
    )


def swapped_kernel_name(
    cfg: Knobs,
    chain: Fp8BsChain,
    interleaved: Optional[bool] = None,
    block: int = QBLOCK,
) -> str:
    """The swapped decode body's object (:mod:`.fp8bs_swap_kernel`): the record carries the weight and the token tile."""
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    il = chain.pair if interleaved is None else interleaved
    return jit.sanitize_symbol_name(
        f"swapped_e4m3bs{_tag(block)}_{chain.describe()}_{knobs.describe()}"
        + ("_il" if il != chain.pair else "")
    )


def quant_kernel_name(
    dtype_str: str,
    wpr: int,
    trigger: str = "none",
    enable_pdl: bool = True,
    block: int = QBLOCK,
) -> str:
    """The standalone quantizer's name: the input dtype, the warps per row (a placement), the PDL trigger placement, PDL and
    the block size where it is not the 128-wide class's; never M or K."""
    return jit.sanitize_symbol_name(
        f"fp8bs_quant_{dtype_str}_w{wpr}_t{trigger}_pdl{int(enable_pdl)}"
        + (f"_b{block}" if check_block(block) != QBLOCK else "")
    )


def gather_scales_name(k: int) -> str:
    return jit.sanitize_symbol_name(f"moe_gather_scales_k{k}")


def gather_rows_scales_name(k: int, wpr: int) -> str:
    return jit.sanitize_symbol_name(
        f"moe_gather_rows_scales_k{k}" + (f"_w{wpr}" if wpr > 1 else "")
    )


def _shape_ok(
    N: int, K: int, E: int, chain: Fp8BsChain, interleaved: Optional[bool], block: int
) -> bool:
    """The operand rule both bodies share: 16-byte e4m3 rows, K a whole number of smem stages (128 elements: every block one
    stage, or four blocks one stage at the 32-wide class), the weight's rows whole scale blocks (the gated chain's interleaved
    weight: whole blocks of the up and the gate half, so a tile's columns read whole up and gate blocks), at most ``MAX_E``
    experts."""
    if K <= 0 or N <= 0 or K % BLOCK_K or N % block:
        return False
    il = chain.pair if interleaved is None else interleaved
    if chain.pair and not il:
        return False  # the gated chain pairs the interleaved weight's columns
    if il and N % (2 * block):
        return False  # the gate / up halves are whole scale blocks each
    n_out = N // 2 if chain.pair else N
    return (
        n_out % OUT_ALIGN_BS[chain.out_fmt] == 0
        and 1 <= E <= MAX_E
        and (not il or N % (2 * GROUP) == 0)
    )


def can_implement_grouped_fp8bs(
    knobs: Knobs,
    N: int,
    K: int,
    E: int,
    chain: Fp8BsChain,
    interleaved: Optional[bool] = None,
    block: int = QBLOCK,
) -> bool:
    """Acceptance of the grouped block-scale body: a record of the dense block-scale class's space at swizzle 1 (the group
    scheduler has no raster swizzle) with a tile of two promotion warpgroups (N 64 / 128 / 256), the operand rule
    (:func:`_shape_ok` at the block), the fused requantization only for the 128-wide class where the tile's outputs are whole
    128-column blocks."""
    if not isinstance(chain, Fp8BsChain) or chain.out_fmt not in OUT_DT_BS:
        return False
    if (
        not knobs.well_formed(IN_FMT)
        or knobs.swizzle_size != 1
        or knobs.tile_n not in TILES_N_GROUPED
    ):
        return False
    if knobs.cluster_k > 1 or knobs.swap_ab or knobs.cta_chunks:
        return False
    if chain.out_fmt == OUT_FMT_QUANT and (
        block != QBLOCK or not fused_quant_ok(chain, knobs.tile_n)
    ):
        return False
    return _shape_ok(N, K, E, chain, interleaved, check_block(block))


def can_implement_swapped_fp8bs(
    knobs: Knobs,
    N: int,
    K: int,
    E: int,
    chain: Fp8BsChain,
    interleaved: Optional[bool] = None,
    block: int = QBLOCK,
) -> bool:
    """Acceptance of the swapped decode body (:mod:`.fp8bs_swap_kernel`): a swapped record of the class (a 64- / 128-row weight
    tile, an 8 / 16 / 32 / 64 token tile, 1-CTA, no cluster), the bf16 / f32 chains (the fused requantization is not a spelling of
    this body), the operand rule (:func:`_shape_ok` at the block)."""
    if not isinstance(chain, Fp8BsChain) or chain.out_fmt not in ("bf16", "f32"):
        return False
    if not is_swapped_record_fp8bs(knobs):
        return False
    return _shape_ok(N, K, E, chain, interleaved, check_block(block))


def can_implement_fp8bs(
    knobs: Knobs,
    N: int,
    K: int,
    E: int,
    chain: Fp8BsChain,
    interleaved: Optional[bool] = None,
    block: int = QBLOCK,
) -> bool:
    """A record either block-scale grouped body takes: the padded body (the dense-derived records) or the swapped decode body
    (the swapped records); the two spaces are disjoint."""
    return can_implement_grouped_fp8bs(
        knobs, N, K, E, chain, interleaved, block
    ) or can_implement_swapped_fp8bs(knobs, N, K, E, chain, interleaved, block)


def compile_swapped_fp8bs_fn(
    cfg: Knobs,
    chain: Fp8BsChain,
    interleaved: Optional[bool] = None,
    block: int = QBLOCK,
) -> Callable[[], Any]:
    """The zero-argument compile closure of one swapped block-scale specialisation: ``kernel(a3, sfa, w3, sfw, out2, counts,
    grid)``."""
    in_dt, al = IN_DT[IN_FMT], ALIGN[IN_FMT]
    out_dt, oal = OUT_DT_BS[chain.out_fmt], OUT_ALIGN_BS[chain.out_fmt]

    def compile_fn():
        sym_r, sym_m, sym_e = cute.sym_int(32), cute.sym_int(32), cute.sym_int(32)
        sym_n = cute.sym_int(32, divisibility=block)
        sym_k = cute.sym_int(32, divisibility=BLOCK_K)
        sym_kb, sym_nb = cute.sym_int(32), cute.sym_int(32)
        sym_no = cute.sym_int(32, divisibility=oal)
        a_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_r, sym_k, 1),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        sfa_fake = cute.runtime.make_fake_tensor(
            Float32,
            (sym_r, sym_kb),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=4,
        )
        w_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_n, sym_k, sym_e),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        sfw_fake = cute.runtime.make_fake_tensor(
            Float32,
            (sym_e, sym_nb, sym_kb),
            (cute.sym_int64(), cute.sym_int64(), 1),
            assumed_align=4,
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
        return cute.compile(
            SwappedBlockScaleFp8Gemm(cfg, chain, interleaved, block),
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
def get_compiled_swapped_fp8bs(
    cfg: Knobs,
    chain: Fp8BsChain,
    interleaved: Optional[bool] = None,
    block: int = QBLOCK,
):
    """``kernel(a3, sfa, w3, sfw, out2, counts, grid)``: A ``(R, K, 1)`` the dense token rows (``R >= Tk + token tile`` keeps the
    boxes in bounds), SFA ``(R, K / block)`` fp32 (any strides), W ``(N, K, E)``, SFW ``(E, N / block, K / block)``, out ``(Tk',
    Nout)``, counts ``(E,)``."""
    return jit.build_and_load(
        MODULE,
        swapped_kernel_name(cfg, chain, interleaved, block),
        compile_swapped_fp8bs_fn(cfg, chain, interleaved, block),
    )


def compile_grouped_fp8bs_fn(
    cfg: Knobs,
    chain: Fp8BsChain,
    interleaved: Optional[bool] = None,
    block: int = QBLOCK,
) -> Callable[[], Any]:
    """The zero-argument compile closure of one grouped block-scale specialisation."""
    in_dt, al = IN_DT[IN_FMT], ALIGN[IN_FMT]
    out_dt, oal = OUT_DT_BS[chain.out_fmt], OUT_ALIGN_BS[chain.out_fmt]

    def compile_fn():
        sym_m = cute.sym_int(32)
        sym_n = cute.sym_int(32, divisibility=block)
        sym_k = cute.sym_int(32, divisibility=BLOCK_K)
        sym_e = cute.sym_int(32)
        sym_kb, sym_nb, sym_ob = cute.sym_int(32), cute.sym_int(32), cute.sym_int(32)
        sym_no = cute.sym_int(32, divisibility=oal)
        a_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_m, sym_k, 1),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        sfa_fake = cute.runtime.make_fake_tensor(
            Float32,
            (sym_m, sym_kb),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=4,
        )
        b_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_n, sym_k, sym_e),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        sfb_fake = cute.runtime.make_fake_tensor(
            Float32,
            (sym_e, sym_nb, sym_kb),
            (cute.sym_int64(), cute.sym_int64(), 1),
            assumed_align=4,
        )
        o_fake = cute.runtime.make_fake_tensor(
            out_dt,
            (sym_m, sym_no),
            (cute.sym_int64(divisibility=oal), 1),
            assumed_align=16,
        )
        # its own row count: unread by the bf16 / f32 chains (a (1, 1) dummy)
        sfo_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_ob, cute.sym_int(32)), (cute.sym_int64(), 1), assumed_align=4
        )
        c_fake = cute.runtime.make_fake_tensor(Int32, (sym_e,), (1,), assumed_align=4)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        grid = GRID_PLACEHOLDER  # a dynamic argument of the kernel: the launch passes the device's value
        return cute.compile(
            GroupedBlockScaleFp8Gemm(cfg, chain, interleaved, block),
            a_fake,
            sfa_fake,
            b_fake,
            sfb_fake,
            o_fake,
            sfo_fake,
            c_fake,
            grid,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return compile_fn


@functools.cache
def get_compiled_grouped_fp8bs(
    cfg: Knobs,
    chain: Fp8BsChain,
    interleaved: Optional[bool] = None,
    block: int = QBLOCK,
):
    """``kernel(a3, sfa, b3, sfb, out2, sfo, counts, grid)`` (:mod:`.fp8bs_kernel`); ``interleaved``: the weight's gate / up row
    interleave (default: the chain's pairing); ``block`` the class (128 / 32)."""
    return jit.build_and_load(
        MODULE,
        kernel_name(cfg, chain, interleaved, block),
        compile_grouped_fp8bs_fn(cfg, chain, interleaved, block),
    )


@functools.cache
def get_compiled_fp8bs_quant(
    dtype_str: str,
    wpr: int,
    trigger: str = "none",
    enable_pdl: bool = True,
    block: int = QBLOCK,
):
    """``kernel(x [M, K] 16-bit, codes [M, K] u8, s (M, KB) f32 with free strides, M, K)``; one kernel per (dtype, warps per row,
    trigger, block)."""
    dtype = get_cutlass_dtype(dtype_str)
    name = quant_kernel_name(dtype_str, wpr, trigger, enable_pdl, block)

    def compile_fn():
        sym_m, sym_k, sym_kb = (
            cute.sym_int(32),
            cute.sym_int(32, divisibility=block),
            cute.sym_int(32),
        )
        x = cute.runtime.make_fake_tensor(
            dtype, (sym_m, sym_k), (cute.sym_int64(divisibility=8), 1), assumed_align=16
        )
        y = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_m, sym_k),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        s = cute.runtime.make_fake_tensor(
            Float32,
            (sym_m, sym_kb),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=4,
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            BSQuantKernel(dtype, wpr, trigger, block),
            x,
            y,
            s,
            Int32(1),
            Int32(block),
            enable_pdl,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, name, compile_fn)


@functools.cache
def get_compiled_gather_rows_scales(k: int, wpr: int = 1):
    """``kernel(x_bytes [T, B] u8, src [>= Tk] i32, x_perm [>= Tk, B] u8, sf (KB, T) f32 any strides, sf_perm (KB, >= Tk) f32,
    Tk, B, KB)``: the fused row + scale gather (one launch); ``wpr`` warps per row (``permute.warps_per_row``)."""

    def compile_fn():
        sym_t, sym_n, sym_p = cute.sym_int(32), cute.sym_int(32), cute.sym_int(32)
        sym_b = cute.sym_int(32, divisibility=16)
        sym_kb, sym_t2 = cute.sym_int(32), cute.sym_int(32)
        x_fake = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_t, sym_b),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        src_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        xp_fake = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_p, sym_b),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        s_fake = cute.runtime.make_fake_tensor(
            Float32,
            (sym_kb, sym_t2),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=4,
        )
        sp_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_kb, sym_p), (cute.sym_int64(), 1), assumed_align=4
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            GatherRowsScalesKernel(k, wpr),
            x_fake,
            src_fake,
            xp_fake,
            s_fake,
            sp_fake,
            Int32(1),
            Int32(16),
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, gather_rows_scales_name(k, wpr), compile_fn)


@functools.cache
def get_compiled_gather_scales(k: int):
    """``kernel(sf (KB, T) f32 any strides, src [>= Tk] i32, sf_perm (KB, Tk) f32, Tk, KB)``."""

    def compile_fn():
        sym_kb, sym_t, sym_n, sym_tk = (
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32),
        )
        s_fake = cute.runtime.make_fake_tensor(
            Float32,
            (sym_kb, sym_t),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=4,
        )
        src_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        sp_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_kb, sym_tk), (cute.sym_int64(), 1), assumed_align=4
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            GatherScalesKernel(k),
            s_fake,
            src_fake,
            sp_fake,
            Int32(1),
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, gather_scales_name(k), compile_fn)


__all__ = [
    "ACTS",
    "F32_CHAIN",
    "GEMM2_CHAIN",
    "MODULE",
    "OUT_ALIGN_BS",
    "OUT_DT_BS",
    "OUT_FMT_QUANT",
    "TILES_N_GROUPED",
    "TOKEN_TILES",
    "WEIGHT_TILES",
    "Fp8BsChain",
    "can_implement_fp8bs",
    "can_implement_grouped_fp8bs",
    "can_implement_swapped_fp8bs",
    "compile_grouped_fp8bs_fn",
    "compile_swapped_fp8bs_fn",
    "gather_rows_scales_name",
    "gather_scales_name",
    "gemm1_chain",
    "get_compiled_fp8bs_quant",
    "get_compiled_gather_rows_scales",
    "get_compiled_gather_scales",
    "get_compiled_grouped_fp8bs",
    "get_compiled_swapped_fp8bs",
    "is_swapped_record_fp8bs",
    "kernel_name",
    "quant_kernel_name",
    "swap_knobs_fp8bs",
    "swapped_kernel_name",
    "swapped_knobs_fp8bs",
]
