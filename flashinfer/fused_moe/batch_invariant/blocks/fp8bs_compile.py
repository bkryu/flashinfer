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
from ....gemm.batch_invariant.spec import BLOCK_K, SCALE_BLOCK_N
from ..spec import ACTS, params_key
from ..spec_fp8bs import BLOCK as QBLOCK
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
from .fp8bs_small import BSQuantKernel, GatherScalesKernel
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


def kernel_name(
    cfg: Knobs, chain: Fp8BsChain, interleaved: Optional[bool] = None
) -> str:
    """The grouped object's name: the chain (kind, activation, output format, parameters), the record without its
    ``grid_clusters``, and the weight interleave where it differs from the chain's own pairing (the f32 oracle on the gated
    weight)."""
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    il = chain.pair if interleaved is None else interleaved
    return jit.sanitize_symbol_name(
        f"grouped_e4m3bs_{chain.describe()}_{knobs.describe()}"
        + ("_il" if il != chain.pair else "")
    )


def quant_kernel_name(
    dtype_str: str, wpr: int, trigger: str = "none", enable_pdl: bool = True
) -> str:
    """The standalone quantizer's name: the input dtype, the warps per row (a placement), the PDL trigger placement and PDL;
    never M or K."""
    return jit.sanitize_symbol_name(
        f"fp8bs_quant_{dtype_str}_w{wpr}_t{trigger}_pdl{int(enable_pdl)}"
    )


def gather_scales_name(k: int) -> str:
    return jit.sanitize_symbol_name(f"moe_gather_scales_k{k}")


def can_implement_grouped_fp8bs(
    knobs: Knobs,
    N: int,
    K: int,
    E: int,
    chain: Fp8BsChain,
    interleaved: Optional[bool] = None,
) -> bool:
    """Acceptance of the grouped block-scale body: a record of the dense block-scale class's space at swizzle 1 (the group
    scheduler has no raster swizzle) with a tile of two promotion warpgroups (N 64 / 128 / 256), 16-byte e4m3 rows, K a whole
    number of scale blocks (every block one smem stage), the weight's rows whole 128-row scale blocks (the gated chain's
    interleaved weight: whole blocks of the up and the gate half, so a tile's columns lie in one up block and one gate block),
    at most ``MAX_E`` experts, the fused requantization only where the tile's outputs are whole 128-column blocks."""
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
    if chain.out_fmt == OUT_FMT_QUANT and not fused_quant_ok(chain, knobs.tile_n):
        return False
    if K <= 0 or N <= 0 or K % BLOCK_K or N % SCALE_BLOCK_N:
        return False
    il = chain.pair if interleaved is None else interleaved
    if chain.pair and not il:
        return False  # the gated chain pairs the interleaved weight's columns
    if il and N % (2 * SCALE_BLOCK_N):
        return False  # the gate / up halves are whole scale blocks each (I % 128 == 0)
    n_out = N // 2 if chain.pair else N
    return (
        n_out % OUT_ALIGN_BS[chain.out_fmt] == 0
        and 1 <= E <= MAX_E
        and (not il or N % (2 * GROUP) == 0)
    )


def compile_grouped_fp8bs_fn(
    cfg: Knobs, chain: Fp8BsChain, interleaved: Optional[bool] = None
) -> Callable[[], Any]:
    """The zero-argument compile closure of one grouped block-scale specialisation."""
    in_dt, al = IN_DT[IN_FMT], ALIGN[IN_FMT]
    out_dt, oal = OUT_DT_BS[chain.out_fmt], OUT_ALIGN_BS[chain.out_fmt]

    def compile_fn():
        sym_m = cute.sym_int(32)
        sym_n = cute.sym_int(32, divisibility=SCALE_BLOCK_N)
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
            GroupedBlockScaleFp8Gemm(cfg, chain, interleaved),
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
    cfg: Knobs, chain: Fp8BsChain, interleaved: Optional[bool] = None
):
    """``kernel(a3, sfa, b3, sfb, out2, sfo, counts, grid)`` (:mod:`.fp8bs_kernel`); ``interleaved``: the weight's gate / up row
    interleave (default: the chain's pairing)."""
    return jit.build_and_load(
        MODULE,
        kernel_name(cfg, chain, interleaved),
        compile_grouped_fp8bs_fn(cfg, chain, interleaved),
    )


@functools.cache
def get_compiled_fp8bs_quant(
    dtype_str: str, wpr: int, trigger: str = "none", enable_pdl: bool = True
):
    """``kernel(x [M, K] 16-bit, codes [M, K] u8, s (M, KB) f32 with free strides, M, K)``; one kernel per (dtype, warps per row,
    trigger)."""
    dtype = get_cutlass_dtype(dtype_str)
    name = quant_kernel_name(dtype_str, wpr, trigger, enable_pdl)

    def compile_fn():
        sym_m, sym_k, sym_kb = (
            cute.sym_int(32),
            cute.sym_int(32, divisibility=QBLOCK),
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
            BSQuantKernel(dtype, wpr, trigger),
            x,
            y,
            s,
            Int32(1),
            Int32(QBLOCK),
            enable_pdl,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, name, compile_fn)


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
    "Fp8BsChain",
    "can_implement_grouped_fp8bs",
    "compile_grouped_fp8bs_fn",
    "gather_scales_name",
    "gemm1_chain",
    "get_compiled_fp8bs_quant",
    "get_compiled_gather_scales",
    "get_compiled_grouped_fp8bs",
    "kernel_name",
    "quant_kernel_name",
]
