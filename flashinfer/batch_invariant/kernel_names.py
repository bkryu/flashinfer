"""Mirror of the compiled kernels' cache names, for the bulk precompiler: it decides whether a specialization is
already on disk (:func:`flashinfer.batch_invariant.jit.is_cached`) without importing the kernel modules' naming code
(which would put this file into their cache-key closure) and without loading anything. The device-free test pins the
mirror against the names the kernel module actually passes to ``jit.build_and_load``."""

from __future__ import annotations

from ..jit.cute_dsl_core import sanitize_symbol_name

MX_QUANT_MODULE = "batch_invariant_mx_quant"
_LAYOUT_TAG = {"128x4": "sw", True: "sw", "8x4": "8x4", "linear": "lin", False: "lin"}


def mx_quant_kernel_name(
    fmt_name: str,
    rule_name: str,
    dtype_str: str,
    Kbp: int,
    cfg_describe: str,
    swizzled,
    enable_pdl: bool,
) -> str:
    """The MX quantizers (module :data:`MX_QUANT_MODULE`, classes `mxfp8.quant` / `mxfp4.quant`): the format, the recipe
    constants' name, the input dtype, the Kbp bucket, the config, the scale-factor layout (``sw`` = 128x4, ``8x4``,
    ``lin``) and PDL; never M (one specialization serves every row count)."""
    tag = _LAYOUT_TAG[swizzled]
    return sanitize_symbol_name(
        f"{fmt_name}_{rule_name}_{dtype_str}_Kbp{Kbp}_{cfg_describe}_{tag}_pdl{int(enable_pdl)}"
    )
