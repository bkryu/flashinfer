# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Persistent tile scheduler with the raster order, the swizzle size and the grid as RUNTIME values (the variant-reduction
study's prototype; ``study/variant-reduction``).

The static scheduler (FlashInfer's copy of the DSL's ``static_persistent_tile_scheduler``) takes ``swizzle_size`` and
``raster_along_m`` as Python values and bakes them into the STRUCTURE of ``problem_layout_ncluster_mnl`` (the nested shape
and its strides), so every (raster, swizzle) pair is its own compiled kernel. This scheduler decodes the persistent linear
cluster index with runtime arithmetic (three FastDivmod divisors formed on the host at launch) and reproduces the static
decode exactly, for every pair:

* raster along m, swizzle ``s``, ``M x N`` clusters, ``N' = round_up(N, s)`` (the static layout ``(M, (s, N'/s), L)`` with
  strides ``(s, (1, s M), M N')``, inverted by ``idx2crd``): ``n0 = idx % s``, ``m = (idx // s) % M``,
  ``n1 = (idx // (s M)) % (N'/s)``, ``l = idx // (M N')``, ``n = n0 + s n1``;
* raster along n: the roles of m and n exchanged;
* ``s = 1``: the static FastDivmod path's order for either raster (the formula above with ``s = 1``).

Generic form with ``F`` the fast (swizzled) axis count, ``S`` the slow axis count, ``F' = round_up(F, s)``:
``f0 = idx % s; r = idx // s; sl = r % S; r = r // S; f1 = r % (F'/s); l = r // (F'/s); f = f0 + s f1``;
``(m, n) = (sl, f)`` for raster m, ``(f, sl)`` for raster n; a work index is valid iff ``idx < S F' L``.

The static scheduler runs the phantom clusters of a swizzled layout whose fast-axis count is not a multiple of ``s`` (its
``round_up``; their TMA stores are clipped and their loads zero-filled); this one runs them too, so a static kernel and its
dynamic twin do the same work in the same order: the tile -> CTA assignment is a placement (the bits never depend on it),
and keeping it identical makes the kernel-time comparison measure the decode arithmetic alone.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cutlass_dsl import (
    Int32,
    Integer,
    dsl_user_op,
    extract_mlir_values,
    min,
    new_from_mlir_values,
)

# FlashInfer's copy of the DSL's scheduler: the work-tile record is shared
from flashinfer.cute_dsl.attention.fmha.helpers.static_persistent_tile_scheduler import (
    WorkTileInfo,
)


class DynSchedParams:
    """The decode's runtime values, formed on the host at launch: the swizzle ``s``, the raster flag (1 = along m), the
    slow-axis count ``S``, the swizzled fast-axis block count ``F'/s``, the valid-work bound ``S F' L`` and the three
    FastDivmod divisors (``s``, ``S``, ``F'/s``). ``cluster_shape_mn`` stays static (the cluster is a launch attribute)."""

    @dsl_user_op
    def __init__(
        self,
        num_tiles_mnl,
        cluster_shape_mn: Tuple[int, int],
        swizzle: Integer,
        raster_m: Integer,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        self.cluster_shape_mn = tuple(int(x) for x in cluster_shape_mn)
        cm = cute.ceil_div(num_tiles_mnl[0], self.cluster_shape_mn[0], loc=loc, ip=ip)
        cn = cute.ceil_div(num_tiles_mnl[1], self.cluster_shape_mn[1], loc=loc, ip=ip)
        L = Int32(num_tiles_mnl[2])
        s = Int32(swizzle)
        rm = Int32(raster_m)
        one_minus = Int32(1) - rm
        S = Int32(cm) * rm + Int32(cn) * one_minus  # the slow axis: m under raster m, n under raster n
        F = Int32(cn) * rm + Int32(cm) * one_minus  # the fast (swizzled) axis
        F1 = (F + s - Int32(1)) // s  # F' / s
        self.swizzle = s
        self.raster_m = rm
        self.S = S
        self.F1 = F1
        self.total = S * F1 * s * L  # S F' L clusters, phantoms included (the static scheduler's size)
        self.fdd_s = cute.fast_divmod_create_divisor_v2(s, loc=loc, ip=ip)
        self.fdd_S = cute.fast_divmod_create_divisor_v2(S, loc=loc, ip=ip)
        self.fdd_F1 = cute.fast_divmod_create_divisor_v2(F1, loc=loc, ip=ip)

    _FIELDS = ("swizzle", "raster_m", "S", "F1", "total", "fdd_s", "fdd_S", "fdd_F1")

    def __extract_mlir_values__(self) -> list:
        values, self._pos = [], []
        for name in self._FIELDS:
            v = extract_mlir_values(getattr(self, name))
            values += v
            self._pos.append(len(v))
        return values

    def __new_from_mlir_values__(self, values: list) -> "DynSchedParams":
        new = object.__new__(DynSchedParams)
        new.cluster_shape_mn = self.cluster_shape_mn
        rest = list(values)
        for name, n in zip(self._FIELDS, self._pos):
            setattr(new, name, new_from_mlir_values(getattr(self, name), rest[:n]))
            rest = rest[n:]
        assert not rest, len(rest)
        return new

    def grid_shape(self, max_active_clusters, *, loc=None, ip=None):
        """``(cluster m, cluster n, persistent clusters)``: the static ``get_grid_shape`` with the cluster count a runtime
        value: ``min(clusters of the problem, max_active_clusters)``."""
        return (*self.cluster_shape_mn, min(self.total, Int32(max_active_clusters)))


class DynTileScheduler:
    """The static scheduler's interface (``create`` / ``initial_work_tile_info`` / ``get_current_work`` /
    ``advance_to_next_work`` / ``num_tiles_executed``) over :class:`DynSchedParams`."""

    def __init__(
        self,
        params: DynSchedParams,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster,
        num_tiles_executed: Int32,
    ):
        self.params = params
        self.num_persistent_clusters = num_persistent_clusters
        self._current_work_linear_idx = current_work_linear_idx
        self.cta_id_in_cluster = cta_id_in_cluster
        self._num_tiles_executed = num_tiles_executed

    def __extract_mlir_values__(self) -> list:
        values = extract_mlir_values(self.num_persistent_clusters)
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self._num_tiles_executed))
        values.extend(extract_mlir_values(self.params))
        return values

    def __new_from_mlir_values__(self, values: list) -> "DynTileScheduler":
        assert len(values) >= 6
        return DynTileScheduler(
            new_from_mlir_values(self.params, values[6:]),
            new_from_mlir_values(self.num_persistent_clusters, [values[0]]),
            new_from_mlir_values(self._current_work_linear_idx, [values[1]]),
            new_from_mlir_values(self.cta_id_in_cluster, values[2:5]),
            new_from_mlir_values(self._num_tiles_executed, [values[5]]),
        )

    @staticmethod
    @dsl_user_op
    def create(
        params: DynSchedParams,
        block_idx,
        grid_dim,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "DynTileScheduler":
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(
            params.cluster_shape_mn, loc=loc, ip=ip
        )
        bidx, bidy, bidz = block_idx
        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )
        return DynTileScheduler(
            params, Int32(num_persistent_clusters), Int32(bidz), cta_id_in_cluster, Int32(0)
        )

    def _work_for(self, idx: Int32, *, loc=None, ip=None) -> WorkTileInfo:
        p = self.params
        is_valid = idx < p.total
        r, f0 = divmod(idx, p.fdd_s)
        r, sl = divmod(r, p.fdd_S)
        l, f1 = divmod(r, p.fdd_F1)
        f = f0 + p.swizzle * f1
        rm = p.raster_m
        nm = Int32(1) - rm
        m = sl * rm + f * nm
        n = f * rm + sl * nm
        cm, cn = p.cluster_shape_mn
        coord = (
            cute.arch.make_warp_uniform(Int32(m) * Int32(cm) + self.cta_id_in_cluster[0]),
            cute.arch.make_warp_uniform(Int32(n) * Int32(cn) + self.cta_id_in_cluster[1]),
            cute.arch.make_warp_uniform(Int32(l) + self.cta_id_in_cluster[2]),
        )
        return WorkTileInfo(coord, cute.arch.make_warp_uniform(is_valid))

    @dsl_user_op
    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self._work_for(self._current_work_linear_idx, loc=loc, ip=ip)

    @dsl_user_op
    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self.get_current_work(loc=loc, ip=ip)

    @dsl_user_op
    @cute.jit
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None) -> None:
        self._current_work_linear_idx += Int32(advance_count) * Int32(self.num_persistent_clusters)
        self._num_tiles_executed += Int32(1)

    @property
    @cute.jit
    def num_tiles_executed(self) -> Int32:
        return self._num_tiles_executed


def build_grid_dyn(st, c: cute.Tensor, max_active_clusters, swizzle, raster_m):
    """The dynamic twin of ``operands.build_grid``: the tile count from C and the CTA tile, the scheduler's runtime values
    from ``swizzle`` / ``raster_m`` (Int32 arguments of the kernel), the persistent grid ``min(clusters, max_active_clusters)``
    with ``max_active_clusters`` an Int32 argument too. Returns ``(params, grid)``."""
    gc = cute.zipped_divide(
        c, tiler=cute.slice_(st.cta_tile_shape_mnk, (None, None, 0))
    )
    num_ctas_mnl = gc[(0, (None, None, None))].shape
    params = DynSchedParams(num_ctas_mnl, st.cluster_shape_mn, swizzle, raster_m)
    return params, params.grid_shape(max_active_clusters)
