"""Genuine LEAF PURE water interval adapter for the scheduled participant (task phase1_leaf_calendar_impl_20261112, Design.md section 3); derived from the accepted f257/Leaf1e468 reference and re-based on the
qualified LEAF delta (`leaf.biocalendar`, `leaf.plant_pet`, `growth.demand_per_plant`) and SYRUP water109. The old `leaf1e468_*` source profiles are NOT accepted here.

MASY is the integrator: SYRUP owns all physical water (hydrology, E, the availability floor, the single actual debit, the lifetime ledger and request sequence), LEAF owns plants
(demand, roots, uptake allocation, growth). This module composes the ACTUAL pure functions of both for ONE dated open interevent sub-interval and returns ONE sealed candidate:

    evolve_interevent (real hydrology+E, once per call)  ->  growth.demand  ->  lifecycle_availability.layer_availability_m  ->  uptake.propose (m3)
      ->  request_withdrawal (ONE debit, depth m)  ->  uptake.accept (ACTUAL debit, never altered)  ->  accumulate D/U/A/T  ->  [explicit growth.grow]

Everything is pure: nothing is written to the caller's head, arrays, root map or tables; a failure anywhere (including a growth failure after the water candidate) drops the whole candidate and
leaves both parents unchanged. Exactly-once is a property of the caller-owned head (`adopt`): the exact head INSTANCE, the exact water parent INSTANCE, the actual
`LifecycleState.request_sequence`, the LEAF generation and the token are compared against the CURRENT head; SYRUP's pure API does not invalidate an old parent and no global registry exists.
No old `leaf.IntereventSession`, `leaf.syrup_water`, `ss.step` or `withdraw_plants` is used (they would double E/hydrology and bypass the lifetime ledger). No CLI, checkpoint, wind, plasticity, PET or water LAW lives here (the calendar decisions are `leaf.biocalendar`'s through `masy.leaf_calendar`; the root map / canopy geometry rebuild is LEAF's `rebuild_geometry`). Unqualified beyond the bounded tests recorded with this task. CPU NumPy/Numba is run here; the CUDA sibling (explicit separate source profile, native9f CUDA lifecycle, all grid fields device-resident through `lop.xp`) is AUTHORED, NOT RUN.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import weakref
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any

import numpy as np

from maple_syrup import lifecycle_availability as la
from maple_syrup import water_lifecycle as wl

from masy import leaf_calendar as lcal
from masy import leaf_source_profile as lsp

__all__ = ["required_head_bytes", "IntervalError", "IntervalMemoryError", "IntervalOperator", "LeafBoundary", "IntervalHead", "IntervalCandidate", "prepare_interval_operator", "new_boundary", "make_head",
           "bind_head", "release_head", "run_interval", "apply_growth", "adopt", "required_interval_bytes", "carry_boundary", "rebind_operator", "required_session_bytes", "VERSION", "MEMORY_SCOPE", "MM_PER_DAY_PER_M_S"]

VERSION = "0.2.0"
MM_PER_DAY_PER_M_S = 86_400_000.0  # m/s -> mm/day, declared once: the terrain PET field (m/s) that drives hydrology E is gathered for plants in mm/day
_F8 = 8
_SEAL = object()
_MINTED: "weakref.WeakSet[Any]" = weakref.WeakSet()
MAX_TOKEN_CHARS = 128
MEMORY_SCOPE = ("fresh arrays of one interval (LEAF D/proposal/feedback/accumulators/growth outputs, the (L,N) availability/A/depth/accepted conversions) admitted stage by stage before the water "
                "candidate; borrowed/resident arrays (caller population, root map, tables, accepted SYRUP state) are reported, not charged; SYRUP stages receive the LEAF-live fresh bytes once as "
                "retained_feedback_bytes; excluded: interpreter objects, allocator pools, JIT, CUDA context, backend reduction scratch beyond the components' own allowances, whole-process RSS. A reading is not a reservation.")


class IntervalError(ValueError):
    """Any adapter refusal (type/shape/gate/binding/closure/adoption). Raised before the candidate exists unless it names a late check; the parents are never changed."""


class IntervalMemoryError(IntervalError):
    """A stage needs more bytes than the strict reading reports; raised before the water candidate."""


# ------------------------------------------------------------------ sealed records
def _mint(obj: Any) -> Any:
    _MINTED.add(obj)
    return obj


def _freeze(a: Any) -> Any:
    if isinstance(a, np.ndarray):
        v = a.view()
        v.flags.writeable = False
        return v
    return a


def _lb(base: Any, **changes: Any) -> Any:
    """A new sealed LeafBoundary equal to `base` except for `changes` (the only constructor path after `new_boundary`)."""
    d = {n: getattr(base, n) for n in LeafBoundary.__slots__ if n != "__weakref__"}
    unknown = set(changes) - set(d)
    if unknown:
        raise IntervalError(f"internal: unknown boundary fields {sorted(unknown)}")
    d.update(changes)
    return LeafBoundary(_SEAL, **d)


class _Sealed:
    __slots__ = ("__weakref__",)

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise IntervalError(f"{type(self).__name__} is minted only by the adapter functions")
        for name in self.__slots__:
            if name != "__weakref__":
                object.__setattr__(self, name, fields.pop(name))
        if fields:
            raise IntervalError(f"internal: unexpected fields {sorted(fields)}")
        _mint(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")


class IntervalOperator(_Sealed):
    """Sealed binding of the verified source receipt, the genuine lifecycle operator and the compiled LEAF species table (CPU NumPy/Numba)."""

    __slots__ = ("receipt", "lop", "table", "leaf", "backend", "leaf_backend", "cuda", "n_layers", "n_cells", "area_m2", "leaf_budget_bytes", "n_species", "catalog", "geometry", "mask_ident")


class LeafBoundary(_Sealed):
    """The LEAF-side accepted boundary (no water): borrowed population handles (read-only views), the CURRENT root map / canopy map, the pending period sums, the single `LeafCalendar` (the only
    pending-credit counter), the generation and the water continuity mark of the last bound lifecycle state."""

    __slots__ = ("B_kg", "E_kg", "species_index", "live", "canopy_area_m2", "root_map", "canopy_map", "base_state", "live_rows", "D_sum", "U_sum", "A_int", "T_int", "calendar", "generation", "mark", "anchor", "geo_elapsed", "water_digest")


class IntervalHead(_Sealed):
    """The scheduler-owned head, replaced whole by `adopt` (O(1): latest watermark/receipt only; no parent chain, id map or ledger copy)."""

    __slots__ = ("op", "water", "leaf", "watermark", "receipt")


class IntervalCandidate(_Sealed):
    """ONE sealed candidate (water half + LEAF half + scalar receipt), minted only after every stage succeeded."""

    __slots__ = ("kind", "op", "water", "leaf", "receipt", "arrays", "_parent_head", "_parent_water", "seq_before", "seq_after", "generation_before", "token", "_state")


# ------------------------------------------------------------------ strict helpers
def _count(v: Any, name: str) -> int:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < 0:
        raise IntervalError(f"{name} must be a non-Boolean integer >= 0, got {type(v).__name__}")
    return int(v)


def _token(v: Any) -> str:
    if type(v) is not str or not v or len(v) > MAX_TOKEN_CHARS or not v.isprintable():
        raise IntervalError(f"interval_token must be a printable non-empty str of <= {MAX_TOKEN_CHARS} chars")
    return v


def _read_free_cold(fn: Any) -> int:
    try:
        return lsp.read_free(fn)
    except lsp.SourceProfileError as exc:
        raise IntervalMemoryError(str(exc)) from None


def _read_free(fn: Any) -> int:
    try:
        return lsp.read_free(fn)
    except lsp.SourceProfileError as exc:
        raise IntervalMemoryError(str(exc)) from None


def _admit(op: IntervalOperator, need: int, stage: str) -> None:
    free = _read_free(op.lop.column.available_bytes_fn)  # fresh strict reading every stage (no cached figure, no default)
    if int(need) > free:
        raise IntervalMemoryError(f"interval stage {stage!r} needs an estimated {int(need)} B of fresh bytes but only {free} B are reported; nothing was computed. Scope: {MEMORY_SCOPE}")


def _admit_host(op: IntervalOperator, need: int, stage: str) -> None:
    """CUDA only: the small host staging (LEAF packets, S-sized forcing, scalar receipts) against the lifecycle's own host reader, strict and fresh."""
    free = _read_free(op.lop._cuda.host_available_bytes_fn)
    if int(need) > free:
        raise IntervalMemoryError(f"interval host stage {stage!r} needs an estimated {int(need)} B of host staging but only {free} B are reported; nothing was computed")


def _typed(op: IntervalOperator, a: Any, name: str, dtype: Any, shape: tuple[int, ...]) -> Any:
    xp = op.lop.xp
    if type(a) is not xp.ndarray or a.dtype != dtype or tuple(a.shape) != shape or not a.flags.c_contiguous:
        raise IntervalError(f"{name} must be an exact C-contiguous {np.dtype(dtype).name} {shape} array of the lifecycle namespace")
    return a


_GEOMETRY_KEYS = ("max_edges", "max_bytes", "canopy_max_edges", "canopy_max_bytes", "builder", "valid_cell_mask")


# ------------------------------------------------------------------ preparation
def prepare_interval_operator(receipt: Any, lop: Any, *, table: Any, leaf_budget_bytes: Any, catalog: Any, geometry: Any) -> IntervalOperator:
    """Bind the verified source receipt, the genuine lifecycle operator, the compiled LEAF table, the species catalog and the declared geometry-rebuild budgets. Cold. A CPU receipt binds only a CPU
    NumPy/Numba lifecycle; the separate CUDA receipt binds only the explicit native9f CUDA lifecycle (genuine column/device preflight runs here); neither receipt is accepted for the other.
    `geometry` = {max_edges, max_bytes (root map), canopy_max_edges, canopy_max_bytes, builder, valid_cell_mask|None}: the sparse CSR caps of `rebuild_geometry` (P/S/E-sized, never P*N)."""
    lsp.check_hot(receipt)
    wl._check_lop(lop)  # genuine sealed operator; on CUDA also the CURRENT-device gate (no switch, no fallback)
    cuda = lop._cuda is not None
    if cuda:
        if receipt.device_kind != "cuda" or lop.backend != "cuda" or lop.xp is np:
            raise IntervalError("a CUDA lifecycle needs the explicit CUDA source receipt (a CPU receipt is never accepted for it)")
        if lop._cuda.profile not in (wl.CUDA9F_PROFILE, wl.CUDA9F_BOUNDARY_PROFILE):
            raise IntervalError("only the explicit native9f CUDA lifecycle profiles are bound by this source profile")
        leaf_backend = "cuda"
    else:
        if receipt.device_kind != "cpu" or lop.backend not in ("numpy", "numba") or lop.xp is not np:
            raise IntervalError("a CPU lifecycle needs the CPU source receipt (a CUDA receipt is never accepted for it)")
        leaf_backend = lop.backend
    leaf = {"uptake": receipt.module("leaf.uptake"), "growth": receipt.module("leaf.growth"), "forcing": receipt.module("leaf.forcing"), "biocalendar": receipt.module("leaf.biocalendar"),
            "plant_pet": receipt.module("leaf.plant_pet"), "state": receipt.module("leaf.state")}
    if type(table) is not receipt._classes["leaf.growth.CompiledSpecies"] or table.backend != leaf_backend:
        raise IntervalError(f"table must be the verified leaf.growth.CompiledSpecies compiled for backend {leaf_backend!r}")
    if cuda and int(table.device) != int(lop.column.device_id):
        raise IntervalError("the compiled table is bound to a different CUDA device than the lifecycle column")
    if type(catalog) is not receipt._classes["leaf.traits.SpeciesCatalog"]:
        raise IntervalError("catalog must be the verified leaf.traits.SpeciesCatalog")
    if type(geometry) is not dict or not {"max_edges", "max_bytes", "canopy_max_edges", "canopy_max_bytes"} <= set(geometry) or set(geometry) - set(_GEOMETRY_KEYS):
        raise IntervalError(f"geometry must be a plain dict with max_edges/max_bytes/canopy_max_edges/canopy_max_bytes (+ optional builder, valid_cell_mask); keys allowed {_GEOMETRY_KEYS}")
    geom = {k: _count(geometry[k], f"geometry.{k}") for k in ("max_edges", "max_bytes", "canopy_max_edges", "canopy_max_bytes")}
    geom["builder"] = geometry.get("builder", "compiled")
    mask, mask_ident = geometry.get("valid_cell_mask"), None
    if mask is not None:  # COLD, admitted: the actual optional valid-cell mask is validated, OWNED (a read-only copy: no mutable borrowed array behind the frozen identity) and hashed ONCE
        if type(mask) is not np.ndarray or mask.dtype != np.bool_ or mask.shape != (int(lop.n_cells),) or not mask.flags.c_contiguous:
            raise IntervalError(f"geometry.valid_cell_mask must be None or a host C-contiguous numpy bool array of shape ({int(lop.n_cells)},) (the native canopy mask layout; nothing is converted)")
        reader = lop._cuda.host_available_bytes_fn if cuda else lop.column.available_bytes_fn  # a HOST copy on both backends
        free = _read_free_cold(reader)
        if 2 * int(mask.nbytes) + 4096 > free:
            raise IntervalMemoryError(f"owning the valid-cell mask needs an estimated {2 * int(mask.nbytes) + 4096} B of host bytes but only {free} B are reported; nothing was copied or hashed")
        owned = mask.copy()
        owned.flags.writeable = False
        mask = owned
        mask_ident = {"shape": [int(mask.shape[0])], "dtype": "bool", "sha256": hashlib.sha256(memoryview(mask).cast("B")).hexdigest()}
    geom["valid_cell_mask"] = mask
    budget = _count(leaf_budget_bytes, "leaf_budget_bytes")
    if budget < 1:
        raise IntervalError("leaf_budget_bytes must be >= 1")
    return IntervalOperator(_SEAL, receipt=receipt, lop=lop, table=table, leaf=leaf, backend=lop.backend, leaf_backend=leaf_backend, cuda=cuda, n_layers=int(lop.n_layers), n_cells=int(lop.n_cells),
                            area_m2=float(lop.column.cell_area_m2), leaf_budget_bytes=budget, n_species=int(table.n_species), catalog=catalog, geometry=MappingProxyType(geom), mask_ident=None if mask_ident is None else MappingProxyType(mask_ident))


def rebind_operator(op: IntervalOperator, new_lop: Any) -> IntervalOperator:
    """A NEW sealed operator over the RESTORED lifecycle operator `new_lop` (the restart path): same verified receipt/table/catalog/geometry caps/budget, nothing minted by class swap. `new_lop` must be a
    genuine lifecycle operator with the same operator digest, backend, layers, cells and (CUDA) device as `op.lop`; `op` is never mutated and the budget/generation/requests are untouched."""
    _check_op(op)
    wl._check_lop(new_lop)
    if (new_lop.sha256 != op.lop.sha256 or new_lop.backend != op.lop.backend or int(new_lop.n_layers) != op.n_layers or int(new_lop.n_cells) != op.n_cells
            or (op.cuda and (new_lop._cuda is None or int(new_lop.column.device_id) != int(op.lop.column.device_id)))):
        raise IntervalError("the restored lifecycle operator differs from the one the LEAF operator was prepared for (digest/backend/layers/cells/device)")
    return IntervalOperator(_SEAL, receipt=op.receipt, lop=new_lop, table=op.table, leaf=op.leaf, backend=new_lop.backend, leaf_backend=op.leaf_backend, cuda=op.cuda, n_layers=op.n_layers,
                            n_cells=op.n_cells, area_m2=float(new_lop.column.cell_area_m2), leaf_budget_bytes=op.leaf_budget_bytes, n_species=op.n_species, catalog=op.catalog, geometry=op.geometry,
                            mask_ident=op.mask_ident)


def _check_op(op: Any) -> None:
    if type(op) is not IntervalOperator or op not in _MINTED:
        raise IntervalError("op must be an IntervalOperator from prepare_interval_operator()")
    lsp.check_hot(op.receipt)
    wl._check_lop(op.lop, preflight=False)


def _mark_of(op: IntervalOperator, water: Any) -> tuple[Any, ...]:
    """Continuity mark of the water lifetime at an accepted lifecycle state: the request sequence, the actual count of T requests, the reset count and the operator digest."""
    return (int(water.request_sequence), int(water.lifetime.n_t_requests), int(water.reset_count), str(water.lop_sha256))


def _canonical_live_rows(op: IntervalOperator, base_state: Any, live_rows: Any, P: int) -> np.ndarray:
    """The caller's live-row index list, validated against the base `PlantState` with LEAF's own status codes: int64, shape (P,), strictly increasing, within [0, base.n) and EXACTLY the plants whose status is
    live (a Boolean/float/ragged input is refused; nothing is truncated)."""
    a = np.asarray(live_rows)
    if a.dtype.kind not in "iu" or a.shape != (P,):
        raise IntervalError("live_rows must be an integer array of shape (P,) (the live rows of base_state in state order)")
    rows = np.ascontiguousarray(a, dtype=np.int64)
    n = int(base_state.n)
    if P and (int(rows.min()) < 0 or int(rows.max()) >= n or bool(np.any(np.diff(rows) <= 0))):
        raise IntervalError("live_rows must be strictly increasing indices within base_state")
    want = np.flatnonzero(np.asarray(base_state.status) == op.leaf["state"].LIVE)
    if not np.array_equal(rows, want):
        raise IntervalError("live_rows are not exactly the live plants of base_state")
    return rows


def new_boundary(op: IntervalOperator, water: Any, *, B_kg: Any, E_kg: Any, species_index: Any, live: Any, canopy_area_m2: Any, root_map: Any, canopy_map: Any, base_state: Any, live_rows: Any,
                 calendar: Any, geo_elapsed: Any = Fraction(0)) -> LeafBoundary:
    """The FIRST LEAF boundary: borrowed population views + the fixed root map / canopy map + the base `PlantState` (the state the schedule anchor stands at) + a `LeafCalendar`. `water` is the
    accepted lifecycle state (boundary or interevent) whose continuity mark is recorded. Zero-filled period sums; generation 0."""
    _check_op(op)
    if type(water) is not wl.LifecycleState or water not in wl._MINTED or water.lop_sha256 != op.lop.sha256:
        raise IntervalError("water must be a genuine LifecycleState of THIS lifecycle operator")
    if type(calendar) is not lcal.LeafCalendar or calendar.bc is not op.leaf["biocalendar"]:
        raise IntervalError("calendar must be a masy.leaf_calendar.LeafCalendar over the verified leaf.biocalendar")
    P = int(getattr(B_kg, "shape", (-1,))[0]) if getattr(B_kg, "ndim", 0) == 1 else -1
    if P < 0:
        raise IntervalError("B_kg must be a 1-D plant array")
    arrays = (_typed(op, B_kg, "B_kg", np.float64, (P,)), _typed(op, E_kg, "E_kg", np.float64, (P,)), _typed(op, species_index, "species_index", np.int32, (P,)),
              _typed(op, live, "live", np.bool_, (P,)), _typed(op, canopy_area_m2, "canopy_area_m2", np.float64, (P,)))
    rm = root_map
    if type(rm) is not op.receipt._classes["leaf.uptake.RootMap"]:
        raise IntervalError("root_map must be the verified leaf.uptake.RootMap class")
    if (rm.backend != op.leaf_backend or (op.cuda and int(rm.device_id) != int(op.lop.column.device_id)) or rm.n_plants != P or rm.n_layers != op.n_layers or rm.n_cells != op.n_cells):
        raise IntervalError(f"root_map ({rm.backend}, P={rm.n_plants}, L={rm.n_layers}, N={rm.n_cells}) does not match the lifecycle/population ({op.leaf_backend}, P={P}, L={op.n_layers}, N={op.n_cells})")
    if canopy_map is not None:
        if type(canopy_map) is not op.receipt._classes["leaf.plant_pet.CanopyMap"] or canopy_map.backend != op.leaf_backend or canopy_map.n_plants != P or canopy_map.n_cells != op.n_cells:
            raise IntervalError("canopy_map must be the verified leaf.plant_pet.CanopyMap of this backend, population and grid")
    if type(base_state) is not op.receipt._classes["leaf.state.PlantState"]:
        raise IntervalError("base_state must be the verified leaf.state.PlantState")
    S = op.n_species
    hb = required_head_bytes(op, P)  # admitted BEFORE the first fresh array (strict fresh readings; CPU host / CUDA device + host kept separate)
    _admit(op, hb["fresh_admitted_side_bytes"], "head_init")
    if op.cuda:
        _admit_host(op, hb["fresh_host_bytes"], "head_init_host")
    rows = _canonical_live_rows(op, base_state, live_rows, P)  # strictly increasing, in range, exactly the live rows of base_state (validated AFTER the admission, BEFORE adoption)
    return LeafBoundary(_SEAL, B_kg=_freeze(arrays[0]), E_kg=_freeze(arrays[1]), species_index=_freeze(arrays[2]), live=_freeze(arrays[3]), canopy_area_m2=_freeze(arrays[4]), root_map=rm,
                        canopy_map=canopy_map, base_state=base_state, live_rows=_freeze(rows), D_sum=op.lop.xp.zeros(P), U_sum=op.lop.xp.zeros(P), A_int=np.zeros(S), T_int=np.zeros(S), calendar=calendar,
                        generation=0, mark=_mark_of(op, water), anchor=_mark_of(op, water), geo_elapsed=Fraction(geo_elapsed), water_digest=str(water.digest))


def _boundary_from_restore(op: IntervalOperator, *, B_kg: Any, E_kg: Any, species_index: Any, live: Any, canopy_area_m2: Any, root_map: Any, canopy_map: Any, base_state: Any, live_rows: Any, D_sum: Any,
                           U_sum: Any, A_int: Any, T_int: Any, calendar: Any, generation: int, mark: Any, anchor: Any, geo_elapsed: Any, water_digest: Any) -> LeafBoundary:
    """A boundary from RESTORED, already-verified components (only the biology codec calls this): same type/shape/class/backend checks as `new_boundary`, the head accumulators are the restored sums."""
    _check_op(op)
    P = int(B_kg.shape[0])
    arrays = (_typed(op, B_kg, "B_kg", np.float64, (P,)), _typed(op, E_kg, "E_kg", np.float64, (P,)), _typed(op, species_index, "species_index", np.int32, (P,)), _typed(op, live, "live", np.bool_, (P,)),
              _typed(op, canopy_area_m2, "canopy_area_m2", np.float64, (P,)), _typed(op, D_sum, "D_sum", np.float64, (P,)), _typed(op, U_sum, "U_sum", np.float64, (P,)))
    S = op.n_species
    if not (type(A_int) is np.ndarray and type(T_int) is np.ndarray and A_int.shape == (S,) and T_int.shape == (S,) and A_int.dtype == np.float64 and T_int.dtype == np.float64):
        raise IntervalError("A_int/T_int must be host float64 (S,) arrays")
    rm = root_map
    dev = int(op.lop.column.device_id) if op.cuda else None
    if type(rm) is not op.receipt._classes["leaf.uptake.RootMap"] or rm.backend != op.leaf_backend or rm.n_plants != P or rm.n_layers != op.n_layers or rm.n_cells != op.n_cells:
        raise IntervalError("root_map does not match the lifecycle/population (type, backend, P, L, N)")
    if op.cuda and int(rm.device_id) != dev:
        raise IntervalError("root_map is bound to a different CUDA device than the lifecycle column")
    if canopy_map is not None:
        if type(canopy_map) is not op.receipt._classes["leaf.plant_pet.CanopyMap"] or canopy_map.backend != op.leaf_backend or canopy_map.n_plants != P or canopy_map.n_cells != op.n_cells:
            raise IntervalError("canopy_map does not match the population/grid/backend")
        if op.cuda and int(canopy_map.device_id) != dev:
            raise IntervalError("canopy_map is bound to a different CUDA device than the lifecycle column")
    if type(base_state) is not op.receipt._classes["leaf.state.PlantState"]:
        raise IntervalError("base_state must be the verified leaf.state.PlantState")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise IntervalError("generation must be a non-negative integer")
    rows = _canonical_live_rows(op, base_state, live_rows, P)
    if type(calendar) is not lcal.LeafCalendar or calendar.bc is not op.leaf["biocalendar"]:
        raise IntervalError("calendar must be a masy.leaf_calendar.LeafCalendar over the verified leaf.biocalendar")
    return LeafBoundary(_SEAL, B_kg=_freeze(arrays[0]), E_kg=_freeze(arrays[1]), species_index=_freeze(arrays[2]), live=_freeze(arrays[3]), canopy_area_m2=_freeze(arrays[4]), root_map=rm,
                        canopy_map=canopy_map, base_state=base_state, live_rows=_freeze(rows), D_sum=arrays[5], U_sum=arrays[6], A_int=A_int, T_int=T_int,
                        calendar=calendar, generation=int(generation), mark=tuple(mark), anchor=tuple(anchor), geo_elapsed=Fraction(geo_elapsed), water_digest=str(water_digest))


def bind_head(op: IntervalOperator, leaf: LeafBoundary, water: Any, *, parent: Any = None) -> IntervalHead:
    """Bind an accepted LEAF boundary to a lifecycle INTEREVENT state for the working interval chain (the non-initial path: NO reset of integrals/generation/roots). The water's continuity
    mark (request sequence, actual T-request count, reset count, operator digest) must equal the one recorded when the boundary was last released/created: a state of a different lifetime
    branch, a replayed request or a reset is refused. The scheduler (not this function) proves the exact lineage by having staged `water` from the accepted parent in the same trial."""
    _check_op(op)
    if type(leaf) is not LeafBoundary or leaf not in _MINTED:
        raise IntervalError("leaf must be a LeafBoundary")
    if type(water) is not wl.LifecycleState or water not in wl._MINTED or water.role != "interevent" or water.lop_sha256 != op.lop.sha256:
        raise IntervalError("water must be a genuine interevent LifecycleState of THIS lifecycle operator")
    if parent is not None and (type(parent) is not wl.LifecycleState or str(parent.digest) != leaf.water_digest):
        raise IntervalError("the boundary was released at a different lifecycle state than the given accepted parent (state digest differs); refused")
    if _mark_of(op, water) != leaf.mark:
        raise IntervalError(f"water continuity mark {_mark_of(op, water)} != the LEAF boundary's recorded mark {leaf.mark} (different lifetime branch, request count or reset); refused. Pass `parent=` (the accepted state `water` was entered from) to also bind the exact state digest")
    return IntervalHead(_SEAL, op=op, water=water, leaf=leaf, watermark=(None, None, int(water.revision), water.digest, 0, None), receipt=None)


def release_head(op: IntervalOperator, head: IntervalHead, closed_water: Any) -> LeafBoundary:
    """Release the working head after the interevent closed: the boundary is re-marked from the CLOSED lifecycle state (role 'boundary'; the close folds no request, so the request sequence and
    the T-request count must equal the head's water). Pending sums, calendar, generation, roots and canopy are carried exactly."""
    _check_op(op)
    _check_head_op(op, head)
    if type(closed_water) is not wl.LifecycleState or closed_water not in wl._MINTED or closed_water.role != "boundary" or closed_water.lop_sha256 != op.lop.sha256:
        raise IntervalError("closed_water must be a genuine closed ('boundary') LifecycleState of THIS lifecycle operator")
    if _mark_of(op, closed_water) != _mark_of(op, head.water):
        raise IntervalError("the closed state's continuity mark differs from the working head's water (a request or reset happened at the close); refused")
    return _lb(head.leaf, mark=_mark_of(op, closed_water), water_digest=str(closed_water.digest))


def carry_boundary(op: IntervalOperator, leaf: LeafBoundary, new_water: Any, *, start: Any, end: Any, parent: Any = None) -> LeafBoundary:
    """Carry an accepted boundary through a TRANSPORT event (rain + recession, wind later): the calendar advances `start`->`end` with no E/T/growth (`LeafCalendar.transport`; typed refusal while enabled
    credit is pending: it must have been flushed at the last growth-enabled endpoint), and the boundary is re-released at the lifecycle state the event produced (`new_water`, role boundary or a
    COMPLETED storm). The water continuity mark must be unchanged (a storm folds no biology request and no reset). Integrals, generation, roots and canopy are untouched; nothing is reset."""
    _check_op(op)
    if type(leaf) is not LeafBoundary or leaf not in _MINTED:
        raise IntervalError("leaf must be a LeafBoundary")
    if (type(new_water) is not wl.LifecycleState or new_water not in wl._MINTED or new_water.lop_sha256 != op.lop.sha256 or new_water.role not in ("boundary", "storm")
            or (new_water.role == "storm" and new_water.payload.completed is not True)):
        raise IntervalError("new_water must be a genuine closed boundary or COMPLETED storm state of THIS lifecycle operator")
    if parent is not None and (type(parent) is not wl.LifecycleState or str(parent.digest) != leaf.water_digest):
        raise IntervalError("the boundary was released at a different lifecycle state than the transport's accepted parent (state digest differs); refused")
    if _mark_of(op, new_water) != leaf.mark:
        raise IntervalError(f"the transport's lifecycle state changed the water continuity mark {_mark_of(op, new_water)} != {leaf.mark} (request/reset during transport); refused")
    try:
        cal, _ = leaf.calendar.transport(start, end)
    except lcal.LeafCalendarError as exc:
        raise IntervalError(str(exc)) from None
    return _lb(leaf, calendar=cal, water_digest=str(new_water.digest))


def make_head(op: IntervalOperator, water: Any, **leaf_kw: Any) -> IntervalHead:
    """The first head: `new_boundary` + `bind_head` for an accepted INTEREVENT state. Non-initial heads use `bind_head` with the carried boundary (never this function)."""
    return bind_head(op, new_boundary(op, water, **leaf_kw), water)


def required_session_bytes(op: IntervalOperator, leaf: LeafBoundary) -> dict[str, int]:
    """Session-level own estimate (pure arithmetic, no reader) of the LEAF side, HOST and DEVICE separate (CPU: everything host). `retained_*` = what stays alive across steps: the population/accumulator
    arrays (8*8P), ALL Q base-state record arrays and the live-row metadata, the host A/T (16S), the root map and canopy CSR resident bytes, and (a bound for) the maps a growth rebuild replaces beside the old ones (`geometry.max_bytes + canopy_max_bytes`).
    `owned_mask_host_bytes` is REPORTED SEPARATELY and not added to `retained_*`: the operator owns that N-bool host array since `prepare_interval_operator` admitted it cold, and a fresh free-memory reading already excludes
    it (no second charge). `stage_*` = the LEAF components' fresh bytes of one interevent fragment, summed over demand/integrals/plan/feedback/canopy-PET/grow (a conservative sum, not a peak) plus the declared rebuild caps."""
    _check_op(op)
    up, gr, pp = op.leaf["uptake"], op.leaf["growth"], op.leaf["plant_pet"]
    P, S, N = int(leaf.B_kg.shape[0]), op.n_species, op.n_cells
    rm, cm = leaf.root_map, leaf.canopy_map
    be = op.leaf_backend
    plan = up.estimate_plan_bytes(P, rm.n_edges, rm.n_resources, be)
    fbk = up.estimate_feedback_bytes(P, rm.n_edges, rm.n_resources, be)
    pet = pp.estimate_pet_bytes(P, 0 if cm is None else cm.n_edges, be)
    ests = [gr.memory_estimate("demand_per_plant", P, S, be, has_canopy_area=True), gr.memory_estimate("integrals", 0, S, be), gr.memory_estimate("grow", P, S, be)]
    sd = sum(_alloc_split(e)[0] for e in ests)
    sh = sum(_alloc_split(e)[1] for e in ests)
    caps = int(op.geometry["max_bytes"]) + int(op.geometry["canopy_max_bytes"])
    maps = int(rm.resident_bytes) + (0 if cm is None else int(cm.resident_bytes))
    pop_c, pop_h = 8 * 8 * P, 2 * _F8 * S
    bs = leaf.base_state
    base_h = sum(int(getattr(bs, k).nbytes) for k in type(bs).ARRAY_NAMES) + int(leaf.live_rows.nbytes)  # ALL Q base records' arrays + the live-row metadata (host, retained by the session)
    digest_staging = int(rm.resident_bytes) + (0 if cm is None else int(cm.export_bytes))  # transient full host export of the CSR maps for the identity digests
    mask_h = 0 if op.mask_ident is None else int(op.geometry["valid_cell_mask"].nbytes)  # the operator-OWNED read-only valid-cell mask (N host bools): retained for the operator's lifetime, ALREADY admitted at prepare
    comp = int(plan["total_bytes"]) + int(fbk["total_bytes"]) + int(pet["total_bytes"]) + 8 * N * 2 + 3 * _F8 * op.n_layers * N
    if op.cuda:
        return {"retained_device": pop_c + maps + caps, "retained_host": pop_h + base_h, "stage_device": sd + comp, "stage_host": sh + 8 * (op.n_species + 2) * 8, "digest_staging_host": digest_staging, "owned_mask_host_bytes": mask_h}
    return {"retained_device": 0, "retained_host": pop_c + pop_h + base_h + maps + caps, "stage_device": 0, "stage_host": sd + sh + comp, "digest_staging_host": digest_staging, "owned_mask_host_bytes": mask_h}


def required_head_bytes(op: IntervalOperator, n_plants: int) -> dict[str, Any]:
    """Fresh bytes of the boundary's own accumulators (pure arithmetic): D_sum + U_sum (8P each, on the compute side) and A_int + T_int (8S each, host). The caller's population views, the root map,
    the canopy map and the table are BORROWED (reported, not charged)."""
    _check_op(op)
    P, S = _count(n_plants, "n_plants"), op.n_species
    dev, host = 2 * _F8 * P, 2 * _F8 * S
    return {"fresh_plant_bytes": dev, "fresh_species_bytes_host": host, "fresh_admitted_side_bytes": (dev if op.cuda else dev + host), "fresh_host_bytes": host,
            "borrowed_reported": {"population_views_bytes": 8 * 3 * P + 4 * P + P + 8 * P, "table_host_bytes": int(op.table.memory()["total_bytes"])}, "scope": MEMORY_SCOPE}


def _check_head_op(op: IntervalOperator, head: Any) -> None:
    """The head (and every candidate) is bound to the EXACT IntervalOperator instance: a different operator/table is refused even with equal dimensions, profile and water digest."""
    if type(head) is not IntervalHead or head not in _MINTED:
        raise IntervalError("head must be an IntervalHead")
    if head.op is not op:
        raise IntervalError("the head was created for a different IntervalOperator instance (operator/table); refused")


def _leaf_cap(op: IntervalOperator, rm: Any) -> int:
    """Effective cap handed to the uptake allocator: an operation budget may only TIGHTEN the map cap (never raised)."""
    return min(int(op.leaf_budget_bytes), int(rm.max_bytes))


_ALLOC_KINDS = ("output", "scratch", "packet")


def _alloc_split(est: dict[str, Any]) -> tuple[int, int]:
    """(device, host) fresh-allocating bytes of a LEAF growth estimate, from its own itemized `where`/`kind` (the caller reserve is not an allocation)."""
    dev = sum(int(i["bytes"]) for i in est["items"] if i["where"] == "device" and i["kind"] in _ALLOC_KINDS)
    host = sum(int(i["bytes"]) for i in est["items"] if i["where"] == "host" and i["kind"] in _ALLOC_KINDS)
    return dev, host


def required_interval_bytes(op: IntervalOperator, head: IntervalHead, *, growth: bool = False, transpiration: bool = True, evaporation: bool = True) -> dict[str, Any]:
    """Pure arithmetic (no reader, I/O or device query) of the interval's fresh bytes, per stage, as two figures: `stages`/`host_stages` = FORECAST (the whole incremental prefix: retained fresh arrays of the
    earlier stages + this stage's allocations) and `execute`/`host_execute` = what the stage is about to ALLOCATE (earlier arrays are already consumed and absent from the free reading).
    CPU: everything is the host side. CUDA (authored, unrun): device/host separate via the public `wl.cuda_entry_plan`. Stage `canopy_pet` = the (N,) mm/day field (8N) + `plant_pet.estimate_pet_bytes`
    (the (P,) per-plant PET, the gather scratch). Geometry rebuild after growth is admitted by LEAF's own `rebuild_geometry` against the declared `geometry.max_bytes` with the live reserve below.
    Retained fresh water after the request is ALL SIX accepted arrays. BORROWED (reported, never subtracted): the original parent state, population views, root/canopy maps, table. Exclusions:
    allocator pools, JIT, context, backend reduction scratch beyond the components' allowances, RSS."""
    _check_op(op)
    _check_head_op(op, head)
    up, gr, pp, lop, cuda = op.leaf["uptake"], op.leaf["growth"], op.leaf["plant_pet"], op.lop, op.cuda
    L, N = op.n_layers, op.n_cells
    LN, P, S, rm, cm = _F8 * L * N, int(head.leaf.B_kg.shape[0]), op.n_species, head.leaf.root_map, head.leaf.canopy_map
    D, SS = _F8 * P, 2 * _F8 * S  # demand vector (compute side); a_inc + t_inc / A_int + T_int (HOST S-sized)
    ACC = int(lop.memory.accepted_bytes)  # all six accepted water arrays of one evolved state
    if transpiration and cm is None and P:
        raise IntervalError("transpiration needs a canopy map (per-plant canopy-area-weighted terrain PET); none is bound")
    pet = pp.estimate_pet_bytes(P, 0 if cm is None else cm.n_edges, op.leaf_backend)
    PET = 8 * N + int(pet["total_bytes"]) if transpiration and P else 0
    dem = gr.memory_estimate("demand_per_plant", P, S, op.leaf_backend, has_canopy_area=True)
    inc = gr.memory_estimate("integrals", 0, S, op.leaf_backend)
    plan = up.estimate_plan_bytes(P, rm.n_edges, rm.n_resources, op.leaf_backend)
    fbk = up.estimate_feedback_bytes(P, rm.n_edges, rm.n_resources, op.leaf_backend)
    plan_t, plan_o, fbk_t, fbk_o = int(plan["total_bytes"]), int(plan["owned_bytes"]), int(fbk["total_bytes"]), int(fbk["owned_bytes"])
    dd, dh = (sum(x) for x in zip(_alloc_split(dem), _alloc_split(inc)))
    ph, ah = (8 * up.NP_PACKET, 8 * up.NA_PACKET) if cuda else (0, 0)
    native_fb = 3 * LN
    live_request = D + _F8 * P + plan_o + LN + (0 if cuda else SS)  # demand + per-plant PET + plan + request depth handed ONCE to the SYRUP request
    avail_extra = D + _F8 * P + ACC + (0 if cuda else SS)
    fc: dict[str, int] = {}
    ex: dict[str, int] = {}
    zero = 0 if evaporation else _F8 * N  # E off: the hydrology receives a fresh (N,) zero rate while the canopy gather reads the real field
    if cuda:
        ev = wl.cuda_entry_plan(lop, "evolve_interevent", parent=head.water, retained_feedback_bytes=zero)
        fc["evolve_interevent"] = int(ev["device_bytes"]) + zero
    else:
        fc["evolve_interevent"] = int(wl.required_evolve_interevent_bytes(lop, head.water, zero)) + zero
    if transpiration and P:
        if cuda:
            rq = wl.cuda_entry_plan(lop, "request", parent=head.water, retained_feedback_bytes=live_request)
            rq_dev = int(rq["device_bytes"])
        else:
            rq_dev = int(wl.required_request_bytes(lop, head.water, live_request))
        av = la.required_availability_bytes(lop, extra_reserve_bytes=avail_extra)
        fc["canopy_pet"] = ACC + PET
        fc["demand+increments"] = ACC + PET + dd + (0 if cuda else dh)
        fc["availability+A_m3"] = int(av.required_bytes) + LN
        fc["propose"] = ACC + D + _F8 * P + (0 if cuda else SS) + LN + plan_t
        fc["request_withdrawal"] = rq_dev
        fc["accept"] = ACC + native_fb + D + _F8 * P + (0 if cuda else SS) + plan_o + LN + fbk_t
        fc["accumulate"] = ACC + D + _F8 * P + (0 if cuda else SS) + plan_o + fbk_o + 2 * D + (0 if cuda else 2 * SS)
        ex.update({"canopy_pet": PET, "demand+increments": dd + (0 if cuda else dh), "availability+A_m3": LN, "propose": plan_t, "request_withdrawal": LN, "accept": LN + fbk_t,
                   "accumulate": 2 * D + (0 if cuda else 2 * SS)})
    gd = gh = 0
    if growth:
        g = gr.memory_estimate("grow", P, S, op.leaf_backend)
        gd, gh = _alloc_split(g)
        ex["growth"] = gd + (0 if cuda else gh) + 2 * D + (0 if cuda else 2 * SS) + 2 * D
        fc["growth"] = ACC + D + (0 if cuda else SS) + 2 * D + (0 if cuda else 2 * SS) + ex["growth"]
    hfc: dict[str, int] = {}
    hex_: dict[str, int] = {}
    if cuda:
        hfc["evolve_interevent"] = int(ev["host_bytes"])
        if transpiration and P:
            hfc["request_withdrawal"] = int(rq["host_bytes"])
            hfc["availability+A_m3"] = int(av.host_staging_bytes)
            hfc["demand+increments"] = hex_["demand+increments"] = dh + SS
            hfc["propose"], hex_["propose"] = ph + SS, ph
            hfc["accept"], hex_["accept"] = ah + SS, ah
            hfc["accumulate"] = hex_["accumulate"] = 3 * SS
        if growth:
            hfc["growth"] = hex_["growth"] = gh + 3 * SS
    geometry_reserve = (int(rm.resident_bytes) + (0 if cm is None else int(cm.resident_bytes)) + 8 * 6 * P + 2 * _F8 * S)
    return {"side": "device+host" if cuda else "host", "stages": fc, "execute": ex, "host_stages": hfc, "host_execute": hex_, "peak_fresh_bytes": max(fc.values()) if fc else 0,
            "live_request_bytes": live_request, "avail_extra_bytes": avail_extra, "geometry_reserve_bytes": geometry_reserve,
            "retained_fresh": {"state1_water": ACC, "state2_water": ACC, "native_feedback_until_accept": native_fb, "plan_owned": plan_o, "feedback_owned": fbk_o, "demand": D, "pet_per_plant": _F8 * P},
            "borrowed_reported": {"root_map_resident": int(rm.resident_bytes), "canopy_map_resident": 0 if cm is None else int(cm.resident_bytes), "table_host": int(op.table.memory()["total_bytes"])},
            "scope": MEMORY_SCOPE}


def _check_leaf_caps(op: IntervalOperator, head: IntervalHead, growth: bool, transpiration: bool = True) -> None:
    """Replay each LEAF component's OWN deterministic admission with the arguments the later call will use, BEFORE any SYRUP work (nothing is raised, no cap is changed)."""
    up, gr = op.leaf["uptake"], op.leaf["growth"]
    rm, P, S, cap = head.leaf.root_map, int(head.leaf.B_kg.shape[0]), op.n_species, int(op.leaf_budget_bytes)
    ests = []
    if transpiration and P:
        ests += [("growth.demand_per_plant", gr.memory_estimate("demand_per_plant", P, S, op.leaf_backend, has_canopy_area=True)), ("growth.integral_increments", gr.memory_estimate("integrals", 0, S, op.leaf_backend))]
    if growth:
        ests.append(("growth.grow", gr.memory_estimate("grow", P, S, op.leaf_backend)))
    for name, est in ests:
        if int(est["total_bytes"]) > cap:
            raise IntervalMemoryError(f"LEAF {name} needs {int(est['total_bytes'])} B > leaf_budget_bytes={cap}; refused before any hydrology (component semantics preserved)")
    if transpiration and P:
        uc = _leaf_cap(op, rm)
        plan = up.estimate_plan_bytes(P, rm.n_edges, rm.n_resources, op.leaf_backend)
        fbk = up.estimate_feedback_bytes(P, rm.n_edges, rm.n_resources, op.leaf_backend)
        try:
            up.admit("uptake.propose", plan, max_edges=rm.max_edges, max_bytes=uc, resident_bytes=rm.resident_bytes)
            up.admit("uptake.accept", fbk, max_edges=rm.max_edges, max_bytes=uc, resident_bytes=rm.resident_bytes + int(plan["owned_bytes"]))
        except up.UptakeError as exc:
            raise IntervalMemoryError(f"LEAF uptake cap (min(leaf_budget_bytes, root_map.max_bytes)={uc}): {exc}; refused before any hydrology") from None


# ------------------------------------------------------------------ the interval
def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _gate(op: IntervalOperator, head: Any, record: Any) -> None:
    _check_head_op(op, head)
    if type(record) is not op.receipt._classes["leaf.forcing.TimestepRecord"]:
        raise IntervalError("record must be the verified leaf.forcing.TimestepRecord")
    if record.process_kind != "interevent" or record.gate != "open":
        raise IntervalError(f"only an OPEN interevent record may spend T/E/growth; got process_kind={record.process_kind!r}, gate={record.gate!r} (wind/rain/recession/closed intervals never reach this API)")
    if record.pet_mm_per_day is None:
        raise IntervalError("record.pet_mm_per_day is required (the shared dated PET)")
    if record.start != head.leaf.calendar.cursor:
        raise IntervalError(f"record.start {record.start.isoformat()} does not join the LEAF cursor {head.leaf.calendar.cursor.isoformat()}")
    if head.water.role != "interevent" or head.water.lop_sha256 != op.lop.sha256:
        raise IntervalError("the head's water is not an interevent state of this lifecycle operator")


def _rebuild(op: IntervalOperator, leaf: LeafBoundary, B: Any, E: Any, endpoint: Any) -> tuple[Any, Any, Any]:
    """LEAF's pure `rebuild_geometry` (root map + canopy area + canopy CSR) for the grown B/E at `endpoint` (the exact calendar instant of the growth): the base state advanced by the exact elapsed
    days since the schedule anchor. Returns (root_map, canopy_area_m2, canopy_map)."""
    bc = op.leaf["biocalendar"]
    g = op.geometry
    canopy = {"max_edges": g["canopy_max_edges"], "max_bytes": g["canopy_max_bytes"], "builder": g["builder"]}
    if g["valid_cell_mask"] is not None:
        canopy["valid_cell_mask"] = g["valid_cell_mask"]
    xp = op.lop.xp
    Bh, Eh = (B, E) if not op.cuda else (B.get(), E.get())
    reserve = int(leaf.root_map.resident_bytes) + (0 if leaf.canopy_map is None else int(leaf.canopy_map.resident_bytes)) + 8 * 6 * int(Bh.shape[0]) + 2 * _F8 * op.n_species
    try:
        geo = bc.rebuild_geometry(leaf.base_state, op.catalog, Bh, Eh, elapsed_days=float(bc.elapsed_days(leaf.calendar.schedule.anchor, endpoint)), max_edges=g["max_edges"], max_bytes=g["max_bytes"],
                                  reserve_bytes=reserve, backend=op.leaf_backend, builder=g["builder"], canopy=canopy, live_rows=np.asarray(leaf.live_rows))
    except bc.BioCalendarError as exc:
        raise IntervalMemoryError(f"geometry rebuild refused: {exc}") from None
    del xp
    return geo.root_map, geo.canopy_area_m2, geo.canopy_map


def _grow(op: IntervalOperator, leaf: LeafBoundary, year_end: bool, endpoint: Any, calendar: Any) -> LeafBoundary:
    """ONE explicit growth over the DECLARED period sums, then the LEAF geometry rebuild; returns the new boundary (generation + 1, zeroed sums, the given calendar)."""
    gr = op.leaf["growth"]
    res = gr.grow(op.table, B_kg=leaf.B_kg, E_kg=leaf.E_kg, species_index=leaf.species_index, live=leaf.live, requested_m3=leaf.D_sum, delivered_m3=leaf.U_sum, activity_integral=leaf.A_int,
                  temperature_integral=leaf.T_int, year_end=bool(year_end), max_bytes=op.leaf_budget_bytes)
    if not res["ledger"].get("population_closed", False):
        raise IntervalError("the aboveground population ledger did not close")
    xp = op.lop.xp
    B2, E2 = xp.array(res["B_new_kg"]), xp.array(res["E_new_kg"])  # copy out of the shared [12, P] buffer
    rec = {"year_end": bool(year_end), "packet": dict(res["packet"]), "ledger": dict(res["ledger"]), "endpoint": endpoint.isoformat()}
    P, S = int(B2.shape[0]), op.n_species
    rm, area, cmap = _rebuild(op, leaf, B2, E2, endpoint)
    endpoint_elapsed = op.leaf["biocalendar"].elapsed_days(leaf.calendar.schedule.anchor, endpoint)
    new = _lb(leaf, B_kg=_freeze(B2), E_kg=_freeze(E2), canopy_area_m2=_freeze(area), root_map=rm, canopy_map=cmap, D_sum=xp.zeros(P), U_sum=xp.zeros(P), A_int=np.zeros(S), T_int=np.zeros(S),
              calendar=calendar, generation=int(leaf.generation) + 1, geo_elapsed=Fraction(endpoint_elapsed))
    return new, rec


def _digest(parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def run_interval(op: IntervalOperator, head: IntervalHead, item: Any, *, pet_terrain_m_s: Any, interval_token: Any, transpiration: bool = True, growth_enabled: bool = True, evaporation: bool = True,
                 max_dt_s: Any = 3600.0, max_accepted_steps: int = 100_000, max_retries: int = 12, min_dt_s: Any = 2.0 ** -20) -> IntervalCandidate:
    """ONE pure open-interevent sub-fragment -> ONE sealed candidate. `item` is the `leaf_calendar.PlanItem` (record + the actual `biocalendar.OpenStep`) for this fragment. Order (OpenStep):
    1 optional season growth on the pending credit (zero water, `year_end=True`) + geometry rebuild; 2 ONE hydrology+E advance, then (if `transpiration`) ONE canopy-PET demand -> availability ->
    proposal -> request_withdrawal -> accept; credit accumulates ONLY when the step earns credit (growth enabled); 3 optional boundary growth + rebuild. `transpiration=False`: E only (the caller zeroes
    the PET field when E is off), no request. `evaporation=False` (E off) hands the hydrology a zero rate while the canopy gather (T) still reads `pet_terrain_m_s`. Growth enabled with T off is REFUSED (no water-limited law exists; no water-free growth is invented). Refusals leave `head` and caller arrays untouched."""
    _check_op(op)
    token = _token(interval_token)
    if type(item) is not lcal.PlanItem:
        raise IntervalError("item must be a masy.leaf_calendar.PlanItem")
    record, step = item.record, item.step
    _gate(op, head, record)
    for name, v in (("transpiration", transpiration), ("growth_enabled", growth_enabled), ("evaporation", evaporation)):
        if type(v) is not bool:
            raise IntervalError(f"{name} must be a bool")
    leaf = head.leaf
    P = int(leaf.B_kg.shape[0])
    has_plants = P > 0  # the population is the LIVE rows only (fixed in Phase 1): a host count, never a device reduction/conversion
    if has_plants and growth_enabled and not transpiration:
        raise IntervalError("growth enabled with transpiration off is not supported in v1 (no water-limited growth law exists for it; water-free growth is never invented)")
    if step.schedule.cursor != record.end or leaf.calendar.cursor != record.start:
        raise IntervalError("the OpenStep does not belong to this record/cursor")
    lop, up, gr, pp = op.lop, op.leaf["uptake"], op.leaf["growth"], op.leaf["plant_pet"]
    xp = lop.xp
    L, N, area = op.n_layers, op.n_cells, op.area_m2
    pet = _typed(op, pet_terrain_m_s, "pet_terrain_m_s", np.float64, (N,))
    if not op.cuda and not (np.isfinite(pet).all() and (pet >= 0.0).all()):
        raise IntervalError("pet_terrain_m_s must be finite and >= 0")
    if op.cuda:
        wl._check_lop(lop)
    do_t = bool(transpiration and has_plants)
    do_grow = bool(step.season_first or step.fire_growth)
    _check_leaf_caps(op, head, do_grow, do_t)
    plan_bytes = required_interval_bytes(op, head, growth=do_grow, transpiration=do_t, evaporation=evaporation)
    needs, exec_needs, host_needs, host_exec = plan_bytes["stages"], plan_bytes["execute"], plan_bytes["host_stages"], plan_bytes["host_execute"]

    def forecast(stage: str) -> None:
        _admit(op, needs[stage], stage)
        if op.cuda and stage in host_needs:
            _admit_host(op, host_needs[stage], stage + "_host")

    def admit(stage: str) -> None:
        _admit(op, exec_needs[stage], stage)
        if op.cuda and stage in host_exec:
            _admit_host(op, host_exec[stage], stage + "_host")

    for name in needs:
        forecast(name)
    dt_s_exact = record.dt_seconds_exact
    parent = head.water
    until = float(Fraction(parent.t_s) + Fraction(dt_s_exact))
    if not until > float(parent.t_s):
        raise IntervalError("the interval does not advance the water clock at float resolution; refused")
    grow_recs = []
    cal = leaf.calendar
    # --- 0. season turnover first (zero water; pending credit, normally 0) -------------------------------------------------------------------------------------
    work = leaf
    if step.season_first:
        admit("growth")
        work, rec = _grow(op, work, True, record.start, cal)
        grow_recs.append({"phase": "season_first", **rec})
    # --- 1. real SYRUP hydrology + E, once ---------------------------------------------------------------------------------------------------------------------
    n_before = int(parent.hydro_count)
    pet_h = pet if evaporation else xp.zeros_like(pet)
    r1 = wl.evolve_interevent(lop, parent, expected_parent=wl.expect_parent(parent), until_s=until, pet_terrain_m_s=pet_h, max_dt_s=max_dt_s, max_accepted_steps=max_accepted_steps,
                              max_retries=max_retries, min_dt_s=min_dt_s)
    if r1.reason != "target" or float(r1.state.t_s) != until:
        raise IntervalError(f"the water clock did not reach the interval end exactly (reason={r1.reason!r}, t_s={r1.state.t_s!r}, wanted {until!r}); nothing adopted")
    state1, accepted_steps, rejected_attempts = r1.state, int(r1.accepted_steps), int(r1.rejected_attempts)
    del r1, pet_h
    state2, tc, fb, D, plan_sum = state1, None, None, None, None
    demand_total = delivered_total = actual_m3 = req_m3 = unmet_m3 = proposed_total = 0.0
    payload_sha = None
    D_sum, U_sum, A_int, T_int = work.D_sum, work.U_sum, work.A_int, work.T_int
    if do_t:
        # --- 2. per-plant canopy-area-weighted terrain PET (mm/day) -> LEAF demand --------------------------------------------------------------------------
        admit("canopy_pet")
        pet_mm = xp.multiply(pet, MM_PER_DAY_PER_M_S)  # m/s -> mm/day ONCE (the same field that drove E)
        cp = pp.canopy_pet(work.canopy_map, pet_mm, max_bytes=op.leaf_budget_bytes, reserve_bytes=0)
        per_plant = cp["pet_mm_per_day"]
        del pet_mm, cp
        admit("demand+increments")
        month, year_days, dt_days = record.start.month, (366 if calendar.isleap(record.start.year) else 365), float(record.dt_days_exact)
        dem = gr.demand_per_plant(op.table, B_kg=work.B_kg, canopy_area_m2=work.canopy_area_m2, species_index=work.species_index, live=work.live, month=month, temperature_c=record.temperature_c,
                                  pet_mm_per_day_per_plant=per_plant, dt_days=dt_days, max_bytes=op.leaf_budget_bytes)
        D = dem["requested_m3"]
        a_inc, t_inc = gr.integral_increments(op.table, month=month, temperature_c=record.temperature_c, dt_days=dt_days, year_days=year_days, max_bytes=op.leaf_budget_bytes)
        demand_total = float(dem["packet"]["sum_requested_m3"])
        del dem, per_plant
        # --- 3. availability (the ONLY floor) -> m3 once ------------------------------------------------------------------------------------------------------
        admit("availability+A_m3")
        avail = la.layer_availability_m(lop, state1, expected_parent=wl.expect_parent(state1), extra_reserve_bytes=int(plan_bytes["avail_extra_bytes"]))
        A_m3 = xp.multiply(avail, area)
        del avail
        # --- 4. LEAF proposal (pure; equal access per cell/layer, no compensation) --------------------------------------------------------------------------
        uc = _leaf_cap(op, work.root_map)
        admit("propose")
        plan = up.propose(work.root_map, D, A_m3, max_bytes=uc)
        del A_m3
        depth = xp.divide(plan.proposed_grid(), area)
        # --- 5. SYRUP: ONE debit ---------------------------------------------------------------------------------------------------------------------------------
        admit("request_withdrawal")
        state2, tc = wl.request_withdrawal(lop, state1, expected_parent=wl.expect_parent(state1), depth_m=depth, identity=token, retained_feedback_bytes=int(plan_bytes["live_request_bytes"]))
        del depth, state1
        # --- 6. LEAF gather from the ACTUAL debit (never altered) -------------------------------------------------------------------------------------------
        admit("accept")
        accepted_m3 = xp.multiply(tc.actual_m, area)
        fb = up.accept(plan, accepted_m3, max_bytes=uc)
        del accepted_m3
        actual_m3, req_m3, unmet_m3, payload_sha = float(tc.receipt.actual_m3), float(tc.receipt.requested_m3), float(tc.receipt.unmet_m3), tc.receipt.payload_sha256
        del tc
        delivered_total = float(fb.summary["delivered_total_m3"])
        tol = up.closure_tolerance_m3(max(abs(actual_m3), abs(delivered_total)), P, work.root_map.n_edges, work.root_map.n_resources) * 4.0 + 4.0 * 2.0 ** -52 * max(abs(actual_m3), 1e-300) * (L * N + 8)
        if not (_finite(actual_m3) and abs(delivered_total - actual_m3) <= tol + 1e-30):
            raise IntervalError(f"water closure: sum(delivered)={delivered_total!r} m3 differs from the native actual debit {actual_m3!r} m3 beyond {tol!r}; nothing adopted")
        proposed_total = float(plan.summary["proposed_total_m3"])
        if abs(req_m3 - proposed_total) > tol + 1e-30 + 4.0 * 2.0 ** -52 * max(proposed_total, 1e-300) * (L * N + 8):
            raise IntervalError(f"the native request {req_m3!r} m3 differs from the LEAF proposal {proposed_total!r} m3; nothing adopted")
        # --- 7. credit (ONLY an enabled step earns credit; a growth-disabled T period adds NOTHING and preserves the earlier credit) ------------------------
        if step.credit_days > 0:
            admit("accumulate")
            D_sum, U_sum = work.D_sum + D, work.U_sum + fb.delivered
            A_int, T_int = work.A_int + np.asarray(a_inc, dtype=np.float64), work.T_int + np.asarray(t_inc, dtype=np.float64)
            if not (_finite(demand_total) and _finite(delivered_total) and np.isfinite(A_int).all() and np.isfinite(T_int).all()):
                raise IntervalError("a non-finite accumulator input was produced; nothing adopted")
    cal2 = cal.with_schedule(step.schedule)
    new = _lb(work, D_sum=D_sum, U_sum=U_sum, A_int=A_int, T_int=T_int, calendar=cal2)
    if step.fire_growth:  # --- 8. boundary growth: a failure here drops the WATER half too ------------------------------------------------------------------------------
        admit("growth")
        new, rec = _grow(op, new, bool(step.year_end), record.end, cal2)
        grow_recs.append({"phase": "boundary", **rec})
    mark = _mark_of(op, state2)
    new = _lb(new, mark=mark)
    receipt = {"kind": "interval", "token": token, "profile": op.receipt.profile, "start": record.start.isoformat(), "end": record.end.isoformat(), "dt_seconds_exact": str(dt_s_exact),
               "water_t_s": float(state2.t_s), "accepted_hydro_steps": accepted_steps, "rejected_attempts": rejected_attempts, "transpiration": do_t, "evaporation": bool(evaporation), "growth_enabled": bool(growth_enabled),
               "credit_days": str(step.credit_days), "hydro_count_delta": int(state2.hydro_count) - n_before, "request_sequence_before": int(parent.request_sequence),
               "request_sequence_after": int(state2.request_sequence), "demand_total_m3": demand_total, "proposed_total_m3": proposed_total, "native_requested_m3": req_m3, "native_actual_m3": actual_m3,
               "native_unmet_m3": unmet_m3, "delivered_total_m3": delivered_total, "payload_sha256": payload_sha, "growth": grow_recs, "pending_days_after": str(new.calendar.pending_days),
               "plan_peak_fresh_bytes": int(plan_bytes["peak_fresh_bytes"]), "plan_stages": dict(needs), "plan_execute": dict(exec_needs)}
    return IntervalCandidate(_SEAL, kind="interval", op=op, water=state2, leaf=new, receipt=receipt, arrays={"delivered_m3": None if fb is None else fb.delivered, "requested_m3": D},
                             _parent_head=weakref.ref(head), _parent_water=weakref.ref(parent), seq_before=int(parent.request_sequence), seq_after=int(state2.request_sequence),
                             generation_before=int(leaf.generation), token=token, _state=[False])


def apply_growth(op: IntervalOperator, head: IntervalHead, *, year_end: bool = False, interval_token: Any) -> IntervalCandidate:
    """The explicit FLUSH at the last growth-enabled endpoint (look-ahead before a transport event or a growth-disabled period): `growth.grow` over the DECLARED pending period sums of the head, the
    LEAF geometry rebuild, `LeafCalendar.flushed` (pending -> 0). No water, no E/T, the request sequence unchanged. A no-pending head is refused (nothing to flush)."""
    _check_op(op)
    token = _token(interval_token)
    _check_head_op(op, head)
    if type(year_end) is not bool:
        raise IntervalError("year_end must be a bool")
    leaf = head.leaf
    if not leaf.calendar.pending_days > 0:
        raise IntervalError("no pending enabled growth credit to flush")
    _check_leaf_caps(op, head, True, False)
    plan_bytes = required_interval_bytes(op, head, growth=True, transpiration=False)
    _admit(op, plan_bytes["execute"]["growth"], "growth")
    if op.cuda:
        _admit_host(op, plan_bytes["host_execute"]["growth"], "growth_host")
    try:
        cal2 = leaf.calendar.flushed(growth_enabled=True)
    except lcal.LeafCalendarError as exc:
        raise IntervalError(str(exc)) from None
    new, rec = _grow(op, leaf, year_end, leaf.calendar.cursor, cal2)
    receipt = {"kind": "growth", "token": token, "profile": op.receipt.profile, "growth": [{"phase": "flush", **rec}], "request_sequence_before": int(head.water.request_sequence),
               "request_sequence_after": int(head.water.request_sequence)}
    return IntervalCandidate(_SEAL, kind="growth", op=op, water=head.water, leaf=new, receipt=receipt, arrays={}, _parent_head=weakref.ref(head), _parent_water=weakref.ref(head.water),
                             seq_before=int(head.water.request_sequence), seq_after=int(head.water.request_sequence), generation_before=int(leaf.generation), token=token, _state=[False])


# ------------------------------------------------------------------ compare-and-adopt (the caller-owned head)
def adopt(op: IntervalOperator, head: IntervalHead, candidate: IntervalCandidate) -> IntervalHead:
    """Replace the head by the candidate's result iff it was derived from THIS head instance, THIS water instance, the CURRENT request sequence/LEAF generation, and was not adopted before.
    A re-run on an old head (alternative equal-scalar child) is refused here; SYRUP's pure API cannot refuse it."""
    _check_op(op)
    _check_head_op(op, head)
    if type(candidate) is not IntervalCandidate or candidate not in _MINTED:
        raise IntervalError("adopt needs a genuine IntervalCandidate")
    if candidate.op is not op:
        raise IntervalError("the candidate was derived under a different IntervalOperator instance (operator/table); refused")
    if candidate._state[0]:
        raise IntervalError("this candidate was already adopted; refused")
    if candidate._parent_head() is not head:
        raise IntervalError("the candidate was derived from a different (old or alternative) head instance; refused")
    if candidate._parent_water() is not head.water:
        raise IntervalError("the candidate's water parent is not the head's current water state instance (an equal-scalar alternative branch is a different parent); refused")
    if int(head.water.request_sequence) != candidate.seq_before:
        raise IntervalError(f"request sequence {head.water.request_sequence} != the candidate's parent sequence {candidate.seq_before}; refused")
    if int(head.leaf.generation) != candidate.generation_before:
        raise IntervalError("the LEAF generation changed since the candidate was derived; refused")
    last = head.watermark[5]
    if candidate.kind == "interval" and last == candidate.token:
        raise IntervalError("the interval token equals the last adopted token; refused (duplicate)")
    candidate._state[0] = True
    n = int(head.watermark[4]) + 1
    wm = (candidate.seq_after, candidate.receipt.get("payload_sha256"), int(candidate.water.revision), candidate.water.digest, n, candidate.token if candidate.kind == "interval" else last)
    return IntervalHead(_SEAL, op=op, water=candidate.water, leaf=candidate.leaf, watermark=wm, receipt=candidate.receipt)
