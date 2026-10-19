"""Thin Dense NumPy CPU adapter over ONE native MAPLE complete bed.

The bed is initialized exactly once (`build_dense_bed`) through the genuine
native sequence `zeros_voxel_column_state -> deposit_surface_mixture_batch ->
initialize_active_layer_from_voxels`, then the genuine ledger, committed
topography, availability and water constructors. It is packaged as SYRUP's
actual `BedState`/`BedContext`. Afterwards no bed is ever rebuilt: every
mapping takes and returns the actual existing voxel / active-layer / ledger /
committed-topography / availability / water objects BY REFERENCE.

Scope (deliberate): genuine dense `VoxelColumnState` only, NumPy host
float64, square grid, full active mask, explicit boundaries, voxel growth
disabled. Windowed/compact beds, CuPy and any other representation refuse
before mutation. No wind stepping, wind finalization, water event, commit,
runner or persistence lives here.

Ownership. Shared bed fields (one object set): voxel_column, active_layer,
ledger, committed_topography, sediment_availability, water; plus the shared
PRESCRIBED landscape fields vegetation and threshold_modifier_state, held by
reference and verified unchanged by digest. Process-private state (wind
mobile cohorts / coupled state / caches; SYRUP graph / storm / velocity /
phase) stays with its caller and is never stored here.

Hashing. `case_record` / `case_sha256` cover the immutable static case only
(native geometry and boundaries, class order and properties, mass
resolution, voxel count, initial time, owned read-only copies of the initial
arrays, the initial availability policy, prescribed vegetation and threshold
modifiers, commit/avalanching specs). Every public transition, scheduler
construction, copy and restore first recomputes the live static inputs
(context objects, geometry and boundary maps, classes, scalar policies,
specs, held arrays, prescribed fields) against that original record. The
evolving bed has a separate `identity_record()` that changes legitimately
whenever native exchange changes it. Stages are additionally bound to the
owning bed instance by a private token that is never hashed. See
docs/dense_integration.md.

Importing this module imports neither numpy nor a native model; all native
imports are lazy.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import hashlib
import json
import math
from types import SimpleNamespace
from typing import Any

from masy.clock import _native_clock, encode_time

__all__ = [
    "BED_IDENTITY_SCHEMA",
    "DENSE_CASE_SCHEMA",
    "SHARED_FIELDS",
    "DenseBed",
    "DenseBedError",
    "DenseBedLimits",
    "DenseStage",
    "build_dense_bed",
    "estimate_initialization_bytes",
]

DENSE_CASE_SCHEMA = "masy.dense_case.v1"
BED_IDENTITY_SCHEMA = "masy.dense_bed_identity.v1"
INITIAL_AVAILABILITY_POLICY = "explicit_fraction_array:initial_split_and_replenishment_fallback"
SHARED_FIELDS = ("voxel_column", "active_layer", "water", "ledger", "committed_topography", "sediment_availability")
# The static BedContext fields, held by identity and re-verified against the case record.
_CONTEXT_FIELDS = ("geometry", "grain_classes", "mass_resolution_kg", "topographic_commit_spec", "avalanching_spec",
                   "initial_available_fraction")
# Conservative initialization peak multipliers (documented in docs/dense_integration.md): voxel-sized
# arrays alive at once (zeros column, deposit working copy, initialization working copy, one kernel
# temporary) and domain-class / domain-sized arrays (request, owned inputs, active layer, availability
# pair, water). Ledger and committed-topography bytes are measured per cell from a 1x1 native probe.
_VOXEL_COPIES = 4
_CLASS_ARRAYS = 8
_CELL_ARRAYS = 4
_FRACTION_SUM_ULPS = 4


class DenseBedError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _native() -> SimpleNamespace:
    """Actual native symbols, imported lazily (exact paths of the frozen sources)."""
    import numpy as np
    from maple.aeolian.scheduler.state import (
        SchedulerPhysicalState,
        SchedulerState,
        carries_coupled_trajectory_history,
        initial_scheduler_state,
    )
    from maple.aeolian.trajectories.drag import TRAJECTORY_ALGORITHM_VERSION
    from maple.core.boundaries import AxisBoundary
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.parameters.geometry import GeometrySpec, validate_geometry
    from maple.core.parameters.grain_classes import GrainClassSet, validate_grain_classes
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.parameters.voxel_growth import DISABLED_VOXEL_GROWTH
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.cohort import MobileCohortState
    from maple.core.types.sediment_availability import (
        SedimentAvailabilityState,
        sediment_availability_state_from_active_layer,
    )
    from maple.core.types.sediment_ledger import SedimentLedgerState, zeros_sediment_ledger_state
    from maple.core.types.threshold_modifiers import ThresholdModifierState, validate_threshold_modifier_state
    from maple.core.types.topographic_commit import CommittedTopographyState, validate_committed_topography_state
    from maple.core.types.vegetation import VegetationState, validate_vegetation_state, vegetation_content_hash
    from maple.core.types.voxel import VoxelColumnState, zeros_voxel_column_state
    from maple.core.types.water import WaterState, water_state_from_depth
    from maple.coupling.sediment_ledger.validation import validate_ledger_state, validate_mass_resolution_kg
    from maple.surface.active_layer.initialization import initialize_active_layer_from_voxels
    from maple.surface.active_layer.validation import (
        check_active_layer_voxel_partition,
        check_voxel_column_geometry_shape,
    )
    from maple.surface.availability.validation import (
        check_sediment_availability_state,
        validate_available_fraction_array,
    )
    from maple.surface.topographic_commit.initialization import (
        initial_committed_topography_state_from_physical_state,
    )
    from maple.surface.voxels.diagnostics import check_voxel_column_state
    from maple.surface.voxels.transfer import deposit_surface_mixture_batch
    from maple.water.validation import validate_water_state
    from maple_syrup.sediment_bed import BedContext, BedState, is_windowed_bed

    return SimpleNamespace(**{name: value for name, value in locals().items()})


def _wrap_native(code: str, call, *args, **kwargs):
    """Run a native validator/constructor; its ValueError becomes a coded DenseBedError (cause kept)."""
    try:
        return call(*args, **kwargs)
    except ValueError as exc:
        if isinstance(exc, DenseBedError):
            raise
        raise DenseBedError(code, f"{getattr(call, '__name__', call)}: {exc}") from exc


def _plain_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise DenseBedError("invalid_limit", f"{name} must be a plain int >= {minimum}, got {value!r}")
    return value


@dataclasses.dataclass(frozen=True)
class DenseBedLimits:
    """Explicit resource caps, checked BEFORE allocation/copy. Resource policy,
    not physics: recorded beside the case, never in its digest."""

    host_memory_cap_bytes: int
    max_voxels_per_column: int
    stage_copy_max_bytes: int

    def __post_init__(self) -> None:
        _plain_int(self.host_memory_cap_bytes, "host_memory_cap_bytes")
        _plain_int(self.max_voxels_per_column, "max_voxels_per_column")
        _plain_int(self.stage_copy_max_bytes, "stage_copy_max_bytes")

    def as_record(self) -> dict[str, int]:
        return dataclasses.asdict(self)


# --- array leaves, digests and canonical static records -----------------------------------------------------------

def _leaves(value: Any, path: str, out: list[tuple[str, Any]]) -> None:
    """Every array-like leaf reachable through dataclasses / sequences / mappings."""
    import numpy as np

    if isinstance(value, np.ndarray):
        out.append((path, value))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _leaves(getattr(value, field.name), f"{path}.{field.name}", out)
    elif hasattr(value, "shape") and hasattr(value, "dtype") and not isinstance(value, np.generic):
        out.append((path, value))  # a non-NumPy (e.g. device) array: refused by _require_host_arrays
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _leaves(item, f"{path}[{index}]", out)
    elif isinstance(value, dict):
        for key, item in value.items():
            _leaves(item, f"{path}[{key!r}]", out)


def _require_host_arrays(value: Any, name: str) -> list[tuple[str, Any]]:
    """Every leaf must be an exact host `numpy.ndarray` (no subclass, no device array)."""
    import numpy as np

    leaves: list[tuple[str, Any]] = []
    _leaves(value, name, leaves)
    for path, array in leaves:
        if type(array) is not np.ndarray:
            raise DenseBedError("unsupported_backend",
                                f"{path} is {type(array).__module__}.{type(array).__qualname__}; only host "
                                "numpy.ndarray (Dense NumPy CPU) is supported")
    return leaves


def _array_digest(array: Any) -> dict[str, Any]:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(memoryview(contiguous).cast("B"))
    return {"array": {"dtype": array.dtype.str, "shape": list(array.shape), "order": "C",
                      "nbytes": int(contiguous.nbytes), "sha256": hasher.hexdigest()}}


def _canonical_static(value: Any, where: str) -> Any:
    """Closed canonical form of a static input; unknown types refuse (fail closed)."""
    import numpy as np

    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise DenseBedError("nonfinite_static_input", f"{where}={value!r}")
        return {"float": value.hex()}
    if isinstance(value, enum.Enum):
        return {"enum": f"{type(value).__module__}.{type(value).__qualname__}",
                "value": _canonical_static(value.value, where)}
    if type(value) is np.ndarray:
        return _array_digest(value)
    if isinstance(value, np.generic):
        return {"numpy_scalar": value.dtype.str, "value": _canonical_static(value.item(), where)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {f.name: _canonical_static(getattr(value, f.name), f"{where}.{f.name}")
                           for f in dataclasses.fields(value)}}
    if isinstance(value, (tuple, list)):
        return {"sequence": type(value).__name__,
                "items": [_canonical_static(item, f"{where}[{i}]") for i, item in enumerate(value)]}
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            if type(key) is not str:
                raise DenseBedError("unhashable_static_input", f"{where} has a non-string key {key!r}")
            items.append([key, _canonical_static(item, f"{where}[{key!r}]")])
        return {"mapping": sorted(items, key=lambda pair: pair[0])}
    if isinstance(value, (set, frozenset)):
        items = [_canonical_static(item, f"{where}{{}}") for item in value]
        return {"set": sorted(items, key=_canonical_json)}
    raise DenseBedError("unhashable_static_input",
                        f"{where} has unsupported type {type(value).__module__}.{type(value).__qualname__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _tree_nbytes(*values: Any) -> int:
    seen: set[int] = set()
    total = 0
    for value in values:
        leaves: list[tuple[str, Any]] = []
        _leaves(value, "", leaves)
        for _path, array in leaves:
            if id(array) not in seen:
                seen.add(id(array))
                total += int(array.nbytes)
    return total


# --- static input checks ---------------------------------------------------------------------------------------------

def _owned_array(n: SimpleNamespace, value: Any, shape: tuple[int, ...], dtype: Any, name: str):
    """Validate an initial input array and return an owned, read-only, C-ordered copy."""
    np = n.np
    if type(value) is not np.ndarray:
        raise DenseBedError("invalid_static_input", f"{name} must be a host numpy.ndarray, got {type(value).__name__}")
    if value.shape != shape:
        raise DenseBedError("invalid_static_input", f"{name} shape {value.shape} != {shape}")
    if value.dtype != dtype:
        raise DenseBedError("invalid_static_input", f"{name} dtype {value.dtype} != {np.dtype(dtype)}")
    owned = np.array(value, dtype=dtype, order="C", copy=True)
    if owned.dtype.kind == "f" and not np.isfinite(owned).all():
        raise DenseBedError("nonfinite_static_input", f"{name} contains non-finite values")
    owned.flags.writeable = False
    return owned


def _check_exact_type(value: Any, expected: type, name: str) -> None:
    if type(value) is not expected:
        raise DenseBedError("unsupported_type",
                            f"{name} must be exactly {expected.__module__}.{expected.__qualname__}, got "
                            f"{type(value).__module__}.{type(value).__qualname__}")


def _probe_per_cell_bytes(n: SimpleNamespace, geometry: Any, n_classes: int, mres: float) -> int:
    """Ledger + committed-topography bytes per cell, measured on a 1x1 empty native probe."""
    try:
        g1 = dataclasses.replace(geometry, nx=1, ny=1)
        column = n.zeros_voxel_column_state(1, 1, 1, n_classes)
        column, active, _ = n.initialize_active_layer_from_voxels(column, g1, mres)
        ledger = n.zeros_sediment_ledger_state(1, 1, n_classes)
        committed = n.initial_committed_topography_state_from_physical_state(column, active, g1, mres)
    except (ValueError, TypeError) as exc:
        raise DenseBedError("memory_probe_failed", f"1x1 native size probe failed ({exc}); refusing") from exc
    return _tree_nbytes(ledger, committed)


def estimate_initialization_bytes(ny: int, nx: int, n_voxels: int, n_classes: int, per_cell_native_bytes: int) -> int:
    """Conservative host-byte bound of one Dense initialization (Python ints, no overflow).
    With `per_cell_native_bytes=0` it is a lower bound of that same estimate from shapes and counts alone."""
    cells = ny * nx
    return 8 * (_VOXEL_COPIES * cells * n_voxels * n_classes + _CLASS_ARRAYS * cells * n_classes
                + _CELL_ARRAYS * cells) + cells * per_cell_native_bytes


def _check_commit_stamp(state: Any, time_s: float) -> None:
    """ClockLedger's rule for a native commit stamp: a native clock value, never after the native time it is
    associated with. Exact comparison: no tolerance, and the stamp is never rewritten."""
    raw = state.committed_topography.last_commit_time_s
    try:
        stamp = _native_clock(raw if isinstance(raw, (bool, int, float)) else float(raw),
                              "committed_topography.last_commit_time_s")
    except (ValueError, TypeError) as exc:
        raise DenseBedError("invalid_commit_time", str(exc)) from exc
    if stamp > time_s:
        raise DenseBedError("future_commit", f"committed_topography.last_commit_time_s {stamp.hex()} is after the "
                            f"native time {time_s.hex()}")


# --- the bed ----------------------------------------------------------------------------------------------------------

_TOKEN = object()


@dataclasses.dataclass(frozen=True)
class DenseStage:
    """A bounded independent copy of the six shared fields at one boundary.
    Not persistence and not history: at most what the caller retains."""

    case_sha256: str
    state: Any
    time_s: float
    identity_sha256: str
    nbytes: int
    _ticket: dict = dataclasses.field(default_factory=dict, compare=False, repr=False)
    # Private owner token of the bed instance that made the stage; never part of any hash.
    _owner: object = dataclasses.field(default=None, compare=False, repr=False)

    @property
    def consumed(self) -> bool:
        return bool(self._ticket.get("consumed"))


class DenseBed:
    """MASY-owned holder of the CURRENT shared bed. Construct only with
    `build_dense_bed`. Mutated only by the explicit `adopt_*` / `restore_stage`
    methods, which validate before replacing anything."""

    def __init__(self, token: object, *, case_record: dict[str, Any], held_inputs: dict[str, Any], context: Any,
                 state: Any, vegetation: Any, threshold_modifier_state: Any, time_s: float, limits: DenseBedLimits,
                 initialization_receipt: dict[str, Any], prescribed_digests: dict[str, str]) -> None:
        if token is not _TOKEN:
            raise DenseBedError("private_constructor", "construct a DenseBed with build_dense_bed")
        self._case_record = case_record
        self._case_sha256 = _sha256(case_record)
        self._held_inputs = held_inputs
        self._context = context
        self._state = state
        self._vegetation = vegetation
        self._threshold_modifier_state = threshold_modifier_state
        self._time_s = time_s
        self._limits = limits
        self._initialization_receipt = initialization_receipt
        self._prescribed_digests = prescribed_digests
        self._context_refs = {name: getattr(context, name) for name in _CONTEXT_FIELDS}
        self._owner = object()  # instance/lineage token for stages; separate from every content digest

    # read-only views ---------------------------------------------------------------------------------------------
    @property
    def case_record(self) -> dict[str, Any]:
        return copy.deepcopy(self._case_record)

    @property
    def case_sha256(self) -> str:
        return self._case_sha256

    @property
    def context(self) -> Any:
        """The one actual SYRUP `BedContext` (geometry/classes/mass resolution/specs reused by every process)."""
        return self._context

    @property
    def state(self) -> Any:
        """The CURRENT actual SYRUP `BedState` (shared objects, by reference)."""
        return self._state

    @property
    def vegetation(self) -> Any:
        return self._vegetation

    @property
    def threshold_modifier_state(self) -> Any:
        return self._threshold_modifier_state

    @property
    def time_s(self) -> float:
        return self._time_s

    @property
    def limits(self) -> DenseBedLimits:
        return self._limits

    @property
    def initialization_receipt(self) -> dict[str, Any]:
        return copy.deepcopy(self._initialization_receipt)

    def vegetation_cover_fraction(self):
        """The canonical prescribed cover FRACTION array (same object; never
        converted). SYRUP's `sediment_physics_step` converts to percent itself."""
        return self._vegetation.raster.cover_fraction

    # verification ------------------------------------------------------------------------------------------------
    def verify_case_inputs(self) -> None:
        """Every live static input and the prescribed fields still match the original case record."""
        self._verify_static(_native())

    def _verify_static(self, n: SimpleNamespace) -> None:
        """Recompute the live static inputs against the original case record; never mutates. Runs first in every
        public transition, scheduler construction, copy and restore."""
        record = self._case_record
        if _sha256(record) != self._case_sha256:
            raise DenseBedError("case_input_mutated", "the held case record no longer reproduces case_sha256")
        context = self._context
        if type(context) is not n.BedContext:
            raise DenseBedError("case_input_mutated", "the held BedContext was replaced")
        for name, ref in self._context_refs.items():
            if getattr(context, name, None) is not ref:
                raise DenseBedError("case_input_mutated", f"context.{name} is no longer the original case object")
        try:
            live = {
                "geometry": _canonical_static(context.geometry, "geometry"),
                "grain_classes": _canonical_static(context.grain_classes, "grain_classes"),
                "class_order": [c.class_id for c in context.grain_classes.classes],
                "mass_resolution_kg": encode_time(context.mass_resolution_kg),
                "topographic_commit_spec": _canonical_static(context.topographic_commit_spec,
                                                             "topographic_commit_spec"),
                "avalanching_spec": _canonical_static(context.avalanching_spec, "avalanching_spec"),
                "initial_available_fraction": _array_digest(context.initial_available_fraction),
            }
        except (ValueError, TypeError, AttributeError) as exc:
            raise DenseBedError("case_input_mutated", f"a live static input no longer canonicalizes ({exc})") from exc
        changed = sorted(key for key, value in live.items() if value != record[key])
        if changed:
            raise DenseBedError("case_input_mutated", f"live static input(s) {changed} differ from the case record")
        for name, array in self._held_inputs.items():
            if array.flags.writeable or record[name] != _array_digest(array):
                raise DenseBedError("case_input_mutated", f"held initial input {name} changed")
        self._check_prescribed(n)

    def _check_prescribed(self, n: SimpleNamespace) -> None:
        current = {
            "vegetation": _sha256(_canonical_static(self._vegetation, "vegetation")),
            "native_vegetation_content_sha256": n.vegetation_content_hash(self._vegetation),
            "threshold_modifier_state": _sha256(_canonical_static(self._threshold_modifier_state,
                                                                  "threshold_modifier_state")),
        }
        if current != self._prescribed_digests:
            changed = sorted(key for key in current if current[key] != self._prescribed_digests[key])
            raise DenseBedError("prescribed_field_changed", f"prescribed field(s) {changed} changed; no growth or "
                                "moisture law exists in this adapter")

    def _validate_shared(self, n: SimpleNamespace, state: Any) -> None:
        """Exact native types, Dense host float64 layout and native validators; never mutates."""
        _check_exact_type(state, n.BedState, "bed state")
        expected_types = {"voxel_column": n.VoxelColumnState, "active_layer": n.ActiveLayerState,
                          "water": n.WaterState, "ledger": n.SedimentLedgerState,
                          "committed_topography": n.CommittedTopographyState,
                          "sediment_availability": n.SedimentAvailabilityState}
        for name in SHARED_FIELDS:
            value = getattr(state, name)
            if value is None:
                raise DenseBedError("missing_shared_field", f"{name} is None; shared fields are never dropped")
            _check_exact_type(value, expected_types[name], name)
            _require_host_arrays(value, name)
        if _wrap_native("unsupported_representation", n.is_windowed_bed, state):
            raise DenseBedError("unsupported_representation", "windowed beds are outside the Dense adapter")
        g, mres = self._context.geometry, self._context.mass_resolution_kg
        nc = len(self._context.grain_classes.classes)
        nv = self._case_record["n_voxels"]
        if state.voxel_column.mass_kg.shape != (g.ny, g.nx, nv, nc) or state.voxel_column.mass_kg.dtype != n.np.float64:
            raise DenseBedError("incompatible_shared_field",
                                f"voxel_column {state.voxel_column.mass_kg.shape}/{state.voxel_column.mass_kg.dtype} "
                                f"!= {(g.ny, g.nx, nv, nc)}/float64")
        _wrap_native("invalid_shared_field", n.check_voxel_column_geometry_shape, state.voxel_column, g)
        _wrap_native("invalid_shared_field", n.check_voxel_column_state, state.voxel_column, g, mres)
        _wrap_native("invalid_shared_field", n.check_active_layer_voxel_partition, state.active_layer,
                     state.voxel_column, g, mres)
        _wrap_native("invalid_shared_field", n.check_sediment_availability_state, state.sediment_availability,
                     state.active_layer, mres)
        _wrap_native("invalid_shared_field", n.validate_water_state, state.water, g.ny, g.nx, nc)
        dims = _wrap_native("invalid_shared_field", n.validate_ledger_state, state.ledger)
        committed_dims = _wrap_native("invalid_shared_field", n.validate_committed_topography_state,
                                      state.committed_topography)
        if tuple(dims) != (g.ny, g.nx, nc) or tuple(committed_dims) != (g.ny, g.nx, nc):
            raise DenseBedError("incompatible_shared_field", f"ledger {dims} / committed {committed_dims} != "
                                f"{(g.ny, g.nx, nc)}")

    def _advance_time(self, time_s: Any) -> float:
        try:
            value = _native_clock(time_s, "time_s")
        except ValueError as exc:
            raise DenseBedError("invalid_time", str(exc)) from exc
        if value < self._time_s:
            raise DenseBedError("time_reversed", f"time {value.hex()} precedes the bed time {self._time_s.hex()}")
        return value

    # mappings ----------------------------------------------------------------------------------------------------
    def fresh_scheduler_state(self, mobile_state: Any, *, time_s: float):
        """A genuine fresh native `SchedulerState` over the shared objects, via
        `initial_scheduler_state` (block 0, no coupled state, no derived caches;
        trajectory semantics derived natively from the empty content).

        `mobile_state` is the caller-owned, process-private wind pool and must
        be genuinely empty; the bed's water mobile load must be exactly zero.
        `time_s` must equal the bed time bit for bit. Prescribed vegetation,
        threshold modifiers, availability and water are passed explicitly,
        never None."""
        n = _native()
        self._verify_static(n)
        if type(time_s) is not float or time_s.hex() != self._time_s.hex():
            raise DenseBedError("time_mismatch", f"time_s {time_s!r} must equal the bed time {self._time_s.hex()}")
        return self._scheduler_over(n, self._state, mobile_state, time_s)

    def stage_scheduler_state(self, stage: DenseStage, mobile_state: Any, *, time_s: float):
        """A genuine fresh native `SchedulerState` over the INDEPENDENT shared
        objects of `stage.state` (never the accepted `self._state`), through the
        same native factory as `fresh_scheduler_state`.

        The stage must be made by this bed instance, unconsumed, staged from the
        CURRENT accepted bed (its recorded identity equals `identity_sha256()`)
        and unchanged since staging. `time_s` must equal the stage and bed time
        bit for bit. The wind pool must be genuinely empty and the staged water
        mobile load exactly zero. The stage is not consumed; nothing here
        mutates the accepted bed or the stage."""
        n = _native()
        self._verify_static(n)
        if type(stage) is not DenseStage or stage.case_sha256 != self._case_sha256:
            raise DenseBedError("foreign_stage", "stage does not belong to this case")
        if stage._owner is not self._owner:
            raise DenseBedError("foreign_stage", "stage was made by a different bed instance")
        if stage.consumed:
            raise DenseBedError("stage_consumed", "a consumed stage cannot back a scheduler")
        if stage.identity_sha256 != self.identity_sha256():
            raise DenseBedError("stale_stage", "the accepted bed changed since this stage was made")
        if _sha256(self._stage_identity(stage)) != stage.identity_sha256:
            raise DenseBedError("stage_mutated", "staged content no longer matches its identity")
        if type(time_s) is not float or time_s.hex() != stage.time_s.hex() or time_s.hex() != self._time_s.hex():
            raise DenseBedError("time_mismatch", f"time_s {time_s!r} must equal the stage and bed time "
                                f"{self._time_s.hex()}")
        self._validate_shared(n, stage.state)
        return self._scheduler_over(n, stage.state, mobile_state, time_s)

    def _scheduler_over(self, n: SimpleNamespace, state: Any, mobile_state: Any, time_s: float):
        """Shared native factory: `initial_scheduler_state` over `state`'s shared objects plus the held
        prescribed fields. Callers have already verified static inputs and the time."""
        _check_exact_type(mobile_state, n.MobileCohortState, "mobile_state")
        _require_host_arrays(mobile_state, "mobile_state")
        _check_commit_stamp(state, time_s)
        probe = n.SchedulerPhysicalState(
            voxel_column=state.voxel_column, active_layer=state.active_layer, mobile_state=mobile_state,
            ledger=state.ledger, committed_topography=state.committed_topography,
            threshold_modifier_state=self._threshold_modifier_state, time_s=time_s)
        if n.carries_coupled_trajectory_history(probe, None):
            raise DenseBedError("mobile_state_not_empty", "a fresh scheduler needs a genuinely empty wind pool")
        if bool(n.np.any(state.water.mobile_mass_by_cell_class_kg != 0.0)):
            raise DenseBedError("water_mobile_not_empty", "water-borne sediment remains; no fresh wind entry")
        scheduler = n.initial_scheduler_state(
            state.voxel_column, state.active_layer, mobile_state, state.ledger, state.committed_topography,
            self._threshold_modifier_state, time_s=time_s, vegetation=self._vegetation,
            sediment_availability=state.sediment_availability, water=state.water)
        physical = scheduler.physical
        if (any(getattr(physical, name) is not getattr(state, name) for name in SHARED_FIELDS)
                or physical.vegetation is not self._vegetation
                or physical.threshold_modifier_state is not self._threshold_modifier_state
                or physical.mobile_state is not mobile_state
                or physical.coupled_trajectory_semantics != n.TRAJECTORY_ALGORITHM_VERSION):
            raise DenseBedError("native_factory_mismatch", "initial_scheduler_state did not carry the shared objects")
        return scheduler

    def adopt_scheduler_physical(self, physical: Any) -> Any:
        """Take the shared fields of a native `SchedulerPhysicalState` back by
        reference (returns the new current `BedState`). The prescribed fields
        must be the held objects; the wind pool must be empty (airborne load is
        resolved natively before any bed handoff, never dropped here)."""
        n = _native()
        self._verify_static(n)
        _check_exact_type(physical, n.SchedulerPhysicalState, "physical")
        if physical.vegetation is not self._vegetation:
            raise DenseBedError("prescribed_field_replaced", "physical.vegetation is not the held prescribed object")
        if physical.threshold_modifier_state is not self._threshold_modifier_state:
            raise DenseBedError("prescribed_field_replaced",
                                "physical.threshold_modifier_state is not the held prescribed object")
        if physical.ledger_reconciliation_baseline is not None:
            raise DenseBedError("unsupported_reconciliation_baseline",
                                "BedState has no slot for ledger_reconciliation_baseline; refusing to drop it")
        _check_exact_type(physical.mobile_state, n.MobileCohortState, "physical.mobile_state")
        if n.carries_coupled_trajectory_history(physical, None):
            raise DenseBedError("mobile_state_not_empty", "airborne wind load remains; no bed handoff")
        candidate = n.BedState(physical.voxel_column, physical.active_layer, physical.water, physical.ledger,
                               physical.committed_topography, physical.sediment_availability)
        return self._adopt(n, candidate, physical.time_s)

    def adopt_bed_state(self, state: Any, *, time_s: float) -> Any:
        """Take an actual SYRUP `BedState` returned by native water code (e.g.
        `DryHandoff.dry_state.bed`) by reference, at its explicit native time."""
        n = _native()
        self._verify_static(n)
        return self._adopt(n, state, time_s)

    def _adopt(self, n: SimpleNamespace, state: Any, time_s: Any) -> Any:
        """Callers have already run `_verify_static`. Every check precedes the single replacement."""
        value = self._advance_time(time_s)
        self._validate_shared(n, state)
        _check_commit_stamp(state, value)
        self._state, self._time_s = state, value
        return state

    # evolving identity and bounded staging -----------------------------------------------------------------------
    def identity_record(self) -> dict[str, Any]:
        """Evolving bed identity (separate from the static case digest)."""
        return {
            "schema": BED_IDENTITY_SCHEMA,
            "case_sha256": self._case_sha256,
            "time_s": encode_time(self._time_s),
            "fields": {name: _canonical_static(getattr(self._state, name), name) for name in SHARED_FIELDS},
            "prescribed": dict(self._prescribed_digests),
        }

    def identity_sha256(self) -> str:
        return _sha256(self.identity_record())

    def stage_copy(self) -> DenseStage:
        """Bounded independent copy of the current shared fields. The byte size
        is checked against `limits.stage_copy_max_bytes` BEFORE copying. The
        stage carries this instance's private owner token."""
        self._verify_static(_native())
        nbytes = _tree_nbytes(*(getattr(self._state, name) for name in SHARED_FIELDS))
        if nbytes > self._limits.stage_copy_max_bytes:
            raise DenseBedError("stage_copy_cap_exceeded",
                                f"{nbytes} bytes exceed stage_copy_max_bytes={self._limits.stage_copy_max_bytes}")
        identity = self.identity_sha256()
        copied = copy.deepcopy(self._state)
        stage = DenseStage(self._case_sha256, copied, self._time_s, identity, nbytes, _owner=self._owner)
        if _sha256(self._stage_identity(stage)) != identity:
            raise DenseBedError("stage_copy_mismatch", "staged copy does not reproduce the bed identity")
        return stage

    def _stage_identity(self, stage: DenseStage) -> dict[str, Any]:
        return {"schema": BED_IDENTITY_SCHEMA, "case_sha256": stage.case_sha256, "time_s": encode_time(stage.time_s),
                "fields": {name: _canonical_static(getattr(stage.state, name), name) for name in SHARED_FIELDS},
                "prescribed": dict(self._prescribed_digests)}

    def restore_stage(self, stage: DenseStage) -> Any:
        """Make a stage made by THIS bed instance the current bed (rollback). A
        stage of another instance, even of an identical case, is foreign. A
        stage is single-use: its arrays become live and it cannot be restored
        twice. Every check precedes consumption."""
        n = _native()
        self._verify_static(n)
        if type(stage) is not DenseStage or stage.case_sha256 != self._case_sha256:
            raise DenseBedError("foreign_stage", "stage does not belong to this case")
        if stage._owner is not self._owner:
            raise DenseBedError("foreign_stage", "stage was made by a different bed instance")
        if stage.consumed:
            raise DenseBedError("stage_consumed", "a stage can be restored only once")
        if _sha256(self._stage_identity(stage)) != stage.identity_sha256:
            raise DenseBedError("stage_mutated", "staged content no longer matches its identity")
        self._validate_shared(n, stage.state)
        _check_commit_stamp(stage.state, stage.time_s)
        stage._ticket["consumed"] = True
        self._state, self._time_s = stage.state, stage.time_s
        return stage.state


def build_dense_bed(*, geometry: Any, grain_classes: Any, mass_resolution_kg: float, n_voxels: int,
                    initial_bed_elevation_m: Any, initial_class_fractions: Any, initial_available_fraction: Any,
                    initial_water_depth_m: Any, vegetation: Any, threshold_modifier_state: Any,
                    topographic_commit_spec: Any, avalanching_spec: Any, active_mask: Any, initial_time_s: float,
                    limits: DenseBedLimits, backend: str = "numpy") -> DenseBed:
    """Validate every static input, check capacities and the host-memory cap,
    then initialize the native bed ONCE. Nothing is allocated at domain size
    before every check has passed; a native failure returns nothing.

    `initial_bed_elevation_m` (ny, nx) is the sediment surface above the voxel
    column base (datum 0), as in the native fixtures; `initial_class_fractions`
    (ny, nx, n_classes) its mass composition in native class order;
    `initial_available_fraction` (ny, nx, n_classes) the explicit initial
    availability policy (initial available/bound split AND the native
    replenishment fallback); `initial_water_depth_m` (ny, nx) the initial
    SYRUP-owned water depth."""
    if backend != "numpy" or type(backend) is not str:
        raise DenseBedError("unsupported_backend", f"backend {backend!r}; only 'numpy' (CPU float64) is supported")
    if type(limits) is not DenseBedLimits:
        raise DenseBedError("invalid_limit", "limits must be a DenseBedLimits")
    n = _native()
    np = n.np
    _check_exact_type(geometry, n.GeometrySpec, "geometry")
    _check_exact_type(grain_classes, n.GrainClassSet, "grain_classes")
    _check_exact_type(geometry.boundary_x, n.AxisBoundary, "geometry.boundary_x")
    _check_exact_type(geometry.boundary_y, n.AxisBoundary, "geometry.boundary_y")
    _check_exact_type(vegetation, n.VegetationState, "vegetation")
    _check_exact_type(threshold_modifier_state, n.ThresholdModifierState, "threshold_modifier_state")
    _check_exact_type(topographic_commit_spec, n.TopographicCommitSpec, "topographic_commit_spec")
    _check_exact_type(avalanching_spec, n.AvalanchingSpec, "avalanching_spec")
    _wrap_native("invalid_grain_classes", n.validate_grain_classes, grain_classes)
    class_ids = tuple(c.class_id for c in grain_classes.classes)
    _wrap_native("invalid_geometry", n.validate_geometry, geometry, set(class_ids))
    mres = _wrap_native("invalid_mass_resolution", n.validate_mass_resolution_kg, mass_resolution_kg)
    if geometry.dx_m != geometry.dy_m:
        raise DenseBedError("non_square_grid", f"dx_m={geometry.dx_m!r} != dy_m={geometry.dy_m!r}")
    if geometry.voxel_growth != n.DISABLED_VOXEL_GROWTH:
        raise DenseBedError("unsupported_voxel_growth", "runtime voxel growth is outside the Dense adapter scope")
    nv = _plain_int(n_voxels, "n_voxels")
    if nv > limits.max_voxels_per_column:
        raise DenseBedError("voxel_capacity_exceeded", f"n_voxels={nv} > max_voxels_per_column="
                            f"{limits.max_voxels_per_column}")
    try:
        t0 = _native_clock(initial_time_s, "initial_time_s")
    except ValueError as exc:
        raise DenseBedError("invalid_time", str(exc)) from exc
    ny, nx, nc = geometry.ny, geometry.nx, len(class_ids)

    # Host memory, cheap first: a lower bound of the conservative estimate from validated shapes and counts only,
    # before any domain-sized owned copy, value scan, native validator scan, probe or native domain allocation.
    lower_bound = estimate_initialization_bytes(ny, nx, nv, nc, 0)
    if lower_bound > limits.host_memory_cap_bytes:
        raise DenseBedError("host_memory_cap_exceeded",
                            f"shape lower bound {lower_bound} bytes exceeds host_memory_cap_bytes="
                            f"{limits.host_memory_cap_bytes}")
    # Then refine with the bounded 1x1 native probe; the full conservative estimate is checked before any copy.
    per_cell = _probe_per_cell_bytes(n, geometry, nc, mres)
    estimate = estimate_initialization_bytes(ny, nx, nv, nc, per_cell)
    if estimate > limits.host_memory_cap_bytes:
        raise DenseBedError("host_memory_cap_exceeded",
                            f"estimated {estimate} bytes exceed host_memory_cap_bytes={limits.host_memory_cap_bytes}")

    # Static inputs: native validators and owned read-only copies (case hash inputs).
    _require_host_arrays(vegetation, "vegetation")
    _require_host_arrays(threshold_modifier_state, "threshold_modifier_state")
    _wrap_native("invalid_vegetation", n.validate_vegetation_state, vegetation, ny, nx, geometry=geometry)
    _wrap_native("invalid_threshold_modifiers", n.validate_threshold_modifier_state, threshold_modifier_state, ny, nx)
    if type(active_mask) is not np.ndarray or active_mask.dtype != np.bool_ or active_mask.shape != (ny, nx) \
            or not bool(active_mask.all()):
        raise DenseBedError("unsupported_active_mask", "only a full (ny, nx) bool active mask is supported")
    elevation = _owned_array(n, initial_bed_elevation_m, (ny, nx), np.float64, "initial_bed_elevation_m")
    fractions = _owned_array(n, initial_class_fractions, (ny, nx, nc), np.float64, "initial_class_fractions")
    available = _owned_array(n, initial_available_fraction, (ny, nx, nc), np.float64, "initial_available_fraction")
    depth = _owned_array(n, initial_water_depth_m, (ny, nx), np.float64, "initial_water_depth_m")
    violations = n.validate_available_fraction_array(available, (ny, nx, nc))
    if violations:
        raise DenseBedError("invalid_static_input", "; ".join(violations))
    if bool((fractions < 0).any() or (fractions > 1).any()):
        raise DenseBedError("invalid_static_input", "initial_class_fractions must lie in [0, 1]")
    if bool((np.abs(fractions.sum(axis=-1) - 1.0) > _FRACTION_SUM_ULPS * nc * np.finfo(np.float64).eps).any()):
        raise DenseBedError("invalid_static_input", "initial_class_fractions must sum to 1 per cell")
    if bool((depth < 0).any()) or bool(np.signbit(depth).any()):
        raise DenseBedError("invalid_static_input", "initial_water_depth_m must be >= +0")
    if bool((elevation < geometry.active_layer_thickness_m).any()):
        raise DenseBedError("insufficient_bed", "initial_bed_elevation_m must reach the active-layer thickness")
    cell_mass_capacity = nv * geometry.voxel_dz_m * geometry.bulk_density_kg_m3 * geometry.dx_m * geometry.dy_m
    cell_mass = elevation * geometry.bulk_density_kg_m3 * geometry.dx_m * geometry.dy_m
    if bool((cell_mass > cell_mass_capacity).any()):
        raise DenseBedError("voxel_capacity_exceeded",
                            f"initial bed mass exceeds the allocated column capacity ({nv} voxels of "
                            f"{geometry.voxel_dz_m} m)")

    case_record = {
        "schema": DENSE_CASE_SCHEMA,
        "backend": "numpy",
        "dtype": "float64",
        "representation": "dense_voxel_column",
        "geometry": _canonical_static(geometry, "geometry"),
        "grain_classes": _canonical_static(grain_classes, "grain_classes"),
        "class_order": list(class_ids),
        "mass_resolution_kg": encode_time(mres),
        "n_voxels": nv,
        "initial_time_s": encode_time(t0),
        "initial_bed_elevation_m": _array_digest(elevation),
        "initial_class_fractions": _array_digest(fractions),
        "initial_available_fraction": _array_digest(available),
        "initial_availability_policy": INITIAL_AVAILABILITY_POLICY,
        "initial_water_depth_m": _array_digest(depth),
        "active_mask": "full",
        "vegetation": _canonical_static(vegetation, "vegetation"),
        "native_vegetation_content_sha256": n.vegetation_content_hash(vegetation),
        "threshold_modifier_state": _canonical_static(threshold_modifier_state, "threshold_modifier_state"),
        "topographic_commit_spec": _canonical_static(topographic_commit_spec, "topographic_commit_spec"),
        "avalanching_spec": _canonical_static(avalanching_spec, "avalanching_spec"),
        "depth_update_rule": "constant_depth",
    }
    prescribed = {
        "vegetation": _sha256(case_record["vegetation"]),
        "native_vegetation_content_sha256": case_record["native_vegetation_content_sha256"],
        "threshold_modifier_state": _sha256(case_record["threshold_modifier_state"]),
    }

    # Genuine one-time native initialization (all local until every check passes).
    request = cell_mass[:, :, None] * fractions
    column = n.zeros_voxel_column_state(ny, nx, nv, nc)
    column, deposit = _wrap_native("native_initialization_failed", n.deposit_surface_mixture_batch, column, request,
                                   geometry, mres)
    if bool(np.any(deposit.limited)):
        raise DenseBedError("deposit_limited", "native deposit reported a resolved shortfall")
    column, active, init = _wrap_native("native_initialization_failed", n.initialize_active_layer_from_voxels,
                                        column, geometry, mres)
    if bool(np.any(init.underfilled_mask)):
        raise DenseBedError("active_layer_underfilled", "native active-layer initialization is underfilled")
    ledger = n.zeros_sediment_ledger_state(ny, nx, nc)
    committed = _wrap_native("native_initialization_failed", n.initial_committed_topography_state_from_physical_state,
                             column, active, geometry, mres, initial_time_s=t0)
    availability = n.sediment_availability_state_from_active_layer(active.mass_kg, available)
    water = n.water_state_from_depth(np.array(depth, copy=True), nc)
    state = n.BedState(column, active, water, ledger, committed, availability)
    context = n.BedContext(geometry, grain_classes, mres, topographic_commit_spec, avalanching_spec, available)

    inventory = column.mass_kg.sum(axis=(0, 1, 2)) + active.mass_kg.sum(axis=(0, 1))
    receipt = {
        "native_sequence": ["zeros_voxel_column_state", "deposit_surface_mixture_batch",
                            "initialize_active_layer_from_voxels", "zeros_sediment_ledger_state",
                            "initial_committed_topography_state_from_physical_state",
                            "sediment_availability_state_from_active_layer", "water_state_from_depth"],
        "requested_by_class_kg": [encode_time(float(v)) for v in request.sum(axis=(0, 1))],
        "deposit_numerical_residual_total_kg": encode_time(float(np.sum(deposit.numerical_residual_total_mass_kg))),
        "init_numerical_residual_total_kg": encode_time(float(np.sum(init.numerical_residual_total_mass_kg))),
        "bed_inventory_by_class_kg": [encode_time(float(v)) for v in inventory],
        "host_memory_lower_bound_bytes": lower_bound,
        "estimated_initialization_bytes": estimate,
        "measured_per_cell_ledger_committed_bytes": per_cell,
        "limits": limits.as_record(),
    }
    bed = DenseBed(_TOKEN, case_record=case_record,
                   held_inputs={"initial_bed_elevation_m": elevation, "initial_class_fractions": fractions,
                                "initial_available_fraction": available, "initial_water_depth_m": depth},
                   context=context, state=state, vegetation=vegetation,
                   threshold_modifier_state=threshold_modifier_state, time_s=t0, limits=limits,
                   initialization_receipt=receipt, prescribed_digests=prescribed)
    bed._verify_static(n)
    bed._validate_shared(n, state)
    _check_commit_stamp(state, t0)
    return bed
