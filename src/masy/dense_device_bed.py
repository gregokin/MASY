"""Unwired private Dense device ownership, not a solver or GPU admission profile.

Accepted arrays never escape: native callers receive independent work copies,
then publication copies their candidate into TWO new private buffers. Exact
byte comparisons happen only at ownership boundaries, with compact native host
packets. Native physical validators remain authoritative. No full bed download,
CPU fallback, device selection, runner, persistence or source-profile claim.

The six-root byte reservation is not a pool/driver fit proof. Comparison and
caller-declared native/reduction scratch reservations are additional; a future
approved worker must qualify their actual peaks. Python private attributes are
an API contract, not a security boundary against hostile pointer access.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import re
import sys
import threading
from types import ModuleType
from typing import Any

from masy.clock import _native_clock
from masy.dense_bed import (
    SHARED_FIELDS,
    DenseBedError,
    _canonical_static,
    _check_commit_stamp,
    _native,
)

__all__ = ["DeviceBedError", "DeviceBedLimits", "DeviceBedStage", "DeviceDenseBedOwner"]


class DeviceBedError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise DeviceBedError("invalid_limit", f"{name} must be plain int >= {minimum}")
    return value


@dataclasses.dataclass(frozen=True)
class DeviceBedLimits:
    """Structural reservations only; scratch sizes are declared, not measured."""

    max_bed_bytes: int
    max_work_bytes: int
    native_validation_scratch_bytes: int
    comparison_reduction_scratch_bytes: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _integer(getattr(self, field.name), field.name)


def _work_reservation(inventory, largest_leaf_bytes, limits):
    """One structural formula shared with the gated test's host preflight."""
    compare = (
        int(largest_leaf_bytes)
        + 8 * (6 * inventory["n_leaves"] + 16)
        + limits.comparison_reduction_scratch_bytes
    )
    return int(compare), int(
        6 * inventory["transfer_bytes"]
        + compare
        + limits.native_validation_scratch_bytes
    )


def _geometry_metadata(n, geometry, n_classes):
    """Close the small host metadata BEFORE canonicalization or deepcopy.

    Native walkers intentionally ignore undeclared attributes. Letting those
    reach deepcopy could copy hidden arrays/resources outside the bed budget.
    The 64KiB metadata cap is scope admission, not a native physics decision.
    """
    from maple.core.boundaries import BoundaryKind
    from maple.core.parameters.voxel_growth import VoxelGrowthSpec

    def holder(value, cls, names):
        if type(value) is not cls:
            raise DeviceBedError(
                "invalid_geometry", "exact native geometry metadata holders required"
            )
        if tuple((f.name, f.init) for f in dataclasses.fields(value)) != tuple(
            (key, True) for key in names
        ):
            raise DeviceBedError(
                "invalid_geometry", "native geometry field/init schema changed"
            )
        attributes = vars(value)
        if len(attributes) != len(names) or any(key not in attributes for key in names):
            raise DeviceBedError(
                "invalid_geometry", "undeclared geometry metadata attributes"
            )
        if "__post_init__" in vars(cls):
            raise DeviceBedError(
                "invalid_geometry", "native geometry reconstruction changed"
            )

    holder(
        geometry,
        n.GeometrySpec,
        (
            "nx",
            "ny",
            "dx_m",
            "dy_m",
            "boundary_x",
            "boundary_y",
            "voxel_dz_m",
            "bulk_density_kg_m3",
            "active_layer_thickness_m",
            "voxel_growth",
        ),
    )
    holder(
        geometry.voxel_growth,
        VoxelGrowthSpec,
        ("enabled", "chunk_levels", "max_levels", "safety_fraction"),
    )
    for key in ("nx", "ny"):
        _integer(getattr(geometry, key), f"geometry.{key}")
    for key in (
        "dx_m",
        "dy_m",
        "voxel_dz_m",
        "bulk_density_kg_m3",
        "active_layer_thickness_m",
    ):
        if type(getattr(geometry, key)) not in (int, float):
            raise DeviceBedError(
                "invalid_geometry", "plain numeric geometry metadata required"
            )
    growth = geometry.voxel_growth
    if (
        type(growth.enabled) is not bool
        or type(growth.chunk_levels) is not int
        or (growth.max_levels is not None and type(growth.max_levels) is not int)
        or type(growth.safety_fraction) not in (int, float)
    ):
        raise DeviceBedError(
            "invalid_geometry", "plain native growth metadata types required"
        )
    values = [
        getattr(geometry, f.name)
        for f in dataclasses.fields(geometry)
        if f.name not in ("boundary_x", "boundary_y", "voxel_growth")
    ]
    values += [
        getattr(geometry.voxel_growth, f.name)
        for f in dataclasses.fields(geometry.voxel_growth)
    ]
    for value in values:
        if type(value) not in (int, float, bool, type(None)):
            raise DeviceBedError(
                "invalid_geometry", "plain host scalar geometry metadata required"
            )
        if (type(value) is float and not math.isfinite(value)) or (
            type(value) is int and value.bit_length() > 63
        ):
            raise DeviceBedError(
                "invalid_geometry", "nonfinite or unbounded geometry metadata"
            )
    metadata_bytes = 4096  # Fixed scalar/field canonicalization allowance.
    for boundary in (geometry.boundary_x, geometry.boundary_y):
        holder(boundary, n.AxisBoundary, ("kind", "inflow_flux_kg_m_s"))
        if (
            type(boundary.kind) is not BoundaryKind
            or type(boundary.kind.value) is not str
        ):
            raise DeviceBedError("invalid_geometry", "native boundary enum required")
        flux = boundary.inflow_flux_kg_m_s
        if flux is None:
            continue
        if type(flux) is not dict or len(flux) > min(n_classes, (64 << 10) // 128):
            raise DeviceBedError(
                "invalid_geometry", "bounded plain boundary metadata mapping required"
            )
        for key, value in flux.items():
            if (
                type(key) is not str
                or len(key) > 256
                or type(value) not in (int, float)
            ):
                raise DeviceBedError(
                    "invalid_geometry", "bounded scalar boundary mapping required"
                )
            if (type(value) is float and not math.isfinite(value)) or (
                type(value) is int and value.bit_length() > 63
            ):
                raise DeviceBedError(
                    "invalid_geometry", "nonfinite or unbounded boundary metadata"
                )
            metadata_bytes += len(key.encode("utf-8")) + 128
        if metadata_bytes > 64 << 10:
            raise DeviceBedError(
                "invalid_geometry", "host geometry metadata exceeds 64KiB scope"
            )
    return _canonical_static(geometry, "geometry")


@dataclasses.dataclass(frozen=True)
class _Accepted:
    state: Any
    mirror: Any
    time_s: float
    generation: int


class DeviceBedStage:
    """One work borrow. Consuming/discarding releases the owner's work refs."""

    __slots__ = ("_consumed", "_generation", "_owner", "_state", "_time")

    def __init__(self, owner, generation, time_s, state):
        self._owner, self._generation, self._time = owner, generation, time_s
        self._state, self._consumed = state, False

    @property
    def state(self):
        if self._consumed:
            raise DeviceBedError("ticket_consumed", "work borrow is consumed")
        return self._state

    @property
    def time_s(self):
        return self._time

    @property
    def consumed(self):
        return self._consumed


class _Prepared:
    __slots__ = ("_accepted", "_consumed", "_generation", "_owner", "_stage")

    def __init__(self, owner, generation, stage, accepted):
        self._owner, self._generation = owner, generation
        self._stage, self._accepted, self._consumed = stage, accepted, False


class _CupyOps:
    """Production adapter; only an already-imported genuine CuPy is accepted."""

    def __init__(self, cp):
        import numpy
        from maple.core.backend import arrays_share_memory, read_transfer_counters
        from maple_syrup.event_backend import host_packet

        self.xp, self.array_type, self.host_np = cp, cp.ndarray, numpy
        self.overlap, self._counters, self._host_packet = (
            arrays_share_memory,
            read_transfer_counters,
            host_packet,
        )

    def current_device(self):
        return int(self.xp.cuda.runtime.getDevice())

    def current_stream(self):
        return int(self.xp.cuda.get_current_stream().ptr)

    def device(self, array):
        return int(array.device.id)

    def clone(self, array):
        return array.copy(order="C")

    def raw_equal(self, left, right):
        # reshape is a view for the pre-admitted C-contiguous arrays, even 0d.
        return self.xp.all(
            left.reshape(-1).view(self.xp.uint8)
            == right.reshape(-1).view(self.xp.uint8)
        )

    def packet(self, flags):
        return self._host_packet(self.xp, flags)

    def vector_packet(self, vector):
        return self._host_packet(self.xp, [], vector)

    def counters(self):
        return dataclasses.asdict(self._counters())


def _schemas(n):
    ledger = (
        "pending_bed_mass_change_kg",
        "pending_compensation_kg",
        "process_totals_kg",
        "process_totals_compensation_kg",
        "numerical_residual_positive_kg",
        "numerical_residual_positive_compensation_kg",
        "numerical_residual_negative_kg",
        "numerical_residual_negative_compensation_kg",
        "actual_mass_positive_kg",
        "actual_mass_positive_compensation_kg",
        "actual_mass_negative_kg",
        "actual_mass_negative_compensation_kg",
        "largest_residual_kg",
        "n_physical_touches",
    )
    return {
        n.BedState: dict(
            zip(
                SHARED_FIELDS,
                (
                    n.VoxelColumnState,
                    n.ActiveLayerState,
                    n.WaterState,
                    n.SedimentLedgerState,
                    n.CommittedTopographyState,
                    n.SedimentAvailabilityState,
                ),
                strict=True,
            )
        ),
        n.VoxelColumnState: {"mass_kg": "array"},
        n.ActiveLayerState: {"mass_kg": "array"},
        n.WaterState: dict.fromkeys(
            ("depth_m", "mobile_mass_by_cell_class_kg"), "array"
        ),
        n.SedimentLedgerState: {
            key: float if key == "largest_residual_kg" else "array" for key in ledger
        },
        n.CommittedTopographyState: dict(
            zip(
                (
                    "reference_elevation_m",
                    "elevation_offset_m",
                    "committed_active_layer_mass_by_class_kg",
                    "committed_voxel_mass_by_class_kg",
                    "last_commit_time_s",
                    "commit_count",
                ),
                ("array", "array", "array", "array", float, int),
                strict=True,
            )
        ),
        n.SedimentAvailabilityState: dict.fromkeys(
            ("available_mass_kg", "bound_mass_kg"), "array"
        ),
    }


def _role_shape(role, geometry, n_voxels, n_classes, n_processes):
    """Native Dense leaf shape of one closed-schema role (pure metadata)."""
    g, nc = geometry, n_classes
    cell, classes = (g.ny, g.nx), (g.ny, g.nx, nc)
    if role == "voxel_column.mass_kg":
        return (*cell, n_voxels, nc)
    if role in (
        "water.depth_m",
        "committed_topography.reference_elevation_m",
        "committed_topography.elevation_offset_m",
    ):
        return cell
    if role.startswith("ledger."):
        key = role.split(".")[1]
        if key in ("process_totals_kg", "process_totals_compensation_kg"):
            return n_processes, nc
        if key not in (
            "pending_bed_mass_change_kg",
            "pending_compensation_kg",
            "n_physical_touches",
        ):
            return n_processes, *classes
    return classes


def _closed_tree(
    schemas, np, value, kind, path, arrays, scalars, *, array_type, shape_of, device_of=None, device=None
):
    """Pure closed walk of the six native holders: exact holder types, field/init
    schemas, no undeclared attributes, exact array type/layout/shape/dtype and
    exact finite host scalars. `device_of=None` is the host (pre-copy) walk."""
    if kind == "array":
        if type(value) is not array_type:
            raise DeviceBedError(
                "unsupported_backend", f"{path}: exact resident array required"
            )
        if not value.flags.c_contiguous or value.dtype.hasobject:
            raise DeviceBedError(
                "unsupported_layout",
                f"{path}: contiguous non-object array required",
            )
        if device_of is not None and device_of(value) != device:
            raise DeviceBedError(
                "wrong_device", f"{path}: leaf is not on pinned device"
            )
        if tuple(value.shape) != shape_of(path):
            raise DeviceBedError(
                "invalid_shape", f"{path}: shape differs from native Dense schema"
            )
        touches = path == "ledger.n_physical_touches"
        if (touches and value.dtype.kind not in "iu") or (
            not touches and value.dtype != np.float64
        ):
            raise DeviceBedError(
                "invalid_dtype", f"{path}: native Dense dtype required"
            )
        arrays.append((path, value))
        return
    if kind in (float, int):
        if type(value) is not kind or (kind is float and not math.isfinite(value)):
            raise DeviceBedError(
                "invalid_scalar",
                f"{path}: exact finite native host scalar required",
            )
        scalars.append((path, value.hex() if kind is float else value))
        return
    if type(value) is not kind:
        raise DeviceBedError(
            "unsupported_holder", f"{path}: exact native holder required"
        )
    schema = schemas[kind]
    if tuple((f.name, f.init) for f in dataclasses.fields(value)) != tuple(
        (key, True) for key in schema
    ):
        raise DeviceBedError(
            "unsupported_schema", f"{path}: native field/init schema changed"
        )
    if "__post_init__" in vars(kind) or set(vars(value)) != set(schema):
        raise DeviceBedError(
            "unsupported_schema",
            f"{path}: native attribute/reconstruction schema changed",
        )
    for key, child_kind in schema.items():
        _closed_tree(
            schemas,
            np,
            getattr(value, key),
            child_kind,
            f"{path}.{key}" if path else key,
            arrays,
            scalars,
            array_type=array_type,
            shape_of=shape_of,
            device_of=device_of,
            device=device,
        )


def _host_bed_schema(state, *, geometry, n_voxels, n_classes):
    """Private pre-copy admission of a genuine CPU source bed: the SAME closed
    six-holder walk the device owner applies, with NumPy as the explicit
    expected array type, bounded geometry metadata first, and native leaf
    discovery required to equal the closed schema (no untracked payload).
    Pure metadata: no copy, clone, allocation or backend injection."""
    from maple.core.backend import array_leaves
    from maple.core.types.sediment_ledger import N_LEDGER_PROCESSES

    n = _native()
    nv, nc = _integer(n_voxels, "n_voxels"), _integer(n_classes, "n_classes")
    _geometry_metadata(n, geometry, nc)
    arrays, scalars = [], []
    _closed_tree(
        _schemas(n),
        n.np,
        state,
        n.BedState,
        "",
        arrays,
        scalars,
        array_type=n.np.ndarray,
        shape_of=lambda role: _role_shape(role, geometry, nv, nc, N_LEDGER_PROCESSES),
    )
    if [id(a) for a in array_leaves(state)] != [id(a) for _, a in arrays]:
        raise DeviceBedError(
            "unsupported_schema", "native leaf discovery disagrees with closed schema"
        )
    for i, (_, left) in enumerate(arrays):
        for _, right in arrays[i + 1 :]:
            if left is not right and n.np.shares_memory(left, right):
                raise DeviceBedError(
                    "overlapping_views", "distinct overlapping leaves unsupported"
                )
    return (
        tuple((p, a.dtype.str, tuple(a.shape)) for p, a in arrays),
        tuple(scalars),
    )


class DeviceDenseBedOwner:
    """Single-thread/current-stream, same-device private six-field owner.

    Validation inputs here are not complete solver/static-case admission.
    Public construction accepts existing genuine device arrays only. Private
    `_create` is a CPU semantic test seam and is never selected by that API.
    """

    def __init__(self):
        raise DeviceBedError("constructor", "use from_device_bed")

    @classmethod
    def from_device_bed(
        cls,
        state,
        *,
        geometry,
        mass_resolution_kg,
        n_voxels,
        n_classes,
        case_sha256,
        time_s,
        limits,
    ):
        cp = sys.modules.get("cupy")
        if (
            not isinstance(cp, ModuleType)
            or getattr(cp, "__name__", None) != "cupy"
            or getattr(getattr(cp, "ndarray", None), "__module__", "").split(".")[0]
            != "cupy"
        ):
            raise DeviceBedError(
                "unsupported_backend",
                "existing genuine CuPy arrays required; no fallback",
            )
        return cls._create(
            state,
            geometry=geometry,
            mass_resolution_kg=mass_resolution_kg,
            n_voxels=n_voxels,
            n_classes=n_classes,
            case_sha256=case_sha256,
            time_s=time_s,
            limits=limits,
            _ops=_CupyOps(cp),
        )

    @classmethod
    def _create(
        cls,
        state,
        *,
        geometry,
        mass_resolution_kg,
        n_voxels,
        n_classes,
        case_sha256,
        time_s,
        limits,
        _ops,
    ):
        from maple.core.backend import array_leaves
        from maple.core.backend.arrays import _map_tree, array_leaf_inventory
        from maple.core.types.sediment_ledger import N_LEDGER_PROCESSES

        if cls is not DeviceDenseBedOwner:
            raise DeviceBedError("unsupported_holder", "owner subclasses unsupported")
        if type(limits) is not DeviceBedLimits or set(vars(limits)) != {
            f.name for f in dataclasses.fields(limits)
        }:
            raise DeviceBedError(
                "invalid_limit", "exact declared DeviceBedLimits required"
            )
        limits.__post_init__()  # Frozen objects can still be forcibly changed.
        n = _native()
        nv, nc = _integer(n_voxels, "n_voxels"), _integer(n_classes, "n_classes")
        geometry_record = _geometry_metadata(n, geometry, nc)
        # Native validation below owns geometry/physical meaning; these are scope/layout restrictions.
        if (
            geometry.dx_m != geometry.dy_m
            or geometry.voxel_growth != n.DISABLED_VOXEL_GROWTH
        ):
            raise DeviceBedError(
                "unsupported_geometry", "square fixed-capacity Dense geometry required"
            )
        _integer(geometry.ny, "geometry.ny")
        _integer(geometry.nx, "geometry.nx")
        if type(mass_resolution_kg) is not float:
            raise DeviceBedError("invalid_mass_resolution", "plain float required")
        try:
            n.validate_mass_resolution_kg(mass_resolution_kg)
        except ValueError as exc:
            raise DeviceBedError("invalid_mass_resolution", str(exc)) from exc
        if (
            type(case_sha256) is not str
            or re.fullmatch("[0-9a-f]{64}", case_sha256) is None
        ):
            raise DeviceBedError(
                "invalid_case",
                "case_sha256 must be a SHA256 label, not source verification",
            )
        if type(time_s) not in (int, float):
            raise DeviceBedError(
                "invalid_time", "plain native int/float clock required"
            )
        try:
            t = _native_clock(time_s, "time_s")
        except ValueError as exc:
            raise DeviceBedError("invalid_time", str(exc)) from exc
        owner = object.__new__(cls)
        owner._n, owner._ops = n, _ops
        owner._leaves, owner._map, owner._inventory = (
            array_leaves,
            _map_tree,
            array_leaf_inventory,
        )
        owner._schemas = _schemas(n)
        owner._geometry, owner._geometry_record = (
            copy.deepcopy(geometry),
            geometry_record,
        )
        owner._nv, owner._nc = nv, nc
        owner._processes, owner._mres, owner._limits = (
            N_LEDGER_PROCESSES,
            mass_resolution_kg,
            limits,
        )
        owner._limit_record = tuple(
            getattr(limits, f.name) for f in dataclasses.fields(limits)
        )
        owner._case, owner._token, owner._active = case_sha256, object(), None
        owner._thread = threading.get_ident()
        owner._device, owner._stream = _ops.current_device(), _ops.current_stream()
        owner._counts = {
            "d2d_enqueued_arrays": 0,
            "d2d_enqueued_bytes": 0,
            "clone_attempts": 0,
            "packet_attempts": 0,
            "packets_completed": 0,
        }
        owner._native_before = _ops.counters()
        owner._admit(state)
        owner._validate(state, t)
        accepted, mirror = owner._clone(state), owner._clone(state)
        owner._disjoint(accepted, state, mirror)
        owner._disjoint(mirror, state)
        owner._compare((state, accepted), (accepted, mirror))
        owner._accepted = _Accepted(accepted, mirror, t, 0)
        return owner

    def _scope(self):
        if threading.get_ident() != self._thread:
            raise DeviceBedError("wrong_thread", "owner is single-threaded")
        if (
            self._ops.current_device() != self._device
            or self._ops.current_stream() != self._stream
        ):
            raise DeviceBedError(
                "device_stream_changed", "current device/stream must remain pinned"
            )
        if _canonical_static(self._geometry, "geometry") != self._geometry_record:
            raise DeviceBedError(
                "geometry_changed", "private validation geometry changed"
            )

    def _shape(self, role):
        return _role_shape(role, self._geometry, self._nv, self._nc, self._processes)

    def _closed(self, value, kind, path, arrays, scalars):
        _closed_tree(
            self._schemas,
            self._n.np,
            value,
            kind,
            path,
            arrays,
            scalars,
            array_type=self._ops.array_type,
            shape_of=self._shape,
            device_of=self._ops.device,
            device=self._device,
        )

    def _describe(self, state):
        arrays, scalars = [], []
        self._closed(state, self._n.BedState, "", arrays, scalars)
        # Discovery/placement are native: repeated refs retain their actual semantics.
        leaves = self._leaves(state)
        if [id(a) for a in leaves] != [id(a) for _, a in arrays]:
            raise DeviceBedError(
                "unsupported_schema",
                "native leaf discovery disagrees with closed schema",
            )
        return arrays, tuple(scalars)

    def _overlap(self, left, right):
        # Also guard this boundary: never reach native strided CuPy fallback.
        if not left.flags.c_contiguous or not right.flags.c_contiguous:
            raise DeviceBedError(
                "unsupported_layout",
                "contiguity required before native overlap predicate",
            )
        return self._ops.overlap(left, right)

    def _admit(self, state):
        self._scope()
        if type(self._limits) is not DeviceBedLimits or set(vars(self._limits)) != {
            f.name for f in dataclasses.fields(DeviceBedLimits)
        }:
            raise DeviceBedError("invalid_limit", "declared limits schema changed")
        self._limits.__post_init__()
        if (
            tuple(
                getattr(self._limits, f.name) for f in dataclasses.fields(self._limits)
            )
            != self._limit_record
        ):
            raise DeviceBedError(
                "invalid_limit", "resource reservation changed after ownership began"
            )
        arrays, _ = self._describe(state)
        for i, (_, left) in enumerate(arrays):
            for _, right in arrays[i + 1 :]:
                if left is not right and self._overlap(left, right):
                    raise DeviceBedError(
                        "overlapping_views", "distinct overlapping leaves unsupported"
                    )
        inventory = self._inventory(state)
        footprint, largest = (
            inventory["transfer_bytes"],
            max(a.nbytes for _, a in arrays),
        )
        compare, required = _work_reservation(inventory, largest, self._limits)
        if (
            footprint > self._limits.max_bed_bytes
            or required > self._limits.max_work_bytes
        ):
            raise DeviceBedError(
                "work_bytes_cap",
                "six roots plus comparison/native scratch exceed reservation",
            )
        return inventory, int(compare), int(required)

    def _validate(self, state, time_s):
        n, g, mres = self._n, self._geometry, self._mres
        try:
            if n.is_windowed_bed(state):
                raise DeviceBedError("unsupported_holder", "windowed bed refused")
            n.check_voxel_column_geometry_shape(state.voxel_column, g)
            n.check_voxel_column_state(state.voxel_column, g, mres)
            n.check_active_layer_voxel_partition(
                state.active_layer, state.voxel_column, g, mres
            )
            n.check_sediment_availability_state(
                state.sediment_availability, state.active_layer, mres
            )
            n.validate_water_state(state.water, g.ny, g.nx, self._nc)
            dims = n.validate_ledger_state(state.ledger)
            committed = n.validate_committed_topography_state(
                state.committed_topography
            )
            if tuple(dims) != (g.ny, g.nx, self._nc) or tuple(committed) != tuple(dims):
                raise DeviceBedError(
                    "invalid_shape",
                    "native validators returned incompatible dimensions",
                )
            _check_commit_stamp(state, time_s)
        except DenseBedError as exc:
            raise DeviceBedError(exc.code, exc.detail) from exc
        except ValueError as exc:
            if isinstance(exc, DeviceBedError):
                raise
            raise DeviceBedError("invalid_native_state", str(exc)) from exc

    def _clone(self, state):
        memo = {}  # Strong source references, scoped to this single reconstruction.

        def leaf(array):
            found = memo.get(id(array))
            if found is not None:
                return found[1]
            self._counts["clone_attempts"] += 1
            moved = self._ops.clone(array)
            self._counts["d2d_enqueued_arrays"] += 1
            self._counts["d2d_enqueued_bytes"] += int(array.nbytes)
            memo[id(array)] = (array, moved)
            return moved

        return self._map(state, leaf)

    def _disjoint(self, state, *others):
        left, _ = self._describe(state)
        for other in others:
            right, _ = self._describe(other)
            if any(self._overlap(a, b) for _, a in left for _, b in right):
                raise DeviceBedError(
                    "private_alias", "work/candidate/private copy allocations overlap"
                )

    def _compare(self, *pairs):
        flags = []
        for left, right in pairs:
            a, sa = self._describe(left)
            b, sb = self._describe(right)
            if sa != sb:
                raise DeviceBedError("integrity_failed", "host scalar bytes differ")
            flags += [
                self._ops.raw_equal(x, y) for (_, x), (_, y) in zip(a, b, strict=True)
            ]
        self._counts["packet_attempts"] += 1
        packet = self._ops.packet(flags)
        self._counts["packets_completed"] += 1
        if packet.shape != (len(flags),) or not all(float(v) == 1.0 for v in packet):
            raise DeviceBedError("integrity_failed", "raw device bytes differ")

    def verify(self):
        self._scope()
        self._admit(self._accepted.state)
        self._admit(self._accepted.mirror)
        self._compare((self._accepted.state, self._accepted.mirror))

    def begin_stage(self):
        self._scope()
        if self._active is not None:
            raise DeviceBedError(
                "transaction_active", "one active work transaction allowed"
            )
        self.verify()
        record = self._accepted
        work = self._clone(record.state)
        self._disjoint(work, record.state, record.mirror)
        self._compare((record.state, work))
        stage = DeviceBedStage(self._token, record.generation, record.time_s, work)
        self._active = stage
        return stage

    def _ticket(self, ticket, kind):
        self._scope()
        if type(ticket) is not kind or ticket._owner is not self._token:
            raise DeviceBedError(
                "foreign_ticket", "ticket belongs to a different owner/type"
            )
        if ticket._consumed:
            raise DeviceBedError("ticket_consumed", "ticket was consumed")
        if (
            type(ticket._generation) is not int
            or self._active is not ticket
            or ticket._generation != self._accepted.generation
        ):
            raise DeviceBedError(
                "stale_ticket", "ticket is not current active generation"
            )

    def prepare_publish(self, stage, candidate, *, time_s):
        self._ticket(stage, DeviceBedStage)
        if type(time_s) not in (int, float):
            raise DeviceBedError(
                "invalid_time", "plain native int/float clock required"
            )
        try:
            t = _native_clock(time_s, "time_s")
        except ValueError as exc:
            raise DeviceBedError("invalid_time", str(exc)) from exc
        if (
            t < self._accepted.time_s
            or stage._time.hex() != self._accepted.time_s.hex()
        ):
            raise DeviceBedError(
                "time_reversed", "native time reversed or stage clock changed"
            )
        self._admit(candidate)
        self._disjoint(candidate, self._accepted.state, self._accepted.mirror)
        self._validate(candidate, t)
        new, mirror = self._clone(candidate), self._clone(candidate)
        self._disjoint(
            new,
            candidate,
            stage.state,
            self._accepted.state,
            self._accepted.mirror,
            mirror,
        )
        self._disjoint(
            mirror, candidate, stage.state, self._accepted.state, self._accepted.mirror
        )
        self._compare(
            (self._accepted.state, self._accepted.mirror),
            (candidate, new),
            (new, mirror),
        )
        # New pair is synchronized, private and never exposed/written after this proof.
        prepared = _Prepared(
            self._token,
            self._accepted.generation,
            stage,
            _Accepted(new, mirror, t, self._accepted.generation + 1),
        )
        self._active = prepared
        return prepared

    def publish(self, prepared):
        self._check_publish(prepared)
        self._swap(prepared)

    def _check_publish(self, prepared):
        """Private: every fallible publication check, no assignment."""
        self._ticket(prepared, _Prepared)
        self.verify()  # Every fallible check precedes the accepted-record replacement.

    def _swap(self, prepared):
        """Private: assignment-only half of `publish`; callers ran `_check_publish`."""
        record = prepared._accepted
        stage = prepared._stage
        self._accepted = record
        # Only assignments follow. No allocation, copying, validation or packet.
        stage._state, stage._consumed = None, True
        prepared._accepted, prepared._stage, prepared._consumed = None, None, True
        self._active = None

    def discard(self, ticket):
        kind = DeviceBedStage if type(ticket) is DeviceBedStage else _Prepared
        self._ticket(ticket, kind)
        if kind is _Prepared:
            ticket._stage._state, ticket._stage._consumed = None, True
            ticket._stage, ticket._accepted = None, None
        else:
            ticket._state = None
        ticket._consumed, self._active = True, None

    def _discard_active(self):
        """Private cleanup: discard whichever ticket (stage or prepared) is active."""
        if self._active is not None:
            self.discard(self._active)

    def _compact(self, fn, *, max_values):
        """Private trusted read: native reduction `fn` over the accepted state,
        returned only as ONE compact host packet of at most `max_values` values.
        The accepted (writable) state itself never leaves the owner."""
        _integer(max_values, "max_values")
        self._scope()
        value = fn(self._accepted.state)
        shape = tuple(getattr(value, "shape", ()))
        if len(shape) > 1 or (shape and shape[0] > max_values):
            raise DeviceBedError("spatial_payload", "compact reductions are class vectors or scalars only")
        self._counts["packet_attempts"] += 1
        packet = self._ops.vector_packet(value)
        self._counts["packets_completed"] += 1
        return packet

    @property
    def time_s(self):
        return self._accepted.time_s

    @property
    def generation(self):
        return self._accepted.generation

    def evidence(self):
        self._scope()
        inventory, compare, required = self._admit(self._accepted.state)
        now = self._ops.counters()
        return {
            "schema": "masy.device_bed_integrity.v1",
            "case_sha256_label": self._case,
            "generation": self.generation,
            "native_time_hex": self.time_s.hex(),
            "visible_device_ordinal": self._device,
            "stream_pointer": self._stream,
            "native_placement_inventory": dict(inventory),
            "bed_root_reservation_count": 6,
            "comparison_reservation_bytes": compare,
            "required_structural_work_bytes": required,
            "declared_native_scratch_bytes": self._limits.native_validation_scratch_bytes,
            "host_geometry_metadata_cap_bytes": 64 << 10,
            "copy_and_packet_counters": dict(self._counts),
            "native_transfer_delta": {
                key: now[key] - value for key, value in self._native_before.items()
            },
            "source_binding_verified_by_owner": False,
            "physical_uuid_verified_by_owner": False,
            "whole_worker_memory_fit": False,
            "gpu_transport_qualified": False,
            "restart_supported": False,
            "limits": dataclasses.asdict(self._limits),
            "evidence_scope": "metadata only; evidence() does not assert a fresh content comparison",
            "scope": "private ownership only; source backing, pools, driver and native peak unqualified",
        }
