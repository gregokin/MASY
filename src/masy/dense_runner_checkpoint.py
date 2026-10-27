"""Passive default-Dense accepted-boundary component; no live adoption or store.

This source-bound prototype prepares an inert private owner. It does not authorize
an event, publish HEAD, migrate sources, or supply a second sediment authority.
Native22/15 are the only physical loader. Original34 owns original typed inputs.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import math
import os
import re
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from masy import dense_bed as db
from masy import dense_checkpoint as bridge
from masy import dense_original_inputs as oi
from masy import dense_runner as dr
from masy.clock import ClockLedger, decode_time, encode_time
from masy.dense_schedule import check_resolved_event, resolve_event

_NATIVE_FLAGS = {
    "RoutingGraph": {"frozen": True, "eq": False, "slots": False},
    "TransportNetwork": {"frozen": True, "eq": False, "slots": False},
    "PhysicsGrid": {"frozen": True, "eq": False, "slots": False},
    "TerrainReference": {"frozen": True, "eq": True, "slots": False},
    "SedimentEventState": {"frozen": True, "eq": False, "slots": False},
}

SCHEMA = "masy.dense_runner_boundary.v1"
PROFILE = "default_dense_numpy_accepted_v1"
_NATIVE_FORMAT = "maple-exact-physical-checkpoint/1"
_ORIGINAL_FORMAT = oi.SCHEMA
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_FRACTION = re.compile(r"(?:0|-?[1-9][0-9]{0,2047})/[1-9][0-9]{0,2047}\Z")
_ARRAY_KEYS = (
    "name",
    "dtype",
    "shape",
    "data_bytes",
    "data_sha256",
    "file_bytes",
    "sha256",
)
_ROLES = (
    "current_soil",
    "total_export",
    "total_input",
    "total_bound",
    "total_internal_terminal_deposition",
    "routing_source",
    "routing_velocity",
)


class BoundaryCheckpointError(ValueError):
    def __init__(self, code, detail):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


class BoundaryPublicationUncertain(OSError):
    """A seal may be visible; publication/durability must be inspected by its owner."""


def _fail(code, detail):
    raise BoundaryCheckpointError(code, detail)


@dataclasses.dataclass(frozen=True)
class BoundaryImageLimits:
    max_numeric_bytes: int = 1 << 20
    max_metadata_bytes: int = 4 << 20
    max_serialized_bytes: int = 6 << 20
    max_header_bytes: int = 4096
    max_nodes: int = 100000
    max_depth: int = 24
    max_files: int = 7
    max_cells: int = 64
    max_classes: int = 6
    max_voxels: int = 16
    max_cohorts: int = 16000
    max_events: int = 8
    max_receipt_bytes: int = 256 << 10
    max_receipts_bytes: int = 2 << 20
    native_max_bytes: int = 64 << 20
    max_global_raw_bytes: int = 512 << 20
    max_cache_bytes: int = 1 << 20

    def __post_init__(self):
        ceiling = (
            1 << 20,
            4 << 20,
            6 << 20,
            4096,
            100000,
            24,
            7,
            64,
            6,
            16,
            16000,
            8,
            256 << 10,
            2 << 20,
            64 << 20,
            512 << 20,
            1 << 20,
        )
        if type(self) is not BoundaryImageLimits or set(vars(self)) != {
            f.name for f in dataclasses.fields(self)
        }:
            _fail("limit_layout", "exact closed BoundaryImageLimits required")
        for field, cap in zip(dataclasses.fields(self), ceiling, strict=True):
            value = getattr(self, field.name)
            if type(value) is not int or not 1 <= value <= cap:
                _fail(
                    "limit_cap", f"{field.name} must be a plain positive int <= {cap}"
                )


_DEFAULT = BoundaryImageLimits()


def _keys(value, names, where):
    try:
        oi._keys(value, names, where)
    except oi.OriginalInputsError as exc:
        _fail(exc.code, exc.detail)


def _sha(value):
    return hashlib.sha256(oi._json(value)).hexdigest()


def _digest(value, where):
    if type(value) is not str or _SHA.fullmatch(value) is None:
        _fail("identity", f"{where}: lowercase SHA256 required")
    return value


def _int(value, cap, where):
    if type(value) is not int or not 0 <= value <= cap:
        _fail("scalar_kind", f"{where}: plain bounded nonnegative int required")
    return value


def _float(value, where):
    if type(value) is not float or not math.isfinite(value):
        _fail("scalar_kind", f"{where}: plain finite float required")
    return value


def _same(a, b, limits):
    try:
        same = oi._same(
            a,
            b,
            oi.OriginalPackLimits(
                max_cells=limits.max_cells,
                max_classes=limits.max_classes,
                max_voxels=limits.max_voxels,
                max_cohorts=limits.max_cohorts,
                max_events=limits.max_events,
            ),
        )
        if not same:
            _fail("content_mismatch", "exact type/bytes/value/order differs")
    except oi.OriginalInputsError as exc:
        _fail("content_mismatch", exc.detail)


def _array(value, shape, where, *, readonly=False):
    import numpy as np

    if (
        type(value) is not np.ndarray
        or value.dtype.str != "<f8"
        or value.shape != tuple(shape)
        or not value.flags.c_contiguous
        or (readonly and value.flags.writeable)
        or not bool(np.isfinite(value).all())
    ):
        _fail("array_layout", f"{where}: exact finite C float64{tuple(shape)} required")
    return value


def _layout(value, cls, names):
    try:
        oi._layout(value, cls, names)
    except oi.OriginalInputsError as exc:
        _fail(exc.code, exc.detail)


@dataclasses.dataclass(frozen=True, eq=False, slots=True)
class _RestoredRouting:
    graph: Any
    terrain: Any
    network: Any
    grid: Any
    source: Any
    velocity: Any
    phase: Any
    logical_identity_sha256: str


@dataclasses.dataclass(frozen=True, eq=False, slots=True)
class _PassiveLoad:
    """Private immediate-loader handoff. Nominal discipline, not a security proof."""

    fields: dict
    evidence: dict


class _PreparedBoundary:
    """Inert private owner. There is deliberately no step/adopt/inner-owner API."""

    __slots__ = ("__evidence", "__owner")

    def __init__(self, owner, evidence):
        self.__owner, self.__evidence = owner, copy.deepcopy(evidence)

    def diagnostics(self):
        return copy.deepcopy(self.__evidence)


def _native_types():
    from maple_syrup.routing import RoutingGraph
    from maple_syrup.sediment_bed import TerrainReference
    from maple_syrup.sediment_event import SedimentEventState
    from maple_syrup.sediment_physics import PhysicsGrid
    from maple_syrup.sediment_transport import TransportNetwork

    return (
        RoutingGraph,
        TransportNetwork,
        PhysicsGrid,
        TerrainReference,
        SedimentEventState,
    )


def _cache_admission(graph, network, grid, terrain, shape, nc, limits):
    """Closed type/shape/raw-byte admission BEFORE canonical traversal/factories."""
    import numpy as np

    ny, nx = shape
    cells = ny * nx
    _int(cells, limits.max_cells, "cells")
    _int(nc, limits.max_classes, "classes")
    classes = _native_types()
    for cls in classes:
        expected = ROUTING_CONTRACT[cls.__name__]["fields"]
        flags = _NATIVE_FLAGS[cls.__name__]
        params = cls.__dataclass_params__
        if (
            tuple(f.name for f in dataclasses.fields(cls)) != tuple(expected)
            or getattr(cls, "__slots__", None) is not None
            or params.frozen is not flags["frozen"]
            or params.eq is not flags["eq"]
        ):
            _fail(
                "restored_routing_layout_changed",
                f"native {cls.__name__} declaration changed",
            )
    event = classes[-1]
    if any(not f.init for f in dataclasses.fields(event)) or hasattr(
        event, "__post_init__"
    ):
        _fail("restored_routing_layout_changed", "native event init protocol changed")
    # Establish dimension symbols only after inspecting exact plain arrays, not array conversions.
    for value, cls in zip((graph, network, grid, terrain), classes[:4], strict=True):
        _layout(value, cls, ROUTING_CONTRACT[cls.__name__]["fields"])
    symbols = {"ny": ny, "nx": nx, "cells": cells}
    for key, value in (
        ("n_active", graph.level_order_host),
        ("n_x_faces", network.cells_x),
        ("n_y_faces", network.cells_y),
    ):
        if type(value) is not np.ndarray or value.ndim != 1 or value.size > cells:
            _fail(
                "restored_routing_layout_changed",
                "native variable dimensions exceed profile",
            )
        symbols[key] = value.size
    # Original34 terrain shape policy, bounded again at each projection reader before hashing.
    terrain_symbols = {"ny": ny, "nx": nx, "ny+2": ny + 2, "nx+2": nx + 2}
    for name, rule in ROUTING_CONTRACT["TerrainReference"]["per_field"].items():
        item, kind = getattr(terrain, name), rule["kind"]
        if "ndarray" in kind:
            expected_shape = tuple(terrain_symbols[v] for v in rule["shape"])
            if (
                type(item) is not np.ndarray
                or item.dtype != np.dtype(rule["dtype"])
                or item.shape != expected_shape
                or not item.flags.c_contiguous
            ):
                _fail(
                    "restored_routing_layout_changed",
                    "bounded original terrain layout required",
                )
        elif kind == "plain_bool":
            if type(item) is not bool:
                _fail("restored_routing_layout_changed", "terrain boolean kind")
        elif item is not None or kind != "None_or_plain_float":
            _float(item, name)
    used = 0
    for value, cls in zip((graph, network, grid), classes[:3], strict=True):
        for name, rule in ROUTING_CONTRACT[cls.__name__]["per_field"].items():
            item, kind = getattr(value, name), rule["kind"]
            if "ndarray" in kind:
                if item is None and kind.startswith("None_or_"):
                    continue
                expected_shape = tuple(
                    symbols[x] if type(x) is str else x for x in rule["shape"]
                )
                if (
                    type(item) is not np.ndarray
                    or item.dtype != np.dtype(rule["dtype"])
                    or item.shape != expected_shape
                    or not item.flags.c_contiguous
                    or item.flags.writeable
                    or item.ndim > 2
                    or item.size > 4 * cells
                ):
                    _fail(
                        "restored_routing_layout_changed",
                        f"{cls.__name__}.{name}: cache layout",
                    )
                used += item.nbytes
                if used > limits.max_cache_bytes:
                    _fail("array_cap", "routing cache byte cap before fingerprint")
            elif kind == "actual_numpy_module":
                if item is not np:
                    _fail("restored_routing_layout_changed", "native namespace differs")
            elif kind == "plain_float":
                _float(item, name)
            elif kind == "plain_int":
                _int(item, cells, name)
            elif kind == "plain_str":
                if type(item) is not str or len(item.encode()) > 4096:
                    _fail("restored_routing_layout_changed", "routing string kind/size")
            elif kind.startswith("tuple["):
                if (
                    type(item) is not tuple
                    or len(item) > cells + 1
                    or any(type(v) is not int or v < 0 or v > cells for v in item)
                ):
                    _fail("restored_routing_layout_changed", "routing tuple layout")
            else:
                _fail(
                    "restored_routing_layout_changed", "unknown native scalar contract"
                )
    if graph.shape != shape or network.shape != shape or grid.shape != shape:
        _fail("restored_routing_layout_changed", "cache geometry differs")
    _digest(graph.input_sha256, "graph input")
    _digest(network.graph_input_sha256, "network graph input")
    if (
        network.n_cells != cells
        or network.n_active != symbols["n_active"]
        or network.n_terminal > network.n_active
        or network.graph_input_sha256 != graph.input_sha256
        or network.dx_m.hex() != graph.dx_m.hex()
        or terrain.dx_m.hex() != graph.dx_m.hex()
        or grid.cell_area_m2.hex() != (graph.dx_m * graph.dx_m).hex()
    ):
        _fail("restored_routing_layout_changed", "native scalar/cache bindings differ")
    from maple_syrup.sediment_bed import graph_policy_flags

    try:
        policy = graph_policy_flags(graph.policy)
    except ValueError as exc:
        _fail("restored_routing_layout_changed", str(exc))
    if policy != {
        name: getattr(terrain, name)
        for name in ("allow_masked_nodata", "allow_pit_storage", "allow_flat_storage")
    }:
        _fail(
            "restored_routing_layout_changed", "graph/original terrain policy differs"
        )
    # Native optional values are distinct; never accept native helper's zero/None equivalence alone.
    if (
        (graph.flat_storage is None) != (terrain.allow_flat_storage is False)
        or ((network.terminal_flat is None) != (network.n_terminal == 0))
        or ((network.terminal_host is None) != (network.n_terminal == 0))
        or ((grid.terminal is None) != (not bool(np.any(graph.pit_storage))))
    ):
        _fail(
            "restored_routing_layout_changed", "native optional cache coupling differs"
        )


def _cache_fingerprint(value):
    import numpy as np

    return db._sha256(dr._canonical_input(np, value, "routing"))


def _routing_identity(terrain_hash, source_hash, graph, network, grid):
    return _sha(
        {
            "terrain": terrain_hash,
            "source": source_hash,
            "graph": graph,
            "network": network,
            "grid": grid,
        }
    )


def _routing_record(held, water, shape, nc, limits):
    import numpy as np

    if (
        type(shape) is not tuple
        or len(shape) != 2
        or any(type(d) is not int or d < 1 for d in shape)
        or math.prod(shape) > limits.max_cells
        or type(nc) is not int
        or not 1 <= nc <= limits.max_classes
        or 8 * math.prod(shape) * (1 + nc) > limits.max_numeric_bytes
    ):
        _fail("profile_cap", "routing dimensions/numeric budget before traversal")
    _layout(
        water,
        dr.DenseWaterInputs,
        (
            "terrain",
            "column",
            "field",
            "vegetation_cover_fraction",
            "sediment",
            "initial_soil_water_m",
        ),
    )
    if type(held) is _RestoredRouting:
        if (
            not _RestoredRouting.__dataclass_params__.frozen
            or _RestoredRouting.__dataclass_params__.eq
            or tuple(f.name for f in dataclasses.fields(held))
            != _RestoredRouting.__slots__
            or any(not hasattr(held, name) for name in _RestoredRouting.__slots__)
        ):
            _fail("restored_routing_layout_changed", "private projection slots missing")
        _digest(held.logical_identity_sha256, "projection identity")
        velocity, phase = held.velocity, held.phase
    else:
        if type(held) is not SimpleNamespace or set(vars(held)) != {
            "graph",
            "terrain",
            "network",
            "grid",
            "source",
            "template",
        }:
            _fail(
                "restored_routing_layout_changed",
                "closed legacy held namespace required",
            )
        cls = _native_types()[-1]
        _layout(held.template, cls, ROUTING_CONTRACT["SedimentEventState"]["fields"])
        for a, b in (
            (held.graph, held.template.graph),
            (held.network, held.template.network),
            (held.grid, held.template.grid),
            (held.source, held.template.routing_source_elevation_m),
            (held.terrain, held.template.terrain),
        ):
            if a is not b:
                _fail("restored_routing_layout_changed", "held/template aliases differ")
        velocity, phase = held.template.sediment_velocity_m_s, held.template.phase
    if held.terrain is not water.terrain:
        _fail(
            "restored_routing_layout_changed",
            "held terrain is not original water terrain",
        )
    if type(velocity) is not np.ndarray:
        _fail(
            "transport_memory_unsupported",
            "retained sediment velocity is not a host NumPy array",
        )
    _array(
        held.source, shape, "routing source", readonly=(type(held) is _RestoredRouting)
    )
    _array(
        velocity,
        (*shape, nc),
        "routing velocity",
        readonly=(type(held) is _RestoredRouting),
    )
    _cache_admission(
        held.graph, held.network, held.grid, held.terrain, shape, nc, limits
    )
    if phase is not None or bool(np.any(velocity != 0.0)):
        _fail(
            "transport_memory_not_empty",
            "upwind entry requires phase None and zero velocity",
        )
    terrain_hash = _cache_fingerprint(water.terrain)
    source_hash = db._sha256(db._array_digest(held.source))
    hashes = tuple(_cache_fingerprint(x) for x in (held.graph, held.network, held.grid))
    identity = _routing_identity(terrain_hash, source_hash, *hashes)
    if type(held) is _RestoredRouting and held.logical_identity_sha256 != identity:
        _fail(
            "restored_routing_changed",
            "retained projection content changed outside runner",
        )
    return dict(
        zip(
            ROUTING_FIELDS,
            (
                terrain_hash,
                "routing_source",
                source_hash,
                held.graph.input_sha256,
                *hashes,
                "routing_velocity",
                None,
                identity,
            ),
            strict=True,
        )
    )


def _check_projection(held, water, shape, nc, limits=_DEFAULT):
    try:
        return _routing_record(held, water, shape, nc, limits)
    except BoundaryCheckpointError as exc:
        raise dr._Stop("refused", exc.code, exc.detail, guarantee=dr._EXTERNAL) from exc


def _restored_memory(held, water, shape, nc):
    if type(held) is not _RestoredRouting:
        raise dr._Stop(
            "refused",
            "restored_routing_layout_changed",
            "private projection required",
            guarantee=dr._EXTERNAL,
        )
    _check_projection(held, water, shape, nc)
    return ("sediment_velocity_m_s",), held.velocity.nbytes


def _restored_guard(held, water, shape, nc):
    _check_projection(held, water, shape, nc)


def _restored_entry(
    held, boundary, graph, network, grid, soil, t, n, *, water, shape, nc
):
    if type(held) is not _RestoredRouting:
        raise dr._Stop(
            "refused",
            "restored_routing_layout_changed",
            "private projection required",
            guarantee=dr._EXTERNAL,
        )
    _check_projection(held, water, shape, nc)
    cls = _native_types()[-1]
    memory = n.np.array(held.velocity, copy=True)
    return cls(
        storm=n.initial_state(graph, boundary.state.water.depth_m, soil, t_s=t),
        bed=boundary.state,
        sediment_velocity_m_s=memory,
        graph=graph,
        network=network,
        grid=grid,
        terrain=held.terrain,
        phase=None,
        routing_source_elevation_m=boundary.routing_source_m,
    )


# Finite, source-derived schema. Not a generic native object decoder.
RUNNER_FIELDS = (
    "_accepted_bed",
    "_accepted_private",
    "_active_evidence",
    "_bed",
    "_control",
    "_inventory0",
    "_inventory_elements",
    "_ledger",
    "_limits",
    "_n_accepted",
    "_n_classes",
    "_next",
    "_pool",
    "_production",
    "_receipts",
    "_routing",
    "_schedule",
    "_soil",
    "_static_parts",
    "_static_parts_sha256",
    "_static_record",
    "_static_sha256",
    "_stop_receipt",
    "_totals",
    "_water",
    "_wind",
)
ROOT_FIELDS = (
    "schema",
    "profile",
    "source_identity",
    "environment_sha256",
    "originals",
    "genesis",
    "physical",
    "state",
    "totals",
    "ledger",
    "receipts",
    "routing",
    "shared_resources",
    "arrays",
    "manifest_sha256",
)
STATE_FIELDS = (
    "next_index",
    "n_accepted",
    "n_classes",
    "inventory_elements",
    "native_time",
    "accepted_bed_sha256",
    "accepted_private_sha256",
    "receipt_chain_sha256",
)
TOTALS_FIELDS = (
    "export",
    "input",
    "bound",
    "internal_terminal_deposition",
    "water",
    "n_entry_reroutes",
    "n_event_reroutes",
)
WATER_FIELDS = (
    "rain_m3",
    "drainage_m3",
    "storm_export_m3",
    "dry_surface_removed_m3",
    "dry_soil_removed_m3",
)
ROUTING_FIELDS = (
    "terrain_original_sha256",
    "source_array",
    "source_sha256",
    "graph_input_sha256",
    "graph_fingerprint",
    "network_fingerprint",
    "grid_fingerprint",
    "velocity_array",
    "phase",
    "routing_identity_sha256",
)
SHARED_RESOURCE_FIELDS = (
    "schema",
    "version",
    "profile",
    "prescribed_identity_sha256",
    "soil_array",
    "update_cursor",
    "update_receipts",
)
ARTIFACT_REF_FIELDS = ("format", "relative_path", "manifest_file_sha256", "identity")
ROUTING_CONTRACT = {
    "RoutingGraph": {
        "fields": [
            "shape",
            "dx_m",
            "aspect",
            "slope",
            "slope_before_edge_rule",
            "edge_rule_applied",
            "friction_factor",
            "active",
            "outlet",
            "receiver",
            "level",
            "level_order_host",
            "level_bounds",
            "conveyance",
            "active_flat",
            "outlet_flat",
            "level_order",
            "conveyance_lo",
            "donor_position",
            "donor_mask",
            "input_sha256",
            "xp",
            "pit_storage",
            "policy",
            "flat_storage",
        ],
        "per_field": {
            "shape": {"kind": "tuple[plain_int,plain_int]"},
            "dx_m": {"kind": "plain_float"},
            "aspect": {
                "kind": "exact_numpy.ndarray",
                "dtype": "int8",
                "shape": ["ny", "nx"],
            },
            "slope": {
                "kind": "exact_numpy.ndarray",
                "dtype": "float64",
                "shape": ["ny", "nx"],
            },
            "slope_before_edge_rule": {
                "kind": "exact_numpy.ndarray",
                "dtype": "float64",
                "shape": ["ny", "nx"],
            },
            "edge_rule_applied": {
                "kind": "exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["ny", "nx"],
            },
            "friction_factor": {
                "kind": "exact_numpy.ndarray",
                "dtype": "float64",
                "shape": ["ny", "nx"],
            },
            "active": {
                "kind": "exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["ny", "nx"],
            },
            "outlet": {
                "kind": "exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["ny", "nx"],
            },
            "receiver": {
                "kind": "exact_numpy.ndarray",
                "dtype": "int64",
                "shape": ["ny", "nx"],
            },
            "level": {
                "kind": "exact_numpy.ndarray",
                "dtype": "int32",
                "shape": ["ny", "nx"],
            },
            "level_order_host": {
                "kind": "exact_numpy.ndarray",
                "dtype": "int64",
                "shape": ["n_active"],
            },
            "level_bounds": {"kind": "tuple[plain_int,...]"},
            "conveyance": {
                "kind": "exact_numpy.ndarray",
                "dtype": "float64",
                "shape": ["cells"],
            },
            "active_flat": {
                "kind": "exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["cells"],
            },
            "outlet_flat": {
                "kind": "exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["cells"],
            },
            "level_order": {
                "kind": "exact_numpy.ndarray",
                "dtype": "int64",
                "shape": ["n_active"],
            },
            "conveyance_lo": {
                "kind": "exact_numpy.ndarray",
                "dtype": "float64",
                "shape": ["n_active"],
            },
            "donor_position": {
                "kind": "exact_numpy.ndarray",
                "dtype": "int64",
                "shape": [4, "n_active"],
            },
            "donor_mask": {
                "kind": "exact_numpy.ndarray",
                "dtype": "bool",
                "shape": [4, "n_active"],
            },
            "input_sha256": {"kind": "plain_str"},
            "xp": {"kind": "actual_numpy_module"},
            "pit_storage": {
                "kind": "exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["ny", "nx"],
            },
            "policy": {"kind": "plain_str"},
            "flat_storage": {
                "kind": "None_or_exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["ny", "nx"],
            },
        },
    },
    "TransportNetwork": {
        "fields": [
            "shape",
            "dx_m",
            "n_cells",
            "n_active",
            "receiver_index",
            "outlet_flat",
            "active_flat",
            "cells_x",
            "rows_x",
            "faces_x",
            "sign_x",
            "cells_y",
            "faces_y",
            "cols_y",
            "sign_y",
            "receiver_host",
            "outlet_host",
            "active_host",
            "graph_input_sha256",
            "xp",
            "terminal_flat",
            "terminal_host",
            "n_terminal",
        ],
        "per_field": {
            "shape": {"kind": "tuple[plain_int,plain_int]"},
            "dx_m": {"kind": "plain_float"},
            "n_cells": {"kind": "plain_int"},
            "n_active": {"kind": "plain_int"},
            "receiver_index": {
                "kind": "exact_numpy.ndarray",
                "dtype": "int64",
                "shape": ["cells"],
            },
            "outlet_flat": {
                "kind": "exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["cells"],
            },
            "active_flat": {
                "kind": "exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["cells"],
            },
            "cells_x": {
                "kind": "exact_numpy.ndarray",
                "dtype": "int64",
                "shape": ["n_x_faces"],
            },
            "rows_x": {
                "kind": "exact_numpy.ndarray",
                "dtype": "int64",
                "shape": ["n_x_faces"],
            },
            "faces_x": {
                "kind": "exact_numpy.ndarray",
                "dtype": "int64",
                "shape": ["n_x_faces"],
            },
            "sign_x": {
                "kind": "exact_numpy.ndarray",
                "dtype": "float64",
                "shape": ["n_x_faces"],
            },
            "cells_y": {
                "kind": "exact_numpy.ndarray",
                "dtype": "int64",
                "shape": ["n_y_faces"],
            },
            "faces_y": {
                "kind": "exact_numpy.ndarray",
                "dtype": "int64",
                "shape": ["n_y_faces"],
            },
            "cols_y": {
                "kind": "exact_numpy.ndarray",
                "dtype": "int64",
                "shape": ["n_y_faces"],
            },
            "sign_y": {
                "kind": "exact_numpy.ndarray",
                "dtype": "float64",
                "shape": ["n_y_faces"],
            },
            "receiver_host": {
                "kind": "exact_numpy.ndarray",
                "dtype": "int64",
                "shape": ["ny", "nx"],
            },
            "outlet_host": {
                "kind": "exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["ny", "nx"],
            },
            "active_host": {
                "kind": "exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["ny", "nx"],
            },
            "graph_input_sha256": {"kind": "plain_str"},
            "xp": {"kind": "actual_numpy_module"},
            "terminal_flat": {
                "kind": "None_or_exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["cells"],
            },
            "terminal_host": {
                "kind": "None_or_exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["ny", "nx"],
            },
            "n_terminal": {"kind": "plain_int"},
        },
    },
    "PhysicsGrid": {
        "fields": ["shape", "cell_area_m2", "slope", "active", "xp", "terminal"],
        "per_field": {
            "shape": {"kind": "tuple[plain_int,plain_int]"},
            "cell_area_m2": {"kind": "plain_float"},
            "slope": {
                "kind": "exact_numpy.ndarray",
                "dtype": "float64",
                "shape": ["ny", "nx"],
            },
            "active": {
                "kind": "exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["ny", "nx"],
            },
            "xp": {"kind": "actual_numpy_module"},
            "terminal": {
                "kind": "None_or_exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["ny", "nx"],
            },
        },
    },
    "TerrainReference": {
        "fields": [
            "initial_full_elevation_m",
            "initial_committed_elevation_m",
            "export_receiver_full",
            "active_mask",
            "friction_factor",
            "dx_m",
            "allow_masked_nodata",
            "nodata_value",
            "allow_pit_storage",
            "allow_flat_storage",
        ],
        "per_field": {
            "initial_full_elevation_m": {
                "kind": "exact_numpy.ndarray",
                "dtype": "float64",
                "shape": ["ny+2", "nx+2"],
            },
            "initial_committed_elevation_m": {
                "kind": "exact_numpy.ndarray",
                "dtype": "float64",
                "shape": ["ny", "nx"],
            },
            "export_receiver_full": {
                "kind": "exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["ny+2", "nx+2"],
            },
            "active_mask": {
                "kind": "exact_numpy.ndarray",
                "dtype": "bool",
                "shape": ["ny", "nx"],
            },
            "friction_factor": {
                "kind": "exact_numpy.ndarray",
                "dtype": "float64",
                "shape": ["ny", "nx"],
            },
            "dx_m": {"kind": "plain_float"},
            "allow_masked_nodata": {"kind": "plain_bool"},
            "nodata_value": {"kind": "None_or_plain_float"},
            "allow_pit_storage": {"kind": "plain_bool"},
            "allow_flat_storage": {"kind": "plain_bool"},
        },
    },
    "SedimentEventState": {
        "fields": [
            "storm",
            "bed",
            "sediment_velocity_m_s",
            "graph",
            "network",
            "grid",
            "terrain",
            "phase",
            "routing_source_elevation_m",
        ],
        "per_field": {},
    },
}
RECEIPT_COMMON = (
    "schema",
    "index",
    "kind",
    "status",
    "code",
    "static_sha256",
    "resolved",
    "clock",
    "bed_identity_before",
    "bed_identity_after",
    "aggregate",
)
RECEIPT_BODIES = {
    "wind": {
        "native": (
            "block",
            "advanced_end_time_s",
            "advanced_duration_s",
            "physical_time_s",
            "process_mode",
            "integration_path",
            "n_accounting_steps",
            "accounting_step_upper_bound",
            "n_commits_this_block",
            "conservation_residual_kg",
        ),
        "coverage": (
            "requested_s",
            "native_plan_steps",
            "native_plan_sum_s",
            "plan_discrepancy_s",
            "plan_rounding_bound_s",
            "elapsed_discrepancy_s",
        ),
        "classes": (
            "bed_change_kg",
            "boundary_input_kg",
            "boundary_export_kg",
            "mobile_before_kg",
            "mobile_after_kg",
            "transport_mobile_after_kg",
            "transport_boundary_export_kg",
            "transport_bed_change_kg",
            "detachment_kg",
            "native_residual_kg",
            "native_bound_kg",
            "event_residual_kg",
            "event_bound_kg",
            "event_summation_terms",
            "event_arithmetic_scale_kg",
            "inventory_before_kg",
            "inventory_after_kg",
            "inventory_residual_kg",
            "inventory_bound_kg",
        ),
        "wind_end_pool": "'genuinely_empty'",
        "finalization": (
            "status",
            "algorithm_version",
            "representation",
            "placement_rule",
            "record_origin_s",
            "record",
            "record_sha256",
            "completed_block_index",
            "completed_block_offset_s",
            "physical_time_s",
            "n_settled",
            "n_exported",
            "airborne_mass_before_by_class_kg",
            "airborne_mass_after_by_class_kg",
            "deposited_mass_by_class_kg",
            "boundary_export_mass_by_class_kg",
            "bed_change_by_class_kg",
            "transfer_identity_gap_by_class_kg",
            "reservoir_identity_gap_by_class_kg",
            "transfer_identity_bound_kg",
            "reservoir_identity_bound_kg",
            "numerical_residual_scalar_kg",
        ),
    },
    "rain": {
        "native": (
            "dry_time_s",
            "rainfall_end_s",
            "recession_guard_end_s",
            "rainfall_edges_s",
            "intensity_mm_h",
            "n_accepted_steps",
            "n_rejected_attempts",
            "n_commits",
            "report_rows",
            "report_rows_bound",
            "storm_steps_lower_bound",
            "transport_scheme",
            "phase",
        ),
        "routing": (
            "first_graph_construction",
            "entry_rerouted",
            "entry_net_displacement_m",
            "graph_retained_by_identity",
            "n_reroutes_in_event",
        ),
        "classes": (
            "actual_pickup_kg",
            "deposition_actual_kg",
            "export_kg",
            "import_kg",
            "internal_terminal_deposition_kg",
            "flow_end_mobile_before_settlement_kg",
            "native_closure_residual_kg",
            "native_closure_tolerance_kg",
            "terminal_sediment_tolerance_kg",
            "inventory_before_kg",
            "inventory_after_kg",
            "inventory_residual_kg",
            "inventory_bound_kg",
        ),
        "water": (
            "rain_m3",
            "surface_initial_m3",
            "soil_initial_m3",
            "surface_final_m3",
            "soil_final_m3",
            "drainage_m3",
            "export_m3",
            "residual_m3",
            "tolerance_m3",
            "dry_surface_removed_m3",
            "dry_soil_removed_m3",
            "water_added_m3",
            "combined_water_residual_m3",
            "dry_reset",
        ),
    },
}


def _reference(value, expected_format):
    _keys(value, ARTIFACT_REF_FIELDS, "artifact reference")
    path = value["relative_path"]
    try:
        path_bytes = path.encode() if type(path) is str else b""
    except UnicodeEncodeError:
        _fail("artifact_reference", "artifact path must be valid UTF-8")
    if (
        type(value["format"]) is not str
        or value["format"] != expected_format
        or type(path) is not str
        or not path
        or len(path_bytes) > 4096
        or path.startswith("/")
        or any(x in ("", ".", "..") for x in path.split("/"))
        or len(path.split("/")) > 24
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
        or any(len(component.encode()) > 255 for component in path.split("/"))
    ):
        _fail("artifact_reference", "closed relative artifact path/format required")
    _digest(value["manifest_file_sha256"], "artifact manifest")
    if expected_format == _ORIGINAL_FORMAT:
        _digest(value["identity"], "original content")
    else:
        identity = value["identity"]
        _keys(identity, bridge._IDENTITY, "native22 identity")
        if type(identity["schema"]) is not str or identity["schema"] != bridge.SCHEMA:
            _fail("artifact_reference", "existing native22 schema required")
        for key in ("original_inputs_sha256", "case_sha256", "bed_identity_sha256"):
            _digest(identity[key], key)
        caller = identity["caller"]
        _keys(
            caller,
            ("schema", "source_identity", "environment_sha256", "static_sha256"),
            "native caller",
        )
        if (
            type(caller["schema"]) is not str
            or caller["schema"] != "masy.dense_runner_boundary_native.v1"
        ):
            _fail("artifact_reference", "native caller schema required")
        oi._source_identity(caller["source_identity"])
        _digest(caller["environment_sha256"], "native environment")
        _digest(caller["static_sha256"], "native static")
    return path


class _Artifacts:
    """Hold no-follow ancestors; admit ALL raw sizes before the first child hash.

    Native delegation remains path-based. Pre/post checks detect changes under the
    quiescent owned-artifact protocol; this is not a hostile-writer atomic reader.
    """

    def __init__(self, root, originals, genesis, current, limits, wrapper_raw=0):
        BoundaryImageLimits.__post_init__(limits)
        for reference, fmt in zip(
            (originals, genesis, current),
            (_ORIGINAL_FORMAT, _NATIVE_FORMAT, _NATIVE_FORMAT),
            strict=True,
        ):
            _reference(reference, fmt)
        self.root, self.refs, self.limits = (
            Path(root).absolute(),
            tuple(copy.deepcopy(value) for value in (originals, genesis, current)),
            limits,
        )
        self.wrapper_raw = _int(
            wrapper_raw, limits.max_serialized_bytes, "admitted wrapper raw bytes"
        )
        self.fds, self.dirs, self.files = [], {}, {}
        self.raw = wrapper_raw
        try:
            for ref, fmt in zip(
                self.refs,
                (_ORIGINAL_FORMAT, _NATIVE_FORMAT, _NATIVE_FORMAT),
                strict=True,
            ):
                relative = _reference(ref, fmt)
                if relative in self.files:
                    if self.files[relative][3] != fmt:
                        _fail(
                            "artifact_reference",
                            "one child path cannot name two formats",
                        )
                    continue
                parent_path = self.root
                for part in (None, *relative.split("/")):
                    if part is not None:
                        parent_path /= part
                    if parent_path not in self.dirs:
                        fd = oi._open_dir(parent_path)
                        self.fds.append(fd)
                        self.dirs[parent_path] = (fd, os.fstat(fd))
                directory = self.dirs[parent_path][0]
                names = []
                with os.scandir(directory) as entries:
                    for entry in entries:
                        names.append(entry.name)
                        if len(names) > (258 if fmt == _ORIGINAL_FORMAT else 4):
                            _fail("file_cap", "bounded child entry count")
                expected = {"checkpoint.json", "native_state.npz", "pending_state.npz"}
                if fmt == _NATIVE_FORMAT:
                    extra = set(names) - expected
                    if not expected <= set(names) or len(extra) > 1:
                        _fail("unlisted_file", "native literal file inventory")
                    for name in extra:
                        if re.fullmatch(r"\.manifest-[0-9a-f]{32}", name) is None:
                            _fail("unlisted_file", "unknown native entry")
                else:
                    if "manifest.json" not in names or any(
                        name not in ("manifest.json", "manifest.tmp")
                        and re.fullmatch(r"a_[0-9]{4}\.npy", name) is None
                        for name in names
                    ):
                        _fail("unlisted_file", "original fixed file inventory")
                sizes, total, rows = {}, 0, {}
                for name in names:
                    s = os.stat(name, dir_fd=directory, follow_symlinks=False)
                    if not stat.S_ISREG(s.st_mode):
                        _fail("file_type", "child is not regular/no-follow")
                    manifest_name = (
                        "manifest.json"
                        if fmt == _ORIGINAL_FORMAT
                        else "checkpoint.json"
                    )
                    metadata = (
                        name == manifest_name
                        or name == "manifest.tmp"
                        or name.startswith(".manifest-")
                    )
                    file_cap = (
                        (4 << 20)
                        if fmt == _ORIGINAL_FORMAT and metadata
                        else (
                            72 << 20
                            if fmt == _ORIGINAL_FORMAT
                            else (
                                1 << 20
                                if metadata
                                else limits.native_max_bytes + 327680
                            )
                        )
                    )
                    if s.st_size > file_cap:
                        _fail("serialized_cap", "raw child per-file cap before hash")
                    if name == "manifest.tmp" or name.startswith(".manifest-"):
                        ms = os.stat(
                            manifest_name, dir_fd=directory, follow_symlinks=False
                        )
                        if oi._stat_id(s) != oi._stat_id(ms) or s.st_nlink != 2:
                            _fail(
                                "unlisted_file",
                                "only exact native/34 leftover seal hardlink permitted",
                            )
                    else:
                        if s.st_nlink != (
                            2
                            if any(
                                x == "manifest.tmp" or x.startswith(".manifest-")
                                for x in names
                            )
                            and name == manifest_name
                            else 1
                        ):
                            _fail("file_type", "foreign hard link refused")
                        total += s.st_size
                        sizes[name] = s.st_size
                    rows[name] = oi._stat_id(s)
                if fmt == _NATIVE_FORMAT:
                    if (
                        sum(sizes[n] for n in ("native_state.npz", "pending_state.npz"))
                        > limits.native_max_bytes + 655360
                    ):
                        _fail(
                            "serialized_cap", "native raw aggregate before hash/archive"
                        )
                elif total > 72 << 20:
                    _fail("serialized_cap", "original raw aggregate before hash/header")
                self.raw += total
                if self.raw > limits.max_global_raw_bytes:
                    _fail(
                        "serialized_cap",
                        "combined raw-input admission before child hash/expansion",
                    )
                self.files[relative] = (parent_path, directory, rows, fmt, sizes)
            # Conservative data/readback reservation bound, no eager allocation.
            self.reservation_bytes = self.raw + 2 * (
                (64 << 20)
                + 2 * limits.native_max_bytes
                + limits.max_numeric_bytes
                + limits.max_cache_bytes
            )
            if self.reservation_bytes > 2 << 30:
                _fail("memory_cap", "bounded component data/readback reservation")
            self.check()
        except BaseException:
            self.close()
            raise

    def close(self):
        for fd in self.fds:
            os.close(fd)
        self.fds.clear()

    def check(self):
        for path, (fd, before) in self.dirs.items():
            other = oi._open_dir(path)
            try:
                now = os.fstat(other)
                if (before.st_dev, before.st_ino) != (now.st_dev, now.st_ino):
                    _fail("file_changed", "held artifact ancestor was replaced")
                if (now.st_dev, now.st_ino) != (
                    os.fstat(fd).st_dev,
                    os.fstat(fd).st_ino,
                ):
                    _fail("file_changed", "held artifact ancestor identity changed")
            finally:
                os.close(other)
        for _path, directory, rows, _fmt, _sizes in self.files.values():
            names = oi._names(
                directory,
                SimpleNamespace(max_files=2 if _fmt == _NATIVE_FORMAT else 256),
            )
            if set(names) != set(rows):
                _fail("file_changed", "artifact entry inventory changed")
            for name, before in rows.items():
                if (
                    oi._stat_id(os.stat(name, dir_fd=directory, follow_symlinks=False))
                    != before
                ):
                    _fail("file_changed", "artifact child changed")

    def hashed(self, ref):
        path, fd, rows, fmt, sizes = self.files[ref["relative_path"]]
        self.check()
        hashes = {}
        if fmt == _NATIVE_FORMAT:
            from masy.runtime import RuntimeRefusal, _hash_path

            for name in ("checkpoint.json", "native_state.npz", "pending_state.npz"):
                try:
                    digest, _blob, count, _raw = _hash_path(
                        path / name, budget=sizes[name]
                    )
                except RuntimeRefusal as exc:
                    _fail(exc.code, exc.detail)
                if count != sizes[name]:
                    _fail("file_changed", "native file size changed")
                hashes[name] = digest
            manifest_name = "checkpoint.json"
        else:
            for name in sorted(sizes):
                opened, s = oi._open_file(fd, name)
                try:
                    if oi._stat_id(s) != rows[name]:
                        _fail("file_changed", "original child changed before hash")
                    hashes[name] = oi._hash_fd(opened, sizes[name])
                    if oi._stat_id(os.fstat(opened)) != rows[name]:
                        _fail("file_changed", "original child changed during hash")
                finally:
                    os.close(opened)
            manifest_name = "manifest.json"
        if hashes[manifest_name] != ref["manifest_file_sha256"]:
            _fail(
                "artifact_identity", "raw child manifest SHA256 differs from reference"
            )
        self.check()
        return hashes


def _decode_wrapper_arrays(directory, descriptors, rows, limits):
    arrays = []
    import numpy as np

    for row in rows:
        fd, before = descriptors[row["name"]]
        oi._verify_written(directory, row, limits)
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "rb") as stream:
            shape, fortran, dtype = oi._read_npy_header(stream, limits)
            if (
                type(shape) is not tuple
                or any(type(dimension) is not int for dimension in shape)
                or list(shape) != row["shape"]
                or fortran is not False
                or dtype.str != "<f8"
                or stream.tell() + row["data_bytes"] != before.st_size
            ):
                _fail("array_layout", "NPY prefix differs")
            flat = np.fromfile(stream, dtype=dtype, count=row["data_bytes"] // 8)
            if flat.size != row["data_bytes"] // 8:
                _fail("file_changed", "truncated wrapper numeric data before reshape")
            a = flat.reshape(shape)
        a.base.flags.writeable = False
        a.flags.writeable = False
        if hashlib.sha256(memoryview(a).cast("B")).hexdigest() != row["data_sha256"]:
            _fail("file_hash", "expanded raw bytes differ")
        arrays.append(a)
    return arrays


def _read_wrapper(
    path,
    expected_manifest_sha256,
    limits,
    *,
    metadata_only=False,
    _manifest_name="manifest.json",
    _admitted_raw=None,
):
    """Fixed5/7 raw NPY decoder reusing34's bounded prefix/FD primitives."""
    _digest(expected_manifest_sha256, "expected boundary seal")
    directory, opened = oi._open_dir(path), []
    try:
        names = oi._names(directory, limits)
        descriptors, total = {}, 0
        private_readback = _manifest_name == "manifest.tmp"
        if _manifest_name not in names or (
            private_readback and "manifest.json" in names
        ):
            _fail(
                "unsealed",
                "expected boundary manifest absent or invalid private readback",
            )
        for name in sorted(names):
            fd, s = oi._open_file(directory, name)
            opened.append(fd)
            descriptors[name] = (fd, s)
            if name == "manifest.tmp" and not private_readback:
                prior = descriptors["manifest.json"][1]
                if oi._stat_id(prior) != oi._stat_id(s) or s.st_nlink != 2:
                    _fail(
                        "unlisted_file",
                        "only exact leftover own seal hardlink permitted",
                    )
            else:
                total += s.st_size
                cap = (
                    limits.max_metadata_bytes
                    if name == _manifest_name
                    else limits.max_numeric_bytes + limits.max_header_bytes
                )
                if s.st_size > cap or total > limits.max_serialized_bytes:
                    _fail(
                        "serialized_cap",
                        "wrapper raw file/aggregate cap before hash/header",
                    )
                if s.st_nlink != (
                    2
                    if not private_readback
                    and name == "manifest.json"
                    and "manifest.tmp" in names
                    else 1
                ):
                    _fail("file_type", "foreign wrapper hardlink refused")
        if _admitted_raw is not None and total != _admitted_raw:
            _fail(
                "file_changed", "wrapper raw bytes changed before second numerical read"
            )
        fd, ms = descriptors[_manifest_name]
        raw = os.read(fd, ms.st_size + 1)
        manifest = oi._strict_json(raw, limits)
        _keys(manifest, ROOT_FIELDS, "boundary manifest")
        seal = manifest["manifest_sha256"]
        if (
            manifest["schema"] != SCHEMA
            or manifest["profile"] != PROFILE
            or seal != expected_manifest_sha256
            or _sha({k: v for k, v in manifest.items() if k != "manifest_sha256"})
            != seal
            or oi._json(manifest) != raw
        ):
            _fail("manifest_identity", "canonical boundary seal/schema/profile differs")
        rows = manifest["arrays"]
        if (
            type(rows) is not list
            or len(rows) not in (5, 7)
            or len(rows) > limits.max_files
        ):
            _fail("array_layout", "exact5/7 wrapper arrays required")
        required = {_manifest_name} | (
            {"manifest.tmp"}
            if not private_readback and "manifest.tmp" in names
            else set()
        )
        numeric = 0
        # All declarations and aggregate caps admitted BEFORE payload hash/header/expansion.
        for i, row in enumerate(rows):
            _keys(row, _ARRAY_KEYS, "wrapper array")
            shape = row["shape"]
            if (
                row["name"] != f"a_{i:04d}.npy"
                or row["dtype"] != "<f8"
                or type(shape) is not list
                or len(shape) not in (1, 2, 3)
                or any(type(d) is not int or not 0 < d <= 64 for d in shape)
            ):
                _fail("array_layout", "fixed numeric dtype/name/shape")
            size = math.prod(shape) * 8
            if type(row["data_bytes"]) is not int or row["data_bytes"] != size:
                _fail("array_layout", "raw numeric byte count")
            numeric += size
            if numeric > limits.max_numeric_bytes or row["name"] not in descriptors:
                _fail(
                    "array_cap", "wrapper aggregate numeric/file cap before expansion"
                )
            _digest(row["sha256"], "numeric file hash")
            _digest(row["data_sha256"], "numeric data hash")
            if (
                type(row["file_bytes"]) is not int
                or descriptors[row["name"]][1].st_size != row["file_bytes"]
            ):
                _fail("array_layout", "raw file size differs")
            required.add(row["name"])
        if set(names) != required:
            _fail("unlisted_file", "unknown boundary artifact entry")
        if metadata_only:
            for name, (handle, before) in descriptors.items():
                if oi._stat_id(os.fstat(handle)) != oi._stat_id(before) or oi._stat_id(
                    os.stat(name, dir_fd=directory, follow_symlinks=False)
                ) != oi._stat_id(before):
                    _fail(
                        "file_changed", "wrapper metadata admission changed during read"
                    )
            return manifest, [], total
        arrays = _decode_wrapper_arrays(directory, descriptors, rows, limits)
        for name, (fd, before) in descriptors.items():
            if oi._stat_id(os.fstat(fd)) != oi._stat_id(before) or oi._stat_id(
                os.stat(name, dir_fd=directory, follow_symlinks=False)
            ) != oi._stat_id(before):
                _fail("file_changed", "wrapper file changed during read")
        return manifest, arrays, total
    finally:
        for fd in opened:
            os.close(fd)
        os.close(directory)


def _vector(value, nc, where):
    if type(value) is not list or len(value) != nc:
        _fail("receipt_layout", f"{where}: class vector length")
    for v in value:
        _float(v, where)


def _receipt_shape(body, nc, limits):
    if type(body) is not dict:
        _fail("receipt_layout", "plain receipt body required")
    kind = body.get("kind")
    if type(kind) is not str or kind not in RECEIPT_BODIES:
        _fail("receipt_layout", "unknown receipt kind")
    spec = RECEIPT_BODIES[kind]
    _keys(body, (*RECEIPT_COMMON, *spec), "receipt")
    for section, names in spec.items():
        if type(names) is tuple:
            _keys(body[section], names, f"receipt.{section}")
    _keys(
        body["aggregate"],
        ("joint_residual_kg", "joint_bound_kg", "n_summation_terms"),
        "aggregate",
    )
    for key in ("joint_residual_kg", "joint_bound_kg"):
        _vector(body["aggregate"][key], nc, key)
    _int(body["aggregate"]["n_summation_terms"], 1 << 30, "sum terms")
    for key in ("static_sha256", "bed_identity_before", "bed_identity_after"):
        _digest(body[key], key)
    _int(body["index"], limits.max_events - 1, "receipt index")
    if (
        body["schema"] != dr.RECEIPT_SCHEMA
        or body["status"] != "accepted"
        or body["code"] != "accepted"
    ):
        _fail("receipt_layout", "only accepted default receipt schema")
    for key, value in body["classes"].items():
        if value is None and key in (
            "detachment_kg",
            "flow_end_mobile_before_settlement_kg",
        ):
            continue
        if kind == "rain" and key == "terminal_sediment_tolerance_kg":
            # Native reservoir_bound_kg returns one scalar; unchanged DenseRunner
            # encodes it as a one-element list, independent of class count.
            _vector(value, 1, key)
        elif key in (
            "native_bound_kg",
            "event_bound_kg",
            "event_arithmetic_scale_kg",
        ) or (kind == "wind" and key == "inventory_bound_kg"):
            _float(value, key)
        elif key == "event_summation_terms":
            if type(value) is not int or value != 6:
                _fail("receipt_layout", "wind event terms")
        else:
            _vector(value, nc, key)
    if kind == "wind":
        if body["wind_end_pool"] != "genuinely_empty":
            _fail("receipt_layout", "wind mobile completion missing")
        _keys(
            body["native"]["block"],
            ("start_time_s", "end_time_s", "duration_s"),
            "native block",
        )
        for value in body["native"]["block"].values():
            decode_time(value)
        for name, value in body["native"].items():
            if name == "block":
                continue
            if name in (
                "advanced_end_time_s",
                "advanced_duration_s",
                "physical_time_s",
            ):
                decode_time(value)
            elif name in ("process_mode", "integration_path"):
                if type(value) is not str or len(value.encode()) > 128:
                    _fail("receipt_layout", "native mode/path")
            elif name == "conservation_residual_kg":
                _float(value, name)
            else:
                _int(value, 1 << 30, name)
        for key, value in body["coverage"].items():
            if key == "requested_s":
                decode_time(value)
            elif key == "native_plan_steps":
                _int(value, 1 << 30, key)
            else:
                if (
                    type(value) is not str
                    or len(value) > 4096
                    or _FRACTION.fullmatch(value) is None
                ):
                    _fail(
                        "receipt_layout",
                        "coverage requires bounded canonical rational syntax",
                    )
                numerator, denominator = (int(part) for part in value.split("/"))
                if math.gcd(numerator, denominator) != 1:
                    _fail("receipt_layout", "noncanonical coverage fraction")
        final = body["finalization"]
        for key, value in final.items():
            if key in ("record_origin_s", "physical_time_s"):
                decode_time(value)
            elif key in ("record",):
                pass  # compared to the complete actual native scalar producer below
            elif key == "record_sha256":
                _digest(value, key)
            elif key in ("completed_block_index", "n_settled", "n_exported"):
                _int(value, 1 << 30, key)
            elif key in (
                "status",
                "algorithm_version",
                "representation",
                "placement_rule",
            ):
                if type(value) is not str or len(value.encode()) > 4096:
                    _fail("receipt_layout", "finalization string")
            elif key.endswith("by_class_kg"):
                _vector(value, nc, key)
            else:
                _float(value, key)
    else:
        if type(body["native"]["rainfall_edges_s"]) is not list:
            _fail("receipt_layout", "plain bounded rainfall edge list required")
        for key, value in body["native"].items():
            if key == "rainfall_edges_s":
                if type(value) is not list or not 2 <= len(value) <= 65:
                    _fail("receipt_layout", "bounded rainfall edges")
                for item in value:
                    decode_time(item)
            elif key == "intensity_mm_h":
                if (
                    type(value) is not list
                    or len(value) != len(body["native"]["rainfall_edges_s"]) - 1
                ):
                    _fail("receipt_layout", "rainfall intensity count")
                for item in value:
                    _float(item, key)
            elif key.endswith("_s"):
                decode_time(value)
            elif key == "transport_scheme":
                if value != "upwind":
                    _fail("receipt_layout", "only upwind receipt")
            elif key == "phase":
                if value is not None:
                    _fail("receipt_layout", "rain phase must be None")
            else:
                _int(value, 1 << 30, key)
        for key, value in body["routing"].items():
            if key == "entry_net_displacement_m":
                if value is not None:
                    _float(value, key)
            elif key == "n_reroutes_in_event":
                _int(value, 1 << 30, key)
            elif type(value) is not bool:
                _fail("receipt_layout", "routing boolean kind")
        for key, value in body["water"].items():
            if key == "dry_reset":
                if (
                    value
                    != "external dry-again removal (native); soil water is zero after the reset"
                ):
                    _fail("receipt_layout", "native dry-reset profile changed")
            else:
                _float(value, key)


def _receipt_rows(receipts, limits):
    if type(receipts) is not list or len(receipts) > limits.max_events:
        _fail("receipt_cap", "bounded plain receipt list required")
    rows, used = [], 0
    for body in receipts:
        raw = dr._canonical_json(body).encode()
        used += len(raw)
        if len(raw) > limits.max_receipt_bytes or used > limits.max_receipts_bytes:
            _fail("receipt_cap", "canonical receipt byte cap")
        rows.append(
            {
                "body": copy.deepcopy(body),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return rows


def _chain(originals, genesis, rows):
    return _sha(
        {
            "originals": originals,
            "genesis": genesis,
            "receipts": [{"bytes": r["bytes"], "sha256": r["sha256"]} for r in rows],
        }
    )


def _totals_wire(totals):
    _keys(totals, TOTALS_FIELDS, "totals")
    _keys(totals["water"], WATER_FIELDS, "water totals")
    water = {
        key: encode_time(_float(value, key)) for key, value in totals["water"].items()
    }
    return {
        "export": "total_export",
        "input": "total_input",
        "bound": "total_bound",
        "internal_terminal_deposition": "total_internal_terminal_deposition",
        "water": water,
        "n_entry_reroutes": totals["n_entry_reroutes"],
        "n_event_reroutes": totals["n_event_reroutes"],
    }


def _validate_history(manifest, originals, genesis, current, arrays, limits):
    import numpy as np

    state = manifest["state"]
    _keys(state, STATE_FIELDS, "state")
    ctx, bed = originals.capsule.context, current.bed
    g, nc = ctx.geometry, len(ctx.grain_classes.classes)
    shape = (g.ny, g.nx)
    k = _int(
        state["next_index"],
        min(limits.max_events, len(originals.schedule.events)),
        "next index",
    )
    for key, expected in (
        ("n_accepted", k),
        ("n_classes", nc),
        (
            "inventory_elements",
            g.ny * g.nx * (originals.capsule.case_record["n_voxels"] + 1),
        ),
    ):
        if type(state[key]) is not int or state[key] != expected:
            _fail("state_identity", f"{key} differs from original/cursor")
    if len(arrays) != (5 if manifest["routing"] is None else 7):
        _fail("array_layout", "routing presence requires exact5/7 arrays")
    soil = _array(arrays[0], shape, "current soil")
    totals_record = manifest["totals"]
    _keys(totals_record, TOTALS_FIELDS, "totals")
    totals = {}
    for i, name in enumerate(
        ("export", "input", "bound", "internal_terminal_deposition"), 1
    ):
        if totals_record[name] != _ROLES[i]:
            _fail("array_layout", "fixed total array roles")
        totals[name] = _array(arrays[i], (nc,), name)
    _keys(totals_record["water"], WATER_FIELDS, "water totals")
    totals["water"] = {
        key: decode_time(totals_record["water"][key]) for key in WATER_FIELDS
    }
    for key in ("n_entry_reroutes", "n_event_reroutes"):
        totals[key] = _int(totals_record[key], 1 << 30, key)
    ledger = ClockLedger.from_record(manifest["ledger"])
    if type(ledger) is not ClockLedger or set(vars(ledger)) != {"epoch", "_entries"}:
        _fail("clock_layout", "exact native clock ledger layout")
    records = ledger.records()
    rows = manifest["receipts"]
    if type(rows) is not list or len(rows) != k or len(records) != k:
        _fail("cursor", "cursor/accepted/ledger/receipt counts differ")
    if (
        encode_time(ledger.epoch) != encode_time(genesis.bed.time_s)
        or encode_time(ledger.epoch) != originals.capsule.case_record["initial_time_s"]
    ):
        _fail("clock_identity", "original/genesis epoch differs")
    if state["native_time"] != encode_time(bed.time_s):
        _fail("clock_identity", "current native time differs")
    _same(
        np.asarray(
            dr._native().bed_inventory(genesis.bed.state), dtype=np.float64
        ).reshape(-1),
        originals.initial_inventory_kg,
        limits,
    )
    _same(genesis.mobile_state, originals.wind.initial_mobile_state, limits)
    _same(
        genesis.bed.state.water.depth_m,
        originals.capsule.held_inputs["initial_water_depth_m"],
        limits,
    )
    n = dr._native()
    _same(
        n.committed_elevation_m(genesis.bed.state.committed_topography),
        originals.water.terrain.initial_committed_elevation_m,
        limits,
    )
    before = genesis.bed.identity_sha256()
    used, receipts, rain_seen = 0, [], False
    zeros = np.zeros(nc, dtype=np.float64)
    fold = {
        key: zeros.copy()
        for key in ("export", "input", "bound", "internal_terminal_deposition")
    }
    fold["water"] = {key: 0.0 for key in WATER_FIELDS}
    fold["n_entry_reroutes"], fold["n_event_reroutes"] = 0, 0
    for index, (row, clock) in enumerate(zip(rows, records, strict=True)):
        _keys(row, ("body", "bytes", "sha256"), "receipt row")
        body = row["body"]
        raw = dr._canonical_json(body).encode()
        used += len(raw)
        if (
            type(row["bytes"]) is not int
            or row["bytes"] != len(raw)
            or len(raw)
            > min(limits.max_receipt_bytes, originals.runner_limits.max_receipt_bytes)
            or used > limits.max_receipts_bytes
            or row["sha256"] != hashlib.sha256(raw).hexdigest()
        ):
            _fail("receipt_identity", "canonical receipt bytes/hash/cap")
        _receipt_shape(body, nc, limits)
        event = originals.schedule.events[index]
        start, end = (
            decode_time(clock["native_start"]),
            decode_time(clock["native_end"]),
        )
        if (
            clock["kind"] != "event"
            or body["index"] != index
            or body["kind"] != event.kind
            or body["static_sha256"] != originals.static_sha256
            or oi._json(body["clock"]) != oi._json(clock)
            or body["bed_identity_before"] != before
        ):
            _fail("receipt_chain", "source/event/clock/bed chain differs")
        expected_returned = (
            body["native"]["advanced_duration_s"] if event.kind == "wind" else None
        )
        if clock["returned_elapsed"] != expected_returned:
            _fail(
                "receipt_clock",
                "native returned duration differs from owning _finish caller",
            )
        if body["aggregate"]["n_summation_terms"] != 2 * state[
            "inventory_elements"
        ] + 2 * (index + 1):
            _fail("aggregate_identity", "historical aggregate arithmetic count differs")
        check_resolved_event(body["resolved"], originals.schedule)
        if oi._json(body["resolved"]) != oi._json(
            resolve_event(originals.schedule, index, start)
        ):
            _fail(
                "receipt_clock",
                "forcing resolution origin differs from original receipt start",
            )
        if index == 0 and clock["commit_before"] != encode_time(
            float(genesis.bed.state.committed_topography.last_commit_time_s)
        ):
            _fail("clock_identity", "genesis committed stamp differs")
        before = body["bed_identity_after"]
        c = body["classes"]
        if event.kind == "wind":
            record = n.resolve_wind_forcing_record(
                {
                    "blocks": [
                        {
                            "shear_velocity_m_s": event.shear_velocity_m_s,
                            "direction_deg": event.direction_deg_cw_from_north,
                            "duration_s": event.duration_s,
                        }
                    ]
                }
            )
            canonical = dr._canonical_input(np, record, "completed_record")
            final = body["finalization"]
            if (
                oi._json(final["record"]) != oi._json(canonical)
                or final["record_sha256"] != db._sha256(canonical)
                or final["record_origin_s"] != encode_time(start)
                or final["physical_time_s"] != encode_time(end)
                or body["native"]["physical_time_s"] != encode_time(end)
            ):
                _fail("receipt_identity", "completed native record/clock differs")
            additions = (
                c["boundary_export_kg"],
                c["boundary_input_kg"],
                np.full(nc, c["inventory_bound_kg"]),
                final["deposited_mass_by_class_kg"],
            )
        else:
            rain_seen = True
            if body["native"]["dry_time_s"] != encode_time(end):
                _fail("receipt_clock", "native dry endpoint differs")
            additions = (
                c["export_kg"],
                c["import_kg"],
                c["inventory_bound_kg"],
                c["internal_terminal_deposition_kg"],
            )
            w = body["water"]
            delta = {
                "rain_m3": w["rain_m3"],
                "drainage_m3": w["drainage_m3"],
                "storm_export_m3": w["export_m3"],
                "dry_surface_removed_m3": w["dry_surface_removed_m3"],
                "dry_soil_removed_m3": w["dry_soil_removed_m3"],
            }
            fold["water"] = {
                key: fold["water"][key] + value for key, value in delta.items()
            }
            fold["n_entry_reroutes"] += int(body["routing"]["entry_rerouted"])
            fold["n_event_reroutes"] += body["routing"]["n_reroutes_in_event"]
        for key, addition in zip(
            ("export", "input", "bound", "internal_terminal_deposition"),
            additions,
            strict=True,
        ):
            fold[key] = fold[key] + np.asarray(addition, dtype=np.float64)
        receipts.append(body)
    _same(fold, totals, limits)
    if not any(body["kind"] == "wind" for body in receipts):
        # Rain passes the same native wind pool through; no first wind has replaced it.
        _same(current.mobile_state, genesis.mobile_state, limits)
    if (
        _chain(manifest["originals"], manifest["genesis"], rows)
        != state["receipt_chain_sha256"]
    ):
        _fail("receipt_chain", "ordered canonical receipt chain differs")
    if rain_seen:
        if bool(np.any(soil != 0.0)):
            _fail(
                "soil_profile",
                "accepted native dry-reset soil must be numerically zero",
            )
    else:
        _same(soil, originals.water.initial_soil_water_m, limits)
    if (manifest["routing"] is not None) != rain_seen:
        _fail("routing_history", "retained routing exists iff rain has been accepted")
    if k:
        if (
            before != bed.identity_sha256()
            or records[-1]["native_end"] != encode_time(bed.time_s)
            or records[-1]["commit_after"]
            != encode_time(float(bed.state.committed_topography.last_commit_time_s))
        ):
            _fail("clock_identity", "last receipt/current bed endpoint differs")
        inventory = dr._vec(np, n.bed_inventory(bed.state))
        joint = (
            inventory
            + totals["export"]
            - totals["input"]
            - originals.initial_inventory_kg
        )
        terms = 2 * state["inventory_elements"] + 2 * k
        scale = max(
            float(np.abs(originals.initial_inventory_kg).sum()),
            float(np.abs(inventory).sum()),
            float(totals["export"].sum()),
            float(totals["input"].sum()),
        )
        allowed = totals["bound"] + n.summation_error_bound_kg(terms, scale)
        if not bool(np.all(np.abs(joint) <= allowed)):
            _fail("aggregate_budget_open", "actual native aggregate bound exceeded")
        _same(
            joint,
            np.asarray(
                receipts[-1]["aggregate"]["joint_residual_kg"], dtype=np.float64
            ),
            limits,
        )
        _same(
            allowed,
            np.asarray(receipts[-1]["aggregate"]["joint_bound_kg"], dtype=np.float64),
            limits,
        )
        if receipts[-1]["aggregate"]["n_summation_terms"] != terms:
            _fail("aggregate_identity", "last aggregate arithmetic count differs")
    elif (
        manifest["genesis"] != manifest["physical"]
        or encode_time(bed.time_s) != encode_time(ledger.epoch)
        or before != bed.identity_sha256()
    ):
        _fail(
            "genesis_identity",
            "cursor0 requires identical closed genesis/current references",
        )
    return soil, totals, ledger, receipts, state


def _preflight_metadata(
    manifest, expected_source_identity, expected_environment_sha256, limits
):
    oi._source_identity(expected_source_identity)
    _digest(expected_environment_sha256, "expected environment")
    if (
        oi._json(manifest["source_identity"]) != oi._json(expected_source_identity)
        or manifest["environment_sha256"] != expected_environment_sha256
    ):
        _fail("source_identity", "boundary source/environment cannot be rebound")
    _reference(manifest["originals"], _ORIGINAL_FORMAT)
    for name in ("genesis", "physical"):
        _reference(manifest[name], _NATIVE_FORMAT)
        identity = manifest[name]["identity"]
        _keys(
            identity,
            (
                "schema",
                "original_inputs_sha256",
                "case_sha256",
                "bed_identity_sha256",
                "caller",
            ),
            "native22 identity",
        )
        caller = identity["caller"]
        _keys(
            caller,
            ("schema", "source_identity", "environment_sha256", "static_sha256"),
            "native caller",
        )
        if (
            caller["schema"] != "masy.dense_runner_boundary_native.v1"
            or oi._json(caller["source_identity"]) != oi._json(expected_source_identity)
            or caller["environment_sha256"] != expected_environment_sha256
        ):
            _fail("source_identity", "native child source/environment differs")
        _digest(caller["static_sha256"], "native caller static")
    if manifest["originals"]["relative_path"] in (
        manifest["genesis"]["relative_path"],
        manifest["physical"]["relative_path"],
    ):
        _fail("artifact_reference", "original and native child paths must be disjoint")
    if manifest["genesis"]["relative_path"] == manifest["physical"][
        "relative_path"
    ] and oi._json(manifest["genesis"]) != oi._json(manifest["physical"]):
        _fail(
            "artifact_reference",
            "one native child path requires one identical closed reference",
        )
    state = manifest["state"]
    _keys(state, STATE_FIELDS, "state")
    k = _int(state["next_index"], limits.max_events, "next")
    if type(state["n_accepted"]) is not int or state["n_accepted"] != k:
        _fail("cursor", "next/count differ")
    nc = _int(state["n_classes"], limits.max_classes, "classes")
    if nc == 0:
        _fail("profile_cap", "classes must be positive")
    _int(
        state["inventory_elements"],
        limits.max_cells * (limits.max_voxels + 1),
        "inventory elements",
    )
    decode_time(state["native_time"])
    for key in (
        "accepted_bed_sha256",
        "accepted_private_sha256",
        "receipt_chain_sha256",
    ):
        _digest(state[key], key)
    rows = manifest["receipts"]
    if type(rows) is not list or len(rows) != k:
        _fail("cursor", "receipt count differs")
    used = 0
    for row in rows:
        _keys(row, ("body", "bytes", "sha256"), "receipt row")
        if type(row["body"]) is not dict:
            _fail("receipt_layout", "plain receipt body")
        _receipt_shape(row["body"], nc, limits)
        raw = dr._canonical_json(row["body"]).encode()
        used += len(raw)
        if (
            type(row["bytes"]) is not int
            or row["bytes"] != len(raw)
            or len(raw) > limits.max_receipt_bytes
            or used > limits.max_receipts_bytes
            or row["sha256"] != hashlib.sha256(raw).hexdigest()
        ):
            _fail(
                "receipt_identity", "canonical receipt hash/size/cap before child load"
            )
    if (
        type(manifest["ledger"]) is not dict
        or type(manifest["ledger"].get("entries")) is not list
        or len(manifest["ledger"]["entries"]) != k
    ):
        _fail("clock_layout", "bounded ledger count before decode")
    ClockLedger.from_record(manifest["ledger"])
    rain_seen = any(row["body"]["kind"] == "rain" for row in rows)
    routing = manifest["routing"]
    if (routing is not None) != rain_seen:
        _fail("routing_history", "retained source exists iff accepted rain")
    if routing is not None:
        _keys(routing, ROUTING_FIELDS, "routing")
        if (
            routing["source_array"] != "routing_source"
            or routing["velocity_array"] != "routing_velocity"
            or routing["phase"] is not None
        ):
            _fail("routing_layout", "fixed saved routing roles and None phase")
        for key in ROUTING_FIELDS:
            if key not in ("source_array", "velocity_array", "phase"):
                _digest(routing[key], key)
    shared = manifest["shared_resources"]
    _keys(shared, SHARED_RESOURCE_FIELDS, "shared resources")
    if (
        shared["schema"] != "masy.shared_surface_resources.v0"
        or type(shared["version"]) is not int
        or shared["version"] != 0
        or shared["profile"] != "static_prescribed_native_dry_reset_v0"
        or shared["soil_array"] != "current_soil"
        or type(shared["update_cursor"]) is not int
        or shared["update_cursor"] != 0
        or type(shared["update_receipts"]) is not list
        or shared["update_receipts"] != []
    ):
        _fail(
            "resource_profile",
            "only static prescribed/native dry-reset resource version0",
        )
    _digest(shared["prescribed_identity_sha256"], "prescribed identity")
    descriptions = manifest["arrays"]
    # Only internal in-memory authoring has []; persisted images have already
    # passed the SAME fixed5/7 reader before this control-record crosscheck.
    if type(descriptions) is not list:
        _fail("array_layout", "plain wrapper descriptions required")
    if descriptions:
        for index, row in enumerate(descriptions):
            _keys(row, _ARRAY_KEYS, "wrapper array description")
            if (
                row["name"] != f"a_{index:04d}.npy"
                or type(row["shape"]) is not list
                or any(type(v) is not int for v in row["shape"])
            ):
                _fail(
                    "array_layout",
                    "fixed positional file names and exact dimension kinds",
                )
        if type(descriptions) is not list or len(descriptions) != (
            7 if rain_seen else 5
        ):
            _fail("array_layout", "closed5/7 role declarations before child expansion")
        shape = descriptions[0]["shape"]
        if (
            type(shape) is not list
            or len(shape) != 2
            or any(type(v) is not int or v < 1 for v in shape)
            or math.prod(shape) > limits.max_cells
        ):
            _fail("array_layout", "bounded soil dimensions before child expansion")
        shapes = [shape, *([[nc]] * 4)] + ([shape, [*shape, nc]] if rain_seen else [])
        for row, expected in zip(descriptions, shapes, strict=True):
            if row["shape"] != expected:
                _fail("array_layout", "exact raw role shapes before child expansion")
    totals = manifest["totals"]
    _keys(totals, TOTALS_FIELDS, "totals")
    _keys(totals["water"], WATER_FIELDS, "water totals")
    for value in totals["water"].values():
        decode_time(value)
    for key in ("n_entry_reroutes", "n_event_reroutes"):
        _int(totals[key], 1 << 30, key)
    if k == 0 and oi._json(manifest["genesis"]) != oi._json(manifest["physical"]):
        _fail("genesis_identity", "cursor0 requires identical closed native references")


def _original_manifest_seal(artifacts, reference, limits):
    # Bounded metadata-only extraction; Original34's typed object decode still runs ONCE.
    _path, directory, rows, _fmt, sizes = artifacts.files[reference["relative_path"]]
    admitted = sizes["manifest.json"]
    fd, before = oi._open_file(directory, "manifest.json")
    try:
        if before.st_size != admitted or oi._stat_id(before) != rows["manifest.json"]:
            _fail("file_changed", "original manifest differs from raw admission")
        raw = os.read(fd, admitted + 1)
        if (
            len(raw) != admitted
            or hashlib.sha256(raw).hexdigest() != reference["manifest_file_sha256"]
            or oi._stat_id(os.fstat(fd)) != rows["manifest.json"]
            or oi._stat_id(
                os.stat("manifest.json", dir_fd=directory, follow_symlinks=False)
            )
            != rows["manifest.json"]
        ):
            _fail(
                "file_changed",
                "bounded original manifest changed during seal extraction",
            )
        m = oi._strict_json(
            raw,
            oi.OriginalPackLimits(
                max_cells=limits.max_cells,
                max_classes=limits.max_classes,
                max_voxels=limits.max_voxels,
                max_cohorts=limits.max_cohorts,
                max_events=limits.max_events,
            ),
        )
        _keys(
            m,
            ("schema", "profile", "content_sha256", "wire", "files", "manifest_sha256"),
            "original manifest",
        )
        return _digest(m["manifest_sha256"], "original seal")
    finally:
        os.close(fd)


def _native_dimensions(artifacts, reference, originals, limits):
    # Native15's own closed raw member validator; no second archive parser/format.
    from maple.aeolian.scheduler import physical_checkpoint as cp

    from masy.runtime import RuntimeRefusal, _read_bounded

    path, _directory, _rows, _fmt, sizes = artifacts.files[reference["relative_path"]]
    artifacts.check()
    try:
        raw = _read_bounded(
            path / "checkpoint.json", sizes["checkpoint.json"], "native_manifest_read"
        )
    except RuntimeRefusal as exc:
        _fail(exc.code, exc.detail)
    if (
        len(raw) != sizes["checkpoint.json"]
        or hashlib.sha256(raw).hexdigest() != reference["manifest_file_sha256"]
    ):
        _fail("file_changed", "native bounded manifest differs from admitted identity")
    m = cp._strict_json(raw)
    cp._keys(m, cp._MANIFEST_KEYS, "manifest")
    cp._keys(m["files"], cp.PAYLOADS, "payload list")
    description = m["files"]["native_state.npz"]
    cp._keys(description, ("sha256", "size_bytes", "members"), "payload record")
    headers = description["members"]
    if type(headers) is not dict or not 1 <= len(headers) <= 256:
        _fail("profile_cap", "native finite member count before load")
    for key, row in headers.items():
        if type(key) is not str or len(key) > 200:
            _fail("profile_cap", "native bounded member name")
        cp._keys(row, ("dtype", "shape", "npy_bytes"), "native member")
        if (
            type(row["dtype"]) is not str
            or len(row["dtype"]) > 128
            or type(row["npy_bytes"]) is not int
            or not 0 <= row["npy_bytes"] <= limits.native_max_bytes
            or type(row["shape"]) is not list
            or len(row["shape"]) > 5
            or any(
                type(v) is not int
                or not 0 <= v <= max(limits.max_cohorts, limits.max_cells + 2)
                for v in row["shape"]
            )
        ):
            _fail("profile_cap", "native finite raw shape before validator")
    cp._raw_base(headers)
    g, nc = (
        originals.capsule.context.geometry,
        len(originals.capsule.context.grain_classes.classes),
    )
    voxel = headers["voxel_mass_kg"]["shape"]
    pool = headers["cohort_mass_kg"]["shape"]
    if (
        len(voxel) != 4
        or voxel[:2] != [g.ny, g.nx]
        or voxel[3] != nc
        or voxel[2] != originals.capsule.case_record["n_voxels"]
        or not 1 <= voxel[2] <= limits.max_voxels
        or len(pool) != 1
        or not 0 <= pool[0] <= limits.max_cohorts
    ):
        _fail(
            "profile_cap", "native voxel/class/grid/cohort dimensions before expansion"
        )
    # Reuse native public dimension groups; no ledger arithmetic or array construction.
    from maple.core.types.sediment_ledger import (
        LEDGER_ACTIVITY_FIELDS,
        LEDGER_PENDING_FIELDS,
        LEDGER_PROCESS_TOTAL_FIELDS,
        LEDGER_RESIDUAL_FIELDS,
        N_LEDGER_PROCESSES,
    )

    pending = m["files"]["pending_state.npz"]
    cp._keys(pending, ("sha256", "size_bytes", "members"), "pending payload record")
    scalars = cp._decode_scalars(m["scalars"])
    if scalars["baseline"] is not None:
        _fail(
            "profile_cap",
            "unchanged22 passive boundary excludes a reconciliation baseline",
        )
    cell_class = [g.ny, g.nx, nc]
    expected = {
        f"ledger__{name}": cell_class
        for name in (*LEDGER_PENDING_FIELDS, "n_physical_touches")
    }
    expected.update(
        {
            f"ledger__{name}": [N_LEDGER_PROCESSES, nc]
            for name in LEDGER_PROCESS_TOTAL_FIELDS
        }
    )
    expected.update(
        {
            f"ledger__{name}": [N_LEDGER_PROCESSES, *cell_class]
            for name in (*LEDGER_RESIDUAL_FIELDS, *LEDGER_ACTIVITY_FIELDS)
        }
    )
    expected.update(
        {
            f"committed__{name}": (
                [g.ny, g.nx]
                if name in ("reference_elevation_m", "elevation_offset_m")
                else cell_class
            )
            for name in cp._COMMITTED_ARRAYS
        }
    )
    if {
        name.removeprefix("ledger__")
        for name in expected
        if name.startswith("ledger__")
    } != set(cp._LEDGER_ARRAYS):
        _fail("profile_cap", "native finite ledger dimension groups changed")
    cp._keys(pending["members"], expected, "native pending member inventory")
    for name, shape in expected.items():
        row = pending["members"][name]
        cp._keys(row, ("dtype", "shape", "npy_bytes"), "native pending member")
        expected_dtype = "<i8" if name == "ledger__n_physical_touches" else "<f8"
        if (
            type(row["dtype"]) is not str
            or row["dtype"] != expected_dtype
            or type(row["npy_bytes"]) is not int
            or not 0 <= row["npy_bytes"] <= limits.native_max_bytes
            or type(row["shape"]) is not list
            or any(type(v) is not int for v in row["shape"])
            or row["shape"] != shape
        ):
            _fail(
                "profile_cap",
                "native pending exact dtype/shape/byte kinds before expansion",
            )
    artifacts.check()


def _load_routing(record, originals, arrays, limits):
    if record is None:
        return None
    n = dr._native()
    g = originals.capsule.context.geometry
    shape, nc = (g.ny, g.nx), len(originals.capsule.context.grain_classes.classes)
    source = _array(arrays[5], shape, "saved routing source", readonly=True)
    velocity = _array(arrays[6], (*shape, nc), "saved routing velocity", readonly=True)
    if bool(n.np.any(velocity != 0.0)):
        _fail("transport_memory_not_empty", "saved velocity is not empty")
    if (
        db._sha256(db._array_digest(source)) != record["source_sha256"]
        or _cache_fingerprint(originals.water.terrain)
        != record["terrain_original_sha256"]
    ):
        _fail("routing_identity", "original terrain/saved hydraulic source differs")
    graph = n.routing_graph_from_elevation(originals.water.terrain, source, xp=n.np)
    network, grid = n.transport_network(graph), n.physics_grid_from_graph(graph)
    _cache_admission(graph, network, grid, originals.water.terrain, shape, nc, limits)
    projection = _RestoredRouting(
        graph,
        originals.water.terrain,
        network,
        grid,
        source,
        velocity,
        None,
        _routing_identity(
            record["terrain_original_sha256"],
            record["source_sha256"],
            *(_cache_fingerprint(value) for value in (graph, network, grid)),
        ),
    )
    actual = _routing_record(projection, originals.water, shape, nc, limits)
    if oi._json(actual) != oi._json(record):
        _fail(
            "routing_identity",
            "native saved-source graph/network/grid full fingerprints differ",
        )
    return projection


def _construct_passive_runner(cls, ticket):
    """Private factory target, immediate checked-loader discipline only."""
    if cls is not dr.DenseRunner or type(ticket) is not _PassiveLoad:
        _fail("restore_ticket", "private immediate passive-loader ticket required")
    fields = ticket.fields
    _keys(fields, RUNNER_FIELDS, "private runner fields")
    if (
        fields["_production"] is not None
        or fields["_stop_receipt"] is not None
        or fields["_active_evidence"] is not None
    ):
        _fail("not_quiescent", "production/stopped/active state unsupported")
    n = dr._native()
    bed, originals = fields["_bed"], ticket.evidence["originals"]
    bed.verify_case_inputs()
    if "record_origin_s" not in inspect.signature(n.finalize_wind_event).parameters:
        _fail(
            "unsupported_native_finalizer",
            "MAPLE must support explicit record_origin_s",
        )
    dr.DenseRunner._check_wind(n, bed, fields["_wind"])
    initial_soil = dr.DenseRunner._check_water(
        n, bed, fields["_water"], fields["_control"], fields["_limits"]
    )
    _same(initial_soil, originals.water.initial_soil_water_m, ticket.evidence["limits"])
    # Full original inputs are independently rehashed; no captured alias/digest is trusted.
    if oi.validate_original_inputs(originals) != ticket.evidence["original_content"]:
        _fail("original_identity", "original inputs changed during preparation")
    probe = SimpleNamespace(**fields)
    parts = dr.DenseRunner._static_parts_record(probe, n)
    if db._sha256(parts) != fields["_static_parts_sha256"]:
        _fail("static_identity", "actual loaded controlling parts differ")
    _same(parts, fields["_static_parts"], ticket.evidence["limits"])
    if db._sha256(fields["_static_record"]) != fields["_static_sha256"]:
        _fail("static_identity", "original static record differs")
    # All26 references prepared above; no initialization/replay/physical reconstruction.
    owner = object.__new__(cls)
    for name in RUNNER_FIELDS:
        setattr(owner, name, fields[name])
    if set(vars(owner)) != set(RUNNER_FIELDS):
        _fail("runner_layout", "closed26 field installation required")
    owner._precheck()
    return owner


def _prepare_material(
    manifest,
    arrays,
    artifact_root,
    expected_source_identity,
    expected_environment_sha256,
    limits,
    *,
    wrapper_raw=0,
    artifacts=None,
):
    _preflight_metadata(
        manifest, expected_source_identity, expected_environment_sha256, limits
    )
    if type(artifacts) is not _Artifacts:
        _fail("restore_ticket", "explicit admitted child resources required")
    BoundaryImageLimits.__post_init__(limits)
    refs = (manifest["originals"], manifest["genesis"], manifest["physical"])
    if (
        artifacts.limits is not limits
        or Path(artifact_root) != artifacts.root
        or oi._json(list(artifacts.refs)) != oi._json(list(refs))
        or type(wrapper_raw) is not int
        or not 0 <= wrapper_raw <= artifacts.wrapper_raw
    ):
        _fail(
            "restore_ticket",
            "manifest/root/limits/raw bytes differ from explicit admission",
        )
    hashes = {r["relative_path"]: artifacts.hashed(r) for r in refs}
    pack = oi.load_original_inputs(
        artifacts.root / refs[0]["relative_path"],
        expected_manifest_sha256=_original_manifest_seal(artifacts, refs[0], limits),
        expected_source_identity=expected_source_identity,
        limits=oi.OriginalPackLimits(
            max_cells=limits.max_cells,
            max_classes=limits.max_classes,
            max_voxels=limits.max_voxels,
            max_cohorts=limits.max_cohorts,
            max_events=limits.max_events,
        ),
    )
    if pack.content_sha256 != refs[0]["identity"]:
        _fail("original_identity", "loaded original content identity differs")
    for ref in refs[1:]:
        if ref["identity"]["caller"]["static_sha256"] != pack.static_sha256:
            _fail(
                "static_identity",
                "native child belongs to different controlling inputs",
            )
    for ref in {r["relative_path"]: r for r in refs[1:]}.values():
        _native_dimensions(artifacts, ref, pack, limits)
    genesis = bridge.load_boundary_physical(
        artifacts.root / refs[1]["relative_path"],
        pack.wind.config,
        refs[1]["identity"],
        inputs=pack.capsule,
        max_bytes=limits.native_max_bytes,
    )
    current = (
        genesis
        if oi._json(refs[1]) == oi._json(refs[2])
        else bridge.load_boundary_physical(
            artifacts.root / refs[2]["relative_path"],
            pack.wind.config,
            refs[2]["identity"],
            inputs=pack.capsule,
            max_bytes=limits.native_max_bytes,
        )
    )
    n = dr._native()
    # Native owning predicate on ACTUAL22/15 current pool, BEFORE routing allocation.
    if type(current.mobile_state) is not n.MobileCohortState:
        _fail("pool_layout", "actual flat native mobile state required")
    if current.mobile_state.mass_kg.size > limits.max_cohorts:
        _fail("profile_cap", "current native cohort capacity")
    if not n.mobile_state_is_genuinely_empty(current.mobile_state):
        _fail("mobile_state_not_empty", "current native pool is not genuinely empty")
    soil, totals, ledger, receipts, state = _validate_history(
        manifest, pack, genesis, current, arrays, limits
    )
    if (
        manifest["shared_resources"]["prescribed_identity_sha256"]
        != db._sha256(pack.capsule.prescribed_digests)
        or current.bed.context is not pack.capsule.context
    ):
        _fail("resource_identity", "original native prescribed/context binding differs")
    routing = _load_routing(manifest["routing"], pack, arrays, limits)
    if current.bed.identity_sha256() != state["accepted_bed_sha256"]:
        _fail("physical_identity", "current accepted bed identity differs")
    if (
        dr.DenseRunner._private_identity(n.np, current.mobile_state, soil)
        != state["accepted_private_sha256"]
    ):
        _fail("private_identity", "current loaded native pool/soil differs")
    inventory0 = n.np.array(pack.initial_inventory_kg, copy=True)
    fields = {
        "_accepted_bed": state["accepted_bed_sha256"],
        "_accepted_private": state["accepted_private_sha256"],
        "_active_evidence": None,
        "_bed": current.bed,
        "_control": pack.control,
        "_inventory0": inventory0,
        "_inventory_elements": state["inventory_elements"],
        "_ledger": ledger,
        "_limits": pack.runner_limits,
        "_n_accepted": state["n_accepted"],
        "_n_classes": state["n_classes"],
        "_next": state["next_index"],
        "_pool": current.mobile_state,
        "_production": None,
        "_receipts": receipts,
        "_routing": routing,
        "_schedule": pack.schedule,
        "_soil": soil,
        "_static_parts": pack.static_parts,
        "_static_parts_sha256": pack.static_parts_sha256,
        "_static_record": pack.static_record,
        "_static_sha256": pack.static_sha256,
        "_stop_receipt": None,
        "_totals": totals,
        "_water": pack.water,
        "_wind": pack.wind,
    }
    ticket = _PassiveLoad(
        fields,
        {"originals": pack, "limits": limits, "original_content": pack.content_sha256},
    )
    owner = dr.DenseRunner._from_passive_boundary_load(ticket)
    artifacts.check()
    for ref in refs:
        if artifacts.hashed(ref) != hashes[ref["relative_path"]]:
            _fail(
                "file_changed", "referenced artifact bytes changed during preparation"
            )
    diagnostics = {
        "schema": SCHEMA,
        "profile": PROFILE,
        "source_identity": copy.deepcopy(expected_source_identity),
        "environment_sha256": expected_environment_sha256,
        "accepted": owner.accepted.as_record(),
        "static_sha256": owner.static_sha256,
        "clock_ledger": owner.ledger_record(),
        "receipt_chain_sha256": state["receipt_chain_sha256"],
        "n_wrapper_arrays": len(arrays),
        "reservation_bytes": artifacts.reservation_bytes,
        "scope": "inert passive source-manifest prototype; no store/adoption/event authorization",
    }
    return _PreparedBoundary(owner, diagnostics)


def prepare_passive_boundary(
    path,
    *,
    artifact_root,
    expected_manifest_sha256,
    expected_source_identity,
    expected_environment_sha256,
    limits=_DEFAULT,
):
    """Prepare and validate a private inert owner; caller receives no live runner."""
    BoundaryImageLimits.__post_init__(limits)
    artifact_root = Path(artifact_root).absolute()
    # Read bounded metadata first, admit ALL child raw sizes before ANY array expansion.
    manifest, _none, raw = _read_wrapper(
        path, expected_manifest_sha256, limits, metadata_only=True
    )
    _preflight_metadata(
        manifest, expected_source_identity, expected_environment_sha256, limits
    )
    artifacts = _Artifacts(
        artifact_root,
        manifest["originals"],
        manifest["genesis"],
        manifest["physical"],
        limits,
        raw,
    )
    try:
        actual, arrays, actual_raw = _read_wrapper(
            path, expected_manifest_sha256, limits, _admitted_raw=raw
        )
        if actual_raw != raw or oi._json(actual) != oi._json(manifest):
            _fail(
                "file_changed",
                "full wrapper read differs from metadata/raw admission basis",
            )
        return _prepare_material(
            actual,
            arrays,
            artifact_root,
            expected_source_identity,
            expected_environment_sha256,
            limits,
            wrapper_raw=raw,
            artifacts=artifacts,
        )
    finally:
        artifacts.close()


def _runner_admission(runner, limits):
    if type(runner) is not dr.DenseRunner or set(vars(runner)) != set(RUNNER_FIELDS):
        _fail("runner_layout", "only closed26-field default DenseRunner")
    if (
        runner._production is not None
        or runner._stop_receipt is not None
        or runner._active_evidence is not None
    ):
        _fail(
            "not_quiescent",
            "only unstopped accepted boundaries without active evidence",
        )
    if type(runner._ledger) is not ClockLedger or set(vars(runner._ledger)) != {
        "epoch",
        "_entries",
    }:
        _fail("clock_layout", "exact ledger layout")
    g, nc = runner._bed.context.geometry, len(runner._bed.context.grain_classes.classes)
    if (
        type(g.ny) is not int
        or type(g.nx) is not int
        or min(g.ny, g.nx) < 1
        or g.ny * g.nx > limits.max_cells
        or not 1 <= nc <= limits.max_classes
        or runner._bed.case_record["n_voxels"] > limits.max_voxels
    ):
        _fail("profile_cap", "bounded Dense dimensions required before traversal")
    for value in (runner._next, runner._n_accepted):
        _int(value, limits.max_events, "accepted counter")
    if (
        runner._next != runner._n_accepted
        or type(runner._receipts) is not list
        or len(runner._receipts) != runner._next
    ):
        _fail("cursor", "closed accepted counters/receipts")
    from maple.aeolian.scheduler import physical_checkpoint as cp

    bridge._check_bed(runner._bed)
    # Reuse34's finite registry for native controlling inputs (including actual NumPy namespace).
    original_limits = oi.OriginalPackLimits(
        max_cells=limits.max_cells,
        max_classes=limits.max_classes,
        max_voxels=limits.max_voxels,
        max_cohorts=limits.max_cohorts,
        max_events=limits.max_events,
    )
    encoder = oi._Encoder(original_limits, oi._registry(), hash_arrays=False)
    encoder.node(
        (
            runner._bed.context,
            runner._wind,
            runner._water,
            runner._control,
            runner._limits,
            runner._inventory0,
            runner._soil,
            runner._static_parts,
            runner._static_record,
            runner._receipts,
        )
    )
    from masy.dense_schedule import DenseSchedule, RainEvent, RainInterval, WindEvent

    _layout(runner._schedule, DenseSchedule, ("events",))
    if (
        type(runner._schedule.events) is not tuple
        or not 1 <= len(runner._schedule.events) <= limits.max_events
    ):
        _fail("profile_cap", "bounded schedule before canonical traversal")
    for event in runner._schedule.events:
        if type(event) is WindEvent:
            _layout(
                event,
                WindEvent,
                ("duration_s", "shear_velocity_m_s", "direction_deg_cw_from_north"),
            )
            encoder.node(vars(event))
        elif type(event) is RainEvent:
            _layout(event, RainEvent, ("intervals", "recession_guard_s"))
            if (
                type(event.intervals) is not tuple
                or not 1 <= len(event.intervals) <= 64
            ):
                _fail("profile_cap", "bounded rainfall intervals before traversal")
            encoder.node(event.recession_guard_s)
            for interval in event.intervals:
                _layout(interval, RainInterval, ("duration_s", "intensity_mm_h"))
                encoder.node(vars(interval))
        else:
            _fail("profile_cap", "unsupported event type")
    bridge._budget(
        cp,
        limits.native_max_bytes,
        runner._bed.context,
        runner._pool,
        runner._bed._held_inputs,
        *(getattr(runner._bed.state, name) for name in db.SHARED_FIELDS),
    )
    if runner._pool.mass_kg.size > limits.max_cohorts:
        _fail("profile_cap", "bounded cohort capacity")
    _array(runner._soil, (g.ny, g.nx), "accepted soil")
    _keys(runner._totals, TOTALS_FIELDS, "accepted totals")
    for name in ("export", "input", "bound", "internal_terminal_deposition"):
        _array(runner._totals[name], (nc,), name)
    if runner._routing is not None:
        _routing_record(runner._routing, runner._water, (g.ny, g.nx), nc, limits)
    for receipt in runner._receipts:
        _receipt_shape(receipt, nc, limits)
    runner._precheck()
    if not dr._native().mobile_state_is_genuinely_empty(runner._pool):
        _fail("mobile_state_not_empty", "accepted native pool not genuinely empty")


def _capture_wrapper_arrays(runner, limits):
    """Only fixed ndarray metadata; wrapper byte/count admission precedes all hashes."""
    import numpy as np

    if type(runner) is not dr.DenseRunner or set(vars(runner)) != set(RUNNER_FIELDS):
        _fail("runner_layout", "closed26-field default DenseRunner required")
    _keys(runner._totals, TOTALS_FIELDS, "accepted totals")
    arrays = [
        runner._soil,
        *(
            runner._totals[name]
            for name in ("export", "input", "bound", "internal_terminal_deposition")
        ),
    ]
    held = runner._routing
    if held is not None:
        if type(held) is _RestoredRouting:
            if any(not hasattr(held, name) for name in _RestoredRouting.__slots__):
                _fail(
                    "restored_routing_layout_changed", "private projection slot absent"
                )
            velocity = held.velocity
        elif type(held) is SimpleNamespace and set(vars(held)) == {
            "graph",
            "terrain",
            "network",
            "grid",
            "source",
            "template",
        }:
            _layout(
                held.template,
                _native_types()[-1],
                ROUTING_CONTRACT["SedimentEventState"]["fields"],
            )
            velocity = held.template.sediment_velocity_m_s
        else:
            _fail("restored_routing_layout_changed", "closed held routing required")
        arrays.extend((held.source, velocity))
    if len(arrays) > limits.max_files or any(
        type(array) is not np.ndarray for array in arrays
    ):
        _fail("array_cap", "fixed host wrapper file inventory before traversal")
    if sum(array.nbytes for array in arrays) > limits.max_numeric_bytes:
        _fail("array_cap", "wrapper numeric budget before content scan/hash")
    return arrays


def _snapshot(
    runner, originals, genesis, physical, source_identity, environment_sha256, limits
):
    for ref, fmt in zip(
        (originals, genesis, physical),
        (_ORIGINAL_FORMAT, _NATIVE_FORMAT, _NATIVE_FORMAT),
        strict=True,
    ):
        _reference(ref, fmt)
    arrays = _capture_wrapper_arrays(runner, limits)
    _runner_admission(runner, limits)
    routing = None
    if runner._routing is not None:
        g = runner._bed.context.geometry
        routing = _routing_record(
            runner._routing, runner._water, (g.ny, g.nx), runner._n_classes, limits
        )
        # Compare native factory output once on capture; do not refresh accepted caches.
        n = dr._native()
        expected_graph = n.routing_graph_from_elevation(
            runner._water.terrain, runner._routing.source, xp=n.np
        )
        expected_network, expected_grid = (
            n.transport_network(expected_graph),
            n.physics_grid_from_graph(expected_graph),
        )
        _cache_admission(
            expected_graph,
            expected_network,
            expected_grid,
            runner._water.terrain,
            (g.ny, g.nx),
            runner._n_classes,
            limits,
        )
        from maple_syrup.sediment_bed import check_graph_evidence_matches

        check_graph_evidence_matches(expected_graph, runner._routing.graph)
        for actual, expected in zip(
            (runner._routing.graph, runner._routing.network, runner._routing.grid),
            (expected_graph, expected_network, expected_grid),
            strict=True,
        ):
            if _cache_fingerprint(actual) != _cache_fingerprint(expected):
                _fail(
                    "routing_identity",
                    "accepted cache differs from native retained-source factory",
                )
    if len(arrays) not in (5, 7) or len(arrays) > limits.max_files:
        _fail("array_cap", "wrapper5/7 file count and configured cap before capture")
    if sum(a.nbytes for a in arrays) > limits.max_numeric_bytes:
        _fail("array_cap", "wrapper final numeric inventory cap")
    rows = _receipt_rows(runner._receipts, limits)
    manifest = {
        "schema": SCHEMA,
        "profile": PROFILE,
        "source_identity": copy.deepcopy(source_identity),
        "environment_sha256": environment_sha256,
        "originals": copy.deepcopy(originals),
        "genesis": copy.deepcopy(genesis),
        "physical": copy.deepcopy(physical),
        "state": {
            "next_index": runner._next,
            "n_accepted": runner._n_accepted,
            "n_classes": runner._n_classes,
            "inventory_elements": runner._inventory_elements,
            "native_time": encode_time(runner._bed.time_s),
            "accepted_bed_sha256": runner._accepted_bed,
            "accepted_private_sha256": runner._accepted_private,
            "receipt_chain_sha256": _chain(originals, genesis, rows),
        },
        "totals": _totals_wire(runner._totals),
        "ledger": runner._ledger.as_record(),
        "receipts": rows,
        "routing": routing,
        "shared_resources": {
            "schema": "masy.shared_surface_resources.v0",
            "version": 0,
            "profile": "static_prescribed_native_dry_reset_v0",
            "prescribed_identity_sha256": db._sha256(runner._bed._prescribed_digests),
            "soil_array": "current_soil",
            "update_cursor": 0,
            "update_receipts": [],
        },
        "arrays": [],
    }
    _preflight_metadata(manifest, source_identity, environment_sha256, limits)
    return manifest, arrays


def _writer_path_identity(path, directory):
    observed = oi._open_dir(path)
    try:
        a, b = os.fstat(observed), os.fstat(directory)
        if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
            _fail(
                "file_changed",
                "writer ancestor/image path no longer names held directory",
            )
    finally:
        os.close(observed)


def save_passive_boundary(
    path,
    runner,
    *,
    artifact_root,
    originals,
    genesis,
    physical,
    source_identity,
    environment_sha256,
    limits=_DEFAULT,
):
    """Seal a NEW immutable wrapper image. No HEAD, adoption or overwrite.

    All referenced originals/native children must already be sealed and retained.
    A post-link exception reports uncertainty and preserves all files for inspection.
    """
    BoundaryImageLimits.__post_init__(limits)
    artifact_root = Path(artifact_root).absolute()
    oi._source_identity(source_identity)
    _digest(environment_sha256, "environment")
    _mixed_static_writer_gate(path, runner, artifact_root, originals,
        source_identity, environment_sha256, limits)
    manifest, arrays = _snapshot(
        runner,
        originals,
        genesis,
        physical,
        source_identity,
        environment_sha256,
        limits,
    )
    estimate = (
        sum(a.nbytes + limits.max_header_bytes for a in arrays)
        + limits.max_metadata_bytes
    )
    if estimate > limits.max_serialized_bytes:
        _fail("serialized_cap", "conservative wrapper serialized preflight")
    artifacts = _Artifacts(
        artifact_root,
        manifest["originals"],
        manifest["genesis"],
        manifest["physical"],
        limits,
        estimate,
    )
    try:
        return _write_passive_boundary(
            path,
            runner,
            artifact_root,
            originals,
            genesis,
            physical,
            source_identity,
            environment_sha256,
            limits,
            manifest,
            arrays,
            artifacts,
        )
    finally:
        artifacts.close()


def _write_passive_boundary(
    path,
    runner,
    artifact_root,
    originals,
    genesis,
    physical,
    source_identity,
    environment_sha256,
    limits,
    manifest,
    arrays,
    artifacts,
):
    _mixed_static_writer_gate(path, runner, artifact_root, originals,
        source_identity, environment_sha256, limits, manifest=manifest)
    # _snapshot already admitted the closed5/7 native array layouts and numeric
    # cap. Live native routing is writable; reader preparation requires private
    # readonly carriers. Pre-check the combined reservation before allocation;
    # _Artifacts holds a static bound, with no mutable allocation/charge ledger.
    # leave the owner references/flags and actual serialized capture untouched.
    routing_copy_bytes = sum(array.nbytes for array in arrays[5:])
    writer_reservation_bytes = artifacts.reservation_bytes + routing_copy_bytes
    if writer_reservation_bytes > 2 << 30:
        _fail("memory_cap", "writer routing copies exceed bounded readback reservation")
    validation_arrays = list(arrays)
    staged = None
    for index in range(5, len(arrays)):
        staged = arrays[index].copy(order="C")
        staged.flags.writeable = False
        validation_arrays[index] = staged
    # Actual child readback validates original/genesis/current anchors BEFORE creating this image.
    checked = _prepare_material(
        manifest,
        validation_arrays,
        artifact_root,
        source_identity,
        environment_sha256,
        limits,
        artifacts=artifacts,
    )
    checked_owner = checked._PreparedBoundary__owner
    _same(checked_owner._pool, runner._pool, limits)
    _same(checked_owner._inventory0, runner._inventory0, limits)
    _same(checked_owner._soil, runner._soil, limits)
    if checked_owner._bed.identity_sha256() != runner._bed.identity_sha256():
        _fail(
            "physical_identity",
            "native reference does not represent the current accepted owner",
        )
    del checked_owner, checked, validation_arrays, staged
    target = Path(path)
    if target.name in ("", ".", ".."):
        _fail("path_refused", "new boundary image name required")
    estimate = (
        sum(a.nbytes + limits.max_header_bytes for a in arrays)
        + limits.max_metadata_bytes
    )
    if estimate > limits.max_serialized_bytes:
        _fail("serialized_cap", "conservative wrapper serialized preflight")
    parent, directory = oi._open_dir(target.parent), None
    linked = False
    created = False
    image_identity = None
    try:
        space = os.fstatvfs(parent)
        if (
            space.f_bavail * space.f_frsize < estimate + (1 << 20)
            or space.f_favail < len(arrays) + 3
        ):
            _fail("storage_cap", "insufficient disk/inodes with bounded headroom")
        os.mkdir(target.name, mode=0o700, dir_fd=parent)
        created = True
        before = os.stat(target.name, dir_fd=parent, follow_symlinks=False)
        directory = os.open(
            target.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent,
        )
        if (os.fstat(directory).st_dev, os.fstat(directory).st_ino) != (
            before.st_dev,
            before.st_ino,
        ) or oi._names(directory, limits):
            _fail("file_changed", "exclusive image directory replaced/nonempty")
        image_identity = (before.st_dev, before.st_ino)
        rows, used = [], 0
        import numpy as np

        for index, array in enumerate(arrays):
            name = f"a_{index:04d}.npy"
            fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=directory,
            )
            with os.fdopen(fd, "wb") as stream:
                np.lib.format.write_array(stream, array, allow_pickle=False)
                os.fchmod(stream.fileno(), 0o400)
                stream.flush()
                os.fsync(stream.fileno())
            fd, s = oi._open_file(directory, name)
            try:
                used += s.st_size
                if (
                    used > limits.max_serialized_bytes
                    or s.st_size > limits.max_numeric_bytes + limits.max_header_bytes
                ):
                    _fail(
                        "serialized_cap",
                        "actual written raw per-file/aggregate cap before hash",
                    )
                rows.append(
                    {
                        "name": name,
                        "dtype": array.dtype.str,
                        "shape": list(array.shape),
                        "data_bytes": array.nbytes,
                        "data_sha256": hashlib.sha256(
                            memoryview(array).cast("B")
                        ).hexdigest(),
                        "file_bytes": s.st_size,
                        "sha256": oi._hash_fd(fd, s.st_size),
                    }
                )
            finally:
                os.close(fd)
        manifest["arrays"] = rows
        for row in rows:
            oi._verify_written(directory, row, limits)
        again, _live_arrays = _snapshot(
            runner,
            originals,
            genesis,
            physical,
            source_identity,
            environment_sha256,
            limits,
        )
        again["arrays"] = rows
        if oi._json(again) != oi._json(manifest):
            _fail("accepted_state_changed", "accepted wrapper changed during writing")
        # Prepare actual written readback privately before sealing any acceptance-visible metadata.
        unsigned = {
            key: manifest[key] for key in ROOT_FIELDS if key != "manifest_sha256"
        }
        manifest["manifest_sha256"] = _sha(unsigned)
        raw = oi._json(manifest)
        _mixed_static_writer_gate(path, runner, artifact_root, originals,
            source_identity, environment_sha256, limits, manifest=manifest, raw=raw)
        if (
            len(raw) > limits.max_metadata_bytes
            or used + len(raw) > limits.max_serialized_bytes
        ):
            _fail("serialized_cap", "actual wrapper metadata/aggregate cap")
        fd = os.open(
            "manifest.tmp",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            os.fchmod(stream.fileno(), 0o400)
            stream.flush()
            os.fsync(stream.fileno())
        fd, s = oi._open_file(directory, "manifest.tmp")
        try:
            if os.read(fd, len(raw) + 1) != raw or oi._stat_id(
                os.fstat(fd)
            ) != oi._stat_id(s):
                _fail("manifest_identity", "written manifest readback differs")
        finally:
            os.close(fd)
        # The SAME complete reader admits the actual unlinked manifest/NPY bytes.
        written_manifest, written_arrays, written_raw = _read_wrapper(
            target, manifest["manifest_sha256"], limits, _manifest_name="manifest.tmp"
        )
        _prepare_material(
            written_manifest,
            written_arrays,
            artifact_root,
            source_identity,
            environment_sha256,
            limits,
            wrapper_raw=written_raw,
            artifacts=artifacts,
        )
        final_record, final_arrays = _snapshot(
            runner,
            originals,
            genesis,
            physical,
            source_identity,
            environment_sha256,
            limits,
        )
        final_record["arrays"] = rows
        if oi._json(final_record) != oi._json(unsigned):
            _fail(
                "accepted_state_changed",
                "accepted wrapper metadata changed before seal",
            )
        for written, live in zip(written_arrays, final_arrays, strict=True):
            _same(written, live, limits)
        for row in rows:
            oi._verify_written(directory, row, limits)
        _writer_path_identity(target.parent, parent)
        _writer_path_identity(target, directory)
        os.fsync(directory)
        artifacts.check()
        linked = True  # Publication can occur from this syscall onward, including interruption.
        os.link(
            "manifest.tmp",
            "manifest.json",
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
        os.unlink("manifest.tmp", dir_fd=directory)
        os.fsync(directory)
        os.fsync(parent)
        _writer_path_identity(target.parent, parent)
        _writer_path_identity(target, directory)
        artifacts.check()
        return manifest["manifest_sha256"]
    except BaseException as exc:
        if linked:
            identity = image_identity
            message = f"boundary seal may be published at {target}, directory {identity}; durability uncertain; inspect retained files"
            if not isinstance(exc, Exception):
                exc.add_note(message)
                raise
            raise BoundaryPublicationUncertain(message) from exc
        if created:
            exc.add_note(
                f"unsealed boundary image retained at {target}; retry requires a new exclusive name"
            )
        raise
    finally:
        if directory is not None:
            os.close(directory)
        os.close(parent)




# Explicit passive dynamic artifacts. C/v1 entrypoints never select these APIs.
SCHEMA_V2 = "masy.dense_runner_boundary.v2"
PROFILE_V2 = "dynamic_manual_dense_numpy_accepted_v1"
_ORIGINAL_FORMAT_V2 = oi.SCHEMA_V2
_NATIVE_CALLER_SCHEMA_V2 = "masy.dense_runner_boundary_native.v2"
RUNNER_FIELDS_V2 = (*RUNNER_FIELDS, "_vegetation_plan", "_vegetation_copied_bytes")
_VEGETATION_BODY_KEYS_V2 = ("input_version", "input_version_sha256",
    "input_native_content_sha256", "output_version", "output_version_sha256",
    "output_native_content_sha256", "binding_sha256", "delta_receipts",
    "water_input_mapping", "water_exchange", "required_snapshot_bytes",
    "delta_copy_bytes", "accepted_run_copy_bytes")
_WATER_MAPPING_V2 = "owned_current_cover_fraction_only;all_five_fields_retained_in_shared_envelope"
_PASSIVE_V2_TOKEN = object()


@dataclasses.dataclass(frozen=True, eq=False, slots=True)
class _PassiveLoadV2:
    fields: dict
    evidence: dict
    _authority: object


class _PreparedBoundaryV2:
    """Inert v2 owner; intentionally distinct from C's accepted v1 carrier."""
    __slots__ = ("__evidence", "__owner")

    def __init__(self, owner, evidence):
        self.__owner, self.__evidence = owner, copy.deepcopy(evidence)

    def diagnostics(self):
        return copy.deepcopy(self.__evidence)


def _vegetation_body_v2(body, limits):
    from masy import landscape_vegetation as lv
    value = body["vegetation"]
    _keys(value, _VEGETATION_BODY_KEYS_V2, "dynamic receipt vegetation")
    lv._json_admit(value, max_bytes=limits.max_receipt_bytes, max_nodes=3551)
    for name in ("input_version", "output_version"):
        _int(value[name], lv._CEILINGS["max_total_updates"], name)
    for name in ("input_version_sha256", "input_native_content_sha256",
            "output_version_sha256", "output_native_content_sha256", "binding_sha256"):
        _digest(value[name], name)
    for name in ("required_snapshot_bytes", "delta_copy_bytes", "accepted_run_copy_bytes"):
        _int(value[name], lv._CEILINGS["max_run_copy_bytes"], name)
    rows = value["delta_receipts"]
    if (type(rows) is not list or len(rows) > lv._CEILINGS["max_total_updates"]
            or value["water_input_mapping"] != _WATER_MAPPING_V2
            or type(value["water_exchange"]) is not list or value["water_exchange"] != []):
        _fail("resource_profile", "closed manual empty-exchange receipt required")
    for row in rows:
        _keys(row, lv._RECEIPT_KEYS, "dynamic delta receipt")


def _preflight_resources_v2(value, limits):
    from masy import landscape_vegetation as lv
    _keys(value, bridge._RESOURCE_KEYS_V2, "dynamic shared resources")
    lv._json_admit(value, max_bytes=limits.max_metadata_bytes, max_nodes=3600)
    if (value["schema"] != bridge.RESOURCE_SCHEMA_V2
            or value["profile"] != PROFILE_V2
            or value["soil_array"] != "current_soil"
            or type(value["water_exchange"]) is not list or value["water_exchange"] != []):
        _fail("resource_profile", "only dynamic manual empty-exchange resources")
    _int(value["version"], lv._CEILINGS["max_total_updates"], "resource version")
    _int(value["accepted_run_copy_bytes"], lv._CEILINGS["max_run_copy_bytes"], "run copies")
    for key in ("prescribed_identity_sha256", "vegetation_metadata_sha256"):
        _digest(value[key], key)
    env = value["vegetation"]
    _keys(env, lv._ENVELOPE_KEYS, "dynamic vegetation envelope")
    lv._json_admit(env, max_bytes=lv._CEILINGS["max_metadata_bytes"], max_nodes=3551)
    if (env["metadata_sha256"] != value["vegetation_metadata_sha256"]
            or type(env["version"]) is not int or env["version"] != value["version"]):
        _fail("resource_identity", "independent metadata/version seals differ")
    soil = value["soil_identity"]
    _keys(soil, ("array",), "soil identity")
    _keys(soil["array"], ("dtype", "shape", "order", "nbytes", "sha256"), "soil array")
    shape = soil["array"]["shape"]
    if (type(shape) is not list or len(shape) != 2
            or any(type(v) is not int or v < 1 for v in shape)
            or math.prod(shape) > limits.max_cells):
        _fail("resource_identity", "bounded current soil dimensions required")
    bridge._soil_identity_v2(soil, tuple(shape))


def _reference_v2(value, expected_format):
    _keys(value, ARTIFACT_REF_FIELDS, "artifact reference")
    path = value["relative_path"]
    try:
        path_bytes = path.encode() if type(path) is str else b""
    except UnicodeEncodeError:
        _fail("artifact_reference", "artifact path must be valid UTF-8")
    if (
        type(value["format"]) is not str
        or value["format"] != expected_format
        or type(path) is not str
        or not path
        or len(path_bytes) > 4096
        or path.startswith("/")
        or any(x in ("", ".", "..") for x in path.split("/"))
        or len(path.split("/")) > 24
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
        or any(len(component.encode()) > 255 for component in path.split("/"))
    ):
        _fail("artifact_reference", "closed relative artifact path/format required")
    _digest(value["manifest_file_sha256"], "artifact manifest")
    if expected_format == _ORIGINAL_FORMAT_V2:
        _digest(value["identity"], "original content")
    else:
        identity = value["identity"]
        _keys(identity, bridge._IDENTITY_V2, "native22 identity")
        if type(identity["schema"]) is not str or identity["schema"] != bridge.SCHEMA_V2:
            _fail("artifact_reference", "existing native22 schema required")
        for key in ("original_inputs_sha256", "case_sha256", "bed_identity_sha256", "plan_sha256", "vegetation_metadata_sha256"):
            _digest(identity[key], key)
        caller = identity["caller"]
        _keys(
            caller,
            ("schema", "source_identity", "environment_sha256", "static_sha256", "shared_resource_record_sha256"),
            "native caller",
        )
        if (
            type(caller["schema"]) is not str
            or caller["schema"] != _NATIVE_CALLER_SCHEMA_V2
        ):
            _fail("artifact_reference", "native caller schema required")
        oi._source_identity(caller["source_identity"])
        _digest(caller["environment_sha256"], "native environment")
        _digest(caller["static_sha256"], "native static")
        _digest(caller["shared_resource_record_sha256"], "native resources")
    return path


class _ArtifactsV2(_Artifacts):
    def __init__(self, root, originals, genesis, current, limits, wrapper_raw=0):
        BoundaryImageLimits.__post_init__(limits)
        for reference, fmt in zip(
            (originals, genesis, current),
            (_ORIGINAL_FORMAT_V2, _NATIVE_FORMAT, _NATIVE_FORMAT),
            strict=True,
        ):
            _reference_v2(reference, fmt)
        self.root, self.refs, self.limits = (
            Path(root).absolute(),
            tuple(copy.deepcopy(value) for value in (originals, genesis, current)),
            limits,
        )
        self.wrapper_raw = _int(
            wrapper_raw, limits.max_serialized_bytes, "admitted wrapper raw bytes"
        )
        self.fds, self.dirs, self.files = [], {}, {}
        self.raw = wrapper_raw
        try:
            for ref, fmt in zip(
                self.refs,
                (_ORIGINAL_FORMAT_V2, _NATIVE_FORMAT, _NATIVE_FORMAT),
                strict=True,
            ):
                relative = _reference_v2(ref, fmt)
                if relative in self.files:
                    if self.files[relative][3] != fmt:
                        _fail(
                            "artifact_reference",
                            "one child path cannot name two formats",
                        )
                    continue
                parent_path = self.root
                for part in (None, *relative.split("/")):
                    if part is not None:
                        parent_path /= part
                    if parent_path not in self.dirs:
                        fd = oi._open_dir(parent_path)
                        self.fds.append(fd)
                        self.dirs[parent_path] = (fd, os.fstat(fd))
                directory = self.dirs[parent_path][0]
                names = []
                with os.scandir(directory) as entries:
                    for entry in entries:
                        names.append(entry.name)
                        if len(names) > (258 if fmt == _ORIGINAL_FORMAT_V2 else 4):
                            _fail("file_cap", "bounded child entry count")
                expected = {"checkpoint.json", "native_state.npz", "pending_state.npz"}
                if fmt == _NATIVE_FORMAT:
                    extra = set(names) - expected
                    if not expected <= set(names) or len(extra) > 1:
                        _fail("unlisted_file", "native literal file inventory")
                    for name in extra:
                        if re.fullmatch(r"\.manifest-[0-9a-f]{32}", name) is None:
                            _fail("unlisted_file", "unknown native entry")
                else:
                    if "manifest.json" not in names or any(
                        name not in ("manifest.json", "manifest.tmp")
                        and re.fullmatch(r"a_[0-9]{4}\.npy", name) is None
                        for name in names
                    ):
                        _fail("unlisted_file", "original fixed file inventory")
                sizes, total, rows = {}, 0, {}
                for name in names:
                    s = os.stat(name, dir_fd=directory, follow_symlinks=False)
                    if not stat.S_ISREG(s.st_mode):
                        _fail("file_type", "child is not regular/no-follow")
                    manifest_name = (
                        "manifest.json"
                        if fmt == _ORIGINAL_FORMAT_V2
                        else "checkpoint.json"
                    )
                    metadata = (
                        name == manifest_name
                        or name == "manifest.tmp"
                        or name.startswith(".manifest-")
                    )
                    file_cap = (
                        (4 << 20)
                        if fmt == _ORIGINAL_FORMAT_V2 and metadata
                        else (
                            72 << 20
                            if fmt == _ORIGINAL_FORMAT_V2
                            else (
                                1 << 20
                                if metadata
                                else limits.native_max_bytes + 327680
                            )
                        )
                    )
                    if s.st_size > file_cap:
                        _fail("serialized_cap", "raw child per-file cap before hash")
                    if name == "manifest.tmp" or name.startswith(".manifest-"):
                        ms = os.stat(
                            manifest_name, dir_fd=directory, follow_symlinks=False
                        )
                        if oi._stat_id(s) != oi._stat_id(ms) or s.st_nlink != 2:
                            _fail(
                                "unlisted_file",
                                "only exact native/34 leftover seal hardlink permitted",
                            )
                    else:
                        if s.st_nlink != (
                            2
                            if any(
                                x == "manifest.tmp" or x.startswith(".manifest-")
                                for x in names
                            )
                            and name == manifest_name
                            else 1
                        ):
                            _fail("file_type", "foreign hard link refused")
                        total += s.st_size
                        sizes[name] = s.st_size
                    rows[name] = oi._stat_id(s)
                if fmt == _NATIVE_FORMAT:
                    if (
                        sum(sizes[n] for n in ("native_state.npz", "pending_state.npz"))
                        > limits.native_max_bytes + 655360
                    ):
                        _fail(
                            "serialized_cap", "native raw aggregate before hash/archive"
                        )
                elif total > 72 << 20:
                    _fail("serialized_cap", "original raw aggregate before hash/header")
                self.raw += total
                if self.raw > limits.max_global_raw_bytes:
                    _fail(
                        "serialized_cap",
                        "combined raw-input admission before child hash/expansion",
                    )
                self.files[relative] = (parent_path, directory, rows, fmt, sizes)
            # Conservative data/readback reservation bound, no eager allocation.
            self.reservation_bytes = self.raw + 2 * (
                (64 << 20)
                + 2 * limits.native_max_bytes
                + limits.max_numeric_bytes
                + limits.max_cache_bytes
            )
            # Total three-C-float64 current-cover/native-soil probe charge.
            self.reservation_bytes += 24 * limits.max_cells
            if self.reservation_bytes > 2 << 30:
                _fail("memory_cap", "bounded component data/readback reservation")
            self.check()
        except BaseException:
            self.close()
            raise


def _read_wrapper_v2(
    path,
    expected_manifest_sha256,
    limits,
    *,
    metadata_only=False,
    _manifest_name="manifest.json",
    _admitted_raw=None,
):
    """Fixed5/7 raw NPY decoder reusing34's bounded prefix/FD primitives."""
    _digest(expected_manifest_sha256, "expected boundary seal")
    directory, opened = oi._open_dir(path), []
    try:
        names = oi._names(directory, limits)
        descriptors, total = {}, 0
        private_readback = _manifest_name == "manifest.tmp"
        if _manifest_name not in names or (
            private_readback and "manifest.json" in names
        ):
            _fail(
                "unsealed",
                "expected boundary manifest absent or invalid private readback",
            )
        for name in sorted(names):
            fd, s = oi._open_file(directory, name)
            opened.append(fd)
            descriptors[name] = (fd, s)
            if name == "manifest.tmp" and not private_readback:
                prior = descriptors["manifest.json"][1]
                if oi._stat_id(prior) != oi._stat_id(s) or s.st_nlink != 2:
                    _fail(
                        "unlisted_file",
                        "only exact leftover own seal hardlink permitted",
                    )
            else:
                total += s.st_size
                cap = (
                    limits.max_metadata_bytes
                    if name == _manifest_name
                    else limits.max_numeric_bytes + limits.max_header_bytes
                )
                if s.st_size > cap or total > limits.max_serialized_bytes:
                    _fail(
                        "serialized_cap",
                        "wrapper raw file/aggregate cap before hash/header",
                    )
                if s.st_nlink != (
                    2
                    if not private_readback
                    and name == "manifest.json"
                    and "manifest.tmp" in names
                    else 1
                ):
                    _fail("file_type", "foreign wrapper hardlink refused")
        if _admitted_raw is not None and total != _admitted_raw:
            _fail(
                "file_changed", "wrapper raw bytes changed before second numerical read"
            )
        fd, ms = descriptors[_manifest_name]
        raw = os.read(fd, ms.st_size + 1)
        manifest = oi._strict_json(raw, limits)
        _keys(manifest, ROOT_FIELDS, "boundary manifest")
        seal = manifest["manifest_sha256"]
        if (
            manifest["schema"] != SCHEMA_V2
            or manifest["profile"] != PROFILE_V2
            or seal != expected_manifest_sha256
            or _sha({k: v for k, v in manifest.items() if k != "manifest_sha256"})
            != seal
            or oi._json(manifest) != raw
        ):
            _fail("manifest_identity", "canonical boundary seal/schema/profile differs")
        rows = manifest["arrays"]
        if (
            type(rows) is not list
            or len(rows) not in (5, 7)
            or len(rows) > limits.max_files
        ):
            _fail("array_layout", "exact5/7 wrapper arrays required")
        required = {_manifest_name} | (
            {"manifest.tmp"}
            if not private_readback and "manifest.tmp" in names
            else set()
        )
        numeric = 0
        # All declarations and aggregate caps admitted BEFORE payload hash/header/expansion.
        for i, row in enumerate(rows):
            _keys(row, _ARRAY_KEYS, "wrapper array")
            shape = row["shape"]
            if (
                row["name"] != f"a_{i:04d}.npy"
                or row["dtype"] != "<f8"
                or type(shape) is not list
                or len(shape) not in (1, 2, 3)
                or any(type(d) is not int or not 0 < d <= 64 for d in shape)
            ):
                _fail("array_layout", "fixed numeric dtype/name/shape")
            size = math.prod(shape) * 8
            if type(row["data_bytes"]) is not int or row["data_bytes"] != size:
                _fail("array_layout", "raw numeric byte count")
            numeric += size
            if numeric > limits.max_numeric_bytes or row["name"] not in descriptors:
                _fail(
                    "array_cap", "wrapper aggregate numeric/file cap before expansion"
                )
            _digest(row["sha256"], "numeric file hash")
            _digest(row["data_sha256"], "numeric data hash")
            if (
                type(row["file_bytes"]) is not int
                or descriptors[row["name"]][1].st_size != row["file_bytes"]
            ):
                _fail("array_layout", "raw file size differs")
            required.add(row["name"])
        if set(names) != required:
            _fail("unlisted_file", "unknown boundary artifact entry")
        if metadata_only:
            for name, (handle, before) in descriptors.items():
                if oi._stat_id(os.fstat(handle)) != oi._stat_id(before) or oi._stat_id(
                    os.stat(name, dir_fd=directory, follow_symlinks=False)
                ) != oi._stat_id(before):
                    _fail(
                        "file_changed", "wrapper metadata admission changed during read"
                    )
            return manifest, [], total
        arrays = _decode_wrapper_arrays(directory, descriptors, rows, limits)
        for name, (fd, before) in descriptors.items():
            if oi._stat_id(os.fstat(fd)) != oi._stat_id(before) or oi._stat_id(
                os.stat(name, dir_fd=directory, follow_symlinks=False)
            ) != oi._stat_id(before):
                _fail("file_changed", "wrapper file changed during read")
        return manifest, arrays, total
    finally:
        for fd in opened:
            os.close(fd)
        os.close(directory)


def _receipt_shape_v2(body, nc, limits):
    if type(body) is not dict:
        _fail("receipt_layout", "plain receipt body required")
    kind = body.get("kind")
    if type(kind) is not str or kind not in RECEIPT_BODIES:
        _fail("receipt_layout", "unknown receipt kind")
    spec = RECEIPT_BODIES[kind]
    _keys(body, (*RECEIPT_COMMON, "vegetation", *spec), "receipt")
    for section, names in spec.items():
        if type(names) is tuple:
            _keys(body[section], names, f"receipt.{section}")
    _keys(
        body["aggregate"],
        ("joint_residual_kg", "joint_bound_kg", "n_summation_terms"),
        "aggregate",
    )
    for key in ("joint_residual_kg", "joint_bound_kg"):
        _vector(body["aggregate"][key], nc, key)
    _int(body["aggregate"]["n_summation_terms"], 1 << 30, "sum terms")
    for key in ("static_sha256", "bed_identity_before", "bed_identity_after"):
        _digest(body[key], key)
    _int(body["index"], limits.max_events - 1, "receipt index")
    if (
        body["schema"] != "masy.dense_runner_event.v2"
        or body["status"] != "accepted"
        or body["code"] != "accepted"
    ):
        _fail("receipt_layout", "only accepted default receipt schema")
    _vegetation_body_v2(body, limits)
    for key, value in body["classes"].items():
        if value is None and key in (
            "detachment_kg",
            "flow_end_mobile_before_settlement_kg",
        ):
            continue
        if kind == "rain" and key == "terminal_sediment_tolerance_kg":
            # Native reservoir_bound_kg returns one scalar; unchanged DenseRunner
            # encodes it as a one-element list, independent of class count.
            _vector(value, 1, key)
        elif key in (
            "native_bound_kg",
            "event_bound_kg",
            "event_arithmetic_scale_kg",
        ) or (kind == "wind" and key == "inventory_bound_kg"):
            _float(value, key)
        elif key == "event_summation_terms":
            if type(value) is not int or value != 6:
                _fail("receipt_layout", "wind event terms")
        else:
            _vector(value, nc, key)
    if kind == "wind":
        if body["wind_end_pool"] != "genuinely_empty":
            _fail("receipt_layout", "wind mobile completion missing")
        _keys(
            body["native"]["block"],
            ("start_time_s", "end_time_s", "duration_s"),
            "native block",
        )
        for value in body["native"]["block"].values():
            decode_time(value)
        for name, value in body["native"].items():
            if name == "block":
                continue
            if name in (
                "advanced_end_time_s",
                "advanced_duration_s",
                "physical_time_s",
            ):
                decode_time(value)
            elif name in ("process_mode", "integration_path"):
                if type(value) is not str or len(value.encode()) > 128:
                    _fail("receipt_layout", "native mode/path")
            elif name == "conservation_residual_kg":
                _float(value, name)
            else:
                _int(value, 1 << 30, name)
        for key, value in body["coverage"].items():
            if key == "requested_s":
                decode_time(value)
            elif key == "native_plan_steps":
                _int(value, 1 << 30, key)
            else:
                if (
                    type(value) is not str
                    or len(value) > 4096
                    or _FRACTION.fullmatch(value) is None
                ):
                    _fail(
                        "receipt_layout",
                        "coverage requires bounded canonical rational syntax",
                    )
                numerator, denominator = (int(part) for part in value.split("/"))
                if math.gcd(numerator, denominator) != 1:
                    _fail("receipt_layout", "noncanonical coverage fraction")
        final = body["finalization"]
        for key, value in final.items():
            if key in ("record_origin_s", "physical_time_s"):
                decode_time(value)
            elif key in ("record",):
                pass  # compared to the complete actual native scalar producer below
            elif key == "record_sha256":
                _digest(value, key)
            elif key in ("completed_block_index", "n_settled", "n_exported"):
                _int(value, 1 << 30, key)
            elif key in (
                "status",
                "algorithm_version",
                "representation",
                "placement_rule",
            ):
                if type(value) is not str or len(value.encode()) > 4096:
                    _fail("receipt_layout", "finalization string")
            elif key.endswith("by_class_kg"):
                _vector(value, nc, key)
            else:
                _float(value, key)
    else:
        if type(body["native"]["rainfall_edges_s"]) is not list:
            _fail("receipt_layout", "plain bounded rainfall edge list required")
        for key, value in body["native"].items():
            if key == "rainfall_edges_s":
                if type(value) is not list or not 2 <= len(value) <= 65:
                    _fail("receipt_layout", "bounded rainfall edges")
                for item in value:
                    decode_time(item)
            elif key == "intensity_mm_h":
                if (
                    type(value) is not list
                    or len(value) != len(body["native"]["rainfall_edges_s"]) - 1
                ):
                    _fail("receipt_layout", "rainfall intensity count")
                for item in value:
                    _float(item, key)
            elif key.endswith("_s"):
                decode_time(value)
            elif key == "transport_scheme":
                if value != "upwind":
                    _fail("receipt_layout", "only upwind receipt")
            elif key == "phase":
                if value is not None:
                    _fail("receipt_layout", "rain phase must be None")
            else:
                _int(value, 1 << 30, key)
        for key, value in body["routing"].items():
            if key == "entry_net_displacement_m":
                if value is not None:
                    _float(value, key)
            elif key == "n_reroutes_in_event":
                _int(value, 1 << 30, key)
            elif type(value) is not bool:
                _fail("receipt_layout", "routing boolean kind")
        for key, value in body["water"].items():
            if key == "dry_reset":
                if (
                    value
                    != "external dry-again removal (native); soil water is zero after the reset"
                ):
                    _fail("receipt_layout", "native dry-reset profile changed")
            else:
                _float(value, key)


def _preflight_metadata_v2(
    manifest, expected_source_identity, expected_environment_sha256, limits
):
    oi._source_identity(expected_source_identity)
    _digest(expected_environment_sha256, "expected environment")
    if (
        oi._json(manifest["source_identity"]) != oi._json(expected_source_identity)
        or manifest["environment_sha256"] != expected_environment_sha256
    ):
        _fail("source_identity", "boundary source/environment cannot be rebound")
    _reference_v2(manifest["originals"], _ORIGINAL_FORMAT_V2)
    for name in ("genesis", "physical"):
        _reference_v2(manifest[name], _NATIVE_FORMAT)
        identity = manifest[name]["identity"]
        _keys(
            identity,
            (
                "schema",
                "original_inputs_sha256",
                "case_sha256",
                "bed_identity_sha256",
                "caller",
                "plan_sha256",
                "vegetation_metadata_sha256",
            ),
            "native22 identity",
        )
        caller = identity["caller"]
        _keys(
            caller,
            ("schema", "source_identity", "environment_sha256", "static_sha256", "shared_resource_record_sha256"),
            "native caller",
        )
        if (
            caller["schema"] != _NATIVE_CALLER_SCHEMA_V2
            or oi._json(caller["source_identity"]) != oi._json(expected_source_identity)
            or caller["environment_sha256"] != expected_environment_sha256
        ):
            _fail("source_identity", "native child source/environment differs")
        _digest(caller["static_sha256"], "native caller static")
    if manifest["originals"]["relative_path"] in (
        manifest["genesis"]["relative_path"],
        manifest["physical"]["relative_path"],
    ):
        _fail("artifact_reference", "original and native child paths must be disjoint")
    if manifest["genesis"]["relative_path"] == manifest["physical"][
        "relative_path"
    ] and oi._json(manifest["genesis"]) != oi._json(manifest["physical"]):
        _fail(
            "artifact_reference",
            "one native child path requires one identical closed reference",
        )
    state = manifest["state"]
    _keys(state, STATE_FIELDS, "state")
    k = _int(state["next_index"], limits.max_events, "next")
    if type(state["n_accepted"]) is not int or state["n_accepted"] != k:
        _fail("cursor", "next/count differ")
    nc = _int(state["n_classes"], limits.max_classes, "classes")
    if nc == 0:
        _fail("profile_cap", "classes must be positive")
    _int(
        state["inventory_elements"],
        limits.max_cells * (limits.max_voxels + 1),
        "inventory elements",
    )
    decode_time(state["native_time"])
    for key in (
        "accepted_bed_sha256",
        "accepted_private_sha256",
        "receipt_chain_sha256",
    ):
        _digest(state[key], key)
    rows = manifest["receipts"]
    if type(rows) is not list or len(rows) != k:
        _fail("cursor", "receipt count differs")
    used = 0
    for row in rows:
        _keys(row, ("body", "bytes", "sha256"), "receipt row")
        if type(row["body"]) is not dict:
            _fail("receipt_layout", "plain receipt body")
        _receipt_shape_v2(row["body"], nc, limits)
        raw = dr._canonical_json(row["body"]).encode()
        used += len(raw)
        if (
            type(row["bytes"]) is not int
            or row["bytes"] != len(raw)
            or len(raw) > limits.max_receipt_bytes
            or used > limits.max_receipts_bytes
            or row["sha256"] != hashlib.sha256(raw).hexdigest()
        ):
            _fail(
                "receipt_identity", "canonical receipt hash/size/cap before child load"
            )
    if (
        type(manifest["ledger"]) is not dict
        or type(manifest["ledger"].get("entries")) is not list
        or len(manifest["ledger"]["entries"]) != k
    ):
        _fail("clock_layout", "bounded ledger count before decode")
    ClockLedger.from_record(manifest["ledger"])
    rain_seen = any(row["body"]["kind"] == "rain" for row in rows)
    routing = manifest["routing"]
    if (routing is not None) != rain_seen:
        _fail("routing_history", "retained source exists iff accepted rain")
    if routing is not None:
        _keys(routing, ROUTING_FIELDS, "routing")
        if (
            routing["source_array"] != "routing_source"
            or routing["velocity_array"] != "routing_velocity"
            or routing["phase"] is not None
        ):
            _fail("routing_layout", "fixed saved routing roles and None phase")
        for key in ROUTING_FIELDS:
            if key not in ("source_array", "velocity_array", "phase"):
                _digest(routing[key], key)
    _preflight_resources_v2(manifest["shared_resources"], limits)
    if (manifest["physical"]["identity"]["vegetation_metadata_sha256"]
            != manifest["shared_resources"]["vegetation_metadata_sha256"]
            or manifest["physical"]["identity"]["caller"]["shared_resource_record_sha256"]
            != _sha(manifest["shared_resources"])):
        _fail("resource_identity", "native current header does not seal complete resources")
    descriptions = manifest["arrays"]
    # Only internal in-memory authoring has []; persisted images have already
    # passed the SAME fixed5/7 reader before this control-record crosscheck.
    if type(descriptions) is not list:
        _fail("array_layout", "plain wrapper descriptions required")
    if descriptions:
        for index, row in enumerate(descriptions):
            _keys(row, _ARRAY_KEYS, "wrapper array description")
            if (
                row["name"] != f"a_{index:04d}.npy"
                or type(row["shape"]) is not list
                or any(type(v) is not int for v in row["shape"])
            ):
                _fail(
                    "array_layout",
                    "fixed positional file names and exact dimension kinds",
                )
        if type(descriptions) is not list or len(descriptions) != (
            7 if rain_seen else 5
        ):
            _fail("array_layout", "closed5/7 role declarations before child expansion")
        shape = descriptions[0]["shape"]
        if (
            type(shape) is not list
            or len(shape) != 2
            or any(type(v) is not int or v < 1 for v in shape)
            or math.prod(shape) > limits.max_cells
        ):
            _fail("array_layout", "bounded soil dimensions before child expansion")
        shapes = [shape, *([[nc]] * 4)] + ([shape, [*shape, nc]] if rain_seen else [])
        for row, expected in zip(descriptions, shapes, strict=True):
            if row["shape"] != expected:
                _fail("array_layout", "exact raw role shapes before child expansion")
    totals = manifest["totals"]
    _keys(totals, TOTALS_FIELDS, "totals")
    _keys(totals["water"], WATER_FIELDS, "water totals")
    for value in totals["water"].values():
        decode_time(value)
    for key in ("n_entry_reroutes", "n_event_reroutes"):
        _int(totals[key], 1 << 30, key)
    if k == 0 and oi._json(manifest["genesis"]) != oi._json(manifest["physical"]):
        _fail("genesis_identity", "cursor0 requires identical closed native references")


def _validate_history_v2(manifest, originals, genesis, current, arrays, limits):
    import numpy as np

    state = manifest["state"]
    _keys(state, STATE_FIELDS, "state")
    ctx, bed = originals.capsule.context, current.bed
    g, nc = ctx.geometry, len(ctx.grain_classes.classes)
    shape = (g.ny, g.nx)
    k = _int(
        state["next_index"],
        min(limits.max_events, len(originals.schedule.events)),
        "next index",
    )
    for key, expected in (
        ("n_accepted", k),
        ("n_classes", nc),
        (
            "inventory_elements",
            g.ny * g.nx * (originals.capsule.case_record["n_voxels"] + 1),
        ),
    ):
        if type(state[key]) is not int or state[key] != expected:
            _fail("state_identity", f"{key} differs from original/cursor")
    if len(arrays) != (5 if manifest["routing"] is None else 7):
        _fail("array_layout", "routing presence requires exact5/7 arrays")
    soil = _array(arrays[0], shape, "current soil")
    totals_record = manifest["totals"]
    _keys(totals_record, TOTALS_FIELDS, "totals")
    totals = {}
    for i, name in enumerate(
        ("export", "input", "bound", "internal_terminal_deposition"), 1
    ):
        if totals_record[name] != _ROLES[i]:
            _fail("array_layout", "fixed total array roles")
        totals[name] = _array(arrays[i], (nc,), name)
    _keys(totals_record["water"], WATER_FIELDS, "water totals")
    totals["water"] = {
        key: decode_time(totals_record["water"][key]) for key in WATER_FIELDS
    }
    for key in ("n_entry_reroutes", "n_event_reroutes"):
        totals[key] = _int(totals_record[key], 1 << 30, key)
    ledger = ClockLedger.from_record(manifest["ledger"])
    if type(ledger) is not ClockLedger or set(vars(ledger)) != {"epoch", "_entries"}:
        _fail("clock_layout", "exact native clock ledger layout")
    records = ledger.records()
    rows = manifest["receipts"]
    if type(rows) is not list or len(rows) != k or len(records) != k:
        _fail("cursor", "cursor/accepted/ledger/receipt counts differ")
    if (
        encode_time(ledger.epoch) != encode_time(genesis.bed.time_s)
        or encode_time(ledger.epoch) != originals.capsule.case_record["initial_time_s"]
    ):
        _fail("clock_identity", "original/genesis epoch differs")
    if state["native_time"] != encode_time(bed.time_s):
        _fail("clock_identity", "current native time differs")
    _same(
        np.asarray(
            dr._native().bed_inventory(genesis.bed.state), dtype=np.float64
        ).reshape(-1),
        originals.initial_inventory_kg,
        limits,
    )
    _same(genesis.mobile_state, originals.wind.initial_mobile_state, limits)
    _same(
        genesis.bed.state.water.depth_m,
        originals.capsule.held_inputs["initial_water_depth_m"],
        limits,
    )
    n = dr._native()
    _same(
        n.committed_elevation_m(genesis.bed.state.committed_topography),
        originals.water.terrain.initial_committed_elevation_m,
        limits,
    )
    before = genesis.bed.identity_sha256()
    used, receipts, rain_seen = 0, [], False
    zeros = np.zeros(nc, dtype=np.float64)
    fold = {
        key: zeros.copy()
        for key in ("export", "input", "bound", "internal_terminal_deposition")
    }
    fold["water"] = {key: 0.0 for key in WATER_FIELDS}
    fold["n_entry_reroutes"], fold["n_event_reroutes"] = 0, 0
    for index, (row, clock) in enumerate(zip(rows, records, strict=True)):
        _keys(row, ("body", "bytes", "sha256"), "receipt row")
        body = row["body"]
        raw = dr._canonical_json(body).encode()
        used += len(raw)
        if (
            type(row["bytes"]) is not int
            or row["bytes"] != len(raw)
            or len(raw)
            > min(limits.max_receipt_bytes, originals.runner_limits.max_receipt_bytes)
            or used > limits.max_receipts_bytes
            or row["sha256"] != hashlib.sha256(raw).hexdigest()
        ):
            _fail("receipt_identity", "canonical receipt bytes/hash/cap")
        _receipt_shape_v2(body, nc, limits)
        event = originals.schedule.events[index]
        start, end = (
            decode_time(clock["native_start"]),
            decode_time(clock["native_end"]),
        )
        if (
            clock["kind"] != "event"
            or body["index"] != index
            or body["kind"] != event.kind
            or body["static_sha256"] != originals.static_sha256
            or oi._json(body["clock"]) != oi._json(clock)
            or body["bed_identity_before"] != before
        ):
            _fail("receipt_chain", "source/event/clock/bed chain differs")
        expected_returned = (
            body["native"]["advanced_duration_s"] if event.kind == "wind" else None
        )
        if clock["returned_elapsed"] != expected_returned:
            _fail(
                "receipt_clock",
                "native returned duration differs from owning _finish caller",
            )
        if body["aggregate"]["n_summation_terms"] != 2 * state[
            "inventory_elements"
        ] + 2 * (index + 1):
            _fail("aggregate_identity", "historical aggregate arithmetic count differs")
        check_resolved_event(body["resolved"], originals.schedule)
        if oi._json(body["resolved"]) != oi._json(
            resolve_event(originals.schedule, index, start)
        ):
            _fail(
                "receipt_clock",
                "forcing resolution origin differs from original receipt start",
            )
        if index == 0 and clock["commit_before"] != encode_time(
            float(genesis.bed.state.committed_topography.last_commit_time_s)
        ):
            _fail("clock_identity", "genesis committed stamp differs")
        before = body["bed_identity_after"]
        c = body["classes"]
        if event.kind == "wind":
            record = n.resolve_wind_forcing_record(
                {
                    "blocks": [
                        {
                            "shear_velocity_m_s": event.shear_velocity_m_s,
                            "direction_deg": event.direction_deg_cw_from_north,
                            "duration_s": event.duration_s,
                        }
                    ]
                }
            )
            canonical = dr._canonical_input(np, record, "completed_record")
            final = body["finalization"]
            if (
                oi._json(final["record"]) != oi._json(canonical)
                or final["record_sha256"] != db._sha256(canonical)
                or final["record_origin_s"] != encode_time(start)
                or final["physical_time_s"] != encode_time(end)
                or body["native"]["physical_time_s"] != encode_time(end)
            ):
                _fail("receipt_identity", "completed native record/clock differs")
            additions = (
                c["boundary_export_kg"],
                c["boundary_input_kg"],
                np.full(nc, c["inventory_bound_kg"]),
                final["deposited_mass_by_class_kg"],
            )
        else:
            rain_seen = True
            if body["native"]["dry_time_s"] != encode_time(end):
                _fail("receipt_clock", "native dry endpoint differs")
            additions = (
                c["export_kg"],
                c["import_kg"],
                c["inventory_bound_kg"],
                c["internal_terminal_deposition_kg"],
            )
            w = body["water"]
            delta = {
                "rain_m3": w["rain_m3"],
                "drainage_m3": w["drainage_m3"],
                "storm_export_m3": w["export_m3"],
                "dry_surface_removed_m3": w["dry_surface_removed_m3"],
                "dry_soil_removed_m3": w["dry_soil_removed_m3"],
            }
            fold["water"] = {
                key: fold["water"][key] + value for key, value in delta.items()
            }
            fold["n_entry_reroutes"] += int(body["routing"]["entry_rerouted"])
            fold["n_event_reroutes"] += body["routing"]["n_reroutes_in_event"]
        for key, addition in zip(
            ("export", "input", "bound", "internal_terminal_deposition"),
            additions,
            strict=True,
        ):
            fold[key] = fold[key] + np.asarray(addition, dtype=np.float64)
        receipts.append(body)
    _same(fold, totals, limits)
    if not any(body["kind"] == "wind" for body in receipts):
        # Rain passes the same native wind pool through; no first wind has replaced it.
        _same(current.mobile_state, genesis.mobile_state, limits)
    if (
        _chain(manifest["originals"], manifest["genesis"], rows)
        != state["receipt_chain_sha256"]
    ):
        _fail("receipt_chain", "ordered canonical receipt chain differs")
    if rain_seen:
        if bool(np.any(soil != 0.0)):
            _fail(
                "soil_profile",
                "accepted native dry-reset soil must be numerically zero",
            )
    else:
        _same(soil, originals.water.initial_soil_water_m, limits)
    if (manifest["routing"] is not None) != rain_seen:
        _fail("routing_history", "retained routing exists iff rain has been accepted")
    if k:
        if (
            before != bed.identity_sha256()
            or records[-1]["native_end"] != encode_time(bed.time_s)
            or records[-1]["commit_after"]
            != encode_time(float(bed.state.committed_topography.last_commit_time_s))
        ):
            _fail("clock_identity", "last receipt/current bed endpoint differs")
        inventory = dr._vec(np, n.bed_inventory(bed.state))
        joint = (
            inventory
            + totals["export"]
            - totals["input"]
            - originals.initial_inventory_kg
        )
        terms = 2 * state["inventory_elements"] + 2 * k
        scale = max(
            float(np.abs(originals.initial_inventory_kg).sum()),
            float(np.abs(inventory).sum()),
            float(totals["export"].sum()),
            float(totals["input"].sum()),
        )
        allowed = totals["bound"] + n.summation_error_bound_kg(terms, scale)
        if not bool(np.all(np.abs(joint) <= allowed)):
            _fail("aggregate_budget_open", "actual native aggregate bound exceeded")
        _same(
            joint,
            np.asarray(
                receipts[-1]["aggregate"]["joint_residual_kg"], dtype=np.float64
            ),
            limits,
        )
        _same(
            allowed,
            np.asarray(receipts[-1]["aggregate"]["joint_bound_kg"], dtype=np.float64),
            limits,
        )
        if receipts[-1]["aggregate"]["n_summation_terms"] != terms:
            _fail("aggregate_identity", "last aggregate arithmetic count differs")
    elif (
        manifest["genesis"] != manifest["physical"]
        or encode_time(bed.time_s) != encode_time(ledger.epoch)
        or before != bed.identity_sha256()
    ):
        _fail(
            "genesis_identity",
            "cursor0 requires identical closed genesis/current references",
        )
    _dynamic_history_v2(manifest, originals, genesis, current, soil, receipts, limits)
    return soil, totals, ledger, receipts, state



def _dynamic_history_v2(manifest, originals, genesis, current, soil, receipts, limits):
    """Cross-bind genuine restored envelopes to every actual accepted body."""
    env, initial = current.bed._dynamic, genesis.bed._dynamic
    if (env.plan is not initial.plan or env.plan.plan_sha256
            != originals.capsule.case_record["vegetation_profile"]["plan_sha256"]):
        _fail("vegetation_identity", "one original plan must own both loaded endpoints")
    plan, k = env.plan, manifest["state"]["next_index"]
    record, bindings, delta_rows = env.as_record(), env.bindings, env.receipts
    if (record["cursor"]["next_event_index"] != k or len(bindings) != k
            or len(receipts) != k or initial.version != 0
            or initial.cursor["next_event_index"] != 0):
        _fail("vegetation_cursor", "accepted bodies/ledger/envelope cursor differ")
    resources = bridge.shared_resources_v2(originals.capsule, plan, record,
        metadata_sha256=record["metadata_sha256"], soil_identity=db._array_digest(soil),
        accepted_run_copy_bytes=manifest["shared_resources"]["accepted_run_copy_bytes"],
        max_bytes=limits.native_max_bytes)
    if oi._json(resources) != oi._json(manifest["shared_resources"]):
        _fail("resource_identity", "loaded native vegetation/soil differ from resources")
    version, version_sha, native_sha = 0, initial.version_sha256, initial.native_content_sha256
    counter, all_deltas = 48 * plan.cells, []
    for index, body in enumerate(receipts):
        value = body["vegetation"]
        rows = [r for r in delta_rows if r["event_index"] == index]
        if (value["input_version"] != version
                or value["input_version_sha256"] != version_sha
                or value["input_native_content_sha256"] != native_sha
                or value["binding_sha256"] != bindings[index]["binding_sha256"]
                or bindings[index]["event_index"] != index
                or bindings[index]["event_kind"] != body["kind"]
                or oi._json(value["delta_receipts"]) != oi._json(rows)):
            _fail("vegetation_receipt_chain", "accepted input/binding/delta chain differs")
        for row in rows:
            if row["base_version"] != version or row["base_version_sha256"] != version_sha:
                _fail("vegetation_receipt_chain", "delta base version differs")
            version, version_sha, native_sha = row["version"], row["version_sha256"], row["native_content_sha256"]
        handoff = 8 * plan.cells if body["kind"] == "rain" else 0
        delta_bytes = sum(row["copied_bytes"] for row in rows)
        counter += handoff + delta_bytes
        if (value["output_version"] != version or value["output_version_sha256"] != version_sha
                or value["output_native_content_sha256"] != native_sha
                or value["required_snapshot_bytes"] != handoff
                or value["delta_copy_bytes"] != delta_bytes
                or value["accepted_run_copy_bytes"] != counter
                or counter > plan.limits.max_run_copy_bytes):
            _fail("vegetation_copy_chain", "accepted endpoint/copy counter differs")
        all_deltas.extend(rows)
    if (oi._json(all_deltas) != oi._json(list(delta_rows))
            or record["cursor"]["copied_bytes"] != sum(row["copied_bytes"] for row in all_deltas)
            or env.version != version or env.version_sha256 != version_sha
            or env.native_content_sha256 != native_sha
            or resources["accepted_run_copy_bytes"] != counter
            or 8 * plan.cells > originals.runner_limits.max_work_bytes):
        _fail("vegetation_copy_chain", "envelope/full receipt/counter endpoint differs")


def _construct_passive_runner_v2(cls, ticket):
    """Install only a prepared dynamic accepted boundary; no constructor replay."""
    if (cls is not dr.DenseRunner or type(ticket) is not _PassiveLoadV2
            or ticket._authority is not _PASSIVE_V2_TOKEN):
        _fail("restore_ticket", "immediate checked dynamic-loader ticket required")
    fields = ticket.fields
    _keys(fields, RUNNER_FIELDS_V2, "private dynamic runner fields")
    decoded = ticket.evidence["decoded"]
    if (type(decoded) is not oi._DecodedOriginalInputsV2
            or decoded._authority is not oi._DECODED_V2_TOKEN
            or fields["_vegetation_plan"] is not decoded.plan):
        _fail("restore_ticket", "same owning original decode and plan required")
    originals, bed, plan = decoded.originals, fields["_bed"], decoded.plan
    if (fields["_production"] is not None or fields["_stop_receipt"] is not None
            or fields["_active_evidence"] is not None or bed._dynamic.plan is not plan
            or bed.context is not originals.capsule.context):
        _fail("not_quiescent", "closed default-flat accepted resources required")
    n = dr._native()
    bed.verify_case_inputs()
    if "record_origin_s" not in inspect.signature(n.finalize_wind_event).parameters:
        _fail("unsupported_native_finalizer", "MAPLE explicit record_origin_s required")
    dr.DenseRunner._check_wind(n, bed, fields["_wind"])
    if 8 * plan.cells > fields["_limits"].max_work_bytes:
        _fail("vegetation_copy_cap", "current-cover validation exceeds actual work cap")
    # Three float64 C-cell temporaries can coexist: this owned current cover,
    # the actual _check_water cover copy and its returned original-soil copy.
    # Original water and current soil remain distinct accepted resources.
    probe_water = dataclasses.replace(fields["_water"],
        vegetation_cover_fraction=bed.vegetation_cover_fraction())
    initial_soil = dr.DenseRunner._check_water(n, bed, probe_water,
        fields["_control"], fields["_limits"])
    _same(initial_soil, originals.water.initial_soil_water_m, ticket.evidence["limits"])
    del initial_soil, probe_water
    if oi.validate_original_inputs_v2(originals, limits=ticket.evidence["oi_limits"],
            admitted_plan=plan) != ticket.evidence["original_content"]:
        _fail("original_identity", "original inputs changed during preparation")
    probe = SimpleNamespace(**fields)
    parts = dr.DenseRunner._static_parts_record(probe, n)
    if db._sha256(parts) != fields["_static_parts_sha256"]:
        _fail("static_identity", "actual loaded dynamic controlling parts differ")
    _same(parts, fields["_static_parts"], ticket.evidence["limits"])
    if db._sha256(fields["_static_record"]) != fields["_static_sha256"]:
        _fail("static_identity", "original dynamic static record differs")
    owner = object.__new__(cls)
    for name in RUNNER_FIELDS_V2:
        setattr(owner, name, fields[name])
    if set(vars(owner)) != set(RUNNER_FIELDS_V2):
        _fail("runner_layout", "closed28-field dynamic installation required")
    owner._precheck()
    return owner


def _prepare_material_v2(manifest, arrays, artifact_root, expected_source_identity,
        expected_environment_sha256, limits, *, wrapper_raw=0, artifacts=None):
    _preflight_metadata_v2(manifest, expected_source_identity, expected_environment_sha256, limits)
    if type(artifacts) is not _ArtifactsV2:
        _fail("restore_ticket", "explicit admitted v2 child resources required")
    BoundaryImageLimits.__post_init__(limits)
    refs = (manifest["originals"], manifest["genesis"], manifest["physical"])
    if (artifacts.limits is not limits or Path(artifact_root) != artifacts.root
            or oi._json(list(artifacts.refs)) != oi._json(list(refs))
            or type(wrapper_raw) is not int or not 0 <= wrapper_raw <= artifacts.wrapper_raw):
        _fail("restore_ticket", "manifest/root/limits/raw admission differs")
    hashes = {ref["relative_path"]: artifacts.hashed(ref) for ref in refs}
    oi_limits = oi.OriginalPackLimits(max_cells=limits.max_cells, max_classes=limits.max_classes,
        max_voxels=limits.max_voxels, max_cohorts=limits.max_cohorts, max_events=limits.max_events)
    decoded = oi.load_original_inputs_v2(artifacts.root / refs[0]["relative_path"],
        expected_manifest_sha256=_original_manifest_seal(artifacts, refs[0], limits),
        expected_source_identity=expected_source_identity,
        limits=oi_limits)
    if (type(decoded) is not oi._DecodedOriginalInputsV2
            or decoded._authority is not oi._DECODED_V2_TOKEN):
        _fail("restore_ticket", "actual owning original-input decoder required")
    pack, plan = decoded.originals, decoded.plan
    if pack.content_sha256 != refs[0]["identity"]:
        _fail("original_identity", "loaded original content identity differs")
    for ref in refs[1:]:
        if ref["identity"]["caller"]["static_sha256"] != pack.static_sha256:
            _fail("static_identity", "native child belongs to other controlling inputs")
    for ref in {ref["relative_path"]: ref for ref in refs[1:]}.values():
        _native_dimensions(artifacts, ref, pack, limits)
    scope = bridge._new_dynamic_decode_scope(plan, pack.capsule, max_bytes=limits.native_max_bytes)
    genesis_ledger = ClockLedger(decode_time(pack.capsule.case_record["initial_time_s"])).as_record()
    genesis_request = bridge._make_dynamic_restore_request(scope, role="genesis",
        ledger_record=genesis_ledger,
        expected_metadata_sha256=refs[1]["identity"]["vegetation_metadata_sha256"],
        max_bytes=limits.native_max_bytes)
    genesis = bridge.load_boundary_physical_v2(artifacts.root / refs[1]["relative_path"],
        pack.wind.config, refs[1]["identity"], inputs=pack.capsule, request=genesis_request,
        genesis_soil_identity=db._array_digest(pack.water.initial_soil_water_m),
        max_bytes=limits.native_max_bytes)
    if oi._json(refs[1]) == oi._json(refs[2]):
        current = genesis
    else:
        request = bridge._make_dynamic_restore_request(scope, role="current",
            ledger_record=manifest["ledger"], envelope_record=manifest["shared_resources"]["vegetation"],
            expected_metadata_sha256=manifest["shared_resources"]["vegetation_metadata_sha256"],
            initial_ticket=genesis.initial_ticket, max_bytes=limits.native_max_bytes)
        current = bridge.load_boundary_physical_v2(artifacts.root / refs[2]["relative_path"],
            pack.wind.config, refs[2]["identity"], inputs=pack.capsule, request=request,
            shared_resources=manifest["shared_resources"], max_bytes=limits.native_max_bytes)
    n = dr._native()
    if (type(current.mobile_state) is not n.MobileCohortState
            or current.mobile_state.mass_kg.size > limits.max_cohorts):
        _fail("pool_layout", "actual bounded flat native mobile state required")
    if not n.mobile_state_is_genuinely_empty(current.mobile_state):
        _fail("mobile_state_not_empty", "current native pool is not genuinely empty")
    soil, totals, ledger, receipts, state = _validate_history_v2(
        manifest, pack, genesis, current, arrays, limits)
    if (manifest["shared_resources"]["prescribed_identity_sha256"]
            != db._sha256(pack.capsule.prescribed_digests)
            or current.bed.context is not pack.capsule.context):
        _fail("resource_identity", "original prescribed/context binding differs")
    routing = _load_routing(manifest["routing"], pack, arrays, limits)
    if current.bed.identity_sha256() != state["accepted_bed_sha256"]:
        _fail("physical_identity", "current accepted bed identity differs")
    if dr.DenseRunner._private_identity(n.np, current.mobile_state, soil) != state["accepted_private_sha256"]:
        _fail("private_identity", "current loaded native pool/soil differs")
    inventory0 = n.np.array(pack.initial_inventory_kg, copy=True)
    fields = {"_accepted_bed": state["accepted_bed_sha256"],
        "_accepted_private": state["accepted_private_sha256"], "_active_evidence": None,
        "_bed": current.bed, "_control": pack.control, "_inventory0": inventory0,
        "_inventory_elements": state["inventory_elements"], "_ledger": ledger,
        "_limits": pack.runner_limits, "_n_accepted": state["n_accepted"],
        "_n_classes": state["n_classes"], "_next": state["next_index"],
        "_pool": current.mobile_state, "_production": None, "_receipts": receipts,
        "_routing": routing, "_schedule": pack.schedule, "_soil": soil,
        "_static_parts": pack.static_parts, "_static_parts_sha256": pack.static_parts_sha256,
        "_static_record": pack.static_record, "_static_sha256": pack.static_sha256,
        "_stop_receipt": None, "_totals": totals, "_water": pack.water, "_wind": pack.wind,
        "_vegetation_plan": plan,
        "_vegetation_copied_bytes": manifest["shared_resources"]["accepted_run_copy_bytes"]}
    ticket = _PassiveLoadV2(fields,
        {"decoded": decoded, "limits": limits, "oi_limits": oi_limits,
         "original_content": pack.content_sha256}, _PASSIVE_V2_TOKEN)
    owner = dr.DenseRunner._from_passive_boundary_load_v2(ticket)
    artifacts.check()
    for ref in refs:
        if artifacts.hashed(ref) != hashes[ref["relative_path"]]:
            _fail("file_changed", "referenced bytes changed during preparation")
    diagnostics = {"schema": SCHEMA_V2, "profile": PROFILE_V2,
        "source_identity": copy.deepcopy(expected_source_identity),
        "environment_sha256": expected_environment_sha256,
        "accepted": owner.accepted.as_record(), "static_sha256": owner.static_sha256,
        "clock_ledger": owner.ledger_record(), "receipt_chain_sha256": state["receipt_chain_sha256"],
        "vegetation_metadata_sha256": manifest["shared_resources"]["vegetation_metadata_sha256"],
        "accepted_run_copy_bytes": owner._vegetation_copied_bytes,
        "n_wrapper_arrays": len(arrays), "reservation_bytes": artifacts.reservation_bytes,
        "scope": "inert passive dynamic source prototype; no C/store/adoption/event authorization"}
    return _PreparedBoundaryV2(owner, diagnostics)


def _resources_from_runner_v2(runner, limits):
    capsule = bridge.DenseBoundaryInputsV2.capture(runner._bed)
    record = runner._bed._dynamic.as_record()
    return bridge.shared_resources_v2(capsule, runner._vegetation_plan, record,
        metadata_sha256=record["metadata_sha256"], soil_identity=db._array_digest(runner._soil),
        accepted_run_copy_bytes=runner._vegetation_copied_bytes, max_bytes=limits.native_max_bytes)


def prepare_passive_boundary_v2(
    path,
    *,
    artifact_root,
    expected_manifest_sha256,
    expected_source_identity,
    expected_environment_sha256,
    limits=_DEFAULT,
):
    """Prepare and validate a private inert owner; caller receives no live runner."""
    BoundaryImageLimits.__post_init__(limits)
    artifact_root = Path(artifact_root).absolute()
    # Read bounded metadata first, admit ALL child raw sizes before ANY array expansion.
    manifest, _none, raw = _read_wrapper_v2(
        path, expected_manifest_sha256, limits, metadata_only=True
    )
    _preflight_metadata_v2(
        manifest, expected_source_identity, expected_environment_sha256, limits
    )
    artifacts = _ArtifactsV2(
        artifact_root,
        manifest["originals"],
        manifest["genesis"],
        manifest["physical"],
        limits,
        raw,
    )
    try:
        actual, arrays, actual_raw = _read_wrapper_v2(
            path, expected_manifest_sha256, limits, _admitted_raw=raw
        )
        if actual_raw != raw or oi._json(actual) != oi._json(manifest):
            _fail(
                "file_changed",
                "full wrapper read differs from metadata/raw admission basis",
            )
        return _prepare_material_v2(
            actual,
            arrays,
            artifact_root,
            expected_source_identity,
            expected_environment_sha256,
            limits,
            wrapper_raw=raw,
            artifacts=artifacts,
        )
    finally:
        artifacts.close()


def _runner_admission_v2(runner, limits):
    if type(runner) is not dr.DenseRunner or set(vars(runner)) != set(RUNNER_FIELDS_V2):
        _fail("runner_layout", "only closed26-field default DenseRunner")
    if (
        runner._production is not None
        or runner._stop_receipt is not None
        or runner._active_evidence is not None
    ):
        _fail(
            "not_quiescent",
            "only unstopped accepted boundaries without active evidence",
        )
    if type(runner._ledger) is not ClockLedger or set(vars(runner._ledger)) != {
        "epoch",
        "_entries",
    }:
        _fail("clock_layout", "exact ledger layout")
    g, nc = runner._bed.context.geometry, len(runner._bed.context.grain_classes.classes)
    if (
        type(g.ny) is not int
        or type(g.nx) is not int
        or min(g.ny, g.nx) < 1
        or g.ny * g.nx > limits.max_cells
        or not 1 <= nc <= limits.max_classes
        or runner._bed.case_record["n_voxels"] > limits.max_voxels
    ):
        _fail("profile_cap", "bounded Dense dimensions required before traversal")
    for value in (runner._next, runner._n_accepted):
        _int(value, limits.max_events, "accepted counter")
    if (
        runner._next != runner._n_accepted
        or type(runner._receipts) is not list
        or len(runner._receipts) != runner._next
    ):
        _fail("cursor", "closed accepted counters/receipts")
    from maple.aeolian.scheduler import physical_checkpoint as cp

    bridge._check_bed_v2(runner._bed)
    # Reuse34's finite registry for native controlling inputs (including actual NumPy namespace).
    original_limits = oi.OriginalPackLimits(
        max_cells=limits.max_cells,
        max_classes=limits.max_classes,
        max_voxels=limits.max_voxels,
        max_cohorts=limits.max_cohorts,
        max_events=limits.max_events,
    )
    encoder = oi._Encoder(original_limits, oi._registry_v2(), hash_arrays=False)
    encoder.node(
        (
            runner._bed.context,
            runner._wind,
            runner._water,
            runner._control,
            runner._limits,
            runner._inventory0,
            runner._soil,
            runner._static_parts,
            runner._static_record,
            runner._receipts,
        )
    )
    from masy.dense_schedule import DenseSchedule, RainEvent, RainInterval, WindEvent

    _layout(runner._schedule, DenseSchedule, ("events",))
    if (
        type(runner._schedule.events) is not tuple
        or not 1 <= len(runner._schedule.events) <= limits.max_events
    ):
        _fail("profile_cap", "bounded schedule before canonical traversal")
    for event in runner._schedule.events:
        if type(event) is WindEvent:
            _layout(
                event,
                WindEvent,
                ("duration_s", "shear_velocity_m_s", "direction_deg_cw_from_north"),
            )
            encoder.node(vars(event))
        elif type(event) is RainEvent:
            _layout(event, RainEvent, ("intervals", "recession_guard_s"))
            if (
                type(event.intervals) is not tuple
                or not 1 <= len(event.intervals) <= 64
            ):
                _fail("profile_cap", "bounded rainfall intervals before traversal")
            encoder.node(event.recession_guard_s)
            for interval in event.intervals:
                _layout(interval, RainInterval, ("duration_s", "intensity_mm_h"))
                encoder.node(vars(interval))
        else:
            _fail("profile_cap", "unsupported event type")
    bridge._budget(
        cp,
        limits.native_max_bytes,
        runner._bed.context,
        runner._pool,
        runner._bed._held_inputs,
        *(getattr(runner._bed.state, name) for name in db.SHARED_FIELDS),
    )
    if runner._pool.mass_kg.size > limits.max_cohorts:
        _fail("profile_cap", "bounded cohort capacity")
    _array(runner._soil, (g.ny, g.nx), "accepted soil")
    _keys(runner._totals, TOTALS_FIELDS, "accepted totals")
    for name in ("export", "input", "bound", "internal_terminal_deposition"):
        _array(runner._totals[name], (nc,), name)
    if runner._routing is not None:
        _routing_record(runner._routing, runner._water, (g.ny, g.nx), nc, limits)
    for receipt in runner._receipts:
        _receipt_shape_v2(receipt, nc, limits)
    plan = runner._vegetation_plan
    from masy import landscape_vegetation as lv
    if type(plan) is not lv.VegetationPlan or runner._bed._dynamic.plan is not plan:
        _fail("vegetation_identity", "one exact admitted native plan required")
    plan.verify()
    if (plan.schedule_sha256 != runner._schedule.digest
            or plan.event_kinds != tuple(event.kind for event in runner._schedule.events)
            or runner._bed._dynamic.cursor["next_event_index"] != runner._next
            or type(runner._vegetation_copied_bytes) is not int
            or not 48 * plan.cells <= runner._vegetation_copied_bytes <= plan.limits.max_run_copy_bytes):
        _fail("vegetation_cursor", "actual accepted dynamic counters differ")
    runner._precheck()
    if not dr._native().mobile_state_is_genuinely_empty(runner._pool):
        _fail("mobile_state_not_empty", "accepted native pool not genuinely empty")


def _capture_wrapper_arrays_v2(runner, limits):
    """Only fixed ndarray metadata; wrapper byte/count admission precedes all hashes."""
    import numpy as np

    if type(runner) is not dr.DenseRunner or set(vars(runner)) != set(RUNNER_FIELDS_V2):
        _fail("runner_layout", "closed26-field default DenseRunner required")
    _keys(runner._totals, TOTALS_FIELDS, "accepted totals")
    arrays = [
        runner._soil,
        *(
            runner._totals[name]
            for name in ("export", "input", "bound", "internal_terminal_deposition")
        ),
    ]
    held = runner._routing
    if held is not None:
        if type(held) is _RestoredRouting:
            if any(not hasattr(held, name) for name in _RestoredRouting.__slots__):
                _fail(
                    "restored_routing_layout_changed", "private projection slot absent"
                )
            velocity = held.velocity
        elif type(held) is SimpleNamespace and set(vars(held)) == {
            "graph",
            "terrain",
            "network",
            "grid",
            "source",
            "template",
        }:
            _layout(
                held.template,
                _native_types()[-1],
                ROUTING_CONTRACT["SedimentEventState"]["fields"],
            )
            velocity = held.template.sediment_velocity_m_s
        else:
            _fail("restored_routing_layout_changed", "closed held routing required")
        arrays.extend((held.source, velocity))
    if len(arrays) > limits.max_files or any(
        type(array) is not np.ndarray for array in arrays
    ):
        _fail("array_cap", "fixed host wrapper file inventory before traversal")
    if sum(array.nbytes for array in arrays) > limits.max_numeric_bytes:
        _fail("array_cap", "wrapper numeric budget before content scan/hash")
    return arrays


def _snapshot_v2(
    runner, originals, genesis, physical, source_identity, environment_sha256, limits
):
    for ref, fmt in zip(
        (originals, genesis, physical),
        (_ORIGINAL_FORMAT_V2, _NATIVE_FORMAT, _NATIVE_FORMAT),
        strict=True,
    ):
        _reference_v2(ref, fmt)
    arrays = _capture_wrapper_arrays_v2(runner, limits)
    _runner_admission_v2(runner, limits)
    routing = None
    if runner._routing is not None:
        g = runner._bed.context.geometry
        routing = _routing_record(
            runner._routing, runner._water, (g.ny, g.nx), runner._n_classes, limits
        )
        # Compare native factory output once on capture; do not refresh accepted caches.
        n = dr._native()
        expected_graph = n.routing_graph_from_elevation(
            runner._water.terrain, runner._routing.source, xp=n.np
        )
        expected_network, expected_grid = (
            n.transport_network(expected_graph),
            n.physics_grid_from_graph(expected_graph),
        )
        _cache_admission(
            expected_graph,
            expected_network,
            expected_grid,
            runner._water.terrain,
            (g.ny, g.nx),
            runner._n_classes,
            limits,
        )
        from maple_syrup.sediment_bed import check_graph_evidence_matches

        check_graph_evidence_matches(expected_graph, runner._routing.graph)
        for actual, expected in zip(
            (runner._routing.graph, runner._routing.network, runner._routing.grid),
            (expected_graph, expected_network, expected_grid),
            strict=True,
        ):
            if _cache_fingerprint(actual) != _cache_fingerprint(expected):
                _fail(
                    "routing_identity",
                    "accepted cache differs from native retained-source factory",
                )
    if len(arrays) not in (5, 7) or len(arrays) > limits.max_files:
        _fail("array_cap", "wrapper5/7 file count and configured cap before capture")
    if sum(a.nbytes for a in arrays) > limits.max_numeric_bytes:
        _fail("array_cap", "wrapper final numeric inventory cap")
    rows = _receipt_rows(runner._receipts, limits)
    manifest = {
        "schema": SCHEMA_V2,
        "profile": PROFILE_V2,
        "source_identity": copy.deepcopy(source_identity),
        "environment_sha256": environment_sha256,
        "originals": copy.deepcopy(originals),
        "genesis": copy.deepcopy(genesis),
        "physical": copy.deepcopy(physical),
        "state": {
            "next_index": runner._next,
            "n_accepted": runner._n_accepted,
            "n_classes": runner._n_classes,
            "inventory_elements": runner._inventory_elements,
            "native_time": encode_time(runner._bed.time_s),
            "accepted_bed_sha256": runner._accepted_bed,
            "accepted_private_sha256": runner._accepted_private,
            "receipt_chain_sha256": _chain(originals, genesis, rows),
        },
        "totals": _totals_wire(runner._totals),
        "ledger": runner._ledger.as_record(),
        "receipts": rows,
        "routing": routing,
        "shared_resources": _resources_from_runner_v2(runner, limits),
        "arrays": [],
    }
    _preflight_metadata_v2(manifest, source_identity, environment_sha256, limits)
    return manifest, arrays


def save_passive_boundary_v2(
    path,
    runner,
    *,
    artifact_root,
    originals,
    genesis,
    physical,
    source_identity,
    environment_sha256,
    limits=_DEFAULT,
):
    """Seal a NEW immutable wrapper image. No HEAD, adoption or overwrite.

    All referenced originals/native children must already be sealed and retained.
    A post-link exception reports uncertainty and preserves all files for inspection.
    """
    BoundaryImageLimits.__post_init__(limits)
    artifact_root = Path(artifact_root).absolute()
    oi._source_identity(source_identity)
    _digest(environment_sha256, "environment")
    manifest, arrays = _snapshot_v2(
        runner,
        originals,
        genesis,
        physical,
        source_identity,
        environment_sha256,
        limits,
    )
    estimate = (
        sum(a.nbytes + limits.max_header_bytes for a in arrays)
        + limits.max_metadata_bytes
    )
    if estimate > limits.max_serialized_bytes:
        _fail("serialized_cap", "conservative wrapper serialized preflight")
    artifacts = _ArtifactsV2(
        artifact_root,
        manifest["originals"],
        manifest["genesis"],
        manifest["physical"],
        limits,
        estimate,
    )
    try:
        return _write_passive_boundary_v2(
            path,
            runner,
            artifact_root,
            originals,
            genesis,
            physical,
            source_identity,
            environment_sha256,
            limits,
            manifest,
            arrays,
            artifacts,
        )
    finally:
        artifacts.close()


def _write_passive_boundary_v2(
    path,
    runner,
    artifact_root,
    originals,
    genesis,
    physical,
    source_identity,
    environment_sha256,
    limits,
    manifest,
    arrays,
    artifacts,
):
    # _snapshot already admitted the closed5/7 native array layouts and numeric
    # cap. Live native routing is writable; reader preparation requires private
    # readonly carriers. Pre-check the combined reservation before allocation;
    # _Artifacts holds a static bound, with no mutable allocation/charge ledger.
    # leave the owner references/flags and actual serialized capture untouched.
    routing_copy_bytes = sum(array.nbytes for array in arrays[5:])
    writer_reservation_bytes = artifacts.reservation_bytes + routing_copy_bytes
    if writer_reservation_bytes > 2 << 30:
        _fail("memory_cap", "writer routing copies exceed bounded readback reservation")
    validation_arrays = list(arrays)
    staged = None
    for index in range(5, len(arrays)):
        staged = arrays[index].copy(order="C")
        staged.flags.writeable = False
        validation_arrays[index] = staged
    # Actual child readback validates original/genesis/current anchors BEFORE creating this image.
    checked = _prepare_material_v2(
        manifest,
        validation_arrays,
        artifact_root,
        source_identity,
        environment_sha256,
        limits,
        artifacts=artifacts,
    )
    checked_owner = checked._PreparedBoundaryV2__owner
    _same(checked_owner._pool, runner._pool, limits)
    _same(checked_owner._inventory0, runner._inventory0, limits)
    _same(checked_owner._soil, runner._soil, limits)
    if checked_owner._bed.identity_sha256() != runner._bed.identity_sha256():
        _fail(
            "physical_identity",
            "native reference does not represent the current accepted owner",
        )
    del checked_owner, checked, validation_arrays, staged
    target = Path(path)
    if target.name in ("", ".", ".."):
        _fail("path_refused", "new boundary image name required")
    estimate = (
        sum(a.nbytes + limits.max_header_bytes for a in arrays)
        + limits.max_metadata_bytes
    )
    if estimate > limits.max_serialized_bytes:
        _fail("serialized_cap", "conservative wrapper serialized preflight")
    parent, directory = oi._open_dir(target.parent), None
    linked = False
    created = False
    image_identity = None
    try:
        space = os.fstatvfs(parent)
        if (
            space.f_bavail * space.f_frsize < estimate + (1 << 20)
            or space.f_favail < len(arrays) + 3
        ):
            _fail("storage_cap", "insufficient disk/inodes with bounded headroom")
        os.mkdir(target.name, mode=0o700, dir_fd=parent)
        created = True
        before = os.stat(target.name, dir_fd=parent, follow_symlinks=False)
        directory = os.open(
            target.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent,
        )
        if (os.fstat(directory).st_dev, os.fstat(directory).st_ino) != (
            before.st_dev,
            before.st_ino,
        ) or oi._names(directory, limits):
            _fail("file_changed", "exclusive image directory replaced/nonempty")
        image_identity = (before.st_dev, before.st_ino)
        rows, used = [], 0
        import numpy as np

        for index, array in enumerate(arrays):
            name = f"a_{index:04d}.npy"
            fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=directory,
            )
            with os.fdopen(fd, "wb") as stream:
                np.lib.format.write_array(stream, array, allow_pickle=False)
                os.fchmod(stream.fileno(), 0o400)
                stream.flush()
                os.fsync(stream.fileno())
            fd, s = oi._open_file(directory, name)
            try:
                used += s.st_size
                if (
                    used > limits.max_serialized_bytes
                    or s.st_size > limits.max_numeric_bytes + limits.max_header_bytes
                ):
                    _fail(
                        "serialized_cap",
                        "actual written raw per-file/aggregate cap before hash",
                    )
                rows.append(
                    {
                        "name": name,
                        "dtype": array.dtype.str,
                        "shape": list(array.shape),
                        "data_bytes": array.nbytes,
                        "data_sha256": hashlib.sha256(
                            memoryview(array).cast("B")
                        ).hexdigest(),
                        "file_bytes": s.st_size,
                        "sha256": oi._hash_fd(fd, s.st_size),
                    }
                )
            finally:
                os.close(fd)
        manifest["arrays"] = rows
        for row in rows:
            oi._verify_written(directory, row, limits)
        again, _live_arrays = _snapshot_v2(
            runner,
            originals,
            genesis,
            physical,
            source_identity,
            environment_sha256,
            limits,
        )
        again["arrays"] = rows
        if oi._json(again) != oi._json(manifest):
            _fail("accepted_state_changed", "accepted wrapper changed during writing")
        # Prepare actual written readback privately before sealing any acceptance-visible metadata.
        unsigned = {
            key: manifest[key] for key in ROOT_FIELDS if key != "manifest_sha256"
        }
        manifest["manifest_sha256"] = _sha(unsigned)
        raw = oi._json(manifest)
        if (
            len(raw) > limits.max_metadata_bytes
            or used + len(raw) > limits.max_serialized_bytes
        ):
            _fail("serialized_cap", "actual wrapper metadata/aggregate cap")
        fd = os.open(
            "manifest.tmp",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            os.fchmod(stream.fileno(), 0o400)
            stream.flush()
            os.fsync(stream.fileno())
        fd, s = oi._open_file(directory, "manifest.tmp")
        try:
            if os.read(fd, len(raw) + 1) != raw or oi._stat_id(
                os.fstat(fd)
            ) != oi._stat_id(s):
                _fail("manifest_identity", "written manifest readback differs")
        finally:
            os.close(fd)
        # The SAME complete reader admits the actual unlinked manifest/NPY bytes.
        written_manifest, written_arrays, written_raw = _read_wrapper_v2(
            target, manifest["manifest_sha256"], limits, _manifest_name="manifest.tmp"
        )
        _prepare_material_v2(
            written_manifest,
            written_arrays,
            artifact_root,
            source_identity,
            environment_sha256,
            limits,
            wrapper_raw=written_raw,
            artifacts=artifacts,
        )
        final_record, final_arrays = _snapshot_v2(
            runner,
            originals,
            genesis,
            physical,
            source_identity,
            environment_sha256,
            limits,
        )
        final_record["arrays"] = rows
        if oi._json(final_record) != oi._json(unsigned):
            _fail(
                "accepted_state_changed",
                "accepted wrapper metadata changed before seal",
            )
        for written, live in zip(written_arrays, final_arrays, strict=True):
            _same(written, live, limits)
        for row in rows:
            oi._verify_written(directory, row, limits)
        _writer_path_identity(target.parent, parent)
        _writer_path_identity(target, directory)
        os.fsync(directory)
        artifacts.check()
        linked = True  # Publication can occur from this syscall onward, including interruption.
        os.link(
            "manifest.tmp",
            "manifest.json",
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
        os.unlink("manifest.tmp", dir_fd=directory)
        os.fsync(directory)
        os.fsync(parent)
        _writer_path_identity(target.parent, parent)
        _writer_path_identity(target, directory)
        artifacts.check()
        return manifest["manifest_sha256"]
    except BaseException as exc:
        if linked:
            identity = image_identity
            message = f"boundary seal may be published at {target}, directory {identity}; durability uncertain; inspect retained files"
            if not isinstance(exc, Exception):
                exc.add_note(message)
                raise
            raise BoundaryPublicationUncertain(message) from exc
        if created:
            exc.add_note(
                f"unsealed boundary image retained at {target}; retry requires a new exclusive name"
            )
        raise
    finally:
        if directory is not None:
            os.close(directory)
        os.close(parent)


# Source-only authoring excerpt. Appended to B by author_b_01.py; never imported here.

_STATIC_PLAN_KEY = object()


class _StaticMetadataPlan:
    """Issued scalar forecast, never a snapshot, physical owner or B manifest."""
    __slots__ = ('_key', 'events', 'accepted', 'max_receipt_bytes', 'wire_upper',
                 'original_manifest_bytes', 'original_manifest_sha256')

    def __init__(self, key, events, accepted, receipt, wire, raw, digest):
        if key is not _STATIC_PLAN_KEY:
            _fail('metadata_plan', 'owning static planner required')
        for name, value in zip(self.__slots__,
                (key, events, accepted, receipt, wire, raw, digest), strict=True):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError('static metadata plans are immutable')

    def __delattr__(self, name):
        raise AttributeError('static metadata plans are immutable')

    def __reduce_ex__(self, protocol):
        raise TypeError('static metadata plans are operation-local')


def _wire_string(text):
    # Only bounded ASCII Source literals reach this helper, never user strings.
    if (type(text) is not str or not text.isascii() or len(text) > 128
            or any(ord(character) < 32 or ord(character) == 127 for character in text)):
        _fail('metadata_plan', 'bounded Source literal required')
    return 2 + len(text) + text.count('"') + text.count('\\')


def _wire_object(fields):
    return 2 + sum(_wire_string(k) + 1 + n for k, n in fields.items()) + max(0, len(fields) - 1)


def _wire_array(widths):
    return 2 + sum(widths) + max(0, len(widths) - 1)


def _static_metadata_wire_upper(accepted):
    """591's Source-width derivation, not a measured-size/default allowance.

    The static v1 serializer/closed keysets and scalar bounds are inverse-bound
    by the Source auditor. Root must also qualify their selected implementations.
    No metadata values, arrays, routing factory, clock or Native decoder is run.
    """
    _int(accepted, 8, 'prospective accepted count')
    sha, encoded_time, fraction = 66, 1006, 962
    source = _wire_object({'kind': _wire_string('source_manifest_prototype'),
        'masy': sha, 'maple': sha, 'syrup': sha})
    caller = _wire_object({'schema': _wire_string('masy.dense_runner_boundary_native.v1'),
        'source_identity': source, 'environment_sha256': sha, 'static_sha256': sha})
    identity = _wire_object({'schema': _wire_string(bridge.SCHEMA),
        'original_inputs_sha256': sha, 'case_sha256': sha, 'bed_identity_sha256': sha,
        'caller': caller})
    path = 2 + 6 * 4096
    original = _wire_object({'format': _wire_string(_ORIGINAL_FORMAT),
        'relative_path': path, 'manifest_file_sha256': sha, 'identity': sha})
    native = _wire_object({'format': _wire_string(_NATIVE_FORMAT),
        'relative_path': path, 'manifest_file_sha256': sha, 'identity': identity})
    entry = _wire_object({'index': 2, 'kind': _wire_string('event'),
        'native_start': encoded_time, 'native_end': encoded_time,
        'global_start': fraction, 'global_end': fraction, 'endpoint_elapsed': fraction,
        'commit_before': encoded_time, 'commit_after': encoded_time,
        'returned_elapsed': encoded_time, 'endpoint_discrepancy': fraction})
    ledger = _wire_object({'schema': _wire_string('masy.clock_ledger.v1'),
        'native_epoch': encoded_time,
        'global_elapsed_definition': _wire_string('Fraction(native_t) - Fraction(native_epoch)'),
        'entries': _wire_array([entry] * accepted)})
    totals = _wire_object({'export': _wire_string('total_export'),
        'input': _wire_string('total_input'), 'bound': _wire_string('total_bound'),
        'internal_terminal_deposition': _wire_string('total_internal_terminal_deposition'),
        'water': _wire_object({k: encoded_time for k in WATER_FIELDS}),
        'n_entry_reroutes': 10, 'n_event_reroutes': 10})
    routing_fields = {k: sha for k in ROUTING_FIELDS}
    routing_fields.update(source_array=_wire_string('routing_source'),
        velocity_array=_wire_string('routing_velocity'), phase=4)
    shared = _wire_object({'schema': _wire_string('masy.shared_surface_resources.v0'),
        'version': 1, 'profile': _wire_string('static_prescribed_native_dry_reset_v0'),
        'prescribed_identity_sha256': sha, 'soil_array': _wire_string('current_soil'),
        'update_cursor': 1, 'update_receipts': 2})
    descriptor = _wire_object({'name': _wire_string('a_0006.npy'),
        'dtype': _wire_string('<f8'), 'shape': _wire_array([2, 2, 2]),
        'data_bytes': 7, 'data_sha256': sha, 'file_bytes': 7, 'sha256': sha})
    state = _wire_object({'next_index': 1, 'n_accepted': 1, 'n_classes': 1,
        'inventory_elements': 4, 'native_time': encoded_time,
        'accepted_bed_sha256': sha, 'accepted_private_sha256': sha,
        'receipt_chain_sha256': sha})
    receipt = _wire_object({'body': 256 << 10, 'bytes': 6, 'sha256': sha})
    return _wire_object({'schema': _wire_string(SCHEMA), 'profile': _wire_string(PROFILE),
        'source_identity': source, 'environment_sha256': sha,
        'originals': original, 'genesis': native, 'physical': native, 'state': state,
        'totals': totals, 'ledger': ledger, 'receipts': _wire_array([receipt] * accepted),
        'routing': _wire_object(routing_fields), 'shared_resources': shared,
        'arrays': _wire_array([descriptor] * 7), 'manifest_sha256': sha})


def _flat_metadata_node(table, index, tag, keys):
    if type(table) is not list or type(index) is not int or not 0 <= index < len(table):
        _fail('metadata_plan', 'bounded exact Original flat node index')
    node = table[index]
    _keys(node, keys, 'prospective Original node')
    if node['t'] != tag:
        _fail('metadata_plan', 'exact Original scalar/container tag')
    return node


def _flat_metadata_fields(table, index, tag, names, *, class_name=None):
    keys = ('t', 'v') if class_name is None else ('t', 'c', 'v')
    node = _flat_metadata_node(table, index, tag, keys)
    if class_name is not None and node['c'] != class_name:
        _fail('metadata_plan', 'genuine Original controlling class tag')
    rows = node['v']
    if type(rows) is not list or len(rows) != len(names):
        _fail('metadata_plan', 'closed exact Original field count')
    fields = {}
    for row in rows:
        if (type(row) is not list or len(row) != 2 or type(row[0]) is not str
                or row[0] not in names or row[0] in fields):
            _fail('metadata_plan', 'missing/duplicate/foreign Original field')
        fields[row[0]] = row[1]
    return fields


def _static_original_scalar_plan(raw, reference, expected_source, kinds, limits, accepted):
    """Metadata only; the unchanged Original/native reader remains mandatory."""
    BoundaryImageLimits.__post_init__(limits)
    if limits.max_receipt_bytes > 256 << 10:
        _fail('metadata_plan', 'owning Source receipt-width ceiling before planning')
    if type(raw) is not bytes or not 1 <= len(raw) <= 4 << 20:
        _fail('metadata_plan', 'raw Original metadata admission before parse')
    _reference(reference, _ORIGINAL_FORMAT)
    if hashlib.sha256(raw).hexdigest() != reference['manifest_file_sha256']:
        _fail('metadata_plan', 'exact Original raw hash required')
    original_limits = oi.OriginalPackLimits(max_cells=limits.max_cells,
        max_classes=limits.max_classes, max_voxels=limits.max_voxels,
        max_cohorts=limits.max_cohorts, max_events=limits.max_events)
    manifest = oi._strict_json(raw, original_limits)
    _keys(manifest, ('schema', 'profile', 'content_sha256', 'wire', 'files',
                     'manifest_sha256'), 'prospective Original manifest')
    unsigned = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if (manifest['schema'] != oi.SCHEMA or manifest['profile'] != oi.PROFILE
            or manifest['content_sha256'] != reference['identity']
            or hashlib.sha256(oi._json(unsigned)).hexdigest() != manifest['manifest_sha256']
            or oi._json(manifest) != raw):
        _fail('metadata_plan', 'exact canonical Original Source/identity/seal')
    wire = manifest['wire']
    _keys(wire, ('anchors', 'body', 'nodes'), 'prospective Original wire')
    table = wire['nodes']
    if type(table) is not list or not 1 <= len(table) <= original_limits.max_nodes:
        _fail('metadata_plan', 'bounded Original metadata node table')
    root = _flat_metadata_fields(table, wire['body'], 'dict',
                                tuple(k for k in oi._ROOT_FIELDS if k != 'content_sha256'))
    source = _flat_metadata_fields(table, root['source_identity'], 'dict',
                                  ('kind', 'masy', 'maple', 'syrup'))
    actual_source = {}
    for name, index in source.items():
        value = _flat_metadata_node(table, index, 'str', ('t', 'v'))['v']
        if type(value) is not str:
            _fail('metadata_plan', 'exact Source identity strings')
        actual_source[name] = value
    oi._source_identity(actual_source)
    if actual_source != expected_source:
        _fail('metadata_plan', 'same actual Source as the accepted store')
    schedule = _flat_metadata_fields(table, root['schedule'], 'dict', ('schema', 'events'))
    events = _flat_metadata_node(table, schedule['events'], 'list', ('t', 'v'))['v']
    if (type(kinds) is not list or type(events) is not list
            or not 1 <= len(events) == len(kinds) <= limits.max_events
            or any(kind not in ('wind', 'rain') for kind in kinds)):
        _fail('metadata_plan', 'same finite original ordered event count')
    for index, event_index in enumerate(events):
        event = _flat_metadata_node(table, event_index, 'dict', ('t', 'v'))['v']
        if type(event) is not list or len(event) > 8:
            _fail('metadata_plan', 'bounded Original event record')
        candidates = [row[1] for row in event if type(row) is list and len(row) == 2
                      and row[0] == 'kind']
        if len(candidates) != 1 or _flat_metadata_node(table, candidates[0], 'str',
                ('t', 'v'))['v'] != kinds[index]:
            _fail('metadata_plan', 'actual Original event kinds differ')
    producer_names = ('max_events', 'max_wind_accounting_steps', 'max_storm_steps',
                      'max_report_rows', 'max_work_bytes', 'max_receipt_bytes')
    producer = _flat_metadata_fields(table, root['runner_limits'], 'class',
        producer_names, class_name='masy.dense_runner:DenseRunnerLimits')
    values = {}
    for name in producer_names:
        value = _flat_metadata_node(table, producer[name], 'int', ('t', 'v'))['v']
        if type(value) is not int or not 1 <= value <= 1 << 40:
            _fail('metadata_plan', 'positive finite producer scalar limit')
        values[name] = value
    if values['max_events'] < len(events) or values['max_receipt_bytes'] > limits.max_receipt_bytes:
        _fail('metadata_plan', 'actual producer receipt cap must precede transport')
    _int(accepted, len(events), 'planned accepted count')
    if limits.max_receipts_bytes < accepted * values['max_receipt_bytes']:
        _fail('metadata_plan', 'aggregate future receipts admitted before Native')
    wire_upper = _static_metadata_wire_upper(accepted)
    if wire_upper > limits.max_metadata_bytes:
        _fail('metadata_plan', 'entire Source-derived future B wire must fit original cap')
    return _StaticMetadataPlan(_STATIC_PLAN_KEY, len(events), accepted,
        values['max_receipt_bytes'], wire_upper, len(raw), reference['manifest_file_sha256'])


def _mixed_static_bridge():
    # sys is an interpreter-owned module; do not import AW/default subjects here.
    import sys
    module = sys.modules.get('masy.dense_active_water_store')
    if module is None or vars(module).get('_MIXED_OPERATION_SLOT') is None:
        return None
    return module


def _mixed_static_writer_gate(path, runner, root, originals, source, environment, limits,
                              *, manifest=None, raw=None):
    module = _mixed_static_bridge()
    if module is not None:
        module._mixed_writer_check(path, runner, root, originals, source,
                                   environment, limits, manifest=manifest, raw=raw)
