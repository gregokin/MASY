"""Private, in-memory native water continuation for the small static Dense CPU profile.

No persistence, physics, checkpoint codec or alternate completion policy lives
here. Imports at module load are standard-library only. Ordinary runner calls
never select this component. A returned token contains no native object.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import math
import os
import struct
import sys
import threading
import weakref
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any

__all__ = ["pause_water", "resume_water", "abandon_water"]

_MAX_CELLS, _MAX_CLASSES, _MAX_VOXELS, _MAX_COHORTS, _MAX_EVENTS = 64, 6, 16, 16000, 8
_MAX_PAUSES, _MAX_ROWS, _MAX_STEPS = 2, 4096, 10000
# After ACTUAL owning binding: MAPLE986 Native bundle; SYRUP603 canonical
# src/pyproject size_bytes+sha256 table. Whole-file logical census335/336;
# not an allocator/heap/JIT/IO fit claim. The whole-worker gate stays required.
_NATIVE_CENSUS_MAPLE = "7fd3caf1b52bcf9b2e4285765db18ba6c70bc7e945b913af4f41d5a2de408abc"
_NATIVE_CENSUS_SYRUP = "390ff7cc0d04b5e7d713156315a8a1607549b8a7f4b75fa5f04aa9b1d1bb2d03"
_ALLOCATION_SITES = MappingProxyType({"hydrology": 1849, "wet_physics": 681, "upwind": 439,
                                      "event_and_terminal": 4605, "maple_dense": 6418})
_MAX_DEPTH, _MAX_NODES, _MAX_ARRAYS, _MAX_KEYS, _MAX_STRING = 32, 65536, 1024, 256, 1024
_MAX_INT_BITS = 256


class _WaterToken:
    __slots__ = ()


@dataclasses.dataclass(slots=True)
class _WaterLease:
    owner: Any
    thread: Any
    pid: int
    index: int
    event: Any
    origin: float
    phase: str = "busy"
    token: Any = None
    prepared: Any = None
    progress: Any = None
    plan: tuple = ()
    ordinal: int = 0
    pauses: int = 0
    resolved: Any = None
    runtime_bytes: bytes = b""
    source: Any = None
    environment: str = ""
    entry_sha256: str = ""
    progress_sha256: str = ""
    stage_started: bool = False
    reservation: Any = None


def _fail(code: str, detail: str) -> None:
    from masy.dense_runner import DenseRunnerError
    raise DenseRunnerError(code, detail)


def _reject_active(runner: Any) -> None:
    """The early step/run guard must always raise, before their catch/_stop."""
    active = runner._active_evidence
    phase = active.phase if type(active) is _WaterLease else "unowned"
    _fail("water_lease_" + phase, "active water ownership refuses fresh/passive/competing work")


def _runner(runner: Any) -> None:
    from masy import dense_runner as dr
    if type(runner) is not dr.DenseRunner:
        _fail("water_pause_profile", "only the exact static default Dense CPU runner")


def _lease(runner: Any, token: Any) -> _WaterLease:
    _runner(runner)
    active = runner._active_evidence
    if (type(active) is not _WaterLease or active.owner() is not runner
            or active.pid != os.getpid() or active.thread() is not threading.current_thread()
            or active.phase != "paused" or type(token) is not _WaterToken or active.token is not token
            or runner._stop_receipt is not None or runner._next != active.index):
        _fail("water_token", "foreign, stale, consumed, busy or wrong-process/thread water token")
    return active


def _declare():
    """Finite source-traced type/import declarations before runtime identity."""
    from masy import dense_boundary_store as store, dense_runner as dr
    from maple_syrup import checkpoint
    n = dr._native()
    schemas = {}
    for (module_name, name), fields in _FIELDS.items():
        cls = getattr(importlib.import_module(module_name), name)
        if not isinstance(cls, type) or tuple(f.name for f in dataclasses.fields(cls)) != fields:
            _fail("water_native_layout", module_name + "." + name)
        schemas[cls] = fields
    for name in ("maple.surface.active_layer.validation", "maple.surface.availability.validation",
                 "maple.coupling.sediment_ledger.validation", "maple.surface.topographic_commit.commit",
                 "maple.water", "maple.water.validation"):
        importlib.import_module(name)
    checkpoint._registry()  # owning declaration, no checkpoint/model construction
    return n, checkpoint, schemas, store


def _profile(runner: Any, n: Any) -> None:
    from masy import dense_runner as dr, dense_runner_checkpoint as boundary
    from maple.core.types.voxel import VoxelColumnState
    from maple_syrup.sediment_event import SedimentEventError
    from maple_syrup.storm import StormError
    if (set(vars(runner)) != set(boundary.RUNNER_FIELDS) or runner._production is not None
            or type(runner._ownership()) is not dr._HostOwnership or runner._stop_receipt is not None):
        _fail("water_pause_profile", "static closed26-field host runner required; dynamic/production/device refused")
    # The ordinary constructor admits only DenseBed, but an unsupported native
    # holder substituted later must refuse BEFORE assuming a .mass_kg field.
    if (type(runner._bed) is not dr.DenseBed or type(runner._bed.state) is not n.BedState
            or type(runner._bed.state.voxel_column) is not VoxelColumnState):
        _fail("water_pause_profile", "original exact Dense native voxel storage required; windowed refused")
    if type(runner._bed.state.voxel_column.mass_kg) is not n.np.ndarray:
        _fail("water_pause_profile", "original NumPy dense mass array required")
    sc = runner._control.sediment_control
    if (type(sc) is not n.SedimentEventControl or type(runner._control.completion_policy) is not n.CompletionPolicy):
        _fail("water_pause_control", "original native controls required")
    if type(runner._limits) is not dr.DenseRunnerLimits:
        _fail("water_pause_profile", "original bounded runner limits required")
    try:
        sc.validated()
        runner._control.completion_policy.validated()
        runner._limits.__post_init__()
    except (TypeError, ValueError, AttributeError, SedimentEventError, StormError) as exc:
        _fail("water_pause_control", "invalid native control/runner limits: " + type(exc).__name__)
    if (sc.commit is not True or sc.force_final_commit is not False
            or sc.transport_scheme != "upwind" or sc.phase_storage != "dense" or sc.phase_max_nnz is not None
            or sc.record_exchange_history is not False or sc.deep_publication != "step"
            or sc.storm.implementation != "array" or sc.storm.root_solver != "bisection"):
        _fail("water_pause_control", "original checkpoint-compatible array/upwind/bisection controls required")
    g = runner._bed.context.geometry
    shape = runner._bed.state.voxel_column.mass_kg.shape
    if (type(runner._next) is not int or not 0 <= runner._next < len(runner._schedule.events)):
        _fail("water_pause_profile", "next original rain event required")
    if (len(shape) != 4 or g.ny * g.nx > _MAX_CELLS or runner._n_classes > _MAX_CLASSES
            or shape[2] > _MAX_VOXELS or len(runner._schedule.events) > _MAX_EVENTS
            or len(runner._pool.mass_kg) > _MAX_COHORTS
            or runner._limits.max_report_rows > _MAX_ROWS or sc.storm.max_steps > _MAX_STEPS
            or len(runner._schedule.events[runner._next].intervals) > _MAX_ROWS):
        _fail("water_pause_profile", "bounded Dense dimensions/cohorts/events required")


def _runtime(store: Any, reference: Any, lease: _WaterLease | None = None):
    from masy import dense_original_inputs as oi
    if lease is None:
        store._plain_metadata(reference, 32 << 10, "water runtime reference", max_nodes=10000, max_depth=16)
    raw = oi._json(reference) if lease is None else lease.runtime_bytes
    if len(raw) > 32 << 10:
        _fail("water_runtime_cap", "finite runtime reference required")
    view = type("_ReferenceLimits", (), {"max_metadata_bytes": 32 << 10, "max_nodes": 10000, "max_depth": 16})()
    copied = oi._strict_json(raw, view)
    source, environment = store._runtime_binding(copied, store.BoundaryStoreLimits())
    if source["maple"] != _NATIVE_CENSUS_MAPLE or source["syrup"] != _NATIVE_CENSUS_SYRUP:
        _fail("water_resource_source", "native bodies lack this explicit resource/continuation census")
    if lease is not None and (source != lease.source or environment != lease.environment):
        _fail("water_runtime_changed", "original source/environment changed while wet")
    return raw, source, environment


def _plan(runner: Any, n: Any):
    from masy.clock import decode_time
    from masy.dense_schedule import resolve_event
    from maple_syrup.storm import plan_boundaries
    resolved = resolve_event(runner._schedule, runner._next, runner._bed.time_s)
    edges = [decode_time(row["native"]) for row in resolved["resolved_edges"]["edges"]]
    event = runner._schedule.events[runner._next]
    schedule = n.RainfallSchedule(edges_s=edges[:-1], intensity_mm_per_h=[i.intensity_mm_h for i in event.intervals],
                                  provenance=n.RainfallProvenance(kind="constant"))
    plan = plan_boundaries(schedule, runner._bed.time_s, edges[-1], runner._control.report_every_s,
                           max_report_rows=runner._limits.max_report_rows)
    return resolved, tuple(float(t) for t in plan)


def _ordinal(value: Any, plan: tuple, previous: int = 0) -> None:
    if type(value) is not int or not previous < value < len(plan):
        _fail("water_pause_ordinal", "a later ORIGINAL unfinished boundary ordinal is required")


def _bounds(runner: Any, plan: tuple) -> dict:
    from maple_syrup.storm import HYDROGRAPH_COLUMNS
    from maple_syrup.sediment_event import sediment_hydrograph_columns
    g = runner._bed.context.geometry
    n, k, z = g.ny * g.nx, runner._n_classes, runner._bed.state.voxel_column.mass_kg.shape[2]
    p = runner._bed.state.ledger.process_totals_kg.shape[0]
    rows = len(plan)
    domain = 8 * max(n * z * k, p * n * k, 8 * n, (g.ny + 2) * (g.nx + 2))
    reports = 8 * rows * (len(HYDROGRAPH_COLUMNS) + len(sediment_hydrograph_columns(k)))
    return {"domain": domain, "reports": reports, "rows": rows,
            "array_cap": max(domain, reports, rows * 8),
            "raw_cap": _MAX_ARRAYS * domain + reports + 2 * rows * 8,
            "sequence_cap": max(rows, runner._control.sediment_control.max_commit_log,
                                runner._control.sediment_control.max_rejections_recorded, 128)}


def _scan(root: Any, n: Any, schemas: dict, bounds: dict) -> dict:
    """Bounded private mutation digest, not a persistence encoder or decoder."""
    np = n.np
    digest = hashlib.sha256()
    arrays = {}
    ancestors = set()
    nodes, raw_bytes, hashed_bytes, alias_visits = 0, 0, 0, 0
    def mark(value: bytes):
        digest.update(len(value).to_bytes(8, "little"));digest.update(value)
    def walk(value: Any, path: str, depth: int):
        nonlocal nodes, raw_bytes, hashed_bytes, alias_visits
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH or len(path) > 8192:
            _fail("water_tree_cap", "bounded native evidence tree required")
        if type(value) is np.ndarray:
            if (value.dtype.hasobject or value.dtype.fields or value.dtype.kind not in "fiub"
                    or not value.dtype.isnative or value.dtype.char not in "?bBhHiIlLqQfd"
                    or value.ndim > 4 or not value.flags.c_contiguous or value.nbytes > bounds["array_cap"]):
                _fail("water_array_layout", path)
            identity = id(value)
            first_visit = identity not in arrays
            if first_visit:
                # ALL raw/leaf/read admission is metadata-only, before hashing
                # this leaf or retaining it in the walker reference table.
                if len(arrays) >= _MAX_ARRAYS or value.nbytes > bounds["raw_cap"] - raw_bytes:
                    _fail("water_array_cap", "native arrays exceed preadmitted byte/leaf/read cap")
                raw_bytes += value.nbytes
                arrays[identity] = value
            else:
                alias_visits += 1
            mark(b"array" if first_visit else b"array_alias")
            mark(value.dtype.str.encode());mark(str(value.shape).encode())
            mark(str((identity, value.flags.writeable)).encode())
            if first_visit:
                mark(memoryview(value).cast("B") if value.nbytes else b"")
                hashed_bytes += value.nbytes
            return
        if isinstance(value, np.generic):
            if value.dtype.kind not in "fiub" or (value.dtype.kind == "f" and not math.isfinite(float(value))):
                _fail("water_scalar", path)
            mark(b"numpy_scalar");mark(value.dtype.str.encode());mark(value.tobytes());return
        if value is None or type(value) in (bool, int, float, str):
            if type(value) is float and not math.isfinite(value):
                _fail("water_scalar", path)
            if type(value) is str and len(value) > _MAX_STRING:
                _fail("water_string_cap", path)
            if type(value) is int and value.bit_length() > _MAX_INT_BITS:
                _fail("water_integer_cap", path)
            # Float/bool/None have bounded spelling; bounded integer bits and
            # string length are admitted BEFORE repr/encode allocates output.
            data = (value.hex() if type(value) is float else repr(value)).encode()
            mark(type(value).__name__.encode());mark(data);return
        if isinstance(value, ModuleType):
            if value is not np:
                _fail("water_backend", path)
            mark(b"numpy_namespace");return
        identity = id(value)
        if identity in ancestors:
            _fail("water_cycle", path)
        ancestors.add(identity)
        try:
            if type(value) in schemas:
                fields = schemas[type(value)]
                if hasattr(value, "__dict__") and set(vars(value)) != set(fields):
                    _fail("water_native_layout", path)
                mark((type(value).__module__ + "." + type(value).__qualname__).encode())
                for field in fields:
                    mark(field.encode());walk(getattr(value, field), path + "." + field, depth + 1)
            elif type(value) in (tuple, list):
                if len(value) > bounds["sequence_cap"]:
                    _fail("water_sequence_cap", path)
                mark(type(value).__name__.encode());mark(str(len(value)).encode())
                for index, item in enumerate(value):walk(item, path + f"[{index}]", depth + 1)
            elif type(value) is dict:
                if len(value) > _MAX_KEYS or any(type(key) is not str or len(key) > 128 for key in value):
                    _fail("water_mapping_cap", path)
                mark(b"dict")
                for key in sorted(value):
                    try:
                        encoded = key.encode()
                    except UnicodeEncodeError:
                        _fail("water_mapping_cap", "mapping key must have bounded UTF-8 encoding")
                    mark(encoded);walk(value[key], path + "." + key, depth + 1)
            else:
                _fail("water_native_type", path + ":" + type(value).__qualname__)
        finally:
            ancestors.remove(identity)
    walk(root, "native", 0)
    return {"sha256": digest.hexdigest(), "arrays": tuple(arrays.values()), "nodes": nodes,
            "raw_bytes": raw_bytes, "array_data_hashed_bytes": hashed_bytes, "array_alias_visits": alias_visits}


def _held_values(held: Any) -> tuple:
    """The existing two host reader layouts, never a synthetic routing state."""
    if held is None:return ()
    from masy import dense_runner_checkpoint as boundary
    if type(held) is SimpleNamespace and set(vars(held)) == {
            "graph", "network", "grid", "terrain", "source", "template"}:
        return ("live", held.graph, held.network, held.grid, held.terrain, held.source, held.template)
    names = ("graph", "terrain", "network", "grid", "source", "velocity", "phase", "logical_identity_sha256")
    if (type(held) is boundary._RestoredRouting
            and tuple(type(held).__slots__) == names and all(hasattr(held, name) for name in names)):
        return ("restored", *(getattr(held, name) for name in names))
    _fail("water_routing_layout", "unchanged closed live or restored host routing holder required")


def _metadata_bytes(rows: int) -> int:
    """Fixed structural preflight/scan bound, before native plan allocation."""
    pointer = struct.calcsize("P")
    node = sys.getsizeof({}) + sys.getsizeof([]) + 4 * sys.getsizeof(0) + 4 * pointer
    item = 4 * _MAX_STRING + sys.getsizeof("") + node
    return ((_MAX_DEPTH + 2) * (_MAX_KEYS * item + 4 * 8192 + 2048)
            + _MAX_NODES * (node + sys.getsizeof("")) + 8 * rows * (sys.getsizeof(0.) + pointer))


def _reserve(runner: Any, plan: tuple) -> dict:
    """Source-bound Python-visible allocation reservation, not a heap/fit claim.

    Every expression site in the closed native files is charged even across
    mutually exclusive branches. Iterative dense kernels replace workspace;
    retained arrays, report-prefix tables and append-only logs are separate.
    This deliberately overreserves visible outputs, but does not estimate
    interpreter/import/NumPy-private allocator or compiled-library memory.
    Those require the independent whole-worker AS/RSS/headroom gate.
    """
    from masy import dense_bed
    g = runner._bed.context.geometry
    cells, classes = g.ny * g.nx, runner._n_classes
    bed = runner._ownership().stage_bytes()
    voxel = runner._bed.state.voxel_column.mass_kg.nbytes
    active = runner._bed.state.active_layer.mass_kg.nbytes
    small, ring = 8 * cells * classes, (g.ny + 2) * (g.nx + 2)
    bounds = _bounds(runner, plan)
    rows = bounds["rows"]
    # Include static referenced holdings as well as accepted evolving owners.
    owners = dense_bed._tree_nbytes(
        runner._bed.state, runner._pool, runner._soil, runner._inventory0, runner._totals,
        runner._bed.context, runner._bed.vegetation, runner._bed.threshold_modifier_state,
        runner._wind, runner._water, runner._control, *_held_values(runner._routing))
    # Original stage plus the retained entry commit's complete result/callback
    # objects, not just boundary.state. The shared native eight-generation
    # lifetime upper is charged separately here and inside resumed execution.
    exchange_commit = 2 * voxel + 2 * (voxel + active)
    original_entry = 9 * bed + exchange_commit + 8 * cells * (classes + 6)
    # EACH full71-field result: <=1024 admitted domain leaves plus the actual
    # two report tables/plan vectors. Prior lease remains alive during resume;
    # native dropping its own continuation reference does not drop ours.
    previous_and_new_progress = 2 * bounds["raw_cap"]
    native_beds = 8 * bed
    # Availability-aware native avalanche callback: original/current, the
    # iteration's exchange candidate and returned candidate coexist. The solver
    # replaces iterations; it does not append a bed history. Charge separately
    # even when the actual static case disables this native callback.
    avalanche_beds = 4 * bed
    native_exchange_commit = exchange_commit
    # Largest non-report tensors in these fixed dense bodies: voxel; full
    # P*N*K ledger; two face extents; D8 ring gathers; original-plan vectors.
    domain = max(bounds["domain"], 2 * small, 64 * ring, 8 * rows)
    native_hydrology = _ALLOCATION_SITES["hydrology"] * max(64 * ring, 8 * rows)
    native_physics = _ALLOCATION_SITES["wet_physics"] * max(small, 8 * cells)
    native_upwind = _ALLOCATION_SITES["upwind"] * max(2 * small, 8 * cells)
    native_dense_and_validation = (_ALLOCATION_SITES["maple_dense"]
                                   + _ALLOCATION_SITES["event_and_terminal"]) * domain
    # Prior-prefix/new/terminal/report-snapshot pairs. The ordinary startup
    # allocations before continuation replacement coexist and are charged.
    reports = 4 * bounds["reports"]
    # Native commit/rejection logs contain metadata only (no array append).
    # Fixed source record <=128 nodes / <=128-char static text; rejection4
    # fields have one <=1024-char reason in the admitted scanner profile.
    pointer = struct.calcsize("P")
    node = sys.getsizeof({}) + sys.getsizeof([]) + 4 * sys.getsizeof(0) + 4 * pointer
    commit_record = 128 * (node + sys.getsizeof("") + 4 * 128)
    rejection_record = 16 * node + sys.getsizeof("") + 4 * _MAX_STRING
    sc = runner._control.sediment_control
    logs = 2 * (sc.max_commit_log * commit_record + sc.max_rejections_recorded * rejection_record)
    # Walker streams bytes. This covers per-depth sorted keys/paths, admitted
    # native scalar metadata, plan tuples and the transient schema/reference.
    metadata = _metadata_bytes(rows)
    total = (owners + original_entry + previous_and_new_progress + native_beds + avalanche_beds
             + native_exchange_commit + native_hydrology + native_physics + native_upwind
             + native_dense_and_validation + reports + logs + metadata)
    runner._work_bytes_check(total)
    return {"owned_reference_bytes": owners, "original_entry_bytes": original_entry,
            "previous_and_new_full_result_bytes": previous_and_new_progress,
            "native_bed_generations_bytes": native_beds, "native_exchange_commit_bytes": native_exchange_commit,
            "native_avalanche_generations_bytes": avalanche_beds,
            "native_hydrology_bytes": native_hydrology, "native_wet_physics_bytes": native_physics,
            "native_upwind_bytes": native_upwind, "native_dense_and_validation_bytes": native_dense_and_validation,
            "report_table_bytes": reports, "native_log_metadata_bytes": logs, "metadata_bytes": metadata,
            "required_bytes": total, "allocation_site_counts": dict(_ALLOCATION_SITES),
            "scope": "source-bound Python-visible Dense CPU reservation; opaque whole-worker gate separate",
            "fit_or_peak_measured": False}


def _entry_digest(runner: Any, prepared: Any, n: Any, schemas: dict, bounds: dict, resolved: dict) -> str:
    from masy import dense_runner as dr
    if (type(prepared) is not dr._RainEntry or prepared.bed is not runner._bed
            or prepared.stage._owner is not runner._bed._owner or prepared.stage.consumed
            or prepared.ctx is not runner._bed.context or prepared.w is not runner._water
            or prepared.sc is not runner._control.sediment_control
            or prepared.policy is not runner._control.completion_policy
            or type(prepared.own) is not dr._HostOwnership or prepared.own._bed is not runner._bed
            or type(prepared.ops) is not dr._HostArrayOps or prepared.ops.xp is not n.np
            or prepared.np is not n.np or prepared.n.np is not n.np):
        _fail("water_entry_owner", "genuine original stage/bed/context required")
    if (type(prepared.first) is not bool or prepared.held is None
            or (prepared.first and runner._routing is not None)
            or (not prepared.first and prepared.held is not runner._routing)):
        _fail("water_routing_changed", "original accepted routing holder changed")
    if prepared.cover is not None and prepared.cover is not runner._water.vegetation_cover_fraction:
        _fail("water_cover_changed", "static original cover identity changed")
    b = prepared.boundary
    facts = (resolved, prepared.stage.case_sha256, prepared.stage.nbytes,
             prepared.stage.identity_sha256, prepared.stage.time_s, prepared.t,
             prepared.first, b.did_commit, b.rerouted, b.net_displacement_m,
             prepared.rows, prepared.lower_steps, prepared.rain_edges, prepared.guard_end,
             prepared.intensities, prepared.every, prepared.inventory_before, prepared.soil, prepared.memory_names,
             prepared.stage.state, b.state, prepared.graph, prepared.network, prepared.grid,
             _held_values(prepared.held), prepared.schedule,
             runner._water.vegetation_cover_fraction, prepared.sc, prepared.policy)
    return _scan(facts, n, schemas, bounds)["sha256"]


def _disjoint(runner: Any, item: Any, n: Any) -> None:
    from masy import dense_bed
    evolving = (item.result.state.bed, item.result.state.storm.soil_water_m,
                item.result.state.sediment_velocity_m_s)
    accepted = (runner._bed.state, runner._pool, runner._soil)
    left = dense_bed._require_host_arrays(evolving, "wet")
    right = dense_bed._require_host_arrays(accepted, "accepted")
    for _, candidate in left:
        for _, original in right:
            if n.np.shares_memory(candidate, original):
                _fail("water_accepted_alias", "wet evolving state aliases the accepted boundary")


def _wet(runner: Any, lease: _WaterLease, item: Any, n: Any, checkpoint: Any, schemas: dict):
    if (type(item) is not n.EventProgress or item.status != "wet" or item.quiet.ready is not False
            or float(item.result.state.t_s) not in lease.plan
            or float(item.result.state.t_s) >= lease.plan[-1]
            or tuple(float(t) for t in item.planned_boundaries) != lease.plan
            or item.result.state.terrain is not lease.prepared.held.terrain):
        _fail("water_pause_result", "native returned no unfinished original-boundary wet state")
    bounds = _bounds(runner, lease.plan)
    first = _scan(item, n, schemas, bounds)
    checkpoint.validate_checkpoint_state(item, runner._bed.context, runner._water.column, runner._water.sediment)
    after = _scan(item, n, schemas, bounds)
    if first["sha256"] != after["sha256"]:
        _fail("water_validator_mutated", "native wet validator changed its input")
    _disjoint(runner, item, n)
    return after


def _callback(target: float):
    def stop(progress):
        # No mutable native object is retained, even when this returnsTrue.
        return bool(float(progress.result.state.t_s) == target)
    return stop


def _halt(runner: Any, lease: _WaterLease, stop: Any):
    lease.phase, lease.token = "halted", None
    lease.prepared, lease.progress = None, None
    try:
        return runner._stop(lease.index, "rain", lease.origin, stop)
    finally:
        lease.prepared, lease.progress = None, None


def _outcome(runner: Any, lease: _WaterLease, item: Any, ordinal: int | None,
             n: Any, checkpoint: Any, schemas: dict, store: Any):
    from masy import dense_runner as dr
    if type(item) is n.DryHandoff and item.status == "complete":
        runner._precheck()
        _runtime(store, None, lease)
        if _entry_digest(runner, lease.prepared, n, schemas, _bounds(runner, lease.plan), lease.resolved) != lease.entry_sha256:
            _fail("water_entry_changed", "original entry changed during final native continuation")
        lease.phase = "finishing"
        result = dr._rain_accept(runner, lease.index, lease.event, lease.resolved, lease.prepared, item)
        lease.phase, lease.prepared, lease.progress, lease.token = "closed", None, None, None
        return result
    proof = _wet(runner, lease, item, n, checkpoint, schemas)
    if ordinal is None or float(item.result.state.t_s) != lease.plan[ordinal - 1]:
        _fail("water_pause_result", "native wet return was not the requested original boundary")
    runner._precheck()
    _runtime(store, None, lease)
    reservation = _reserve(runner, lease.plan)
    entry_sha = _entry_digest(runner, lease.prepared, n, schemas, _bounds(runner, lease.plan), lease.resolved)
    if lease.entry_sha256 and lease.entry_sha256 != entry_sha:
        _fail("water_entry_changed", "original entry witness changed")
    token = _WaterToken()
    lease.entry_sha256, lease.progress_sha256 = entry_sha, proof["sha256"]
    lease.progress, lease.ordinal, lease.pauses = item, ordinal, lease.pauses + 1
    lease.reservation, lease.token, lease.phase = reservation, token, "paused"
    return token


def _execute(runner: Any, lease: _WaterLease, ordinal: int | None, continuation: Any,
             n: Any, checkpoint: Any, schemas: dict, store: Any):
    from masy import dense_runner as dr
    try:
        item = dr._rain_native(runner, lease.prepared, continuation=continuation,
                               checkpoint_callback=None if ordinal is None else _callback(lease.plan[ordinal - 1]))
        return _outcome(runner, lease, item, ordinal, n, checkpoint, schemas, store)
    except dr._Stop as stop:
        return _halt(runner, lease, stop)
    except Exception as exc:
        if runner._active_evidence is None and runner._next == lease.index + 1:
            raise  # acceptance completed; never convert a lost acknowledgement to a failed event
        guarantee = ("failure during final adoption; publication may be partial; owner permanently halted"
                     if lease.phase == "finishing" else dr._UNCHANGED)
        lease.phase, lease.token, lease.prepared, lease.progress = "halted", None, None, None
        return _halt(runner, lease, dr._Stop("runner_error", "water_pause_failure", dr._detail(exc),
                                           guarantee=guarantee, exception=exc))
    except BaseException:
        if runner._active_evidence is None and runner._next == lease.index + 1:
            raise  # genuine acceptance visible; no stop/reset/replay
        guarantee = ("interrupted during final adoption; publication may be partial; owner permanently halted"
                     if lease.phase == "finishing" else dr._UNCHANGED)
        lease.phase, lease.token, lease.prepared, lease.progress = "halted", None, None, None
        try:
            _halt(runner, lease, dr._Stop("runner_error", "water_interrupted", "native water interrupted",
                                        guarantee=guarantee))
        except BaseException:
            pass  # preserve the original interrupt; the lease already refuses entry
        raise


def pause_water(runner: Any, *, runtime_reference: Any, boundary_ordinal: int):
    """Start one rain event; return an opaque wet token OR its normal dry/stop receipt.

    Callers serialize initial runner entry, as for DenseRunner itself. Once a
    lease is issued it is process/thread-bound and reentrant entry is refused.
    """
    from masy import dense_runner as dr
    from masy.dense_schedule import RainEvent
    _runner(runner)
    if runner._stop_receipt is not None:_fail("runner_halted", "stopped runner cannot pause")
    if runner._active_evidence is not None:_reject_active(runner)
    if runner._next >= len(runner._schedule.events):_fail("schedule_complete", "no next event")
    event = runner._schedule.events[runner._next]
    if type(event) is not RainEvent:_fail("water_pause_event", "next event must be rain")
    if type(boundary_ordinal) is not int or boundary_ordinal < 1:_fail("water_pause_ordinal", "plain positive ordinal required")
    lease = _WaterLease(weakref.ref(runner), weakref.ref(threading.current_thread()), os.getpid(),
                        runner._next, event, runner._bed.time_s)
    runner._active_evidence = lease
    try:
        n, checkpoint, schemas, store = _declare()
        _profile(runner, n)
        runner._precheck()
        runner._work_bytes_check(_metadata_bytes(runner._limits.max_report_rows))
        lease.runtime_bytes, lease.source, lease.environment = _runtime(store, runtime_reference)
        lease.resolved, lease.plan = _plan(runner, n)
        _ordinal(boundary_ordinal, lease.plan)
        _scan(lease.resolved, n, schemas, _bounds(runner, lease.plan))
        lease.reservation = _reserve(runner, lease.plan)
    except dr._Stop as stop:
        return _halt(runner, lease, stop)
    except BaseException:
        if runner._active_evidence is lease:runner._active_evidence = None
        lease.phase = "discarded"
        raise
    try:
        lease.stage_started = True
        lease.prepared = dr._rain_enter(runner, lease.index, lease.event, lease.resolved)
        from maple_syrup.storm import plan_boundaries
        prepared = lease.prepared
        actual_plan = tuple(float(t) for t in plan_boundaries(
            prepared.schedule, prepared.t, prepared.guard_end, prepared.every,
            max_report_rows=prepared.limits.max_report_rows))
        if (actual_plan != lease.plan or prepared.guard_end != lease.plan[-1]
                or prepared.every != runner._control.report_every_s):
            _fail("water_entry_plan", "prepared native plan differs from the admitted original plan")
        lease.entry_sha256 = _entry_digest(runner, lease.prepared, n, schemas, _bounds(runner, lease.plan), lease.resolved)
    except dr._Stop as stop:
        return _halt(runner, lease, stop)
    except Exception as exc:
        lease.phase, lease.token, lease.prepared, lease.progress = "halted", None, None, None
        return _halt(runner, lease, dr._Stop("runner_error", "water_entry_failure", dr._detail(exc), exception=exc))
    except BaseException:
        lease.phase, lease.token, lease.prepared, lease.progress = "halted", None, None, None
        try:
            _halt(runner, lease, dr._Stop("runner_error", "water_entry_interrupted", "native entry interrupted"))
        except BaseException:
            pass  # preserve the original interrupt; the lease already refuses entry
        raise
    return _execute(runner, lease, boundary_ordinal, None, n, checkpoint, schemas, store)


def resume_water(runner: Any, token: Any, *, boundary_ordinal: int | None = None):
    """Consume one process-local wet token; never restart original event entry."""
    lease = _lease(runner, token)
    if boundary_ordinal is not None:
        _ordinal(boundary_ordinal, lease.plan, lease.ordinal)
        if lease.pauses >= _MAX_PAUSES:_fail("water_pause_limit", "at most two wet pauses per event")
    n, checkpoint, schemas, store = _declare()
    _profile(runner, n)
    from masy import dense_runner as dr
    try:
        runner._precheck()
        _runtime(store, None, lease)
        _reserve(runner, lease.plan)
    except dr._Stop as stop:
        return _halt(runner, lease, stop)
    bounds = _bounds(runner, lease.plan)
    if _entry_digest(runner, lease.prepared, n, schemas, bounds, lease.resolved) != lease.entry_sha256:
        _fail("water_entry_changed", "original entry witness changed")
    if _wet(runner, lease, lease.progress, n, checkpoint, schemas)["sha256"] != lease.progress_sha256:
        _fail("water_progress_changed", "private wet progress changed while paused")
    # Recheck after every fallible preflight, before the indivisible owned call.
    _lease(runner, token)
    lease.phase, lease.token = "busy", None
    return _execute(runner, lease, boundary_ordinal, lease.progress, n, checkpoint, schemas, store)


def abandon_water(runner: Any, token: Any):
    """Drop only this private attempt and halt; no native reset or accepted change."""
    from masy import dense_runner as dr
    lease = _lease(runner, token)
    # The published ORIGINAL ordinal is the last native-validated pause time.
    # Do not report a subsequently modified private progress clock as known.
    elapsed = dr._elapsed(lease.origin, lease.plan[lease.ordinal - 1])
    return _halt(runner, lease, dr._Stop("incomplete", "water_pause_abandoned", "private water attempt abandoned",
                                       elapsed=elapsed))


# Closed native source layout; immutable metadata, never a process-state registry.
_FIELDS = MappingProxyType({
    ('maple_syrup.rainfall', 'RainfallProvenance'): ('kind', 'convention', 'path', 'sha256', 'size_bytes', 'start_clock', 'start_clock_s', 'rollover_record', 'n_records', 'blank_lines_skipped'),
    ('maple_syrup.rainfall', 'RainfallSchedule'): ('edges_s', 'intensity_mm_per_h', 'provenance', 'rate_m_per_s'),
    ('maple_syrup.complete_event', 'CompletionPolicy'): ('max_depth_m', 'surface_volume_m3', 'max_unit_discharge_m2_s', 'outlet_discharge_m3_s', 'hold_s', 'mobile_per_cell_class_kg', 'terminal_storage_water', 'deposit_mobile_at_flow_end'),
    ('maple_syrup.complete_event', 'QuietProgress'): ('since_s', 'last_t_s', 'ready'),
    ('maple_syrup.complete_event', 'EventProgress'): ('result', 'policy', 'quiet', 'control', 'planned_boundaries', 'origin_t_s', 'initial_surface_m3', 'initial_soil_m3', 'rainfall_end_s', 'report_every_s', 'max_report_rows', 'status'),
    ('maple_syrup.sediment_event', 'SedimentEventControl'): ('storm', 'sediment_courant_max', 'max_transport_substeps', 'commit', 'force_final_commit', 'max_commit_log', 'max_rejections_recorded', 'transport_scheme', 'phase_bins', 'transport_implementation', 'phase_storage', 'phase_max_nnz', 'record_exchange_history', 'deep_publication', 'deep_publication_high_water_records'),
    ('maple_syrup.sediment_event', 'SedimentEventState'): ('storm', 'bed', 'sediment_velocity_m_s', 'graph', 'network', 'grid', 'terrain', 'phase', 'routing_source_elevation_m'),
    ('maple_syrup.sediment_event', 'SedimentEventResult'): ('state', 'hydrograph', 'sediment_hydrograph', 'sediment_columns', 'boundaries', 'cumulative_rain_m', 'cumulative_intake_m', 'cumulative_saturation_return_m', 'cumulative_drainage_m', 'cumulative_export_m3', 'peak_depth_m', 'peak_velocity_m_s', 'last_velocity_m_s', 'peak_outlet_discharge_m3_s', 'time_of_peak_outlet_s', 'max_courant_old', 'max_courant_new', 'max_routing_cell_balance_residual_m', 'max_constitutive_residual_m', 'cell_steps_no_runon', 'cell_steps_partial_runon', 'cell_steps_complete_runon', 'cumulative_pickup_kg', 'cumulative_deposition_kg', 'cumulative_export_request_kg', 'by_class', 'initial_bed_by_cell_class_kg', 'initial_bed_inventory_kg', 'initial_mobile_kg', 'numerical_residual_scalar_abs_kg', 'peak_mobile_kg', 'time_of_peak_mobile_s', 'peak_export_rate_kg_s', 'time_of_peak_export_s', 'max_sediment_courant', 'max_decay_exponent', 'max_transport_cell_residual_kg', 'max_transport_substeps_used', 'n_transport_rejections', 'regime_cell_steps', 'ledger_process_totals_reset_kg', 'committed_net_bed_change_kg', 'n_commits', 'n_forced_commits', 'n_graph_changes', 'rerouted_cells_total', 'commit_log', 'max_phase_reconciliation_residual_kg', 'n_phase_canonicalized', 'n_phase_rounding_remnants', 'phase_steered_cells_total', 'n_accepted_steps', 'n_rejected_attempts', 'rejections', 'min_accepted_dt_s', 'max_accepted_dt_s', 'n_maple_water_calls', 'first_step_wall_s', 'first_step_cpu_s', 'remaining_wall_s', 'remaining_cpu_s', 'commit_wall_s', 'n_inventory_terms', 'exchange_history', 'deep_publication', 'n_reroutes', 'n_net_commits', 'n_routing_gate_evaluations', 'max_unrouted_net_displacement_m', 'routing_source_origin', 'routing_source_t_s'),
    ('maple_syrup.storm', 'StormControl'): ('max_dt_s', 'min_dt_s', 'max_retries', 'max_steps', 'courant_max', 'bisection_iterations', 'root_tolerance_m', 'implementation', 'root_solver', 'newton_max_iterations'),
    ('maple_syrup.storm', 'StormState'): ('t_s', 'depth_m', 'soil_water_m', 'discharge_m2_s'),
    ('maple_syrup.sediment_bed', 'BedState'): ('voxel_column', 'active_layer', 'water', 'ledger', 'committed_topography', 'sediment_availability'),
    ('maple_syrup.sediment_bed', 'TerrainReference'): ('initial_full_elevation_m', 'initial_committed_elevation_m', 'export_receiver_full', 'active_mask', 'friction_factor', 'dx_m', 'allow_masked_nodata', 'nodata_value', 'allow_pit_storage', 'allow_flat_storage'),
    ('maple_syrup.routing', 'RoutingGraph'): ('shape', 'dx_m', 'aspect', 'slope', 'slope_before_edge_rule', 'edge_rule_applied', 'friction_factor', 'active', 'outlet', 'receiver', 'level', 'level_order_host', 'level_bounds', 'conveyance', 'active_flat', 'outlet_flat', 'level_order', 'conveyance_lo', 'donor_position', 'donor_mask', 'input_sha256', 'xp', 'pit_storage', 'policy', 'flat_storage'),
    ('maple_syrup.sediment_transport', 'TransportNetwork'): ('shape', 'dx_m', 'n_cells', 'n_active', 'receiver_index', 'outlet_flat', 'active_flat', 'cells_x', 'rows_x', 'faces_x', 'sign_x', 'cells_y', 'faces_y', 'cols_y', 'sign_y', 'receiver_host', 'outlet_host', 'active_host', 'graph_input_sha256', 'xp', 'terminal_flat', 'terminal_host', 'n_terminal'),
    ('maple_syrup.sediment_physics', 'PhysicsGrid'): ('shape', 'cell_area_m2', 'slope', 'active', 'xp', 'terminal'),
    ('maple.core.types.voxel', 'VoxelColumnState'): ('mass_kg',),
    ('maple.core.types.active_layer', 'ActiveLayerState'): ('mass_kg',),
    ('maple.core.types.sediment_availability', 'SedimentAvailabilityState'): ('available_mass_kg', 'bound_mass_kg'),
    ('maple.core.types.sediment_ledger', 'SedimentLedgerState'): ('pending_bed_mass_change_kg', 'pending_compensation_kg', 'process_totals_kg', 'process_totals_compensation_kg', 'numerical_residual_positive_kg', 'numerical_residual_positive_compensation_kg', 'numerical_residual_negative_kg', 'numerical_residual_negative_compensation_kg', 'actual_mass_positive_kg', 'actual_mass_positive_compensation_kg', 'actual_mass_negative_kg', 'actual_mass_negative_compensation_kg', 'largest_residual_kg', 'n_physical_touches'),
    ('maple.core.types.topographic_commit', 'CommittedTopographyState'): ('reference_elevation_m', 'elevation_offset_m', 'committed_active_layer_mass_by_class_kg', 'committed_voxel_mass_by_class_kg', 'last_commit_time_s', 'commit_count'),
    ('maple.core.types.water', 'WaterState'): ('depth_m', 'mobile_mass_by_cell_class_kg'),
})
