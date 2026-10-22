"""Explicit in-memory default-flat Dense NumPy wind continuation.

No persistence, physics, reconstructed bed, alternate finalizer or native token
is exposed. Ordinary DenseRunner calls do not select this opt-in component.
SOURCE_UNACTIVATED until independent Root P2 acceptance.
The first source family requires a genuine explicit development composition;
old CPU locks and a matching file name are never runtime authority.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import struct
import sys
import threading
import weakref
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

__all__ = ["pause_wind", "resume_wind", "abandon_wind"]

_MAX_CELLS, _MAX_CLASSES, _MAX_VOXELS, _MAX_COHORTS = 64, 6, 16, 16000
_MAX_STEPS, _MAX_EVENTS, _MAX_PAUSES = 4096, 8, 2
_MAX_REFERENCE, _MAX_MANIFEST = 32 << 10, 1 << 20
# Canonical complete src/pyproject TABLE digests, not a claimed bundle ID.
# Locked committed Native fa2 (370 rows) / SYRUP 603 (65 rows) Source census;
# current-family wind pause/resume qualification remains pending Root.
_NATIVE_TABLE = "dd6b7575fc83a11178e8fa31738e1d5fc5df0d879a8c2e20337c18f7859a5640"
_SYRUP_TABLE = "390ff7cc0d04b5e7d713156315a8a1607549b8a7f4b75fa5f04aa9b1d1bb2d03"
_PROGRESS_FIELDS = ("native_time_s", "next_step_index", "n_steps", "advanced_start_time_s",
                    "advanced_duration_s", "advanced_end_time_s")


class _WindToken:
    __slots__ = ()


@dataclasses.dataclass(slots=True)
class _WindLease:
    owner: Any
    thread: Any
    pid: int
    index: int
    event: Any
    origin: float
    accepted: tuple
    accepted_record: dict
    phase: str = "busy"
    native_started: bool = False
    token: Any = None
    native_token: Any = None
    prepared: Any = None
    resolved: Any = None
    progress: Any = None
    progress_sha256: str = ""
    progress_bytes: bytes = b""
    ordinal: int = 0
    pauses: int = 0
    runtime_bytes: bytes = b""
    source: Any = None
    environment: str = ""
    entry_sha256: str = ""
    entry_metadata_sha256: str = ""
    entry_ids: tuple = ()
    reservation: Any = None


def _fail(code: str, detail: str) -> None:
    from masy.dense_runner import DenseRunnerError
    raise DenseRunnerError(code, detail)


def _reject_active(runner: Any) -> None:
    item = runner._active_evidence
    phase = item.phase if type(item) is _WindLease else "unowned"
    _fail("wind_lease_" + phase, "active wind ownership refuses fresh/passive/competing work")


def _runner(runner: Any) -> None:
    from masy import dense_runner as dr
    if type(runner) is not dr.DenseRunner:
        _fail("wind_pause_profile", "only the exact static default Dense CPU runner")


def _lease(runner: Any, token: Any) -> _WindLease:
    _runner(runner)
    item = runner._active_evidence
    if (type(item) is not _WindLease or item.owner() is not runner
            or item.pid != os.getpid() or item.thread() is not threading.current_thread()
            or item.phase != "paused" or type(token) is not _WindToken or item.token is not token
            or runner._stop_receipt is not None or runner._next != item.index):
        _fail("wind_token", "foreign, stale, consumed, busy or wrong-process/thread wind token")
    return item


def _declare():
    from masy import dense_boundary_store as store, dense_runner as dr
    from maple.aeolian.scheduler import baseline_continuation as continuation, block
    from maple.core.types.voxel import VoxelColumnState
    from maple.core.types.cohort import MobileCohortState
    continuation._native_types()  # actual closed import/type declaration before owning C origin/environment sweep
    if tuple(f.name for f in dataclasses.fields(continuation.WindBlockProgress)) != _PROGRESS_FIELDS:
        _fail("wind_native_layout", "the actual six-field native scalar progress API is required")
    return SimpleNamespace(n=dr._native(), store=store, block=block, continuation=continuation,
                           voxel=VoxelColumnState, mobile=MobileCohortState)


def _profile(runner: Any, api: Any) -> None:
    from masy import dense_runner as dr, dense_runner_checkpoint as boundary
    n, bed, cfg = api.n, runner._bed, runner._wind.config
    if (set(vars(runner)) != set(boundary.RUNNER_FIELDS) or runner._production is not None
            or type(runner._ownership()) is not dr._HostOwnership
            or type(bed) is not dr.DenseBed or type(bed.state) is not n.BedState
            or type(bed.state.voxel_column) is not api.voxel
            or type(runner._pool) is not api.mobile or type(runner._limits) is not dr.DenseRunnerLimits):
        _fail("wind_pause_profile", "closed26-field static host Dense profile required")
    runner._limits.__post_init__()
    if (type(cfg) is not n.SchedulerPhysicsConfig or cfg.execution_backend != "numpy"
            or cfg.detachment_source.mode != "aerodynamic" or cfg.dormancy_spec.enabled
            or cfg.particle_transport.mobile_state_representation != "flat"
            or cfg.particle_transport.routing_model != "hop_distance"
            or cfg.particle_transport.hop_distance_distribution != "lognormal"
            or cfg.particle_transport.direction_mode != "regional"
            or cfg.impact_rebound_splash_spec.response != "deposit_all"
            or cfg.water_coupling_spec.enabled or cfg.topographic_wind_spec.enabled
            or cfg.flow_separation_spec.enabled or cfg.vegetation_wake_spec.enabled
            or cfg.geometry.voxel_growth.enabled or cfg.prescribed_external_shear is not None
            or cfg.reduced_detachment is not None or cfg.prescribed_detachment is not None
            or cfg.surface_creep_spec is not None and cfg.surface_creep_spec.enabled):
        _fail("wind_pause_profile", "unchanged native default aerodynamic/dormancy-OFF flat NumPy profile")
    mass, g = bed.state.voxel_column.mass_kg, bed.context.geometry
    if (type(mass) is not n.np.ndarray or mass.ndim != 4
            or type(g.ny) is not int or type(g.nx) is not int
            or not 1 <= g.ny * g.nx <= _MAX_CELLS
            or not 1 <= runner._n_classes <= _MAX_CLASSES or not 1 <= mass.shape[2] <= _MAX_VOXELS
            or type(runner._pool.mass_kg) is not n.np.ndarray
            or not 0 <= len(runner._pool.mass_kg) <= _MAX_COHORTS
            or not 1 <= len(runner._schedule.events) <= _MAX_EVENTS
            or runner._limits.max_wind_accounting_steps > _MAX_STEPS):
        _fail("wind_pause_profile", "bounded original Dense dimensions/classes/steps/cohorts/events required")
    if (bed.vegetation is not None and bed.vegetation.objects is not None
            or any(getattr(runner._pool, name) is not None for name in (
                "launch_air_horizontal_speed_m_s", "live_velocity_m_s", "live_travelled_m",
                "live_turn_displacement_m", "live_has_turned"))):
        _fail("wind_pause_profile", "native object rasterization/live trajectories are outside this profile")


def _ordinal(value: Any, maximum: int, previous: int = 0) -> None:
    if type(value) is not int or not previous < value <= maximum:
        _fail("wind_pause_ordinal", "a later plain ORIGINAL completed-step ordinal within admitted work is required")


def _runtime(api: Any, reference: Any, lease: _WindLease | None = None):
    from masy import dense_original_inputs as oi
    if lease is None:
        api.store._plain_metadata(reference, _MAX_REFERENCE, "wind runtime reference", max_nodes=10000, max_depth=16)
        raw = oi._json(reference)
    else:
        raw = lease.runtime_bytes
    if len(raw) > _MAX_REFERENCE:
        _fail("wind_runtime_cap", "bounded canonical runtime reference required")
    view = SimpleNamespace(max_metadata_bytes=_MAX_REFERENCE, max_nodes=10000, max_depth=16)
    copied = oi._strict_json(raw, view)
    # This initial unwired adapter requires an explicit genuine new SOURCE
    # composition. It never treats old7f0/9eab or a supplied hash as a bundle.
    if copied.get("kind") != "source_manifest_prototype":
        _fail("wind_runtime_kind", "explicit new immutable development composition required for M1")
    source, environment = api.store._runtime_binding(copied, api.store.BoundaryStoreLimits(max_runtime_metadata_bytes=_MAX_MANIFEST))
    source_raw = api.store._read(copied["source_manifest_path"], _MAX_MANIFEST)
    if hashlib.sha256(source_raw).hexdigest() != copied["source_manifest_file_sha256"]:
        _fail("wind_runtime_changed", "source manifest changed after genuine owning binding")
    manifest = oi._strict_json(source_raw, SimpleNamespace(max_metadata_bytes=_MAX_MANIFEST,
                                                          max_nodes=100000, max_depth=24))
    if (manifest["source_identity"] != source
            or hashlib.sha256(oi._json(manifest["files"]["maple"])).hexdigest() != _NATIVE_TABLE
            or hashlib.sha256(oi._json(manifest["files"]["maple_syrup"])).hexdigest() != _SYRUP_TABLE):
        _fail("wind_resource_source", "complete owning source lacks this explicit native continuation/callee census")
    if lease is not None and (source != lease.source or environment != lease.environment):
        _fail("wind_runtime_changed", "original actual source/environment changed while active")
    return raw, source, environment


def _accepted_refs(runner: Any) -> tuple:
    return (runner._bed, runner._bed.state, runner._pool, runner._soil, runner._routing,
            runner._ledger, runner._totals, runner._receipts, runner._schedule,
            runner._accepted_bed, runner._accepted_private, runner._next, runner._n_accepted)


def _accepted_guard(runner: Any, lease: _WindLease) -> None:
    now = _accepted_refs(runner)
    if (any(a is not b for a, b in zip(now[:9], lease.accepted[:9], strict=True))
            or now[9:] != lease.accepted[9:]):
        from masy import dense_runner as dr
        raise dr._Stop("refused", "accepted_state_externally_mutated",
                       "an accepted boundary holder changed while wind was active", guarantee=dr._EXTERNAL)


def _reserve(runner: Any, api: Any) -> dict:
    """Additional named overlapping roles; opaque native peak is separately gated.

    Native continuation retains its own per-family max_work and raw/read caps.
    Its accepted45-callee census,8GiB whole-worker AS/CPU/wall and actual compiled
    guards remain required; neither this sum nor a native family proves fit.
    """
    from masy import dense_bed
    n, g, k = api.n, runner._bed.context.geometry, runner._n_classes
    own = runner._ownership()
    bed, pool = own.stage_bytes(), own.pool_bytes(runner._pool)
    from masy.dense_water_pause import _held_values
    owners = dense_bed._tree_nbytes(runner._bed.state, runner._pool, runner._soil,
        runner._inventory0, runner._totals, runner._bed.context, runner._bed.vegetation,
        runner._bed.threshold_modifier_state, runner._wind, runner._water, runner._control,
        *_held_values(runner._routing))  # one actual unique-array walk, including routing aliases
    native = api.continuation.WindContinuationLimits()
    rows = runner._limits.max_wind_accounting_steps
    original_entry = 4 * bed + 3 * pool + 8 * runner._inventory_elements * k + 32 * rows
    # At a paused seal anchors+current carry are one <=64MiB native raw set.
    # The next RHS can overlap a prior and next family; the native census is
    # separately mandatory and no physics-history list is introduced here.
    retained_native = native.max_raw_bytes
    pointer = struct.calcsize("P")
    node = sys.getsizeof({}) + sys.getsizeof([]) + 4 * sys.getsizeof(0) + 4 * pointer
    native_scan = native.max_nodes * node + native.max_arrays * 16 * pointer + 2 * native.max_metadata_bytes
    source_stream = (native.max_source_files * (4 * 4096 + node) + 65536
                     + 8192 * (2 * sys.getsizeof(()) + 4 * pointer) + 48 * 4 * 4096)
    # Genuine C binding parses a <=1MiB source manifest with100000 admitted
    # nodes, plus bounded32KiB reference and environment/origin records.
    binding = 100000 * node + 4 * _MAX_MANIFEST + 4 * _MAX_REFERENCE
    # Unchanged genuine bundle defaults:16MiB manifest/20000 rows; at
    # most four1MiB dist-info records +1MiB pyproject, duplicated conservatively.
    # All native source/archive hashing is streamed; runtime._hash_path keep=0.
    native_bundle = 4 * (16 << 20) + 20000 * node + 10 * (1 << 20)
    metadata = 128 * node + 8 * rows * (sys.getsizeof("") + 64 + pointer) + 8192
    # Native6B natural-commit generations, original owners and bounded
    # checkpoint/accounting scalar-family scratch; same source roles as217.
    commit = 6 * 8 * g.ny * g.nx * k * (runner._bed.state.voxel_column.mass_kg.shape[2] + 1)
    validation = 8 * max(g.ny * g.nx * k * 64, 4 * g.ny * g.nx * k,
                         runner._inventory_elements * k)
    total = owners + original_entry + retained_native + native_scan + source_stream + binding + native_bundle + metadata + commit + validation
    runner._work_bytes_check(total)
    return {"accepted_and_static_reference_bytes": owners, "original_entry_bytes": original_entry,
            "retained_native_raw_cap_bytes": retained_native, "native_scan_metadata_bytes": native_scan,
            "native_source_stream_metadata_bytes": source_stream, "genuine_binding_metadata_bytes": binding, "native_bundle_metadata_bytes": native_bundle,
            "wrapper_metadata_bytes": metadata, "native_commit_generations_bytes": commit,
            "native_validation_family_bytes": validation, "required_bytes": total,
            "native_largest_family_and_whole_worker_separate": True, "peak_or_fit_measured": False}


def _entry_metadata(runner: Any, lease: _WindLease, api: Any):
    from masy import dense_runner as dr, dense_original_inputs as oi
    p, n = lease.prepared, api.n
    if (type(p) is not dr._WindEntry or p.bed is not runner._bed or p.cfg is not runner._wind.config
            or type(p.own) is not dr._HostOwnership or p.own._bed is not runner._bed
            or type(p.ops) is not dr._HostArrayOps or p.ops.xp is not n.np or p.np is not n.np
            or p.n.np is not n.np or type(p.stage) is not dr._bed.DenseStage
            or p.stage._owner is not runner._bed._owner or p.stage.consumed
            or p.t != lease.origin or p.scheduler.physical.time_s != lease.origin):
        _fail("wind_entry_owner", "genuine unchanged original MASY prologue/bed/stage required")
    ids = tuple(id(getattr(p, f.name)) for f in dataclasses.fields(dr._WindEntry))
    if lease.entry_ids and ids != lease.entry_ids:
        _fail("wind_entry_changed", "an original retained entry object was replaced")
    if (type(p.plan) is not list or not 1 <= len(p.plan) <= min(_MAX_STEPS, runner._limits.max_wind_accounting_steps)
            or any(type(v) is not float or not math.isfinite(v) or v <= 0. for v in p.plan)):
        _fail("wind_entry_plan", "bounded original native coverage-plan scalars required before encoding")
    vector = p.inventory_before
    if (type(vector) is not n.np.ndarray or vector.dtype != n.np.dtype("float64")
            or vector.shape != (runner._n_classes,) or not vector.flags.c_contiguous):
        _fail("wind_entry_inventory", "original bounded canonical class vector required")
    facts = {"ids": list(ids), "resolved": lease.resolved, "stage_case": p.stage.case_sha256,
             "stage_identity": p.stage.identity_sha256, "stage_bytes": p.stage.nbytes,
             "time": p.t.hex(), "end": p.end.hex(), "upper": p.upper,
             "inventory_strides": list(vector.strides),
             "inventory_flags": [bool(vector.flags.writeable), bool(vector.flags.owndata),
                                 bool(vector.flags.aligned), bool(vector.flags.c_contiguous)],
             "inventory_pointer": vector.__array_interface__["data"][0],
             "plan": [float(v).hex() for v in p.plan],
             "quotient": str(p.quotient), "requested": str(p.requested),
             "covered": str(p.covered), "rounding": str(p.rounding)}
    api.store._plain_metadata(facts, _MAX_REFERENCE, "wind entry metadata", max_nodes=20000, max_depth=12)
    if not lease.entry_ids:
        lease.entry_ids = ids
    return vector, hashlib.sha256(oi._json(facts)).hexdigest()


def _entry_digest(runner: Any, lease: _WindLease, api: Any) -> str:
    vector, metadata = _entry_metadata(runner, lease, api)
    if not lease.entry_metadata_sha256:
        lease.entry_metadata_sha256 = metadata
    h = hashlib.sha256(metadata.encode("ascii"))
    h.update(memoryview(vector).cast("B"))  # at most8K borrowed bytes, no array copy
    return h.hexdigest()


def _progress_digest(lease: _WindLease, api: Any) -> str:
    """Wrapper scalar-integrity DATA only; native resume remains sole native seal."""
    from masy import dense_original_inputs as oi
    value = {"progress": lease.progress, "ordinal": lease.ordinal, "pauses": lease.pauses,
             "native_token_identity": id(lease.native_token)}
    api.store._plain_metadata(value, 4096, "wind scalar progress", max_nodes=128, max_depth=8)
    if oi._json(lease.progress) != lease.progress_bytes:
        _fail("wind_progress_changed", "validated scalar snapshot changed")
    return hashlib.sha256(oi._json(value)).hexdigest()


def _callback(previous: int, target: int):
    count = previous
    def pause_requested():
        nonlocal count
        count += 1
        return count == target
    return pause_requested


def _discard_native(lease: _WindLease, api: Any, primary: BaseException) -> None:
    token, lease.native_token = lease.native_token, None
    if token is None:
        return
    if api is None:
        return  # no further native import after an unqualified declaration failure
    try:
        api.block.abandon_wind_block(token)
    except BaseException as cleanup:
        # Genuine consumed/halted native owners may correctly refuse stale;
        # this cleanup is not validation, rewind, reset or a second attempt.
        try:
            primary.add_note("wind cleanup unavailable: " + type(cleanup).__name__)
        except BaseException:
            pass


def _halt(runner: Any, lease: _WindLease, stop: Any, api: Any):
    from masy import dense_runner as dr
    phase = lease.phase
    lease.phase, lease.token = "halted", None
    _discard_native(lease, api, stop.exception if stop.exception is not None else stop)
    guarantee = stop.guarantee
    if phase == "finishing" and guarantee != dr._EXTERNAL:
        guarantee = "failure during final adoption; publication may be partial; owner permanently halted"
    from masy import dense_original_inputs as oi
    last = None
    if type(lease.progress_bytes) is bytes and lease.progress_bytes:
        try:
            last = oi._strict_json(lease.progress_bytes, SimpleNamespace(
                max_metadata_bytes=4096, max_nodes=128, max_depth=8))
        except Exception:
            pass  # scalar diagnostic unavailable, never retain tampered native/array objects
    evidence = {"schema": "masy.dense_wind_pause.stop/1", "phase": "halted",
                "last_validated": last, "entry_native_time": lease.origin.hex(),
                "unknown_current_endpoint": lease.native_started, "last_accepted": lease.accepted_record,
                "persistent_active_restart": False}
    elapsed = None if type(last) is not dict else last.get("elapsed_local_s")
    closed = dr._Stop(stop.status, stop.code, stop.detail, exception=stop.exception,
                      elapsed=stop.elapsed if stop.elapsed is not None else elapsed,
                      evidence=evidence, guarantee=guarantee)
    try:
        return runner._stop(lease.index, "wind", lease.origin, closed)
    finally:
        lease.prepared = lease.native_token = None


def _outcome(runner: Any, lease: _WindLease, item: Any, ordinal: int | None, api: Any):
    from masy import dense_runner as dr
    from masy.clock import encode_time
    # Genuine complete result wins, even if the requested poll was beyond the
    # last unfinished original boundary. Never finalize a partial token.
    if type(item) is api.block.WindBlockResult:
        runner._precheck(); _accepted_guard(runner, lease)
        _runtime(api, None, lease)
        if _entry_digest(runner, lease, api) != lease.entry_sha256:
            _fail("wind_entry_changed", "original entry changed before final acceptance")
        if lease.progress is not None:
            actual = (item.advanced_start_time_s, item.advanced_duration_s, item.advanced_end_time_s)
            if tuple(lease.prepared.ops.scalar(v).hex() for v in actual) != tuple(lease.progress['advanced_hex']):
                _fail("wind_progress_changed", "native original advanced anchors changed")
        lease.phase = "finishing"
        result = dr._wind_accept(runner, lease.index, lease.event, lease.resolved, lease.prepared, item)
        lease.phase, lease.prepared, lease.native_token, lease.token = "closed", None, None, None
        return result
    if type(item) is not api.continuation.WindContinuationToken or ordinal is None:
        _fail("wind_pause_result", "only genuine native completion or requested paused token is accepted")
    lease.native_token = item
    progress = api.block.wind_block_progress(item)  # exactly once/newly published pause, NEVER on resume
    if (type(progress) is not api.continuation.WindBlockProgress
            or type(progress.next_step_index) is not int or type(progress.n_steps) is not int
            or progress.next_step_index != ordinal or not lease.ordinal < ordinal < progress.n_steps
            or progress.n_steps > _MAX_STEPS):
        _fail("wind_pause_result", "requested ORIGINAL unfinished native boundary was not returned")
    native_t = lease.prepared.ops.scalar(progress.native_time_s)
    global_t = runner._ledger.global_elapsed(native_t)
    if runner._ledger.native_from_global(global_t).hex() != native_t.hex():
        _fail("wind_pause_clock", "native epoch/global clock inverse changed the binary64 endpoint")
    advanced = tuple(lease.prepared.ops.scalar(getattr(progress, name)).hex() for name in _PROGRESS_FIELDS[3:])
    if lease.progress is not None and advanced != tuple(lease.progress['advanced_hex']):
        _fail("wind_progress_changed", "native original advanced anchors changed between pauses")
    runner._precheck(); _accepted_guard(runner, lease)
    _runtime(api, None, lease)
    if _entry_metadata(runner, lease, api)[1] != lease.entry_metadata_sha256:
        _fail("wind_entry_changed", "original entry metadata changed while publishing pause")
    token = _WindToken()
    status = {"native_time": encode_time(native_t), "elapsed_local_s": dr._elapsed(lease.origin, native_t),
              "global_elapsed_s": dr._fraction_text(global_t), "advanced_hex": list(advanced),
              "next_step_index": progress.next_step_index, "n_steps": progress.n_steps}
    lease.progress, lease.ordinal, lease.pauses = status, ordinal, lease.pauses + 1
    from masy import dense_original_inputs as oi
    lease.progress_bytes = oi._json(status)
    lease.progress_sha256 = _progress_digest(lease, api)
    lease.token, lease.phase = token, "paused"
    return token


def _execute(runner: Any, lease: _WindLease, ordinal: int | None, api: Any, native_token: Any = None):
    from masy import dense_runner as dr
    callback = None if ordinal is None else _callback(lease.ordinal, ordinal)
    lease.native_token = native_token  # retain only this owned token for passive-failure abandonment
    lease.native_started = True
    try:
        try:
            if native_token is None:
                p = lease.prepared
                item = api.block.advance_wind_block_pausable(p.scheduler, p.block, p.cfg,
                    pause_requested=callback, limits=api.continuation.WindContinuationLimits())
            else:
                item = api.block.resume_wind_block(native_token, pause_requested=callback)
        except Exception as exc:
            status = "native_refused" if isinstance(exc, ValueError) else "native_error"
            raise dr._Stop(status, "wind_native_failure", dr._detail(exc), exception=exc) from exc
        # Retain at most one current native token, including the failure path.
        lease.native_token = item if type(item) is api.continuation.WindContinuationToken else None
        return _outcome(runner, lease, item, ordinal, api)
    except dr._Stop as stop:
        if runner._active_evidence is None and runner._next == lease.index + 1:
            raise
        return _halt(runner, lease, stop, api)
    except Exception as exc:
        if runner._active_evidence is None and runner._next == lease.index + 1:
            raise  # acceptance completed, never turn lost acknowledgement into stop/replay
        return _halt(runner, lease, dr._Stop("runner_error", "runner_internal_error", dr._detail(exc), exception=exc), api)
    except BaseException:
        if runner._active_evidence is None and runner._next == lease.index + 1:
            raise
        try:
            _halt(runner, lease, dr._Stop("runner_error", "wind_interrupted", "native wind interrupted"), api)
        except BaseException:
            pass  # preserve primary interruption; phase/token already refuse reuse
        raise


def pause_wind(runner: Any, *, runtime_reference: Any, boundary_ordinal: int):
    """Run one native wind event until a genuine unfinished boundary or completion."""
    from masy import dense_runner as dr
    from masy.dense_schedule import WindEvent
    _runner(runner)
    index, event, start = dr._step_start(runner)
    if type(event) is not WindEvent:
        _fail("wind_pause_event", "next event must be wind")
    _ordinal(boundary_ordinal, min(_MAX_STEPS, runner._limits.max_wind_accounting_steps))
    api = _declare()
    _profile(runner, api)
    # No accepted array is copied or republished by this small reference record.
    refs = _accepted_refs(runner)
    record = {"event_index": index, "native_time": start.hex(),
              "bed_identity": runner._accepted_bed, "private_identity": runner._accepted_private,
              "n_accepted": runner._n_accepted}
    lease = _WindLease(weakref.ref(runner), weakref.ref(threading.current_thread()), os.getpid(),
                       index, event, start, refs, record)
    runner._active_evidence = lease
    try:
        lease.resolved = dr._step_resolve(runner, index, start)
        lease.reservation = _reserve(runner, api)
        lease.runtime_bytes, lease.source, lease.environment = _runtime(api, runtime_reference)
        lease.prepared = dr._wind_enter(runner, index, event, lease.resolved)
        lease.entry_sha256 = _entry_digest(runner, lease, api)
    except dr._Stop as stop:
        return _halt(runner, lease, stop, api)
    except Exception as exc:
        return _halt(runner, lease, dr._Stop("runner_error", "runner_internal_error", dr._detail(exc), exception=exc), api)
    except BaseException:
        try:
            _halt(runner, lease, dr._Stop("runner_error", "wind_interrupted", "native wind entry interrupted"), api)
        except BaseException:
            pass
        raise
    return _execute(runner, lease, boundary_ordinal, api)


def resume_wind(runner: Any, token: Any, *, boundary_ordinal: int | None = None):
    """Consume one wrapper token and call genuine native resume, never original entry."""
    from masy import dense_runner as dr
    lease = _lease(runner, token)
    if boundary_ordinal is not None:
        _ordinal(boundary_ordinal, min(_MAX_STEPS, runner._limits.max_wind_accounting_steps), lease.ordinal)
        if lease.pauses >= _MAX_PAUSES:
            _fail("wind_pause_limit", "at most two wind pauses per original event")
    api = None
    try:
        api = _declare()
        _profile(runner, api)
        runner._precheck(); _accepted_guard(runner, lease)
        _runtime(api, None, lease)
        _reserve(runner, api)
        if _progress_digest(lease, api) != lease.progress_sha256:
            _fail("wind_progress_changed", "last independently validated wrapper scalar progress changed")
        if _entry_digest(runner, lease, api) != lease.entry_sha256:
            _fail("wind_entry_changed", "original entry changed while paused")
    except dr._Stop as stop:
        return _halt(runner, lease, stop, api)
    except Exception as exc:
        return _halt(runner, lease, dr._Stop("runner_error", "runner_internal_error", dr._detail(exc), exception=exc), api)
    except BaseException:
        try:
            _halt(runner, lease, dr._Stop("runner_error", "wind_interrupted", "wind resume admission interrupted"), api)
        except BaseException:
            pass
        raise
    # Last wrapper admission AFTER every fallible preflight. No P1 call here:
    # native resume alone performs the original passive seal before consumption.
    _lease(runner, token)
    original = lease.native_token
    lease.phase, lease.token, lease.native_token = "busy", None, None
    return _execute(runner, lease, boundary_ordinal, api, original)


def abandon_wind(runner: Any, token: Any):
    """Discard only private work and halt incomplete; no next event/reset/replay."""
    from masy import dense_runner as dr
    lease = _lease(runner, token)
    api = _declare()
    return _halt(runner, lease, dr._Stop("incomplete", "wind_pause_abandoned", "private wind attempt abandoned"), api)
