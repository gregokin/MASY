"""Bounded private companion to the actual Native paused-wind graph codec.

C supplies history/custody. Native supplies its codec, validation and transport.
This module never observes a private Native owner/carry or finalizes a pause.
"""
from __future__ import annotations

import dataclasses
import os
import stat
import sys
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

from masy import dense_active_checkpoint as a

SCHEMA = "masy.dense_active_wind_checkpoint/1"
EXPORT_SCHEMA = "masy.dense_active_wind_export/1"
_CAP = 1 << 20
_WIRE_CAP = 8 << 20  # unchanged first-component owning file allowance
_FILE = "continuation.bin"
_ISSUED = object()
_CAPSULE = ("schema", "parent", "source", "environment", "operation", "attempt",
            "index", "resolved", "origin", "ordinal", "pauses", "entry", "progress",
            "static_sha256", "prescribed", "native_limits")
_ENTRY = ("case", "identity", "time", "nbytes", "stamp", "inventory", "plan", "end",
          "quotient", "upper", "requested", "covered", "rounding")
_RAW = ("name", "size", "sha256", "dev", "ino", "mtime_ns", "ctime_ns", "mode",
        "parent_dev", "parent_ino", "parent_mtime_ns", "parent_ctime_ns")
_COMPLETION_FIELDS = (12, 8, 31)  # actual fa2 physical/state/result Source fields


@dataclasses.dataclass(frozen=True, slots=True)
class _ColdBinding:
    key: object
    parent: object
    owner: object
    path: Path
    export_sha256: str
    capsule_raw: bytes


def _declare():
    from masy import dense_boundary_store as c, dense_runner as dr, dense_wind_pause as pause
    from maple.aeolian.scheduler import baseline_continuation as continuation, baseline_continuation_codec as codec
    from maple.aeolian.scheduler.state import SchedulerPhysicalState, SchedulerState
    from maple.core.types.vegetation import VegetationState, VegetationRasterState, validate_vegetation_state
    from maple.core.types.threshold_modifiers import ThresholdModifierState, validate_threshold_modifier_state
    api = pause._declare()
    return SimpleNamespace(c=c, dr=dr, pause=pause, api=api, n=api.n,
        continuation=continuation, codec=codec, physical=SchedulerPhysicalState, state=SchedulerState,
        vegetation=VegetationState, raster=VegetationRasterState,
        threshold=ThresholdModifierState, validate_vegetation=validate_vegetation_state,
        validate_threshold=validate_threshold_modifier_state)


def _limits(d):
    value = d.continuation.WindContinuationLimits()
    value.__post_init__()
    return value


def _limits_record(value):
    return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}


def _codec_charge(value):
    """Additive Source-defined serialization receivers; no whole-heap fit claim.

    Graph child visits bound total reference slots, not only distinct nodes.
    The Native codec retains at most9 slots per record and rank6 array tuples.
    Native per-owner work and external interpreter/provider/AS guards remain.
    """
    n, arrays, meta = value.max_nodes, value.max_arrays, value.max_metadata_bytes
    pointer = sys.getsizeof((None,)) - sys.getsizeof(())
    integer = sys.getsizeof(sys.maxsize)
    sequence = sys.getsizeof(()) + sys.getsizeof([])
    # Three graph receiver roles are conservatively charged independently.
    # Export retains original artifact/scratch decode/canonical encode; restore
    # retains caller artifact/decode-parser/canonical encode. Both have3E+N.
    record = (sys.getsizeof(()) + 9 * pointer + 9 * integer
              + sys.getsizeof("") + sys.getsizeof(b""))
    # Encoder/walk visits<=n. Before the parser's reachable-graph walk, each
    # stored reference still occupies>=4 wire bytes; admit all metadata/4
    # references as well, including refusal before that walk completes.
    edges = n + meta // 4
    references = edges * (2 * pointer + 2 * integer)
    # Five dict/set maps occur across planner/parser/build (memo, active,
    # values, visited, walk-active); the Source node projection is conservative.
    maps = 5 * n * (sys.getsizeof({}) + sys.getsizeof(set()) + 4 * pointer + 2 * integer)
    array_records = arrays * (2 * sys.getsizeof(()) + 12 * (pointer + integer))
    text_headers = (n + meta // 4) * (sys.getsizeof("") + sys.getsizeof(b"") + 4 * pointer)
    graph = n * record + references + maps + array_records + text_headers + 6 * sequence
    # Source74-class/522-field registry, qualified<=256/field<=128 text.
    registry = (74 * (3 * sys.getsizeof(()) + 3 * pointer + sys.getsizeof("") + 4 * 256)
                + 522 * (pointer + sys.getsizeof("") + 4 * 128))
    wire = min(value.max_read_bytes, 40 + meta + value.max_raw_bytes + 32)
    # Writer capture is a closed Source-generated graph, NOT an arbitrary JSON
    # parse:<=32768 existing capsule nodes (actual new lists<=4096+6 leaves),
    # container/text/scalar headers per node, plus6 full wire/text receivers.
    # Existing Source/parent/resolved/progress objects remain separately held.
    # Readers still use the larger raw-parser projection on exact stat sizes.
    # No new limit is introduced or inherited parser admission weakened.
    wrapper_node = (sys.getsizeof({}) + sys.getsizeof([]) + sys.getsizeof(())
                    + sys.getsizeof(set()) + sys.getsizeof("") + sys.getsizeof(b"")
                    + 8 * pointer + integer)
    wrapper_metadata = a._NODES * wrapper_node + 6 * _CAP
    # Native _source_observation snapshots<=8192 actual module registry rows,
    # while hashing<=1024 Source files in bounded chunks. Two receiver families
    # may survive a primary exception; they do not authorize any new imports.
    source_observers = (2 * 8192 * (3 * sys.getsizeof(()) + 6 * pointer)
                        + 2 * (1 << 20))
    # Four10-row layout snapshots, their rank2 shape/stride tuples and loop
    # lists/tuples, plus three exact non-array completed shells. Native/provider
    # validation remains independent; the explicit boolean expression peak
    # from the two owning raster validators is four C-sized masks.
    control_row = (sys.getsizeof(()) + 10 * pointer + 4 * integer + sys.getsizeof("") + 12
                   + 2 * (sys.getsizeof(()) + 2 * integer))
    control = (40 * control_row + 8 * (sys.getsizeof(()) + sys.getsizeof([]) + 10 * pointer)
               + sum(_COMPLETION_FIELDS) * (sys.getsizeof({}) + 4 * pointer)
               + 4 * value.max_cells)
    return {"codec_wire_bound": wire, "codec_live_wire_and_scratch_bytes": 3 * wire + value.max_raw_bytes,
            "codec_graph_metadata_bytes": 3 * graph + 3 * registry + 6 * meta,
            "codec_parser_pre_walk_reference_bound": edges,
            "codec_text_header_bound_each_graph": n + meta // 4,
            "wrapper_capsule_descriptor_metadata_bytes": wrapper_metadata,
            "native_source_observer_metadata_bytes": source_observers,
            "completion_control_metadata_and_explicit_masks_bytes": control,
            "codec_frame_provider_heap_separately_qualified": True,
            "required_bytes": (3 * wire + value.max_raw_bytes + 3 * graph + 3 * registry + 6 * meta
                               + wrapper_metadata + source_observers + control)}


def reservation_increment():
    """Actual owning typed Native limits; no policy copied into a new codec."""
    from maple.aeolian.scheduler.baseline_continuation import WindContinuationLimits
    limits = WindContinuationLimits()
    limits.__post_init__()
    return _codec_charge(limits)


def _admit(parent, runner, d, *, metadata_bytes=0):
    a._parent(parent, runner)
    if type(metadata_bytes) is not int or not 0 <= metadata_bytes <= 2 * _CAP:
        a._fail("wind_metadata_cap", "bounded raw descriptor/capsule occurrences required")
    d.pause._profile(runner, d.api)
    d.pause._reserve(runner, d.api)  # unchanged original native/entry reservation
    from masy import dense_active_water_store as store
    handle = parent._handle
    monitored = store._monitored_wind_codec(handle, runner, metadata_bytes)
    if monitored is not None:
        return monitored
    chain = (store._metadata_stat_charge(handle, store._inventory(handle))
             if d.c._active_store(handle._header) else 0)
    base = store._reservation(handle._limits, runner._limits.max_work_bytes, chain)
    extra = reservation_increment()
    memory = base["memory_bytes"] + extra["required_bytes"] + 2 * store._parse_charge(metadata_bytes)
    if memory > handle._limits.max_memory_bytes:
        d.c._fail("memory_cap", "complete original C/active owners plus wind codec/metadata")
    return {**extra, "memory_bytes": memory, "Source_logical_not_heap_fit": True}


def _entry(runner, lease, d):
    p = lease.prepared
    if type(p) is not d.dr._WindEntry:
        a._fail("wind_cold_entry", "actual original MASY WindEntry required")
    d.pause._entry_digest(runner, lease, d.api)
    return {"case": p.stage.case_sha256, "identity": p.stage.identity_sha256,
        "time": p.t.hex(), "nbytes": p.stage.nbytes,
        "stamp": float(p.stage.state.committed_topography.last_commit_time_s).hex(),
        "inventory": [float(x).hex() for x in p.inventory_before],
        "plan": [x.hex() for x in p.plan], "end": p.end.hex(),
        "quotient": str(p.quotient), "upper": p.upper, "requested": str(p.requested),
        "covered": str(p.covered), "rounding": str(p.rounding)}


def _prescribed(runner):
    return dict(runner._bed._prescribed_digests)


def _capture(parent, runner, lease, path, d):
    return {"schema": SCHEMA, "parent": a._parent_record(parent, runner),
        "source": lease.source, "environment": lease.environment,
        "operation": a._operation(parent, runner), "attempt": a._sha(str(path).encode()),
        "index": lease.index, "resolved": lease.resolved, "origin": lease.origin.hex(),
        "ordinal": lease.ordinal, "pauses": lease.pauses, "entry": _entry(runner, lease, d),
        "progress": lease.progress, "static_sha256": runner._static_sha256,
        "prescribed": _prescribed(runner), "native_limits": _limits_record(_limits(d))}


def _admit_capsule(parent, runner, facts, path, d):
    from masy.dense_schedule import WindEvent
    d.c._keys(facts, _CAPSULE, "wind capsule")
    d.c._keys(facts["entry"], _ENTRY, "original wind entry")
    if (facts["schema"] != SCHEMA
            or d.c.oi._json(facts["parent"]) != d.c.oi._json(a._parent_record(parent, runner))
            or a._metadata(facts["source"], _CAP, "source") != parent._source
            or facts["environment"] != parent._environment
            or facts["operation"] != a._operation(parent, runner)
            or type(facts["index"]) is not int or facts["index"] != runner._next
            or not 0 <= runner._next < len(runner._schedule.events)
            or type(runner._schedule.events[runner._next]) is not WindEvent
            or a._float(facts["origin"]).hex() != runner._bed.time_s.hex()
            or facts["static_sha256"] != runner._static_sha256
            or d.c.oi._json(facts["prescribed"]) != d.c.oi._json(_prescribed(runner))
            or d.c.oi._json(facts["native_limits"]) != d.c.oi._json(_limits_record(_limits(d)))):
        a._fail("wind_capsule_parent", "exact actual accepted parent/event/source/controls required")
    d.c._digest(facts["attempt"], "actual wind writer attempt")
    if facts["attempt"] != a._sha(str(path).encode()):
        a._fail("wind_capsule_path", "original writer final canonical path differs")
    a._integer(facts["pauses"], d.pause._MAX_PAUSES, "original wind pause count", 1)
    d.pause._ordinal(facts["ordinal"], min(d.pause._MAX_STEPS, runner._limits.max_wind_accounting_steps))
    d.c._keys(facts["progress"], ("native_time", "elapsed_local_s", "global_elapsed_s", "advanced_hex",
                                 "next_step_index", "n_steps"), "wind scalar progress")


def _stat(path, cap, d):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid()
            or info.st_mode & 0o222 or not 1 <= info.st_size <= cap):
        a._fail("wind_raw_file", "owned readonly single-link bounded original writer file required")
    return info


def _raw(path, d, *, expected=None):
    rootfd = a._directory(path, d.c)
    try:
        with os.scandir(rootfd) as entries:
            names = []
            for item in entries:
                if len(names) >= 1 or item.name != _FILE:
                    a._fail("wind_raw_inventory", "exact single Native byte artifact required")
                names.append(item.name)
        if names != [_FILE]:
            a._fail("wind_raw_inventory", "missing genuine Native byte artifact")
        file = path / _FILE
        info = _stat(file, _WIRE_CAP, d)
        parent = os.fstat(rootfd)
        digest = d.c.runtime._hash_path(file, budget=_WIRE_CAP)[0]
        after = file.lstat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
        if any(getattr(info, n) != getattr(after, n) for n in fields):
            a._fail("wind_raw_changed", "original Native file changed while hashing")
        row = dict(zip(_RAW, (_FILE, info.st_size, digest, info.st_dev, info.st_ino,
            info.st_mtime_ns, info.st_ctime_ns, info.st_mode, parent.st_dev, parent.st_ino,
            parent.st_mtime_ns, parent.st_ctime_ns), strict=True))
        d.c._same_directory(path, rootfd)
        if expected is not None and d.c.oi._json(row) != d.c.oi._json(expected):
            a._fail("wind_raw_changed", "exact original writer raw custody differs")
        return row
    finally:
        os.close(rootfd)


def _progress(lease, value, runner, d):
    from masy.clock import encode_time
    if type(value) is not d.continuation.WindBlockProgress:
        a._fail("wind_progress", "actual six-scalar Native progress required")
    t = lease.prepared.ops.scalar(value.native_time_s)
    global_t = runner._ledger.global_elapsed(t)
    if runner._ledger.native_from_global(global_t).hex() != t.hex():
        a._fail("wind_clock", "exact original epoch/global inverse required")
    return {"native_time": encode_time(t), "elapsed_local_s": d.dr._elapsed(lease.origin, t),
        "global_elapsed_s": d.dr._fraction_text(global_t),
        "advanced_hex": [lease.prepared.ops.scalar(getattr(value, n)).hex() for n in d.pause._PROGRESS_FIELDS[3:]],
        "next_step_index": value.next_step_index, "n_steps": value.n_steps}


def _check(parent, runner, lease, token, d):
    a._parent(parent, runner)
    d.pause._lease(runner, token)
    runner._precheck()
    d.pause._accepted_guard(runner, lease)
    d.pause._runtime(d.api, None, lease)
    if d.pause._entry_digest(runner, lease, d.api) != lease.entry_sha256:
        a._fail("wind_entry_changed", "original accepted entry changed during export")
    if d.pause._progress_digest(lease, d.api) != lease.progress_sha256:
        a._fail("wind_progress_changed", "original published scalar progress changed")
    if lease.cold_binding is not None:
        _check_binding(runner, lease, d)


def _check_binding(runner, lease, d):
    binding = lease.cold_binding
    if (type(binding) is not _ColdBinding or binding.key is not _ISSUED
            or binding.owner() is not runner or binding.parent() is None):
        a._fail("wind_cold_binding", "actual issued cold C relation required")
    parent = binding.parent()
    a._parent(parent, runner)
    _admit(parent, runner, d, metadata_bytes=len(binding.capsule_raw))
    facts = a._decode(binding.capsule_raw, _CAP)
    _admit_capsule(parent, runner, facts, binding.path, d)
    if a._metadata(_entry(runner, lease, d), _CAP, "current original entry") != a._metadata(facts["entry"], _CAP, "saved original entry"):
        a._fail("wind_cold_entry", "real local original entry no longer equals the writer")
    _stat(binding.path / "export.json", _CAP, d)
    if a._sha(d.c._read(binding.path / "export.json", _CAP)) != binding.export_sha256:
        a._fail("wind_export_changed", "selected original descriptor drift")
    if d.c._read(binding.path / "entry/capsule.json", _CAP) != binding.capsule_raw:
        a._fail("wind_capsule_changed", "selected original entry drift")
    return parent, facts


def export_wind_pause(parent, runner, token, *, attempt):
    """Actual nonconsuming Native export; C alone may publish its descriptor."""
    d = _declare()
    lease = d.pause._lease(runner, token)
    _admit(parent, runner, d)
    _check(parent, runner, lease, token, d)
    path = a._attempt_path(parent, attempt)
    reserve = _WIRE_CAP + 2 * _CAP
    free = os.statvfs(path.parent)
    handle = parent._handle
    if (free.f_bavail * free.f_frsize < handle._limits.free_reserve_bytes + reserve
            or free.f_favail < handle._limits.free_reserve_inodes + 6):
        d.c._fail("storage_cap", "original free reserve plus exact child files/directories")
    raw = None
    fd = None
    try:
        os.mkdir(path, 0o700)
        os.mkdir(path / "entry", 0o700)
        os.mkdir(path / "native", 0o700)
        fd = a._directory(path, d.c)
        # Native owns all binary parsing/packing/schema/array policy.
        raw = d.codec.export_paused_continuation(lease.native_token)
        if type(raw) is not bytes or not 1 <= len(raw) <= _WIRE_CAP:
            a._fail("wind_wire_cap", "original8MiB file allowance before write")
        capsule_raw = a._metadata(_capture(parent, runner, lease, path, d), _CAP, "actual original wind capsule")
        d.c._write(path / "entry/capsule.json", capsule_raw)
        os.chmod(path / "entry/capsule.json", 0o444, follow_symlinks=False)
        d.c._write(path / "native" / _FILE, raw)
        raw = None
        os.chmod(path / "native" / _FILE, 0o444, follow_symlinks=False)
        os.chmod(path / "native", 0o555, follow_symlinks=False)
        row = _raw(path / "native", d)
        descriptor = {"schema": EXPORT_SCHEMA, "capsule_sha256": a._sha(capsule_raw),
            "capsule_bytes": len(capsule_raw), "raw_file": row}
        export_raw = a._metadata(descriptor, _CAP, "wind export descriptor")
        d.c._write(path / "export.json", export_raw)
        os.chmod(path / "export.json", 0o444, follow_symlinks=False)
        _check(parent, runner, lease, token, d)
        _raw(path / "native", d, expected=row)
        d.c._same_directory(path, fd)
        for name in ("entry", "native"):
            subfd = a._directory(path / name, d.c)
            try:
                os.fsync(subfd)
            finally:
                os.close(subfd)
        os.fsync(fd)
        return a._sha(export_raw)
    except BaseException as primary:
        try:
            _check(parent, runner, lease, token, d)
        except BaseException as closing:
            try:
                d.pause._halt(runner, lease, d.dr._Stop("runner_error", "wind_export_changed",
                    "private export changed/unknown; selected C authority remains", exception=None), d.api)
            except BaseException:
                pass
            primary.add_note("closing wind export check refused: " + type(closing).__name__[:64])
        raise
    finally:
        raw = None
        if fd is not None:
            os.close(fd)


def _metadata_read(path, admitted, d):
    # Reuse the original no-follow, fstat/read/fstat owner. Its read cap is the
    # exact admitted size, not merely the larger format ceiling.
    raw, identity = d.c._read(path, admitted.st_size, _with_identity=True)
    current = _stat(path, _CAP, d)
    if (len(raw) != admitted.st_size or identity != d.c.oi._stat_id(admitted)
            or d.c.oi._stat_id(current) != d.c.oi._stat_id(admitted)
            or current.st_mode != admitted.st_mode):
        a._fail("wind_metadata_changed", "metadata differs from its exact admitted raw occurrence")
    return raw


def _read_descriptor(path, expected_export_sha256, d, *, admitted=None):
    d.c._digest(expected_export_sha256, "external original wind descriptor SHA")
    descriptor_stat, capsule_stat = (admitted if admitted is not None else
        (_stat(path / "export.json", _CAP, d), _stat(path / "entry/capsule.json", _CAP, d)))
    _stat(path / "native" / _FILE, _WIRE_CAP, d)
    raw = _metadata_read(path / "export.json", descriptor_stat, d)
    if a._sha(raw) != expected_export_sha256:
        a._fail("wind_export_changed", "actual original descriptor bytes differ")
    value = a._decode(raw, _CAP)
    d.c._keys(value, ("schema", "capsule_sha256", "capsule_bytes", "raw_file"), "wind descriptor")
    if value["schema"] != EXPORT_SCHEMA:
        a._fail("wind_export_schema", "exact owning wind descriptor required")
    d.c._digest(value["capsule_sha256"], "actual capsule identity")
    a._integer(value["capsule_bytes"], _CAP, "actual capsule bytes", 1)
    d.c._keys(value["raw_file"], _RAW, "actual raw wind record")
    _raw(path / "native", d, expected=value["raw_file"])
    capsule = _metadata_read(path / "entry/capsule.json", capsule_stat, d)
    if len(capsule) != value["capsule_bytes"] or a._sha(capsule) != value["capsule_sha256"]:
        a._fail("wind_capsule_changed", "actual original capsule differs")
    return value, capsule


def load_wind_pause(parent, runner, *, attempt, expected_export_sha256, runtime_reference):
    """Actual codec restore plus passive MASY staging; zero wind advancement."""
    d = _declare()
    d.pause._runner(runner)
    if runner._active_evidence is not None or runner._stop_receipt is not None:
        a._fail("wind_runner_busy", "actual accepted unoccupied runner required")
    path = a._attempt_path(parent, attempt)
    # Native/file/capsule counts and regular/readonly/size roles precede every
    # new parser/graph/canonical receiver. No guessed maximum-reader fit.
    descriptor_stat = _stat(path / "export.json", _CAP, d)
    capsule_stat = _stat(path / "entry/capsule.json", _CAP, d)
    _stat(path / "native" / _FILE, _WIRE_CAP, d)
    _admit(parent, runner, d, metadata_bytes=descriptor_stat.st_size + capsule_stat.st_size)
    fd = a._directory(path, d.c)
    native_token = lease = None
    try:
        descriptor, capsule_raw = _read_descriptor(path, expected_export_sha256, d,
                                                   admitted=(descriptor_stat, capsule_stat))
        facts = a._decode(capsule_raw, _CAP)
        _admit_capsule(parent, runner, facts, path, d)
        runner._precheck()
        raw_runtime, source, environment = d.pause._runtime(d.api, runtime_reference)
        if d.c.oi._json(source) != parent._source or environment != parent._environment:
            a._fail("wind_runtime", "actual C/source/environment differs")
        index, event, origin = d.dr._step_start(runner)
        resolved = d.dr._step_resolve(runner, index, origin)
        # Existing MASY stage/factory/plan ONLY, never Native begin/prologue.
        prepared = d.dr._wind_enter(runner, index, event, resolved)
        accepted = d.pause._accepted_refs(runner)
        record = {"event_index": index, "native_time": origin.hex(),
            "bed_identity": runner._accepted_bed, "private_identity": runner._accepted_private,
            "n_accepted": runner._n_accepted}
        lease = d.pause._WindLease(weakref.ref(runner), weakref.ref(threading.current_thread()),
            os.getpid(), index, event, origin, accepted, record)
        lease.prepared, lease.resolved = prepared, resolved
        lease.runtime_bytes, lease.source, lease.environment = raw_runtime, source, environment
        lease.reservation = d.pause._reserve(runner, d.api)
        lease.entry_sha256 = d.pause._entry_digest(runner, lease, d.api)
        if (d.c.oi._json(resolved) != d.c.oi._json(facts["resolved"])
                or a._metadata(_entry(runner, lease, d), _CAP, "cold entry") != a._metadata(facts["entry"], _CAP, "writer entry")):
            a._fail("wind_cold_entry", "genuine accepted-A MASY entry differs from writer")
        artifact = d.c._read(path / "native" / _FILE, _WIRE_CAP)
        if len(artifact) != descriptor["raw_file"]["size"] or a._sha(artifact) != descriptor["raw_file"]["sha256"]:
            a._fail("wind_raw_changed", "artifact bytes differ before owning restore")
        native_token = d.codec.restore_paused_continuation(artifact, limits=_limits(d))
        lease.native_token = native_token
        artifact = None
        progress = d.api.block.wind_block_progress(native_token)  # once per loaded pause
        actual = _progress(lease, progress, runner, d)
        if (type(progress.next_step_index) is not int or type(progress.n_steps) is not int
                or progress.next_step_index != facts["ordinal"]
                or not 0 < progress.next_step_index < progress.n_steps <= d.pause._MAX_STEPS
                or a._metadata(actual, _CAP, "real progress") != a._metadata(facts["progress"], _CAP, "writer progress")):
            a._fail("wind_progress", "actual restored six-scalar progress differs")
        lease.ordinal, lease.pauses, lease.progress = facts["ordinal"], facts["pauses"], actual
        lease.progress_bytes = d.c.oi._json(actual)
        lease.native_token = native_token
        lease.progress_sha256 = d.pause._progress_digest(lease, d.api)
        lease.cold_binding = _ColdBinding(_ISSUED, weakref.ref(parent), weakref.ref(runner), path,
                                         expected_export_sha256, capsule_raw)
        _check_binding(runner, lease, d)
        _raw(path / "native", d, expected=descriptor["raw_file"])
        d.c._same_directory(path, fd)
        runner._precheck()
        d.pause._runtime(d.api, None, lease)
        a._parent(parent, runner)
        token = d.pause._WindToken()
        lease.phase, lease.token = "paused", token
        if runner._active_evidence is not None or runner._stop_receipt is not None:
            a._fail("wind_runner_busy", "receiver changed before local lease publication")
        # No fallible descriptor close remains after local lease publication.
        closing, fd = fd, None
        os.close(closing)
        runner._active_evidence = lease
        native_token = None
        return token
    except BaseException as primary:
        if native_token is not None:
            try:
                d.api.block.abandon_wind_block(native_token)
            except BaseException as closing:
                # A genuine unreturned Native owner could not be retired.
                # Keep the actual private parent guard, never claim idle.
                lease.phase, lease.token, lease.native_token = "busy", None, native_token
                runner._active_evidence = lease
                primary.add_note("unreturned wind owner cleanup refused: " + type(closing).__name__[:64])
        if fd is not None:
            closing, fd = fd, None
            try:
                os.close(closing)
            except BaseException as closing_error:
                primary.add_note("wind reader descriptor close refused: " + type(closing_error).__name__[:64])
        raise


def _control_pair(left, right, cls, names, shape, d):
    if type(left) is not cls or type(right) is not cls:
        a._fail("wind_cold_controls", "exact Native controlling types required")
    for value in (left, right):
        if set(vars(value)) != set(names):
            a._fail("wind_cold_controls", "exact Native controlling fields required")
        for name in names:
            array = getattr(value, name)
            if (type(array) is not d.n.np.ndarray or array.dtype != d.n.np.dtype("float64")
                    or not array.dtype.isnative or array.dtype.metadata is not None
                    or array.dtype.fields is not None or array.dtype.subdtype is not None or array.shape != shape
                    or not array.flags.c_contiguous or not array.flags.aligned):
                a._fail("wind_cold_controls", "canonical host primitive controlling arrays required")
    if not d.c.oi._same(left, right, d.c.oi.OriginalPackLimits()):
        a._fail("wind_cold_controls", "full exact controlling values differ")


def _control_layouts(vegetation, threshold, shape, d):
    """Exactly10 bounded controlling arrays; preserve actual flags and bases.

    C-contiguous singleton-axis strides need not equal an invented canonical
    stride tuple. Native's original codec already preserves their real layout.
    """
    if (type(vegetation) is not d.vegetation or type(vegetation.raster) is not d.raster
            or type(threshold) is not d.threshold):
        a._fail("wind_cold_controls", "exact Native raster/threshold types before metadata")
    arrays = tuple(getattr(vegetation.raster, name) for name in
        ("height_m", "cover_fraction", "porosity", "drag_coefficient", "frontal_area_density"))
    arrays += tuple(getattr(threshold, name) for name in
        ("gravimetric_moisture_fraction", "clay_fraction", "physical_crust_multiplier",
         "chemical_cementation_multiplier", "biological_binding_multiplier"))
    rows = []
    for array in arrays:
        if (type(array) is not d.n.np.ndarray or array.dtype != d.n.np.dtype("float64")
                or not array.dtype.isnative or array.dtype.metadata is not None
                or array.dtype.fields is not None or array.dtype.subdtype is not None or array.shape != shape
                or not array.flags.c_contiguous or not array.flags.aligned):
            a._fail("wind_cold_controls", "primitive native float64 controls before pointer metadata")
        base = array.base
        if base is not None and type(base) is not d.n.np.ndarray:
            a._fail("wind_cold_controls", "foreign controlling backing is outside this profile")
        if base is not None and (base.dtype != d.n.np.dtype("float64") or not base.dtype.isnative
                or base.dtype.metadata is not None or base.dtype.fields is not None or base.dtype.subdtype is not None):
            a._fail("wind_cold_controls", "primitive controlling backing before pointer metadata")
        rows.append((id(array), array.dtype.str, array.shape, array.strides,
            bool(array.flags.writeable), bool(array.flags.owndata), bool(array.flags.aligned),
            bool(array.flags.c_contiguous), id(base), array.__array_interface__["data"][0]))
    return tuple(rows)


def bind_completed_result(runner, lease, result):
    """Explicit two-reference representation binding AFTER real completion only."""
    d = _declare()
    if (lease.phase != "busy" or lease.native_token is not None
            or type(result) is not d.api.block.WindBlockResult or type(result.state) is not d.state
            or type(result.state.physical) is not d.physical or result.state.coupled is not None
            or result.state.topographic_wind_cache is not None or result.state.vegetation_wind_cache is not None):
        a._fail("wind_cold_result", "actual completed supported Native result required")
    if tuple(len(dataclasses.fields(cls)) for cls in (d.physical, d.state, d.api.block.WindBlockResult)) != _COMPLETION_FIELDS:
        a._fail("wind_cold_result", "actual Source-bound completed shell layouts changed")
    p, bed, n = result.state.physical, runner._bed, d.n
    g = bed.context.geometry
    vegetation, threshold = p.vegetation, p.threshold_modifier_state
    held_vegetation, held_threshold = bed.vegetation, bed.threshold_modifier_state
    if (type(vegetation) is not d.vegetation or type(held_vegetation) is not d.vegetation
            or set(vars(vegetation)) != {"source", "raster", "objects"}
            or set(vars(held_vegetation)) != {"source", "raster", "objects"}
            or vegetation.objects is not None or held_vegetation.objects is not None
            or type(vegetation.source) is not str or vegetation.source != held_vegetation.source):
        a._fail("wind_cold_controls", "exact prescribed raster-only vegetation required")
    shape = (g.ny, g.nx)
    layouts = (_control_layouts(vegetation, threshold, shape, d),
               _control_layouts(held_vegetation, held_threshold, shape, d))
    _check_binding(runner, lease, d)
    d.validate_vegetation(vegetation, g.ny, g.nx)
    d.validate_vegetation(held_vegetation, g.ny, g.nx)
    d.validate_threshold(threshold, g.ny, g.nx)
    d.validate_threshold(held_threshold, g.ny, g.nx)
    raster_names = ("height_m", "cover_fraction", "porosity", "drag_coefficient", "frontal_area_density")
    threshold_names = ("gravimetric_moisture_fraction", "clay_fraction", "physical_crust_multiplier",
                       "chemical_cementation_multiplier", "biological_binding_multiplier")
    _control_pair(vegetation.raster, held_vegetation.raster, d.raster, raster_names, (g.ny, g.nx), d)
    _control_pair(threshold, held_threshold, d.threshold, threshold_names, (g.ny, g.nx), d)
    # Compare the complete source object as well as all declared ndarray bytes.
    if not d.c.oi._same(vegetation, held_vegetation, d.c.oi.OriginalPackLimits()):
        a._fail("wind_cold_controls", "full Native vegetation values differ")
    runner._precheck()
    _check_binding(runner, lease, d)
    if layouts != (_control_layouts(vegetation, threshold, shape, d),
                   _control_layouts(held_vegetation, held_threshold, shape, d)):
        a._fail("wind_cold_controls", "actual controlling layout/flags/backing changed")
    rebound = dataclasses.replace(p, vegetation=held_vegetation, threshold_modifier_state=held_threshold)
    for field in dataclasses.fields(d.physical):
        if field.name not in ("vegetation", "threshold_modifier_state") and getattr(rebound, field.name) is not getattr(p, field.name):
            a._fail("wind_cold_result", "noncontrolling Native reference changed")
    state = dataclasses.replace(result.state, physical=rebound)
    answer = dataclasses.replace(result, state=state)
    if type(state) is not d.state or type(answer) is not d.api.block.WindBlockResult:
        a._fail("wind_cold_result", "actual Native shell types changed")
    for old, new, replaced in ((result.state, state, "physical"), (result, answer, "state")):
        for field in dataclasses.fields(type(old)):
            if field.name != replaced and getattr(old, field.name) is not getattr(new, field.name):
                a._fail("wind_cold_result", "non-replaced Native result/state field changed")
    # Ordinary _wind_accept/finalizer/adoption follow at their original point.
    return answer


def validate_retained_reference(root, header, limits, head, generation, managed):
    """Bounded raw/capsule provenance only, never historical Native LOAD/step."""
    d = _declare()
    ref = generation["active_ref"]
    d.c._keys(ref, ("process_kind", "attempt_name", "export_sha256", "ordinal", "pauses",
                     "origin_s", "native_time_s", "plan_sha256"), "wind active reference")
    if (ref["process_kind"] != "wind" or head["kind"] != "wind_paused"
            or ref["attempt_name"] != head["generation_name"]
            or type(ref["ordinal"]) is not int or ref["ordinal"] != head["pause_ordinal"]
            or type(ref["pauses"]) is not int or ref["pauses"] != head["pause_count"]):
        d.c._fail("active_wind_reference", "exact selected wind attempt/progress required")
    a._float(ref["origin_s"])
    a._float(ref["native_time_s"])
    d.c._digest(ref["plan_sha256"], "original wind plan")
    path = managed / ref["attempt_name"]
    _value, raw = _read_descriptor(path, ref["export_sha256"], d)
    facts = a._decode(raw, _CAP)
    d.c._keys(facts, _CAPSULE, "retained wind capsule")
    d.c._keys(facts["entry"], _ENTRY, "retained original wind entry")
    d.c._keys(facts["progress"], ("native_time", "elapsed_local_s", "global_elapsed_s", "advanced_hex",
                                  "next_step_index", "n_steps"), "retained wind progress")
    a._integer(facts["ordinal"], d.pause._MAX_STEPS, "retained original wind ordinal", 1)
    a._integer(facts["pauses"], d.pause._MAX_PAUSES, "retained original wind pause count", 1)
    a._integer(facts["progress"]["next_step_index"], d.pause._MAX_STEPS, "retained native cursor", 1)
    a._integer(facts["progress"]["n_steps"], d.pause._MAX_STEPS, "retained native plan", 1)
    if (facts["progress"]["next_step_index"] != facts["ordinal"]
            or not facts["ordinal"] < facts["progress"]["n_steps"]):
        a._fail("wind_capsule_reference", "retained pause is a strict original-plan prefix")
    plan = facts["entry"]["plan"]
    if type(plan) is not list or not 1 <= len(plan) <= d.pause._MAX_STEPS:
        a._fail("wind_capsule_reference", "bounded original wind plan required")
    for duration in plan:
        if a._float(duration) <= 0:
            a._fail("wind_capsule_reference", "positive original native accounting durations required")
    if len(plan) != facts["progress"]["n_steps"]:
        a._fail("wind_capsule_reference", "original MASY/Native plan lengths differ")
    from masy.clock import encode_time, decode_time
    decode_time(facts["progress"]["native_time"])
    anchor = generation["entry_anchor_head_record"]
    expected_parent = {"store_id": header["store_id"], "token": d.c._token(anchor),
        "accepted_sequence": head["accepted_sequence"],
        "store_file_sha256": head["store_file_sha256"],
        "head_file_sha256": a._sha(d.c.oi._json(anchor)),
        "image": generation["image"], "originals": header["originals"]}
    if (facts["schema"] != SCHEMA or facts["attempt"] != a._sha(str(path).encode())
            or type(facts["ordinal"]) is not int or facts["ordinal"] != ref["ordinal"]
            or type(facts["pauses"]) is not int or facts["pauses"] != ref["pauses"]
            or facts["origin"] != a._float(ref["origin_s"]).hex()
            or d.c.oi._json(facts["progress"]["native_time"]) != d.c.oi._json(encode_time(a._float(ref["native_time_s"])))
            or a._sha(d.c.oi._json(facts["entry"]["plan"])) != ref["plan_sha256"]
            or d.c.oi._json(facts["parent"]) != d.c.oi._json(expected_parent)
            or d.c.oi._json(facts["source"]) != d.c.oi._json(header["source_identity"])
            or facts["environment"] != header["environment_sha256"]
            or facts["operation"] != generation["event_operation_id"]
            or type(facts["index"]) is not int or facts["index"] != head["accepted_sequence"]):
        a._fail("wind_capsule_reference", "exact retained source/anchor/progress/plan relation differs")
