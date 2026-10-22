"""Private DEFAULT NumPy water-entry export and cold continuation, StageA.

The existing C HEAD remains the accepted authority.  This module publishes no
active C revision or acknowledgment.  Native SYRUP alone writes/loads physical
continuation state.  Imports at module load are standard-library only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import stat
import struct
import sys
import threading
import weakref
import uuid
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCHEMA = "masy-private-water-entry/1"
EXPORT_SCHEMA = "masy-private-water-export/1"
_CAPSULE_BYTES = 1 << 20
_NATIVE_BYTES = 8 << 20
_NATIVE_JSON_BYTES = 2 << 20
_NODES, _DEPTH, _STRING, _BITS = 32768, 32, 1024, 256
_ISSUED = object()
_CAPSULE_FIELDS = frozenset(("schema", "parent", "source", "environment", "operation", "attempt",
                             "index", "resolved", "origin", "plan", "ordinal", "pauses", "entry",
                             "schedule", "native_semantic_sha256"))
_ENTRY_FIELDS = frozenset(("stage_case", "stage_label", "stage_time", "stage_nbytes", "stamp_before",
                           "inventory_before", "first", "did_commit", "rerouted", "net_displacement",
                           "graph_retained", "rain_edges", "guard_end", "intensities", "rows",
                           "lower_steps", "every", "nc", "memory_names"))


def _fail(code, detail):
    from masy.dense_runner import DenseRunnerError
    raise DenseRunnerError(code, detail)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value):
    if type(value) is not float or not math.isfinite(value):
        _fail("active_scalar", "finite original plain float required")
    return value.hex()


def _float(value):
    if type(value) is not str or len(value) > 64:
        _fail("active_scalar", "bounded canonical float hex required")
    try:
        result = float.fromhex(value)
    except (ValueError, OverflowError):
        _fail("active_scalar", "invalid original float hex")
    if not math.isfinite(result) or result.hex() != value:
        _fail("active_scalar", "noncanonical or nonfinite float hex")
    return result


def _integer(value, maximum, label, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum or value.bit_length() > _BITS:
        _fail("active_integer", label)
    return value


def _metadata(value, cap, label):
    from masy import dense_boundary_store as c
    # Existing bounded metadata admission precedes serialization/allocation.
    c._plain_metadata(value, cap, label, max_nodes=_NODES, max_depth=_DEPTH)
    raw = c.oi._json(value)
    if len(raw) > cap:
        _fail("active_metadata", label)
    return raw


def _decode(raw, cap):
    from masy import dense_boundary_store as c
    view = SimpleNamespace(max_metadata_bytes=cap, max_nodes=_NODES, max_depth=_DEPTH)
    value = c.oi._strict_json(raw, view)
    if _metadata(value, cap, "active decoded metadata") != raw:
        _fail("active_metadata", "canonical whole bytes required")
    return value


def _declare():
    """Declare the actual owning APIs before the original runtime binding."""
    from masy import dense_water_pause as pause
    from maple.core import backend
    from maple.io.outputs import snapshot
    from maple.aeolian.scheduler import physical_checkpoint
    n, checkpoint, schemas, c = pause._declare()
    # These are the real loaded modules, not injected backend/provider objects.
    return n, checkpoint, schemas, c, backend, snapshot, physical_checkpoint


def _parent(parent, runner):
    from masy import dense_boundary_store as c
    if type(parent) is not c._WaterParent:
        _fail("active_parent", "actual locally entered C parent borrow required")
    parent.check(runner)
    return parent


def _parent_record(parent, runner):
    from masy import dense_boundary_store as c
    _parent(parent, runner)
    h = parent._handle
    _record, head_raw = h._head()
    return {"store_id": h._header["store_id"], "token": parent._token,
            "accepted_sequence": h._state.sequence, "store_file_sha256": c._sha(h._store_raw),
            "head_file_sha256": c._sha(head_raw), "image": _decode(parent._image, 128 << 10),
            "originals": h._header["originals"]}


def _operation(parent, runner):
    from masy import dense_boundary_store as c
    return c._operation(parent._handle._header["store_id"], _decode(parent._head, 32 << 10),
                        runner._next, runner._schedule.digest)


def _array_hash(array, np):
    if (type(array) is not np.ndarray or array.dtype.hasobject or array.dtype.fields
            or array.dtype.kind not in "fiub" or not array.dtype.isnative
            or not array.flags.c_contiguous or array.ndim > 4):
        _fail("active_array", "closed native contiguous numeric array required")
    digest = hashlib.sha256()
    if array.nbytes:
        with memoryview(array) as view, view.cast("B") as raw:
            for start in range(0, raw.nbytes, 65536):
                digest.update(raw[start:start + 65536])
    return {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": digest.hexdigest()}


def _native_projection(item, n, checkpoint, backend, schemas, bounds):
    """Compare Native wire values, not historical ids/flags or array labels.

    Admission is done by the caller. The original bounded scanner closes all
    numeric leaves before Native WRITE traversal. The Native encoder's array
    labels are replaced with raw descriptors so JSON key sorting/occurrence
    normalization cannot make a different semantic identity after LOAD.
    """
    from masy import dense_water_pause as pause
    pause._scan(item, n, schemas, bounds)
    primary = checkpoint._primary(item)
    arrays = {}
    encoded = checkpoint._encode_wire(item, primary, arrays, checkpoint._registry(),
                                      backend.is_array, backend.to_host)
    cache = {}

    def array(value):
        identity = id(value)
        if identity not in cache:
            cache[identity] = _array_hash(value, n.np)
        return cache[identity]

    def replace(value):
        if type(value) is dict:
            if set(value) == {"array", "dtype", "shape"}:
                return {"native_array": array(arrays[value["array"]])}
            return {key: replace(child) for key, child in value.items()}
        if type(value) is list:
            return [replace(child) for child in value]
        return value

    # Native wire represents primary sediment through its MAPLE snapshot.
    # Charge/hash every physical field separately; no surface reconstruction.
    bed = {}
    for root in ("voxel_column", "active_layer", "water", "ledger", "committed_topography",
                 "sediment_availability"):
        obj = getattr(primary.bed, root)
        values = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            if type(value) is n.np.ndarray:
                values[field.name] = {"native_array": array(value)}
            else:
                if isinstance(value, n.np.generic):
                    value = value.item()  # precisely Native wire's scalar normalization
                values[field.name] = value
        bed[root] = values
    return _sha(_metadata({"native": replace(encoded), "primary_bed": bed},
                          _NATIVE_JSON_BYTES, "Native semantic projection"))


@dataclasses.dataclass(slots=True)
class _ColdPrepared:
    key: Any
    parent: Any
    owner: Any
    capsule_sha256: str
    facts: Any
    n: Any
    plan: Any
    schedule: Any
    inventory_before: Any
    stage: Any
    cover: Any


def _check_prepared(runner, prepared):
    from masy import dense_runner as dr
    if (type(runner) is not dr.DenseRunner or type(prepared) is not _ColdPrepared
            or prepared.key is not _ISSUED or prepared.owner() is not runner
            or type(prepared.stage) is not dr._ColdRainStage
            or prepared.stage._capsule != prepared.capsule_sha256):
        _fail("active_entry_owner", "exact locally issued cold entry required")
    parent = prepared.parent()
    _parent(parent, runner)
    if type(prepared.plan) is not tuple or not 1 <= len(prepared.plan) <= runner._limits.max_report_rows:
        _fail("active_plan", "original bounded cold plan required")
    # Every resumed cold scan/Native call retains these extra original-entry
    # owners. Re-admit them in addition to the unchanged warm reservation.
    _reserve_active(runner, prepared.plan)
    if _sha(_metadata(prepared.facts, _CAPSULE_BYTES, "cold entry facts")) != prepared.capsule_sha256:
        _fail("active_entry_changed", "original capsule facts changed")
    if prepared.cover is not runner._water.vegetation_cover_fraction:
        _fail("active_cover_changed", "original cover identity changed")
    return parent


def _stage_value(runner, stage, field):
    from masy import dense_runner as dr
    if (type(stage) is not dr._ColdRainStage or stage._owner() is not runner
            or runner._active_evidence is None):
        _fail("active_stage_owner", "locally issued cold stage facts required")
    _parent(stage._parent(), runner)
    bed = runner._bed
    if (bed.case_sha256 != stage._case or bed.identity_sha256() != stage._label
            or bed.time_s.hex() != stage._time.hex()
            or runner._ownership().stage_bytes() != stage._nbytes
            or float(bed.state.committed_topography.last_commit_time_s).hex() != stage._stamp.hex()):
        _fail("active_stage_changed", "accepted parent no longer matches original entry stage")
    if field == "stamp":
        return stage._stamp
    if field == "label":
        return stage._label
    _fail("active_stage_field", "closed cold stage accessor")


def _entry_value(runner, prepared, field):
    _check_prepared(runner, prepared)
    facts = prepared.facts["entry"]
    if field == "terrain":
        # Parent B preparation already validates original routing/source/terrain
        # identity. Native603 owns its decoded terrain comparison/substitution.
        if runner._routing is not None:
            from masy import dense_runner_checkpoint as b
            b._routing_record(runner._routing, runner._water,
                              (runner._bed.context.geometry.ny, runner._bed.context.geometry.nx),
                              runner._n_classes, prepared.parent()._handle._limits.image)
        return runner._water.terrain
    if field == "rerouted":
        return facts["rerouted"]
    if field == "net_displacement_m":
        value = facts["net_displacement"]
        return None if value is None else _float(value)
    if field == "graph_retained":
        return facts["graph_retained"]
    _fail("active_entry_field", "closed cold entry accessor")


def _native_fields(runner, prepared, continuation):
    _check_prepared(runner, prepared)
    if continuation is None:
        _fail("active_fresh_entry", "cold continuation cannot invoke fresh native entry")
    e = prepared.facts["entry"]
    return (prepared.n, runner._bed.context, runner._water, prepared.schedule,
            runner._control.sediment_control, runner._control.completion_policy,
            _float(e["every"]), runner._limits, _float(e["guard_end"]))


def _accept_fields(runner, prepared):
    _check_prepared(runner, prepared)
    n, e = prepared.n, prepared.facts["entry"]
    return (n, n.np, runner._bed, runner._bed.context, runner._array_ops(n.np), runner._ownership(),
            runner._control.sediment_control, runner._control.completion_policy, _float(e["every"]),
            runner._water, runner._limits, _float(prepared.facts["origin"]),
            [_float(x) for x in e["rain_edges"]], _float(e["guard_end"]),
            [_float(x) for x in e["intensities"]], e["rows"], e["lower_steps"], e["nc"],
            prepared.stage, prepared.inventory_before, e["first"], runner._bed.context.geometry,
            prepared.schedule)


def _cold_entry_seal(runner, prepared, n, schemas, bounds, resolved):
    from masy import dense_water_pause as pause
    _check_prepared(runner, prepared)
    stage = prepared.stage
    facts = (prepared.facts, prepared.plan, resolved, stage._case, stage._label, stage._nbytes, stage._time,
             stage._stamp, prepared.inventory_before, prepared.schedule, prepared.cover,
             runner._control.sediment_control, runner._control.completion_policy)
    if prepared.n is not n:
        # _declare creates a fresh namespace; compare actual bound module/class
        # objects, not a transient SimpleNamespace identity.
        if prepared.n.np is not n.np or prepared.n.EventProgress is not n.EventProgress:
            _fail("active_native_changed", "original native classes/backend changed")
    return pause._scan(facts, n, schemas, bounds)["sha256"]


def _schedule_record(schedule):
    return {"edges": [_hex(float(x)) for x in schedule.edges_s],
            "intensity": [_hex(float(x)) for x in schedule.intensity_mm_per_h],
            "rate": [_hex(float(x)) for x in schedule.rate_m_per_s],
            "provenance": {f.name: getattr(schedule.provenance, f.name)
                           for f in dataclasses.fields(schedule.provenance)}}


def _capture(parent, runner, lease, semantic, attempt):
    p, n = lease.prepared, lease.prepared.n
    from masy import dense_runner as dr
    if type(p) is not dr._RainEntry:
        _fail("active_export_entry", "first StageA export requires genuine original warm entry")
    # Export of a loaded cold pause is intentionally a later protocol: do not
    # infer/reconstruct a missing original DenseStage/NativeBedCommit.
    stage = p.stage
    entry = {"stage_case": stage.case_sha256, "stage_label": stage.identity_sha256,
             "stage_time": _hex(stage.time_s), "stage_nbytes": stage.nbytes,
             "stamp_before": _hex(float(stage.state.committed_topography.last_commit_time_s)),
             "inventory_before": [_hex(float(x)) for x in p.inventory_before],
             "first": p.first, "did_commit": p.boundary.did_commit, "rerouted": p.boundary.rerouted,
             "net_displacement": (None if p.boundary.net_displacement_m is None
                                  else _hex(float(p.boundary.net_displacement_m))),
             "graph_retained": p.graph is p.held.graph,
             "rain_edges": [_hex(float(x)) for x in p.rain_edges], "guard_end": _hex(p.guard_end),
             "intensities": [_hex(float(x)) for x in p.intensities], "rows": p.rows,
             "lower_steps": p.lower_steps, "every": _hex(p.every), "nc": p.nc,
             "memory_names": list(p.memory_names)}
    return {"schema": SCHEMA, "parent": _parent_record(parent, runner),
            "source": lease.source, "environment": lease.environment,
            "operation": _operation(parent, runner), "attempt": attempt,
            "index": lease.index, "resolved": lease.resolved, "origin": _hex(lease.origin),
            "plan": [_hex(x) for x in lease.plan], "ordinal": lease.ordinal, "pauses": lease.pauses,
            "entry": entry, "schedule": _schedule_record(p.schedule),
            "native_semantic_sha256": semantic}


def _admit_capsule(parent, runner, capsule, n, c):
    from masy import dense_water_pause as pause
    from masy.dense_schedule import RainEvent
    c._keys(capsule, _CAPSULE_FIELDS, "water entry capsule")
    e = capsule["entry"]
    c._keys(e, _ENTRY_FIELDS, "water original entry")
    if (capsule["schema"] != SCHEMA or type(capsule["schema"]) is not str
            or c.oi._json(capsule["parent"]) != c.oi._json(_parent_record(parent, runner))
            or capsule["operation"] != _operation(parent, runner)
            or capsule["environment"] != parent._environment
            or c.oi._json(capsule["source"]) != parent._source):
        _fail("active_parent_changed", "capsule does not bind actual current accepted C parent")
    c._digest(capsule["native_semantic_sha256"], "Native semantic digest")
    c._digest(capsule["attempt"], "private attempt identity")
    if (type(runner._next) is not int or not 0 <= runner._next < len(runner._schedule.events)
            or capsule["index"] != runner._next or type(capsule["index"]) is not int
            or type(runner._schedule.events[runner._next]) is not RainEvent
            or _float(capsule["origin"]).hex() != runner._bed.time_s.hex()):
        _fail("active_event", "original next rain/origin required")
    resolved, plan = pause._plan(runner, n)  # validation only; no step/entry/replay
    if (c.oi._json(capsule["resolved"]) != c.oi._json(resolved)
            or type(capsule["plan"]) is not list or len(capsule["plan"]) != len(plan)
            or tuple(_float(x).hex() for x in capsule["plan"]) != tuple(x.hex() for x in plan)):
        _fail("active_plan", "original complete event plan differs")
    pause._ordinal(capsule["ordinal"], plan)
    _integer(capsule["pauses"], pause._MAX_PAUSES, "total pause count", 1)
    for key in ("first", "did_commit", "rerouted", "graph_retained"):
        if type(e[key]) is not bool:
            _fail("active_entry", "original exact entry relation bool required")
    if e["did_commit"] is not True or e["first"] != (runner._routing is None):
        _fail("active_entry", "original first/accepted entry relation differs")
    if e["net_displacement"] is not None:
        _float(e["net_displacement"])
    bed, own = runner._bed, runner._ownership()
    if (e["stage_case"] != bed.case_sha256 or e["stage_label"] != bed.identity_sha256()
            or _float(e["stage_time"]).hex() != bed.time_s.hex()
            or type(e["stage_nbytes"]) is not int or e["stage_nbytes"] != own.stage_bytes()
            or _float(e["stamp_before"]).hex() != float(bed.state.committed_topography.last_commit_time_s).hex()
            or type(e["nc"]) is not int or e["nc"] != runner._n_classes):
        _fail("active_stage", "original stage facts differ from actual accepted parent")
    expected_inventory = runner._array_ops(n.np).inventory(n, bed.state)
    if (type(e["inventory_before"]) is not list or len(e["inventory_before"]) != runner._n_classes
            or [_float(x).hex() for x in e["inventory_before"]] != [float(x).hex() for x in expected_inventory]):
        _fail("active_inventory", "pre-entry inventory differs from accepted physical inventory")
    event = runner._schedule.events[runner._next]
    from masy.clock import decode_time
    edges = [decode_time(row["native"]) for row in resolved["resolved_edges"]["edges"]]
    expected_edges = [x.hex() for x in edges[:-1]]
    if (type(e["rain_edges"]) is not list or e["rain_edges"] != expected_edges
            or e["intensities"] != [_hex(float(i.intensity_mm_h)) for i in event.intervals]
            or _float(e["guard_end"]).hex() != edges[-1].hex()
            or _float(e["every"]).hex() != runner._control.report_every_s.hex()):
        _fail("active_forcing", "complete original forcing/report controls differ")
    sc, policy = runner._control.sediment_control, runner._control.completion_policy
    need = (Fraction(edges[-2]) - Fraction(bed.time_s) + Fraction(policy.hold_s)) / Fraction(sc.storm.max_dt_s)
    lower = -(-need.numerator // need.denominator)
    span = (Fraction(edges[-1]) - Fraction(bed.time_s)) / Fraction(runner._control.report_every_s)
    rows = span.numerator // span.denominator + len(edges) - 1 + 2
    if type(e["rows"]) is not int or e["rows"] != rows or type(e["lower_steps"]) is not int or e["lower_steps"] != lower:
        _fail("active_entry_budget", "original engineering row/step preflight differs")
    names = () if runner._routing is None else own.routing_memory(
        n.np, runner._routing, runner._water, (bed.context.geometry.ny, bed.context.geometry.nx), runner._n_classes)[0]
    if type(e["memory_names"]) is not list or e["memory_names"] != list(names):
        _fail("active_entry_memory", "original retained routing roles differ")
    schedule = capsule["schedule"]
    c._keys(schedule, {"edges", "intensity", "rate", "provenance"}, "original rainfall schedule")
    if (schedule["edges"] != expected_edges or schedule["intensity"] != e["intensities"]
            or type(schedule["rate"]) is not list or len(schedule["rate"]) != len(event.intervals)
            or type(schedule["provenance"]) is not dict
            or c.oi._json(schedule["provenance"]) != c.oi._json({
                f.name: getattr(n.RainfallProvenance(kind="constant"), f.name)
                for f in dataclasses.fields(n.RainfallProvenance)})):
        _fail("active_schedule", "complete original schedule/provenance differs")
    return resolved, plan


def _attempt_path(parent, attempt):
    from masy import dense_boundary_store as c
    if type(attempt) not in (str, type(Path())):
        _fail("active_path", "plain owned attempt path required")
    path = Path(attempt).absolute()
    if (len(str(path)) > 4096 or str(path) != os.path.realpath(path)
            or path == parent._handle._root or parent._handle._root in path.parents):
        _fail("active_path", "private attempt must be canonical and outside existing C store")
    directory = c.oi._open_dir(path.parent)
    os.close(directory)
    return path


def _directory(path, c):
    fd = c.oi._open_dir(path)
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or info.st_mode & 0o022:
        os.close(fd)
        _fail("active_directory", "owned private directory required")
    return fd


def _raw_files(path, checkpoint, c, *, expected=None):
    """Stream through the unchanged owning reader; no payload materialization."""
    fd = _directory(path, c)
    try:
        names = set()
        with os.scandir(path) as entries:
            for entry in entries:
                if len(names) == 3 or entry.name not in checkpoint._CHILD_NAMES:
                    _fail("active_inventory", "exact three native child files required")
                names.add(entry.name)
        if names != set(checkpoint._CHILD_NAMES):
            _fail("active_inventory", "missing Native child")
        directory = os.fstat(fd)
        rows = []
        if expected is not None and (type(expected) is not list or len(expected) != 3):
            _fail("active_raw", "exact three admitted custody records required")
        for index, name in enumerate(checkpoint._CHILD_NAMES):
            target = path / name
            info = target.lstat()
            cap = _NATIVE_JSON_BYTES if name == "checkpoint.json" else _NATIVE_BYTES + (1 << 20)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid()
                    or info.st_mode & 0o222 or not 1 <= info.st_size <= cap):
                _fail("active_raw", "readonly owned single-link bounded Native child required")
            digest = c.runtime._hash_path(target, budget=cap)[0]
            row = checkpoint.CheckpointRawFile(name, info.st_size, digest,
                info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns, info.st_mode,
                directory.st_dev, directory.st_ino, directory.st_mtime_ns, directory.st_ctime_ns)
            checkpoint._child_file_stat(target, row)
            if expected is not None and dataclasses.asdict(row) != expected[index]:
                _fail("active_raw_changed", "actual raw child differs from admitted writer custody")
            rows.append(row)
        checkpoint._child_inventory(path, tuple(rows))
        c._same_directory(path, fd)
        return tuple(rows)
    finally:
        os.close(fd)


def _p1_limits(checkpoint):
    # Narrower caller IO limits, never a change to Native ceilings or physics.
    return checkpoint.CheckpointWriteLimits(max_uncompressed_bytes=_NATIVE_BYTES,
                                           max_json_bytes=_NATIVE_JSON_BYTES)


def _space(parent, path, layout, capsule_bytes, c):
    """Admit one exclusive attempt and preserve the existing C free reserve.

    This accounts for the sibling artifact, rather than charging the old C
    store's publication reservation or pretending it is an active revision.
    Earlier attempts remain retained and consume actual filesystem space.
    """
    needed = layout.native_child_bytes + capsule_bytes + _CAPSULE_BYTES
    limits = parent._handle._limits
    if needed > limits.max_store_bytes:
        _fail("active_disk_cap", "one complete sibling attempt exceeds existing storage cap")
    descriptor = c.oi._open_dir(path.parent)
    try:
        available = os.fstatvfs(descriptor)
        if available.f_bavail * available.f_frsize < needed + limits.free_reserve_bytes:
            _fail("active_disk_space", "complete Native child/capsule/descriptor plus fixed reserve required")
        if available.f_favail < 8 + limits.free_reserve_inodes:
            _fail("active_disk_inodes", "three directories and five files plus fixed reserve required")
        c._same_directory(path.parent, descriptor)
    finally:
        os.close(descriptor)
    return needed


def _native_identity(capsule, capsule_sha):
    return {"schema": SCHEMA, "source": capsule["source"], "environment": capsule["environment"],
            "parent": capsule["parent"], "operation": capsule["operation"],
            "attempt": capsule["attempt"], "capsule_sha256": capsule_sha}


def _header_admission(path, layout, native, c, *, bounds):
    """Actual Native header parser, then bounded context-derived admission."""
    if type(layout) is not dict or type(layout.get("archives")) is not list or len(layout["archives"]) != 2:
        _fail("active_layout", "actual writer's two-archive layout metadata required")
    used = 0
    for index, name in enumerate(("maple_state.npz", "continuation.npz")):
        observed, count = native._audit_archive(path / name, _NATIVE_BYTES - used)
        used += count
        archive = layout["archives"][index]
        if (type(archive) is not dict or archive.get("filename") != name
                or type(archive.get("members")) is not list or not 1 <= len(archive["members"]) <= 256
                or tuple(observed) != tuple(row["key"] for row in archive["members"])):
            _fail("active_layout", "actual archive member inventory differs from writer P1")
        for row in archive["members"]:
            if observed[row["key"]] != {"dtype": row["dtype"], "shape": row["shape"], "npy_bytes": row["npy_bytes"]}:
                _fail("active_layout", "actual native archive header differs from writer P1")
            # The native parser has already admitted dtype/header size/raw bytes.
            # The selected companion's numeric shape fits the unchanged pause
            # scanner envelope BEFORE any np.load. Snapshot scalar metadata has
            # its separate finite Native JSON/uncompressed limits.
            if index == 1:
                shape = row["shape"]
                if len(shape) > 4 or row["data_bytes"] > bounds["array_cap"]:
                    _fail("active_layout", "numeric member exceeds original scanner envelope")
        if count != archive["npy_bytes"]:
            _fail("active_layout", "actual uncompressed archive differs from writer P1")
    if used != layout.get("uncompressed_payload_bytes"):
        _fail("active_layout", "summed Native uncompressed identity differs")
    raw = c._read(path / "checkpoint.json", _NATIVE_JSON_BYTES)
    view = SimpleNamespace(max_metadata_bytes=_NATIVE_JSON_BYTES, max_nodes=_NODES, max_depth=_DEPTH)
    data = c.oi._strict_json(raw, view)
    c._plain_metadata(data, _NATIVE_JSON_BYTES, "native metadata preadmission", max_nodes=_NODES, max_depth=_DEPTH)
    states = 0

    def count_states(value):
        nonlocal states
        if type(value) is dict:
            if len(value) == 4 and set(value) == {"state", "graph_digest", "graph_binding", "routing_source"}:
                states += 1
            for child in value.values():
                count_states(child)
        elif type(value) is list:
            for child in value:
                count_states(child)
    count_states(data)
    # This is a preallocation restriction of the already required single wet
    # Native603 anchor profile, not another model decoder or physical validator.
    if states != 1:
        _fail("active_native_states", "exactly one dense wet state before Native LOAD")
    return data


def _same_control(value, expected, primary, checkpoint, backend, c):
    left, right = {}, {}
    a = checkpoint._encode_wire(value, primary, left, checkpoint._registry(), backend.is_array, backend.to_host)
    b = checkpoint._encode_wire(expected, primary, right, checkpoint._registry(), backend.is_array, backend.to_host)
    if left or right or c.oi._json(a) != c.oi._json(b):
        _fail("active_controls", "complete original Native control/policy bits differ")


def _mint(parent, runner, capsule, capsule_sha, item, runtime_reference, declared):
    from masy import dense_runner as dr, dense_water_pause as pause
    n, checkpoint, schemas, c, backend, _snapshot, _native = declared
    resolved, plan = _admit_capsule(parent, runner, capsule, n, c)
    if runner._active_evidence is not None or runner._stop_receipt is not None:
        _fail("active_runner_busy", "cold mint requires actual unoccupied accepted runner")
    if (type(item) is not n.EventProgress or item.status != "wet" or item.quiet.ready is not False
            or float(item.result.state.t_s).hex() != plan[capsule["ordinal"] - 1].hex()
            or type(item.origin_t_s) is not float
            or item.origin_t_s.hex() != _float(capsule["origin"]).hex()
            or item.report_every_s != runner._control.report_every_s
            or item.max_report_rows != runner._limits.max_report_rows):
        _fail("active_progress", "actual loaded wet original boundary required")
    _same_control(item.control, runner._control.sediment_control, item.result.state, checkpoint, backend, c)
    _same_control(item.policy, runner._control.completion_policy, item.result.state, checkpoint, backend, c)
    e, saved = capsule["entry"], capsule["schedule"]
    schedule = n.RainfallSchedule(edges_s=[_float(x) for x in saved["edges"]],
                                  intensity_mm_per_h=[_float(x) for x in saved["intensity"]],
                                  provenance=n.RainfallProvenance(**saved["provenance"]))
    if c.oi._json(_schedule_record(schedule)) != c.oi._json(saved):
        _fail("active_schedule", "passive Native constructor did not reproduce all original bits")
    stage = object.__new__(dr._ColdRainStage)
    stage._parent, stage._owner = weakref.ref(parent), weakref.ref(runner)
    stage._label, stage._case, stage._nbytes = e["stage_label"], e["stage_case"], e["stage_nbytes"]
    stage._time, stage._stamp, stage._capsule = _float(e["stage_time"]), _float(e["stamp_before"]), capsule_sha
    inventory = n.np.array([_float(x) for x in e["inventory_before"]], dtype=n.np.float64)
    inventory.flags.writeable = False
    prepared = _ColdPrepared(_ISSUED, weakref.ref(parent), weakref.ref(runner), capsule_sha, capsule,
                              n, plan, schedule, inventory, stage, runner._water.vegetation_cover_fraction)
    raw, source, environment = pause._runtime(c, runtime_reference)
    if c.oi._json(source) != parent._source or environment != parent._environment:
        _fail("active_runtime", "actual runtime differs from borrowed accepted parent")
    lease = pause._WaterLease(weakref.ref(runner), weakref.ref(threading.current_thread()), os.getpid(),
                              runner._next, runner._schedule.events[runner._next], _float(capsule["origin"]))
    lease.prepared, lease.plan, lease.resolved = prepared, plan, resolved
    lease.runtime_bytes, lease.source, lease.environment = raw, source, environment
    lease.ordinal, lease.pauses, lease.progress = capsule["ordinal"], capsule["pauses"], item
    lease.reservation = _reserve_active(runner, plan)
    proof = pause._wet(runner, lease, item, n, checkpoint, schemas)
    lease.entry_sha256 = _cold_entry_seal(runner, prepared, n, schemas, pause._bounds(runner, plan), resolved)
    lease.progress_sha256 = proof["sha256"]
    token = pause._WaterToken()
    lease.phase, lease.token = "paused", token
    runner._precheck()
    pause._runtime(c, None, lease)
    _parent(parent, runner)
    if runner._active_evidence is not None or runner._stop_receipt is not None:
        _fail("active_runner_busy", "cold mint receiver changed before final publication")
    # All fallible parsing/validation/copying is complete. No fresh DenseStage,
    # Native entry, BedCommit, solver step, reset or C publication was invoked.
    runner._active_evidence = lease
    return token


def _reserve_active(runner, plan, *, phase="resume", layout=None, raw_metadata_bytes=0):
    """Additive Source-visible role admission, not total heap/JIT fit.

    Native solver/validator scratch remains under the unchanged qualified
    pause reservation. New archive/decode/factory/projection owners are added;
    that inherited reservation is never replaced by an IO byte limit.
    """
    from masy import dense_water_pause as pause
    inherited = pause._reserve(runner, plan)
    g = runner._bed.context.geometry
    n, ring, k = g.ny * g.nx, (g.ny + 2) * (g.nx + 2), runner._n_classes
    if phase not in ("resume", "preflight", "writer", "reader"):
        _fail("active_resource_phase", "closed Source resource phase required")
    pointer = struct.calcsize("P")
    # Closed capsule constructor roles: plan, three schedule vectors, original
    # interval/edge metadata, inventory and fixed parent/stage/source records.
    # Every float hex has <=64 code points, admitted before constructing it.
    values = len(plan) + 4 * len(runner._schedule.events[runner._next].intervals) + k
    scalar = sys.getsizeof("") + 4 * 64 + pointer
    capsule_values = values * scalar + 256 * (sys.getsizeof({}) + 16 * pointer)
    parent_metadata = (32 + 128 + 128) << 10
    # Raw parser input is admitted by lstat before reading/parsing. At most one
    # JSON value per input byte, each bounded built-in container/scalar role,
    # plus maximal 4-byte Unicode and simultaneous original raw bytes. This
    # covers rejection before the existing post-parse node/bit/depth checks;
    # it is not an allocator fragmentation or native runtime bound.
    _integer(raw_metadata_bytes, 2 * _CAPSULE_BYTES + _NATIVE_JSON_BYTES, "metadata raw admission")
    json_node = sys.getsizeof({}) + sys.getsizeof([]) + sys.getsizeof(0.) + 8 * pointer
    parsed_metadata = raw_metadata_bytes * (json_node + 5)
    roles = {"cold_capsule_values": capsule_values, "parent_control_metadata": parent_metadata,
             "preadmitted_raw_parse_and_value_roles": parsed_metadata,
             # Actual runtime._hash_path and Native sha256_file both stream
             # 1MiB; old/new blocks coexist at assignment. Array hashing below
             # borrows <=64KiB views, not additional copied payload buffers.
             "hash_buffers": 2 << 20, "schedule_constructor_vectors": 8 * (3 * len(plan) + k)}
    if phase in ("preflight", "writer", "reader"):
        # Metadata-only Native P1 and semantic projection are admitted before
        # their traversals. The original finite scanner/table law is inherited
        # unchanged; its temporary walk has ended before Native P1 starts.
        roles["native_layout_and_projection_metadata"] = pause._metadata_bytes(len(plan))
    if phase == "writer":
        if layout is None:
            _fail("active_resource_layout", "actual Native P1 required before SAVE")
        roles["native_serialization_increment"] = layout.additional_serialization_upper_bytes
    if phase == "reader":
        # Total two-archive U bounds occurrence payloads, not factories. The
        # archives' arrays + readonly decoder copies and retained snapshot/
        # terrain validation generations coexist. Do not substitute a 2T-only
        # claim: malformed terrain nodes share the complete admitted U.
        roles["snapshot_companion_copy_and_terrain_occurrences"] = 4 * _NATIVE_BYTES
        roles["snapshot_repeated_active_reads"] = 3 * 8 * n * k
    if phase in ("writer", "reader"):
        # Actual fixed D4 factory families, including all four donor rounds,
        # retained outputs and old/new expression temporaries. Dimensions are
        # ring, N, and (4,N); no NxN/volume/history allocation is inferred.
        # NumPy to_host/to_device/freeze themselves return the SAME array.
        graph_rows = {
            "terrain_ring_copy_and_grid_normalization": 4 * 8 * ring + ring + 25 * n,
            "ring_and_cell_predicates": 4 * ring + 32 * n,
            "zmin_and_four_D4_select_generations": 6 * 8 * n,
            "slopes_and_conveyance_arithmetic": 12 * 8 * n,
            "coordinates_steps_receivers_and_gathers": 16 * 8 * n,
            "four_donor_rounds_and_flat_output": (4 * 8 + 4 * (2 * 8 + 7 + 4 * 8 + 3 * 8 + 5 + 8)) * n,
            "level_sort_counts_bounds_and_positions": 16 * 8 * (n + 1),
            "donor_gather_clip_position_and_masks": 5 * 4 * 8 * n,
            "transport_network_indexes_faces_masks_copies": 32 * 8 * n,
            "physics_grid_copies_predicates": 6 * 8 * n,
            "level_and_outlet_python_records": (n + 1) * (
                sys.getsizeof(()) + 2 * pointer + 2 * (sys.getsizeof(0) + pointer) + pointer
                + sys.getsizeof(0) + pointer),
        }
        # SAVE checks a rebuild against its retained warm graph; LOAD retains
        # its decoded graph and validator/error frames. Charge both roles in
        # either phase, without alias credit or concurrent solver execution.
        roles["native_retained_and_rebuilt_graph_factory_roles"] = 2 * sum(graph_rows.values())
    required = inherited["required_bytes"] + sum(roles.values())
    runner._work_bytes_check(required)
    return {"phase": phase, "inherited_pause": inherited, "additional_roles": roles,
            "required_bytes": required, "heap_peak_or_fit_proven": False,
            "opaque_runtime": "independent unchanged whole-worker AS/CPU/wall/headroom gate required"}


def _normalize_metadata(value):
    # Actual Native P1 dataclass metadata only; no model dataclass/array enters.
    if type(value) is tuple:
        return [_normalize_metadata(x) for x in value]
    if type(value) is list:
        return [_normalize_metadata(x) for x in value]
    if type(value) is dict:
        return {key: _normalize_metadata(x) for key, x in value.items()}
    if value is None or type(value) in (str, bool, int, float):
        return value
    _fail("active_metadata_type", "only closed Native descriptor metadata")


def _export_check(parent, runner, lease, token, declared):
    from masy import dense_water_pause as pause
    n, checkpoint, schemas, c, _backend, _snapshot, _native = declared
    if (runner._active_evidence is not lease or lease.owner() is not runner
            or lease.pid != os.getpid() or lease.thread() is not threading.current_thread()
            or lease.token is not token or lease.phase not in ("paused", "exporting")):
        _fail("active_export_lease", "original local paused export owner changed")
    _parent(parent, runner)
    runner._precheck()
    pause._runtime(c, None, lease)
    bounds = pause._bounds(runner, lease.plan)
    if pause._entry_digest(runner, lease.prepared, n, schemas, bounds, lease.resolved) != lease.entry_sha256:
        _fail("active_export_entry", "original warm entry changed")
    if pause._wet(runner, lease, lease.progress, n, checkpoint, schemas)["sha256"] != lease.progress_sha256:
        _fail("active_export_progress", "original warm progress changed")


def export_water_pause(parent, runner, token, *, attempt):
    """Non-consuming warm export; returns only the external metadata SHA.

    Caller retains this SHA locally for a separately gated fresh process. The
    original C HEAD is unchanged. This is not a durable active acknowledgment.
    """
    from masy import dense_runner as dr, dense_water_pause as pause
    lease = pause._lease(runner, token)
    if type(lease.prepared) is not dr._RainEntry:
        _fail("active_export_entry", "only the genuine original warm entry can export in StageA")
    declared = _declare()
    n, checkpoint, schemas, c, backend, _snapshot, _native = declared
    pause._profile(runner, n)
    _parent(parent, runner)
    _reserve_active(runner, lease.plan, phase="preflight")
    _export_check(parent, runner, lease, token, declared)
    path = _attempt_path(parent, attempt)
    if path.exists() or path.is_symlink():
        _fail("active_attempt_exists", "new exclusive attempt required; old versions retained")
    limits = _p1_limits(checkpoint)
    # This real, metadata-only descriptor admits the projection before its
    # allocation. Its identity contains real parent/operation only; it is NOT
    # a SAVE identity or fabricated capsule hash. Final SAVE has its own P1.
    preliminary = {"parent": _parent_record(parent, runner), "operation": _operation(parent, runner)}
    projection_layout = checkpoint.checkpoint_write_budget(lease.progress, runner._bed.context,
        runner._water.column, runner._water.sediment, preliminary, limits=limits)
    _reserve_active(runner, lease.plan, phase="writer", layout=projection_layout)
    semantic = _native_projection(lease.progress, n, checkpoint, backend, schemas, pause._bounds(runner, lease.plan))
    capsule = _capture(parent, runner, lease, semantic, _sha(uuid.uuid4().bytes))
    capsule_raw = _metadata(capsule, _CAPSULE_BYTES, "original rain entry capsule")
    capsule_sha = _sha(capsule_raw)
    identity = _native_identity(capsule, capsule_sha)
    layout = checkpoint.checkpoint_write_budget(lease.progress, runner._bed.context,
        runner._water.column, runner._water.sediment, identity, limits=limits)
    _reserve_active(runner, lease.plan, phase="writer", layout=layout)
    _space(parent, path, layout, len(capsule_raw), c)
    _export_check(parent, runner, lease, token, declared)
    pause._lease(runner, token)  # final non-consuming check before exclusive IO
    lease.phase = "exporting"
    parentfd = rootfd = None
    try:
        parentfd = c.oi._open_dir(path.parent)
        os.mkdir(path.name, 0o700, dir_fd=parentfd)
        os.fsync(parentfd)
        rootfd = _directory(path, c)
        os.mkdir("entry", 0o700, dir_fd=rootfd)
        c._write(path / "entry/capsule.json", capsule_raw)
        os.chmod(path / "entry/capsule.json", 0o444, follow_symlinks=False)
        checkpoint.save_checkpoint(path / "native", lease.progress, runner._bed.context,
                                   runner._water.column, runner._water.sediment, identity)
        for name in checkpoint._CHILD_NAMES:
            os.chmod(path / "native" / name, 0o444, follow_symlinks=False)
        os.chmod(path / "native", 0o555, follow_symlinks=False)
        rows = _raw_files(path / "native", checkpoint, c)
        checkpoint.audit_checkpoint_child(path / "native", lease.progress, runner._bed.context,
            runner._water.column, runner._water.sediment, identity, limits=limits,
            expected_layout=layout, raw_files=rows)  # retained ORIGINAL writer, never loaded substitute
        _export_check(parent, runner, lease, token, declared)
        export = {"schema": EXPORT_SCHEMA, "capsule_sha256": capsule_sha,
                  "capsule_bytes": len(capsule_raw), "identity": identity,
                  "raw_files": [dataclasses.asdict(row) for row in rows],
                  "layout": _normalize_metadata(dataclasses.asdict(layout))}
        export_raw = _metadata(export, _CAPSULE_BYTES, "export descriptor")
        c._write(path / "export.json", export_raw)
        os.chmod(path / "export.json", 0o444, follow_symlinks=False)
        c._same_directory(path, rootfd)
        _raw_files(path / "native", checkpoint, c, expected=export["raw_files"])
        os.fsync(rootfd)
        lease.phase = "paused"
        pause._lease(runner, token)
        return _sha(export_raw)
    except BaseException as primary:
        try:
            _export_check(parent, runner, lease, token, declared)
        except BaseException as closing:
            try:
                pause._halt(runner, lease, dr._Stop("runner_error", "active_export_changed",
                    "export input/authority changed; private attempt halted",
                    exception=None, elapsed=None))
            except BaseException:
                pass  # preserve original Native/IO/interrupt exception
            primary.add_note("closing export check refused: " + type(closing).__name__[:64])
        else:
            lease.phase = "paused"  # input unchanged: IO failure did not consume token
        raise
    finally:
        if rootfd is not None:
            os.close(rootfd)
        if parentfd is not None:
            os.close(parentfd)


def load_water_pause(parent, runner, *, attempt, expected_export_sha256, runtime_reference):
    """Fresh-process private restore; Native LOAD, no entry/commit/reset/replay."""
    from masy import dense_water_pause as pause
    declared = _declare()
    n, checkpoint, schemas, c, backend, _snapshot, native = declared
    pause._runner(runner)
    if runner._active_evidence is not None or runner._stop_receipt is not None:
        _fail("active_runner_busy", "accepted unoccupied runner required before any child read")
    pause._profile(runner, n)
    _parent(parent, runner)
    runner._precheck()
    _raw, source, environment = pause._runtime(c, runtime_reference)
    if c.oi._json(source) != parent._source or environment != parent._environment:
        _fail("active_runtime", "actual selected runtime differs from C accepted parent")
    c._digest(expected_export_sha256, "external writer export descriptor SHA")
    # Same existing initial plan-allocation guard as pause_water. This computes
    # validation-only original plan, never starts a Native event or stage.
    runner._work_bytes_check(pause._metadata_bytes(runner._limits.max_report_rows))
    _resolved, plan = pause._plan(runner, n)
    path = _attempt_path(parent, attempt)
    rootfd = _directory(path, c)
    item = capsule = export = None
    try:
        info = (path / "export.json").lstat()
        _integer(info.st_size, _CAPSULE_BYTES, "export metadata bytes", 1)
        _reserve_active(runner, plan, raw_metadata_bytes=info.st_size)
        export_raw = c._read(path / "export.json", _CAPSULE_BYTES)
        if info.st_mode & 0o222 or _sha(export_raw) != expected_export_sha256:
            _fail("active_export_changed", "external actual writer whole bytes/readonly custody differ")
        export = _decode(export_raw, _CAPSULE_BYTES)
        c._keys(export, {"schema", "capsule_sha256", "capsule_bytes", "identity", "raw_files", "layout"}, "export")
        if export["schema"] != EXPORT_SCHEMA:
            _fail("active_export_schema", "closed real writer descriptor required")
        c._digest(export["capsule_sha256"], "capsule digest")
        _integer(export["capsule_bytes"], _CAPSULE_BYTES, "capsule raw bytes", 1)
        _reserve_active(runner, plan, raw_metadata_bytes=info.st_size + export["capsule_bytes"])
        capsule_raw = c._read(path / "entry/capsule.json", _CAPSULE_BYTES)
        if (len(capsule_raw) != export["capsule_bytes"] or _sha(capsule_raw) != export["capsule_sha256"]
                or (path / "entry/capsule.json").lstat().st_mode & 0o222):
            _fail("active_capsule_changed", "whole raw capsule admission before parsing")
        capsule = _decode(capsule_raw, _CAPSULE_BYTES)
        resolved, plan = _admit_capsule(parent, runner, capsule, n, c)
        identity = _native_identity(capsule, export["capsule_sha256"])
        if c.oi._json(identity) != c.oi._json(export["identity"]):
            _fail("active_native_identity", "capsule/Native identity chain differs")
        layout = export["layout"]
        c._keys(layout, tuple(f.name for f in dataclasses.fields(checkpoint.CheckpointWriteBudget)),
                "actual Native P1 descriptor")
        declared_json = _integer(layout["checkpoint_json_bytes"], _NATIVE_JSON_BYTES,
                                 "Native declared JSON bytes", 1)
        actual_json = (path / "native/checkpoint.json").lstat()
        if (not stat.S_ISREG(actual_json.st_mode) or actual_json.st_nlink != 1
                or actual_json.st_uid != os.geteuid() or actual_json.st_mode & 0o222
                or _integer(actual_json.st_size, _NATIVE_JSON_BYTES, "Native actual JSON bytes", 1) != declared_json):
            _fail("active_layout", "actual bounded Native JSON size differs before parser reservation")
        _reserve_active(runner, plan, phase="reader", raw_metadata_bytes=(
            info.st_size + export["capsule_bytes"] + actual_json.st_size))
        _raw_files(path / "native", checkpoint, c, expected=export["raw_files"])
        native_data = _header_admission(path / "native", export["layout"], native, c,
                                         bounds=pause._bounds(runner, plan))
        if c.oi._json(native_data.get("identity")) != c.oi._json(identity):
            _fail("active_native_identity", "actual native JSON identity differs before LOAD")
        del native_data
        _parent(parent, runner)
        item = checkpoint.load_checkpoint(path / "native", runner._bed.context, runner._water.column,
            runner._water.sediment, identity, allow_complete=False, max_bytes=_NATIVE_BYTES,
            terrain_anchor=runner._water.terrain)
        if _native_projection(item, n, checkpoint, backend, schemas, pause._bounds(runner, plan)) != capsule["native_semantic_sha256"]:
            _fail("active_native_semantics", "Native LOAD values differ from actual writer projection")
        _raw_files(path / "native", checkpoint, c, expected=export["raw_files"])
        if _sha(c._read(path / "entry/capsule.json", _CAPSULE_BYTES)) != export["capsule_sha256"]:
            _fail("active_capsule_changed", "closing capsule identity drift")
        if _sha(c._read(path / "export.json", _CAPSULE_BYTES)) != expected_export_sha256:
            _fail("active_export_changed", "closing descriptor drift")
        c._same_directory(path, rootfd)
        token = _mint(parent, runner, capsule, export["capsule_sha256"], item, runtime_reference, declared)
        item = capsule = export = None
        return token
    finally:
        item = capsule = export = None
        os.close(rootfd)


