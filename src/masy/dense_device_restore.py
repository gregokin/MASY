"""Private cold HOST physical evidence for the default device-checkpoint format.

Native MAPLE serialization is the sole physical codec. This component neither
restores a device nor creates an executable runner. Artifacts and original
inputs must remain caller-private and quiescent throughout each operation.
The nominal Python carrier is not a security or provenance boundary.
"""
from __future__ import annotations

from collections import Counter
import dataclasses
import hashlib
import math
import os
from pathlib import Path
import re
import stat

__all__: list[str] = []

_SCHEMA = "masy.device_physical_cold_host.v1"
_SCOPE = "physical_only_cold_host;no_device_or_runner_authority;soil_and_routing_excluded"
_IDENTITY_SCOPE = "physical_only;soil_and_routing_payloads_excluded;no_restore_authority"
_RECORD = {"schema", "identity", "identity_sha256", "static_record", "files", "layout", "reservation", "scope"}
_IDENTITY = {"schema", "caller", "capsule_sha256", "case_sha256", "static_sha256", "config_sha256",
             "physical_sha256", "native_time_hex", "next_event", "bed_generation", "resource_generation",
             "limits", "scope"}
_STATIC = {"schema", "profile", "case_sha256", "schedule_sha256", "case_schedule_sha256",
           "controlling_inputs_sha256", "limits", "device_limits", "bed_limits", "resource_limits",
           "initial_evolving", "source"}
_PLAN = {"numeric_bytes", "largest_leaf_bytes", "leaves", "device_bytes", "device_arrays"}
_STAT = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_LEFTOVER = re.compile(r"\.manifest-[0-9a-f]{32}\Z")
_MINT = object()


def _fail(code, detail):
    from .dense_device_checkpoint import DeviceCheckpointError
    raise DeviceCheckpointError(code, detail)


def _keys(value, names, where):
    if type(value) is not dict or set(value) != set(names):
        _fail("cold_record", where + ": exact fields required")


def _integer(value, low, high, where):
    if type(value) is not int or not low <= value <= high:
        _fail("cold_record", where + ": bounded plain integer required")
    return value


def _sha(value, where):
    if type(value) is not str or _SHA.fullmatch(value) is None:
        _fail("cold_identity", where + ": SHA256 required")
    return value


def _path(value):
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path:
        _fail("cold_path", "canonical absolute artifact paths required")
    return path


def _stat_record(value):
    return {name: getattr(value, name) for name in _STAT}


def _read(path, maximum):
    """Bounded no-follow regular-file read. Native manifest hard links remain valid."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum:
        _fail("cold_file", "bounded regular file required")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        opened = os.fstat(fd)
        blocks, size = [], 0
        while size <= maximum:
            block = os.read(fd, min(1 << 20, maximum + 1 - size))
            if not block:
                break
            blocks.append(block); size += len(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (size != before.st_size or not _stat_record(before) == _stat_record(opened)
            == _stat_record(after) == _stat_record(path.lstat())):
        _fail("cold_file", "artifact changed during bounded read")
    return b"".join(blocks), _stat_record(before)


def _names(root, codec):
    before = root.lstat()
    if not stat.S_ISDIR(before.st_mode):
        _fail("cold_file", "native bundle must be a regular directory")
    names = set()
    with os.scandir(root) as entries:
        for entry in entries:
            names.add(entry.name)
            if len(names) > 4:
                _fail("cold_file", "native bundle has more than four entries")
    required = {codec.MANIFEST, *codec.PAYLOADS}
    if not required <= names:
        _fail("cold_file", "native bundle is incomplete")
    for name in names - required:
        path = root / name
        if (_LEFTOVER.fullmatch(name) is None or path.is_symlink()
                or not path.is_file() or not os.path.samefile(path, root / codec.MANIFEST)):
            _fail("cold_file", "only the native same-file manifest leftover is supported")
    return names, _stat_record(before)


def _file_bounds(files, root, codec, limits):
    if type(files) is not dict or not 3 <= len(files) <= 4:
        _fail("cold_file", "closed finite native file inventory required")
    names = set(files)
    required = {codec.MANIFEST, *codec.PAYLOADS}
    if (not required <= names or any(type(n) is not str or _LEFTOVER.fullmatch(n) is None
                                    for n in names - required)):
        _fail("cold_file", "unknown or missing native file")
    payload_total, total = 0, 0
    for name, row in files.items():
        _keys(row, {"path", "size_bytes", "sha256", "stat"}, "native file row")
        _keys(row["stat"], _STAT, "native file stat")
        if row["path"] != str(root / name):
            _fail("cold_file", "native file reference path differs")
        _sha(row["sha256"], "native file")
        for key, value in row["stat"].items():
            _integer(value, 0, (1 << 127) - 1, "native file " + key)
        size = _integer(row["size_bytes"], 0, limits.max_native_bytes + codec._ZIP_METADATA_SLACK, "raw file size")
        if row["stat"]["st_size"] != size or not stat.S_ISREG(row["stat"]["st_mode"]):
            _fail("cold_file", "native file size/type differs")
        if name in codec.PAYLOADS:
            payload_total += size
        elif size > limits.max_metadata_bytes:
            _fail("cold_file", "native manifest metadata exceeds cap")
        total += size
    if (payload_total > limits.max_native_bytes + len(codec.PAYLOADS) * codec._ZIP_METADATA_SLACK
            or total > limits.max_native_bytes + len(codec.PAYLOADS) * codec._ZIP_METADATA_SLACK
            + 2 * limits.max_metadata_bytes):
        _fail("cold_file", "aggregate native raw bytes exceed cap")


def _snapshot(root, codec, limits):
    names, root_stat = _names(root, codec)
    files = {}
    # Admit EVERY raw file before hashing any payload or parsing a ZIP directory.
    for name in sorted(names):
        path = root / name; value = path.lstat()
        files[name] = {"path": str(path), "size_bytes": value.st_size,
                       "sha256": "0" * 64, "stat": _stat_record(value)}
    _file_bounds(files, root, codec, limits)
    for name, row in files.items():
        path = root / name; value = path.lstat(); h = hashlib.sha256(); size = 0
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if _stat_record(os.fstat(fd)) != row["stat"]:
                _fail("cold_file", "native file identity changed before hashing")
            while size <= row["size_bytes"]:
                block = os.read(fd, min(1 << 20, row["size_bytes"] + 1 - size))
                if not block:
                    break
                size += len(block); h.update(block)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if (size != row["size_bytes"] or not _stat_record(value) == row["stat"]
                == _stat_record(after) == _stat_record(path.lstat())):
            _fail("cold_file", "native file changed during hashing")
        row["sha256"] = h.hexdigest()
    final_names, final_root = _names(root, codec)
    if final_names != names or final_root != root_stat:
        _fail("cold_file", "native directory changed during observation")
    return files, root_stat


def _originals(checkpoint, config, inputs, limits):
    bed, bridge, n, codec, transport = checkpoint._native()
    static_bytes = bridge._budget(codec, limits.max_host_data_bytes, inputs, config)
    codec._fields(inputs, bridge.DenseBoundaryInputs, bridge._INPUTS)
    codec._fields(inputs.context, n.BedContext, bridge._CONTEXT)
    bridge._keys(inputs.case_record, bridge._CASE, "original case")
    g = inputs.context.geometry
    if type(g) is not n.GeometrySpec or type(inputs.case_record) is not dict:
        _fail("unsupported_profile", "exact original geometry/case metadata required")
    for name in ("nx", "ny"):
        _integer(getattr(g, name), 1, 64, name)
    _integer(g.nx * g.ny, 1, 64, "cells")
    _integer(inputs.case_record["n_voxels"], 1, 64, "voxels")
    if (type(inputs.context.grain_classes) is not n.GrainClassSet
            or type(inputs.context.grain_classes.classes) is not tuple
            or len(inputs.context.grain_classes.classes) != 6 or type(config) is not transport.SchedulerPhysicsConfig
            or config.execution_backend != "numpy"
            or config.particle_transport.mobile_state_representation != "flat"):
        _fail("unsupported_profile", "original default six-class NumPy storage config required")
    _integer(config.particle_transport.max_active_cohorts, 1, 16000, "flat cohorts")
    bridge._validate_inputs(inputs, n, codec, limits.max_host_data_bytes)
    seal = {"config": codec._fingerprint(config), "capsule": inputs.binding_sha256,
            "case": inputs.case_sha256, "limits": dataclasses.asdict(limits)}
    return bed, bridge, n, codec, transport, static_bytes, seal


def _static_record(value, identity, inputs, bed, bridge, codec, limits):
    from . import dense_device_runner as device
    bridge._budget(codec, limits.max_metadata_bytes, value, plain_json=True)
    _keys(value, _STATIC, "default device static record")
    if (value["schema"] != device.DEVICE_STATIC_SCHEMA
            or codec._json(value["profile"]) != codec._json(device.DEVICE_PROFILE)
            or value["case_sha256"] != inputs.case_sha256):
        _fail("cold_static", "default static schema/profile/original case differs")
    for name in ("case_sha256", "schedule_sha256", "case_schedule_sha256", "controlling_inputs_sha256"):
        _sha(value[name], name)
    from .dense_device_bed import DeviceBedLimits
    from .dense_device_resources import DeviceResourceLimits
    from .dense_runner import DenseRunnerLimits
    for name, cls in (("limits", DenseRunnerLimits), ("device_limits", device.DeviceRunnerLimits),
                      ("bed_limits", DeviceBedLimits), ("resource_limits", DeviceResourceLimits)):
        _keys(value[name], {f.name for f in dataclasses.fields(cls)}, name)
        cls(**value[name])  # The actual metadata-only owning validators, no bed or device construction.
    _integer(value["limits"]["max_events"], 1, limits.max_events, "original event cap")
    _keys(value["initial_evolving"], {"mobile_state_sha256", "soil_water_m"}, "initial evolving labels")
    _sha(value["initial_evolving"]["mobile_state_sha256"], "original mobile label")
    soil = value["initial_evolving"]["soil_water_m"]
    _keys(soil, {"array"}, "original soil digest label")
    _keys(soil["array"], {"dtype", "shape", "order", "nbytes", "sha256"}, "original soil array label")
    _sha(soil["array"]["sha256"], "original soil label")
    shape = soil["array"]["shape"]
    if type(shape) is not list or len(shape) != 2:
        _fail("cold_static", "original soil array label rank differs")
    for dimension in shape:
        _integer(dimension, 1, 64, "original soil dimension label")
    if (soil["array"]["dtype"] != bed._native().np.dtype("float64").str or soil["array"]["order"] != "C"
            or soil["array"]["shape"] != [inputs.context.geometry.ny, inputs.context.geometry.nx]):
        _fail("cold_static", "original soil array label layout differs")
    _integer(soil["array"]["nbytes"], 0, limits.max_data_bytes, "original soil byte label")
    if soil["array"]["nbytes"] != inputs.context.geometry.nx * inputs.context.geometry.ny * 8:
        _fail("cold_static", "original soil array byte label differs")
    _keys(value["source"], {"bed_identity_sha256", "conversion"}, "original source labels")
    _sha(value["source"]["bed_identity_sha256"], "original bed label")
    if type(value["source"]["conversion"]) is not str or len(value["source"]["conversion"]) > 4096:
        _fail("cold_static", "bounded conversion metadata required")
    if bed._sha256(value) != identity["static_sha256"]:
        _fail("cold_static", "complete static record hash differs")


def _identity_record(identity, identity_sha, config, inputs, checkpoint, bed, bridge, codec, limits):
    bridge._budget(codec, limits.max_metadata_bytes, identity, plain_json=True)
    _keys(identity, _IDENTITY, "device-format physical identity")
    codec._identity(identity)
    _keys(identity["limits"], dataclasses.asdict(limits), "physical identity limits")
    for name, bound in dataclasses.asdict(limits).items():
        _integer(identity["limits"][name], bound, bound, "physical identity " + name)
    _sha(identity_sha, "device-format identity digest")
    if (identity["schema"] != checkpoint.SCHEMA or identity["scope"] != _IDENTITY_SCOPE
            or identity["limits"] != dataclasses.asdict(limits)
            or identity["case_sha256"] != inputs.case_sha256
            or identity["capsule_sha256"] != inputs.binding_sha256
            or identity["config_sha256"] != codec._fingerprint(config)
            or checkpoint._json_seal(codec, bridge, limits, identity) != identity_sha):
        _fail("cold_identity", "original config/capsule/identity differs")
    for name in ("case_sha256", "capsule_sha256", "config_sha256", "static_sha256", "physical_sha256"):
        _sha(identity[name], name)
    for name in ("bed_generation", "resource_generation"):
        _integer(identity[name], 0, (1 << 63) - 1, name)
    _integer(identity["next_event"], 0, limits.max_events, "event cursor")
    if type(identity["native_time_hex"]) is not str or len(identity["native_time_hex"]) > 32:
        _fail("cold_identity", "native clock hex required")
    try:
        native_time = float.fromhex(identity["native_time_hex"])
        bridge._native_clock(native_time, "cold physical time")
    except (ValueError, TypeError, OverflowError) as exc:
        _fail("cold_identity", "invalid native clock: " + type(exc).__name__)
    if native_time.hex() != identity["native_time_hex"]:
        _fail("cold_identity", "native clock spelling differs")


def _headers(layout, codec, np, limits):
    _keys(layout, {"host", "archives", "multiset", "scalar_bytes"}, "host layout")
    host = layout["host"]; _keys(host, _PLAN, "host leaf plan")
    for key in _PLAN:
        _integer(host[key], 0, limits.max_data_bytes, "host plan " + key)
    if host["device_bytes"] or host["device_arrays"] or not 1 <= host["leaves"] <= limits.max_leaves:
        _fail("cold_layout", "cold layout must describe host leaves only")
    _keys(layout["archives"], codec.PAYLOADS, "native archive layout")
    counter, scalar_bytes, npy_bytes, numeric, largest, leaves = Counter(), 0, 0, 0, 0, 0
    for members in layout["archives"].values():
        if type(members) is not dict or not 1 <= len(members) <= 256:
            _fail("cold_layout", "bounded native member inventory required")
        for name, spec in members.items():
            if (type(name) is not str or not name or len(name) > 196
                    or "/" in name or "\\" in name or name in {".", ".."}):
                _fail("cold_layout", "unsafe native member name")
            _keys(spec, {"dtype", "shape", "npy_bytes"}, "native member metadata")
            match = re.fullmatch(r"[<>=|]([fiubSU])([0-9]{1,7})", spec["dtype"]) if type(spec["dtype"]) is str else None
            if match is None or not 1 <= int(match[2]) <= limits.max_metadata_bytes:
                _fail("cold_layout", "bounded native dtype required")
            dtype = np.dtype(spec["dtype"])
            if dtype.str != spec["dtype"] or dtype.hasobject or dtype.fields or not 0 < dtype.itemsize <= limits.max_metadata_bytes:
                _fail("cold_layout", "canonical nonobject dtype required")
            shape = spec["shape"]
            if type(shape) is not list or len(shape) > 4:
                _fail("cold_layout", "bounded native rank required")
            for value in shape:
                _integer(value, 0, limits.max_data_bytes, "native dimension")
            size = math.prod(shape) * dtype.itemsize
            npy_bytes += _integer(spec["npy_bytes"], size, limits.max_native_bytes, "NPY bytes")
            if npy_bytes > limits.max_native_bytes:
                _fail("cold_layout", "aggregate native declared bytes exceed cap")
            if shape:
                if dtype.kind not in "fiub" or size > limits.max_data_bytes:
                    _fail("cold_layout", "physical array dtype/bytes differ")
                numeric += size; largest = max(largest, size); leaves += 1
                counter[(dtype.str, tuple(shape))] += 1
            else:
                scalar_bytes += dtype.itemsize
    expected = {"numeric_bytes": numeric, "largest_leaf_bytes": largest, "leaves": leaves,
                "device_bytes": 0, "device_arrays": 0}
    multiset = [[dtype, list(shape), count] for (dtype, shape), count in sorted(counter.items())]
    if (host != expected or codec._json(layout["multiset"]) != codec._json(multiset)
            or layout["scalar_bytes"] != scalar_bytes
            or type(layout["scalar_bytes"]) is not int or scalar_bytes > limits.max_metadata_bytes
            or numeric > limits.max_data_bytes):
        _fail("cold_layout", "host multiset/aggregate scalar or data budget differs")
    return host


def _archive_layout(root, physical, checkpoint, codec, np, limits):
    codec._serialized_budget(root, limits.max_native_bytes)
    expected = {codec.PAYLOADS[0]: codec._expected_base(physical), codec.PAYLOADS[1]: set(codec._pending(physical))}
    archives, used = {}, 0
    for name in codec.PAYLOADS:
        members, size = codec._audit_archive(root / name, limits.max_native_bytes - used); used += size
        if set(members) != expected[name]:
            _fail("cold_layout", "native field inventory differs")
        archives[name] = members
    arrays = list(codec._arrays(physical))
    counter = Counter((a.dtype.str, tuple(a.shape)) for a in arrays)
    layout = {"host": checkpoint._array_plan((physical,), np.ndarray, np, limits), "archives": archives,
              "multiset": [[dtype, list(shape), count] for (dtype, shape), count in sorted(counter.items())],
              "scalar_bytes": sum(np.dtype(spec["dtype"]).itemsize for members in archives.values()
                                  for spec in members.values() if not spec["shape"])}
    _headers(layout, codec, np, limits)
    return layout


def _physical(physical, identity, config, inputs, limits, bed, bridge, n, codec, transport):
    from .dense_device_bed import _host_bed_schema
    bridge._budget(codec, limits.max_host_data_bytes, physical, require_readonly=True)
    if (type(physical) is not n.SchedulerPhysicalState or physical.vegetation is None
            or physical.vegetation.objects is not None):
        _fail("unsupported_profile", "exact native readonly raster physical state required")
    bridge._validate_physical(physical, config, inputs, n, codec, limits.max_host_data_bytes)
    if not transport.mobile_state_is_genuinely_empty(physical.mobile_state):
        _fail("cold_mobile", "wind mobile state is not genuinely empty")
    prescribed = {"vegetation": bed._sha256(bed._canonical_static(physical.vegetation, "vegetation")),
                  "native_vegetation_content_sha256": n.vegetation_content_hash(physical.vegetation),
                  "threshold_modifier_state": bed._sha256(bed._canonical_static(
                      physical.threshold_modifier_state, "threshold_modifier_state"))}
    if prescribed != inputs.prescribed_digests:
        _fail("cold_prescribed", "original prescribed fields differ")
    shallow = n.BedState(**{name: getattr(physical, name) for name in bed.SHARED_FIELDS})
    _host_bed_schema(shallow, geometry=inputs.context.geometry,
                     n_voxels=inputs.case_record["n_voxels"], n_classes=6)
    bed._check_commit_stamp(shallow, physical.time_s)
    if (codec._fingerprint(physical) != identity["physical_sha256"]
            or physical.time_s.hex() != identity["native_time_hex"]):
        _fail("cold_physical", "native physical bytes/clock differ")


class ColdRecordPublicationUncertain(OSError):
    """Visible companion failed durability or post-publication verification.

    The destination is retained; this exception grants no restoration authority.
    Its cause distinguishes fsync uncertainty from a later validation failure.
    """
    def __init__(self, path, digest):
        self.path, self.record_sha256 = Path(path), digest
        super().__init__("cold companion publication/durability uncertain; retain destination")


def _publish(path, raw, codec):
    digest = hashlib.sha256(raw).hexdigest()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        try:
            used = 0
            while used < len(raw):
                n = os.write(fd, raw[used:])
                if n <= 0:
                    raise OSError("short companion write")
                used += n
            os.fsync(fd)
        finally:
            os.close(fd)
        codec._fsync(path.parent, directory=True)
    except codec.CheckpointPublicationUncertain:
        raise
    except OSError as exc:
        raise ColdRecordPublicationUncertain(path, digest) from exc
    return digest


def seal_device_boundary_restore(bundle, record_path, export, config, *, inputs, static_record, limits):
    """Seal caller-private native physical data; exact issued export remains mandatory."""
    from . import dense_device_checkpoint as checkpoint
    checkpoint._limits(limits)
    root, record_path = _path(bundle), _path(record_path)
    if record_path.is_relative_to(root) or os.path.lexists(record_path):
        _fail("cold_path", "companion must be a new path outside native bundle")
    checkpoint._validate_export(export, config, inputs, limits)
    bed, bridge, n, codec, transport, static_bytes, original = _originals(checkpoint, config, inputs, limits)
    _identity_record(export.identity, export.identity_sha256, config, inputs, checkpoint, bed, bridge, codec, limits)
    _static_record(static_record, export.identity, inputs, bed, bridge, codec, limits)
    before, root_before = _snapshot(root, codec, limits)
    loaded = checkpoint.verify_device_boundary_roundtrip(root, export, config, inputs=inputs, limits=limits)
    _physical(loaded, export.identity, config, inputs, limits, bed, bridge, n, codec, transport)
    layout = _archive_layout(root, export.physical, checkpoint, codec, n.np, limits)
    reservation = checkpoint._reserve(layout["host"], static_bytes, limits)
    record = {"schema": _SCHEMA, "identity": codec._identity(export.identity),
              "identity_sha256": export.identity_sha256, "static_record": static_record,
              "files": before, "layout": layout, "reservation": reservation, "scope": _SCOPE}
    bridge._budget(codec, limits.max_metadata_bytes, record, plain_json=True)
    raw = codec._json(record).encode()
    if len(raw) > limits.max_metadata_bytes:
        _fail("cold_record", "companion exceeds metadata cap")
    if _snapshot(root, codec, limits) != (before, root_before):
        _fail("cold_file", "native artifact changed before companion publication")
    if _originals(checkpoint, config, inputs, limits)[-1] != original:
        _fail("cold_identity", "original inputs changed during sealing")
    _static_record(static_record, export.identity, inputs, bed, bridge, codec, limits)
    checkpoint._validate_export(export, config, inputs, limits)
    del loaded
    digest = _publish(record_path, raw, codec)
    try:
        current, _ = _read(record_path, limits.max_metadata_bytes)
        if current != raw or _snapshot(root, codec, limits) != (before, root_before):
            _fail("cold_file", "published companion/native artifact changed; evidence retained")
        if _originals(checkpoint, config, inputs, limits)[-1] != original:
            _fail("cold_identity", "original inputs changed during publication; evidence retained")
        _static_record(static_record, export.identity, inputs, bed, bridge, codec, limits)
    except codec.CheckpointPublicationUncertain:
        raise
    except Exception as exc:
        raise ColdRecordPublicationUncertain(record_path, digest) from exc
    return digest


class _ColdPhysical:
    __slots__ = ("_physical", "_inputs", "_config", "_limits", "_record", "_bundle", "_record_path",
                 "_record_observation", "_snapshot", "_record_sha256", "_mint", "__weakref__")

    def __new__(cls, *args, **kwargs):
        _fail("cold_ticket", "only the actual cold native loader creates this nominal carrier")

    def diagnostics(self):
        import json
        if getattr(self, "_mint", None) is not _MINT:
            _fail("cold_ticket", "unminted cold carrier")
        record = self._record
        value = {"scope": _SCOPE, "record_sha256": self._record_sha256, "released": self._physical is None,
                 "device_restore": False, "combined_restart": False, "gpu_qualified": False,
                 "memory_fit": False, "restart_supported": False}
        if record is not None:
            value.update(identity=record["identity"], layout=record["layout"], reservation=record["reservation"])
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))

    def _release(self):
        if getattr(self, "_mint", None) is not _MINT:
            _fail("cold_ticket", "unminted cold carrier")
        self._physical = self._inputs = self._config = self._limits = self._record = self._snapshot = None


def load_device_boundary_cold(bundle, record_path, config, *, inputs, limits, expected_record_sha256):
    """Return readonly host physical evidence; no device, runner or event is admitted."""
    from . import dense_device_checkpoint as checkpoint
    checkpoint._limits(limits)
    _sha(expected_record_sha256, "external companion digest")
    root, record_path = _path(bundle), _path(record_path)
    if record_path.is_relative_to(root):
        _fail("cold_path", "companion must remain outside native bundle")
    raw, record_observation = _read(record_path, limits.max_metadata_bytes)
    if hashlib.sha256(raw).hexdigest() != expected_record_sha256:
        _fail("cold_record", "external raw companion identity differs")
    bed, bridge, n, codec, transport, static_bytes, original = _originals(checkpoint, config, inputs, limits)
    record = codec._strict_json(raw)
    bridge._budget(codec, limits.max_metadata_bytes, record, plain_json=True)
    _keys(record, _RECORD, "cold companion")
    if record["schema"] != _SCHEMA or record["scope"] != _SCOPE or codec._json(record).encode() != raw:
        _fail("cold_record", "canonical physical-only companion schema differs")
    _identity_record(record["identity"], record["identity_sha256"], config, inputs, checkpoint, bed, bridge, codec, limits)
    _static_record(record["static_record"], record["identity"], inputs, bed, bridge, codec, limits)
    plan = _headers(record["layout"], codec, n.np, limits)
    if codec._json(record["reservation"]) != codec._json(checkpoint._reserve(plan, static_bytes, limits)):
        _fail("host_bytes_cap", "fresh host reservation differs from recorded metadata")
    _file_bounds(record["files"], root, codec, limits)
    before, root_before = _snapshot(root, codec, limits)
    if before != record["files"]:
        _fail("cold_file", "raw native artifact identity differs")
    codec._serialized_budget(root, limits.max_native_bytes)
    used = 0
    for name in codec.PAYLOADS:
        headers, size = codec._audit_archive(root / name, limits.max_native_bytes - used); used += size
        if headers != record["layout"]["archives"][name]:
            _fail("cold_layout", "actual native archive headers differ")
    physical = codec.load_exact_physical_checkpoint(root, config, record["identity"], max_bytes=limits.max_native_bytes)
    _physical(physical, record["identity"], config, inputs, limits, bed, bridge, n, codec, transport)
    if checkpoint._array_plan((physical,), n.np.ndarray, n.np, limits) != plan:
        _fail("cold_layout", "loaded host physical layout differs")
    for name, expected in ((codec.PAYLOADS[0], codec._expected_base(physical)),
                           (codec.PAYLOADS[1], set(codec._pending(physical)))):
        if set(record["layout"]["archives"][name]) != expected:
            _fail("cold_layout", "loaded native field inventory differs")
    if _snapshot(root, codec, limits) != (before, root_before):
        _fail("cold_file", "native artifact changed during cold load")
    if _read(record_path, limits.max_metadata_bytes) != (raw, record_observation):
        _fail("cold_file", "companion changed during cold load")
    if _originals(checkpoint, config, inputs, limits)[-1] != original:
        _fail("cold_identity", "original inputs changed during cold load")
    result = object.__new__(_ColdPhysical)
    result._physical, result._inputs, result._config, result._limits = physical, inputs, config, limits
    result._record, result._bundle, result._record_path = record, root, record_path
    result._record_observation, result._snapshot = record_observation, (before, root_before)
    result._record_sha256, result._mint = expected_record_sha256, _MINT
    return result


# Private bed-only device admission. All preceding cold-HOST definitions are unchanged.
def _admission_roots(values, np, codec, limits, *, forbidden_holders=()):
    """Bound and record the temporary array/root plan for cold bed admission.

    Metadata-only walk: no hashing, copying, content reads or flag changes.
    Aliases are visited and charged each time they occur; each unique terminal
    NumPy root is charged once at its full allocation size before any hash.
    """
    import dataclasses
    import math
    from enum import Enum

    native = ("maple", "maple_syrup")
    wrappers = frozenset({
        ("masy.dense_checkpoint", "DenseBoundaryInputs"),
        ("masy.dense_bed", "DenseBedLimits"),
    })
    max_leaves = getattr(limits, "max_leaves", None)
    max_host = getattr(limits, "max_host_data_bytes", None)
    if type(max_leaves) is not int or type(max_host) is not int or max_leaves < 1 or max_host < 1:
        _fail("admission_layout", "limits must carry exact positive int caps")
    if type(forbidden_holders) not in (tuple, frozenset) or len(forbidden_holders) > 6:
        _fail("admission_layout", "forbidden_holders must be a bounded tuple of ids")
    if any(type(item) is not int for item in forbidden_holders):
        _fail("admission_layout", "forbidden_holders must contain exact int ids")
    forbidden = frozenset(forbidden_holders)
    if type(values) is not tuple or not 1 <= len(values) <= 3:
        _fail("admission_layout", "values must be a bounded exact tuple")

    arrays, roots, active = [], {}, set()
    nodes = leaves = visible = root_bytes = 0

    def name(value):
        return str(type(value).__qualname__)[:64]

    def native_type(value):
        module = type(value).__module__
        return type(module) is str and module.split(".", 1)[0] in native

    def hold(value):
        if id(value) in forbidden:
            _fail("admission_layout", f"forbidden holder encountered: {name(value)}")

    def admit(array, path):
        nonlocal leaves, visible, root_bytes
        if leaves >= max_leaves:
            _fail("admission_host_cap", f"array visits exceed max_leaves={max_leaves}")
        leaves += 1
        dtype = array.dtype
        if (
            not array.flags.c_contiguous
            or array.ndim > 4
            or dtype.hasobject
            or dtype.kind not in "fiub"
            or not dtype.isnative
        ):
            _fail("admission_layout", f"unsupported leaf layout: ndim={array.ndim} kind={dtype.kind}")
        visible += array.nbytes
        if visible > max_host:
            _fail("admission_host_cap", "alias-counted visible bytes exceed HOST cap")
        root, chain, hops = array, {id(array)}, 0
        while root.base is not None:
            base = root.base
            if type(base) is not np.ndarray:
                _fail("admission_layout", f"array base is not an exact ndarray: {name(base)}")
            hold(base)
            hops += 1
            if hops > 24 or id(base) in chain:
                _fail("admission_layout", "array base chain is cyclic or exceeds depth 24")
            chain.add(id(base))
            root = base
        flags, rdtype = root.flags, root.dtype
        if (
            not flags.owndata
            or not (flags.c_contiguous or flags.f_contiguous)
            or rdtype.hasobject
            or not rdtype.isnative
        ):
            _fail("admission_layout", f"unsupported terminal root: kind={rdtype.kind} ndim={root.ndim}")
        if id(root) not in roots:
            if root_bytes + root.nbytes > max_host:
                _fail("admission_host_cap", "unique root allocations exceed max_host_data_bytes")
            root_bytes += root.nbytes
            roots[id(root)] = root
        arrays.append((path, array, root))

    def visit(value, path, depth):
        nonlocal nodes
        nodes += 1
        if depth > 24 or nodes > 100_000:
            _fail("admission_host_cap", "admission tree exceeds depth/node bound")
        kind = type(value)
        if value is None or kind is bool or kind is int or kind is str:
            return
        if kind is float:
            if not math.isfinite(value):
                _fail("admission_layout", "non-finite float in admission tree")
            return
        hold(value)
        if kind is np.ndarray:
            admit(value, path)
            return
        if id(value) in active:
            _fail("admission_layout", f"cycle detected at {name(value)}")
        if kind is dict:
            if len(value) > 100_000 - nodes:
                _fail("admission_host_cap", "admission mapping exceeds node bound")
            if any(type(key) is not str for key in value):
                _fail("admission_layout", "admission mapping keys must be strings")
            items = value.items()
        elif kind is list or kind is tuple:
            if len(value) > 100_000 - nodes:
                _fail("admission_host_cap", "admission sequence exceeds node bound")
            items = enumerate(value)
        elif isinstance(value, Enum):
            if not native_type(value):
                _fail("admission_layout", f"non-native enum: {name(value)}")
            items = (("value", value.value),)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            ident = (type(value).__module__, type(value).__qualname__)
            if not native_type(value) and ident not in wrappers:
                _fail("admission_layout", f"unsupported dataclass: {name(value)}")
            items = codec._instance(value).items()
        else:
            _fail("admission_layout", f"unsupported admission value type: {name(value)}")
        active.add(id(value))
        for key, item in items:
            visit(item, path + (key,), depth + 1)
        active.discard(id(value))

    for entry in values:
        if type(entry) is not tuple or len(entry) != 2 or type(entry[0]) is not str:
            _fail("admission_layout", "values entries must be exact (role, value) tuples")
        visit(entry[1], (entry[0],), 0)

    return {
        "arrays": tuple(arrays),
        "roots": tuple(roots.values()),
        "visible_bytes": visible,
        "root_bytes": root_bytes,
        "nodes": nodes,
    }


_ADMISSION_MINT = object()
_RESIDUAL_FIELDS = ("mobile_state", "threshold_modifier_state", "time_s", "vegetation",
                    "coupled_trajectory_semantics", "ledger_reconciliation_baseline")
_COUNTER_FIELDS = ("host_to_device", "host_to_device_bytes", "device_to_host",
                   "device_to_host_bytes", "scalar_reads", "synchronizations")
_OWNER_COUNTER_FIELDS = ("d2d_enqueued_arrays", "d2d_enqueued_bytes", "clone_attempts",
                         "packet_attempts", "packets_completed")
_COLD_SLOTS = ("_physical", "_inputs", "_config", "_limits", "_record", "_bundle", "_record_path",
               "_record_observation", "_snapshot", "_record_sha256", "_mint", "__weakref__")


class DeviceBedAdmissionError(ValueError):
    """Detached private-component refusal; evidence contains metadata only."""

    def __init__(self, code, detail, evidence):
        self.code, self.detail, self.evidence = code, detail, evidence
        super().__init__(code + ": " + detail)


def _admission_slots(cold):
    if (type(cold) is not _ColdPhysical or tuple(_ColdPhysical.__slots__) != _COLD_SLOTS
            or getattr(cold, "_mint", None) is not _MINT
            or any(not hasattr(cold, key) for key in _COLD_SLOTS if key != "__weakref__")
            or cold._physical is None):
        _fail("admission_cold", "exact live native cold-loader carrier required")
    path_type = type(Path("/"))
    for path in (cold._bundle, cold._record_path):
        if type(path) is not path_type or len(str(path)) > 4096 or _path(path) != path:
            _fail("admission_cold", "cold artifact references must remain canonical paths")
    _sha(cold._record_sha256, "cold record digest")
    _keys(cold._record_observation, _STAT, "cold record observation")
    for item in cold._record_observation.values():
        _integer(item, 0, (1 << 127) - 1, "cold stat")
    if type(cold._snapshot) is not tuple or len(cold._snapshot) != 2:
        _fail("admission_cold", "closed cold artifact snapshot required")


def _admission_root_disjoint(bed_roots, other_roots, np):
    for left in bed_roots:
        for right in other_roots:
            if left is right or np.shares_memory(left, right):
                _fail("admission_retention", "residual/originals retain a bed allocation root")


def _admission_layout_stamp(plan):
    return tuple((path, id(array), id(root), array.dtype.str, tuple(array.shape), tuple(array.strides),
                  bool(array.flags.writeable), int(array.__array_interface__["data"][0]),
                  int(root.nbytes), bool(root.flags.writeable))
                 for path, array, root in plan["arrays"])


def _prepare_bed_admission(cold, bed_limits):
    """HOST/native preflight only. Returns temporary references, never authority.

    Existing native/bridge layout, original-input validation and physical codec
    remain authoritative. This preparation does not load a second payload,
    initialize a bed, choose/import a device backend, or mutate any source.
    """
    from . import dense_device_checkpoint as checkpoint
    from .dense_device_bed import DeviceBedLimits, _host_bed_schema, _work_reservation
    from maple.core.backend import array_leaves
    from maple.core.backend.arrays import array_leaf_inventory

    _admission_slots(cold)
    physical, inputs, config, limits, record = (cold._physical, cold._inputs, cold._config,
                                               cold._limits, cold._record)
    bundle, record_path = cold._bundle, cold._record_path
    record_observation, source_snapshot, record_sha256 = (cold._record_observation,
                                                         cold._snapshot, cold._record_sha256)
    checkpoint._limits(limits)
    limit_values = tuple(getattr(limits, f.name) for f in dataclasses.fields(limits))
    if (type(bed_limits) is not DeviceBedLimits
            or set(vars(bed_limits)) != {f.name for f in dataclasses.fields(DeviceBedLimits)}):
        _fail("admission_limit", "exact DeviceBedLimits required")
    bed_limits.__post_init__()
    bed, bridge, n, codec, transport = checkpoint._native()
    if (len(bed.SHARED_FIELDS) != 6 or len(_RESIDUAL_FIELDS) != 6
            or set(bed.SHARED_FIELDS) & set(_RESIDUAL_FIELDS)
            or set(codec._PHYSICAL) != set(bed.SHARED_FIELDS) | set(_RESIDUAL_FIELDS)):
        _fail("admission_layout", "native physical fields must partition exactly into six bed and six residual fields")
    bridge._budget(codec, limits.max_metadata_bytes, record, record_observation,
                   *source_snapshot, plain_json=True)
    _keys(record, _RECORD, "cold companion")
    _keys(record["identity"], _IDENTITY, "cold physical identity")
    _keys(record["static_record"], _STATIC, "cold static identity")
    if (record["schema"] != _SCHEMA or record["scope"] != _SCOPE
            or dataclasses.asdict(bed_limits) != record["static_record"]["bed_limits"]):
        _fail("admission_limit", "bound original profile/bed limits differ")
    header_plan = _headers(record["layout"], codec, n.np, limits)
    _file_bounds(record["files"], bundle, codec, limits)
    static_used = bridge._budget(codec, limits.max_host_data_bytes, inputs, config)
    old_reservation = checkpoint._reserve(header_plan, static_used, limits)
    if codec._json(record["reservation"]) != codec._json(old_reservation):
        _fail("admission_host_cap", "cold reservation differs from actual owning formula")
    used = bridge._budget(codec, limits.max_host_data_bytes, physical, inputs,
                          config, record)
    bridge._budget(codec, limits.max_host_data_bytes, physical, require_readonly=True)
    whole = _admission_roots((("physical", physical), ("inputs", inputs),
                              ("config", config)), n.np, codec, limits)
    # Close native field/holder inventory BEFORE its nonmemoized reconstruction.
    codec._fields(physical, n.SchedulerPhysicalState, codec._PHYSICAL)
    if (physical.coupled_trajectory_semantics is not None
            or physical.ledger_reconciliation_baseline is not None
            or physical.vegetation is None):
        _fail("admission_profile", "current boundary excludes baseline/trajectory history and requires raster vegetation")
    codec._fields(physical.mobile_state, codec.MobileCohortState, codec._MOBILE)
    codec._fields(physical.threshold_modifier_state, codec.ThresholdModifierState, codec._THRESHOLD)
    codec._fields(physical.vegetation, codec.VegetationState, ("source", "raster", "objects"))
    if physical.vegetation.objects is not None:
        _fail("admission_profile", "current boundary admits prescribed raster vegetation only")
    codec._fields(physical.vegetation.raster, codec.VegetationRasterState, codec._RASTER)
    shallow = n.BedState(**{name: getattr(physical, name) for name in bed.SHARED_FIELDS})
    codec._fields(inputs, bridge.DenseBoundaryInputs, bridge._INPUTS)
    codec._fields(inputs.context, n.BedContext, bridge._CONTEXT)
    bridge._keys(inputs.case_record, bridge._CASE, "original case")
    g = inputs.context.geometry
    if type(g) is not n.GeometrySpec:
        _fail("admission_layout", "exact original geometry required")
    nv = _integer(inputs.case_record["n_voxels"], 1, 64, "original voxels")
    for dimension in (g.nx, g.ny):
        _integer(dimension, 1, 64, "original horizontal dimension")
    _integer(g.nx * g.ny, 1, 64, "original cells")
    if type(physical.time_s) is not float:
        _fail("admission_layout", "accepted cold boundary preserves a plain float native clock")
    metadata, scalars = _host_bed_schema(shallow, geometry=g, n_voxels=nv, n_classes=6)
    bed_plan = _admission_roots((("bed", shallow),), n.np, codec, limits)
    residual = tuple(getattr(physical, key) for key in _RESIDUAL_FIELDS)
    others = _admission_roots((("residual", residual), ("inputs", inputs),
                               ("config", config)), n.np, codec, limits,
                              forbidden_holders=tuple(id(getattr(shallow, key)) for key in bed.SHARED_FIELDS))
    _admission_root_disjoint(bed_plan["roots"], others["roots"], n.np)
    arrays, inventory = tuple(array_leaves(shallow)), array_leaf_inventory(shallow)
    if (len(arrays) != 23 or len({id(a) for a in arrays}) != 23
            or inventory["n_leaves"] != 23 or inventory["n_unique_leaves"] != 23
            or inventory["n_aliased_leaves"] or inventory["transfer_bytes"] != sum(a.nbytes for a in arrays)):
        _fail("admission_layout", "closed23 unique native bed leaves required")
    archives = record["layout"]["archives"]
    for name, expected in ((codec.PAYLOADS[0], codec._expected_base(physical)),
                           (codec.PAYLOADS[1], set(codec._pending(physical)))):
        _keys(archives[name], expected, "native field inventory")
    codec._raw_base(archives[codec.PAYLOADS[0]])
    touches_dtype = archives[codec.PAYLOADS[1]]["ledger__n_physical_touches"]["dtype"]
    if (shallow.ledger.n_physical_touches.dtype != n.np.dtype(touches_dtype)
            or shallow.ledger.n_physical_touches.dtype != n.np.dtype("int64")):
        _fail("admission_layout", "native saved touch dtype differs")
    total, largest = int(inventory["transfer_bytes"]), max(int(a.nbytes) for a in arrays)
    comparison, required_device = _work_reservation(inventory, largest, bed_limits)
    device_record = record["static_record"]["device_limits"]
    if type(device_record) is not dict or "max_device_bytes" not in device_record:
        _fail("admission_limit", "original device byte cap required")
    device_cap = _integer(device_record["max_device_bytes"], 1, 1 << 40, "device byte cap")
    if total > bed_limits.max_bed_bytes or required_device > min(bed_limits.max_work_bytes, device_cap):
        _fail("admission_device_cap", "unchanged six-root/native/compare reservation exceeds cap")
    # The unchanged _snapshot streams at most1MiB per os.read; assigning the
    # next block can coexist with the previous block. Companion _read's
    # chunks/join and detached/planner records have a separate metadata term.
    metadata_used = _integer(used - whole["visible_bytes"], 0, limits.max_host_data_bytes,
                             "existing tree metadata")
    terms = {"unique_host_allocation_roots": whole["root_bytes"],
             "existing_tree_metadata": metadata_used,
             "one_leaf_readback_and_comparison": 2 * largest,
             "native_stream_hash_buffers": 2 * (1 << 20),
             "companion_detached_and_planner_metadata": 8 * limits.max_metadata_bytes}
    new_required = sum(terms.values())
    extra_backing = max(0, whole["root_bytes"] - whole["visible_bytes"])
    required_host = max(new_required, old_reservation["required_host_data_bytes"] + extra_backing)
    if required_host > limits.max_host_data_bytes:
        _fail("admission_host_cap", "HOST root/hash/copy data reservation exceeds cap")
    # ALL allocation-root and raw-size admission above precedes fresh hashes.
    original = _originals(checkpoint, config, inputs, limits)[-1]
    _identity_record(record["identity"], record["identity_sha256"], config, inputs,
                     checkpoint, bed, bridge, codec, limits)
    _static_record(record["static_record"], record["identity"], inputs, bed, bridge, codec, limits)
    raw_record, observation = _read(record_path, limits.max_metadata_bytes)
    if (observation != record_observation or hashlib.sha256(raw_record).hexdigest() != record_sha256
            or codec._json(record).encode() != raw_record
            or _snapshot(bundle, codec, limits) != source_snapshot
            or source_snapshot[0] != record["files"]):
        _fail("admission_source", "retained local companion/native artifact changed")
    _physical(physical, record["identity"], config, inputs,
              limits, bed, bridge, n, codec, transport)
    if checkpoint._array_plan((physical,), n.np.ndarray, n.np, limits) != header_plan:
        _fail("admission_layout", "loaded physical layout differs from saved layout")
    binding = {"schema": "masy.device_bed_cold_admission.v1", "scope": "device_bed_plus_inert_host_residual",
        "source_identity": codec._identity(record["identity"]), "record_sha256": record_sha256,
        "static_record": codec._strict_json(codec._json(record["static_record"]).encode()),
        "residual_sha256": codec._fingerprint(dict(zip(_RESIDUAL_FIELDS, residual, strict=True))),
        "bed_scalars": [list(row) for row in scalars],
        "bed_layout": [[role, dtype, list(shape)] for role, dtype, shape in metadata], "bed_bytes": total,
        "bed_leaves": 23, "largest_bed_leaf_bytes": largest,
        "saved_bed_generation": record["identity"]["bed_generation"],
        "saved_resource_generation": record["identity"]["resource_generation"],
        "native_commit_count": shallow.committed_topography.commit_count,
        "native_commit_time_hex": shallow.committed_topography.last_commit_time_s.hex(),
        "native_time_hex": physical.time_s.hex(), "native_time_kind": type(physical.time_s).__name__,
        "reservation": {"host_data_terms": terms, "required_host_data_bytes": required_host,
                        "inherited_cold_data_bytes": old_reservation["required_host_data_bytes"],
                        "extra_backing_allocation_bytes": extra_backing,
                        "bed_six_root_required_bytes": required_device, "comparison_bytes": comparison,
                        "scratch_status": "DECLARED_UNMEASURED", "memory_fit": False}}
    bridge._budget(codec, limits.max_metadata_bytes, binding, plain_json=True)
    # Pin what was admitted/hashed to what is actually returned. This is a
    # caller-private quiescence protocol, not hostile-writer atomicity.
    if _originals(checkpoint, config, inputs, limits)[-1] != original:
        _fail("admission_source", "original inputs changed during native validation")
    _admission_slots(cold)
    checkpoint._limits(limits)
    bed_limits.__post_init__()
    if (cold._physical is not physical or cold._inputs is not inputs or cold._config is not config
            or cold._limits is not limits or cold._record is not record
            or cold._bundle != bundle or cold._record_path != record_path
            or cold._record_observation is not record_observation or cold._snapshot is not source_snapshot
            or cold._record_sha256 != record_sha256 or record_observation != observation
            or codec._json(record).encode() != raw_record
            or tuple(getattr(limits, f.name) for f in dataclasses.fields(limits)) != limit_values
            or dataclasses.asdict(bed_limits) != record["static_record"]["bed_limits"]
            or any(getattr(physical, key) is not getattr(shallow, key) for key in bed.SHARED_FIELDS)
            or any(getattr(physical, key) is not value for key, value in zip(_RESIDUAL_FIELDS, residual, strict=True))):
        _fail("admission_source", "validated source holders/limits/metadata changed during preparation")
    rewalk = _admission_roots((("physical", physical), ("inputs", inputs), ("config", config)),
                              n.np, codec, limits)
    layout_stamp = _admission_layout_stamp(whole)
    if _admission_layout_stamp(rewalk) != layout_stamp:
        _fail("admission_source", "source array/root layout changed during preparation")
    stamp = (tuple(id(value) for value in (physical, inputs, config, limits, record)),
             original, layout_stamp, codec._json(binding))
    return {"shallow": shallow, "residual": residual, "inputs": inputs, "config": config,
            "limits": limits, "n": n, "codec": codec, "binding": binding, "stamp": stamp,
            "geometry": g, "n_voxels": nv, "mass_resolution_kg": inputs.context.mass_resolution_kg,
            "case_sha256": inputs.case_sha256, "time_s": physical.time_s}



def _admission_counter_record(value):
    from maple.core.backend.arrays import TransferCounters
    if (type(value) is not TransferCounters or set(vars(value)) != set(_COUNTER_FIELDS)
            or tuple(f.name for f in dataclasses.fields(value)) != _COUNTER_FIELDS):
        _fail("admission_counters", "exact native counter holder/layout required")
    record = {key: getattr(value, key) for key in _COUNTER_FIELDS}
    if any(type(v) is not int or v < 0 for v in record.values()):
        _fail("admission_counters", "closed native counter record required")
    return record


def _admission_counter_delta(before, after):
    if (type(before) is not dict or type(after) is not dict
            or set(before) != set(_COUNTER_FIELDS) or set(after) != set(_COUNTER_FIELDS)):
        _fail("admission_counters", "closed counter fields required")
    if any(type(before[k]) is not int or type(after[k]) is not int
           or not 0 <= before[k] <= after[k] for k in _COUNTER_FIELDS):
        _fail("admission_counters", "native counters must be monotonic plain integers")
    return {key: after[key] - before[key] for key in _COUNTER_FIELDS}


def _admission_expected_delta(before, after, **expected):
    if set(expected) - set(_COUNTER_FIELDS) or any(type(v) is not int or v < 0 for v in expected.values()):
        _fail("admission_counters", "unknown or non-integer phase expectation")
    if _admission_counter_delta(before, after) != {key: expected.get(key, 0) for key in _COUNTER_FIELDS}:
        _fail("admission_counters", "native phase counter delta differs from owning source contract")


def _admission_owner_counts(owner):
    if type(owner._counts) is not dict or len(owner._counts) != len(_OWNER_COUNTER_FIELDS):
        _fail("admission_counters", "closed owner-private counter layout required")
    result = dict(owner._counts)
    if set(result) != set(_OWNER_COUNTER_FIELDS) or any(type(v) is not int or v < 0 for v in result.values()):
        _fail("admission_counters", "closed owner-private counter record required")
    return result


class _DeviceBedPhysical:
    """Inert nominal private carrier, never a native mixed-backend physical state.

    It retains no cold ticket/full physical/six HOST bed roots. It offers no
    runner/event/continuation API. Source/device/fit authority is external.
    """
    __slots__ = ("_owner", "_residual", "_inputs", "_config", "_binding", "_mint", "__weakref__")

    def __new__(cls, *args, **kwargs):
        _fail("admission_ticket", "only actual private device admission creates this carrier")

    def diagnostics(self):
        import json
        if type(self) is not _DeviceBedPhysical or getattr(self, "_mint", None) is not _ADMISSION_MINT:
            _fail("admission_ticket", "unminted private bed carrier")
        return json.loads(json.dumps({"binding": self._binding, "released": self._owner is None,
            "scope": "device_bed_plus_complete_inert_host_residual", "device_restore": False,
            "gpu_qualified": False, "memory_fit": False, "restart_supported": False,
            "combined_restart": False, "runner_ready": False}, sort_keys=True, allow_nan=False))

    def _release(self):
        if type(self) is not _DeviceBedPhysical or getattr(self, "_mint", None) is not _ADMISSION_MINT:
            _fail("admission_ticket", "unminted private bed carrier")
        self._owner = self._residual = self._inputs = self._config = None


def _admission_backend():
    import sys
    from types import ModuleType
    cp = sys.modules.get("cupy")
    if (type(cp) is not ModuleType or cp.__name__ != "cupy"
            or getattr(getattr(cp, "ndarray", None), "__module__", "").split(".")[0] != "cupy"):
        _fail("admission_backend", "already imported genuine CuPy required; no backend injection or fallback")
    return cp


def _admission_device_scope(cp):
    import threading
    return (threading.get_ident(), int(cp.cuda.runtime.getDevice()), int(cp.cuda.get_current_stream().ptr))


def _convert_cold_bed(cold, bed_limits, evidence):
    """Ordinary Exception refusals pass through the public detaching boundary."""
    plan = converted = owner = leaves = shallow = post = result = None
    read_counters = None
    snapshots = evidence["native_snapshots"]
    try:
        _admission_slots(cold)
        cp = _admission_backend()
        from .dense_device_bed import DeviceDenseBedOwner
        from .dense_device_runner import _verify_source_copy
        from maple.core.backend import array_leaves, read_transfer_counters, to_device_tree
        read_counters = read_transfer_counters
        plan = _prepare_bed_admission(cold, bed_limits)
        source_scope = _admission_device_scope(cp)  # Actual protected caller established context before entry.
        shallow, np = plan["shallow"], plan["n"].np
        total, nleaves = plan["binding"]["bed_bytes"], plan["binding"]["bed_leaves"]
        snapshots.append(_admission_counter_record(read_transfer_counters()))
        evidence["phase"] = "upload"
        converted = to_device_tree(shallow, cp)
        snapshots.append(_admission_counter_record(read_transfer_counters()))
        _admission_expected_delta(snapshots[0], snapshots[1], host_to_device=nleaves, host_to_device_bytes=total)
        leaves = tuple(array_leaves(converted))
        if (len(leaves) != nleaves or any(type(a) is not cp.ndarray or not a.dtype.isnative
                or int(a.device.id) != source_scope[1] for a in leaves)
                or _admission_device_scope(cp) != source_scope):
            _fail("admission_copy", "every converted leaf must be genuine native-order same-device CuPy")
        evidence["phase"] = "raw_verification"
        copied = _verify_source_copy(cp, np, [(shallow, converted)])
        if (type(copied) is not dict or set(copied) != {"verified_device_leaves", "downloaded_bytes"}
                or any(type(v) is not int for v in copied.values())
                or copied != {"verified_device_leaves": nleaves, "downloaded_bytes": total}):
            _fail("admission_copy", "all23 uploaded native bed leaves must be verified exactly")
        evidence["source_copy"] = dict(copied)
        snapshots.append(_admission_counter_record(read_transfer_counters()))
        _admission_expected_delta(snapshots[1], snapshots[2], device_to_host=nleaves, device_to_host_bytes=total)
        evidence["phase"] = "owner_construction"
        owner = DeviceDenseBedOwner.from_device_bed(converted, geometry=plan["geometry"],
            mass_resolution_kg=plan["mass_resolution_kg"], n_voxels=plan["n_voxels"], n_classes=6,
            case_sha256=plan["case_sha256"], time_s=plan["time_s"], limits=bed_limits)
        evidence["owner_returned"] = True
        evidence["owner_private_counts"] = _admission_owner_counts(owner)
        snapshots.append(_admission_counter_record(read_transfer_counters()))
        _admission_expected_delta(snapshots[2], snapshots[3], device_to_host=1,
                                  device_to_host_bytes=16 * nleaves, scalar_reads=31)
        # Clear ALL converted source references before the final private comparison.
        converted = leaves = shallow = None
        plan["shallow"] = None
        evidence["phase"] = "owner_verify"
        owner.verify()
        evidence["owner_private_counts"] = _admission_owner_counts(owner)
        snapshots.append(_admission_counter_record(read_transfer_counters()))
        _admission_expected_delta(snapshots[3], snapshots[4], device_to_host=1, device_to_host_bytes=8 * nleaves)
        if evidence["owner_private_counts"] != {"d2d_enqueued_arrays": 2 * nleaves,
                "d2d_enqueued_bytes": 2 * total, "clone_attempts": 2 * nleaves,
                "packet_attempts": 2, "packets_completed": 2}:
            _fail("admission_counters", "owner-private clone/packet counters differ")
        evidence["phase"] = "source_recheck"
        post = _prepare_bed_admission(cold, bed_limits)
        if post["stamp"] != plan["stamp"] or _admission_device_scope(cp) != source_scope:
            _fail("admission_source", "source identities/layout/device scope changed during admission")
        if (type(owner) is not DeviceDenseBedOwner or owner.generation != 0
                or type(owner.time_s) is not type(plan["time_s"]) or owner.time_s.hex() != plan["time_s"].hex()):
            _fail("admission_owner", "new local generation/time differs from preserved physical clock")
        _admission_expected_delta(snapshots[4], _admission_counter_record(read_counters()))
        binding = plan["binding"]
        binding.update(owner_local_generation=0, device_ordinal=source_scope[1], stream_pointer=source_scope[2],
            owner_local_identity=id(owner), accounting={"native_snapshots": list(snapshots),
                "native_delta": _admission_counter_delta(snapshots[0], snapshots[4]),
                "owner_private_counts": dict(evidence["owner_private_counts"]),
                "compact_packets": {"constructor_flags": 2 * nleaves, "verify_flags": nleaves,
                                    "float64_bytes": 24 * nleaves}})
        from . import dense_checkpoint as bridge
        bridge._budget(plan["codec"], plan["limits"].max_metadata_bytes, binding, plain_json=True)
        result = object.__new__(_DeviceBedPhysical)
        result._owner, result._residual = owner, plan["residual"]
        result._inputs, result._config, result._binding = plan["inputs"], plan["config"], binding
        result._mint = _ADMISSION_MINT
        evidence["phase"] = "returned_private_no_authority"
        return result
    finally:
        # Native counter read is metadata only. Constructor failure can precede
        # returning its owner, so unobserved D2D counts MUST remain unknown.
        if read_counters is not None:
            try:
                evidence["last_native_counters"] = _admission_counter_record(read_counters())
            except Exception as counter_error:
                evidence["counter_capture_failure"] = type(counter_error).__name__[:128]
        plan = converted = owner = leaves = shallow = post = result = None


def admit_device_bed_from_cold(cold, *, bed_limits):
    """Private inert conversion only; caller retains cold until real post-check.

    No source/UUID/PCI/grant/device-fit authority is conferred. No backend
    injection, fresh bed, native clock/commit reset or event API is supported.
    Failure raises outside the original exception handler, retaining metadata
    rather than partial device objects through its cause/context/traceback.
    """
    evidence = {"phase": "preflight", "native_snapshots": [], "owner_returned": False,
                "owner_private_counts": None, "source_copy": None,
                "scope": "inert_bed_only_not_device_or_restart_authority"}
    result = failure = None
    try:
        result = _convert_cold_bed(cold, bed_limits, evidence)
    except Exception as exc:
        code, detail = getattr(exc, "code", None), getattr(exc, "detail", None)
        code = code if type(code) is str and len(code) <= 128 else "admission_failure"
        detail = detail if type(detail) is str and len(detail) <= 1024 else type(exc).__name__[:128]
        failure = (code, detail)
    if failure is not None:
        result = None
        raise DeviceBedAdmissionError(failure[0], failure[1], evidence) from None
    return result
