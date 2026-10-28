"""Closed Small GPU accepted-boundary codec and existing-store adapter.

Native windowed snapshots own sediment serialization.  This companion owns
only the missing MASY state.  Files contain data, never Python capabilities.
The qualification claim remains bounded/observational; opaque allocations are
not supplied by this codec.  All Native imports and device operations are lazy.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import io
import json
import math
import os
import re
import stat
import sys
import threading
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PosixPath
from types import ModuleType, SimpleNamespace

from . import dense_original_inputs as oi
from . import dense_runner_checkpoint as boundary
from . import dense_boundary_store as store
from .clock import ClockLedger, decode_time, encode_time

SCHEMA = "masy.windowed_device_boundary_image.v1"
PROFILE = "small_C16_K1_V6_W3_gpu_accepted_boundary_v1"
STORE_SCHEMA = "masy.windowed_device_boundary_store.v1"
ORIGINAL_SCHEMA = "masy.windowed_device_originals.v1"
PHYSICAL_SCHEMA = "masy.windowed_device_native_snapshot.v1"
RUNTIME_SCHEMA = STORE_SCHEMA + ".runtime"
CLAIM = "bounded physical qualification; known subset only; no allocation proof"
_KEY = object()
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_DTYPES = {"|b1": 1, "|u1": 1, "|i1": 1, "<i4": 4, "<i8": 8,
           "<u4": 4, "<u8": 8, "<f4": 4, "<f8": 8}
_BED = ("voxel_column", "active_layer", "water", "ledger", "committed_topography",
        "sediment_availability", "reconciliation_baseline")
_COUNTERS = ("gate_count", "logical_work_debited", "logical_read_bytes_debited",
             "versions_reserved", "h2d_bytes_reserved", "d2h_bytes_reserved",
             "scalar_reads_reserved", "maximum_host_predicate", "maximum_device_predicate")
_INPUT_FIELDS = ("case", "context", "statics", "host_config", "model_config", "record_limits",
    "profile_source", "calibration", "label", "terrain", "control", "runner_limits", "resource_limits",
    "bed_limits", "schedule", "trajectory", "static_parts", "static_record", "static_sha256",
    "static_parts_sha256", "native_audit")
_CURRENT_FIELDS = ("auxiliary", "pool", "soil", "bookkeeping", "routing")
_ORIGINAL_FIELDS = ("inputs", "initial_pool", "initial_soil", "inventory0")


class WindowedDeviceCheckpointError(ValueError):
    def __init__(self, code, detail):
        self.code, self.detail = code, str(detail)[:2048]
        super().__init__(f"{code}: {self.detail}")


def _fail(code, detail):
    raise WindowedDeviceCheckpointError(code, detail)


def _keys(value, names, where):
    if type(value) is not dict or set(value) != set(names):
        _fail("closed_record", where)


def _integer(value, high, where, low=0):
    if type(value) is not int or not low <= value <= high:
        _fail("integer", where)
    return value


def _digest(value, where):
    if type(value) is not str or _SHA.fullmatch(value) is None:
        _fail("identity", where)
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _plain(value, limits):
    store._plain_metadata(value, limits.max_metadata_bytes, "GPU image",
                          max_nodes=limits.max_nodes, max_depth=limits.max_depth)
    return value


def _path(value):
    text = str(value)
    if (not 1 <= len(text) <= 4096 or "\0" in text or not Path(text).is_absolute()
            or str(Path(text)) != text or os.path.normpath(text) != text or ".." in Path(text).parts):
        _fail("path", "canonical absolute path required")
    return text


def _finite_hex(value):
    if type(value) is not str or len(value) > 128:
        _fail("float", "bounded canonical float hex required")
    try:
        number = float.fromhex(value)
    except (ValueError, OverflowError):
        _fail("float", "invalid float")
    if not math.isfinite(number) or number.hex() != value:
        _fail("float", "noncanonical/nonfinite float")
    return number


def _class_table():
    # Both tables are code-owned Source rosters, not class names from a file.
    result = {module + ":" + name: (fields, enum)
              for module, name, fields, enum in oi._TYPES}
    for module, name, fields in _EXTRA_TYPES:
        result[module + ":" + name] = (fields, False)
    return result


def _class(tag):
    table = _class_table()
    if type(tag) is not str or tag not in table:
        _fail("class", "class outside frozen Source roster")
    module, name = tag.split(":")
    cls = getattr(importlib.import_module(module), name)
    names, enum = table[tag]
    if not enum and (not dataclasses.is_dataclass(cls)
            or tuple(f.name for f in dataclasses.fields(cls)) != names
            or any(not f.init for f in dataclasses.fields(cls))):
        _fail("class_schema", tag)
    return cls, names, enum


class _Encoder:
    """Bound metadata first; no download/hash/copy until complete admission."""
    def __init__(self, limits, cp):
        import numpy as np
        self.limits, self.cp, self.np = limits, cp, np
        self.seen, self.active, self.arrays, self.rows = {}, set(), [], []
        self.nodes, self.numeric = 0, 0

    def node(self, value, depth=0):
        import enum
        self.nodes += 1
        if self.nodes > self.limits.max_nodes or depth > self.limits.max_depth:
            _fail("metadata_cap", "typed input nodes/depth")
        kind = type(value)
        if value is None or kind in (bool, str, int):
            if kind is str and len(value) > 4096 or kind is int and value.bit_length() > 128:
                _fail("scalar_cap", "typed scalar")
            return {"t": "scalar", "v": value}
        if kind is float:
            if not math.isfinite(value):
                _fail("float", "nonfinite input")
            return {"t": "float", "v": value.hex()}
        if value is self.np or value is self.cp:
            return {"t": "module", "v": "numpy" if value is self.np else "cupy"}
        if kind is PosixPath:
            return {"t": "path", "v": _path(value)}
        if kind in (self.np.int32, self.np.int64, self.np.float32, self.np.float64, self.np.bool_):
            return {"t": "numpy_scalar", "dtype": value.dtype.str,
                    "v": self.node(value.item(), depth + 1)}
        tag = kind.__module__ + ":" + kind.__qualname__
        if isinstance(value, enum.Enum):
            cls, _names, is_enum = _class(tag)
            if kind is not cls or not is_enum:
                _fail("class", "unknown enum")
            return {"t": "enum", "c": tag, "v": self.node(value.value, depth + 1)}
        key = id(value)
        if key in self.active:
            _fail("cycle", "cyclic input graph unsupported")
        if key in self.seen:
            return {"t": "ref", "v": self.seen[key]}
        index = len(self.seen)
        self.seen[key] = index
        self.active.add(key)
        try:
            if kind is self.np.ndarray or kind is self.cp.ndarray:
                dtype, shape = value.dtype.str, tuple(value.shape)
                if (dtype not in _DTYPES or not value.flags.c_contiguous or value.ndim > 4
                        or any(type(d) is not int or not 0 <= d <= self.limits.max_cohorts for d in shape)):
                    _fail("array_layout", "closed C numeric/bool leaf")
                size = math.prod(shape) * _DTYPES[dtype]
                if size != value.nbytes or self.numeric + size > self.limits.max_numeric_bytes:
                    _fail("numeric_cap", "complete companion payload before download")
                row = dict(dtype=dtype, shape=list(shape), offset=self.numeric, bytes=size,
                           backend="numpy" if kind is self.np.ndarray else "cupy")
                self.numeric += size
                array_index = len(self.arrays)
                self.arrays.append(value); self.rows.append(row)
                return {"t": "array", "i": index, "v": array_index}
            if kind in (tuple, list, dict):
                if len(value) > self.limits.max_nodes:
                    _fail("metadata_cap", "container count")
                if kind is dict:
                    if any(type(k) is not str or len(k) > 4096 for k in value):
                        _fail("dictionary", "bounded string keys only")
                    children = [[k, self.node(v, depth + 1)] for k, v in value.items()]
                else:
                    children = [self.node(v, depth + 1) for v in value]
                return {"t": kind.__name__, "i": index, "v": children}
            cls, names, is_enum = _class(tag)
            if kind is not cls or is_enum:
                _fail("class", "exact registered holder required")
            if hasattr(value, "__dict__") and set(vars(value)) != set(names):
                _fail("class_fields", tag)
            return {"t": "class", "i": index, "c": tag,
                    "v": [[name, self.node(getattr(value, name), depth + 1)] for name in names]}
        finally:
            self.active.remove(key)


def _wire_preflight(wire, limits):
    """Pure closed data validation, before class construction or expansion."""
    _keys(wire, ("root", "arrays", "numeric_bytes"), "wire")
    _plain(wire, limits)
    total = _integer(wire["numeric_bytes"], limits.max_numeric_bytes, "numeric bytes")
    rows = wire["arrays"]
    if type(rows) is not list or len(rows) > limits.max_nodes:
        _fail("array_cap", "typed array rows")
    offset = 0
    for row in rows:
        _keys(row, ("dtype", "shape", "offset", "bytes", "backend"), "array row")
        if row["dtype"] not in _DTYPES or row["backend"] not in ("numpy", "cupy"):
            _fail("array_layout", "dtype/backend")
        shape = row["shape"]
        if type(shape) is not list or len(shape) > 4:
            _fail("array_layout", "shape")
        for d in shape:
            _integer(d, limits.max_cohorts, "dimension")
        size = math.prod(shape) * _DTYPES[row["dtype"]]
        if type(row["offset"]) is not int or row["offset"] != offset or type(row["bytes"]) is not int or row["bytes"] != size:
            _fail("array_layout", "exact contiguous offset/shape bytes")
        offset += size
        if offset > total:
            _fail("numeric_cap", "array offsets")
    if offset != total:
        _fail("numeric_cap", "numeric total")
    complete, active, array_used, count = set(), set(), set(), [0]
    def visit(node, depth):
        count[0] += 1
        if count[0] > limits.max_nodes or depth > limits.max_depth or type(node) is not dict:
            _fail("metadata_cap", "typed tree")
        tag = node.get("t")
        if tag in ("scalar", "float", "module", "path", "ref"):
            _keys(node, ("t", "v"), "scalar/ref node")
            v = node["v"]
            if tag == "scalar":
                if v is not None and type(v) not in (str, bool, int):
                    _fail("scalar", "plain scalar")
                if type(v) is int and v.bit_length() > 128 or type(v) is str and len(v) > 4096:
                    _fail("scalar", "scalar cap")
            elif tag == "float": _finite_hex(v)
            elif tag == "module":
                if v not in ("numpy", "cupy"): _fail("module", "backend")
            elif tag == "path": _path(v)
            elif type(v) is not int or v not in complete:
                _fail("reference", "forward/cyclic/unknown reference")
            return
        if tag == "numpy_scalar":
            _keys(node, ("t", "dtype", "v"), "NumPy scalar")
            if node["dtype"] not in ("<i4", "<i8", "<f4", "<f8", "|b1"):
                _fail("scalar", "NumPy scalar dtype")
            child = node["v"]
            if type(child) is not dict or child.get("t") not in ("scalar", "float"):
                _fail("scalar", "NumPy scalar must contain a scalar value")
            value = child.get("v")
            if ((node["dtype"] in ("<i4", "<i8") and (child["t"] != "scalar" or type(value) is not int
                    or not -(1 << (8 * _DTYPES[node["dtype"]] - 1)) <= value < (1 << (8 * _DTYPES[node["dtype"]] - 1))))
                    or (node["dtype"] == "|b1" and (child["t"] != "scalar" or type(value) is not bool))
                    or (node["dtype"] in ("<f4", "<f8") and child["t"] != "float")):
                _fail("scalar", "exact NumPy scalar kind/range")
            visit(node["v"], depth + 1); return
        if tag == "enum":
            _keys(node, ("t", "c", "v"), "enum")
            if node["c"] not in _class_table() or not _class_table()[node["c"]][1]:
                _fail("class", "enum roster")
            visit(node["v"], depth + 1); return
        keys = ("t", "i", "c", "v") if tag == "class" else ("t", "i", "v")
        _keys(node, keys, "holder/container node")
        i = _integer(node["i"], limits.max_nodes, "node identity")
        if i in complete or i in active:
            _fail("reference", "duplicate identity")
        active.add(i)
        if tag == "array":
            a = _integer(node["v"], len(rows) - 1, "array index")
            if a in array_used: _fail("array_layout", "duplicate array definition")
            array_used.add(a)
        elif tag in ("tuple", "list", "dict", "class"):
            children = node["v"]
            if type(children) is not list or len(children) > limits.max_nodes:
                _fail("metadata_cap", "children")
            if tag in ("dict", "class"):
                if any(type(v) is not list or len(v) != 2 or type(v[0]) is not str for v in children):
                    _fail("class_fields", "named children")
                names = tuple(v[0] for v in children)
                if len(set(names)) != len(names): _fail("class_fields", "duplicate key")
                if tag == "class" and (node["c"] not in _class_table()
                        or names != _class_table()[node["c"]][0] or _class_table()[node["c"]][1]):
                    _fail("class_fields", "frozen Source field order")
                children = [v[1] for v in children]
            for child in children: visit(child, depth + 1)
        else:
            _fail("node_kind", "unknown wire node")
        active.remove(i); complete.add(i)
    visit(wire["root"], 0)
    if array_used != set(range(len(rows))):
        _fail("array_layout", "unreferenced array")
    return total


def _wire_root(wire, names, where):
    root = wire["root"]
    if root.get("t") != "dict" or tuple(v[0] for v in root["v"]) != names:
        _fail("closed_record", where)


def _original_preflight(record, source, environment, limits):
    _keys(record, ("schema", "wire", "payload", "profile", "source_identity",
                  "environment_sha256", "manifest_sha256"), "original capsule")
    if (record["schema"] != ORIGINAL_SCHEMA or record["profile"] != PROFILE
            or _json(record["source_identity"]) != _json(source)
            or record["environment_sha256"] != environment):
        _fail("original_identity", "original profile/Source/environment")
    _wire_preflight(record["wire"], limits)
    _wire_root(record["wire"], _ORIGINAL_FIELDS, "original root")
    first = record["wire"]["root"]["v"][0][1]
    if first.get("t") != "dict" or tuple(v[0] for v in first["v"]) != _INPUT_FIELDS:
        _fail("original_inputs", "complete ordered original inputs")


def _decode(wire, payload, limits, cp, *, host=False, inputs_only=False):
    import numpy as np
    from maple.core.backend import to_device
    total = _wire_preflight(wire, limits)
    if type(payload) is not np.ndarray or payload.dtype != np.uint8 or payload.shape != (total,):
        _fail("payload", "exact admitted raw byte vector")
    seen = {}
    def node(n):
        t, v = n["t"], n["v"]
        if t == "scalar": return v
        if t == "float": return _finite_hex(v)
        if t == "module": return cp if v == "cupy" and not host else np
        if t == "path": return Path(v)
        if t == "ref": return seen[v]
        if t == "numpy_scalar": return np.dtype(n["dtype"]).type(node(v))
        if t == "enum": return _class(n["c"])[0](node(v))
        if t == "array":
            r = wire["arrays"][v]
            value = payload[r["offset"]:r["offset"] + r["bytes"]].view(np.dtype(r["dtype"])).reshape(r["shape"]).copy(order="C")
            if r["backend"] == "cupy" and not host:
                value = to_device(value, cp)
            else:
                value.flags.writeable = False
        elif t in ("list", "tuple"):
            value = [node(x) for x in v]
            if t == "tuple": value = tuple(value)
        elif t == "dict": value = {k: node(x) for k, x in v}
        else: value = _class(n["c"])[0](**{k: node(x) for k, x in v})
        seen[n["i"]] = value
        return value
    root = wire["root"]
    if inputs_only:
        if root["t"] != "dict" or root["v"][0][0] != "inputs":
            _fail("original_inputs", "inputs must be the first independent original root")
        root = root["v"][0][1]
    return node(root)


def _environment(cp):
    import numpy as np
    return dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                cupy=cp.__version__, byteorder=sys.byteorder)


def _budget_identity(record):
    """Stable scientific/Source operands; actual role references stay evidence."""
    result = copy.deepcopy(record)
    result.pop("current_runtime_ref")
    result["factory_source"] = {k: result["factory_source"][k] for k in ("size", "sha256")}
    result["source_bindings"] = [dict(module=r["module"], size=r["size"], sha256=r["sha256"])
                                 for r in result["source_bindings"]]
    return result


class _Authority:
    """In-process handoff AFTER Root consumed this actual role's launch grant.

    This is not a launch/grant issuer. Root's existing protected worker owns
    fresh-role/expiry/UUID/runtime/cold-import verification. Normal owning
    constructors still run the full genuine qualification runtime guards.
    """
    __slots__ = ("budget", "cp", "record", "pid", "thread", "device", "stream")

    def __init__(self, budget):
        from .dense_device_runner import _genuine_cupy
        from .dense_device_bed import _CupyOps
        from maple_syrup import windowed_shared_bed_device as shared
        self.cp = _genuine_cupy()
        if type(budget) is not shared.DeviceWaterQualificationBudget:
            _fail("authority", "genuine fresh qualification budget required")
        budget.validated()
        # Source, grant bytes and loaded origins are independently rechecked;
        # this does not reconsume Root's outer one-use launch authority.
        for ref in (budget.current_runtime_ref, budget.factory_source): shared._read_reference(ref)
        if shared._read_reference(budget.grant) != shared.qualification_budget_bytes(budget):
            _fail("authority", "actual complete product budget bytes differ")
        for name, ref in budget.source_bindings:
            module = sys.modules.get(name)
            origin = None if type(module) is not ModuleType else vars(module).get("__file__")
            if type(origin) is not str or Path(origin).resolve() != Path(ref.path).resolve():
                _fail("source_origin", name)
            shared._read_reference(ref)
        self.budget = budget
        evidence = shared.qualification_budget_record(budget)
        self.record = dict(schema=RUNTIME_SCHEMA, source_identity=_budget_identity(evidence),
                           environment=_environment(self.cp), evidence=evidence,
                           grant=dict(path=budget.grant.path, size=budget.grant.size,
                                      sha256=budget.grant.sha256))
        self.record["environment_sha256"] = _sha(self.record["environment"])
        ops = _CupyOps(self.cp)
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.device, self.stream = ops.current_device(), ops.current_stream()

    def __reduce_ex__(self, protocol):
        raise TypeError("execution authority is process-local")

    def check(self, saved):
        from .dense_device_bed import _CupyOps
        from maple_syrup import windowed_shared_bed_device as shared
        _keys(saved, ("schema", "source_identity", "environment", "environment_sha256", "evidence", "grant"), "runtime")
        ops = _CupyOps(self.cp)
        if (self.pid != os.getpid() or self.thread != threading.get_ident()
                or self.device != ops.current_device() or self.stream != ops.current_stream()):
            _fail("authority_scope", "actual process/thread/device/stream changed")
        self.budget.validated()
        current = shared.qualification_budget_record(self.budget)
        if (saved["schema"] != RUNTIME_SCHEMA or _json(saved["source_identity"]) != _json(_budget_identity(current))
                or saved["environment"] != _environment(self.cp)
                or saved["environment_sha256"] != _sha(saved["environment"])
                or _json(_budget_identity(saved["evidence"])) != _json(saved["source_identity"])):
            _fail("runtime_identity", "original Source/profile/caps/UUID/environment differ")
        # Do not demand the OLD role still be unexpired, change its bytes or
        # silently reuse it. It is a bounded immutable provenance record only.
        for ref in (self.budget.current_runtime_ref, self.budget.factory_source): shared._read_reference(ref)
        if shared._read_reference(self.budget.grant) != shared.qualification_budget_bytes(self.budget):
            _fail("authority", "fresh role grant changed")
        for name, ref in self.budget.source_bindings:
            module = sys.modules.get(name)
            origin = None if type(module) is not ModuleType else vars(module).get("__file__")
            if type(origin) is not str or Path(origin).resolve() != Path(ref.path).resolve():
                _fail("source_origin", name)
            shared._read_reference(ref)
        return copy.deepcopy(saved["source_identity"]), saved["environment_sha256"]


def _read_file(path, cap):
    return store._read(path, cap)


def _file_row(path, cap):
    """Stream a pinned ordinary file; never allocate a complete deep payload."""
    p = Path(path); parent = oi._open_dir(p.parent)
    fd = None
    try:
        info = os.stat(p.name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink not in (1, 2) or not 0 < info.st_size <= cap:
            _fail("file", "ordinary bounded file required")
        fd = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent)
        held = os.fstat(fd)
        if oi._stat_id(info) != oi._stat_id(held): _fail("file_changed", str(p))
        digest, count = hashlib.sha256(), 0
        while True:
            raw = os.read(fd, min(1 << 20, info.st_size + 1 - count))
            if not raw: break
            count += len(raw)
            if count > info.st_size: _fail("file_changed", str(p))
            digest.update(raw)
        if count != info.st_size or oi._stat_id(os.fstat(fd)) != oi._stat_id(held):
            _fail("file_changed", str(p))
        if oi._stat_id(os.stat(p.name, dir_fd=parent, follow_symlinks=False)) != oi._stat_id(held):
            _fail("file_changed", str(p))
        return dict(path=_path(p), size=count, sha256=digest.hexdigest())
    finally:
        if fd is not None: os.close(fd)
        os.close(parent)


def _check_file(row, cap):
    _keys(row, ("path", "size", "sha256"), "file reference")
    _path(row["path"]); _integer(row["size"], cap, "file size", 1); _digest(row["sha256"], "file hash")
    if _file_row(row["path"], row["size"]) != row:
        _fail("file_changed", row["path"])


def _write_wire(directory, material, *, schema, limits, authority, extra):
    """One admitted uint8 NPY plus closed manifest fits old inventory roles."""
    import numpy as np
    from maple.core.backend import to_host
    directory = Path(directory)
    encoder = _Encoder(limits, authority.cp)
    tree = encoder.node(material)
    wire = dict(root=tree, arrays=encoder.rows, numeric_bytes=encoder.numeric)
    _wire_preflight(wire, limits)
    _boundary_admit(encoder.numeric, encoder.numeric, limits, authority)
    directory.mkdir(mode=0o700)
    payload = np.empty(encoder.numeric, dtype=np.uint8)
    for array, row in zip(encoder.arrays, encoder.rows, strict=True):
        host = np.asarray(to_host(array))
        if host.dtype.str != row["dtype"] or list(host.shape) != row["shape"] or not host.flags.c_contiguous:
            _fail("array_changed", "download metadata changed")
        payload[row["offset"]:row["offset"] + row["bytes"]] = host.reshape(-1).view(np.uint8).reshape(-1)
    path = directory / "a_0000.npy"
    with path.open("xb") as handle:
        np.save(handle, payload, allow_pickle=False); handle.flush(); os.fsync(handle.fileno())
    os.chmod(path, 0o400)
    row = _file_row(path, limits.max_serialized_bytes)
    row.pop("path")
    record = dict(schema=schema, wire=wire, payload=row, **extra)
    if schema == ORIGINAL_SCHEMA:
        _original_preflight(dict(record, manifest_sha256="0" * 64),
                            record["source_identity"], record["environment_sha256"], limits)
    _plain(record, limits)
    record["manifest_sha256"] = _sha(record)
    store._sealed_write(directory, "manifest.json", record, limits.max_metadata_bytes)
    return record


def _read_wire(directory, *, expected_file_sha=None, expected_seal=None, schema, limits, expand):
    import numpy as np
    directory = Path(directory)
    raw = _read_file(directory / "manifest.json", limits.max_metadata_bytes)
    if expected_file_sha is not None and hashlib.sha256(raw).hexdigest() != expected_file_sha:
        _fail("manifest_hash", "raw manifest hash")
    record = oi._strict_json(raw, limits)
    if type(record) is not dict or record.get("schema") != schema:
        _fail("schema", "exact companion/original schema")
    body = dict(record); seal = body.pop("manifest_sha256", None)
    if _sha(body) != seal or expected_seal is not None and seal != expected_seal:
        _fail("manifest_hash", "manifest seal")
    total = _wire_preflight(record["wire"], limits)
    _keys(record["payload"], ("size", "sha256"), "raw payload")
    row = dict(path=str(directory / "a_0000.npy"), **record["payload"])
    _check_file(row, limits.max_serialized_bytes)
    with (directory / "a_0000.npy").open("rb") as handle:
        shape, fortran, dtype = oi._read_npy_header(handle, limits)
        if dtype != np.uint8 or shape != (total,) or fortran or handle.tell() + total != row["size"]:
            _fail("payload", "exact bounded byte-vector header before expansion")
        payload = np.fromfile(handle, dtype=np.uint8, count=total) if expand else None
    _check_file(row, limits.max_serialized_bytes)
    return record, payload, raw


def _boundary_admit(host, device, limits, authority, *, retained_host=0, retained_device=0):
    """Positive boundary copy overlap, in addition to unchanged Native gates.

    Nine payload occurrences cover input/download, sealed/readback, two
    simultaneous owners/mirrors and first-error/caller residuals; actual bed
    and resource constructors add their original full Native producers too.
    This prices selected visible bytes only. Opaque providers remain None.
    """
    if any(type(v) is not int or v < 0 for v in (host, device, retained_host, retained_device)):
        _fail("boundary_payload", "positive exact integers")
    # Each complete family is bounded by the admitted aggregate payload, not
    # an operator-supplied multiplier. No ownership-release credit is taken.
    banks = ("caller_input", "parent_input", "parent_accepted", "parent_mirror",
             "readback_input", "readback_accepted", "readback_mirror",
             "current_next_temporary", "first_error")
    host_request = sum(host for _ in banks) + retained_host + (64 << 20) + (2 << 20)
    device_request = sum(device for _ in banks) + retained_device + (256 << 20)
    if host_request > 5 << 30 or device_request > 2 << 30:
        _fail("boundary_payload", "boundary copies exceed original caps")
    from maple.aeolian.cohorts.capacity import default_available_memory_bytes
    from maple.surface.active_layer import bed_exchange_device
    available = default_available_memory_bytes()
    if type(available) is not int or 2 * host_request > available:
        _fail("boundary_headroom", "genuine HOST half-headroom")
    bed_exchange_device._check_device_admission(device_request, device_id=authority.device, safety_fraction=0.5)
    return dict(host_payload_bytes=host_request, device_payload_bytes=device_request,
                positive_payload_banks=banks,
                known_subset_only=True, constructor_allocation=None, external_runtime=None, claim=CLAIM)


def _disk_admit(context, deep_store, *, disk_bytes, entries):
    """The actual Native whole-run inventory also covers this new store.

    Snapshot/companion publication does not write a deep version. Native
    event writers keep their own unchanged deep-write preflights. The common
    store's complete additional reservation is charged to the run root too.
    """
    from maple_syrup.windowed_shared_bed_device import _inventory_disk
    _integer(disk_bytes, 512 << 20, "boundary disk reservation")
    _integer(entries, 4096, "boundary entry reservation")
    total, _store_bytes, count = _inventory_disk(context, Path(deep_store))
    if (total + disk_bytes > context.bridge_limits.max_run_bytes
            or count + entries > context.bridge_limits.max_entries):
        _fail("run_storage", "whole Native run plus complete boundary attempt")


def _snapshot_preflight(path, limits):
    """All Native NPZ headers, member sizes and closed metadata BEFORE load."""
    import numpy as np
    from maple.surface.voxels.windowed_prefix import packet_terms_for
    expected = {
        "window_kg": ((4, 4, 3, 1), "<f8"), "base": ((4, 4), "<i8"),
        "deep_by_class_kg": ((4, 4, 1), "<f8"), "deep_total_kg": ((4, 4), "<f8"),
        "deep_room_kg": ((4, 4), "<f8"),
        "prefix_packet": ((4, 4, packet_terms_for(6, 1)), "<f8"),
        "active_mass_kg": ((4, 4, 1), "<f8"), "available_mass_kg": ((4, 4, 1), "<f8"),
        "bound_mass_kg": ((4, 4, 1), "<f8"), "water_depth_m": ((4, 4), "<f8"),
        "mobile_mass_kg": ((4, 4, 1), "<f8"),
    }
    numeric, meta = 0, None
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        if (len(names) != 12 or len(set(names)) != 12
                or set(names) != {name + ".npy" for name in (*expected, "metadata")}
                or sum(i.file_size for i in infos) > limits.native_max_bytes):
            _fail("native_archive", "closed Native snapshot roster/raw cap")
        for info in infos:
            if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
                _fail("native_archive", "Native uncompressed unencrypted format required")
            with archive.open(info) as handle:
                shape, fortran, dtype = oi._read_npy_header(handle, limits)
                if fortran or len(shape) > 4:
                    _fail("native_header", "Native C array required")
                name = info.filename[:-4]
                if name == "metadata":
                    if shape != () or dtype.kind != "U" or not 0 < dtype.itemsize <= limits.max_metadata_bytes:
                        _fail("native_header", "bounded scalar Unicode metadata")
                    raw = handle.read(dtype.itemsize + 1)
                    if len(raw) != dtype.itemsize:
                        _fail("native_header", "metadata exact bytes")
                    text = str(np.frombuffer(raw, dtype=dtype, count=1)[0])
                    meta = oi._strict_json(text.encode("utf-8"), limits)
                else:
                    if (tuple(shape), dtype.str) != expected[name]:
                        _fail("native_header", name)
                    numeric += math.prod(shape) * dtype.itemsize
                    if handle.tell() + math.prod(shape) * dtype.itemsize != info.file_size:
                        _fail("native_header", "exact Native payload size")
    from maple.surface.voxels.windowed_snapshot import META_KEYS, SCHEMA as native_schema
    _keys(meta, META_KEYS, "Native metadata")
    if meta["schema"] != native_schema or type(meta["time_s"]) not in (int, float):
        _fail("native_metadata", "schema/time")
    encode_time(float(meta["time_s"]))
    _integer(meta["step"], 4, "Native step")
    _integer(meta["generation"], (1 << 63) - 1, "Native generation")
    if meta["n_voxels"] != 6 or type(meta["n_voxels"]) is not int:
        _fail("native_metadata", "voxel count")
    _path(meta["store_dir"]); _digest(meta["manifest_digest"], "Native manifest")
    _digest(meta["header_digest"], "Native header")
    return meta, numeric


def _deep_files(version, limits):
    """Pin existing files, including old versions/orphans; never delete/rollback.

    Later legitimate publication may append files. Every previously pinned
    file remains immutable. Source/native validation still owns version rules.
    """
    root = Path(version.store_dir)
    files, entries, total = [], 1, 0
    descriptor = root / "store.json"
    files.append(_file_row(descriptor, limits.max_metadata_bytes))
    total += files[-1]["size"]
    for directory, pattern in ((root / "chunks", r"[0-9a-f]{64}(?:\.ovl)?\.f64"),
                               (root / "versions", r"[0-9a-f]{64}\.json")):
        held = oi._open_dir(directory)
        try:
            with os.scandir(held) as iterator:
                for entry in iterator:
                    entries += 1
                    if entries > 4096:
                        _fail("deep_entries", "retained Native files exceed original4096")
                    if not re.fullmatch(pattern, entry.name) or not entry.is_file(follow_symlinks=False):
                        _fail("deep_file", "unexpected Native store file")
                    row = _file_row(directory / entry.name, 64 << 20)
                    total += row["size"]
                    if total > 64 << 20:
                        _fail("deep_bytes", "retained Native store exceeds original64MiB")
                    files.append(row)
        finally:
            os.close(held)
    return sorted(files, key=lambda r: r["path"])


def _provenance(runner, source, parent):
    return dict(schema=PHYSICAL_SCHEMA + ".provenance", source_identity_sha256=_sha(source),
                case_sha256=runner._bed.case_sha256, schedule_sha256=runner._schedule.digest,
                accepted_cursor=runner._next, parent_head_token=parent)


def _physical_write(path, runner, handle):
    from maple.surface.voxels import windowed_snapshot as native
    runner_admission(runner, handle._limits.image)
    path = Path(path); path.mkdir(mode=0o700)
    state, limits = runner._bed._accepted.state, handle._limits.image
    rows, _ = runner._bed._describe(state)
    bytes_ = sum(int(a.nbytes) for _, a in rows)
    _boundary_admit(2 * bytes_, bytes_, limits, handle._authority)
    provenance = _provenance(runner, handle._header["source_identity"], handle._state.token)
    native.save_windowed_state_snapshot(state.voxel_column, state.active_layer, runner._bed.time_s,
        path / "native_state.npz", step=runner._next, geometry=runner._bed.context.geometry,
        mass_resolution_kg=runner._bed.context.mass_resolution_kg,
        sediment_availability=state.sediment_availability, water=state.water, provenance=provenance)
    os.chmod(path / "native_state.npz", 0o400)
    meta, numeric = _snapshot_preflight(path / "native_state.npz", limits)
    if meta["provenance"] != provenance:
        _fail("native_provenance", "Native saved metadata changed")
    deep = _deep_files(state.voxel_column.deep_store, limits)
    # Native writes fsync files. Combined HEAD must also retain/sync directory entries.
    store._sync_tree(state.voxel_column.deep_store.store_dir)
    record = dict(schema=PHYSICAL_SCHEMA, snapshot=_file_row(path / "native_state.npz", limits.native_max_bytes),
                  deep_files=deep, metadata=meta, numeric_bytes=numeric,
                  source_identity=handle._header["source_identity"])
    record["manifest_sha256"] = _sha(record)
    store._sealed_write(path, "checkpoint.json", record, min(1 << 20, limits.max_metadata_bytes))
    return dict(format=PHYSICAL_SCHEMA, relative_path=str(path.relative_to(handle._root)),
                manifest_file_sha256=_file_row(path / "checkpoint.json", min(1 << 20, limits.max_metadata_bytes))["sha256"],
                identity=record["manifest_sha256"])


def _reference(ref, fmt):
    _keys(ref, ("format", "relative_path", "manifest_file_sha256", "identity"), "artifact reference")
    if ref["format"] != fmt or type(ref["relative_path"]) is not str:
        _fail("artifact_format", fmt)
    relative = Path(ref["relative_path"])
    if relative.is_absolute() or ".." in relative.parts or str(relative) != ref["relative_path"]:
        _fail("artifact_path", "relative retained child required")
    _digest(ref["manifest_file_sha256"], "artifact bytes"); _digest(ref["identity"], "artifact identity")


def _physical_read(root, ref, limits):
    _reference(ref, PHYSICAL_SCHEMA)
    directory = Path(root) / ref["relative_path"]
    raw = _read_file(directory / "checkpoint.json", min(1 << 20, limits.max_metadata_bytes))
    if hashlib.sha256(raw).hexdigest() != ref["manifest_file_sha256"]:
        _fail("physical_hash", "physical manifest bytes")
    rec = oi._strict_json(raw, limits)
    _keys(rec, ("schema", "snapshot", "deep_files", "metadata", "numeric_bytes", "source_identity", "manifest_sha256"), "physical")
    body = dict(rec); seal = body.pop("manifest_sha256")
    if rec["schema"] != PHYSICAL_SCHEMA or _sha(body) != seal or seal != ref["identity"]:
        _fail("physical_hash", "physical manifest identity")
    if rec["snapshot"]["path"] != str(directory / "native_state.npz"):
        _fail("physical_path", "snapshot must be own child")
    _check_file(rec["snapshot"], limits.native_max_bytes)
    meta, numeric = _snapshot_preflight(rec["snapshot"]["path"], limits)
    if meta != rec["metadata"] or numeric != rec["numeric_bytes"]:
        _fail("physical_metadata", "Native preflight differs")
    if type(rec["deep_files"]) is not list or not 1 <= len(rec["deep_files"]) <= 4096:
        _fail("deep_entries", "bounded retained deep roster")
    seen, total = set(), 0
    for row in rec["deep_files"]:
        _keys(row, ("path", "size", "sha256"), "deep file")
        p, base = Path(_path(row["path"])), Path(meta["store_dir"])
        _integer(row["size"], 64 << 20, "deep file bytes", 1); _digest(row["sha256"], "deep file hash")
        relative = p.relative_to(base) if p.is_relative_to(base) else None
        valid = (relative == Path("store.json") or relative is not None and len(relative.parts) == 2
            and ((relative.parts[0] == "versions" and re.fullmatch(r"[0-9a-f]{64}\.json", relative.name))
                 or (relative.parts[0] == "chunks" and re.fullmatch(r"[0-9a-f]{64}(?:\.ovl)?\.f64", relative.name))))
        if p in seen or not valid:
            _fail("deep_path", "duplicate/foreign deep file")
        seen.add(p); total += row["size"]
        if total > 64 << 20: _fail("deep_bytes", "deep aggregate cap before hash")
    for row in rec["deep_files"]: _check_file(row, 64 << 20)
    required = Path(meta["store_dir"]) / "versions" / (meta["manifest_digest"] + ".json")
    if required not in seen or Path(meta["store_dir"]) / "store.json" not in seen:
        _fail("deep_reference", "current Native descriptor/version not retained")
    return rec


def _inputs(runner):
    from .dense_device_resources import STATIC_NAMES
    ctx = runner._bed.context
    names = tuple(f.name for f in dataclasses.fields(type(ctx)))
    context = {name: getattr(ctx, name) for name in names if name not in ("runtime_bounds", "initial_available_fraction")}
    return dict(case=runner._bed.case_record, context=context,
        statics={name: runner._device_resources.static(name) for name in STATIC_NAMES},
        host_config=runner._host_config, model_config=runner._wind_profile.model_config,
        record_limits=runner._wind_profile.record_limits,
        profile_source=runner._wind_profile.source_identity, calibration=runner._wind_profile.calibration,
        label=runner._wind_profile.label, terrain=runner._water.terrain,
        control=runner._control, runner_limits=runner._limits, resource_limits=runner._resource_limits,
        bed_limits=runner._bed.limits, schedule=runner._schedule.as_record(),
        trajectory=runner._bed._semantics, static_parts=runner._static_parts,
        static_record=runner._static_record, static_sha256=runner._static_sha256,
        static_parts_sha256=runner._static_parts_sha256, native_audit=runner._native_audit)


def _book(runner):
    return dict(next_index=runner._next, n_accepted=runner._n_accepted,
        native_time=encode_time(runner._bed.time_s), ledger=runner._ledger.as_record(),
        inventory0=runner._inventory0, totals=runner._totals, receipts=runner._receipts,
        water_counters=runner._water_counters, water_record=runner._water_record,
        bed_generation=runner._bed.generation, resource_generation=runner._device_resources.generation,
        routing_issue_counter=runner._device_own._routing_generation,
        accepted_bed_label=runner._accepted_bed, accepted_private_label=runner._accepted_private)


def _fingerprint(value, limits, cp):
    from maple.core.backend import to_host
    encoder = _Encoder(limits, cp); root = encoder.node(value)
    wire = dict(root=root, arrays=encoder.rows, numeric_bytes=encoder.numeric)
    _wire_preflight(wire, limits)
    hashes = [hashlib.sha256(to_host(a).tobytes(order="C")).hexdigest() for a in encoder.arrays]
    return _sha(dict(wire=wire, data_hashes=hashes))


def _routing(runner, limits):
    held = runner._routing
    if held is None: return None
    runner._device_own._held_routing(held, runner._water)
    runner._device_own.routing_guard(None, held, runner._water, (4, 4), 1)
    return dict(source=held.source, velocity=held.velocity, phase=held.phase,
        generation=held.generation, terrain_sha256=_fingerprint(held.terrain, limits, runner._device_ops.xp),
        cache_fingerprints=[_fingerprint(value, limits, runner._device_ops.xp)
                            for value in (held.graph, held.network, held.grid)])


def runner_admission(runner, limits):
    from .windowed_device_runner import DeviceWindowedRunner, _process_check
    from maple.aeolian.cohorts.dispatch import mobile_state_is_genuinely_empty
    if type(runner) is not DeviceWindowedRunner or set(vars(runner)) != set(_RUNNER_FIELDS):
        _fail("runner", "exact closed device-windowed owner")
    if (runner._stop_receipt is not None or runner._active_evidence is not None
            or runner._pending_water_counters is not None or runner._bed._active is not None
            or runner._device_resources.active or runner._device_own._stage is not None
            or runner._device_own._prepared is not None):
        _fail("not_quiescent", "accepted idle boundary only")
    boundary.BoundaryImageLimits.__post_init__(limits)
    if limits.max_cells < 16 or limits.max_classes < 1 or limits.max_voxels < 6 or limits.max_events < 4:
        _fail("profile_cap", "closed C16/K1/V6/four-event profile")
    runner._bed.verify(); runner._device_resources.verify()
    runner._bed._validate(runner._bed._accepted.state, runner._bed.time_s)
    if not mobile_state_is_genuinely_empty(runner._device_resources._accepted.pool):
        _fail("mobile", "genuine finalized empty pool required")
    _process_check(runner._bed.context, runner._worker_identity, checkpoint="boundary_checkpoint")
    _integer(runner._next, 4, "cursor")
    if runner._n_accepted != runner._next or len(runner._receipts) != runner._next:
        _fail("cursor", "accepted/receipt counts")
    runner._precheck()
    return runner


def _image_write(directory, runner, physical, handle):
    limits = handle._limits.image
    runner_admission(runner, limits)
    state = runner._bed._accepted.state
    material = dict(auxiliary={name: getattr(state, name) for name in ("ledger", "committed_topography", "reconciliation_baseline")},
                    pool=runner._device_resources._accepted.pool, soil=runner._device_resources._accepted.soil,
                    bookkeeping=_book(runner), routing=_routing(runner, limits))
    rows = boundary._receipt_rows(runner._receipts, limits)
    route = material["routing"]
    extra = dict(profile=PROFILE, source_identity=handle._header["source_identity"],
        environment_sha256=handle._header["environment_sha256"], originals=handle._header["originals"],
        genesis=handle._header["genesis_physical"], physical=physical,
        case_sha256=runner._bed.case_sha256, schedule_sha256=runner._schedule.digest,
        state=dict(next_index=runner._next, n_accepted=runner._n_accepted,
                   native_time=encode_time(runner._bed.time_s), accepted_bed_label=runner._accepted_bed),
        routing=None if route is None else dict(source_sha256=_fingerprint(route["source"], limits, handle._authority.cp)),
        receipts=rows, ledger=runner._ledger.as_record(), water_counters=copy.deepcopy(runner._water_counters))
    rec = _write_wire(directory, material, schema=SCHEMA, limits=limits, authority=handle._authority, extra=extra)
    return dict(format=SCHEMA, relative_path=str(Path(directory).relative_to(handle._root)),
                manifest_file_sha256=_file_row(Path(directory) / "manifest.json", limits.max_metadata_bytes)["sha256"],
                manifest_sha256=rec["manifest_sha256"])


def _summary(manifest):
    state, route = manifest["state"], manifest["routing"]
    return dict(next_index=state["next_index"], time_s=state["native_time"],
                accepted_bed_label=state["accepted_bed_label"],
                n_accepted_events=state["n_accepted"], routing_held=route is not None,
                routing_source_sha256=None if route is None else route["source_sha256"])


def _preflight(manifest, source, environment, limits):
    _keys(manifest, ("schema", "profile", "wire", "payload", "source_identity", "environment_sha256",
        "originals", "genesis", "physical", "case_sha256", "schedule_sha256", "state", "routing", "receipts", "ledger", "water_counters", "manifest_sha256"), "GPU image")
    _plain(manifest, limits)
    if (manifest["schema"] != SCHEMA or manifest["profile"] != PROFILE
            or _json(manifest["source_identity"]) != _json(source)
            or manifest["environment_sha256"] != environment):
        _fail("image_identity", "exact profile/Source/environment")
    _reference(manifest["originals"], ORIGINAL_SCHEMA)
    _digest(manifest["case_sha256"], "case"); _digest(manifest["schedule_sha256"], "schedule")
    for name in ("genesis", "physical"): _reference(manifest[name], PHYSICAL_SCHEMA)
    _keys(manifest["state"], ("next_index", "n_accepted", "native_time", "accepted_bed_label"), "state")
    k = _integer(manifest["state"]["next_index"], 4, "cursor")
    if type(manifest["state"]["n_accepted"]) is not int or manifest["state"]["n_accepted"] != k:
        _fail("cursor", "accepted count")
    ledger = ClockLedger.from_record(manifest["ledger"])
    records = ledger.records()
    end = ledger.epoch if not records else decode_time(records[-1]["native_end"])
    if len(records) != k or encode_time(end) != manifest["state"]["native_time"]:
        _fail("clock", "cursor/ledger/time")
    rows = manifest["receipts"]
    if type(rows) is not list or len(rows) != k:
        _fail("receipt", "exact accepted receipt count")
    used = 0
    for index, row in enumerate(rows):
        _keys(row, ("body", "bytes", "sha256"), "receipt row")
        body = row["body"]; raw = _json(body); used += len(raw)
        if (type(row["bytes"]) is not int or len(raw) != row["bytes"]
                or hashlib.sha256(raw).hexdigest() != row["sha256"]
                or len(raw) > limits.max_receipt_bytes or used > limits.max_receipts_bytes
                or type(body) is not dict or body.get("schema") != "masy.windowed_device_event_receipt.v1"
                or body.get("index") != index or type(body.get("index")) is not int
                or body.get("status") != "accepted" or body.get("code") != "accepted"
                or body.get("clock") != records[index]
                or type(body.get("resolved")) is not dict
                or body.get("kind") != ("wind" if index % 2 == 0 else "rain")):
            _fail("receipt", "canonical accepted event/clock identity")
        resolved = body["resolved"]
        _digest(resolved.get("schedule_sha256"), "resolved schedule")
        edges = resolved.get("resolved_edges")
        if (resolved["schedule_sha256"] != manifest["schedule_sha256"]
                or type(resolved.get("index")) is not int or resolved["index"] != index
                or resolved.get("kind") != body["kind"] or type(edges) is not dict
                or edges.get("native_start") != records[index]["native_start"]):
            _fail("receipt", "original schedule/index/kind/actual clock start join")
    _keys(manifest["water_counters"], _COUNTERS, "water counters")
    for v in manifest["water_counters"].values(): _integer(v, (1 << 63) - 1, "water counter")
    if manifest["routing"] is not None:
        _keys(manifest["routing"], ("source_sha256",), "routing")
        _digest(manifest["routing"]["source_sha256"], "routing source")
    _wire_preflight(manifest["wire"], limits)
    _wire_root(manifest["wire"], _CURRENT_FIELDS, "current root")


def _image_metadata(root, image, limits):
    _keys(image, ("format", "relative_path", "manifest_file_sha256", "manifest_sha256"), "image reference")
    if image["format"] != SCHEMA: _fail("image_format", "GPU image only")
    relative = Path(image["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        _fail("image_path", "relative own image")
    rec, _payload, raw = _read_wire(Path(root) / relative,
        expected_file_sha=image["manifest_file_sha256"], expected_seal=image["manifest_sha256"],
        schema=SCHEMA, limits=limits.image, expand=False)
    _preflight(rec, rec["source_identity"], rec["environment_sha256"], limits.image)
    return rec, raw


class _Artifacts:
    def __init__(self, root, originals, genesis, physical, limits, wrapper_raw):
        self.root, self.limits = Path(root), limits
        self.refs = tuple({r["relative_path"]: r for r in (originals, genesis, physical)}.values())
        self.records, self.identities, self.raw_sizes = {}, {}, {}
        self.wrapper_bytes = len(wrapper_raw)
        self.reservation_bytes = 0
        for ref in self.refs:
            self.hashed(ref)

    def hashed(self, ref):
        if ref["format"] == PHYSICAL_SCHEMA:
            rec = _physical_read(self.root, ref, self.limits)
            size = rec["snapshot"]["size"] + sum(r["size"] for r in rec["deep_files"])
            paths = [dict(path=str(self.root / ref["relative_path"] / "checkpoint.json"),
                          size=len(_json(rec))), rec["snapshot"], *rec["deep_files"]]
        elif ref["format"] == ORIGINAL_SCHEMA:
            _reference(ref, ORIGINAL_SCHEMA)
            rec, _p, _raw = _read_wire(self.root / ref["relative_path"], expected_file_sha=ref["manifest_file_sha256"],
                expected_seal=ref["identity"], schema=ORIGINAL_SCHEMA, limits=self.limits, expand=False)
            size = rec["payload"]["size"]
            paths = [dict(path=str(self.root / ref["relative_path"] / "manifest.json"), size=len(_json(rec))),
                     dict(path=str(self.root / ref["relative_path"] / "a_0000.npy"), size=size)]
        else: _fail("artifact_format", "unknown artifact family")
        self.records[ref["relative_path"]] = (copy.deepcopy(ref), rec)
        for row in paths:
            p = Path(row["path"]); identity = oi._stat_id(p.lstat())
            if str(p) in self.identities and self.identities[str(p)] != identity:
                _fail("artifact_changed", "retained inode/mtime/size changed")
            self.identities[str(p)] = identity
            self.raw_sizes[str(p)] = p.lstat().st_size
        self.reservation_bytes = self.wrapper_bytes + sum(self.raw_sizes.values())
        if size > self.limits.max_global_raw_bytes or self.reservation_bytes > self.limits.max_global_raw_bytes:
            _fail("raw_cap", "retained artifact aggregate before expansion")
        return ref["manifest_file_sha256"]

    def check(self):
        for ref, old in tuple(self.records.values()):
            self.hashed(ref)
            if self.records[ref["relative_path"]][1] != old:
                _fail("artifact_changed", "quiescent retained bytes changed")
        for path, identity in self.identities.items():
            if oi._stat_id(Path(path).lstat()) != identity:
                _fail("artifact_changed", "quiescent inode/mtime/size changed")

    def close(self):
        self.records.clear()


class _RestoreTicket:
    __slots__ = ("key", "inputs", "current", "host_snapshot", "authority", "limits", "manifest", "consumed")
    def __init__(self, key, **values):
        if key is not _KEY: _fail("ticket", "private loader only")
        self.key, self.consumed = key, False
        for name, value in values.items(): setattr(self, name, value)
    def __reduce_ex__(self, protocol):
        raise TypeError("restore ticket cannot be serialized")


def _checked_ticket(ticket):
    if type(ticket) is not _RestoreTicket or ticket.key is not _KEY or ticket.consumed:
        _fail("ticket", "actual unconsumed checked loader ticket required")
    return ticket


class _Prepared:
    __slots__ = ("owner", "evidence")
    def __init__(self, key, owner, evidence):
        if key is not _KEY: _fail("prepared", "private loader only")
        self.owner, self.evidence = owner, copy.deepcopy(evidence)
    def diagnostics(self): return copy.deepcopy(self.evidence)


def _owner(prepared):
    from .windowed_device_runner import DeviceWindowedRunner
    if type(prepared) is not _Prepared or type(prepared.owner) is not DeviceWindowedRunner:
        _fail("prepared", "exact actual GPU loader result")
    return prepared.owner


def _validate_book(inputs, current, manifest, snapshot):
    import numpy as np
    _keys(inputs, _INPUT_FIELDS, "restored inputs")
    _keys(current, _CURRENT_FIELDS, "restored current")
    _keys(current["auxiliary"], ("ledger", "committed_topography", "reconciliation_baseline"), "auxiliary")
    book = current["bookkeeping"]
    _keys(book, ("next_index", "n_accepted", "native_time", "ledger", "inventory0", "totals", "receipts",
        "water_counters", "water_record", "bed_generation", "resource_generation", "routing_issue_counter",
        "accepted_bed_label", "accepted_private_label"), "bookkeeping")
    k = _integer(book["next_index"], 4, "cursor")
    if (book["n_accepted"] != k or type(book["n_accepted"]) is not int
            or book["bed_generation"] != k or type(book["bed_generation"]) is not int
            or book["resource_generation"] != k or type(book["resource_generation"]) is not int
            or book["ledger"] != manifest["ledger"] or book["water_counters"] != manifest["water_counters"]
            or book["native_time"] != manifest["state"]["native_time"]
            or book["accepted_bed_label"] != manifest["state"]["accepted_bed_label"]
            or _json(book["receipts"]) != _json([r["body"] for r in manifest["receipts"]])):
        _fail("bookkeeping", "complete image/book/cursor/generation join")
    time = decode_time(book["native_time"])
    committed = current["auxiliary"]["committed_topography"]
    if (float(snapshot["time_s"]).hex() != time.hex() or snapshot["step"] != k
            or float(committed.last_commit_time_s).hex() != time.hex()):
        _fail("clock", "Native snapshot/commit/clock bits differ")
    receipts, counters = book["receipts"], dict.fromkeys(_COUNTERS, 0)
    fold = {name: np.zeros(1, dtype=np.float64) for name in ("export", "input", "bound", "internal_terminal_deposition")}
    water = dict.fromkeys(("rain_m3", "drainage_m3", "storm_export_m3", "dry_surface_removed_m3", "dry_soil_removed_m3"), 0.0)
    reroutes, last_water = [0, 0], None
    from .dense_schedule import DenseSchedule, check_resolved_event
    schedule = DenseSchedule.from_record(inputs["schedule"])
    for index, receipt in enumerate(receipts):
        check_resolved_event(receipt["resolved"], schedule)
        c = receipt["classes"]
        if receipt["kind"] == "wind":
            additions = (c["boundary_export_kg"], c["boundary_input_kg"], c["inventory_bound_kg"],
                         receipt["finalization"]["deposited_mass_by_class_kg"])
        else:
            additions = (c["export_kg"], c["import_kg"], c["inventory_bound_kg"], c["internal_terminal_deposition_kg"])
            w = receipt["water"]
            for name in water: water[name] += w["export_m3" if name == "storm_export_m3" else name]
            reroutes[0] += int(receipt["routing"]["entry_rerouted"])
            reroutes[1] += receipt["routing"]["n_reroutes_in_event"]
            last_water = receipt["device_water_qualification"]
            for summary in (last_water["entry"], last_water["event"]):
                for name in _COUNTERS:
                    v = _integer(summary[name], (1 << 63) - 1, "actual summary counter")
                    counters[name] = max(counters[name], v) if name.startswith("maximum_") else counters[name] + v
            if counters != last_water["cumulative"]: _fail("water_counters", "receipt cumulative fold")
        for name, value in zip(fold, additions, strict=True):
            a = np.asarray(value, dtype=np.float64).reshape(-1)
            if a.shape != (1,) or not bool(np.all(np.isfinite(a))): _fail("totals", "class vector")
            fold[name] = fold[name] + a
    totals = book["totals"]
    _keys(totals, (*fold, "water", "n_entry_reroutes", "n_event_reroutes"), "totals")
    _keys(totals["water"], tuple(water), "water totals")
    if any(type(v) is not float or not math.isfinite(v) for v in totals["water"].values()):
        _fail("totals", "finite actual water totals")
    _integer(totals["n_entry_reroutes"], (1 << 63) - 1, "entry reroutes")
    _integer(totals["n_event_reroutes"], (1 << 63) - 1, "event reroutes")
    for name, expected in fold.items():
        actual = totals[name]
        if type(actual) is not np.ndarray or actual.shape != (1,) or actual.dtype != np.float64 or actual.tobytes() != expected.tobytes():
            _fail("totals", name)
    if ({k: v.hex() for k, v in totals["water"].items()} != {k: v.hex() for k, v in water.items()}
            or totals["n_entry_reroutes"] != reroutes[0] or totals["n_event_reroutes"] != reroutes[1]
            or counters != book["water_counters"] or _json(last_water) != _json(book["water_record"])):
        _fail("totals", "water/reroute/debit fold")
    limits = inputs["context"]["bridge_limits"]
    for name, ceiling in (("logical_work_debited", limits.max_work),
                          ("logical_read_bytes_debited", limits.max_logical_read_bytes),
                          ("versions_reserved", limits.max_new_versions)):
        if counters[name] > ceiling: _fail("water_cumulative_ceiling", name)
    if (type(book["inventory0"]) is not np.ndarray or book["inventory0"].shape != (1,)
            or book["inventory0"].dtype != np.float64):
        _fail("inventory", "original exact K1 inventory")
    route = current["routing"]
    issue = _integer(book["routing_issue_counter"], 8, "routing issue counter")
    if route is None:
        if k > 1 or issue != 0 or manifest["routing"] is not None:
            _fail("routing", "absence/history")
    else:
        _keys(route, ("source", "velocity", "phase", "generation", "terrain_sha256", "cache_fingerprints"), "held routing")
        _integer(route["generation"], issue, "held generation", 1)
        if route["phase"] is not None or bool(np.any(route["velocity"] != 0.0)):
            _fail("routing", "actual dry transport memory")
    return schedule


def _prepare(path, *, artifact_root, expected_manifest_sha256, expected_source_identity,
             expected_environment_sha256, limits, authority):
    from maple.surface.voxels.windowed_snapshot import load_windowed_state_snapshot
    from .windowed_device_runner import _restore_windowed_device_runner
    authority.check(authority.record)
    manifest, payload, raw = _read_wire(path, expected_seal=expected_manifest_sha256,
                                       schema=SCHEMA, limits=limits, expand=False)
    _preflight(manifest, expected_source_identity, expected_environment_sha256, limits)
    artifacts = _Artifacts(artifact_root, manifest["originals"], manifest["genesis"], manifest["physical"], limits, raw)
    try:
        original_ref = manifest["originals"]
        original, _p, _raw = _read_wire(Path(artifact_root) / original_ref["relative_path"],
            expected_file_sha=original_ref["manifest_file_sha256"], expected_seal=original_ref["identity"],
            schema=ORIGINAL_SCHEMA, limits=limits, expand=False)
        _original_preflight(original, expected_source_identity, expected_environment_sha256, limits)
        physical = _physical_read(artifact_root, manifest["physical"], limits)
        total = original["wire"]["numeric_bytes"] + manifest["wire"]["numeric_bytes"] + physical["numeric_bytes"]
        _boundary_admit(total, total, limits, authority)
        _original, original_payload, _raw = _read_wire(Path(artifact_root) / original_ref["relative_path"],
            expected_file_sha=original_ref["manifest_file_sha256"], expected_seal=original_ref["identity"],
            schema=ORIGINAL_SCHEMA, limits=limits, expand=True)
        _manifest, payload, _raw = _read_wire(path, expected_seal=expected_manifest_sha256, schema=SCHEMA, limits=limits, expand=True)
        # Actual typed HOST holders first; device-tagged module fields use NumPy
        # only in this inert preflight projection, never as a transport fallback.
        originals_host = _decode(original["wire"], original_payload, limits, authority.cp, host=True)
        current_host = _decode(manifest["wire"], payload, limits, authority.cp, host=True)
        _keys(originals_host, _ORIGINAL_FIELDS, "decoded originals")
        inputs_host = originals_host["inputs"]
        ctx = inputs_host["context"]
        saved = load_windowed_state_snapshot(physical["snapshot"]["path"], geometry=ctx["geometry"],
            grain_classes=ctx["grain_classes"], mass_resolution_kg=ctx["mass_resolution_kg"], max_bytes=limits.native_max_bytes)
        _validate_book(inputs_host, current_host, manifest, saved)
        if (originals_host["inventory0"].tobytes() != current_host["bookkeeping"]["inventory0"].tobytes()
                or _json(physical["source_identity"]) != _json(expected_source_identity)):
            _fail("original_identity", "original inventory/Source changed")
        provenance = physical["metadata"]["provenance"]
        _keys(provenance, ("schema", "source_identity_sha256", "case_sha256", "schedule_sha256", "accepted_cursor", "parent_head_token"), "Native provenance")
        if (provenance["schema"] != PHYSICAL_SCHEMA + ".provenance"
                or provenance["source_identity_sha256"] != _sha(expected_source_identity)
                or provenance["case_sha256"] != _sha(inputs_host["case"])
                or provenance["accepted_cursor"] != manifest["state"]["next_index"]
                or provenance["schedule_sha256"] != _sha(inputs_host["schedule"])):
            _fail("native_provenance", "closed Source/case/schedule/cursor join")
        artifacts.check()  # Native reopens paths: explicit quiescent before/after contract.
        inputs = _decode(original["wire"], original_payload, limits, authority.cp, inputs_only=True)
        current = _decode(manifest["wire"], payload, limits, authority.cp)
        ticket = _RestoreTicket(_KEY, inputs=inputs, current=current, host_snapshot=saved,
                                authority=authority, limits=limits, manifest=manifest)
        owner = _restore_windowed_device_runner(ticket)
        artifacts.check()
        return _Prepared(_KEY, owner, dict(accepted=_summary(manifest), source_identity=expected_source_identity,
            native_version=saved["voxel_column"].manifest_digest, known_subset_only=True, claim=CLAIM))
    finally:
        artifacts.close()


# Exact headers derived from pinned ordinary Source; no runtime discovery.
_EXTRA_TYPES = (('maple.core.parameters', 'ModelConfig', ('run_name', 'seed', 'output_dir', 'geometry', 'grain_classes', 'time_and_forcing', 'numerics', 'execution', 'outputs_and_diagnostics', 'topographic_commit', 'avalanching', 'aeolian_process_mode', 'prescribed_detachment', 'particle_transport', 'fluid_threshold', 'aerodynamic_detachment', 'reduced_detachment', 'detachment_source', 'impact_rebound_splash', 'wind_grain_feedback', 'process_scheduler', 'stationary_continuation', 'dormancy', 'physical_window_admission', 'transport_accuracy', 'topographic_wind', 'flow_separation', 'vegetation_wake', 'prescribed_external_shear', 'water_coupling', 'surface_creep')), ('maple.core.parameters.prescribed_external_shear', 'PrescribedExternalShearSpec', ('enabled', 'multiplier', 'multiplier_map')), ('maple.core.parameters.numerics', 'NumericsSpec', ('mass_resolution_kg', 'cohort_calendar_max_bytes', 'cohort_calendar_memory_safety_fraction', 'cohort_calendar_memory_override_confirmed')), ('maple.core.parameters.wind_forcing', 'WindBlock', ('shear_velocity_m_s', 'direction_deg', 'duration_s', 'start_time_s', 'end_time_s')), ('maple.core.parameters.wind_forcing', 'WindForcingRecordSpec', ('blocks', 'source_path', 'source_sha256')), ('maple.core.parameters.surface_creep', 'SurfaceCreepSpec', ('enabled', 'displacement_scale_m', 'slope_coefficient')), ('maple.core.parameters.reduced_detachment', 'CalibratedGrain', ('class_id', 'diameter_m', 'particle_density_kg_m3', 'mass_fraction')), ('maple.core.parameters.reduced_detachment', 'ReducedDetachmentSpec', ('calibration_id', 'calibration_schema', 'coefficient_c', 'impact_threshold_m_s', 'coefficient_c_range', 'impact_threshold_bracket_m_s', 'valid_shear_velocity_range_m_s', 'out_of_envelope_policy', 'activation_policy', 'calibrated_air_density_kg_m3', 'calibrated_gravity_m_s2', 'calibrated_grains', 'grain_property_relative_tolerance', 'composition_relative_tolerance')), ('maple.core.parameters.scalar_fields', 'UniformFieldSpec', ('value',)), ('maple.core.parameters.scalar_fields', 'StepFieldSpec', ('low_value', 'high_value', 'position_fraction', 'orientation_deg_from_x')), ('maple.core.parameters.scalar_fields', 'StripesFieldSpec', ('low_value', 'high_value', 'wavelength_m', 'duty_cycle', 'orientation_deg_from_x')), ('maple.core.parameters.scalar_fields', 'PatchFieldSpec', ('background_value', 'patch_value', 'x_min_m', 'x_max_m', 'y_min_m', 'y_max_m')), ('maple.core.parameters.scalar_fields', 'CheckerboardFieldSpec', ('low_value', 'high_value', 'cell_size_x_m', 'cell_size_y_m')), ('maple.core.parameters.scalar_fields', 'GradientFieldSpec', ('start_value', 'end_value', 'orientation_deg_from_x')), ('maple.core.parameters.scalar_fields', 'RandomPatchesFieldSpec', ('background_value', 'patch_value', 'n_patches', 'patch_radius_m')), ('maple.core.parameters.scalar_fields', 'ScalarFieldSpec', ('builder', 'uniform', 'step', 'stripes', 'patch', 'checkerboard', 'gradient', 'random_patches', 'imported')), ('maple.core.parameters.transport_accuracy', 'TransportAccuracySpec', ('mode', 'policy')), ('maple.core.parameters.prescribed_detachment', 'DetachmentRegion', ('row_start', 'row_end', 'col_start', 'col_end')), ('maple.core.parameters.prescribed_detachment', 'PrescribedDetachmentSource', ('source_type', 'units', 'start_time_s', 'end_time_s', 'region', 'rate_by_class_kg_m2_s', 'rate_array_kg_m2_s')), ('maple.core.parameters.prescribed_detachment', 'PrescribedDetachmentSpec', ('sources',)), ('maple.core.parameters.execution', 'ExecutionSpec', ('backend', 'device_id', 'deterministic_scatter')), ('maple.core.parameters.polygon_geometry', 'PolygonAreaMoments', ('signed_area_m2', 'centroid_x_m', 'centroid_y_m', 'mu_xx', 'mu_yy', 'mu_xy')), ('maple.core.parameters.polygon_geometry', 'PolygonPrincipalAxes', ('angle_from_x_rad', 'lambda_major', 'lambda_minor')), ('maple.core.parameters.outputs_and_diagnostics', 'OutputsAndDiagnosticsSpec', ('visualization_cadence_s', 'conservation_cadence_s', 'exact_checkpoint_cadence_s', 'keyframe_cadence_s', 'enable_exact_reconstruction', 'blosc_cname', 'blosc_clevel', 'blosc_shuffle', 'viz_float32_elevation_change', 'viz_float32_tolerance_m', 'visualization_time_chunk', 'visualization_y_chunk', 'visualization_x_chunk', 'conservation_time_chunk', 'exact_event_chunk', 'exact_checkpoint_chunk')), ('maple.core.parameters.case_initialization', 'DuneSpec', ('height_m', 'wavelength_m', 'orientation_deg_from_x', 'asymmetry')), ('maple.core.parameters.case_initialization', 'PlaneSpec', ('slope_m_m', 'orientation_deg_from_x')), ('maple.core.parameters.case_initialization', 'RidgeSpec', ('height_m', 'width_m', 'orientation_deg_from_x')), ('maple.core.parameters.case_initialization', 'HillSpec', ('height_m', 'width_m')), ('maple.core.parameters.case_initialization', 'PeriodicRoughSpec', ('relief_m', 'correlation_length_m', 'correlation_length_x_m', 'correlation_length_y_m')), ('maple.core.parameters.case_initialization', 'RandomPerturbationSpec', ('relief_m', 'correlation_length_m', 'correlation_length_x_m', 'correlation_length_y_m')), ('maple.core.parameters.case_initialization', 'TopographySpec', ('base', 'nz', 'extra_top_layers', 'dune', 'plane', 'ridge', 'hill', 'periodic_rough', 'sinusoid', 'sinusoid_2d', 'triangular_dune', 'brink_dune', 'cone', 'fractal', 'scarp', 'additional_bases', 'perturbation')), ('maple.core.parameters.case_initialization', 'GrainMixtureSpec', ('fractions',)), ('maple.core.parameters.case_initialization', 'PlantSpec', ('x_m', 'y_m', 'height_m', 'semi_major_m', 'semi_minor_m', 'orientation_deg', 'porosity', 'drag_coefficient')), ('maple.core.parameters.case_initialization', 'PlantLatticeSpec', ('spacing_x_m', 'spacing_y_m', 'height_m', 'semi_major_m', 'semi_minor_m', 'offset_x_m', 'offset_y_m', 'orientation_deg', 'porosity', 'drag_coefficient')), ('maple.core.parameters.case_initialization', 'UniformVegetationRasterSpec', ('height_m', 'cover_fraction', 'porosity', 'drag_coefficient', 'frontal_area_density')), ('maple.core.parameters.case_initialization', 'VegetationSpec', ('source', 'plants', 'lattice', 'raster', 'row', 'staggered_lattice', 'poisson', 'clustered', 'inhibited', 'polygons', 'raster_fields')), ('maple.core.parameters.case_initialization', 'SedimentAvailabilitySpec', ('global_available_fraction', 'available_fraction_by_class', 'available_fraction_map_by_class', 'builder', 'builder_by_class')), ('maple.core.parameters.case_initialization', 'ThresholdModifierSpec', ('gravimetric_moisture_fraction', 'clay_fraction', 'physical_crust_multiplier', 'chemical_cementation_multiplier', 'biological_binding_multiplier')), ('maple.core.parameters.case_initialization', 'StratigraphicLayerSpec', ('fractions', 'thickness_m', 'fraction_of_column', 'lateral')), ('maple.core.parameters.case_initialization', 'StratigraphySpec', ('layers',)), ('maple.core.parameters.case_initialization', 'SinusoidSpec', ('amplitude_m', 'wavelength_m', 'orientation_deg_from_x', 'phase_deg')), ('maple.core.parameters.case_initialization', 'Sinusoid2DSpec', ('amplitude_m', 'wavelength_x_m', 'wavelength_y_m')), ('maple.core.parameters.case_initialization', 'TriangularDuneSpec', ('height_m', 'wavelength_m', 'orientation_deg_from_x', 'stoss_fraction')), ('maple.core.parameters.case_initialization', 'BrinkDuneSpec', ('height_m', 'wavelength_m', 'orientation_deg_from_x', 'brink_position_fraction', 'lee_angle_deg')), ('maple.core.parameters.case_initialization', 'ConeSpec', ('height_m', 'base_radius_m', 'center_x_m', 'center_y_m')), ('maple.core.parameters.case_initialization', 'FractalSpec', ('relief_m', 'hurst_exponent')), ('maple.core.parameters.case_initialization', 'ScarpSpec', ('height_m', 'wavelength_m', 'plateau_fraction', 'lee_angle_deg', 'flat_fraction', 'orientation_deg_from_x')), ('maple.core.parameters.case_initialization', 'SurfaceComponentSpec', ('base', 'dune', 'plane', 'ridge', 'hill', 'periodic_rough', 'sinusoid', 'sinusoid_2d', 'triangular_dune', 'brink_dune', 'cone', 'fractal', 'scarp')), ('maple.core.parameters.case_initialization', 'PlantRowSpec', ('height_m', 'semi_major_m', 'semi_minor_m', 'y_m', 'spacing_x_m', 'offset_x_m', 'orientation_deg', 'porosity', 'drag_coefficient')), ('maple.core.parameters.case_initialization', 'StaggeredLatticeSpec', ('height_m', 'semi_major_m', 'semi_minor_m', 'spacing_x_m', 'spacing_y_m', 'offset_x_m', 'offset_y_m', 'row_offset_fraction', 'orientation_deg', 'porosity', 'drag_coefficient')), ('maple.core.parameters.case_initialization', 'PoissonVegetationSpec', ('height_m', 'semi_major_m', 'semi_minor_m', 'n_plants', 'orientation_deg', 'porosity', 'drag_coefficient')), ('maple.core.parameters.case_initialization', 'ClusteredVegetationSpec', ('height_m', 'semi_major_m', 'semi_minor_m', 'n_clusters', 'plants_per_cluster', 'cluster_radius_m', 'orientation_deg', 'porosity', 'drag_coefficient')), ('maple.core.parameters.case_initialization', 'InhibitedVegetationSpec', ('height_m', 'semi_major_m', 'semi_minor_m', 'n_plants', 'min_spacing_m', 'max_attempts_per_plant', 'orientation_deg', 'porosity', 'drag_coefficient')), ('maple.core.parameters.case_initialization', 'CrownPolygonSpec', ('vertices', 'height_m', 'porosity', 'drag_coefficient')), ('maple.core.parameters.case_initialization', 'VegetationRasterFieldsSpec', ('height_m', 'cover_fraction', 'porosity', 'drag_coefficient', 'frontal_area_density')), ('maple.core.parameters.case_initialization', 'InitialMobileSeedSpec', ('mass_by_class_kg', 'launch_direction_deg')), ('maple.core.parameters.case_initialization', 'WaterInitialDepthSpec', ('depth_m',)), ('maple.core.parameters.time_and_forcing', 'TimeAndForcingSpec', ('outer_timestep_s', 'wind_block_duration_s')), ('maple.core.parameters.case_import', 'TransformStepSpec', ('name', 'parameters')), ('maple.core.parameters.case_import', 'ImportedSourceSpec', ('path', 'format', 'base_dir', 'band', 'array_name', 'row_order', 'nodata_value', 'value_kind', 'value_units', 'horizontal_units', 'regularity_tolerance_m', 'columns', 'delimiter', 'has_header')), ('maple.core.parameters.case_import', 'ImportedFieldSpec', ('source', 'transforms')), ('maple.core.parameters.case_import', 'DepthIntervalSpec', ('top_depth_m', 'bottom_depth_m', 'fractions')), ('maple.core.parameters.case_import', 'SoilProfileSpec', ('intervals',)), ('maple.core.parameters.case_import', 'PointSamplesImportSpec', ('x_m', 'y_m', 'profiles', 'interpolation', 'idw_power')), ('maple.core.parameters.case_import', 'CategoricalMapImportSpec', ('categories', 'profiles')), ('maple.core.parameters.case_import', 'DirectVoxelImportSpec', ('source', 'class_order')), ('maple.core.parameters.case_import', 'ActiveLayerSampleImportSpec', ('fractions',)), ('maple.core.parameters.case_import', 'SedimentImportSpec', ('mode', 'profile', 'point_samples', 'categorical_map', 'direct_voxel_state', 'active_layer_sample', 'bulk_density')), ('maple.core.parameters.case_import', 'VegetationImportSpec', ('objects', 'polygons', 'default_porosity', 'default_drag_coefficient')), ('maple.core.parameters.case_import', 'ForcingImportSpec', ('source', 'direction_convention', 'block_duration_s', 'aggregation', 'samples_per_block', 'min_direction_resultant')), ('maple.core.parameters.case_import', 'EmpiricalDomainSpec', ('label',)), ('maple.core.parameters.case_import', 'CaseImportSpec', ('elevation', 'sediment', 'vegetation', 'forcing', 'domain')), ('maple.core.types.active_layer', 'ActiveLayerState', ('mass_kg',)), ('maple.core.types.water', 'WaterState', ('depth_m', 'mobile_mass_by_cell_class_kg')), ('maple.core.types.sediment_ledger', 'SedimentLedgerState', ('pending_bed_mass_change_kg', 'pending_compensation_kg', 'process_totals_kg', 'process_totals_compensation_kg', 'numerical_residual_positive_kg', 'numerical_residual_positive_compensation_kg', 'numerical_residual_negative_kg', 'numerical_residual_negative_compensation_kg', 'actual_mass_positive_kg', 'actual_mass_positive_compensation_kg', 'actual_mass_negative_kg', 'actual_mass_negative_compensation_kg', 'largest_residual_kg', 'n_physical_touches')), ('maple.core.types.topographic_commit', 'CommittedTopographyState', ('reference_elevation_m', 'elevation_offset_m', 'committed_active_layer_mass_by_class_kg', 'committed_voxel_mass_by_class_kg', 'last_commit_time_s', 'commit_count')), ('maple.core.types.sediment_availability', 'SedimentAvailabilityState', ('available_mass_kg', 'bound_mass_kg')), ('maple.core.types.ledger_reconciliation', 'LedgerReconciliationBaseline', ('active_layer_mass_by_class_kg', 'voxel_mass_by_class_kg', 'last_reconciliation_time_s', 'reconciliation_count')), ('maple.core.types.threshold_modifiers', 'ThresholdModifierState', ('gravimetric_moisture_fraction', 'clay_fraction', 'physical_crust_multiplier', 'chemical_cementation_multiplier', 'biological_binding_multiplier')), ('maple.core.types.vegetation', 'VegetationObjectSet', ('center_x_m', 'center_y_m', 'semi_major_m', 'semi_minor_m', 'orientation_deg', 'height_m', 'porosity', 'drag_coefficient')), ('maple.core.types.vegetation', 'VegetationRasterState', ('height_m', 'cover_fraction', 'porosity', 'drag_coefficient', 'frontal_area_density')), ('maple.core.types.vegetation', 'VegetationState', ('source', 'raster', 'objects')), ('maple.aeolian.scheduler.state', 'SchedulerPhysicalState', ('voxel_column', 'active_layer', 'mobile_state', 'ledger', 'committed_topography', 'threshold_modifier_state', 'time_s', 'vegetation', 'sediment_availability', 'water', 'coupled_trajectory_semantics', 'ledger_reconciliation_baseline')), ('maple.aeolian.scheduler.state', 'StationaryContinuationState', ('calibration_duration_s', 'launched_mass_rate_by_class_kg_s', 'impact_mass_rate_by_class_kg_s', 'deposited_mass_rate_by_class_kg_s', 'rebounded_mass_rate_by_class_kg_s', 'boundary_input_rate_by_class_kg_s', 'boundary_export_rate_by_class_kg_s', 'bed_change_rate_by_class_kg_s', 'mobile_mass_net_rate_by_class_kg_s', 'numerical_residual_scalar_rate_kg_s', 'face_flux_rate_by_class_kg_s', 'impact_state_rate_kg_s', 'actual_detachment_rate_by_cell_class_kg_s', 'splash_removal_rate_by_cell_class_kg_s', 'deposited_rate_by_cell_class_kg_s', 'x_face_net_crossing_rate_by_class_kg_s', 'y_face_net_crossing_rate_by_class_kg_s', 'domain_momentum_exchange_rate_kg_m_s', 'grain_borne_stress_reference', 'calibrated_at_time_s', 'last_probe_time_s', 'source_config_identity_sha256', 'environmental_fingerprint')), ('maple.aeolian.scheduler.state', 'PhysicalAdmissionCertificate', ('source_config_identity_sha256', 'source_explicit_slice_s', 'admitted_at_time_s', 'consecutive_agreeing_windows', 'required_consecutive_agreeing_windows', 'worst_quantity', 'worst_violation', 'previous_window_duration_s', 'current_window_duration_s')), ('maple.aeolian.scheduler.state', 'SchedulerCoupledState', ('grain_borne_stress', 'status', 'last_equilibrium_result', 'environmental_fingerprint', 'stationary_rate', 'window_evidence', 'window_admission_streak', 'physical_admission', 'window_evidence_context_sha256', 'window_admission_transition_elapsed_s', 'window_admission_context_time_s', 'calibration_stream', 'probe_stream', 'probe_stall_count')), ('maple.aeolian.scheduler.state', 'TopographicWindCache', ('cache_key', 'modifier', 'integrity_hash', 'last_observed_elevation_m', 'n_surface_changes_since_recompute', 'cumulative_elevation_change_m', 'last_recompute_time_s', 'recompute_reason', 'stale_under_policy')), ('maple.aeolian.scheduler.state', 'VegetationWindCache', ('cache_key', 'modifier', 'integrity_hash', 'last_recompute_time_s', 'recompute_reason')), ('maple.aeolian.scheduler.state', 'ActiveLayerCompositionCache', ('threshold_result', 'exposed_area_weight', 'detachable_mask', 'cache_key', 'integrity_hash', 'last_recompute_time_s', 'recompute_reason', 'source_contract_identity')), ('maple.aeolian.scheduler.state', 'SchedulerState', ('physical', 'coupled', 'block_index', 'block_offset_s', 'last_resolved_wind', 'topographic_wind_cache', 'vegetation_wind_cache', 'active_layer_composition_cache')), ('maple.aeolian.scheduler.windowed_record', 'WindowedWindRecordLimits', ('block', 'max_blocks', 'max_total_outer_steps', 'max_accounting_reasons')), ('maple.aeolian.scheduler.windowed_device', 'DeviceWindowedWindBlockLimits', ('step', 'solve', 'validation', 'commit', 'max_outer_steps', 'max_block_host_bytes', 'max_block_device_bytes', 'source_control_host_bytes', 'max_image_operators', 'max_reference_items', 'max_block_h2d_bytes', 'max_block_d2h_bytes', 'reserve_host_bytes', 'device_reserve_bytes', 'device_safety_fraction')), ('maple.surface.avalanche.windowed_device', 'DeviceWindowedAvalancheLimits', ('bed_limits', 'max_iterations', 'max_snapshot_device_bytes', 'max_total_logical_deep_read_bytes', 'max_total_miss_batches', 'max_total_h2d_bytes', 'max_total_d2h_bytes', 'boundary_host_bytes', 'reserve_host_bytes', 'device_reserve_bytes', 'device_safety_fraction')), ('maple.surface.avalanche.windowed_device', 'DeviceWindowedAvalancheValidationLimits', ('bed_limits', 'delta_budgets', 'max_snapshot_device_bytes', 'full_partition_host_bytes', 'max_h2d_bytes', 'max_d2h_bytes', 'reserve_host_bytes', 'device_reserve_bytes', 'device_safety_fraction')), ('maple.surface.topographic_commit.windowed_device', 'DeviceWindowedCommitLimits', ('max_guard_device_bytes', 'max_guard_host_bytes', 'publication_max_records', 'publication_encoding', 'publication_overlay_max_bytes', 'reserve_host_bytes', 'device_reserve_bytes', 'device_safety_fraction')), ('maple.surface.active_layer.bed_exchange_device', 'DeviceBedExchangeLimits', ('max_operations', 'max_gather_columns', 'max_journal_records', 'max_cells', 'max_classes', 'max_voxels', 'max_impact_rows', 'max_new_device_bytes', 'max_host_staging_bytes', 'max_logical_deep_read_bytes', 'max_mapped_bytes', 'device_reserve_bytes', 'host_reserve_bytes', 'device_safety_fraction')), ('maple.aeolian.flux.windowed_step_device', 'DeviceWindowedReducedStepLimits', ('bed_limits', 'creep_limits', 'delta_budgets', 'max_mobile_slots', 'max_state_leaves', 'max_config_leaves', 'max_leaf_bytes', 'max_snapshot_device_bytes', 'max_snapshot_host_bytes', 'transport_body_device_bytes', 'transport_body_host_bytes', 'boundary_host_bytes', 'max_step_h2d_bytes', 'max_step_d2h_bytes', 'reserve_host_bytes', 'device_reserve_bytes', 'device_safety_fraction')), ('masy.windowed_device_runner', 'DeviceWindowedWindProfile', ('model_config', 'wind_config', 'record_limits', 'source_identity', 'calibration', 'label')), ('masy.windowed_device_bed', 'DeviceWindowedBedLimits', ('max_bed_device_bytes', 'max_known_host_bytes', 'max_known_device_bytes', 'max_stage_logical_read_bytes')), ('masy.dense_device_resources', 'DeviceResourceLimits', ('max_private_bytes', 'max_static_bytes', 'max_work_bytes', 'comparison_reduction_scratch_bytes')), ('maple_syrup.windowed_shared_bed_device', 'PositiveWindowedDeviceWaterLimits', ('max_host_bytes', 'max_device_bytes', 'max_mapped_bytes', 'max_logical_read_bytes', 'max_work', 'max_store_bytes', 'max_run_bytes', 'max_new_versions', 'max_entries', 'max_external_backing_bytes', 'max_event_steps', 'max_retries', 'max_transport_substeps', 'max_report_rows', 'reserve_host_bytes', 'reserve_device_bytes', 'safety_fraction', 'run_dir')), ('maple_syrup.routing', 'RoutingGraph', ('shape', 'dx_m', 'aspect', 'slope', 'slope_before_edge_rule', 'edge_rule_applied', 'friction_factor', 'active', 'outlet', 'receiver', 'level', 'level_order_host', 'level_bounds', 'conveyance', 'active_flat', 'outlet_flat', 'level_order', 'conveyance_lo', 'donor_position', 'donor_mask', 'input_sha256', 'xp', 'pit_storage', 'policy', 'flat_storage')), ('maple_syrup.sediment_transport', 'TransportNetwork', ('shape', 'dx_m', 'n_cells', 'n_active', 'receiver_index', 'outlet_flat', 'active_flat', 'cells_x', 'rows_x', 'faces_x', 'sign_x', 'cells_y', 'faces_y', 'cols_y', 'sign_y', 'receiver_host', 'outlet_host', 'active_host', 'graph_input_sha256', 'xp', 'terminal_flat', 'terminal_host', 'n_terminal')), ('maple_syrup.sediment_physics', 'PhysicsGrid', ('shape', 'cell_area_m2', 'slope', 'active', 'xp', 'terminal')))
_RUNNER_FIELDS = ('_bed', '_schedule', '_wind', '_water', '_control', '_limits', '_n_classes', '_inventory_elements', '_inventory0', '_pool', '_soil', '_routing', '_ledger', '_next', '_n_accepted', '_totals', '_receipts', '_stop_receipt', '_active_evidence', '_water_record', '_water_counters', '_pending_water_counters', '_production', '_production_limits', '_worker_identity', '_wind_profile', '_host_config', '_native_audit', '_device_resources', '_resource_limits', '_constructor_frontier', '_device_ops', '_device_own', '_wind_enter_bound', '_wind_accept_bound', '_static_parts', '_static_parts_sha256', '_static_record', '_static_sha256', '_accepted_bed', '_accepted_private')


def _compare(work, readback, limits):
    runner_admission(work, limits); runner_admission(readback, limits)
    work._bed._compare((work._bed._accepted.state, readback._bed._accepted.state))
    left, right = work._device_resources._accepted, readback._device_resources._accepted
    work._device_resources._compare((left.pool, right.pool), (left.soil, right.soil),
        *((work._device_resources.static(name), readback._device_resources.static(name))
          for name in work._device_resources._statics))
    cp = work._device_ops.xp
    if (work._static_sha256 != readback._static_sha256
            or _fingerprint(_book(work), limits, cp) != _fingerprint(_book(readback), limits, cp)
            or _fingerprint(_routing(work, limits), limits, cp) != _fingerprint(_routing(readback, limits), limits, cp)):
        _fail("readback_identity", "all physical/private/static/bookkeeping/routing data must match")


def _reservation(limits, max_work_bytes):
    """Named existing caps reserve the whole operation before its writes."""
    image = limits.image
    # Parent/work plus independent readback, their HOST payloads and one
    # complete first-error residual are simultaneous positive occurrences.
    raw = 2 * image.native_max_bytes + 2 * image.max_serialized_bytes
    memory = 2 * max_work_bytes + 3 * raw + 3 * image.max_metadata_bytes + (64 << 20)
    if memory > limits.max_memory_bytes or limits.max_memory_bytes > 5 << 30:
        _fail("memory_cap", "full boundary/readback/current/error plus original work")
    return dict(disk_bytes=raw + 3 * image.max_metadata_bytes + (64 << 20)
                + 2 * limits.max_head_bytes + limits.max_generation_bytes + limits.max_intent_bytes,
                entries=2 * image.max_files + 16, memory_bytes=memory,
                known_subset_only=True, constructor_allocation=None, external_runtime=None, claim=CLAIM)


class _GPUStore(store.DenseBoundaryStore):
    """Fresh authority is outside immutable disk data and old Dense slots."""
    __slots__ = ("_authority", "_disk_context", "_deep_store_path")

    def _admit(self, accepted_names, max_work_bytes, *, initial=False):
        super()._admit(accepted_names, max_work_bytes, initial=initial)
        _disk_admit(self._disk_context, self._deep_store_path,
                    disk_bytes=self._reservation_record["disk_bytes"],
                    entries=self._reservation_record["entries"])

    def _stable(self):
        self._binding()
        inventory = store._inventory(self._root, self._limits, expected_root=os.fstat(self._rootfd))
        raw = store._read(self._root / "STORE.json", 128 << 10,
                          links=store._published_links(self._root, "STORE.json"))
        if raw != self._store_raw: _fail("identity_drift", "immutable STORE changed")
        self._temporary_heads(inventory); self._storage()
        source, environment = self._authority.check(self._header["runtime_reference"])
        if _json(source) != _json(self._header["source_identity"]) or environment != self._header["environment_sha256"]:
            _fail("runtime_identity", "full original profile/Source/environment")

    def _load(self, head):
        chain = store._generation_chain(self._root, self._header, store._sha(self._store_raw), head, self._limits)
        _reservation(self._limits, 0)
        image = chain[0][1]["image"]
        prepared = _prepare(self._root / image["relative_path"], artifact_root=self._root,
            expected_manifest_sha256=image["manifest_sha256"], expected_source_identity=self._header["source_identity"],
            expected_environment_sha256=self._header["environment_sha256"], limits=self._limits.image,
            authority=self._authority)
        owner = _owner(prepared)
        self._disk_context = owner._bed.context
        self._deep_store_path = owner._bed._accepted.state.voxel_column.deep_store.store_dir
        _disk_admit(self._disk_context, self._deep_store_path, disk_bytes=0, entries=0)
        if _json(prepared.diagnostics()["accepted"]) != _json(chain[0][1]["accepted_summary"]):
            _fail("readback_identity", "selected accepted summary")
        for selected, generation, _manifest in chain:
            parent = generation["parent_head_record"]
            index = 0 if selected["sequence"] == 0 else selected["sequence"] - 1
            if generation["operation_id"] != store._operation(self._header["store_id"], parent, index, owner._schedule.digest):
                _fail("operation_identity", "original schedule/parent/cursor")
        return prepared, chain

    def _extra_original(self, manifest, wrapper_raw):
        ref = self._header["originals"]
        rec, _payload, _raw = _read_wire(self._root / ref["relative_path"], expected_file_sha=ref["manifest_file_sha256"],
            expected_seal=ref["identity"], schema=ORIGINAL_SCHEMA, limits=self._limits.image, expand=False)
        if manifest["originals"] != ref: _fail("original_identity", "immutable original reference")
        return rec

    def _save_physical(self, path, owner, pack):
        if pack["schema"] != ORIGINAL_SCHEMA: _fail("original_identity", "actual original capsule")
        return _physical_write(path, owner, self)

    def _save_image(self, directory, owner, physical):
        return _image_write(directory, owner, physical, self)

    def report(self):
        value = super().report()
        value["scope"] = "Small same-backend GPU accepted-boundary profile; no active-event resume"
        value["qualification"] = dict(known_subset_only=True, claim=CLAIM, allocation_or_fit_proof=None)
        return value


_GPU_DEFAULT = dataclasses.replace(store._DEFAULT, max_store_bytes=512 << 20,
    max_memory_bytes=5 << 30, max_accepted=5,
    image=boundary.BoundaryImageLimits(max_cells=16, max_classes=1, max_voxels=6, max_cohorts=4000, max_events=4))


def create_windowed_device_boundary_store(path, *, runner, qualification_budget,
        limits=_GPU_DEFAULT, storage_contract):
    """Create only a genuine untouched genesis; Root already owns this role."""
    store.BoundaryStoreLimits.__post_init__(limits); store.StorageContract.__post_init__(storage_contract)
    runner_admission(runner, limits.image)
    if (runner._next != 0 or runner._n_accepted != 0 or runner._receipts
            or runner._bed.context.runtime_bounds is not qualification_budget):
        _fail("fresh_runner", "untouched genuine runner and its exact fresh role budget required")
    if limits.max_store_bytes > 512 << 20 or limits.max_memory_bytes > 5 << 30:
        _fail("limits", "original Root storage/HOST caps")
    path = Path(_path(path))
    if not path.is_relative_to(Path(runner._bed.context.bridge_limits.run_dir)):
        _fail("run_scope", "combined store must remain in the Native/Root governed run directory")
    authority = _Authority(qualification_budget)
    initial_reservation = _reservation(limits, runner._limits.max_work_bytes)
    _disk_admit(runner._bed.context, runner._bed._accepted.state.voxel_column.deep_store.store_dir,
                disk_bytes=initial_reservation["disk_bytes"] + (72 << 20),
                entries=initial_reservation["entries"])
    handle = _GPUStore._open(path, limits, create=True)
    handle._authority = authority
    handle._disk_context = runner._bed.context
    handle._deep_store_path = runner._bed._accepted.state.voxel_column.deep_store.store_dir
    try:
        probe, probe_sha = store._probe(handle._root, handle._rootfd)
        os.mkdir(handle._root / "originals", 0o700); os.mkdir(handle._root / "generations", 0o700)
        handle._header = dict(schema=STORE_SCHEMA, store_id=uuid.uuid4().hex, profile=PROFILE,
            source_identity=copy.deepcopy(authority.record["source_identity"]), runtime_reference=copy.deepcopy(authority.record),
            environment_sha256=authority.record["environment_sha256"],
            storage_contract=dict(schema=STORE_SCHEMA + ".storage", operator_declares_local=True,
                operator_declares_fsync_durable=True, observed_fs_type=probe["observation"]["fs_type"],
                observed_fs_id=probe["f_fsid"], root_path=str(handle._root), root_dev=probe["root_dev"],
                root_ino=probe["root_ino"], probe_file_sha256=probe_sha),
            limits=store._limits_record(limits), originals=None, genesis_physical=None)
        handle._admit([], runner._limits.max_work_bytes, initial=True)
        original_path = handle._root / "originals" / ("o-" + uuid.uuid4().hex)
        original = _write_wire(original_path,
            dict(inputs=_inputs(runner), initial_pool=runner._device_resources._accepted.pool,
                 initial_soil=runner._device_resources._accepted.soil, inventory0=runner._inventory0),
            schema=ORIGINAL_SCHEMA, limits=limits.image, authority=authority,
            extra=dict(profile=PROFILE, source_identity=handle._header["source_identity"],
                       environment_sha256=handle._header["environment_sha256"]))
        handle._header["originals"] = dict(format=ORIGINAL_SCHEMA, relative_path=str(original_path.relative_to(handle._root)),
            manifest_file_sha256=_file_row(original_path / "manifest.json", limits.image.max_metadata_bytes)["sha256"],
            identity=original["manifest_sha256"])
        directory, intent, halt = handle._reserve(0, None, runner._schedule.digest)
        physical = _physical_write(directory / "physical", runner, handle)
        handle._header["genesis_physical"] = physical
        handle._header = store._seal(handle._header, "store_sha256")
        store._sealed_write(handle._root, "STORE.json", handle._header, 128 << 10)
        handle._store_raw = oi._json(handle._header)
        image = _image_write(directory / "image", runner, physical, handle)
        head = handle._seal_generation(directory, intent, None, image)
        prepared, chain = handle._load(head)
        _compare(runner, _owner(prepared), limits.image)
        state = handle._idle_state(head, prepared, chain[0][1])
        del prepared
        handle._publish(head, state, None, halt)
        return handle
    except BaseException:
        handle.close(); raise


def recover_windowed_device_boundary_store(path, *, expected_store_file_sha256, qualification_budget, limits=_GPU_DEFAULT):
    """Selected HEAD only; fresh granted role, original immutable evidence."""
    store.BoundaryStoreLimits.__post_init__(limits); _digest(expected_store_file_sha256, "external STORE hash")
    authority = _Authority(qualification_budget)
    handle = _GPUStore._open(path, limits, create=False); handle._authority = authority
    try:
        store._inventory(handle._root, limits, expected_root=os.fstat(handle._rootfd))
        header, raw = store._record(handle._root / "STORE.json", 128 << 10,
                                    links=store._published_links(handle._root, "STORE.json"))
        store._check_seal(header, store._STORE, "store_sha256", STORE_SCHEMA)
        if (type(header["store_id"]) is not str or store._NONCE.fullmatch(header["store_id"]) is None
                or store._sha(raw) != expected_store_file_sha256 or header["profile"] != PROFILE
                or store._limits_record(store._limits_load(header["limits"])) != store._limits_record(limits)):
            _fail("store_identity", "exact store/profile/limits required")
        authority.check(header["runtime_reference"])
        handle._header, handle._store_raw = header, raw
        handle._stable(); head, head_raw = handle._head()
        prepared, chain = handle._load(head)
        state = handle._idle_state(head, prepared, chain[0][1]); del prepared
        store._sync_tree(handle._root); handle._stable()
        if handle._head()[1] != head_raw: _fail("parent_drift", "HEAD changed during recovery")
        os.fsync(handle._rootfd); handle._state = state
        return handle
    except BaseException:
        handle.close(); raise


def _history(root, header, chain, limits):
    """Metadata/bytes only: Native snapshot and wrapper share ONE clock/cursor."""
    genesis = chain[-1][2]
    epoch = genesis["state"]["native_time"]
    for head, generation, manifest in chain:
        physical = _physical_read(root, manifest["physical"], limits.image)
        meta, parent = physical["metadata"], generation["parent_head_record"]
        provenance = meta["provenance"]
        _keys(provenance, ("schema", "source_identity_sha256", "case_sha256", "schedule_sha256",
                           "accepted_cursor", "parent_head_token"), "Native provenance")
        expected = dict(schema=PHYSICAL_SCHEMA + ".provenance", source_identity_sha256=_sha(header["source_identity"]),
            case_sha256=manifest["case_sha256"], schedule_sha256=manifest["schedule_sha256"],
            accepted_cursor=head["sequence"], parent_head_token=None if parent is None else store._token(parent))
        if (provenance != expected or type(meta["step"]) is not int or meta["step"] != head["sequence"]
                or float(meta["time_s"]).hex() != decode_time(manifest["state"]["native_time"]).hex()
                or _json(physical["source_identity"]) != _json(header["source_identity"])
                or manifest["ledger"]["native_epoch"] != epoch
                or manifest["case_sha256"] != genesis["case_sha256"]
                or manifest["schedule_sha256"] != genesis["schedule_sha256"]):
            _fail("native_history", "exact accepted snapshot/provenance/clock/cursor/original joins")
    for child, parent in zip(chain, chain[1:]):
        if child[2]["ledger"]["entries"][:-1] != parent[2]["ledger"]["entries"]:
            _fail("clock_history", "old clock entries cannot be rewritten")
