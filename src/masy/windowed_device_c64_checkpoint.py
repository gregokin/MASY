"""Closed factory C64 same-backend accepted-boundary persistence.

Native owns physical snapshots/storage/physics. This sibling owns only typed
companions, checked restoration and the existing atomic MASY boundary protocol.
No imported CaseBudget, active-event resume or general allocation proof.
"""
from __future__ import annotations

import contextlib
import ast
import copy
import dataclasses
import hashlib
import importlib
import json
import math
import os
import re
import resource
import stat
import sys
import threading
import uuid
import zipfile
from pathlib import Path, PosixPath
from types import ModuleType, FunctionType, CodeType, SimpleNamespace

from . import dense_original_inputs as oi
from . import dense_runner_checkpoint as boundary
from . import dense_boundary_store as store
from . import windowed_device_checkpoint as codec
from .clock import ClockLedger, decode_time, encode_time

SCHEMA = "masy.windowed_device_c64_boundary_image.v1"
PROFILE = "C64_K1_V20_W3_gpu_accepted_boundary_v1"
STORE_SCHEMA = "masy.windowed_device_c64_boundary_store.v1"
ORIGINAL_SCHEMA = "masy.windowed_device_c64_originals.v1"
PHYSICAL_SCHEMA = "masy.windowed_device_c64_native_snapshot.v1"
RUNTIME_SCHEMA = STORE_SCHEMA + ".runtime"
CLAIM = codec.CLAIM
E1_MODULE = "maple.core.backend.prefix_cell_reductions"
PREFIX_MODULE = "maple.surface.voxels.windowed_prefix"
_KEY = object()
_SHARED_STORE = store
_ACTIVE_SCOPE = None
_DTYPES, _BED, _COUNTERS = codec._DTYPES, codec._BED, codec._COUNTERS
_INPUT_FIELDS, _CURRENT_FIELDS, _ORIGINAL_FIELDS = codec._INPUT_FIELDS, codec._CURRENT_FIELDS, codec._ORIGINAL_FIELDS
_RUNNER_FIELDS = codec._RUNNER_FIELDS
_fail = codec._fail
_keys = codec._keys
_integer = codec._integer
_digest = codec._digest
_json = codec._json
_sha = codec._sha
_plain = codec._plain
_wire_json_limits = codec._wire_json_limits
_record_json_limits = codec._record_json_limits
_plain_wire = codec._plain_wire
_plain_record = codec._plain_record
_finite_hex = codec._finite_hex
_Encoder = codec._Encoder
_wire_preflight = codec._wire_preflight
_wire_root = codec._wire_root
_decode = codec._decode
_environment = codec._environment
_file_row = codec._file_row
_check_file = codec._check_file
_reference = codec._reference
_fingerprint = codec._fingerprint
_reservation = codec._reservation


def _path(value):
    if type(value) not in (str, PosixPath):
        _fail("path", "exact str or PosixPath required before conversion")
    return codec._path(value)


def _shared():
    module = sys.modules.get("maple_syrup.windowed_shared_bed_device")
    if type(module) is not ModuleType:
        _fail("source", "genuine already-loaded owning SYRUP module required")
    return module


class _RestoreTicket:
    __slots__ = ("key", "pid", "thread", "phase", "authority", "limits", "inputs",
                 "current", "host_snapshot", "manifest", "before", "diagnostics", "pending_refs")

    def __init__(self, key):
        if key is not _KEY:
            _fail("ticket", "private C64 coordinator only")
        self.key, self.pid, self.thread = key, os.getpid(), threading.get_ident()
        self.phase, self.authority = "constructing", None
        self.limits = self.inputs = self.current = self.host_snapshot = self.manifest = None
        self.before = self.diagnostics = self.pending_refs = None

    def __reduce_ex__(self, protocol):
        raise TypeError("C64 restore tickets cannot be serialized")


class _Scope:
    __slots__ = ("pid", "thread", "authority", "ticket")

    def __init__(self, authority, ticket):
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.authority, self.ticket = authority, ticket


def _checked_ticket(ticket):
    if (type(ticket) is not _RestoreTicket or ticket.key is not _KEY
            or ticket.phase != "constructing" or ticket.pid != os.getpid()
            or ticket.thread != threading.get_ident()
            or type(_ACTIVE_SCOPE) is not _Scope or _ACTIVE_SCOPE.ticket is not ticket
            or _ACTIVE_SCOPE.authority is not ticket.authority):
        _fail("ticket", "same actual constructing PID/thread/authority ticket required")
    return ticket


@contextlib.contextmanager
def _attempt(ticket, authority=None):
    global _ACTIVE_SCOPE
    if (type(ticket) is not _RestoreTicket or ticket.key is not _KEY
            or ticket.phase != "constructing" or ticket.pid != os.getpid()
            or ticket.thread != threading.get_ident()):
        _fail("ticket", "one same-PID/thread attempt only")
    previous = _ACTIVE_SCOPE
    if previous is not None and previous.ticket is not None:
        _fail("ticket", "restore reentry is unsupported")
    ticket.authority = authority
    _ACTIVE_SCOPE = _Scope(authority, ticket)
    try:
        yield ticket
        if ticket.phase != "consumed":
            _fail("ticket", "complete coordinator must consume its ticket")
    except BaseException as primary:
        try:
            ticket.phase = "poisoned"
        except BaseException:
            try:
                primary.add_note("C64 restore ticket poison was unconfirmed")
            except BaseException:
                pass
        raise
    finally:
        _ACTIVE_SCOPE = previous


@contextlib.contextmanager
def _writer_scope(authority):
    global _ACTIVE_SCOPE
    if _ACTIVE_SCOPE is not None:
        _fail("scope", "nested writer operation is unsupported")
    _ACTIVE_SCOPE = _Scope(authority, None)
    try:
        yield
    finally:
        _ACTIVE_SCOPE = None


def _scope_authority():
    scope = _ACTIVE_SCOPE
    if (type(scope) is not _Scope or scope.pid != os.getpid()
            or scope.thread != threading.get_ident()
            or type(scope.authority) is not _Authority):
        _fail("scope", "actual current C64 IO/restore scope required")
    if scope.ticket is not None:
        _checked_ticket(scope.ticket)
    return scope.authority


def _read_file(path, maximum):
    _scope_authority().prelude()
    return codec._read_file(path, maximum)


def _code_charge():
    """Original positive Source-word coefficients; allocator terms stay opaque."""
    bed = sys.modules.get("masy.windowed_device_c64_bed")
    resources = sys.modules.get("masy.dense_device_resources")
    if type(bed) is not ModuleType or type(resources) is not ModuleType:
        _fail("source", "already-loaded genuine MASY owners required before read admission")
    codes = slots = text = bindings = 0
    def visit(code, depth=0):
        nonlocal codes, slots, text
        if depth > 24 or codes >= 8192:
            _fail("code_frontier", "closed Source code forest exceeded")
        codes += 1
        slots += 1 + code.co_nlocals + code.co_stacksize + len(code.co_cellvars) + len(code.co_freevars)
        text += len(code.co_code) + len(code.co_linetable) + len(code.co_exceptiontable)
        text += sum(len(v) for v in code.co_consts if type(v) is str)
        for value in code.co_consts:
            if type(value) is CodeType:
                visit(value, depth + 1)
    # Bed/runner's original source producer already charges those two images.
    for module in (sys.modules[__name__], codec, store, resources):
        for namespace, name, descriptor, code in bed._capture_source_bindings(vars(module)):
            bindings += 32 + 4*8 + 4*len(name)
            visit(code)
    frames = 25*(32*codes + 8*slots)
    return 3*(32*codes + 8*slots + frames + 4*text + bindings) + bed._wrapper_source_frontier()["host_source_bytes"]


def _read_plan(image, accepted, reference_bytes):
    """Original named read families, three current/next/error occurrences."""
    C64BoundaryImageLimits.__post_init__(image)
    _integer(accepted, 5, "accepted boundary capacity", 1)
    _integer(reference_bytes, 64 << 20, "complete Source reference occurrences", 1)
    count, meta, native, header = accepted, image.max_metadata_bytes, image.native_max_bytes, image.max_header_bytes
    families = dict(source_references=reference_bytes,
        image_manifests=(count+1)*meta,
        physical_manifests=(count+1)*min(1 << 20, meta),
        native_archives=(count+1)*native,
        deep_metadata=2*(count+1)*meta,
        control=(128 << 10)+(count+1)*(32 << 10)+count*((128 << 10)+(32 << 10)),
        native_headers=12*(count+1)*(header+12),
        masy_hash_buffer=1 << 20, native_hash_buffer=8 << 20,
        own_source_reads=6*(2 << 20))
    return dict(families=families, host_bytes=3*sum(families.values()),
        banks=("current", "next", "first_error"), known_subset_only=True,
        constructor_allocation=None, external_runtime=None, claim=CLAIM)


class _Authority:
    """Fresh Root-owned role facts; this object issues no launch authority."""
    __slots__ = ("budget", "bridge_limits", "limits", "native_source_refs", "cp", "pid",
                 "thread", "device", "stream", "record", "source_rows", "native_images", "class_images", "native_code_bytes")

    def __init__(self, budget, bridge_limits, native_source_refs, limits):
        shared = _shared()
        if type(budget) is not shared.DeviceWaterC64QualificationBudget:
            _fail("authority", "exact factory C64 budget required; CaseBudget unsupported")
        if type(bridge_limits) is not shared.PositiveWindowedDeviceWaterLimits:
            _fail("authority", "genuine explicit fresh bridge limits required")
        C64BoundaryStoreLimits.__post_init__(limits)
        bridge_limits.__post_init__(); budget.validated()
        if (type(native_source_refs) is not tuple or len(native_source_refs) != 2
                or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not str for row in native_source_refs)
                or tuple(row[0] for row in native_source_refs) != (E1_MODULE, PREFIX_MODULE)):
            _fail("native_source", "exact ordered E1/prefix Root refs required")
        for name, ref in native_source_refs:
            if type(ref) is not shared.SourceReference:
                _fail("native_source", "genuine SourceReference required")
            ref.validated()
        cp = sys.modules.get("cupy")
        if type(cp) is not ModuleType:
            _fail("authority", "genuine already-loaded CuPy required")
        self.budget, self.bridge_limits, self.limits = budget, bridge_limits, limits
        self.native_source_refs, self.cp = native_source_refs, cp
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.device = int(cp.cuda.runtime.getDevice())
        self.stream = int(cp.cuda.get_current_stream().ptr)
        self.source_rows, self.native_images, self.class_images = (), (), ()
        self.native_code_bytes = 0
        if type(_ACTIVE_SCOPE) is not _Scope:
            _fail("scope", "ticket/writer scope precedes authority construction")
        _ACTIVE_SCOPE.authority = self
        if _ACTIVE_SCOPE.ticket is not None:
            _ACTIVE_SCOPE.ticket.authority = self
            _ACTIVE_SCOPE.ticket.limits = limits.image
        self.prelude()  # No reference/file/import operation precedes admission.
        self._authenticate()
        evidence = shared.c64_qualification_budget_record(budget)
        self.record = dict(schema=RUNTIME_SCHEMA,
            source_identity=self._identity(evidence), environment=_environment(cp),
            evidence=evidence, grant=dataclasses.asdict(budget.grant),
            native_source_refs=[dict(module=n, **dataclasses.asdict(r)) for n, r in native_source_refs],
            masy_source=[dict(row) for row in self.source_rows])
        self.record["environment_sha256"] = _sha(self.record["environment"])

    def __reduce_ex__(self, protocol):
        raise TypeError("C64 execution authority is process-local")

    def prelude(self):
        if (self.pid != os.getpid() or self.thread != threading.get_ident()
                or self.device != int(self.cp.cuda.runtime.getDevice())
                or self.stream != int(self.cp.cuda.get_current_stream().ptr)):
            _fail("authority_scope", "fresh role process/thread/device/stream changed")
        _check_module_image()
        for kind, expected in ((resource.RLIMIT_CPU, (480, 540)),
                               (resource.RLIMIT_FSIZE, (64 << 20, 64 << 20)),
                               (resource.RLIMIT_CORE, (0, 0))):
            if resource.getrlimit(kind) != expected:
                _fail("process_limits", "original qualification OS limits required")
        shared = _shared()
        refs = (*shared._c64_refs(self.budget), *(ref for _, ref in self.native_source_refs))
        raw = sum(ref.size for ref in refs)
        plan = _read_plan(self.limits.image, self.limits.max_accepted, raw)
        host = plan["host_bytes"] + 3*raw + shared._c64_code_charge() + _code_charge() + self.native_code_bytes + self.bridge_limits.reserve_host_bytes
        device = self.bridge_limits.reserve_device_bytes
        if (host > min(self.limits.max_memory_bytes, self.bridge_limits.max_host_bytes, 5 << 30)
                or device > min(self.bridge_limits.max_device_bytes, 2 << 30)):
            _fail("read_frontier", "complete positive read/Source/reserve banks exceed original caps")
        capacity = sys.modules.get("maple.aeolian.cohorts.capacity")
        memory = sys.modules.get("maple.aeolian.cohorts.memory_budget")
        bed_exchange_device = sys.modules.get("maple.surface.active_layer.bed_exchange_device")
        if any(type(module) is not ModuleType for module in (capacity, memory, bed_exchange_device)):
            _fail("source", "genuine loaded Native admission owners required before IO")
        default_available_memory_bytes = vars(capacity).get("default_available_memory_bytes")
        default_device_memory_snapshot = vars(memory).get("default_device_memory_snapshot")
        DeviceMemorySnapshot = vars(memory).get("DeviceMemorySnapshot")
        if (type(default_available_memory_bytes) is not FunctionType
                or default_available_memory_bytes.__globals__ is not vars(capacity)
                or type(default_device_memory_snapshot) is not FunctionType
                or default_device_memory_snapshot.__globals__ is not vars(memory)
                or type(DeviceMemorySnapshot) is not type):
            _fail("source", "actual Native admission bindings changed")
        available, snapshot = default_available_memory_bytes(), default_device_memory_snapshot()
        if type(available) is not int or 2*host > available:
            _fail("read_headroom", "genuine HOST half-headroom")
        if (type(snapshot) is not DeviceMemorySnapshot or snapshot.device_id != self.device
                or snapshot.allocator_supported is not True or snapshot.pool_limit_bytes != 2 << 30):
            _fail("pool", "genuine original active2GiB pool required")
        bed_exchange_device._check_device_admission(device, device_id=self.device, safety_fraction=0.5)
        return dict(host_bytes=host, device_bytes=device, read_plan=plan, known_subset_only=True,
                    constructor_allocation=None, external_runtime=None, claim=CLAIM)

    def _identity(self, evidence):
        identity = _budget_identity(evidence)
        identity["native_source_refs"] = [dict(module=n, size=r.size, sha256=r.sha256) for n, r in self.native_source_refs]
        identity["masy_source"] = [{k: row[k] for k in ("module", "size", "sha256")} for row in self.source_rows]
        return identity

    def _authenticate(self):
        self.prelude()
        shared = _shared()
        for ref in (self.budget.current_runtime_ref, self.budget.factory_source):
            shared._read_reference(ref)
        if shared._read_reference(self.budget.grant) != shared.c64_qualification_budget_bytes(self.budget):
            _fail("authority", "fresh genuine complete C64 budget bytes changed")
        for name, ref in self.budget.source_bindings:
            module = sys.modules.get(name)
            if (type(module) is not ModuleType or type(vars(module).get("__file__")) is not str
                    or vars(module)["__file__"] != ref.path):
                _fail("source_origin", name)
            shared._read_reference(ref)
        source_rows = []
        for name in ("windowed_device_c64_checkpoint", "windowed_device_checkpoint", "dense_boundary_store",
                     "dense_device_resources", "windowed_device_c64_bed", "windowed_device_c64_runner"):
            path = Path(__file__).with_name(name + ".py")
            row = _file_row(path, 2 << 20)
            if path.lstat().st_mode & 0o222:
                _fail("source_custody", "locked read-only MASY Source required")
            source_rows.append(dict(module="masy." + name, **row))
        if self.source_rows and self.source_rows != tuple(source_rows):
            _fail("source_currentness", "actual MASY Source bytes changed")
        self.source_rows = tuple(source_rows)
        self.native_images = _authenticate_native_sources(self.native_source_refs)
        from .windowed_device_c64_bed import _require_completed_scale_api
        _require_completed_scale_api(preflight=True)  # Existing complete Native image/Source gate.
        _authenticate_classes(self)

    def check(self, saved):
        _keys(saved, ("schema", "source_identity", "environment", "environment_sha256", "evidence", "grant",
                      "native_source_refs", "masy_source"), "C64 runtime")
        self.budget.validated(); self._authenticate()
        current = _shared().c64_qualification_budget_record(self.budget)
        if (saved["schema"] != RUNTIME_SCHEMA or _json(saved["source_identity"]) != _json(self._identity(current))
                or saved["environment"] != _environment(self.cp)
                or saved["environment_sha256"] != _sha(saved["environment"])
                or _json(_budget_identity(saved["evidence"])) != _json(_budget_identity(current))
                or [{k: row[k] for k in ("module", "size", "sha256")} for row in saved["native_source_refs"]]
                   != [dict(module=n, size=r.size, sha256=r.sha256) for n, r in self.native_source_refs]
                or [{k: row[k] for k in ("module", "size", "sha256")} for row in saved["masy_source"]]
                   != [{k: row[k] for k in ("module", "size", "sha256")} for row in self.source_rows]):
            _fail("runtime_identity", "original exact scientific/Source/profile/caps/UUID/runtime differ")
        return copy.deepcopy(saved["source_identity"]), saved["environment_sha256"]


def _generated_binding_methods(cls, source):
    """The one Source-declared frozen dataclass: genuine stdlib metadata only.

    Generate method CodeTypes from its checked real Field metadata, never a
    Native instance/class substitute and never execute the compiled module.
    """
    if cls.__name__ != "PrefixPacketBinding" or cls.__module__ != E1_MODULE:
        _fail("native_image", "unknown generated Native class")
    names = tuple(n.target.id for n in source.body if isinstance(n, ast.AnnAssign)
                  and isinstance(n.target, ast.Name))
    fields = dataclasses.fields(cls)
    if (tuple(f.name for f in fields) != names or len(fields) != 16
            or any(type(f) is not dataclasses.Field or not f.init or not f.repr or not f.compare
                   or f.kw_only or f.default_factory is not dataclasses.MISSING for f in fields)):
        _fail("native_image", "Source-exact PrefixPacketBinding fields required")
    if any(f.default is not dataclasses.MISSING for f in fields[:-2]) or any(f.default is not None for f in fields[-2:]):
        _fail("native_image", "Source-exact optional version/generation defaults required")
    params = cls.__dict__.get("__dataclass_params__")
    if (type(params) is not dataclasses._DataclassParams or params.init is not True
            or params.repr is not True or params.eq is not False or params.order is not False
            or params.unsafe_hash is not False or params.frozen is not True):
        _fail("native_image", "Source dataclass policy changed")
    ns = sys.modules[E1_MODULE].__dict__
    std, kw = dataclasses._fields_in_init_order(fields)
    init = dataclasses._init_fn(fields, std, kw, True, True, "self", ns, False)
    repr_ = dataclasses._repr_fn(fields, ns)
    set_, del_ = dataclasses._frozen_get_del_attr(cls, fields, ns)
    if cls.__dict__.get("__match_args__") != names:
        _fail("native_image", "Source dataclass match fields changed")
    return dict(__init__=init, __repr__=repr_, __setattr__=set_, __delattr__=del_)


def _same_function_image(actual, expected):
    if (type(actual) is not FunctionType or type(expected) is not FunctionType
            or actual.__code__ != expected.__code__ or actual.__defaults__ != expected.__defaults__
            or actual.__kwdefaults__ != expected.__kwdefaults__
            or actual.__globals__ is not expected.__globals__):
        _fail("native_image", "generated method image/default/global drift")
    left, right = actual.__closure__ or (), expected.__closure__ or ()
    if len(left) != len(right):
        _fail("native_image", "generated closure layout")
    for a, b in zip(left, right, strict=True):
        a, b = a.cell_contents, b.cell_contents
        if type(a) is FunctionType and type(b) is FunctionType:
            _same_function_image(a, b)
        elif type(a) is set and type(b) is set:
            if a or b:
                _fail("native_image", "active generated repr closure")
        elif a is not b:
            _fail("native_image", "generated closure binding changed")


def _authenticate_native_sources(refs):
    """Exactly two locked Sources; compilation makes metadata, never executes."""
    authority = _scope_authority()
    authority.prelude()
    shared, result, compiled_images = _shared(), [], []
    maple = sys.modules.get("maple")
    if type(maple) is not ModuleType or type(vars(maple).get("__file__")) is not str:
        _fail("native_source", "actual locked Native package required")
    package_root = Path(vars(maple)["__file__"]).parent
    for name, ref in refs:
        path = package_root.joinpath(*name.split(".")[1:]).with_suffix(".py")
        if ref.path != str(path):
            _fail("native_source", "ref must name the selected Native package Source")
        raw = shared._read_reference(ref)
        source_tree = ast.parse(raw, filename=ref.path)
        image = compile(raw, ref.path, "exec", dont_inherit=True, optimize=0)
        compiled_images.append(image)
        # Genuine CodeType metadata only. Initial raw/parser constructor costs
        # remain opaque; every visible code/slot/text/current-next-error word
        # is positive before import, class lookup or Native function dispatch.
        authority.native_code_bytes = max(authority.native_code_bytes, _native_code_words(compiled_images))
        authority.prelude()
        expected = {}
        def collect(code):
            expected.setdefault(code.co_qualname, []).append(code)
            for child in code.co_consts:
                if type(child) is CodeType:
                    collect(child)
        collect(image)
        module = importlib.import_module(name)
        if (type(module) is not ModuleType or vars(module).get("__file__") != ref.path
                or vars(module).get("__spec__") is None
                or vars(module)["__spec__"].origin != ref.path):
            _fail("native_source", "actual imported/cached Native origin changed")
        ns = vars(module)
        classes = {n.name: n for n in source_tree.body if isinstance(n, ast.ClassDef)}
        for declaration in source_tree.body:
            if isinstance(declaration, ast.FunctionDef):
                function = dict.get(ns, declaration.name)
                if type(function) is not FunctionType or function.__globals__ is not ns:
                    _fail("native_image", "missing or foreign Source function: " + declaration.name)
            elif isinstance(declaration, ast.ClassDef):
                kind = dict.get(ns, declaration.name)
                if type(kind) is not type or type.__getattribute__(kind, "__module__") != name:
                    _fail("native_image", "missing or foreign Source class: " + declaration.name)
                for method in declaration.body:
                    if isinstance(method, ast.FunctionDef) and method.name not in type.__getattribute__(kind, "__dict__"):
                        _fail("native_image", "missing Source method: " + declaration.name + "." + method.name)
        bindings = []
        for member, value in tuple(ns.items()):
            if type(value) is FunctionType and value.__module__ == name:
                codes = expected.get(value.__code__.co_qualname, ())
                if not any(value.__code__ == code for code in codes) or value.__globals__ is not ns:
                    _fail("native_image", name + ":" + member)
                bindings.append((ns, member, value, value.__code__))
            elif type(value) is type and value.__module__ == name:
                declaration = classes.get(value.__name__)
                if declaration is None:
                    _fail("native_image", "unknown class in locked Native image")
                generated = (_generated_binding_methods(value, declaration)
                             if value.__name__ == "PrefixPacketBinding" else {})
                for field, descriptor in tuple(vars(value).items()):
                    fn = (descriptor.__func__ if type(descriptor) in (classmethod, staticmethod)
                          else descriptor.fget if type(descriptor) is property else descriptor)
                    if type(fn) is not FunctionType:
                        continue
                    if field in generated:
                        _same_function_image(fn, generated[field])
                    else:
                        codes = expected.get(fn.__code__.co_qualname, ())
                        if not any(fn.__code__ == code for code in codes) or fn.__globals__ is not ns:
                            _fail("native_image", name + ":" + value.__name__ + "." + field)
                    bindings.append((vars(value), field, descriptor, fn.__code__))
        if not bindings:
            _fail("native_image", "empty locked Native image is unsupported")
        if name == PREFIX_MODULE and ns.get("E1_MODULE") != E1_MODULE:
            _fail("native_image", "actual prefix E1 binding changed")
        # Literal immutable controls participate; mutable diagnostic counters
        # are retained, never required to equal their import-time zero values.
        immutable_controls = {"LEAF", "LANES", "MAX_DEPTH", "MAX_ENCODE_CELLS", "THREADS",
                              "MAX_BLOCKS", "_SOURCE", "E1_MODULE"}
        for assignment in source_tree.body:
            if isinstance(assignment, ast.Assign) and type(assignment.value) is ast.Constant:
                for target in assignment.targets:
                    if isinstance(target, ast.Name) and target.id in immutable_controls:
                        actual, wanted = ns[target.id], assignment.value.value
                        if type(actual) is not type(wanted) or actual != wanted:
                            _fail("native_image", "literal control changed: " + target.id)
        for assignment in source_tree.body:
            if not isinstance(assignment, ast.Assign):
                continue
            for target in assignment.targets:
                if (isinstance(target, ast.Name) and target.id in {"IMPLEMENTATIONS", "SUPPORTED_BASE_DTYPES", "NAMESPACES", "OPTIONS"}
                        and isinstance(assignment.value, ast.Tuple)
                        and all(isinstance(v, ast.Constant) and type(v.value) is str for v in assignment.value.elts)):
                    wanted = tuple(v.value for v in assignment.value.elts)
                    actual = dict.get(ns, target.id)
                    if type(actual) is not tuple or any(type(v) is not str for v in actual) or actual != wanted:
                        _fail("native_image", "Source tuple control changed: " + target.id)
        if shared._read_reference(ref) != raw:
            _fail("native_source", "Source changed across metadata import/image check")
        for owner, member, original, code in bindings:
            current = owner.get(member)
            fn = (current.__func__ if type(current) in (classmethod, staticmethod)
                  else current.fget if type(current) is property else current)
            if current is not original or type(fn) is not FunctionType or fn.__code__ is not code:
                _fail("native_image", "Native binding changed during authentication")
        result.append((module, tuple(bindings), image))
    return tuple(result)


def _native_code_words(images):
    codes = slots = text = 0
    def visit(code, depth=0):
        nonlocal codes, slots, text
        if type(code) is not CodeType or depth > 24 or codes >= 8192:
            _fail("native_image", "bounded genuine Native CodeType forest required")
        codes += 1
        slots += 1 + code.co_nlocals + code.co_stacksize + len(code.co_cellvars) + len(code.co_freevars)
        text += len(code.co_code) + len(code.co_linetable) + len(code.co_exceptiontable)
        text += sum(len(value) for value in code.co_consts if type(value) is str)
        for value in code.co_consts:
            if type(value) is CodeType:
                visit(value, depth+1)
    for code in images:
        visit(code)
    return 3*(32*codes + 8*slots + 25*(32*codes + 8*slots) + 4*text)


def _native_prefix():
    authority = _scope_authority()
    authority.prelude()
    images = _authenticate_native_sources(authority.native_source_refs)
    prefix = images[1][0]
    terms = vars(prefix)["packet_terms_for"](20, 1)
    _integer(terms, 4000, "genuine E1 packet terms", 1)
    for module, bindings, image in images:
        for owner, name, original, code in bindings:
            descriptor = owner.get(name)
            function = (descriptor.__func__ if type(descriptor) in (classmethod, staticmethod)
                        else descriptor.fget if type(descriptor) is property else descriptor)
            if descriptor is not original or function.__code__ is not code:
                _fail("native_image", "prefix owner changed across packet call")
    return terms

@dataclasses.dataclass(frozen=True)
class C64BoundaryImageLimits:
    max_numeric_bytes: int = 1 << 20
    max_metadata_bytes: int = 4 << 20
    max_serialized_bytes: int = 6 << 20
    max_header_bytes: int = 4096
    max_nodes: int = 100000
    max_depth: int = 24
    max_files: int = 7
    max_cells: int = 64
    max_classes: int = 1
    max_voxels: int = 20
    max_cohorts: int = 4000
    max_events: int = 4
    max_receipt_bytes: int = 256 << 10
    max_receipts_bytes: int = 2 << 20
    native_max_bytes: int = 64 << 20
    max_global_raw_bytes: int = 512 << 20
    max_cache_bytes: int = 1 << 20

    def __post_init__(self):
        ceiling = (1 << 20, 4 << 20, 6 << 20, 4096, 100000, 24, 7, 64, 1, 20, 4000, 4, 256 << 10, 2 << 20, 64 << 20, 512 << 20, 1 << 20,)
        if type(self) is not C64BoundaryImageLimits or set(vars(self)) != {
            f.name for f in dataclasses.fields(self)
        }:
            _fail("limit_layout", "exact closed C64BoundaryImageLimits required")
        for field, cap in zip(dataclasses.fields(self), ceiling, strict=True):
            value = getattr(self, field.name)
            if type(value) is not int or not 1 <= value <= cap:
                _fail(
                    "limit_cap", f"{field.name} must be a plain positive int <= {cap}"
                )

        if (self.max_cells, self.max_classes, self.max_voxels, self.max_cohorts, self.max_events) != (64, 1, 20, 4000, 4):
            _fail("profile_cap", "exact C64/K1/V20/4000/four-event dimensional capacities required")


@dataclasses.dataclass(frozen=True)
class C64BoundaryStoreLimits:
    max_store_bytes: int = 512 << 20
    max_entries: int = 4096
    max_accepted: int = 5
    max_orphans: int = 4
    free_reserve_bytes: int = 1 << 30
    free_reserve_inodes: int = 256
    max_memory_bytes: int = 5 << 30
    max_head_bytes: int = 32 << 10
    max_generation_bytes: int = 128 << 10
    max_intent_bytes: int = 32 << 10
    max_runtime_metadata_bytes: int = 8 << 20
    image: C64BoundaryImageLimits = dataclasses.field(
        default_factory=C64BoundaryImageLimits
    )

    def __post_init__(self):
        names = tuple(f.name for f in dataclasses.fields(C64BoundaryStoreLimits))
        if type(self) is not C64BoundaryStoreLimits or set(vars(self)) != set(names):
            _fail("limit_layout", "exact closed C64BoundaryStoreLimits required")
        caps = (512 << 20, 4096, 5, 4, 1 << 30, 256, 5 << 30, 32 << 10, 128 << 10, 32 << 10, 8 << 20,)
        for name, cap in zip(names[:-1], caps, strict=True):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= cap:
                _fail("limit_cap", name)
        if self.free_reserve_bytes != 1 << 30 or self.free_reserve_inodes != 256:
            _fail("limit_cap", "v1 retains the full1GiB/256inode free reserve")
        C64BoundaryImageLimits.__post_init__(self.image)

        if self.max_accepted != 5:
            _fail("profile_cap", "five accepted boundaries required")


_C64_TYPES = (('masy.windowed_device_c64_bed', 'DeviceWindowedC64BedLimits', ('max_bed_device_bytes', 'max_known_host_bytes', 'max_known_device_bytes', 'max_stage_logical_read_bytes')), ('maple.surface.active_layer.bed_exchange_device', 'DeviceWindowedK1DimensionProfile', ('ny', 'nx', 'n_voxels', 'window_voxels', 'n_classes', 'hop_bins')), ('maple.surface.active_layer.bed_exchange_device', 'DeviceK1DimensionImpactLimits', ('profile', 'max_operations', 'max_gather_columns', 'max_journal_records', 'max_cells', 'max_classes', 'max_voxels', 'max_impact_rows', 'max_new_device_bytes', 'max_host_staging_bytes', 'max_logical_deep_read_bytes', 'max_mapped_bytes', 'device_reserve_bytes', 'host_reserve_bytes', 'device_safety_fraction')), ('maple.aeolian.flux.windowed_step_device', 'DeviceWindowedK1DimensionStepLimits', ('profile', 'bed_limits', 'creep_limits', 'delta_budgets', 'max_mobile_slots', 'max_state_leaves', 'max_config_leaves', 'max_leaf_bytes', 'max_snapshot_device_bytes', 'max_snapshot_host_bytes', 'transport_body_device_bytes', 'transport_body_host_bytes', 'boundary_host_bytes', 'max_step_h2d_bytes', 'max_step_d2h_bytes', 'reserve_host_bytes', 'device_reserve_bytes', 'device_safety_fraction')), ('maple.aeolian.scheduler.windowed_device', 'DeviceWindowedK1DimensionWindBlockLimits', ('profile', 'step', 'solve', 'validation', 'commit', 'max_outer_steps', 'max_block_host_bytes', 'max_block_device_bytes', 'source_control_host_bytes', 'max_image_operators', 'max_reference_items', 'max_block_h2d_bytes', 'max_block_d2h_bytes', 'reserve_host_bytes', 'device_reserve_bytes', 'device_safety_fraction')))

def _class_table(limits):
    C64BoundaryImageLimits.__post_init__(limits)
    result = codec._class_table()
    for module, name, fields in _C64_TYPES:
        tag = module + ":" + name
        if tag in result:
            _fail("class_schema", "C64 tag overlaps original roster")
        result[tag] = (fields, False)
    return result


def _class(tag, limits):
    table = _class_table(limits)
    if type(tag) is not str or tag not in table:
        _fail("class", "outside the closed original plus five C64 rows")
    additions = {module + ":" + name for module, name, _fields in _C64_TYPES}
    if tag not in additions:
        return codec._class(tag)
    authority = _scope_authority()  # No new-class construction from an unissued codec call.
    _check_class_images(authority)
    module_name, name = tag.split(":")
    module = sys.modules.get(module_name)
    if type(module) is not ModuleType:
        _fail("class_source", "already authenticated owning module required")
    ns = module.__dict__
    cls = dict.get(ns, name)
    fields, enum = table[tag]
    if (type(cls) is not type or type.__getattribute__(cls, "__module__") != module_name
            or type.__getattribute__(cls, "__qualname__") != name
            or not dataclasses.is_dataclass(cls)
            or tuple(f.name for f in dataclasses.fields(cls)) != fields
            or any(not f.init for f in dataclasses.fields(cls))):
        _fail("class_schema", tag)
    if not any(saved_class is cls for _module, _name, saved_class, _rows in authority.class_images):
        _fail("class_source", "class is not the freshly authenticated owning image")
    _check_class_images(authority)
    return cls, fields, enum


def _authenticate_classes(authority):
    rows = []
    for module_name, name, expected_fields in _C64_TYPES:
        module = sys.modules.get(module_name)
        if type(module) is not ModuleType:
            _fail("class_source", "actual already-loaded Native/MASY class owner required")
        cls = dict.get(module.__dict__, name)
        if (type(cls) is not type or type.__getattribute__(cls, "__module__") != module_name
                or tuple(f.name for f in dataclasses.fields(cls)) != expected_fields
                or any(not f.init for f in dataclasses.fields(cls))):
            _fail("class_source", "closed Source-declared class/fields differ")
        image = []
        for key, descriptor in tuple(type.__getattribute__(cls, "__dict__").items()):
            fn = (descriptor.__func__ if type(descriptor) in (classmethod, staticmethod)
                  else descriptor.fget if type(descriptor) is property else descriptor)
            if type(fn) is FunctionType:
                image.append((key, descriptor, fn.__code__, fn.__globals__, fn.__defaults__, fn.__kwdefaults__))
        rows.append((module, name, cls, tuple(image)))
    if authority.class_images:
        _check_class_images(authority)
        if any(left[:3] != right[:3] for left, right in zip(authority.class_images, rows, strict=True)):
            _fail("class_source", "actual class owning objects changed")
    else:
        authority.class_images = tuple(rows)
    _check_class_images(authority)


def _check_class_images(authority):
    if type(authority.class_images) is not tuple or len(authority.class_images) != 5:
        _fail("class_source", "five authenticated class images required")
    for module, name, cls, rows in authority.class_images:
        if type(module) is not ModuleType or dict.get(module.__dict__, name) is not cls:
            _fail("class_source", "class binding changed")
        namespace = type.__getattribute__(cls, "__dict__")
        for key, original, code, globals_, defaults, kwdefaults in rows:
            descriptor = namespace.get(key)
            fn = (descriptor.__func__ if type(descriptor) in (classmethod, staticmethod)
                  else descriptor.fget if type(descriptor) is property else descriptor)
            if (descriptor is not original or type(fn) is not FunctionType or fn.__code__ is not code
                    or fn.__globals__ is not globals_ or fn.__defaults__ is not defaults or fn.__kwdefaults__ is not kwdefaults):
                _fail("class_source", "actual class function/code/globals/defaults changed")


def _budget_identity(record):
    """Stable scientific/Source operands; actual role references stay evidence."""
    result = copy.deepcopy(record)
    result.pop("current_runtime_ref")
    result["factory_source"] = {k: result["factory_source"][k] for k in ("size", "sha256")}
    result["source_bindings"] = [dict(module=r["module"], size=r["size"], sha256=r["sha256"])
                                 for r in result["source_bindings"]]
    for name in ("opaque_external_bounds", "opaque_primitive_units"):
        if name in result:
            result[name] = [list(row) for row in result[name]]
    return result


def _original_preflight(record, source, environment, limits):
    _keys(record, ("schema", "wire", "payload", "profile", "source_identity",
                  "environment_sha256", "manifest_sha256"), "original capsule")
    _plain_record(record, limits)
    if (record["schema"] != ORIGINAL_SCHEMA or record["profile"] != PROFILE
            or _json(record["source_identity"]) != _json(source)
            or record["environment_sha256"] != environment):
        _fail("original_identity", "original profile/Source/environment")
    _wire_preflight(record["wire"], limits)
    _wire_root(record["wire"], _ORIGINAL_FIELDS, "original root")
    first = record["wire"]["root"]["v"][0][1]
    if first.get("t") != "dict" or tuple(v[0] for v in first["v"]) != _INPUT_FIELDS:
        _fail("original_inputs", "complete ordered original inputs")


def _write_wire(directory, material, *, schema, limits, authority, extra):
    """One admitted uint8 NPY plus closed manifest fits old inventory roles."""
    import numpy as np
    from maple.core.backend import to_host
    authority.prelude()
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
    _plain_record(dict(record, manifest_sha256="0" * 64), limits)
    record["manifest_sha256"] = _sha(record)
    _plain_record(record, limits)
    store._sealed_write(directory, "manifest.json", record, limits.max_metadata_bytes)
    return record


def _read_wire(directory, *, expected_file_sha=None, expected_seal=None, schema, limits, expand):
    import numpy as np
    _scope_authority().prelude()
    directory = Path(directory)
    raw = _read_file(directory / "manifest.json", limits.max_metadata_bytes)
    if expected_file_sha is not None and hashlib.sha256(raw).hexdigest() != expected_file_sha:
        _fail("manifest_hash", "raw manifest hash")
    record = oi._strict_json(raw, _record_json_limits(limits))
    if type(record) is not dict or record.get("schema") != schema:
        _fail("schema", "exact companion/original schema")
    body = dict(record); seal = body.pop("manifest_sha256", None)
    if _sha(body) != seal or expected_seal is not None and seal != expected_seal:
        _fail("manifest_hash", "manifest seal")
    _plain_record(record, limits)
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
    prelude = authority.prelude()
    host_request = sum(host for _ in banks) + retained_host + (64 << 20) + (2 << 20) + prelude["host_bytes"]
    device_request = sum(device for _ in banks) + retained_device + (256 << 20) + prelude["device_bytes"]
    if (host_request > min(5 << 30, authority.limits.max_memory_bytes, authority.bridge_limits.max_host_bytes)
            or device_request > min(2 << 30, authority.bridge_limits.max_device_bytes)):
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


def _snapshot_preflight(path, limits, snapshot_row):
    """Authenticate the entire bounded archive BEFORE each ZipFile parser."""
    import numpy as np
    _scope_authority().prelude()
    _keys(snapshot_row, ("path", "size", "sha256"), "Native archive row")
    if snapshot_row["path"] != str(path):
        _fail("native_path", "exact authenticated archive path required")
    _check_file(snapshot_row, limits.native_max_bytes)
    terms = _native_prefix()
    expected = {
        "window_kg": ((8, 8, 3, 1), "<f8"), "base": ((8, 8), "<i4"),
        "deep_by_class_kg": ((8, 8, 1), "<f8"), "deep_total_kg": ((8, 8), "<f8"),
        "deep_room_kg": ((8, 8), "<f8"), "prefix_packet": ((8, 8, terms), "<f8"),
        "active_mass_kg": ((8, 8, 1), "<f8"), "available_mass_kg": ((8, 8, 1), "<f8"),
        "bound_mass_kg": ((8, 8, 1), "<f8"), "water_depth_m": ((8, 8), "<f8"),
        "mobile_mass_kg": ((8, 8, 1), "<f8"),
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
                    if (shape != () or dtype.kind != "U" or dtype.itemsize % 4
                            or not 0 < dtype.itemsize <= limits.max_metadata_bytes
                            or handle.tell() + dtype.itemsize != info.file_size):
                        _fail("native_header", "bounded exact scalar Unicode metadata")
                    raw = handle.read(dtype.itemsize + 1)
                    if len(raw) != dtype.itemsize:
                        _fail("native_header", "metadata exact bytes")
                    text = str(np.frombuffer(raw, dtype=dtype, count=1)[0])
                    meta = oi._strict_json(text.encode("utf-8"), limits)
                else:
                    if (tuple(shape), dtype.str) != expected[name]:
                        _fail("native_header", name)
                    size = math.prod(shape) * dtype.itemsize
                    numeric += size
                    if handle.tell() + size != info.file_size:
                        _fail("native_header", "exact Native payload size")
    from maple.surface.voxels.windowed_snapshot import META_KEYS, SCHEMA as native_schema
    _keys(meta, META_KEYS, "Native metadata")
    if (meta["schema"] != native_schema or type(meta["time_s"]) not in (int, float)
            or type(meta["class_ids"]) is not list or len(meta["class_ids"]) != 1
            or type(meta["class_ids"][0]) is not str or not 1 <= len(meta["class_ids"][0]) <= 4096):
        _fail("native_metadata", "closed schema/time/K1 class ordering")
    encode_time(float(meta["time_s"]))
    _integer(meta["step"], 4, "Native step")
    _integer(meta["generation"], (1 << 63) - 1, "Native generation")
    if meta["n_voxels"] != 20 or type(meta["n_voxels"]) is not int:
        _fail("native_metadata", "voxel count")
    _deep_root(meta["store_dir"])
    _digest(meta["manifest_digest"], "Native manifest")
    _digest(meta["header_digest"], "Native header")
    _keys(meta["provenance"], ("schema", "source_identity_sha256", "case_sha256", "schedule_sha256",
                              "accepted_cursor", "parent_head_token"), "Native provenance")
    provenance = meta["provenance"]
    if provenance["schema"] != PHYSICAL_SCHEMA + ".provenance":
        _fail("native_provenance", "closed C64 provenance schema")
    for name in ("source_identity_sha256", "case_sha256", "schedule_sha256"):
        _digest(provenance[name], "Native provenance " + name)
    cursor = _integer(provenance["accepted_cursor"], 4, "Native provenance cursor")
    if cursor != meta["step"] or (provenance["parent_head_token"] is None) != (cursor == 0):
        _fail("native_provenance", "parent absence iff genesis and same cursor")
    if cursor:
        _digest(provenance["parent_head_token"], "Native parent token")
    _check_file(snapshot_row, limits.native_max_bytes)
    return meta, numeric


def _deep_root(value):
    if type(value) is not str:
        _fail("deep_path", "plain canonical absolute Native store_dir required")
    root = Path(_path(value))
    held = oi._open_dir(root)  # All parents are traversed with O_NOFOLLOW.
    try:
        if str(root.resolve(strict=True)) != value:
            _fail("deep_path", "Native store_dir must be canonical absolute")
    finally:
        os.close(held)
    return root


def _deep_files(version, limits):
    """Complete lstat-only roster FIRST; overlays refused BEFORE payload hash."""
    _scope_authority().prelude()
    root = _deep_root(str(version.store_dir))
    paths, total, entries = [root / "store.json"], 0, 1
    for directory, pattern in ((root / "chunks", r"[0-9a-f]{64}\.f64"),
                               (root / "versions", r"[0-9a-f]{64}\.json")):
        held = oi._open_dir(directory)
        try:
            with os.scandir(held) as iterator:
                for entry in iterator:
                    entries += 1
                    if entries > 4096:
                        _fail("deep_entries", "retained Native files exceed4096")
                    if (not re.fullmatch(pattern, entry.name)
                            or not entry.is_file(follow_symlinks=False)):
                        _fail("deep_file", "closed full-v1 roster; overlay/special file refused")
                    paths.append(directory / entry.name)
        finally:
            os.close(held)
    for path in paths:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink not in (1, 2) or not 0 < info.st_size <= 64 << 20:
            _fail("deep_file", "bounded ordinary deep file")
        total += info.st_size
        if total > 64 << 20:
            _fail("deep_bytes", "retained Native store exceeds64MiB")
    # Every full-v1 metadata object is closed before hashing any chunk payload.
    rows = [_file_row(path, limits.max_metadata_bytes) for path in paths if path.suffix == ".json"]
    _deep_metadata(root, rows, limits)
    rows.extend(_file_row(path, 64 << 20) for path in paths if path.suffix == ".f64")
    return sorted(rows, key=lambda row: row["path"])


def _deep_metadata(root, rows, limits, *, selected=None):
    """Strict duplicate-free bounded JSON before genuine Native load_version."""
    from maple.surface.voxels.deep_store import DeepStoreHeader
    by_path = {row["path"]: row for row in rows}
    descriptor = str(root / "store.json")
    if descriptor not in by_path:
        _fail("deep_reference", "retained store.json required")
    def read(path):
        row = by_path[str(path)]
        _keys(row, ("path", "size", "sha256"), "deep metadata row")
        _check_file(row, limits.max_metadata_bytes)
        raw = _read_file(path, limits.max_metadata_bytes)
        if len(raw) != row["size"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            _fail("deep_metadata", "authenticated exact metadata before parse")
        return oi._strict_json(raw, limits)
    descriptor_record = read(root / "store.json")
    _keys(descriptor_record, ("format", "kind", "header", "header_digest", "max_store_bytes"), "deep descriptor")
    if descriptor_record["format"] != "maple-deep-store/1" or descriptor_record["kind"] != "store":
        _fail("deep_format", "full-v1 Native descriptor only")
    header = DeepStoreHeader.from_json_dict(descriptor_record["header"])
    count = (64 + header.chunk_columns - 1) // header.chunk_columns
    if ((header.ny, header.nx, header.n_voxels, header.window_voxels, header.n_classes) != (8, 8, 20, 3, 1)
            or count != 1 or header.n_chunks != count or header.column_bytes != 20*1*8
            or descriptor_record["header_digest"] != header.digest()):
        _fail("deep_header", "genuine C64/K1/V20/W3 single-chunk full-v1 header required")
    _integer(descriptor_record["max_store_bytes"], 512 << 20, "Native deep storage cap", 1)
    n_versions, selected_record = 0, None
    for row in rows:
        path = Path(row["path"])
        if path.parent != root / "versions":
            continue
        record = read(path)
        _keys(record, ("format", "kind", "header", "header_digest", "version_id", "parent_version_id",
                       "parent_manifest_digest", "journal_digest", "chunks"), "Native full-v1 version")
        if (record["format"] != "maple-deep-store/1" or record["kind"] != "version"
                or record["header"] != descriptor_record["header"] or record["header_digest"] != header.digest()
                or row["sha256"] != path.stem):
            _fail("deep_format", "exact full-v1 metadata/header/content-address join")
        version = _integer(record["version_id"], (1 << 63) - 1, "Native version")
        parent, digest = record["parent_version_id"], record["parent_manifest_digest"]
        if (parent is None) != (digest is None):
            _fail("deep_parent", "complete Native parent provenance required")
        if parent is not None:
            _integer(parent, (1 << 63) - 1, "Native parent version")
            _digest(digest, "Native parent digest")
            if parent != version - 1:
                _fail("deep_parent", "consecutive Native immutable version provenance")
        elif version != 0:
            _fail("deep_parent", "only genesis may lack Native parent")
        _digest(record["journal_digest"], "Native journal")
        chunks = record["chunks"]
        if type(chunks) is not list or len(chunks) != count:
            _fail("deep_chunks", "exact derived single chunk required")
        chunk = chunks[0]
        _keys(chunk, ("index", "cell_start", "n_cells", "sha256", "nbytes"), "Native full chunk")
        for name, value in (("index", 0), ("cell_start", 0), ("n_cells", 64), ("nbytes", 64*20*1*8)):
            if type(chunk[name]) is not int or chunk[name] != value:
                _fail("deep_chunk", name)
        _digest(chunk["sha256"], "Native chunk hash")
        # On read the complete authenticated file roster is already supplied.
        if any(Path(r["path"]).suffix == ".f64" for r in rows):
            payload = by_path.get(str(root / "chunks" / (chunk["sha256"] + ".f64")))
            if payload is None or payload["size"] != chunk["nbytes"] or payload["sha256"] != chunk["sha256"]:
                _fail("deep_chunk", "current and retained version chunk reference missing")
        n_versions += 1
        if path.stem == selected:
            selected_record = record
    if not n_versions or selected is not None and selected_record is None:
        _fail("deep_reference", "selected full-v1 version must be retained")
    # Other parsed version objects are not accumulated into a second history.
    # Their immutable files remain retained; work/hash checks are all positive.
    return header, selected_record


def _provenance(runner, source, parent):
    if (parent is None) != (runner._next == 0):
        _fail("native_provenance", "actual parent token absence iff genesis")
    if parent is not None:
        _digest(parent, "actual Native parent token")
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
    snapshot_row = _file_row(path / "native_state.npz", limits.native_max_bytes)
    meta, numeric = _snapshot_preflight(path / "native_state.npz", limits, snapshot_row)
    if meta["provenance"] != provenance:
        _fail("native_provenance", "Native saved metadata changed")
    deep = _deep_files(state.voxel_column.deep_store, limits)
    # Native writes fsync files. Combined HEAD must also retain/sync directory entries.
    store._sync_tree(state.voxel_column.deep_store.store_dir)
    record = dict(schema=PHYSICAL_SCHEMA, snapshot=snapshot_row,
                  deep_files=deep, metadata=meta, numeric_bytes=numeric,
                  source_identity=handle._header["source_identity"])
    record["manifest_sha256"] = _sha(record)
    store._sealed_write(path, "checkpoint.json", record, min(1 << 20, limits.max_metadata_bytes))
    return dict(format=PHYSICAL_SCHEMA, relative_path=str(path.relative_to(handle._root)),
                manifest_file_sha256=_file_row(path / "checkpoint.json", min(1 << 20, limits.max_metadata_bytes))["sha256"],
                identity=record["manifest_sha256"])


def _physical_read(root, ref, limits):
    _scope_authority().prelude()
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
    _keys(rec["snapshot"], ("path", "size", "sha256"), "snapshot row")
    if rec["snapshot"]["path"] != str(directory / "native_state.npz"):
        _fail("physical_path", "snapshot must be own child")
    meta, numeric = _snapshot_preflight(rec["snapshot"]["path"], limits, rec["snapshot"])
    if meta != rec["metadata"] or type(rec["numeric_bytes"]) is not int or numeric != rec["numeric_bytes"]:
        _fail("physical_metadata", "Native preflight differs")
    if type(rec["deep_files"]) is not list or not 1 <= len(rec["deep_files"]) <= 4096:
        _fail("deep_entries", "bounded retained deep roster")
    base = _deep_root(meta["store_dir"])
    seen, total = set(), 0
    # Validate the ENTIRE roster, including rejecting overlays, before any chunk hash.
    for row in rec["deep_files"]:
        _keys(row, ("path", "size", "sha256"), "deep file")
        p = Path(_path(row["path"]))
        _integer(row["size"], 64 << 20, "deep file bytes", 1); _digest(row["sha256"], "deep file hash")
        relative = p.relative_to(base) if p.is_relative_to(base) else None
        valid = (relative == Path("store.json") or relative is not None and len(relative.parts) == 2
            and ((relative.parts[0] == "versions" and re.fullmatch(r"[0-9a-f]{64}\.json", relative.name))
                 or (relative.parts[0] == "chunks" and re.fullmatch(r"[0-9a-f]{64}\.f64", relative.name))))
        if p in seen or not valid:
            _fail("deep_path", "duplicate/foreign/overlay deep file")
        seen.add(p); total += row["size"]
        if total > 64 << 20:
            _fail("deep_bytes", "deep aggregate cap before hash")
    header, version = _deep_metadata(base, rec["deep_files"], limits, selected=meta["manifest_digest"])
    if header.digest() != meta["header_digest"] or list(header.class_ids) != meta["class_ids"]:
        _fail("deep_header", "Native archive/retained header and class ordering differ")
    for row in rec["deep_files"]:
        _check_file(row, 64 << 20)
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
        accepted_bed_label=runner._accepted_bed, accepted_private_label=runner._accepted_private,
        stop_receipt=runner._stop_receipt, active_evidence=runner._active_evidence,
        pending_water_counters=runner._pending_water_counters)


def _routing(runner, limits):
    held = runner._routing
    if held is None: return None
    runner._device_own._held_routing(held, runner._water)
    runner._device_own.routing_guard(None, held, runner._water, (8, 8), 1)
    return dict(source=held.source, velocity=held.velocity, phase=held.phase,
        generation=held.generation, terrain_sha256=_fingerprint(held.terrain, limits, runner._device_ops.xp),
        cache_fingerprints=[_fingerprint(value, limits, runner._device_ops.xp)
                            for value in (held.graph, held.network, held.grid)])


def runner_admission(runner, limits):
    from .windowed_device_c64_runner import DeviceWindowedC64Runner, _process_check
    from maple.aeolian.cohorts.dispatch import mobile_state_is_genuinely_empty
    if type(runner) is not DeviceWindowedC64Runner or set(vars(runner)) != set(_RUNNER_FIELDS):
        _fail("runner", "exact closed device-windowed owner")
    if (runner._stop_receipt is not None or runner._active_evidence is not None
            or runner._pending_water_counters is not None or runner._bed._active is not None
            or runner._device_resources.active or runner._device_own._stage is not None
            or runner._device_own._prepared is not None):
        _fail("not_quiescent", "accepted idle boundary only")
    C64BoundaryImageLimits.__post_init__(limits)
    if limits.max_cells < 64 or limits.max_classes < 1 or limits.max_voxels < 20 or limits.max_events < 4:
        _fail("profile_cap", "closed C64/K1/V20/four-event profile")
    runner._bed.verify(); runner._device_resources.verify()
    if type(runner._bed.context.runtime_bounds) is not _shared().DeviceWaterC64QualificationBudget:
        _fail("qualification_budget", "factory-only C64 boundary; imported CaseBudget refused")
    runner._bed._validate(runner._bed._accepted.state, runner._bed.time_s)
    cp = runner._device_ops.xp
    passive = runner._bed._compact(lambda state: cp.stack((
        cp.any(state.water.depth_m != 0.0) | cp.any(state.water.mobile_mass_by_cell_class_kg != 0.0),)), max_values=1)
    if passive.shape != (1,) or float(passive[0]) != 0.0:
        _fail("not_passive", "genuine native dry surface/mobile boundary required")
    soil = runner._device_resources._accepted.soil
    if runner._device_ops.any_nonzero(~cp.isfinite(soil) | (soil != 0.0) | cp.signbit(soil)):
        _fail("not_passive", "finite nonnegative positive-zero native dry-reset soil required")
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
    _plain_record(manifest, limits)
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
                or type(body) is not dict or body.get("schema") != "masy.windowed_device_c64_event_receipt.v1"
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


class _Prepared:
    __slots__ = ("owner", "evidence")
    def __init__(self, key, owner, evidence):
        if key is not _KEY: _fail("prepared", "private loader only")
        self.owner, self.evidence = owner, copy.deepcopy(evidence)
    def diagnostics(self): return copy.deepcopy(self.evidence)


def _owner(prepared):
    from .windowed_device_c64_runner import DeviceWindowedC64Runner
    if type(prepared) is not _Prepared or type(prepared.owner) is not DeviceWindowedC64Runner:
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
        "accepted_bed_label", "accepted_private_label", "stop_receipt", "active_evidence", "pending_water_counters"), "bookkeeping")
    if any(book[name] is not None for name in ("stop_receipt", "active_evidence", "pending_water_counters")):
        _fail("not_quiescent", "C64 companion must explicitly retain three passive None markers")
    from .windowed_device_c64_runner import _combine_water_counters
    bridge = inputs["context"]["bridge_limits"]
    authority = _scope_authority()
    if type(bridge) is not type(authority.bridge_limits) or dataclasses.asdict(bridge) != dataclasses.asdict(authority.bridge_limits):
        _fail("bridge_limits", "saved and genuine fresh full bridge fields differ")
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
            counters = _combine_water_counters(counters, last_water["entry"], last_water["event"], bridge)
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
                          ("versions_reserved", limits.max_new_versions),
                          ("maximum_host_predicate", limits.max_host_bytes),
                          ("maximum_device_predicate", limits.max_device_bytes)):
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


def _history(root, header, chain, limits, *, authority=None, ticket=None):
    """Metadata/bytes only: Native snapshot and wrapper share ONE clock/cursor."""
    _chain_admit(authority, ticket, limits)
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





def _chain_admit(authority, ticket, limits):
    actual = _scope_authority()
    if authority is not None and authority is not actual:
        _fail("authority", "chain requires the actual same fresh role authority")
    if ticket is not None:
        _checked_ticket(ticket)
        if ticket.authority is not actual:
            _fail("ticket", "chain ticket/authority mismatch")
    C64BoundaryStoreLimits.__post_init__(limits)
    actual.prelude()


def _transfer_snapshot():
    from maple.core.backend.arrays import read_transfer_counters
    counters = read_transfer_counters()  # Genuine pure-HOST locked observation.
    result = dataclasses.asdict(counters)
    names = ("host_to_device", "host_to_device_bytes", "device_to_host", "device_to_host_bytes", "scalar_reads", "synchronizations")
    # The actual owning DTO roster is checked below rather than inferred from
    # a DEVICE allocation or a compensated/fabricated zero.
    if set(result) != set(names) or any(type(k) is not str or type(v) is not int or v < 0 for k, v in result.items()):
        _fail("transfer_counters", "genuine complete six-field TransferCounters required")
    return result


def _prepare(path, *, artifact_root, expected_manifest_sha256, expected_source_identity,
             expected_environment_sha256, limits, authority, ticket):
    _checked_ticket(ticket)
    if ticket.authority is not authority:
        _fail("ticket", "same actual owning authority required")
    authority.check(authority.record)
    from maple.surface.voxels.windowed_snapshot import load_windowed_state_snapshot
    from .windowed_device_c64_runner import _restore_windowed_device_c64_runner
    ticket.before = dict(native=_transfer_snapshot())
    manifest, _payload, raw = _read_wire(path, expected_seal=expected_manifest_sha256,
                                       schema=SCHEMA, limits=limits, expand=False)
    _preflight(manifest, expected_source_identity, expected_environment_sha256, limits)
    artifacts = _Artifacts(artifact_root, manifest["originals"], manifest["genesis"], manifest["physical"], limits, raw)
    try:
        ref = manifest["originals"]
        original_path = Path(artifact_root) / ref["relative_path"]
        original, _payload, _raw = _read_wire(original_path, expected_file_sha=ref["manifest_file_sha256"],
            expected_seal=ref["identity"], schema=ORIGINAL_SCHEMA, limits=limits, expand=False)
        _original_preflight(original, expected_source_identity, expected_environment_sha256, limits)
        physical = _physical_read(artifact_root, manifest["physical"], limits)
        total = original["wire"]["numeric_bytes"] + manifest["wire"]["numeric_bytes"] + physical["numeric_bytes"]
        _boundary_admit(total, total, limits, authority)
        original2, original_payload, _raw = _read_wire(original_path,
            expected_file_sha=ref["manifest_file_sha256"], expected_seal=ref["identity"],
            schema=ORIGINAL_SCHEMA, limits=limits, expand=True)
        manifest2, payload, _raw = _read_wire(path, expected_seal=expected_manifest_sha256,
            schema=SCHEMA, limits=limits, expand=True)
        if original != original2 or manifest != manifest2:
            _fail("manifest_changed", "metadata changed before expansion")
        originals_host = _decode(original["wire"], original_payload, limits, authority.cp, host=True)
        current_host = _decode(manifest["wire"], payload, limits, authority.cp, host=True)
        _keys(originals_host, _ORIGINAL_FIELDS, "decoded originals")
        inputs_host = originals_host["inputs"]
        ctx = inputs_host["context"]
        # Physical archive, canonical deep root and every full-v1 metadata/chunk
        # row were authenticated above, BEFORE this genuine Native HOST loader.
        artifacts.check()
        saved = load_windowed_state_snapshot(physical["snapshot"]["path"], geometry=ctx["geometry"],
            grain_classes=ctx["grain_classes"], mass_resolution_kg=ctx["mass_resolution_kg"], max_bytes=limits.native_max_bytes)
        version = saved["voxel_column"].deep_store
        metadata = physical["metadata"]
        if (str(version.store_dir.resolve()) != metadata["store_dir"]
                or version.manifest_digest != metadata["manifest_digest"]
                or version.header.digest() != metadata["header_digest"]
                or list(version.header.class_ids) != metadata["class_ids"]
                or saved["voxel_column"].generation != metadata["generation"]):
            _fail("native_reference", "actual Native returned handle/header/version/class/root differs")
        _validate_book(inputs_host, current_host, manifest, saved)
        if (originals_host["inventory0"].tobytes() != current_host["bookkeeping"]["inventory0"].tobytes()
                or _json(physical["source_identity"]) != _json(expected_source_identity)):
            _fail("original_identity", "original inventory/Source changed")
        provenance = metadata["provenance"]
        if (provenance["source_identity_sha256"] != _sha(expected_source_identity)
                or provenance["case_sha256"] != _sha(inputs_host["case"])
                or provenance["accepted_cursor"] != manifest["state"]["next_index"]
                or provenance["schedule_sha256"] != _sha(inputs_host["schedule"])):
            _fail("native_provenance", "closed Source/case/schedule/cursor join")
        artifacts.check()
        inputs = _decode(original["wire"], original_payload, limits, authority.cp, inputs_only=True)
        current = _decode(manifest["wire"], payload, limits, authority.cp)
        ticket.inputs, ticket.current, ticket.host_snapshot = inputs, current, saved
        ticket.manifest, ticket.limits = manifest, limits
        owner = _restore_windowed_device_c64_runner(ticket)
        if (_fingerprint(_book(owner), limits, authority.cp) != _fingerprint(current["bookkeeping"], limits, authority.cp)
                or _fingerprint(_routing(owner, limits), limits, authority.cp) != _fingerprint(current["routing"], limits, authority.cp)):
            _fail("restore_identity", "complete saved book/routing differs after checked construction")
        artifacts.check()
        return _Prepared(_KEY, owner, dict(accepted=_summary(manifest), source_identity=expected_source_identity,
            native_version=version.manifest_digest, known_subset_only=True, claim=CLAIM))
    finally:
        artifacts.close()


def _complete_restore(ticket, prepared, handle, state=None):
    """All mandated queries FIRST; counts FIRST then HOST transfers; refs LAST."""
    _checked_ticket(ticket)
    owner = _owner(prepared)
    runner_admission(owner, ticket.limits)
    owner.report()  # Includes the actual evidence queries; never hides traffic.
    handle._authority.check(handle._header["runtime_reference"])
    pending = ticket.pending_refs
    if type(pending) is not tuple or len(pending) != 4 or pending[0] is not owner._bed:
        _fail("restore_publication", "one complete prepared reference set required")
    bed, context, vegetation, threshold = pending
    if (context.runtime_bounds is not handle._authority.budget
            or owner._water.vegetation_cover_fraction is not vegetation.raster.cover_fraction
            or bed._runner_consumed is not True):
        _fail("restore_publication", "fresh budget/cover/permanent constructor latch")
    # Plain copies: do not call evidence(), verify(), label or packet queries
    # after the bed snapshot and before the genuine pure-HOST transfer snapshot.
    bed_after = dict(bed._counts)
    native_after = _transfer_snapshot()
    bed_before, native_before = ticket.before["bed_after_constructor"], ticket.before["native"]
    if (set(bed_after) != {"d2d_enqueued_arrays", "d2d_enqueued_bytes", "clone_attempts", "packet_attempts", "packets_completed"}
            or set(native_before) != set(native_after)
            or any(type(v) is not int or v < 0 for row in (bed_before, bed_after, native_before, native_after) for v in row.values())
            or any(bed_after[k] < bed_before[k] for k in bed_after)
            or any(native_after[k] < native_before[k] for k in native_after)):
        _fail("restore_diagnostics", "actual monotone complete diagnostic observations required")
    diagnostics = dict(schema=STORE_SCHEMA + ".restore_diagnostics", phase="after_all_restore_queries_before_reference_publication",
        bed=dict(before_phase="after_genuine_bed_constructor", before=bed_before, after=bed_after,
                 delta={k: bed_after[k]-bed_before[k] for k in bed_after}, constructor_observed=bed_before),
        native=dict(before_phase="before_native_HOST_snapshot_load_and_device_construction", before=native_before,
                    after=native_after, delta={k: native_after[k]-native_before[k] for k in native_after}),
        saved_water_counters=copy.deepcopy(owner._water_counters), no_counter_reset=True)
    prepared.evidence["restore_diagnostics"] = copy.deepcopy(diagnostics)
    next_state = None if state is None else dataclasses.replace(state, diagnostics=prepared.diagnostics())
    handle._restore_diagnostics = copy.deepcopy(diagnostics)
    ticket.diagnostics = diagnostics
    # Same-valued reference publication only; no subsequent constructor/query.
    bed._context, bed._vegetation, bed._threshold = context, vegetation, threshold
    ticket.phase = "consumed"
    return next_state


class _C64GPUStore(store.DenseBoundaryStore):
    __slots__ = ("_authority", "_disk_context", "_deep_store_path", "_restore_diagnostics")

    def _admit(self, accepted_names, max_work_bytes, *, initial=False):
        self._authority.prelude()
        super()._admit(accepted_names, max_work_bytes, initial=initial)
        _disk_admit(self._disk_context, self._deep_store_path,
                    disk_bytes=self._reservation_record["disk_bytes"], entries=self._reservation_record["entries"])

    def _stable(self):
        self._binding()
        self._authority.prelude()
        inventory = store._inventory(self._root, self._limits, expected_root=os.fstat(self._rootfd))
        raw = store._read(self._root / "STORE.json", 128 << 10,
                          links=store._published_links(self._root, "STORE.json"))
        if raw != self._store_raw:
            _fail("identity_drift", "immutable STORE changed")
        self._temporary_heads(inventory); self._storage()
        source, environment = self._authority.check(self._header["runtime_reference"])
        if _json(source) != _json(self._header["source_identity"]) or environment != self._header["environment_sha256"]:
            _fail("runtime_identity", "full original profile/Source/environment")

    def _load(self, head):
        scope = _ACTIVE_SCOPE
        if type(scope) is _Scope and scope.ticket is not None:
            return self._load_checked(head, _checked_ticket(scope.ticket))
        ticket = _RestoreTicket(_KEY)
        with _attempt(ticket, self._authority):
            prepared, chain = self._load_checked(head, ticket)
            _complete_restore(ticket, prepared, self)
            return prepared, chain

    def _load_checked(self, head, ticket):
        _checked_ticket(ticket)
        self._authority.prelude()
        chain = store._generation_chain(self._root, self._header, store._sha(self._store_raw), head, self._limits,
            authority=self._authority, ticket=ticket)
        _reservation(self._limits, 0)
        image = chain[0][1]["image"]
        prepared = _prepare(self._root / image["relative_path"], artifact_root=self._root,
            expected_manifest_sha256=image["manifest_sha256"], expected_source_identity=self._header["source_identity"],
            expected_environment_sha256=self._header["environment_sha256"], limits=self._limits.image,
            authority=self._authority, ticket=ticket)
        owner = _owner(prepared)
        self._disk_context = owner._bed.context
        self._deep_store_path = owner._bed._accepted.state.voxel_column.deep_store.store_dir
        _disk_admit(self._disk_context, self._deep_store_path, disk_bytes=0, entries=0)
        if _json(prepared.diagnostics()["accepted"]) != _json(chain[0][1]["accepted_summary"]):
            _fail("readback_identity", "selected accepted summary")
        for selected, generation, _manifest in chain:
            parent = generation["parent_head_record"]
            index = 0 if selected["sequence"] == 0 else selected["sequence"]-1
            if generation["operation_id"] != store._operation(self._header["store_id"], parent, index, owner._schedule.digest):
                _fail("operation_identity", "original schedule/parent/cursor")
        return prepared, chain

    def _extra_original(self, manifest, wrapper_raw):
        ref = self._header["originals"]
        rec, _payload, _raw = _read_wire(self._root / ref["relative_path"], expected_file_sha=ref["manifest_file_sha256"],
            expected_seal=ref["identity"], schema=ORIGINAL_SCHEMA, limits=self._limits.image, expand=False)
        if manifest["originals"] != ref:
            _fail("original_identity", "immutable original reference")
        return rec

    def _save_physical(self, path, owner, pack):
        if pack["schema"] != ORIGINAL_SCHEMA:
            _fail("original_identity", "actual original capsule")
        return _physical_write(path, owner, self)

    def _save_image(self, directory, owner, physical):
        return _image_write(directory, owner, physical, self)

    def step(self, expected_parent):
        with _writer_scope(self._authority):
            return super().step(expected_parent)

    def report(self):
        value = super().report()
        value["scope"] = "C64 same-backend factory GPU accepted-boundary profile; no active-event/CaseBudget resume"
        value["qualification"] = dict(known_subset_only=True, claim=CLAIM, allocation_or_fit_proof=None)
        value["restore_diagnostics"] = copy.deepcopy(self._restore_diagnostics)
        return value


_DEFAULT = C64BoundaryStoreLimits()


def create_windowed_device_c64_boundary_store(path, *, runner, qualification_budget,
        bridge_limits, native_source_refs, limits=_DEFAULT, storage_contract):
    """Factory-only accepted genesis, with a fresh genuine role authority."""
    path = Path(_path(path))  # Exact plain syntax is the only pre-ticket work.
    ticket = _RestoreTicket(_KEY)
    handle = None
    try:
        with _attempt(ticket):
            C64BoundaryStoreLimits.__post_init__(limits)
            store.StorageContract.__post_init__(storage_contract)
            authority = _Authority(qualification_budget, bridge_limits, native_source_refs, limits)
            runner_admission(runner, limits.image)
            if (runner._next != 0 or runner._n_accepted != 0 or runner._receipts
                    or runner._bed.context.runtime_bounds is not qualification_budget
                    or dataclasses.asdict(runner._bed.context.bridge_limits) != dataclasses.asdict(bridge_limits)):
                _fail("fresh_runner", "untouched genuine runner and its exact fresh role budget/limits required")
            if not path.is_relative_to(Path(bridge_limits.run_dir)):
                _fail("run_scope", "store remains inside the genuine governed run directory")
            initial = _reservation(limits, runner._limits.max_work_bytes)
            _disk_admit(runner._bed.context, runner._bed._accepted.state.voxel_column.deep_store.store_dir,
                disk_bytes=initial["disk_bytes"]+(72 << 20), entries=initial["entries"])
            handle = _C64GPUStore._open(path, limits, create=True)
            handle._authority, handle._restore_diagnostics = authority, None
            handle._disk_context = runner._bed.context
            handle._deep_store_path = runner._bed._accepted.state.voxel_column.deep_store.store_dir
            probe, probe_sha = store._probe(handle._root, handle._rootfd)
            os.mkdir(handle._root / "originals", 0o700); os.mkdir(handle._root / "generations", 0o700)
            handle._header = dict(schema=STORE_SCHEMA, store_id=uuid.uuid4().hex, profile=PROFILE,
                source_identity=copy.deepcopy(authority.record["source_identity"]), runtime_reference=copy.deepcopy(authority.record),
                environment_sha256=authority.record["environment_sha256"],
                storage_contract=dict(schema=STORE_SCHEMA+".storage", operator_declares_local=True,
                    operator_declares_fsync_durable=True, observed_fs_type=probe["observation"]["fs_type"],
                    observed_fs_id=probe["f_fsid"], root_path=str(handle._root), root_dev=probe["root_dev"],
                    root_ino=probe["root_ino"], probe_file_sha256=probe_sha),
                limits=store._limits_record(limits), originals=None, genesis_physical=None)
            handle._admit([], runner._limits.max_work_bytes, initial=True)
            original_path = handle._root / "originals" / ("o-"+uuid.uuid4().hex)
            original = _write_wire(original_path,
                dict(inputs=_inputs(runner), initial_pool=runner._device_resources._accepted.pool,
                    initial_soil=runner._device_resources._accepted.soil, inventory0=runner._inventory0),
                schema=ORIGINAL_SCHEMA, limits=limits.image, authority=authority,
                extra=dict(profile=PROFILE, source_identity=handle._header["source_identity"],
                    environment_sha256=handle._header["environment_sha256"]))
            handle._header["originals"] = dict(format=ORIGINAL_SCHEMA, relative_path=str(original_path.relative_to(handle._root)),
                manifest_file_sha256=_file_row(original_path/"manifest.json", limits.image.max_metadata_bytes)["sha256"], identity=original["manifest_sha256"])
            directory, intent, halt = handle._reserve(0, None, runner._schedule.digest)
            physical = _physical_write(directory/"physical", runner, handle)
            handle._header["genesis_physical"] = physical
            handle._header = store._seal(handle._header, "store_sha256")
            store._sealed_write(handle._root, "STORE.json", handle._header, 128 << 10)
            handle._store_raw = oi._json(handle._header)
            image = _image_write(directory/"image", runner, physical, handle)
            head = handle._seal_generation(directory, intent, None, image)
            prepared, chain = handle._load(head)
            _compare(runner, _owner(prepared), limits.image)
            state = handle._idle_state(head, prepared, chain[0][1])
            state = _complete_restore(ticket, prepared, handle, state)
        with _writer_scope(authority):
            handle._publish(head, state, None, halt)
        return handle
    except BaseException as primary:
        if handle is not None:
            try:
                handle.close()
            except BaseException:
                try:
                    primary.add_note("C64 handle cleanup was unconfirmed")
                except BaseException:
                    pass
        raise


def recover_windowed_device_c64_boundary_store(path, *, expected_store_file_sha256,
        qualification_budget, bridge_limits, native_source_refs, limits=_DEFAULT):
    """Selected immutable HEAD, fresh same-backend/UUID role, no replay."""
    path = Path(_path(path))
    _digest(expected_store_file_sha256, "external STORE hash")
    ticket = _RestoreTicket(_KEY)
    with _attempt(ticket):
        authority = _Authority(qualification_budget, bridge_limits, native_source_refs, limits)
        handle = _C64GPUStore._open(path, limits, create=False)
        handle._authority, handle._restore_diagnostics = authority, None
        try:
            store._inventory(handle._root, limits, expected_root=os.fstat(handle._rootfd))
            header, raw = store._record(handle._root/"STORE.json", 128 << 10,
                links=store._published_links(handle._root, "STORE.json"))
            store._check_seal(header, store._STORE, "store_sha256", STORE_SCHEMA)
            if (type(header["store_id"]) is not str or store._NONCE.fullmatch(header["store_id"]) is None
                    or store._sha(raw) != expected_store_file_sha256 or header["profile"] != PROFILE
                    or store._limits_record(store._limits_load_for(header, header["limits"])) != store._limits_record(limits)):
                _fail("store_identity", "exact store/profile/limits required")
            authority.check(header["runtime_reference"])
            handle._header, handle._store_raw = header, raw
            handle._stable(); head, head_raw = handle._head()
            prepared, chain = handle._load(head)
            state = handle._idle_state(head, prepared, chain[0][1])
            store._sync_tree(handle._root); handle._stable()
            if handle._head()[1] != head_raw:
                _fail("parent_drift", "HEAD changed during recovery")
            os.fsync(handle._rootfd)
            state = _complete_restore(ticket, prepared, handle, state)
            handle._state = state
            return handle
        except BaseException as primary:
            try:
                handle.close()
            except BaseException:
                try:
                    primary.add_note("C64 handle cleanup was unconfirmed")
                except BaseException:
                    pass
            raise


def _check_module_image():
    """Current in-process locked owner image; no IO and no foreign methods."""
    from importlib.machinery import ModuleSpec, SourceFileLoader
    module = sys.modules.get(__name__)
    ns = globals()
    if type(module) is not ModuleType or module.__dict__ is not ns:
        _fail("source", "actual current C64 module identity required")
    spec, loader = ns.get("__spec__"), ns.get("__loader__")
    origin = str(Path(codec.__file__).with_name("windowed_device_c64_checkpoint.py"))
    if (type(spec) is not ModuleSpec or type(spec.origin) is not str or spec.origin != origin
            or type(loader) is not SourceFileLoader or type(loader.path) is not str or loader.path != origin
            or ns.get("__file__") != origin or ns.get("_SHARED_STORE") is not store):
        _fail("source", "current C64 file/spec/loader/shared owner differ")
    for name, original in _CLASS_BINDINGS:
        if dict.get(ns, name) is not original:
            _fail("source", "actual C64 class binding changed")
    for owner, name, descriptor, code in _SOURCE_BINDINGS:
        current = owner.get(name)
        function = (current.__func__ if type(current) in (classmethod, staticmethod)
                    else current.fget if type(current) is property else current)
        if current is not descriptor or type(function) is not FunctionType or function.__code__ is not code:
            _fail("source", "actual C64 owning function/code changed")
    for name, original in _DATA_BINDINGS:
        if dict.get(ns, name) is not original:
            _fail("source", "closed C64 constants/roster/defaults changed")


# Captured once by the genuine locked Source import, never reconstructed from
# a file tag or a caller-provided class name. No Native import occurs here.
from .windowed_device_c64_bed import _capture_source_bindings
_CLASS_BINDINGS = tuple((name, value) for name, value in tuple(globals().items())
    if type(value) is type and value.__module__ == __name__)
_DATA_BINDINGS = tuple((name, globals()[name]) for name in (
    "SCHEMA", "PROFILE", "STORE_SCHEMA", "ORIGINAL_SCHEMA", "PHYSICAL_SCHEMA", "RUNTIME_SCHEMA",
    "CLAIM", "E1_MODULE", "PREFIX_MODULE", "_KEY", "_DTYPES", "_BED", "_COUNTERS",
    "_INPUT_FIELDS", "_CURRENT_FIELDS", "_ORIGINAL_FIELDS", "_RUNNER_FIELDS", "_C64_TYPES", "_DEFAULT"))
_SOURCE_BINDINGS = _capture_source_bindings(globals())
