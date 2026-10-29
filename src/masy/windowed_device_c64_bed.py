"""Separate unaccepted C64 device-windowed bed ownership prototype.

Storage, validation and copying are Native operations.  This sibling adds no
transport law.  Its reservations cover named payloads only; Python, CUDA and
compiler allocation terms remain opaque.  Root's whole-attempt controls are
required for physical qualification.  Dense owners and CPU windowed paths are
unchanged and continue to reject this representation.
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import os
import inspect
import threading
import types
from dataclasses import dataclass
from typing import Any

from .dense_bed import _canonical_static, _native_clock
from .dense_device_bed import (
    DeviceBedError, DeviceDenseBedOwner, _Accepted, _CupyOps,
)

WINDOWED_FIELDS = (
    "voxel_column", "active_layer", "water", "ledger", "committed_topography",
    "sediment_availability", "reconciliation_baseline",
)
_COLUMN_ARRAYS = ("window_kg", "base", "deep_by_class_kg", "deep_total_kg", "deep_room_kg")
CLAIM = "bounded physical qualification; known subset only; no allocation proof"
# Exact immutable1047/unchanged1038 Source facts only. These are UNACCEPTED bytes;
# no runtime, provider, ABI, grant, actual image or fit fact is implied.
def _expected_native_source_pins():
    return (
        ("maple.aeolian.scheduler.windowed_device", 158912,
         "8f1c837609cfa370f2f227c98bae4f670d82c88b5e6eb59ed2315024fe16b52f"),
        ("maple.aeolian.scheduler.windowed_record", 89546,
         "18501ffaec8a7260e6bc6e2bf9431b5209d8fbfbf9f8751607a9ac9b5f57fe10"),
    )


def _expected_scale_helpers():
    return (
        ("_k1_cs_subject", ("owner",)),
        ("_k1_cs_check_source", ("subject", "images", "work_policy")),
        ("_k1_cs_unit", ("name", "calls", "iterations")),
        ("_k1_cs_work", ("dimensions", "row_count", "fields", "reasons", "blocks", "packet_terms", "binding_bytes", "accounting_bytes")),
        ("_k1_cs_work_ceiling", ("subject", "before")),
        ("_k1_cs_transfer_terms", ("subject", "before", "accounting")),
        ("_k1_cs_admit", ("subject",)),
        ("_completed_k1_dimension_record_reservoir_arithmetic_scale_kg", ("owner",)),
    )


_NATIVE_K1_SOURCE_PINS = _expected_native_source_pins()
_K1_SCALE_HELPERS = _expected_scale_helpers()


def _check_data_policy():
    if (_NATIVE_K1_SOURCE_PINS is not _expected_native_source_pins()
            or _K1_SCALE_HELPERS is not _expected_scale_helpers()
            or type(WINDOWED_FIELDS) is not tuple or WINDOWED_FIELDS != (
                "voxel_column", "active_layer", "water", "ledger", "committed_topography",
                "sediment_availability", "reconciliation_baseline")
            or type(_COLUMN_ARRAYS) is not tuple or _COLUMN_ARRAYS != (
                "window_kg", "base", "deep_by_class_kg", "deep_total_kg", "deep_room_kg")
            or type(CLAIM) is not str
            or CLAIM != "bounded physical qualification; known subset only; no allocation proof"):
        raise DeviceBedError("native_scale_source", "closed own Source pins/helper/data policy changed")


def _require_completed_scale_api(*, preflight):
    """Real Native dispatch/images before MASY copies; no manufactured owner.

    The no-owner image check is the genuine Native preflight. During Native
    work only the captured callable identities are rechecked here; Native's
    actual owner/record paths retain their own complete image checks.
    """
    if type(preflight) is not bool:
        raise DeviceBedError("native_scale_source", "plain preflight flag required")
    _check_data_policy()
    pins = _NATIVE_K1_SOURCE_PINS
    if type(pins) is not tuple or len(pins) != 2:
        raise DeviceBedError("native_scale_source", "actual frozen1047 Source join is unissued")
    from maple.aeolian.scheduler import windowed_device as device, windowed_record as record
    namespaces = (device, record)
    for module, row in zip(namespaces, pins, strict=True):
        if (type(module) is not types.ModuleType or type(row) is not tuple or len(row) != 3
                or module.__name__ != row[0] or type(row[1]) is not int or row[1] <= 0
                or type(row[2]) is not str or len(row[2]) != 64):
            raise DeviceBedError("native_scale_source", "closed actual Native Source pins required")
    public = vars(record.WindowedWindRecord).get("reservoir_arithmetic_scale_kg")
    if (type(public) is not types.FunctionType or public.__globals__ is not vars(record)
            or tuple(inspect.signature(public).parameters) != ("self",)
            or not {"_K1DimensionDeviceRecordPlan", "_completed_k1_dimension_record_reservoir_arithmetic_scale_kg"}
                <= set(public.__code__.co_names)):
        raise DeviceBedError("native_scale_api", "genuine zero-argument K1 public dispatch required")
    bindings = [(public, public.__code__)]
    for name, parameters in _K1_SCALE_HELPERS:
        function = vars(device).get(name)
        if (type(function) is not types.FunctionType or function.__module__ != device.__name__
                or function.__globals__ is not vars(device)
                or tuple(inspect.signature(function).parameters) != parameters):
            raise DeviceBedError("native_scale_api", "complete genuine K1 scalar helper family required")
        bindings.append((function, function.__code__))
    domain = getattr(device, "_K1_CS_WORK_DOMAIN", None)
    if (type(domain) is not tuple or any(type(value) is not int for value in domain)
            or domain != (64, 1, 20, 3, 4000, 320, 4, 1, 1, 8, 512)):
        raise DeviceBedError("native_scale_domain", "exact genuine C64 completed-record work domain required")
    checker = vars(device).get("_check_record_image")
    if (type(checker) is not types.FunctionType or checker.__globals__ is not vars(device)
            or tuple(inspect.signature(checker).parameters) != ("plan",)):
        raise DeviceBedError("native_scale_image", "original Native no-owner image checker required")
    bindings.append((checker, checker.__code__))
    if preflight:
        _exchange, shared = _native()
        source = _wrapper_source_frontier()
        if source["host_source_bytes"] > 5 << 30:
            raise DeviceBedError("wrapper_source", "positive Source/read frontier exceeds original HOST ceiling")
        from maple.aeolian.cohorts.capacity import default_available_memory_bytes
        available = default_available_memory_bytes()
        if type(available) is not int or 2*source["host_source_bytes"] > available:
            raise DeviceBedError("wrapper_source", "genuine HOST half-headroom required before Source reads")
        for module, (_name, size, sha256) in zip(namespaces, pins, strict=True):
            source = vars(module).get("__file__")
            if type(source) is not str:
                raise DeviceBedError("native_scale_source", "genuine Native Source origin required")
            try:
                shared._read_reference(shared.SourceReference(source, size, sha256))
            except (shared.BedIntegrationError, OSError) as exc:
                # Preserve the actual immutable-file refusal or OS cause;
                # this does not admit writable checkout Source origins.
                raise DeviceBedError("native_scale_source", "immutable Native Source preflight refused") from exc
        try:
            checker()  # Actual idle Native image/currentness receiver, never a flag.
        except (ValueError, device.closed._ClosedCommitMetadataRefusal) as exc:
            raise DeviceBedError("native_scale_image", "original Native image preflight refused") from exc
    return tuple(bindings)


def _capture_source_bindings(namespace):
    """Finite own function/descriptor forest for the two new wrapper modules."""
    rows = []
    for name, value in tuple(namespace.items()):
        if type(value) is types.FunctionType and value.__module__ == namespace["__name__"]:
            rows.append((namespace, name, value, value.__code__))
        elif type(value) is type and value.__module__ == namespace["__name__"]:
            for member, descriptor in tuple(vars(value).items()):
                function = (descriptor.__func__ if type(descriptor) in (classmethod, staticmethod)
                            else descriptor.fget if type(descriptor) is property else descriptor)
                if type(function) is types.FunctionType:
                    rows.append((vars(value), member, descriptor, function.__code__))
    return tuple(rows)


def _wrapper_source_frontier():
    """Positive logical Source words for these exact two owners, not heap ABI.

    Keep three occurrence banks and both real common-wind namespace copies.
    Constructor/allocator/error-runtime terms remain opaque independently.
    """
    from . import windowed_device_c64_runner as runner
    codes = slots = text = bindings = 0
    def visit(code, depth=0):
        nonlocal codes, slots, text
        if depth > 24 or codes >= 8192:
            raise DeviceBedError("wrapper_source", "finite own-code frontier exceeded")
        codes += 1
        slots += 1 + code.co_nlocals + code.co_stacksize + len(code.co_cellvars) + len(code.co_freevars)
        text += len(code.co_code) + len(code.co_linetable) + len(code.co_exceptiontable)
        text += sum(len(x) for x in code.co_consts if type(x) is str)
        for child in code.co_consts:
            if type(child) is types.CodeType:
                visit(child, depth+1)
    for rows in (_SOURCE_BINDINGS, runner._SOURCE_BINDINGS):
        for namespace, name, original, code in rows:
            descriptor = namespace.get(name)
            function = (descriptor.__func__ if type(descriptor) in (classmethod, staticmethod)
                        else descriptor.fget if type(descriptor) is property else descriptor)
            if descriptor is not original or getattr(function, "__code__", None) is not code:
                raise DeviceBedError("wrapper_source", "actual own binding/code changed")
            bindings += 32 + 4*8 + 4*len(name)
            visit(code)
    _check_data_policy()
    runner._check_data_policy()
    common = vars(runner._cpu_windowed)
    namespaces = 2*(32 + 8*(len(common)+3) + 4*sum(len(name) for name in common))
    if type(_NATIVE_K1_SOURCE_PINS) is not tuple or len(_NATIVE_K1_SOURCE_PINS) != 2:
        raise DeviceBedError("native_scale_source", "Source read operands are unissued, never zero")
    source_raw = 3*sum(row[1] for row in _NATIVE_K1_SOURCE_PINS)
    # visit admits depths0..24, while physical _auxiliary admits0..12.
    # All other own call paths are nonrecursive; each own code/slot row gets
    # the larger25-frame occurrence envelope, including nested comprehensions.
    # This is a positive logical word row, not a CPython frame-size assertion.
    frames = 25*(32*codes + 8*slots)
    host = 3*(32*codes + 8*slots + frames + 4*text + bindings + namespaces) + source_raw
    return dict(host_source_bytes=host, codes=codes, slots=slots, literal_and_code_units=text,
                binding_bytes=bindings, common_namespace_bytes=namespaces,
                frame_word_bytes=frames, frame_occurrences_per_code=25,
                native_source_raw_generations=source_raw, known_subset_only=True,
                constructor_allocation=None, external_runtime=None, claim=CLAIM)


def _integer(value: Any, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise DeviceBedError("invalid_limit", f"{name}: positive plain int <= {maximum} required")
    return value


@dataclass(frozen=True, repr=False, eq=False)
class DeviceWindowedC64BedLimits:
    """Caller-declared ceilings for the named payload projection; no ABI fee."""
    max_bed_device_bytes: int
    max_known_host_bytes: int
    max_known_device_bytes: int
    max_stage_logical_read_bytes: int

    def __post_init__(self):
        for name, maximum in (("max_bed_device_bytes", 2 << 30),
                              ("max_known_host_bytes", 5 << 30),
                              ("max_known_device_bytes", 2 << 30),
                              ("max_stage_logical_read_bytes", 16 << 30)):
            _integer(getattr(self, name), name, maximum)
        if set(vars(self)) != {f.name for f in dataclasses.fields(type(self))}:
            raise DeviceBedError("invalid_limit", "closed device-windowed limits required")

    def as_record(self):
        return {**dataclasses.asdict(self), "known_subset_only": True, "claim": CLAIM}


def _native():
    from maple.surface.active_layer import bed_exchange_device as exchange
    from maple_syrup import windowed_shared_bed_device as shared
    return exchange, shared


def _auxiliary(value, path, array_type, leaves, *, clone=None, depth=0):
    """Closed physical-holder walk after genuine Native pair/profile validation.

    The opaque column, its journal/cache/packet bindings and immutable version
    never enter this walker.  HOST physical arrays are refused, not converted.
    """
    if depth > 12 or len(leaves) > 512 or len(path) > 2048:
        raise DeviceBedError("holder_frontier", "bounded auxiliary holder frontier exceeded")
    if type(value) is array_type:
        if len(leaves) >= 512 or value.ndim > 4 or not value.flags.c_contiguous or value.dtype.hasobject:
            raise DeviceBedError("device_layout", f"{path}: exact C device array required")
        leaves.append((path, value))
        return clone(value) if clone is not None else ("array", tuple(value.shape), value.dtype.str)
    if value is None or type(value) in (bool, int, float, str):
        if (type(value) is int and value.bit_length() > 128
                or type(value) is str and len(value) > 4096
                or type(value) is float and not math.isfinite(value)):
            raise DeviceBedError("holder_scalar", f"{path}: bounded finite metadata required")
        return value
    if isinstance(value, enum.Enum):
        return value if clone is not None else (type(value).__module__, type(value).__qualname__, value.name)
    import numpy as np
    if type(value) in (np.int64, np.int32, np.float64, np.bool_):
        item = value.item()
        if type(item) is float and not math.isfinite(item):
            raise DeviceBedError("holder_scalar", f"{path}: nonfinite Native scalar")
        return value if clone is not None else (type(value).__name__, item)
    if type(value) is tuple and len(value) <= 128:
        return tuple(_auxiliary(item, path + f"[{i}]", array_type, leaves,
                     clone=clone, depth=depth+1) for i, item in enumerate(value))
    kind = type(value)
    from .dense_bed import _native as native_symbols
    from maple.core.types.ledger_reconciliation import LedgerReconciliationBaseline
    n = native_symbols()
    allowed = (n.ActiveLayerState, n.WaterState, n.SedimentLedgerState,
               n.CommittedTopographyState, n.SedimentAvailabilityState,
               LedgerReconciliationBaseline)
    if kind not in allowed:
        raise DeviceBedError("physical_holder", f"{path}: unsupported physical holder {kind.__qualname__}")
    fields = dataclasses.fields(kind)
    if len(fields) > 128 or any(not field.init for field in fields):
        raise DeviceBedError("physical_holder", f"{path}: unqualified auxiliary schema")
    if set(vars(value)) != {field.name for field in fields}:
        raise DeviceBedError("physical_holder", f"{path}: undeclared physical attribute")
    mapped = {field.name: _auxiliary(getattr(value, field.name), path + "." + field.name,
              array_type, leaves, clone=clone, depth=depth+1) for field in fields}
    if clone is not None:
        return kind(**mapped)
    return (kind.__module__, kind.__qualname__, mapped)


class DeviceWindowedC64BedOwner(DeviceDenseBedOwner):
    """Exact seven-field receiver; inherited ticket/swap code is layout neutral.

    Every Dense schema/copy/admission/validation receiver is replaced here.
    Only the existing same-device ticket, compare, publication and compact-read
    mechanics are reused.  Public construction cannot inject a fake backend.
    """

    def __init_subclass__(cls, **kwargs):
        raise TypeError("device-windowed owner subclasses are unsupported")

    @classmethod
    def from_device_bed(cls, state, context, *, case_record, time_s, limits,
                        vegetation, threshold_modifier_state, trajectory_semantics, _case_admission=None):
        if cls is not DeviceWindowedC64BedOwner or type(limits) is not DeviceWindowedC64BedLimits:
            raise DeviceBedError("invalid_input", "exact device-windowed owner/limits required")
        limits.__post_init__()
        support = _require_completed_scale_api(preflight=True)
        from .dense_device_runner import _genuine_cupy
        cp = _genuine_cupy()
        exchange, shared = _native()
        if type(state) is not shared.WindowedSharedBedState or type(context) is not shared.WindowedDeviceWaterC64Context:
            raise DeviceBedError("invalid_input", "genuine seven-field device water state/context required")
        if (type(context.runtime_bounds) is not shared.DeviceWaterC64QualificationBudget
                and type(context.runtime_bounds) is not shared.DeviceWaterC64CaseBudget):
            raise DeviceBedError("invalid_input", "distinct qualification budget required")
        if type(context.runtime_bounds) is not shared.DeviceWaterC64QualificationBudget:
            from .case_device_c64 import _check_case_admission
            _check_case_admission(_case_admission, state, context, context.runtime_bounds, case_record)
        shared._c64_qualification_runtime(context, verify_files=True)
        shared._profile(state, context)
        if tuple(f.name for f in dataclasses.fields(type(state))) != WINDOWED_FIELDS:
            raise DeviceBedError("native_schema_changed", "seven-field shared schema changed")
        t = _native_clock(time_s, "time_s")
        if type(case_record) is not dict:
            raise DeviceBedError("invalid_case", "explicit controlling case metadata required")
        text = json.dumps(case_record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(text.encode()) > 65536:
            raise DeviceBedError("invalid_case", "case metadata exceeds finite profile")
        owner = object.__new__(cls)
        owner._context, owner._exchange, owner._shared = context, exchange, shared
        if type(context.runtime_bounds) is not shared.DeviceWaterC64QualificationBudget:
            owner._case_admission = _case_admission
        owner._native_scale_support = support
        owner._ops = _CupyOps(cp)
        owner._geometry, owner._mres = context.geometry, context.mass_resolution_kg
        owner._geometry_record = _canonical_static(owner._geometry, "geometry")
        owner._limits, owner._limit_record = limits, dataclasses.asdict(limits)
        owner._case_record = json.loads(text)
        owner._case = hashlib.sha256(text.encode()).hexdigest()
        owner._vegetation, owner._threshold = vegetation, threshold_modifier_state
        from maple.aeolian.trajectories.drag import TRAJECTORY_ALGORITHM_VERSION
        if trajectory_semantics != TRAJECTORY_ALGORITHM_VERSION:
            raise DeviceBedError("trajectory_semantics", "genuine current pre-transport marker required")
        owner._semantics = trajectory_semantics
        owner._token, owner._active = object(), None
        # A construction attempt permanently consumes this owner; it never
        # points back to a runner and cannot create an ownership cycle.
        owner._runner_consumed = False
        owner._zero_op_copy_roles = ()
        owner._thread, owner._pid = threading.get_ident(), os.getpid()
        owner._device, owner._stream = owner._ops.current_device(), owner._ops.current_stream()
        owner._counts = dict(d2d_enqueued_arrays=0, d2d_enqueued_bytes=0,
                             clone_attempts=0, packet_attempts=0, packets_completed=0)
        owner._native_before = owner._ops.counters()
        owner._plans = []
        owner._nc, owner._nv = 1, 20
        owner._admit(state)
        owner._validate(state, t)
        try:
            accepted, mirror = owner._clone(state), owner._clone(state)
            owner._disjoint(accepted, state, mirror)
            owner._disjoint(mirror, state)
            owner._compare((state, accepted), (accepted, mirror))
        finally:
            owner._zero_op_copy_roles = None
        owner._accepted = _Accepted(accepted, mirror, t, 0)
        if type(context.runtime_bounds) is not shared.DeviceWaterC64QualificationBudget:
            from .case_device_c64 import _bind_case_admission_owner
            _bind_case_admission_owner(_case_admission, owner)
        return owner

    @classmethod
    def _from_restored_c64_boundary(cls, ticket, state, context, *, case_record, time_s,
                                    limits, vegetation, threshold_modifier_state,
                                    trajectory_semantics, generation, _case_admission=None):
        """Actual Native constructor first; only a MASY generation is rebound."""
        from .windowed_device_c64_checkpoint import _checked_ticket, _integer
        _checked_ticket(ticket)
        if cls is not DeviceWindowedC64BedOwner:
            raise DeviceBedError("restore_kind", "exact C64 owner required")
        _integer(generation, 4, "saved C64 bed generation")
        if generation != ticket.current["bookkeeping"]["next_index"]:
            raise DeviceBedError("restore_generation", "bed/cursor mismatch")
        if type(context.runtime_bounds) is not _native()[1].DeviceWaterC64QualificationBudget:
            shared = _native()[1]
            if (not ticket.authority.case_mode or type(context.runtime_bounds) is not shared.DeviceWaterC64CaseBudget
                    or context.runtime_bounds is not ticket.authority.budget):
                raise DeviceBedError("qualification_budget", "factory C64 only; CaseBudget unsupported")
            from .case_device_c64 import _check_case_admission
            _check_case_admission(_case_admission, state, context, context.runtime_bounds, case_record)
        owner = cls.from_device_bed(state, context, case_record=case_record, time_s=time_s,
            limits=limits, vegetation=vegetation, threshold_modifier_state=threshold_modifier_state,
            trajectory_semantics=trajectory_semantics, _case_admission=_case_admission)
        if owner.generation != 0 or owner._active is not None:
            raise DeviceBedError("restore_generation", "fresh idle Native owner required")
        owner._accepted = dataclasses.replace(owner._accepted, generation=generation)
        owner.verify()
        return owner

    @property
    def context(self):
        self._scope()
        return self._context

    @property
    def case_record(self):
        return json.loads(json.dumps(self._case_record, sort_keys=True, allow_nan=False))

    @property
    def case_sha256(self):
        return self._case

    @property
    def vegetation(self):
        return self._vegetation

    @property
    def threshold_modifier_state(self):
        return self._threshold

    def vegetation_cover_fraction(self):
        return self._vegetation.raster.cover_fraction

    @property
    def limits(self):
        return self._limits

    def verify_case_inputs(self):
        self.verify()

    def _scope(self):
        if (os.getpid() != self._pid or threading.get_ident() != self._thread
                or self._ops.current_device() != self._device
                or self._ops.current_stream() != self._stream):
            raise DeviceBedError("device_scope_changed", "same process/thread/device/stream required")
        if _canonical_static(self._geometry, "geometry") != self._geometry_record:
            raise DeviceBedError("geometry_changed", "controlling Native geometry changed")
        self._limits.__post_init__()
        if _require_completed_scale_api(preflight=False) != self._native_scale_support:
            raise DeviceBedError("native_scale_source", "genuine completed-K1 capability changed")
        if dataclasses.asdict(self._limits) != self._limit_record:
            raise DeviceBedError("limits_changed", "accepted resource declaration changed")

    def _plan(self, state):
        self._scope()
        self._shared._profile(state, self._context)
        plan = self._exchange._plan_device_windowed_bed_exchange(
            state.voxel_column, state.active_layer, self._geometry,
            class_ids=tuple(c.class_id for c in self._context.grain_classes.classes),
            limits=self._context.native_solve_limits.bed_limits)
        if plan.shape != (8, 8, 20, 3, 1) or plan.device_id != self._device or plan.carried_records > 320:
            raise DeviceBedError("unsupported_profile", "exact C64/K1/V20/W3/J320 device profile required")
        return plan

    def _describe(self, state):
        plan = self._plan(state)
        column = state.voxel_column
        leaves = [("voxel_column." + name, getattr(column, name)) for name in _COLUMN_ARRAYS]
        if column.window_cache is not None:
            leaves.append(("voxel_column.cache", column.window_cache.window_totals_kg))
        leaves.extend((("voxel_column.packet", column.height_packet.packet),
                       ("voxel_column.encoded_base", column.height_packet.binding.encoded_base)))
        metadata = {name: _auxiliary(getattr(state, name), name, self._ops.array_type, leaves)
                    for name in WINDOWED_FIELDS[1:]}
        store = column.deep_store
        pending = column.pending_deep
        journal = None if pending is None else pending.journal
        metadata["column"] = (column.n_voxels, column.generation, column.manifest_digest,
            str(store.store_dir), str(store.manifest_path), store.version_id,
            store.format_version, store.encoding,
            None if journal is None else (journal.header_digest,
                tuple((name, getattr(journal, name).dtype.str, tuple(getattr(journal, name).shape),
                       hashlib.sha256(getattr(journal, name).tobytes()).hexdigest())
                      for name in ("cell_flat", "voxel", "values_kg"))))
        for path, value in leaves:
            if type(value) is not self._ops.array_type or self._ops.device(value) != self._device:
                raise DeviceBedError("device_layout", f"{path}: exact same-device array required")
        return leaves, metadata

    def _admit(self, state):
        """Positive current/next/error payload law BEFORE every resident copy.

        Seven full state generations cover input, accepted+mirror, work,
        candidate and next accepted+mirror.  Native's complete session includes
        its actual packet/cache/journal/scratch terms; auxiliary copies and
        full byte-comparison temporaries are additional. No release/alias credit.
        """
        plan = self._plan(state)
        arrays, _metadata = self._describe(state)
        footprint = sum(int(value.nbytes) for _, value in arrays)
        auxiliary = sum(int(value.nbytes) for path, value in arrays
                        if not path.startswith("voxel_column."))
        largest = max(int(value.nbytes) for _, value in arrays)
        pending = state.voxel_column.pending_deep
        journal = 0 if pending is None else sum(int(getattr(pending.journal, name).nbytes)
            for name in ("cell_flat", "voxel", "values_kg"))
        device = 7*footprint + 2*auxiliary + 2*plan.new_device_bytes_peak + 2*largest
        source = _wrapper_source_frontier()
        # At most two directed source/copy pairs exist during one copying
        # operation. Six pointer roles per current/next/error occurrence are
        # positive logical words; tuple/allocator costs remain opaque.
        copy_role_reference_bytes = 3*8*(2*2+2)
        host = (8*journal + 2*plan.new_host_bytes_peak + 2*plan.mapped_bytes_max
                + source["host_source_bytes"] + copy_role_reference_bytes)
        reads = 2*plan.logical_deep_read_bytes
        if (footprint > self._limits.max_bed_device_bytes
                or device > self._limits.max_known_device_bytes
                or host > self._limits.max_known_host_bytes
                or reads > self._limits.max_stage_logical_read_bytes):
            raise DeviceBedError("work_bytes_cap", "positive windowed current/next/error frontier exceeds declaration")
        from maple.aeolian.cohorts.capacity import default_available_memory_bytes
        available = default_available_memory_bytes()
        if type(available) is not int or 2*host > available:
            raise DeviceBedError("host_availability", "genuine HOST half-headroom check refused")
        self._exchange._check_device_admission(device, device_id=self._device, safety_fraction=0.5)
        self._last_known = dict(device_payload_bytes=device, host_payload_bytes=host,
            logical_deep_read_bytes=reads, generations=7, known_subset_only=True,
            copy_role_reference_bytes=copy_role_reference_bytes,
            wrapper_source=source, constructor_allocation=None, external_runtime=None, claim=CLAIM)
        return dict(transfer_bytes=footprint), 2*largest, host+device

    def _validate(self, state, time_s):
        self._plan(state)
        if float(state.committed_topography.last_commit_time_s).hex() != float(time_s).hex():
            raise DeviceBedError("commit_stamp_mismatch", "published native stamp must equal actual boundary time")
        if state.voxel_column.pending_deep is not None:
            raise DeviceBedError("unpublished_boundary", "accepted/event boundary needs actual Native publication")
        self._shared._validate(state, self._context)
        from .dense_bed import _native as native_symbols
        from maple.core.types.ledger_reconciliation import validate_ledger_reconciliation_baseline
        n = native_symbols()
        g = self._geometry
        n.check_sediment_availability_state(state.sediment_availability, state.active_layer, self._mres)
        n.validate_water_state(state.water, g.ny, g.nx, 1)
        dimensions = n.validate_ledger_state(state.ledger)
        committed = n.validate_committed_topography_state(state.committed_topography)
        if tuple(dimensions) != (8, 8, 1) or tuple(committed) != (8, 8, 1):
            raise DeviceBedError("invalid_shape", "Native ledger/committed dimensions differ")
        baseline = state.reconciliation_baseline
        if baseline is not None:
            if tuple(validate_ledger_reconciliation_baseline(baseline)) != (8, 8, 1):
                raise DeviceBedError("invalid_shape", "Native baseline dimensions differ")
            if float(baseline.last_reconciliation_time_s) > float(time_s):
                raise DeviceBedError("invalid_baseline_clock", "baseline cannot follow the physical boundary")

    def _clone(self, state):
        self._admit(state)
        cp_type, leaves = self._ops.array_type, []

        def copied(array):
            self._counts["clone_attempts"] += 1
            result = self._ops.clone(array)
            self._counts["d2d_enqueued_arrays"] += 1
            self._counts["d2d_enqueued_bytes"] += int(array.nbytes)
            return result

        # Native copying begin passes the active layer by reference: supply
        # this separately admitted independent copy before opening it.
        active = _auxiliary(state.active_layer, "active_layer", cp_type, leaves, clone=copied)
        auxiliary = {name: _auxiliary(getattr(state, name), name, cp_type, leaves, clone=copied)
                     for name in WINDOWED_FIELDS[2:]}
        column = state.voxel_column
        with self._exchange._open_device_windowed_bed_exchange(
                column, active, self._geometry, self._mres,
                class_ids=tuple(c.class_id for c in self._context.grain_classes.classes),
                admission=self._context.native_solve_limits.bed_limits) as exchange:
            outcome = exchange.finish()
        new = outcome.new_column
        if (outcome.n_operations != 0 or outcome.new_active_layer is not active
                or new.deep_store is not column.deep_store
                or new.generation != column.generation or new.manifest_digest != column.manifest_digest):
            raise DeviceBedError("native_stage_changed", "zero-operation Native copy changed physical authority")
        if (self._zero_op_copy_roles is not None and column.window_cache is None
                and new.window_cache is not None):
            if len(self._zero_op_copy_roles) >= 2:
                raise DeviceBedError("native_stage_changed", "only two current zero-operation copy roles allowed")
            self._zero_op_copy_roles += ((column, new),)
        return self._shared.WindowedSharedBedState(voxel_column=new, active_layer=active, **auxiliary)

    def begin_stage(self):
        self._zero_op_copy_roles = ()
        try:
            return super().begin_stage()
        finally:
            self._zero_op_copy_roles = None

    def prepare_publish(self, stage, candidate, *, time_s):
        self._zero_op_copy_roles = ()
        try:
            return super().prepare_publish(stage, candidate, time_s=time_s)
        finally:
            self._zero_op_copy_roles = None

    def _disjoint(self, left, *others):
        a, _ = self._describe(left)
        for other in others:
            b, _ = self._describe(other)
            for _path, x in a:
                for _name, y in b:
                    if self._overlap(x, y):
                        raise DeviceBedError("stage_not_independent", "resident device copies overlap")

    def _closed(self, state):
        return self._describe(state)

    def _compare(self, *pairs):
        flags = []
        for left, right in pairs:
            a, sa = self._describe(left)
            b, sb = self._describe(right)
            if sa != sb:
                raise DeviceBedError("integrity_failed", "scalar/version/journal metadata differs")
            a, b = dict(a), dict(b)
            # Only an exact source -> current zero-operation copy may gain
            # the Native-derived cache. Accepted/mirror and unrelated pairs
            # never receive a lifecycle-wide exemption.
            cache = "voxel_column.cache"
            if (set(a) ^ set(b)) - {cache}:
                raise DeviceBedError("integrity_failed", "physical field layouts differ")
            if cache in (set(a) ^ set(b)):
                roles = self._zero_op_copy_roles
                if (cache in a or cache not in b or type(roles) is not tuple
                        or not any(left.voxel_column is source and right.voxel_column is copied
                                   for source, copied in roles)):
                    raise DeviceBedError("integrity_failed", "derived-cache change lacks an exact current copy role")
            for path in sorted(set(a) & set(b)):
                if (a[path].shape != b[path].shape or a[path].dtype != b[path].dtype):
                    raise DeviceBedError("integrity_failed", "physical array layouts differ")
                flags.append(self._ops.raw_equal(a[path], b[path]))
        self._counts["packet_attempts"] += 1
        packet = self._ops.packet(flags)
        self._counts["packets_completed"] += 1
        if packet.shape != (len(flags),) or not all(float(value) == 1.0 for value in packet):
            raise DeviceBedError("integrity_failed", "device bytes differ")

    def _compact(self, function, *, max_values):
        _integer(max_values, "max_values", 16)
        self._scope()
        self.verify()
        value = function(self._accepted.state)
        shape = tuple(getattr(value, "shape", ()))
        if len(shape) > 1 or shape and shape[0] > max_values:
            raise DeviceBedError("spatial_payload", "only bounded compact reductions may leave the device")
        self._counts["packet_attempts"] += 1
        packet = self._ops.vector_packet(value)
        self._counts["packets_completed"] += 1
        return packet

    def evidence(self):
        self.verify()
        return dict(schema="masy.windowed_device_c64_bed.v1", case_sha256=self._case,
            generation=self.generation, time_s=self.time_s.hex(), device_id=self._device,
            counters=dict(self._counts), known_frontier=dict(self._last_known),
            known_subset_only=True, opaque_terms=self._shared._QUALIFICATION_OPAQUE_TERMS, claim=CLAIM)


_SOURCE_BINDINGS = _capture_source_bindings(globals())
