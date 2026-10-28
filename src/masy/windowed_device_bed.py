"""Small, explicitly qualified device-windowed bed ownership.

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
import threading
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


def _integer(value: Any, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise DeviceBedError("invalid_limit", f"{name}: positive plain int <= {maximum} required")
    return value


@dataclass(frozen=True)
class DeviceWindowedBedLimits:
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


class DeviceWindowedBedOwner(DeviceDenseBedOwner):
    """Exact seven-field receiver; inherited ticket/swap code is layout neutral.

    Every Dense schema/copy/admission/validation receiver is replaced here.
    Only the existing same-device ticket, compare, publication and compact-read
    mechanics are reused.  Public construction cannot inject a fake backend.
    """

    def __init_subclass__(cls, **kwargs):
        raise TypeError("device-windowed owner subclasses are unsupported")

    @classmethod
    def from_device_bed(cls, state, context, *, case_record, time_s, limits,
                        vegetation, threshold_modifier_state, trajectory_semantics):
        if cls is not DeviceWindowedBedOwner or type(limits) is not DeviceWindowedBedLimits:
            raise DeviceBedError("invalid_input", "exact device-windowed owner/limits required")
        limits.__post_init__()
        from .dense_device_runner import _genuine_cupy
        cp = _genuine_cupy()
        exchange, shared = _native()
        if type(state) is not shared.WindowedSharedBedState or type(context) is not shared.WindowedDeviceWaterContext:
            raise DeviceBedError("invalid_input", "genuine seven-field device water state/context required")
        if type(context.runtime_bounds) is not shared.DeviceWaterQualificationBudget:
            raise DeviceBedError("invalid_input", "distinct qualification budget required")
        shared._qualification_runtime(context, verify_files=True)
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
        owner._thread, owner._pid = threading.get_ident(), os.getpid()
        owner._device, owner._stream = owner._ops.current_device(), owner._ops.current_stream()
        owner._counts = dict(d2d_enqueued_arrays=0, d2d_enqueued_bytes=0,
                             clone_attempts=0, packet_attempts=0, packets_completed=0)
        owner._native_before = owner._ops.counters()
        owner._plans = []
        owner._nc, owner._nv = 1, 6
        owner._admit(state)
        owner._validate(state, t)
        accepted, mirror = owner._clone(state), owner._clone(state)
        owner._disjoint(accepted, state, mirror)
        owner._disjoint(mirror, state)
        owner._compare((state, accepted), (accepted, mirror))
        owner._accepted = _Accepted(accepted, mirror, t, 0)
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
        if dataclasses.asdict(self._limits) != self._limit_record:
            raise DeviceBedError("limits_changed", "accepted resource declaration changed")

    def _plan(self, state):
        self._scope()
        self._shared._profile(state, self._context)
        plan = self._exchange._plan_device_windowed_bed_exchange(
            state.voxel_column, state.active_layer, self._geometry,
            class_ids=tuple(c.class_id for c in self._context.grain_classes.classes),
            limits=self._context.native_solve_limits.bed_limits)
        if plan.shape != (4, 4, 6, 3, 1) or plan.device_id != self._device or plan.carried_records > 320:
            raise DeviceBedError("unsupported_profile", "exact C16/K1/V6/W3/J320 device profile required")
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
        host = 8*journal + 2*plan.new_host_bytes_peak + 2*plan.mapped_bytes_max
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
            constructor_allocation=None, external_runtime=None, claim=CLAIM)
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
        if tuple(dimensions) != (4, 4, 1) or tuple(committed) != (4, 4, 1):
            raise DeviceBedError("invalid_shape", "Native ledger/committed dimensions differ")
        baseline = state.reconciliation_baseline
        if baseline is not None:
            if tuple(validate_ledger_reconciliation_baseline(baseline)) != (4, 4, 1):
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
        return self._shared.WindowedSharedBedState(voxel_column=new, active_layer=active, **auxiliary)

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
            # A genuine zero-operation Native finish creates its derived
            # window cache even when an initial Native column had none.
            # Every later accepted/mirror pair has that cache on BOTH sides;
            # its bytes then participate in the ordinary currentness check.
            cache = "voxel_column.cache"
            if (set(a) ^ set(b)) - {cache}:
                raise DeviceBedError("integrity_failed", "physical field layouts differ")
            if cache in (set(a) ^ set(b)) and hasattr(self, "_accepted"):
                raise DeviceBedError("integrity_failed", "accepted derived-cache ownership changed")
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
        return dict(schema="masy.windowed_device_bed.v1", case_sha256=self._case,
            generation=self.generation, time_s=self.time_s.hex(), device_id=self._device,
            counters=dict(self._counts), known_frontier=dict(self._last_known),
            known_subset_only=True, opaque_terms=self._shared._QUALIFICATION_OPAQUE_TERMS, claim=CLAIM)

    @classmethod
    def _from_restored_boundary(cls, ticket, state, context, *, case_record, time_s,
                                limits, vegetation, threshold_modifier_state,
                                trajectory_semantics, generation):
        """Actual constructors/mirrors first; checked MASY generation second.

        No Native version/commit time/counter or input array is rewritten.
        The nominal ticket is internal loader discipline, not a security proof.
        """
        from .windowed_device_checkpoint import _checked_ticket, _integer
        _checked_ticket(ticket)
        _integer(generation, 4, "saved MASY bed generation")
        if generation != ticket.current["bookkeeping"]["next_index"]:
            raise DeviceBedError("restore_generation", "accepted cursor/generation differs")
        owner = cls.from_device_bed(state, context, case_record=case_record, time_s=time_s,
            limits=limits, vegetation=vegetation, threshold_modifier_state=threshold_modifier_state,
            trajectory_semantics=trajectory_semantics)
        if owner.generation != 0 or owner._active is not None:
            raise DeviceBedError("restore_generation", "fresh idle owning constructor required")
        owner._accepted = dataclasses.replace(owner._accepted, generation=generation)
        owner.verify()
        return owner
