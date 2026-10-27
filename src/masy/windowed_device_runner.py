"""Exact Small GPU windowed W/R/W/R qualification adapter.

Actual Native wind records and actual SYRUP water events share one evolving
seven-field bed.  The original MASY event transaction, clocks and accounting
remain the engine.  The strict proof lane and all old profiles are unchanged.
This interface reports known payloads and opaque costs, never a peak/fit proof.
"""
from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import math
import os
import resource
import threading
import types
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

from . import dense_bed as _bed
from . import dense_device_runner as _device
from . import dense_runner as _dense
from . import windowed_runner as _cpu_windowed
from .clock import ClockLedger, encode_time
from .dense_device_bed import DeviceBedError, _CupyOps
from .dense_device_resources import (
    DeviceDenseResources, DeviceResourceError, DeviceResourceLimits,
    STATIC_NAMES, resource_reservation,
)
from .dense_runner import (
    DenseRunner, DenseRunnerControl, DenseRunnerError, DenseRunnerLimits,
    DenseWaterInputs, DenseWindInputs, _Stop,
)
from .dense_schedule import DenseSchedule, RainEvent, WindEvent, case_schedule_digest
from .windowed_device_bed import CLAIM, DeviceWindowedBedOwner, WINDOWED_FIELDS

SCOPE = "small_C16_K1_V6_W3_GPU_windowed_WRWR_qualification_v1"
REPORT_ROWS = 4096
_TOKEN = object()


@dataclass(frozen=True)
class DeviceWindowedWindProfile:
    """Original resolved HOST model/config provenance plus genuine device limits."""
    model_config: Any
    wind_config: Any
    record_limits: Any
    source_identity: str
    calibration: str
    label: str


def _native():
    from maple.aeolian.scheduler import windowed_device as device
    from maple.aeolian.scheduler.state import SchedulerState, SchedulerPhysicalState, SchedulerPhysicsConfig
    from maple.aeolian.scheduler.accounting import BlockAccounting
    from maple.aeolian.scheduler.record import WindRecordResult
    from maple.aeolian.scheduler.windowed_record import (
        WindowedWindRecord, WindowedWindRecordResult, WindowedWindRecordLimits,
    )
    from maple.surface.voxels.windowed_inventory import resident_reduction_bound_by_cell_class_kg
    from maple_syrup import windowed_shared_bed_device as shared
    from maple_syrup.complete_event import DryHandoff
    from maple_syrup.sediment_bed import BedCommit, BedIntegrationError, reroute_threshold_m
    return SimpleNamespace(**locals())


def _require_entry_api(w):
    expected = {
        "commit_qualification_device_entry": ("state", "context", "graph", "reference", "current_time_s",
            "routing_source_m", "control"),
        "complete_qualification_device_event": ("state", "context", "column", "field", "schedule",
            "vegetation", "sediment", "max_end_s", "control", "policy", "report_every_s"),
    }
    for name, names in expected.items():
        function = getattr(w.shared, name, None)
        if (type(function) is not types.FunctionType or function.__module__ != w.shared.__name__
                or function.__globals__ is not vars(w.shared)
                or tuple(inspect.signature(function).parameters) != names):
            raise DenseRunnerError("unsupported_device_water_api", f"genuine exact {name} required")


def _schedule_profile(schedule):
    if type(schedule) is not DenseSchedule or len(schedule.events) != 4:
        raise DenseRunnerError("device_schedule", "exact four-event sequential schedule required")
    if tuple(type(event) for event in schedule.events) != (WindEvent, RainEvent, WindEvent, RainEvent):
        raise DenseRunnerError("device_schedule", "exact wind/rain/wind/rain order required")
    if any(event.duration_s != 1.0 for event in schedule.events if type(event) is WindEvent):
        raise DenseRunnerError("device_schedule", "each declared wind event is one actual one-second block")


def _process_check(context, identity=None, *, upcoming_wrapper_bytes=0, checkpoint="event_entry"):
    """Check genuine inherited OS/pool controls; Root owns whole-tree monitoring.

    This does not convert observed RSS/driver/storage into an allocation proof,
    start a new timeout, or install/modify any limit or allocator.
    """
    w = _native()
    if (type(context) is not w.shared.WindowedDeviceWaterContext
            or type(context.runtime_bounds) is not w.shared.DeviceWaterQualificationBudget):
        raise DenseRunnerError("qualification_context", "distinct genuine qualification context required")
    w.shared._qualification_runtime(context, verify_files=True)
    if type(upcoming_wrapper_bytes) is not int or not 0 <= upcoming_wrapper_bytes <= 5 << 30:
        raise DenseRunnerError("wrapper_payload_cap", "known positive wrapper payload exceeds fixed ceiling")
    cp = _device._genuine_cupy()
    current = dict(pid=os.getpid(), thread=threading.get_ident(), device=int(cp.cuda.runtime.getDevice()))
    if identity is not None and current != identity:
        raise DenseRunnerError("process_identity_changed", "same process/thread/device required")
    for kind, required in ((resource.RLIMIT_CPU, (480, 540)),
                           (resource.RLIMIT_FSIZE, (64 << 20, 64 << 20)),
                           (resource.RLIMIT_CORE, (0, 0))):
        if resource.getrlimit(kind) != required:
            raise DenseRunnerError("qualification_process_limits", "original exact OS policy required")
    from maple.aeolian.cohorts.memory_budget import DeviceMemorySnapshot, default_device_memory_snapshot
    snapshot = default_device_memory_snapshot()
    if (type(snapshot) is not DeviceMemorySnapshot or snapshot.device_id != current["device"]
            or snapshot.allocator_supported is not True or snapshot.pool_limit_bytes != 2 << 30):
        raise DenseRunnerError("qualification_pool", "original actual 2GiB active pool limit required")
    return dict(process_identity=current, checkpoint=checkpoint,
        hard_controls="CPU480/540;file64MiB;core0;active_pool2GiB",
        external_Root_controls="whole600wall;sampledRSS5GiB;drivergrowth9GiB;run512MiB;streams4MiB",
        known_subset_only=True, opaque_terms=w.shared._QUALIFICATION_OPAQUE_TERMS, claim=CLAIM)


class _Evidence:
    __slots__ = ()
    SCOPE = SCOPE
    DenseProductionError = DenseRunnerError
    check_process = staticmethod(_process_check)
    commit_evidence = staticmethod(_cpu_windowed._windowed_commit_evidence)


_EVIDENCE = _Evidence()


def _resident_fold_bound(n, w, column, geometry):
    """Unchanged Native fold/policy with device scalar/K-vector receivers."""
    values = w.resident_reduction_bound_by_cell_class_kg(column, geometry)
    ops = _device._DeviceArrayOps(_device._genuine_cupy(), 1)
    folded = ops.class_vector(values.sum(axis=(0, 1)))
    rounding = n.summation_error_bound_kg(int(geometry.ny * geometry.nx), ops.scalar(values.max()))
    return folded + rounding


def _wind_binding(function):
    """Bind only the existing wrapper orchestration, never a Native physics body."""
    allowed = (_cpu_windowed.WindowedRunner._wind_event, _cpu_windowed.WindowedRunner._wind_accept_record)
    if function not in allowed or function.__globals__ is not vars(_cpu_windowed):
        raise DenseRunnerError("source_binding_changed", "only the two genuine wrapper wind bodies may bind")
    namespace = dict(vars(_cpu_windowed))
    namespace.update(_windowed_native=_native,
        _prepare_record=lambda *args, **kwargs: _native().device.prepare_device_windowed_small_feature_wind_record(
            *args, **kwargs), _resident_fold_bound=_resident_fold_bound)
    bound = types.FunctionType(function.__code__, namespace, function.__name__,
        function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    return bound


@dataclass(frozen=True)
class _Stage:
    bed: Any
    resources: Any
    generation: int
    time_s: float
    nbytes: int
    pending_records_in: int
    pending_records_out: int


class _Ownership(_device._DeviceOwnership):
    """Existing paired device transaction mechanics with seven-field receivers."""
    receipt_schema = "masy.windowed_device_event_receipt.v1"

    def __init__(self, bed, resources, ops, stage_bytes):
        super().__init__(bed, resources, ops, 1, stage_bytes, None)

    def verify_case_inputs(self):
        try:
            self._owner.verify()
            self._res.verify_static()
        except (DeviceBedError, DeviceResourceError) as exc:
            raise _bed.DenseBedError(exc.code, exc.detail) from exc

    def begin(self):
        pending = self._owner._accepted.state.voxel_column.pending_deep
        incoming = 0 if pending is None else int(pending.journal.n_records)
        stage = super().begin()
        pending = stage.bed.state.voxel_column.pending_deep
        outgoing = 0 if pending is None else int(pending.journal.n_records)
        self._stage = _Stage(stage.bed, stage.resources, stage.generation, stage.time_s,
                             self._stage_bytes, incoming, outgoing)
        return self._stage

    def scheduler(self, stage, pool, time_s):
        n, w = _dense._native(), _native()
        state = stage.bed.state
        if type(time_s) is not float or time_s.hex() != stage.time_s.hex():
            raise _bed.DenseBedError("time_mismatch", "fresh scheduler keeps actual boundary time")
        if not n.mobile_state_is_genuinely_empty(pool) or self._ops.any_nonzero(state.water.mobile_mass_by_cell_class_kg):
            raise _bed.DenseBedError("mobile_state_not_empty", "no airborne/water-mobile load at wind entry")
        physical = w.SchedulerPhysicalState(
            voxel_column=state.voxel_column, active_layer=state.active_layer, mobile_state=pool,
            ledger=state.ledger, committed_topography=state.committed_topography,
            threshold_modifier_state=self._res.static("threshold_modifier_state"), time_s=time_s,
            vegetation=self._res.static("vegetation"), sediment_availability=state.sediment_availability,
            water=state.water, coupled_trajectory_semantics=self._owner._semantics,
            ledger_reconciliation_baseline=state.reconciliation_baseline)
        return w.SchedulerState(physical=physical, coupled=None, block_index=0, block_offset_s=0.0,
            last_resolved_wind=None, topographic_wind_cache=None, vegetation_wind_cache=None,
            active_layer_composition_cache=None)

    def prepare_scheduler(self, physical):
        n, w = _dense._native(), _native()
        if (type(physical) is not w.SchedulerPhysicalState
                or physical.vegetation is not self._res.static("vegetation")
                or physical.threshold_modifier_state is not self._res.static("threshold_modifier_state")
                or not n.mobile_state_is_genuinely_empty(physical.mobile_state)
                or physical.coupled_trajectory_semantics != self._owner._semantics):
            raise _Stop("refused", "device_wind_handoff", "exact owned prescribed fields and empty Native pool required")
        candidate = w.shared.WindowedSharedBedState(voxel_column=physical.voxel_column,
            active_layer=physical.active_layer, water=physical.water, ledger=physical.ledger,
            committed_topography=physical.committed_topography,
            sediment_availability=physical.sediment_availability,
            reconciliation_baseline=physical.ledger_reconciliation_baseline)
        return self._prepare(candidate, physical.time_s, pool=physical.mobile_state)

    def retain_routing(self, state):
        if state.phase is not None or self._ops.any_nonzero(state.sediment_velocity_m_s):
            raise _Stop("incomplete", "transport_memory_not_empty", "genuine Native dry transport memory required")
        self._routing_generation += 1
        # Retain the actual reset velocity; do not synthesize another zero grid.
        return SimpleNamespace(graph=state.graph, terrain=state.terrain, network=state.network, grid=state.grid,
            source=state.routing_source_elevation_m, velocity=state.sediment_velocity_m_s, phase=state.phase,
            generation=self._routing_generation, stamp=self._routing_stamp)

    def production_require_dry(self, production, np):
        cp = self._ops.xp
        flag = self._owner._compact(lambda state: cp.any(state.water.depth_m != 0.0)
            | cp.any(state.water.mobile_mass_by_cell_class_kg != 0.0), max_values=1)
        if float(flag[0]) != 0.0:
            raise DenseRunnerError("wind_water_exchange", "actual accepted water must be dry")

    def production_wind_evidence(self, production, n, result, final):
        cp = self._ops.xp
        targets = (result.record.state.physical.water, final.state.physical.water)
        def changed(state):
            value = cp.any(state.water.depth_m != 0.0) | cp.any(state.water.mobile_mass_by_cell_class_kg != 0.0)
            for target in targets:
                for name in ("depth_m", "mobile_mass_by_cell_class_kg"):
                    left, right = getattr(state.water, name), getattr(target, name)
                    if (type(right) is not cp.ndarray or right.shape != left.shape
                            or right.dtype != left.dtype or not right.flags.c_contiguous):
                        raise DenseRunnerError("wind_water_layout", "unchanged device water schema required")
                    value = value | cp.any(right != 0.0) | ~self._owner._ops.raw_equal(left, right)
            return value
        if float(self._owner._compact(changed, max_values=1)[0]) != 0.0:
            raise DenseRunnerError("wind_water_exchange", "Native wind changed the dry water")
        a = result.record.aggregate_accounting
        exchanges = {name: None if getattr(a, name) is None else self._ops.scalar(getattr(a, name))
                     for name in ("water_clamp_added_volume_m3", "water_displaced_volume_m3")}
        if any(value not in (None, 0.0) for value in exchanges.values()):
            raise DenseRunnerError("wind_water_exchange", "Native wind reported water exchange")
        if final.state.topographic_wind_cache is not None or final.state.active_layer_composition_cache is not None:
            raise DenseRunnerError("wind_cache_not_invalidated", "Native finalizer cache invalidation required")
        cache = result.record.state.topographic_wind_cache
        modifier = None if cache is None else cache.modifier
        topography = {"status": "absent"} if modifier is None else dict(status="native_modifier",
            algorithm_version=modifier.algorithm_version,
            wind_direction_deg=self._ops.scalar(modifier.wind_direction_deg),
            separation_present=modifier.separation is not None,
            separation_bubble_cells=None if modifier.separation is None else
                int(self._ops.scalar(cp.count_nonzero(modifier.separation.bubble_mask))))
        return dict(scope=SCOPE, dry_water_preserved=True, water_rule="constant_depth",
            water_volume_exchanges_m3=exchanges, topographic_wind=topography,
            transport_launched_mass_by_class_kg=self._ops.float_values(a.launched_mass_by_class_kg),
            latest_wind_commit={"status": "not_retained_by_native_record_owner"},
            seventh_baseline="retained_in_owned_shared_bed", known_subset_only=True, claim=CLAIM)

    def production_retained_owners(self, production, wind, water, control, profile, pool, soil, routing, n_receipts):
        if type(n_receipts) is not int or not 0 <= n_receipts <= 4:
            raise DenseRunnerError("receipt_frontier", "four-event receipt frontier exceeded")
        return dict(scope=SCOPE, bed_owner=self._owner.evidence(), resources=self._res.evidence(),
            private_pool_bytes=self._res.pool_bytes(), private_soil_bytes=self._res.soil_bytes(),
            receipts=n_receipts, routing="absent" if routing is None else self.routing_label(routing),
            known_subset_only=True, opaque_terms=_native().shared._QUALIFICATION_OPAQUE_TERMS, claim=CLAIM)


def _summary(w, summary, *, entry):
    if type(summary) is not w.shared.QualificationAdmissionSummary:
        raise _Stop("incomplete", "qualification_summary", "actual distinct qualification summary required")
    w.shared._holder(summary, w.shared.QualificationAdmissionSummary)
    w.shared._qualification_labels(summary.known_subset_only, summary.opaque_terms, summary.claim)
    if entry and summary.hydraulics is not None:
        raise _Stop("incomplete", "entry_hydraulics", "commit-only entry cannot invent a hydraulic lifecycle")
    names = ("gate_count", "logical_work_debited", "logical_read_bytes_debited", "versions_reserved",
             "h2d_bytes_reserved", "d2h_bytes_reserved", "scalar_reads_reserved",
             "maximum_host_predicate", "maximum_device_predicate")
    values = {name: getattr(summary, name) for name in names}
    if any(type(value) is not int or not 0 <= value < 1 << 63 for value in values.values()):
        raise _Stop("incomplete", "qualification_summary", "closed nonnegative actual int63 counters required")
    return dict(values, hydraulics=summary.hydraulics, known_subset_only=True,
                opaque_terms=summary.opaque_terms, claim=summary.claim)


class _WaterLane:
    def __init__(self, runner, w):
        self.runner, self.w, self.entry, self.event = runner, w, None, None

    def _profile(self, state, context):
        if (type(state) is not self.w.shared.WindowedSharedBedState
                or context is not self.runner._bed.context
                or type(context.runtime_bounds) is not self.w.shared.DeviceWaterQualificationBudget
                or self.runner._limits.max_report_rows != REPORT_ROWS
                or context.bridge_limits.max_report_rows != REPORT_ROWS):
            raise _Stop("refused", "device_water_profile", "exact qualified seven-field/4096 profile required")

    def native(self):
        values = dict(vars(_dense._native()))
        values["commit_bed"] = self.commit_bed
        return SimpleNamespace(**values)

    def commit_bed(self, state, context, graph, reference, time_s, *, force, routing_source_m):
        self._profile(state, context)
        if force is not True or self.entry is not None:
            raise _Stop("refused", "device_entry_commit", "one actual forced entry commit required")
        self.entry = {}  # sticky before invocation, including its original failure path
        result, summary = self.w.shared.commit_qualification_device_entry(state, context, graph, reference,
            time_s, routing_source_m, control=self.runner._control.sediment_control)
        evidence = {"entry_commit": result, "qualification_summary": summary}
        if (type(result) is not self.w.BedCommit or result.did_commit is not True
                or type(result.rerouted) is not bool or result.routing_source_m is None):
            raise _Stop("incomplete", "device_entry_commit", "genuine committed Native routing result required",
                        evidence=evidence)
        try:
            self.entry = _summary(self.w, summary, entry=True)
        except _Stop as stop:
            stop.evidence = evidence
            raise
        except Exception as exc:
            raise _Stop("incomplete", "device_entry_summary", _dense._detail(exc), exception=exc,
                        evidence=evidence) from exc
        return result

    def complete(self, state, context, column, field, schedule, vegetation, sediment,
                 *, max_end_s, control, policy, report_every_s, max_report_rows, **extra):
        self._profile(state.bed, context)
        if (extra or max_report_rows != REPORT_ROWS or self.entry is None or self.event is not None
                or vegetation is not self.runner._device_resources.static("vegetation").raster.cover_fraction):
            raise _Stop("refused", "device_complete_profile", "one callback-free real event with owned cover required")
        self.event = {}
        wet, summary = self.w.shared.complete_qualification_device_event(state, context, column, field,
            schedule, vegetation, sediment, max_end_s=max_end_s, control=control, policy=policy,
            report_every_s=report_every_s)
        evidence = {"handoff": wet, "qualification_summary": summary}
        try:
            self.event = _summary(self.w, summary, entry=False)
        except _Stop as stop:
            stop.evidence = evidence
            raise
        except Exception as exc:
            raise _Stop("incomplete", "device_event_summary", _dense._detail(exc), exception=exc,
                        evidence=evidence) from exc
        return wet


class DeviceWindowedRunner(DenseRunner):
    """Closed new runner; no public resume, migration, ecology or CPU fallback."""
    def __init_subclass__(cls, **kwargs):
        raise TypeError("device-windowed runner subclasses unsupported")

    def __init__(self, token, **values):
        if token is not _TOKEN:
            raise TypeError("use build_windowed_device_runner")
        self.__dict__.update(values)

    def _ownership(self):
        return self._device_own

    def _array_ops(self, np):
        return self._device_ops

    def _static_parts_record(self, n):
        self._device_own.verify_case_inputs()
        record = {"wind_config": _dense._canonical_input(n.np, self._host_config, "wind.config"),
            "terrain": _dense._canonical_input(n.np, self._water.terrain, "water.terrain"),
            "control": _dense._canonical_input(n.np, self._control, "control"),
            "record_limits": _dense._canonical_input(n.np, self._wind_profile.record_limits, "record_limits"),
            "runner_limits": self._limits.as_record(), "bed_limits": self._bed.limits.as_record(),
            "resource_limits": dataclasses.asdict(self._resource_limits),
            "schedule": self._schedule.as_record(), "native_config_audit": self._native_audit,
            "source_identity": self._wind_profile.source_identity}
        # Current bytes of every static device input are checked against the
        # resource owner's independent mirrors above; this is metadata only.
        record["static_schemas"] = {name: self._device_resources._metadata(
            self._device_resources.static(name), name, allow_host=True)[0] for name in STATIC_NAMES}
        for field in dataclasses.fields(self._host_config):
            expected = (self._device_resources.static("initial_available_fraction")
                        if field.name == "initial_available_fraction" else getattr(self._host_config, field.name))
            if getattr(self._wind.config, field.name) is not expected:
                raise DenseRunnerError("device_config_diverged", f"{field.name} departed from its authored owner")
        return record

    def _wind_event(self, index, event, resolved):
        return self._wind_enter_bound(self, index, event, resolved)

    def _wind_accept_record(self, *args, **kwargs):
        return self._wind_accept_bound(self, *args, **kwargs)

    def _finish(self, n, index, kind, resolved, stage, candidate, t, t_end, returned_elapsed,
                inflow, export, event_bound, body, publish, **kwargs):
        if kind == "wind":
            body["native"]["record"]["api"] = (
                "maple.aeolian.scheduler.windowed_device.prepare_device_windowed_small_feature_wind_record")
        body["qualification"] = dict(scope=SCOPE, known_subset_only=True,
            opaque_terms=_native().shared._QUALIFICATION_OPAQUE_TERMS, claim=CLAIM)
        returned = DenseRunner._finish(self, n, index, kind, resolved, stage, candidate, t, t_end,
            returned_elapsed, inflow, export, event_bound, body, publish, **kwargs)
        # The common engine already published. Only prepared references are
        # assigned here; no new arithmetic/validation/allocation follows it.
        if kind == "rain":
            self._water_counters = self._pending_water_counters
            self._pending_water_counters = None
        return returned

    def _rain_event(self, index, event, resolved):
        w = _native()
        lane = _WaterLane(self, w)
        self._water_record = None
        enter = _cpu_windowed._source_bound(_dense._rain_enter, _native=lane.native)
        native = _cpu_windowed._source_bound(_dense._rain_native, _complete_event=lane.complete)
        prepared = enter(self, index, event, resolved)
        wet = native(self, prepared)
        if lane.entry is None or lane.event is None:
            raise _Stop("incomplete", "missing_admission", "real entry and event summaries are required")
        names = ("gate_count", "logical_work_debited", "logical_read_bytes_debited", "versions_reserved",
                 "h2d_bytes_reserved", "d2h_bytes_reserved", "scalar_reads_reserved")
        counters = {name: self._water_counters[name] + lane.entry[name] + lane.event[name] for name in names}
        for name in ("maximum_host_predicate", "maximum_device_predicate"):
            counters[name] = max(self._water_counters[name], lane.entry[name], lane.event[name])
        limits = self._bed.context.bridge_limits
        if (any(value >= 1 << 63 for value in counters.values())
                or counters["logical_work_debited"] > limits.max_work
                or counters["logical_read_bytes_debited"] > limits.max_logical_read_bytes
                or counters["versions_reserved"] > limits.max_new_versions):
            raise _Stop("refused", "water_cumulative_ceiling", "all short-entry/event owners share monotone original ceilings",
                        elapsed=_dense._elapsed(prepared.t, float(wet.dry_state.t_s)), evidence=wet)
        self._pending_water_counters = counters
        self._water_record = dict(entry=lane.entry, event=lane.event, cumulative=dict(counters))
        try:
            return _dense._rain_accept(self, index, event, resolved, prepared, wet)
        except BaseException:
            self._water_record = None
            raise

    def _rain_receipt_extra(self):
        return {"device_water_qualification": copy.deepcopy(self._water_record)}

    def _abandon(self):
        self._device_own.abandon()
        self._pending_water_counters = None

    @property
    def mobile_state(self):
        raise DenseRunnerError("device_private", "accepted wind pool stays private")

    @property
    def soil_water_m(self):
        raise DenseRunnerError("device_private", "accepted soil water stays private")

    def report(self):
        result = DenseRunner.report(self)
        result["schema"] = "masy.windowed_device_run_report.v1"
        result["qualification"] = dict(scope=SCOPE, known_subset_only=True,
            opaque_terms=_native().shared._QUALIFICATION_OPAQUE_TERMS, claim=CLAIM,
            full_physical_acceptance="Root-owned; not issued by this report",
            allocation_or_fit_proof=None, restart_supported=False)
        return result


def _profile(n, w, bed, wind, schedule, profile, record_limits):
    """Central Native production audit on original HOST controls, before copies."""
    from .dense_wind_profile import DenseProductionWindProfile, audit_dense_production_wind_config
    from .dense_production import schedule_envelope
    if type(profile) is not DeviceWindowedWindProfile or profile.wind_config is not wind.config:
        raise DenseRunnerError("invalid_wind_profile", "explicit original model/config profile required")
    cfg, ctx = wind.config, bed.context
    if type(cfg) is not w.SchedulerPhysicsConfig or record_limits is not profile.record_limits:
        raise DenseRunnerError("invalid_wind_profile", "exact Native config/device record limits required")
    if (type(record_limits) is not w.WindowedWindRecordLimits
            or type(record_limits.block) is not w.device.DeviceWindowedWindBlockLimits
            or type(record_limits.block.step) is not w.device.step.DeviceWindowedReducedStepLimits
            or record_limits.block.max_outer_steps != 1):
        raise DenseRunnerError("invalid_wind_profile", "real Native Small DEVICE record policy required")
    if (profile.source_identity != ctx.source_identity or type(profile.source_identity) is not str
            or len(profile.source_identity) != 64):
        raise DenseRunnerError("source_identity", "same genuine Root-bound Native source tag required")
    for name in ("geometry", "grain_classes", "avalanching_spec", "topographic_commit_spec"):
        if getattr(cfg, name) is not getattr(ctx, name):
            raise DenseRunnerError("incompatible_wind_config", f"{name} must share the actual model owner")
    for field, model_field in (("geometry", "geometry"), ("grain_classes", "grain_classes"),
                               ("avalanching_spec", "avalanching"),
                               ("topographic_commit_spec", "topographic_commit"), ("dormancy_spec", "dormancy")):
        if getattr(cfg, field) is not getattr(profile.model_config, model_field):
            raise DenseRunnerError("incompatible_wind_config", f"{field} must retain its original model owner")
    if (float(cfg.mass_resolution_kg).hex() != float(ctx.mass_resolution_kg).hex()
            or any(c not in "0123456789abcdef" for c in profile.source_identity)
            or type(record_limits.max_blocks) is not int or not 1 <= record_limits.max_blocks <= 2
            or type(record_limits.max_total_outer_steps) is not int
            or not 1 <= record_limits.max_total_outer_steps <= 2):
        raise DenseRunnerError("incompatible_wind_config", "same Native units/tag and bounded two-block record declarations required")
    if (cfg.particle_transport.mobile_state_representation != "flat"
            or cfg.particle_transport.quantile_bins != 8 or cfg.particle_transport.max_active_cohorts != 4000):
        raise DenseRunnerError("unsupported_mobile_state", "exact genuine Small flat/J8/N4000 input required")
    if (cfg.geometry.ny, cfg.geometry.nx, len(cfg.grain_classes.classes)) != (4, 4, 1):
        raise DenseRunnerError("unsupported_dimensions", "exact C16/K1 profile required")
    if (cfg.outer_timestep_s != 1.0 or cfg.detachment_source.mode != "reduced"
            or cfg.aeolian_process_mode.mode != "detachment_transport_deposition"
            or not cfg.topographic_wind_spec.enabled or not cfg.flow_separation_spec.enabled
            or cfg.surface_creep_spec is None or not cfg.surface_creep_spec.active
            or not cfg.avalanching_spec.enabled or cfg.vegetation_wake_spec.enabled
            or cfg.spatial_block_diagnostics_enabled or cfg.water_coupling_spec.enabled
            or cfg.water_coupling_spec.depth_update_rule != "constant_depth"):
        raise DenseRunnerError("unsupported_features", "unchanged fully enabled reduced/FFT/separation/creep/avalanche profile required")
    envelope, binding = schedule_envelope(schedule)
    audited = audit_dense_production_wind_config(
        DenseProductionWindProfile(profile.model_config, profile.calibration, profile.label), cfg,
        wind_envelope_record=envelope, initial_available_fraction=cfg.initial_available_fraction)
    return dict(native_config_audit=audited, schedule_binding=binding)


def _constructor_frontier(bed, pool, soil, statics, config, resource_limits):
    """Join the existing two positive payload laws BEFORE resource copies.

    Every occurrence is additive. HOST static copies and the original HOST
    config tables are included separately from the unchanged device-resource
    producer. Object/constructor/provider terms remain opaque, not zero.
    """
    from maple.core.backend import array_leaves
    cp = _device._genuine_cupy()
    import numpy as np
    pool_arrays = tuple(array_leaves(pool))
    static_arrays = tuple(array for value in statics.values() for array in array_leaves(value))
    if any(type(array) is not cp.ndarray for array in pool_arrays):
        raise DenseRunnerError("private_layout", "all actual pool leaves must be device arrays")
    if any(type(array) not in (cp.ndarray, np.ndarray) for array in static_arrays):
        raise DenseRunnerError("static_layout", "exact Native HOST metadata/device leaves required")
    private = sum(int(array.nbytes) for array in pool_arrays) + int(soil.nbytes)
    device_static = sum(int(array.nbytes) for array in static_arrays if type(array) is cp.ndarray)
    host_static = sum(int(array.nbytes) for array in static_arrays if type(array) is np.ndarray)
    config_arrays = tuple(array_leaves(config))
    if any(type(array) is not np.ndarray for array in config_arrays):
        raise DenseRunnerError("wind_config", "original central-audited HOST wind metadata tables required")
    config_bytes = sum(int(array.nbytes) for array in config_arrays)
    device_arrays = tuple(array for array in (*pool_arrays, soil, *static_arrays) if type(array) is cp.ndarray)
    _compare, device_resources = resource_reservation(private, device_static, len(device_arrays),
        max(int(array.nbytes) for array in device_arrays), resource_limits)
    bed._admit(bed._accepted.state)
    known = bed._last_known
    # Two positive availability-policy comparison arrays plus its predicate.
    policy_compare = 2*16*8 + 16
    device = known["device_payload_bytes"] + device_resources + policy_compare
    host = known["host_payload_bytes"] + 3*host_static + 3*config_bytes
    if device > 2 << 30 or host > 5 << 30 or device_resources > resource_limits.max_work_bytes:
        raise DenseRunnerError("constructor_payload_cap", "joined positive constructor frontiers exceed original ceilings")
    from maple.aeolian.cohorts.capacity import default_available_memory_bytes
    available = default_available_memory_bytes()
    if type(available) is not int or 2*host > available:
        raise DenseRunnerError("constructor_host_availability", "genuine HOST half-headroom refused")
    bed._exchange._check_device_admission(device, device_id=bed._device, safety_fraction=0.5)
    return dict(host_payload_bytes=host, device_payload_bytes=device,
        private_payload_bytes=private, static_device_payload_bytes=device_static,
        static_HOST_payload_bytes=host_static, HOST_config_payload_bytes=config_bytes,
        known_subset_only=True, constructor_allocation=None, external_runtime=None, claim=CLAIM)


def build_windowed_device_runner(bed, schedule, *, wind, water, control, profile,
        record_limits, runner_limits, resource_limits, qualification_budget):
    """Build only after genuine entry API and Root qualification controls exist."""
    if type(bed) is not DeviceWindowedBedOwner:
        raise DenseRunnerError("invalid_input", "exact seven-field device-windowed bed owner required")
    for value, kind in ((wind, DenseWindInputs), (water, DenseWaterInputs),
                        (control, DenseRunnerControl), (runner_limits, DenseRunnerLimits),
                        (resource_limits, DeviceResourceLimits)):
        if type(value) is not kind:
            raise DenseRunnerError("invalid_input", f"exact {kind.__name__} required")
    _schedule_profile(schedule)
    if runner_limits.max_events < 4 or runner_limits.max_report_rows != REPORT_ROWS:
        raise DenseRunnerError("runner_limits", "four events and exact original4096 report rows required")
    n, w = _dense._native(), _native()
    _require_entry_api(w)  # Missing separately accepted SYRUP seam refuses before copies.
    ctx = bed.context
    if (type(qualification_budget) is not w.shared.DeviceWaterQualificationBudget
            or qualification_budget is not ctx.runtime_bounds):
        raise DenseRunnerError("qualification_budget", "exact budget owned by the shared context required")
    process = _process_check(ctx, checkpoint="construction")
    audit = _profile(n, w, bed, wind, schedule, profile, record_limits)
    bed.verify()
    cp = _device._genuine_cupy()
    ops = _device._DeviceArrayOps(cp, 1)
    sc, policy = control.sediment_control, control.completion_policy
    if (type(sc) is not n.SedimentEventControl or type(policy) is not n.CompletionPolicy
            or sc.transport_scheme != "upwind" or sc.storm.implementation != "cuda" or sc.cuda is not True
            or sc.commit is not True or sc.record_exchange_history is not False or sc.deep_publication != "step"):
        raise DenseRunnerError("water_control", "genuine existing CUDA/upwind/commit/step control required")
    sc.validated(); policy.validated()
    w.shared._control_projection(ctx, sc)
    if sc.storm.max_steps > runner_limits.max_storm_steps:
        raise DenseRunnerError("storm_step_limit", "Native storm guard exceeds runner declaration")
    if type(control.report_every_s) is not float or not math.isfinite(control.report_every_s) or control.report_every_s <= 0:
        raise DenseRunnerError("water_control", "positive explicit report interval required")
    if (type(water.terrain) is not n.TerrainReference or not water.terrain.strict
            or water.terrain.allow_masked_nodata or water.terrain.dx_m != ctx.geometry.dx_m
            or water.terrain.initial_committed_elevation_m.shape != (4, 4)
            or type(water.field) is not n.RainfallField or water.field.xp is not cp or water.field.shape != (4, 4)):
        raise DenseRunnerError("water_geometry", "strict unmasked existing drainage and actual device rainfall field required")
    if water.vegetation_cover_fraction is not bed.vegetation.raster.cover_fraction:
        raise DenseRunnerError("vegetation_identity", "water must use the canonical full vegetation holder's cover")
    soil, pool = water.initial_soil_water_m, wind.initial_mobile_state
    if (type(soil) is not cp.ndarray or soil.shape != (4, 4) or soil.dtype != cp.float64
            or not soil.flags.c_contiguous or type(pool) is not n.MobileCohortState
            or not n.mobile_state_is_genuinely_empty(pool)):
        raise DenseRunnerError("private_layout", "genuine empty device wind pool and explicit device soil required")
    if ops.any_nonzero(~cp.isfinite(soil) | (soil < 0.0) | cp.signbit(soil)):
        raise DenseRunnerError("private_layout", "finite nonnegative positive-zero initial soil required")
    if ctx.bridge_limits.max_report_rows != REPORT_ROWS:
        raise DenseRunnerError("report_rows", "Native and wrapper must both retain4096")
    # The unchanged resource producer walks all actual input leaves before its
    # copies; the bed producer has already retained input/current/next/error.
    statics = dict(zip(STATIC_NAMES, (bed.vegetation, bed.threshold_modifier_state,
        ctx.initial_available_fraction, water.column, water.field, water.sediment), strict=True))
    frontier = _constructor_frontier(bed, pool, soil, statics, wind.config, resource_limits)
    from maple.core.backend import to_device
    policy_source = wind.config.initial_available_fraction
    if (type(policy_source) is not n.np.ndarray or policy_source.shape != (4, 4, 1)
            or policy_source.dtype != n.np.float64 or not policy_source.flags.c_contiguous):
        raise DenseRunnerError("initial_availability", "actual central-audited HOST policy required")
    if ops.any_nonzero(to_device(policy_source, cp) != ctx.initial_available_fraction):
        raise DenseRunnerError("initial_availability", "wind and water policies must agree before ownership")
    resources = DeviceDenseResources._create(pool=pool, soil=soil, statics=statics,
        shape=(4, 4), limits=resource_limits, _ops=_CupyOps(cp))
    owned_context = dataclasses.replace(ctx, initial_available_fraction=resources.static("initial_available_fraction"))
    bed._context = owned_context
    bed._vegetation, bed._threshold = resources.static("vegetation"), resources.static("threshold_modifier_state")
    cfg = dataclasses.replace(wind.config, initial_available_fraction=resources.static("initial_available_fraction"))
    device_wind = dataclasses.replace(wind, config=cfg, initial_mobile_state=None)
    device_water = dataclasses.replace(water, column=resources.static("column"), field=resources.static("field"),
        sediment=resources.static("sediment"), vegetation_cover_fraction=bed.vegetation.raster.cover_fraction,
        initial_soil_water_m=None)
    stage_bytes = bed._admit(bed._accepted.state)[2]
    own = _Ownership(bed, resources, ops, stage_bytes)
    zero = n.np.zeros(1, dtype=n.np.float64)
    ref = _device._PrivateRef(resources.generation)
    runner = DeviceWindowedRunner(_TOKEN, _bed=bed, _schedule=schedule, _wind=device_wind,
        _water=device_water, _control=control, _limits=runner_limits, _n_classes=1,
        _inventory_elements=16*7, _inventory0=own.accepted_inventory(ops, n).copy(),
        _pool=ref, _soil=ref, _routing=None, _ledger=ClockLedger(bed.time_s), _next=0, _n_accepted=0,
        _totals=dict(export=zero.copy(), input=zero.copy(), bound=zero.copy(), internal_terminal_deposition=zero.copy(),
            water=dict(rain_m3=0.0, drainage_m3=0.0, storm_export_m3=0.0,
                       dry_surface_removed_m3=0.0, dry_soil_removed_m3=0.0), n_entry_reroutes=0, n_event_reroutes=0),
        _receipts=[], _stop_receipt=None, _active_evidence=None, _water_record=None,
        _water_counters={name: 0 for name in ("gate_count", "logical_work_debited", "logical_read_bytes_debited",
            "versions_reserved", "h2d_bytes_reserved", "d2h_bytes_reserved", "scalar_reads_reserved",
            "maximum_host_predicate", "maximum_device_predicate")}, _pending_water_counters=None,
        _production=_EVIDENCE, _production_limits=owned_context, _worker_identity=process["process_identity"],
        _wind_profile=dataclasses.replace(profile, wind_config=cfg), _host_config=wind.config,
        _native_audit=audit, _device_resources=resources, _resource_limits=resource_limits,
        _constructor_frontier=frontier,
        _device_ops=ops, _device_own=own,
        _wind_enter_bound=_wind_binding(_cpu_windowed.WindowedRunner._wind_event),
        _wind_accept_bound=_wind_binding(_cpu_windowed.WindowedRunner._wind_accept_record))
    runner._static_parts = runner._static_parts_record(n)
    runner._static_parts_sha256 = _bed._sha256(runner._static_parts)
    runner._static_record = dict(schema="masy.windowed_device_static.v1", scope=SCOPE,
        case_sha256=bed.case_sha256, schedule_sha256=schedule.digest,
        case_schedule_sha256=case_schedule_digest(bed.case_sha256, schedule),
        controlling_inputs_sha256=runner._static_parts_sha256, known_subset_only=True,
        opaque_terms=w.shared._QUALIFICATION_OPAQUE_TERMS, claim=CLAIM)
    runner._static_sha256 = _bed._sha256(runner._static_record)
    runner._accepted_bed = own.accepted_label()
    runner._accepted_private = own.private_label(n.np, ref, ref)
    return runner
