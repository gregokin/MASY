"""Explicit imported-case boundary for the closed C64 device profile.

This module constructs no landscape and supplies no scientific defaults. It
places a genuine, already adopted HOST case and calls the existing Native
and MASY owners. Root's real whole-attempt controls remain mandatory.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import resource
import sys
import threading
import types
from pathlib import Path
from typing import Any

from . import dense_runner as _dense
from .case_schedule import CaseRainGroup, CaseSchedule, CaseWindGroup, WindDeclaration
from .case_windowed import CaseWindowedInputs, PreparedCaseWindowedWater
from .dense_schedule import DenseSchedule, WindEvent
from .windowed_device_c64_bed import (
    DeviceWindowedC64BedLimits, DeviceWindowedC64BedOwner,
    _capture_source_bindings, _require_completed_scale_api,
)
from .windowed_device_c64_runner import (
    DeviceWindowedC64WindProfile, build_windowed_device_c64_runner,
)

_TOKEN = object()
_ADMISSION_TOKEN = object()
_BINDING_SCHEMA = "masy.device_c64_case_request.v1"
_BACKEND_FIELDS = ("sediment_control.storm.implementation",)
_CHECKPOINT_ORIGIN_SCHEMA = "masy.device_c64_case_origin.v1"
_CHECKPOINT_VALIDATION_TOKEN = object()


class CaseDeviceC64Error(ValueError):
    def __init__(self, code, detail):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclasses.dataclass(frozen=True, eq=False, repr=False, slots=True)
class _CaseAdmission:
    token: Any
    binding_sha256: str
    state: Any
    context: Any
    budget: Any
    case_record_bytes: bytes
    owner_id: int | None = None


def _check_case_admission(admission, state, context, budget, case_record):
    """Producer-only immediate genesis join, not historical byte evidence."""
    if (type(admission) is not _CaseAdmission or admission.token is not _ADMISSION_TOKEN
            or admission.owner_id is not None
            or admission.state is not state or admission.context is not context
            or admission.budget is not budget
            or admission.binding_sha256 != budget.case_request_binding_sha256
            or type(case_record) is not dict
            or case_record.get("schema") != "masy.imported_device_c64_case.v1"
            or case_record.get("case_request_binding_sha256") != admission.binding_sha256
            or _bytes(case_record, 65536) != admission.case_record_bytes):
        raise CaseDeviceC64Error("case_admission", "genuine producer admission for this placed genesis is required")
    return admission


def _bind_case_admission_owner(admission, owner):
    if (type(admission) is not _CaseAdmission or admission.token is not _ADMISSION_TOKEN
            or admission.owner_id is not None or owner._case_admission is not admission
            or owner.context is not admission.context
            or _bytes(owner._case_record, 65536) != admission.case_record_bytes):
        raise CaseDeviceC64Error("case_admission", "one genuine constructed owner must publish admission")
    object.__setattr__(admission, "owner_id", id(owner))  # No owner back-reference/cycle.


def _check_case_bed_admission(bed, budget):
    admission = getattr(bed, "_case_admission", None)
    if (type(admission) is not _CaseAdmission or admission.token is not _ADMISSION_TOKEN
            or admission.owner_id != id(bed) or admission.budget is not budget
            or admission.binding_sha256 != budget.case_request_binding_sha256
            or bed._case_record.get("schema") != "masy.imported_device_c64_case.v1"
            or bed._case_record.get("case_request_binding_sha256") != admission.binding_sha256
            or _bytes(bed._case_record, 65536) != admission.case_record_bytes):
        raise CaseDeviceC64Error("case_admission", "this genuine owner requires its producer request admission")
    return admission


@dataclasses.dataclass(frozen=True)
class CaseDeviceC64PlacementLimits:
    max_known_host_bytes: int
    max_known_device_bytes: int
    max_metadata_bytes: int

    def __post_init__(self):
        for name, cap in (("max_known_host_bytes", 5 << 30),
                          ("max_known_device_bytes", 2 << 30),
                          ("max_metadata_bytes", 4 << 20)):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= cap:
                raise CaseDeviceC64Error("placement_limit", name)


def _bytes(value, maximum):
    """Closed primitive image, bounded before encoded chunks are retained."""
    todo = [(value, 0)]
    nodes = chars = 0
    while todo:
        item, depth = todo.pop()
        nodes += 1
        if depth > 32 or nodes > 131072:
            raise CaseDeviceC64Error("metadata_limit", "primitive graph exceeds the closed domain")
        kind = type(item)
        if kind is dict:
            if 2 * len(item) + len(todo) > 131072 - nodes:
                raise CaseDeviceC64Error("metadata_limit", "dictionary expansion exceeds quota")
            if any(type(key) is not str for key in item):
                raise CaseDeviceC64Error("metadata_type", "plain string keys required")
            todo.extend((part, depth + 1) for pair in item.items() for part in pair)
        elif kind in (tuple, list):
            if len(item) + len(todo) > 131072 - nodes:
                raise CaseDeviceC64Error("metadata_limit", "sequence expansion exceeds quota")
            todo.extend((part, depth + 1) for part in item)
        elif kind is str:
            chars += len(item)
        elif kind is float:
            if not math.isfinite(item):
                raise CaseDeviceC64Error("metadata_type", "finite metadata required")
        elif kind is int:
            if not -(1 << 63) < item < (1 << 63):
                raise CaseDeviceC64Error("metadata_type", "int63 metadata required")
        elif item is not None and kind is not bool:
            raise CaseDeviceC64Error("metadata_type", "primitive metadata required")
        if chars > maximum:
            raise CaseDeviceC64Error("metadata_limit", "metadata text limit")
    output = bytearray()
    for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
        encoded = chunk.encode()
        if len(output) + len(encoded) > maximum:
            raise CaseDeviceC64Error("metadata_limit", "encoded metadata limit")
        output.extend(encoded)
    return bytes(output)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _case_api(shared):
    kind = vars(shared).get("DeviceWaterC64CaseBudget")
    if type(kind) is not type or kind.__module__ != shared.__name__:
        raise CaseDeviceC64Error("case_api", "separately qualified Native case contract is absent")
    for name in ("c64_case_budget_record", "c64_case_budget_bytes", "_c64_case_budget_runtime"):
        function = vars(shared).get(name)
        if (type(function) is not types.FunctionType or function.__globals__ is not vars(shared)
                or function.__module__ != shared.__name__):
            raise CaseDeviceC64Error("case_api", "genuine owning case contract API required")
    return kind


def _canonical(value):
    import numpy as np
    return _dense._canonical_input(np, value, "device_case")


def _bridge_image(limits):
    # Native accepts a genuine bounded absolute Path. Canonicalize only after
    # that domain check; never replace the Path retained by Native limits.
    path = limits.run_dir
    if (type(path) is not type(Path()) or not path.is_absolute()
            or any(part == ".." for part in path.parts)
            or sum(len(part) + 1 for part in path.parts) > 4096):
        raise CaseDeviceC64Error("run_path", "bounded genuine absolute Native run Path required")
    text = str(path)
    if len(text) > 4096 or "\x00" in text:
        raise CaseDeviceC64Error("run_path", "Native run path domain")
    return {f.name: text if f.name == "run_dir" else _canonical(getattr(limits, f.name))
            for f in dataclasses.fields(limits)}


def _control_join(host, cuda):
    """Exactly two explicit backend selectors; all other fields stay exact."""
    from maple_syrup.complete_event import CompletionPolicy
    from maple_syrup.sediment_event import SedimentEventControl
    from maple_syrup.storm import StormControl
    if type(host) is not _dense.DenseRunnerControl or type(cuda) is not _dense.DenseRunnerControl:
        raise CaseDeviceC64Error("water_control", "genuine controls required")
    for control in (host, cuda):
        if (type(control.sediment_control) is not SedimentEventControl
                or type(control.sediment_control.storm) is not StormControl
                or type(control.completion_policy) is not CompletionPolicy):
            raise CaseDeviceC64Error("water_control", "genuine Native control holders required")
        control.sediment_control.validated()
        control.completion_policy.validated()
    hs, cs = host.sediment_control, cuda.sediment_control
    if (hs.storm.implementation != "array" or hs.cuda is not False
            or cs.storm.implementation != "cuda" or cs.cuda is not True):
        raise CaseDeviceC64Error("water_backend", "explicit array/False to cuda/True selection required")
    # Native cuda is a read-only property of storm.implementation, not an
    # independently authorable dataclass field. Compare the complete holder
    # after reversing its single backend field; the derived flag follows it.
    comparison = dataclasses.replace(cuda, sediment_control=dataclasses.replace(cs,
        storm=dataclasses.replace(cs.storm, implementation="array")))
    if _canonical(host) != _canonical(comparison):
        raise CaseDeviceC64Error("water_control_changed", "a non-backend control field differs")
    if (cs.transport_scheme != "upwind" or cs.commit is not True
            or cs.record_exchange_history is not False or cs.deep_publication != "step"
            or cuda.completion_policy.terminal_storage_water is not True
            or cuda.completion_policy.deposit_mobile_at_flow_end is not True):
        raise CaseDeviceC64Error("water_policy", "explicit existing C64 and accounted dry-handoff policy required")


def _host_domain(windowed, prepared, cuda, record_limits):
    import numpy as np
    from maple.aeolian.scheduler import windowed_record as records
    from maple.aeolian.scheduler.windowed_device import DeviceWindowedK1DimensionWindBlockLimits
    from maple.surface.voxels.windowed_prefix import require_prefix_packets
    if type(windowed) is not CaseWindowedInputs or type(prepared) is not PreparedCaseWindowedWater:
        raise CaseDeviceC64Error("host_owner", "genuine adopted HOST case and water owners required")
    windowed.verify()
    prepared.verify(windowed)
    state, ctx, cfg = windowed.state, windowed.context, windowed.config
    col, g = state.voxel_column, ctx.geometry
    packets = require_prefix_packets(col, operation="prepare_device_c64_case")
    header = col.deep_store.header
    if ((g.ny, g.nx, len(ctx.grain_classes.classes), col.n_voxels, header.window_voxels) != (8, 8, 1, 20, 3)
            or g.dx_m != g.dy_m or cfg.geometry.voxel_growth.enabled is not False
            or col.pending_deep is not None or col.window_cache is not None
            or col.deep_store.format_version != 1 or col.deep_store.encoding != "full"
            or col.deep_store.overlay_max_bytes is not None
            or not 64 <= header.chunk_columns <= 1024 or packets.packet.shape != (8, 8, 8)):
        raise CaseDeviceC64Error("case_domain", "published full-v1 single-chunk C64/K1/V20/W3/E1P8 required")
    if (windowed.initial_mobile_state is not None or windowed.ledger_reconciliation_baseline is not None
            or not bool(np.all(state.water.depth_m == 0.0))
            or not bool(np.all(state.water.mobile_mass_by_cell_class_kg == 0.0))):
        raise CaseDeviceC64Error("initial_wet_or_mobile", "initial wind requires the actual dry unseeded case; no reset is applied")
    if float(state.committed_topography.last_commit_time_s).hex() != float(windowed.time_s).hex():
        raise CaseDeviceC64Error("initial_clock", "actual committed and case clocks differ")
    resolved = windowed.static_inputs["resolved"]
    if (cfg.time_and_forcing.outer_timestep_s != 1.0 or cfg.detachment_source.mode != "reduced"
            or cfg.aeolian_process_mode.mode != "detachment_transport_deposition"
            or not cfg.topographic_wind.enabled or not cfg.flow_separation.enabled
            or not cfg.surface_creep.active or not cfg.avalanching.enabled
            or cfg.vegetation_wake.enabled or resolved["requested_diagnostics"].spatial_snapshots
            or resolved["prescribed_external_shear"] is not None
            or cfg.water_coupling.enabled is not False
            or cfg.water_coupling.depth_update_rule != "constant_depth"
            or cfg.particle_transport.mobile_state_representation != "flat"
            or cfg.particle_transport.quantile_bins != 8
            or cfg.particle_transport.max_active_cohorts != 4000):
        raise CaseDeviceC64Error("case_model", "authored model is outside the unchanged C64 feature/step domain")
    water = prepared.water
    from .windowed_device_c64_runner import _routing_policy
    if (not _routing_policy(water.terrain) or not bool(np.all(water.column.active_mask))
            or water.vegetation_cover_fraction is not windowed.vegetation.raster.cover_fraction):
        raise CaseDeviceC64Error("water_domain", "unmasked actual terrain and canonical cover required")
    _control_join(prepared.control, cuda)
    if (type(record_limits) is not records.WindowedWindRecordLimits
            or type(record_limits.block) is not DeviceWindowedK1DimensionWindBlockLimits):
        raise CaseDeviceC64Error("record_limits", "genuine complete Native C64 limits required")
    record_limits.block.__post_init__()
    records._count(record_limits.max_accounting_reasons, "max_accounting_reasons", records.admission._MAX_LEAVES)
    from .windowed_device_c64_runner import _dimension_join
    limits = record_limits.block
    p = limits.profile
    _dimension_join(p, p, limits.step.profile, limits.step.bed_limits.profile,
        shape=(p.ny, p.nx, p.n_voxels, p.window_voxels, p.n_classes, p.hop_bins),
        impact_rows=limits.step.bed_limits.max_impact_rows,
        creep_rows=limits.step.creep_limits.max_impact_rows,
        solve_rows=limits.solve.bed_limits.max_impact_rows,
        validation_rows=limits.validation.bed_limits.max_impact_rows,
        block_steps=limits.max_outer_steps, blocks=record_limits.max_blocks,
        total_steps=record_limits.max_total_outer_steps)


def _schedule(windowed, prepared, schedule_input, runner_limits):
    from maple.core.parameters.wind_forcing import validate_wind_forcing_record, resolve_wind_forcing_record
    original = windowed.static_inputs["resolved"]["wind_record"]
    validate_wind_forcing_record(original)
    if len(original.blocks) != 2 or any(b.duration_s != 1.0 for b in original.blocks):
        raise CaseDeviceC64Error("wind_record", "exact two complete authored one-second blocks required")
    # Compare all five Native block fields, including original derived clocks.
    rebuilt = resolve_wind_forcing_record({"blocks": [dict(duration_s=b.duration_s,
        shear_velocity_m_s=b.shear_velocity_m_s, direction_deg=b.direction_deg) for b in original.blocks]})
    if _canonical(original.blocks) != _canonical(rebuilt.blocks):
        raise CaseDeviceC64Error("wind_record", "original Native block fields cannot be represented losslessly")
    declarations = tuple(WindDeclaration(i, b.shear_velocity_m_s, b.direction_deg, b.duration_s)
                         for i, b in enumerate(original.blocks))
    grouped = CaseSchedule.from_input(schedule_input, blocks=declarations,
        case_sha256=windowed.identity_sha256, wind_record_sha256=_sha(_bytes(_canonical(original), 4 << 20)),
        water_profile_id=prepared.profile_id, water_identity_sha256=prepared.identity_sha256,
        max_events=runner_limits.max_events)
    if tuple(type(e) for e in grouped.events) != (CaseWindGroup, CaseRainGroup, CaseWindGroup, CaseRainGroup):
        raise CaseDeviceC64Error("schedule", "exact four-event WRWR case partition required")
    events = []
    for event in grouped.events:
        if type(event) is CaseWindGroup:
            if len(event.blocks) != 1:
                raise CaseDeviceC64Error("schedule", "one original block per wind event required")
            b = event.blocks[0]
            events.append(WindEvent(b.duration_s, b.shear_velocity_m_s, b.direction_deg))
        else:
            events.append(event.event)
    result = DenseSchedule(tuple(events))
    from .windowed_device_c64_runner import _schedule_profile
    _schedule_profile(result)
    return original, grouped, result


def _metadata_charge(maximum):
    # Three live generations: final projection, config image and full wind
    # image. Canonical reference/text words stay positive separately from
    # encoded buffers; these are logical occurrences, never Python ABI fees.
    images = 2 * maximum + (4 << 20)
    return 3 * (images + 4 * images + 2 * 8 * 131072)


def _metadata_admit(limits):
    from maple.aeolian.cohorts.capacity import default_available_memory_bytes
    known = _metadata_charge(limits.max_metadata_bytes)
    if known > limits.max_known_host_bytes:
        raise CaseDeviceC64Error("metadata_limit", "prospective canonical and encoded images exceed HOST ceiling")
    available = default_available_memory_bytes()
    if type(available) is not int or 2 * known > available:
        raise CaseDeviceC64Error("host_availability", "metadata projection half-headroom refused")
    return known


def device_c64_request_binding(windowed, prepared_host_water, *, schedule_input, cuda_control,
        record_limits, bridge_limits, bed_limits, resource_limits, runner_limits,
        placement_limits, calibration, label):
    """Actual HOST projection used by Root before issuing the bound CaseBudget.

    This verifies input custody and maps schedule metadata; no device placement,
    transport, grant fabrication or Native planner is performed.
    """
    if type(placement_limits) is not CaseDeviceC64PlacementLimits:
        raise CaseDeviceC64Error("placement_limit", "exact placement limits required")
    placement_limits.__post_init__()
    metadata_charge = _metadata_admit(placement_limits)
    _host_domain(windowed, prepared_host_water, cuda_control, record_limits)
    from .dense_device_resources import DeviceResourceLimits
    if (type(bed_limits) is not DeviceWindowedC64BedLimits
            or type(resource_limits) is not DeviceResourceLimits
            or type(runner_limits) is not _dense.DenseRunnerLimits):
        raise CaseDeviceC64Error("limits", "genuine MASY C64/resource/runner limits required")
    for limits in (bed_limits, resource_limits, runner_limits):
        limits.__post_init__()
    from maple_syrup.windowed_shared_bed_device import PositiveWindowedDeviceWaterLimits
    if type(bridge_limits) is not PositiveWindowedDeviceWaterLimits:
        raise CaseDeviceC64Error("limits", "genuine water bridge limits required")
    bridge_limits.__post_init__()
    if type(calibration) is not str or calibration != "synthetic_test_fixture":
        raise CaseDeviceC64Error("calibration_pending", "this first imported interface profile is synthetic_test_fixture only")
    if type(label) is not str or not label.strip() or len(label) > 4096:
        raise CaseDeviceC64Error("label", "explicit bounded nonempty interface label required")
    _, grouped, schedule = _schedule(windowed, prepared_host_water, schedule_input, runner_limits)
    record = windowed.identity_record()
    projection = dict(schema=_BINDING_SCHEMA, windowed_identity_sha256=windowed.identity_sha256,
        water_identity_sha256=prepared_host_water.identity_sha256,
        native_case_identity_sha256=record["native_case_identity_sha256"],
        authored_file_pins=record["file_pins"], storage_choices=record["storage_choices"],
        native_committed_regrouping=record["native_committed_regrouping"],
        host_model_sha256=_sha(_bytes(_canonical(windowed.config), placement_limits.max_metadata_bytes)),
        case_schedule=grouped.as_record(), projected_schedule=schedule.as_record(),
        cuda_control=_canonical(cuda_control), backend_changes=list(_BACKEND_FIELDS),
        derived_cuda={"kind": "Native read-only property", "host": False, "device": True},
        record_limits=_canonical(record_limits), bridge_limits=_bridge_image(bridge_limits),
        bed_limits=dataclasses.asdict(bed_limits), resource_limits=dataclasses.asdict(resource_limits),
        runner_limits=dataclasses.asdict(runner_limits), placement_limits=dataclasses.asdict(placement_limits),
        metadata_projection_logical_bytes=metadata_charge,
        calibration=calibration, label=label, initial_time_hex=windowed.time_s.hex())
    return json.loads(_bytes(projection, placement_limits.max_metadata_bytes))


def _source_frontier():
    codes = slots = text = 0
    def visit(code, depth=0):
        nonlocal codes, slots, text
        if depth > 24 or codes >= 8192:
            raise CaseDeviceC64Error("source_image", "finite producer code frontier")
        codes += 1
        slots += 1 + code.co_nlocals + code.co_stacksize + len(code.co_cellvars) + len(code.co_freevars)
        text += len(code.co_code) + len(code.co_linetable) + len(code.co_exceptiontable)
        text += sum(len(x) for x in code.co_consts if type(x) is str)
        for child in code.co_consts:
            if type(child) is types.CodeType:
                visit(child, depth + 1)
    for namespace, name, descriptor, code in _SOURCE_BINDINGS:
        actual = namespace.get(name)
        function = (actual.__func__ if type(actual) in (classmethod, staticmethod)
                    else actual.fget if type(actual) is property else actual)
        if actual is not descriptor or getattr(function, "__code__", None) is not code:
            raise CaseDeviceC64Error("source_image", "producer binding/code changed")
        visit(code)
    # Positive word/metadata occurrences; never CPython allocation/ABI fees.
    return 3 * (32 * codes + 8 * slots + 25 * (32 * codes + 8 * slots) + 4 * text)


def _precopy_process(plan, placement_limits, expected=None):
    from .dense_device_runner import _genuine_cupy
    from maple.aeolian.cohorts.capacity import default_available_memory_bytes
    from maple.aeolian.cohorts.memory_budget import DeviceMemorySnapshot, default_device_memory_snapshot
    for kind, required in ((resource.RLIMIT_CPU, (480, 540)),
                           (resource.RLIMIT_FSIZE, (64 << 20, 64 << 20)),
                           (resource.RLIMIT_CORE, (0, 0))):
        if resource.getrlimit(kind) != required:
            raise CaseDeviceC64Error("process_limits", "original whole-attempt policy required before placement")
    cp = _genuine_cupy()
    identity = dict(pid=os.getpid(), thread=threading.get_ident(), device=int(cp.cuda.runtime.getDevice()))
    if expected is not None and identity != expected:
        raise CaseDeviceC64Error("process_changed", "same process/thread/device required")
    if (plan["known_host_bytes"] > placement_limits.max_known_host_bytes
            or plan["known_device_bytes"] > placement_limits.max_known_device_bytes):
        raise CaseDeviceC64Error("placement_cap", "positive caller+placement occurrences exceed declared limits")
    available = default_available_memory_bytes()
    if type(available) is not int or 2 * plan["known_host_bytes"] > available:
        raise CaseDeviceC64Error("host_availability", "genuine half-headroom refused")
    snapshot = default_device_memory_snapshot()
    if (type(snapshot) is not DeviceMemorySnapshot or snapshot.device_id != identity["device"]
            or snapshot.allocator_supported is not True or snapshot.pool_limit_bytes != 2 << 30):
        raise CaseDeviceC64Error("pool_limit", "genuine original2GiB active pool required")
    from maple.surface.active_layer import bed_exchange_device
    bed_exchange_device._check_device_admission(plan["known_device_bytes"],
        device_id=identity["device"], safety_fraction=0.5)
    return cp, identity


def _plan(windowed, prepared, pool, budget, metadata_size, metadata_limit):
    from maple.core.backend.arrays import array_leaf_inventory
    col = windowed.state.voxel_column
    physical = (windowed.state, windowed.vegetation, windowed.threshold_modifiers)
    host = array_leaf_inventory((physical, windowed.context, windowed.config, windowed.static_inputs,
                                 prepared.water))
    # Native cohort.py: ten float64, four int32 and one bool C4000 arrays.
    # None denotes the prospective constructor, never a fake Native pool.
    pool_bytes, pool_leaves = 97 * 4000, 15
    if pool is not None:
        inventory = array_leaf_inventory(pool)
        if (inventory["transfer_bytes"], inventory["n_leaves"]) != (pool_bytes, pool_leaves):
            raise CaseDeviceC64Error("pool_layout", "actual Native empty-pool payload differs from Source")
    # Terrain stays HOST; canonical cover is placed with vegetation, not twice.
    sediment_arrays = tuple(getattr(prepared.water.sediment, f.name)
        for f in dataclasses.fields(prepared.water.sediment) if f.name.startswith("d_"))
    moving = (physical, windowed.context.initial_available_fraction, prepared.water.column,
              prepared.water.field, sediment_arrays, prepared.water.initial_soil_water_m)
    visits = array_leaf_inventory(moving)
    packets = col.height_packet.packet.nbytes
    encoded_base = col.height_packet.binding.encoded_base.nbytes
    references = (budget.current_runtime_ref, budget.producer_source, budget.grant,
                  *(ref for _, ref in budget.source_bindings))
    raw = sum(ref.size for ref in references)
    from .windowed_device_c64_bed import _wrapper_source_frontier
    source_words = _source_frontier() + _wrapper_source_frontier()["host_source_bytes"]
    initial = windowed.identity_record()["construction_plan"]
    # The Native mapper reuses the model/fraction and makes two K1/J8 fp64
    # outputs (2*1*8*8=128B). Price all model leaves as an overcoverage of
    # the mapped subset, then verify the actual config inventory below.
    wind_config = array_leaf_inventory((windowed.config, windowed.context.initial_available_fraction))["transfer_bytes"] + 128
    # Every role is additive, including alias visits. No dedup/free credit.
    terms = dict(caller_retained_dense_input_bytes=initial["caller_retained_input_root_bytes"],
        retained_HOST_leaf_occurrence_bytes=host["transfer_bytes"] + pool_bytes,
        retained_prefix_HOST_bytes=packets + encoded_base,
        HOST_pool_and_error_bytes=2 * pool_bytes,
        actual_H2D_payload_bytes=visits["transfer_bytes"] + pool_bytes + packets,
        device_encoded_base_bytes=encoded_base,
        placement_verification_current_next_error_bytes=3 * (visits["transfer_bytes"] + pool_bytes + packets + encoded_base),
        Source_raw_current_next_error_bytes=3 * raw,
        producer_Source_word_bytes=source_words,
        metadata_current_next_error_bytes=3 * metadata_size,
        intermediate_metadata_logical_bytes=_metadata_charge(metadata_limit),
        prospective_wind_config_current_next_error_bytes=3 * wind_config,
        producer_admission_reference_words=3 * 7 * 8,
        verification_HOST_download_bytes=2 * (visits["transfer_bytes"] + pool_bytes + packets + encoded_base),
        retained_store_and_IO_overlap_bytes=3 * windowed._limits.max_store_bytes)
    device = (terms["actual_H2D_payload_bytes"] + encoded_base
              + terms["placement_verification_current_next_error_bytes"])
    return dict(schema="masy.device_c64_case_prefix.v1", terms=terms,
        known_host_bytes=sum(terms.values()), known_device_bytes=device,
        transfer_leaf_visits=visits["n_leaves"] + pool_leaves, alias_credit_taken=False,
        packet_transfer_bytes=packets, encoded_base_bytes=encoded_base,
        downstream_bed_resource_Native_charges="separate and unchanged",
        constructor_allocation=None, external_runtime=None, whole_process_fit=False)


def _placed_tree(host, cp):
    """Actual Native placement plus bounded byte-exact per-leaf readback."""
    import numpy as np
    from maple.core.backend import to_device_tree, to_host
    from maple.core.backend.arrays import _map_tree
    def visits(value):
        result = []
        def keep(array):
            result.append(array)
            return array
        _map_tree(value, keep)  # Exact same init=True, alias-counted Native walk.
        return result
    result = to_device_tree(host, cp)
    left, right = visits(host), visits(result)
    if len(left) != len(right):
        raise CaseDeviceC64Error("placement_layout", "Native leaf layout changed")
    for old, new in zip(left, right, strict=True):
        if (type(old) is not np.ndarray or type(new) is not cp.ndarray
                or old.shape != new.shape or old.dtype != new.dtype):
            raise CaseDeviceC64Error("placement_layout", "exact dtype/shape/backend required")
        readback = to_host(new)
        if not np.array_equal(old.view(np.uint8).reshape(-1), readback.view(np.uint8).reshape(-1)):
            raise CaseDeviceC64Error("placement_changed", "Native placement changed bytes")
    return result


def _place_water(water, cp, vegetation):
    import numpy as np
    arrays = lambda value: {f.name for f in dataclasses.fields(value)
                            if type(getattr(value, f.name)) is np.ndarray}
    column_names = arrays(water.column)
    sediment_names = {"d_diameter", "d_raindrop_a", "d_raindrop_b", "d_raindrop_c", "d_attenuation",
                      "d_raindrop_max", "d_settling", "d_particle_mass_g"}
    host_metadata_names = {"diameter_m", "raindrop_a", "raindrop_b", "raindrop_c",
        "raindrop_depth_attenuation_per_cm", "raindrop_max_depth_per_reference_s_m",
        "settling_velocity_m_s", "particle_mass_g"}
    if (arrays(water.field) != {"multiplier"}
            or arrays(water.sediment) != sediment_names | host_metadata_names):
        raise CaseDeviceC64Error("water_layout", "unqualified new Native parameter array field")
    def holder(value, names):
        moved = {name: _placed_tree(getattr(value, name), cp) for name in names}
        return dataclasses.replace(value, xp=cp, **moved)
    return dataclasses.replace(water,
        column=holder(water.column, column_names), field=holder(water.field, {"multiplier"}),
        sediment=holder(water.sediment, sediment_names),
        vegetation_cover_fraction=vegetation.raster.cover_fraction,
        initial_soil_water_m=_placed_tree(water.initial_soil_water_m, cp))


@dataclasses.dataclass(repr=False, eq=False)
class PreparedDeviceC64Case:
    """One placed case/runner; no dense compiled forest or replacement bed."""
    bed: Any
    schedule: Any
    wind: Any
    water: Any
    control: Any
    profile: Any
    record_limits: Any
    runner_limits: Any
    resource_limits: Any
    qualification_budget: Any
    _runner: Any
    _host: Any
    _prepared_host_water: Any
    _request_kwargs: dict
    _binding_bytes: bytes
    _prefix: dict
    _identity: dict
    _token: Any
    _failed: bool = False
    _checkpoint_handed_off: bool = False

    def build_runner(self):
        self.verify()
        return self  # Same consumed runner, reached only through guarded step.

    @property
    def accepted(self):
        return self._runner.accepted  # Genuine frozen AcceptedBoundary.

    @property
    def stop_receipt(self):
        return self._runner.stop_receipt  # Existing owner returns a deep copy.

    def verify(self):
        if self._checkpoint_handed_off is not False:
            raise CaseDeviceC64Error("case_handed_off", "accepted-boundary store owns this stepping attempt")
        if self._token is not _TOKEN or self._failed:
            raise CaseDeviceC64Error("case_stopped", "case boundary was refused; no replay")
        try:
            return self._verify_current()
        except BaseException:
            self._failed = True
            raise

    def _verify_current(self):
        from maple_syrup import windowed_shared_bed_device as shared
        actual = device_c64_request_binding(self._host, self._prepared_host_water, **self._request_kwargs)
        if _bytes(actual, self._request_kwargs["placement_limits"].max_metadata_bytes) != self._binding_bytes:
            raise CaseDeviceC64Error("case_changed", "original input/request projection changed")
        shared._c64_case_budget_runtime(self.qualification_budget, verify_files=True)
        _precopy_process(self._prefix, self._request_kwargs["placement_limits"], self._identity)
        self.bed.verify()
        return self

    def step(self):
        self.verify()
        try:
            result = self._runner.step()
            self.verify()
            return result
        except BaseException:
            self._failed = True
            raise  # Existing runner retains genuine accepted/active evidence.

    def report(self):
        return dict(schema="masy.device_c64_imported_case_report.v1",
            case_request_binding_sha256=_sha(self._binding_bytes),
            request_binding=json.loads(self._binding_bytes), known_prefix=copy.deepcopy(self._prefix),
            backend_changes=list(_BACKEND_FIELDS),
            derived_cuda={"kind": "Native read-only property", "host": False, "device": True},
            input_guard_failed=self._failed,
            runner=self._runner.report(), calibration="synthetic interface; production pending",
            restart_qualified=False, general_domain_qualified=False,
            external_controls="one600wall;CPU480/540;pool2GiB;RSS5GiB;driver9GiB;run512MiB;file64MiB;streams4MiB")


def prepare_device_c64_case(windowed, prepared_host_water, *, schedule_input, cuda_control,
        record_limits, bridge_limits, qualification_budget, bed_limits, resource_limits,
        runner_limits, placement_limits, calibration, label, request_binding):
    """Prepare one real imported case under a genuine separately issued budget.

    Root must install and monitor the original entire-attempt controls before
    calling. Failed calls can leave Native artifacts; no deep rollback/GC or
    automatic retry is supplied. Constructor/provider allocations stay opaque.
    """
    from maple_syrup import windowed_shared_bed_device as shared
    if type(qualification_budget) is not _case_api(shared):
        raise CaseDeviceC64Error("case_budget_required", "synthetic factory authority cannot admit a supplied case")
    qualification_budget.validated()
    kwargs = dict(schedule_input=schedule_input, cuda_control=cuda_control, record_limits=record_limits,
        bridge_limits=bridge_limits, bed_limits=bed_limits, resource_limits=resource_limits,
        runner_limits=runner_limits, placement_limits=placement_limits, calibration=calibration, label=label)
    actual = device_c64_request_binding(windowed, prepared_host_water, **kwargs)
    binding = _bytes(actual, placement_limits.max_metadata_bytes)
    if (_bytes(request_binding, placement_limits.max_metadata_bytes) != binding
            or qualification_budget.case_request_binding_sha256 != _sha(binding)):
        raise CaseDeviceC64Error("request_binding", "actual complete case projection differs from Root admission")
    original, grouped, schedule = _schedule(windowed, prepared_host_water, schedule_input, runner_limits)
    prefix = _plan(windowed, prepared_host_water, None, qualification_budget, len(binding), placement_limits.max_metadata_bytes)
    cp, identity = _precopy_process(prefix, placement_limits)
    _require_completed_scale_api(preflight=True)
    # The priced raw/current-next/error prefix precedes actual Source/grant I/O.
    shared._c64_case_budget_runtime(qualification_budget, verify_files=True)
    from maple.core.types.cohort import zeros_mobile_cohort_state
    pool = zeros_mobile_cohort_state(4000)
    actual_prefix = _plan(windowed, prepared_host_water, pool, qualification_budget, len(binding), placement_limits.max_metadata_bytes)
    if actual_prefix != prefix:
        raise CaseDeviceC64Error("pool_layout", "actual Native constructor differs from prospective admission")
    from .dense_wind_profile import DenseProductionWindProfile, build_dense_production_wind_config
    wind_config = build_dense_production_wind_config(
        DenseProductionWindProfile(windowed.config, calibration, label), wind_envelope_record=original,
        initial_available_fraction=windowed.context.initial_available_fraction)
    from maple.core.backend.arrays import array_leaf_inventory
    if 3 * array_leaf_inventory(wind_config)["transfer_bytes"] > prefix["terms"]["prospective_wind_config_current_next_error_bytes"]:
        raise CaseDeviceC64Error("wind_config_layout", "Native mapper exceeded prospective array occurrence envelope")
    _precopy_process(prefix, placement_limits, identity)
    if _bytes(device_c64_request_binding(windowed, prepared_host_water, **kwargs), placement_limits.max_metadata_bytes) != binding:
        raise CaseDeviceC64Error("case_changed", "case changed during pre-copy admission")
    from maple.core.backend import read_transfer_counters
    before = dataclasses.asdict(read_transfer_counters())
    state = _placed_tree(windowed.state, cp)
    from maple.surface.voxels.windowed_prefix import move_prefix_packets, require_prefix_packets
    host_column = windowed.state.voxel_column
    column = move_prefix_packets(state.voxel_column, host_column, cp)
    if (column.deep_store is not host_column.deep_store or column.manifest_digest != host_column.manifest_digest
            or column.generation != host_column.generation):
        raise CaseDeviceC64Error("store_identity", "placement changed the original immutable deep-store handle/version")
    packet = require_prefix_packets(column, operation="imported_c64_placement")
    import numpy as np
    from maple.core.backend import to_host
    for old, new in ((host_column.height_packet.packet, packet.packet),
                     (host_column.height_packet.binding.encoded_base, packet.binding.encoded_base)):
        if not np.array_equal(old.view(np.uint8).reshape(-1), to_host(new).view(np.uint8).reshape(-1)):
            raise CaseDeviceC64Error("packet_changed", "packet or E1 encoded-base placement changed bytes")
    vegetation = _placed_tree(windowed.vegetation, cp)
    threshold = _placed_tree(windowed.threshold_modifiers, cp)
    fraction = _placed_tree(windowed.context.initial_available_fraction, cp)
    device_pool = _placed_tree(pool, cp)
    water = _place_water(prepared_host_water.water, cp, vegetation)
    b = record_limits.block
    context_values = {f.name: getattr(windowed.context, f.name) for f in dataclasses.fields(windowed.context)}
    context_values.update(initial_available_fraction=fraction,
        native_solve_limits=b.solve, native_validation_limits=b.validation,
        native_commit_limits=dataclasses.replace(b.commit, publication_max_records=320),
        native_dimension_profile=b.profile, source_identity=qualification_budget.native_source_identity,
        bridge_limits=bridge_limits, runtime_bounds=qualification_budget, window_max_journal_records=320)
    context = shared.WindowedDeviceWaterC64Context(**context_values)
    prefix["actual_placement_transfer_after_placement"] = dataclasses.asdict(read_transfer_counters())
    from maple_syrup.windowed_shared_bed import WindowedSharedBedState
    shared_state = WindowedSharedBedState(column, state.active_layer, state.water, state.ledger,
        state.committed_topography, state.sediment_availability, None)
    from maple.aeolian.trajectories.drag import TRAJECTORY_ALGORITHM_VERSION
    case_record = dict(schema="masy.imported_device_c64_case.v1",
        windowed_identity_sha256=windowed.identity_sha256, water_identity_sha256=prepared_host_water.identity_sha256,
        case_request_binding_sha256=_sha(binding), calibration=calibration, label=label,
        complete_provenance="bound HOST owner/request; not a synthetic factory",
        source_identity=qualification_budget.native_source_identity)
    admission = _CaseAdmission(_ADMISSION_TOKEN, _sha(binding), shared_state, context,
        qualification_budget, _bytes(case_record, 65536))
    bed = DeviceWindowedC64BedOwner.from_device_bed(shared_state, context,
        case_record=case_record, time_s=windowed.time_s, limits=bed_limits,
        vegetation=vegetation, threshold_modifier_state=threshold, trajectory_semantics=TRAJECTORY_ALGORITHM_VERSION,
        _case_admission=admission)
    profile = DeviceWindowedC64WindProfile(windowed.config, wind_config, record_limits,
        qualification_budget.native_source_identity, calibration, label)
    wind = _dense.DenseWindInputs(wind_config, device_pool)
    runner = build_windowed_device_c64_runner(bed, schedule, wind=wind, water=water, control=cuda_control,
        profile=profile, record_limits=record_limits, runner_limits=runner_limits,
        resource_limits=resource_limits, qualification_budget=qualification_budget)
    prefix["actual_placement_transfer_before"] = before
    prefix["actual_placement_transfer_after_construction"] = dataclasses.asdict(read_transfer_counters())
    prefix["case_schedule_mapping"] = grouped.as_record()
    result = PreparedDeviceC64Case(bed, schedule, wind, water, cuda_control, profile, record_limits,
        runner_limits, resource_limits, qualification_budget, runner, windowed, prepared_host_water,
        kwargs, binding, prefix, identity, _TOKEN)
    return result.verify()


@dataclasses.dataclass(frozen=True, eq=False, repr=False, slots=True)
class _CheckpointOriginValidation:
    token: Any
    ticket_key: Any
    budget: Any
    case_bytes: bytes
    request_sha256: str
    used: bool = False


def _checkpoint_source_check(origin):
    """Current genuine producer bindings; never call a foreign mapping method."""
    from importlib.machinery import ModuleSpec, SourceFileLoader
    module = sys.modules.get(__name__)
    ns = globals()
    spec, loader = ns.get("__spec__"), ns.get("__loader__")
    if (type(module) is not types.ModuleType or module.__dict__ is not ns
            or type(origin) is not str or type(ns.get("__file__")) is not str or ns["__file__"] != origin
            or type(spec) is not ModuleSpec or type(spec.origin) is not str or spec.origin != origin
            or type(loader) is not SourceFileLoader or type(loader.path) is not str or loader.path != origin):
        raise CaseDeviceC64Error("checkpoint_source", "genuine loaded producer origin required")
    rows = ns.get("_SOURCE_BINDINGS")
    if type(rows) is not tuple or not 0 < len(rows) <= 8192:
        raise CaseDeviceC64Error("checkpoint_source", "closed producer binding image required")
    for row in rows:
        if type(row) is not tuple or len(row) != 4:
            raise CaseDeviceC64Error("checkpoint_source", "closed producer binding row required")
        owner, name, descriptor, code = row
        if (type(owner) not in (dict, types.MappingProxyType)
                or type(owner) is dict and owner is not ns
                or type(name) is not str or type(code) is not types.CodeType):
            raise CaseDeviceC64Error("checkpoint_source", "actual producer mapping/code required")
        current = dict.get(owner, name) if type(owner) is dict else owner[name]
        function = (current.__func__ if type(current) in (classmethod, staticmethod)
                    else current.fget if type(current) is property else current)
        if (current is not descriptor or type(function) is not types.FunctionType
                or function.__code__ is not code
                or code.co_filename == origin and function.__globals__ is not ns):
            raise CaseDeviceC64Error("checkpoint_source", "producer function/code/global changed")
    data = ns.get("_CHECKPOINT_DATA_BINDINGS")
    names = ("_TOKEN", "_ADMISSION_TOKEN", "_CHECKPOINT_VALIDATION_TOKEN", "_CHECKPOINT_ORIGIN_SCHEMA",
             "_CheckpointOriginValidation", "_CaseAdmission", "PreparedDeviceC64Case", "_SOURCE_BINDINGS")
    if (type(data) is not tuple or len(data) != len(names)
            or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not str for row in data)
            or tuple(row[0] for row in data) != names):
        raise CaseDeviceC64Error("checkpoint_source", "closed producer data image required")
    for name, value in data:
        if dict.get(ns, name) is not value:
            raise CaseDeviceC64Error("checkpoint_source", "producer private capability binding changed")
    return _source_frontier()


def _checkpoint_pack_record(raw, maximum):
    """An exact raw byte record represented by existing bounded scalar tags."""
    if type(maximum) is not int or not 0 < maximum <= 4 << 20:
        raise CaseDeviceC64Error("checkpoint_origin", "original metadata ceiling required")
    if type(raw) is not bytes or not 0 < len(raw) <= maximum:
        raise CaseDeviceC64Error("checkpoint_origin", "bounded original raw bytes required")
    chunks = tuple(raw[start:start+2048].hex() for start in range(0, len(raw), 2048))
    return dict(size=len(raw), sha256=_sha(raw), hex_chunks=chunks)


def _checkpoint_unpack_record(record, maximum):
    if type(maximum) is not int or not 0 < maximum <= 4 << 20:
        raise CaseDeviceC64Error("checkpoint_origin", "unchanged metadata ceiling required")
    if (type(record) is not dict or set(record) != {"size", "sha256", "hex_chunks"}
            or type(record["size"]) is not int or not 0 < record["size"] <= maximum
            or type(record["sha256"]) is not str or len(record["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in record["sha256"])
            or type(record["hex_chunks"]) is not tuple):
        raise CaseDeviceC64Error("checkpoint_origin", "closed raw-record entry required")
    size, chunks = record["size"], record["hex_chunks"]
    if len(chunks) != (size+2047)//2048:
        raise CaseDeviceC64Error("checkpoint_origin", "exact raw-record chunk count required")
    for index, chunk in enumerate(chunks):
        expected = 4096 if index+1 < len(chunks) else 2*(size-2048*index)
        if (type(chunk) is not str or len(chunk) != expected
                or any(c not in "0123456789abcdef" for c in chunk)):
            raise CaseDeviceC64Error("checkpoint_origin", "canonical bounded raw-record hex required")
    raw = bytes.fromhex("".join(chunks))
    if len(raw) != size or _sha(raw) != record["sha256"]:
        raise CaseDeviceC64Error("checkpoint_origin", "original raw-record digest differs")
    if _checkpoint_pack_record(raw, maximum) != record:
        raise CaseDeviceC64Error("checkpoint_origin", "raw record failed exact round trip")
    return raw


def _checkpoint_origin_charge(origin, maximum):
    """Separate raw/hex/parser occurrence frontier, never a preparation plan.

    A JSON lexical byte can introduce at most one value/key/container. Count
    each such occurrence with the existing32-byte logical header, two8-byte
    references, an8-byte scalar word and four-byte text. Raw, joined/repacked
    hex, parsed text and re-encoding buffers coexist in each current/next/
    first-error bank. The existing metadata term additionally retains derived
    comparison DOMs and paired encodings. These are positive Source words,
    not Python/provider allocator sizes.
    """
    if type(maximum) is not int or not 0 < maximum <= 4 << 20:
        raise CaseDeviceC64Error("checkpoint_origin", "unchanged metadata ceiling required")
    if (type(origin) is not dict or tuple(origin) != ("schema", "request", "host", "water")
            or origin["schema"] != _CHECKPOINT_ORIGIN_SCHEMA):
        raise CaseDeviceC64Error("checkpoint_origin", "exact imported raw capsule required")
    size = chunks = 0
    for name in ("request", "host", "water"):
        entry = origin[name]
        if (type(entry) is not dict or set(entry) != {"size", "sha256", "hex_chunks"}
                or type(entry["size"]) is not int or not 0 < entry["size"] <= maximum
                or type(entry["hex_chunks"]) is not tuple
                or len(entry["hex_chunks"]) != (entry["size"]+2047)//2048):
            raise CaseDeviceC64Error("checkpoint_origin", "bounded raw capsule sizes required")
        size += entry["size"]
        chunks += len(entry["hex_chunks"])
    # The complete encoded origin, not just individual strings, shares the
    # existing metadata cap with all other original typed inputs.
    if 2*size > maximum:
        raise CaseDeviceC64Error("checkpoint_origin", "hex raw capsule exceeds unchanged metadata cap")
    return _checkpoint_raw_words(size, chunks) + _metadata_charge(maximum)


def _checkpoint_raw_words(size, chunks):
    # Origin root+three record dictionaries, their three chunk tuples and
    # three raw byte objects; validation6/admission7 slots, three new ticket
    # references/key, canonical case bytes (the existing65536 case ceiling),
    # and the64-character request digest remain positive per occurrence.
    # Besides joined hex2S and decoded text4S, exact repack retains another2S
    # of hex; canonical re-encoding retains bytearray+final bytes up to2S.
    fixed = 32*(1+3+3+3+1+1+1) + 8*(4+3*3+6+7+3) + 65536 + 4*64
    return 3*(size + 2*size + 2*size + 4*size + 2*size
              + (32+2*8+8)*size + 8*chunks + fixed)


def _checkpoint_restore_charge(maximum):
    """Prospective capsule/parser banks BEFORE any imported reference IO."""
    if type(maximum) is not int or not 0 < maximum <= 4 << 20:
        raise CaseDeviceC64Error("checkpoint_origin", "unchanged metadata ceiling required")
    # Hex text alone is2*raw; the complete original envelope shares maximum.
    # Sum of three ceil(size/2048) counts is <= ceil(total/2048)+2.
    size = maximum//2
    chunks = (size+2047)//2048 + 3
    return _checkpoint_raw_words(size, chunks) + _metadata_charge(maximum)


def _checkpoint_retained(prepared):
    """Actual create caller roots, retained through this same-process handle.

    Do not use this function on recovery: no initial HOST forest is created
    there. It rederives the genuine initial plan rather than trusting a caller
    mutation of prepared._prefix. Original accepted/mirror/static resources
    are additional positive payload occurrences on both sides of admission.
    """
    from .dense_device_resources import DeviceDenseResources
    from maple.core.backend.arrays import array_leaf_inventory
    if (type(prepared) is not PreparedDeviceC64Case or prepared._token is not _TOKEN
            or prepared._failed is not False or prepared._checkpoint_handed_off is not False
            or type(prepared._host) is not CaseWindowedInputs
            or type(prepared._prepared_host_water) is not PreparedCaseWindowedWater
            or type(prepared.bed) is not DeviceWindowedC64BedOwner
            or any(type(raw) is not bytes for raw in (prepared._binding_bytes,
                prepared._host._record_bytes, prepared._prepared_host_water._record_bytes))):
        raise CaseDeviceC64Error("checkpoint_handoff", "unconsumed genuine imported case required")
    runner = prepared._runner
    from .windowed_device_c64_runner import DeviceWindowedC64Runner
    if type(runner) is not DeviceWindowedC64Runner or runner._bed is not prepared.bed:
        raise CaseDeviceC64Error("checkpoint_handoff", "one actual runner/bed required")
    resources = runner._device_resources
    if type(resources) is not DeviceDenseResources:
        raise CaseDeviceC64Error("checkpoint_handoff", "actual resource owner required")
    placement = prepared._request_kwargs["placement_limits"]
    prefix = _plan(prepared._host, prepared._prepared_host_water, None,
        prepared.qualification_budget, len(prepared._binding_bytes), placement.max_metadata_bytes)
    bed, private = prepared.bed._accepted, resources._accepted
    leaves = array_leaf_inventory((bed.state, bed.mirror, private.pool, private.pool_mirror,
        private.soil, private.soil_mirror, resources._statics, resources._static_mirrors,
        runner._host_config, runner._wind_profile, runner._wind, runner._water,
        prepared.bed._context, prepared.bed._vegetation, prepared.bed._threshold))
    visible = leaves["transfer_bytes"]
    raw = sum(len(value) for value in (prepared._binding_bytes,
        prepared._host._record_bytes, prepared._prepared_host_water._record_bytes))
    return dict(host_bytes=prefix["known_host_bytes"]+visible+raw,
        device_bytes=prefix["known_device_bytes"]+visible)


def _capture_checkpoint_origin(prepared, budget, maximum):
    if (type(prepared) is not PreparedDeviceC64Case
            or prepared.qualification_budget is not budget):
        raise CaseDeviceC64Error("checkpoint_handoff", "this exact genuine case budget required")
    prepared.verify()
    host, water = prepared._host, prepared._prepared_host_water
    request, host_raw, water_raw = prepared._binding_bytes, host._record_bytes, water._record_bytes
    if (type(request) is not bytes or type(host_raw) is not bytes or type(water_raw) is not bytes
            or _sha(request) != budget.case_request_binding_sha256
            or _sha(host_raw) != host.identity_sha256 or _sha(water_raw) != water.identity_sha256):
        raise CaseDeviceC64Error("checkpoint_origin", "actual owning raw identities required")
    origin = dict(schema=_CHECKPOINT_ORIGIN_SCHEMA,
        request=_checkpoint_pack_record(request, maximum), host=_checkpoint_pack_record(host_raw, maximum),
        water=_checkpoint_pack_record(water_raw, maximum))
    _checkpoint_origin_charge(origin, maximum)
    prepared.verify()
    if (prepared._host is not host or prepared._prepared_host_water is not water
            or prepared._binding_bytes != request or host._record_bytes != host_raw or water._record_bytes != water_raw):
        raise CaseDeviceC64Error("checkpoint_origin", "owning raw inputs changed during capture")
    # Distinct from scientific/input failure; no constructor/event may reuse
    # this public stepping capability after any store construction attempt.
    prepared._checkpoint_handed_off = True
    return origin


def _checkpoint_json(raw, maximum):
    def pairs(rows):
        result = {}
        for name, value in rows:
            if type(name) is not str or name in result:
                raise CaseDeviceC64Error("checkpoint_origin", "unique primitive JSON keys required")
            result[name] = value
        return result
    def integer(text):
        digits = text[1:] if text.startswith("-") else text
        if len(digits) > 19:
            raise CaseDeviceC64Error("checkpoint_origin", "owning signed63 JSON integer required")
        value = int(text)
        if not -(1 << 63) < value < (1 << 63):
            raise CaseDeviceC64Error("checkpoint_origin", "owning signed63 JSON integer required")
        return value
    def constant(_text):
        raise CaseDeviceC64Error("checkpoint_origin", "finite canonical JSON required")
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_int=integer, parse_constant=constant)
        if type(value) is not dict or _bytes(value, maximum) != raw:
            raise CaseDeviceC64Error("checkpoint_origin", "exact owning canonical JSON bytes required")
        return value
    except (ValueError, TypeError, RecursionError, OverflowError) as exc:
        if type(exc) is CaseDeviceC64Error:
            raise
        raise CaseDeviceC64Error("checkpoint_origin", "invalid bounded original JSON") from exc


def _validate_checkpoint_origin(ticket, origin, inputs, genesis_time):
    """C5 joins on admitted HOST views, BEFORE the Native snapshot loader."""
    from . import windowed_device_c64_checkpoint as checkpoint
    checkpoint._checked_ticket(ticket)
    authority = ticket.authority
    if (authority.case_mode is not True or ticket.case_validation is not None
            or ticket.case_genesis_time != genesis_time):
        raise CaseDeviceC64Error("checkpoint_validation", "one actual imported ticket/epoch required")
    maximum = ticket.limits.max_metadata_bytes
    _checkpoint_origin_charge(origin, maximum)
    raws = {name: _checkpoint_unpack_record(origin[name], maximum) for name in ("request", "host", "water")}
    request, host, water = (_checkpoint_json(raws[name], maximum) for name in ("request", "host", "water"))
    case = inputs["case"]
    closed = ((request, ("schema", "windowed_identity_sha256", "water_identity_sha256", "native_case_identity_sha256",
        "authored_file_pins", "storage_choices", "native_committed_regrouping", "host_model_sha256", "case_schedule",
        "projected_schedule", "cuda_control", "backend_changes", "derived_cuda", "record_limits", "bridge_limits",
        "bed_limits", "resource_limits", "runner_limits", "placement_limits", "metadata_projection_logical_bytes",
        "calibration", "label", "initial_time_hex")),
        (host, ("schema", "native_case_identity_sha256", "case_directory", "file_pins", "input_limits",
        "complete_initial_provenance", "held", "native_committed_regrouping", "limits", "construction_plan",
        "storage_choices", "ledger_reconciliation_baseline", "initial_mobile_state", "backend", "event_eligibility",
        "device_placement", "caller_dense_input_may_remain_owned", "whole_process_fit_or_release_claim", "known_plan_excludes")),
        (water, ("schema", "windowed_case_sha256", "native_inputs", "event_eligibility", "device_placement")),
        (case, ("schema", "windowed_identity_sha256", "water_identity_sha256", "case_request_binding_sha256",
        "calibration", "label", "complete_provenance", "source_identity")))
    if any(type(value) is not dict or set(value) != set(names) for value, names in closed):
        raise CaseDeviceC64Error("checkpoint_origin", "complete closed original record fields required")
    if (type(case) is not dict or case.get("schema") != "masy.imported_device_c64_case.v1"
            or request.get("schema") != _BINDING_SCHEMA
            or host.get("schema") != "masy.case_windowed_inputs.v1"
            or water.get("schema") != "masy.case_windowed_water_inputs.v1"):
        raise CaseDeviceC64Error("checkpoint_origin", "exact imported origin schemas required")
    def same(left, right, name):
        if _bytes(left, maximum) != _bytes(right, maximum):
            raise CaseDeviceC64Error("checkpoint_origin_join", name)
    digest = _sha(raws["request"])
    same(digest, authority.budget.case_request_binding_sha256, "fresh budget request")
    same(digest, case.get("case_request_binding_sha256"), "case request")
    same(case["source_identity"], authority.budget.native_source_identity, "case Native Source")
    for name, field in (("host", "windowed_identity_sha256"), ("water", "water_identity_sha256")):
        same(_sha(raws[name]), request.get(field), field)
        same(request.get(field), case.get(field), "case "+field)
    same(water.get("windowed_case_sha256"), request.get("windowed_identity_sha256"), "water HOST")
    for target, original in (("native_case_identity_sha256", "native_case_identity_sha256"),
            ("authored_file_pins", "file_pins"), ("storage_choices", "storage_choices"),
            ("native_committed_regrouping", "native_committed_regrouping")):
        same(request.get(target), host.get(original), target)
    same(request.get("projected_schedule"), inputs["schedule"], "projected schedule")
    same(request.get("cuda_control"), _canonical(inputs["control"]), "CUDA control")
    same(request.get("host_model_sha256"), _sha(_bytes(_canonical(inputs["model_config"]), maximum)), "original model")
    same(request.get("record_limits"), _canonical(inputs["record_limits"]), "Native record limits")
    for name in ("bed_limits", "resource_limits", "runner_limits"):
        same(request.get(name), dataclasses.asdict(inputs[name]), name)
    same(request.get("bridge_limits"), _bridge_image(authority.bridge_limits), "full bridge/run_dir")
    same(request["backend_changes"], list(_BACKEND_FIELDS), "only explicit backend selection")
    same(request["derived_cuda"], dict(kind="Native read-only property", host=False, device=True), "derived CUDA")
    for name in ("label", "calibration"):
        same(request.get(name), inputs[name], name)
        same(request.get(name), case.get(name), "case "+name)
    same(request.get("calibration"), "synthetic_test_fixture", "calibration domain")
    if type(request["label"]) is not str or not request["label"].strip() or len(request["label"]) > 4096:
        raise CaseDeviceC64Error("checkpoint_origin_join", "original bounded label required")
    same(request.get("initial_time_hex"), checkpoint.decode_time(genesis_time).hex(), "original epoch")
    proof = _CheckpointOriginValidation(_CHECKPOINT_VALIDATION_TOKEN, ticket.case_key,
        authority.budget, _bytes(case, 65536), digest)
    ticket.case_validation = proof
    return proof


def _mint_checkpoint_admission(ticket, state, context, budget, case_record):
    from . import windowed_device_c64_checkpoint as checkpoint
    checkpoint._checked_ticket(ticket)
    proof = ticket.case_validation
    shared = checkpoint._shared()
    if (ticket.authority.case_mode is not True or ticket.authority.budget is not budget
            or type(state) is not shared.WindowedSharedBedState
            or type(context) is not shared.WindowedDeviceWaterC64Context
            or type(proof) is not _CheckpointOriginValidation
            or proof.token is not _CHECKPOINT_VALIDATION_TOKEN or proof.ticket_key is not ticket.case_key
            or proof.budget is not budget or proof.used is not False
            or proof.request_sha256 != budget.case_request_binding_sha256
            or _bytes(ticket.inputs["case"], 65536) != proof.case_bytes
            or _bytes(case_record, 65536) != proof.case_bytes
            or context.runtime_bounds is not budget):
        raise CaseDeviceC64Error("checkpoint_admission", "one validated live ticket construction required")
    object.__setattr__(proof, "used", True)  # Failed construction cannot remint.
    return _CaseAdmission(_ADMISSION_TOKEN, proof.request_sha256, state, context, budget, proof.case_bytes)


def create_device_c64_case_boundary_store(path, *, prepared_case, qualification_budget,
        bridge_limits, native_source_refs, limits=None, storage_contract):
    """Handoff an imported genesis to the accepted-boundary atomic store API.

    Root owns one-use/fresh-role authority and the original complete controls.
    This does not add a public resume command or a restart qualification claim.
    """
    if type(prepared_case) is not PreparedDeviceC64Case:
        raise CaseDeviceC64Error("checkpoint_handoff", "exact genuine prepared imported case required")
    from . import windowed_device_c64_checkpoint as checkpoint
    return checkpoint._create_imported_boundary_store(path, prepared_case=prepared_case,
        qualification_budget=qualification_budget, bridge_limits=bridge_limits,
        native_source_refs=native_source_refs,
        limits=checkpoint._DEFAULT if limits is None else limits, storage_contract=storage_contract)


def recover_device_c64_case_boundary_store(path, *, expected_store_file_sha256,
        qualification_budget, bridge_limits, native_source_refs, limits=None):
    """Recover an imported accepted boundary without compile/adopt/genesis."""
    from . import windowed_device_c64_checkpoint as checkpoint
    return checkpoint._recover_imported_boundary_store(path,
        expected_store_file_sha256=expected_store_file_sha256, qualification_budget=qualification_budget,
        bridge_limits=bridge_limits, native_source_refs=native_source_refs,
        limits=checkpoint._DEFAULT if limits is None else limits)


_SOURCE_BINDINGS = _capture_source_bindings(globals())
_CHECKPOINT_DATA_BINDINGS = tuple((name, globals()[name]) for name in (
    "_TOKEN", "_ADMISSION_TOKEN", "_CHECKPOINT_VALIDATION_TOKEN", "_CHECKPOINT_ORIGIN_SCHEMA",
    "_CheckpointOriginValidation", "_CaseAdmission", "PreparedDeviceC64Case", "_SOURCE_BINDINGS"))

__all__ = ["CaseDeviceC64Error", "CaseDeviceC64PlacementLimits", "PreparedDeviceC64Case",
           "device_c64_request_binding", "prepare_device_c64_case"]
