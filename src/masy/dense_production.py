"""Opt-in, OS-capped small CPU interface helpers; no transport or limit mutation.

The OS address-space ceiling covers the whole dedicated worker. Known wrapper owners
are diagnostics and a necessary headroom check, never a complete native peak estimate.
This is not production calibration, arbitrary-size admission, GPU or restart support.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
from pathlib import Path
from typing import Any

SCOPE = "dedicated_os_capped_small_cpu_interface_only"
THREAD_ENV = {
    "MALLOC_ARENA_MAX": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
}


class DenseProductionError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclasses.dataclass(frozen=True)
class DenseProductionLimits:
    """Already-enforced worker policy. Values may tighten the accepted test ceiling."""
    address_space_bytes: int = 4 << 30
    cpu_soft_s: int = 120
    cpu_hard_s: int = 135
    max_cells: int = 64
    max_classes: int = 1
    max_voxels: int = 16

    def __post_init__(self) -> None:
        ceilings = (4 << 30, 120, 135, 64, 1, 16)
        for field, ceiling in zip(dataclasses.fields(self), ceilings, strict=True):
            value = getattr(self, field.name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise DenseProductionError("invalid_production_limit", f"{field.name} must be a plain int in [1,{ceiling}]")
        if self.cpu_hard_s < self.cpu_soft_s:
            raise DenseProductionError("invalid_production_limit", "CPU hard limit must be >= soft limit")

    def as_record(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "core_soft_hard_bytes": [0, 0],
                "file_size_soft_hard_bytes": [128 << 20] * 2, "scope": SCOPE,
                "native_peak_estimate": None,
                "native_peak_estimate_missing": ["flat_transport", "FFT_and_separation_scratch",
                                                   "avalanche_and_commit_scratch", "native_output_lifetimes"]}


def process_identity() -> dict[str, Any]:
    try:
        # Linux stat field22: field3 is index0 after the LAST ')' of comm.
        stat = Path("/proc/self/stat").read_text()
        start = int(stat[stat.rfind(")") + 1:].split()[19])
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if not boot or start <= 0:
            raise ValueError("empty boot id or invalid start ticks")
        return {"pid": os.getpid(), "start_ticks": start, "boot_id": boot}
    except (OSError, ValueError, IndexError) as exc:
        raise DenseProductionError("worker_identity_unavailable", str(exc)) from exc


def check_process(limits: DenseProductionLimits, expected_identity: dict[str, Any] | None = None,
                  *, upcoming_wrapper_bytes: int = 0, checkpoint: str = "unspecified") -> dict[str, Any]:
    """Read-only Linux guard; also called immediately before the one publication."""
    if type(limits) is not DenseProductionLimits:
        raise DenseProductionError("invalid_production_limit", "production_limits must be DenseProductionLimits")
    limits.__post_init__()  # revalidate live values, including forced mutation of the frozen declaration
    if sys.platform != "linux":
        raise DenseProductionError("unsupported_resource_platform", "dedicated Linux RLIMIT_AS worker required")
    import resource

    try:
        observed = {name: list(resource.getrlimit(getattr(resource, name)))
                    for name in ("RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_CORE", "RLIMIT_FSIZE")}
    except (AttributeError, OSError, ValueError) as exc:
        raise DenseProductionError("resource_limit_unavailable", str(exc)) from exc
    wanted = {"RLIMIT_AS": [limits.address_space_bytes] * 2,
              "RLIMIT_CPU": [limits.cpu_soft_s, limits.cpu_hard_s], "RLIMIT_CORE": [0, 0],
              "RLIMIT_FSIZE": [128 << 20] * 2}
    if observed != wanted:
        raise DenseProductionError("resource_limit_mismatch", f"required already enforced {wanted}; observed {observed}")
    identity = process_identity()
    if expected_identity is not None and identity != expected_identity:
        raise DenseProductionError("worker_identity_changed", "PID/start/boot identity differs from construction")
    environment = {key: os.environ.get(key) for key in THREAD_ENV}
    if environment != THREAD_ENV:
        raise DenseProductionError("worker_environment_mismatch", "dedicated CPU/thread/arena environment is not the declared policy")
    try:
        virtual_bytes = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError) as exc:
        raise DenseProductionError("worker_virtual_size_unavailable", str(exc)) from exc
    if type(upcoming_wrapper_bytes) is not int or upcoming_wrapper_bytes < 0:
        raise DenseProductionError("invalid_wrapper_headroom", "upcoming wrapper byte count must be a nonnegative int")
    if virtual_bytes >= limits.address_space_bytes or upcoming_wrapper_bytes > limits.address_space_bytes - virtual_bytes:
        raise DenseProductionError("address_space_headroom", "current virtual mappings and known wrapper work cannot fit the AS ceiling")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"scope": SCOPE, "checkpoint": checkpoint,
            "process_identity": identity, "limits": observed, "environment": environment,
            "virtual_size_bytes": virtual_bytes, "known_upcoming_wrapper_bytes": upcoming_wrapper_bytes,
            "virtual_headroom_bytes": limits.address_space_bytes - virtual_bytes,
            "self_cpu_s": float(usage.ru_utime + usage.ru_stime),
            "peak_rss_native_units": int(usage.ru_maxrss), "native_peak_estimate": None}


def schedule_envelope(schedule: Any) -> tuple[Any, dict[str, Any]]:
    from maple.core.parameters.wind_forcing import resolve_wind_forcing_record

    from masy.dense_schedule import WindEvent
    events = [(i, e) for i, e in enumerate(schedule.events) if type(e) is WindEvent]
    if not events:
        raise DenseProductionError("wind_envelope_empty", "the actual schedule has no wind event")
    try:
        record = resolve_wind_forcing_record({"blocks": [
            {"shear_velocity_m_s": e.shear_velocity_m_s, "direction_deg": e.direction_deg_cw_from_north,
             "duration_s": e.duration_s} for _, e in events]})
    except (ValueError, TypeError) as exc:
        raise DenseProductionError("invalid_wind_envelope", str(exc)) from exc
    return record, {"role": "forcing_bound_only_not_execution_schedule", "schedule_sha256": schedule.digest,
                    "source_event_indices": [i for i, _ in events]}


def controlling_record(n: Any, bed: Any, wind: Any, schedule: Any, profile: Any,
                       limits: DenseProductionLimits) -> dict[str, Any]:
    """Full native rebuild/audit and exact shared context checks; no hand-written mapper."""
    from maple_syrup.sediment_bed import reroute_threshold_m

    from masy.dense_bed import DenseBedError, _require_host_arrays
    from masy.dense_wind_profile import (
        ProductionProfileError,
        audit_dense_production_wind_config,
    )
    ctx, cfg = bed.context, wind.config
    if type(cfg) is not n.SchedulerPhysicsConfig:
        raise DenseProductionError("invalid_wind_config", "exact native SchedulerPhysicsConfig required")
    for name in ("geometry", "grain_classes", "avalanching_spec", "topographic_commit_spec", "initial_available_fraction"):
        if getattr(cfg, name) is not getattr(ctx, name):
            raise DenseProductionError("incompatible_wind_config", f"{name} is not the shared context object")
    if float(cfg.mass_resolution_kg).hex() != float(ctx.mass_resolution_kg).hex():
        raise DenseProductionError("incompatible_wind_config", "mass resolution differs from the shared bed")
    g = ctx.geometry
    dimensions = {"cells": int(g.ny * g.nx), "classes": len(ctx.grain_classes.classes),
                  "voxels": bed.case_record["n_voxels"]}
    if any(dimensions[name] > maximum for name, maximum in
           (("cells", limits.max_cells), ("classes", limits.max_classes), ("voxels", limits.max_voxels))):
        raise DenseProductionError("production_dimension_limit", f"case dimensions {dimensions} exceed the bounded CPU policy")
    try:
        envelope, binding = schedule_envelope(schedule)
        audit = audit_dense_production_wind_config(profile, cfg, wind_envelope_record=envelope,
                                                  initial_available_fraction=ctx.initial_available_fraction)
    except ProductionProfileError as exc:
        raise DenseProductionError(exc.code, exc.detail) from exc
    for scheduler_name, model_name in (("geometry", "geometry"), ("grain_classes", "grain_classes"),
            ("avalanching_spec", "avalanching"), ("topographic_commit_spec", "topographic_commit")):
        if getattr(cfg, scheduler_name) is not getattr(profile.model_config, model_name):
            raise DenseProductionError("incompatible_wind_config", f"{scheduler_name} is not the authored model object")
    model_water = profile.model_config.water_coupling
    if (model_water.enabled is not False or cfg.water_coupling_spec.enabled is not False
            or model_water.depth_update_rule != "constant_depth"
            or cfg.water_coupling_spec.depth_update_rule != "constant_depth" or ctx.depth_update_rule != "constant_depth"):
        raise DenseProductionError("incompatible_water_rule", "explicit disabled wind water transport and shared constant_depth required")
    if cfg.particle_transport.mobile_state_representation != "flat":
        raise DenseProductionError("unsupported_mobile_state", "only native flat cohorts are qualified")
    pool = wind.initial_mobile_state
    if type(pool) is not n.MobileCohortState:
        raise DenseProductionError("unsupported_mobile_state", "exact native flat MobileCohortState required")
    try:
        _require_host_arrays(pool, "initial_mobile_state")
    except DenseBedError as exc:
        raise DenseProductionError(exc.code, exc.detail) from exc
    if not n.mobile_state_is_genuinely_empty(pool):
        raise DenseProductionError("mobile_state_not_empty", "the initial native wind pool must be genuinely empty")
    outer = cfg.outer_timestep_s
    if type(outer) is not float or not math.isfinite(outer) or outer <= 0:
        raise DenseProductionError("invalid_wind_config", "finite positive float outer_timestep_s required")
    return {"scope": SCOPE, "production_calibration": "pending_not_qualified_by_interface_tests",
            "limits": limits.as_record(), "dimensions": dimensions, "water_rule": "constant_depth",
            "hydraulic_routing_policy": {"native_reader": "maple_syrup.sediment_bed.reroute_threshold_m",
                                         "threshold_m_hex": float(reroute_threshold_m(ctx)).hex(),
                                         "decision": "actual_native_commit_bed_only"},
            "schedule_binding": binding, "native_config_audit": audit}


def require_dry_wind_water(np: Any, water: Any, *, previous: Any = None) -> None:
    if bool(np.any(water.depth_m != 0.0)) or bool(np.any(water.mobile_mass_by_cell_class_kg != 0.0)):
        raise DenseProductionError("wind_water_not_dry", "wind requires and must preserve native dry surface water and empty load")
    if previous is not None and (not np.array_equal(water.depth_m, previous.depth_m)
                                or not np.array_equal(water.mobile_mass_by_cell_class_kg,
                                                      previous.mobile_mass_by_cell_class_kg)):
        raise DenseProductionError("wind_water_changed", "native wind changed accepted dry water state")


def retained_wrapper_owners(bed: Any, wind: Any, water: Any, control: Any, profile: Any,
                           pool: Any, soil: Any, routing: Any, n_receipts: int) -> dict[str, Any]:
    """Reachable array payloads only; owner categories can alias. Never native peak admission."""
    import numpy as np

    def payload_bytes(*roots: Any) -> int:
        # Only real arrays count. NumPy's namespace also has shape/dtype names,
        # so the old host-array validator's broad array-like scan cannot be used
        # on configuration trees containing an xp module.
        seen: set[int] = set()
        def visit(value: Any) -> int:
            if id(value) in seen:
                return 0
            seen.add(id(value))
            if type(value) is np.ndarray:
                return int(value.nbytes)
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                return sum(visit(getattr(value, field.name)) for field in dataclasses.fields(value))
            if isinstance(value, (tuple, list)):
                return sum(visit(item) for item in value)
            if isinstance(value, dict):
                return sum(visit(item) for item in value.values())
            return 0
        return sum(visit(root) for root in roots)

    return {"scope": "retained_array_payload_diagnostics_only_categories_may_alias",
            "accepted_bed": payload_bytes(bed.state, bed.vegetation, bed.threshold_modifier_state),
            "held_case_inputs": payload_bytes(bed._held_inputs),
            "static_native_inputs": payload_bytes(wind.config, water, control, profile.model_config),
            "accepted_mobile_pool": payload_bytes(pool), "accepted_soil": payload_bytes(soil),
            "retained_water_routing_and_template": 0 if routing is None else payload_bytes(vars(routing)),
            "accepted_receipt_count": n_receipts,
            "python_objects_headers_allocator_and_native_scratch": "not_estimated_OS_ceiling_applies"}


def commit_evidence(np: Any, commit: Any) -> dict[str, Any]:
    if commit is None:
        return {"status": "no_commit", "scope": "latest_native_commit_only"}
    diag = commit.avalanche_diagnostics
    names = ("source_removed_by_class_kg", "destination_added_by_class_kg", "gross_moved_by_class_kg",
             "global_net_by_class_kg", "gross_physical_activity_by_class_kg")
    return {"status": "committed", "scope": "latest_native_commit_only", "avalanche_applied": bool(commit.avalanche_applied),
            "avalanche_diagnostics": {name: None if diag is None or name not in diag else
                                       [float(v) for v in np.asarray(diag[name]).reshape(-1)] for name in names}}


def wind_evidence(n: Any, result: Any, final: Any, previous_water: Any) -> dict[str, Any]:
    require_dry_wind_water(n.np, result.state.physical.water, previous=previous_water)
    require_dry_wind_water(n.np, final.state.physical.water, previous=previous_water)
    for name in ("water_clamp_added_volume_m3", "water_displaced_volume_m3"):
        value = getattr(result.accounting, name)
        if value is not None and float(value) != 0.0:
            raise DenseProductionError("wind_water_exchange", f"native wind reported nonzero {name}")
    physical = final.state.physical
    if physical.ledger_reconciliation_baseline is not None:
        raise DenseProductionError("unsupported_reconciliation_baseline", "native wind returned a reconciliation baseline without a shared-bed owner")
    if final.state.topographic_wind_cache is not None or final.state.active_layer_composition_cache is not None:
        raise DenseProductionError("wind_cache_not_invalidated", "native finalizer did not invalidate topographic/composition caches")
    cache = result.state.topographic_wind_cache
    modifier = None if cache is None else cache.modifier
    topography = {"status": "absent"} if modifier is None else {
        "status": "native_modifier", "algorithm_version": modifier.algorithm_version,
        "wind_direction_deg": float(modifier.wind_direction_deg),
        "separation_present": modifier.separation is not None,
        "separation_bubble_cells": (None if modifier.separation is None else
                                     int(n.np.count_nonzero(modifier.separation.bubble_mask))),
        "envelope_role": "wind_diagnostic_never_bed_input"}
    return {"scope": SCOPE, "water_rule": "constant_depth", "dry_water_preserved": True,
            "water_volume_exchanges_m3": {name: None if getattr(result.accounting, name) is None else
                                           float(getattr(result.accounting, name)) for name in
                                           ("water_clamp_added_volume_m3", "water_displaced_volume_m3")},
            "topographic_wind": topography,
            "transport_launched_mass_by_class_kg": [float(v) for v in result.accounting.launched_mass_by_class_kg],
            "latest_wind_commit": commit_evidence(n.np, result.last_scheduler_commit_result),
            "creep_evidence": "signed_domain_net_is_not_a_gross_activity_witness"}
