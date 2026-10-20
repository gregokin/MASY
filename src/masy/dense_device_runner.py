"""Private explicit Dense GPU binding of the ONE existing `DenseRunner` engine.

Not exported, not imported by `dense_runner`, any CLI or default path. The only entry is the private factory
`_create_dense_device_runner`; it is unused by default and holds no run/source/allocation authority (a later
protected GPU worker owns that). It builds a NEW run from an actual validated CPU `DenseBed` plus
`DenseWindInputs` / `DenseWaterInputs` / `DenseRunnerControl`:

1. exact first-profile admission: the unchanged strict default Dense physics (the CPU `_check_wind` guards, which
   refuse every enabled unsupported feature incl. topographic FFT), six grain classes, SYRUP `upwind` with the
   native `cuda` hydraulic implementation; no production wind profile; genuine already-imported CuPy only;
2. host source identity (case inputs, bed identity, controlling static record, initial pool/soil) BEFORE;
3. the exact native `cuda_memory_preflight` over a named case view of the HOST source bed, plus the device-bed
   owner six-root reservation and the private pool/soil/static reservation, against the native free bytes and an
   explicit cap (native preflight does NOT estimate wind cohorts or FFT; FFT is refused here);
4. ONE native `to_device_tree` conversion per source object, adopted by `DeviceDenseBedOwner` (six bed holders)
   and `DeviceDenseResources` (pool, soil, statics), which clone into private accepted+mirror pairs;
5. host source identity AFTER (the CPU source must be unchanged by the declared lossless copy).

The CPU physical bed is not retained (only its host `BedContext`, case metadata and the host static inputs that
the CPU engine also re-verifies). Steps run through the SAME `DenseRunner.step/_wind_event/_rain_event/_finish`
code; this module only supplies private ownership / array-op / rain-lifecycle hook bodies, including the
engine's three closed retained-routing calls over the device namespace (restored CPU projections stay CPU-only).
GPU labels are verified-mirror / generation / host-metadata labels, never bed-content SHAs. Not qualified: GPU
transport, CPU-GPU event comparison (not run; bounds policy in `device_evidence`), memory fit, windowed/dynamic
vegetation/restart/scale (docs/dense_device_runner.md).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from masy import dense_bed as _bed
from masy import dense_runner as _runner
from masy.clock import ClockLedger, _native_clock
from masy.dense_bed import DenseBed, DenseBedError
from masy.dense_device_bed import (
    DeviceBedError,
    DeviceBedLimits,
    DeviceDenseBedOwner,
    _CupyOps,
    _work_reservation,
)
from masy.dense_device_resources import (
    STATIC_NAMES,
    DeviceDenseResources,
    DeviceResourceError,
    DeviceResourceLimits,
    resource_reservation,
)
from masy.dense_runner import (
    PROFILE,
    DenseRunner,
    DenseRunnerControl,
    DenseRunnerError,
    DenseRunnerLimits,
    DenseWaterInputs,
    DenseWindInputs,
    _Stop,
    case_schedule_digest,
)
from masy.dense_schedule import DenseSchedule

__all__: list[str] = []  # private module: nothing is exported

DEVICE_STATIC_SCHEMA = "masy.dense_device_runner_static.v1"
DEVICE_RECEIPT_SCHEMA = "masy.dense_device_runner_event.v1"
DEVICE_REPORT_SCHEMA = "masy.dense_device_runner_report.v1"
DEVICE_PROFILE = {
    **PROFILE, "backend": "cupy", "hydraulic_implementation": "cuda", "topographic_wind_fft": "refused",
    "grain_classes": 6, "scope": "first_small_dense_default_gpu_profile",
    "calibration": "unchanged_default_physics_not_gpu_qualified",
}
_N_CLASSES = 6
_CUDA_MODES = ("auto", "fused", "split")
# SedimentEventState: the exact nine native fields (a native addition is refused, never dropped).
_EVENT_FIELDS = ("storm", "bed", "sediment_velocity_m_s", "graph", "network", "grid", "terrain", "phase",
                 "routing_source_elevation_m")
# The device retained-routing namespace (`_DeviceOwnership.retain_routing`): intentionally no native template.
# `stamp` is the creating adapter's opaque per-instance token (never the owner, adapter or bed itself).
_HELD_ROUTING_FIELDS = frozenset({"graph", "terrain", "network", "grid", "source", "velocity", "phase", "generation",
                                  "stamp"})
_PREFLIGHT_KEYS = ("case_field_bytes", "coexisting_state_bytes", "maple_working_bytes", "commit_snapshot_bytes",
                   "phase_bytes", "transport_working_bytes", "hydrology_bytes", "accounting_bytes",
                   "required_bytes", "device_free_bytes")
_SUMMARY_CHARS = 4096
# Finite first-profile engineering ceilings (the synthetic six-class alternation fixture is 6x1 cells, 3 voxels,
# 16000 flat cohorts, 8 hop bins). Unsupported inputs are refused; no physics is changed or disabled.
_FIRST_PROFILE_CAPS = {"max_cells": 64, "max_voxels": 64, "n_classes": _N_CLASSES, "max_flat_cohorts": 16000,
                       "max_hop_bins": 1024, "max_wind_accounting_steps": 64, "max_pool_leaves": 64}


@dataclasses.dataclass(frozen=True)
class DeviceRunnerLimits:
    """Explicit device policy of the private factory. Structural, never physics."""

    max_device_bytes: int
    hydraulics_max_log: int
    # Declared UNMEASURED allowances, ADDED to (never standing in for) the named source-derived holdings of
    # `_workspace_mapping`: flat flux/step/routing owners beyond the per-row lower reservations (wind) and
    # routing/reroute temporaries beyond the named held graph/network/grid/source/velocity arrays (routing). A
    # later protected worker measures the peak.
    wind_workspace_bytes: int
    routing_workspace_bytes: int
    cuda_mode: str = "auto"

    def __post_init__(self) -> None:
        for name in ("max_device_bytes", "hydraulics_max_log", "wind_workspace_bytes", "routing_workspace_bytes"):
            _runner._plain_int(getattr(self, name), name)
        if self.cuda_mode not in _CUDA_MODES:
            raise DenseRunnerError("invalid_limit", f"cuda_mode must be one of {_CUDA_MODES}")

    def as_record(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _genuine_cupy() -> ModuleType:
    cp = sys.modules.get("cupy")
    if (not isinstance(cp, ModuleType) or getattr(cp, "__name__", None) != "cupy"
            or getattr(getattr(cp, "ndarray", None), "__module__", "").split(".")[0] != "cupy"):
        raise DenseRunnerError("unsupported_backend", "an already-imported genuine CuPy is required; no fallback")
    return cp


def _device_error(exc: BaseException) -> DenseBedError:
    """Owner errors cross the engine's existing `DenseBedError` refusal seams with their own code."""
    return DenseBedError(getattr(exc, "code", "device_owner_error"), getattr(exc, "detail", str(exc)))


# --- admission ---------------------------------------------------------------------------------------------------

def _check_profile_caps(n: SimpleNamespace, bed: DenseBed, wind: DenseWindInputs, limits: DenseRunnerLimits) -> None:
    """Finite first profile BEFORE any source canonicalization, hash, soil copy or preflight: plain-int metadata
    reads only, against `_FIRST_PROFILE_CAPS`."""
    from maple.core.backend import array_leaves

    caps = _FIRST_PROFILE_CAPS

    def bounded(value: Any, low: int, high: int, what: str) -> int:
        if type(value) is not int or not low <= value <= high:
            raise DenseRunnerError("unsupported_profile", f"{what}={value!r} outside the first profile [{low}, {high}]")
        return value

    g = bed.context.geometry
    ny, nx = bounded(g.ny, 1, caps["max_cells"], "geometry.ny"), bounded(g.nx, 1, caps["max_cells"], "geometry.nx")
    bounded(ny * nx, 1, caps["max_cells"], "cells")
    record = bed.case_record
    bounded(record.get("n_voxels") if type(record) is dict else None, 1, caps["max_voxels"], "n_voxels")
    classes = bed.context.grain_classes.classes
    if type(classes) is not tuple or len(classes) != caps["n_classes"]:
        raise DenseRunnerError("unsupported_profile", f"first device profile is the {caps['n_classes']}-class fixture")
    cfg = wind.config
    if type(cfg) is not n.SchedulerPhysicsConfig:
        raise DenseRunnerError("invalid_wind_config", "wind.config must be a native SchedulerPhysicsConfig")
    particle = cfg.particle_transport
    if particle.mobile_state_representation != "flat":
        raise DenseRunnerError("unsupported_profile", "only the flat mobile representation is in the first profile")
    cohorts = bounded(particle.max_active_cohorts, 1, caps["max_flat_cohorts"], "max_active_cohorts")
    bounded(particle.quantile_bins, 1, caps["max_hop_bins"], "quantile_bins")
    shape = tuple(getattr(getattr(cfg.hop_bins, "hop_distance_m", None), "shape", ()))
    if len(shape) != 2 or shape[0] != caps["n_classes"] or not 1 <= shape[1] <= caps["max_hop_bins"]:
        raise DenseRunnerError("unsupported_profile", f"hop_bins {shape} outside (6, <= {caps['max_hop_bins']})")
    bounded(limits.max_wind_accounting_steps, 1, caps["max_wind_accounting_steps"], "max_wind_accounting_steps")
    leaves = []
    for leaf in array_leaves(wind.initial_mobile_state):  # native flat pool: parallel (max_cohorts,) slot arrays
        leaves.append(leaf)
        if len(leaves) > caps["max_pool_leaves"]:
            raise DenseRunnerError("unsupported_profile", "too many wind pool leaves")
        if type(leaf) is not n.np.ndarray or leaf.ndim < 1 or leaf.shape[0] != cohorts:
            raise DenseRunnerError("unsupported_profile", "pool leaves must be host slot arrays of max_active_cohorts")


def _host_bed_admission(bed: DenseBed) -> tuple[Any, Any]:
    """Live physical schema of the CPU source bed BEFORE any host->device copy: the device owner's closed
    six-holder walk with NumPy as the expected array type (`dense_device_bed._host_bed_schema`). The device
    owner's native validators still run after conversion."""
    from masy.dense_device_bed import _host_bed_schema

    try:
        return _host_bed_schema(bed.state, geometry=bed.context.geometry, n_voxels=bed.case_record["n_voxels"],
                                n_classes=len(bed.context.grain_classes.classes))
    except DeviceBedError as exc:
        raise DenseRunnerError(exc.code, exc.detail) from exc


def _neutral_metadata(value: Any, array_types: tuple[type, ...], xp: ModuleType) -> tuple[Any, ...]:
    """Backend-neutral closed metadata (array module roots and the xp name dropped): host and device holders
    of the SAME native values must agree exactly on every type/field/scalar/enum/dtype/shape."""
    from masy.dense_device_resources import _closed_metadata

    try:
        record, _, _ = _closed_metadata(value, array_types, (xp,))
    except DeviceResourceError as exc:
        raise DenseRunnerError(exc.code, exc.detail) from exc
    return tuple((e[0], "array", *e[3:]) if e[1] == "array" else (e[0], "module") if e[1] == "module" else e
                 for e in record)


def _bind_device_inputs(np: Any, cp: ModuleType, host_wind: Any, host_water: Any, device_wind: Any,
                        device_water: Any, resources: DeviceDenseResources) -> None:
    """The shadow device wind config / water inputs cannot diverge from their controlling host sources: every
    wind config field IS the host field except the owned availability static; terrain IS the host terrain;
    column / field / sediment are the owned statics and carry exactly the host native metadata (scalars, model,
    enums, dtypes, shapes)."""
    host_cfg, cfg = host_wind.config, device_wind.config
    if type(cfg) is not type(host_cfg) or set(vars(cfg)) != set(vars(host_cfg)):
        raise DenseRunnerError("device_config_diverged", "device wind config holder/attributes differ from host")
    for field in dataclasses.fields(host_cfg):
        want = (resources.static("initial_available_fraction") if field.name == "initial_available_fraction"
                else getattr(host_cfg, field.name))
        if getattr(cfg, field.name) is not want:
            raise DenseRunnerError("device_config_diverged", f"device wind config {field.name} is not bound")
    if device_wind.initial_mobile_state is not None or device_water.initial_soil_water_m is not None:
        raise DenseRunnerError("device_config_diverged", "evolving inputs must stay owner-private")
    if device_water.terrain is not host_water.terrain:
        raise DenseRunnerError("device_config_diverged", "device terrain is not the host terrain reference")
    for name in ("column", "field", "sediment"):
        device = getattr(device_water, name)
        if device is not resources.static(name):
            raise DenseRunnerError("device_config_diverged", f"device water {name} is not the owned static")
        if (_neutral_metadata(device, (cp.ndarray, np.ndarray), cp)
                != _neutral_metadata(getattr(host_water, name), (np.ndarray,), np)):
            raise DenseRunnerError("device_config_diverged", f"device water {name} metadata differs from host")
    if device_water.vegetation_cover_fraction is not resources.static("vegetation").raster.cover_fraction:
        raise DenseRunnerError("device_config_diverged", "device vegetation cover is not the owned static")


def _workspace_mapping(n: SimpleNamespace, bed: DenseBed, wind: DenseWindInputs, soil: Any,
                       native: dict[str, Any], device_limits: DeviceRunnerLimits) -> dict[str, Any]:
    """Named finite source-derived device holdings the owner reservations and the native WATER preflight do not
    cover, plus the declared unmeasured allowances on top. MAPLE's `flat_pool_unbudgeted_bytes` is a reported
    PARTIAL flat estimate with NO preflight gate (`calendar_workspace_bytes` does not describe flat at all); it is
    charged here at the conservative reconstructed-trajectory size, not claimed as a gate or peak fit."""
    from maple.aeolian.cohorts import memory_budget as mb
    from maple.core.backend import array_leaves

    g, nc = bed.context.geometry, len(bed.context.grain_classes.classes)
    cells = g.ny * g.nx
    pool_bytes = sum(int(a.nbytes) for a in array_leaves(wind.initial_mobile_state))
    rows = int(wind.config.particle_transport.max_active_cohorts)
    flat = mb.flat_pool_unbudgeted_bytes(rows, reconstructed_splash_trajectory=True)
    # Native holder shapes traced from source (n_active <= cells; device = graph namespace):
    #   SedimentEventState.sediment_velocity_m_s float64 (ny, nx, nc)                 sediment_event.py
    #   routing_source_elevation_m = committed_elevation_m(...) float64 (ny, nx)       sediment_bed.py:367
    #   RoutingGraph runtime: conveyance f8, active_flat b1, outlet_flat b1, level_order i8, conveyance_lo f8,
    #     donor_position i8 (4, n_active), donor_mask b1 (4, n_active) -> 62 B/cell   routing.py:530-550
    #   TransportNetwork: receiver_index i8, outlet_flat b1, active_flat b1, terminal_flat b1, and per active
    #     cell exactly one of the x/y face sets (cells i8, rows|cols i8, faces i8, sign f8) -> 43 B/cell
    #                                                                                  sediment_transport.py:237
    #   PhysicsGrid: slope f8, active b1, terminal b1 -> 10 B/cell                     sediment_physics.py:497
    #   StormState entry: depth, soil, discharge f8 (ny, nx) -> 24 B/cell              storm.py:247
    velocity = cells * nc * mb.F8
    source = cells * mb.F8
    graph = cells * (2 * mb.F8 + 2 * mb.B1 + mb.I8 + 4 * (mb.I8 + mb.B1))
    network = cells * (mb.I8 + 3 * mb.B1 + 3 * mb.I8 + mb.F8)
    grid = cells * (mb.F8 + 2 * mb.B1)
    # Flat wind rows: every active slot can route one impact row per step, so `max_active_cohorts` rows is a
    # LOWER bound on the per-step row owners, charged with MAPLE's own per-row coefficients of the shared
    # face_flux / aggregation / cohorts.routing / deposition functions the flat `flux/step.py` calls. The
    # calendar estimators (`calendar_workspace_bytes`) do NOT describe flat and are not used.
    # Rank: `n_index_components=2` is a MINIMUM; the native class-indexed face/deposition scatters over
    # (ny, nx, nc)-shaped targets may carry more index components (not audited here), so these remain lower
    # terms. Flat transient rows can exceed stored cohorts; `max_active_cohorts` does not bound the row count.
    faces = g.ny * (g.nx + 1) + (g.ny + 1) * g.nx
    face_scatter = mb.scatter_owner_bytes(rows, 2, faces * nc)["peak_bytes"]
    deposit_scatter = mb.scatter_owner_bytes(4 * rows, 2, cells * nc)["peak_bytes"]
    known = {
        # wind: scheduler state shares the owner's work objects (identity-checked); the native step returns a
        # NEW candidate pool held with the work pool until publication; flat stress runs unchunked.
        "wind_native_candidate_pool_bytes": pool_bytes,
        "wind_flat_pool_native_partial_persistent_bytes": max(int(flat["persistent_bytes"]), pool_bytes),
        "wind_flat_grain_stress_native_partial_bytes": int(flat["stress_workspace_bytes"]),
        "wind_flat_routing_boundary_rows_lower_bytes": rows * mb.BOUNDARY_BYTES_PER_ROW,
        "wind_flat_routing_group_rows_lower_bytes": rows * mb.GROUP_BYTES_PER_ROW,
        "wind_flat_face_row_global_lower_bytes": rows * mb.FACE_ROW_GLOBAL_BYTES_PER_ROW,
        "wind_flat_face_grid_lower_bytes": rows * mb.FACE_GRID_BYTES_PER_ELEMENT,  # >= 1 element per row
        "wind_flat_face_scatter_peak_lower_bytes": int(face_scatter),
        "wind_flat_aggregation_rows_lower_bytes": rows * mb.AGGREGATION_BYTES_PER_ROW,
        # deposition of the step AND the finalizer's settlement (sequential: charged once at the larger owner)
        "wind_flat_deposition_rows_lower_bytes": rows * mb.DEPOSIT_PRODUCER_BYTES_PER_ROW,
        "wind_flat_deposition_scatter_peak_lower_bytes": int(deposit_scatter),
        "wind_flat_deposition_export_partial_lower_bytes": mb.pairwise_export_partial_bytes(rows, rows, nc),
        # routing: old AND new held context coexist across an event (x2)
        "routing_held_zero_velocity_bytes": 2 * velocity,
        "routing_entry_velocity_work_copy_bytes": velocity,
        "routing_native_dry_velocity_bytes": velocity,
        "routing_held_source_elevation_bytes": 2 * source,
        "routing_held_graph_runtime_bytes": 2 * graph,
        "routing_held_network_bytes": 2 * network,
        "routing_held_grid_bytes": 2 * grid,
        # may overlap the native preflight hydrology term; charged anyway (conservative)
        "rain_entry_storm_state_bytes": 3 * source,
        # finalizer / bed candidate / new entry / comparison packets: inside the bed owner's six-root and the
        # resource owner's five-copy + comparison reservations (counted there, not twice).
        "finalizer_bed_candidate_bytes": 0,
        "soil_candidate_bytes": int(soil.nbytes),
    }
    declared = {"declared_unmeasured_wind_flat_flux_step_routing_bytes": device_limits.wind_workspace_bytes,
                "declared_unmeasured_routing_temporaries_bytes": device_limits.routing_workspace_bytes}
    host_graph = cells * (2 * mb.B1 + 3 * mb.F8 + mb.B1 + mb.B1 + mb.I8 + mb.I4 + mb.I8 + mb.B1)
    return {"named_device_bytes": known, "declared_unmeasured_device_bytes": declared,
            "named_total_bytes": int(sum(known.values())), "declared_total_bytes": int(sum(declared.values())),
            "native_flat_estimate": {k: flat[k] for k in ("persistent_bytes", "stress_workspace_bytes",
                                                          "gated_by_preflight")},
            "native_coefficients": {"estimator_version": mb.ESTIMATOR_VERSION, "lower_bound_rows": rows,
                                    "scope": "partial: per-row owners of shared functions at rows = "
                                             "max_active_cohorts; crossings per row, impact-generated launches "
                                             "and expression temporaries are NOT bounded here"},
            "native_water_preflight_required_bytes": int(native["required_bytes"]),
            "host_only": {"report_summary_chars": _SUMMARY_CHARS,
                          "static_host_sediment_bytes_owned_and_mirror": 2 * 8 * nc * 8,
                          "held_graph_host_arrays_bytes_x2": 2 * host_graph,
                          "held_network_host_arrays_bytes_x2": 2 * cells * (mb.I8 + 3 * mb.B1),
                          "terrain": "the caller's host TerrainReference by reference (not copied)"},
            "scope": "named source-derived holdings (partial lower reservations where marked) + declared "
                     "unmeasured allowances; no peak fit claimed; CPU driver VM/RSS is distinct from device "
                     "bytes; a protected worker must enforce pool/resource caps and measure the first profile"}


def _check_device_water(n: SimpleNamespace, bed: DenseBed, water: DenseWaterInputs, control: DenseRunnerControl,
                        limits: DenseRunnerLimits) -> Any:
    """The CPU `_check_water` guards, except that the hydraulic implementation must be native `cuda` (the CPU
    profile requires `array`). Every other guard is identical; nothing is relaxed or disabled."""
    np, g = n.np, bed.context.geometry
    sc, policy = control.sediment_control, control.completion_policy
    if type(sc) is not n.SedimentEventControl:
        raise DenseRunnerError("invalid_water_control", "sediment_control must be a native SedimentEventControl")
    if sc.transport_scheme != "upwind":
        raise DenseRunnerError("unsupported_transport_scheme", f"{sc.transport_scheme!r}; only 'upwind'")
    if sc.storm.implementation != "cuda":
        raise DenseRunnerError("unsupported_hydraulic_implementation",
                               f"{sc.storm.implementation!r}; the device profile requires 'cuda'")
    if sc.commit is not True or sc.record_exchange_history is not False or sc.deep_publication != "step":
        raise DenseRunnerError("unsupported_water_control",
                               "requires commit=True, record_exchange_history=False, deep_publication='step'")
    if type(policy) is not n.CompletionPolicy:
        raise DenseRunnerError("invalid_water_control", "completion_policy must be a native CompletionPolicy")
    try:
        sc.validated()
        policy.validated()
    except ValueError as exc:
        raise DenseRunnerError("invalid_water_control", _runner._detail(exc)) from exc
    if sc.storm.max_steps > limits.max_storm_steps:
        raise DenseRunnerError("storm_step_limit", f"native max_steps {sc.storm.max_steps} > max_storm_steps="
                               f"{limits.max_storm_steps}")
    every = control.report_every_s
    if type(every) is not float or not math.isfinite(every) or every <= 0:
        raise DenseRunnerError("invalid_water_control", f"report_every_s {every!r} must be a finite float > 0")
    if type(water.terrain) is not n.TerrainReference:
        raise DenseRunnerError("invalid_water_inputs", "terrain must be a native TerrainReference")
    if water.terrain.dx_m != g.dx_m or water.terrain.initial_committed_elevation_m.shape != (g.ny, g.nx):
        raise DenseRunnerError("incompatible_water_inputs", "terrain does not match the bed geometry")
    if type(water.field) is not n.RainfallField or water.field.xp is not np or water.field.shape != (g.ny, g.nx):
        raise DenseRunnerError("incompatible_water_inputs", "field must be the NumPy source RainfallField on the grid")
    veg, cover = water.vegetation_cover_fraction, bed.vegetation_cover_fraction()
    if (type(veg) is not np.ndarray or veg.dtype != cover.dtype or veg.shape != cover.shape
            or not bool(np.array_equal(veg, cover))):
        raise DenseRunnerError("vegetation_mismatch", "water vegetation must be bitwise the bed's canonical "
                               "prescribed cover fraction (no percent conversion, no second vegetation)")
    soil = water.initial_soil_water_m
    if (type(soil) is not np.ndarray or soil.dtype != np.float64 or soil.shape != (g.ny, g.nx)
            or not bool(np.isfinite(soil).all()) or bool((soil < 0).any()) or bool(np.signbit(soil).any())):
        raise DenseRunnerError("invalid_water_inputs", "initial_soil_water_m must be finite (ny, nx) float64 >= +0")
    return np.array(soil, dtype=np.float64, copy=True)


def _source_identity(n: SimpleNamespace, bed: DenseBed, view: SimpleNamespace, soil: Any) -> dict[str, str]:
    """Host identity of the CPU source (no device work)."""
    bed.verify_case_inputs()
    parts = DenseRunner._static_parts_record(view, n)
    return {"bed_identity_sha256": bed.identity_sha256(),
            "controlling_inputs_sha256": _bed._sha256(parts),
            "private_sha256": DenseRunner._private_identity(n.np, view._wind.initial_mobile_state, soil)}


def _preflight_view(state: Any, geometry: Any) -> SimpleNamespace:
    """The named case view the native CUDA preflight reads (`PREFLIGHT_CASE_FIELDS` plus config.geometry)."""
    return SimpleNamespace(config=SimpleNamespace(geometry=geometry), voxel_column=state.voxel_column,
                           active_layer=state.active_layer, water=state.water, sediment_ledger=state.ledger,
                           committed_topography=state.committed_topography,
                           sediment_availability=state.sediment_availability)


# Native water parameter holders: exact field schemas (a native field addition is refused, never dropped).
_COLUMN_ARRAYS = ("ksat_m_per_s", "suction_m", "drainage_parameter", "theta_sat", "soil_thickness_m",
                  "storage_max_m", "lambda_m_per_s", "active_mask")
_COLUMN_FIELDS = ("model", *_COLUMN_ARRAYS, "xp")
_SEDIMENT_HOST = ("diameter_m", "raindrop_a", "raindrop_b", "raindrop_c", "raindrop_depth_attenuation_per_cm",
                  "raindrop_max_depth_per_reference_s_m", "settling_velocity_m_s", "particle_mass_g")
_SEDIMENT_DEVICE = ("d_diameter", "d_raindrop_a", "d_raindrop_b", "d_raindrop_c", "d_attenuation", "d_raindrop_max",
                    "d_settling", "d_particle_mass_g")


def _exact_holder(value: Any, cls: type, names: tuple[str, ...], what: str) -> None:
    if type(value) is not cls:
        raise DenseRunnerError("invalid_water_inputs", f"{what} must be the exact native {cls.__name__}")
    fields = dataclasses.fields(cls)
    if (tuple(f.name for f in fields) != names or any(not f.init for f in fields) or "__post_init__" in vars(cls)
            or set(vars(value)) != set(names)):
        raise DenseRunnerError("unsupported_schema", f"native {cls.__name__} field/attribute schema changed")


def _water_parameter_sources(np: Any, column: Any, sediment: Any, shape: tuple[int, int],
                             nc: int) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
    """Close the native `ColumnParameters` / `SedimentPhysicsParameters` source contracts BEFORE any copy.
    Returns (leaves to place on the device, host leaves the native contract keeps on the host)."""
    from maple_syrup.infiltration import ColumnParameters
    from maple_syrup.sediment_physics import SedimentPhysicsParameters

    _exact_holder(column, ColumnParameters, _COLUMN_FIELDS, "water.column")
    _exact_holder(sediment, SedimentPhysicsParameters,
                  tuple(f.name for f in dataclasses.fields(SedimentPhysicsParameters)), "water.sediment")
    if not set(_SEDIMENT_HOST) | set(_SEDIMENT_DEVICE) | {"xp"} <= set(vars(sediment)):
        raise DenseRunnerError("unsupported_schema", "native SedimentPhysicsParameters array fields changed")
    if column.xp is not np or sediment.xp is not np:
        raise DenseRunnerError("invalid_water_inputs", "water parameter sources must be the NumPy native builds")
    if type(sediment.n_classes) is not int or sediment.n_classes != nc:
        raise DenseRunnerError("incompatible_water_inputs", "sediment parameters do not match the six classes")
    device, host = [], []
    for name in _COLUMN_ARRAYS:
        array = getattr(column, name)
        if array is None and name == "lambda_m_per_s":
            continue
        dtype = np.bool_ if name == "active_mask" else np.float64
        if (type(array) is not np.ndarray or array.dtype != dtype or array.shape != shape
                or not array.flags.c_contiguous):
            raise DenseRunnerError("invalid_water_inputs", f"column.{name}: contiguous {shape} native array required")
        device.append((f"column.{name}", array))
    for name in _SEDIMENT_DEVICE:
        array = getattr(sediment, name)
        if (type(array) is not np.ndarray or array.dtype != np.float64 or array.shape != (1, 1, nc)
                or not array.flags.c_contiguous):
            raise DenseRunnerError("invalid_water_inputs", f"sediment.{name}: contiguous (1, 1, nc) float64 required")
        device.append((f"sediment.{name}", array))
    for name in _SEDIMENT_HOST:
        array = getattr(sediment, name)
        if type(array) is not np.ndarray or array.dtype != np.float64 or array.shape != (nc,):
            raise DenseRunnerError("invalid_water_inputs", f"sediment.{name}: (nc,) float64 host array required")
        host.append((f"sediment.{name}", array))
    for name, value in vars(sediment).items():  # every remaining native field is a host scalar/str by contract
        if name not in _SEDIMENT_HOST and name not in _SEDIMENT_DEVICE and name != "xp" and type(value) not in (
                int, float, str):
            raise DenseRunnerError("unsupported_schema", f"sediment.{name}: undeclared payload refused")
    if type(column.model) is not str:
        raise DenseRunnerError("unsupported_schema", "column.model must be the native model string")
    return device, host


def _device_water_parameters(cp: ModuleType, column: Any, sediment: Any) -> tuple[Any, Any]:
    """Explicit lossless native dataclass state bridge (no native placement API exists for these holders): the
    SAME validated native values, with the native `xp` namespace switched to CuPy and exactly the fields the
    native builders put in `xp` placed there by native `to_device` (one counted host->device copy each).
    Nothing is recomputed: `storage_max_m`, `lambda_m_per_s` and the derived sediment coefficients are the
    already-validated native values. SedimentPhysicsParameters keeps its per-class HOST arrays and scalars on
    the host, exactly as `sediment_physics_parameters(xp=cupy)` builds them."""
    from maple.core.backend import freeze, to_device

    def place(array: Any) -> Any:
        return None if array is None else freeze(to_device(array, cp))

    device_column = dataclasses.replace(column, xp=cp, **{name: place(getattr(column, name))
                                                           for name in _COLUMN_ARRAYS})
    device_sediment = dataclasses.replace(sediment, xp=cp, **{name: place(getattr(sediment, name))
                                                               for name in _SEDIMENT_DEVICE})
    return device_column, device_sediment


def _closed_host_sources(np: Any, bed: DenseBed, wind: DenseWindInputs, water: DenseWaterInputs) -> None:
    """Closed host metadata of every native CPU pool/static source that will be converted, BEFORE any source hash
    or copy: the resource owner's bounded `_closed_metadata` walk (finite depth/item/byte caps; unknown extra
    holder attributes, payloads, dtypes and layouts refused; legitimate native aliases kept), plus exact coverage
    of the native `array_leaves` the reservations count. Engineering guard only; no value policy."""
    from maple.core.backend import array_leaves
    from masy.dense_device_resources import _closed_metadata

    sources = (("pool", wind.initial_mobile_state), ("vegetation", bed._vegetation),
               ("threshold_modifier_state", bed._threshold_modifier_state),
               ("initial_available_fraction", wind.config.initial_available_fraction),
               ("multiplier", water.field.multiplier), ("column", water.column), ("sediment", water.sediment))
    for name, value in sources:
        try:
            _, arrays, _ = _closed_metadata(value, (np.ndarray,), (np,))
        except DeviceResourceError as exc:
            raise DenseRunnerError(exc.code, f"{name}: {exc.detail}") from exc
        if {id(a) for a in arrays} != {id(a) for a in array_leaves(value)}:
            raise DenseRunnerError("unsupported_schema", f"{name}: closed arrays differ from native array_leaves")


def _host_reservations(bed: DenseBed, wind: DenseWindInputs, water: DenseWaterInputs, soil: Any,
                       water_device: list[tuple[str, Any]], bed_limits: DeviceBedLimits,
                       resource_limits: DeviceResourceLimits, device_limits: DeviceRunnerLimits) -> dict[str, int]:
    """Owner reservations from host METADATA of the exact source leaves that will be converted, with the SAME
    caps the owners apply AFTER conversion enforced here BEFORE the first domain copy."""
    from maple.core.backend import array_leaves
    from maple.core.backend.arrays import array_leaf_inventory

    for value in (bed_limits, resource_limits):  # frozen objects can still be forcibly changed
        try:
            value.__post_init__()
        except (DeviceBedError, DeviceResourceError) as exc:
            raise DenseRunnerError(exc.code, exc.detail) from exc
    inventory = array_leaf_inventory(bed.state)
    bed_leaves = list(array_leaves(bed.state))
    largest = max(int(a.nbytes) for a in bed_leaves)
    _, bed_required = _work_reservation(inventory, largest, bed_limits)
    if inventory["transfer_bytes"] > bed_limits.max_bed_bytes or bed_required > bed_limits.max_work_bytes:
        raise DenseRunnerError("work_bytes_cap", "six bed roots plus comparison/native scratch exceed bed_limits")
    pool_leaves = list(array_leaves(wind.initial_mobile_state))
    device_static = [*array_leaves(bed._vegetation), *array_leaves(bed._threshold_modifier_state),
                     *array_leaves(wind.config.initial_available_fraction), *array_leaves(water.field.multiplier),
                     *(a for _, a in water_device)]
    private_bytes = sum(int(a.nbytes) for a in pool_leaves) + int(soil.nbytes)
    static_bytes = sum(int(a.nbytes) for a in device_static)
    leaves = [*pool_leaves, soil, *device_static]
    _, resources_required = resource_reservation(private_bytes, static_bytes, len(leaves),
                                                 max(int(a.nbytes) for a in leaves), resource_limits)
    if (private_bytes > resource_limits.max_private_bytes or static_bytes > resource_limits.max_static_bytes
            or resources_required > resource_limits.max_work_bytes):
        raise DenseRunnerError("work_bytes_cap", "pool/soil/static copies plus comparison exceed resource_limits")
    # Construction-only lifetimes the owner reservations do not hold: the converted static sources coexist with
    # the owned static + mirror until dropped (statics reserve 2 copies), and the bounded host->device source
    # verification downloads one leaf at a time (host scratch = largest converted leaf).
    largest_converted = max(largest, *(int(a.nbytes) for a in leaves))
    return {"bed_owner_six_root_required_bytes": int(bed_required),
            "private_resources_required_bytes": int(resources_required),
            "construction_static_source_bytes": int(static_bytes),
            "bed_transfer_bytes": int(inventory["transfer_bytes"]),
            # + the owned availability downloaded a second time against the host BedContext map
            "construction_verification_download_bytes": int(inventory["transfer_bytes"]) + private_bytes
                                                        + static_bytes
                                                        + int(bed.context.initial_available_fraction.nbytes),
            "construction_verification_host_scratch_bytes": int(largest_converted)}


def _verify_source_copy(cp: ModuleType, np: Any, pairs: list[tuple[Any, Any]]) -> dict[str, int]:
    """Bounded construction-only raw bit-exact host->device check of the declared copy: native leaf discovery
    on host source and converted object in the SAME order; each device leaf is downloaded once through native
    `to_host` (counted), compared as raw bytes with its host source and dropped before the next. Not repeated
    per step."""
    from maple.core.backend import array_leaves, to_host

    n_leaves = n_bytes = 0
    for source, converted in pairs:
        host_leaves, device_leaves = list(array_leaves(source)), list(array_leaves(converted))
        if len(host_leaves) != len(device_leaves):
            raise DenseRunnerError("source_copy_mismatch", "converted leaf count differs from the host source")
        for host, device in zip(host_leaves, device_leaves, strict=True):
            if type(host) is not np.ndarray:
                raise DenseRunnerError("source_copy_mismatch", "host source leaf is not a NumPy array")
            if type(device) is np.ndarray:  # a native host leaf kept on the host by contract: same object
                if device is not host:
                    raise DenseRunnerError("source_copy_mismatch", "a host-resident leaf was copied unexpectedly")
                continue
            if type(device) is not cp.ndarray or device.dtype != host.dtype or tuple(device.shape) != host.shape:
                raise DenseRunnerError("source_copy_mismatch", "converted leaf type/dtype/shape differs")
            back = to_host(device)
            equal = bool(np.array_equal(np.ascontiguousarray(back).reshape(-1).view(np.uint8),
                                        np.ascontiguousarray(host).reshape(-1).view(np.uint8)))
            n_leaves, n_bytes = n_leaves + 1, n_bytes + int(back.nbytes)
            back = None  # noqa: F841 - one leaf of host scratch at a time
            if not equal:
                raise DenseRunnerError("source_copy_mismatch", "raw device bytes differ from the host source")
    return {"verified_device_leaves": n_leaves, "downloaded_bytes": n_bytes}


# --- private facade / ops ------------------------------------------------------------------------------------------

def _device_context(host: Any, availability: Any) -> Any:
    """The native `BedContext` the engine hands to SYRUP on the device path: a genuine `dataclasses.replace` of
    the ORIGINAL host context whose `initial_available_fraction` references the OWNED device availability static
    (the same object the device wind config binds). Every other native field (geometry, grain classes, mass
    resolution, commit/avalanche specs, depth rule, history/window/deep-storage flags) is the host object itself.
    Native `bed_from_case(xp=cupy)` places exactly this one field with the state namespace; SYRUP's
    `resolve_operand_namespace(..., context.initial_available_fraction)` refuses a NumPy fraction next to a CuPy
    bed. No value is recomputed, reset or defaulted."""
    context = dataclasses.replace(host, initial_available_fraction=availability)
    _check_device_context(host, context, availability)
    return context


def _check_device_context(host: Any, context: Any, availability: Any) -> None:
    """Closed binding of the device context to the original host context (host metadata only, no download):
    exact type, exact declared attribute set, every field but availability the SAME object, availability the
    owned static with the host dtype/shape. Raw device availability content is guarded by the resource owner's
    static mirror; host availability content by the case record."""
    names = tuple(f.name for f in dataclasses.fields(type(host)))
    if (type(context) is not type(host) or tuple(f.name for f in dataclasses.fields(type(context))) != names
            or set(vars(context)) != set(names) or set(vars(host)) != set(names)):
        raise DenseBedError("case_input_mutated", "device BedContext type/attribute schema differs from the host")
    for name in names:
        if name == "initial_available_fraction":
            continue
        if getattr(context, name) is not getattr(host, name):
            raise DenseBedError("case_input_mutated", f"device context.{name} is not the original host object")
    held, fraction = host.initial_available_fraction, context.initial_available_fraction
    if fraction is not availability:
        raise DenseBedError("case_input_mutated", "device context availability is not the owned static")
    if (fraction.dtype != held.dtype or tuple(fraction.shape) != tuple(held.shape)
            or not fraction.flags.c_contiguous):
        raise DenseBedError("case_input_mutated", "device context availability dtype/shape differs from the host")


class _HostMetadataBed:
    """What the engine reads from `self._bed` on the device path: host context/case metadata, the device
    `BedContext` bridge and the owner's native time. No physical array of the CPU source bed is held here."""

    __slots__ = ("_context_refs", "_device_context", "_host_context", "_owner", "_resources", "case_record",
                 "case_sha256")

    def __init__(self, owner: DeviceDenseBedOwner, source: DenseBed, resources: DeviceDenseResources) -> None:
        self._owner, self._resources, self._host_context = owner, resources, source.context
        self.case_sha256, self.case_record = source.case_sha256, copy.deepcopy(source.case_record)
        # The SAME native context object relationships the CPU bed binds (`DenseBed._context_refs`).
        self._context_refs = dict(source._context_refs)
        self._device_context = _device_context(source.context, resources.static("initial_available_fraction"))

    @property
    def context(self) -> Any:
        """The device `BedContext` (native SYRUP namespace rules); the host original stays `host_context`."""
        return self._device_context

    @property
    def host_context(self) -> Any:
        return self._host_context

    @property
    def time_s(self) -> float:
        return self._owner.time_s

    def verify(self, n: SimpleNamespace) -> None:
        """`DenseBed._verify_static` over the retained host metadata only (no physical bed array is held):
        case record SHA, exact native context type and object relationships, and every live canonical static
        input (geometry / grain classes / class order / mass resolution / commit / avalanche / availability)
        against the original case record; then the device context binding (`_check_device_context`)."""
        record = self.case_record
        if _bed._sha256(record) != self.case_sha256:
            raise DenseBedError("case_input_mutated", "the held case record no longer reproduces case_sha256")
        self._verify_host(n, record)
        if type(self._device_context) is not n.BedContext:
            raise DenseBedError("case_input_mutated", "the device BedContext was replaced")
        _check_device_context(self._host_context, self._device_context,
                              self._resources.static("initial_available_fraction"))

    def _verify_host(self, n: SimpleNamespace, record: dict[str, Any]) -> None:
        context = self._host_context
        if type(context) is not n.BedContext:
            raise DenseBedError("case_input_mutated", "the held BedContext was replaced")
        for name, ref in self._context_refs.items():
            if getattr(context, name, None) is not ref:
                raise DenseBedError("case_input_mutated", f"context.{name} is no longer the original case object")
        try:
            live = {
                "geometry": _bed._canonical_static(context.geometry, "geometry"),
                "grain_classes": _bed._canonical_static(context.grain_classes, "grain_classes"),
                "class_order": [c.class_id for c in context.grain_classes.classes],
                "mass_resolution_kg": _bed.encode_time(context.mass_resolution_kg),
                "topographic_commit_spec": _bed._canonical_static(context.topographic_commit_spec,
                                                                  "topographic_commit_spec"),
                "avalanching_spec": _bed._canonical_static(context.avalanching_spec, "avalanching_spec"),
                "initial_available_fraction": _bed._array_digest(context.initial_available_fraction),
            }
        except (ValueError, TypeError, AttributeError) as exc:
            raise DenseBedError("case_input_mutated", f"a live static input no longer canonicalizes ({exc})") from exc
        changed = sorted(key for key, value in live.items() if value != record[key])
        if changed:
            raise DenseBedError("case_input_mutated", f"live static input(s) {changed} differ from the case record")


class _DeviceArrayOps:
    """Explicit compact host reads of device values: ONE native `host_packet` per read; spatial payloads are
    refused; native scalar / length-1 / nc-length shapes and negative-zero bytes are kept as returned."""

    __slots__ = ("_cp", "_np", "_packet", "_max", "xp")

    def __init__(self, cp: ModuleType, n_classes: int) -> None:
        import numpy
        from maple_syrup.event_backend import host_packet

        self._cp, self._np, self._packet, self._max, self.xp = cp, numpy, host_packet, n_classes, cp

    def _host(self, value: Any) -> Any:
        np = self._np
        if type(value) is self._cp.ndarray:
            if value.ndim > 1 or value.size > self._max:
                raise DenseRunnerError("spatial_payload", "compact reads are class vectors or scalars only")
            return self._packet(self._cp, [], value).reshape(-1)
        if type(value) is list:
            if len(value) > self._max:
                raise DenseRunnerError("spatial_payload", "compact reads are class vectors or scalars only")
            if any(type(v) not in (float, int) for v in value):
                raise DenseRunnerError("unsupported_value", "host lists must hold plain floats or ints")
            return np.asarray(value, dtype=np.float64).reshape(-1)
        if type(value) is np.ndarray or type(value) in (float, int) or isinstance(value, (np.floating, np.integer)):
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            if array.size > self._max:
                raise DenseRunnerError("spatial_payload", "compact reads are class vectors or scalars only")
            return array
        raise DenseRunnerError("unsupported_value", f"no compact read for {type(value).__name__}")

    def class_vector(self, value: Any) -> Any:
        return self._host(value)

    def float_values(self, value: Any) -> list[float]:
        return [float(v) for v in self._host(value)]

    def inventory(self, n: SimpleNamespace, state: Any) -> Any:
        return self._host(n.bed_inventory(state))

    def any_nonzero(self, value: Any) -> bool:
        if type(value) is self._cp.ndarray:
            return float(self._packet(self._cp, [self._cp.any(value != 0.0)])[0]) == 1.0
        return bool(self._np.any(value != 0.0))

    def equal_class_vector(self, vector: Any, value: Any) -> bool:
        return bool(self._np.array_equal(vector, self._host(value)))

    def scalar(self, value: Any) -> float:
        array = self._host(value)
        if array.shape != (1,):
            raise DenseRunnerError("unsupported_value", "a native scalar must be 0-d or length 1")
        return float(array[0])


class _PrivateRef:
    """Opaque boundary reference to the owner's private pool/soil generation (never the arrays)."""

    __slots__ = ("generation",)

    def __init__(self, generation: int) -> None:
        self.generation = generation


@dataclasses.dataclass(frozen=True, eq=False)
class _DeviceStage:
    bed: Any
    resources: Any
    generation: int
    time_s: float


@dataclasses.dataclass(frozen=True, eq=False)
class _DevicePublication:
    bed: Any
    resources: Any
    generation: int


class _DeviceOwnership:
    """Device bodies of the engine's `_HostOwnership` operations, over the two private owners. Work copies come
    from owner tickets; every publication is check-both-then-swap-both (assignment only after all checks)."""

    receipt_schema = DEVICE_RECEIPT_SCHEMA

    def __init__(self, bed: DeviceDenseBedOwner, resources: DeviceDenseResources, ops: _DeviceArrayOps,
                 n_classes: int, stage_bytes: int, metadata: _HostMetadataBed) -> None:
        self._owner, self._res, self._ops, self._nc = bed, resources, ops, n_classes
        self._metadata = metadata
        self._stage_bytes = stage_bytes
        self._stage: _DeviceStage | None = None
        self._prepared: _DevicePublication | None = None
        self._routing_generation = 0
        # Opaque per-adapter identity of every held routing namespace this adapter issues: a bare object, so the
        # namespace keeps no reference to the adapter, its owners or the native bed. Transient; never serialized.
        self._routing_stamp = object()
        self.cleanup_failures = 0

    # labels (verified mirror / generation / host metadata; never content SHAs) ---------------------------------
    @staticmethod
    def _bed_label(generation: int, time_s: float) -> str:
        return f"device-bed:mirror-verified:generation={generation}:time={time_s.hex()}"

    def accepted_state(self) -> Any:
        raise DenseRunnerError("device_private", "the accepted device bed never leaves its owner")

    def accepted_inventory(self, ops: Any, n: SimpleNamespace) -> Any:
        return self._owner._compact(n.bed_inventory, max_values=self._nc).reshape(-1)

    def verify_case_inputs(self) -> None:
        self._metadata.verify(_bed._native())  # DenseBedError crosses the engine's existing seam unchanged
        try:
            self._owner.verify()
            self._res.verify_static()
        except (DeviceBedError, DeviceResourceError) as exc:
            raise _device_error(exc) from exc

    def accepted_label(self) -> str:
        try:
            self._owner.verify()
        except DeviceBedError:
            return "device-bed:mirror-verification-failed"
        return self._bed_label(self._owner.generation, self._owner.time_s)

    def private_label(self, np: Any, pool: Any, soil: Any) -> str:
        if type(pool) is not _PrivateRef or pool is not soil:
            return "device-private:not-an-owner-reference"
        if pool.generation == self._res.generation:
            try:
                self._res.verify()
            except DeviceResourceError:
                return "device-private:mirror-verification-failed"
        elif self._prepared is None or pool.generation != self._prepared.generation:
            return "device-private:stale-generation"
        return f"device-private:mirror-verified:generation={pool.generation}"

    def routing_label(self, routing: SimpleNamespace) -> str:
        return f"device-routing:held-source:generation={routing.generation}"

    # staging -----------------------------------------------------------------------------------------------------
    def begin(self) -> _DeviceStage:
        try:
            bed_stage = self._owner.begin_stage()
        except DeviceBedError as exc:
            raise _device_error(exc) from exc
        try:
            res_stage = self._res.begin_stage()
        except DeviceResourceError as exc:
            self._owner.discard(bed_stage)
            raise _device_error(exc) from exc
        self._stage = _DeviceStage(bed_stage, res_stage, self._owner.generation, bed_stage.time_s)
        return self._stage

    def stage_state(self, stage: _DeviceStage) -> Any:
        return stage.bed.state

    def stage_label(self, stage: _DeviceStage) -> str:
        return self._bed_label(stage.generation, stage.time_s)

    def candidate_label(self, candidate: Any, time_s: Any) -> str:
        """EXACTLY the label `accepted_label` returns once this candidate is published: the owner swaps in
        generation + 1 at the native clock `prepare_publish` stored. A failed event publishes nothing, so the
        owner generation and the runner's accepted label do not advance."""
        t = time_s if type(time_s) is float else self._ops.scalar(time_s)
        try:
            t = _native_clock(t, "time_s")
        except ValueError as exc:
            raise _Stop("refused", "invalid_time", _runner._detail(exc), exception=exc) from exc
        return self._bed_label(self._owner.generation + 1, t)

    def stage_bytes(self) -> int:
        return self._stage_bytes

    def pool_bytes(self, pool: Any) -> int:
        return self._res.pool_bytes()

    def _current(self) -> _DeviceStage:
        if self._stage is None:
            raise DenseBedError("no_active_stage", "no device work transaction is active")
        return self._stage

    def work_pool(self, pool: Any) -> Any:
        try:
            return self._res.work_pool(self._current().resources)
        except DeviceResourceError as exc:
            raise _device_error(exc) from exc

    def work_soil(self, np: Any, soil: Any) -> Any:
        try:
            return self._res.work_soil(self._current().resources)
        except DeviceResourceError as exc:
            raise _device_error(exc) from exc

    def scheduler(self, stage: _DeviceStage, pool: Any, time_s: float) -> Any:
        """Native `initial_scheduler_state` over the staged device objects and the OWNED device statics (the
        same native factory and identity checks as `DenseBed._scheduler_over`)."""
        n = _bed._native()
        state = stage.bed.state
        if type(time_s) is not float or time_s.hex() != stage.time_s.hex():
            raise DenseBedError("time_mismatch", "scheduler time must equal the stage time bit for bit")
        _bed._check_commit_stamp(state, time_s)
        vegetation, threshold = self._res.static("vegetation"), self._res.static("threshold_modifier_state")
        probe = n.SchedulerPhysicalState(
            voxel_column=state.voxel_column, active_layer=state.active_layer, mobile_state=pool,
            ledger=state.ledger, committed_topography=state.committed_topography,
            threshold_modifier_state=threshold, time_s=time_s)
        if n.carries_coupled_trajectory_history(probe, None):
            raise DenseBedError("mobile_state_not_empty", "a fresh scheduler needs a genuinely empty wind pool")
        if self._ops.any_nonzero(state.water.mobile_mass_by_cell_class_kg):
            raise DenseBedError("water_mobile_not_empty", "water-borne sediment remains; no fresh wind entry")
        scheduler = n.initial_scheduler_state(
            state.voxel_column, state.active_layer, pool, state.ledger, state.committed_topography, threshold,
            time_s=time_s, vegetation=vegetation, sediment_availability=state.sediment_availability,
            water=state.water)
        physical = scheduler.physical
        if (any(getattr(physical, name) is not getattr(state, name) for name in _bed.SHARED_FIELDS)
                or physical.vegetation is not vegetation or physical.threshold_modifier_state is not threshold
                or physical.mobile_state is not pool
                or physical.coupled_trajectory_semantics != n.TRAJECTORY_ALGORITHM_VERSION):
            raise DenseBedError("native_factory_mismatch", "initial_scheduler_state did not carry the shared objects")
        return scheduler

    # retained routing: graph / terrain / network / grid / source plus a private zero velocity ---------------------
    def memory_names(self, np: Any, held: SimpleNamespace) -> tuple[str, ...]:
        return ("sediment_velocity_m_s",) if type(held.velocity) is self._ops.xp.ndarray else ()

    def memory_bytes(self, held: SimpleNamespace, names: tuple[str, ...]) -> int:
        return int(held.velocity.nbytes) if names else 0

    def work_memory(self, np: Any, held: SimpleNamespace, names: tuple[str, ...]) -> dict[str, Any]:
        return {name: held.velocity.copy() for name in names}

    def entry_memory_nonempty(self, ops: Any, held: SimpleNamespace) -> bool:
        return held.phase is not None or ops.any_nonzero(held.velocity)

    def entry_state(self, n: SimpleNamespace, held: SimpleNamespace, replaced: dict[str, Any],
                    memory: dict[str, Any]) -> Any:
        from maple_syrup.sediment_event import SedimentEventState

        fields = tuple(f.name for f in dataclasses.fields(SedimentEventState))
        if sorted(fields) != sorted(_EVENT_FIELDS) or any(not f.init for f in dataclasses.fields(SedimentEventState)):
            raise DenseBedError("unsupported_schema", f"native SedimentEventState fields changed: {fields}")
        values = {**replaced, **memory}
        if "sediment_velocity_m_s" not in values:  # first entry: the seed's own native zero velocity
            values["sediment_velocity_m_s"] = held.velocity
        if sorted(values) != sorted(_EVENT_FIELDS):
            raise DenseBedError("unsupported_schema", "entry fields do not match the native event state")
        return SedimentEventState(**values)

    # The engine's three closed retained-routing calls (`_HostOwnership.routing_memory/_guard/_entry`). Only this
    # owner's own namespace is accepted; a CPU restored projection or any other holder refuses before transport.
    # The generation is bounded by the ISSUE counter (not "== latest"): `retain_routing` runs before `_finish`
    # publishes, and `abandon` does not roll the counter back, so the accepted routing can be older than a newer
    # namespace issued for an event that then failed before publication.
    def _held_routing(self, held: Any, water: Any) -> SimpleNamespace:
        if (type(held) is not SimpleNamespace or set(vars(held)) != _HELD_ROUTING_FIELDS
                or held.stamp is not self._routing_stamp
                or type(held.generation) is not int or not 0 < held.generation <= self._routing_generation
                or held.terrain is not water.terrain):
            raise _Stop("refused", "restored_routing_layout_changed", "unknown retained device routing",
                        guarantee=_runner._EXTERNAL)
        return held

    def routing_memory(self, np: Any, held: Any, water: Any, shape: tuple[int, int],
                       nc: int) -> tuple[tuple[str, ...], int]:
        held = self._held_routing(held, water)
        names = self.memory_names(np, held)
        if ("sediment_velocity_m_s" not in names or held.velocity.dtype != self._ops.xp.float64
                or tuple(held.velocity.shape) != (*shape, nc)):
            raise _Stop("refused", "transport_memory_unsupported",
                        "retained sediment velocity is not an owned device grid array; it cannot be work-copied")
        return names, self.memory_bytes(held, names)

    def routing_guard(self, np: Any, held: Any, water: Any, shape: tuple[int, int], nc: int) -> None:
        held = self._held_routing(held, water)
        if self.entry_memory_nonempty(self._ops, held):
            raise _Stop("refused", "transport_memory_not_empty", "upwind entry needs phase None and zero memory")

    def routing_entry(self, n: SimpleNamespace, held: Any, boundary: Any, graph: Any, network: Any, grid: Any,
                      soil: Any, t: float, names: tuple[str, ...], *, water: Any, shape: tuple[int, int],
                      nc: int) -> Any:
        held = self._held_routing(held, water)
        memory = self.work_memory(n.np, held, names)
        return self.entry_state(n, held, dict(
            bed=boundary.state, graph=graph, network=network, grid=grid, terrain=held.terrain,
            routing_source_elevation_m=boundary.routing_source_m, phase=None,
            storm=n.initial_state(graph, boundary.state.water.depth_m, soil, t_s=t)), memory)

    def retain_routing(self, state: Any) -> SimpleNamespace:
        self._routing_generation += 1
        velocity = self._ops.xp.zeros(state.sediment_velocity_m_s.shape, dtype=self._ops.xp.float64)
        return SimpleNamespace(graph=state.graph, terrain=state.terrain, network=state.network, grid=state.grid,
                               source=state.routing_source_elevation_m, velocity=velocity, phase=None,
                               generation=self._routing_generation, stamp=self._routing_stamp)

    # preparation / publication ------------------------------------------------------------------------------
    def _prepare(self, candidate: Any, time_s: Any, *, pool: Any = None, soil: Any = None) -> _DevicePublication:
        stage = self._current()
        t = time_s if type(time_s) is float else self._ops.scalar(time_s)
        try:
            bed_prepared = self._owner.prepare_publish(stage.bed, candidate, time_s=t)
            res_prepared = self._res.prepare_publish(stage.resources, pool=pool, soil=soil)
        except (DeviceBedError, DeviceResourceError) as exc:
            raise _Stop("refused", getattr(exc, "code", "device_owner_error"), exc.detail, exception=exc) from exc
        self._prepared = _DevicePublication(bed_prepared, res_prepared, self._res.generation + 1)
        return self._prepared

    def prepare_scheduler(self, physical: Any) -> _DevicePublication:
        n = _bed._native()
        # The CPU `DenseBed.adopt_scheduler_physical` guards, in the same order, over the owned statics.
        try:
            self._metadata.verify(n)
            _bed._check_exact_type(physical, n.SchedulerPhysicalState, "physical")
        except DenseBedError as exc:
            raise _Stop("refused", exc.code, exc.detail, exception=exc) from exc
        if physical.vegetation is not self._res.static("vegetation"):
            raise _Stop("refused", "prescribed_field_replaced", "physical.vegetation is not the owned static")
        if physical.threshold_modifier_state is not self._res.static("threshold_modifier_state"):
            raise _Stop("refused", "prescribed_field_replaced", "threshold_modifier_state is not the owned static")
        if physical.ledger_reconciliation_baseline is not None:
            raise _Stop("refused", "unsupported_reconciliation_baseline", "BedState has no slot for it")
        try:
            _bed._check_exact_type(physical.mobile_state, n.MobileCohortState, "physical.mobile_state")
        except DenseBedError as exc:
            raise _Stop("refused", exc.code, exc.detail, exception=exc) from exc
        if n.carries_coupled_trajectory_history(physical, None):
            raise _Stop("refused", "mobile_state_not_empty", "airborne wind load remains; no bed handoff")
        candidate = n.BedState(physical.voxel_column, physical.active_layer, physical.water, physical.ledger,
                               physical.committed_topography, physical.sediment_availability)
        return self._prepare(candidate, physical.time_s, pool=physical.mobile_state)

    def prepare_bed(self, state: Any, time_s: float, *, soil: Any = None) -> _DevicePublication:
        return self._prepare(state, time_s, soil=soil)

    def retained_private(self, pool: Any, soil: Any) -> tuple[Any, Any]:
        if self._prepared is None:
            raise DenseBedError("no_prepared_publication", "private references follow a prepared publication")
        ref = _PrivateRef(self._prepared.generation)
        return ref, ref

    def commit(self, prepared: Any) -> None:
        if type(prepared) is not _DevicePublication or prepared is not self._prepared:
            raise TypeError(f"not the current prepared device publication: {type(prepared).__name__}")
        try:
            self._owner._check_publish(prepared.bed)
            self._res._check_publish(prepared.resources)
        except (DeviceBedError, DeviceResourceError) as exc:
            raise _device_error(exc) from exc
        # Assignment only from here (both owners already checked).
        self._owner._swap(prepared.bed)
        self._res._swap(prepared.resources)
        self._stage = None
        self._prepared = None

    def abandon(self) -> None:
        """Ordinary-failure cleanup: discard every active stage/prepared ticket; accepted pairs are untouched."""
        for owner in (self._owner, self._res):
            try:
                owner._discard_active()
            except Exception:  # noqa: BLE001 - counted; the stop receipt is still written
                self.cleanup_failures += 1
        self._stage = None
        self._prepared = None


# --- the bound engine ------------------------------------------------------------------------------------------------

def _bounded_summary(summary: Any) -> dict[str, Any]:
    """Scalar entries of the native `CudaHydraulics.summary()` only (the log list and any context are dropped)."""
    def keep(value: Any, depth: int) -> Any:
        if type(value) in (bool, int, str) or value is None:
            return value
        if type(value) is float:
            return value if math.isfinite(value) else repr(value)
        if type(value) is dict and depth < 2:
            return {str(k): keep(v, depth + 1) for k, v in value.items() if type(v) in (bool, int, float, str, dict)
                    or v is None}
        return None

    record = keep(summary, 0) if type(summary) is dict else {"summary_type": type(summary).__name__}
    if len(json.dumps(record, sort_keys=True)) > _SUMMARY_CHARS:
        record = {"summary_truncated": True}
    return record


class _DeviceDenseRunner(DenseRunner):
    """The SAME engine with the private device hook bodies; built only by `_create_dense_device_runner`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - public construction is refused
        raise DenseRunnerError("constructor", "private factory only")

    def _ownership(self) -> Any:
        return self._device_own

    def _array_ops(self, np: Any) -> Any:
        return self._device_ops

    def _static_parts_record(self, n: SimpleNamespace) -> dict[str, Any]:
        try:
            self._device_resources.verify_static()
        except DeviceResourceError as exc:
            raise DenseRunnerError(exc.code, exc.detail) from exc
        view = self._host_view
        _bind_device_inputs(n.np, self._cp, view._wind, view._water, self._wind, self._water,
                            self._device_resources)
        return DenseRunner._static_parts_record(view, n)

    def _rain_native_kwargs(self) -> dict[str, Any]:
        from maple_syrup.sediment_event import CudaHydraulics

        # Fresh lifecycle per rain event; it has no close/free API, so only references are dropped afterwards.
        self._hydraulics = CudaHydraulics(mode=self._device_limits.cuda_mode,
                                          max_log=self._device_limits.hydraulics_max_log)
        self._hydraulics_summary = None
        return {"hydraulics": self._hydraulics}

    def _rain_native_release(self) -> None:
        lifecycle, self._hydraulics = self._hydraulics, None
        try:
            self._hydraulics_summary = None if lifecycle is None else _bounded_summary(lifecycle.summary())
        except Exception as exc:  # noqa: BLE001 - bounded metadata only; no traceback/context retained
            self._hydraulics_summary = {"summary_error": type(exc).__name__}
        lifecycle = None  # noqa: F841 - explicit drop of the last local reference

    def _rain_receipt_extra(self) -> dict[str, Any]:
        return {"device": {"cuda_hydraulics": self._hydraulics_summary,
                           "lifecycle": "fresh native CudaHydraulics per rain; references dropped after the call; "
                                        "driver/pool deallocation not asserted"}}

    def _abandon(self) -> None:
        self._hydraulics = None
        self._device_own.abandon()

    def _stop_evidence(self, stop: _Stop) -> dict[str, Any]:
        """Bounded host metadata only: no native result/handoff/progress tree (they reference device candidates
        and contexts), no exception object, traceback or CUDA context, no invented duration."""
        evidence = stop.evidence
        return {"evidence_type": None if evidence is None else type(evidence).__name__,
                "evidence_retained": False,
                "native_unaccepted_elapsed_s": stop.elapsed if type(stop.elapsed) is str else None,
                "exception_type": None if stop.exception is None else type(stop.exception).__name__,
                "cleanup_failures": self._device_own.cleanup_failures,
                "cuda_hydraulics": self._hydraulics_summary}

    def report(self) -> dict[str, Any]:
        record = super().report()
        record["schema"] = DEVICE_REPORT_SCHEMA
        return record

    @property
    def mobile_state(self) -> Any:
        raise DenseRunnerError("device_private", "the accepted device wind pool never leaves its owner")

    @property
    def soil_water_m(self) -> Any:
        raise DenseRunnerError("device_private", "the accepted device soil water never leaves its owner")

    @property
    def construction_evidence(self) -> dict[str, Any]:
        return copy.deepcopy(self._construction_evidence)

    def device_evidence(self) -> dict[str, Any]:
        return {"bed": self._device_bed.evidence(), "resources": self._device_resources.evidence(),
                "cleanup_failures": self._device_own.cleanup_failures,
                "qualification": {"gpu_transport_qualified": False,
                                  "cpu_gpu_event_comparison": "not_run; policy: each model's existing native "
                                                              "comparison bounds, never relaxed; exact copies, "
                                                              "handoffs and discrete decisions exact where required",
                                  "memory_fit_qualified": False, "restart_supported": False}}


# --- the private explicit factory ------------------------------------------------------------------------------------

def _create_dense_device_runner(bed: DenseBed, schedule: DenseSchedule, *, wind: DenseWindInputs,
                                water: DenseWaterInputs, control: DenseRunnerControl, limits: DenseRunnerLimits,
                                device_limits: DeviceRunnerLimits, bed_limits: DeviceBedLimits,
                                resource_limits: DeviceResourceLimits, device_id: int) -> _DeviceDenseRunner:
    """New-run CPU -> GPU initialization of the first bounded profile (module docstring). Every refusal raises
    `DenseRunnerError` with no runner; any device objects made before an ordinary failure are dropped, and the
    raised error carries no traceback reference to them."""
    failure: tuple[str, str] | None = None
    made: dict[str, Any] = {}
    try:
        runner = _build(bed, schedule, wind, water, control, limits, device_limits, bed_limits, resource_limits,
                        device_id, made)
    except DenseRunnerError as exc:
        failure = (exc.code, exc.detail)
    except (DenseBedError, DeviceBedError, DeviceResourceError) as exc:
        failure = (exc.code, exc.detail)
    except Exception as exc:  # noqa: BLE001 - native conversion / preflight failure
        failure = ("device_construction_failure", _runner._detail(exc))
    if failure is not None:
        made.clear()  # drop every converted/owned device object before raising outside the except clause
        raise DenseRunnerError(*failure)
    return runner


def _build(bed: Any, schedule: Any, wind: Any, water: Any, control: Any, limits: Any, device_limits: Any,
           bed_limits: Any, resource_limits: Any, device_id: Any, made: dict[str, Any]) -> _DeviceDenseRunner:
    if type(bed) is not DenseBed:
        raise DenseRunnerError("invalid_input", "bed must be a validated CPU DenseBed")
    if type(schedule) is not DenseSchedule:
        raise DenseRunnerError("invalid_input", "schedule must be a DenseSchedule")
    for value, expected in ((wind, DenseWindInputs), (water, DenseWaterInputs), (control, DenseRunnerControl),
                            (limits, DenseRunnerLimits), (device_limits, DeviceRunnerLimits),
                            (bed_limits, DeviceBedLimits), (resource_limits, DeviceResourceLimits)):
        if type(value) is not expected:
            raise DenseRunnerError("invalid_input", f"expected {expected.__name__}, got {type(value).__name__}")
    device_limits.__post_init__()
    if type(device_id) is not int or device_id < 0:
        raise DenseRunnerError("invalid_input", "device_id must be a plain int >= 0")
    if len(schedule.events) > limits.max_events:
        raise DenseRunnerError("event_limit", f"{len(schedule.events)} events > max_events={limits.max_events}")
    if hasattr(bed, "_dynamic"):  # the CPU engine runs evolving vegetation only with its explicit plan
        raise DenseRunnerError("unsupported_profile", "dynamic vegetation beds are CPU-only; no evolving GPU vegetation")
    n = _runner._native()
    np = n.np
    # Finite first profile, then the closed live CPU bed schema, BEFORE any canonicalization/hash/copy.
    _check_profile_caps(n, bed, wind, limits)
    host_schema = _host_bed_admission(bed)
    cp = _genuine_cupy()
    bed.verify_case_inputs()
    import inspect

    if "record_origin_s" not in inspect.signature(n.finalize_wind_event).parameters:
        raise DenseRunnerError("unsupported_native_finalizer", "MAPLE must support explicit record_origin_s")
    DenseRunner._check_wind(n, bed, wind)  # strict default profile; every enabled unsupported feature refused
    soil = _check_device_water(n, bed, water, control, limits)
    ctx, g = bed.context, bed.context.geometry
    nc = len(ctx.grain_classes.classes)
    if nc != _N_CLASSES:
        raise DenseRunnerError("unsupported_profile", f"first device profile is the six-class fixture, got {nc}")
    if int(cp.cuda.runtime.getDevice()) != device_id:
        raise DenseRunnerError("device_mismatch", "device_id must be the current CuPy device")

    host_view = SimpleNamespace(_water=water, _control=control, _wind=wind, _schedule=schedule, _production=None)
    # Exact native parameter holders and the closed host pool/static metadata BEFORE the first source hash/copy.
    water_device, _ = _water_parameter_sources(np, water.column, water.sediment, (g.ny, g.nx), nc)
    _closed_host_sources(np, bed, wind, water)
    before = _source_identity(n, bed, host_view, soil)

    # Known construction admission BEFORE the first domain copy: owner caps, the native CUDA preflight (exact
    # routine, named host case view), named PARTIAL lower reservations and declared allowances. This is NOT a
    # complete full-transport admission or peak fit (see `_workspace_mapping`).
    reserves = _host_reservations(bed, wind, water, soil, water_device, bed_limits, resource_limits, device_limits)
    from maple_syrup.sediment_experiment import cuda_memory_preflight

    native = cuda_memory_preflight(_preflight_view(bed.state, g), phase_bins=1, device_id=device_id,
                                   transport_scheme="upwind")
    missing = [key for key in _PREFLIGHT_KEYS if key not in native]
    if missing:
        raise DenseRunnerError("native_preflight_schema", f"native preflight lacks {missing}")
    workspace = _workspace_mapping(n, bed, wind, soil, native, device_limits)
    required = (int(native["required_bytes"]) + reserves["bed_owner_six_root_required_bytes"]
                + reserves["private_resources_required_bytes"] + reserves["construction_static_source_bytes"]
                + workspace["named_total_bytes"] + workspace["declared_total_bytes"])
    if required > int(native["device_free_bytes"]) or required > device_limits.max_device_bytes:
        raise DenseRunnerError("device_bytes_cap", f"native estimate + owner/construction/named/declared terms "
                               f"{required} bytes exceed device free {native['device_free_bytes']} or "
                               f"max_device_bytes={device_limits.max_device_bytes}")

    # ONE declared lossless native conversion per source object (new run only).
    from maple.core.backend import to_device_tree
    from maple_syrup.rainfall import rainfall_field

    made["state"] = to_device_tree(bed.state, cp)
    made["pool"] = to_device_tree(wind.initial_mobile_state, cp)
    made["soil"] = to_device_tree(soil, cp)
    made["multiplier"] = to_device_tree(water.field.multiplier, cp)
    made["column"], made["sediment"] = _device_water_parameters(cp, water.column, water.sediment)
    made["statics"] = {
        "vegetation": to_device_tree(bed._vegetation, cp),
        "threshold_modifier_state": to_device_tree(bed._threshold_modifier_state, cp),
        "initial_available_fraction": to_device_tree(wind.config.initial_available_fraction, cp),
        "column": made["column"],
        "field": rainfall_field(g.ny, g.nx, scale=made["multiplier"], xp=cp),
        "sediment": made["sediment"],
    }
    if tuple(made["statics"]) != STATIC_NAMES:
        raise DenseRunnerError("invalid_static", "static names differ from the resource owner schema")
    if made["statics"]["column"].xp is not cp or made["statics"]["sediment"].xp is not cp:
        raise DenseRunnerError("invalid_static", "native water parameters must share the CuPy graph namespace")
    statics = made["statics"]
    copy_check = _verify_source_copy(cp, np, [
        (bed.state, made["state"]), (wind.initial_mobile_state, made["pool"]), (soil, made["soil"]),
        (water.field.multiplier, made["multiplier"]), (bed._vegetation, statics["vegetation"]),
        (bed._threshold_modifier_state, statics["threshold_modifier_state"]),
        (wind.config.initial_available_fraction, statics["initial_available_fraction"]),
        # the device BedContext availability is this same static: bit-exact against the HOST context's own map too
        (ctx.initial_available_fraction, statics["initial_available_fraction"]),
        (water.column, statics["column"]), (water.sediment, statics["sediment"])])
    statics = None
    made["bed_owner"] = DeviceDenseBedOwner.from_device_bed(
        made["state"], geometry=g, mass_resolution_kg=ctx.mass_resolution_kg, n_voxels=bed.case_record["n_voxels"],
        n_classes=nc, case_sha256=bed.case_sha256, time_s=bed.time_s, limits=bed_limits)
    made["resources"] = DeviceDenseResources._create(pool=made["pool"], soil=made["soil"], statics=made["statics"],
                                                     shape=(g.ny, g.nx), limits=resource_limits,
                                                     _ops=_CupyOps(cp))
    for name in ("state", "pool", "soil", "multiplier", "column", "sediment", "statics"):  # owners hold clones
        made.pop(name)

    after = _source_identity(n, bed, host_view, soil)
    if after != before or _host_bed_admission(bed) != host_schema:
        raise DenseRunnerError("source_changed_during_conversion", "the CPU source identity changed across the copy")

    owner, resources = made["bed_owner"], made["resources"]
    ops = _DeviceArrayOps(cp, nc)
    metadata = _HostMetadataBed(owner, bed, resources)  # device context bound to the owned availability static
    metadata.verify(n)
    own = _DeviceOwnership(owner, resources, ops, nc, reserves["bed_transfer_bytes"], metadata)
    runner = object.__new__(_DeviceDenseRunner)
    runner._production = None
    runner._device_bed, runner._device_resources = owner, resources
    runner._device_ops, runner._device_own = ops, own
    runner._device_limits, runner._hydraulics, runner._hydraulics_summary = device_limits, None, None
    runner._host_view = host_view
    runner._bed = metadata
    runner._schedule = schedule
    runner._wind = DenseWindInputs(
        config=dataclasses.replace(wind.config,
                                   initial_available_fraction=resources.static("initial_available_fraction")),
        initial_mobile_state=None)
    runner._water = DenseWaterInputs(
        terrain=water.terrain, column=resources.static("column"), field=resources.static("field"),
        vegetation_cover_fraction=resources.static("vegetation").raster.cover_fraction,
        sediment=resources.static("sediment"), initial_soil_water_m=None)
    runner._cp = cp
    _bind_device_inputs(np, cp, wind, water, runner._wind, runner._water, resources)
    runner._control, runner._limits = control, limits
    runner._static_parts = DenseRunner._static_parts_record(host_view, n)
    runner._static_parts_sha256 = _bed._sha256(runner._static_parts)
    runner._static_record = {
        "schema": DEVICE_STATIC_SCHEMA,
        "profile": dict(DEVICE_PROFILE),
        "case_sha256": bed.case_sha256,
        "schedule_sha256": schedule.digest,
        "case_schedule_sha256": case_schedule_digest(bed.case_sha256, schedule),
        "controlling_inputs_sha256": runner._static_parts_sha256,
        "limits": limits.as_record(),
        "device_limits": device_limits.as_record(),
        "bed_limits": dataclasses.asdict(bed_limits),
        "resource_limits": dataclasses.asdict(resource_limits),
        "initial_evolving": {
            "mobile_state_sha256": _bed._sha256(_runner._canonical_input(np, wind.initial_mobile_state, "mobile")),
            "soil_water_m": _bed._array_digest(soil),
        },
        "source": {"bed_identity_sha256": before["bed_identity_sha256"],
                   "conversion": "native to_device_tree, one declared copy at new-run construction; host source "
                                 "identity equal before and after; device labels are mirror/generation labels"},
    }
    runner._static_sha256 = _bed._sha256(runner._static_record)
    runner._construction_evidence = {
        "native_cuda_preflight": {key: native[key] for key in _PREFLIGHT_KEYS},
        "native_preflight_scope": "MAPLE/SYRUP water residency only; wind cohorts and FFT not estimated (FFT "
                                  "refused); wind/finalizer and routing terms are named source-derived (partial "
                                  "lower where marked) holdings plus DECLARED unmeasured allowances",
        "owner_reservations": reserves, "workspace_mapping": workspace, "required_device_bytes": required,
        "first_profile_caps": dict(_FIRST_PROFILE_CAPS),
        "source_copy_verification": {**copy_check, "scope": "construction only, raw bytes host vs device, one "
                                                            "counted native to_host per device leaf"},
    }
    runner._n_classes = nc
    runner._inventory_elements = g.ny * g.nx * (bed.case_record["n_voxels"] + 1)
    runner._inventory0 = own.accepted_inventory(ops, n).copy()
    zeros = np.zeros(nc, dtype=np.float64)
    ref = _PrivateRef(resources.generation)
    runner._pool, runner._soil, runner._routing = ref, ref, None
    runner._ledger = ClockLedger(bed.time_s)
    runner._next = runner._n_accepted = 0
    runner._totals = {"export": zeros.copy(), "input": zeros.copy(), "bound": zeros.copy(),
                      "internal_terminal_deposition": zeros.copy(),
                      "water": {"rain_m3": 0.0, "drainage_m3": 0.0, "storm_export_m3": 0.0,
                                "dry_surface_removed_m3": 0.0, "dry_soil_removed_m3": 0.0},
                      "n_entry_reroutes": 0, "n_event_reroutes": 0}
    runner._receipts, runner._stop_receipt, runner._active_evidence = [], None, None
    runner._accepted_bed = own.accepted_label()
    runner._accepted_private = own.private_label(np, ref, ref)
    if runner._accepted_private != f"device-private:mirror-verified:generation={resources.generation}":
        raise DenseRunnerError("integrity_failed", "initial private pool/soil mirror verification failed")
    made.clear()  # the runner is now the sole holder of the owners
    return runner
