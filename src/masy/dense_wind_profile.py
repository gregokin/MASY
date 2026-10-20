"""Dense production wind configuration foundation (MASY-10, configuration only).

A standalone factory that turns an exact native MAPLE `ModelConfig` into the actual native
`SchedulerPhysicsConfig` through the one native mapper, `build_scheduler_physics_config`, plus a
transparent canonical audit of every caller-controlling input. Nothing here transports, allocates a bed,
advances a schedule or chooses a calibration, and the result is NOT qualified for transport.

The caller-supplied `WindForcingRecordSpec` is validated solely as the native forcing-envelope input (it
sets the scheduler's supported shear bound); it is not an execution schedule and defines no clock or water
interval. The calibration declaration is caller-supplied provenance, never scientific certification.
Environment and source locks remain the caller runtime lock's responsibility. numpy, MAPLE and the dense
runner are imported only inside calls.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

__all__ = [
    "DenseProductionWindProfile",
    "DenseSyntheticVegetationWakeProfile",
    "ProductionProfileError",
    "audit_dense_production_wind_config",
    "audit_dense_synthetic_vegetation_wake_config",
    "build_dense_production_wind_config",
    "build_dense_synthetic_vegetation_wake_config",
    "DenseProductionGpuWindProfile",
    "build_dense_production_gpu_wind_config",
    "audit_dense_production_gpu_wind_config",
]

FACTORY_VERSION = "masy-dense-production-wind-config-v1"
AUDIT_SCHEMA = "masy-dense-production-wind-config-audit-v1"
QUALIFICATION = "configuration_only_unqualified_for_transport"
ENVELOPE_ROLE = "forcing_bound_only_not_execution_schedule"
AVAILABILITY_POLICY = "explicit_caller_host_float64_ny_nx_nclasses_finite_unit_interval_identity_preserved"
DECLARATION_MEANING = "caller_supplied_provenance_not_scientific_certification"
REVIEWED_USER_CALIBRATION = "reviewed_user_calibration"
SYNTHETIC_TEST_FIXTURE = "synthetic_test_fixture"
CALIBRATION_STATUSES = (REVIEWED_USER_CALIBRATION, SYNTHETIC_TEST_FIXTURE)
_SYNTHETIC_SCHEMA = "synthetic-test-fixture-v1"
_SYNTHETIC_CALIBRATION_ID = "synthetic-scheduler-fixture-sand100um-NOT-MEASURED"
SYNTHETIC_WAKE_FEATURE = "masy.synthetic_native_vegetation_wake_interface.v1"
SYNTHETIC_WAKE_AUDIT_SCHEMA = "masy-dense-synthetic-vegetation-wake-config-audit-v1"
SYNTHETIC_WAKE_FACTORY_VERSION = "masy-dense-synthetic-vegetation-wake-config-v1"
GPU_FACTORY_VERSION = "masy-dense-production-gpu-wind-config-v1"
GPU_AUDIT_SCHEMA = "masy-dense-production-gpu-wind-config-audit-v1"
_NATIVE_BUILDER = "maple.experiment.scheduler_adapter.build_scheduler_physics_config"

# The only modes this foundation accepts, and the switches it refuses unless exactly False. Everything else
# (inert prescribed/reduced specs, default transport accuracy, creep, topographic wind, flow separation,
# avalanching, ...) is left to native defaults and native validation.
_REQUIRED = (
    ("aeolian_process_mode", "mode", "detachment_transport_deposition"),
    ("detachment_source", "mode", "reduced"),
    ("execution", "backend", "numpy"),
    ("reduced_detachment", "out_of_envelope_policy", "reject"),
)
# The same scientific requirements, differing only in the native execution backend.
_GPU_REQUIRED = (
    ("aeolian_process_mode", "mode", "detachment_transport_deposition"),
    ("detachment_source", "mode", "reduced"),
    ("execution", "backend", "cupy"),
    ("reduced_detachment", "out_of_envelope_policy", "reject"),
)
_REFUSED_UNLESS_FALSE = (
    ("water_coupling", "enabled"),
    ("vegetation_wake", "enabled"),
    ("prescribed_external_shear", "enabled"),
    ("topographic_commit", "dry_aeolian_commit_separation_enabled"),
)


class ProductionProfileError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclasses.dataclass(frozen=True)
class DenseProductionWindProfile:
    """An exact native `ModelConfig` plus the caller's calibration declaration (provenance only)."""

    model_config: Any
    calibration_status: str
    calibration_reference: str


@dataclasses.dataclass(frozen=True)
class DenseSyntheticVegetationWakeProfile(DenseProductionWindProfile):
    """Explicit authored synthetic-only configuration, with native default wake coefficients.

    The ordinary factory APIs still refuse this type. No calibration or transport
    certification is implied by the marker or by a caller-supplied declaration.
    """

    interface_feature: str = dataclasses.field(default=SYNTHETIC_WAKE_FEATURE, init=False)


@dataclasses.dataclass(frozen=True)
class DenseProductionGpuWindProfile:
    """An exact native `ModelConfig` (cupy backend) plus the caller's calibration declaration (provenance only)."""

    model_config: Any
    calibration_status: str
    calibration_reference: str


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _native() -> SimpleNamespace:
    """The actual native objects, imported only when a call needs them."""
    import numpy as np
    from maple.aeolian.scheduler.state import SchedulerPhysicsConfig
    from maple.core.parameters import ModelConfig, config_to_dict, validate_config
    from maple.core.parameters.wind_forcing import (
        WindBlock,
        WindForcingRecordSpec,
        validate_wind_forcing_record,
    )
    from maple.experiment.scheduler_adapter import build_scheduler_physics_config

    return SimpleNamespace(
        np=np, SchedulerPhysicsConfig=SchedulerPhysicsConfig, ModelConfig=ModelConfig,
        config_to_dict=config_to_dict, validate_config=validate_config, WindBlock=WindBlock,
        WindForcingRecordSpec=WindForcingRecordSpec, validate_wind_forcing_record=validate_wind_forcing_record,
        build_scheduler_physics_config=build_scheduler_physics_config,
    )


def _check_profile(native: SimpleNamespace, profile: Any) -> Any:
    if type(profile) is not DenseProductionWindProfile:
        raise ProductionProfileError("wrong_native_type",
                                     f"profile is {_type_name(profile)}; expected DenseProductionWindProfile")
    return _check_profile_rules(native, profile, _REQUIRED)


def _check_profile_rules(native: SimpleNamespace, profile: Any, required: tuple[tuple[str, str, str], ...]) -> Any:
    config = profile.model_config
    if type(config) is not native.ModelConfig:
        raise ProductionProfileError("wrong_native_type",
                                     f"profile.model_config is {_type_name(config)}; expected exact native ModelConfig")
    status, reference = profile.calibration_status, profile.calibration_reference
    if type(status) is not str or status not in CALIBRATION_STATUSES:
        raise ProductionProfileError("invalid_calibration_declaration",
                                     f"calibration_status={status!r}; expected one of {list(CALIBRATION_STATUSES)}")
    if type(reference) is not str or not reference.strip():
        raise ProductionProfileError("invalid_calibration_declaration",
                                     f"calibration_reference={reference!r}; expected a nonempty string")
    try:
        native.validate_config(config)
    except (ValueError, TypeError) as exc:
        raise ProductionProfileError("invalid_model_config", str(exc)) from exc
    reduced = config.reduced_detachment
    if status == REVIEWED_USER_CALIBRATION and (
            reduced.calibration_schema == _SYNTHETIC_SCHEMA or reduced.calibration_id == _SYNTHETIC_CALIBRATION_ID):
        raise ProductionProfileError(
            "calibration_declaration_refused",
            f"reduced_detachment {reduced.calibration_id!r} (schema {reduced.calibration_schema!r}) is the known "
            f"synthetic test fixture and cannot be declared {REVIEWED_USER_CALIBRATION!r}")
    for group, name, wanted in required:
        value = getattr(getattr(config, group), name)
        if type(value) is not str or value != wanted:
            raise ProductionProfileError("unsupported_mode",
                                         f"{group}.{name}={value!r}; this configuration foundation requires {wanted!r}")
    for group, name in _REFUSED_UNLESS_FALSE:
        value = getattr(getattr(config, group), name)
        if value is not False:
            raise ProductionProfileError(
                "unsupported_mode", f"{group}.{name}={value!r}; refused as unqualified for this configuration foundation")
    return config


def _check_synthetic_wake_profile(native: SimpleNamespace, profile: Any) -> Any:
    from maple.core.parameters.geometry import GeometrySpec
    from maple.core.parameters.grain_classes import GrainClassSet
    from maple.core.parameters.vegetation_wake import (
        VegetationWakeSpec,
        VegetationWindwardSpec,
    )

    if type(profile) is not DenseSyntheticVegetationWakeProfile:
        raise ProductionProfileError("wrong_native_type", "exact DenseSyntheticVegetationWakeProfile required")
    config = profile.model_config
    if type(config) is not native.ModelConfig:
        raise ProductionProfileError("wrong_native_type", "exact native ModelConfig required")
    if type(profile.interface_feature) is not str or profile.interface_feature != SYNTHETIC_WAKE_FEATURE:
        raise ProductionProfileError("invalid_interface_feature", "fixed synthetic native wake marker required")
    if type(profile.calibration_status) is not str or profile.calibration_status != SYNTHETIC_TEST_FIXTURE:
        raise ProductionProfileError("invalid_calibration_declaration", "synthetic_test_fixture status required")
    reference = profile.calibration_reference
    if type(reference) is not str or not 1 <= len(reference) <= 4096 or not reference.strip():
        raise ProductionProfileError("invalid_calibration_declaration", "bounded nonempty synthetic fixture reference required")
    geometry, grains = config.geometry, config.grain_classes
    if type(geometry) is not GeometrySpec or type(grains) is not GrainClassSet:
        raise ProductionProfileError("wrong_native_type", "exact native geometry and grain-class holders required")
    if (type(geometry.ny) is not int or type(geometry.nx) is not int
            or not 1 <= geometry.ny <= 64 or not 1 <= geometry.nx <= 64
            or geometry.ny * geometry.nx > 64 or type(grains.classes) is not tuple or len(grains.classes) != 1):
        raise ProductionProfileError("synthetic_interface_scope", "at most64cells and exactly1native class required")
    wake = config.vegetation_wake
    if type(wake) is not VegetationWakeSpec or type(wake.windward) is not VegetationWindwardSpec:
        raise ProductionProfileError("wrong_native_type", "exact native wake and windward holders required")
    if wake.enabled is not True:
        raise ProductionProfileError("unsupported_mode", "synthetic wake interface requires native enabled=True")
    try:
        native.validate_config(config)
    except (ValueError, TypeError) as exc:
        raise ProductionProfileError("invalid_model_config", str(exc)) from exc
    reduced = config.reduced_detachment
    if (reduced.calibration_schema != _SYNTHETIC_SCHEMA or reduced.calibration_id != _SYNTHETIC_CALIBRATION_ID):
        raise ProductionProfileError("calibration_declaration_refused", "only the known native synthetic test fixture is admitted")
    # Canonical exact comparison retains native holder/scalar types and signed zero.
    # The caller config is never modified; the temporary disabled spec is comparison-only.
    try:
        actual = _canonical(native.np, dataclasses.replace(wake, enabled=False), "synthetic_wake_defaults")
        defaults = _canonical(native.np, VegetationWakeSpec(), "native_wake_defaults")
    except (ValueError, TypeError) as exc:
        if isinstance(exc, ProductionProfileError):
            raise
        raise ProductionProfileError("invalid_native_wake_spec", "native wake default comparison refused") from exc
    if actual != defaults:
        raise ProductionProfileError("unsupported_mode", "synthetic interface requires exact native default wake/windward coefficients")
    for group, name, wanted in _REQUIRED:
        value = getattr(getattr(config, group), name)
        if type(value) is not str or value != wanted:
            raise ProductionProfileError("unsupported_mode", f"{group}.{name} requires {wanted!r}")
    for group, name in _REFUSED_UNLESS_FALSE:
        if group == "vegetation_wake":
            continue
        if getattr(getattr(config, group), name) is not False:
            raise ProductionProfileError("unsupported_mode", f"{group}.{name} remains refused")
    return config


def _check_envelope(native: SimpleNamespace, record: Any) -> None:
    if type(record) is not native.WindForcingRecordSpec:
        raise ProductionProfileError("wrong_native_type",
                                     f"wind_envelope_record is {_type_name(record)}; expected exact native WindForcingRecordSpec")
    if type(record.blocks) is not tuple or any(type(block) is not native.WindBlock for block in record.blocks):
        raise ProductionProfileError("wrong_native_type",
                                     "wind_envelope_record.blocks must be a tuple of exact native WindBlock")
    try:
        native.validate_wind_forcing_record(record)
    except (ValueError, TypeError) as exc:
        raise ProductionProfileError("invalid_wind_envelope", str(exc)) from exc


def _check_availability(native: SimpleNamespace, config: Any, fraction: Any) -> None:
    """Inspect only: the caller's array is never normalized, copied or replaced."""
    np = native.np
    if fraction is None:
        raise ProductionProfileError("invalid_availability", "initial_available_fraction must be supplied explicitly")
    if type(fraction) is not np.ndarray:
        raise ProductionProfileError("invalid_availability",
                                     f"initial_available_fraction is {_type_name(fraction)}; only host numpy.ndarray")
    if fraction.dtype != np.dtype(np.float64):
        raise ProductionProfileError("invalid_availability",
                                     f"initial_available_fraction dtype is {fraction.dtype.str}; only native float64")
    expected = (config.geometry.ny, config.geometry.nx, len(config.grain_classes.classes))
    if fraction.shape != expected:
        raise ProductionProfileError("invalid_availability",
                                     f"initial_available_fraction shape is {fraction.shape}; geometry/classes require {expected}")
    if not bool(np.isfinite(fraction).all()):
        raise ProductionProfileError("invalid_availability", "initial_available_fraction has non-finite entries")
    if not bool(((fraction >= 0.0) & (fraction <= 1.0)).all()):
        raise ProductionProfileError("invalid_availability", "initial_available_fraction has entries outside [0, 1]")


def build_dense_production_wind_config(profile: DenseProductionWindProfile, *, wind_envelope_record: Any,
                                       initial_available_fraction: Any) -> Any:
    """The actual native `SchedulerPhysicsConfig`, mapped only by the native builder. Configuration only;
    unqualified for transport."""
    native = _native()
    config = _check_profile(native, profile)
    _check_envelope(native, wind_envelope_record)
    _check_availability(native, config, initial_available_fraction)
    return _build_checked(native, config, wind_envelope_record, initial_available_fraction)


def _build_checked(native: SimpleNamespace, config: Any, wind_envelope_record: Any,
                   initial_available_fraction: Any) -> Any:
    try:
        scheduler = native.build_scheduler_physics_config(
            config, spatial_block_diagnostics_enabled=False,
            initial_available_fraction=initial_available_fraction,
            prescribed_external_shear=None, wind_record=wind_envelope_record)
    except (ValueError, TypeError) as exc:
        raise ProductionProfileError("native_builder_refused", str(exc)) from exc
    if type(scheduler) is not native.SchedulerPhysicsConfig:
        raise ProductionProfileError("wrong_native_type", f"native builder returned {_type_name(scheduler)}")
    if scheduler.initial_available_fraction is not initial_available_fraction:
        raise ProductionProfileError("availability_identity_lost",
                                     "native builder did not carry the caller's initial_available_fraction object")
    if scheduler.supported_external_shear_velocity_m_s is None:
        raise ProductionProfileError("unbounded_wind_envelope",
                                     "native builder could not derive a supported shear bound from wind_envelope_record")
    return scheduler



def build_dense_synthetic_vegetation_wake_config(profile: DenseSyntheticVegetationWakeProfile, *,
        wind_envelope_record: Any, initial_available_fraction: Any) -> Any:
    """Explicit small synthetic configuration path; native mapper only, not transport qualification."""
    native = _native()
    config = _check_synthetic_wake_profile(native, profile)
    _check_envelope(native, wind_envelope_record)
    _check_availability(native, config, initial_available_fraction)
    return _build_checked(native, config, wind_envelope_record, initial_available_fraction)


def _canonical(np: Any, value: Any, where: str) -> Any:
    """The existing dense-runner canonical form; its refusals are translated, never bypassed."""
    from masy.dense_runner import DenseRunnerError, _canonical_input

    try:
        return _canonical_input(np, value, where)
    except DenseRunnerError as exc:
        raise ProductionProfileError("uncanonical_input", getattr(exc, "detail", str(exc))) from exc


def audit_dense_production_wind_config(profile: DenseProductionWindProfile, config: Any, *,
                                       wind_envelope_record: Any, initial_available_fraction: Any) -> dict[str, Any]:
    """Rebuild through the native builder and refuse unless `config` is canonically identical in every field
    and carries the caller's availability object, then return a transparent record. `audit_sha256` is
    `masy.dense_bed._sha256` over the record without that key."""
    native = _native()
    if type(config) is not native.SchedulerPhysicsConfig:
        raise ProductionProfileError("wrong_native_type",
                                     f"config is {_type_name(config)}; expected exact native SchedulerPhysicsConfig")
    expected = build_dense_production_wind_config(profile, wind_envelope_record=wind_envelope_record,
                                                  initial_available_fraction=initial_available_fraction)
    return _audit_checked(native, profile, config, expected, wind_envelope_record, initial_available_fraction)


def audit_dense_synthetic_vegetation_wake_config(profile: DenseSyntheticVegetationWakeProfile, config: Any, *,
        wind_envelope_record: Any, initial_available_fraction: Any) -> dict[str, Any]:
    """Exact native rebuild and transparent synthetic-only marker/default-spec audit."""
    native = _native()
    if type(config) is not native.SchedulerPhysicsConfig:
        raise ProductionProfileError("wrong_native_type", "exact native SchedulerPhysicsConfig required")
    expected = build_dense_synthetic_vegetation_wake_config(profile,
        wind_envelope_record=wind_envelope_record, initial_available_fraction=initial_available_fraction)
    return _audit_checked(native, profile, config, expected, wind_envelope_record,
                          initial_available_fraction, synthetic_wake=True)


def _audit_checked(native: SimpleNamespace, profile: Any, config: Any, expected: Any,
        wind_envelope_record: Any, initial_available_fraction: Any, *, synthetic_wake: bool = False) -> dict[str, Any]:
    np = native.np
    scheduler = _canonical(np, expected, "expected_scheduler_config")
    supplied = _canonical(np, config, "scheduler_config")
    if supplied != scheduler:
        differing = sorted(name for name, item in scheduler["fields"].items() if supplied["fields"].get(name) != item)
        raise ProductionProfileError("scheduler_config_mismatch",
                                     f"supplied scheduler config differs from the native rebuild in field(s) {differing}")
    if config.initial_available_fraction is not initial_available_fraction:
        raise ProductionProfileError("availability_identity_lost",
                                     "supplied scheduler config does not carry the caller's initial_available_fraction object")

    from masy.dense_bed import _sha256

    model = profile.model_config
    reduced = model.reduced_detachment
    model_record = _canonical(np, native.config_to_dict(model), "model_config")
    availability = _canonical(np, initial_available_fraction, "initial_available_fraction")
    envelope = _canonical(np, wind_envelope_record, "wind_envelope_record")
    record = {
        "schema": AUDIT_SCHEMA,
        "qualification": QUALIFICATION,
        "transport_ready": False,
        "envelope_role": ENVELOPE_ROLE,
        "environment_and_source_lock": "not_claimed_caller_runtime_lock_responsibility",
        "factory_policy": {
            "factory_version": FACTORY_VERSION,
            "native_builder": _NATIVE_BUILDER,
            "builder_arguments": {"spatial_block_diagnostics_enabled": False, "prescribed_external_shear": None,
                                  "initial_available_fraction": "caller_supplied_identity_preserved",
                                  "wind_record": "caller_supplied_envelope_only"},
            "required": [[f"{group}.{name}", wanted] for group, name, wanted in _REQUIRED],
            "refused_unless_false": [f"{group}.{name}" for group, name in _REFUSED_UNLESS_FALSE],
            "unbounded_envelope": "refused",
            "reviewed_refused_for_synthetic": {"calibration_schema": _SYNTHETIC_SCHEMA,
                                               "calibration_id": _SYNTHETIC_CALIBRATION_ID},
        },
        "calibration_declaration": {
            "status": profile.calibration_status,
            "reference": profile.calibration_reference,
            "reduced_calibration_id": reduced.calibration_id,
            "reduced_calibration_schema": reduced.calibration_schema,
            "meaning": DECLARATION_MEANING,
        },
        "model_config": {"config_to_dict": model_record, "sha256": _sha256(model_record)},
        "availability": {"policy": AVAILABILITY_POLICY, "array": availability, "sha256": _sha256(availability)},
        "wind_envelope": {
            "record": envelope,
            "supported_external_shear_velocity_m_s": _canonical(
                np, expected.supported_external_shear_velocity_m_s, "supported_external_shear_velocity_m_s"),
            "sha256": _sha256(envelope),
        },
        "scheduler_config": {"canonical": scheduler, "sha256": _sha256(scheduler)},
    }
    if synthetic_wake:
        wake = _canonical(np, model.vegetation_wake, "native_synthetic_wake_spec")
        record["schema"] = SYNTHETIC_WAKE_AUDIT_SCHEMA
        record["qualification"] = "configuration_only_authored_synthetic_native_wake_unqualified_for_transport"
        record["factory_policy"]["factory_version"] = SYNTHETIC_WAKE_FACTORY_VERSION
        record["factory_policy"]["refused_unless_false"] = [
            f"{group}.{name}" for group, name in _REFUSED_UNLESS_FALSE if group != "vegetation_wake"]
        record["factory_policy"]["synthetic_wake_required"] = "enabled_is_True_and_exact_native_default_coefficients"
        record["synthetic_interface"] = {
            "feature": profile.interface_feature, "profile_type": _type_name(profile),
            "calibration": "authored_test_fixture_NOT_MEASURED_not_production_calibration",
            "configuration_scope": {"max_cells": 64, "classes": 1},
            "native_wake_spec": {"canonical": wake, "sha256": _sha256(wake)},
            "native_default_coefficients_only": True,
            "raster_binding": "not_supplied_to_factory; actual_bed_and_event_native_vegetation_bindings_required",
            "runtime_resource_admission": "not_observed_here; unchanged_DenseProductionLimits_and_check_process_required"}
    digest = _sha256(record)
    return {**record, "audit_sha256": digest}


# Closed first GPU configuration scope. This is a host-at-rest mapper, not
# a device/workspace reservation or physical constructor.
_GPU_ADMISSION = {
    "max_cells": 64, "classes": 1, "max_flat_cohorts": 4000, "max_hop_bins": 8,
    "max_wind_blocks": 8, "max_depth": 16, "max_nodes_per_root": 4096,
    "max_container_entries": 32, "max_string_utf8_bytes": 4096,
    "max_total_string_utf8_bytes_per_root": 32768, "integer_bits": 63,
    "availability_max_bytes": 512, "hop_array_max_bytes": 64,
}
# Actual89 native holder symbols AND their complete declared field inventories.
# These constant imports never come from caller metadata. A different native
# layout refuses even when its class name is unchanged. No native code is copied.
_GPU_NATIVE_LAYOUTS = (
    ('maple.core.parameters', 'ModelConfig', ('run_name', 'seed', 'output_dir', 'geometry', 'grain_classes', 'time_and_forcing', 'numerics', 'execution', 'outputs_and_diagnostics', 'topographic_commit', 'avalanching', 'aeolian_process_mode', 'prescribed_detachment', 'particle_transport', 'fluid_threshold', 'aerodynamic_detachment', 'reduced_detachment', 'detachment_source', 'impact_rebound_splash', 'wind_grain_feedback', 'process_scheduler', 'stationary_continuation', 'dormancy', 'physical_window_admission', 'transport_accuracy', 'topographic_wind', 'flow_separation', 'vegetation_wake', 'prescribed_external_shear', 'water_coupling', 'surface_creep')),
    ('maple.core.parameters.geometry', 'GeometrySpec', ('nx', 'ny', 'dx_m', 'dy_m', 'boundary_x', 'boundary_y', 'voxel_dz_m', 'bulk_density_kg_m3', 'active_layer_thickness_m', 'voxel_growth')),
    ('maple.core.parameters.grain_classes', 'GrainClassSet', ('classes',)),
    ('maple.core.parameters.time_and_forcing', 'TimeAndForcingSpec', ('outer_timestep_s', 'wind_block_duration_s')),
    ('maple.core.parameters.numerics', 'NumericsSpec', ('mass_resolution_kg', 'cohort_calendar_max_bytes', 'cohort_calendar_memory_safety_fraction', 'cohort_calendar_memory_override_confirmed')),
    ('maple.core.parameters.execution', 'ExecutionSpec', ('backend', 'device_id', 'deterministic_scatter')),
    ('maple.core.parameters.outputs_and_diagnostics', 'OutputsAndDiagnosticsSpec', ('visualization_cadence_s', 'conservation_cadence_s', 'exact_checkpoint_cadence_s', 'keyframe_cadence_s', 'enable_exact_reconstruction', 'blosc_cname', 'blosc_clevel', 'blosc_shuffle', 'viz_float32_elevation_change', 'viz_float32_tolerance_m', 'visualization_time_chunk', 'visualization_y_chunk', 'visualization_x_chunk', 'conservation_time_chunk', 'exact_event_chunk', 'exact_checkpoint_chunk')),
    ('maple.core.parameters.topographic_commit', 'TopographicCommitSpec', ('commit_interval_s', 'pending_elevation_trigger_m', 'active_layer_turnover_fraction_trigger', 'dry_aeolian_commit_separation_enabled', 'dry_aeolian_geometric_height_threshold_m')),
    ('maple.core.parameters.avalanching', 'AvalanchingSpec', ('enabled', 'repose_angle_deg', 'convergence_angle_tolerance_deg', 'near_repose_margin_deg', 'max_iterations', 'mixture_rule')),
    ('maple.core.parameters.aeolian_process_mode', 'AeolianProcessModeSpec', ('mode',)),
    ('maple.core.parameters.prescribed_detachment', 'PrescribedDetachmentSpec', ('sources',)),
    ('maple.core.parameters.particle_transport', 'ParticleTransportSpec', ('routing_model', 'hop_distance_distribution', 'quantile_bins', 'hop_distance_by_class', 'max_active_cohorts', 'mobile_state_representation', 'direction_mode')),
    ('maple.core.parameters.fluid_threshold', 'FluidThresholdSpec', ('law', 'air_density_kg_m3', 'gravity_m_s2', 'air_dynamic_viscosity_pa_s', 'aerodynamic_coefficient', 'cohesion_parameter_kg_s2', 'bulk_mixture_rule', 'modifiers')),
    ('maple.core.parameters.aerodynamic_detachment', 'AerodynamicDetachmentSpec', ('law', 'reference_rate_kg_m2_s', 'excess_exponent')),
    ('maple.core.parameters.reduced_detachment', 'ReducedDetachmentSpec', ('calibration_id', 'calibration_schema', 'coefficient_c', 'impact_threshold_m_s', 'coefficient_c_range', 'impact_threshold_bracket_m_s', 'valid_shear_velocity_range_m_s', 'out_of_envelope_policy', 'activation_policy', 'calibrated_air_density_kg_m3', 'calibrated_gravity_m_s2', 'calibrated_grains', 'grain_property_relative_tolerance', 'composition_relative_tolerance')),
    ('maple.core.parameters.detachment_source', 'DetachmentSourceSpec', ('mode',)),
    ('maple.core.parameters.impact_rebound_splash', 'ImpactReboundSplashSpec', ('response', 'impact_assignment', 'gravity_m_s2', 'impact_state_bins', 'rebound', 'splash', 'launch_bins')),
    ('maple.core.parameters.wind_grain_feedback', 'WindGrainFeedbackSpec', ('transient', 'equilibrium', 'momentum')),
    ('maple.core.parameters.process_scheduler', 'ProcessSchedulerSpec', ('coupled_transition_duration_s', 'speed_change_absolute_threshold_m_s', 'speed_change_relative_threshold', 'direction_change_threshold_deg', 'recompute_on_active_layer_change', 'active_layer_change_relative_threshold', 'recompute_on_availability_change', 'recompute_on_topography_change', 'recompute_on_modifier_change')),
    ('maple.core.parameters.stationary_continuation', 'StationaryContinuationSpec', ('enabled', 'calibration_drift_relative_tolerance', 'calibration_drift_absolute_tolerance_kg', 'n_probe_slices', 'probe_interval_outer_steps', 'probe_rate_relative_tolerance', 'probe_rate_absolute_tolerance_kg_s', 'probe_impact_distribution_total_variation_tolerance', 'probe_stress_relative_tolerance', 'probe_stress_absolute_tolerance_pa', 'max_coast_interval_calibration_multiple')),
    ('maple.core.parameters.dormancy', 'AeolianDormancySpec', ('enabled',)),
    ('maple.core.parameters.physical_window_admission', 'PhysicalWindowAdmissionSpec', ('enabled', 'support_span_multiplier', 'rate_relative_tolerance', 'required_consecutive_agreeing_windows', 'probe_window_multiples')),
    ('maple.core.parameters.transport_accuracy', 'TransportAccuracySpec', ('mode', 'policy')),
    ('maple.core.parameters.topographic_wind', 'TopographicWindSpec', ('enabled', 'model', 'aerodynamic_roughness_m', 'filter', 'spectral', 'cache_normalized_modifier', 'debugging', 'update_policy')),
    ('maple.core.parameters.flow_separation', 'FlowSeparationSpec', ('enabled', 'model', 'separation_onset_angle_deg', 'max_envelope_descent_angle_deg', 'sampling_step_m', 'max_bubble_length_m', 'envelope_clearance_tolerance_m', 'descent_slope_tolerance', 'crest_arrival_angle_deg', 'crest_lookahead_m', 'launch_tangent_method', 'launch_tangent_baseline_m')),
    ('maple.core.parameters.vegetation_wake', 'VegetationWakeSpec', ('enabled', 'model', 'immediate_lee_shear_velocity_ratio', 'wake_efolding_distance_plant_heights', 'max_wake_length_plant_heights', 'sampling_step_m', 'cache_direction_dependent_fields', 'windward')),
    ('maple.core.parameters.prescribed_external_shear', 'PrescribedExternalShearSpec', ('enabled', 'multiplier', 'multiplier_map')),
    ('maple.core.parameters.water_coupling', 'WaterCouplingSpec', ('enabled', 'event_kind', 'depth_update_rule', 'adapter', 'prescribed')),
    ('maple.core.parameters.surface_creep', 'SurfaceCreepSpec', ('enabled', 'displacement_scale_m', 'slope_coefficient')),
    ('maple.core.boundaries', 'AxisBoundary', ('kind', 'inflow_flux_kg_m_s')),
    ('maple.core.parameters.voxel_growth', 'VoxelGrowthSpec', ('enabled', 'chunk_levels', 'max_levels', 'safety_fraction')),
    ('maple.core.parameters.grain_classes', 'GrainClass', ('class_id', 'diameter_m', 'particle_density_kg_m3', 'is_aggregate', 'diameter_min_m', 'diameter_max_m')),
    ('maple.core.parameters.prescribed_detachment', 'PrescribedDetachmentSource', ('source_type', 'units', 'start_time_s', 'end_time_s', 'region', 'rate_by_class_kg_m2_s', 'rate_array_kg_m2_s')),
    ('maple.core.parameters.particle_transport', 'HopDistanceClassSpec', ('median_hop_distance_m', 'geometric_stddev', 'representative_flight_speed_m_s', 'representative_impact_speed_m_s', 'representative_impact_angle_from_horizontal_deg')),
    ('maple.core.parameters.fluid_threshold', 'FluidThresholdModifiers', ('moisture_law', 'moisture_enabled', 'physical_crust_enabled', 'chemical_cementation_enabled', 'biological_binding_enabled')),
    ('maple.core.parameters.reduced_detachment', 'CalibratedGrain', ('class_id', 'diameter_m', 'particle_density_kg_m3', 'mass_fraction')),
    ('maple.core.parameters.impact_rebound_splash', 'ImpactStateBinSpec', ('speed_edges_m_s', 'incidence_angle_edges_deg')),
    ('maple.core.parameters.impact_rebound_splash', 'ReboundSpec', ('probability_law', 'probability_by_class', 'speed_retention_by_class', 'angle_from_horizontal_deg_by_class', 'gamma_s_per_m_by_class', 'rebound_energy_fraction_distribution', 'rebound_energy_fraction_mean', 'rebound_energy_fraction_std', 'rebound_energy_fraction_quadrature', 'rebound_angle_distribution', 'rebound_angle_quadrature')),
    ('maple.core.parameters.impact_rebound_splash', 'SplashSpec', ('law', 'energy_conversion_efficiency', 'momentum_conversion_efficiency', 'ejecta_speed_m_s_by_class', 'ejecta_angle_from_horizontal_deg_by_class', 'ejectability_exponent', 'hiding_exposure_exponent', 'cohesion_energy_j_kg_by_class', 'ejecta_speed_distribution', 'ejecta_speed_quadrature', 'ejecta_angle_distribution', 'ejecta_angle_quadrature', 'ejecta_speed_mean_law', 'ejecta_speed_mean_law_params')),
    ('maple.core.parameters.impact_rebound_splash', 'LaunchBinSpec', ('rebound_hop_bin_count', 'splash_hop_bin_count')),
    ('maple.core.parameters.wind_grain_feedback', 'TransientSpec', ('internal_slice_s', 'duration_s', 'allow_diagnostic_duration_override')),
    ('maple.core.parameters.wind_grain_feedback', 'EquilibriumSpec', ('max_iterations', 'relaxation_factor', 'stopping_profile', 'mobile_mass_relative_tolerance', 'mobile_mass_absolute_tolerance_kg', 'face_flux_relative_tolerance', 'face_flux_absolute_tolerance_kg_s', 'grain_stress_relative_tolerance', 'grain_stress_absolute_tolerance_pa', 'impact_distribution_relative_tolerance', 'impact_distribution_absolute_tolerance_kg_s', 'required_consecutive_converged_iterations')),
    ('maple.core.parameters.wind_grain_feedback', 'MomentumSpec', ('law', 'saltation_layer_air_speed_factor_by_class', 'grain_stress_attribution')),
    ('maple.core.parameters.topographic_wind', 'TopographicWindFilterSpec', ('kind', 'length_scale_m')),
    ('maple.core.parameters.topographic_wind', 'TopographicWindSpectralSpec', ('low_wavenumber_policy', 'high_wavenumber_filter')),
    ('maple.core.parameters.topographic_wind', 'TopographicWindDebuggingSpec', ('local_slope_mode',)),
    ('maple.core.parameters.topographic_wind', 'TopographicWindUpdatePolicySpec', ('mode', 'recompute_every_n_surface_changes', 'max_elevation_change_since_recompute_m')),
    ('maple.core.parameters.vegetation_wake', 'VegetationWindwardSpec', ('enabled', 'profile', 'support_plant_heights', 'max_shear_velocity_reduction', 'profile_exponent')),
    ('maple.core.parameters.scalar_fields', 'ScalarFieldSpec', ('builder', 'uniform', 'step', 'stripes', 'patch', 'checkerboard', 'gradient', 'random_patches', 'imported')),
    ('maple.core.parameters.water_coupling', 'WaterAdapterSpec', ('name', 'source_model', 'source_model_variant', 'length_unit', 'particle_density_kg_m3', 'density_kind', 'detachment_integration', 'boundary_export_derived', 'composition_feedback', 'active_layer_sensitivity_mm', 'class_max_detachment_depth_mm')),
    ('maple.core.parameters.water_coupling', 'PrescribedWaterSpec', ('sources',)),
    ('maple.core.parameters.prescribed_detachment', 'DetachmentRegion', ('row_start', 'row_end', 'col_start', 'col_end')),
    ('maple.core.parameters.impact_rebound_splash', 'ReboundEnergyFractionQuadratureSpec', ('rule', 'panels', 'nodes_per_panel')),
    ('maple.core.parameters.impact_rebound_splash', 'EjectaAngleQuadratureSpec', ('rule', 'angle_panels', 'angle_nodes_per_panel')),
    ('maple.core.parameters.impact_rebound_splash', 'EjectaSpeedQuadratureSpec', ('rule', 'speed_cutoff_over_mean', 'body_panels', 'body_nodes_per_panel', 'tail_nodes')),
    ('maple.core.parameters.impact_rebound_splash', 'EjectaSpeedMeanLawSpec', ('law', 'alpha_ej', 'a', 'impact_speed_scale', 'characteristic_diameter_m', 'launch_level_count', 'lowest_resolved_impact_speed_m_s')),
    ('maple.core.parameters.scalar_fields', 'UniformFieldSpec', ('value',)),
    ('maple.core.parameters.scalar_fields', 'StepFieldSpec', ('low_value', 'high_value', 'position_fraction', 'orientation_deg_from_x')),
    ('maple.core.parameters.scalar_fields', 'StripesFieldSpec', ('low_value', 'high_value', 'wavelength_m', 'duty_cycle', 'orientation_deg_from_x')),
    ('maple.core.parameters.scalar_fields', 'PatchFieldSpec', ('background_value', 'patch_value', 'x_min_m', 'x_max_m', 'y_min_m', 'y_max_m')),
    ('maple.core.parameters.scalar_fields', 'CheckerboardFieldSpec', ('low_value', 'high_value', 'cell_size_x_m', 'cell_size_y_m')),
    ('maple.core.parameters.scalar_fields', 'GradientFieldSpec', ('start_value', 'end_value', 'orientation_deg_from_x')),
    ('maple.core.parameters.scalar_fields', 'RandomPatchesFieldSpec', ('background_value', 'patch_value', 'n_patches', 'patch_radius_m')),
    ('maple.core.parameters.water_coupling', 'PrescribedWaterSource', ('source_type', 'units', 'start_time_s', 'end_time_s', 'region', 'removal_rate_by_class_kg_m2_s', 'deposition_rate_by_class_kg_m2_s', 'removal_rate_array_kg_m2_s', 'deposition_rate_array_kg_m2_s')),
    ('maple.aeolian.scheduler.state', 'SchedulerPhysicsConfig', ('geometry', 'grain_classes', 'fluid_threshold_spec', 'aerodynamic_detachment_spec', 'particle_transport', 'hop_bins', 'aeolian_process_mode', 'impact_rebound_splash_spec', 'feedback_spec', 'avalanching_spec', 'process_scheduler_spec', 'topographic_commit_spec', 'mass_resolution_kg', 'outer_timestep_s', 'detachment_source', 'prescribed_detachment', 'reduced_detachment', 'stationary_continuation_spec', 'dormancy_spec', 'water_coupling_spec', 'topographic_wind_spec', 'flow_separation_spec', 'vegetation_wake_spec', 'spatial_block_diagnostics_enabled', 'cohort_calendar_max_bytes', 'cohort_calendar_memory_safety_fraction', 'cohort_calendar_memory_override_confirmed', 'execution_backend', 'initial_available_fraction', 'prescribed_external_shear', 'supported_external_shear_velocity_m_s', 'transport_accuracy', 'physical_window_admission_spec', 'surface_creep_spec')),
    ('maple.aeolian.trajectories.quantile_bins', 'HopQuantileBins', ('hop_distance_m', 'weight', 'algorithm_version')),
    ('maple.core.parameters.wind_forcing', 'WindBlock', ('shear_velocity_m_s', 'direction_deg', 'duration_s', 'start_time_s', 'end_time_s')),
    ('maple.core.parameters.wind_forcing', 'WindForcingRecordSpec', ('blocks', 'source_path', 'source_sha256')),
)


def _gpu_native_layouts() -> dict[type, tuple[str, ...]]:
    import importlib

    result = {}
    for module, name, names in _GPU_NATIVE_LAYOUTS:
        cls = getattr(importlib.import_module(module), name)
        if not dataclasses.is_dataclass(cls) or tuple(f.name for f in dataclasses.fields(cls)) != names:
            raise ProductionProfileError("gpu_raw_layout", "native declared holder layout differs")
        result[cls] = names
    return result


def _gpu_raw_tree(native: SimpleNamespace, value: Any, where: str, *,
                  arrays: dict[str, tuple[tuple[int, ...], Any]] | None = None) -> None:
    """Bound a source-closed input tree before validation/serialization/mapper work.

    No data normalization, defaults, registry persistence or array copies. Native
    hints are read ONLY from exact code-owned classes in the closed layout table.
    Repeated aliases count again, matching later canonical/asdict expansion cost.
    """
    import math
    import types
    import typing
    from maple.core.boundaries import BoundaryKind

    layouts = _gpu_native_layouts()
    arrays = {} if arrays is None else arrays
    active: set[int] = set()
    hints: dict[type, dict[str, Any]] = {}
    nodes = text_bytes = 0
    np = native.np

    def refuse(detail: str) -> None:
        raise ProductionProfileError("gpu_raw_admission", detail)

    def matches(item: Any, hint: Any) -> bool:
        if hint is Any:
            return True  # the closed value walk below still refuses unknown values
        origin, args = typing.get_origin(hint), typing.get_args(hint)
        if origin in (typing.Union, types.UnionType):
            return any(matches(item, member) for member in args)
        if origin in (tuple, dict):
            return type(item) is origin
        if hint is float:
            return type(item) in (float, int)
        if hint in (str, bool, int, type(None), np.ndarray, BoundaryKind) or hint in layouts:
            return type(item) is hint
        return False

    def walk(item: Any, label: str, hint: Any, depth: int) -> None:
        nonlocal nodes, text_bytes
        nodes += 1
        if nodes > _GPU_ADMISSION["max_nodes_per_root"] or depth > _GPU_ADMISSION["max_depth"]:
            refuse("input node/depth cap exceeded")
        if not matches(item, hint):
            refuse("input field type differs at " + label)
        if typing.get_origin(hint) in (typing.Union, types.UnionType):
            hint = next(member for member in typing.get_args(hint) if matches(item, member))
        kind = type(item)
        if label in arrays and kind is not np.ndarray:
            refuse("native array field required at " + label)
        if item is None or kind is bool:
            return
        if kind is int:
            if not -(1 << 63) < item < (1 << 63):
                refuse("plain integer exceeds bounded raw scope")
            return
        if kind is float:
            if not math.isfinite(item):
                refuse("nonfinite raw scalar")
            return
        if kind is str:
            if len(item) > _GPU_ADMISSION["max_string_utf8_bytes"]:
                refuse("raw text cap exceeded")
            try:
                size = len(item.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise ProductionProfileError("gpu_raw_admission", "raw text is not UTF-8") from exc
            text_bytes += size
            if size > _GPU_ADMISSION["max_string_utf8_bytes"] or text_bytes > _GPU_ADMISSION["max_total_string_utf8_bytes_per_root"]:
                refuse("raw UTF-8 text budget exceeded")
            return
        if kind is BoundaryKind:
            if not any(item is member for member in BoundaryKind):
                refuse("unknown native boundary member")
            return
        if kind is np.ndarray:
            contract = arrays.get(label)
            if contract is None:
                refuse("raw array is unsupported at " + label)
            shape, identity = contract
            if (item.dtype != np.dtype(np.float64) or item.shape != shape
                    or not item.flags.c_contiguous or item.nbytes > 8 * math.prod(shape)
                    or (identity is not None and item is not identity)):
                refuse("raw array header/layout/identity differs at " + label)
            if not bool(np.isfinite(item).all()):
                refuse("raw array has nonfinite entries")
            return
        if kind not in (tuple, dict) and kind not in layouts:
            refuse("unsupported raw input type")
        key = id(item)
        if key in active:
            refuse("cyclic raw input")
        active.add(key)
        try:
            if kind is tuple:
                if len(item) > _GPU_ADMISSION["max_container_entries"]:
                    refuse("raw tuple cap exceeded")
                args = typing.get_args(hint)
                if args and args[-1] is not Ellipsis and len(args) != len(item):
                    refuse("fixed native tuple length differs")
                for index, child in enumerate(item):
                    expected = Any if not args else args[0] if args[-1] is Ellipsis else args[index]
                    walk(child, label + "[" + str(index) + "]", expected, depth + 1)
            elif kind is dict:
                if len(item) > _GPU_ADMISSION["max_container_entries"]:
                    refuse("raw mapping cap exceeded")
                args = typing.get_args(hint)
                expected = args[1] if args else Any
                for name, child in item.items():
                    if type(name) is not str:
                        refuse("raw mapping keys must be plain text")
                    walk(name, label + ".key", str, depth + 1)
                    walk(child, label + ".value", expected, depth + 1)
            else:
                names = layouts[kind]
                actual = vars(item)
                if len(actual) != len(names) or set(actual) != set(names):
                    refuse("extra or absent native instance fields at " + label)
                if kind not in hints:
                    hints[kind] = typing.get_type_hints(kind)
                for name in names:
                    walk(actual[name], label + "." + name, hints[kind][name], depth + 1)
        finally:
            active.remove(key)

    walk(value, where, type(value), 0)


def _gpu_raw_admission(native: SimpleNamespace, profile: Any, record: Any, fraction: Any) -> Any:
    if type(profile) is not DenseProductionGpuWindProfile:
        raise ProductionProfileError("wrong_native_type", "exact DenseProductionGpuWindProfile required")
    if len(vars(profile)) != 3 or set(vars(profile)) != {"model_config", "calibration_status", "calibration_reference"}:
        raise ProductionProfileError("gpu_raw_layout", "exact present GPU declaration fields required")
    config = profile.model_config
    if type(config) is not native.ModelConfig or type(record) is not native.WindForcingRecordSpec:
        raise ProductionProfileError("wrong_native_type", "exact native model and wind record required")
    for name in ("calibration_status", "calibration_reference"):
        value = getattr(profile, name)
        if type(value) is not str:
            raise ProductionProfileError("invalid_calibration_declaration", "plain bounded declaration text required")
        _gpu_raw_tree(native, value, name)
    _gpu_raw_tree(native, config, "model")
    _gpu_raw_tree(native, record, "envelope")
    g, p = config.geometry, config.particle_transport
    if (type(g.ny) is not int or type(g.nx) is not int or not 1 <= g.ny <= 64
            or not 1 <= g.nx <= 64 or g.ny * g.nx > 64
            or len(config.grain_classes.classes) != 1):
        raise ProductionProfileError("gpu_configuration_scope", "at most64cells and exactly1native class required")
    if (type(p.quantile_bins) is not int or not 1 <= p.quantile_bins <= 8
            or type(p.max_active_cohorts) is not int or not 1 <= p.max_active_cohorts <= 4000
            or p.mobile_state_representation != "flat"):
        raise ProductionProfileError("gpu_configuration_scope", "bounded flat cohort/bin profile required")
    if not 1 <= len(record.blocks) <= 8:
        raise ProductionProfileError("gpu_configuration_scope", "one to8wind-envelope blocks required")
    np = native.np
    shape = (g.ny, g.nx, 1)
    if (type(fraction) is not np.ndarray or fraction.dtype != np.dtype(np.float64)
            or fraction.shape != shape or not fraction.flags.c_contiguous or fraction.nbytes > 512):
        raise ProductionProfileError("invalid_availability", "bounded contiguous host float64 availability required")
    _check_availability(native, config, fraction)
    return config


def _gpu_scheduler_admission(native: SimpleNamespace, scheduler: Any, config: Any, fraction: Any) -> None:
    if type(scheduler) is not native.SchedulerPhysicsConfig:
        raise ProductionProfileError("wrong_native_type", "exact native SchedulerPhysicsConfig required")
    bins = config.particle_transport.quantile_bins
    _gpu_raw_tree(native, scheduler, "scheduler", arrays={
        "scheduler.initial_available_fraction": ((config.geometry.ny, config.geometry.nx, 1), fraction),
        "scheduler.hop_bins.hop_distance_m": ((1, bins), None),
        "scheduler.hop_bins.weight": ((1, bins), None),
    })


# Exact authored native100um fixture policy, from the immutable native
# tests.phase22b.test_reduced_integration SYNTHETIC_FIXTURE_CALIBRATION and
# ReducedDetachmentSpec native defaults. The existing CPU fixture explicitly
# strengthens native fixture 'warn' to 'reject'; no measured calibration claim.
_GPU_SYNTHETIC_CALIBRATION = {
    "calibration_id": _SYNTHETIC_CALIBRATION_ID, "calibration_schema": _SYNTHETIC_SCHEMA,
    "coefficient_c": 3.2300485895517235, "impact_threshold_m_s": 0.281,
    "coefficient_c_range": (3.227028359170264, 3.2331553271948033),
    "impact_threshold_bracket_m_s": (0.280, 0.282),
    "valid_shear_velocity_range_m_s": (0.4, 0.6), "out_of_envelope_policy": "reject",
    "activation_policy": "impact_only", "calibrated_air_density_kg_m3": 1.225,
    "calibrated_gravity_m_s2": 9.80665, "grain_property_relative_tolerance": 1e-9,
    "composition_relative_tolerance": 1e-9,
}
_GPU_EXTRA_REQUIRED = (
    ("execution", "deterministic_scatter", True),
    ("topographic_wind", "enabled", True), ("flow_separation", "enabled", True),
    ("surface_creep", "enabled", True), ("avalanching", "enabled", True),
    ("water_coupling", "event_kind", "aeolian"),
    ("water_coupling", "depth_update_rule", "constant_depth"),
    ("physical_window_admission", "enabled", False),
    ("numerics", "cohort_calendar_max_bytes", None),
    ("numerics", "cohort_calendar_memory_safety_fraction", 0.5),
    ("numerics", "cohort_calendar_memory_override_confirmed", False),
)


def _check_gpu_profile_rules(native: SimpleNamespace, profile: Any, config: Any) -> Any:
    if profile.calibration_status != SYNTHETIC_TEST_FIXTURE:
        raise ProductionProfileError("invalid_calibration_declaration", "synthetic_test_fixture status required")
    for group, name, wanted in _GPU_EXTRA_REQUIRED:
        value = getattr(getattr(config, group), name)
        if type(value) is not type(wanted) or value != wanted:
            raise ProductionProfileError("unsupported_mode", "GPU configuration requires " + group + "." + name)
    if config.geometry.voxel_growth.enabled is not False:
        raise ProductionProfileError("unsupported_mode", "voxel growth is outside this GPU configuration profile")
    if config.prescribed_detachment.sources or config.water_coupling.prescribed.sources:
        raise ProductionProfileError("unsupported_mode", "prescribed source profiles remain unsupported")
    reduced = config.reduced_detachment
    for name, wanted in _GPU_SYNTHETIC_CALIBRATION.items():
        value = getattr(reduced, name)
        if type(value) is not type(wanted) or value != wanted:
            raise ProductionProfileError("calibration_declaration_refused", "known native synthetic calibration required: " + name)
    if len(reduced.calibrated_grains) != 1:
        raise ProductionProfileError("calibration_declaration_refused", "exact native synthetic calibrated grain required")
    grain = reduced.calibrated_grains[0]
    if (grain.class_id != "sand" or type(grain.diameter_m) is not float or grain.diameter_m != 1e-4
            or type(grain.particle_density_kg_m3) is not float or grain.particle_density_kg_m3 != 2650.0
            or type(grain.mass_fraction) is not float or grain.mass_fraction != 1.0):
        raise ProductionProfileError("calibration_declaration_refused", "exact native synthetic100um calibrated grain required")
    # Native reduced-source dormancy cannot prove a skip (dormancy.py548-566),
    # and stationary continuation is coupled-only. Keep those inert native
    # defaults and the caller's actual values; no flags or physics are rewritten.
    try:
        return _check_profile_rules(native, profile, _GPU_REQUIRED)
    except OverflowError as exc:
        raise ProductionProfileError("invalid_model_config", "native config numeric operation refused") from exc


def build_dense_production_gpu_wind_config(profile: DenseProductionGpuWindProfile, *,
        wind_envelope_record: Any, initial_available_fraction: Any) -> Any:
    """Map the original bounded cupy-labelled native model, host-at-rest only.

    This allocates small native NumPy/JAX hop-bin configuration arrays, never
    a GPU context/bed/owner/pool or transport permission.
    """
    native = _native()
    config = _gpu_raw_admission(native, profile, wind_envelope_record, initial_available_fraction)
    _check_gpu_profile_rules(native, profile, config)
    _check_envelope(native, wind_envelope_record)
    return _build_checked(native, config, wind_envelope_record, initial_available_fraction)


def audit_dense_production_gpu_wind_config(profile: DenseProductionGpuWindProfile, config: Any, *,
        wind_envelope_record: Any, initial_available_fraction: Any) -> dict[str, Any]:
    """Full native rebuild/every-field audit; bounded before any expansion.

    Scheduler-affecting drift refuses. New valid metadata can leave the mapped
    scheduler equivalent, but always changes the complete recorded identity.
    """
    native = _native()
    model = _gpu_raw_admission(native, profile, wind_envelope_record, initial_available_fraction)
    _gpu_scheduler_admission(native, config, model, initial_available_fraction)
    expected = build_dense_production_gpu_wind_config(profile,
        wind_envelope_record=wind_envelope_record, initial_available_fraction=initial_available_fraction)
    record = _audit_checked(native, profile, config, expected, wind_envelope_record, initial_available_fraction)
    from masy.dense_bed import _sha256

    record.pop("audit_sha256")
    record["schema"] = GPU_AUDIT_SCHEMA
    record["qualification"] = "configuration_only_authored_synthetic_production_gpu_unqualified_for_transport"
    policy = record["factory_policy"]
    policy["factory_version"] = GPU_FACTORY_VERSION
    policy["required"] = [[group + "." + name, wanted] for group, name, wanted in _GPU_REQUIRED]
    policy["gpu_required"] = [[group + "." + name, wanted] for group, name, wanted in _GPU_EXTRA_REQUIRED]
    policy["raw_admission"] = dict(_GPU_ADMISSION)
    policy["native_holder_layouts"] = [[module, name, list(names)] for module, name, names in _GPU_NATIVE_LAYOUTS]
    policy["unsupported"] = ["calendar_storage", "physical_window_admission", "voxel_growth",
                             "prescribed_detachment_sources", "prescribed_water_sources",
                             "programmatic_model_arrays", "noncontiguous_availability", "unknown_instance_fields"]
    policy["inert_native_acceleration_defaults"] = "preserved; reduced source refuses dormancy proof; stationary path coupled_only"
    record["synthetic_gpu_interface"] = {
        "profile_type": _type_name(profile), "calibration": "native100um_authored_fixture_NOT_MEASURED_production_calibration_pending",
        "required_native_calibration": _canonical(native.np, _GPU_SYNTHETIC_CALIBRATION, "required_native_calibration"),
        "calibrated_grain": {"class_id": "sand", "diameter_m": 1e-4, "particle_density_kg_m3": 2650.0, "mass_fraction": 1.0},
        "placement": "host_numpy_float64_at_rest; native_execution_backend_cupy_recorded; no_device_placement_or_construction",
        "availability": "same_caller_object_and_flags; later_explicit_owner_conversion_required",
        "runtime_resources": "not_observed_here; no_owner_or_execution_admission",
    }
    return {**record, "audit_sha256": _sha256(record)}
