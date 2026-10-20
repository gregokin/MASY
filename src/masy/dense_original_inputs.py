"""Closed, bounded original-input artifact for the default Dense CPU profile.

This passive prerequisite restores typed inputs, never a bed, runner or event.
Current physical sediment remains the native exact-checkpoint codec's authority.
Native imports are lazy; the finite registry below is code-owned, never supplied
by a payload. Original inventory is an audit anchor, not validated genesis mass
until the later retained native genesis checkpoint is available.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
import math
import os
import re
import stat
from pathlib import Path
from types import ModuleType, SimpleNamespace, UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

SCHEMA = "masy.dense_original_inputs.v1"
PROFILE = "default_dense_numpy_v1"
_ROOT_FIELDS = (
    "capsule",
    "schedule",
    "wind",
    "water",
    "control",
    "runner_limits",
    "initial_inventory_kg",
    "static_parts",
    "static_parts_sha256",
    "static_record",
    "static_sha256",
    "source_identity",
    "recipes",
    "content_sha256",
)
_RUNNER_FIELDS = {
    "_bed",
    "_schedule",
    "_wind",
    "_water",
    "_control",
    "_limits",
    "_production",
    "_static_parts",
    "_static_parts_sha256",
    "_static_record",
    "_static_sha256",
    "_n_classes",
    "_inventory_elements",
    "_inventory0",
    "_pool",
    "_soil",
    "_routing",
    "_ledger",
    "_next",
    "_n_accepted",
    "_totals",
    "_receipts",
    "_stop_receipt",
    "_active_evidence",
    "_accepted_bed",
    "_accepted_private",
}
_RECIPES = {
    "column": "native_fixed_ksat",
    "sediment": "native_plot1_defaults",
    "hop_bins": "native_quantile_bins",
    "rainfall": "native_resolved_multiplier",
}
_ANCHORS = (
    "geometry",
    "grain_classes",
    "topographic_commit_spec",
    "avalanching_spec",
    "initial_available_fraction",
    "context",
)
_DTYPES = {"<f8", "<f4", "<i8", "<i4", "<u8", "<u4", "|u1", "|i1", "|b1"}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ARRAY_NAME = re.compile(r"a_[0-9]{4}\.npy\Z")
_LIMIT_NAMES = (
    "max_numeric_bytes",
    "max_serialized_bytes",
    "max_metadata_bytes",
    "max_files",
    "max_header_bytes",
    "max_nodes",
    "max_depth",
    "max_cells",
    "max_classes",
    "max_voxels",
    "max_cohorts",
    "max_events",
)


class OriginalInputsError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code, self.detail = code, detail
        super().__init__(f"{code}: {detail}")


class OriginalPackPublicationUncertain(OSError):
    """The manifest link may exist; publication/durability is unconfirmed."""


@dataclasses.dataclass(frozen=True)
class OriginalPackLimits:
    max_numeric_bytes: int = 64 << 20
    max_serialized_bytes: int = 72 << 20
    max_metadata_bytes: int = 4 << 20
    max_files: int = 256
    max_header_bytes: int = 4096
    max_nodes: int = 100000
    max_depth: int = 24
    max_cells: int = 64
    max_classes: int = 6
    max_voxels: int = 16
    max_cohorts: int = 16000
    max_events: int = 8

    def __post_init__(self):
        _layout(self, OriginalPackLimits, _LIMIT_NAMES)
        ceilings = (
            64 << 20,
            72 << 20,
            4 << 20,
            256,
            4096,
            100000,
            24,
            64,
            6,
            16,
            16000,
            8,
        )
        for name, ceiling in zip(_LIMIT_NAMES, ceilings, strict=True):
            if (
                type(getattr(self, name)) is not int
                or not 1 <= getattr(self, name) <= ceiling
            ):
                _fail(
                    "invalid_limit", f"{name} must be a plain integer in [1,{ceiling}]"
                )


@dataclasses.dataclass(frozen=True)
class OriginalInputs:
    capsule: Any
    schedule: Any
    wind: Any
    water: Any
    control: Any
    runner_limits: Any
    initial_inventory_kg: Any
    static_parts: dict
    static_parts_sha256: str
    static_record: dict
    static_sha256: str
    source_identity: dict
    recipes: dict
    content_sha256: str


def _fail(code, detail):
    raise OriginalInputsError(code, detail)


def _layout(value, cls, names):
    if (
        type(value) is not cls
        or not dataclasses.is_dataclass(value)
        or tuple(f.name for f in dataclasses.fields(value)) != tuple(names)
        or set(vars(value)) != set(names)
    ):
        _fail(
            "unsupported_layout",
            f"closed declared/instance fields required: {cls.__name__}",
        )


def _keys(value, names, where):
    if (
        type(value) is not dict
        or any(type(key) is not str for key in value)
        or set(value) != set(names)
    ):
        _fail("metadata_fields", f"{where}: unknown or missing fields")


def _source_identity(value):
    _keys(value, ("kind", "masy", "maple", "syrup"), "source identity")
    if (
        type(value["kind"]) is not str
        or value["kind"] != "source_manifest_prototype"
        or any(
            type(value[k]) is not str or not _SHA.fullmatch(value[k])
            for k in ("masy", "maple", "syrup")
        )
    ):
        _fail(
            "source_identity",
            "stageA requires explicit prototype-only source SHA256 references",
        )


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _strict_json(data, limits):
    if len(data) > limits.max_metadata_bytes:
        _fail("metadata_cap", "metadata exceeds raw byte cap")

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                _fail("metadata_duplicate", "duplicate JSON field")
            value[key] = item
        return value

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda _: _fail("metadata_nonfinite", "nonfinite JSON"),
        )
    except OriginalInputsError:
        raise
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise OriginalInputsError("metadata_invalid", "invalid bounded JSON") from exc
    nodes = 0

    def visit(v, depth):
        nonlocal nodes
        nodes += 1
        if nodes > limits.max_nodes or depth > limits.max_depth:
            _fail("metadata_cap", "metadata node/depth cap")
        if type(v) is dict:
            for k, item in v.items():
                if type(k) is not str or len(k) > 4096:
                    _fail("metadata_invalid", "invalid key")
                visit(item, depth + 1)
        elif type(v) is list:
            for item in v:
                visit(item, depth + 1)
        elif type(v) is str and len(v) > 4096:
            _fail("metadata_cap", "string cap")
        elif type(v) is int and v.bit_length() > 4096:
            _fail("metadata_cap", "integer cap")
        elif type(v) is float and not math.isfinite(v):
            _fail("metadata_nonfinite", "overflowed/nonfinite JSON number")

    visit(value, 0)
    return value


def _native():
    from maple.aeolian.flux.validation import validate_mobile_state_union
    from maple.aeolian.trajectories.quantile_bins import build_hop_quantile_bins
    from maple_syrup.infiltration import column_parameters
    from maple_syrup.rainfall import rainfall_field
    from maple_syrup.sediment_physics import plot1_sediment_parameters

    from masy import dense_bed, dense_checkpoint, dense_runner
    from masy.dense_schedule import DenseSchedule

    bed_native, cp = dense_checkpoint._native()
    n = SimpleNamespace(**{**vars(bed_native), **vars(dense_runner._native())})
    return SimpleNamespace(
        n=n,
        cp=cp,
        bed=dense_bed,
        bridge=dense_checkpoint,
        runner=dense_runner,
        schedule=DenseSchedule,
        pool_validate=validate_mobile_state_union,
        bins=build_hop_quantile_bins,
        column=column_parameters,
        rainfall=rainfall_field,
        sediment=plot1_sediment_parameters,
    )


# REGISTRY_LITERAL is generated once from exact audited sources, then reviewed.
_TYPES = (
    ("maple.core.boundaries", "BoundaryKind", (), True),
    ("maple.core.boundaries", "AxisBoundary", ("kind", "inflow_flux_kg_m_s"), False),
    (
        "maple.core.parameters.geometry",
        "GeometrySpec",
        (
            "nx",
            "ny",
            "dx_m",
            "dy_m",
            "boundary_x",
            "boundary_y",
            "voxel_dz_m",
            "bulk_density_kg_m3",
            "active_layer_thickness_m",
            "voxel_growth",
        ),
        False,
    ),
    (
        "maple.core.parameters.grain_classes",
        "GrainClass",
        (
            "class_id",
            "diameter_m",
            "particle_density_kg_m3",
            "is_aggregate",
            "diameter_min_m",
            "diameter_max_m",
        ),
        False,
    ),
    ("maple.core.parameters.grain_classes", "GrainClassSet", ("classes",), False),
    (
        "maple.core.parameters.grain_classes",
        "ResolvedGrainClassArrays",
        ("diameter_m", "particle_density_kg_m3"),
        False,
    ),
    (
        "maple.core.parameters.fluid_threshold",
        "FluidThresholdModifiers",
        (
            "moisture_law",
            "moisture_enabled",
            "physical_crust_enabled",
            "chemical_cementation_enabled",
            "biological_binding_enabled",
        ),
        False,
    ),
    (
        "maple.core.parameters.fluid_threshold",
        "FluidThresholdSpec",
        (
            "law",
            "air_density_kg_m3",
            "gravity_m_s2",
            "air_dynamic_viscosity_pa_s",
            "aerodynamic_coefficient",
            "cohesion_parameter_kg_s2",
            "bulk_mixture_rule",
            "modifiers",
        ),
        False,
    ),
    (
        "maple.core.parameters.aerodynamic_detachment",
        "AerodynamicDetachmentSpec",
        ("law", "reference_rate_kg_m2_s", "excess_exponent"),
        False,
    ),
    (
        "maple.core.parameters.particle_transport",
        "HopDistanceClassSpec",
        (
            "median_hop_distance_m",
            "geometric_stddev",
            "representative_flight_speed_m_s",
            "representative_impact_speed_m_s",
            "representative_impact_angle_from_horizontal_deg",
        ),
        False,
    ),
    (
        "maple.core.parameters.particle_transport",
        "ParticleTransportSpec",
        (
            "routing_model",
            "hop_distance_distribution",
            "quantile_bins",
            "hop_distance_by_class",
            "max_active_cohorts",
            "mobile_state_representation",
            "direction_mode",
        ),
        False,
    ),
    (
        "maple.aeolian.trajectories.quantile_bins",
        "HopQuantileBins",
        ("hop_distance_m", "weight", "algorithm_version"),
        False,
    ),
    (
        "maple.core.parameters.aeolian_process_mode",
        "AeolianProcessModeSpec",
        ("mode",),
        False,
    ),
    (
        "maple.core.parameters.aeolian_process_mode",
        "ResolvedProcessMode",
        (
            "requested_mode",
            "impact_response",
            "wind_grain_feedback",
            "rebound",
            "splash",
            "impact_generated_launch",
            "required_detachment_source",
            "coupled_iteration",
        ),
        False,
    ),
    (
        "maple.core.parameters.impact_rebound_splash",
        "ImpactStateBinSpec",
        ("speed_edges_m_s", "incidence_angle_edges_deg"),
        False,
    ),
    (
        "maple.core.parameters.impact_rebound_splash",
        "ReboundSpec",
        (
            "probability_law",
            "probability_by_class",
            "speed_retention_by_class",
            "angle_from_horizontal_deg_by_class",
            "gamma_s_per_m_by_class",
            "rebound_energy_fraction_distribution",
            "rebound_energy_fraction_mean",
            "rebound_energy_fraction_std",
            "rebound_energy_fraction_quadrature",
            "rebound_angle_distribution",
            "rebound_angle_quadrature",
        ),
        False,
    ),
    (
        "maple.core.parameters.impact_rebound_splash",
        "ReboundEnergyFractionQuadratureSpec",
        ("rule", "panels", "nodes_per_panel"),
        False,
    ),
    (
        "maple.core.parameters.impact_rebound_splash",
        "EjectaSpeedQuadratureSpec",
        (
            "rule",
            "speed_cutoff_over_mean",
            "body_panels",
            "body_nodes_per_panel",
            "tail_nodes",
        ),
        False,
    ),
    (
        "maple.core.parameters.impact_rebound_splash",
        "SplashSpec",
        (
            "law",
            "energy_conversion_efficiency",
            "momentum_conversion_efficiency",
            "ejecta_speed_m_s_by_class",
            "ejecta_angle_from_horizontal_deg_by_class",
            "ejectability_exponent",
            "hiding_exposure_exponent",
            "cohesion_energy_j_kg_by_class",
            "ejecta_speed_distribution",
            "ejecta_speed_quadrature",
            "ejecta_angle_distribution",
            "ejecta_angle_quadrature",
            "ejecta_speed_mean_law",
            "ejecta_speed_mean_law_params",
        ),
        False,
    ),
    (
        "maple.core.parameters.impact_rebound_splash",
        "EjectaAngleQuadratureSpec",
        ("rule", "angle_panels", "angle_nodes_per_panel"),
        False,
    ),
    (
        "maple.core.parameters.impact_rebound_splash",
        "EjectaSpeedMeanLawSpec",
        (
            "law",
            "alpha_ej",
            "a",
            "impact_speed_scale",
            "characteristic_diameter_m",
            "launch_level_count",
            "lowest_resolved_impact_speed_m_s",
        ),
        False,
    ),
    (
        "maple.core.parameters.impact_rebound_splash",
        "LaunchBinSpec",
        ("rebound_hop_bin_count", "splash_hop_bin_count"),
        False,
    ),
    (
        "maple.core.parameters.impact_rebound_splash",
        "ImpactReboundSplashSpec",
        (
            "response",
            "impact_assignment",
            "gravity_m_s2",
            "impact_state_bins",
            "rebound",
            "splash",
            "launch_bins",
        ),
        False,
    ),
    (
        "maple.core.parameters.wind_grain_feedback",
        "TransientSpec",
        ("internal_slice_s", "duration_s", "allow_diagnostic_duration_override"),
        False,
    ),
    (
        "maple.core.parameters.wind_grain_feedback",
        "EquilibriumSpec",
        (
            "max_iterations",
            "relaxation_factor",
            "stopping_profile",
            "mobile_mass_relative_tolerance",
            "mobile_mass_absolute_tolerance_kg",
            "face_flux_relative_tolerance",
            "face_flux_absolute_tolerance_kg_s",
            "grain_stress_relative_tolerance",
            "grain_stress_absolute_tolerance_pa",
            "impact_distribution_relative_tolerance",
            "impact_distribution_absolute_tolerance_kg_s",
            "required_consecutive_converged_iterations",
        ),
        False,
    ),
    (
        "maple.core.parameters.wind_grain_feedback",
        "MomentumSpec",
        (
            "law",
            "saltation_layer_air_speed_factor_by_class",
            "grain_stress_attribution",
        ),
        False,
    ),
    (
        "maple.core.parameters.wind_grain_feedback",
        "WindGrainFeedbackSpec",
        ("transient", "equilibrium", "momentum"),
        False,
    ),
    (
        "maple.core.parameters.avalanching",
        "AvalanchingSpec",
        (
            "enabled",
            "repose_angle_deg",
            "convergence_angle_tolerance_deg",
            "near_repose_margin_deg",
            "max_iterations",
            "mixture_rule",
        ),
        False,
    ),
    (
        "maple.core.parameters.process_scheduler",
        "ProcessSchedulerSpec",
        (
            "coupled_transition_duration_s",
            "speed_change_absolute_threshold_m_s",
            "speed_change_relative_threshold",
            "direction_change_threshold_deg",
            "recompute_on_active_layer_change",
            "active_layer_change_relative_threshold",
            "recompute_on_availability_change",
            "recompute_on_topography_change",
            "recompute_on_modifier_change",
        ),
        False,
    ),
    (
        "maple.core.parameters.topographic_commit",
        "TopographicCommitSpec",
        (
            "commit_interval_s",
            "pending_elevation_trigger_m",
            "active_layer_turnover_fraction_trigger",
            "dry_aeolian_commit_separation_enabled",
            "dry_aeolian_geometric_height_threshold_m",
        ),
        False,
    ),
    (
        "maple.core.parameters.detachment_source",
        "DetachmentSourceSpec",
        ("mode",),
        False,
    ),
    (
        "maple.core.parameters.stationary_continuation",
        "StationaryContinuationSpec",
        (
            "enabled",
            "calibration_drift_relative_tolerance",
            "calibration_drift_absolute_tolerance_kg",
            "n_probe_slices",
            "probe_interval_outer_steps",
            "probe_rate_relative_tolerance",
            "probe_rate_absolute_tolerance_kg_s",
            "probe_impact_distribution_total_variation_tolerance",
            "probe_stress_relative_tolerance",
            "probe_stress_absolute_tolerance_pa",
            "max_coast_interval_calibration_multiple",
        ),
        False,
    ),
    ("maple.core.parameters.dormancy", "AeolianDormancySpec", ("enabled",), False),
    (
        "maple.core.parameters.water_coupling",
        "PrescribedWaterSource",
        (
            "source_type",
            "units",
            "start_time_s",
            "end_time_s",
            "region",
            "removal_rate_by_class_kg_m2_s",
            "deposition_rate_by_class_kg_m2_s",
            "removal_rate_array_kg_m2_s",
            "deposition_rate_array_kg_m2_s",
        ),
        False,
    ),
    (
        "maple.core.parameters.water_coupling",
        "PrescribedWaterSpec",
        ("sources",),
        False,
    ),
    (
        "maple.core.parameters.water_coupling",
        "WaterAdapterSpec",
        (
            "name",
            "source_model",
            "source_model_variant",
            "length_unit",
            "particle_density_kg_m3",
            "density_kind",
            "detachment_integration",
            "boundary_export_derived",
            "composition_feedback",
            "active_layer_sensitivity_mm",
            "class_max_detachment_depth_mm",
        ),
        False,
    ),
    (
        "maple.core.parameters.water_coupling",
        "WaterCouplingSpec",
        ("enabled", "event_kind", "depth_update_rule", "adapter", "prescribed"),
        False,
    ),
    (
        "maple.core.parameters.topographic_wind",
        "TopographicWindFilterSpec",
        ("kind", "length_scale_m"),
        False,
    ),
    (
        "maple.core.parameters.topographic_wind",
        "TopographicWindSpectralSpec",
        ("low_wavenumber_policy", "high_wavenumber_filter"),
        False,
    ),
    (
        "maple.core.parameters.topographic_wind",
        "TopographicWindDebuggingSpec",
        ("local_slope_mode",),
        False,
    ),
    (
        "maple.core.parameters.topographic_wind",
        "TopographicWindUpdatePolicySpec",
        (
            "mode",
            "recompute_every_n_surface_changes",
            "max_elevation_change_since_recompute_m",
        ),
        False,
    ),
    (
        "maple.core.parameters.topographic_wind",
        "TopographicWindSpec",
        (
            "enabled",
            "model",
            "aerodynamic_roughness_m",
            "filter",
            "spectral",
            "cache_normalized_modifier",
            "debugging",
            "update_policy",
        ),
        False,
    ),
    (
        "maple.core.parameters.flow_separation",
        "FlowSeparationSpec",
        (
            "enabled",
            "model",
            "separation_onset_angle_deg",
            "max_envelope_descent_angle_deg",
            "sampling_step_m",
            "max_bubble_length_m",
            "envelope_clearance_tolerance_m",
            "descent_slope_tolerance",
            "crest_arrival_angle_deg",
            "crest_lookahead_m",
            "launch_tangent_method",
            "launch_tangent_baseline_m",
        ),
        False,
    ),
    (
        "maple.core.parameters.vegetation_wake",
        "VegetationWindwardSpec",
        (
            "enabled",
            "profile",
            "support_plant_heights",
            "max_shear_velocity_reduction",
            "profile_exponent",
        ),
        False,
    ),
    (
        "maple.core.parameters.vegetation_wake",
        "VegetationWakeSpec",
        (
            "enabled",
            "model",
            "immediate_lee_shear_velocity_ratio",
            "wake_efolding_distance_plant_heights",
            "max_wake_length_plant_heights",
            "sampling_step_m",
            "cache_direction_dependent_fields",
            "windward",
        ),
        False,
    ),
    (
        "maple.core.parameters.physical_window_admission",
        "PhysicalWindowAdmissionSpec",
        (
            "enabled",
            "support_span_multiplier",
            "rate_relative_tolerance",
            "required_consecutive_agreeing_windows",
            "probe_window_multiples",
        ),
        False,
    ),
    (
        "maple.core.parameters.voxel_growth",
        "VoxelGrowthSpec",
        ("enabled", "chunk_levels", "max_levels", "safety_fraction"),
        False,
    ),
    (
        "maple.core.types.cohort",
        "MobileCohortState",
        (
            "mass_kg",
            "grain_class_index",
            "position_x_m",
            "position_y_m",
            "velocity_m_s",
            "angle_rad",
            "launch_mode",
            "active_mask",
            "launch_time_s",
            "impact_time_s",
            "launch_bin_index",
            "hop_distance_m",
            "boundary_origin",
            "impact_speed_m_s",
            "impact_angle_from_horizontal_rad",
            "launch_air_horizontal_speed_m_s",
            "live_velocity_m_s",
            "live_travelled_m",
            "live_turn_displacement_m",
            "live_has_turned",
        ),
        False,
    ),
    (
        "maple.aeolian.scheduler.state",
        "SchedulerPhysicsConfig",
        (
            "geometry",
            "grain_classes",
            "fluid_threshold_spec",
            "aerodynamic_detachment_spec",
            "particle_transport",
            "hop_bins",
            "aeolian_process_mode",
            "impact_rebound_splash_spec",
            "feedback_spec",
            "avalanching_spec",
            "process_scheduler_spec",
            "topographic_commit_spec",
            "mass_resolution_kg",
            "outer_timestep_s",
            "detachment_source",
            "prescribed_detachment",
            "reduced_detachment",
            "stationary_continuation_spec",
            "dormancy_spec",
            "water_coupling_spec",
            "topographic_wind_spec",
            "flow_separation_spec",
            "vegetation_wake_spec",
            "spatial_block_diagnostics_enabled",
            "cohort_calendar_max_bytes",
            "cohort_calendar_memory_safety_fraction",
            "cohort_calendar_memory_override_confirmed",
            "execution_backend",
            "initial_available_fraction",
            "prescribed_external_shear",
            "supported_external_shear_velocity_m_s",
            "transport_accuracy",
            "physical_window_admission_spec",
            "surface_creep_spec",
        ),
        False,
    ),
    (
        "maple_syrup.sediment_bed",
        "BedContext",
        (
            "geometry",
            "grain_classes",
            "mass_resolution_kg",
            "topographic_commit_spec",
            "avalanching_spec",
            "initial_available_fraction",
            "depth_update_rule",
            "record_exchange_history",
            "window_max_journal_records",
            "deep_chunk_encoding",
            "deep_overlay_max_bytes",
        ),
        False,
    ),
    (
        "maple_syrup.sediment_bed",
        "TerrainReference",
        (
            "initial_full_elevation_m",
            "initial_committed_elevation_m",
            "export_receiver_full",
            "active_mask",
            "friction_factor",
            "dx_m",
            "allow_masked_nodata",
            "nodata_value",
            "allow_pit_storage",
            "allow_flat_storage",
        ),
        False,
    ),
    (
        "maple_syrup.infiltration",
        "ColumnParameters",
        (
            "model",
            "ksat_m_per_s",
            "suction_m",
            "drainage_parameter",
            "theta_sat",
            "soil_thickness_m",
            "storage_max_m",
            "lambda_m_per_s",
            "active_mask",
            "xp",
        ),
        False,
    ),
    ("maple_syrup.rainfall", "RainfallField", ("multiplier", "xp"), False),
    (
        "maple_syrup.sediment_physics",
        "SedimentPhysicsParameters",
        (
            "n_classes",
            "diameter_m",
            "raindrop_a",
            "raindrop_b",
            "raindrop_c",
            "raindrop_depth_attenuation_per_cm",
            "raindrop_max_depth_per_reference_s_m",
            "particle_density_kg_m3",
            "flow_detachment_depth_scale_m",
            "kinematic_viscosity_m2_s",
            "reference_interval_s",
            "recession_factor_per_reference_s",
            "ke_model",
            "ke_vegetation_form",
            "raindrop_composition_scaling",
            "distance_convention",
            "dstar_convention",
            "bagnold_depth_units",
            "sigma",
            "excess_density_kg_m3",
            "bagnold_density_scale",
            "dstar_const_per_m",
            "settling_velocity_m_s",
            "particle_mass_g",
            "xp",
            "d_diameter",
            "d_raindrop_a",
            "d_raindrop_b",
            "d_raindrop_c",
            "d_attenuation",
            "d_raindrop_max",
            "d_settling",
            "d_particle_mass_g",
        ),
        False,
    ),
    (
        "maple_syrup.sediment_event",
        "SedimentEventControl",
        (
            "storm",
            "sediment_courant_max",
            "max_transport_substeps",
            "commit",
            "force_final_commit",
            "max_commit_log",
            "max_rejections_recorded",
            "transport_scheme",
            "phase_bins",
            "transport_implementation",
            "phase_storage",
            "phase_max_nnz",
            "record_exchange_history",
            "deep_publication",
            "deep_publication_high_water_records",
        ),
        False,
    ),
    (
        "maple_syrup.storm",
        "StormControl",
        (
            "max_dt_s",
            "min_dt_s",
            "max_retries",
            "max_steps",
            "courant_max",
            "bisection_iterations",
            "root_tolerance_m",
            "implementation",
            "root_solver",
            "newton_max_iterations",
        ),
        False,
    ),
    (
        "maple_syrup.complete_event",
        "CompletionPolicy",
        (
            "max_depth_m",
            "surface_volume_m3",
            "max_unit_discharge_m2_s",
            "outlet_discharge_m3_s",
            "hold_s",
            "mobile_per_cell_class_kg",
            "terminal_storage_water",
            "deposit_mobile_at_flow_end",
        ),
        False,
    ),
    (
        "masy.dense_checkpoint",
        "DenseBoundaryInputs",
        (
            "case_record",
            "held_inputs",
            "context",
            "limits",
            "initialization_receipt",
            "prescribed_digests",
            "case_sha256",
            "binding_sha256",
        ),
        False,
    ),
    (
        "masy.dense_bed",
        "DenseBedLimits",
        ("host_memory_cap_bytes", "max_voxels_per_column", "stage_copy_max_bytes"),
        False,
    ),
    (
        "masy.dense_runner",
        "DenseRunnerLimits",
        (
            "max_events",
            "max_wind_accounting_steps",
            "max_storm_steps",
            "max_report_rows",
            "max_work_bytes",
            "max_receipt_bytes",
        ),
        False,
    ),
    ("masy.dense_runner", "DenseWindInputs", ("config", "initial_mobile_state"), False),
    (
        "masy.dense_runner",
        "DenseWaterInputs",
        (
            "terrain",
            "column",
            "field",
            "vegetation_cover_fraction",
            "sediment",
            "initial_soil_water_m",
        ),
        False,
    ),
    (
        "masy.dense_runner",
        "DenseRunnerControl",
        ("sediment_control", "completion_policy", "report_every_s"),
        False,
    ),
)


def _registry():
    result = {}
    for module, name, fields, enum in _TYPES:
        cls = getattr(
            importlib.import_module(module), name
        )  # code-owned constants only
        tag = module + ":" + name
        if not enum and tuple(f.name for f in dataclasses.fields(cls)) != fields:
            _fail("native_schema_changed", tag)
        result[tag] = (cls, fields, enum)
    return result


def _hint(value, hint, depth=0, *, budget):
    budget[0] -= 1
    if budget[0] < 0:
        _fail("metadata_cap", "aggregate native hint traversal cap")
    if depth > 24 or type(value) in (tuple, list, dict) and len(value) > 100000:
        return False
    if hint is Any:
        return True
    origin, args = get_origin(hint), get_args(hint)

    def check(v, h):
        return _hint(v, h, depth + 1, budget=budget)

    if origin in (Union, UnionType):
        return any(check(value, item) for item in args)
    if origin is Literal:
        return any(type(value) is type(item) and value == item for item in args)
    if origin in (tuple, list):
        if type(value) is not origin:
            return False
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            return all(check(item, args[0]) for item in value)
        return (
            all(check(item, args[0]) for item in value)
            if origin is list
            else len(value) == len(args)
            and all(check(v, h) for v, h in zip(value, args, strict=True))
        )
    if origin is dict:
        return type(value) is dict and all(
            check(k, args[0]) and check(v, args[1]) for k, v in value.items()
        )
    if hint is type(None):
        return value is None
    if hint is ModuleType:
        import numpy as np

        return value is np
    return type(value) is hint


def _check_class(value, entry, *, budget=None):
    budget = [100000] if budget is None else budget
    cls, fields, enum = entry
    if enum:
        if type(value) is not cls:
            _fail("unsupported_type", "enum type mismatch")
        return
    _layout(value, cls, fields)
    for name, hint in get_type_hints(cls).items():
        if name in fields and not _hint(vars(value)[name], hint, budget=budget):
            _fail("field_type", f"{cls.__name__}.{name} has unsupported exact type")


def _material(pack):
    _layout(pack, OriginalInputs, _ROOT_FIELDS)
    result = {
        name: getattr(pack, name) for name in _ROOT_FIELDS if name != "content_sha256"
    }
    result["schedule"] = pack.schedule.as_record()
    return result


class _Encoder:
    def __init__(self, limits, registry, *, hash_arrays, hint_budget=None):
        import numpy as np

        self.np, self.limits, self.registry, self.hash_arrays = (
            np,
            limits,
            registry,
            hash_arrays,
        )
        self.by_type = {entry[0]: (tag, entry) for tag, entry in registry.items()}
        self.arrays, self.array_ids, self.nodes, self.numeric = [], {}, 0, 0
        self.metadata = 0
        self.anchor_ids = {}
        self.hint_budget = [limits.max_nodes] if hint_budget is None else hint_budget

    def charge(self, size):
        self.metadata += size
        if self.metadata > self.limits.max_metadata_bytes:
            _fail("metadata_cap", "conservative metadata preflight")

    def node(self, value, depth=0, *, defining=None):
        self.nodes += 1
        self.charge(128)
        if self.nodes > self.limits.max_nodes or depth > self.limits.max_depth:
            _fail("metadata_cap", "input node/depth cap")
        if id(value) in self.anchor_ids and self.anchor_ids[id(value)] != defining:
            return {"t": "ref", "v": self.anchor_ids[id(value)]}
        if value is self.np:
            return {"t": "numpy"}
        if type(value) is self.np.ndarray:
            a = value
            dimension_cap = max(
                self.limits.max_cohorts,
                self.limits.max_cells + 2,
                self.limits.max_classes,
                self.limits.max_voxels,
            )
            if (
                a.dtype.str not in _DTYPES
                or not a.flags.c_contiguous
                or a.ndim > 4
                or any(n > dimension_cap for n in a.shape)
            ):
                _fail(
                    "array_layout", "only bounded contiguous numeric/bool NumPy arrays"
                )
            if id(a) not in self.array_ids:
                self.numeric += a.nbytes
                if (
                    self.numeric > self.limits.max_numeric_bytes
                    or len(self.arrays) >= self.limits.max_files
                ):
                    _fail("array_cap", "numeric byte/file cap before array hashing")
                index = len(self.arrays)
                self.array_ids[id(a)] = index
                self.arrays.append(a)
            index = self.array_ids[id(a)]
            return {"t": "array", "v": index}
        if type(value) in self.by_type:
            tag, entry = self.by_type[type(value)]
            _check_class(value, entry, budget=self.hint_budget)
            self.charge(12 * len(tag) + sum(12 * len(name) + 8 for name in entry[1]))
            if entry[2]:
                return {"t": "enum", "c": tag, "v": self.node(value.value, depth + 1)}
            return {
                "t": "class",
                "c": tag,
                "v": [
                    [name, self.node(vars(value)[name], depth + 1)] for name in entry[1]
                ],
            }
        if type(value) in (dict, list, tuple):
            if len(value) > self.limits.max_nodes - self.nodes:
                _fail("metadata_cap", "container cap")
            if type(value) is dict:
                for key in value:
                    if type(key) is not str or len(key) > 4096:
                        _fail(
                            "unsupported_type",
                            "plain bounded string mapping keys required",
                        )
                    self.charge(12 * len(key) + 8)
                return {
                    "t": "dict",
                    "v": [[k, self.node(item, depth + 1)] for k, item in value.items()],
                }
            return {
                "t": "list" if type(value) is list else "tuple",
                "v": [self.node(item, depth + 1) for item in value],
            }
        if type(value) is float:
            if not math.isfinite(value):
                _fail("unsupported_scalar", "nonfinite float")
            return {"t": "float", "v": value.hex()}
        if value is None or type(value) is bool:
            return {"t": "none"} if value is None else {"t": "bool", "v": value}
        if type(value) is int and value.bit_length() <= 4096:
            self.charge(value.bit_length() + 4)
            return {"t": "int", "v": value}
        if type(value) is str and len(value) <= 4096:
            self.charge(12 * len(value) + 4)
            return {"t": "str", "v": value}
        if isinstance(value, self.np.generic) and value.dtype.str in _DTYPES:
            if value.dtype.kind == "f" and not bool(self.np.isfinite(value)):
                _fail("unsupported_scalar", "nonfinite NumPy scalar")
            return {"t": "scalar", "d": value.dtype.str, "v": value.tobytes().hex()}
        _fail("unsupported_type", type(value).__name__)

    def encode(self, pack):
        context = pack.capsule.context
        anchors = {name: getattr(context, name) for name in _ANCHORS[:-1]}
        anchors["context"] = context
        if any(
            not dataclasses.is_dataclass(v) and type(v) is not self.np.ndarray
            for v in anchors.values()
        ):
            _fail(
                "anchor_layout",
                "named anchors are native objects/arrays, never scalar singletons",
            )
        if len({id(v) for v in anchors.values()}) != len(anchors):
            _fail("anchor_layout", "shared anchors must have distinct roles")
        self.anchor_ids = {id(v): name for name, v in anchors.items()}
        definitions = [
            [name, self.node(value, defining=name)] for name, value in anchors.items()
        ]
        body = self.node(_material(pack))
        arrays = []
        for index, a in enumerate(self.arrays):
            arrays.append(
                {
                    "name": f"a_{index:04d}.npy",
                    "dtype": a.dtype.str,
                    "shape": list(a.shape),
                    "data_bytes": a.nbytes,
                    "data_sha256": hashlib.sha256(
                        memoryview(a).cast("B") if a.size else b""
                    ).hexdigest()
                    if self.hash_arrays
                    else "0" * 64,
                }
            )
        return {"anchors": definitions, "body": body, "arrays": arrays}


def _encoded(pack, limits, *, hashed):
    import numpy as np

    from masy.dense_schedule import DenseSchedule, RainEvent, RainInterval, WindEvent

    _layout(pack, OriginalInputs, _ROOT_FIELDS)
    registry, budget = _registry(), [limits.max_nodes]
    _source_identity(pack.source_identity)
    roots = {
        "capsule": "masy.dense_checkpoint:DenseBoundaryInputs",
        "wind": "masy.dense_runner:DenseWindInputs",
        "water": "masy.dense_runner:DenseWaterInputs",
        "control": "masy.dense_runner:DenseRunnerControl",
        "runner_limits": "masy.dense_runner:DenseRunnerLimits",
    }
    for name, tag in roots.items():
        _check_class(getattr(pack, name), registry[tag], budget=budget)
    for name, cls in (
        ("initial_inventory_kg", np.ndarray),
        ("static_parts", dict),
        ("static_parts_sha256", str),
        ("static_record", dict),
        ("static_sha256", str),
        ("source_identity", dict),
        ("recipes", dict),
        ("content_sha256", str),
    ):
        if type(getattr(pack, name)) is not cls:
            _fail("root_type", f"exact original root type required: {name}")
    _check_class(
        pack.capsule.context,
        registry["maple_syrup.sediment_bed:BedContext"],
        budget=budget,
    )
    _check_class(
        pack.wind.config,
        registry["maple.aeolian.scheduler.state:SchedulerPhysicsConfig"],
        budget=budget,
    )
    _check_class(
        pack.wind.initial_mobile_state,
        registry["maple.core.types.cohort:MobileCohortState"],
        budget=budget,
    )
    for name, tag in (
        ("terrain", "maple_syrup.sediment_bed:TerrainReference"),
        ("column", "maple_syrup.infiltration:ColumnParameters"),
        ("field", "maple_syrup.rainfall:RainfallField"),
        ("sediment", "maple_syrup.sediment_physics:SedimentPhysicsParameters"),
    ):
        _check_class(getattr(pack.water, name), registry[tag], budget=budget)
    _check_class(
        pack.control.sediment_control,
        registry["maple_syrup.sediment_event:SedimentEventControl"],
        budget=budget,
    )
    _check_class(
        pack.control.completion_policy,
        registry["maple_syrup.complete_event:CompletionPolicy"],
        budget=budget,
    )
    cfg, ctx = pack.wind.config, pack.capsule.context
    for name in _ANCHORS[:-1]:
        if getattr(cfg, name) is not getattr(ctx, name):
            _fail(
                "anchor_layout",
                f"first pack profile requires one shared anchor: {name}",
            )
    _layout(pack.schedule, DenseSchedule, ("events",))
    if (
        type(pack.schedule.events) is not tuple
        or not 1 <= len(pack.schedule.events) <= limits.max_events
    ):
        _fail("schedule_invalid", "bounded event tuple required")
    for event in pack.schedule.events:
        if type(event) is WindEvent:
            _layout(
                event,
                WindEvent,
                ("duration_s", "shear_velocity_m_s", "direction_deg_cw_from_north"),
            )
            if any(type(v) is not float for v in vars(event).values()):
                _fail("schedule_invalid", "exact native wind floats required")
        elif type(event) is RainEvent:
            _layout(event, RainEvent, ("intervals", "recession_guard_s"))
            if type(event.recession_guard_s) is not float:
                _fail("schedule_invalid", "exact native rain guard float required")
            if (
                type(event.intervals) is not tuple
                or not 1 <= len(event.intervals) <= 64
            ):
                _fail(
                    "profile_cap",
                    "first pack profile permits at most64 rainfall intervals/event",
                )
            for interval in event.intervals:
                _layout(interval, RainInterval, ("duration_s", "intensity_mm_h"))
                if any(type(v) is not float for v in vars(interval).values()):
                    _fail("schedule_invalid", "exact native rain floats required")
        else:
            _fail("schedule_invalid", "unsupported schedule event type")
    encoder = _Encoder(limits, registry, hash_arrays=hashed, hint_budget=budget)
    result = encoder.encode(pack)
    if len(_json(result)) > limits.max_metadata_bytes:
        _fail("metadata_cap", "encoded metadata cap")
    return result, encoder.arrays


def _same(left, right, limits):
    # Exact factory validation ignores incidental aliases, never values/types.
    import numpy as np

    if type(left) is not type(right):
        return False
    if type(left) is np.ndarray:
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and (memoryview(left).cast("B") if left.size else b"")
            == (memoryview(right).cast("B") if right.size else b"")
        )
    if left is np:
        return right is np
    if dataclasses.is_dataclass(left):
        return set(vars(left)) == set(vars(right)) and all(
            _same(vars(left)[k], vars(right)[k], limits) for k in vars(left)
        )
    if type(left) is dict:
        return list(left) == list(right) and all(
            _same(left[k], right[k], limits) for k in left
        )
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(
            _same(a, b, limits) for a, b in zip(left, right, strict=True)
        )
    if type(left) is float:
        return left.hex() == right.hex()
    if isinstance(left, np.generic):
        return left.dtype == right.dtype and left.tobytes() == right.tobytes()
    return left == right


def _validate(pack, limits):
    from masy.clock import decode_time
    from masy.dense_schedule import case_schedule_digest

    d = _native()
    np, ctx, cfg, water = d.n.np, pack.capsule.context, pack.wind.config, pack.water
    d.bridge._validate_inputs(
        pack.capsule, d.n, d.cp, max(limits.max_numeric_bytes, 1 << 20)
    )
    g, nc = ctx.geometry, len(ctx.grain_classes.classes)
    nv = pack.capsule.case_record["n_voxels"]
    if (
        g.ny * g.nx > limits.max_cells
        or nc > limits.max_classes
        or type(nv) is not int
        or not 1 <= nv <= limits.max_voxels
        or len(pack.schedule.events) > limits.max_events
    ):
        _fail("profile_cap", "default Dense dimension/event cap")
    if (
        type(pack.schedule) is not d.schedule
        or not _same(
            pack.schedule, d.schedule.from_record(pack.schedule.as_record()), limits
        )
        or _json(pack.schedule.as_record())
        != _json(d.schedule.from_record(pack.schedule.as_record()).as_record())
    ):
        _fail("schedule_invalid", "schedule is not exact supported record")
    _keys(pack.recipes, _RECIPES, "recipes")
    if pack.recipes != _RECIPES:
        _fail(
            "recipe_unsupported",
            "only fixed_ksat/nativePlot1defaults/resolved rainfall/native bins",
        )
    _source_identity(pack.source_identity)
    d.runner.DenseRunner._check_wind(d.n, SimpleNamespace(context=ctx), pack.wind)
    d.pool_validate(pack.wind.initial_mobile_state)
    pool = pack.wind.initial_mobile_state
    if pool.mass_kg.size > limits.max_cohorts:
        _fail("profile_cap", "flat cohort cap")
    for name in d.cp._MOBILE:
        value = getattr(pool, name)
        if value is None:
            continue
        dtype = (
            np.dtype("int32")
            if name in d.cp._MOBILE_INTS
            else np.dtype(
                "bool" if name in ("active_mask", "live_has_turned") else "float64"
            )
        )
        if (
            type(value) is not np.ndarray
            or value.dtype != dtype
            or value.shape != pool.mass_kg.shape
        ):
            _fail("pool_layout", f"original pool {name} dtype/shape")
    if any(getattr(pool, name) is not None for name in d.cp._MOBILE[-4:]):
        _fail(
            "profile_unsupported", "active trajectory/live continuation is unsupported"
        )
    bins = cfg.particle_transport.quantile_bins
    if (
        type(bins) is not int
        or not 1 <= bins <= 64
        or cfg.hop_bins.hop_distance_m.shape != (nc, bins)
        or cfg.hop_bins.weight.shape != (nc, bins)
    ):
        _fail(
            "profile_cap",
            "bounded declared/resolved native hop bins must agree before factory allocation",
        )
    if not _same(
        cfg.hop_bins, d.bins(cfg.particle_transport, ctx.grain_classes), limits
    ):
        _fail("recipe_mismatch", "all native hop-bin fields must be exact")
    column = water.column
    if column.model != "fixed_ksat":
        _fail("recipe_unsupported", "only native fixed_ksat column")
    for name in (
        "ksat_m_per_s",
        "suction_m",
        "drainage_parameter",
        "theta_sat",
        "soil_thickness_m",
        "active_mask",
    ):
        array = getattr(column, name)
        if (
            type(array) is not np.ndarray
            or array.shape != (g.ny, g.nx)
            or array.dtype != (np.bool_ if name == "active_mask" else np.float64)
        ):
            _fail(
                "column_layout",
                "native column inputs must match the bounded original grid",
            )
    made = d.column(
        model="fixed_ksat",
        **{
            name: getattr(column, name)
            for name in (
                "ksat_m_per_s",
                "suction_m",
                "drainage_parameter",
                "theta_sat",
                "soil_thickness_m",
                "active_mask",
            )
        },
    )
    if not _same(column, made, limits) or not _same(
        water.sediment, d.sediment(xp=np), limits
    ):
        _fail("recipe_mismatch", "all column/Plot1 resolved fields must be exact")
    if water.sediment.n_classes != nc:
        _fail(
            "profile_unsupported", "native Plot1 class count differs from original bed"
        )
    if not _same(
        water.field,
        d.rainfall(g.ny, g.nx, scale=water.field.multiplier, active_mask=None, xp=np),
        limits,
    ):
        _fail("recipe_mismatch", "all resolved rainfall fields must be exact")
    terrain = water.terrain
    for name in (
        "initial_full_elevation_m",
        "initial_committed_elevation_m",
        "export_receiver_full",
        "active_mask",
        "friction_factor",
    ):
        a = getattr(terrain, name)
        shape = (
            (g.ny + 2, g.nx + 2)
            if name in ("initial_full_elevation_m", "export_receiver_full")
            else (g.ny, g.nx)
        )
        dtype = (
            np.bool_ if name in ("export_receiver_full", "active_mask") else np.float64
        )
        if type(a) is not np.ndarray or a.shape != shape or a.dtype != dtype:
            _fail("terrain_layout", name)
    if terrain.dx_m != g.dx_m:
        _fail("terrain_layout", "terrain spacing differs")
    # Native static graph validation only; no bed, storm, commit or routing receipt.
    d.n.routing_graph_from_elevation(
        terrain, terrain.initial_committed_elevation_m, xp=np
    )
    cover = water.vegetation_cover_fraction
    if (
        d.bed._array_digest(cover)
        != pack.capsule.case_record["vegetation"]["fields"]["raster"]["fields"][
            "cover_fraction"
        ]
    ):
        _fail(
            "prescribed_mismatch", "original water cover differs from prescribed case"
        )
    soil = water.initial_soil_water_m
    if (
        type(soil) is not np.ndarray
        or soil.dtype != np.float64
        or soil.shape != (g.ny, g.nx)
        or not bool(np.isfinite(soil).all())
        or bool((soil < 0).any())
        or bool(np.signbit(soil).any())
    ):
        _fail("soil_layout", "original native soil must be finite float64 >=+0")
    sc, policy = pack.control.sediment_control, pack.control.completion_policy
    sc.validated()
    policy.validated()
    if (
        sc.transport_scheme != "upwind"
        or sc.storm.implementation != "array"
        or sc.commit is not True
        or sc.record_exchange_history is not False
        or sc.deep_publication != "step"
    ):
        _fail("profile_unsupported", "native water profile differs")
    pack.runner_limits.__post_init__()
    if (
        len(pack.schedule.events) > pack.runner_limits.max_events
        or sc.storm.max_steps > pack.runner_limits.max_storm_steps
        or type(pack.control.report_every_s) is not float
        or not math.isfinite(pack.control.report_every_s)
        or pack.control.report_every_s <= 0
    ):
        _fail("control_invalid", "original limits/cadence differ")
    inv = pack.initial_inventory_kg
    if (
        type(inv) is not np.ndarray
        or inv.dtype != np.float64
        or inv.shape != (nc,)
        or not bool(np.isfinite(inv).all())
        or bool((inv < 0).any())
    ):
        _fail("inventory_layout", "original inventory audit vector layout")
    view = SimpleNamespace(
        _wind=pack.wind,
        _water=water,
        _control=pack.control,
        _schedule=pack.schedule,
        _production=None,
    )
    parts = d.runner.DenseRunner._static_parts_record(view, d.n)
    if (
        not _same(parts, pack.static_parts, limits)
        or d.bed._sha256(parts) != pack.static_parts_sha256
    ):
        _fail("static_mismatch", "original controlling fields differ")
    expected = {
        "schema": d.runner.STATIC_SCHEMA,
        "profile": dict(d.runner.PROFILE),
        "case_sha256": pack.capsule.case_sha256,
        "schedule_sha256": pack.schedule.digest,
        "case_schedule_sha256": case_schedule_digest(
            pack.capsule.case_sha256, pack.schedule
        ),
        "controlling_inputs_sha256": pack.static_parts_sha256,
        "limits": pack.runner_limits.as_record(),
        "initial_evolving": {
            "mobile_state_sha256": d.bed._sha256(
                d.runner._canonical_input(np, pool, "mobile")
            ),
            "soil_water_m": d.bed._array_digest(soil),
        },
    }
    if (
        not _same(expected, pack.static_record, limits)
        or d.bed._sha256(expected) != pack.static_sha256
    ):
        _fail("static_mismatch", "original static/initial-evolving record differs")
    decode_time(pack.capsule.case_record["initial_time_s"])


_DEFAULT_LIMITS = OriginalPackLimits()


def validate_original_inputs(pack, *, limits=_DEFAULT_LIMITS):
    """Rehash exact original aliases; no bed/runner construction or transport."""
    OriginalPackLimits.__post_init__(limits)
    _encoded(
        pack, limits, hashed=False
    )  # Complete closed/cap preflight BEFORE hashing/factories.
    _validate(pack, limits)
    value, _ = _encoded(pack, limits, hashed=True)
    digest = hashlib.sha256(_json(value)).hexdigest()
    if pack.content_sha256 != digest:
        _fail("content_mismatch", "original aliases/content changed")
    return digest


def capture_original_inputs(
    runner, *, source_identity, recipes=None, limits=_DEFAULT_LIMITS
):
    """Capture a quiescent NEW default runner by reference; never initialize one."""
    d = _native()
    from masy.clock import ClockLedger, decode_time

    if type(runner) is not d.runner.DenseRunner or set(vars(runner)) != _RUNNER_FIELDS:
        _fail("runner_layout", "only exact default constructor fields")
    if type(runner._ledger) is not ClockLedger or set(vars(runner._ledger)) != {
        "epoch",
        "_entries",
    }:
        _fail("not_genesis", "exact initial clock ledger required")
    if (
        runner._production is not None
        or type(runner._next) is not int
        or runner._next != 0
        or type(runner._n_accepted) is not int
        or runner._n_accepted != 0
        or type(runner._receipts) is not list
        or runner._receipts != []
        or runner._routing is not None
        or runner._active_evidence is not None
        or runner._stop_receipt is not None
        or type(runner._ledger._entries) is not list
        or runner._ledger._entries != []
        or type(runner._ledger.epoch) is not float
        or runner._ledger.records() != []
        or runner._pool is not runner._wind.initial_mobile_state
    ):
        _fail("not_genesis", "original pack capture requires a NEW runner")
    OriginalPackLimits.__post_init__(limits)
    bed = runner._bed
    # Build only a reference carrier for a COMPLETE cap/layout pass. No deep copy,
    # source hash, native inventory scan or native capsule capture precedes it.
    capsule = d.bridge.DenseBoundaryInputs(
        bed._case_record,
        bed._held_inputs,
        bed.context,
        bed.limits,
        bed._initialization_receipt,
        bed._prescribed_digests,
        bed.case_sha256,
        "",
    )
    pack = OriginalInputs(
        capsule,
        runner._schedule,
        runner._wind,
        runner._water,
        runner._control,
        runner._limits,
        runner._inventory0,
        runner._static_parts,
        runner._static_parts_sha256,
        runner._static_record,
        runner._static_sha256,
        source_identity,
        dict(_RECIPES) if recipes is None else recipes,
        "",
    )
    _encoded(pack, limits, hashed=False)
    expected_totals = {
        "export",
        "input",
        "bound",
        "internal_terminal_deposition",
        "water",
        "n_entry_reroutes",
        "n_event_reroutes",
    }
    _keys(runner._totals, expected_totals, "genesis totals")
    nc = len(bed.context.grain_classes.classes)
    if (
        type(runner._n_classes) is not int
        or runner._n_classes != nc
        or type(runner._inventory_elements) is not int
        or runner._inventory_elements
        != bed.context.geometry.ny
        * bed.context.geometry.nx
        * (bed.case_record["n_voxels"] + 1)
        or type(bed.time_s) is not float
        or bed.time_s.hex()
        != decode_time(pack.capsule.case_record["initial_time_s"]).hex()
    ):
        _fail("not_genesis", "original dimensions/epoch differ")
    for name in ("export", "input", "bound", "internal_terminal_deposition"):
        a = runner._totals[name]
        if (
            type(a) is not d.n.np.ndarray
            or a.dtype != d.n.np.float64
            or a.shape != (nc,)
            or bool(d.n.np.any(a.view(d.n.np.uint64) != 0))
        ):
            _fail("not_genesis", "initial cumulative class totals must be exact +0")
    _keys(
        runner._totals["water"],
        (
            "rain_m3",
            "drainage_m3",
            "storm_export_m3",
            "dry_surface_removed_m3",
            "dry_soil_removed_m3",
        ),
        "genesis water totals",
    )
    if any(
        type(v) is not float or v.hex() != (0.0).hex()
        for v in runner._totals["water"].values()
    ) or any(
        type(runner._totals[k]) is not int or runner._totals[k] != 0
        for k in ("n_entry_reroutes", "n_event_reroutes")
    ):
        _fail("not_genesis", "initial cumulative water/reroute totals differ")
    bed.verify_case_inputs()
    if (
        bed.time_s.hex() != runner._ledger.epoch.hex()
        or bed.identity_sha256() != runner._accepted_bed
        or runner._private_identity(d.n.np, runner._pool, runner._soil)
        != runner._accepted_private
    ):
        _fail("not_genesis", "initial boundary changed")
    if not _same(runner._soil, runner._water.initial_soil_water_m, limits):
        _fail("not_genesis", "initial soil changed")
    pack = dataclasses.replace(
        pack,
        capsule=d.bridge.DenseBoundaryInputs.capture(bed),
        static_parts=copy.deepcopy(pack.static_parts),
        static_record=copy.deepcopy(pack.static_record),
        source_identity=copy.deepcopy(source_identity),
        recipes=copy.deepcopy(pack.recipes),
    )
    _encoded(pack, limits, hashed=False)
    _validate(pack, limits)
    if not _same(
        pack.water.terrain.initial_committed_elevation_m,
        d.n.committed_elevation_m(bed.state.committed_topography),
        limits,
    ):
        _fail(
            "not_genesis",
            "terrain reference differs from actual initial native committed state",
        )
    if not _same(
        pack.initial_inventory_kg,
        d.n.np.asarray(d.n.bed_inventory(bed.state), dtype=d.n.np.float64),
        limits,
    ):
        _fail("not_genesis", "original inventory differs from actual source bed")
    encoded, _ = _encoded(pack, limits, hashed=True)
    pack = dataclasses.replace(
        pack, content_sha256=hashlib.sha256(_json(encoded)).hexdigest()
    )
    validate_original_inputs(pack, limits=limits)
    return pack


def _stat_id(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_dir(path):
    path = Path(path)
    fd = os.open(
        "/" if path.is_absolute() else ".", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    )
    try:
        for name in path.parts:
            if name in ("/", "."):
                continue
            if name == "..":
                _fail("path_refused", "parent path traversal")
            child = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=fd,
            )
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_file(directory, name):
    if (
        type(name) is not str
        or not _ARRAY_NAME.fullmatch(name)
        and name not in ("manifest.json", "manifest.tmp")
    ):
        _fail("path_refused", "closed artifact file names required")
    before = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        _fail("file_type", "regular file required before open")
    fd = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOCTTY,
        dir_fd=directory,
    )
    try:
        s = os.fstat(fd)
    except BaseException:
        os.close(fd)
        raise
    if not stat.S_ISREG(s.st_mode):
        os.close(fd)
        _fail("file_type", "regular file required")
    if (s.st_dev, s.st_ino) != (before.st_dev, before.st_ino):
        os.close(fd)
        _fail("file_changed", "prechecked file identity differs from opened file")
    return fd, s


def _hash_fd(fd, size):
    h = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    remaining = size
    while remaining:
        block = os.read(fd, min(remaining, 1 << 20))
        if not block:
            _fail("file_changed", "truncated while hashing")
        h.update(block)
        remaining -= len(block)
    if os.read(fd, 1):
        _fail("file_changed", "grew while hashing")
    return h.hexdigest()


def _names(directory, limits):
    names = []
    with os.scandir(directory) as entries:
        for entry in entries:
            names.append(entry.name)
            if len(names) > limits.max_files + 2:
                _fail("file_cap", "artifact entry count")
    return names


def _read_npy_header(stream, limits):
    """Bound the raw declared header BEFORE NumPy reads or parses that header."""
    import numpy as np

    try:
        version = np.lib.format.read_magic(stream)
    except ValueError as exc:
        raise OriginalInputsError("array_layout", "invalid NPY magic") from exc
    if version not in ((1, 0), (2, 0)):
        _fail("array_layout", "NPY version")
    width = 2 if version == (1, 0) else 4
    start = stream.tell()
    prefix = stream.read(width)
    if len(prefix) != width:
        _fail("array_layout", "truncated NPY header length")
    if int.from_bytes(prefix, "little") > limits.max_header_bytes:
        _fail("header_cap", "raw NPY header length cap before header read")
    stream.seek(start)
    reader = (
        np.lib.format.read_array_header_1_0
        if version == (1, 0)
        else np.lib.format.read_array_header_2_0
    )
    try:
        return reader(stream, max_header_size=limits.max_header_bytes)
    except ValueError as exc:
        raise OriginalInputsError("array_layout", "invalid bounded NPY header") from exc


def _verify_written(directory, row, limits):
    """Audit actual NPY header/data without materializing an extra array."""
    fd, before = _open_file(directory, row["name"])
    try:
        if before.st_nlink != 1:
            _fail("file_type", "written numerical file has a foreign hardlink")
        with os.fdopen(os.dup(fd), "rb") as stream:
            shape, fortran, dtype = _read_npy_header(stream, limits)
            offset = stream.tell()
        if (
            list(shape) != row["shape"]
            or fortran is not False
            or dtype.str != row["dtype"]
            or offset + row["data_bytes"] != before.st_size
            or before.st_size != row["file_bytes"]
        ):
            _fail("array_layout", "written NPY differs from exact source layout")
        os.lseek(fd, offset, os.SEEK_SET)
        remaining, h = row["data_bytes"], hashlib.sha256()
        while remaining:
            block = os.read(fd, min(remaining, 1 << 20))
            if not block:
                _fail("file_changed", "written array truncated")
            h.update(block)
            remaining -= len(block)
        if os.read(fd, 1) or h.hexdigest() != row["data_sha256"]:
            _fail("array_hash", "written data differ from original bytes")
        if _hash_fd(fd, before.st_size) != row["sha256"]:
            _fail("file_hash", "written full-file identity changed")
        if _stat_id(os.fstat(fd)) != _stat_id(before):
            _fail("file_changed", "written data changed")
    finally:
        os.close(fd)


def _flat_wire(encoded):
    """Keep the JSON depth cap without multiplying native value nesting by tags."""
    nodes = []

    def add(node):
        row = dict(node)
        kind = row["t"]
        if kind == "class" or kind == "dict":
            row["v"] = [[key, add(value)] for key, value in row["v"]]
        elif kind in ("list", "tuple"):
            row["v"] = [add(value) for value in row["v"]]
        elif kind == "enum":
            row["v"] = add(row["v"])
        index = len(nodes)
        nodes.append(row)
        return index

    anchors = [[name, add(value)] for name, value in encoded["anchors"]]
    body = add(encoded["body"])
    return {"anchors": anchors, "body": body, "nodes": nodes}


def save_original_inputs(path, pack, *, limits=_DEFAULT_LIMITS):
    """Write a new exclusive sealed pack. No HEAD or acceptance publication."""
    OriginalPackLimits.__post_init__(limits)
    digest = validate_original_inputs(pack, limits=limits)
    encoded, arrays = _encoded(pack, limits, hashed=True)
    target = Path(path)
    if target.name in ("", ".", ".."):
        _fail("path_refused", "new pack name required")
    estimate = (
        sum(a.nbytes + limits.max_header_bytes for a in arrays)
        + limits.max_metadata_bytes
    )
    if estimate > limits.max_serialized_bytes:
        _fail("serialized_cap", "conservative serialized preflight exceeds cap")
    parent = _open_dir(target.parent)
    directory = None
    try:
        space = os.fstatvfs(parent)
        if (
            space.f_bavail * space.f_frsize < estimate + (1 << 20)
            or space.f_favail < len(arrays) + 2
        ):
            _fail("storage_cap", "insufficient free space/inodes with bounded headroom")
        os.mkdir(
            target.name, mode=0o700, dir_fd=parent
        )  # Exclusive; never replaces an existing entry.
        created = os.stat(target.name, dir_fd=parent, follow_symlinks=False)
        directory = os.open(
            target.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent,
        )
        actual = os.fstat(directory)
        if (actual.st_dev, actual.st_ino) != (created.st_dev, created.st_ino) or _names(
            directory, limits
        ):
            _fail("file_changed", "new pack directory replaced/nonempty")
    except BaseException:
        if directory is not None:
            os.close(directory)
        os.close(parent)
        raise
    linked = False
    try:
        files = []
        used = 0
        for description, a in zip(encoded["arrays"], arrays, strict=True):
            fd = os.open(
                description["name"],
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=directory,
            )
            with os.fdopen(fd, "wb") as stream:
                import numpy as np

                np.lib.format.write_array(stream, a, allow_pickle=False)
                os.fchmod(stream.fileno(), 0o400)
                stream.flush()
                os.fsync(stream.fileno())
            fd, s = _open_file(directory, description["name"])
            try:
                used += s.st_size
                if used > limits.max_serialized_bytes:
                    _fail("serialized_cap", "serialized bytes exceed cap")
                files.append(
                    {
                        **description,
                        "file_bytes": s.st_size,
                        "sha256": _hash_fd(fd, s.st_size),
                    }
                )
            finally:
                os.close(fd)
        # A source write during I/O must refuse before the manifest can be sealed.
        if validate_original_inputs(pack, limits=limits) != digest:
            _fail("content_mismatch", "original identity changed during writing")
        for row in files:
            _verify_written(directory, row, limits)
        manifest = {
            "schema": SCHEMA,
            "profile": PROFILE,
            "content_sha256": digest,
            "wire": _flat_wire(encoded),
            "files": files,
        }
        manifest["manifest_sha256"] = hashlib.sha256(_json(manifest)).hexdigest()
        raw = _json(manifest)
        if (
            len(raw) > limits.max_metadata_bytes
            or used + len(raw) > limits.max_serialized_bytes
        ):
            _fail("serialized_cap", "aggregate pack/metadata cap")
        fd = os.open(
            "manifest.tmp",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            os.fchmod(stream.fileno(), 0o400)
            stream.flush()
            os.fsync(stream.fileno())
        fd, ms = _open_file(directory, "manifest.tmp")
        try:
            if (
                ms.st_nlink != 1
                or ms.st_size != len(raw)
                or os.read(fd, len(raw) + 1) != raw
                or _stat_id(os.fstat(fd)) != _stat_id(ms)
            ):
                _fail(
                    "manifest_identity", "actual written manifest differs before seal"
                )
        finally:
            os.close(fd)
        os.fsync(directory)
        linked = True  # Publication may happen from this syscall onward.
        os.link(
            "manifest.tmp",
            "manifest.json",
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
        os.unlink("manifest.tmp", dir_fd=directory)
        os.fsync(directory)
        os.fsync(parent)
        return manifest["manifest_sha256"]
    except BaseException as exc:
        if linked:
            message = f"original pack seal may have been published at {target}; durability uncertain"
            if not isinstance(exc, Exception):
                exc.add_note(message)
                raise
            raise OriginalPackPublicationUncertain(message) from exc
        raise
    finally:
        os.close(directory)
        os.close(parent)


class _Decoder:
    def __init__(self, arrays, registry, limits, nodes):
        import numpy as np

        self.np, self.arrays, self.registry, self.limits = np, arrays, registry, limits
        self.anchors, self.nodes, self.table, self.used = {}, 0, nodes, set()
        self.used_arrays = set()
        self.hint_budget = [limits.max_nodes]

    def node(self, index, depth=0):
        self.nodes += 1
        if (
            self.nodes > self.limits.max_nodes
            or depth > self.limits.max_depth
            or type(index) is not int
            or not 0 <= index < len(self.table)
        ):
            _fail("wire_invalid", "typed node/depth/count")
        self.used.add(index)
        node = self.table[index]
        if type(node) is not dict:
            _fail("wire_invalid", "typed node object")
        kind = node.get("t")
        if type(kind) is not str:
            _fail("wire_invalid", "node kind must be a string")
        expected = {
            "none": {"t"},
            "numpy": {"t"},
            "ref": {"t", "v"},
            "array": {"t", "v"},
            "class": {"t", "c", "v"},
            "enum": {"t", "c", "v"},
            "scalar": {"t", "d", "v"},
        }
        _keys(node, expected.get(kind, {"t", "v"}), "typed node")
        if kind == "none":
            return None
        if kind == "numpy":
            return self.np
        if kind == "ref":
            if type(node["v"]) is not str or node["v"] not in self.anchors:
                _fail("anchor_invalid", "unknown/forward/cyclic anchor")
            return self.anchors[node["v"]]
        if kind == "array":
            if type(node["v"]) is not int or not 0 <= node["v"] < len(self.arrays):
                _fail("wire_invalid", "array index")
            self.used_arrays.add(node["v"])
            return self.arrays[node["v"]]
        if kind == "scalar":
            if (
                type(node["d"]) is not str
                or node["d"] not in _DTYPES
                or type(node["v"]) is not str
            ):
                _fail("wire_invalid", "scalar dtype/bytes")
            dtype = self.np.dtype(node["d"])
            if len(node["v"]) != 2 * dtype.itemsize:
                _fail("wire_invalid", "scalar hex length")
            try:
                raw = bytes.fromhex(node["v"])
            except ValueError as exc:
                raise OriginalInputsError("wire_invalid", "scalar hex") from exc
            if len(raw) != dtype.itemsize:
                _fail("wire_invalid", "scalar size")
            return self.np.frombuffer(raw, dtype=dtype)[0]
        if kind in ("class", "enum"):
            if type(node["c"]) is not str or node["c"] not in self.registry:
                _fail("type_refused", "unknown native type tag")
            cls, fields, enum = self.registry[node["c"]]
            if (kind == "enum") != enum:
                _fail("wire_invalid", "native tag kind")
            if enum:
                return cls(self.node(node["v"], depth + 1))
            rows = node["v"]
            if (
                type(rows) is not list
                or any(type(r) is not list or len(r) != 2 for r in rows)
                or [r[0] for r in rows] != list(fields)
            ):
                _fail("wire_invalid", "closed ordered native fields")
            value = cls(**{name: self.node(item, depth + 1) for name, item in rows})
            _check_class(value, (cls, fields, enum), budget=self.hint_budget)
            return value
        if kind in ("dict", "list", "tuple"):
            rows = node["v"]
            if type(rows) is not list:
                _fail("wire_invalid", "container rows")
            if kind != "dict":
                values = [self.node(v, depth + 1) for v in rows]
                return tuple(values) if kind == "tuple" else values
            if any(
                type(r) is not list or len(r) != 2 or type(r[0]) is not str
                for r in rows
            ) or len({r[0] for r in rows}) != len(rows):
                _fail("wire_invalid", "closed unique ordered mapping")
            return {k: self.node(v, depth + 1) for k, v in rows}
        value = node.get("v")
        if kind == "float" and type(value) is str:
            if len(value) > 64:
                _fail("wire_invalid", "float hex length")
            try:
                result = float.fromhex(value)
            except ValueError as exc:
                raise OriginalInputsError("wire_invalid", "float hex") from exc
            if not math.isfinite(result) or result.hex() != value:
                _fail("wire_invalid", "noncanonical finite float")
            return result
        types = {"str": str, "bool": bool, "int": int}
        if kind in types and type(value) is types[kind]:
            if (
                kind == "str"
                and len(value) > 4096
                or kind == "int"
                and value.bit_length() > 4096
            ):
                _fail("wire_invalid", "scalar length cap")
            return value
        _fail("wire_invalid", "unknown or wrong scalar kind")


def load_original_inputs(
    path, *, expected_manifest_sha256, expected_source_identity, limits=_DEFAULT_LIMITS
):
    """Load only original typed inputs; native physical genesis validation is later."""
    OriginalPackLimits.__post_init__(limits)
    _source_identity(expected_source_identity)
    expected_source_identity = dict(expected_source_identity)
    if type(expected_manifest_sha256) is not str or not _SHA.fullmatch(
        expected_manifest_sha256
    ):
        _fail("manifest_identity", "expected immutable pack identity is required")
    directory = _open_dir(path)
    opened = []
    try:
        names = _names(directory, limits)
        if len(names) > limits.max_files + 2 or "manifest.json" not in names:
            _fail("file_cap", "artifact entry count/unsealed pack")
        # Bound ALL raw file sizes/count/aggregate BEFORE hashing or NPY parsing.
        total = 0
        descriptors = {}
        for name in sorted(names):
            fd, s = _open_file(directory, name)
            opened.append(fd)
            descriptors[name] = (fd, s)
            if name not in ("manifest.json", "manifest.tmp") and s.st_nlink != 1:
                _fail("file_type", "foreign numeric hardlink refused")
            if name == "manifest.tmp":
                prior = descriptors["manifest.json"][1]
                if _stat_id(s) != _stat_id(prior) or s.st_nlink != 2:
                    _fail(
                        "unlisted_file",
                        "only exact leftover seal hardlink is recoverable",
                    )
            else:
                total += s.st_size
            if total > limits.max_serialized_bytes:
                _fail(
                    "serialized_cap", "raw aggregate serialized cap before hash/header"
                )
            if (
                name in ("manifest.json", "manifest.tmp")
                and s.st_size > limits.max_metadata_bytes
            ):
                _fail("metadata_cap", "raw metadata cap")
        mfd, ms = descriptors["manifest.json"]
        raw = os.read(mfd, ms.st_size + 1)
        manifest = _strict_json(raw, limits)
        _keys(
            manifest,
            ("schema", "profile", "content_sha256", "wire", "files", "manifest_sha256"),
            "manifest",
        )
        seal = manifest["manifest_sha256"]
        unsigned = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if (
            manifest["schema"] != SCHEMA
            or manifest["profile"] != PROFILE
            or seal != expected_manifest_sha256
            or hashlib.sha256(_json(unsigned)).hexdigest() != seal
            or _json(manifest) != raw
        ):
            _fail("manifest_identity", "schema/profile/canonical seal mismatch")
        _keys(manifest["wire"], ("anchors", "body", "nodes"), "wire")
        if (
            type(manifest["wire"]["nodes"]) is not list
            or len(manifest["wire"]["nodes"]) > limits.max_nodes
        ):
            _fail("metadata_cap", "flat typed node count")
        # Check the caller's independently verified source reference BEFORE native
        # class construction, payload hashing, factory invocation or expansion.
        table = manifest["wire"]["nodes"]
        body_index = manifest["wire"]["body"]
        if type(body_index) is not int or not 0 <= body_index < len(table):
            _fail("wire_invalid", "body index")
        root = table[body_index]
        if (
            type(root) is not dict
            or root.get("t") != "dict"
            or type(root.get("v")) is not list
            or any(
                type(r) is not list or len(r) != 2 or type(r[0]) is not str
                for r in root["v"]
            )
        ):
            _fail("wire_invalid", "original body rows")
        source_rows = [r[1] for r in root["v"] if r[0] == "source_identity"]
        if len(source_rows) != 1:
            _fail("source_identity", "missing/duplicate source identity")
        recorded_source = _Decoder([], {}, limits, table).node(source_rows[0])
        _source_identity(recorded_source)
        _source_identity(expected_source_identity)
        if recorded_source != expected_source_identity:
            _fail(
                "source_identity",
                "original pack belongs to a different expected source identity",
            )
        rows = manifest["files"]
        if type(rows) is not list or len(rows) > limits.max_files:
            _fail("file_cap", "manifest numerical file count")
        required = {"manifest.json"}
        if "manifest.tmp" in descriptors:
            _tfd, ts = descriptors["manifest.tmp"]
            if _stat_id(ts) != _stat_id(ms) or ts.st_nlink != 2:
                _fail(
                    "unlisted_file", "only exact leftover seal hardlink is recoverable"
                )
            required.add("manifest.tmp")
        elif ms.st_nlink != 1:
            _fail("file_type", "foreign manifest hardlink refused")
        import numpy as np

        numeric, arrays = 0, []
        for index, row in enumerate(rows):
            _keys(
                row,
                (
                    "name",
                    "dtype",
                    "shape",
                    "data_bytes",
                    "data_sha256",
                    "file_bytes",
                    "sha256",
                ),
                "array description",
            )
            if (
                row["name"] != f"a_{index:04d}.npy"
                or type(row["dtype"]) is not str
                or row["dtype"] not in _DTYPES
            ):
                _fail("array_layout", "closed array name/dtype")
            shape = row["shape"]
            dimension_cap = max(
                limits.max_cohorts,
                limits.max_cells + 2,
                limits.max_classes,
                limits.max_voxels,
            )
            if (
                type(shape) is not list
                or len(shape) > 4
                or any(type(v) is not int or not 0 <= v <= dimension_cap for v in shape)
            ):
                _fail("array_layout", "shape")
            size = np.dtype(row["dtype"]).itemsize
            for dim in shape:
                size *= dim
                if size > limits.max_numeric_bytes:
                    _fail("array_cap", "shape product cap before expansion")
            if type(row["data_bytes"]) is not int or size != row["data_bytes"]:
                _fail("array_layout", "data byte count")
            numeric += size
            if numeric > limits.max_numeric_bytes:
                _fail("array_cap", "aggregate expansion cap")
            if row["name"] not in descriptors:
                _fail("missing_file", "numerical file missing")
            fd, s = descriptors[row["name"]]
            if type(row["file_bytes"]) is not int or s.st_size != row["file_bytes"]:
                _fail("array_layout", "raw file size differs")
            if (
                type(row["sha256"]) is not str
                or not _SHA.fullmatch(row["sha256"])
                or type(row["data_sha256"]) is not str
                or not _SHA.fullmatch(row["data_sha256"])
            ):
                _fail("array_layout", "exact data/file SHA256 fields required")
            required.add(row["name"])
        if set(names) != required:
            _fail("unlisted_file", "unknown entries")
        # Every raw/expanded aggregate and shape has passed before first payload
        # hash, NPY header parse or numerical allocation.
        for row in rows:
            fd, s = descriptors[row["name"]]
            shape, size = row["shape"], row["data_bytes"]
            if _stat_id(os.fstat(fd)) != _stat_id(s):
                _fail("file_changed", "changed before payload read")
            if _hash_fd(fd, s.st_size) != row["sha256"]:
                _fail("file_hash", "numerical hash mismatch")
            os.lseek(fd, 0, os.SEEK_SET)
            with os.fdopen(os.dup(fd), "rb") as stream:
                actual_shape, fortran, dtype = _read_npy_header(stream, limits)
                if (
                    list(actual_shape) != shape
                    or fortran is not False
                    or dtype.str != row["dtype"]
                    or dtype.hasobject
                    or dtype.fields is not None
                    or stream.tell() + size != s.st_size
                ):
                    _fail("array_layout", "raw NPY header contract")
                a = np.fromfile(stream, dtype=dtype, count=size // dtype.itemsize)
            if a.size != size // dtype.itemsize:
                _fail("file_changed", "truncated numerical data")
            a = a.reshape(tuple(shape))
            a.base.flags.writeable = False
            if (
                hashlib.sha256(memoryview(a).cast("B") if a.size else b"").hexdigest()
                != row["data_sha256"]
            ):
                _fail("array_hash", "expanded raw bytes differ")
            a.flags.writeable = False
            arrays.append(a)
        decoder = _Decoder(arrays, _registry(), limits, manifest["wire"]["nodes"])
        anchors = manifest["wire"]["anchors"]
        if (
            type(anchors) is not list
            or any(type(r) is not list or len(r) != 2 for r in anchors)
            or [r[0] for r in anchors] != list(_ANCHORS)
        ):
            _fail("anchor_invalid", "closed ordered anchors")
        for name, node in anchors:
            decoder.anchors[name] = decoder.node(node)
        body = decoder.node(manifest["wire"]["body"])
        if decoder.used != set(range(len(decoder.table))):
            _fail("wire_invalid", "unreferenced typed nodes")
        if decoder.used_arrays != set(range(len(arrays))):
            _fail("wire_invalid", "unreferenced numerical files")
        _keys(
            body,
            tuple(n for n in _ROOT_FIELDS if n != "content_sha256"),
            "original body",
        )
        d = _native()
        schedule_record = body["schedule"]
        if (
            type(schedule_record) is not dict
            or type(schedule_record.get("events")) is not list
            or not 1 <= len(schedule_record["events"]) <= limits.max_events
        ):
            _fail("profile_cap", "bounded raw schedule before native construction")
        for event in schedule_record["events"]:
            if (
                type(event) is not dict
                or event.get("kind") == "rain"
                and (
                    type(event.get("intervals")) is not list
                    or not 1 <= len(event["intervals"]) <= 64
                )
            ):
                _fail(
                    "profile_cap",
                    "bounded raw rainfall intervals before native construction",
                )
        body["schedule"] = d.schedule.from_record(body["schedule"])
        pack = OriginalInputs(**body, content_sha256=manifest["content_sha256"])
        validate_original_inputs(pack, limits=limits)
        for name, (fd, before) in descriptors.items():
            if _stat_id(os.fstat(fd)) != _stat_id(before):
                _fail("file_changed", "opened file changed during read")
            live = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if _stat_id(live) != _stat_id(before):
                _fail("file_changed", "artifact name changed during read")
        if set(_names(directory, limits)) != set(names):
            _fail("file_changed", "artifact entries changed")
        return pack
    finally:
        for fd in opened:
            os.close(fd)
        os.close(directory)
