"""Authored native fixture for the Dense runner checks (test-only; never imported by product code).

Every object is built with actual MAPLE / SYRUP constructors and explicitly
authored parameters matching the verified source-only native alternation
reference (source_context/native_alternation_reference.py): the six-class
MAHLERAN chain of 6 x 1 cells (z0 0.12 m, 0.01 m per row, dx 0.5 m, 3 voxels
of 0.1 m, bulk 1250 kg/m3, active layer 0.002 m, default mass resolution),
25% initial availability, aerodynamic detachment_transport_deposition wind
(reference rate 1e-3, 8 hop bins, median hop 0.3 m, 16000 cohorts, outer
step 1 s), the ringed SYRUP routing graph (walls +1 m, south export ring
0.02 m below row 0, friction factor 1, routing parameter 0.5), fixed-ksat
1e-5 m/s soil column, zero entry soil water, upwind transport with array
hydraulics (max_dt 1 s, max_steps 10000), deposit_mobile_at_flow_end and a
10 s reporting cadence. The values reproduce the reference; they are not a
calibration.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from maple.aeolian.scheduler.state import SchedulerPhysicsConfig
from maple.aeolian.trajectories.quantile_bins import build_hop_quantile_bins
from maple.core.boundaries import AxisBoundary, BoundaryKind
from maple.core.parameters.aeolian_process_mode import AeolianProcessModeSpec
from maple.core.parameters.aerodynamic_detachment import AerodynamicDetachmentSpec
from maple.core.parameters.avalanching import AvalanchingSpec
from maple.core.parameters.fluid_threshold import FluidThresholdSpec
from maple.core.parameters.geometry import GeometrySpec
from maple.core.parameters.grain_classes import GrainClass, GrainClassSet
from maple.core.parameters.impact_rebound_splash import ImpactReboundSplashSpec
from maple.core.parameters.numerics import DEFAULT_MASS_RESOLUTION_KG
from maple.core.parameters.particle_transport import HopDistanceClassSpec, ParticleTransportSpec
from maple.core.parameters.process_scheduler import ProcessSchedulerSpec
from maple.core.parameters.topographic_commit import TopographicCommitSpec
from maple.core.parameters.water_coupling import MAHLERAN_1_2_1_CLASS_DIAMETERS_M
from maple.core.parameters.wind_grain_feedback import WindGrainFeedbackSpec
from maple.core.types.cohort import zeros_mobile_cohort_state
from maple.core.types.threshold_modifiers import neutral_threshold_modifier_state
from maple.core.types.topographic_commit import committed_elevation_m
from maple.core.types.vegetation import empty_vegetation_state
from maple_syrup.complete_event import CompletionPolicy
from maple_syrup.infiltration import column_parameters
from maple_syrup.rainfall import rainfall_field
from maple_syrup.routing import build_routing_graph
from maple_syrup.sediment_bed import terrain_reference
from maple_syrup.sediment_event import SedimentEventControl
from maple_syrup.sediment_physics import plot1_sediment_parameters
from maple_syrup.storm import StormControl

from masy.dense_bed import DenseBedLimits, build_dense_bed
from masy.dense_runner import (
    DenseRunner,
    DenseRunnerControl,
    DenseRunnerLimits,
    DenseWaterInputs,
    DenseWindInputs,
)
from masy.dense_schedule import DenseSchedule

NY, NX = 6, 1
DX = 0.5
N_VOXELS, VOXEL_DZ, BULK, ACTIVE = 3, 0.1, 1250.0, 0.002
CLASS_IDS = tuple(f"phi_{k}" for k in range(1, 7))
FRACTIONS = (0.10, 0.15, 0.25, 0.25, 0.15, 0.10)
AVAILABLE = 0.25
MAX_COHORTS = 16000
SHEAR, DIRECTION, WIND_S = 0.5, 270.0, 2.0
RAIN_S, RAIN_MM_H, GUARD_S = 60.0, 72.0, 180.0
KSAT = 1e-5
REPORT_EVERY_S = 10.0


def grain_classes() -> GrainClassSet:
    return GrainClassSet(classes=tuple(
        GrainClass(class_id=c, diameter_m=float(d), particle_density_kg_m3=2650.0, is_aggregate=False)
        for c, d in zip(CLASS_IDS, MAHLERAN_1_2_1_CLASS_DIAMETERS_M, strict=True)))


def geometry(classes: GrainClassSet) -> GeometrySpec:
    def boundary():
        return AxisBoundary(kind=BoundaryKind.PRESCRIBED, inflow_flux_kg_m_s={c: 0.0 for c in CLASS_IDS})

    return GeometrySpec(nx=NX, ny=NY, dx_m=DX, dy_m=DX, boundary_x=boundary(), boundary_y=boundary(),
                        voxel_dz_m=VOXEL_DZ, bulk_density_kg_m3=BULK, active_layer_thickness_m=ACTIVE)


def chain_elevation() -> np.ndarray:
    return 0.12 + 0.01 * np.arange(NY, dtype=np.float64)[:, None]


def build_bed(*, available=None, commit_spec=None, limits=None):
    """The reference chain bed through the qualified one-time native Dense initialization."""
    classes = grain_classes()
    fractions = np.array(np.broadcast_to(np.asarray(FRACTIONS, dtype=np.float64), (NY, NX, len(CLASS_IDS))))
    available = np.full((NY, NX, len(CLASS_IDS)), AVAILABLE) if available is None else available
    return build_dense_bed(
        geometry=geometry(classes), grain_classes=classes, mass_resolution_kg=float(DEFAULT_MASS_RESOLUTION_KG),
        n_voxels=N_VOXELS, initial_bed_elevation_m=chain_elevation(), initial_class_fractions=fractions,
        initial_available_fraction=available, initial_water_depth_m=np.zeros((NY, NX)),
        vegetation=empty_vegetation_state(NY, NX), threshold_modifier_state=neutral_threshold_modifier_state(NY, NX),
        topographic_commit_spec=commit_spec or TopographicCommitSpec(), avalanching_spec=AvalanchingSpec(enabled=False),
        active_mask=np.ones((NY, NX), dtype=bool), initial_time_s=0.0,
        limits=limits or DenseBedLimits(host_memory_cap_bytes=1 << 30, max_voxels_per_column=64,
                                        stage_copy_max_bytes=1 << 24))


def wind_config(bed, **overrides) -> SchedulerPhysicsConfig:
    ctx = bed.context
    particle = ParticleTransportSpec(quantile_bins=8, hop_distance_by_class={
        c.class_id: HopDistanceClassSpec(median_hop_distance_m=0.3, geometric_stddev=2.0,
                                         representative_flight_speed_m_s=3.0, representative_impact_speed_m_s=3.0,
                                         representative_impact_angle_from_horizontal_deg=30.0)
        for c in ctx.grain_classes.classes}, max_active_cohorts=MAX_COHORTS)
    config = SchedulerPhysicsConfig(
        geometry=ctx.geometry, grain_classes=ctx.grain_classes, fluid_threshold_spec=FluidThresholdSpec(),
        aerodynamic_detachment_spec=AerodynamicDetachmentSpec(reference_rate_kg_m2_s=1e-3),
        particle_transport=particle, hop_bins=build_hop_quantile_bins(particle, ctx.grain_classes),
        aeolian_process_mode=AeolianProcessModeSpec(mode="detachment_transport_deposition"),
        impact_rebound_splash_spec=ImpactReboundSplashSpec(), feedback_spec=WindGrainFeedbackSpec(),
        avalanching_spec=ctx.avalanching_spec, process_scheduler_spec=ProcessSchedulerSpec(),
        topographic_commit_spec=ctx.topographic_commit_spec, mass_resolution_kg=ctx.mass_resolution_kg,
        outer_timestep_s=1.0, initial_available_fraction=ctx.initial_available_fraction,
        spatial_block_diagnostics_enabled=True)
    return dataclasses.replace(config, **overrides) if overrides else config


def empty_pool():
    return zeros_mobile_cohort_state(MAX_COHORTS)


def routing_for(state):
    """The reference ringed routing graph and terrain reference around a bed state's committed terrain."""
    zc = np.asarray(committed_elevation_m(state.committed_topography))
    ny, nx = zc.shape
    z = np.full((ny + 2, nx + 2), float(zc.max()) + 1.0)
    z[1:-1, 1:-1] = zc
    z[0, 1:-1] = zc[0] - 0.02
    exports = np.zeros(z.shape, dtype=bool)
    exports[0, 1:-1] = True
    graph = build_routing_graph(z, exports, np.full((ny, nx), 1.0), 0.5)
    return graph, terrain_reference(z, exports, state, graph)


def column(shape):
    def full(v):
        return np.full(shape, float(v))

    return column_parameters(model="fixed_ksat", ksat_m_per_s=full(KSAT), suction_m=full(0.0),
                             drainage_parameter=full(0.0), theta_sat=full(0.4), soil_thickness_m=full(0.3))


def water_inputs(bed, **overrides) -> DenseWaterInputs:
    graph, terrain = routing_for(bed.state)
    inputs = DenseWaterInputs(terrain=terrain, column=column(graph.shape), field=rainfall_field(*graph.shape),
                              vegetation_cover_fraction=np.zeros(graph.shape), sediment=plot1_sediment_parameters(),
                              initial_soil_water_m=np.zeros(graph.shape))
    return dataclasses.replace(inputs, **overrides) if overrides else inputs


def sediment_control(**storm) -> SedimentEventControl:
    options = dict(max_dt_s=1.0, max_steps=10000, implementation="array")
    options.update(storm)
    return SedimentEventControl(storm=StormControl(**options), transport_scheme="upwind")


def policy() -> CompletionPolicy:
    return CompletionPolicy(deposit_mobile_at_flow_end=True)


def runner_control(**overrides) -> DenseRunnerControl:
    control = DenseRunnerControl(sediment_control=sediment_control(), completion_policy=policy(),
                                 report_every_s=REPORT_EVERY_S)
    return dataclasses.replace(control, **overrides) if overrides else control


def limits(**overrides) -> DenseRunnerLimits:
    values = dict(max_events=8, max_wind_accounting_steps=4096, max_storm_steps=10000, max_report_rows=4096,
                  max_work_bytes=1 << 28, max_receipt_bytes=1 << 20)
    values.update(overrides)
    return DenseRunnerLimits(**values)


def wind(duration_s=WIND_S, shear=SHEAR, direction=DIRECTION) -> dict:
    return {"kind": "wind", "duration_s": duration_s, "shear_velocity_m_s": shear,
            "direction_deg_cw_from_north": direction}


def rain(intervals=((RAIN_S, RAIN_MM_H),), guard_s=GUARD_S) -> dict:
    return {"kind": "rain", "recession_guard_s": guard_s,
            "intervals": [{"duration_s": d, "intensity_mm_h": i} for d, i in intervals]}


def schedule(*events) -> DenseSchedule:
    return DenseSchedule.from_input({"schema": "masy.dense_schedule.v1", "events": list(events)})


def reference_schedule() -> DenseSchedule:
    return schedule(wind(), rain(), wind())


def make_runner(bed, events: DenseSchedule, *, config=None, pool=None, water=None, control=None, limit=None):
    return DenseRunner(bed, events, wind=DenseWindInputs(config or wind_config(bed), pool or empty_pool()),
                       water=water or water_inputs(bed), control=control or runner_control(),
                       limits=limit or limits())
