"""Authored synthetic 100um interface fixture, NOT a measured calibration.

Reuses native labelled reduced-test coefficients, range and composition guard.
Explicitly tightens that fixture's native warn policy to reject. Wet Plot1 class-1 coefficients are
explicitly reused at the shared 100um diameter: this changes the native 62.5um wet
fixture and is NOT its unchanged calibration. No production science is qualified.
Periodic wind / open water and constant_depth were explicitly accepted by user.
"""
from __future__ import annotations

import copy
import dataclasses

import fixture_dense_alternation as legacy
import numpy as np
from maple.core.parameters import resolve_config, validate_config
from maple.core.types.cohort import zeros_mobile_cohort_state
from maple.core.types.threshold_modifiers import neutral_threshold_modifier_state
from maple.core.types.topographic_commit import committed_elevation_m
from maple.core.types.vegetation import empty_vegetation_state
from maple_syrup import sediment_physics as wet_physics
from maple_syrup.rainfall import rainfall_field
from maple_syrup.routing import build_routing_graph
from maple_syrup.sediment_bed import terrain_reference
from tests.phase22b.test_reduced_detachment_config import _MODEL_RAW
from tests.phase22b.test_reduced_integration import SYNTHETIC_FIXTURE_CALIBRATION

from masy.dense_bed import DenseBedLimits, build_dense_bed
from masy.dense_production import DenseProductionLimits, schedule_envelope
from masy.dense_runner import DenseRunner, DenseWaterInputs, DenseWindInputs
from masy.dense_wind_profile import (
    DenseProductionWindProfile,
    build_dense_production_wind_config,
)


def model(*, outer=1.0, creep=True, avalanche=True, shape=(4, 4)):
    raw = copy.deepcopy(_MODEL_RAW)
    raw["geometry"].update(ny=shape[0], nx=shape[1])
    calibration = copy.deepcopy(SYNTHETIC_FIXTURE_CALIBRATION)
    assert calibration["out_of_envelope_policy"] == "warn"
    calibration["out_of_envelope_policy"] = "reject"  # stricter authored interface policy; no warning waiver
    raw["detachment_source"] = {"mode": "reduced"}
    raw["reduced_detachment"] = calibration
    raw["grain_classes"][0]["diameter_m"] = calibration["calibrated_grains"][0]["diameter_m"]
    raw["particle_transport"] = {
        "quantile_bins": 8, "max_active_cohorts": 4000, "mobile_state_representation": "flat",
        "hop_distance_by_class": {"sand": {"median_hop_distance_m": 0.3,
            "geometric_stddev": 2.0, "representative_flight_speed_m_s": 3.0}}}
    raw["time_and_forcing"] = {"outer_timestep_s": outer}
    raw["topographic_wind"] = {"enabled": True}
    raw["flow_separation"] = {"enabled": True}
    raw["surface_creep"] = {"enabled": creep}
    raw["avalanching"] = {"enabled": avalanche}
    raw.setdefault("water_coupling", {}).update(enabled=False, depth_update_rule="constant_depth")
    # Existing native numerical commit fixture trigger, not a wrapper hydraulic gate.
    raw.setdefault("topographic_commit", {}).update(pending_elevation_trigger_m=1e-9)
    cfg = resolve_config(raw)
    validate_config(cfg)
    return cfg


def build_bed(cfg, *, mound=True):
    g = cfg.geometry
    # Small cross slope gives an actual D4 drainage path around the isolated mound.
    z = 0.128 + 0.01 * np.arange(g.ny)[:, None] + 0.002 * np.arange(g.nx)[None, :]
    if mound:
        z[g.ny // 2, g.nx // 2] += 0.256
    return build_dense_bed(
        geometry=g, grain_classes=cfg.grain_classes, mass_resolution_kg=cfg.numerics.mass_resolution_kg,
        n_voxels=6, initial_bed_elevation_m=z, initial_class_fractions=np.ones((g.ny, g.nx, 1)),
        initial_available_fraction=np.full((g.ny, g.nx, 1), 0.25), initial_water_depth_m=np.zeros((g.ny, g.nx)),
        vegetation=empty_vegetation_state(g.ny, g.nx),
        threshold_modifier_state=neutral_threshold_modifier_state(g.ny, g.nx),
        topographic_commit_spec=cfg.topographic_commit, avalanching_spec=cfg.avalanching,
        active_mask=np.ones((g.ny, g.nx), dtype=bool), initial_time_s=0.0,
        limits=DenseBedLimits(host_memory_cap_bytes=1 << 28, max_voxels_per_column=16,
                              stage_copy_max_bytes=1 << 24))


def water_inputs(bed, *, all_edges_open=False):
    p = wet_physics
    sediment = p.sediment_physics_parameters(
        diameter_m=[bed.context.grain_classes.classes[0].diameter_m], raindrop_a=p.PLOT1_RAINDROP_A[:1],
        raindrop_b=p.PLOT1_RAINDROP_B[:1], raindrop_c=p.PLOT1_RAINDROP_C[:1],
        raindrop_max_depth_mm=p.PLOT1_RAINDROP_MAX_MM[:1], particle_density_g_cm3=p.PLOT1_PARTICLE_DENSITY_G_CM3,
        active_layer_sensitivity_mm=p.PLOT1_ACTIVE_LAYER_SENSITIVITY_MM, ke_model=p.PLOT1_KE_MODEL,
        raindrop_depth_attenuation_per_cm=p.LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM[:1])
    zc = committed_elevation_m(bed.state.committed_topography)
    ny, nx = zc.shape
    z = np.full((ny + 2, nx + 2), float(zc.max()) + 1.0)
    z[1:-1, 1:-1] = zc
    z[0, 1:-1] = zc[0] - 0.02
    exports = np.zeros(z.shape, dtype=bool)
    exports[0, 1:-1] = True
    if all_edges_open:
        # Separate authored 3x3 avalanche witness with native open edge outlets.
        # No sink policy, filling or carving is enabled.
        z[0, 1:-1] = z[-1, 1:-1] = 0.0
        z[1:-1, 0] = z[1:-1, -1] = 0.0
        exports[0, 1:-1] = exports[-1, 1:-1] = True
        exports[1:-1, 0] = exports[1:-1, -1] = True
    graph = build_routing_graph(z, exports, np.ones((ny, nx)), bed.context.geometry.dx_m)
    terrain = terrain_reference(z, exports, bed.state, graph)
    return DenseWaterInputs(terrain=terrain, column=legacy.column(graph.shape), field=rainfall_field(*graph.shape),
                           vegetation_cover_fraction=bed.vegetation_cover_fraction(), sediment=sediment,
                           initial_soil_water_m=np.zeros(graph.shape))


def four_events():
    return legacy.schedule(legacy.wind(1.0), legacy.rain(), legacy.wind(1.0), legacy.rain())


def components(*, events=None, cfg=None, mound=True):
    cfg = model() if cfg is None else cfg
    events = four_events() if events is None else events
    bed = build_bed(cfg, mound=mound)
    profile = DenseProductionWindProfile(cfg, "synthetic_test_fixture",
        "authored MASY16 interface fixture; native synthetic100um reduced coefficients; wet Plot1 class1 reused at100um; NOT MEASURED")
    envelope, _ = schedule_envelope(events)
    wind_cfg = build_dense_production_wind_config(profile, wind_envelope_record=envelope,
                                                 initial_available_fraction=bed.context.initial_available_fraction)
    wind = DenseWindInputs(wind_cfg, zeros_mobile_cohort_state(cfg.particle_transport.max_active_cohorts))
    return bed, events, wind, water_inputs(bed), profile


def make_runner(*, events=None, cfg=None, mound=True, **overrides):
    bed, events, wind, water, profile = components(events=events, cfg=cfg, mound=mound)
    options = {"wind": wind, "water": water, "control": legacy.runner_control(), "limits": legacy.limits(),
               "wind_profile": profile, "production_limits": DenseProductionLimits()}
    options.update(overrides)
    return DenseRunner(bed, events, **options)


def direct_control(bed, cfg, events, water):
    """Independent native calls only. Does not call DenseRunner or its delegates."""
    from maple.aeolian.scheduler.block import advance_wind_block
    from maple.aeolian.scheduler.finalization import finalize_wind_event
    from maple.aeolian.scheduler.state import initial_scheduler_state
    from maple.core.parameters.wind_forcing import (
        WindBlock,
        resolve_wind_forcing_record,
    )
    from maple_syrup.complete_event import DryHandoff, complete_event
    from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule
    from maple_syrup.sediment_bed import commit_bed, routing_graph_from_elevation
    from maple_syrup.sediment_event import initial_event_state
    from maple_syrup.sediment_physics import physics_grid_from_graph
    from maple_syrup.sediment_transport import transport_network
    from maple_syrup.storm import initial_state

    from masy.dense_schedule import WindEvent
    state, ctx, t = bed.state, bed.context, bed.time_s
    pool = zeros_mobile_cohort_state(cfg.particle_transport.max_active_cohorts)
    soil = np.array(water.initial_soil_water_m, copy=True)
    held = None
    control = legacy.sediment_control()
    for event in events.events:
        if type(event) is WindEvent:
            scheduler = initial_scheduler_state(
                state.voxel_column, state.active_layer, pool, state.ledger, state.committed_topography,
                bed.threshold_modifier_state, time_s=t, vegetation=bed.vegetation,
                sediment_availability=state.sediment_availability, water=state.water)
            block = WindBlock(event.shear_velocity_m_s, event.direction_deg_cw_from_north, event.duration_s,
                              t, t + event.duration_s)
            result = advance_wind_block(scheduler, block, cfg)
            assert result.conservation_ok
            record = resolve_wind_forcing_record({"blocks": [{"duration_s": event.duration_s,
                "shear_velocity_m_s": event.shear_velocity_m_s, "direction_deg": event.direction_deg_cw_from_north}]})
            completed = dataclasses.replace(result.state, block_index=1, block_offset_s=0.0)
            final = finalize_wind_event(completed, cfg, record, record_origin_s=t)
            assert final.completed and final.water_handoff_ready
            p = final.state.physical
            state = dataclasses.replace(state, voxel_column=p.voxel_column, active_layer=p.active_layer,
                ledger=p.ledger, committed_topography=p.committed_topography, water=p.water,
                sediment_availability=p.sediment_availability)
            pool, t = p.mobile_state, p.time_s
            yield state, t, {"kind": "wind", "result": result, "final": final}
            continue
        if held is None:
            graph = routing_graph_from_elevation(water.terrain, committed_elevation_m(state.committed_topography), xp=np)
            held = initial_event_state(graph, water.terrain, state, ctx, soil, t_s=t, control=control)
        boundary = commit_bed(state, ctx, held.graph, held.terrain, t, force=True,
                              routing_source_m=held.routing_source_elevation_m)
        graph = boundary.graph
        entry = dataclasses.replace(held, bed=boundary.state, graph=graph,
            network=transport_network(graph) if boundary.rerouted else held.network,
            grid=physics_grid_from_graph(graph) if boundary.rerouted else held.grid,
            routing_source_elevation_m=boundary.routing_source_m,
            storm=initial_state(graph, boundary.state.water.depth_m, soil, t_s=t))
        offsets = np.cumsum([interval.duration_s for interval in event.intervals]).tolist()
        rainfall = RainfallSchedule(edges_s=[t] + [t + value for value in offsets],
            intensity_mm_per_h=[interval.intensity_mm_h for interval in event.intervals],
            provenance=RainfallProvenance(kind="constant"))
        wet = complete_event(entry, ctx, water.column, water.field, rainfall,
            water.vegetation_cover_fraction, water.sediment, max_end_s=t + offsets[-1] + event.recession_guard_s,
            control=control, policy=legacy.policy(), report_every_s=legacy.REPORT_EVERY_S, max_report_rows=4096)
        assert type(wet) is DryHandoff
        held, state, t = wet.dry_state, wet.dry_state.bed, wet.dry_state.t_s
        soil = held.storm.soil_water_m
        yield state, t, {"kind": "rain", "wet": wet, "entry_commit": boundary}
