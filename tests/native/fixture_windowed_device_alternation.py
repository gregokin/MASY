"""Labelled Small C16/K1/V6/W3 GPU W/R/W/R fixture; Root alone runs it.

All physical values come from the existing Native Small genesis, prescribed
805 vegetation recipe and 832/994 all-edge-open wet fixture. The independent
oracle calls upstream record/entry/event APIs directly; it never calls a MASY
event delegate. Its work and all retained inputs share the ONE outer budget.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import fixture_dense_alternation as legacy
import fixture_dense_production as production
import numpy as np
from tests.windowed_wind_gpu.test_device_reduced_step import (
    _small_feature_genesis, _device_column, _limits as step_limits,
)
from tests.windowed_wind_gpu.test_device_windowed_record import _limits as block_limits
from tests.windowed_wind_gpu.test_device_avalanche import _solve_limits, _validation_limits
from tests.windowed_wind_gpu.test_device_commit import _commit_limits
from maple.aeolian.cohorts.dispatch import mobile_state_is_genuinely_empty
from maple.aeolian.scheduler import windowed_device as native_wind
from maple.aeolian.scheduler.state import SchedulerPhysicalState, SchedulerState, initial_scheduler_state
from maple.aeolian.scheduler.windowed_record import WindowedWindRecordLimits
from maple.aeolian.trajectories.drag import TRAJECTORY_ALGORITHM_VERSION
from maple.core.backend import to_device, to_device_tree
from maple.core.parameters.wind_forcing import resolve_wind_forcing_record
from maple.core.types.topographic_commit import committed_elevation_m
from maple.core.types.vegetation import uniform_vegetation_raster
from maple.surface.topographic_commit.initialization import initial_committed_topography_state_from_physical_state
from maple.surface.topographic_commit.windowed_commit import windowed_committed_state_from_dense_baseline
from maple_syrup import windowed_shared_bed_device as device_water
from maple_syrup.complete_event import DryHandoff
from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule
from maple_syrup.sediment_bed import routing_graph_from_elevation
from maple_syrup.sediment_event import initial_event_state
from maple_syrup.sediment_physics import physics_grid_from_graph
from maple_syrup.sediment_transport import transport_network
from maple_syrup.storm import initial_state
from maple_syrup.windowed_shared_bed import WindowedSharedBedState

from masy.dense_device_resources import DeviceResourceLimits
from masy.dense_runner import DenseWindInputs
from masy.dense_schedule import WindEvent
from masy.windowed_device_bed import DeviceWindowedBedOwner, DeviceWindowedBedLimits
from masy.windowed_device_runner import DeviceWindowedWindProfile, build_windowed_device_runner

ORIGIN_S = 7.25
VEGETATION_FIELDS = ("height_m", "cover_fraction", "porosity", "drag_coefficient", "frontal_area_density")
AUTHORED_VEGETATION = {"height_m": 0.5, "cover_fraction": 0.25}
CALIBRATION = "synthetic_test_fixture"
LABEL = "authored synthetic100um Small GPU interface; prescribed raster; NOT MEASURED"


def events():
    return legacy.schedule(legacy.wind(1.0, shear=0.5, direction=90.0), legacy.rain(),
                           legacy.wind(1.0, shear=0.5, direction=90.0), legacy.rain())


def bridge_limits(root):
    """Existing994 declarations, unchanged whole-attempt caps and Native0.5."""
    return device_water.PositiveWindowedDeviceWaterLimits(
        max_host_bytes=5 << 30, max_device_bytes=2 << 30, max_mapped_bytes=128 << 20,
        max_logical_read_bytes=(1 << 63)-1, max_work=(1 << 63)-1,
        max_store_bytes=64 << 20, max_run_bytes=512 << 20, max_new_versions=1000000,
        max_entries=4096, max_external_backing_bytes=64 << 20, max_event_steps=10000,
        max_retries=10, max_transport_substeps=64, max_report_rows=4096,
        reserve_host_bytes=64 << 20, reserve_device_bytes=256 << 20,
        safety_fraction=0.5, run_dir=root)


def _water_inputs(host, cp, vegetation):
    """Lossless initial placement of the actual832/994 inputs, no formula copy."""
    water = production.water_inputs(host, all_edges_open=True)
    column = dataclasses.replace(water.column, xp=cp,
        **{f.name: to_device(getattr(water.column, f.name), cp)
           for f in dataclasses.fields(water.column) if type(getattr(water.column, f.name)) is np.ndarray})
    field = dataclasses.replace(water.field, xp=cp, multiplier=to_device(water.field.multiplier, cp))
    sediment = dataclasses.replace(water.sediment, xp=cp,
        **{f.name: to_device(getattr(water.sediment, f.name), cp)
           for f in dataclasses.fields(water.sediment) if f.name.startswith("d_")})
    return dataclasses.replace(water, column=column, field=field, sediment=sediment,
        vegetation_cover_fraction=vegetation.raster.cover_fraction,
        initial_soil_water_m=to_device(water.initial_soil_water_m, cp))


def components(root, cp, qualification_budget):
    """Native genesis/preflight only, no physical event and no observed-fit claim.

    Root has consumed one outer budget fixture before calling this function.
    The original Native oracle frontier covers the fixture+independent oracle;
    it does not start another allowance or supply any opaque allocation proof.
    """
    device_water.reserve_qualification_device_water_oracle_frontier(
        qualification_budget, device_id=int(cp.cuda.runtime.getDevice()))
    root.mkdir(parents=True, exist_ok=False)
    fx = _small_feature_genesis(root)
    p, dense = fx.state.physical, fx.dense_state.physical
    vegetation = uniform_vegetation_raster(4, 4, **AUTHORED_VEGETATION)
    assert tuple(f.name for f in dataclasses.fields(vegetation.raster)) == VEGETATION_FIELDS
    committed = initial_committed_topography_state_from_physical_state(
        dense.voxel_column, dense.active_layer, fx.geo, fx.mres, initial_time_s=ORIGIN_S)
    seed = initial_scheduler_state(p.voxel_column, p.active_layer, p.mobile_state, p.ledger,
        committed, p.threshold_modifier_state, time_s=ORIGIN_S, vegetation=vegetation,
        sediment_availability=p.sediment_availability, water=p.water,
        coupled_trajectory_semantics=TRAJECTORY_ALGORITHM_VERSION)
    placed = to_device_tree(seed, cp)
    column = _device_column(p.voxel_column, cp)
    regrouped, regroup = windowed_committed_state_from_dense_baseline(
        placed.physical.committed_topography, column, placed.physical.active_layer, fx.geo, trusted_packet=True)
    assert regroup["exact_height_checked"] is True
    physical = dataclasses.replace(placed.physical, voxel_column=column, committed_topography=regrouped)
    assert physical.ledger_reconciliation_baseline is None
    state = WindowedSharedBedState(physical.voxel_column, physical.active_layer, physical.water,
        physical.ledger, physical.committed_topography, physical.sediment_availability,
        physical.ledger_reconciliation_baseline)
    fx.step_limits = step_limits()
    policy = to_device(fx.config.initial_available_fraction, cp)
    context = device_water.WindowedDeviceWaterContext(geometry=fx.geo,
        grain_classes=fx.model.grain_classes, mass_resolution_kg=fx.mres,
        topographic_commit_spec=fx.config.topographic_commit_spec, avalanching_spec=fx.config.avalanching_spec,
        initial_available_fraction=policy, window_max_journal_records=320,
        native_solve_limits=_solve_limits(fx.config.avalanching_spec), native_validation_limits=_validation_limits(),
        native_commit_limits=dataclasses.replace(_commit_limits(), publication_max_records=320),
        source_identity=qualification_budget.native_source_identity,
        bridge_limits=bridge_limits(root), runtime_bounds=qualification_budget)
    bed = DeviceWindowedBedOwner.from_device_bed(state, context,
        case_record={"schema": "masy1005.authored_small.v1", "n_voxels": 6, "window_voxels": 3,
            "cells": 16, "classes": 1, "origin_s": ORIGIN_S.hex(), "calibration": CALIBRATION,
            "vegetation": AUTHORED_VEGETATION, "wet_inputs": "existing832_994_all_edges_open"},
        time_s=ORIGIN_S, limits=DeviceWindowedBedLimits(1 << 24, 5 << 30, 2 << 30, 16 << 30),
        vegetation=physical.vegetation, threshold_modifier_state=physical.threshold_modifier_state,
        trajectory_semantics=TRAJECTORY_ALGORITHM_VERSION)
    host = production.build_bed(production.model())  # retained tiny initial terrain/input witness only
    assert host.context.geometry == fx.geo and host.context.grain_classes == fx.model.grain_classes
    water = _water_inputs(host, cp, physical.vegetation)
    control = legacy.runner_control(sediment_control=legacy.sediment_control(
        implementation="cuda", root_solver="bisection"))
    limits = WindowedWindRecordLimits(block_limits(fx), 2, 2, 64)
    profile = DeviceWindowedWindProfile(fx.model, fx.config, limits,
        qualification_budget.native_source_identity, CALIBRATION, LABEL)
    return SimpleNamespace(fx=fx, bed=bed, initial=state, physical=physical, host=host,
        schedule=events(), wind=DenseWindInputs(fx.config, physical.mobile_state), water=water,
        control=control, profile=profile, record_limits=limits, runner_limits=legacy.limits(),
        resource_limits=DeviceResourceLimits(64 << 20, 64 << 20, 2 << 30, 4 << 20),
        qualification_budget=qualification_budget)


def make_runner(case, **overrides):
    options = dict(wind=case.wind, water=case.water, control=case.control, profile=case.profile,
        record_limits=case.record_limits, runner_limits=case.runner_limits,
        resource_limits=case.resource_limits, qualification_budget=case.qualification_budget)
    options.update(overrides)
    return build_windowed_device_runner(case.bed, case.schedule, **options)


def _scheduler(case, state, pool, time_s):
    """Actual twelve-field Native physical owner, preserving the seventh field."""
    return SchedulerState(physical=SchedulerPhysicalState(voxel_column=state.voxel_column,
        active_layer=state.active_layer, mobile_state=pool, ledger=state.ledger,
        committed_topography=state.committed_topography,
        threshold_modifier_state=case.physical.threshold_modifier_state, time_s=time_s,
        vegetation=case.physical.vegetation, sediment_availability=state.sediment_availability,
        water=state.water, coupled_trajectory_semantics=TRAJECTORY_ALGORITHM_VERSION,
        ledger_reconciliation_baseline=state.reconciliation_baseline), coupled=None, block_index=0,
        block_offset_s=0.0, last_resolved_wind=None, topographic_wind_cache=None,
        vegetation_wind_cache=None, active_layer_composition_cache=None)


def direct_events(case, cp):
    """Sequential Native record+entry+complete calls, independent of the runner."""
    state, ctx, t = case.bed._accepted.state, case.bed.context, case.bed.time_s
    pool, soil = case.wind.initial_mobile_state, case.water.initial_soil_water_m.copy()
    cfg = dataclasses.replace(case.wind.config, initial_available_fraction=ctx.initial_available_fraction)
    held = None
    for event in case.schedule.events:
        if type(event) is WindEvent:
            assert mobile_state_is_genuinely_empty(pool)
            record = resolve_wind_forcing_record({"blocks": [{"duration_s": event.duration_s,
                "shear_velocity_m_s": event.shear_velocity_m_s,
                "direction_deg": event.direction_deg_cw_from_north}]})
            owner = native_wind.prepare_device_windowed_small_feature_wind_record(
                _scheduler(case, state, pool, t), cfg, record, limits=case.record_limits,
                source_identity=case.profile.source_identity, record_origin_s=t)
            result = owner.advance()
            assert owner.status == "completed" and owner.block_count == owner.finalizer_call_count == 1
            assert result.completed and result.finalization.completed and result.finalization.water_handoff_ready
            p = result.state.physical
            assert mobile_state_is_genuinely_empty(p.mobile_state)
            state = WindowedSharedBedState(p.voxel_column, p.active_layer, p.water, p.ledger,
                p.committed_topography, p.sediment_availability, p.ledger_reconciliation_baseline)
            pool, t = p.mobile_state, p.time_s
            yield SimpleNamespace(kind="wind", state=state, time_s=t, pool=pool, soil=soil,
                routing=held, owner=owner, result=result)
            continue
        w = case.water
        if held is None:
            graph = routing_graph_from_elevation(w.terrain,
                committed_elevation_m(state.committed_topography), xp=cp)
            held = initial_event_state(graph, w.terrain, state, ctx, soil, t_s=t,
                control=case.control.sediment_control)
        boundary, entry_summary = device_water.commit_qualification_device_entry(state, ctx,
            held.graph, held.terrain, t, held.routing_source_elevation_m,
            control=case.control.sediment_control)
        graph = boundary.graph
        entry = dataclasses.replace(held, bed=boundary.state, graph=graph,
            network=transport_network(graph) if boundary.rerouted else held.network,
            grid=physics_grid_from_graph(graph) if boundary.rerouted else held.grid,
            routing_source_elevation_m=boundary.routing_source_m,
            storm=initial_state(graph, boundary.state.water.depth_m, soil, t_s=t),
            sediment_velocity_m_s=held.sediment_velocity_m_s.copy(), phase=None)
        edges, offset = [t], 0.0
        for interval in event.intervals:
            offset += interval.duration_s
            edges.append(t+offset)
        rainfall = RainfallSchedule(edges_s=edges,
            intensity_mm_per_h=[value.intensity_mm_h for value in event.intervals],
            provenance=RainfallProvenance(kind="constant"))
        wet, event_summary = device_water.complete_qualification_device_event(entry, ctx, w.column,
            w.field, rainfall, w.vegetation_cover_fraction, w.sediment,
            max_end_s=edges[-1]+event.recession_guard_s, control=case.control.sediment_control,
            policy=case.control.completion_policy, report_every_s=case.control.report_every_s)
        assert type(wet) is DryHandoff and wet.status == "complete"
        held, state, t = wet.dry_state, wet.dry_state.bed, wet.dry_state.t_s
        soil = held.storm.soil_water_m
        yield SimpleNamespace(kind="rain", state=state, time_s=t, pool=pool, soil=soil, routing=held,
            wet=wet, entry_commit=boundary, entry_summary=entry_summary, event_summary=event_summary)
