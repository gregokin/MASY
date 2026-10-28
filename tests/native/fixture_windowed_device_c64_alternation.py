"""Labelled C64/K1/V20/W3 GPU W/R/W/R fixture; Root alone runs it.

All physical values come from the existing Native C64 genesis, prescribed
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
from tests.windowed_wind_gpu.test_device_k1_dimension import _dimension_fixture
from maple.aeolian.cohorts.dispatch import mobile_state_is_genuinely_empty
from maple.aeolian.scheduler import windowed_device as native_wind
from maple.aeolian.scheduler.state import SchedulerPhysicalState, SchedulerState
from maple.aeolian.trajectories.drag import TRAJECTORY_ALGORITHM_VERSION
from maple.core.backend import to_device, to_device_tree
from maple.core.parameters.wind_forcing import resolve_wind_forcing_record
from maple.core.types.topographic_commit import committed_elevation_m
from maple.core.types.vegetation import uniform_vegetation_raster
from maple_syrup import windowed_shared_bed_device as device_water
from maple_syrup.complete_event import DryHandoff
from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule
from maple_syrup.sediment_bed import BedContext, routing_graph_from_elevation
from maple_syrup.sediment_event import initial_event_state
from maple_syrup.sediment_physics import physics_grid_from_graph
from maple_syrup.sediment_transport import transport_network
from maple_syrup.storm import initial_state
from maple_syrup.windowed_shared_bed import WindowedSharedBedState

from masy.dense_device_resources import DeviceResourceLimits
from masy.clock import decode_time
from masy.dense_runner import DenseWindInputs
from masy.dense_schedule import WindEvent, resolve_event
from masy.windowed_device_c64_bed import (
    DeviceWindowedC64BedOwner, DeviceWindowedC64BedLimits, _require_completed_scale_api,
)
from masy.windowed_device_c64_runner import DeviceWindowedC64WindProfile, build_windowed_device_c64_runner

ORIGIN_S = 1234.0
VEGETATION_FIELDS = ("height_m", "cover_fraction", "porosity", "drag_coefficient", "frontal_area_density")
AUTHORED_VEGETATION = {"height_m": 0.5, "cover_fraction": 0.25}
CALIBRATION = "synthetic_test_fixture"
LABEL = "authored synthetic100um C64 GPU interface; prescribed raster; NOT MEASURED"


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
    # User-selected native strict-pit storage; preserve the original ring and flat-storage=False.
    return dataclasses.replace(water, terrain=dataclasses.replace(water.terrain, allow_pit_storage=True),
        column=column, field=field, sediment=sediment,
        vegetation_cover_fraction=vegetation.raster.cover_fraction,
        initial_soil_water_m=to_device(water.initial_soil_water_m, cp))


@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class _NativeWaterView:
    """Initial terrain/input witness over the genuine C64 Dense fixture only."""
    state: object
    context: object
    vegetation: object
    time_s: float

    def vegetation_cover_fraction(self):
        return self.vegetation.raster.cover_fraction


def components(root, cp, qualification_budget):
    """One genuine Native C64 genesis; Root supplies whole-coupled authority.

    A standalone water descriptor alone does not authorize this fixture, two
    initializations, four events, independent oracle or retained MASY owners.
    The separate Root outer grant must bind this complete Source/caller role.
    """
    _require_completed_scale_api(preflight=True)
    device_water.reserve_c64_qualification_device_water_oracle_frontier(
        qualification_budget, device_id=int(cp.cuda.runtime.getDevice()))
    root.mkdir(parents=True, exist_ok=False)
    fx = _dimension_fixture(root, cp)
    p = fx.device_state.physical
    assert p.time_s.hex() == ORIGIN_S.hex()
    assert (fx.geo.ny, fx.geo.nx, p.voxel_column.n_voxels,
            p.voxel_column.deep_store.header.window_voxels) == (8, 8, 20, 3)
    assert fx.profile is fx.limits.block.profile is fx.limits.block.step.profile
    assert fx.regrouping["exact_height_checked"] is True
    host_vegetation = uniform_vegetation_raster(8, 8, **AUTHORED_VEGETATION)
    assert tuple(f.name for f in dataclasses.fields(host_vegetation.raster)) == VEGETATION_FIELDS
    vegetation = to_device_tree(host_vegetation, cp)
    physical = dataclasses.replace(p, vegetation=vegetation,
        coupled_trajectory_semantics=TRAJECTORY_ALGORITHM_VERSION)
    assert physical.ledger_reconciliation_baseline is None
    state = WindowedSharedBedState(physical.voxel_column, physical.active_layer, physical.water,
        physical.ledger, physical.committed_topography, physical.sediment_availability,
        physical.ledger_reconciliation_baseline)
    policy = to_device(fx.config.initial_available_fraction, cp)
    context = device_water.WindowedDeviceWaterC64Context(geometry=fx.geo,
        grain_classes=fx.model.grain_classes, mass_resolution_kg=fx.mres,
        topographic_commit_spec=fx.config.topographic_commit_spec, avalanching_spec=fx.config.avalanching_spec,
        initial_available_fraction=policy, window_max_journal_records=320,
        native_solve_limits=fx.limits.block.solve, native_validation_limits=fx.limits.block.validation,
        native_commit_limits=dataclasses.replace(fx.limits.block.commit, publication_max_records=320),
        native_dimension_profile=fx.profile, source_identity=qualification_budget.native_source_identity,
        bridge_limits=bridge_limits(root), runtime_bounds=qualification_budget)
    bed = DeviceWindowedC64BedOwner.from_device_bed(state, context,
        case_record={"schema": "masy1040.authored_C64.v1", "n_voxels": 20, "window_voxels": 3,
            "cells": 64, "classes": 1, "origin_s": ORIGIN_S.hex(), "calibration": CALIBRATION,
            "vegetation": AUTHORED_VEGETATION, "wet_inputs": "existing832_994_all_edges_open",
            "water_routing": {"allow_pit_storage": True, "allow_flat_storage": False},
            "water_completion": {"schema": "masy.interim_accounted_ponded_water.v1",
                "terminal_storage_water": True, "deposit_mobile_at_flow_end": True,
                "removal": "native_external_dry_reset", "evapotranspiration_simulated": False}},
        time_s=ORIGIN_S, limits=DeviceWindowedC64BedLimits(1 << 24, 5 << 30, 2 << 30, 16 << 30),
        vegetation=physical.vegetation, threshold_modifier_state=physical.threshold_modifier_state,
        trajectory_semantics=TRAJECTORY_ALGORITHM_VERSION)
    host_context = BedContext(geometry=fx.geo, grain_classes=fx.model.grain_classes,
        mass_resolution_kg=fx.mres, topographic_commit_spec=fx.config.topographic_commit_spec,
        avalanching_spec=fx.config.avalanching_spec, initial_available_fraction=fx.config.initial_available_fraction)
    host = _NativeWaterView(fx.dense_state.physical, host_context, host_vegetation, ORIGIN_S)
    water = _water_inputs(host, cp, physical.vegetation)
    control = legacy.runner_control(sediment_control=legacy.sediment_control(
        implementation="cuda", root_solver="bisection"))
    control = dataclasses.replace(control,
        completion_policy=dataclasses.replace(control.completion_policy, terminal_storage_water=True))
    limits = fx.limits
    assert limits.max_blocks == limits.max_total_outer_steps == limits.block.max_outer_steps == 1
    profile = DeviceWindowedC64WindProfile(fx.model, fx.config, limits,
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
    return build_windowed_device_c64_runner(case.bed, case.schedule, **options)


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
    for index, event in enumerate(case.schedule.events):
        if type(event) is WindEvent:
            assert mobile_state_is_genuinely_empty(pool)
            record = resolve_wind_forcing_record({"blocks": [{"duration_s": event.duration_s,
                "shear_velocity_m_s": event.shear_velocity_m_s,
                "direction_deg": event.direction_deg_cw_from_north}]})
            owner = native_wind.prepare_device_windowed_k1_dimension_wind_record(
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
        boundary, entry_summary = device_water.commit_c64_qualification_device_entry(state, ctx,
            held.graph, held.terrain, t, held.routing_source_elevation_m,
            control=case.control.sediment_control)
        graph = boundary.graph
        entry = dataclasses.replace(held, bed=boundary.state, graph=graph,
            network=transport_network(graph) if boundary.rerouted else held.network,
            grid=physics_grid_from_graph(graph) if boundary.rerouted else held.grid,
            routing_source_elevation_m=boundary.routing_source_m,
            storm=initial_state(graph, boundary.state.water.depth_m, soil, t_s=t),
            sediment_velocity_m_s=held.sediment_velocity_m_s.copy(), phase=None)
        resolved = resolve_event(case.schedule, index, t)
        native_edges = resolved["resolved_edges"]["edges"]
        edges = [decode_time(edge["native"]) for edge in native_edges[:-1]]
        rainfall = RainfallSchedule(edges_s=edges,
            intensity_mm_per_h=[value.intensity_mm_h for value in event.intervals],
            provenance=RainfallProvenance(kind="constant"))
        wet, event_summary = device_water.complete_c64_qualification_device_event(entry, ctx, w.column,
            w.field, rainfall, w.vegetation_cover_fraction, w.sediment,
            max_end_s=decode_time(native_edges[-1]["native"]), control=case.control.sediment_control,
            policy=case.control.completion_policy, report_every_s=case.control.report_every_s)
        assert type(wet) is DryHandoff and wet.status == "complete"
        held, state, t = wet.dry_state, wet.dry_state.bed, wet.dry_state.t_s
        soil = held.storm.soil_water_m
        yield SimpleNamespace(kind="rain", state=state, time_s=t, pool=pool, soil=soil, routing=held,
            wet=wet, entry_commit=boundary, entry_summary=entry_summary, event_summary=event_summary)
