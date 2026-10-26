"""Synthetic K1/C16/V6/W3 CPU interface fixture; Root alone executes it.

The genuine Native Small factory supplies the existing production-reduced,
FFT/separation/creep/avalanche profile and its metadata-planned limits. Its
labelled 100um calibration is not measured production calibration. Initial
physical time 7.25 is supplied to real Native constructors BEFORE transport.
The prescribed raster recipe is the existing805 fixture: authored height0.5m,
cover0.25; its other three canonical arrays use Native defaults. Both models
receive that one complete VegetationState and its exact cover object.

No adapter constructs a Dense bed at an event handoff. The small Dense genesis
retained by the upstream test factory is an initial fixture witness only.
All original worker, Native and persistent-store limits remain authoritative.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import fixture_dense_alternation as legacy
import fixture_dense_production as production
import numpy as np
from maple.aeolian.cohorts.dispatch import mobile_state_is_genuinely_empty
from maple.aeolian.scheduler.state import SchedulerPhysicalState, SchedulerState
from maple.aeolian.scheduler.windowed_record import (
    WindowedWindRecordLimits, prepare_windowed_small_feature_wind_record,
)
from maple.core.backend.depth_class_reductions import qualify_depth_class_order
from maple.core.parameters.wind_forcing import resolve_wind_forcing_record
from maple.core.types.cohort import zeros_mobile_cohort_state
from maple.core.types.topographic_commit import initial_committed_topography_state, committed_elevation_m
from maple.core.types.vegetation import uniform_vegetation_raster
from maple_syrup.complete_event import DryHandoff
from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule
from maple_syrup.sediment_bed import routing_graph_from_elevation
from maple_syrup.sediment_event import initial_event_state
from maple_syrup.sediment_physics import physics_grid_from_graph
from maple_syrup.sediment_transport import transport_network
from maple_syrup.storm import initial_state
from maple_syrup.windowed_shared_bed import (
    PositiveWindowedWaterBridgeLimits, WindowedAvalancheContext, WindowedSharedBedState,
    commit_positive_windowed_entry, complete_positive_windowed_event,
)
from tests.windowed_wind import test_w6_windowed_event_finalization as native_small

from masy.windowed_process import WindowedProductionLimits, check_process
from masy.dense_runner import DenseWindInputs
from masy.dense_schedule import WindEvent
from masy.windowed_bed import WindowedBedLimits, build_windowed_bed
from masy.windowed_runner import WindowedWindProfile, build_windowed_runner

ORIGIN_S = 7.25
AUTHORED_VEGETATION = {"height_m": 0.5, "cover_fraction": 0.25}
VEGETATION_FIELDS = ("height_m", "cover_fraction", "porosity", "drag_coefficient", "frontal_area_density")
CALIBRATION = "synthetic_test_fixture"
LABEL = "authored synthetic100um Small interface; wet Plot1class1 at100um; prescribed raster; NOT MEASURED"


def events(*, count=4):
    if type(count) is not int or count not in (2, 4):
        raise ValueError("only the ordered two-event gate or four-event interface case")
    ordered = (legacy.wind(1.0, shear=0.5, direction=90.0), legacy.rain(),
               legacy.wind(1.0, shear=0.5, direction=90.0), legacy.rain())
    return legacy.schedule(*ordered[:count])


def bridge_limits(run_dir):
    """Exact existing832/815 positive-water policy, never a fit inferred from telemetry."""
    return PositiveWindowedWaterBridgeLimits(
        max_host_bytes=4 << 30, max_mapped_bytes=128 << 20,
        max_logical_read_bytes=(1 << 63) - 1, max_work=(1 << 63) - 1,
        max_store_bytes=64 << 20, max_run_bytes=128 << 20,
        max_new_versions=1000000, max_entries=4096,
        max_external_backing_bytes=64 << 20, max_event_steps=10000,
        max_retries=10, max_transport_substeps=64, max_report_rows=4096,
        reserve_host_bytes=64 << 20, run_dir=str(run_dir))


def components(tmp_path, *, count=4, process_profile="ownership"):
    """Genuine constructors/preflights only; does no event transport.

    The Native Small fixture retains its initial Dense witness and immutable
    old store. Neither is freed/deleted or credited to an admission here.
    Root must qualify the final full Source/runtime/cold images independently.
    """
    production_limits = WindowedProductionLimits.for_profile(process_profile)
    check_process(production_limits, checkpoint="fixture_before_genesis")
    tmp_path.mkdir(parents=True, exist_ok=True)
    fx = native_small._small_feature_fixture(tmp_path, _record_host_envelope=True)
    p = fx.state.physical
    assert (fx.geo.ny, fx.geo.nx, len(fx.ids), p.voxel_column.n_voxels,
            p.voxel_column.deep_store.header.window_voxels) == (4, 4, 1, 6, 3)
    assert fx.config.avalanching_spec.enabled and fx.config.flow_separation_spec.enabled
    assert fx.config.topographic_wind_spec.enabled and fx.config.surface_creep_spec.active
    assert fx.config.particle_transport.quantile_bins == 8
    assert fx.config.particle_transport.max_active_cohorts == 4000
    assert fx.limits.step.bed_limits().max_new_host_bytes == 64 << 20
    assert fx.limits.step.evidence_limits().max_evidence_bytes == 4 << 20
    # The initial mass baselines come from the genuine Native diagnosis in
    # the upstream genesis, before either event. No evolved clock is rebased.
    old = p.committed_topography
    committed = initial_committed_topography_state(
        committed_elevation_m(old), old.committed_active_layer_mass_by_class_kg,
        old.committed_voxel_mass_by_class_kg, initial_time_s=ORIGIN_S)
    vegetation = uniform_vegetation_raster(4, 4, **AUTHORED_VEGETATION)
    assert tuple(f.name for f in dataclasses.fields(vegetation.raster)) == VEGETATION_FIELDS
    physical = SchedulerPhysicalState(
        voxel_column=p.voxel_column, active_layer=p.active_layer,
        mobile_state=zeros_mobile_cohort_state(4000), ledger=p.ledger,
        committed_topography=committed, threshold_modifier_state=p.threshold_modifier_state,
        time_s=ORIGIN_S, vegetation=vegetation, sediment_availability=p.sediment_availability,
        water=p.water, coupled_trajectory_semantics=p.coupled_trajectory_semantics,
        ledger_reconciliation_baseline=p.ledger_reconciliation_baseline)
    assert physical.ledger_reconciliation_baseline is None  # no synthetic post-run reference
    qualify_depth_class_order(6, 1)  # existing genuine order qualification, no forged cache marker
    l = fx.limits
    context = WindowedAvalancheContext(
        geometry=fx.geo, grain_classes=fx.model.grain_classes, mass_resolution_kg=fx.mres,
        topographic_commit_spec=fx.config.topographic_commit_spec, avalanching_spec=fx.config.avalanching_spec,
        initial_available_fraction=fx.config.initial_available_fraction, depth_update_rule="constant_depth",
        record_exchange_history=False, window_max_journal_records=l.iteration.max_journal_records,
        native_iteration_limits=l.iteration, native_evidence_limits=l.evidence,
        native_solve_limits=l.solve, native_validation_limits=l.validation,
        native_publication_limits=l.publication, native_qualification_limits=l.qualification,
        source_identity=native_small.w5.SOURCE_IDENTITY, bridge_limits=bridge_limits(tmp_path))
    bed = build_windowed_bed(physical=physical, context=context,
        limits=WindowedBedLimits(stage_copy_max_bytes=1 << 24, held_input_max_bytes=1 << 24))
    schedule = events(count=count)
    profile = WindowedWindProfile(fx.model, fx.config,
        WindowedWindRecordLimits(l, max_blocks=2, max_total_outer_steps=2, max_accounting_reasons=64),
        native_small.w5.SOURCE_IDENTITY, CALIBRATION, LABEL)
    wind = DenseWindInputs(fx.config, physical.mobile_state)
    water = production.water_inputs(bed, all_edges_open=True)  # existing all-edge Native outlets
    assert water.vegetation_cover_fraction is vegetation.raster.cover_fraction
    return SimpleNamespace(fx=fx, bed=bed, schedule=schedule, profile=profile, wind=wind, water=water,
        control=legacy.runner_control(), limits=legacy.limits(), production_limits=production_limits)


def make_runner(case, **overrides):
    options = dict(wind=case.wind, water=case.water, control=case.control, limits=case.limits,
                   profile=case.profile, production_limits=case.production_limits)
    options.update(overrides)
    return build_windowed_runner(case.bed, case.schedule, **options)


def _scheduler(bed, state, pool, time_s):
    """Direct Native twelve-field holder; never calls a MASY event delegate."""
    return SchedulerState(physical=SchedulerPhysicalState(
        voxel_column=state.voxel_column, active_layer=state.active_layer, mobile_state=pool, ledger=state.ledger,
        committed_topography=state.committed_topography, threshold_modifier_state=bed.threshold_modifier_state,
        time_s=time_s, vegetation=bed.vegetation, sediment_availability=state.sediment_availability,
        water=state.water, coupled_trajectory_semantics=bed.trajectory_semantics,
        ledger_reconciliation_baseline=state.reconciliation_baseline), coupled=None, block_index=0,
        block_offset_s=0.0, last_resolved_wind=None, topographic_wind_cache=None,
        vegetation_wind_cache=None, active_layer_composition_cache=None)


def direct_events(case):
    """Independent actual Native calls on a separately constructed case.

    No WindowedRunner step/delegates/acceptance/publication are used here.
    All old Native results and version handles may be retained by the caller.
    """
    bed, cfg = case.bed, case.wind.config
    state, ctx, t = bed.state, bed.context, bed.time_s
    pool = case.wind.initial_mobile_state
    soil = np.array(case.water.initial_soil_water_m, copy=True)
    held = None
    for event in case.schedule.events:
        if type(event) is WindEvent:
            assert mobile_state_is_genuinely_empty(pool)
            scheduler = _scheduler(bed, state, pool, t)
            record = resolve_wind_forcing_record({"blocks": [{"duration_s": event.duration_s,
                "shear_velocity_m_s": event.shear_velocity_m_s,
                "direction_deg": event.direction_deg_cw_from_north}]})
            owner = prepare_windowed_small_feature_wind_record(scheduler, cfg, record,
                limits=case.profile.record_limits, source_identity=case.profile.source_identity, record_origin_s=t)
            result = owner.advance()
            assert result.completed and result.finalization.water_handoff_ready
            assert owner.status == "completed" and owner.block_count == 1 and owner.finalizer_call_count == 1
            p = result.state.physical
            assert mobile_state_is_genuinely_empty(p.mobile_state)
            state = WindowedSharedBedState(p.voxel_column, p.active_layer, p.water, p.ledger,
                p.committed_topography, p.sediment_availability, p.ledger_reconciliation_baseline)
            pool, t = p.mobile_state, p.time_s
            yield SimpleNamespace(kind="wind", state=state, time_s=t, pool=pool, soil=soil, routing=held,
                                  result=result, owner=owner)
            continue
        w = case.water
        if held is None:
            graph = routing_graph_from_elevation(w.terrain, committed_elevation_m(state.committed_topography), xp=np)
            held = initial_event_state(graph, w.terrain, state, ctx, soil, t_s=t,
                                       control=case.control.sediment_control)
        boundary, entry_summary = commit_positive_windowed_entry(state, ctx, held.graph, held.terrain, t,
            held.routing_source_elevation_m, control=case.control.sediment_control)
        graph = boundary.graph
        entry = dataclasses.replace(held, bed=boundary.state, graph=graph,
            network=transport_network(graph) if boundary.rerouted else held.network,
            grid=physics_grid_from_graph(graph) if boundary.rerouted else held.grid,
            routing_source_elevation_m=boundary.routing_source_m,
            storm=initial_state(graph, boundary.state.water.depth_m, soil, t_s=t),
            sediment_velocity_m_s=np.array(held.sediment_velocity_m_s, copy=True), phase=None)
        elapsed = 0.0
        edges = [t]
        for interval in event.intervals:
            elapsed += interval.duration_s
            edges.append(t + elapsed)
        rainfall = RainfallSchedule(edges_s=edges,
            intensity_mm_per_h=[v.intensity_mm_h for v in event.intervals],
            provenance=RainfallProvenance(kind="constant"))
        wet, event_summary = complete_positive_windowed_event(entry, ctx, w.column, w.field, rainfall,
            w.vegetation_cover_fraction, w.sediment, max_end_s=edges[-1]+event.recession_guard_s,
            control=case.control.sediment_control, policy=case.control.completion_policy,
            report_every_s=case.control.report_every_s)
        assert type(wet) is DryHandoff and wet.status == "complete"
        held, state, t = wet.dry_state, wet.dry_state.bed, wet.dry_state.t_s
        soil = held.storm.soil_water_m
        yield SimpleNamespace(kind="rain", state=state, time_s=t, pool=pool, soil=soil, routing=held,
                              wet=wet, entry_commit=boundary, entry_summary=entry_summary,
                              event_summary=event_summary)
