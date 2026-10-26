"""Synthetic >64-cell/V>16/two-class W-R-W interface qualification.

No values are collaborator calibration/defaults. Root alone runs this file in
an immutable, current-source CPU pair with explicit singleton CPU affinity.
"""
from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
import resource

import numpy as np
import pytest
import yaml

from masy import dense_runner as dr
from masy.case_cli import inspect_case
from masy.case_runner import CaseCPUProcessPolicy, CaseRunner
from masy.case_schedule import CaseWindGroup
from masy.case_water import CaseWaterCopyLimits, CaseWaterSpec, CaseWetClass, prepare_case_water
from masy.case_workflow import CaseInputLimits, adopt_case_inputs, resolve_case_inputs
from masy.dense_bed import DenseBedLimits, SHARED_FIELDS
from masy.dense_runner import DenseRunnerControl, DenseRunnerLimits

NY, NX, NV = 9, 8, 20
RAIN_S, RAIN_MM_H, GUARD_S = 60., 72., 180.  # existing fixture_dense_alternation positive hydrology
IO = CaseInputLimits(1 << 20, 256, 8 << 20, 16 << 20, 16 << 20, 128 << 20)
BED = DenseBedLimits(2 << 30, 32, 128 << 20)
WATER = CaseWaterCopyLimits(NY * NX, 2, 128 << 20, 4 << 20)
LIMITS = DenseRunnerLimits(8, 128, 10000, 1000, 512 << 20, 8 << 20)


def _package(root):
    """Actual MAPLE three-file recipe, genuine NPY DEM and stratigraphy."""
    root.mkdir(); (root / 'source').mkdir()
    rows, cols = np.indices((NY, NX), dtype=np.float64)
    dem = np.ascontiguousarray(0.31 + 0.001 * rows + 0.0005 * cols)
    np.save(root / 'source/dem.npy', dem, allow_pickle=False)
    case = {'run_name': 'synthetic_case_cpu_events', 'seed': 9281, 'output_dir': 'outputs/case_cpu',
        'geometry': {'nx': NX, 'ny': NY, 'dx_m': .5, 'dy_m': .5, 'boundary_x': 'periodic', 'boundary_y': 'periodic',
                     'voxel_dz_m': .05, 'bulk_density_kg_m3': 1250., 'active_layer_thickness_m': .02},
        'grain_classes': [{'class_id': key, 'diameter_m': size, 'particle_density_kg_m3': 2650., 'is_aggregate': False}
                          for key, size in (('fine', .00025), ('coarse', .0005))],
        'topography': {'base': 'imported', 'nz': NV, 'perturbation': {'relief_m': 0., 'correlation_length_m': 1.}},
        'import': {'elevation': {'source': {'path': 'source/dem.npy', 'format': 'npy', 'row_order': 'south_to_north',
                                           'value_kind': 'length', 'value_units': 'm'}, 'transforms': []}},
        'stratigraphy': {'layers': [{'fractions': {'fine': .2, 'coarse': .8}},
                                    {'thickness_m': .07, 'fractions': {'fine': .9, 'coarse': .1}}]},
        'vegetation': {'source': 'raster', 'raster': {'height_m': .1, 'cover_fraction': .1, 'porosity': .2,
                                                       'drag_coefficient': 1., 'frontal_area_density': .1}},
        'sediment_availability': {'global_available_fraction': .7},
        'requested_diagnostics': {'spatial_snapshots': True}}
    model = {'time_and_forcing': {'outer_timestep_s': .25}, 'numerics': {'mass_resolution_kg': 1e-10},
        'detachment_source': {'mode': 'aerodynamic'}, 'aerodynamic_detachment': {'reference_rate_kg_m2_s': .0001},
        'avalanching': {'enabled': True},
        'topographic_commit': {'dry_aeolian_commit_separation_enabled': False},
        'particle_transport': {'max_active_cohorts': 8192, 'quantile_bins': 4, 'mobile_state_representation': 'flat',
            'hop_distance_by_class': {key: {'median_hop_distance_m': .1, 'geometric_stddev': 1.5,
                                           'representative_flight_speed_m_s': 3.} for key in ('fine', 'coarse')}}}
    wind = {'wind_forcing': {'blocks': [
        {'shear_velocity_m_s': .4, 'direction_deg': 90., 'duration_s': .5},
        {'shear_velocity_m_s': .45, 'direction_deg': 120., 'duration_s': .75},
        {'shear_velocity_m_s': .4, 'direction_deg': 180., 'duration_s': .5}]}}
    for name, value in (('case.yaml', case), ('model.yaml', model), ('wind.yaml', wind)):
        (root / name).write_text(yaml.safe_dump(value, sort_keys=False))
    return root


def _water(bed):
    from maple_syrup.complete_event import CompletionPolicy
    from maple_syrup.sediment_event import SedimentEventControl
    from maple_syrup.storm import StormControl
    g = bed.context.geometry; ny, nx = g.ny, g.nx
    from maple.core.types.topographic_commit import committed_elevation_m
    surface = committed_elevation_m(bed.state.committed_topography)
    ring = np.full((ny + 2, nx + 2), 1., dtype=np.float64); ring[1:-1, 1:-1] = surface
    ring[0, 1:-1] = surface[0] - .02
    exports = np.zeros(ring.shape, dtype=bool); exports[0, 1:-1] = True
    def grid(value): return np.full((ny, nx), value, dtype=np.float64)
    spec = CaseWaterSpec('synthetic-events-v1', 'authored:case-event-interface;not-calibration',
        ring, exports, np.ones((ny, nx), dtype=bool), grid(1.), False, None, False, False,
        'fixed_ksat', grid(1e-5), grid(0.), grid(0.), grid(.4), grid(.3), None, grid(0.), None, grid(1.),
        (CaseWetClass('fine', 1., 1.2, .4, .01, .6), CaseWetClass('coarse', 2., 1.1, .5, .02, .7)),
        1., 'verstraeten_exp', 1.003e-6, 1., .9, 'legacy_literal', 'legacy_none',
        'legacy_literal', 'legacy_sigma_minus_one', 'legacy_mm')
    control = DenseRunnerControl(SedimentEventControl(storm=StormControl(max_steps=10000, implementation='array'),
        transport_scheme='upwind', transport_implementation='array'),
        CompletionPolicy(deposit_mobile_at_flow_end=True), 10.)
    return prepare_case_water(bed, spec, control, limits=WATER)


def _policy():
    # The Root actor explicitly selects and applies these exact limits before
    # importing this module/Native. Tests never raise a hard ceiling themselves.
    return CaseCPUProcessPolicy(4 << 30, 600, 620, 900., 8 << 20, 256 << 20, 1 << 20,
                               16384, 4096, 16 << 20, 128 << 20, 8 << 20)


def _runner(root):
    inputs = resolve_case_inputs(_package(root), mode='compile', input_limits=IO)
    adopted = adopt_case_inputs(inputs, bed_limits=BED); water = _water(adopted.bed)
    schedule = {'schema': 'masy.case_schedule.v1', 'events': [
        {'kind': 'wind', 'first_block': 0, 'stop_block': 2},
        {'kind': 'rain', 'water_profile_id': water.profile_id,
         'intervals': [{'duration_s': RAIN_S, 'intensity_mm_h': RAIN_MM_H}], 'recession_guard_s': GUARD_S},
        {'kind': 'wind', 'first_block': 2, 'stop_block': 3}]}
    return CaseRunner(adopted, water, schedule, limits=LIMITS, process_policy=_policy())


def _same(a, b):
    assert type(a) is type(b)
    if type(a) is np.ndarray:
        assert a.shape == b.shape and a.dtype == b.dtype
        assert a.tobytes(order='C') == b.tobytes(order='C')
    elif dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a): _same(getattr(a, f.name), getattr(b, f.name))
    elif type(a) is float: assert a.hex() == b.hex()
    elif type(a) in (list, tuple):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True): _same(x, y)
    elif type(a) is dict:
        assert a.keys() == b.keys()
        for k in a: _same(a[k], b[k])
    else: assert a == b


def _direct_wind(runner, bed, start, group):
    from maple.aeolian.scheduler.state import initial_scheduler_state
    from maple.aeolian.scheduler.block import advance_wind_block, check_block_conservation
    from maple.aeolian.scheduler.finalization import finalize_wind_event
    from maple.aeolian.scheduler.record import BlockAccountingAccumulator
    from maple.core.parameters.wind_forcing import WindBlock, resolve_wind_forcing_record
    from maple.core.types.cohort import zeros_mobile_cohort_state
    from maple_syrup.sediment_bed import BedState
    cfg = runner._wind.config
    state = initial_scheduler_state(bed.voxel_column, bed.active_layer,
        zeros_mobile_cohort_state(runner._case_effective.particle_transport.max_active_cohorts),
        bed.ledger, bed.committed_topography, runner._bed._threshold_modifier_state,
        time_s=start, vegetation=runner._bed.vegetation, sediment_availability=bed.sediment_availability, water=bed.water)
    aggregate = BlockAccountingAccumulator(2)
    record = resolve_wind_forcing_record({'blocks': [{'shear_velocity_m_s': b.shear_velocity_m_s,
        'direction_deg': b.direction_deg, 'duration_s': b.duration_s} for b in group.blocks]})
    for i, b in enumerate(group.blocks):
        t = float(state.physical.time_s)
        result = advance_wind_block(state, WindBlock(b.shear_velocity_m_s, b.direction_deg, b.duration_s, t, t+b.duration_s), cfg)
        assert check_block_conservation(result.accounting, final_physical=result.state.physical)[0]
        aggregate.add(result.accounting)
        state = dataclasses.replace(result.state, block_index=i+1, block_offset_s=0.)
    assert aggregate.n_blocks == len(group.blocks)
    assert check_block_conservation(aggregate.total, final_physical=state.physical)[0]
    final = finalize_wind_event(state, cfg, record, record_origin_s=start)
    assert final.completed and final.water_handoff_ready
    physical = final.state.physical
    return BedState(physical.voxel_column, physical.active_layer, physical.water, physical.ledger,
                    physical.committed_topography, physical.sediment_availability), float(physical.time_s), final


def test_multiblock_wind_rain_wind_matches_real_native_endpoints(tmp_path):
    from maple_syrup.complete_event import complete_event, DryHandoff
    from maple_syrup.sediment_bed import commit_bed, routing_graph_from_elevation
    from maple_syrup.sediment_event import initial_event_state
    from maple_syrup.rainfall import RainfallSchedule, RainfallProvenance
    from maple.core.types.topographic_commit import committed_elevation_m
    runner = _runner(tmp_path / 'native_case')
    assert NY*NX > 64 and NV > 16 and len(runner._bed.context.grain_classes.classes) == 2
    assert runner._bed.context.avalanching_spec.enabled is True
    initial = copy.deepcopy(runner._bed.state)
    native, t, first_final = _direct_wind(runner, initial, 0., runner._schedule.events[0])
    first = runner.step(); assert first['status'] == 'accepted'
    for name in SHARED_FIELDS: _same(getattr(runner._bed.state, name), getattr(native, name))
    assert first['case_wind']['n_blocks'] == 2
    assert sum(first['classes']['detachment_kg']) > 0
    w, ctx, control = runner._water, runner._bed.context, runner._control
    source = committed_elevation_m(native.committed_topography)
    graph = routing_graph_from_elevation(w.terrain, source, xp=np)
    seed = initial_event_state(graph, w.terrain, native, ctx, runner._soil.copy(), t_s=t, control=control.sediment_control)
    boundary = commit_bed(native, ctx, graph, w.terrain, t, force=True, routing_source_m=seed.routing_source_elevation_m)
    from maple_syrup.storm import initial_state
    from maple_syrup.sediment_transport import transport_network
    from maple_syrup.sediment_physics import physics_grid_from_graph
    if boundary.rerouted:
        network, grid = transport_network(boundary.graph), physics_grid_from_graph(boundary.graph)
    else:
        assert boundary.graph is seed.graph and boundary.routing_source_m is seed.routing_source_elevation_m
        network, grid = seed.network, seed.grid
    # Preserve the actual accepted routing source; a new initial_event_state
    # would infer the new surface even when Native deliberately did not reroute.
    entry = dataclasses.replace(seed, bed=boundary.state, graph=boundary.graph, network=network, grid=grid,
        routing_source_elevation_m=boundary.routing_source_m, phase=None,
        sediment_velocity_m_s=seed.sediment_velocity_m_s.copy(),
        storm=initial_state(boundary.graph, boundary.state.water.depth_m, runner._soil.copy(), t_s=t))
    rain = RainfallSchedule(edges_s=[t, t+RAIN_S], intensity_mm_per_h=[RAIN_MM_H], provenance=RainfallProvenance(kind='constant'))
    wet = complete_event(entry, ctx, w.column, w.field, rain, w.vegetation_cover_fraction, w.sediment,
        max_end_s=t+RAIN_S+GUARD_S, control=control.sediment_control, policy=control.completion_policy,
        report_every_s=control.report_every_s, max_report_rows=LIMITS.max_report_rows)
    assert type(wet) is DryHandoff and wet.status == 'complete'
    second = runner.step(); assert second['status'] == 'accepted'
    for name in SHARED_FIELDS: _same(getattr(runner._bed.state, name), getattr(wet.dry_state.bed, name))
    assert runner._bed.time_s.hex() == float(wet.dry_state.t_s).hex()
    assert sum(second['classes']['actual_pickup_kg']) > 0
    assert second['water']['rain_m3'] > 0
    assert runner._routing is not None and not np.any(runner._soil)
    final_bed, final_time, _ = _direct_wind(runner, copy.deepcopy(wet.dry_state.bed), float(wet.dry_state.t_s), runner._schedule.events[2])
    third = runner.step(); assert third['status'] == 'accepted'
    for name in SHARED_FIELDS: _same(getattr(runner._bed.state, name), getattr(final_bed, name))
    assert runner._bed.time_s.hex() == final_time.hex()
    assert third['case_wind']['origin_s'] == float(wet.dry_state.t_s).hex()
    assert runner.report()['status'] == 'complete' and len(runner.receipts) == 3
    assert first_final.completed


def test_second_block_failure_leaves_original_boundary_unpublished(tmp_path, monkeypatch):
    runner = _runner(tmp_path / 'failed_case')
    before = runner.accepted.as_record(); bed = runner._bed.state
    snapshots = copy.deepcopy(bed)
    pool, soil, routing = runner._pool, runner._soil, runner._routing
    pool_before, soil_before = copy.deepcopy(pool), soil.copy()
    real = dr._advance_wind_block; calls = []; finals = []
    def fail_second(state, block, config):
        calls.append(state.block_index)
        if len(calls) == 2: raise RuntimeError('synthetic failure after one genuine completed block')
        return real(state, block, config)
    def forbidden_finalizer(*args, **kwargs):
        finals.append(True); raise AssertionError('no finalizer after failed intermediate block')
    # Only the declared MASY delegates may be patched; Native physics unchanged.
    monkeypatch.setattr(dr, '_advance_wind_block', fail_second)
    monkeypatch.setattr(dr, '_finalize_wind_event', forbidden_finalizer)
    stopped = runner.step()
    assert stopped['status'] == 'native_error' and stopped['code'] == 'wind_native_failure'
    assert calls == [0, 1] and not finals
    assert runner.accepted.as_record() == before and not runner.receipts
    assert runner._bed.state is bed and runner._pool is pool and runner._soil is soil and runner._routing is routing
    _same(runner._bed.state, snapshots); _same(runner._pool, pool_before); _same(runner._soil, soil_before)
    assert runner._active_evidence is not None


def test_post_native_resource_refusal_keeps_real_active_evidence(tmp_path, monkeypatch):
    runner = _runner(tmp_path / 'post_native_cap')
    before = runner.accepted.as_record(); accepted = runner._bed.state
    real_advance, real_check = dr._advance_wind_block, runner._work_bytes_check
    actual_results = []
    def observed_advance(state, block, config):
        result = real_advance(state, block, config); actual_results.append(result); return result
    def refuse_after_actual_native(estimate):
        if actual_results:
            raise dr._Stop('refused', 'work_bytes_cap', 'synthetic post-Native resource refusal')
        return real_check(estimate)
    monkeypatch.setattr(dr, '_advance_wind_block', observed_advance)
    monkeypatch.setattr(runner, '_work_bytes_check', refuse_after_actual_native)
    stop = runner.step()
    assert stop['code'] == 'work_bytes_cap' and stop['status'] == 'refused'
    assert runner.accepted.as_record() == before and runner._bed.state is accepted
    assert len(actual_results) == 1 and not runner.receipts
    assert stop['unaccepted_elapsed_s'] is not None
    assert runner._active_evidence['last_returned_result'] is actual_results[0]
    assert float(actual_results[0].state.physical.time_s) > 0.
