"""Labelled synthetic imported 9x8/K2/V20 Native adoption; no events.

The recipe is the existing case-event interface recipe, not calibration.
Root alone runs these checks in an immutable qualified CPU environment.
"""
from __future__ import annotations

import dataclasses
import json
from types import FunctionType, MethodType

import numpy as np
import pytest
import yaml

from masy import case_workflow as cw
from masy import dense_bed as db
from masy.case_water import CaseWaterCopyLimits, CaseWaterError, CaseWaterSpec, CaseWetClass, _identity
from masy.case_windowed import (
    CaseWindowedError, CaseWindowedInputs, CaseWindowedLimits,
    adopt_windowed_case, prepare_case_windowed_water,
)
from masy.case_workflow import CaseInputLimits, CaseWorkflowError, resolve_case_inputs
from masy.dense_runner import DenseRunnerControl

NY, NX, NV, K = 9, 8, 20, 2
IO = CaseInputLimits(1 << 20, 256, 8 << 20, 16 << 20, 16 << 20, 128 << 20)
LIMITS = CaseWindowedLimits(NY * NX, K, NV, 128 << 20, 4 << 20, 16 << 20)
WATER = CaseWaterCopyLimits(NY * NX, K, 128 << 20, 4 << 20)

def _package(root, *, growth=False, seeded=False):
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
    if growth:
        case['geometry']['voxel_growth'] = {'enabled': True, 'chunk_levels': 4, 'max_levels': 32}
    if seeded:
        case['initial_mobile_seed'] = {'mass_by_class_kg': {'fine': .0001, 'coarse': .0002},
            'launch_direction_deg': 90.}
    for name, value in (('case.yaml', case), ('model.yaml', model), ('wind.yaml', wind)):
        (root / name).write_text(yaml.safe_dump(value, sort_keys=False))
    return root


def _water(inputs):
    from maple_syrup.complete_event import CompletionPolicy
    from maple_syrup.sediment_event import SedimentEventControl
    from maple_syrup.storm import StormControl
    g = inputs.context.geometry; ny, nx = g.ny, g.nx
    from maple.core.types.topographic_commit import committed_elevation_m
    surface = committed_elevation_m(inputs.state.committed_topography)
    ring = np.full((ny + 2, nx + 2), 1., dtype=np.float64); ring[1:-1, 1:-1] = surface
    # Synthetic open outlet: fixed 2 cm below Native mass-equivalent zero,
    # rather than 2 cm below the initial bed. This is not a real-case repair
    # or a guarantee that later interior topography cannot form a strict pit.
    ring[0, 1:-1] = -.02
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
    return prepare_case_windowed_water(inputs, spec, control, limits=WATER)


def _inputs(root, **choices):
    return resolve_case_inputs(_package(root, **choices), mode="compile", input_limits=IO)


def _adopt(inputs, store, **changes):
    options = dict(store_dir=store, datum_offset_m=0.0, window_voxels=3,
                   chunk_columns=64, rows_per_band=2, with_prefix_packets=True, limits=LIMITS)
    options.update(changes)
    return adopt_windowed_case(inputs, **options)


def _compact(value, seen=None):
    """Check the actual returned graph, excluding methods on opaque Native handles."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    assert type(value) not in (cw.CaseInputs, cw.AdoptedCase, db.DenseBed, FunctionType, MethodType)
    assert type(value).__name__ != "CompiledCase"
    if isinstance(value, np.ndarray):
        assert value.shape != (NY, NX, NV, K)
        assert value.size != NY * NX * NV * K
        assert not value.flags.writeable
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _compact(getattr(value, field.name), seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _compact(item, seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            _compact(key, seen); _compact(item, seen)


def test_imported_stratified_case_is_lossless_compact_and_prepares_native_water(tmp_path):
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple.surface.voxels.windowed_inventory import column_mass_by_class_kg, gather_full_columns
    from maple_syrup.sediment_bed import BedContext, BedState

    original = _inputs(tmp_path / "case")
    before = cw._input_record(original.resolved, original.compiled, original.file_pins)
    flags = [(array, bool(array.flags.writeable)) for _name, array in
             db._require_host_arrays((original.resolved, original.compiled), "original")]
    adopted = _adopt(original, tmp_path / "deep")
    assert type(adopted) is CaseWindowedInputs
    assert type(adopted.state) is BedState and type(adopted.context) is BedContext
    assert adopted.config.geometry is adopted.context.geometry
    assert adopted.state.voxel_column.logical_shape == (NY, NX, NV, K)
    assert adopted.state.voxel_column.deep_store.header.chunk_columns == 64
    assert tuple((ref.cell_start, ref.n_cells) for ref in
                 adopted.state.voxel_column.deep_store.chunks) == ((0, 64), (64, 8))
    assert adopted.initial_mobile_state is None and adopted.ledger_reconciliation_baseline is None
    flat = original.compiled.voxel_column.mass_kg.reshape(NY * NX, NV, K)
    cursor = 0
    for ref in adopted.state.voxel_column.deep_store.chunks:
        assert ref.cell_start == cursor and 0 < ref.n_cells <= 64
        ids = np.arange(ref.cell_start, ref.cell_start + ref.n_cells, dtype=np.int64)
        actual = gather_full_columns(adopted.state.voxel_column, ids, max_columns=64)
        assert actual.tobytes() == flat[ref.cell_start:ref.cell_start + ref.n_cells].tobytes()
        cursor += ref.n_cells
    assert cursor == NY * NX
    for name, expected in (("active_layer", original.compiled.active_layer), ("water", original.compiled.water),
                           ("ledger", original.compiled.sediment_ledger),
                           ("sediment_availability", original.compiled.sediment_availability)):
        assert cw._canonical(getattr(adopted.state, name)) == cw._canonical(expected)
    for field in dataclasses.fields(original.compiled.committed_topography):
        if field.name != "committed_voxel_mass_by_class_kg":
            assert cw._canonical(getattr(adopted.state.committed_topography, field.name)) == cw._canonical(
                getattr(original.compiled.committed_topography, field.name))
    np.testing.assert_array_equal(adopted.state.committed_topography.committed_voxel_mass_by_class_kg,
                                  column_mass_by_class_kg(adopted.state.voxel_column))
    assert cw._canonical(adopted.vegetation) == cw._canonical(original.compiled.vegetation)
    assert cw._canonical(adopted.threshold_modifiers) == cw._canonical(original.compiled.threshold_modifier_state)
    physical_fields = {"voxel_column", "active_layer", "water", "sediment_ledger", "committed_topography",
                       "sediment_availability", "vegetation", "threshold_modifier_state", "initial_mobile_state", "config"}
    for group, source, excluded in (("compiled", original.compiled, physical_fields),
                                     ("resolved", original.resolved, {"config"})):
        names = {field.name for field in dataclasses.fields(source)} - excluded
        assert adopted.static_inputs[group].keys() == names
        for name in names:
            assert cw._canonical(adopted.static_inputs[group][name]) == cw._canonical(getattr(source, name))
    assert cw._canonical(adopted.config) == cw._canonical(original.resolved.config)
    _compact(adopted)
    assert not adopted.state.voxel_column.height_packet.packet.flags.writeable
    _compact(adopted.state.voxel_column.height_packet.binding)
    assert not np.shares_memory(adopted.state.voxel_column.height_packet.binding.encoded_base,
                                adopted.state.voxel_column.base)
    assert adopted.verify() is adopted
    record = adopted.identity_record()
    assert record["complete_initial_provenance"] == before
    assert record["native_committed_regrouping"]["exact_height_checked"] is True
    assert record["storage_choices"]["trusted_packet"] is False
    assert record["construction_plan"]["caller_retained_input_root_bytes"] > 0
    plan = record["construction_plan"]
    column = adopted.state.voxel_column
    binding = column.height_packet.binding
    assert column.base.dtype == binding.encoded_base.dtype == np.int32
    assert binding.packet_terms == column.height_packet.packet.shape[-1]
    assert sum(getattr(column, name).nbytes for name in
               ("window_kg", "base", "deep_by_class_kg", "deep_total_kg", "deep_room_kg")) <= plan["candidate_resident_array_bytes"]
    assert column.height_packet.packet.nbytes == plan["candidate_prefix_packet_bytes"]
    assert binding.encoded_base.nbytes == plan["candidate_prefix_encoded_base_bytes"]
    assert adopted.context.initial_available_fraction.nbytes == plan["candidate_available_fraction_bytes"]
    assert record["caller_dense_input_may_remain_owned"] is True
    assert record["event_eligibility"] is False and record["device_placement"] is False
    prepared = _water(adopted)
    assert prepared.verify(adopted) is prepared
    assert prepared.water.vegetation_cover_fraction is adopted.vegetation.raster.cover_fraction
    np.testing.assert_array_equal(prepared.water.terrain.initial_committed_elevation_m,
                                  committed_elevation_m(adopted.state.committed_topography))
    assert prepared.identity_record()["event_eligibility"] is False
    assert cw._input_record(original.resolved, original.compiled, original.file_pins) == before
    assert all(bool(array.flags.writeable) == flag for array, flag in flags)
    for _name, result_array in db._require_host_arrays((adopted.state, adopted.static_inputs, adopted.config,
            adopted.context, adopted.vegetation, adopted.threshold_modifiers), "owned"):
        assert not any(np.shares_memory(result_array, source_array) for source_array, _flag in flags)


def test_explicit_storage_choices_and_known_limits_refuse_before_artifacts(tmp_path):
    original = _inputs(tmp_path / "case")
    before = cw._input_record(original.resolved, original.compiled, original.file_pins)
    for index, (changes, code) in enumerate((
        ({"datum_offset_m": 1.0}, "unsupported_datum"),
        ({"datum_offset_m": -0.0}, "unsupported_datum"),
        ({"with_prefix_packets": False}, "prefix_packets_required"),
        ({"window_voxels": NV + 1}, "shape_limit"),
        ({"chunk_columns": True}, "invalid_storage_choice"),
        ({"rows_per_band": NY + 1}, "shape_limit"),
        ({"limits": dataclasses.replace(LIMITS, max_known_host_bytes=1)}, "known_copy_limit"),
        ({"limits": dataclasses.replace(LIMITS, max_metadata_bytes=1)}, "metadata_limit"),
        ({"limits": dataclasses.replace(LIMITS, max_cells=NY * NX - 1)}, "shape_limit"),
    )):
        store = tmp_path / f"refused{index}"
        with pytest.raises(CaseWindowedError) as caught:
            _adopt(original, store, **changes)
        assert caught.value.code == code
        assert not store.exists()
    with pytest.raises(CaseWindowedError, match="invalid_store_directory"):
        _adopt(original, original.case_dir / "deep")
    assert not (original.case_dir / "deep").exists()
    assert cw._input_record(original.resolved, original.compiled, original.file_pins) == before


def test_authored_growth_mobile_and_non_none_baseline_are_not_discarded(tmp_path):
    from maple.core.types.ledger_reconciliation import ledger_reconciliation_baseline_from_committed_state

    growth = _inputs(tmp_path / "growth", growth=True)
    assert growth.resolved.config.geometry.voxel_growth.enabled is True
    with pytest.raises(ValueError, match="voxel_growth is ENABLED"):
        _adopt(growth, tmp_path / "growth_store")
    assert not (tmp_path / "growth_store").exists()
    seeded = _inputs(tmp_path / "seeded", seeded=True)
    before = cw._input_record(seeded.resolved, seeded.compiled, seeded.file_pins)
    with pytest.raises(CaseWindowedError, match="unsupported_initial_mobile"):
        _adopt(seeded, tmp_path / "seeded_store")
    assert not (tmp_path / "seeded_store").exists()
    assert cw._input_record(seeded.resolved, seeded.compiled, seeded.file_pins) == before
    plain = _inputs(tmp_path / "baseline")
    baseline = ledger_reconciliation_baseline_from_committed_state(plain.compiled.committed_topography)
    object.__setattr__(plain.compiled, "ledger_reconciliation_baseline", baseline)
    # CaseInputs.verify owns this refusal. The adapter's later duplicate
    # check is defensive-only, not a separately exercised initialization path.
    with pytest.raises(CaseWorkflowError, match="unsupported_reconciliation_baseline"):
        _adopt(plain, tmp_path / "baseline_store")
    assert not (tmp_path / "baseline_store").exists()


def test_postconstruction_metadata_refusal_retains_store_and_requires_fresh_path(tmp_path):
    original = _inputs(tmp_path / "case")
    before = cw._input_record(original.resolved, original.compiled, original.file_pins)
    # The source record fits exactly; the full owner record additionally holds
    # the actual store binding/receipt/plan and must exceed this same cap.
    provenance_size = len(json.dumps(before, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii"))
    limits = dataclasses.replace(LIMITS, max_metadata_bytes=provenance_size)
    store = tmp_path / "postconstruction"
    with pytest.raises(CaseWindowedError, match="metadata_limit"):
        _adopt(original, store, limits=limits)
    assert store.is_dir() and any(path.is_file() for path in store.rglob("*"))
    with pytest.raises(CaseWindowedError, match="invalid_store_directory"):
        _adopt(original, store)
    assert cw._input_record(original.resolved, original.compiled, original.file_pins) == before


def test_prepared_identity_cannot_be_rebound_with_matching_mutated_byte_fields(tmp_path):
    adopted = _adopt(_inputs(tmp_path / "case"), tmp_path / "deep")
    prepared = _water(adopted)
    changed_spec = dataclasses.replace(prepared._spec, source_reference=prepared._spec.source_reference + ":changed")
    ksat = prepared.water.column.ksat_m_per_s.copy()
    ksat[0, 0] *= 2.0
    ksat.flags.writeable = False
    changed_water = dataclasses.replace(prepared.water,
        column=dataclasses.replace(prepared.water.column, ksat_m_per_s=ksat))
    changed_control = dataclasses.replace(prepared.control, report_every_s=20.0)
    for name, value, byte_name in (("_spec", changed_spec, "_spec_bytes"),
                                     ("water", changed_water, "_water_bytes"),
                                     ("control", changed_control, "_control_bytes")):
        forged = dataclasses.replace(prepared, **{name: value,
            byte_name: _identity(np, value, WATER.max_metadata_bytes)})
        assert forged.identity_sha256 == prepared.identity_sha256
        with pytest.raises(CaseWindowedError, match="controlling_input_changed"):
            forged.verify()
    for changes in ({"_limits": object()}, {"_water_bytes": "foreign"}):
        with pytest.raises(CaseWindowedError, match="controlling_input_changed"):
            dataclasses.replace(prepared, **changes).verify()
    # Shared wet-control failures deliberately retain Dense API error parity.
    with pytest.raises(CaseWaterError, match="invalid_control"):
        prepare_case_windowed_water(adopted, prepared._spec, object(), limits=WATER)
    assert prepared.verify() is prepared


def test_owned_source_and_prepared_input_currentness_are_enforced(tmp_path):
    original = _inputs(tmp_path / "case")
    adopted = _adopt(original, tmp_path / "deep")
    prepared = _water(adopted)
    detached = adopted.identity_record(); detached["event_eligibility"] = True
    assert adopted.identity_record()["event_eligibility"] is False
    with pytest.raises(CaseWindowedError, match="foreign_owner"):
        prepared.verify(dataclasses.replace(adopted))
    owned = adopted.state.active_layer.mass_kg
    owned.flags.writeable = True
    with pytest.raises(CaseWindowedError, match="mutable_or_nonhost_input"):
        adopted.verify()
    owned.flags.writeable = False
    assert adopted.verify() is adopted
    encoded = adopted.state.voxel_column.height_packet.binding.encoded_base
    before = encoded[0, 0].copy()
    encoded.flags.writeable = True; encoded[0, 0] += 1; encoded.flags.writeable = False
    with pytest.raises(CaseWindowedError, match="controlling_input_changed"):
        adopted.verify()
    encoded.flags.writeable = True; encoded[0, 0] = before; encoded.flags.writeable = False
    assert adopted.verify() is adopted
    returned = prepared.water.column.ksat_m_per_s
    returned.flags.writeable = True; returned[0, 0] *= 2.0
    with pytest.raises(CaseWindowedError, match="controlling_input_changed"):
        prepared.verify()
    source = original.case_dir / "source/dem.npy"
    with source.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(CaseWindowedError, match="source_changed"):
        adopted.verify()
