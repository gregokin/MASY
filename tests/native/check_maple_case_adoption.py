"""Future locked-pair Native adoption checks; no wind/water transport here.

Synthetic declarations exercise unequal layers/classes/availability. They are
not collaborator defaults or calibration. Root alone executes these checks.
"""
from __future__ import annotations

import dataclasses
import copy
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from masy.case_workflow import (
    AdoptedCase, CaseInputLimits, CaseInspection, CaseWorkflowError, adopt_case_inputs, resolve_case_inputs,
)
from masy.dense_bed import DenseBedError, DenseBedLimits, SHARED_FIELDS


IO = CaseInputLimits(1 << 20, 128, 8 << 20, 16 << 20, 8 << 20, 64 << 20)
BED = DenseBedLimits(1 << 30, 32, 64 << 20)


def _package(root: Path, *, layered: bool = True, seeded: bool = False, imported: bool = False) -> Path:
    root.mkdir()
    case = {
        "run_name": "synthetic_compiled_adoption", "seed": 9281, "output_dir": "outputs/adoption",
        "geometry": {"nx": 3, "ny": 2, "dx_m": 1.0, "dy_m": 1.0, "boundary_x": "periodic",
                     "boundary_y": "periodic", "voxel_dz_m": 0.1, "bulk_density_kg_m3": 1250.0,
                     "active_layer_thickness_m": 0.04},
        "grain_classes": [{"class_id": name, "diameter_m": diameter, "particle_density_kg_m3": 2650.0,
                           "is_aggregate": False} for name, diameter in (("fine", 0.00025), ("coarse", 0.0005))],
        "topography": {"base": "flat", "nz": 8, "extra_top_layers": 2,
                       "perturbation": {"relief_m": 0.01, "correlation_length_m": 1.0}},
        "vegetation": {"source": "objects", "plants": [{"x_m": 0.5, "y_m": 0.5, "height_m": 0.2,
                       "semi_major_m": 0.2, "semi_minor_m": 0.15, "orientation_deg": 0.0, "porosity": 0.2}]},
        "sediment_availability": {"global_available_fraction": 0.7,
            "builder": {"builder": "step", "step": {"low_value": 0.2, "high_value": 0.8,
                "position_fraction": 0.5, "orientation_deg_from_x": 0.0}}},
        "threshold_modifiers": {"physical_crust_multiplier": 1.2},
    }
    if layered:
        case["stratigraphy"] = {"layers": [{"fractions": {"fine": 0.2, "coarse": 0.8}},
            {"thickness_m": 0.07, "fractions": {"fine": 0.9, "coarse": 0.1}}]}
    else:
        case["grain_mixture"] = {"fractions": {"fine": 0.4, "coarse": 0.6}}
    if seeded:
        case["initial_mobile_seed"] = {"mass_by_class_kg": {"fine": 0.0001, "coarse": 0.0002},
                                       "launch_direction_deg": 90.0}
    if imported:
        (root / "source").mkdir()
        np.save(root / "source" / "available.npy", np.array([[0.2, 0.5, 0.8], [0.3, 0.6, 0.9]], dtype=np.float64))
        case["sediment_availability"]["builder"] = {"builder": "imported", "imported": {
            "source": {"path": "source/available.npy", "format": "npy", "row_order": "south_to_north",
                       "value_kind": "dimensionless", "value_units": "fraction"}, "transforms": []}}
    model = {"time_and_forcing": {"outer_timestep_s": 1.0},
             "numerics": {"mass_resolution_kg": 1e-10},
             "fluid_threshold": {"modifiers": {"physical_crust_enabled": True}},
             "detachment_source": {"mode": "aerodynamic"},
             "aerodynamic_detachment": {"reference_rate_kg_m2_s": 0.0001},
             "particle_transport": {"max_active_cohorts": 1024, "quantile_bins": 4, "mobile_state_representation": "flat",
                "hop_distance_by_class": {name: {"median_hop_distance_m": 0.1, "geometric_stddev": 1.5,
                    "representative_flight_speed_m_s": 3.0} for name in ("fine", "coarse")}}}
    wind = {"wind_forcing": {"blocks": [{"shear_velocity_m_s": 0.4, "direction_deg": 90.0, "duration_s": 1.0}]}}
    for name, value in (("case.yaml", case), ("model.yaml", model), ("wind.yaml", wind)):
        (root / name).write_text(json.dumps(value))  # JSON is authored YAML, not a second parser.
    return root


def _same(left, right) -> None:
    """Independent recursive byte comparison, including NaN metadata bits."""
    assert type(left) is type(right)
    if type(left) is np.ndarray:
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes(order="C") == right.tobytes(order="C")
    elif dataclasses.is_dataclass(left):
        assert [field.name for field in dataclasses.fields(left)] == [field.name for field in dataclasses.fields(right)]
        for field in dataclasses.fields(left):
            _same(getattr(left, field.name), getattr(right, field.name))
    elif type(left) is dict:
        assert left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    elif type(left) in (tuple, list):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _same(a, b)
    elif type(left) is float:
        assert struct.pack(">d", left) == struct.pack(">d", right)
    else:
        assert left == right


def _inputs(root: Path, **options):
    return resolve_case_inputs(_package(root, **options), mode="compile", input_limits=IO)


def test_stratified_native_case_is_adopted_losslessly(tmp_path):
    inputs = _inputs(tmp_path / "stratified", imported=True)
    source = inputs.compiled
    before = copy.deepcopy(source)
    adopted = adopt_case_inputs(inputs, bed_limits=BED)
    assert type(adopted) is AdoptedCase and (adopted.bed.context.geometry.ny, adopted.bed.context.geometry.nx) == (2, 3)
    assert len(adopted.bed.context.grain_classes.classes) == 2
    assert adopted.bed.context.geometry is adopted.inputs.resolved.config.geometry
    assert adopted.bed.context.grain_classes is adopted.inputs.resolved.config.grain_classes
    assert adopted.bed.context.topographic_commit_spec is adopted.inputs.resolved.config.topographic_commit
    names = {"ledger": "sediment_ledger"}
    for name in SHARED_FIELDS:
        _same(getattr(adopted.bed.state, name), getattr(source, names.get(name, name)))
    _same(adopted.inputs.compiled, source)
    _same(adopted.inputs.resolved, inputs.resolved)
    for name in ("expected_total_mass_by_class_kg", "expected_mass_by_cell_class_kg", "expected_mass_by_level_class_kg"):
        _same(getattr(adopted.inputs.compiled, name), getattr(source, name))
    assert source.stratigraphy is not None and np.ptp(adopted.bed.context.initial_available_fraction) > 0
    assert np.any(adopted.bed.vegetation.raster.cover_fraction > 0)
    assert adopted.inputs.compiled.provenance_record is not source.provenance_record
    assert not np.shares_memory(adopted.bed.state.voxel_column.mass_kg, source.voxel_column.mass_kg)
    assert not np.shares_memory(adopted.bed.state.voxel_column.mass_kg, adopted.inputs.compiled.voxel_column.mass_kg)
    assert not adopted.inputs.compiled.voxel_column.mass_kg.flags.writeable
    _same(before, source)
    inputs.verify()
    adopted.bed.verify_case_inputs()


def test_uniform_case_keeps_its_native_inventory_and_new_schema(tmp_path):
    inputs = _inputs(tmp_path / "uniform", layered=False)
    adopted = adopt_case_inputs(inputs, bed_limits=BED)
    assert inputs.compiled.stratigraphy is None
    _same(adopted.bed.state.voxel_column, inputs.compiled.voxel_column)
    _same(adopted.bed.state.active_layer, inputs.compiled.active_layer)
    assert adopted.bed.case_record["schema"] == "masy.maple_case_initial.v1"
    assert adopted.bed.initialization_receipt["native_initialization_repeated"] is False
    # Old builder/default/schema continuity is an independent exact Source
    # inverse; a different initial-case schema is deliberately not relabelled.


def test_seeded_case_preserves_mobile_and_post_seed_holdings(tmp_path):
    from maple.case_tools.validators.compiled_case import validate_seeded_case_state
    inputs = _inputs(tmp_path / "seeded", seeded=True)
    adopted = adopt_case_inputs(inputs, bed_limits=BED)
    c, cfg = adopted.inputs.compiled, adopted.inputs.resolved.config
    assert adopted.initial_mobile_state is not None and np.any(adopted.initial_mobile_state.active_mask)
    assert adopted.initial_mobile_state is adopted.inputs.compiled.initial_mobile_state
    assert not adopted.initial_mobile_state.active_mask.flags.writeable
    _same(adopted.initial_mobile_state, inputs.compiled.initial_mobile_state)
    _same(adopted.bed.state.sediment_availability, inputs.compiled.sediment_availability)
    validate_seeded_case_state(adopted.bed.state.voxel_column, adopted.bed.state.active_layer,
        adopted.bed.state.committed_topography, adopted.bed.state.sediment_availability, adopted.initial_mobile_state,
        c.expected_total_mass_by_class_kg, cfg.geometry, cfg.grain_classes, cfg.numerics.mass_resolution_kg)
    _same(c.expected_mass_by_level_class_kg, inputs.compiled.expected_mass_by_level_class_kg)


def test_complete_case_mutation_is_seen_by_ordinary_bed_stage(tmp_path):
    inputs = _inputs(tmp_path / "metadata")
    adopted = adopt_case_inputs(inputs, bed_limits=BED)
    adopted.inputs.compiled.provenance_record["deliberate_mutation"] = True
    with pytest.raises((CaseWorkflowError, DenseBedError), match="mutat"):
        adopted.bed.stage_copy()
    assert "deliberate_mutation" not in inputs.compiled.provenance_record


def test_authored_and_imported_source_changes_refuse(tmp_path):
    inputs = _inputs(tmp_path / "source", imported=True)
    original = inputs.compiled.voxel_column.mass_kg.copy()
    path = inputs.case_dir / "source" / "available.npy"
    raw = path.read_bytes()
    path.write_bytes(raw + b"changed")
    with pytest.raises(CaseWorkflowError, match="changed"):
        adopt_case_inputs(inputs, bed_limits=BED)
    assert np.array_equal(inputs.compiled.voxel_column.mass_kg, original)


def test_named_copy_and_voxel_limits_refuse_before_adoption(tmp_path):
    inputs = _inputs(tmp_path / "limits")
    for limits, code in ((dataclasses.replace(BED, host_memory_cap_bytes=1), "host_memory_cap_exceeded"),
                         (dataclasses.replace(BED, stage_copy_max_bytes=1), "stage_copy_cap_exceeded"),
                         (dataclasses.replace(BED, max_voxels_per_column=1), "voxel_capacity_exceeded")):
        with pytest.raises(CaseWorkflowError) as error:
            adopt_case_inputs(inputs, bed_limits=limits)
        assert error.value.code == code
    inputs.verify()


def test_dtype_geometry_and_class_mutations_refuse(tmp_path):
    inputs = _inputs(tmp_path / "types")
    original = inputs.compiled.voxel_column
    object.__setattr__(inputs.compiled, "voxel_column", dataclasses.replace(original, mass_kg=original.mass_kg.astype(np.float32)))
    with pytest.raises(CaseWorkflowError) as error:
        adopt_case_inputs(inputs, bed_limits=BED)
    assert error.value.code == "unsupported_case_array"
    object.__setattr__(inputs.compiled, "voxel_column", original)
    cfg = inputs.compiled.config
    for changed in (dataclasses.replace(cfg, geometry=dataclasses.replace(cfg.geometry, nx=cfg.geometry.nx + 1)),
                    dataclasses.replace(cfg, grain_classes=dataclasses.replace(cfg.grain_classes, classes=tuple(reversed(cfg.grain_classes.classes))))):
        object.__setattr__(inputs.compiled, "config", changed)
        with pytest.raises(CaseWorkflowError) as error:
            adopt_case_inputs(inputs, bed_limits=BED)
        assert error.value.code == "case_config_mismatch"
    object.__setattr__(inputs.compiled, "config", cfg)
    inputs.verify()


def test_genuine_non_none_reconciliation_baseline_is_refused(tmp_path):
    from maple.core.types.cohort import zeros_mobile_cohort_state
    from maple.core.types.ledger_reconciliation import ledger_reconciliation_baseline_from_committed_state
    adopted = adopt_case_inputs(_inputs(tmp_path / "baseline"), bed_limits=BED)
    bed = adopted.bed
    scheduler = bed.fresh_scheduler_state(zeros_mobile_cohort_state(
        max_cohorts=adopted.inputs.resolved.config.particle_transport.max_active_cohorts), time_s=0.0)
    assert scheduler.physical.water is bed.state.water
    assert scheduler.physical.sediment_availability is bed.state.sediment_availability
    baseline = ledger_reconciliation_baseline_from_committed_state(bed.state.committed_topography)
    changed = dataclasses.replace(scheduler.physical, ledger_reconciliation_baseline=baseline)
    with pytest.raises(DenseBedError) as error:
        bed.adopt_scheduler_physical(changed)
    assert error.value.code == "unsupported_reconciliation_baseline"
    assert bed.time_s == 0.0


def test_validate_mode_has_no_compiled_or_bed_and_limits_are_typed(tmp_path):
    case = _package(tmp_path / "validate")
    inspection = resolve_case_inputs(case, mode="validate", input_limits=IO)
    assert type(inspection) is CaseInspection and not hasattr(inspection, "compiled")
    assert not (case / "processed").exists()
    with pytest.raises(CaseWorkflowError) as error:
        resolve_case_inputs(case, mode="validate", input_limits=None)
    assert error.value.code == "invalid_input_limit"
    with pytest.raises(CaseWorkflowError):
        resolve_case_inputs(case, mode="validate", input_limits=dataclasses.replace(IO, max_authored_file_bytes=1))
