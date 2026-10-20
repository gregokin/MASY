"""Actual native CONFIGURATION-only GPU-labelled mapper checks.

The admitted CPU worker supplies immutable native tests and the exact e88 CPU
module template. No bed, pool, owner, runner, graph or transport is constructed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

import fixture_dense_production as fixture
import check_dense_wind_profile as old_checks
from maple.core.parameters.execution import ExecutionSpec
import masy.dense_wind_profile as dwp


def _inputs():
    cpu = fixture.model()
    model = dataclasses.replace(cpu, execution=ExecutionSpec(backend="cupy", deterministic_scatter=True))
    profile = dwp.DenseProductionGpuWindProfile(model, "synthetic_test_fixture", "authored native100um interface; NOT MEASURED")
    return cpu, profile, old_checks._envelope(), np.full((4, 4, 1), 0.25, dtype=np.float64)


def _build(profile, envelope, available):
    return dwp.build_dense_production_gpu_wind_config(profile,
        wind_envelope_record=envelope, initial_available_fraction=available)


def _audit(profile, config, envelope, available):
    return dwp.audit_dense_production_gpu_wind_config(profile, config,
        wind_envelope_record=envelope, initial_available_fraction=available)


def _replace(profile, group, **fields):
    model = dataclasses.replace(profile.model_config,
        **{group: dataclasses.replace(getattr(profile.model_config, group), **fields)})
    return dataclasses.replace(profile, model_config=model)


def _baseline():
    path = Path(os.environ["MASY130_CPU_BASELINE_PROFILE"])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "830ebca6694b8da2c897ca50f0e611c644b674f64ec4bde051a9a54488afe7f5"
    # Preserve the original canonical class/module names for the full wake audit.
    # This is a transient test-only source comparison, not runtime provenance.
    name = dwp.__name__
    baseline = types.ModuleType(name)
    baseline.__file__ = str(path)
    previous = sys.modules[name]
    try:
        sys.modules[name] = baseline
        exec(compile(raw, str(path), "exec"), baseline.__dict__)
    finally:
        sys.modules[name] = previous
    return baseline


def _without_expansion(monkeypatch):
    import maple.core.parameters as parameters
    import maple.experiment.scheduler_adapter as adapter

    def forbidden(*args, **kwargs):
        raise AssertionError("native validation/mapper/canonical expansion reached before admission")

    monkeypatch.setattr(parameters, "validate_config", forbidden)
    monkeypatch.setattr(adapter, "build_scheduler_physics_config", forbidden)
    monkeypatch.setattr(dwp, "_canonical", forbidden)


def test_original_native_model_and_actual_mapper_are_preserved():
    from maple.core.parameters import config_to_dict
    from maple.aeolian.scheduler.state import SchedulerPhysicsConfig
    from maple.experiment.scheduler_adapter import build_scheduler_physics_config

    cpu, profile, envelope, available = _inputs()
    before = old_checks._canon(profile.model_config)
    left, right = config_to_dict(cpu), config_to_dict(profile.model_config)
    assert left.pop("execution") == {"backend": "numpy", "device_id": 0, "deterministic_scatter": True}
    assert right.pop("execution") == {"backend": "cupy", "device_id": 0, "deterministic_scatter": True}
    assert left == right
    actual = _build(profile, envelope, available)
    expected = build_scheduler_physics_config(profile.model_config,
        initial_available_fraction=available, wind_record=envelope)
    assert type(actual) is SchedulerPhysicsConfig
    assert actual.execution_backend == "cupy"
    assert actual.initial_available_fraction is available
    assert actual.geometry is profile.model_config.geometry
    assert actual.reduced_detachment is profile.model_config.reduced_detachment
    assert actual.supported_external_shear_velocity_m_s is not None
    assert old_checks._canon(actual) == old_checks._canon(expected)
    assert old_checks._canon(profile.model_config) == before


def test_full_audit_records_and_whole_identity():
    from masy.dense_bed import _sha256

    _, profile, envelope, available = _inputs()
    config = _build(profile, envelope, available)
    audit = _audit(profile, config, envelope, available)
    assert audit["schema"] == dwp.GPU_AUDIT_SCHEMA
    assert audit["transport_ready"] is False
    assert audit["environment_and_source_lock"] == "not_claimed_caller_runtime_lock_responsibility"
    assert set(("model_config", "availability", "wind_envelope", "scheduler_config", "calibration_declaration")) <= set(audit)
    assert audit["scheduler_config"]["canonical"] == old_checks._canon(config)
    assert audit["availability"]["array"] == old_checks._canon(available)
    assert audit["wind_envelope"]["record"] == old_checks._canon(envelope)
    assert audit["factory_policy"]["factory_version"] == dwp.GPU_FACTORY_VERSION
    assert ["execution.backend", "cupy"] in audit["factory_policy"]["required"]
    assert len(audit["factory_policy"]["native_holder_layouts"]) == 68
    assert audit["factory_policy"]["raw_admission"]["max_hop_bins"] == 8
    assert audit["audit_sha256"] == _sha256({key: value for key, value in audit.items() if key != "audit_sha256"})


@pytest.mark.parametrize("synthetic", [False, True])
def test_cpu_complete_audit_bytes_equal_original_source(synthetic):
    baseline = _baseline()
    model = fixture.model() if synthetic else old_checks._config()
    status = "synthetic_test_fixture" if synthetic else "reviewed_user_calibration"
    profile = dwp.DenseProductionWindProfile(model, status, "unchanged CPU declaration")
    old_profile = baseline.DenseProductionWindProfile(model, status, "unchanged CPU declaration")
    envelope, available = old_checks._envelope(), old_checks._availability()
    kwargs = dict(wind_envelope_record=envelope, initial_available_fraction=available)
    actual = dwp.build_dense_production_wind_config(profile, **kwargs)
    expected = baseline.build_dense_production_wind_config(old_profile, **kwargs)
    assert old_checks._canon(actual) == old_checks._canon(expected)
    assert dwp.audit_dense_production_wind_config(profile, actual, **kwargs) == baseline.audit_dense_production_wind_config(old_profile, expected, **kwargs)


@pytest.mark.parametrize("backend", ["cupy", "auto", "bad"])
def test_cpu_refusal_code_and_detail_equal_original_source(backend):
    baseline = _baseline()
    model = old_checks._config()
    model = dataclasses.replace(model, execution=dataclasses.replace(model.execution, backend=backend))
    kwargs = dict(wind_envelope_record=old_checks._envelope(), initial_available_fraction=old_checks._availability())
    errors = []
    for owner in (dwp, baseline):
        with pytest.raises(owner.ProductionProfileError) as info:
            owner.build_dense_production_wind_config(owner.DenseProductionWindProfile(model, "reviewed_user_calibration", "same"), **kwargs)
        errors.append((info.value.code, info.value.detail))
    assert errors[0] == errors[1]


def test_old_synthetic_wake_complete_audit_bytes_equal_original_source():
    baseline = _baseline()
    raw = old_checks._synthetic_raw(out_of_envelope_policy="reject")
    raw["vegetation_wake"] = {"enabled": True}
    model = old_checks._config(raw)
    kwargs = dict(wind_envelope_record=old_checks._envelope(), initial_available_fraction=old_checks._availability())
    rows = []
    for owner in (dwp, baseline):
        profile = owner.DenseSyntheticVegetationWakeProfile(model, "synthetic_test_fixture", "same synthetic wake fixture")
        config = owner.build_dense_synthetic_vegetation_wake_config(profile, **kwargs)
        rows.append(owner.audit_dense_synthetic_vegetation_wake_config(profile, config, **kwargs))
    assert rows[0] == rows[1]


@pytest.mark.parametrize("group,field,value", [
    ("execution", "backend", "numpy"), ("execution", "deterministic_scatter", False),
    ("topographic_wind", "enabled", False), ("flow_separation", "enabled", False),
    ("surface_creep", "enabled", False), ("avalanching", "enabled", False),
    ("water_coupling", "depth_update_rule", "constant_free_surface"),
    ("water_coupling", "enabled", True), ("vegetation_wake", "enabled", True),
    ("physical_window_admission", "enabled", True),
    ("particle_transport", "mobile_state_representation", "calendar"),
])
def test_unsupported_profile_refuses_without_mutation(group, field, value):
    _, profile, envelope, available = _inputs()
    bad = _replace(profile, group, **{field: value})
    before = old_checks._canon(bad.model_config)
    with pytest.raises(dwp.ProductionProfileError):
        _build(bad, envelope, available)
    assert old_checks._canon(bad.model_config) == before


@pytest.mark.parametrize("field,value", [
    ("coefficient_c", 3.0), ("coefficient_c_range", (3.0, 4.0)),
    ("valid_shear_velocity_range_m_s", (0.3, 0.7)),
    ("composition_relative_tolerance", 0.01), ("out_of_envelope_policy", "warn"),
    ("calibration_id", "a different declaration"),
])
def test_synthetic_calibration_cannot_be_relabelled_or_relaxed(field, value):
    _, profile, envelope, available = _inputs()
    with pytest.raises(dwp.ProductionProfileError, match="calibration_declaration_refused"):
        _build(_replace(profile, "reduced_detachment", **{field: value}), envelope, available)


def test_cross_profile_and_exact_subclass_refusals():
    cpu, profile, envelope, available = _inputs()

    @dataclasses.dataclass(frozen=True)
    class Derived(dwp.DenseProductionGpuWindProfile):
        pass

    wrong = [dwp.DenseProductionWindProfile(cpu, "synthetic_test_fixture", "CPU"),
             Derived(profile.model_config, profile.calibration_status, profile.calibration_reference)]
    for item in wrong:
        with pytest.raises(dwp.ProductionProfileError, match="wrong_native_type"):
            _build(item, envelope, available)
    with pytest.raises(dwp.ProductionProfileError, match="wrong_native_type"):
        dwp.build_dense_production_wind_config(profile, wind_envelope_record=envelope, initial_available_fraction=available)


@pytest.mark.parametrize("group,field,value", [
    ("particle_transport", "quantile_bins", 9), ("particle_transport", "quantile_bins", 1 << 100),
    ("particle_transport", "quantile_bins", True), ("particle_transport", "max_active_cohorts", 4001),
    ("geometry", "nx", 65), ("geometry", "ny", 0),
])
def test_count_caps_precede_validation_mapping_and_canonical(monkeypatch, group, field, value):
    _, profile, envelope, available = _inputs()
    bad = _replace(profile, group, **{field: value})
    _without_expansion(monkeypatch)
    with pytest.raises(dwp.ProductionProfileError):
        _build(bad, envelope, available)


@pytest.mark.parametrize("damage", ["text", "mapping", "extra", "deleted", "holder", "raw_array", "declaration"])
def test_closed_raw_layout_refuses_before_expansion(monkeypatch, damage):
    _, profile, envelope, available = _inputs()
    model = profile.model_config
    if damage == "text":
        profile = dataclasses.replace(profile, model_config=dataclasses.replace(model, run_name="x" * 4097))
    elif damage == "mapping":
        profile = _replace(profile, "particle_transport", hop_distance_by_class={str(i): 0.0 for i in range(33)})
    elif damage == "extra":
        object.__setattr__(model, "unrecorded", 1)
    elif damage == "deleted":
        object.__delattr__(model.execution, "deterministic_scatter")
    elif damage == "holder":
        profile = dataclasses.replace(profile, model_config=dataclasses.replace(model, geometry=model.execution))
    elif damage == "raw_array":
        profile = _replace(profile, "prescribed_external_shear", multiplier_map=np.zeros((1,)))
    else:
        profile = dataclasses.replace(profile, calibration_reference="x" * 4097)
    _without_expansion(monkeypatch)
    with pytest.raises(dwp.ProductionProfileError):
        _build(profile, envelope, available)


@pytest.mark.parametrize("damage", ["dtype", "shape", "strides", "nan", "range", "subclass"])
def test_availability_refuses_before_expansion(monkeypatch, damage):
    _, profile, envelope, available = _inputs()
    if damage == "dtype": available = available.astype(np.float32)
    elif damage == "shape": available = available[:-1]
    elif damage == "strides": available = available[:, ::-1]
    elif damage == "nan": available[0, 0, 0] = np.nan
    elif damage == "subclass": available = available.view(type("Sub", (np.ndarray,), {}))
    else: available[0, 0, 0] = 1.01
    _without_expansion(monkeypatch)
    with pytest.raises(dwp.ProductionProfileError):
        _build(profile, envelope, available)


def test_envelope_and_class_counts_refuse_before_expansion(monkeypatch):
    _, profile, envelope, available = _inputs()
    from maple.core.parameters.wind_forcing import resolve_wind_forcing_record
    long = resolve_wind_forcing_record({"blocks": [{"shear_velocity_m_s": 0.5, "direction_deg": 270.0, "duration_s": 1.0}] * 9})
    grains = profile.model_config.grain_classes
    bad = dataclasses.replace(profile, model_config=dataclasses.replace(profile.model_config,
        grain_classes=dataclasses.replace(grains, classes=grains.classes * 2)))
    _without_expansion(monkeypatch)
    available_two = np.full((4, 4, 2), 0.25, dtype=np.float64)
    for item, record, array in ((profile, long, available), (bad, envelope, available_two)):
        with pytest.raises(dwp.ProductionProfileError, match="gpu_configuration_scope"):
            _build(item, record, array)


def test_readonly_borrow_and_all_flags_preserved():
    _, profile, envelope, available = _inputs()
    available.flags.writeable = False
    before = (available.tobytes(), str(available.flags), available.__array_interface__["data"][0])
    config = _build(profile, envelope, available)
    _audit(profile, config, envelope, available)
    assert config.initial_available_fraction is available
    assert (available.tobytes(), str(available.flags), available.__array_interface__["data"][0]) == before


@pytest.mark.parametrize("damage", ["field", "array", "layout"])
def test_supplied_scheduler_is_bounded_before_native_rebuild(monkeypatch, damage):
    _, profile, envelope, available = _inputs()
    config = _build(profile, envelope, available)
    if damage == "field": object.__setattr__(config, "extra", "x")
    elif damage == "layout": object.__delattr__(config, "surface_creep_spec")
    else: config = dataclasses.replace(config, hop_bins=dataclasses.replace(config.hop_bins, weight=np.zeros((1, 9))))
    _without_expansion(monkeypatch)
    with pytest.raises(dwp.ProductionProfileError): _audit(profile, config, envelope, available)


def test_scheduler_controlling_drift_and_metadata_identity():
    _, profile, envelope, available = _inputs()
    config = _build(profile, envelope, available)
    audit = _audit(profile, config, envelope, available)
    with pytest.raises(dwp.ProductionProfileError, match="scheduler_config_mismatch"):
        _audit(profile, dataclasses.replace(config, outer_timestep_s=2.0), envelope, available)
    changed = dataclasses.replace(profile, calibration_reference="another valid authored reference")
    another = _audit(changed, config, envelope, available)
    assert another["scheduler_config"] == audit["scheduler_config"]
    assert another["audit_sha256"] != audit["audit_sha256"]


def test_inert_native_acceleration_defaults_are_preserved():
    _, profile, envelope, available = _inputs()
    config = _build(profile, envelope, available)
    assert config.stationary_continuation_spec is profile.model_config.stationary_continuation
    assert config.dormancy_spec is profile.model_config.dormancy
    assert config.stationary_continuation_spec.enabled is True
    assert config.dormancy_spec.enabled is True
    assert config.detachment_source.mode == "reduced"
    assert config.aeolian_process_mode.mode == "detachment_transport_deposition"


def test_maximum_admitted_domain_bins_and_envelope_roundtrip():
    from maple.core.parameters.wind_forcing import resolve_wind_forcing_record
    cpu = fixture.model(shape=(8, 8))
    model = dataclasses.replace(cpu, execution=ExecutionSpec(backend="cupy", deterministic_scatter=True))
    profile = dwp.DenseProductionGpuWindProfile(model, "synthetic_test_fixture", "maximal bounded interface")
    envelope = resolve_wind_forcing_record({"blocks": [{"shear_velocity_m_s": 0.5,
        "direction_deg": 270.0, "duration_s": 1.0}] * 8})
    available = np.full((8, 8, 1), 0.25, dtype=np.float64)
    config = _build(profile, envelope, available)
    result = _audit(profile, config, envelope, available)
    assert config.initial_available_fraction is available
    assert config.particle_transport.max_active_cohorts == 4000
    assert config.hop_bins.weight.shape == (1, 8)
    assert result["wind_envelope"]["record"] == old_checks._canon(envelope)


@pytest.mark.parametrize("field,value", [("calibration_status", "reviewed_user_calibration"),
    ("calibration_reference", "  ")])
def test_invalid_declarations_are_not_certified(field, value):
    _, profile, envelope, available = _inputs()
    with pytest.raises(dwp.ProductionProfileError):
        _build(dataclasses.replace(profile, **{field: value}), envelope, available)


@pytest.mark.parametrize("field,value", [("weight", 1.0), ("hop_distance_m", {})])
def test_native_any_array_slots_cannot_be_replaced_by_scalars(monkeypatch, field, value):
    _, profile, envelope, available = _inputs()
    config = _build(profile, envelope, available)
    changed = dataclasses.replace(config, hop_bins=dataclasses.replace(config.hop_bins, **{field: value}))
    _without_expansion(monkeypatch)
    with pytest.raises(dwp.ProductionProfileError):
        _audit(profile, changed, envelope, available)


def test_geometry_cap_isolated_by_consistent_shape_and_error_code(monkeypatch):
    from maple.core.parameters.wind_forcing import resolve_wind_forcing_record
    cpu = fixture.model(shape=(9, 9))
    model = dataclasses.replace(cpu, execution=ExecutionSpec(backend="cupy", deterministic_scatter=True))
    profile = dwp.DenseProductionGpuWindProfile(model, "synthetic_test_fixture", "oversized negative configuration")
    envelope = resolve_wind_forcing_record({"blocks": [{"shear_velocity_m_s": 0.5,
        "direction_deg": 270.0, "duration_s": 1.0}]})
    available = np.full((9, 9, 1), 0.25, dtype=np.float64)
    _without_expansion(monkeypatch)
    # Matching the scope code isolates geometry admission from the independent
    # availability-byte ceiling: deleting the geometry guard must fail this test.
    with pytest.raises(dwp.ProductionProfileError, match="gpu_configuration_scope"):
        _build(profile, envelope, available)
