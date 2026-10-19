"""Configuration-only checks for `masy.dense_wind_profile` (MASY-10 foundation).

No beds, no transport, no simulation: the only array is the explicit availability policy input. The
runner supplies the frozen native MAPLE test modules (`tests.phase22b.*`) and the immutable native package
on the import path. The measured example is the already labelled SINGLE-CLASS reduced fixture, not a
six-class production qualification.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import masy.dense_wind_profile as dwp
from masy.dense_wind_profile import (
    DenseProductionWindProfile,
    ProductionProfileError,
    audit_dense_production_wind_config,
    build_dense_production_wind_config,
)

REVIEWED = "reviewed_user_calibration"
SYNTHETIC = "synthetic_test_fixture"
_REFERENCE = "maple:docs/phase22b2/calibration.md"
# Hop routing values copied from the native scheduler fixture (tests/unit/_scheduler_helpers.py): routing
# configuration the native builder requires, not a calibration and not a physical choice made here.
_HOP = {"median_hop_distance_m": 0.3, "geometric_stddev": 2.0, "representative_flight_speed_m_s": 3.0}
_WIND_TEXT = "wind_forcing: {blocks: [{shear_velocity_m_s: 0.5, direction_deg: 270.0, duration_s: 60.0}]}"


def _config_tests():
    return importlib.import_module("tests.phase22b.test_reduced_detachment_config")


def _synthetic_calibration():
    module = importlib.import_module("tests.phase22b.test_reduced_integration")
    return copy.deepcopy(module.SYNTHETIC_FIXTURE_CALIBRATION)


def _measured_raw():
    upstream = _config_tests()
    raw = copy.deepcopy(upstream._MODEL_RAW)
    raw["detachment_source"] = {"mode": "reduced"}
    raw["reduced_detachment"] = upstream._raw()
    raw["particle_transport"] = {"hop_distance_by_class": {"sand": dict(_HOP)}}
    return raw


def _synthetic_raw(**calibration_over):
    raw = _measured_raw()
    calibration = {**_synthetic_calibration(), **calibration_over}
    raw["grain_classes"][0]["diameter_m"] = calibration["calibrated_grains"][0]["diameter_m"]
    raw["reduced_detachment"] = calibration
    return raw


def _config(raw=None):
    from maple.core.parameters import resolve_config, validate_config

    config = resolve_config(_measured_raw() if raw is None else raw)
    validate_config(config)
    return config


def _profile(config=None, status=REVIEWED, reference=_REFERENCE):
    return DenseProductionWindProfile(_config() if config is None else config, status, reference)


def _envelope(shear=0.5, direction=270.0, source_path="cases/fixture/wind.yaml", source_text=_WIND_TEXT):
    from maple.core.parameters.wind_forcing import (
        compute_wind_forcing_source_sha256,
        resolve_wind_forcing_record,
    )

    record = resolve_wind_forcing_record(
        {"blocks": [{"shear_velocity_m_s": shear, "direction_deg": direction, "duration_s": 60.0}]})
    return dataclasses.replace(record, source_path=source_path,
                               source_sha256=compute_wind_forcing_source_sha256(source_text))


def _availability(value=1.0):
    return np.full((4, 4, 1), value, dtype=np.float64)


def _canon(value):
    from masy.dense_runner import _canonical_input

    return _canonical_input(np, value, "check")


def _build(profile=None, envelope=None, availability=None):
    profile = _profile() if profile is None else profile
    envelope = _envelope() if envelope is None else envelope
    availability = _availability() if availability is None else availability
    scheduler = build_dense_production_wind_config(
        profile, wind_envelope_record=envelope, initial_available_fraction=availability)
    return profile, envelope, availability, scheduler


def _raises(code, fn, *args, **kwargs):
    with pytest.raises(ProductionProfileError) as info:
        fn(*args, **kwargs)
    assert info.value.code == code, info.value
    return info.value


def _forbid_native_build(monkeypatch):
    import maple.experiment.scheduler_adapter as adapter

    def _forbidden(*args, **kwargs):
        raise AssertionError("native builder reached after a refusal")

    monkeypatch.setattr(adapter, "build_scheduler_physics_config", _forbidden)


def test_exact_native_adapter_output():
    from maple.aeolian.scheduler.state import SchedulerPhysicsConfig
    from maple.experiment.scheduler_adapter import build_scheduler_physics_config

    profile, envelope, availability, scheduler = _build()
    expected = build_scheduler_physics_config(
        profile.model_config, initial_available_fraction=availability, wind_record=envelope)
    assert type(scheduler) is SchedulerPhysicsConfig
    assert _canon(scheduler) == _canon(expected)
    assert scheduler.initial_available_fraction is availability
    assert scheduler.geometry is profile.model_config.geometry
    assert scheduler.reduced_detachment is profile.model_config.reduced_detachment
    assert scheduler.supported_external_shear_velocity_m_s is not None
    assert scheduler.supported_external_shear_velocity_m_s == expected.supported_external_shear_velocity_m_s


def test_native_defaults_are_accepted_unchanged():
    from maple.core.parameters.prescribed_detachment import PrescribedDetachmentSpec
    from maple.core.parameters.transport_accuracy import TransportAccuracySpec

    config = _config()
    assert type(config.prescribed_detachment) is PrescribedDetachmentSpec
    assert type(config.transport_accuracy) is TransportAccuracySpec
    _, _, _, scheduler = _build(_profile(config))
    for scheduler_name, config_name in (
        ("prescribed_detachment", "prescribed_detachment"),
        ("transport_accuracy", "transport_accuracy"),
        ("stationary_continuation_spec", "stationary_continuation"),
        ("dormancy_spec", "dormancy"),
        ("physical_window_admission_spec", "physical_window_admission"),
        ("surface_creep_spec", "surface_creep"),
        ("avalanching_spec", "avalanching"),
    ):
        assert getattr(scheduler, scheduler_name) is getattr(config, config_name)


def test_native_topographic_wind_default_is_accepted():
    raw = _measured_raw()
    del raw["topographic_wind"]
    config = _config(raw)
    assert config.topographic_wind.enabled is True
    _, _, _, scheduler = _build(_profile(config))
    assert scheduler.topographic_wind_spec is config.topographic_wind
    assert scheduler.flow_separation_spec is config.flow_separation


_ACTIVE = [
    ("aeolian_process_mode", "mode", "coupled_saltation"),
    ("detachment_source", "mode", "aerodynamic"),
    ("execution", "backend", "cupy"),
    ("reduced_detachment", "out_of_envelope_policy", "warn"),
    ("water_coupling", "enabled", True),
    ("vegetation_wake", "enabled", True),
    ("prescribed_external_shear", "enabled", True),
    ("topographic_commit", "dry_aeolian_commit_separation_enabled", True),
]


@pytest.mark.parametrize("group, field, value", _ACTIVE)
def test_active_modes_are_refused(monkeypatch, group, field, value):
    import maple.core.parameters as params

    config = _config()
    bad = dataclasses.replace(config, **{group: dataclasses.replace(getattr(config, group), **{field: value})})
    before = _canon(bad)
    with pytest.raises(ProductionProfileError):
        build_dense_production_wind_config(
            _profile(bad), wind_envelope_record=_envelope(), initial_available_fraction=_availability())
    assert _canon(bad) == before
    # With the native validator bypassed, this foundation's own refusal still fires before the builder.
    monkeypatch.setattr(params, "validate_config", lambda config: None)
    _forbid_native_build(monkeypatch)
    err = _raises("unsupported_mode", build_dense_production_wind_config, _profile(bad),
                  wind_envelope_record=_envelope(), initial_available_fraction=_availability())
    assert f"{group}.{field}" in err.detail
    assert _canon(bad) == before


def _invalid_configs():
    config = _config()
    grains = config.grain_classes
    return {
        "seed": dataclasses.replace(config, seed=-1),
        "fluid": dataclasses.replace(
            config, fluid_threshold=dataclasses.replace(config.fluid_threshold, air_density_kg_m3=1.0)),
        "grain": dataclasses.replace(config, grain_classes=dataclasses.replace(
            grains, classes=(dataclasses.replace(grains.classes[0], diameter_m=1.0e-4),))),
    }


@pytest.mark.parametrize("name", ["seed", "fluid", "grain"])
def test_native_invalid_config_detail_is_preserved_without_mutation(monkeypatch, name):
    from maple.core.parameters import validate_config

    bad = _invalid_configs()[name]
    with pytest.raises(ValueError) as native:
        validate_config(bad)
    before = _canon(bad)
    _forbid_native_build(monkeypatch)
    err = _raises("invalid_model_config", build_dense_production_wind_config, _profile(bad),
                  wind_envelope_record=_envelope(), initial_available_fraction=_availability())
    assert err.detail == str(native.value)
    assert _canon(bad) == before


def test_native_builder_refusal_detail_is_preserved():
    from maple.experiment.scheduler_adapter import build_scheduler_physics_config

    config = _config()
    bad = dataclasses.replace(
        config, particle_transport=dataclasses.replace(config.particle_transport, hop_distance_by_class={}))
    with pytest.raises(ValueError) as native:
        build_scheduler_physics_config(bad, initial_available_fraction=_availability(), wind_record=_envelope())
    err = _raises("native_builder_refused", _build, _profile(bad))
    assert err.detail == str(native.value)


def test_unbounded_envelope_is_refused(monkeypatch):
    import maple.experiment.scheduler_adapter as adapter

    monkeypatch.setattr(adapter, "_supported_forcing_bound_m_s", lambda wind_record: None)
    _raises("unbounded_wind_envelope", _build)


def test_wrong_native_types_are_refused(monkeypatch):
    from maple.core.parameters import ModelConfig, config_to_dict
    from maple.core.parameters.wind_forcing import WindForcingRecordSpec

    @dataclasses.dataclass(frozen=True)
    class _SubConfig(ModelConfig):
        pass

    @dataclasses.dataclass(frozen=True)
    class _SubRecord(WindForcingRecordSpec):
        pass

    config = _config()
    envelope = _envelope()
    availability = _availability()
    sub_config = _SubConfig(**{f.name: getattr(config, f.name) for f in dataclasses.fields(config)})
    sub_record = _SubRecord(**{f.name: getattr(envelope, f.name) for f in dataclasses.fields(envelope)})
    _forbid_native_build(monkeypatch)
    profiles = [
        SimpleNamespace(model_config=config, calibration_status=REVIEWED, calibration_reference=_REFERENCE),
        _profile(config_to_dict(config)),
        _profile(sub_config),
    ]
    for profile in profiles:
        _raises("wrong_native_type", build_dense_production_wind_config, profile,
                wind_envelope_record=envelope, initial_available_fraction=availability)
    records = [
        {"blocks": [{"shear_velocity_m_s": 0.5, "direction_deg": 270.0, "duration_s": 60.0}]},
        sub_record,
        dataclasses.replace(envelope, blocks=list(envelope.blocks)),
        dataclasses.replace(envelope, blocks=({"shear_velocity_m_s": 0.5},)),
    ]
    for record in records:
        _raises("wrong_native_type", build_dense_production_wind_config, _profile(config),
                wind_envelope_record=record, initial_available_fraction=availability)


class _ArraySubclass(np.ndarray):
    pass


def _with_entry(value):
    array = _availability()
    array[0, 0, 0] = value
    return array


_BAD_AVAILABILITY = {
    "missing": lambda: None,
    "list": lambda: _availability().tolist(),
    "float32": lambda: _availability().astype(np.float32),
    "big_endian": lambda: _availability().astype(">f8"),
    "subclass": lambda: _availability().view(_ArraySubclass),
    "two_d": lambda: np.ones((4, 4), dtype=np.float64),
    "wrong_classes": lambda: np.ones((4, 4, 2), dtype=np.float64),
    "wrong_order": lambda: np.ones((1, 4, 4), dtype=np.float64),
    "nan": lambda: _with_entry(np.nan),
    "inf": lambda: _with_entry(np.inf),
    "negative": lambda: _with_entry(-0.1),
    "above_one": lambda: _with_entry(np.nextafter(1.0, 2.0)),
}


@pytest.mark.parametrize("name", sorted(_BAD_AVAILABILITY))
def test_invalid_availability_is_refused_without_mutation(monkeypatch, name):
    value = _BAD_AVAILABILITY[name]()
    snapshot = copy.deepcopy(value)
    _forbid_native_build(monkeypatch)
    _raises("invalid_availability", build_dense_production_wind_config, _profile(),
            wind_envelope_record=_envelope(), initial_available_fraction=value)
    if isinstance(value, np.ndarray):
        assert value.dtype == snapshot.dtype
        assert np.array_equal(value, snapshot, equal_nan=True)
    else:
        assert value == snapshot


def test_availability_identity_and_values_are_preserved():
    availability = np.linspace(0.0, 1.0, 16, dtype=np.float64).reshape(4, 4, 1)
    snapshot = availability.copy()
    writeable = availability.flags.writeable
    _, _, _, scheduler = _build(availability=availability)
    assert scheduler.initial_available_fraction is availability
    assert scheduler.resolved_initial_available_fraction() is availability
    assert np.array_equal(availability, snapshot)
    assert availability.flags.writeable == writeable


def test_malformed_envelope_keeps_native_detail(monkeypatch):
    from maple.core.parameters.wind_forcing import (
        WindForcingRecordSpec,
        validate_wind_forcing_record,
    )

    envelope = _envelope()
    block = envelope.blocks[0]
    cases = [
        WindForcingRecordSpec(blocks=()),
        dataclasses.replace(envelope, blocks=(dataclasses.replace(block, end_time_s=block.end_time_s + 1.0),)),
        dataclasses.replace(envelope, blocks=(dataclasses.replace(block, shear_velocity_m_s=-0.1),)),
        dataclasses.replace(envelope, blocks=(
            dataclasses.replace(block, duration_s=0.0, end_time_s=block.start_time_s),)),
    ]
    _forbid_native_build(monkeypatch)
    for record in cases:
        with pytest.raises(ValueError) as native:
            validate_wind_forcing_record(record)
        before = _canon(record)
        err = _raises("invalid_wind_envelope", build_dense_production_wind_config, _profile(),
                      wind_envelope_record=record, initial_available_fraction=_availability())
        assert err.detail == str(native.value)
        assert _canon(record) == before


def test_calibration_declaration_honesty():
    _build(_profile(status=REVIEWED))
    # Labelling a measured fixture as synthetic under-claims; it is allowed and recorded as such.
    _build(_profile(status=SYNTHETIC))
    synthetic = _config(_synthetic_raw())
    _raises("calibration_declaration_refused", _build, _profile(synthetic, status=REVIEWED))
    renamed_schema = _config(_synthetic_raw(calibration_schema="maple-detachment-calibration-v5"))
    _raises("calibration_declaration_refused", _build, _profile(renamed_schema, status=REVIEWED))
    raw = _measured_raw()
    raw["reduced_detachment"]["calibration_schema"] = "synthetic-test-fixture-v1"
    _raises("calibration_declaration_refused", _build, _profile(_config(raw), status=REVIEWED))
    # Declared honestly, the synthetic fixture is still refused for its warn envelope policy.
    err = _raises("unsupported_mode", _build, _profile(synthetic, status=SYNTHETIC))
    assert "reduced_detachment.out_of_envelope_policy" in err.detail


def test_honestly_labelled_synthetic_config_keeps_native_calibration_guards():
    raw = _synthetic_raw(out_of_envelope_policy="reject")
    profile, envelope, availability, scheduler = _build(_profile(_config(raw), status=SYNTHETIC))
    record = audit_dense_production_wind_config(
        profile, scheduler, wind_envelope_record=envelope, initial_available_fraction=availability)
    assert record["calibration_declaration"]["status"] == SYNTHETIC
    assert record["calibration_declaration"]["reduced_calibration_id"] == raw["reduced_detachment"]["calibration_id"]
    assert scheduler.reduced_detachment is profile.model_config.reduced_detachment
    assert scheduler.reduced_detachment.out_of_envelope_policy == "reject"
    assert record["transport_ready"] is False


@pytest.mark.parametrize("status, reference", [
    ("reviewed", _REFERENCE), ("", _REFERENCE), (None, _REFERENCE), ("REVIEWED_USER_CALIBRATION", _REFERENCE),
    (REVIEWED, ""), (REVIEWED, "   "), (REVIEWED, None), (REVIEWED, 3),
])
def test_invalid_calibration_declarations_are_refused(monkeypatch, status, reference):
    profile = _profile(status=status, reference=reference)
    _forbid_native_build(monkeypatch)
    _raises("invalid_calibration_declaration", _build, profile)


def test_audit_record_is_transparent_and_recomputable():
    from maple.core.parameters import config_to_dict

    from masy.dense_bed import _sha256

    profile, envelope, availability, scheduler = _build()
    record = audit_dense_production_wind_config(
        profile, scheduler, wind_envelope_record=envelope, initial_available_fraction=availability)
    again = audit_dense_production_wind_config(
        profile, scheduler, wind_envelope_record=envelope, initial_available_fraction=availability)
    assert record == again
    body = {key: value for key, value in record.items() if key != "audit_sha256"}
    assert record["audit_sha256"] == _sha256(body)
    json.dumps(record, allow_nan=False, sort_keys=True)
    assert record["qualification"] == "configuration_only_unqualified_for_transport"
    assert record["envelope_role"] == "forcing_bound_only_not_execution_schedule"
    assert record["transport_ready"] is False
    declaration = record["calibration_declaration"]
    assert declaration["status"] == REVIEWED and declaration["reference"] == _REFERENCE
    assert declaration["meaning"] == "caller_supplied_provenance_not_scientific_certification"
    assert declaration["reduced_calibration_id"] == profile.model_config.reduced_detachment.calibration_id
    assert record["scheduler_config"]["canonical"] == _canon(scheduler)
    assert record["model_config"]["config_to_dict"] == _canon(config_to_dict(profile.model_config))
    assert record["wind_envelope"]["record"] == _canon(envelope)
    assert record["availability"]["array"] == _canon(availability)
    for section, payload in (("scheduler_config", "canonical"), ("model_config", "config_to_dict"),
                             ("wind_envelope", "record"), ("availability", "array")):
        assert record[section]["sha256"] == _sha256(record[section][payload])


def test_audit_digest_detects_every_caller_controlling_change():
    config = _config()
    hop_raw = _measured_raw()
    hop_raw["particle_transport"]["hop_distance_by_class"]["sand"]["median_hop_distance_m"] = 0.31
    variants = {
        "baseline": {},
        "status": {"profile": _profile(config, status=SYNTHETIC)},
        "reference": {"profile": _profile(config, reference=_REFERENCE + "#rev2")},
        "seed": {"profile": _profile(dataclasses.replace(config, seed=8))},
        "run_name": {"profile": _profile(dataclasses.replace(config, run_name="reduced_foundation_b"))},
        "hop": {"profile": _profile(_config(hop_raw))},
        "availability": {"availability": _availability(0.5)},
        "shear": {"envelope": _envelope(shear=0.55)},
        "direction": {"envelope": _envelope(direction=90.0)},
        "source_path": {"envelope": _envelope(source_path="cases/other/wind.yaml")},
        "source_sha256": {"envelope": _envelope(source_text=_WIND_TEXT + " # edited")},
    }
    digests, sections = {}, {}
    for name, over in variants.items():
        over = {"profile": _profile(config), **over}
        profile, envelope, availability, scheduler = _build(**over)
        record = audit_dense_production_wind_config(
            profile, scheduler, wind_envelope_record=envelope, initial_available_fraction=availability)
        digests[name] = record["audit_sha256"]
        sections[name] = {key: record[key]["sha256"]
                          for key in ("model_config", "availability", "wind_envelope", "scheduler_config")}
    assert len(set(digests.values())) == len(digests), digests
    baseline = sections["baseline"]
    for name in ("status", "reference"):
        assert sections[name] == baseline
    for name in ("seed", "run_name"):
        assert sections[name]["model_config"] != baseline["model_config"]
        assert sections[name]["scheduler_config"] == baseline["scheduler_config"]
    for name in ("source_path", "source_sha256", "direction"):
        assert sections[name]["wind_envelope"] != baseline["wind_envelope"]
    for name in ("source_path", "source_sha256"):
        assert sections[name]["scheduler_config"] == baseline["scheduler_config"]
    assert sections["availability"]["availability"] != baseline["availability"]
    assert sections["hop"]["scheduler_config"] != baseline["scheduler_config"]
    # Same content in a fresh object is the same record: the digest is content-deterministic.
    profile, envelope, _, _ = _build(_profile(config))
    fresh = _availability()
    scheduler = build_dense_production_wind_config(profile, wind_envelope_record=envelope,
                                                   initial_available_fraction=fresh)
    record = audit_dense_production_wind_config(profile, scheduler, wind_envelope_record=envelope,
                                                initial_available_fraction=fresh)
    assert record["audit_sha256"] == digests["baseline"]


def test_audit_refuses_scheduler_built_from_other_inputs():
    profile, envelope, availability, scheduler = _build()
    _raises("scheduler_config_mismatch", audit_dense_production_wind_config, profile, scheduler,
            wind_envelope_record=_envelope(shear=0.55), initial_available_fraction=availability)
    hop_raw = _measured_raw()
    hop_raw["particle_transport"]["hop_distance_by_class"]["sand"]["median_hop_distance_m"] = 0.31
    _raises("scheduler_config_mismatch", audit_dense_production_wind_config, _profile(_config(hop_raw)), scheduler,
            wind_envelope_record=envelope, initial_available_fraction=availability)
    _raises("availability_identity_lost", audit_dense_production_wind_config, profile, scheduler,
            wind_envelope_record=envelope, initial_available_fraction=availability.copy())
    _raises("wrong_native_type", audit_dense_production_wind_config, profile, object(),
            wind_envelope_record=envelope, initial_available_fraction=availability)


class _NoTamper(Exception):
    pass


def _tampered(value):
    """A canonically different value of the same shape of thing; test-only, never simulated."""
    if value is None:
        return 1
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is float:
        if not math.isfinite(value):
            raise _NoTamper
        return value * 2.0 + 1.0
    if type(value) is str:
        return value + "-tampered"
    if isinstance(value, enum.Enum):
        others = [member for member in type(value) if member is not value]
        if not others:
            raise _NoTamper
        return others[0]
    if isinstance(value, np.bool_):
        return np.bool_(not value)
    if isinstance(value, np.generic):
        return value.dtype.type(_tampered(value.item()))
    if type(value) is np.ndarray:
        if value.size == 0:
            return np.zeros((1,), dtype=value.dtype)
        changed = np.array(value, copy=True)
        flat = changed.reshape(-1)
        if changed.dtype.kind == "b":
            flat[0] = not flat[0]
        elif changed.dtype.kind in "iu":
            flat[0] = flat[0] + 1
        else:
            flat[0] = 0.25 if flat[0] != 0.25 else 0.5
        return changed
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            try:
                replacement = _tampered(getattr(value, f.name))
            except _NoTamper:
                continue
            clone = copy.copy(value)
            object.__setattr__(clone, f.name, replacement)
            return clone
        raise _NoTamper
    if isinstance(value, tuple):
        return (_tampered(value[0]),) + tuple(value[1:]) if value else (1,)
    if isinstance(value, list):
        return [_tampered(value[0]), *value[1:]] if value else [1]
    if isinstance(value, dict):
        return {**value, "masy_tampered": 1}
    raise _NoTamper


def test_every_scheduler_field_tamper_is_refused():
    profile, envelope, availability, scheduler = _build()
    untouched = copy.copy(scheduler)
    audit_dense_production_wind_config(
        profile, untouched, wind_envelope_record=envelope, initial_available_fraction=availability)
    names = [f.name for f in dataclasses.fields(scheduler)]
    assert "initial_available_fraction" in names and "supported_external_shear_velocity_m_s" in names
    before = _canon(scheduler)
    for name in names:
        tampered = copy.copy(scheduler)
        object.__setattr__(tampered, name, _tampered(getattr(scheduler, name)))
        err = _raises("scheduler_config_mismatch", audit_dense_production_wind_config, profile, tampered,
                      wind_envelope_record=envelope, initial_available_fraction=availability)
        assert repr(name) in err.detail, (name, err.detail)
    assert _canon(scheduler) == before


def test_module_imports_lazily():
    code = "; ".join([
        "import sys",
        "sys.path.insert(0, " + repr(str(Path(dwp.__file__).resolve().parents[1])) + ")",
        "import sys, masy",
        "before = set(sys.modules)",
        "import masy.dense_wind_profile",
        "added = sorted(set(sys.modules) - before)",
        "bad = [m for m in added if m.split('.')[0] in ('numpy', 'maple') or m in ('masy.dense_runner', 'masy.dense_bed')]",
        "assert not bad, bad",
    ])
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], env=dict(os.environ),
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
