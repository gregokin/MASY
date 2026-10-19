"""Dense production wind configuration foundation (MASY-10, configuration only).

A standalone factory that turns an exact native MAPLE `ModelConfig` into the actual native
`SchedulerPhysicsConfig` through the one native mapper, `build_scheduler_physics_config`, plus a
transparent canonical audit of every caller-controlling input. Nothing here transports, allocates a bed,
advances a schedule or chooses a calibration, and the result is NOT qualified for transport.

The caller-supplied `WindForcingRecordSpec` is validated solely as the native forcing-envelope input (it
sets the scheduler's supported shear bound); it is not an execution schedule and defines no clock or water
interval. The calibration declaration is caller-supplied provenance, never scientific certification.
Environment and source locks remain the caller runtime lock's responsibility. numpy, MAPLE and the dense
runner are imported only inside calls.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

__all__ = [
    "DenseProductionWindProfile",
    "ProductionProfileError",
    "audit_dense_production_wind_config",
    "build_dense_production_wind_config",
]

FACTORY_VERSION = "masy-dense-production-wind-config-v1"
AUDIT_SCHEMA = "masy-dense-production-wind-config-audit-v1"
QUALIFICATION = "configuration_only_unqualified_for_transport"
ENVELOPE_ROLE = "forcing_bound_only_not_execution_schedule"
AVAILABILITY_POLICY = "explicit_caller_host_float64_ny_nx_nclasses_finite_unit_interval_identity_preserved"
DECLARATION_MEANING = "caller_supplied_provenance_not_scientific_certification"
REVIEWED_USER_CALIBRATION = "reviewed_user_calibration"
SYNTHETIC_TEST_FIXTURE = "synthetic_test_fixture"
CALIBRATION_STATUSES = (REVIEWED_USER_CALIBRATION, SYNTHETIC_TEST_FIXTURE)
_SYNTHETIC_SCHEMA = "synthetic-test-fixture-v1"
_SYNTHETIC_CALIBRATION_ID = "synthetic-scheduler-fixture-sand100um-NOT-MEASURED"
_NATIVE_BUILDER = "maple.experiment.scheduler_adapter.build_scheduler_physics_config"

# The only modes this foundation accepts, and the switches it refuses unless exactly False. Everything else
# (inert prescribed/reduced specs, default transport accuracy, creep, topographic wind, flow separation,
# avalanching, ...) is left to native defaults and native validation.
_REQUIRED = (
    ("aeolian_process_mode", "mode", "detachment_transport_deposition"),
    ("detachment_source", "mode", "reduced"),
    ("execution", "backend", "numpy"),
    ("reduced_detachment", "out_of_envelope_policy", "reject"),
)
_REFUSED_UNLESS_FALSE = (
    ("water_coupling", "enabled"),
    ("vegetation_wake", "enabled"),
    ("prescribed_external_shear", "enabled"),
    ("topographic_commit", "dry_aeolian_commit_separation_enabled"),
)


class ProductionProfileError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclasses.dataclass(frozen=True)
class DenseProductionWindProfile:
    """An exact native `ModelConfig` plus the caller's calibration declaration (provenance only)."""

    model_config: Any
    calibration_status: str
    calibration_reference: str


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _native() -> SimpleNamespace:
    """The actual native objects, imported only when a call needs them."""
    import numpy as np
    from maple.aeolian.scheduler.state import SchedulerPhysicsConfig
    from maple.core.parameters import ModelConfig, config_to_dict, validate_config
    from maple.core.parameters.wind_forcing import (
        WindBlock,
        WindForcingRecordSpec,
        validate_wind_forcing_record,
    )
    from maple.experiment.scheduler_adapter import build_scheduler_physics_config

    return SimpleNamespace(
        np=np, SchedulerPhysicsConfig=SchedulerPhysicsConfig, ModelConfig=ModelConfig,
        config_to_dict=config_to_dict, validate_config=validate_config, WindBlock=WindBlock,
        WindForcingRecordSpec=WindForcingRecordSpec, validate_wind_forcing_record=validate_wind_forcing_record,
        build_scheduler_physics_config=build_scheduler_physics_config,
    )


def _check_profile(native: SimpleNamespace, profile: Any) -> Any:
    if type(profile) is not DenseProductionWindProfile:
        raise ProductionProfileError("wrong_native_type",
                                     f"profile is {_type_name(profile)}; expected DenseProductionWindProfile")
    config = profile.model_config
    if type(config) is not native.ModelConfig:
        raise ProductionProfileError("wrong_native_type",
                                     f"profile.model_config is {_type_name(config)}; expected exact native ModelConfig")
    status, reference = profile.calibration_status, profile.calibration_reference
    if type(status) is not str or status not in CALIBRATION_STATUSES:
        raise ProductionProfileError("invalid_calibration_declaration",
                                     f"calibration_status={status!r}; expected one of {list(CALIBRATION_STATUSES)}")
    if type(reference) is not str or not reference.strip():
        raise ProductionProfileError("invalid_calibration_declaration",
                                     f"calibration_reference={reference!r}; expected a nonempty string")
    try:
        native.validate_config(config)
    except (ValueError, TypeError) as exc:
        raise ProductionProfileError("invalid_model_config", str(exc)) from exc
    reduced = config.reduced_detachment
    if status == REVIEWED_USER_CALIBRATION and (
            reduced.calibration_schema == _SYNTHETIC_SCHEMA or reduced.calibration_id == _SYNTHETIC_CALIBRATION_ID):
        raise ProductionProfileError(
            "calibration_declaration_refused",
            f"reduced_detachment {reduced.calibration_id!r} (schema {reduced.calibration_schema!r}) is the known "
            f"synthetic test fixture and cannot be declared {REVIEWED_USER_CALIBRATION!r}")
    for group, name, wanted in _REQUIRED:
        value = getattr(getattr(config, group), name)
        if type(value) is not str or value != wanted:
            raise ProductionProfileError("unsupported_mode",
                                         f"{group}.{name}={value!r}; this configuration foundation requires {wanted!r}")
    for group, name in _REFUSED_UNLESS_FALSE:
        value = getattr(getattr(config, group), name)
        if value is not False:
            raise ProductionProfileError(
                "unsupported_mode", f"{group}.{name}={value!r}; refused as unqualified for this configuration foundation")
    return config


def _check_envelope(native: SimpleNamespace, record: Any) -> None:
    if type(record) is not native.WindForcingRecordSpec:
        raise ProductionProfileError("wrong_native_type",
                                     f"wind_envelope_record is {_type_name(record)}; expected exact native WindForcingRecordSpec")
    if type(record.blocks) is not tuple or any(type(block) is not native.WindBlock for block in record.blocks):
        raise ProductionProfileError("wrong_native_type",
                                     "wind_envelope_record.blocks must be a tuple of exact native WindBlock")
    try:
        native.validate_wind_forcing_record(record)
    except (ValueError, TypeError) as exc:
        raise ProductionProfileError("invalid_wind_envelope", str(exc)) from exc


def _check_availability(native: SimpleNamespace, config: Any, fraction: Any) -> None:
    """Inspect only: the caller's array is never normalized, copied or replaced."""
    np = native.np
    if fraction is None:
        raise ProductionProfileError("invalid_availability", "initial_available_fraction must be supplied explicitly")
    if type(fraction) is not np.ndarray:
        raise ProductionProfileError("invalid_availability",
                                     f"initial_available_fraction is {_type_name(fraction)}; only host numpy.ndarray")
    if fraction.dtype != np.dtype(np.float64):
        raise ProductionProfileError("invalid_availability",
                                     f"initial_available_fraction dtype is {fraction.dtype.str}; only native float64")
    expected = (config.geometry.ny, config.geometry.nx, len(config.grain_classes.classes))
    if fraction.shape != expected:
        raise ProductionProfileError("invalid_availability",
                                     f"initial_available_fraction shape is {fraction.shape}; geometry/classes require {expected}")
    if not bool(np.isfinite(fraction).all()):
        raise ProductionProfileError("invalid_availability", "initial_available_fraction has non-finite entries")
    if not bool(((fraction >= 0.0) & (fraction <= 1.0)).all()):
        raise ProductionProfileError("invalid_availability", "initial_available_fraction has entries outside [0, 1]")


def build_dense_production_wind_config(profile: DenseProductionWindProfile, *, wind_envelope_record: Any,
                                       initial_available_fraction: Any) -> Any:
    """The actual native `SchedulerPhysicsConfig`, mapped only by the native builder. Configuration only;
    unqualified for transport."""
    native = _native()
    config = _check_profile(native, profile)
    _check_envelope(native, wind_envelope_record)
    _check_availability(native, config, initial_available_fraction)
    try:
        scheduler = native.build_scheduler_physics_config(
            config, spatial_block_diagnostics_enabled=False,
            initial_available_fraction=initial_available_fraction,
            prescribed_external_shear=None, wind_record=wind_envelope_record)
    except (ValueError, TypeError) as exc:
        raise ProductionProfileError("native_builder_refused", str(exc)) from exc
    if type(scheduler) is not native.SchedulerPhysicsConfig:
        raise ProductionProfileError("wrong_native_type", f"native builder returned {_type_name(scheduler)}")
    if scheduler.initial_available_fraction is not initial_available_fraction:
        raise ProductionProfileError("availability_identity_lost",
                                     "native builder did not carry the caller's initial_available_fraction object")
    if scheduler.supported_external_shear_velocity_m_s is None:
        raise ProductionProfileError("unbounded_wind_envelope",
                                     "native builder could not derive a supported shear bound from wind_envelope_record")
    return scheduler


def _canonical(np: Any, value: Any, where: str) -> Any:
    """The existing dense-runner canonical form; its refusals are translated, never bypassed."""
    from masy.dense_runner import DenseRunnerError, _canonical_input

    try:
        return _canonical_input(np, value, where)
    except DenseRunnerError as exc:
        raise ProductionProfileError("uncanonical_input", getattr(exc, "detail", str(exc))) from exc


def audit_dense_production_wind_config(profile: DenseProductionWindProfile, config: Any, *,
                                       wind_envelope_record: Any, initial_available_fraction: Any) -> dict[str, Any]:
    """Rebuild through the native builder and refuse unless `config` is canonically identical in every field
    and carries the caller's availability object, then return a transparent record. `audit_sha256` is
    `masy.dense_bed._sha256` over the record without that key."""
    native = _native()
    if type(config) is not native.SchedulerPhysicsConfig:
        raise ProductionProfileError("wrong_native_type",
                                     f"config is {_type_name(config)}; expected exact native SchedulerPhysicsConfig")
    expected = build_dense_production_wind_config(profile, wind_envelope_record=wind_envelope_record,
                                                  initial_available_fraction=initial_available_fraction)
    np = native.np
    scheduler = _canonical(np, expected, "expected_scheduler_config")
    supplied = _canonical(np, config, "scheduler_config")
    if supplied != scheduler:
        differing = sorted(name for name, item in scheduler["fields"].items() if supplied["fields"].get(name) != item)
        raise ProductionProfileError("scheduler_config_mismatch",
                                     f"supplied scheduler config differs from the native rebuild in field(s) {differing}")
    if config.initial_available_fraction is not initial_available_fraction:
        raise ProductionProfileError("availability_identity_lost",
                                     "supplied scheduler config does not carry the caller's initial_available_fraction object")

    from masy.dense_bed import _sha256

    model = profile.model_config
    reduced = model.reduced_detachment
    model_record = _canonical(np, native.config_to_dict(model), "model_config")
    availability = _canonical(np, initial_available_fraction, "initial_available_fraction")
    envelope = _canonical(np, wind_envelope_record, "wind_envelope_record")
    record = {
        "schema": AUDIT_SCHEMA,
        "qualification": QUALIFICATION,
        "transport_ready": False,
        "envelope_role": ENVELOPE_ROLE,
        "environment_and_source_lock": "not_claimed_caller_runtime_lock_responsibility",
        "factory_policy": {
            "factory_version": FACTORY_VERSION,
            "native_builder": _NATIVE_BUILDER,
            "builder_arguments": {"spatial_block_diagnostics_enabled": False, "prescribed_external_shear": None,
                                  "initial_available_fraction": "caller_supplied_identity_preserved",
                                  "wind_record": "caller_supplied_envelope_only"},
            "required": [[f"{group}.{name}", wanted] for group, name, wanted in _REQUIRED],
            "refused_unless_false": [f"{group}.{name}" for group, name in _REFUSED_UNLESS_FALSE],
            "unbounded_envelope": "refused",
            "reviewed_refused_for_synthetic": {"calibration_schema": _SYNTHETIC_SCHEMA,
                                               "calibration_id": _SYNTHETIC_CALIBRATION_ID},
        },
        "calibration_declaration": {
            "status": profile.calibration_status,
            "reference": profile.calibration_reference,
            "reduced_calibration_id": reduced.calibration_id,
            "reduced_calibration_schema": reduced.calibration_schema,
            "meaning": DECLARATION_MEANING,
        },
        "model_config": {"config_to_dict": model_record, "sha256": _sha256(model_record)},
        "availability": {"policy": AVAILABILITY_POLICY, "array": availability, "sha256": _sha256(availability)},
        "wind_envelope": {
            "record": envelope,
            "supported_external_shear_velocity_m_s": _canonical(
                np, expected.supported_external_shear_velocity_m_s, "supported_external_shear_velocity_m_s"),
            "sha256": _sha256(envelope),
        },
        "scheduler_config": {"canonical": scheduler, "sha256": _sha256(scheduler)},
    }
    digest = _sha256(record)
    return {**record, "audit_sha256": digest}
