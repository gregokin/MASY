"""Bounded Dense NumPy executor of ONE complete native wind occurrence on an actual current bed.

This is a thin, opt-in native wind COMPONENT. It is not a wind-water daily run, it adopts no SYRUP lifecycle state, owns no day
cursor and makes no water update (no drainage, E/T, moisture reset, sink or coupling). Wind assumes dry regardless of soil or
surface water: carried ponded depth is preserved exactly and untouched; water-borne sediment must be genuinely empty.

Every physical operation is an actual MAPLE call: ``build_scheduler_physics_config`` (from the admitted compiled config, the
current availability policy and the occurrence's own forcing envelope), ``advance_wind_block`` once per block in exact row order,
the native record loop's logical cursor mapping, and ``finalize_wind_event(..., record_origin_s=start)``. The native record
entry ``advance_wind_record`` enforces time relative to 0 and cannot start at an arbitrary physical origin, so, exactly like
``dense_runner``, blocks are driven one by one at the actual start; no native clock is rewritten and no timestep is rounded.

Importing this module imports neither NumPy nor any native package. See docs/daily_wind_executor.md.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from masy.clock import ClockError, _native_clock, encode_time
from masy.daily_csv import WindOccurrence

__all__ = ["DailyWindCandidate", "DailyWindError", "DailyWindExecutor", "DailyWindLimits", "FinalizedDailyWind", "MAPLE_COMMIT",
           "MAPLE_PACKAGE_SHA256", "PROFILE", "RECEIPT_SCHEMA", "SYRUP_PACKAGE_SHA256", "prepare_daily_wind"]

PROFILE = "masy.daily_native_wind.cpu.v1"
_HANDOFF_PROFILE = "masy.daily_wind_handoff.cpu.v1"
RECEIPT_SCHEMA = "masy.daily_native_wind_receipt.v1"
# Same explicit qualified source pair as the water connection (MAPLE 9f commit/package, SYRUP N15 CPU package).
SYRUP_PACKAGE_SHA256 = "82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107"
MAPLE_COMMIT = "9f466d2cfcd52991b24496cf67f4562d6268afa9"
MAPLE_PACKAGE_SHA256 = "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"
SUPPORTED_WIND_MODE = "detachment_transport_deposition"
SUPPORTED_PATH = "baseline_direct"
_SHARED_FIELDS = ("voxel_column", "active_layer", "water", "ledger", "committed_topography", "sediment_availability")
_DETAIL_CHARS = 400


class DailyWindError(ValueError):
    """A refusal. ``evidence`` is a bounded scalar dict (never arrays, native results or tracebacks)."""

    def __init__(self, code: str, detail: str, *, evidence: dict[str, Any] | None = None):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail[:_DETAIL_CHARS]
        self.evidence = dict(evidence or {})


def _refuse(code: str, detail: str, **evidence: Any) -> DailyWindError:
    return DailyWindError(code, detail, evidence=evidence)


def _plain_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise DailyWindError("limits", f"{name} must be a plain int >= 1, got {value!r}")
    return value


@dataclasses.dataclass(frozen=True)
class DailyWindLimits:
    """Explicit caller resource policy, checked BEFORE any array copy or native physics. Never physics."""

    max_cells: int
    max_voxels: int
    max_cohorts: int
    max_blocks: int
    max_accounting_steps: int
    max_stage_bytes: int
    max_work_bytes: int
    max_receipt_bytes: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _plain_int(getattr(self, field.name), field.name)

    def as_record(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DailyWindCandidate:
    """Staged, NOT adopted result: a native ``BedState`` from the returned complete physical fields, the actual native end
    time and the bounded scalar receipt. Nothing here advances a lifecycle or a day cursor."""

    state: Any
    end_s: float
    receipt_json: str

    def receipt(self) -> dict[str, Any]:
        return json.loads(self.receipt_json)


@dataclasses.dataclass(frozen=True)
class FinalizedDailyWind:
    """Transient owning-model handoff evidence, returned only on explicit request.

    ``finalization`` is the actual MAPLE result, including its native scheduler
    and empty pool. Consumers should release it after native handoff staging;
    it is neither a scalar receipt nor a portable checkpoint. Ordinary execute
    returns only DailyWindCandidate and does not retain this wind workspace.
    """

    candidate: DailyWindCandidate
    finalization: Any = dataclasses.field(repr=False, compare=False)


def _native() -> SimpleNamespace:
    """Actual native symbols, imported lazily (same frozen paths as ``dense_runner`` / ``dense_bed``)."""
    import numpy as np
    from maple.aeolian.cohorts.dispatch import mobile_mass_by_class_kg, mobile_state_is_genuinely_empty
    from maple.aeolian.scheduler.accounting import accounting_step_durations_s
    from maple.aeolian.scheduler.block import reservoir_arithmetic_scale_kg
    from maple.aeolian.scheduler.finalization import WindEventFinalizationResult
    from maple.aeolian.scheduler.record import _EPS_S as record_end_epsilon_s
    from maple.aeolian.scheduler.record import BlockAccountingAccumulator
    from maple.aeolian.scheduler.state import (SchedulerPhysicalState, SchedulerPhysicsConfig,
                                               carries_coupled_trajectory_history, initial_scheduler_state)
    from maple.coupling.sediment_ledger.validation import validate_ledger_state
    from maple.core.parameters.wind_forcing import (WindBlock, resolve_wind_forcing_record,
                                                    validate_wind_forcing_record)
    from maple.core.types.cohort import MobileCohortState, zeros_mobile_cohort_state
    from maple.core.types.topographic_commit import validate_committed_topography_state
    from maple.experiment.scheduler_adapter import build_scheduler_physics_config
    from maple.surface.active_layer.validation import (check_active_layer_voxel_partition,
                                                       check_voxel_column_geometry_shape)
    from maple.surface.availability.validation import check_sediment_availability_state
    from maple.surface.voxels._numerics import summation_error_bound_kg
    from maple.surface.voxels.diagnostics import check_voxel_column_state
    from maple.water.validation import validate_water_state
    from maple_syrup.sediment_bed import BedContext, BedState, bed_inventory, is_windowed_bed

    return SimpleNamespace(**{name: value for name, value in locals().items()})


# --- native delegates (the only names negative tests may patch) ----------------------------------------------------

def _advance_wind_block(state: Any, block: Any, config: Any) -> Any:
    from maple.aeolian.scheduler.block import advance_wind_block

    return advance_wind_block(state, block, config)


def _finalize_wind_event(state: Any, config: Any, record: Any, *, record_origin_s: float) -> Any:
    from maple.aeolian.scheduler.finalization import finalize_wind_event

    return finalize_wind_event(state, config, record, record_origin_s=record_origin_s)


# --- helpers -----------------------------------------------------------------------------------------------------------

def _detail(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:_DETAIL_CHARS]


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _wrapper_sources(profile: str = PROFILE) -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = ("daily_wind.py", "daily_csv.py", "clock.py", "dense_bed.py", "dense_runner.py")
    if profile == _HANDOFF_PROFILE:
        names += ("daily_wind_handoff.py",)
    elif profile != PROFILE:
        raise _refuse("unsupported_profile", "unknown wind source profile")
    try:
        return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                names}
    except OSError as exc:
        raise _refuse("wrapper_source", _detail(exc)) from None


def _vec(np: Any, value: Any) -> Any:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _floats(np: Any, value: Any) -> list[float]:
    return [float(v) for v in _vec(np, value)]


def _hexes(np: Any, value: Any) -> list[str]:
    return [float(v).hex() for v in _vec(np, value)]


def _bed_helpers() -> SimpleNamespace:
    from masy import dense_bed as bed
    return SimpleNamespace(leaves=bed._leaves, nbytes=bed._tree_nbytes, canonical=bed._canonical_static,
                           array_digest=bed._array_digest, host=bed._require_host_arrays, error=bed.DenseBedError)


def _canonical(np: Any, value: Any, where: str) -> Any:
    from masy.dense_runner import DenseRunnerError, _canonical_input
    try:
        return _canonical_input(np, value, where)
    except DenseRunnerError as exc:
        detail = exc.detail
    raise _refuse("identity", detail)


def _bed_digest(bed: Any) -> str:
    h = _bed_helpers()
    try:
        return _sha({name: h.canonical(getattr(bed, name), name) for name in _SHARED_FIELDS})
    except h.error as exc:
        detail = exc.detail
    raise _refuse("identity", detail)


def _prescribed_digest(compiled: Any) -> str:
    h = _bed_helpers()
    try:
        return _sha({"vegetation": h.canonical(compiled.vegetation, "vegetation"),
                     "threshold_modifier_state": h.canonical(compiled.threshold_modifier_state, "threshold")})
    except h.error as exc:
        detail = exc.detail
    raise _refuse("identity", detail)


def _check_limits(limits: Any) -> DailyWindLimits:
    if type(limits) is not DailyWindLimits:
        raise DailyWindError("limits", "explicit DailyWindLimits are required")
    return limits


# --- source / admission / context ------------------------------------------------------------------------------------

def _source_facts(source: Any, admission: Any) -> dict[str, Any]:
    """Recheck genuine admission, CPU capability and the exact shared MAPLE source.

    Each explicit public profile additionally checks its own exact SYRUP pin.
    """
    try:
        from maple_syrup import native_case_profile as ncp
    except ImportError as exc:
        detail = _detail(exc)
    else:
        detail = None
    if detail is not None:
        raise _refuse("native_unavailable", detail)
    if type(source) is not ncp.NativeSourceProfile or type(admission) is not ncp.NativeCaseAdmission:
        raise _refuse("native_source", "genuine NativeSourceProfile and NativeCaseAdmission are required")
    detail = None
    try:
        source.recheck()
        admission.recheck()
        binding = admission.latitude_binding()
        facts = {"profile": source.profile, "maple_commit": source.maple_commit, "bundle_id": source.bundle_id,
                 "package_digest": source.package_digest, "project_digest": source.project_digest,
                 "syrup_package_digest": source.syrup_package_digest, "python": source.python,
                 "device_runtime_capability": source.device_runtime_capability,
                 "admission_profile": admission.profile_name, "admission_bundle_id": admission.bundle_id,
                 "admission_package_digest": admission.package_digest,
                 "admission_syrup_package_digest": admission.syrup_package_digest,
                 "case_identity_sha256": admission.case_identity_sha256, "latitude_deg": binding["value"],
                 "files": [[name, sha, size] for name, _path, sha, size in admission.files]}
    except ncp.NativeCaseProfileError as exc:
        detail = _detail(exc)
    if detail is not None:
        raise _refuse("native_source", detail)
    if (facts["profile"] != ncp.PROFILE_NAME or facts["device_runtime_capability"]
            or facts["maple_commit"] != MAPLE_COMMIT
            or facts["package_digest"] != MAPLE_PACKAGE_SHA256):
        raise _refuse("native_source", "the qualified MAPLE 9f CPU source is required; no automatic rebinding")
    if (facts["admission_profile"] != facts["profile"] or facts["admission_bundle_id"] != facts["bundle_id"]
            or facts["admission_package_digest"] != facts["package_digest"]
            or facts["admission_syrup_package_digest"] != facts["syrup_package_digest"]):
        raise _refuse("native_source", "the admission was not produced under this source profile")
    return facts


def _check_source(source: Any, admission: Any) -> dict[str, Any]:
    """The historical public factory remains pinned to the qualified N15 pair."""
    facts = _source_facts(source, admission)
    if facts["syrup_package_digest"] != SYRUP_PACKAGE_SHA256:
        raise _refuse("native_source", "explicit MAPLE 9f / SYRUP N15 CPU source pair required; no automatic rebinding")
    return facts


def _check_source_for_profile(source: Any, admission: Any, profile: str) -> dict[str, Any]:
    if profile == PROFILE:
        return _check_source(source, admission)
    if profile == _HANDOFF_PROFILE:
        from masy.daily_wind_handoff import _check_handoff_source
        return _check_handoff_source(source, admission)
    raise _refuse("unsupported_profile", "unknown wind source profile")


def _same(a: Any, b: Any) -> bool:
    return a is b or bool(a == b)


def _check_context(n: SimpleNamespace, admission: Any, context: Any) -> None:
    np = n.np
    if type(context) is not n.BedContext:
        raise _refuse("context", "context must be a native BedContext")
    config = admission.compiled.config
    for name, mine, theirs in (("geometry", context.geometry, config.geometry),
                               ("grain_classes", context.grain_classes, config.grain_classes),
                               ("avalanching_spec", context.avalanching_spec, config.avalanching),
                               ("topographic_commit_spec", context.topographic_commit_spec, config.topographic_commit)):
        if not _same(mine, theirs):
            raise _refuse("context", f"context.{name} is not the admitted case's")
    if float(context.mass_resolution_kg).hex() != float(config.numerics.mass_resolution_kg).hex():
        raise _refuse("context", "context mass_resolution_kg differs from the admitted case")
    g, nc = context.geometry, len(context.grain_classes.classes)
    fraction = context.initial_available_fraction
    if (type(fraction) is not np.ndarray or fraction.dtype != np.float64 or fraction.shape != (g.ny, g.nx, nc)
            or not bool(np.isfinite(fraction).all())):
        raise _refuse("context", "context.initial_available_fraction must be finite host float64 (ny, nx, n_classes)")
    if bool(context.avalanching_spec.enabled):
        raise _refuse("unsupported_profile", "avalanching must be disabled")
    if str(context.depth_update_rule) != "constant_depth":
        raise _refuse("context", "only the constant_depth water rule is carried")


def _check_config_controls(config: Any) -> None:
    """Admitted native configuration controls. Water-enabled or alternate configurations are REFUSED, never changed."""
    if bool(config.water_coupling.enabled):
        raise _refuse("water_coupling_enabled", "the admitted configuration enables native water coupling; refused")
    if bool(config.topographic_commit.dry_aeolian_commit_separation_enabled):
        raise _refuse("unsupported_profile", "dry_aeolian_commit_separation is enabled; refused")
    if bool(config.prescribed_external_shear.enabled):
        raise _refuse("unsupported_profile", "a prescribed external shear field is not connected; refused")
    if config.aeolian_process_mode.mode != SUPPORTED_WIND_MODE:
        raise _refuse("unsupported_profile", f"wind mode {config.aeolian_process_mode.mode!r}; only {SUPPORTED_WIND_MODE}")
    if getattr(config.execution, "backend", "numpy") != "numpy":
        raise _refuse("unsupported_profile", "only the NumPy CPU backend is connected")


def _effective_wind_config(config: Any, context: Any) -> Any:
    """Explicit user-directed wind view; actual water and authored inputs stay intact.

    MAPLE's ordinary free-surface commit rule changes water volume when the bed
    moves even with water coupling disabled. The existing BedContext instead
    declares constant depth, which preserves the SYRUP-owned water volume.
    Suppressing native moisture feedback implements the user's dry-wind rule.
    Every other native control is passed through unchanged.
    """
    return dataclasses.replace(
        config,
        water_coupling=dataclasses.replace(config.water_coupling, depth_update_rule=context.depth_update_rule),
        fluid_threshold=dataclasses.replace(config.fluid_threshold, modifiers=dataclasses.replace(
            config.fluid_threshold.modifiers, moisture_enabled=False)))


# --- the executor -------------------------------------------------------------------------------------------------------

def prepare_daily_wind(source: Any, admission: Any, context: Any, *, limits: DailyWindLimits) -> "DailyWindExecutor":
    """Validate the explicit source pair, admission, context and limits; return the executor (no bed, no physics)."""
    return _prepare_daily_wind(source, admission, context, limits=limits, profile=PROFILE)


def _prepare_daily_wind(source: Any, admission: Any, context: Any, *, limits: DailyWindLimits, profile: str) -> "DailyWindExecutor":
    """Shared mechanics behind two explicitly selected, separately qualified profiles."""
    limits = _check_limits(limits)
    facts = _check_source_for_profile(source, admission, profile)
    n = _native()
    _check_context(n, admission, context)
    _check_config_controls(admission.compiled.config)
    g = context.geometry
    if g.ny * g.nx > limits.max_cells:
        raise _refuse("cell_limit", f"{g.ny * g.nx} cells > max_cells={limits.max_cells}")
    identity = {"profile": profile, "source": facts, "latitude_deg": encode_time(float(facts["latitude_deg"])),
                "context": {"mass_resolution_kg": encode_time(float(context.mass_resolution_kg)),
                            "initial_available_fraction": _bed_helpers().array_digest(context.initial_available_fraction),
                            "geometry": _canonical(n.np, context.geometry, "geometry"),
                            "grain_classes": _canonical(n.np, context.grain_classes, "grain_classes"),
                            "depth_update_rule": context.depth_update_rule},
                "prescribed_sha256": _prescribed_digest(admission.compiled),
                "admitted_config_sha256": _sha(_canonical(n.np, admission.compiled.config, "model_config")),
                "wrapper_sources": _wrapper_sources(profile)}
    return DailyWindExecutor(source, admission, context, limits, identity)


class DailyWindExecutor:
    """Holds only the admitted source/case/context references, limits and identity: no bed, no history."""

    def __init__(self, source: Any, admission: Any, context: Any, limits: DailyWindLimits, identity: dict[str, Any]):
        self._source, self._admission, self._context, self._limits = source, admission, context, limits
        self._identity_json = _json(identity)
        self._identity_sha256 = hashlib.sha256(self._identity_json.encode()).hexdigest()

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    def identity(self) -> dict[str, Any]:
        return json.loads(self._identity_json)

    def _recheck(self) -> None:
        profile = self.identity()["profile"]
        if _wrapper_sources(profile) != self.identity()["wrapper_sources"]:
            raise _refuse("wrapper_source_changed", "wrapper sources changed since preparation")
        if _check_source_for_profile(self._source, self._admission, profile) != self.identity()["source"]:
            raise _refuse("native_source", "source/admission association changed since preparation")
        if _prescribed_digest(self._admission.compiled) != self.identity()["prescribed_sha256"]:
            raise _refuse("prescribed_changed", "native prescribed vegetation/threshold fields changed")
        n = _native()
        _check_context(n, self._admission, self._context)
        if _bed_helpers().array_digest(self._context.initial_available_fraction) != self.identity()["context"]["initial_available_fraction"]:
            raise _refuse("context_changed", "the admitted availability policy changed")
        _check_config_controls(self._admission.compiled.config)
        if _sha(_canonical(n.np, self._admission.compiled.config, "model_config")) != self.identity()["admitted_config_sha256"]:
            raise _refuse("config_changed", "the admitted model configuration changed")

    def execute(self, bed: Any, occurrence: Any, start_s: float) -> DailyWindCandidate:
        """Run ONE complete occurrence from the actual current ``bed`` at physical ``start_s``. Never mutates ``bed``."""
        return _execute(self, bed, occurrence, start_s)

    def execute_finalized(self, bed: Any, occurrence: Any, start_s: float) -> FinalizedDailyWind:
        """Run the same checks and return the actual transient MAPLE completion too.

        No lifecycle is adopted. Release the completion after staging with the
        owning water API; it retains native wind workspace until released.
        """
        return _execute(self, bed, occurrence, start_s, with_completion=True)


def _plan_blocks(n: SimpleNamespace, record: Any, outer: float, limits: DailyWindLimits) -> list[dict[str, Any]]:
    """Exact accounting-step bounds and the native split's coverage, per block; before any array copy."""
    out, total = [], 0
    for block in record.blocks:
        quotient = Fraction(block.duration_s) / Fraction(outer)
        upper = -(-quotient.numerator // quotient.denominator) + 1
        total += upper
        if total > limits.max_accounting_steps:
            raise _refuse("accounting_step_limit", f"up to {total} accounting steps > {limits.max_accounting_steps}")
        plan = n.accounting_step_durations_s(block.duration_s, outer)
        requested = Fraction(block.duration_s)
        covered = sum((Fraction(dt) for dt in plan), Fraction(0))
        rounding = Fraction(2, 2 ** 53) * max(requested, covered)
        if len(plan) > upper or covered <= 0 or abs(covered - requested) > rounding:
            raise _refuse("unsupported_wind_duration", f"native accounting split covers {float(covered)!r} s of the "
                          f"requested {block.duration_s!r} s beyond its own rounding")
        out.append({"upper": upper, "plan_steps": len(plan), "plan_discrepancy": covered - requested})
    return out


def _validate_bed(n: SimpleNamespace, bed: Any, context: Any, start_s: float) -> tuple[int, int]:
    """Exact types, host layout, native validators; returns (n_voxels, n_classes). Never mutates."""
    np, h = n.np, _bed_helpers()
    if type(bed) is not n.BedState:
        raise _refuse("invalid_bed", "bed must be a native BedState")
    for name in _SHARED_FIELDS:
        if getattr(bed, name) is None:
            raise _refuse("invalid_bed", f"{name} is None; shared fields are never dropped")
        try:
            h.host(getattr(bed, name), name)
        except h.error as exc:
            detail = exc.detail
        else:
            continue
        raise _refuse("unsupported_backend", detail)
    if n.is_windowed_bed(bed):
        raise _refuse("unsupported_profile", "windowed beds are unqualified here")
    g, nc = context.geometry, len(context.grain_classes.classes)
    mass = bed.voxel_column.mass_kg
    if mass.ndim != 4 or mass.shape[:2] != (g.ny, g.nx) or mass.shape[3] != nc or mass.dtype != np.float64:
        raise _refuse("invalid_bed", f"voxel mass {mass.shape}/{mass.dtype} is not (ny, nx, nv, nc) float64")
    return int(mass.shape[2]), nc


def _validate_native(n: SimpleNamespace, bed: Any, context: Any, start_s: float) -> None:
    g, mres, nc = context.geometry, context.mass_resolution_kg, len(context.grain_classes.classes)
    detail = None
    try:
        n.check_voxel_column_geometry_shape(bed.voxel_column, g)
        n.check_voxel_column_state(bed.voxel_column, g, mres)
        n.check_active_layer_voxel_partition(bed.active_layer, bed.voxel_column, g, mres)
        n.check_sediment_availability_state(bed.sediment_availability, bed.active_layer, mres)
        n.validate_water_state(bed.water, g.ny, g.nx, nc)
        dims = n.validate_ledger_state(bed.ledger)
        committed = n.validate_committed_topography_state(bed.committed_topography)
        if tuple(dims) != (g.ny, g.nx, nc) or tuple(committed) != (g.ny, g.nx, nc):
            detail = "ledger/committed topography dimensions differ from the context"
        stamp = _native_clock(float(bed.committed_topography.last_commit_time_s), "last_commit_time_s")
        if detail is None and stamp > start_s:
            detail = f"committed topography stamp {stamp.hex()} is after the start {start_s.hex()}"
    except (ValueError, TypeError) as exc:
        detail = _detail(exc)
    if detail is not None:
        raise _refuse("invalid_bed", detail)
    mobile = bed.water.mobile_mass_by_cell_class_kg
    if not bool(n.np.isfinite(mobile).all()) or bool(n.np.any(mobile != 0.0)):
        raise _refuse("water_mobile_not_empty", "water-borne sediment is present; a dry wind entry needs none")


def _execute(self: DailyWindExecutor, bed: Any, occurrence: Any, start_s: float, *, with_completion: bool = False) -> DailyWindCandidate | FinalizedDailyWind:
    limits, context, compiled = self._limits, self._context, self._admission.compiled
    if type(occurrence) is not WindOccurrence or not occurrence.blocks:
        raise _refuse("occurrence", "a parsed WindOccurrence with at least one block is required")
    detail = None
    try:
        start_s = _native_clock(start_s, "start_s")
    except ClockError as exc:
        detail = exc.detail
    if detail is not None:
        raise _refuse("start_time", detail)
    if type(start_s) is not float:
        raise _refuse("start_time", "start_s must be a float native physical clock")
    if len(occurrence.blocks) > limits.max_blocks:
        raise _refuse("block_limit", f"{len(occurrence.blocks)} blocks > max_blocks={limits.max_blocks}")
    self._recheck()
    n = _native()
    np, h = n.np, _bed_helpers()
    nv, nc = _validate_bed(n, bed, context, start_s)
    g = context.geometry
    if g.ny * g.nx > limits.max_cells or nv > limits.max_voxels:
        raise _refuse("cell_limit", f"{g.ny * g.nx} cells / {nv} voxels exceed max_cells/max_voxels")
    stage_bytes = h.nbytes(*(getattr(bed, name) for name in _SHARED_FIELDS))
    if stage_bytes > limits.max_stage_bytes:
        raise _refuse("stage_bytes_cap", f"{stage_bytes} bytes > max_stage_bytes={limits.max_stage_bytes}")

    # Forcing record (native resolver; local to zero) and native config with the record's own envelope.
    declared = [{"shear_velocity_m_s": b.shear_velocity_m_s, "direction_deg": b.direction_deg, "duration_s": b.duration_s}
                for b in occurrence.blocks]
    detail = None
    try:
        record = n.resolve_wind_forcing_record({"blocks": declared})
        n.validate_wind_forcing_record(record)
        effective = _effective_wind_config(compiled.config, context)
        config = n.build_scheduler_physics_config(
            effective, initial_available_fraction=context.initial_available_fraction, wind_record=record)
    except (ValueError, TypeError, KeyError) as exc:
        detail = _detail(exc)
    if detail is not None:
        raise _refuse("native_config", detail)
    fraction = config.initial_available_fraction
    if (type(config) is not n.SchedulerPhysicsConfig or fraction is None
            or not bool(np.array_equal(fraction, context.initial_available_fraction))
            or bool(config.water_coupling_spec.enabled) or bool(config.topographic_commit_spec.dry_aeolian_commit_separation_enabled)
            or config.water_coupling_spec.depth_update_rule != "constant_depth"
            or config.fluid_threshold_spec.modifiers.moisture_enabled
            or float(config.mass_resolution_kg).hex() != float(context.mass_resolution_kg).hex()
            or type(config.outer_timestep_s) is not float or not math.isfinite(config.outer_timestep_s)
            or config.outer_timestep_s <= 0.0):
        raise _refuse("native_config", "native physics config is not the dry, admitted, availability-preserving profile")
    if config.particle_transport.max_active_cohorts > limits.max_cohorts:
        raise _refuse("cohort_limit", f"{config.particle_transport.max_active_cohorts} cohorts > max_cohorts="
                      f"{limits.max_cohorts}")
    plans = _plan_blocks(n, record, config.outer_timestep_s, limits)
    upper_total = sum(p["upper"] for p in plans)
    # Inspect only the zero-capacity native factory's dtypes before admitting a
    # real pool. No array of configured cohort/domain size is allocated yet.
    empty_pool = n.zeros_mobile_cohort_state(max_cohorts=0)
    leaves = []
    h.leaves(empty_pool, "mobile", leaves)
    pool_bytes = config.particle_transport.max_active_cohorts * sum(a.dtype.itemsize for _, a in leaves)
    work_bytes = 4 * stage_bytes + 3 * pool_bytes + 8 * g.ny * g.nx * (nv + 1) * nc + 32 * upper_total
    if work_bytes > limits.max_work_bytes:
        raise _refuse("work_bytes_cap", "estimated work copies exceed max_work_bytes")
    from maple_syrup.subsurface import default_host_available_bytes
    if default_host_available_bytes() < work_bytes:
        raise _refuse("host_memory", "available host memory does not admit the bounded wind workspace")
    pool = n.zeros_mobile_cohort_state(max_cohorts=config.particle_transport.max_active_cohorts)
    if h.nbytes(pool) != pool_bytes:
        raise _refuse("pool_layout", "native cohort allocation differs from its declared dtype layout")
    if not n.mobile_state_is_genuinely_empty(pool):
        raise _refuse("mobile_state_not_empty", "the fresh native pool is not genuinely empty")
    _validate_native(n, bed, context, start_s)

    parent_sha, prescribed_sha = _bed_digest(bed), _prescribed_digest(compiled)
    inventory_before = _vec(np, n.bed_inventory(bed)).copy()
    staged = copy.deepcopy(bed)
    if _bed_digest(staged) != parent_sha:
        raise _refuse("stage_mismatch", "staged copy does not reproduce the parent identity")
    scheduler = n.initial_scheduler_state(
        staged.voxel_column, staged.active_layer, pool, staged.ledger, staged.committed_topography,
        compiled.threshold_modifier_state, time_s=start_s, vegetation=compiled.vegetation,
        sediment_availability=staged.sediment_availability, water=staged.water)
    probe = scheduler.physical
    if (n.carries_coupled_trajectory_history(probe, None) or float(probe.time_s).hex() != start_s.hex()
            or any(getattr(probe, name) is not getattr(staged, name) for name in _SHARED_FIELDS)):
        raise _refuse("native_factory_mismatch", "initial_scheduler_state did not carry the staged shared objects")

    state, acc = scheduler, n.BlockAccountingAccumulator(nc)
    blocks, bounds, scale, previous = [], [], 0.0, start_s
    for index, rb in enumerate(record.blocks):
        t0 = float(state.physical.time_s)
        block = n.WindBlock(shear_velocity_m_s=rb.shear_velocity_m_s, direction_deg=rb.direction_deg,
                            duration_s=rb.duration_s, start_time_s=t0, end_time_s=t0 + rb.duration_s)
        stop = None
        try:
            result = _advance_wind_block(state, block, config)
        except Exception as exc:  # noqa: BLE001 - every native failure refuses; no traceback is retained
            stop = ("native_failure", _detail(exc))
        if stop is not None:
            raise _refuse(stop[0], stop[1], stage="block", block_index=index)
        step = _check_block(self, n, index, block, result, plans[index], upper_total, acc)
        t1 = float(result.state.physical.time_s)
        if not (t1 > t0 >= previous):
            raise _refuse("wind_no_elapsed", "native physical time did not advance through the block", block_index=index)
        previous = t1
        scale += step["scale"]
        bounds.append(step["bound"])
        blocks.append({"index": index, "expected_duration_s": encode_time(rb.duration_s),
                       "start_s": encode_time(t0), "nominal_end_s": encode_time(block.end_time_s),
                       "realized_end_s": encode_time(t1),
                       "realized_elapsed_s": _fraction_text(Fraction(t1) - Fraction(t0)),
                       "elapsed_discrepancy_s": _fraction_text(Fraction(t1) - Fraction(t0) - Fraction(rb.duration_s)),
                       "plan_steps": plans[index]["plan_steps"],
                       "plan_discrepancy_s": _fraction_text(plans[index]["plan_discrepancy"]),
                       "n_accounting_steps": step["steps"], "native_residual_max_kg": step["residual"].hex(),
                       "native_bound_kg": step["bound"].hex(), "n_commits": step["commits"]})
        # The native record loop's terminal cursor mapping, applied at the same point of each block.
        state = dataclasses.replace(result.state, block_index=index + 1, block_offset_s=0.0)
        del result
    total = acc.total
    t_end = float(state.physical.time_s)

    stop = None
    try:
        final = _finalize_wind_event(state, config, record, record_origin_s=start_s)
    except Exception as exc:  # noqa: BLE001 - refuses unpublished; the mobile pool is never discarded
        stop = _detail(exc)
    if stop is not None:
        raise _refuse("finalization_failure", stop, stage="finalize", elapsed_s=_fraction_text(Fraction(t_end) - Fraction(start_s)))
    classes, candidate, checks = _check_final(self, n, final, state, total, bounds, scale, inventory_before, bed,
                                              record, start_s, t_end, upper_total, nv, nc)
    # Native pure on the parent: verify it, the prescribed fields and the source are unchanged.
    if _bed_digest(bed) != parent_sha or _prescribed_digest(compiled) != prescribed_sha:
        raise _refuse("parent_mutated", "the input bed or prescribed fields changed during native execution")
    self._recheck()
    receipt = {
        "schema": RECEIPT_SCHEMA, "profile": self.identity()["profile"], "executor_identity_sha256": self._identity_sha256,
        "executor_identity": self.identity(), "limits": limits.as_record(),
        "config_sha256": _sha(_canonical(np, config, "config")),
        "wind_policy": {"soil_moisture_feedback_enabled": False,
                        "admitted_moisture_feedback_enabled": bool(compiled.config.fluid_threshold.modifiers.moisture_enabled),
                        "water_depth_update_rule": config.water_coupling_spec.depth_update_rule,
                        "admitted_water_depth_update_rule": compiled.config.water_coupling.depth_update_rule,
                        "actual_water_state_reset": False},
        "record_sha256": _sha(_canonical(np, record, "record")),
        "occurrence": {"index": occurrence.index, "occurrence_id": occurrence.occurrence_id,
                       "occurrence_sha256": occurrence.occurrence_sha256, "date": occurrence.date.isoformat(),
                       "reference": occurrence.reference, "file_sha256": occurrence.sha256,
                       "declared_duration_s": encode_time(occurrence.declared_duration_s)},
        "parent_bed_sha256": parent_sha, "candidate_bed_sha256": _bed_digest(candidate),
        "clocks": {"start_s": encode_time(start_s), "end_s": encode_time(t_end),
                   "expected_s": _fraction_text(sum((Fraction(b.duration_s) for b in record.blocks), Fraction(0))),
                   "realized_s": _fraction_text(Fraction(t_end) - Fraction(start_s)),
                   "record_origin_s": encode_time(start_s), "settled_clock_s": encode_time(t_end)},
        "blocks": blocks, "n_accounting_steps": int(total.n_accounting_steps), "classes": classes, "checks": checks,
        "water": {"ponded_depth_preserved_bitwise": True, "water_borne_sediment_empty": True,
                  "physical_water_update": "none"},
        "scope": "native_wind_component_not_a_wind_water_daily_run; no lifecycle or day cursor adopted",
    }
    text = _json(receipt)
    if len(text.encode()) > limits.max_receipt_bytes:
        raise _refuse("receipt_bytes_cap", f"receipt {len(text.encode())} bytes > max_receipt_bytes")
    out = DailyWindCandidate(candidate, t_end, text)
    return FinalizedDailyWind(out, final) if with_completion else out


def _check_block(self: DailyWindExecutor, n: SimpleNamespace, index: int, block: Any, result: Any,
                 plan: dict[str, Any], upper_total: int, acc: Any) -> dict[str, Any]:
    """Native per-block checks (same as ``dense_runner``); folds the block into the native accumulator."""
    np = n.np
    physical, a = result.state.physical, result.accounting

    def bad(code: str, detail: str) -> DailyWindError:
        return _refuse(code, detail, stage="block", block_index=index)

    if result.process_mode != SUPPORTED_WIND_MODE or result.integration_path != SUPPORTED_PATH:
        raise bad("unsupported_wind_path", f"{result.process_mode}/{result.integration_path}")
    if result.conservation_ok is not True:
        raise bad("wind_conservation_failed", "native check_block_conservation reported failure")
    for name in ("mobile_mass_before_by_class_kg", "mobile_mass_after_by_class_kg", "boundary_input_by_class_kg",
                 "boundary_export_by_class_kg"):
        value = getattr(a, f"water_{name}", None)
        if value is not None and bool(np.any(_vec(np, value) != 0.0)):
            raise bad("wind_water_pass", "native wind accounting reports water-borne exchange")
    if a.n_accounting_steps > plan["upper"] or a.n_accounting_steps > self._limits.max_accounting_steps:
        raise bad("accounting_step_limit", f"{a.n_accounting_steps} native accounting steps")
    if bool(np.any(physical.water.mobile_mass_by_cell_class_kg != 0.0)):
        raise bad("water_mobile_not_empty", "water-borne load appeared during wind")
    if result.state.block_index != index or result.state.block_offset_s < block.duration_s - n.record_end_epsilon_s:
        raise bad("wind_record_incomplete", "native block did not reach its terminal cursor")
    bed_change = _vec(np, a.bed_change_by_class_kg)
    delta = _vec(np, a.mobile_mass_after_by_class_kg) - _vec(np, a.mobile_mass_before_by_class_kg)
    inflow, export = _vec(np, a.boundary_input_by_class_kg), _vec(np, a.boundary_export_by_class_kg)
    residual = bed_change + delta - inflow + export
    bound = float(n.summation_error_bound_kg(max(a.n_accounting_steps, 1), n.reservoir_arithmetic_scale_kg(a, physical)))
    if not bool(np.all(np.abs(residual) <= bound)):
        raise bad("wind_budget_open", "native per-class reservoir identity exceeds its native bound")
    acc.add(a)
    return {"steps": int(a.n_accounting_steps), "commits": int(result.n_commits_this_block),
            "residual": float(np.abs(residual).max()), "bound": bound,
            "scale": sum(float(np.abs(v).sum()) for v in (bed_change, inflow, export))}


def _check_final(self: DailyWindExecutor, n: SimpleNamespace, final: Any, state: Any, total: Any, bounds: list[float],
                 scale: float, inventory_before: Any, parent: Any, record: Any, start_s: float, t_end: float,
                 upper_total: int, nv: int, nc: int) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Finalizer checks and whole-bed + boundary per-class closure, using native tolerance helpers only."""
    np, h, compiled, context = n.np, _bed_helpers(), self._admission.compiled, self._context
    elapsed = _fraction_text(Fraction(t_end) - Fraction(start_s))

    def bad(code: str, detail: str) -> DailyWindError:
        return _refuse(code, detail, stage="finalize", elapsed_s=elapsed)

    if type(final) is not n.WindEventFinalizationResult:
        raise bad("invalid_wind_finalization", "unexpected native finalizer result type")
    if not final.completed or not final.water_handoff_ready:
        raise bad("wind_finalization_incomplete", str(final.reasons)[:_DETAIL_CHARS])
    settled = final.state
    physical = settled.physical
    if (float(physical.time_s).hex() != t_end.hex() or float(final.time_s).hex() != t_end.hex()
            or settled.block_index != len(record.blocks) or settled.block_offset_s != 0.0):
        raise bad("wind_finalization_clock_changed", "native finalizer changed the clock or cursor")
    if not n.mobile_state_is_genuinely_empty(physical.mobile_state):
        raise bad("wind_finalization_mobile_nonempty", "native finalizer left airborne material")
    names = ("airborne_mass_before_by_class_kg", "airborne_mass_after_by_class_kg", "deposited_mass_by_class_kg",
             "boundary_export_mass_by_class_kg", "bed_change_by_class_kg", "transfer_identity_gap_by_class_kg",
             "reservoir_identity_gap_by_class_kg")
    fv = {name: _vec(np, getattr(final, name)) for name in names}
    if any(v.shape != (nc,) or not bool(np.isfinite(v).all()) for v in fv.values()):
        raise bad("wind_finalization_budget_invalid", "nonfinite or incompatible class vector")
    for prefix in ("transfer", "reservoir"):
        bound = float(getattr(final, prefix + "_identity_bound_kg"))
        if not math.isfinite(bound) or bound < 0.0 or not bool(np.all(np.abs(fv[prefix + "_identity_gap_by_class_kg"]) <= bound)):
            raise bad("wind_finalization_budget_open", "native finalizer class budget failed")
    last_mobile = n.mobile_mass_by_class_kg(state.physical.mobile_state, nc)
    if bool(np.any(fv["airborne_mass_after_by_class_kg"] != 0.0)) or not bool(
            np.array_equal(fv["airborne_mass_before_by_class_kg"], _vec(np, last_mobile))):
        raise bad("wind_finalization_mobile_budget", "native mobile inventory disagrees")
    mobile_first = _vec(np, total.mobile_mass_before_by_class_kg)
    if bool(np.any(mobile_first != 0.0)):
        raise bad("mobile_state_not_empty", "the first native block did not start from an empty pool")
    if not bool(np.array_equal(_vec(np, total.mobile_mass_after_by_class_kg), _vec(np, last_mobile))):
        raise bad("wind_budget_open", "accumulated mobile inventory differs from the last native block")

    inflow, transport_export = _vec(np, total.boundary_input_by_class_kg), _vec(np, total.boundary_export_by_class_kg)
    transport_bed_change = _vec(np, total.bed_change_by_class_kg)
    export = transport_export + fv["boundary_export_mass_by_class_kg"]
    bed_change = transport_bed_change + fv["bed_change_by_class_kg"]
    candidate = n.BedState(physical.voxel_column, physical.active_layer, physical.water, physical.ledger,
                           physical.committed_topography, physical.sediment_availability)
    if (physical.vegetation is not compiled.vegetation or physical.threshold_modifier_state is not compiled.threshold_modifier_state
            or physical.ledger_reconciliation_baseline is not None):
        raise bad("prescribed_field_replaced", "native result replaced prescribed fields or carries a baseline")
    _validate_native(n, candidate, context, t_end)
    # Ponded depth (and every water leaf) is carried exactly; nothing in wind updates water.
    water_in, water_out = [], []
    h.leaves(parent.water, "water", water_in)
    h.leaves(candidate.water, "water", water_out)
    if len(water_in) != len(water_out) or any(
            a[1].dtype != b[1].dtype or a[1].shape != b[1].shape or not bool(np.array_equal(a[1], b[1]))
            for a, b in zip(water_in, water_out)):
        raise bad("water_changed", "carried water state differs from the parent; wind must not update water")

    inventory_after = _vec(np, n.bed_inventory(candidate))
    mobile_delta = fv["airborne_mass_after_by_class_kg"] - mobile_first
    residual = bed_change + mobile_delta - inflow + export
    n_blocks = len(record.blocks)
    event_scale = scale + sum(float(np.abs(v).sum()) for v in (
        fv["bed_change_by_class_kg"], fv["airborne_mass_after_by_class_kg"], mobile_first,
        fv["boundary_export_mass_by_class_kg"]))
    reservoir_bound = float(final.reservoir_identity_bound_kg)
    native_bound = float(sum(bounds))
    event_bound = native_bound + reservoir_bound + float(n.summation_error_bound_kg(6 * n_blocks, event_scale))
    if not bool(np.all(np.abs(residual) <= event_bound)):
        raise bad("wind_event_budget_open", "combined transport/finalizer class identity failed")
    g = context.geometry
    inv_residual = inventory_after - inventory_before + mobile_delta - inflow + export
    inv_scale = max(float(np.abs(inventory_before).sum()), float(np.abs(inventory_after).sum()),
                    float(inflow.sum()), float(export.sum()))
    inv_bound = native_bound + reservoir_bound + float(
        n.summation_error_bound_kg(2 * g.ny * g.nx * (nv + 1) + 6 * n_blocks, inv_scale))
    if not bool(np.all(np.abs(inv_residual) <= inv_bound)):
        raise bad("wind_inventory_open", "actual bed inventories do not close within native bounds")
    classes = {
        "class_ids": [c.class_id for c in context.grain_classes.classes],
        "boundary_input_kg": _floats(np, inflow), "boundary_input_hex": _hexes(np, inflow),
        "boundary_export_kg": _floats(np, export), "boundary_export_hex": _hexes(np, export),
        "transport_boundary_export_kg": _floats(np, transport_export),
        "bed_change_kg": _floats(np, bed_change), "bed_change_hex": _hexes(np, bed_change),
        "transport_bed_change_kg": _floats(np, transport_bed_change),
        "settled_deposition_kg": _floats(np, fv["deposited_mass_by_class_kg"]),
        "settled_boundary_export_kg": _floats(np, fv["boundary_export_mass_by_class_kg"]),
        "settled_airborne_before_kg": _floats(np, fv["airborne_mass_before_by_class_kg"]),
        "settled_airborne_after_kg": _floats(np, fv["airborne_mass_after_by_class_kg"]),
        "inventory_before_kg": _floats(np, inventory_before), "inventory_after_kg": _floats(np, inventory_after),
        "event_residual_kg": _floats(np, residual), "event_bound_kg": event_bound,
        "inventory_residual_kg": _floats(np, inv_residual), "inventory_bound_kg": inv_bound,
    }
    checks = {"native_block_checks": True, "finalizer_checks": True, "event_class_identity": True,
              "inventory_closure_with_boundary_export": True, "finalization_status": str(final.status),
              "n_settled": int(final.n_settled), "n_exported": int(final.n_exported),
              "settlement_counted_once": "finalizer deposition is internal to the returned bed; only its boundary export is added",
              "transfer_identity_bound_kg": float(final.transfer_identity_bound_kg),
              "reservoir_identity_bound_kg": reservoir_bound, "native_block_bound_sum_kg": native_bound,
              "accounting_step_upper_bound": upper_total}
    return classes, candidate, checks
