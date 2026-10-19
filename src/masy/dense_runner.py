"""Thin Dense NumPy CPU runner: ordered wind / rain events on ONE native evolving bed.

`DenseRunner` executes a `DenseSchedule` event by event on one `DenseBed`.
It holds no physics and makes no physical decision: every transport step,
commit, reroute decision, completion and dry reset is an actual MAPLE /
SYRUP call (`advance_wind_block`, `commit_bed`, `complete_event`, ...). The
caller supplies every native input (wind config and empty pool, terrain,
column, rainfall field, vegetation cover fraction, sediment parameters,
controls); this module never imports a test helper or a case CLI.

Supported profile (refused otherwise, before any mutation): NumPy host
float64 Dense square full grid with fixed voxel capacity (`DenseBed`); flat
wind cohorts; aerodynamic `detachment_transport_deposition` with no MAPLE
water pass in wind; avalanching disabled; SYRUP `upwind` transport (phase
None) with the `array` hydraulic implementation.

Transaction. Before any work copy or native call, the case inputs, every
controlling static input and the accepted bed / exposed pool / soil water are
re-verified against their accepted identities (a change made outside the
runner is refused as such and never repaired). Every event then runs on
`DenseBed.stage_copy()` plus private work copies of the wind pool, soil water
and the retained transport memory (e.g. sediment velocity). Per-event and
aggregate per-class budgets, all candidate totals / counters, routing, soil,
pool, the candidate `ClockLedger`, receipt list, returned receipt and
static / accepted-identity rechecks are all prepared BEFORE the single bed
adoption; afterwards only prepared references are assigned. Any refusal,
native failure, limit or incomplete native outcome stops the runner with
the last accepted boundary intact (bed, clocks, counters, routing, soil,
pool, cursor). Unaccepted native output is kept only as the latest
`active_evidence`; there is no resume of a stopped event. See
docs/dense_runner.md.

Importing this module imports neither numpy nor a native model.
An explicit opt-in reduced profile is bounded separately in docs/dense_production_wind.md.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import inspect
import math
from fractions import Fraction
from types import ModuleType, SimpleNamespace
from typing import Any

from masy import dense_bed as _bed
from masy.clock import ClockError, ClockLedger, decode_time, encode_time
from masy.dense_bed import SHARED_FIELDS, DenseBed, DenseBedError
from masy.dense_schedule import DenseSchedule, DenseScheduleError, WindEvent, case_schedule_digest, resolve_event

__all__ = [
    "PROFILE",
    "RECEIPT_SCHEMA",
    "REPORT_SCHEMA",
    "STATIC_SCHEMA",
    "AcceptedBoundary",
    "DenseRunner",
    "DenseRunnerControl",
    "DenseRunnerError",
    "DenseRunnerLimits",
    "DenseWaterInputs",
    "DenseWindInputs",
]

STATIC_SCHEMA = "masy.dense_runner_static.v1"
RECEIPT_SCHEMA = "masy.dense_runner_event.v1"
REPORT_SCHEMA = "masy.dense_runner_report.v1"
SUPPORTED_WIND_MODE = "detachment_transport_deposition"
PROFILE = {
    "backend": "numpy", "dtype": "float64", "representation": "dense_voxel_column", "grid": "square_full",
    "voxel_capacity": "fixed", "wind_cohorts": "flat", "wind_mode": SUPPORTED_WIND_MODE,
    "wind_detachment": "aerodynamic", "wind_water_pass": "none", "avalanching": "disabled",
    "water_transport_scheme": "upwind", "water_phase": None, "hydraulic_implementation": "array",
    "wind_end": "native_instantaneous_finalization",
}
# Wind config fields that select an alternate (unqualified) mode when set.
_UNSUPPORTED_WIND_FIELDS = ("prescribed_detachment", "reduced_detachment", "prescribed_external_shear",
                            "surface_creep_spec", "transport_accuracy", "cohort_calendar_max_bytes")
_DETAIL_CHARS = 400
# SedimentEventState fields the runner replaces at every rain entry; every OTHER array field is retained
# process-private memory and is handed to native code only as a work copy.
_ENTRY_REPLACED = frozenset({"bed", "graph", "network", "grid", "terrain", "routing_source_elevation_m", "phase",
                             "storm"})
_UNCHANGED = "last accepted boundary unchanged; no next event; no resume of this event"
_EXTERNAL = ("the runner published nothing and repaired nothing; the change was made OUTSIDE the runner, so the "
             "reported accepted identity is the live (externally changed) one; no next event")


class DenseRunnerError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class _Stop(Exception):
    """Internal: stop the current event before publication."""

    def __init__(self, status: str, code: str, detail: str, *, exception: BaseException | None = None,
                 elapsed: str | None = None, evidence: Any = None, guarantee: str = _UNCHANGED) -> None:
        super().__init__(code)
        self.status, self.code, self.detail = status, code, detail
        self.exception, self.elapsed, self.evidence = exception, elapsed, evidence
        self.guarantee = guarantee


def _native() -> SimpleNamespace:
    """Actual native symbols, imported lazily (exact paths of the frozen sources)."""
    import numpy as np
    from maple.aeolian.cohorts.dispatch import mobile_state_is_genuinely_empty, mobile_mass_by_class_kg
    from maple.aeolian.scheduler.accounting import accounting_step_durations_s
    from maple.aeolian.scheduler.block import reservoir_arithmetic_scale_kg
    from maple.aeolian.scheduler.state import SchedulerPhysicsConfig
    from maple.aeolian.scheduler.finalization import finalize_wind_event, WindEventFinalizationResult
    from maple.aeolian.scheduler.record import _EPS_S as record_end_epsilon_s
    from maple.core.parameters.wind_forcing import WindBlock, resolve_wind_forcing_record
    from maple.core.types.cohort import MobileCohortState
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple.surface.voxels._numerics import summation_error_bound_kg
    from maple_syrup.complete_event import (
        CompletionPolicy,
        DryHandoff,
        EventProgress,
        IncompleteEventError,
    )
    from maple_syrup.rainfall import RainfallField, RainfallProvenance, RainfallSchedule
    from maple_syrup.sediment_bed import (
        BedState,
        TerrainReference,
        bed_inventory,
        commit_bed,
        routing_graph_from_elevation,
    )
    from maple_syrup.sediment_event import SedimentEventControl, initial_event_state, sediment_hydrograph_columns
    from maple_syrup.sediment_physics import physics_grid_from_graph
    from maple_syrup.sediment_transport import transport_network
    from maple_syrup.storm import HYDROGRAPH_COLUMNS, initial_state

    return SimpleNamespace(**{name: value for name, value in locals().items()})


# --- native delegates (the only names negative tests may patch) ----------------------------------------------------

def _advance_wind_block(state: Any, block: Any, config: Any) -> Any:
    from maple.aeolian.scheduler.block import advance_wind_block

    return advance_wind_block(state, block, config)


def _finalize_wind_event(state: Any, config: Any, record: Any, *, record_origin_s: float) -> Any:
    from maple.aeolian.scheduler.finalization import finalize_wind_event

    return finalize_wind_event(state, config, record, record_origin_s=record_origin_s)


def _complete_event(state: Any, context: Any, column: Any, field: Any, schedule: Any, vegetation: Any,
                    sediment: Any, **kwargs: Any) -> Any:
    from maple_syrup.complete_event import complete_event

    return complete_event(state, context, column, field, schedule, vegetation, sediment, **kwargs)


# --- typed inputs ----------------------------------------------------------------------------------------------------

def _plain_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise DenseRunnerError("invalid_limit", f"{name} must be a plain int >= 1, got {value!r}")
    return value


@dataclasses.dataclass(frozen=True)
class DenseRunnerLimits:
    """Resource policy for the tiny CPU profile; checked BEFORE each native call. Never physics."""

    max_events: int
    max_wind_accounting_steps: int
    max_storm_steps: int
    max_report_rows: int
    max_work_bytes: int
    max_receipt_bytes: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _plain_int(getattr(self, field.name), field.name)

    def as_record(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DenseWindInputs:
    """`config`: the native `SchedulerPhysicsConfig` sharing the bed context's geometry, classes, mass
    resolution, availability policy and specs. `initial_mobile_state`: a genuinely empty flat native pool."""

    config: Any
    initial_mobile_state: Any


@dataclasses.dataclass(frozen=True)
class DenseWaterInputs:
    """Native SYRUP water inputs. `terrain` is the `TerrainReference` bound to the initial bed (static);
    the first routing graph is built from it natively at the first rain entry. `vegetation_cover_fraction`
    must be bitwise the bed's canonical cover FRACTION (SYRUP converts to percent itself).
    `initial_soil_water_m` is the retained soil water at the first rain entry (owned copy)."""

    terrain: Any
    column: Any
    field: Any
    vegetation_cover_fraction: Any
    sediment: Any
    initial_soil_water_m: Any


@dataclasses.dataclass(frozen=True)
class DenseRunnerControl:
    """`sediment_control`: native `SedimentEventControl`; `completion_policy`: native `CompletionPolicy`;
    `report_every_s`: native reporting cadence."""

    sediment_control: Any
    completion_policy: Any
    report_every_s: float


@dataclasses.dataclass(frozen=True)
class AcceptedBoundary:
    next_index: int
    time_s: float
    bed_identity_sha256: str
    n_accepted_events: int
    routing_held: bool
    routing_source_sha256: str | None

    def as_record(self) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        record["time_s"] = encode_time(self.time_s)
        return record


# --- canonical static inputs -----------------------------------------------------------------------------------------

def _canonical_input(np: Any, value: Any, where: str) -> Any:
    """Closed canonical form of a controlling input. The array namespace must be the actual numpy module
    (recorded by name and version, never by repr); every other leaf goes through the bed's fail-closed
    canonicalizer (unknown types refuse)."""
    if isinstance(value, ModuleType):
        if value is not np:
            raise DenseRunnerError("unsupported_backend", f"{where} is namespace {value.__name__!r}; only numpy")
        return {"array_namespace": "numpy", "numpy_version": np.__version__}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {f.name: _canonical_input(np, getattr(value, f.name), f"{where}.{f.name}")
                           for f in dataclasses.fields(value)}}
    if isinstance(value, (tuple, list)):
        return {"sequence": type(value).__name__,
                "items": [_canonical_input(np, item, f"{where}[{i}]") for i, item in enumerate(value)]}
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            if type(key) is not str:
                raise DenseRunnerError("unhashable_input", f"{where} has a non-string key {key!r}")
            items.append([key, _canonical_input(np, item, f"{where}[{key!r}]")])
        return {"mapping": sorted(items, key=lambda pair: pair[0])}
    try:
        return _bed._canonical_static(value, where)
    except DenseBedError as exc:
        raise DenseRunnerError("unhashable_input", exc.detail) from exc


def _vec(np: Any, value: Any) -> Any:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _floats(np: Any, value: Any) -> list[float]:
    return [float(v) for v in _vec(np, value)]


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _elapsed(start: float, end: Any) -> str:
    return _fraction_text(Fraction(float(end)) - Fraction(start))


def _detail(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:_DETAIL_CHARS]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


# --- the runner ------------------------------------------------------------------------------------------------------

class DenseRunner:
    """Executes the schedule's ordered events on ONE `DenseBed`. `step()` runs one event transactionally and
    returns its receipt (accepted, or the stop receipt); `run()` steps until the schedule is done or stops."""

    def __init__(self, bed: DenseBed, schedule: DenseSchedule, *, wind: DenseWindInputs, water: DenseWaterInputs,
                 control: DenseRunnerControl, limits: DenseRunnerLimits, wind_profile: Any = None,
                 production_limits: Any = None) -> None:
        if type(bed) is not DenseBed:
            raise DenseRunnerError("invalid_input", "bed must be a DenseBed")
        if type(schedule) is not DenseSchedule:
            raise DenseRunnerError("invalid_input", "schedule must be a DenseSchedule")
        for value, expected in ((wind, DenseWindInputs), (water, DenseWaterInputs), (control, DenseRunnerControl),
                                (limits, DenseRunnerLimits)):
            if type(value) is not expected:
                raise DenseRunnerError("invalid_input", f"expected {expected.__name__}, got {type(value).__name__}")
        if len(schedule.events) > limits.max_events:
            raise DenseRunnerError("event_limit", f"{len(schedule.events)} events > max_events={limits.max_events}")
        # The default path imports no production/resource module and retains its original preflight.
        self._production = None
        if wind_profile is None:
            if production_limits is not None:
                raise DenseRunnerError("invalid_input", "production_limits requires an explicit wind_profile")
        else:
            from masy import dense_production
            try:
                process = dense_production.check_process(production_limits, checkpoint="construction")
            except dense_production.DenseProductionError as exc:
                raise DenseRunnerError(exc.code, exc.detail) from exc
            self._production = dense_production
            self._wind_profile, self._production_limits = wind_profile, production_limits
            self._worker_identity = process["process_identity"]
        n = _native()
        np = n.np
        try:
            bed.verify_case_inputs()
        except DenseBedError as exc:
            raise DenseRunnerError(exc.code, exc.detail) from exc
        if "record_origin_s" not in inspect.signature(n.finalize_wind_event).parameters:
            raise DenseRunnerError("unsupported_native_finalizer", "MAPLE must support explicit record_origin_s")
        if self._production is None:
            self._check_wind(n, bed, wind)
        else:
            try:
                self._production.controlling_record(n, bed, wind, schedule, wind_profile, production_limits)
            except self._production.DenseProductionError as exc:
                raise DenseRunnerError(exc.code, exc.detail) from exc
        soil = self._check_water(n, bed, water, control, limits)

        self._bed, self._schedule = bed, schedule
        self._wind, self._water, self._control, self._limits = wind, water, control, limits
        self._static_parts = self._static_parts_record(n)
        self._static_parts_sha256 = _bed._sha256(self._static_parts)
        self._static_record = {
            "schema": STATIC_SCHEMA,
            "profile": (dict(PROFILE) if self._production is None else
                        {**PROFILE, "wind_detachment": "native_reduced", "avalanching": "native_configured",
                         "scope": self._production.SCOPE, "calibration": "not_qualified_by_interface_tests"}),
            "case_sha256": bed.case_sha256,
            "schedule_sha256": schedule.digest,
            "case_schedule_sha256": case_schedule_digest(bed.case_sha256, schedule),
            "controlling_inputs_sha256": self._static_parts_sha256,
            "limits": limits.as_record(),
            # Initial values of evolving process-private state: recorded, never re-verified as static.
            "initial_evolving": {
                "mobile_state_sha256": _bed._sha256(_canonical_input(np, wind.initial_mobile_state, "mobile")),
                "soil_water_m": _bed._array_digest(soil),
            },
        }
        self._static_sha256 = _bed._sha256(self._static_record)

        g = bed.context.geometry
        self._n_classes = len(bed.context.grain_classes.classes)
        self._inventory_elements = g.ny * g.nx * (bed.case_record["n_voxels"] + 1)
        self._inventory0 = _vec(np, n.bed_inventory(bed.state)).copy()
        zeros = np.zeros(self._n_classes, dtype=np.float64)
        # accepted boundary (replaced only by publication)
        self._pool = wind.initial_mobile_state
        self._soil = soil
        self._routing: SimpleNamespace | None = None
        self._ledger = ClockLedger(bed.time_s)
        self._next = 0
        self._n_accepted = 0
        self._totals = {"export": zeros.copy(), "input": zeros.copy(), "bound": zeros.copy(),
                        "internal_terminal_deposition": zeros.copy(),
                        "water": {"rain_m3": 0.0, "drainage_m3": 0.0, "storm_export_m3": 0.0,
                                  "dry_surface_removed_m3": 0.0, "dry_soil_removed_m3": 0.0},
                        "n_entry_reroutes": 0, "n_event_reroutes": 0}
        self._receipts: list[dict[str, Any]] = []
        self._stop_receipt: dict[str, Any] | None = None
        self._active_evidence: Any = None
        # Accepted identities, re-verified before every event's first work copy / native call.
        self._accepted_bed = bed.identity_sha256()
        self._accepted_private = self._private_identity(np, self._pool, self._soil)

    # preflight -------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_wind(n: SimpleNamespace, bed: DenseBed, wind: DenseWindInputs) -> None:
        np, ctx, cfg = n.np, bed.context, wind.config
        if type(cfg) is not n.SchedulerPhysicsConfig:
            raise DenseRunnerError("invalid_wind_config", "wind.config must be a native SchedulerPhysicsConfig")
        shared = {"geometry": cfg.geometry is ctx.geometry, "grain_classes": cfg.grain_classes is ctx.grain_classes,
                  "avalanching_spec": cfg.avalanching_spec is ctx.avalanching_spec,
                  "topographic_commit_spec": cfg.topographic_commit_spec is ctx.topographic_commit_spec}
        wrong = sorted(name for name, same in shared.items() if not same)
        if wrong:
            raise DenseRunnerError("incompatible_wind_config", f"wind config {wrong} are not the bed context objects")
        if float(cfg.mass_resolution_kg).hex() != float(ctx.mass_resolution_kg).hex():
            raise DenseRunnerError("incompatible_wind_config", "wind mass_resolution_kg differs from the bed")
        fraction, held = cfg.initial_available_fraction, ctx.initial_available_fraction
        if (type(fraction) is not np.ndarray or fraction.dtype != held.dtype or fraction.shape != held.shape
                or not bool(np.array_equal(fraction, held))):
            raise DenseRunnerError("incompatible_wind_config",
                                   "wind initial_available_fraction must equal the bed availability policy bitwise")
        if bool(getattr(ctx.avalanching_spec, "enabled", True)):
            raise DenseRunnerError("unsupported_avalanching", "only explicitly disabled avalanching is supported")
        if cfg.aeolian_process_mode.mode != SUPPORTED_WIND_MODE:
            raise DenseRunnerError("unsupported_wind_mode", f"mode {cfg.aeolian_process_mode.mode!r}")
        if bool(cfg.water_coupling_spec.enabled):
            raise DenseRunnerError("unsupported_wind_mode", "a MAPLE water pass during wind is not supported")
        if bool(cfg.topographic_wind_spec.enabled):
            raise DenseRunnerError("unsupported_wind_mode", "topographic wind is not in the supported profile")
        set_fields = sorted(name for name in _UNSUPPORTED_WIND_FIELDS if getattr(cfg, name, None) is not None)
        if set_fields:
            raise DenseRunnerError("unsupported_wind_mode", f"alternate wind mode field(s) {set_fields} are set")
        outer = cfg.outer_timestep_s
        if type(outer) is not float or not math.isfinite(outer) or outer <= 0:
            raise DenseRunnerError("invalid_wind_config", f"outer_timestep_s {outer!r} must be a finite float > 0")
        pool = wind.initial_mobile_state
        if type(pool) is not n.MobileCohortState:
            raise DenseRunnerError("unsupported_mobile_state", "only a flat native MobileCohortState is supported")
        try:
            _bed._require_host_arrays(pool, "initial_mobile_state")
        except DenseBedError as exc:
            raise DenseRunnerError(exc.code, exc.detail) from exc
        if not n.mobile_state_is_genuinely_empty(pool):
            raise DenseRunnerError("mobile_state_not_empty", "the initial wind pool must be genuinely empty")

    @staticmethod
    def _check_water(n: SimpleNamespace, bed: DenseBed, water: DenseWaterInputs, control: DenseRunnerControl,
                     limits: DenseRunnerLimits):
        np, g = n.np, bed.context.geometry
        sc, policy = control.sediment_control, control.completion_policy
        if type(sc) is not n.SedimentEventControl:
            raise DenseRunnerError("invalid_water_control", "sediment_control must be a native SedimentEventControl")
        if sc.transport_scheme != "upwind":
            raise DenseRunnerError("unsupported_transport_scheme", f"{sc.transport_scheme!r}; only 'upwind'")
        if sc.storm.implementation != "array":
            raise DenseRunnerError("unsupported_hydraulic_implementation", f"{sc.storm.implementation!r}; only 'array'")
        if sc.commit is not True or sc.record_exchange_history is not False or sc.deep_publication != "step":
            raise DenseRunnerError("unsupported_water_control",
                                   "requires commit=True, record_exchange_history=False, deep_publication='step'")
        if type(policy) is not n.CompletionPolicy:
            raise DenseRunnerError("invalid_water_control", "completion_policy must be a native CompletionPolicy")
        try:
            sc.validated()
            policy.validated()
        except ValueError as exc:
            raise DenseRunnerError("invalid_water_control", _detail(exc)) from exc
        if sc.storm.max_steps > limits.max_storm_steps:
            raise DenseRunnerError("storm_step_limit", f"native max_steps {sc.storm.max_steps} > max_storm_steps="
                                   f"{limits.max_storm_steps}")
        every = control.report_every_s
        if type(every) is not float or not math.isfinite(every) or every <= 0:
            raise DenseRunnerError("invalid_water_control", f"report_every_s {every!r} must be a finite float > 0")
        if type(water.terrain) is not n.TerrainReference:
            raise DenseRunnerError("invalid_water_inputs", "terrain must be a native TerrainReference")
        if water.terrain.dx_m != g.dx_m or water.terrain.initial_committed_elevation_m.shape != (g.ny, g.nx):
            raise DenseRunnerError("incompatible_water_inputs", "terrain does not match the bed geometry")
        if type(water.field) is not n.RainfallField or water.field.xp is not np or water.field.shape != (g.ny, g.nx):
            raise DenseRunnerError("incompatible_water_inputs", "field must be a NumPy RainfallField on the bed grid")
        veg, cover = water.vegetation_cover_fraction, bed.vegetation_cover_fraction()
        if (type(veg) is not np.ndarray or veg.dtype != cover.dtype or veg.shape != cover.shape
                or not bool(np.array_equal(veg, cover))):
            raise DenseRunnerError("vegetation_mismatch", "water vegetation must be bitwise the bed's canonical "
                                   "prescribed cover fraction (no percent conversion, no second vegetation)")
        soil = water.initial_soil_water_m
        if (type(soil) is not np.ndarray or soil.dtype != np.float64 or soil.shape != (g.ny, g.nx)
                or not bool(np.isfinite(soil).all()) or bool((soil < 0).any()) or bool(np.signbit(soil).any())):
            raise DenseRunnerError("invalid_water_inputs", "initial_soil_water_m must be finite (ny, nx) float64 >= +0")
        return np.array(soil, dtype=np.float64, copy=True)

    def _static_parts_record(self, n: SimpleNamespace) -> dict[str, Any]:
        """Every controlling static input (recomputed live before each publication)."""
        np, w, c = n.np, self._water, self._control
        record = {
            "wind_config": _canonical_input(np, self._wind.config, "wind.config"),
            "terrain": _canonical_input(np, w.terrain, "water.terrain"),
            "column": _canonical_input(np, w.column, "water.column"),
            "field": _canonical_input(np, w.field, "water.field"),
            "vegetation_cover_fraction": _canonical_input(np, w.vegetation_cover_fraction, "water.vegetation"),
            "sediment": _canonical_input(np, w.sediment, "water.sediment"),
            "sediment_control": _canonical_input(np, c.sediment_control, "control.sediment_control"),
            "completion_policy": _canonical_input(np, c.completion_policy, "control.completion_policy"),
            "report_every_s": encode_time(c.report_every_s),
            "schedule": self._schedule.as_record(),
        }
        if self._production is not None:
            try:
                record["production"] = self._production.controlling_record(
                    n, self._bed, self._wind, self._schedule, self._wind_profile, self._production_limits)
            except self._production.DenseProductionError as exc:
                raise DenseRunnerError(exc.code, exc.detail) from exc
        return record

    @staticmethod
    def _private_identity(np: Any, pool: Any, soil: Any) -> str:
        """Digest of the exposed accepted process-private arrays (wind pool host arrays, soil water)."""
        if not dataclasses.is_dataclass(pool):
            arrays = _canonical_input(np, pool, "mobile_state")
        else:
            arrays = {f.name: _bed._array_digest(getattr(pool, f.name)) for f in dataclasses.fields(pool)
                      if type(getattr(pool, f.name)) is np.ndarray}
        return _bed._sha256({"mobile_state": arrays, "soil_water_m": _bed._array_digest(soil)})

    def _precheck(self) -> None:
        """Before any event work copy or native transport call: case inputs, every controlling static
        input and the accepted bed / pool / soil must still be what the runner accepted. A difference can only
        come from outside the runner; it is refused as such, and nothing is repaired or replaced.
        Opt-in native configuration audits may allocate small records before work staging."""
        self._production_process_check()
        n = _native()
        try:
            self._bed.verify_case_inputs()
            live = _bed._sha256(self._static_parts_record(n))
        except (DenseBedError, DenseRunnerError) as exc:
            raise _Stop("refused", "case_input_mutated_before_event", exc.detail, exception=exc,
                        guarantee=_EXTERNAL) from exc
        if live != self._static_parts_sha256:
            raise _Stop("refused", "static_input_mutated_before_event", "a controlling runner input changed after "
                        "construction; no native call was made", guarantee=_EXTERNAL)
        if (self._bed.identity_sha256() != self._accepted_bed
                or self._private_identity(n.np, self._pool, self._soil) != self._accepted_private):
            raise _Stop("refused", "accepted_state_externally_mutated", "the accepted bed, wind pool or soil water "
                        "was changed outside the runner since its acceptance", guarantee=_EXTERNAL)

    # views -----------------------------------------------------------------------------------------------------
    @property
    def static_record(self) -> dict[str, Any]:
        return copy.deepcopy(self._static_record)

    @property
    def static_sha256(self) -> str:
        return self._static_sha256

    @property
    def accepted(self) -> AcceptedBoundary:
        source = None if self._routing is None else _bed._sha256(_bed._array_digest(self._routing.source))
        return AcceptedBoundary(self._next, self._bed.time_s, self._bed.identity_sha256(), self._n_accepted,
                                self._routing is not None, source)

    @property
    def receipts(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._receipts)

    @property
    def stop_receipt(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._stop_receipt)

    @property
    def active_evidence(self) -> Any:
        """Latest unaccepted native output of a stopped event (never adopted, never resumable here)."""
        return self._active_evidence

    @property
    def mobile_state(self) -> Any:
        return self._pool

    @property
    def soil_water_m(self) -> Any:
        return self._soil

    def ledger_record(self) -> dict[str, Any]:
        return self._ledger.as_record()

    def report(self) -> dict[str, Any]:
        n = _native()
        done = self._next == len(self._schedule.events)
        return {"schema": REPORT_SCHEMA, "static_sha256": self._static_sha256,
                "status": "stopped" if self._stop_receipt else ("complete" if done else "in_progress"),
                "accepted": self.accepted.as_record(), "clock_ledger": self._ledger.as_record(),
                "receipts": self.receipts, "stop": self.stop_receipt,
                "aggregate": self._aggregate_record(n.np, self._totals, n.bed_inventory(self._bed.state))}

    # stepping --------------------------------------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        while self._stop_receipt is None and self._next < len(self._schedule.events):
            self.step()
        return self.report()

    def step(self) -> dict[str, Any]:
        if self._stop_receipt is not None:
            raise DenseRunnerError("runner_halted", f"stopped at event {self._stop_receipt['index']} "
                                   f"({self._stop_receipt['code']}); no further event runs")
        if self._next >= len(self._schedule.events):
            raise DenseRunnerError("schedule_complete", "every event has been accepted")
        index = self._next
        event = self._schedule.events[index]
        start = self._bed.time_s
        try:
            self._precheck()
            try:
                resolved = resolve_event(self._schedule, index, start)
            except (ClockError, DenseScheduleError) as exc:
                raise _Stop("refused", "unresolvable_event", _detail(exc), exception=exc) from exc
            if type(event) is WindEvent:
                return self._wind_event(index, event, resolved)
            return self._rain_event(index, event, resolved)
        except _Stop as stop:
            if self._production is not None and isinstance(stop.exception, MemoryError):
                stop = _Stop("runner_error", "address_space_exhausted", _detail(stop.exception),
                             exception=stop.exception, elapsed=stop.elapsed, evidence=stop.evidence,
                             guarantee=stop.guarantee)
            return self._stop(index, event.kind, start, stop)
        except Exception as exc:  # noqa: BLE001 - publication is the last fallible step; stop unpublished
            code = ("address_space_exhausted" if self._production is not None and isinstance(exc, MemoryError)
                    else "runner_internal_error")
            return self._stop(index, event.kind, start,
                              _Stop("runner_error", code, _detail(exc), exception=exc))

    def _stop(self, index: int, kind: str, start: float, stop: _Stop) -> dict[str, Any]:
        receipt = {"schema": RECEIPT_SCHEMA, "index": index, "kind": kind, "status": stop.status, "code": stop.code,
                   "detail": stop.detail[:_DETAIL_CHARS],
                   "exception_type": None if stop.exception is None else type(stop.exception).__name__,
                   "native_start": encode_time(start), "unaccepted_elapsed_s": stop.elapsed,
                   "accepted": self.accepted.as_record(),
                   "guarantee": stop.guarantee}
        self._stop_receipt, self._active_evidence = receipt, stop.evidence
        return copy.deepcopy(receipt)

    def _production_process_check(self, upcoming_wrapper_bytes: int = 0, *, checkpoint: str = "event_entry",
                                  elapsed: str | None = None, evidence: Any = None) -> Any:
        if self._production is None:
            return None
        try:
            return self._production.check_process(self._production_limits, self._worker_identity,
                                                  upcoming_wrapper_bytes=upcoming_wrapper_bytes, checkpoint=checkpoint)
        except self._production.DenseProductionError as exc:
            raise _Stop("refused", exc.code, exc.detail, exception=exc, elapsed=elapsed, evidence=evidence) from exc
        except MemoryError as exc:
            raise _Stop("runner_error", "address_space_exhausted", _detail(exc), exception=exc,
                        elapsed=elapsed, evidence=evidence) from exc

    def _work_bytes_check(self, estimate: int) -> None:
        if estimate > self._limits.max_work_bytes:
            raise _Stop("refused", "work_bytes_cap", f"estimated work copies {estimate} bytes > max_work_bytes="
                        f"{self._limits.max_work_bytes}")
        self._production_process_check(estimate, checkpoint="wrapper_work")

    def _stage(self) -> Any:
        try:
            return self._bed.stage_copy()
        except DenseBedError as exc:
            raise _Stop("refused", exc.code, exc.detail, exception=exc) from exc

    # wind ------------------------------------------------------------------------------------------------------
    def _wind_event(self, index: int, event: WindEvent, resolved: dict[str, Any]) -> dict[str, Any]:
        n = _native()
        np, bed, cfg = n.np, self._bed, self._wind.config
        t = bed.time_s
        end = decode_time(resolved["resolved_edges"]["edges"][-1]["native"])
        # Theoretical native accounting work (exact upper bound, no step list): ceil(duration / outer) + 1.
        quotient = Fraction(event.duration_s) / Fraction(cfg.outer_timestep_s)
        upper = -(-quotient.numerator // quotient.denominator) + 1
        if upper > self._limits.max_wind_accounting_steps:
            raise _Stop("refused", "wind_step_limit", f"up to {upper} native accounting steps > "
                        f"max_wind_accounting_steps={self._limits.max_wind_accounting_steps}")
        stage_bytes = _bed._tree_nbytes(*(getattr(bed.state, name) for name in SHARED_FIELDS))
        pool_bytes = _bed._tree_nbytes(self._pool)
        # Bound staged inputs, transport/finalizer outputs, empty replacement pool,
        # native deposition request and the native accounting plan. Dense CPU only.
        self._work_bytes_check(4 * stage_bytes + 3 * pool_bytes
                               + 8 * self._inventory_elements * self._n_classes + 32 * upper)
        # The native baseline splits the block with `accounting_step_durations_s` (a list of at most `upper`
        # floats, guarded above). Its exact sum must cover the requested duration up to the rounding of the
        # split's own two operations (n * outer and duration - n * outer, unit roundoff 2**-53 each). A dropped
        # or extended interval is an unsupported duration: refused before any model work, time never rewritten.
        plan = n.accounting_step_durations_s(event.duration_s, cfg.outer_timestep_s)
        requested = Fraction(event.duration_s)
        covered = sum((Fraction(dt) for dt in plan), Fraction(0))
        rounding = Fraction(2, 2 ** 53) * max(requested, covered)
        if len(plan) > upper or covered <= 0 or abs(covered - requested) > rounding:
            raise _Stop("refused", "unsupported_wind_duration",
                        f"native accounting steps ({len(plan)}) cover {float(covered)!r} s of the requested "
                        f"{event.duration_s!r} s at outer_timestep_s={cfg.outer_timestep_s!r}; the difference "
                        "exceeds the split's own FP rounding (the native split drops or extends scheduled time)")


        if self._production is not None:
            try:
                self._production.require_dry_wind_water(np, bed.state.water)
            except self._production.DenseProductionError as exc:
                raise _Stop("refused", exc.code, exc.detail, exception=exc) from exc
        stage = self._stage()
        pool = copy.deepcopy(self._pool)
        try:
            scheduler = bed.stage_scheduler_state(stage, pool, time_s=t)
        except DenseBedError as exc:
            raise _Stop("refused", exc.code, exc.detail, exception=exc) from exc
        block = n.WindBlock(shear_velocity_m_s=event.shear_velocity_m_s,
                            direction_deg=event.direction_deg_cw_from_north, duration_s=event.duration_s,
                            start_time_s=t, end_time_s=end)
        inventory_before = _vec(np, n.bed_inventory(stage.state))
        try:
            result = _advance_wind_block(scheduler, block, cfg)
        except Exception as exc:  # noqa: BLE001 - every native failure stops the event unpublished
            status = "native_refused" if isinstance(exc, ValueError) else "native_error"
            raise _Stop(status, "wind_native_failure", _detail(exc), exception=exc) from exc
        physical, a = result.state.physical, result.accounting
        t_end = float(physical.time_s)
        elapsed = _elapsed(t, t_end)

        def incomplete(code: str, detail: str) -> _Stop:
            return _Stop("incomplete", code, detail, elapsed=elapsed, evidence=result)

        if result.process_mode != SUPPORTED_WIND_MODE or result.integration_path != "baseline_direct":
            raise incomplete("unsupported_wind_path", f"{result.process_mode}/{result.integration_path}")
        if result.conservation_ok is not True:
            raise incomplete("wind_conservation_failed", "native check_block_conservation reported failure")
        water_fields = [getattr(a, f"water_{name}", None) for name in
                        ("mobile_mass_before_by_class_kg", "mobile_mass_after_by_class_kg",
                         "boundary_input_by_class_kg", "boundary_export_by_class_kg")]
        if any(v is not None and bool(np.any(_vec(np, v) != 0.0)) for v in water_fields):
            raise incomplete("wind_water_pass", "native wind accounting reports water-borne exchange")
        if a.n_accounting_steps > upper or a.n_accounting_steps > self._limits.max_wind_accounting_steps:
            raise incomplete("wind_step_limit", f"{a.n_accounting_steps} native accounting steps")
        if bool(np.any(physical.water.mobile_mass_by_cell_class_kg != 0.0)):
            raise incomplete("water_mobile_not_empty", "water-borne load appeared during wind")
        if t_end < t:
            raise incomplete("time_reversed", "native wind end precedes its start")
        if t_end == t:
            raise incomplete("wind_no_elapsed", "positive requested duration but native physical time did not advance")

        bed_change = _vec(np, a.bed_change_by_class_kg)
        mobile_delta = _vec(np, a.mobile_mass_after_by_class_kg) - _vec(np, a.mobile_mass_before_by_class_kg)
        inflow, export = _vec(np, a.boundary_input_by_class_kg), _vec(np, a.boundary_export_by_class_kg)
        native_residual = bed_change + mobile_delta - inflow + export
        native_bound = n.summation_error_bound_kg(max(a.n_accounting_steps, 1),
                                                  n.reservoir_arithmetic_scale_kg(a, physical))
        if not bool(np.all(np.abs(native_residual) <= native_bound)):
            raise incomplete("wind_budget_open", "native per-class reservoir identity exceeds its native bound")
        # advance_wind_block leaves the consumed block cursor in place. Apply
        # exactly the native record loop's terminal cursor mapping, at this
        # deliberate event boundary only. No native physical timestamp changes.
        if (result.state.block_index != 0
                or result.state.block_offset_s < event.duration_s - n.record_end_epsilon_s):
            raise incomplete("wind_record_incomplete", "native block did not reach its terminal cursor")
        record = n.resolve_wind_forcing_record({"blocks": [{
            "shear_velocity_m_s": event.shear_velocity_m_s,
            "direction_deg": event.direction_deg_cw_from_north, "duration_s": event.duration_s}]})
        completed = dataclasses.replace(result.state, block_index=len(record.blocks), block_offset_s=0.0)
        evidence = {"transport": result}
        try:
            final = _finalize_wind_event(completed, cfg, record, record_origin_s=t)
        except Exception as exc:  # native errors never publish this event or admit water
            raise _Stop("native_error", "wind_finalization_failure", _detail(exc), exception=exc,
                        elapsed=elapsed, evidence=evidence) from exc
        evidence["finalization"] = final
        if type(final) is not n.WindEventFinalizationResult:
            raise _Stop("incomplete", "invalid_wind_finalization", "unexpected native finalizer result type",
                        elapsed=elapsed, evidence=evidence)
        if not final.completed or not final.water_handoff_ready:
            raise _Stop("incomplete", "wind_finalization_incomplete", str(final.reasons)[:_DETAIL_CHARS],
                        elapsed=elapsed, evidence=evidence)
        settled = final.state
        if (float(settled.physical.time_s).hex() != t_end.hex()
                or float(final.time_s).hex() != t_end.hex()
                or settled.block_index != completed.block_index or settled.block_offset_s != 0.0
                or final.block_index != completed.block_index or final.block_offset_s != 0.0):
            raise _Stop("incomplete", "wind_finalization_clock_changed", "native finalizer changed time or cursor",
                        elapsed=elapsed, evidence=evidence)
        if not n.mobile_state_is_genuinely_empty(settled.physical.mobile_state):
            raise _Stop("incomplete", "wind_finalization_mobile_nonempty", "native finalizer left airborne material",
                        elapsed=elapsed, evidence=evidence)
        # Finalizer deposition is INTERNAL: its native returned bed/ledger already
        # includes it. Only its native boundary export is added to external export.
        final_vectors = {name: _vec(np, getattr(final, name)) for name in (
            "airborne_mass_before_by_class_kg", "airborne_mass_after_by_class_kg",
            "deposited_mass_by_class_kg", "boundary_export_mass_by_class_kg", "bed_change_by_class_kg",
            "transfer_identity_gap_by_class_kg", "reservoir_identity_gap_by_class_kg")}
        if any(v.shape != (self._n_classes,) or not bool(np.isfinite(v).all()) for v in final_vectors.values()):
            raise _Stop("incomplete", "wind_finalization_budget_invalid", "nonfinite or incompatible class vector",
                        elapsed=elapsed, evidence=evidence)
        for prefix in ("transfer", "reservoir"):
            bound = float(getattr(final, prefix + "_identity_bound_kg"))
            if (not math.isfinite(bound) or bound < 0.0
                    or not bool(np.all(np.abs(final_vectors[prefix + "_identity_gap_by_class_kg"]) <= bound))):
                raise _Stop("incomplete", "wind_finalization_budget_open", "native finalizer class budget failed",
                            elapsed=elapsed, evidence=evidence)
        if (bool(np.any(final_vectors["airborne_mass_after_by_class_kg"] != 0.0))
                or not bool(np.array_equal(final_vectors["airborne_mass_before_by_class_kg"],
                                          n.mobile_mass_by_class_kg(physical.mobile_state, self._n_classes)))):
            raise _Stop("incomplete", "wind_finalization_mobile_budget", "native mobile inventory disagrees",
                        elapsed=elapsed, evidence=evidence)
        transport_export = export
        export = transport_export + final_vectors["boundary_export_mass_by_class_kg"]
        transport_bed_change = bed_change
        bed_change = bed_change + final_vectors["bed_change_by_class_kg"]
        physical = settled.physical
        candidate = n.BedState(physical.voxel_column, physical.active_layer, physical.water, physical.ledger,
                               physical.committed_topography, physical.sediment_availability)
        inventory_after = _vec(np, n.bed_inventory(candidate))
        event_mobile_delta = final_vectors["airborne_mass_after_by_class_kg"] - _vec(np, a.mobile_mass_before_by_class_kg)
        event_residual = bed_change + event_mobile_delta - inflow + export
        # Use the original operands, so cancellation of erosion and settlement
        # cannot shrink the arithmetic scale of the six combining operations.
        event_scale = sum(float(np.abs(value).sum()) for value in (
            transport_bed_change, final_vectors["bed_change_by_class_kg"],
            final_vectors["airborne_mass_after_by_class_kg"], _vec(np, a.mobile_mass_before_by_class_kg),
            inflow, transport_export, final_vectors["boundary_export_mass_by_class_kg"]))
        event_bound = (native_bound + float(final.reservoir_identity_bound_kg)
                       + n.summation_error_bound_kg(6, event_scale))
        if not bool(np.all(np.abs(event_residual) <= event_bound)):
            raise _Stop("incomplete", "wind_event_budget_open", "combined transport/finalizer class identity failed",
                        elapsed=elapsed, evidence=evidence)
        inventory_residual = inventory_after - inventory_before + event_mobile_delta - inflow + export
        scale = max(float(np.abs(inventory_before).sum()), float(np.abs(inventory_after).sum()),
                    float(inflow.sum()), float(export.sum()))
        inventory_bound = (native_bound + float(final.reservoir_identity_bound_kg)
                           + n.summation_error_bound_kg(2 * self._inventory_elements + 6, scale))
        if not bool(np.all(np.abs(inventory_residual) <= inventory_bound)):
            raise _Stop("incomplete", "wind_inventory_open", "actual bed inventories do not close within native bounds",
                        elapsed=elapsed, evidence=evidence)

        spatial = getattr(a, "spatial", None)
        body = {
            "native": {
                "block": {"start_time_s": encode_time(float(result.start_time_s)),
                          "end_time_s": encode_time(float(result.end_time_s)),
                          "duration_s": encode_time(float(result.duration_s))},
                "advanced_end_time_s": encode_time(float(result.advanced_end_time_s)),
                "advanced_duration_s": encode_time(float(result.advanced_duration_s)),
                "physical_time_s": encode_time(t_end),
                "process_mode": result.process_mode, "integration_path": result.integration_path,
                "n_accounting_steps": int(a.n_accounting_steps), "accounting_step_upper_bound": upper,
                "n_commits_this_block": int(result.n_commits_this_block),
                "conservation_residual_kg": float(result.conservation_residual_kg),
            },
            # Native FP accumulation is kept as is (endpoints unchanged); only the discrepancy is recorded.
            "coverage": {"requested_s": encode_time(event.duration_s), "native_plan_steps": len(plan),
                         "native_plan_sum_s": _fraction_text(covered),
                         "plan_discrepancy_s": _fraction_text(covered - requested),
                         "plan_rounding_bound_s": _fraction_text(rounding),
                         "elapsed_discrepancy_s": _fraction_text(Fraction(t_end) - Fraction(t) - requested)},
            "classes": {
                "bed_change_kg": _floats(np, bed_change), "boundary_input_kg": _floats(np, inflow),
                "boundary_export_kg": _floats(np, export),
                "mobile_before_kg": _floats(np, a.mobile_mass_before_by_class_kg),
                "mobile_after_kg": _floats(np, final.airborne_mass_after_by_class_kg),
                "transport_mobile_after_kg": _floats(np, a.mobile_mass_after_by_class_kg),
                "transport_boundary_export_kg": _floats(np, transport_export),
                "transport_bed_change_kg": _floats(np, transport_bed_change),
                "detachment_kg": (None if spatial is None
                                  else _floats(np, spatial.detachment_by_cell_class_kg.sum(axis=(0, 1)))),
                "native_residual_kg": _floats(np, native_residual), "native_bound_kg": float(native_bound),
                "event_residual_kg": _floats(np, event_residual), "event_bound_kg": float(event_bound),
                "event_summation_terms": 6, "event_arithmetic_scale_kg": event_scale,
                "inventory_before_kg": _floats(np, inventory_before), "inventory_after_kg": _floats(np, inventory_after),
                "inventory_residual_kg": _floats(np, inventory_residual),
                "inventory_bound_kg": float(inventory_bound),
            },
            "wind_end_pool": "genuinely_empty",
            "finalization": {
                "status": final.status, "algorithm_version": final.algorithm_version,
                "representation": final.representation, "placement_rule": final.placement_rule,
                "record_origin_s": encode_time(t),
                "record": _canonical_input(np, record, "completed_record"),
                "record_sha256": _bed._sha256(_canonical_input(np, record, "completed_record")),
                "completed_block_index": settled.block_index, "completed_block_offset_s": settled.block_offset_s,
                "physical_time_s": encode_time(float(physical.time_s)),
                "n_settled": int(final.n_settled), "n_exported": int(final.n_exported),
                **{name: _floats(np, value) for name, value in final_vectors.items()},
                "transfer_identity_bound_kg": float(final.transfer_identity_bound_kg),
                "reservoir_identity_bound_kg": float(final.reservoir_identity_bound_kg),
                "numerical_residual_scalar_kg": float(final.numerical_residual_scalar_kg),
            },
        }

        if self._production is not None:
            try:
                body["production"] = self._production.wind_evidence(n, result, final, bed.state.water)
                body["production"]["worker"] = self._production_process_check(
                    checkpoint="after_native_wind", elapsed=elapsed, evidence=evidence)
                body["production"]["retained_wrapper_owners"] = self._production.retained_wrapper_owners(
                    bed, self._wind, self._water, self._control, self._wind_profile,
                    self._pool, self._soil, self._routing, len(self._receipts))
            except _Stop:
                raise
            except MemoryError as exc:
                raise _Stop("runner_error", "address_space_exhausted", _detail(exc), exception=exc,
                            elapsed=elapsed, evidence=evidence) from exc
            except self._production.DenseProductionError as exc:
                raise _Stop("incomplete", exc.code, exc.detail, exception=exc,
                            elapsed=elapsed, evidence=evidence) from exc
            except Exception as exc:
                raise _Stop("runner_error", "production_evidence_failure", _detail(exc), exception=exc,
                            elapsed=elapsed, evidence=evidence) from exc

        def publish() -> None:
            bed.adopt_scheduler_physical(physical)

        # The legitimate empty native pool is the next accepted pool; caches are rebuilt at re-entry.
        return self._finish(n, index, "wind", resolved, stage, candidate, t, t_end, float(result.advanced_duration_s),
                            inflow, export, np.full(self._n_classes, inventory_bound), body, publish,
                            pool=physical.mobile_state, soil=self._soil, routing=self._routing,
                            terminal=final_vectors["deposited_mass_by_class_kg"],
                            elapsed=elapsed, evidence=evidence)

    # rain ------------------------------------------------------------------------------------------------------
    def _rain_event(self, index: int, event: Any, resolved: dict[str, Any]) -> dict[str, Any]:
        n = _native()
        np, bed, ctx = n.np, self._bed, self._bed.context
        sc, policy, every = self._control.sediment_control, self._control.completion_policy, self._control.report_every_s
        w, limits = self._water, self._limits
        t = bed.time_s
        edges = [decode_time(edge["native"]) for edge in resolved["resolved_edges"]["edges"]]
        rain_edges, guard_end = edges[:-1], edges[-1]
        intensities = [interval.intensity_mm_h for interval in event.intervals]
        # Preflight (no allocation yet): steps needed to reach the rainfall end plus the native hold at max_dt_s.
        need = (Fraction(rain_edges[-1]) - Fraction(t) + Fraction(policy.hold_s)) / Fraction(sc.storm.max_dt_s)
        lower_steps = -(-need.numerator // need.denominator)
        if lower_steps > sc.storm.max_steps:
            raise _Stop("refused", "storm_step_limit", f"at least {lower_steps} storm steps > native max_steps="
                        f"{sc.storm.max_steps}")
        span = (Fraction(guard_end) - Fraction(t)) / Fraction(every)
        rows = span.numerator // span.denominator + len(rain_edges) + 2
        if rows > limits.max_report_rows:
            raise _Stop("refused", "report_rows_limit", f"up to {rows} report rows > max_report_rows="
                        f"{limits.max_report_rows}")
        g = ctx.geometry
        cells, nc = g.ny * g.nx, self._n_classes
        stage_bytes = _bed._tree_nbytes(*(getattr(bed.state, name) for name in SHARED_FIELDS))
        columns = len(n.HYDROGRAPH_COLUMNS) + len(n.sediment_hydrograph_columns(nc))
        # Retained process-private memory of the accepted template (sediment velocity in particular): every array
        # field the entry does not replace is handed to native code only as a work copy, never by reference.
        retained = self._routing
        memory_names: tuple[str, ...] = ()
        memory_bytes = 0
        if retained is not None:
            memory_names = tuple(f.name for f in dataclasses.fields(retained.template) if f.init
                                 and f.name not in _ENTRY_REPLACED
                                 and type(getattr(retained.template, f.name)) is np.ndarray)
            if "sediment_velocity_m_s" not in memory_names:
                raise _Stop("refused", "transport_memory_unsupported",
                            "retained sediment velocity is not a host NumPy array; it cannot be work-copied")
            memory_bytes = sum(getattr(retained.template, name).nbytes for name in memory_names)
        # stage + committed candidate + dry bed; storm / soil / velocity work grids; two report tables; memory copies.
        self._work_bytes_check(3 * stage_bytes + 8 * cells * (6 + nc) + 2 * 8 * rows * columns + memory_bytes)
        try:
            schedule = n.RainfallSchedule(edges_s=rain_edges, intensity_mm_per_h=intensities,
                                          provenance=n.RainfallProvenance(kind="constant"))
        except ValueError as exc:
            raise _Stop("refused", "invalid_rainfall", _detail(exc), exception=exc) from exc

        stage = self._stage()
        stage_bed = stage.state
        if bool(np.any(stage_bed.water.mobile_mass_by_cell_class_kg != 0.0)):
            raise _Stop("refused", "water_mobile_not_empty", "rain entry needs an empty water-borne load")
        inventory_before = _vec(np, n.bed_inventory(stage_bed))
        soil = np.array(self._soil, copy=True)
        first = self._routing is None
        try:
            if first:  # first-time construction: the native graph of the CURRENT committed elevation
                source = n.committed_elevation_m(stage_bed.committed_topography)
                graph0 = n.routing_graph_from_elevation(w.terrain, source, xp=np)
                seed = n.initial_event_state(graph0, w.terrain, stage_bed, ctx, soil, t_s=t, control=sc)
                held = SimpleNamespace(graph=seed.graph, terrain=seed.terrain, network=seed.network, grid=seed.grid,
                                       source=seed.routing_source_elevation_m, template=seed)
            else:
                held = retained
            if held.template.phase is not None or bool(np.any(held.template.sediment_velocity_m_s != 0.0)):
                raise _Stop("refused", "transport_memory_not_empty", "upwind entry needs phase None and zero memory")
            boundary = n.commit_bed(stage_bed, ctx, held.graph, held.terrain, t, force=True,
                                    routing_source_m=held.source)
        except _Stop:
            raise
        except Exception as exc:  # noqa: BLE001
            status = "native_refused" if isinstance(exc, ValueError) else "native_error"
            raise _Stop(status, "rain_entry_failure", _detail(exc), exception=exc) from exc
        if boundary.did_commit is not True or type(boundary.rerouted) is not bool or boundary.routing_source_m is None:
            raise _Stop("incomplete", "rain_entry_commit", "native entry commit did not return a held routing source")
        graph = boundary.graph
        if boundary.rerouted:
            network, grid = n.transport_network(graph), n.physics_grid_from_graph(graph)
        else:
            if graph is not held.graph or boundary.routing_source_m is not held.source:
                raise _Stop("incomplete", "routing_identity_changed", "no reroute but the routing pair was replaced")
            network, grid = held.network, held.grid
        try:
            # Same values (native reset semantics kept), independent buffers: a native write to the entry's memory
            # can never reach the accepted template.
            memory = {name: np.array(getattr(held.template, name), copy=True) for name in memory_names}
            entry = dataclasses.replace(
                held.template, bed=boundary.state, graph=graph, network=network, grid=grid, terrain=held.terrain,
                routing_source_elevation_m=boundary.routing_source_m, phase=None,
                storm=n.initial_state(graph, boundary.state.water.depth_m, soil, t_s=t), **memory)
            wet = _complete_event(entry, ctx, w.column, w.field, schedule, w.vegetation_cover_fraction, w.sediment,
                                  max_end_s=guard_end, control=sc, policy=policy, report_every_s=every,
                                  max_report_rows=limits.max_report_rows)
        except n.IncompleteEventError as exc:
            raise _Stop("incomplete", "rain_incomplete_native_limit", _detail(exc), exception=exc) from exc
        except Exception as exc:  # noqa: BLE001
            status = "native_refused" if isinstance(exc, ValueError) else "native_error"
            raise _Stop(status, "rain_native_failure", _detail(exc), exception=exc) from exc
        if type(wet) is not n.DryHandoff or wet.status != "complete":
            progress = wet if type(wet) is n.EventProgress else None
            elapsed = None if progress is None else _elapsed(t, progress.result.state.t_s)
            raise _Stop("incomplete", "rain_not_dry", f"native returned {type(wet).__name__}, not a DryHandoff "
                        "(wet progress kept as evidence only; no combined resume)", elapsed=elapsed, evidence=wet)
        dry, acc, r = wet.dry_state, wet.accounting, wet.progress.result
        t_end = float(dry.t_s)
        elapsed = _elapsed(t, t_end)

        def incomplete(code: str, detail: str) -> _Stop:
            return _Stop("incomplete", code, detail, elapsed=elapsed, evidence=wet)

        closure, budget = acc["sediment_closure"], acc["storm_water_budget"]
        if closure["closed"] is not True or closure["request_reconciled"] is not True or budget["closed"] is not True:
            raise incomplete("rain_budget_open", "native sediment closure / request / water budget not closed")
        if not abs(acc["combined_water_residual_m3"]) <= budget["tolerance_m3"]:
            raise incomplete("dry_water_budget_open", "native combined storm + dry-reset water budget not closed")
        if (bool(np.any(dry.bed.water.mobile_mass_by_cell_class_kg != 0.0)) or bool(np.any(dry.storm.depth_m != 0.0))
                or bool(np.any(dry.storm.soil_water_m != 0.0)) or dry.phase is not None
                or bool(np.any(dry.sediment_velocity_m_s != 0.0))):
            raise incomplete("dry_reset_not_empty", "the native dry state is not the documented empty reset")
        if dry.terrain is not held.terrain:
            raise incomplete("terrain_replaced", "the native dry state carries a different terrain reference")
        if not rain_edges[-1] <= t_end <= guard_end:
            raise incomplete("rain_end_outside_guard", "dry time outside [rainfall end, recession guard end]")

        export = _vec(np, r.by_class["export_actual"])
        inflow = np.zeros(nc, dtype=np.float64)  # the supported SYRUP event has no sediment boundary input
        inventory_after = _vec(np, n.bed_inventory(dry.bed))
        residual = inventory_after - inventory_before + export
        scale = max(float(np.abs(inventory_before).sum()), float(np.abs(inventory_after).sum()), float(export.sum()))
        bound = (_vec(np, closure["tolerance_kg"]) + _vec(np, acc["terminal_sediment_tolerance_kg"])
                 + n.summation_error_bound_kg(2 * self._inventory_elements + 4, scale))
        if not bool(np.all(np.abs(residual) <= bound)):
            raise incomplete("rain_inventory_open", "actual bed inventories do not close within native bounds")
        terminal = _vec(np, acc["terminal_deposition_by_class_kg"])
        settlement = acc.get("flow_end_settlement")
        body = {
            "native": {
                "dry_time_s": encode_time(t_end), "rainfall_end_s": encode_time(rain_edges[-1]),
                "recession_guard_end_s": encode_time(guard_end),
                "rainfall_edges_s": [encode_time(v) for v in rain_edges], "intensity_mm_h": intensities,
                "n_accepted_steps": int(r.n_accepted_steps), "n_rejected_attempts": int(r.n_rejected_attempts),
                "n_commits": int(r.n_commits), "report_rows": int(len(r.boundaries)), "report_rows_bound": rows,
                "storm_steps_lower_bound": lower_steps, "transport_scheme": "upwind", "phase": None,
            },
            "routing": {"first_graph_construction": first, "entry_rerouted": boundary.rerouted,
                        "entry_net_displacement_m": (None if boundary.net_displacement_m is None
                                                     else float(boundary.net_displacement_m)),
                        "graph_retained_by_identity": graph is held.graph,
                        "n_reroutes_in_event": int(r.n_reroutes)},
            "classes": {
                "actual_pickup_kg": _floats(np, r.by_class["actual_pickup"]),
                "deposition_actual_kg": _floats(np, r.by_class["deposition_actual"]),
                "export_kg": _floats(np, export), "import_kg": _floats(np, inflow),
                "internal_terminal_deposition_kg": _floats(np, terminal),
                "flow_end_mobile_before_settlement_kg": (None if settlement is None else
                                                         [float(v) for v in
                                                          settlement["mobile_before_settlement_by_class_kg"]]),
                "native_closure_residual_kg": _floats(np, closure["residual_kg"]),
                "native_closure_tolerance_kg": _floats(np, closure["tolerance_kg"]),
                "terminal_sediment_tolerance_kg": _floats(np, acc["terminal_sediment_tolerance_kg"]),
                "inventory_before_kg": _floats(np, inventory_before), "inventory_after_kg": _floats(np, inventory_after),
                "inventory_residual_kg": _floats(np, residual), "inventory_bound_kg": _floats(np, bound),
            },
            "water": {name: float(budget[name]) for name in
                      ("rain_m3", "surface_initial_m3", "soil_initial_m3", "surface_final_m3", "soil_final_m3",
                       "drainage_m3", "export_m3", "residual_m3", "tolerance_m3")}
            | {"dry_surface_removed_m3": float(acc["surface_removed_m3"]),
               "dry_soil_removed_m3": float(acc["soil_removed_m3"]), "water_added_m3": float(acc["water_added_m3"]),
               "combined_water_residual_m3": float(acc["combined_water_residual_m3"]),
               "dry_reset": "external dry-again removal (native); soil water is zero after the reset"},
        }
        if self._production is not None:
            try:
                body["production"] = {"scope": self._production.SCOPE,
                                      "worker": self._production_process_check(
                                          checkpoint="after_native_rain", elapsed=elapsed, evidence=wet),
                                      "entry_commit": self._production.commit_evidence(np, boundary.maple_result),
                                      "retained_wrapper_owners": self._production.retained_wrapper_owners(
                                          bed, self._wind, self._water, self._control, self._wind_profile,
                                          self._pool, self._soil, self._routing, len(self._receipts))}
            except _Stop:
                raise
            except MemoryError as exc:
                raise _Stop("runner_error", "address_space_exhausted", _detail(exc), exception=exc,
                            elapsed=elapsed, evidence=wet) from exc
            except Exception as exc:
                raise _Stop("runner_error", "production_evidence_failure", _detail(exc), exception=exc,
                            elapsed=elapsed, evidence=wet) from exc
        water_delta = {"rain_m3": float(budget["rain_m3"]), "drainage_m3": float(budget["drainage_m3"]),
                       "storm_export_m3": float(budget["export_m3"]),
                       "dry_surface_removed_m3": float(acc["surface_removed_m3"]),
                       "dry_soil_removed_m3": float(acc["soil_removed_m3"])}

        def publish() -> None:
            bed.adopt_bed_state(dry.bed, time_s=t_end)

        routing = SimpleNamespace(graph=dry.graph, terrain=dry.terrain, network=dry.network, grid=dry.grid,
                                  source=dry.routing_source_elevation_m, template=dry)
        return self._finish(n, index, "rain", resolved, stage, dry.bed, t, t_end, None, inflow, export, bound, body,
                            publish, pool=self._pool, soil=dry.storm.soil_water_m, routing=routing,
                            terminal=terminal, water=water_delta,
                            reroutes=(int(boundary.rerouted), int(r.n_reroutes)), elapsed=elapsed, evidence=wet)

    # atomic acceptance -----------------------------------------------------------------------------------------
    def _aggregate_record(self, np: Any, totals: dict[str, Any], inventory: Any) -> dict[str, Any]:
        return {"cumulative_export_kg": _floats(np, totals["export"]), "cumulative_import_kg": _floats(np, totals["input"]),
                "cumulative_native_bound_kg": _floats(np, totals["bound"]),
                "cumulative_internal_terminal_deposition_kg": _floats(np, totals["internal_terminal_deposition"]),
                "initial_inventory_kg": _floats(np, self._inventory0), "inventory_kg": _floats(np, inventory),
                "water": dict(totals["water"]), "n_entry_reroutes": totals["n_entry_reroutes"],
                "n_event_reroutes": totals["n_event_reroutes"]}

    def _finish(self, n: SimpleNamespace, index: int, kind: str, resolved: dict[str, Any], stage: Any,
                candidate: Any, t: float, t_end: float, returned_elapsed: float | None, inflow: Any, export: Any,
                event_bound: Any, body: dict[str, Any], publish, *, pool: Any, soil: Any, routing: Any,
                terminal: Any = None, water: dict[str, float] | None = None, reroutes: tuple[int, int] = (0, 0),
                elapsed: str, evidence: Any):
        """Prepare EVERYTHING (totals, counters, ledger, receipts, return value, identities), verify, adopt once,
        then assign only prepared references."""
        np, bed = n.np, self._bed

        def stop(status: str, code: str, detail: str, exc: BaseException | None = None,
                 guarantee: str = _UNCHANGED) -> _Stop:
            return _Stop(status, code, detail, exception=exc, elapsed=elapsed, evidence=evidence, guarantee=guarantee)

        # Complete candidate totals (a NEW mapping; the accepted one is never modified in place).
        old = self._totals
        totals = {"export": old["export"] + export, "input": old["input"] + inflow,
                  "bound": old["bound"] + event_bound,
                  "internal_terminal_deposition": (old["internal_terminal_deposition"] if terminal is None
                                                   else old["internal_terminal_deposition"] + terminal),
                  "water": (dict(old["water"]) if water is None
                            else {k: old["water"][k] + v for k, v in water.items()}),
                  "n_entry_reroutes": old["n_entry_reroutes"] + reroutes[0],
                  "n_event_reroutes": old["n_event_reroutes"] + reroutes[1]}
        # Aggregate per-class budget over every accepted event plus this candidate.
        inventory = _vec(np, n.bed_inventory(candidate))
        joint = inventory + totals["export"] - totals["input"] - self._inventory0
        n_terms = 2 * self._inventory_elements + 2 * (self._n_accepted + 1)
        scale = max(float(np.abs(self._inventory0).sum()), float(np.abs(inventory).sum()),
                    float(totals["export"].sum()), float(totals["input"].sum()))
        allowed = totals["bound"] + n.summation_error_bound_kg(n_terms, scale)
        if not bool(np.all(np.abs(joint) <= allowed)):
            raise stop("incomplete", "aggregate_budget_open", "joint per-class closure exceeds accumulated native bounds")
        # Candidate clock ledger (round trip through its own record, then append the native endpoints).
        stamp_before = float(stage.state.committed_topography.last_commit_time_s)
        stamp_after = float(candidate.committed_topography.last_commit_time_s)
        try:
            ledger = ClockLedger.from_record(self._ledger.as_record())
            clock = ledger.append("event", native_start=t, native_end=t_end, commit_before=stamp_before,
                                  commit_after=stamp_after, returned_elapsed=returned_elapsed)
        except ClockError as exc:
            raise stop("incomplete", "clock_rejected", _detail(exc), exc) from exc
        candidate_identity = _bed._sha256(bed._stage_identity(
            SimpleNamespace(case_sha256=bed.case_sha256, time_s=t_end, state=candidate)))
        receipt = {"schema": RECEIPT_SCHEMA, "index": index, "kind": kind, "status": "accepted", "code": "accepted",
                   "static_sha256": self._static_sha256, "resolved": resolved, "clock": clock,
                   "bed_identity_before": stage.identity_sha256, "bed_identity_after": candidate_identity,
                   **body,
                   "aggregate": {"joint_residual_kg": _floats(np, joint), "joint_bound_kg": _floats(np, allowed),
                                 "n_summation_terms": n_terms}}
        try:
            text = _canonical_json(receipt)
        except (TypeError, ValueError) as exc:
            raise stop("incomplete", "receipt_not_serializable", _detail(exc), exc) from exc
        if len(text.encode()) > self._limits.max_receipt_bytes:
            raise stop("refused", "receipt_cap_exceeded", f"{len(text.encode())} bytes > max_receipt_bytes="
                       f"{self._limits.max_receipt_bytes}")
        receipt = json.loads(text)
        receipts = [*self._receipts, receipt]
        returned = copy.deepcopy(receipt)
        next_index, n_accepted = index + 1, self._n_accepted + 1
        private_identity = self._private_identity(np, pool, soil)
        # Static and accepted-identity rechecks immediately before the one publication (changes DURING the call;
        # the runner's own work is staged copies, so any difference here was made outside the runner).
        try:
            bed.verify_case_inputs()
            live = _bed._sha256(self._static_parts_record(n))
        except (DenseBedError, DenseRunnerError) as exc:
            raise stop("refused", "case_input_mutated", exc.detail, exc, _EXTERNAL) from exc
        if live != self._static_parts_sha256:
            raise stop("refused", "static_input_mutated", "a controlling runner input changed during the event",
                       guarantee=_EXTERNAL)
        if (bed.identity_sha256() != stage.identity_sha256
                or self._private_identity(np, self._pool, self._soil) != self._accepted_private):
            raise stop("refused", "accepted_state_changed", "the accepted bed / pool / soil changed outside the "
                       "runner while the event was staged", guarantee=_EXTERNAL)
        # Allocation-light final OS/process check after identities, immediately before adoption.
        # Its refusal is internal resource failure, not an assertion of outside bed mutation.
        if self._production is not None:
            try:
                self._production.check_process(self._production_limits, self._worker_identity,
                                               checkpoint="before_publication")
            except self._production.DenseProductionError as exc:
                raise stop("refused", exc.code, exc.detail, exc) from exc
            except MemoryError as exc:
                raise stop("runner_error", "address_space_exhausted", _detail(exc), exc) from exc
            except Exception as exc:
                raise stop("runner_error", "production_resource_observation_failure", _detail(exc), exc) from exc
        try:
            publish()  # validates fully before its single replacement; a refusal publishes nothing
        except DenseBedError as exc:
            raise stop("refused", exc.code, exc.detail, exc) from exc
        # Publication done: assignment of already prepared references only (no computation, allocation,
        # copying or validation follows).
        self._pool = pool
        self._soil = soil
        self._routing = routing
        self._totals = totals
        self._ledger = ledger
        self._next = next_index
        self._n_accepted = n_accepted
        self._receipts = receipts
        self._accepted_bed = candidate_identity
        self._accepted_private = private_identity
        self._active_evidence = None
        return returned
