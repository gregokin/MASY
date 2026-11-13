"""Opt-in, fixed-geometry CPU water connection for the daily CSV schedule.

This delegates every physical operation to the explicitly qualified SYRUP
Phase 1.5 source pair. It is not the older MASY wind/water runner and does not
qualify wind exchange, LEAF, CUDA or a combined physical restart. Import is
metadata-only; native modules are loaded by ``open_water_session``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import itertools
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from masy.daily_csv import DailyScenario
from masy.daily_schedule import build_daily_plan

PROFILE = "masy.daily_native_water.cpu.v1"
SYRUP_PACKAGE_SHA256 = "82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107"
MAPLE_COMMIT = "9f466d2cfcd52991b24496cf67f4562d6268afa9"
MAPLE_PACKAGE_SHA256 = "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"


class DailyNativeError(ValueError):
    """A refusal. ``active_state`` is optional native evidence, never acceptance.

    At most one candidate is retained; callers must not accumulate these states
    in a history. The session's accepted boundary remains available separately.
    """

    def __init__(self, code: str, detail: str, *, active_state: Any = None):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail, self.active_state = code, detail, active_state


def preflight_water_only(scenario: DailyScenario) -> None:
    """Refuse unsupported work anywhere in the scenario before native mutation."""
    if type(scenario) is not DailyScenario:
        raise DailyNativeError("scenario", "a parsed DailyScenario is required")
    if scenario.occurrences:
        raise DailyNativeError("wind_bridge_unqualified", "daily persistent-water/wind exchange is not qualified")
    if scenario.settings.transpiration_enabled or scenario.settings.growth_enabled:
        raise DailyNativeError("leaf_bridge_pending", "LEAF plant-only demand/feedback/growth connection is required")


def _json(value: Any) -> str:
    def plain(item):
        if isinstance(item, Mapping):
            return {k: plain(v) for k, v in item.items()}
        if type(item) in (tuple, list):
            return [plain(v) for v in item]
        return item
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _positive(value: Any, name: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise DailyNativeError("limits", f"{name} must be finite and positive")
    return float(value)


def _source_identity() -> dict[str, str]:
    # Source identity, not an import-path override or a replacement runtime lock.
    root = Path(__file__).resolve().parent
    try:
        return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("daily_csv.py", "daily_schedule.py", "daily_native.py")}
    except OSError as exc:
        raise DailyNativeError("masy_source", "readable source files are required for source-bound execution") from exc


@dataclasses.dataclass(frozen=True)
class AcceptedWaterDay:
    """Bounded scalar receipt; no retained process owners or simulation arrays."""

    json_record: str

    def record(self) -> dict[str, Any]:
        return json.loads(self.json_record)


class DailyWaterSession:
    """One accepted boundary, with explicit, bounded native storm execution.

    Construct with ``open_water_session``. ``advance`` adopts only after the
    native daily operation, close and audits succeed. On refusal it retains the
    previous accepted boundary and one optional active candidate, and stops.
    Native Phase 1.5 calls are transactional; no wrapper bed copy is made.
    Output/checkpoint flags are requests for the caller; ``advance`` does not
    write model output or silently save a checkpoint.
    """

    def __init__(self, *, scenario, setup, admission, source, terrain, control,
                 quiet_policy, storm_limit_s, pet_max_bytes, checkpoint_path,
                 checkpoint_limits, host_available_bytes_fn):
        preflight_water_only(scenario)
        if checkpoint_path is not None and (checkpoint_limits is None or host_available_bytes_fn is None):
            raise DailyNativeError("checkpoint", "explicit native checkpoint quotas and host-memory reader required")
        # These imports must resolve to the caller's explicit immutable source
        # pair. The native verifier checks both package digests and module origins.
        try:
            ncp = importlib.import_module("maple_syrup.native_case_profile")
            nd = importlib.import_module("maple_syrup.native_dense_profile")
            lc = importlib.import_module("maple_syrup.layered_case")
            wp = importlib.import_module("maple_syrup.weather_pet")
            self._dw = importlib.import_module("maple_syrup.daily_water")
            self._wl = importlib.import_module("maple_syrup.water_lifecycle")
            self._v3 = importlib.import_module("maple_syrup.lifecycle_checkpoint_v3")
        except ImportError as exc:
            raise DailyNativeError("native_unavailable", str(exc)) from exc
        if type(source) is not ncp.NativeSourceProfile or type(admission) is not ncp.NativeCaseAdmission:
            raise DailyNativeError("native_source", "genuine native source profile and case admission are required")
        if (source.profile != ncp.PROFILE_NAME or source.syrup_package_digest != SYRUP_PACKAGE_SHA256
                or source.maple_commit != MAPLE_COMMIT or source.package_digest != MAPLE_PACKAGE_SHA256):
            raise DailyNativeError("native_source", "explicit Phase 1.5 CPU source pair required; no automatic rebinding")
        try:
            source.recheck()
            admission.recheck()
            binding = admission.latitude_binding()
        except ncp.NativeCaseProfileError as exc:
            raise DailyNativeError("native_source", str(exc)) from exc
        if type(setup) is not lc.LayeredCaseSetup:
            raise DailyNativeError("native_setup", "genuine layered-case setup required")
        lop = setup.lifecycle_operator
        ids = setup.identities
        expected = {"native_profile": nd.PROFILE_NAME, "native_bundle_id": source.bundle_id,
                    "native_package_digest": source.package_digest, "syrup_package_digest": source.syrup_package_digest,
                    "case_identity_sha256": admission.case_identity_sha256,
                    "latitude_deg": binding["value"], "latitude_binding": binding,
                    "lifecycle_operator_sha256": lop.sha256}
        mismatches = [key for key, value in expected.items() if ids.get(key) != value]
        if mismatches:
            raise DailyNativeError("native_setup", "case/source/operator association differs: " + ", ".join(mismatches))
        if ids.get("backend") not in ("numpy", "numba") or getattr(lop, "_cuda", None) is not None:
            raise DailyNativeError("backend", "only the explicit CPU profile is connected")
        if (type(terrain) is not wp.TerrainFactor or terrain.backend != "numpy"
                or terrain.shape != (lop.n_cells,) or terrain.latitude_deg != binding["value"]
                or not terrain.verify_identity()):
            raise DailyNativeError("terrain", "explicit native per-cell CPU TerrainFactor with the case latitude required")
        # Native factories own terrain science. There is no implicit flat-terrain
        # assumption or wrapper conversion from MAPLE aspect conventions here.
        ev = importlib.import_module("maple_syrup.layered_event")
        if type(control) is not ev.LayeredEventControl or type(quiet_policy) is not ev.LayeredQuietPolicy:
            raise DailyNativeError("native_controls", "explicit native storm control and quiet policy required")
        try:
            control.validated()
            quiet_policy.validated()
        except ev.LayeredEventError as exc:
            raise DailyNativeError("native_controls", str(exc)) from exc
        limit = _positive(storm_limit_s, "storm_limit_s")
        if type(pet_max_bytes) is not int or pet_max_bytes <= 0 or 32 * lop.n_cells > pet_max_bytes:
            raise DailyNativeError("limits", "pet_max_bytes must admit the bounded CPU PET workspace (32 bytes/cell)")
        self._scenario, self._terrain, self._wp = scenario, terrain, wp
        self._lop, self._source = lop, source
        self._control, self._quiet_policy, self._storm_limit = control, quiet_policy, limit
        self._latitude = binding["value"]
        self._failure = None
        self._pet_workspace = 32 * lop.n_cells
        if host_available_bytes_fn is None:
            host_available_bytes_fn = importlib.import_module("maple_syrup.subsurface").default_host_available_bytes
        if not callable(host_available_bytes_fn):
            raise DailyNativeError("limits", "host_available_bytes_fn must be callable")
        self._host_available = host_available_bytes_fn
        identity = {"profile": PROFILE, "scenario_sha256": scenario.identity_sha256,
                    "masy_source_sha256": _source_identity(), "native": expected,
                    "layered_config": ids["config"], "texture_source_sha256": ids["texture_source_sha256"],
                    "terrain": terrain.identity(), "storm_control": dataclasses.asdict(control),
                    "quiet_policy": dataclasses.asdict(quiet_policy), "storm_limit_s_hex": limit.hex(),
                    "pet_max_bytes": pet_max_bytes}
        # Absolute paths record provenance but don't define mathematical identity.
        identity["native"]["latitude_binding"] = {k: v for k, v in binding.items() if k != "case_yaml_path"}
        self._identity_json = _json(identity)
        self._run_digest = hashlib.sha256(self._identity_json.encode()).hexdigest()
        self._masy_source = identity["masy_source_sha256"]
        # Native rain validation is metadata-only. Validate all nominal rows up
        # front rather than discovering depth underflow on a late day. Actual
        # forcing still binds to each occurrence's realized physical clock.
        rc = importlib.import_module("maple_syrup.recharge")
        for day in build_daily_plan(scenario):
            try:
                rc.plan_daily_rain(day.weather.precipitation_mm, day.weather.precipitation_intensity_mm_h,
                                   scenario.weather.precipitation_threshold_mm, run_digest=self._run_digest,
                                   ordinal=day.day_number - 1, label=day.date.isoformat())
            except rc.RechargeError as exc:
                raise DailyNativeError("native_rain_plan", f"{day.date}: {exc}") from exc
        self._reference_bed = setup.state.payload.bed
        if checkpoint_path is None:
            state = self._dw.start_daily_lineage(lop, setup.state, expected_parent=self._wl.expect_parent(setup.state),
                                                 run_digest=self._run_digest)
        else:
            if checkpoint_limits is None:
                raise DailyNativeError("checkpoint", "explicit native checkpoint quotas and host-memory reader required")
            restored = self._v3.load_lifecycle_checkpoint_v3(
                checkpoint_path, lop, self.checkpoint_identity(), expected_role="boundary", limits=checkpoint_limits,
                host_available_bytes_fn=self._host_available, reference_bed=self._reference_bed)
            self._lop, state = restored.lop, restored.state
        if (state.role != "boundary" or state.daily is None or state.daily.pending is not None
                or state.daily.run_digest != self._run_digest
                or not 0 <= state.daily.ordinal <= scenario.settings.day_count):
            raise DailyNativeError("cursor", "native accepted boundary does not match this daily scenario")
        self._state = state
        self._days = iter(itertools.islice(build_daily_plan(scenario), state.daily.ordinal, None))

    @property
    def state(self):
        """Current accepted native boundary, read-only under the native contract."""
        return self._state

    @property
    def lifecycle_operator(self):
        return self._lop

    @property
    def failure(self):
        # Keep exception tracebacks (and their large native frame locals) out of
        # the long-lived session. Active evidence is returned on the exception.
        return None if self._failure is None else dict(self._failure)

    def checkpoint_identity(self) -> dict[str, Any]:
        return json.loads(self._identity_json)

    def save_checkpoint(self, path, *, limits, host_available_bytes_fn, **native_quota_options):
        """Native V3 atomic boundary publication, with explicit caller quotas.

        This water-only checkpoint binds the full parsed scenario and native
        daily cursor. It does not contain a LEAF or wind state.
        """
        return self._v3.save_lifecycle_checkpoint_v3(
            path, self._lop, self._state, self.checkpoint_identity(), limits=limits,
            host_available_bytes_fn=host_available_bytes_fn, **native_quota_options)

    def advance(self) -> AcceptedWaterDay:
        if self._failure is not None:
            raise DailyNativeError("session_stopped", "restore an accepted boundary after inspecting the earlier refusal")
        if self._state.daily.ordinal == self._scenario.settings.day_count:
            raise StopIteration
        try:
            day = next(self._days)
        except BaseException as exc:
            code = "cursor" if isinstance(exc, StopIteration) else "interrupted"
            self._failure = {"code": code, "detail": "daily plan stopped before schedule completion"}
            if isinstance(exc, StopIteration):
                raise DailyNativeError(code, self._failure["detail"]) from exc
            raise
        parent, lop, dw, wl = self._state, self._lop, self._dw, self._wl
        active = None
        stage = "weather"
        try:
            if parent.daily.ordinal != day.day_number - 1:
                raise DailyNativeError("cursor", "native ordinal differs from the nominal date")
            if _source_identity() != self._masy_source:
                raise DailyNativeError("masy_source_changed", "connection source changed since session admission")
            available = self._host_available()
            if type(available) is not int or available < self._pet_workspace:
                raise DailyNativeError("pet_workspace_unavailable", "host-memory reading does not admit the PET workspace")
            w = day.weather
            wp = self._wp
            native_weather = wp.DailyWeather(w.date, w.tmin_c, w.tmean_c, w.tmax_c, pet_mm_day=w.pet_mm_day)
            demand = wp.potential_demand(native_weather, self._latitude, method=self._scenario.weather.pet_method)
            applied = wp.apply_terrain(demand, self._terrain)
            pet = wp.pet_m_s(applied.pet_mm_day_cells).reshape(lop.n_cells)
            if not day.evaporation_enabled:
                pet.fill(0.0)
            plan = dw.next_plan(parent, w.precipitation_mm, w.precipitation_intensity_mm_h,
                                self._scenario.weather.precipitation_threshold_mm, label=day.date.isoformat())
            stage = "daily_water"
            result = None
            fallback = None
            if plan.selection != "storm":
                result = dw.daily_water_action(lop, parent, expected_parent=wl.expect_parent(parent), plan=plan,
                                               pet_terrain_m_s=pet, policy=self._quiet_policy)
                if result.status == "capacity_failure":
                    fallback = result.fallback
            if result is None or result.status == "capacity_failure":
                stage = "storm"
                active = dw.begin_daily_storm(lop, parent, expected_parent=wl.expect_parent(parent), plan=plan,
                                              fallback_of=fallback, control=self._control, policy=self._quiet_policy)
                until = float(active.t_s) + self._storm_limit
                if not math.isfinite(until) or until <= active.t_s:
                    raise DailyNativeError("clock_limit", "storm guard is not representable on the native clock", active_state=active)
                evolved = wl.evolve_storm(lop, active, expected_parent=wl.expect_parent(active), until_s=until)
                active = evolved.state
                if not evolved.complete:
                    raise DailyNativeError("storm_incomplete", str(evolved.reason), active_state=active)
                stage = "daily_after_storm"
                result = dw.finish_daily_storm(lop, active, expected_parent=wl.expect_parent(active), plan=plan, pet_terrain_m_s=pet)
            if result.status != "complete" or result.state is None or result.receipt is None:
                raise DailyNativeError("native_result", "native daily operation did not accept a day", active_state=active)
            active = result.state
            stage = "close"
            closed = wl.close_interevent(lop, active, expected_parent=wl.expect_parent(active))
            active = closed
            audit = wl.audit_lifecycle(lop, closed)
            if (closed.daily.ordinal != day.day_number or closed.daily.pending is not None
                    or closed.daily.run_digest != self._run_digest or closed.role != "boundary"):
                raise DailyNativeError("cursor", "native close did not consume exactly one occurrence", active_state=closed)
            if _source_identity() != self._masy_source:
                raise DailyNativeError("masy_source_changed", "connection source changed during the candidate day", active_state=closed)
            stage = "record"
            receipt = result.receipt
            record = {"profile": PROFILE, "date": day.date.isoformat(), "accepted_days": closed.daily.ordinal,
                      "scenario_sha256": self._scenario.identity_sha256, "run_digest": self._run_digest,
                      "native_receipt": receipt.as_dict(), "native_receipt_sha256": receipt.digest(),
                      "physical_clock_hex": {"occurrence_start_s": float(receipt.occurrence_start_s).hex(),
                                             "daily_start_s": float(receipt.t_start_s).hex(), "end_s": float(closed.t_s).hex()},
                      "native_audit": audit, "reference_pet_mm_day": demand.pet_mm_day,
                      "evaporation_enabled": day.evaporation_enabled, "output_requested": day.output_due,
                      "checkpoint_requested": day.checkpoint_due,
                      "event_boundary_output_requested": day.event_boundary_output_requested,
                      "is_final_day": day.is_final_day}
            accepted = AcceptedWaterDay(_json(record))
        except Exception as exc:
            error = exc if isinstance(exc, DailyNativeError) else DailyNativeError(
                "record_failed" if stage == "record" else getattr(exc, "code", "native_refused"),
                f"{stage}: {exc}", active_state=active)
            self._failure = {"code": error.code, "detail": error.detail}
            if error is exc:
                raise
            raise error from exc
        except BaseException:
            self._failure = {"code": "interrupted", "detail": f"{stage}: execution interrupted before adoption"}
            raise
        self._state = closed
        return accepted


def open_water_session(scenario: DailyScenario, *, setup, admission, source, terrain,
                       control, quiet_policy, storm_limit_s, pet_max_bytes,
                       checkpoint_path=None, checkpoint_limits=None,
                       host_available_bytes_fn=None) -> DailyWaterSession:
    """Open the explicit Phase 1.5 CPU connection on real native factory products.

    ``setup`` must be prepared by SYRUP's native layered-case factory; MAPLE
    initializes its landscape. Native source/profile verification is mandatory.
    Storm guard and PET workspace are explicit caller limits, not weather or
    calendar controls. No physics, completion tolerance or method is invented.
    """
    return DailyWaterSession(scenario=scenario, setup=setup, admission=admission, source=source, terrain=terrain,
                             control=control, quiet_policy=quiet_policy, storm_limit_s=storm_limit_s,
                             pet_max_bytes=pet_max_bytes, checkpoint_path=checkpoint_path,
                             checkpoint_limits=checkpoint_limits, host_available_bytes_fn=host_available_bytes_fn)
