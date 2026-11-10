"""Mechanical temporal coordinator for wind / rain / interevent occurrences (no physics).

The coordinator owns TEMPORAL coordination only: it walks a `Scenario` cursor,
builds one sealed dispatch per occurrence, hands an independent copy to the
explicitly supplied adapter of that kind, validates the adapter's realized
outcome against its own sealed frame and a fresh re-derivation of the intended
event, and advances the boundary atomically. It never constructs native
MAPLE/SYRUP/LEAF state, never initializes a landscape or vegetation, never
applies a process, never retimes forcing.

Adapters are REQUIRED inputs: any kind present in the scenario without an
adapter refuses before anything runs. An adapter object must provide
`capabilities() -> AdapterCapabilities` (every field explicit; in particular
`continuous_soil_moisture` has no default, so the existing dry-reset profile
cannot be advertised continuous by omission, and `native_staging` must declare
that the adapter stages native state until the coordinator accepts) and
`run(dispatch) -> AdapterOutcome`. Capability preflight (duration policy,
calendar support) runs for EVERY occurrence of that adapter's kind before the
adapter is called, independently of weather or process gates.

Dispatch integrity. The adapter receives a deep copy of the sealed dispatch;
the copy is compared with the sealed frame after `run` and any change refuses
(`dispatch_mutated`). Validation never reads the adapter-visible object: the
planned edges are re-derived from the scenario's own event file and the
sealed native start. Public accessors (`dispatch()`, `realized`, `ledger`,
`as_record()`) return independent deep copies; mutating them cannot change
accepted state.

Atomicity scope. Every check runs on a trial copy of the existing
`masy.clock.ClockLedger`; ledger, realized list and cursor are replaced
together only after all checks pass, so a refusal leaves the previous
temporal boundary intact and keeps the adapter's output as `active_evidence`.
That atomicity covers the coordinator's OWN cursor only: it cannot roll back
a native model an adapter already mutated in place. Hence the protocol
requires `native_staging="staged_until_accepted"`: an adapter must stage
native changes (bed, soil water, plants, process-private state) and publish
them only after the coordinator accepted the boundary, or discard them on
refusal, exactly as the existing `DenseRunner` stages bed copies. No native
checkpoint or transaction is implemented or claimed here.

Restart: `as_record()` is an immutable temporal-cursor record (scenario
digest, ledger, realized occurrences, adapter identities, latitude binding);
`Coordinator.resume` rebuilds from it, refuses any scenario/adapter/latitude
identity mismatch, re-derives every realized record and requires
`as_record()` to round-trip identically. It is NOT a native state checkpoint.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from masy.calendar import CalendarError, CalendarStart, fraction_text, noleap_day_index
from masy.clock import ClockError, ClockLedger, decode_time, encode_time, resolve_forcing_edges
from masy.dense_schedule import _exact_decimal
from masy.event_files import EVENT_KINDS, INTEREVENT_PROCESSES, IntereventFile, RainEventFile, process_gate
from masy.scenario import Occurrence, Scenario
from masy.scenario_io import canonical_json
from masy.weather import LatitudeBinding, WeatherError, thaw

__all__ = [
    "CALENDAR_SUPPORTS",
    "CURSOR_SCHEMA",
    "DISPATCH_SCHEMA",
    "DURATION_POLICIES",
    "STAGING_POLICIES",
    "AdapterCapabilities",
    "AdapterOutcome",
    "Coordinator",
    "CoordinatorError",
]

CURSOR_SCHEMA = "masy.coordinator_cursor.v1"
DISPATCH_SCHEMA = "masy.dispatch.v1"
OCCURRENCE_SCHEMA = "masy.realized_occurrence.v1"
DURATION_POLICIES = ("any_positive_duration", "whole_days_86400_only")
CALENDAR_SUPPORTS = ("proleptic_gregorian_dates", "noleap_365_day_index")
CHECKPOINT_SCOPES = ("not_provided", "adapter_owned")
STAGING_POLICIES = ("staged_until_accepted", "in_place_unstaged")
STATE_SCOPE = ("temporal cursor only: scenario identity, exact clock ledger, realized occurrences and adapter "
               "identities; native bed/soil/plant/process state checkpoints are adapter-owned and NOT included; "
               "cursor atomicity cannot roll back a native model mutated in place, hence native_staging is required")
PROCESS_GATE_CONTRACT = ("evaporation/transpiration/growth are dispatched only inside interevent occurrences; wind and "
                         "rain (recession included) carry an all-false gate; calendar time advances through every "
                         "occurrence; no process is applied retroactively over an excluded duration")
DISPATCH_CONTRACT = ("read-only: the adapter receives an independent copy of the sealed dispatch; any modification of "
                     "that copy refuses the outcome (dispatch_mutated); validation uses the coordinator's sealed frame "
                     "and a fresh derivation from the scenario, never the adapter-visible object")


class CoordinatorError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _text(value: Any, where: str, maximum: int = 256) -> str:
    if type(value) is not str or not 0 < len(value) <= maximum or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise CoordinatorError("invalid_capability", f"{where} must be a non-empty printable string of at most {maximum} chars")
    return value


def _deep_copy(value: Any) -> Any:
    """Independent plain-JSON copy (all coordinator records are plain JSON data)."""
    return json.loads(canonical_json(value))


@dataclass(frozen=True)
class AdapterCapabilities:
    """Explicit adapter declaration; no field has a default so nothing is implied."""

    kind: str
    adapter_id: str
    source_identity: str
    continuous_soil_moisture: bool
    duration_policy: str
    calendar_support: str
    native_state_checkpoint: str
    native_staging: str
    notes: str

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise CoordinatorError("invalid_capability", f"kind must be one of {EVENT_KINDS}")
        _text(self.adapter_id, "adapter_id")
        _text(self.source_identity, "source_identity", 512)
        if type(self.continuous_soil_moisture) is not bool:
            raise CoordinatorError("invalid_capability", "continuous_soil_moisture must be an explicit bool")
        if self.duration_policy not in DURATION_POLICIES:
            raise CoordinatorError("invalid_capability", f"duration_policy must be one of {DURATION_POLICIES}")
        if self.calendar_support not in CALENDAR_SUPPORTS:
            raise CoordinatorError("invalid_capability", f"calendar_support must be one of {CALENDAR_SUPPORTS}")
        if self.native_state_checkpoint not in CHECKPOINT_SCOPES:
            raise CoordinatorError("invalid_capability", f"native_state_checkpoint must be one of {CHECKPOINT_SCOPES}")
        if self.native_staging not in STAGING_POLICIES:
            raise CoordinatorError("invalid_capability", f"native_staging must be one of {STAGING_POLICIES}")
        if type(self.notes) is not str or len(self.notes) > 2048:
            raise CoordinatorError("invalid_capability", "notes must be a string of at most 2048 chars")

    def record(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_record(cls, record: Any) -> "AdapterCapabilities":
        names = tuple(f.name for f in dataclasses.fields(cls))
        if type(record) is not dict or set(record) != set(names):
            raise CoordinatorError("invalid_capability", f"capability record requires exactly {names}")
        return cls(**record)


@dataclass(frozen=True)
class AdapterOutcome:
    """What an adapter realized. `status` must be exactly "complete" for the
    boundary to advance; anything else is kept as evidence only."""

    status: str
    native_end_s: float | None
    commit_before_s: float | None = None
    commit_after_s: float | None = None
    returned_elapsed_s: float | None = None
    evidence: Any = None


class Coordinator:
    def __init__(self, scenario: Scenario, adapters: dict[str, Any], *, latitude: LatitudeBinding | None = None,
                 initial_commit_s: float | None = None) -> None:
        if type(scenario) is not Scenario:
            raise CoordinatorError("invalid_scenario", "scenario must be a masy.scenario.Scenario")
        self.scenario = scenario
        self._adapters, self._capabilities = _bind_adapters(scenario, adapters)
        if latitude is not None and type(latitude) is not LatitudeBinding:
            raise CoordinatorError("invalid_latitude", "latitude must be a masy.weather.LatitudeBinding or None")
        weather = scenario.weather
        if weather is not None and weather.pet.needs_latitude and latitude is None:
            raise CoordinatorError("latitude_required",
                                   f"weather pet.method {weather.pet_method_name!r} needs the bound MAPLE case latitude; "
                                   "there is no equator default")
        self.latitude = latitude
        try:
            self._ledger = ClockLedger(scenario.native_epoch_s)
        except ClockError as exc:
            raise CoordinatorError("clock_refused", str(exc)) from exc
        commit = scenario.native_epoch_s if initial_commit_s is None else initial_commit_s
        try:
            ClockLedger(commit)  # same native-clock value rules: finite, non-negative, not -0.0, never a bool
        except ClockError as exc:
            raise CoordinatorError("invalid_initial_commit", str(exc)) from exc
        if float(commit) > scenario.native_epoch_s:
            raise CoordinatorError("invalid_initial_commit", "initial_commit_s must not exceed native_epoch_s")
        self._initial_commit = float(commit)
        self._realized: list[dict[str, Any]] = []
        self._next = 0
        self.active_evidence: Any = None
        self.last_refusal: dict[str, Any] | None = None
        # Immutable identity bound at construction: the scenario digest (every event/weather record and file hash)
        # and the latitude record. Any later change of the loaded objects is refused before dispatch, acceptance,
        # publication or resume; bounded metadata only (no native state is hashed here).
        self._bound_identity = {"scenario_sha256": scenario.digest,
                                "latitude": None if latitude is None else latitude.record()}

    def _check_bound_identity(self) -> None:
        current = self.scenario.digest
        if current != self._bound_identity["scenario_sha256"]:
            raise CoordinatorError("scenario_changed",
                                   f"the loaded scenario changed after binding ({self._bound_identity['scenario_sha256']} -> "
                                   f"{current}); the coordinator refuses to dispatch, accept, publish or resume against it")
        latitude = None if self.latitude is None else self.latitude.record()
        if latitude != self._bound_identity["latitude"]:
            raise CoordinatorError("latitude_changed", "the bound latitude record changed after construction")

    # ------------------------------------------------------------------ state
    @property
    def next_index(self) -> int:
        return self._next

    @property
    def finished(self) -> bool:
        return self._next >= len(self.scenario.occurrences)

    @property
    def ledger(self) -> ClockLedger:
        return ClockLedger.from_record(self._ledger.as_record())  # independent copy; the live ledger is private

    @property
    def realized(self) -> tuple[dict[str, Any], ...]:
        return tuple(_deep_copy(r) for r in self._realized)

    def capabilities(self) -> dict[str, AdapterCapabilities]:
        return dict(self._capabilities)

    def soil_moisture_continuity(self) -> dict[str, Any]:
        declared = {kind: caps.continuous_soil_moisture for kind, caps in sorted(self._capabilities.items())}
        return {"continuous_soil_moisture_across_events": all(declared.values()),
                "declared_by_adapter": declared,
                "meaning": "True only when EVERY supplied adapter explicitly declares it; the existing native "
                           "dry-reset rain handoff declares False and is never advertised continuous"}

    def _last_commit(self) -> float:
        records = self._ledger.records()
        return decode_time(records[-1]["commit_after"]) if records else self._initial_commit

    def _previous_end(self) -> tuple[float, Fraction]:
        records = self._ledger.records()
        if not records:
            return self._ledger.epoch, Fraction(0)
        native = decode_time(records[-1]["native_end"])
        return native, self._ledger.global_elapsed(native)

    # --------------------------------------------------------------- dispatch
    def _planned(self, occurrence: Occurrence, native_start: float) -> dict[str, Any]:
        """Fresh derivation of the intended edges from the scenario's own event file."""
        offsets, _ = occurrence.event.planned_edges()
        try:
            return resolve_forcing_edges(native_start, [_exact_decimal(o) for o in offsets])
        except ClockError as exc:
            raise CoordinatorError("planned_edges_refused", f"{occurrence.occurrence_id}: {exc}") from exc

    def dispatch(self) -> dict[str, Any]:
        """Pure: an independent copy of the sealed dispatch for the next occurrence, or a refusal."""
        return _deep_copy(self._sealed_dispatch())

    def _sealed_dispatch(self, *, with_pet: bool = True) -> dict[str, Any]:
        self._check_bound_identity()
        if self.finished:
            raise CoordinatorError("scenario_finished", "every occurrence has been realized")
        occurrence = self.scenario.occurrences[self._next]
        previous_native, previous_global = self._previous_end()
        gap_entry = None
        native_start = previous_native
        if occurrence.scheduled_start_s is not None:
            scheduled = occurrence.scheduled_start_s
            if scheduled < previous_global:
                end_instant = self.scenario.calendar.instant(previous_global)
                raise CoordinatorError("scheduled_start_overlap",
                                       f"{occurrence.occurrence_id} is scheduled at {occurrence.scheduled_start} but the "
                                       f"previous occurrence realized its end at {end_instant.record()['iso']} "
                                       f"(+{fraction_text(end_instant.subsecond)} s); forcing is never truncated or shifted")
            if scheduled > previous_global and self.scenario.gap_policy != "idle":
                raise CoordinatorError("uncovered_gap_refused",
                                       f"{occurrence.occurrence_id} starts {fraction_text(scheduled - previous_global)} s "
                                       "after the realized previous end; gap_policy 'refuse' rejects uncovered time")
            try:
                native_start = self._ledger.native_from_global(scheduled)
            except ClockError as exc:
                raise CoordinatorError("scheduled_start_not_native", str(exc)) from exc
            if scheduled > previous_global:
                gap_entry = {"kind": "idle", "native_start": encode_time(previous_native), "native_end": encode_time(native_start),
                             "processes": {name: False for name in INTEREVENT_PROCESSES},
                             "meaning": "explicit uncovered interval: no process dispatched, calendar advanced"}
        event = occurrence.event
        _, roles = event.planned_edges()
        planned = self._planned(occurrence, native_start)
        start_global = self._ledger.global_elapsed(native_start)
        planned_end_native = decode_time(planned["edges"][-1]["native"])
        planned_end_global = self._ledger.global_elapsed(planned_end_native)
        calendar = self.scenario.calendar
        caps = self._capabilities[event.kind]
        try:
            cal_start = calendar.instant(start_global)
            cal: dict[str, Any] = {"start": cal_start.record(), "planned_end": calendar.instant(planned_end_global).record()}
            if type(event) is RainEventFile:
                rainfall_end = decode_time(planned["edges"][-2]["native"])
                cal["rainfall_end"] = calendar.instant(self._ledger.global_elapsed(rainfall_end)).record()
                cal["note"] = "realized end lies in [rainfall_end, planned_end] (recession guard); it is not predicted"
            preflight = _capability_preflight(occurrence, caps, calendar, start_global, planned_end_global)
        except CalendarError as exc:
            raise CoordinatorError(exc.code, f"{occurrence.occurrence_id}: {exc}") from exc
        cal.update(preflight)
        gate = process_gate(event)
        weather = None
        if type(event) is IntereventFile and any(gate.values()):
            weather = self._weather_slice(occurrence, start_global, planned_end_global, with_pet=with_pet)
        return {
            "schema": DISPATCH_SCHEMA, "scenario_sha256": self.scenario.digest, "scenario_id": self.scenario.scenario_id,
            "occurrence_index": occurrence.index, "occurrence_id": occurrence.occurrence_id, "kind": event.kind,
            "event_file": occurrence.file, "event_file_sha256": occurrence.file_sha256, "event_digest": event.digest,
            "event": event.record(), "native_start": encode_time(native_start), "global_start_s": fraction_text(start_global),
            "planned_edges": planned, "edge_roles": roles, "calendar": cal, "processes": gate,
            "process_gate_contract": PROCESS_GATE_CONTRACT, "dispatch_contract": DISPATCH_CONTRACT, "weather": weather,
            "latitude": None if self.latitude is None else self.latitude.record(),
            "adapter": caps.record(), "gap_entry": gap_entry,
            "commit_before_expected": encode_time(self._last_commit()),
        }

    def _weather_slice(self, occurrence: Occurrence, start_g: Fraction, end_g: Fraction, *, with_pet: bool = True) -> dict[str, Any]:
        """Coverage is always checked; daily PET is evaluated by the owning library for a live dispatch and
        skipped during cursor replay (`with_pet=False`), where no adapter receives the dispatch."""
        calendar = self.scenario.calendar
        weather = self.scenario.weather
        assert weather is not None  # Scenario static check guarantees it
        dates = calendar.dates_touched(start_g, end_g)
        try:
            rows = weather.coverage(dates)  # independent copies of the accepted rows
            pet = [weather.daily_pet(row, self.latitude) for row in rows] if with_pet else None
        except WeatherError as exc:
            raise CoordinatorError(exc.code, f"{occurrence.occurrence_id}: {exc}") from exc
        return {"weather_id": weather.weather_id, "weather_sha256": weather.source_sha256, "dates": [d.isoformat() for d in dates],
                "rows": list(rows), "daily_pet": pet, "library": thaw(weather.library),
                "units": "degC, RH fraction, VPD kPa, radiation MJ m-2 day-1, PET mm day-1",
                "coverage": "exactly the calendar dates intersecting this interevent interval; nothing earlier"}

    # ------------------------------------------------------------------- step
    def step(self) -> dict[str, Any]:
        """Dispatch the next occurrence, validate the outcome, advance atomically."""
        sealed = self._sealed_dispatch()
        handed = _deep_copy(sealed)
        adapter = self._adapters[sealed["kind"]]
        try:
            outcome = adapter.run(handed)
        except Exception as exc:  # noqa: BLE001 - adapter failure is evidence, never a partial advance
            self.active_evidence = exc
            self.last_refusal = {"code": "adapter_failure", "occurrence_id": sealed["occurrence_id"],
                                 "detail": f"{type(exc).__name__}: {exc}"[:2000]}
            raise CoordinatorError("adapter_failure", f"{sealed['occurrence_id']}: {type(exc).__name__}: {exc}") from exc
        try:
            if canonical_json(handed) != canonical_json(sealed):
                raise CoordinatorError("dispatch_mutated",
                                       f"{sealed['occurrence_id']}: the adapter modified its dispatch copy; the dispatch is "
                                       "read-only and the outcome is refused (cursor unchanged)")
            realized, trial = self._validate(sealed, outcome)
        except CoordinatorError as exc:
            self.active_evidence = outcome
            self.last_refusal = {"code": exc.code, "occurrence_id": sealed["occurrence_id"], "detail": exc.detail}
            raise
        except (TypeError, ValueError) as exc:  # a non-JSON object smuggled into the copy, etc.
            self.active_evidence = outcome
            self.last_refusal = {"code": "dispatch_mutated", "occurrence_id": sealed["occurrence_id"], "detail": str(exc)[:2000]}
            raise CoordinatorError("dispatch_mutated", f"{sealed['occurrence_id']}: dispatch copy is no longer plain data ({exc})") from exc
        # atomic boundary advance: every check passed; replace references together
        self._ledger = trial
        self._realized.append(realized)
        self._next += 1
        self.active_evidence = None
        self.last_refusal = None
        return _deep_copy(realized)

    def run(self) -> list[dict[str, Any]]:
        while not self.finished:
            self.step()
        return list(self.realized)

    def _validate(self, sealed: dict[str, Any], outcome: Any) -> tuple[dict[str, Any], ClockLedger]:
        self._check_bound_identity()  # recheck after adapter.run, before any accepted boundary advances
        occurrence = self.scenario.occurrences[self._next]
        occurrence_id = occurrence.occurrence_id
        if type(outcome) is not AdapterOutcome:
            raise CoordinatorError("invalid_outcome", f"{occurrence_id}: adapter returned {type(outcome).__name__}, not AdapterOutcome")
        if outcome.status != "complete":
            raise CoordinatorError("event_incomplete",
                                   f"{occurrence_id}: adapter status {outcome.status!r} is not completion; a limit or error is "
                                   "not completion; cursor unchanged, output kept as active_evidence")
        if outcome.native_end_s is None:
            raise CoordinatorError("invalid_outcome", f"{occurrence_id}: complete outcome lacks native_end_s")
        native_start = decode_time(sealed["native_start"])
        planned = self._planned(occurrence, native_start)  # fresh derivation, never the adapter-visible copy
        edges = planned["edges"]
        planned_end = decode_time(edges[-1]["native"])
        try:
            trial = ClockLedger.from_record(self._ledger.as_record())
            expected_commit = self._last_commit()
            commit_before = expected_commit if outcome.commit_before_s is None else outcome.commit_before_s
            if type(commit_before) is not float or commit_before.hex() != expected_commit.hex():
                raise CoordinatorError("commit_chain_refused",
                                       f"{occurrence_id}: commit_before {commit_before!r} must equal the accepted chain stamp "
                                       f"{expected_commit!r} (the initial commit for the first occurrence)")
            commit_after = commit_before if outcome.commit_after_s is None else outcome.commit_after_s
            gap_index = None
            if sealed["gap_entry"] is not None:
                gap_index = len(trial.records())
                trial.append("idle", native_start=decode_time(sealed["gap_entry"]["native_start"]),
                             native_end=native_start, commit_before=self._last_commit(), commit_after=self._last_commit())
            entry_index = len(trial.records())
            if occurrence.kind == "interevent" and commit_after != commit_before:
                raise CoordinatorError("interevent_commit_refused",
                                       f"{occurrence_id}: an interevent occurrence performs no bed commit; stamps must be equal")
            end = outcome.native_end_s
            if occurrence.kind == "rain":
                rainfall_end = decode_time(edges[-2]["native"])
                if not (type(end) is float and rainfall_end <= end <= planned_end):
                    raise CoordinatorError("rain_end_outside_guard",
                                           f"{occurrence_id}: realized end {end!r} not in [rainfall end {rainfall_end!r}, "
                                           f"recession guard end {planned_end!r}]")
            elif type(end) is not float or end.hex() != planned_end.hex():
                raise CoordinatorError("realized_end_mismatch",
                                       f"{occurrence_id}: {occurrence.kind} must end exactly at its planned end "
                                       f"{planned_end.hex()}, adapter realized {end!r}")
            trial.append("event", native_start=native_start, native_end=end, commit_before=commit_before,
                         commit_after=commit_after, returned_elapsed=outcome.returned_elapsed_s)
        except ClockError as exc:
            raise CoordinatorError("clock_refused", f"{occurrence_id}: {exc}") from exc
        realized = self._realized_record(occurrence, entry_index, gap_index, trial.records(),
                                         evidence_present=outcome.evidence is not None)
        return realized, trial

    def _realized_record(self, occurrence: Occurrence, entry_index: int, gap_index: int | None,
                         entries: list[dict[str, Any]], *, evidence_present: bool) -> dict[str, Any]:
        """Every field except `evidence_present` is re-derived from the scenario and the ledger entry."""
        entry = entries[entry_index]
        start = decode_time(entry["native_start"])
        end = decode_time(entry["native_end"])
        planned = self._planned(occurrence, start)
        planned_end = decode_time(planned["edges"][-1]["native"])
        start_g, end_g = Fraction(entry["global_start"]), Fraction(entry["global_end"])
        calendar = self.scenario.calendar
        gate = process_gate(occurrence.event)
        dates = None
        if type(occurrence.event) is IntereventFile and any(gate.values()):
            dates = [d.isoformat() for d in calendar.dates_touched(start_g, end_g)]
        return {
            "schema": OCCURRENCE_SCHEMA, "index": occurrence.index, "occurrence_id": occurrence.occurrence_id,
            "kind": occurrence.kind, "event_file_sha256": occurrence.file_sha256, "event_digest": occurrence.event.digest,
            "ledger_entry_index": entry_index, "gap_ledger_entry_index": gap_index,
            "native_start": entry["native_start"], "native_end": entry["native_end"],
            "global_start_s": entry["global_start"], "global_end_s": entry["global_end"],
            "planned_end": encode_time(planned_end),
            "realized_minus_planned_end_s": fraction_text(Fraction(end) - Fraction(planned_end)),
            "calendar_start": calendar.instant(start_g).record(), "calendar_end": calendar.instant(end_g).record(),
            "processes": gate, "weather_dates": dates, "status": "complete", "evidence_present": evidence_present,
        }

    # ---------------------------------------------------------------- records
    def as_record(self) -> dict[str, Any]:
        self._check_bound_identity()
        return _deep_copy({
            "schema": CURSOR_SCHEMA, "scenario_sha256": self.scenario.digest, "scenario_id": self.scenario.scenario_id,
            "native_epoch": encode_time(self.scenario.native_epoch_s), "initial_commit": encode_time(self._initial_commit),
            "ledger": self._ledger.as_record(), "occurrences": list(self._realized),
            "next_index": self._next, "finished": self.finished,
            "adapters": {kind: caps.record() for kind, caps in sorted(self._capabilities.items())},
            "latitude": None if self.latitude is None else self.latitude.record(),
            "soil_moisture_continuity": self.soil_moisture_continuity(),
            "process_gate_contract": PROCESS_GATE_CONTRACT, "state_scope": STATE_SCOPE,
        })

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.as_record()).encode("ascii")).hexdigest()

    @classmethod
    def resume(cls, scenario: Scenario, record: Any, adapters: dict[str, Any], *,
               latitude: LatitudeBinding | None = None) -> "Coordinator":
        """Rebuild the temporal cursor; any identity or derived-field mismatch refuses."""
        keys = ("schema", "scenario_sha256", "scenario_id", "native_epoch", "initial_commit", "ledger", "occurrences",
                "next_index", "finished", "adapters", "latitude", "soil_moisture_continuity", "process_gate_contract",
                "state_scope")
        if type(record) is not dict or set(record) != set(keys):
            raise CoordinatorError("malformed_cursor", f"cursor record requires exactly {keys}")
        if record["schema"] != CURSOR_SCHEMA:
            raise CoordinatorError("unsupported_cursor_schema", f"{record['schema']!r} is not {CURSOR_SCHEMA!r}")
        if type(scenario) is not Scenario:
            raise CoordinatorError("invalid_scenario", "scenario must be a masy.scenario.Scenario")
        if record["scenario_sha256"] != scenario.digest:
            raise CoordinatorError("scenario_source_mismatch",
                                   f"cursor binds scenario {record['scenario_sha256']}, the loaded scenario is {scenario.digest} "
                                   "(an event/weather file or the scenario document changed)")
        try:
            initial_commit = decode_time(record["initial_commit"])
            ledger = ClockLedger.from_record(record["ledger"])
        except ClockError as exc:
            raise CoordinatorError("malformed_cursor", str(exc)) from exc
        coordinator = cls(scenario, adapters, latitude=latitude, initial_commit_s=initial_commit)
        recorded_caps = record["adapters"]
        live_caps = {kind: caps.record() for kind, caps in coordinator._capabilities.items()}
        if type(recorded_caps) is not dict or recorded_caps != live_caps:
            raise CoordinatorError("adapter_identity_mismatch", "supplied adapters differ from the recorded capabilities")
        live_lat = None if latitude is None else latitude.record()
        if record["latitude"] != live_lat:
            raise CoordinatorError("latitude_mismatch", "latitude binding differs from the recorded one")
        occurrences = record["occurrences"]
        next_index = record["next_index"]
        if (type(occurrences) is not list or type(next_index) is not int or next_index != len(occurrences)
                or next_index > len(scenario.occurrences)):
            raise CoordinatorError("cursor_inconsistent", "occurrence count, next_index and scenario length disagree")
        entries = ledger.records()
        # Replay the accepted prefix through the SAME sealed-dispatch and validation rules as a live run, without
        # calling any adapter: scheduled starts, gap policy, capability preflight, weather coverage, planned/realized
        # ends, interevent no-commit, commit chain (initial commit for the first entry) and exact ledger metadata.
        for position, realized in enumerate(occurrences):
            occurrence = scenario.occurrences[position]
            if type(realized) is not dict or type(realized.get("evidence_present")) is not bool:
                raise CoordinatorError("cursor_inconsistent", f"{occurrence.occurrence_id}: realized occurrence record malformed")
            index = realized.get("ledger_entry_index")
            if type(index) is not int or not 0 <= index < len(entries):
                raise CoordinatorError("cursor_inconsistent", f"{occurrence.occurrence_id}: ledger entry index invalid")
            entry = entries[index]
            consumed = len(coordinator._ledger.records())
            try:
                sealed = coordinator._sealed_dispatch(with_pet=False)
                outcome = AdapterOutcome(
                    "complete", decode_time(entry["native_end"]), decode_time(entry["commit_before"]),
                    decode_time(entry["commit_after"]),
                    None if entry["returned_elapsed"] is None else decode_time(entry["returned_elapsed"]),
                    evidence=_REPLAYED_EVIDENCE if realized["evidence_present"] else None)
                derived, trial = coordinator._validate(sealed, outcome)
            except CoordinatorError as exc:
                raise CoordinatorError("cursor_inconsistent",
                                       f"{occurrence.occurrence_id}: replay of the accepted prefix refused ({exc.code}: {exc.detail})") from exc
            except ClockError as exc:
                raise CoordinatorError("cursor_inconsistent", f"{occurrence.occurrence_id}: {exc}") from exc
            if canonical_json(derived) != canonical_json(realized):
                raise CoordinatorError("cursor_inconsistent",
                                       f"{occurrence.occurrence_id}: realized record differs from its replayed derivation")
            replayed = trial.records()[consumed:]
            if canonical_json(replayed) != canonical_json(entries[consumed:consumed + len(replayed)]):
                raise CoordinatorError("cursor_inconsistent",
                                       f"{occurrence.occurrence_id}: ledger entries differ from the replayed boundary")
            coordinator._ledger = trial
            coordinator._realized.append(derived)
            coordinator._next += 1
        if len(coordinator._ledger.records()) != len(entries):
            raise CoordinatorError("cursor_inconsistent", "ledger holds entries not referenced by a realized occurrence")
        if canonical_json(coordinator._ledger.as_record()) != canonical_json(record["ledger"]):
            raise CoordinatorError("cursor_inconsistent", "replayed clock ledger differs from the serialized ledger")
        if canonical_json(coordinator.as_record()) != canonical_json(record):
            raise CoordinatorError("cursor_inconsistent", "derived cursor fields differ from the serialized record")
        return coordinator


_REPLAYED_EVIDENCE = object()  # marker: the original adapter evidence is not persisted, only its presence


def _capability_preflight(occurrence: Occurrence, caps: AdapterCapabilities, calendar: CalendarStart,
                          start_g: Fraction, end_g: Fraction) -> dict[str, Any]:
    """Declared duration/calendar capabilities are enforced for EVERY occurrence
    of the adapter's kind, before the adapter is called and independently of
    weather or process gates. Never truncates or shifts the physical duration."""
    whole_days = calendar.whole_days(start_g, end_g)
    if caps.duration_policy == "whole_days_86400_only" and whole_days is None:
        raise CoordinatorError("adapter_duration_policy_violation",
                               f"{occurrence.occurrence_id}: adapter {caps.adapter_id!r} accepts only whole 86400 s days starting "
                               f"at midnight; this {occurrence.kind} occurrence starts {calendar.instant(start_g).record()['iso']} "
                               f"and spans {fraction_text(end_g - start_g)} s (planned); the duration is not truncated")
    dates = calendar.dates_touched(start_g, end_g)
    noleap = None
    if caps.calendar_support == "noleap_365_day_index":
        try:
            noleap = [noleap_day_index(day) for day in dates]
        except CalendarError as exc:
            raise CoordinatorError("noleap_adapter_leap_day",
                                   f"{occurrence.occurrence_id}: adapter {caps.adapter_id!r} uses a 365-day calendar; {exc}") from exc
    return {"dates_touched": [d.isoformat() for d in dates], "whole_days": whole_days, "noleap_day_index": noleap,
            "capability_preflight": {"duration_policy": caps.duration_policy, "calendar_support": caps.calendar_support,
                                     "checked_before_dispatch": True}}


def _bind_adapters(scenario: Scenario, adapters: Any) -> tuple[dict[str, Any], dict[str, AdapterCapabilities]]:
    if type(adapters) is not dict:
        raise CoordinatorError("adapter_missing", "adapters must be a dict kind -> adapter object")
    present = scenario.kinds_present
    missing = sorted(present - set(adapters))
    if missing:
        raise CoordinatorError("adapter_missing", f"scenario needs adapters for {missing}; none supplied (no stub is substituted)")
    extra = sorted(set(adapters) - present)
    if extra:
        raise CoordinatorError("adapter_unused", f"adapters supplied for kinds not in the scenario: {extra}")
    bound: dict[str, Any] = {}
    capabilities: dict[str, AdapterCapabilities] = {}
    for kind in sorted(present):
        adapter = adapters[kind]
        if not callable(getattr(adapter, "capabilities", None)) or not callable(getattr(adapter, "run", None)):
            raise CoordinatorError("invalid_adapter", f"adapter for {kind!r} must provide capabilities() and run(dispatch)")
        caps = adapter.capabilities()
        if type(caps) is not AdapterCapabilities:
            raise CoordinatorError("invalid_capability", f"adapter for {kind!r} returned {type(caps).__name__}, not AdapterCapabilities")
        if caps.kind != kind:
            raise CoordinatorError("invalid_capability", f"adapter bound as {kind!r} declares kind {caps.kind!r}")
        if caps.native_staging != "staged_until_accepted":
            raise CoordinatorError("adapter_staging_required",
                                   f"the {kind} adapter {caps.adapter_id!r} declares native_staging={caps.native_staging!r}; the "
                                   "coordinator's cursor atomicity cannot roll back a native model mutated in place, so adapters "
                                   "must stage native state until the boundary is accepted")
        if scenario.requires_continuous_soil_moisture and not caps.continuous_soil_moisture:
            raise CoordinatorError("continuous_soil_moisture_unavailable",
                                   f"scenario requires continuous soil moisture but the {kind} adapter {caps.adapter_id!r} "
                                   "declares a dry-reset / non-continuous handoff")
        bound[kind] = adapter
        capabilities[kind] = caps
    return bound, capabilities
