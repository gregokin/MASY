"""Dated, source-bound scenario: an ordered list of event-file occurrences (data only).

A scenario document (`masy.scenario.v1`) references reusable event files
(`masy.event_files`) by relative path plus SHA-256 and gives each use a
distinct `occurrence_id`, e.g. interevent1, wind1, interevent2, rain1, ...
Any order of kinds is accepted; nothing requires alternation. The scenario
carries the explicit dated calendar start (`masy.calendar`), the native epoch
of the shared bed clock, the optional shared weather document and the
scheduling policy:

- **Relative sequential dispatch (default).** Each occurrence starts at the
  REALIZED native end of the previous one, including rain recession and any
  settlement time the owning model counted. Nothing here predicts a realized
  end.
- **Explicit `scheduled_start`.** An occurrence may carry a calendar
  date-time. It is honoured only when it does not collide: at dispatch the
  coordinator refuses a scheduled start earlier than the realized previous
  end (`scheduled_start_overlap`); forcing is never truncated or shifted. A
  gap before a scheduled start is refused by default (`gap_policy =
  "refuse"`) or recorded as an explicit no-process `idle` interval
  (`gap_policy = "idle"`); it never runs evaporation, transpiration or growth
  silently. Statically, a scheduled start may not precede the minimum
  possible end of everything before it (wind/interevent durations are exact,
  rain contributes its rainfall duration; recession is unknown).

Identity: `digest` is the SHA-256 of the canonical normalized record, which
includes every event file's byte hash and lossless event record, the weather
hash and the calendar; changing one byte of any referenced file changes the
digest. Loading never touches native models; latitude is NOT a scenario field
(see `masy.weather.LatitudeBinding`).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from fractions import Fraction
from pathlib import Path
from typing import Any

from masy.calendar import CalendarError, CalendarStart, fraction_text
from masy.clock import ClockError, ClockLedger, encode_time
from masy.event_files import EventFile, EventFileError, IntereventFile, parse_event_file
from masy.scenario_io import IOLimits, ScenarioIOError, canonical_json, load_document, relative_reference
from masy.weather import WeatherError, WeatherSeries

__all__ = [
    "GAP_POLICIES",
    "SCENARIO_SCHEMA",
    "Occurrence",
    "Scenario",
    "ScenarioError",
    "load_scenario",
]

SCENARIO_SCHEMA = "masy.scenario.v1"
GAP_POLICIES = ("refuse", "idle")
_TOP_KEYS = ("schema", "scenario_id", "calendar", "native_epoch_s", "gap_policy", "requires_continuous_soil_moisture",
             "weather", "events")
_EVENT_KEYS = ("occurrence_id", "file", "sha256", "scheduled_start")
_REF_KEYS = ("file", "sha256")
_OCCURRENCE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


class ScenarioError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _closed(value: Any, names: tuple[str, ...], where: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(names):
        raise ScenarioError("invalid_keys", f"{where} requires exactly the keys {names}")
    return value


def _sha(value: Any, where: str) -> str:
    if type(value) is not str or not _SHA.match(value):
        raise ScenarioError("invalid_digest", f"{where} must be 64 lowercase hex digits")
    return value


@dataclass(frozen=True)
class Occurrence:
    index: int
    occurrence_id: str
    file: str
    file_sha256: str
    event: EventFile
    scheduled_start: str | None
    scheduled_start_s: Fraction | None

    @property
    def kind(self) -> str:
        return self.event.kind

    def record(self) -> dict[str, Any]:
        return {"index": self.index, "occurrence_id": self.occurrence_id, "kind": self.kind, "file": self.file,
                "file_sha256": self.file_sha256, "event_digest": self.event.digest, "event": self.event.record(),
                "scheduled_start": self.scheduled_start,
                "scheduled_start_s": None if self.scheduled_start_s is None else fraction_text(self.scheduled_start_s)}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    calendar: CalendarStart
    native_epoch_s: float
    gap_policy: str
    requires_continuous_soil_moisture: bool
    occurrences: tuple[Occurrence, ...]
    weather: WeatherSeries | None
    weather_ref: Mapping[str, str] | None
    source_sha256: str | None = None
    base_dir: str | None = None

    def __post_init__(self) -> None:
        if (self.weather is None) != (self.weather_ref is None):
            raise ScenarioError("invalid_keys", "weather and weather_ref must be given together")
        if self.weather_ref is not None:
            if not isinstance(self.weather_ref, Mapping) or set(self.weather_ref) != {"file", "sha256"}:
                raise ScenarioError("invalid_keys", "weather_ref requires exactly file and sha256")
            object.__setattr__(self, "weather_ref", MappingProxyType(dict(self.weather_ref)))  # read-only
        if type(self.occurrences) is not tuple or not self.occurrences:
            raise ScenarioError("empty_scenario", "a scenario needs at least one occurrence")
        ids = [o.occurrence_id for o in self.occurrences]
        if len(set(ids)) != len(ids):
            raise ScenarioError("duplicate_occurrence_id", "occurrence ids must be unique")
        if any(o.index != i for i, o in enumerate(self.occurrences)):
            raise ScenarioError("invalid_occurrence_index", "occurrence indices must be 0..n-1 in order")
        if self.gap_policy not in GAP_POLICIES:
            raise ScenarioError("invalid_gap_policy", f"gap_policy must be one of {GAP_POLICIES}")
        try:
            ClockLedger(self.native_epoch_s)
        except ClockError as exc:
            raise ScenarioError("invalid_native_epoch", str(exc)) from exc
        self._static_checks()

    def _static_checks(self) -> None:
        needs_weather = any(type(o.event) is IntereventFile and any(o.event.processes.values()) for o in self.occurrences)
        if needs_weather and self.weather is None:
            raise ScenarioError("weather_required",
                                "an interevent occurrence enables evaporation/transpiration/growth; a weather document is required")
        if self.weather is not None:
            if self.weather.utc_offset != self.calendar.offset_text:
                raise ScenarioError("weather_offset_mismatch",
                                    f"weather utc_offset {self.weather.utc_offset} differs from calendar {self.calendar.offset_text}")
            if self.weather.first_date > self.calendar.start.date():
                raise ScenarioError("weather_starts_after_scenario",
                                    f"weather begins {self.weather.first_date.isoformat()}, after the scenario start")
        lower = Fraction(0)
        for occurrence in self.occurrences:
            if occurrence.scheduled_start_s is not None:
                if occurrence.scheduled_start_s < lower:
                    raise ScenarioError("scheduled_start_before_minimum_end",
                                        f"{occurrence.occurrence_id}: scheduled start {occurrence.scheduled_start} precedes the "
                                        f"minimum possible end of the preceding occurrences ({fraction_text(lower)} s)")
                lower = occurrence.scheduled_start_s
            lower += occurrence.event.minimum_duration_s

    @property
    def kinds_present(self) -> frozenset[str]:
        return frozenset(o.kind for o in self.occurrences)

    def record(self) -> dict[str, Any]:
        return {"schema": SCENARIO_SCHEMA, "scenario_id": self.scenario_id, "calendar": self.calendar.record(),
                "native_epoch_s": encode_time(self.native_epoch_s), "gap_policy": self.gap_policy,
                "requires_continuous_soil_moisture": self.requires_continuous_soil_moisture,
                "weather": None if self.weather is None else {**self.weather_ref, "weather_digest": self.weather.digest,
                                                                 "pet_method": self.weather.pet_method_name},
                "events": [o.record() for o in self.occurrences], "source_sha256": self.source_sha256}

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.record()).encode("ascii")).hexdigest()


def _parse_scenario(value: Any, base_dir: Path, limits: IOLimits, *, max_events: int, source_sha256: str | None,
                    where: str) -> Scenario:
    _closed(value, _TOP_KEYS, where)
    if value["schema"] != SCENARIO_SCHEMA:
        raise ScenarioError("invalid_schema", f"{where}.schema must be {SCENARIO_SCHEMA!r}")
    scenario_id = value["scenario_id"]
    if type(scenario_id) is not str or not _OCCURRENCE_ID.match(scenario_id):
        raise ScenarioError("invalid_scenario_id", f"{where}.scenario_id must match {_OCCURRENCE_ID.pattern}")
    try:
        calendar = CalendarStart.from_input(value["calendar"], f"{where}.calendar")
    except CalendarError as exc:
        raise ScenarioError(exc.code, str(exc)) from exc
    epoch = value["native_epoch_s"]
    if type(epoch) not in (int, float) or (type(epoch) is int and float(epoch) != epoch) or not math.isfinite(float(epoch)):
        raise ScenarioError("invalid_native_epoch", f"{where}.native_epoch_s must be a finite plain number")
    epoch = float(epoch)
    gap_policy = value["gap_policy"]
    if gap_policy not in GAP_POLICIES:
        raise ScenarioError("invalid_gap_policy", f"{where}.gap_policy must be one of {GAP_POLICIES}")
    requires = value["requires_continuous_soil_moisture"]
    if type(requires) is not bool:
        raise ScenarioError("invalid_keys", f"{where}.requires_continuous_soil_moisture must be a bool")
    rows = value["events"]
    if type(max_events) is not int or max_events < 1:
        raise ScenarioError("invalid_limit", "max_events must be a positive plain int")
    if type(rows) is not list or not 1 <= len(rows) <= max_events:
        raise ScenarioError("event_limit", f"{where}.events must hold 1..{max_events} occurrences")

    cache: dict[Path, tuple[EventFile, str]] = {}

    def load_reference(ref: dict[str, Any], label: str) -> tuple[Path, str]:
        try:
            path = relative_reference(ref["file"], base_dir, limits, f"{label}.file")
        except ScenarioIOError as exc:
            raise ScenarioError(exc.code, str(exc)) from exc
        return path, _sha(ref["sha256"], f"{label}.sha256")

    weather = weather_ref = None
    if value["weather"] is not None:
        ref = _closed(value["weather"], _REF_KEYS, f"{where}.weather")
        path, declared = load_reference(ref, f"{where}.weather")
        try:
            document, pin = load_document(path, limits, where=f"{where}.weather -> {ref['file']}")
        except ScenarioIOError as exc:
            raise ScenarioError(exc.code, str(exc)) from exc
        if pin["sha256"] != declared:
            raise ScenarioError("weather_source_hash_mismatch",
                                f"{where}.weather: file {ref['file']} has sha256 {pin['sha256']}, declared {declared}")
        try:
            weather = WeatherSeries.from_input(document, where=f"{where}.weather -> {ref['file']}", source_sha256=pin["sha256"])
        except WeatherError as exc:
            raise ScenarioError(exc.code, str(exc)) from exc
        weather_ref = {"file": ref["file"], "sha256": declared}

    occurrences: list[Occurrence] = []
    for index, row in enumerate(rows):
        label = f"{where}.events[{index}]"
        _closed(row, _EVENT_KEYS, label)
        occurrence_id = row["occurrence_id"]
        if type(occurrence_id) is not str or not _OCCURRENCE_ID.match(occurrence_id):
            raise ScenarioError("invalid_occurrence_id", f"{label}.occurrence_id must match {_OCCURRENCE_ID.pattern}")
        path, declared = load_reference(row, label)
        if path not in cache:
            if len(cache) >= limits.max_files:
                raise ScenarioError("file_limit", f"more than {limits.max_files} distinct event files")
            try:
                document, pin = load_document(path, limits, where=f"{label} -> {row['file']}")
            except ScenarioIOError as exc:
                raise ScenarioError(exc.code, str(exc)) from exc
            try:
                event = parse_event_file(document, where=f"{label} -> {row['file']}")
            except EventFileError as exc:
                raise ScenarioError(exc.code, str(exc)) from exc
            cache[path] = (event, pin["sha256"])
        event, actual = cache[path]
        if actual != declared:
            raise ScenarioError("event_source_hash_mismatch",
                                f"{label}: file {row['file']} has sha256 {actual}, declared {declared}")
        scheduled = row["scheduled_start"]
        scheduled_s = None
        if scheduled is not None:
            try:
                scheduled_s = calendar.elapsed_of(scheduled, f"{label}.scheduled_start")
            except CalendarError as exc:
                raise ScenarioError(exc.code, str(exc)) from exc
        occurrences.append(Occurrence(index, occurrence_id, row["file"], declared, event, scheduled, scheduled_s))
    return Scenario(scenario_id, calendar, epoch, gap_policy, requires, tuple(occurrences), weather, weather_ref,
                    source_sha256, str(base_dir))


def load_scenario(path: Path, limits: IOLimits | None = None, *, max_events: int = 4096) -> Scenario:
    """Strictly load a scenario document and every referenced event/weather file."""
    limits = IOLimits() if limits is None else limits
    path = Path(path)
    try:
        value, pin = load_document(path, limits, where=str(path))
    except ScenarioIOError as exc:
        raise ScenarioError(exc.code, str(exc)) from exc
    return _parse_scenario(value, path.parent, limits, max_events=max_events, source_sha256=pin["sha256"], where=str(path))
