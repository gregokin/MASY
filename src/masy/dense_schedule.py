"""Ordered sequential Dense wind/rain event schedule (data only).

Inputs are relative durations and forcing ONLY. A schedule is an ordered
list of events, each `wind` (one duration with one constant forcing for this
initial scope: shear velocity m/s and direction in degrees clockwise from
north) or `rain` (piecewise positive interval durations with intensities in
mm/h, plus a finite positive recession guard). Any sequence of kinds is
accepted; nothing requires alternation.

Absolute or calendar times are refused: no input carries a native start/end,
an epoch, a date or a timestamp (`absolute_time_refused`). Native times are
execution evidence, resolved per event by `resolve_event` from the realized
native start of that event (the native epoch for event 0) through the
existing `masy.clock.resolve_forcing_edges`. The intended duration record
and the resolved native edges are kept side by side; nothing is retimed or
rescaled, and the realized end of a rain event (after physical recession) is
never predicted here.

Durations and forcing values are plain JSON numbers: bool, NaN, +-inf, -0.0,
numpy scalars and integers not exactly representable as binary64 refuse.
The normalized record stores every value as binary64 hex plus its exact
rational (`masy.clock.encode_time`), so `from_record(as_record())` is
lossless and the digest is stable. Importing this module imports neither
numpy nor any native model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from masy.clock import ClockError, decode_time, encode_time, resolve_forcing_edges

__all__ = [
    "CASE_SCHEDULE_SCHEMA",
    "EVENT_KINDS",
    "RESOLVED_EVENT_SCHEMA",
    "SCHEDULE_SCHEMA",
    "DenseSchedule",
    "DenseScheduleError",
    "RainEvent",
    "RainInterval",
    "WindEvent",
    "case_schedule_digest",
    "check_resolved_event",
    "resolve_event",
]

SCHEDULE_SCHEMA = "masy.dense_schedule.v1"
RESOLVED_EVENT_SCHEMA = "masy.dense_resolved_event.v1"
CASE_SCHEDULE_SCHEMA = "masy.dense_case_schedule.v1"
EVENT_KINDS = ("wind", "rain")

_TOP_KEYS = frozenset({"schema", "events"})
_WIND_KEYS = frozenset({"kind", "duration_s", "shear_velocity_m_s", "direction_deg_cw_from_north"})
_RAIN_KEYS = frozenset({"kind", "intervals", "recession_guard_s"})
_INTERVAL_KEYS = frozenset({"duration_s", "intensity_mm_h"})
_WIND_RECORD_KEYS = _WIND_KEYS | {"index"}
_RAIN_RECORD_KEYS = _RAIN_KEYS | {"index", "rainfall_duration_s"}
_RESOLVED_KEYS = frozenset({"schema", "schedule_sha256", "index", "kind", "intended", "edge_roles", "resolved_edges"})
# Any key mentioning one of these tokens claims an absolute/calendar time.
_ABSOLUTE_TOKENS = ("start", "end", "epoch", "date", "calendar", "timestamp", "utc", "timezone", "clock",
                    "native", "time_s", "at_s")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DenseScheduleError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _number(value: Any, where: str) -> float:
    """A plain finite JSON number (int exactly representable, or float); never bool/-0.0/numpy."""
    if type(value) not in (int, float):
        raise DenseScheduleError("invalid_number", f"{where} must be a plain int or float, got {type(value).__name__}")
    if type(value) is int:
        converted = float(value)
        if converted != value:
            raise DenseScheduleError("invalid_number", f"{where}={value} is not exactly representable as binary64")
        value = converted
    if not math.isfinite(value):
        raise DenseScheduleError("nonfinite_value", f"{where} must be finite, got {value!r}")
    if value == 0 and math.copysign(1.0, value) < 0:
        raise DenseScheduleError("negative_zero", f"{where} is -0.0; refused rather than rewritten")
    return value


def _duration(value: Any, where: str) -> float:
    value = _number(value, where)
    if value <= 0:
        raise DenseScheduleError("nonpositive_duration", f"{where} must be > 0 (empty/collapsed refused), got {value!r}")
    return value


def _nonnegative(value: Any, where: str) -> float:
    value = _number(value, where)
    if value < 0:
        raise DenseScheduleError("negative_value", f"{where} must be >= 0, got {value!r}")
    return value


def _direction(value: Any, where: str) -> float:
    value = _number(value, where)
    if not 0.0 <= value < 360.0:
        raise DenseScheduleError("invalid_direction", f"{where} must lie in [0, 360) degrees, got {value!r}")
    return value


@dataclass(frozen=True)
class WindEvent:
    """One wind event: a duration under one constant forcing (initial scope)."""

    duration_s: float
    shear_velocity_m_s: float
    direction_deg_cw_from_north: float

    kind = "wind"

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration_s", _duration(self.duration_s, "wind.duration_s"))
        object.__setattr__(self, "shear_velocity_m_s",
                           _nonnegative(self.shear_velocity_m_s, "wind.shear_velocity_m_s"))
        object.__setattr__(self, "direction_deg_cw_from_north",
                           _direction(self.direction_deg_cw_from_north, "wind.direction_deg_cw_from_north"))


@dataclass(frozen=True)
class RainInterval:
    duration_s: float
    intensity_mm_h: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration_s", _duration(self.duration_s, "rain.interval.duration_s"))
        object.__setattr__(self, "intensity_mm_h", _nonnegative(self.intensity_mm_h, "rain.interval.intensity_mm_h"))


@dataclass(frozen=True)
class RainEvent:
    """Piecewise-constant rainfall followed by physical recession, bounded by
    a finite positive `recession_guard_s` after the rainfall ends. Reaching
    the guard without native completion is incomplete, never complete."""

    intervals: tuple[RainInterval, ...]
    recession_guard_s: float

    kind = "rain"

    def __post_init__(self) -> None:
        if type(self.intervals) is not tuple or not self.intervals:
            raise DenseScheduleError("invalid_rain_intervals", "rain.intervals must be a non-empty tuple")
        if any(type(interval) is not RainInterval for interval in self.intervals):
            raise DenseScheduleError("invalid_rain_intervals", "rain.intervals must contain RainInterval objects")
        if not any(interval.intensity_mm_h > 0 for interval in self.intervals):
            raise DenseScheduleError("dry_rain_event", "a rain event needs at least one positive intensity")
        object.__setattr__(self, "recession_guard_s", _duration(self.recession_guard_s, "rain.recession_guard_s"))

    @property
    def rainfall_duration_s(self) -> Fraction:
        """Exact sum of the interval durations (never a float accumulation)."""
        return sum((Fraction(interval.duration_s) for interval in self.intervals), Fraction(0))


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _exact_decimal(value: Fraction) -> str:
    """Exact plain decimal of a non-negative dyadic rational (every binary64 sum is one)."""
    numerator, denominator = value.numerator, value.denominator
    k = denominator.bit_length() - 1
    if numerator < 0 or denominator != 1 << k:
        raise DenseScheduleError("internal_error", f"{value} is not a non-negative dyadic rational")
    digits = str(numerator * 5 ** k)
    if k == 0:
        return digits
    digits = digits.rjust(k + 1, "0")
    whole, fraction = digits[:-k], digits[-k:].rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _check_keys(table: Any, allowed: frozenset[str], where: str) -> dict[str, Any]:
    if not isinstance(table, dict):
        raise DenseScheduleError("malformed_schedule", f"{where} must be an object, got {type(table).__name__}")
    for key in table:
        if not isinstance(key, str):
            raise DenseScheduleError("malformed_schedule", f"{where} has a non-string key {key!r}")
    unknown = set(table) - allowed
    absolute = sorted(key for key in unknown if any(token in key.lower() for token in _ABSOLUTE_TOKENS))
    if absolute:
        raise DenseScheduleError("absolute_time_refused",
                                 f"{where} carries absolute/calendar/native time field(s) {absolute}; inputs are "
                                 "relative durations only")
    if unknown:
        raise DenseScheduleError("unknown_key", f"{where} has unsupported key(s) {sorted(unknown)}")
    missing = allowed - set(table)
    if missing:
        raise DenseScheduleError("missing_key", f"{where} lacks required key(s) {sorted(missing)}")
    return table


def _event_kind(table: dict[str, Any], where: str) -> str:
    kind = table.get("kind") if isinstance(table, dict) else None
    if not isinstance(kind, str) or kind not in EVENT_KINDS:
        raise DenseScheduleError("invalid_event_kind", f"{where}.kind must be one of {list(EVENT_KINDS)}, got {kind!r}")
    return kind


@dataclass(frozen=True)
class DenseSchedule:
    """Immutable ordered event list. Construct with `from_input` (plain JSON
    numbers) or `from_record` (normalized lossless record)."""

    events: tuple[WindEvent | RainEvent, ...]

    def __post_init__(self) -> None:
        if type(self.events) is not tuple or not self.events:
            raise DenseScheduleError("empty_schedule", "a schedule needs at least one event")
        if any(type(event) not in (WindEvent, RainEvent) for event in self.events):
            raise DenseScheduleError("invalid_event", "events must be WindEvent or RainEvent objects")

    @classmethod
    def from_input(cls, raw: Any) -> "DenseSchedule":
        table = _check_keys(raw, _TOP_KEYS, "schedule")
        if table["schema"] != SCHEDULE_SCHEMA:
            raise DenseScheduleError("unsupported_schema", f"schema {table['schema']!r:.100} is not {SCHEDULE_SCHEMA!r}")
        events = table["events"]
        if not isinstance(events, list) or not events:
            raise DenseScheduleError("empty_schedule", "schedule.events must be a non-empty list")
        parsed: list[WindEvent | RainEvent] = []
        for index, event in enumerate(events):
            where = f"events[{index}]"
            if not isinstance(event, dict):
                raise DenseScheduleError("malformed_schedule", f"{where} must be an object")
            kind = _event_kind(event, where)
            if kind == "wind":
                _check_keys(event, _WIND_KEYS, where)
                parsed.append(WindEvent(event["duration_s"], event["shear_velocity_m_s"],
                                        event["direction_deg_cw_from_north"]))
            else:
                _check_keys(event, _RAIN_KEYS, where)
                intervals = event["intervals"]
                if not isinstance(intervals, list) or not intervals:
                    raise DenseScheduleError("invalid_rain_intervals", f"{where}.intervals must be a non-empty list")
                parsed.append(RainEvent(
                    tuple(RainInterval(**_check_keys(item, _INTERVAL_KEYS, f"{where}.intervals[{i}]"))
                          for i, item in enumerate(intervals)),
                    event["recession_guard_s"],
                ))
        return cls(tuple(parsed))

    def as_input(self) -> dict[str, Any]:
        """Plain-number input form; `from_input(as_input())` equals self."""
        events: list[dict[str, Any]] = []
        for event in self.events:
            if type(event) is WindEvent:
                events.append({"kind": "wind", "duration_s": event.duration_s,
                               "shear_velocity_m_s": event.shear_velocity_m_s,
                               "direction_deg_cw_from_north": event.direction_deg_cw_from_north})
            else:
                events.append({"kind": "rain", "recession_guard_s": event.recession_guard_s,
                               "intervals": [{"duration_s": i.duration_s, "intensity_mm_h": i.intensity_mm_h}
                                             for i in event.intervals]})
        return {"schema": SCHEDULE_SCHEMA, "events": events}

    def as_record(self) -> dict[str, Any]:
        """Normalized lossless plain-JSON record (binary64 hex + exact fraction)."""
        events: list[dict[str, Any]] = []
        for index, event in enumerate(self.events):
            if type(event) is WindEvent:
                events.append({
                    "index": index, "kind": "wind", "duration_s": encode_time(event.duration_s),
                    "shear_velocity_m_s": encode_time(event.shear_velocity_m_s),
                    "direction_deg_cw_from_north": encode_time(event.direction_deg_cw_from_north),
                })
            else:
                events.append({
                    "index": index, "kind": "rain",
                    "intervals": [{"duration_s": encode_time(i.duration_s),
                                   "intensity_mm_h": encode_time(i.intensity_mm_h)} for i in event.intervals],
                    "rainfall_duration_s": _fraction_text(event.rainfall_duration_s),
                    "recession_guard_s": encode_time(event.recession_guard_s),
                })
        return {"schema": SCHEDULE_SCHEMA, "events": events}

    @classmethod
    def from_record(cls, record: Any) -> "DenseSchedule":
        """Strictly rebuild a normalized record; every derived field must equal
        its recomputed value and the canonical JSON must match exactly."""
        table = _check_keys(record, _TOP_KEYS, "schedule record")
        if table["schema"] != SCHEDULE_SCHEMA:
            raise DenseScheduleError("unsupported_schema", f"schema {table['schema']!r:.100} is not {SCHEDULE_SCHEMA!r}")
        events = table["events"]
        if not isinstance(events, list) or not events:
            raise DenseScheduleError("empty_schedule", "schedule record events must be a non-empty list")
        parsed: list[WindEvent | RainEvent] = []
        try:
            for index, event in enumerate(events):
                where = f"events[{index}]"
                kind = _event_kind(event, where)
                if kind == "wind":
                    _check_keys(event, _WIND_RECORD_KEYS, where)
                    parsed.append(WindEvent(decode_time(event["duration_s"]), decode_time(event["shear_velocity_m_s"]),
                                            decode_time(event["direction_deg_cw_from_north"])))
                else:
                    _check_keys(event, _RAIN_RECORD_KEYS, where)
                    if not isinstance(event["intervals"], list):
                        raise DenseScheduleError("invalid_rain_intervals", f"{where}.intervals must be a list")
                    intervals = tuple(
                        RainInterval(decode_time(item["duration_s"]), decode_time(item["intensity_mm_h"]))
                        for item in (_check_keys(i, _INTERVAL_KEYS, f"{where}.intervals[]") for i in event["intervals"]))
                    parsed.append(RainEvent(intervals, decode_time(event["recession_guard_s"])))
        except ClockError as exc:
            raise DenseScheduleError("malformed_schedule", f"invalid time/value record ({exc})") from exc
        schedule = cls(tuple(parsed))
        try:
            same = _canonical(schedule.as_record()) == _canonical(record)
        except (TypeError, ValueError) as exc:
            raise DenseScheduleError("malformed_schedule", f"record is not plain JSON data ({exc})") from exc
        if not same:
            raise DenseScheduleError("inconsistent_schedule_record", "derived fields differ from the serialized record")
        return schedule

    @property
    def digest(self) -> str:
        """SHA-256 of the canonical normalized record."""
        return hashlib.sha256(_canonical(self.as_record()).encode()).hexdigest()


def _event_edges(event: WindEvent | RainEvent) -> tuple[list[Fraction], list[str]]:
    if type(event) is WindEvent:
        return [Fraction(0), Fraction(event.duration_s)], ["event_start", "wind_end"]
    offsets, roles, total = [Fraction(0)], ["event_start"], Fraction(0)
    for i, interval in enumerate(event.intervals):
        total += Fraction(interval.duration_s)
        offsets.append(total)
        roles.append(f"rain_interval_{i}_end")
    roles[-1] = "rainfall_end"
    offsets.append(total + Fraction(event.recession_guard_s))
    roles.append("recession_guard_end")
    return offsets, roles


def resolve_event(schedule: DenseSchedule, index: int, native_start_s: float) -> dict[str, Any]:
    """Resolution evidence for event `index` starting at the actual native
    time `native_start_s` (the native epoch for event 0; afterwards the
    realized native end of the preceding event, recorded as returned).

    Edges are exact cumulative offsets passed to
    `masy.clock.resolve_forcing_edges` as exact decimals; its intended and
    native records are returned unchanged next to the intended input
    record. Collapsed or non-finite native edges raise `ClockError`."""
    if type(schedule) is not DenseSchedule:
        raise DenseScheduleError("invalid_schedule", "schedule must be a DenseSchedule")
    if type(index) is not int or not 0 <= index < len(schedule.events):
        raise DenseScheduleError("invalid_event_index", f"index {index!r} outside 0..{len(schedule.events) - 1}")
    event = schedule.events[index]
    offsets, roles = _event_edges(event)
    resolved = resolve_forcing_edges(native_start_s, [_exact_decimal(offset) for offset in offsets])
    return {
        "schema": RESOLVED_EVENT_SCHEMA,
        "schedule_sha256": schedule.digest,
        "index": index,
        "kind": event.kind,
        "intended": schedule.as_record()["events"][index],
        "edge_roles": roles,
        "resolved_edges": resolved,
    }


def check_resolved_event(record: Any, schedule: DenseSchedule) -> dict[str, Any]:
    """Re-derive a serialized resolution record from its own native start and
    the schedule; any difference (or a foreign schedule) refuses."""
    if type(schedule) is not DenseSchedule:
        raise DenseScheduleError("invalid_schedule", "schedule must be a DenseSchedule")
    if not isinstance(record, dict) or set(record) != _RESOLVED_KEYS:
        raise DenseScheduleError("malformed_resolved_event", f"keys must be exactly {sorted(_RESOLVED_KEYS)}")
    table = record
    if table["schema"] != RESOLVED_EVENT_SCHEMA or table["schedule_sha256"] != schedule.digest:
        raise DenseScheduleError("foreign_resolved_event", "record schema or schedule digest does not match")
    try:
        start = decode_time(table["resolved_edges"]["native_start"])
    except (ClockError, KeyError, TypeError) as exc:
        raise DenseScheduleError("malformed_resolved_event", f"native start unreadable ({exc})") from exc
    expected = resolve_event(schedule, table["index"], start)
    try:
        same = _canonical(expected) == _canonical(record)
    except (TypeError, ValueError) as exc:
        raise DenseScheduleError("malformed_resolved_event", f"record is not plain JSON data ({exc})") from exc
    if not same:
        raise DenseScheduleError("inconsistent_resolved_event", "record differs from its re-derived resolution")
    return expected


def case_schedule_digest(case_static_sha256: str, schedule: DenseSchedule) -> str:
    """Combined identity of an immutable static case digest (see
    `masy.dense_bed`) and a schedule; evolving bed state is never part of it."""
    if not isinstance(case_static_sha256, str) or not _SHA256.match(case_static_sha256):
        raise DenseScheduleError("invalid_digest", "case_static_sha256 must be 64 lowercase hex characters")
    if type(schedule) is not DenseSchedule:
        raise DenseScheduleError("invalid_schedule", "schedule must be a DenseSchedule")
    payload = {"schema": CASE_SCHEDULE_SCHEMA, "case_static_sha256": case_static_sha256,
               "schedule_sha256": schedule.digest}
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()
