"""Strict readers for the NEW opt-in daily weather / daily scenario CSV formats (metadata only).

Formats (`masy.daily_weather.v1`, `masy.daily_scenario.v1`) are UTF-8 documents:
a `# key: value` preamble followed by one ordinary CSV table. They are read with
the standard library, bounded before any graph is built, and bound to the exact
bytes of every referenced file (weather CSV, wind YAML) by automatically
computed SHA-256 values; no hash is ever authored by hand.

This module only validates and records declarations. It contains no PET
equation, never imports NumPy, PyYAML (except lazily for wind YAML through
`masy.scenario_io.parse_strict_yaml`), MAPLE, SYRUP or LEAF, runs no model and
makes no physical claim. Wind YAML is the native MAPLE `wind_forcing` record;
its scalars are validated with the existing `masy.event_files.WindEventFile` /
`masy.case_schedule.WindDeclaration` rules. Native runtime validation of that
record remains a separate adapter qualification.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from masy.case_schedule import WindDeclaration
from masy.clock import encode_time
from masy.event_files import WIND_SCHEMA, EventFileError, WindEventFile
from masy.scenario_io import (IOLimits, ScenarioIOError, canonical_json, parse_strict_yaml, read_bounded,
                              relative_reference)

__all__ = [
    "DEFAULT_LIMITS",
    "EXPLICIT_COLUMNS",
    "GROWTH_INTERVAL_RANGE",
    "HARGREAVES_COLUMNS",
    "IDENTITY_SCHEMA",
    "PET_EXPLICIT",
    "PET_HARGREAVES",
    "SCENARIO_COLUMNS",
    "SCENARIO_FORMAT",
    "WEATHER_FORMAT",
    "DailyCsvError",
    "DailyCsvLimits",
    "DailyScenario",
    "DailySettings",
    "DailyWeatherDay",
    "DailyWeatherTable",
    "WindOccurrence",
    "load_daily_scenario",
    "parse_weather_csv",
    "parse_wind_yaml",
]

WEATHER_FORMAT = "masy.daily_weather.v1"
SCENARIO_FORMAT = "masy.daily_scenario.v1"
IDENTITY_SCHEMA = "masy.daily_source_identity.v1"
PET_HARGREAVES = "ecohyd_hargreaves"
PET_EXPLICIT = "explicit"
HARGREAVES_COLUMNS = ("date", "tmin_c", "tmean_c", "tmax_c", "precipitation_mm", "precipitation_intensity_mm_h")
EXPLICIT_COLUMNS = ("date", "tmean_c", "pet_mm_day", "precipitation_mm", "precipitation_intensity_mm_h")
SCENARIO_COLUMNS = ("date", "event_type", "event_file")
# LEAF's existing qualified daily-to-weekly growth range; never widened here.
GROWTH_INTERVAL_RANGE = (1, 7)

_WEATHER_KEYS = ("format", "pet_method", "precipitation_threshold_mm", "source", "description")
_WEATHER_REQUIRED = ("format", "pet_method", "precipitation_threshold_mm")
_SCENARIO_KEYS = ("format", "scenario_id", "weather_file", "start_date", "end_date", "evaporation_enabled",
                  "transpiration_enabled", "growth_enabled", "growth_interval_days", "output_interval_days",
                  "checkpoint_interval_days", "output_at_event_boundaries")
_CONTROL = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")
_META = re.compile(r"# ([a-z][a-z0-9_]*): (\S(?:.*\S)?)")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_NUMBER = re.compile(r"-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_INTERVAL = re.compile(r"[1-9][0-9]{0,8}")
_SCENARIO_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_MAX_NUMBER_CHARS = 64


class DailyCsvError(ScenarioIOError):
    """Stable-coded refusal (`.code`, `.detail`); also used for wrapped I/O refusals."""


@dataclass(frozen=True)
class DailyCsvLimits:
    """Explicit positive-integer input quotas (input bounds, not a process memory theorem)."""

    max_file_bytes: int = 16 << 20
    max_weather_rows: int = 100000
    max_scenario_rows: int = 100000
    max_files: int = 256
    max_total_bytes: int = 256 << 20
    max_header_lines: int = 32
    max_header_chars: int = 1024
    max_line_chars: int = 8192
    max_cell_chars: int = 4096
    max_path_chars: int = 4096

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if type(value) is not int or value < 1:
                raise DailyCsvError("invalid_limit", f"DailyCsvLimits.{name} must be a positive plain int")

    def io_limits(self) -> IOLimits:
        return IOLimits(max_file_bytes=self.max_file_bytes, max_path_chars=self.max_path_chars,
                        max_files=self.max_files)


DEFAULT_LIMITS = DailyCsvLimits()


# ----------------------------------------------------------------------------- text/CSV layer

def _split_document(data: bytes, limits: DailyCsvLimits, where: str, allowed: tuple[str, ...],
                    required: tuple[str, ...], max_rows: int) -> tuple[dict[str, str], list[str]]:
    if type(data) is not bytes:
        raise DailyCsvError("invalid_input", f"{where} requires bytes")
    if len(data) > limits.max_file_bytes:
        raise DailyCsvError("document_too_large", f"{where} exceeds {limits.max_file_bytes} bytes")
    if data.count(b"\n") > max_rows + limits.max_header_lines + 2:
        raise DailyCsvError("row_quota_exceeded", f"{where} has more than {max_rows} table rows")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DailyCsvError("invalid_encoding", f"{where} is not UTF-8: {exc}") from exc
    if text.startswith("﻿"):
        raise DailyCsvError("invalid_encoding", f"{where} starts with a byte-order mark; refused")
    if not text:
        raise DailyCsvError("empty_document", f"{where} is empty")
    text = text.replace("\r\n", "\n")
    if _CONTROL.search(text):
        raise DailyCsvError("invalid_control_character", f"{where} contains control characters (tabs and bare CR included)")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()  # exactly one terminating newline is allowed
    meta: dict[str, str] = {}
    index = 0
    while index < len(lines) and lines[index].startswith("#"):
        if index >= limits.max_header_lines:
            raise DailyCsvError("header_quota_exceeded", f"{where} has more than {limits.max_header_lines} preamble lines")
        line = lines[index]
        if len(line) > limits.max_header_chars:
            raise DailyCsvError("header_quota_exceeded", f"{where} preamble line {index + 1} exceeds {limits.max_header_chars} chars")
        match = _META.fullmatch(line)
        if match is None:
            raise DailyCsvError("invalid_preamble", f"{where} preamble line {index + 1} must be '# key: value'")
        key, value = match.group(1), match.group(2)
        if key in meta:
            raise DailyCsvError("duplicate_metadata", f"{where} repeats preamble key {key!r}")
        if key not in allowed:
            raise DailyCsvError("unknown_metadata", f"{where} has unknown preamble key {key!r}")
        meta[key] = value
        index += 1
    missing = [key for key in required if key not in meta]
    if missing:
        raise DailyCsvError("missing_metadata", f"{where} lacks required preamble keys {missing}")
    return meta, lines[index:]


def _table(lines: list[str], columns: tuple[str, ...], limits: DailyCsvLimits, where: str,
           max_rows: int) -> list[tuple[str, ...]]:
    if not lines:
        raise DailyCsvError("missing_table_header", f"{where} has no CSV header after the preamble")
    header = lines[0]
    if len(header) > limits.max_line_chars:
        raise DailyCsvError("invalid_header", f"{where} header exceeds {limits.max_line_chars} chars")
    if header.split(",") != list(columns):
        raise DailyCsvError("invalid_header", f"{where} header must be exactly {','.join(columns)}")
    body = lines[1:]
    if len(body) > max_rows:
        raise DailyCsvError("row_quota_exceeded", f"{where} has more than {max_rows} table rows")
    rows: list[tuple[str, ...]] = []
    for number, line in enumerate(body, 1):
        if not line:
            raise DailyCsvError("malformed_row", f"{where} row {number} is blank")
        if len(line) > limits.max_line_chars:
            raise DailyCsvError("malformed_row", f"{where} row {number} exceeds {limits.max_line_chars} chars")
        try:
            cells = tuple(next(csv.reader([line], strict=True)))
        except (csv.Error, StopIteration) as exc:
            raise DailyCsvError("malformed_row", f"{where} row {number} is not valid CSV") from exc
        if len(cells) != len(columns):
            raise DailyCsvError("malformed_row", f"{where} row {number} has {len(cells)} cells, expected {len(columns)}")
        for cell in cells:
            if len(cell) > limits.max_cell_chars:
                raise DailyCsvError("malformed_row", f"{where} row {number} has a cell over {limits.max_cell_chars} chars")
            if cell != cell.strip():
                raise DailyCsvError("malformed_row", f"{where} row {number} has a cell with surrounding whitespace")
        rows.append(cells)
    return rows


def _date(text: str, where: str) -> _dt.date:
    if not _DATE.fullmatch(text):
        raise DailyCsvError("invalid_date", f"{where}: {text[:40]!r} must be canonical YYYY-MM-DD")
    try:
        value = _dt.date.fromisoformat(text)
    except ValueError as exc:
        raise DailyCsvError("invalid_date", f"{where}: {text!r} is not a Gregorian date") from exc
    if value.isoformat() != text:
        raise DailyCsvError("invalid_date", f"{where}: {text!r} is not canonical")
    return value


def _number(text: str, where: str, *, nonnegative: bool) -> float:
    if len(text) > _MAX_NUMBER_CHARS or not _NUMBER.fullmatch(text):
        raise DailyCsvError("invalid_number", f"{where}: {text[:40]!r} is not a plain decimal number")
    value = float(text)
    if not math.isfinite(value):
        raise DailyCsvError("nonfinite_number", f"{where}: {text!r} is not finitely representable as binary64")
    if value == 0.0:
        if re.search(r"[1-9]", re.split("[eE]", text)[0]):
            raise DailyCsvError("number_underflow", f"{where}: {text!r} underflows binary64")
        if text.startswith("-"):
            raise DailyCsvError("negative_zero", f"{where}: negative zero is refused rather than rewritten")
    elif nonnegative and value < 0:
        raise DailyCsvError("negative_value", f"{where}: {text!r} must not be negative")
    return value


def _bool(text: str, where: str) -> bool:
    if text not in ("true", "false"):
        raise DailyCsvError("invalid_boolean", f"{where} must be the literal true or false")
    return text == "true"


def _interval(text: str, where: str) -> int:
    if not _INTERVAL.fullmatch(text):
        raise DailyCsvError("invalid_interval", f"{where} must be a strict positive integer of at most 9 digits")
    return int(text)


# ----------------------------------------------------------------------------- weather

@dataclass(frozen=True)
class DailyWeatherDay:
    """One accepted weather day, native binary64 values. Optional fields depend on `pet_method`."""

    date: _dt.date
    tmean_c: float
    precipitation_mm: float
    precipitation_intensity_mm_h: float
    tmin_c: float | None = None
    tmax_c: float | None = None
    pet_mm_day: float | None = None

    @property
    def wet_duration_s(self) -> float | None:
        """Nominal P/I wet-period duration in seconds (None when dry); excludes any recession."""
        if self.precipitation_mm == 0.0:
            return None
        return self.precipitation_mm / self.precipitation_intensity_mm_h * 3600.0

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {"date": self.date.isoformat()}
        for name in ("tmin_c", "tmean_c", "tmax_c", "pet_mm_day", "precipitation_mm", "precipitation_intensity_mm_h"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out

    def _hex_row(self) -> list[str]:
        row = [self.date.isoformat()]
        for name in ("tmin_c", "tmean_c", "tmax_c", "pet_mm_day", "precipitation_mm", "precipitation_intensity_mm_h"):
            value = getattr(self, name)
            row.append("-" if value is None else value.hex())
        return row


@dataclass(frozen=True)
class DailyWeatherTable:
    pet_method: str
    precipitation_threshold_mm: float
    days: tuple[DailyWeatherDay, ...]
    sha256: str
    size: int
    values_sha256: str
    source: str | None = None
    description: str | None = None
    format: str = WEATHER_FORMAT

    @property
    def first_date(self) -> _dt.date:
        return self.days[0].date

    @property
    def last_date(self) -> _dt.date:
        return self.days[-1].date

    def day_index(self, date: _dt.date) -> int:
        offset = (date - self.first_date).days
        if not 0 <= offset < len(self.days):
            raise DailyCsvError("weather_coverage", f"{date.isoformat()} is outside the weather table")
        return offset

    def summary(self, reference: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "format": self.format, "pet_method": self.pet_method,
            "precipitation_threshold_mm": self.precipitation_threshold_mm,
            "days": len(self.days), "first_date": self.first_date.isoformat(),
            "last_date": self.last_date.isoformat(), "sha256": self.sha256, "size": self.size,
            "values_sha256": self.values_sha256, "source": self.source, "description": self.description}
        if reference is not None:
            out["reference"] = reference
        return out


def parse_weather_csv(data: bytes, limits: DailyCsvLimits = DEFAULT_LIMITS, where: str = "weather") -> DailyWeatherTable:
    """Parse one weather CSV (bytes) strictly. Gaps, duplicates and reordering refuse."""
    meta, lines = _split_document(data, limits, where, _WEATHER_KEYS, _WEATHER_REQUIRED, limits.max_weather_rows)
    if meta["format"] != WEATHER_FORMAT:
        raise DailyCsvError("invalid_format", f"{where} format must be {WEATHER_FORMAT!r}")
    method = meta["pet_method"]
    if method not in (PET_HARGREAVES, PET_EXPLICIT):
        raise DailyCsvError("unsupported_pet_method", f"{where} pet_method must be {PET_HARGREAVES!r} or {PET_EXPLICIT!r}")
    threshold = _number(meta["precipitation_threshold_mm"], f"{where} precipitation_threshold_mm", nonnegative=True)
    columns = HARGREAVES_COLUMNS if method == PET_HARGREAVES else EXPLICIT_COLUMNS
    rows = _table(lines, columns, limits, where, limits.max_weather_rows)
    if not rows:
        raise DailyCsvError("weather_empty", f"{where} has no weather rows")
    days: list[DailyWeatherDay] = []
    digest = hashlib.sha256()
    previous: _dt.date | None = None
    for number, row in enumerate(rows, 1):
        label = f"{where} row {number}"
        date = _date(row[0], f"{label} date")
        if previous is not None:
            if date == previous:
                raise DailyCsvError("weather_date_duplicate", f"{label}: {row[0]} repeats the previous date")
            if date < previous:
                raise DailyCsvError("weather_date_out_of_order", f"{label}: {row[0]} precedes the previous date")
            if date != previous + _dt.timedelta(days=1):
                raise DailyCsvError("weather_date_missing", f"{label}: dates after {previous.isoformat()} are missing")
        previous = date
        if method == PET_HARGREAVES:
            tmin = _number(row[1], f"{label} tmin_c", nonnegative=False)
            tmean = _number(row[2], f"{label} tmean_c", nonnegative=False)
            tmax = _number(row[3], f"{label} tmax_c", nonnegative=False)
            if not tmin <= tmean <= tmax:
                raise DailyCsvError("temperature_order", f"{label}: requires tmin_c <= tmean_c <= tmax_c")
            pet = None
            rest = row[4:]
        else:
            tmin = tmax = None
            tmean = _number(row[1], f"{label} tmean_c", nonnegative=False)
            pet = _number(row[2], f"{label} pet_mm_day", nonnegative=True)
            rest = row[3:]
        precipitation = _number(rest[0], f"{label} precipitation_mm", nonnegative=True)
        intensity = _number(rest[1], f"{label} precipitation_intensity_mm_h", nonnegative=True)
        if precipitation > 0.0:
            if intensity == 0.0:
                raise DailyCsvError("rain_without_intensity", f"{label}: positive precipitation requires positive intensity")
            wet = precipitation / intensity * 3600.0
            if not math.isfinite(wet) or wet <= 0.0:
                raise DailyCsvError("rain_duration_not_representable", f"{label}: P/I wet duration is not a positive finite binary64")
        elif intensity != 0.0:
            raise DailyCsvError("intensity_without_rain", f"{label}: zero precipitation requires zero intensity")
        day = DailyWeatherDay(date, tmean, precipitation, intensity, tmin, tmax, pet)
        days.append(day)
        digest.update(canonical_json(day._hex_row()).encode("ascii") + b"\n")
    return DailyWeatherTable(method, threshold, tuple(days), hashlib.sha256(data).hexdigest(), len(data),
                             digest.hexdigest(), meta.get("source"), meta.get("description"))


# ----------------------------------------------------------------------------- wind

def parse_wind_yaml(data: bytes, limits: DailyCsvLimits = DEFAULT_LIMITS,
                    where: str = "wind") -> tuple[WindDeclaration, ...]:
    """Validate a native MAPLE wind YAML (`wind_forcing: {blocks: [...]}`) as complete declarations.

    Uses the existing strict YAML reader and `WindEventFile` scalar rules. Records are
    never truncated, retimed or rescaled. This is metadata validation only.
    """
    try:
        value = parse_strict_yaml(data, limits.io_limits(), where)
    except DailyCsvError:
        raise
    except ScenarioIOError as exc:
        raise DailyCsvError(exc.code, exc.detail) from exc
    if type(value) is not dict or set(value) != {"wind_forcing"}:
        raise DailyCsvError("invalid_wind_record", f"{where} must contain exactly the top-level key wind_forcing")
    forcing = value["wind_forcing"]
    if type(forcing) is not dict or set(forcing) != {"blocks"}:
        raise DailyCsvError("invalid_wind_record", f"{where}.wind_forcing must contain exactly the key blocks")
    try:
        event = WindEventFile.from_input({"schema": WIND_SCHEMA, "event_id": "daily", "wind_forcing": forcing}, where)
    except EventFileError as exc:
        raise DailyCsvError("invalid_wind_record", f"{exc.code}: {exc.detail}") from exc
    try:
        duration = float(sum((Fraction(b.duration_s) for b in event.blocks), Fraction(0)))
    except OverflowError as exc:
        raise DailyCsvError("invalid_wind_duration", f"{where} total wind duration overflows binary64") from exc
    if not math.isfinite(duration):
        raise DailyCsvError("invalid_wind_duration", f"{where} total wind duration must be finite")
    return event.blocks


# ----------------------------------------------------------------------------- scenario records

@dataclass(frozen=True)
class DailySettings:
    scenario_id: str
    start_date: _dt.date
    end_date: _dt.date
    evaporation_enabled: bool
    transpiration_enabled: bool
    growth_enabled: bool
    growth_interval_days: int
    output_interval_days: int
    checkpoint_interval_days: int
    output_at_event_boundaries: bool

    @property
    def day_count(self) -> int:
        return (self.end_date - self.start_date).days + 1

    def record(self) -> dict[str, Any]:
        return {"scenario_id": self.scenario_id, "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(), "evaporation_enabled": self.evaporation_enabled,
                "transpiration_enabled": self.transpiration_enabled, "growth_enabled": self.growth_enabled,
                "growth_interval_days": self.growth_interval_days, "output_interval_days": self.output_interval_days,
                "checkpoint_interval_days": self.checkpoint_interval_days,
                "output_at_event_boundaries": self.output_at_event_boundaries}


@dataclass(frozen=True)
class WindOccurrence:
    """One scenario wind row: a complete native record with its own derived identity."""

    index: int
    occurrence_id: str
    occurrence_sha256: str
    date: _dt.date
    event_type: str
    reference: str
    size: int
    sha256: str
    blocks: tuple[WindDeclaration, ...]

    @property
    def declared_duration_s(self) -> float:
        """Sum of declared block durations (exact rational, rounded once); not a physical claim."""
        return float(sum((Fraction(b.duration_s) for b in self.blocks), Fraction(0)))

    def record(self) -> dict[str, Any]:
        return {"index": self.index, "occurrence_id": self.occurrence_id,
                "occurrence_sha256": self.occurrence_sha256, "date": self.date.isoformat(),
                "event_type": self.event_type, "reference": self.reference, "size": self.size,
                "sha256": self.sha256, "blocks": [b.record() for b in self.blocks]}


def _identity_json(scenario: "DailyScenario") -> str:
    weather = scenario.weather
    files = {}
    events = []
    for occ in scenario.occurrences:
        # Reused files must not multiply an entire block declaration per row.
        if occ.reference not in files:
            files[occ.reference] = {"size": occ.size, "sha256": occ.sha256,
                                    "blocks": [b.record() for b in occ.blocks]}
        events.append({"index": occ.index, "occurrence_id": occ.occurrence_id,
                       "occurrence_sha256": occ.occurrence_sha256, "date": occ.date.isoformat(),
                       "event_type": occ.event_type, "reference": occ.reference, "sha256": occ.sha256})
    record = {
        "schema": IDENTITY_SCHEMA,
        "scenario": {"sha256": scenario.scenario_sha256, "size": scenario.scenario_size},
        "settings": scenario.settings.record(),
        "weather": {**weather.summary(scenario.weather_reference),
                    "precipitation_threshold_mm": encode_time(weather.precipitation_threshold_mm)},
        "wind_files": files,
        "events": events,
    }
    return canonical_json(record)


@dataclass(frozen=True)
class DailyScenario:
    """Immutable parsed scenario. Identity covers exact input byte hashes, scenario-relative
    paths, normalized settings, full weather values and every wind declaration in row order."""

    settings: DailySettings
    weather: DailyWeatherTable
    weather_reference: str
    scenario_size: int
    scenario_sha256: str
    occurrences: tuple[WindOccurrence, ...]
    identity_json: str = field(init=False, repr=False)
    identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (type(self.settings) is not DailySettings or type(self.weather) is not DailyWeatherTable
                or type(self.occurrences) is not tuple
                or any(type(o) is not WindOccurrence for o in self.occurrences)):
            raise DailyCsvError("invalid_scenario", "DailyScenario requires parsed immutable records")
        text = _identity_json(self)
        object.__setattr__(self, "identity_json", text)
        object.__setattr__(self, "identity_sha256", hashlib.sha256(text.encode("ascii")).hexdigest())

    def identity_record(self) -> dict[str, Any]:
        """Fresh JSON-compatible copy of the canonical identity; mutating it changes nothing."""
        return json.loads(self.identity_json)

    @property
    def wind_files(self) -> tuple[tuple[str, int, str], ...]:
        """Distinct (reference, size, sha256) wind files in first-use order."""
        seen: dict[str, tuple[str, int, str]] = {}
        for occ in self.occurrences:
            seen.setdefault(occ.reference, (occ.reference, occ.size, occ.sha256))
        return tuple(seen.values())

    def summary(self) -> dict[str, Any]:
        return {"identity_sha256": self.identity_sha256, "settings": self.settings.record(),
                "weather": self.weather.summary(self.weather_reference),
                "scenario": {"size": self.scenario_size, "sha256": self.scenario_sha256},
                "wind_occurrences": len(self.occurrences), "distinct_wind_files": len(self.wind_files)}


# ----------------------------------------------------------------------------- loader

def _settings(meta: dict[str, str], where: str) -> DailySettings:
    if meta["format"] != SCENARIO_FORMAT:
        raise DailyCsvError("invalid_format", f"{where} format must be {SCENARIO_FORMAT!r}")
    if not _SCENARIO_ID.fullmatch(meta["scenario_id"]):
        raise DailyCsvError("invalid_scenario_id", f"{where} scenario_id must match {_SCENARIO_ID.pattern}")
    start = _date(meta["start_date"], f"{where} start_date")
    end = _date(meta["end_date"], f"{where} end_date")
    if end < start:
        raise DailyCsvError("invalid_date_range", f"{where} end_date precedes start_date")
    growth = _bool(meta["growth_enabled"], f"{where} growth_enabled")
    growth_days = _interval(meta["growth_interval_days"], f"{where} growth_interval_days")
    low, high = GROWTH_INTERVAL_RANGE
    if growth and not low <= growth_days <= high:
        raise DailyCsvError("growth_interval_out_of_range",
                            f"{where} growth_interval_days must be within {low}..{high} when growth is enabled")
    return DailySettings(
        meta["scenario_id"], start, end,
        _bool(meta["evaporation_enabled"], f"{where} evaporation_enabled"),
        _bool(meta["transpiration_enabled"], f"{where} transpiration_enabled"),
        growth, growth_days,
        _interval(meta["output_interval_days"], f"{where} output_interval_days"),
        _interval(meta["checkpoint_interval_days"], f"{where} checkpoint_interval_days"),
        _bool(meta["output_at_event_boundaries"], f"{where} output_at_event_boundaries"))


def load_daily_scenario(path: Path | str, limits: DailyCsvLimits = DEFAULT_LIMITS) -> DailyScenario:
    """Read a scenario CSV plus its weather and wind files; refuse anything unsafe or inconsistent."""
    if type(limits) is not DailyCsvLimits:
        raise DailyCsvError("invalid_limit", "limits must be a DailyCsvLimits")
    try:
        return _load(Path(path), limits)
    except DailyCsvError:
        raise
    except ScenarioIOError as exc:
        raise DailyCsvError(exc.code, exc.detail) from exc


def _load(path: Path, limits: DailyCsvLimits) -> DailyScenario:
    io = limits.io_limits()
    if limits.max_files < 2:
        raise DailyCsvError("file_quota_exceeded", "a scenario and its weather require at least two files")
    base = Path(os.path.realpath(path.parent))
    data, pin = read_bounded(path, limits.max_file_bytes)
    scenario_size, scenario_sha = pin["size"], pin["sha256"]
    total = scenario_size
    meta, lines = _split_document(data, limits, "scenario", _SCENARIO_KEYS, _SCENARIO_KEYS, limits.max_scenario_rows)
    settings = _settings(meta, "scenario")
    rows = _table(lines, SCENARIO_COLUMNS, limits, "scenario", limits.max_scenario_rows)
    weather_ref = meta["weather_file"]
    weather_path = relative_reference(weather_ref, base, io, "weather_file")

    refs: dict[str, Path] = {}
    entries: list[tuple[_dt.date, str]] = []
    previous: _dt.date | None = None
    for number, row in enumerate(rows, 1):
        label = f"scenario event row {number}"
        date = _date(row[0], f"{label} date")
        if row[1] != "wind":
            raise DailyCsvError("unsupported_event_type", f"{label}: event_type must be 'wind'")
        if not settings.start_date <= date <= settings.end_date:
            raise DailyCsvError("event_date_out_of_range", f"{label}: {row[0]} is outside the inclusive scenario dates")
        if previous is not None and date < previous:
            raise DailyCsvError("event_dates_out_of_order", f"{label}: wind dates must be nondecreasing")
        previous = date
        ref = row[2]
        if ref not in refs:
            if len(refs) + 3 > limits.max_files:
                raise DailyCsvError("file_quota_exceeded", f"more than {limits.max_files} distinct referenced files")
            if Path(ref).suffix.lower() not in (".yaml", ".yml"):
                raise DailyCsvError("unsupported_wind_suffix", f"{label}: wind files must be .yaml or .yml")
            refs[ref] = relative_reference(ref, base, io, f"{label} event_file")
        entries.append((date, ref))

    wdata, wpin = read_bounded(weather_path, limits.max_file_bytes)
    total += wpin["size"]
    if total > limits.max_total_bytes:
        raise DailyCsvError("aggregate_quota_exceeded", f"referenced files exceed {limits.max_total_bytes} bytes")
    weather = parse_weather_csv(wdata, limits, "weather")
    del wdata
    if settings.start_date < weather.first_date or settings.end_date > weather.last_date:
        raise DailyCsvError("weather_coverage", "weather dates must cover the inclusive scenario start..end dates")

    wind: dict[str, tuple[int, str, tuple[WindDeclaration, ...]]] = {}
    for ref, resolved in refs.items():
        raw, rpin = read_bounded(resolved, limits.max_file_bytes)
        total += rpin["size"]
        if total > limits.max_total_bytes:
            raise DailyCsvError("aggregate_quota_exceeded", f"referenced files exceed {limits.max_total_bytes} bytes")
        wind[ref] = (rpin["size"], rpin["sha256"], parse_wind_yaml(raw, limits, f"wind file {ref}"))

    # Snapshot check: every distinct file is read again and must still match its first read.
    for label, again, size, sha in ([("scenario", path, scenario_size, scenario_sha),
                                     ("weather", weather_path, weather.size, weather.sha256)]
                                    + [(f"wind file {r}", refs[r], v[0], v[1]) for r, v in wind.items()]):
        _, second = read_bounded(again, limits.max_file_bytes)
        if second["size"] != size or second["sha256"] != sha:
            raise DailyCsvError("document_changed", f"{label} changed between reads")

    occurrences = []
    for index, (date, ref) in enumerate(entries):
        size, sha, blocks = wind[ref]
        derived = canonical_json({"scenario_id": settings.scenario_id, "index": index,
                                  "date": date.isoformat(), "reference": ref, "sha256": sha})
        occurrences.append(WindOccurrence(index, f"wind-{index:06d}", hashlib.sha256(derived.encode("ascii")).hexdigest(),
                                          date, "wind", ref, size, sha, blocks))
    return DailyScenario(settings, weather, weather_ref, scenario_size, scenario_sha, tuple(occurrences))
