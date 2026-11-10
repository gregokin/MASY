"""Reusable, source-bound event files: wind, rain and interevent (data only).

One file describes one reusable event; a scenario (`masy.scenario`) references
files by relative path plus SHA-256 and gives each use a distinct occurrence
id. Forcing semantics stay with the owning model:

- **wind** (`masy.wind_event.v1`): the complete native MAPLE wind-forcing
  record form, `{"wind_forcing": {"blocks": [{shear_velocity_m_s,
  direction_deg, duration_s}, ...]}}`, exactly the document shape MAPLE's
  `resolve_wind_forcing_record` consumes (`core/parameters/wind_forcing.py`).
  Every occurrence runs the COMPLETE record in order; a scenario cannot
  select, split, retime or rescale blocks. Scalars are validated with the
  existing `masy.case_schedule.WindDeclaration` rules (finite, non-negative
  shear, direction in [0, 360), positive duration). Wind ends at the record
  end; MAPLE's native finalization is instantaneous at that clock.
- **rain** (`masy.rain_event.v1`): SYRUP semantics through the existing
  `masy.dense_schedule.RainEvent` / `RainInterval` validation: positive
  piecewise interval durations with intensities in mm/h, at least one wet
  interval, and a positive finite `recession_guard_s`. The realized end of a
  rain occurrence (after physical recession) is never predicted here; it
  lies in [rainfall end, recession guard end] or the event is incomplete.
- **interevent** (`masy.interevent.v1`): a positive duration in seconds plus
  the explicit process gate (`evaporation`, `transpiration`, `growth`, each a
  bool). These processes exist ONLY in interevent occurrences; wind and rain
  dispatches always carry an all-false gate and nothing is ever applied
  retroactively over them.

Durations and forcing values are plain JSON numbers (binary64; bool, NaN,
+-inf, -0.0 and ints not exactly representable refuse). Records are lossless
(`masy.clock.encode_time`) and the digest is the SHA-256 of the canonical
record. Importing this module imports neither NumPy nor a native model.
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

from masy.case_schedule import CaseScheduleError, WindDeclaration
from masy.clock import encode_time
from masy.dense_schedule import DenseScheduleError, RainEvent, RainInterval
from masy.scenario_io import IOLimits, canonical_json, load_document

__all__ = [
    "EVENT_KINDS",
    "EVENT_SCHEMAS",
    "INTEREVENT_PROCESSES",
    "INTEREVENT_SCHEMA",
    "RAIN_SCHEMA",
    "WIND_SCHEMA",
    "EventFileError",
    "IntereventFile",
    "RainEventFile",
    "WindEventFile",
    "event_id_text",
    "load_event_file",
    "parse_event_file",
    "process_gate",
]

WIND_SCHEMA = "masy.wind_event.v1"
RAIN_SCHEMA = "masy.rain_event.v1"
INTEREVENT_SCHEMA = "masy.interevent.v1"
EVENT_SCHEMAS = {WIND_SCHEMA: "wind", RAIN_SCHEMA: "rain", INTEREVENT_SCHEMA: "interevent"}
EVENT_KINDS = ("wind", "rain", "interevent")
INTEREVENT_PROCESSES = ("evaporation", "transpiration", "growth")
_EVENT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_WIND_BLOCK_KEYS = ("shear_velocity_m_s", "direction_deg", "duration_s")


class EventFileError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _closed(value: Any, names: tuple[str, ...], where: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(names):
        raise EventFileError("invalid_keys", f"{where} requires exactly the keys {names}")
    return value


def _number(value: Any, where: str, *, positive: bool = False, minimum: float = 0.0) -> float:
    if type(value) not in (int, float):
        raise EventFileError("invalid_number", f"{where} must be a plain int or float, got {type(value).__name__}")
    if type(value) is int:
        converted = float(value)
        if converted != value:
            raise EventFileError("invalid_number", f"{where}={value} is not exactly representable as binary64")
        value = converted
    if not math.isfinite(value):
        raise EventFileError("nonfinite_number", f"{where} must be finite")
    if value == 0 and math.copysign(1.0, value) < 0:
        raise EventFileError("negative_zero", f"{where} is -0.0; refused rather than rewritten")
    if value < minimum or (positive and value <= minimum):
        raise EventFileError("out_of_domain", f"{where}={value!r} must be {'>' if positive else '>='} {minimum}")
    return value


def event_id_text(value: Any, where: str) -> str:
    if type(value) is not str or not _EVENT_ID.match(value):
        raise EventFileError("invalid_event_id", f"{where} must match {_EVENT_ID.pattern}")
    return value


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(record).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class WindEventFile:
    event_id: str
    blocks: tuple[WindDeclaration, ...]
    kind: str = "wind"

    def __post_init__(self) -> None:
        event_id_text(self.event_id, "wind.event_id")
        if (type(self.blocks) is not tuple or not self.blocks
                or any(type(b) is not WindDeclaration or b.index != i for i, b in enumerate(self.blocks))):
            raise EventFileError("invalid_wind_record", "complete ordered WindDeclaration blocks required")
        for block in self.blocks:
            block.__post_init__()

    @classmethod
    def from_input(cls, value: Any, where: str = "wind") -> "WindEventFile":
        _closed(value, ("schema", "event_id", "wind_forcing"), where)
        if value["schema"] != WIND_SCHEMA:
            raise EventFileError("invalid_schema", f"{where}.schema must be {WIND_SCHEMA!r}")
        forcing = _closed(value["wind_forcing"], ("blocks",), f"{where}.wind_forcing")
        rows = forcing["blocks"]
        if type(rows) is not list or not rows:
            raise EventFileError("invalid_wind_record", f"{where}.wind_forcing.blocks must be a non-empty list")
        blocks = []
        for index, row in enumerate(rows):
            label = f"{where}.wind_forcing.blocks[{index}]"
            _closed(row, _WIND_BLOCK_KEYS, label)
            try:
                blocks.append(WindDeclaration(
                    index,
                    _number(row["shear_velocity_m_s"], f"{label}.shear_velocity_m_s"),
                    _number(row["direction_deg"], f"{label}.direction_deg"),
                    _number(row["duration_s"], f"{label}.duration_s", positive=True),
                ))
            except CaseScheduleError as exc:
                raise EventFileError("invalid_wind_record", f"{label}: {exc}") from exc
        return cls(event_id_text(value["event_id"], f"{where}.event_id"), tuple(blocks))

    def planned_edges(self) -> tuple[list[Fraction], list[str]]:
        offsets, roles, total = [Fraction(0)], ["event_start"], Fraction(0)
        for block in self.blocks:
            total += Fraction(block.duration_s)
            offsets.append(total)
            roles.append(f"original_block_{block.index}_end")
        roles[-1] = "wind_end"
        return offsets, roles

    @property
    def minimum_duration_s(self) -> Fraction:
        return self.planned_edges()[0][-1]

    @property
    def maximum_duration_s(self) -> Fraction:
        return self.minimum_duration_s

    def record(self) -> dict[str, Any]:
        return {"schema": WIND_SCHEMA, "event_id": self.event_id, "kind": self.kind,
                "forcing_semantics": "complete native MAPLE wind_forcing record per occurrence; no block selection",
                "wind_forcing": {"blocks": [b.record() for b in self.blocks]},
                "total_duration_s": _fraction_text(self.minimum_duration_s)}

    @property
    def digest(self) -> str:
        return _digest(self.record())


@dataclass(frozen=True)
class RainEventFile:
    event_id: str
    water_profile_id: str
    event: RainEvent
    kind: str = "rain"

    def __post_init__(self) -> None:
        event_id_text(self.event_id, "rain.event_id")
        if type(self.water_profile_id) is not str or not self.water_profile_id:
            raise EventFileError("invalid_water_profile", "rain.water_profile_id must be a non-empty string")
        if type(self.event) is not RainEvent:
            raise EventFileError("invalid_rain", "rain.event must be a dense_schedule.RainEvent")
        self.event.__post_init__()

    @classmethod
    def from_input(cls, value: Any, where: str = "rain") -> "RainEventFile":
        _closed(value, ("schema", "event_id", "water_profile_id", "intervals", "recession_guard_s"), where)
        if value["schema"] != RAIN_SCHEMA:
            raise EventFileError("invalid_schema", f"{where}.schema must be {RAIN_SCHEMA!r}")
        rows = value["intervals"]
        if type(rows) is not list or not rows:
            raise EventFileError("invalid_rain", f"{where}.intervals must be a non-empty list")
        try:
            intervals = []
            for index, row in enumerate(rows):
                label = f"{where}.intervals[{index}]"
                _closed(row, ("duration_s", "intensity_mm_h"), label)
                intervals.append(RainInterval(_number(row["duration_s"], f"{label}.duration_s", positive=True),
                                              _number(row["intensity_mm_h"], f"{label}.intensity_mm_h")))
            event = RainEvent(tuple(intervals), _number(value["recession_guard_s"], f"{where}.recession_guard_s",
                                                        positive=True))
        except DenseScheduleError as exc:
            raise EventFileError("invalid_rain", f"{where}: {exc}") from exc
        profile = value["water_profile_id"]
        if type(profile) is not str or not profile:
            raise EventFileError("invalid_water_profile", f"{where}.water_profile_id must be a non-empty string")
        return cls(event_id_text(value["event_id"], f"{where}.event_id"), profile, event)

    def planned_edges(self) -> tuple[list[Fraction], list[str]]:
        offsets, roles, total = [Fraction(0)], ["event_start"], Fraction(0)
        for index, interval in enumerate(self.event.intervals):
            total += Fraction(interval.duration_s)
            offsets.append(total)
            roles.append(f"rain_interval_{index}_end")
        roles[-1] = "rainfall_end"
        offsets.append(total + Fraction(self.event.recession_guard_s))
        roles.append("recession_guard_end")
        return offsets, roles

    @property
    def minimum_duration_s(self) -> Fraction:
        """Rainfall only; recession adds realized time up to the guard."""
        return self.event.rainfall_duration_s

    @property
    def maximum_duration_s(self) -> Fraction:
        return self.event.rainfall_duration_s + Fraction(self.event.recession_guard_s)

    def record(self) -> dict[str, Any]:
        return {"schema": RAIN_SCHEMA, "event_id": self.event_id, "kind": self.kind,
                "water_profile_id": self.water_profile_id,
                "forcing_semantics": "native SYRUP piecewise rainfall plus physical recession bounded by the guard",
                "intervals": [{"duration_s": encode_time(i.duration_s), "intensity_mm_h": encode_time(i.intensity_mm_h)}
                              for i in self.event.intervals],
                "rainfall_duration_s": _fraction_text(self.event.rainfall_duration_s),
                "recession_guard_s": encode_time(self.event.recession_guard_s)}

    @property
    def digest(self) -> str:
        return _digest(self.record())


@dataclass(frozen=True)
class IntereventFile:
    event_id: str
    duration_s: float
    processes: Mapping[str, bool]
    kind: str = "interevent"

    def __post_init__(self) -> None:
        event_id_text(self.event_id, "interevent.event_id")
        _number(self.duration_s, "interevent.duration_s", positive=True)
        if type(self.duration_s) is not float:
            raise EventFileError("invalid_number", "interevent.duration_s must be a float")
        if (not isinstance(self.processes, Mapping) or tuple(self.processes) != INTEREVENT_PROCESSES
                or any(type(v) is not bool for v in self.processes.values())):
            raise EventFileError("invalid_process_gate",
                                 f"interevent.processes must map exactly {INTEREVENT_PROCESSES} to bools, in order")
        object.__setattr__(self, "processes", MappingProxyType(dict(self.processes)))  # read-only after validation

    @classmethod
    def from_input(cls, value: Any, where: str = "interevent") -> "IntereventFile":
        _closed(value, ("schema", "event_id", "duration_s", "processes"), where)
        if value["schema"] != INTEREVENT_SCHEMA:
            raise EventFileError("invalid_schema", f"{where}.schema must be {INTEREVENT_SCHEMA!r}")
        gate = _closed(value["processes"], INTEREVENT_PROCESSES, f"{where}.processes")
        for name in INTEREVENT_PROCESSES:
            if type(gate[name]) is not bool:
                raise EventFileError("invalid_process_gate", f"{where}.processes.{name} must be a bool")
        return cls(event_id_text(value["event_id"], f"{where}.event_id"),
                   _number(value["duration_s"], f"{where}.duration_s", positive=True),
                   {name: gate[name] for name in INTEREVENT_PROCESSES})

    def planned_edges(self) -> tuple[list[Fraction], list[str]]:
        return [Fraction(0), Fraction(self.duration_s)], ["event_start", "interevent_end"]

    @property
    def minimum_duration_s(self) -> Fraction:
        return Fraction(self.duration_s)

    @property
    def maximum_duration_s(self) -> Fraction:
        return self.minimum_duration_s

    def record(self) -> dict[str, Any]:
        return {"schema": INTEREVENT_SCHEMA, "event_id": self.event_id, "kind": self.kind,
                "duration_s": encode_time(self.duration_s), "processes": dict(self.processes),
                "process_semantics": "evaporation/transpiration/growth are dispatched only inside this interval; "
                                     "never retroactively over wind/rain durations"}

    @property
    def digest(self) -> str:
        return _digest(self.record())


EventFile = WindEventFile | RainEventFile | IntereventFile


def process_gate(event: Any) -> dict[str, bool]:
    """The process gate of an occurrence: the interevent file's declared
    gate, or all False for wind and rain (storm/wind time never runs
    evaporation, transpiration or growth, recession included)."""
    if type(event) is IntereventFile:
        return dict(event.processes)
    if type(event) in (WindEventFile, RainEventFile):
        return {name: False for name in INTEREVENT_PROCESSES}
    raise EventFileError("invalid_event", "process_gate requires a parsed event file")


def parse_event_file(value: Any, where: str = "event") -> EventFile:
    if type(value) is not dict or value.get("schema") not in EVENT_SCHEMAS:
        raise EventFileError("invalid_schema", f"{where}.schema must be one of {sorted(EVENT_SCHEMAS)}")
    kind = EVENT_SCHEMAS[value["schema"]]
    if kind == "wind":
        return WindEventFile.from_input(value, where)
    if kind == "rain":
        return RainEventFile.from_input(value, where)
    return IntereventFile.from_input(value, where)


def load_event_file(path: Path, limits: IOLimits) -> tuple[EventFile, dict[str, Any]]:
    """Strictly load one event document; returns (event, source pin with sha256)."""
    value, pin = load_document(path, limits, where=str(path))
    return parse_event_file(value, where=str(path)), pin
