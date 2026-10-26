"""Explicit event grouping of a genuine resolved MAPLE wind record.

This module handles metadata only. It never constructs a Native result/state,
changes forcing, or treats an individual forcing block as an event boundary.
"""
from __future__ import annotations

import dataclasses
from fractions import Fraction
import hashlib
import json
import math
import re
from typing import Any

from masy.clock import ClockError, decode_time, encode_time, resolve_forcing_edges
from masy.dense_schedule import DenseScheduleError, RainEvent, RainInterval, _exact_decimal

SCHEMA = "masy.case_schedule.v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class CaseScheduleError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def closed(value: Any, names: tuple[str, ...], where: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(names):
        raise CaseScheduleError("invalid_keys", f"{where} requires exactly {names}")
    return value


def integer(value: Any, where: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise CaseScheduleError("invalid_integer", f"{where} must be a plain int >= {minimum}")
    return value


def number(value: Any, where: str, *, minimum: float = 0.0, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise CaseScheduleError("invalid_number", f"{where} requires a finite binary64 number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise CaseScheduleError("invalid_number", where) from error
    if (not math.isfinite(result) or (type(value) is int and int(result) != value)
            or result < minimum or (positive and result == minimum)
            or (result == 0 and math.copysign(1.0, result) < 0)):
        raise CaseScheduleError("invalid_number", f"{where} outside its exact finite domain")
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class WindDeclaration:
    """Source-bound scalar declaration, not a Native WindBlock/transport result."""
    index: int
    shear_velocity_m_s: float
    direction_deg: float
    duration_s: float

    def __post_init__(self):
        integer(self.index, "block index", minimum=0)
        for key in ("shear_velocity_m_s", "direction_deg", "duration_s"):
            value = getattr(self, key)
            if type(value) is not float:
                raise CaseScheduleError("invalid_block", "Native wind declaration scalars must be plain floats")
            number(value, key, positive=key == "duration_s")
        if self.direction_deg >= 360:
            raise CaseScheduleError("invalid_block", "direction must be below360 degrees")

    def record(self):
        return {"index": self.index, "shear_velocity_m_s": encode_time(self.shear_velocity_m_s),
                "direction_deg": encode_time(self.direction_deg), "duration_s": encode_time(self.duration_s)}


@dataclasses.dataclass(frozen=True)
class CaseWindGroup:
    first_block: int
    stop_block: int
    blocks: tuple[WindDeclaration, ...]
    kind: str = dataclasses.field(default="wind", init=False)

    def __post_init__(self):
        integer(self.first_block, "first_block", minimum=0)
        integer(self.stop_block, "stop_block")
        if (not self.first_block < self.stop_block or type(self.blocks) is not tuple
                or len(self.blocks) != self.stop_block - self.first_block
                or any(type(b) is not WindDeclaration or b.index != self.first_block + i
                       for i, b in enumerate(self.blocks))):
            raise CaseScheduleError("wind_partition", "group must retain exactly its declared original range")

    @property
    def duration_s(self) -> float:
        """Display/aggregate extent; individual Native durations stay exact."""
        return float(sum((Fraction(b.duration_s) for b in self.blocks), Fraction(0)))


@dataclasses.dataclass(frozen=True)
class CaseRainGroup:
    event: RainEvent
    water_profile_id: str
    kind: str = dataclasses.field(default="rain", init=False)

    def __post_init__(self):
        if type(self.event) is not RainEvent or type(self.water_profile_id) is not str or not self.water_profile_id:
            raise CaseScheduleError("invalid_rain", "genuine rain metadata and explicit profile required")
        self.event.__post_init__()


@dataclasses.dataclass(frozen=True)
class CaseSchedule:
    events: tuple[CaseWindGroup | CaseRainGroup, ...]
    original_blocks: tuple[WindDeclaration, ...]
    case_sha256: str
    wind_record_sha256: str
    water_profile_id: str
    water_identity_sha256: str

    def __post_init__(self):
        if (type(self.original_blocks) is not tuple or not self.original_blocks
                or any(type(b) is not WindDeclaration or b.index != i
                       for i, b in enumerate(self.original_blocks))):
            raise CaseScheduleError("invalid_blocks", "complete original declarations required")
        for digest in (self.case_sha256, self.wind_record_sha256, self.water_identity_sha256):
            if type(digest) is not str or not _SHA.fullmatch(digest):
                raise CaseScheduleError("invalid_digest", "lowercase SHA256 bindings required")
        if type(self.water_profile_id) is not str or not self.water_profile_id:
            raise CaseScheduleError("invalid_profile", "explicit water profile required")
        if type(self.events) is not tuple or not self.events:
            raise CaseScheduleError("invalid_events", "nonempty immutable event tuple required")
        cursor = 0
        for event in self.events:
            if type(event) is CaseWindGroup:
                event.__post_init__()
                if (event.first_block != cursor or event.stop_block > len(self.original_blocks)
                        or event.blocks != self.original_blocks[event.first_block:event.stop_block]):
                    raise CaseScheduleError("wind_partition", "groups must cover genuine original declarations in order")
                cursor = event.stop_block
            elif type(event) is CaseRainGroup:
                event.__post_init__()
                if event.water_profile_id != self.water_profile_id:
                    raise CaseScheduleError("water_profile_mismatch", "rain profile differs")
            else:
                raise CaseScheduleError("invalid_event", "foreign event type")
        if cursor != len(self.original_blocks):
            raise CaseScheduleError("wind_partition", "full original record required")

    @classmethod
    def from_input(cls, value: Any, *, blocks: tuple[WindDeclaration, ...], case_sha256: str,
                   wind_record_sha256: str, water_profile_id: str, water_identity_sha256: str,
                   max_events: int) -> CaseSchedule:
        closed(value, ("schema", "events"), "schedule")
        if value["schema"] != SCHEMA:
            raise CaseScheduleError("invalid_schema", "unsupported case schedule")
        integer(max_events, "max_events")
        if (type(blocks) is not tuple or not blocks or any(type(b) is not WindDeclaration or b.index != i
                                                       for i, b in enumerate(blocks))):
            raise CaseScheduleError("invalid_blocks", "complete ordered original declarations required")
        for digest in (case_sha256, wind_record_sha256, water_identity_sha256):
            if type(digest) is not str or not _SHA.fullmatch(digest):
                raise CaseScheduleError("invalid_digest", "binding requires a lowercase SHA256")
        if type(water_profile_id) is not str or not water_profile_id:
            raise CaseScheduleError("invalid_profile", "one explicit water profile required")
        raw = value["events"]
        if type(raw) is not list or not 1 <= len(raw) <= max_events:
            raise CaseScheduleError("event_limit", "event list is empty or exceeds its declared bound")
        events, cursor = [], 0
        for row in raw:
            if type(row) is not dict:
                raise CaseScheduleError("invalid_event", "events must be objects")
            if row.get("kind") == "wind":
                closed(row, ("kind", "first_block", "stop_block"), "wind event")
                first = integer(row["first_block"], "first_block", minimum=0)
                stop = integer(row["stop_block"], "stop_block")
                if first != cursor or not first < stop <= len(blocks):
                    raise CaseScheduleError("wind_partition", "wind ranges must be positive and cover original blocks in order")
                events.append(CaseWindGroup(first, stop, blocks[first:stop]))
                cursor = stop
            elif row.get("kind") == "rain":
                closed(row, ("kind", "water_profile_id", "intervals", "recession_guard_s"), "rain event")
                if row["water_profile_id"] != water_profile_id:
                    raise CaseScheduleError("water_profile_mismatch", "all rain events require the supplied shared profile")
                if type(row["intervals"]) is not list or not row["intervals"]:
                    raise CaseScheduleError("invalid_rain", "explicit positive rain intervals required")
                intervals = []
                for interval in row["intervals"]:
                    closed(interval, ("duration_s", "intensity_mm_h"), "rain interval")
                    intervals.append(RainInterval(number(interval["duration_s"], "duration_s", positive=True),
                                                  number(interval["intensity_mm_h"], "intensity_mm_h")))
                try:
                    event = RainEvent(tuple(intervals), number(row["recession_guard_s"], "recession_guard_s", positive=True))
                except DenseScheduleError as error:
                    raise CaseScheduleError("invalid_rain", str(error)) from error
                events.append(CaseRainGroup(event, water_profile_id))
            else:
                raise CaseScheduleError("invalid_event", "only relative wind groups and rain events are supported")
        if cursor != len(blocks):
            raise CaseScheduleError("wind_partition", "the full original wind record must be consumed exactly once")
        return cls(tuple(events), blocks, case_sha256, wind_record_sha256, water_profile_id, water_identity_sha256)

    def as_record(self) -> dict[str, Any]:
        rows = []
        for event in self.events:
            if type(event) is CaseWindGroup:
                rows.append({"kind": "wind", "first_block": event.first_block, "stop_block": event.stop_block,
                             "local_to_original": [b.index for b in event.blocks]})
            elif type(event) is CaseRainGroup:
                rows.append({"kind": "rain", "water_profile_id": event.water_profile_id,
                             "intervals": [{"duration_s": encode_time(i.duration_s),
                                            "intensity_mm_h": encode_time(i.intensity_mm_h)} for i in event.event.intervals],
                             "recession_guard_s": encode_time(event.event.recession_guard_s)})
            else:
                raise CaseScheduleError("invalid_event", "foreign event type")
        return {"schema": SCHEMA, "case_sha256": self.case_sha256, "wind_record_sha256": self.wind_record_sha256,
                "water_profile_id": self.water_profile_id, "water_identity_sha256": self.water_identity_sha256,
                "original_blocks": [b.record() for b in self.original_blocks], "events": rows}

    @property
    def digest(self) -> str:
        return _digest(self.as_record())

    def combined_digest(self, bed_sha256: str) -> str:
        if type(bed_sha256) is not str or not _SHA.fullmatch(bed_sha256):
            raise CaseScheduleError("invalid_digest", "bed binding requires lowercase SHA256")
        return _digest({"schema": "masy.case_cpu_binding.v1", "bed_sha256": bed_sha256, "schedule_sha256": self.digest})

    def resolve(self, index: int, start: float) -> dict[str, Any]:
        if type(index) is not int or not 0 <= index < len(self.events):
            raise CaseScheduleError("invalid_event_index", "event index out of range")
        start = number(start, "native event start")
        event = self.events[index]
        offsets, roles, total = [Fraction(0)], ["event_start"], Fraction(0)
        if type(event) is CaseWindGroup:
            for b in event.blocks:
                total += Fraction(b.duration_s)
                offsets.append(total)
                roles.append(f"original_block_{b.index}_end")
            roles[-1] = "wind_end"
        else:
            for i, interval in enumerate(event.event.intervals):
                total += Fraction(interval.duration_s)
                offsets.append(total)
                roles.append(f"rain_interval_{i}_end")
            roles[-1] = "rainfall_end"
            offsets.append(total + Fraction(event.event.recession_guard_s))
            roles.append("recession_guard_end")
        return {"schema": "masy.case_resolved_event.v1", "schedule_sha256": self.digest, "index": index,
                "kind": event.kind, "intended": self.as_record()["events"][index], "edge_roles": roles,
                "resolved_edges": resolve_forcing_edges(start, [_exact_decimal(t) for t in offsets])}

    def check_resolved(self, value: Any) -> dict[str, Any]:
        closed(value, ("schema", "schedule_sha256", "index", "kind", "intended", "edge_roles", "resolved_edges"), "resolved")
        edges = value["resolved_edges"]
        if type(edges) is not dict or "native_start" not in edges:
            raise CaseScheduleError("invalid_resolution", "resolved edges lack a native start")
        try:
            result = self.resolve(value["index"], decode_time(edges["native_start"]))
        except ClockError as error:
            raise CaseScheduleError("invalid_resolution", str(error)) from error
        if _digest(value) != _digest(result):
            raise CaseScheduleError("changed_resolution", "resolution changed from its genuine scalar derivation")
        return result
