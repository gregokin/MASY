"""Explicit dated calendar layer over the exact native/global clock (stdlib only).

The existing `masy.clock` keeps native binary64 solver times and the exact
rational global elapsed `G(t) = Fraction(t) - Fraction(T0)`. This module adds
a dated layer ON TOP of that mapping; it never rewrites, rounds or replaces a
native time. A `CalendarStart` is an explicit proleptic-Gregorian date-time
with a fixed UTC offset (no named zones, no daylight-saving rules, no leap
seconds: every calendar day has exactly 86400 SI seconds). Instants are the
start plus an exact non-negative rational number of seconds: the whole
seconds go into a `datetime`, the sub-second remainder is kept as an exact
fraction next to it, so nothing is lost.

Day arithmetic (day-of-year, leap years, dates touched by an interval, whole
day detection) is done with `datetime.date`, which already handles leap years
and year boundaries of the proleptic Gregorian calendar.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

__all__ = [
    "CALENDAR_CONVENTION",
    "CALENDAR_SCHEMA",
    "DAY_S",
    "LEAP_SECONDS_POLICY",
    "CalendarError",
    "CalendarInstant",
    "CalendarStart",
    "days_in_year",
    "fraction_text",
    "noleap_day_index",
    "parse_fraction",
    "parse_offset",
]

CALENDAR_SCHEMA = "masy.calendar.v1"
CALENDAR_CONVENTION = "proleptic_gregorian"
LEAP_SECONDS_POLICY = "none"
DAY_S = 86400
_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})([+-]\d{2}:\d{2})$")
_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_OFFSET = re.compile(r"^([+-])(\d{2}):(\d{2})$")
_FRACTION = re.compile(r"^(-?)(0|[1-9][0-9]*)/([1-9][0-9]*)$")


class CalendarError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def parse_fraction(text: Any, where: str) -> Fraction:
    if type(text) is not str or not _FRACTION.match(text):
        raise CalendarError("invalid_fraction", f"{where} must be a reduced 'p/q' fraction string")
    value = Fraction(text)
    if fraction_text(value) != text:
        raise CalendarError("invalid_fraction", f"{where} {text!r} is not in lowest terms")
    return value


def parse_offset(text: Any, where: str) -> _dt.timezone:
    """A fixed `+HH:MM`/`-HH:MM` offset; `Z`, names and seconds refuse."""
    if type(text) is not str:
        raise CalendarError("invalid_offset", f"{where} must be a string like '+00:00'")
    match = _OFFSET.match(text)
    if not match:
        raise CalendarError("invalid_offset", f"{where} {text!r} must be a fixed offset '+HH:MM' or '-HH:MM'")
    sign, hours, minutes = match.groups()
    hours_i, minutes_i = int(hours), int(minutes)
    if hours_i > 23 or minutes_i > 59:
        raise CalendarError("invalid_offset", f"{where} {text!r} is outside +-23:59")
    delta = _dt.timedelta(hours=hours_i, minutes=minutes_i)
    if sign == "-":
        if delta == _dt.timedelta(0):
            raise CalendarError("invalid_offset", f"{where} '-00:00' is refused; write '+00:00'")
        delta = -delta
    return _dt.timezone(delta)


def _offset_text(offset: _dt.timezone) -> str:
    delta = offset.utcoffset(None)
    assert delta is not None
    total = int(delta.total_seconds())
    sign = "-" if total < 0 else "+"
    total = abs(total)
    return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"


def days_in_year(year: int) -> int:
    """365 or 366 on the proleptic Gregorian calendar (`datetime` decides)."""
    return _dt.date(year, 12, 31).timetuple().tm_yday


def noleap_day_index(day: _dt.date) -> int:
    """0-based day index on a 365-day no-leap calendar (1 January = 0),
    the convention `leaf.stepwat` documents. 29 February has no image and
    refuses: a no-leap consumer cannot receive that date."""
    if day.month == 2 and day.day == 29:
        raise CalendarError("noleap_leap_day", f"{day.isoformat()} has no image on a 365-day no-leap calendar")
    index = day.timetuple().tm_yday - 1
    if days_in_year(day.year) == 366 and (day.month, day.day) > (2, 29):
        index -= 1
    return index


@dataclass(frozen=True)
class CalendarInstant:
    """The calendar image of an exact elapsed offset from a `CalendarStart`."""

    elapsed_s: Fraction
    whole: _dt.datetime  # start + floor(elapsed_s) seconds, fixed offset
    subsecond: Fraction  # 0 <= subsecond < 1, exact

    @property
    def date(self) -> _dt.date:
        return self.whole.date()

    @property
    def day_of_year(self) -> int:
        return self.whole.timetuple().tm_yday

    @property
    def year(self) -> int:
        return self.whole.year

    @property
    def at_midnight(self) -> bool:
        return self.whole.time() == _dt.time(0, 0, 0) and self.subsecond == 0

    def record(self) -> dict[str, Any]:
        return {
            "iso": self.whole.isoformat(),
            "subsecond": fraction_text(self.subsecond),
            "elapsed_s": fraction_text(self.elapsed_s),
            "date": self.date.isoformat(),
            "day_of_year": self.day_of_year,
            "days_in_year": days_in_year(self.year),
        }


@dataclass(frozen=True)
class CalendarStart:
    """Explicit dated origin of global elapsed time zero."""

    start: _dt.datetime

    def __post_init__(self) -> None:
        if type(self.start) is not _dt.datetime or self.start.tzinfo is None or self.start.microsecond != 0:
            raise CalendarError("invalid_calendar_start", "start must be an offset-aware datetime with whole seconds")
        if type(self.start.tzinfo) is not _dt.timezone:
            raise CalendarError("invalid_calendar_start", "start offset must be a fixed datetime.timezone")

    @classmethod
    def parse(cls, text: Any, where: str = "calendar.start") -> "CalendarStart":
        """`YYYY-MM-DDTHH:MM:SS+HH:MM`, nothing else (no 'Z', no fractions)."""
        if type(text) is not str:
            raise CalendarError("invalid_calendar_start", f"{where} must be a string")
        match = _ISO.match(text)
        if not match:
            raise CalendarError("invalid_calendar_start",
                                f"{where} {text!r} must be 'YYYY-MM-DDTHH:MM:SS+HH:MM' with an explicit fixed offset")
        year, month, day, hour, minute, second, offset = match.groups()
        tz = parse_offset(offset, f"{where} offset")
        try:
            start = _dt.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=tz)
        except ValueError as exc:
            raise CalendarError("invalid_calendar_start", f"{where} {text!r}: {exc}") from exc
        if start.isoformat() != text:
            raise CalendarError("invalid_calendar_start", f"{where} {text!r} is not canonical ({start.isoformat()})")
        return cls(start)

    @property
    def offset_text(self) -> str:
        return _offset_text(self.start.tzinfo)  # type: ignore[arg-type]

    def record(self) -> dict[str, Any]:
        return {"schema": CALENDAR_SCHEMA, "start": self.start.isoformat(), "convention": CALENDAR_CONVENTION,
                "utc_offset": self.offset_text, "leap_seconds": LEAP_SECONDS_POLICY, "day_s": DAY_S}

    @classmethod
    def from_input(cls, value: Any, where: str = "calendar") -> "CalendarStart":
        """Closed input table: start, convention and leap_seconds must be exactly the supported ones."""
        required = ("start", "convention", "leap_seconds")
        if type(value) is not dict or set(value) != set(required):
            raise CalendarError("invalid_calendar", f"{where} requires exactly the keys {required}")
        if value["convention"] != CALENDAR_CONVENTION:
            raise CalendarError("unsupported_calendar_convention",
                                f"{where}.convention must be {CALENDAR_CONVENTION!r}")
        if value["leap_seconds"] != LEAP_SECONDS_POLICY:
            raise CalendarError("unsupported_leap_seconds", f"{where}.leap_seconds must be {LEAP_SECONDS_POLICY!r}")
        return cls.parse(value["start"], f"{where}.start")

    def instant(self, elapsed_s: Fraction) -> CalendarInstant:
        if type(elapsed_s) is not Fraction:
            raise CalendarError("invalid_elapsed", "elapsed_s must be a Fraction")
        if elapsed_s < 0:
            raise CalendarError("negative_elapsed", "calendar instants before the scenario start are refused")
        whole_seconds = elapsed_s.numerator // elapsed_s.denominator
        subsecond = elapsed_s - whole_seconds
        try:
            whole = self.start + _dt.timedelta(seconds=whole_seconds)
        except OverflowError as exc:
            raise CalendarError("calendar_range", "instant exceeds the supported calendar range (year 9999)") from exc
        return CalendarInstant(elapsed_s, whole, subsecond)

    def elapsed_of(self, text: Any, where: str) -> Fraction:
        """Exact elapsed seconds of a canonical date-time string (same offset required)."""
        other = CalendarStart.parse(text, where)
        if other.start.utcoffset() != self.start.utcoffset():
            raise CalendarError("offset_mismatch",
                                f"{where} uses offset {other.offset_text}, the scenario uses {self.offset_text}")
        delta = other.start - self.start
        seconds = delta.days * DAY_S + delta.seconds
        if seconds < 0:
            raise CalendarError("before_calendar_start", f"{where} precedes the scenario start")
        return Fraction(seconds)

    def dates_touched(self, start_s: Fraction, end_s: Fraction) -> tuple[_dt.date, ...]:
        """Calendar dates (in the start's fixed offset) whose day intersects the
        half-open interval [start_s, end_s). An empty interval touches nothing."""
        if type(start_s) is not Fraction or type(end_s) is not Fraction:
            raise CalendarError("invalid_elapsed", "interval bounds must be Fractions")
        if end_s < start_s:
            raise CalendarError("reversed_interval", "end precedes start")
        if end_s == start_s:
            return ()
        first = self.instant(start_s).date
        last_instant = self.instant(end_s)
        last = last_instant.date
        if last_instant.at_midnight:  # [.., midnight) does not touch the next day
            last = last - _dt.timedelta(days=1)
        count = (last - first).days + 1
        return tuple(first + _dt.timedelta(days=i) for i in range(count))

    def whole_days(self, start_s: Fraction, end_s: Fraction) -> int | None:
        """Number of whole calendar days if [start_s, end_s) starts at midnight
        and spans an integer number of 86400 s days; otherwise None."""
        if end_s <= start_s or not self.instant(start_s).at_midnight:
            return None
        span = end_s - start_s
        if span.denominator != 1 or span.numerator % DAY_S:
            return None
        return span.numerator // DAY_S


def parse_date(text: Any, where: str) -> _dt.date:
    if type(text) is not str or not _DATE.match(text):
        raise CalendarError("invalid_date", f"{where} must be 'YYYY-MM-DD'")
    try:
        value = _dt.date.fromisoformat(text)
    except ValueError as exc:
        raise CalendarError("invalid_date", f"{where} {text!r}: {exc}") from exc
    if value.isoformat() != text:
        raise CalendarError("invalid_date", f"{where} {text!r} is not canonical")
    return value


__all__.append("parse_date")
