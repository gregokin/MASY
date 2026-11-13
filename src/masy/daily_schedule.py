"""Immutable daily plan over a parsed `masy.daily_csv.DailyScenario` (metadata only).

The plan enumerates every nominal Gregorian date from `start_date` to `end_date`
inclusive, including dry and wind-free days, leap days and year crossings. Each
day carries its weather, its wind occurrences in authoritative row order, the
process flags and the requested cadence flags. Cadences anchor to the scenario
start: an interval of N is due after N accepted days (accepted day number % N == 0).
There is no implicit final flush and no biological year-end decision; calendar
boundary information is exposed only.

NEW clock convention (separate from `masy.clock` and every older scenario
meaning): nominal calendar days are distinct from native physical elapsed
seconds. Each day is wind(s), then precipitation (including full storm
recession when required), then ONE native daily soil/E update, then LEAF
transactions when available. Wind/storm time can make a nominal year longer
than 365/366 physical days; nothing here predicts it.

The precipitation selection is the INTENDED request only: P=0 none, 0<P<=T a
recharge attempt (which native code may still turn into a full storm on a
closed-column capacity failure), P>T a required native full storm. Soil
capacity, recession, physical duration, budgets and acceptance belong to native
SYRUP and are not predicted. No model is executed here.
"""

from __future__ import annotations

import datetime as _dt
import copy
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from masy.daily_csv import DailyCsvError, DailyScenario, DailyWeatherDay, WindOccurrence

__all__ = [
    "CLOCK_CONVENTION",
    "PRECIPITATION_NONE",
    "PRECIPITATION_RECHARGE_ATTEMPT",
    "PRECIPITATION_FULL_STORM",
    "DailyPlan",
    "DailyPlanDay",
    "build_daily_plan",
    "select_precipitation",
]

PRECIPITATION_NONE = "none"
PRECIPITATION_RECHARGE_ATTEMPT = "recharge_attempt"
PRECIPITATION_FULL_STORM = "full_storm_required"
CLOCK_CONVENTION = {
    "name": "masy.daily_clock.v1",
    "nominal_day": "Gregorian calendar date; independent of native physical elapsed seconds",
    "per_day_order": ["wind occurrences in row order", "precipitation (full storm recession if required)",
                      "one native daily soil/E update", "LEAF transactions when available"],
    "subdaily_interevent_processes": False,
    "biological_catch_up_over_transport_durations": False,
    "physical_year_may_exceed_nominal_year": True,
    "cadence_anchor": "scenario start; interval N is due after N accepted days",
    "implicit_final_growth_flush": False,
}


def select_precipitation(precipitation_mm: float, threshold_mm: float) -> str:
    """Intended precipitation path only: P=0 none; 0<P<=T recharge attempt; P>T full storm."""
    if precipitation_mm == 0.0:
        return PRECIPITATION_NONE
    return PRECIPITATION_RECHARGE_ATTEMPT if precipitation_mm <= threshold_mm else PRECIPITATION_FULL_STORM


@dataclass(frozen=True)
class DailyPlanDay:
    day_number: int
    date: _dt.date
    weather: DailyWeatherDay
    winds: tuple[WindOccurrence, ...]
    precipitation_selection: str
    recharge_may_fall_back_to_full_storm: bool
    evaporation_enabled: bool
    transpiration_enabled: bool
    growth_enabled: bool
    growth_due: bool
    output_due: bool
    checkpoint_due: bool
    event_boundary_output_requested: bool
    is_final_day: bool

    @property
    def is_leap_day(self) -> bool:
        return self.date.month == 2 and self.date.day == 29

    @property
    def is_calendar_year_start(self) -> bool:
        return self.date.month == 1 and self.date.day == 1

    @property
    def is_calendar_year_end(self) -> bool:
        return self.date.month == 12 and self.date.day == 31

    def record(self, *, include_winds: bool = True) -> dict[str, Any]:
        wet = self.weather.wet_duration_s
        return {
            "day_number": self.day_number, "date": self.date.isoformat(),
            "calendar": {"year": self.date.year, "day_of_year": self.date.timetuple().tm_yday,
                         "is_leap_day": self.is_leap_day, "is_calendar_year_start": self.is_calendar_year_start,
                         "is_calendar_year_end": self.is_calendar_year_end},
            "weather": self.weather.record(), "nominal_wet_period_s": wet,
            "wind_occurrences": [w.record() for w in self.winds] if include_winds else [],
            "precipitation_selection": self.precipitation_selection,
            "recharge_may_fall_back_to_full_storm": self.recharge_may_fall_back_to_full_storm,
            "evaporation_enabled": self.evaporation_enabled, "transpiration_enabled": self.transpiration_enabled,
            "growth_enabled": self.growth_enabled, "growth_due": self.growth_due,
            "output_due": self.output_due, "checkpoint_due": self.checkpoint_due,
            "event_boundary_output_requested": self.event_boundary_output_requested,
            "is_final_day": self.is_final_day}


@dataclass(frozen=True)
class DailyPlan:
    scenario: DailyScenario

    def __post_init__(self) -> None:
        if type(self.scenario) is not DailyScenario:
            raise DailyCsvError("invalid_scenario", "DailyPlan requires a parsed DailyScenario")
        settings, weather = self.scenario.settings, self.scenario.weather
        if (settings.end_date < settings.start_date or not weather.days
                or settings.start_date < weather.first_date or settings.end_date > weather.last_date):
            raise DailyCsvError("weather_coverage", "weather must cover the inclusive scenario range")
        previous = None
        for event in self.scenario.occurrences:
            if not settings.start_date <= event.date <= settings.end_date:
                raise DailyCsvError("event_date_out_of_range", "wind occurrence is outside the scenario range")
            if previous is not None and event.date < previous:
                raise DailyCsvError("event_dates_out_of_order", "wind occurrence dates must be nondecreasing")
            previous = event.date

    @property
    def day_count(self) -> int:
        return self.scenario.settings.day_count

    def __iter__(self) -> Iterator[DailyPlanDay]:
        return self.days()

    def days(self) -> Iterator[DailyPlanDay]:
        s = self.scenario.settings
        weather = self.scenario.weather
        offset = weather.day_index(s.start_date)
        occurrences = self.scenario.occurrences
        cursor = 0
        for number in range(1, s.day_count + 1):
            date = s.start_date + _dt.timedelta(days=number - 1)
            day_weather = weather.days[offset + number - 1]
            if day_weather.date != date:
                raise DailyCsvError("weather_coverage", "internal weather/date misalignment")
            winds = []
            while cursor < len(occurrences) and occurrences[cursor].date == date:
                winds.append(occurrences[cursor])
                cursor += 1
            selection = select_precipitation(day_weather.precipitation_mm, weather.precipitation_threshold_mm)
            yield DailyPlanDay(
                day_number=number, date=date, weather=day_weather, winds=tuple(winds),
                precipitation_selection=selection,
                recharge_may_fall_back_to_full_storm=selection == PRECIPITATION_RECHARGE_ATTEMPT,
                evaporation_enabled=s.evaporation_enabled, transpiration_enabled=s.transpiration_enabled,
                growth_enabled=s.growth_enabled,
                growth_due=s.growth_enabled and number % s.growth_interval_days == 0,
                output_due=number % s.output_interval_days == 0,
                checkpoint_due=number % s.checkpoint_interval_days == 0,
                event_boundary_output_requested=s.output_at_event_boundaries and bool(
                    winds or selection != PRECIPITATION_NONE),
                is_final_day=number == s.day_count)

    def summary(self) -> dict[str, Any]:
        counts = {"days": 0, "days_with_wind": 0, "wind_occurrences": 0, PRECIPITATION_NONE: 0,
                  PRECIPITATION_RECHARGE_ATTEMPT: 0, PRECIPITATION_FULL_STORM: 0, "growth_due_days": 0,
                  "output_due_days": 0, "checkpoint_due_days": 0, "event_boundary_output_days": 0,
                  "leap_days": 0, "calendar_year_end_days": 0}
        years: set[int] = set()
        for day in self.days():
            counts["days"] += 1
            counts["days_with_wind"] += bool(day.winds)
            counts["wind_occurrences"] += len(day.winds)
            counts[day.precipitation_selection] += 1
            counts["growth_due_days"] += day.growth_due
            counts["output_due_days"] += day.output_due
            counts["checkpoint_due_days"] += day.checkpoint_due
            counts["event_boundary_output_days"] += day.event_boundary_output_requested
            counts["leap_days"] += day.is_leap_day
            counts["calendar_year_end_days"] += day.is_calendar_year_end
            years.add(day.date.year)
        s = self.scenario.settings
        return {"start_date": s.start_date.isoformat(), "end_date": s.end_date.isoformat(),
                "calendar_years_touched": sorted(years),
                "precipitation_threshold_mm": self.scenario.weather.precipitation_threshold_mm,
                "precipitation_selection_counts": {k: counts.pop(k) for k in (
                    PRECIPITATION_NONE, PRECIPITATION_RECHARGE_ATTEMPT, PRECIPITATION_FULL_STORM)},
                "counts": counts, "clock": copy.deepcopy(CLOCK_CONVENTION),
                "precipitation_note": "intended selection only; native SYRUP decides capacity fallback, "
                                      "recession, physical duration and acceptance"}


def build_daily_plan(scenario: DailyScenario) -> DailyPlan:
    return DailyPlan(scenario)
