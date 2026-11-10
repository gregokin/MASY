"""Helpers for the Phase 0 scheduler tests: synthetic scenario trees in temp dirs (no native model code).

Weather rows are validated by the OWNING SYRUP library `maple_syrup.weather_pet`
through the lazy delegate in `masy.pet`. Tests that need it are skipped with an
explicit reason unless that module is importable (development: put the SYRUP
PET candidate `src` on PYTHONPATH next to this candidate's `src`); the actual
module origin is reported by `library_report()`. This is a development import,
not a qualified native runtime.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from typing import Any

import _support  # noqa: F401 - puts MASY src on sys.path

from masy.clock import decode_time
from masy.coordinator import AdapterCapabilities, AdapterOutcome
from masy.pet import PETError, library_origin, weather_pet_library
from masy.weather import LatitudeBinding

CAL = {"start": "2027-12-30T00:00:00+00:00", "convention": "proleptic_gregorian", "leap_seconds": "none"}
UNITS = {"temperature": "degC", "relative_humidity": "fraction", "vpd": "kPa", "shortwave_radiation": "MJ_m2_day",
         "pet": "mm_day"}
OWNER = "maple_syrup.weather_pet"


def library_report() -> tuple[bool, str]:
    try:
        origin = library_origin(weather_pet_library())
    except PETError as exc:
        return False, f"{exc.code}: {exc.detail}"
    return True, f"{origin['module']} from {origin['file']} sha256 {origin['sha256']}"


LIBRARY_AVAILABLE, LIBRARY_REPORT = library_report()
requires_library = unittest.skipUnless(
    LIBRARY_AVAILABLE, f"owning PET library unavailable, test deferred explicitly: {LIBRARY_REPORT}")


def lat(value: float = 32.6) -> LatitudeBinding:
    return LatitudeBinding(value, {"kind": "maple_case_yaml", "path": "/case/case.yaml", "sha256": "0" * 64,
                                   "field": "location.latitude_deg"})


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=1) + "\n").encode()
    path.write_bytes(data)
    return sha(data)


def interevent(duration_s: float = 172800, evaporation: bool = True, transpiration: bool = True, growth: bool = True,
               event_id: str = "interevent_x") -> dict[str, Any]:
    return {"schema": "masy.interevent.v1", "event_id": event_id, "duration_s": duration_s,
            "processes": {"evaporation": evaporation, "transpiration": transpiration, "growth": growth}}


def wind(blocks: tuple[tuple[float, float, float], ...] = ((0.42, 270.0, 1800.0), (0.55, 250.0, 1800.0)),
         event_id: str = "wind_x") -> dict[str, Any]:
    return {"schema": "masy.wind_event.v1", "event_id": event_id, "wind_forcing": {"blocks": [
        {"shear_velocity_m_s": s, "direction_deg": d, "duration_s": t} for s, d, t in blocks]}}


def rain(intervals: tuple[tuple[float, float], ...] = ((600.0, 30.0), (1200.0, 12.0)), guard: float = 7200.0,
         event_id: str = "rain_x") -> dict[str, Any]:
    return {"schema": "masy.rain_event.v1", "event_id": event_id, "water_profile_id": "profile",
            "intervals": [{"duration_s": d, "intensity_mm_h": i} for d, i in intervals], "recession_guard_s": guard}


def weather(dates: tuple[str, ...], *, method: str = "ecohyd_hargreaves", humidity: str = "rh", tmean: str | None = None,
            extrema: bool = True, radiation: bool = False, offset: str = "+00:00", tolerance: float | None = None,
            provenance: str = "prescribed", rows_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    hum: dict[str, Any] = {"supplied": humidity}
    if humidity != "none":
        hum["derivation"] = OWNER
    if tolerance is not None:
        hum["consistency_tolerance_kpa"] = tolerance
    tmean_mode = tmean or ("arithmetic_mean_of_tmin_tmax" if extrema else "supplied")
    rows = []
    for index, date in enumerate(dates):
        row: dict[str, Any] = {"date": date}
        if extrema:
            row["tmin_c"] = 1.0 + index
            row["tmax_c"] = 15.0 + index
        if tmean_mode == "supplied":
            row["tmean_c"] = 8.0 + index
        if humidity in ("rh", "both"):
            row["relative_humidity_fraction"] = 0.4
        if humidity in ("vpd", "both"):
            row["vpd_kpa"] = 0.6
        if radiation:
            row["radiation_mj_m2_day"] = 12.0
        if method == "explicit":
            row["pet_mm_day"] = 2.5
        row.update(rows_extra or {})
        rows.append(row)
    return {"schema": "masy.weather.v1", "weather_id": "w", "utc_offset": offset, "sampling": "daily", "units": dict(UNITS),
            "temperature": {"tmean": tmean_mode, "extrema": extrema}, "humidity": hum,
            "radiation": {"supplied": radiation}, "pet": {"method": method},
            "provenance": {"kind": provenance, "description": "test fixture", "source": "unit test"},
            "rows": rows}


WEEK = ("2027-12-28", "2027-12-29", "2027-12-30", "2027-12-31", "2028-01-01", "2028-01-02", "2028-01-03", "2028-01-04",
        "2028-01-05", "2028-01-06")


def build_tree(root: Path, events: list[tuple[str, str, dict[str, Any], str | None]], *, weather_doc: dict[str, Any] | None,
               gap_policy: str = "refuse", requires_continuous: bool = False, calendar: dict[str, Any] | None = None,
               native_epoch_s: float = 0.0, extra: dict[str, Any] | None = None) -> Path:
    """events: (occurrence_id, relative file, document, scheduled_start). Files are written once per path."""
    hashes: dict[str, str] = {}
    for _, rel, doc, _ in events:
        if rel not in hashes:
            hashes[rel] = write_json(root / rel, doc)
    weather_ref = None
    if weather_doc is not None:
        weather_ref = {"file": "weather.json", "sha256": write_json(root / "weather.json", weather_doc)}
    scenario = {"schema": "masy.scenario.v1", "scenario_id": "test_scenario", "calendar": calendar or dict(CAL),
                "native_epoch_s": native_epoch_s, "gap_policy": gap_policy,
                "requires_continuous_soil_moisture": requires_continuous, "weather": weather_ref,
                "events": [{"occurrence_id": oid, "file": rel, "sha256": hashes[rel], "scheduled_start": start}
                           for oid, rel, _, start in events]}
    scenario.update(extra or {})
    write_json(root / "scenario.json", scenario)
    return root / "scenario.json"


def standard_events() -> list[tuple[str, str, dict[str, Any], str | None]]:
    return [("interevent1", "events/ie2d.json", interevent(), None),
            ("wind1", "events/wind.json", wind(), None),
            ("interevent2", "events/ie23h.json", interevent(82800, growth=False, event_id="ie23h"), None),
            ("rain1", "events/rain.json", rain(), None),
            ("interevent3", "events/ie2d.json", interevent(), None),
            ("wind2", "events/wind.json", wind(), None)]


def quiet_events(duration_s: float = 3600.0) -> list[tuple[str, str, dict[str, Any], str | None]]:
    """wind then an interevent with every process disabled: no weather document needed."""
    return [("w", "events/wind.json", wind(), None),
            ("q", "events/quiet.json", interevent(duration_s, False, False, False, event_id="quiet"), None)]


NO_OVERRIDE = object()


class FakeAdapter:
    """Mechanical test adapter: realizes the planned end (rain: rainfall end + `recession_s`). Holds no native
    state, so `staged_until_accepted` is trivially true; `mutate` injects a dispatch modification to test refusal."""

    def __init__(self, kind: str, *, continuous: bool = False, duration_policy: str = "any_positive_duration",
                 calendar_support: str = "proleptic_gregorian_dates", recession_s: float = 100.0, status: str = "complete",
                 end_override: Any = NO_OVERRIDE, raise_exc: BaseException | None = None, commit: bool = True,
                 adapter_id: str | None = None, staging: str = "staged_until_accepted", mutate: Any = None) -> None:
        self.kind, self.continuous, self.duration_policy = kind, continuous, duration_policy
        self.calendar_support, self.recession_s, self.status = calendar_support, recession_s, status
        self.end_override, self.raise_exc, self.commit = end_override, raise_exc, commit
        self.adapter_id = adapter_id or f"fake_{kind}"
        self.staging, self.mutate = staging, mutate
        self.dispatches: list[dict[str, Any]] = []

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(self.kind, self.adapter_id, "test-source", self.continuous, self.duration_policy,
                                   self.calendar_support, "not_provided", self.staging, "synthetic test adapter; no physics")

    def run(self, dispatch: dict[str, Any]) -> AdapterOutcome:
        self.dispatches.append(json.loads(json.dumps(dispatch)))
        if self.raise_exc is not None:
            raise self.raise_exc
        edges = dispatch["planned_edges"]["edges"]
        end = decode_time(edges[-1]["native"])
        if self.kind == "rain":
            end = decode_time(edges[-2]["native"]) + self.recession_s
        if self.end_override is not NO_OVERRIDE:
            end = self.end_override
        if self.mutate is not None:
            end = self.mutate(dispatch, end)
        after = end if (self.commit and self.kind != "interevent") else None
        return AdapterOutcome(self.status, end, None, after, returned_elapsed_s=None, evidence={"fake": True})


def adapters(**overrides: Any) -> dict[str, FakeAdapter]:
    base = {"wind": FakeAdapter("wind"), "rain": FakeAdapter("rain"), "interevent": FakeAdapter("interevent", continuous=True)}
    base.update(overrides)
    return base
