"""Shared daily weather stream for SYRUP and LEAF (schema and binding only; physics delegated).

One source-bound weather document (`masy.weather.v1`) is the ONLY weather
stream of a scenario: SYRUP evaporative demand and LEAF temperature /
humidity / radiation responses read the same validated rows, the same
calendar dates and the same derived humidity values. MASY owns the schema,
the calendar binding and the provenance record; every physical evaluation
(saturation vapour pressure, RH/VPD derivation and consistency, Hargreaves
PET, polar/cold behaviour) is performed by the owning SYRUP library
`maple_syrup.weather_pet` (`DailyWeather`, `potential_demand`), reached through
the narrow lazy delegate in `masy.pet`. No formula is duplicated here, so the
two models cannot disagree about atmospheric conditions.

Row fields use the library's names so one record maps 1:1 onto `DailyWeather`:
`date`, `tmean_c`, optional `tmin_c`/`tmax_c` (together), optional
`relative_humidity_fraction` (0..1, never percent), `vpd_kpa`,
`radiation_mj_m2_day`, `pet_mm_day`. Declarations are closed and explicit:

- `units`: exactly `degC`, RH `fraction`, VPD `kPa`, radiation `MJ_m2_day`,
  PET `mm_day`; any other spelling refuses rather than being converted.
- `temperature.extrema` (bool) says whether rows carry `tmin_c`/`tmax_c`;
  `temperature.tmean` is `supplied` (rows carry `tmean_c`) or
  `arithmetic_mean_of_tmin_tmax` (a declared MASY convention, flagged
  `derived`, allowed only with extrema). `ecohyd_hargreaves` and any humidity
  field require extrema (`extrema_required`); `explicit` PET with mean
  temperature works without extrema or latitude.
- `humidity.supplied`: `none` / `rh` / `vpd` / `both`; `humidity.derivation`
  must name the owning library (`maple_syrup.weather_pet`); the relation it
  applied is recorded from the library per row. With `both`, an optional
  `consistency_tolerance_kpa` is passed to the library only when its
  `DailyWeather` exposes such a parameter; otherwise the library default
  applies and a declared tolerance refuses (`humidity_tolerance_unsupported`).
- `radiation.supplied` (bool); `pet.method` from the `masy.pet` registry
  (`ecohyd_hargreaves`, `explicit`; `penman_monteith` refuses truthfully).
- `provenance.kind` is the library's label (`observed` / `prescribed` /
  `estimated`) with description and source; simplified forcing declares
  `prescribed` or `estimated`, never `observed`.
- Rows are contiguous calendar days (missing-data policy: contiguous rows
  required, no nulls) in the declared fixed `utc_offset`, which must equal the
  scenario calendar offset.

Latitude is NOT a weather field. It is MAPLE's existing `case.yaml`
`location.latitude_deg` (`maple/core/parameters/location.py`: descriptive
metadata, no default, omitted != 0). `latitude_binding_from_case_yaml` reads
that section with the same closed rules and records the file identity so the
PET consumer binds an actual value and source, never an equator default.
MAPLE's own resolver remains the authority at case compile time.

Importing this module imports neither NumPy nor `maple_syrup`; `from_input`
and `daily_pet` import the owning library lazily and refuse with an actionable
message when it is unavailable.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from masy.calendar import CalendarError, parse_date, parse_offset
from masy.clock import ClockError, decode_time, encode_time
from masy.pet import PETError, PETMethod, daily_weather_parameters, library_origin, pet_method, weather_pet_library
from masy.scenario_io import IOLimits, ScenarioIOError, canonical_json, load_document

__all__ = [
    "CANONICAL_UNITS",
    "freeze",
    "thaw",
    "HUMIDITY_OWNER",
    "LATITUDE_SCHEMA",
    "PROVENANCE_KINDS",
    "WEATHER_SCHEMA",
    "LatitudeBinding",
    "WeatherError",
    "WeatherSeries",
    "latitude_binding_from_case_yaml",
    "load_weather",
]

WEATHER_SCHEMA = "masy.weather.v1"
LATITUDE_SCHEMA = "masy.latitude_binding.v1"
HUMIDITY_OWNER = "maple_syrup.weather_pet"
CANONICAL_UNITS = {"temperature": "degC", "relative_humidity": "fraction", "vpd": "kPa",
                   "shortwave_radiation": "MJ_m2_day", "pet": "mm_day"}
PROVENANCE_KINDS = ("observed", "prescribed", "estimated")
_TOP_KEYS = ("schema", "weather_id", "utc_offset", "sampling", "units", "temperature", "humidity", "radiation",
             "pet", "provenance", "rows")
_TMEAN_MODES = ("supplied", "arithmetic_mean_of_tmin_tmax")
_HUMIDITY_MODES = ("none", "rh", "vpd", "both")
_TOLERANCE_PARAMETERS = ("humidity_consistency_kpa", "humidity_tolerance_kpa", "consistency_tolerance_kpa")
_MAPLE_LOCATION_SEMANTICS = ("MAPLE core/parameters/location.py: descriptive metadata, degrees north positive, "
                             "no default when omitted, null refused; first physical use (PET) is bound here explicitly")


class WeatherError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _closed(value: Any, names: tuple[str, ...], where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise WeatherError("invalid_keys", f"{where} requires exactly the keys {names}")
    return value


def _text(value: Any, where: str, *, maximum: int = 4096) -> str:
    if type(value) is not str or not 0 < len(value) <= maximum or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise WeatherError("invalid_text", f"{where} must be a non-empty printable string of at most {maximum} chars")
    return value


def _number(value: Any, where: str) -> float:
    if type(value) not in (int, float):
        raise WeatherError("invalid_number", f"{where} must be a plain int or float")
    if type(value) is int:
        converted = float(value)
        if converted != value:
            raise WeatherError("invalid_number", f"{where} is not exactly representable as binary64")
        value = converted
    if not math.isfinite(value):
        raise WeatherError("nonfinite_number", f"{where} must be finite")
    if value == 0 and math.copysign(1.0, value) < 0:
        raise WeatherError("negative_zero", f"{where} is -0.0; refused")
    return value


def _choice(value: Any, options: tuple[str, ...], where: str) -> str:
    if type(value) is not str or value not in options:
        raise WeatherError("unsupported_declaration", f"{where} must be one of {list(options)}")
    return value


def _encode(value: Any) -> Any:
    if type(value) is float:
        return encode_time(value)
    if isinstance(value, Mapping):
        return {k: _encode(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return [_encode(v) for v in value]
    return value


def freeze(value: Any) -> Any:
    """Read-only view of plain data: mappings become `MappingProxyType`, lists become tuples (recursively).
    Accepted source-bound records are stored frozen so no public attribute lets a caller change them."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if type(value) in (list, tuple):
        return tuple(freeze(v) for v in value)
    return value


def thaw(value: Any) -> Any:
    """Independent plain-data copy of a frozen (or plain) record."""
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return [thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class LatitudeBinding:
    """An actual latitude value bound to its source identity (never a default)."""

    latitude_deg: float
    source: dict[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.latitude_deg, bool) or type(self.latitude_deg) is not float:
            raise WeatherError("invalid_latitude", "latitude_deg must be a float")
        if not math.isfinite(self.latitude_deg) or not -90.0 <= self.latitude_deg <= 90.0:
            raise WeatherError("invalid_latitude", f"latitude_deg {self.latitude_deg!r} outside [-90, 90]")
        _closed(self.source, ("kind", "path", "sha256", "field"), "latitude.source")
        if self.source["kind"] != "maple_case_yaml" or self.source["field"] != "location.latitude_deg":
            raise WeatherError("invalid_latitude_source", "latitude must come from MAPLE case.yaml location.latitude_deg")
        _text(self.source["path"], "latitude.source.path")
        sha = self.source["sha256"]
        if type(sha) is not str or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise WeatherError("invalid_latitude_source", "latitude.source.sha256 must be 64 lowercase hex digits")
        object.__setattr__(self, "source", freeze(self.source))  # read-only identity after validation

    def record(self) -> dict[str, Any]:
        return {"schema": LATITUDE_SCHEMA, "latitude_deg": encode_time(self.latitude_deg), "source": thaw(self.source),
                "semantics": _MAPLE_LOCATION_SEMANTICS}

    @classmethod
    def from_record(cls, record: Any) -> "LatitudeBinding":
        _closed(record, ("schema", "latitude_deg", "source", "semantics"), "latitude binding record")
        if record["schema"] != LATITUDE_SCHEMA or record["semantics"] != _MAPLE_LOCATION_SEMANTICS:
            raise WeatherError("invalid_latitude_record", "unsupported latitude binding record")
        try:
            value = decode_time(record["latitude_deg"])
        except ClockError as exc:
            raise WeatherError("invalid_latitude_record", str(exc)) from exc
        return cls(value, dict(record["source"]))


def latitude_binding_from_case_yaml(path: Path, limits: IOLimits) -> LatitudeBinding:
    """Read `location.latitude_deg` from an actual MAPLE case.yaml with MAPLE's
    closed semantics: omitted section refuses `latitude_unspecified`, null
    refuses `latitude_null`, unknown keys / non-numbers / out of range refuse.
    The returned binding carries the file path and SHA-256."""
    path = Path(path)
    try:
        value, pin = load_document(path, limits, where=str(path))
    except ScenarioIOError as exc:
        raise WeatherError("case_yaml_unreadable", str(exc)) from exc
    if type(value) is not dict:
        raise WeatherError("case_yaml_invalid", f"{path} is not a mapping")
    if "location" not in value:
        raise WeatherError("latitude_unspecified",
                           f"{path} has no `location` section; PET needs an authored latitude (no equator default)")
    section = value["location"]
    if section is None:
        raise WeatherError("latitude_null", f"{path} `location` is null; MAPLE refuses this too")
    if type(section) is not dict or set(section) != {"latitude_deg"}:
        raise WeatherError("latitude_invalid", f"{path} `location` must be a mapping with exactly `latitude_deg`")
    raw = section["latitude_deg"]
    if raw is None:
        raise WeatherError("latitude_null", f"{path} `location.latitude_deg` is null")
    if isinstance(raw, bool) or type(raw) not in (int, float) or not math.isfinite(float(raw)):
        raise WeatherError("latitude_invalid", f"{path} `location.latitude_deg` must be a finite non-Boolean number")
    numeric = float(raw)
    if not -90.0 <= numeric <= 90.0:
        raise WeatherError("latitude_invalid", f"{path} `location.latitude_deg` {numeric!r} outside [-90, 90]")
    return LatitudeBinding(numeric, {"kind": "maple_case_yaml", "path": pin["path"], "sha256": pin["sha256"],
                                     "field": "location.latitude_deg"})


def _row_fields(tmean_mode: str, extrema: bool, humidity_mode: str, radiation: bool, method: PETMethod) -> tuple[str, ...]:
    fields = ["date"]
    if tmean_mode == "supplied":
        fields.append("tmean_c")
    if extrema:
        fields += ["tmin_c", "tmax_c"]
    if humidity_mode in ("rh", "both"):
        fields.append("relative_humidity_fraction")
    if humidity_mode in ("vpd", "both"):
        fields.append("vpd_kpa")
    if radiation:
        fields.append("radiation_mj_m2_day")
    if method.name == "explicit":
        fields.append("pet_mm_day")
    return tuple(fields)


def _library_error(exc: BaseException, where: str) -> WeatherError:
    return WeatherError("weather_library_refused", f"{where}: {HUMIDITY_OWNER} refused the record: {str(exc)[:600]}")


@dataclass(frozen=True)
class WeatherSeries:
    weather_id: str
    utc_offset: str
    tmean_mode: str
    extrema_supplied: bool
    humidity_mode: str
    consistency_tolerance_kpa: float | None
    radiation_supplied: bool
    pet_method_name: str
    provenance: Mapping[str, str]
    rows: tuple[Mapping[str, Any], ...]  # normalized: date (iso), library-named floats, derived values from the library
    library: Mapping[str, Any]  # origin of the library that validated the rows (development identity)
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        # Accepted records are read-only; `coverage()`/`record()` return independent copies.
        object.__setattr__(self, "provenance", freeze(self.provenance))
        object.__setattr__(self, "library", freeze(self.library))
        object.__setattr__(self, "rows", tuple(freeze(row) for row in self.rows))

    @property
    def pet(self) -> PETMethod:
        return pet_method(self.pet_method_name)

    @property
    def first_date(self) -> _dt.date:
        return _dt.date.fromisoformat(self.rows[0]["date"])

    @property
    def last_date(self) -> _dt.date:
        return _dt.date.fromisoformat(self.rows[-1]["date"])

    @classmethod
    def from_input(cls, value: Any, where: str = "weather", source_sha256: str | None = None) -> "WeatherSeries":
        _closed(value, _TOP_KEYS, where)
        if value["schema"] != WEATHER_SCHEMA:
            raise WeatherError("invalid_schema", f"{where}.schema must be {WEATHER_SCHEMA!r}")
        weather_id = _text(value["weather_id"], f"{where}.weather_id", maximum=128)
        try:
            parse_offset(value["utc_offset"], f"{where}.utc_offset")
        except CalendarError as exc:
            raise WeatherError("invalid_offset", str(exc)) from exc
        if value["sampling"] != "daily":
            raise WeatherError("unsupported_declaration", f"{where}.sampling must be 'daily'")
        units = _closed(value["units"], tuple(CANONICAL_UNITS), f"{where}.units")
        for name, unit in CANONICAL_UNITS.items():
            if units[name] != unit:
                raise WeatherError("unsupported_unit", f"{where}.units.{name} must be {unit!r} (no silent conversion)")
        temperature = _closed(value["temperature"], ("tmean", "extrema"), f"{where}.temperature")
        tmean_mode = _choice(temperature["tmean"], _TMEAN_MODES, f"{where}.temperature.tmean")
        extrema = temperature["extrema"]
        if type(extrema) is not bool:
            raise WeatherError("unsupported_declaration", f"{where}.temperature.extrema must be a bool")
        if tmean_mode == "arithmetic_mean_of_tmin_tmax" and not extrema:
            raise WeatherError("extrema_required", f"{where}: deriving tmean needs temperature.extrema = true")
        humidity = value["humidity"]
        if type(humidity) is not dict or "supplied" not in humidity:
            raise WeatherError("invalid_keys", f"{where}.humidity requires 'supplied'")
        humidity_mode = _choice(humidity["supplied"], _HUMIDITY_MODES, f"{where}.humidity.supplied")
        tolerance = None
        if humidity_mode == "none":
            _closed(humidity, ("supplied",), f"{where}.humidity")
        else:
            allowed = ("supplied", "derivation") + (("consistency_tolerance_kpa",) if humidity_mode == "both" else ())
            if set(humidity) - set(allowed) or "derivation" not in humidity:
                raise WeatherError("invalid_keys", f"{where}.humidity allows exactly {allowed} (tolerance optional)")
            if humidity["derivation"] != HUMIDITY_OWNER:
                raise WeatherError("unsupported_declaration",
                                   f"{where}.humidity.derivation must name the owning library {HUMIDITY_OWNER!r}")
            if "consistency_tolerance_kpa" in humidity:
                tolerance = _number(humidity["consistency_tolerance_kpa"], f"{where}.humidity.consistency_tolerance_kpa")
                if tolerance < 0:
                    raise WeatherError("invalid_number", f"{where}.humidity.consistency_tolerance_kpa must be >= 0")
            if not extrema:
                raise WeatherError("extrema_required",
                                   f"{where}: the owning library's humidity relation needs tmin_c/tmax_c (temperature.extrema)")
        radiation = _closed(value["radiation"], ("supplied",), f"{where}.radiation")["supplied"]
        if type(radiation) is not bool:
            raise WeatherError("unsupported_declaration", f"{where}.radiation.supplied must be a bool")
        try:
            method = pet_method(_closed(value["pet"], ("method",), f"{where}.pet")["method"])
        except PETError as exc:
            raise WeatherError(exc.code, str(exc)) from exc
        if "tmin_c" in method.required_fields and not extrema:
            raise WeatherError("extrema_required", f"{where}: pet.method {method.name!r} needs tmin_c/tmax_c (temperature.extrema)")
        provenance = _closed(value["provenance"], ("kind", "description", "source"), f"{where}.provenance")
        kind = _choice(provenance["kind"], PROVENANCE_KINDS, f"{where}.provenance.kind")
        prov = {"kind": kind, "description": _text(provenance["description"], f"{where}.provenance.description"),
                "source": _text(provenance["source"], f"{where}.provenance.source")}
        raw_rows = value["rows"]
        if type(raw_rows) is not list or not raw_rows:
            raise WeatherError("empty_weather", f"{where}.rows must be a non-empty list")
        fields = _row_fields(tmean_mode, extrema, humidity_mode, radiation, method)
        try:
            library = weather_pet_library()
            parameters = daily_weather_parameters(library)
        except PETError as exc:
            raise WeatherError(exc.code, str(exc)) from exc
        tolerance_parameter = next((name for name in _TOLERANCE_PARAMETERS if name in parameters), None)
        if tolerance is not None and tolerance_parameter is None:
            raise WeatherError("humidity_tolerance_unsupported",
                               f"{where}.humidity.consistency_tolerance_kpa is declared but {HUMIDITY_OWNER}.DailyWeather "
                               f"exposes no tolerance parameter ({sorted(parameters)}); omit it to use the library default")
        rows: list[dict[str, Any]] = []
        previous: _dt.date | None = None
        for index, raw in enumerate(raw_rows):
            label = f"{where}.rows[{index}]"
            _closed(raw, fields, label)
            try:
                day = parse_date(raw["date"], f"{label}.date")
            except CalendarError as exc:
                raise WeatherError("invalid_date", str(exc)) from exc
            if previous is not None and day != previous + _dt.timedelta(days=1):
                raise WeatherError("weather_rows_not_contiguous",
                                   f"{label}.date {day.isoformat()} does not follow {previous.isoformat()} by one day")
            previous = day
            rows.append(_normalize_row(raw, day, label, tmean_mode, extrema, kind, library, tolerance, tolerance_parameter))
        return cls(weather_id, value["utc_offset"], tmean_mode, extrema, humidity_mode, tolerance, radiation, method.name,
                   prov, tuple(rows), library_origin(library), source_sha256)

    def record(self) -> dict[str, Any]:
        return {"schema": WEATHER_SCHEMA, "weather_id": self.weather_id, "utc_offset": self.utc_offset,
                "sampling": "daily", "units": dict(CANONICAL_UNITS),
                "temperature": {"tmean": self.tmean_mode, "extrema": self.extrema_supplied},
                "humidity": {"supplied": self.humidity_mode, "derivation": HUMIDITY_OWNER,
                             "consistency_tolerance_kpa": None if self.consistency_tolerance_kpa is None
                             else encode_time(self.consistency_tolerance_kpa)},
                "radiation": {"supplied": self.radiation_supplied},
                "pet": {"method": self.pet_method_name, "owner": self.pet.owner, "needs_latitude": self.pet.needs_latitude,
                        "required_fields": list(self.pet.required_fields)},
                "provenance": thaw(self.provenance), "source_sha256": self.source_sha256,
                "library": _encode(self.library),
                "first_date": self.rows[0]["date"], "last_date": self.rows[-1]["date"],
                "rows": [_encode(row) for row in self.rows]}

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.record()).encode("ascii")).hexdigest()

    def coverage(self, dates: tuple[_dt.date, ...]) -> tuple[dict[str, Any], ...]:
        """Independent plain copies of the rows for exactly these dates; any uncovered date refuses."""
        by_date = {row["date"]: row for row in self.rows}
        out = []
        for day in dates:
            row = by_date.get(day.isoformat())
            if row is None:
                raise WeatherError("weather_coverage_missing",
                                   f"no weather row for {day.isoformat()} (series {self.rows[0]['date']}..{self.rows[-1]['date']})")
            out.append(thaw(row))
        return tuple(out)

    def daily_weather(self, row: Mapping[str, Any], library: Any = None) -> Any:
        """The owning library's `DailyWeather` for one normalized row (keyword construction only)."""
        try:
            library = weather_pet_library() if library is None else library
        except PETError as exc:
            raise WeatherError(exc.code, str(exc)) from exc
        kwargs: dict[str, Any] = {"date": _dt.date.fromisoformat(row["date"]), "tmean_c": row["tmean_c"],
                                  "tmin_c": row.get("tmin_c"), "tmax_c": row.get("tmax_c"),
                                  "provenance": self.provenance["kind"]}
        for name in ("relative_humidity_fraction", "vpd_kpa", "radiation_mj_m2_day", "pet_mm_day"):
            if name in row:
                kwargs[name] = row[name]
        if self.consistency_tolerance_kpa is not None:
            parameter = next((n for n in _TOLERANCE_PARAMETERS if n in daily_weather_parameters(library)), None)
            if parameter is None:
                raise WeatherError("humidity_tolerance_unsupported", f"{HUMIDITY_OWNER}.DailyWeather exposes no tolerance parameter")
            kwargs[parameter] = self.consistency_tolerance_kpa
        try:
            return library.DailyWeather(**kwargs)
        except library.WeatherError as exc:
            raise _library_error(exc, row["date"]) from exc

    def daily_pet(self, row: Mapping[str, Any], latitude: LatitudeBinding | None) -> dict[str, Any]:
        """PET for one covered row, computed by the owning library under the
        selected method; latitude is the bound MAPLE value or None (methods
        that need it refuse here before the library is called)."""
        method = self.pet
        if method.needs_latitude and latitude is None:
            raise WeatherError("latitude_required",
                               f"pet.method {method.name!r} needs the bound MAPLE case latitude; there is no equator default")
        try:
            library = weather_pet_library()
        except PETError as exc:
            raise WeatherError(exc.code, str(exc)) from exc
        weather = self.daily_weather(row, library)
        try:
            demand = library.potential_demand(weather, None if latitude is None else latitude.latitude_deg, method.name)
        except library.UnsupportedPETMethodError as exc:
            raise WeatherError("pet_method_unimplemented", f"{HUMIDITY_OWNER}: {exc}") from exc
        except library.WeatherError as exc:
            raise _library_error(exc, row["date"]) from exc
        return {"method": method.name, "date": row["date"], "pet_mm_day": float(demand.pet_mm_day),
                "clamped": bool(demand.clamped), "ra_equiv_mm_day": demand.ra_equiv_mm_day,
                "provenance": demand.provenance(), "library": library_origin(library),
                "latitude_source": None if latitude is None else thaw(latitude.source),
                "qualification": "daily reference value from the owning library; partition into soil evaporation and "
                                 "plant transpiration is owned by SYRUP/LEAF and not spent twice"}


def _normalize_row(raw: dict[str, Any], day: _dt.date, label: str, tmean_mode: str, extrema: bool, provenance: str,
                   library: Any, tolerance: float | None, tolerance_parameter: str | None) -> dict[str, Any]:
    row: dict[str, Any] = {"date": day.isoformat()}
    derived: dict[str, Any] = {}
    if extrema:
        row["tmin_c"] = _number(raw["tmin_c"], f"{label}.tmin_c")
        row["tmax_c"] = _number(raw["tmax_c"], f"{label}.tmax_c")
    if tmean_mode == "supplied":
        row["tmean_c"] = _number(raw["tmean_c"], f"{label}.tmean_c")
    else:
        row["tmean_c"] = (row["tmin_c"] + row["tmax_c"]) / 2.0
        derived["tmean_c"] = "arithmetic_mean_of_tmin_tmax (declared MASY convention)"
    for name in ("relative_humidity_fraction", "vpd_kpa", "radiation_mj_m2_day", "pet_mm_day"):
        if name in raw:
            row[name] = _number(raw[name], f"{label}.{name}")
    kwargs: dict[str, Any] = {"date": day, "tmean_c": row["tmean_c"], "tmin_c": row.get("tmin_c"), "tmax_c": row.get("tmax_c"),
                              "provenance": provenance}
    kwargs.update({name: row[name] for name in ("relative_humidity_fraction", "vpd_kpa", "radiation_mj_m2_day", "pet_mm_day")
                   if name in row})
    if tolerance is not None and tolerance_parameter is not None:
        kwargs[tolerance_parameter] = tolerance
    try:
        weather = library.DailyWeather(**kwargs)
    except library.WeatherError as exc:
        raise _library_error(exc, label) from exc
    except TypeError as exc:
        raise WeatherError("pet_library_incompatible", f"{label}: {HUMIDITY_OWNER}.DailyWeather rejected the keyword record ({exc})") from exc
    record = weather.record()
    if record.get("humidity_sources"):
        derived.update({"rh_fraction": record["rh_fraction"], "vapor_pressure_deficit_kpa": record["vapor_pressure_deficit_kpa"],
                        "saturation_vapor_pressure_mean_kpa": record["saturation_vapor_pressure_mean_kpa"],
                        "humidity_sources": dict(record["humidity_sources"]), "humidity_relation": record["humidity_relation"],
                        "owner": HUMIDITY_OWNER})
    row["derived"] = derived
    return row


def load_weather(path: Path, limits: IOLimits) -> tuple[WeatherSeries, dict[str, Any]]:
    """Strictly load one weather document; returns (series, source pin)."""
    try:
        value, pin = load_document(path, limits, where=str(path))
    except ScenarioIOError as exc:
        raise WeatherError(exc.code, str(exc)) from exc
    return WeatherSeries.from_input(value, where=str(path), source_sha256=pin["sha256"]), pin
