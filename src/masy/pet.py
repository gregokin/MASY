"""Declarative PET method registry and a narrow lazy delegate to SYRUP's `maple_syrup.weather_pet`.

MASY owns temporal coordination, not physical weather calculations. This module
therefore contains NO evapotranspiration, radiation, humidity or polar/cold
equation. It declares which PET methods a scenario may select and which daily
weather fields each needs, and it delegates every physical evaluation to the
ACTUAL owning library, SYRUP's `maple_syrup.weather_pet` (Phase 0B candidate,
first computed method `ecohyd_hargreaves` with the downloaded EcoHyD
coefficients 15.392 / 0.4093 / 1.405, actual Gregorian day of year with the
reference's 365 denominator, analytic polar handling and zero clamped cold
demand; `explicit` PET plus mean temperature; `penman_monteith` declared but
refusing). Nothing from that library is copied here.

Importing this module imports neither NumPy nor `maple_syrup`. The library is
imported lazily by `weather_pet_library()` at validation/derivation time; when
it cannot be imported the refusal (`pet_library_unavailable`) says what to put
on the import path. The import records the module's actual file origin and
SHA-256 so a consumer can bind what was used; that origin is a development
identity, NOT a qualified locked runtime (the existing source-qualified MASY
runtime and its guards are unchanged and are not bypassed by this delegate).
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = [
    "LIBRARY_MODULE",
    "LIBRARY_REQUIRED_NAMES",
    "PET_METHODS",
    "PETError",
    "PETMethod",
    "library_origin",
    "pet_method",
    "weather_pet_library",
]

LIBRARY_MODULE = "maple_syrup.weather_pet"
LIBRARY_REQUIRED_NAMES = ("DailyWeather", "potential_demand", "METHODS", "SOURCE_IDENTITY", "WeatherError",
                          "UnsupportedPETMethodError", "WeatherValidationError")
IMPORT_HINT = ("put the SYRUP source root that provides maple_syrup.weather_pet on the import path (development: "
               "PYTHONPATH=<syrup_candidate>/src; qualified runs: the source-locked MASY runtime), nothing is installed "
               "or substituted here")


class PETError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class PETMethod:
    """Declarative registry row: which daily weather fields the owning library
    needs for this method, whether the bound MAPLE case latitude is required,
    and whether the library implements it. No equation lives here."""

    name: str
    required_fields: tuple[str, ...]
    needs_latitude: bool
    implemented: bool
    owner: str
    note: str


PET_METHODS: dict[str, PETMethod] = {
    "ecohyd_hargreaves": PETMethod(
        "ecohyd_hargreaves", ("tmean_c", "tmin_c", "tmax_c"), True, True, LIBRARY_MODULE,
        "downloaded EcoHyD Hargreaves (15.392 / 0.4093 / 1.405), dated Gregorian J over 365, analytic polar "
        "handling, cold demand clamped to zero by the owning library"),
    "explicit": PETMethod(
        "explicit", ("tmean_c", "pet_mm_day"), False, True, LIBRARY_MODULE,
        "caller-declared daily reference PET (mm/day) plus mean temperature; extrema and latitude are not required"),
    "penman_monteith": PETMethod(
        "penman_monteith", (), True, False, LIBRARY_MODULE,
        "declared physical hook; the owning library refuses it until wind speed, net radiation, pressure and "
        "resistances are defined and qualified"),
}


def pet_method(name: Any) -> PETMethod:
    """The registry row, or a truthful refusal: unknown names refuse, and a
    declared-but-unimplemented method refuses `pet_method_unimplemented`."""
    if type(name) is not str or name not in PET_METHODS:
        raise PETError("pet_method_unsupported", f"pet.method {name!r}; declared methods: {sorted(PET_METHODS)}")
    method = PET_METHODS[name]
    if not method.implemented:
        raise PETError("pet_method_unimplemented",
                       f"pet.method {name!r} is a declared hook without an implementation in {method.owner}; {method.note}")
    return method


def weather_pet_library() -> ModuleType:
    """Lazily import the owning SYRUP library and check the names this delegate
    relies on. Refuses with an actionable message; never substitutes a fallback."""
    try:
        module = importlib.import_module(LIBRARY_MODULE)
    except ImportError as exc:
        raise PETError("pet_library_unavailable", f"cannot import {LIBRARY_MODULE} ({exc}); {IMPORT_HINT}") from exc
    missing = [name for name in LIBRARY_REQUIRED_NAMES if not hasattr(module, name)]
    if missing:
        raise PETError("pet_library_incompatible",
                       f"{LIBRARY_MODULE} at {getattr(module, '__file__', '?')} lacks {missing}; the delegate binds the "
                       "documented Phase 0B names and does not adapt to a different API silently")
    return module


def library_origin(module: ModuleType | None = None) -> dict[str, Any]:
    """Actual file origin and SHA-256 of the imported owning library plus its
    declared equation source identity. Development identity only: it records
    what was imported, it does not certify an immutable qualified runtime."""
    module = weather_pet_library() if module is None else module
    origin = getattr(module, "__file__", None)
    digest = None
    if origin:
        try:
            digest = hashlib.sha256(Path(origin).read_bytes()).hexdigest()
        except OSError:
            digest = None
    source = getattr(module, "SOURCE_IDENTITY", None)
    return {"module": LIBRARY_MODULE, "file": origin, "sha256": digest,
            "declared_methods": list(getattr(module, "METHODS", ())),
            "source_identity": dict(source) if isinstance(source, dict) else source,
            "qualification": "development import recorded by origin; not a source-locked qualified runtime"}


def daily_weather_parameters(module: ModuleType | None = None) -> frozenset[str]:
    """Constructor parameter names of the library's `DailyWeather` (keyword use only)."""
    module = weather_pet_library() if module is None else module
    try:
        return frozenset(inspect.signature(module.DailyWeather).parameters)
    except (TypeError, ValueError) as exc:
        raise PETError("pet_library_incompatible", f"{LIBRARY_MODULE}.DailyWeather signature unavailable ({exc})") from exc


__all__.append("daily_weather_parameters")
