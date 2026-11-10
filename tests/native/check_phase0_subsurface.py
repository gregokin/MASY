#!/usr/bin/env python
"""Bounded native Phase 0 qualification DRIVER: dated MASY Coordinator -> existing SYRUP soil column (interevent only).

NOT collected automatically (no `test_` prefix, no package `__init__`). Run it explicitly from an interpreter that
has NumPy (and Numba / CuPy for those backends), e.g. the existing MAPLE virtual environments:

    /home/okin/MAPLE/.venv/bin/python -I tests/native/check_phase0_subsurface.py --backend numpy --output /tmp/p0/numpy.json
    /home/okin/MAPLE/.venv/bin/python -I tests/native/check_phase0_subsurface.py --backend numba --output /tmp/p0/numba.json
    CUDA_VISIBLE_DEVICES=0 /home/okin/MAPLE/.venv-cupy/bin/python -I tests/native/check_phase0_subsurface.py --backend cuda --output /tmp/p0/cuda.json

What it does (and nothing more):

* builds a small GENUINE multilayer column with explicit geometry / hydraulic values and a known initial water state
  (3 layers, 4 cells of which one is inactive, cell area 6.25 m2) through the actual `maple_syrup.subsurface`
  (`soil_layers`, `prepare_column`, `column_state`, `step`, `initialize_profile`, `storage_m3`, `CumulativeBudget`);
* writes reproducible temporary scenario trees (`masy.scenario.v1` + `masy.interevent.v1` + `masy.weather.v1`) and
  drives them through the ACTUAL dated `masy.scenario.load_scenario` / `masy.coordinator.Coordinator`, whose weather
  rows and daily PET are validated and computed by the owning SYRUP library `maple_syrup.weather_pet` (reached through
  the actual `masy.pet` delegate; one scenario declares explicit PET plus mean temperature, the other computed
  `ecohyd_hargreaves` from daily extrema with the latitude bound from a MAPLE `case.yaml` location excerpt);
* a TEST-LOCAL staging adapter (`NativeIntereventTestAdapter`) maps every interevent dispatch to EXACT calendar-day
  chunks (rational seconds, split at midnight by the actual `masy.calendar`), checks each chunk's duration is exactly
  representable as a binary64 native duration before it is used, admits ZERO soil intake, and advances the column
  with `subsurface.step` (actual soil evaporation from the owning library's PET rate, gravity redistribution/drainage
  and diffusive redistribution). The run never writes the accepted state; the DRIVER publishes the staged result only
  after `Coordinator.step()` returned acceptance, and discards it on any refusal;
* compares the coordinator-driven path against an INDEPENDENTLY initialized direct-native control: its own column,
  its own state array, chunks derived with the standard library from the scenario documents, PET from
  `maple_syrup.weather_pet` directly, advanced through `subsurface.initialize_profile`. The control is a test
  reference, not a second simulation inventory;
* injects a wrong reported end and an invalid native forcing (non-zero potential evaporation on the inactive cell)
  after staged computation and proves the accepted column inputs and the coordinator cursor / ledger are unchanged;
* validates per-step native budgets under the existing MAPLE-derived bounds, positive evaporation / unmet
  evaporation / percolation / diffusion / drainage, an exactly zero surface return, aggregate closure with the native
  tolerance function, exact clocks, byte-unchanged inputs and exact counter / result comparisons;
* writes ONE JSON receipt with scalar summaries only (no time x cell histories) plus the exact module paths and
  SHA-256 of every source actually used.

Source arrangement (development overlay, NOT a locked production runtime and NOT a bypass of any guard): the driver
builds an isolated same-package import root under its work directory, `overlay/maple_syrup/`, holding byte-identical
copies (verified by SHA-256) of the CURRENT SYRUP `subsurface.py`, `subsurface_numba.py`, `subsurface_cuda.py`,
`conservation.py`, `__init__.py`, the import-only closure `routing_cuda.py` / `routing.py` / `routing_newton.py` (the
CUDA soil backend takes its compile options from `routing_cuda`; no routing code runs) and the Phase 0B candidate
`weather_pet.py`, so that
`maple_syrup.subsurface` and `maple_syrup.weather_pet` import from one package without sys.modules replacement or
fake modules. The low-level MAPLE backend (`maple.core.backend`) is the IMMUTABLE MAPLE39 source bundle listed in the
task baseline, never the live MAPLE checkout. After import every module's `__file__` is verified to lie inside its
declared root; otherwise the driver refuses (exit 3). No source lock is created, mutated or rebound.

Backend: `--backend numpy|numba|cuda` selects the actual `maple_syrup.subsurface` backend; an unavailable backend
(no Numba, no CuPy, no visible CUDA device) is a clean refusal receipt (exit 2), never a fallback.

Explicitly NOT accepted / NOT claimed by a passing run: persistent rain coupling (the native rain handoff still
resets soil water), plant growth / transpiration (no LEAF adapter), soil remapping after erosion / deposition,
combined native restart (the coordinator cursor is not a native checkpoint; there is no native commit hook on the
Coordinator and no combined transaction), wind / rain integration, and a production coupled adapter. Mixed-event
gate proofs remain the MECHANICAL protocol tests of the scheduler candidate. Nothing downloaded is executed.

Exit codes: 0 all checks passed; 1 one or more checks failed (receipt written); 2 selected backend unavailable
(refusal receipt written); 3 setup / source-isolation refusal (receipt written when possible).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # never write __pycache__ into the read-only dependency trees or a Root-prepared overlay

import argparse
import datetime as _dt
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import time
import traceback
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
CANDIDATE_ROOT = HERE.parents[2]
TASK_ROOT = HERE.parents[3]
BASELINE_JSON = TASK_ROOT / "baseline.json"
RECEIPT_SCHEMA = "syrup.phase0_native_interevent_receipt.v1"
ADAPTER_ID = "phase0_native_interevent_test_adapter"
# routing_cuda / routing / routing_newton are copied ONLY because `subsurface_cuda` imports its NVRTC compile options from
# `maple_syrup.routing_cuda`, whose module-level imports reach `routing` and `routing_newton`; no routing code is executed.
OVERLAY_FROM_SYRUP = ("__init__.py", "conservation.py", "subsurface.py", "subsurface_numba.py", "subsurface_cuda.py",
                      "routing_cuda.py", "routing.py", "routing_newton.py")
OVERLAY_FROM_PET = ("weather_pet.py",)
MASY_MODULES = ("masy", "masy.calendar", "masy.clock", "masy.coordinator", "masy.event_files", "masy.pet", "masy.scenario",
                "masy.scenario_io", "masy.weather", "masy.case_schedule", "masy.dense_schedule")
DAY_S = Fraction(86400)

# --- explicit column fixture (dyadic values so every hand-checkable depth is exact in FP64) ----------------------------
DEPTH_EDGES_M = (0.0, 0.125, 0.375, 1.0)  # three layers: 0.125 m, 0.25 m, 0.625 m
THETA_RESIDUAL, THETA_FIELD_CAPACITY, THETA_SATURATION, THETA_STRESS = 0.0625, 0.25, 0.5, 0.25
DIFFUSIVITY_M2_S = 2.0 ** -22  # about 2.4e-7 m2/s, explicit value (no texture rule)
EVAPORATION_FACTOR = 1.2
CELL_AREA_M2 = 6.25  # explicit non-unit area (2.5 m square cells)
ACTIVE_MASK = (True, True, False, True)  # cell 2 inactive
INACTIVE_CELL = 2
# theta per cell and layer: cell 0 above field capacity everywhere (cascade + bottom drainage in the first step), cell 1
# in the quadratic evaporation regime with an upward moisture gradient (diffusion), cell 2 inactive (identity), cell 3
# barely above residual in layer 0 so demand exceeds the withdrawable water (unmet evaporation).
THETA_BY_CELL = (
    (0.375, 0.3125, 0.3125),
    (0.125, 0.25, 0.125),
    (0.1, 0.1, 0.1),
    (0.0625 + 2.0 ** -10, 0.0625, 0.0625),
)

# --- scenario fixtures ---------------------------------------------------------------------------------------------------
CALENDAR_START = "2028-02-28T18:00:00+00:00"  # 2028 is a leap year: the first occurrence crosses midnight into 29 Feb
NATIVE_EPOCH_S = 86400.0  # non-zero native epoch: native clock = epoch + exact global elapsed
WEATHER_DATES = ("2028-02-28", "2028-02-29", "2028-03-01", "2028-03-02")
EXPLICIT_ROWS = tuple({"date": d, "tmean_c": t, "pet_mm_day": p}
                      for d, t, p in zip(WEATHER_DATES, (14.0, 15.0, 12.0, 16.0), (4.0, 4.5, 3.5, 5.0)))
HARGREAVES_ROWS = tuple({"date": d, "tmin_c": lo, "tmean_c": mid, "tmax_c": hi}
                        for d, lo, mid, hi in zip(WEATHER_DATES, (5.0, 6.0, 4.0, 7.0), (14.0, 15.0, 12.0, 16.0),
                                                  (23.0, 24.0, 20.0, 25.0)))
LATITUDE_DEG = 32.6
CASE_LOCATION_EXCERPT = ("# EXCERPT of a MAPLE case.yaml `location` section for the Phase 0 qualification driver; not a "
                         "complete MAPLE case.\nname: phase0_native_interevent_excerpt\nlocation:\n  latitude_deg: 32.6\n")
PROVENANCE = {"kind": "prescribed", "description": "Invented simplified forcing for the Phase 0 native interevent "
              "qualification driver; not observations, not a calibration.",
              "source": "check_phase0_subsurface.py (authored fixture)"}
UNITS = {"temperature": "degC", "relative_humidity": "fraction", "vpd": "kPa", "shortwave_radiation": "MJ_m2_day",
         "pet": "mm_day"}
# occurrence plan shared by both physical scenarios: 12 h (crosses leap midnight, partial day), 36 h, then the SAME
# 12 h event file again (reuse; consecutive interevents with no soil reset)
OCCURRENCES = (("interevent1", "events/interevent_12h.json"), ("interevent2", "events/interevent_36h.json"),
               ("interevent3", "events/interevent_12h.json"))
EVENT_FILES = {"events/interevent_12h.json": ("interevent_12h", 43200.0, True, False, False),
               "events/interevent_36h.json": ("interevent_36h", 129600.0, True, False, False),
               "events/interevent_transpiration.json": ("interevent_transpiration_1h", 3600.0, True, True, False)}


# --- small utilities -----------------------------------------------------------------------------------------------------
class DriverSetupError(RuntimeError):
    """Source isolation / environment refusal: nothing was computed."""


class CheckFailure(AssertionError):
    """A qualification check failed; the receipt records it."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


class Checks:
    """Ordered pass/fail records; `require` stops dependent work, `record` continues."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def record(self, name: str, ok: bool, detail: Any = None) -> bool:
        self.items.append({"check": name, "passed": bool(ok), "detail": detail})
        return bool(ok)

    def require(self, name: str, ok: bool, detail: Any = None) -> None:
        if not self.record(name, ok, detail):
            raise CheckFailure(f"{name}: {detail}")

    @property
    def failed(self) -> list[str]:
        return [i["check"] for i in self.items if not i["passed"]]


# --- source binding ------------------------------------------------------------------------------------------------------
def default_roots() -> dict[str, str]:
    """Dependency roots recorded by the task baseline (read-only dependencies), overridable on the command line."""
    if not BASELINE_JSON.is_file():
        return {}
    deps = json.loads(BASELINE_JSON.read_text()).get("dependencies_read_only", {})
    return {"masy_src": str(Path(deps["scheduler"]) / "src") if "scheduler" in deps else None,
            "pet_src": str(Path(deps["weather_pet"]) / "src") if "weather_pet" in deps else None,
            "syrup_src": str(Path(deps["existing_soil"]).parent) if "existing_soil" in deps else None,
            "maple_src": str(Path(deps["maple_immutable"]) / "src") if "maple_immutable" in deps else None}


def build_overlay(work_dir: Path, syrup_src: Path, pet_src: Path) -> dict[str, Any]:
    """Isolated same-package import root with byte-identical copies of the actual modules (hash-verified)."""
    overlay = work_dir / "overlay"
    package = overlay / "maple_syrup"
    package.mkdir(parents=True, exist_ok=True)
    files = []
    for origin_root, names in ((syrup_src / "maple_syrup", OVERLAY_FROM_SYRUP), (pet_src / "maple_syrup", OVERLAY_FROM_PET)):
        for name in names:
            origin = origin_root / name
            if not origin.is_file():
                raise DriverSetupError(f"overlay origin missing: {origin}")
            target = package / name
            shutil.copyfile(origin, target)
            origin_sha, copy_sha = sha256_file(origin), sha256_file(target)
            if origin_sha != copy_sha:
                raise DriverSetupError(f"overlay copy of {origin} is not byte-identical")
            files.append({"name": name, "origin": str(origin), "sha256": origin_sha, "copy": str(target)})
    (overlay / "OVERLAY_NOT_A_RUNTIME.txt").write_text(
        "Development import overlay built by check_phase0_subsurface.py: byte-identical copies of the current SYRUP "
        "soil modules and the Phase 0B weather_pet candidate in ONE package for this qualification run. It is not a "
        "source-locked production runtime, not a dependency bundle and bypasses no guard. Rebuilt on every run.\n")
    return {"root": str(overlay), "package": "maple_syrup", "files": files,
            "meaning": "development source overlay (hash-verified copies); NOT a locked production runtime"}


def prepared_overlay(overlay_src: Path, identity_path: Path | None) -> dict[str, Any]:
    """Use a Root-prepared same-package import root as-is (never edited, never rebuilt). Every file of its
    `maple_syrup` package is hashed; when an identity manifest is given, each hashed file must match the manifest
    entry (`source_files[<relative path>].sha256`) or the driver refuses."""
    package = overlay_src / "maple_syrup"
    if not (package / "__init__.py").is_file():
        raise DriverSetupError(f"prepared overlay {overlay_src} has no maple_syrup/__init__.py")
    files = [{"name": p.name, "origin": str(p), "sha256": sha256_file(p), "copy": None} for p in sorted(package.glob("*.py"))]
    record: dict[str, Any] = {"root": str(overlay_src), "package": "maple_syrup", "files": files, "prepared_by_root": True,
                              "identity_manifest": None,
                              "meaning": "Root-prepared development overlay used as-is (hashed, not edited); NOT a locked production runtime"}
    if identity_path is not None:
        manifest = json.loads(identity_path.read_text())
        entries = manifest.get("source_files", {})
        by_name = {Path(rel).name: entry for rel, entry in entries.items()}
        mismatched = [f["name"] for f in files if f["name"] not in by_name or by_name[f["name"]].get("sha256") != f["sha256"]]
        if mismatched:
            raise DriverSetupError(f"prepared overlay files differ from the identity manifest {identity_path}: {mismatched}")
        for f in files:
            f["manifest_source"] = by_name[f["name"]].get("source")
        record["identity_manifest"] = {"path": str(identity_path), "sha256": sha256_file(identity_path),
                                       "status": manifest.get("status"), "created_utc": manifest.get("created_utc"),
                                       "verified_files": len(files)}
    return record


def bind_sources(maple_src: Path, masy_src: Path, overlay_root: Path, backend: str) -> tuple[SimpleNamespace, dict[str, Any]]:
    """Put the three declared roots first on sys.path, import the actual modules and verify where each resolved."""
    for name in ("maple", "maple_syrup", "masy"):
        if name in sys.modules or any(m == name or m.startswith(name + ".") for m in sys.modules):
            raise DriverSetupError(f"{name!r} is already imported in this interpreter; run the driver in a fresh process")
    for root in (maple_src, masy_src, overlay_root):
        if not root.is_dir():
            raise DriverSetupError(f"declared source root is not a directory: {root}")
    for root in (overlay_root, masy_src, maple_src):  # inserted in reverse so maple_src ends first
        sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    try:
        maple = importlib.import_module("maple")
        backend_mod = importlib.import_module("maple.core.backend")
        masy_mods = {name: importlib.import_module(name) for name in MASY_MODULES}
        subsurface = importlib.import_module("maple_syrup.subsurface")
        conservation = importlib.import_module("maple_syrup.conservation")
        weather_pet = importlib.import_module("maple_syrup.weather_pet")
        extra = {}
        if backend == "numba":
            extra["maple_syrup.subsurface_numba"] = importlib.import_module("maple_syrup.subsurface_numba")
        if backend == "cuda":
            extra["maple_syrup.subsurface_cuda"] = importlib.import_module("maple_syrup.subsurface_cuda")
    except ImportError as exc:
        raise DriverSetupError(f"import failed under the declared roots: {type(exc).__name__}: {exc}") from exc
    origins: dict[str, dict[str, str]] = {}

    def verify(module: Any, root: Path) -> None:
        file = Path(getattr(module, "__file__", "") or "").resolve()
        if not file.is_file() or not file.is_relative_to(root.resolve()):
            raise DriverSetupError(f"{module.__name__} resolved to {file}, outside the declared root {root}")
        origins[module.__name__] = {"file": str(file), "sha256": sha256_file(file)}

    verify(maple, maple_src)
    verify(backend_mod, maple_src)
    for module in masy_mods.values():
        verify(module, masy_src)
    for module in (subsurface, conservation, weather_pet, *extra.values()):
        verify(module, overlay_root)
    for name in ("freeze", "to_device", "to_host", "to_float", "to_int", "errstate", "cupy_module", "gpu_execution_available"):
        if not hasattr(backend_mod, name):
            raise DriverSetupError(f"maple.core.backend from {maple_src} lacks {name!r}")
    modules = SimpleNamespace(maple=maple, backend=backend_mod, subsurface=subsurface, conservation=conservation,
                              weather_pet=weather_pet, extra=extra, **{name.replace(".", "_"): mod for name, mod in masy_mods.items()})
    record = {"sys_path_prepended": [str(maple_src), str(masy_src), str(overlay_root)],
              "module_origins": origins,
              "maple_bundle_dir": str(maple_src.parent.parent) if maple_src.name == "src" else str(maple_src),
              "weather_pet_source_identity": dict(weather_pet.SOURCE_IDENTITY),
              "masy_version": getattr(masy_mods["masy"], "__version__", None),
              "meaning": "development identities of what was actually imported; not a source-locked qualified runtime"}
    return modules, record


# --- scenario trees ------------------------------------------------------------------------------------------------------
def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=1, sort_keys=False, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return sha256_bytes(data)


def weather_document(method: str, rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    explicit = method == "explicit"
    return {"schema": "masy.weather.v1", "weather_id": f"phase0_native_{method}", "utc_offset": "+00:00",
            "sampling": "daily", "units": dict(UNITS),
            "temperature": {"tmean": "supplied", "extrema": not explicit},
            "humidity": {"supplied": "none"}, "radiation": {"supplied": False}, "pet": {"method": method},
            "provenance": dict(PROVENANCE), "rows": [dict(r) for r in rows]}


def write_scenario_tree(root: Path, scenario_id: str, method: str, rows: tuple[dict[str, Any], ...],
                        occurrences: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for _, rel in occurrences:
        if rel not in hashes:
            event_id, duration, evaporation, transpiration, growth = EVENT_FILES[rel]
            hashes[rel] = write_json(root / rel, {"schema": "masy.interevent.v1", "event_id": event_id,
                                                   "duration_s": duration,
                                                   "processes": {"evaporation": evaporation, "transpiration": transpiration,
                                                                 "growth": growth}})
    weather_sha = write_json(root / "weather.json", weather_document(method, rows))
    scenario = {"schema": "masy.scenario.v1", "scenario_id": scenario_id,
                "calendar": {"start": CALENDAR_START, "convention": "proleptic_gregorian", "leap_seconds": "none"},
                "native_epoch_s": NATIVE_EPOCH_S, "gap_policy": "refuse", "requires_continuous_soil_moisture": False,
                "weather": {"file": "weather.json", "sha256": weather_sha},
                "events": [{"occurrence_id": oid, "file": rel, "sha256": hashes[rel], "scheduled_start": None}
                           for oid, rel in occurrences]}
    write_json(root / "scenario.json", scenario)
    (root / "case_location_excerpt.yaml").write_text(CASE_LOCATION_EXCERPT)
    return {"dir": str(root), "scenario": str(root / "scenario.json"), "method": method,
            "case_yaml": str(root / "case_location_excerpt.yaml"), "files_sha256": tree_hashes(root)}


# --- exact chunking ------------------------------------------------------------------------------------------------------
class NotRepresentableError(ValueError):
    """An exact rational chunk duration is not a binary64 value; it is refused, never rounded."""


def native_duration_s(seconds: Fraction) -> float:
    """The exact rational duration as a float, accepted only when exactly representable and positive."""
    if type(seconds) is not Fraction or seconds <= 0:
        raise NotRepresentableError(f"chunk duration must be a positive Fraction, got {seconds!r}")
    value = float(seconds)
    if not math.isfinite(value) or Fraction(value) != seconds:
        raise NotRepresentableError(f"chunk duration {fraction_text(seconds)} s is not exactly representable as binary64")
    return value


def exact_day_chunks(calendar: Any, start_g: Fraction, end_g: Fraction) -> list[tuple[_dt.date, Fraction]]:
    """[start_g, end_g) split at calendar midnights into exact (date, rational seconds) chunks via the actual
    `masy.calendar.CalendarStart.instant` (whole seconds in datetime, sub-second remainder exact)."""
    out: list[tuple[_dt.date, Fraction]] = []
    cursor = start_g
    while cursor < end_g:
        instant = calendar.instant(cursor)
        since_midnight = Fraction(instant.whole.hour * 3600 + instant.whole.minute * 60 + instant.whole.second) + instant.subsecond
        piece_end = min(end_g, cursor + (DAY_S - since_midnight))
        out.append((instant.date, piece_end - cursor))
        cursor = piece_end
    return out


def control_day_chunks(start_iso: str, offset_g: Fraction, duration: Fraction) -> list[tuple[_dt.date, Fraction]]:
    """Independent stdlib derivation of the same chunks from the scenario's calendar start and an event duration."""
    start = _dt.datetime.fromisoformat(start_iso)
    out: list[tuple[_dt.date, Fraction]] = []
    cursor, end = offset_g, offset_g + duration
    while cursor < end:
        whole = cursor.numerator // cursor.denominator
        instant = start + _dt.timedelta(seconds=whole)
        next_midnight = _dt.datetime.combine(instant.date() + _dt.timedelta(days=1), _dt.time.min, tzinfo=instant.tzinfo)
        to_midnight = Fraction(int((next_midnight - instant).total_seconds())) - (cursor - whole)
        piece_end = min(end, cursor + to_midnight)
        out.append((instant.date(), piece_end - cursor))
        cursor = piece_end
    return out


def chunk_records(chunks: list[tuple[_dt.date, Fraction]]) -> list[dict[str, str]]:
    return [{"date": d.isoformat(), "seconds": fraction_text(s)} for d, s in chunks]


# --- column construction -------------------------------------------------------------------------------------------------
def make_layers(m: SimpleNamespace) -> Any:
    return m.subsurface.soil_layers(DEPTH_EDGES_M, theta_residual=THETA_RESIDUAL, theta_field_capacity=THETA_FIELD_CAPACITY,
                                    theta_saturation=THETA_SATURATION, diffusivity_m2_s=DIFFUSIVITY_M2_S,
                                    theta_evaporation_stress=THETA_STRESS, evaporation_factor=EVAPORATION_FACTOR,
                                    gravity_drainage=True)


def initial_water_host(layers: Any) -> np.ndarray:
    theta = np.array(THETA_BY_CELL, dtype=np.float64).T  # (L, N)
    return np.ascontiguousarray(theta * np.array(layers.thickness_m)[:, None])


def device_namespace(m: SimpleNamespace, backend: str) -> tuple[Any, int | None]:
    """The array namespace of the selected backend; CUDA through the actual SYRUP probe (refusal, never fallback)."""
    if backend == "cuda":
        cuda = m.extra["maple_syrup.subsurface_cuda"]
        cp = cuda.require_cuda()
        return cp, cuda.current_device_id(cp)
    if backend == "numba":
        m.extra["maple_syrup.subsurface_numba"].require_numba()
    return np, None


def prepare(m: SimpleNamespace, backend: str, xp: Any, layers: Any, reserve_bytes: int) -> Any:
    mask = np.array(ACTIVE_MASK, dtype=np.bool_)
    mask_xp = mask if xp is np else xp.asarray(mask)
    # default available_bytes_fn: the real host (sysconf) or device (driver + pool) reading; nothing is faked
    return m.subsurface.prepare_column(layers, mask_xp, CELL_AREA_M2, backend=backend, reserve_bytes=int(reserve_bytes))


def make_state(m: SimpleNamespace, column: Any, water_host: np.ndarray, t_s: float) -> Any:
    water = water_host if column.xp is np else column.xp.asarray(water_host)
    return m.subsurface.column_state(column, water, t_s)


def water_hash(m: SimpleNamespace, state: Any) -> str:
    return sha256_bytes(np.ascontiguousarray(m.backend.to_host(state.water_m)).tobytes())


def budget_totals(c: Any) -> dict[str, float]:
    return {"n_steps": int(c.n_steps), "elapsed_s": float(c.elapsed_s), "added_m3": float(c.added_m3),
            "surface_return_m3": float(c.surface_return_m3), "drainage_m3": float(c.drainage_m3),
            "evaporation_m3": float(c.evaporation_m3), "evaporation_unmet_m3": float(c.evaporation_unmet_m3)}


# --- the test-local staging adapter ---------------------------------------------------------------------------------------
class NativeIntereventTestAdapter:
    """TEST-LOCAL adapter for `masy.coordinator.Coordinator` (kind "interevent").

    `run(dispatch)` reads the ONE accepted native column state, advances a staged copy through `subsurface.step` over
    exact day chunks with zero intake and the dispatch's daily PET (owning library rate), and reports the realized
    end. It NEVER writes the accepted state or any input array (the native step is pure). Publication is the
    DRIVER's job: `publish()` only after `Coordinator.step()` returned acceptance, `discard()` on refusal. There is
    no native commit hook on the Coordinator, no transaction and no checkpoint; this helper is not a production
    coupled adapter and must not be advertised as one. `injection` ("wrong_end" / "invalid_forcing") exists only for
    the refusal checks and is reset by the driver."""

    def __init__(self, m: SimpleNamespace, source_identity: str) -> None:
        self.m = m
        self.source_identity = source_identity
        self.column: Any = None
        self.accepted: Any = None
        self.calendar: Any = None
        self.staged: dict[str, Any] | None = None
        self.injection: str | None = None
        self.runs = self.published = self.discarded = self.refused = 0
        self.added: Any = None  # zero intake, allocated once, never written
        self.pet_rate: Any = None  # (N,) refilled per chunk
        self.inactive: Any = None

    def bind(self, column: Any, state: Any, calendar: Any) -> None:
        xp = column.xp
        self.column, self.accepted, self.calendar = column, state, calendar
        self.added = xp.zeros((column.n_layers, column.n_cells), dtype=np.float64)
        self.pet_rate = xp.zeros(column.n_cells, dtype=np.float64)
        self.inactive = ~column.active

    def capabilities(self) -> Any:
        C = self.m.masy_coordinator.AdapterCapabilities
        return C("interevent", ADAPTER_ID, self.source_identity, True, "any_positive_duration", "proleptic_gregorian_dates",
                 "not_provided", "staged_until_accepted",
                 "TEST-LOCAL staging helper: actual maple_syrup.subsurface column advanced over exact day chunks with zero "
                 "intake and the owning library's PET; interevent-only, so continuous_soil_moisture=True states only that "
                 "consecutive interevent occurrences share one persistent column state (no rain/wind adapter exists; the "
                 "native rain handoff still resets soil). Transpiration/growth refuse (no LEAF). Not a production adapter; "
                 "publication is performed by the driver after acceptance; no native checkpoint.")

    def run(self, dispatch: dict[str, Any]) -> Any:
        m = self.m
        Outcome = m.masy_coordinator.AdapterOutcome
        self.runs += 1
        self.staged = None
        gate = dispatch["processes"]
        if gate["transpiration"] or gate["growth"]:
            self.refused += 1
            return Outcome("refused_unsupported_process", None, evidence={
                "reason": "this test adapter has no LEAF; transpiration/growth are not performed and not faked", "gate": dict(gate)})
        native_start = m.masy_clock.decode_time(dispatch["native_start"])
        if self.accepted.t_s.hex() != native_start.hex():
            self.refused += 1
            return Outcome("refused_native_start_mismatch", None, evidence={
                "accepted_t_s": self.accepted.t_s.hex(), "dispatch_native_start": native_start.hex()})
        start_g = m.masy_calendar.parse_fraction(dispatch["calendar"]["start"]["elapsed_s"], "calendar.start.elapsed_s")
        end_g = m.masy_calendar.parse_fraction(dispatch["calendar"]["planned_end"]["elapsed_s"], "calendar.planned_end.elapsed_s")
        pet_by_date: dict[str, float] = {}
        if gate["evaporation"]:
            pet_by_date = {row["date"]: float(row["pet_mm_day"]) for row in dispatch["weather"]["daily_pet"]}
        chunks = exact_day_chunks(self.calendar, start_g, end_g)
        xp, column = self.column.xp, self.column
        sub = m.subsurface
        state = self.accepted
        cumulative = sub.CumulativeBudget()
        acc = {name: xp.zeros((), dtype=np.float64) for name in ("percolation_m", "abs_diffusion_m", "inactive_abs_flux_m",
                                                                  "max_step_residual_over_bound")}
        elapsed = Fraction(0)
        input_hash = water_hash(m, self.accepted)
        for index, (date, seconds) in enumerate(chunks):
            dt_s = native_duration_s(seconds)  # refuses non-representable exact durations; never rounds
            rate = m.weather_pet.pet_m_s(pet_by_date[date.isoformat()]) if gate["evaporation"] else 0.0
            self.pet_rate[...] = 0.0
            self.pet_rate[column.active] = rate
            if self.injection == "invalid_forcing" and index == len(chunks) - 1:
                self.pet_rate[self.inactive] = rate if rate > 0.0 else 1e-9  # forcing on an inactive cell: native refusal
            result = sub.step(column, state, dt_s, added_water_m=self.added, potential_evaporation_m_s=self.pet_rate)
            b = result.budget
            cumulative = cumulative.accumulate(b, result.dt_s)
            acc["percolation_m"] += xp.sum(result.percolation_m)
            acc["abs_diffusion_m"] += xp.sum(xp.abs(result.diffusion_m))
            acc["inactive_abs_flux_m"] += (xp.sum(xp.abs(result.percolation_m[:, self.inactive]))
                                           + xp.sum(xp.abs(result.diffusion_m[:, self.inactive]))
                                           + xp.sum(xp.abs(result.surface_return_m[self.inactive]))
                                           + xp.sum(xp.abs(result.drainage_m[self.inactive]))
                                           + xp.sum(xp.abs(result.evaporation_m[self.inactive]))
                                           + xp.sum(xp.abs(result.evaporation_unmet_m[self.inactive])))
            ratio = abs(b.residual_m3) / b.bound_m3 if b.bound_m3 > 0 else (0.0 if b.residual_m3 == 0 else math.inf)
            acc["max_step_residual_over_bound"] = xp.maximum(acc["max_step_residual_over_bound"], xp.asarray(ratio))
            elapsed += seconds
            state = result.state
            del result, b
            if Fraction(state.t_s) != Fraction(native_start) + elapsed:
                self.refused += 1
                return Outcome("refused_time_not_exact", None, evidence={"t_s": state.t_s.hex(), "expected": fraction_text(Fraction(native_start) + elapsed)})
        if water_hash(m, self.accepted) != input_hash:
            self.refused += 1
            return Outcome("refused_input_mutated", None, evidence={"reason": "the accepted state array changed during run"})
        to_float = m.backend.to_float
        summary = {"n_steps": int(cumulative.n_steps), "chunks": chunk_records(chunks),
                   "dates": sorted({d.isoformat() for d, _ in chunks}), "pet_mm_day_by_date": pet_by_date,
                   "budget": budget_totals(cumulative),
                   "percolation_m": to_float(acc["percolation_m"]), "abs_diffusion_m": to_float(acc["abs_diffusion_m"]),
                   "inactive_abs_flux_m": to_float(acc["inactive_abs_flux_m"]),
                   "max_step_residual_over_bound": to_float(acc["max_step_residual_over_bound"]),
                   "elapsed_s_exact": fraction_text(elapsed), "t_end_hex": state.t_s.hex(), "injection": self.injection}
        self.staged = {"state": state, "cumulative": cumulative, "summary": summary, "native_start": native_start}
        end = state.t_s
        if self.injection == "wrong_end":
            end = end + 1.0
        returned = float(elapsed) if Fraction(float(elapsed)) == elapsed else None
        return Outcome("complete", float(end), None, None, returned_elapsed_s=returned, evidence=dict(summary))

    def publish(self) -> dict[str, Any]:
        """DRIVER-called after Coordinator acceptance: the staged state becomes the ONE accepted native state."""
        if self.staged is None:
            raise RuntimeError("nothing staged")
        staged, self.staged = self.staged, None
        self.accepted = staged["state"]
        self.published += 1
        return staged

    def discard(self) -> None:
        self.staged = None
        self.discarded += 1


# --- independent direct-native control -------------------------------------------------------------------------------------
def pet_records(m: SimpleNamespace, method: str, rows: list[dict[str, Any]], latitude: float | None) -> dict[_dt.date, Any]:
    """PET per date straight from the owning library (no MASY): explicit or ecohyd_hargreaves."""
    wp = m.weather_pet
    out = {}
    for row in rows:
        date = _dt.date.fromisoformat(row["date"])
        if method == "explicit":
            weather = wp.DailyWeather.explicit_pet(date, row["tmean_c"], row["pet_mm_day"], provenance=PROVENANCE["kind"])
        else:
            weather = wp.DailyWeather(date, row["tmin_c"], row["tmean_c"], row["tmax_c"], provenance=PROVENANCE["kind"])
        out[date] = wp.potential_demand(weather, latitude, method)
    return out


def direct_control(m: SimpleNamespace, backend: str, xp: Any, tree: dict[str, Any], layers: Any, water_host: np.ndarray,
                   reserve_bytes: int) -> dict[str, Any]:
    """Independently initialized direct-native reference: own column, own state, stdlib chunking from the documents,
    PET from maple_syrup.weather_pet, `initialize_profile` per occurrence. Scalar summaries only."""
    root = Path(tree["dir"])
    scenario = json.loads((root / "scenario.json").read_text())
    weather = json.loads((root / scenario["weather"]["file"]).read_text())
    method = weather["pet"]["method"]
    latitude = LATITUDE_DEG if method == "ecohyd_hargreaves" else None
    records = pet_records(m, method, weather["rows"], latitude)
    column = prepare(m, backend, xp, layers, reserve_bytes)
    state = make_state(m, column, water_host, float(scenario["native_epoch_s"]))
    sub, wp = m.subsurface, m.weather_pet
    added = xp.zeros((column.n_layers, column.n_cells), dtype=np.float64)
    pet_rate = xp.zeros(column.n_cells, dtype=np.float64)
    start_iso = scenario["calendar"]["start"]
    offset = Fraction(0)
    occurrences = []
    storage0 = sub.storage_m3(column, state)
    for event in scenario["events"]:
        document = json.loads((root / event["file"]).read_text())
        duration = Fraction(document["duration_s"])
        chunks = control_day_chunks(start_iso, offset, duration)
        depth_sum_m = 0.0

        def forcing(chunks=chunks, evaporate=document["processes"]["evaporation"]):
            nonlocal depth_sum_m
            for date, seconds in chunks:
                dt_s = native_duration_s(seconds)
                rate = wp.pet_m_s(records[date].pet_mm_day) if evaporate else 0.0
                depth_sum_m += rate * dt_s
                pet_rate[...] = 0.0
                pet_rate[column.active] = rate
                yield sub.ForcingRecord(dt_s, added, pet_rate)

        profile = sub.initialize_profile(column, state, forcing())
        state = profile.state
        integrated = wp.integrate_interevent_pet_chunks(chunks, records, "interevent")
        offset += duration
        occurrences.append({"occurrence_id": event["occurrence_id"], "chunks": chunk_records(chunks),
                            "budget": budget_totals(profile.cumulative), "residual_m3": float(profile.residual_m3),
                            "bound_m3": float(profile.bound_m3), "n_terms": int(profile.n_terms),
                            "storage_initial_m3": float(profile.storage_initial_m3), "storage_final_m3": float(profile.storage_final_m3),
                            "t_end_hex": state.t_s.hex(), "water_sha256": water_hash(m, state),
                            "pet_depth_sum_m": depth_sum_m, "library_integrated_depth_mm": float(integrated.depth_mm),
                            "library_chunk_count": int(integrated.chunk_count),
                            "library_total_seconds": float(integrated.total_seconds)})
    return {"backend": column.backend, "namespace": column.xp.__name__, "storage_initial_m3": storage0,
            "storage_final_m3": sub.storage_m3(column, state), "occurrences": occurrences, "method": method,
            "pet_mm_day_by_date": {d.isoformat(): r.pet_mm_day for d, r in sorted(records.items())},
            "meaning": "independent test reference (own column/state/chunking/PET, initialize_profile); not a second inventory"}


# --- coordinator-driven run -----------------------------------------------------------------------------------------------
def run_scenario(m: SimpleNamespace, backend: str, xp: Any, tree: dict[str, Any], layers: Any, water_host: np.ndarray,
                 source_identity: str, checks: Checks, *, inject_at: int | None) -> dict[str, Any]:
    scenario = m.masy_scenario.load_scenario(Path(tree["scenario"]), m.masy_scenario_io.IOLimits())
    label = scenario.scenario_id
    checks.record(f"{label}.scenario_loaded", scenario.kinds_present == frozenset({"interevent"}) and len(scenario.occurrences) == 3,
                  {"digest": scenario.digest, "occurrences": [o.occurrence_id for o in scenario.occurrences]})
    latitude = None
    if scenario.weather.pet.needs_latitude:
        latitude = m.masy_weather.latitude_binding_from_case_yaml(Path(tree["case_yaml"]), m.masy_scenario_io.IOLimits())
        checks.record(f"{label}.latitude_bound_from_case_yaml_excerpt", latitude.latitude_deg == LATITUDE_DEG,
                      {"latitude_deg": latitude.latitude_deg, "source": latitude.source})
    adapter = NativeIntereventTestAdapter(m, source_identity)
    coordinator = m.masy_coordinator.Coordinator(scenario, {"interevent": adapter}, latitude=latitude)
    canonical = m.masy_scenario_io.canonical_json
    L, N = len(DEPTH_EDGES_M) - 1, len(ACTIVE_MASK)
    # explicit reserve for the live coordinator / PET / input objects held while the column exists, declared BEFORE
    # prepare_column (a declaration added to the admitted requirement, not a measurement of Python object overhead)
    live_bytes = (len(canonical(scenario.record()).encode()) + len(canonical(scenario.weather.record()).encode())
                  + len(canonical(coordinator.dispatch()).encode()) + len(canonical(coordinator.as_record()).encode()))
    array_bytes = 8 * (2 * L * N + 2 * (L * N + N) + L * N)  # staged + control states, two input sets, control copy
    reserve = int(live_bytes + array_bytes)
    t0 = time.perf_counter()
    column = prepare(m, backend, xp, layers, reserve)
    prepare_s = time.perf_counter() - t0
    checks.require(f"{label}.backend_selected", column.backend == backend,
                   {"backend": column.backend, "namespace": column.xp.__name__, "device_id": column.device_id,
                    "reserve_bytes": reserve, "required_bytes": column.memory.required_bytes, "prepare_s": prepare_s})
    state0 = make_state(m, column, water_host, scenario.native_epoch_s)
    initial_hash = water_hash(m, state0)
    adapter.bind(column, state0, scenario.calendar)
    sub = m.subsurface
    storage_initial = sub.storage_m3(column, state0)
    realized_records, occurrence_summaries, refusals = [], [], []
    totals = {"added_m3": 0.0, "surface_return_m3": 0.0, "drainage_m3": 0.0, "evaporation_m3": 0.0,
              "evaporation_unmet_m3": 0.0, "n_steps": 0, "percolation_m": 0.0, "abs_diffusion_m": 0.0,
              "inactive_abs_flux_m": 0.0, "max_step_residual_over_bound": 0.0}
    while not coordinator.finished:
        index = coordinator.next_index
        if inject_at is not None and index == inject_at:
            for injection, expected_code in (("wrong_end", "realized_end_mismatch"), ("invalid_forcing", "adapter_failure")):
                before_cursor = canonical(coordinator.as_record())
                before_ledger = canonical(coordinator.ledger.as_record())
                before_hash = water_hash(m, adapter.accepted)
                before_t = adapter.accepted.t_s.hex()
                adapter.injection = injection
                code = None
                try:
                    coordinator.step()
                except m.masy_coordinator.CoordinatorError as exc:
                    code = exc.code
                finally:
                    adapter.injection = None
                staged_present = adapter.staged is not None
                adapter.discard()
                unchanged = (canonical(coordinator.as_record()) == before_cursor and canonical(coordinator.ledger.as_record()) == before_ledger
                             and water_hash(m, adapter.accepted) == before_hash and adapter.accepted.t_s.hex() == before_t
                             and coordinator.next_index == index)
                detail = {"injection": injection, "refusal_code": code, "expected_code": expected_code,
                          "last_refusal": coordinator.last_refusal, "staged_before_discard": staged_present,
                          "stage_discarded": adapter.staged is None, "accepted_state_and_cursor_unchanged": unchanged}
                refusals.append(detail)
                checks.record(f"{label}.refusal_{injection}_leaves_accepted_state_and_cursor_unchanged",
                              code == expected_code and unchanged and adapter.staged is None, detail)
                if injection == "invalid_forcing":
                    refusal_detail = str((coordinator.last_refusal or {}).get("detail", ""))
                    checks.record(f"{label}.invalid_forcing_refused_natively_before_staging",
                                  not staged_present and code == "adapter_failure" and "SubsurfaceStepError" in refusal_detail
                                  and "inactive" in refusal_detail, {"detail": coordinator.last_refusal})
        storage_before = sub.storage_m3(column, adapter.accepted)
        hash_before = water_hash(m, adapter.accepted)
        dispatch = coordinator.dispatch()
        try:
            realized = coordinator.step()
        except m.masy_coordinator.CoordinatorError as exc:
            adapter.discard()
            checks.require(f"{label}.{dispatch['occurrence_id']}.accepted", False, {"code": exc.code, "detail": exc.detail})
        staged = adapter.publish()  # ONLY after acceptance
        summary = staged["summary"]
        realized_records.append(realized)
        end_native = m.masy_clock.decode_time(realized["native_end"])
        ledger_entry = coordinator.ledger.records()[realized["ledger_entry_index"]]
        occurrence = {"occurrence_id": realized["occurrence_id"], "calendar_start": realized["calendar_start"]["iso"],
                      "calendar_end": realized["calendar_end"]["iso"], "weather_dates": realized["weather_dates"],
                      "whole_days": dispatch["calendar"]["whole_days"], "native_start_hex": m.masy_clock.decode_time(realized["native_start"]).hex(),
                      "native_end_hex": end_native.hex(), "realized_minus_planned_end_s": realized["realized_minus_planned_end_s"],
                      "ledger_endpoint_discrepancy": ledger_entry["endpoint_discrepancy"],
                      "storage_before_m3": storage_before, "storage_after_m3": sub.storage_m3(column, adapter.accepted),
                      "water_sha256_before": hash_before, "water_sha256_after": water_hash(m, adapter.accepted),
                      **summary}
        previous_after_hash = occurrence_summaries[-1]["water_sha256_after"] if occurrence_summaries else initial_hash
        previous_after_storage = occurrence_summaries[-1]["storage_after_m3"] if occurrence_summaries else storage_initial
        occurrence_summaries.append(occurrence)
        for key in ("added_m3", "surface_return_m3", "drainage_m3", "evaporation_m3", "evaporation_unmet_m3"):
            totals[key] += summary["budget"][key]
        totals["n_steps"] += summary["n_steps"]
        for key in ("percolation_m", "abs_diffusion_m", "inactive_abs_flux_m"):
            totals[key] += summary[key]
        totals["max_step_residual_over_bound"] = max(totals["max_step_residual_over_bound"], summary["max_step_residual_over_bound"])
        oid = realized["occurrence_id"]
        checks.record(f"{label}.{oid}.exact_times", adapter.accepted.t_s.hex() == end_native.hex()
                      and realized["realized_minus_planned_end_s"] == "0/1" and ledger_entry["endpoint_discrepancy"] == "0/1"
                      and summary["elapsed_s_exact"] == ledger_entry["endpoint_elapsed"],
                      {"t_s": adapter.accepted.t_s.hex(), "native_end": end_native.hex(), "endpoint_elapsed": ledger_entry["endpoint_elapsed"]})
        checks.record(f"{label}.{oid}.no_soil_reset_between_occurrences",
                      hash_before == previous_after_hash and storage_before == previous_after_storage
                      and occurrence["water_sha256_after"] != hash_before,
                      {"storage_before_m3": storage_before, "previous_after_m3": previous_after_storage,
                       "state_carried_bitwise": hash_before == previous_after_hash, "state_advanced": occurrence["water_sha256_after"] != hash_before})
        checks.record(f"{label}.{oid}.per_step_residual_within_native_bound", summary["max_step_residual_over_bound"] <= 1.0,
                      {"max_step_residual_over_bound": summary["max_step_residual_over_bound"]})
        checks.record(f"{label}.{oid}.zero_surface_return", summary["budget"]["surface_return_m3"] == 0.0 and summary["budget"]["added_m3"] == 0.0,
                      {"surface_return_m3": summary["budget"]["surface_return_m3"], "added_m3": summary["budget"]["added_m3"]})
        checks.record(f"{label}.{oid}.inactive_cell_has_no_flux", summary["inactive_abs_flux_m"] == 0.0,
                      {"inactive_abs_flux_m": summary["inactive_abs_flux_m"]})
    final_hash = water_hash(m, adapter.accepted)
    storage_final = sub.storage_m3(column, adapter.accepted)
    final_host = np.ascontiguousarray(m.backend.to_host(adapter.accepted.water_m))
    checks.record(f"{label}.coordinator_finished_all_occurrences", coordinator.finished and len(realized_records) == 3
                  and adapter.published == 3 and [r["status"] for r in realized_records] == ["complete"] * 3,
                  {"published": adapter.published, "discarded": adapter.discarded, "runs": adapter.runs, "refused": adapter.refused})
    checks.record(f"{label}.leap_midnight_partial_day_crossed", realized_records[0]["weather_dates"] == ["2028-02-28", "2028-02-29"]
                  and occurrence_summaries[0]["whole_days"] is None and len(occurrence_summaries[0]["chunks"]) == 2
                  and realized_records[0]["calendar_end"]["iso"] == "2028-02-29T06:00:00+00:00",
                  {"weather_dates": realized_records[0]["weather_dates"], "chunks": occurrence_summaries[0]["chunks"]})
    checks.record(f"{label}.positive_evaporation_unmet_percolation_diffusion_drainage",
                  totals["evaporation_m3"] > 0.0 and totals["evaporation_unmet_m3"] > 0.0 and totals["percolation_m"] > 0.0
                  and totals["abs_diffusion_m"] > 0.0 and totals["drainage_m3"] > 0.0, dict(totals))
    checks.record(f"{label}.inactive_cell_state_identity", bool(np.array_equal(final_host[:, INACTIVE_CELL], water_host[:, INACTIVE_CELL])),
                  {"cell": INACTIVE_CELL})
    checks.record(f"{label}.initial_state_array_unchanged", water_hash(m, state0) == initial_hash
                  and sha256_bytes(water_host.tobytes()) == sha256_bytes(initial_water_host(layers).tobytes()),
                  {"initial_sha256": initial_hash})
    closure_terms = 4 * L * N * max(totals["n_steps"], 1) + 7
    residual = (storage_final - storage_initial) - (((totals["added_m3"] - totals["surface_return_m3"]) - totals["drainage_m3"]) - totals["evaporation_m3"])
    scale = max(abs(storage_initial), abs(storage_final), totals["added_m3"], totals["surface_return_m3"], totals["drainage_m3"], totals["evaporation_m3"])
    bound = m.conservation.volume_roundoff_bound_m3(closure_terms, scale)
    checks.record(f"{label}.aggregate_closure_within_native_bound", math.isfinite(residual) and abs(residual) <= bound,
                  {"residual_m3": residual, "bound_m3": bound, "n_terms": closure_terms, "storage_initial_m3": storage_initial,
                   "storage_final_m3": storage_final})
    continuity = coordinator.soil_moisture_continuity()
    return {"scenario_id": label, "scenario_digest": scenario.digest, "coordinator_cursor_digest": coordinator.digest,
            "ledger": coordinator.ledger.records(), "realized": realized_records, "occurrences": occurrence_summaries,
            "totals": totals, "storage_initial_m3": storage_initial, "storage_final_m3": storage_final,
            "aggregate_closure": {"residual_m3": residual, "bound_m3": bound, "n_terms": closure_terms},
            "final_water_sha256": final_hash, "final_t_s_hex": adapter.accepted.t_s.hex(), "refusal_injections": refusals,
            "adapter_counters": {"runs": adapter.runs, "published": adapter.published, "discarded": adapter.discarded, "refused": adapter.refused},
            "soil_moisture_continuity": {**continuity, "caveat": "interevent-only scenario: consecutive interevent occurrences share one "
                                         "persistent column; no rain/wind adapter exists and the native rain handoff still resets soil"},
            "column": column.summary(), "reserve_bytes": reserve, "prepare_s": prepare_s,
            "latitude_binding": None if latitude is None else latitude.record()}


def compare_with_control(label: str, driven: dict[str, Any], control: dict[str, Any], checks: Checks, m: SimpleNamespace) -> dict[str, Any]:
    comparisons = []
    for d, c in zip(driven["occurrences"], control["occurrences"]):
        same_chunks = d["chunks"] == c["chunks"]
        same_steps = d["n_steps"] == c["budget"]["n_steps"]
        same_budget = all(d["budget"][k] == c["budget"][k] for k in ("added_m3", "surface_return_m3", "drainage_m3", "evaporation_m3",
                                                                      "evaporation_unmet_m3", "elapsed_s"))
        same_state = d["t_end_hex"] == c["t_end_hex"] and d["water_sha256_after"] == c["water_sha256"]
        same_storage = d["storage_after_m3"] == c["storage_final_m3"]
        pet_scale = max(abs(c["pet_depth_sum_m"]), abs(c["library_integrated_depth_mm"] * 1e-3))
        pet_bound = m.conservation.volume_roundoff_bound_m3(len(c["chunks"]) + 2, pet_scale)
        pet_consistent = abs(c["pet_depth_sum_m"] - c["library_integrated_depth_mm"] * 1e-3) <= pet_bound and c["library_chunk_count"] == len(c["chunks"])
        same_pet = d["pet_mm_day_by_date"] == {k: v for k, v in control["pet_mm_day_by_date"].items() if k in d["pet_mm_day_by_date"]}
        row = {"occurrence_id": d["occurrence_id"], "same_chunks": same_chunks, "same_step_count": same_steps,
               "same_budget_exact": same_budget, "same_final_state_bitwise": same_state, "same_storage_exact": same_storage,
               "control_residual_m3": c["residual_m3"], "control_bound_m3": c["bound_m3"], "control_within_bound": abs(c["residual_m3"]) <= c["bound_m3"],
               "pet_chunk_sum_vs_library_integral_consistent": pet_consistent, "pet_bound_m": pet_bound,
               "same_daily_pet_values": same_pet}
        comparisons.append(row)
        checks.record(f"{label}.{d['occurrence_id']}.coordinator_driven_equals_independent_control",
                      same_chunks and same_steps and same_budget and same_state and same_storage and row["control_within_bound"]
                      and pet_consistent and same_pet, row)
    checks.record(f"{label}.final_state_equals_control", driven["final_water_sha256"] == control["occurrences"][-1]["water_sha256"]
                  and driven["storage_final_m3"] == control["storage_final_m3"] and driven["storage_initial_m3"] == control["storage_initial_m3"],
                  {"driven": driven["final_water_sha256"], "control": control["occurrences"][-1]["water_sha256"]})
    return {"occurrences": comparisons}


def run_unsupported_process_scenario(m: SimpleNamespace, tree: dict[str, Any], source_identity: str, checks: Checks) -> dict[str, Any]:
    """A transpiration-enabled interevent must be refused by the LEAF-less test adapter with the cursor unchanged."""
    scenario = m.masy_scenario.load_scenario(Path(tree["scenario"]), m.masy_scenario_io.IOLimits())
    adapter = NativeIntereventTestAdapter(m, source_identity)
    coordinator = m.masy_coordinator.Coordinator(scenario, {"interevent": adapter}, latitude=None)
    before = m.masy_scenario_io.canonical_json(coordinator.as_record())
    code = None
    try:
        coordinator.step()
    except m.masy_coordinator.CoordinatorError as exc:
        code = exc.code
    unchanged = m.masy_scenario_io.canonical_json(coordinator.as_record()) == before and coordinator.next_index == 0
    detail = {"refusal_code": code, "last_refusal": coordinator.last_refusal, "cursor_unchanged": unchanged, "staged": adapter.staged is not None}
    checks.record("unsupported_transpiration_gate_refused_without_native_work", code == "event_incomplete" and unchanged
                  and adapter.staged is None and adapter.refused == 1, detail)
    return detail


# --- main ----------------------------------------------------------------------------------------------------------------
def environment_record(backend: str, m: SimpleNamespace | None, xp: Any, device_id: int | None) -> dict[str, Any]:
    record: dict[str, Any] = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                              "numpy": np.__version__, "cwd": os.getcwd(), "backend": backend,
                              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if backend == "numba" and m is not None:
        record["numba_versions"] = m.extra["maple_syrup.subsurface_numba"].numba_versions()
    if backend == "cuda" and m is not None and xp is not None and xp is not np:
        try:
            props = xp.cuda.runtime.getDeviceProperties(device_id)
            record["cuda_device"] = {"id": device_id, "name": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
                                     "cupy": xp.__version__}
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal
            record["cuda_device"] = {"id": device_id, "query_error": f"{type(exc).__name__}: {exc}"}
        record["kernel_provenance"] = m.extra["maple_syrup.subsurface_cuda"].kernel_provenance()
    return record


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=1, sort_keys=False, allow_nan=False, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    defaults = default_roots()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", required=True, choices=("numpy", "numba", "cuda"))
    parser.add_argument("--output", required=True, type=Path, help="receipt JSON path")
    parser.add_argument("--work-dir", type=Path, default=None, help="scenario trees + overlay (default: <output>.work)")
    parser.add_argument("--reuse-work-dir", action="store_true", help="allow regenerating files in an existing work dir")
    parser.add_argument("--masy-src", type=Path, default=defaults.get("masy_src"), help="MASY candidate src root")
    parser.add_argument("--pet-src", type=Path, default=defaults.get("pet_src"), help="SYRUP weather_pet candidate src root")
    parser.add_argument("--syrup-src", type=Path, default=defaults.get("syrup_src"), help="current SYRUP src root (soil modules)")
    parser.add_argument("--maple-src", type=Path, default=defaults.get("maple_src"), help="IMMUTABLE MAPLE39 bundle source/src")
    parser.add_argument("--overlay-src", type=Path, default=None,
                        help="use a Root-prepared same-package import root (contains maple_syrup/) AS-IS instead of building the overlay")
    parser.add_argument("--overlay-identity", type=Path, default=None,
                        help="identity manifest JSON (source_files[rel].sha256) that every file of --overlay-src must match")
    args = parser.parse_args(argv)
    if args.overlay_identity is not None and args.overlay_src is None:
        parser.error("--overlay-identity requires --overlay-src")
    args.output = args.output.resolve()  # absolute paths in the receipt, so recorded identities do not depend on the cwd
    if args.work_dir is not None:
        args.work_dir = args.work_dir.resolve()
    started = time.perf_counter()
    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "created_utc": utc_now(), "driver": {"file": str(HERE), "sha256": sha256_file(HERE)},
                               "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                               "status": "incomplete", "checks": [], "not_accepted": [
                                   "persistent rain coupling (native rain handoff still resets soil water)",
                                   "plant transpiration / growth (no LEAF adapter; transpiration gates refuse)",
                                   "soil remapping after erosion / deposition", "combined native restart or native checkpoint",
                                   "wind / rain integration or mixed-event proofs (those remain mechanical scheduler tests)",
                                   "a production-safe coupled adapter (the staging helper is test-local; the driver publishes)",
                                   "source lock / qualified runtime (development overlay identities only)"]}
    checks = Checks()
    work = args.work_dir or args.output.with_name(args.output.name + ".work")
    try:
        required = ("masy_src", "maple_src") + (() if args.overlay_src is not None else ("pet_src", "syrup_src"))
        for name in required:
            if getattr(args, name) is None:
                raise DriverSetupError(f"--{name.replace('_', '-')} is required (no baseline.json default found)")
        if work.exists() and any(work.iterdir()) and not args.reuse_work_dir:
            raise DriverSetupError(f"work dir {work} exists and is not empty; pass --reuse-work-dir to regenerate its files")
        work.mkdir(parents=True, exist_ok=True)
        receipt["work_dir"] = str(work)
        if args.overlay_src is not None:
            receipt["overlay"] = prepared_overlay(args.overlay_src.resolve(),
                                                  None if args.overlay_identity is None else args.overlay_identity.resolve())
        else:
            receipt["overlay"] = build_overlay(work, Path(args.syrup_src).resolve(), Path(args.pet_src).resolve())
        m, sources = bind_sources(Path(args.maple_src).resolve(), Path(args.masy_src).resolve(), Path(receipt["overlay"]["root"]), args.backend)
        receipt["sources"] = sources
        checks.record("source_isolation_verified", True, {"roots": sources["sys_path_prepended"]})
    except (DriverSetupError, OSError) as exc:  # OSError: unwritable work dir / unreadable source root
        receipt.update(status="setup_refused", refusal=f"{type(exc).__name__}: {exc}", checks=checks.items)
        try:
            write_receipt(args.output, receipt)
        except OSError as write_exc:
            print(f"SETUP REFUSED (receipt not writable: {write_exc})", file=sys.stderr)
        print(f"SETUP REFUSED: {exc}", file=sys.stderr)
        return 3
    xp: Any = None
    device_id = None
    try:
        xp, device_id = device_namespace(m, args.backend)
    except m.subsurface.SubsurfaceBackendUnavailableError as exc:
        receipt.update(status="refused_backend_unavailable", refusal=str(exc), checks=checks.items,
                       environment=environment_record(args.backend, m, None, None))
        write_receipt(args.output, receipt)
        print(f"BACKEND REFUSED (no fallback): {exc}", file=sys.stderr)
        return 2
    receipt["environment"] = environment_record(args.backend, m, xp, device_id)
    origins = sources["module_origins"]
    source_identity = (f"subsurface {origins['maple_syrup.subsurface']['sha256'][:16]} weather_pet "
                       f"{origins['maple_syrup.weather_pet']['sha256'][:16]} masy.coordinator {origins['masy.coordinator']['sha256'][:16]} "
                       f"maple.core.backend {origins['maple.core.backend']['sha256'][:16]} (development overlay identities)")
    trees = {"explicit": write_scenario_tree(work / "scenario_explicit_pet", "phase0_native_explicit_pet", "explicit", EXPLICIT_ROWS, OCCURRENCES),
             "hargreaves": write_scenario_tree(work / "scenario_hargreaves", "phase0_native_hargreaves", "ecohyd_hargreaves", HARGREAVES_ROWS, OCCURRENCES),
             "unsupported": write_scenario_tree(work / "scenario_unsupported_transpiration", "phase0_native_unsupported", "explicit", EXPLICIT_ROWS,
                                                (("transpiring", "events/interevent_transpiration.json"),))}
    receipt["scenario_trees"] = trees
    receipt["column_fixture"] = {"depth_edges_m": list(DEPTH_EDGES_M), "theta_residual": THETA_RESIDUAL, "theta_field_capacity": THETA_FIELD_CAPACITY,
                                 "theta_saturation": THETA_SATURATION, "theta_evaporation_stress": THETA_STRESS,
                                 "diffusivity_m2_s": DIFFUSIVITY_M2_S, "evaporation_factor": EVAPORATION_FACTOR,
                                 "cell_area_m2": CELL_AREA_M2, "active_mask": list(ACTIVE_MASK), "theta_by_cell": [list(t) for t in THETA_BY_CELL],
                                 "intake": "zero (added_water_m == 0 everywhere)", "native_epoch_s": NATIVE_EPOCH_S}
    results: dict[str, Any] = {}
    try:
        layers = make_layers(m)
        water_host = initial_water_host(layers)
        receipt["column_fixture"]["initial_water_sha256"] = sha256_bytes(water_host.tobytes())
        try:
            native_duration_s(Fraction(1, 3))
            checks.record("non_representable_duration_refused", False, "Fraction(1,3) accepted")
        except NotRepresentableError as exc:
            checks.record("non_representable_duration_refused", True, str(exc))
        checks.record("representable_duration_accepted_exactly", native_duration_s(Fraction(21600)) == 21600.0 and native_duration_s(Fraction(1, 4)) == 0.25, None)
        for key, inject_at in (("explicit", 1), ("hargreaves", 1)):
            tree = trees[key]
            label = json.loads(Path(tree["scenario"]).read_text())["scenario_id"]
            t_run = time.perf_counter()
            driven = run_scenario(m, args.backend, xp, tree, layers, water_host, source_identity, checks, inject_at=inject_at)
            control = direct_control(m, args.backend, xp, tree, layers, water_host, driven["reserve_bytes"])
            comparison = compare_with_control(label, driven, control, checks, m)
            checks.record(f"{label}.scenario_files_byte_unchanged", tree_hashes(Path(tree["dir"])) == tree["files_sha256"], None)
            results[key] = {"coordinator_driven": driven, "direct_control": control, "comparison": comparison,
                            "wall_s": time.perf_counter() - t_run}
        results["unsupported"] = run_unsupported_process_scenario(m, trees["unsupported"], source_identity, checks)
    except CheckFailure as exc:
        receipt["stopped_on"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - any other failure is a failed run with the traceback in the receipt
        checks.record("driver_exception", False, {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    receipt["results"] = results
    receipt["checks"] = checks.items
    receipt["summary"] = {"n_checks": len(checks.items), "n_failed": len(checks.failed), "failed": checks.failed,
                          "wall_s": time.perf_counter() - started}
    receipt["status"] = "passed" if not checks.failed and results.get("unsupported") is not None else "failed"
    write_receipt(args.output, receipt)
    print(f"{receipt['status'].upper()}: {len(checks.items) - len(checks.failed)}/{len(checks.items)} checks passed "
          f"(backend {args.backend}); receipt {args.output}")
    for name in checks.failed:
        print(f"  FAILED {name}")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
