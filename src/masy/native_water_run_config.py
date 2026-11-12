"""Closed run-configuration schema `masy.native_water_run.v1` for the configured water-only native calendar run.

The file holds REFERENCES and explicit execution/resource/output limits only: an existing MAPLE case directory, a SYRUP layered-water parameter file, an existing MASY
scenario (with its event/weather files), the named source profile with the native bundle and the two expected manifests, reserves, caps and output cadence. It never
initializes a landscape or vegetation, supplies no physical default or site fallback, never resamples or resets native water. Parsing imports neither NumPy nor any native
model (stdlib + the existing strict document reader), so a refused configuration costs no physical work.

Paths: `maple_case_dir`, `syrup_parameters`, `scenario` and the two manifests are canonical relative references inside the configuration file's directory (the existing
`scenario_io.relative_reference` rules: no `..`, no absolute path, real path stays inside the directory; the case directory may not be a symlink). `source.native_target`,
`source.masy_root`, `source.syrup_root` are the declared absolute source roots (declarative expectation; the actual census/namespace/distribution/recheck are the authority) and
`output.run_dir` is the new run directory (relative values resolve against the configuration directory).
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

from masy.scenario_io import IOLimits, ScenarioIOError, load_document, parse_strict_json, parse_strict_yaml, read_bounded, relative_reference

__all__ = ["CONFIG_SCHEMA", "CONFIG_SCHEMA_V2", "LEAF_PROFILES", "SOURCE_MODES", "MAP_PIN_MAX_BYTES", "MAP_HASH_CHUNK", "MAX_MAP_PINS", "PARENT_BUFFER_BYTES", "PIN_METADATA_BYTES", "STRUCTURED_PARSE_FACTOR", "host_available_bytes", "make_availability", "PHYSICAL_CHECKPOINT_MODES", "STATUS_RESERVE_BYTES", "RunConfig", "RunConfigError", "load_manifest", "load_run_config",
           "parse_run_config", "resolved_config"]

CONFIG_SCHEMA = "masy.native_water_run.v1"
CONFIG_SCHEMA_V2 = "masy.native_water_run.v2"  # explicit opt-in sibling: the v1 keys plus the REQUIRED closed `leaf` block (LEAF participant + joint water/biology checkpoint)
PHYSICAL_CHECKPOINT_MODES = ("none", "terminal_standalone", "combined_terminal")  # combined_terminal: the sealed calendar+physical resumable artifact (masy.native_water_checkpoint)
STATUS_RESERVE_BYTES = 8192  # the terminal status is reserved INSIDE max_output_bytes / max_file_bytes (never added on top)
MAP_PIN_MAX_BYTES = 256 * 1024 * 1024  # = the layered factory's default per-map/total cap (LayeredCaseLimits.max_map_bytes); a larger referenced map refuses at pinning
MANIFEST_MAX_BYTES = 16 << 20
_SEAL = object()
STRUCTURED_PARSE_FACTOR = 12  # declared working-memory model of a strictly parsed YAML/JSON text: 12 x its bytes
MAX_MAP_PINS = 32  # = the layered factory's default max_map_files
PIN_METADATA_BYTES = 64 * 1024  # retained pin dictionaries (<= 32 names of <= 255 chars + 64-hex digests + case/scenario/params pins), counted once
MAP_HASH_CHUNK = 1 << 20
PARENT_BUFFER_BYTES = 4096  # the parent's own modelled buffer beyond the caller reserve before it launches any child
_TOP = ("schema", "run_id", "maple_case_dir", "syrup_parameters", "scenario", "profile_id", "source", "execution", "limits", "resources", "output")
SOURCE_MODES = ("development_snapshot", "production_committed")
_SOURCE_DEV = ("mode", "profile", "native_target", "masy_root", "masy_manifest", "syrup_root", "syrup_manifest", "origin")
_SOURCE_PROD = ("mode", "profile", "production_store", "production_lock")
_SOURCE_PROD_V2 = ("mode", "profile", "recipe", "recipe_sha256", "receipt", "receipt_sha256", "production_store", "production_lock", "release_masy_root", "leaf_capsule_namespace")  # v2 ONLY (explicit sibling)
_ORIGIN = ("label", "declared_by")
_STORM = ("max_dt_s", "min_dt_s", "max_retries", "max_steps", "max_recession_s")
_LIMITS = ("interevent_max_dt_s", "interevent_min_dt_s", "interevent_max_retries", "storm", "max_occurrences", "max_dates_per_interevent",
           "max_accepted_steps_per_occurrence", "call_step_budget", "max_record_bytes")
_RESOURCES = ("caller_reserve_bytes", "temporal_reserve_bytes", "lifecycle_extra_reserve_bytes", "preparation_reserve_bytes", "host_available_cap_bytes")
_OUTPUT = ("run_dir", "step_cadence", "max_output_bytes", "max_step_lines", "max_file_bytes", "physical_checkpoint", "compile_output_reserve_bytes")
_EXEC = ("quiet_policy",)
_CASE_FILES = ("case.yaml", "model.yaml", "wind.yaml")
#: CLOSED pure-literal source-profile capability table, kept IN this module because the parent closure is exactly five masy modules (`native_source_gate.PARENT_CLOSURE`) and
#: `native_water_backend` is not one of them. `native_water_backend.PROFILE_CAPABILITY` is a deliberate duplicate; a test pins the two equal and equal to `SOURCE_PROFILES` names.
PROFILE_CAPABILITY = {"c3_literal308": "cpu", "native9f_layered_factory_95ef_cpu_v1": "cpu", "native9f_layered_factory_afca_cpu_v1": "cpu", "native9f_layered_factory_afca_cuda_v1": "cuda"}
_LEAF = ("leaf_variant", "leaf_root", "catalog", "plants", "cadence_days", "geometry", "budgets", "io")
_LEAF_PROD = tuple(k for k in _LEAF if k != "leaf_root")  # production: the LEAF root is DERIVED from the immutable capsule, never supplied
_LEAF_GEOM = ("max_edges", "max_bytes", "canopy_max_edges", "canopy_max_bytes", "builder")
_LEAF_BUD = ("leaf_budget_bytes", "checkpoint")
_LEAF_CK = ("max_host_bytes", "max_device_bytes", "max_disk_bytes", "max_biology_bytes")
_LEAF_IO = ("max_catalog_bytes", "max_plants_bytes", "max_plant_records")
LEAF_PROFILES = ("native9f_layered_factory_afca_cpu_v1", "native9f_layered_factory_afca_cuda_v1")  # the LEAF participant is qualified only with the 109 AFCA water profiles
LEAF_VARIANTS = ("8da",)  # the file CLI serves the 8da LEAF family (plain CSV + companion metadata); d4fe has no such loader
_CUDA = ("device_id", "device_reserve_bytes", "device_cap_bytes", "native_reserve_bytes")  # OPTIONAL top-level `cuda` block: REQUIRED iff the profile declares the cuda capability
_IO = IOLimits(max_file_bytes=1 << 20, max_entries=4096, max_text_chars=1 << 20, max_depth=8, max_path_chars=1024, max_files=64)


class RunConfigError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _closed(value: Any, keys: tuple, where: str) -> dict:
    if not isinstance(value, dict):
        raise RunConfigError("invalid_config", f"{where} must be a mapping")
    unknown, missing = sorted(set(value) - set(keys)), sorted(set(keys) - set(value))
    if unknown or missing:
        raise RunConfigError("invalid_config", f"{where}: unknown keys {unknown}, missing keys {missing}")
    return value


def _int(value: Any, where: str, lo: int = 0) -> int:
    if type(value) is not int or value < lo:
        raise RunConfigError("invalid_config", f"{where} must be a plain integer >= {lo}, got {value!r}")
    return value


def _real(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise RunConfigError("invalid_config", f"{where} must be a finite positive number, got {value!r}")
    return float(value)


def _text(value: Any, where: str, limit: int = 1024) -> str:
    if type(value) is not str or not value or len(value) > limit or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise RunConfigError("invalid_config", f"{where} must be a non-empty control-free string of at most {limit} characters")
    return value


def _absolute(value: Any, where: str) -> str:
    text = _text(value, where)
    if not os.path.isabs(text) or os.path.normpath(text) != text:
        raise RunConfigError("invalid_config", f"{where} must be a normalised absolute path")
    return text


@dataclass(frozen=True, eq=False, init=False)
class RunConfig:
    """Resolved, immutable, SEALED run configuration (paths resolved, referenced files pinned by SHA-256 of the bytes actually read). Constructed only by the parser:
    `dataclasses.replace` (which calls the constructor without the private seal) refuses, and `run_configured` revalidates the seal and the config bytes at entry.
    The two source manifests are referenced by path and read only by the worker after the availability cap is known (`load_manifest`)."""

    run_id: str
    config_path: str
    config_sha256: str
    config_bytes: bytes
    maple_case_dir: str
    case_files: Any
    syrup_parameters: str
    syrup_parameters_sha256: str
    syrup_map_pins: Any
    scenario: str
    scenario_sha256: str
    profile_id: str
    source: Any
    quiet_policy: str
    limits: Any
    resources: Any
    output: Any
    pinned: bool
    cuda: Any = None  # explicit CUDA placement (device id + caller device reserve + declared device cap + native reserve); None for every CPU profile
    leaf: Any = None  # v2 only: the resolved/pinned closed `leaf` block (None for every v1 configuration)
    schema: str = CONFIG_SCHEMA  # the schema the file declared (v1 default: every historical constructor call is unchanged)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        names = [f.name for f in fields(self)]
        if not args or args[0] is not _SEAL or kwargs:
            raise RunConfigError("sealed_config", "a RunConfig is created only by the run-config loader (dataclasses.replace / direct construction refuses)")
        args = list(args)
        defaults = {"cuda": None, "leaf": None, "schema": CONFIG_SCHEMA}
        while len(args) < len(names) + 1 and names[len(args) - 1] in defaults:  # historical constructor calls omit the trailing optional fields
            args.append(defaults[names[len(args) - 1]])
        if len(args) != len(names) + 1:
            raise RunConfigError("sealed_config", "internal: wrong field count")
        for name, value in zip(names, args[1:]):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_sealed", True)

    def revalidate(self) -> None:
        """Entry check of the worker: sealed, config bytes still hash to the recorded pin."""
        if self.__dict__.get("_sealed") is not True or hashlib.sha256(self.config_bytes).hexdigest() != self.config_sha256:
            raise RunConfigError("sealed_config", "the run configuration is not the sealed object the loader produced")
        if self.pinned is not True:
            raise RunConfigError("unpinned_config", "the run configuration was loaded without pinning its references (parent form); the worker needs the pinned form")

    def record(self) -> dict[str, Any]:
        return {"schema": self.schema, "run_id": self.run_id, "config_sha256": self.config_sha256, "maple_case_dir": self.maple_case_dir,
                "case_files": dict(self.case_files), "syrup_parameters": self.syrup_parameters, "syrup_parameters_sha256": self.syrup_parameters_sha256, "syrup_map_pins": dict(self.syrup_map_pins),
                "scenario": self.scenario, "scenario_sha256": self.scenario_sha256, "profile_id": self.profile_id, "source": dict(self.source),
                "quiet_policy": self.quiet_policy, "limits": _thaw(self.limits), "resources": dict(self.resources), "output": dict(self.output),
                **({} if self.cuda is None else {"cuda": dict(self.cuda)}),  # the CPU record is byte-identical (no `cuda` key)
                **({} if self.leaf is None else {"leaf": _thaw(self.leaf)})}  # the v1 record has no `leaf` key


def resolved_config(cfg: "RunConfig", source: Any, leaf_root: str) -> "RunConfig":
    """The v2 production configuration with its `source` completed by the verified sealed roots (development-shaped keys the worker already consumes) and the capsule-derived LEAF root.
    Sealed like every RunConfig (same config bytes/pins); only the production resolution (`masy.production_runtime.resolve`) produces the arguments."""
    if cfg.schema != CONFIG_SCHEMA_V2 or cfg.source["mode"] != "production_committed" or cfg.leaf is None:
        raise RunConfigError("invalid_config", "resolved_config applies only to a v2 production_committed configuration")
    values = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    values["source"] = MappingProxyType({**dict(cfg.source), **dict(source)})
    values["leaf"] = MappingProxyType({**dict(cfg.leaf), "leaf_root": str(leaf_root)})
    return RunConfig(_SEAL, *[values[f.name] for f in fields(cfg)])


def _thaw(value: Any) -> Any:
    if isinstance(value, MappingProxyType) or isinstance(value, dict):
        return {k: _thaw(v) for k, v in value.items()}
    return value


def host_available_bytes() -> int:
    """Stdlib equivalent of SYRUP's `default_host_available_bytes` (`sysconf(SC_AVPHYS_PAGES) * SC_PAGE_SIZE`): a reading, not a reservation."""
    try:
        pages, page = os.sysconf("SC_AVPHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError) as exc:
        raise RunConfigError("invalid_availability", f"the host cannot report available memory ({type(exc).__name__}: {exc})") from None
    if pages < 0 or page <= 0:
        raise RunConfigError("invalid_availability", "the host reported an invalid available-memory figure")
    return int(pages) * int(page)


def _strict_nonneg(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunConfigError("invalid_availability", f"{what} must be a non-Boolean integer >= 0, got {value!r}")
    return int(value)


def make_availability(cap: int | None, fn: Any = None):
    """The ONE availability reading: `fn` (unit controls only; default the host helper), validated as a non-Boolean integer >= 0 on every call, bounded by the declared cap."""
    if cap is not None:
        cap = _strict_nonneg(cap, "host_available_cap_bytes")
    if fn is not None and not callable(fn):
        raise RunConfigError("invalid_availability", "available_bytes_fn must be callable")
    base = fn if fn is not None else host_available_bytes

    def reading() -> int:
        value = _strict_nonneg(base(), "the availability reading")
        return value if cap is None else min(value, cap)
    return reading


class _ColdBudget:
    """Fresh availability reading against `caller_reserve + buffer model` BEFORE each reference body is opened/read/hashed/parsed (typed `cold_memory`)."""

    def __init__(self, avail: Any, reserve: int, stages: list | None) -> None:
        self.avail, self.reserve, self.stages = avail, int(reserve), stages

    def require(self, nbytes: int, what: str) -> None:
        if self.stages is not None:
            self.stages.append((what, int(nbytes)))
        reading = self.avail()
        if reading < self.reserve + int(nbytes):
            raise RunConfigError("cold_memory", f"{what} needs {int(nbytes)} B + the {self.reserve}-B caller reserve but only {reading} B are available")


def _ref(base: Path, value: Any, where: str) -> Path:
    try:
        return relative_reference(value, base, _IO, where)
    except ScenarioIOError as exc:
        raise RunConfigError("invalid_reference", str(exc)) from exc


def _size_of(path: Path, where: str) -> int:
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise RunConfigError("reference_unreadable", f"{where}: {exc}") from exc
    return int(st.st_size)


def _pin(path: Path, where: str, budget: "_ColdBudget | None" = None) -> str:
    if budget is not None:
        budget.require(2 * _size_of(path, where) + 4096, f"reading {where}")
    try:
        _data, pin = read_bounded(path, _IO.max_file_bytes)
    except ScenarioIOError as exc:
        raise RunConfigError("reference_unreadable", f"{where}: {exc}") from exc
    return pin["sha256"]


def load_manifest(path: Path, where: str, max_bytes: int = MANIFEST_MAX_BYTES, admit: Any = None) -> dict[str, str]:
    """The worker's manifest read: ONE closed parser, `masy.native_source_gate.read_manifest` (size cap before the body, lexical preflight before `json.loads`, exact parse
    admission), with the 16 MiB cap never raised. `admit(nbytes, what)` is the caller's fresh availability + reserve check (it may raise)."""
    from masy import native_source_gate as gate

    if max_bytes > MANIFEST_MAX_BYTES:
        raise RunConfigError("manifest_too_large", f"{where}: the {MANIFEST_MAX_BYTES}-B manifest cap cannot be raised")
    try:
        files, _sha = gate.read_manifest(str(path), gate.GateLimits(max_manifest_bytes=max_bytes), admit or (lambda n, w: None))
    except gate.GateRefusal as exc:
        raise RunConfigError(exc.code, f"{where}: {exc.detail}") from None
    return MappingProxyType(dict(files))


def load_run_config(path: str | os.PathLike, *, references: bool = True, available_bytes_fn: Any = None, _stages: list | None = None) -> RunConfig:
    """Read the configuration file ONCE (bounded, no-follow); the SHA-256, the parsed document and `config_bytes` are all of those same bytes."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".json", ".yaml", ".yml"):
        raise RunConfigError("config_unreadable", f"{path}: unsupported suffix {suffix!r}")
    try:
        data, pin = read_bounded(path, _IO.max_file_bytes)
        value = parse_strict_json(data, _IO, str(path)) if suffix == ".json" else parse_strict_yaml(data, _IO, str(path))
    except ScenarioIOError as exc:
        raise RunConfigError("config_unreadable", str(exc)) from exc
    return parse_run_config(value, base_dir=path.resolve().parent, config_path=str(path), config_sha256=pin["sha256"], config_bytes=data, references=references,
                            available_bytes_fn=available_bytes_fn, _stages=_stages)


def _walk_files(value: Any, out: list) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "file" and isinstance(v, str):
                out.append(v)
            else:
                _walk_files(v, out)
    elif isinstance(value, list):
        for v in value:
            _walk_files(v, out)


def _pin_params_and_maps(params: Path, budget: "_ColdBudget") -> tuple[str, dict[str, str]]:
    """The layered parameter file is read ONCE: its SHA-256 and its referenced map names come from the same bytes. Each referenced `.npy` map (single-component name beside it) is
    hashed in `MAP_HASH_CHUNK` chunks, no-follow, size-capped, each BEFORE its stream is opened guarded by the cold budget. The worker re-reads and re-verifies everything."""
    budget.require(2 * _size_of(params, "syrup_parameters") + 4096, "reading syrup_parameters")
    try:
        data, pin = read_bounded(params, 1 << 20)
    except ScenarioIOError as exc:
        raise RunConfigError("reference_unreadable", f"syrup_parameters: {exc}") from exc
    budget.require(STRUCTURED_PARSE_FACTOR * len(data) + 4096, "parsing syrup_parameters")
    try:
        doc = parse_strict_yaml(data, _IO, "syrup_parameters") if params.suffix.lower() in (".yaml", ".yml") else parse_strict_json(data, _IO, "syrup_parameters")
    except ScenarioIOError as exc:
        raise RunConfigError("reference_unreadable", f"syrup_parameters: {exc}") from exc
    names: list[str] = []
    _walk_files(doc, names)
    del doc
    unique = sorted(set(names))
    if len(unique) > MAX_MAP_PINS:
        raise RunConfigError("invalid_reference", f"syrup_parameters references {len(unique)} maps, over the {MAX_MAP_PINS}-map pin limit")
    pins: dict[str, str] = {}
    if not unique:
        return pin["sha256"], pins
    try:
        dirfd = os.open(params.parent, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        raise RunConfigError("reference_unreadable", f"parameter directory cannot be opened: {exc}") from exc
    try:
        for name in unique:
            if not name or "/" in name or "\\" in name or name in (".", "..") or name.startswith("~") or "\x00" in name or len(name) > 255:
                raise RunConfigError("invalid_reference", f"syrup_parameters map name {name!r} must be one plain file name")
            budget.require(2 * MAP_HASH_CHUNK + 4096, f"hashing map {name}")
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dirfd)
            except OSError as exc:
                raise RunConfigError("reference_unreadable", f"map {name!r}: {exc}") from exc
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAP_PIN_MAX_BYTES:
                    raise RunConfigError("reference_unreadable", f"map {name!r} is not a regular file within {MAP_PIN_MAX_BYTES} B")
                digest, total = hashlib.sha256(), 0
                while True:
                    chunk = os.read(fd, MAP_HASH_CHUNK)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAP_PIN_MAX_BYTES:
                        raise RunConfigError("reference_unreadable", f"map {name!r} grew beyond {MAP_PIN_MAX_BYTES} B while pinned")
                    digest.update(chunk)
                if total != info.st_size:
                    raise RunConfigError("reference_unreadable", f"map {name!r} changed while pinned")
                pins[name] = digest.hexdigest()
            finally:
                os.close(fd)
    finally:
        os.close(dirfd)
    return pin["sha256"], pins


def _hex64(value: Any, where: str) -> str:
    if type(value) is not str or len(value) != 64 or not set(value) <= set("0123456789abcdef"):
        raise RunConfigError("invalid_config", f"{where} must be 64 lowercase hex digits")
    return value


def _parse_leaf(raw: Any, base: Path, production: bool = False) -> Any:
    """Closed stdlib validation of the v2 `leaf` block (no reference is touched): the LEAF family/root, the species catalog and initial plant CSV references (the CSV's companion metadata sits beside it),
    the growth cadence (whole days, <= the LEAF weekly cap), the sparse root/canopy caps, the LEAF component budget, the SEPARATE joint-checkpoint caps and the plant-file I/O caps."""
    leaf = _closed(raw, _LEAF_PROD if production else _LEAF, "leaf")
    if leaf["leaf_variant"] not in LEAF_VARIANTS:
        raise RunConfigError("invalid_config", f"leaf.leaf_variant must be one of {LEAF_VARIANTS}, got {leaf['leaf_variant']!r}")
    cadence = _int(leaf["cadence_days"], "leaf.cadence_days", 1)
    if cadence > 7:
        raise RunConfigError("invalid_config", "leaf.cadence_days must be an integer in 1..7 (the LEAF weekly growth envelope)")
    g = _closed(leaf["geometry"], _LEAF_GEOM, "leaf.geometry")
    if g["builder"] not in ("numpy", "compiled"):
        raise RunConfigError("invalid_config", "leaf.geometry.builder must be 'numpy' or 'compiled'")
    b = _closed(leaf["budgets"], _LEAF_BUD, "leaf.budgets")
    ck = _closed(b["checkpoint"], _LEAF_CK, "leaf.budgets.checkpoint")
    io = _closed(leaf["io"], _LEAF_IO, "leaf.io")
    catalog = _ref(base, leaf["catalog"], "leaf.catalog")
    plants = _ref(base, leaf["plants"], "leaf.plants")
    if str(plants).endswith(".metadata.json"):
        raise RunConfigError("invalid_config", "leaf.plants names the plain CSV, not its companion metadata file")
    meta = Path(os.path.splitext(str(plants))[0] + ".metadata.json")
    return MappingProxyType({
        "leaf_variant": leaf["leaf_variant"], "leaf_root": "" if production else _absolute(leaf["leaf_root"], "leaf.leaf_root"), "catalog": str(catalog), "plants": str(plants), "plants_metadata": str(meta), "cadence_days": cadence,
        "geometry": MappingProxyType({"max_edges": _int(g["max_edges"], "leaf.geometry.max_edges", 1), "max_bytes": _int(g["max_bytes"], "leaf.geometry.max_bytes", 1),
                                      "canopy_max_edges": _int(g["canopy_max_edges"], "leaf.geometry.canopy_max_edges", 1), "canopy_max_bytes": _int(g["canopy_max_bytes"], "leaf.geometry.canopy_max_bytes", 1),
                                      "builder": g["builder"]}),
        "budgets": MappingProxyType({"leaf_budget_bytes": _int(b["leaf_budget_bytes"], "leaf.budgets.leaf_budget_bytes", 1),
                                     "checkpoint": MappingProxyType({k: _int(ck[k], f"leaf.budgets.checkpoint.{k}", 0) for k in _LEAF_CK})}),
        "io": MappingProxyType({"max_catalog_bytes": _int(io["max_catalog_bytes"], "leaf.io.max_catalog_bytes", 1), "max_plants_bytes": _int(io["max_plants_bytes"], "leaf.io.max_plants_bytes", 1),
                                "max_plant_records": _int(io["max_plant_records"], "leaf.io.max_plant_records", 1)}),
        "pins": MappingProxyType({})})


def _pin_leaf(leaf: Any, budget: "_ColdBudget") -> Any:
    """Pin the three LEAF input files by the SHA-256 of the bytes actually read, each guarded by a fresh reading + the caller reserve BEFORE it is opened, each under its OWN declared I/O cap (the 1 MiB
    generic reader cap does not apply to a population file)."""
    io = leaf["io"]
    pins = {}
    for key, path, cap in (("catalog_sha256", leaf["catalog"], io["max_catalog_bytes"]), ("plants_sha256", leaf["plants"], io["max_plants_bytes"]),
                           ("metadata_sha256", leaf["plants_metadata"], io["max_catalog_bytes"])):
        size = _size_of(Path(path), f"leaf file {os.path.basename(path)}")
        if size > cap:
            raise RunConfigError("reference_too_large", f"{path} is {size} B, over its declared leaf.io cap {cap} B (checked from metadata before any read)")
        budget.require(2 * size + 4096, f"reading leaf file {os.path.basename(path)}")
        try:
            _data, pin = read_bounded(Path(path), cap)
        except ScenarioIOError as exc:
            raise RunConfigError("reference_unreadable", f"{path}: {exc}") from exc
        pins[key] = pin["sha256"]
    out = dict(leaf)
    out["pins"] = MappingProxyType(pins)
    return MappingProxyType(out)


def parse_run_config(value: Any, *, base_dir: Path, config_path: str, config_sha256: str, config_bytes: bytes, references: bool = True, available_bytes_fn: Any = None,
                     _stages: list | None = None) -> RunConfig:
    """Closed validation of EVERY value first (no reference is touched); only then, with `references=True`, availability is read and each reference body is pinned guarded
    by a fresh reading + the caller reserve + an explicit buffer model BEFORE it is opened. `references=False` (the parent's form) never opens a reference."""
    cuda_raw = value.get("cuda") if isinstance(value, dict) else None
    has_cuda = isinstance(value, dict) and "cuda" in value
    is_v2 = isinstance(value, dict) and value.get("schema") == CONFIG_SCHEMA_V2
    if is_v2 and "leaf" not in value:
        raise RunConfigError("invalid_config", f"a {CONFIG_SCHEMA_V2} configuration REQUIRES the closed `leaf` block (use {CONFIG_SCHEMA} for a water-only run)")
    leaf_raw = value.get("leaf") if is_v2 else None
    drop = {"cuda"} if has_cuda else set()
    if is_v2:
        drop.add("leaf")
    top = _closed({k: v for k, v in value.items() if k not in drop} if drop else value, _TOP, "run config")  # a v1 file carrying `leaf` stays an unknown key (refused)
    schema_ok = CONFIG_SCHEMA_V2 if is_v2 else CONFIG_SCHEMA
    if top["schema"] != schema_ok:
        raise RunConfigError("invalid_config", f"schema must be {CONFIG_SCHEMA!r} or {CONFIG_SCHEMA_V2!r}, got {top['schema']!r}")
    run_id = _text(top["run_id"], "run_id", 128)
    base = Path(os.path.realpath(base_dir))
    case_dir = _ref(base, top["maple_case_dir"], "maple_case_dir")
    params = _ref(base, top["syrup_parameters"], "syrup_parameters")
    scenario = _ref(base, top["scenario"], "scenario")
    profile_id = _text(top["profile_id"], "profile_id", 128)
    src_raw = top["source"]
    mode = src_raw.get("mode") if isinstance(src_raw, dict) else None
    if mode not in SOURCE_MODES:
        raise RunConfigError("invalid_config", f"source.mode is REQUIRED and must be one of {SOURCE_MODES} (no implicit default), got {mode!r}")
    if mode == "production_committed" and is_v2:
        src = _closed(src_raw, _SOURCE_PROD_V2, "source")
        source = {"mode": mode, "profile": _text(src["profile"], "source.profile", 128), "production_store": _absolute(src["production_store"], "source.production_store"),
                  "production_lock": _hex64(src["production_lock"], "source.production_lock"), "recipe": _absolute(src["recipe"], "source.recipe"),
                  "recipe_sha256": _hex64(src["recipe_sha256"], "source.recipe_sha256"), "receipt": _absolute(src["receipt"], "source.receipt"),
                  "receipt_sha256": _hex64(src["receipt_sha256"], "source.receipt_sha256"), "release_masy_root": _absolute(src["release_masy_root"], "source.release_masy_root"),
                  "leaf_capsule_namespace": _absolute(src["leaf_capsule_namespace"], "source.leaf_capsule_namespace")}
    elif mode == "production_committed":
        src = _closed(src_raw, _SOURCE_PROD, "source")
        source = {"mode": mode, "profile": _text(src["profile"], "source.profile", 128), "production_store": _absolute(src["production_store"], "source.production_store"),
                  "production_lock": _text(src["production_lock"], "source.production_lock", 1024)}
    else:
        src = _closed(src_raw, _SOURCE_DEV, "source")
        origin = _closed(src["origin"], _ORIGIN, "source.origin")
        source = {"mode": mode, "profile": _text(src["profile"], "source.profile", 128), "native_target": _absolute(src["native_target"], "source.native_target"),
                  "masy_root": _absolute(src["masy_root"], "source.masy_root"), "syrup_root": _absolute(src["syrup_root"], "source.syrup_root"),
                  "masy_manifest": str(_ref(base, src["masy_manifest"], "source.masy_manifest")), "syrup_manifest": str(_ref(base, src["syrup_manifest"], "source.syrup_manifest")),
                  "origin": {"label": _text(origin["label"], "source.origin.label", 512), "declared_by": _text(origin["declared_by"], "source.origin.declared_by", 256)}}
    capability = PROFILE_CAPABILITY.get(source["profile"])
    cuda = None
    if capability == "cuda":
        if not has_cuda:
            raise RunConfigError("invalid_config", "the CUDA source profile requires the explicit top-level `cuda` block (device_id, device_reserve_bytes, device_cap_bytes, native_reserve_bytes); there is no default device or cap")
        c = _closed(cuda_raw, _CUDA, "cuda")
        cuda = MappingProxyType({"device_id": _int(c["device_id"], "cuda.device_id", 0), "device_reserve_bytes": _int(c["device_reserve_bytes"], "cuda.device_reserve_bytes", 0),
                                 "device_cap_bytes": _int(c["device_cap_bytes"], "cuda.device_cap_bytes", 1), "native_reserve_bytes": _int(c["native_reserve_bytes"], "cuda.native_reserve_bytes", 0)})
    elif has_cuda:
        raise RunConfigError("invalid_config", f"the `cuda` block is refused for source profile {source['profile']!r} (capability {capability!r}); it is accepted only with a CUDA profile")
    ex = _closed(top["execution"], _EXEC, "execution")
    if ex["quiet_policy"] != "default":
        raise RunConfigError("invalid_config", "execution.quiet_policy must be 'default' (the actual LayeredQuietPolicy defaults; no override in this slice)")
    lim = _closed(top["limits"], _LIMITS, "limits")
    storm = _closed(lim["storm"], _STORM, "limits.storm")
    limits = MappingProxyType({
        "interevent_max_dt_s": _real(lim["interevent_max_dt_s"], "limits.interevent_max_dt_s"), "interevent_min_dt_s": _real(lim["interevent_min_dt_s"], "limits.interevent_min_dt_s"),
        "interevent_max_retries": _int(lim["interevent_max_retries"], "limits.interevent_max_retries", 0),
        "storm": MappingProxyType({"max_dt_s": _real(storm["max_dt_s"], "limits.storm.max_dt_s"), "min_dt_s": _real(storm["min_dt_s"], "limits.storm.min_dt_s"),
                                   "max_retries": _int(storm["max_retries"], "limits.storm.max_retries", 0), "max_steps": _int(storm["max_steps"], "limits.storm.max_steps", 1),
                                   "max_recession_s": _real(storm["max_recession_s"], "limits.storm.max_recession_s")}),
        "max_occurrences": _int(lim["max_occurrences"], "limits.max_occurrences", 1), "max_dates_per_interevent": _int(lim["max_dates_per_interevent"], "limits.max_dates_per_interevent", 1),
        "max_accepted_steps_per_occurrence": _int(lim["max_accepted_steps_per_occurrence"], "limits.max_accepted_steps_per_occurrence", 1),
        "call_step_budget": _int(lim["call_step_budget"], "limits.call_step_budget", 1), "max_record_bytes": _int(lim["max_record_bytes"], "limits.max_record_bytes", 1)})
    if limits["interevent_min_dt_s"] > limits["interevent_max_dt_s"] or limits["storm"]["min_dt_s"] > limits["storm"]["max_dt_s"]:
        raise RunConfigError("invalid_config", "a minimum dt exceeds its maximum dt")
    res = _closed(top["resources"], _RESOURCES, "resources")
    cap = res["host_available_cap_bytes"]
    resources = MappingProxyType({"caller_reserve_bytes": _int(res["caller_reserve_bytes"], "resources.caller_reserve_bytes"),
                                  "temporal_reserve_bytes": _int(res["temporal_reserve_bytes"], "resources.temporal_reserve_bytes"),
                                  "lifecycle_extra_reserve_bytes": _int(res["lifecycle_extra_reserve_bytes"], "resources.lifecycle_extra_reserve_bytes"),
                                  "preparation_reserve_bytes": _int(res["preparation_reserve_bytes"], "resources.preparation_reserve_bytes"),
                                  "host_available_cap_bytes": None if cap is None else _int(cap, "resources.host_available_cap_bytes")})
    out = _closed(top["output"], _OUTPUT, "output")
    mode = out["physical_checkpoint"]
    if mode not in PHYSICAL_CHECKPOINT_MODES:
        raise RunConfigError("invalid_config", f"output.physical_checkpoint must be one of {PHYSICAL_CHECKPOINT_MODES}")
    run_dir = _text(out["run_dir"], "output.run_dir", 1024)
    run_dir = run_dir if os.path.isabs(run_dir) else str(Path(os.path.realpath(base)) / run_dir)
    output = MappingProxyType({"run_dir": os.path.normpath(run_dir), "step_cadence": _int(out["step_cadence"], "output.step_cadence", 1),
                               "max_output_bytes": _int(out["max_output_bytes"], "output.max_output_bytes", 4 * STATUS_RESERVE_BYTES), "max_step_lines": _int(out["max_step_lines"], "output.max_step_lines", 1),
                               "max_file_bytes": _int(out["max_file_bytes"], "output.max_file_bytes", STATUS_RESERVE_BYTES), "physical_checkpoint": mode,
                               "compile_output_reserve_bytes": _int(out["compile_output_reserve_bytes"], "output.compile_output_reserve_bytes", 16)})
    if output["compile_output_reserve_bytes"] + STATUS_RESERVE_BYTES >= output["max_output_bytes"]:
        raise RunConfigError("invalid_config", "output.compile_output_reserve_bytes plus the status reserve must be smaller than output.max_output_bytes")
    leaf = None
    if is_v2:
        if source["profile"] not in LEAF_PROFILES:
            raise RunConfigError("invalid_config", f"the LEAF participant requires one of the AFCA profiles {LEAF_PROFILES}, got {source['profile']!r}")
        if output["physical_checkpoint"] == "terminal_standalone":
            raise RunConfigError("invalid_config", "output.physical_checkpoint 'terminal_standalone' is the WATER-ONLY physical bundle and would drop the biology: a LEAF run uses 'none' or 'combined_terminal' (the joint artifact)")
        leaf = _parse_leaf(leaf_raw, base, production=source["mode"] == "production_committed")
    case_files: dict[str, str] = {}
    params_sha, map_pins, scenario_sha = "", {}, ""
    if references:
        if case_dir.is_symlink() or not case_dir.is_dir():
            raise RunConfigError("invalid_reference", "maple_case_dir must be a real directory (not a symlink)")
        avail = make_availability(resources["host_available_cap_bytes"], available_bytes_fn)
        budget = _ColdBudget(avail, resources["caller_reserve_bytes"], _stages)
        budget.require(PIN_METADATA_BYTES, "retained pin metadata")
        for name in _CASE_FILES:
            case_files[name] = _pin(case_dir / name, f"maple_case_dir/{name}", budget)
        params_sha, map_pins = _pin_params_and_maps(params, budget)
        scenario_sha = _pin(scenario, "scenario", budget)
        if leaf is not None:
            leaf = _pin_leaf(leaf, budget)
    return RunConfig(_SEAL, run_id, config_path, config_sha256, bytes(config_bytes), str(case_dir), MappingProxyType(case_files), str(params), params_sha,
                     MappingProxyType(map_pins), str(scenario), scenario_sha, profile_id, MappingProxyType(source), "default", limits, resources, output, references, cuda, leaf,
                     CONFIG_SCHEMA_V2 if is_v2 else CONFIG_SCHEMA)
