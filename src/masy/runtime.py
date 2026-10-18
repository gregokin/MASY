"""MASY-03 locked CPU runtime: `prepare` and `verify` (parent side).

Entered only through `tools/masy_runtime.py` (`-I -B`). Stdlib plus `packaging` (present in the MAPLE CPU `.venv`);
this process never imports `maple` or `maple_syrup`. Every upstream call runs in a fresh bounded child that imports
only sealed copies (`masy.runtime_worker`). Nothing is installed, built, downloaded or garbage-collected, and no
transport entry point exists: a locked/verified runtime is CPU runtime identity and native interface delivery only
(`transport_ready` is always false).

prepare (selection = current local committed HEAD of each configured upstream and of MASY itself):

1. closed-schema config; interpreter identity (the configured venv spelling is kept for launching; its realpath,
   hash and environment are recorded); store location separation; existing store layout check (read-only);
2. BEFORE observation with the existing `sourceobs.observe_upstream` guards for MAPLE, SYRUP and MASY (clean
   committed scope, blob = HEAD, no untracked/ignored runtime input, symlink or special file) plus the MASY
   launcher's Git status/blob, `HEAD^{tree}` and the installed MAPLE dist-info bytes;
3. read-only pass over exactly the committed `src/` + `pyproject.toml` blobs (+ MASY launcher): sha256 and Git
   blob of the bytes on disk, cross-checked with the observation; static metadata/`__version__` checks;
4. disk/quota preflight BEFORE the store, a run directory or any object exists;
5. store and run directory creation; content-addressed, read-only source objects (staged, fully verified against
   their final identity, published with renameat2(RENAME_NOREPLACE); an existing object is fully verified and
   reused, never repaired); run-owned metadata overlays for `maple-syrup` and `masy` (labelled, not installations);
6. child `export`: SYRUP's own `dependency_bundle.export_bundle` from the sealed SYRUP copy; parent cross-checks the
   upstream manifest and publishes it no-replace (or verifies an identical existing bundle);
7. child `materialize`: upstream `verify_bundle` / `materialize_bundle` / `verify_materialized`;
8. child `runtime`: upstream `check_runtime` (unmodified resolver on the frozen cache), MASY native audit, both
   native checkpoint registries, origins, selected metadata, digests, environment and compiled identities;
9. AFTER observation must equal BEFORE; the lock (stable identity only) is validated with the verify validator and
   published O_EXCL + link (no replace; identical bytes are reused, different bytes refuse).

verify (explicit lock; never reads current upstream repositories, never selects anything new): bounded strict lock
read, full re-verification of every referenced object, interpreter identity, sealed bootstrap hash, one fresh
`runtime` child with the lock's own sealed code, then strict comparison of the closed stable identity. Per-run
operational paths (pycache prefix, caches, home, request, pid) are validated against the new run directory and
saved in its receipts, never compared with older runs.

Every stage appends to `<store>/runs/<run_id>/stages.jsonl`; refusals keep the run directory, staging and worker
outputs. Exit codes: 0 runtime_locked/runtime_verified, 2 refused, 3 incompatible, 1 internal error.
"""

from __future__ import annotations

import argparse
import ast
import base64
import configparser
import ctypes
import email.parser
import errno
import hashlib
import json
import math
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from masy import runtime_worker as _worker
from masy.sourceobs import Limits, SourceRefusal, canonical_json, diff_observations, observe_upstream, run_git

__all__ = [
    "CONFIG_SCHEMA",
    "DEFAULT_STORE",
    "LOCK_SCHEMA",
    "OVERLAY_SCHEMA",
    "SOURCE_OBJECT_SCHEMA",
    "UNIMPLEMENTED",
    "BoundedResult",
    "RuntimeConfig",
    "RuntimeLimits",
    "RuntimeRefusal",
    "call_worker",
    "check_overlay_semantics",
    "check_store_location",
    "disk_preflight",
    "generate_overlay",
    "load_lock_ref",
    "load_runtime_config",
    "main",
    "parse_runtime_config",
    "prepare",
    "publish_lock",
    "run_bounded",
    "seal_source",
    "store_conflicts",
    "validate_lock",
    "validate_static_project",
    "verify",
    "verify_source_object",
    "worker_env",
]

CONFIG_SCHEMA = "masy.runtime.v1"
LOCK_SCHEMA = "masy.runtime_lock.v1"
SOURCE_OBJECT_SCHEMA = "masy.source_object.v1"
OVERLAY_SCHEMA = "masy.metadata_overlay.v1"
EVIDENCE_SCHEMA = "masy.runtime_evidence.v1"
OUTPUT_SCHEMA = "masy.runtime_output.v1"
DEFAULT_STORE = "/home/okin/MASY/runs/runtime-store"
LAUNCHER_RELPATH = "tools/masy_runtime.py"
SCOPE_PATHSPECS = ("src", "pyproject.toml")
# name -> (import package, distribution/project name). Fixed; MASY's root is the launcher's own repository.
SOURCES = {"maple": ("maple", "maple"), "syrup": ("maple_syrup", "maple-syrup"), "masy": ("masy", "masy")}
OVERLAY_KINDS = ("syrup", "masy")
ALLOWED_PROJECT_KEYS = frozenset({"name", "version", "description", "requires-python", "dependencies",
                                  "optional-dependencies", "scripts"})
INSTALLER = "masy.runtime"
EXIT_CODES = {"runtime_locked": 0, "runtime_verified": 0, "refused": 2, "incompatible": 3}
EXIT_INTERNAL = 1
MAX_MESSAGE = 4000
MAX_CONFIG_BYTES = 1 << 20
MAX_SMALL_FILE_BYTES = 1 << 20
MAX_BUNDLE_MANIFEST_BYTES = 16 << 20  # upstream dependency_bundle.Limits.max_manifest_bytes default
MAX_DIST_INFO_FILES = 256
MAX_STAGE_LINE_BYTES = 256 << 10
MAX_OUTPUT_EVIDENCE_CHARS = 64 << 10
STRAGGLER_GRACE_S = 2.0
_CHUNK = 1 << 20
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_RUN_ID = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]{0,127}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._+!-]{0,127}$")
_EXTRA_NAME = re.compile(r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$")
_SCRIPT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SCRIPT_TARGET = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")
_REGULAR_MODES = frozenset({"100644", "100755"})
_FLOAT_LIMITS = frozenset({"git_timeout_s", "worker_timeout_s"})
OBJECT_SUBDIRS = ("sources", "maple-bundles", "maple-materialized")

SELECTION = {
    "policy": "current_local_committed_head",
    "development_snapshot": False,
    "scope": "committed src/ and pyproject.toml blobs of each source; MASY also its tools/masy_runtime.py",
    "generated_metadata": "live src/*.egg-info|*.dist-info are recorded in the observation and never copied; run "
                          "metadata comes only from the upstream MAPLE bundle overlay and labelled MASY overlays",
}
UNIMPLEMENTED = (
    "wind_adapter", "water_adapter", "event_schedule_runner", "transition_accounting", "combined_restart",
    "routing_reentry_adapter", "cache_invalidation_adapter", "wind_mobile_completion", "case_input_hashing",
    "windowed_storage_qualification", "gpu_and_venv_cupy_qualification", "performance_qualification",
    "physical_science_qualification",
)
INTERPRETATION = {
    "transport_ready": False,
    "scope": "CPU interpreter runtime identity and native interface delivery only; Dense checkpoint registries are "
             "probed by class identity, never exercised; no transport, restart, science or performance claim",
    "environment": "the existing interpreter environment; third-party wheels and native libraries are identified "
                   "by version and hash, not bundled and not independent of that environment",
    "gpu_qualified": False,
    "unimplemented": list(UNIMPLEMENTED),
}
OPERATIONAL_FIELDS = ("run_dir", "request_path", "pycache_prefix", "pid", *_worker.OPERATIONAL_ENV_KEYS)
PYCACHE_POLICY = "-B plus a run-owned -X pycache_prefix that is never created: no bytecode written or read from " \
                 "source objects; every module compiles from its sealed source"
_ENVIRONMENT_KEYS = frozenset({"interpreter_identity", "numpy", "optional_distribution_versions",
                               "distribution_versions", "compiled", "optional_modules_loaded"})


def _read_generator_identity() -> str:
    """sha256 of this module's bytes: the metadata-overlay generator identity bound into overlay object ids."""
    with open(__file__, "rb") as handle:
        data = handle.read((8 << 20) + 1)
    if len(data) > 8 << 20:
        raise RuntimeError("masy/runtime.py is unexpectedly large")
    return hashlib.sha256(data).hexdigest()


GENERATOR_SHA256 = _read_generator_identity()


class RuntimeRefusal(Exception):
    """A runtime guard failed. `code` is stable; `severity` is refusal or incompatible; evidence is JSON-safe."""

    def __init__(self, code: str, detail: str, *, severity: str = "refusal", **evidence: Any) -> None:
        super().__init__(f"{code}: {detail}")
        if severity not in ("refusal", "incompatible"):
            raise ValueError(f"invalid severity {severity!r}")
        self.code = code
        self.detail = str(detail)[:MAX_MESSAGE]
        self.severity = severity
        self.evidence = evidence


def _from_source_refusal(exc: SourceRefusal, **extra: Any) -> RuntimeRefusal:
    return RuntimeRefusal(exc.code, exc.detail, **extra, **exc.evidence)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    return repr(value)[:MAX_MESSAGE]


def _canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("ascii")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------------------------- limits


@dataclass(frozen=True)
class RuntimeLimits:
    """Every bound is a positive finite number (bool, NaN, infinity, zero and negatives are invalid)."""

    git_timeout_s: float = 30.0
    git_max_output_bytes: int = 8 << 20
    max_files: int = 5_000
    max_bytes: int = 64 << 20
    max_lock_bytes: int = 4 << 20
    max_evidence_bytes: int = 32 << 20
    worker_timeout_s: float = 900.0
    worker_max_output_bytes: int = 8 << 20
    max_compiled_files: int = 4_000
    max_compiled_bytes: int = 4 << 30
    store_max_new_bytes: int = 4 << 30
    disk_reserve_bytes: int = 1 << 30

    def __post_init__(self) -> None:
        for item in fields(self):
            _check_limit(item.name, getattr(self, item.name), f"limits.{item.name}")

    def source_limits(self) -> Limits:
        return Limits(git_timeout_s=self.git_timeout_s, git_max_output_bytes=self.git_max_output_bytes,
                      max_files=self.max_files, max_bytes=self.max_bytes, max_lock_bytes=self.max_lock_bytes)

    def record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, table: Any, where: str = "limits") -> RuntimeLimits:
        if not isinstance(table, dict):
            raise RuntimeRefusal("invalid_config_type", f"{where} must be a table")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(table) - known)
        if unknown:
            raise RuntimeRefusal("unknown_config_field", f"{where} has unknown field(s) {unknown}")
        values = {}
        for name, value in table.items():
            _check_limit(name, value, f"{where}.{name}")
            values[name] = float(value) if name in _FLOAT_LIMITS else value
        return cls(**values)


def _check_limit(name: str, value: Any, where: str) -> None:
    if name in _FLOAT_LIMITS:
        ok = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        if not ok:
            raise RuntimeRefusal("invalid_config_value", f"{where} must be a positive finite number, got {value!r}")
    elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeRefusal("invalid_config_value", f"{where} must be a positive integer, got {value!r}")


# ------------------------------------------------------------------------------------------------- config


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root: str
    package: str
    project_name: str
    expected_revision: str | None


@dataclass(frozen=True)
class RuntimeConfig:
    path: str | None
    sha256: str | None
    maple: SourceSpec
    syrup: SourceSpec
    masy_expected_revision: str | None
    maple_distribution_metadata: str
    python: str
    store: str
    limits: RuntimeLimits

    def record(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "schema": CONFIG_SCHEMA,
                "sources": {"maple": {"root": self.maple.root, "expected_revision": self.maple.expected_revision,
                                      "distribution_metadata": self.maple_distribution_metadata},
                            "syrup": {"root": self.syrup.root, "expected_revision": self.syrup.expected_revision},
                            "masy": {"expected_revision": self.masy_expected_revision}},
                "runtime": {"python": self.python, "store": self.store}, "limits": self.limits.record()}


_TOP_KEYS = {"schema", "sources", "runtime", "limits"}
_SOURCE_KEYS = {"maple": {"root", "distribution_metadata", "expected_revision"},
                "syrup": {"root", "expected_revision"}, "masy": {"expected_revision"}}
_RUNTIME_KEYS = {"python", "store"}


def _table(value: Any, allowed: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeRefusal("invalid_config_type", f"{where} must be a table")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RuntimeRefusal("unknown_config_field", f"{where} has unknown field(s) {unknown}")
    return value


def _path_value(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or _CONTROL.search(value):
        raise RuntimeRefusal("invalid_config_type", f"{where} must be a non-empty string without control characters")
    if not os.path.isabs(value) or os.path.normpath(value) != value:
        raise RuntimeRefusal("invalid_config_value", f"{where} must be an absolute, normalized path")
    return value


def _revision(value: Any, where: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not _OBJECT_ID.match(value)):
        raise RuntimeRefusal("invalid_config_value", f"{where} must be a full 40/64-hex object id")
    return value


def parse_runtime_config(raw: Any, *, path: str | None = None, sha256: str | None = None) -> RuntimeConfig:
    _table(raw, _TOP_KEYS, "<top>")
    if raw.get("schema") != CONFIG_SCHEMA:
        raise RuntimeRefusal("invalid_config_value", f"schema must be {CONFIG_SCHEMA!r}")
    sources = _table(raw.get("sources"), set(_SOURCE_KEYS), "sources")
    specs = {}
    for name in ("maple", "syrup"):
        if name not in sources:
            raise RuntimeRefusal("missing_config_field", f"[sources.{name}] is required")
        table = _table(sources[name], _SOURCE_KEYS[name], f"sources.{name}")
        if "root" not in table:
            raise RuntimeRefusal("missing_config_field", f"sources.{name}.root is required")
        package, project = SOURCES[name]
        specs[name] = SourceSpec(name, _path_value(table["root"], f"sources.{name}.root"), package, project,
                                 _revision(table.get("expected_revision"), f"sources.{name}.expected_revision"))
    if "distribution_metadata" not in sources["maple"]:
        raise RuntimeRefusal("missing_config_field", "sources.maple.distribution_metadata is required")
    dist = _path_value(sources["maple"]["distribution_metadata"], "sources.maple.distribution_metadata")
    masy = _table(sources.get("masy", {}), _SOURCE_KEYS["masy"], "sources.masy")
    runtime = _table(raw.get("runtime"), _RUNTIME_KEYS, "runtime")
    if "python" not in runtime:
        raise RuntimeRefusal("missing_config_field", "runtime.python is required")
    limits = RuntimeLimits.from_mapping(raw.get("limits", {}))
    return RuntimeConfig(
        path=path, sha256=sha256, maple=specs["maple"], syrup=specs["syrup"],
        masy_expected_revision=_revision(masy.get("expected_revision"), "sources.masy.expected_revision"),
        maple_distribution_metadata=dist, python=_path_value(runtime["python"], "runtime.python"),
        store=_path_value(runtime.get("store", DEFAULT_STORE), "runtime.store"), limits=limits,
    )


def load_runtime_config(path: str) -> RuntimeConfig:
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_CONFIG_BYTES + 1)
    except OSError as exc:
        raise RuntimeRefusal("config_unreadable", f"{path}: {exc}") from exc
    if len(data) > MAX_CONFIG_BYTES:
        raise RuntimeRefusal("config_unreadable", f"{path} is larger than {MAX_CONFIG_BYTES} bytes")
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeRefusal("config_unreadable", f"{path}: {exc}") from exc
    return parse_runtime_config(raw, path=os.path.abspath(path), sha256=_sha256(data))


# ------------------------------------------------------------------------------------- location separation


def _forms(path: str) -> set[str]:
    return {os.path.abspath(path), os.path.realpath(path)}


def _contains(outer: str, inner: str) -> bool:
    try:
        return os.path.commonpath([outer, inner]) == outer
    except ValueError:
        return False


def store_conflicts(store: str, *, wrapper_roots: list[str], protected: dict[str, str]) -> list[str]:
    """Protected inputs the store equals, contains or lies inside (lexical and symlink-resolved forms), plus the
    wrapper rule: the store may lie inside a wrapper root only strictly beneath its `runs/` directory, and may
    never equal a wrapper root or be one of its ancestors."""
    conflicts = []
    store_forms = _forms(store)
    for label, path in protected.items():
        if any(_contains(p, s) or _contains(s, p) for p in _forms(path) for s in store_forms):
            conflicts.append(f"{label} ({path})")
    for root in wrapper_roots:
        for w in _forms(root):
            runs = os.path.join(w, "runs")
            for s in store_forms:
                if _contains(s, w):
                    conflicts.append(f"wrapper root {root} (the store equals or contains it)")
                elif _contains(w, s) and (s == runs or not _contains(runs, s)):
                    conflicts.append(f"wrapper root {root} (inside it the store must lie strictly under runs/)")
    return sorted(set(conflicts))


def check_store_path(store: str) -> None:
    if not isinstance(store, str) or not os.path.isabs(store) or os.path.normpath(store) != store:
        raise RuntimeRefusal("store_path_invalid", f"store {store!r} must be an absolute, normalized path")
    if os.path.realpath(store) != store:
        raise RuntimeRefusal("store_not_canonical", f"store {store} resolves through a symlink to "
                             f"{os.path.realpath(store)}; symlinked store components are refused")


def check_store_location(store: str, *, wrapper_roots: list[str], protected: dict[str, str]) -> None:
    check_store_path(store)
    conflicts = store_conflicts(store, wrapper_roots=wrapper_roots, protected=protected)
    if conflicts:
        raise RuntimeRefusal("store_overlaps_input", f"store {store} overlaps protected input(s) {conflicts}",
                             conflicts=conflicts)


def _protected_inputs(*, maple_root: str, syrup_root: str, dist_info: str, python: dict[str, Any],
                      wrapper_root: str) -> dict[str, str]:
    protected = {"sources.maple.root": maple_root, "sources.syrup.root": syrup_root,
                 "sources.maple.distribution_metadata": dist_info,
                 "runtime.python (resolved executable)": python["realpath"]}
    if python["venv"] is not None:
        protected["runtime.python environment"] = python["venv"]["dir"]
    for relative in ("src", "tools", "pyproject.toml", ".git"):
        protected[f"wrapper {relative}"] = os.path.join(wrapper_root, relative)
    return protected


# ------------------------------------------------------------------------------------------ file utilities


def _lstat(path: Path | str, code: str) -> os.stat_result:
    try:
        return os.lstat(path)
    except FileNotFoundError as exc:
        raise RuntimeRefusal(code, f"{path} does not exist") from exc
    except OSError as exc:
        raise RuntimeRefusal("unreadable_file", f"cannot stat {path}: {exc}") from exc


def _require_real_dir(path: Path | str, code: str = "store_component_invalid") -> None:
    info = _lstat(path, code)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeRefusal(code, f"{path} is not a real directory (symlinks are refused)")


_IDENTITY_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return tuple(getattr(info, name) for name in _IDENTITY_FIELDS)


def _hash_path(path: Path | str, *, budget: int, blob_algorithm: str | None = None, keep: int | bool = 0,
               ) -> tuple[str, str | None, int, bytes | None]:
    """sha256, optional Git blob id, size and (when the file is at most `keep` bytes; `keep=True` means
    MAX_SMALL_FILE_BYTES) the bytes of one regular file opened without following symlinks.

    The path lstat before open, the descriptor fstat after open, the descriptor fstat after the last read and the
    path lstat after the read must all carry the same device, inode, mode, size, mtime_ns and ctime_ns, and the bytes
    read must equal that size: a replaced path or a same-size in-place write during the read refuses as
    `source_changed_during_read` instead of returning a digest of bytes the file no longer holds."""
    if keep is True:
        keep = MAX_SMALL_FILE_BYTES
    initial = _lstat(path, "unreadable_file")
    if stat.S_ISLNK(initial.st_mode):
        raise RuntimeRefusal("symlink_in_scope", f"{path} is a symlink")
    if not stat.S_ISREG(initial.st_mode):
        raise RuntimeRefusal("non_regular_file", f"{path} is not a regular file")
    if initial.st_size > budget:
        raise RuntimeRefusal("max_bytes_exceeded", f"{path} ({initial.st_size} bytes) exceeds the remaining byte "
                             f"budget {budget}")
    identity = _file_identity(initial)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError as exc:
        raise RuntimeRefusal("source_changed_during_read", f"{path} disappeared before it was opened") from exc
    except OSError as exc:
        code = "symlink_in_scope" if exc.errno == errno.ELOOP else "unreadable_file"
        raise RuntimeRefusal(code, f"cannot open {path}: {exc}") from exc
    try:
        if _file_identity(os.fstat(fd)) != identity:
            raise RuntimeRefusal("source_changed_during_read", f"{path} changed between lstat and open")
        size = initial.st_size
        sha = hashlib.sha256()
        blob = hashlib.new(blob_algorithm) if blob_algorithm else None
        if blob is not None:
            blob.update(b"blob %d\0" % size)
        kept = [] if keep and size <= keep else None
        read = 0
        while True:
            chunk = os.read(fd, min(_CHUNK, size + 1 - read))
            if not chunk:
                break
            read += len(chunk)
            if read > size:
                raise RuntimeRefusal("source_changed_during_read", f"{path} grew while it was read")
            sha.update(chunk)
            if blob is not None:
                blob.update(chunk)
            if kept is not None:
                kept.append(chunk)
        if read != size or _file_identity(os.fstat(fd)) != identity:
            raise RuntimeRefusal("source_changed_during_read", f"{path} was modified while it was read")
    finally:
        os.close(fd)
    try:
        final = os.lstat(path)
    except OSError as exc:
        raise RuntimeRefusal("source_changed_during_read", f"{path} is missing or unreadable after it was read: "
                             f"{exc}") from exc
    if _file_identity(final) != identity:
        raise RuntimeRefusal("source_changed_during_read", f"{path} was replaced or modified while it was read")
    return sha.hexdigest(), (blob.hexdigest() if blob else None), read, (b"".join(kept) if kept is not None else None)


def _read_bounded(path: Path | str, max_bytes: int, code: str) -> bytes:
    info = _lstat(path, code)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeRefusal(code, f"{path} is not a regular file (symlinks are refused)")
    if info.st_size > max_bytes:
        raise RuntimeRefusal(code, f"{path} is {info.st_size} bytes, above its bound {max_bytes}")
    try:
        _sha, _blob, _size, data = _hash_path(path, budget=max_bytes, keep=max_bytes)
    except RuntimeRefusal as exc:
        raise RuntimeRefusal(code, exc.detail) from exc
    assert data is not None  # the size was bounded by max_bytes = keep
    return data


def _strict_json(data: bytes, code: str, what: str) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate key {key!r}")
            out[key] = value
        return out

    def constant(name):
        raise ValueError(f"non-finite number {name}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeRefusal(code, f"{what} is not strict UTF-8 JSON: {exc}") from exc


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(errno.EIO, "write made no progress")
        view = view[written:]


def _write_new(path: Path, data: bytes, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path | str) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _freeze(root: Path) -> None:
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            os.chmod(os.path.join(dirpath, name), 0o444)  # staging holds only files this process created
        os.chmod(dirpath, 0o555)


def _rename_noreplace(src: Path, dst: Path) -> None:
    """Atomic Linux renameat2(RENAME_NOREPLACE); FileExistsError if `dst` exists in any form."""
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise RuntimeRefusal("atomic_publication_unavailable", "renameat2 is unavailable; refusing to publish")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    at_fdcwd, rename_noreplace = -100, 1
    if function(at_fdcwd, os.fsencode(str(src)), at_fdcwd, os.fsencode(str(dst)), rename_noreplace) != 0:
        err = ctypes.get_errno()
        if err == errno.EEXIST:
            raise FileExistsError(err, os.strerror(err), str(dst))
        if err in (errno.ENOSYS, errno.EINVAL):
            raise RuntimeRefusal("atomic_publication_unavailable", f"renameat2 NOREPLACE failed: {os.strerror(err)}")
        raise OSError(err, os.strerror(err), str(dst))


def _tree_digest(entries: list[tuple[str, str]]) -> str:
    """Same algorithm as `sourceobs.DIGEST_ALGORITHM` / `maple_syrup.provenance.SOURCE_DIGEST_ALGORITHM`."""
    hasher = hashlib.sha256()
    for relative, digest in sorted(entries):
        hasher.update(relative.encode("utf-8", "surrogateescape") + b"\0" + digest.encode("ascii") + b"\n")
    return hasher.hexdigest()


def _package_digest(files: list[list[Any]], package: str) -> str:
    prefix = f"src/{package}/"
    return _tree_digest([(path[len(prefix):], sha) for path, _size, sha, _blob in files if path.startswith(prefix)])


# --------------------------------------------------------------------------------------- interpreter & dist


def interpreter_identity(python: str, limits: RuntimeLimits) -> dict[str, Any]:
    """The configured spelling is kept (a venv `bin/python` symlink keeps its environment); only that final
    component may be a symlink, explicitly recorded. Realpath bytes are hashed in bounded chunks."""
    _path_value(python, "runtime.python")
    parent = os.path.dirname(python)
    if os.path.realpath(parent) != parent:
        raise RuntimeRefusal("interpreter_path_invalid", f"{parent} resolves through a symlink; only the "
                             "interpreter's own final path component may be a symlink")
    info = _lstat(python, "interpreter_missing")
    is_link = stat.S_ISLNK(info.st_mode)
    if not is_link and not stat.S_ISREG(info.st_mode):
        raise RuntimeRefusal("interpreter_path_invalid", f"{python} is neither a regular file nor a symlink")
    real = os.path.realpath(python)
    if not os.path.isfile(real) or not os.access(real, os.X_OK):
        raise RuntimeRefusal("interpreter_missing", f"{python} resolves to {real}, not an executable file")
    sha, _blob, size, _data = _hash_path(real, budget=limits.max_compiled_bytes)
    venv_dir = os.path.dirname(parent)
    cfg = os.path.join(venv_dir, "pyvenv.cfg")
    venv = None
    if os.path.lexists(cfg):
        venv = {"dir": venv_dir, "pyvenv_cfg_sha256": _sha256(_read_bounded(cfg, MAX_SMALL_FILE_BYTES,
                                                                            "interpreter_path_invalid"))}
        if os.path.realpath(venv_dir) != venv_dir:
            raise RuntimeRefusal("interpreter_path_invalid", f"environment {venv_dir} is not canonical")
    return {"spelling": python, "is_symlink": is_link, "link_target": os.readlink(python) if is_link else None,
            "realpath": real, "size": size, "sha256": sha, "venv": venv,
            "scope": "CPU interpreter; no GPU or .venv-cupy qualification"}


def distribution_metadata_record(path: str) -> dict[str, Any]:
    """sha256 of every file in the installed MAPLE dist-info directory (bounded; symlinks/special files refuse)."""
    _path_value(path, "sources.maple.distribution_metadata")
    if os.path.realpath(path) != path:
        raise RuntimeRefusal("distribution_metadata_invalid", f"{path} resolves through a symlink")
    _require_real_dir(path, "distribution_metadata_invalid")
    files: dict[str, dict[str, Any]] = {}
    pending = [Path(path)]
    while pending:
        current = pending.pop()
        for entry in sorted(os.listdir(current)):
            full = current / entry
            info = _lstat(full, "distribution_metadata_invalid")
            if stat.S_ISDIR(info.st_mode):
                pending.append(full)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeRefusal("distribution_metadata_invalid", f"{full} is a symlink or special file")
            if len(files) >= MAX_DIST_INFO_FILES:
                raise RuntimeRefusal("distribution_metadata_invalid", f"{path} holds more than "
                                     f"{MAX_DIST_INFO_FILES} files")
            sha, _blob, size, _data = _hash_path(full, budget=MAX_SMALL_FILE_BYTES)
            files[full.relative_to(path).as_posix()] = {"size": size, "sha256": sha}
    return {"path": path, "files": dict(sorted(files.items()))}


# ---------------------------------------------------------------------------------------------- git & scope


def _git(root: str, args: list[str], limits: RuntimeLimits) -> bytes:
    try:
        return run_git(root, args, limits.source_limits())
    except SourceRefusal as exc:
        raise _from_source_refusal(exc) from exc


def head_tree(root: str, head: str, limits: RuntimeLimits) -> str:
    tree = _git(root, ["rev-parse", "--verify", f"{head}^{{tree}}"], limits).decode("ascii", "replace").strip()
    if not _OBJECT_ID.match(tree):
        raise RuntimeRefusal("head_unreadable", f"{head}^{{tree}} did not resolve to an object id")
    return tree


def ls_scope(root: str, head: str, pathspecs: list[str], limits: RuntimeLimits) -> list[tuple[str, str, str]]:
    """(path, mode, blob id) of the committed regular files of `head` under `pathspecs`."""
    out = _git(root, ["ls-tree", "-r", "-z", "--full-tree", head, "--", *pathspecs], limits)
    entries = []
    for record in out.decode("utf-8", "surrogateescape").split("\0"):
        if not record:
            continue
        meta, sep, path = record.partition("\t")
        parts = meta.split(" ")
        if not sep or len(parts) != 3:
            raise RuntimeRefusal("git_output_unparseable", f"unexpected ls-tree record {record[:200]!r}")
        mode, kind, oid = parts
        if mode == "120000":
            raise RuntimeRefusal("symlink_in_scope", f"committed {path} is a symlink")
        if kind != "blob" or mode not in _REGULAR_MODES or not _OBJECT_ID.match(oid):
            raise RuntimeRefusal("unsupported_tracked_entry", f"committed {path} is {kind} mode {mode}")
        entries.append((path, mode, oid))
    if len(entries) > limits.max_files + 1:
        raise RuntimeRefusal("max_files_exceeded", f"{root}: more than max_files={limits.max_files} files in scope")
    return sorted(entries)


def _check_relpath(path: str, *, launcher_allowed: bool) -> str:
    parts = path.split("/") if isinstance(path, str) else []
    if not isinstance(path, str) or not path or _CONTROL.search(path) or "\\" in path or path.startswith("/") or \
            len(path.encode("utf-8", "surrogateescape")) > 1024 or any(p in ("", ".", "..") for p in parts):
        raise RuntimeRefusal("unsupported_tracked_entry", f"unsafe scope path {path!r}")
    if ".git" in parts or "__pycache__" in parts or path.endswith((".pyc", ".pyo")):
        raise RuntimeRefusal("unsupported_tracked_entry", f"{path!r}: VCS metadata or bytecode is not a source")
    if not (path == "pyproject.toml" or path.startswith("src/") or (launcher_allowed and path == LAUNCHER_RELPATH)):
        raise RuntimeRefusal("unsupported_tracked_entry", f"{path!r} is outside the sealed scope")
    return path


def observe_launcher(root: str, head: str, object_format: str, launcher: str, limits: RuntimeLimits
                     ) -> dict[str, Any]:
    """The running MASY launcher must be this repository's committed, unmodified tools/masy_runtime.py."""
    expected = os.path.join(root, LAUNCHER_RELPATH)
    if os.path.realpath(launcher) != expected:
        raise RuntimeRefusal("launcher_outside_wrapper_root", f"launcher {launcher} is not {expected}")
    _require_real_dir(os.path.join(root, "tools"), "symlink_in_scope")
    status = _git(root, ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none",
                         "--", LAUNCHER_RELPATH], limits)
    if status:
        raise RuntimeRefusal("dirty_scope", f"{LAUNCHER_RELPATH} has uncommitted changes", source="masy")
    entries = ls_scope(root, head, [LAUNCHER_RELPATH], limits)
    if [e[0] for e in entries] != [LAUNCHER_RELPATH]:
        raise RuntimeRefusal("launcher_not_committed", f"{LAUNCHER_RELPATH} is not a committed regular file")
    sha, blob, size, _data = _hash_path(expected, budget=MAX_SMALL_FILE_BYTES, blob_algorithm=object_format)
    if blob != entries[0][2]:
        raise RuntimeRefusal("content_differs_from_head", f"{LAUNCHER_RELPATH} differs from HEAD", source="masy")
    return {"path": LAUNCHER_RELPATH, "git_blob": blob, "sha256": sha, "size": size, "mode": entries[0][1]}


def observe_source(name: str, root: str, *, expected_revision: str | None, limits: RuntimeLimits
                   ) -> dict[str, Any]:
    package, project = SOURCES[name]
    try:
        observation = observe_upstream(name, root, package=package, project_name=project,
                                       expected_revision=expected_revision, limits=limits.source_limits())
    except SourceRefusal as exc:
        raise _from_source_refusal(exc, source=name) from exc
    observation["tree"] = head_tree(observation["root"], observation["head"], limits)
    return observation


def observe_selection(config: RuntimeConfig, *, masy_root: str, launcher: str) -> dict[str, Any]:
    limits = config.limits
    observation = {
        "maple": observe_source("maple", config.maple.root, expected_revision=config.maple.expected_revision,
                                limits=limits),
        "syrup": observe_source("syrup", config.syrup.root, expected_revision=config.syrup.expected_revision,
                                limits=limits),
        "masy": observe_source("masy", masy_root, expected_revision=config.masy_expected_revision, limits=limits),
    }
    masy = observation["masy"]
    masy["launcher"] = observe_launcher(masy["root"], masy["head"], masy["object_format"], launcher, limits)
    observation["maple"]["installed_distribution_metadata"] = \
        distribution_metadata_record(config.maple_distribution_metadata)
    return observation


def check_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    differences = diff_observations(before, after)
    if differences:
        raise RuntimeRefusal("source_changed_during_prepare", f"before/after observations differ at {differences}",
                             differences=differences)


def hash_scope(observation: dict[str, Any], limits: RuntimeLimits, *,
               _after_hash: Callable[[], None] | None = None) -> tuple[list[list[Any]], dict[str, bytes]]:
    """Read-only pass over exactly the committed scope of `observation["head"]`: [[path, size, sha256, blob]] of
    the bytes on disk (each blob must equal the commit's), cross-checked with the observation digests. Small
    metadata inputs (pyproject.toml, src/<pkg>/__init__.py) are captured for semantic checks."""
    name, root, head = observation["name"], observation["root"], observation["head"]
    object_format = observation["object_format"]
    package = SOURCES[name][0]
    pathspecs = list(SCOPE_PATHSPECS) + ([LAUNCHER_RELPATH] if name == "masy" else [])
    entries = ls_scope(root, head, pathspecs, limits)
    keep = {"pyproject.toml", f"src/{package}/__init__.py"}
    files, captured, checked_dirs = [], {}, set()
    remaining = limits.max_bytes + MAX_SMALL_FILE_BYTES  # observed scope bound plus the small launcher
    for path, _mode, oid in entries:
        _check_relpath(path, launcher_allowed=name == "masy")
        parent = os.path.dirname(path)
        while parent and parent not in checked_dirs:
            _require_real_dir(os.path.join(root, parent), "symlink_in_scope")
            checked_dirs.add(parent)
            parent = os.path.dirname(parent)
        sha, blob, size, data = _hash_path(os.path.join(root, path), budget=remaining, blob_algorithm=object_format,
                                           keep=path in keep)
        remaining -= size
        if blob != oid:
            raise RuntimeRefusal("content_differs_from_head", f"{path} differs from commit {head}", source=name)
        files.append([path, size, sha, blob])
        if data is not None:
            captured[path] = data
    if _after_hash is not None:
        _after_hash()
    scoped = [(path, sha) for path, _size, sha, _blob in files if path != LAUNCHER_RELPATH]
    tracked = observation["tracked_scope"]
    if len(scoped) != tracked["file_count"] or _tree_digest(scoped) != tracked["digest_sha256"] or \
            _package_digest(files, package) != observation["package"]["digest_sha256"]:
        raise RuntimeRefusal("source_changed_during_prepare", f"{name}: sealed scope disagrees with the observation",
                             source=name)
    if set(keep) - set(captured):
        raise RuntimeRefusal("package_not_committed", f"{name}: {sorted(set(keep) - set(captured))} missing")
    return files, captured


# --------------------------------------------------------------------------------------- static metadata


def _packaging():
    try:
        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.utils import canonicalize_name
        from packaging.version import InvalidVersion, Version
    except ImportError as exc:
        raise RuntimeRefusal("packaging_unavailable", "the `packaging` module of the existing interpreter is required "
                             "to validate static metadata") from exc
    return Requirement, InvalidRequirement, SpecifierSet, InvalidSpecifier, Version, InvalidVersion, canonicalize_name


def _project_table(data: bytes, where: str) -> dict[str, Any]:
    try:
        project = tomllib.loads(data.decode("utf-8")).get("project")
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeRefusal("pyproject_unreadable", f"{where}: {exc}") from exc
    if not isinstance(project, dict):
        raise RuntimeRefusal("pyproject_unreadable", f"{where}: [project] is missing or not a table")
    return project


def project_identity(data: bytes, expected_name: str, where: str) -> dict[str, Any]:
    project = _project_table(data, where)
    if project.get("name") != expected_name or not isinstance(project.get("version"), str):
        raise RuntimeRefusal("pyproject_project_mismatch", f"{where} must name project {expected_name!r} with a "
                             f"version (got {project.get('name')!r} {project.get('version')!r})")
    return {"name": project["name"], "version": project["version"], "pyproject_sha256": _sha256(data)}


def validate_static_project(data: bytes, expected_name: str, where: str = "pyproject.toml") -> dict[str, Any]:
    """Closed static [project] table for a generated overlay: only name, version, description, requires-python,
    dependencies, optional-dependencies and scripts; `dynamic` and every other key refuse; the version must be
    canonical PEP 440. Returns the normalized values."""
    Requirement, InvalidRequirement, SpecifierSet, InvalidSpecifier, Version, InvalidVersion, canon = _packaging()
    project = _project_table(data, where)
    if "dynamic" in project:
        raise RuntimeRefusal("pyproject_dynamic_refused", f"{where}: dynamic metadata {project['dynamic']!r} cannot "
                             "be generated statically")
    unknown = sorted(set(project) - ALLOWED_PROJECT_KEYS)
    if unknown:
        raise RuntimeRefusal("pyproject_key_unsupported", f"{where}: [project] keys {unknown} are not supported by "
                             "the static overlay generator")
    if project.get("name") != expected_name:
        raise RuntimeRefusal("pyproject_project_mismatch", f"{where} names {project.get('name')!r}, expected "
                             f"{expected_name!r}")
    version = project.get("version")
    try:
        canonical = str(Version(version)) if isinstance(version, str) else None
    except InvalidVersion:
        canonical = None
    if canonical != version or not _SAFE_COMPONENT.match(version):
        raise RuntimeRefusal("pyproject_version_invalid", f"{where}: version {version!r} is not a canonical PEP 440 "
                             "version")

    def text(key: str) -> str | None:
        value = project.get(key)
        if value is not None and (not isinstance(value, str) or "\n" in value or "\r" in value or not value.strip()):
            raise RuntimeRefusal("pyproject_value_invalid", f"{where}: {key} must be a one-line string")
        return value

    def requirements(values: Any, key: str) -> list[str]:
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise RuntimeRefusal("pyproject_value_invalid", f"{where}: {key} must be a list of strings")
        try:
            return [str(Requirement(v)) for v in values]
        except InvalidRequirement as exc:
            raise RuntimeRefusal("pyproject_value_invalid", f"{where}: {key}: {exc}") from exc

    requires_python = text("requires-python")
    if requires_python is not None:
        try:
            SpecifierSet(requires_python)
        except InvalidSpecifier as exc:
            raise RuntimeRefusal("pyproject_value_invalid", f"{where}: requires-python: {exc}") from exc
    optional_raw = project.get("optional-dependencies", {})
    if not isinstance(optional_raw, dict):
        raise RuntimeRefusal("pyproject_value_invalid", f"{where}: optional-dependencies must be a table")
    optional = {}
    for extra, values in optional_raw.items():
        if not _EXTRA_NAME.match(extra) or canon(extra) != extra:
            raise RuntimeRefusal("pyproject_value_invalid", f"{where}: extra name {extra!r} is not normalized")
        optional[extra] = requirements(values, f"optional-dependencies.{extra}")
    scripts = project.get("scripts", {})
    if not isinstance(scripts, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and _SCRIPT_NAME.match(k) and _SCRIPT_TARGET.match(v)
            for k, v in scripts.items()):
        raise RuntimeRefusal("pyproject_value_invalid", f"{where}: scripts must map names to module:attribute")
    return {"name": project["name"], "version": version, "description": text("description"),
            "requires_python": requires_python,
            "dependencies": requirements(project.get("dependencies", []), "dependencies"),
            "optional_dependencies": dict(sorted(optional.items())), "scripts": dict(sorted(scripts.items())),
            "pyproject_sha256": _sha256(data)}


def module_version(data: bytes, where: str) -> str:
    """The single top-level `__version__ = "<literal>"` of a module, read from its AST (never imported)."""
    try:
        tree = ast.parse(data.decode("utf-8"), filename=where)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise RuntimeRefusal("version_alignment_unverifiable", f"{where}: {exc}") from exc
    values = [node.value.value for node in tree.body
              if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id == "__version__" and isinstance(node.value, ast.Constant)
              and isinstance(node.value.value, str)]
    if len(values) != 1:
        raise RuntimeRefusal("version_alignment_unverifiable", f"{where} must assign one literal __version__")
    return values[0]


def check_version_alignment(project: dict[str, Any], init_bytes: bytes, where: str) -> None:
    found = module_version(init_bytes, where)
    if found != project["version"]:
        raise RuntimeRefusal("version_mismatch", f"{where} __version__ {found!r} != pyproject version "
                             f"{project['version']!r}")


def dist_info_name(name: str, version: str) -> str:
    return f"{re.sub(r'[-_.]+', '_', name).lower()}-{version}.dist-info"


def _extra_requirement(value: str, extra: str) -> str:
    Requirement = _packaging()[0]
    requirement = Requirement(value)
    marker = requirement.marker
    requirement.marker = None
    if marker is None:
        return f'{requirement}; extra == "{extra}"'
    return f'{requirement}; ({marker}) and extra == "{extra}"'


def _overlay_label(project: dict[str, Any], *, object_id: str, generator_sha256: str) -> dict[str, Any]:
    return {"kind": "run_owned_metadata_overlay", "not_an_installation": True, "schema": OVERLAY_SCHEMA,
            "generator_sha256": generator_sha256, "source_object": object_id, "distribution": project["name"],
            "version": project["version"], "pyproject_sha256": project["pyproject_sha256"],
            "derived_from": "the archived static pyproject.toml of this source object"}


def generate_overlay(project: dict[str, Any], *, package: str, object_dir: Path | str, object_id: str,
                     generator_sha256: str = GENERATOR_SHA256) -> dict[str, bytes]:
    """Byte-deterministic dist-info overlay (relative to `<object>/editable`) derived only from the validated
    static project table; bound to the object's final path and id. Not an installation; nothing is executed."""
    info = dist_info_name(project["name"], project["version"])
    lines = ["Metadata-Version: 2.1", f"Name: {project['name']}", f"Version: {project['version']}"]
    if project["description"] is not None:
        lines.append(f"Summary: {project['description']}")
    if project["requires_python"] is not None:
        lines.append(f"Requires-Python: {project['requires_python']}")
    lines += [f"Requires-Dist: {dep}" for dep in project["dependencies"]]
    for extra, values in project["optional_dependencies"].items():
        lines.append(f"Provides-Extra: {extra}")
        lines += [f"Requires-Dist: {_extra_requirement(value, extra)}" for value in values]
    files = {
        f"{info}/METADATA": ("\n".join(lines) + "\n").encode("utf-8"),
        f"{info}/top_level.txt": f"{package}\n".encode("ascii"),
        f"{info}/INSTALLER": f"{INSTALLER}\n".encode("ascii"),
        f"{info}/direct_url.json": json.dumps(
            {"dir_info": {"editable": True}, "url": "file://" + quote(str(Path(object_dir) / "source"))},
            sort_keys=True).encode("ascii"),
        f"{info}/MASY_OVERLAY.json": _canonical_bytes(_overlay_label(project, object_id=object_id,
                                                                     generator_sha256=generator_sha256)),
    }
    if project["scripts"]:
        files[f"{info}/entry_points.txt"] = ("[console_scripts]\n" + "".join(
            f"{name} = {target}\n" for name, target in project["scripts"].items())).encode("ascii")
    record = []
    for rel in sorted(files):
        digest = base64.urlsafe_b64encode(hashlib.sha256(files[rel]).digest()).rstrip(b"=").decode()
        record.append(f"{rel},sha256={digest},{len(files[rel])}")
    record.append(f"{info}/RECORD,,")
    files[f"{info}/RECORD"] = ("\n".join(record) + "\n").encode("ascii")
    return dict(sorted(files.items()))


def check_overlay_semantics(files: dict[str, bytes], project: dict[str, Any], *, package: str,
                            object_dir: Path | str, object_id: str, generator_sha256: str) -> None:
    """Parse the ACTUAL overlay bytes and require them to state exactly the archived static metadata, the object's
    final location and its label; independent of the generator's byte layout."""
    Requirement = _packaging()[0]
    info = dist_info_name(project["name"], project["version"])
    required = {"METADATA", "top_level.txt", "INSTALLER", "direct_url.json", "MASY_OVERLAY.json", "RECORD"}
    if project["scripts"]:
        required.add("entry_points.txt")
    if set(files) != {f"{info}/{name}" for name in required}:
        raise RuntimeRefusal("overlay_invalid", f"overlay files {sorted(files)} are not exactly {sorted(required)} "
                             f"under {info}")

    def text(name: str) -> str:
        try:
            return files[f"{info}/{name}"].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeRefusal("overlay_invalid", f"{name} is not UTF-8") from exc

    meta = email.parser.Parser().parsestr(text("METADATA"))
    allowed = {"Metadata-Version", "Name", "Version", "Summary", "Requires-Python", "Requires-Dist", "Provides-Extra"}
    names = [key for key, _value in meta.items()]
    if set(names) - allowed or meta.get_payload() or any(
            names.count(key) > 1 for key in allowed - {"Requires-Dist", "Provides-Extra"}):
        raise RuntimeRefusal("overlay_invalid", f"METADATA has unexpected or repeated fields {names}")
    expected = {"Metadata-Version": "2.1", "Name": project["name"], "Version": project["version"],
                "Summary": project["description"], "Requires-Python": project["requires_python"]}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeRefusal("overlay_invalid", f"METADATA {key} {meta.get(key)!r} != {value!r}")
    try:
        got = {str(Requirement(v)) for v in meta.get_all("Requires-Dist") or []}
        want = {str(Requirement(v)) for v in project["dependencies"]} | {
            str(Requirement(_extra_requirement(v, extra)))
            for extra, values in project["optional_dependencies"].items() for v in values}
    except Exception as exc:  # noqa: BLE001 - InvalidRequirement in tampered bytes
        raise RuntimeRefusal("overlay_invalid", f"METADATA Requires-Dist unparseable: {exc}") from exc
    if got != want or set(meta.get_all("Provides-Extra") or []) != set(project["optional_dependencies"]):
        raise RuntimeRefusal("overlay_invalid", "METADATA Requires-Dist/Provides-Extra differ from the pyproject")
    if text("top_level.txt").split() != [package] or text("INSTALLER") != f"{INSTALLER}\n":
        raise RuntimeRefusal("overlay_invalid", "top_level.txt or INSTALLER differ from the expected values")
    direct = _strict_json(files[f"{info}/direct_url.json"], "overlay_invalid", "direct_url.json")
    if direct != {"dir_info": {"editable": True}, "url": "file://" + quote(str(Path(object_dir) / "source"))}:
        raise RuntimeRefusal("overlay_invalid", f"direct_url.json {direct!r} is not bound to {object_dir}/source")
    label = _strict_json(files[f"{info}/MASY_OVERLAY.json"], "overlay_invalid", "MASY_OVERLAY.json")
    if label != _overlay_label(project, object_id=object_id, generator_sha256=generator_sha256):
        raise RuntimeRefusal("overlay_invalid", "MASY_OVERLAY.json label differs (it must say not_an_installation "
                             "and bind this object, generator and pyproject)")
    if project["scripts"]:
        parser = configparser.ConfigParser(delimiters=("=",), interpolation=None)
        parser.optionxform = str  # type: ignore[assignment,method-assign]
        try:
            parser.read_string(text("entry_points.txt"))
        except configparser.Error as exc:
            raise RuntimeRefusal("overlay_invalid", f"entry_points.txt unparseable: {exc}") from exc
        if parser.sections() != ["console_scripts"] or dict(parser["console_scripts"]) != project["scripts"]:
            raise RuntimeRefusal("overlay_invalid", "entry_points.txt differs from [project.scripts]")
    rows = text("RECORD").splitlines()
    record = {}
    for row in rows:
        rel, _sep, rest = row.partition(",")
        record[rel] = rest
    if len(record) != len(rows) or set(record) != set(files) or record.pop(f"{info}/RECORD") != ",":
        raise RuntimeRefusal("overlay_invalid", "RECORD does not list exactly the overlay files")
    for rel, rest in record.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(files[rel]).digest()).rstrip(b"=").decode()
        if rest != f"sha256={digest},{len(files[rel])}":
            raise RuntimeRefusal("overlay_invalid", f"RECORD entry for {rel} differs from its bytes")


# ------------------------------------------------------------------------------------------- source objects


def object_identity(kind: str, *, object_format: str, files: list[list[Any]], project: dict[str, Any],
                    generator_sha256: str = GENERATOR_SHA256) -> dict[str, Any]:
    """The hashed identity of a source object: source bytes (path, size, sha256, Git blob), object format, project
    identity and, for overlay kinds, the overlay schema AND generator identity, so changed generator code can
    never collide with an older identical-source object. No path or revision is hashed (no self-reference)."""
    package, distribution = SOURCES[kind]
    overlay = None
    if kind in OVERLAY_KINDS:
        overlay = {"schema": OVERLAY_SCHEMA, "generator_sha256": generator_sha256, "distribution": distribution,
                   "dist_info": dist_info_name(distribution, project["version"])}
    return {"schema": SOURCE_OBJECT_SCHEMA, "kind": kind, "object_format": object_format, "package": package,
            "project": {"name": project["name"], "version": project["version"],
                        "pyproject_sha256": project["pyproject_sha256"]},
            "files": sorted([list(item) for item in files]), "overlay": overlay}


def identity_id(identity: dict[str, Any]) -> str:
    return _sha256(canonical_json(identity).encode("ascii"))


def _validate_identity(identity: Any) -> dict[str, Any]:
    keys = {"schema", "kind", "object_format", "package", "project", "files", "overlay"}
    if not isinstance(identity, dict) or set(identity) != keys or identity["schema"] != SOURCE_OBJECT_SCHEMA or \
            identity["kind"] not in SOURCES or identity["object_format"] not in ("sha1", "sha256") or \
            identity["package"] != SOURCES[identity["kind"]][0]:
        raise RuntimeRefusal("object_tamper", "object identity has an invalid schema/kind/format/package")
    kind = identity["kind"]
    project = identity["project"]
    if not isinstance(project, dict) or set(project) != {"name", "version", "pyproject_sha256"} or \
            project["name"] != SOURCES[kind][1] or not isinstance(project["version"], str) or \
            not isinstance(project["pyproject_sha256"], str) or not _HEX64.match(project["pyproject_sha256"]):
        raise RuntimeRefusal("object_tamper", "object identity project record is malformed")
    files = identity["files"]
    if not isinstance(files, list) or not files:
        raise RuntimeRefusal("object_tamper", "object identity has no files")
    previous = None
    for item in files:
        if not (isinstance(item, list) and len(item) == 4 and isinstance(item[0], str) and isinstance(item[1], int)
                and not isinstance(item[1], bool) and item[1] >= 0 and isinstance(item[2], str)
                and _HEX64.match(item[2]) and isinstance(item[3], str) and _OBJECT_ID.match(item[3])):
            raise RuntimeRefusal("object_tamper", f"object identity file entry {item!r} is malformed")
        _check_relpath(item[0], launcher_allowed=kind == "masy")
        if previous is not None and item[0] <= previous:
            raise RuntimeRefusal("object_tamper", "object identity files are not strictly sorted")
        previous = item[0]
    paths = {item[0] for item in files}
    if "pyproject.toml" not in paths or f"src/{identity['package']}/__init__.py" not in paths or \
            (kind == "masy") != (LAUNCHER_RELPATH in paths):
        raise RuntimeRefusal("object_tamper", "object identity lacks its pyproject/package/launcher entries")
    overlay = identity["overlay"]
    if (kind in OVERLAY_KINDS) != (overlay is not None):
        raise RuntimeRefusal("object_tamper", "object identity overlay presence does not match its kind")
    if overlay is not None and (
            not isinstance(overlay, dict) or set(overlay) != {"schema", "generator_sha256", "distribution",
                                                               "dist_info"}
            or overlay["schema"] != OVERLAY_SCHEMA or overlay["distribution"] != SOURCES[kind][1]
            or not isinstance(overlay["generator_sha256"], str) or not _HEX64.match(overlay["generator_sha256"])
            or overlay["dist_info"] != dist_info_name(SOURCES[kind][1], project["version"])):
        raise RuntimeRefusal("object_tamper", "object identity overlay record is malformed")
    return identity


def _walk_object(location: Path, max_entries: int) -> tuple[dict[str, os.stat_result], set[str]]:
    files: dict[str, os.stat_result] = {}
    dirs: set[str] = set()
    pending = [location]
    while pending:
        current = pending.pop()
        try:
            names = sorted(os.listdir(current))
        except OSError as exc:
            raise RuntimeRefusal("object_tamper", f"cannot list {current}: {exc}") from exc
        for name in names:
            full = current / name
            info = os.lstat(full)
            rel = full.relative_to(location).as_posix()
            if stat.S_ISLNK(info.st_mode):
                raise RuntimeRefusal("object_tamper", f"{full} is a symlink inside a sealed object")
            if info.st_mode & 0o222:
                raise RuntimeRefusal("object_tamper", f"{full} is writable; sealed objects are read-only")
            if stat.S_ISDIR(info.st_mode):
                dirs.add(rel)
                pending.append(full)
            elif stat.S_ISREG(info.st_mode):
                files[rel] = info
            else:
                raise RuntimeRefusal("object_tamper", f"{full} is a special file inside a sealed object")
            if len(files) + len(dirs) > max_entries:
                raise RuntimeRefusal("object_tamper", f"{location} holds more entries than its manifest allows")
    return files, dirs


def verify_source_object(location: Path | str, expected_id: str, limits: RuntimeLimits, *,
                         identity_dir: Path | str | None = None,
                         current_generator: str = GENERATOR_SHA256) -> dict[str, Any]:
    """Full read-only verification of one source object at `location` whose final identity is `identity_dir`
    (default: `location`): canonical manifest, id = sha256(identity) = directory name, exact file/dir set,
    read-only modes, every byte (sha256 and Git blob), static project semantics, `__version__` alignment and the
    overlay's parsed semantics; exact overlay bytes are additionally regenerated when the recorded generator is
    this one. Never repairs, rewrites or deletes anything."""
    location = Path(location)
    identity_dir = Path(identity_dir) if identity_dir is not None else location
    _require_real_dir(location, "object_tamper")
    if identity_dir.name != expected_id:
        raise RuntimeRefusal("object_tamper", f"object directory {identity_dir.name} is not its id {expected_id}")
    data = _read_bounded(location / "manifest.json", limits.max_lock_bytes, "object_tamper")
    manifest = _strict_json(data, "object_tamper", f"{location}/manifest.json")
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "object_id", "identity", "overlay_files"} or \
            manifest["schema"] != SOURCE_OBJECT_SCHEMA or manifest["object_id"] != expected_id or \
            data != _canonical_bytes(manifest):
        raise RuntimeRefusal("object_tamper", f"{location}/manifest.json is not this object's canonical manifest")
    identity = _validate_identity(manifest["identity"])
    if identity_id(identity) != expected_id:
        raise RuntimeRefusal("object_tamper", f"{location}: identity digest is not {expected_id}")
    overlay_files = manifest["overlay_files"]
    if (identity["overlay"] is None) != (overlay_files is None) or overlay_files is not None and (
            not isinstance(overlay_files, list) or not all(
                isinstance(e, list) and len(e) == 3 and isinstance(e[0], str) and isinstance(e[1], int)
                and isinstance(e[2], str) and _HEX64.match(e[2]) for e in overlay_files)):
        raise RuntimeRefusal("object_tamper", f"{location}: overlay file list is malformed")
    expected: dict[str, tuple[int, str, str | None]] = {
        f"source/{path}": (size, sha, blob) for path, size, sha, blob in identity["files"]}
    for rel, size, sha in overlay_files or []:
        if rel.startswith("/") or ".." in rel.split("/") or not rel.startswith(identity["overlay"]["dist_info"] + "/"):
            raise RuntimeRefusal("object_tamper", f"overlay entry {rel!r} is outside its dist-info directory")
        expected[f"editable/{rel}"] = (size, sha, None)
    expected_dirs = {"source"} | ({"editable"} if overlay_files is not None else set())
    for rel in expected:
        parent = os.path.dirname(rel)
        while parent:
            expected_dirs.add(parent)
            parent = os.path.dirname(parent)
    found, dirs = _walk_object(location, len(expected) + len(expected_dirs) + 1)
    if set(found) != set(expected) | {"manifest.json"} or dirs != expected_dirs:
        extra = sorted((set(found) - set(expected) - {"manifest.json"}) | (dirs - expected_dirs))
        missing = sorted((set(expected) - set(found)) | (expected_dirs - dirs))
        raise RuntimeRefusal("object_tamper", f"{location}: file set differs from the manifest", extra=extra[:50],
                             missing=missing[:50])
    budget = sum(size for size, _sha, _blob in expected.values()) + 1
    captured: dict[str, bytes] = {}
    package = identity["package"]
    keep = {"source/pyproject.toml", f"source/src/{package}/__init__.py"}
    for rel, (size, sha, blob) in expected.items():
        try:
            got_sha, got_blob, got_size, small = _hash_path(
                location / rel, budget=budget, blob_algorithm=identity["object_format"] if blob else None,
                keep=rel in keep or rel.startswith("editable/"))
        except RuntimeRefusal as exc:
            raise RuntimeRefusal("object_tamper", exc.detail) from exc
        if (got_size, got_sha) != (size, sha) or (blob is not None and got_blob != blob):
            raise RuntimeRefusal("object_tamper", f"{location / rel} differs from the object manifest")
        if small is not None:
            captured[rel] = small
    kind = identity["kind"]
    pyproject = captured["source/pyproject.toml"]
    regeneration = "not_applicable"
    if identity["overlay"] is None:
        project = project_identity(pyproject, SOURCES[kind][1], f"{location}/source/pyproject.toml")
    else:
        project = validate_static_project(pyproject, SOURCES[kind][1], f"{location}/source/pyproject.toml")
        check_version_alignment(project, captured[f"source/src/{package}/__init__.py"],
                                f"{location}/source/src/{package}/__init__.py")
        overlay = {rel[len("editable/"):]: data for rel, data in captured.items() if rel.startswith("editable/")}
        generator = identity["overlay"]["generator_sha256"]
        check_overlay_semantics(overlay, project, package=package, object_dir=identity_dir, object_id=expected_id,
                                generator_sha256=generator)
        if generator == current_generator:
            if overlay != generate_overlay(project, package=package, object_dir=identity_dir, object_id=expected_id,
                                           generator_sha256=generator):
                raise RuntimeRefusal("object_tamper", f"{location}: overlay bytes differ from the generator output")
            regeneration = "exact"
        else:
            regeneration = "semantic_only_generator_differs"
    if {key: project[key] for key in ("name", "version", "pyproject_sha256")} != identity["project"]:
        raise RuntimeRefusal("object_tamper", f"{location}: archived pyproject disagrees with the identity")
    return {"object_id": expected_id, "path": str(identity_dir), "kind": kind,
            "file_count": len(identity["files"]), "total_bytes": sum(item[1] for item in identity["files"]),
            "manifest_sha256": _sha256(data), "package_digest": _package_digest(identity["files"], package),
            "overlay_files": {rel: sha for rel, _size, sha in overlay_files} if overlay_files is not None else None,
            "dist_info": identity["overlay"]["dist_info"] if identity["overlay"] else None,
            "overlay_regeneration": regeneration, "project": identity["project"]}


def _copy_verified(source: Path, target: Path, size: int, sha: str) -> None:
    """Copy one regular file (no symlink follow) and require the copied bytes to be the pass-1 bytes."""
    os.makedirs(target.parent, 0o755, exist_ok=True)
    try:
        src = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise RuntimeRefusal("source_changed_during_read", f"cannot reopen {source}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(src).st_mode):
            raise RuntimeRefusal("source_changed_during_read", f"{source} is no longer a regular file")
        dst = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            hasher, copied = hashlib.sha256(), 0
            while True:
                chunk = os.read(src, min(_CHUNK, size + 1 - copied))
                if not chunk:
                    break
                copied += len(chunk)
                if copied > size:
                    break
                hasher.update(chunk)
                _write_all(dst, chunk)
            os.fsync(dst)
        finally:
            os.close(dst)
    finally:
        os.close(src)
    if copied != size or hasher.hexdigest() != sha:
        raise RuntimeRefusal("source_changed_during_read", f"{source} changed between hashing and copying")


def seal_source(store: str, kind: str, observation: dict[str, Any], files: list[list[Any]],
                captured: dict[str, bytes], *, run_id: str, limits: RuntimeLimits,
                _after_copy: Callable[[], None] | None = None) -> dict[str, Any]:
    """Seal one committed scope into `<store>/objects/sources/<id>` (content-addressed, read-only).

    An existing object with the id is fully verified and reused (a corrupt one refuses and is left untouched).
    Otherwise the copy is staged in `objects/sources/.staging-<run_id>-<kind>`, frozen, fully verified against its
    final identity and published with renameat2(RENAME_NOREPLACE). Failed staging is retained as evidence."""
    package, distribution = SOURCES[kind]
    if kind in OVERLAY_KINDS:
        project = validate_static_project(captured["pyproject.toml"], distribution, f"{kind} pyproject.toml")
        check_version_alignment(project, captured[f"src/{package}/__init__.py"], f"{kind} src/{package}/__init__.py")
    else:
        project = project_identity(captured["pyproject.toml"], distribution, f"{kind} pyproject.toml")
    identity = object_identity(kind, object_format=observation["object_format"], files=files, project=project)
    object_id = identity_id(identity)
    sources_dir = Path(store) / "objects" / "sources"
    _require_real_dir(sources_dir)
    final = sources_dir / object_id
    if os.path.lexists(final):
        return {**verify_source_object(final, object_id, limits), "reused": True}
    staging = sources_dir / f".staging-{run_id}-{kind}"
    try:
        os.mkdir(staging, 0o700)
    except FileExistsError as exc:
        raise RuntimeRefusal("staging_exists", f"{staging} already exists") from exc
    root = Path(observation["root"])
    try:
        for path, size, sha, _blob in files:
            _copy_verified(root / path, staging / "source" / path, size, sha)
        if _after_copy is not None:
            _after_copy()
        overlay_files = None
        if kind in OVERLAY_KINDS:
            overlay = generate_overlay(project, package=package, object_dir=final, object_id=object_id)
            for rel, data in overlay.items():
                os.makedirs((staging / "editable" / rel).parent, 0o755, exist_ok=True)
                _write_new(staging / "editable" / rel, data)
            overlay_files = [[rel, len(data), _sha256(data)] for rel, data in overlay.items()]
        manifest = {"schema": SOURCE_OBJECT_SCHEMA, "object_id": object_id, "identity": identity,
                    "overlay_files": overlay_files}
        _write_new(staging / "manifest.json", _canonical_bytes(manifest))
        _freeze(staging)
        verify_source_object(staging, object_id, limits, identity_dir=final)
        try:
            _rename_noreplace(staging, final)
        except FileExistsError:
            # A concurrent prepare published the same id first: verify it; our staging stays as evidence.
            return {**verify_source_object(final, object_id, limits), "reused": True,
                    "retained_staging": str(staging)}
    except RuntimeRefusal as exc:
        exc.evidence.setdefault("retained_staging", str(staging))
        raise
    except OSError as exc:
        raise RuntimeRefusal("object_publication_failed", f"{kind}: {exc}", retained_staging=str(staging)) from exc
    _fsync_dir(sources_dir)
    return {**verify_source_object(final, object_id, limits), "reused": False}


# ------------------------------------------------------------------------------------------------- the store


_STORE_ENTRY_RULES: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "sources": [(_HEX64, "dir"), (re.compile(r"^\.staging-[0-9A-Za-z._-]+-(?:maple|syrup|masy)$"), "dir")],
    "maple-bundles": [(_HEX64, "dir"), (re.compile(r"^\.export-[0-9A-Za-z._-]+$"), "dir"),
                      (re.compile(r"^\.\.export-[0-9A-Za-z._-]+\.staging-[A-Za-z0-9_]+$"), "dir")],
    "maple-materialized": [(_HEX64, "dir"), (re.compile(r"^\.[0-9a-f]{64}\.staging-[A-Za-z0-9_]+$"), "dir")],
    "locks": [(re.compile(r"^[0-9a-f]{64}\.json$"), "file"), (re.compile(r"^\.staging-[0-9A-Za-z._-]+\.json$"), "file")],
    "runs": [(_RUN_ID, "dir")],
}


def validate_store_tree(store: str) -> None:
    """Read-only layout check of an existing store: only the documented entries, no symlinks or special files at
    any layout level, owned by this user. Unknown content refuses; nothing is repaired."""
    _require_real_dir(store)
    if os.lstat(store).st_uid != os.getuid():
        raise RuntimeRefusal("store_component_invalid", f"{store} is not owned by this user")
    top = set(os.listdir(store))
    if top - {"objects", "locks", "runs"}:
        raise RuntimeRefusal("store_unexpected_entry", f"{store} has unexpected entries {sorted(top)}")
    areas = {name: os.path.join(store, name) for name in ("locks", "runs") if name in top}
    if "objects" in top:
        objects = os.path.join(store, "objects")
        _require_real_dir(objects)
        names = set(os.listdir(objects))
        if names - set(OBJECT_SUBDIRS):
            raise RuntimeRefusal("store_unexpected_entry", f"{objects} has unexpected entries {sorted(names)}")
        areas.update({name: os.path.join(objects, name) for name in names})
    for area, path in areas.items():
        _require_real_dir(path)
        for name in os.listdir(path):
            info = os.lstat(os.path.join(path, name))
            kind = "dir" if stat.S_ISDIR(info.st_mode) else "file" if stat.S_ISREG(info.st_mode) else "other"
            if not any(pattern.match(name) and kind == want for pattern, want in _STORE_ENTRY_RULES[area]):
                raise RuntimeRefusal("store_unexpected_entry", f"{path}/{name} ({kind}) is not a valid {area} entry")


def ensure_store(store: str) -> None:
    """Create the store and its fixed layout (parent must already exist); never follows a symlink."""
    if not os.path.lexists(store):
        parent = os.path.dirname(store)
        _require_real_dir(parent, "store_parent_missing")
        try:
            os.mkdir(store, 0o755)
        except FileExistsError:
            pass
    for relative in ("objects", "locks", "runs", *(f"objects/{name}" for name in OBJECT_SUBDIRS)):
        path = os.path.join(store, relative)
        try:
            os.mkdir(path, 0o755)
        except FileExistsError:
            pass
    check_store_path(store)
    validate_store_tree(store)


class _Run:
    """One run directory `<store>/runs/<run_id>`; every stage is appended to stages.jsonl as it happens."""

    def __init__(self, store: str, run_id: str, operation: str) -> None:
        if not isinstance(run_id, str) or not _RUN_ID.match(run_id):
            raise RuntimeRefusal("run_id_invalid", f"run id {run_id!r} is not a safe single component")
        self.store, self.run_id, self.operation = store, run_id, operation
        self.dir = Path(store) / "runs" / run_id
        self.pycache = str(self.dir / "cache" / "pycache")
        self.workers: list[dict[str, Any]] = []
        self.log_failures: list[str] = []

    def create(self) -> _Run:
        _require_real_dir(Path(self.store) / "runs")
        try:
            os.mkdir(self.dir, 0o700)
        except FileExistsError as exc:
            raise RuntimeRefusal("run_exists", f"run directory {self.dir} already exists") from exc
        for relative in ("cache", "cache/numba", "cache/matplotlib", "cache/xdg", "home", "tmp"):
            os.mkdir(self.dir / relative, 0o700)
        return self

    def stage(self, name: str, status: str, **data: Any) -> None:
        record = {"stage": name, "status": status, "time_unix_s": time.time(), "monotonic_s": time.monotonic(),
                  **_jsonable(data)}
        line = (canonical_json(record) + "\n").encode("ascii")
        if len(line) > MAX_STAGE_LINE_BYTES:
            line = (canonical_json({"stage": name, "status": status, "truncated": True, "bytes": len(line)})
                    + "\n").encode("ascii")
        try:
            fd = os.open(self.dir / "stages.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                _write_all(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:  # e.g. ENOSPC: keep going, report it in the output
            self.log_failures.append(f"{name}: {exc}")

    def write(self, name: str, data: bytes) -> Path:
        path = self.dir / name
        _write_new(path, data)
        return path

    def evidence(self, record: dict[str, Any], max_bytes: int) -> str | None:
        data = _canonical_bytes(_jsonable({"schema": EVIDENCE_SCHEMA, **record, "workers": self.workers,
                                           "log_failures": self.log_failures}))
        if len(data) > max_bytes:
            data = _canonical_bytes({"schema": EVIDENCE_SCHEMA, "truncated": True, "bytes": len(data),
                                     "verdict": record.get("verdict"), "refusal": _jsonable(record.get("refusal"))})
        try:
            self.write("evidence.json", data)
        except OSError as exc:
            self.log_failures.append(f"evidence.json: {exc}")
            print(f"masy_runtime: cannot write evidence.json: {exc}", file=sys.stderr)
            return None
        return str(self.dir / "evidence.json")


def _run_dir_bytes(limits: RuntimeLimits) -> int:
    return 4 * limits.worker_max_output_bytes + 2 * limits.max_lock_bytes + limits.max_evidence_bytes + (1 << 20)


def prepare_estimate(before: dict[str, Any], limits: RuntimeLimits) -> int:
    """Conservative new-byte estimate assuming nothing can be reused: MAPLE scope x4 (source object, upstream
    archive, a duplicate run export, materialized cache) with per-file block/tar overhead, SYRUP and MASY scopes
    x2 (copy + slack), overlays, dist-info copies, lock and the run directory."""
    def scope(name: str) -> int:
        tracked = before[name]["tracked_scope"]
        return tracked["total_bytes"] + 4096 * (tracked["file_count"] + 1)
    dist = sum(item["size"] + 4096 for item in before["maple"]["installed_distribution_metadata"]["files"].values())
    return 4 * scope("maple") + 3 * dist + 2 * (scope("syrup") + scope("masy")) + 4 * (64 << 10) + \
        limits.max_lock_bytes + _run_dir_bytes(limits)


def disk_preflight(store: str, estimate: int, limits: RuntimeLimits, *,
                   statvfs: Callable[[str], Any] = os.statvfs) -> dict[str, Any]:
    """Before anything is created: the estimate must fit `store_max_new_bytes`, and the free space available to
    this user on the store's filesystem must cover the estimate plus `disk_reserve_bytes`. Per-user quotas that
    statvfs does not report are not visible here."""
    probe = store
    while not os.path.lexists(probe):
        probe = os.path.dirname(probe)
    info = statvfs(probe)
    free = int(info.f_bavail) * int(info.f_frsize)
    record = {"probe": probe, "free_bytes": free, "estimate_bytes": estimate,
              "store_max_new_bytes": limits.store_max_new_bytes, "disk_reserve_bytes": limits.disk_reserve_bytes}
    if estimate > limits.store_max_new_bytes:
        raise RuntimeRefusal("disk_budget_exceeded", f"estimated {estimate} new bytes exceed store_max_new_bytes="
                             f"{limits.store_max_new_bytes}; nothing was created", **record)
    if free < estimate + limits.disk_reserve_bytes:
        raise RuntimeRefusal("disk_quota_insufficient", f"{free} free bytes on {probe} < estimate {estimate} + "
                             f"reserve {limits.disk_reserve_bytes}; nothing was created", **record)
    return record


# ------------------------------------------------------------------------------------------ bounded children


@dataclass(frozen=True)
class BoundedResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    elapsed_s: float
    pid: int


def _leader_exited(pid: int) -> bool:
    """True once the child has exited, WITHOUT reaping it (its pid, hence its process group id, stays reserved
    until we kill the group and reap)."""
    try:
        return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT | os.WNOHANG) is not None
    except ChildProcessError:
        return True


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_bounded(argv: list[str], *, env: dict[str, str], timeout_s: float, max_output_bytes: int,
                cwd: str = "/") -> BoundedResult:
    """Run one child in its own session/process group. stdout+stderr are read while it runs and capped by their
    COMBINED size; wall time is bounded. Whatever the outcome (success, timeout, overflow, error), the whole
    process group is SIGKILLed before the leader is reaped, so no straggler survives. Raises RuntimeRefusal on
    timeout/overflow with bounded partial output in `.partial`."""
    start = time.monotonic()
    deadline = start + timeout_s
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=env, cwd=cwd, start_new_session=True, close_fds=True)
    except OSError as exc:
        raise RuntimeRefusal("worker_launch_failed", f"cannot start {argv[0]!r}: {exc}") from exc
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    total, failure, exited_at = 0, None, None
    selector = selectors.DefaultSelector()
    try:
        selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
        selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map() and failure is None:
            now = time.monotonic()
            if now >= deadline:
                failure = ("worker_timeout", f"child exceeded {timeout_s} s")
                break
            if exited_at is None and _leader_exited(proc.pid):
                exited_at = now
            if exited_at is not None and now - exited_at > STRAGGLER_GRACE_S:
                break  # descendants still hold the pipes; they are killed below
            for key, _events in selector.select(min(deadline - now, 0.05)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                room = max_output_bytes - total
                total += len(chunk)
                buffers[key.data] += chunk[:max(room, 0)]
                if total > max_output_bytes:
                    failure = ("worker_output_exceeded",
                               f"child printed more than {max_output_bytes} bytes (stdout+stderr)")
                    break
        while failure is None and not _leader_exited(proc.pid):
            if time.monotonic() >= deadline:
                failure = ("worker_timeout", f"child exceeded {timeout_s} s")
                break
            time.sleep(0.01)
    finally:
        _kill_group(proc.pid)
        returncode = proc.wait()
        selector.close()
        proc.stdout.close()
        proc.stderr.close()
    elapsed = time.monotonic() - start
    out, err = bytes(buffers["stdout"]), bytes(buffers["stderr"])
    if failure is not None:
        exc = RuntimeRefusal(failure[0], failure[1], pid=proc.pid, elapsed_s=round(elapsed, 3),
                             returncode_after_kill=returncode,
                             stdout_head=out[:2000].decode("utf-8", "replace"),
                             stderr_tail=err[-2000:].decode("utf-8", "replace"))
        exc.partial = (out, err)  # type: ignore[attr-defined]
        raise exc
    return BoundedResult(returncode, out, err, elapsed, proc.pid)


def worker_env(run_dir: Path | str) -> dict[str, str]:
    """The complete child environment, built from nothing: CPU-only device switches and run-owned caches/home."""
    run = Path(run_dir)
    return {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "HOME": str(run / "home"), "TMPDIR": str(run / "tmp"),
            "XDG_CACHE_HOME": str(run / "cache" / "xdg"), "NUMBA_CACHE_DIR": str(run / "cache" / "numba"),
            "MPLCONFIGDIR": str(run / "cache" / "matplotlib"), "NUMBA_DISABLE_CUDA": "1",
            "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}


def worker_command(python: str, bootstrap: str, run: _Run, request_path: Path) -> list[str]:
    return [python, "-I", "-B", "-X", f"pycache_prefix={run.pycache}", "-c", bootstrap, str(request_path)]


def sealed_bootstrap(masy_object: Path | str) -> str:
    """BOOTSTRAP read from the SEALED MASY object's runtime_worker.py source (AST literal; never imported here)."""
    path = Path(masy_object) / "source" / "src" / "masy" / "runtime_worker.py"
    data = _read_bounded(path, MAX_SMALL_FILE_BYTES * 4, "bootstrap_unavailable")
    try:
        tree = ast.parse(data.decode("utf-8"), filename=str(path))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise RuntimeRefusal("bootstrap_unavailable", f"{path}: {exc}") from exc
    values = [node.value.value for node in tree.body
              if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id == "BOOTSTRAP" and isinstance(node.value, ast.Constant)
              and isinstance(node.value.value, str)]
    if len(values) != 1:
        raise RuntimeRefusal("bootstrap_unavailable", f"{path} must define one literal BOOTSTRAP")
    return values[0]


def call_worker(run: _Run, operation: str, body: dict[str, Any], *, python: str, bootstrap: str,
                limits: RuntimeLimits) -> dict[str, Any]:
    """Write the request, run one bounded child, save its raw outputs and receipt, and return its `ok` result.
    A refused/incompatible result raises RuntimeRefusal with the worker evidence; anything else (non-zero exit
    without a result, unparseable or incomplete output) refuses as `worker_failed`; never a fake success."""
    index = len(run.workers) + 1
    request = {"schema": _worker.REQUEST_SCHEMA, "operation": operation, **body}
    request_path = run.write(f"request-{index}-{operation}.json", _canonical_bytes(request))
    argv = worker_command(python, bootstrap, run, request_path)
    receipt: dict[str, Any] = {"index": index, "operation": operation, "request_path": str(request_path),
                               "request_sha256": _sha256(_canonical_bytes(request)),
                               "argv": [*argv[:6], f"<BOOTSTRAP sha256 {_sha256(bootstrap.encode())}>", argv[7]],
                               "run_dir": str(run.dir)}
    run.workers.append(receipt)
    run.stage(f"worker_{index}_{operation}", "started", request=str(request_path))
    try:
        result = run_bounded(argv, env=worker_env(run.dir), timeout_s=limits.worker_timeout_s,
                             max_output_bytes=limits.worker_max_output_bytes)
    except RuntimeRefusal as exc:
        out, err = getattr(exc, "partial", (b"", b""))
        for suffix, data in (("stdout", out), ("stderr", err)):
            receipt[suffix] = str(run.write(f"worker-{index}-{operation}.{suffix}", data))
        receipt.update(status="failed", code=exc.code)
        run.stage(f"worker_{index}_{operation}", "failed", code=exc.code)
        raise
    for suffix, data in (("stdout", result.stdout), ("stderr", result.stderr)):
        receipt[suffix] = str(run.write(f"worker-{index}-{operation}.{suffix}", data))
        receipt[f"{suffix}_sha256"] = _sha256(data)
    receipt.update(returncode=result.returncode, elapsed_s=round(result.elapsed_s, 3), pid=result.pid)
    lines = result.stdout.decode("utf-8", "replace").splitlines()
    parsed: Any = None
    if len(lines) == 1:
        try:
            parsed = _strict_json(lines[0].encode("utf-8"), "worker_failed", "worker result")
        except RuntimeRefusal:
            parsed = None
    valid = isinstance(parsed, dict) and parsed.get("schema") == _worker.RESULT_SCHEMA and \
        parsed.get("operation") in (operation, "bootstrap") and parsed.get("status") in _worker.STATUS_EXIT and \
        result.returncode == _worker.STATUS_EXIT[parsed["status"]]
    if not valid:
        receipt.update(status="failed", code="worker_failed")
        run.stage(f"worker_{index}_{operation}", "failed", code="worker_failed", returncode=result.returncode)
        raise RuntimeRefusal("worker_failed", f"{operation} child exited {result.returncode} without one valid result "
                             f"line", stderr_tail=result.stderr[-2000:].decode("utf-8", "replace"),
                             stdout_head=result.stdout[:2000].decode("utf-8", "replace"))
    receipt.update(status=parsed["status"], operational=parsed.get("operational"))
    run.stage(f"worker_{index}_{operation}", parsed["status"], code=parsed.get("code"))
    if parsed["status"] != "ok":
        raise RuntimeRefusal(parsed.get("code", "worker_refused"), parsed.get("detail", ""),
                             severity="incompatible" if parsed["status"] == "incompatible" else "refusal",
                             worker=operation, worker_evidence=parsed.get("evidence"))
    if set(parsed) != {"schema", "operation", "status", "stable", "operational", "receipt"}:
        raise RuntimeRefusal("worker_failed", f"{operation} result has keys {sorted(parsed)}")
    check_operational(parsed["operational"], run)
    return parsed


def check_operational(operational: Any, run: _Run) -> None:
    """Per-run values must point into THIS run directory exactly; they are then never compared across runs."""
    env = worker_env(run.dir)
    expected_env = {key: env[key] for key in _worker.OPERATIONAL_ENV_KEYS}
    if not isinstance(operational, dict) or operational.get("pycache_prefix") != run.pycache or \
            operational.get("environment") != expected_env or not isinstance(operational.get("pid"), int):
        raise RuntimeRefusal("worker_operational_mismatch", "child pycache prefix/caches/home are not this run's",
                             operational=operational, expected={"pycache_prefix": run.pycache, "environment":
                                                                expected_env})


# ------------------------------------------------------------------------------------------- request shapes


def _object_paths(store: str, *, object_ids: dict[str, str], bundle_id: str) -> dict[str, str]:
    root = os.path.join(store, "objects")
    paths = {kind: os.path.join(root, "sources", object_ids[kind]) for kind in SOURCES}
    paths["bundle"] = os.path.join(root, "maple-bundles", bundle_id)
    paths["materialized"] = os.path.join(root, "maple-materialized", bundle_id)
    return paths


def runtime_sys_path(paths: dict[str, str]) -> list[str]:
    """Exact sealed prefix: metadata overlays first (so importlib.metadata selects them), then package roots."""
    return [os.path.join(paths["materialized"], "editable"), os.path.join(paths["syrup"], "editable"),
            os.path.join(paths["masy"], "editable"), os.path.join(paths["masy"], "source", "src"),
            os.path.join(paths["syrup"], "source", "src"), os.path.join(paths["materialized"], "source", "src")]


def bundle_sys_path(paths: dict[str, str]) -> list[str]:
    return [os.path.join(paths["masy"], "source", "src"), os.path.join(paths["syrup"], "source", "src")]


def check_import_roots(prefix: list[str]) -> None:
    """The sealed src roots must not offer the same top-level name twice; overlay dirs hold only metadata."""
    seen: dict[str, str] = {}
    for entry in prefix:
        names = os.listdir(entry)
        if entry.endswith("/editable"):
            bad = [n for n in names if not (n.endswith(".dist-info") or (n.startswith("__editable__.")
                                                                       and n.endswith(".pth")))]
            if bad:
                raise RuntimeRefusal("import_root_invalid", f"{entry} holds non-metadata entries {bad[:20]}")
            continue
        for name in names:
            top = name[:-3] if name.endswith(".py") else name
            if top in seen:
                raise RuntimeRefusal("import_root_collision", f"{top!r} is offered by both {seen[top]} and {entry}")
            seen[top] = entry


def _python_request(python: dict[str, Any]) -> dict[str, Any]:
    return {"spelling": python["spelling"], "realpath": python["realpath"],
            "venv_dir": python["venv"]["dir"] if python["venv"] else None}


def _request_common(run: _Run, python: dict[str, Any], prefix: list[str], packages: dict[str, Any],
                    limits: RuntimeLimits) -> dict[str, Any]:
    return {"python": _python_request(python),
            "run": {"run_dir": str(run.dir), "pycache_prefix": run.pycache, "env": worker_env(run.dir)},
            "sys_path_prefix": prefix, "packages": packages,
            "limits": {"max_files": limits.max_files, "max_bytes": limits.max_bytes,
                       "max_compiled_files": limits.max_compiled_files,
                       "max_compiled_bytes": limits.max_compiled_bytes}}


def _runtime_request(lock: dict[str, Any], run: _Run, limits: RuntimeLimits) -> dict[str, Any]:
    objects = lock["objects"]
    paths = _object_paths(lock["store"], object_ids={k: v["object_id"] for k, v in objects["sources"].items()},
                          bundle_id=objects["maple_bundle"]["bundle_id"])
    packages = {
        "maple": {"dir": os.path.join(paths["materialized"], "source", "src", "maple"),
                  "digest": objects["maple_bundle"]["package_digest"]},
        "maple_syrup": {"dir": os.path.join(paths["syrup"], "source", "src", "maple_syrup"),
                        "digest": objects["sources"]["syrup"]["package_digest"]},
        "masy": {"dir": os.path.join(paths["masy"], "source", "src", "masy"),
                 "digest": objects["sources"]["masy"]["package_digest"]},
    }
    body = _request_common(run, lock["runtime"]["python"], lock["runtime"]["sys_path_prefix"], packages, limits)
    body["bundle"] = {"dir": paths["bundle"], "bundle_id": objects["maple_bundle"]["bundle_id"],
                      "target": paths["materialized"],
                      "tool_module_sha256": objects["maple_bundle"]["tool_module_sha256"]}
    body["distributions"] = [{"name": name, "dir": spec["overlay_dir"], "files": spec["files"]}
                             for name, spec in sorted(lock["distributions"].items())]
    return body


# --------------------------------------------------------------------------------------------- MAPLE bundle


def _origin_record(maple: dict[str, Any], object_id: str) -> dict[str, Any]:
    return {"schema": "maple_syrup.dependency_origin.v1",
            "description": f"MASY runtime prepare: scoped copy (src/, pyproject.toml) of committed MAPLE {maple['head']}"
                           f" sealed as MASY source object {object_id}",
            "original_source": {"kind": "git_commit_scoped_copy", "revision": maple["head"],
                                "revision_absence_reason": None, "dirty_snapshot_note": None},
            "patch_chain": [],
            "patch_chain_absence_reason": "the committed HEAD was selected with a clean scope; no patch was applied"}


def check_bundle_manifest(data: bytes, *, bundle_id: str, tool_sha256: str, maple_object: dict[str, Any],
                          maple_files: list[list[Any]], distribution: dict[str, Any], origin: dict[str, Any],
                          package_digest: str) -> dict[str, Any]:
    """Parent cross-check of the upstream export manifest against what MASY sealed and observed (the full
    upstream verification runs in the sealed child)."""
    manifest = _strict_json(data, "bundle_manifest_mismatch", "bundle-manifest.json")
    problems = []
    if not isinstance(manifest, dict) or manifest.get("bundle_id") != bundle_id:
        raise RuntimeRefusal("bundle_manifest_mismatch", "manifest bundle_id differs from the export result")
    if manifest.get("tool", {}).get("module_sha256") != tool_sha256:
        problems.append("tool.module_sha256 is not the sealed SYRUP dependency_bundle.py")
    project = manifest.get("project", {})
    if project.get("exported_from") != os.path.join(maple_object["path"], "source"):
        problems.append(f"exported_from {project.get('exported_from')!r} is not the sealed MAPLE object")
    if project.get("package_source_digest", {}).get("digest_sha256") != package_digest:
        problems.append("package_source_digest differs from the observed MAPLE package digest")
    if manifest.get("origin") != origin:
        problems.append("origin record differs from the one MASY wrote")
    if manifest.get("distribution", {}).get("source_dir") != distribution["path"]:
        problems.append("distribution.source_dir is not the configured dist-info")
    files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    source = sorted([e.get("path", "")[len("source/"):], e.get("size"), e.get("sha256")]
                    for e in files if isinstance(e, dict) and str(e.get("path", "")).startswith("source/"))
    if source != sorted([path, size, sha] for path, size, sha, _blob in maple_files):
        problems.append("archived source members differ from the sealed MAPLE object files")
    for entry in files:
        if isinstance(entry, dict) and str(entry.get("path", "")).startswith("distribution/"):
            observed = distribution["files"].get(entry["path"][len("distribution/"):])
            if observed is None or observed["sha256"] != entry.get("sha256"):
                problems.append(f"{entry['path']} differs from the observed installed dist-info")
    if problems:
        raise RuntimeRefusal("bundle_manifest_mismatch", "; ".join(problems))
    return manifest


def _remove_own_export(path: Path, run_id: str) -> None:
    """Remove only this run's own duplicate export after an identical bundle was verified (never an object)."""
    if path.name != f".export-{run_id}" or sorted(os.listdir(path)) != ["bundle-manifest.json", "maple-project.tar"]:
        raise RuntimeRefusal("store_component_invalid", f"refusing to remove unexpected {path}")
    os.chmod(path, 0o700)
    for name in ("bundle-manifest.json", "maple-project.tar"):
        if not stat.S_ISREG(os.lstat(path / name).st_mode):
            raise RuntimeRefusal("store_component_invalid", f"{path / name} is not a regular file")
        os.unlink(path / name)
    os.rmdir(path)


def _editable_files(editable: Path) -> dict[str, str]:
    files, pending = {}, [editable]
    while pending:
        current = pending.pop()
        for name in sorted(os.listdir(current)):
            full = current / name
            info = os.lstat(full)
            if stat.S_ISDIR(info.st_mode):
                pending.append(full)
            elif stat.S_ISREG(info.st_mode) and len(files) < MAX_DIST_INFO_FILES:
                files[full.relative_to(editable).as_posix()] = _hash_path(full, budget=MAX_SMALL_FILE_BYTES)[0]
            else:
                raise RuntimeRefusal("materialized_overlay_invalid", f"{full} is not an expected overlay file")
    return dict(sorted(files.items()))


# ------------------------------------------------------------------------------------------------------ lock


_LOCK_KEYS = {"schema", "lock_id", "selection", "store", "sources", "objects", "distributions", "runtime",
              "verification", "limits", "interpretation"}


def _source_lock_record(observation: dict[str, Any], object_id: str) -> dict[str, Any]:
    record = {key: observation[key] for key in ("root", "head", "tree", "object_format", "expected_revision",
                                                "pyproject", "package", "tracked_scope")}
    record["live_generated_metadata_observed"] = observation["generated_metadata"]
    record["object_id"] = object_id
    for extra in ("launcher", "installed_distribution_metadata"):
        if extra in observation:
            record[extra] = observation[extra]
    return record


def lock_body_id(lock: dict[str, Any]) -> str:
    return _sha256(canonical_json({k: v for k, v in lock.items() if k != "lock_id"}).encode("ascii"))


def validate_lock(lock: Any, *, store: str, lock_name: str | None = None) -> dict[str, Any]:
    """The single strict validator used before publication and before every verify: closed schema, self-hash,
    filename, store, object references/paths, exact sealed sys.path prefix and interpretation."""
    def bad(detail: str) -> RuntimeRefusal:
        return RuntimeRefusal("lock_invalid", detail)

    if not isinstance(lock, dict) or set(lock) != _LOCK_KEYS:
        raise bad(f"lock keys {sorted(lock) if isinstance(lock, dict) else type(lock).__name__} != "
                  f"{sorted(_LOCK_KEYS)}")
    if lock["schema"] != LOCK_SCHEMA:
        raise bad(f"schema {lock['schema']!r} is not {LOCK_SCHEMA!r}")
    if not isinstance(lock["lock_id"], str) or not _HEX64.match(lock["lock_id"]) or \
            lock_body_id(lock) != lock["lock_id"]:
        raise bad("lock_id is not sha256 of the canonical lock body")
    if lock_name is not None and lock_name != f"{lock['lock_id']}.json":
        raise bad(f"lock file name {lock_name} is not {lock['lock_id']}.json")
    if lock["store"] != store:
        raise bad(f"lock store {lock['store']!r} is not {store!r}")
    selection = lock["selection"]
    if not isinstance(selection, dict) or selection.get("policy") != "current_local_committed_head" or \
            selection.get("development_snapshot") is not False:
        raise bad("selection must be the current-local-committed-head policy without a development snapshot")
    RuntimeLimits.from_mapping(lock["limits"], "lock.limits")
    sources, objects = lock["sources"], lock["objects"]
    if not isinstance(sources, dict) or set(sources) != set(SOURCES) or not isinstance(objects, dict) or \
            set(objects) != {"sources", "maple_bundle", "maple_materialized"} or \
            not isinstance(objects["sources"], dict) or set(objects["sources"]) != set(SOURCES):
        raise bad("lock sources/objects do not name exactly maple, syrup and masy")
    ids = {}
    for kind in SOURCES:
        source, obj = sources[kind], objects["sources"][kind]
        if not isinstance(source, dict) or not isinstance(obj, dict) or \
                not all(isinstance(source.get(k), str) and _OBJECT_ID.match(source[k]) for k in ("head", "tree")) or \
                not isinstance(source.get("root"), str) or not os.path.isabs(source["root"]) or \
                not isinstance(obj.get("object_id"), str) or not _HEX64.match(obj["object_id"]) or \
                source.get("object_id") != obj["object_id"] or \
                not isinstance(obj.get("package_digest"), str) or not _HEX64.match(obj["package_digest"]):
            raise bad(f"lock source/object record for {kind} is malformed")
        ids[kind] = obj["object_id"]
    bundle, materialized = objects["maple_bundle"], objects["maple_materialized"]
    if not isinstance(bundle, dict) or not isinstance(materialized, dict) or \
            not all(isinstance(bundle.get(k), str) and _HEX64.match(bundle[k]) for k in (
                "bundle_id", "manifest_sha256", "archive_sha256", "tool_module_sha256", "package_digest")) or \
            materialized.get("bundle_id") != bundle["bundle_id"]:
        raise bad("lock MAPLE bundle/materialized records are malformed")
    paths = _object_paths(store, object_ids=ids, bundle_id=bundle["bundle_id"])
    for kind in SOURCES:
        if objects["sources"][kind].get("path") != paths[kind]:
            raise bad(f"lock {kind} object path is not {paths[kind]}")
    if bundle.get("path") != paths["bundle"] or materialized.get("path") != paths["materialized"]:
        raise bad("lock MAPLE bundle/materialized paths are not derived from the bundle id")
    distributions = lock["distributions"]
    overlay_dirs = {"maple": os.path.join(paths["materialized"], "editable"),
                    "maple-syrup": os.path.join(paths["syrup"], "editable"),
                    "masy": os.path.join(paths["masy"], "editable")}
    if not isinstance(distributions, dict) or set(distributions) != set(overlay_dirs):
        raise bad("lock distributions must name exactly maple, maple-syrup and masy")
    for name, spec in distributions.items():
        if not isinstance(spec, dict) or spec.get("overlay_dir") != overlay_dirs[name] or \
                not isinstance(spec.get("files"), dict) or not spec["files"] or not all(
                    isinstance(k, str) and isinstance(v, str) and _HEX64.match(v) for k, v in spec["files"].items()):
            raise bad(f"lock distribution record for {name} is malformed")
    runtime = lock["runtime"]
    if not isinstance(runtime, dict) or runtime.get("sys_path_prefix") != runtime_sys_path(paths):
        raise bad("lock sys_path_prefix is not the exact sealed prefix of its objects")
    python = runtime.get("python")
    if not isinstance(python, dict) or not isinstance(python.get("spelling"), str) or \
            not os.path.isabs(python["spelling"]) or not isinstance(python.get("realpath"), str) or \
            not isinstance(python.get("sha256"), str) or not _HEX64.match(python["sha256"]) or \
            not (python.get("venv") is None or isinstance(python["venv"], dict)) or \
            not isinstance(runtime.get("bootstrap_sha256"), str) or not _HEX64.match(runtime["bootstrap_sha256"]):
        raise bad("lock runtime python/bootstrap record is malformed")
    if not isinstance(lock["interpretation"], dict) or lock["interpretation"].get("transport_ready") is not False:
        raise bad("lock interpretation must state transport_ready=false")
    verification = lock["verification"]
    if not isinstance(verification, dict) or not isinstance(verification.get("bundle"), dict) or \
            verification["bundle"].get("bundle_id") != bundle["bundle_id"] or \
            verification.get("process", {}).get("sys_path_prefix") != runtime["sys_path_prefix"]:
        raise bad("lock verification record does not match its objects")
    return lock


def publish_lock(store: str, lock: dict[str, Any], *, run_id: str, limits: RuntimeLimits) -> tuple[Path, bool]:
    """O_EXCL staging + link (no replace) to `locks/<lock_id>.json`, read back. Identical existing bytes are
    reused; different bytes refuse (`lock_collision`) and the original is never touched."""
    validate_lock(lock, store=store)
    data = _canonical_bytes(lock)
    if len(data) > limits.max_lock_bytes:
        raise RuntimeRefusal("lock_oversized", f"lock is {len(data)} bytes > max_lock_bytes={limits.max_lock_bytes}")
    locks = Path(store) / "locks"
    _require_real_dir(locks)
    final = locks / f"{lock['lock_id']}.json"
    staging = locks / f".staging-{run_id}.json"
    _write_new(staging, data, 0o444)
    reused = False
    try:
        os.link(staging, final, follow_symlinks=False)
    except FileExistsError:
        existing = _read_bounded(final, limits.max_lock_bytes, "lock_collision")
        if existing != data:
            raise RuntimeRefusal("lock_collision", f"{final} exists with different bytes; it was left untouched",
                                 retained_staging=str(staging)) from None
        reused = True
    os.unlink(staging)
    _fsync_dir(locks)
    if _read_bounded(final, limits.max_lock_bytes, "lock_readback_failed") != data:
        raise RuntimeRefusal("lock_readback_failed", f"{final} does not read back as the published bytes")
    return final, reused


def load_lock_ref(store: str, ref: str, max_bytes: int) -> tuple[dict[str, Any], Path]:
    """`ref` is a 64-hex lock id or the absolute path of `<store>/locks/<id>.json`; bounded strict read."""
    locks = Path(store) / "locks"
    if isinstance(ref, str) and _HEX64.match(ref):
        path = locks / f"{ref}.json"
    elif isinstance(ref, str) and os.path.isabs(ref) and os.path.normpath(ref) == ref and \
            os.path.dirname(ref) == str(locks) and os.path.realpath(os.path.dirname(ref)) == str(locks):
        path = Path(ref)
    else:
        raise RuntimeRefusal("lock_ref_invalid", f"--lock {ref!r} must be a lock id or a file directly in {locks}")
    data = _read_bounded(path, max_bytes, "lock_unreadable")
    lock = _strict_json(data, "lock_unreadable", str(path))
    if data != _canonical_bytes(lock):
        raise RuntimeRefusal("lock_invalid", f"{path} is not canonical lock JSON")
    return validate_lock(lock, store=store, lock_name=path.name), path


# -------------------------------------------------------------------------------------------- prepare/verify


def _output(verdict: str, **values: Any) -> dict[str, Any]:
    return {"schema": OUTPUT_SCHEMA, "verdict": verdict, "exit_code": EXIT_CODES[verdict],
            "transport_ready": False, "transport_entry_point": None, "unimplemented": list(UNIMPLEMENTED),
            **_jsonable(values)}


def refusal_output(exc: RuntimeRefusal, *, run: _Run | None = None, evidence_path: str | None = None,
                   **values: Any) -> dict[str, Any]:
    evidence = _jsonable(exc.evidence)
    text = canonical_json(evidence)
    if len(text) > MAX_OUTPUT_EVIDENCE_CHARS:
        evidence = {"truncated": True, "chars": len(text), "see": evidence_path}
    verdict = "incompatible" if exc.severity == "incompatible" else "refused"
    return _output(verdict, code=exc.code, detail=exc.detail, evidence=evidence, lock_published=False,
                   run_dir=str(run.dir) if run else None, evidence_path=evidence_path,
                   log_failures=run.log_failures if run else [], **values)


def _check_launch(launch: dict[str, Any], store: str) -> None:
    for key in ("launcher", "masy_root", "run_id", "store"):
        if not isinstance(launch.get(key), str):
            raise RuntimeRefusal("launch_invalid", f"launch record lacks {key}")
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise RuntimeRefusal("launcher_flags_missing", "the parent must run with -I -B")
    if launch["store"] != store:
        raise RuntimeRefusal("launch_invalid", f"launcher store {launch['store']} != {store}")
    expected = os.path.join(store, "runs", launch["run_id"], "cache", "pycache")
    if sys.pycache_prefix != expected:
        raise RuntimeRefusal("launch_invalid", f"sys.pycache_prefix {sys.pycache_prefix!r} is not {expected!r}")
    from masy.importcheck import module_origin_violations

    violations = module_origin_violations(dict(sys.modules),
                                          {"masy": os.path.join(launch["masy_root"], "src", "masy")})
    if violations:
        raise RuntimeRefusal("parent_module_origin_mismatch", "parent masy modules are not the launcher's tree",
                             violations=violations[:50])
    preloaded = sorted(n for n in sys.modules if n.partition(".")[0] in ("maple", "maple_syrup"))
    if preloaded:
        raise RuntimeRefusal("parent_upstream_imported", f"the parent must not import upstream code: {preloaded[:20]}")


def _python_stable(python: dict[str, Any]) -> dict[str, Any]:
    return {key: python[key] for key in ("spelling", "is_symlink", "link_target", "realpath", "size", "sha256",
                                         "venv")}


def prepare(config: RuntimeConfig, *, launch: dict[str, Any], statvfs: Callable[[str], Any] = os.statvfs
            ) -> dict[str, Any]:
    limits, store = config.limits, config.store
    _check_launch(launch, store)
    masy_root = launch["masy_root"]
    python = interpreter_identity(config.python, limits)
    protected = _protected_inputs(maple_root=config.maple.root, syrup_root=config.syrup.root,
                                  dist_info=config.maple_distribution_metadata, python=python, wrapper_root=masy_root)
    check_store_location(store, wrapper_roots=[masy_root], protected=protected)
    if os.path.lexists(store):
        validate_store_tree(store)
    before = observe_selection(config, masy_root=masy_root, launcher=launch["launcher"])
    hashed = {kind: hash_scope(before[kind], limits) for kind in SOURCES}
    runtime_entry = [e for e in hashed["masy"][0] if e[0] == "src/masy/runtime.py"]
    if not runtime_entry or runtime_entry[0][2] != GENERATOR_SHA256:
        raise RuntimeRefusal("source_changed_during_prepare", "the running masy/runtime.py (overlay generator) is not "
                             "the committed one", source="masy")
    for kind in OVERLAY_KINDS:  # static metadata/version checks before anything is written
        files, captured = hashed[kind]
        package, distribution = SOURCES[kind]
        project = validate_static_project(captured["pyproject.toml"], distribution, f"{kind} pyproject.toml")
        check_version_alignment(project, captured[f"src/{package}/__init__.py"], f"{kind} src/{package}/__init__.py")
    if "src/maple_syrup/dependency_bundle.py" not in {e[0] for e in hashed["syrup"][0]}:
        raise RuntimeRefusal("bundle_tool_missing", "committed SYRUP lacks src/maple_syrup/dependency_bundle.py")
    disk = disk_preflight(store, prepare_estimate(before, limits), limits, statvfs=statvfs)

    ensure_store(store)
    run = _Run(store, launch["run_id"], "prepare").create()
    record: dict[str, Any] = {"operation": "prepare", "run_id": run.run_id, "config": config.record(),
                              "launch": launch, "python": python, "disk_preflight": disk, "before": before}
    run.stage("observed", "ok", heads={k: before[k]["head"] for k in SOURCES}, disk=disk)
    try:
        sealed = {}
        for kind in ("maple", "syrup", "masy"):
            files, captured = hashed[kind]
            sealed[kind] = seal_source(store, kind, before[kind], files, captured, run_id=run.run_id, limits=limits)
            run.stage(f"sealed_{kind}", "ok", object_id=sealed[kind]["object_id"], reused=sealed[kind]["reused"])
        record["objects"] = sealed
        bootstrap = sealed_bootstrap(sealed["masy"]["path"])
        if bootstrap != _worker.BOOTSTRAP:
            raise RuntimeRefusal("source_changed_during_prepare", "sealed BOOTSTRAP differs from the running worker")
        tool_sha = next(e[2] for e in hashed["syrup"][0] if e[0] == "src/maple_syrup/dependency_bundle.py")
        paths = {kind: sealed[kind]["path"] for kind in SOURCES}
        bundle_prefix = bundle_sys_path(paths)
        check_import_roots(bundle_prefix)
        bundle_packages = {
            "maple_syrup": {"dir": os.path.join(paths["syrup"], "source", "src", "maple_syrup"),
                            "digest": sealed["syrup"]["package_digest"]},
            "masy": {"dir": os.path.join(paths["masy"], "source", "src", "masy"),
                     "digest": sealed["masy"]["package_digest"]}}

        origin = _origin_record(before["maple"], sealed["maple"]["object_id"])
        origin_path = run.write("origin.json", _canonical_bytes(origin))
        bundles = Path(store) / "objects" / "maple-bundles"
        export = bundles / f".export-{run.run_id}"
        if os.path.lexists(export):
            raise RuntimeRefusal("staging_exists", f"{export} already exists")
        body = _request_common(run, python, bundle_prefix, bundle_packages, limits)
        body.update(export={"project": os.path.join(paths["maple"], "source"),
                            "distribution": config.maple_distribution_metadata, "origin": str(origin_path),
                            "output": str(export)}, tool_module_sha256=tool_sha)
        exported = call_worker(run, "export", body, python=config.python, bootstrap=bootstrap, limits=limits)
        bundle_id = exported["stable"]["bundle"]["bundle_id"]
        if not isinstance(bundle_id, str) or not _HEX64.match(bundle_id):
            raise RuntimeRefusal("bundle_manifest_mismatch", f"export returned bundle id {bundle_id!r}")
        manifest_bytes = _read_bounded(export / "bundle-manifest.json", MAX_BUNDLE_MANIFEST_BYTES,
                                       "bundle_manifest_mismatch")
        manifest = check_bundle_manifest(
            manifest_bytes, bundle_id=bundle_id, tool_sha256=tool_sha, maple_object=sealed["maple"],
            maple_files=hashed["maple"][0], distribution=before["maple"]["installed_distribution_metadata"],
            origin=origin, package_digest=before["maple"]["package"]["digest_sha256"])
        final_bundle = bundles / bundle_id
        duplicate = os.path.lexists(final_bundle)
        if not duplicate:
            try:
                _rename_noreplace(export, final_bundle)
                _fsync_dir(bundles)
            except FileExistsError:
                duplicate = True
        if duplicate and _read_bounded(final_bundle / "bundle-manifest.json", MAX_BUNDLE_MANIFEST_BYTES,
                                       "bundle_collision") != manifest_bytes:
            raise RuntimeRefusal("bundle_collision", f"{final_bundle} exists with a different manifest; it was left "
                                 "untouched", retained_export=str(export))
        run.stage("bundle_published", "ok", bundle_id=bundle_id, reused=duplicate)
        paths = _object_paths(store, object_ids={k: sealed[k]["object_id"] for k in SOURCES}, bundle_id=bundle_id)

        body = _request_common(run, python, bundle_prefix, bundle_packages, limits)
        body["bundle"] = {"dir": paths["bundle"], "bundle_id": bundle_id, "cache": str(Path(paths["materialized"]).parent),
                          "target": paths["materialized"], "tool_module_sha256": tool_sha}
        call_worker(run, "materialize", body, python=config.python, bootstrap=bootstrap, limits=limits)
        if duplicate:
            _remove_own_export(export, run.run_id)
        materialized = Path(paths["materialized"])
        _require_real_dir(materialized, "materialized_overlay_invalid")
        if _read_bounded(materialized / "bundle-manifest.json", MAX_BUNDLE_MANIFEST_BYTES,
                         "materialized_overlay_invalid") != manifest_bytes:
            raise RuntimeRefusal("materialized_overlay_invalid", f"{materialized} manifest copy differs")
        prefix = runtime_sys_path(paths)
        check_import_roots(prefix)

        lock: dict[str, Any] = {
            "schema": LOCK_SCHEMA, "lock_id": "", "selection": SELECTION, "store": store,
            "sources": {kind: _source_lock_record(before[kind], sealed[kind]["object_id"]) for kind in SOURCES},
            "objects": {
                "sources": {kind: {key: sealed[kind][key] for key in (
                    "object_id", "path", "file_count", "total_bytes", "manifest_sha256", "package_digest",
                    "dist_info", "overlay_regeneration")} for kind in SOURCES},
                "maple_bundle": {"bundle_id": bundle_id, "path": paths["bundle"],
                                 "manifest_sha256": _sha256(manifest_bytes),
                                 "archive_sha256": manifest["archive"]["sha256"], "tool_module_sha256": tool_sha,
                                 "package_digest": manifest["project"]["package_source_digest"]["digest_sha256"]},
                "maple_materialized": {"bundle_id": bundle_id, "path": paths["materialized"],
                                       "manifest_sha256": _sha256(manifest_bytes)}},
            "distributions": {
                "maple": {"origin": "upstream maple_syrup.dependency_bundle overlay of the installed dist-info",
                          "installed_dist_info": before["maple"]["installed_distribution_metadata"],
                          "overlay_dir": os.path.join(paths["materialized"], "editable"),
                          "files": _editable_files(materialized / "editable")},
                **{SOURCES[kind][1]: {"origin": "MASY run-owned metadata overlay generated from the archived static "
                                                "pyproject.toml; not an installation",
                                      "overlay_dir": os.path.join(paths[kind], "editable"),
                                      "files": sealed[kind]["overlay_files"]} for kind in OVERLAY_KINDS}},
            "runtime": {"python": _python_stable(python), "bootstrap_sha256": _sha256(bootstrap.encode("utf-8")),
                        "sys_path_prefix": prefix,
                        "command": [config.python, "-I", "-B", "-X", "pycache_prefix=<run>/cache/pycache", "-c",
                                    "<BOOTSTRAP>", "<run>/request-<n>-runtime.json"],
                        "stable_environment": {key: worker_env("/")[key] for key in _worker.STABLE_ENV_KEYS},
                        "operational_fields": list(OPERATIONAL_FIELDS), "pycache_policy": PYCACHE_POLICY},
            "verification": {}, "limits": limits.record(), "interpretation": INTERPRETATION,
        }
        result = call_worker(run, "runtime", _runtime_request(lock, run, limits), python=config.python,
                             bootstrap=bootstrap, limits=limits)
        record["runtime_receipt"] = result["receipt"]
        lock["verification"] = result["stable"]

        after = observe_selection(config, masy_root=masy_root, launcher=launch["launcher"])
        record["after"] = after
        check_unchanged(before, after)
        if _python_stable(interpreter_identity(config.python, limits)) != _python_stable(python):
            raise RuntimeRefusal("environment_changed", "the interpreter changed during prepare")
        run.stage("after_observation", "ok")
        lock["lock_id"] = lock_body_id(lock)
        lock_path, reused = publish_lock(store, lock, run_id=run.run_id, limits=limits)
        run.stage("lock_published", "ok", lock_id=lock["lock_id"], reused=reused)
    except RuntimeRefusal as exc:
        run.stage("refused", exc.severity, code=exc.code, detail=exc.detail)
        record.update(verdict="incompatible" if exc.severity == "incompatible" else "refused",
                      refusal={"code": exc.code, "detail": exc.detail, "evidence": exc.evidence})
        return refusal_output(exc, run=run, evidence_path=run.evidence(record, limits.max_evidence_bytes))
    record.update(verdict="runtime_locked", lock_id=lock["lock_id"], lock_path=str(lock_path), lock_reused=reused)
    evidence = run.evidence(record, limits.max_evidence_bytes)
    return _output("runtime_locked", lock_id=lock["lock_id"], lock_path=str(lock_path), lock_reused=reused,
                   run_dir=str(run.dir), evidence_path=evidence, log_failures=run.log_failures,
                   selection="current_local_committed_head",
                   heads={kind: before[kind]["head"] for kind in SOURCES},
                   roots={kind: before[kind]["root"] for kind in SOURCES})


def _lock_protected(lock: dict[str, Any]) -> dict[str, str]:
    python = lock["runtime"]["python"]
    protected = {"lock sources.maple.root": lock["sources"]["maple"]["root"],
                 "lock sources.syrup.root": lock["sources"]["syrup"]["root"],
                 "lock runtime.python (resolved executable)": python["realpath"]}
    dist = lock["sources"]["maple"].get("installed_distribution_metadata", {}).get("path")
    if isinstance(dist, str):
        protected["lock sources.maple.distribution_metadata"] = dist
    if isinstance(python.get("venv"), dict) and isinstance(python["venv"].get("dir"), str):
        protected["lock runtime.python environment"] = python["venv"]["dir"]
    return protected


def verify(store: str, lock_ref: str, *, launch: dict[str, Any], max_lock_bytes: int | None = None,
           statvfs: Callable[[str], Any] = os.statvfs) -> dict[str, Any]:
    _check_launch(launch, store)
    check_store_path(store)
    validate_store_tree(store)
    lock, lock_path = load_lock_ref(store, lock_ref, max_lock_bytes or RuntimeLimits().max_lock_bytes)
    limits = RuntimeLimits.from_mapping(lock["limits"], "lock.limits")
    wrapper_roots = sorted({launch["masy_root"], lock["sources"]["masy"]["root"]})
    protected = _lock_protected(lock)
    for root in wrapper_roots:
        protected.update({f"wrapper {rel} ({root})": os.path.join(root, rel) for rel in ("src", "tools",
                                                                                        "pyproject.toml", ".git")})
    check_store_location(store, wrapper_roots=wrapper_roots, protected=protected)
    disk = disk_preflight(store, _run_dir_bytes(limits), limits, statvfs=statvfs)
    run = _Run(store, launch["run_id"], "verify").create()
    record: dict[str, Any] = {"operation": "verify", "run_id": run.run_id, "lock_id": lock["lock_id"],
                              "lock_path": str(lock_path), "launch": launch, "disk_preflight": disk,
                              "selection": "explicit_lock", "asserts_current": False}
    run.stage("lock_loaded", "ok", lock_id=lock["lock_id"])
    try:
        objects = lock["objects"]

        def verify_objects(stage: str) -> dict[str, Any]:
            summaries = {}
            for kind in SOURCES:
                entry = objects["sources"][kind]
                summary = verify_source_object(entry["path"], entry["object_id"], limits)
                for key in ("file_count", "total_bytes", "manifest_sha256", "package_digest"):
                    if summary[key] != entry[key]:
                        raise RuntimeRefusal("lock_runtime_mismatch", f"{kind} object {key} differs from the lock")
                summaries[kind] = summary
            for key, entry in (("maple_bundle", objects["maple_bundle"]),
                               ("maple_materialized", objects["maple_materialized"])):
                data = _read_bounded(Path(entry["path"]) / "bundle-manifest.json", MAX_BUNDLE_MANIFEST_BYTES,
                                     "object_tamper")
                if _sha256(data) != entry["manifest_sha256"]:
                    raise RuntimeRefusal("object_tamper", f"{entry['path']}/bundle-manifest.json differs from the lock")
            run.stage(stage, "ok")
            return summaries

        record["objects"] = verify_objects("objects_verified")
        bootstrap = sealed_bootstrap(objects["sources"]["masy"]["path"])
        if _sha256(bootstrap.encode("utf-8")) != lock["runtime"]["bootstrap_sha256"]:
            raise RuntimeRefusal("lock_runtime_mismatch", "sealed BOOTSTRAP hash differs from the lock")
        spelling = lock["runtime"]["python"]["spelling"]
        python = interpreter_identity(spelling, limits)
        record["python"] = python
        if _python_stable(python) != lock["runtime"]["python"]:
            raise RuntimeRefusal("environment_changed", "interpreter identity differs from the lock",
                                 differences=diff_observations({"python": lock["runtime"]["python"]},
                                                               {"python": _python_stable(python)}))
        check_import_roots(lock["runtime"]["sys_path_prefix"])
        result = call_worker(run, "runtime", _runtime_request(lock, run, limits), python=spelling,
                             bootstrap=bootstrap, limits=limits)
        record["runtime_receipt"] = result["receipt"]
        differences = diff_observations(lock["verification"], result["stable"])
        if differences:
            top = {item.split(".", 1)[0] for item in differences}
            code = "environment_changed" if top <= _ENVIRONMENT_KEYS else "lock_runtime_mismatch"
            raise RuntimeRefusal(code, f"fresh runtime identity differs from the lock at {differences[:50]}",
                                 differences=differences)
        verify_objects("objects_reverified")
    except RuntimeRefusal as exc:
        run.stage("refused", exc.severity, code=exc.code, detail=exc.detail)
        record.update(verdict="incompatible" if exc.severity == "incompatible" else "refused",
                      refusal={"code": exc.code, "detail": exc.detail, "evidence": exc.evidence})
        return refusal_output(exc, run=run, evidence_path=run.evidence(record, limits.max_evidence_bytes),
                              lock_id=lock["lock_id"])
    record["verdict"] = "runtime_verified"
    evidence = run.evidence(record, limits.max_evidence_bytes)
    return _output("runtime_verified", lock_id=lock["lock_id"], lock_path=str(lock_path), run_dir=str(run.dir),
                   evidence_path=evidence, log_failures=run.log_failures, selection="explicit_lock",
                   asserts_current=False)


# ------------------------------------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="masy_runtime", description="MASY-03 locked CPU runtime prepare/verify",
                                     allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", allow_abbrev=False, help="seal the current committed pair and publish a lock")
    p.add_argument("--config", required=True, help="runtime TOML (schema masy.runtime.v1)")
    v = sub.add_parser("verify", allow_abbrev=False, help="re-verify one explicit lock with a fresh child")
    v.add_argument("--store", required=True, help="absolute runtime store path")
    v.add_argument("--lock", required=True, help="lock id or <store>/locks/<id>.json")
    v.add_argument("--max-lock-bytes", type=int, default=None, help="bound for reading the lock file")
    return parser


def main(argv: list[str], *, launch: dict[str, Any]) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(load_runtime_config(args.config), launch=launch)
        else:
            if args.max_lock_bytes is not None:
                _check_limit("max_lock_bytes", args.max_lock_bytes, "--max-lock-bytes")
            result = verify(args.store, args.lock, launch=launch, max_lock_bytes=args.max_lock_bytes)
    except RuntimeRefusal as exc:
        result = refusal_output(exc)
    print(canonical_json(result))
    return result["exit_code"]
