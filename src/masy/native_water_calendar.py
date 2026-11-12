"""Native layered-water calendar adapter for the MASY coordinator (CPU, NumPy/Numba, native9f Dense, fixed material).

MASY stays temporal-only: this module owns NO physical formula. Every physical step is the actual SYRUP lifecycle
(`maple_syrup.water_lifecycle`: begin/evolve storm, enter/evolve/close interevent), the actual rainfall factories,
the actual `weather_pet` PET/terrain factories and the actual native MAPLE bed carried by the lifecycle. SYRUP never
imports MASY. See `docs/native_water_calendar.md` for the contract, limits and qualification status.

`NativeWaterSession` owns an ACCEPTED PAIR `(Coordinator, LifecycleState)` and at most one bounded unaccepted progress
slot. `Coordinator.step` advances its own cursor before deep-copying its return record, so the authoritative coordinator
is never stepped: a TRIAL coordinator is rebuilt with `Coordinator.resume` from the bounded accepted record, the real
adapter stages a pure physical child, the combined pair and evidence are fully built, and ONE reference assignment
publishes them. Any earlier failure discards the trial pair and leaves the accepted pair identical.

Not provided: wind, positive idle gaps, transpiration/growth gates, Windowed/CUDA/remapped beds, durable combined restart
(`native_state_checkpoint='not_provided'`), CLI/default adoption, production source locks.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import importlib.metadata
import math
import os
import stat
import sys
from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup import layered_event as ev
from maple_syrup import water_lifecycle as wl
from maple_syrup.pond_bed import steepest_d4_descent
from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule, rainfall_field

from masy import native_water_backend as nwb
from masy import native_water_checkpoint as nck
from masy.calendar import CalendarError
from masy.clock import ClockError, ClockLedger, decode_time
from masy.coordinator import AdapterCapabilities, AdapterOutcome, Coordinator, CoordinatorError
from masy.event_files import IntereventFile, RainEventFile
from masy.pet import PETError, weather_pet_library
from masy.scenario import Scenario
from masy.scenario_io import canonical_json
from masy.weather import LatitudeBinding, WeatherError

__all__ = [
    "ADAPTER_VERSION",
    "DEFAULT_SOURCE_PROFILE",
    "SOURCE_PROFILES",
    "NativeCalendarLimits",
    "NativeWaterCalendarError",
    "NativeWaterCalendarMemoryError",
    "NativeWaterSession",
    "LeafCheckpointLimits",
    "SourceFence", "delivery_source_fence",
    "StepResult",
    "CensusLimits",
    "begin_storm_bytes_formula",
    "event_extra_bytes_formula",
    "preparation_plan_bytes",
    "rain_forcing_bytes",
    "source_facts",
    "distribution_records",
    "join_latitude",
    "namespace_census",
    "source_census",
    "validate_source",
    "terrain_factor_from_topology",
]

ADAPTER_VERSION = "masy.native_water_calendar/1"
_F8 = 8
_D4_LABEL = np.array(["flat", "N", "E", "S", "W"])
_NEW_TOKEN = object()


class NativeWaterCalendarError(ValueError):
    def __init__(self, code: str, detail: str, *, terminal: bool = False, steps: int | None = None) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.terminal = terminal
        self.steps_total = steps


class NativeWaterCalendarMemoryError(NativeWaterCalendarError):
    def __init__(self, detail: str) -> None:
        super().__init__("memory_refused", detail)


def _refuse(code: str, detail: str, *, terminal: bool = False, steps: int | None = None) -> NativeWaterCalendarError:
    return NativeWaterCalendarError(code, detail, terminal=terminal, steps=steps)


# --- limits ------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class NativeCalendarLimits:
    """Bounded caller limits. `interevent_*` controls are separate from the storm control and have no default.
    `temporal_reserve_bytes` is the caller's EXPLICIT reserve for the coordinator's Python metadata (accepted record, trial replay,
    dispatch, returned copy); it is declared, never derived from serialized JSON size. `max_record_bytes` caps the serialized record."""

    interevent_max_dt_s: float
    interevent_min_dt_s: float
    interevent_max_retries: int
    storm_control: ev.LayeredEventControl
    caller_reserve_bytes: int
    temporal_reserve_bytes: int
    max_occurrences: int = 64
    max_dates_per_interevent: int = 400
    max_record_bytes: int = 1 << 20
    max_accepted_steps_per_occurrence: int = 100_000
    call_step_budget: int = 100_000

    def __post_init__(self) -> None:
        for name in ("interevent_max_dt_s", "interevent_min_dt_s"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise _refuse("invalid_limits", f"{name} must be a positive finite float")
        if self.interevent_min_dt_s > self.interevent_max_dt_s:
            raise _refuse("invalid_limits", "interevent_min_dt_s exceeds interevent_max_dt_s")
        if type(self.storm_control) is not ev.LayeredEventControl:
            raise _refuse("invalid_limits", "storm_control must be a genuine LayeredEventControl")
        for name in ("interevent_max_retries", "caller_reserve_bytes", "temporal_reserve_bytes", "max_occurrences",
                     "max_dates_per_interevent", "max_record_bytes", "max_accepted_steps_per_occurrence", "call_step_budget"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in ("caller_reserve_bytes", "temporal_reserve_bytes") else 1):
                raise _refuse("invalid_limits", f"{name} must be a plain int within range")

    def record(self) -> dict[str, Any]:
        return {"interevent_max_dt_s": self.interevent_max_dt_s.hex(), "interevent_min_dt_s": self.interevent_min_dt_s.hex(),
                "interevent_max_retries": self.interevent_max_retries,
                "storm_control": {k: (v.hex() if isinstance(v, float) else v)
                                  for k, v in dataclasses.asdict(self.storm_control).items()},
                "caller_reserve_bytes": self.caller_reserve_bytes, "temporal_reserve_bytes": self.temporal_reserve_bytes,
                "max_occurrences": self.max_occurrences,
                "max_dates_per_interevent": self.max_dates_per_interevent, "max_record_bytes": self.max_record_bytes,
                "max_accepted_steps_per_occurrence": self.max_accepted_steps_per_occurrence,
                "call_step_budget": self.call_step_budget}


# --- bounded cold source census ----------------------------------------------------------------------------------------
_HEX64 = frozenset("0123456789abcdef")


@dataclasses.dataclass(frozen=True)
class CensusLimits:
    max_files: int = 1024
    max_depth: int = 12
    max_path_len: int = 200
    max_file_bytes: int = 4 << 20
    max_total_bytes: int = 32 << 20
    chunk_bytes: int = 1 << 20

    def __post_init__(self) -> None:
        for name in dataclasses.asdict(self):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise _refuse("invalid_census_limits", f"{name} must be a positive int")

    def staging_bytes(self, n_files: int) -> int:
        """Cold staging charged by the create admission: one hash chunk + bounded per-node metadata (explicit, not measured)."""
        return int(self.chunk_bytes + 512 * max(int(n_files), 1))


def _safe_relative(path: Any, limits: CensusLimits) -> tuple[str, ...]:
    if type(path) is not str or not path or len(path) > limits.max_path_len or "\x00" in path or "\\" in path or path.startswith("/"):
        raise _refuse("census_path_unsafe", f"manifest path {path!r} is not a canonical relative path")
    parts = tuple(path.split("/"))
    if any(p in ("", ".", "..") for p in parts) or len(parts) > limits.max_depth:
        raise _refuse("census_path_unsafe", f"manifest path {path!r} is not a canonical relative path within the depth limit")
    return parts


def _manifest_digest(manifest: Any) -> str:
    return hashlib.sha256(canonical_json(sorted(dict(manifest).items())).encode("ascii")).hexdigest()


def _validate_manifest(manifest: Any, limits: CensusLimits) -> dict[str, tuple[str, ...]]:
    if not isinstance(manifest, Mapping) or not 0 < len(manifest) <= limits.max_files:
        raise _refuse("census_manifest_invalid", f"manifest must be a non-empty mapping of at most {limits.max_files} files")
    parsed = {}
    for path, digest in manifest.items():
        parsed[path] = _safe_relative(path, limits)
        if type(digest) is not str or len(digest) != 64 or not set(digest) <= _HEX64:
            raise _refuse("census_manifest_invalid", f"manifest digest of {path!r} is not 64 lowercase hex digits")
    return parsed


def _open_dir(root_fd: int, parts: tuple[str, ...]) -> int:
    """Open a directory by walking no-follow from the root descriptor; every intermediate descriptor is closed on every path."""
    fd = os.dup(root_fd)
    try:
        for part in parts:
            nxt = os.open(part, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = nxt
    except BaseException:
        os.close(fd)
        raise
    return fd


def _scan_tree(root_fd: int, limits: CensusLimits) -> tuple[set[str], list[str]]:
    """Streamed no-follow enumeration. The work list holds only relative part-tuples (no pending descriptors); each directory is opened when
    popped and closed in `finally`. Entries are consumed lazily from the scandir iterator and every global bound is enforced per entry, so at most
    one entry beyond a limit is ever discovered. Symlinks/special nodes are reported as unsafe."""
    files: set[str] = set()
    unsafe: list[str] = []
    pending: list[tuple[str, ...]] = [()]
    nodes = 0
    while pending:
        parts = pending.pop()
        fd = _open_dir(root_fd, parts)
        try:
            with os.scandir(fd) as it:
                for entry in it:
                    name = entry.name
                    if name == "__pycache__" or name.endswith(".pyc"):
                        continue
                    nodes += 1
                    if nodes > 4 * limits.max_files:
                        raise _refuse("census_limit", "source tree holds more nodes than the census limit")
                    if len(parts) + 1 > limits.max_depth:
                        raise _refuse("census_limit", f"{'/'.join((*parts, name))}: depth exceeds the census limit")
                    rel = "/".join((*parts, name))
                    if len(rel) > limits.max_path_len:
                        raise _refuse("census_limit", "a source path exceeds the census path-length limit")
                    st = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(st.st_mode):
                        pending.append((*parts, name))
                    elif stat.S_ISREG(st.st_mode):
                        if len(files) >= limits.max_files:
                            raise _refuse("census_limit", "source tree holds more files than the census limit")
                        files.add(rel)
                    else:
                        unsafe.append(rel)
        finally:
            os.close(fd)
    return files, unsafe


def _open_member(root_fd: int, parts: tuple[str, ...]) -> int:
    fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = nxt
        out = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
    finally:
        os.close(fd)
    return out


def source_census(root: Any, manifest: Any, limits: CensusLimits | None = None) -> dict[str, Any]:
    """Bounded cold census of `root` against `{canonical relative path: sha256}`. Manifest paths are validated and sizes are checked from
    metadata BEFORE any file is read; files are opened with no-follow descriptors and hashed in chunks; a size/mtime/inode change during
    the read is a race refusal. Never call per physical step."""
    limits = CensusLimits() if limits is None else limits
    parsed = _validate_manifest(manifest, limits)
    root = str(root)
    try:
        lst = os.lstat(root)
        if not stat.S_ISDIR(lst.st_mode):
            raise _refuse("census_root_unsafe", "census root is not a real directory (symlink or special refused)")
        root_fd = os.open(root, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as exc:
        raise _refuse("census_root_unsafe", f"census root cannot be opened without following links: {exc}") from exc
    changed, missing = [], []
    bytes_read = 0
    try:
        present, unsafe = _scan_tree(root_fd, limits)
        metadata_total = 0
        sizes = {}
        for rel in sorted(parsed):
            if rel not in present:
                continue
            fd = _open_member(root_fd, parsed[rel])
            try:
                st = os.fstat(fd)
            finally:
                os.close(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_size > limits.max_file_bytes:
                raise _refuse("census_limit", f"{rel}: not a regular file within {limits.max_file_bytes} B")
            metadata_total += st.st_size
            if metadata_total > limits.max_total_bytes:
                raise _refuse("census_limit", f"declared sources exceed {limits.max_total_bytes} B in total; nothing was read")
            sizes[rel] = st
        for rel in sorted(parsed):
            if rel not in present:
                missing.append(rel)
                continue
            fd = _open_member(root_fd, parsed[rel])
            try:
                before = os.fstat(fd)
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (sizes[rel].st_size, sizes[rel].st_mtime_ns, sizes[rel].st_ino):
                    raise _refuse("census_race", f"{rel} changed between metadata admission and read")
                h, seen = hashlib.sha256(), 0
                while True:
                    chunk = os.read(fd, limits.chunk_bytes)
                    if not chunk:
                        break
                    seen += len(chunk)
                    if seen > before.st_size:
                        raise _refuse("census_race", f"{rel} grew while being read")
                    h.update(chunk)
                after = os.fstat(fd)
                if seen != before.st_size or (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                    raise _refuse("census_race", f"{rel} changed while being read")
            finally:
                os.close(fd)
            bytes_read += seen
            if h.hexdigest() != manifest[rel]:
                changed.append(rel)
    finally:
        os.close(root_fd)
    unexpected = sorted(present - set(parsed))
    return {"root": root, "files": len(parsed), "changed": changed, "missing": missing, "unexpected": unexpected, "unsafe": sorted(unsafe),
            "bytes_read": bytes_read, "manifest_sha256": _manifest_digest(manifest),
            "clean": not (changed or missing or unexpected or unsafe)}


# --- pinned literals + fence --------------------------------------------------------------------------------------------
NATIVE_COMMIT = "9f466d2cfcd52991b24496cf67f4562d6268afa9"
NATIVE_PACKAGE_DIGEST = "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"
NATIVE_BUNDLE_ID = "55d1485a03c385240bab57edf7fba6cad5230628729b30b8a19d96d094ddaa5c"
SYRUP_PACKAGE_DIGEST = "d00ad390c2761cd2ef6c22fb5437d5fefc12e7bedb2d2752efc67c652a08b0ac"
DEFAULT_SOURCE_PROFILE = "c3_literal308"
#: Closed registry of NAMED source profiles: profile -> the literal SYRUP package digest it qualifies (native 9f, CPU NumPy/Numba, fixed material). The historical C3
#: literal-308 profile stays the default and its records/identities are unchanged; a new profile is registered here explicitly, never accepted from a caller digest.
SOURCE_PROFILES = MappingProxyType({DEFAULT_SOURCE_PROFILE: SYRUP_PACKAGE_DIGEST,
                                    "native9f_layered_factory_95ef_cpu_v1": "95efc6dcf38bbd599f746fd8089722660b26799de7ecafd12e5736bad080796d",
                                    # the SAME qualified full-water SYRUP 109-module digest under two distinct literal capabilities (`native_water_backend.PROFILE_CAPABILITY`)
                                    "native9f_layered_factory_afca_cpu_v1": "afcad53691b9d295ed560116d313419c08fc5ffe9b227a4bdd9c922f93c52671",
                                    "native9f_layered_factory_afca_cuda_v1": "afcad53691b9d295ed560116d313419c08fc5ffe9b227a4bdd9c922f93c52671"})


@dataclasses.dataclass(frozen=True)
class SourceFence:
    """Declarative EXPECTED record (never authority): validated against the actual preparation, imports and bounded manifests."""

    native_commit: str
    native_package_digest: str
    bundle_id: str
    syrup_package_digest: str
    masy_root: str
    syrup_root: str
    native_target: str
    masy_manifest: Any
    syrup_manifest: Any
    census_limits: CensusLimits = CensusLimits()
    profile: str = DEFAULT_SOURCE_PROFILE
    #: Explicit closed verified-delivery record (`masy.native_delivery.v1`). NOT a constructor argument (the historical dataclass signature is unchanged): it is bound only by the explicit factory
    #: `delivery_source_fence`, which validates/joins the closed record. None keeps the historical record/identity and the mandatory reference-bundle literal EXACTLY. The pure `_source_facts` joins
    #: scalars; the ACTUAL bundle-manifest bytes are verified in the cold `_validate_source` under its admission.
    delivery: Any = dataclasses.field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.profile) is not str or self.profile not in SOURCE_PROFILES:
            raise _refuse("invalid_fence", f"source profile {self.profile!r} is not one of {sorted(SOURCE_PROFILES)}")
        for name in ("masy_manifest", "syrup_manifest"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise _refuse("invalid_fence", f"{name} must be a mapping")
            object.__setattr__(self, name, MappingProxyType(dict(value)))

    @property
    def delivery_core(self) -> Any:
        if self.delivery is None:
            return None
        from masy import production_runtime as prod

        return prod.delivery_core(self.delivery)

    @property
    def masy_manifest_sha256(self) -> str:
        return _manifest_digest(self.masy_manifest)

    @property
    def syrup_manifest_sha256(self) -> str:
        return _manifest_digest(self.syrup_manifest)

    def record(self) -> dict[str, Any]:
        out = {"native_commit": self.native_commit, "native_package_digest": self.native_package_digest, "bundle_id": self.bundle_id,
               "syrup_package_digest": self.syrup_package_digest, "masy_root": self.masy_root, "syrup_root": self.syrup_root,
               "native_target": self.native_target, "masy_manifest_sha256": self.masy_manifest_sha256,
               "syrup_manifest_sha256": self.syrup_manifest_sha256}
        if self.profile != DEFAULT_SOURCE_PROFILE:  # the historical record is byte-identical for the default profile
            out["source_profile"] = self.profile
        if self.delivery is not None:  # additive: the historical record is byte-identical when no delivery is bound
            out["delivery_core"] = self.delivery_core
        return out

    def identity(self) -> str:
        base = (f"{ADAPTER_VERSION};native={self.native_commit[:12]}:{self.native_package_digest[:16]};"
                f"syrup={self.syrup_package_digest[:16]};masy={self.masy_manifest_sha256[:16]};syrupm={self.syrup_manifest_sha256[:16]}")
        out = base if self.profile == DEFAULT_SOURCE_PROFILE else f"{base};profile={self.profile}"
        if self.delivery is not None:
            core = self.delivery_core
            out += f";delivery={core['bundle_id'][:16]}:{core['manifest_sha256'][:16]}:{core['archive_sha256'][:16]}"
        return out


def delivery_source_fence(base: SourceFence, delivery: Any) -> SourceFence:
    """Explicit sibling constructor of a delivery-bound `SourceFence`: the bundle id is DERIVED from the closed, validated `masy.native_delivery.v1` record (there is no free id argument), the record is
    joined to the fence (target, masy/syrup roots, device) and the result is a genuine `SourceFence` whose historical constructor signature is unchanged. A record from `masy.production_runtime.resolve`
    (plain dict or its immutable proxy) is the only intended producer; the actual bundle bytes are verified cold in `_validate_source`."""
    if type(base) is not SourceFence or base.delivery is not None:
        raise _refuse("invalid_fence", "delivery_source_fence requires a genuine reference SourceFence without a delivery")
    from masy import production_runtime as prod  # stdlib-only; imported only for a delivery

    try:
        record = prod.validate_delivery(delivery)
    except Exception as exc:  # noqa: BLE001 - the closed validator's GateRefusal
        raise _refuse("invalid_fence", f"fence delivery is not the closed masy.native_delivery.v1 record: {getattr(exc, 'detail', exc)}") from None
    new = dataclasses.replace(base, bundle_id=record["bundle_id"])
    joins = (("target", new.native_target), ("masy_root", os.path.join(new.masy_root, "src", "masy")), ("syrup_root", os.path.join(new.syrup_root, "src", "maple_syrup")),
             ("device", nwb.PROFILE_CAPABILITY[new.profile]))
    for key, want in joins:
        if record[key] != want:
            raise _refuse("invalid_fence", f"fence delivery.{key} {record[key]!r} differs from the fence ({want!r})")
    object.__setattr__(new, "delivery", MappingProxyType(record))
    return new


def _under(path: str | None, root: str) -> bool:
    if not path:
        return False
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def _rel_under(path: str, root: str) -> str | None:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return None


def namespace_census(fence: SourceFence, *, max_modules: int = 4096) -> dict[str, Any]:
    return _namespace_census(nwb.CPU_V1, fence, max_modules=max_modules)


def _namespace_census(policy: Any, fence: SourceFence, *, max_modules: int = 4096) -> dict[str, Any]:
    """Every loaded `masy.*`/`maple_syrup.*` module (file, spec origin, package path) must be a member of its frozen manifest under the fenced
    root; every `maple.*` module must lie under the bundle `source/src`. Cold, bounded."""
    policy = nwb.policy_of(policy)  # exact-object selection BEFORE any module inspection
    names = [n for n in list(sys.modules) if n.split(".")[0] in ("masy", "maple_syrup", "maple")]
    if len(names) > max_modules:
        raise _refuse("namespace_limit", f"{len(names)} loaded modules in the fenced namespaces exceed {max_modules}")
    native_src = str(Path(fence.native_target) / "source" / "src")
    counts = {"masy": 0, "maple_syrup": 0, "maple": 0}
    for name in sorted(names):
        module = sys.modules.get(name)
        top = name.split(".")[0]
        root, manifest = {"masy": (fence.masy_root, fence.masy_manifest), "maple_syrup": (fence.syrup_root, fence.syrup_manifest),
                          "maple": (native_src, None)}[top]
        locations = []
        file = getattr(module, "__file__", None)
        if file:
            locations.append(file)
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if origin and origin not in ("built-in", "frozen"):
            locations.append(origin)
        locations.extend(list(getattr(module, "__path__", []) or []))
        if not locations and top != "maple":
            raise _refuse("namespace_unlocated", f"module {name} has no file/spec/path to associate with its source")
        for location in locations:
            rel = _rel_under(location, root)
            if rel is None:
                raise _refuse("module_origin_mismatch", f"module {name} location {location!r} lies outside fenced root {root!r}")
            if manifest is not None and Path(location).is_file() and rel not in manifest:
                raise _refuse("module_origin_mismatch", f"module {name} file {rel!r} is not a member of the frozen manifest")
        counts[top] += 1
    if not all(counts[k] for k in counts):
        raise _refuse("namespace_missing", f"a fenced namespace has no loaded module: {counts}")
    if policy.cupy_imported_refused():
        raise _refuse("gpu_import", "CuPy is imported; the CPU profile refuses GPU modules")
    return counts


def distribution_records(fence: SourceFence | None = None) -> dict[str, Any]:
    """Actual distribution metadata. `maple` must be located in the bundle `editable/` overlay when a fence is given. MASY/SYRUP are not
    installed distributions (recorded None honestly; their identity is the manifest + namespace census)."""
    records: dict[str, Any] = {}
    for name in ("maple", "maple-syrup", "masy"):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            records[name] = None
            continue
        records[name] = {"version": dist.version, "location": str(Path(str(dist.locate_file(""))).resolve())}
    if fence is not None:
        overlay = str((Path(fence.native_target) / "editable").resolve())
        found = records["maple"]
        if found is None or found["location"] != overlay:
            raise _refuse("distribution_mismatch", f"maple distribution {found!r} is not the bundle overlay {overlay!r}")
    return records


def source_facts(fence: SourceFence, preparation: Any) -> dict[str, Any]:
    """Pure metadata checks (types, pinned literals, sealed-record scalars). No census, hash, file read or recheck happens here."""
    return _source_facts(nwb.CPU_V1, fence, preparation)


def _source_facts(policy: Any, fence: SourceFence, preparation: Any) -> dict[str, Any]:
    policy = nwb.policy_of(policy)  # exact-object selection FIRST: a forged policy must not import any SYRUP module
    from maple_syrup import native_case_profile as ncp

    if type(fence) is not SourceFence:
        raise _refuse("invalid_fence", "fence must be a SourceFence")
    if nwb.policy_for_profile(fence.profile) is not policy:
        raise _refuse("profile_backend_mismatch", f"source profile {fence.profile!r} declares the {nwb.PROFILE_CAPABILITY[fence.profile]!r} capability, not this backend policy ({policy.name})")
    literals = [("native_commit", fence.native_commit, NATIVE_COMMIT), ("native_package_digest", fence.native_package_digest, NATIVE_PACKAGE_DIGEST),
                ("syrup_package_digest", fence.syrup_package_digest, SOURCE_PROFILES[fence.profile])]
    if fence.delivery is None:  # historical mandatory reference-bundle literal (unchanged); a delivery is verified below against actual bytes instead
        literals.insert(2, ("bundle_id", fence.bundle_id, NATIVE_BUNDLE_ID))
    for name, claimed, literal in literals:
        if claimed != literal:
            raise _refuse("source_fence_mismatch", f"fence {name} {claimed!r} is not the pinned literal {literal!r}")
    if ncp.NATIVE_COMMIT != NATIVE_COMMIT or ncp.PACKAGE_DIGEST != NATIVE_PACKAGE_DIGEST:
        raise _refuse("source_fence_mismatch", "the imported native_case_profile pins a different native source (not native9f)")
    if fence.delivery is not None:
        _check_delivery_facts(fence, ncp)
    if type(preparation) is not policy.prep_type():
        raise _refuse("preparation_required", f"a genuine {policy.prep_type().__module__.rsplit('.', 1)[-1]}.{policy.prep_type().__name__} is required, got {type(preparation).__name__}")
    try:
        facts = {"profile": preparation.profile_name, "package": preparation.package_digest, "bundle": preparation.bundle_id,
                 "syrup": preparation.syrup_package_digest, "case_identity": preparation.case_identity_sha256, "route": preparation.route,
                 "latitude": preparation.latitude_deg, "admission": preparation.admission}
    except policy.prep_errors() as exc:
        raise _refuse("preparation_required", f"unauthenticated preparation: {exc}") from exc
    if facts["profile"] != policy.prep_profile_name() or facts["package"] != fence.native_package_digest or facts["bundle"] != fence.bundle_id \
            or facts["syrup"] != fence.syrup_package_digest:
        raise _refuse("source_fence_mismatch", "the sealed preparation's profile/package/bundle/SYRUP digest differ from the fence")
    return facts


def _check_delivery_facts(fence: SourceFence, ncp: Any) -> None:
    """PURE scalar join (no file read, no hashing: `source_facts` stays metadata-only): the record's archive sha256 is the literal pinned in the genuine native_case_profile. The actual bundle bytes are
    verified in `_check_delivery_manifest` during the cold source admission."""
    if fence.delivery["archive_sha256"] != ncp.ARCHIVE_SHA256:
        raise _refuse("source_fence_mismatch", "delivery.archive_sha256 is not the pinned 9f archive sha256")


def _check_delivery_manifest(fence: SourceFence, reserve_bytes: int, available_bytes_fn: Any) -> None:
    """COLD verification of the ACTUAL materialized bundle manifest against the delivery record. Metadata first: `lstat` must describe a REGULAR file (a symlink/FIFO/device/directory is refused without
    opening it, so a malformed special file cannot block) of at most 16 MiB; then a fresh admitted availability reading (caller reserve + ONE fixed 1 MiB hash chunk) BEFORE the file is opened; it is
    opened no-follow/non-blocking and the opened `fstat` must again be regular with the same device/inode/size; streaming sha256 holds exactly one chunk at a time (`del part` before the next read); the
    final `fstat` must be unchanged. Called by `_validate_source` (create and resume); never cached as authority."""
    path = Path(fence.native_target) / "bundle-manifest.json"
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise _refuse("source_fence_mismatch", f"the materialized bundle manifest cannot be examined: {exc}") from None
    if not stat.S_ISREG(before.st_mode):
        raise _refuse("source_fence_mismatch", "the materialized bundle manifest is not a regular file (symlink/special refused; not opened)")
    size = int(before.st_size)
    if size > (16 << 20):
        raise _refuse("source_fence_mismatch", f"the materialized bundle manifest is {size} B, over the 16 MiB bound (checked before the body)")
    chunk = 1 << 20
    reading = available_bytes_fn()
    if type(reading) is bool or not isinstance(reading, (int, np.integer)) or int(reading) < 0:
        raise _refuse("invalid_availability", f"available_bytes_fn returned {reading!r}")
    if int(reading) < int(reserve_bytes) + chunk:
        raise NativeWaterCalendarMemoryError(f"the delivery manifest check needs {int(reserve_bytes) + chunk} B (reserve + one hash chunk) but its availability query reports {int(reading)} B")
    digest, total = hashlib.sha256(), 0
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
        try:
            held = os.fstat(fd)
            if not stat.S_ISREG(held.st_mode) or (held.st_dev, held.st_ino, held.st_size) != (before.st_dev, before.st_ino, before.st_size) or int(held.st_size) > (16 << 20):
                raise _refuse("source_fence_mismatch", "the opened bundle manifest is not the regular file that was examined (replaced or not regular)")
            while True:
                part = os.read(fd, chunk)
                if not part:
                    break
                total += len(part)
                if total > size:
                    raise _refuse("source_fence_mismatch", "the materialized bundle manifest grew while read")
                digest.update(part)
                del part  # exactly ONE chunk is live at a time: the previous buffer is released before the next read allocates
            after = os.fstat(fd)
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (held.st_dev, held.st_ino, held.st_size, held.st_mtime_ns):
                raise _refuse("source_fence_mismatch", "the materialized bundle manifest changed while read")
        finally:
            os.close(fd)
    except OSError as exc:
        raise _refuse("source_fence_mismatch", f"the materialized bundle manifest cannot be read: {exc}") from None
    if total != size or digest.hexdigest() != fence.delivery["manifest_sha256"]:
        raise _refuse("source_fence_mismatch", "the bytes of the materialized bundle manifest differ from the delivery record")


def preparation_plan_bytes(preparation: Any) -> int:
    """Whole-call requirement of the genuine preparation recheck (metadata only: `nd.plan_preparation`), reserve excluded."""
    return _preparation_plan_bytes(nwb.CPU_V1, preparation)


def _preparation_plan_bytes(policy: Any, preparation: Any) -> int:
    policy = nwb.policy_of(policy)
    try:
        return int(policy.plan_prep_bytes(preparation))
    except policy.prep_errors() as exc:
        raise _refuse("preparation_plan_failed", str(exc)) from exc
    except ValueError as exc:  # the CUDA planner's own typed placement/plan error
        if getattr(exc, "code", None) == "preparation_plan_failed":
            raise _refuse("preparation_plan_failed", str(exc)) from exc
        raise


def validate_source(fence: SourceFence, preparation: Any, *, reserve_bytes: int, available_bytes_fn: Any) -> dict[str, Any]:
    """Cold source/case capability check BEFORE any physical validation. Call only after the whole-call create admission."""
    return _validate_source(nwb.CPU_V1, fence, preparation, reserve_bytes=reserve_bytes, available_bytes_fn=available_bytes_fn)


def _validate_source(policy: Any, fence: SourceFence, preparation: Any, *, reserve_bytes: int, available_bytes_fn: Any, dev: Any = None,
                     device_available_bytes_fn: Any = None) -> dict[str, Any]:
    # The CPU census/distribution helpers are called through their PUBLIC module names (patchable by the existing tests); the namespace census is policy-bound.
    policy = nwb.policy_of(policy)
    facts = _source_facts(policy, fence, preparation)
    if fence.delivery is not None:  # cold admission of the actual delivery bytes (create AND resume reach this function)
        _check_delivery_manifest(fence, reserve_bytes, available_bytes_fn)
    limits = fence.census_limits
    reports = {}
    for label, root, manifest in (("masy", fence.masy_root, fence.masy_manifest), ("syrup", fence.syrup_root, fence.syrup_manifest)):
        report = source_census(root, manifest, limits)
        if not report["clean"]:
            raise _refuse("source_census_failed", f"{label} census: changed={report['changed'][:5]} missing={report['missing'][:5]} "
                          f"unexpected={report['unexpected'][:5]} unsafe={report['unsafe'][:5]}")
        reports[label] = {k: report[k] for k in ("files", "bytes_read", "manifest_sha256")}
    namespace = namespace_census(fence) if policy is nwb.CPU_V1 else _namespace_census(policy, fence)
    dists = distribution_records(fence)
    need = _preparation_plan_bytes(policy, preparation) + int(reserve_bytes)

    def fresh_reading() -> int:
        """Every nested query takes a fresh actual reading; insufficient -> typed memory refusal; otherwise the actual reading is returned.
        (A reading is not a reservation; nothing is cached and no native error text is parsed.)"""
        reading = available_bytes_fn()
        if type(reading) is bool or not isinstance(reading, (int, np.integer)) or int(reading) < 0:
            raise _refuse("invalid_availability", f"available_bytes_fn returned {reading!r}")
        if need > int(reading):
            raise NativeWaterCalendarMemoryError(f"preparation recheck needs {need} B (native plan + composed nested reserve) but its own "
                                                 f"availability query reports {int(reading)} B")
        return int(reading)

    try:
        if policy is nwb.CPU_V1:
            policy.recheck(preparation, reserve_bytes, fresh_reading)
        else:  # CUDA: the cold recheck needs the explicit device reserve and a device reader; the host figure is the same fresh-reading convention
            if dev is None or device_available_bytes_fn is None:
                raise _refuse("invalid_placement", "the CUDA cold recheck needs the explicit device placement and its device availability reader")
            policy.recheck(preparation, reserve_bytes, fresh_reading, dev.device_reserve_bytes, device_available_bytes_fn)
    except MemoryError as exc:
        raise NativeWaterCalendarMemoryError(f"{type(exc).__name__}: {exc}") from exc
    except policy.recheck_errors() as exc:
        raise _refuse("preparation_recheck_failed", str(exc)) from exc
    return {"case_identity_sha256": facts["case_identity"], "route": facts["route"], "latitude_deg": facts["latitude"],
            "latitude_binding": preparation.latitude_binding(), "census": reports, "namespace": namespace, "distributions": dists}


# --- metadata-only rain plan (mirrors of two small SYRUP formulas; equivalence is pinned by tests) ----------------------------
def event_extra_bytes_formula(lop: Any, n_intervals: int) -> int:
    """Pure arithmetic mirror of `water_lifecycle._event_extra_bytes` for a schedule of `n_intervals` float64 intervals
    (edges 8(n+1) B, intensity 8n B, rate 8n B). Tests compare it with the real function on real schedules and pin the real source text."""
    n_cells, n_layers, n_faces = int(lop.n_cells), int(lop.n_layers), int(lop.memory.n_faces)
    edges, intensity, rate = 8 * (n_intervals + 1), 8 * n_intervals, 8 * n_intervals
    extra = (16 * 8 + 1 + n_layers * 8) * n_cells + 4 * edges + 2 * intensity + 2 * rate + 4 * n_faces + 4096
    extra += ((n_layers + 3) * n_cells + n_faces) * 8
    if lop.context is not None:
        extra += n_cells * 8
    return int(extra)


def begin_storm_bytes_formula(lop: Any, n_intervals: int) -> int:
    """Pure arithmetic mirror of `water_lifecycle.required_begin_storm_bytes` (no schedule object)."""
    m = lop.memory
    return int(m.storm_required_bytes + m.owned_fixed_bytes + m.begin_copy_bytes
               + max(m.storm["build_bytes"], m.storm["initialization_bytes"], event_extra_bytes_formula(lop, n_intervals)))


def rain_forcing_bytes(n_intervals: int) -> int:
    """The adapter's own simultaneously live forcing leaves while the real schedule is built: its edges (n+1) and intensity (n) arrays, the
    schedule's private edges/intensity/rate copies (n+1, n, n) and the Python edge list (explicit 40 B per float incl. its pointer)."""
    return int(8 * ((n_intervals + 1) + n_intervals + (n_intervals + 1) + n_intervals + n_intervals) + 40 * (n_intervals + 1))


# --- latitude join -----------------------------------------------------------------------------------------------------
def join_latitude(masy_binding: LatitudeBinding, native_binding: dict[str, Any]) -> None:
    """Exact join of MASY's authored-case latitude record and the sealed native admission's association."""
    source = masy_binding.source
    pairs = (("path", str(Path(source["path"]).resolve()), str(Path(native_binding["case_yaml_path"]).resolve())),
             ("sha256", source["sha256"], native_binding["case_yaml_sha256"]),
             ("field", source["field"], native_binding["field"]),
             ("value", masy_binding.latitude_deg.hex(), float(native_binding["value"]).hex()))
    for name, left, right in pairs:
        if left != right:
            raise _refuse("latitude_join_mismatch", f"latitude {name}: MASY {left!r} != native {right!r}")


# --- terrain -----------------------------------------------------------------------------------------------------------
def terrain_factor_from_topology(topology: Any, latitude_deg: float) -> Any:
    """Actual `weather_pet.terrain_factor` from the HELD topology's real bed: steepest D4 drop/distance -> `atan`,
    D4 codes 1/2/3/4 (code 1 = `+row` neighbour, as `steepest_d4_descent` documents) -> labels N/E/S/W, inactive -> flat.
    Flat classification is the factory's strict `< 5 deg`; the hemisphere table is the factory's. Built once; row-major `(N,)`."""
    wp = weather_pet_library()
    ratio, code = steepest_d4_descent(topology.source_z_full, np.asarray(topology.active), float(topology.dx_m),
                                      topology.outlets)
    slope = np.arctan(np.asarray(ratio, dtype=np.float64)).reshape(-1)
    labels = _D4_LABEL[np.asarray(code, dtype=np.int64)].reshape(-1)
    return wp.terrain_factor(wp.aspect_codes_from_labels(labels), slope, float(latitude_deg))


def _host_reading(backend: Any, lop: Any) -> int:
    """The host availability reading of create/step. CPU: the historical `int(reader())` (behaviour preserved). CUDA: STRICT (non-Boolean integer >= 0, no coercion, typed refusal)."""
    if backend is nwb.CPU_V1:
        return int(backend.host_available_fn(lop)())
    from masy import native_water_cuda as nwcu

    try:
        return nwcu.strict_host_reading(backend.host_available_fn(lop), "host availability")
    except nwcu.CudaPlacementError as exc:
        raise _refuse(exc.code, exc.detail) from None


def _admit_device_total(backend: Any, lop: Any, dev: Any, need: int, what: str) -> None:
    """Fresh STRICT device reading vs the named device total (outside reserve already inside it) and the declared cap. Never a synthetic value."""
    if need > dev.device_cap_bytes:
        raise NativeWaterCalendarMemoryError(f"{what} needs {need} B on the device but the declared device_cap_bytes is {dev.device_cap_bytes}; nothing was done")
    try:
        reading = nck.read_availability(backend.device_available_fn(lop), "device")
    except nck.CheckpointError as exc:
        raise _refuse("invalid_availability", exc.detail) from None
    if need > reading:
        raise NativeWaterCalendarMemoryError(f"{what} needs {need} B on the device but only {reading} B are reported; process/context overhead excluded")


def _dev_record(dev: Any) -> dict[str, int]:
    from masy import native_water_cuda as nwc

    return nwc.placement_record(dev)


def _check_placement(backend: Any, dev: Any, preparation: Any = None) -> Any:
    """CPU: no placement is accepted. CUDA: an exact `CudaPlacement` whose device is the preparation's device (no other device is ever selected)."""
    if backend is nwb.CPU_V1:
        if dev is not None:
            raise _refuse("invalid_placement", "the CPU backend accepts no CUDA placement")
        return None
    from masy import native_water_cuda as nwc

    if type(dev) is not nwc.CudaPlacement:
        raise _refuse("invalid_placement", "the CUDA backend needs an explicit masy.native_water_cuda.CudaPlacement")
    if preparation is not None and getattr(preparation, "device_id", None) != dev.device_id:
        raise _refuse("device_mismatch", f"the preparation's device differs from the declared placement device {dev.device_id}")
    return dev


def adapter_capabilities(kind: str, fence: SourceFence, case_identity_sha256: str, latitude_deg: Any) -> AdapterCapabilities:
    """Pure capability declaration of the water-only adapter (historical signature, unchanged)."""
    return _adapter_capabilities(kind, fence, case_identity_sha256, latitude_deg, False)


def _adapter_capabilities(kind: str, fence: SourceFence, case_identity_sha256: str, latitude_deg: Any, leaf: bool) -> AdapterCapabilities:
    """Pure capability declaration (factored from the session so the restart loader can verify the recorded cursor identity BEFORE any physical read; the
    strings are byte-identical to the historical ones)."""
    identity = f"{fence.identity()};case={case_identity_sha256[:16]};lat={latitude_deg!r}"
    if leaf:  # explicit sibling record: the participant is part of the sealed adapter identity (the water-only strings below are untouched)
        return AdapterCapabilities(
            kind, f"masy.native_water_calendar.leaf_water.{kind}", (identity + ";participant=leaf_water_v1")[:512], True, "any_positive_duration", "proleptic_gregorian_dates", "not_provided",
            "staged_until_accepted", "native9f Dense layered water + LEAF participant (T/growth gates, interevent only); no wind/gaps; no durable combined restart (checkpoint refused)")
    return AdapterCapabilities(
        kind, f"masy.native_water_calendar.{kind}", identity[:512], True, "any_positive_duration",
        "proleptic_gregorian_dates", "not_provided", "staged_until_accepted",
        "native9f Dense CPU fixed-material layered water; no wind/gaps/T/growth; no durable combined restart")


def rain_schedule(occurrence: Any) -> RainfallSchedule:
    """The actual `RainfallSchedule` of one rain occurrence (exact `Fraction` edge accumulation converted once)."""
    intervals = occurrence.event.event.intervals
    cumulative, edges = Fraction(0), [0.0]
    for interval in intervals:
        cumulative += Fraction(interval.duration_s)
        edges.append(float(cumulative))
    return RainfallSchedule(np.asarray(edges, dtype=np.float64), np.asarray([i.intensity_mm_h for i in intervals], dtype=np.float64),
                            RainfallProvenance(kind="masy_rain_event", path=occurrence.file, sha256=occurrence.file_sha256, n_records=len(intervals)))


def storm_parts(lop: Any, limits: "NativeCalendarLimits", occurrence: Any, guard: float, schedule_fn: Any = None) -> tuple[RainfallSchedule, Any, Any]:
    return _storm_parts(nwb.CPU_V1, lop, limits, occurrence, guard, schedule_fn)


def _storm_parts(policy: Any, lop: Any, limits: "NativeCalendarLimits", occurrence: Any, guard: float, schedule_fn: Any = None) -> tuple[RainfallSchedule, Any, Any]:
    """(schedule, rainfall field, `LayeredEventControl`) of one rain occurrence: the ONE builder shared by `_run_rain` and the restart loader (which needs the
    codec's expected `StormSpec`)."""
    policy = nwb.policy_of(policy)  # BEFORE the schedule is built
    schedule = (rain_schedule if schedule_fn is None else schedule_fn)(occurrence)
    field = policy.rainfall_field(lop)
    sc = limits.storm_control
    control = ev.LayeredEventControl(sc.max_dt_s, sc.min_dt_s, sc.max_retries, limits.max_accepted_steps_per_occurrence, guard)
    return schedule, field, control


def check_scenario(scenario: Scenario, limits: "NativeCalendarLimits", profile_id: str, latitude: LatitudeBinding | None) -> bool:
    """The create-time scenario fences of the water-only profile (historical signature, unchanged)."""
    return _check_scenario(scenario, limits, profile_id, latitude, False)


def _check_scenario(scenario: Scenario, limits: "NativeCalendarLimits", profile_id: str, latitude: LatitudeBinding | None, leaf: bool) -> bool:
    """The create-time scenario fences (wind, positive gaps, T/growth gates, water profile, occurrence limit, weather/latitude need); returns `needs_pet`."""
    if len(scenario.occurrences) > limits.max_occurrences:
        raise _refuse("occurrence_limit", f"{len(scenario.occurrences)} occurrences exceed {limits.max_occurrences}")
    if scenario.gap_policy != "refuse":
        raise _refuse("unsupported_gap_policy", "positive idle gaps have no qualified water transition; gap_policy must be 'refuse'")
    needs_pet = False
    for occurrence in scenario.occurrences:
        event = occurrence.event
        if event.kind == "wind":
            raise _refuse("wind_unsupported", f"{occurrence.occurrence_id}: wind has no qualified shared-water transition here")
        if type(event) is RainEventFile and event.water_profile_id != profile_id:
            raise _refuse("profile_mismatch", f"{occurrence.occurrence_id}: water_profile_id {event.water_profile_id!r} != {profile_id!r}")
        if type(event) is IntereventFile:
            if not leaf and (event.processes["transpiration"] or event.processes["growth"]):
                raise _refuse("unsupported_gate", f"{occurrence.occurrence_id}: transpiration/growth gates are not part of this water-only profile")
            if leaf and event.processes["growth"] and not event.processes["transpiration"]:
                raise _refuse("unsupported_gate", f"{occurrence.occurrence_id}: growth enabled with transpiration off is not supported (no water-limited law; water-free growth is never invented)")
            needs_pet = needs_pet or event.processes["evaporation"] or (leaf and bool(event.processes["transpiration"]))
    if needs_pet and (latitude is None or scenario.weather is None):
        raise _refuse("weather_latitude_required", "evaporation/transpiration need scenario weather and the bound case latitude")
    return needs_pet


def _check_inputs(value: Any) -> dict | None:
    if value is None:
        return None
    if type(value) is not dict or not 0 < len(value) <= 64 or any(type(k) is not str or not 0 < len(k) <= 128 or type(v) is not str or len(v) != 64
                                                                    or not set(v) <= _HEX64 for k, v in value.items()):
        raise nck.CheckpointError("invalid_inputs", "inputs_sha256 must be None or a dict of at most 64 {name: 64-hex sha256}")
    return dict(sorted(value.items()))


# --- records -----------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class StepResult:
    status: str  # 'complete' | 'incomplete'
    occurrence_id: str
    record: dict[str, Any] | None
    evidence: dict[str, Any]


class _Pair:
    """The accepted triple: coordinator + actual lifecycle water + (LEAF participant only) the accepted `LeafBoundary`; replaced whole by ONE reference swap. `leaf` is None for every water-only session."""

    __slots__ = ("coordinator", "state", "serial", "leaf")

    def __init__(self, coordinator: Coordinator, state: Any, serial: int, leaf: Any = None) -> None:
        self.coordinator, self.state, self.serial, self.leaf = coordinator, state, serial, leaf


@dataclasses.dataclass(frozen=True)
class _Progress:
    key: tuple
    state: Any
    stage: str
    fragment: int
    steps_total: int
    calls: int


class _Stage:
    __slots__ = ("pair", "progress", "occurrence_index", "child", "pending", "evidence", "adopted", "child_leaf")

    def __init__(self, pair: _Pair, progress: _Progress | None, occurrence_index: int) -> None:
        self.pair, self.progress, self.occurrence_index = pair, progress, occurrence_index
        self.child = None
        self.child_leaf = None  # LEAF participant only: the staged child boundary (never visible before the swap)
        self.pending: _Progress | None = None
        self.evidence: dict[str, Any] = {}
        self.adopted = False


def _hex(value: float) -> str:
    return float(value).hex()


def _freeze(value: Any) -> Any:
    """Recursive deep immutable copy: dict -> MappingProxyType over an OWNED dict, list/tuple -> tuple."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


# --- session -----------------------------------------------------------------------------------------------------------
def _leaf_dt_for(scenario: Scenario, native_t: float) -> Any:
    """The exact aware datetime of a native solver time through the EXISTING exact clock/calendar bridge; an instant LEAF cannot represent exactly is a typed refusal (never rounded/restamped)."""
    from masy import leaf_calendar as lcal

    try:
        return lcal.exact_datetime(scenario.calendar.start, ClockLedger(scenario.native_epoch_s).global_elapsed(float(native_t)))
    except (lcal.LeafCalendarError, ClockError, CalendarError) as exc:
        raise _refuse("leaf_instant_not_exact", f"native time {float(native_t).hex()} has no exact LEAF datetime: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class LeafCheckpointLimits:
    """Explicit separate caps of the JOINT (water + biology) checkpoint; plain positive ints, no defaults, never silently raised. The physical codec keeps its own `CheckpointLimits`."""

    max_host_bytes: int
    max_device_bytes: int
    max_disk_bytes: int
    max_biology_bytes: int

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if type(v) is not int or v < 0:
                raise _refuse("invalid_limits", f"{f.name} must be a non-negative plain int")


def _leaf_bind(backend: Any, scenario: Scenario, lop: Any, state: Any, fence: SourceFence, op: Any, boundary: Any, initial: bool = True) -> tuple[SimpleNamespace, dict[str, Any]]:
    """Cold binding of the LEAF participant to THIS session (before any census/read; lazy imports: a water-only session never imports LEAF or the bridge). Returns (bundle, identity record)."""
    from masy import leaf_water_interval as lwi

    if type(op) is not lwi.IntervalOperator or op not in lwi._MINTED or type(boundary) is not lwi.LeafBoundary or boundary not in lwi._MINTED:
        raise _refuse("leaf_inputs", "leaf_op/leaf_boundary must be a genuine IntervalOperator / LeafBoundary of masy.leaf_water_interval")
    cuda = backend is not nwb.CPU_V1
    want_profile = "native9f_layered_factory_afca_cuda_v1" if cuda else "native9f_layered_factory_afca_cpu_v1"
    if fence.profile != want_profile:
        raise _refuse("leaf_profile", f"the LEAF participant is qualified only with the 109 AFCA water profile {want_profile!r}, not {fence.profile!r}")
    try:
        lwi.lsp.check_hot(op.receipt)
    except lwi.lsp.SourceProfileError as exc:
        raise _refuse("leaf_source", str(exc)) from None
    rc = op.receipt
    if op.lop is not lop or op.cuda != cuda or rc.device_kind != ("cuda" if cuda else "cpu"):
        raise _refuse("leaf_operator", "the LEAF operator is bound to a different lifecycle operator or backend capability")
    if (rc.digests["syrup"] != fence.syrup_package_digest or rc.digests["maple"] != fence.native_package_digest or not _under(str(rc.masy_root), str(fence.masy_root))
            or not _under(str(rc.syrup_root), str(fence.syrup_root)) or str(rc.native_root) != str(fence.native_target)):
        raise _refuse("leaf_source", "the LEAF source receipt does not match the session's source fence (SYRUP/MAPLE digests, MASY root, native target)")
    cal = boundary.calendar
    start = scenario.calendar.start
    if initial and (cal.cursor != start or boundary.generation != 0 or cal.pending_days != 0):
        raise _refuse("leaf_calendar_join", "the initial LEAF boundary must stand at the scenario calendar start (anchor == cursor == calendar.start, generation 0, nothing pending)")
    if cal.schedule.anchor != start:
        raise _refuse("leaf_calendar_join", "the LEAF schedule anchor must equal the scenario calendar start")
    if boundary.water_digest != str(state.digest) or boundary.mark != lwi._mark_of(op, state) or (initial and boundary.anchor != boundary.mark):
        raise _refuse("leaf_water_join", "the LEAF boundary was not made at this lifecycle state (state digest / continuity mark" + (" / anchor)" if initial else ")"))
    ident = {"receipt_profile": rc.profile, "leaf_package_digest": rc.digests["leaf"], "catalog_hash": str(op.catalog.hash), "species": list(op.table.species_ids),
             "geometry": {k: v for k, v in dict(op.geometry).items() if k != "valid_cell_mask"}, "mask": None if op.mask_ident is None else dict(op.mask_ident),  # the ACTUAL mask content identity (shape/dtype/sha256, hashed once at prepare)
             "leaf_budget_bytes": int(op.leaf_budget_bytes), "cadence_seconds": str(int(cal.schedule.cadence.total_seconds())), "n_plants": int(boundary.B_kg.shape[0]),
             "n_records": int(boundary.base_state.n)}  # STATIC identity only (config digest): the changing boundary tokens (CSR/area/water digests, generation, cursor) live in the biology member, so a restart after growth is valid
    est = lwi.required_session_bytes(op, boundary)  # pure arithmetic from shapes/nbytes
    cold = {"host": int(est["retained_host"]), "device": int(est["retained_device"])}  # the CSR export/hash staging is a CHECKPOINT-time cost (no digest is computed at create)
    return SimpleNamespace(op=op, cold=cold), ident  # NO retained initial boundary: the accepted generation lives only in the pair


class _Terminal:
    """ONE bounded terminal-failure token (no physical arrays): the typed error and cumulative work of the failed frame."""

    __slots__ = ("frame", "code", "detail", "steps_total", "refusals")

    def __init__(self, frame: tuple, code: str, detail: str, steps_total: int | None) -> None:
        self.frame, self.code, self.detail, self.steps_total, self.refusals = frame, code, detail, steps_total, 0


class NativeWaterSession:
    """Host adapter + accepted pair. Build with `NativeWaterSession.create`. Immutable after create: attribute assignment/deletion is
    refused and the sealed configuration is re-verified before every continuation (trusted private-attribute editing is out of scope)."""

    def __init__(self, token: object, **fields: Any) -> None:
        if token is not _NEW_TOKEN:
            raise _refuse("factory_only", "NativeWaterSession is built by NativeWaterSession.create")
        for name, value in fields.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("NativeWaterSession is immutable; its configuration is sealed at create")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("NativeWaterSession is immutable; its configuration is sealed at create")

    def _config_digest(self) -> str:
        return hashlib.sha256(canonical_json({
            "adapter": ADAPTER_VERSION, "fence": _thaw(self._fence_record), "scenario": self._scenario_digest, "profile": self.profile_id,
            "limits": self.limits.record(), "policy": dataclasses.asdict(self.policy), "lop": self.lop.sha256,
            "terrain": None if self.terrain is None else self.terrain.sha256, "retained_native_bytes": self.retained_native_bytes,
            "case_identity": self._exports["case_identity_sha256"], "exports_sha256": self._exports_sha256,
            "latitude": None if self.latitude is None else self.latitude.record(),
            **({} if getattr(self, "_dev", None) is None else {"device": _dev_record(self._dev)}),
            **self._participant_record()}).encode("ascii")).hexdigest()

    def _participant_record(self) -> dict[str, Any]:
        """Empty for a water-only session (the historical digest is untouched); the sealed participant identity otherwise."""
        if self._participant is nwb.WATER_ONLY_V1:
            return {}
        return {"participant": {"name": self._participant.name, **_thaw(self._leaf_identity)}}

    def _check_config(self) -> None:
        if self._config_digest() != self.config_token:
            raise _refuse("config_changed", "the sealed session configuration no longer matches its creation token; refused before any work")

    @staticmethod
    def plan_create(*, lop: Any, state: Any, preparation: Any, fence: SourceFence, limits: NativeCalendarLimits, needs_pet: bool,
                    retained_native_bytes: int = 0) -> dict[str, Any]:
        """Metadata-only named whole-call create requirement: reads sealed scalars and `nd.plan_preparation` only (no census, hash, file read, recheck).
        Disjoint cold phases take the max; retained native, temporal and caller reserves are added once and the nested recheck receives
        `retained + caller_reserve` so its own admission (`plan + reserve`) is bounded by this total."""
        facts = source_facts(fence, preparation)
        del facts
        m = lop.memory
        n_files = len(fence.masy_manifest) + len(fence.syrup_manifest)
        common = int(m.storm["resident_bytes"] + m.owned_fixed_bytes + m.reserves_bytes)  # B: live through every cold phase (lifecycle convention)
        scratch = {"preparation_recheck": preparation_plan_bytes(preparation),
                   "cold_census_staging": fence.census_limits.staging_bytes(n_files),
                   "terrain_staging": 12 * int(lop.n_cells) * _F8 if needs_pet else 0}
        whole_audit = int(wl.required_audit_bytes(lop, state))  # == B + audit scratch (B counted once)
        phases = {"preparation_recheck": common + scratch["preparation_recheck"], "cold_census_staging": common + scratch["cold_census_staging"],
                  "lifecycle_audit": whole_audit, "terrain_staging": common + scratch["terrain_staging"]}
        extras = {"caller_retained_native": int(retained_native_bytes), "temporal_metadata_reserve": limits.temporal_reserve_bytes,
                  "caller_reserve": limits.caller_reserve_bytes}
        return {"common_resident_bytes": common, "scratch": scratch, "phases": phases, "extras": extras,
                "nested_reserve_bytes": common + sum(extras.values()), "total": max(phases.values()) + sum(extras.values()),
                "scope": "B = lifecycle resident + owned fixed + existing reserves stays live; scratch phases are disjoint (max); native compile/config/coords/forcing, "
                         "caller-input containers and process/JIT/allocator overhead are excluded; not a peak measurement"}

    @classmethod
    def _assemble(cls, *, scenario: Scenario, lop: Any, exports: dict, fence: SourceFence, latitude: LatitudeBinding | None, limits: "NativeCalendarLimits",
                  profile_id: str, policy: Any, terrain: Any, retained_native_bytes: int, resume_receipt: Any = None, backend: Any = None, dev: Any = None,
                  participant: Any = None, leaf_bundle: Any = None, leaf_identity: Any = None) -> "NativeWaterSession":
        """Field assembly shared by `create` and `resume` (no accepted pair yet). `exports` holds exactly the keys `create` retains. `backend` is the sealed private backend
        policy handle (`_backend`; `None` = CPU_V1) — NOT `policy`, which is the storm quiet policy; it is excluded from the config digest."""
        backend = nwb.policy_of(backend)
        frozen = _freeze(exports)
        self = cls(_NEW_TOKEN, scenario=scenario, lop=lop, _exports=frozen, _exports_sha256=hashlib.sha256(canonical_json(_thaw(frozen)).encode("ascii")).hexdigest(),
                   _fence_record=_freeze(fence.record()), _scenario_digest=scenario.digest, fence=fence, latitude=latitude, limits=limits,
                   profile_id=profile_id, policy=policy, terrain=terrain, retained_native_bytes=retained_native_bytes, config_token="",
                   _stage=None, _progress=None, _terminal=None, _pair=None, _adapters=None, _resume_receipt=resume_receipt, _backend=backend, _dev=dev,
                   _participant=nwb.participant_of(participant), _leaf=leaf_bundle, _leaf_identity=_freeze(leaf_identity or {}))
        object.__setattr__(self, "config_token", self._config_digest())
        object.__setattr__(self, "_adapters", {kind: _Adapter(self, kind) for kind in sorted(scenario.kinds_present)})
        return self

    @classmethod
    def create(cls, *, scenario: Scenario, lop: Any, state: Any, preparation: Any, fence: SourceFence,
               latitude: LatitudeBinding | None, limits: NativeCalendarLimits, profile_id: str,
               quiet_policy: ev.LayeredQuietPolicy | None = None, retained_native_bytes: int = 0) -> "NativeWaterSession":
        """`preparation` is the genuine initial `NativeDensePreparation`; it is validated cold and NOT retained (only digests/ids are exported).
        `retained_native_bytes` declares caller-retained native case/graph bytes that stay alive and are charged in every later admission."""
        return cls._create_engine(nwb.CPU_V1, scenario=scenario, lop=lop, state=state, preparation=preparation, fence=fence, latitude=latitude, limits=limits,
                                  profile_id=profile_id, quiet_policy=quiet_policy, retained_native_bytes=retained_native_bytes)

    @classmethod
    def create_leaf(cls, *, scenario: Scenario, lop: Any, state: Any, preparation: Any, fence: SourceFence, latitude: LatitudeBinding | None, limits: NativeCalendarLimits, profile_id: str,
                    participant: Any, leaf_op: Any, leaf_boundary: Any, quiet_policy: ev.LayeredQuietPolicy | None = None, retained_native_bytes: int = 0) -> "NativeWaterSession":
        """EXPLICIT LEAF-participant sibling of `create` (CPU NumPy/Numba). `participant` must be exactly `nwb.LEAF_WATER_V1`; `leaf_op` is the bound `leaf_water_interval.IntervalOperator` (verified source
        receipt, table, catalog, geometry caps) and `leaf_boundary` the initial `LeafBoundary` made at `state`. The water-only `create` is unchanged. Checkpoint/resume of this session are explicitly refused
        until the joint artifact exists."""
        return cls._create_engine(nwb.CPU_V1, scenario=scenario, lop=lop, state=state, preparation=preparation, fence=fence, latitude=latitude, limits=limits, profile_id=profile_id,
                                  quiet_policy=quiet_policy, retained_native_bytes=retained_native_bytes, participant=participant, leaf_op=leaf_op, leaf_boundary=leaf_boundary)

    @classmethod
    def create_leaf_cuda(cls, *, scenario: Scenario, lop: Any, state: Any, preparation: Any, fence: SourceFence, latitude: LatitudeBinding | None, limits: NativeCalendarLimits, profile_id: str,
                         placement: Any, participant: Any, leaf_op: Any, leaf_boundary: Any, quiet_policy: ev.LayeredQuietPolicy | None = None,
                         retained_native_bytes: int = 0) -> "NativeWaterSession":
        """CUDA sibling of `create_leaf` (AUTHORED, never run on hardware here): the same checks under the sealed CUDA policy and the explicit CUDA source receipt."""
        return cls._create_engine(nwb.CUDA_V1, scenario=scenario, lop=lop, state=state, preparation=preparation, fence=fence, latitude=latitude, limits=limits, profile_id=profile_id,
                                  quiet_policy=quiet_policy, retained_native_bytes=retained_native_bytes, dev=placement, participant=participant, leaf_op=leaf_op, leaf_boundary=leaf_boundary)

    @classmethod
    def _create_engine(cls, _backend: Any, *, scenario: Scenario, lop: Any, state: Any, preparation: Any, fence: SourceFence,
                       latitude: LatitudeBinding | None, limits: NativeCalendarLimits, profile_id: str,
                       quiet_policy: ev.LayeredQuietPolicy | None = None, retained_native_bytes: int = 0, dev: Any = None,
                       participant: Any = None, leaf_op: Any = None, leaf_boundary: Any = None) -> "NativeWaterSession":
        backend = nwb.policy_of(_backend)  # selected by exact object identity BEFORE any other check or side effect
        part = nwb.participant_of(participant)  # exact sealed literal or None (water-only); nothing else
        dev = _check_placement(backend, dev)
        if type(scenario) is not Scenario:
            raise _refuse("invalid_scenario", "scenario must be a masy.scenario.Scenario")
        if type(limits) is not NativeCalendarLimits:
            raise _refuse("invalid_limits", "limits must be NativeCalendarLimits")
        if type(retained_native_bytes) is not int or retained_native_bytes < 0:
            raise _refuse("invalid_limits", "retained_native_bytes must be a non-negative plain int")
        if type(lop) is not wl.LifecycleOperator or type(state) is not wl.LifecycleState:
            raise _refuse("invalid_lifecycle", "a genuine LifecycleOperator and initialized LifecycleState are required")
        if not backend.backend_ok(lop.backend):
            raise _refuse("unsupported_backend", f"backend {lop.backend!r}; " + ("NumPy and Numba only" if backend is nwb.CPU_V1 else "the genuine CUDA lifecycle only"))
        dev = _check_placement(backend, dev, preparation)
        if state.role != "boundary" or state.payload.t_s.hex() != scenario.native_epoch_s.hex() or state.t_s.hex() != scenario.native_epoch_s.hex():
            raise _refuse("invalid_initial_state", "the lifecycle state must be a boundary exactly at the scenario native epoch")
        is_leaf = part is nwb.LEAF_WATER_V1
        if not is_leaf and (leaf_op is not None or leaf_boundary is not None):
            raise _refuse("participant_inputs_without_participant", "leaf_op/leaf_boundary require the explicit LEAF_WATER_V1 participant (create_leaf)")
        needs_pet = _check_scenario(scenario, limits, profile_id, latitude, is_leaf)
        leaf_bundle = leaf_identity = None
        if is_leaf:  # BEFORE any census/read: the participant must be bound to THIS lifecycle, fence, scenario calendar and initial state
            leaf_bundle, leaf_identity = _leaf_bind(backend, scenario, lop, state, fence, leaf_op, leaf_boundary)
        if backend is nwb.CPU_V1:
            plan = cls.plan_create(lop=lop, state=state, preparation=preparation, fence=fence, limits=limits, needs_pet=needs_pet,
                                   retained_native_bytes=retained_native_bytes)
        else:
            plan = cls._plan_create_cuda(backend, lop=lop, state=state, preparation=preparation, fence=fence, limits=limits, needs_pet=needs_pet,
                                         retained_native_bytes=retained_native_bytes, dev=dev)
        available = _host_reading(backend, lop)
        leaf_host = 0 if leaf_bundle is None else leaf_bundle.cold["host"]  # retained biology + base-state records + the digest staging the session itself holds
        if plan["total"] + leaf_host > available:  # BEFORE any census/read/hash/recheck/CSR export
            raise NativeWaterCalendarMemoryError(f"create needs {plan['total']} B ({plan['phases']}, extras {plan['extras']})" + (f" + {leaf_host} B of LEAF participant host bytes" if leaf_host else "")
                                                 + f" but only {available} B are reported; process/JIT overhead excluded")
        if backend is nwb.CPU_V1:
            exports = validate_source(fence, preparation, reserve_bytes=plan["nested_reserve_bytes"],
                                      available_bytes_fn=backend.host_available_fn(lop))
        else:
            _admit_device_total(backend, lop, dev, plan["device_total"] + (0 if leaf_bundle is None else leaf_bundle.cold["device"]), "create")
            exports = _validate_source(backend, fence, preparation, reserve_bytes=plan["nested_reserve_bytes"], available_bytes_fn=backend.host_available_fn(lop),
                                       dev=dev, device_available_bytes_fn=backend.device_available_fn(lop))
        if state.payload.bed is not preparation.bed or lop.context is not preparation.context:
            raise _refuse("bed_identity_mismatch", "the lifecycle state/operator are not bound to this preparation's native bed/context")
        if latitude is not None:
            join_latitude(latitude, exports["latitude_binding"])
        try:
            wl.audit_lifecycle(lop, state)
        except MemoryError as exc:
            raise NativeWaterCalendarMemoryError(str(exc)) from exc
        terrain = terrain_factor_from_topology(lop.topology, latitude.latitude_deg) if needs_pet else None
        policy = ev.LayeredQuietPolicy() if quiet_policy is None else quiet_policy
        exports = {k: exports[k] for k in ("case_identity_sha256", "route", "latitude_deg", "census", "namespace", "distributions")}
        self = cls._assemble(scenario=scenario, lop=lop, exports=exports, fence=fence, latitude=latitude, limits=limits, profile_id=profile_id,
                             policy=policy, terrain=terrain, retained_native_bytes=retained_native_bytes, backend=backend, dev=dev,
                             participant=part, leaf_bundle=leaf_bundle, leaf_identity=leaf_identity)
        coordinator = Coordinator(scenario, self._adapters, latitude=latitude)
        object.__setattr__(self, "_pair", _Pair(coordinator, state, 0, None if leaf_bundle is None else leaf_boundary))
        return self

    @classmethod
    def _plan_create_cuda(cls, backend: Any, *, lop: Any, state: Any, preparation: Any, fence: SourceFence, limits: NativeCalendarLimits, needs_pet: bool,
                          retained_native_bytes: int, dev: Any) -> dict[str, Any]:
        """CUDA create requirement from metadata only, with SEPARATE host and device figures (never summed): the host phases are the preparation recheck, the cold census
        staging, the terrain staging and the lifecycle audit's own host figure; the device phases are the preparation recheck and the lifecycle audit (`cuda_entry_plan`).
        Host extras (retained native, temporal, caller reserve) are counted once on the host; the caller's device reserve once on the device."""
        from masy import native_water_cuda as nwc

        _source_facts(backend, fence, preparation)
        n_files = len(fence.masy_manifest) + len(fence.syrup_manifest)
        audit_host, audit_dev = nwc.entry_figures(lop, "audit_boundary", parent=state)
        host_phases = {"preparation_recheck": _preparation_plan_bytes(backend, preparation), "cold_census_staging": fence.census_limits.staging_bytes(n_files),
                       "lifecycle_audit": audit_host, "terrain_staging": 12 * int(lop.n_cells) * _F8 if needs_pet else 0}
        dev_phases = {"preparation_recheck": nwc.plan_prep_device_bytes(preparation), "lifecycle_audit": audit_dev}
        extras = {"caller_retained_native": int(retained_native_bytes), "temporal_metadata_reserve": limits.temporal_reserve_bytes, "caller_reserve": limits.caller_reserve_bytes}
        device_total = max(dev_phases.values()) + dev.device_reserve_bytes
        return {"phases": host_phases, "device_phases": dev_phases, "extras": extras, "nested_reserve_bytes": sum(extras.values()), "total": max(host_phases.values()) + sum(extras.values()),
                "device_total": device_total,
                "scope": "CUDA create: host and device figures are separate (never summed); lifecycle figures are its own cuda_entry_plan; the device reserve is the caller's explicit "
                         "outside reserve; process/JIT/context/allocator overhead excluded; not a peak measurement"}

    @classmethod
    def create_cuda(cls, *, scenario: Scenario, lop: Any, state: Any, preparation: Any, fence: SourceFence, latitude: LatitudeBinding | None, limits: NativeCalendarLimits,
                    profile_id: str, placement: Any, quiet_policy: ev.LayeredQuietPolicy | None = None, retained_native_bytes: int = 0) -> "NativeWaterSession":
        """CUDA sibling of `create` (the CPU `create` is unchanged and stays CPU). `preparation` is the genuine initial `NativeDenseCudaPreparation`, `placement` the explicit
        `native_water_cuda.CudaPlacement`; the fence must be the registered CUDA profile."""
        return cls._create_engine(nwb.CUDA_V1, scenario=scenario, lop=lop, state=state, preparation=preparation, fence=fence, latitude=latitude, limits=limits,
                                  profile_id=profile_id, quiet_policy=quiet_policy, retained_native_bytes=retained_native_bytes, dev=placement)

    # ---- read-only views (copies; no live coordinator exposed) ----
    @property
    def exports(self) -> dict[str, Any]:
        """A freshly detached plain copy of the retained provenance: mutating it cannot change the sealed (deeply immutable) record or any token."""
        return _thaw(self._exports)

    @property
    def state(self) -> Any:
        return self._pair.state

    @property
    def leaf(self) -> Any:
        """The accepted `LeafBoundary` (immutable; None for a water-only session)."""
        return self._pair.leaf

    @property
    def participant(self) -> str:
        return self._participant.name

    @property
    def finished(self) -> bool:
        return self._pair.coordinator.finished

    def as_record(self) -> dict[str, Any]:
        return self._pair.coordinator.as_record()

    @property
    def realized(self) -> tuple[dict[str, Any], ...]:
        return self._pair.coordinator.realized

    def progress_summary(self) -> dict[str, Any] | None:
        progress = self._progress
        if progress is None or progress.key[0] != self._pair.serial:
            return None
        return {"stage": progress.stage, "fragment": progress.fragment, "steps_total": progress.steps_total, "calls": progress.calls,
                "occurrence_id": progress.key[2]}

    def terminal_summary(self) -> dict[str, Any] | None:
        t = self._terminal
        return None if t is None else {"code": t.code, "steps_total": t.steps_total, "refusals": t.refusals, "occurrence_id": t.frame[2]}

    def capabilities_of(self, kind: str) -> AdapterCapabilities:
        ex = self._exports
        return _adapter_capabilities(kind, self.fence, ex["case_identity_sha256"], ex["latitude_deg"], self._participant is nwb.LEAF_WATER_V1)

    def _rain_schedule(self, occurrence: Any, origin: float) -> RainfallSchedule:
        return rain_schedule(occurrence)

    # ---- resources ----
    def _requirements(self, pair: _Pair, progress: _Progress | None, occurrence: Any) -> dict[str, Any]:
        """The water requirement (below), plus - for the LEAF participant only - its own retained biology and the stage bytes of an interevent occurrence, HOST and DEVICE separate (before any allocation)."""
        req = self._requirements_water(pair, progress, occurrence)
        if self._participant is not nwb.LEAF_WATER_V1:
            return req
        est = self._leaf_est(pair)
        host_extra = int(est["retained_host"])
        dev_extra = int(est["retained_device"])
        stage_h = int(est["stage_host"]) if occurrence.kind == "interevent" else 0
        stage_d = int(est["stage_device"]) if occurrence.kind == "interevent" else 0
        req = dict(req)
        req["extras"] = {**req["extras"], "leaf_retained_biology": host_extra}
        req["phases"] = {**req["phases"], "leaf_stage": stage_h}
        req["total"] = max(req["phases"].values()) + sum(req["extras"].values())
        if self._backend is not nwb.CPU_V1:
            req["device_phases"] = {**req["device_phases"], "leaf_stage": stage_d}
            req["device_total"] = max(req["device_phases"].values()) + self._dev.device_reserve_bytes + dev_extra
        return req

    def _leaf_est(self, pair: _Pair) -> dict[str, int]:
        """The CURRENT biology estimate from the accepted `pair.leaf` (O(1) shape/nbytes arithmetic, no hash/transfer/rebuild): after a growth the maps can have more edges, so nothing is cached from creation."""
        from masy import leaf_water_interval as lwi

        return lwi.required_session_bytes(self._leaf.op, pair.leaf)

    def _requirements_water(self, pair: _Pair, progress: _Progress | None, occurrence: Any) -> dict[str, Any]:
        """Metadata-only named whole-call requirement (no record, hash, array or operator is created). Disjoint phases use max; the
        accepted parent beside a progress candidate, PET scratch, caller-retained native bytes and the explicit temporal/caller reserves
        are added once. The lifecycle's own phase figures already include its static/native/extra reserves."""
        if self._backend is not nwb.CPU_V1:
            return self._requirements_cuda(pair, progress, occurrence)
        lop, m, n = self.lop, self.lop.memory, int(self.lop.n_cells)
        parent = pair.state if progress is None else progress.state
        phases: dict[str, int] = {}
        if occurrence.kind == "rain":
            n_intervals = len(occurrence.event.event.intervals)  # pure arithmetic: no schedule array exists before admission
            extra = event_extra_bytes_formula(lop, n_intervals)
            if progress is None:
                if parent.role == "storm":
                    phases["admin_enter_interevent"] = int(wl.required_enter_interevent_bytes(lop, parent))
                    _, gm = wl._interevent_estimates(lop, parent.binding.op)
                    phases["admin_close"] = int(max(gm.audit_required_bytes, gm.resident_bytes + gm.reserves_bytes + gm.column_reserve_bytes
                                                    + m.audit_scratch_bytes) + m.owned_fixed_bytes)
                phases["begin_storm"] = int(begin_storm_bytes_formula(lop, n_intervals))
                during = m.storm_required_bytes + m.owned_fixed_bytes + extra
                after = m.storm["resident_bytes"] + m.call_parent_bytes + m.audit_scratch_bytes + m.reserves_bytes + m.owned_fixed_bytes
                phases["evolve_storm"] = int(max(during, after))
            else:
                phases["evolve_storm"] = int(wl.required_evolve_storm_bytes(lop, parent))
            phases["audit_storm_end"] = int(m.storm["resident_bytes"] + extra + m.reserves_bytes + m.audit_scratch_bytes + m.owned_fixed_bytes)
            pet = rain_forcing_bytes(n_intervals)  # live forcing arrays/metadata of the real schedule (no PET in rain)
        else:
            if progress is None:
                if parent.role == "storm":
                    phases["enter_interevent"] = int(wl.required_enter_interevent_bytes(lop, parent))
                    _, gm = wl._interevent_estimates(lop, parent.binding.op)
                else:
                    phases["enter_from_boundary"] = int(wl.required_enter_from_boundary_bytes(lop, parent))
                    _, _, gm = wl._boundary_estimates(lop)
            else:
                gm = parent.binding.gop.memory
            after = gm.resident_bytes + gm.reserves_bytes + gm.column_reserve_bytes + m.audit_scratch_bytes
            phases["evolve_interevent"] = int(max(int(gm.step_required_bytes), int(gm.audit_required_bytes), after)
                                              + m.owned_fixed_bytes + m.call_parent_bytes)
            phases["close_interevent"] = int(max(int(gm.audit_required_bytes), after) + m.owned_fixed_bytes)
            phases["audit_boundary_end"] = int(m.storm["resident_bytes"] + m.reserves_bytes + m.audit_scratch_bytes + m.owned_fixed_bytes)
            pet = 3 * n * _F8 if occurrence.event.processes["evaporation"] else n * _F8
        extras = {"old_accepted_parent_beside_progress": int(m.accepted_bytes) if progress is not None else 0, "pet_scratch_or_rain_forcing": pet,
                  "persistent_terrain": 8 * n if self.terrain is not None else 0,
                  "caller_retained_native": self.retained_native_bytes, "temporal_metadata_reserve": self.limits.temporal_reserve_bytes,
                  "caller_reserve": self.limits.caller_reserve_bytes}
        return {"phases": phases, "extras": extras, "total": max(phases.values()) + sum(extras.values())}

    def _requirements_cuda(self, pair: _Pair, progress: _Progress | None, occurrence: Any) -> dict[str, Any]:
        """CUDA whole-call pre-admission from the lifecycle's OWN `cuda_entry_plan` (metadata only; separate host and device figures; the entries re-admit themselves at run time).
        Phases that need a parent that does not exist before the call (the storm operator, the entered interevent) are covered by the entry that creates them (`begin_storm`,
        `enter_*`), whose plan includes their build stages; later phases of the same call are admitted by the lifecycle entry itself. Device extras: the PET packet and the
        rainfall mask; the caller's device reserve is counted once."""
        from masy import native_water_cuda as nwc

        lop, n = self.lop, int(self.lop.n_cells)
        parent = pair.state if progress is None else progress.state
        host: dict[str, int] = {}
        device: dict[str, int] = {}

        def add(name: str, entry: str, **kw: Any) -> None:
            host[name], device[name] = nwc.entry_figures(lop, entry, **kw)

        if occurrence.kind == "rain":
            n_intervals = len(occurrence.event.event.intervals)
            fb = nwc.rain_field_bytes(n)
            if progress is None:
                if parent.role == "storm":
                    add("admin_enter_interevent", "enter_interevent", parent=parent)
                host["begin_storm"], device["begin_storm"] = nwc.begin_storm_figures(lop, n_intervals)  # integer composer (no schedule object is passed to the strict real guard)
                device["begin_storm"] += fb["device_persistent"]  # the built field (8N) is alive BESIDE the begin_storm allocations
                host["rain_field_build"], device["rain_field_build"] = fb["host_build"], fb["device_build_peak"]  # mask N + ones 8N + where 8N before begin_storm
            else:
                add("evolve_storm", "evolve_storm", parent=parent)
            host_extra = rain_forcing_bytes(n_intervals)
        else:
            if progress is None:
                if parent.role == "storm":
                    add("enter_interevent", "enter_interevent", parent=parent)
                else:
                    add("enter_from_boundary", "enter_from_boundary")
            else:
                add("evolve_interevent", "evolve_interevent", parent=parent)
            host_extra = 3 * n * _F8 if occurrence.event.processes["evaporation"] else n * _F8
            device["evolve_interevent"] = device.get("evolve_interevent", 0) + n * _F8  # the GRID-sized PET upload is alive beside the evolve entry
        extras = {"pet_scratch_or_rain_forcing": host_extra, "persistent_terrain": 8 * n if self.terrain is not None else 0,
                  "caller_retained_native": self.retained_native_bytes, "temporal_metadata_reserve": self.limits.temporal_reserve_bytes, "caller_reserve": self.limits.caller_reserve_bytes}
        return {"phases": host, "device_phases": device, "extras": extras, "total": max(host.values()) + sum(extras.values()),
                "device_total": max(device.values()) + self._dev.device_reserve_bytes,
                "scope": "CURRENT-ENTRY pre-admission only (the entry/entries that start this call, the rainfall-field build peak 17N and the borrowed field 8N / GRID-sized PET 8N beside them); it is NOT "
                         "a forecast of every later phase of the call: each lifecycle entry re-admits itself freshly (host and device) when it runs; the device reserve is counted once; "
                         "CUDA memcpy/allocator overhead is excluded"}

    def required_call_bytes(self) -> dict[str, Any]:
        """Named whole-call requirement of the NEXT `step()` (metadata only; honest scope in the docs)."""
        pair = self._pair
        occurrence = self.scenario.occurrences[pair.coordinator.next_index]
        progress = self._progress
        if progress is not None and progress.key[0] != pair.serial:
            progress = None
        return self._requirements(pair, progress, occurrence)

    # ---- combined calendar + physical checkpoint (accepted occurrence boundaries only; docs/native_water_combined_checkpoint.md) ----
    @property
    def resume_receipt(self) -> dict[str, Any] | None:
        """Disclosure record of the `resume` that built this session (None for `create`): seal, boundary, changed resource limits, plan, conservatism."""
        return None if self._resume_receipt is None else _thaw(self._resume_receipt)

    def _boundary_role(self, pair: _Pair) -> tuple[str, str | None]:
        n = pair.coordinator.next_index
        if n == 0:
            return "boundary", None
        kind = self.scenario.occurrences[n - 1].kind
        return ("storm" if kind == "rain" else "boundary"), kind

    @staticmethod
    def _bed_of(state: Any) -> Any:
        return state.payload.water.bed if state.role == "storm" else state.payload.bed

    def _codec_extra(self) -> int:
        lim = self.limits
        return nck.codec_extra_reserve_bytes(max_record_bytes=lim.max_record_bytes, caller_reserve_bytes=lim.caller_reserve_bytes,
                                             temporal_reserve_bytes=lim.temporal_reserve_bytes, retained_native_bytes=self.retained_native_bytes)

    def _plan_checkpoint(self) -> tuple[dict[str, Any], Any]:
        pair = self._pair
        state = pair.state
        role, kind = self._boundary_role(pair)
        if state.role != role or (role == "storm" and getattr(state.payload, "completed", None) is not True):
            raise nck.CheckpointError("role_not_accepted", f"the accepted physical state is role {state.role!r} (completed="
                                      f"{getattr(state.payload, 'completed', None)!r}) but the cursor (last kind {kind!r}) requires a {role!r}"
                                      f"{' that is COMPLETED' if role == 'storm' else ''}")
        codec = self._backend.codec()
        extra_ev = max(int(state.binding.event.required_bytes) - int(state.binding.op.memory.required_bytes), 0) if role == "storm" else 0
        budget = self._backend.map_budget(self._backend.estimate_save(codec, self.lop, role, state, self._bed_of(state), extra_ev, self._codec_extra(), getattr(self, "_dev", None)))
        lim = self.limits
        n_files = len(self.fence.masy_manifest) + len(self.fence.syrup_manifest)
        plan = nck.plan_save(codec_budget=budget, max_record_bytes=lim.max_record_bytes, census_staging_bytes=self.fence.census_limits.staging_bytes(n_files),
                             caller_reserve_bytes=lim.caller_reserve_bytes, temporal_reserve_bytes=lim.temporal_reserve_bytes,
                             retained_native_bytes=self.retained_native_bytes)
        plan["role"] = role
        return plan, budget

    def plan_checkpoint(self) -> dict[str, Any]:
        """Metadata-only whole-call save requirement of the CURRENT accepted boundary (no record, hash, serialization or I/O)."""
        if self._participant is nwb.LEAF_WATER_V1:
            raise nck.CheckpointError("leaf_checkpoint_unsupported", "a LEAF-participant session is planned with plan_checkpoint_leaf (the joint artifact); nothing is planned")
        return self._plan_checkpoint()[0]

    def _fresh_census(self) -> dict[str, Any]:
        """Fresh bounded source census of both fenced trees + namespace + distributions; the path-free census summary must equal the one sealed at create/resume."""
        fence, out = self.fence, {}
        for label, root, manifest in (("masy", fence.masy_root, fence.masy_manifest), ("syrup", fence.syrup_root, fence.syrup_manifest)):
            report = source_census(root, manifest, fence.census_limits)
            if not report["clean"]:
                raise nck.CheckpointError("source_census_failed", f"{label} census: changed={report['changed'][:5]} missing={report['missing'][:5]} "
                                          f"unexpected={report['unexpected'][:5]} unsafe={report['unsafe'][:5]}")
            out[label] = {k: report[k] for k in ("files", "bytes_read", "manifest_sha256")}
        namespace_census(fence) if self._backend is nwb.CPU_V1 else _namespace_census(self._backend, fence)
        distribution_records(fence)
        if out != _thaw(self._exports["census"]):
            raise nck.CheckpointError("source_changed", "the fresh source census differs from the census sealed when this session was created/resumed")
        return out

    def _identity_sections(self) -> dict[str, Any]:
        """The path-free sections both `checkpoint` and `resume` derive from live objects (compared field by field at resume)."""
        ex = _thaw(self._exports)
        fence, lat = self.fence, self.latitude
        top = self.lop.topology
        sc = self.scenario
        return {
            "scenario": {"scenario_sha256": sc.digest, "source_sha256": sc.source_sha256, "scenario_id": sc.scenario_id,
                         "occurrences_sha256": hashlib.sha256(canonical_json([[o.occurrence_id, o.file_sha256, o.event.digest] for o in sc.occurrences]).encode("ascii")).hexdigest(),
                         "n_occurrences": len(sc.occurrences), "weather": sc.record()["weather"], "calendar": sc.calendar.record(),
                         "native_epoch_hex": sc.native_epoch_s.hex(), "gap_policy": sc.gap_policy},
            "latitude_path_free": None if lat is None else {"kind": lat.source["kind"], "field": lat.source["field"], "sha256": lat.source["sha256"],
                                                            "latitude_deg_hex": lat.latitude_deg.hex()},
            "source": {"identity": fence.identity(), "profile": fence.profile, "adapter_version": ADAPTER_VERSION, "native_commit": fence.native_commit,
                       "native_package_digest": fence.native_package_digest, "bundle_id": fence.bundle_id, "syrup_package_digest": fence.syrup_package_digest,
                       "masy_manifest_sha256": fence.masy_manifest_sha256, "syrup_manifest_sha256": fence.syrup_manifest_sha256, "census": ex["census"],
                       **({} if fence.delivery is None else {"delivery": fence.delivery_core})},  # additive: an old (reference) checkpoint has no `delivery` and can never match a delivery fence
            "case": {"case_identity_sha256": ex["case_identity_sha256"], "route": ex["route"], "profile_id": self.profile_id, "lop_sha256": self.lop.sha256,
                     "topology": {"identity_sha256": top.identity_sha256, "source_sha256": top.source_sha256, "topology_sha256": top.topology_sha256,
                                  "generation": int(top.generation)},
                     "quiet_policy": dataclasses.asdict(self.policy), "latitude_deg_hex": float(ex["latitude_deg"]).hex()},
            "limits": nck.split_limits(self.limits.record()),
        }

    def checkpoint(self, path: Any, *, checkpoint_limits: Any, host_available_bytes_fn: Any = None, disk_available_bytes_fn: Any = None,
                   inputs_sha256: dict | None = None, lineage: dict | None = None) -> "nck.CheckpointReceipt":
        """Seal the CURRENT accepted occurrence boundary (cursor + physical lifecycle) into a NEW artifact directory. Never mutates the session: the accepted pair,
        the clock and every ledger are untouched whatever happens. Refuses (typed, nothing written) on any pending `incomplete` progress, a terminal failure, an
        active stage, a role that is not the accepted role of the cursor (closed table), or a source that no longer matches. The receipt is returned only after
        the artifact is sealed and durable; a failure after the seal link is `CheckpointIndeterminate`."""
        if self._participant is nwb.LEAF_WATER_V1:  # BEFORE any plan/read/hash/file: a water-only artifact would silently drop the biology
            raise nck.CheckpointError("leaf_checkpoint_unsupported", "a LEAF-participant session is checkpointed with checkpoint_leaf (the joint water + biology artifact); the water-only checkpoint would drop the biology and is refused; nothing is written")
        pair = self._pair
        if self._stage is not None:
            raise nck.CheckpointError("stage_active", "a step is in progress")
        if self._progress is not None:
            raise nck.CheckpointError("pending_progress", "an incomplete occurrence is pending; its cumulative step budget cannot be reset, so a partial occurrence is never "
                                      "checkpointed, abandoned or restarted (finish it first)")
        if self._terminal is not None:
            raise nck.CheckpointError("terminal_failure", "the session has a sticky terminal failure; nothing is checkpointed")
        self._check_config()
        if type(checkpoint_limits) is not self._backend.checkpoint_limits_type():
            raise nck.CheckpointError("invalid_limits", "checkpoint_limits must be a maple_syrup.layered_checkpoint.CheckpointLimits")
        inputs = _check_inputs(inputs_sha256)
        if lineage is not None and (type(lineage) is not dict or len(canonical_json(lineage)) > 4096):
            raise nck.CheckpointError("invalid_lineage", "lineage must be a small dict (informational)")
        avail = self._backend.host_available_fn(self.lop) if host_available_bytes_fn is None else host_available_bytes_fn
        plan, budget = self._plan_checkpoint()  # metadata only; may refuse role_not_accepted BEFORE any admission/serialization
        if plan["total"] > int(checkpoint_limits.max_host_bytes) or plan["phases"]["physical_codec_whole"] > int(checkpoint_limits.max_host_bytes):
            raise nck.CheckpointMemoryError(f"checkpoint needs {plan['total']} B but limits.max_host_bytes is {checkpoint_limits.max_host_bytes}; nothing was read or written")
        if int(budget.serialized_bytes) > int(checkpoint_limits.max_serialized_bytes):
            raise nck.CheckpointMemoryError(f"the physical archive needs up to {budget.serialized_bytes} B but limits.max_serialized_bytes is "
                                            f"{checkpoint_limits.max_serialized_bytes}; nothing was written")
        nck.admit(plan["total"], avail, "combined checkpoint save")
        if self._backend is not nwb.CPU_V1:  # CUDA: the separate device figure (incremental + the caller's device reserve) vs the declared caps and a fresh device reading
            need_d = int(budget.device_required_with_reserves)
            if need_d > int(checkpoint_limits.max_device_bytes):
                raise nck.CheckpointMemoryError(f"checkpoint needs {need_d} B on the device but limits.max_device_bytes is {checkpoint_limits.max_device_bytes}; nothing was read or written")
            try:
                _admit_device_total(self._backend, self.lop, self._dev, need_d, "combined checkpoint save")
            except NativeWaterCalendarMemoryError as exc:
                raise nck.CheckpointMemoryError(exc.detail) from None
        disk_fn = disk_available_bytes_fn if disk_available_bytes_fn is not None else (lambda: nck.default_disk_available(path))
        nck.admit(plan["disk_bytes"], disk_fn, "combined checkpoint disk")
        # ---- only now: the temporal record, hashes and the fresh source census ----
        record = pair.coordinator.as_record()
        temporal = canonical_json(record).encode("ascii")
        if len(temporal) > self.limits.max_record_bytes:
            raise nck.CheckpointError("temporal_record_limit", f"accepted temporal record is {len(temporal)} B > {self.limits.max_record_bytes}")
        cursor_digest = hashlib.sha256(temporal).hexdigest()
        boundary = nck.expected_boundary(record)
        state = pair.state
        if (boundary["role"] != state.role or float(state.t_s).hex() != boundary["state_t_s_hex"]
                or (boundary["role"] == "storm" and state.payload.completed is not True)):
            raise nck.CheckpointError("time_bridge_mismatch", f"physical state (role {state.role!r}, t {float(state.t_s).hex()}) is not the cursor boundary "
                                      f"(role {boundary['role']!r}, native end {boundary['state_t_s_hex']})")
        self._fresh_census()
        sections = self._identity_sections()
        codec = self._backend.codec()
        ex = _thaw(self._exports)
        identity = nck.physical_identity(cursor_digest=cursor_digest, scenario_sha256=self.scenario.digest, next_index=record["next_index"],
                                         fence_identity=self.fence.identity(), case_identity_sha256=ex["case_identity_sha256"], state_t_s_hex=boundary["state_t_s_hex"])
        extra = self._codec_extra()

        def build_physical(physical_path: str) -> Any:
            quota_err, base_err = self._backend.physical_errors()

            try:
                return codec.save(physical_path, self.lop, state, identity, limits=checkpoint_limits,
                                  host_fn=lambda: nck.read_availability(avail, "codec host"), disk_fn=disk_fn, extra_reserve_bytes=extra,
                                  **self._backend.codec_kw(self.lop, getattr(self, "_dev", None)))
            except quota_err as exc:
                raise nck.CheckpointMemoryError(f"physical codec quota: {exc}") from None
            except base_err as exc:  # incl. its own Indeterminate: the combined artifact is still unsealed and wholly ours
                raise nck.CheckpointError("physical_save_refused", f"{type(exc).__name__}: {exc}") from None

        def make_manifest(rec: Any) -> dict:
            if rec.state_digest != state.digest or float(rec.t_s).hex() != boundary["state_t_s_hex"] or rec.role != boundary["role"]:
                raise nck.CheckpointError("physical_mismatch", "the sealed physical archive does not describe the accepted state")
            body = {"schema": nck.SCHEMA, "artifact_kind": nck.ARTIFACT_KIND,
                    "codec": {"id": codec.id, "profile": codec.profile, "device": codec.device, "physical_dir": nck.PHYSICAL_DIR},
                    "boundary": {**boundary, "next_index": record["next_index"], "finished": record["finished"]},
                    "temporal": {"file": nck.TEMPORAL_NAME, "sha256": cursor_digest, "bytes": len(temporal), "cursor_digest": cursor_digest},
                    "scenario": sections["scenario"], "latitude": {"record": None if self.latitude is None else self.latitude.record(),
                                                                 "path_free": sections["latitude_path_free"]},
                    "source": {**sections["source"], "informational": {"masy_root": self.fence.masy_root, "syrup_root": self.fence.syrup_root,
                                                                       "native_target": self.fence.native_target, **({} if self.fence.delivery is None else {"delivery": dict(self.fence.delivery)})}},
                    "case": sections["case"], "inputs": inputs,
                    "limits": {"behavioural": sections["limits"][0], "resource": sections["limits"][1]},
                    "capabilities": record["adapters"],
                    "physical": {"dir": nck.PHYSICAL_DIR, "manifest_sha256": rec.manifest_sha256, "state_digest": rec.state_digest,
                                 "physical_digest": rec.physical_digest, "role": rec.role, "revision": int(rec.revision),
                                 "t_s_hex": float(rec.t_s).hex(), "identity_sha256": hashlib.sha256(canonical_json(identity).encode("ascii")).hexdigest(),
                                 "profile": rec.profile, "budget": {"host_save_bytes": int(budget.host_save_bytes), "serialized_bytes": int(budget.serialized_bytes),
                                                                     "resident_old_bytes": int(budget.resident_old_bytes), "plan_total": int(plan["total"])}},
                    "writer": {"backend": self.lop.backend, "runtime": self._backend.runtime_record(self.lop, rec)},
                    "lineage": lineage}
            return nck.build_manifest(body)

        if self._backend is nwb.CPU_V1:
            out = nck.publish(path, build_physical=build_physical, temporal_bytes=temporal, make_manifest=make_manifest)
        else:
            out = nck.publish_cuda(path, build_physical=build_physical, temporal_bytes=temporal, make_manifest=make_manifest)
        man, rec = out["manifest"], out["physical"]
        return nck.CheckpointReceipt(os.path.abspath(os.fspath(path)), man["manifest_sha256"], rec.manifest_sha256, rec.state_digest, rec.physical_digest,
                                     cursor_digest, int(record["next_index"]), boundary["role"], boundary["state_t_s_hex"], len(temporal), out["manifest_bytes"], plan)

    # ---- JOINT (water + LEAF biology) checkpoint ----
    def _plan_checkpoint_leaf(self) -> tuple[dict[str, Any], Any]:
        """Composite save plan: metadata-only arithmetic (no hash, export, CSV text or population array exists) except the FIRST per-operator `metadata_cap` pass (an admitted actual empty-`PlantState`/S-sized LEAF call, then cached): the unchanged water plan, the biology encode peak (host/device separate, honest full-host staging) and the
        retained biology members R that stay alive while the physical codec runs. R is charged ONCE in the whole-call PREFIX phase (the physical codec phase beside the retained members) and is NOT
        added to the codec's own extra reserve afterwards: a fresh free reading already excludes the bytes R drew."""
        if self._participant is not nwb.LEAF_WATER_V1:
            raise nck.CheckpointError("leaf_session_required", "checkpoint_leaf needs a session created with create_leaf*")
        from masy import leaf_biology_codec_v3 as cdc

        if self._leaf.op.receipt.variant != "8da":  # the joint artifact (codec v3) serves ONLY the 8da family: typed refusal BEFORE any plan, metadata probe, allocation or file
            raise nck.CheckpointError("leaf_variant_unsupported", f"the joint checkpoint serves the 8da LEAF family only; this session's operator is {self._leaf.op.receipt.variant!r} (profile {self._leaf.op.receipt.profile!r})")
        plan, budget = self._plan_checkpoint()
        op, leaf = self._leaf.op, self._pair.leaf
        try:
            need = cdc.required_encode_bytes(op, leaf)
        except cdc.BiologyCodecError as exc:
            raise nck.CheckpointError("biology_" + exc.code, exc.detail) from None
        P, S, Q = int(leaf.B_kg.shape[0]), op.n_species, int(leaf.base_state.n)
        lim = cdc.member_limits(Q, P, S, cdc._max_name(op.catalog.species_ids), cdc.metadata_cap(op))
        r_cap = lim["arrays.bin"] + lim["base_plants.csv"] + lim["base_plants.metadata.json"] + lim["biology.json"]
        extras = plan["extras"]
        phases = dict(plan["phases"])
        phases["biology_encode"] = int(need["host"]) + extras
        phases["physical_codec_beside_retained_biology"] = int(phases["physical_codec_whole"]) + r_cap
        phases["combined_manifest"] = 3 * nck.MANIFEST_MAX_BYTES_LEAF + self.limits.max_record_bytes + extras + r_cap
        out = dict(plan, phases=phases, total=max(phases.values()), disk_bytes=int(plan["disk_bytes"]) + r_cap + nck.MANIFEST_MAX_BYTES_LEAF - nck.MANIFEST_MAX_BYTES,
                   nodes=plan["nodes"] + 1 + len(nck.BIOLOGY_MEMBERS), held_fds_engine=nck.MAX_HELD_FDS_LEAF)
        out["biology"] = {"n_base": Q, "n_live": P, "n_species": S, "member_caps": lim, "retained_member_bytes_cap": r_cap, "encode_host_bytes": int(need["host"]), "encode_device_bytes": int(need["device"]),
                          "retained_boundary": self._leaf_est(self._pair)}
        out["scope"] = plan["scope"] + "; biology: full-host P/S/CSR staging admitted before the allocation (no chunk/stream claim), retained members charged once in the prefix, no deep bed or water grid copy"
        return out, budget

    def plan_checkpoint_leaf(self) -> dict[str, Any]:
        """Metadata-only whole-call save requirement of the CURRENT accepted triple (no record, hash, export, serialization or I/O)."""
        return self._plan_checkpoint_leaf()[0]

    def checkpoint_leaf(self, path: Any, *, checkpoint_limits: Any, leaf_limits: Any, host_available_bytes_fn: Any = None, disk_available_bytes_fn: Any = None,
                        inputs_sha256: dict | None = None, lineage: dict | None = None) -> "nck.CheckpointReceipt":
        """Seal the CURRENT accepted occurrence boundary (cursor + physical lifecycle + LEAF biology) into a NEW joint artifact directory: `physical/` (the unchanged SYRUP codec), `biology/` (four
        closed members written by the publisher engine), `temporal.json`, the sealed manifest linked LAST. Never mutates the session; refuses (typed, nothing written) on pending progress, a terminal
        failure, an active stage, a non-accepted role, a source change or any admission failure; encode happens BEFORE any file exists."""
        if self._participant is not nwb.LEAF_WATER_V1:
            raise nck.CheckpointError("leaf_session_required", "checkpoint_leaf needs a session created with create_leaf*")
        pair = self._pair
        if self._stage is not None:
            raise nck.CheckpointError("stage_active", "a step is in progress")
        if self._progress is not None:
            raise nck.CheckpointError("pending_progress", "an incomplete occurrence is pending; a partial occurrence is never checkpointed")
        if self._terminal is not None:
            raise nck.CheckpointError("terminal_failure", "the session has a sticky terminal failure; nothing is checkpointed")
        self._check_config()
        if type(checkpoint_limits) is not self._backend.checkpoint_limits_type():
            raise nck.CheckpointError("invalid_limits", "checkpoint_limits must be a maple_syrup.layered_checkpoint.CheckpointLimits")
        if type(leaf_limits) is not LeafCheckpointLimits:
            raise nck.CheckpointError("invalid_limits", "leaf_limits must be LeafCheckpointLimits")
        inputs = _check_inputs(inputs_sha256)
        if lineage is not None and (type(lineage) is not dict or len(canonical_json(lineage)) > 4096):
            raise nck.CheckpointError("invalid_lineage", "lineage must be a small dict (informational)")
        from masy import leaf_biology_codec_v3 as cdc
        from masy import leaf_water_interval as lwi

        avail = self._backend.host_available_fn(self.lop) if host_available_bytes_fn is None else host_available_bytes_fn
        plan, budget = self._plan_checkpoint_leaf()  # metadata only
        bio = plan["biology"]
        if plan["total"] > int(leaf_limits.max_host_bytes) or plan["phases"]["physical_codec_whole"] > int(checkpoint_limits.max_host_bytes):
            raise nck.CheckpointMemoryError(f"joint checkpoint needs {plan['total']} B but leaf_limits.max_host_bytes is {leaf_limits.max_host_bytes} (physical limit {checkpoint_limits.max_host_bytes}); nothing was read or written")
        if bio["retained_member_bytes_cap"] > int(leaf_limits.max_biology_bytes):
            raise nck.CheckpointMemoryError(f"the biology members need up to {bio['retained_member_bytes_cap']} B but leaf_limits.max_biology_bytes is {leaf_limits.max_biology_bytes}; nothing was written")
        if int(budget.serialized_bytes) > int(checkpoint_limits.max_serialized_bytes):
            raise nck.CheckpointMemoryError(f"the physical archive needs up to {budget.serialized_bytes} B but limits.max_serialized_bytes is {checkpoint_limits.max_serialized_bytes}; nothing was written")
        if plan["disk_bytes"] > int(leaf_limits.max_disk_bytes):
            raise nck.CheckpointMemoryError(f"the joint artifact needs up to {plan['disk_bytes']} B of disk but leaf_limits.max_disk_bytes is {leaf_limits.max_disk_bytes}; nothing was written")
        nck.admit(plan["total"], avail, "joint checkpoint save")
        if self._backend is not nwb.CPU_V1:  # CUDA: the physical device stage + the biology device figure (nothing today) + the retained boundary, vs the declared caps and a fresh strict reading
            need_d = int(budget.device_required_with_reserves) + int(bio["encode_device_bytes"])
            if need_d > int(checkpoint_limits.max_device_bytes) or need_d > int(leaf_limits.max_device_bytes):
                raise nck.CheckpointMemoryError(f"the joint checkpoint needs {need_d} B on the device but the declared device limits are {checkpoint_limits.max_device_bytes}/{leaf_limits.max_device_bytes}; nothing was read or written")
            try:
                _admit_device_total(self._backend, self.lop, self._dev, need_d, "joint checkpoint save")
            except NativeWaterCalendarMemoryError as exc:
                raise nck.CheckpointMemoryError(exc.detail) from None
        disk_fn = disk_available_bytes_fn if disk_available_bytes_fn is not None else (lambda: nck.default_disk_available(path))
        nck.admit(plan["disk_bytes"], disk_fn, "joint checkpoint disk")
        # ---- only now: the temporal record, the source census, then ENCODE (cold: CSR export/hash) BEFORE any file exists ----
        record = pair.coordinator.as_record()
        temporal = canonical_json(record).encode("ascii")
        if len(temporal) > self.limits.max_record_bytes:
            raise nck.CheckpointError("temporal_record_limit", f"accepted temporal record is {len(temporal)} B > {self.limits.max_record_bytes}")
        cursor_digest = hashlib.sha256(temporal).hexdigest()
        boundary = nck.expected_boundary(record)
        state = pair.state
        if (boundary["role"] != state.role or float(state.t_s).hex() != boundary["state_t_s_hex"] or (boundary["role"] == "storm" and state.payload.completed is not True)):
            raise nck.CheckpointError("time_bridge_mismatch", f"physical state (role {state.role!r}, t {float(state.t_s).hex()}) is not the cursor boundary (role {boundary['role']!r}, native end {boundary['state_t_s_hex']})")
        self._fresh_census()
        leaf = pair.leaf
        if leaf.water_digest != str(state.digest) or leaf.mark != lwi._mark_of(self._leaf.op, state) or leaf.calendar.cursor != self._leaf_dt(float(state.t_s)):
            raise nck.CheckpointError("leaf_join_failed", "the accepted LEAF boundary does not join the accepted lifecycle state / clock")
        try:
            members = cdc.encode(self._leaf.op, leaf)
        except cdc.BiologyCodecError as exc:
            raise nck.CheckpointError("biology_" + exc.code, exc.detail) from None
        except lwi.IntervalMemoryError as exc:
            raise nck.CheckpointMemoryError(str(exc)) from None
        doc = json.loads(bytes(members["biology.json"]).decode("ascii"))
        sections = self._identity_sections()
        codec = self._backend.codec()
        ex = _thaw(self._exports)
        identity = nck.physical_identity(cursor_digest=cursor_digest, scenario_sha256=self.scenario.digest, next_index=record["next_index"], fence_identity=self.fence.identity(),
                                         case_identity_sha256=ex["case_identity_sha256"], state_t_s_hex=boundary["state_t_s_hex"])
        extra = self._codec_extra()  # UNCHANGED: the retained members are not added again (a fresh reading already excludes them)

        def build_physical(physical_path: str) -> Any:
            quota_err, base_err = self._backend.physical_errors()
            try:
                return codec.save(physical_path, self.lop, state, identity, limits=checkpoint_limits, host_fn=lambda: nck.read_availability(avail, "codec host"), disk_fn=disk_fn,
                                  extra_reserve_bytes=extra, **self._backend.codec_kw(self.lop, getattr(self, "_dev", None)))
            except quota_err as exc:
                raise nck.CheckpointMemoryError(f"physical codec quota: {exc}") from None
            except base_err as exc:
                raise nck.CheckpointError("physical_save_refused", f"{type(exc).__name__}: {exc}") from None

        static_sha = hashlib.sha256(canonical_json(_thaw(self._leaf_identity)).encode("ascii")).hexdigest()
        tokens = {k: doc[k] for k in ("water_digest", "mark", "anchor", "generation", "root_digest", "canopy_digest", "area_digest", "schedule", "geo_elapsed_days", "live_rows_sha256", "initial")}

        def make_manifest(rec: Any) -> dict:
            if rec.state_digest != state.digest or float(rec.t_s).hex() != boundary["state_t_s_hex"] or rec.role != boundary["role"]:
                raise nck.CheckpointError("physical_mismatch", "the sealed physical archive does not describe the accepted state")
            body = {"schema": nck.SCHEMA_LEAF, "artifact_kind": nck.ARTIFACT_KIND_LEAF,
                    "codec": {"id": codec.id, "profile": codec.profile, "device": codec.device, "physical_dir": nck.PHYSICAL_DIR},
                    "boundary": {**boundary, "next_index": record["next_index"], "finished": record["finished"]},
                    "temporal": {"file": nck.TEMPORAL_NAME, "sha256": cursor_digest, "bytes": len(temporal), "cursor_digest": cursor_digest},
                    "scenario": sections["scenario"], "latitude": {"record": None if self.latitude is None else self.latitude.record(), "path_free": sections["latitude_path_free"]},
                    "source": {**sections["source"], "informational": {"masy_root": self.fence.masy_root, "syrup_root": self.fence.syrup_root, "native_target": self.fence.native_target,
                                                                       **({} if self.fence.delivery is None else {"delivery": dict(self.fence.delivery)})}},
                    "case": sections["case"], "inputs": inputs, "limits": {"behavioural": sections["limits"][0], "resource": sections["limits"][1]}, "capabilities": record["adapters"],
                    "physical": {"dir": nck.PHYSICAL_DIR, "manifest_sha256": rec.manifest_sha256, "state_digest": rec.state_digest, "physical_digest": rec.physical_digest, "role": rec.role,
                                 "revision": int(rec.revision), "t_s_hex": float(rec.t_s).hex(), "identity_sha256": hashlib.sha256(canonical_json(identity).encode("ascii")).hexdigest(),
                                 "profile": rec.profile, "budget": {"host_save_bytes": int(budget.host_save_bytes), "serialized_bytes": int(budget.serialized_bytes),
                                                                     "resident_old_bytes": int(budget.resident_old_bytes), "plan_total": int(plan["total"])}},
                    "biology": {"dir": nck.BIOLOGY_DIR, "version": cdc.VERSION, "profile": doc["profile"], "variant": self._leaf.op.receipt.variant, "leaf_package_digest": doc["leaf_package_digest"],
                                "n_base": doc["n_base"], "n_live": doc["n_live"], "n_species": doc["n_species"],
                                "members": {n: {"sha256": hashlib.sha256(members[n]).hexdigest(), "bytes": len(members[n])} for n in nck.BIOLOGY_MEMBERS}, "static_identity_sha256": static_sha,
                                "boundary_tokens": tokens},
                    "writer": {"backend": self.lop.backend, "runtime": self._backend.runtime_record(self.lop, rec)}, "lineage": lineage}
            return nck.build_manifest_leaf(body)

        pub = nck.publish_leaf if self._backend is nwb.CPU_V1 else nck.publish_leaf_cuda
        out = pub(path, build_physical=build_physical, biology_members=members, temporal_bytes=temporal, make_manifest=make_manifest)
        man, rec = out["manifest"], out["physical"]
        return nck.CheckpointReceipt(os.path.abspath(os.fspath(path)), man["manifest_sha256"], rec.manifest_sha256, rec.state_digest, rec.physical_digest, cursor_digest, int(record["next_index"]),
                                     boundary["role"], boundary["state_t_s_hex"], len(temporal), out["manifest_bytes"], plan)

    # ---- restart: `resume` returns an UNADOPTED candidate session (never `create`, never a physical transition) ----
    @staticmethod
    def _resume_args(kw: dict[str, Any]) -> SimpleNamespace:
        a = SimpleNamespace(**kw)
        a._backend = nwb.policy_of(kw.get("_backend"))  # exact-object selection BEFORE any other check
        try:
            a._dev = _check_placement(a._backend, kw.get("_dev"), kw.get("preparation"))
        except NativeWaterCalendarError as exc:
            raise nck.CheckpointError(exc.code, exc.detail) from None
        if type(a.scenario) is not Scenario:
            raise nck.CheckpointError("invalid_scenario", "scenario must be a masy.scenario.Scenario")
        if type(a.limits) is not NativeCalendarLimits:
            raise nck.CheckpointError("invalid_limits", "limits must be NativeCalendarLimits")
        if type(a.fence) is not SourceFence:
            raise nck.CheckpointError("invalid_fence", "fence must be a SourceFence")
        if type(a.retained_native_bytes) is not int or a.retained_native_bytes < 0:
            raise nck.CheckpointError("invalid_limits", "retained_native_bytes must be a non-negative plain int")
        if type(a.lop) is not wl.LifecycleOperator or type(a.reference_state) is not wl.LifecycleState:
            raise nck.CheckpointError("invalid_lifecycle", "a genuine reference LifecycleOperator and its initial LifecycleState are required")
        if a.latitude_live is not None and type(a.latitude_live) is not LatitudeBinding:
            raise nck.CheckpointError("invalid_latitude", "latitude_live must be a masy.weather.LatitudeBinding or None")
        if type(a.checkpoint_limits) is not a._backend.checkpoint_limits_type():
            raise nck.CheckpointError("invalid_limits", "checkpoint_limits must be a maple_syrup.layered_checkpoint.CheckpointLimits")
        if a.expected_seal_sha256 is not None and not nck._is_sha(a.expected_seal_sha256):
            raise nck.CheckpointError("invalid_seal", "expected_seal_sha256 must be 64 lowercase hex digits")
        if a.quiet_policy is not None and type(a.quiet_policy) is not ev.LayeredQuietPolicy:
            raise nck.CheckpointError("invalid_policy", "quiet_policy must be a LayeredQuietPolicy")
        a.inputs = _check_inputs(a.inputs_sha256)
        if not a._backend.backend_ok(a.lop.backend):
            raise nck.CheckpointError("unsupported_backend", f"backend {a.lop.backend!r}; " + ("NumPy and Numba only" if a._backend is nwb.CPU_V1 else "the genuine CUDA lifecycle only"))
        ref = a.reference_state
        if ref.role != "boundary" or ref.payload.bed is not a.preparation.bed or a.lop.context is not a.preparation.context:
            raise nck.CheckpointError("reference_invalid", "the reference state/operator must be the INITIAL boundary bound to this preparation's native bed/context "
                                      "(the codec's expected components; the reference is never adopted)")
        a.avail = a._backend.host_available_fn(a.lop) if a.host_available_bytes_fn is None else a.host_available_bytes_fn
        return a

    @classmethod
    def _resume_open(cls, kw: dict[str, Any]) -> tuple[SimpleNamespace, "nck.Artifact", dict, dict]:
        a = cls._resume_args(kw)
        lim = a.limits
        try:
            (source_facts(a.fence, a.preparation) if a._backend is nwb.CPU_V1 else _source_facts(a._backend, a.fence, a.preparation))  # pure metadata (pinned literals / sealed scalars)
            leaf = getattr(a, "leaf_op", None) is not None
            needs_pet = _check_scenario(a.scenario, lim, a.profile_id, a.latitude_live, leaf)
        except NativeWaterCalendarError as exc:
            raise nck.CheckpointError(exc.code, exc.detail) from None
        a.needs_pet, a.leaf = needs_pet, leaf
        if leaf:
            from masy import leaf_water_interval as lwi

            if type(a.leaf_op) is not lwi.IntervalOperator or a.leaf_op not in lwi._MINTED or a.leaf_op.lop is not a.lop or type(getattr(a, "leaf_limits", None)) is not LeafCheckpointLimits:
                raise nck.CheckpointError("leaf_inputs", "resume_leaf needs a genuine IntervalOperator prepared for the REFERENCE lifecycle operator and LeafCheckpointLimits")
            if a.leaf_op.receipt.variant != "8da":
                raise nck.CheckpointError("leaf_variant_unsupported", f"the joint artifact serves the 8da LEAF family only; the given operator is {a.leaf_op.receipt.variant!r}")
            if a._backend is nwb.CPU_V1 and a.leaf_op.cuda or (a._backend is not nwb.CPU_V1 and not a.leaf_op.cuda):
                raise nck.CheckpointError("leaf_inputs", "the LEAF operator's backend capability differs from the resume backend")
        # S0: the explicitly admitted bounded manifest header
        art = nck._open_artifact(a.checkpoint_dir, members=a._backend.physical_members, admit_header=lambda n: nck.admit(3 * n + lim.caller_reserve_bytes, a.avail, "combined manifest header"),
                                temporal_cap=lim.max_record_bytes, leaf=leaf)
        try:
            man = art.manifest
            if a.expected_seal_sha256 is not None and man["manifest_sha256"] != a.expected_seal_sha256:
                raise nck.CheckpointError("seal_mismatch", "the manifest seal differs from the expected seal")
            a._backend.codec_for_manifest(man["codec"]["id"], man["codec"]["device"])
            if man["codec"]["profile"] != a._backend.codec_profile:
                raise nck.CheckpointError("codec_unavailable", f"profile {man['codec']['profile']!r} is not {a._backend.codec_profile!r}")
            n_next = man["boundary"]["next_index"]
            if n_next > len(a.scenario.occurrences) or man["boundary"]["finished"] != (n_next == len(a.scenario.occurrences)):
                raise nck.CheckpointError("scenario_mismatch", "the sealed cursor position does not fit the loaded scenario")
            # cheap joins BEFORE any heavy work (all path-free)
            probe = SimpleNamespace(scenario=a.scenario, fence=a.fence, latitude=a.latitude_live, lop=a.lop, policy=ev.LayeredQuietPolicy() if a.quiet_policy is None else a.quiet_policy,
                                    profile_id=a.profile_id, limits=lim,
                                    _exports=_freeze({"census": man["source"]["census"], "case_identity_sha256": a.preparation.case_identity_sha256,
                                                      "route": a.preparation.route, "latitude_deg": a.preparation.latitude_deg}))
            live = cls._identity_sections(probe)
            for key, code in (("scenario", "scenario_mismatch"), ("source", "source_mismatch"), ("case", "case_mismatch")):
                if live[key] != man[key] if key != "source" else {k: v for k, v in man["source"].items() if k != "informational"} != live["source"]:
                    diff = sorted(k for k in live[key] if live[key][k] != (man[key] if key != "source" else man["source"]).get(k))
                    raise nck.CheckpointError(code, f"{key} differs from the sealed artifact: {diff}")
            if live["latitude_path_free"] != man["latitude"]["path_free"]:
                raise nck.CheckpointError("latitude_mismatch", "the live case latitude (sha256/field/kind/value) differs from the sealed one (path is historical, not compared)")
            if live["limits"][0] != man["limits"]["behavioural"]:
                diff = sorted(k for k in live["limits"][0] if live["limits"][0][k] != man["limits"]["behavioural"].get(k))
                raise nck.CheckpointError("limits_mismatch", f"behavioural limits differ from the writer's: {diff}")
            if man["inputs"] != a.inputs:
                raise nck.CheckpointError("inputs_mismatch", "inputs_sha256 differs from the sealed input pins")
            for kind in sorted(a.scenario.kinds_present):
                want = _adapter_capabilities(kind, a.fence, a.preparation.case_identity_sha256, a.preparation.latitude_deg, leaf).record()
                if man["capabilities"].get(kind) != want:
                    raise nck.CheckpointError("adapter_identity_mismatch", f"the {kind} adapter identity differs from the sealed capabilities (writer source/case/latitude)")
            if set(man["capabilities"]) != set(a.scenario.kinds_present):
                raise nck.CheckpointError("adapter_identity_mismatch", "adapter kinds differ from the scenario")
            if not a._backend.runtime_matches_early(man["writer"], a.lop):
                raise nck.CheckpointError("runtime_mismatch", "writer backend/runtime differs (no cross-backend or cross-runtime migration)")
            if leaf:
                bm, op0 = man["biology"], a.leaf_op
                from masy import leaf_biology_codec_v3 as cdc

                if (bm["version"] != cdc.VERSION or bm["profile"] != op0.receipt.profile or bm["variant"] != op0.receipt.variant or bm["leaf_package_digest"] != op0.receipt.digests["leaf"]
                        or bm["n_species"] != op0.n_species):
                    raise nck.CheckpointError("biology_source_mismatch", "the artifact's biology schema/profile/variant/LEAF package differs from the live LEAF operator (no cross-profile restore)")
            role = man["boundary"]["role"]
            if (n_next == 0) != (man["boundary"]["last_kind"] is None) or role != ("storm" if man["boundary"]["last_kind"] == "rain" else "boundary"):
                raise nck.CheckpointError("role_not_accepted", "the sealed boundary role does not follow the closed accepted-role table")
            # S1: whole-call admission BEFORE the temporal body read/hash, census, recheck, native load
            m = a.lop.memory
            codec = a._backend.codec_for_manifest(man["codec"]["id"], man["codec"]["device"])
            extra_ev = 0
            if role == "storm":
                extra_ev = event_extra_bytes_formula(a.lop, len(a.scenario.occurrences[n_next - 1].event.event.intervals))
            lim_extra = nck.codec_extra_reserve_bytes(max_record_bytes=lim.max_record_bytes, caller_reserve_bytes=lim.caller_reserve_bytes,
                                                      temporal_reserve_bytes=lim.temporal_reserve_bytes, retained_native_bytes=a.retained_native_bytes)
            n_files = len(a.fence.masy_manifest) + len(a.fence.syrup_manifest)
            if a._backend is nwb.CPU_V1:
                budget = a._backend.map_budget(a._backend.estimate_load(codec, a.lop, role, a.preparation.bed, None, extra_ev, lim_extra, a._dev))
                plan = nck.plan_load(codec_budget=budget, common_bytes=int(m.storm["resident_bytes"] + m.owned_fixed_bytes + m.reserves_bytes),
                                     temporal_bytes=man["temporal"]["bytes"], preparation_recheck_bytes=_preparation_plan_bytes(a._backend, a.preparation),
                                     census_staging_bytes=a.fence.census_limits.staging_bytes(n_files),
                                     terrain_staging_bytes=12 * int(a.lop.n_cells) * _F8 if needs_pet else 0, audit_scratch_bytes=int(m.audit_scratch_bytes),
                                     caller_reserve_bytes=lim.caller_reserve_bytes, temporal_reserve_bytes=lim.temporal_reserve_bytes,
                                     retained_native_bytes=a.retained_native_bytes)
            else:
                from masy import native_water_cuda as nwcu

                # METADATA ONLY: no rainfall field / StormSpec / schedule / device array exists at this point (integer interval count)
                n_int = len(a.scenario.occurrences[n_next - 1].event.event.intervals) if role == "storm" else None
                budget = a._backend.map_budget(a._backend.estimate_load(codec, a.lop, role, a.preparation.bed, n_int, 0, lim_extra, a._dev))
                plan = nck.plan_load_cuda(codec_budget=budget, temporal_bytes=man["temporal"]["bytes"], preparation_recheck_bytes=_preparation_plan_bytes(a._backend, a.preparation),
                                          census_staging_bytes=a.fence.census_limits.staging_bytes(n_files), terrain_staging_bytes=12 * int(a.lop.n_cells) * _F8 if needs_pet else 0,
                                          caller_reserve_bytes=lim.caller_reserve_bytes, temporal_reserve_bytes=lim.temporal_reserve_bytes,
                                          retained_native_bytes=a.retained_native_bytes)
                dr = a._dev.device_reserve_bytes
                field = nwcu.rain_field_bytes(int(a.lop.n_cells)) if role == "storm" else {"device_build_peak": 0, "device_persistent": 0, "host_build": 0}
                stages = {"source_recheck": nwcu.plan_prep_device_bytes(a.preparation) + dr,
                          "load_with_borrowed_field": field["device_persistent"] + int(budget.device_required_with_reserves),  # PREFIX figure (whole-call cap/limit check BEFORE any allocation)
                          "load_after_field": int(budget.device_required_with_reserves)}  # the codec figure already holds the device reserve ONCE; used AFTER the field is drawn from free
                if role == "storm":
                    stages["field_build"] = field["device_build_peak"] + dr  # mask N + ones 8N + where 8N; the codec load has not started
                plan["device"] = stages
                plan["device_total"] = max(stages.values())
                plan["rain_field"] = field
                plan["scope"] = (plan["scope"] + "; DEVICE stages (separate from the host plan): source recheck, rainfall-field build peak (17N), then the codec load stage-prefix peak beside the "
                                 "borrowed persistent field (8N); the reserve is inside each figure once; the reference bed is resident (borrowed, never charged: readings are free memory); "
                                 "post-load audit and every later lifecycle entry re-admit freshly")
            if leaf:  # composite plan: the biology decode stage joins the phases (max); on CUDA its rebuild caps are a separate DEVICE stage
                bm, op0 = man["biology"], a.leaf_op
                breq = cdc.required_decode_bytes(op0, bm["n_base"], bm["n_live"], bm["n_species"], sum(v["bytes"] for v in bm["members"].values()))
                plan["phases"]["biology_decode"] = int(breq["host"]) + plan["extras"]
                plan["total"] = max(plan["phases"].values())
                plan["biology"] = {"host": int(breq["host"]), "device": int(breq["device"]), "n_base": bm["n_base"], "n_live": bm["n_live"], "members": {k: v["bytes"] for k, v in bm["members"].items()}}
                if a._backend is not nwb.CPU_V1:
                    plan["device"]["biology_decode_device"] = int(breq["device"]) + a._dev.device_reserve_bytes
                    plan["device_total"] = max(plan["device"].values())
            plan["role"], plan["codec_extra_reserve_bytes"], plan["event_extra_bytes"] = role, lim_extra, extra_ev
            a.budget, a.extra_ev, a.lim_extra = budget, extra_ev, lim_extra
            return a, art, man, plan
        except BaseException:
            art.close()
            raise

    @classmethod
    def plan_resume(cls, **kw: Any) -> dict[str, Any]:
        """Metadata-only whole-call load requirement (reads only the small admitted manifest header; same keyword arguments as `resume`)."""
        return cls._plan_resume_engine(nwb.CPU_V1, kw)

    @classmethod
    def plan_resume_cuda(cls, *, placement: Any, **kw: Any) -> dict[str, Any]:
        """CUDA sibling of `plan_resume` (same keyword arguments plus the explicit `CudaPlacement`; reads only the small admitted manifest header)."""
        return cls._plan_resume_engine(nwb.CUDA_V1, dict(kw, _dev=placement))

    @classmethod
    def resume_cuda(cls, *, checkpoint_dir: Any, scenario: Scenario, preparation: Any, lop: Any, reference_state: Any, fence: SourceFence, latitude_live: LatitudeBinding | None,
                    limits: NativeCalendarLimits, profile_id: str, checkpoint_limits: Any, placement: Any, quiet_policy: Any = None, retained_native_bytes: int = 0,
                    inputs_sha256: dict | None = None, expected_seal_sha256: str | None = None, host_available_bytes_fn: Any = None,
                    disk_available_bytes_fn: Any = None) -> "NativeWaterSession":
        """CUDA sibling of `resume`: a NEW UNADOPTED candidate session from the sealed CUDA V2 combined artifact. The reference `preparation`/`lop`/`reference_state` are the caller's
        fresh INITIAL CUDA setup (a second cold device bed coexisting with the restored one until the caller drops it; the load budget admits that coexistence); never `create`,
        replay, restamp or reset."""
        return cls._resume_engine(nwb.CUDA_V1, checkpoint_dir=checkpoint_dir, scenario=scenario, preparation=preparation, lop=lop, reference_state=reference_state, fence=fence,
                                  latitude_live=latitude_live, limits=limits, profile_id=profile_id, checkpoint_limits=checkpoint_limits, quiet_policy=quiet_policy,
                                  retained_native_bytes=retained_native_bytes, inputs_sha256=inputs_sha256, expected_seal_sha256=expected_seal_sha256,
                                  host_available_bytes_fn=host_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn, dev=placement)

    @classmethod
    def _plan_resume_engine(cls, _backend: Any, kw: dict[str, Any]) -> dict[str, Any]:
        kw = dict(kw, _backend=nwb.policy_of(_backend))
        kw.setdefault("quiet_policy", None), kw.setdefault("retained_native_bytes", 0), kw.setdefault("inputs_sha256", None)
        kw.setdefault("expected_seal_sha256", None), kw.setdefault("host_available_bytes_fn", None), kw.setdefault("disk_available_bytes_fn", None)
        a, art, man, plan = cls._resume_open(kw)
        art.close()
        return plan

    @classmethod
    def plan_resume_leaf(cls, *, leaf_op: Any, leaf_limits: Any, **kw: Any) -> dict[str, Any]:
        """Metadata-only whole-call load requirement of the JOINT artifact (reads only the small admitted manifest header)."""
        return cls._plan_resume_engine(nwb.CPU_V1, dict(kw, leaf_op=leaf_op, leaf_limits=leaf_limits))

    @classmethod
    def plan_resume_leaf_cuda(cls, *, placement: Any, leaf_op: Any, leaf_limits: Any, **kw: Any) -> dict[str, Any]:
        return cls._plan_resume_engine(nwb.CUDA_V1, dict(kw, _dev=placement, leaf_op=leaf_op, leaf_limits=leaf_limits))

    @classmethod
    def resume_leaf(cls, *, checkpoint_dir: Any, scenario: Scenario, preparation: Any, lop: Any, reference_state: Any, fence: SourceFence, latitude_live: LatitudeBinding | None,
                    limits: NativeCalendarLimits, profile_id: str, checkpoint_limits: Any, leaf_op: Any, leaf_limits: Any, quiet_policy: Any = None, retained_native_bytes: int = 0,
                    inputs_sha256: dict | None = None, expected_seal_sha256: str | None = None, host_available_bytes_fn: Any = None, disk_available_bytes_fn: Any = None) -> "NativeWaterSession":
        """Restore a JOINT artifact as a NEW UNADOPTED LEAF-participant session: new native bed/water through the unchanged V2 loader, a genuine interval-operator REBIND to the restored lifecycle
        operator, the public biology decode/rebuild, exact clock and static source/config joins, `Coordinator.resume`, ONE final triple. Never `create`, replay, reset, drying or fake zero rain. The
        reference `preparation`/`lop`/`reference_state` and `leaf_op` (prepared for that reference lop) are the caller's and are never adopted or mutated."""
        return cls._resume_engine(nwb.CPU_V1, checkpoint_dir=checkpoint_dir, scenario=scenario, preparation=preparation, lop=lop, reference_state=reference_state, fence=fence,
                                  latitude_live=latitude_live, limits=limits, profile_id=profile_id, checkpoint_limits=checkpoint_limits, quiet_policy=quiet_policy,
                                  retained_native_bytes=retained_native_bytes, inputs_sha256=inputs_sha256, expected_seal_sha256=expected_seal_sha256,
                                  host_available_bytes_fn=host_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn, leaf_op=leaf_op, leaf_limits=leaf_limits)

    @classmethod
    def resume_leaf_cuda(cls, *, checkpoint_dir: Any, scenario: Scenario, preparation: Any, lop: Any, reference_state: Any, fence: SourceFence, latitude_live: LatitudeBinding | None,
                         limits: NativeCalendarLimits, profile_id: str, checkpoint_limits: Any, placement: Any, leaf_op: Any, leaf_limits: Any, quiet_policy: Any = None,
                         retained_native_bytes: int = 0, inputs_sha256: dict | None = None, expected_seal_sha256: str | None = None, host_available_bytes_fn: Any = None,
                         disk_available_bytes_fn: Any = None) -> "NativeWaterSession":
        """CUDA sibling of `resume_leaf` (AUTHORED, never run on hardware here)."""
        return cls._resume_engine(nwb.CUDA_V1, checkpoint_dir=checkpoint_dir, scenario=scenario, preparation=preparation, lop=lop, reference_state=reference_state, fence=fence,
                                  latitude_live=latitude_live, limits=limits, profile_id=profile_id, checkpoint_limits=checkpoint_limits, quiet_policy=quiet_policy,
                                  retained_native_bytes=retained_native_bytes, inputs_sha256=inputs_sha256, expected_seal_sha256=expected_seal_sha256,
                                  host_available_bytes_fn=host_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn, dev=placement, leaf_op=leaf_op, leaf_limits=leaf_limits)

    @classmethod
    def resume(cls, *, checkpoint_dir: Any, scenario: Scenario, preparation: Any, lop: Any, reference_state: Any, fence: SourceFence,
               latitude_live: LatitudeBinding | None, limits: NativeCalendarLimits, profile_id: str, checkpoint_limits: Any,
               quiet_policy: Any = None, retained_native_bytes: int = 0, inputs_sha256: dict | None = None, expected_seal_sha256: str | None = None,
               host_available_bytes_fn: Any = None, disk_available_bytes_fn: Any = None) -> "NativeWaterSession":
        """Restore the sealed accepted boundary as a NEW unadopted session (the caller adopts it by one assignment). `preparation`/`lop`/`reference_state` are the
        caller-built INITIAL native reference (borrowed; used only as the codec's expected components and for the cold source recheck; never adopted, never mutated;
        drop them after adopting the candidate). Never calls `create`, never begins/enters/evolves/closes anything, never restamps a clock or replays rain/E:
        the cursor comes from `Coordinator.resume` (adapter-free replay) and the physical state from the unchanged SYRUP codec, joined by exact path-free identity."""
        return cls._resume_engine(nwb.CPU_V1, checkpoint_dir=checkpoint_dir, scenario=scenario, preparation=preparation, lop=lop, reference_state=reference_state, fence=fence,
                                  latitude_live=latitude_live, limits=limits, profile_id=profile_id, checkpoint_limits=checkpoint_limits, quiet_policy=quiet_policy,
                                  retained_native_bytes=retained_native_bytes, inputs_sha256=inputs_sha256, expected_seal_sha256=expected_seal_sha256,
                                  host_available_bytes_fn=host_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn)

    @classmethod
    def _resume_engine(cls, _backend: Any, *, checkpoint_dir: Any, scenario: Scenario, preparation: Any, lop: Any, reference_state: Any, fence: SourceFence,
                       latitude_live: LatitudeBinding | None, limits: NativeCalendarLimits, profile_id: str, checkpoint_limits: Any,
                       quiet_policy: Any = None, retained_native_bytes: int = 0, inputs_sha256: dict | None = None, expected_seal_sha256: str | None = None,
                       host_available_bytes_fn: Any = None, disk_available_bytes_fn: Any = None, dev: Any = None, leaf_op: Any = None, leaf_limits: Any = None) -> "NativeWaterSession":
        backend = nwb.policy_of(_backend)
        quota_err, base_err = backend.physical_errors()
        kw = dict(leaf_op=leaf_op, leaf_limits=leaf_limits, _backend=backend, checkpoint_dir=checkpoint_dir, scenario=scenario, preparation=preparation, lop=lop, reference_state=reference_state, fence=fence,
                  latitude_live=latitude_live, limits=limits, profile_id=profile_id, checkpoint_limits=checkpoint_limits, quiet_policy=quiet_policy,
                  retained_native_bytes=retained_native_bytes, inputs_sha256=inputs_sha256, expected_seal_sha256=expected_seal_sha256,
                  host_available_bytes_fn=host_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn, _dev=dev)
        a, art, man, plan = cls._resume_open(kw)
        try:
            lim, avail = a.limits, a.avail
            if plan["total"] > int(a.checkpoint_limits.max_host_bytes) or (a.leaf and plan["total"] > int(a.leaf_limits.max_host_bytes)):
                raise nck.CheckpointMemoryError(f"resume needs {plan['total']} B but the declared host limits are {a.checkpoint_limits.max_host_bytes}" + (f"/{a.leaf_limits.max_host_bytes}" if a.leaf else "") + "; nothing was read")
            if a.leaf and sum(plan["biology"]["members"].values()) > int(a.leaf_limits.max_biology_bytes):
                raise nck.CheckpointMemoryError(f"the biology members are {sum(plan['biology']['members'].values())} B but leaf_limits.max_biology_bytes is {a.leaf_limits.max_biology_bytes}; nothing was read")
            nck.admit(plan["total"], avail, "combined checkpoint resume")
            def dev_admit(need: int, what: str) -> None:
                try:
                    _admit_device_total(a._backend, a.lop, a._dev, int(need), what)
                except NativeWaterCalendarMemoryError as exc:
                    raise nck.CheckpointMemoryError(exc.detail) from None

            if a._backend is not nwb.CPU_V1:  # CUDA: the largest device stage vs max_device_bytes, the cap and a fresh STRICT reading; each later allocating stage re-reads below
                if int(plan["device_total"]) > int(a.checkpoint_limits.max_device_bytes) or (a.leaf and int(plan["device_total"]) > int(a.leaf_limits.max_device_bytes)):
                    raise nck.CheckpointMemoryError(f"resume needs {plan['device_total']} B on the device but limits.max_device_bytes is {a.checkpoint_limits.max_device_bytes}; nothing was read")
                dev_admit(plan["device_total"], "combined checkpoint resume")
            # ---- temporal body (bounded, pinned), then the adapter-free cursor replay with capability stand-ins ----
            raw = art.read_temporal(man["temporal"]["sha256"], man["temporal"]["bytes"], lim.max_record_bytes)
            record = nck.parse_canonical(raw, "temporal.json")
            historical = None
            if man["latitude"]["record"] is not None:
                try:
                    historical = LatitudeBinding.from_record(man["latitude"]["record"])
                except WeatherError as exc:
                    raise nck.CheckpointError("latitude_mismatch", f"the sealed latitude record is invalid: {exc}") from None
                if historical.record() != man["latitude"]["record"] or historical.latitude_deg.hex() != man["latitude"]["path_free"]["latitude_deg_hex"]:
                    raise nck.CheckpointError("latitude_mismatch", "the sealed latitude record is not self-consistent")
            standins = {k: _CapStandIn(_adapter_capabilities(k, a.fence, a.preparation.case_identity_sha256, a.preparation.latitude_deg, a.leaf)) for k in sorted(a.scenario.kinds_present)}
            try:
                trial = Coordinator.resume(a.scenario, record, standins, latitude=historical)
            except CoordinatorError as exc:
                raise nck.CheckpointError(f"cursor_{exc.code}", exc.detail) from None
            if trial.digest != man["temporal"]["cursor_digest"] or canonical_json(trial.as_record()).encode("ascii") != raw:
                raise nck.CheckpointError("temporal_mismatch", "the replayed cursor differs from the sealed cursor digest")
            boundary = nck.expected_boundary(record)
            if {**boundary, "next_index": record["next_index"], "finished": record["finished"]} != man["boundary"]:
                raise nck.CheckpointError("time_bridge_mismatch", "the sealed boundary section differs from the one derived from the cursor record")
            del trial, standins, record
            # ---- source / case: the cold recheck runs on the INITIAL reference preparation only (never on the restored state) ----
            try:
                _res = plan["extras"] + int(a.lop.memory.storm["resident_bytes"] + a.lop.memory.owned_fixed_bytes + a.lop.memory.reserves_bytes)
                if a._backend is nwb.CPU_V1:
                    exports = validate_source(a.fence, a.preparation, reserve_bytes=_res, available_bytes_fn=avail)
                else:
                    exports = _validate_source(a._backend, a.fence, a.preparation, reserve_bytes=plan["extras"], available_bytes_fn=avail, dev=a._dev,
                                               device_available_bytes_fn=a._backend.device_available_fn(a.lop))
                if a.latitude_live is not None:
                    join_latitude(a.latitude_live, exports["latitude_binding"])
            except NativeWaterCalendarMemoryError as exc:
                raise nck.CheckpointMemoryError(exc.detail) from None
            except NativeWaterCalendarError as exc:
                raise nck.CheckpointError(exc.code, exc.detail) from None
            if {k: exports["census"][k] for k in ("masy", "syrup")} != man["source"]["census"]:
                raise nck.CheckpointError("source_mismatch", "the fresh source census differs from the sealed census")
            terrain = terrain_factor_from_topology(a.lop.topology, historical.latitude_deg) if a.needs_pet else None
            policy = ev.LayeredQuietPolicy() if a.quiet_policy is None else a.quiet_policy
            ident = nck.physical_identity(cursor_digest=man["temporal"]["cursor_digest"], scenario_sha256=a.scenario.digest, next_index=man["boundary"]["next_index"],
                                          fence_identity=a.fence.identity(), case_identity_sha256=exports["case_identity_sha256"],
                                          state_t_s_hex=man["boundary"]["state_t_s_hex"])
            if hashlib.sha256(canonical_json(ident).encode("ascii")).hexdigest() != man["physical"]["identity_sha256"]:
                raise nck.CheckpointError("identity_mismatch", "the derived physical identity differs from the sealed one")
            role = man["boundary"]["role"]
            storm = None
            if role == "storm":
                if a._backend is not nwb.CPU_V1:  # the field is built ONLY now: after the strict host/device admission and the temporal/source/identity joins
                    nck.admit(plan["rain_field"]["host_build"] + plan["extras"], avail, "storm rainfall field build (host)")
                    dev_admit(plan["device"]["field_build"], "storm rainfall field build (device peak 17N)")
                occ = a.scenario.occurrences[man["boundary"]["next_index"] - 1]
                schedule, field, control = _storm_parts(a._backend, a.lop, lim, occ, float(occ.event.event.recession_guard_s))
                storm = a._backend.storm_spec(schedule, field, float.fromhex(man["boundary"]["origin_native_hex"]), control, policy)
            if a._backend is not nwb.CPU_V1:
                # POST-field INCREMENTAL check: the borrowed field (8N) has already been drawn from the free reading, so only the codec peak + the outside reserve (inside the codec figure once)
                # is needed now; the whole prefix (8N + codec peak) was compared with the free reading, the cap and max_device_bytes BEFORE the field was allocated (`dev_admit(plan["device_total"])`)
                dev_admit(plan["device"]["load_after_field"], "codec load after the field was drawn from free" if role == "storm" else "codec load")
            art.check_identity()
            codec = a._backend.codec_for_manifest(man["codec"]["id"], man["codec"]["device"])
            try:
                restored = codec.load(art.physical_path, a.lop, ident, role=role, storm=storm, reference_bed=a.preparation.bed, limits=a.checkpoint_limits,
                                      host_fn=lambda: nck.read_availability(avail, "codec host"), disk_fn=a.disk_available_bytes_fn, extra_reserve_bytes=a.lim_extra,
                                      **a._backend.codec_kw(a.lop, a._dev))
            except quota_err as exc:
                raise nck.CheckpointMemoryError(f"physical codec quota: {exc}") from None
            except base_err as exc:
                raise nck.CheckpointError("physical_load_refused", f"{type(exc).__name__}: {exc}") from None
            art.check_identity()
            state, new_lop = restored.state, restored.lop
            rec = restored.receipt
            if not a._backend.runtime_matches_late(man["writer"], rec):
                raise nck.CheckpointError("runtime_mismatch", "the codec's fresh runtime record differs from the sealed writer runtime (no cross-runtime migration)")
            if (state.role != role or float(state.t_s).hex() != man["boundary"]["state_t_s_hex"] or (role == "storm" and state.payload.completed is not True)
                    or state.digest != man["physical"]["state_digest"] or rec["state_digest"] != man["physical"]["state_digest"]
                    or rec["physical_digest"] != man["physical"]["physical_digest"] or int(state.revision) != man["physical"]["revision"]):
                raise nck.CheckpointError("physical_mismatch", "the restored physical state differs from the sealed role/clock/digests")
            try:
                wl.audit_lifecycle(new_lop, state)
            except MemoryError as exc:
                raise nck.CheckpointMemoryError(str(exc)) from None
            leaf_bundle = leaf_ident = leaf_boundary = None
            if a.leaf:
                leaf_boundary, leaf_bundle, leaf_ident = cls._restore_leaf(a, art, man, state, new_lop, avail, dev_admit)
            ex = {k: exports[k] for k in ("case_identity_sha256", "route", "latitude_deg", "census", "namespace", "distributions")}
            changed = {k: [man["limits"]["resource"][k], v] for k, v in nck.split_limits(lim.record())[1].items() if man["limits"]["resource"][k] != v}
            receipt = {"seal_sha256": man["manifest_sha256"], "next_index": man["boundary"]["next_index"], "role": role, "state_digest": state.digest,
                       "cursor_digest": man["temporal"]["cursor_digest"], "resource_limits_changed": changed, "plan_total": plan["total"],
                       "conservatism_upper_bound_bytes": plan["conservatism_upper_bound_bytes"], "historical_latitude_path": None if historical is None else historical.source["path"],
                       "live_latitude_path": None if a.latitude_live is None else a.latitude_live.source["path"]}
            self = cls._assemble(scenario=a.scenario, lop=new_lop, exports=ex, fence=a.fence, latitude=historical, limits=lim, profile_id=a.profile_id,
                                 policy=policy, terrain=terrain, retained_native_bytes=a.retained_native_bytes, resume_receipt=receipt, backend=a._backend, dev=a._dev,
                                 participant=nwb.LEAF_WATER_V1 if a.leaf else None, leaf_bundle=leaf_bundle, leaf_identity=leaf_ident)
            coordinator = Coordinator.resume(a.scenario, nck.parse_canonical(raw, "temporal.json"), self._adapters, latitude=historical)
            if coordinator.digest != man["temporal"]["cursor_digest"]:
                raise nck.CheckpointError("temporal_mismatch", "the adopted coordinator differs from the sealed cursor digest")
            object.__setattr__(self, "_pair", _Pair(coordinator, state, int(man["boundary"]["next_index"]), leaf_boundary))
            return self
        finally:
            art.close()

    @staticmethod
    def _restore_leaf(a: SimpleNamespace, art: Any, man: dict, state: Any, new_lop: Any, avail: Any, dev_admit: Any) -> tuple[Any, SimpleNamespace, dict]:
        """The biology half of a joint resume (after the physical load + audit): closed members read through the held artifact descriptor (sizes from fstat BEFORE the body, admitted), a genuine
        operator REBIND to the restored lifecycle operator, the v3 decode/rebuild with the exact clock/anchor/water joins, the static source/config join, and the manifest-vs-member token join. Returns
        (boundary, bundle, static identity); nothing is adopted here."""
        from masy import leaf_biology_codec_v3 as cdc
        from masy import leaf_calendar as lcal
        from masy import leaf_water_interval as lwi

        bm = man["biology"]
        try:
            op2 = lwi.rebind_operator(a.leaf_op, new_lop)
            lim_b = cdc.member_limits(bm["n_base"], bm["n_live"], bm["n_species"], cdc._max_name(op2.catalog.species_ids), cdc.metadata_cap(op2))
        except (lwi.IntervalError, cdc.BiologyCodecError) as exc:
            raise nck.CheckpointError("leaf_rebind_refused", str(exc)) from None
        if a._backend is not nwb.CPU_V1:
            dev_admit(int(cdc.required_decode_bytes(op2, bm["n_base"], bm["n_live"], bm["n_species"], 0)["device"]), "biology decode (device rebuild caps)")
        members = cdc.BiologyMembers()
        for name in nck.BIOLOGY_MEMBERS:
            cap = lim_b[name]
            data = art.read_biology(name, cap, lambda n: nck.admit(n + a.limits.caller_reserve_bytes, avail, f"biology member {name}"))
            rec = bm["members"][name]
            if len(data) != rec["bytes"] or hashlib.sha256(data).hexdigest() != rec["sha256"]:
                raise nck.CheckpointError("biology_member_mismatch", f"biology/{name} differs from the sealed manifest (size/sha256)")
            members[name] = data
        cursor = _leaf_dt_for(a.scenario, float(state.t_s))
        try:
            boundary = cdc.decode(op2, members, state, calendar_cls=lcal.LeafCalendar, expected_cursor=cursor, expected_anchor=a.scenario.calendar.start,
                                  expected_counts=(bm["n_base"], bm["n_live"], bm["n_species"]))
        except cdc.BiologyCodecError as exc:
            raise nck.CheckpointError("biology_" + exc.code, exc.detail) from None
        except lwi.IntervalMemoryError as exc:
            raise nck.CheckpointMemoryError(str(exc)) from None
        except lwi.IntervalError as exc:
            raise nck.CheckpointError("leaf_restore_refused", str(exc)) from None
        doc = json.loads(bytes(members["biology.json"]).decode("ascii"))
        if {k: doc[k] for k in bm["boundary_tokens"]} != bm["boundary_tokens"] or set(bm["boundary_tokens"]) != {k for k in ("water_digest", "mark", "anchor", "generation", "root_digest", "canopy_digest",
                                                                                                                       "area_digest", "schedule", "geo_elapsed_days", "live_rows_sha256", "initial")}:
            raise nck.CheckpointError("biology_token_mismatch", "the sealed manifest's boundary tokens differ from the biology member")
        if (bm["n_base"], bm["n_live"], bm["n_species"]) != (int(boundary.base_state.n), int(boundary.B_kg.shape[0]), op2.n_species):
            raise nck.CheckpointError("biology_token_mismatch", "the sealed manifest's record/live/species counts differ from the restored biology")
        try:
            bundle, ident = _leaf_bind(a._backend, a.scenario, new_lop, state, a.fence, op2, boundary, initial=False)
        except NativeWaterCalendarError as exc:
            raise nck.CheckpointError(exc.code, exc.detail) from None
        if hashlib.sha256(canonical_json(ident).encode("ascii")).hexdigest() != bm["static_identity_sha256"]:
            raise nck.CheckpointError("leaf_static_identity_mismatch", "the live static LEAF identity (profile, catalog, species, geometry caps, budget, cadence, counts) differs from the sealed one")
        return boundary, bundle, ident

    # ---- the atomic step ----
    def step(self) -> StepResult:
        pair = self._pair
        coordinator = pair.coordinator
        if coordinator.finished:
            raise _refuse("scenario_finished", "every occurrence has been realized")
        index = coordinator.next_index
        occurrence = self.scenario.occurrences[index]
        self._check_config()
        frame = (pair.serial, pair.state.digest, occurrence.occurrence_id, occurrence.index, self.config_token)
        terminal = self._terminal
        if terminal is not None and terminal.frame == frame:
            terminal.refusals += 1  # sticky: repeat the typed refusal BEFORE any admission/copy/trial/operator
            raise _refuse(terminal.code, f"{terminal.detail} [sticky terminal: cumulative steps {terminal.steps_total}, refused calls {terminal.refusals}]",
                          terminal=True, steps=terminal.steps_total)
        progress = self._progress
        if progress is not None and progress.key[0] != pair.serial:
            progress = None  # stale (an earlier occurrence was accepted since)
        req = self._requirements(pair, progress, occurrence)  # BEFORE the temporal record, hashes, arrays or operators exist
        available = _host_reading(self._backend, self.lop)
        if req["total"] > available:
            raise NativeWaterCalendarMemoryError(
                f"step needs {req['total']} B ({req['phases']}, extras {req['extras']}) but only {available} B are reported; "
                "Python/JIT/allocator overhead and the coordinator's Python object graph beyond the declared temporal reserve are excluded")
        if self._backend is not nwb.CPU_V1:
            _admit_device_total(self._backend, self.lop, self._dev, req["device_total"], "step")
        if index > self.limits.max_occurrences:
            raise _refuse("occurrence_limit", "cursor exceeds the declared maximum occurrences")
        record = coordinator.as_record()
        size = len(canonical_json(record))
        if size > self.limits.max_record_bytes:
            raise _refuse("temporal_record_limit", f"accepted temporal record is {size} B > {self.limits.max_record_bytes}")
        stage = _Stage(pair, progress, index)
        trial = Coordinator.resume(self.scenario, record, self._adapters, latitude=self.latitude)
        object.__setattr__(self, "_stage", stage)
        try:
            realized = trial.step()
        except CoordinatorError as exc:
            if exc.code == "event_incomplete" and stage.pending is not None:
                object.__setattr__(self, "_progress", stage.pending)
                return StepResult("incomplete", occurrence.occurrence_id, None, dict(stage.evidence))
            cause = exc.__cause__
            if isinstance(cause, NativeWaterCalendarError):
                if cause.terminal:
                    object.__setattr__(self, "_terminal", _Terminal(frame, cause.code, cause.detail, cause.steps_total))
                    object.__setattr__(self, "_progress", None)
                raise cause
            if isinstance(cause, MemoryError):
                raise NativeWaterCalendarMemoryError(f"{type(cause).__name__}: {cause}") from cause
            raise
        finally:
            object.__setattr__(self, "_stage", None)
        if stage.child is None or stage.adopted:
            raise _refuse("stage_not_adoptable", "no staged physical child (or already adopted)")
        if self._participant is nwb.LEAF_WATER_V1:
            self._leaf_join(stage)  # clock/water/lineage/source joins of the staged triple BEFORE the one swap (typed refusal leaves the accepted triple untouched)
        stage.adopted = True
        result = StepResult("complete", realized["occurrence_id"], realized, dict(stage.evidence))
        new_pair = _Pair(trial, stage.child, pair.serial + 1, stage.child_leaf)
        # single publication: everything above is already built
        object.__setattr__(self, "_pair", new_pair)
        object.__setattr__(self, "_progress", None)
        return result

    def run(self) -> list[StepResult]:
        results = []
        while not self.finished:
            result = self.step()
            results.append(result)
            if result.status != "complete":
                break
        return results

    # ---- called by the protocol adapters (inside trial.step) ----
    def _staged(self, kind: str, dispatch: dict[str, Any]) -> tuple[_Stage, Any, Any]:
        stage = self._stage
        if stage is None or stage.adopted:
            raise _refuse("no_stage", "adapter called outside a session step")
        occurrence = self.scenario.occurrences[stage.occurrence_index]
        parent = stage.pair.state
        if (dispatch["occurrence_id"] != occurrence.occurrence_id or dispatch["occurrence_index"] != occurrence.index
                or dispatch["kind"] != kind or occurrence.kind != kind
                or dispatch["event_file_sha256"] != occurrence.file_sha256 or dispatch["event_digest"] != occurrence.event.digest
                or dispatch["scenario_sha256"] != self.scenario.digest):
            raise _refuse("dispatch_identity_mismatch", "the sealed dispatch does not match the accepted occurrence")
        if decode_time(dispatch["native_start"]).hex() != _hex(parent.t_s):
            raise _refuse("parent_clock_mismatch", "dispatch native_start differs from the accepted physical clock")
        if self._participant is not nwb.LEAF_WATER_V1 and any(dispatch["processes"].get(name) for name in ("transpiration", "growth")):
            raise _refuse("unsupported_gate", "transpiration/growth gates are not supported")
        expected_lat = None if self.latitude is None else self.latitude.record()
        if dispatch["latitude"] != expected_lat:
            raise _refuse("latitude_mismatch", "dispatch latitude differs from the bound latitude")
        token = hashlib.sha256(canonical_json(dispatch).encode("ascii")).hexdigest()
        key = (stage.pair.serial, parent.digest, occurrence.occurrence_id, occurrence.index, token, self.config_token)
        if stage.progress is not None and stage.progress.key != key:
            raise _refuse("progress_key_mismatch", "pending progress belongs to a different parent/dispatch/source/forcing; it is not resumed")
        return stage, occurrence, key

    def _budget(self, stage: _Stage) -> tuple[int, int]:
        used = 0 if stage.progress is None else stage.progress.steps_total
        remaining_total = self.limits.max_accepted_steps_per_occurrence - used
        return used, min(remaining_total, self.limits.call_step_budget)

    def _run_rain(self, dispatch: dict[str, Any]) -> AdapterOutcome:
        stage, occurrence, key = self._staged("rain", dispatch)
        lop, limits = self.lop, self.limits
        event = occurrence.event
        parent = stage.pair.state
        intervals = event.event.intervals
        planned = [decode_time(e["native"]) for e in dispatch["planned_edges"]["edges"]]
        origin = float(parent.t_s)
        cumulative, edges = Fraction(0), [0.0]
        for interval in intervals:
            cumulative += Fraction(interval.duration_s)
            edges.append(float(cumulative))
        if len(planned) != len(intervals) + 2:
            raise _refuse("rain_edge_count_mismatch", "sealed dispatch edge count differs from the rain intervals")
        for k, edge in enumerate(edges):
            if (origin + edge).hex() != planned[k].hex():
                raise _refuse("rain_edge_mismatch", f"rain edge {k}: native origin+edge {(origin + edge).hex()} != sealed {planned[k].hex()}")
        guard = float(event.event.recession_guard_s)
        deadline = (origin + edges[-1]) + guard
        if deadline.hex() != planned[-1].hex():
            raise _refuse("recession_deadline_mismatch",
                          f"actual storm deadline {deadline.hex()} != MASY planned end {planned[-1].hex()} (one-ULP incompatibility refused; "
                          "no snapping, no guard stretch)")
        used, call = self._budget(stage)
        state = None if stage.progress is None else stage.progress.state
        steps_total, calls = used, 0 if stage.progress is None else stage.progress.calls
        if state is None:
            s = parent
            if s.role == "storm":
                s = wl.enter_interevent_phase(lop, s, expected_parent=wl.expect_parent(s))
                s = wl.close_interevent(lop, s, expected_parent=wl.expect_parent(s))
            if s.role != "boundary":
                raise _refuse("invalid_parent_role", f"rain needs a boundary or completed storm parent, got {s.role!r}")
            schedule, field, control = _storm_parts(self._backend, lop, limits, occurrence, guard, lambda o: self._rain_schedule(o, origin))
            state = wl.begin_storm(lop, s, expected_parent=wl.expect_parent(s), schedule=schedule, field=field,
                                   control=control, policy=self.policy)
        if call <= 0:
            reason, n, rejected = "max_steps", 0, 0
        else:
            result = wl.evolve_storm(lop, state, expected_parent=wl.expect_parent(state), until_s=deadline, max_accepted_steps=call)
            state, reason, n, rejected = result.state, result.reason, result.accepted_steps, result.rejected_attempts
        steps_total += n
        calls += 1
        evidence = {"kind": "rain", "reason": reason, "accepted_steps": n, "rejected_attempts": rejected, "steps_total": steps_total,
                    "end_s": _hex(state.t_s), "deadline_s": _hex(deadline), "calls": calls}
        stage.evidence = evidence
        if reason == "completed" and state.payload.completed is True:
            evidence["audit"] = wl.audit_lifecycle(lop, state)
            evidence["lifetime"] = state.lifetime.as_dict()
            if self._participant is nwb.LEAF_WATER_V1:  # rain + recession is transport: the calendar advances over the REALIZED native span, no E/T/growth, no catch-up
                from masy import leaf_water_interval as lwi

                try:
                    stage.child_leaf = lwi.carry_boundary(self._leaf.op, stage.pair.leaf, state, start=self._leaf_dt(float(parent.t_s)), end=self._leaf_dt(float(state.t_s)), parent=parent)
                except lwi.IntervalError as exc:
                    raise _refuse("participant_carry_refused", str(exc)) from exc
            stage.child = state
            stage.pending = None
            return AdapterOutcome("complete", float(state.t_s), None, None, None, evidence)
        if reason == "max_steps" and steps_total < limits.max_accepted_steps_per_occurrence:
            stage.pending = _Progress(key, state, "storm", 0, steps_total, calls)
            return AdapterOutcome("incomplete", None, evidence=evidence)
        raise _refuse("storm_incomplete_terminal", f"storm stopped with {reason!r} after {steps_total} accepted steps; not completion", terminal=True, steps=steps_total)

    def _fragments(self, dispatch: dict[str, Any], start: float, end: float) -> list[tuple[Any, float]]:
        calendar = self.scenario.calendar
        ledger = ClockLedger(self.scenario.native_epoch_s)
        try:
            start_g, end_g = ledger.global_elapsed(start), ledger.global_elapsed(end)
            dates = calendar.dates_touched(start_g, end_g)
            if len(dates) > self.limits.max_dates_per_interevent:
                raise _refuse("date_limit", f"{len(dates)} dates exceed {self.limits.max_dates_per_interevent}")
            out = []
            for index, day in enumerate(dates):
                if index + 1 < len(dates):
                    mid = calendar.elapsed_of(f"{dates[index + 1].isoformat()}T00:00:00{calendar.offset_text}", "midnight")
                    native = ledger.native_from_global(mid)
                else:
                    native = end
                out.append((day, native))
        except (CalendarError, ClockError) as exc:
            raise _refuse("fragment_not_native", f"calendar fragment cannot be resolved exactly: {exc}") from exc
        previous = start
        for _, native in out:
            if not native > previous:
                raise _refuse("fragment_collapsed", "a date fragment collapses on the native clock")
            previous = native
        if out[-1][1].hex() != end.hex():
            raise _refuse("fragment_end_mismatch", "last fragment does not end at the planned end")
        return out

    def _pet_rate(self, day: Any, evaporation: bool, pet_log: list) -> np.ndarray:
        n = int(self.lop.n_cells)
        if not evaporation:
            return np.zeros(n, dtype=np.float64)
        wp = weather_pet_library()
        weather = self.scenario.weather
        try:
            row = weather.coverage((day,))[0]
            daily = weather.daily_weather(row)
            lat = None if self.latitude is None else self.latitude.latitude_deg
            demand = wp.potential_demand(daily, lat, weather.pet_method_name)
            applied = wp.apply_terrain(demand, self.terrain)
            rate = np.array(wp.pet_m_s(np.asarray(applied.pet_mm_day_cells)), dtype=np.float64)
        except (WeatherError, PETError, ValueError) as exc:
            raise _refuse("pet_refused", f"{day.isoformat()}: {exc}") from exc
        pet_log.append({"date": day.isoformat(), "method": weather.pet_method_name, "pet_mm_day": float(demand.pet_mm_day),
                        "terrain_sha256": self.terrain.sha256, "weather_row_date": row["date"]})
        del applied, demand
        return rate

    # ---- LEAF participant (only reachable from a `create_leaf*` session) ----
    def _leaf_dt(self, native_t: float) -> Any:
        """The exact aware datetime of a native solver time through the EXISTING exact clock/calendar bridge (global elapsed Fraction -> calendar start + exact microseconds). An instant LEAF's datetimes
        cannot represent exactly is a bounded typed refusal that preserves the accepted state: nothing is rounded, restamped or tolerated."""
        return _leaf_dt_for(self.scenario, native_t)

    def _pet_scalars(self, day: Any, need: bool, pet_log: list) -> SimpleNamespace:
        """The dated weather row's scalars WITHOUT any grid array: (demand object, scalar PET mm/day, mean temperature C). Evaluated whenever E OR T needs it (the E gate never suppresses T)."""
        if not need:
            return SimpleNamespace(demand=None, pet_mm=0.0, tmean=0.0)
        wp = weather_pet_library()
        weather = self.scenario.weather
        try:
            row = weather.coverage((day,))[0]
            daily = weather.daily_weather(row)
            lat = None if self.latitude is None else self.latitude.latitude_deg
            demand = wp.potential_demand(daily, lat, weather.pet_method_name)
        except (WeatherError, PETError, ValueError) as exc:
            raise _refuse("pet_refused", f"{day.isoformat()}: {exc}") from exc
        pet_log.append({"date": day.isoformat(), "method": weather.pet_method_name, "pet_mm_day": float(demand.pet_mm_day), "terrain_sha256": self.terrain.sha256, "weather_row_date": row["date"]})
        return SimpleNamespace(demand=demand, pet_mm=float(demand.pet_mm_day), tmean=float(daily.tmean_c))

    def _pet_field(self, scal: SimpleNamespace) -> Any:
        """ONE terrain PET field (m/s) per date fragment from the already-evaluated scalar demand (terrain factor applied once by the existing `apply_terrain`), uploaded once through the backend."""
        wp = weather_pet_library()
        n = int(self.lop.n_cells)
        if scal.demand is None:
            return self._backend.pet_array(self.lop, np.zeros(n, dtype=np.float64))
        try:
            applied = wp.apply_terrain(scal.demand, self.terrain)
            rate = np.array(wp.pet_m_s(np.asarray(applied.pet_mm_day_cells)), dtype=np.float64)
        except (PETError, ValueError) as exc:
            raise _refuse("pet_refused", str(exc)) from exc
        del applied
        return self._backend.pet_array(self.lop, rate)

    def _leaf_join(self, stage: _Stage) -> None:
        from masy import leaf_water_interval as lwi

        leaf, child = stage.child_leaf, stage.child
        if leaf is None or child is None:
            raise _refuse("leaf_join_failed", "the staged child has no LEAF boundary / water state")
        if type(leaf) is not lwi.LeafBoundary or leaf not in lwi._MINTED:
            raise _refuse("leaf_join_failed", "the staged LEAF boundary is not a genuine LeafBoundary")
        op = self._leaf.op
        if leaf.water_digest != str(child.digest) or leaf.mark != lwi._mark_of(op, child):
            raise _refuse("leaf_join_failed", "the staged LEAF boundary is not bound to the staged lifecycle state (state digest / continuity mark)")
        if leaf.calendar.cursor != self._leaf_dt(float(child.t_s)):
            raise _refuse("leaf_join_failed", "the LEAF calendar cursor differs from the exact datetime of the staged physical clock")
        if leaf.calendar.schedule.anchor != self.scenario.calendar.start or leaf.base_state is not stage.pair.leaf.base_state:
            raise _refuse("leaf_join_failed", "the staged LEAF boundary changed its anchor or base plant state")

    def _run_interevent_leaf(self, dispatch: dict[str, Any]) -> AdapterOutcome:
        from masy import leaf_water_interval as lwi
        from masy import native_water_participant as npp

        stage, occurrence, key = self._staged("interevent", dispatch)
        if stage.progress is not None:
            raise _refuse("participant_no_progress", "a LEAF participant occurrence is never partially retained (no progress candidate can exist)")
        lop, limits, op = self.lop, self.limits, self._leaf.op
        parent, leaf = stage.pair.state, stage.pair.leaf
        pr = occurrence.event.processes
        e_on, t_on, g_on = bool(pr["evaporation"]), bool(pr["transpiration"]), bool(pr["growth"])
        planned_end = decode_time(dispatch["planned_edges"]["edges"][-1]["native"])
        fragments = self._fragments(dispatch, float(parent.t_s), planned_end)
        pet_log: list[dict[str, Any]] = []
        windows, start = [], float(parent.t_s)
        for day, until in fragments:
            scal = self._pet_scalars(day, e_on or t_on, pet_log)  # scalars only: NO grid array here

            def field(scal=scal) -> Any:  # ONE terrain field per date fragment, built lazily when its window runs (bounded memory), shared by hydrology E and the canopy gather
                return self._pet_field(scal)

            windows.append(npp.Window(self._leaf_dt(start), self._leaf_dt(until), scal.tmean, scal.pet_mm, field, e_on, t_on, g_on))
            start = until
        nxt = self.scenario.occurrences[occurrence.index + 1] if occurrence.index + 1 < len(self.scenario.occurrences) else None
        next_kind = None if nxt is None else nxt.kind
        next_growth = None if nxt is None or nxt.kind != "interevent" else bool(nxt.event.processes["growth"])
        try:
            out = npp.run_interevent(op, leaf, parent, tuple(windows), next_kind=next_kind, next_growth=next_growth, token=f"{occurrence.occurrence_id}:{stage.pair.serial}",
                                     max_accepted_steps=min(int(limits.max_accepted_steps_per_occurrence), int(limits.call_step_budget)), max_dt_s=limits.interevent_max_dt_s, max_retries=limits.interevent_max_retries,
                                     min_dt_s=limits.interevent_min_dt_s)
        except npp.ParticipantError as exc:
            raise _refuse("participant_" + exc.code, exc.detail) from exc
        except lwi.IntervalMemoryError as exc:
            raise NativeWaterCalendarMemoryError(str(exc)) from exc
        except lwi.IntervalError as exc:
            raise _refuse("participant_refused", str(exc)) from exc
        closed = out.closed
        if _hex(closed.t_s) != planned_end.hex():
            raise _refuse("realized_end_mismatch", "closed interevent clock differs from the planned end", terminal=True)
        evidence = {"kind": "interevent", "participant": "leaf_water_v1", "accepted_steps": int(out.evidence["accepted_hydro_steps"]), "items": out.evidence["items"], "flushed": out.evidence["flushed"],
                    "generation": out.evidence["generation"], "pending_days": out.evidence["pending_days"], "request_sequence": out.evidence["request_sequence"], "n_fragments": len(fragments),
                    "evaporation": e_on, "transpiration": t_on, "growth": g_on, "pet": pet_log}
        evidence["audit"] = wl.audit_lifecycle(lop, closed)
        evidence["lifetime"] = closed.lifetime.as_dict()
        stage.evidence = evidence
        stage.child, stage.child_leaf, stage.pending = closed, out.leaf, None
        return AdapterOutcome("complete", float(closed.t_s), None, None, None, evidence)

    def _run_interevent(self, dispatch: dict[str, Any]) -> AdapterOutcome:
        if self._participant is nwb.LEAF_WATER_V1:
            return self._run_interevent_leaf(dispatch)
        stage, occurrence, key = self._staged("interevent", dispatch)
        lop, limits = self.lop, self.limits
        evaporation = bool(occurrence.event.processes["evaporation"])
        parent = stage.pair.state
        planned_end = decode_time(dispatch["planned_edges"]["edges"][-1]["native"])
        fragments = self._fragments(dispatch, float(parent.t_s), planned_end)
        used, call = self._budget(stage)
        steps_total, calls = used, 0 if stage.progress is None else stage.progress.calls
        if stage.progress is None:
            if parent.role == "storm":
                state = wl.enter_interevent_phase(lop, parent, expected_parent=wl.expect_parent(parent))
            elif parent.role == "boundary":
                state = wl.enter_interevent_from_boundary(lop, parent, expected_parent=wl.expect_parent(parent), policy=self.policy)
            else:
                raise _refuse("invalid_parent_role", f"interevent needs a boundary or completed storm parent, got {parent.role!r}")
            fragment = 0
        else:
            state, fragment = stage.progress.state, stage.progress.fragment
        pet_log: list[dict[str, Any]] = []
        total_accepted = rejected_total = 0
        stopped = None
        while fragment < len(fragments):
            day, until = fragments[fragment]
            if call - total_accepted <= 0:
                stopped = "max_steps"
                break
            rate = self._backend.pet_array(lop, self._pet_rate(day, evaporation, pet_log))
            result = wl.evolve_interevent(lop, state, expected_parent=wl.expect_parent(state), until_s=until, pet_terrain_m_s=rate,
                                          max_dt_s=limits.interevent_max_dt_s, max_accepted_steps=call - total_accepted,
                                          max_retries=limits.interevent_max_retries, min_dt_s=limits.interevent_min_dt_s)
            del rate
            state = result.state
            total_accepted += result.accepted_steps
            rejected_total += result.rejected_attempts
            if result.reason == "max_steps":
                stopped = "max_steps"
                break
            if result.reason != "target":
                raise _refuse("interevent_terminal", f"interevent stopped with {result.reason!r}", terminal=True, steps=steps_total + total_accepted)
            fragment += 1
        steps_total += total_accepted
        calls += 1
        evidence = {"kind": "interevent", "accepted_steps": total_accepted, "rejected_attempts": rejected_total,
                    "steps_total": steps_total, "fragment": fragment, "n_fragments": len(fragments), "calls": calls,
                    "evaporation": evaporation, "pet": pet_log}
        stage.evidence = evidence
        if stopped is None:
            closed = wl.close_interevent(lop, state, expected_parent=wl.expect_parent(state))
            if _hex(closed.t_s) != planned_end.hex():
                raise _refuse("realized_end_mismatch", "closed interevent clock differs from the planned end", terminal=True, steps=steps_total)
            evidence["audit"] = wl.audit_lifecycle(lop, closed)
            evidence["lifetime"] = closed.lifetime.as_dict()
            stage.child = closed
            stage.pending = None
            return AdapterOutcome("complete", float(closed.t_s), None, None, None, evidence)
        if steps_total < limits.max_accepted_steps_per_occurrence:
            stage.pending = _Progress(key, state, "interevent", fragment, steps_total, calls)
            return AdapterOutcome("incomplete", None, evidence=evidence)
        raise _refuse("work_cap_exhausted", f"{steps_total} accepted steps reached the per-occurrence cap", terminal=True, steps=steps_total)


class _CapStandIn:
    """Capability-only stand-in used by `resume` to replay the sealed cursor BEFORE any physical read; `run` is never legitimate."""

    def __init__(self, caps: AdapterCapabilities) -> None:
        self._caps = caps

    def capabilities(self) -> AdapterCapabilities:
        return self._caps

    def run(self, dispatch: dict[str, Any]) -> AdapterOutcome:
        raise _refuse("standin_run", "a capability stand-in cannot run an occurrence")


class _Adapter:
    """Genuine protocol adapter (`capabilities`, `run`) bound to a session; holds no state of its own."""

    def __init__(self, session: NativeWaterSession, kind: str) -> None:
        self._session, self._kind = session, kind

    def capabilities(self) -> AdapterCapabilities:
        return self._session.capabilities_of(self._kind)

    def run(self, dispatch: dict[str, Any]) -> AdapterOutcome:
        if self._kind == "rain":
            return self._session._run_rain(dispatch)
        return self._session._run_interevent(dispatch)
