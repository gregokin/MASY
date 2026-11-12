"""Combined calendar + physical-water checkpoint artifact for `NativeWaterSession` (CPU, accepted occurrence boundaries only).

PURE layer (stdlib + `masy.clock/scenario_io` at import; SYRUP is imported lazily inside the CPU codec entry only). It owns:

* the closed artifact layout and sealed manifest (`masy.native_water_combined_checkpoint.v1`), bounded and strictly parsed;
* the atomic no-clobber publisher and the descriptor-relative reader of the artifact directory (outer directory and `physical/`);
* the accepted-role / time-bridge table that joins a coordinator cursor record to the one current physical lifecycle role;
* the path-free identity handed to the physical codec so a physical archive cannot be paired with another cursor;
* named whole-call memory plans (`plan_save`, `plan_load`) and strict availability readings;
* the closed physical-codec registry. Only the CPU native9f Dense entry exists; any other id/device is refused before I/O.

Layout (`<ckpt>/`): `physical/` (EXACTLY the unchanged SYRUP `lifecycle_checkpoint_v2` bundle), `temporal.json` (the canonical coordinator cursor
record) and `combined_checkpoint.json` (the sealed manifest, linked LAST). See `docs/native_water_combined_checkpoint.md`.

No physics lives here and nothing is copied from SYRUP. Hashes demonstrate consistency/integrity of a trusted-writer artifact; they are not signatures.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

from masy.clock import ClockError, decode_time
from masy.scenario_io import canonical_json

__all__ = [
    "CheckpointReceipt", "ARTIFACT_KIND", "CODEC_ID", "COMBINED_NAME", "MANIFEST_MAX_BYTES", "PHYSICAL_CODECS", "PHYSICAL_DIR", "SCHEMA", "TEMPORAL_NAME",
    "Artifact", "CheckpointError", "CheckpointIndeterminate", "CheckpointMemoryError", "CpuNative9fCodec", "admit", "build_manifest",
    "codec_for", "expected_boundary", "manifest_seal", "open_artifact", "physical_identity", "plan_load", "plan_save", "publish", "split_limits",
    "BIOLOGY_DIR", "BIOLOGY_MEMBERS", "SCHEMA_LEAF", "ARTIFACT_KIND_LEAF", "MAX_HELD_FDS_LEAF", "MANIFEST_MAX_BYTES_LEAF", "build_manifest_leaf", "validate_manifest_leaf", "open_artifact_leaf", "open_artifact_leaf_cuda", "publish_leaf", "publish_leaf_cuda",
    "strict_reading", "validate_manifest",
]

SCHEMA = "masy.native_water_combined_checkpoint.v1"
ARTIFACT_KIND = "calendar_physical_accepted_occurrence_boundary"
CODEC_ID = "cpu_native9f_dense_lifecycle_v2"
PHYSICAL_PROFILE = "native9f_dense_lifecycle_cpu_v2"
PHYSICAL_DIR = "physical"
#: The JOINT (water + LEAF biology) layout: a SECOND closed member directory written by the engine itself. These literals are the only layout; no member list is a caller argument.
BIOLOGY_DIR = "biology"
BIOLOGY_MEMBERS = ("base_plants.csv", "base_plants.metadata.json", "arrays.bin", "biology.json")
SCHEMA_LEAF = "masy.native_water_leaf_checkpoint.v1"
ARTIFACT_KIND_LEAF = "calendar_physical_biology_accepted_occurrence_boundary"
#: The CUDA V2 physical codec (separate closed identity; the CPU constants above are unchanged). Members are the actual V2 policy's closed `allowed_names`.
CUDA_CODEC_ID = "cuda_native9f_dense_lifecycle_v2"
CUDA_PHYSICAL_PROFILE = "native9f_dense_lifecycle_cuda_v2"
CUDA_PHYSICAL_MEMBERS = ("lifecycle_checkpoint_cuda_v2.json", "maple_state.npz", "lifecycle_state.npz")
TEMPORAL_NAME = "temporal.json"
COMBINED_NAME = "combined_checkpoint.json"
_TMP_NAME = ".combined_checkpoint.tmp"
N_NODES_ON_DISK = 4  # physical/, temporal.json, combined_checkpoint.json (+ the transient temp name during publication)
RESOURCE_LIMITS = ("caller_reserve_bytes", "temporal_reserve_bytes", "max_record_bytes")
_HEX64 = frozenset("0123456789abcdef")

# --- derived manifest bounds -------------------------------------------------------------------------------------------------------------------------
#: Every string in the manifest is capped at STRING_CAP characters (validated). The widest content is the string total below; ASCII escaping can widen a
#: character to six bytes (`\uXXXX`), so MANIFEST_MAX_BYTES = 6 x (declared string total) rounded up to a power of two with the numeric/structure allowance.
STRING_CAP = 4096
_STRING_TOTAL = (4 * STRING_CAP      # informational root/target paths
                 + 2 * STRING_CAP    # latitude historical path + live path
                 + 2 * (512 + 2048 + 512)  # two adapter capability records (identity, notes, ids)
                 + 8 * STRING_CAP)   # ids, digests, ISO calendar texts, runtime strings, lineage
MANIFEST_MAX_BYTES = 1 << (6 * _STRING_TOTAL + 16384 - 1).bit_length()
MANIFEST_MAX_BYTES_LEAF = 1 << (6 * (_STRING_TOTAL + 4 * STRING_CAP) + 16384 - 1).bit_length()  # + the `biology` section strings (identity/profile/digest texts)
MANIFEST_MAX_DEPTH = 6   # root -> limits -> behavioural -> storm_control -> value (4); latitude -> record -> source -> field (4); slack 2 for versioned subrecords
MANIFEST_MAX_NODES = 4096


class CheckpointError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class CheckpointMemoryError(CheckpointError):
    def __init__(self, detail: str) -> None:
        super().__init__("memory_refused", detail)


class CheckpointIndeterminate(CheckpointError):
    """A failure AFTER the atomic seal link: the artifact may be durable. Nothing is deleted or retried over it; no receipt exists."""

    def __init__(self, detail: str) -> None:
        super().__init__("indeterminate_after_seal", detail)


def _refuse(code: str, detail: str) -> CheckpointError:
    return CheckpointError(code, detail)


# --- strict availability / admission -----------------------------------------------------------------------------------------------------------------
def strict_reading(value: Any, what: str) -> int:
    """A reading is a non-Boolean integer >= 0 (0 is valid; NumPy integers accepted). Never truthiness, never coerced."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 0:
        raise _refuse("invalid_availability", f"{what}: the availability reading must be a non-Boolean integer >= 0, got {value!r}")
    return int(value)


def read_availability(fn: Any, what: str) -> int:
    if not callable(fn):
        raise _refuse("invalid_availability", f"{what}: an availability callable is required")
    return strict_reading(fn(), what)


def admit(need: int, fn: Any, what: str, detail: str = "") -> int:
    """Fresh reading per call; `need > reading` -> typed memory refusal BEFORE the guarded work. A reading is not a reservation."""
    if isinstance(need, bool) or not isinstance(need, int) or need < 0:
        raise _refuse("invalid_requirement", f"{what}: requirement {need!r} is not a non-negative int")
    reading = read_availability(fn, what)
    if need > reading:
        raise CheckpointMemoryError(f"{what} needs {need} B but only {reading} B are reported available{detail}; nothing was read, hashed or written")
    return reading


@dataclass(frozen=True)
class CheckpointReceipt:
    """Returned by `NativeWaterSession.checkpoint` ONLY after the artifact is sealed and the directory fsyncs completed."""

    path: str
    manifest_sha256: str
    physical_manifest_sha256: str
    state_digest: str
    physical_digest: str
    cursor_digest: str
    next_index: int
    role: str
    t_s_hex: str
    temporal_bytes: int
    manifest_bytes: int
    plan: dict


# --- closed codec registry (CPU only) ----------------------------------------------------------------------------------------------------------------
class CpuNative9fCodec:
    """The unchanged SYRUP `lifecycle_checkpoint_v2` native9f Dense CPU profile. SYRUP is imported lazily; no GPU module is ever touched."""

    id = CODEC_ID
    profile = PHYSICAL_PROFILE
    device = "cpu"

    @staticmethod
    def _lk2() -> Any:
        if "cupy" in sys.modules:
            raise _refuse("gpu_import", "CuPy is imported; the CPU combined checkpoint refuses GPU modules")
        from maple_syrup import lifecycle_checkpoint_v2 as lk2
        return lk2

    def check_operator(self, lop: Any) -> None:
        if getattr(lop, "backend", None) not in ("numpy", "numba"):
            raise _refuse("unsupported_backend", f"operator backend {getattr(lop, 'backend', None)!r}; the CPU codec supports NumPy and Numba only")
        if getattr(lop, "context", None) is None:
            raise _refuse("unsupported_bed", "the native9f profile is Dense-only (a bedless lifecycle has no profile)")

    def estimate(self, lop: Any, role: str, bed: Any, event_extra_bytes: int, extra_reserve_bytes: int) -> Any:
        self.check_operator(lop)
        return self._lk2().estimate_lifecycle_budget_v2(lop, role, profile=self.profile, source_kind=None, bed=bed,
                                                        event_extra_bytes=int(event_extra_bytes), extra_reserve_bytes=int(extra_reserve_bytes))

    def save(self, path: str, lop: Any, state: Any, identity: dict, *, limits: Any, host_fn: Any, disk_fn: Any, extra_reserve_bytes: int) -> Any:
        self.check_operator(lop)
        return self._lk2().save_lifecycle_checkpoint_v2(path, lop, state, identity, profile=self.profile, limits=limits, host_available_bytes_fn=host_fn,
                                                        disk_available_bytes_fn=disk_fn, extra_reserve_bytes=int(extra_reserve_bytes))

    def load(self, path: str, lop: Any, identity: dict, *, role: str, storm: Any, reference_bed: Any, limits: Any, host_fn: Any, disk_fn: Any,
             extra_reserve_bytes: int) -> Any:
        self.check_operator(lop)
        return self._lk2().load_lifecycle_checkpoint_v2(path, lop, identity, profile=self.profile, expected_role=role, expected_source_kind=None,
                                                        limits=limits, host_available_bytes_fn=host_fn, reference_bed=reference_bed, storm=storm,
                                                        disk_available_bytes_fn=disk_fn, extra_reserve_bytes=int(extra_reserve_bytes))


PHYSICAL_CODECS = MappingProxyType({CODEC_ID: CpuNative9fCodec()})


def codec_for(codec_id: Any, device: Any) -> CpuNative9fCodec:
    """Closed lookup. Unknown id, a non-CPU device or an unregistered GPU codec -> typed refusal before any I/O."""
    if type(codec_id) is not str or codec_id not in PHYSICAL_CODECS:
        raise _refuse("codec_unavailable", f"physical codec {codec_id!r} is not registered (registered: {sorted(PHYSICAL_CODECS)}); no GPU codec exists")
    codec = PHYSICAL_CODECS[codec_id]
    if device != codec.device:
        raise _refuse("codec_unavailable", f"device {device!r} is not {codec.device!r} for codec {codec_id!r}")
    return codec


class CudaNative9fV2Codec:
    """The registered CUDA sibling: the unchanged SYRUP `lifecycle_checkpoint_cuda_v2` engines (native9f Dense, boundary / completed storm) plus the accepted budget facade
    `lifecycle_checkpoint_cuda_calendar`. SYRUP is imported lazily; nothing runs at import. Every device figure is the caller's explicit `device_reserve_bytes` (no default)."""

    id = CUDA_CODEC_ID
    profile = CUDA_PHYSICAL_PROFILE
    device = "cuda"
    members = CUDA_PHYSICAL_MEMBERS

    @staticmethod
    def _mods() -> tuple:
        from maple_syrup import lifecycle_checkpoint_cuda as cs
        from maple_syrup import lifecycle_checkpoint_cuda_calendar as cal
        from maple_syrup import lifecycle_checkpoint_cuda_v2 as v2
        return cs, cal, v2

    @staticmethod
    def _dr(device_reserve_bytes: Any) -> int:
        if isinstance(device_reserve_bytes, bool) or not isinstance(device_reserve_bytes, int) or device_reserve_bytes < 0:
            raise _refuse("invalid_limits", "the CUDA codec needs an explicit non-negative integer device_reserve_bytes (no default)")
        return int(device_reserve_bytes)

    def check_operator(self, lop: Any) -> None:
        if getattr(lop, "backend", None) != "cuda":
            raise _refuse("unsupported_backend", f"operator backend {getattr(lop, 'backend', None)!r}; the CUDA codec supports the genuine CUDA lifecycle only")
        if getattr(lop, "context", None) is None:
            raise _refuse("unsupported_bed", "the native9f profile is Dense-only (a bedless lifecycle has no profile)")

    def estimate_save(self, lop: Any, state: Any, extra_reserve_bytes: int, *, device_reserve_bytes: Any) -> Any:
        self.check_operator(lop)
        _, cal, _ = self._mods()
        return cal.estimate_save_cuda_v2(lop, state, caller_host_reserve_bytes=0, caller_device_reserve_bytes=self._dr(device_reserve_bytes), extra_reserve_bytes=int(extra_reserve_bytes))

    def estimate_load(self, lop: Any, role: str, reference_bed: Any, n_intervals: Any, extra_reserve_bytes: int, *, device_reserve_bytes: Any) -> Any:
        """METADATA-ONLY load budget (no rainfall field is built): `n_intervals` is the integer interval count of the completed storm's rain event (None for a boundary)."""
        self.check_operator(lop)
        from masy import native_water_cuda as nwcu

        return nwcu.estimate_load_metadata(lop, expected_role=role, reference_bed=reference_bed, n_intervals=n_intervals, caller_host_reserve_bytes=0,
                                           caller_device_reserve_bytes=self._dr(device_reserve_bytes), extra_reserve_bytes=int(extra_reserve_bytes))

    def save(self, path: str, lop: Any, state: Any, identity: dict, *, limits: Any, host_fn: Any, disk_fn: Any, extra_reserve_bytes: int, device_reserve_bytes: Any = None,
             device_fn: Any = None) -> Any:
        self.check_operator(lop)
        _, _, v2 = self._mods()
        return v2.save_lifecycle_checkpoint_cuda_v2(path, lop, state, identity, limits=limits, caller_host_reserve_bytes=0, caller_device_reserve_bytes=self._dr(device_reserve_bytes),
                                                    host_available_bytes_fn=host_fn, device_available_bytes_fn=device_fn, disk_available_bytes_fn=disk_fn,
                                                    extra_reserve_bytes=int(extra_reserve_bytes))

    def load(self, path: str, lop: Any, identity: dict, *, role: str, storm: Any, reference_bed: Any, limits: Any, host_fn: Any, disk_fn: Any, extra_reserve_bytes: int,
             device_reserve_bytes: Any = None, device_fn: Any = None) -> Any:
        self.check_operator(lop)
        _, _, v2 = self._mods()
        return v2.load_lifecycle_checkpoint_cuda_v2(path, lop, identity, expected_role=role, expected_source_kind=None, storm=storm, limits=limits, reference_bed=reference_bed,
                                                    caller_host_reserve_bytes=0, caller_device_reserve_bytes=self._dr(device_reserve_bytes), host_available_bytes_fn=host_fn,
                                                    device_available_bytes_fn=device_fn, disk_available_bytes_fn=disk_fn, extra_reserve_bytes=int(extra_reserve_bytes))


CUDA_CODECS = MappingProxyType({CUDA_CODEC_ID: CudaNative9fV2Codec()})


def cuda_codec_for(codec_id: Any, device: Any) -> CudaNative9fV2Codec:
    """Closed lookup of the CUDA registry (separate from `codec_for`, whose CPU-only contract is unchanged). Unknown id or a non-CUDA device -> typed refusal before any I/O."""
    if type(codec_id) is not str or codec_id not in CUDA_CODECS:
        raise _refuse("codec_unavailable", f"physical codec {codec_id!r} is not a registered CUDA codec (registered: {sorted(CUDA_CODECS)})")
    codec = CUDA_CODECS[codec_id]
    if device != codec.device:
        raise _refuse("codec_unavailable", f"device {device!r} is not {codec.device!r} for codec {codec_id!r}")
    return codec


# --- limits split / role-time table / identity ----------------------------------------------------------------------------------------------------------
def split_limits(record: dict) -> tuple[dict, dict]:
    """`NativeCalendarLimits.record()` -> (behavioural, resource). Resource fields may change at resume only under fresh admission and are disclosed."""
    if type(record) is not dict:
        raise _refuse("invalid_limits", "limits record must be a dict")
    resource = {k: record[k] for k in RESOURCE_LIMITS}
    behavioural = {k: v for k, v in record.items() if k not in RESOURCE_LIMITS}
    return behavioural, resource


def expected_boundary(record: dict) -> dict:
    """Closed accepted-role table derived ONLY from a coordinator cursor record (the caller has validated it by `Coordinator.resume`/`as_record`).

    * no accepted occurrence: closed `boundary` at the native epoch;
    * last accepted `interevent`: closed `boundary` at that entry's native_end;
    * last accepted `rain`: the COMPLETED `storm` at that entry's native_end (never closed, never restamped).
    The last commit is the ledger's own `commit_after` (or the record's initial_commit); no value is assumed."""
    try:
        occurrences, entries = record["occurrences"], record["ledger"]["entries"]
        if not occurrences:
            return {"role": "boundary", "completed": None, "last_kind": None, "last_occurrence_id": None, "origin_native_hex": None,
                    "state_t_s_hex": decode_time(record["native_epoch"]).hex(), "initial_commit_hex": decode_time(record["initial_commit"]).hex(),
                    "last_commit_hex": decode_time(record["initial_commit"]).hex(), "ledger_last_native_end_hex": decode_time(record["native_epoch"]).hex(),
                    "ledger_entries": len(entries)}
        last = occurrences[-1]
        entry = entries[last["ledger_entry_index"]]
        kind = last["kind"]
        if kind not in ("rain", "interevent"):
            raise _refuse("role_not_accepted", f"last accepted occurrence kind {kind!r} has no accepted physical role in the water-only profile")
        end = decode_time(entry["native_end"]).hex()
        return {"role": "storm" if kind == "rain" else "boundary", "completed": True if kind == "rain" else None, "last_kind": kind,
                "last_occurrence_id": last["occurrence_id"], "origin_native_hex": decode_time(entry["native_start"]).hex(), "state_t_s_hex": end,
                "initial_commit_hex": decode_time(record["initial_commit"]).hex(), "last_commit_hex": decode_time(entry["commit_after"]).hex(),
                "ledger_last_native_end_hex": end, "ledger_entries": len(entries)}
    except (KeyError, TypeError, IndexError, ClockError) as exc:
        raise _refuse("temporal_record_invalid", f"cursor record does not expose the closed boundary fields: {type(exc).__name__}: {exc}") from None


def physical_identity(*, cursor_digest: str, scenario_sha256: str, next_index: int, fence_identity: str, case_identity_sha256: str,
                      state_t_s_hex: str) -> dict:
    """The path-free identity passed to the physical codec (<= 64 KiB, strict JSON). A physical archive from another cursor/run/step refuses inside the codec."""
    return {"kind": "masy_combined_physical_part/1", "cursor_digest": cursor_digest, "scenario_sha256": scenario_sha256, "next_index": int(next_index),
            "fence_identity": fence_identity, "case_identity_sha256": case_identity_sha256, "state_t_s_hex": state_t_s_hex}


# --- named whole-call plans -----------------------------------------------------------------------------------------------------------------------------
def plan_save(*, codec_budget: Any, max_record_bytes: int, census_staging_bytes: int, caller_reserve_bytes: int, temporal_reserve_bytes: int,
              retained_native_bytes: int) -> dict:
    """Metadata-only whole-call save requirement (before any serialization, record, hash or read).

    `codec_budget` was estimated WITH `extra_reserve_bytes == codec_extra_reserve_bytes(...)`, so the codec's own host figure (which counts the live
    owner, `resident_old_bytes`) already includes the held temporal text and the caller reserves exactly once. The codec figure is compared with the plain strict
    reading (unchanged conservative admission; the conservatism is at most `resident_old_bytes` and is reported)."""
    extras = int(retained_native_bytes) + int(temporal_reserve_bytes) + int(caller_reserve_bytes)
    phases = {"cold_census_staging": int(census_staging_bytes) + extras,
              "temporal_serialization": 3 * int(max_record_bytes) + extras,
              "physical_codec_whole": int(codec_budget.host_save_bytes),
              "combined_manifest": 3 * MANIFEST_MAX_BYTES + int(max_record_bytes) + extras}
    disk = int(codec_budget.serialized_bytes) + int(max_record_bytes) + MANIFEST_MAX_BYTES
    return {"phases": phases, "extras": extras, "total": max(phases.values()), "disk_bytes": disk, "nodes": N_NODES_ON_DISK,
            "codec_resident_old_bytes": int(codec_budget.resident_old_bytes), "conservatism_upper_bound_bytes": int(codec_budget.resident_old_bytes),
            "scope": "codec whole figure counts the live owner and is compared with a free reading (conservative by <= resident_old); caller/temporal/retained "
                     "reserves counted once via the codec extra_reserve; process/JIT/allocator overhead excluded; not a peak measurement"}


def codec_extra_reserve_bytes(*, max_record_bytes: int, caller_reserve_bytes: int, temporal_reserve_bytes: int, retained_native_bytes: int) -> int:
    """The single pass-through to the codec's `extra_reserve_bytes` (held temporal text bound + reserves), so nothing is counted twice."""
    return int(max_record_bytes) + int(caller_reserve_bytes) + int(temporal_reserve_bytes) + int(retained_native_bytes)


def plan_load(*, codec_budget: Any, common_bytes: int, temporal_bytes: int, preparation_recheck_bytes: int, census_staging_bytes: int, terrain_staging_bytes: int,
              audit_scratch_bytes: int, caller_reserve_bytes: int, temporal_reserve_bytes: int, retained_native_bytes: int) -> dict:
    """Metadata-only whole-call load requirement AFTER the admitted bounded manifest header and BEFORE the temporal body read/hash, census, recheck or codec.

    `common_bytes` = the reference owner's live resident (lifecycle resident + owned fixed + reserves) that stays live through every cold phase; the codec's own
    figure already contains it as `resident_old_bytes`, so it is not added to the codec phase."""
    extras = int(retained_native_bytes) + int(temporal_reserve_bytes) + int(caller_reserve_bytes)
    c = int(common_bytes)
    phases = {"temporal_replay": c + 12 * int(temporal_bytes) + extras,
              "preparation_recheck": c + int(preparation_recheck_bytes) + extras,
              "cold_census_staging": c + int(census_staging_bytes) + extras,
              "terrain_staging": c + int(terrain_staging_bytes) + extras,
              "physical_codec_whole": int(codec_budget.host_load_bytes),
              "post_load_audit": int(codec_budget.resident_old_bytes) + int(codec_budget.resident_new_bytes) + int(audit_scratch_bytes) + extras}
    return {"phases": phases, "extras": extras, "total": max(phases.values()), "serialized_bytes": int(codec_budget.serialized_bytes),
            "codec_resident_old_bytes": int(codec_budget.resident_old_bytes), "conservatism_upper_bound_bytes": int(codec_budget.resident_old_bytes),
            "scope": "reference preparation/setup are caller-built borrowed objects (their factories keep their own planners); the codec whole figure counts the "
                     "reference owner and is compared with a free reading (conservative by <= resident_old); process/JIT/allocator overhead excluded"}


def plan_load_cuda(*, codec_budget: Any, temporal_bytes: int, preparation_recheck_bytes: int, census_staging_bytes: int, terrain_staging_bytes: int,
                   caller_reserve_bytes: int, temporal_reserve_bytes: int, retained_native_bytes: int) -> dict:
    """CUDA-specific normalization of the combined load plan (the CPU `plan_load` above is unchanged). HOST phases only: the CPU plan adds the lifecycle's resident/scratch figures
    (`m.storm["resident_bytes"] + owned_fixed + reserves`, `audit_scratch_bytes`) which on CUDA are DEVICE quantities, so they are NOT host terms here; device figures live in the
    separate device plan built by the caller from the actual V2 load budget (stage prefixes, field prefix, preparation recheck). `codec_budget` is the `map_budget` adapter of the ACTUAL
    `CudaLoadBudget`: its host figure (`host_required_with_reserves`) already holds the codec's host peak, the wrapper's retained host bytes and the outside reserves/`extra` exactly once
    (the caller passes `codec_extra_reserve_bytes` as the codec's extra), so the codec phase is NOT topped up again. The post-load `audit_lifecycle` is a lifecycle entry that admits
    itself freshly (host and device) when it runs; the codec's own `restore_audit` stage is inside the codec figures."""
    extras = int(retained_native_bytes) + int(temporal_reserve_bytes) + int(caller_reserve_bytes)
    phases = {"temporal_replay": 12 * int(temporal_bytes) + extras, "preparation_recheck": int(preparation_recheck_bytes) + extras,
              "cold_census_staging": int(census_staging_bytes) + extras, "terrain_staging": int(terrain_staging_bytes) + extras,
              "physical_codec_whole": int(codec_budget.host_load_bytes)}
    return {"phases": phases, "extras": extras, "total": max(phases.values()), "serialized_bytes": int(codec_budget.serialized_bytes), "codec_resident_old_bytes": 0,
            "conservatism_upper_bound_bytes": 0,
            "scope": "CUDA HOST plan only: device quantities are never host terms; outside reserves once (the codec extra holds the temporal text bound, caller/temporal/retained reserves); "
                     "the CUDA load figures are INCREMENTAL, so no resident-owner top-up is added; later lifecycle entries re-admit freshly"}


# --- sealed manifest -------------------------------------------------------------------------------------------------------------------------------------
_TOP_KEYS = frozenset({"schema", "artifact_kind", "codec", "boundary", "temporal", "scenario", "latitude", "source", "case", "inputs", "limits",
                       "capabilities", "physical", "writer", "lineage", "manifest_sha256"})
_NESTED = {
    "codec": ("id", "profile", "device", "physical_dir"),
    "boundary": ("next_index", "finished", "last_occurrence_id", "last_kind", "role", "completed", "state_t_s_hex", "ledger_last_native_end_hex",
                 "initial_commit_hex", "last_commit_hex", "ledger_entries", "origin_native_hex"),
    "temporal": ("file", "sha256", "bytes", "cursor_digest"),
    "scenario": ("scenario_sha256", "source_sha256", "scenario_id", "occurrences_sha256", "n_occurrences", "weather", "calendar", "native_epoch_hex",
                 "gap_policy"),
    "latitude": ("record", "path_free"),
    "source": ("identity", "profile", "adapter_version", "native_commit", "native_package_digest", "bundle_id", "syrup_package_digest",
               "masy_manifest_sha256", "syrup_manifest_sha256", "census", "informational"),
    "case": ("case_identity_sha256", "route", "profile_id", "lop_sha256", "topology", "quiet_policy", "latitude_deg_hex"),
    "limits": ("behavioural", "resource"),
    "physical": ("dir", "manifest_sha256", "state_digest", "physical_digest", "role", "revision", "t_s_hex", "identity_sha256", "profile", "budget"),
    "writer": ("backend", "runtime"),
}


def manifest_seal(body: dict) -> str:
    return hashlib.sha256(canonical_json({k: v for k, v in body.items() if k != "manifest_sha256"}).encode("ascii")).hexdigest()


def build_manifest(body: dict) -> dict:
    """Return the sealed manifest (adds `manifest_sha256`); validates the closed shape first."""
    doc = dict(body)
    doc["manifest_sha256"] = ""
    validate_manifest(doc, check_seal=False)
    doc["manifest_sha256"] = manifest_seal(doc)
    return doc


_DELIVERY_CORE = ("schema", "bundle_id", "manifest_sha256", "archive_sha256", "device", "leaf_profile", "leaf_variant")  # == masy.production_runtime.DELIVERY_CORE_KEYS (pinned by a test)
_NESTED_LEAF = ("dir", "version", "profile", "variant", "leaf_package_digest", "n_base", "n_live", "n_species", "members", "static_identity_sha256", "boundary_tokens")


def build_manifest_leaf(body: dict) -> dict:
    """Sealed manifest of the JOINT artifact (schema `masy.native_water_leaf_checkpoint.v1`): the water-only closed sections plus the closed `biology` section."""
    doc = dict(body)
    doc["manifest_sha256"] = ""
    validate_manifest_leaf(doc, check_seal=False)
    doc["manifest_sha256"] = manifest_seal(doc)
    return doc


def _is_sha(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _HEX64


def _walk(value: Any, depth: int, counter: list, where: str) -> None:
    counter[0] += 1
    if counter[0] > MANIFEST_MAX_NODES:
        raise _refuse("manifest_invalid", f"{where}: more than {MANIFEST_MAX_NODES} values")
    if depth > MANIFEST_MAX_DEPTH:
        raise _refuse("manifest_invalid", f"{where}: deeper than {MANIFEST_MAX_DEPTH}")
    if isinstance(value, dict):
        for k, v in value.items():
            if type(k) is not str or len(k) > 256:
                raise _refuse("manifest_invalid", f"{where}: invalid key")
            _walk(v, depth + 1, counter, where)
    elif isinstance(value, list):
        if len(value) > 256:
            raise _refuse("manifest_invalid", f"{where}: list longer than 256")
        for v in value:
            _walk(v, depth + 1, counter, where)
    elif isinstance(value, str):
        if len(value) > STRING_CAP:
            raise _refuse("manifest_invalid", f"{where}: string longer than {STRING_CAP}")
    elif isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise _refuse("manifest_invalid", f"{where}: non-finite number")
    else:
        raise _refuse("manifest_invalid", f"{where}: unsupported value type {type(value).__name__}")


def validate_manifest(doc: Any, *, check_seal: bool = True) -> dict:
    """Closed key sets (top and every nested record), types of the identity leaves, caps, depth, nodes and (optionally) the seal. The historical public signature, unchanged: WATER-ONLY schema."""
    return _validate_manifest_engine(doc, check_seal, False)


def validate_manifest_leaf(doc: Any, *, check_seal: bool = True) -> dict:
    """The JOINT schema (`masy.native_water_leaf_checkpoint.v1`, adds the closed `biology` section); one private engine with `validate_manifest`, so no validator is copied. A water-only manifest is never
    valid here and vice versa."""
    return _validate_manifest_engine(doc, check_seal, True)


def _validate_manifest_engine(doc: Any, check_seal: bool, leaf: bool) -> dict:
    top = _TOP_KEYS | {"biology"} if leaf else _TOP_KEYS
    if type(doc) is not dict or set(doc) != top:
        raise _refuse("manifest_invalid", f"manifest requires exactly the keys {sorted(top)}")
    if (doc["schema"], doc["artifact_kind"]) != ((SCHEMA_LEAF, ARTIFACT_KIND_LEAF) if leaf else (SCHEMA, ARTIFACT_KIND)):
        raise _refuse("manifest_invalid", "unsupported schema/artifact kind")
    if leaf:
        bio = doc["biology"]
        if type(bio) is not dict or set(bio) != set(_NESTED_LEAF) or bio["dir"] != BIOLOGY_DIR:
            raise _refuse("manifest_invalid", f"manifest.biology requires exactly the keys {list(_NESTED_LEAF)} and the closed directory name")
        if type(bio["members"]) is not dict or set(bio["members"]) != set(BIOLOGY_MEMBERS) or not all(
                type(v) is dict and set(v) == {"sha256", "bytes"} and _is_sha(v["sha256"]) and type(v["bytes"]) is int and v["bytes"] >= 0 for v in bio["members"].values()):
            raise _refuse("manifest_invalid", "manifest.biology.members must record exactly the four closed members (sha256 + bytes)")
        if not _is_sha(bio["static_identity_sha256"]) or any(type(bio[k]) is not int or bio[k] < 0 for k in ("n_base", "n_live", "n_species")):
            raise _refuse("manifest_invalid", "manifest.biology identity/count fields are malformed")
    for name, keys in _NESTED.items():
        allowed = [set(keys)] + ([set(keys) | {"delivery"}] if name == "source" else [])  # `source.delivery` (verified-delivery core) is the ONLY optional member; absent for every historical manifest
        if type(doc[name]) is not dict or set(doc[name]) not in allowed:
            raise _refuse("manifest_invalid", f"manifest.{name} requires exactly the keys {list(keys)}")
    if "delivery" in doc["source"]:
        d = doc["source"]["delivery"]
        if type(d) is not dict or set(d) != set(_DELIVERY_CORE) or d["schema"] != "masy.native_delivery.v1" or not all(_is_sha(d[k]) for k in ("bundle_id", "manifest_sha256", "archive_sha256")) \
                or d["device"] not in ("cpu", "cuda") or type(d["leaf_profile"]) is not str or d["leaf_variant"] != "8da":
            raise _refuse("manifest_invalid", "manifest.source.delivery is not the closed verified-delivery core")
    _walk(doc, 1, [0], "manifest")
    t, p, s = doc["temporal"], doc["physical"], doc["scenario"]
    if t["file"] != TEMPORAL_NAME or p["dir"] != PHYSICAL_DIR or doc["codec"]["physical_dir"] != PHYSICAL_DIR:
        raise _refuse("manifest_invalid", "member names are closed")
    if not (_is_sha(t["sha256"]) and _is_sha(t["cursor_digest"]) and _is_sha(p["manifest_sha256"]) and _is_sha(p["state_digest"])
            and _is_sha(p["physical_digest"]) and _is_sha(p["identity_sha256"]) and _is_sha(s["scenario_sha256"]) and _is_sha(s["occurrences_sha256"])):
        raise _refuse("manifest_invalid", "a digest field is not 64 lowercase hex digits")
    for name in ("bytes",):
        if type(t[name]) is not int or t[name] < 0:
            raise _refuse("manifest_invalid", "temporal.bytes must be a non-negative int")
    b = doc["boundary"]
    if type(b["next_index"]) is not int or b["next_index"] < 0 or type(b["finished"]) is not bool or b["role"] not in ("boundary", "storm"):
        raise _refuse("manifest_invalid", "boundary fields have the wrong type/value")
    if set(doc["limits"]["resource"]) != set(RESOURCE_LIMITS):
        raise _refuse("manifest_invalid", "limits.resource requires exactly the closed resource fields")
    if check_seal:
        if not _is_sha(doc["manifest_sha256"]) or doc["manifest_sha256"] != manifest_seal(doc):
            raise _refuse("manifest_seal_mismatch", "the sealed manifest hash does not match its content")
    return doc


def scan_json_bytes(data: bytes) -> None:
    """Constant-memory byte scan BEFORE `json.loads`: depth, string length and node count are enforced while traversing, so a hostile document is refused with
    scratch independent of its size (the read itself is already capped)."""
    depth = nodes = run = 0
    in_str = esc = False
    for byte in data:
        if in_str:
            run += 1
            if run > 6 * STRING_CAP + 8:
                raise _refuse("manifest_invalid", "a string exceeds the closed cap")
            if esc:
                esc = False
            elif byte == 0x5C:
                esc = True
            elif byte == 0x22:
                in_str = False
            continue
        if byte == 0x22:
            in_str, run = True, 0
        elif byte in (0x7B, 0x5B):
            depth += 1
            nodes += 1
            if depth > MANIFEST_MAX_DEPTH:
                raise _refuse("manifest_invalid", f"nesting deeper than {MANIFEST_MAX_DEPTH}")
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth < 0:
                raise _refuse("manifest_invalid", "unbalanced brackets")
        elif byte in (0x2C, 0x3A):
            nodes += 1
        if nodes > 2 * MANIFEST_MAX_NODES:
            raise _refuse("manifest_invalid", "too many structural tokens")
    if in_str or depth != 0:
        raise _refuse("manifest_invalid", "truncated document")


def _no_dupes(pairs: list) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise _refuse("manifest_invalid", f"duplicate key {k!r}")
        out[k] = v
    return out


def _no_constant(name: str) -> Any:
    raise _refuse("manifest_invalid", f"non-finite constant {name}")


def parse_canonical(raw: bytes, what: str) -> Any:
    """Strict JSON: ASCII, no duplicate keys, no NaN/Infinity, and the bytes must be EXACTLY the canonical serialization (one encoding per document)."""
    try:
        text = raw.decode("ascii")
        value = json.loads(text, object_pairs_hook=_no_dupes, parse_constant=_no_constant)
    except CheckpointError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise _refuse("manifest_invalid", f"{what} is not strict ASCII JSON: {type(exc).__name__}") from None
    if canonical_json(value).encode("ascii") != raw:
        raise _refuse("manifest_invalid", f"{what} is not in canonical form")
    return value


# --- descriptor-relative artifact reader -----------------------------------------------------------------------------------------------------------------
_O = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _stamp(st: os.stat_result) -> tuple:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _read_member(dirfd: int, name: str, cap: int, what: str, admit_size: Callable[[int], None] | None = None) -> bytes:
    """No-follow, descriptor-relative bounded read: size from metadata BEFORE the body (admitted), fstat join before/after, exact length."""
    try:
        fd = os.open(name, _O, dir_fd=dirfd)
    except OSError as exc:
        raise _refuse("artifact_layout_refused", f"{what}: cannot open without following links: {exc.strerror or exc}") from None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise _refuse("artifact_layout_refused", f"{what} is not a regular file")
        if before.st_size > cap:
            raise _refuse("artifact_limit", f"{what} is {before.st_size} B, over its cap {cap} B (checked from metadata before any body read)")
        if admit_size is not None:
            admit_size(int(before.st_size))
        chunks, seen = [], 0
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            seen += len(chunk)
            if seen > before.st_size:
                raise _refuse("artifact_race", f"{what} grew while being read")
            chunks.append(chunk)
        after = os.fstat(fd)
        if seen != before.st_size or _stamp(after) != _stamp(before):
            raise _refuse("artifact_race", f"{what} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(fd)


PHYSICAL_MEMBERS = ("lifecycle_checkpoint.json", "maple_state.npz", "lifecycle_state.npz")  # the closed member set of the SYRUP v2 bundle (a name list only; no SYRUP import)


def _short(name: Any) -> str:
    return str(name)[:64]


def _scan_closed(dirfd: int, allowed: tuple, what: str) -> set:
    """Bounded descriptor scan: membership against a CLOSED name set while iterating; stops at the first foreign name or after `len(allowed) + 1` entries. Never builds
    a listing of the directory and never puts more than one truncated foreign name into a message."""
    seen: set = set()
    limit = len(allowed) + 1
    with os.scandir(dirfd) as it:
        for count, entry in enumerate(it, 1):
            if count > limit or entry.name not in allowed:
                raise _refuse("artifact_layout_refused", f"{what} holds an entry outside its closed layout (first: {_short(entry.name)!r}); at most {limit} entries are inspected")
            seen.add(entry.name)
    return seen


def _preflight_physical(dirfd: int, members: tuple = PHYSICAL_MEMBERS) -> None:
    """Bounded closed preflight of `physical/` BEFORE the old CPU codec opens its payload: no-follow real directory, closed member names, regular no-follow members."""
    try:
        pfd = os.open(PHYSICAL_DIR, _O | os.O_DIRECTORY, dir_fd=dirfd)
    except OSError as exc:
        raise _refuse("artifact_layout_refused", f"physical/ cannot be opened without following links: {exc.strerror or exc}") from None
    try:
        names = _scan_closed(pfd, members, "physical/")
        if names != set(members):
            raise _refuse("artifact_layout_refused", "physical/ lacks a member of the closed bundle layout")
        for name in names:
            if not stat.S_ISREG(os.lstat(name, dir_fd=pfd).st_mode):
                raise _refuse("artifact_layout_refused", f"physical/{name} is not a regular file")
    finally:
        os.close(pfd)


@dataclass
class Artifact:
    """An opened artifact directory (descriptor held). Use as a context manager; `close()` is idempotent."""

    path: str
    dirfd: int
    ident: tuple
    manifest_raw: bytes
    manifest: dict
    temporal_size: int
    physical_members: tuple = PHYSICAL_MEMBERS  # closed member names of the codec bundle (backend policy seam; the CPU default is the old tuple)
    layout: str = "water"  # "water" (physical/ temporal.json manifest) or "leaf" (+ the closed biology/ directory)

    def check_identity(self) -> None:
        """The path still names the very directory this descriptor holds (not replaced, not a symlink)."""
        try:
            st = os.lstat(self.path)
        except OSError:
            raise _refuse("artifact_race", "the artifact directory vanished") from None
        if stat.S_ISLNK(st.st_mode) or (st.st_dev, st.st_ino) != self.ident:
            raise _refuse("artifact_race", "the artifact directory was replaced after it was opened")

    @property
    def physical_path(self) -> str:
        """The path the codec will open, returned only after the bounded closed preflight of `physical/` (descriptor-relative, identity re-checked)."""
        self.check_identity()
        _preflight_physical(self.dirfd, self.physical_members)
        return os.path.join(self.path, PHYSICAL_DIR)

    def read_biology(self, name: str, cap: int, admit_size: Callable[[int], None] | None = None) -> bytes:
        """Read ONE closed biology member (joint layout only): the directory is opened no-follow relative to the held descriptor, its closed member set is preflighted, the size comes from metadata BEFORE
        the body (admitted), fstat joins before/after; the identity of the artifact directory is re-checked."""
        if self.layout != "leaf" or name not in BIOLOGY_MEMBERS:
            raise _refuse("artifact_layout_refused", "biology members exist only in the joint layout and only under the four closed names")
        self.check_identity()
        try:
            bfd = os.open(BIOLOGY_DIR, _O | os.O_DIRECTORY, dir_fd=self.dirfd)
        except OSError as exc:
            raise _refuse("artifact_layout_refused", f"biology/ cannot be opened without following links: {exc.strerror or exc}") from None
        try:
            names = _scan_closed(bfd, BIOLOGY_MEMBERS, "biology/")
            if names != set(BIOLOGY_MEMBERS):
                raise _refuse("artifact_layout_refused", "biology/ lacks a member of the closed layout")
            return _read_member(bfd, name, cap, f"biology/{name}", admit_size)
        finally:
            os.close(bfd)

    def read_temporal(self, expected_sha256: str, expected_bytes: int, cap: int) -> bytes:
        raw = _read_member(self.dirfd, TEMPORAL_NAME, cap, "temporal.json")
        if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise _refuse("temporal_mismatch", "temporal.json differs from the sealed manifest (size/sha256)")
        return raw

    def close(self) -> None:
        if self.dirfd >= 0:
            fd, self.dirfd = self.dirfd, -1
            os.close(fd)

    def __enter__(self) -> "Artifact":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def open_artifact(path: Any, *, admit_header: Callable[[int], None], temporal_cap: int) -> Artifact:
    return _open_artifact(path, admit_header=admit_header, temporal_cap=temporal_cap, members=PHYSICAL_MEMBERS)


def open_artifact_cuda(path: Any, *, admit_header: Callable[[int], None], temporal_cap: int) -> Artifact:
    """The explicit CUDA sibling of `open_artifact`: the closed CUDA member set is fixed here (never a caller argument)."""
    return _open_artifact(path, admit_header=admit_header, temporal_cap=temporal_cap, members=CUDA_PHYSICAL_MEMBERS)


def open_artifact_leaf(path: Any, *, admit_header: Callable[[int], None], temporal_cap: int) -> Artifact:
    """The JOINT layout opener (CPU): exactly {physical/, biology/, temporal.json, combined_checkpoint.json}; a water-only artifact is refused as `artifact_layout_refused` (and `open_artifact`
    refuses a joint one the same way because its closed top-level set has three names)."""
    return _open_artifact(path, admit_header=admit_header, temporal_cap=temporal_cap, members=PHYSICAL_MEMBERS, leaf=True)


def open_artifact_leaf_cuda(path: Any, *, admit_header: Callable[[int], None], temporal_cap: int) -> Artifact:
    return _open_artifact(path, admit_header=admit_header, temporal_cap=temporal_cap, members=CUDA_PHYSICAL_MEMBERS, leaf=True)


def _open_artifact(path: Any, *, admit_header: Callable[[int], None], temporal_cap: int, members: tuple, leaf: bool = False) -> Artifact:
    """Open and fence an artifact directory: exactly {physical/, temporal.json, combined_checkpoint.json}, no symlink/special node, sizes from metadata, the
    manifest read through the descriptor after `admit_header(manifest_bytes)` (the explicitly admitted first stage), strictly parsed and sealed.
    Nothing is written. Any refusal closes the descriptor."""
    target = os.path.abspath(os.fspath(path))
    try:
        st = os.lstat(target)
        if not stat.S_ISDIR(st.st_mode):
            raise _refuse("artifact_layout_refused", "the checkpoint path is not a real directory (symlink or special refused)")
        dirfd = os.open(target, _O | os.O_DIRECTORY)
    except OSError as exc:
        raise _refuse("artifact_layout_refused", f"cannot open the checkpoint directory without following links: {exc.strerror or exc}") from None
    try:
        closed = (COMBINED_NAME, TEMPORAL_NAME, PHYSICAL_DIR) + ((BIOLOGY_DIR,) if leaf else ())
        names = _scan_closed(dirfd, closed, "the artifact directory")
        if COMBINED_NAME not in names:
            raise _refuse("artifact_unsealed", "no sealed combined manifest (unsealed or partial artifact)")
        if names != set(closed):
            raise _refuse("artifact_layout_refused", f"the artifact must contain exactly {list(closed)}")
        pst = os.lstat(PHYSICAL_DIR, dir_fd=dirfd)
        if not stat.S_ISDIR(pst.st_mode):
            raise _refuse("artifact_layout_refused", "physical/ is not a real directory")
        if leaf and not stat.S_ISDIR(os.lstat(BIOLOGY_DIR, dir_fd=dirfd).st_mode):
            raise _refuse("artifact_layout_refused", "biology/ is not a real directory")
        tst = os.lstat(TEMPORAL_NAME, dir_fd=dirfd)
        if not stat.S_ISREG(tst.st_mode):
            raise _refuse("artifact_layout_refused", "temporal.json is not a regular file")
        if tst.st_size > temporal_cap:
            raise _refuse("artifact_limit", f"temporal.json is {tst.st_size} B, over the cap {temporal_cap} B (from metadata)")
        raw = _read_member(dirfd, COMBINED_NAME, MANIFEST_MAX_BYTES_LEAF if leaf else MANIFEST_MAX_BYTES, "combined manifest", admit_header)
        scan_json_bytes(raw)
        doc = (validate_manifest_leaf if leaf else validate_manifest)(parse_canonical(raw, "combined manifest"))
        top = os.fstat(dirfd)
        return Artifact(target, dirfd, (top.st_dev, top.st_ino), raw, doc, int(tst.st_size), tuple(members), "leaf" if leaf else "water")
    except BaseException:
        os.close(dirfd)
        raise


# --- atomic publisher ---------------------------------------------------------------------------------------------------------------------------------------
_fsync = os.fsync
_link = os.link
_mkdir = os.mkdir


#: Descriptors `publish` may hold at the same time: parent dir, outer dir, `physical/`, <= 3 member fds, temporal fd, temp-manifest fd (1+1+1+3+1+1 = 8).
#: Fixed metadata, never raised; all are closed before `publish` returns or raises.
MAX_HELD_FDS = 8
#: Joint layout: the 8 above + the `biology/` directory + its 4 members = 13 held at the peak (engine-owned holds only; descriptors a codec opens inside `build_physical` are its own scope).
MAX_HELD_FDS_LEAF = 13


def _write_new(dirfd: int, name: str, data: bytes, hold: list | None = None) -> tuple:
    """Create `name` exclusively and write it. Returns the created inode `(st_dev, st_ino)`. With `hold` (a list) the ORIGINAL creating descriptor is retained there on
    success (the caller closes it): an open descriptor keeps the inode allocated even after its name is unlinked, so the inode number cannot be reused by a replacement
    while it is compared. If its own write/fsync fails it rolls back only its own file (generation proved with the still-open fd) BEFORE closing, then re-raises."""
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), 0o600, dir_fd=dirfd)
    st = os.fstat(fd)
    ident = (st.st_dev, st.st_ino)
    keep = False
    try:
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            view = view[n:]
        _fsync(fd)
        if hold is not None:
            hold.append(fd)
            keep = True
    except BaseException:
        _unlink_if_owned(dirfd, name, ident, fd)
        raise
    finally:
        if not keep:
            os.close(fd)
    return ident


def _same_inode(a: os.stat_result, b: os.stat_result) -> bool:
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _unlink_if_owned(dirfd: int, name: str, ident: tuple, fd: int | None = None) -> bool:
    """Unlink `name` only if it is still a regular file of the proved generation. With a HELD `fd` the proof is `lstat(name)` == the LIVE `fstat(fd)` (the held descriptor
    pins the inode, so a replacement can never carry its number); ctime/mtime/nlink are never used. Without `fd` (legacy helper-local use) only the recorded identity is
    compared, which is NOT a generation proof."""
    try:
        st = os.lstat(name, dir_fd=dirfd)
        if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) != ident:
            return False
        if fd is not None and not _same_inode(st, os.fstat(fd)):
            return False
        os.unlink(name, dir_fd=dirfd)
        return True
    except OSError:
        return False


def _record_physical(dfd: int, hold: list, members: tuple = PHYSICAL_MEMBERS, dirname: str = PHYSICAL_DIR) -> dict | None:
    """Ownership evidence of the codec bundle, valid ONLY right after a genuine `build_physical` success. HOLDS (appended to `hold`, closed by the caller) the `physical/`
    directory fd (no-follow) and read-only no-follow fds of the REGULAR members of the closed codec member set (bounded scan, <= 3 recorded). No data is read. Unknown
    entries are not recorded (not ours). Returns None when `physical/` is not a real directory of our outer directory."""
    try:
        st = os.lstat(dirname, dir_fd=dfd)
        if not stat.S_ISDIR(st.st_mode):
            return None
        pfd = os.open(dirname, _O | os.O_DIRECTORY, dir_fd=dfd)
    except OSError:
        return None
    hold.append(pfd)
    try:
        if not _same_inode(os.fstat(pfd), st):
            return None
        children: dict = {}
        names = []
        with os.scandir(pfd) as it:
            for count, entry in enumerate(it, 1):
                if count > len(members) + 1:
                    break  # a bundle with extra entries: the extras are simply not recorded
                if entry.name in members:
                    names.append(entry.name)
        for name in names:
            try:
                cfd = os.open(name, _O, dir_fd=pfd)
            except OSError:
                continue
            hold.append(cfd)
            cst = os.fstat(cfd)
            if stat.S_ISREG(cst.st_mode):
                children[name] = ((cst.st_dev, cst.st_ino), cfd)
        return {"ident": (st.st_dev, st.st_ino), "fd": pfd, "children": children}
    except OSError:
        return None


def publish(path: Any, *, build_physical: Callable[[str], Any], temporal_bytes: bytes, make_manifest: Callable[[Any], dict]) -> dict:
    """The CPU layout (closed member set `PHYSICAL_MEMBERS`): the historical public signature, unchanged. See `_publish_engine` for the body and guarantees."""
    from masy import native_water_backend as nwb

    return _publish_engine(nwb.CPU_V1, path, build_physical=build_physical, temporal_bytes=temporal_bytes, make_manifest=make_manifest)


def publish_cuda(path: Any, *, build_physical: Callable[[str], Any], temporal_bytes: bytes, make_manifest: Callable[[Any], dict]) -> dict:
    """The explicit CUDA sibling: the same engine under the sealed `CUDA_V1` policy (closed member set `CUDA_PHYSICAL_MEMBERS`). No member name is a caller argument."""
    from masy import native_water_backend as nwb

    return _publish_engine(nwb.CUDA_V1, path, build_physical=build_physical, temporal_bytes=temporal_bytes, make_manifest=make_manifest)


def publish_leaf(path: Any, *, build_physical: Callable[[str], Any], biology_members: Any, temporal_bytes: bytes, make_manifest: Callable[[Any], dict]) -> dict:
    """The JOINT (water + biology) CPU layout: `physical/` (unchanged SYRUP codec), `biology/` (the four CLOSED members written by this engine from in-memory bytes-like objects), `temporal.json`,
    the sealed manifest linked last. Same private held-FD engine as `publish`; no member name is an argument (the dict must carry exactly `BIOLOGY_MEMBERS`)."""
    from masy import native_water_backend as nwb

    _check_biology_members(biology_members)  # None/absent is a refusal here, never a silent water-only layout
    return _publish_engine(nwb.CPU_V1, path, build_physical=build_physical, temporal_bytes=temporal_bytes, make_manifest=make_manifest, biology_members=biology_members)


def publish_leaf_cuda(path: Any, *, build_physical: Callable[[str], Any], biology_members: Any, temporal_bytes: bytes, make_manifest: Callable[[Any], dict]) -> dict:
    """CUDA sibling of `publish_leaf` (sealed `CUDA_V1` policy; the physical member set is the CUDA V2 one)."""
    from masy import native_water_backend as nwb

    _check_biology_members(biology_members)
    return _publish_engine(nwb.CUDA_V1, path, build_physical=build_physical, temporal_bytes=temporal_bytes, make_manifest=make_manifest, biology_members=biology_members)


def _check_biology_members(members: Any) -> dict:
    if not isinstance(members, dict) or set(members) != set(BIOLOGY_MEMBERS) or any(not isinstance(v, (bytes, bytearray, memoryview)) for v in members.values()):
        raise _refuse("biology_members_invalid", f"biology_members must be a dict of exactly {BIOLOGY_MEMBERS} bytes-like objects")
    return members


def _publish_engine(backend: Any, path: Any, *, build_physical: Callable[[str], Any], temporal_bytes: bytes, make_manifest: Callable[[Any], dict], biology_members: Any = None) -> dict:
    """The ONE shared private publishing engine. `backend` must be EXACTLY one of the two sealed module-literal policies (`CPU_V1`/`CUDA_V1`; `None`, names, copies, imposters are refused
    BEFORE any path, descriptor or callback is touched); the closed member names are derived from that policy here, never accepted from a caller. Create the artifact atomically: exclusive `mkdir` -> the physical codec writes `physical/` -> `temporal.json` (O_EXCL) -> sealed manifest linked LAST
    (temp name + `os.link`, no clobber). Before the link, any failure removes ONLY objects whose generation this call proves with the ORIGINAL descriptors it still holds
    (see `_rollback`); everything else (replacements, foreign entries, a `physical/` whose ownership is unknown because the codec callback failed) is LEFT and reported in
    `exc.cleanup`. After the link, any failure is `CheckpointIndeterminate` (nothing deleted). At most `MAX_HELD_FDS` (8: parent, outer, `physical/`, 3 physical members, temporal, temp manifest) descriptors are held for the WATER-ONLY layout and `MAX_HELD_FDS_LEAF` (13 = those 8 + `biology/` + its 4 members) for the JOINT layout (engine-owned holds only; descriptors a physical codec opens inside `build_physical` are a separate scope); ALL are closed before
    return/raise. Returns `{"physical": <codec receipt>, "manifest": <sealed dict>, "manifest_bytes": int}` only after the final directory fsyncs."""
    from masy import native_water_backend as nwb

    if backend is not nwb.CPU_V1 and backend is not nwb.CUDA_V1:
        raise nwb.BackendPolicyError("policy_forged", "the publishing engine needs exactly a sealed module-literal backend policy (no None, name, copy or imposter)")
    members = backend.physical_members
    joint = biology_members is not None
    if joint:
        _check_biology_members(biology_members)  # BEFORE any path, descriptor or callback
    target = os.path.abspath(os.fspath(path))
    parent, name = os.path.split(target)
    if not name or name in (".", ".."):
        raise _refuse("invalid_path", "the checkpoint path needs a final name")
    try:
        pfd = os.open(parent, _O | os.O_DIRECTORY)
    except OSError as exc:
        raise _refuse("invalid_path", f"the parent directory does not exist or is a link: {exc.strerror or exc}") from None
    created_outer = False
    ident: tuple | None = None
    dfd = -1
    sealed = False
    holds: list = []  # every descriptor this call opened for ownership proof (closed in `finally`)
    owned_files: dict = {}
    physical: dict | None = None
    biology: dict | None = None
    peak_fds = 0
    try:
        try:
            os.lstat(name, dir_fd=pfd)
        except FileNotFoundError:
            pass
        else:
            raise _refuse("path_exists", f"the checkpoint path already exists (no overwrite): {target}")
        try:
            _mkdir(name, 0o700, dir_fd=pfd)
        except FileExistsError:
            raise _refuse("path_exists", f"the checkpoint path already exists (no overwrite): {target}") from None
        created_outer = True
        st = os.lstat(name, dir_fd=pfd)
        ident = (st.st_dev, st.st_ino)
        dfd = os.open(name, _O | os.O_DIRECTORY, dir_fd=pfd)
        if (os.fstat(dfd).st_dev, os.fstat(dfd).st_ino) != ident:
            raise _refuse("artifact_race", "the new directory was replaced during creation")

        def same() -> None:
            now = os.lstat(name, dir_fd=pfd)
            if stat.S_ISLNK(now.st_mode) or not _same_inode(now, os.fstat(dfd)):
                raise _refuse("artifact_race", "the artifact directory was replaced during publication")

        same()
        receipt = build_physical(os.path.join(target, PHYSICAL_DIR))
        physical = _record_physical(dfd, holds, members)  # genuine success evidence only; a raising callback leaves `physical` None (ownership unknown)
        same()
        if joint:  # the ENGINE writes the closed biology members itself: every inode is created here with its descriptor held, so ownership is known by construction
            bfd = _make_dir_held(dfd, BIOLOGY_DIR, holds)
            biology = {"ident": (os.fstat(bfd).st_dev, os.fstat(bfd).st_ino), "fd": bfd, "children": {}}
            for bname in BIOLOGY_MEMBERS:
                ident = _write_new(bfd, bname, biology_members[bname], holds)
                biology["children"][bname] = (ident, holds[-1])
                peak_fds = max(peak_fds, 2 + len(holds))
            _fsync(bfd)
            same()
        owned_files[TEMPORAL_NAME] = (_write_new(dfd, TEMPORAL_NAME, temporal_bytes, holds), holds[-1])
        sealed_doc = make_manifest(receipt)
        data = canonical_json(sealed_doc).encode("ascii")
        cap = MANIFEST_MAX_BYTES_LEAF if joint else MANIFEST_MAX_BYTES
        if len(data) > cap:
            raise _refuse("artifact_limit", f"the sealed manifest is {len(data)} B, over {cap} B")
        owned_files[_TMP_NAME] = (_write_new(dfd, _TMP_NAME, data, holds), holds[-1])
        same()
        try:
            _link(_TMP_NAME, COMBINED_NAME, src_dir_fd=dfd, dst_dir_fd=dfd)  # THE SEAL (atomic, no clobber)
        except FileExistsError:
            raise _refuse("path_exists", "a combined manifest already exists") from None
        sealed = True
        os.unlink(_TMP_NAME, dir_fd=dfd)
        _fsync(dfd)
        _fsync(pfd)
        out = {"physical": receipt, "manifest": sealed_doc, "manifest_bytes": len(data)}
        if joint:
            out["peak_held_fds"] = max(peak_fds, 2 + len(holds))  # engine-owned holds only (parent + outer + the list); a codec's temporary descriptors are a separate scope
        return out
    except BaseException as exc:
        if sealed:
            raise CheckpointIndeterminate(f"{type(exc).__name__}: {exc}; the sealed artifact at {target} may be durable and was left untouched") from exc
        if created_outer and ident is not None:
            report = _rollback(pfd, name, dfd, owned_files, physical, biology)
            try:
                exc.cleanup = report  # type: ignore[attr-defined]
                if hasattr(exc, "add_note"):
                    exc.add_note(f"unsealed artifact rollback: removed={report['removed']} left={report['left']}")
            except Exception:  # noqa: BLE001
                pass
        raise
    finally:
        for fd in reversed(holds):
            try:
                os.close(fd)
            except OSError:
                pass
        if dfd >= 0:
            os.close(dfd)
        os.close(pfd)


def _make_dir_held(dfd: int, dirname: str, holds: list) -> int:
    """Create `dirname` exclusively (0700) inside the held outer directory and return its no-follow descriptor, HELD in `holds`; the opened inode must be the created one."""
    os.mkdir(dirname, 0o700, dir_fd=dfd)
    st = os.lstat(dirname, dir_fd=dfd)
    fd = os.open(dirname, _O | os.O_DIRECTORY, dir_fd=dfd)
    holds.append(fd)
    if not stat.S_ISDIR(st.st_mode) or not _same_inode(st, os.fstat(fd)):
        raise _refuse("artifact_race", f"{dirname}/ was replaced during creation")
    return fd


def _rollback(pfd: int, name: str, dfd: int, owned_files: dict, physical: dict | None, biology: dict | None = None) -> dict:
    """Pre-seal rollback by GENERATION PROOF ONLY: the held outer directory fd `dfd` (and every held file/dir fd) is compared live with the path (`lstat`) before each
    unlink/rmdir (never by name or recorded number alone, never `rmtree`). If the outer path is no longer the held directory nothing is touched. Returns
    `{"removed": [...], "left": [...]}`; leftovers are an unsealed, unloadable partial (no manifest)."""
    removed: list = []
    left: list = []
    try:
        now = os.lstat(name, dir_fd=pfd)
        if stat.S_ISLNK(now.st_mode) or not _same_inode(now, os.fstat(dfd)):
            return {"removed": removed, "left": ["<outer directory is not the created inode: nothing touched>"]}
    except OSError:
        return {"removed": removed, "left": ["<outer directory vanished>"]}
    for fname, (fid, ffd) in owned_files.items():
        if _unlink_if_owned(dfd, fname, fid, ffd):
            removed.append(fname)
        elif _exists(dfd, fname):
            left.append(f"{fname} (not the created inode)")
    if physical is None:
        if _exists(dfd, PHYSICAL_DIR):
            left.append(f"{PHYSICAL_DIR}/ (ownership unknown: the codec callback did not report success)")
    else:
        _rollback_physical(dfd, physical, removed, left)
    if biology is not None:
        _rollback_physical(dfd, biology, removed, left, BIOLOGY_DIR)
    elif _exists(dfd, BIOLOGY_DIR):
        left.append(f"{BIOLOGY_DIR}/ (created but never recorded: ownership unknown, left)")
    try:
        now = os.lstat(name, dir_fd=pfd)
        if stat.S_ISLNK(now.st_mode) or not _same_inode(now, os.fstat(dfd)):
            left.append(f"{name}/ (outer replaced during rollback)")
        else:
            os.rmdir(name, dir_fd=pfd)
            removed.append(name + "/ (outer)")
    except OSError:
        left.append(f"{name}/ (outer, not empty or not removable)")
    return {"removed": removed, "left": left}


def _exists(dirfd: int, name: str) -> bool:
    try:
        os.lstat(name, dir_fd=dirfd)
        return True
    except OSError:
        return False


def _rollback_physical(dfd: int, physical: dict, removed: list, left: list, dirname: str = PHYSICAL_DIR) -> None:
    pfd = physical["fd"]  # HELD since the directory was recorded: its inode cannot be reallocated while it is open
    try:
        st = os.lstat(dirname, dir_fd=dfd)
    except OSError:
        return
    if not stat.S_ISDIR(st.st_mode) or not _same_inode(st, os.fstat(pfd)):
        left.append(f"{dirname}/ (replaced: not the created inode)")
        return
    for cname, (cid, cfd) in physical["children"].items():
        if _unlink_if_owned(pfd, cname, cid, cfd):
            removed.append(f"{dirname}/{cname}")
        elif _exists(pfd, cname):
            left.append(f"{dirname}/{cname} (not the created inode)")
    try:
        st = os.lstat(dirname, dir_fd=dfd)
        if not stat.S_ISDIR(st.st_mode) or not _same_inode(st, os.fstat(pfd)):
            left.append(f"{dirname}/ (replaced during rollback)")
            return
        os.rmdir(dirname, dir_fd=dfd)
        removed.append(dirname + "/")
    except OSError:
        left.append(f"{dirname}/ (holds entries that are not ours)")


def default_disk_available(path: Any) -> int:
    """Free bytes of the checkpoint's parent directory (a reading, not a reservation); an unusable parent is a typed refusal before any I/O."""
    parent = os.path.dirname(os.path.abspath(os.fspath(path))) or "."
    try:
        return int(shutil.disk_usage(parent).free)
    except OSError as exc:
        raise _refuse("invalid_path", f"the parent directory of the checkpoint path is not usable: {exc.strerror or exc}") from None
