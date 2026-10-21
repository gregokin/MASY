"""Device-backed dense production, first part: host metadata view, controlling record, commit evidence.

Scope is the exact bounded synthetic interface case only. Physical authority stays with the native SYRUP/MAPLE
code; this module never copies native physics, changes native defaults, or holds the original CPU bed.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from types import ModuleType
from typing import Any

from masy import dense_production as _dense_production

DenseProductionError = _dense_production.DenseProductionError

SCOPE = ("dense_device_production_v1:exact_1class_4x4_6voxels_flat_le4000_bins_le8_events_le8:"
         "constant_depth_wind_water_disabled:no_vegetation_plan_no_gpu_wake")

_SCOPE_DIMENSIONS = {"cells": 16, "classes": 1, "voxels": 6}
_CASE_ROOT_KEYS = frozenset((
    "schema", "backend", "dtype", "representation", "geometry", "grain_classes", "class_order",
    "mass_resolution_kg", "n_voxels", "initial_time_s", "initial_bed_elevation_m", "initial_class_fractions",
    "initial_available_fraction", "initial_availability_policy", "initial_water_depth_m", "active_mask",
    "vegetation", "native_vegetation_content_sha256", "threshold_modifier_state", "topographic_commit_spec",
    "avalanching_spec", "depth_update_rule"))
_CASE_MAX_DEPTH, _CASE_MAX_ITEMS, _CASE_MAX_ENTRIES, _CASE_MAX_UTF8 = 8, 128, 4096, 65536
_DIAGNOSTIC_NAMES = ("source_removed_by_class_kg", "destination_added_by_class_kg", "gross_moved_by_class_kg",
                     "global_net_by_class_kg", "gross_physical_activity_by_class_kg")
_CONTEXT_FIELDS = ("geometry", "grain_classes", "mass_resolution_kg", "topographic_commit_spec",
                   "avalanching_spec", "initial_available_fraction", "depth_update_rule", "record_exchange_history",
                   "window_max_journal_records", "deep_chunk_encoding", "deep_overlay_max_bytes")
_CREATE_TOKEN = object()


def _check_context(context: Any) -> None:
    """Exact native `BedContext`, closed declared layout, native defaults only for the non-dense options."""
    from maple_syrup.sediment_bed import BedContext

    if type(context) is not BedContext:
        raise DenseProductionError("invalid_host_context", "exact native BedContext required")
    declared = tuple(field.name for field in dataclasses.fields(BedContext))
    if declared != _CONTEXT_FIELDS:
        raise DenseProductionError("unsupported_context_schema", "native BedContext field schema changed")
    layout = getattr(context, "__dict__", None)
    if type(layout) is not dict or len(layout) != len(declared) or set(layout) != set(declared):
        raise DenseProductionError("invalid_host_context", "BedContext instance layout is not the declared fields")
    if (context.record_exchange_history is not False or context.window_max_journal_records is not None
            or type(context.deep_chunk_encoding) is not str or context.deep_chunk_encoding != "full"
            or context.deep_overlay_max_bytes is not None):
        raise DenseProductionError("invalid_host_context", "native default exchange/window/deep options required")
    if type(context.depth_update_rule) is not str or context.depth_update_rule != "constant_depth":
        raise DenseProductionError("incompatible_water_rule", "shared constant_depth required")
    mres = context.mass_resolution_kg
    if type(mres) is not float or not math.isfinite(mres) or mres <= 0:
        raise DenseProductionError("invalid_host_context", "finite positive float mass_resolution_kg required")


def _copy_case(value: Any, depth: int, budget: list[int], *, make_copy: bool = True) -> Any:
    """Bounded primitive-only copy; every bound is checked before any expansion."""
    kind = type(value)
    if value is None or kind is bool:
        return value
    if kind is int:
        if value.bit_length() > 63:
            raise DenseProductionError("case_record_limit", "case metadata integer exceeds 63 bits")
        return value
    if kind is float:
        if not math.isfinite(value):
            raise DenseProductionError("invalid_case_record", "non-finite float in case metadata")
        return value
    if kind is str:
        # At most 16KiB can be allocated by encoding this <=4096-codepoint string.
        if len(value) > 4096:
            raise DenseProductionError("case_record_limit", "case metadata string exceeds 4096 codepoints")
        try:
            budget[1] += len(value.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise DenseProductionError("invalid_case_record", "case metadata string is not UTF-8") from exc
        if budget[1] > _CASE_MAX_UTF8:
            raise DenseProductionError("case_record_limit", "case metadata UTF-8 bound exceeded")
        return value
    if kind is not list and kind is not dict:
        raise DenseProductionError("invalid_case_record", f"unsupported case metadata type {kind.__name__}")
    if depth >= _CASE_MAX_DEPTH:
        raise DenseProductionError("case_record_limit", "case metadata depth bound exceeded")
    if len(value) > _CASE_MAX_ITEMS:
        raise DenseProductionError("case_record_limit", "case metadata container item bound exceeded")
    budget[0] += len(value)
    if budget[0] > _CASE_MAX_ENTRIES:
        raise DenseProductionError("case_record_limit", "case metadata entry bound exceeded")
    if kind is list:
        if make_copy:
            return [_copy_case(item, depth + 1, budget) for item in value]
        for item in value:
            _copy_case(item, depth + 1, budget, make_copy=False)
        return None
    copied = {} if make_copy else None
    for key in value:
        if type(key) is not str:
            raise DenseProductionError("invalid_case_record", "case metadata keys must be str")
        checked_key = _copy_case(key, depth + 1, budget, make_copy=make_copy)
        checked_value = _copy_case(value[key], depth + 1, budget, make_copy=make_copy)
        if make_copy:
            copied[checked_key] = checked_value
    return copied


def _case_admission(case_record: Any, case_sha256: Any) -> None:
    """Read-only finite primitive walk before the native bed/source schemas, with no copy or hash."""
    from masy import dense_bed as _bed

    if type(case_record) is not dict or len(case_record) != len(_CASE_ROOT_KEYS) or set(case_record) != _CASE_ROOT_KEYS:
        raise DenseProductionError("invalid_case_record", "exact static DenseCase v1 root keys required")
    if (type(case_sha256) is not str or len(case_sha256) != 64
            or any(c not in "0123456789abcdef" for c in case_sha256)):
        raise DenseProductionError("invalid_case_record", "lowercase SHA256 case_sha256 required")
    _copy_case(case_record, 0, [0, 0], make_copy=False)
    if (case_record["schema"] != _bed.DENSE_CASE_SCHEMA or case_record["backend"] != "numpy"
            or case_record["dtype"] != "float64" or case_record["representation"] != "dense_voxel_column"
            or case_record["active_mask"] != "full" or case_record["depth_update_rule"] != "constant_depth"
            or type(case_record["n_voxels"]) is not int):
        raise DenseProductionError("invalid_case_record", "case record is not the static dense v1 CPU case")


def _checked_case(case_record: Any, case_sha256: Any) -> dict[str, Any]:
    from masy import dense_bed as _bed

    _case_admission(case_record, case_sha256)
    copied = _copy_case(case_record, 0, [0, 0])
    if _bed._sha256(copied) != case_sha256:
        raise DenseProductionError("case_input_mutated", "case record does not reproduce case_sha256")
    return copied


class _HostBedView:
    """Metadata-only host view: original host context plus a bounded primitive copy of the case record."""

    __slots__ = ("context", "case_record", "case_sha256")

    def __init__(self, token: Any, context: Any, case_record: dict[str, Any], case_sha256: str) -> None:
        if token is not _CREATE_TOKEN:
            raise DenseProductionError("private_host_view", "use _HostBedView.create")
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "case_record", case_record)
        object.__setattr__(self, "case_sha256", case_sha256)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("_HostBedView is closed")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("_HostBedView is frozen")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("_HostBedView is frozen")

    def __reduce_ex__(self, protocol: Any) -> Any:
        raise TypeError("_HostBedView is not copyable")

    @classmethod
    def create(cls, n: Any, context: Any, case_record: Any, case_sha256: Any) -> _HostBedView:
        _check_context(context)
        return cls(_CREATE_TOKEN, context, _checked_case(case_record, case_sha256), case_sha256)

    def verify(self) -> None:
        if type(self) is not _HostBedView:
            raise DenseProductionError("private_host_view", "exact _HostBedView required")
        _check_context(self.context)
        _checked_case(self.case_record, self.case_sha256)

    def bind_metadata(self, metadata: Any) -> None:
        """Read-only check that device metadata wraps this same host context and case identity."""
        from masy.dense_device_runner import _HostMetadataBed

        if type(metadata) is not _HostMetadataBed:
            raise DenseProductionError("incompatible_host_metadata", "exact private _HostMetadataBed required")
        self.verify()
        if metadata.host_context is not self.context:
            raise DenseProductionError("incompatible_host_metadata", "metadata host_context is not the view context")
        if metadata.case_sha256 != self.case_sha256 or metadata.case_record != self.case_record:
            raise DenseProductionError("incompatible_host_metadata", "metadata case record/identity differs")


def controlling_record(n: Any, bed: Any, wind: Any, schedule: Any, profile: Any, limits: Any) -> dict[str, Any]:
    """Native GPU-profile rebuild/audit and exact shared context checks over the host view; no copied mapper."""
    from maple_syrup.sediment_bed import reroute_threshold_m

    from masy.dense_bed import DenseBedError, _require_host_arrays
    from masy.dense_production import schedule_envelope
    from masy.dense_wind_profile import (
        DenseProductionGpuWindProfile,
        ProductionProfileError,
        audit_dense_production_gpu_wind_config,
    )
    if type(bed) is not _HostBedView:
        raise DenseProductionError("invalid_host_view", "exact _HostBedView required; dynamic beds refused")
    bed.verify()
    if type(limits) is not DeviceProductionLimits:
        raise DenseProductionError("invalid_limits", "exact DeviceProductionLimits required")
    limits.__post_init__()
    if type(profile) is not DenseProductionGpuWindProfile:
        raise DenseProductionError("unsupported_profile", "exact DenseProductionGpuWindProfile required; wake refused")
    ctx, cfg, model = bed.context, wind.config, profile.model_config
    if type(cfg) is not n.SchedulerPhysicsConfig:
        raise DenseProductionError("invalid_wind_config", "exact native SchedulerPhysicsConfig required")
    for name in ("geometry", "grain_classes", "avalanching_spec", "topographic_commit_spec", "initial_available_fraction"):
        if getattr(cfg, name) is not getattr(ctx, name):
            raise DenseProductionError("incompatible_wind_config", f"{name} is not the shared context object")
    if float(cfg.mass_resolution_kg).hex() != float(ctx.mass_resolution_kg).hex():
        raise DenseProductionError("incompatible_wind_config", "mass resolution differs from the shared bed")
    for context_name, model_name in (("geometry", "geometry"), ("grain_classes", "grain_classes"),
            ("avalanching_spec", "avalanching"), ("topographic_commit_spec", "topographic_commit")):
        if getattr(cfg, context_name) is not getattr(model, model_name) or getattr(ctx, context_name) is not getattr(model, model_name):
            raise DenseProductionError("incompatible_wind_config", f"{context_name} is not the authored model object")
    g = ctx.geometry
    if type(g.ny) is not int or type(g.nx) is not int or g.ny != 4 or g.nx != 4:
        raise DenseProductionError("production_dimension_limit", "exact plain-int 4x4 geometry required")
    dimensions = {"cells": g.ny * g.nx, "classes": len(ctx.grain_classes.classes),
                  "voxels": bed.case_record["n_voxels"]}
    if dimensions != _SCOPE_DIMENSIONS:
        raise DenseProductionError("production_dimension_limit", f"case dimensions {dimensions} are outside {SCOPE}")
    particle = cfg.particle_transport
    if (type(particle.max_active_cohorts) is not int or not 1 <= particle.max_active_cohorts <= 4000
            or type(particle.quantile_bins) is not int or not 1 <= particle.quantile_bins <= 8
            or not 1 <= len(schedule.events) <= limits.max_events):
        raise DenseProductionError("production_profile_limit", "flat/bin/event metadata exceeds bounded scope")
    try:
        envelope, binding = schedule_envelope(schedule)
        audit = audit_dense_production_gpu_wind_config(profile, cfg, wind_envelope_record=envelope,
                                                       initial_available_fraction=ctx.initial_available_fraction)
    except ProductionProfileError as exc:
        raise DenseProductionError(exc.code, exc.detail) from exc
    model_water = model.water_coupling
    if (model_water.enabled is not False or cfg.water_coupling_spec.enabled is not False
            or model_water.depth_update_rule != "constant_depth"
            or cfg.water_coupling_spec.depth_update_rule != "constant_depth" or ctx.depth_update_rule != "constant_depth"):
        raise DenseProductionError("incompatible_water_rule", "explicit disabled wind water transport and shared constant_depth required")
    if cfg.particle_transport.mobile_state_representation != "flat":
        raise DenseProductionError("unsupported_mobile_state", "only native flat cohorts are qualified")
    pool = wind.initial_mobile_state
    if type(pool) is not n.MobileCohortState:
        raise DenseProductionError("unsupported_mobile_state", "exact native flat MobileCohortState required")
    try:
        _require_host_arrays(pool, "initial_mobile_state")
    except DenseBedError as exc:
        raise DenseProductionError(exc.code, exc.detail) from exc
    if not n.mobile_state_is_genuinely_empty(pool):
        raise DenseProductionError("mobile_state_not_empty", "the initial native wind pool must be genuinely empty")
    outer = cfg.outer_timestep_s
    if type(outer) is not float or not math.isfinite(outer) or outer <= 0:
        raise DenseProductionError("invalid_wind_config", "finite positive float outer_timestep_s required")
    return {"scope": SCOPE, "production_calibration": "pending_not_qualified_by_interface_tests",
            "limits": limits.as_record(), "dimensions": dimensions, "water_rule": "constant_depth",
            "hydraulic_routing_policy": {"native_reader": "maple_syrup.sediment_bed.reroute_threshold_m",
                                         "threshold_m_hex": float(reroute_threshold_m(ctx)).hex(),
                                         "decision": "actual_native_commit_bed_only"},
            "schedule_binding": binding, "native_config_audit": audit,
            "case_identity": {"schema": bed.case_record["schema"], "case_sha256": bed.case_sha256}}


def _diagnostic_values(np: Any, to_host: Any, name: str, value: Any) -> list[float] | None:
    if value is None:
        return None
    cp = sys.modules.get("cupy")
    genuine = (type(cp) is ModuleType and cp.__name__ == "cupy"
               and getattr(getattr(cp, "ndarray", None), "__module__", "").split(".")[0] == "cupy")
    if type(value) is not np.ndarray and (not genuine or type(value) is not cp.ndarray):
        raise DenseProductionError("invalid_commit_evidence", f"{name} must be a NumPy or loaded CuPy ndarray")
    if value.dtype != np.float64 or value.shape != (1,) or not value.flags.c_contiguous:
        raise DenseProductionError("invalid_commit_evidence", f"{name} must be float64 C-contiguous shape (1,)")
    host = to_host(value)
    if type(host) is not np.ndarray or host.dtype != np.float64 or host.shape != (1,):
        raise DenseProductionError("invalid_commit_evidence", f"{name} host transfer is not float64 shape (1,)")
    values = host.tolist()
    if not all(type(v) is float and math.isfinite(v) for v in values):
        raise DenseProductionError("invalid_commit_evidence", f"{name} must be finite")
    return values


def commit_evidence(np: Any, commit: Any) -> dict[str, Any]:
    if commit is None:
        return {"status": "no_commit", "scope": "latest_native_commit_only"}
    from maple.core.backend.arrays import to_host
    from maple.surface.topographic_commit.commit import CommitResult

    if type(commit) is not CommitResult:
        raise DenseProductionError("invalid_commit_evidence", "exact native CommitResult or None required")
    applied = commit.avalanche_applied
    if type(applied) is not bool:
        raise DenseProductionError("invalid_commit_evidence", "native plain bool avalanche_applied required")
    diag = commit.avalanche_diagnostics
    if diag is not None and type(diag) is not dict:
        raise DenseProductionError("invalid_commit_evidence", "avalanche_diagnostics must be a dict or None")
    return {"status": "committed", "scope": "latest_native_commit_only", "avalanche_applied": applied,
            "avalanche_diagnostics": {name: None if diag is None or name not in diag else
                                      _diagnostic_values(np, to_host, name, diag[name]) for name in _DIAGNOSTIC_NAMES}}


_MIB, _GIB, _TIB = 1 << 20, 1 << 30, 1 << 40
DECLARED_UNMEASURED = ("fft_workspace_reserve_bytes", "jit_driver_reserve_bytes", "native_transient_reserve_bytes",
                       "diagnostic_scratch_bytes")
_LIMIT_BOUNDS = {name: (1, _GIB) for name in DECLARED_UNMEASURED}
_LIMIT_BOUNDS.update({"cpu_soft_s": (1, 480), "cpu_hard_s": (1, 540), "file_size_bytes": (1, 64 * _MIB),
                      "nofile": (1, 1024), "max_host_rss_bytes": (1, 5 * _GIB), "min_host_free_bytes": (_GIB, _TIB),
                      "max_device_bytes": (1, 8 * _GIB), "min_device_free_bytes": (10 * _GIB, _TIB),
                      "max_separation_samples": (1, 128), "max_separation_owners": (1, 16), "max_events": (1, 8)})
_RESERVE_TOTAL_MAX = 4 * _GIB
# These are the actual unchanged gpu_runtime_worker.worker_env literals. Native imports may add only
# the separately checked installed-JAX additions in the protected worker's post-import adapter.
_ENV_POLICY = (("JAX_PLATFORMS", "cpu"), ("NUMBA_DISABLE_CUDA", "1"))
_THREAD_ARENA_ENV = ("MALLOC_ARENA_MAX", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")
_IDENTITY_KEYS = ("pid", "start_ticks", "boot_id", "device_id", "stream_ptr", "cuda_visible_devices")
_FIXED_CHECKPOINTS = frozenset(("unspecified", "constructing", "event_entry", "wrapper_work",
                                 "after_native_wind", "after_native_rain", "before_publication"))
_POLICY_AUTHORITY = "read_only_library_policy_external_protected_gates_required"


@dataclasses.dataclass(frozen=True)
class DeviceProductionLimits:
    """Static read-only dedicated GPU process and reserve policy; reserve fields are DECLARED_UNMEASURED.

    Reserves are declarations, not fit or peak evidence. Device driver VM is not host CPU address space, so no
    RLIMIT_AS term is defined here; the existing CPU thread/AS checker is separate and unchanged.
    """

    fft_workspace_reserve_bytes: int
    jit_driver_reserve_bytes: int
    native_transient_reserve_bytes: int
    diagnostic_scratch_bytes: int
    cpu_soft_s: int = 480
    cpu_hard_s: int = 540
    file_size_bytes: int = 64 * _MIB
    nofile: int = 1024
    max_host_rss_bytes: int = 5 * _GIB
    min_host_free_bytes: int = _GIB
    max_device_bytes: int = 8 * _GIB
    min_device_free_bytes: int = 10 * _GIB
    max_separation_samples: int = 128
    max_separation_owners: int = 16
    max_events: int = 8

    def __post_init__(self) -> None:
        if type(self) is not DeviceProductionLimits:
            raise DenseProductionError("invalid_limits", "exact DeviceProductionLimits required")
        if type(self.__dict__) is not dict or len(self.__dict__) != len(_LIMIT_BOUNDS) or set(self.__dict__) != set(_LIMIT_BOUNDS):
            raise DenseProductionError("invalid_limits", "DeviceProductionLimits layout is not the declared fields")
        for name, (low, high) in _LIMIT_BOUNDS.items():
            value = self.__dict__[name]
            if type(value) is not int or not low <= value <= high:
                raise DenseProductionError("invalid_limits", f"{name} must be a plain int in [{low}, {high}]")
        if self.__dict__["cpu_hard_s"] < self.__dict__["cpu_soft_s"]:
            raise DenseProductionError("invalid_limits", "cpu_hard_s must be >= cpu_soft_s")
        if self.reserve_total_bytes > _RESERVE_TOTAL_MAX:
            raise DenseProductionError("invalid_limits", "declared unmeasured reserve total exceeds 4 GiB")

    @property
    def reserve_total_bytes(self) -> int:
        return sum(self.__dict__[name] for name in DECLARED_UNMEASURED)

    def as_record(self) -> dict[str, Any]:
        """Static policy only; no PID, device, or resource observation."""
        self.__post_init__()
        reserves = {name: self.__dict__[name] for name in DECLARED_UNMEASURED}
        return {"scope": SCOPE, "full_production_peak": "unqualified_not_estimated",
                "policy_authority": _POLICY_AUTHORITY,
                "declared_unmeasured_reserves_bytes": reserves,
                "declared_unmeasured_reserve_total_bytes": sum(reserves.values()),
                "reserve_status": "declared_unmeasured_not_fit_or_peak_evidence",
                "host_address_space": "not_applied_device_driver_vm_is_not_host_cpu_as",
                "profile": {"classes": 1, "grid": [4, 4], "voxels": 6, "mobile_state": "flat",
                            "max_flat_entries": 4000, "max_bins": 8, "max_events": self.max_events},
                "policy": {name: self.__dict__[name] for name in _LIMIT_BOUNDS if name not in reserves}}


def _read_bounded(path: str, bound: int) -> bytes:
    try:
        with open(path, "rb") as handle:
            data = handle.read(bound + 1)
    except OSError as exc:
        raise DenseProductionError("process_observation", f"cannot read {path}") from exc
    if not data or len(data) > bound:
        raise DenseProductionError("process_observation", f"{path} is empty or exceeds {bound} bytes")
    return data


def _proc_int(text: bytes, name: str, high: int) -> int:
    if not text.isdigit() or len(text) > 20 or int(text) > high:
        raise DenseProductionError("process_observation", f"{name} is not a bounded decimal")
    return int(text)


def _loaded_cupy() -> Any:
    """Already-loaded genuine CuPy module only; never imports CuPy or discovers an accelerator."""
    cp = sys.modules.get("cupy")
    if type(cp) is not type(sys) or getattr(cp, "__name__", None) != "cupy":
        raise DenseProductionError("cupy_not_loaded", "a genuine loaded cupy module is required")
    ndarray = getattr(cp, "ndarray", None)
    module = getattr(ndarray, "__module__", None)
    if not isinstance(ndarray, type) or type(module) is not str or not (module == "cupy" or module.startswith("cupy.")):
        raise DenseProductionError("cupy_not_loaded", "cupy.ndarray is not the genuine CuPy array type")
    return cp


def process_identity() -> dict[str, Any]:
    """Actual worker observation; the PID is namespace-local. Source UUID/PCI/grants are handled outside."""
    import os

    stat = _read_bounded("/proc/self/stat", 4096)
    head, tail = stat.find(b" ("), stat.rfind(b")")
    rest = stat[tail + 2:].split() if 0 < head < tail else []
    if len(rest) < 20:
        raise DenseProductionError("process_observation", "/proc/self/stat layout not recognized")
    pid = _proc_int(stat[:head], "pid", 1 << 22)
    if pid < 1 or pid != os.getpid():
        raise DenseProductionError("process_observation", "/proc/self/stat pid is not this process")
    start_ticks = _proc_int(rest[19], "start_ticks", (1 << 63) - 1)
    if start_ticks < 1:
        raise DenseProductionError("process_observation", "positive start_ticks required")
    boot = _read_bounded("/proc/sys/kernel/random/boot_id", 64)
    boot_id = boot[:-1].decode("ascii", "replace") if boot.endswith(b"\n") else ""
    if (len(boot_id) != 36 or any(boot_id[i] != "-" for i in (8, 13, 18, 23))
            or not set(boot_id.replace("-", "")) <= set("0123456789abcdef")):
        raise DenseProductionError("process_observation", "boot_id is not a lowercase UUID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (type(visible) is not str or not visible.isascii() or not 1 <= len(visible) <= 128
            or not visible.isprintable() or any(ch.isspace() for ch in visible)):
        raise DenseProductionError("process_observation", "nonempty bounded CUDA_VISIBLE_DEVICES required")
    cp = _loaded_cupy()
    try:
        device_id, stream_ptr = cp.cuda.runtime.getDevice(), cp.cuda.get_current_stream().ptr
    except Exception as exc:
        raise DenseProductionError("device_observation", f"device identity read failed: {type(exc).__name__}") from exc
    if type(device_id) is not int or not 0 <= device_id <= 1024:
        raise DenseProductionError("device_observation", "device ordinal must be a bounded int >= 0")
    if type(stream_ptr) is not int or not 0 <= stream_ptr < 1 << 64:
        raise DenseProductionError("device_observation", "stream pointer must be a bounded int >= 0")
    return {"pid": pid, "start_ticks": start_ticks, "boot_id": boot_id, "device_id": device_id,
            "stream_ptr": stream_ptr, "cuda_visible_devices": visible}


def _host_mem_available_bytes() -> int:
    for line in _read_bounded("/proc/meminfo", 16384).split(b"\n"):
        parts = line.split()
        if len(parts) == 3 and parts[0] == b"MemAvailable:" and parts[2] == b"kB":
            return _proc_int(parts[1], "MemAvailable", _TIB) * 1024
    raise DenseProductionError("process_observation", "MemAvailable not found in /proc/meminfo")


def _check_observed_device_memory(limits: Any, *, free: Any, total: Any, used: Any, held: Any,
                                  cap: Any, upcoming: Any, checkpoint: Any) -> None:
    """Original scalar predicates over already observed counters; passing is not admission."""
    if not all(type(v) is int and 0 <= v <= _TIB for v in (free, total, used, held, cap)) or free > total or used > held:
        raise DenseProductionError("device_observation", "device memory counters are not bounded consistent ints")
    if held > limits.max_device_bytes:
        raise DenseProductionError("device_pool_limit", "pool held bytes exceed max_device_bytes")
    if cap != limits.max_device_bytes:
        raise DenseProductionError("device_pool_limit", "default pool hard limit must equal max_device_bytes")
    if used + upcoming > limits.max_device_bytes:
        raise DenseProductionError("device_budget", "pool used plus upcoming bytes exceeds max_device_bytes")
    if upcoming > free:
        raise DenseProductionError("device_budget", "upcoming bytes exceed current driver free memory")
    if checkpoint == "constructing" and free < limits.min_device_free_bytes:
        error = DenseProductionError("device_free_floor", "driver free memory below min_device_free_bytes")
        try:
            error._device_memory_refusal = (free, total, used, held, cap, upcoming,
                                             limits.min_device_free_bytes, limits.max_device_bytes)
        except Exception:
            pass
        raise error


def _copy_device_memory_refusal(error: Any) -> tuple[int, ...] | None:
    """Optional primitive DATA only; unavailable data never changes refusal or admission."""
    if type(error) is not DenseProductionError:
        return None
    fields = error.__dict__
    if type(fields) is not dict:
        return None
    code = fields.get("code")
    if type(code) is not str or code != "device_free_floor":
        return None
    data = fields.get("_device_memory_refusal")
    if type(data) is not tuple or len(data) != 8:
        return None
    if not all(type(value) is int and 0 <= value <= _TIB for value in data):
        return None
    free, total, used, held, cap, upcoming, minimum, maximum = data
    if (free > total or used > held or held > maximum or cap != maximum
            or used + upcoming > maximum or upcoming > free or free >= minimum):
        return None
    return data


def check_process(limits: Any, identity: Any, *, upcoming_wrapper_bytes: Any = 0,
                  checkpoint: Any = "unspecified") -> dict[str, Any]:
    """Read-only check of already-enforced process policy, expected worker identity, host and device budgets."""
    if type(limits) is not DeviceProductionLimits:
        raise DenseProductionError("invalid_limits", "exact DeviceProductionLimits required")
    limits.__post_init__()
    upcoming = upcoming_wrapper_bytes
    if type(upcoming) is not int or not 0 <= upcoming <= _TIB:
        raise DenseProductionError("invalid_process_request", "upcoming_wrapper_bytes must be a plain int in [0, 1 TiB]")
    if type(checkpoint) is not str or checkpoint not in _FIXED_CHECKPOINTS:
        raise DenseProductionError("invalid_process_request", "unknown checkpoint")
    if checkpoint in ("after_native_wind", "after_native_rain", "before_publication") and upcoming != 0:
        raise DenseProductionError("invalid_process_request", f"{checkpoint} admits no upcoming device bytes")
    if checkpoint == "constructing" and upcoming < limits.reserve_total_bytes:
        raise DenseProductionError("invalid_process_request", "constructing requires the full admitted bytes")
    if type(identity) is not dict or len(identity) != len(_IDENTITY_KEYS) or set(identity) != set(_IDENTITY_KEYS):
        raise DenseProductionError("worker_identity_mismatch", "expected worker identity must be the closed dict")
    import os
    import resource

    flags = sys.flags
    if (sys.platform != "linux" or flags.isolated != 1 or flags.no_user_site != 1
            or flags.dont_write_bytecode != 1 or flags.optimize != 0):
        raise DenseProductionError("process_policy", "Linux with -I -B and optimize 0 required")
    for rlimit, expected in ((resource.RLIMIT_CPU, (limits.cpu_soft_s, limits.cpu_hard_s)),
                             (resource.RLIMIT_CORE, (0, 0)),
                             (resource.RLIMIT_FSIZE, (limits.file_size_bytes, limits.file_size_bytes)),
                             (resource.RLIMIT_NOFILE, (limits.nofile, limits.nofile))):
        if resource.getrlimit(rlimit) != expected:
            raise DenseProductionError("process_rlimit", f"rlimit {rlimit} is not the enforced {expected}")
    for name, expected in _ENV_POLICY:
        if os.environ.get(name) != expected:
            raise DenseProductionError("process_environment", f"{name} must be {expected!r}")
    if any(name in os.environ for name in _THREAD_ARENA_ENV):
        raise DenseProductionError("process_environment", "thread/arena overrides are absent in the native GPU policy")
    actual = process_identity()
    for key in _IDENTITY_KEYS:
        if type(identity[key]) is not type(actual[key]) or identity[key] != actual[key]:
            raise DenseProductionError("worker_identity_mismatch", f"{key} differs from the actual worker")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_s = usage.ru_utime + usage.ru_stime
    if not math.isfinite(cpu_s) or cpu_s >= limits.cpu_soft_s:
        raise DenseProductionError("host_cpu_budget", "process CPU time reached the soft limit")
    statm = _read_bounded("/proc/self/statm", 256).split()
    if len(statm) < 2:
        raise DenseProductionError("process_observation", "/proc/self/statm layout not recognized")
    rss = _proc_int(statm[1], "resident_pages", _TIB) * resource.getpagesize()
    if rss > limits.max_host_rss_bytes:
        raise DenseProductionError("host_rss_limit", "host RSS exceeds max_host_rss_bytes")
    mem_available = _host_mem_available_bytes()
    if mem_available < limits.min_host_free_bytes:
        raise DenseProductionError("host_free_floor", "system MemAvailable below min_host_free_bytes")
    cp = _loaded_cupy()
    try:
        free, total = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        used, held, cap = pool.used_bytes(), pool.total_bytes(), pool.get_limit()
    except Exception as exc:
        raise DenseProductionError("device_observation", f"device memory read failed: {type(exc).__name__}") from exc
    _check_observed_device_memory(limits, free=free, total=total, used=used, held=held,
                                  cap=cap, upcoming=upcoming, checkpoint=checkpoint)
    return {"checkpoint": checkpoint, "policy_authority": _POLICY_AUTHORITY,
            "worker_pid_namespace_local": actual["pid"], "device_id": actual["device_id"],
            "cpu_used_s": cpu_s, "host_rss_bytes": rss, "host_mem_available_bytes": mem_available,
            "host_mem_available_scope": "global_meminfo_not_cgroup_headroom",
            "thread_arena_policy": "overrides_absent_unbounded_by_environment_external_whole_worker_caps_required",
            "device_free_bytes": free, "device_total_bytes": total, "pool_used_bytes": used,
            "pool_held_bytes": held, "pool_limit_bytes": cap, "upcoming_wrapper_bytes": upcoming,
            "device_free_floor": "checked_at_construction" if checkpoint == "constructing" else "not_applied_during_work"}


def construction_admission(n: Any, bed: Any, wind: Any, schedule: Any, limits: Any, base_required_bytes: Any,
                           device_limits: Any) -> dict[str, Any]:
    """Static construction admission for the one audited dense GPU profile.

    Every flow-separation number is delegated to the native resolvers. The charge is a declared structural
    shape allowance plus the DECLARED_UNMEASURED reserves; it is not a native peak. No kernel, allocation,
    copy or device observation happens here. The caller's fresh
    `check_process(..., upcoming=required, checkpoint="constructing")` immediately before the first domain
    copy stays the only pool-cap / current-free / floor gate.
    """
    import math
    from maple.core.parameters.flow_separation import (resolved_crest_lookahead_m, resolved_sampling_step_m,
                                                       separation_sample_count)

    if type(limits) is not DeviceProductionLimits:
        raise DenseProductionError("invalid_limits", "exact DeviceProductionLimits required")
    limits.__post_init__()
    cap = device_limits.max_device_bytes
    if type(cap) is not int or not 0 < cap <= limits.max_device_bytes:
        raise DenseProductionError("invalid_limits", "device max_device_bytes must be a plain int in (0, limits cap]")
    if type(base_required_bytes) is not int or not 0 <= base_required_bytes <= limits.max_device_bytes:
        raise DenseProductionError("invalid_required_bytes", "base_required_bytes must be a bounded plain int >= 0")
    if type(bed) is not _HostBedView:
        raise DenseProductionError("invalid_bed", "construction admission reads a _HostBedView only")
    bed.verify()
    g = bed.context.geometry
    shape, dx, dy = (g.ny, g.nx), g.dx_m, g.dy_m
    if (shape != (4, 4) or len(bed.context.grain_classes.classes) != 1
            or bed.case_record["n_voxels"] != 6 or type(dx) is not float or type(dy) is not float
            or not (math.isfinite(dx) and math.isfinite(dy) and dx > 0.0 and dy > 0.0)):
        raise DenseProductionError("invalid_bed", "bed metadata is not the one-class 4x4 / 6-voxel profile")
    from masy.dense_schedule import DenseSchedule, RainEvent, WindEvent
    if (type(schedule) is not DenseSchedule or len(schedule.events) != 4
            or len(schedule.events) > limits.max_events
            or tuple(type(event) for event in schedule.events) != (WindEvent, RainEvent, WindEvent, RainEvent)):
        raise DenseProductionError("invalid_schedule", "the original ordered wind/rain/wind/rain schedule is required")

    spec = wind.config.flow_separation_spec
    if spec is None or spec.enabled is not True:
        raise DenseProductionError("flow_separation_disabled", "the audited profile requires enabled flow separation")
    bound = float(spec.max_bubble_length_m)
    if not math.isfinite(bound) or bound <= 0.0:
        raise DenseProductionError("invalid_flow_separation", "max_bubble_length_m must be finite and positive")
    step = resolved_sampling_step_m(spec, dx, dy)
    lookahead = resolved_crest_lookahead_m(spec, dx, dy)
    samples = separation_sample_count(bound, step)
    crest = separation_sample_count(lookahead, step)
    ny, nx = shape
    cells = ny * nx
    owners = cells
    # Guard the sample ladders and owner count before anything could size a sampling/FFT buffer.
    if type(samples) is not int or not 2 <= samples <= limits.max_separation_samples:
        raise DenseProductionError("separation_samples", f"bubble ladder {samples!r} exceeds the declared bound")
    if type(crest) is not int or not 0 <= crest <= limits.max_separation_samples:
        raise DenseProductionError("separation_samples", f"crest ladder {crest!r} exceeds the declared bound")
    if owners > limits.max_separation_owners:
        raise DenseProductionError("separation_owners", f"{owners} separation owners exceed the declared bound")

    charged_samples = max(samples, crest)
    sample_matrix_pair = 2 * (8 + 1) * charged_samples * owners
    scalar_stencil = 8 * charged_samples * 4
    reserves = {name: limits.__dict__[name] for name in DECLARED_UNMEASURED}
    required = base_required_bytes + sample_matrix_pair + scalar_stencil + sum(reserves.values())
    if required > cap:
        raise DenseProductionError("device_bytes_exceeded", f"required {required} B exceeds max_device_bytes {cap} B")
    return {"scope": SCOPE, "full_production_peak": "unqualified_not_estimated",
            "flow_separation": {"max_bubble_length_m": bound, "sampling_step_m": float(step),
                                "crest_lookahead_m": float(lookahead), "bubble_samples": samples,
                                "crest_samples": crest, "owners": owners},
            "payload_lower_observations_bytes": {"full_complex128_spectrum": 16 * cells,
                                                 "half_complex128_spectrum": 16 * ny * (nx // 2 + 1),
                                                 "real_float64_field": 8 * cells},
            "payload_status": "per_payload_lower_observation_not_whole_peak",
            "structural_allowance_bytes": {"reference_cubic_sample_matrix_pair": sample_matrix_pair,
                                           "scalar_stencil": scalar_stencil},
            "structural_status": "declared_shape_allowance_not_native_peak",
            "declared_unmeasured_reserves_bytes": reserves,
            "base_required_bytes": base_required_bytes, "required_bytes": required, "max_device_bytes": cap,
            "device_observation": "none_caller_rechecks_process_before_first_domain_copy"}


def profile_admission(n: Any, context: Any, case_record: Any, case_sha256: Any, wind: Any, schedule: Any,
                      profile: Any, limits: Any, runner_limits: Any) -> dict[str, Any]:
    """Finite metadata only before closed physical/static schemas: no case copy/hash or mapper/audit.

    The existing closed source schemas run next in the factory. Only then is the metadata view copied/hashed;
    _source_identity owns the complete controlling_record/native audit before/after conversion.
    """
    from masy.dense_device_resources import DeviceResourceError, _closed_metadata
    from masy.dense_schedule import DenseSchedule
    from masy.dense_wind_profile import DenseProductionGpuWindProfile

    _check_context(context)
    _case_admission(case_record, case_sha256)
    if type(limits) is not DeviceProductionLimits or type(profile) is not DenseProductionGpuWindProfile:
        raise DenseProductionError("unsupported_profile", "exact production limits/profile required")
    limits.__post_init__()
    if type(schedule) is not DenseSchedule or not 1 <= len(schedule.events) <= limits.max_events:
        raise DenseProductionError("production_profile_limit", "bounded native DenseSchedule required")
    g = context.geometry
    if (type(g.ny) is not int or type(g.nx) is not int or (g.ny, g.nx) != (4, 4)
            or case_record["n_voxels"] != 6 or len(context.grain_classes.classes) != 1):
        raise DenseProductionError("production_dimension_limit", "exact 1-class/4x4/6-voxel profile required")
    cfg = wind.config
    if type(cfg) is not n.SchedulerPhysicsConfig:
        raise DenseProductionError("invalid_wind_config", "exact native scheduler config required")
    particle = cfg.particle_transport
    rows, bins = particle.max_active_cohorts, particle.quantile_bins
    if (particle.mobile_state_representation != "flat" or type(rows) is not int or not 1 <= rows <= 4000
            or type(bins) is not int or not 1 <= bins <= 8
            or type(runner_limits.max_wind_accounting_steps) is not int
            or not 1 <= runner_limits.max_wind_accounting_steps <= 64):
        raise DenseProductionError("production_profile_limit", "flat/bin/accounting metadata outside finite scope")
    if type(wind.initial_mobile_state) is not n.MobileCohortState:
        raise DenseProductionError("unsupported_mobile_state", "exact native flat pool required")
    try:
        _, leaves, _ = _closed_metadata(wind.initial_mobile_state, (n.np.ndarray,), (n.np,))
    except DeviceResourceError as exc:
        raise DenseProductionError(exc.code, exc.detail) from exc
    if len(leaves) > 64 or any(a.ndim != 1 or a.shape != (rows,) for a in leaves):
        raise DenseProductionError("production_profile_limit", "bounded native host slot arrays required")
    return {"scope": SCOPE, "metadata_only_admission": True, "case_copied_or_hashed": False,
            "native_config_audit": "deferred_until_closed_bed_and_static_sources"}
