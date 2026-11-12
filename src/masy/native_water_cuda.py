"""Small CUDA wrappers of the native water calendar (S2; written WITHOUT any GPU/CuPy/device execution; UNQUALIFIED until Root runs the hardware matrix).

Not a calendar clone: the calendar stays in `native_water_calendar` and reaches this module only through the sealed `CUDA_V1` policy of `native_water_backend`. Importing this module
imports no CuPy and no SYRUP module (everything device- or SYRUP-dependent is imported lazily inside the functions that need it). It provides:

* `CudaPlacement`: the explicit, closed device placement (id + caller device reserve + declared cap); no default device, no unlimited cap, no device picker;
* `plan_prep_host_bytes`: the pure host requirement of the CUDA preparation recheck (`ndc.plan_native_dense_cuda`);
* `device_packet` / `device_rainfall_field`: the only host->device packets of the calendar (one GRID-sized `(N,)` PET upload per dated fragment — N cells, so it scales with the grid; never per hydraulic step; the storm rainfall field built with a device mask);
* `map_budget`: the budget ADAPTER record mapping the ACTUAL `CudaSaveBudget`/`CudaLoadBudget` onto the names the host-side plan functions read (no invented quantity);
* `requirements`/`plan_create`: the CUDA step/create pre-admission built from the lifecycle's own authoritative `cuda_entry_plan` (separate host and device figures).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

__all__ = ["CudaPlacement", "CudaPlacementError", "device_packet", "device_rainfall_field", "entry_figures", "map_budget", "plan_prep_device_bytes", "plan_prep_host_bytes", "placement_record", "begin_storm_plan", "begin_storm_raw_bytes", "begin_storm_figures", "event_extra_bytes", "estimate_load_metadata", "rain_field_bytes", "strict_host_reading"]


class CudaPlacementError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class CudaPlacement:
    """Explicit placement of one GPU calendar run. `device_id` is the ONLY device used (A4's current-device gate must find it current; the calendar never selects another);
    `device_reserve_bytes` is the caller's outside-reserve on the device (counted once, never subtracted from a reading); `device_cap_bytes` is the declared upper bound of
    device bytes this run may use (a positive finite integer; no default, no 'unlimited')."""

    device_id: int
    device_reserve_bytes: int
    device_cap_bytes: int
    native_reserve_bytes: int

    def __post_init__(self) -> None:
        for name in ("device_id", "device_reserve_bytes", "device_cap_bytes", "native_reserve_bytes"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise CudaPlacementError("invalid_placement", f"{name} must be a non-Boolean integer >= 0, got {v!r}")
        if self.device_cap_bytes < 1:
            raise CudaPlacementError("invalid_placement", "device_cap_bytes must be a positive integer (there is no default and no unlimited device cap)")


def placement_record(p: CudaPlacement) -> dict:
    return {"device_id": p.device_id, "device_reserve_bytes": p.device_reserve_bytes, "device_cap_bytes": p.device_cap_bytes, "native_reserve_bytes": p.native_reserve_bytes}


def plan_prep_host_bytes(preparation: Any) -> int:
    """Pure host requirement of the CUDA preparation recheck (reserve excluded): `ndc.plan_native_dense_cuda(admission, device_id, 0, 0).host_required_bytes`. No CuPy, no device query."""
    from maple_syrup import native_dense_cuda_profile as ndc

    try:
        plan = ndc.plan_native_dense_cuda(preparation.admission, device_id=int(preparation.device_id), caller_host_reserve_bytes=0, caller_device_reserve_bytes=0)
    except ndc.NativeDenseCudaProfileError as exc:
        raise CudaPlacementError("preparation_plan_failed", str(exc)) from exc
    return int(plan.host_required_bytes)


def plan_prep_device_bytes(preparation: Any) -> int:
    """Pure device requirement of the CUDA preparation recheck (reserve excluded)."""
    from maple_syrup import native_dense_cuda_profile as ndc

    try:
        plan = ndc.plan_native_dense_cuda(preparation.admission, device_id=int(preparation.device_id), caller_host_reserve_bytes=0, caller_device_reserve_bytes=0)
    except ndc.NativeDenseCudaProfileError as exc:
        raise CudaPlacementError("preparation_plan_failed", str(exc)) from exc
    return int(plan.device_required_bytes)


def device_packet(lop: Any, rate: Any) -> Any:
    """ONE GRID-sized `(N,)` float64 host->device upload (the constant terrain PET of a dated fragment; N = cells, 8N bytes). Once per dated fragment, never per hydraulic step or per voxel."""
    import numpy as np
    from maple.core.backend import to_device

    arr = np.ascontiguousarray(np.asarray(rate, dtype=np.float64))
    if arr.shape != (int(lop.n_cells),):
        raise CudaPlacementError("invalid_packet", f"PET packet shape {arr.shape} is not ({int(lop.n_cells)},)")
    return to_device(arr, lop.xp)


def device_rainfall_field(lop: Any) -> Any:
    """The storm rainfall field on the lifecycle's device: `rainfall_field(ny, nx, active_mask=<device bool (ny, nx)>, xp=cupy)` (the helper does its single validation read)."""
    import numpy as np
    from maple.core.backend import to_device
    from maple_syrup.rainfall import rainfall_field

    mask = to_device(np.ascontiguousarray(np.asarray(lop.topology.active, dtype=bool)), lop.xp)
    return rainfall_field(*lop.shape, active_mask=mask, xp=lop.xp)


def map_budget(budget: Any) -> Any:
    """Adapter record over the ACTUAL planner budget (exact type check; no quantity is invented). The CUDA codec charges incremental bytes only and never the resident
    owner, so `resident_old/new_bytes` are 0; host and device figures stay separate (`device_required_with_reserves`)."""
    from maple_syrup import lifecycle_checkpoint_cuda_load_plan as lp
    from maple_syrup import lifecycle_checkpoint_cuda_plan as sp

    if type(budget) not in (sp.CudaSaveBudget, lp.CudaLoadBudget):
        raise CudaPlacementError("invalid_budget", f"expected an actual CudaSaveBudget/CudaLoadBudget, got {type(budget).__name__}")
    host = int(budget.host_required_with_reserves)
    return SimpleNamespace(host_save_bytes=host, host_load_bytes=host, serialized_bytes=int(budget.serialized_bytes), disk_bytes=int(budget.disk_bytes),
                           resident_old_bytes=0, resident_new_bytes=0, device_required_with_reserves=int(budget.device_required_with_reserves), cuda=budget)


# --- CUDA pre-admission (lifecycle-authoritative `cuda_entry_plan`; separate host and device figures) ----------------------------------------------------
def entry_figures(lop: Any, entry: str, **kw: Any) -> tuple[int, int]:
    """(host_bytes, device_bytes) of one public CUDA lifecycle entry from the lifecycle's own pure metadata plan (no reading, no device query)."""
    from maple_syrup import water_lifecycle as wl

    plan = wl.cuda_entry_plan(lop, entry, **kw)
    return int(plan["host_bytes"]), int(plan["device_bytes"])


# --- rainfall-field device accounting (read from the actual `rainfall.rainfall_field`) -----------------------------------------------------------------------------------------
def rain_field_bytes(n_cells: int) -> dict[str, int]:
    """Device/host bytes of building ONE storm rainfall field with `device_rainfall_field` (N = cells). Device peak while building = the uploaded bool mask (N) + `xp.ones` base (8N) +
    the `xp.where` result (8N) = 17N; the returned multiplier (8N) is what persists BESIDE the later stages (`begin_storm`, the V2 load); the mask/base die on return. The host builds one
    bool `(ny, nx)` array (N). CUDA memcpy staging / allocator rounding / pool growth are NOT counted here (documented outside the figures)."""
    n = int(n_cells)
    return {"device_build_peak": 17 * n, "device_persistent": 8 * n, "host_build": n}


def strict_host_reading(fn: Any, what: str) -> int:
    """ONE strict host availability reading for the CUDA branches: `fn` must be callable and return a non-Boolean integer >= 0 (NumPy integers accepted, floats/None/bools refused, no
    `int()` coercion). Raises `CudaPlacementError('invalid_availability')`."""
    import numpy as np

    if not callable(fn):
        raise CudaPlacementError("invalid_availability", f"{what}: an availability callable is required")
    value = fn()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise CudaPlacementError("invalid_availability", f"{what}: the reading must be a non-Boolean integer >= 0, got {value!r}")
    return int(value)


def _count(n_intervals: Any) -> int:
    """STRICT interval count: a plain/NumPy integer >= 1 (no bool, no float, no truncation)."""
    import numpy as np

    if isinstance(n_intervals, (bool, np.bool_)) or not isinstance(n_intervals, (int, np.integer)) or int(n_intervals) < 1:
        raise CudaPlacementError("invalid_interval_count", f"n_intervals must be a non-Boolean integer >= 1, got {n_intervals!r}")
    return int(n_intervals)


def event_extra_bytes(lop: Any, n_intervals: int) -> int:
    """Arithmetic of `water_lifecycle._event_extra_bytes` for a schedule of `n_intervals` float64 intervals (edges 8(n+1), intensity 8n, rate 8n bytes). Pure integers; no array, no schedule
    object. Equality with the real function on REAL `RainfallSchedule`s is pinned by CPU tests (real CPU lifecycle) and by the 109 source pins."""
    n = _count(n_intervals)
    N, L, F = int(lop.n_cells), int(lop.n_layers), int(lop.memory.n_faces)
    extra = (16 * 8 + 1 + L * 8) * N + 4 * (8 * (n + 1)) + 2 * (8 * n) + 2 * (8 * n) + 4 * F + 4096
    extra += ((L + 3) * N + F) * 8
    if lop.context is not None:
        extra += N * 8
    return int(extra)


def begin_storm_raw_bytes(lop: Any, n_intervals: int) -> int:
    """The raw device figure of `water_lifecycle._raw_begin_storm_bytes` from an integer interval count (pure arithmetic over the lifecycle's sealed `memory`; works on a real CPU lifecycle too,
    which is how the CPU tests compare it with the real function on REAL `RainfallSchedule`s)."""
    m = lop.memory
    return int(m.storm_required_bytes + m.owned_fixed_bytes + m.begin_copy_bytes + max(m.storm["build_bytes"], m.storm["initialization_bytes"], event_extra_bytes(lop, _count(n_intervals))))


def begin_storm_plan(lop: Any, n_intervals: int) -> dict:
    """The complete host+device plan of the CUDA `begin_storm` entry from INTEGER primitives (no schedule/array/duck): the authoritative pure composer `water_lifecycle_cuda.compose_entry_plan`
    with the raw device figure of `water_lifecycle._raw_begin_storm_bytes` (same arithmetic; its strict `RainfallSchedule` type guard is NOT bypassed: no schedule object is made or passed),
    the sealed lifecycle's own `host_reserve_bytes`/profile record bytes, the actual source-cold flag and the schedule byte sizes (8(n+1), 8n, 8n). The genuine-lifecycle/source/cover gates are
    `water_lifecycle._check_lop(lop, preflight=False)` (no device query). The result must equal `wl.cuda_entry_plan(lop, "begin_storm", schedule=<real schedule>)` EXACTLY (GPU test)."""
    from maple_syrup import interevent_transpiration as it
    from maple_syrup import water_lifecycle as wl
    from maple_syrup import water_lifecycle_cuda as wlc

    n = _count(n_intervals)
    try:
        wl._check_lop(lop, preflight=False)
    except wl.LifecycleError as exc:
        raise CudaPlacementError("lifecycle_refused", str(exc)) from None
    cap = lop._cuda
    if cap is None:
        raise CudaPlacementError("lifecycle_refused", "begin_storm_plan exists only for a CUDA lifecycle")
    m = lop.memory
    raw = begin_storm_raw_bytes(lop, n)
    return wlc.compose_entry_plan(lop.n_layers, lop.n_cells, "begin_storm", device_raw_bytes=raw, host_reserve_bytes=cap.host_reserve_bytes, source_cold=it._windowed_source_cold(),
                                  schedule_bytes=(8 * (n + 1), 8 * n, 8 * n), n_faces=m.n_faces, storm_host_stage_bytes=int(m.storm.get("host_bytes", 0)), interevent_build_host_bytes=0,
                                  source_record_bytes=wl._cuda_source_record_bytes(cap.profile))


def begin_storm_figures(lop: Any, n_intervals: int) -> tuple[int, int]:
    plan = begin_storm_plan(lop, n_intervals)
    return int(plan["host_bytes"]), int(plan["device_bytes"])


def estimate_load_metadata(lop: Any, *, expected_role: str, reference_bed: Any, n_intervals: Any, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int,
                           extra_reserve_bytes: int = 0) -> Any:
    """The ACTUAL V2 load budget (`CudaLoadBudget`) of a boundary or COMPLETED-storm restore WITHOUT any rainfall field/StormSpec/schedule/duck. Fences in the facade's order (reserves, exact role,
    `_fence_lop`, current-device owner gate, `_fence_reference_bed`), then the real `ld._gather` (boundary leaf arithmetic), and for the storm role EXACTLY the composition `_gather_v2` performs —
    `dataclasses.replace(save.role='storm')` and `StormLoadTerms(begin_storm host/device from `begin_storm_plan`, event_extra_bytes, storm_audit_device_bytes)` — but from integer primitives
    (`begin_storm_plan`, `event_extra_bytes`) instead of a schedule object; the real V2 planner then composes the budget. `n_intervals` is the strict interval count for the storm role, None for
    a boundary. The real StormSpec is validated by the actual loader later (no fence bypassed). Exact equality with the facade/loader is a GPU test; the transitive water109 pins are CPU tests."""
    import dataclasses

    from maple_syrup import lifecycle_checkpoint_cuda as cs
    from maple_syrup import lifecycle_checkpoint_cuda_load as ld
    from maple_syrup import lifecycle_checkpoint_cuda_load_plan as lp
    from maple_syrup import lifecycle_checkpoint_cuda_plan as sp
    from maple_syrup import lifecycle_checkpoint_cuda_v2 as v2
    from maple_syrup import lifecycle_checkpoint_cuda_v2_plan as v2plan
    from maple_syrup import subsurface as ss
    from maple_syrup import water_lifecycle as wl

    hr = cs._reserve(caller_host_reserve_bytes, "caller_host_reserve_bytes")
    dr = cs._reserve(caller_device_reserve_bytes, "caller_device_reserve_bytes")
    ex = cs._reserve(extra_reserve_bytes, "extra_reserve_bytes")
    if type(expected_role) is not str or expected_role not in ("boundary", "storm"):
        cs._refuse("the calendar restores only a closed 'boundary' or a COMPLETED 'storm'")
    if (expected_role == "storm") != (n_intervals is not None):
        cs._refuse("an interval count is required for (and only for) the storm role")
    if n_intervals is not None:
        try:
            _count(n_intervals)
        except CudaPlacementError as exc:
            cs._refuse(str(exc))
    ld._fence_lop(lop)
    try:
        owner = wl._cuda_checkpoint_owner(lop)
    except (wl.LifecycleError, ss.SubsurfaceError) as exc:
        raise cs._convert(exc) from None
    if isinstance(owner.device_id, bool) or not isinstance(owner.device_id, int):
        cs._refuse("the lifecycle's bound device id is not an integer")
    ld._fence_reference_bed(lop, reference_bed, owner)
    try:
        meta = ld._gather(lop, owner, "boundary", None, reference_bed)  # the role-independent leaf arithmetic, exactly as `_gather_v2` does for a storm
        terms = None
        if expected_role == "storm":
            n = _count(n_intervals)
            plan = begin_storm_plan(lop, n)
            extra = event_extra_bytes(lop, n)
            mem = lop.memory
            audit = int(mem.storm["resident_bytes"] + extra + mem.reserves_bytes + mem.audit_scratch_bytes + mem.owned_fixed_bytes)
            meta = dataclasses.replace(meta, save=dataclasses.replace(meta.save, role="storm"))
            terms = v2plan.StormLoadTerms(begin_storm_device_bytes=int(plan["device_bytes"]), begin_storm_host_bytes=int(plan["host_bytes"]), event_extra_bytes=extra, storm_audit_device_bytes=audit)
        return v2.estimate_load_budget_cuda_v2(meta, caller_host_reserve_bytes=hr, caller_device_reserve_bytes=dr, extra_reserve_bytes=ex, storm=terms)
    except CudaPlacementError as exc:
        raise cs._convert(exc) from None
    except (wl.LifecycleError, sp.CudaSavePlanError, lp.CudaLoadPlanError) as exc:
        raise cs._convert(exc) from None
