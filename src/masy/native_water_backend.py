"""Private closed backend policy seam of the native water calendar (S1a: CPU behaviour byte-preserved, no GPU execution).

`CPU_V1` is the ONE sealed module-literal policy: an instance of a private final class that only this module can construct. A policy is selected by EXACT OBJECT
identity (`policy_of`): `None` means `CPU_V1`; copies, subclass instances, forged namespaces, mappings and strings (`"cpu"`, `"cuda"`, ...) are refused with a typed
`BackendPolicyError` before any side effect. It carries only code literals (the statements that used to sit inline in `native_water_calendar`/`native_water_checkpoint`);
there are no caller-selected formulas or hooks. The reserved CUDA capability is an explicit unimplemented refusal (`cuda_unimplemented`); nothing here imports CuPy.

The policy is stored on a session as the separate private handle `_backend` (never `policy`, which is the storm quiet policy) and is excluded from the CPU configuration
digest, records and artifacts, so CPU identities are unchanged. SYRUP is imported lazily inside methods; the module imports only the standard library.
"""

from __future__ import annotations

import sys
from typing import Any

__all__ = ["BackendPolicyError", "CPU_V1", "cuda_capability", "policy_of"]  # + module attributes WATER_ONLY_V1 / LEAF_WATER_V1 / participant_of (the old __all__ is pinned by old tests)  # unchanged (old seam tests pin it); `CUDA_V1`, `PROFILE_CAPABILITY`, `policy_for_profile` are module attributes

#: CLOSED pure-literal table (stdlib only, so the parent preflight may import it): registered water SOURCE PROFILE name -> declared backend capability.
#: The old profiles are CPU; the two new literal 109 profiles fix the SAME full water digest (see `native_water_calendar.SOURCE_PROFILES`) and differ only in capability.
CPU_PROFILES = ("c3_literal308", "native9f_layered_factory_95ef_cpu_v1", "native9f_layered_factory_afca_cpu_v1")
CUDA_PROFILES = ("native9f_layered_factory_afca_cuda_v1",)
PROFILE_CAPABILITY = {**{name: "cpu" for name in CPU_PROFILES}, **{name: "cuda" for name in CUDA_PROFILES}}


class BackendPolicyError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


_SEAL = object()


class _Policy:
    """Private final class: constructed only by this module (seal), immutable, not copyable, not subclassable."""

    __slots__ = ("name", "kind", "backends", "refuses_cupy", "cuda")

    def __init_subclass__(cls, **kw: Any) -> None:
        raise BackendPolicyError("policy_sealed", "the backend policy class cannot be subclassed")

    def __init__(self, seal: object, name: str, backends: tuple, refuses_cupy: bool, kind: str = "cpu") -> None:
        if seal is not _SEAL:
            raise BackendPolicyError("policy_sealed", "a backend policy is created only by native_water_backend")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "backends", tuple(backends))
        object.__setattr__(self, "refuses_cupy", bool(refuses_cupy))
        object.__setattr__(self, "cuda", "explicit_unimplemented" if kind == "cpu" else "implemented")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a backend policy is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a backend policy is immutable")

    def __copy__(self) -> "_Policy":
        raise BackendPolicyError("policy_sealed", "a backend policy cannot be copied")

    def __deepcopy__(self, memo: Any) -> "_Policy":
        raise BackendPolicyError("policy_sealed", "a backend policy cannot be copied")

    def __reduce__(self) -> Any:
        raise BackendPolicyError("policy_sealed", "a backend policy cannot be pickled or reconstructed")

    def __repr__(self) -> str:
        return f"<masy backend policy {self.name}>"

    # ---- seam 1: operator backend tuple (callers keep their own old error type/message) ----
    def backend_ok(self, backend: Any) -> bool:
        return backend in self.backends

    # ---- seam 2: imported-CuPy refusal of the namespace census ----
    def cupy_imported_refused(self) -> bool:
        return self.refuses_cupy and "cupy" in sys.modules

    # ---- seams 3/4: the dense preparation module (type, PROFILE_NAME, errors, plan) and the cold recheck ----
    def dense_profile(self) -> Any:
        if self.kind == "cuda":
            raise BackendPolicyError("cuda_no_cpu_module", "the CUDA policy has no CPU dense-profile module")
        from maple_syrup import native_dense_profile as nd
        return nd

    def prep_type(self) -> type:
        if self.kind == "cuda":
            from maple_syrup import native_dense_cuda_profile as ndc
            return ndc.NativeDenseCudaPreparation
        return self.dense_profile().NativeDensePreparation

    def prep_profile_name(self) -> str:
        if self.kind == "cuda":
            from maple_syrup import native_dense_cuda_profile as ndc
            return ndc.PROFILE_NAME
        return self.dense_profile().PROFILE_NAME

    def prep_errors(self) -> type:
        if self.kind == "cuda":
            from maple_syrup import native_dense_cuda_profile as ndc
            return ndc.NativeDenseCudaProfileError
        return self.dense_profile().NativeDenseProfileError

    def plan_prep_bytes(self, preparation: Any, device_id: Any = None) -> int:
        """Whole-call host requirement of the genuine preparation recheck (metadata only, reserve excluded)."""
        if self.kind == "cuda":
            from masy import native_water_cuda as nwc
            return nwc.plan_prep_host_bytes(preparation)
        nd = self.dense_profile()
        return int(nd.plan_preparation(preparation.admission, caller_reserve_bytes=0)["required_bytes"])

    def recheck(self, preparation: Any, reserve_bytes: int, fresh_reading: Any, device_reserve_bytes: int = 0, device_reading: Any = None) -> None:
        if self.kind == "cuda":
            preparation.recheck(caller_host_reserve_bytes=int(reserve_bytes), caller_device_reserve_bytes=int(device_reserve_bytes), host_available_bytes_fn=fresh_reading,
                                device_available_bytes_fn=device_reading)
            return
        preparation.recheck(caller_reserve_bytes=int(reserve_bytes), available_bytes_fn=fresh_reading)

    def recheck_errors(self) -> tuple:
        from maple_syrup import native_case_profile as ncp
        return (self.prep_errors(), ncp.NativeCaseProfileError)

    # ---- seam 5: host availability reader ----
    def host_available_fn(self, lop: Any) -> Any:
        if self.kind == "cuda":  # under CUDA `lop.column.available_bytes_fn` is the DEVICE reader; the host reader is the SYRUP default
            from maple_syrup import subsurface as ss
            return ss.default_host_available_bytes
        return lop.column.available_bytes_fn

    def device_available_fn(self, lop: Any) -> Any:
        if self.kind != "cuda":
            raise BackendPolicyError("no_device_reader", "the CPU policy has no device availability reader")
        return lop.column.available_bytes_fn

    def pet_array(self, lop: Any, rate: Any) -> Any:
        """The constant terrain PET `(N,)` handed to `evolve_interevent`: CPU = the NumPy array unchanged; CUDA = ONE GRID-sized `(N,)` float64 upload on the device per dated fragment (not fixed-size, never per hydraulic step)."""
        if self.kind == "cuda":
            from masy import native_water_cuda as nwc
            return nwc.device_packet(lop, rate)
        return rate

    # ---- seam 6: rainfall field namespace ----
    def rainfall_field(self, lop: Any) -> Any:
        if self.kind == "cuda":
            from masy import native_water_cuda as nwc
            return nwc.device_rainfall_field(lop)
        import numpy as np
        from maple_syrup.rainfall import rainfall_field
        return rainfall_field(*lop.shape, active_mask=np.asarray(lop.topology.active, dtype=bool), xp=np)

    # ---- seam 7: physical codec selection ----
    def codec(self) -> Any:
        from masy import native_water_checkpoint as nck
        return nck.cuda_codec_for(nck.CUDA_CODEC_ID, "cuda") if self.kind == "cuda" else nck.codec_for(nck.CODEC_ID, "cpu")

    def codec_for_manifest(self, codec_id: Any, device: Any) -> Any:
        """Closed lookup in THIS policy's own registry (a CPU artifact under CUDA and the reverse refuse here, before any payload)."""
        from masy import native_water_checkpoint as nck
        return nck.cuda_codec_for(codec_id, device) if self.kind == "cuda" else nck.codec_for(codec_id, device)

    # ---- CUDA-only budget/estimate/runtime seams (CPU methods reproduce the old inline statements) ----
    def estimate_save(self, codec: Any, lop: Any, role: str, state: Any, bed: Any, extra_ev: int, extra: int, dev: Any) -> Any:
        if self.kind == "cuda":
            return codec.estimate_save(lop, state, extra, device_reserve_bytes=None if dev is None else dev.device_reserve_bytes)
        return codec.estimate(lop, role, bed, extra_ev, extra)

    def estimate_load(self, codec: Any, lop: Any, role: str, reference_bed: Any, n_intervals: Any, extra_ev: int, extra: int, dev: Any) -> Any:
        if self.kind == "cuda":  # metadata only: `n_intervals` is the integer interval count (None for a boundary), never a schedule/StormSpec/field
            return codec.estimate_load(lop, role, reference_bed, n_intervals, extra, device_reserve_bytes=None if dev is None else dev.device_reserve_bytes)
        return codec.estimate(lop, role, reference_bed, extra_ev, extra)

    def codec_kw(self, lop: Any, dev: Any) -> dict:
        """Extra keyword arguments of `codec.save/load` (CPU: none)."""
        if self.kind != "cuda":
            return {}
        return {"device_reserve_bytes": None if dev is None else dev.device_reserve_bytes, "device_fn": lop.column.available_bytes_fn}

    def runtime_matches_early(self, manifest_writer: Any, lop: Any) -> bool:
        """Join of the sealed writer record BEFORE the payload. CPU: the fresh interpreter record must equal it. CUDA: the backend label only; the runtime record is joined after the
        codec's own load (its loader compares the manifest runtime with a fresh CUDA record and returns it in the receipt)."""
        if self.kind == "cuda":
            return manifest_writer["backend"] == lop.backend and isinstance(manifest_writer["runtime"], dict)
        return manifest_writer["backend"] == lop.backend and manifest_writer["runtime"] == self.runtime_record(lop)

    def runtime_matches_late(self, manifest_writer: Any, receipt: Any) -> bool:
        if self.kind != "cuda":
            return True
        return manifest_writer["runtime"] == dict(receipt["runtime"])

    @property
    def codec_profile(self) -> str:
        from masy import native_water_checkpoint as nck
        return nck.CUDA_PHYSICAL_PROFILE if self.kind == "cuda" else nck.PHYSICAL_PROFILE

    # ---- seam 8: codec error family (quota, base) ----
    def physical_errors(self) -> tuple:
        if self.kind == "cuda":
            from maple_syrup import lifecycle_checkpoint_cuda as cs
            return (cs.LifecycleCheckpointCudaQuotaError, cs.LifecycleCheckpointCudaError)
        from maple_syrup import lifecycle_checkpoint as lk
        return (lk.LifecycleCheckpointQuotaError, lk.LifecycleCheckpointError)

    def indeterminate_error(self) -> Any:
        if self.kind == "cuda":
            from maple_syrup import lifecycle_checkpoint_cuda as cs
            return cs.LifecycleCheckpointCudaIndeterminate
        from maple_syrup import lifecycle_checkpoint as lk
        return getattr(lk, "LifecycleCheckpointIndeterminate", None)

    # ---- seam 9: codec budget mapping (CPU budgets are attribute objects: identity) ----
    def map_budget(self, budget: Any) -> Any:
        if self.kind == "cuda":
            from masy import native_water_cuda as nwc
            return nwc.map_budget(budget)
        return budget

    # ---- seam 10: writer runtime record ----
    def runtime_record(self, lop: Any, receipt: Any = None) -> Any:
        if self.kind == "cuda":  # the CUDA runtime record comes from the codec's own receipt (never `runtime_profile("cuda")`, which is the CPU record)
            if receipt is None:
                raise BackendPolicyError("runtime_needs_receipt", "the CUDA runtime record is taken from the codec receipt")
            return dict(receipt.runtime if hasattr(receipt, "runtime") else receipt["runtime"])
        from maple_syrup import layered_checkpoint as lc_
        return lc_.runtime_profile(lop.backend)

    # ---- seam 11: closed physical member names (read path) ----
    @property
    def physical_members(self) -> tuple:
        from masy import native_water_checkpoint as nck
        return nck.CUDA_PHYSICAL_MEMBERS if self.kind == "cuda" else nck.PHYSICAL_MEMBERS

    # ---- seam 12/13: codec storm spec class and the checkpoint quota record type ----
    def storm_spec(self, schedule: Any, field: Any, origin_s: float, control: Any, policy: Any) -> Any:  # the same exact StormSpec type serves both codecs
        from maple_syrup import lifecycle_checkpoint as lk
        return lk.StormSpec(schedule, field, origin_s, control, policy)

    def checkpoint_limits_type(self) -> type:
        from maple_syrup import layered_checkpoint as lcp
        return lcp.CheckpointLimits


CPU_V1 = _Policy(_SEAL, "cpu_v1", ("numpy", "numba"), True, "cpu")
CUDA_V1 = _Policy(_SEAL, "cuda_v1", ("cuda",), False, "cuda")


def policy_of(obj: Any = None) -> _Policy:
    """Select a policy by EXACT object identity. `None` -> `CPU_V1`; `CUDA_V1` only when that exact module literal is passed; everything else is refused BEFORE any side effect."""
    if obj is None or obj is CPU_V1:
        return CPU_V1
    if obj is CUDA_V1:
        return CUDA_V1
    raise BackendPolicyError("policy_forged", f"only a sealed module-literal backend policy is accepted (got {type(obj).__name__}); copies, subclasses, mappings and names are refused")


def policy_for_profile(profile: Any) -> _Policy:
    """The module-literal policy of a REGISTERED source profile (closed table lookup; a name is never a policy by itself)."""
    if type(profile) is not str or profile not in PROFILE_CAPABILITY:
        raise BackendPolicyError("profile_unknown", f"source profile {profile!r} is not registered (registered: {sorted(PROFILE_CAPABILITY)})")
    return CUDA_V1 if PROFILE_CAPABILITY[profile] == "cuda" else CPU_V1


def cuda_capability(*_args: Any, **_kw: Any) -> Any:
    """The historical S1a reserved-capability refusal, preserved verbatim for the old CPU surface (the explicit CUDA policy is the module literal `CUDA_V1`, selected only by object identity)."""
    raise BackendPolicyError("cuda_unimplemented", "the CUDA calendar backend is reserved and not implemented in this stage (no GPU execution, no CuPy)")


class _Participant:
    """Private final sealed participant capability (orthogonal to the CPU/CUDA backend policy): `WATER_ONLY_V1` (the historical behaviour, the default) or `LEAF_WATER_V1`. Selected by EXACT object
    identity only; never inferred from a keyword, a class, a module name or duck attributes."""

    __slots__ = ("name",)

    def __init_subclass__(cls, **kw: Any) -> None:
        raise BackendPolicyError("participant_sealed", "the participant policy class cannot be subclassed")

    def __init__(self, seal: object, name: str) -> None:
        if seal is not _SEAL:
            raise BackendPolicyError("participant_sealed", "a participant policy is created only by native_water_backend")
        object.__setattr__(self, "name", name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a participant policy is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a participant policy is immutable")

    def __copy__(self) -> "_Participant":
        raise BackendPolicyError("participant_sealed", "a participant policy cannot be copied")

    def __deepcopy__(self, memo: Any) -> "_Participant":
        raise BackendPolicyError("participant_sealed", "a participant policy cannot be copied")

    def __reduce__(self) -> Any:
        raise BackendPolicyError("participant_sealed", "a participant policy cannot be pickled")


WATER_ONLY_V1 = _Participant(_SEAL, "water_only_v1")
LEAF_WATER_V1 = _Participant(_SEAL, "leaf_water_v1")


def participant_of(obj: Any = None) -> _Participant:
    """None -> `WATER_ONLY_V1`; otherwise EXACTLY one of the two module literals (copies, subclasses, names, mappings and look-alikes are refused before any side effect)."""
    if obj is None:
        return WATER_ONLY_V1
    if obj is WATER_ONLY_V1 or obj is LEAF_WATER_V1:
        return obj
    raise BackendPolicyError("participant_forged", "the participant must be exactly nwb.WATER_ONLY_V1 or nwb.LEAF_WATER_V1")
