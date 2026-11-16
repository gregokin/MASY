"""Explicit two-pool admission and metadata helpers for the Dense CUDA profile.

No native provider, allocation or GPU is imported at module import time.
These are engineering controls; all physical updates remain upstream.
"""
from __future__ import annotations

import dataclasses
import numbers
import sys
import uuid
from types import ModuleType


class GpuContractError(ValueError):
    pass


def count(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise GpuContractError(f"{name} must be a plain integer >= {minimum}")
    return value


@dataclasses.dataclass(frozen=True)
class GpuPolicy:
    device_id: int
    physical_uuid: str
    max_host_bytes: int
    max_device_bytes: int
    host_reserve_bytes: int
    device_reserve_bytes: int
    host_available_bytes_fn: object = dataclasses.field(repr=False, compare=False)
    device_available_bytes_fn: object = dataclasses.field(repr=False, compare=False)

    def __post_init__(self):
        count(self.device_id, "device_id")
        if type(self.physical_uuid) is not str or not self.physical_uuid.startswith("GPU-") or len(self.physical_uuid) != 40:
            raise GpuContractError("the parent-selected physical GPU UUID is required")
        try:
            if "GPU-" + str(uuid.UUID(self.physical_uuid[4:])) != self.physical_uuid:
                raise ValueError("noncanonical UUID")
        except ValueError:
            raise GpuContractError("canonical physical GPU UUID required") from None
        for name in ("max_host_bytes", "max_device_bytes"):
            count(getattr(self, name), name, 1)
        for name in ("host_reserve_bytes", "device_reserve_bytes"):
            count(getattr(self, name), name)
        if not callable(self.host_available_bytes_fn) or not callable(self.device_available_bytes_fn):
            raise GpuContractError("actual separate host and device available-byte readers are required")

    def record(self):
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if not f.name.endswith("_fn")}

    def gate(self):
        cp = sys.modules.get("cupy")
        if not isinstance(cp, ModuleType) or getattr(cp, "__name__", None) != "cupy":
            raise GpuContractError("the explicitly selected CUDA runtime must import genuine CuPy first")
        if getattr(getattr(cp, "ndarray", None), "__module__", "").partition(".")[0] != "cupy":
            raise GpuContractError("foreign CuPy array class")
        if int(cp.cuda.runtime.getDevice()) != self.device_id:
            raise GpuContractError("current CUDA device differs from this run; no automatic device switch")
        raw_uuid = cp.cuda.runtime.getDeviceProperties(self.device_id).get("uuid")
        if type(raw_uuid) is not bytes or len(raw_uuid) != 16 or "GPU-" + str(uuid.UUID(bytes=raw_uuid)) != self.physical_uuid:
            raise GpuContractError("actual CUDA physical UUID differs from the scoped run")
        return cp

    def admit(self, host, device, what, *, host_extra=0, device_extra=0):
        self.gate()  # before readers or array work
        hn = count(host, "host payload") + self.host_reserve_bytes + count(host_extra, "host extra")
        dn = count(device, "device payload") + self.device_reserve_bytes + count(device_extra, "device extra")
        if hn > self.max_host_bytes or dn > self.max_device_bytes:
            raise GpuContractError(f"{what} exceeds the declared separate host/device byte caps")
        ha, da = self.host_available_bytes_fn(), self.device_available_bytes_fn()
        count(ha, "host reader result")
        count(da, "device reader result")
        if hn > ha or dn > da:
            raise GpuContractError(f"{what} cannot fit its separate host/device memory admission")
        return hn, dn


def leaves(value, path="", out=None):
    """Metadata traversal; borrowed arrays may occur more than once conservatively."""
    if out is None:
        out = []
    if value is None or isinstance(value, (ModuleType, str, bytes, numbers.Number)):
        return out
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "nbytes"):
        out.append((path, value))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            leaves(getattr(value, f.name), path + "." + f.name, out)
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            leaves(v, f"{path}[{i}]", out)
    elif isinstance(value, dict):
        for k, v in value.items():
            leaves(v, f"{path}[{k!r}]", out)
    return out


def footprint(*values):
    """Structural host/device payload, counting repeated aliases conservatively."""
    host = device = 0
    for value in values:
        for _path, array in leaves(value):
            if hasattr(array, "__cuda_array_interface__"):
                device += int(array.nbytes)
            else:
                host += int(array.nbytes)
    return host, device


def device_leaves(value, policy, *, what="state"):
    cp = policy.gate()
    rows = leaves(value, what)
    for name, a in rows:
        if type(a) is not cp.ndarray or int(a.device.id) != policy.device_id:
            raise GpuContractError(f"{name} is not an exact array on the selected CUDA device")
        if a.dtype.hasobject:
            raise GpuContractError(f"{name} has an unsupported object dtype")
    return rows


def clone_device_tree(value, policy):
    """One bounded Dense stage. The caller admits the complete payload first."""
    cp = policy.gate()
    memo = {}

    def copy(v):
        if type(v) is cp.ndarray:
            if id(v) not in memo:
                memo[id(v)] = v.copy(order="C")
            return memo[id(v)]
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return dataclasses.replace(v, **{f.name: copy(getattr(v, f.name)) for f in dataclasses.fields(v) if f.init})
        if isinstance(v, tuple):
            values = tuple(copy(x) for x in v)
            return type(v)(*values) if hasattr(v, "_fields") else values
        if isinstance(v, list):
            return [copy(x) for x in v]
        if isinstance(v, dict):
            return {k: copy(x) for k, x in v.items()}
        return v

    return copy(value)


def host_vector(value, *, size):
    """Explicit bounded class/scalar packet, never a whole landscape download."""
    import numpy as np
    from maple.core.backend import to_host

    if int(getattr(value, "size", 1)) > size:
        raise GpuContractError("device receipt packet exceeds its declared class/scalar size")
    result = np.asarray(to_host(value) if hasattr(value, "__cuda_array_interface__") else value, dtype=np.float64).reshape(-1)
    if result.shape != (size,):
        raise GpuContractError("native class packet has another class count")
    return result
