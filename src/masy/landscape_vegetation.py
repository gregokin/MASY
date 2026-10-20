"""Dynamic raster vegetation envelope for the Dense CPU profile (MASY51 phase 2).

One `LandscapeVegetation` owns exactly one native MAPLE `VegetationState`
(raster form). It adds identity and transaction metadata only; the native
container stays the only physical-field container, and the native
`validate_vegetation_state` / `vegetation_content_hash` stay the authority.

Scope (first hook):

- explicit authored +/- canonical replacements only (`CellReplacement`); no
  growth, mortality, uptake, ET, response law or soil-water policy;
- `source="raster"` and `source="none"` (the first accepted delta declares the
  transition `none->raster` in its receipt); `source="objects"` is refused;
- the native raster schema is exactly the five declared fields; any other
  holder/field layout refuses (`requalification_required`);
- a typed joint-resource seam (`JointResourceCandidate`, `WaterExchange`):
  only the empty exchange is admitted, and the candidate soil object is
  returned unchanged (`is`); any named channel refuses, even at zero.

Identity: `native_content_sha256` is the native `vegetation_content_hash`;
`version_sha256 = sha256(schema, parent version_sha256, delta_id,
native_content_sha256)`. The two are distinct and never compared.

Use (the later bed/runner integration):

    plan = build_vegetation_plan(deltas, shape=(ny, nx), n_classes=..., n_voxels=...,
                                 event_kinds=("rain", "wind", ...), schedule_sha256=...)
    env = LandscapeVegetation.from_native(native_state, plan=plan)
    # after event k, with the COMPLETE candidate ClockLedger.as_record():
    binding = BoundaryBinding.from_ledger_record(record, event_index=k, event_kind=plan.event_kinds[k],
                                                  schedule_sha256=plan.schedule_sha256)
    prepared = prepare_vegetation(env, binding, resource=JointResourceCandidate(soil))
    # ... all other fallible preparation; then publish prepared.envelope by reference.

`prepare_vegetation` is pure: it never mutates the envelope, the caller's
input, the soil or any native pool. Changed fields only are copied; arrays the
envelope holds are read-only owned copies; public native states are safe copies. Receipts and the cursor are plain
bounded metadata (no arrays, no parent envelope). Private-attribute tampering
is detected by native-hash rechecks (`verify`), not prevented; this is an integrity contract, not a Python security boundary.

This module imports only the standard library; NumPy and MAPLE are imported
lazily inside the explicit APIs that need them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from types import SimpleNamespace
from typing import Any

from masy.clock import ClockError, ClockLedger

__all__ = [
    "BINDING_SCHEMA",
    "DELTA_SCHEMA",
    "ENVELOPE_SCHEMA",
    "EVENT_KINDS",
    "PLAN_SCHEMA",
    "PROPOSERS",
    "RASTER_FIELDS",
    "REASONS",
    "RECEIPT_SCHEMA",
    "VEGETATION_PROFILE",
    "VERSION_SCHEMA",
    "BoundaryBinding",
    "CellReplacement",
    "JointResourceCandidate",
    "LandscapeVegetation",
    "PlannedDelta",
    "PreparedVegetation",
    "VegetationError",
    "VegetationLimits",
    "VegetationPlan",
    "WaterExchange",
    "build_vegetation_plan",
    "candidate_peak_bytes",
    "prepare_vegetation",
]

VEGETATION_PROFILE = "masy.dense_vegetation.dynamic_raster.v1"
PLAN_SCHEMA = "masy.vegetation_plan.v1"
DELTA_SCHEMA = "masy.vegetation_delta.v1"
VERSION_SCHEMA = "masy.landscape_vegetation_version.v1"
RECEIPT_SCHEMA = "masy.vegetation_receipt.v1"
BINDING_SCHEMA = "masy.vegetation_binding.v1"
ENVELOPE_SCHEMA = "masy.landscape_vegetation.v1"
CASE_SCHEMA_V2 = "masy.dense_case.v2"
BED_IDENTITY_SCHEMA_V2 = "masy.dense_bed_identity.v2"
STATIC_SCHEMA_V2 = "masy.dense_runner_static.v2"
RECEIPT_SCHEMA_V2 = "masy.dense_runner_event.v2"
REPORT_SCHEMA_V2 = "masy.dense_runner_report.v2"

# Exact native raster schema (MAPLE VegetationRasterState); anything else needs requalification.
RASTER_FIELDS = (
    "height_m",
    "cover_fraction",
    "porosity",
    "drag_coefficient",
    "frontal_area_density",
)
_STATE_FIELDS = ("source", "raster", "objects")
_ADMITTED_SOURCES = frozenset({"none", "raster"})
EVENT_KINDS = frozenset({"wind", "rain"})
_PROPOSER_BY_EVENT = {"wind": "wind", "rain": "water"}
PROPOSERS = frozenset({"wind", "water", "prescribed"})
REASONS = frozenset(
    {"prescribed_increase", "prescribed_decrease", "prescribed_replacement"}
)
_PAYLOAD_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_F64 = 8
_N_FIELDS = len(RASTER_FIELDS)
# Peak coexisting per-cell float64 arrays within one event: initial v0, accepted,
# current candidate, next changed copy (each F fields), plus water cover copy and
# public snapshot: (4F + 2).
_PEAK_ARRAYS = 4 * _N_FIELDS + 2
# Declared reserves (not measured): native validation boolean/compare temporaries
# (<= 4 full-cell float64-sized temporaries at a time) and fixed metadata.
_VALIDATION_RESERVE_ARRAYS = 4
_METADATA_RESERVE_BYTES = 16 * 1024


class VegetationError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _native() -> SimpleNamespace:
    """Actual native symbols, imported lazily."""
    import numpy as np
    from maple.core.types.vegetation import (
        DEFAULT_VEGETATION_DRAG_COEFFICIENT,
        DEFAULT_VEGETATION_POROSITY,
        VegetationRasterState,
        VegetationState,
        validate_vegetation_raster_state,
        validate_vegetation_state,
        vegetation_content_hash,
    )

    return SimpleNamespace(
        np=np,
        DEFAULT_VEGETATION_DRAG_COEFFICIENT=DEFAULT_VEGETATION_DRAG_COEFFICIENT,
        DEFAULT_VEGETATION_POROSITY=DEFAULT_VEGETATION_POROSITY,
        VegetationRasterState=VegetationRasterState,
        VegetationState=VegetationState,
        validate_vegetation_raster_state=validate_vegetation_raster_state,
        validate_vegetation_state=validate_vegetation_state,
        vegetation_content_hash=vegetation_content_hash,
    )


# Closed metadata admission. These limits concern logical JSON/nodes, not Python heap
# measurement; the qualification worker also enforces an actual 4 GiB OS AS cap.
_MAX_JSON_NODES = 120_000
_MAX_STRING_CHARS = 2048  # covers native finite-binary64 rational time records
_MAX_JSON_DEPTH = 12
_CLOCK_RECORD_BYTES = 1024 * 1024
# Source upper bounds: a cell record has six keys, two <=63 indices, and
# five <=24-character binary64 hex values (<256 bytes, <24 nodes).
# A delta's fixed fields, payload128/reason256 caps, digest wrapper fit1024.
_PLAN_FIXED_BYTES = 4096
_CELL_RECORD_BYTES = 512
_DELTA_RECORD_BYTES = 1024
_RECEIPT_RECORD_BYTES = 2048
_CURSOR_RECORD_BYTES = 1024
_ENVELOPE_FIXED_BYTES = 2048
_BOUNDARY_DIGEST_RECORD_BYTES = 512


def _json_admit(
    value: Any, *, max_bytes: int = 4 * 1024 * 1024, max_nodes: int = _MAX_JSON_NODES
) -> int:
    """Count plain closed-size JSON BEFORE encoder/parser allocation.

    No repr or conversion of arbitrary objects/integers. Container lengths,
    depth, node count and strings are checked before their children. Exact
    ASCII-escaped string lengths are counted without constructing strings.
    """
    nodes = 0
    total = 0

    def add(amount: int) -> None:
        nonlocal total
        total += amount
        if total > max_bytes:
            raise VegetationError(
                "metadata_cap_exceeded", "metadata byte admission exceeded"
            )

    def walk(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes or depth > _MAX_JSON_DEPTH:
            raise VegetationError(
                "metadata_cap_exceeded", "metadata node/depth admission exceeded"
            )
        kind = type(item)
        if kind is str:
            if len(item) > _MAX_STRING_CHARS:
                raise VegetationError(
                    "metadata_cap_exceeded", "metadata string admission exceeded"
                )
            add(2)
            for char in item:
                code = ord(char)
                add(
                    2
                    if char in '"\\\b\f\n\r\t'
                    else 6
                    if code < 32 or 127 <= code <= 65535
                    else 12
                    if code > 65535
                    else 1
                )
        elif item is None:
            add(4)
        elif kind is bool:
            add(4 if item else 5)
        elif kind is int:
            if item.bit_length() > 63:
                raise VegetationError(
                    "metadata_cap_exceeded", "metadata integer admission exceeded"
                )
            add(len(str(item)))
        elif kind in (list, dict):
            if len(item) > 4096:
                raise VegetationError(
                    "metadata_cap_exceeded", "metadata container admission exceeded"
                )
            add(2 + max(0, len(item) - 1))
            if kind is list:
                for child in item:
                    walk(child, depth + 1)
            else:
                for key, child in item.items():
                    if type(key) is not str:
                        raise VegetationError(
                            "invalid_metadata", "metadata keys must be plain strings"
                        )
                    walk(key, depth + 1)
                    add(1)
                    walk(child, depth + 1)
        else:
            raise VegetationError(
                "invalid_metadata", "metadata must be bounded plain JSON data"
            )

    walk(value, 0)
    return total


def _canonical_json(value: Any) -> str:
    _json_admit(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise VegetationError(
            "unhashable_metadata", "metadata encoding refused"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _plain_int(
    value: Any, name: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise VegetationError(
            "invalid_input",
            f"{name} must be a plain integer within the declared bounds",
        )
    return value


def _bounded_str(value: Any, name: str, max_chars: int) -> str:
    if type(value) is not str:
        raise VegetationError("invalid_input", f"{name} must be a plain string")
    if len(value) > max_chars:
        raise VegetationError(
            "metadata_cap_exceeded", f"{name} exceeds its character bound"
        )
    return value


def _digest(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or not _SHA256.fullmatch(value):
        raise VegetationError(
            "invalid_input", f"{name} must be a lowercase sha256 digest"
        )
    return value


def _canonical_float(value: Any, name: str) -> float:
    """Plain finite binary64 or exactly representable plain int; no bool/-0."""
    if type(value) is int:
        if value.bit_length() > 1024:
            raise VegetationError(
                "noncanonical_value", f"{name} integer exceeds binary64 admission"
            )
        try:
            converted = float(value)
        except OverflowError as exc:
            raise VegetationError(
                "noncanonical_value", f"{name} integer exceeds binary64 admission"
            ) from exc
        if converted != value:
            raise VegetationError(
                "noncanonical_value", f"{name} integer is not exactly binary64"
            )
        value = converted
    if type(value) is not float:
        raise VegetationError(
            "noncanonical_value", f"{name} must be a plain binary64 value"
        )
    if not math.isfinite(value):
        raise VegetationError("nonfinite_value", f"{name} must be finite")
    if value == 0.0 and math.copysign(1.0, value) < 0:
        raise VegetationError("noncanonical_value", f"{name} is negative zero")
    return value


# --------------------------------------------------------------------------- limits

_CEILINGS = {
    "max_cells": 64,
    "max_classes": 6,
    "max_voxels": 16,
    "max_events": 8,
    "max_deltas_per_boundary": 8,
    "max_total_updates": 64,
    "max_cells_per_delta": 64,
    "max_receipts": 64,
    "max_reason_chars": 256,
    "max_payload_id_chars": 128,
    "max_candidate_peak_bytes": 64 * 1024,
    "max_event_copy_bytes": 64 * 1024,
    "max_run_copy_bytes": 512 * 1024,
    "max_plan_record_bytes": 1024 * 1024,
    "max_metadata_bytes": 4 * 1024 * 1024,
}


@dataclasses.dataclass(frozen=True)
class VegetationLimits:
    """Finite engineering caps. Defaults are the source ceilings; a caller may
    choose tighter plain-int values but never larger ones."""

    max_cells: int = _CEILINGS["max_cells"]
    max_classes: int = _CEILINGS["max_classes"]
    max_voxels: int = _CEILINGS["max_voxels"]
    max_events: int = _CEILINGS["max_events"]
    max_deltas_per_boundary: int = _CEILINGS["max_deltas_per_boundary"]
    max_total_updates: int = _CEILINGS["max_total_updates"]
    max_cells_per_delta: int = _CEILINGS["max_cells_per_delta"]
    max_receipts: int = _CEILINGS["max_receipts"]
    max_reason_chars: int = _CEILINGS["max_reason_chars"]
    max_payload_id_chars: int = _CEILINGS["max_payload_id_chars"]
    max_candidate_peak_bytes: int = _CEILINGS["max_candidate_peak_bytes"]
    max_event_copy_bytes: int = _CEILINGS["max_event_copy_bytes"]
    max_run_copy_bytes: int = _CEILINGS["max_run_copy_bytes"]
    max_plan_record_bytes: int = _CEILINGS["max_plan_record_bytes"]
    max_metadata_bytes: int = _CEILINGS["max_metadata_bytes"]

    def __post_init__(self) -> None:
        for name, ceiling in _CEILINGS.items():
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise VegetationError(
                    "invalid_limit", f"{name} must be a plain int >= 1"
                )
            if value > ceiling:
                raise VegetationError(
                    "limit_above_ceiling",
                    f"{name} exceeds the source ceiling {ceiling}",
                )

    def as_record(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in _CEILINGS}


def candidate_peak_bytes(cells: int) -> int:
    """Declared conservative peak for one event: (4F+2)*8*cells plus reserves."""
    _plain_int(cells, "cells", minimum=1, maximum=_CEILINGS["max_cells"])
    return (
        _PEAK_ARRAYS + _VALIDATION_RESERVE_ARRAYS
    ) * _F64 * cells + _METADATA_RESERVE_BYTES


# --------------------------------------------------------------------------- authored plan


@dataclasses.dataclass(frozen=True)
class CellReplacement:
    """New canonical native values for one cell (all five raster fields).

    `height_m` is the CANONICAL cover-weighted height (native raster
    convention), not an authored plant height. A bare cell (cover 0) must
    carry height 0, frontal_area_density 0 and the native default porosity
    and drag constants exactly."""

    cell: tuple[int, int]
    cover_fraction: float
    height_m: float
    porosity: float
    drag_coefficient: float
    frontal_area_density: float

    def __post_init__(self) -> None:
        if type(self.cell) is not tuple or len(self.cell) != 2:
            raise VegetationError("invalid_input", "cell must be a (row, col) tuple")
        for axis in self.cell:
            _plain_int(axis, "cell index", maximum=_CEILINGS["max_cells"] - 1)
        for name in RASTER_FIELDS:
            object.__setattr__(self, name, _canonical_float(getattr(self, name), name))

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {"cell": list(self.cell)}
        out.update({name: getattr(self, name).hex() for name in RASTER_FIELDS})
        return out


@dataclasses.dataclass(frozen=True)
class WaterExchange:
    """Reserved typed seam for a future SYRUP-validated exchange. The first
    hook admits only the empty exchange; any named channel refuses (even 0)."""

    channels: tuple = ()

    def __post_init__(self) -> None:
        _refuse_water_exchange(self.channels)


def _refuse_water_exchange(channels: Any) -> None:
    if type(channels) is not tuple:
        raise VegetationError(
            "invalid_input",
            f"water_exchange must be a tuple, got {type(channels).__name__}",
        )
    if channels:
        raise VegetationError(
            "water_exchange_unsupported",
            "no water-exchange channel is supported by this profile, even at zero",
        )


@dataclasses.dataclass(frozen=True)
class PlannedDelta:
    """One authored delta, bound to the boundary after schedule event
    `after_event_index`; it takes effect from the following event."""

    proposer: str
    sequence: int
    payload_id: str
    after_event_index: int
    replacements: tuple[CellReplacement, ...]
    reason: str
    water_exchange: tuple = ()

    def __post_init__(self) -> None:
        if type(self.proposer) is not str or self.proposer not in PROPOSERS:
            raise VegetationError(
                "invalid_input", f"proposer must be one of {sorted(PROPOSERS)}"
            )
        _plain_int(
            self.sequence, "sequence", maximum=_CEILINGS["max_total_updates"] - 1
        )
        _plain_int(
            self.after_event_index,
            "after_event_index",
            maximum=_CEILINGS["max_events"] - 1,
        )
        _bounded_str(self.payload_id, "payload_id", _CEILINGS["max_payload_id_chars"])
        if not _PAYLOAD_ID.fullmatch(self.payload_id):
            raise VegetationError(
                "invalid_input",
                f"payload_id {self.payload_id!r} is not [a-z0-9][a-z0-9._:-]*",
            )
        _bounded_str(self.reason, "reason", _CEILINGS["max_reason_chars"])
        if self.reason not in REASONS:
            raise VegetationError(
                "invalid_input", f"reason must be one of {sorted(REASONS)}"
            )
        if type(self.replacements) is not tuple or not self.replacements:
            raise VegetationError(
                "invalid_input", "replacements must be a non-empty tuple"
            )
        if len(self.replacements) > _CEILINGS["max_cells_per_delta"]:
            raise VegetationError("metadata_cap_exceeded", "too many replacements")
        if not all(type(item) is CellReplacement for item in self.replacements):
            raise VegetationError(
                "invalid_input", "replacements must all be CellReplacement"
            )
        _refuse_water_exchange(self.water_exchange)

    def payload_record(self) -> dict[str, Any]:
        return {
            "schema": DELTA_SCHEMA,
            "proposer": self.proposer,
            "sequence": self.sequence,
            "payload_id": self.payload_id,
            "after_event_index": self.after_event_index,
            "replacements": [item.record() for item in self.replacements],
            "water_exchange": [],
            "reason": self.reason,
        }

    @property
    def payload_sha256(self) -> str:
        return _sha256(self.payload_record())


def _admit_plan_parts(
    deltas: Any,
    *,
    shape: Any,
    n_classes: Any,
    n_voxels: Any,
    event_kinds: Any,
    schedule_sha256: Any,
    limits: Any,
) -> int:
    """Raw source admission BEFORE record/native/JSON/list allocations."""
    if type(limits) is not VegetationLimits:
        raise VegetationError("invalid_input", "limits must be VegetationLimits")
    limits.__post_init__()
    if type(shape) is not tuple or len(shape) != 2:
        raise VegetationError("invalid_input", "shape must be a two-integer tuple")
    for axis in shape:
        _plain_int(axis, "shape axis", minimum=1, maximum=limits.max_cells)
    cells = shape[0] * shape[1]
    if cells > limits.max_cells:
        raise VegetationError(
            "domain_cap_exceeded", "cell count exceeds the declared bound"
        )
    _plain_int(n_classes, "n_classes", minimum=1, maximum=limits.max_classes)
    _plain_int(n_voxels, "n_voxels", minimum=1, maximum=limits.max_voxels)
    if type(event_kinds) is not tuple or not 1 <= len(event_kinds) <= limits.max_events:
        raise VegetationError(
            "domain_cap_exceeded", "event count exceeds the declared bound"
        )
    for kind in event_kinds:
        if type(kind) is not str or kind not in EVENT_KINDS:
            raise VegetationError(
                "invalid_input", "schedule event kinds must be wind/rain"
            )
    _digest(schedule_sha256, "schedule_sha256")
    if type(deltas) is not tuple or len(deltas) > min(
        limits.max_total_updates, limits.max_receipts
    ):
        raise VegetationError(
            "update_cap_exceeded", "delta tuple exceeds its declared bound"
        )
    size = _PLAN_FIXED_BYTES
    for delta in deltas:
        if type(delta) is not PlannedDelta:
            raise VegetationError(
                "invalid_input", "deltas must contain exact PlannedDelta values"
            )
        # Frozen dataclasses are not a Python security boundary. Recheck actual
        # held values before any payload_record traversal or native call.
        delta.__post_init__()
        _bounded_str(delta.payload_id, "payload_id", limits.max_payload_id_chars)
        _bounded_str(delta.reason, "reason", limits.max_reason_chars)
        if len(delta.replacements) > limits.max_cells_per_delta:
            raise VegetationError(
                "update_cap_exceeded", "replacement count exceeds the declared bound"
            )
        for item in delta.replacements:
            if type(item.cell) is not tuple or len(item.cell) != 2:
                raise VegetationError(
                    "invalid_input", "replacement cell must be a two-integer tuple"
                )
            for axis in item.cell:
                _plain_int(axis, "cell index", maximum=limits.max_cells - 1)
            for name in RASTER_FIELDS:
                value = getattr(item, name)
                if type(value) is not float:
                    raise VegetationError(
                        "noncanonical_value",
                        "held replacement values must be canonical floats",
                    )
                _canonical_float(value, name)
        size += _DELTA_RECORD_BYTES + len(delta.replacements) * _CELL_RECORD_BYTES
        if size > limits.max_plan_record_bytes:
            raise VegetationError(
                "metadata_cap_exceeded",
                "source plan byte bound exceeds max_plan_record_bytes",
            )
    if size > limits.max_plan_record_bytes:
        raise VegetationError(
            "metadata_cap_exceeded",
            "fixed plan byte bound exceeds max_plan_record_bytes",
        )
    if candidate_peak_bytes(cells) > limits.max_candidate_peak_bytes:
        raise VegetationError(
            "memory_cap_exceeded",
            "candidate array bound exceeds max_candidate_peak_bytes",
        )
    full_copy = _N_FIELDS * _F64 * cells
    if len(deltas) * full_copy > limits.max_run_copy_bytes:
        raise VegetationError(
            "copy_cap_exceeded", "run copy bound exceeds max_run_copy_bytes"
        )
    for index in range(len(event_kinds)):
        count = sum(delta.after_event_index == index for delta in deltas)
        if count > limits.max_deltas_per_boundary:
            raise VegetationError(
                "update_cap_exceeded", "boundary delta count exceeds its source bound"
            )
        if count * full_copy > limits.max_event_copy_bytes:
            raise VegetationError(
                "copy_cap_exceeded", "event copy bound exceeds max_event_copy_bytes"
            )
    return size


def _plan_record(
    deltas: tuple,
    *,
    shape: tuple,
    n_classes: int,
    n_voxels: int,
    event_kinds: tuple,
    schedule_sha256: str,
    limits: VegetationLimits,
) -> dict[str, Any]:
    return {
        "schema": PLAN_SCHEMA,
        "profile": VEGETATION_PROFILE,
        "shape": list(shape),
        "n_classes": n_classes,
        "n_voxels": n_voxels,
        "event_kinds": list(event_kinds),
        "schedule_sha256": schedule_sha256,
        "limits": limits.as_record(),
        "deltas": [
            {"payload": delta.payload_record(), "payload_sha256": delta.payload_sha256}
            for delta in deltas
        ],
    }


_PLAN_TOKEN = object()


class VegetationPlan:
    """Statically validated, immutable authored plan. Build with `build_vegetation_plan`."""

    __slots__ = (
        "_deltas",
        "_event_kinds",
        "_limits",
        "_n_classes",
        "_n_voxels",
        "_record_json",
        "_schedule_sha256",
        "_sha256",
        "_shape",
    )

    def __init__(self, token: object, **parts: Any) -> None:
        if token is not _PLAN_TOKEN:
            raise VegetationError(
                "private_constructor",
                "build a VegetationPlan with build_vegetation_plan",
            )
        for name, value in parts.items():
            object.__setattr__(self, "_" + name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise VegetationError("immutable", "a VegetationPlan is immutable")

    shape = property(lambda self: self._shape)
    cells = property(lambda self: self._shape[0] * self._shape[1])
    n_classes = property(lambda self: self._n_classes)
    n_voxels = property(lambda self: self._n_voxels)
    event_kinds = property(lambda self: self._event_kinds)
    schedule_sha256 = property(lambda self: self._schedule_sha256)
    limits = property(lambda self: self._limits)
    deltas = property(lambda self: self._deltas)
    plan_sha256 = property(lambda self: self._sha256)

    def verify(self) -> None:
        _admit_plan_parts(
            self._deltas,
            shape=self._shape,
            n_classes=self._n_classes,
            n_voxels=self._n_voxels,
            event_kinds=self._event_kinds,
            schedule_sha256=self._schedule_sha256,
            limits=self._limits,
        )
        if (
            type(self._record_json) is not str
            or len(self._record_json) > self._limits.max_plan_record_bytes
        ):
            raise VegetationError(
                "plan_externally_mutated", "held plan record exceeds its bound"
            )
        record = _plan_record(
            self._deltas,
            shape=self._shape,
            n_classes=self._n_classes,
            n_voxels=self._n_voxels,
            event_kinds=self._event_kinds,
            schedule_sha256=self._schedule_sha256,
            limits=self._limits,
        )
        text = _canonical_json(record)
        if (
            text != self._record_json
            or hashlib.sha256(text.encode("ascii")).hexdigest() != self._sha256
        ):
            raise VegetationError(
                "plan_externally_mutated",
                "held plan values differ from their canonical identity",
            )

    def plan_record(self) -> dict[str, Any]:
        self.verify()
        return json.loads(self._record_json)

    def deltas_at(self, event_index: int) -> tuple[PlannedDelta, ...]:
        _plain_int(event_index, "event_index", maximum=len(self._event_kinds) - 1)
        return tuple(
            delta for delta in self._deltas if delta.after_event_index == event_index
        )


def build_vegetation_plan(
    deltas: Any,
    *,
    shape: Any,
    n_classes: int,
    n_voxels: int,
    event_kinds: Any,
    schedule_sha256: str,
    limits: VegetationLimits | None = None,
) -> VegetationPlan:
    """Run every static check before any event: bounds, contiguity, uniqueness,
    proposer/kind, canonical FP64, native per-cell validity, bare-cell
    constants and all caps (including declared copy budgets)."""
    limits = VegetationLimits() if limits is None else limits
    _admit_plan_parts(
        deltas,
        shape=shape,
        n_classes=n_classes,
        n_voxels=n_voxels,
        event_kinds=event_kinds,
        schedule_sha256=schedule_sha256,
        limits=limits,
    )
    if type(limits) is not VegetationLimits:
        raise VegetationError("invalid_input", "limits must be VegetationLimits")
    if type(shape) is not tuple or len(shape) != 2:
        raise VegetationError(
            "invalid_input", f"shape must be a (ny, nx) tuple, got {shape!r:.80}"
        )
    ny, nx = (_plain_int(axis, "shape axis", minimum=1) for axis in shape)
    cells = ny * nx
    if cells > limits.max_cells:
        raise VegetationError(
            "domain_cap_exceeded", f"{cells} cells > max_cells {limits.max_cells}"
        )
    _plain_int(n_classes, "n_classes", minimum=1)
    if n_classes > limits.max_classes:
        raise VegetationError(
            "domain_cap_exceeded", f"{n_classes} classes > {limits.max_classes}"
        )
    _plain_int(n_voxels, "n_voxels", minimum=1)
    if n_voxels > limits.max_voxels:
        raise VegetationError(
            "domain_cap_exceeded", f"{n_voxels} voxels > {limits.max_voxels}"
        )
    if type(event_kinds) is not tuple or not event_kinds:
        raise VegetationError("invalid_input", "event_kinds must be a non-empty tuple")
    if len(event_kinds) > limits.max_events:
        raise VegetationError(
            "domain_cap_exceeded", f"{len(event_kinds)} events > {limits.max_events}"
        )
    if not all(type(kind) is str and kind in EVENT_KINDS for kind in event_kinds):
        raise VegetationError(
            "invalid_input", f"event kinds must be in {sorted(EVENT_KINDS)}"
        )
    if type(schedule_sha256) is not str or not _SHA256.match(schedule_sha256):
        raise VegetationError(
            "invalid_input", "schedule_sha256 must be a lowercase sha256 hex digest"
        )
    peak = candidate_peak_bytes(cells)
    if peak > limits.max_candidate_peak_bytes:
        raise VegetationError(
            "memory_cap_exceeded",
            f"candidate peak {peak} B > {limits.max_candidate_peak_bytes}",
        )

    if type(deltas) is not tuple:
        raise VegetationError("invalid_input", "deltas must be a tuple of PlannedDelta")
    if len(deltas) > limits.max_total_updates:
        raise VegetationError(
            "update_cap_exceeded", f"{len(deltas)} deltas > {limits.max_total_updates}"
        )
    if len(deltas) > limits.max_receipts:
        raise VegetationError(
            "receipt_cap_exceeded",
            f"{len(deltas)} deltas > max_receipts {limits.max_receipts}",
        )
    n = _native() if deltas else None
    payload_ids: set[str] = set()
    next_sequence: dict[str, int] = {}
    per_boundary: dict[int, int] = {}
    last_index = -1
    full_copy = _N_FIELDS * _F64 * cells
    for position, delta in enumerate(deltas):
        where = f"deltas[{position}]"
        if type(delta) is not PlannedDelta:
            raise VegetationError("invalid_input", f"{where} is not a PlannedDelta")
        if len(delta.payload_id) > limits.max_payload_id_chars:
            raise VegetationError(
                "metadata_cap_exceeded",
                f"{where}.payload_id exceeds {limits.max_payload_id_chars}",
            )
        if len(delta.reason) > limits.max_reason_chars:
            raise VegetationError(
                "metadata_cap_exceeded",
                f"{where}.reason exceeds {limits.max_reason_chars}",
            )
        if delta.after_event_index >= len(event_kinds):
            raise VegetationError(
                "invalid_boundary",
                f"{where} after_event_index {delta.after_event_index} is "
                f"outside the {len(event_kinds)}-event schedule",
            )
        if delta.after_event_index < last_index:
            raise VegetationError(
                "unordered_plan", f"{where} precedes an earlier boundary"
            )
        last_index = delta.after_event_index
        kind = event_kinds[delta.after_event_index]
        if (
            delta.proposer != "prescribed"
            and delta.proposer != _PROPOSER_BY_EVENT[kind]
        ):
            raise VegetationError(
                "proposer_kind_mismatch",
                f"{where} proposer {delta.proposer!r} at a {kind!r} event boundary",
            )
        if delta.payload_id in payload_ids:
            raise VegetationError(
                "duplicate_payload_id",
                f"{where} payload_id {delta.payload_id!r} repeats",
            )
        payload_ids.add(delta.payload_id)
        expected = next_sequence.get(delta.proposer, 0)
        if delta.sequence != expected:
            raise VegetationError(
                "sequence_not_contiguous",
                f"{where} {delta.proposer} sequence {delta.sequence} != expected {expected}",
            )
        next_sequence[delta.proposer] = expected + 1
        per_boundary[delta.after_event_index] = (
            per_boundary.get(delta.after_event_index, 0) + 1
        )
        if per_boundary[delta.after_event_index] > limits.max_deltas_per_boundary:
            raise VegetationError(
                "update_cap_exceeded",
                f"more than {limits.max_deltas_per_boundary} deltas after "
                f"event {delta.after_event_index}",
            )
        if len(delta.replacements) > limits.max_cells_per_delta:
            raise VegetationError(
                "update_cap_exceeded",
                f"{where} touches > {limits.max_cells_per_delta} cells",
            )
        seen: set[tuple[int, int]] = set()
        for item in delta.replacements:
            row, col = item.cell
            if row >= ny or col >= nx:
                raise VegetationError(
                    "cell_out_of_bounds", f"{where} cell {item.cell} outside {(ny, nx)}"
                )
            if item.cell in seen:
                raise VegetationError(
                    "duplicate_cell", f"{where} replaces cell {item.cell} twice"
                )
            seen.add(item.cell)
            _check_replacement_native(n, item, where)
    for index, count in per_boundary.items():
        if count * full_copy > limits.max_event_copy_bytes:
            raise VegetationError(
                "copy_cap_exceeded",
                f"event {index} may copy {count * full_copy} B > "
                f"{limits.max_event_copy_bytes}",
            )
    if len(deltas) * full_copy > limits.max_run_copy_bytes:
        raise VegetationError(
            "copy_cap_exceeded",
            f"run may copy {len(deltas) * full_copy} B > {limits.max_run_copy_bytes}",
        )

    record = _plan_record(
        deltas,
        shape=(ny, nx),
        n_classes=n_classes,
        n_voxels=n_voxels,
        event_kinds=event_kinds,
        schedule_sha256=schedule_sha256,
        limits=limits,
    )
    record_json = _canonical_json(record)
    if len(record_json) > limits.max_plan_record_bytes:
        raise VegetationError(
            "metadata_cap_exceeded",
            f"plan record {len(record_json)} B > {limits.max_plan_record_bytes}",
        )
    return VegetationPlan(
        _PLAN_TOKEN,
        shape=(ny, nx),
        n_classes=n_classes,
        n_voxels=n_voxels,
        event_kinds=event_kinds,
        schedule_sha256=schedule_sha256,
        limits=limits,
        deltas=deltas,
        record_json=record_json,
        sha256=hashlib.sha256(record_json.encode("ascii")).hexdigest(),
    )


def _check_replacement_native(
    n: SimpleNamespace, item: CellReplacement, where: str
) -> None:
    """Per-cell validity through the NATIVE raster validator on a 1x1 raster,
    plus the declared bare-cell rule (native default constants)."""
    np = n.np
    raster = n.VegetationRasterState(
        **{
            name: np.full((1, 1), getattr(item, name), dtype=np.float64)
            for name in RASTER_FIELDS
        }
    )
    try:
        n.validate_vegetation_raster_state(raster, 1, 1)
    except ValueError as exc:
        raise VegetationError(
            "invalid_replacement", f"{where} cell {item.cell}: {exc}"
        ) from exc
    if item.cover_fraction == 0.0:
        bare = {
            "height_m": 0.0,
            "frontal_area_density": 0.0,
            "porosity": n.DEFAULT_VEGETATION_POROSITY,
            "drag_coefficient": n.DEFAULT_VEGETATION_DRAG_COEFFICIENT,
        }
        wrong = sorted(
            name
            for name, value in bare.items()
            if getattr(item, name).hex() != float(value).hex()
        )
        if wrong:
            raise VegetationError(
                "bare_cell_not_canonical",
                f"{where} bare cell {item.cell} field(s) {wrong} are "
                "not the native bare-cell values",
            )


# --------------------------------------------------------------------------- binding


class BoundaryBinding:
    """Realized COMPLETE candidate ClockLedger after a sequential schedule event.

    Native from_record/as_record owns every clock invariant, redundant field,
    and returned-elapsed discrepancy. This profile admits one event entry per
    schedule event; idle/calendar entries are outside the current runner API.
    """

    __slots__ = ("_event_index", "_event_kind", "_json", "_schedule_sha256", "_sha256")

    def __init__(
        self,
        token: object,
        event_index: int,
        event_kind: str,
        schedule_sha256: str,
        record_json: str,
    ) -> None:
        if token is not _PLAN_TOKEN:
            raise VegetationError(
                "private_constructor", "use BoundaryBinding.from_ledger_record"
            )
        for name, value in (
            ("event_index", event_index),
            ("event_kind", event_kind),
            ("schedule_sha256", schedule_sha256),
            ("json", record_json),
            ("sha256", hashlib.sha256(record_json.encode("ascii")).hexdigest()),
        ):
            object.__setattr__(self, "_" + name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise VegetationError("immutable", "a BoundaryBinding is immutable")

    @classmethod
    def from_ledger_record(
        cls, record: Any, *, event_index: int, event_kind: str, schedule_sha256: str
    ) -> BoundaryBinding:
        _plain_int(event_index, "event_index", maximum=_CEILINGS["max_events"] - 1)
        if type(event_kind) is not str or event_kind not in EVENT_KINDS:
            raise VegetationError(
                "invalid_binding", "schedule event kind must be wind/rain"
            )
        _digest(schedule_sha256, "schedule_sha256")
        if type(record) is not dict or len(record) != 4:
            raise VegetationError(
                "invalid_binding", "a complete bounded ClockLedger record is required"
            )
        entries = record.get("entries")
        if type(entries) is not list or len(entries) != event_index + 1:
            raise VegetationError(
                "invalid_binding", "complete ledger length must equal event_index + 1"
            )
        # Raw node/string/byte bounds before native roundtrip, record construction,
        # or canonicalization. The native parser remains the only clock parser.
        _json_admit(record, max_bytes=_CLOCK_RECORD_BYTES, max_nodes=1024)
        try:
            ledger = ClockLedger.from_record(record)
            canonical = ledger.as_record()
        except (
            ClockError,
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
        ) as exc:
            raise VegetationError(
                "invalid_binding", "native complete ClockLedger validation refused"
            ) from exc
        if any(entry["kind"] != "event" for entry in canonical["entries"]):
            raise VegetationError(
                "invalid_binding", "only sequential event entries are supported"
            )
        if canonical["entries"][-1]["index"] != event_index:
            raise VegetationError("invalid_binding", "ledger/event index mismatch")
        body = {
            "schema": BINDING_SCHEMA,
            "event_index": event_index,
            "event_kind": event_kind,
            "schedule_sha256": schedule_sha256,
            "clock_ledger": canonical,
        }
        return cls(
            _PLAN_TOKEN, event_index, event_kind, schedule_sha256, _canonical_json(body)
        )

    event_index = property(lambda self: self._event_index)
    event_kind = property(lambda self: self._event_kind)
    schedule_sha256 = property(lambda self: self._schedule_sha256)
    binding_sha256 = property(lambda self: self._sha256)

    def verify(self) -> None:
        if type(self._json) is not str or len(self._json) > _CLOCK_RECORD_BYTES:
            raise VegetationError(
                "invalid_binding", "held binding exceeds its byte bound"
            )
        if not self._json.isascii():
            raise VegetationError(
                "invalid_binding", "binding canonical JSON must be ASCII"
            )
        if hashlib.sha256(self._json.encode("ascii")).hexdigest() != self._sha256:
            raise VegetationError("invalid_binding", "binding canonical bytes changed")
        record = json.loads(self._json)
        checked = type(self).from_ledger_record(
            record["clock_ledger"],
            event_index=self._event_index,
            event_kind=self._event_kind,
            schedule_sha256=self._schedule_sha256,
        )
        if checked._json != self._json:
            raise VegetationError(
                "invalid_binding", "binding fields differ from held canonical bytes"
            )

    def record(self) -> dict[str, Any]:
        self.verify()
        return json.loads(self._json)


# --------------------------------------------------------------------------- resource seam


@dataclasses.dataclass(frozen=True)
class JointResourceCandidate:
    """The event's candidate soil (the runner's separately owned soil object,
    i.e. the `_finish` candidate, possibly already changed by rain) plus the
    reserved exchange. Soil is opaque here and returned unchanged (`is`)."""

    soil: Any
    water_exchange: WaterExchange = WaterExchange()
    vegetation_version_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.water_exchange) is not WaterExchange:
            raise VegetationError(
                "invalid_input", "water_exchange must be a WaterExchange"
            )
        if self.vegetation_version_sha256 is not None:
            _digest(self.vegetation_version_sha256, "vegetation_version_sha256")


# --------------------------------------------------------------------------- envelope

_ENV_TOKEN = object()


def _check_native_layout(
    n: SimpleNamespace, state: Any, shape: tuple[int, int]
) -> None:
    if type(state) is not n.VegetationState:
        raise VegetationError(
            "unsupported_type",
            f"expected native VegetationState, got {type(state).__name__}",
        )
    if tuple(f.name for f in dataclasses.fields(state)) != _STATE_FIELDS:
        raise VegetationError(
            "requalification_required", "VegetationState field layout changed"
        )
    if state.source == "objects" or state.objects is not None:
        raise VegetationError(
            "objects_source_unsupported",
            "objects-sourced vegetation has no stable plant IDs",
        )
    if state.source not in _ADMITTED_SOURCES:
        raise VegetationError("unsupported_source", f"source {state.source!r:.40}")
    raster = state.raster
    if type(raster) is not n.VegetationRasterState:
        raise VegetationError(
            "requalification_required",
            f"raster holder {type(raster).__name__} is not native",
        )
    if tuple(f.name for f in dataclasses.fields(raster)) != RASTER_FIELDS:
        raise VegetationError(
            "requalification_required", "VegetationRasterState field schema changed"
        )
    for name in RASTER_FIELDS:
        array = getattr(raster, name)
        if type(array) is not n.np.ndarray:
            raise VegetationError(
                "unsupported_backend", f"raster.{name} must be a host numpy.ndarray"
            )
        if (
            array.dtype != n.np.float64
            or array.shape != shape
            or not array.flags.c_contiguous
        ):
            raise VegetationError(
                "invalid_raster",
                f"raster.{name} must be C-contiguous float64 {shape}, got "
                f"{array.dtype} {array.shape}",
            )
        if not n.np.isfinite(array).all():
            raise VegetationError("nonfinite_value", f"raster.{name} is not finite")


def _validate_native(n: SimpleNamespace, state: Any, shape: tuple[int, int]) -> str:
    try:
        n.validate_vegetation_state(state, shape[0], shape[1])
    except ValueError as exc:
        raise VegetationError("native_validation_failed", str(exc)) from exc
    return n.vegetation_content_hash(state)


def _owned_readonly(np: Any, array: Any) -> Any:
    owned = np.array(array, dtype=np.float64, order="C", copy=True)
    owned.setflags(write=False)
    return owned


def _version_sha256(parent: str | None, delta_id: str | None, native_sha: str) -> str:
    return _sha256(
        {
            "schema": VERSION_SCHEMA,
            "parent_version_sha256": parent,
            "delta_id": delta_id,
            "native_content_sha256": native_sha,
        }
    )


def _copy_native(n: SimpleNamespace, state: Any) -> Any:
    return n.VegetationState(
        source=state.source,
        raster=n.VegetationRasterState(
            **{
                name: _owned_readonly(n.np, getattr(state.raster, name))
                for name in RASTER_FIELDS
            }
        ),
        objects=None,
    )


def _envelope_admit(
    limits: VegetationLimits,
    *,
    receipts_bytes: int,
    bindings_bytes: int,
    last_binding_bytes: int,
    new_receipts: int = 0,
    new_boundary: bool = False,
) -> int:
    """Source logical-serialization bound BEFORE prospective records/array copies.

    Cursor has three fields, <=3 proposer counters and <=512KiB copy integer;
    each receipt's fixed23 fields, <=5 changed names, <=128 ID and <=256 reason
    fit2048 bytes. Digest-only history grows <=512B/boundary. Only the newest
    complete native clock binding is retained, never a parent/array chain.
    """
    size = (
        _ENVELOPE_FIXED_BYTES
        + _CURSOR_RECORD_BYTES
        + receipts_bytes
        + bindings_bytes
        + last_binding_bytes
        + new_receipts * _RECEIPT_RECORD_BYTES
        + (_BOUNDARY_DIGEST_RECORD_BYTES if new_boundary else 0)
    )
    if size > limits.max_metadata_bytes:
        raise VegetationError(
            "metadata_cap_exceeded",
            "prospective envelope metadata exceeds its source bound",
        )
    return size


class LandscapeVegetation:
    """Owned initial/current native states plus bounded sealed transaction metadata.

    Public native states are independent five-array copies. Private native
    borrows are trusted integration references verified again at a boundary;
    readonly flags and seals detect accidental tampering, not malicious Python.
    No parent envelope/array chain is retained. Arbitrary caller-retained public
    copies are outside the internal event peak; each snapshot copies5*8*cells.
    A future runner must charge its own required snapshots to work/traffic caps.
    """

    __slots__ = (
        "_bindings_json",
        "_current",
        "_cursor_json",
        "_initial",
        "_initial_native_sha256",
        "_last_binding_json",
        "_native_sha256",
        "_plan",
        "_receipts_json",
        "_seal_sha256",
        "_source_transition",
        "_version",
        "_version_sha256",
    )

    def __init__(self, token: object, **parts: Any) -> None:
        if token is not _ENV_TOKEN:
            raise VegetationError(
                "private_constructor", "use LandscapeVegetation.from_native"
            )
        for name, value in parts.items():
            object.__setattr__(self, "_" + name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise VegetationError(
            "immutable", "a LandscapeVegetation envelope is immutable"
        )

    @classmethod
    def from_native(cls, state: Any, *, plan: VegetationPlan) -> LandscapeVegetation:
        if type(plan) is not VegetationPlan:
            raise VegetationError("invalid_input", "plan must be a VegetationPlan")
        plan.verify()
        _envelope_admit(
            plan.limits, receipts_bytes=2, bindings_bytes=2, last_binding_bytes=0
        )
        n = _native()
        _check_native_layout(n, state, plan.shape)
        owned = _copy_native(n, state)
        native_sha = _validate_native(n, owned, plan.shape)
        if native_sha != n.vegetation_content_hash(state):
            raise VegetationError(
                "copy_mismatch", "owned copy differs from the native content hash"
            )
        cursor = {"next_event_index": 0, "last_sequence": {}, "copied_bytes": 0}
        env = cls(
            _ENV_TOKEN,
            plan=plan,
            initial=owned,
            initial_native_sha256=native_sha,
            current=owned,
            version=0,
            version_sha256=_version_sha256(None, None, native_sha),
            native_sha256=native_sha,
            receipts_json="[]",
            cursor_json=_canonical_json(cursor),
            source_transition=None,
            bindings_json="[]",
            last_binding_json=None,
        )
        env._seal()
        return env

    plan = property(lambda self: self._plan)
    version = property(lambda self: self._version)
    version_sha256 = property(lambda self: self._version_sha256)
    native_content_sha256 = property(lambda self: self._native_sha256)
    initial_native_content_sha256 = property(lambda self: self._initial_native_sha256)
    snapshot_copy_bytes = property(lambda self: _N_FIELDS * _F64 * self._plan.cells)

    @property
    def metadata_sha256(self) -> str:
        """Stored accepted metadata identity; call verify() before native work."""
        return self._seal_sha256

    @property
    def state(self) -> Any:
        """Caller-owned native snapshot, including all five independent arrays."""
        self.verify()
        return _copy_native(_native(), self._current)

    @property
    def initial_state(self) -> Any:
        """Caller-owned v0 native snapshot; no authority storage is exposed."""
        self.verify()
        return _copy_native(_native(), self._initial)

    def _native_borrow(self, *, initial: bool = False) -> Any:
        """Private trusted native handoff only; external mutation is checked later."""
        if type(initial) is not bool:
            raise VegetationError("invalid_input", "initial must be a bool")
        self.verify()
        return self._initial if initial else self._current

    @property
    def receipts(self) -> tuple[dict[str, Any], ...]:
        self._verify_metadata()
        return tuple(json.loads(self._receipts_json))

    @property
    def cursor(self) -> dict[str, Any]:
        self._verify_metadata()
        return json.loads(self._cursor_json)

    @property
    def bindings(self) -> tuple[dict[str, Any], ...]:
        self._verify_metadata()
        return tuple(json.loads(self._bindings_json))

    def cover_fraction_snapshot(self) -> Any:
        """One caller-owned writable array (8*cells bytes)."""
        self.verify()
        return _native().np.array(self._current.raster.cover_fraction, copy=True)

    def as_record(self) -> dict[str, Any]:
        self.verify()
        return {
            "schema": ENVELOPE_SCHEMA,
            "profile": VEGETATION_PROFILE,
            "plan_sha256": self._plan.plan_sha256,
            "version": self._version,
            "version_sha256": self._version_sha256,
            "native_content_sha256": self._native_sha256,
            "initial_native_content_sha256": self._initial_native_sha256,
            "source": self._current.source,
            "source_transition": self._source_transition,
            "receipts": json.loads(self._receipts_json),
            "cursor": json.loads(self._cursor_json),
            "bindings": json.loads(self._bindings_json),
            "last_boundary_binding": None
            if self._last_binding_json is None
            else json.loads(self._last_binding_json),
            "metadata_sha256": self._seal_sha256,
        }

    def _check_metadata(self) -> None:
        if type(self._plan) is not VegetationPlan:
            raise VegetationError(
                "accepted_metadata_externally_mutated", "plan type changed"
            )
        self._plan.limits.__post_init__()
        _digest(self._plan.plan_sha256, "plan_sha256")
        _plain_int(self._version, "version", maximum=self._plan.limits.max_receipts)
        for name in ("_version_sha256", "_native_sha256", "_initial_native_sha256"):
            _digest(getattr(self, name), name)
        if self._source_transition is not None and (
            type(self._source_transition) is not str
            or self._source_transition != "none->raster"
        ):
            raise VegetationError(
                "accepted_metadata_externally_mutated", "source transition changed"
            )
        size = _ENVELOPE_FIXED_BYTES
        for name in (
            "_receipts_json",
            "_cursor_json",
            "_bindings_json",
            "_last_binding_json",
        ):
            text = getattr(self, name)
            if text is None and name == "_last_binding_json":
                continue
            if (
                type(text) is not str
                or len(text) > self._plan.limits.max_metadata_bytes
            ):
                raise VegetationError(
                    "metadata_cap_exceeded", "held metadata exceeds its byte bound"
                )
            if not text.isascii():
                raise VegetationError(
                    "invalid_metadata", "held canonical JSON must be ASCII"
                )
            size += len(text)
        if size > self._plan.limits.max_metadata_bytes:
            raise VegetationError(
                "metadata_cap_exceeded", "held metadata exceeds max_metadata_bytes"
            )

    def _metadata_digest(self) -> str:
        self._check_metadata()
        digest = hashlib.sha256()
        # Framed exact canonical bytes: no JSON parse or combined JSON allocation.
        for value in (
            ENVELOPE_SCHEMA,
            self._plan.plan_sha256,
            str(self._version),
            self._version_sha256,
            self._native_sha256,
            self._initial_native_sha256,
            self._source_transition,
            self._receipts_json,
            self._cursor_json,
            self._bindings_json,
            self._last_binding_json,
        ):
            data = b"N" if value is None else b"S" + value.encode("ascii")
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
        return digest.hexdigest()

    def _seal(self) -> None:
        object.__setattr__(self, "_seal_sha256", self._metadata_digest())

    def _verify_metadata(self) -> None:
        self._check_metadata()
        if self._metadata_digest() != self._seal_sha256:
            raise VegetationError(
                "accepted_metadata_externally_mutated",
                "cursor/receipts/version/bindings changed",
            )

    def verify(self) -> None:
        self._plan.verify()
        self._verify_metadata()  # before decoding any held metadata
        n = _native()
        for label, state, expected in (
            ("initial", self._initial, self._initial_native_sha256),
            ("current", self._current, self._native_sha256),
        ):
            try:
                _check_native_layout(n, state, self._plan.shape)
                actual = _validate_native(n, state, self._plan.shape)
            except VegetationError as exc:
                raise VegetationError(
                    "accepted_state_externally_mutated",
                    label + " native validation failed",
                ) from exc
            if actual != expected:
                raise VegetationError(
                    "accepted_state_externally_mutated",
                    label + " native content changed",
                )
            if any(
                getattr(state.raster, name).flags.writeable for name in RASTER_FIELDS
            ):
                raise VegetationError(
                    "accepted_state_externally_mutated",
                    label + " native arrays became writable",
                )


@dataclasses.dataclass(frozen=True)
class PreparedVegetation:
    """Result of `prepare_vegetation`. `base` is the accepted envelope it was
    prepared from (lineage check at publication); `envelope` is the envelope
    to publish (`base` itself for a recognized duplicate)."""

    base: LandscapeVegetation
    envelope: LandscapeVegetation
    binding_sha256: str
    new_receipts: tuple[dict[str, Any], ...]
    resource: JointResourceCandidate
    duplicate: bool

    @property
    def base_version(self) -> int:
        return self.base.version

    @property
    def base_version_sha256(self) -> str:
        return self.base.version_sha256


def _delta_id(payload_sha: str, binding_sha: str, base_version_sha: str) -> str:
    return _sha256(
        {
            "schema": RECEIPT_SCHEMA,
            "payload_sha256": payload_sha,
            "binding_sha256": binding_sha,
            "base_version_sha256": base_version_sha,
        }
    )


def prepare_vegetation(
    env: LandscapeVegetation,
    binding: BoundaryBinding,
    *,
    resource: JointResourceCandidate,
) -> PreparedVegetation:
    """Pure preparation of the boundary after `binding.event_index`.

    Order: typed resource/exchange refusal; accepted envelope and binding
    integrity; current resource-version check; duplicate recognition (an already published boundary returns the CURRENT
    envelope and the retained receipts; a conflicting payload/binding refuses
    `delta_conflict`; nothing rewinds); cursor order; caps before allocation;
    then per delta: changed-field copy, native validate, native hash."""
    if type(env) is not LandscapeVegetation or type(binding) is not BoundaryBinding:
        raise VegetationError(
            "invalid_input", "env/binding must be LandscapeVegetation/BoundaryBinding"
        )
    if type(resource) is not JointResourceCandidate:
        raise VegetationError(
            "invalid_input", "resource must be a JointResourceCandidate"
        )
    resource.__post_init__()
    _refuse_water_exchange(resource.water_exchange.channels)
    env.verify()
    binding.verify()
    if (
        resource.vegetation_version_sha256 is not None
        and resource.vegetation_version_sha256 != env.version_sha256
    ):
        raise VegetationError(
            "resource_version_mismatch",
            "resource candidate refers to another vegetation version",
        )
    plan = env.plan
    index = binding.event_index
    if (
        index >= len(plan.event_kinds)
        or binding.event_kind != plan.event_kinds[index]
        or binding.schedule_sha256 != plan.schedule_sha256
    ):
        raise VegetationError(
            "binding_mismatch",
            f"event {index} kind {binding.event_kind!r} is not the plan's",
        )
    deltas = plan.deltas_at(index)
    cursor = env.cursor
    receipts = env.receipts
    bound = JointResourceCandidate(
        resource.soil, resource.water_exchange, env.version_sha256
    )

    if index < cursor["next_event_index"]:
        held = env.bindings[index]
        if (
            held["event_index"] != index
            or held["binding_sha256"] != binding.binding_sha256
        ):
            raise VegetationError(
                "delta_conflict", "boundary replay differs from its retained binding"
            )
        by_id = {receipt["payload_id"]: receipt for receipt in receipts}
        retained = []
        for delta in deltas:
            receipt = by_id.get(delta.payload_id)
            if (
                receipt is None
                or receipt["payload_sha256"] != delta.payload_sha256
                or receipt["binding_sha256"] != binding.binding_sha256
                or receipt["event_index"] != index
                or receipt["delta_id"]
                != _delta_id(
                    delta.payload_sha256,
                    binding.binding_sha256,
                    receipt["base_version_sha256"],
                )
            ):
                raise VegetationError(
                    "delta_conflict",
                    f"{delta.payload_id!r} at event {index} differs from the "
                    "published receipt",
                )
            retained.append(receipt)
        if not deltas and any(
            r["event_index"] == index for r in receipts
        ):  # pragma: no cover - plan-immutable
            raise VegetationError(
                "delta_conflict", "receipts at a boundary the plan leaves empty"
            )
        return PreparedVegetation(
            env, env, binding.binding_sha256, tuple(retained), bound, True
        )
    if index > cursor["next_event_index"]:
        raise VegetationError(
            "boundary_skipped",
            f"event {index} after cursor {cursor['next_event_index']}",
        )

    limits = plan.limits
    _envelope_admit(
        limits,
        receipts_bytes=len(env._receipts_json),
        bindings_bytes=len(env._bindings_json),
        last_binding_bytes=len(binding._json),
        new_receipts=len(deltas),
        new_boundary=True,
    )
    if index:
        prefix = binding.record()["clock_ledger"]
        prefix["entries"] = prefix["entries"][:-1]
        previous = BoundaryBinding.from_ledger_record(
            prefix,
            event_index=index - 1,
            event_kind=plan.event_kinds[index - 1],
            schedule_sha256=plan.schedule_sha256,
        )
        if previous._json != env._last_binding_json:
            raise VegetationError(
                "binding_history_conflict",
                "complete ledger changed an accepted earlier interval",
            )
    cells = plan.cells
    full_copy = _N_FIELDS * _F64 * cells
    if len(receipts) + len(deltas) > limits.max_receipts:
        raise VegetationError(
            "receipt_cap_exceeded", "receipts would exceed max_receipts"
        )
    if len(deltas) * full_copy > limits.max_event_copy_bytes:
        raise VegetationError("copy_cap_exceeded", "event copy budget")
    if cursor["copied_bytes"] + len(deltas) * full_copy > limits.max_run_copy_bytes:
        raise VegetationError("copy_cap_exceeded", "run copy budget")

    n = _native() if deltas else None
    current = (
        env._current
    )  # verified private borrow; public snapshots are independent copies
    version = env.version
    version_sha = env.version_sha256
    native_sha = env.native_content_sha256
    transition = env._source_transition
    last_sequence = dict(cursor["last_sequence"])
    copied = cursor["copied_bytes"]
    new_receipts = []
    for delta in deltas:
        if delta.sequence != last_sequence.get(delta.proposer, -1) + 1:
            raise VegetationError("sequence_not_contiguous", f"{delta.payload_id!r}")
        np = n.np
        fields = {name: getattr(current.raster, name) for name in RASTER_FIELDS}
        changed = []
        for name in RASTER_FIELDS:
            base_array = fields[name]
            if all(
                float(base_array[item.cell]).hex() == getattr(item, name).hex()
                for item in delta.replacements
            ):
                continue
            array = np.array(base_array, copy=True)
            for item in delta.replacements:
                array[item.cell] = getattr(item, name)
            array.setflags(write=False)
            fields[name] = array
            changed.append(name)
        step_transition = "none->raster" if current.source == "none" else None
        candidate = n.VegetationState(
            source="raster", raster=n.VegetationRasterState(**fields), objects=None
        )
        new_native = _validate_native(n, candidate, plan.shape)
        delta_id = _delta_id(delta.payload_sha256, binding.binding_sha256, version_sha)
        new_version_sha = _version_sha256(version_sha, delta_id, new_native)
        step_bytes = len(changed) * _F64 * cells
        copied += step_bytes
        new_receipts.append(
            {
                "schema": RECEIPT_SCHEMA,
                "payload_id": delta.payload_id,
                "proposer": delta.proposer,
                "sequence": delta.sequence,
                "reason": delta.reason,
                "payload_sha256": delta.payload_sha256,
                "event_index": index,
                "event_kind": binding.event_kind,
                "binding_sha256": binding.binding_sha256,
                "ledger_index": index,
                "base_version": version,
                "base_version_sha256": version_sha,
                "delta_id": delta_id,
                "version": version + 1,
                "version_sha256": new_version_sha,
                "native_content_sha256": new_native,
                "changed_fields": changed,
                "changed_cells": len(delta.replacements),
                "copied_bytes": step_bytes,
                "source_transition": step_transition,
                "water_exchange": [],
            }
        )
        last_sequence[delta.proposer] = delta.sequence
        transition = step_transition or transition
        current, version, version_sha, native_sha = (
            candidate,
            version + 1,
            new_version_sha,
            new_native,
        )
        del candidate, fields

    new_cursor = {
        "next_event_index": index + 1,
        "last_sequence": last_sequence,
        "copied_bytes": copied,
    }
    all_receipts = list(receipts) + new_receipts
    boundaries = [
        *env.bindings,
        {
            "event_index": index,
            "event_kind": binding.event_kind,
            "schedule_sha256": binding.schedule_sha256,
            "binding_sha256": binding.binding_sha256,
        },
    ]
    new_env = LandscapeVegetation(
        _ENV_TOKEN,
        plan=plan,
        initial=env._initial,
        initial_native_sha256=env.initial_native_content_sha256,
        current=current,
        version=version,
        version_sha256=version_sha,
        native_sha256=native_sha,
        receipts_json=_canonical_json(all_receipts),
        cursor_json=_canonical_json(new_cursor),
        source_transition=transition,
        bindings_json=_canonical_json(boundaries),
        last_binding_json=binding._json,
    )
    new_env._seal()
    out_resource = JointResourceCandidate(
        resource.soil, resource.water_exchange, version_sha
    )
    return PreparedVegetation(
        env,
        new_env,
        binding.binding_sha256,
        tuple(json.loads(_canonical_json(new_receipts))),
        out_resource,
        False,
    )
