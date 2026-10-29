"""Lossless HOST adoption of a Native compiled case into windowed storage.

This input boundary performs no event, settlement, hydraulic commit, device
placement or restart publication. Native compilation remains dense. The caller
can still own that dense input after this function returns.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import importlib
import json
import math
import os
import re
import stat
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

from masy import case_workflow as _case
from masy import dense_bed as _bed
from masy.case_water import (
    CaseWaterCopyLimits,
    CaseWaterError,
    CaseWaterSpec,
    _identity,
    _prepare_water_inputs,
)
from masy.dense_runner import DenseRunnerControl, DenseWaterInputs

__all__ = ["CaseWindowedError", "CaseWindowedLimits", "CaseWindowedInputs",
           "PreparedCaseWindowedWater", "adopt_windowed_case", "prepare_case_windowed_water",
           "save_initial_windowed_case", "reopen_initial_windowed_case"]

_TOKEN = object()
_PHYSICAL_FIELDS = {"voxel_column", "active_layer", "water", "sediment_ledger",
                    "committed_topography", "sediment_availability", "vegetation",
                    "threshold_modifier_state", "initial_mobile_state", "config"}


class CaseWindowedError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclasses.dataclass(frozen=True)
class CaseWindowedLimits:
    """Explicit input/known-owner ceilings, not a whole-process heap theorem."""
    max_cells: int
    max_classes: int
    max_voxels: int
    max_known_host_bytes: int
    max_metadata_bytes: int
    max_store_bytes: int

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or not 0 < value < (1 << 63):
                raise CaseWindowedError("invalid_limit", f"{field.name} requires a positive plain signed-63 integer")


def _bytes(value, maximum):
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    parts, total = [], 0
    for text in encoder.iterencode(value):
        part = text.encode("ascii")
        total += len(part)
        if total > maximum:
            raise CaseWindowedError("metadata_limit", "windowed case metadata exceeds its declared limit")
        parts.append(part)
    return b"".join(parts)


def _freeze(value):
    for _name, array in _bed._require_host_arrays(value, "windowed_case"):
        array.flags.writeable = False


def _column_record(column):
    version = column.deep_store
    binding = column.height_packet.binding
    return {"type": f"{type(column).__module__}.{type(column).__qualname__}",
            "arrays": {name: _bed._array_digest(getattr(column, name)) for name in
                       ("window_kg", "base", "deep_by_class_kg", "deep_total_kg", "deep_room_kg")},
            "prefix_packet": _bed._array_digest(column.height_packet.packet),
            "prefix_binding": {field.name: _case._canonical(getattr(binding, field.name))
                               for field in dataclasses.fields(binding) if field.name not in ("packet", "base")},
            "n_voxels": column.n_voxels, "generation": column.generation,
            "manifest_digest": column.manifest_digest, "header": version.header.to_json_dict(),
            "store_dir": str(version.store_dir), "header_digest": version.header.digest(),
            "pending_deep": None, "window_cache": None}


def _held_record(value):
    state = value.state
    return {"column": _column_record(state.voxel_column),
            "noncolumn": _case._canonical((state.active_layer, state.water, state.ledger,
                state.committed_topography, state.sediment_availability)),
            "context": _case._canonical(value.context), "config": _case._canonical(value.config),
            "vegetation": _case._canonical(value.vegetation),
            "threshold_modifiers": _case._canonical(value.threshold_modifiers),
            "initial_mobile_state": _case._canonical(value.initial_mobile_state),
            "ledger_reconciliation_baseline": _case._canonical(value.ledger_reconciliation_baseline),
            "static_inputs": _case._canonical(value.static_inputs), "time_s": _case._canonical(value.time_s)}


@dataclasses.dataclass(frozen=True)
class CaseWindowedInputs:
    """Owned HOST window plus noncolumn inputs and immutable provenance text.

    No CaseInputs, CompiledCase, DenseBed, bound method or closure is held.
    ``state`` is the genuine six-field SYRUP BedState. Its absent seventh
    landscape field is explicit below, and non-None baselines are refused.
    """
    state: Any
    context: Any
    config: Any
    vegetation: Any
    threshold_modifiers: Any
    initial_mobile_state: Any
    ledger_reconciliation_baseline: Any
    static_inputs: dict
    time_s: float
    identity_sha256: str
    _case_dir: Path = dataclasses.field(repr=False)
    _file_pins: Any = dataclasses.field(repr=False)
    _input_limits: Any = dataclasses.field(repr=False)
    _limits: CaseWindowedLimits = dataclasses.field(repr=False)
    _record_bytes: bytes = dataclasses.field(repr=False)
    _token: Any = dataclasses.field(repr=False)

    @property
    def case_sha256(self):
        return self.identity_sha256

    def identity_record(self):
        if hashlib.sha256(self._record_bytes).hexdigest() != self.identity_sha256:
            raise CaseWindowedError("identity_changed", "windowed input record changed")
        return json.loads(self._record_bytes)

    def verify(self):
        import numpy as np
        from maple.core.types.voxel import WindowedVoxelColumnState
        from maple.surface.voxels.deep_store import load_version
        from maple.surface.voxels.windowed_snapshot import validate_windowed_partition
        from maple_syrup.sediment_bed import BedContext, BedState

        if (self._token is not _TOKEN or type(self._limits) is not CaseWindowedLimits
                or type(self._input_limits) is not _case.CaseInputLimits
                or type(self.state) is not BedState or type(self.context) is not BedContext):
            raise CaseWindowedError("invalid_owner", "requires the genuine adopted HOST owner")
        self._limits.__post_init__()
        record = self.identity_record()
        if (record["limits"] != dataclasses.asdict(self._limits)
                or record["case_directory"] != str(self._case_dir)
                or record["file_pins"] != _case._canonical(self._file_pins)
                or record["input_limits"] != dataclasses.asdict(self._input_limits)):
            raise CaseWindowedError("identity_changed", "case file custody or declared limits changed")
        column = self.state.voxel_column
        if (type(column) is not WindowedVoxelColumnState or column.pending_deep is not None
                or column.window_cache is not None or column.height_packet is None
                or self.initial_mobile_state is not None or self.ledger_reconciliation_baseline is not None):
            raise CaseWindowedError("unsupported_state", "only the published initial HOST/baseline-None family is admitted")
        if (self.context.geometry is not self.config.geometry
                or self.context.grain_classes is not self.config.grain_classes
                or self.context.topographic_commit_spec is not self.config.topographic_commit
                or self.context.avalanching_spec is not self.config.avalanching):
            raise CaseWindowedError("config_authority", "context must retain the one copied Native config authority")
        leaves = _bed._require_host_arrays((self.config, self.state, self.context,
            self.vegetation, self.threshold_modifiers, self.static_inputs), "windowed_case")
        leaves.append(("prefix_packet", column.height_packet.packet))
        leaves.append(("prefix_encoded_base", column.height_packet.binding.encoded_base))
        if any(type(array) is not np.ndarray or array.flags.writeable for _name, array in leaves):
            raise CaseWindowedError("mutable_or_nonhost_input", "owned initial arrays must remain readonly exact NumPy arrays")
        if _held_record(self) != record["held"]:
            raise CaseWindowedError("controlling_input_changed", "owned Native input, statics or store binding changed")
        if _case._inventory(self._case_dir, self._input_limits, compiled=True) != list(self._file_pins):
            raise CaseWindowedError("source_changed", "authored/compiled case files changed after adoption")
        g, mres = self.context.geometry, self.context.mass_resolution_kg
        ids = tuple(c.class_id for c in self.context.grain_classes.classes)
        # load_version binds manifest/header and hashes referenced file identities.
        # verify_data=False defers physical values to the actual streamed
        # partition/packet validator below. No full dense reconstruction.
        load_version(column.deep_store.store_dir, column.manifest_digest, geometry=g,
            mass_resolution_kg=mres, class_ids=ids,
            expected_header_digest=column.deep_store.header.digest(), verify_data=False)
        validate_windowed_partition(column, self.state.active_layer, g, mres)
        n = _bed._native()
        n.check_sediment_availability_state(self.state.sediment_availability, self.state.active_layer, mres)
        n.validate_water_state(self.state.water, g.ny, g.nx, len(ids))
        if n.validate_ledger_state(self.state.ledger) != (g.ny, g.nx, len(ids)):
            raise CaseWindowedError("ledger_shape", "Native ledger dimensions differ from the case")
        n.validate_committed_topography_state(self.state.committed_topography)
        n.validate_vegetation_state(self.vegetation, g.ny, g.nx, geometry=g)
        n.validate_threshold_modifier_state(self.threshold_modifiers, g.ny, g.nx)
        return self


def _store_destination(store_dir, case_dir):
    if (type(store_dir) is not type(Path()) or not store_dir.is_absolute()
            or ".." in store_dir.parts or store_dir.exists() or store_dir.is_symlink()
            or not store_dir.parent.is_dir() or store_dir.parent.resolve() != store_dir.parent
            or store_dir.is_relative_to(case_dir)):
        raise CaseWindowedError("invalid_store_directory", "requires a fresh absolute Path outside the case under a real existing parent")


def adopt_windowed_case(inputs: _case.CaseInputs, *, store_dir: Path, datum_offset_m: float,
                       window_voxels: int, chunk_columns: int, rows_per_band: int,
                       with_prefix_packets: bool, limits: CaseWindowedLimits) -> CaseWindowedInputs:
    """Convert the complete actual column through Native streamed construction.

    W/chunk/band/datum/packet choices are required arguments. Datum zero is
    the Native dense frame; a different datum or disabled height packets is
    unsupported by this exact-height adoption boundary.

    Once Native construction starts, a refusal can leave partial files or a
    published genesis at store_dir. This call does not delete those artifacts
    or any referenced store. Retain the failure evidence and select a fresh
    store path for a retry; post-construction refusal is not a rollback.
    """
    if type(inputs) is not _case.CaseInputs or type(limits) is not CaseWindowedLimits:
        raise CaseWindowedError("invalid_input_type", "requires genuine CaseInputs and CaseWindowedLimits")
    limits.__post_init__()
    for name, value in (("window_voxels", window_voxels), ("chunk_columns", chunk_columns), ("rows_per_band", rows_per_band)):
        if type(value) is not int or not 0 < value < (1 << 63):
            raise CaseWindowedError("invalid_storage_choice", f"{name} must be an explicit positive plain integer")
    if type(datum_offset_m) is not float or datum_offset_m.hex() != (0.0).hex():
        raise CaseWindowedError("unsupported_datum", "requires explicit +0.0 in the Native dense physical frame")
    if type(with_prefix_packets) is not bool or with_prefix_packets is not True:
        raise CaseWindowedError("prefix_packets_required", "exact-height committed adoption requires explicit True")
    inputs.verify()
    cfg, compiled = inputs.resolved.config, inputs.compiled
    from maple.surface.topographic_commit.windowed_commit import windowed_committed_state_from_dense_baseline
    from maple.surface.voxels.windowed_inventory import require_fixed_windowed_capacity, windowed_column_from_dense
    from maple.surface.voxels.windowed_prefix import packet_terms_for
    from maple_syrup.sediment_bed import BedContext, BedState

    # Keep the Native authored-growth refusal independent of the compiled
    # column's current headroom and the adapter's shape ceilings.
    require_fixed_windowed_capacity(cfg.geometry, operation="masy.adopt_windowed_case")
    g, mass = cfg.geometry, compiled.voxel_column.mass_kg
    ny, nx, nv, nc = mass.shape
    if (ny * nx > limits.max_cells or nc > limits.max_classes or nv > limits.max_voxels
            or window_voxels > nv or rows_per_band > ny):
        raise CaseWindowedError("shape_limit", "case or storage choices exceed the declared shape domain")
    if g.dx_m != g.dy_m:
        raise CaseWindowedError("unsupported_geometry", "current common water input path requires square cells")
    if compiled.initial_mobile_state is not None:
        raise CaseWindowedError("unsupported_initial_mobile", "initial airborne seed needs separately qualified windowed transport ownership")
    if getattr(compiled, "ledger_reconciliation_baseline", None) is not None:
        raise CaseWindowedError("unsupported_reconciliation_baseline", "non-None initial baseline is outside current case admission")
    _store_destination(store_dir, inputs.case_dir)
    static = {"resolved": {field.name: getattr(inputs.resolved, field.name)
                          for field in dataclasses.fields(inputs.resolved) if field.name != "config"},
              "compiled": {field.name: getattr(compiled, field.name)
                           for field in dataclasses.fields(compiled) if field.name not in _PHYSICAL_FIELDS}}
    noncolumn = (compiled.active_layer, compiled.water, compiled.sediment_ledger,
                 compiled.committed_topography, compiled.sediment_availability,
                 compiled.vegetation, compiled.threshold_modifier_state, cfg, static)
    noncolumn_bytes, _ = _case._array_cost(noncolumn)
    _input_occurrences, input_roots = _case._array_cost((inputs.resolved, compiled))
    cells, packet_terms = ny * nx, packet_terms_for(nv, nc)
    plan = {"caller_retained_input_root_bytes": input_roots,
            "candidate_noncolumn_array_occurrence_bytes": noncolumn_bytes,
            "candidate_resident_array_bytes": cells * (8 * window_voxels * nc + 4 + 8 * nc + 16),
            "candidate_prefix_packet_bytes": 8 * cells * packet_terms,
            "candidate_prefix_encoded_base_bytes": 4 * cells,
            "native_new_committed_baseline_bytes": 8 * cells * nc,
            "native_availability_map_envelope_bytes": 8 * cells * (6 * nc + 2),
            "candidate_available_fraction_bytes": 8 * cells * nc,
            "native_band_payload_bytes": 8 * rows_per_band * nx * nv * nc,
            "native_chunk_payload_bytes": 8 * chunk_columns * nv * nc,
            "metadata_text_reservation_bytes": 3 * limits.max_metadata_bytes}
    plan["known_owner_and_payload_bytes"] = sum(plan.values())
    if plan["known_owner_and_payload_bytes"] > limits.max_known_host_bytes:
        raise CaseWindowedError("known_copy_limit", "caller originals plus candidate/band/chunk/metadata owners exceed the declared limit")
    source_record = _case._input_record(inputs.resolved, compiled, inputs.file_pins)
    provenance_bytes = _bytes(source_record, limits.max_metadata_bytes)
    fallback = _case._fallback(inputs.resolved, compiled, inputs.case_dir)
    inputs.verify()
    _case._validate_compiled(compiled, inputs.resolved, fallback)
    owned = copy.deepcopy(noncolumn)
    active, water, ledger, committed, availability, vegetation, threshold, config, statics = owned
    import numpy as np

    available_fraction = np.array(fallback, dtype=np.float64, copy=True, order="C")
    context = BedContext(config.geometry, config.grain_classes, config.numerics.mass_resolution_kg,
                         config.topographic_commit, config.avalanching, available_fraction)
    column = windowed_column_from_dense(compiled.voxel_column, g, context.mass_resolution_kg,
        store_dir=store_dir, class_ids=tuple(c.class_id for c in config.grain_classes.classes),
        max_store_bytes=limits.max_store_bytes, datum_offset_m=datum_offset_m,
        window_voxels=window_voxels, chunk_columns=chunk_columns, rows_per_band=rows_per_band,
        with_prefix_packets=with_prefix_packets)
    rebound, receipt = windowed_committed_state_from_dense_baseline(committed, column, active,
        config.geometry, require_exact_height=True, trusted_packet=False)
    state = BedState(column, active, water, ledger, rebound, availability)
    _freeze((state, context, config, vegetation, threshold, statics))
    column.height_packet.packet.flags.writeable = False
    inputs.verify()
    if _bytes(_case._input_record(inputs.resolved, compiled, inputs.file_pins), limits.max_metadata_bytes) != provenance_bytes:
        raise CaseWindowedError("input_changed", "complete Native case changed during construction")
    envelope = CaseWindowedInputs(state, context, config, vegetation, threshold, None, None,
        statics, 0.0, "", inputs.case_dir, copy.deepcopy(inputs.file_pins), inputs.input_limits,
        limits, b"", _TOKEN)
    record = {"schema": "masy.case_windowed_inputs.v1", "native_case_identity_sha256": inputs.identity_sha256,
        "case_directory": str(inputs.case_dir), "file_pins": _case._canonical(inputs.file_pins),
        "input_limits": dataclasses.asdict(inputs.input_limits),
        "complete_initial_provenance": json.loads(provenance_bytes), "held": _held_record(envelope),
        "native_committed_regrouping": receipt, "limits": dataclasses.asdict(limits), "construction_plan": plan,
        "storage_choices": {"datum_offset_m": datum_offset_m, "window_voxels": window_voxels,
            "chunk_columns": chunk_columns, "rows_per_band": rows_per_band, "with_prefix_packets": True,
            "require_exact_height": True, "trusted_packet": False},
        "ledger_reconciliation_baseline": None, "initial_mobile_state": None,
        "backend": "numpy", "event_eligibility": False, "device_placement": False,
        "caller_dense_input_may_remain_owned": True, "whole_process_fit_or_release_claim": False,
        "known_plan_excludes": "Native temporary reductions, Python/provider/allocator state; original bounded worker must admit actual construction"}
    record_bytes = _bytes(record, limits.max_metadata_bytes)
    result = dataclasses.replace(envelope, identity_sha256=hashlib.sha256(record_bytes).hexdigest(), _record_bytes=record_bytes)
    return result.verify()


@dataclasses.dataclass(frozen=True)
class PreparedCaseWindowedWater:
    """Genuine Native HOST inputs; neither a runner nor a transport admission."""
    water: DenseWaterInputs
    control: DenseRunnerControl
    profile_id: str
    identity_sha256: str
    _inputs: CaseWindowedInputs = dataclasses.field(repr=False)
    _spec: CaseWaterSpec = dataclasses.field(repr=False)
    _limits: CaseWaterCopyLimits = dataclasses.field(repr=False)
    _spec_bytes: bytes = dataclasses.field(repr=False)
    _water_bytes: bytes = dataclasses.field(repr=False)
    _control_bytes: bytes = dataclasses.field(repr=False)
    _record_bytes: bytes = dataclasses.field(repr=False)

    def identity_record(self):
        if hashlib.sha256(self._record_bytes).hexdigest() != self.identity_sha256:
            raise CaseWindowedError("identity_changed", "water input record changed")
        return json.loads(self._record_bytes)

    def verify(self, inputs=None):
        import numpy as np

        held = self._inputs if inputs is None else inputs
        if type(held) is not CaseWindowedInputs or held is not self._inputs:
            raise CaseWindowedError("foreign_owner", "water inputs belong to one adopted windowed owner")
        held.verify()
        record = self.identity_record()
        if (type(self._limits) is not CaseWaterCopyLimits
                or any(type(value) is not bytes for value in
                       (self._spec_bytes, self._water_bytes, self._control_bytes))):
            raise CaseWindowedError("controlling_input_changed", "prepared copy limits or identity byte fields changed")
        maximum = self._limits.max_metadata_bytes
        if (record["windowed_case_sha256"] != held.identity_sha256
                or record["native_inputs"]["profile_id"] != self.profile_id
                or record["native_inputs"]["copy_limits"] != dataclasses.asdict(self._limits)
                or hashlib.sha256(self._spec_bytes).hexdigest() != record["native_inputs"]["spec_sha256"]
                or hashlib.sha256(self._water_bytes).hexdigest() != record["native_inputs"]["water_sha256"]
                or hashlib.sha256(self._control_bytes).hexdigest() != record["native_inputs"]["control_sha256"]
                or _identity(np, self._spec, maximum) != self._spec_bytes
                or _identity(np, self.water, maximum) != self._water_bytes
                or _identity(np, self.control, maximum) != self._control_bytes
                or self.water.vegetation_cover_fraction is not held.vegetation.raster.cover_fraction):
            raise CaseWindowedError("controlling_input_changed", "windowed owner, Native water or authored inputs changed")
        return self


def prepare_case_windowed_water(inputs: CaseWindowedInputs, spec: CaseWaterSpec,
                               control: DenseRunnerControl, *, limits: CaseWaterCopyLimits) -> PreparedCaseWindowedWater:
    """Use the same Native factories and explicit wet law as the Dense API.

    Adapter/owner failures use CaseWindowedError; shared wet-input/control
    failures retain CaseWaterError, including the Dense control-type check.
    """
    if (type(inputs) is not CaseWindowedInputs or type(spec) is not CaseWaterSpec
            or type(limits) is not CaseWaterCopyLimits):
        raise CaseWindowedError("invalid_input_type", "requires actual windowed owner, CaseWaterSpec and CaseWaterCopyLimits")
    if type(control) is not DenseRunnerControl:
        raise CaseWaterError("invalid_control", "requires existing DenseRunnerControl")
    inputs.verify()
    state, context, vegetation, clock = inputs.state, inputs.context, inputs.vegetation, inputs.time_s
    water, spec_bytes, water_bytes, control_bytes, native_record = _prepare_water_inputs(
        state, context, vegetation, inputs.identity_sha256, spec, control, limits)
    inputs.verify()
    if (inputs.state is not state or inputs.context is not context
            or inputs.vegetation is not vegetation or inputs.time_s != clock):
        raise CaseWindowedError("input_changed", "windowed owner changed during Native preparation")
    record = {"schema": "masy.case_windowed_water_inputs.v1", "windowed_case_sha256": inputs.identity_sha256,
              "native_inputs": json.loads(native_record), "event_eligibility": False, "device_placement": False}
    raw = _bytes(record, limits.max_metadata_bytes)
    return PreparedCaseWindowedWater(water, control, spec.profile_id, hashlib.sha256(raw).hexdigest(),
        inputs, spec, limits, spec_bytes, water_bytes, control_bytes, raw).verify()


# Initial passive C64 capsules. These are input ownership, not event restart.
_INITIAL_SCHEMA = "masy.initial_windowed_case.v1"
_INITIAL_MEMBERS = ("native.npz", "committed-voxel.npy", "initial-available-fraction.npy",
                    "host-record.json", "case-input-record.json")
_INITIAL_NATIVE_PROVENANCE = "masy.initial_windowed_native.v1"
_INITIAL_SOURCES = (__name__, "masy.case_workflow", "masy.case_water", "masy.dense_bed",
    "maple.core.types.voxel", "maple.surface.voxels.windowed_snapshot",
    "maple.surface.voxels.deep_store", "maple.surface.voxels.windowed_inventory",
    "maple.surface.voxels.windowed_prefix", "maple.core.backend.prefix_cell_reductions",
    "maple_syrup.sediment_bed")


def _initial_path(value, *, directory=False, fresh=False):
    if (type(value) is not type(Path()) or not value.is_absolute() or ".." in value.parts
            or len(str(value)) > 4096 or len(value.parts) > 64):
        raise CaseWindowedError("capsule_path", "requires a bounded canonical absolute Path")
    parent = value.parent if fresh else value
    for path in (parent, *parent.parents):
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise CaseWindowedError("capsule_path", "all directory ancestors must be ordinary, never symlinks")
    if value.resolve() != value or (fresh and (value.exists() or value.is_symlink())):
        raise CaseWindowedError("capsule_path", "requires an unchanged real path and an exclusive fresh destination")
    if directory and not fresh and not stat.S_ISDIR(value.lstat().st_mode):
        raise CaseWindowedError("capsule_path", "capsule is not an ordinary directory")
    return value


def _initial_io(io, requested):
    # Actual requested read-buffer occurrences, positive even when sequential.
    charged = io["read_buffer_occurrence_bytes"] + requested
    if io["known_base_bytes"] + charged > io["known_limit_bytes"]:
        raise CaseWindowedError("known_copy_limit", "adapter read occurrences exceed the unchanged known HOST ceiling")
    io["read_buffer_occurrence_bytes"] = charged


def _initial_pin(path, ceiling, io):
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or path.resolve() != path
            or before.st_size < 1 or before.st_size > ceiling):
        raise CaseWindowedError("capsule_file", "requires a nonempty bounded ordinary file")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while True:
            requested = min(1 << 20, before.st_size - size + 1)
            _initial_io(io, requested)
            block = stream.read(requested)
            if not block:
                break
            size += len(block)
            if size > ceiling:
                raise CaseWindowedError("capsule_file", "file grew beyond its admitted extent")
            digest.update(block)
        held = os.fstat(stream.fileno())
    stamp = _case._stamp(before)
    if stamp != _case._stamp(held) or stamp != _case._stamp(path.lstat()) or size != before.st_size:
        raise CaseWindowedError("capsule_changed", "file changed while authenticating")
    io["hash_calls"] += 1
    io["hash_bytes"] += size
    return {"size": size, "sha256": digest.hexdigest(), "stamp": list(stamp)}


def _initial_same_pin(path, expected, ceiling, io):
    if (type(expected) is not dict or set(expected) not in ({"size", "sha256"}, {"size", "sha256", "stamp"})
            or type(expected["size"]) is not int or not 0 < expected["size"] <= ceiling
            or type(expected["sha256"]) is not str or re.fullmatch("[0-9a-f]{64}", expected["sha256"]) is None):
        raise CaseWindowedError("capsule_metadata", "closed positive file identity required")
    actual = _initial_pin(path, ceiling, io)
    if any(actual[key] != expected[key] for key in ("size", "sha256")):
        raise CaseWindowedError("capsule_changed", "authenticated file identity differs")
    return actual


def _initial_json(raw, maximum):
    def pairs(items):
        result = {}
        for key, value in items:
            if type(key) is not str or key in result:
                raise CaseWindowedError("capsule_metadata", "duplicate or nontext metadata key")
            result[key] = value
        return result
    if type(raw) is not bytes or not 0 < len(raw) <= maximum:
        raise CaseWindowedError("capsule_metadata", "bounded metadata bytes required")
    # All genuine writers here emit ASCII (including escaped Unicode).
    # Bound JSON depth and token/container count BEFORE constructing its DOM.
    quoted = escaped = token = False
    depth = nodes = token_length = 0
    try:
        text = raw.decode("ascii")
        for char in text:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                nodes += 1
                quoted = True
                token = False
            elif char in "[{":
                nodes += 1
                depth += 1
                token = False
            elif char in "]}":
                depth -= 1
                token = False
            elif char in " \t\n\r,:":
                token = False
            else:
                if not token:
                    nodes += 1
                    token_length = 0
                token = True
                token_length += 1
                if token_length > 64:
                    raise ValueError("overlong primitive token")
            if depth < 0 or depth > 32 or nodes > min(maximum, 131072):
                raise ValueError("JSON lexical frontier")
        if quoted or depth:
            raise ValueError("unterminated JSON")
        def integer(value):
            if len(value) > 20:
                raise ValueError("overlong integer")
            return int(value)
        value = json.loads(text, object_pairs_hook=pairs, parse_int=integer,
                           parse_constant=lambda _x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise CaseWindowedError("capsule_metadata", "invalid closed JSON") from exc
    todo, nodes = [(value, 0)], 0
    while todo:
        item, depth = todo.pop()
        nodes += 1
        if nodes > min(maximum, 131072) or depth > 32:
            raise CaseWindowedError("capsule_metadata", "metadata graph exceeds the closed domain")
        if type(item) in (dict, list):
            count = 2 * len(item) if type(item) is dict else len(item)
            if nodes + len(todo) + count > min(maximum, 131072):
                raise CaseWindowedError("capsule_metadata", "metadata expansion exceeds quota")
            children = (part for pair in item.items() for part in pair) if type(item) is dict else iter(item)
            todo.extend((part, depth + 1) for part in children)
        elif type(item) is float:
            if not math.isfinite(item):
                raise CaseWindowedError("capsule_metadata", "nonfinite metadata")
        elif type(item) is int:
            if not -(1 << 63) < item < (1 << 63):
                raise CaseWindowedError("capsule_metadata", "integer exceeds int63")
        elif item is not None and type(item) not in (str, bool):
            raise CaseWindowedError("capsule_metadata", "primitive metadata required")
    return value


def _initial_read(path, expected, maximum, io):
    before = _initial_same_pin(path, expected, maximum, io)
    with path.open("rb") as stream:
        _initial_io(io, before["size"] + 1)
        raw = stream.read(before["size"] + 1)
    io["read_bytes"] += len(raw)
    after = _initial_pin(path, maximum, io)
    if len(raw) != before["size"] or before != after:
        raise CaseWindowedError("capsule_changed", "file changed across bounded read")
    return raw


def _initial_admit(inputs, limits, maximum):
    if (type(inputs) is not _case.CaseInputs or inputs._token is not _case._INPUT_TOKEN
            or type(limits) is not CaseWindowedLimits or type(maximum) is not int
            or not 0 < maximum <= 64 << 20):
        raise CaseWindowedError("capsule_domain", "genuine inputs/limits and explicit IO bound <=64MiB required")
    limits.__post_init__()
    g, c = inputs.resolved.config.geometry, inputs.compiled
    if ((g.ny, g.nx, len(inputs.resolved.config.grain_classes.classes), c.voxel_column.mass_kg.shape)
            != (8, 8, 1, (8, 8, 20, 1)) or g.dx_m != g.dy_m
            or g.voxel_growth.enabled is not False or c.initial_mobile_state is not None
            or getattr(c, "ledger_reconciliation_baseline", None) is not None
            or c.transform_logs != () or not c.import_sources
            or any(type(row) is not dict for row in c.import_sources)
            or limits.max_cells < 64 or limits.max_classes < 1 or limits.max_voxels < 20):
        raise CaseWindowedError("capsule_domain", "requires an actual loaded imported dry C64/K1/V20 fixed-capacity subject")
    occurrences, roots = _case._array_cost((inputs.resolved, c))
    file_bytes = 0
    if type(inputs.file_pins) is not tuple or len(inputs.file_pins) > inputs.input_limits.max_source_files + 4:
        raise CaseWindowedError("capsule_domain", "closed original input inventory required")
    for row in inputs.file_pins:
        if type(row) is not dict or type(row.get("size")) is not int or not 0 < row["size"] < 1 << 63:
            raise CaseWindowedError("capsule_domain", "positive original file extents required")
        file_bytes += row["size"]
    # Positive occurrences, not allocator fees or a release/deduplication claim.
    # Eight Native hash buffers cover the fixed reopen/water/projector chain
    # plus Native snapshot load; every extra caller verification is additional.
    plan = {"caller_array_occurrence_bytes": occurrences, "caller_retained_root_bytes": roots,
        "candidate_copy_and_error_array_occurrences": 3 * occurrences,
        "input_inventory_read_occurrences": 8 * file_bytes,
        "canonical_HOST_input_current_next_error_bytes": 4 * limits.max_metadata_bytes,
        "Native_metadata_UTF32_UTF8_and_error_bytes": 3 * limits.max_metadata_bytes,
        "JSON_raw_ASCII_current_next_error_bytes": 3 * limits.max_metadata_bytes,
        "JSON_node_reference_word_occurrences": 3 * 8 * min(limits.max_metadata_bytes, 131072),
        "alias_position_reference_word_occurrences": 4 * 8 * min(limits.max_metadata_bytes, 131072),
        "Native_hash_buffer_occurrences": 8 * (8 << 20),
        "Source_read_buffer_occurrences": len(_INITIAL_SOURCES) * (1 << 20),
        "Native_snapshot_array_copy_occurrences": 3 * 9984,
        "companion_and_discarded_baseline_occurrences": 4 * 1024 + 512,
        "Native_chunk_prefix_validation_occurrences": 8 * (10240 + 4096 + 256),
        "NPY_ZIP_header_reservation": 14 * 4096 + 12 * 1024 + 98}
    plan["known_owner_and_frontier_bytes"] = sum(plan.values())
    if plan["known_owner_and_frontier_bytes"] > limits.max_known_host_bytes:
        raise CaseWindowedError("known_copy_limit", "positive caller/copy/metadata/read frontier exceeds the unchanged HOST limit")
    # Archive IO is a separate finite ceiling; it is never counted as a
    # whole in-memory NPZ copy. Numeric members have fixed C64 extents.
    plan["capsule_IO_eligibility_bytes"] = maximum
    from maple.aeolian.cohorts.capacity import default_available_memory_bytes
    available = default_available_memory_bytes()
    if type(available) is not int or 2 * plan["known_owner_and_frontier_bytes"] > available:
        raise CaseWindowedError("capsule_headroom", "genuine HOST half-headroom refused")
    return plan


def _initial_sources(io):
    # These real imports own the physical loader and E1 binding. No device import.
    from maple.surface.voxels import windowed_snapshot  # noqa: F401
    from maple.surface.voxels import windowed_prefix
    windowed_prefix._e1()
    rows = []
    for name in _INITIAL_SOURCES:
        module = importlib.import_module(name)
        if type(module) is not types.ModuleType or type(vars(module).get("__file__")) is not str:
            raise CaseWindowedError("capsule_source", "actual ordinary Source module required")
        path = Path(vars(module)["__file__"])
        _initial_path(path.parent, directory=True)
        extent = path.lstat().st_size
        if not 0 < extent <= 1 << 20:
            raise CaseWindowedError("capsule_source", "Source file exceeds its admitted frontier")
        pin = _initial_pin(path, extent, io)
        rows.append({"module": name, "path": str(path), "size": pin["size"], "sha256": pin["sha256"]})
    return rows


def _initial_deep(record, limits, io, expected=None):
    col = record["held"]["column"]
    store = _initial_path(Path(col["store_dir"]), directory=True)
    _initial_path(store / "versions", directory=True)
    _initial_path(store / "chunks", directory=True)
    digest = col["manifest_digest"]
    if type(digest) is not str or re.fullmatch("[0-9a-f]{64}", digest) is None:
        raise CaseWindowedError("capsule_deep", "canonical original manifest digest required")
    paths = (store / "store.json", store / "versions" / (digest + ".json"))
    pins = [_initial_pin(path, limits.max_metadata_bytes, io) for path in paths]
    descriptor, manifest = [_initial_json(_initial_read(path, pin, limits.max_metadata_bytes, io),
        limits.max_metadata_bytes) for path, pin in zip(paths, pins, strict=True)]
    if (set(descriptor) != {"format", "kind", "header", "header_digest", "max_store_bytes"}
            or set(manifest) != {"format", "kind", "header", "header_digest", "version_id",
                "parent_version_id", "parent_manifest_digest", "journal_digest", "chunks"}
            or descriptor["format"] != "maple-deep-store/1" or descriptor["kind"] != "store"
            or manifest["format"] != "maple-deep-store/1" or manifest["kind"] != "version"
            or type(manifest["version_id"]) is not int or manifest["version_id"] != 0
            or manifest["parent_version_id"] is not None or manifest["parent_manifest_digest"] is not None
            or pins[1]["sha256"] != digest or col["generation"] != 0
            or type(descriptor["max_store_bytes"]) is not int
            or not 0 < descriptor["max_store_bytes"] <= limits.max_store_bytes
            or _bytes(descriptor["header"], limits.max_metadata_bytes) != _bytes(col["header"], limits.max_metadata_bytes)
            or _bytes(manifest["header"], limits.max_metadata_bytes) != _bytes(col["header"], limits.max_metadata_bytes)
            or descriptor["header_digest"] != col["header_digest"] or manifest["header_digest"] != col["header_digest"]):
        raise CaseWindowedError("capsule_deep", "original full-v1 genesis descriptor/manifest mismatch")
    header = manifest["header"]
    if (tuple(header.get(name) for name in ("ny", "nx", "n_voxels", "n_classes", "window_voxels")) != (8, 8, 20, 1, 3)
            or any(type(header.get(name)) is not int for name in ("ny", "nx", "n_voxels", "n_classes", "window_voxels", "chunk_columns"))
            or not 64 <= header["chunk_columns"] <= 1024 or header["dx_m"] != header["dy_m"]
            or type(header["datum_offset_m"]) is not float or header["datum_offset_m"].hex() != (0.0).hex()
            or header["dtype"] != "<f8" or header["layout"] != "C(cell_flat, voxel, class)"
            or type(manifest["chunks"]) is not list or len(manifest["chunks"]) != 1):
        raise CaseWindowedError("capsule_deep", "bounded C64 single-chunk physical header required")
    ref = manifest["chunks"][0]
    if (set(ref) != {"index", "cell_start", "n_cells", "sha256", "nbytes"}
            or any(type(ref.get(name)) is not int for name in ("index", "cell_start", "n_cells", "nbytes"))
            or tuple(ref[name] for name in ("index", "cell_start", "n_cells", "nbytes")) != (0, 0, 64, 10240)
            or type(ref["sha256"]) is not str or re.fullmatch("[0-9a-f]{64}", ref["sha256"]) is None):
        raise CaseWindowedError("capsule_deep", "one actual complete C64 chunk reference required")
    chunk = _initial_same_pin(store / "chunks" / (ref["sha256"] + ".f64"),
        {"size": 10240, "sha256": ref["sha256"]}, 10240, io)
    # Count every retained version/orphan; never compare the grown tree to an
    # old smaller inventory or delete siblings. No concurrent writer is allowed.
    pending, total, entries = [(store, 0)], 0, 0
    while pending:
        directory, depth = pending.pop()
        if depth > 16:
            raise CaseWindowedError("capsule_deep", "retained store depth exceeds the original domain")
        with os.scandir(directory) as iterator:
            for entry in iterator:
                entries += 1
                if entries > 4096 or len(entry.path) > 4096:
                    raise CaseWindowedError("capsule_deep", "retained store metadata domain exceeded")
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append((Path(entry.path), depth + 1))
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
                else:
                    raise CaseWindowedError("capsule_deep", "store symlink or special entry refused")
                if total > min(limits.max_store_bytes, descriptor["max_store_bytes"]):
                    raise CaseWindowedError("capsule_deep", "all retained store files exceed the unchanged quota")
    result = {"store_dir": str(store), "descriptor": {k: pins[0][k] for k in ("size", "sha256")},
        "manifest": {k: pins[1][k] for k in ("size", "sha256")},
        "chunk": {k: chunk[k] for k in ("size", "sha256")}, "mapping_cache_chunks": 8}
    if expected is not None and result != expected:
        raise CaseWindowedError("capsule_deep", "saved original deep reference changed")
    return result


def _initial_aliases(value, maximum):
    import numpy as np
    roots = (value.state.active_layer, value.state.water, value.state.ledger,
        value.state.committed_topography, value.state.sediment_availability, value.context,
        value.config, value.vegetation, value.threshold_modifiers, value.static_inputs)
    first, active, rows = {}, set(), []
    nodes = characters = 0
    def visit(item, path, depth):
        nonlocal nodes, characters
        nodes += 1
        characters += len(path)
        if characters > maximum:
            raise CaseWindowedError("capsule_alias", "aggregate alias text exceeds quota")
        if depth > 32 or nodes > min(maximum, 131072) or len(path) > 4096:
            raise CaseWindowedError("capsule_alias", "alias positions exceed their metadata domain")
        kind = type(item)
        if kind is np.ndarray or kind in (dict, list, tuple) or dataclasses.is_dataclass(item) and not isinstance(item, type):
            ident = id(item)
            if ident in active:
                raise CaseWindowedError("capsule_alias", "cyclic input forest is unsupported")
            if ident in first:
                rows.append([path, first[ident], f"{kind.__module__}.{kind.__qualname__}"])
                return
            first[ident] = path
            if kind is np.ndarray:
                return
            active.add(ident)
            if kind is dict:
                if any(type(key) not in (str, int, bool, float) and key is not None for key in item):
                    raise CaseWindowedError("capsule_alias", "plain canonical static keys required")
                count = len(item)
                if count > min(maximum, 131072) - nodes:
                    raise CaseWindowedError("capsule_alias", "alias expansion exceeds quota")
                # Bound all key images BEFORE retaining the sort keys.
                key_text = sum(len(key) if type(key) is str else 64 for key in item)
                if characters + key_text > maximum:
                    raise CaseWindowedError("capsule_alias", "canonical alias keys exceed quota")
                keys = sorted(item, key=lambda key: _bytes(_case._canonical(key), maximum))
                children = ((_bytes(_case._canonical(key), maximum).decode("ascii"), item[key]) for key in keys)
            elif kind in (list, tuple):
                count = len(item)
                children = ((str(i), child) for i, child in enumerate(item))
            else:
                fields = dataclasses.fields(item)
                count = len(fields)
                children = ((field.name, getattr(item, field.name)) for field in fields)
            if count > min(maximum, 131072) - nodes:
                raise CaseWindowedError("capsule_alias", "alias expansion exceeds quota")
            for segment, child in children:
                if len(path) + 1 + len(segment) > 4096 or characters + len(path) + 1 + len(segment) > maximum:
                    raise CaseWindowedError("capsule_alias", "alias path expansion exceeds quota")
                visit(child, path + "/" + segment, depth + 1)
            active.remove(ident)
    for index, root in enumerate(roots):
        visit(root, str(index), 0)
    _bytes(rows, maximum)
    return rows


def _initial_domain(windowed, record):
    import numpy as np
    column = windowed.state.voxel_column
    if (bool(np.any(windowed.state.water.depth_m != 0.0))
            or bool(np.any(windowed.state.water.mobile_mass_by_cell_class_kg != 0.0))
            or windowed.time_s.hex() != (0.0).hex() or windowed.initial_mobile_state is not None
            or windowed.ledger_reconciliation_baseline is not None
            or column.generation != 0 or column.manifest_digest != record["held"]["column"]["manifest_digest"]
            or column.pending_deep is not None or column.window_cache is not None
            or column.height_packet is None or column.deep_store.mapping_cache_chunks != 8
            or column.deep_store.format_version != 1 or column.deep_store.encoding != "full"
            or column.deep_store.overlay_max_bytes is not None
            or record["storage_choices"]["datum_offset_m"].hex() != (0.0).hex()
            or record["storage_choices"]["window_voxels"] != 3
            or record["storage_choices"]["with_prefix_packets"] is not True):
        raise CaseWindowedError("capsule_domain", "published initial C64/gen0/cache8/packet owner required")


def _initial_verify(windowed, record, limits, io, deep):
    _initial_deep(record, limits, io, deep)
    _initial_domain(windowed, record)
    windowed.verify()
    _initial_deep(record, limits, io, deep)


def _initial_npy(stream, extent, shape, dtype, maximum):
    magic = stream.read(8)
    if magic[:6] != b"\x93NUMPY" or magic[6:] not in (b"\x01\x00", b"\x02\x00"):
        raise CaseWindowedError("capsule_header", "supported ordinary NPY header required")
    width = 2 if magic[6] == 1 else 4
    size_bytes = stream.read(width)
    length = int.from_bytes(size_bytes, "little")
    if len(size_bytes) != width or not 0 < length <= 4096:
        raise CaseWindowedError("capsule_header", "bounded NPY header required")
    text = stream.read(length)
    try:
        header = ast.literal_eval(text.decode("ascii"))
    except (ValueError, SyntaxError, UnicodeError, RecursionError) as exc:
        raise CaseWindowedError("capsule_header", "invalid NPY header") from exc
    if type(header) is not dict or set(header) != {"descr", "fortran_order", "shape"} or header["fortran_order"] is not False:
        raise CaseWindowedError("capsule_header", "exact nonobject C-order NPY header required")
    if header["shape"] != shape or type(header["shape"]) is not tuple or any(type(n) is not int for n in header["shape"]):
        raise CaseWindowedError("capsule_header", "actual physical shape differs")
    if dtype is None:
        match = re.fullmatch(r"<U([1-9][0-9]*)", header["descr"]) if type(header["descr"]) is str else None
        if match is None:
            raise CaseWindowedError("capsule_header", "scalar Unicode Native metadata required")
        payload = 4 * int(match.group(1))
    else:
        if header["descr"] != dtype:
            raise CaseWindowedError("capsule_header", "actual physical dtype differs")
        payload = math.prod(shape) * (4 if dtype == "<i4" else 8)
    if payload > maximum or 8 + width + length + payload != extent:
        raise CaseWindowedError("capsule_header", "NPY payload extent exceeds the admitted shape/header")
    return payload


def _initial_native_metadata(record):
    return {"schema": "maple-windowed-snapshot/1", "store_dir": record["held"]["column"]["store_dir"],
        "manifest_digest": record["held"]["column"]["manifest_digest"], "generation": 0,
        "header_digest": record["held"]["column"]["header_digest"], "n_voxels": 20, "time_s": 0.0,
        "step": 0, "class_ids": record["held"]["column"]["header"]["class_ids"],
        "provenance": {"schema": _INITIAL_NATIVE_PROVENANCE,
            "native_case_identity_sha256": record["native_case_identity_sha256"],
            "windowed_identity_sha256": hashlib.sha256(_bytes(record, 64 << 20)).hexdigest()}}


def _initial_archive(path, member, record, maximum, metadata_maximum, io):
    before = _initial_same_pin(path, member, maximum, io)
    shapes = {"window_kg": ((8, 8, 3, 1), "<f8"), "base": ((8, 8), "<i4"),
        "deep_by_class_kg": ((8, 8, 1), "<f8"), "deep_total_kg": ((8, 8), "<f8"),
        "deep_room_kg": ((8, 8), "<f8"), "prefix_packet": ((8, 8, 8), "<f8"),
        "active_mass_kg": ((8, 8, 1), "<f8"), "available_mass_kg": ((8, 8, 1), "<f8"),
        "bound_mass_kg": ((8, 8, 1), "<f8"), "water_depth_m": ((8, 8), "<f8"),
        "mobile_mass_kg": ((8, 8, 1), "<f8"), "metadata": ((), None)}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if (len(infos) != 12 or {i.filename for i in infos} != {name + ".npy" for name in shapes}
                or any(i.compress_type != zipfile.ZIP_STORED or i.flag_bits & 1 for i in infos)
                or sum(i.file_size for i in infos) > maximum):
            raise CaseWindowedError("capsule_header", "exact twelve uncompressed Native members required")
        metadata = None
        for info in infos:
            shape, dtype = shapes[info.filename[:-4]]
            with archive.open(info) as stream:
                payload = _initial_npy(stream, info.file_size, shape, dtype,
                    metadata_maximum if dtype is None else maximum)
                if dtype is None:
                    _initial_io(io, payload + 1)
                    raw = stream.read(payload + 1)
                    if len(raw) != payload:
                        raise CaseWindowedError("capsule_header", "Native metadata extent differs")
                    metadata = _initial_json(raw.decode("utf-32-le").encode("utf-8"), metadata_maximum)
    expected = _initial_native_metadata(record)
    if _bytes(metadata, maximum) != _bytes(expected, maximum) or before != _initial_pin(path, maximum, io):
        raise CaseWindowedError("capsule_changed", "Native metadata or archive currentness differs")
    return before


def _initial_rebuild(inputs, directory, capsule, record_bytes, input_bytes, limits, maximum, io):
    import numpy as np
    from maple.surface.voxels.windowed_snapshot import load_windowed_state_snapshot
    from maple_syrup.sediment_bed import BedContext, BedState
    record = _initial_json(record_bytes, limits.max_metadata_bytes)
    if (record["native_case_identity_sha256"] != inputs.identity_sha256
            or record["case_directory"] != str(inputs.case_dir)
            or record["limits"] != dataclasses.asdict(limits)
            or _bytes(_case._input_record(inputs.resolved, inputs.compiled, inputs.file_pins), limits.max_metadata_bytes) != input_bytes):
        raise CaseWindowedError("capsule_input", "same original loaded case and limits required")
    _initial_deep(record, limits, io, capsule["deep"])
    before = _initial_archive(directory / "native.npz", capsule["members"]["native.npz"], record, maximum, limits.max_metadata_bytes, io)
    g, classes = inputs.resolved.config.geometry, inputs.resolved.config.grain_classes
    loaded = load_windowed_state_snapshot(directory / "native.npz", geometry=g, grain_classes=classes,
        mass_resolution_kg=inputs.resolved.config.numerics.mass_resolution_kg, max_bytes=maximum)
    if before != _initial_pin(directory / "native.npz", maximum, io):
        raise CaseWindowedError("capsule_changed", "Native archive changed across the genuine loader")
    companions = []
    for name in ("committed-voxel.npy", "initial-available-fraction.npy"):
        path = directory / name
        pin = _initial_same_pin(path, capsule["members"][name], maximum, io)
        with path.open("rb") as stream:
            _initial_npy(stream, pin["size"], (8, 8, 1), "<f8", maximum)
        array = np.load(path, allow_pickle=False)
        if pin != _initial_pin(path, maximum, io):
            raise CaseWindowedError("capsule_changed", "companion changed while loading")
        array.flags.writeable = False
        companions.append(array)
    c = inputs.compiled
    static = {"resolved": {f.name: getattr(inputs.resolved, f.name) for f in dataclasses.fields(inputs.resolved) if f.name != "config"},
        "compiled": {f.name: getattr(c, f.name) for f in dataclasses.fields(c) if f.name not in _PHYSICAL_FIELDS}}
    memo = {}
    for original, restored in ((c.active_layer, loaded["active_layer"]), (c.water, loaded["water"]),
                               (c.sediment_availability, loaded["sediment_availability"])):
        for source, target in ((original, restored), *((getattr(original, f.name), getattr(restored, f.name))
                                                      for f in dataclasses.fields(original))):
            if id(source) in memo and memo[id(source)] is not target:
                raise CaseWindowedError("capsule_alias", "original physical alias has inconsistent restored targets")
            memo[id(source)] = target
    noncolumn = (c.active_layer, c.water, c.sediment_ledger, c.committed_topography,
        c.sediment_availability, c.vegetation, c.threshold_modifier_state, inputs.resolved.config, static)
    active, water, ledger, committed, availability, vegetation, threshold, config, statics = copy.deepcopy(noncolumn, memo)
    committed = dataclasses.replace(committed, committed_voxel_mass_by_class_kg=companions[0])
    context = BedContext(config.geometry, config.grain_classes, config.numerics.mass_resolution_kg,
        config.topographic_commit, config.avalanching, companions[1])
    state = BedState(loaded["voxel_column"], active, water, ledger, committed, availability)
    _freeze((state, context, config, vegetation, threshold, statics))
    state.voxel_column.height_packet.packet.flags.writeable = False
    state.voxel_column.height_packet.binding.encoded_base.flags.writeable = False
    candidate = CaseWindowedInputs(state, context, config, vegetation, threshold, None, None, statics,
        loaded["time_s"], hashlib.sha256(record_bytes).hexdigest(), inputs.case_dir,
        copy.deepcopy(inputs.file_pins), inputs.input_limits, limits, record_bytes, None)
    if (_bytes(_held_record(candidate), limits.max_metadata_bytes) != _bytes(record["held"], limits.max_metadata_bytes)
            or _initial_aliases(candidate, limits.max_metadata_bytes) != capsule["alias_positions"]):
        raise CaseWindowedError("capsule_identity", "actual held values or original object-alias positions differ")
    _initial_domain(candidate, record)
    _initial_deep(record, limits, io, capsule["deep"])
    inputs.verify()
    if _initial_sources(io) != capsule["source_refs"]:
        raise CaseWindowedError("capsule_source", "owning Source changed before owner mint")
    result = dataclasses.replace(candidate, _token=_TOKEN)
    _initial_verify(result, record, limits, io, capsule["deep"])
    return result


def _initial_write(path, value, *, array=False):
    import numpy as np
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(fd, "wb") as stream:
        if array:
            np.save(stream, value, allow_pickle=False)
        else:
            stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _initial_sync(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _initial_receipt(operation, plan, io, ref):
    return {"schema": "masy.initial_windowed_case_receipt.v1", "operation": operation,
        "capsule": ref, "current_known_plan": dict(plan,
            adapter_read_buffer_occurrence_bytes=io["read_buffer_occurrence_bytes"],
            current_known_total_bytes=plan["known_owner_and_frontier_bytes"] + io["read_buffer_occurrence_bytes"]),
        "actual_adapter_io": dict(io),
        "carried_historical_fields": ["native_committed_regrouping", "construction_plan", "storage_choices.rows_per_band"],
        "runtime_cache_state_equality": False, "event_eligibility": False,
        "whole_process_fit_claim": False, "opaque_allocator_provider_IO_service_bytes": None,
        "original_caller_Dense_inputs_remain_positive": True}


def save_initial_windowed_case(inputs: _case.CaseInputs, windowed: CaseWindowedInputs, *,
                              capsule_dir: Path, max_capsule_bytes: int) -> dict:
    """Persist a genuine passive loaded C64/gen0 owner, publishing CAPSULE last.

    Referenced immutable deep files are retained in their original store. Any
    failure leaves truthful partial files; this call deletes or resets nothing.
    Root owns the serial-role/process/whole-run policy, not this capsule.
    """
    if type(windowed) is not CaseWindowedInputs or windowed._token is not _TOKEN:
        raise CaseWindowedError("invalid_owner", "genuine initial windowed owner required")
    limits = windowed._limits
    plan = _initial_admit(inputs, limits, max_capsule_bytes)
    io = {"hash_calls": 0, "hash_bytes": 0, "read_bytes": 0,
        "read_buffer_occurrence_bytes": 0, "known_base_bytes": plan["known_owner_and_frontier_bytes"],
        "known_limit_bytes": limits.max_known_host_bytes}
    directory = _initial_path(capsule_dir, directory=True, fresh=True)
    sources = _initial_sources(io)
    inputs.verify()
    record_bytes = windowed._record_bytes
    record = _initial_json(record_bytes, limits.max_metadata_bytes)
    if (record["native_case_identity_sha256"] != inputs.identity_sha256
            or windowed._case_dir != inputs.case_dir or windowed._input_limits != inputs.input_limits):
        raise CaseWindowedError("capsule_input", "original loaded input subject differs")
    deep = _initial_deep(record, limits, io)
    _initial_verify(windowed, record, limits, io, deep)
    input_bytes = _bytes(_case._input_record(inputs.resolved, inputs.compiled, inputs.file_pins), limits.max_metadata_bytes)
    aliases = _initial_aliases(windowed, limits.max_metadata_bytes)
    if directory.is_relative_to(inputs.case_dir) or directory.is_relative_to(Path(deep["store_dir"])):
        raise CaseWindowedError("capsule_path", "capsule must be outside original case and deep store")
    native_text = json.dumps(_initial_native_metadata(record), sort_keys=True, allow_nan=False)
    native_upper = 9984 + 4 * len(native_text) + 12 * (4096 + 1024) + 98
    upper = {"native.npz": native_upper, "committed-voxel.npy": 512 + 4096,
        "initial-available-fraction.npy": 512 + 4096,
        "host-record.json": len(record_bytes), "case-input-record.json": len(input_bytes)}
    prospective = {"schema": _INITIAL_SCHEMA, "limits": dataclasses.asdict(limits),
        "members": {name: {"size": size, "sha256": "0" * 64} for name, size in upper.items()},
        "windowed_identity_sha256": windowed.identity_sha256, "native_case_identity_sha256": inputs.identity_sha256,
        "deep": deep, "source_refs": sources, "alias_positions": aliases}
    if sum(upper.values()) + len(_bytes(prospective, limits.max_metadata_bytes)) > max_capsule_bytes:
        raise CaseWindowedError("capsule_file_limit", "prospective actual-domain capsule exceeds the declared IO limit")
    from maple.surface.voxels.windowed_snapshot import save_windowed_state_snapshot
    directory.mkdir()
    state = windowed.state
    save_windowed_state_snapshot(state.voxel_column, state.active_layer, windowed.time_s, directory / "native.npz",
        step=0, geometry=windowed.context.geometry, mass_resolution_kg=windowed.context.mass_resolution_kg,
        sediment_availability=state.sediment_availability, water=state.water,
        provenance={"schema": _INITIAL_NATIVE_PROVENANCE, "native_case_identity_sha256": inputs.identity_sha256,
                    "windowed_identity_sha256": windowed.identity_sha256})
    os.chmod(directory / "native.npz", 0o444)
    _initial_write(directory / "committed-voxel.npy", state.committed_topography.committed_voxel_mass_by_class_kg, array=True)
    _initial_write(directory / "initial-available-fraction.npy", windowed.context.initial_available_fraction, array=True)
    _initial_write(directory / "host-record.json", record_bytes)
    _initial_write(directory / "case-input-record.json", input_bytes)
    members = {}
    total = 0
    for name in _INITIAL_MEMBERS:
        pin = _initial_pin(directory / name, max_capsule_bytes, io)
        members[name] = {key: pin[key] for key in ("size", "sha256")}
        total += pin["size"]
    capsule = {"schema": _INITIAL_SCHEMA, "limits": dataclasses.asdict(limits), "members": members,
        "windowed_identity_sha256": windowed.identity_sha256, "native_case_identity_sha256": inputs.identity_sha256,
        "deep": deep, "source_refs": sources, "alias_positions": aliases}
    raw = _bytes(capsule, limits.max_metadata_bytes)
    if total + len(raw) > max_capsule_bytes:
        raise CaseWindowedError("capsule_file_limit", "six actual capsule files exceed their declared IO limit")
    _initial_rebuild(inputs, directory, capsule, record_bytes, input_bytes, limits, max_capsule_bytes, io)
    _initial_verify(windowed, record, limits, io, deep)
    inputs.verify()
    if (_initial_sources(io) != sources or windowed._record_bytes != record_bytes
            or _initial_aliases(windowed, limits.max_metadata_bytes) != aliases):
        raise CaseWindowedError("capsule_changed", "source or owner changed during save/readback")
    for name, pin in members.items():
        _initial_same_pin(directory / name, pin, max_capsule_bytes, io)
    _initial_sync(directory)
    _initial_write(directory / "CAPSULE.json", raw)
    _initial_sync(directory)
    ref = _initial_pin(directory / "CAPSULE.json", limits.max_metadata_bytes, io)
    return _initial_receipt("save_initial", plan, io, {"path": str(directory / "CAPSULE.json"),
        "size": ref["size"], "sha256": ref["sha256"]})


def reopen_initial_windowed_case(inputs: _case.CaseInputs, *, capsule_dir: Path,
                                expected_capsule_sha256: str, limits: CaseWindowedLimits,
                                max_capsule_bytes: int) -> tuple[CaseWindowedInputs, dict]:
    """Reopen the SAME initial physical snapshot with already-owned inputs.

    No compile/load/adopt/regroup/DEM reconstruction is called. Later genuine
    sibling versions may remain; serial-role quiescence and actual Root limits
    are required. The old construction receipt is carried, never recomputed.
    """
    plan = _initial_admit(inputs, limits, max_capsule_bytes)
    if type(expected_capsule_sha256) is not str or re.fullmatch("[0-9a-f]{64}", expected_capsule_sha256) is None:
        raise CaseWindowedError("capsule_identity", "actual externally authenticated capsule SHA required")
    io = {"hash_calls": 0, "hash_bytes": 0, "read_bytes": 0,
        "read_buffer_occurrence_bytes": 0, "known_base_bytes": plan["known_owner_and_frontier_bytes"],
        "known_limit_bytes": limits.max_known_host_bytes}
    directory = _initial_path(capsule_dir, directory=True)
    with os.scandir(directory) as iterator:
        names = []
        for entry in iterator:
            if len(names) >= 6 or not stat.S_ISREG(entry.stat(follow_symlinks=False).st_mode):
                raise CaseWindowedError("capsule_members", "exact six ordinary capsule files required")
            names.append(entry.name)
    if set(names) != {*_INITIAL_MEMBERS, "CAPSULE.json"}:
        raise CaseWindowedError("capsule_members", "exact six capsule members required")
    final = _initial_pin(directory / "CAPSULE.json", limits.max_metadata_bytes, io)
    if final["sha256"] != expected_capsule_sha256:
        raise CaseWindowedError("capsule_identity", "capsule differs from original external reference")
    raw = _initial_read(directory / "CAPSULE.json", final, limits.max_metadata_bytes, io)
    capsule = _initial_json(raw, limits.max_metadata_bytes)
    if (type(capsule) is not dict or set(capsule) != {"schema", "limits", "members", "windowed_identity_sha256",
            "native_case_identity_sha256", "deep", "source_refs", "alias_positions"}
            or capsule["schema"] != _INITIAL_SCHEMA
            or _bytes(capsule["limits"], limits.max_metadata_bytes) != _bytes(dataclasses.asdict(limits), limits.max_metadata_bytes)
            or set(capsule["members"]) != set(_INITIAL_MEMBERS)
            or capsule["native_case_identity_sha256"] != inputs.identity_sha256):
        raise CaseWindowedError("capsule_metadata", "closed original capsule/input/limits required")
    total = final["size"]
    for name in _INITIAL_MEMBERS:
        total += _initial_same_pin(directory / name, capsule["members"][name], max_capsule_bytes, io)["size"]
    if total > max_capsule_bytes:
        raise CaseWindowedError("capsule_file_limit", "entire capsule exceeds its unchanged IO limit")
    if _initial_sources(io) != capsule["source_refs"]:
        raise CaseWindowedError("capsule_source", "original owning Source differs")
    inputs.verify()
    host_raw = _initial_read(directory / "host-record.json", capsule["members"]["host-record.json"], limits.max_metadata_bytes, io)
    input_raw = _initial_read(directory / "case-input-record.json", capsule["members"]["case-input-record.json"], limits.max_metadata_bytes, io)
    if hashlib.sha256(host_raw).hexdigest() != capsule["windowed_identity_sha256"]:
        raise CaseWindowedError("capsule_identity", "original verbatim HOST record differs")
    result = _initial_rebuild(inputs, directory, capsule, host_raw, input_raw, limits, max_capsule_bytes, io)
    inputs.verify()
    if _initial_sources(io) != capsule["source_refs"] or final != _initial_pin(directory / "CAPSULE.json", limits.max_metadata_bytes, io):
        raise CaseWindowedError("capsule_changed", "source or final marker changed before return")
    for name in _INITIAL_MEMBERS:
        _initial_same_pin(directory / name, capsule["members"][name], max_capsule_bytes, io)
    return result, _initial_receipt("reopen_initial", plan, io,
        {"path": str(directory / "CAPSULE.json"), "size": final["size"], "sha256": final["sha256"]})
