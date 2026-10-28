"""Lossless HOST adoption of a Native compiled case into windowed storage.

This input boundary performs no event, settlement, hydraulic commit, device
placement or restart publication. Native compilation remains dense. The caller
can still own that dense input after this function returns.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
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
           "PreparedCaseWindowedWater", "adopt_windowed_case", "prepare_case_windowed_water"]

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
