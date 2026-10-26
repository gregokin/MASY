"""Closed MASY holder of ONE genuine Native windowed bed (seven shared fields).

`WindowedBed` holds the CURRENT actual SYRUP `WindowedSharedBedState`:
voxel_column (resident window, maintained prefix packets, pending journal and
the immutable full-v1 deep-store version), active_layer, water, ledger,
committed_topography, sediment_availability and reconciliation_baseline. It
also holds the actual physical time, the canonical PRESCRIBED vegetation and
threshold modifiers, the trajectory semantics of the empty wind content and the
exact `WindowedAvalancheContext`. It never rebuilds a bed from elevation,
fractions or a case and never materializes or copies the deep store.

Construction is `build_windowed_bed` only, from a caller-built genuine Native
`SchedulerPhysicalState` (all twelve fields) and context. The profile is the
Small CPU one: NumPy float64, square 4x4 grid, V6/W3/K1, published full-v1
store, tracked availability, real dry constant-depth WaterState, empty initial
wind pool. Anything else refuses before mutation.

Staging (`stage_copy`) prices every copy BEFORE allocating, then makes exact
typed copies of the active/water/ledger/committed/availability/baseline holders
and an independent resident column through the admitted Native copying
exchange (`open_windowed_exchange`, zero operations, genuine `finish`). That
finish keeps the immutable deep version, copies the resident window, metadata
and maintained packets, and keeps any prior pending journal through its own
`working_bed_journal`. The stage is then checked for distinct resident
ownership, an unchanged deep authority, identity equal to the accepted bed and
the actual Native partition/value validation, all before any event uses it.

Adoption (`adopt_*`) validates the complete candidate first (exact seven-field
type, prescribed references, semantics, empty wind pool, actual Native shared
validation, clock and commit stamp) and then replaces the held state once.
Importing this module imports neither numpy nor a native model. See
docs/windowed_cpu_integration.md.
"""

from __future__ import annotations

import dataclasses
import enum
from types import SimpleNamespace
from typing import Any

from masy import dense_bed as _dense
from masy.clock import _native_clock, encode_time
from masy.dense_bed import SHARED_FIELDS, DenseBedError

__all__ = [
    "PHYSICAL_FIELDS",
    "SCHEDULER_FIELDS",
    "WINDOWED_BED_IDENTITY_SCHEMA",
    "WINDOWED_CASE_SCHEMA",
    "WINDOWED_FIELDS",
    "WindowedBed",
    "WindowedBedError",
    "WindowedBedLimits",
    "WindowedStage",
    "build_windowed_bed",
]

WINDOWED_CASE_SCHEMA = "masy.windowed_case.v1"
WINDOWED_BED_IDENTITY_SCHEMA = "masy.windowed_bed_identity.v1"
# Exactly SYRUP's `_BED_FIELDS` order: the six Dense shared fields plus the seventh.
WINDOWED_FIELDS = SHARED_FIELDS + ("reconciliation_baseline",)
# The twelve annotated Native `SchedulerPhysicalState` fields and the eight `SchedulerState` fields.
PHYSICAL_FIELDS = ("voxel_column", "active_layer", "mobile_state", "ledger", "committed_topography",
                   "threshold_modifier_state", "time_s", "vegetation", "sediment_availability", "water",
                   "coupled_trajectory_semantics", "ledger_reconciliation_baseline")
SCHEDULER_FIELDS = ("physical", "coupled", "block_index", "block_offset_s", "last_resolved_wind",
                    "topographic_wind_cache", "vegetation_wind_cache", "active_layer_composition_cache")
CONTEXT_FIELDS = ("geometry", "grain_classes", "mass_resolution_kg", "topographic_commit_spec", "avalanching_spec",
                  "initial_available_fraction", "depth_update_rule", "record_exchange_history",
                  "window_max_journal_records", "deep_chunk_encoding", "deep_overlay_max_bytes",
                  "native_iteration_limits", "native_evidence_limits", "native_solve_limits",
                  "native_validation_limits", "native_publication_limits", "native_qualification_limits",
                  "source_identity", "bridge_limits")
NATIVE_POLICIES = ("native_iteration_limits", "native_evidence_limits", "native_solve_limits",
                   "native_validation_limits", "native_publication_limits", "native_qualification_limits")
# Exact Small positive-water profile (C16/V6/W3/K1, journal 320); the SYRUP lane re-checks it natively.
PROFILE_SHAPE = {"ny": 4, "nx": 4, "n_voxels": 6, "window_voxels": 3, "n_classes": 1}
JOURNAL_RECORDS = 320
INITIAL_AVAILABILITY_POLICY = "explicit_fraction_array:initial_split_and_replenishment_fallback"
_COLUMN_ARRAYS = ("window_kg", "base", "deep_by_class_kg", "deep_total_kg", "deep_room_kg")
# Logical Source projections, NOT measured ndarray/Python ABI:32 reference
# words per array/header/shape role plus one bounded error-text frontier.
# Actual allocator/opaque header costs remain under Native reserve/OS guards.
_ARRAY_HEADER_BYTES = 32 * 8
_ERROR_TEXT_BYTES = 4096


class WindowedBedError(DenseBedError):
    """A coded refusal of the windowed holder (a DenseBedError, so the runner's existing taxonomy applies)."""


def _native() -> SimpleNamespace:
    """Actual native symbols, imported lazily (exact paths of the frozen sources)."""
    import numpy as np
    from maple.aeolian.cohorts.dispatch import mobile_state_is_genuinely_empty
    from maple.aeolian.scheduler.state import SchedulerPhysicalState, SchedulerState
    from maple.aeolian.trajectories.drag import TRAJECTORY_ALGORITHM_VERSION
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.cohort import MobileCohortState
    from maple.core.types.ledger_reconciliation import LedgerReconciliationBaseline
    from maple.core.types.sediment_availability import SedimentAvailabilityState
    from maple.core.types.sediment_ledger import SedimentLedgerState
    from maple.core.types.threshold_modifiers import ThresholdModifierState, validate_threshold_modifier_state
    from maple.core.types.topographic_commit import CommittedTopographyState
    from maple.core.types.vegetation import VegetationState, validate_vegetation_state, vegetation_content_hash
    from maple.core.types.voxel import WindowedVoxelColumnState
    from maple.core.types.water import WaterState
    from maple.surface.active_layer.windowed_session import open_windowed_exchange
    from maple.surface.active_layer.bed_exchange import _plan_windowed_bed_exchange
    from maple_syrup import windowed_shared_bed as shared
    from maple_syrup.sediment_bed import BedIntegrationError, is_windowed_bed

    return SimpleNamespace(**{name: value for name, value in locals().items()})


def _refuse(code: str, detail: str) -> WindowedBedError:
    return WindowedBedError(code, detail)


def _wrap(code: str, call, *args, **kwargs):
    """Run a native validator/constructor; its ValueError/TypeError becomes a coded refusal (cause kept)."""
    try:
        return call(*args, **kwargs)
    except DenseBedError:
        raise
    except (ValueError, TypeError) as exc:
        raise _refuse(code, f"{getattr(call, '__name__', call)}: {type(exc).__name__}: {exc}"[:400]) from exc


def _plain_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise _refuse("invalid_limit", f"{name} must be a plain int >= 1")
    return value


def _exact(value: Any, expected: type, name: str) -> None:
    if type(value) is not expected:
        raise _refuse("unsupported_type", f"{name} must be exactly {expected.__module__}.{expected.__qualname__}")


def _field_names(value: Any) -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(value))


@dataclasses.dataclass(frozen=True)
class WindowedBedLimits:
    """Explicit caps checked BEFORE a copy (resource policy, never physics)."""

    stage_copy_max_bytes: int
    held_input_max_bytes: int

    def __post_init__(self) -> None:
        _plain_int(self.stage_copy_max_bytes, "stage_copy_max_bytes")
        _plain_int(self.held_input_max_bytes, "held_input_max_bytes")

    def as_record(self) -> dict[str, int]:
        return dataclasses.asdict(self)


# --- exact typed copies and identity projections ---------------------------------------------------------------------

def _leaf_copy(np: Any, item: Any, where: str) -> Any:
    """Owned copy of one typed field value; arrays keep dtype, layout and the Native freeze flag."""
    if type(item) is np.ndarray:
        copied = np.array(item, copy=True, order="K")
        if not item.flags.writeable:
            copied.flags.writeable = False
        return copied
    if (item is None or type(item) is bool or type(item) is int or type(item) is float or type(item) is str
            or isinstance(item, (np.generic, enum.Enum))):
        return item
    if type(item) is tuple:
        return tuple(_leaf_copy(np, value, f"{where}[{i}]") for i, value in enumerate(item))
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return _typed_copy(np, item, where)
    raise _refuse("uncopyable_holder", f"{where} has unsupported type {type(item).__qualname__}")


def _typed_copy(np: Any, value: Any, where: str) -> Any:
    """Copy a Native holder through its exact typed field schema. An init=False owner that is set cannot be
    reconstructed faithfully by a constructor; it is refused instead of being silently dropped."""
    kwargs = {}
    for f in dataclasses.fields(value):
        item = getattr(value, f.name)
        if not f.init:
            if item is not None:
                raise _refuse("uncopyable_holder", f"{where}.{f.name} is an init=False owner")
            continue
        kwargs[f.name] = _leaf_copy(np, item, f"{where}.{f.name}")
    return type(value)(**kwargs)


def _holder_bytes(*values: Any) -> tuple[int, int]:
    """(bytes, array count) of every array leaf; additive per holder, no alias credit."""
    total, count = 0, 0
    for value in values:
        leaves: list[tuple[str, Any]] = []
        _dense._leaves(value, "", leaves)
        total += sum(int(array.nbytes) for _, array in leaves)
        count += len(leaves)
    return total, count


def _pending_records(column: Any) -> int:
    pending = column.pending_deep
    return 0 if pending is None else int(pending.journal.n_records)


def _pending_identity(column: Any) -> Any:
    """The journal is part of the physical bed, not just a record count.

    Native validate_pending_binding owns the immutable version/array binding;
    these hashes also detect an outside write into any of its three buffers.
    """
    pending = column.pending_deep
    if pending is None:
        return None
    journal = pending.journal
    return {"header_digest": journal.header_digest,
            "cell_flat": _dense._array_digest(journal.cell_flat),
            "voxel": _dense._array_digest(journal.voxel),
            "values_kg": _dense._array_digest(journal.values_kg)}


def _column_identity(column: Any) -> dict[str, Any]:
    """Evolving identity of the resident column and its deep authority (no deep read, no derived cache)."""
    store = column.deep_store
    packet = column.height_packet
    return {
        "resident": {name: _dense._array_digest(getattr(column, name)) for name in _COLUMN_ARRAYS},
        "n_voxels": _dense._canonical_static(column.n_voxels, "voxel_column.n_voxels"),
        "generation": _dense._canonical_static(column.generation, "voxel_column.generation"),
        "manifest_digest": _dense._canonical_static(column.manifest_digest, "voxel_column.manifest_digest"),
        "deep_store": {"store_dir": str(store.store_dir), "manifest_path": str(store.manifest_path),
                       "version_id": _dense._canonical_static(store.version_id, "deep_store.version_id"),
                       "format_version": _dense._canonical_static(store.format_version, "deep_store.format_version"),
                       "encoding": _dense._canonical_static(store.encoding, "deep_store.encoding")},
        "pending_records": _pending_records(column),
        "pending_journal": _pending_identity(column),
        "prefix_packet": None if packet is None else _dense._array_digest(packet.packet),
    }


def _field_identity(state: Any) -> dict[str, Any]:
    record = {"voxel_column": _column_identity(state.voxel_column)}
    for name in WINDOWED_FIELDS[1:]:
        record[name] = _dense._canonical_static(getattr(state, name), name)
    return record


# --- the bed ----------------------------------------------------------------------------------------------------------

_TOKEN = object()


@dataclasses.dataclass(frozen=True)
class WindowedStage:
    """A bounded independent resident copy of the seven shared fields at one boundary (never persistence)."""

    case_sha256: str
    state: Any
    time_s: float
    identity_sha256: str
    nbytes: int
    cost_terms: tuple
    pending_records_in: int
    pending_records_out: int
    # Private owner token of the bed instance that made the stage; never part of any hash.
    _owner: object = dataclasses.field(default=None, compare=False, repr=False)


class WindowedBed:
    """MASY-owned holder of the CURRENT windowed bed. Construct only with `build_windowed_bed`.
    Mutated only by `adopt_*`, which validate before replacing anything."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("WindowedBed is closed")

    def __init__(self, token: object = None, **fields: Any) -> None:
        if token is not _TOKEN:
            raise WindowedBedError("private_constructor", "construct a WindowedBed with build_windowed_bed")
        self._case_record = fields["case_record"]
        self._case_sha256 = _dense._sha256(self._case_record)
        self._held_inputs = fields["held_inputs"]
        self._context = fields["context"]
        self._state = fields["state"]
        self._vegetation = fields["vegetation"]
        self._threshold_modifier_state = fields["threshold_modifier_state"]
        self._semantics = fields["semantics"]
        self._time_s = fields["time_s"]
        self._limits = fields["limits"]
        self._prescribed_digests = fields["prescribed_digests"]
        self._context_refs = {name: getattr(self._context, name) for name in CONTEXT_FIELDS}
        self._owner = object()  # instance/lineage token for stages; separate from every content digest

    # read-only views ---------------------------------------------------------------------------------------------
    @property
    def case_record(self) -> dict[str, Any]:
        import copy
        return copy.deepcopy(self._case_record)

    @property
    def case_sha256(self) -> str:
        return self._case_sha256

    @property
    def context(self) -> Any:
        """The one actual SYRUP `WindowedAvalancheContext`."""
        return self._context

    @property
    def state(self) -> Any:
        """The CURRENT actual SYRUP `WindowedSharedBedState` (shared objects, by reference)."""
        return self._state

    @property
    def vegetation(self) -> Any:
        return self._vegetation

    @property
    def threshold_modifier_state(self) -> Any:
        return self._threshold_modifier_state

    @property
    def trajectory_semantics(self) -> str:
        return self._semantics

    @property
    def time_s(self) -> float:
        return self._time_s

    @property
    def limits(self) -> WindowedBedLimits:
        return self._limits

    def vegetation_cover_fraction(self):
        """The canonical prescribed cover FRACTION array (same object; never converted)."""
        return self._vegetation.raster.cover_fraction

    # verification ------------------------------------------------------------------------------------------------
    def verify_case_inputs(self) -> None:
        self._verify_static(_native())

    def _live_static(self, n: SimpleNamespace) -> dict[str, Any]:
        ctx = self._context
        try:
            return {
                "geometry": _dense._canonical_static(ctx.geometry, "geometry"),
                "grain_classes": _dense._canonical_static(ctx.grain_classes, "grain_classes"),
                "class_order": [c.class_id for c in ctx.grain_classes.classes],
                "mass_resolution_kg": encode_time(ctx.mass_resolution_kg),
                "topographic_commit_spec": _dense._canonical_static(ctx.topographic_commit_spec,
                                                                    "topographic_commit_spec"),
                "avalanching_spec": _dense._canonical_static(ctx.avalanching_spec, "avalanching_spec"),
                "initial_available_fraction": _dense._array_digest(ctx.initial_available_fraction),
                "context_options": {name: _dense._canonical_static(getattr(ctx, name), name) for name in (
                    "depth_update_rule", "record_exchange_history", "window_max_journal_records",
                    "deep_chunk_encoding", "deep_overlay_max_bytes", "source_identity")},
                "native_limits": {name: _dense._canonical_static(getattr(ctx, name), name) for name in NATIVE_POLICIES},
                "bridge_limits": _bridge_record(ctx.bridge_limits),
                "bed_limits": self._limits.as_record(),
            }
        except (DenseBedError, ValueError, TypeError, AttributeError) as exc:
            raise _refuse("case_input_mutated", f"a live static input no longer canonicalizes ({exc})"[:400]) from exc

    def _verify_static(self, n: SimpleNamespace) -> None:
        """Live static inputs, held inputs and prescribed fields against the original record; never mutates."""
        record = self._case_record
        if _dense._sha256(record) != self._case_sha256:
            raise _refuse("case_input_mutated", "the held case record no longer reproduces case_sha256")
        if type(self._context) is not n.shared.WindowedAvalancheContext:
            raise _refuse("case_input_mutated", "the held WindowedAvalancheContext was replaced")
        for name, ref in self._context_refs.items():
            if getattr(self._context, name, None) is not ref:
                raise _refuse("case_input_mutated", f"context.{name} is no longer the original case object")
        live = self._live_static(n)
        changed = sorted(key for key, value in live.items() if value != record[key])
        if changed:
            raise _refuse("case_input_mutated", f"live static input(s) {changed} differ from the case record")
        for name, array in self._held_inputs.items():
            if array.flags.writeable or record["held_inputs"][name] != _dense._array_digest(array):
                raise _refuse("case_input_mutated", f"held initial input {name} changed")
        current = {
            "vegetation": _dense._sha256(_dense._canonical_static(self._vegetation, "vegetation")),
            "native_vegetation_content_sha256": n.vegetation_content_hash(self._vegetation),
            "threshold_modifier_state": _dense._sha256(_dense._canonical_static(self._threshold_modifier_state,
                                                                                "threshold_modifier_state")),
        }
        if current != self._prescribed_digests:
            changed = sorted(key for key in current if current[key] != self._prescribed_digests[key])
            raise _refuse("prescribed_field_changed", f"prescribed field(s) {changed} changed; no growth or "
                                                      "moisture law exists in this adapter")
        if self._semantics != record["trajectory_semantics"]:
            raise _refuse("case_input_mutated", "held trajectory semantics changed")

    def _validate_shared(self, n: SimpleNamespace, state: Any, time_s: float) -> None:
        """Exact types, Small profile shape, commit stamp and the actual Native shared validation."""
        _exact(state, n.shared.WindowedSharedBedState, "bed state")
        if _field_names(state) != WINDOWED_FIELDS:
            raise _refuse("incompatible_shared_field", "the shared bed is not the exact seven-field schema")
        expected = {"voxel_column": n.WindowedVoxelColumnState, "active_layer": n.ActiveLayerState,
                    "water": n.WaterState, "ledger": n.SedimentLedgerState,
                    "committed_topography": n.CommittedTopographyState,
                    "sediment_availability": n.SedimentAvailabilityState}
        for name, cls in expected.items():
            value = getattr(state, name)
            if value is None:
                raise _refuse("missing_shared_field", f"{name} is None; shared fields are never dropped")
            _exact(value, cls, name)
            if name != "voxel_column":
                _dense._require_host_arrays(value, name)
        baseline = state.reconciliation_baseline
        if baseline is not None:
            _exact(baseline, n.LedgerReconciliationBaseline, "reconciliation_baseline")
            _dense._require_host_arrays(baseline, "reconciliation_baseline")
        if _wrap("unsupported_representation", n.is_windowed_bed, state) is not True:
            raise _refuse("unsupported_representation", "only the Native windowed column is supported")
        column = state.voxel_column
        for name in _COLUMN_ARRAYS:
            if type(getattr(column, name)) is not n.np.ndarray:
                raise _refuse("unsupported_backend", f"voxel_column.{name} is not a host numpy.ndarray")
        header, record = column.deep_store.header, self._case_record
        shape = (header.ny, header.nx, header.n_voxels, header.window_voxels, header.n_classes)
        if (shape != tuple(PROFILE_SHAPE.values()) or column.n_voxels != record["n_voxels"]
                or column.deep_store.format_version != 1 or column.deep_store.encoding != "full"
                or str(column.deep_store.store_dir) != record["initial_deep_store"]["store_dir"]):
            raise _refuse("incompatible_shared_field", f"windowed column {shape} is outside the C16/V6/W3/K1 "
                                                       "full-v1 profile of this case store")
        _dense._check_commit_stamp(state, time_s)
        _wrap("invalid_shared_field", n.shared.validate_shared_bed, state, self._context,
              current_time_s=time_s, published=column.pending_deep is None)

    def _advance_time(self, time_s: Any) -> float:
        if type(time_s) is not float:
            raise _refuse("invalid_time", "a plain Native float clock is required")
        try:
            value = _native_clock(time_s, "time_s")
        except ValueError as exc:
            raise _refuse("invalid_time", str(exc)) from exc
        if value < self._time_s:
            raise _refuse("time_reversed", f"time {value.hex()} precedes the bed time {self._time_s.hex()}")
        return value

    # identity and bounded staging --------------------------------------------------------------------------------
    def _identity_of(self, state: Any, time_s: float, *, case_sha256: str | None = None) -> dict[str, Any]:
        try:
            fields = _field_identity(state)
        except (DenseBedError, AttributeError, TypeError) as exc:
            raise _refuse("unhashable_shared_field", f"{type(exc).__name__}: {exc}"[:400]) from exc
        return {"schema": WINDOWED_BED_IDENTITY_SCHEMA,
                "case_sha256": self._case_sha256 if case_sha256 is None else case_sha256,
                "time_s": encode_time(time_s), "fields": fields,
                "prescribed": dict(self._prescribed_digests), "trajectory_semantics": self._semantics}

    def identity_record(self) -> dict[str, Any]:
        """Evolving bed identity (separate from the static case digest)."""
        return self._identity_of(self._state, self._time_s)

    def identity_sha256(self) -> str:
        return _dense._sha256(self.identity_record())

    def candidate_identity_sha256(self, state: Any, time_s: float) -> str:
        return _dense._sha256(self._identity_of(state, time_s))

    def stage_cost(self) -> dict[str, int]:
        """Positive logical bytes of ONE stage, planned before resident copies. Python metadata and the
        genuine Native planner have their own positive roles; this is not a zero-allocation claim. Every copied
        holder, the Native begin copies (window, optional totals, metadata, packets), the fresh window cache and,
        when prior pending records exist, the retained journal plus its concatenation/canonicalization/new journal
        are additive. Alias, release and garbage-collection credit are never taken."""
        state = self._state
        column = state.voxel_column
        ctx = self._context
        # Reuse the complete genuine Native copying/finish/journal/partition
        # validator envelope, including its cold-order and declared callee
        # scratch terms. The context's two-operation policy overcharges this
        # zero-operation stage. A plan is only a budget, never a physical
        # proof/token: the public begin/finish still perform their own checks.
        native_plan = _wrap("native_stage_admission", _native()._plan_windowed_bed_exchange,
            column, state.active_layer, ctx.geometry,
            class_ids=tuple(c.class_id for c in ctx.grain_classes.classes), limits=ctx.native_iteration_limits)
        holders, holder_arrays = _holder_bytes(*(getattr(state, name) for name in WINDOWED_FIELDS[1:]
                                                 if getattr(state, name) is not None))
        resident = sum(int(getattr(column, name).nbytes) for name in _COLUMN_ARRAYS)
        window = int(column.window_kg.nbytes)
        packet = column.height_packet
        if packet is None:
            raise _refuse("missing_prefix_packet", "the accepted column carries no maintained prefix packet")
        packet_bytes = int(packet.packet.nbytes) + int(packet.binding.encoded_base.nbytes)
        h = column.deep_store.header
        cache = 4 * h.ny * h.nx * h.window_voxels * 8 + window
        # PendingDeepBinding is deliberately opaque to the generic dataclass
        # walker. Name its three genuine journal buffers explicitly. Public
        # begin retains the prior arrays; finish concatenates/canonicalizes,
        # then PendingDeepBinding makes independent copies of the result.
        j = None if column.pending_deep is None else column.pending_deep.journal
        pending_bytes = 0 if j is None else sum(int(a.nbytes) for a in (j.cell_flat, j.voxel, j.values_kg))
        records = _pending_records(column)
        pending = 8 * pending_bytes + 8 * 8 * records  # buffers plus sort/keep/index roles, no release credit
        arrays = holder_arrays + len(_COLUMN_ARRAYS) + 8 + (64 if j is not None else 8)
        terms = {"typed_holder_copies": holders, "begin_resident_copies": resident,
                 "begin_optional_totals": window, "begin_packet_copy": packet_bytes,
                 "finish_packet_and_binding": packet_bytes, "finish_window_cache": cache,
                 "prior_pending_journal_roles": pending,
                 "native_copying_finish_and_complete_validator_envelope": int(native_plan.new_host_bytes_peak),
                 "native_mapping_envelope": int(native_plan.mapped_bytes_max),
                 "typed_constructor_kwargs_and_field_reference_words": 8 * 8 * sum(
                     len(dataclasses.fields(getattr(state, name))) for name in WINDOWED_FIELDS[1:]
                     if getattr(state, name) is not None),
                 "logical_headers_and_error_text": _ARRAY_HEADER_BYTES * arrays + _ERROR_TEXT_BYTES}
        terms["total"] = sum(terms.values())
        return terms

    def stage_copy(self) -> WindowedStage:
        """Bounded independent resident copy of the current seven fields through the admitted Native copying
        exchange. Priced and capped BEFORE any copy; validated before it is returned."""
        n = _native()
        np = n.np
        self._verify_static(n)
        cost = self.stage_cost()
        if cost["total"] > self._limits.stage_copy_max_bytes:
            raise _refuse("stage_copy_cap_exceeded", f"{cost['total']} bytes exceed stage_copy_max_bytes="
                                                     f"{self._limits.stage_copy_max_bytes}")
        identity = self.identity_sha256()
        state, ctx = self._state, self._context
        column = state.voxel_column
        ids = tuple(c.class_id for c in ctx.grain_classes.classes)
        # The admitted session passes the active layer by reference: copy it first, then the other holders.
        active = _typed_copy(np, state.active_layer, "active_layer")
        copies = {name: (None if getattr(state, name) is None else _typed_copy(np, getattr(state, name), name))
                  for name in ("water", "ledger", "committed_topography", "sediment_availability",
                               "reconciliation_baseline")}
        session = _wrap("native_stage_refused", n.open_windowed_exchange, column, active, ctx.geometry,
                        ctx.mass_resolution_kg, class_ids=ids,
                        max_gather_columns=ctx.native_iteration_limits.max_gather_columns,
                        max_journal_records=ctx.window_max_journal_records, max_operations=1,
                        trust_window_cache=False)
        try:
            outcome = session.finish()  # zero exchange operations; genuine journal/packet/cache assembly
        except BaseException:
            session.abandon()
            raise
        if outcome.n_operations != 0 or outcome.new_active_layer is not active:
            raise _refuse("native_stage_changed", "the zero-operation session changed the staged active layer")
        new = outcome.new_column
        if (new.deep_store is not column.deep_store or new.generation != column.generation
                or new.manifest_digest != column.manifest_digest or new.height_packet is None):
            raise _refuse("native_stage_changed", "the staged column is not bound to the accepted deep version")
        for name in _COLUMN_ARRAYS:
            if np.shares_memory(getattr(new, name), getattr(column, name)):
                raise _refuse("stage_not_independent", f"staged voxel_column.{name} shares the accepted resident array")
        if np.shares_memory(new.height_packet.packet, column.height_packet.packet):
            raise _refuse("stage_not_independent", "staged prefix packet shares the accepted packet")
        staged = n.shared.WindowedSharedBedState(voxel_column=new, active_layer=active, **copies)
        if _dense._sha256(self._identity_of(staged, self._time_s)) != identity:
            raise _refuse("stage_copy_mismatch", "staged copy does not reproduce the bed identity")
        self._validate_shared(n, staged, self._time_s)
        return WindowedStage(self._case_sha256, staged, self._time_s, identity, cost["total"],
                             tuple(sorted(cost.items())), _pending_records(column), int(outcome.pending_records),
                             _owner=self._owner)

    def _require_stage(self, stage: Any) -> None:
        if type(stage) is not WindowedStage or stage.case_sha256 != self._case_sha256:
            raise _refuse("foreign_stage", "stage does not belong to this case")
        if stage._owner is not self._owner:
            raise _refuse("foreign_stage", "stage was made by a different bed instance")
        if stage.identity_sha256 != self.identity_sha256():
            raise _refuse("stale_stage", "the accepted bed changed since this stage was made")
        if _dense._sha256(self._identity_of(stage.state, stage.time_s, case_sha256=stage.case_sha256)) \
                != stage.identity_sha256:
            raise _refuse("stage_mutated", "staged content no longer matches its identity")

    def stage_scheduler_state(self, stage: WindowedStage, mobile_state: Any, *, time_s: float):
        """A genuine fresh Native `SchedulerState` over the stage: all twelve physical fields explicit (the
        reconciliation baseline included), cursor (0, +0.0), coupled None and absent derived caches."""
        n = _native()
        self._verify_static(n)
        self._require_stage(stage)
        if type(time_s) is not float or time_s.hex() != stage.time_s.hex() or time_s.hex() != self._time_s.hex():
            raise _refuse("time_mismatch", f"time_s {time_s!r} must equal the stage and bed time {self._time_s.hex()}")
        _exact(mobile_state, n.MobileCohortState, "mobile_state")
        _dense._require_host_arrays(mobile_state, "mobile_state")
        if not n.mobile_state_is_genuinely_empty(mobile_state):
            raise _refuse("mobile_state_not_empty", "a fresh windowed wind entry needs a genuinely empty pool")
        state = stage.state
        if bool(n.np.any(state.water.mobile_mass_by_cell_class_kg != 0.0)):
            raise _refuse("water_mobile_not_empty", "water-borne sediment remains; no fresh wind entry")
        physical = n.SchedulerPhysicalState(
            voxel_column=state.voxel_column, active_layer=state.active_layer, mobile_state=mobile_state,
            ledger=state.ledger, committed_topography=state.committed_topography,
            threshold_modifier_state=self._threshold_modifier_state, time_s=time_s, vegetation=self._vegetation,
            sediment_availability=state.sediment_availability, water=state.water,
            coupled_trajectory_semantics=self._semantics,
            ledger_reconciliation_baseline=state.reconciliation_baseline)
        scheduler = n.SchedulerState(physical=physical, coupled=None, block_index=0, block_offset_s=0.0,
                                     last_resolved_wind=None, topographic_wind_cache=None,
                                     vegetation_wind_cache=None, active_layer_composition_cache=None)
        if _field_names(physical) != PHYSICAL_FIELDS or _field_names(scheduler) != SCHEDULER_FIELDS:
            raise _refuse("native_schema_changed", "Native scheduler carriers are not the pinned schemas")
        return scheduler

    # adoption ----------------------------------------------------------------------------------------------------
    def _prepare_scheduler_physical(self, n: SimpleNamespace, physical: Any):
        """Complete wind-adoption guards: the seven-field map keeps the reconciliation baseline."""
        _exact(physical, n.SchedulerPhysicalState, "physical")
        if _field_names(physical) != PHYSICAL_FIELDS:
            raise _refuse("native_schema_changed", "SchedulerPhysicalState is not the pinned twelve-field schema")
        if physical.vegetation is not self._vegetation:
            raise _refuse("prescribed_field_replaced", "physical.vegetation is not the held prescribed object")
        if physical.threshold_modifier_state is not self._threshold_modifier_state:
            raise _refuse("prescribed_field_replaced", "physical.threshold_modifier_state is not the held object")
        if physical.coupled_trajectory_semantics != self._semantics:
            raise _refuse("trajectory_semantics_changed", "returned wind content carries different semantics")
        _exact(physical.mobile_state, n.MobileCohortState, "physical.mobile_state")
        if not n.mobile_state_is_genuinely_empty(physical.mobile_state):
            raise _refuse("mobile_state_not_empty", "airborne wind load remains; no bed handoff")
        state = n.shared.WindowedSharedBedState(
            voxel_column=physical.voxel_column, active_layer=physical.active_layer, water=physical.water,
            ledger=physical.ledger, committed_topography=physical.committed_topography,
            sediment_availability=physical.sediment_availability,
            reconciliation_baseline=physical.ledger_reconciliation_baseline)
        return state, physical.time_s

    def _prepare_adopt(self, n: SimpleNamespace, state: Any, time_s: Any):
        value = self._advance_time(time_s)
        self._validate_shared(n, state, value)
        return state, value

    def adopt_scheduler_physical(self, physical: Any) -> Any:
        """Native wind adoption: every check precedes the single replacement."""
        n = _native()
        self._verify_static(n)
        state, time_s = self._prepare_scheduler_physical(n, physical)
        state, value = self._prepare_adopt(n, state, time_s)
        self._state, self._time_s = state, value
        return state

    def adopt_bed_state(self, state: Any, *, time_s: float) -> Any:
        """Take the actual seven-field SYRUP bed returned by native water code (`DryHandoff.dry_state.bed`)."""
        n = _native()
        self._verify_static(n)
        state, value = self._prepare_adopt(n, state, time_s)
        self._state, self._time_s = state, value
        return state


def _bridge_record(limits: Any) -> dict[str, Any]:
    if type(limits.run_dir) is not str or not 0 < len(limits.run_dir) <= 4096:
        raise _refuse("invalid_bridge_limits", "the closed MASY profile requires a bounded plain run-directory string")
    record = {f.name: getattr(limits, f.name) for f in dataclasses.fields(limits) if f.name != "run_dir"}
    if any(type(value) is not int for value in record.values()):
        raise _refuse("invalid_bridge_limits", "bridge limit declarations must be plain ints")
    record["run_dir"] = limits.run_dir
    record["type"] = f"{type(limits).__module__}.{type(limits).__qualname__}"
    return record


def build_windowed_bed(*, physical: Any, context: Any, limits: WindowedBedLimits) -> WindowedBed:
    """Validate a caller-built genuine Native windowed `SchedulerPhysicalState` and its exact
    `WindowedAvalancheContext`, then hold their seven shared fields. Nothing is copied before every cheap
    check passes; the actual Native shared validation runs before the holder is returned."""
    if type(limits) is not WindowedBedLimits:
        raise _refuse("invalid_limit", "limits must be a WindowedBedLimits")
    n = _native()
    np = n.np
    _exact(physical, n.SchedulerPhysicalState, "physical")
    if _field_names(physical) != PHYSICAL_FIELDS:
        raise _refuse("native_schema_changed", "SchedulerPhysicalState is not the pinned twelve-field schema")
    _exact(context, n.shared.WindowedAvalancheContext, "context")
    if _field_names(context) != CONTEXT_FIELDS:
        raise _refuse("native_schema_changed", "WindowedAvalancheContext is not the pinned schema")
    _exact(context.bridge_limits, n.shared.PositiveWindowedWaterBridgeLimits, "context.bridge_limits")
    _bridge_record(context.bridge_limits)  # before Native validation and any held-input copy
    g = context.geometry
    classes = context.grain_classes.classes
    if g.dx_m != g.dy_m or (g.ny, g.nx) != (PROFILE_SHAPE["ny"], PROFILE_SHAPE["nx"]) or len(classes) != 1:
        raise _refuse("unsupported_profile", "only the square 4x4 K1 Small profile is supported")
    if (context.depth_update_rule != "constant_depth" or context.record_exchange_history is not False
            or context.deep_chunk_encoding != "full" or context.deep_overlay_max_bytes is not None
            or type(context.window_max_journal_records) is not int
            or context.window_max_journal_records != JOURNAL_RECORDS
            or context.avalanching_spec.enabled is not True):
        raise _refuse("unsupported_profile", "constant_depth, no history, full-v1, journal 320 and enabled "
                                             "Native avalanching are required")
    expected = {"voxel_column": n.WindowedVoxelColumnState, "active_layer": n.ActiveLayerState,
                "mobile_state": n.MobileCohortState, "ledger": n.SedimentLedgerState,
                "committed_topography": n.CommittedTopographyState, "water": n.WaterState,
                "sediment_availability": n.SedimentAvailabilityState, "vegetation": n.VegetationState,
                "threshold_modifier_state": n.ThresholdModifierState}
    for name, cls in expected.items():
        _exact(getattr(physical, name), cls, f"physical.{name}")
    if physical.ledger_reconciliation_baseline is not None:
        _exact(physical.ledger_reconciliation_baseline, n.LedgerReconciliationBaseline, "reconciliation_baseline")
    column = physical.voxel_column
    if column.pending_deep is not None or column.height_packet is None:
        raise _refuse("unsupported_profile", "the initial column must be published with maintained prefix packets")
    if type(physical.time_s) is not float:
        raise _refuse("invalid_time", "physical.time_s must be a plain float")
    try:
        t0 = _native_clock(physical.time_s, "physical.time_s")
    except ValueError as exc:
        raise _refuse("invalid_time", str(exc)) from exc
    _dense._require_host_arrays(physical.mobile_state, "mobile_state")
    if not n.mobile_state_is_genuinely_empty(physical.mobile_state):
        raise _refuse("mobile_state_not_empty", "the initial windowed bed carries airborne wind content")
    if physical.coupled_trajectory_semantics != n.TRAJECTORY_ALGORITHM_VERSION:
        raise _refuse("trajectory_semantics_unknown", "empty initial wind content must carry the current semantics")
    water = physical.water
    _dense._require_host_arrays(water, "water")
    if (bool(np.any(water.depth_m != 0.0)) or bool(np.signbit(water.depth_m).any())
            or bool(np.any(water.mobile_mass_by_cell_class_kg != 0.0))):
        raise _refuse("water_not_dry", "the initial constant-depth water must be dry (+0 depth, empty load)")
    vegetation, threshold = physical.vegetation, physical.threshold_modifier_state
    _dense._require_host_arrays(vegetation, "vegetation")
    _dense._require_host_arrays(threshold, "threshold_modifier_state")
    _wrap("invalid_vegetation", n.validate_vegetation_state, vegetation, g.ny, g.nx, geometry=g)
    _wrap("invalid_threshold_modifiers", n.validate_threshold_modifier_state, threshold, g.ny, g.nx)
    fraction = context.initial_available_fraction
    if (type(fraction) is not np.ndarray or fraction.dtype != np.float64 or fraction.shape != (g.ny, g.nx, 1)
            or not bool(np.isfinite(fraction).all())):
        raise _refuse("invalid_static_input", "initial_available_fraction must be a finite (ny, nx, 1) float64 array")
    if int(fraction.nbytes) > limits.held_input_max_bytes:
        raise _refuse("held_input_cap_exceeded", "initial availability copy exceeds held_input_max_bytes")
    held = np.array(fraction, dtype=np.float64, order="C", copy=True)
    held.flags.writeable = False
    store, header = column.deep_store, column.deep_store.header
    try:
        case_record = {
            "schema": WINDOWED_CASE_SCHEMA, "backend": "numpy", "dtype": "float64",
            "representation": "windowed_voxel_column_full_v1",
            "geometry": _dense._canonical_static(g, "geometry"),
            "grain_classes": _dense._canonical_static(context.grain_classes, "grain_classes"),
            "class_order": [c.class_id for c in classes],
            "mass_resolution_kg": encode_time(context.mass_resolution_kg),
            "n_voxels": column.n_voxels, "window_voxels": header.window_voxels,
            "initial_time_s": encode_time(t0),
            "initial_available_fraction": _dense._array_digest(fraction),
            "initial_availability_policy": INITIAL_AVAILABILITY_POLICY,
            "topographic_commit_spec": _dense._canonical_static(context.topographic_commit_spec,
                                                                "topographic_commit_spec"),
            "avalanching_spec": _dense._canonical_static(context.avalanching_spec, "avalanching_spec"),
            "context_options": {name: _dense._canonical_static(getattr(context, name), name) for name in (
                "depth_update_rule", "record_exchange_history", "window_max_journal_records",
                "deep_chunk_encoding", "deep_overlay_max_bytes", "source_identity")},
            "native_limits": {name: _dense._canonical_static(getattr(context, name), name)
                              for name in NATIVE_POLICIES},
            "bridge_limits": _bridge_record(context.bridge_limits),
            "bed_limits": limits.as_record(),
            "held_inputs": {"initial_available_fraction": _dense._array_digest(held)},
            "vegetation": _dense._canonical_static(vegetation, "vegetation"),
            "native_vegetation_content_sha256": n.vegetation_content_hash(vegetation),
            "threshold_modifier_state": _dense._canonical_static(threshold, "threshold_modifier_state"),
            "trajectory_semantics": physical.coupled_trajectory_semantics,
            "initial_deep_store": {"store_dir": str(store.store_dir), "version_id": int(store.version_id),
                                   "format_version": int(store.format_version), "encoding": str(store.encoding),
                                   "header": {"ny": int(header.ny), "nx": int(header.nx),
                                              "n_voxels": int(header.n_voxels),
                                              "window_voxels": int(header.window_voxels),
                                              "n_classes": int(header.n_classes), "n_chunks": int(header.n_chunks),
                                              "class_ids": [str(c) for c in header.class_ids]}},
        }
    except DenseBedError as exc:
        raise _refuse(exc.code, exc.detail) from exc
    prescribed = {"vegetation": _dense._sha256(case_record["vegetation"]),
                  "native_vegetation_content_sha256": case_record["native_vegetation_content_sha256"],
                  "threshold_modifier_state": _dense._sha256(case_record["threshold_modifier_state"])}
    state = n.shared.WindowedSharedBedState(
        voxel_column=column, active_layer=physical.active_layer, water=water, ledger=physical.ledger,
        committed_topography=physical.committed_topography, sediment_availability=physical.sediment_availability,
        reconciliation_baseline=physical.ledger_reconciliation_baseline)
    bed = WindowedBed(_TOKEN, case_record=case_record, held_inputs={"initial_available_fraction": held},
                      context=context, state=state, vegetation=vegetation, threshold_modifier_state=threshold,
                      semantics=physical.coupled_trajectory_semantics, time_s=t0, limits=limits,
                      prescribed_digests=prescribed)
    bed._verify_static(n)
    bed._validate_shared(n, state, t0)
    return bed
