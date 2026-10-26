"""Closed windowed CPU runner: ordered wind / rain events on ONE Native windowed bed.

`WindowedRunner` reuses the existing `DenseRunner` engine (schedule resolution,
precheck, ClockLedger, `_finish` budgets/receipts/identity rechecks and the one
adoption, stop receipts) with a separate owner of a `WindowedBed`. It holds no
physics. Every transport, commit, settlement and dry reset is an actual MAPLE /
SYRUP call:

* wind: `prepare_windowed_small_feature_wind_record` ONCE and the returned
  `WindowedWindRecord.advance()` ONCE (one 1 s Native block, Native pending
  publication, full boundary and exactly one Native finalizer). The record's
  complete result is checked and handed to the inherited `_finish` with the
  seven-field candidate. The Dense `_wind_accept` (six-field rebuild, second
  finalizer) is never used.
* rain: the ORIGINAL `_rain_enter` / `_rain_native` code objects, bound to a
  copy of their module globals in which only the commit delegate (callback-free
  `commit_positive_windowed_entry`) and the event delegate
  (`complete_positive_windowed_event`, unpacked to its DryHandoff) differ, then
  the original `_rain_accept` unchanged. Entry and event admission summaries are
  aggregated and refuse adoption above the selected cumulative ceilings.

Construction is `build_windowed_runner` only (the ordinary constructor refuses).
It establishes every original DenseRunner field and guard with windowed
counterparts where the Dense projection is incompatible. Only the explicit
Small profile is supported; everything else refuses before any native call.
Not a restart, GPU, K6, calibration or production-scale path. See
docs/windowed_cpu_integration.md. Importing this module imports neither numpy
nor a native model.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import math
import types
from fractions import Fraction
from types import ModuleType, SimpleNamespace
from typing import Any

from masy import dense_bed as _bed
from masy import dense_production as _production
from masy import dense_runner as _dense
from masy import windowed_bed as _windowed_bed
from masy import windowed_process as _windowed_process
from masy.clock import decode_time, encode_time
from masy.dense_bed import DenseBedError
from masy.dense_runner import (
    DenseRunner,
    DenseRunnerControl,
    DenseRunnerError,
    DenseRunnerLimits,
    DenseWaterInputs,
    DenseWindInputs,
    _BedPublication,
    _canonical_input,
    _detail,
    _elapsed,
    _floats,
    _fraction_text,
    _HostOwnership,
    _SchedulerPublication,
    _Stop,
)
from masy.dense_schedule import DenseSchedule, RainEvent, WindEvent, case_schedule_digest
from masy.windowed_bed import WindowedBed

__all__ = [
    "PROFILE",
    "RECEIPT_SCHEMA",
    "REPORT_SCHEMA",
    "STATIC_SCHEMA",
    "WindowedOwnership",
    "WindowedRunner",
    "WindowedWindProfile",
    "build_windowed_runner",
]

STATIC_SCHEMA = "masy.windowed_runner_static.v1"
RECEIPT_SCHEMA = "masy.windowed_runner_event.v1"
REPORT_SCHEMA = "masy.windowed_runner_report.v1"
SCOPE = "windowed_small_K1_C16_V6_W3_full_v1_cpu_interface_only"
REPORT_ROWS = 4096
PROFILE = {
    **_dense.PROFILE, "representation": "windowed_voxel_column_full_v1", "grid": "square_full_4x4",
    "wind_detachment": "native_reduced", "avalanching": "native_enabled",
    "wind_features": "native_topographic_fft_resolved_separation_surface_creep",
    "wind_end": "native_windowed_small_record_single_finalization",
    "water": "syrup_positive_windowed_shared_bed", "water_depth_rule": "constant_depth",
    "scope": SCOPE, "calibration": "not_qualified_by_interface_tests",
}
_TOKEN = object()


def _require_windowed_scale_receiver(w: SimpleNamespace) -> None:
    """Require the genuine completed-owner zero-argument Native receiver shape.

    This checks the actual plain Python signature headers, without calling
    Native or following synthetic ``__signature__`` / ``__wrapped__`` metadata.
    Native owns Source/image/boundary/resource validation at the real call;
    Root separately qualifies the changed immutable Native subject.
    """
    owner = w.WindowedWindRecord
    if (type(owner) is not type
            or owner.__module__ != "maple.aeolian.scheduler.windowed_record"
            or owner.__name__ != "WindowedWindRecord"):
        raise DenseRunnerError("native_windowed_scale_unavailable", "exact Native record class required")
    receiver = vars(owner).get("reservoir_arithmetic_scale_kg")
    if (type(receiver) is not types.FunctionType
            or receiver.__module__ != owner.__module__
            or receiver.__qualname__ != "WindowedWindRecord.reservoir_arithmetic_scale_kg"):
        raise DenseRunnerError("native_windowed_scale_unavailable", "Native completed-owner method required")
    code = receiver.__code__
    if (code.co_argcount != 1 or code.co_posonlyargcount != 0 or code.co_kwonlyargcount != 0
            or code.co_varnames[:1] != ("self",)
            or code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS | inspect.CO_GENERATOR
                                | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR)
            or receiver.__defaults__ is not None or receiver.__kwdefaults__ is not None
            or receiver.__closure__ is not None):
        raise DenseRunnerError("native_windowed_scale_unavailable", "Native receiver must bind only self")


def _metadata_terms(np: Any, roots: tuple[Any, ...], receipt_bytes: int, events: int) -> tuple[tuple[str, int], ...]:
    """Pre-walk the NEW profile's controlling metadata before canonicalization.

    These are positive logical Source reference/text/copy reservations, NOT
    Python heap/RSS estimates. The closed finite domains guard every expanded
    occurrence (no identity/alias credit). Native scratch/provider/allocator
    owners remain additional under their own real admission and OS guards.
    """
    nodes = text = paths = payload = arrays = comparisons = 0

    def visit(value: Any, depth: int, path: int, lossless_schedule: bool = False) -> None:
        nonlocal nodes, text, paths, payload, arrays, comparisons
        nodes += 1
        paths += path
        if nodes > 16384 or depth > 32 or path > 2048 or paths > 2 << 20:
            raise DenseRunnerError("windowed_metadata_domain", "expanded controlling metadata/path domain exceeded")
        if type(value) is np.ndarray:
            if value.ndim > 4 or value.dtype.hasobject or value.nbytes > 64 << 20:
                raise DenseRunnerError("windowed_metadata_domain", "primitive bounded input array required")
            arrays += 1
            payload += int(value.nbytes)
            text += 512  # dtype/shape/digest/type keys, including rank4 and 64hex SHA
        elif value is None or type(value) is bool:
            text += 5
        elif type(value) is int:
            if value.bit_length() > 128:
                raise DenseRunnerError("windowed_metadata_domain", "bounded controlling integer required")
            text += 42
        elif type(value) is float:
            if not math.isfinite(value):
                raise DenseRunnerError("windowed_metadata_domain", "finite controlling float required")
            # Schedule.as_record/encode_time also emit the exact rational.
            # Any finite binary64 numerator/denominator needs <1024 chars;
            # ordinary canonical inputs emit only the hexadecimal form.
            text += 1024 if lossless_schedule else 40
        elif type(value) is str:
            if len(value) > 4096:
                raise DenseRunnerError("windowed_metadata_domain", "bounded controlling text required")
            text += len(value)
        elif isinstance(value, enum.Enum):
            text += 512
            visit(value.value, depth + 1, path + 6, lossless_schedule)
        elif isinstance(value, np.generic):
            text += 128
            visit(value.item(), depth + 1, path + 6, lossless_schedule)
        elif isinstance(value, ModuleType):
            if value is not np:
                raise DenseRunnerError("unsupported_backend", "only genuine NumPy namespace")
            text += 512
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            lossless_schedule = lossless_schedule or type(value) is DenseSchedule
            fields = dataclasses.fields(value)
            if (len(fields) > 128 or len(type(value).__module__) + len(type(value).__qualname__) + 1 > 512):
                raise DenseRunnerError("windowed_metadata_domain", "bounded Native holder schema required")
            text += 512  # fully qualified type plus canonical wrapper keys
            if lossless_schedule and type(value) is RainEvent:
                # as_record includes the exact sum of at most4096 interval
                # rationals in addition to each encoded duration/intensity.
                text += 1024
            for field in fields:
                if len(field.name) > 128:
                    raise DenseRunnerError("windowed_metadata_domain", "bounded field name required")
                text += len(field.name)
                visit(getattr(value, field.name), depth + 1, path + len(field.name) + 1, lossless_schedule)
        elif (type(value) is tuple or type(value) is list or type(value) is dict
              or type(value) is set or type(value) is frozenset):
            if len(value) > 4096:
                raise DenseRunnerError("windowed_metadata_domain", "bounded metadata container required")
            comparisons += len(value) * len(value)  # finite sort/comparison work; never a retained-byte debit
            if type(value) is dict:
                for key, item in value.items():
                    if type(key) is not str or len(key) > 4096:
                        raise DenseRunnerError("windowed_metadata_domain", "bounded string metadata keys required")
                    text += len(key)
                    # _canonical_input uses key!r in eager diagnostic paths;
                    # include escaping/encoding before creating those paths.
                    visit(item, depth + 1, path + 12 * len(key) + 4, lossless_schedule)
            else:
                for item in value:
                    visit(item, depth + 1, path + 8, lossless_schedule)
        else:
            raise DenseRunnerError("windowed_metadata_domain", "unknown controlling metadata kind")
        if text > 1 << 20 or payload > 64 << 20:
            raise DenseRunnerError("windowed_metadata_domain", "whole controlling text/array domain exceeded")

    for root in roots:
        visit(root, 0, 32)
    # A canonical occurrence adds at most16 fixed wrapper/value/field/shape
    # nodes at rank<=4. Thirty-two logical reference words per such node
    # cover current/expected/partial/returned metadata and error frontiers.
    # Text includes JSON's worst12-character escaping and eager path strings.
    # The complete held/current receipt capacity is additive, with no GC or
    # same-object credit. Actual sys/interpreter allocation is unclaimed.
    global_refs = len(vars(_dense))
    if global_refs > 512:
        raise DenseRunnerError("windowed_image_domain", "original runner symbol domain exceeded")
    return (("canonical_reference_words", 8 * 32 * (16 * nodes + 64 * arrays + 128)),
            ("canonical_text_and_eager_paths", 12 * text + 2 * paths),
            ("hash_contiguity_and_input_projection_buffers", 2 * payload + 32 * 8 * arrays),
            ("two_source_bound_functions_globals_and_error_slots", 8 * (8 * global_refs + 256) + 8 * 4096),
            ("held_current_encoded_parsed_and_returned_receipts", 6 * (events + 1) * receipt_bytes),
            ("finite_sort_and_projection_visits",
             comparisons * (12 * text + 32 * nodes + 1) + nodes + 12 * text + paths),
            *_own_image_terms(), *_windowed_process.binding_metadata_terms())


def _own_image_terms() -> tuple[tuple[str, int], ...]:
    """Count actual Source-bound code/constant/ref metadata of these two modules.

    No function is invoked and no Source file/provider/array is read. This
    prices the final code (including this helper) rather than a stale literal
    census. It is a logical image/frontier reservation, not a code-object ABI
    or whole-interpreter heap certificate. Root's full immutable Source and
    compiled-origin validation remains an independent prerequisite.
    """
    count = slots = raw = chars = frame_words = 0

    def constant(value: Any) -> None:
        nonlocal count, slots, raw, chars, frame_words
        count += 1
        if count > 131072:
            raise DenseRunnerError("windowed_image_domain", "own code/constant census exceeded")
        slots += 1
        if type(value) is types.CodeType:
            raw += len(value.co_code) + len(value.co_linetable) + len(value.co_exceptiontable)
            frame_words += value.co_nlocals + value.co_stacksize + len(value.co_cellvars) + len(value.co_freevars)
            for item in (value.co_name, value.co_qualname, value.co_filename):
                constant(item)
            for sequence in (value.co_consts, value.co_names, value.co_varnames,
                             value.co_cellvars, value.co_freevars):
                for item in sequence:
                    constant(item)
        elif type(value) is str:
            chars += len(value)
        elif type(value) is bytes:
            raw += len(value)
        elif type(value) is tuple or type(value) is frozenset:
            for item in value:
                constant(item)
        elif not (value is None or value is Ellipsis or type(value) is bool or type(value) is int
                  or type(value) is float or type(value) is complex):
            raise DenseRunnerError("windowed_image_domain", "unknown own compiled constant kind")
        if raw > 4 << 20 or chars > 4 << 20:
            raise DenseRunnerError("windowed_image_domain", "own code/text domain exceeded")

    def inert(value: Any, depth: int = 0) -> None:
        """Only the exact inert metadata emitted by these Source classes.

        The three dataclasses have annotations as strings, no factories,
        empty field metadata, and either MISSING or the one None default.
        Unknown descriptors/holders are not silently treated as free code.
        """
        nonlocal count, slots, chars
        count += 1
        slots += 4
        if count > 131072 or depth > 32:
            raise DenseRunnerError("windowed_image_domain", "own class metadata domain exceeded")
        if value is None or value is dataclasses.MISSING:
            return
        if type(value) is str:
            chars += len(value)
        elif type(value) is bool or type(value) is int:
            if type(value) is int and value.bit_length() > 128:
                raise DenseRunnerError("windowed_image_domain", "own scalar metadata domain exceeded")
        elif type(value) is float:
            if not math.isfinite(value):
                raise DenseRunnerError("windowed_image_domain", "finite own scalar metadata required")
            chars += 40
        elif (type(value) is tuple or type(value) is list or type(value) is dict
              or type(value) is set or type(value) is frozenset):
            if len(value) > 512:
                raise DenseRunnerError("windowed_image_domain", "own metadata container domain exceeded")
            if type(value) is dict:
                for key, item in value.items():
                    if type(key) is not str:
                        raise DenseRunnerError("windowed_image_domain", "own metadata string key required")
                    inert(key, depth + 1)
                    inert(item, depth + 1)
            else:
                for item in value:
                    inert(item, depth + 1)
        elif type(value) is dataclasses.Field:
            policy_field = (vars(_windowed_process.WindowedProductionLimits)["__dataclass_fields__"].get(value.name)
                            is value)
            if (value.default_factory is not dataclasses.MISSING or len(value.metadata) != 0
                    or value._field_type is not dataclasses._FIELD
                    or not (value.default is None or value.default is dataclasses.MISSING
                            or policy_field and type(value.default) in (str, int))):
                raise DenseRunnerError("windowed_image_domain", "unsupported own dataclass field behavior")
            if policy_field:
                inert(value.default, depth + 1)  # new Source policy's eight genuine defaults
            inert(value.name, depth + 1)
            inert(value.type, depth + 1)
            for item in (value.init, value.repr, value.compare, value.hash, value.kw_only):
                inert(item, depth + 1)
        elif type(value) is dataclasses._DataclassParams:
            # Exact ten boolean members of the Source-pinned 3.12 producer.
            for name in ("init", "repr", "eq", "order", "unsafe_hash", "frozen", "match_args",
                         "kw_only", "slots", "weakref_slot"):
                item = getattr(value, name)
                if type(item) is not bool:
                    raise DenseRunnerError("windowed_image_domain", "own dataclass parameter domain exceeded")
                inert(item, depth + 1)
        else:
            raise DenseRunnerError("windowed_image_domain", "unknown own class metadata kind")
        if chars > 4 << 20:
            raise DenseRunnerError("windowed_image_domain", "own text domain exceeded")

    def function(value: Any, depth: int = 0) -> None:
        nonlocal slots, chars
        if type(value) is not types.FunctionType or depth > 32:
            raise DenseRunnerError("windowed_image_domain", "exact own/reused Python method required")
        constant(value.__code__)
        inert(value.__defaults__)
        inert(value.__kwdefaults__)
        inert(value.__annotations__)
        closure = value.__closure__
        if closure is not None:
            if len(closure) > 64:
                raise DenseRunnerError("windowed_image_domain", "own function closure domain exceeded")
            slots += 4 * len(closure)
            for cell in closure:
                item = cell.cell_contents
                if type(item) is types.FunctionType:
                    function(item, depth + 1)  # generated repr wrapper's actual inner function
                elif type(item) is type:
                    # Frozen methods capture their declaring class and error
                    # class. No constructor/method runs; Source owns the code.
                    chars += len(item.__module__) + len(item.__qualname__)
                    slots += 4
                else:
                    inert(item)

    def own_class(value: type, module_name: str, depth: int = 0) -> None:
        nonlocal slots, chars
        namespace = vars(value)
        if depth > 16 or len(namespace) > 512:
            raise DenseRunnerError("windowed_image_domain", "own class symbol domain exceeded")
        slots += 4 * len(namespace)
        for name, attribute in namespace.items():
            if type(name) is not str or len(name) > 512:
                raise DenseRunnerError("windowed_image_domain", "own class name domain exceeded")
            chars += len(name)
            if type(attribute) is types.FunctionType:
                function(attribute)  # includes direct HostOwnership aliases
            elif type(attribute) is staticmethod or type(attribute) is classmethod:
                function(attribute.__func__)
            elif type(attribute) is property:
                for item in (attribute.fget, attribute.fset, attribute.fdel):
                    if item is not None:
                        function(item)
                inert(attribute.__doc__)
            elif type(attribute) is type and attribute.__module__ == module_name:
                own_class(attribute, module_name, depth + 1)
            elif attribute is _production.DenseProductionError:
                # The only external class alias in the owning Source roster.
                # Its original Source/compiled guards are independent.
                slots += 4
                chars += len(attribute.__module__) + len(attribute.__qualname__)
            elif type(attribute) is types.GetSetDescriptorType or type(attribute) is types.MemberDescriptorType:
                if attribute.__objclass__ is not value or attribute.__name__ != name:
                    raise DenseRunnerError("windowed_image_domain", "foreign own-class descriptor")
                slots += 32  # positive logical descriptor/header reference role
            else:
                inert(attribute)

    for namespace in (globals(), vars(_windowed_bed), vars(_windowed_process)):
        if len(namespace) > 512:
            raise DenseRunnerError("windowed_image_domain", "own module symbol domain exceeded")
        slots += 4 * len(namespace)
        module_name = namespace["__name__"]
        for name, value in namespace.items():
            chars += len(name)
            if namespace is vars(_windowed_process) and name in _windowed_process.BINDING_FIELDS:
                # The bound Original code/default/annotation/closure image is
                # walked whole; its owned globals/holders have a separate bank.
                if name == "_BOUND_CHECK":
                    function(value)
                slots += 4
            elif name in ("__builtins__", "__spec__", "__loader__"):
                slots += 32  # original interpreter/import owner, not our payload
            elif type(value) is ModuleType:
                slots += 32  # independent original module/Source guards
            elif type(value) is types.FunctionType:
                if value.__module__ == module_name or value.__globals__ is namespace:
                    function(value)
                else:
                    slots += 32  # explicitly imported original function
            elif value is Any or (name == "annotations" and value is annotations):
                # The exact pre-existing typing/future imports. Their foreign
                # producer owns their contents; only this new reference is ours.
                slots += 32
            elif isinstance(value, type):
                if value.__module__ == module_name:
                    if type(value) is not type:
                        raise DenseRunnerError("windowed_image_domain", "own custom metaclass is unsupported")
                    own_class(value, module_name)
                else:
                    slots += 32  # original imported class, independently guarded
            elif type(value) is types.BuiltinFunctionType:
                if type(value.__self__) is not ModuleType:
                    raise DenseRunnerError("windowed_image_domain", "bound container builtin is unsupported")
                slots += 32
            elif name == "_TOKEN" and type(value) is object:
                slots += 32  # Source-declared sentinel name and exact object kind
            elif name == "_EVIDENCE" and type(value) is _WindowedEvidence:
                slots += 32  # Source-owned zero-slot receiver; its class is walked
            else:
                inert(value)  # including complete lists/tuples/dicts/constants
    if chars > 4 << 20:
        raise DenseRunnerError("windowed_image_domain", "whole own text domain exceeded")
    # Four simultaneously retained code/metadata/error/reference frontiers;
    # raw and Unicode encoding/hash buffers are separate positive roles.
    return (("own_control_code_and_raw_buffers", 4 * raw),
            ("own_control_text_encode_hash_error_buffers", 12 * chars),
            ("own_control_reference_and_frame_words", 4 * 8 * (slots + frame_words + 128)))


# --- native delegates (the only names negative tests may patch) ----------------------------------------------------

def _prepare_record(state: Any, config: Any, record: Any, *, limits: Any, source_identity: str,
                    record_origin_s: float) -> Any:
    from maple.aeolian.scheduler.windowed_record import prepare_windowed_small_feature_wind_record

    return prepare_windowed_small_feature_wind_record(state, config, record, limits=limits,
                                                      source_identity=source_identity,
                                                      record_origin_s=record_origin_s)


def _advance_record(owner: Any) -> Any:
    return owner.advance()


def _commit_entry(state: Any, context: Any, graph: Any, reference: Any, current_time_s: float,
                  routing_source_m: Any, *, control: Any) -> Any:
    from maple_syrup.windowed_shared_bed import commit_positive_windowed_entry

    return commit_positive_windowed_entry(state, context, graph, reference, current_time_s, routing_source_m,
                                          control=control)


def _complete_positive(state: Any, context: Any, column: Any, field: Any, schedule: Any, vegetation: Any,
                       sediment: Any, *, max_end_s: float, control: Any, policy: Any, report_every_s: float) -> Any:
    from maple_syrup.windowed_shared_bed import complete_positive_windowed_event

    return complete_positive_windowed_event(state, context, column, field, schedule, vegetation, sediment,
                                            max_end_s=max_end_s, control=control, policy=policy,
                                            report_every_s=report_every_s)


def _windowed_native() -> SimpleNamespace:
    """Actual windowed native symbols, imported lazily."""
    from maple.aeolian.flux.windowed_step_admission import SmallFeatureWindowedReducedStepLimits
    from maple.aeolian.scheduler.state import SchedulerPhysicsConfig, SchedulerPhysicalState, SchedulerState
    from maple.aeolian.scheduler.accounting import BlockAccounting
    from maple.aeolian.scheduler.record import WindRecordResult
    from maple.aeolian.scheduler.windowed_finalization import finalize_windowed_small_feature_wind_event
    from maple.aeolian.scheduler.windowed_record import (
        WindowedWindRecord, WindowedWindRecordLimits, WindowedWindRecordResult,
    )
    from maple.surface.voxels.windowed_inventory import resident_reduction_bound_by_cell_class_kg
    from maple_syrup import windowed_shared_bed as shared
    from maple_syrup.complete_event import DryHandoff
    from maple_syrup.sediment_bed import BedIntegrationError, reroute_threshold_m

    return SimpleNamespace(**{name: value for name, value in locals().items()})


def _resident_fold_bound(n: SimpleNamespace, w: SimpleNamespace, column: Any, geometry: Any) -> Any:
    """Native per-cell bounds plus Native policy for this actual C-term fold.

    K1 is the closed profile. Each nonnegative bound entry is at most the
    observed maximum; no inventory/capacity/gap substitute sets the scale.
    This is the same Native summation policy, with its actual primitive count.
    """
    values = w.resident_reduction_bound_by_cell_class_kg(column, geometry)
    folded = values.sum(axis=(0, 1))
    rounding = n.summation_error_bound_kg(int(geometry.ny * geometry.nx), float(values.max()))
    return _dense._vec(n.np, folded) + rounding


# --- typed profile -------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class WindowedWindProfile:
    """Explicit opt-in Small wind profile: the resolved Native model, the scheduler config built from it, the
    Native record limits (genuine metadata planning) and the Source identity tag (metadata, not authority)."""

    model_config: Any
    wind_config: Any
    record_limits: Any
    source_identity: str
    calibration: str
    label: str


# --- windowed evidence receiver: genuine process/dry checks, windowed-only projections ------------------------------

def _dense_record_refused(*args: Any, **kwargs: Any) -> Any:
    raise _production.DenseProductionError("dense_record_not_applicable",
                                           "the Dense controlling record is never applied to a windowed subject")


def _windowed_commit_evidence(np: Any, commit: Any) -> dict[str, Any]:
    """Projection of the latest Native windowed commit result (fields actually present only)."""
    if commit is None:
        return {"status": "no_commit", "scope": "latest_native_commit_only"}
    names = [f.name for f in dataclasses.fields(commit)] if dataclasses.is_dataclass(commit) else []
    applied = getattr(commit, "avalanche_applied", None) if "avalanche_applied" in names else None
    return {"status": "committed", "scope": "latest_native_commit_only",
            "result_type": f"{type(commit).__module__}.{type(commit).__qualname__}",
            "avalanche_applied": None if applied is None else bool(applied),
            "avalanche_applied_exposed": "avalanche_applied" in names}


def _windowed_wind_evidence(n: Any, result: Any, final: Any, previous_water: Any) -> dict[str, Any]:
    """The Dense evidence with the windowed record: the dry predicates are the genuine ones; the seventh-field
    baseline is retained by the windowed owner instead of being refused."""
    record = result.record
    _production.require_dry_wind_water(n.np, record.state.physical.water, previous=previous_water)
    _production.require_dry_wind_water(n.np, final.state.physical.water, previous=previous_water)
    accounting = record.aggregate_accounting
    for name in ("water_clamp_added_volume_m3", "water_displaced_volume_m3"):
        value = getattr(accounting, name, None)
        if value is not None and float(value) != 0.0:
            raise _production.DenseProductionError("wind_water_exchange", f"native wind reported nonzero {name}")
    if final.state.topographic_wind_cache is not None or final.state.active_layer_composition_cache is not None:
        raise _production.DenseProductionError("wind_cache_not_invalidated",
                                               "native finalizer did not invalidate topographic/composition caches")
    baseline = final.state.physical.ledger_reconciliation_baseline
    cache = record.state.topographic_wind_cache
    modifier = None if cache is None else cache.modifier
    topography = {"status": "absent"} if modifier is None else {
        "status": "native_modifier", "algorithm_version": modifier.algorithm_version,
        "wind_direction_deg": float(modifier.wind_direction_deg),
        "separation_present": modifier.separation is not None,
        "separation_bubble_cells": (None if modifier.separation is None else
                                    int(n.np.count_nonzero(modifier.separation.bubble_mask))),
        "envelope_role": "wind_diagnostic_never_bed_input"}
    return {"scope": SCOPE, "water_rule": "constant_depth", "dry_water_preserved": True,
            "water_volume_exchanges_m3": {name: None if getattr(accounting, name, None) is None else
                                          float(getattr(accounting, name)) for name in
                                          ("water_clamp_added_volume_m3", "water_displaced_volume_m3")},
            "topographic_wind": topography,
            "transport_launched_mass_by_class_kg": [float(v) for v in accounting.launched_mass_by_class_kg],
            "latest_wind_commit": {"status": "not_retained_by_native_record_owner"},
            "reconciliation_baseline": (None if baseline is None else {
                "owner": "windowed_bed_seventh_field",
                "last_reconciliation_time_s": encode_time(float(baseline.last_reconciliation_time_s)),
                "reconciliation_count": int(baseline.reconciliation_count)}),
            "creep_evidence": "signed_domain_net_is_not_a_gross_activity_witness"}


class _WindowedEvidence:
    """Closed receiver bound as the runner's `_production`. Process checks, dry predicates and retained-owner
    diagnostics are the genuine dense_production functions; only the incompatible Dense projections differ."""

    __slots__ = ()
    SCOPE = SCOPE
    DenseProductionError = _production.DenseProductionError
    check_process = staticmethod(_windowed_process.check_process)
    require_dry_wind_water = staticmethod(_production.require_dry_wind_water)
    retained_wrapper_owners = staticmethod(_production.retained_wrapper_owners)
    commit_evidence = staticmethod(_windowed_commit_evidence)
    wind_evidence = staticmethod(_windowed_wind_evidence)
    controlling_record = staticmethod(_dense_record_refused)


_EVIDENCE = _WindowedEvidence()


# --- private ownership protocol ---------------------------------------------------------------------------------------

class WindowedOwnership:
    """Ownership of the runner's accepted `WindowedBed` boundary under the complete `_HostOwnership` protocol.
    Bed-independent bodies (private labels, pool/soil copies, retained routing template/entry/guards, prepared
    publication carriers) ARE the original host functions; bed-dependent ones go through the WindowedBed."""

    __slots__ = ("_bed",)
    receipt_schema = RECEIPT_SCHEMA

    def __init__(self, bed: WindowedBed) -> None:
        if type(bed) is not WindowedBed:
            raise DenseRunnerError("invalid_input", "windowed ownership requires a WindowedBed")
        self._bed = bed

    private_label = _HostOwnership.private_label
    routing_label = _HostOwnership.routing_label
    pool_bytes = _HostOwnership.pool_bytes
    work_pool = _HostOwnership.work_pool
    work_soil = _HostOwnership.work_soil
    memory_names = _HostOwnership.memory_names
    memory_bytes = _HostOwnership.memory_bytes
    work_memory = _HostOwnership.work_memory
    retain_routing = _HostOwnership.retain_routing
    routing_memory = _HostOwnership.routing_memory
    routing_guard = _HostOwnership.routing_guard
    routing_entry = _HostOwnership.routing_entry
    prepare_scheduler = _HostOwnership.prepare_scheduler
    prepare_bed = _HostOwnership.prepare_bed
    retained_private = _HostOwnership.retained_private

    def accepted_state(self) -> Any:
        return self._bed.state

    def accepted_inventory(self, ops: Any, n: SimpleNamespace) -> Any:
        return ops.inventory(n, self._bed.state)

    def verify_case_inputs(self) -> None:
        self._bed.verify_case_inputs()

    def accepted_label(self) -> str:
        return self._bed.identity_sha256()

    def begin(self) -> Any:
        return self._bed.stage_copy()

    def stage_state(self, stage: Any) -> Any:
        return stage.state

    def stage_label(self, stage: Any) -> str:
        return stage.identity_sha256

    def candidate_label(self, candidate: Any, time_s: float) -> str:
        return self._bed.candidate_identity_sha256(candidate, time_s)

    def prepare_dynamic(self, *args: Any, **kwargs: Any) -> Any:
        raise DenseBedError("invalid_vegetation_publication", "dynamic vegetation is outside the windowed profile")

    def verify_prepared_dynamic(self, prepared: Any) -> Any:
        raise DenseBedError("invalid_vegetation_publication", "dynamic vegetation is outside the windowed profile")

    def publish_prepared_dynamic(self, prepared: Any) -> Any:
        raise DenseBedError("invalid_vegetation_publication", "dynamic vegetation is outside the windowed profile")

    def stage_bytes(self) -> int:
        return self._bed.stage_cost()["total"]

    def scheduler(self, stage: Any, pool: Any, time_s: float) -> Any:
        return self._bed.stage_scheduler_state(stage, pool, time_s=time_s)

    def commit(self, prepared: Any) -> None:
        if type(prepared) is _SchedulerPublication:
            self._bed.adopt_scheduler_physical(prepared.physical)
        elif type(prepared) is _BedPublication:
            self._bed.adopt_bed_state(prepared.state, time_s=prepared.time_s)
        else:
            raise TypeError(f"not a prepared publication: {type(prepared).__name__}")

    def production_require_dry(self, production: Any, np: Any) -> None:
        production.require_dry_wind_water(np, self._bed.state.water)

    def production_wind_evidence(self, production: Any, n: SimpleNamespace, result: Any,
                                 final: Any) -> dict[str, Any]:
        return production.wind_evidence(n, result, final, self._bed.state.water)

    def production_retained_owners(self, production: Any, wind: Any, water: Any, control: Any,
                                   profile: Any, pool: Any, soil: Any, routing: Any,
                                   n_receipts: int) -> dict[str, Any]:
        return production.retained_wrapper_owners(self._bed, wind, water, control, profile, pool, soil, routing,
                                                  n_receipts)


def _protocol_names(cls: type) -> set[str]:
    return {name for name, value in vars(cls).items()
            if not name.startswith("__") and (callable(value) or name == "receipt_schema")}


def _require_complete_protocol() -> None:
    missing = sorted(_protocol_names(_HostOwnership) - set(dir(WindowedOwnership)))
    if missing:
        raise DenseRunnerError("ownership_protocol_incomplete", f"WindowedOwnership lacks {missing}")


# --- source-bound original rain functions -----------------------------------------------------------------------------

def _source_bound(function: Any, **replacements: Any) -> Any:
    """The ORIGINAL code object over a copy of its module globals in which only `replacements` differ. Each
    replaced name must already be a module global the original code actually loads."""
    namespace = dict(vars(_dense))
    names = set(function.__code__.co_names)
    wrong = sorted(name for name in replacements if name not in names or name not in namespace)
    if wrong or function.__globals__ is not vars(_dense):
        raise DenseRunnerError("source_binding_changed", f"{function.__name__} does not load {wrong}")
    namespace.update(replacements)
    bound = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__,
                               function.__closure__)
    bound.__kwdefaults__ = None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    return bound


def _summary_record(w: SimpleNamespace, summary: Any, role: str, *, evidence: Any = None,
                    elapsed: str | None = None) -> dict[str, int]:
    if type(summary) is not w.shared.PositiveWaterAdmissionSummary:
        raise _Stop("incomplete", "water_admission_summary_missing", f"{role} returned no actual admission summary",
                    evidence=evidence, elapsed=elapsed)
    names = ("gate_count", "logical_work_debited", "logical_read_bytes_debited",
             "versions_reserved", "maximum_predicate_bytes")
    if tuple(f.name for f in dataclasses.fields(summary)) != names or set(vars(summary)) != set(names):
        raise _Stop("incomplete", "water_admission_summary_invalid", f"{role} summary schema changed",
                    evidence=evidence, elapsed=elapsed)
    record = {f.name: getattr(summary, f.name) for f in dataclasses.fields(summary)}
    if any(type(value) is not int or value < 0 for value in record.values()) or record["gate_count"] < 1:
        raise _Stop("incomplete", "water_admission_summary_invalid", f"{role} summary is not plain admitted counters",
                    evidence=evidence, elapsed=elapsed)
    return record


class _WaterLane:
    """Per-rain-event adapters: only the commit and event delegates of the original functions change."""

    __slots__ = ("runner", "w", "entry", "event")

    def __init__(self, runner: WindowedRunner, w: SimpleNamespace) -> None:
        self.runner, self.w = runner, w
        self.entry = self.event = None

    def _profile(self, state: Any, context: Any) -> None:
        runner, w = self.runner, self.w
        if (type(state) is not w.shared.WindowedSharedBedState or context is not runner._bed.context
                or type(context.bridge_limits) is not w.shared.PositiveWindowedWaterBridgeLimits
                or runner._limits.max_report_rows != REPORT_ROWS
                or context.bridge_limits.max_report_rows != REPORT_ROWS):
            raise _Stop("refused", "windowed_water_profile", "exact seven-field bed, held context and the original "
                        "4096 report declaration (MASY and SYRUP) are required before entry")

    def native(self) -> SimpleNamespace:
        values = dict(vars(_dense._native()))
        values["commit_bed"] = self.commit_bed
        return SimpleNamespace(**values)

    def commit_bed(self, state: Any, context: Any, graph: Any, reference: Any, time_s: float, *, force: bool,
                   routing_source_m: Any) -> Any:
        self._profile(state, context)
        if force is not True or self.entry is not None:
            raise _Stop("refused", "windowed_entry_commit", "exactly one forced entry commit per rain event")
        result, summary = _commit_entry(state, context, graph, reference, time_s, routing_source_m,
                                        control=self.runner._control.sediment_control)
        self.entry = {}  # sticky invocation marker, also when actual summary validation refuses
        self.entry = _summary_record(self.w, summary, "entry commit", evidence=result)
        return result

    def complete(self, state: Any, context: Any, column: Any, field: Any, schedule: Any, vegetation: Any,
                 sediment: Any, *, max_end_s: float, control: Any, policy: Any, report_every_s: float,
                 max_report_rows: int, **extra: Any) -> Any:
        self._profile(state.bed, context)
        if extra or max_report_rows != REPORT_ROWS or self.entry is None or self.event is not None:
            raise _Stop("refused", "windowed_event_profile", "one callback-free positive event after the entry "
                        "commit, with the original 4096 report declaration")
        wet, summary = _complete_positive(state, context, column, field, schedule, vegetation, sediment,
                                          max_end_s=max_end_s, control=control, policy=policy,
                                          report_every_s=report_every_s)
        self.event = {}  # never admit a second call after a returned native result
        elapsed = (_elapsed(float(state.t_s), float(wet.dry_state.t_s))
                   if type(wet) is self.w.DryHandoff else None)
        self.event = _summary_record(self.w, summary, "event", evidence=wet, elapsed=elapsed)
        return wet


# --- the runner -------------------------------------------------------------------------------------------------------

class WindowedRunner(DenseRunner):
    """The DenseRunner engine over ONE `WindowedBed`. Build only with `build_windowed_runner`."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("WindowedRunner is closed")

    def __init__(self, token: object = None, **fields: Any) -> None:
        if token is not _TOKEN:
            raise DenseRunnerError("private_constructor", "construct a WindowedRunner with build_windowed_runner")
        for name, value in fields.items():
            setattr(self, name, value)

    @classmethod
    def _from_passive_boundary_load(cls, loaded: Any) -> Any:
        raise DenseRunnerError("restart_unsupported", "no windowed checkpoint/restart codec exists")

    @classmethod
    def _from_passive_boundary_load_v2(cls, loaded: Any) -> Any:
        raise DenseRunnerError("restart_unsupported", "no windowed checkpoint/restart codec exists")

    # private operation hooks -----------------------------------------------------------------------------------
    def _ownership(self) -> Any:
        return WindowedOwnership(self._bed)

    def _rain_receipt_extra(self) -> dict[str, Any]:
        return {"windowed_water": self._water_record}

    def _metadata_reservation(self, n: SimpleNamespace) -> tuple[tuple[str, int], ...]:
        roots = (self._bed._case_record, self._bed.limits, self._bed.context,
                 self._bed.vegetation, self._bed.threshold_modifier_state,
                 self._wind.config, self._water, self._control, self._wind_profile.model_config,
                 self._wind_profile.record_limits, self._schedule, self._limits, self._production_limits,
                 (self._wind_profile.calibration, self._wind_profile.label, self._wind_profile.source_identity))
        return _metadata_terms(n.np, roots, self._limits.max_receipt_bytes, len(self._schedule.events))

    def _reserve_metadata(self, n: SimpleNamespace, estimate: int = 0) -> tuple[tuple[str, int], ...]:
        terms = self._metadata_reservation(n)
        # Logical work is a separate finite counter, never converted into a
        # retained-byte debit. Known reference/text/input/image/receipt roles
        # are additional to the original array stage/report estimate.
        amount = sum(value for name, value in terms if name != "finite_sort_and_projection_visits")
        DenseRunner._work_bytes_check(self, estimate + amount)
        return terms

    def _work_bytes_check(self, estimate: int) -> None:
        self._reserve_metadata(_dense._native(), estimate)

    def _static_parts_record(self, n: SimpleNamespace) -> dict[str, Any]:
        """Every controlling static input (recomputed live before each event and publication)."""
        self._reserve_metadata(n)  # before Native rebuild / canonical records / input hash buffers
        np, w, c = n.np, self._water, self._control
        record = {
            "wind_config": _canonical_input(np, self._wind.config, "wind.config"),
            "terrain": _canonical_input(np, w.terrain, "water.terrain"),
            "column": _canonical_input(np, w.column, "water.column"),
            "field": _canonical_input(np, w.field, "water.field"),
            "vegetation_cover_fraction": _canonical_input(np, w.vegetation_cover_fraction, "water.vegetation"),
            "sediment": _canonical_input(np, w.sediment, "water.sediment"),
            "sediment_control": _canonical_input(np, c.sediment_control, "control.sediment_control"),
            "completion_policy": _canonical_input(np, c.completion_policy, "control.completion_policy"),
            "report_every_s": encode_time(c.report_every_s),
            "runner_limits": self._limits.as_record(), "bed_limits": self._bed.limits.as_record(),
            "schedule": self._schedule.as_record(),
            "windowed_profile": _profile_record(n, _windowed_native(), self._bed, self._wind, self._schedule,
                                                self._wind_profile, self._production_limits),
        }
        return record

    def report(self) -> dict[str, Any]:
        report = super().report()
        report["schema"] = REPORT_SCHEMA
        return report

    # wind ------------------------------------------------------------------------------------------------------
    def _wind_event(self, index: int, event: WindEvent, resolved: dict[str, Any]) -> dict[str, Any]:
        n, w = _dense._native(), _windowed_native()
        np, bed, cfg, profile = n.np, self._bed, self._wind.config, self._wind_profile
        ops, own = self._array_ops(np), self._ownership()
        t = bed.time_s
        end = decode_time(resolved["resolved_edges"]["edges"][-1]["native"])
        quotient = Fraction(event.duration_s) / Fraction(cfg.outer_timestep_s)
        upper = -(-quotient.numerator // quotient.denominator) + 1
        if upper > self._limits.max_wind_accounting_steps:
            raise _Stop("refused", "wind_step_limit", f"up to {upper} native accounting steps > "
                        f"max_wind_accounting_steps={self._limits.max_wind_accounting_steps}")
        plan = n.accounting_step_durations_s(event.duration_s, cfg.outer_timestep_s)
        requested = Fraction(event.duration_s)
        covered = sum((Fraction(dt) for dt in plan), Fraction(0))
        rounding = Fraction(2, 2 ** 53) * max(requested, covered)
        if len(plan) > upper or covered <= 0 or abs(covered - requested) > rounding:
            raise _Stop("refused", "unsupported_wind_duration",
                        f"native accounting steps ({len(plan)}) cover {float(covered)!r} s of the requested "
                        f"{event.duration_s!r} s; the native split drops or extends scheduled time")
        stage_bytes = own.stage_bytes()
        pool_bytes = own.pool_bytes(self._pool)
        # Wrapper-owned roles only: one priced resident stage, the work pool and the returned empty pool, two
        # inventory/fold class vectors and the accounting plan. The Native record admits its own block,
        # finalizer, publication and boundary roles before any transport.
        cells = int(cfg.geometry.ny * cfg.geometry.nx)
        # Two independent Native bound calls: four C*K float arrays in each
        # native maximum/minimum/scale/product chain, fold and returned K
        # vectors, plus positive logical array/header/error reference roles.
        fold_roles = 2 * (4 * 8 * cells * self._n_classes + 4 * 8 * self._n_classes + 8 * 32 * 8)
        self._work_bytes_check(stage_bytes + 2 * pool_bytes + 8 * self._inventory_elements * self._n_classes
                               + 32 * upper + fold_roles)
        try:
            own.production_require_dry(self._production, np)
        except self._production.DenseProductionError as exc:
            raise _Stop("refused", exc.code, exc.detail, exception=exc) from exc
        stage = self._stage()
        pool = own.work_pool(self._pool)
        try:
            scheduler = own.scheduler(stage, pool, t)
        except DenseBedError as exc:
            raise _Stop("refused", exc.code, exc.detail, exception=exc) from exc
        record = n.resolve_wind_forcing_record({"blocks": [{
            "shear_velocity_m_s": event.shear_velocity_m_s,
            "direction_deg": event.direction_deg_cw_from_north, "duration_s": event.duration_s}]})
        staged = own.stage_state(stage)
        inventory_before = ops.inventory(n, staged)
        fold_before = _resident_fold_bound(n, w, staged.voxel_column, cfg.geometry)
        try:
            owner = _prepare_record(scheduler, cfg, record, limits=profile.record_limits,
                                    source_identity=profile.source_identity, record_origin_s=t)
        except Exception as exc:  # noqa: BLE001 - refused before any Native block
            status = "native_refused" if isinstance(exc, ValueError) else "native_error"
            raise _Stop(status, "wind_record_refused", _detail(exc), exception=exc) from exc
        evidence = {"record_owner": owner}
        try:
            result = _advance_record(owner)
        except Exception as exc:  # noqa: BLE001 - the owner keeps its historical/unaccepted Native evidence
            status = "native_refused" if isinstance(exc, ValueError) else "native_error"
            raise _Stop(status, "wind_native_failure", _detail(exc), exception=exc, evidence=evidence) from exc
        evidence["result"] = result
        return self._wind_accept_record(n, w, index, event, resolved, SimpleNamespace(
            t=t, end=end, upper=upper, plan=plan, requested=requested, covered=covered, rounding=rounding,
            stage=stage, record=record, owner=owner, inventory_before=inventory_before,
            fold_before=fold_before), result, evidence)

    def _wind_accept_record(self, n: SimpleNamespace, w: SimpleNamespace, index: int, event: WindEvent,
                            resolved: dict[str, Any], entry: SimpleNamespace, result: Any,
                            evidence: dict[str, Any]) -> dict[str, Any]:
        np, cfg, nc = n.np, self._wind.config, self._n_classes
        ops, own = self._array_ops(np), self._ownership()
        t, owner, record = entry.t, entry.owner, entry.record
        if (type(owner) is not w.WindowedWindRecord or type(result) is not w.WindowedWindRecordResult
                or owner.result is not result):
            raise _Stop("incomplete", "invalid_wind_record_result", "unexpected native record owner/result association",
                        evidence=evidence)
        rec, final = result.record, result.finalization
        if (type(rec) is not w.WindRecordResult or type(final) is not n.WindEventFinalizationResult
                or type(rec.state) is not w.SchedulerState or type(final.state) is not w.SchedulerState
                or type(rec.state.physical) is not w.SchedulerPhysicalState
                or type(final.state.physical) is not w.SchedulerPhysicalState
                or type(rec.aggregate_accounting) is not w.BlockAccounting):
            raise _Stop("incomplete", "invalid_wind_record_result", "unexpected nested Native result type",
                        evidence=evidence)
        before = rec.state
        t_end = ops.scalar(before.physical.time_s)
        elapsed = _elapsed(t, t_end)

        def incomplete(code: str, detail: str) -> _Stop:
            return _Stop("incomplete", code, detail, elapsed=elapsed, evidence=evidence)

        if (owner.status != "completed" or owner.block_count != 1 or owner.finalizer_call_count != 1
                or owner.current_state is not final.state or rec.n_blocks != 1 or rec.block_results != ()
                or rec.block_results_collected is not False):
            raise incomplete("wind_record_incomplete", f"native record owner status {owner.status!r}")
        if rec.all_blocks_conserved is not True:
            raise incomplete("wind_conservation_failed", "native block conservation reported failure")
        a = rec.aggregate_accounting
        try:
            water_fields = [getattr(a, f"water_{name}") for name in
                            ("mobile_mass_before_by_class_kg", "mobile_mass_after_by_class_kg",
                             "boundary_input_by_class_kg", "boundary_export_by_class_kg")]
        except AttributeError as exc:
            raise incomplete("wind_record_incomplete", "Native water accounting field is absent") from exc
        if any(v is not None and ops.any_nonzero(ops.class_vector(v)) for v in water_fields):
            raise incomplete("wind_water_pass", "native wind accounting reports water-borne exchange")
        if a.n_accounting_steps > entry.upper or a.n_accounting_steps > self._limits.max_wind_accounting_steps:
            raise incomplete("wind_step_limit", f"{a.n_accounting_steps} native accounting steps")
        if ops.any_nonzero(before.physical.water.mobile_mass_by_cell_class_kg):
            raise incomplete("water_mobile_not_empty", "water-borne load appeared during wind")
        if t_end < t:
            raise incomplete("time_reversed", "native wind end precedes its start")
        if t_end == t:
            raise incomplete("wind_no_elapsed", "positive requested duration but native physical time did not advance")
        # Repeat the genuine Native completed-record end-time condition; no
        # new clock epsilon and no rebasing/dropped elapsed time.
        if abs(t_end - entry.end) > n.record_end_epsilon_s:
            raise incomplete("wind_record_clock", "physical end is outside the Native record end bound")
        if (type(rec.total_duration_s) is not float
                or rec.total_duration_s.hex() != (t_end - t).hex()):
            raise incomplete("wind_record_clock", "Native returned elapsed disagrees with unchanged endpoints")
        if before.block_index != len(record.blocks) or before.block_offset_s != 0.0:
            raise incomplete("wind_record_incomplete", "native record did not reach its completed cursor")
        if type(final) is not n.WindEventFinalizationResult:
            raise incomplete("invalid_wind_finalization", "unexpected native finalizer result type")
        if not final.completed or not final.water_handoff_ready:
            raise incomplete("wind_finalization_incomplete", str(final.reasons)[:_dense._DETAIL_CHARS])
        settled = final.state
        if (ops.scalar(settled.physical.time_s).hex() != t_end.hex() or ops.scalar(final.time_s).hex() != t_end.hex()
                or settled.block_index != before.block_index or settled.block_offset_s != 0.0
                or final.block_index != before.block_index or final.block_offset_s != 0.0):
            raise incomplete("wind_finalization_clock_changed", "native finalizer changed time or cursor")
        if not n.mobile_state_is_genuinely_empty(settled.physical.mobile_state):
            raise incomplete("wind_finalization_mobile_nonempty", "native finalizer left airborne material")
        final_vectors = {name: ops.class_vector(getattr(final, name)) for name in (
            "airborne_mass_before_by_class_kg", "airborne_mass_after_by_class_kg",
            "deposited_mass_by_class_kg", "boundary_export_mass_by_class_kg", "bed_change_by_class_kg",
            "transfer_identity_gap_by_class_kg", "reservoir_identity_gap_by_class_kg")}
        if any(v.shape != (nc,) or not bool(np.isfinite(v).all()) for v in final_vectors.values()):
            raise incomplete("wind_finalization_budget_invalid", "nonfinite or incompatible class vector")
        for prefix in ("transfer", "reservoir"):
            bound = ops.scalar(getattr(final, prefix + "_identity_bound_kg"))
            if (not math.isfinite(bound) or bound < 0.0
                    or not bool(np.all(np.abs(final_vectors[prefix + "_identity_gap_by_class_kg"]) <= bound))):
                raise incomplete("wind_finalization_budget_open", "native finalizer class budget failed")
        if (bool(np.any(final_vectors["airborne_mass_after_by_class_kg"] != 0.0))
                or not ops.equal_class_vector(final_vectors["airborne_mass_before_by_class_kg"],
                                              n.mobile_mass_by_class_kg(before.physical.mobile_state, nc))):
            raise incomplete("wind_finalization_mobile_budget", "native mobile inventory disagrees")

        # Native block reservoir identity over the record's one aggregate (operands as the native scale defines).
        transport_bed_change = ops.class_vector(a.bed_change_by_class_kg)
        mobile_before = ops.class_vector(a.mobile_mass_before_by_class_kg)
        mobile_after = ops.class_vector(a.mobile_mass_after_by_class_kg)
        if not ops.equal_class_vector(mobile_after, final_vectors["airborne_mass_before_by_class_kg"]):
            raise incomplete("wind_finalization_mobile_budget", "transport/finalizer mobile handoff differs")
        inflow, transport_export = (ops.class_vector(a.boundary_input_by_class_kg),
                                    ops.class_vector(a.boundary_export_by_class_kg))
        inventory_before = entry.inventory_before
        native_residual = transport_bed_change + (mobile_after - mobile_before) - inflow + transport_export
        # Genuine completed Native owner: its receiver selects rec.state.physical
        # and aggregate accounting BEFORE the one finalizer, not settled.physical.
        # Native owns its fresh boundary/image/resource admission and exact formula.
        native_scale = owner.reservoir_arithmetic_scale_kg()
        native_bound = ops.scalar(n.summation_error_bound_kg(max(a.n_accounting_steps, 1), native_scale))
        if not bool(np.all(np.abs(native_residual) <= native_bound)):
            raise incomplete("wind_budget_open", "native per-class reservoir identity exceeds its native bound")
        export = transport_export + final_vectors["boundary_export_mass_by_class_kg"]
        bed_change = transport_bed_change + final_vectors["bed_change_by_class_kg"]
        physical = settled.physical
        candidate = w.shared.WindowedSharedBedState(
            voxel_column=physical.voxel_column, active_layer=physical.active_layer, water=physical.water,
            ledger=physical.ledger, committed_topography=physical.committed_topography,
            sediment_availability=physical.sediment_availability,
            reconciliation_baseline=physical.ledger_reconciliation_baseline)
        inventory_after = ops.inventory(n, candidate)
        fold_after = _resident_fold_bound(n, w, physical.voxel_column, cfg.geometry)
        event_mobile_delta = final_vectors["airborne_mass_after_by_class_kg"] - mobile_before
        event_residual = bed_change + event_mobile_delta - inflow + export
        event_scale = sum(float(np.abs(value).sum()) for value in (
            transport_bed_change, final_vectors["bed_change_by_class_kg"],
            final_vectors["airborne_mass_after_by_class_kg"], mobile_before,
            inflow, transport_export, final_vectors["boundary_export_mass_by_class_kg"]))
        reservoir_bound = ops.scalar(final.reservoir_identity_bound_kg)
        event_bound = native_bound + reservoir_bound + n.summation_error_bound_kg(6, event_scale)
        if not bool(np.all(np.abs(event_residual) <= event_bound)):
            raise incomplete("wind_event_budget_open", "combined transport/finalizer class identity failed")
        inventory_residual = inventory_after - inventory_before + event_mobile_delta - inflow + export
        scale = max(float(np.abs(inventory_before).sum()), float(np.abs(inventory_after).sum()),
                    float(inflow.sum()), float(export.sum()))
        # The streamed windowed inventories are Native folds; their own resident reduction bounds are added.
        inventory_bound = (native_bound + reservoir_bound + entry.fold_before + fold_after
                           + n.summation_error_bound_kg(2 * self._inventory_elements + 6, scale))
        if not bool(np.all(np.abs(inventory_residual) <= inventory_bound)):
            raise incomplete("wind_inventory_open", "actual bed inventories do not close within native bounds")

        baseline = physical.ledger_reconciliation_baseline
        settlement = {"completed": "native_completed_settlement",
                      "already_empty": "native_already_empty_no_settlement_claimed"}.get(final.status, final.status)
        body = {
            "native": {
                "record": {"api": "maple.aeolian.scheduler.windowed_record."
                                  "prepare_windowed_small_feature_wind_record",
                           "prepare_calls": 1, "advance_calls": 1, "owner_status": owner.status,
                           "n_blocks": int(rec.n_blocks), "finalizer_calls": int(owner.finalizer_call_count),
                           "record_origin_s": encode_time(t),
                           "total_duration_s": encode_time(ops.scalar(rec.total_duration_s)),
                           "all_blocks_conserved": True},
                "resolved_end_s": encode_time(entry.end), "physical_time_s": encode_time(t_end),
                "n_accounting_steps": int(a.n_accounting_steps), "accounting_step_upper_bound": entry.upper,
            },
            "coverage": {"requested_s": encode_time(event.duration_s), "native_plan_steps": len(entry.plan),
                         "native_plan_sum_s": _fraction_text(entry.covered),
                         "plan_discrepancy_s": _fraction_text(entry.covered - entry.requested),
                         "plan_rounding_bound_s": _fraction_text(entry.rounding),
                         "elapsed_discrepancy_s": _fraction_text(Fraction(t_end) - Fraction(t) - entry.requested)},
            "classes": {
                "bed_change_kg": _floats(np, bed_change), "boundary_input_kg": _floats(np, inflow),
                "boundary_export_kg": _floats(np, export),
                "mobile_before_kg": _floats(np, mobile_before),
                "mobile_after_kg": _floats(np, final_vectors["airborne_mass_after_by_class_kg"]),
                "transport_mobile_after_kg": _floats(np, mobile_after),
                "transport_boundary_export_kg": _floats(np, transport_export),
                "transport_bed_change_kg": _floats(np, transport_bed_change),
                "native_residual_kg": _floats(np, native_residual), "native_bound_kg": float(native_bound),
                "event_residual_kg": _floats(np, event_residual), "event_bound_kg": float(event_bound),
                "event_summation_terms": 6, "event_arithmetic_scale_kg": event_scale,
                "inventory_before_kg": _floats(np, inventory_before),
                "inventory_after_kg": _floats(np, inventory_after),
                "inventory_residual_kg": _floats(np, inventory_residual),
                "inventory_bound_kg": _floats(np, inventory_bound),
                "resident_fold_bound_before_kg": _floats(np, entry.fold_before),
                "resident_fold_bound_after_kg": _floats(np, fold_after),
            },
            "wind_end_pool": "genuinely_empty",
            "finalization": {
                "status": final.status, "settlement_claim": settlement,
                "algorithm_version": final.algorithm_version, "representation": final.representation,
                "placement_rule": final.placement_rule, "record_origin_s": encode_time(t),
                "record": _canonical_input(np, record, "completed_record"),
                "record_sha256": _bed._sha256(_canonical_input(np, record, "completed_record")),
                "completed_block_index": settled.block_index, "completed_block_offset_s": settled.block_offset_s,
                "physical_time_s": encode_time(ops.scalar(physical.time_s)),
                "n_settled": int(final.n_settled), "n_exported": int(final.n_exported),
                **{name: _floats(np, value) for name, value in final_vectors.items()},
                "transfer_identity_bound_kg": ops.scalar(final.transfer_identity_bound_kg),
                "reservoir_identity_bound_kg": reservoir_bound,
                "numerical_residual_scalar_kg": ops.scalar(final.numerical_residual_scalar_kg),
            },
            "reconciliation_baseline": None if baseline is None else {
                "last_reconciliation_time_s": encode_time(float(baseline.last_reconciliation_time_s)),
                "reconciliation_count": int(baseline.reconciliation_count)},
            "stage": {"nbytes": entry.stage.nbytes, "pending_records_in": entry.stage.pending_records_in,
                      "pending_records_out": entry.stage.pending_records_out},
        }
        try:
            body["production"] = own.production_wind_evidence(self._production, n, result, final)
            body["production"]["worker"] = self._production_process_check(
                checkpoint="after_native_wind", elapsed=elapsed, evidence=evidence)
            body["production"]["retained_wrapper_owners"] = own.production_retained_owners(
                self._production, self._wind, self._water, self._control, self._wind_profile,
                self._pool, self._soil, self._routing, len(self._receipts))
        except _Stop:
            raise
        except MemoryError as exc:
            raise _Stop("runner_error", "address_space_exhausted", _detail(exc), exception=exc,
                        elapsed=elapsed, evidence=evidence) from exc
        except self._production.DenseProductionError as exc:
            raise _Stop("incomplete", exc.code, exc.detail, exception=exc, elapsed=elapsed,
                        evidence=evidence) from exc
        except Exception as exc:
            raise _Stop("runner_error", "production_evidence_failure", _detail(exc), exception=exc,
                        elapsed=elapsed, evidence=evidence) from exc
        publish = own.prepare_scheduler(physical)
        return self._finish(n, index, "wind", resolved, entry.stage, candidate, t, t_end,
                            ops.scalar(rec.total_duration_s), inflow, export, inventory_bound, body, publish,
                            pool=physical.mobile_state, soil=self._soil, routing=self._routing,
                            terminal=final_vectors["deposited_mass_by_class_kg"], elapsed=elapsed,
                            evidence=evidence, publication_source=physical)

    # rain ------------------------------------------------------------------------------------------------------
    def _rain_event(self, index: int, event: Any, resolved: dict[str, Any]) -> dict[str, Any]:
        self._reserve_metadata(_dense._native())  # before both copied globals/functions and lane headers
        w = _windowed_native()
        lane = _WaterLane(self, w)
        self._water_record = None
        enter = _source_bound(_dense._rain_enter, _native=lane.native)
        native = _source_bound(_dense._rain_native, _complete_event=lane.complete)
        prepared = enter(self, index, event, resolved)
        wet = native(self, prepared)
        self._water_record = self._water_ceilings(prepared, lane, wet)
        try:
            return _dense._rain_accept(self, index, event, resolved, prepared, wet)
        except BaseException:
            # Only an adopted event may leave an accepted admission record.
            # The real wet/Native evidence remains on the original stop path.
            self._water_record = None
            raise

    def _water_ceilings(self, prepared: Any, lane: _WaterLane, wet: Any) -> dict[str, Any]:
        """Aggregate the actual entry and event admission counters; refuse adoption above the selected
        cumulative ceilings (persistent store/run limits are the SYRUP owners' own, unchanged)."""
        n = prepared.n
        elapsed = _elapsed(prepared.t, float(wet.dry_state.t_s)) if type(wet) is n.DryHandoff else None
        if lane.entry is None or lane.event is None:
            raise _Stop("incomplete", "water_admission_missing", "entry or event admission summary missing",
                        elapsed=elapsed, evidence=wet)
        limits = self._bed.context.bridge_limits
        names = (("logical_work_debited", "max_work"), ("logical_read_bytes_debited", "max_logical_read_bytes"),
                 ("versions_reserved", "max_new_versions"))
        aggregate = {name: lane.entry[name] + lane.event[name] for name, _ in names}
        over = [name for name, ceiling in names if aggregate[name] > getattr(limits, ceiling)]
        if over:
            raise _Stop("refused", "water_cumulative_ceiling", f"entry+event {over} exceed the selected ceilings",
                        elapsed=elapsed, evidence=wet)
        return {"entry_commit_api": "maple_syrup.windowed_shared_bed.commit_positive_windowed_entry",
                "event_api": "maple_syrup.windowed_shared_bed.complete_positive_windowed_event",
                "entry_admission": dict(lane.entry), "event_admission": dict(lane.event),
                "aggregate": {**aggregate, "maximum_predicate_bytes": max(lane.entry["maximum_predicate_bytes"],
                                                                          lane.event["maximum_predicate_bytes"]),
                              "ceilings": {ceiling: getattr(limits, ceiling) for _, ceiling in names}},
                "report_rows_declared": REPORT_ROWS}


# --- closed factory ---------------------------------------------------------------------------------------------------

def _hex64(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _profile_record(n: SimpleNamespace, w: SimpleNamespace, bed: WindowedBed, wind: DenseWindInputs,
                    schedule: DenseSchedule, profile: Any, limits: Any) -> dict[str, Any]:
    """The windowed controlling record: exact shared objects and the Small profile guards (no Dense subject)."""
    ctx, cfg = bed.context, wind.config
    if type(profile) is not WindowedWindProfile or profile.wind_config is not cfg:
        raise DenseRunnerError("invalid_wind_profile", "an explicit WindowedWindProfile over wind.config is required")
    if type(cfg) is not w.SchedulerPhysicsConfig:
        raise DenseRunnerError("invalid_wind_config", "exact native SchedulerPhysicsConfig required")
    # Reuse the genuine existing central model validation/native builder audit.
    # This API validates a scheduler, not a Dense bed: the incompatible Dense
    # controlling_record / six-field bed projection is never invoked.
    from masy.dense_wind_profile import (
        DenseProductionWindProfile, ProductionProfileError, audit_dense_production_wind_config,
    )
    try:
        envelope, schedule_binding = _production.schedule_envelope(schedule)
        audit = audit_dense_production_wind_config(
            DenseProductionWindProfile(profile.model_config, profile.calibration, profile.label), cfg,
            wind_envelope_record=envelope, initial_available_fraction=ctx.initial_available_fraction)
    except ProductionProfileError as exc:
        raise DenseRunnerError(exc.code, exc.detail) from exc
    for name in ("geometry", "grain_classes", "avalanching_spec", "topographic_commit_spec",
                 "initial_available_fraction"):
        if getattr(cfg, name) is not getattr(ctx, name):
            raise DenseRunnerError("incompatible_wind_config", f"{name} is not the shared context object")
    if float(cfg.mass_resolution_kg).hex() != float(ctx.mass_resolution_kg).hex():
        raise DenseRunnerError("incompatible_wind_config", "mass resolution differs from the shared bed")
    model = profile.model_config
    for scheduler_name, model_name in (("geometry", "geometry"), ("grain_classes", "grain_classes"),
                                       ("avalanching_spec", "avalanching"),
                                       ("topographic_commit_spec", "topographic_commit"),
                                       ("dormancy_spec", "dormancy")):
        if getattr(cfg, scheduler_name) is not getattr(model, model_name):
            raise DenseRunnerError("incompatible_wind_config", f"{scheduler_name} is not the authored model object")
    g = ctx.geometry
    dimensions = {"cells": int(g.ny * g.nx), "classes": len(ctx.grain_classes.classes),
                  "voxels": bed.case_record["n_voxels"], "window_voxels": bed.case_record["window_voxels"]}
    if any(dimensions[name] > maximum for name, maximum in
           (("cells", limits.max_cells), ("classes", limits.max_classes), ("voxels", limits.max_voxels))):
        raise DenseRunnerError("production_dimension_limit", f"case dimensions {dimensions} exceed the CPU policy")
    if (cfg.aeolian_process_mode.mode != _dense.SUPPORTED_WIND_MODE or cfg.detachment_source.mode != "reduced"
            or cfg.reduced_detachment is None or not cfg.reduced_detachment.configured):
        raise DenseRunnerError("unsupported_wind_mode", "configured native reduced baseline detachment required")
    if (model.water_coupling.enabled is not False or cfg.water_coupling_spec.enabled is not False
            or model.water_coupling.depth_update_rule != "constant_depth"
            or cfg.water_coupling_spec.depth_update_rule != "constant_depth"
            or ctx.depth_update_rule != "constant_depth"):
        raise DenseRunnerError("incompatible_water_rule", "disabled wind water transport and shared constant_depth")
    surface_creep = cfg.surface_creep_spec
    if (cfg.avalanching_spec.enabled is not True or not cfg.topographic_wind_spec.enabled
            or not cfg.flow_separation_spec.enabled or surface_creep is None or not surface_creep.active
            or cfg.vegetation_wake_spec.enabled or cfg.spatial_block_diagnostics_enabled):
        raise DenseRunnerError("unsupported_wind_mode", "the Small profile is enabled avalanching, topographic FFT, "
                               "resolved separation and creep, without vegetation wake or spatial diagnostics")
    if cfg.particle_transport.mobile_state_representation != "flat":
        raise DenseRunnerError("unsupported_mobile_state", "only native flat cohorts are qualified")
    if type(cfg.outer_timestep_s) is not float or cfg.outer_timestep_s != 1.0:
        raise DenseRunnerError("invalid_wind_config", "the Small record requires outer_timestep_s 1.0")
    record_limits = profile.record_limits
    if (type(record_limits) is not w.WindowedWindRecordLimits
            or type(record_limits.block.step) is not w.SmallFeatureWindowedReducedStepLimits
            or record_limits.block.max_outer_steps != 1 or type(record_limits.max_blocks) is not int
            or record_limits.max_blocks < 1):
        raise DenseRunnerError("invalid_wind_profile", "explicit Small record limits with one outer step required")
    if not _hex64(profile.source_identity) or profile.source_identity != ctx.source_identity:
        raise DenseRunnerError("invalid_wind_profile", "the record and water Source tags must be the same 64-hex tag")
    winds = [i for i, e in enumerate(schedule.events) if type(e) is WindEvent]
    if not winds or any(schedule.events[i].duration_s != 1.0 for i in winds):
        raise DenseRunnerError("unsupported_wind_duration", "every Small wind event is exactly one 1 s Native block")
    try:
        threshold = float(w.reroute_threshold_m(ctx)).hex()
    except w.BedIntegrationError as exc:
        raise DenseRunnerError("invalid_water_inputs", _detail(exc)) from exc
    return {"scope": SCOPE, "calibration": profile.calibration, "label": profile.label,
            "production_calibration": "pending_not_qualified_by_interface_tests",
            "limits": limits.as_record(), "dimensions": dimensions, "water_rule": "constant_depth",
            "hydraulic_routing_policy": {"native_reader": "maple_syrup.sediment_bed.reroute_threshold_m",
                                         "threshold_m_hex": threshold, "decision": "actual_native_commit_bed_only"},
            "native_config_audit": audit, "schedule_binding": schedule_binding,
            "record_limits": _canonical_input(n.np, record_limits, "profile.record_limits"),
            "source_identity": profile.source_identity, "wind_event_indices": winds,
            "case_sha256": bed.case_sha256}


def _require_native_apis(w: SimpleNamespace) -> None:
    """The finalizer origin keyword and the exact callback-free SYRUP entry/event APIs must exist."""
    if "record_origin_s" not in inspect.signature(w.finalize_windowed_small_feature_wind_event).parameters:
        raise DenseRunnerError("unsupported_native_finalizer", "MAPLE must support explicit record_origin_s")
    expected = {"commit_positive_windowed_entry": ("state", "context", "graph", "reference", "current_time_s",
                                                   "routing_source_m", "control"),
                "complete_positive_windowed_event": ("state", "context", "column", "field", "schedule", "vegetation",
                                                     "sediment", "max_end_s", "control", "policy", "report_every_s")}
    for name, parameters in expected.items():
        function = getattr(w.shared, name, None)
        if function is None or tuple(inspect.signature(function).parameters) != parameters:
            raise DenseRunnerError("unsupported_syrup_entry_api", f"SYRUP {name} with exact parameters is required")


def build_windowed_runner(bed: WindowedBed, schedule: DenseSchedule, *, wind: DenseWindInputs,
                          water: DenseWaterInputs, control: DenseRunnerControl, limits: DenseRunnerLimits,
                          profile: WindowedWindProfile, production_limits: Any) -> WindowedRunner:
    """Closed factory: every original DenseRunner guard and field, with the windowed owners. Refuses before
    any native transport, commit or work copy."""
    if type(bed) is not WindowedBed:
        raise DenseRunnerError("invalid_input", "bed must be a WindowedBed")
    if type(schedule) is not DenseSchedule:
        raise DenseRunnerError("invalid_input", "schedule must be a DenseSchedule")
    for value, expected in ((wind, DenseWindInputs), (water, DenseWaterInputs), (control, DenseRunnerControl),
                            (limits, DenseRunnerLimits), (profile, WindowedWindProfile)):
        if type(value) is not expected:
            raise DenseRunnerError("invalid_input", f"expected {expected.__name__}, got {type(value).__name__}")
    if len(schedule.events) > limits.max_events:
        raise DenseRunnerError("event_limit", f"{len(schedule.events)} events > max_events={limits.max_events}")
    _require_complete_protocol()
    try:
        process = _windowed_process.check_process(production_limits, checkpoint="construction")
    except _production.DenseProductionError as exc:
        raise DenseRunnerError(exc.code, exc.detail) from exc
    n, w = _dense._native(), _windowed_native()
    _require_windowed_scale_receiver(w)
    np = n.np
    try:
        bed.verify_case_inputs()
    except DenseBedError as exc:
        raise DenseRunnerError(exc.code, exc.detail) from exc
    _require_native_apis(w)
    if hasattr(bed, "_dynamic"):
        raise DenseRunnerError("invalid_vegetation_profile", "dynamic vegetation is outside the windowed profile")
    if type(water.initial_soil_water_m) is not np.ndarray:
        raise DenseRunnerError("invalid_water_inputs", "initial_soil_water_m must be an exact host array")
    constructor_terms = _metadata_terms(np, (bed._case_record, bed.limits, bed.context,
        bed.vegetation, bed.threshold_modifier_state,
        wind.config, water, control, profile.model_config, profile.record_limits, schedule,
        limits, production_limits, (profile.calibration, profile.label, profile.source_identity)),
        limits.max_receipt_bytes, len(schedule.events))
    amount = sum(value for name, value in constructor_terms if name != "finite_sort_and_projection_visits")
    # Complete known constructor roles: initial inventory/vector seed, five
    # cumulative class vectors, private soil and both canonical pool passes.
    amount += 8 * len(bed.context.grain_classes.classes) * 8
    amount += int(water.initial_soil_water_m.nbytes) + 2 * _bed._tree_nbytes(wind.initial_mobile_state)
    if amount > limits.max_work_bytes:
        raise DenseRunnerError("work_bytes_cap", "known windowed constructor roles exceed original max_work_bytes")
    try:
        _windowed_process.check_process(production_limits, process["process_identity"],
                                  upcoming_wrapper_bytes=amount, checkpoint="before_windowed_constructor")
    except _production.DenseProductionError as exc:
        raise DenseRunnerError(exc.code, exc.detail) from exc
    _profile_record(n, w, bed, wind, schedule, profile, production_limits)
    pool = wind.initial_mobile_state
    if type(pool) is not n.MobileCohortState:
        raise DenseRunnerError("unsupported_mobile_state", "only a flat native MobileCohortState is supported")
    try:
        _bed._require_host_arrays(pool, "initial_mobile_state")
    except DenseBedError as exc:
        raise DenseRunnerError(exc.code, exc.detail) from exc
    if not n.mobile_state_is_genuinely_empty(pool):
        raise DenseRunnerError("mobile_state_not_empty", "the initial wind pool must be genuinely empty")
    soil = DenseRunner._check_water(n, bed, water, control, limits)
    ctx = bed.context
    if limits.max_report_rows != REPORT_ROWS or ctx.bridge_limits.max_report_rows != REPORT_ROWS:
        raise DenseRunnerError("report_rows_limit", "the original 4096 report declaration is required (MASY and SYRUP)")
    if water.terrain.allow_masked_nodata or not water.terrain.strict:
        raise DenseRunnerError("incompatible_water_inputs", "the shared profile uses a strict unmasked terrain")
    try:
        w.shared._positive_control_projection(ctx, control.sediment_control)
    except w.BedIntegrationError as exc:
        raise DenseRunnerError("unsupported_water_control", _detail(exc)) from exc

    g = ctx.geometry
    n_classes = len(ctx.grain_classes.classes)
    zeros = np.zeros(n_classes, dtype=np.float64)
    runner = WindowedRunner(
        _TOKEN, _production=_EVIDENCE, _wind_profile=profile, _production_limits=production_limits,
        _worker_identity=process["process_identity"], _bed=bed, _schedule=schedule, _wind=wind, _water=water,
        _control=control, _limits=limits, _n_classes=n_classes,
        _inventory_elements=g.ny * g.nx * (bed.case_record["n_voxels"] + 1),
        _inventory0=_dense._vec(np, n.bed_inventory(bed.state)).copy(),
        _pool=pool, _soil=soil, _routing=None, _ledger=_dense.ClockLedger(bed.time_s), _next=0, _n_accepted=0,
        _totals={"export": zeros.copy(), "input": zeros.copy(), "bound": zeros.copy(),
                 "internal_terminal_deposition": zeros.copy(),
                 "water": {"rain_m3": 0.0, "drainage_m3": 0.0, "storm_export_m3": 0.0,
                           "dry_surface_removed_m3": 0.0, "dry_soil_removed_m3": 0.0},
                 "n_entry_reroutes": 0, "n_event_reroutes": 0},
        _receipts=[], _stop_receipt=None, _active_evidence=None, _water_record=None)
    runner._static_parts = runner._static_parts_record(n)
    runner._static_parts_sha256 = _bed._sha256(runner._static_parts)
    runner._static_record = {
        "schema": STATIC_SCHEMA, "profile": dict(PROFILE), "case_sha256": bed.case_sha256,
        "schedule_sha256": schedule.digest, "case_schedule_sha256": case_schedule_digest(bed.case_sha256, schedule),
        "controlling_inputs_sha256": runner._static_parts_sha256, "limits": limits.as_record(),
        "bed_limits": bed.limits.as_record(),
        # Initial values of evolving process-private state: recorded, never re-verified as static.
        "initial_evolving": {
            "mobile_state_sha256": _bed._sha256(_canonical_input(np, pool, "mobile")),
            "soil_water_m": _bed._array_digest(soil)},
    }
    runner._static_sha256 = _bed._sha256(runner._static_record)
    runner._accepted_bed = bed.identity_sha256()
    runner._accepted_private = DenseRunner._private_identity(np, runner._pool, runner._soil)
    return runner
