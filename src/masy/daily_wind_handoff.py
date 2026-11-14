"""Opt-in CSV wind transactions on the native floating-water lifecycle.

The water boundary belongs to SYRUP. This adapter supplies actual MAPLE wind
completion, explicit source qualification and the bounded schedule cursor.
It does not implement water physics, run a day or mint a native water record.
Native imports are lazy. Historical daily_native / N15 factories stay strict.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from fractions import Fraction
from typing import Any

from masy import daily_wind as dw
from masy.daily_csv import WindOccurrence
from masy.daily_schedule import DailyPlan

PROFILE = dw._HANDOFF_PROFILE
RECORD_SCHEMA = "masy.wind_handoff_cursor.v1"
RECEIPT_SCHEMA = "masy.wind_handoff.v1"
# Filled only after the owning source is frozen and independently qualified.
# Never calculate/accept a new runtime package hash automatically.
SYRUP_PACKAGE_SHA256 = "0bc7dc17c7eaa0e7445896759654bb3fff0649a14bcf43e79115b0cee2afed76"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_RECORD_KEYS = frozenset({"schema", "profile", "scenario_sha256", "executor_sha256", "run_digest", "native_manifest_sha256", "native_head", "native_context", "cursor", "policy"})
_CURSOR_KEYS = frozenset({"next_index", "last_id", "last_sha256", "last_receipt_sha256", "wind_elapsed_numerator", "wind_elapsed_denominator", "sediment"})


class WindHandoffError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code, self.detail = code, detail[:500]
        super().__init__(f"{code}: {self.detail}")


@dataclasses.dataclass(frozen=True)
class WindBridgeLimits:
    max_receipt_bytes: int
    max_record_bytes: int

    def __post_init__(self) -> None:
        for name in ("max_receipt_bytes", "max_record_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise WindHandoffError("limits", f"{name} must be a plain positive integer")


@dataclasses.dataclass(frozen=True)
class _Cursor:
    next_index: int = 0
    last_id: str | None = None
    last_sha256: str | None = None
    last_receipt_sha256: str | None = None
    elapsed: Fraction = Fraction(0)
    initial_inventory_hex: tuple[str, ...] = ()
    boundary_export: tuple[Fraction, ...] = ()
    boundary_input: tuple[Fraction, ...] = ()
    inventory_bound: Fraction = Fraction(0)

    def record(self) -> dict:
        return {"next_index": self.next_index, "last_id": self.last_id, "last_sha256": self.last_sha256,
                "last_receipt_sha256": self.last_receipt_sha256, "wind_elapsed_numerator": str(self.elapsed.numerator),
                "wind_elapsed_denominator": str(self.elapsed.denominator),
                "sediment": {"initial_inventory_hex": list(self.initial_inventory_hex),
                             "boundary_export": [_fraction_record(v) for v in self.boundary_export],
                             "boundary_input": [_fraction_record(v) for v in self.boundary_input],
                             "inventory_bound": _fraction_record(self.inventory_bound)}}


@dataclasses.dataclass(frozen=True)
class _Binding:
    lineage: Any
    context: Any
    run_digest: str


@dataclasses.dataclass(frozen=True)
class _Pending:
    candidate: Any
    cursor: _Cursor
    receipt_json: str


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("ascii")).hexdigest()


def _fraction_record(value: Fraction) -> list[str]:
    return [str(value.numerator), str(value.denominator)]


def _read_fraction(value: Any) -> Fraction:
    if type(value) is not list or len(value) != 2 or any(type(v) is not str or len(v) > 1000 for v in value):
        raise WindHandoffError("record", "a bounded exact fraction is required")
    try:
        out = Fraction(int(value[0]), int(value[1]))
    except (ValueError, ZeroDivisionError):
        raise WindHandoffError("record", "invalid exact fraction") from None
    if out < 0 or _fraction_record(out) != value:
        raise WindHandoffError("record", "a canonical nonnegative exact fraction is required")
    return out


def _hex(value: Any, name: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise WindHandoffError("record", f"{name} must be a lowercase SHA-256")
    return value


def _bounded(value: Any, limit: int, name: str) -> str:
    text = _json(value)
    if len(text.encode("ascii")) > limit:
        raise WindHandoffError("record_limit", f"{name} exceeds its explicit byte cap")
    return text


def _native():
    from maple_syrup import floating_event_lifecycle as lc
    from maple_syrup import floating_wind_handoff as wh
    return lc, wh


def _check_handoff_source(source: Any, admission: Any) -> dict:
    facts = dw._source_facts(source, admission)
    if SYRUP_PACKAGE_SHA256 == "0" * 64:
        raise dw.DailyWindError("unqualified_profile", "the new native wind-handoff source has not been qualified")
    if facts["syrup_package_digest"] != SYRUP_PACKAGE_SHA256:
        raise dw.DailyWindError("native_source", "the exact qualified native wind-handoff source is required; no automatic rebinding")
    return facts


def prepare_wind_handoff(plan: DailyPlan, source: Any, admission: Any, context: Any, *, wind_limits: dw.DailyWindLimits,
                         handoff_limits: Any, bridge_limits: WindBridgeLimits, available_bytes_fn: Any = None,
                         reserve_bytes: int = 0, batch_cells: int = 64) -> "WindHandoffExecutor":
    """Prepare the explicit new profile; no water owner, bed or day is changed.

    The caller supplies the native floating configuration and static event
    factory at start. Their physical parameters are never invented here.
    """
    if type(plan) is not DailyPlan:
        raise WindHandoffError("plan", "a parsed DailyPlan is required")
    if type(wind_limits) is not dw.DailyWindLimits or type(bridge_limits) is not WindBridgeLimits:
        raise WindHandoffError("limits", "explicit wind and bridge limits are required")
    if type(reserve_bytes) is not int or reserve_bytes < 0 or type(batch_cells) is not int or batch_cells < 1:
        raise WindHandoffError("limits", "reserve_bytes and batch_cells must be explicit non-Boolean integers in range")
    wind = dw._prepare_daily_wind(source, admission, context, limits=wind_limits, profile=PROFILE)
    _lc, wh = _native()
    if type(handoff_limits) is not wh.WindHandoffLimits or batch_cells > handoff_limits.max_batch_cells:
        raise WindHandoffError("limits", "native WindHandoffLimits and an admitted batch size are required")
    if handoff_limits.max_cohorts < admission.compiled.config.particle_transport.max_active_cohorts:
        raise WindHandoffError("limits", "the native handoff cohort cap must admit the configured wind pool before execution")
    if available_bytes_fn is None:
        from maple_syrup.subsurface import default_host_available_bytes
        available_bytes_fn = default_host_available_bytes
    if not callable(available_bytes_fn):
        raise WindHandoffError("limits", "available_bytes_fn must be callable")
    return WindHandoffExecutor(plan, wind, context, handoff_limits, bridge_limits, available_bytes_fn, reserve_bytes, batch_cells)


class WindHandoffExecutor:
    """One live native lineage plus scalar wind progress; no event-state history.

    Native water APIs may advance ``lineage`` between wind calls. Its accepted
    head is authoritative; this adapter never keeps a second water inventory.
    """

    __slots__ = ("_plan", "_wind", "_context", "_limits", "_bridge_limits", "_reader", "_reserve", "_batch",
                 "_binding", "_cursor", "_last_json", "_busy", "_pending", "_pending_start")

    def __init__(self, plan, wind, context, limits, bridge_limits, reader, reserve, batch):
        self._plan, self._wind, self._context = plan, wind, context
        self._limits, self._bridge_limits, self._reader, self._reserve, self._batch = limits, bridge_limits, reader, reserve, batch
        self._binding, self._cursor, self._last_json = None, _Cursor(), None
        self._busy, self._pending, self._pending_start = False, None, None

    @property
    def lineage(self):
        if self._binding is None:
            raise WindHandoffError("state", "the wind bridge has not been started or restored")
        return self._binding.lineage

    @property
    def run_digest(self) -> str:
        if self._binding is None:
            raise WindHandoffError("state", "the run has not been started or restored")
        return self._binding.run_digest

    @property
    def next_wind_index(self) -> int:
        return self._cursor.next_index

    def _policy(self) -> dict:
        return {"wind": self._wind._limits.as_record(), "handoff": self._limits.as_record(),
                "bridge": dataclasses.asdict(self._bridge_limits), "reserve_bytes": self._reserve, "batch_cells": self._batch}

    def _digest(self, context_record: dict) -> str:
        return _sha({"profile": PROFILE, "scenario_sha256": self._plan.scenario.identity_sha256,
                     "executor_sha256": self._wind.identity_sha256, "context": context_record, "policy": self._policy()})

    def _new_binding(self, lineage, boundary, context):
        # Called by the owning initializer with its real staged proposal before
        # retirement/publication. All allocating coordinator work happens here.
        self._wind._recheck()
        binding = _Binding(lineage, context, self._digest(context.record()))
        from maple_syrup.sediment_bed import bed_inventory
        inventory = tuple(float(v).hex() for v in bed_inventory(boundary.bed))
        zeros = (Fraction(0),) * len(inventory)
        cursor = _Cursor(initial_inventory_hex=inventory, boundary_export=zeros, boundary_input=zeros)
        self._pending_start = (binding, boundary, cursor)

    def start(self, owner, bed, event_factory, *, preferential=None):
        """Start before the first wind/day using the genuine native initializer.

        No fictitious rain or daily interval is inserted. If interrupted after
        native publication, ``lineage`` still exposes the accepted initial head.
        """
        if self._binding is not None or self._busy:
            raise WindHandoffError("state", "start needs a fresh idle wind bridge")
        _lc, wh = _native()
        self._busy = True
        try:
            self._wind._recheck()
            wh.start_wind_boundary(owner, bed, event_factory, bed_context=self._context, limits=self._limits,
                                   available_bytes_fn=self._reader, preferential=preferential, reserve_bytes=self._reserve,
                                   batch_cells=self._batch, before_publish=self._new_binding)
        finally:
            pending = self._pending_start
            if pending is not None and pending[0].lineage.head is pending[1]:
                self._binding, self._cursor = pending[0], pending[2]
            self._pending_start, self._busy = None, False
        if self._binding is None:
            raise WindHandoffError("interface", "native initialization did not publish its proposed boundary")
        return self.lineage.head

    def _head(self):
        lc, _wh = _native()
        try:
            kind, head = lc.saveable_head(self.lineage)
        except lc.LifecycleError as exc:
            raise WindHandoffError("not_ready", str(exc)) from None
        if kind != "boundary":
            raise WindHandoffError("not_ready", "wind needs a completed Boundary; hand back the closed day first")
        return head

    def _check_day(self, boundary, occurrence):
        cursor = boundary.cursor
        ordinal = 0 if cursor is None else cursor.ordinal
        if cursor is not None and cursor.run_digest != self.run_digest:
            raise WindHandoffError("calendar", "the native water cursor belongs to another resolved run")
        target = (occurrence.date - self._plan.scenario.settings.start_date).days
        if ordinal != target:
            raise WindHandoffError("calendar", "wind must run on its own nominal date, before that date's rain/day update")

    def _check_progress(self, boundary, cursor):
        native = boundary.cursor
        ordinal = 0 if native is None else native.ordinal
        if native is not None and native.run_digest != self.run_digest:
            raise WindHandoffError("calendar", "the native water cursor belongs to another resolved run")
        if type(ordinal) is not int or not 0 <= ordinal <= self._plan.day_count:
            raise WindHandoffError("calendar", "native water progress is outside this scenario")
        rows, start = self._plan.scenario.occurrences, self._plan.scenario.settings.start_date
        if cursor.next_index and (rows[cursor.next_index - 1].date - start).days > ordinal:
            raise WindHandoffError("calendar", "the accepted wind prefix is ahead of the native calendar")
        if cursor.next_index < len(rows) and (rows[cursor.next_index].date - start).days < ordinal:
            raise WindHandoffError("calendar", "native daily progress skipped an unexecuted wind row")

    def _settle_pending(self):
        pending = self._pending
        if pending is None:
            return
        ref = pending.candidate.accepted_boundary_ref
        accepted = None if ref is None else ref()
        if accepted is not None and self.lineage.head is accepted:
            self._cursor, self._last_json = pending.cursor, pending.receipt_json
        self._pending = None

    def execute(self, occurrence: WindOccurrence):
        """Run and adopt the next parsed wind row, returning the genuine Boundary.

        Surface return may make that boundary pending recession. Actual native
        water completion is then required before another wind or daily update.
        The current scalar receipt is available separately via last_receipt().
        """
        if self._busy:
            raise WindHandoffError("state", "another bridge transaction is active")
        rows, index = self._plan.scenario.occurrences, self._cursor.next_index
        if type(occurrence) is not WindOccurrence or index >= len(rows) or occurrence != rows[index] or occurrence.index != index:
            raise WindHandoffError("cursor", "only the next wind row of the resolved scenario can execute")
        parent = self._head()
        self._check_day(parent, occurrence)
        _lc, wh = _native()
        self._busy = True
        completed = None
        try:
            completed = self._wind.execute_finalized(parent.bed, occurrence, float(parent.t_s))
            native = wh.stage_wind_handoff(self.lineage, parent, completed.finalization, completed.candidate.state,
                                          context=self._binding.context, occurrence_id=occurrence.occurrence_id,
                                          occurrence_sha256=occurrence.occurrence_sha256, start_s=float(parent.t_s),
                                          limits=self._limits, available_bytes_fn=self._reader,
                                          reserve_bytes=self._reserve, batch_cells=self._batch)
            self._wind._recheck()
            wind_receipt = completed.candidate.receipt()
            handoff_receipt = native.receipt()
            receipt = _bounded({"schema": RECEIPT_SCHEMA, "profile": PROFILE, "run_digest": self.run_digest,
                                "scenario_sha256": self._plan.scenario.identity_sha256, "wind": wind_receipt,
                                "handoff": handoff_receipt, "next_wind_index": index + 1},
                               self._bridge_limits.max_receipt_bytes, "combined wind receipt")
            end_s = completed.candidate.end_s
            classes = wind_receipt["classes"]
            exports = tuple(Fraction(float.fromhex(v)) for v in classes["boundary_export_hex"])
            inputs = tuple(Fraction(float.fromhex(v)) for v in classes["boundary_input_hex"])
            if len(exports) != len(self._cursor.boundary_export) or len(inputs) != len(self._cursor.boundary_input):
                raise WindHandoffError("accounting", "native class order/count changed across the wind transaction")
            cursor = _Cursor(index + 1, occurrence.occurrence_id, occurrence.occurrence_sha256,
                             hashlib.sha256(receipt.encode("ascii")).hexdigest(),
                             self._cursor.elapsed + Fraction(end_s) - Fraction(float(parent.t_s)),
                             self._cursor.initial_inventory_hex,
                             tuple(a + b for a, b in zip(self._cursor.boundary_export, exports)),
                             tuple(a + b for a, b in zip(self._cursor.boundary_input, inputs)),
                             self._cursor.inventory_bound + Fraction(float(classes["inventory_bound_kg"])))
            self._pending = _Pending(native, cursor, receipt)
            # Native state is already staged. Release the transient wind result
            # and empty mobile pool before publishing the water boundary.
            completed = None
            wh.adopt_wind_handoff(self.lineage, parent, native, available_bytes_fn=self._reader,
                                 reserve_bytes=self._reserve, before_publish=self._wind._recheck)
        finally:
            completed = None
            try:
                self._settle_pending()
            finally:
                self._busy = False
        if self._cursor.next_index != index + 1:
            raise WindHandoffError("interface", "native adoption did not publish the staged boundary")
        return self.lineage.head

    def last_receipt(self) -> dict | None:
        return None if self._last_json is None else json.loads(self._last_json)

    def wind_accounting(self) -> dict:
        """Exact accumulation of actual native wind inputs/exports/bounds only.

        Water-process sediment receipts remain the water coordinator's input to
        the combined class budget; wind and water exports must not be conflated.
        """
        return self._cursor.record()["sediment"]

    def checkpoint_record(self, *, native_manifest_sha256: str) -> str:
        """Bind scalar schedule progress to a verified native boundary file.

        The caller publishes this record with the native file atomically and
        keeps its external SHA. This method writes no files or model arrays.
        """
        if self._busy or self._pending is not None:
            raise WindHandoffError("state", "only an idle accepted boundary can be recorded")
        native_sha = _hex(native_manifest_sha256, "native_manifest_sha256")
        head = self._head()
        self._check_progress(head, self._cursor)
        lc, _wh = _native()
        self._wind._recheck()
        return _bounded({"schema": RECORD_SCHEMA, "profile": PROFILE,
                         "scenario_sha256": self._plan.scenario.identity_sha256, "executor_sha256": self._wind.identity_sha256,
                         "run_digest": self.run_digest, "native_manifest_sha256": native_sha,
                         "native_head": lc.head_record(head), "native_context": self._binding.context.record(),
                         "cursor": self._cursor.record(), "policy": self._policy()},
                        self._bridge_limits.max_record_bytes, "wind restart record")

    def restore(self, lineage, boundary, record_json: str, *, expected_record_sha256: str, native_manifest_sha256: str):
        """Bind a fresh bridge to a genuinely verified restored native Boundary.

        Static runtime inputs must already be prepared and source-verified by
        the caller. This function never initializes terrain or a water owner.
        It is not an automatic complete-model file/runtime provisioning CLI.
        """
        if self._binding is not None or self._busy:
            raise WindHandoffError("state", "restore needs a fresh idle bridge")
        if type(record_json) is not str or len(record_json) > self._bridge_limits.max_record_bytes:
            raise WindHandoffError("record_limit", "restart record is absent or exceeds its byte cap")
        try:
            raw = record_json.encode("ascii")
        except UnicodeEncodeError:
            raise WindHandoffError("record", "restart record must be canonical ASCII JSON") from None
        expected = _hex(expected_record_sha256, "expected_record_sha256")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise WindHandoffError("record", "restart record differs from the expected external SHA")
        try:
            record = json.loads(record_json)
        except ValueError:
            raise WindHandoffError("record", "restart record is not JSON") from None
        if type(record) is not dict or set(record) != _RECORD_KEYS or record["schema"] != RECORD_SCHEMA or record["profile"] != PROFILE:
            raise WindHandoffError("record", "restart record has an unknown schema or fields")
        lc, wh = _native()
        kind, actual = lc.saveable_head(lineage)
        if kind != "boundary" or actual is not boundary:
            raise WindHandoffError("state", "the restored Boundary is not the lineage's accepted head")
        self._wind._recheck()
        if (record["scenario_sha256"] != self._plan.scenario.identity_sha256 or record["executor_sha256"] != self._wind.identity_sha256
                or record["native_manifest_sha256"] != _hex(native_manifest_sha256, "native_manifest_sha256")
                or record["native_head"] != lc.head_record(boundary) or record["policy"] != self._policy()):
            raise WindHandoffError("record", "restart source, case, schedule, policy or native boundary differs")
        c = record["cursor"]
        if type(c) is not dict or set(c) != _CURSOR_KEYS or type(c["next_index"]) is not int or not 0 <= c["next_index"] <= len(self._plan.scenario.occurrences):
            raise WindHandoffError("record", "restart wind cursor is malformed")
        try:
            if any(type(c[k]) is not str or len(c[k]) > 1000 for k in ("wind_elapsed_numerator", "wind_elapsed_denominator")):
                raise ValueError("invalid exact elapsed interval")
            elapsed = Fraction(int(c["wind_elapsed_numerator"]), int(c["wind_elapsed_denominator"]))
            if elapsed < 0 or str(elapsed.numerator) != c["wind_elapsed_numerator"] or str(elapsed.denominator) != c["wind_elapsed_denominator"]:
                raise ValueError("noncanonical exact interval")
        except (ValueError, ZeroDivisionError):
            raise WindHandoffError("record", "restart exact wind interval is malformed") from None
        index = c["next_index"]
        if index == 0:
            if any(c[k] is not None for k in ("last_id", "last_sha256", "last_receipt_sha256")) or elapsed != 0:
                raise WindHandoffError("record", "initial wind cursor contains an accepted occurrence")
        else:
            previous = self._plan.scenario.occurrences[index - 1]
            if c["last_id"] != previous.occurrence_id or c["last_sha256"] != previous.occurrence_sha256:
                raise WindHandoffError("record", "restart cursor does not match the accepted wind prefix")
            _hex(c["last_receipt_sha256"], "last_receipt_sha256")
        context = wh.restore_wind_context(boundary, self._context, record["native_context"])
        digest = self._digest(context.record())
        if record["run_digest"] != digest or (boundary.cursor is not None and boundary.cursor.run_digest != digest):
            raise WindHandoffError("record", "restart water and wind run identities differ")
        binding = _Binding(lineage, context, digest)
        sed = c["sediment"]
        keys = {"initial_inventory_hex", "boundary_export", "boundary_input", "inventory_bound"}
        nc = len(self._context.grain_classes.classes)
        if type(sed) is not dict or set(sed) != keys or any(type(sed[k]) is not list or len(sed[k]) != nc for k in ("initial_inventory_hex", "boundary_export", "boundary_input")):
            raise WindHandoffError("record", "restart wind sediment accounting has another class count or fields")
        inventory = tuple(sed["initial_inventory_hex"])
        try:
            for value in inventory:
                if type(value) is not str or len(value) > 40:
                    raise ValueError("invalid inventory")
                x = float.fromhex(value)
                if not math.isfinite(x) or x < 0 or x.hex() != value:
                    raise ValueError("invalid inventory")
        except ValueError:
            raise WindHandoffError("record", "initial sediment inventory is not canonical finite native FP64") from None
        exports = tuple(_read_fraction(v) for v in sed["boundary_export"])
        inputs = tuple(_read_fraction(v) for v in sed["boundary_input"])
        bound = _read_fraction(sed["inventory_bound"])
        if index == 0 and (any(exports) or any(inputs) or bound):
            raise WindHandoffError("record", "initial cursor contains previously accepted wind exchanges")
        cursor = _Cursor(index, c["last_id"], c["last_sha256"], c["last_receipt_sha256"], elapsed, inventory, exports, inputs, bound)
        # Validate calendar consistency before installing any coordinator state.
        native_cursor = boundary.cursor
        ordinal = 0 if native_cursor is None else native_cursor.ordinal
        rows = self._plan.scenario.occurrences
        if (type(ordinal) is not int or not 0 <= ordinal <= self._plan.day_count
                or (index and (rows[index - 1].date - self._plan.scenario.settings.start_date).days > ordinal)
                or (index < len(rows) and (rows[index].date - self._plan.scenario.settings.start_date).days < ordinal)):
            raise WindHandoffError("calendar", "restart native day and accepted wind prefix differ")
        self._wind._recheck()
        self._binding, self._cursor = binding, cursor
        # The accepted native state plus the scalar cursor are authoritative.
        # A prior receipt payload is deliberately not regenerated on restore.
        self._last_json = None
        return boundary
