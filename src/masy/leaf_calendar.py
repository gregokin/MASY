"""Pure LEAF-calendar sequencing for the scheduled LEAF-water participant (Design.md section 4). No water, no plants, no I/O.

MASY only sequences dated windows: every biology rule (growth grid <= 7 days, local midnights, new years, deferred season, pending ENABLED credit, flush/transport semantics) is the actual
`leaf.biocalendar` (reached through the verified source receipt; this module takes the module object as an argument and never imports LEAF itself, so the default water-only workers never
import it). `LeafCalendar` is an immutable wrapper of ONE `biocalendar.Schedule`: the single source of the pending-credit Fraction, the grid and the deferred-season flag. Every method returns a new
calendar; nothing is mutated, so a refused/failed trial leaves the accepted calendar untouched. Wind, rain, recession and closed-gate intervals only advance the calendar (`transport`) and
never carry E, T or growth; boundaries crossed by transport are skipped (no catch-up) and a year end inside transport defers the season turnover to the next open growth-enabled interval.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
from fractions import Fraction
from typing import Any

__all__ = ["LeafCalendarError", "LeafCalendar", "PlanItem", "exact_datetime", "MAX_FRAGMENTS_DEFAULT"]

MAX_FRAGMENTS_DEFAULT = 4096


class LeafCalendarError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def exact_datetime(start: _dt.datetime, elapsed_s: Any) -> _dt.datetime:
    """The EXACT fixed-offset datetime `start + elapsed_s`. `elapsed_s` must be the exact `Fraction` the EXISTING accepted bridge returns (`ClockLedger.global_elapsed(native_t)` / `CalendarStart.instant`
    -> `elapsed_s`), never a re-derived binary elapsed. LEAF records are datetimes (microsecond resolution), so an instant that is not an exact multiple of one microsecond is REFUSED with a typed error
    (no rounding, no tolerance, no restamp); whole-second scheduler instants (midnights, integral durations) always pass."""
    if type(elapsed_s) is not Fraction:
        raise LeafCalendarError("bad_elapsed", "elapsed seconds must be an exact Fraction")
    micro = elapsed_s * 1_000_000
    if micro.denominator != 1:
        raise LeafCalendarError("instant_not_exact", f"elapsed {elapsed_s} s is not an exact number of microseconds; the LEAF calendar refuses to round it")
    try:
        return start + _dt.timedelta(microseconds=int(micro))
    except OverflowError as exc:
        raise LeafCalendarError("calendar_range", str(exc)) from None


@dataclasses.dataclass(frozen=True)
class PlanItem:
    """One sub-fragment of an open interval and the LEAF decisions that go with it (`step` is the actual `biocalendar.OpenStep`)."""

    record: Any
    step: Any


@dataclasses.dataclass(frozen=True)
class LeafCalendar:
    bc: Any  # the verified leaf.biocalendar module object
    schedule: Any  # leaf.biocalendar.Schedule

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, self.bc.Schedule):
            raise LeafCalendarError("bad_schedule", "schedule must be the verified leaf.biocalendar.Schedule")

    @classmethod
    def new(cls, bc: Any, anchor: _dt.datetime, cadence: _dt.timedelta) -> "LeafCalendar":
        try:
            return cls(bc, bc.new_schedule(anchor, cadence))
        except bc.BioCalendarError as exc:
            raise LeafCalendarError("bad_calendar", str(exc)) from None

    @classmethod
    def from_schedule(cls, bc: Any, schedule: Any) -> "LeafCalendar":
        return cls(bc, schedule)

    @property
    def cursor(self) -> _dt.datetime:
        return self.schedule.cursor

    @property
    def pending_days(self) -> Fraction:
        return self.schedule.pending_days

    def plan_open(self, record: Any, *, growth_enabled: bool, has_plants: bool, max_fragments: int = MAX_FRAGMENTS_DEFAULT) -> tuple[PlanItem, ...]:
        """The ordered sub-fragments (split at local midnights, growth-grid points and new years) of ONE open interevent record, each with its `OpenStep`. The chain is pure: item k+1 starts
        from `items[k].step.schedule`; the calendar itself is unchanged (the caller adopts the last schedule with `with_schedule` only after the whole occurrence succeeded)."""
        bc = self.bc
        if isinstance(max_fragments, bool) or not isinstance(max_fragments, int) or max_fragments < 1:
            raise LeafCalendarError("bad_cap", "max_fragments must be a non-Boolean integer >= 1")
        bound = self.fragment_bound(record)  # derived from the dates/cadence BEFORE any piece is materialized
        if bound > max_fragments:
            raise LeafCalendarError("fragment_limit", f"the record can split into up to {bound} calendar fragments, over the declared cap {max_fragments}; refused before any split")
        try:
            pieces = bc.split_record(self.schedule, record)
            sched, items = self.schedule, []
            for piece in pieces:
                step = bc.open_step(sched, piece, growth_enabled=growth_enabled, has_plants=has_plants)
                items.append(PlanItem(piece, step))
                sched = step.schedule
        except bc.BioCalendarError as exc:
            raise LeafCalendarError("calendar_refused", str(exc)) from None
        return tuple(items)

    def fragment_bound(self, record: Any) -> int:
        """Upper bound on the number of pieces `split_record` can return, computed arithmetically from the local dates and the growth grid (no list of cut points is built): 1 + local midnights strictly
        inside the record + grid points strictly inside it (a coinciding cut only lowers the real count). A non-open record is one piece."""
        if not record.active:
            return 1
        mid = (record.end.date() - record.start.date()).days - (1 if record.end.time() == _dt.time(0) and record.end.microsecond == 0 else 0)
        grid = 0
        nb = self.schedule.next_boundary
        if nb < record.end:
            grid = int((record.end - nb) // self.schedule.cadence) + (1 if (record.end - nb) % self.schedule.cadence else 0)
        return 1 + max(int(mid), 0) + grid

    def with_schedule(self, schedule: Any) -> "LeafCalendar":
        return LeafCalendar(self.bc, schedule)

    def flush_required(self, *, next_growth_enabled: bool, next_is_transport: bool) -> bool:
        """Look-ahead (the scheduler knows what follows): True when enabled credit is pending and the next period is a transport event or growth-disabled, so the LAST growth-enabled
        endpoint must finish the partial period BEFORE it."""
        return bool(self.bc.flush_required(self.schedule, next_growth_enabled=next_growth_enabled, next_is_transport=next_is_transport))

    def flushed(self, *, growth_enabled: bool) -> "LeafCalendar":
        """The calendar after the flush AT the cursor (pending -> 0; grid/season unchanged). Refused with growth off and pending credit (the flush belongs to the last enabled endpoint)."""
        bc = self.bc
        try:
            return LeafCalendar(bc, bc.flush_step(self.schedule, growth_enabled=growth_enabled))
        except bc.BioCalendarError as exc:
            raise LeafCalendarError("flush_refused", str(exc)) from None

    def transport(self, start: _dt.datetime, end: _dt.datetime) -> tuple["LeafCalendar", Any]:
        """Advance over wind / rain / recession / a closed-gate pause: no E/T/growth, no catch-up; the pending credit must already be flushed (typed refusal otherwise)."""
        bc = self.bc
        try:
            step = bc.transport_step(self.schedule, start, end)
        except bc.BioCalendarError as exc:
            raise LeafCalendarError("transport_refused", str(exc)) from None
        return LeafCalendar(bc, step.schedule), step

    def to_dict(self) -> dict[str, Any]:
        return self.schedule.to_dict()
