"""Scheduled LEAF-water participant orchestration (Design.md section 5): the pure sequencing that `NativeWaterSession` hooks call for one INTEREVENT or RAIN occurrence when a LEAF participant is
configured. It composes only accepted pieces: the actual SYRUP lifecycle (enter/close), `masy.leaf_water_interval` (ONE hydrology+E advance then ONE actual T withdrawal per eligible fragment),
`masy.leaf_calendar` (the actual `leaf.biocalendar` decisions) and the biology codec. It owns no physics, no calendar law and no water store.

Capabilities: `WATER_ONLY` (no participant: every old record, signature and digest is unchanged) and `LEAF_WATER` (this module), orthogonal to the sealed CPU/CUDA backend policy.

Everything returns NEW objects; the caller (the session's staged child) publishes `(coordinator, lifecycle state, LeafBoundary)` by ONE assignment, so a refusal anywhere (typed, including the bounded
step budget) leaves the accepted pair untouched. There is no partial-T/replay path: a hydrology budget that is too small REFUSES the occurrence. No growth, E or T ever runs for rain/recession/wind
(`carry_boundary`), and a look-ahead flush finishes a pending enabled period at the LAST growth-enabled endpoint BEFORE a transport event or a growth-disabled interevent.

NOT provided here (see docs/native_leaf_calendar.md): the session/CLI/config wiring, the joint artifact layout and wind.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
from typing import Any

import numpy as np

from maple_syrup import water_lifecycle as wl

from masy import leaf_water_interval as lwi

__all__ = ["WATER_ONLY", "LEAF_WATER", "ParticipantError", "Window", "InterEventOutcome", "run_interevent", "carry_through_rain"]

WATER_ONLY = "water_only"
LEAF_WATER = "leaf_water"


class ParticipantError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclasses.dataclass(frozen=True)
class Window:
    """One calendar fragment of an interevent (the session's date fragments): exact aware datetimes, the shared dated mean temperature and scalar PET (mm/day, the existing Hargreaves value) and the
    TERRAIN PET field (m/s, the existing `terrain factor` applied ONCE) that drives both hydrology E and the canopy gather; the three explicit process gates of the occurrence."""

    start: _dt.datetime
    end: _dt.datetime
    temperature_c: float
    pet_scalar_mm_day: float
    pet_terrain_m_s: Any
    evaporation: bool
    transpiration: bool
    growth: bool


@dataclasses.dataclass(frozen=True)
class InterEventOutcome:
    closed: Any  # the closed 'boundary' LifecycleState
    leaf: Any  # the released LeafBoundary
    evidence: dict[str, Any]


def run_interevent(op: Any, leaf: Any, parent: Any, windows: Any, *, next_kind: str | None, next_growth: bool | None, token: str, max_accepted_steps: int = 100_000,
                   max_dt_s: float = 3600.0, max_retries: int = 12, min_dt_s: float = 2.0 ** -20) -> InterEventOutcome:
    """One whole interevent occurrence: enter from `parent` (a closed boundary or a COMPLETED storm), run every window through `leaf_water_interval.run_interval` (each window is split by LEAF into its
    midnight/grid sub-fragments), flush at the last growth-enabled endpoint when the NEXT period is transport or growth-disabled (`next_kind` = the next occurrence kind or None at the end of the
    scenario; `next_growth` = its growth gate when it is an interevent; at the END of the scenario (`None`) nothing is flushed: the partial period stays pending and restartable), close, and release the boundary. A total hydrology budget `max_accepted_steps` that is exceeded REFUSES the whole occurrence."""
    lop = op.lop
    if type(windows) is not tuple or not windows or not all(type(w) is Window for w in windows):
        raise ParticipantError("bad_windows", "windows must be a non-empty tuple of Window")
    if parent.role == "storm":
        state = wl.enter_interevent_phase(lop, parent, expected_parent=wl.expect_parent(parent))
    elif parent.role == "boundary":
        state = wl.enter_interevent_from_boundary(lop, parent, expected_parent=wl.expect_parent(parent))
    else:
        raise ParticipantError("invalid_parent_role", f"interevent needs a boundary or completed storm parent, got {parent.role!r}")
    head = lwi.bind_head(op, leaf, state, parent=parent)
    forcing = op.leaf["forcing"]
    steps = items = 0
    last_growth = None
    for k, w in enumerate(windows):
        pet_field = w.pet_terrain_m_s() if callable(w.pet_terrain_m_s) else w.pet_terrain_m_s  # a callable defers the per-date terrain PET build to this window (bounded memory)
        rec = forcing.TimestepRecord(start=w.start, end=w.end, temperature_c=w.temperature_c, pet_mm_per_day=w.pet_scalar_mm_day, process_kind="interevent", gate="open")
        has_plants = int(head.leaf.B_kg.shape[0]) > 0  # host count of the fixed live population (no device conversion)
        try:
            plan = head.leaf.calendar.plan_open(rec, growth_enabled=w.growth, has_plants=has_plants)
        except Exception as exc:
            raise ParticipantError("calendar_refused", str(exc)) from None
        for item in plan:
            remaining = int(max_accepted_steps) - steps
            if remaining <= 0:
                raise ParticipantError("step_budget", f"the occurrence exhausted its {int(max_accepted_steps)} accepted hydrology steps; refused (no partial transpiration, no replay)")
            token_i = f"{token}:{k}:{items}"
            try:
                cand = lwi.run_interval(op, head, item, pet_terrain_m_s=pet_field, interval_token=token_i, transpiration=w.transpiration, growth_enabled=w.growth, evaporation=w.evaporation,
                                        max_dt_s=max_dt_s, max_accepted_steps=remaining, max_retries=max_retries, min_dt_s=min_dt_s)
            except lwi.IntervalError as exc:
                if "did not reach the interval end" in str(exc):
                    raise ParticipantError("step_budget", f"{exc}; refused (no partial transpiration, no replay)") from None
                raise
            steps += int(cand.receipt["accepted_hydro_steps"])
            head = lwi.adopt(op, head, cand)
            items += 1
        last_growth = w.growth
    # look-ahead: finish the partial ENABLED period at the LAST growth-enabled endpoint, never during a disabled period or a transport event
    next_is_transport = next_kind is not None and next_kind != "interevent"
    flushed = False
    if next_kind is not None and head.leaf.calendar.pending_days > 0 and head.leaf.calendar.flush_required(next_growth_enabled=bool(next_growth) if next_kind == "interevent" else False, next_is_transport=next_is_transport):
        if not last_growth:
            raise ParticipantError("flush_endpoint", "pending enabled credit exists but the last endpoint is growth-disabled (it must have been flushed at the last enabled endpoint)")
        head = lwi.adopt(op, head, lwi.apply_growth(op, head, year_end=False, interval_token=f"{token}:flush"))
        flushed = True
    closed = wl.close_interevent(lop, head.water, expected_parent=wl.expect_parent(head.water))
    new_leaf = lwi.release_head(op, head, closed)
    ev = {"kind": "interevent", "token": token, "items": items, "accepted_hydro_steps": steps, "flushed": flushed, "generation": int(new_leaf.generation), "cursor": new_leaf.calendar.cursor.isoformat(),
          "pending_days": str(new_leaf.calendar.pending_days), "request_sequence": int(closed.request_sequence)}
    return InterEventOutcome(closed, new_leaf, ev)


def carry_through_rain(op: Any, leaf: Any, storm_state: Any, *, start: _dt.datetime, end: _dt.datetime, parent: Any = None) -> Any:
    """The biology across a REAL rain occurrence: the calendar advances over [start, end) (rain duration + recession, the REALIZED native end converted by the caller through the existing exact clock
    bridge) with NO E/T/growth/catch-up, and the boundary is re-released at the completed storm state."""
    return lwi.carry_boundary(op, leaf, storm_state, start=start, end=end, parent=parent)
