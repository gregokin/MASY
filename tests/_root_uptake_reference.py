"""TEST/DOCS reference arithmetic for the agreed LEAF/SYRUP plant-water contract (stdlib only; NOT production masy code).

This is documentation-grade reference code for the contract recorded in
`docs/subsurface_hydrology/development_phases_20261109.md` (SYRUP) and
`docs/shared_scheduler_phase0.md` here. It is NOT a production adapter, not
LEAF physiology and not SYRUP's inventory: it exists so a reviewer can check
the agreed numbers (including the user's example) without NumPy/Numba.

Contract, per coordinator timestep, in physical water VOLUME (m3):

- LEAF: plant p requests `D_p` and distributes `R_pcl = D_p * w_pcl` over its
  actually rooted cells c and layers l with `sum_cl w_pcl = 1`.
- SYRUP: reports withdrawable water `A_cl >= 0` per cell/layer, within its
  declared bounds; converts depth <-> volume with the explicit cell area
  exactly once (`depth_m * cell_area_m2`).
- Per cell/layer: `Q_cl = sum_p R_pcl`; `f_cl = min(1, A_cl / Q_cl)` when
  `Q_cl > 0`, else no withdrawal. Every plant rooted there gets
  `U_pcl = R_pcl * f_cl` (same factor, no priority, no order dependence, no
  plasticity, no reallocation of unmet demand to wetter locations).
- SYRUP debits `sum_p U_pcl <= A_cl` exactly once and reports accepted
  amounts and shortfalls; LEAF keeps no second soil-water store.
- Water-supported growth fraction for `D_p > 0`: `S_p = sum_cl U_pcl / D_p`
  (root-weighted satisfaction, NOT the minimum layer satisfaction). `D_p = 0`
  is explicit: no withdrawal and no invented growth factor (None).

User example: D = 100, weights 0.5/0.5, available 25/200 -> requests 50/50,
accepted 25/50, remaining 0/150, S = 0.75.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping

__all__ = ["ContractError", "UptakeAllocation", "allocate_uptake", "depth_to_volume_m3", "volume_to_depth_m"]

Number = int | float | Fraction
Location = tuple[Any, Any]  # (cell, layer) keys as the caller labels them


class ContractError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _exact(value: Any, where: str, *, allow_zero: bool = True) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ContractError("invalid_value", f"{where} must be a real number")
    if isinstance(value, float) and value != value:
        raise ContractError("invalid_value", f"{where} must be finite")
    try:
        exact = Fraction(value)
    except (OverflowError, ValueError) as exc:
        raise ContractError("invalid_value", f"{where} must be finite") from exc
    if exact < 0 or (not allow_zero and exact == 0):
        raise ContractError("invalid_value", f"{where} must be {'>=' if allow_zero else '>'} 0")
    return exact


def depth_to_volume_m3(depth_m: Number, cell_area_m2: Number) -> Fraction:
    """Explicit single conversion used by the SYRUP side (depth * area)."""
    return _exact(depth_m, "depth_m") * _exact(cell_area_m2, "cell_area_m2", allow_zero=False)


def volume_to_depth_m(volume_m3: Number, cell_area_m2: Number) -> Fraction:
    return _exact(volume_m3, "volume_m3") / _exact(cell_area_m2, "cell_area_m2", allow_zero=False)


@dataclass(frozen=True)
class UptakeAllocation:
    requests: dict[Any, dict[Location, Fraction]]      # R_pcl
    local_demand: dict[Location, Fraction]             # Q_cl
    fulfillment: dict[Location, Fraction]              # f_cl (0 where Q_cl = 0)
    accepted: dict[Any, dict[Location, Fraction]]      # U_pcl
    debited: dict[Location, Fraction]                  # sum_p U_pcl
    remaining: dict[Location, Fraction]                # A_cl - debited
    shortfall: dict[Location, Fraction]                # Q_cl - debited
    growth_fraction: dict[Any, Fraction | None]        # S_p, None for D_p = 0

    def as_floats(self) -> dict[str, Any]:
        def f(mapping: Mapping[Any, Any]) -> dict[Any, Any]:
            return {k: (f(v) if isinstance(v, dict) else (None if v is None else float(v))) for k, v in mapping.items()}
        return {"requests": f(self.requests), "local_demand": f(self.local_demand), "fulfillment": f(self.fulfillment),
                "accepted": f(self.accepted), "debited": f(self.debited), "remaining": f(self.remaining),
                "shortfall": f(self.shortfall), "growth_fraction": f(self.growth_fraction)}


def allocate_uptake(demands: Mapping[Any, Number], weights: Mapping[Any, Mapping[Location, Number]],
                    available: Mapping[Location, Number], *, weight_tolerance: Fraction = Fraction(0)) -> UptakeAllocation:
    """Exact rational evaluation of the contract. `weights[p]` must sum to 1
    (within `weight_tolerance`, default exact) over the plant's rooted
    locations; a location absent from `available` is a contract violation
    (LEAF may only request where SYRUP reports water)."""
    avail = {loc: _exact(v, f"available[{loc}]") for loc, v in available.items()}
    requests: dict[Any, dict[Location, Fraction]] = {}
    local: dict[Location, Fraction] = {loc: Fraction(0) for loc in avail}
    for plant, demand in demands.items():
        d = _exact(demand, f"demand[{plant}]")
        if plant not in weights:
            raise ContractError("missing_weights", f"plant {plant!r} has no root weights")
        rows = {loc: _exact(w, f"weights[{plant}][{loc}]") for loc, w in weights[plant].items()}
        if not rows:
            raise ContractError("no_roots", f"plant {plant!r} has no rooted location")
        total = sum(rows.values(), Fraction(0))
        if abs(total - 1) > weight_tolerance:
            raise ContractError("weights_not_normalized", f"plant {plant!r} root weights sum to {total}, not 1")
        requests[plant] = {}
        for loc, w in rows.items():
            if loc not in avail:
                raise ContractError("unreported_location", f"plant {plant!r} requests water at {loc} not reported by SYRUP")
            r = d * w
            requests[plant][loc] = r
            local[loc] += r
    fulfillment = {loc: (min(Fraction(1), avail[loc] / q) if q > 0 else Fraction(0)) for loc, q in local.items()}
    accepted = {plant: {loc: r * fulfillment[loc] for loc, r in rows.items()} for plant, rows in requests.items()}
    debited = {loc: Fraction(0) for loc in avail}
    for rows in accepted.values():
        for loc, u in rows.items():
            debited[loc] += u
    for loc, taken in debited.items():
        if taken > avail[loc]:  # cannot happen with exact rationals; guards float-input misuse upstream
            raise ContractError("overdraw", f"debited {taken} exceeds available {avail[loc]} at {loc}")
    remaining = {loc: avail[loc] - debited[loc] for loc in avail}
    shortfall = {loc: local[loc] - debited[loc] for loc in avail}
    growth: dict[Any, Fraction | None] = {}
    for plant, demand in demands.items():
        d = Fraction(demand)
        growth[plant] = (sum(accepted[plant].values(), Fraction(0)) / d) if d > 0 else None
    return UptakeAllocation(requests, local, fulfillment, accepted, debited, remaining, shortfall, growth)
