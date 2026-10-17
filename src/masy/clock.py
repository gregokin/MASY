"""Exact global/native clock ledger and forcing-edge resolution (data only).

Native solver times are binary64 values and are never recomputed, rounded
or rewritten. Native clocks and commit stamps are non-negative (the native
solvers require it); negative values and negative zero refuse. Global
elapsed is the exact rational difference `G(t) = Fraction(t) - Fraction(T0)`
from a (possibly nonzero) native epoch `T0`, so `G(T0) == 0` and
`native_from_global(G(t)) == t` bit for bit for every accepted `t`. A
rational value alone cannot distinguish +0.0 from -0.0; that is why -0.0 is
refused rather than silently rewritten to +0.0, and why serialized times
keep the binary64 hex next to the fraction.

These helpers validate a ledger of realized intervals; they do not run or
certify any solver adapter. See docs/clock_contract.md.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

__all__ = [
    "ENTRY_KINDS",
    "ClockError",
    "ClockLedger",
    "decode_time",
    "encode_time",
    "resolve_forcing_edges",
]

ENTRY_KINDS = frozenset({"event", "idle"})
LEDGER_SCHEMA = "masy.clock_ledger.v1"
GLOBAL_ELAPSED_DEFINITION = "Fraction(native_t) - Fraction(native_epoch)"
_DECIMAL = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
_LEDGER_KEYS = frozenset({"schema", "native_epoch", "global_elapsed_definition", "entries"})
_ENTRY_KEYS = frozenset({
    "index", "kind", "native_start", "native_end", "global_start", "global_end", "endpoint_elapsed",
    "commit_before", "commit_after", "returned_elapsed", "endpoint_discrepancy",
})


class ClockError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _native(value: Any, name: str) -> float:
    """Accept a finite float (or an int exactly representable as one); never a bool."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClockError("invalid_time_value", f"{name} must be a float, got {type(value).__name__}")
    if isinstance(value, int):
        converted = float(value)
        if converted != value:
            raise ClockError("invalid_time_value", f"{name}={value} is not exactly representable as binary64")
        value = converted
    if not math.isfinite(value):
        raise ClockError("invalid_time_value", f"{name} must be finite, got {value!r}")
    return value


def _native_clock(value: Any, name: str) -> float:
    """A native clock value, commit stamp or duration: finite and >= +0.0."""
    value = _native(value, name)
    if value < 0:
        raise ClockError("negative_time_value", f"{name}={value!r} is negative; native clocks are non-negative")
    if math.copysign(1.0, value) < 0:
        raise ClockError("negative_zero_time", f"{name} is -0.0; refused rather than rewritten to +0.0")
    return value


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def encode_time(value: float) -> dict[str, str]:
    """Lossless record: binary64 hex plus its exact rational value."""
    value = _native(value, "value")
    return {"hex": value.hex(), "fraction": _fraction_text(Fraction(value))}


def decode_time(record: Any) -> float:
    """Inverse of `encode_time`; the hex must be canonical and agree with the fraction."""
    if not isinstance(record, dict) or set(record) != {"hex", "fraction"} or not all(
            isinstance(record[key], str) for key in ("hex", "fraction")):
        raise ClockError("malformed_time_record", f"expected {{'hex': str, 'fraction': str}}, got {record!r:.200}")
    try:
        value = float.fromhex(record["hex"])
    except (ValueError, OverflowError) as exc:
        raise ClockError("malformed_time_record", f"hex {record['hex']!r:.100} is not a binary64 value") from exc
    if not math.isfinite(value) or value.hex() != record["hex"]:
        raise ClockError("malformed_time_record", f"hex {record['hex']!r:.100} is not canonical finite binary64")
    if _fraction_text(Fraction(value)) != record["fraction"]:
        raise ClockError("inconsistent_time_record", f"hex {record['hex']} does not equal {record['fraction']}")
    return value


def _same_bits(a: float, b: float) -> bool:
    return a.hex() == b.hex()


@dataclass(frozen=True)
class _Entry:
    kind: str
    native_start: float
    native_end: float
    commit_before: float
    commit_after: float
    returned_elapsed: float | None


class ClockLedger:
    """Contiguous sequence of realized intervals on one monotone native epoch.

    Invariants enforced by `append` (each violation raises ClockError):

    - the first entry starts exactly at the epoch; each next entry starts
      exactly (bit for bit) at the preceding entry's end; gaps exist only as
      explicit `idle` entries;
    - `native_end >= native_start`;
    - `commit_before <= native_start` and `commit_after <= native_end`
      (no commit stamp from the future);
    - commit stamps are monotone: `commit_after >= commit_before`, and each
      entry's `commit_before` equals the preceding `commit_after` (an idle
      interval performs no commit, so its stamps are equal);
    - every value is finite, not a bool, non-negative and not -0.0
      (`negative_time_value` / `negative_zero_time`).

    `returned_elapsed` (an upstream-reported duration) is stored separately;
    its exact discrepancy from the endpoint difference is recorded and never
    used to correct the interval.
    """

    def __init__(self, native_epoch_s: float) -> None:
        self.epoch = _native_clock(native_epoch_s, "native_epoch_s")
        self._entries: list[_Entry] = []

    def global_elapsed(self, native_t: float) -> Fraction:
        """Exact offset of a native clock value from the epoch; may be negative
        for a valid native time earlier than the epoch."""
        return Fraction(_native_clock(native_t, "native_t")) - Fraction(self.epoch)

    def native_from_global(self, global_elapsed: Fraction) -> float:
        """Inverse of `global_elapsed`: the result must be an exactly
        representable native clock value (finite, >= +0.0)."""
        exact = Fraction(self.epoch) + Fraction(global_elapsed)
        value = float(exact)
        if Fraction(value) != exact:
            raise ClockError("not_a_native_value", f"epoch + {global_elapsed} is not a binary64 native time")
        return _native_clock(value, "native_from_global result")

    def append(self, kind: str, *, native_start: float, native_end: float, commit_before: float,
               commit_after: float, returned_elapsed: float | None = None) -> dict[str, Any]:
        if not isinstance(kind, str) or kind not in ENTRY_KINDS:
            raise ClockError("invalid_entry_kind", f"kind must be one of {sorted(ENTRY_KINDS)}, got {kind!r}")
        start = _native_clock(native_start, "native_start")
        end = _native_clock(native_end, "native_end")
        before = _native_clock(commit_before, "commit_before")
        after = _native_clock(commit_after, "commit_after")
        returned = None if returned_elapsed is None else _native_clock(returned_elapsed, "returned_elapsed")

        expected_start = self._entries[-1].native_end if self._entries else self.epoch
        if not _same_bits(start, expected_start):
            raise ClockError(
                "non_contiguous_interval",
                f"native_start {start.hex()} must equal {expected_start.hex()}; record gaps as idle entries",
            )
        if end < start:
            raise ClockError("reversed_interval", f"native_end {end.hex()} precedes native_start {start.hex()}")
        if before > start:
            raise ClockError("future_commit", f"commit_before {before.hex()} is after native_start {start.hex()}")
        if after > end:
            raise ClockError("future_commit", f"commit_after {after.hex()} is after native_end {end.hex()}")
        if after < before:
            raise ClockError("non_monotone_commit", f"commit_after {after.hex()} precedes commit_before")
        if self._entries and not _same_bits(before, self._entries[-1].commit_after):
            raise ClockError(
                "non_monotone_commit",
                f"commit_before {before.hex()} must equal the preceding commit_after "
                f"{self._entries[-1].commit_after.hex()}",
            )
        if kind == "idle" and not _same_bits(before, after):
            raise ClockError("idle_commit", "an idle interval performs no commit; commit stamps must be equal")
        self._entries.append(_Entry(kind, start, end, before, after, returned))
        return self.records()[-1]

    def records(self) -> list[dict[str, Any]]:
        out = []
        for index, entry in enumerate(self._entries):
            endpoint_elapsed = Fraction(entry.native_end) - Fraction(entry.native_start)
            record: dict[str, Any] = {
                "index": index,
                "kind": entry.kind,
                "native_start": encode_time(entry.native_start),
                "native_end": encode_time(entry.native_end),
                "global_start": _fraction_text(self.global_elapsed(entry.native_start)),
                "global_end": _fraction_text(self.global_elapsed(entry.native_end)),
                "endpoint_elapsed": _fraction_text(endpoint_elapsed),
                "commit_before": encode_time(entry.commit_before),
                "commit_after": encode_time(entry.commit_after),
                "returned_elapsed": None,
                "endpoint_discrepancy": None,
            }
            if entry.returned_elapsed is not None:
                record["returned_elapsed"] = encode_time(entry.returned_elapsed)
                record["endpoint_discrepancy"] = _fraction_text(Fraction(entry.returned_elapsed) - endpoint_elapsed)
            out.append(record)
        return out

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": LEDGER_SCHEMA,
            "native_epoch": encode_time(self.epoch),
            "global_elapsed_definition": GLOBAL_ELAPSED_DEFINITION,
            "entries": self.records(),
        }

    @classmethod
    def from_record(cls, record: Any) -> "ClockLedger":
        """Rebuild and re-validate a serialized ledger (lossless round trip).

        The schema and the global-elapsed definition must be exactly the
        declared ones (a record cannot redefine its own meaning); key sets
        are closed; every redundant derived field must equal its recomputed
        value. Any malformed record raises ClockError."""
        if not isinstance(record, dict) or set(record) != _LEDGER_KEYS:
            raise ClockError("malformed_ledger_record", f"ledger record keys must be exactly {sorted(_LEDGER_KEYS)}")
        if record["schema"] != LEDGER_SCHEMA:
            raise ClockError("unsupported_ledger_schema", f"schema {record['schema']!r:.100} is not {LEDGER_SCHEMA!r}")
        if record["global_elapsed_definition"] != GLOBAL_ELAPSED_DEFINITION:
            raise ClockError(
                "unsupported_global_elapsed_definition",
                f"{record['global_elapsed_definition']!r:.200} is not {GLOBAL_ELAPSED_DEFINITION!r}",
            )
        entries = record["entries"]
        if not isinstance(entries, list) or not all(
                isinstance(entry, dict) and set(entry) == _ENTRY_KEYS for entry in entries):
            raise ClockError("malformed_ledger_record", f"entries must be a list of objects with keys "
                                                        f"{sorted(_ENTRY_KEYS)}")
        ledger = cls(decode_time(record["native_epoch"]))
        for entry in entries:
            returned = entry["returned_elapsed"]
            ledger.append(
                entry["kind"],
                native_start=decode_time(entry["native_start"]),
                native_end=decode_time(entry["native_end"]),
                commit_before=decode_time(entry["commit_before"]),
                commit_after=decode_time(entry["commit_after"]),
                returned_elapsed=None if returned is None else decode_time(returned),
            )
        try:  # canonical JSON comparison: also rejects bool-for-int and other type substitutions
            same = _canonical(ledger.as_record()) == _canonical(record)
        except (TypeError, ValueError) as exc:
            raise ClockError("malformed_ledger_record", f"record is not plain JSON data ({exc})") from exc
        if not same:
            raise ClockError("inconsistent_ledger_record", "derived fields differ from the serialized ledger")
        return ledger


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def resolve_forcing_edges(native_start_s: float, edges_s: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Record intended decimal edges and the native binary64 boundaries
    `float(native_start + float(edge))` they resolve to.

    Inexact float sums (e.g. 0.1 + 0.2) are accepted and their exact offset
    from the intended value is recorded; nothing is retimed or rescaled.
    Non-decimal, non-finite, collapsed (equal successive) or reversed edges
    refuse. The origin counts as a predecessor: a positive first edge whose
    native time equals `native_start` (e.g. start 2**53, edge "1") would lose
    the origin-to-first-edge interval and refuses. Whether a solver realizes
    these boundaries is not qualified here.
    """
    start = _native_clock(native_start_s, "native_start_s")
    if len(edges_s) < 2:
        raise ClockError("invalid_forcing_edges", "at least two edges are required")
    intended: list[dict[str, Any]] = []
    previous_exact: Fraction | None = None
    previous_native: float | None = None
    for index, edge in enumerate(edges_s):
        if not isinstance(edge, str) or not _DECIMAL.match(edge):
            raise ClockError("invalid_forcing_edges", f"edge {index} {edge!r} is not a plain non-negative decimal")
        exact = Fraction(edge)
        native = start + float(exact)
        if not math.isfinite(native):
            raise ClockError("invalid_forcing_edges", f"edge {index} resolves to a non-finite native time")
        if previous_exact is None:
            if exact > 0 and native <= start:
                raise ClockError("collapsed_forcing_edges",
                                 f"edge 0 {edge!r} collapses onto the origin {start.hex()} natively")
        else:
            if exact < previous_exact or native < previous_native:
                raise ClockError("reversed_forcing_edges", f"edge {index} {edge!r} precedes edge {index - 1}")
            if exact == previous_exact or native == previous_native:
                raise ClockError("collapsed_forcing_edges", f"edge {index} {edge!r} collapses onto edge {index - 1}")
        offset = Fraction(native) - (Fraction(start) + exact)
        intended.append({
            "index": index,
            "intended_decimal": edge,
            "intended_fraction": _fraction_text(exact),
            "native": encode_time(native),
            "native_minus_intended": _fraction_text(offset),
        })
        previous_exact, previous_native = exact, native
    return {
        "native_start": encode_time(start),
        "resolution_rule": "float(native_start + float(Fraction(edge)))",
        "edges": intended,
        "adapter_status": "forcing_adapter_unqualified",
    }
