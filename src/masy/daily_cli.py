"""Read-only `inspect` CLI for the daily scenario CSV format (metadata only).

    python -I -B tools/masy_daily.py inspect SCENARIO.csv [--sample-days N] [quota flags]

Prints bounded JSON: identity, settings, weather summary, wind occurrence list
(capped), plan summary and the truthful limitations. It never imports a native
model, runs nothing and creates nothing. Exit codes: 0 inspected, 2 refused
(JSON with a stable `error_code`), 1 internal error.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, TextIO

from masy.daily_csv import DEFAULT_LIMITS, DailyCsvError, DailyCsvLimits, load_daily_scenario
from masy.daily_schedule import build_daily_plan
from masy.scenario_io import ScenarioIOError

__all__ = ["main"]

SCHEMA = "masy.daily_inspect.v1"
MAX_LISTED_EVENTS = 64
MAX_SAMPLE_DAYS = 366
LIMITATIONS = (
    "Input delivery only: CSV/YAML metadata was read and validated; no simulation was run.",
    "No physical transport, soil/E update, transpiration or growth occurred; nothing here predicts physical durations or budgets.",
    "Wind YAML scalars were checked with existing MASY declaration rules; native MAPLE wind runtime validation and execution are a separate adapter qualification.",
    "PET is not computed in MASY; native SYRUP owns PET (Hargreaves latitude stays in the native MAPLE case).",
    "Precipitation selection is the intended request only; native SYRUP Phase 1.5 decides capacity fallback, recession and acceptance.",
    "Enabled transpiration/growth require the future LEAF connection; no no-op model is substituted and no LEAF adapter exists in this stage.",
    "Native solver settings remain in the owner configs.",
)
EXECUTION = {"executed": False, "native_models_imported": False, "physical_transport_occurred": False,
             "leaf_adapter_available": False, "execution_ready": False,
             "qualification": "none; input metadata and nominal plan only"}


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:  # noqa: ANN401
        raise DailyCsvError("invalid_arguments", message)


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="masy_daily", allow_abbrev=False,
                     description="MASY daily scenario CSV tools (read-only metadata; no model is run)")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    inspect = sub.add_parser("inspect", allow_abbrev=False, help="validate a scenario CSV and print bounded JSON")
    inspect.add_argument("scenario", help="path to a masy.daily_scenario.v1 CSV")
    inspect.add_argument("--sample-days", type=int, default=0,
                         help=f"also print the first N plan days (0..{MAX_SAMPLE_DAYS}; default 0)")
    for name, default in vars(DEFAULT_LIMITS).items():
        inspect.add_argument("--" + name.replace("_", "-"), dest=name, type=_positive, default=default,
                             help=f"input quota (default {default})")
    return parser


def _inspect(args: argparse.Namespace) -> dict[str, Any]:
    if not 0 <= args.sample_days <= MAX_SAMPLE_DAYS:
        raise DailyCsvError("invalid_arguments", f"--sample-days must be within 0..{MAX_SAMPLE_DAYS}")
    limits = DailyCsvLimits(**{name: getattr(args, name) for name in vars(DEFAULT_LIMITS)})
    scenario = load_daily_scenario(args.scenario, limits)
    plan = build_daily_plan(scenario)
    events = [{"index": o.index, "occurrence_id": o.occurrence_id, "date": o.date.isoformat(),
               "reference": o.reference, "sha256": o.sha256, "block_count": len(o.blocks),
               "declared_wind_duration_s": o.declared_duration_s}
              for o in scenario.occurrences[:MAX_LISTED_EVENTS]]
    sample = []
    if args.sample_days:
        for day in plan.days():
            if len(sample) >= args.sample_days:
                break
            # A sampled day can contain every repeated row in a large scenario;
            # keep both the event list and block details out of this summary.
            record = day.record(include_winds=False)
            record["wind_occurrence_count"] = len(day.winds)
            record["wind_occurrences"] = [w.occurrence_id for w in day.winds[:MAX_LISTED_EVENTS]]
            record["wind_occurrences_truncated"] = len(day.winds) > MAX_LISTED_EVENTS
            sample.append(record)
    return {"schema": SCHEMA, "status": "inspected", **scenario.summary(),
            "wind_events": {"count": len(scenario.occurrences), "listed": events,
                            "truncated": len(scenario.occurrences) > MAX_LISTED_EVENTS},
            "plan": plan.summary(), "sample_days": sample,
            "execution": dict(EXECUTION), "limitations": list(LIMITATIONS)}


def _emit(stream: TextIO, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main(argv: list[str], *, stdout: TextIO | None = None) -> int:
    out = stdout if stdout is not None else sys.stdout
    try:
        args = _build_parser().parse_args(argv)
        _emit(out, _inspect(args))
        return 0
    except SystemExit as exc:  # --help only; argument errors are raised as DailyCsvError
        return int(exc.code or 0)
    except ScenarioIOError as exc:
        _emit(out, {"schema": SCHEMA, "status": "refused", "error_code": exc.code, "detail": str(exc.detail)[:1000],
                    "execution": dict(EXECUTION)})
        return 2
