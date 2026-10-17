"""Strict closed-schema preflight configuration (TOML, stdlib only).

Unknown keys refuse. Calendar/absolute-time fields are refused by exact key
name with a specific `unsupported_calendar_field` code at any table level;
there is no substring matching, so IDs and paths containing dates are valid.
Duration budgets and forcing edges are decimal strings kept exactly
(`fractions.Fraction`); nothing is rescaled.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import tomllib
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from masy.sourceobs import Limits

__all__ = [
    "SCHEMA",
    "UNSUPPORTED_CALENDAR_FIELDS",
    "CaseInput",
    "ConfigError",
    "PreflightConfig",
    "UpstreamSource",
    "load_config",
    "parse_config",
]

SCHEMA = "masy.preflight.v1"
MAX_CONFIG_BYTES = 1024 * 1024
UNSUPPORTED_CALENDAR_FIELDS = frozenset({
    "start_datetime", "start_date", "calendar", "timezone", "epoch_utc", "absolute_start_s", "fixed_calendar",
})
# name -> (import package, pyproject project name). Fixed, not configurable.
UPSTREAMS = {"maple": ("maple", "maple"), "syrup": ("maple_syrup", "maple-syrup")}
_DECIMAL = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

_TOP_KEYS = {"schema", "sources", "limits", "budgets", "case", "clock", "forcing"}
_SOURCE_KEYS = {"root", "expected_revision"}
_LIMIT_KEYS = {"git_timeout_s", "git_max_output_bytes", "max_files", "max_bytes", "max_report_bytes"}
_BUDGET_KEYS = {"wind_settlement_max_s", "wind_settlement_max_steps", "water_max_end_s", "run_disk_bytes"}
_CASE_KEYS = {"id", "inputs"}
_CASE_INPUT_KEYS = {"name", "path", "sha256"}
_CLOCK_KEYS = {"native_epoch_s"}
_FORCING_KEYS = {"id", "edges_s"}


class ConfigError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class UpstreamSource:
    name: str
    root: str
    package: str
    project_name: str
    expected_revision: str | None


@dataclass(frozen=True)
class CaseInput:
    name: str
    path: str
    declared_sha256: str | None


@dataclass(frozen=True)
class PreflightConfig:
    path: str | None
    sha256: str | None
    masy_root: str
    upstreams: tuple[UpstreamSource, ...]
    limits: Limits
    max_report_bytes: int
    budgets: dict[str, Any] | None
    case_id: str | None
    case_inputs: tuple[CaseInput, ...]
    native_epoch_s: float
    forcing_id: str | None
    forcing_edges_s: tuple[str, ...] | None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def upstream(self, name: str) -> UpstreamSource:
        return next(source for source in self.upstreams if source.name == name)


def _check_keys(table: Any, allowed: set[str], where: str) -> dict[str, Any]:
    if not isinstance(table, dict):
        raise ConfigError("invalid_config_type", f"{where} must be a table")
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError("unknown_config_field", f"{where} has unknown field(s) {unknown}")
    return table


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or _CONTROL.search(value):
        raise ConfigError("invalid_config_type", f"{where} must be a non-empty string without control characters")
    return value


def _absolute_path(value: Any, where: str) -> str:
    text = _text(value, where)
    if not os.path.isabs(text):
        raise ConfigError("invalid_config_value", f"{where} must be an absolute path")
    return text


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError("invalid_config_value", f"{where} must be a positive integer")
    return value


def _positive_seconds(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConfigError("invalid_config_value", f"{where} must be a positive finite number")
    return float(value)


def _positive_decimal(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _DECIMAL.match(value) or Fraction(value) <= 0:
        raise ConfigError("invalid_config_value", f"{where} must be a positive plain decimal string, e.g. \"3600\"")
    return value


def _refuse_calendar_fields(value: Any, where: str) -> None:
    """Exact-key scan of every nested table first, so the specific refusal is
    reported even when other parts of the file are also invalid."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in UNSUPPORTED_CALENDAR_FIELDS:
                raise ConfigError(
                    "unsupported_calendar_field",
                    f"{where}.{key} is a calendar/absolute-time field; fixed-calendar input is not supported "
                    "in this schema",
                )
            _refuse_calendar_fields(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _refuse_calendar_fields(item, f"{where}[{index}]")


def parse_config(raw: dict[str, Any], *, masy_root: str, path: str | None = None,
                 sha256: str | None = None) -> PreflightConfig:
    _refuse_calendar_fields(raw, "<top>")
    _check_keys(raw, _TOP_KEYS, "<top>")
    if raw.get("schema") != SCHEMA:
        raise ConfigError("invalid_config_value", f"schema must be {SCHEMA!r}")

    sources = raw.get("sources")
    if not isinstance(sources, dict):
        raise ConfigError("invalid_config_type", "[sources] must be a table")
    _check_keys(sources, set(UPSTREAMS), "sources")
    upstreams = []
    for name, (package, project_name) in UPSTREAMS.items():
        if name not in sources:
            raise ConfigError("missing_config_field", f"[sources.{name}] is required")
        table = _check_keys(sources[name], _SOURCE_KEYS, f"sources.{name}")
        if "root" not in table:
            raise ConfigError("missing_config_field", f"sources.{name}.root is required")
        revision = table.get("expected_revision")
        if revision is not None and (not isinstance(revision, str) or not _OBJECT_ID.match(revision)):
            raise ConfigError("invalid_config_value", f"sources.{name}.expected_revision must be a full object id")
        upstreams.append(UpstreamSource(
            name, _absolute_path(table["root"], f"sources.{name}.root"), package, project_name, revision,
        ))

    limits_table = _check_keys(raw.get("limits", {}), _LIMIT_KEYS, "limits")
    defaults = Limits()
    limits = Limits(
        git_timeout_s=_positive_seconds(limits_table.get("git_timeout_s", defaults.git_timeout_s),
                                        "limits.git_timeout_s"),
        git_max_output_bytes=_positive_int(limits_table.get("git_max_output_bytes", defaults.git_max_output_bytes),
                                           "limits.git_max_output_bytes"),
        max_files=_positive_int(limits_table.get("max_files", defaults.max_files), "limits.max_files"),
        max_bytes=_positive_int(limits_table.get("max_bytes", defaults.max_bytes), "limits.max_bytes"),
    )
    max_report_bytes = _positive_int(limits_table.get("max_report_bytes", 16 * 1024 * 1024),
                                     "limits.max_report_bytes")

    budgets = None
    if "budgets" in raw:
        table = _check_keys(raw["budgets"], _BUDGET_KEYS, "budgets")
        budgets = {}
        for key in sorted(table):
            if key.endswith("_s"):
                budgets[key] = _positive_decimal(table[key], f"budgets.{key}")
            else:
                budgets[key] = _positive_int(table[key], f"budgets.{key}")

    case_id, case_inputs = None, []
    if "case" in raw:
        case = _check_keys(raw["case"], _CASE_KEYS, "case")
        case_id = _text(case.get("id"), "case.id")
        inputs = case.get("inputs", [])
        if not isinstance(inputs, list):
            raise ConfigError("invalid_config_type", "case.inputs must be an array of tables")
        seen = set()
        for index, item in enumerate(inputs):
            where = f"case.inputs[{index}]"
            item = _check_keys(item, _CASE_INPUT_KEYS, where)
            name = _text(item.get("name"), f"{where}.name")
            if name in seen:
                raise ConfigError("invalid_config_value", f"{where}.name {name!r} is duplicated")
            seen.add(name)
            declared = item.get("sha256")
            if declared is not None and (not isinstance(declared, str) or not _HEX64.match(declared)):
                raise ConfigError("invalid_config_value", f"{where}.sha256 must be 64 lowercase hex digits")
            case_inputs.append(CaseInput(name, _absolute_path(item.get("path"), f"{where}.path"), declared))

    clock = _check_keys(raw.get("clock", {}), _CLOCK_KEYS, "clock")
    epoch = clock.get("native_epoch_s", 0.0)
    if isinstance(epoch, bool) or not isinstance(epoch, (int, float)) or not math.isfinite(epoch):
        raise ConfigError("invalid_config_value", "clock.native_epoch_s must be a finite number")
    if isinstance(epoch, int) and float(epoch) != epoch:
        raise ConfigError("invalid_config_value", "clock.native_epoch_s is not exactly representable as binary64")
    if epoch < 0 or math.copysign(1.0, float(epoch)) < 0:
        raise ConfigError("invalid_config_value",
                          "clock.native_epoch_s must be non-negative and not -0.0 (native clocks are non-negative)")

    forcing_id, edges = None, None
    if "forcing" in raw:
        forcing = _check_keys(raw["forcing"], _FORCING_KEYS, "forcing")
        forcing_id = _text(forcing.get("id"), "forcing.id")
        values = forcing.get("edges_s")
        if not isinstance(values, list) or len(values) < 2 or not all(isinstance(v, str) for v in values):
            raise ConfigError("invalid_config_value", "forcing.edges_s must list at least two decimal strings")
        edges = tuple(values)

    return PreflightConfig(
        path=path, sha256=sha256, masy_root=masy_root, upstreams=tuple(upstreams), limits=limits,
        max_report_bytes=max_report_bytes, budgets=budgets, case_id=case_id, case_inputs=tuple(case_inputs),
        native_epoch_s=float(epoch), forcing_id=forcing_id, forcing_edges_s=edges, raw=raw,
    )


def load_config(path: str, *, masy_root: str) -> PreflightConfig:
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            data = handle.read(MAX_CONFIG_BYTES + 1)
    except OSError as exc:
        raise ConfigError("config_unreadable", f"{path}: {exc}") from exc
    if len(data) > MAX_CONFIG_BYTES:
        raise ConfigError("config_unreadable", f"{path} is larger than {MAX_CONFIG_BYTES} bytes")
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError("config_unreadable", f"{path}: {exc}") from exc
    return parse_config(raw, masy_root=masy_root, path=os.path.abspath(path),
                        sha256=hashlib.sha256(data).hexdigest())
