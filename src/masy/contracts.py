"""Native resolver and API-surface evidence for the selected pair.

Three independent records, all retained in the report:

1. `native_resolver`: SYRUP's own `resolve_maple_dependency(expected MAPLE
   root)`, called unmodified. A `MapleApiIncompatibleError` is an API
   incompatibility; any other `MapleDependencyError` (for example the current
   `source_root=None` identity failure) or unexpected error is a concrete
   refusal. The guard is never bypassed or repaired.
2. `syrup_required_api`: SYRUP's `check_required_api()` (its own list).
3. `masy_required_api`: functions MASY's later adapters would build on
   (must be non-class callables with the named parameters), the shared
   state/handoff dataclasses (must be dataclass types declaring the named
   fields) and genuine classes (the public module attribute must bind a real
   class, including re-exports/aliases, e.g. an `isinstance` target of the
   observed SYRUP Dense path; module origins are checked separately),
   inspected by name only. Nothing is called or constructed and no
   configured value (e.g. the reroute trigger) is read. Passing is interface
   evidence, never physical or transport qualification. The SYRUP native list
   is narrower, so it can pass while this one fails.

Optional modules (the windowed paired transaction) are located with
`importlib.util.find_spec`, not imported; absence is informational and does
not fail the dense audit. A TYPE that the observed SYRUP imports on every bed
path (`WindowedVoxelColumnState`, finding 11) is a required class instead,
whatever its name: requiring it does not make Windowed storage required or
supported.

`crosscheck_provenance` compares the stdlib observation with SYRUP's
`source_tree_digest` / `scoped_git_status`; any disagreement refuses.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import inspect
from dataclasses import dataclass
from types import ModuleType
from typing import Any

__all__ = [
    "MASY_REQUIRED_API",
    "OPTIONAL_MODULES",
    "REQUIREMENT_KINDS",
    "Requirement",
    "check_masy_api",
    "check_optional_modules",
    "crosscheck_provenance",
    "run_native_audit",
]

MAX_MESSAGE = 4000


REQUIREMENT_KINDS = ("callable", "dataclass", "class")


@dataclass(frozen=True)
class Requirement:
    """`kind="callable"`: a non-class callable whose signature has every name
    in `parameters`. `kind="dataclass"`: a dataclass type declaring every
    name in `fields`. `kind="class"`: the public binding `module.attribute`
    is a class, whatever name it was defined under (re-exports and aliases
    pass; no parameters or fields). `required_by` names the observed caller, for the report. Names
    only; no call, construction, instance check or value read."""

    module: str
    attribute: str
    parameters: tuple[str, ...] = ()
    kind: str = "callable"
    fields: tuple[str, ...] = ()
    required_by: str = ""

    def __post_init__(self) -> None:
        if self.kind not in REQUIREMENT_KINDS:
            raise ValueError(f"Requirement.kind must be one of {REQUIREMENT_KINDS}, got {self.kind!r}")
        if self.kind != "callable" and self.parameters or self.kind != "dataclass" and self.fields:
            raise ValueError(f"{self.module}.{self.attribute}: parameters apply to callables, fields to dataclasses")


# Source-backed surface (see docs/preflight.md for the source lines). Exact
# names from the observed MAPLE/SYRUP sources; extend only when an adapter
# actually starts calling something. No threshold VALUE is read or frozen.
MASY_REQUIRED_API: tuple[Requirement, ...] = (
    Requirement("maple.aeolian.scheduler.block", "advance_wind_block", ("state", "block", "config")),
    Requirement("maple.aeolian.scheduler.state", "initial_scheduler_state", (
        "voxel_column", "active_layer", "mobile_state", "ledger", "committed_topography",
        "threshold_modifier_state", "time_s", "coupled", "vegetation", "sediment_availability", "water",
        "coupled_trajectory_semantics",
    )),
    Requirement("maple.aeolian.scheduler.state", "SchedulerPhysicalState", kind="dataclass", fields=(
        "voxel_column", "active_layer", "mobile_state", "ledger", "committed_topography",
        "threshold_modifier_state", "time_s", "vegetation", "sediment_availability", "water",
        "coupled_trajectory_semantics",
    )),
    Requirement("maple.aeolian.scheduler.state_validation", "validate_event_boundary", (
        "physical", "coupled", "fluvial", "geometry", "grain_classes",
    )),
    Requirement("maple.aeolian.cohorts.dispatch", "mobile_state_mass_is_finite", ("mobile_state",)),
    Requirement("maple.aeolian.cohorts.dispatch", "mobile_state_is_genuinely_empty", ("mobile_state",)),
    Requirement("maple.surface.topographic_commit.commit", "commit_topography", (
        "current_time_s", "water_callback", "routing_callback",
    )),
    Requirement("maple.core.parameters.topographic_commit", "TopographicCommitSpec", kind="dataclass",
                fields=("pending_elevation_trigger_m",)),
    Requirement("maple_syrup.sediment_bed", "BedState", kind="dataclass", fields=(
        "voxel_column", "active_layer", "water", "ledger", "committed_topography", "sediment_availability",
    )),
    Requirement("maple_syrup.sediment_bed", "BedContext", kind="dataclass", fields=(
        "geometry", "grain_classes", "mass_resolution_kg", "topographic_commit_spec", "initial_available_fraction",
    )),
    # Dense bed path (finding 11). SYRUP 9fabdca sediment_bed.py:610-613: `is_windowed_bed` runs
    # `from maple.core.types.voxel import WindowedVoxelColumnState` before looking at the column, and
    # bed_inventory / apply_bed_demand / bed_by_cell_class / bed_voxel_count / bed_already_committed all
    # call it first. So the type is required by the CURRENT SYRUP caller even for Dense beds; this does not
    # make Windowed storage required or supported. Re-derive if an owning-SYRUP change removes the import.
    Requirement("maple.core.types.voxel", "VoxelColumnState", kind="class",
                required_by="the genuine Dense voxel column type carried in BedState.voxel_column"),
    Requirement("maple.core.types.voxel", "WindowedVoxelColumnState", kind="class",
                required_by="maple_syrup.sediment_bed.is_windowed_bed (sediment_bed.py:610-613 at SYRUP 9fabdca) "
                            "imports it unconditionally on every Dense bed path; not a Windowed-storage requirement"),
    Requirement("maple_syrup.sediment_bed", "is_windowed_bed", ("state",)),
    Requirement("maple_syrup.sediment_bed", "bed_inventory", ("state",)),
    Requirement("maple_syrup.sediment_bed", "apply_bed_demand", ("state", "context", "demand")),
    Requirement("maple_syrup.sediment_bed", "bed_by_cell_class", ("state",)),
    Requirement("maple_syrup.sediment_bed", "bed_voxel_count", ("state",)),
    Requirement("maple_syrup.sediment_bed", "bed_already_committed", ("state", "context", "time_s")),
    Requirement("maple.aeolian.flux.step", "bed_inventory_by_class_kg", ("voxel_column", "active_layer"),
                required_by="maple_syrup.sediment_bed.bed_inventory (Dense branch)"),
    Requirement("maple.water", "apply_water_process_demand", (
        "voxel_column", "active_layer", "water", "ledger", "demand", "geometry", "grain_classes",
        "mass_resolution_kg", "sediment_availability", "initial_available_fraction", "adapter_name",
        "detachment_integration",
    ), required_by="maple_syrup.sediment_bed.apply_bed_demand (Dense branch)"),
    Requirement("maple.surface.topographic_commit.commit", "is_already_committed", (
        "ledger", "committed_state", "voxel_column", "active_layer", "geometry", "mass_resolution_kg",
        "current_time_s",
    ), required_by="maple_syrup.sediment_bed.bed_already_committed (Dense branch)"),
    Requirement("maple_syrup.sediment_bed", "reroute_threshold_m", ("context",)),
    Requirement("maple_syrup.sediment_bed", "current_routing_surface_m", ("state", "context")),
    Requirement("maple_syrup.sediment_bed", "net_routing_displacement_m", (
        "elevation_m", "routing_source_m", "active",
    )),
    Requirement("maple_syrup.sediment_bed", "commit_bed", (
        "state", "context", "graph", "reference", "time_s", "routing_source_m",
    )),
    Requirement("maple_syrup.sediment_event", "initial_event_state", (
        "graph", "terrain", "bed", "context", "soil_water_m", "t_s", "control",
    )),
    Requirement("maple_syrup.complete_event", "DryHandoff", kind="dataclass", fields=(
        "progress", "dry_state", "pre_reset_surface_m", "pre_reset_soil_m", "terminal_deposition_kg",
        "accounting", "status",
    )),
    Requirement("maple_syrup.complete_event", "complete_event", (
        "state", "context", "max_end_s", "control", "continuation",
    )),
    Requirement("maple_syrup.complete_event", "finalize_event", ("progress", "context", "column")),
)

OPTIONAL_MODULES: tuple[tuple[str, str], ...] = (
    ("maple.water.windowed_transaction", "paired windowed pickup/transport/deposit transaction"),
)


def _message(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:MAX_MESSAGE]


def check_masy_api(requirements: tuple[Requirement, ...] = MASY_REQUIRED_API) -> list[dict[str, Any]]:
    results = []
    for req in requirements:
        result: dict[str, Any] = {"module": req.module, "attribute": req.attribute, "kind": req.kind,
                                  "parameters": list(req.parameters), "fields": list(req.fields),
                                  "required_by": req.required_by}
        reason = _requirement_failure(req)
        results.append({**result, "status": "pass"} if reason is None else
                       {**result, "status": "fail", "reason": reason})
    return results


def _requirement_failure(req: Requirement) -> str | None:
    try:
        module = importlib.import_module(req.module)
    except Exception as exc:  # noqa: BLE001 - recorded verbatim
        return f"module import failed ({_message(exc)})"
    if not hasattr(module, req.attribute):
        return "attribute missing"
    obj = getattr(module, req.attribute)
    if req.kind == "class":
        return None if isinstance(obj, type) else f"not a class (got {type(obj).__name__})"
    if req.kind == "dataclass":
        if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
            return f"not a dataclass type (got {type(obj).__name__})"
        declared = {item.name for item in dataclasses.fields(obj)}
        missing = [name for name in req.fields if name not in declared]
        return f"missing field(s) {missing}" if missing else None
    if isinstance(obj, type) or not callable(obj):
        return f"not a callable function (got {type(obj).__name__})"
    try:
        present = set(inspect.signature(obj).parameters)
    except (TypeError, ValueError) as exc:
        return f"signature unavailable ({exc})"
    missing = [name for name in req.parameters if name not in present]
    return f"missing parameter(s) {missing}" if missing else None


def check_optional_modules(optional: tuple[tuple[str, str], ...] = OPTIONAL_MODULES) -> list[dict[str, Any]]:
    results = []
    for name, purpose in optional:
        try:
            spec = importlib.util.find_spec(name)
        except Exception as exc:  # noqa: BLE001
            results.append({"module": name, "purpose": purpose, "status": "lookup_error", "reason": _message(exc)})
            continue
        results.append({"module": name, "purpose": purpose, "status": "present_not_imported" if spec else "absent",
                        "origin": getattr(spec, "origin", None)})
    return results


def run_native_audit(
    dependency: ModuleType,
    maple_root: str,
    *,
    masy_requirements: tuple[Requirement, ...] = MASY_REQUIRED_API,
    optional_modules: tuple[tuple[str, str], ...] = OPTIONAL_MODULES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolver, SYRUP API check, MASY API check and optional modules.

    Every part runs regardless of the others; returns (record, findings).
    """
    findings: list[dict[str, Any]] = []
    record: dict[str, Any] = {}

    resolver: dict[str, Any] = {"call": "resolve_maple_dependency(expected_source_root)",
                                "expected_source_root": maple_root}
    try:
        resolved = dependency.resolve_maple_dependency(maple_root)
    except dependency.MapleApiIncompatibleError as exc:
        resolver.update(status="api_incompatible", error=_message(exc))
        findings.append({"code": "native_resolver_api_incompatible", "severity": "incompatible",
                         "detail": resolver["error"]})
    except dependency.MapleDependencyError as exc:
        resolver.update(status="refused", error=_message(exc))
        findings.append({"code": "native_resolver_refused", "severity": "refusal", "detail": resolver["error"]})
    except Exception as exc:  # noqa: BLE001
        resolver.update(status="error", error=_message(exc))
        findings.append({"code": "native_resolver_error", "severity": "refusal", "detail": resolver["error"]})
    else:
        resolver.update(status="resolved", dependency=resolved.as_record())
        if resolver["dependency"].get("source_root") != maple_root:
            findings.append({"code": "native_resolver_root_mismatch", "severity": "refusal",
                             "detail": f"resolver returned source_root {resolver['dependency'].get('source_root')}"})
    record["native_resolver"] = resolver

    syrup_api: dict[str, Any] = {"call": "check_required_api()"}
    try:
        dependency.check_required_api()
    except dependency.MapleApiIncompatibleError as exc:
        syrup_api.update(status="fail", error=_message(exc))
        findings.append({"code": "syrup_required_api_missing", "severity": "incompatible",
                         "detail": syrup_api["error"]})
    except Exception as exc:  # noqa: BLE001
        syrup_api.update(status="error", error=_message(exc))
        findings.append({"code": "syrup_required_api_error", "severity": "refusal", "detail": syrup_api["error"]})
    else:
        syrup_api.update(status="pass", requirement_count=len(dependency.REQUIRED_MAPLE_API))
    record["syrup_required_api"] = syrup_api

    masy_api = check_masy_api(masy_requirements)
    record["masy_required_api"] = masy_api
    failed = [f"{item['module']}.{item['attribute']}: {item['reason']}"
              + (f" (required by {item['required_by']})" if item["required_by"] else "")
              for item in masy_api if item["status"] != "pass"]
    if failed:
        findings.append({"code": "masy_required_api_missing", "severity": "incompatible",
                         "detail": "; ".join(failed)[:MAX_MESSAGE]})

    optional = check_optional_modules(optional_modules)
    record["optional_modules"] = optional
    for item in optional:
        findings.append({"code": f"optional_module_{item['status']}", "severity": "info",
                         "detail": f"{item['module']} ({item['purpose']}); not required by the dense audit"})
    record["interpretation"] = "interface evidence only; no physical, restart or transport qualification"
    return record, findings


def crosscheck_provenance(provenance: ModuleType, observation: dict[str, Any], *,
                          max_files: int, max_bytes: int, git_timeout_s: float) -> tuple[dict[str, Any], list]:
    """Compare stdlib observation with SYRUP's provenance helpers."""
    record: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    for name in ("maple", "syrup"):
        observed = observation[name]
        entry: dict[str, Any] = {}
        try:
            digest = provenance.source_tree_digest(observed["package"]["dir"], max_files=max_files,
                                                   max_bytes=max_bytes)
            entry["source_tree_digest"] = digest.as_record()
            entry["scoped_git_status"] = provenance.scoped_git_status(
                observed["root"], ("src", "pyproject.toml"), timeout_s=git_timeout_s)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = _message(exc)
            findings.append({"code": "upstream_provenance_error", "severity": "refusal",
                             "detail": f"{name}: {entry['error']}"})
            record[name] = entry
            continue
        package = observed["package"]
        disagreements = [
            key for key, theirs in (("digest_sha256", digest.digest_sha256), ("file_count", digest.file_count),
                                    ("total_bytes", digest.total_bytes), ("algorithm", digest.algorithm))
            if package[key] != theirs
        ]
        if entry["scoped_git_status"].get("status") != "clean":
            disagreements.append(f"scoped_git_status={entry['scoped_git_status'].get('status')}")
        entry["agrees"] = not disagreements
        if disagreements:
            findings.append({"code": "upstream_provenance_disagreement", "severity": "refusal",
                             "detail": f"{name}: {disagreements}"})
        record[name] = entry
    return record, findings
