"""MASY-01 preflight orchestration. Entered only through tools/masy_preflight.py.

Stage order and gating:

1. strict config (refusal skips everything below except publication);
2. MASY module origin;
3. stdlib source observation of MASY, MAPLE and SYRUP, then the optional
   lock; any failure skips all upstream imports;
4. upstream imports and module origins;
5. native resolver, SYRUP API check, MASY API check, optional modules
   (a resolver failure does not skip the API checks);
6. SYRUP provenance cross-check, final origin sweep, distribution metadata;
7. environment record;
8. post-audit re-observation, run whenever the first observation succeeded;
9. data-only clock/forcing, budgets and case-input declarations.

The report is published exclusively into the new run directory. If not even
the minimal refusal report fits `max_report_bytes`, nothing is published and
the exit is 2. There is no transport entry point and no success verdict.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any

from masy.clock import ClockError, encode_time, resolve_forcing_edges
from masy.config import UNSUPPORTED_CALENDAR_FIELDS, ConfigError, PreflightConfig, load_config
from masy.contracts import crosscheck_provenance, run_native_audit
from masy.importcheck import (
    OriginError, distribution_records, environment_record, import_upstreams, module_origin_violations,
)
from masy.report import EXIT_CODES, ReportTooLarge, publish_report, render_report
from masy.sourceobs import SourceRefusal, compare_lock, diff_observations, load_lock, observe_sources

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="masy_preflight", description="MASY-01 read-only source preflight",
                                     allow_abbrev=False)
    parser.add_argument("--config", required=True, help="preflight TOML (schema masy.preflight.v1)")
    parser.add_argument("--run-dir", required=True, help="new run directory (created exclusively by the launcher)")
    parser.add_argument("--expect-lock", help="JSON copy of a previous report's source_observation block")
    return parser


def _refusal(code: str, detail: str, **evidence: Any) -> dict[str, Any]:
    finding: dict[str, Any] = {"code": code, "severity": "refusal", "detail": detail}
    if evidence:
        finding["evidence"] = evidence
    return finding


def _clock_section(config: PreflightConfig, findings: list[dict[str, Any]]) -> dict[str, Any]:
    section: dict[str, Any] = {
        "contract": "docs/clock_contract.md",
        "native_epoch": encode_time(config.native_epoch_s),
        "unsupported_calendar_fields": sorted(UNSUPPORTED_CALENDAR_FIELDS),
        "forcing": None,
    }
    if config.forcing_edges_s is not None:
        try:
            resolved = resolve_forcing_edges(config.native_epoch_s, config.forcing_edges_s)
        except ClockError as exc:
            findings.append(_refusal("forcing_edges_refused", f"{exc.code}: {exc.detail}"))
        else:
            section["forcing"] = {"id": config.forcing_id,
                                  "resolved_at": "native_epoch (data-only illustration; the schedule sets the "
                                                 "actual event start)",
                                  **resolved}
    return section


def _declarations(config: PreflightConfig, findings: list[dict[str, Any]]) -> dict[str, Any]:
    if config.budgets is None:
        budgets: dict[str, Any] = {"status": "absent"}
        findings.append({"code": "budgets_undeclared", "severity": "blocking",
                         "detail": "no finite settlement/resource budgets declared"})
    else:
        budgets = {"status": "declared_not_enforced", "values": config.budgets}
    if config.case_id is None:
        case: dict[str, Any] = {"status": "absent"}
        findings.append({"code": "case_inputs_absent", "severity": "blocking",
                         "detail": "no case or static inputs declared"})
    else:
        case = {"status": "declared_unverified", "case_id": config.case_id, "inputs": [
            {"name": item.name, "path": item.path, "declared_sha256": item.declared_sha256,
             "status": "declared_unverified"} for item in config.case_inputs]}
        findings.append({"code": "case_inputs_unverified", "severity": "blocking",
                         "detail": "case inputs are declared but not read or hashed by MASY-01"})
    return {"budgets": budgets, "case_inputs": case}


def _upstream_stages(config: PreflightConfig, before: dict[str, Any], expected_dirs: dict[str, str],
                     body: dict[str, Any], findings: list[dict[str, Any]]) -> None:
    upstream_src = [os.path.join(before[name]["root"], "src") for name in ("maple", "syrup")]
    try:
        body["imports"] = import_upstreams(upstream_src, expected_dirs)
    except OriginError as exc:
        body["imports"] = exc.evidence.pop("record", None)
        findings.append(_refusal(exc.code, exc.detail, **exc.evidence))
        return
    try:
        dependency = importlib.import_module("maple_syrup.dependency")
        provenance = importlib.import_module("maple_syrup.provenance")
    except Exception as exc:  # noqa: BLE001
        findings.append(_refusal("upstream_import_failed", f"{type(exc).__name__}: {exc}"[:4000]))
        return
    audit, audit_findings = run_native_audit(dependency, before["maple"]["root"])
    body["native_audit"] = audit
    findings.extend(audit_findings)
    limits = config.limits
    body["provenance_crosscheck"], cross_findings = crosscheck_provenance(
        provenance, before, max_files=limits.max_files, max_bytes=limits.max_bytes,
        git_timeout_s=limits.git_timeout_s)
    findings.extend(cross_findings)
    violations = module_origin_violations(dict(sys.modules), expected_dirs)
    body["final_origin_violations"] = violations[:50]
    if violations:
        findings.append(_refusal("module_origin_mismatch",
                                 f"{len(violations)} loaded module(s) with a foreign origin after the audit"))
    body["distribution_metadata"], dist_findings = distribution_records({
        "maple": before["maple"]["root"], "maple-syrup": before["syrup"]["root"], "masy": before["masy"]["root"],
    })
    findings.extend(dist_findings)


def main(argv: list[str], *, launch: dict[str, Any]) -> int:
    args = build_parser().parse_args(argv)
    findings: list[dict[str, Any]] = []
    body: dict[str, Any] = {"launch": launch, "stages_skipped": []}
    max_report_bytes = 16 * 1024 * 1024

    def finish() -> int:
        report, _ = render_report(body, findings)
        try:
            receipt = publish_report(launch["run_dir"], report, max_report_bytes=max_report_bytes)
        except ReportTooLarge as exc:
            print(f"masy_preflight: refused: {exc}; no report published", file=sys.stderr)
            return EXIT_CODES["refused"]
        print(json.dumps({"verdict": receipt["verdict"], "exit_code": receipt["exit_code"],
                          "report": receipt["path"], "sha256": receipt["sha256"]}, sort_keys=True))
        return receipt["exit_code"]

    try:
        config = load_config(args.config, masy_root=launch["masy_root"])
    except ConfigError as exc:
        body["config"] = {"path": os.path.abspath(args.config), "error": exc.code}
        findings.append(_refusal(exc.code, exc.detail))
        body["stages_skipped"] = ["source_observation", "imports", "native_audit", "post_observation"]
        return finish()
    max_report_bytes = config.max_report_bytes
    body["config"] = {"path": config.path, "sha256": config.sha256}

    masy_dir = os.path.join(launch["masy_root"], "src", "masy")
    masy_violations = module_origin_violations(dict(sys.modules), {"masy": masy_dir})
    if masy_violations:
        findings.append(_refusal("module_origin_mismatch", "masy modules with a foreign origin",
                                 violations=masy_violations[:50]))

    try:
        before = observe_sources(config, launch["launcher"])
    except SourceRefusal as exc:
        findings.append(_refusal(exc.code, exc.detail, **exc.evidence))
        body["source_observation"] = None
        body["stages_skipped"] = ["lock", "imports", "native_audit", "post_observation"]
        body.update(_declarations(config, findings))
        body["clock"] = _clock_section(config, findings)
        body["environment"] = environment_record()
        return finish()
    body["source_observation"] = before

    guards_ok = not masy_violations
    body["lock"] = {"status": "not_requested"}
    if args.expect_lock:
        try:
            compare_lock(load_lock(args.expect_lock, config.limits), before)
        except SourceRefusal as exc:
            body["lock"] = {"status": "refused", "path": os.path.abspath(args.expect_lock)}
            findings.append(_refusal(exc.code, exc.detail, **exc.evidence))
            guards_ok = False
        else:
            body["lock"] = {"status": "matched", "path": os.path.abspath(args.expect_lock)}

    if guards_ok:
        expected_dirs = {"masy": before["masy"]["package"]["dir"], "maple": before["maple"]["package"]["dir"],
                         "maple_syrup": before["syrup"]["package"]["dir"]}
        _upstream_stages(config, before, expected_dirs, body, findings)
    else:
        body["stages_skipped"] = ["imports", "native_audit"]
    body["environment"] = environment_record()

    try:
        after = observe_sources(config, launch["launcher"])
    except SourceRefusal as exc:
        body["source_observation_after"] = None
        findings.append(_refusal("post_observation_refused", f"{exc.code}: {exc.detail}", **exc.evidence))
    else:
        body["source_observation_after"] = after
        differences = diff_observations(before, after)
        body["source_consistency"] = {"status": "changed" if differences else "unchanged",
                                      "differences": differences}
        if differences:
            findings.append(_refusal("source_changed_during_preflight", f"changed: {differences}"))

    body.update(_declarations(config, findings))
    body["clock"] = _clock_section(config, findings)
    return finish()
