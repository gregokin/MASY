"""Verdicts, standing unqualified findings and exclusive report publication.

There is no success verdict. Precedence: `refused` (exit 2) >
`incompatible` (exit 3) > `not_transport_ready` (exit 4). Exit 1 is an
internal error and publishes no verdict.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from masy.sourceobs import canonical_json

__all__ = [
    "EXIT_CODES",
    "REPORT_SCHEMA",
    "SEVERITIES",
    "STANDING_FINDINGS",
    "ReportTooLarge",
    "compute_verdict",
    "publish_report",
    "render_report",
]

REPORT_SCHEMA = "masy.preflight.report.v1"
SEVERITIES = ("refusal", "incompatible", "blocking", "info")
EXIT_CODES = {"refused": 2, "incompatible": 3, "not_transport_ready": 4}

# Always present: nothing in MASY-01 qualifies these.
STANDING_FINDINGS: tuple[dict[str, str], ...] = (
    {"code": "immutable_import_unverified", "severity": "blocking",
     "detail": "live working-tree imports; no immutable bundle/overlay is built or verified in MASY-01"},
    {"code": "transport_not_implemented", "severity": "blocking",
     "detail": "MASY-01 has no simulation entry point; no wind or water transport is launched"},
    {"code": "alternating_physics_unqualified", "severity": "blocking",
     "detail": "no alternating MAPLE/SYRUP event has been executed or validated"},
    {"code": "wind_mobile_completion_unqualified", "severity": "blocking",
     "detail": "candidate: zero-input-wind native stepping under finite wind_settlement budgets; not probed"},
    {"code": "routing_reentry_unqualified", "severity": "blocking",
     "detail": "retained SYRUP graph/source reentry after wind needs a validated adapter; not implemented"},
    {"code": "cache_invalidation_unqualified", "severity": "blocking",
     "detail": "post-water MAPLE scheduler cache invalidation adapter not implemented"},
    {"code": "native_restart_unqualified", "severity": "blocking",
     "detail": "combined restart/checkpoint not implemented or certified"},
    {"code": "forcing_adapter_unqualified", "severity": "blocking",
     "detail": "data-only edge resolution; native solver realization of boundaries not qualified"},
)


class ReportTooLarge(Exception):
    """No report, not even the minimal refusal, fits the hard byte bound."""


def compute_verdict(findings: list[dict[str, Any]]) -> str:
    severities = {finding["severity"] for finding in findings}
    unknown = severities - set(SEVERITIES)
    if unknown:
        raise ValueError(f"unknown finding severities {sorted(unknown)}")
    if "refusal" in severities:
        return "refused"
    if "incompatible" in severities:
        return "incompatible"
    return "not_transport_ready"


def render_report(body: dict[str, Any], findings: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    all_findings = list(findings) + [dict(item) for item in STANDING_FINDINGS]
    verdict = compute_verdict(all_findings)
    report = {
        **body,
        "schema": REPORT_SCHEMA,
        "verdict": verdict,
        "exit_code": EXIT_CODES[verdict],
        "transport_ready": False,
        "findings": all_findings,
    }
    return report, EXIT_CODES[verdict]


def _write_exclusive(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_report(run_dir: str, report: dict[str, Any], *, max_report_bytes: int) -> dict[str, Any]:
    """Write `report.json` and `report.json.sha256` into the run directory.

    Each file is staged with O_EXCL and hard-linked into place, which fails
    rather than overwrite an existing file. An oversized report is replaced
    by a minimal `refused` report that states the original size. The bound
    is hard for `report.json` in both cases: if the minimal report does not
    fit either, `ReportTooLarge` is raised before anything is written (the
    fixed-size `.sha256` sidecar is not counted).
    """
    data = (canonical_json(report) + "\n").encode("ascii")
    if len(data) > max_report_bytes:
        minimal = {
            "schema": REPORT_SCHEMA,
            "verdict": "refused",
            "exit_code": EXIT_CODES["refused"],
            "transport_ready": False,
            "findings": [{"code": "report_oversized", "severity": "refusal",
                          "detail": f"full report was {len(data)} bytes > max_report_bytes={max_report_bytes}; "
                                    f"its original verdict was {report.get('verdict')!r}"}],
        }
        minimal_data = (canonical_json(minimal) + "\n").encode("ascii")
        if len(minimal_data) > max_report_bytes:
            raise ReportTooLarge(
                f"report_unpublishable: full report {len(data)} bytes and minimal refusal report "
                f"{len(minimal_data)} bytes both exceed max_report_bytes={max_report_bytes}"
            )
        data = minimal_data
        report = minimal
    digest = hashlib.sha256(data).hexdigest()
    root = Path(run_dir)
    for name, content in (("report.json", data), ("report.json.sha256", f"{digest}  report.json\n".encode())):
        staged = root / f".{name}.partial"
        _write_exclusive(staged, content)
        try:
            os.link(staged, root / name)
        finally:
            os.unlink(staged)
    return {"path": str(root / "report.json"), "sha256": digest, "bytes": len(data), "verdict": report["verdict"],
            "exit_code": report["exit_code"]}
