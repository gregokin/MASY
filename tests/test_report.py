"""Verdict precedence, canonical JSON and exclusive publication."""

from __future__ import annotations

import hashlib
import json
import os
import unittest

from _support import TempDirCase

from masy.report import (
    EXIT_CODES, STANDING_FINDINGS, ReportTooLarge, compute_verdict, publish_report, render_report,
)


def finding(severity: str, code: str = "x") -> dict:
    return {"code": code, "severity": severity, "detail": ""}


class VerdictTests(unittest.TestCase):
    def test_precedence(self) -> None:
        cases = [
            ([], "not_transport_ready"),
            ([finding("info"), finding("blocking")], "not_transport_ready"),
            ([finding("incompatible"), finding("blocking")], "incompatible"),
            ([finding("incompatible"), finding("refusal")], "refused"),
            ([finding("refusal"), finding("info")], "refused"),
        ]
        for findings, verdict in cases:
            with self.subTest(findings=findings):
                self.assertEqual(compute_verdict(findings), verdict)

    def test_unknown_severity_is_an_error_not_a_pass(self) -> None:
        with self.assertRaises(ValueError):
            compute_verdict([finding("ok")])

    def test_no_success_path(self) -> None:
        self.assertEqual(set(EXIT_CODES), {"refused", "incompatible", "not_transport_ready"})
        self.assertNotIn(0, EXIT_CODES.values())
        report, code = render_report({}, [])
        self.assertEqual((report["verdict"], code, report["transport_ready"]), ("not_transport_ready", 4, False))
        standing = {f["code"] for f in report["findings"]}
        self.assertTrue({"immutable_import_unverified", "transport_not_implemented",
                         "alternating_physics_unqualified"} <= standing)
        self.assertTrue(all(f["severity"] == "blocking" for f in STANDING_FINDINGS))


class PublicationTests(TempDirCase):
    def test_canonical_report_and_digest(self) -> None:
        report, _ = render_report({"b": 1, "a": [1.5]}, [finding("refusal")])
        receipt = publish_report(str(self.tmp), report, max_report_bytes=1 << 20)
        data = (self.tmp / "report.json").read_bytes()
        self.assertEqual(data, (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode())
        self.assertEqual(receipt["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual((self.tmp / "report.json.sha256").read_text(), f"{receipt['sha256']}  report.json\n")
        self.assertEqual(sorted(os.listdir(self.tmp)), ["report.json", "report.json.sha256"])

    def test_existing_report_is_never_overwritten(self) -> None:
        (self.tmp / "report.json").write_text("accepted")
        report, _ = render_report({}, [])
        with self.assertRaises(FileExistsError):
            publish_report(str(self.tmp), report, max_report_bytes=1 << 20)
        self.assertEqual((self.tmp / "report.json").read_text(), "accepted")
        self.assertFalse((self.tmp / ".report.json.partial").exists())

    def test_oversized_report_is_replaced_by_minimal_refusal(self) -> None:
        report, _ = render_report({"payload": "x" * 5000}, [])
        receipt = publish_report(str(self.tmp), report, max_report_bytes=2000)
        published = json.loads((self.tmp / "report.json").read_text())
        self.assertEqual((receipt["verdict"], receipt["exit_code"]), ("refused", 2))
        self.assertEqual(published["findings"][0]["code"], "report_oversized")
        self.assertIn("not_transport_ready", published["findings"][0]["detail"])

    def test_minimal_report_obeys_the_hard_bound(self) -> None:
        report, _ = render_report({"payload": "x" * 5000}, [])
        for bound in (400, 2000):
            run = self.tmp / f"bound{bound}"
            run.mkdir()
            with self.subTest(bound=bound):
                receipt = publish_report(str(run), report, max_report_bytes=bound)
                self.assertLessEqual(len((run / "report.json").read_bytes()), bound)
                self.assertEqual(receipt["bytes"], len((run / "report.json").read_bytes()))

    def test_unfittable_report_refuses_and_publishes_nothing(self) -> None:
        report, _ = render_report({"payload": "x" * 5000}, [])
        with self.assertRaises(ReportTooLarge) as ctx:
            publish_report(str(self.tmp), report, max_report_bytes=64)
        self.assertIn("max_report_bytes=64", str(ctx.exception))
        self.assertEqual(os.listdir(self.tmp), [])

    def test_unfittable_report_preserves_existing_files(self) -> None:
        (self.tmp / "report.json").write_text("accepted")
        report, _ = render_report({"payload": "x" * 5000}, [])
        with self.assertRaises(ReportTooLarge):
            publish_report(str(self.tmp), report, max_report_bytes=64)
        self.assertEqual(sorted(os.listdir(self.tmp)), ["report.json"])
        self.assertEqual((self.tmp / "report.json").read_text(), "accepted")

    def test_non_finite_values_are_not_serialized(self) -> None:
        report, _ = render_report({"bad": float("nan")}, [])
        with self.assertRaises(ValueError):
            publish_report(str(self.tmp), report, max_report_bytes=1 << 20)
        self.assertFalse((self.tmp / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
