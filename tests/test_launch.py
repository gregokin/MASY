"""Isolated launcher invocation end to end, on synthetic upstream repos only.

The fake SYRUP `dependency` module mimics the observed live failure
(`resolve_maple_dependency` refusing with source_root=None) while its API
check passes. No real upstream code, transport or dataset is involved.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from _support import LAUNCHER, TempDirCase, make_fake_pair, requires_git, write, write_config


def run_launcher(*args: str, flags: tuple[str, ...] = ("-B", "-I"), cwd: Path | None = None):
    return subprocess.run([sys.executable, *flags, str(LAUNCHER), *args], capture_output=True, timeout=300,
                          cwd=cwd)


def codes(report: dict, severity: str | None = None) -> list[str]:
    return [f["code"] for f in report["findings"] if severity is None or f["severity"] == severity]


class LauncherGuardTests(TempDirCase):
    def test_missing_isolation_flags_refuse_before_run_dir(self) -> None:
        for flags in (("-B",), ("-I",), ()):
            run_dir = self.tmp / f"run{len(flags)}{''.join(flags)}"
            with self.subTest(flags=flags):
                result = run_launcher("--config", "unused.toml", "--run-dir", str(run_dir), flags=flags)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(b"-B -I", result.stderr)
                self.assertFalse(run_dir.exists())

    def test_existing_run_dir_refuses_and_is_untouched(self) -> None:
        config = write(self.tmp / "bad.toml", 'schema = "masy.preflight.v1"\nupdate = true\n')
        run_dir = self.tmp / "run"
        run_dir.mkdir()
        (run_dir / "report.json").write_text("accepted")
        result = run_launcher("--config", str(config), "--run-dir", str(run_dir))
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"already exists", result.stderr)
        self.assertEqual([p.name for p in run_dir.iterdir()], ["report.json"])
        self.assertEqual((run_dir / "report.json").read_text(), "accepted")

    def test_isolated_invocation_finds_masy_and_reports_config_refusal(self) -> None:
        config = write(self.tmp / "bad.toml", 'schema = "masy.preflight.v1"\nupdate = true\n')
        run_dir = self.tmp / "run"
        result = run_launcher("--config", str(config), "--run-dir", str(run_dir), cwd=self.tmp)
        self.assertEqual(result.returncode, 2, result.stderr)
        report = json.loads((run_dir / "report.json").read_text())
        self.assertEqual(report["verdict"], "refused")
        self.assertEqual(codes(report, "refusal"), ["unknown_config_field"])
        self.assertEqual(report["launch"]["masy_src"], str(LAUNCHER.parents[1] / "src"))
        self.assertEqual(report["launch"]["pycache_prefix"], str(run_dir.resolve() / "cache" / "pycache"))
        self.assertFalse((run_dir / "cache" / "pycache").exists())  # future prefix, never created
        self.assertTrue((run_dir / "cache" / "numba").is_dir())
        self.assertEqual(report["launch"]["environment_overrides"]["JAX_PLATFORMS"], "cpu")
        self.assertEqual(json.loads(result.stdout)["verdict"], "refused")

    def test_help_is_informational_and_creates_nothing(self) -> None:
        for args in (("--help",), ("-h",), ("--config", "x.toml", "--help")):
            with self.subTest(args=args):
                before = sorted(self.tmp.iterdir())
                result = run_launcher(*args, cwd=self.tmp)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(b"--run-dir", result.stdout)
                self.assertIn(b"Informational help only", result.stdout)
                self.assertEqual(sorted(self.tmp.iterdir()), before)

    def test_missing_run_dir_without_help_is_a_usage_refusal(self) -> None:
        result = run_launcher("--config", "x.toml", cwd=self.tmp)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(sorted(self.tmp.iterdir()), [])


def tree_state(root: Path) -> list[tuple[str, bool, int]]:
    """Every path under root (and root itself) with its type and mtime."""
    paths = [root, *root.rglob("*")]
    return sorted((str(p.relative_to(root)), p.is_dir(), p.lstat().st_mtime_ns) for p in paths)


class InputSeparationTests(TempDirCase):
    """The run directory may never be created inside a read-only input.

    Synthetic source trees need no Git: the overlap refusal happens before
    any source observation, and the tests assert the trees are byte- and
    mtime-identical afterwards, not just a nonzero exit."""

    def setUp(self) -> None:
        super().setUp()
        self.maple = self.tmp / "MAPLE"
        self.syrup = self.tmp / "SYRUP"
        write(self.maple / "pyproject.toml", '[project]\nname = "maple"\n')
        write(self.maple / "src" / "maple" / "__init__.py", "")
        write(self.syrup / "src" / "maple_syrup" / "__init__.py", "")
        self.config = write_config(self.tmp / "preflight.toml", self.maple, self.syrup)

    def assertRefusedWithoutMutation(self, run_dir: Path, *roots: Path, config: Path | None = None,
                                     message: bytes = b"overlaps") -> None:
        before = {root: tree_state(root) for root in roots}
        result = run_launcher("--config", str(config or self.config), "--run-dir", str(run_dir), cwd=self.tmp)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(message, result.stderr)
        self.assertIn(b"nothing was created", result.stderr)
        self.assertFalse(os.path.lexists(run_dir))
        for root in roots:
            self.assertEqual(tree_state(root), before[root], root)

    def test_run_dir_inside_configured_source_refuses_before_any_write(self) -> None:
        for run_dir in (self.maple / "src" / "maple" / "audit_output", self.maple / "audit", self.syrup / "x"):
            with self.subTest(run_dir=run_dir):
                self.assertRefusedWithoutMutation(run_dir, self.maple, self.syrup)

    def test_symlinked_parent_alias_into_source_refuses(self) -> None:
        alias = self.tmp / "alias"
        os.symlink(self.maple / "src", alias)
        self.assertRefusedWithoutMutation(alias / "maple" / "audit_output", self.maple)

    def test_symlinked_source_root_alias_in_config_refuses(self) -> None:
        alias = self.tmp / "maple-alias"
        os.symlink(self.maple, alias)
        config = write_config(self.tmp / "alias.toml", alias, self.syrup)
        self.assertRefusedWithoutMutation(self.maple / "src" / "audit", self.maple, config=config)

    def test_otherwise_invalid_config_still_protects_its_sources(self) -> None:
        config = write_config(self.tmp / "invalid.toml", self.maple, self.syrup, "update = true\n")
        self.assertRefusedWithoutMutation(self.maple / "src" / "maple" / "audit_output", self.maple, config=config)

    def test_invalid_source_roots_are_still_protected(self) -> None:
        missing = self.tmp / "NOPE"
        config = write_config(self.tmp / "missing-root.toml", missing, self.syrup)
        self.assertRefusedWithoutMutation(missing / "audit", self.syrup, config=config)
        self.assertFalse(missing.exists())
        not_a_repo = self.tmp / "plain"
        write(not_a_repo / "data.txt", "x")
        config = write_config(self.tmp / "plain-root.toml", not_a_repo, self.syrup)
        self.assertRefusedWithoutMutation(not_a_repo / "audit", not_a_repo, config=config)

    def test_unreadable_or_unparseable_config_refuses_before_any_write(self) -> None:
        broken = write(self.tmp / "broken.toml", "[sources.maple\nroot = 1\n")
        for config, message in ((broken, b"cannot parse config"), (self.tmp / "missing.toml", b"cannot read config")):
            with self.subTest(config=config.name):
                self.assertRefusedWithoutMutation(self.maple / "src" / "maple" / "audit_output", self.maple,
                                                  config=config, message=message)

    def test_run_dir_inside_masy_runtime_refuses(self) -> None:
        for run_dir in (LAUNCHER.parents[1] / "src" / "masy" / "masy01_test_audit",
                        LAUNCHER.parent / "masy01_test_audit"):
            with self.subTest(run_dir=run_dir):
                try:
                    self.assertRefusedWithoutMutation(run_dir, run_dir.parent)
                finally:
                    if run_dir.is_dir() and not any(run_dir.iterdir()):
                        run_dir.rmdir()

    def test_sibling_with_source_name_prefix_is_an_ordinary_run_dir(self) -> None:
        config = write_config(self.tmp / "invalid.toml", self.maple, self.syrup, "update = true\n")
        before = tree_state(self.maple)
        run_dir = self.tmp / "MAPLE-run"
        result = run_launcher("--config", str(config), "--run-dir", str(run_dir), cwd=self.tmp)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(codes(json.loads((run_dir / "report.json").read_text()), "refusal"),
                         ["unknown_config_field"])
        self.assertEqual(tree_state(self.maple), before)

    def test_calendar_field_refuses_through_launcher(self) -> None:
        config = write(self.tmp / "cal.toml",
                       'schema = "masy.preflight.v1"\n[forcing]\nid = "f"\nedges_s = ["0", "1"]\n'
                       'start_datetime = "2026-10-17T00:00:00Z"\n')
        result = run_launcher("--config", str(config), "--run-dir", str(self.tmp / "run"))
        report = json.loads((self.tmp / "run" / "report.json").read_text())
        self.assertEqual((result.returncode, codes(report, "refusal")), (2, ["unsupported_calendar_field"]))


@requires_git
class EndToEndTests(TempDirCase):
    def run_pair(self, *, mutate: bool = False, extra_args: tuple[str, ...] = (), name: str = "run"):
        if not (self.tmp / "MAPLE").exists():
            self.maple, self.syrup = make_fake_pair(self.tmp, mutate=mutate)
        config = write_config(self.tmp / "preflight.toml", self.maple, self.syrup,
                              '[forcing]\nid = "storm-2026-10-17"\nedges_s = ["0", "0.1", "600"]\n')
        run_dir = self.tmp / name
        result = run_launcher("--config", str(config), "--run-dir", str(run_dir), *extra_args)
        report = json.loads((run_dir / "report.json").read_text())
        return result, report

    def test_resolver_refusal_retains_api_evidence_and_sources_unchanged(self) -> None:
        result, report = self.run_pair()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(report["verdict"], "refused")
        self.assertFalse(report["transport_ready"])
        self.assertIn("native_resolver_refused", codes(report, "refusal"))
        self.assertIn("source root None", report["native_audit"]["native_resolver"]["error"])
        self.assertEqual(report["native_audit"]["syrup_required_api"]["status"], "pass")
        self.assertIn("masy_required_api_missing", codes(report, "incompatible"))
        self.assertEqual(report["source_consistency"], {"status": "unchanged", "differences": []})
        self.assertTrue(report["provenance_crosscheck"]["maple"]["agrees"])
        self.assertEqual(report["final_origin_violations"], [])
        self.assertEqual(report["imports"]["packages"]["maple"]["file"],
                         str(self.maple / "src" / "maple" / "__init__.py"))
        self.assertEqual(report["source_observation"]["maple"]["head"],
                         report["source_observation_after"]["maple"]["head"])
        self.assertEqual(report["clock"]["forcing"]["edges"][1]["intended_decimal"], "0.1")
        self.assertIn("immutable_import_unverified", codes(report, "blocking"))
        self.assertFalse(report["environment"]["gpu_queried"])
        self.assertFalse(any(p.name == "__pycache__" for p in self.maple.rglob("__pycache__")))

    def test_dirty_source_skips_imports(self) -> None:
        self.maple, self.syrup = make_fake_pair(self.tmp)
        write(self.syrup / "src" / "maple_syrup" / "dependency.py", "raise SystemExit('must not import')\n")
        result, report = self.run_pair()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(codes(report, "refusal"), ["dirty_scope"])
        self.assertIsNone(report["source_observation"])
        self.assertNotIn("imports", report)
        self.assertNotIn("native_audit", report)

    def test_lock_round_trip_then_mismatch(self) -> None:
        _, first = self.run_pair(name="run1")
        lock = self.tmp / "lock.json"
        lock.write_text(json.dumps(first["source_observation"]))
        _, matched = self.run_pair(extra_args=("--expect-lock", str(lock)), name="run2")
        self.assertEqual(matched["lock"]["status"], "matched")
        tampered = json.loads(lock.read_text())
        tampered["syrup"]["head"] = "0" * 40
        lock.write_text(json.dumps(tampered))
        result, refused = self.run_pair(extra_args=("--expect-lock", str(lock)), name="run3")
        self.assertEqual(result.returncode, 2)
        self.assertIn("lock_mismatch", codes(refused, "refusal"))
        self.assertNotIn("imports", refused)
        self.assertEqual(refused["source_consistency"]["status"], "unchanged")

    def test_concurrent_change_during_preflight_refuses(self) -> None:
        result, report = self.run_pair(mutate=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        refusals = codes(report, "refusal")
        self.assertIn("post_observation_refused", refusals)
        self.assertIn("upstream_provenance_disagreement", refusals)
        self.assertEqual(report["native_audit"]["syrup_required_api"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
