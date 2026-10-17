"""Source observation guards on synthetic temp Git repositories."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import _support
from _support import TempDirCase, commit_all, git, make_repo, requires_git, write

from masy import sourceobs
from masy.sourceobs import (
    Limits, SourceRefusal, compare_lock, diff_observations, load_lock, observe_masy, observe_upstream,
)

LIMITS = Limits(git_timeout_s=30.0)


def observe(root: Path, limits: Limits = LIMITS, expected_revision: str | None = None) -> dict:
    return observe_upstream("maple", str(root), package="maple", project_name="maple",
                            expected_revision=expected_revision, limits=limits)


@requires_git
class UpstreamObservationTests(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        self.repo = make_repo(self.tmp, "MAPLE", "maple", "maple", files={
            "src/maple/core.py": "VALUE = 1\n",
            ".gitignore": "*.egg-info/\n__pycache__/\n*.pyc\n*.so\nignored_link\n",
        })

    def refusal(self, code: str, **kwargs) -> SourceRefusal:
        with self.assertRaises(SourceRefusal) as ctx:
            observe(self.repo, **kwargs)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)
        return ctx.exception

    def test_clean_repo_records_head_and_independent_package_digest(self) -> None:
        obs = observe(self.repo)
        self.assertEqual(obs["head"], git(self.repo, "rev-parse", "HEAD").strip())
        self.assertEqual(obs["git_status"], "clean")
        self.assertEqual(obs["pyproject"]["project_name"], "maple")
        self.assertEqual(obs["pyproject"]["sha256"],
                         hashlib.sha256((self.repo / "pyproject.toml").read_bytes()).hexdigest())
        package_dir = self.repo / "src" / "maple"
        entries = sorted((p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in package_dir.iterdir())
        expected = hashlib.sha256(b"".join(f"{r}\0{h}\n".encode() for r, h in entries)).hexdigest()
        self.assertEqual(obs["package"]["digest_sha256"], expected)
        self.assertEqual(obs["package"]["file_count"], 2)

    def test_expected_revision_mismatch_refuses(self) -> None:
        self.refusal("expected_revision_mismatch", expected_revision="0" * 40)

    def test_expected_revision_match_passes(self) -> None:
        head = git(self.repo, "rev-parse", "HEAD").strip()
        self.assertEqual(observe(self.repo, expected_revision=head)["head"], head)

    def test_tracked_edit_refuses(self) -> None:
        write(self.repo / "src/maple/core.py", "VALUE = 22\n")
        self.refusal("dirty_scope")

    def test_edit_hidden_from_status_is_caught_by_blob_comparison(self) -> None:
        git(self.repo, "update-index", "--assume-unchanged", "src/maple/core.py")
        write(self.repo / "src/maple/core.py", "VALUE = 222\n")
        self.assertEqual(git(self.repo, "status", "--porcelain", "--", "src"), "")
        self.refusal("content_differs_from_head")

    def test_untracked_file_in_scope_refuses(self) -> None:
        write(self.repo / "src/maple/extra.py", "X = 1\n")
        self.refusal("dirty_scope")

    def test_deleted_tracked_file_refuses(self) -> None:
        (self.repo / "src/maple/core.py").unlink()
        self.refusal("dirty_scope")

    def test_uncommitted_pyproject_change_refuses(self) -> None:
        write(self.repo / "pyproject.toml", '[project]\nname = "maple"\nversion = "1"\n')
        self.refusal("dirty_scope")

    def test_committed_pyproject_with_wrong_project_refuses(self) -> None:
        write(self.repo / "pyproject.toml", '[project]\nname = "not-maple"\n')
        commit_all(self.repo)
        self.refusal("pyproject_project_mismatch")

    def test_missing_pyproject_refuses(self) -> None:
        git(self.repo, "rm", "-q", "pyproject.toml")
        commit_all(self.repo)
        self.refusal("pyproject_not_committed")

    def test_ignored_runtime_file_in_scope_refuses(self) -> None:
        write(self.repo / "src/maple/_ext.so", "binary")
        self.assertEqual(git(self.repo, "status", "--porcelain", "--", "src"), "")  # git alone sees nothing
        exc = self.refusal("untracked_runtime_input")
        self.assertEqual(exc.evidence["path"], "src/maple/_ext.so")

    def test_ignored_sourceless_bytecode_outside_cache_refuses(self) -> None:
        write(self.repo / "src/maple/hidden.pyc", "bytecode")
        self.refusal("untracked_runtime_input")

    def test_pycache_directory_is_excluded(self) -> None:
        write(self.repo / "src/maple/__pycache__/core.cpython-312.pyc", "cache")
        self.assertEqual(observe(self.repo)["package"]["file_count"], 2)

    def test_generated_metadata_is_recorded_not_refused(self) -> None:
        write(self.repo / "src/maple.egg-info/PKG-INFO", "Name: maple\n")
        obs = observe(self.repo)
        self.assertEqual([e["path"] for e in obs["generated_metadata"]["entries"]], ["src/maple.egg-info/PKG-INFO"])
        self.assertEqual(obs["generated_metadata"]["policy"], "recorded_not_repaired")

    def test_out_of_scope_untracked_files_are_ignored(self) -> None:
        write(self.repo / "cases/2026-10-17/run.txt", "unrelated case\n")
        write(self.repo / "notes.md", "unrelated\n")
        observe(self.repo)

    def test_committed_symlink_refuses(self) -> None:
        os.symlink("core.py", self.repo / "src/maple/alias.py")
        commit_all(self.repo)
        self.refusal("symlink_in_scope")

    def test_ignored_symlink_in_scope_refuses(self) -> None:
        os.symlink(self.tmp, self.repo / "src/maple/ignored_link")
        self.assertEqual(git(self.repo, "status", "--porcelain", "--", "src"), "")
        self.refusal("symlink_in_scope")

    def test_file_count_bound_refuses(self) -> None:
        self.refusal("max_files_exceeded", limits=Limits(max_files=2))

    def test_byte_bound_refuses(self) -> None:
        self.refusal("max_bytes_exceeded", limits=Limits(max_bytes=20))

    def test_git_output_bound_refuses(self) -> None:
        self.refusal("git_output_exceeded", limits=Limits(git_max_output_bytes=8))

    def test_missing_git_refuses(self) -> None:
        with mock.patch.object(sourceobs, "GIT_EXECUTABLE", str(self.tmp / "no-such-git")):
            self.refusal("git_unavailable")

    def test_subdirectory_root_refuses(self) -> None:
        with self.assertRaises(SourceRefusal) as ctx:
            observe(self.repo / "src")
        self.assertEqual(ctx.exception.code, "not_repository_root")

    def test_symlinked_root_refuses(self) -> None:
        link = self.tmp / "maple-link"
        os.symlink(self.repo, link)
        with self.assertRaises(SourceRefusal) as ctx:
            observe(link)
        self.assertEqual(ctx.exception.code, "source_root_not_canonical")

    def test_new_commit_between_observations_is_detected(self) -> None:
        before = observe(self.repo)
        write(self.repo / "src/maple/core.py", "VALUE = 3\n")
        commit_all(self.repo)
        after = observe(self.repo)
        changed = diff_observations({"maple": before}, {"maple": after})
        self.assertIn("maple.head", changed)
        self.assertIn("maple.package", changed)

    def test_generated_metadata_change_between_observations_is_detected(self) -> None:
        write(self.repo / "src/maple.egg-info/PKG-INFO", "Name: maple\n")
        before = observe(self.repo)
        write(self.repo / "src/maple.egg-info/PKG-INFO", "Name: maple\nVersion: 9\n")
        after = observe(self.repo)
        self.assertEqual(diff_observations({"maple": before}, {"maple": after}), ["maple.generated_metadata"])

    def test_identical_observations_have_no_difference(self) -> None:
        self.assertEqual(diff_observations({"maple": observe(self.repo)}, {"maple": observe(self.repo)}), [])


class GitOutputBoundTests(TempDirCase):
    """Synthetic read-only `git` stand-ins (no repository is touched)."""

    def fake_git(self, body: str) -> Path:
        path = self.tmp / "fake-git"
        path.write_text(f"#!{sys.executable}\nimport sys, time\n{body}\n")
        path.chmod(0o755)
        return path

    def run_fake(self, body: str, limits: Limits) -> bytes:
        with mock.patch.object(sourceobs, "GIT_EXECUTABLE", str(self.fake_git(body))):
            return sourceobs.run_git(self.tmp, ["status"], limits)

    def test_stderr_counts_against_the_output_cap(self) -> None:
        body = "sys.stderr.write('e' * 4096); sys.stderr.flush(); sys.stdout.write('abc')"
        with self.assertRaises(SourceRefusal) as ctx:
            self.run_fake(body, Limits(git_max_output_bytes=64))
        self.assertEqual(ctx.exception.code, "git_output_exceeded")

    def test_stderr_only_output_is_capped_while_the_child_runs(self) -> None:
        body = "sys.stderr.write('e' * 100000); sys.stderr.flush(); time.sleep(60)"
        started = time.monotonic()
        with self.assertRaises(SourceRefusal) as ctx:
            self.run_fake(body, Limits(git_max_output_bytes=64, git_timeout_s=30.0))
        self.assertEqual(ctx.exception.code, "git_output_exceeded")
        self.assertLess(time.monotonic() - started, 20.0)

    def test_small_combined_output_passes_and_failure_keeps_bounded_stderr(self) -> None:
        ok = "sys.stderr.write('warn'); sys.stdout.write('abc')"
        self.assertEqual(self.run_fake(ok, Limits(git_max_output_bytes=64)), b"abc")
        failing = "sys.stderr.write('fatal: synthetic'); sys.exit(3)"
        with self.assertRaises(SourceRefusal) as ctx:
            self.run_fake(failing, Limits(git_max_output_bytes=64))
        self.assertEqual(ctx.exception.code, "git_failed")
        self.assertIn("fatal: synthetic", ctx.exception.detail)

    def test_timeout_kills_and_refuses(self) -> None:
        started = time.monotonic()
        with self.assertRaises(SourceRefusal) as ctx:
            self.run_fake("time.sleep(60)", Limits(git_timeout_s=0.5))
        self.assertEqual(ctx.exception.code, "git_timeout")
        self.assertLess(time.monotonic() - started, 20.0)


@requires_git
class LockTests(TempDirCase):
    def test_lock_round_trip_and_mismatch(self) -> None:
        repo = make_repo(self.tmp, "MAPLE", "maple", "maple")
        block = {"masy": {"root": "/x"}, "maple": observe(repo), "syrup": {"root": "/y"}}
        lock_path = self.tmp / "lock.json"
        lock_path.write_text(json.dumps(block))
        compare_lock(load_lock(str(lock_path), LIMITS), block)
        altered = json.loads(json.dumps(block))
        altered["maple"]["head"] = "f" * 40
        with self.assertRaises(SourceRefusal) as ctx:
            compare_lock(load_lock(str(lock_path), LIMITS), altered)
        self.assertEqual(ctx.exception.code, "lock_mismatch")
        self.assertEqual(ctx.exception.evidence["differences"], ["maple.head"])

    def test_malformed_lock_refuses(self) -> None:
        for content in ("not json", json.dumps({"source_observation": {}}), json.dumps([1])):
            path = self.tmp / "lock.json"
            path.write_text(content)
            with self.assertRaises(SourceRefusal) as ctx:
                load_lock(str(path), LIMITS)
            self.assertEqual(ctx.exception.code, "lock_unreadable")


class MasyObservationTests(TempDirCase):
    def make_masy(self) -> Path:
        root = self.tmp / "MASY"
        write(root / "src/masy/__init__.py", "")
        write(root / "tools/masy_preflight.py", "")
        return root

    def test_masy_files_are_hashed_without_git(self) -> None:
        root = self.make_masy()
        obs = observe_masy(str(root), str(root / "tools/masy_preflight.py"), LIMITS)
        self.assertEqual(obs["guard"], "explicit_lock_only")
        self.assertIsNone(obs["pyproject"])
        self.assertEqual(obs["files_digest"]["file_count"], 2)

    def test_masy_symlink_refuses(self) -> None:
        root = self.make_masy()
        os.symlink(root / "src/masy/__init__.py", root / "src/masy/alias.py")
        with self.assertRaises(SourceRefusal) as ctx:
            observe_masy(str(root), str(root / "tools/masy_preflight.py"), LIMITS)
        self.assertEqual(ctx.exception.code, "symlink_in_scope")

    def test_foreign_launcher_refuses(self) -> None:
        root = self.make_masy()
        other = write(self.tmp / "elsewhere/masy_preflight.py", "")
        with self.assertRaises(SourceRefusal) as ctx:
            observe_masy(str(root), str(other), LIMITS)
        self.assertEqual(ctx.exception.code, "launcher_outside_masy_root")


if __name__ == "__main__":
    unittest.main()
