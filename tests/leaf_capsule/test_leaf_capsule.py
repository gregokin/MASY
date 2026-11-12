"""Engineering tests of the immutable LEAF capsule helpers. Synthetic temporary Git repos only: never scientific, provenance or production proof."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from masy import leaf_capsule as cap  # noqa: E402
from masy import leaf_capsule_seal as seal  # noqa: E402
from masy.native_source_gate import GateLimits, GateRefusal  # noqa: E402
from masy.runtime import RuntimeLimits, RuntimeRefusal  # noqa: E402

PYPROJECT = '[project]\nname = "leaf"\nversion = "0.1.0"\n'
LEAF8DA = ROOT.parent.parent / "phase1_leaf_calendar_impl_20261112" / "leaf8da_source_reference"
LITERAL_8DA_PACKAGE = "8da24af5a672138bab335d8fd1d4518c5a17ae90db9b400c5d211cc0c768fa4b"


def git(root, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


def rmtree(path):
    for dp, dn, fn in os.walk(path):
        os.chmod(dp, 0o755)
        for n in fn:
            p = os.path.join(dp, n)
            if not os.path.islink(p):
                os.chmod(p, 0o644)
    shutil.rmtree(path, ignore_errors=True)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="leafcap_"))
        self.addCleanup(rmtree, self.tmp)
        self.repo = os.path.join(self.tmp, "leafrepo")
        self.ns = os.path.join(self.tmp, "wrapper", "runs", "store-leaf-capsules")
        os.makedirs(os.path.dirname(self.ns))
        self.wrapper = os.path.join(self.tmp, "wrapper")
        self.make_repo({"pyproject.toml": PYPROJECT, "leaf/__init__.py": "A = 1\n", "leaf/sub/mod.py": "B = 2\n"})

    def make_repo(self, files):
        os.makedirs(self.repo)
        git(self.repo, "init", "-q")
        for rel, text in files.items():
            p = Path(self.repo, rel)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "c")
        self.head = git(self.repo, "rev-parse", "HEAD")

    def observe(self, **kw):
        return seal.observe_leaf_scope(self.repo, expected_revision=kw.pop("rev", self.head), **kw)

    def seal(self, obs=None, run="r1", **kw):
        obs = obs or self.observe()
        prot = {"wrapper src": os.path.join(self.wrapper, "src")}
        return seal.seal_leaf_capsule(self.ns, obs, run_id=run, wrapper_roots=[self.wrapper], protected=prot, **kw)

    def code(self, exc_cm):
        return exc_cm.exception.code


class HappyPath(Base):
    def test_seal_verify_reseal_same_id(self):
        r = self.seal()
        self.assertFalse(r["reused"])
        loc = r["location"]
        self.assertEqual(stat.S_IMODE(os.stat(loc).st_mode), 0o555)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.join(loc, "source", "leaf", "__init__.py")).st_mode), 0o444)
        v = cap.verify_capsule(self.ns, r["object_id"], expected_manifest_sha256=r["manifest_sha256"])
        self.assertEqual(v["provenance"]["head"], self.head)
        self.assertEqual(v["provenance"]["tree"], git(self.repo, "rev-parse", "HEAD^{tree}"))
        self.assertEqual(v["file_count"], 3)
        self.assertNotEqual(v["tree_digest"], v["package_digest"])  # full-tree vs package digest stay distinct
        r2 = self.seal(run="r2")
        self.assertTrue(r2["reused"])
        self.assertEqual(r2["object_id"], r["object_id"])
        self.assertFalse(any(n.startswith(".staging") for n in os.listdir(os.path.join(self.ns, "objects"))))

    def test_id_is_commit_and_path_independent(self):
        a = self.seal()["object_id"]
        Path(self.repo, "README").write_text("x")  # outside scope
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "second")
        head2 = git(self.repo, "rev-parse", "HEAD")
        b = self.seal(self.observe(rev=head2), run="r2")
        self.assertTrue(b["reused"])
        self.assertEqual(a, b["object_id"])
        self.assertEqual(cap.verify_capsule(self.ns, a)["provenance"]["head"], self.head)  # first commit's provenance retained

    def test_pycache_excluded_but_not_silently_ignored_sources(self):
        Path(self.repo, ".git/info/exclude").write_text("__pycache__/\n")
        os.makedirs(os.path.join(self.repo, "leaf", "__pycache__"))
        Path(self.repo, "leaf", "__pycache__", "x.pyc").write_bytes(b"c")
        self.assertEqual(len(self.observe()["files"]), 3)

    @unittest.skipUnless(LEAF8DA.is_dir(), "frozen 8da reference tree absent")
    def test_real_8da_tree_in_synthetic_repo(self):
        shutil.rmtree(self.repo)
        shutil.copytree(LEAF8DA, self.repo)
        for dp, dn, fn in os.walk(self.repo):
            os.chmod(dp, 0o755)
            for n in fn:
                os.chmod(os.path.join(dp, n), 0o644)
        git(self.repo, "init", "-q")
        git(self.repo, "add", "-A")
        git(self.repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
        self.head = git(self.repo, "rev-parse", "HEAD")
        r = self.seal()
        self.assertEqual(r["file_count"], 24)
        self.assertEqual(r["package_digest"], LITERAL_8DA_PACKAGE)  # synthetic provenance: NOT the original commit and NOT qualification
        sys.path.insert(0, str(ROOT / "src"))
        from masy.leaf_source_profile import LEAF_MANIFEST_8DA
        lit = dict(LEAF_MANIFEST_8DA)
        self.assertEqual(r["files"], lit)


class Observation(Base):
    def test_expected_revision(self):
        with self.assertRaises(RuntimeRefusal) as c:
            seal.observe_leaf_scope(self.repo, expected_revision="0" * 40)
        self.assertEqual(self.code(c), "expected_revision_mismatch")
        with self.assertRaises(RuntimeRefusal) as c:
            seal.observe_leaf_scope(self.repo, expected_revision="main")
        self.assertEqual(self.code(c), "expected_revision_required")

    def test_dirty(self):
        Path(self.repo, "leaf/__init__.py").write_text("A = 2\n")
        with self.assertRaises(RuntimeRefusal) as c:
            self.observe()
        self.assertEqual(self.code(c), "dirty_scope")

    def test_untracked_and_ignored(self):
        Path(self.repo, "leaf/new.py").write_text("x")
        with self.assertRaises(RuntimeRefusal) as c:
            self.observe()
        self.assertEqual(self.code(c), "dirty_scope")
        os.remove(Path(self.repo, "leaf/new.py"))
        Path(self.repo, ".git/info/exclude").write_text("leaf/hidden.py\n")
        Path(self.repo, "leaf/hidden.py").write_text("x")
        with self.assertRaises(RuntimeRefusal) as c:
            self.observe()
        self.assertEqual(self.code(c), "untracked_runtime_input")

    def test_symlink_special(self):
        os.symlink("__init__.py", os.path.join(self.repo, "leaf", "lnk.py"))
        with self.assertRaises(RuntimeRefusal) as c:
            self.observe()
        self.assertIn(self.code(c), {"dirty_scope", "symlink_in_scope"})
        os.remove(os.path.join(self.repo, "leaf", "lnk.py"))
        git(self.repo, "status", "--short")
        os.mkfifo(os.path.join(self.repo, "leaf", "fifo"))
        with self.assertRaises(RuntimeRefusal) as c:
            self.observe()
        self.assertIn(self.code(c), {"non_regular_file", "dirty_scope"})

    def test_committed_symlink(self):
        os.symlink("__init__.py", os.path.join(self.repo, "leaf", "lnk.py"))
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "l")
        with self.assertRaises(RuntimeRefusal) as c:
            seal.observe_leaf_scope(self.repo, expected_revision=git(self.repo, "rev-parse", "HEAD"))
        self.assertEqual(self.code(c), "symlink_in_scope")

    def test_file_count_bound(self):
        with self.assertRaises(RuntimeRefusal) as c:
            self.observe(gate_limits=GateLimits(max_files=2))
        self.assertEqual(self.code(c), "max_files_exceeded")

    def test_zero_admission_before_body(self):
        calls = []

        def admit(n, what):
            calls.append(what)
            raise GateRefusal("cold_memory", what)
        with self.assertRaises(GateRefusal) as c:
            self.observe(admit=admit)
        self.assertEqual(c.exception.code, "cold_memory")
        self.assertEqual(calls, ["LEAF observation buffers"])

    def test_wrong_project_name(self):
        shutil.rmtree(self.repo)
        self.make_repo({"pyproject.toml": '[project]\nname = "other"\nversion = "1"\n', "leaf/__init__.py": ""})
        with self.assertRaises(RuntimeRefusal) as c:
            self.observe()
        self.assertEqual(self.code(c), "pyproject_project_mismatch")


class Publication(Base):
    def test_namespace_conflicts(self):
        obs = self.observe()
        for ns in (os.path.join(self.repo, "caps"), os.path.join(self.wrapper, "caps"), os.path.join(self.wrapper, "runs", "store", "leaf-capsules")):
            with self.assertRaises(RuntimeRefusal) as c:
                seal.seal_leaf_capsule(ns, obs, run_id="r", wrapper_roots=[self.wrapper], protected={"wrapper src": os.path.join(self.wrapper, "src"), "store": os.path.join(self.wrapper, "runs", "store")})
            self.assertEqual(self.code(c), "store_overlaps_input")

    def test_interruption_retains_staging_no_final(self):
        class Boom(Exception):
            pass

        def hook():
            raise RuntimeRefusal("injected", "x")
        with self.assertRaises(RuntimeRefusal) as c:
            self.seal(_after_copy=hook)
        self.assertIn("retained_staging", c.exception.evidence)
        names = os.listdir(os.path.join(self.ns, "objects"))
        self.assertEqual(len(names), 1)
        self.assertTrue(names[0].startswith(".staging-r1"))
        self.assertTrue(self.seal(run="r2")["object_id"])  # a later run publishes; evidence retained
        self.assertEqual(len(os.listdir(os.path.join(self.ns, "objects"))), 2)

    def test_copy_race_changed_source(self):
        obs = self.observe()
        Path(self.repo, "leaf/sub/mod.py").write_text("B = 9\n")  # same size, different bytes, after observation
        with self.assertRaises(RuntimeRefusal) as c:
            self.seal(obs)
        self.assertEqual(self.code(c), "source_changed_during_read")
        self.assertFalse([n for n in os.listdir(os.path.join(self.ns, "objects")) if not n.startswith(".staging")])

    def test_dirty_after_observation_refused(self):
        obs = self.observe()
        Path(self.repo, "leaf/extra.py").write_text("x")
        with self.assertRaises(RuntimeRefusal) as c:
            self.seal(obs)
        self.assertEqual(self.code(c), "source_changed_during_seal")

    def test_staging_exists_and_bad_run_id(self):
        os.makedirs(os.path.join(self.ns, "objects"))
        os.mkdir(os.path.join(self.ns, "objects", ".staging-r1-leaf"))
        with self.assertRaises(RuntimeRefusal) as c:
            self.seal()
        self.assertEqual(self.code(c), "staging_exists")
        with self.assertRaises(RuntimeRefusal) as c:
            self.seal(run="../x")
        self.assertEqual(self.code(c), "run_id_invalid")

    def test_corrupt_existing_never_overwritten(self):
        r = self.seal()
        f = os.path.join(r["location"], "source", "leaf", "__init__.py")
        os.chmod(r["location"], 0o755)
        os.chmod(os.path.dirname(f), 0o755)
        os.chmod(f, 0o644)
        Path(f).write_text("A = 7\n")
        os.chmod(f, 0o444)
        os.chmod(os.path.dirname(f), 0o555)
        os.chmod(r["location"], 0o555)
        with self.assertRaises(RuntimeRefusal):
            self.seal(run="r2")
        self.assertEqual(Path(f).read_text(), "A = 7\n")  # untouched evidence


class Tamper(Base):
    def setUp(self):
        super().setUp()
        r = self.seal()
        self.loc, self.id = r["location"], r["object_id"]

    def unlock(self, *paths):
        for p in paths:
            os.chmod(p, 0o755 if os.path.isdir(p) else 0o644)

    def refused(self, code=None):
        with self.assertRaises(GateRefusal) as c:
            cap.verify_capsule(self.ns, self.id)
        if code:
            self.assertEqual(c.exception.code, code)

    def test_extra_top_entry(self):
        self.unlock(self.loc)
        Path(self.loc, "extra").write_text("x")
        os.chmod(os.path.join(self.loc, "extra"), 0o444)
        os.chmod(self.loc, 0o555)
        self.refused("capsule_entries")

    def test_missing_manifest(self):
        self.unlock(self.loc)
        os.remove(os.path.join(self.loc, "manifest.json"))
        os.chmod(self.loc, 0o555)
        self.refused("capsule_entries")

    def test_writable_file_and_dir(self):
        f = os.path.join(self.loc, "source", "leaf", "__init__.py")
        os.chmod(f, 0o644)
        self.refused("writable_file")
        os.chmod(f, 0o444)
        os.chmod(os.path.join(self.loc, "source", "leaf"), 0o755)
        self.refused("writable_directory")

    def test_changed_byte_extra_missing_and_empty_dir(self):
        d = os.path.join(self.loc, "source", "leaf")
        f = os.path.join(d, "__init__.py")
        self.unlock(d, f)
        Path(f).write_text("A = 2\n")
        os.chmod(f, 0o444)
        os.chmod(d, 0o555)
        self.refused("snapshot_differs")
        self.unlock(d)
        Path(f).chmod(0o644)
        Path(f).write_text("A = 1\n")
        os.chmod(f, 0o444)
        os.mkdir(os.path.join(d, "emptydir"))
        os.chmod(os.path.join(d, "emptydir"), 0o555)
        os.chmod(d, 0o555)
        self.refused("capsule_entries")

    def test_symlink_in_source(self):
        d = os.path.join(self.loc, "source", "leaf")
        self.unlock(d)
        os.symlink("__init__.py", os.path.join(d, "l.py"))
        os.chmod(d, 0o555)
        self.refused()

    def rewrite(self, name, data):
        self.unlock(self.loc)
        p = os.path.join(self.loc, name)
        os.chmod(p, 0o644)
        Path(p).write_bytes(data)
        os.chmod(p, 0o444)
        os.chmod(self.loc, 0o555)

    def test_manifest_forgeries(self):
        m = json.loads(Path(self.loc, "manifest.json").read_bytes())
        bad = dict(m, object_id="0" * 64)
        self.rewrite("manifest.json", cap.canonical_bytes(bad))
        self.refused("capsule_id_mismatch")
        ident = dict(m["identity"], package_digest="0" * 64)
        self.rewrite("manifest.json", cap.canonical_bytes(dict(m, identity=ident)))
        self.refused("capsule_manifest_invalid")
        files = [list(f) for f in m["identity"]["files"]]
        files[0][1] += 1
        ident = dict(m["identity"], files=files)
        forged = dict(m, identity=ident, object_id=hashlib.sha256(cap.canonical_bytes(ident)).hexdigest())
        self.rewrite("manifest.json", cap.canonical_bytes(forged))
        with self.assertRaises(GateRefusal):
            cap.verify_capsule_dir(self.loc, forged["object_id"], name_dir=os.path.join(os.path.dirname(self.loc), forged["object_id"]))

    def test_strict_json_dup_nonfinite_unknown(self):
        raw = Path(self.loc, "manifest.json").read_bytes()
        for data in (raw[:-1] + b',"schema":"x"}', b'{"schema":NaN}', raw[:-1] + b',"zz":1}', b"[" * 40 + b"]" * 40, b"\xff"):
            self.rewrite("manifest.json", data)
            self.refused("capsule_manifest_invalid")

    def test_source_manifest_mismatch(self):
        self.rewrite("source-manifest.json", cap.canonical_bytes({"files": {"pyproject.toml": "0" * 64}, "note": "x"}))
        self.refused("capsule_manifest_sha_mismatch")

    def test_expected_manifest_sha_and_wrong_id(self):
        with self.assertRaises(GateRefusal) as c:
            cap.verify_capsule(self.ns, self.id, expected_manifest_sha256="0" * 64)
        self.assertEqual(c.exception.code, "capsule_manifest_sha_mismatch")
        with self.assertRaises(GateRefusal):
            cap.verify_capsule(self.ns, "1" * 64)

    def test_deep_blob_mismatch_detected_only_by_deep_verifier(self):
        m = json.loads(Path(self.loc, "manifest.json").read_bytes())
        files = [list(f) for f in m["identity"]["files"]]
        files[0][3] = "1" * 40
        ident = dict(m["identity"], files=files)
        newid = hashlib.sha256(cap.canonical_bytes(ident)).hexdigest()
        new = os.path.join(os.path.dirname(self.loc), newid)
        shutil.copytree(self.loc, new, symlinks=True)
        os.chmod(new, 0o755)
        p = os.path.join(new, "manifest.json")
        os.chmod(p, 0o644)
        Path(p).write_bytes(cap.canonical_bytes(dict(m, identity=ident, object_id=newid)))
        os.chmod(p, 0o444)
        os.chmod(new, 0o555)
        cap.verify_capsule(self.ns, newid)  # closure verifier does not re-derive blob ids
        with self.assertRaises(RuntimeRefusal) as c:
            seal.verify_leaf_capsule(new, newid)
        self.assertEqual(c.exception.code, "object_tamper")

    def test_admission_before_body(self):
        seen = []

        def admit(n, what):
            seen.append(what)
            raise GateRefusal("cold_memory", what)
        with self.assertRaises(GateRefusal) as c:
            cap.verify_capsule(self.ns, self.id, admit=admit)
        self.assertEqual(c.exception.code, "cold_memory")
        self.assertTrue(seen[-1].startswith("reading capsule manifest.json") or "census" in seen[-1])

    def test_manifest_size_cap_before_body(self):
        with self.assertRaises(GateRefusal) as c:
            cap.verify_capsule_dir(self.loc, self.id, GateLimits(max_manifest_bytes=10))
        self.assertEqual(c.exception.code, "capsule_manifest_invalid")

    def test_namespace_layout(self):
        os.mkdir(os.path.join(self.ns, "stray"))
        with self.assertRaises(GateRefusal) as c:
            cap.validate_namespace(self.ns)
        self.assertEqual(c.exception.code, "namespace_unexpected_entry")
        os.rmdir(os.path.join(self.ns, "stray"))
        Path(self.ns, "objects", "junk").write_text("x")
        with self.assertRaises(GateRefusal):
            cap.validate_namespace(self.ns)

    def test_symlinked_capsule_dir(self):
        link = os.path.join(self.ns, "objects", "2" * 64)
        os.symlink(self.loc, link)
        with self.assertRaises(GateRefusal):
            cap.verify_capsule(self.ns, "2" * 64)


class Correction01(Base):
    def setUp(self):
        super().setUp()
        r = self.seal()
        self.loc, self.id = r["location"], r["object_id"]

    def test_no_listdir_and_admission_precedes_any_enumeration(self):
        from unittest import mock
        events = []
        real_listdir, real_scandir = os.listdir, os.scandir

        def admit(n, what):
            events.append("admission")
            raise GateRefusal("cold_memory", what)
        with mock.patch.object(os, "listdir", side_effect=lambda *a, **k: (events.append("listdir"), real_listdir(*a, **k))[1]), \
                mock.patch.object(os, "scandir", side_effect=lambda *a, **k: (events.append("scandir"), real_scandir(*a, **k))[1]):
            with self.assertRaises(GateRefusal):
                cap.verify_capsule(self.ns, self.id, admit=admit)
        self.assertEqual(events, ["admission"])  # Root's probe saw ['listdir','listdir','admission']
        with mock.patch.object(os, "listdir", side_effect=AssertionError("listdir used")):
            cap.verify_capsule(self.ns, self.id)  # a full successful verification never lists a directory unboundedly

    def test_oversized_namespace_refused_while_iterating(self):
        for i in range(60):
            os.mkdir(os.path.join(self.ns, "objects", f"{i:064x}"))
        calls = []
        with self.assertRaises(GateRefusal) as c:
            cap.validate_namespace(self.ns, admit=lambda n, w: calls.append(w), max_objects=10)
        self.assertEqual(c.exception.code, "namespace_too_many_objects")
        self.assertLessEqual(len(calls), 10 + 3)  # admissions stop at the cap; the other 50 entries were never retained
        with self.assertRaises(GateRefusal) as c:
            cap.validate_namespace(self.ns, max_objects=0)
        self.assertEqual(c.exception.code, "namespace_invalid")

    def test_oversized_capsule_top_and_source_dirs(self):
        os.chmod(self.loc, 0o755)
        for i in range(30):
            Path(self.loc, f"x{i}").write_text("x")
        os.chmod(self.loc, 0o555)
        calls = []
        with self.assertRaises(GateRefusal) as c:
            cap.verify_capsule(self.ns, self.id, admit=lambda n, w: calls.append(w))
        self.assertEqual(c.exception.code, "capsule_entries")
        self.assertLessEqual(len(calls), 3 + 3 + 3)
        os.chmod(self.loc, 0o755)
        for i in range(30):
            os.remove(os.path.join(self.loc, f"x{i}"))
        os.chmod(self.loc, 0o555)
        with self.assertRaises(GateRefusal) as c:
            cap.verify_capsule(self.ns, self.id, GateLimits(max_dir_entries=1))
        self.assertEqual(c.exception.code, "capsule_entries")

    def test_zero_and_short_admission_before_each_stage(self):
        stages = []

        def record(n, what):
            stages.append(what)
        cap.verify_capsule(self.ns, self.id, admit=record)
        for k in range(len(stages)):
            seen = []

            def admit(n, what, k=k):
                seen.append(what)
                if len(seen) > k:
                    raise GateRefusal("cold_memory", what)
            with self.assertRaises(GateRefusal) as c:
                cap.verify_capsule(self.ns, self.id, admit=admit)
            self.assertEqual(c.exception.code, "cold_memory", (k, seen))

    def test_per_file_size_swap_rejected_by_closure_and_deep(self):
        m = json.loads(Path(self.loc, "manifest.json").read_bytes())
        files = [list(f) for f in m["identity"]["files"]]
        files[0][1] += 1
        files[1][1] -= 1  # total unchanged
        ident = dict(m["identity"], files=files)
        newid = hashlib.sha256(cap.canonical_bytes(ident)).hexdigest()
        new = os.path.join(os.path.dirname(self.loc), newid)
        shutil.copytree(self.loc, new, symlinks=True)
        os.chmod(new, 0o755)
        p = os.path.join(new, "manifest.json")
        os.chmod(p, 0o644)
        Path(p).write_bytes(cap.canonical_bytes(dict(m, identity=ident, object_id=newid)))
        os.chmod(p, 0o444)
        os.chmod(new, 0o555)
        with self.assertRaises(GateRefusal) as c:
            cap.verify_capsule(self.ns, newid)
        self.assertEqual(c.exception.code, "snapshot_differs")
        with self.assertRaises(RuntimeRefusal):
            seal.verify_leaf_capsule(new, newid)
        cap.verify_capsule(self.ns, self.id)  # control: the genuine capsule still verifies

    def test_deep_verifier_does_not_reread_manifest_unbounded(self):
        from unittest import mock
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("unbounded read")):
            seal.verify_leaf_capsule(self.loc, self.id)

    def test_read_file_bounded_chunks_and_joins(self):
        from unittest import mock
        empty, small = Path(self.tmp, "empty"), Path(self.tmp, "small")
        empty.write_bytes(b"")
        small.write_bytes(b"abc")
        sizes = []
        real_read = os.read
        with mock.patch.object(os, "read", side_effect=lambda fd, n: (sizes.append(n), real_read(fd, n))[1]):
            self.assertEqual(cap._read_file(str(empty), 10, lambda n, w: None, "e", "c"), b"")
            self.assertEqual(cap._read_file(str(small), 10, lambda n, w: None, "s", "c"), b"abc")
        self.assertEqual(sizes, [1, 4, 1])  # never a fixed 1 MiB buffer for a tiny file

    def test_ignored_runtime_input_added_after_observation(self):
        Path(self.repo, ".git/info/exclude").write_text("leaf/late.py\n")
        Path(self.repo, "leaf/__init__.py").write_text("A = 3\n")  # new content => a new capsule id (the setUp capsule would otherwise be reused)
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "n")
        obs = self.observe(rev=git(self.repo, "rev-parse", "HEAD"))
        Path(self.repo, "leaf/late.py").write_text("x")  # ignored: invisible to git status
        with self.assertRaises(RuntimeRefusal) as c:
            self.seal(obs, run="late")
        self.assertEqual(self.code(c), "source_changed_during_seal")
        self.assertIn("retained_staging", c.exception.evidence)
        self.assertTrue(os.path.isdir(c.exception.evidence["retained_staging"]))

    def test_tool_rejects_negative_numbers_before_any_read(self):
        tool = str(ROOT / "tools" / "masy_leaf_capsule.py")
        ns = os.path.join(self.tmp, "never-created")
        for extra in (["--reserve-bytes", "-1"], ["--reserve-bytes", "0", "--available-cap-bytes", "-5"]):
            args = [a for a in ["--namespace", ns, "--capsule-id", self.id] + extra]
            if "--reserve-bytes" not in extra:
                args += ["--reserve-bytes", "0"]
            out = subprocess.run([sys.executable, "-I", "-B", tool, "verify", *args], capture_output=True, text=True)
            self.assertEqual(out.returncode, 2, out.stderr)
            self.assertEqual(json.loads(out.stdout)["code"], "invalid_argument")
        self.assertFalse(os.path.exists(ns))
        zero = subprocess.run([sys.executable, "-I", "-B", tool, "verify", "--namespace", self.ns, "--capsule-id", self.id, "--reserve-bytes", "0", "--available-cap-bytes", "0"],
                              capture_output=True, text=True)
        self.assertEqual(json.loads(zero.stdout)["code"], "cold_memory")  # zero cap is valid and refuses every admission


class ClosureAndTool(Base):
    def test_closed_import_closure(self):
        code = ("import sys; sys.path.insert(0, %r); import masy.leaf_capsule; "
                "bad=[m for m in sys.modules if m.split('.')[0] in ('numpy','numba','cupy','scipy','maple','maple_syrup','leaf') or m in ('masy.runtime','masy.sourceobs','masy.runtime_worker')]; "
                "ok={m for m in sys.modules if m=='masy' or m.startswith('masy.')}; print(bad, sorted(ok))" % str(ROOT / "src"))
        out = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, check=True).stdout
        self.assertTrue(out.startswith("[] "), out)
        self.assertEqual(out.split("] ", 1)[1].strip(), "['masy', 'masy.leaf_capsule', 'masy.native_source_gate']")

    def test_tool_seal_and_verify_cli(self):
        tool = str(ROOT / "tools" / "masy_leaf_capsule.py")
        common = ["--namespace", self.ns, "--reserve-bytes", "0"]
        out = subprocess.run([sys.executable, "-I", "-B", tool, "seal", *common, "--leaf-root", self.repo, "--expected-revision", self.head, "--run-id", "t1", "--wrapper-root", self.wrapper],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr + out.stdout)
        cid = json.loads(out.stdout)["object_id"]
        out = subprocess.run([sys.executable, "-I", "-B", tool, "verify", *common, "--capsule-id", cid], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        out = subprocess.run([sys.executable, "-I", "-B", tool, "verify", "--namespace", self.ns, "--reserve-bytes", str(1 << 62), "--capsule-id", cid], capture_output=True, text=True)
        self.assertEqual(out.returncode, 2)
        self.assertEqual(json.loads(out.stdout)["code"], "cold_memory")

    def test_atomic_helper_no_replace(self):
        a, b = Path(self.tmp, "a"), Path(self.tmp, "b")
        a.mkdir()
        b.mkdir()
        with self.assertRaises(FileExistsError):
            seal._rename_noreplace(a, b)


if __name__ == "__main__":
    unittest.main()
