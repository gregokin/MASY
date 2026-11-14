"""Pure metadata tests: owned synthetic Git source, never native/provider imports.

Run with unittest under Root's bounded controller. These fixtures are source
identity examples, not fake native model owners or physical qualification.
"""
from __future__ import annotations

import importlib.machinery
import json
import os
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from masy import triplet_lock as tl
from masy import triplet_runtime as tr


class TripletSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repos, self.sources = {}, {}
        for name in tl.MODELS:
            root = self.root / name
            root.mkdir()
            self.repos[name] = root
            package = root / ("leaf" if name == "leaf" else "src/" + tl.PACKAGE[name])
            package.mkdir(parents=True)
            (package / "__init__.py").write_text('__version__ = "0.1.0"\n')
            (package / "sample.py").write_text('VALUE = "committed"\n')
            (root / "pyproject.toml").write_text('[project]\nname = ' + json.dumps(tl.PROJECT[name]) + '\nversion = "0.1.0"\n')
            if name == "leaf":
                (root / "benchmarks").mkdir()
                (root / "benchmarks/spatial_mapping.py").write_text('HELPER = "pinned"\n')
            if name == "masy":
                (root / "tools").mkdir()
                (root / "tools/masy_triplet.py").write_bytes((Path(__file__).resolve().parents[1] / "tools/masy_triplet.py").read_bytes())
                for module in ("triplet_lock.py", "triplet_runtime.py"):
                    (package / module).write_bytes((Path(__file__).resolve().parents[1] / "src/masy" / module).read_bytes())
            self.git(name, "init", "-q", "--initial-branch=main")
            self.git(name, "add", ".")
            self.git(name, "-c", "user.name=Source Test", "-c", "user.email=source@example.invalid", "commit", "-qm", "source")
            self.sources[name] = {"repo": str(root), "ref": "refs/heads/main"}
        self.specs = tl.resolve(self.sources)["sources"]
        for spec in self.specs.values():
            spec.pop("active_excluded")

    def git(self, name, *args):
        return subprocess.run(["git", "--no-optional-locks", "-C", str(self.repos[name]), *args],
                              env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5).stdout

    def test_explicit_ref_uses_committed_objects_and_records_excluded_dirty_work(self):
        path = self.repos["maple"] / "src/maple/sample.py"
        path.write_text('VALUE = "dirty active work"\n')
        selected = tl.resolve(self.sources)
        self.assertEqual(selected["status"], "UNQUALIFIED_OBSERVATION")
        spec = selected["sources"]["maple"]
        self.assertEqual(spec["files"], self.specs["maple"]["files"])
        self.assertIn("src/maple/sample.py", [r[0] for r in spec["active_excluded"]["tracked_worktree"]])
        row = next(r for r in spec["active_excluded"]["tracked_worktree"] if r[0] == "src/maple/sample.py")
        self.assertEqual(row[3], tl.digest(path.read_bytes()))
        self.assertFalse(spec["active_excluded"]["git_status_or_clean_filters_used"])
        self.assertTrue(spec["active_excluded"]["excluded_from_committed_snapshot"])
        self.assertEqual(path.read_text(), 'VALUE = "dirty active work"\n')
        self.assertFalse(any(n in sys.modules for n in ("maple", "maple_syrup", "leaf", "numpy", "cupy")))

    def test_advancing_qualified_branch_requires_new_qualification(self):
        path = self.repos["maple"] / "src/maple/sample.py"
        path.write_text('VALUE = "new commit"\n')
        self.git("maple", "add", ".")
        self.git("maple", "-c", "user.name=Source Test", "-c", "user.email=source@example.invalid", "commit", "-qm", "advance")
        with self.assertRaises(tl.TripletLockError) as ctx:
            tl._current({"sources": self.specs, "limits": tl.DEFAULT_LIMITS})
        self.assertEqual(ctx.exception.code, "ref_advanced")

    def test_leaf_helper_and_namespace_are_part_of_selected_source(self):
        leaf = self.specs["leaf"]
        row = next(r for r in leaf["files"] if r[0] == "benchmarks/spatial_mapping.py")
        self.assertEqual(row[2], tl.digest(b'HELPER = "pinned"\n'))
        (self.repos["leaf"] / "benchmarks/__init__.py").write_text('# concrete package\n')
        self.git("leaf", "add", ".")
        self.git("leaf", "-c", "user.name=Source Test", "-c", "user.email=source@example.invalid", "commit", "-qm", "namespace change")
        with self.assertRaises(tl.TripletLockError) as ctx:
            tl.resolve(self.sources)
        self.assertEqual(ctx.exception.code, "helper_namespace")

    def test_snapshot_drift_refuses_without_publishing_a_lock(self):
        qpath, qsha, q = self.qualification()
        destination = self.root / "new_code"
        original = tl._git
        def changed(repo, args, limits, **kw):
            data = original(repo, args, limits, **kw)
            if args[:2] == ["cat-file", "blob"] and destination.exists():
                return data + b"# changed during copy\n"
            return data
        with patch.object(tl, "_git", changed), self.assertRaises(tl.TripletLockError) as ctx:
            tl.prepare(qpath, expected_sha256=qsha, destination=str(destination))
        self.assertEqual(ctx.exception.code, "source_changed")
        self.assertTrue(destination.exists())  # truthful partial evidence kept
        self.assertFalse((destination / "triplet_lock.json").exists())

    def test_tampered_pinned_helper_is_refused_by_snapshot_verification(self):
        objects = self.root / "objects"
        objects.mkdir()
        obj = tl._snapshot("leaf", self.specs["leaf"], objects, tl.DEFAULT_LIMITS)
        helper = Path(obj["path"]) / "source/benchmarks/spatial_mapping.py"
        helper.chmod(0o644)
        helper.write_text('HELPER = "tampered"\n')
        helper.chmod(0o444)
        with self.assertRaises(tl.TripletLockError) as ctx:
            tl._verify_object("leaf", self.specs["leaf"], obj, tl.DEFAULT_LIMITS)
        self.assertEqual(ctx.exception.code, "source_changed")

    def test_actual_foreign_import_origin_is_refused_before_source_read(self):
        lock = {"objects": {n: {"path": str(self.root / "objects" / n)} for n in tl.MODELS},
                "qualification": {"limits": tl.DEFAULT_LIMITS, "sources": self.specs,
                                  "maple_bundle": {"target": str(self.root / "bundle")}}}
        module = ModuleType("leaf.sample")
        module.__file__ = str(self.repos["leaf"] / "leaf/sample.py")
        module.__spec__ = importlib.machinery.ModuleSpec("leaf.sample", loader=None, origin=module.__file__)
        with self.assertRaises(tl.TripletLockError) as ctx:
            tr.check_import_sources(lock, {"leaf.sample": module})
        self.assertEqual(ctx.exception.code, "module_origin")

    def test_qualification_hash_is_external_and_zero_never_qualifies(self):
        p = self.root / "unqualified.json"
        p.write_bytes(b'{}\n')
        for sha, expected in (("0" * 64, "unqualified_profile"), ("1" * 64, "qualification_sha256")):
            with self.subTest(sha=sha), self.assertRaises(tl.TripletLockError) as ctx:
                tl.load_qualification(p, expected_sha256=sha)
            self.assertEqual(ctx.exception.code, expected)
        with self.assertRaises(tl.TripletLockError):
            tl._rel("../outside.py")

    def test_alternate_benchmarks_root_is_refused(self):
        expected = self.root / "objects/leaf/source/benchmarks"
        expected.mkdir(parents=True)
        (expected / "spatial_mapping.py").write_text('# source\n')
        other = self.root / "other/benchmarks"
        other.mkdir(parents=True)
        (other / "spatial_mapping.py").write_text('# other\n')
        lock = {"objects": {"leaf": {"path": str(self.root / "objects/leaf")}}}
        with self.assertRaises(tl.TripletLockError) as ctx:
            tr._namespace(lock, [str(expected.parent), str(other.parent)])
        self.assertEqual(ctx.exception.code, "helper_namespace")

    def qualification(self):
        # Synthetic Source metadata only: never import these fake native package
        # fixtures, never invoke native check_runtime or claim physical authority.
        target = self.root / ("e" * 64)
        target.mkdir()
        rows = []
        for rel, _, _, blob in self.specs["maple"]["files"]:
            data = self.git("maple", "cat-file", "blob", blob)
            path = target / "source" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            rows.append({"path": "source/" + rel, "size": len(data), "sha256": tl.digest(data)})
        distribution = {"METADATA": b"Metadata-Version: 2.1\nName: maple\nVersion: 0.1.0\n",
                        "WHEEL": b"Wheel-Version: 1.0\n", "top_level.txt": b"maple\n"}
        for name, data in distribution.items():
            path = target / "distribution" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)
            rows.append({"path": "distribution/" + name, "size": len(data), "sha256": tl.digest(data)})
        record = {"schema": "maple_syrup.dependency_bundle.v1", "bundle_id": target.name,
                  "origin": {"original_source": {"revision": self.specs["maple"]["commit"]}, "patch_chain": []},
                  "project": {"name": "maple", "version": "0.1.0", "package_source_digest": {"digest_sha256": self.specs["maple"]["package_sha256"]}},
                  "files": rows}
        for rel, data in tl._native_overlay(record, target, distribution).items():
            path = target / "editable" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        manifest = target / "bundle-manifest.json"
        manifest.write_bytes(tl.canonical(record))
        (target / "materialization.json").write_bytes(tl.canonical({
            "schema": "maple_syrup.dependency_materialization.v1", "bundle_id": target.name, "target": str(target),
            "layout": {"source": "source/", "distribution": "distribution/", "overlay": "editable/", "manifest": "bundle-manifest.json"}}))
        q = {"schema": tl.QUALIFICATION_SCHEMA, "profile": tl.PROFILE, "status": "qualified", "sources": self.specs,
             "limits": dict(tl.DEFAULT_LIMITS), "entrypoints": [], "native_profile": "synthetic_source_metadata_only",
             "interpreter": {"path": tl.PYTHON, "sha256": tl.file_reference(str(Path(tl.PYTHON).resolve()), tl.DEFAULT_LIMITS["max_source_bytes"])["sha256"],
                             "version": list(sys.version_info[:3])},
             "maple_bundle": {"target": str(target), "manifest": tl.file_reference(str(manifest), tl.DEFAULT_LIMITS["max_manifest_bytes"])}}
        path = self.root / "qualification.json"
        path.write_bytes(tl.canonical(q))
        return path, tl.digest(path.read_bytes()), q

    def test_source_prepare_load_verify_roundtrip_and_fresh_pycache_command(self):
        qpath, qsha, q = self.qualification()
        destination = self.root / "code"
        receipt = tl.prepare(qpath, expected_sha256=qsha, destination=str(destination))
        ref = receipt["lock"]
        first = tl.load(ref["path"], expected_sha256=ref["sha256"])
        checked = tl.verify(ref["path"], expected_sha256=ref["sha256"])
        self.assertEqual(first, checked)
        tool = Path(__file__).resolve().parents[1] / "tools/masy_triplet.py"
        cli = subprocess.run([tl.PYTHON, "-I", "-B", str(tool), "verify", "--lock", ref["path"], "--lock-sha256", ref["sha256"]],
                             check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        self.assertEqual(json.loads(cli.stdout)["status"], "source_verified")
        self.assertEqual(checked["qualification"]["sources"], q["sources"])
        self.assertFalse(checked["interpretation"]["transport_qualified_by_lock"])
        run = self.root / "operational"
        run.mkdir()
        command = tr.bootstrap_spec(ref["path"], expected_sha256=ref["sha256"], run_dir=str(run))
        self.assertIn("-X", command["argv"])
        self.assertIn("pycache_prefix=" + str(run / "pycache"), command["argv"])
        self.assertEqual(list((run / "pycache").iterdir()), [])
        with self.assertRaises(tl.TripletLockError) as ctx:
            tr.bootstrap_spec(ref["path"], expected_sha256=ref["sha256"], run_dir=str(run))
        self.assertEqual(ctx.exception.code, "pycache")
        with self.assertRaises(tl.TripletLockError) as ctx:
            tl.prepare(qpath, expected_sha256=qsha, destination=str(destination))
        self.assertEqual(ctx.exception.code, "output_exists")

    def test_bundle_shadow_and_bytecode_extras_refuse_before_native_import(self):
        _, _, q = self.qualification()
        target = Path(q["maple_bundle"]["target"])
        for rel in ("source/src/subprocess.py", "editable/numpy/__init__.py", "source/src/maple/__pycache__/sample.cpython-312.pyc"):
            with self.subTest(rel=rel):
                path = target / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"UNVERIFIED EXTRA")
                with self.assertRaises(tl.TripletLockError) as ctx:
                    tl._bundle(q)
                self.assertEqual(ctx.exception.code, "bundle_extra")
                path.unlink()
                parent = path.parent
                while parent != target and not any(parent.iterdir()):
                    parent.rmdir(); parent = parent.parent
        self.assertFalse(any(n in sys.modules for n in ("maple", "maple_syrup", "leaf", "numpy")))

    def test_repo_local_fsmonitor_and_clean_driver_are_never_executed(self):
        marker = self.root / "driver_ran"
        script = self.root / "driver.sh"
        script.write_text("#!/bin/sh\nprintf ran > " + str(marker) + "\ncat\n")
        script.chmod(0o755)
        self.git("maple", "config", "core.fsmonitor", str(script))
        self.git("maple", "config", "filter.triplet.clean", str(script))
        (self.repos["maple"] / ".gitattributes").write_text("src/maple/*.py filter=triplet\n")
        (self.repos["maple"] / "src/maple/sample.py").write_text("dirty bytes\n")
        selected = tl.resolve(self.sources)
        self.assertFalse(marker.exists())
        self.assertFalse(selected["sources"]["maple"]["active_excluded"]["git_status_or_clean_filters_used"])

    def test_sealed_import_roots_follow_stdlib_before_site_tail(self):
        shadow = self.root / "shadow"
        shadow.mkdir()
        (shadow / "calendar.py").write_text("raise AssertionError('shadow executed')\n")
        standard = tr._standard_paths(sys.path)
        self.assertTrue(standard)
        tail = str(self.root / "site-packages")
        with patch.object(sys, "path", standard + [tail]):
            tr.install_prefix([str(shadow)])
            self.assertEqual(sys.path[:len(standard)], standard)
            self.assertEqual(sys.path[len(standard):], [str(shadow), tail])
            spec = importlib.machinery.PathFinder.find_spec("calendar", sys.path)
            self.assertIsNotNone(spec)
            self.assertNotEqual(spec.origin, str(shadow / "calendar.py"))


if __name__ == "__main__":
    unittest.main()
