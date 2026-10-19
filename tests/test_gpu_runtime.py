"""Actual source capture and synthetic profile tests, no model/GPU execution."""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from masy import gpu_runtime as g

ROOT = Path(g.__file__).resolve().parents[2]
REPO = "/home/okin/MASY"
BASE = "22aa6ab65ad73a58fb372ffe09e3f7e0ca9893a1"
GPU = "GPU-5c8bd3d4-2885-82c6-a0f0-ad2b7f87b871"
GPU_MODULES = ("src/masy/gpu_runtime.py", "src/masy/gpu_runtime_worker.py")
GPU_LAUNCHER = "tools/masy_gpu_runtime.py"


def source_fixture(tree):
    """Small historical source fixture, independent of later accepted files.

    Git plumbing reads immutable22aa blobs directly, independently of the
    runtime's source walker. Only three current GPU source files are copied;
    no live workspace directory, saved payload or cache is traversed.
    """
    tree.mkdir()
    listing = subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "-z", BASE, "--",
        "pyproject.toml", "src/masy", "tools/masy_runtime.py"],
        check=True, capture_output=True, timeout=10).stdout
    baseline = set()
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        info, relative = entry.split(b"\t", 1)
        mode, kind, _blob = info.split()
        if mode != b"100644" or kind != b"blob":
            raise AssertionError("fixture requires regular immutable source blobs")
        relative = relative.decode()
        data = subprocess.run(["git", "-C", REPO, "show", f"{BASE}:{relative}"],
            check=True, capture_output=True, timeout=10).stdout
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        baseline.add(relative)
    if not {"pyproject.toml", "src/masy/__init__.py", "tools/masy_runtime.py"} <= baseline:
        raise AssertionError("incomplete immutable source fixture")
    for relative in (*GPU_MODULES, GPU_LAUNCHER):
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    return baseline | set(GPU_MODULES)


def config(root=ROOT, store="/tmp/masy18-synthetic-store"):
    return {"schema": g.CONFIG_SCHEMA,
        "sources": {"maple": {"root": "/home/okin/MAPLE", "expected_revision": "39bcf79e1e67b6ed960f78a923ddfedf1d173e68",
            "distribution_metadata": "/home/okin/MASY/agent_handoffs/tasks/masy08_storage_pair_20261019/metadata_build/distribution/maple-0.0.1.dist-info"},
            "syrup": {"root": "/home/okin/SYRUP", "expected_revision": "d05d8c6c66d9fd2969c007681916254e8d1c2c7c"},
            "masy": {"mode": "development_snapshot", "root": str(root), "base_repository": REPO, "base_revision": BASE}},
        "runtime": {"python": g.worker.PYTHON, "store": store},
        "device": {"uuid": GPU, "pci_bus_id": "00000000:17:00.0", "visible_ordinal": 0},
        "limits": {}, "metadata_receipt": None}


class ProfileTests(unittest.TestCase):
    def test_closed_config_rejects_rebinding_or_unknown_fields(self):
        parsed = g.parse_gpu_config(config())
        self.assertEqual(parsed.native.python, g.worker.PYTHON)
        for mutate in (lambda v: v.update(extra=True),
                lambda v: v["runtime"].update(python="/home/okin/MAPLE/.venv/bin/python"),
                lambda v: v["runtime"].update(store=g.cpu.DEFAULT_STORE),
                lambda v: v["sources"]["masy"].update(mode="implicit_dirty")):
            changed = config()
            mutate(changed)
            with self.assertRaises(g.Refusal):
                g.parse_gpu_config(changed)

    def test_actual_complete_snapshot_preserves_base_and_exact_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            expected = source_fixture(tree)
            parsed = g.parse_gpu_config(config(root=tree))
            first, files, captured = g.observe_masy(parsed.masy, parsed.native.limits)
            second, again, _captured = g.observe_masy(parsed.masy, parsed.native.limits)
            self.assertEqual(first, second)
            self.assertEqual(files, again)
            self.assertEqual(first["source_status"], "development_snapshot")
            self.assertIs(first["committed"], False)
            self.assertEqual(first["base_revision"], BASE)
            self.assertEqual(first["delta"], {"added": sorted(GPU_MODULES), "modified": [], "deleted": []})
            self.assertEqual(set(captured), {"pyproject.toml", "src/masy/__init__.py"})
            self.assertEqual({r[0] for r in files}, expected)

    def test_snapshot_refuses_modified_deleted_extra_or_symlink(self):
        for mode in ("modified", "deleted", "extra", "symlink"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                tree = Path(tmp) / "tree"
                source_fixture(tree)
                path = tree / "src/masy/runtime.py"
                if mode == "modified":
                    path.write_text(path.read_text() + "\n# synthetic edit\n")
                elif mode == "deleted":
                    path.unlink()
                elif mode == "extra":
                    (tree / "src/masy/unknown.py").write_text("# synthetic\n")
                else:
                    path.unlink()
                    path.symlink_to(ROOT / "src/masy/runtime.py")
                parsed = g.parse_gpu_config(config(root=tree))
                with self.assertRaises(g.Refusal):
                    g.observe_masy(parsed.masy, parsed.native.limits)

    def test_snapshot_is_a_complete_native_source_object_not_import_stitching(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = str(Path(tmp) / "store")
            g._ensure_store(store)
            tree = Path(tmp) / "tree"
            source_fixture(tree)
            parsed = g.parse_gpu_config(config(root=tree, store=store))
            observed, files, captured = g.observe_masy(parsed.masy, parsed.native.limits)
            sealed = g.cpu.seal_source(store, "masy", observed, files, captured, run_id="synthetic", limits=parsed.native.limits)
            checked = g.cpu.verify_source_object(sealed["path"], sealed["object_id"], parsed.native.limits)
            self.assertEqual(checked["package_digest"], sealed["package_digest"])
            source = Path(sealed["path"]) / "source/src/masy"
            self.assertTrue((source / "runtime.py").is_file())
            self.assertTrue((source / "gpu_runtime_worker.py").is_file())
            self.assertEqual(g._bootstrap(sealed), g.worker.BOOTSTRAP)
            self.assertNotEqual(sealed["object_id"], "4c5c9897b05b10ff5d1810b12b53c9eb65f62cb348c7087fca8ae0cba2dd61fa")
            self.assertFalse(any(source.rglob("*.pyc")))

    def test_noninstalled_metadata_needs_genuine_bound_receipt(self):
        parsed = g.parse_gpu_config(config())
        # Immutable historical source fixture, independent of parallel native
        # releases. This is not a current-HEAD runtime selection.
        object_path = Path(REPO) / "runs/runtime-store/objects/sources/208bb26ccb6410b52f5c8d94a1445631a28af5dda3f3c72179c6b6a1e693def0"
        g.cpu.verify_source_object(object_path, object_path.name, parsed.native.limits)
        files = g._read_json(object_path / "manifest.json")["identity"]["files"]
        observed = {"head": "39bcf79e1e67b6ed960f78a923ddfedf1d173e68"}
        with self.assertRaises(g.Refusal):
            g._metadata(parsed, observed, files)
        receipt = Path(REPO) / "agent_handoffs/tasks/masy08_storage_pair_20261019/metadata_build/build_receipt.json"
        raw = config()
        raw["metadata_receipt"] = {"path": str(receipt), "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}
        selected = g.parse_gpu_config(raw)
        metadata = g._metadata(selected, observed, files)
        self.assertEqual(metadata["origin"], "wheel_build_receipt")
        changed = copy.deepcopy(observed)
        changed["head"] = "0" * 40
        with self.assertRaises(g.Refusal):
            g._metadata(selected, changed, files)

    def test_publication_no_replace_and_store_unknown_entry_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = str(Path(tmp) / "store")
            g._ensure_store(store)
            value, path = g._publish(store, "profiles", {"profile_id": "", "synthetic": True}, "profile_id", "one")
            first = path.read_bytes()
            again, same = g._publish(store, "profiles", {"profile_id": "", "synthetic": True}, "profile_id", "two")
            self.assertEqual(value, again)
            self.assertEqual(path, same)
            self.assertEqual(path.read_bytes(), first)
            self.assertTrue((Path(store) / "profiles/.staging-two.json").exists())
            (Path(store) / "objects/sources/.unrecognized").mkdir()
            with self.assertRaises(g.Refusal):
                g.validate_store(store)
            (Path(store) / "objects/sources/.unrecognized").rmdir()
            (Path(store) / "unexpected").write_text("synthetic")
            with self.assertRaises(g.Refusal):
                g.validate_store(store)

    def test_referenced_raw_stream_changes_refuse_independently_of_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = str(Path(tmp) / "store")
            g._ensure_store(store)
            run = g._Run(store, "synthetic-evidence")
            children = []
            for index, operation in enumerate(("export", "materialize", "audit", "audit"), 1):
                child = {"operation": operation, "status": "ok", "exit_code": 0, "reaped": True,
                    "pid": 100 + index, "process": {"pid": 100 + index, "start_ticks": "123"}}
                for key in ("request", "stdout", "stderr"):
                    child[key] = g._file(run.write(f"synthetic-{index}-{key}", {"synthetic": True}))
                children.append(child)
                run.write(f"worker-{index}-receipt.json", child)
            path = run.write("prepare-receipt.json", {"launch": {}, "config": None,
                "before_after_unchanged": True, "workers": children, "disk_preflight": {}})
            record = g._file(path)
            g._verify_preparation_receipt(store, record, g.cpu.RuntimeLimits())
            Path(children[0]["stdout"]["path"]).write_text("synthetic tampering")
            with self.assertRaises(g.Refusal):
                g._verify_preparation_receipt(store, record, g.cpu.RuntimeLimits())
            path.write_text(json.dumps({"synthetic": "replaced receipt"}))
            with self.assertRaises(g.Refusal):
                g._verify_preparation_receipt(store, record, g.cpu.RuntimeLimits())

    def test_snapshot_base_inputs_protected_without_imposing_live_wrapper_output_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            candidate = Path(tmp) / "candidate"
            for path in (base / "src", base / "tools", base / ".git", base / "agent_handoffs/task", candidate):
                path.mkdir(parents=True, exist_ok=True)
            python = {"realpath": str(Path(sys.executable).resolve())}
            g._protected(str(base / "agent_handoffs/task/store"), str(candidate), [], python,
                base_repository=str(base))
            for relative in ("src", "tools", ".git"):
                with self.subTest(relative=relative), self.assertRaises(g.Refusal):
                    g._protected(str(base / relative / "store"), str(candidate), [], python,
                        base_repository=str(base))
    def test_launcher_help_is_native_free_and_does_not_write_source_bytecode(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            source_fixture(tree)
            result = subprocess.run([sys.executable, "-I", "-B", str(tree / GPU_LAUNCHER), "--help"],
                capture_output=True, text=True, timeout=10, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("{prepare,verify,probe}", result.stdout)
            self.assertFalse(any(tree.rglob("*.pyc")))

    def test_cpu_lock_and_product_bytes_unchanged(self):
        expected = {"src/masy/runtime.py": "3eecfd189d096a0fe8f14247096ff4445a0d82620caa61d65184477ae24ed570",
            "src/masy/runtime_worker.py": "5d6bf66576f3e52a7f59dd2515209b3ebde5f8f09fcf1784671d278e5569436e",
            "tools/masy_runtime.py": "57ed19bb80094a1818ee0bcd2bfbab52f6f22dc00b6a58a69abf1b89cc619cf9"}
        for rel, sha in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), sha)
            self.assertEqual(hashlib.sha256((Path(REPO) / rel).read_bytes()).hexdigest(), sha)
        lock = Path(REPO) / "runs/runtime-store/locks/43976b9a6c3aa5384d0cdfdcf7c61f90d6a2f0ced45a0890e92dad5fb6a5cd33.json"
        self.assertEqual(hashlib.sha256(lock.read_bytes()).hexdigest(),
            "867c48b18247b5a2778aeaef75d0191e2536204dcd2766c54a42a5aa23efee55")


if __name__ == "__main__":
    unittest.main()
