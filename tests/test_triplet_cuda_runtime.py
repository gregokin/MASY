"""Native-free source/runtime checks for the explicit CUDA triplet profile.

Synthetic Git repositories and data-only descriptors only: no MAPLE, SYRUP,
LEAF, NumPy or CuPy import, no device, no model. Run with unittest under
Root's bounded controller. AUTHORED, NOT EXECUTED by the runtime author.
"""
from __future__ import annotations

import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from masy import triplet_lock as tl
from masy import triplet_runtime as tr
from masy import triplet_cuda_runtime as tcr

ROOT = Path(__file__).resolve().parents[1]
UUID = "GPU-0123abcd-4567-89ef-0123-456789abcdef"
SHA = "1" * 64


class CudaTripletSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
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
                for rel in tcr.GENERATOR_FILES:
                    (root / rel).write_bytes((ROOT / rel).read_bytes())
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

    def descriptor(self, **changes):
        q = {"schema": tcr.QUALIFICATION_SCHEMA, "profile": tcr.PROFILE, "status": "qualified", "sources": self.specs,
             "limits": dict(tl.DEFAULT_LIMITS), "entrypoints": [], "native_profile": tcr.NATIVE_PROFILE,
             "interpreter": {"path": tcr.PYTHON, "sha256": SHA, "version": [3, 12, 0]},
             "maple_bundle": {"target": str(self.root / "bundle"), "manifest": {"path": "x", "size": 0, "sha256": SHA}},
             "backend": {"name": "cuda", "provider": "cupy"}, "device": {"physical_uuid": UUID, "logical_ordinal": 0}}
        q.update(changes)
        return q

    def test_cuda_descriptor_is_separate_and_closed(self):
        self.assertNotEqual((tcr.QUALIFICATION_SCHEMA, tcr.LOCK_SCHEMA, tcr.PROFILE, tcr.PYTHON),
                            (tl.QUALIFICATION_SCHEMA, tl.LOCK_SCHEMA, tl.PROFILE, tl.PYTHON))
        good = tl._validate_qualification(self.descriptor(), tcr._POLICY)
        self.assertEqual(good["device"]["logical_ordinal"], 0)
        with self.assertRaises(tl.TripletLockError) as ctx:  # CPU validator refuses the CUDA shape
            tl._validate_qualification(self.descriptor())
        self.assertEqual(ctx.exception.code, "schema")
        cases = [({"schema": tl.QUALIFICATION_SCHEMA, "profile": tl.PROFILE}, "unqualified_profile"),
                 ({"interpreter": {"path": tl.PYTHON, "sha256": SHA, "version": [3, 12, 0]}}, "interpreter"),
                 ({"status": "unqualified"}, "unqualified_profile"),
                 ({"native_profile": "native_maple_1dca_case_latitude_cpu_v1"}, "unqualified_profile"),
                 ({"backend": {"name": "cuda", "provider": "numba"}}, "backend"),
                 ({"device": {"physical_uuid": UUID, "logical_ordinal": 1}}, "device"),
                 ({"device": {"physical_uuid": UUID.upper(), "logical_ordinal": 0}}, "device"),
                 ({"device": {"physical_uuid": UUID}}, "schema"),
                 ({"device": {"physical_uuid": UUID, "logical_ordinal": 0, "pci": "x"}}, "schema")]
        for changes, code in cases:
            with self.subTest(changes=changes), self.assertRaises(tl.TripletLockError) as ctx:
                tl._validate_qualification(self.descriptor(**changes), tcr._POLICY)
            self.assertEqual(ctx.exception.code, code)
        p = self.root / "q.json"
        p.write_bytes(tl.canonical(self.descriptor()))
        with self.assertRaises(tl.TripletLockError) as ctx:
            tcr.load_qualification(p, expected_sha256="0" * 64)
        self.assertEqual(ctx.exception.code, "unqualified_profile")
        with self.assertRaises(tl.TripletLockError) as ctx:
            tcr.load_qualification(p, expected_sha256=SHA)
        self.assertEqual(ctx.exception.code, "qualification_sha256")

    def test_draft_is_unqualified_and_records_device(self):
        draft = tcr.draft_qualification(self.sources, interpreter={"path": tcr.PYTHON, "sha256": SHA, "version": [3, 12, 0]},
                                        maple_bundle={"target": "/x", "manifest": {}}, device={"physical_uuid": UUID, "logical_ordinal": 0})
        q = draft["qualification"]
        self.assertEqual((q["schema"], q["profile"], q["status"], q["native_profile"]),
                         (tcr.QUALIFICATION_SCHEMA, tcr.PROFILE, "unqualified", tcr.NATIVE_PROFILE))
        self.assertEqual(q["device"], {"physical_uuid": UUID, "logical_ordinal": 0})
        self.assertIn("leaf", draft["excluded_active_work"])

    def test_cpu_policy_and_bootstrap_are_unchanged(self):
        self.assertEqual(tl.PYTHON, "/home/okin/MAPLE/.venv/bin/python")
        self.assertEqual((tl._CPU.python, tl._CPU.profile, tl._CPU.lock_schema), (tl.PYTHON, tl.PROFILE, tl.LOCK_SCHEMA))
        self.assertEqual(dict(tl._CPU.interpretation)["backend"], "numpy")
        self.assertTrue(tr.BOOTSTRAP.startswith(tr._BOOTSTRAP_HEAD))
        self.assertTrue(tr.BOOTSTRAP.endswith("from masy.triplet_runtime import worker_main\nsys.exit(worker_main(sys.argv[1:]))\n"))
        self.assertIn("triplet_cuda_runtime", tcr.BOOTSTRAP)
        self.assertNotIn("triplet_cuda_runtime", tr.BOOTSTRAP)
        with self.assertRaises(Exception):
            tl._CPU.python = tcr.PYTHON  # frozen; no mutable global switch
        run = self.root / "cpu_run"
        run.mkdir()
        spec = tr._bootstrap_spec({"qualification": {"entrypoints": []}}, "/x/lock.json", SHA, run_dir=str(run),
                                  entrypoint=None, argv=(), policy=tr._CPU)
        self.assertEqual(spec["env"], {"PATH": str(Path(tl.PYTHON).parent) + ":/usr/bin:/bin", "LC_ALL": "C.UTF-8",
                                       "HOME": str(run / "home"), "TMPDIR": str(run / "tmp"), "XDG_CACHE_HOME": str(run / "cache"),
                                       "NUMBA_CACHE_DIR": str(run / "cache/numba"), "MPLCONFIGDIR": str(run / "cache/matplotlib"),
                                       "NUMBA_DISABLE_CUDA": "1", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "GIT_OPTIONAL_LOCKS": "0"})
        self.assertEqual(spec["argv"][:7], [tl.PYTHON, "-I", "-B", "-X", "pycache_prefix=" + str(run / "pycache"), "-c", tr.BOOTSTRAP])
        lock = {"objects": {n: {"path": str(self.root / "objects" / n)} for n in tl.MODELS},
                "qualification": {"limits": tl.DEFAULT_LIMITS, "sources": self.specs, "maple_bundle": {"target": str(self.root / "bundle")}}}
        cupy = ModuleType("cupy")
        for check, code in ((tr.check_import_sources, "backend"), (tcr.check_import_sources, "backend")):
            with self.assertRaises(tl.TripletLockError) as ctx:
                check(lock, {"cupy": cupy})
            self.assertEqual(ctx.exception.code, code)
        foreign = ModuleType("leaf.sample")
        foreign.__file__ = str(self.repos["leaf"] / "leaf/sample.py")
        foreign.__spec__ = importlib.machinery.ModuleSpec("leaf.sample", loader=None, origin=foreign.__file__)
        with self.assertRaises(tl.TripletLockError) as ctx:
            tcr.check_import_sources(lock, {"leaf.sample": foreign})
        self.assertEqual(ctx.exception.code, "module_origin")

    def test_cuda_bootstrap_spec_selects_locked_device_without_launching(self):
        run = self.root / "gpu_run"
        run.mkdir()
        lock = {"qualification": {"entrypoints": ["masy.x:main"], "device": {"physical_uuid": UUID, "logical_ordinal": 0}}}
        spec = tr._bootstrap_spec(lock, "/x/lock.json", SHA, run_dir=str(run), entrypoint="masy.x:main", argv=("a",), policy=tcr._RUNTIME)
        self.assertEqual(spec["argv"][0], tcr.PYTHON)
        self.assertEqual(spec["argv"][6:], [tcr.BOOTSTRAP, "/x/lock.json", SHA, "masy.x:main", '["a"]\n', str(run / "pycache")])
        env = spec["env"]
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], UUID)
        self.assertEqual((env["NUMBA_DISABLE_CUDA"], env["JAX_PLATFORMS"]), ("1", "cpu"))
        self.assertEqual(env["CUPY_CACHE_DIR"], str(run / "cache/cupy"))
        self.assertEqual(env["PATH"].split(":")[0], str(Path(tcr.PYTHON).parent))
        self.assertEqual(sorted(p.name for p in run.iterdir()), ["pycache"])
        self.assertEqual(list((run / "pycache").iterdir()), [])
        with self.assertRaises(tl.TripletLockError) as ctx:
            tr._bootstrap_spec(lock, "/x/lock.json", SHA, run_dir=str(run), entrypoint="masy.other:main", argv=(), policy=tcr._RUNTIME)
        self.assertEqual(ctx.exception.code, "entrypoint")

    def test_device_provider_census_authenticates_origin_only(self):
        site = self.root / "site-packages"
        pkg = site / "cupy"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("# provider\n")
        (pkg / "ext.cpython-312-x86_64-linux-gnu.so").write_bytes(b"\x7fELF")
        def module(name, origin, path=None):
            m = ModuleType(name)
            m.__file__ = origin
            m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, origin=origin)
            if path is not None:
                m.__path__ = path
            return m
        good = {"cupy": module("cupy", str(pkg / "__init__.py"), [str(pkg)]),
                "cupy.ext": module("cupy.ext", str(pkg / "ext.cpython-312-x86_64-linux-gnu.so")), "json": json}
        census = tcr.device_runtime_census(good, provider_root=str(site))
        self.assertEqual(census["cupy"]["sha256"], tl.digest(b"# provider\n"))
        self.assertEqual((census["cupy.ext"]["kind"], census["cupy.ext"]["hashed"], census["cupy.ext"]["sha256"]), ("extension", False, None))
        # A binary is recorded from metadata, without a source-body read. The
        # real CuPy thrust extension is larger than the default64MiB source cap.
        separate = tcr.device_runtime_census({"cupy.ext": good["cupy.ext"]}, provider_root=str(site),
            limits={**tl.DEFAULT_LIMITS, "max_file_bytes": 3})
        self.assertEqual(separate["cupy.ext"]["size"], 4)
        with self.assertRaises(tl.TripletLockError):
            tcr.device_runtime_census({"cupy": good["cupy"]}, provider_root=str(site),
                limits={**tl.DEFAULT_LIMITS, "max_file_bytes": 3})
        other = self.root / "other/cupy"
        other.mkdir(parents=True)
        (other / "__init__.py").write_text("# foreign\n")
        with self.assertRaises(tl.TripletLockError) as ctx:
            tcr.device_runtime_census({"cupy": module("cupy", str(other / "__init__.py"), [str(other)])}, provider_root=str(site))
        self.assertEqual(ctx.exception.code, "backend")
        with self.assertRaises(tl.TripletLockError):
            tcr._compare_census(census, {**census, "cupy": {**census["cupy"], "sha256": "0" * 64}})
        tcr._compare_census(census, {**census, "cupy.new": {"path": "x"}})  # growth after authentication is permitted
        ctx_obj = tcr.CudaRuntimeContext("/x", SHA, None, {"physical_uuid": UUID, "logical_ordinal": 0})
        ctx_obj.record_device_runtime({"observed_uuid": UUID, "compute_capability": [8, 9]})
        self.assertEqual(ctx_obj.device_runtime_reports()[0]["verified_by_runtime"], False)
        for bad in (["list"], {"nan": float("nan")}, {1: "key"}):
            with self.assertRaises(tl.TripletLockError):
                ctx_obj.record_device_runtime(bad)

    def test_provider_lazy_module_is_inspected_without_execution(self):
        import importlib.util
        site = self.root / "lazy-site"
        pkg = site / "cupy"
        pkg.mkdir(parents=True)
        path = pkg / "lazy.py"
        path.write_text("raise AssertionError('census executed lazy module')\n")
        spec = importlib.util.spec_from_file_location("cupy.lazy", path)
        loader = importlib.util.LazyLoader(spec.loader)
        spec.loader = loader
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        result = tcr.device_runtime_census({"cupy.lazy": module}, provider_root=str(site))
        self.assertEqual(result["cupy.lazy"]["sha256"], tl.digest(path.read_bytes()))
        self.assertIs(type(module), importlib.util._LazyModule)

    def test_fft_config_alias_requires_exact_authenticated_owner_singleton(self):
        site = self.root / "alias-site"
        pkg = site / "cupy/fft"
        pkg.mkdir(parents=True)
        path = pkg / "_config.py"
        path.write_text("# labelled synthetic provider metadata only\n")
        owner = ModuleType("cupy.fft._config")
        owner.__file__ = str(path)
        owner.__spec__ = importlib.machinery.ModuleSpec(owner.__name__, loader=None, origin=str(path))
        cls = type("_FFTConfig", (), {"__module__": "cupy.fft._config"})
        owner._FFTConfig = cls
        owner.config = cls()
        modules = {"cupy.fft._config": owner, "cupy.fft.config": owner.config}
        result = tcr.device_runtime_census(modules, provider_root=str(site))
        self.assertEqual(result["cupy.fft.config"]["sha256"], tl.digest(path.read_bytes()))
        for alias in (object(), cls()):
            with self.assertRaises(tl.TripletLockError):
                tcr.device_runtime_census({**modules, "cupy.fft.config": alias}, provider_root=str(site))
        path = pkg / "bad.pyc"
        path.write_bytes(b"unqualified bytecode")
        bad = ModuleType("cupy.fft.bad")
        bad.__file__ = str(path)
        bad.__spec__ = importlib.machinery.ModuleSpec(bad.__name__, loader=None, origin=str(path))
        with self.assertRaises(tl.TripletLockError):
            tcr.device_runtime_census({"cupy.fft.bad": bad}, provider_root=str(site))

    def test_cold_bootstrap_refuses_before_any_masy_import(self):
        sealed = self.root / "sealed/source"
        (sealed / "src/masy").mkdir(parents=True)
        rows = []
        for rel in ("src/masy/__init__.py", *tcr.GENERATOR_FILES):
            data = (ROOT / rel).read_bytes()
            target = sealed / rel
            target.write_bytes(data)
            target.chmod(0o444)
            rows.append([rel, len(data), tl.digest(data), "0" * 40])
        for d in (sealed, sealed / "src", sealed / "src/masy"):
            d.chmod(0o555)
        self.addCleanup(lambda: [d.chmod(0o755) for d in (sealed / "src/masy", sealed / "src", sealed)])
        def lock_file(name, schema=tcr.LOCK_SCHEMA):
            lock = {"schema": schema, "profile": tcr.PROFILE, "objects": {"masy": {"path": str(sealed.parent)}},
                    "qualification": {"sources": {"masy": {"files": rows}}, "device": {"physical_uuid": UUID, "logical_ordinal": 0}}}
            p = self.root / name
            p.write_bytes(tl.canonical(lock))
            return p, tl.digest(p.read_bytes())
        def run(path, sha, env_uuid=UUID):
            cache = self.root / ("cache_" + path.stem)
            cache.mkdir(exist_ok=True)
            env = {"PATH": os.defpath}
            if env_uuid is not None:
                env["CUDA_VISIBLE_DEVICES"] = env_uuid
            return subprocess.run([sys.executable, "-I", "-B", "-X", "pycache_prefix=" + str(cache), "-c", tcr.BOOTSTRAP,
                                   str(path), sha, "verify", "[]\n", str(cache)], env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=60)
        good, sha = lock_file("good.json")
        for label, path, used, env_uuid, message in (
                ("sha", good, "f" * 64, UUID, "external lock SHA differs"),
                ("schema", lock_file("cpu.json", tl.LOCK_SCHEMA)[0], lock_file("cpu.json", tl.LOCK_SCHEMA)[1], UUID, "explicit CUDA triplet lock required"),
                ("device", good, sha, "GPU-ffffffff-0000-0000-0000-000000000000", "selected physical device differs"),
                ("no_device", good, sha, None, "selected physical device differs")):
            with self.subTest(label=label):
                result = run(path, used, env_uuid)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr.decode())
                self.assertNotIn("triplet_cuda_runtime.py", result.stderr.decode())
        # Everything authentic: the head imports the sealed worker, whose strict loader refuses this minimal lock.
        result = run(good, sha)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("schema", result.stderr.decode())
        self.assertIn("triplet_lock.py", result.stderr.decode())

    @unittest.skipUnless(Path(tcr.PYTHON).exists(), "existing .venv-cupy interpreter required for prepare/verify")
    def test_prepare_load_verify_roundtrip_keeps_dirty_work_out_and_refuses_cpu_loader(self):
        # Synthetic MAPLE bundle metadata (never imported) mirrors tests/test_triplet_lock.py.
        target = self.root / ("e" * 64)
        target.mkdir()
        rows = []
        for rel, _, _, blob in self.specs["maple"]["files"]:
            data = self.git("maple", "cat-file", "blob", blob)
            path = target / "source" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            rows.append({"path": "source/" + rel, "size": len(data), "sha256": tl.digest(data)})
        distribution = {"METADATA": b"Metadata-Version: 2.1\nName: maple\nVersion: 0.1.0\n", "WHEEL": b"Wheel-Version: 1.0\n", "top_level.txt": b"maple\n"}
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
        q = self.descriptor(interpreter={"path": tcr.PYTHON, "version": [3, 12, 0],
                                         "sha256": tl.file_reference(str(Path(tcr.PYTHON).resolve()), tl.DEFAULT_LIMITS["max_source_bytes"])["sha256"]},
                            maple_bundle={"target": str(target), "manifest": tl.file_reference(str(manifest), tl.DEFAULT_LIMITS["max_manifest_bytes"])})
        qpath = self.root / "qualification.json"
        qpath.write_bytes(tl.canonical(q))
        qsha = tl.digest(qpath.read_bytes())
        (self.repos["leaf"] / "benchmarks/spatial_mapping.py").write_text('HELPER = "dirty"\n')  # excluded, recorded
        destination = self.root / "code"
        receipt = tcr.prepare(qpath, expected_sha256=qsha, destination=str(destination))
        ref = receipt["lock"]
        lock = tcr.verify(ref["path"], expected_sha256=ref["sha256"])
        self.assertEqual(lock, tcr.load(ref["path"], expected_sha256=ref["sha256"]))
        self.assertEqual((lock["schema"], lock["profile"], lock["interpretation"]["backend"]), (tcr.LOCK_SCHEMA, tcr.PROFILE, "cuda"))
        self.assertFalse(lock["interpretation"]["combined_restart"] or lock["interpretation"]["device_qualified_by_lock"]
                         or lock["interpretation"]["transport_qualified_by_lock"])
        helper = Path(lock["objects"]["leaf"]["path"]) / "source/benchmarks/spatial_mapping.py"
        self.assertEqual(helper.read_bytes(), b'HELPER = "pinned"\n')
        self.assertTrue(any(r[0] == "benchmarks/spatial_mapping.py" and r[1] == "regular"
                            for r in lock["excluded_active_work"]["leaf"]["tracked_worktree"]))
        with self.assertRaises(tl.TripletLockError) as ctx:
            tl.load(ref["path"], expected_sha256=ref["sha256"])
        self.assertEqual(ctx.exception.code, "schema")
        with self.assertRaises(tl.TripletLockError) as ctx:
            tcr.prepare(qpath, expected_sha256=qsha, destination=str(destination))
        self.assertEqual(ctx.exception.code, "output_exists")
        run = self.root / "operational"
        run.mkdir()
        spec = tcr.bootstrap_spec(ref["path"], expected_sha256=ref["sha256"], run_dir=str(run))
        self.assertEqual((spec["argv"][0], spec["env"]["CUDA_VISIBLE_DEVICES"], spec["device"]["physical_uuid"]), (tcr.PYTHON, UUID, UUID))
        helper.chmod(0o644)
        helper.write_text('HELPER = "tampered"\n')
        helper.chmod(0o444)
        with self.assertRaises(tl.TripletLockError) as ctx:
            tcr.verify(ref["path"], expected_sha256=ref["sha256"])
        self.assertEqual(ctx.exception.code, "source_changed")


if __name__ == "__main__":
    unittest.main()
