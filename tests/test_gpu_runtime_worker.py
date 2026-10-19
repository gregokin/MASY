"""Synthetic interface tests only: never import CuPy or query a device."""
import base64
import contextlib
import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from masy import gpu_runtime_worker as w

GPU = "GPU-5c8bd3d4-2885-82c6-a0f0-ad2b7f87b871"
DEVICE = {"uuid": GPU, "pci_bus_id": "00000000:17:00.0", "visible_ordinal": 0}
PROFILE = "a" * 64


def gate(now=1000):
    resource = {"schema": "masy.gpu_resource_observation.v1", "observed_unix_s": now - 1,
        "gpus": [{**DEVICE, "memory_used_mib": 0, "memory_total_mib": 11264, "utilization_percent": 0}],
        "processes": [], "command_receipts": {"synthetic": True}}
    auth = {"schema": "masy.gpu_identity_authorization.v1", "authorized": True, "operation": "identity_probe",
        "profile_id": PROFILE, "uuid": GPU, "expires_unix_s": now + 10, "max_resource_age_s": 5,
        "max_memory_used_mib": 0, "max_utilization_percent": 0}
    return regate({"authorization": auth, "resource": resource, "resource_sha256": ""})


def regate(value):
    value["resource_sha256"] = hashlib.sha256(json.dumps(value["resource"], sort_keys=True,
        separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    return value


class WorkerTests(unittest.TestCase):
    def test_process_identity_and_exact_policy_with_synthetic_process_fields(self):
        from masy import gpu_runtime as g

        with tempfile.TemporaryDirectory() as tmp:
            store = str(Path(tmp) / "store")
            g._ensure_store(store)
            run = g._Run(store, "synthetic-process")
            package = Path(tmp) / "package"
            package.mkdir()
            env = w.worker_env(run.dir, GPU)
            request = {"python": {"spelling": w.PYTHON, "realpath": os.path.realpath(w.PYTHON), "venv_dir": w.VENV},
                "run": {"run_dir": str(run.dir), "pycache_prefix": run.pycache, "env": env},
                "sys_path_prefix": [str(package)], "device": DEVICE, "limits": {"max_files": 1}}
            with patch.dict(os.environ, env, clear=True), patch.object(sys, "executable", w.PYTHON), \
                    patch.object(sys, "prefix", w.VENV), patch.object(sys, "pycache_prefix", run.pycache), \
                    patch.object(sys, "path", [str(package)]):
                observed = w.check_process(request, [])
                self.assertEqual(observed["executable"], w.PYTHON)
                changed = copy.deepcopy(request)
                changed["run"]["env"]["HOME"] = "synthetic-illegal-key-only"
                with self.assertRaises(w.WorkerRefusal):
                    w.check_process(changed, [])
                with patch.dict(os.environ, {"UNREQUESTED_BACKEND_FLAG": "1"}), self.assertRaises(w.WorkerRefusal):
                    w.check_process(request, [])
                with patch.object(sys, "prefix", "/wrong-venv"), self.assertRaises(w.WorkerRefusal):
                    w.check_process(request, [])
                with patch.object(sys, "pycache_prefix", None), self.assertRaises(w.WorkerRefusal):
                    w.check_process(request, [])

    def test_exact_environment_has_all_owned_caches_and_no_home(self):
        env = w.worker_env("/tmp/synthetic-run", GPU)
        self.assertEqual(set(env), set(w._STABLE_ENV) | set(w._CACHE_PATHS) | {"CUDA_VISIBLE_DEVICES"})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], GPU)
        for key in w._CACHE_PATHS:
            self.assertTrue(env[key].startswith("/tmp/synthetic-run/"))
        self.assertFalse({"HOME", "PYTHONPATH", "LD_LIBRARY_PATH", "CUPY_CACHE_SAVE_CUDA_SOURCE"} & set(env))

    def test_closed_device_rejects_ordinal_aliases_and_malformed_identity(self):
        self.assertEqual(w.check_device(DEVICE), DEVICE)
        for field, value in (("visible_ordinal", True), ("visible_ordinal", 1), ("uuid", "0"),
                ("uuid", "MIG-GPU-example"), ("pci_bus_id", "garbage")):
            changed = {**DEVICE, field: value}
            with self.subTest(field=field, value=value), self.assertRaises(w.WorkerRefusal):
                w.check_device(changed)
        with self.assertRaises(w.WorkerRefusal):
            w.check_device({**DEVICE, "extra": 1})

    def test_uuid_and_pci_both_must_match(self):
        props = {"uuid": uuid.UUID(GPU[4:]).bytes, "pciDomainID": 0, "pciBusID": 0x17, "pciDeviceID": 0}
        self.assertEqual(w.check_observed_device(DEVICE, props, "0000:17:00.0"), DEVICE)
        for bad in ({**props, "uuid": bytes(16)}, {**props, "pciBusID": 0x65},
                {**props, "uuid": b"short"}, {k: v for k, v in props.items() if k != "uuid"}):
            with self.subTest(props=bad), self.assertRaises(w.WorkerRefusal):
                w.check_observed_device(DEVICE, bad, "0000:17:00.0")
        with self.assertRaises(w.WorkerRefusal):
            w.check_observed_device(DEVICE, props, "0000:65:00.0")

    def test_gate_requires_exact_authorized_scope_before_any_import(self):
        w.check_probe_gate(gate(), PROFILE, DEVICE, now=1000)
        for field, value in (("authorized", False), ("operation", "transport"), ("profile_id", "b" * 64),
                ("uuid", "GPU-00000000-0000-0000-0000-000000000000"), ("expires_unix_s", 999),
                ("max_resource_age_s", True), ("max_resource_age_s", 0)):
            changed = gate()
            changed["authorization"][field] = value
            with self.subTest(field=field), self.assertRaises(w.WorkerRefusal):
                w.check_probe_gate(changed, PROFILE, DEVICE, now=1000)
        invalid = gate()
        invalid["authorization"]["authorized"] = False
        blocker = w.AcceleratorBlocker()
        sys.meta_path.insert(0, blocker)
        try:
            with self.assertRaises(w.WorkerRefusal):
                w.identity_probe({"probe_gate": invalid, "profile_id": PROFILE, "device": DEVICE})
            self.assertEqual(blocker.attempts, [])
        finally:
            sys.meta_path.remove(blocker)

    def test_busy_stale_ambiguous_or_tampered_resource_refuses(self):
        for mode in ("busy_memory", "busy_util", "process", "ambiguous", "stale", "tampered", "schema"):
            value = gate()
            if mode == "busy_memory":
                value["resource"]["gpus"][0]["memory_used_mib"] = 948
            elif mode == "busy_util":
                value["resource"]["gpus"][0]["utilization_percent"] = 3
            elif mode == "process":
                value["resource"]["processes"] = [{"uuid": GPU, "pid": 123, "memory_mib": "0"}]
            elif mode == "ambiguous":
                value["resource"]["gpus"].append(copy.deepcopy(value["resource"]["gpus"][0]))
            elif mode == "stale":
                value["resource"]["observed_unix_s"] = 990
            elif mode == "schema":
                value["resource"]["schema"] = "not-the-resource-schema"
            if mode != "tampered":
                regate(value)
            else:
                value["resource_sha256"] = "0" * 64
            with self.subTest(mode=mode), self.assertRaises(w.WorkerRefusal):
                w.check_probe_gate(value, PROFILE, DEVICE, now=1000)

    def test_gate_refuses_malformed_process_rows_before_any_accelerator_import(self):
        other = "GPU-00000000-0000-0000-0000-000000000001"
        valid = {"uuid": other, "pid": 123, "memory_mib": "[N/A]"}
        value = gate()
        value["resource"]["processes"] = [valid]
        w.check_probe_gate(regate(value), PROFILE, DEVICE, now=1000)
        malformed = (None, {}, {"gpu_uuid": GPU, "pid": 123},
            {**valid, "uuid": None}, {**valid, "uuid": "0"}, {**valid, "pid": 0},
            {**valid, "pid": -1}, {**valid, "pid": True}, {**valid, "pid": "123"},
            {**valid, "memory_mib": None}, {**valid, "memory_mib": " "}, {**valid, "extra": 1})
        blocker = w.AcceleratorBlocker()
        sys.meta_path.insert(0, blocker)
        try:
            for process in malformed:
                value = gate()
                value["resource"]["processes"] = [process]
                with self.subTest(process=process), self.assertRaises(w.WorkerRefusal):
                    w.check_probe_gate(regate(value), PROFILE, DEVICE, now=1000)
            self.assertEqual(blocker.attempts, [])
        finally:
            sys.meta_path.remove(blocker)

    def test_import_blocker_records_attempts_even_when_caller_catches(self):
        blocker = w.AcceleratorBlocker()
        sys.meta_path.insert(0, blocker)
        try:
            for name in ("cupy", "cupy_backends", "cuda", "pynvml"):
                with self.assertRaises(w.WorkerRefusal):
                    importlib.import_module(name)
            self.assertEqual(blocker.attempts, ["cupy", "cupy_backends", "cuda", "pynvml"])
        finally:
            sys.meta_path.remove(blocker)
        self.assertNotIn("cupy", sys.modules)

    def test_maps_scanner_keeps_external_libraries_and_special_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / "external driver.so.1"
            library.write_bytes(b"synthetic binary")
            maps = Path(tmp) / "maps"
            escaped = str(library).replace(" ", r"\040")
            maps.write_text(f"0-1 r-xp 0 0:0 1 {escaped}\n0-1 rw-p 0 0:0 1 /dev/null\n")
            records = w.mapped_files(str(maps))
            self.assertEqual({r["path"]: r["kind"] for r in records},
                {str(library): "binary", "/dev/null": "special_mapping"})
            maps.write_text(f"0-1 r-xp 0 0:0 1 {escaped} (deleted)\n")
            with self.assertRaises(w.WorkerRefusal):
                w.mapped_files(str(maps))
            with self.assertRaises(w.WorkerRefusal):
                w.mapped_files(str(Path(tmp) / "missing"))

    def test_maps_and_compiled_bounds_refuse_instead_of_truncate(self):
        with tempfile.TemporaryDirectory() as tmp:
            maps = Path(tmp) / "maps"
            maps.write_bytes(b"x" * (w.MAX_MAPS + 1))
            with self.assertRaises(w.WorkerRefusal):
                w.mapped_files(str(maps))
        with patch.object(w, "mapped_files", return_value=[{"path": "/a", "kind": "binary"},
                {"path": "/b", "kind": "binary"}]), self.assertRaises(w.WorkerRefusal):
            w.compiled_inventory({"max_compiled_files": 1, "max_compiled_bytes": 1 << 30}, {})

    def test_compiled_record_mismatch_is_real_file_hash_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "synthetic.so"
            binary.write_bytes(b"source-only fake binary")
            owner = {"distribution": "synthetic", "mode": "sha256", "value": "bad", "size": binary.stat().st_size}
            with patch.object(w, "mapped_files", return_value=[{"path": str(binary), "kind": "binary"}]):
                with self.assertRaises(w.WorkerRefusal) as caught:
                    w.compiled_inventory({"max_compiled_files": 1000, "max_compiled_bytes": 1 << 30}, {str(binary): [owner]})
                self.assertEqual(caught.exception.code, "compiled_record_mismatch")

    def test_metadata_inventory_binds_actual_record_bytes_without_gpu_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp) / "synthetic-1.dist-info"
            dist.mkdir()
            (Path(tmp) / "synthetic.so").write_bytes(b"declared")
            (dist / "METADATA").write_text("Name: synthetic\nVersion: 1\n")
            digest = base64.urlsafe_b64encode(hashlib.sha256(b"declared").digest()).rstrip(b"=").decode()
            (dist / "RECORD").write_text(f"synthetic.so,sha256={digest},8\nsynthetic-1.dist-info/METADATA,,\nsynthetic-1.dist-info/RECORD,,\n")
            records, index = w.environment_metadata(path=[tmp])
            self.assertEqual(records[0]["name"], "synthetic")
            self.assertEqual(len(records[0]["metadata_files"]), 2)
            self.assertEqual(index[str(Path(tmp) / "synthetic.so")][0]["value"], digest)
            self.assertNotIn("cupy", sys.modules)

    def test_no_readiness_from_metadata(self):
        self.assertTrue(all(value is False for value in w.INTERPRETATION.values()))

    def test_synthetic_probe_adapters_refuse_either_incomplete_native_lookup(self):
        properties = {"uuid": uuid.UUID(GPU[4:]).bytes, "pciDomainID": 0, "pciBusID": 0x17, "pciDeviceID": 0}
        runtime = types.SimpleNamespace(getDeviceCount=lambda: 1, getDeviceProperties=lambda _n: properties,
            deviceGetPCIBusId=lambda _n: "0000:17:00.0", getDevice=lambda: 0,
            driverGetVersion=lambda: 12090, runtimeGetVersion=lambda: 12090)
        cp = types.ModuleType("cupy")  # Pure fake module: no real CuPy/native call.
        cp.__version__ = "SYNTHETIC"
        cp.cuda = types.SimpleNamespace(runtime=runtime, get_local_runtime_version=lambda: 12090,
            nvrtc=types.SimpleNamespace(getVersion=lambda: (12, 9)))
        backend = types.SimpleNamespace(device_scope=contextlib.nullcontext,
            fingerprint={"source_only": "synthetic"}, deterministic_scatter=True)
        selection = types.ModuleType("maple.core.backend.selection")
        selection.resolve_backend = lambda *_args, **_kwargs: backend
        declarations = {name: {"nvrtc": [12, 9], "provenance_incomplete": False}
            for name in ("routing", "hydrology")}
        original_import = w.importlib.import_module

        def fake_import(name, *args, **kwargs):
            for kind, declaration in declarations.items():
                if name == f"maple_syrup.{kind}_cuda":
                    return types.SimpleNamespace(kernel_provenance=lambda value=declaration: value.copy())
            return original_import(name, *args, **kwargs)

        with patch.dict(sys.modules, {"cupy": cp, "maple.core.backend.selection": selection}), \
                patch.object(w, "check_probe_gate"), patch.object(w.importlib, "import_module", side_effect=fake_import):
            request = {"probe_gate": {}, "profile_id": PROFILE, "device": DEVICE}
            accepted = w.identity_probe(request)
            self.assertEqual(set(accepted["native_kernel_declarations"]), {"routing", "hydrology"})
            for kind, declaration in declarations.items():
                for field, value in (("provenance_incomplete", True), ("nvrtc", None)):
                    old = declaration[field]
                    declaration[field] = value
                    with self.subTest(kind=kind, field=field), self.assertRaises(w.WorkerRefusal) as caught:
                        w.identity_probe(request)
                    self.assertEqual(caught.exception.code, "device_provenance_incomplete")
                    declaration[field] = old
        self.assertNotIn("cupy", sys.modules)

    def test_actual_native_declared_hashes_agree_without_cupy_import(self):
        root = Path(__file__).resolve().parents[1]
        lock_path = Path("/home/okin/MASY/runs/runtime-store/locks/43976b9a6c3aa5384d0cdfdcf7c61f90d6a2f0ced45a0890e92dad5fb6a5cd33.json")
        lock = json.loads(lock_path.read_text())
        old_masy = lock["objects"]["sources"]["masy"]["path"]
        prefix = [str(root / "src"), *[p for p in lock["runtime"]["sys_path_prefix"] if not p.startswith(old_masy + "/")]]
        code = """import sys,json
sys.path[:0]=json.loads(sys.argv[1])
from masy import gpu_runtime_worker as w
b=w.AcceleratorBlocker(); sys.meta_path.insert(0,b)
result=w.declared_kernels()
assert not b.attempts and 'cupy' not in sys.modules
sys.modules['cupy']=None
from maple_syrup import routing_cuda,hydrology_cuda
r=routing_cuda.kernel_provenance(); h=hydrology_cuda.kernel_provenance()
assert result['routing']['_KERNEL_SOURCE_sha256']==r['source_sha256']
assert result['routing']['_BLOCK_SOURCE_sha256']==r['block_source_sha256']
assert result['hydrology']['_SOURCE_sha256']==h['hydrology_source_sha256']
assert result['compile_options']==r['compile_options']
print(json.dumps({'pass':True,'source_only':True}))
"""
        with tempfile.TemporaryDirectory() as tmp:
            env = w.worker_env(tmp, GPU)
            for relative in w._CACHE_PATHS.values():
                (Path(tmp) / relative).mkdir(parents=True, exist_ok=True)
            env["CUDA_VISIBLE_DEVICES"] = ""  # CPU source-test process, no profile/admission claim
            result = subprocess.run([sys.executable, "-I", "-B", "-X", f"pycache_prefix={tmp}/absent", "-c", code,
                json.dumps(prefix)], env=env, capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {"pass": True, "source_only": True})


if __name__ == "__main__":
    unittest.main()
