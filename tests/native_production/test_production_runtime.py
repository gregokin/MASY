"""C4B tests (CPU, MAPLE .venv, -B). Synthetic engineering fixtures: the real V1 sealer, the real C4A capsule sealer and the real production resolver code; the genuine
V1 `prepare`/`verify` child on REAL C1 sources, the real native worker and any GPU run are NOT exercised here (deferred to Root against the actual C1 commits)."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from _fixture import ROOT, Fixture, git, rmtree, sha  # noqa: F401
from masy import native_source_gate as gate
from masy import production_runtime as prod


class Positive(Fixture):
    def test_resolve_builds_closed_identity_and_sealed_roots(self):
        r = self.resolve()
        i = r.identity
        self.assertEqual((i["schema"], i["mode"], i["production_qualified"]), ("masy.native_water_run_bootstrap.v2", "production_committed", False))
        self.assertEqual(i["lock"]["lock_id"], self.lock_id)
        self.assertEqual(i["sources"]["leaf"]["capsule_id"], self.capsule["object_id"])
        self.assertEqual(r.leaf_root, os.path.join(self.ns, "objects", self.capsule["object_id"], "source"))
        self.assertEqual(r.source["masy_root"], os.path.join(self.store, "objects", "sources", self.sealed["masy"]["object_id"], "source"))
        self.assertIn("src/masy/__init__.py", r.manifests["masy"])
        self.assertEqual(r.identity_sha256, self.resolve().identity_sha256)  # parent and worker recompute the same identity
        worker = prod.resolve(self.source, "f" * 64, capability="cpu", role="worker", admit=lambda n, w: None, check_running_code=False)
        self.assertEqual(worker.identity_sha256, r.identity_sha256)
        self.assertEqual(r.source["mode"], "production_committed")
        self.assertEqual(r.identity["interpreter"]["spelling"], sys.executable)  # the identity carries the full V1 interpreter record, incl. the venv fields

    def test_resolved_manifests_equal_sealed_source_exactly(self):
        # the worker's own `nwc.source_census` needs the native stack (not importable in this author environment): the equivalent strict read-only snapshot check is run instead
        r = self.resolve()
        for kind, root in (("masy", r.source["masy_root"]), ("syrup", r.source["syrup_root"])):
            self.assertEqual(gate.verify_snapshot(root, dict(r.manifests[kind]), gate.GateLimits())["file_count"], len(r.manifests[kind]), kind)
        self.assertEqual(gate.verify_snapshot(r.leaf_root, dict(self.capsule["files"]), gate.GateLimits())["file_count"], len(self.capsule["files"]))

    def test_running_code_join_is_enforced(self):
        with self.assertRaises(gate.GateRefusal) as c:  # the real running masy modules are NOT the synthetic sealed MASY: must refuse (no caller label accepted)
            prod.resolve(self.source, "f" * 64, capability="cpu", role="parent", admit=lambda n, w: None)
        self.assertIn(c.exception.code, {"parent_code_mismatch", "parent_code_unexpected_module"})


class DeliveryBinding(Fixture):
    def test_resolve_builds_the_delivery_record_bound_to_the_identity(self):
        r = self.resolve()
        d = dict(r.delivery)
        self.assertEqual(prod.validate_delivery(d), d)
        self.assertEqual(d["production_identity_sha256"], r.identity_sha256)
        self.assertEqual((d["bundle_id"], d["manifest_sha256"], d["archive_sha256"], d["target"], d["lock_id"]),
                         (self.maple_bundle, self.bundle_manifest_sha, self.archive_sha, self.materialized, self.lock_id))
        self.assertEqual(d["leaf_root"], r.leaf_root)
        self.assertEqual(d["syrup_root"], os.path.join(r.source["syrup_root"], "src", "maple_syrup"))
        self.assertEqual(r.identity["sources"]["maple"]["bundle_manifest_sha256"], self.bundle_manifest_sha)
        self.assertEqual(self.resolve().delivery, r.delivery)  # parent and worker derive the same record

    def test_actual_resolve_output_is_accepted_by_every_consumer_and_the_algorithm_is_noncircular(self):
        from collections import OrderedDict
        from types import MappingProxyType
        r = self.resolve()
        self.assertIs(type(r.delivery), MappingProxyType)  # the real producer value (Root's probe: this used to be refused)
        self.assertEqual(prod.validate_delivery(r.delivery), dict(r.delivery))
        core = prod.delivery_core(r.delivery)
        self.assertEqual(r.identity["delivery_core"], core)  # (b) the identity includes the CORE
        self.assertEqual(tuple(core), prod.DELIVERY_CORE_KEYS)
        flat = json.dumps(r.identity, sort_keys=True)
        for outer_only in ("production_identity_sha256", r.identity_sha256, r.delivery["leaf_root"], r.delivery["target"]):
            self.assertNotIn(outer_only, flat)  # the identity never contains the outer record / its own digest: no fixed point
        self.assertEqual(r.delivery["production_identity_sha256"], gate.bootstrap_identity_sha(r.identity))  # (c)(d) outer record carries the digest of the core-bearing identity
        for bad in (OrderedDict(dict(r.delivery)), [("a", 1)], "x", None):
            with self.assertRaises(gate.GateRefusal):
                prod.validate_delivery(bad)
        with self.assertRaises(gate.GateRefusal):
            prod.validate_delivery(MappingProxyType({**r.delivery, "extra": 1}))

    def test_materialized_manifest_bytes_must_match_recipe_and_lock(self):
        Path(self.materialized, "bundle-manifest.json").write_bytes(b'{"tampered": true}\n')
        self.refused("delivery_manifest_mismatch")

    def test_lock_and_recipe_delivery_pins_must_agree(self):
        lock = self.make_lock()
        lock["objects"]["maple_bundle"]["archive_sha256"] = "9" * 64
        self.write_lock(lock)
        self.repin(recipe=False)
        self.refused("lock_recipe_mismatch")
        lock = self.make_lock()
        lock["objects"]["maple_materialized"]["manifest_sha256"] = "9" * 64
        self.write_lock(lock)
        self.repin(recipe=False)
        self.refused("lock_recipe_mismatch")

    def test_recipe_requires_the_new_delivery_keys(self):
        for key in ("bundle_manifest_sha256", "archive_sha256"):
            self.recipe_doc = self.make_recipe()
            del self.recipe_doc["sources"]["maple"][key]
            self.repin()
            self.refused("production_record_invalid")

    def test_missing_manifest_file_refused(self):
        os.remove(os.path.join(self.materialized, "bundle-manifest.json"))
        self.refused("delivery_manifest_unreadable")


class RecordRefusals(Fixture):
    def test_recipe_sha_pin(self):
        self.source = {**self.source, "recipe_sha256": "0" * 64}
        self.refused("recipe_sha_mismatch")

    def test_receipt_sha_pin(self):
        self.source = {**self.source, "receipt_sha256": "0" * 64}
        self.refused("receipt_sha_mismatch")

    def test_draft_and_nonindependent_receipt(self):
        for over in ({"status": "unreviewed_draft"}, {"independent": False}):
            self.write_receipt(**over)
            self.repin(recipe=False)
            self.refused("recipe_unreviewed")

    def test_receipt_names_other_recipe(self):
        self.write_receipt(recipe_sha256="1" * 64)
        self.repin(recipe=False)
        self.refused("receipt_recipe_mismatch")

    def test_unknown_profile_and_device(self):
        self.source = {**self.source, "profile": "native9f_layered_factory_afca_cuda_v1"}
        self.refused("production_recipe_unavailable")
        self.source = {**self.source, "profile": "native9f_layered_factory_afca_cpu_v1"}
        self.recipe_doc["profiles"] = {"native9f_layered_factory_afca_cpu_v1": {"device": "cuda", "leaf_profile": prod.LEAF_PROFILES["cuda"], "leaf_variant": "8da"}}
        self.repin()
        self.refused("production_device_mismatch")

    def test_recipe_closed_keys_and_literals(self):
        for mutate in (lambda d: d.update(extra=1), lambda d: d["profiles"]["native9f_layered_factory_afca_cpu_v1"].update(leaf_profile="leaf23_d4fe_water109_native9f_cpu_v1"),
                       lambda d: d["sources"]["leaf"].update(commit="main"), lambda d: d["sources"].pop("maple")):
            self.recipe_doc = self.make_recipe()
            mutate(self.recipe_doc)
            self.repin()
            self.refused("production_record_invalid")

    def test_oversized_record_refused_before_body(self):
        seen = []
        prod_limits = prod.ProductionLimits(max_recipe_bytes=10)
        with self.assertRaises(gate.GateRefusal) as c:
            prod.resolve(self.source, "f" * 64, capability="cpu", role="parent", plim=prod_limits, admit=lambda n, w: seen.append(w), check_running_code=False)
        self.assertEqual(c.exception.code, "recipe_unreadable")
        self.assertEqual(seen, [])  # the cap is checked from metadata before any admission/body


class LockRefusals(Fixture):
    def rewrite(self, mutate, expect):
        lock = self.make_lock()
        mutate(lock)
        self.write_lock(lock)
        self.repin(recipe=False)
        self.refused(expect)

    def test_lock_head_object_package_bundle(self):
        self.rewrite(lambda l: l["sources"]["syrup"].update(head="a" * 40), "lock_recipe_mismatch")
        self.rewrite(lambda l: l["objects"]["sources"]["masy"].update(object_id="a" * 64), "lock_recipe_mismatch")
        self.rewrite(lambda l: l["objects"]["sources"]["syrup"].update(package_digest="a" * 64), "lock_recipe_mismatch")
        self.rewrite(lambda l: l["objects"]["maple_bundle"].update(bundle_id="a" * 64), "lock_recipe_mismatch")
        self.rewrite(lambda l: l["objects"]["maple_materialized"].update(path="/tmp/x"), "lock_invalid")

    def test_lock_bytes_and_id_tamper(self):
        path = os.path.join(self.store, "locks", self.lock_id + ".json")
        data = Path(path).read_bytes()
        Path(path).write_bytes(data.replace(b'"store"', b'"stor2"'))
        self.refused("lock_invalid")
        Path(path).write_bytes(data[:-1])  # no trailing newline: not canonical
        self.refused("lock_invalid")
        Path(path).write_bytes(data)
        self.resolve()

    def test_bad_lock_ref_and_missing(self):
        self.source = {**self.source, "production_lock": "../x"}
        self.refused("lock_ref_invalid")
        self.source = {**self.source, "production_lock": "9" * 64}
        self.refused("lock_unreadable")

    def test_interpreter_mismatch(self):
        for key, bad in (("sha256", "0" * 64), ("spelling", "/usr/bin/python3"), ("is_symlink", not self.lock_doc["runtime"]["python"]["is_symlink"]), ("venv", None),
                         ("size", 1), ("realpath", "/usr/bin/python3")):
            self.rewrite(lambda l, key=key, bad=bad: l["runtime"]["python"].update({key: bad}), "interpreter_mismatch")
        self.rewrite(lambda l: l["runtime"]["python"].update(venv={**self.lock_doc["runtime"]["python"]["venv"], "pyvenv_cfg_sha256": "0" * 64}) if self.lock_doc["runtime"]["python"]["venv"] else None,
                     "interpreter_mismatch") if self.lock_doc["runtime"]["python"]["venv"] else None

    @unittest.skipUnless(os.path.exists("/home/okin/MAPLE/.venv-cupy/bin/python") and os.path.exists(sys.executable), "needs the two genuine existing virtual environments")
    def test_other_real_venv_over_the_same_base_binary_is_refused_with_valid_control(self):
        from masy import runtime
        from _fixture import LIMITS
        other = runtime._python_stable(runtime.interpreter_identity("/home/okin/MAPLE/.venv-cupy/bin/python", LIMITS))
        mine = runtime._python_stable(runtime.interpreter_identity(sys.executable, LIMITS))
        if other["spelling"] == mine["spelling"]:
            self.skipTest("the running interpreter is the cupy environment")
        self.assertEqual((other["realpath"], other["sha256"]), (mine["realpath"], mine["sha256"]))  # the Root probe condition: identical base binary, different environment
        self.rewrite(lambda l: l["runtime"].update(python=other), "interpreter_mismatch")
        self.rewrite(lambda l: l["runtime"].update(python=mine), "release_root_mismatch") if False else None
        self.write_lock(self.make_lock())  # valid control: the running interpreter's own record resolves
        self.repin(recipe=False)
        self.assertEqual(self.resolve().identity["interpreter"], mine)

    def test_release_root_mismatch(self):
        self.rewrite(lambda l: l["sources"]["masy"].update(root="/somewhere/else"), "release_root_mismatch")


class ObjectRefusals(Fixture):
    def unlock(self, p):
        os.chmod(p, 0o755 if os.path.isdir(p) else 0o644)

    def test_changed_sealed_byte_and_writable(self):
        f = os.path.join(self.sealed["syrup"]["path"], "source", "src", "maple_syrup", "mod.py")
        os.chmod(f, 0o644)
        self.refused("writable_file")
        Path(f).write_text("Y = 3\n")
        os.chmod(f, 0o444)
        self.refused("snapshot_differs")

    def test_extra_and_missing_entries(self):
        obj = self.sealed["masy"]["path"]
        os.chmod(obj, 0o755)
        Path(obj, "stray").write_text("x")
        self.refused("object_tamper")
        os.remove(os.path.join(obj, "stray"))
        os.chmod(obj, 0o555)
        d = os.path.join(obj, "source", "src", "masy")
        os.chmod(d, 0o755)
        os.mkdir(os.path.join(d, "emptydir"))
        os.chmod(os.path.join(d, "emptydir"), 0o555)
        os.chmod(d, 0o555)
        self.refused("capsule_entries")

    def test_object_manifest_forgery(self):
        obj = self.sealed["syrup"]["path"]
        m = os.path.join(obj, "manifest.json")
        os.chmod(obj, 0o755)
        os.chmod(m, 0o644)
        doc = json.loads(Path(m).read_text())
        doc["identity"]["project"]["version"] = "9"
        Path(m).write_text(prod.canonical_json(doc) + "\n")
        self.refused("object_tamper")

    def test_symlinked_object(self):
        obj = self.sealed["syrup"]["path"]
        moved = obj + ".moved"
        os.rename(obj, moved)
        os.symlink(moved, obj)
        self.refused("object_tamper")

    def test_wrong_capsule_and_commit(self):
        self.recipe_doc["sources"]["leaf"]["commit"] = "a" * 40
        self.repin()
        self.refused("capsule_recipe_mismatch")
        self.recipe_doc = self.make_recipe()
        self.recipe_doc["sources"]["leaf"]["capsule_id"] = "a" * 64
        self.repin()
        self.refused("capsule_missing")
        self.recipe_doc = self.make_recipe()
        self.recipe_doc["sources"]["leaf"]["package_digest"] = "a" * 64
        self.repin()
        self.refused("capsule_recipe_mismatch")

    def test_capsule_tamper(self):
        f = os.path.join(self.capsule["location"], "source", "leaf", "mod.py")
        os.chmod(f, 0o644)
        self.refused("writable_file")

    def test_release_root_drift(self):
        Path(self.release, "src", "masy", "extra.py").write_text("X = 2\n")
        self.refused("release_root_differs")
        Path(self.release, "src", "masy", "extra.py").write_text("X = 1\n")
        Path(self.release, "tools", "masy_runtime.py").write_text("changed\n")
        self.refused("release_root_differs")
        Path(self.release, "tools", "masy_runtime.py").write_text('"""launcher fixture"""\n')
        Path(self.release, "src", "masy", "new.py").write_text("")
        self.refused("release_root_differs")


class FrozenLayout(Fixture):
    """The V1 sealed-object layout/modes claimed by the verifier are ENFORCED (worker path included: resolve without the V1 child)."""

    def refuses_object(self, kind="syrup"):
        with self.assertRaises(gate.GateRefusal) as c:
            prod.resolve(self.source, "f" * 64, capability="cpu", role="worker", admit=lambda n, w: None, check_running_code=False)
        self.assertIn(c.exception.code, {"object_tamper", "writable_file", "writable_directory", "snapshot_differs", "capsule_entries", "symlink"}, c.exception.detail)

    def test_valid_control(self):
        prod.resolve(self.source, "f" * 64, capability="cpu", role="worker", admit=lambda n, w: None, check_running_code=False)

    def test_writable_object_root_manifest_editable_and_source_members(self):
        for kind in ("syrup", "masy"):
            obj = self.sealed[kind]["path"]
            for target in (obj, os.path.join(obj, "manifest.json"), os.path.join(obj, "editable"), os.path.join(obj, "source")):
                mode = os.stat(target).st_mode & 0o7777
                os.chmod(target, mode | 0o200)
                self.refuses_object()
                os.chmod(target, mode)
            ed = os.path.join(obj, "editable")
            first = next(iter(sorted(os.listdir(ed))))
            f = os.path.join(ed, first)
            mode = os.stat(f).st_mode & 0o7777
            os.chmod(f, mode | 0o200)
            self.refuses_object()
            os.chmod(f, mode)
        self.test_valid_control()

    def test_editable_overlay_tamper_extra_symlink_and_missing(self):
        obj = self.sealed["masy"]["path"]
        ed = os.path.join(obj, "editable")
        dist = os.path.join(ed, sorted(os.listdir(ed))[0])
        meta = os.path.join(dist, sorted(os.listdir(dist))[0])
        os.chmod(dist, 0o755)
        os.chmod(meta, 0o644)
        Path(meta).write_bytes(Path(meta).read_bytes() + b"x")
        os.chmod(meta, 0o444)
        os.chmod(dist, 0o555)
        self.refuses_object()
        os.chmod(dist, 0o755)
        os.symlink(meta, os.path.join(dist, "link"))
        os.chmod(dist, 0o555)
        self.refuses_object()

    def test_extra_top_level_member_and_symlinked_manifest(self):
        obj = self.sealed["syrup"]["path"]
        os.chmod(obj, 0o755)
        Path(obj, "stray").write_text("x")
        os.chmod(obj, 0o555)
        self.refuses_object()
        os.chmod(obj, 0o755)
        os.remove(os.path.join(obj, "stray"))
        m = os.path.join(obj, "manifest.json")
        os.rename(m, m + ".real")
        os.symlink(m + ".real", m)
        os.chmod(obj, 0o555)
        self.refuses_object()

    def test_object_without_editable_when_manifest_lists_it(self):
        obj = self.sealed["masy"]["path"]
        os.chmod(obj, 0o755)
        shutil.rmtree(os.path.join(obj, "editable"), onerror=lambda f, p, e: (os.chmod(os.path.dirname(p), 0o755), os.chmod(p, 0o755), f(p)))
        os.chmod(obj, 0o555)
        self.refuses_object()


class AdmissionBeforeReads(Fixture):
    def test_zero_admission_before_every_stage(self):
        stages = []
        self.resolve(admit=lambda n, w: stages.append(w))
        self.assertGreater(len(stages), 10)
        for k in range(len(stages)):
            seen = []

            def admit(n, w, k=k):
                seen.append(w)
                if len(seen) > k:
                    raise gate.GateRefusal("cold_memory", w)
            with self.assertRaises(gate.GateRefusal) as c:
                self.resolve(admit=admit)
            self.assertEqual(c.exception.code, "cold_memory", (k, seen[-1]))

    def test_refused_first_admission_reads_no_file_body(self):
        reads = []
        real = os.read
        with mock.patch.object(os, "read", side_effect=lambda fd, n: (reads.append(n), real(fd, n))[1]):
            with self.assertRaises(gate.GateRefusal) as c:
                self.resolve(admit=lambda n, w: (_ for _ in ()).throw(gate.GateRefusal("cold_memory", w)))
        self.assertEqual(c.exception.code, "cold_memory")
        self.assertEqual(reads, [])  # nothing was read, hashed or parsed before the first admission was granted


class VerifyChild(Fixture):
    def child(self, body: str, **limits):
        launcher = self.tmp / "launcher.py"
        launcher.write_text(body)
        plim = prod.ProductionLimits(**limits) if limits else prod.ProductionLimits()
        return prod._run_verify_child(sys.executable, str(launcher), self.store, "a" * 64, plim, lambda n, w: None)

    OK = ('import json;print(json.dumps({"schema":"masy.runtime_output.v1","verdict":"runtime_verified","exit_code":0,"lock_id":"%s","selection":"explicit_lock",'
          '"asserts_current":False,"transport_ready":False}))' % ("a" * 64))

    def test_accepts_only_the_closed_success_output(self):
        self.assertEqual(self.child(self.OK)["verdict"], "runtime_verified")
        for body, code in ((self.OK.replace("a" * 64, "b" * 64), "verify_child_failed"), (self.OK.replace("runtime_verified", "refused"), "verify_child_failed"),
                           (self.OK.replace("False,\"transport", "True,\"transport"), "verify_child_failed"), ("print('not json')", "verify_child_failed"),
                           ("import sys;sys.exit(2)", "verify_child_failed"), ("print('x'*100000)", "verify_child_output_too_large"),
                           ("import time;time.sleep(30)", "verify_child_timeout")):
            limits = {"verify_max_output_bytes": 4096} if "x'*" in body else ({"verify_timeout_s": 1.0} if "sleep" in body else {})
            with self.assertRaises(gate.GateRefusal) as c:
                self.child(body, **limits)
            self.assertEqual(c.exception.code, code, body)

    def test_child_environment_cpu_hides_and_cuda_forwards_only_the_bound_value(self):
        probe = 'import json,os;print(json.dumps(dict(os.environ)))'
        for cap_, vis, want in (("cpu", "0", ""), ("cpu", None, ""), ("cuda", "GPU-abc", "GPU-abc")):
            env = prod._child_env(cap_, vis)
            out = subprocess.run([sys.executable, "-I", "-B", "-c", probe], env=env, capture_output=True, text=True, check=True).stdout
            got = json.loads(out)
            self.assertEqual(got.get("CUDA_VISIBLE_DEVICES"), want, (cap_, vis))
            self.assertLessEqual(set(env), {"PATH", "LC_ALL", "CUDA_VISIBLE_DEVICES"})  # no other caller variable leaks into the child
        for bad in (None, "", "  "):
            with self.assertRaises(gate.GateRefusal) as c:
                prod._child_env("cuda", bad)
            self.assertEqual(c.exception.code, "cuda_visibility_unbound")

    def test_child_admission_precedes_spawn(self):
        with mock.patch.object(prod.subprocess, "Popen", side_effect=AssertionError("spawned")):
            with self.assertRaises(gate.GateRefusal):
                prod._run_verify_child(sys.executable, "/x", self.store, "a" * 64, prod.ProductionLimits(), lambda n, w: (_ for _ in ()).throw(gate.GateRefusal("cold_memory", w)))


class ConfigAndClosure(Fixture):
    def render(self, **over):
        out = self.tmp / "cfg"
        args = [sys.executable, "-I", "-B", str(ROOT / "tools" / "masy_production.py"), "render-config", str(out), "--recipe", self.recipe_path, "--receipt", self.receipt_path, "--store", self.store,
                "--lock", self.lock_id, "--release-root", self.release, "--capsule-namespace", self.ns]
        done = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        return out / "run.yaml"

    def test_rendered_v2_production_config_loads_closed_and_pinned(self):
        from masy.native_water_run_config import load_run_config, resolved_config
        path = self.render()
        cfg = load_run_config(str(path), references=False)
        self.assertEqual((cfg.schema, cfg.source["mode"]), ("masy.native_water_run.v2", "production_committed"))
        self.assertEqual(cfg.leaf["leaf_root"], "")
        full = load_run_config(str(path))
        r = self.resolve()
        rc = resolved_config(full, r.source, r.leaf_root)
        self.assertEqual(rc.leaf["leaf_root"], r.leaf_root)
        self.assertEqual(rc.source["masy_root"], r.source["masy_root"])
        self.assertEqual(rc.config_sha256, full.config_sha256)
        rc.revalidate()
        self.assertEqual(full.leaf["leaf_root"], "")  # the sealed original is untouched

    def test_closed_key_refusals(self):
        from masy.native_water_run_config import RunConfigError, load_run_config
        text = self.render().read_text()
        base = self.tmp / "cfg"
        for name, new in (("leaf_root", text.replace("  catalog: catalog.json", "  leaf_root: /x\n  catalog: catalog.json")), ("extra_source", text.replace("  recipe: ", "  extra: 1\n  recipe: ")),
                          ("rel_recipe", text.replace("recipe: " + self.recipe_path, "recipe: recipe.json")), ("bad_sha", text.replace(self.source["recipe_sha256"], "xyz")),
                          ("dev_profile", text.replace("native9f_layered_factory_afca_cpu_v1", "native9f_layered_factory_95ef_cpu_v1")), ("variant", text.replace("leaf_variant: 8da", "leaf_variant: d4fe"))):
            (base / "bad.yaml").write_text(new)
            with self.assertRaises(RunConfigError, msg=name):
                load_run_config(str(base / "bad.yaml"), references=False)

    def test_v1_production_and_v2_dev_contracts_preserved(self):
        from masy import native_water_run as run
        base = self.tmp / "cfg"
        text = self.render().read_text()
        v1 = text.replace("schema: masy.native_water_run.v2", "schema: masy.native_water_run.v1")
        a, b = v1.index("source:"), v1.index("execution:")
        v1 = v1[:a] + f"source:\n  mode: production_committed\n  profile: native9f_layered_factory_afca_cpu_v1\n  production_store: {self.store}\n  production_lock: {self.lock_id}\n" + v1[b:v1.index("leaf:")]
        (base / "v1.yaml").write_text(v1)
        buf = io.StringIO()
        with redirect_stdout(buf), mock.patch.object(prod, "resolve", side_effect=AssertionError("v2 path must not run for v1")):
            rc = run.main(["run", str(base / "v1.yaml")])
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads(buf.getvalue().split("MASY_RUN_RESULT:")[1])["code"], "production_recipe_unavailable")

    def test_zero_cap_refuses_before_any_production_read(self):
        from masy import native_water_run as run
        base = self.tmp / "cfg"
        text = self.render().read_text().replace("host_available_cap_bytes: null", "host_available_cap_bytes: 0")
        (base / "zero.yaml").write_text(text)
        buf = io.StringIO()
        with redirect_stdout(buf), mock.patch.object(prod, "resolve", side_effect=AssertionError("resolve called")), mock.patch.object(prod.cap, "_read_file", side_effect=AssertionError("read")):
            rc = run.main(["run", str(base / "zero.yaml")])
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads(buf.getvalue().split("MASY_RUN_RESULT:")[1])["code"], "cold_memory")

    def test_development_closures_unchanged_and_production_import_is_lazy(self):
        self.assertEqual(gate.PARENT_CLOSURE, ("__init__", "scenario_io", "native_water_run", "native_water_run_config", "native_source_gate"))
        self.assertEqual(gate.WORKER_ALLOWED, gate.PARENT_CLOSURE + ("native_water_worker",))
        self.assertEqual(gate.PRODUCTION_PARENT_CLOSURE, gate.PARENT_CLOSURE + ("production_runtime", "leaf_capsule"))
        code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); import masy.native_water_run, masy.native_water_worker; "
                "print(sorted(m for m in sys.modules if m.startswith('masy.') and m.split('.')[1] in ('production_runtime','leaf_capsule','runtime','sourceobs')))")
        out = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, check=True).stdout
        self.assertEqual(out.strip(), "[]")

    def test_production_modules_import_closure_has_no_native_or_runtime(self):
        code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); import masy.production_runtime; "
                "bad=[m for m in sys.modules if m.split('.')[0] in ('numpy','maple','maple_syrup','leaf','numba','cupy') or m in ('masy.runtime','masy.sourceobs','masy.runtime_worker')]; "
                "print(bad, sorted(m for m in sys.modules if m.startswith('masy')))")
        out = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, check=True).stdout
        self.assertTrue(out.startswith("[] "), out)
        self.assertIn("'masy.production_runtime'", out)

    def test_leaf_profile_literals_pinned_to_the_source_profile(self):
        from masy import leaf_source_profile as lsp
        self.assertEqual(dict(prod.LEAF_PROFILES), {"cpu": lsp.CPU_PROFILE_8DA_ID, "cuda": lsp.CUDA_PROFILE_8DA_ID})
        from masy.native_water_run_config import PROFILE_CAPABILITY
        self.assertEqual(PROFILE_CAPABILITY["native9f_layered_factory_afca_cpu_v1"], "cpu")

    def test_worker_refuses_production_without_parent_identity(self):
        from masy import native_water_worker as worker
        path = self.render()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = worker.main([str(path), "--expected-config-sha256", sha(path.read_bytes())])
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads(buf.getvalue().split("MASY_RUN_RESULT:")[1])["code"], "bootstrap_identity_required")


class OperatorTool(Fixture):
    def test_draft_recipe_derives_from_real_lock_and_capsule_and_receipt_draft_is_refused(self):
        tool = [sys.executable, "-I", "-B", str(ROOT / "tools" / "masy_production.py")]
        out = self.tmp / "drafted.json"
        done = subprocess.run([*tool, "draft-recipe", "--store", self.store, "--lock", self.lock_id, "--capsule-namespace", self.ns, "--capsule-id", self.capsule["object_id"], "--name", "n",
                               "--profile", "native9f_layered_factory_afca_cpu_v1", "--device", "cpu", "--reserve-bytes", "0", "--out", str(out)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        drafted = json.loads(out.read_text())
        self.assertEqual(drafted["sources"], self.recipe_doc["sources"])  # derived, never typed in
        rec = self.tmp / "draft_receipt.json"
        self.assertEqual(subprocess.run([*tool, "draft-receipt", "--recipe", str(out), "--out", str(rec)], capture_output=True, text=True).returncode, 0)
        with self.assertRaises(gate.GateRefusal) as c:
            prod.parse_receipt(json.loads(rec.read_text()), sha(out.read_bytes()))
        self.assertEqual(c.exception.code, "recipe_unreviewed")
        neg = subprocess.run([*tool, "draft-recipe", "--store", self.store, "--lock", self.lock_id, "--capsule-namespace", self.ns, "--capsule-id", self.capsule["object_id"], "--name", "n",
                              "--profile", "p", "--device", "cpu", "--reserve-bytes", "-1", "--out", str(self.tmp / "x.json")], capture_output=True, text=True)
        self.assertNotEqual(neg.returncode, 0)


if __name__ == "__main__":
    unittest.main()
