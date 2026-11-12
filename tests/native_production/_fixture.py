"""Authored synthetic fixtures for the production-runtime tests: real V1 sealing (`runtime.seal_source`) of synthetic MASY/SYRUP repos, a real C4A LEAF capsule of a synthetic LEAF repo,
a self-consistent synthetic lock, recipe and receipt. ENGINEERING fixtures only: not the original sources, not a production or scientific proof."""

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
sys.path.insert(0, str(ROOT / "tests"))

from _support import commit_all, git, make_repo  # noqa: E402,F401

from masy import leaf_capsule_seal as capseal  # noqa: E402
from masy import native_source_gate as gate  # noqa: E402
from masy import production_runtime as prod  # noqa: E402
from masy import runtime  # noqa: E402
from masy.runtime import RuntimeLimits  # noqa: E402

LIMITS = RuntimeLimits(git_timeout_s=30.0)
SYRUP_PYPROJECT = '[project]\nname = "maple-syrup"\nversion = "0.1.0"\ndescription = "synthetic"\nrequires-python = ">=3.11"\ndependencies = []\n'
MASY_PYPROJECT = '[project]\nname = "masy"\nversion = "0.0.3.dev0"\ndescription = "synthetic"\nrequires-python = ">=3.11"\ndependencies = []\n'
LEAF_PYPROJECT = '[project]\nname = "leaf"\nversion = "0.1.0"\n'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rmtree(path) -> None:
    for dp, dn, fn in os.walk(path):
        os.chmod(dp, 0o755)
        for n in fn:
            p = os.path.join(dp, n)
            if not os.path.islink(p):
                os.chmod(p, 0o644)
    shutil.rmtree(path, ignore_errors=True)


class Fixture(unittest.TestCase):
    """Builds, in one temp dir: store (sealed masy+syrup), release root (a copy of the sealed masy scope), LEAF capsule namespace, lock, recipe, receipt and a `source` block."""

    def setUp(self):
        self.tmp = Path(os.path.realpath(tempfile.mkdtemp(prefix="prod_")))
        self.addCleanup(rmtree, self.tmp)
        self.store = str(self.tmp / "wrapper" / "runs" / "store")
        os.makedirs(os.path.join(self.store, "objects", "sources"))
        os.makedirs(os.path.join(self.store, "locks"))
        masy = make_repo(self.tmp, "MASYREPO", "masy", "masy", files={"pyproject.toml": MASY_PYPROJECT, "src/masy/__init__.py": '"""m"""\n__version__ = "0.0.3.dev0"\n', "src/masy/extra.py": "X = 1\n", **{f"src/masy/{n}.py": f"# stub {n}\n" for n in ("scenario_io", "native_water_run", "native_water_run_config", "native_source_gate", "production_runtime", "leaf_capsule", "native_water_worker")},
                                                                       "tools/masy_runtime.py": '"""launcher fixture"""\n'})
        syrup = make_repo(self.tmp, "SYRUPREPO", "maple_syrup", "maple-syrup", files={"pyproject.toml": SYRUP_PYPROJECT, "src/maple_syrup/__init__.py": '"""s"""\n__version__ = "0.1.0"\n',
                                                                                     "src/maple_syrup/mod.py": "Y = 2\n"})
        self.sealed = {}
        for kind, repo in (("masy", masy), ("syrup", syrup)):
            obs = runtime.observe_source(kind, str(repo), expected_revision=None, limits=LIMITS)
            files, captured = runtime.hash_scope(obs, LIMITS)
            self.sealed[kind] = runtime.seal_source(self.store, kind, obs, files, captured, run_id=f"s-{kind}", limits=LIMITS)
            self.sealed[kind]["head"] = obs["head"]
        # release root: a plain (writable) copy of the sealed MASY scope, like a detached release worktree
        self.release = str(self.tmp / "release")
        shutil.copytree(os.path.join(self.sealed["masy"]["path"], "source"), self.release)
        for dp, dn, fn in os.walk(self.release):
            os.chmod(dp, 0o755)
            for n in fn:
                os.chmod(os.path.join(dp, n), 0o644)
        # LEAF capsule from a synthetic LEAF repo
        leaf = make_repo(self.tmp, "LEAFREPO", "leaf", "leaf", files={"pyproject.toml": LEAF_PYPROJECT, "leaf/__init__.py": "A = 1\n", "leaf/mod.py": "B = 2\n"})
        shutil.rmtree(leaf / "src")
        git(leaf, "add", "-A")
        git(leaf, "commit", "-q", "-m", "layout")
        self.leaf_head = git(leaf, "rev-parse", "HEAD").strip()
        self.ns = str(self.tmp / "wrapper" / "runs" / "store-leaf-capsules")
        obs = capseal.observe_leaf_scope(str(leaf), expected_revision=self.leaf_head)
        self.capsule = capseal.seal_leaf_capsule(self.ns, obs, run_id="c1", wrapper_roots=[str(self.tmp / "wrapper")], protected={"wrapper src": str(self.tmp / "wrapper" / "src")})
        interp = gate.interpreter_identity(gate.GateLimits())
        self.interp = interp
        self.maple_bundle = "b" * 64
        self.materialized = os.path.join(self.store, "objects", "maple-materialized", self.maple_bundle)
        os.makedirs(self.materialized)
        Path(self.materialized, "bundle-manifest.json").write_bytes(b'{"synthetic": "bundle manifest fixture"}\n')
        self.bundle_manifest_sha = sha(Path(self.materialized, "bundle-manifest.json").read_bytes())
        self.archive_sha = "a" * 64
        self.lock_doc = self.make_lock()
        self.write_lock(self.lock_doc)
        self.recipe_doc = self.make_recipe()
        self.recipe_path = str(self.tmp / "recipe.json")
        self.write_recipe(self.recipe_doc)
        self.receipt_path = str(self.tmp / "receipt.json")
        self.write_receipt()
        self.source = self.make_source()

    # ---- builders (tests mutate these documents, then rewrite)
    def make_lock(self) -> dict:
        def src(kind):
            return {"head": self.sealed[kind]["head"], "root": self.release if kind == "masy" else "/nonexistent/" + kind}
        sources = {"maple": {"head": "c" * 40, "root": "/nonexistent/maple"}, "syrup": src("syrup"), "masy": {**src("masy"), "launcher": {"sha256": sha(Path(self.release, "tools", "masy_runtime.py").read_bytes())}}}
        objects = {"sources": {"maple": {"object_id": "d" * 64, "package_digest": "e" * 64}, **{k: {"object_id": self.sealed[k]["object_id"], "package_digest": self.sealed[k]["package_digest"]} for k in ("syrup", "masy")}},
                   "maple_bundle": {"bundle_id": self.maple_bundle, "manifest_sha256": self.bundle_manifest_sha, "archive_sha256": self.archive_sha},
                   "maple_materialized": {"path": self.materialized, "manifest_sha256": self.bundle_manifest_sha}}
        lock = {"schema": prod.LOCK_SCHEMA, "lock_id": "", "store": self.store, "sources": sources, "objects": objects,
                "runtime": {"python": runtime._python_stable(runtime.interpreter_identity(sys.executable, LIMITS))}}  # the GENUINE V1 record of the running interpreter
        return lock

    def write_lock(self, lock: dict) -> str:
        lock = dict(lock)
        lock["lock_id"] = ""
        lock["lock_id"] = sha(prod.canonical_json({k: v for k, v in lock.items() if k != "lock_id"}).encode("ascii"))
        path = os.path.join(self.store, "locks", lock["lock_id"] + ".json")
        if os.path.exists(path):
            os.chmod(path, 0o644)
        Path(path).write_bytes((prod.canonical_json(lock) + "\n").encode("ascii"))
        self.lock_id = lock["lock_id"]
        return lock["lock_id"]

    def make_recipe(self) -> dict:
        s = self.sealed
        return {"schema": prod.RECIPE_SCHEMA, "name": "fixture", "profiles": {"native9f_layered_factory_afca_cpu_v1": {"device": "cpu", "leaf_profile": prod.LEAF_PROFILES["cpu"], "leaf_variant": "8da"}},
                "sources": {"maple": {"commit": "c" * 40, "object_id": "d" * 64, "package_digest": "e" * 64, "bundle_id": self.maple_bundle, "bundle_manifest_sha256": self.bundle_manifest_sha,
                                      "archive_sha256": self.archive_sha},
                            "syrup": {"commit": s["syrup"]["head"], "object_id": s["syrup"]["object_id"], "package_digest": s["syrup"]["package_digest"]},
                            "masy": {"commit": s["masy"]["head"], "object_id": s["masy"]["object_id"], "package_digest": s["masy"]["package_digest"],
                                     "launcher_sha256": sha(Path(self.release, "tools", "masy_runtime.py").read_bytes())},
                            "leaf": {"commit": self.leaf_head, "capsule_id": self.capsule["object_id"], "package_digest": self.capsule["package_digest"], "tree_digest": self.capsule["tree_digest"]}}}

    def write_recipe(self, doc: dict) -> None:
        Path(self.recipe_path).write_text(json.dumps(doc, sort_keys=True))

    def write_receipt(self, **over) -> None:
        doc = {"schema": prod.RECEIPT_SCHEMA, "status": "root_reviewed", "independent": True, "recipe_sha256": sha(Path(self.recipe_path).read_bytes()), "scope": "fixture scope",
               "evidence": ["fixture evidence"], "limits": ["fixture limits"], **over}
        Path(self.receipt_path).write_text(json.dumps(doc, sort_keys=True))

    def make_source(self) -> dict:
        return {"mode": "production_committed", "profile": "native9f_layered_factory_afca_cpu_v1", "recipe": self.recipe_path, "recipe_sha256": sha(Path(self.recipe_path).read_bytes()),
                "receipt": self.receipt_path, "receipt_sha256": sha(Path(self.receipt_path).read_bytes()), "production_store": self.store, "production_lock": self.lock_id,
                "release_masy_root": self.release, "leaf_capsule_namespace": self.ns}

    def repin(self, recipe=True, receipt=True) -> None:
        """Rewrite pins after a deliberate edit (so the test exercises the INTENDED check, not the sha pin)."""
        if recipe:
            self.write_recipe(self.recipe_doc)
            self.write_receipt()
        self.source = {**self.source, "recipe_sha256": sha(Path(self.recipe_path).read_bytes()), "receipt_sha256": sha(Path(self.receipt_path).read_bytes()), "production_lock": self.lock_id}

    def resolve(self, source=None, admit=None, **kw):
        return prod.resolve(source or self.source, "f" * 64, capability="cpu", role="parent", admit=admit or (lambda n, w: None), check_running_code=False, **kw)

    def refused(self, code: str, **kw):
        with self.assertRaises(gate.GateRefusal) as c:
            self.resolve(**kw)
        self.assertEqual(c.exception.code, code, c.exception.detail)
        return c.exception
