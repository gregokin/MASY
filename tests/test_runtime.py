"""MASY-03 locked CPU runtime (`masy.runtime` parent, `masy.runtime_worker` child helpers).

Every fixture is a tiny authored file, module or synthetic Git repository inside a temporary directory
(`_support.TempDirCase`). Nothing here reads MAPLE/SYRUP sources or installed environments, imports upstream code,
installs anything, touches arrays/datasets or runs the launcher end to end; the full committed-pair prepare/verify
is qualified separately. Sealed fixture objects are read-only: `RuntimeCase.tearDown` restores write permission
on temporary directories only. Process-group, /proc, FIFO and link/renameat2 publication tests are Linux/CPU
scoped; child waits are a few seconds at most.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import stat
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from _support import TempDirCase, commit_all, git, make_repo, requires_git, write

from masy import runtime, runtime_worker
from masy.runtime import RuntimeLimits, RuntimeRefusal
from masy.runtime_worker import WorkerRefusal
from masy.sourceobs import canonical_json

LIMITS = RuntimeLimits(git_timeout_s=30.0)
LINUX = sys.platform.startswith("linux")
requires_linux = unittest.skipUnless(LINUX, "Linux only (process groups, /proc, FIFOs, link/renameat2 publication)")
requires_packaging = unittest.skipUnless(importlib.util.find_spec("packaging") is not None,
                                         "static metadata checks need the interpreter's `packaging` module")

# ----------------------------------------------------------------------------------------- authored fixtures

SYRUP_PYPROJECT = """[project]
name = "maple-syrup"
version = "0.1.0"
description = "synthetic MASY runtime fixture"
requires-python = ">=3.11"
dependencies = ["packaging>=23"]

[project.optional-dependencies]
test = ["pytest>=7; python_version >= '3.11'"]

[project.scripts]
maple-syrup-fixture = "maple_syrup.cli:main"
"""
SYRUP_INIT = '"""synthetic maple_syrup"""\n__version__ = "0.1.0"\n'
SYRUP_CLI = "def main():\n    return 0\n"
SYRUP_DIST_INFO = "maple_syrup-0.1.0.dist-info"
MASY_PYPROJECT = """[project]
name = "masy"
version = "0.0.3.dev0"
description = "synthetic MASY wrapper fixture"
requires-python = ">=3.11"
dependencies = []
"""
MASY_INIT = '"""synthetic masy"""\n__version__ = "0.0.3.dev0"\n'
LAUNCHER_FIXTURE = '"""synthetic launcher fixture; never executed"""\n'
OVERLAY_NAMES = ("INSTALLER", "MASY_OVERLAY.json", "METADATA", "RECORD", "direct_url.json", "top_level.txt")


def make_syrup(base: Path, name: str = "SYRUP") -> Path:
    """Clean committed synthetic SYRUP: static pyproject (deps/extra/script) and a literal aligned __version__."""
    return make_repo(base, name, "maple_syrup", "maple-syrup", files={
        "pyproject.toml": SYRUP_PYPROJECT, "src/maple_syrup/__init__.py": SYRUP_INIT,
        "src/maple_syrup/cli.py": SYRUP_CLI, ".gitignore": "*.so\n__pycache__/\n",
    })


def make_masy(base: Path) -> Path:
    return make_repo(base, "MASYFIX", "masy", "masy", files={
        "pyproject.toml": MASY_PYPROJECT, "src/masy/__init__.py": MASY_INIT,
        "tools/masy_runtime.py": LAUNCHER_FIXTURE,
    })


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def committed_scope(root: Path, *pathspecs: str) -> list[list]:
    """[[path, size, sha256, git blob]] of HEAD under `pathspecs`, from `git ls-tree` and independent file reads."""
    entries = []
    for line in git(root, "ls-tree", "-r", "HEAD", "--", *pathspecs).splitlines():
        meta, path = line.split("\t", 1)
        oid = meta.split()[2]
        data = (root / path).read_bytes()
        entries.append([path, len(data), sha256(data), oid])
    return sorted(entries)


def tree_files(root: Path) -> dict[str, bytes]:
    """Every file under `root` (all must be regular files) -> bytes."""
    found = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            assert stat.S_ISREG(os.lstat(path).st_mode), path
            found[path.relative_to(root).as_posix()] = path.read_bytes()
    return found


class RuntimeCase(TempDirCase):
    def tearDown(self) -> None:
        # Sealed fixture objects are 0o555/0o444; restore owner write on temporary directories only.
        for dirpath, _dirnames, _filenames in os.walk(self.tmp):
            os.chmod(dirpath, stat.S_IRWXU)
        super().tearDown()

    def refuses(self, code: str, call, *args, **kwargs) -> RuntimeRefusal:
        with self.assertRaises(RuntimeRefusal) as ctx:
            call(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)
        return ctx.exception

    def new_store(self) -> Path:
        store = self.tmp / "store"
        (store / "objects" / "sources").mkdir(parents=True)
        return store

    @contextlib.contextmanager
    def writable(self, directory: Path):
        os.chmod(directory, 0o755)
        try:
            yield directory
        finally:
            os.chmod(directory, 0o555)

    def replace_bytes(self, path: Path, data: bytes) -> None:
        """Rewrite one sealed read-only file in place (fixture tamper), keeping its read-only mode."""
        with self.writable(path.parent):
            os.chmod(path, 0o644)
            path.write_bytes(data)
            os.chmod(path, 0o444)


# --------------------------------------------------------------------- B1: hashing races (parent and worker)

RACE_ORIGINAL = b"synthetic hash race fixture 32b\n"
RACE_MUTATED = b"SYNTHETIC HASH RACE FIXTURE 32B\n"
assert len(RACE_ORIGINAL) == len(RACE_MUTATED) == 32


@contextlib.contextmanager
def mutate_after_first_read(target: Path, mutation):
    """Wrap os.read so `mutation` runs once, right after the first non-empty read from `target`'s inode."""
    real_read = os.read
    info = os.stat(target)
    state = {"mutated": False}

    def read(fd, size):
        data = real_read(fd, size)
        if data and not state["mutated"]:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino) == (info.st_dev, info.st_ino):
                state["mutated"] = True
                mutation()
        return data

    with mock.patch.object(os, "read", read):
        yield state


def rewrite_in_place(path: Path) -> None:
    """Same inode, same size, different bytes. mtime is stepped explicitly so the change is visible even where
    the filesystem timestamp granularity is coarse."""
    before = os.stat(path)
    with open(path, "r+b") as handle:
        handle.write(RACE_MUTATED)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))


def replace_file(path: Path) -> None:
    """A different inode of the same size renamed over the path."""
    other = path.with_name(path.name + ".new")
    other.write_bytes(RACE_MUTATED)
    os.replace(other, path)


class ParentHashTests(RuntimeCase):
    """`runtime._hash_path`: bounded no-follow regular-file hashing that must refuse any change during the read."""

    def setUp(self) -> None:
        super().setUp()
        self.target = self.tmp / "race.bin"
        self.target.write_bytes(RACE_ORIGINAL)

    def test_bounded_hash_blob_and_byte_keep_limit(self) -> None:
        blob = hashlib.sha1(b"blob 32\0" + RACE_ORIGINAL).hexdigest()
        digest, got_blob, size, kept = runtime._hash_path(self.target, budget=32, blob_algorithm="sha1", keep=64)
        self.assertEqual((digest, got_blob, size, kept), (sha256(RACE_ORIGINAL), blob, 32, RACE_ORIGINAL))
        self.assertIsNone(runtime._hash_path(self.target, budget=32, keep=0)[3])
        self.assertIsNone(runtime._hash_path(self.target, budget=32, keep=31)[3])  # keep is a byte limit
        self.assertEqual(runtime._hash_path(self.target, budget=32, keep=32)[3], RACE_ORIGINAL)

    def test_byte_budget_refuses_before_reading(self) -> None:
        self.refuses("max_bytes_exceeded", runtime._hash_path, self.target, budget=31)

    def test_symlink_is_not_followed(self) -> None:
        link = self.tmp / "link.bin"
        link.symlink_to(self.target)
        self.refuses("symlink_in_scope", runtime._hash_path, link, budget=64)

    @requires_linux
    def test_special_file_refuses(self) -> None:
        fifo = self.tmp / "fifo"
        os.mkfifo(fifo)
        self.refuses("non_regular_file", runtime._hash_path, fifo, budget=64)

    def test_in_place_same_size_change_during_read_refuses(self) -> None:
        with mutate_after_first_read(self.target, lambda: rewrite_in_place(self.target)) as state:
            with self.assertRaises(RuntimeRefusal) as ctx:
                runtime._hash_path(self.target, budget=64, blob_algorithm="sha1", keep=64)
        self.assertTrue(state["mutated"], "the fixture race did not happen")
        self.assertEqual(ctx.exception.code, "source_changed_during_read", ctx.exception.detail)
        self.assertEqual(self.target.read_bytes(), RACE_MUTATED)

    def test_replacement_during_read_refuses(self) -> None:
        with mutate_after_first_read(self.target, lambda: replace_file(self.target)) as state:
            with self.assertRaises(RuntimeRefusal) as ctx:
                runtime._hash_path(self.target, budget=64, keep=64)
        self.assertTrue(state["mutated"], "the fixture race did not happen")
        self.assertEqual(ctx.exception.code, "source_changed_during_read", ctx.exception.detail)


class WorkerHashTests(RuntimeCase):
    """`runtime_worker._sha256_file` (compiled identities, selected metadata, bundle tool) under the same races."""

    def setUp(self) -> None:
        super().setUp()
        self.target = self.tmp / "race.bin"
        self.target.write_bytes(RACE_ORIGINAL)

    def worker_refuses(self, path: Path, budget: int) -> WorkerRefusal:
        with self.assertRaises(WorkerRefusal) as ctx:
            runtime_worker._sha256_file(str(path), budget)
        self.assertEqual(ctx.exception.status, "refused")
        return ctx.exception

    def test_bounded_hash(self) -> None:
        self.assertEqual(runtime_worker._sha256_file(str(self.target), 32), (sha256(RACE_ORIGINAL), 32))

    def test_byte_budget_refuses(self) -> None:
        self.assertEqual(self.worker_refuses(self.target, 31).code, "compiled_identity_bound_exceeded")

    def test_symlink_and_special_file_refuse(self) -> None:
        link = self.tmp / "link.bin"
        link.symlink_to(self.target)
        self.worker_refuses(link, 64)
        if LINUX:
            fifo = self.tmp / "fifo"
            os.mkfifo(fifo)
            self.worker_refuses(fifo, 64)

    def test_in_place_same_size_change_during_read_refuses(self) -> None:
        with mutate_after_first_read(self.target, lambda: rewrite_in_place(self.target)) as state:
            self.worker_refuses(self.target, 64)
        self.assertTrue(state["mutated"], "the fixture race did not happen")

    def test_replacement_during_read_refuses(self) -> None:
        with mutate_after_first_read(self.target, lambda: replace_file(self.target)) as state:
            self.worker_refuses(self.target, 64)
        self.assertTrue(state["mutated"], "the fixture race did not happen")


# ------------------------------------------------------- B2: clean committed capture -> seal -> verify -> reuse


@requires_git
@requires_packaging
class SourceCaptureTests(RuntimeCase):
    def capture(self, kind: str, root: Path) -> tuple[dict, list, dict]:
        observation = runtime.observe_source(kind, str(root), expected_revision=None, limits=LIMITS)
        files, captured = runtime.hash_scope(observation, LIMITS)
        return observation, files, captured

    def test_clean_syrup_capture_seal_verify_reuse(self) -> None:
        syrup = make_syrup(self.tmp)
        observation, files, captured = self.capture("syrup", syrup)
        self.assertEqual(observation["head"], git(syrup, "rev-parse", "HEAD").strip())
        self.assertEqual(observation["tree"], git(syrup, "rev-parse", "HEAD^{tree}").strip())
        expected = committed_scope(syrup, "src", "pyproject.toml")
        self.assertEqual([entry[0] for entry in expected],
                         ["pyproject.toml", "src/maple_syrup/__init__.py", "src/maple_syrup/cli.py"])
        self.assertEqual(files, expected)
        self.assertEqual(captured, {"pyproject.toml": SYRUP_PYPROJECT.encode(),
                                    "src/maple_syrup/__init__.py": SYRUP_INIT.encode()})

        store = self.new_store()
        first = runtime.seal_source(str(store), "syrup", observation, files, captured, run_id="run-1", limits=LIMITS)
        object_id = first["object_id"]
        location = store / "objects" / "sources" / object_id
        self.assertFalse(first["reused"])
        self.assertEqual(first["path"], str(location))
        self.assertEqual(sorted(os.listdir(store / "objects" / "sources")), [object_id])  # no staging left

        manifest = json.loads((location / "manifest.json").read_bytes())
        identity = manifest["identity"]
        self.assertEqual(object_id, sha256(canonical_json(identity).encode("ascii")))
        self.assertEqual(identity["files"], expected)
        self.assertEqual(identity["project"], {"name": "maple-syrup", "version": "0.1.0",
                                               "pyproject_sha256": sha256(SYRUP_PYPROJECT.encode())})
        self.assertEqual(identity["overlay"]["generator_sha256"], runtime.GENERATOR_SHA256)
        self.assertEqual(runtime.GENERATOR_SHA256, sha256(Path(runtime.__file__).read_bytes()))

        self.assertEqual(tree_files(location / "source"), {path: (syrup / path).read_bytes() for path, *_ in expected})
        overlay = tree_files(location / "editable")
        self.assertEqual(set(overlay), {f"{SYRUP_DIST_INFO}/{name}"
                                        for name in (*OVERLAY_NAMES, "entry_points.txt")})
        label = json.loads(overlay[f"{SYRUP_DIST_INFO}/MASY_OVERLAY.json"])
        self.assertIs(label["not_an_installation"], True)
        self.assertEqual((label["source_object"], label["distribution"], label["version"]),
                         (object_id, "maple-syrup", "0.1.0"))
        direct = json.loads(overlay[f"{SYRUP_DIST_INFO}/direct_url.json"])
        self.assertEqual(direct["url"], (location / "source").as_uri())
        for dirpath, dirnames, filenames in os.walk(location):
            for name in [*dirnames, *filenames]:
                self.assertEqual(os.lstat(os.path.join(dirpath, name)).st_mode & 0o222, 0, name)

        # Ordinary importlib.metadata reads the generated overlay (no discovery patching).
        dists = list(importlib.metadata.distributions(name="maple-syrup", path=[str(location / "editable")]))
        self.assertEqual([d.version for d in dists], ["0.1.0"])
        self.assertEqual([(ep.name, ep.value) for ep in dists[0].entry_points.select(group="console_scripts")],
                         [("maple-syrup-fixture", "maple_syrup.cli:main")])

        summary = runtime.verify_source_object(location, object_id, LIMITS)
        self.assertEqual(summary["overlay_regeneration"], "exact")
        self.assertEqual(summary["overlay_files"], {rel: sha256(data) for rel, data in overlay.items()})
        self.assertEqual(summary["package_digest"], observation["package"]["digest_sha256"])

        second = runtime.seal_source(str(store), "syrup", observation, files, captured, run_id="run-2",
                                     limits=LIMITS)
        self.assertTrue(second["reused"])
        self.assertEqual(second["object_id"], object_id)
        self.assertEqual(sorted(os.listdir(store / "objects" / "sources")), [object_id])

    def test_seal_from_independently_read_committed_inputs(self) -> None:
        """The isolated seal path: inputs built here from `git ls-tree` and plain reads, not from hash_scope."""
        syrup = make_syrup(self.tmp)
        observation = runtime.observe_source("syrup", str(syrup), expected_revision=None, limits=LIMITS)
        files = committed_scope(syrup, "src", "pyproject.toml")
        captured = {path: (syrup / path).read_bytes() for path in ("pyproject.toml", "src/maple_syrup/__init__.py")}
        store = self.new_store()
        sealed = runtime.seal_source(str(store), "syrup", observation, files, captured, run_id="iso", limits=LIMITS)
        self.assertFalse(sealed["reused"])
        self.assertEqual(sealed["dist_info"], SYRUP_DIST_INFO)
        self.assertEqual(runtime.verify_source_object(sealed["path"], sealed["object_id"], LIMITS)["object_id"],
                         sealed["object_id"])

    def test_masy_capture_records_launcher_and_aligned_version(self) -> None:
        masy = make_masy(self.tmp)
        observation, files, captured = self.capture("masy", masy)
        expected = committed_scope(masy, "src", "pyproject.toml", "tools/masy_runtime.py")
        self.assertEqual(files, expected)
        self.assertIn("tools/masy_runtime.py", [entry[0] for entry in files])
        self.assertEqual(set(captured), {"pyproject.toml", "src/masy/__init__.py"})
        sealed = runtime.seal_source(str(self.new_store()), "masy", observation, files, captured, run_id="m",
                                     limits=LIMITS)
        location = Path(sealed["path"])
        self.assertEqual((location / "source" / "tools" / "masy_runtime.py").read_bytes(), LAUNCHER_FIXTURE.encode())
        self.assertEqual(set(tree_files(location / "editable")),
                         {f"masy-0.0.3.dev0.dist-info/{name}" for name in OVERLAY_NAMES})

    def test_maple_object_has_no_overlay(self) -> None:
        maple = make_repo(self.tmp, "MAPLE", "maple", "maple", files={"src/maple/core.py": "VALUE = 1\n"})
        observation, files, captured = self.capture("maple", maple)
        self.assertEqual(files, committed_scope(maple, "src", "pyproject.toml"))
        sealed = runtime.seal_source(str(self.new_store()), "maple", observation, files, captured, run_id="p",
                                     limits=LIMITS)
        self.assertIsNone(sealed["overlay_files"])
        self.assertFalse((Path(sealed["path"]) / "editable").exists())


# ------------------------------------------------------------------------------- closed config and limits


def config_table(**overrides) -> dict:
    table = {
        "schema": "masy.runtime.v1",
        "sources": {"maple": {"root": "/synthetic/MAPLE",
                              "distribution_metadata": "/synthetic/venv/lib/site-packages/maple-0.dist-info"},
                    "syrup": {"root": "/synthetic/SYRUP"}},
        "runtime": {"python": "/synthetic/venv/bin/python"},
    }
    table.update(overrides)
    return table


class ConfigAndLimitsTests(RuntimeCase):
    def test_minimal_config_uses_default_store_and_limits(self) -> None:
        config = runtime.parse_runtime_config(config_table())
        self.assertEqual(config.store, "/home/okin/MASY/runs/runtime-store")
        self.assertEqual(config.store, runtime.DEFAULT_STORE)
        self.assertEqual(config.limits, RuntimeLimits())
        self.assertEqual((config.maple.package, config.syrup.package), ("maple", "maple_syrup"))
        self.assertIsNone(config.masy_expected_revision)

    def test_closed_schema(self) -> None:
        cases = {
            "unknown_config_field": [
                config_table(extra=1),
                config_table(sources={**config_table()["sources"], "maple2": {}}),
                config_table(runtime={"python": "/synthetic/venv/bin/python", "gpu": True}),
                config_table(limits={"max_file": 3}),
                config_table(sources={"maple": {**config_table()["sources"]["maple"], "patch": "x"},
                                      "syrup": {"root": "/synthetic/SYRUP"}}),
            ],
            "missing_config_field": [
                config_table(sources={"maple": {"root": "/synthetic/MAPLE"}, "syrup": {"root": "/synthetic/SYRUP"}}),
                config_table(sources={"maple": config_table()["sources"]["maple"]}),
                config_table(runtime={}),
            ],
            "invalid_config_value": [
                config_table(schema="masy.runtime.v0"),
                config_table(runtime={"python": "relative/python"}),
                config_table(runtime={"python": "/synthetic/venv/../bin/python"}),
                config_table(sources={**config_table()["sources"],
                                      "masy": {"expected_revision": "abc123"}}),
            ],
            "invalid_config_type": [
                config_table(runtime={"python": ""}),
                config_table(sources=[]),
            ],
        }
        for code, tables in cases.items():
            for table in tables:
                with self.subTest(code=code, table=table):
                    self.refuses(code, runtime.parse_runtime_config, table)

    def test_limits_must_be_typed_positive_finite(self) -> None:
        invalid = [("max_files", True), ("max_files", 0), ("max_files", -1), ("max_files", 1.5),
                   ("max_bytes", "64"), ("worker_timeout_s", float("nan")), ("worker_timeout_s", float("inf")),
                   ("git_timeout_s", 0.0), ("git_timeout_s", -1.0), ("git_timeout_s", False)]
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                self.refuses("invalid_config_value", runtime.parse_runtime_config,
                             config_table(limits={name: value}))
                self.refuses("invalid_config_value", RuntimeLimits, **{name: value})
        config = runtime.parse_runtime_config(config_table(limits={"worker_timeout_s": 5, "max_files": 7}))
        self.assertEqual((config.limits.worker_timeout_s, config.limits.max_files), (5.0, 7))
        self.assertIsInstance(config.limits.worker_timeout_s, float)

    def test_toml_non_finite_and_record(self) -> None:
        head = ('schema = "masy.runtime.v1"\n[sources.maple]\nroot = "/synthetic/MAPLE"\n'
                'distribution_metadata = "/synthetic/maple-0.dist-info"\n[sources.syrup]\nroot = "/synthetic/SYRUP"\n'
                '[runtime]\npython = "/synthetic/venv/bin/python"\nstore = "/tmp/synthetic-store"\n')
        for literal in ("nan", "inf", "-inf"):
            path = write(self.tmp / f"bad-{literal}.toml", head + f"[limits]\nworker_timeout_s = {literal}\n")
            with self.subTest(literal=literal):
                self.refuses("invalid_config_value", runtime.load_runtime_config, str(path))
        good = write(self.tmp / "good.toml", head)
        config = runtime.load_runtime_config(str(good))
        self.assertEqual(config.sha256, sha256(good.read_bytes()))
        self.assertEqual(config.record()["runtime"], {"python": "/synthetic/venv/bin/python",
                                                      "store": "/tmp/synthetic-store"})
        self.refuses("config_unreadable", runtime.load_runtime_config, str(self.tmp / "missing.toml"))


# ----------------------------------------------------------------------------- store/output separation


class StoreLocationTests(RuntimeCase):
    def setUp(self) -> None:
        super().setUp()
        self.maple = self.tmp / "MAPLE"
        self.venv = self.tmp / "venv"
        self.wrapper = self.tmp / "MASY"
        for path in (self.maple, self.venv, self.wrapper / "src", self.wrapper / "runs"):
            path.mkdir(parents=True)
        self.protected = {"sources.maple.root": str(self.maple), "runtime.python environment": str(self.venv),
                          **{f"wrapper {rel}": str(self.wrapper / rel)
                             for rel in ("src", "tools", "pyproject.toml", ".git")}}

    def check(self, store: Path | str, protected: dict | None = None) -> None:
        runtime.check_store_location(str(store), wrapper_roots=[str(self.wrapper)],
                                     protected=self.protected if protected is None else protected)

    def test_store_under_wrapper_runs_is_allowed(self) -> None:
        self.check(self.wrapper / "runs" / "runtime-store")
        self.check(self.tmp / "elsewhere" / "store")

    def test_overlapping_stores_refuse(self) -> None:
        for store in (self.maple, self.maple / "store", self.venv / "store", self.tmp,
                      self.wrapper, self.wrapper / "runs", self.wrapper / "src" / "store",
                      self.wrapper / "other-store", self.wrapper / ".git" / "store"):
            with self.subTest(store=store):
                exc = self.refuses("store_overlaps_input", self.check, store)
                self.assertTrue(exc.evidence["conflicts"])

    def test_symlink_aliases(self) -> None:
        alias = self.tmp / "maple-alias"
        alias.symlink_to(self.maple, target_is_directory=True)
        protected = {**self.protected, "sources.maple.root": str(alias)}
        self.refuses("store_overlaps_input", self.check, self.maple / "store", protected)
        # A store spelled through a symlinked component is refused outright.
        self.refuses("store_not_canonical", self.check, alias / "store")
        real = self.tmp / "real-parent"
        real.mkdir()
        (self.tmp / "parent-alias").symlink_to(real, target_is_directory=True)
        self.refuses("store_not_canonical", self.check, self.tmp / "parent-alias" / "store")

    def test_store_path_must_be_absolute_and_normalized(self) -> None:
        self.refuses("store_path_invalid", self.check, "relative/store")
        self.refuses("store_path_invalid", self.check, str(self.tmp) + "/x/../store")


# ------------------------------------------------------------------------- committed source selection guards


@requires_git
class SourceSelectionTests(RuntimeCase):
    def setUp(self) -> None:
        super().setUp()
        self.syrup = make_syrup(self.tmp)

    def observe(self, limits: RuntimeLimits = LIMITS) -> dict:
        return runtime.observe_source("syrup", str(self.syrup), expected_revision=None, limits=limits)

    def refusal(self, code: str, limits: RuntimeLimits = LIMITS) -> RuntimeRefusal:
        exc = self.refuses(code, self.observe, limits)
        self.assertEqual(exc.evidence.get("source"), "syrup")
        return exc

    def test_clean_head_and_blob_identity(self) -> None:
        observation = self.observe()
        self.assertEqual(observation["head"], git(self.syrup, "rev-parse", "HEAD").strip())
        self.assertEqual(observation["git_status"], "clean")
        blob = git(self.syrup, "rev-parse", "HEAD:pyproject.toml").strip()
        self.assertEqual(observation["pyproject"]["git_blob"], blob)
        self.assertEqual(observation["tracked_scope"]["file_count"], 3)

    def test_expected_revision_is_enforced(self) -> None:
        head = git(self.syrup, "rev-parse", "HEAD").strip()
        self.assertEqual(runtime.observe_source("syrup", str(self.syrup), expected_revision=head,
                                                limits=LIMITS)["head"], head)
        self.refuses("expected_revision_mismatch", runtime.observe_source, "syrup", str(self.syrup),
                     expected_revision="0" * 40, limits=LIMITS)

    def test_dirty_and_extra_sources_refuse(self) -> None:
        write(self.syrup / "src/maple_syrup/cli.py", "def main():\n    return 9\n")
        self.refusal("dirty_scope")
        git(self.syrup, "checkout", "--", "src/maple_syrup/cli.py")
        write(self.syrup / "src/maple_syrup/extra.py", "X = 1\n")
        self.refusal("dirty_scope")

    def test_ignored_binary_refuses(self) -> None:
        (self.syrup / "src/maple_syrup/_native.so").write_bytes(b"\x7fELF synthetic")
        self.assertEqual(git(self.syrup, "status", "--porcelain", "--", "src"), "")
        self.refusal("untracked_runtime_input")

    def test_symlinks_refuse(self) -> None:
        (self.syrup / "src/maple_syrup/alias.so").symlink_to("cli.py")  # ignored, so only the walk sees it
        self.refusal("symlink_in_scope")
        (self.syrup / "src/maple_syrup/alias.so").unlink()
        (self.syrup / "src/maple_syrup/alias.py").symlink_to("cli.py")
        commit_all(self.syrup, "committed symlink")
        self.refusal("symlink_in_scope")

    @requires_linux
    def test_special_file_refuses(self) -> None:
        os.mkfifo(self.syrup / "src/maple_syrup/pipe.so")
        self.refusal("non_regular_file")

    def test_file_and_byte_limits(self) -> None:
        self.refusal("max_files_exceeded", RuntimeLimits(max_files=1))
        self.refusal("max_bytes_exceeded", RuntimeLimits(max_bytes=8))

    def test_concurrent_change_after_observation_refuses(self) -> None:
        observation = self.observe()
        write(self.syrup / "src/maple_syrup/cli.py", "def main():\n    return 7\n")  # same size, uncommitted
        exc = self.refuses("content_differs_from_head", runtime.hash_scope, observation, LIMITS)
        self.assertEqual(exc.evidence.get("source"), "syrup")

    def test_new_commit_between_observations_refuses(self) -> None:
        before = self.observe()
        write(self.syrup / "src/maple_syrup/cli.py", "def main():\n    return 8\n")
        commit_all(self.syrup, "concurrent commit")
        after = self.observe()
        self.refuses("source_changed_during_prepare", runtime.check_unchanged, before, after)
        self.refuses("content_differs_from_head", runtime.hash_scope, before, LIMITS)

    def test_scope_disagreeing_with_observation_refuses(self) -> None:
        observation = self.observe()
        drifted = copy.deepcopy(observation)
        drifted["tracked_scope"]["digest_sha256"] = "0" * 64
        self.refuses("source_changed_during_prepare", runtime.hash_scope, drifted, LIMITS)
        drifted = copy.deepcopy(observation)
        drifted["package"]["digest_sha256"] = "0" * 64
        self.refuses("source_changed_during_prepare", runtime.hash_scope, drifted, LIMITS)


# ------------------------------------------------------------- sealed objects: tamper, no-clobber, retention


@requires_git
@requires_packaging
class SealedObjectTests(RuntimeCase):
    def setUp(self) -> None:
        super().setUp()
        self.syrup = make_syrup(self.tmp)
        self.store = self.new_store()
        self.sources = self.store / "objects" / "sources"
        sealed = self.seal("seed")
        self.object_id = sealed["object_id"]
        self.location = Path(sealed["path"])

    def seal(self, run_id: str, **hooks) -> dict:
        observation = runtime.observe_source("syrup", str(self.syrup), expected_revision=None, limits=LIMITS)
        files, captured = runtime.hash_scope(observation, LIMITS)
        return runtime.seal_source(str(self.store), "syrup", observation, files, captured, run_id=run_id,
                                   limits=LIMITS, **hooks)

    def verify(self, object_id: str | None = None, **kwargs) -> dict:
        return runtime.verify_source_object(self.location, object_id or self.object_id, LIMITS, **kwargs)

    def test_tampered_source_refuses_and_is_never_repaired(self) -> None:
        target = self.location / "source/src/maple_syrup/cli.py"
        tampered = SYRUP_CLI.replace("0", "5").encode()
        self.replace_bytes(target, tampered)
        self.refuses("object_tamper", self.verify)
        self.refuses("object_tamper", self.seal, "again")  # reuse verifies first; nothing is rewritten
        self.assertEqual(target.read_bytes(), tampered)
        self.assertEqual(sorted(os.listdir(self.sources)), [self.object_id])

    def test_added_file_refuses(self) -> None:
        package = self.location / "source/src/maple_syrup"
        with self.writable(package):
            (package / "extra.py").write_text("X = 1\n")
            os.chmod(package / "extra.py", 0o444)
        self.refuses("object_tamper", self.verify)

    def test_symlink_inside_object_refuses(self) -> None:
        package = self.location / "source/src/maple_syrup"
        with self.writable(package):
            (package / "alias.py").symlink_to("cli.py")
        self.refuses("object_tamper", self.verify)

    def test_writable_mode_refuses(self) -> None:
        os.chmod(self.location / "source/src/maple_syrup/cli.py", 0o644)
        self.refuses("object_tamper", self.verify)

    def test_overlay_tamper_refuses(self) -> None:
        metadata = self.location / "editable" / SYRUP_DIST_INFO / "METADATA"
        self.replace_bytes(metadata, metadata.read_bytes().replace(b"Version: 0.1.0", b"Version: 0.1.1"))
        self.refuses("object_tamper", self.verify)

    def test_wrong_object_id_refuses(self) -> None:
        self.refuses("object_tamper", self.verify, "0" * 64)

    def test_other_generator_is_checked_semantically(self) -> None:
        self.assertEqual(self.verify()["overlay_regeneration"], "exact")
        self.assertEqual(self.verify(current_generator="0" * 64)["overlay_regeneration"],
                         "semantic_only_generator_differs")

    def test_new_commit_seals_new_object_and_retains_old(self) -> None:
        write(self.syrup / "src/maple_syrup/cli.py", "def main():\n    return 2\n")
        commit_all(self.syrup, "next")
        newer = self.seal("next")
        self.assertFalse(newer["reused"])
        self.assertNotEqual(newer["object_id"], self.object_id)
        self.assertEqual(sorted(os.listdir(self.sources)), sorted([self.object_id, newer["object_id"]]))
        self.assertEqual(self.verify()["object_id"], self.object_id)  # the old object is retained intact

    def test_source_change_before_copy_refuses_and_retains_staging(self) -> None:
        write(self.syrup / "src/maple_syrup/cli.py", "def main():\n    return 3\n")
        commit_all(self.syrup, "racing")
        observation = runtime.observe_source("syrup", str(self.syrup), expected_revision=None, limits=LIMITS)
        files, captured = runtime.hash_scope(observation, LIMITS)
        write(self.syrup / "src/maple_syrup/cli.py", "def main():\n    return 4\n")  # after hashing
        exc = self.refuses("source_changed_during_read", runtime.seal_source, str(self.store), "syrup", observation,
                           files, captured, run_id="race", limits=LIMITS)
        staging = self.sources / ".staging-race-syrup"
        self.assertEqual(exc.evidence.get("retained_staging"), str(staging))
        self.assertTrue(staging.is_dir())
        self.assertEqual(sorted(name for name in os.listdir(self.sources) if not name.startswith(".")),
                         [self.object_id])

    def test_staged_copy_tamper_refuses_and_retains_staging(self) -> None:
        write(self.syrup / "src/maple_syrup/cli.py", "def main():\n    return 6\n")
        commit_all(self.syrup, "hooked")
        staging = self.sources / ".staging-hooked-syrup"

        def tamper_staging() -> None:
            (staging / "source/src/maple_syrup/cli.py").write_text("def main():\n    return 1\n")

        exc = self.refuses("object_tamper", self.seal, "hooked", _after_copy=tamper_staging)
        self.assertEqual(exc.evidence.get("retained_staging"), str(staging))
        self.assertTrue(staging.is_dir())
        self.assertEqual(sorted(name for name in os.listdir(self.sources) if not name.startswith(".")),
                         [self.object_id])


# -------------------------------------------------------------------- static metadata and generated overlays


def record_rows(files: dict[str, bytes], info: str) -> dict[str, bytes]:
    """Recompute a wheel-style RECORD for `files` (so semantic checks, not RECORD digests, are what refuses)."""
    rows = []
    for rel in sorted(files):
        if rel != f"{info}/RECORD":
            digest = base64.urlsafe_b64encode(hashlib.sha256(files[rel]).digest()).rstrip(b"=").decode()
            rows.append(f"{rel},sha256={digest},{len(files[rel])}")
    rows.append(f"{info}/RECORD,,")
    return {**files, f"{info}/RECORD": ("\n".join(rows) + "\n").encode()}


@requires_packaging
class StaticMetadataTests(RuntimeCase):
    def validate(self, text: str, name: str = "maple-syrup") -> dict:
        return runtime.validate_static_project(text.encode(), name)

    def test_static_fields_are_normalized(self) -> None:
        project = self.validate(SYRUP_PYPROJECT)
        self.assertEqual((project["name"], project["version"], project["requires_python"]),
                         ("maple-syrup", "0.1.0", ">=3.11"))
        self.assertEqual(project["dependencies"], ["packaging>=23"])
        self.assertEqual(list(project["optional_dependencies"]), ["test"])
        self.assertEqual(project["scripts"], {"maple-syrup-fixture": "maple_syrup.cli:main"})
        self.assertEqual(project["pyproject_sha256"], sha256(SYRUP_PYPROJECT.encode()))

    def test_unsupported_or_invalid_static_metadata_refuses(self) -> None:
        base = '[project]\nname = "maple-syrup"\n'
        cases = [
            ("pyproject_dynamic_refused", base + 'dynamic = ["version"]\n'),
            ("pyproject_key_unsupported", base + 'version = "0.1.0"\nlicense = "MIT"\n'),
            ("pyproject_key_unsupported", base + 'version = "0.1.0"\n[project.entry-points.x]\ny = "a:b"\n'),
            ("pyproject_project_mismatch", '[project]\nname = "other"\nversion = "0.1.0"\n'),
            ("pyproject_version_invalid", base + 'version = "1.0.0-dev1"\n'),
            ("pyproject_version_invalid", base + 'version = "v1"\n'),
            ("pyproject_version_invalid", base + 'version = "not a version"\n'),
            ("pyproject_value_invalid", base + 'version = "0.1.0"\ndependencies = ["numpy >>> 1"]\n'),
            ("pyproject_value_invalid", base + 'version = "0.1.0"\ndependencies = "numpy"\n'),
            ("pyproject_value_invalid", base + 'version = "0.1.0"\nrequires-python = "three"\n'),
            ("pyproject_value_invalid", base + 'version = "0.1.0"\n[project.optional-dependencies]\nTest = []\n'),
            ("pyproject_value_invalid", base + 'version = "0.1.0"\n[project.scripts]\nx = "no-colon"\n'),
            ("pyproject_unreadable", "[project\n"),
        ]
        for code, text in cases:
            with self.subTest(code=code, text=text):
                self.refuses(code, self.validate, text)

    def test_missing_or_non_string_version_refuses(self) -> None:
        # A typed refusal, not an internal error, when [project] has no usable static version.
        for text in ('[project]\nname = "maple-syrup"\n', '[project]\nname = "maple-syrup"\nversion = 1\n'):
            with self.subTest(text=text):
                self.refuses("pyproject_version_invalid", self.validate, text)

    def test_literal_version_alignment(self) -> None:
        project = self.validate(SYRUP_PYPROJECT)
        runtime.check_version_alignment(project, SYRUP_INIT.encode(), "init")
        self.refuses("version_mismatch", runtime.check_version_alignment, project, b'__version__ = "0.1.1"\n', "init")
        for body in (b'"""no version"""\n', b'__version__ = "0.1.0"\n__version__ = "0.1.0"\n',
                     b"from importlib.metadata import version\n__version__ = version('maple-syrup')\n",
                     b"def broken(:\n"):
            with self.subTest(body=body):
                self.refuses("version_alignment_unverifiable", runtime.check_version_alignment, project, body, "init")


@requires_packaging
class OverlayTests(RuntimeCase):
    OBJECT_ID = "a" * 64

    def setUp(self) -> None:
        super().setUp()
        self.project = runtime.validate_static_project(SYRUP_PYPROJECT.encode(), "maple-syrup")
        self.object_dir = self.tmp / "objects" / self.OBJECT_ID
        self.files = runtime.generate_overlay(self.project, package="maple_syrup", object_dir=self.object_dir,
                                              object_id=self.OBJECT_ID)

    def check(self, files: dict[str, bytes], **overrides) -> None:
        kwargs = {"package": "maple_syrup", "object_dir": self.object_dir, "object_id": self.OBJECT_ID,
                  "generator_sha256": runtime.GENERATOR_SHA256, **overrides}
        runtime.check_overlay_semantics(files, self.project, **kwargs)

    def test_generated_overlay_is_ordinary_readable_metadata(self) -> None:
        self.check(self.files)
        self.assertEqual(set(self.files), {f"{SYRUP_DIST_INFO}/{n}" for n in (*OVERLAY_NAMES, "entry_points.txt")})
        overlay = self.tmp / "overlay"
        for rel, data in self.files.items():
            (overlay / rel).parent.mkdir(parents=True, exist_ok=True)
            (overlay / rel).write_bytes(data)
        [dist] = importlib.metadata.distributions(name="maple-syrup", path=[str(overlay)])
        self.assertEqual((dist.metadata["Name"], dist.version, dist.metadata["Requires-Python"]),
                         ("maple-syrup", "0.1.0", ">=3.11"))
        self.assertEqual(dist.metadata.get_all("Provides-Extra"), ["test"])
        from packaging.requirements import Requirement

        requirements = [Requirement(text) for text in dist.requires]
        self.assertEqual(sorted(r.name for r in requirements), ["packaging", "pytest"])
        pytest_req = next(r for r in requirements if r.name == "pytest")
        self.assertTrue(pytest_req.marker.evaluate({"extra": "test", "python_version": "3.12"}))
        self.assertFalse(pytest_req.marker.evaluate({"extra": "", "python_version": "3.12"}))
        self.assertEqual({(ep.name, ep.value) for ep in dist.entry_points.select(group="console_scripts")},
                         {("maple-syrup-fixture", "maple_syrup.cli:main")})
        self.assertEqual(dist.read_text("INSTALLER"), "masy.runtime\n")
        label = json.loads(dist.read_text("MASY_OVERLAY.json"))
        self.assertIs(label["not_an_installation"], True)
        self.assertEqual(sorted(str(p) for p in dist.files), sorted(self.files))
        self.assertEqual(self.files, runtime.generate_overlay(self.project, package="maple_syrup",
                                                              object_dir=self.object_dir, object_id=self.OBJECT_ID))

    def test_semantic_tamper_refuses_even_with_consistent_record(self) -> None:
        info = SYRUP_DIST_INFO

        def edited(name: str, old: bytes, new: bytes) -> dict[str, bytes]:
            files = dict(self.files)
            self.assertIn(old, files[f"{info}/{name}"])
            files[f"{info}/{name}"] = files[f"{info}/{name}"].replace(old, new)
            return record_rows(files, info)

        label = json.loads(self.files[f"{info}/MASY_OVERLAY.json"])
        cases = {
            "version": edited("METADATA", b"Version: 0.1.0", b"Version: 0.1.1"),
            "dependency": edited("METADATA", b"packaging>=23", b"packaging>=24"),
            "script": edited("entry_points.txt", b"maple_syrup.cli:main", b"maple_syrup.cli:other"),
            "top_level": edited("top_level.txt", b"maple_syrup", b"maple"),
            "installer": edited("INSTALLER", b"masy.runtime", b"pip"),
            "label": record_rows({**self.files, f"{info}/MASY_OVERLAY.json": json.dumps(
                {**label, "not_an_installation": False}).encode()}, info),
            "extra_file": record_rows({**self.files, f"{info}/extra.txt": b"x\n"}, info),
            "record_digest": {**self.files, f"{info}/top_level.txt": b"maple_syrup\n\n"},
        }
        for name, files in cases.items():
            with self.subTest(name=name):
                self.refuses("overlay_invalid", self.check, files)
        with self.subTest(name="other_object_dir"):
            self.refuses("overlay_invalid", self.check, self.files, object_dir=self.tmp / "elsewhere")
        with self.subTest(name="other_object_id"):
            self.refuses("overlay_invalid", self.check, self.files, object_id="b" * 64)
        with self.subTest(name="other_generator"):
            self.refuses("overlay_invalid", self.check, self.files, generator_sha256="c" * 64)


# --------------------------------------------------- worker: ordinary importlib.metadata selection and shadows


@requires_packaging
class SelectedDistributionTests(RuntimeCase):
    def setUp(self) -> None:
        super().setUp()
        project = runtime.validate_static_project(SYRUP_PYPROJECT.encode(), "maple-syrup")
        files = runtime.generate_overlay(project, package="maple_syrup", object_dir=self.tmp / "obj",
                                         object_id="a" * 64)
        self.sealed = self.tmp / "sealed-editable"
        for rel, data in files.items():
            (self.sealed / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.sealed / rel).write_bytes(data)
        self.shadow = self.tmp / "shadow"
        write(self.shadow / "maple_syrup-0.0.1.dist-info/METADATA",
              "Metadata-Version: 2.1\nName: maple-syrup\nVersion: 0.0.1\n")
        self.spec = {"name": "maple-syrup", "dir": str(self.sealed),
                     "files": {rel: sha256(data) for rel, data in files.items()}}

    def worker_refuses(self, code: str, specs: list, path: list[str]) -> None:
        with self.assertRaises(WorkerRefusal) as ctx:
            runtime_worker.selected_distributions(specs, path=path)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)

    def test_sealed_overlay_selected_and_shadow_reported(self) -> None:
        selected, shadowed = runtime_worker.selected_distributions([self.spec],
                                                                   path=[str(self.sealed), str(self.shadow)])
        self.assertEqual((selected["maple-syrup"]["location"], selected["maple-syrup"]["version"]),
                         (str(self.sealed), "0.1.0"))
        self.assertEqual(shadowed["maple-syrup"],
                         [{"location": str(self.shadow), "name": "maple-syrup", "version": "0.0.1"}])

    def test_shadowing_order_missing_and_file_mismatch_refuse(self) -> None:
        self.worker_refuses("selected_distribution_mismatch", [self.spec], [str(self.shadow), str(self.sealed)])
        (self.tmp / "empty").mkdir()
        self.worker_refuses("selected_distribution_missing", [self.spec], [str(self.tmp / "empty")])
        fewer = {**self.spec, "files": dict(list(self.spec["files"].items())[1:])}
        self.worker_refuses("selected_distribution_files_mismatch", [fewer], [str(self.sealed)])
        top_level = self.sealed / SYRUP_DIST_INFO / "top_level.txt"
        top_level.write_bytes(b"maple_syruP\n")  # same size, different bytes
        self.worker_refuses("selected_distribution_files_mismatch", [self.spec], [str(self.sealed)])


# ----------------------------------------------------------------------------------------------------- locks


def rehash(lock: dict) -> dict:
    lock["lock_id"] = sha256(canonical_json({k: v for k, v in lock.items() if k != "lock_id"}).encode("ascii"))
    return lock


def synthetic_lock(store: str) -> dict:
    """A structurally complete lock whose object references are derived from `store` and fixed synthetic ids.
    It names no real object; only the lock schema/self-hash/path rules are exercised."""
    ids = {"maple": "a" * 64, "syrup": "b" * 64, "masy": "c" * 64}
    bundle_id = "d" * 64
    obj = {kind: f"{store}/objects/sources/{object_id}" for kind, object_id in ids.items()}
    bundle = f"{store}/objects/maple-bundles/{bundle_id}"
    materialized = f"{store}/objects/maple-materialized/{bundle_id}"
    # Overlay metadata directories first (selected by importlib.metadata), then the sealed package roots.
    prefix = [f"{materialized}/editable", f"{obj['syrup']}/editable", f"{obj['masy']}/editable",
              f"{obj['masy']}/source/src", f"{obj['syrup']}/source/src", f"{materialized}/source/src"]
    return rehash({
        "schema": "masy.runtime_lock.v1", "lock_id": "",
        "selection": {"policy": "current_local_committed_head", "development_snapshot": False},
        "store": store,
        "sources": {kind: {"root": f"/synthetic/{kind}", "head": "1" * 40, "tree": "2" * 40, "object_id": oid}
                    for kind, oid in ids.items()},
        "objects": {
            "sources": {kind: {"object_id": oid, "path": obj[kind], "package_digest": "e" * 64}
                        for kind, oid in ids.items()},
            "maple_bundle": {"bundle_id": bundle_id, "path": bundle, "manifest_sha256": "f" * 64,
                             "archive_sha256": "0" * 64, "tool_module_sha256": "9" * 64, "package_digest": "8" * 64},
            "maple_materialized": {"bundle_id": bundle_id, "path": materialized, "manifest_sha256": "f" * 64},
        },
        "distributions": {
            "maple": {"overlay_dir": f"{materialized}/editable", "files": {"maple-0.dist-info/METADATA": "7" * 64}},
            "maple-syrup": {"overlay_dir": f"{obj['syrup']}/editable",
                            "files": {f"{SYRUP_DIST_INFO}/METADATA": "7" * 64}},
            "masy": {"overlay_dir": f"{obj['masy']}/editable",
                     "files": {"masy-0.0.3.dev0.dist-info/METADATA": "7" * 64}},
        },
        "runtime": {"python": {"spelling": "/synthetic/venv/bin/python", "realpath": "/synthetic/bin/python3",
                               "sha256": "6" * 64, "venv": None},
                    "bootstrap_sha256": "5" * 64, "sys_path_prefix": prefix},
        "verification": {"bundle": {"bundle_id": bundle_id}, "process": {"sys_path_prefix": prefix}},
        "limits": RuntimeLimits().record(),
        "interpretation": {"transport_ready": False},
    })


def lock_bytes(lock: dict) -> bytes:
    return (canonical_json(lock) + "\n").encode("ascii")


class LockTests(RuntimeCase):
    def setUp(self) -> None:
        super().setUp()
        self.store = str(self.tmp / "store")
        self.locks = self.tmp / "store" / "locks"
        self.locks.mkdir(parents=True)
        self.lock = synthetic_lock(self.store)
        self.name = f"{self.lock['lock_id']}.json"

    def invalid(self, lock: dict, **kwargs) -> None:
        self.refuses("lock_invalid", runtime.validate_lock, lock, store=kwargs.pop("store", self.store), **kwargs)

    def test_valid_lock(self) -> None:
        self.assertIs(runtime.validate_lock(self.lock, store=self.store, lock_name=self.name), self.lock)

    def test_closed_schema_self_hash_name_and_store(self) -> None:
        edited = copy.deepcopy(self.lock)
        edited["sources"]["maple"]["head"] = "3" * 40  # body changed, id not recomputed
        self.invalid(edited)
        self.invalid(rehash({**copy.deepcopy(self.lock), "extra": True}))
        self.invalid(rehash({**copy.deepcopy(self.lock), "schema": "masy.runtime_lock.v2"}))
        without = copy.deepcopy(self.lock)
        del without["verification"]
        self.invalid(rehash(without))
        self.invalid(self.lock, lock_name="0" * 64 + ".json")
        self.invalid(self.lock, store=str(self.tmp / "other-store"))

    def test_object_derived_paths_prefix_and_interpretation(self) -> None:
        def mutated(change) -> dict:
            lock = copy.deepcopy(self.lock)
            change(lock)
            return rehash(lock)

        cases = {
            "object_path": lambda l: l["objects"]["sources"]["syrup"].update(path=f"{self.store}/elsewhere"),
            "object_id_mismatch": lambda l: l["sources"]["masy"].update(object_id="f" * 64),
            "bundle_path": lambda l: l["objects"]["maple_bundle"].update(path=f"{self.store}/objects/x"),
            "materialized_id": lambda l: l["objects"]["maple_materialized"].update(bundle_id="e" * 64),
            "overlay_dir": lambda l: l["distributions"]["masy"].update(overlay_dir="/synthetic/editable"),
            "prefix_order": lambda l: l["runtime"]["sys_path_prefix"].reverse(),
            "transport_ready": lambda l: l["interpretation"].update(transport_ready=True),
            "development_snapshot": lambda l: l["selection"].update(development_snapshot=True),
            "missing_distribution": lambda l: l["distributions"].pop("maple-syrup"),
            "short_head": lambda l: l["sources"]["maple"].update(head="1234"),
        }
        for name, change in cases.items():
            with self.subTest(name=name):
                self.invalid(mutated(change))
        with self.subTest(name="limits"):
            with self.assertRaises(RuntimeRefusal):
                runtime.validate_lock(mutated(lambda l: l["limits"].update(max_files=0)), store=self.store)

    @requires_linux
    def test_publish_reuse_and_load(self) -> None:
        path, reused = runtime.publish_lock(self.store, self.lock, run_id="p1", limits=LIMITS)
        self.assertEqual((path, reused), (self.locks / self.name, False))
        self.assertEqual(path.read_bytes(), lock_bytes(self.lock))
        self.assertEqual(os.lstat(path).st_mode & 0o222, 0)
        self.assertEqual(os.listdir(self.locks), [self.name])
        self.assertEqual(runtime.publish_lock(self.store, self.lock, run_id="p2", limits=LIMITS), (path, True))
        self.assertEqual(os.listdir(self.locks), [self.name])
        for ref in (self.lock["lock_id"], str(path)):
            with self.subTest(ref=ref):
                self.assertEqual(runtime.load_lock_ref(self.store, ref, LIMITS.max_lock_bytes), (self.lock, path))

    @requires_linux
    def test_collision_keeps_original(self) -> None:
        original = self.locks / self.name
        original.write_bytes(b'{"original":true}\n')
        os.chmod(original, 0o444)
        exc = self.refuses("lock_collision", runtime.publish_lock, self.store, self.lock, run_id="c1", limits=LIMITS)
        self.assertEqual(original.read_bytes(), b'{"original":true}\n')
        self.assertTrue(os.path.exists(exc.evidence["retained_staging"]))

    def test_oversized_lock_refuses_before_writing(self) -> None:
        self.refuses("lock_oversized", runtime.publish_lock, self.store, self.lock, run_id="o1",
                     limits=RuntimeLimits(max_lock_bytes=64))
        self.assertEqual(os.listdir(self.locks), [])

    def test_load_refusals(self) -> None:
        self.refuses("lock_ref_invalid", runtime.load_lock_ref, self.store, "../" + self.name, 1 << 20)
        elsewhere = self.tmp / self.name
        elsewhere.write_bytes(lock_bytes(self.lock))
        self.refuses("lock_ref_invalid", runtime.load_lock_ref, self.store, str(elsewhere), 1 << 20)
        self.refuses("lock_unreadable", runtime.load_lock_ref, self.store, self.lock["lock_id"], 1 << 20)  # absent
        target = self.locks / self.name
        target.symlink_to(elsewhere)
        self.refuses("lock_unreadable", runtime.load_lock_ref, self.store, self.lock["lock_id"], 1 << 20)
        target.unlink()
        target.write_bytes(lock_bytes(self.lock))
        self.refuses("lock_unreadable", runtime.load_lock_ref, self.store, self.lock["lock_id"],
                     len(lock_bytes(self.lock)) - 1)
        target.write_text(json.dumps(self.lock, indent=1))
        self.refuses("lock_invalid", runtime.load_lock_ref, self.store, self.lock["lock_id"], 1 << 20)
        target.write_bytes(b'{"schema":1,"schema":2}\n')
        self.refuses("lock_unreadable", runtime.load_lock_ref, self.store, self.lock["lock_id"], 1 << 20)
        target.write_bytes(b'{"x":NaN}\n')
        self.refuses("lock_unreadable", runtime.load_lock_ref, self.store, self.lock["lock_id"], 1 << 20)


# ------------------------------------------------------------------------------------------ disk preflight


class DiskPreflightTests(RuntimeCase):
    def statvfs(self, free_bytes: int):
        seen = []

        def fake(path: str):
            seen.append(path)
            return types.SimpleNamespace(f_bavail=free_bytes // 4096, f_frsize=4096)

        return fake, seen

    def test_refusals_happen_before_anything_exists(self) -> None:
        store = self.tmp / "absent-parent" / "store"
        limits = RuntimeLimits(store_max_new_bytes=1 << 20, disk_reserve_bytes=1 << 20)
        fake, seen = self.statvfs(16 << 20)
        exc = self.refuses("disk_budget_exceeded", runtime.disk_preflight, str(store), (1 << 20) + 1, limits,
                           statvfs=fake)
        self.assertEqual(exc.evidence["estimate_bytes"], (1 << 20) + 1)
        fake, seen = self.statvfs(1 << 20)
        exc = self.refuses("disk_quota_insufficient", runtime.disk_preflight, str(store), 4096, limits, statvfs=fake)
        self.assertEqual(seen, [str(self.tmp)])  # the nearest existing ancestor is probed
        self.assertEqual(exc.evidence["free_bytes"], 1 << 20)
        self.assertEqual(os.listdir(self.tmp), [])

    def test_sufficient_space_is_recorded(self) -> None:
        fake, _seen = self.statvfs(64 << 20)
        record = runtime.disk_preflight(str(self.tmp / "store"), 4096,
                                        RuntimeLimits(store_max_new_bytes=1 << 20, disk_reserve_bytes=1 << 20),
                                        statvfs=fake)
        self.assertEqual((record["probe"], record["free_bytes"], record["estimate_bytes"]),
                         (str(self.tmp), 64 << 20, 4096))
        self.assertEqual(os.listdir(self.tmp), [])


# ------------------------------------------------------------------------------- bounded children and env


def process_alive(pid: int) -> bool:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return text.rsplit(")", 1)[1].split()[0] not in ("Z", "X")


def assert_reaped(case: unittest.TestCase, pid: int) -> None:
    with case.assertRaises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


@requires_linux
class BoundedProcessTests(RuntimeCase):
    def run_child(self, code: str, *args: str, timeout_s: float = 10.0, max_output_bytes: int = 1 << 16):
        return runtime.run_bounded([sys.executable, "-I", "-B", "-c", code, *args],
                                   env=runtime.worker_env(self.tmp / "run"), timeout_s=timeout_s,
                                   max_output_bytes=max_output_bytes)

    def test_success_and_exact_environment(self) -> None:
        result = self.run_child("import json, os; print(json.dumps(dict(os.environ)))")
        self.assertEqual(result.returncode, 0)
        seen = json.loads(result.stdout)
        seen.pop("LC_CTYPE", None)  # CPython locale coercion may add it
        self.assertEqual(seen, runtime.worker_env(self.tmp / "run"))
        assert_reaped(self, result.pid)

    def test_worker_env_is_cpu_only_and_run_owned(self) -> None:
        run = self.tmp / "run"
        env = runtime.worker_env(run)
        self.assertEqual(set(env), set(runtime_worker.STABLE_ENV_KEYS) | set(runtime_worker.OPERATIONAL_ENV_KEYS))
        self.assertEqual((env["CUDA_VISIBLE_DEVICES"], env["NUMBA_DISABLE_CUDA"], env["JAX_PLATFORMS"]),
                         ("", "1", "cpu"))
        for key in runtime_worker.OPERATIONAL_ENV_KEYS:
            self.assertTrue(env[key].startswith(str(run) + "/"), key)
        self.assertEqual({k: env[k] for k in runtime_worker.STABLE_ENV_KEYS},
                         {k: runtime.worker_env(self.tmp / "other")[k] for k in runtime_worker.STABLE_ENV_KEYS})

    def test_timeout_kills_and_reaps(self) -> None:
        start = time.monotonic()
        with self.assertRaises(RuntimeRefusal) as ctx:
            self.run_child("import time; time.sleep(30)", timeout_s=0.5)
        self.assertEqual(ctx.exception.code, "worker_timeout")
        self.assertLess(time.monotonic() - start, 10)
        assert_reaped(self, ctx.exception.evidence["pid"])

    def test_combined_stdout_stderr_cap(self) -> None:
        code = ("import sys, time\nsys.stdout.write('o' * 3000); sys.stdout.flush()\n"
                "sys.stderr.write('e' * 3000); sys.stderr.flush()\ntime.sleep(30)\n")
        start = time.monotonic()
        with self.assertRaises(RuntimeRefusal) as ctx:
            self.run_child(code, max_output_bytes=4000)  # each stream alone fits; together they do not
        self.assertEqual(ctx.exception.code, "worker_output_exceeded")
        self.assertLess(time.monotonic() - start, 10)
        out, err = ctx.exception.partial
        self.assertLessEqual(len(out) + len(err), 4000)
        assert_reaped(self, ctx.exception.evidence["pid"])

    def test_descendants_are_killed_with_the_group(self) -> None:
        pid_file = self.tmp / "grandchild.pid"
        code = ("import subprocess, sys\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                "open(sys.argv[1], 'w').write(str(child.pid))\n")
        result = self.run_child(code, str(pid_file), timeout_s=10.0)
        self.assertEqual(result.returncode, 0)
        assert_reaped(self, result.pid)
        grandchild = int(pid_file.read_text())
        deadline = time.monotonic() + 3.0
        while process_alive(grandchild) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(process_alive(grandchild), "a descendant survived the bounded run")


# ------------------------------------------------------------------ bootstrap and pure worker-side helpers


class WorkerHelperTests(RuntimeCase):
    def test_preload_violations_use_package_families(self) -> None:
        modules = {"maple": None, "maple.core": None, "maple_syrup": None, "masy.runtime": None,
                   "maple_syrupx": None, "maplex": None, "json": None}
        self.assertEqual(runtime_worker.preload_violations(modules),
                         ["maple", "maple.core", "maple_syrup", "masy.runtime"])

    def test_compiled_identities_are_bounded(self) -> None:
        extension = self.tmp / f"fixture_ext{importlib.machinery.EXTENSION_SUFFIXES[0]}"
        extension.write_bytes(b"synthetic extension bytes\n")
        library = self.tmp / "lib" / "libfixture.so"
        write(library, "synthetic library bytes\n")
        maps = write(self.tmp / "maps", f"7f00-7f01 r-xp 00000000 00:00 0 {library}\n"
                                        "7f01-7f02 r--p 00000000 00:00 0 /elsewhere/libother.so\n")
        extension_module = types.ModuleType("fixture_ext")
        extension_module.__file__ = str(extension)
        plain_module = types.ModuleType("plain")
        plain_module.__file__ = str(self.tmp / "plain.py")
        modules = {"fixture_ext": extension_module, "plain": plain_module, "builtin": None}

        def identities(**bounds):
            return runtime_worker.compiled_identities(modules, maps_path=str(maps),
                                                      library_roots=[str(self.tmp / "lib")], **bounds)

        records = identities(max_count=2, max_bytes=1000)
        self.assertEqual([(r["path"], r["kind"], r["sha256"]) for r in records], sorted([
            (str(extension), "extension_module", sha256(extension.read_bytes())),
            (str(library), "mapped_library", sha256(library.read_bytes()))]))
        for bounds in ({"max_count": 1, "max_bytes": 1000}, {"max_count": 2, "max_bytes": 30}):
            with self.subTest(bounds=bounds):
                with self.assertRaises(WorkerRefusal) as ctx:
                    identities(**bounds)
                self.assertEqual(ctx.exception.code, "compiled_identity_bound_exceeded")


@requires_linux
class BootstrapTests(RuntimeCase):
    """The literal BOOTSTRAP runs in a fresh -I -B child before any sealed worker is imported."""

    def run_bootstrap(self, request: dict, preamble: str = ""):
        path = self.tmp / "request.json"
        path.write_text(json.dumps(request))
        return runtime.run_bounded([sys.executable, "-I", "-B", "-c", preamble + runtime_worker.BOOTSTRAP, str(path)],
                                   env=runtime.worker_env(self.tmp / "run"), timeout_s=10.0,
                                   max_output_bytes=1 << 16)

    def test_preloaded_family_refuses_before_worker_import(self) -> None:
        sealed = self.tmp / "sealed"
        write(sealed / "masy/__init__.py", "raise SystemExit('the sealed worker must not be imported')\n")
        preamble = "import sys, types\nsys.modules['maple_syrup'] = types.ModuleType('maple_syrup')\n"
        result = self.run_bootstrap({"sys_path_prefix": [str(sealed)]}, preamble)
        self.assertEqual(result.returncode, 2, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual((record["status"], record["code"]), ("refused", "worker_preloaded_modules"))
        self.assertEqual(record["evidence"]["modules"], ["maple_syrup"])

    def test_invalid_prefix_refuses(self) -> None:
        for prefix in ([], ["relative/src"], "/abs"):
            with self.subTest(prefix=prefix):
                result = self.run_bootstrap({"sys_path_prefix": prefix})
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(json.loads(result.stdout)["code"], "worker_request_invalid")

    def test_exact_prefix_is_inserted_before_worker_import(self) -> None:
        sealed = self.tmp / "sealed"
        write(sealed / "masy/__init__.py", "")
        write(sealed / "masy/runtime_worker.py", (
            "import json, sys\n"
            "def main(request, base_sys_path):\n"
            "    sys.stdout.write(json.dumps({'head': sys.path[:1], 'file': __file__,\n"
            "                                 'in_base': request['sys_path_prefix'][0] in base_sys_path}) + '\\n')\n"
            "    return 0\n"))
        result = self.run_bootstrap({"sys_path_prefix": [str(sealed)]})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"head": [str(sealed)],
                                                     "file": str(sealed / "masy/runtime_worker.py"),
                                                     "in_base": False})


if __name__ == "__main__":
    unittest.main()
