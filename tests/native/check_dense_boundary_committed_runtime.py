"""Closed committed-binding protocols; metadata/unit delegates, no model run.

MASY91_CPU_LOCK names an explicitly admitted canonical lock metadata file. It
supplies a real format example, never a positive provenance claim for this new
candidate. No test constructs a bed, materializes a bundle, or starts a child.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from masy import dense_boundary_store as c
from masy import runtime


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def object_metadata(kind="syrup", *, extra_rows=(), overlay=None):
    """Illustrative metadata only; these bytes are not a source object on disk."""
    package, project_name = runtime.SOURCES[kind]
    rows = [
        ["pyproject.toml", 3, "a" * 64, "b" * 40],
        [f"src/{package}/__init__.py", 5, "c" * 64, "d" * 40],
        *copy.deepcopy(list(extra_rows)),
    ]
    if kind == "masy":
        rows.append([runtime.LAUNCHER_RELPATH, 7, "e" * 64, "f" * 40])
    identity = runtime.object_identity(
        kind, object_format="sha1", files=rows,
        project={"name": project_name, "version": "0.1", "pyproject_sha256": "a" * 64},
    )
    if overlay is None and identity["overlay"] is not None:
        overlay = [[identity["overlay"]["dist_info"] + "/METADATA", 4, "e" * 64]]
    record = {
        "schema": runtime.SOURCE_OBJECT_SCHEMA,
        "object_id": runtime.identity_id(identity),
        "identity": identity,
        "overlay_files": overlay,
    }
    return record


def admitted(record):
    return c._admit_source_object(
        runtime._canonical_bytes(record), kind=record["identity"]["kind"],
        object_id=record["object_id"],
    )


class CommittedBindingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ["MASY91_CPU_LOCK"])
        if path.stat().st_size > c._CL_LOCK_BYTES:
            raise AssertionError("test lock raw admission")
        cls.raw_lock = path.read_bytes()
        cls.lock = json.loads(cls.raw_lock)
        cls.reference = c.committed_runtime_reference(
            cpu_store=cls.lock["store"], cpu_lock_id=cls.lock["lock_id"],
            cpu_lock_file_sha256=sha(cls.raw_lock), lock=cls.lock,
        )

    def refusal(self, code, call):
        with self.assertRaises(c.BoundaryStoreError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_reference_is_pure_and_keeps_null_manifest_slots(self):
        with mock.patch("builtins.open", side_effect=AssertionError("IO")), \
             mock.patch.object(os, "stat", side_effect=AssertionError("stat")), \
             mock.patch.object(os, "lstat", side_effect=AssertionError("lstat")), \
             mock.patch.object(os.path, "realpath", side_effect=AssertionError("realpath")), \
             mock.patch.object(runtime, "_hash_path", side_effect=AssertionError("hash")):
            result = c.committed_runtime_reference(
                cpu_store=self.lock["store"], cpu_lock_id=self.lock["lock_id"],
                cpu_lock_file_sha256=sha(self.raw_lock), lock=self.lock,
            )
        self.assertEqual(result, self.reference)
        self.assertEqual(set(result), set(c._RUNTIME))
        self.assertIsNone(result["source_manifest_path"])
        self.assertIsNone(result["source_manifest_file_sha256"])
        self.assertEqual(result["module_roots"]["masy"], self.lock["runtime"]["sys_path_prefix"][3])

    def test_reference_wrong_raw_sha_refuses(self):
        self.refusal("runtime_reference", lambda: c.committed_runtime_reference(
            cpu_store=self.lock["store"], cpu_lock_id=self.lock["lock_id"],
            cpu_lock_file_sha256="0" * 64, lock=self.lock,
        ))

    def test_reference_wrong_id_and_lexical_store_refuse(self):
        for store, lock_id in ((self.lock["store"], "0" * 64), ("/a/../b", self.lock["lock_id"])):
            with self.subTest(store=store, lock_id=lock_id), self.assertRaises(c.BoundaryStoreError):
                c.committed_runtime_reference(
                    cpu_store=store, cpu_lock_id=lock_id,
                    cpu_lock_file_sha256=sha(self.raw_lock), lock=self.lock,
                )

    def test_reference_rejects_nonplain_metadata_before_native_validator(self):
        value = copy.deepcopy(self.lock)
        value["extra"] = object()
        with mock.patch.object(runtime, "validate_lock", side_effect=AssertionError("validator")):
            self.refusal("metadata_invalid", lambda: c.committed_runtime_reference(
                cpu_store=self.lock["store"], cpu_lock_id=self.lock["lock_id"],
                cpu_lock_file_sha256=sha(self.raw_lock), lock=value,
            ))

    def test_reference_metadata_cycle_and_wire_cap(self):
        cycle = []
        cycle.append(cycle)
        self.refusal("metadata_invalid", lambda: c._plain_metadata(cycle, 1024, "unit"))
        self.refusal("metadata_cap", lambda: c._plain_metadata(["x" * 1024], 100, "unit"))
        self.refusal("metadata_cap", lambda: c._plain_metadata([1, 2], 100, "unit", max_nodes=2))

    def test_source_maple_none_is_distinct_from_present_overlay(self):
        record = object_metadata("maple")
        got, _rows, overlay, size = admitted(record)
        self.assertIsNone(got["overlay_files"])
        self.assertEqual(overlay, [])
        self.assertEqual(size, 8)
        record["overlay_files"] = []
        self.refusal("source_object", lambda: admitted(record))

    def test_source_overlay_plain_size_sorted_unique_and_namespace(self):
        info = object_metadata()["identity"]["overlay"]["dist_info"]
        error = self.refusal("source_object", lambda: c._overlay_path(info + "/M\ud800", info))
        self.assertEqual(error.detail, "unsafe overlay path")
        for change in ("bool", "negative", "duplicate", "escape", "depth", "huge", "surrogate"):
            record = object_metadata()
            row = record["overlay_files"][0]
            if change == "bool": row[1] = True
            elif change == "negative": row[1] = -1
            elif change == "duplicate": record["overlay_files"].append(copy.deepcopy(row))
            elif change == "escape": row[0] = "../METADATA"
            elif change == "depth": row[0] = record["identity"]["overlay"]["dist_info"] + "/" + "a/" * 16 + "METADATA"
            elif change == "huge": row[1] = (1 << 20) + 1
            elif change == "surrogate": row[0] += "\ud800"
            with self.subTest(change=change), self.assertRaises(c.BoundaryStoreError):
                admitted(record)

    def test_source_caps_and_unsafe_scope_refuse_before_file_delegates(self):
        records = []
        records.append(object_metadata(extra_rows=[["src/maple_syrup/huge.py", (16 << 20) + 1, "a" * 64, "b" * 40]]))
        records.append(object_metadata(extra_rows=[["other/input.py", 1, "a" * 64, "b" * 40]]))
        records.append(object_metadata(extra_rows=[["src/" + "a/" * 16 + "x.py", 1, "a" * 64, "b" * 40]]))
        with mock.patch.object(runtime, "_hash_path", side_effect=AssertionError("full hash")), \
             mock.patch.object(runtime, "verify_source_object", side_effect=AssertionError("full verify")):
            for record in records:
                with self.subTest(files=record["identity"]["files"][-1]), self.assertRaises(c.BoundaryStoreError):
                    admitted(record)
            record = object_metadata()
            self.refusal("metadata_cap", lambda: c._admit_source_object(
                runtime._canonical_bytes(record), kind="syrup", object_id=record["object_id"], max_bytes=8,
            ))

    def test_source_closed_canonical_identity_refusals(self):
        record = object_metadata()
        raw = runtime._canonical_bytes(record)
        self.refusal("metadata_noncanonical", lambda: c._admit_source_object(
            raw + b" ", kind="syrup", object_id=record["object_id"],
        ))
        self.refusal("source_object", lambda: c._admit_source_object(
            raw, kind="masy", object_id=record["object_id"],
        ))
        record["extra"] = None
        self.refusal("metadata_layout", lambda: admitted(record))

    def test_inventory_raw_cap_precedes_byte_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.json"
            path.write_bytes(b"x" * 16)
            path.chmod(0o444)
            with c._HeldInventory(1024) as held, mock.patch.object(c, "_read", side_effect=AssertionError("read")):
                self.refusal("metadata_cap", lambda: held.read(str(path), 8))

    def test_inventory_type_mode_nlink_and_exact_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regular = root / "regular"
            regular.write_bytes(b"one")
            with c._HeldInventory(1024) as held:
                self.refusal("inventory_file", lambda: held.admit(str(regular), 3))
            regular.chmod(0o444)
            for name, create in (
                ("symlink", lambda p: p.symlink_to(regular)),
                ("directory", lambda p: p.mkdir()),
                ("hardlink", lambda p: os.link(regular, p)),
            ):
                path = root / name
                create(path)
                with self.subTest(name=name), c._HeldInventory(1024) as held:
                    self.refusal("inventory_file", lambda path=path: held.admit(str(path), 3))
                if name == "hardlink": path.unlink()
            with c._HeldInventory(1024) as held:
                self.refusal("inventory_file", lambda: held.admit(str(regular), 2))

    def test_inventory_shared_roles_charge_once_and_replacement_refuses(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "owned"
            path.write_bytes(b"one")
            path.chmod(0o444)
            with c._HeldInventory(3) as held:
                held.admit(str(path), 3)
                held.admit(str(path), 3)
                self.assertEqual(held._bytes, 3)
                self.assertEqual(held.read(str(path), 3), b"one")
                self.refusal("inventory_conflict", lambda: held.admit(str(path), 2))
                replacement = path.with_name("replacement")
                replacement.write_bytes(b"two")
                replacement.chmod(0o444)
                os.replace(replacement, path)
                self.refusal("identity_drift", held.check)

    def test_inventory_finite_caps_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "owned"
            path.write_bytes(b"one")
            path.chmod(0o444)
            for budget, kwargs in ((3, {"max_files": 0}), (3, {"max_parents": 0}), (2, {})):
                with c._HeldInventory(budget, **kwargs) as held:
                    self.refusal("inventory_cap", lambda: held.admit(str(path), 3))
            held = c._HeldInventory(3)
            descriptors = []
            with self.assertRaisesRegex(RuntimeError, "unit failure"), held:
                held.admit(str(path), 3)
                descriptors = [fd for fd, _identity in held._parents.values()]
                raise RuntimeError("unit failure")
            self.assertFalse(held._parents)
            for descriptor in descriptors:
                with self.assertRaises(OSError):
                    os.fstat(descriptor)
            self.refusal("inventory_closed", held.check)

    def test_existing_unknown_kind_and_prototype_cpu_slot_refusals(self):
        reference = copy.deepcopy(self.reference)
        reference["kind"] = "unknown"
        error = self.refusal("runtime_kind_unsupported_v1", lambda: c._runtime_binding(reference, c._DEFAULT))
        self.assertEqual(error.detail, "v1 has no committed-lock success branch")
        reference["kind"] = "source_manifest_prototype"
        with mock.patch.object(c, "_read", side_effect=AssertionError("read")):
            self.refusal("runtime_reference", lambda: c._runtime_binding(reference, c._DEFAULT))

    def test_committed_shape_refuses_before_process_and_read(self):
        for field, value in (("source_manifest_path", "/unexpected"), ("cpu_lock_id", "bad"), ("cpu_store", "/a/../b")):
            reference = copy.deepcopy(self.reference)
            reference[field] = value
            with self.subTest(field=field), \
                 mock.patch.object(c, "_committed_cpu_flags", side_effect=AssertionError("flags")), \
                 mock.patch.object(c, "_read", side_effect=AssertionError("read")), \
                 self.assertRaises(c.BoundaryStoreError):
                c._runtime_binding(reference, c._DEFAULT)

    def test_committed_lock_sha_refuses_before_loader_and_full_delegates(self):
        reference = copy.deepcopy(self.reference)
        reference["cpu_lock_file_sha256"] = "0" * 64
        with mock.patch.object(c, "_committed_cpu_flags", return_value={}), \
             mock.patch.object(c._HeldInventory, "read", return_value=self.raw_lock), \
             mock.patch.object(runtime, "load_lock_ref", side_effect=AssertionError("loader")), \
             mock.patch.object(runtime, "verify_source_object", side_effect=AssertionError("verify")):
            self.refusal("committed_lock", lambda: c._runtime_binding(reference, c._DEFAULT))

    def test_cpu_flag_and_environment_refusals_are_predelegate(self):
        flags = SimpleNamespace(isolated=1, dont_write_bytecode=0, ignore_environment=1,
                                no_user_site=1, safe_path=True, utf8_mode=1)
        with mock.patch.object(c.sys, "flags", flags):
            self.refusal("committed_runtime", c._committed_cpu_flags)
        flags.dont_write_bytecode = 1
        with mock.patch.object(c.sys, "flags", flags), mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            self.refusal("committed_runtime", c._committed_cpu_flags)

    def test_actual_process_prefix_and_locked_record_mismatch(self):
        with mock.patch.object(c.sys, "path", []):
            self.refusal("committed_runtime", lambda: c._committed_process(self.lock, {}))
        lock = copy.deepcopy(self.lock)
        lock["verification"]["process"]["machine"] = "different"
        flags = self.lock["verification"]["process"]["flags"]
        stable = self.lock["runtime"]["stable_environment"]
        present = {key: value for key, value in stable.items() if value is not None}
        with mock.patch.object(c.sys, "path", list(lock["runtime"]["sys_path_prefix"])), \
             mock.patch.dict(os.environ, present):
            for key, value in stable.items():
                if value is None:
                    os.environ.pop(key, None)
            # Unit prefix injection only, never an end-to-end provenance claim.
            self.assertEqual(
                c._committed_process(self.lock, flags), self.lock["verification"]["process"]
            )
            error = self.refusal("committed_runtime", lambda: c._committed_process(lock, flags))
            self.assertEqual(error.detail, "current stable process record differs from lock")

    def test_four_summary_fields_are_independently_required(self):
        lock = copy.deepcopy(self.lock)
        sources = {}
        for kind in ("maple", "syrup", "masy"):
            record = object_metadata(kind)
            sources[kind] = admitted(record)[:3]
        spec = lock["objects"]["sources"]["maple"]
        valid = {k: spec[k] for k in ("file_count", "total_bytes", "manifest_sha256", "package_digest")}
        valid["overlay_files"] = None
        for field in ("file_count", "total_bytes", "manifest_sha256", "package_digest"):
            summary = {**valid, field: "different"}
            with self.subTest(field=field), \
                 mock.patch.object(runtime, "verify_source_object", return_value=summary):
                error = self.refusal("source_identity", lambda: c._committed_verify_sources(
                    lock, sources, SimpleNamespace(check=lambda: None), c._CL_OBJECT_BYTES,
                ))
                self.assertEqual(error.detail, "maple: four source summary fields differ")

    def native_metadata_unit(self, *, lock=None, reference=None, admitted_change=None):
        """Real native pure validator/generator; inventory is a labelled unit seam.

        Only lock-referenced small control/distribution metadata is read. The
        native archive/source files are neither opened nor physically verified.
        """
        from maple_syrup import dependency_bundle as native

        base_lock = self.lock
        if lock is None:
            lock = copy.deepcopy(base_lock)
        if reference is None:
            reference = copy.deepcopy(self.reference)
        sources = {}
        for kind in ("maple", "syrup", "masy"):
            spec = base_lock["objects"]["sources"][kind]
            raw = c._read(Path(spec["path"]) / "manifest.json", c._CL_OBJECT_BYTES)
            self.assertEqual(sha(raw), spec["manifest_sha256"])
            sources[kind] = c._admit_source_object(
                raw, kind=kind, object_id=spec["object_id"],
            )[:3]
        if admitted_change is not None:
            admitted_change(sources)
        bundle = Path(base_lock["objects"]["maple_bundle"]["path"])
        target = Path(base_lock["objects"]["maple_materialized"]["path"])
        native_raw = c._read(bundle / native.MANIFEST_NAME, c._CL_LOCK_BYTES)
        self.assertEqual(sha(native_raw), base_lock["objects"]["maple_bundle"]["manifest_sha256"])
        record = json.loads(native_raw)
        controls = {
            str(bundle / native.MANIFEST_NAME): native_raw,
            str(target / native.MANIFEST_NAME): c._read(target / native.MANIFEST_NAME, c._CL_LOCK_BYTES),
            str(target / native.MATERIALIZATION_NAME): c._read(target / native.MATERIALIZATION_NAME, 64 << 10),
        }
        for name in record["distribution"]["files"]:
            controls[str(target / "distribution" / name)] = c._read(target / "distribution" / name, 1 << 20)
        admissions, reads = [], []

        def read(path, cap, declared_size=None):
            data = controls[path]  # finite literal control set; never arbitrary paths
            self.assertLessEqual(len(data), cap)
            if declared_size is not None:
                self.assertEqual(len(data), declared_size)
            reads.append(path)
            return data

        held = SimpleNamespace(read=read, admit=lambda path, size: admissions.append((path, size)))
        with mock.patch.object(native, "verify_bundle", side_effect=AssertionError("full native verify")), \
             mock.patch.object(native, "check_runtime", side_effect=AssertionError("native resolver")):
            result = c._committed_native(
                lock, reference, sources, held, c._CL_LOCK_BYTES, native,
            )
        return result, admissions, reads

    def test_native_metadata_unit_reuses_real_validator_and_overlay_limits(self):
        (record, limits, controls), admissions, reads = self.native_metadata_unit()
        self.assertEqual(record["bundle_id"], self.lock["objects"]["maple_bundle"]["bundle_id"])
        self.assertEqual((limits.max_files, limits.max_file_bytes, limits.max_total_bytes), (4000, 16 << 20, 64 << 20))
        self.assertEqual(limits.max_origin_bytes, 64 << 10)
        self.assertEqual(len(controls), 2)
        self.assertIn((str(Path(self.lock["objects"]["maple_bundle"]["path"]) / "maple-project.tar"), record["archive"]["size"]), admissions)
        self.assertTrue(all("/distribution/" in p or p.endswith(("bundle-manifest.json", "materialization.json")) for p in reads))

    def test_native_metadata_unit_each_crosslink_refuses_at_own_predicate(self):
        for owner, field in (("maple_bundle", "archive_sha256"), ("maple_bundle", "tool_module_sha256"),
                             ("maple_bundle", "package_digest"), ("maple_source", "package_digest")):
            lock = copy.deepcopy(self.lock)
            spec = lock["objects"]["sources"]["maple"] if owner == "maple_source" else lock["objects"][owner]
            spec[field] = "0" * 64
            with self.subTest(owner=owner, field=field):
                error = self.refusal("native_binding", lambda lock=lock: self.native_metadata_unit(lock=lock))
                self.assertEqual(error.detail, "native source/tool/archive/package cross-links differ")
        reference = copy.deepcopy(self.reference)
        reference["native_bundle_manifest_file_sha256"] = "0" * 64
        error = self.refusal("native_binding", lambda: self.native_metadata_unit(reference=reference))
        self.assertEqual(error.detail, "native bundle/materialized manifest pins differ")

    def identity_unit(self, *, origin=None, python=None, bootstrap="unit", environment=None):
        """Isolated helper wiring, deliberately not end-to-end committed binding."""
        lock = copy.deepcopy(self.lock)
        lock["runtime"]["bootstrap_sha256"] = sha(b"unit")
        actual_python = {**lock["runtime"]["python"], "scope": "unit"}
        if python is not None:
            actual_python.update(python)
        executable = lock["runtime"]["python"]
        if environment is None:
            environment = {"interpreter_identity": {
                "path": executable["realpath"], "size": executable["size"],
                "sha256": executable["sha256"],
            }, "unit_only": True}
        sources = {kind: admitted(object_metadata(kind))[:3] for kind in ("maple", "syrup", "masy")}
        module = SimpleNamespace(__file__=origin or self.reference["module_roots"]["masy"] + "/masy/__init__.py")
        with mock.patch.object(c, "_committed_process"), \
             mock.patch.object(runtime, "interpreter_identity", return_value=actual_python), \
             mock.patch.object(runtime, "sealed_bootstrap", return_value=bootstrap), \
             mock.patch.object(runtime, "check_import_roots"), \
             mock.patch.object(c.rw, "selected_distributions") as selected, \
             mock.patch.object(c, "_environment_identity", return_value=environment), \
             mock.patch.object(c.sys, "modules", {"masy": module}):
            result = c._committed_identity(lock, sources, self.reference["module_roots"], {})
        return result, selected.call_args, sources, environment

    def test_identity_unit_keeps_legacy_tuple_and_real_resolver_contract(self):
        (ids, environment_sha), selected, sources, environment = self.identity_unit()
        self.assertEqual(ids["kind"], "source_manifest_prototype")
        self.assertEqual(set(ids), {"kind", "masy", "maple", "syrup"})
        self.assertEqual(ids["maple"], self.lock["objects"]["maple_bundle"]["bundle_id"])
        table = {p: {"size_bytes": size, "sha256": digest} for p, size, digest, _blob in sources["masy"][1] if p.startswith("src/") or p == "pyproject.toml"}
        self.assertEqual(ids["masy"], sha(c.oi._json(table)))
        self.assertEqual(environment_sha, sha(c.oi._json(environment)))
        self.assertIsNone(selected.kwargs["path"])
        self.assertEqual([s["name"] for s in selected.args[0]], ["maple", "maple-syrup", "masy"])
        for spec in selected.args[0]:
            self.assertEqual(set(spec), {"name", "dir", "files"})
            self.assertEqual(spec["dir"], self.lock["distributions"][spec["name"]]["overlay_dir"])

    def test_identity_unit_interpreter_bootstrap_environment_and_origin_refuse(self):
        self.refusal("committed_runtime", lambda: self.identity_unit(python={"sha256": "0" * 64}))
        error = self.refusal("committed_runtime", lambda: self.identity_unit(bootstrap="different"))
        self.assertEqual(error.detail, "sealed bootstrap SHA differs from lock")
        self.refusal("environment_identity", lambda: self.identity_unit(environment={"interpreter_identity": {"path": "wrong", "size": 0, "sha256": "0" * 64}}))
        self.refusal("module_origin", lambda: self.identity_unit(origin="/tmp/foreign/masy.py"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
