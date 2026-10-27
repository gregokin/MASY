"""Pure stdlib controls for the current-pair restart composition.

These checks execute selected real guard/metadata bodies extracted from Source.
Synthetic holders exercise refusal and local lifetime only; no MASY/Native module,
physical constructor, codec, provider or transport is imported or executed.
They do not establish actual runtime, ABI, storage, physical restart or fit facts.
"""
from __future__ import annotations

import ast
import __future__
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest


_SOURCE = Path(__file__).resolve().parents[1] / "src" / "masy"


class _Refusal(RuntimeError):
    def __init__(self, code, detail):
        super().__init__(code, detail)
        self.code, self.detail = code, detail


def _fail(code, detail):
    raise _Refusal(code, detail)


def _tree(name):
    return ast.parse((_SOURCE / name).read_text(encoding="utf-8"), filename=name)


def _node(name, key):
    nodes = _tree(name).body
    for part in key.split("."):
        item = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part), None)
        if item is None:
            raise AssertionError("missing Source symbol " + key)
        nodes = item.body
    return copy.deepcopy(item)


def _literal(name, key):
    for item in _tree(name).body:
        if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == key for t in item.targets):
            return ast.literal_eval(item.value)
    raise AssertionError("missing literal " + key)


def _load(name, keys, namespace):
    # Preserve the owning Source's compiler semantics without executing its
    # imports or fabricating annotation names in the synthetic namespace.
    flags = 0
    for item in _tree(name).body:
        if isinstance(item, ast.ImportFrom) and item.module == "__future__":
            for feature in item.names:
                flags |= getattr(__future__, feature.name).compiler_flag
    module = ast.Module(body=[_node(name, key) for key in keys], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, name, "exec", flags=flags, dont_inherit=True), namespace)
    return namespace


def _original_json():
    env = {"json": json, "math": math, "_fail": _fail,
           "OriginalInputsError": _Refusal, "_CANONICAL_METADATA_SCOPE": None}
    _load("dense_original_inputs.py", ("_json", "_strict_json"), env)
    return SimpleNamespace(_json=env["_json"], _strict_json=env["_strict_json"])


def _validation_namespace():
    # No synthetic holder purports to be a real store or physical model.
    holder = type("SyntheticMetadataHandle", (), {})
    env = {"DenseBoundaryStore": holder, "_fail": _fail}
    for key in ("SCHEMA_ACTIVE_WATER", "PROFILE_ACTIVE_WATER", "_CL_KIND"):
        env[key] = _literal("dense_boundary_store.py", key)
    _load("dense_boundary_store.py", ("_active_water_store", "_new_runtime_validation"), env)
    return holder, env


def _parent_namespace():
    env = {"_fail": _fail, "_WATER_PARENT_KEY": object(), "os": os,
           "threading": threading, "sys": SimpleNamespace(modules={})}
    _load("dense_boundary_store.py", ("_WaterParent",), env)
    return env


class TestCurrentPairRestartComposition(unittest.TestCase):
    def test_ordinary_profiles_return_before_new_private_attribute(self):
        holder, env = _validation_namespace()
        headers = [None, {},
                   {"schema": env["SCHEMA_ACTIVE_WATER"], "profile": "different"},
                   {"schema": env["SCHEMA_ACTIVE_WATER"], "profile": env["PROFILE_ACTIVE_WATER"],
                    "runtime_reference": {"kind": "source_manifest_prototype"}}]
        for header in headers:
            with self.subTest(header=header):
                handle = holder()
                handle._header = header
                self.assertNotIn("_runtime_validation", vars(handle))
                self.assertIsNone(env["_new_runtime_validation"](handle))
                self.assertEqual(vars(handle), {"_header": header})

    def test_active_validation_refuses_nested_or_not_entered_owner(self):
        holder, env = _validation_namespace()
        for nested, entered in ((object(), True), (None, False)):
            with self.subTest(nested=nested is not None, entered=entered):
                handle = holder()
                handle._header = {"schema": env["SCHEMA_ACTIVE_WATER"],
                    "profile": env["PROFILE_ACTIVE_WATER"], "runtime_reference": {"kind": env["_CL_KIND"]}}
                handle._runtime_validation, handle._entered = nested, entered
                with self.assertRaises(_Refusal) as error:
                    env["_new_runtime_validation"](handle)
                self.assertEqual(error.exception.code, "runtime_scope")
                self.assertIs(handle._runtime_validation, nested)
        with self.assertRaises(_Refusal) as error:
            env["_new_runtime_validation"](object())
        self.assertEqual(error.exception.code, "runtime_scope")

    def test_complete_source_pair_checks_both_tables_and_subject(self):
        oi = _original_json()
        imports = []
        def source_import(name, globals=None, locals=None, fromlist=(), level=0):
            imports.append((name, tuple(fromlist), level))
            if (name, tuple(fromlist), level) != ("masy", ("dense_original_inputs",), 0):
                raise AssertionError("unexpected import")
            return SimpleNamespace(dense_original_inputs=oi)
        builtin = dict(vars(__import__("builtins")))
        builtin["__import__"] = source_import
        native = {"src/maple/example.py": {"size_bytes": 3, "sha256": "a" * 64}}
        syrup = {"src/maple_syrup/example.py": {"size_bytes": 4, "sha256": "b" * 64}}
        source = {"kind": "source_manifest_prototype", "masy": "c" * 64,
                  "maple": "d" * 64, "syrup": "e" * 64}
        manifest = {"source_identity": source, "files": {"maple": native, "maple_syrup": syrup}}
        env = {"__builtins__": builtin, "hashlib": hashlib,
               "_REVIEWED_CURRENT_NATIVE_TABLE_937": hashlib.sha256(oi._json(native)).hexdigest(),
               "_REVIEWED_CURRENT_SYRUP_TABLE_937": hashlib.sha256(oi._json(syrup)).hexdigest()}
        _load("dense_wind_pause.py", ("_reviewed_current_source_pair",), env)
        check = env["_reviewed_current_source_pair"]
        self.assertTrue(check(manifest, source))  # predicate only, not runtime admission
        for package in ("maple", "maple_syrup"):
            changed = copy.deepcopy(manifest)
            next(iter(changed["files"][package].values()))["size_bytes"] += 1
            self.assertFalse(check(changed, source))
        self.assertFalse(check(manifest, {**source, "masy": "f" * 64}))
        self.assertTrue(imports)
        self.assertTrue(all(row == ("masy", ("dense_original_inputs",), 0) for row in imports))

    def test_unknown_or_malformed_source_pins_refuse(self):
        oi, imports = _original_json(), []
        def source_import(name, globals=None, locals=None, fromlist=(), level=0):
            imports.append((name, tuple(fromlist), level))
            if (name, tuple(fromlist), level) != ("masy", ("dense_original_inputs",), 0):
                raise AssertionError("unexpected import")
            return SimpleNamespace(dense_original_inputs=oi)
        builtin = dict(vars(__import__("builtins"))); builtin["__import__"] = source_import
        tables = {"maple": {"src/maple/example.py": {"size_bytes": 3, "sha256": "a" * 64}},
                  "maple_syrup": {"src/maple_syrup/example.py": {"size_bytes": 4, "sha256": "b" * 64}}}
        source = {"kind": "source_manifest_prototype", "masy": "c" * 64,
                  "maple": "d" * 64, "syrup": "e" * 64}
        manifest = {"source_identity": source, "files": tables}
        genuine = {"_REVIEWED_CURRENT_NATIVE_TABLE_937": hashlib.sha256(oi._json(tables["maple"])).hexdigest(),
                   "_REVIEWED_CURRENT_SYRUP_TABLE_937": hashlib.sha256(oi._json(tables["maple_syrup"])).hexdigest()}
        env = {"__builtins__": builtin, "hashlib": hashlib, **genuine}
        _load("dense_wind_pause.py", ("_reviewed_current_source_pair",), env)
        for key in genuine:
            for bad in (None, True, "a" * 63, "A" * 64, "z" * 64):
                with self.subTest(side=key, pin=bad):
                    env.update(genuine)
                    self.assertTrue(env["_reviewed_current_source_pair"](manifest, source))
                    env[key] = bad
                    self.assertFalse(env["_reviewed_current_source_pair"](manifest, source))
        self.assertTrue(imports)
        self.assertTrue(all(row == ("masy", ("dense_original_inputs",), 0) for row in imports))
        self.assertEqual(_literal("dense_wind_pause.py", "_REVIEWED_CURRENT_NATIVE_TABLE_937"),
                         "ae7a5ee454ca63d365190e78fbd3349c8c5f5ce14b970ae2d2ffcd976a82a706")
        self.assertEqual(_literal("dense_wind_pause.py", "_REVIEWED_CURRENT_SYRUP_TABLE_937"),
                         "c87b75a81bfec10274163f7348dcd812305986c29c024f5cdfd89c0a2daab098")
    def test_water_mismatch_refuses_without_importing_active_owner(self):
        oi, imports = _original_json(), []
        def source_import(name, globals=None, locals=None, fromlist=(), level=0):
            imports.append((name, tuple(fromlist), level))
            if (name, tuple(fromlist), level) != ("masy", ("dense_original_inputs",), 0):
                raise AssertionError("ordinary mismatch introduced a new import")
            return SimpleNamespace(dense_original_inputs=oi)
        builtin = dict(vars(__import__("builtins"))); builtin["__import__"] = source_import
        active_env = {"_MONITORED_RESTART_SLOT": None}
        _load("dense_active_water_store.py", ("_monitored_runtime_matches",), active_env)
        for modules in ({}, {"masy.dense_active_water_store": SimpleNamespace(
                _monitored_runtime_matches=active_env["_monitored_runtime_matches"])}):
            with self.subTest(already_loaded=bool(modules)):
                env = {"__builtins__": builtin, "_fail": _fail, "sys": SimpleNamespace(modules=modules),
                       "_NATIVE_CENSUS_MAPLE": "expected_native", "_NATIVE_CENSUS_SYRUP": "expected_syrup"}
                _load("dense_water_pause.py", ("_runtime",), env)
                store = SimpleNamespace(_plain_metadata=lambda *a, **k: None,
                    _runtime_binding=lambda *a: ({"maple": "wrong", "syrup": "wrong"}, "environment"),
                    BoundaryStoreLimits=lambda: object())
                with self.assertRaises(_Refusal) as error:
                    env["_runtime"](store, {})
                self.assertEqual(error.exception.code, "water_resource_source")
        self.assertTrue(imports)
        self.assertTrue(all(row == ("masy", ("dense_original_inputs",), 0) for row in imports))

    def test_parent_slot_and_never_entered_close_preserve_private_lifecycle(self):
        env = _parent_namespace(); cls = env["_WaterParent"]
        parent = cls(env["_WATER_PARENT_KEY"], object(), "token", b"head", b"source", "environment")
        self.assertIn("_selection", cls.__slots__)
        self.assertIsNone(parent._selection)
        self.assertFalse(hasattr(parent, "__dict__"))
        with self.assertRaises(_Refusal) as error:
            cls(object(), object(), "token", b"head", b"source", "environment")
        self.assertEqual(error.exception.code, "private_carrier")
        parent.close()
        self.assertEqual(parent._phase, "closed")
        self.assertEqual(parent.cleanup_record, ("never_entered", None))
        self.assertIsNone(parent._handle)
        parent.close()
        self.assertEqual(parent.cleanup_record, ("never_entered", None))
        with self.assertRaises(_Refusal):
            parent.__reduce_ex__(4)

    def test_active_parent_release_retains_outer_handle_guard(self):
        env = _parent_namespace(); active = {}
        _load("dense_active_water_store.py", ("_release_parent",), active)
        env["_active_module"] = lambda: SimpleNamespace(_release_parent=active["_release_parent"])
        handle = SimpleNamespace(_entered=True)
        parent = env["_WaterParent"](env["_WATER_PARENT_KEY"], handle, "token", b"head", b"source", "environment")
        parent._selection, parent._phase = object(), "entered"
        parent._release()
        self.assertEqual(parent._phase, "closed")
        self.assertIsNone(parent._selection)
        self.assertIsNone(parent._handle)
        self.assertTrue(handle._entered)

    def test_case_and_adopted_carriers_refuse_before_store_work(self):
        function = _node("dense_active_water_store.py", "_create_active_store")
        guards = [n for n in function.body if isinstance(n, ast.If)
                  and ("type(runner)" in ast.unparse(n.test) or "_case_policy" in ast.unparse(n.test))]
        self.assertEqual(len(guards), 2)
        first_work = next(i for i, n in enumerate(function.body)
                          if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                          and isinstance(n.value.func, ast.Name) and n.value.func.id == "_reservation")
        self.assertTrue(all(function.body.index(n) < first_work for n in guards))
        runner_type = type("SyntheticMetadataRunner", (), {})
        c = SimpleNamespace(dr=SimpleNamespace(DenseRunner=runner_type), _fail=_fail)
        module = ast.Module(body=guards, type_ignores=[]); ast.fix_missing_locations(module)
        compiled = compile(module, "actual_create_profile_guards", "exec", dont_inherit=True)
        for mode in ("wrong_type", "case_policy", "adopted"):
            with self.subTest(mode=mode):
                runner = object() if mode == "wrong_type" else runner_type()
                if mode != "wrong_type": runner._bed = SimpleNamespace()
                if mode == "case_policy": runner._case_policy = None
                if mode == "adopted": runner._bed._adopted_case_inputs = None
                with self.assertRaises(_Refusal) as error:
                    exec(compiled, {"c": c, "runner": runner})
                self.assertEqual(error.exception.code, "runner_profile")

    def test_actual_legacy_terminal_discriminator_keeps_default_lane(self):
        tree = _tree("dense_runner.py")
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DenseRunner")
        self.assertFalse(cls.bases)
        self.assertFalse(any(isinstance(n, ast.FunctionDef) and n.name in ("__getattr__", "__getattribute__") for n in cls.body))
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == "_case_policy" and isinstance(n.ctx, ast.Store) for n in ast.walk(tree)))
        terminal = _node("dense_runner.py", "_wind_terminal")
        condition = ast.parse("hasattr(self, '_case_policy')", mode="eval").body
        branches = [n for n in terminal.body if isinstance(n, ast.If)
                    and ast.dump(n.test) == ast.dump(condition)]
        self.assertEqual(len(branches), 1)
        chunk = ast.Module(body=branches, type_ignores=[]); ast.fix_missing_locations(chunk)
        body = {"native": {}}
        scope = {"self": SimpleNamespace(), "case_summary": None, "body": body}
        exec(compile(chunk, "actual_legacy_terminal_branch", "exec", dont_inherit=True), scope)
        self.assertEqual(body, {"native": {}})
        scope = {"self": SimpleNamespace(_case_policy=None), "case_summary": None,
                 "index": 0, "elapsed": 0.0, "evidence": {}, "_Stop": lambda *a, **k: _Refusal(a[1], a[2])}
        with self.assertRaises(_Refusal) as error:
            exec(compile(chunk, "actual_case_terminal_branch", "exec", dont_inherit=True), scope)
        self.assertEqual(error.exception.code, "case_summary_identity")

    def test_active_cleanup_refuses_live_foreign_but_retains_stopped_evidence(self):
        imports = []
        marker = type("SyntheticWaterLeaseMarker", (), {})
        def source_import(name, globals=None, locals=None, fromlist=(), level=0):
            imports.append((name, tuple(fromlist), level))
            if (name, tuple(fromlist), level) != ("masy", ("dense_water_pause",), 0):
                raise AssertionError("unexpected cleanup import")
            return SimpleNamespace(dense_water_pause=SimpleNamespace(_WaterLease=marker))
        builtin = dict(vars(__import__("builtins"))); builtin["__import__"] = source_import
        env = {"__builtins__": builtin, "sys": SimpleNamespace(modules={}), "_fail": _fail}
        _load("dense_boundary_store.py", ("_abandon_active_issued",), env)
        foreign = object()
        runner = SimpleNamespace(_active_evidence=foreign, _stop_receipt=None)
        with self.assertRaises(_Refusal) as error:
            env["_abandon_active_issued"](runner)
        self.assertEqual(error.exception.code, "wind_cleanup")
        self.assertIs(runner._active_evidence, foreign)
        stopped = {"status": "incomplete", "code": "synthetic_control_failure"}
        evidence = {"unaccepted": object()}
        runner._stop_receipt, runner._active_evidence = stopped, evidence
        self.assertEqual(env["_abandon_active_issued"](runner), "released")
        self.assertIs(runner._active_evidence, evidence)
        self.assertIs(runner._stop_receipt, stopped)
        runner._active_evidence, runner._stop_receipt = None, None
        self.assertEqual(env["_abandon_active_issued"](runner), "released")
        self.assertTrue(imports)
        self.assertTrue(all(row == ("masy", ("dense_water_pause",), 0) for row in imports))

    def test_all_source_census_constant_mutations_change_image(self):
        wind_names = ("_NATIVE_TABLE", "_SYRUP_TABLE", "_REVIEWED_NATIVE_TABLE_726",
                      "_REVIEWED_SYRUP_TABLE_726", "_REVIEWED_CURRENT_NATIVE_TABLE_937",
                      "_REVIEWED_CURRENT_SYRUP_TABLE_937")
        water_names = ("_NATIVE_CENSUS_MAPLE", "_NATIVE_CENSUS_SYRUP")
        wind = SimpleNamespace(**{k: _literal("dense_wind_pause.py", k) for k in wind_names})
        water = SimpleNamespace(**{k: _literal("dense_water_pause.py", k) for k in water_names})
        image = _node("dense_active_water_store.py", "_monitored_image")
        returned = next(n for n in image.body if isinstance(n, ast.Return))
        # Execute the exact constant-image expression only. No constructor,
        # provider, Native declaration or physical image is manufactured.
        expression = ast.Expression(body=returned.value.elts[2]); ast.fix_missing_locations(expression)
        names = {n.id for n in ast.walk(expression) if isinstance(n, ast.Name)}
        scope = {name: "synthetic_metadata_" + name for name in names}
        scope.update(wind=wind, water=water, legacy=((), (), ("legacy",)), storage128=False)
        code = compile(expression, "actual_monitored_constant_image", "eval", dont_inherit=True)
        env = {}; _load("runtime_worker.py", ("_same_compiled_image",), env)
        before = ((), (), eval(code, scope))
        self.assertTrue(env["_same_compiled_image"](before, before))
        for owner, keys in ((wind, wind_names), (water, water_names)):
            for key in keys:
                with self.subTest(key=key):
                    original = getattr(owner, key)
                    setattr(owner, key, "0" * 64 if original != "0" * 64 else "1" * 64)
                    after = ((), (), eval(code, scope))
                    self.assertFalse(env["_same_compiled_image"](before, after))
                    setattr(owner, key, original)
        self.assertTrue(env["_same_compiled_image"](before, ((), (), eval(code, scope))))

    def test_current_public_observer_and_held_checker_remain_direct(self):
        public = _node("runtime_worker.py", "compiled_identities")
        names = {n.id for n in ast.walk(public) if isinstance(n, ast.Name)}
        self.assertNotIn("_compiled_identities_scoped", names)
        self.assertNotIn("_compiled_identities_body", names)
        held = _node("dense_boundary_store.py", "_HeldInventory")
        attributes = {n.attr for n in ast.walk(held) if isinstance(n, ast.Attribute)}
        self.assertIn("_check_file", attributes)
        self.assertNotIn("_check_held_file", attributes)
        for filename in ("dense_original_inputs.py", "dense_active_checkpoint.py"):
            self.assertIsNone(_literal(filename, "_CANONICAL_METADATA_SCOPE"))


if __name__ == "__main__":
    unittest.main()
