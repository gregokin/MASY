"""Native resolver vs API evidence classification (fake modules only)."""

from __future__ import annotations

import dataclasses
import inspect
import sys
import types
import unittest
from dataclasses import dataclass
from typing import Any

import _support  # noqa: F401 - puts MASY src on sys.path

from masy.contracts import MASY_REQUIRED_API, Requirement, check_masy_api, crosscheck_provenance, run_native_audit
from masy.report import compute_verdict, render_report

API_MODULE = "masy_test_fake_native_api"
DRY_FIELDS = ("progress", "dry_state", "pre_reset_surface_m", "pre_reset_soil_m", "terminal_deposition_kg",
              "accounting", "status")


@dataclass(frozen=True)
class FakeDryHandoff:
    progress: Any
    dry_state: Any
    pre_reset_surface_m: Any
    pre_reset_soil_m: Any
    terminal_deposition_kg: Any
    accounting: dict
    status: str = "complete"


def _fake_api() -> types.ModuleType:
    mod = types.ModuleType(API_MODULE)

    def commit_bed(state, context, graph, reference, time_s, *, force=False, routing_source_m=None):
        return None

    mod.commit_bed = commit_bed
    mod.DryHandoff = FakeDryHandoff
    return mod


REQUIREMENTS_PASS = (
    Requirement(API_MODULE, "commit_bed", ("state", "time_s", "routing_source_m")),
    Requirement(API_MODULE, "DryHandoff", kind="dataclass", fields=DRY_FIELDS),
)
OPTIONAL_ABSENT = (("masy_test_absent_windowed_module", "windowed transaction"),)
DENSE_PATH_TYPE = Requirement(API_MODULE, "WindowedVoxelColumnState", kind="class",
                              required_by="is_windowed_bed on every Dense bed path")


def genuine_class(name: str, module: str = API_MODULE) -> type:
    return type(name, (), {"__module__": module})


def fake_dependency(resolver_exc: str | None = None, api_ok: bool = True, source_root: str = "/r/MAPLE"):
    dep = types.ModuleType("fake_dependency")

    class MapleDependencyError(RuntimeError):
        pass

    class MapleApiIncompatibleError(MapleDependencyError):
        pass

    @dataclass
    class Resolved:
        def as_record(self):
            return {"source_root": source_root, "source_kind": "editable_working_tree"}

    def resolve_maple_dependency(expected_source_root=None):
        if resolver_exc == "identity":
            raise MapleDependencyError(f"expected MAPLE source root {expected_source_root}, but ... source root None")
        if resolver_exc == "api":
            raise MapleApiIncompatibleError("missing maple.water.apply_water_process_demand")
        if resolver_exc == "other":
            raise KeyError("unexpected")
        return Resolved()

    def check_required_api():
        if not api_ok:
            raise MapleApiIncompatibleError("attribute missing")

    dep.MapleDependencyError = MapleDependencyError
    dep.MapleApiIncompatibleError = MapleApiIncompatibleError
    dep.REQUIRED_MAPLE_API = (1, 2, 3)
    dep.resolve_maple_dependency = resolve_maple_dependency
    dep.check_required_api = check_required_api
    return dep


class NativeAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.modules[API_MODULE] = _fake_api()

    def tearDown(self) -> None:
        sys.modules.pop(API_MODULE, None)

    def audit(self, dep, requirements=REQUIREMENTS_PASS):
        return run_native_audit(dep, "/r/MAPLE", masy_requirements=requirements, optional_modules=OPTIONAL_ABSENT)

    def codes(self, findings, severity):
        return [f["code"] for f in findings if f["severity"] == severity]

    def test_resolver_identity_failure_is_refusal_with_api_evidence_retained(self) -> None:
        record, findings = self.audit(fake_dependency("identity"))
        self.assertEqual(record["native_resolver"]["status"], "refused")
        self.assertIn("source root None", record["native_resolver"]["error"])
        self.assertEqual(record["syrup_required_api"], {"call": "check_required_api()", "status": "pass",
                                                        "requirement_count": 3})
        self.assertTrue(all(item["status"] == "pass" for item in record["masy_required_api"]))
        self.assertEqual(self.codes(findings, "refusal"), ["native_resolver_refused"])
        self.assertEqual(compute_verdict(findings), "refused")

    def test_resolver_api_failure_is_incompatible(self) -> None:
        record, findings = self.audit(fake_dependency("api"))
        self.assertEqual(record["native_resolver"]["status"], "api_incompatible")
        self.assertEqual(self.codes(findings, "refusal"), [])
        self.assertEqual(compute_verdict(findings), "incompatible")

    def test_unexpected_resolver_error_is_refusal(self) -> None:
        _, findings = self.audit(fake_dependency("other"))
        self.assertEqual(self.codes(findings, "refusal"), ["native_resolver_error"])

    def test_resolved_with_matching_root_is_best_not_transport_ready(self) -> None:
        _, findings = self.audit(fake_dependency())
        self.assertEqual(self.codes(findings, "refusal") + self.codes(findings, "incompatible"), [])
        report, exit_code = render_report({}, findings)
        self.assertEqual((report["verdict"], exit_code, report["transport_ready"]), ("not_transport_ready", 4, False))

    def test_resolved_with_other_root_refuses(self) -> None:
        _, findings = self.audit(fake_dependency(source_root=None))
        self.assertEqual(self.codes(findings, "refusal"), ["native_resolver_root_mismatch"])

    def test_missing_syrup_api_is_incompatible(self) -> None:
        record, findings = self.audit(fake_dependency(api_ok=False))
        self.assertEqual(record["syrup_required_api"]["status"], "fail")
        self.assertEqual(self.codes(findings, "incompatible"), ["syrup_required_api_missing"])

    def test_missing_masy_api_is_incompatible(self) -> None:
        requirements = REQUIREMENTS_PASS + (
            Requirement(API_MODULE, "finalize_event"),
            Requirement(API_MODULE, "commit_bed", ("force_reroute",)),
            Requirement("masy_test_no_such_module", "x"),
        )
        record, findings = self.audit(fake_dependency(), requirements)
        statuses = [item["status"] for item in record["masy_required_api"]]
        self.assertEqual(statuses, ["pass", "pass", "fail", "fail", "fail"])
        self.assertEqual(self.codes(findings, "incompatible"), ["masy_required_api_missing"])
        self.assertEqual(compute_verdict(findings), "incompatible")

    def test_absent_optional_windowed_module_is_info_only(self) -> None:
        record, findings = self.audit(fake_dependency())
        self.assertEqual(record["optional_modules"][0]["status"], "absent")
        self.assertEqual([f["severity"] for f in findings if f["code"].startswith("optional_module_")], ["info"])
        self.assertEqual(compute_verdict(findings), "not_transport_ready")

    def test_check_masy_api_reports_parameter_names(self) -> None:
        result = check_masy_api((Requirement(API_MODULE, "commit_bed", ("graph", "missing_one")),))
        self.assertEqual(result[0]["status"], "fail")
        self.assertIn("missing_one", result[0]["reason"])

    def test_dry_handoff_must_be_a_dataclass_with_its_fields(self) -> None:
        requirement = (Requirement(API_MODULE, "DryHandoff", kind="dataclass", fields=DRY_FIELDS),)
        partial = dataclasses.make_dataclass("DryHandoff", [(name, Any) for name in DRY_FIELDS if name != "status"])
        cases = {
            "none": (None, "not a dataclass type"),
            "plain_class": (type("DryHandoff", (), {}), "not a dataclass type"),
            "instance": (FakeDryHandoff(1, 2, 3, 4, 5, {}), "not a dataclass type"),
            "missing_field": (partial, "missing field(s) ['status']"),
        }
        for label, (value, reason) in cases.items():
            sys.modules[API_MODULE].DryHandoff = value
            with self.subTest(case=label):
                result = check_masy_api(requirement)[0]
                self.assertEqual(result["status"], "fail")
                self.assertIn(reason, result["reason"])

    def test_function_requirement_refuses_non_callables_and_classes(self) -> None:
        requirement = (Requirement(API_MODULE, "commit_bed", ("state",)),)
        for value in (None, 3, FakeDryHandoff):
            sys.modules[API_MODULE].commit_bed = value
            with self.subTest(value=value):
                result = check_masy_api(requirement)[0]
                self.assertEqual(result["status"], "fail")
                self.assertIn("not a callable function", result["reason"])

    def test_requirement_kind_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            Requirement(API_MODULE, "x", kind="anything")
        with self.assertRaises(ValueError):
            Requirement(API_MODULE, "x", ("a",), kind="dataclass")
        with self.assertRaises(ValueError):
            Requirement(API_MODULE, "x", ("a",), kind="class")
        with self.assertRaises(ValueError):
            Requirement(API_MODULE, "x", kind="class", fields=("a",))

    def test_class_requirement_refuses_non_classes(self) -> None:
        def function():
            return None
        cases = {
            "none": (None, "not a class (got NoneType)"),
            "function": (function, "not a class (got function)"),
            "instance": (genuine_class("WindowedVoxelColumnState")(), "not a class (got WindowedVoxelColumnState)"),
            "int": (3, "not a class (got int)"),
        }
        for label, (value, reason) in cases.items():
            sys.modules[API_MODULE].WindowedVoxelColumnState = value
            with self.subTest(case=label):
                result = check_masy_api((DENSE_PATH_TYPE,))[0]
                self.assertEqual(result["status"], "fail")
                self.assertIn(reason, result["reason"])

    def test_class_requirement_accepts_re_exported_and_aliased_classes(self) -> None:
        cases = {
            "defined_here": genuine_class("WindowedVoxelColumnState"),
            "re_exported": genuine_class("WindowedVoxelColumnState", "native_implementation"),
            "aliased": genuine_class("InternalColumn", "native_implementation"),
            "renamed_here": genuine_class("Other"),
            "nested": type("Outer", (), {"Inner": genuine_class("Inner")}).Inner,
        }
        for label, value in cases.items():
            sys.modules[API_MODULE].WindowedVoxelColumnState = value
            with self.subTest(case=label):
                result = check_masy_api((DENSE_PATH_TYPE,))[0]
                self.assertEqual(result["status"], "pass", result.get("reason"))

    def test_missing_dense_path_type_is_incompatible_and_resolver_refusal_retained(self) -> None:
        record, findings = self.audit(fake_dependency("identity"), REQUIREMENTS_PASS + (DENSE_PATH_TYPE,))
        self.assertEqual(record["syrup_required_api"]["status"], "pass")
        self.assertEqual(record["masy_required_api"][-1]["reason"], "attribute missing")
        self.assertEqual(self.codes(findings, "refusal"), ["native_resolver_refused"])
        self.assertEqual(self.codes(findings, "incompatible"), ["masy_required_api_missing"])
        detail = next(f["detail"] for f in findings if f["code"] == "masy_required_api_missing")
        self.assertIn("WindowedVoxelColumnState: attribute missing (required by is_windowed_bed on every Dense bed "
                      "path)", detail)
        self.assertEqual(compute_verdict(findings), "refused")
        self.assertEqual(compute_verdict([f for f in findings if f["severity"] != "refusal"]), "incompatible")

    def test_dense_path_type_is_distinct_from_the_optional_transaction_module(self) -> None:
        sys.modules[API_MODULE].WindowedVoxelColumnState = genuine_class("WindowedVoxelColumnState")
        record, findings = self.audit(fake_dependency(), REQUIREMENTS_PASS + (DENSE_PATH_TYPE,))
        self.assertTrue(all(item["status"] == "pass" for item in record["masy_required_api"]))
        self.assertEqual(record["optional_modules"][0]["status"], "absent")
        self.assertEqual(self.codes(findings, "incompatible"), [])
        self.assertEqual(compute_verdict(findings), "not_transport_ready")
        del sys.modules[API_MODULE].WindowedVoxelColumnState
        _, findings = self.audit(fake_dependency(), REQUIREMENTS_PASS + (DENSE_PATH_TYPE,))
        self.assertEqual(self.codes(findings, "incompatible"), ["masy_required_api_missing"])
        self.assertEqual([f["severity"] for f in findings if f["code"].startswith("optional_module_")], ["info"])


SURFACE_PREFIX = "masy_test_surface"


def _surface(requirements=MASY_REQUIRED_API) -> tuple[tuple[Requirement, ...], dict[str, types.ModuleType]]:
    """Synthetic modules (renamed under SURFACE_PREFIX) exposing exactly the
    names the real MASY table requires: keyword-only signatures for functions,
    dataclasses for types, plain classes defined in the renamed module for
    classes. The negative tests remove one item at a time."""
    renamed, modules = [], {}
    for req in requirements:
        name = f"{SURFACE_PREFIX}.{req.module}"
        module = modules.setdefault(name, types.ModuleType(name))
        if req.kind == "class":
            setattr(module, req.attribute, genuine_class(req.attribute, name))
        elif req.kind == "dataclass":
            setattr(module, req.attribute, dataclasses.make_dataclass(req.attribute, [(f, Any) for f in req.fields]))
        else:
            def function(*args, **kwargs):
                return None
            function.__signature__ = inspect.Signature(
                [inspect.Parameter(p, inspect.Parameter.KEYWORD_ONLY) for p in req.parameters])
            setattr(module, req.attribute, function)
        renamed.append(dataclasses.replace(req, module=name))
    return tuple(renamed), modules


class RequiredSurfaceTests(unittest.TestCase):
    """The real MASY_REQUIRED_API table, checked against synthetic modules."""

    def setUp(self) -> None:
        self.requirements, self.modules = _surface()
        sys.modules.update(self.modules)

    def tearDown(self) -> None:
        for name in self.modules:
            sys.modules.pop(name, None)

    def module(self, real: str) -> types.ModuleType:
        return self.modules[f"{SURFACE_PREFIX}.{real}"]

    def failures(self) -> list[tuple[str, str]]:
        return [(r["attribute"], r["reason"]) for r in check_masy_api(self.requirements) if r["status"] != "pass"]

    def test_table_names_the_source_backed_surface(self) -> None:
        table = {(r.module, r.attribute): r for r in MASY_REQUIRED_API}
        self.assertEqual(table["maple.aeolian.scheduler.block", "advance_wind_block"].parameters,
                         ("state", "block", "config"))
        self.assertIn("coupled_trajectory_semantics",
                      table["maple.aeolian.scheduler.state", "initial_scheduler_state"].parameters)
        self.assertIn("coupled", table["maple.aeolian.scheduler.state", "initial_scheduler_state"].parameters)
        self.assertEqual(table["maple_syrup.sediment_bed", "reroute_threshold_m"].parameters, ("context",))
        self.assertEqual(table["maple.core.parameters.topographic_commit", "TopographicCommitSpec"].fields,
                         ("pending_elevation_trigger_m",))
        self.assertIn("continuation", table["maple_syrup.complete_event", "complete_event"].parameters)
        for key in (("maple.aeolian.scheduler.state_validation", "validate_event_boundary"),
                    ("maple.aeolian.cohorts.dispatch", "mobile_state_mass_is_finite"),
                    ("maple.aeolian.cohorts.dispatch", "mobile_state_is_genuinely_empty")):
            self.assertIn(key, table)
        for key in (("maple.aeolian.scheduler.state", "SchedulerPhysicalState"),
                    ("maple_syrup.sediment_bed", "BedState"), ("maple_syrup.complete_event", "DryHandoff")):
            self.assertEqual(table[key].kind, "dataclass")

    def test_table_names_the_dense_bed_path(self) -> None:
        table = {(r.module, r.attribute): r for r in MASY_REQUIRED_API}
        for name in ("VoxelColumnState", "WindowedVoxelColumnState"):
            self.assertEqual(table["maple.core.types.voxel", name].kind, "class")
        self.assertIn("is_windowed_bed", table["maple.core.types.voxel", "WindowedVoxelColumnState"].required_by)
        expected = {
            "is_windowed_bed": ("state",), "bed_inventory": ("state",),
            "apply_bed_demand": ("state", "context", "demand"), "bed_by_cell_class": ("state",),
            "bed_voxel_count": ("state",), "bed_already_committed": ("state", "context", "time_s"),
        }
        for attribute, parameters in expected.items():
            self.assertEqual(table["maple_syrup.sediment_bed", attribute].parameters, parameters)
        self.assertEqual(table["maple.aeolian.flux.step", "bed_inventory_by_class_kg"].parameters,
                         ("voxel_column", "active_layer"))
        self.assertIn("demand", table["maple.water", "apply_water_process_demand"].parameters)
        self.assertIn("current_time_s",
                      table["maple.surface.topographic_commit.commit", "is_already_committed"].parameters)
        self.assertIn("mass_resolution_kg", table["maple_syrup.sediment_bed", "BedContext"].fields)
        self.assertNotIn("maple.water.windowed_transaction", {r.module for r in MASY_REQUIRED_API})

    def test_complete_synthetic_surface_passes(self) -> None:
        self.assertEqual(self.failures(), [])

    def test_missing_windowed_type_fails_the_dense_surface(self) -> None:
        del self.module("maple.core.types.voxel").WindowedVoxelColumnState
        self.assertEqual(self.failures(), [("WindowedVoxelColumnState", "attribute missing")])

    def test_dense_voxel_type_bound_to_function_fails(self) -> None:
        def VoxelColumnState(mass_kg):  # noqa: N802 - a callable factory is not an isinstance target
            return None
        self.module("maple.core.types.voxel").VoxelColumnState = VoxelColumnState
        self.assertEqual(self.failures(), [("VoxelColumnState", "not a class (got function)")])

    def test_dense_voxel_type_re_exported_from_implementation_module_passes(self) -> None:
        self.module("maple.core.types.voxel").VoxelColumnState = genuine_class(
            "InternalColumn", "native_implementation")
        self.assertEqual(self.failures(), [])

    def test_removed_wind_dispatcher_fails(self) -> None:
        del self.module("maple.aeolian.scheduler.block").advance_wind_block
        self.assertEqual(self.failures(), [("advance_wind_block", "attribute missing")])

    def test_dispatcher_without_config_parameter_fails(self) -> None:
        def advance_wind_block(state, block):
            return None
        self.module("maple.aeolian.scheduler.block").advance_wind_block = advance_wind_block
        self.assertEqual(self.failures(), [("advance_wind_block", "missing parameter(s) ['config']")])

    def test_removed_reroute_threshold_fails(self) -> None:
        del self.module("maple_syrup.sediment_bed").reroute_threshold_m
        self.assertEqual(self.failures(), [("reroute_threshold_m", "attribute missing")])

    def test_commit_spec_without_trigger_field_fails(self) -> None:
        module = self.module("maple.core.parameters.topographic_commit")
        module.TopographicCommitSpec = dataclasses.make_dataclass("TopographicCommitSpec", [("other_m", float)])
        self.assertEqual(self.failures(),
                         [("TopographicCommitSpec", "missing field(s) ['pending_elevation_trigger_m']")])

    def test_scheduler_state_without_trajectory_semantics_field_fails(self) -> None:
        module = self.module("maple.aeolian.scheduler.state")
        kept = [f.name for f in dataclasses.fields(module.SchedulerPhysicalState)
                if f.name != "coupled_trajectory_semantics"]
        module.SchedulerPhysicalState = dataclasses.make_dataclass("SchedulerPhysicalState", kept)
        self.assertEqual(self.failures(),
                         [("SchedulerPhysicalState", "missing field(s) ['coupled_trajectory_semantics']")])

    def test_dry_handoff_none_fails(self) -> None:
        self.module("maple_syrup.complete_event").DryHandoff = None
        self.assertEqual(self.failures(), [("DryHandoff", "not a dataclass type (got NoneType)")])


class ProvenanceCrosscheckTests(unittest.TestCase):
    OBSERVATION = {
        name: {"root": f"/r/{name}", "package": {"dir": f"/r/{name}/src/pkg", "digest_sha256": "a" * 64,
                                                  "file_count": 2, "total_bytes": 10, "algorithm": "alg"}}
        for name in ("maple", "syrup")
    }

    def provenance(self, digest: str, status: str = "clean"):
        @dataclass
        class Digest:
            digest_sha256: str
            file_count: int = 2
            total_bytes: int = 10
            algorithm: str = "alg"

            def as_record(self):
                return {"digest_sha256": self.digest_sha256}

        mod = types.ModuleType("fake_provenance")
        mod.source_tree_digest = lambda root, *, max_files, max_bytes: Digest(digest)
        mod.scoped_git_status = lambda root, pathspecs, *, timeout_s: {"status": status}
        return mod

    def test_agreement_passes(self) -> None:
        record, findings = crosscheck_provenance(self.provenance("a" * 64), self.OBSERVATION,
                                                 max_files=10, max_bytes=10, git_timeout_s=1.0)
        self.assertEqual(findings, [])
        self.assertTrue(record["maple"]["agrees"])

    def test_digest_or_status_disagreement_refuses(self) -> None:
        for provenance in (self.provenance("b" * 64), self.provenance("a" * 64, status="dirty")):
            _, findings = crosscheck_provenance(provenance, self.OBSERVATION, max_files=10, max_bytes=10,
                                                git_timeout_s=1.0)
            self.assertEqual({f["code"] for f in findings}, {"upstream_provenance_disagreement"})
            self.assertEqual(compute_verdict(findings), "refused")


if __name__ == "__main__":
    unittest.main()
