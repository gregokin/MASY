"""Native daily wind executor checks.

Ordinary discovery runs the metadata/import/limits tests with no native runtime. Set MASY_DAILY_NATIVE_BUNDLE and
MASY_DAILY_NATIVE_SYRUP (the explicit MAPLE 9f bundle and SYRUP N15 source already on the interpreter's import path) for the
genuine native CPU tests, which follow the factory chain of tests/test_daily_native.py. Nothing installs packages or alters an
upstream tree. Numerical inputs are illustrative, not calibration.
"""
import dataclasses
import datetime
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from masy import daily_wind
from masy.case_schedule import WindDeclaration
from masy.daily_csv import WindOccurrence
from masy.daily_wind import DailyWindError, DailyWindLimits, prepare_daily_wind

BUNDLE = os.environ.get("MASY_DAILY_NATIVE_BUNDLE")
SYRUP = os.environ.get("MASY_DAILY_NATIVE_SYRUP")

LIMITS = dict(max_cells=1 << 10, max_voxels=64, max_cohorts=1 << 14, max_blocks=8, max_accounting_steps=1 << 12,
              max_stage_bytes=1 << 24, max_work_bytes=1 << 28, max_receipt_bytes=1 << 20)


def occurrence(index=0, durations=(10.0,), shear=0.4, reference="wind.yaml", day=datetime.date(2028, 3, 1)):
    blocks = tuple(WindDeclaration(i, shear, 90.0, d) for i, d in enumerate(durations))
    return WindOccurrence(index, f"occ-{index}", f"{index:064x}", day, "wind", reference, 100, "ab" * 32, blocks)


class Metadata(unittest.TestCase):
    def test_import_is_native_free(self):
        code = ("import sys; sys.path.insert(0, %r); import masy.daily_wind; "
                "bad = [m for m in ('numpy', 'maple', 'maple_syrup') if m in sys.modules]; "
                "sys.exit(1 if bad else 0)" % str(ROOT / "src"))
        self.assertEqual(subprocess.run([sys.executable, "-I", "-B", "-c", code]).returncode, 0)

    def test_source_pair_constants_match_water_connection(self):
        from masy import daily_native
        self.assertEqual(daily_wind.SYRUP_PACKAGE_SHA256, daily_native.SYRUP_PACKAGE_SHA256)
        self.assertEqual(daily_wind.MAPLE_COMMIT, daily_native.MAPLE_COMMIT)
        self.assertEqual(daily_wind.MAPLE_PACKAGE_SHA256, daily_native.MAPLE_PACKAGE_SHA256)

    def test_limits_are_explicit_plain_positive_ints(self):
        self.assertEqual(DailyWindLimits(**LIMITS).as_record(), LIMITS)
        for name in LIMITS:
            for bad in (0, -1, 1.0, True, None):
                with self.subTest(name=name, bad=bad), self.assertRaises(DailyWindError) as ctx:
                    DailyWindLimits(**{**LIMITS, name: bad})
                self.assertEqual(ctx.exception.code, "limits")

    def test_prepare_requires_limits_and_genuine_native_records(self):
        with self.assertRaises(DailyWindError) as ctx:
            prepare_daily_wind(None, None, None, limits=None)
        self.assertEqual(ctx.exception.code, "limits")
        with self.assertRaises(DailyWindError) as ctx:
            prepare_daily_wind(None, None, None, limits=DailyWindLimits(**LIMITS))
        self.assertIn(ctx.exception.code, ("native_unavailable", "native_source"))

    def test_error_keeps_only_bounded_scalar_evidence(self):
        error = DailyWindError("code", "x" * 5000, evidence={"block_index": 1})
        self.assertLessEqual(len(error.detail), 400)
        self.assertEqual(error.evidence, {"block_index": 1})
        self.assertIsNone(error.__cause__)


@unittest.skipUnless(BUNDLE and SYRUP, "explicit native MAPLE 9f / SYRUP N15 CPU source pair not selected")
class NativePhysical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import yaml
        from maple_syrup import native_case_profile as ncp, native_dense_profile as nd
        cls.np, cls.yaml, cls.ncp, cls.nd = np, yaml, ncp, nd
        cls.source = ncp.verify_source_profile(BUNDLE, expected_syrup_digest=daily_wind.SYRUP_PACKAGE_SHA256)
        cls.public = Path(SYRUP) / "docs/manuals/examples/native_leaf_calendar_cli"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        case = Path(self.tmp.name) / "case"
        shutil.copytree(self.public / "case", case)
        for file in case.rglob("*"):
            if file.is_file():
                file.chmod(0o644)  # owned synthetic input, copied from a read-only source archive
        self.admission = self.ncp.admit_package_case(self.source, case)
        self.prep = self.nd.prepare_native_dense_case(self.admission, caller_reserve_bytes=1 << 20)
        self.bed, self.context = self.prep.bed, self.prep.context

    def executor(self, **limits):
        return prepare_daily_wind(self.source, self.admission, self.context,
                                  limits=DailyWindLimits(**{**LIMITS, **limits}))

    def digest(self, bed):
        return daily_wind._bed_digest(bed)

    def ponded(self, depth=0.01):
        np = self.np
        water = dataclasses.replace(self.bed.water, depth_m=np.full_like(self.bed.water.depth_m, depth))
        return dataclasses.replace(self.bed, water=water)

    def test_wind_transports_preserves_ponded_water_and_input(self):
        np = self.np
        bed = self.ponded()
        before = self.digest(bed)
        candidate = self.executor().execute(bed, occurrence(), 12.5)
        self.assertEqual(self.digest(bed), before)
        self.assertNotEqual(self.digest(candidate.state), before)
        self.assertTrue(np.array_equal(candidate.state.water.depth_m, bed.water.depth_m))
        self.assertGreater(float(np.abs(candidate.state.water.depth_m).sum()), 0.0)
        receipt = candidate.receipt()
        self.assertEqual(receipt["clocks"]["start_s"]["hex"], (12.5).hex())
        self.assertEqual(receipt["clocks"]["end_s"]["hex"], candidate.end_s.hex())
        self.assertGreater(candidate.end_s, 12.5)
        self.assertTrue(any(v != 0.0 for v in receipt["classes"]["bed_change_kg"] + receipt["classes"]["boundary_export_kg"]
                            + receipt["classes"]["settled_deposition_kg"]))
        self.assertTrue(receipt["checks"]["inventory_closure_with_boundary_export"])
        availability = candidate.state.sediment_availability
        self.assertTrue(np.all(availability.available_mass_kg >= 0))
        self.assertTrue(np.any(availability.bound_mass_kg > 0))
        self.assertFalse(receipt["wind_policy"]["soil_moisture_feedback_enabled"])
        self.assertEqual(receipt["wind_policy"]["water_depth_update_rule"], "constant_depth")

    def test_repeated_same_file_events_are_distinct_with_exact_origins(self):
        executor = self.executor()
        first = executor.execute(self.bed, occurrence(0), 100.0)
        second = executor.execute(first.state, occurrence(1), first.end_s)
        a, b = first.receipt(), second.receipt()
        self.assertEqual(a["occurrence"]["reference"], b["occurrence"]["reference"])
        self.assertNotEqual(a["occurrence"]["occurrence_sha256"], b["occurrence"]["occurrence_sha256"])
        self.assertEqual(b["clocks"]["start_s"]["hex"], first.end_s.hex())
        self.assertEqual(b["clocks"]["record_origin_s"]["hex"], first.end_s.hex())
        self.assertEqual(b["parent_bed_sha256"], a["candidate_bed_sha256"])

    def test_dry_wind_policy_overrides_are_explicit_and_case_is_unchanged(self):
        case = Path(self.tmp.name) / "wet_policy_case"
        shutil.copytree(self.public / "case", case)
        for file in case.rglob("*"):
            if file.is_file():
                file.chmod(0o644)
        file = case / "model.yaml"
        raw = self.yaml.safe_load(file.read_text())
        raw["fluid_threshold"] = {"modifiers": {"moisture_enabled": True}}
        file.write_text(self.yaml.safe_dump(raw))
        file = case / "case.yaml"
        raw = self.yaml.safe_load(file.read_text())
        raw["water_initial_depth"] = {"depth_m": 0.001}
        file.write_text(self.yaml.safe_dump(raw))
        admission = self.ncp.admit_package_case(self.source, case)
        prep = self.nd.prepare_native_dense_case(admission, caller_reserve_bytes=1 << 20)
        original = admission.compiled.config
        before = self.digest(prep.bed)
        executor = prepare_daily_wind(self.source, admission, prep.context, limits=DailyWindLimits(**LIMITS))
        result = executor.execute(prep.bed, occurrence(durations=(2.0, 1.0)), 86400.0)
        policy = result.receipt()["wind_policy"]
        self.assertTrue(original.fluid_threshold.modifiers.moisture_enabled)
        self.assertEqual(original.water_coupling.depth_update_rule, "constant_free_surface")
        self.assertFalse(policy["soil_moisture_feedback_enabled"])
        self.assertTrue(policy["admitted_moisture_feedback_enabled"])
        self.assertEqual(policy["water_depth_update_rule"], "constant_depth")
        self.assertEqual(self.digest(prep.bed), before)
        self.np.testing.assert_array_equal(result.state.water.depth_m, prep.bed.water.depth_m)

    def test_multiple_blocks_run_in_row_order_with_exact_intervals(self):
        candidate = self.executor().execute(self.bed, occurrence(durations=(4.0, 6.0)), 3.0)
        blocks = candidate.receipt()["blocks"]
        self.assertEqual([b["index"] for b in blocks], [0, 1])
        self.assertEqual(blocks[1]["start_s"]["hex"], blocks[0]["realized_end_s"]["hex"])
        self.assertEqual(blocks[0]["expected_duration_s"]["hex"], (4.0).hex())

    def test_limits_refuse_before_work_and_leave_input_identical(self):
        before = self.digest(self.bed)
        for name, value, code in (("max_cells", 1, "cell_limit"), ("max_voxels", 1, "cell_limit"),
                                  ("max_blocks", 1, "block_limit"), ("max_accounting_steps", 1, "accounting_step_limit"),
                                  ("max_stage_bytes", 1, "stage_bytes_cap"), ("max_work_bytes", 1, "work_bytes_cap"),
                                  ("max_cohorts", 1, "cohort_limit"), ("max_receipt_bytes", 1, "receipt_bytes_cap")):
            with self.subTest(name=name):
                limits = {name: value}
                try:
                    executor = self.executor(**limits)
                except DailyWindError as exc:
                    self.assertEqual(exc.code, code)
                    continue
                with self.assertRaises(DailyWindError) as ctx:
                    executor.execute(self.bed, occurrence(durations=(4.0, 6.0)), 0.0)
                self.assertEqual(ctx.exception.code, code)
                self.assertEqual(self.digest(self.bed), before)

    def test_water_borne_load_is_refused(self):
        np = self.np
        mobile = np.array(self.bed.water.mobile_mass_by_cell_class_kg, copy=True)
        mobile.flat[0] = 1e-3
        bed = dataclasses.replace(self.bed, water=dataclasses.replace(self.bed.water, mobile_mass_by_cell_class_kg=mobile))
        before = self.digest(bed)
        with self.assertRaises(DailyWindError) as ctx:
            self.executor().execute(bed, occurrence(), 0.0)
        self.assertEqual(ctx.exception.code, "water_mobile_not_empty")
        self.assertEqual(self.digest(bed), before)

    def test_unqualified_inputs_are_refused(self):
        executor = self.executor()
        for bed, occ, start, code in ((None, occurrence(), 0.0, "invalid_bed"), (self.bed, None, 0.0, "occurrence"),
                                      (self.bed, occurrence(), -1.0, "start_time"),
                                      (self.bed, occurrence(), float("nan"), "start_time")):
            with self.subTest(code=code), self.assertRaises(DailyWindError) as ctx:
                executor.execute(bed, occ, start)
            self.assertEqual(ctx.exception.code, code)

    def test_enabled_native_water_coupling_is_refused_not_changed(self):
        config = self.admission.compiled.config
        coupling = dataclasses.replace(config.water_coupling, enabled=True)
        patched = dataclasses.replace(config, water_coupling=coupling)
        compiled = self.admission.compiled
        # Inject an invalid cached configuration on the genuine compiled holder;
        # a class attribute cannot shadow a frozen instance's own config field.
        object.__setattr__(compiled, "config", patched)
        try:
            with self.assertRaises(DailyWindError) as ctx:
                self.executor()
        finally:
            object.__setattr__(compiled, "config", config)
        self.assertEqual(ctx.exception.code, "water_coupling_enabled")

    def test_native_failure_stops_with_bounded_evidence_and_intact_input(self):
        from unittest import mock
        before = self.digest(self.bed)
        with mock.patch.object(daily_wind, "_advance_wind_block", side_effect=ValueError("injected")):
            with self.assertRaises(DailyWindError) as ctx:
                self.executor().execute(self.bed, occurrence(), 0.0)
        self.assertEqual(ctx.exception.code, "native_failure")
        self.assertEqual(ctx.exception.evidence["block_index"], 0)
        self.assertIsNone(ctx.exception.__context__)
        self.assertEqual(self.digest(self.bed), before)

    def test_incomplete_finalization_is_refusal(self):
        from unittest import mock
        with mock.patch.object(daily_wind, "_finalize_wind_event", side_effect=ValueError("injected")):
            with self.assertRaises(DailyWindError) as ctx:
                self.executor().execute(self.bed, occurrence(), 0.0)
        self.assertEqual(ctx.exception.code, "finalization_failure")


if __name__ == "__main__":
    unittest.main()
