"""The additional native wind/water interface; actual native cases are opt-in."""
import dataclasses
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from masy import daily_wind as dw, daily_wind_handoff as mh
import wind_handoff_support as hs


class Metadata(unittest.TestCase):
    def test_import_is_native_free(self):
        code = "import sys; sys.path.insert(0,%r); import masy.daily_wind_handoff; assert not any(n in sys.modules for n in ('numpy','maple','maple_syrup'))" % str(ROOT / "src")
        subprocess.run([sys.executable, "-I", "-B", "-c", code], check=True)

    def test_resource_policy_and_profile_are_explicit(self):
        for name in ("max_receipt_bytes", "max_record_bytes"):
            for value in (0, -1, True, 1.0):
                with self.subTest(name=name, value=value), self.assertRaises(mh.WindHandoffError):
                    mh.WindBridgeLimits(**{**dict(max_receipt_bytes=4096, max_record_bytes=4096), name: value})
        self.assertNotEqual(mh.PROFILE, dw.PROFILE)
        self.assertNotEqual(mh.SYRUP_PACKAGE_SHA256, dw.SYRUP_PACKAGE_SHA256)
        with self.assertRaises(mh.WindHandoffError):
            mh.prepare_wind_handoff(None, None, None, None, wind_limits=None, handoff_limits=None, bridge_limits=None)


@unittest.skipUnless(os.environ.get("MASY_DAILY_WIND_HANDOFF_NATIVE") == "1",
                     "explicit newly qualified native wind-handoff CPU source not selected")
class NativeInterface(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, *, same_day=False, depth=0.0, threshold=5.0, precipitation=0.0, **kwargs):
        plan = hs.make_plan(self.root / "inputs", same_day=same_day, threshold=threshold, precipitation=precipitation)
        native = hs.make_native(self.root, depth=depth, **kwargs)
        return plan, native, self.prepare(plan, native)

    def prepare(self, plan, native, *, reader=None, bridge_limits=None):
        return mh.prepare_wind_handoff(plan, native.source, native.admission, native.context,
            wind_limits=hs.wind_limits(), handoff_limits=hs.handoff_limits(),
            bridge_limits=bridge_limits or mh.WindBridgeLimits(max_receipt_bytes=2 << 20, max_record_bytes=1 << 16),
            available_bytes_fn=reader or native.available)

    def start(self, bridge, native):
        return bridge.start(native.owner, native.bed, native.factory)

    def test_impossible_native_cohort_cap_refuses_during_preparation(self):
        plan, f, _unused = self.fixture()
        cohort_cap = f.admission.compiled.config.particle_transport.max_active_cohorts - 1
        with self.assertRaises(mh.WindHandoffError) as raised:
            mh.prepare_wind_handoff(plan, f.source, f.admission, f.context,
                wind_limits=hs.wind_limits(), handoff_limits=dataclasses.replace(hs.handoff_limits(), max_cohorts=cohort_cap),
                bridge_limits=mh.WindBridgeLimits(max_receipt_bytes=2 << 20, max_record_bytes=1 << 16),
                available_bytes_fn=f.available)
        self.assertEqual(raised.exception.code, "limits")
        self.assertTrue(f.owner.is_current)

    def test_native_pending_head_errors_have_the_adapter_not_ready_code(self):
        from maple_syrup import floating_event_lifecycle as lc
        plan, f, bridge = self.fixture()
        parent = self.start(bridge, f)
        # Fault injection of the actual native refusal type, not a model result.
        with patch.object(lc, "saveable_head", side_effect=lc.LifecycleError("pending recession is not saveable")):
            for operation in (lambda: bridge.execute(plan.scenario.occurrences[0]),
                              lambda: bridge.checkpoint_record(native_manifest_sha256="f" * 64)):
                with self.assertRaises(mh.WindHandoffError) as raised:
                    operation()
                self.assertEqual(raised.exception.code, "not_ready")
        self.assertIs(bridge.lineage.head, parent)
        self.assertEqual(bridge.next_wind_index, 0)

    def test_actual_csv_wind_preserves_ponded_water_soil_history_and_routing(self):
        import numpy as np
        plan, f, bridge = self.fixture(depth=0.001)
        parent = self.start(bridge, f)
        self.assertFalse(f.owner.is_current)
        before = parent.root.water_state.water_m.copy()
        history = parent.root.history
        out = bridge.execute(plan.scenario.occurrences[0])
        self.assertIs(bridge.lineage.head, out)
        self.assertEqual(bridge.next_wind_index, 1)
        self.assertTrue(np.array_equal(out.bed.water.depth_m, parent.bed.water.depth_m))
        self.assertGreater(float(out.bed.water.depth_m.sum()), 0)
        self.assertTrue(np.array_equal(out.root.water_state.water_m, before))
        self.assertTrue(np.array_equal(out.root.history.wetting_m, history.wetting_m))
        self.assertTrue(np.array_equal(out.root.history.cumulative_intake_m, history.cumulative_intake_m))
        self.assertIs(out.routing, parent.routing)
        self.assertEqual(out.t_s, out.root.water_state.t_s)
        self.assertEqual(out.t_s.hex(), bridge.last_receipt()["wind"]["clocks"]["end_s"]["hex"])
        self.assertNotEqual(dw._bed_digest(out.bed), dw._bed_digest(parent.bed))
        self.assertTrue(bridge.last_receipt()["wind"]["checks"]["inventory_closure_with_boundary_export"])
        with self.assertRaises(dw.DailyWindError) as err:
            dw.prepare_daily_wind(f.source, f.admission, f.context, limits=hs.wind_limits())
        self.assertEqual(err.exception.code, "native_source")

    def test_same_file_same_day_order_no_replay_and_exact_aggregate(self):
        plan, f, bridge = self.fixture(same_day=True)
        self.start(bridge, f)
        bridge.execute(plan.scenario.occurrences[0])
        first = bridge.last_receipt()
        with self.assertRaises(mh.WindHandoffError):
            bridge.execute(plan.scenario.occurrences[0])
        bridge.execute(plan.scenario.occurrences[1])
        second = bridge.last_receipt()
        self.assertEqual(first["wind"]["occurrence"]["reference"], second["wind"]["occurrence"]["reference"])
        self.assertNotEqual(first["wind"]["occurrence"]["occurrence_id"], second["wind"]["occurrence"]["occurrence_id"])
        self.assertEqual(second["wind"]["clocks"]["start_s"], first["wind"]["clocks"]["end_s"])
        got = bridge.wind_accounting()["boundary_export"]
        for k, pair in enumerate(got):
            expected = sum((Fraction(float.fromhex(r["wind"]["classes"]["boundary_export_hex"][k])) for r in (first, second)), Fraction(0))
            self.assertEqual(Fraction(int(pair[0]), int(pair[1])), expected)
        self.assertIsNone(bridge.lineage.head.cursor)

    def test_reduced_rain_and_daily_update_then_wind_use_the_live_water(self):
        import numpy as np
        plan, f, bridge = self.fixture(precipitation=0.1)
        initial = self.start(bridge, f).root.water_state.water_m.copy()
        bridge.execute(plan.scenario.occurrences[0])
        day, audits = hs.run_native_day(bridge, f, next(plan.days()))
        self.assertEqual(audits, [])
        self.assertEqual(day.cursor.ordinal, 1)
        live = day.root.water_state.water_m.copy()
        self.assertFalse(np.array_equal(initial, live))
        history = day.root.history
        out = bridge.execute(plan.scenario.occurrences[1])
        self.assertEqual(out.cursor.ordinal, 1)
        self.assertTrue(np.array_equal(out.root.water_state.water_m, live))
        self.assertTrue(np.array_equal(out.root.history.wetting_m, history.wetting_m))
        self.assertTrue(np.array_equal(out.root.history.cumulative_intake_m, history.cumulative_intake_m))

    def test_full_planned_native_storm_and_day_then_wind_close_class_budgets(self):
        import numpy as np
        from maple_syrup.sediment_bed import bed_inventory
        # Deliberately small explicit threshold for a short full-storm control;
        # the shipped user example retains the initial 5 mm threshold.
        plan, f, bridge = self.fixture(threshold=0.005, precipitation=0.01)
        self.start(bridge, f)
        bridge.execute(plan.scenario.occurrences[0])
        day, audits = hs.run_native_day(bridge, f, next(plan.days()))
        self.assertTrue(audits)
        self.assertTrue(all(a["water"]["closes"] and a["sediment"]["closes"] for a in audits))
        self.assertTrue(any(a["water"]["rain_m3"] > 0 for a in audits))
        live = day.root.water_state.water_m.copy()
        history = day.root.history
        self.assertTrue(np.any(history.cumulative_intake_m > 0.02))
        out = bridge.execute(plan.scenario.occurrences[1])
        self.assertEqual(out.root.water_state.t_s, out.t_s)
        self.assertTrue(np.array_equal(out.root.water_state.water_m, live))
        self.assertTrue(np.array_equal(out.root.history.wetting_m, history.wetting_m))
        self.assertEqual(out.cursor.ordinal, 1)
        totals = bridge.wind_accounting()
        bound = Fraction(*map(int, totals["inventory_bound"])) + sum((Fraction(float(a["sediment"]["class_closure_bound_kg"])) for a in audits), Fraction(0))
        current = bed_inventory(out.bed)
        for k in range(len(current)):
            water_export = sum((Fraction(float(np.asarray(a["sediment"]["export_wet_kg"])[k] + np.asarray(a["sediment"]["export_splash_kg"])[k])) for a in audits), Fraction(0))
            residual = (Fraction(float(current[k])) - Fraction(float.fromhex(totals["initial_inventory_hex"][k]))
                        + Fraction(*map(int, totals["boundary_export"][k])) - Fraction(*map(int, totals["boundary_input"][k])) + water_export)
            self.assertLessEqual(abs(residual), bound)

    def test_wrong_date_and_mutated_wind_row_refuse_before_transport(self):
        plan, f, bridge = self.fixture()
        head = self.start(bridge, f)
        with self.assertRaises(mh.WindHandoffError):
            bridge.execute(plan.scenario.occurrences[1])
        with self.assertRaises(mh.WindHandoffError):
            bridge.execute(dataclasses.replace(plan.scenario.occurrences[0], reference="different.yaml"))
        self.assertIs(bridge.lineage.head, head)
        self.assertEqual(bridge.next_wind_index, 0)

    def test_native_quota_failure_is_retryable_without_cursor_or_parent_change(self):
        from maple_syrup import native_floating_soil as nfs
        plan, f, _unused = self.fixture(same_day=True)
        budget = [1 << 40]
        bridge = self.prepare(plan, f, reader=lambda: budget[0])
        parent = self.start(bridge, f)
        fingerprint = dw._bed_digest(parent.bed)
        budget[0] = 0
        with self.assertRaises(nfs.NativeFloatingMemoryError):
            bridge.execute(plan.scenario.occurrences[0])
        self.assertIs(bridge.lineage.head, parent)
        self.assertEqual(dw._bed_digest(parent.bed), fingerprint)
        self.assertEqual(bridge.next_wind_index, 0)
        self.assertIsNone(bridge.last_receipt())
        budget[0] = 1 << 40
        bridge.execute(plan.scenario.occurrences[0])
        self.assertEqual(bridge.next_wind_index, 1)

    def test_interrupt_immediately_after_native_publication_updates_cursor_once(self):
        from maple_syrup import floating_wind_handoff as wh
        plan, f, bridge = self.fixture(same_day=True)
        parent = self.start(bridge, f)
        real = wh.adopt_wind_handoff

        def after(*args, **kwargs):
            real(*args, **kwargs)
            raise KeyboardInterrupt("injected after successful native publication")

        with patch.object(wh, "adopt_wind_handoff", after), self.assertRaises(KeyboardInterrupt):
            bridge.execute(plan.scenario.occurrences[0])
        self.assertIsNot(bridge.lineage.head, parent)
        self.assertEqual(bridge.next_wind_index, 1)
        self.assertIsNotNone(bridge.last_receipt())
        with self.assertRaises(mh.WindHandoffError):
            bridge.execute(plan.scenario.occurrences[0])
        bridge.execute(plan.scenario.occurrences[1])
        self.assertEqual(bridge.next_wind_index, 2)

    def test_failed_initial_retirement_preserves_owner_and_can_retry(self):
        from maple_syrup import native_floating_soil as nfs
        plan, f, bridge = self.fixture()
        with patch.object(nfs, "retire_native_owner_to_lineage", side_effect=MemoryError("injected")), self.assertRaises(MemoryError):
            self.start(bridge, f)
        self.assertTrue(f.owner.is_current)
        with self.assertRaises(mh.WindHandoffError):
            _ = bridge.lineage
        self.start(bridge, f)
        self.assertFalse(f.owner.is_current)

    def test_finalized_wind_workspace_is_released_before_water_adoption(self):
        from maple_syrup import floating_wind_handoff as wh
        plan, f, bridge = self.fixture()
        self.start(bridge, f)
        refs = []
        real_wind = bridge._wind.execute_finalized
        real_adopt = wh.adopt_wind_handoff

        def observe(*args, **kwargs):
            out = real_wind(*args, **kwargs)
            refs.extend((weakref.ref(out.finalization), weakref.ref(out.finalization.state.physical.mobile_state)))
            return out

        def assert_released(*args, **kwargs):
            self.assertTrue(refs and all(r() is None for r in refs))
            return real_adopt(*args, **kwargs)

        with patch.object(bridge._wind, "execute_finalized", observe), patch.object(wh, "adopt_wind_handoff", assert_released):
            bridge.execute(plan.scenario.occurrences[0])

    def test_native_file_and_schedule_cursor_resume_without_wind_replay(self):
        import numpy as np
        from maple_syrup import floating_lifecycle_checkpoint as lcp, native_floating_soil as nfs
        plan, f, bridge = self.fixture(precipitation=0.1)
        self.start(bridge, f)
        bridge.execute(plan.scenario.occurrences[0])
        hs.run_native_day(bridge, f, next(plan.days()))
        identity = {"test": "wind_handoff", "scenario": plan.scenario.identity_sha256}
        saved = lcp.save_floating_lifecycle_checkpoint(self.root / "checkpoint", bridge.lineage, f.factory, identity,
            original_topology=f.original_topology, available_bytes_fn=f.available)
        record = bridge.checkpoint_record(native_manifest_sha256=saved.manifest_sha256)
        sha = hashlib.sha256(record.encode()).hexdigest()
        resumed = self.prepare(plan, f)
        with patch.object(nfs, "initialize_native_floating_soil", side_effect=AssertionError("no owner genesis")), \
             patch.object(nfs, "_batch_properties", side_effect=AssertionError("no property reinitialization")):
            loaded = lcp.load_floating_lifecycle_checkpoint(self.root / "checkpoint", f.factory, identity,
                saved.manifest_sha256, original_topology=f.original_topology, available_bytes_fn=f.available)
            resumed.restore(loaded.lineage, loaded.head, record, expected_record_sha256=sha,
                            native_manifest_sha256=loaded.manifest_sha256)
        self.assertEqual(resumed.next_wind_index, 1)
        with self.assertRaises(mh.WindHandoffError):
            resumed.execute(plan.scenario.occurrences[0])
        a = bridge.execute(plan.scenario.occurrences[1])
        b = resumed.execute(plan.scenario.occurrences[1])
        self.assertEqual(dw._bed_digest(a.bed), dw._bed_digest(b.bed))
        self.assertTrue(np.array_equal(a.root.water_state.water_m, b.root.water_state.water_m))
        self.assertTrue(np.array_equal(a.root.history.wetting_m, b.root.history.wetting_m))
        self.assertEqual(a.ledger, b.ledger)
        self.assertEqual(a.t_s.hex(), b.t_s.hex())
        self.assertEqual(bridge.wind_accounting(), resumed.wind_accounting())
        wrong = self.prepare(plan, f)
        with self.assertRaises(mh.WindHandoffError):
            wrong.restore(loaded.lineage, loaded.head, record, expected_record_sha256="0" * 64,
                          native_manifest_sha256=loaded.manifest_sha256)

    def test_fresh_process_native_restart_without_original_bed_or_water_genesis(self):
        import numpy as np
        from maple_syrup import floating_lifecycle_checkpoint as lcp
        plan, f, bridge = self.fixture(precipitation=0.1)
        self.start(bridge, f)
        bridge.execute(plan.scenario.occurrences[0])
        hs.run_native_day(bridge, f, next(plan.days()))
        bridge.execute(plan.scenario.occurrences[1])
        before = hs.physical_record(bridge.lineage.head)
        identity = {"test": "fresh_wind_handoff", "scenario": plan.scenario.identity_sha256}
        saved = lcp.save_floating_lifecycle_checkpoint(self.root / "checkpoint", bridge.lineage, f.factory, identity,
            original_topology=f.original_topology, available_bytes_fn=f.available)
        top = f.original_topology
        capsule = self.root / "static_topology.npz"
        np.savez(capsule, source_z_full=top.source_z_full, active=top.active, friction=top.friction,
                 outlets=np.array([o.as_tuple() for o in top.outlets], dtype=np.int64))
        seal = {"identity": identity, "native_manifest_sha256": saved.manifest_sha256,
                "native_context": bridge._binding.context.record(), "run_digest": bridge.run_digest,
                "static_sha256": hashlib.sha256(capsule.read_bytes()).hexdigest(),
                "topology_sha256": top.topology_sha256, "before": before}
        (self.root / "fresh_seal.json").write_text(json.dumps(seal))
        continuous, audits = hs.run_native_day(bridge, f, list(plan.days())[1])
        self.assertEqual(audits, [])
        # No saved payload leaves this temporary local directory.
        child = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "tests/fresh_wind_handoff_worker.py"), str(self.root)],
                               capture_output=True, text=True, timeout=90)
        self.assertEqual(child.returncode, 0, child.stderr[-4000:])
        self.assertEqual(json.loads((self.root / "fresh_result.json").read_text()), hs.physical_record(continuous))


if __name__ == "__main__":
    unittest.main()
