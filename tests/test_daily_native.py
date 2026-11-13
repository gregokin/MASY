"""Daily connection checks; physical tests require an explicit immutable source pair.

Ordinary discovery runs the metadata/refusal tests without native imports.
Set MASY_DAILY_NATIVE_BUNDLE and MASY_DAILY_NATIVE_SYRUP, with that source
pair already on the isolated interpreter's import path, for actual native CPU
tests. Nothing installs packages or alters an upstream tree.
"""
import datetime
import dataclasses
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from masy.daily_csv import load_daily_scenario
from masy.daily_native import (DailyNativeError, SYRUP_PACKAGE_SHA256,
                               open_water_session, preflight_water_only)

BUNDLE = os.environ.get("MASY_DAILY_NATIVE_BUNDLE")
SYRUP = os.environ.get("MASY_DAILY_NATIVE_SYRUP")


def scenario(path, rain=(0,), intensity=None, *, explicit_pet=0, evaporation=True,
             transpiration=False, growth=False, wind=False, hargreaves=False):
    path.mkdir(parents=True, exist_ok=True)
    start = datetime.date(2028, 2, 28)
    end = start + datetime.timedelta(days=len(rain) - 1)
    heads = {"format": "masy.daily_scenario.v1", "scenario_id": "native_test", "weather_file": "weather.csv",
             "start_date": start, "end_date": end, "evaporation_enabled": str(evaporation).lower(),
             "transpiration_enabled": str(transpiration).lower(), "growth_enabled": str(growth).lower(),
             "growth_interval_days": 7, "output_interval_days": 1, "checkpoint_interval_days": 2,
             "output_at_event_boundaries": "true"}
    text = "".join(f"# {k}: {v}\n" for k, v in heads.items()) + "date,event_type,event_file\n"
    if wind:
        text += f"{end},wind,wind.yaml\n"
        (path / "wind.yaml").write_text("wind_forcing:\n  blocks:\n  - shear_velocity_m_s: 0.4\n    direction_deg: 90.0\n    duration_s: 10.0\n")
    (path / "scenario.csv").write_text(text)
    method = "ecohyd_hargreaves" if hargreaves else "explicit"
    weather = f"# format: masy.daily_weather.v1\n# pet_method: {method}\n# precipitation_threshold_mm: 5\n"
    weather += ("date,tmin_c,tmean_c,tmax_c,precipitation_mm,precipitation_intensity_mm_h\n" if hargreaves else
                "date,tmean_c,pet_mm_day,precipitation_mm,precipitation_intensity_mm_h\n")
    for i, p in enumerate(rain):
        rate = (intensity[i] if intensity is not None else 360) if p else 0
        values = "18,25,32" if hargreaves else f"25,{explicit_pet}"
        weather += f"{start + datetime.timedelta(days=i)},{values},{p},{rate}\n"
    (path / "weather.csv").write_text(weather)
    return load_daily_scenario(path / "scenario.csv")


class Preflight(unittest.TestCase):
    def test_missing_leaf_is_refused_before_native_import(self):
        with tempfile.TemporaryDirectory() as td:
            s = scenario(Path(td), transpiration=True)
            with self.assertRaises(DailyNativeError) as ctx:
                open_water_session(s, setup=None, admission=None, source=None, terrain=None,
                                   control=None, quiet_policy=None, storm_limit_s=1, pet_max_bytes=1)
            self.assertEqual(ctx.exception.code, "leaf_bridge_pending")

    def test_future_wind_is_refused_at_preflight(self):
        if importlib.util.find_spec("yaml") is None:
            self.skipTest("optional YAML reader absent")
        with tempfile.TemporaryDirectory() as td:
            s = scenario(Path(td), rain=(0, 0), wind=True)
            with self.assertRaises(DailyNativeError) as ctx:
                preflight_water_only(s)
            self.assertEqual(ctx.exception.code, "wind_bridge_unqualified")

    def test_water_only_and_invalid_type(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(preflight_water_only(scenario(Path(td))))
        with self.assertRaises(DailyNativeError) as ctx:
            preflight_water_only(None)
        self.assertEqual(ctx.exception.code, "scenario")


@unittest.skipUnless(BUNDLE and SYRUP, "explicit native Phase 1.5 CPU source pair not selected")
class NativePhysical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import yaml
        from maple_syrup import native_case_profile as ncp, native_dense_profile as nd
        from maple_syrup import layered_case as lc, layered_event as ev, weather_pet as wp
        from maple_syrup import lifecycle_checkpoint_v3 as ck, layered_checkpoint as limits
        from maple_syrup.subsurface import default_host_available_bytes
        cls.np, cls.yaml, cls.ncp, cls.nd, cls.lc, cls.wp, cls.ck = np, yaml, ncp, nd, lc, wp, ck
        cls.host_reader = staticmethod(default_host_available_bytes)
        cls.limits = limits.CheckpointLimits(1 << 28, 1 << 28, 0, 1 << 28)
        cls.source = ncp.verify_source_profile(BUNDLE, expected_syrup_digest=SYRUP_PACKAGE_SHA256)
        cls.public = Path(SYRUP) / "docs/manuals/examples/native_leaf_calendar_cli"
        cls.control = ev.LayeredEventControl(max_dt_s=2, max_steps=100000, max_recession_s=20000)
        cls.quiet = ev.LayeredQuietPolicy()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def prepare(self, *, theta=0.20, bottom="open", flat=False):
        case = self.work / "case"
        shutil.copytree(self.public / "case", case)
        for file in case.rglob("*"):
            if file.is_file():
                file.chmod(0o644)  # owned synthetic input, copied from a read-only source archive
        if flat:
            file = case / "case.yaml"
            cfg = self.yaml.safe_load(file.read_text())
            cfg["topography"]["perturbation"]["relief_m"] = 0.0
            file.write_text(self.yaml.safe_dump(cfg))
        adm = self.ncp.admit_package_case(self.source, case)
        prep = self.nd.prepare_native_dense_case(adm, caller_reserve_bytes=1 << 20)
        cfg = self.yaml.safe_load((self.public / "layered_water.yaml").read_text())
        cfg["soil"]["bottom"] = bottom
        cfg["initial"]["soil_moisture"]["by_layer"] = [theta, theta]
        if flat:
            cfg["domain"]["outlets"] = []
        layered = self.work / "layered.yaml"
        layered.write_text(self.yaml.safe_dump(cfg))
        setup = self.lc.prepare_layered_case(prep, self.lc.load_layered_config(layered),
                                             caller_reserve_bytes=1 << 20, lifecycle_extra_reserve_bytes=1 << 20)
        # Explicit illustrative fixture terrain, not a derived MAPLE aspect map.
        terrain = self.wp.terrain_factor(self.np.zeros(setup.lifecycle_operator.n_cells, dtype=self.np.int32),
                                         self.np.zeros(setup.lifecycle_operator.n_cells), adm.latitude_deg)
        return setup, adm, terrain

    def open(self, s, setup, adm, terrain, **kw):
        args = dict(setup=setup, admission=adm, source=self.source, terrain=terrain,
                    control=self.control, quiet_policy=self.quiet, storm_limit_s=20000, pet_max_bytes=1 << 20)
        args.update(kw)
        return open_water_session(s, **args)

    def assert_closed(self, record):
        audit = record["native_audit"]
        self.assertLessEqual(abs(audit["residual_m3"]), audit["bound_m3"])
        self.assertEqual(audit["transpiration_m3"], 0)

    def test_daily_equality_dry_leap_day_and_restart_once(self):
        s = scenario(self.work / "forcing", rain=(5, 0, 2))
        setup, adm, terrain = self.prepare()
        session = self.open(s, setup, adm, terrain)
        first = session.advance().record()
        self.assertEqual(first["native_receipt"]["selection"], "recharge")
        self.assertEqual(first["native_receipt"]["branch"], "recharge")
        self.assertAlmostEqual(first["native_audit"]["rain_m3"], 0.005 * 24 * 0.25 ** 2)
        self.assertEqual(session.state.t_s, 86400)
        self.assert_closed(first)
        checkpoint = self.work / "checkpoint"
        session.save_checkpoint(checkpoint, limits=self.limits, host_available_bytes_fn=self.host_reader)
        restored = self.open(s, setup, adm, terrain, checkpoint_path=checkpoint,
                             checkpoint_limits=self.limits, host_available_bytes_fn=self.host_reader)
        self.assertEqual(restored.state.daily.ordinal, 1)
        for expected_date in ("2028-02-29", "2028-03-01"):
            a, b = session.advance(), restored.advance()
            self.assertEqual(a.record()["date"], expected_date)
            self.assertEqual(a.record(), b.record())
            self.assert_closed(a.record())
        self.assertEqual(session.state.daily.ordinal, 3)
        self.assertEqual(session.state.t_s, 3 * 86400)
        self.assertAlmostEqual(b.record()["native_audit"]["rain_m3"], 0.007 * 24 * 0.25 ** 2)
        with self.assertRaises(StopIteration):
            restored.advance()

    def test_gravity_open_bottom_drains(self):
        s = scenario(self.work / "forcing", rain=(2,))
        setup, adm, terrain = self.prepare(theta=0.30)
        record = self.open(s, setup, adm, terrain).advance().record()
        self.assertGreater(record["native_receipt"]["drainage_m3"], 0)
        self.assert_closed(record)

    def test_closed_bottom_retains_above_field_capacity(self):
        s = scenario(self.work / "forcing", rain=(2,))
        setup, adm, terrain = self.prepare(theta=0.25, bottom="closed")
        record = self.open(s, setup, adm, terrain).advance().record()
        self.assertEqual(record["native_receipt"]["drainage_m3"], 0)
        self.assertEqual(record["native_receipt"]["branch"], "recharge")
        self.assert_closed(record)

    def test_capacity_fallback_full_original_rain_and_extra_clock(self):
        s = scenario(self.work / "forcing", rain=(1,), explicit_pet=8.64)
        setup, adm, terrain = self.prepare(theta=0.50, bottom="closed", flat=True)
        session = self.open(s, setup, adm, terrain)
        record = session.advance().record()
        self.assertEqual(record["native_receipt"]["branch"], "storm_fallback")
        self.assertEqual(record["native_receipt"]["capacity_failing_columns"], 24)
        self.assertAlmostEqual(record["native_audit"]["rain_m3"], 0.001 * 24 * 0.25 ** 2)
        self.assertGreater(session.state.t_s, 86400)
        self.assert_closed(record)

    def test_daily_router_refusal_keeps_boundary_and_evidence(self):
        s = scenario(self.work / "forcing", rain=(1,), explicit_pet=0.864)
        setup, adm, terrain = self.prepare(theta=0.50, bottom="closed", flat=True)
        session = self.open(s, setup, adm, terrain)
        parent = session.state
        with self.assertRaises(DailyNativeError) as ctx:
            session.advance()
        self.assertEqual(ctx.exception.code, "router_step_rejected")
        self.assertIs(session.state, parent)
        self.assertEqual(parent.daily.ordinal, 0)
        self.assertEqual(ctx.exception.active_state.role, "storm")
        with self.assertRaises(DailyNativeError) as stopped:
            session.advance()
        self.assertEqual(stopped.exception.code, "session_stopped")

    def test_direct_storm_incomplete_is_not_a_day(self):
        s = scenario(self.work / "forcing", rain=(6,))
        setup, adm, terrain = self.prepare()
        session = self.open(s, setup, adm, terrain, storm_limit_s=0.01)
        parent = session.state
        with self.assertRaises(DailyNativeError) as ctx:
            session.advance()
        self.assertEqual(ctx.exception.code, "storm_incomplete")
        self.assertIs(session.state, parent)
        self.assertEqual(parent.daily.ordinal, 0)
        self.assertEqual(ctx.exception.active_state.daily.ordinal, 0)

    def test_direct_storm_completes_before_one_daily_update(self):
        s = scenario(self.work / "forcing", rain=(6,))
        setup, adm, terrain = self.prepare()
        session = self.open(s, setup, adm, terrain)
        record = session.advance().record()
        receipt = record["native_receipt"]
        self.assertEqual((receipt["selection"], receipt["branch"]), ("storm", "storm"))
        self.assertEqual(receipt["t_end_s"] - receipt["t_start_s"], 86400)
        self.assertGreater(receipt["t_start_s"], receipt["occurrence_start_s"])
        self.assertAlmostEqual(record["native_audit"]["rain_m3"], 0.006 * 24 * 0.25 ** 2)
        self.assert_closed(record)

    def test_terrain_and_workspace_refused_before_lineage(self):
        s = scenario(self.work / "forcing")
        setup, adm, terrain = self.prepare()
        parent = setup.state
        for kw, expected in (({"terrain": None}, "terrain"), ({"pet_max_bytes": 1}, "limits"),
                             ({"source": None}, "native_source"),
                             ({"control": dataclasses.replace(self.control, max_steps=0)}, "native_controls")):
            with self.subTest(expected=expected), self.assertRaises(DailyNativeError) as ctx:
                args = dict(setup=setup, admission=adm, source=self.source, terrain=terrain, control=self.control,
                            quiet_policy=self.quiet, storm_limit_s=20000, pet_max_bytes=1 << 20)
                args.update(kw)
                open_water_session(s, **args)
            self.assertEqual(ctx.exception.code, expected)
        self.assertIs(setup.state, parent)
        self.assertIsNone(parent.daily)

    def test_interrupt_memory_and_source_failures_stop_at_boundary(self):
        s = scenario(self.work / "forcing")
        setup, adm, terrain = self.prepare()
        for mode in ("interrupt", "memory", "source"):
            with self.subTest(mode=mode):
                session = self.open(s, setup, adm, terrain)
                parent = session.state
                if mode == "interrupt":
                    def interrupted():
                        raise KeyboardInterrupt()
                    session._host_available = interrupted
                    with self.assertRaises(KeyboardInterrupt):
                        session.advance()
                    expected = "interrupted"
                elif mode == "memory":
                    session._host_available = lambda: 0
                    with self.assertRaises(DailyNativeError) as ctx:
                        session.advance()
                    expected = "pet_workspace_unavailable"
                    self.assertEqual(ctx.exception.code, expected)
                else:
                    with mock.patch("masy.daily_native._source_identity", return_value={}):
                        with self.assertRaises(DailyNativeError) as ctx:
                            session.advance()
                    expected = "masy_source_changed"
                    self.assertEqual(ctx.exception.code, expected)
                self.assertIs(session.state, parent)
                self.assertEqual(session.failure["code"], expected)
                self.assertIsInstance(session.failure, dict)
                with self.assertRaises(DailyNativeError) as ctx:
                    session.advance()
                self.assertEqual(ctx.exception.code, "session_stopped")

    def test_record_failure_keeps_boundary_and_closed_candidate_evidence(self):
        s = scenario(self.work / "forcing")
        setup, adm, terrain = self.prepare()
        session = self.open(s, setup, adm, terrain)
        parent = session.state
        with mock.patch("masy.daily_native._json", side_effect=TypeError("injected receipt failure")):
            with self.assertRaises(DailyNativeError) as ctx:
                session.advance()
        self.assertEqual(ctx.exception.code, "record_failed")
        self.assertIs(session.state, parent)
        self.assertEqual(parent.daily.ordinal, 0)
        self.assertEqual(ctx.exception.active_state.role, "boundary")
        self.assertEqual(ctx.exception.active_state.daily.ordinal, 1)

    def test_interrupted_or_exhausted_iterator_is_not_schedule_completion(self):
        s = scenario(self.work / "forcing")
        setup, adm, terrain = self.prepare()
        for mode in ("interrupt", "exhaustion"):
            with self.subTest(mode=mode):
                session = self.open(s, setup, adm, terrain)
                parent = session.state
                if mode == "interrupt":
                    def interrupted():
                        raise KeyboardInterrupt()
                        yield  # make this an interrupted generator, not a regular function
                    session._days = interrupted()
                    with self.assertRaises(KeyboardInterrupt):
                        session.advance()
                    expected = "interrupted"
                else:
                    session._days = iter(())
                    with self.assertRaises(DailyNativeError) as ctx:
                        session.advance()
                    expected = "cursor"
                    self.assertEqual(ctx.exception.code, expected)
                self.assertEqual(session.failure["code"], expected)
                self.assertIs(session.state, parent)
                with self.assertRaises(DailyNativeError) as ctx:
                    session.advance()
                self.assertEqual(ctx.exception.code, "session_stopped")

    def test_extreme_late_rain_refused_before_daily_lineage(self):
        # P is representable in mm and P/I duration is finite, but its native
        # conversion to metres cannot be represented. Only SYRUP decides that.
        s = scenario(self.work / "forcing", rain=(0, 1e-321), intensity=(0, 1e-300))
        setup, adm, terrain = self.prepare()
        with self.assertRaises(DailyNativeError) as ctx:
            self.open(s, setup, adm, terrain)
        self.assertEqual(ctx.exception.code, "native_rain_plan")
        self.assertIsNone(setup.state.daily)

    def test_hargreaves_uses_case_latitude_native_pet(self):
        s = scenario(self.work / "forcing", hargreaves=True)
        setup, adm, terrain = self.prepare()
        record = self.open(s, setup, adm, terrain).advance().record()
        expected = self.wp.potential_demand(self.wp.DailyWeather(datetime.date(2028, 2, 28), 18, 25, 32),
                                           adm.latitude_deg, method="ecohyd_hargreaves")
        self.assertEqual(record["reference_pet_mm_day"], expected.pet_mm_day)
        self.assertGreater(record["native_audit"]["soil_evaporation_m3"], 0)
        self.assert_closed(record)

    def test_disabled_evaporation_and_changed_weather_restart_refused(self):
        s = scenario(self.work / "forcing", explicit_pet=8.64, evaporation=False)
        setup, adm, terrain = self.prepare()
        session = self.open(s, setup, adm, terrain)
        record = session.advance().record()
        self.assertEqual(record["native_audit"]["soil_evaporation_m3"], 0)
        self.assertEqual(record["native_audit"]["surface_evaporation_m3"], 0)
        session.save_checkpoint(self.work / "checkpoint", limits=self.limits, host_available_bytes_fn=self.host_reader)
        changed = scenario(self.work / "forcing", explicit_pet=8.63, evaporation=False)
        with self.assertRaises(self.ck.LifecycleCheckpointError) as ctx:
            self.open(changed, setup, adm, terrain, checkpoint_path=self.work / "checkpoint",
                      checkpoint_limits=self.limits, host_available_bytes_fn=self.host_reader)
        self.assertIn("identity", str(ctx.exception).lower())
