"""Daily plan and read-only CLI tests (metadata only; nothing here runs a model).

Expected values are derived by hand from the shipped example and small synthetic
calendars, independent of the implementation. Wind YAML needs optional PyYAML.
"""
import datetime
import dataclasses
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from masy import daily_cli  # noqa: E402
from masy.daily_csv import DailyCsvError, load_daily_scenario  # noqa: E402
from masy.daily_schedule import (CLOCK_CONVENTION, DailyPlan, build_daily_plan,  # noqa: E402
                                 select_precipitation)

EXAMPLE = ROOT / "examples" / "daily_scenario" / "scenario.csv"
LAUNCHER = ROOT / "tools" / "masy_daily.py"
needs_yaml = unittest.skipUnless(importlib.util.find_spec("yaml") is not None, "PyYAML not installed")

WEATHER_HEAD = ("# format: masy.daily_weather.v1\n# pet_method: ecohyd_hargreaves\n"
                "# precipitation_threshold_mm: 5.0\n")
HEADER = "date,tmin_c,tmean_c,tmax_c,precipitation_mm,precipitation_intensity_mm_h\n"
WIND = "wind_forcing:\n  blocks:\n    - shear_velocity_m_s: 0.45\n      direction_deg: 90.0\n      duration_s: 600.0\n"


def weather_text(start, days, rain=None):
    first = datetime.date.fromisoformat(start)
    rain = rain or {}
    rows = ""
    for i in range(days):
        p = rain.get(i, 0)
        rows += f"{first + datetime.timedelta(days=i)},18,25,32,{p},{10 if p else 0}\n"
    return WEATHER_HEAD + HEADER + rows


def scenario_text(start, end, events=(), **over):
    keys = {"format": "masy.daily_scenario.v1", "scenario_id": "plan_t", "weather_file": "weather.csv",
            "start_date": start, "end_date": end, "evaporation_enabled": "true", "transpiration_enabled": "false",
            "growth_enabled": "true", "growth_interval_days": "7", "output_interval_days": "1",
            "checkpoint_interval_days": "7", "output_at_event_boundaries": "false"}
    keys.update(over)
    return ("".join(f"# {k}: {v}\n" for k, v in keys.items()) + "date,event_type,event_file\n"
            + "".join(f"{d},wind,{f}\n" for d, f in events))


class PlanCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def build(self, start, end, rain=None, events=(), weather_start=None, weather_days=None, **over):
        ws = weather_start or start
        n = weather_days or ((datetime.date.fromisoformat(end) - datetime.date.fromisoformat(ws)).days + 1)
        (self.dir / "weather.csv").write_text(weather_text(ws, n, rain))
        (self.dir / "wind").mkdir(exist_ok=True)
        (self.dir / "wind" / "a.yaml").write_text(WIND)
        (self.dir / "scenario.csv").write_text(scenario_text(start, end, events, **over))
        return build_daily_plan(load_daily_scenario(self.dir / "scenario.csv"))


class SelectionTests(unittest.TestCase):
    def test_threshold_rule(self):
        self.assertEqual(select_precipitation(0.0, 5.0), "none")
        self.assertEqual(select_precipitation(0.001, 5.0), "recharge_attempt")
        self.assertEqual(select_precipitation(5.0, 5.0), "recharge_attempt")  # equality is recharge
        self.assertEqual(select_precipitation(5.000001, 5.0), "full_storm_required")
        self.assertEqual(select_precipitation(12.0, 5.0), "full_storm_required")
        self.assertEqual(select_precipitation(0.0, 0.0), "none")
        self.assertEqual(select_precipitation(0.5, 0.0), "full_storm_required")

    def test_clock_convention_is_separate_and_truthful(self):
        self.assertTrue(CLOCK_CONVENTION["physical_year_may_exceed_nominal_year"])
        self.assertFalse(CLOCK_CONVENTION["subdaily_interevent_processes"])
        self.assertFalse(CLOCK_CONVENTION["implicit_final_growth_flush"])
        self.assertEqual(CLOCK_CONVENTION["per_day_order"][2], "one native daily soil/E update")


class PlanTests(PlanCase):
    def test_equality_threshold_in_plan(self):
        plan = self.build("2030-06-01", "2030-06-03", rain={0: 5, 1: "5.0000001", 2: 4.9999})
        self.assertEqual([d.precipitation_selection for d in plan],
                         ["recharge_attempt", "full_storm_required", "recharge_attempt"])
        self.assertEqual([d.recharge_may_fall_back_to_full_storm for d in plan], [True, False, True])

    def test_every_date_exists_without_events_or_rain(self):
        plan = self.build("2030-06-01", "2030-06-10", weather_start="2030-05-25", weather_days=30)
        days = list(plan)
        self.assertEqual(len(days), 10)
        self.assertEqual([d.date.day for d in days], list(range(1, 11)))
        self.assertTrue(all(d.winds == () and d.precipitation_selection == "none" for d in days))
        self.assertEqual([d.day_number for d in days], list(range(1, 11)))
        self.assertEqual([d.is_final_day for d in days], [False] * 9 + [True])

    def test_iterator_is_repeatable_and_lazy(self):
        plan = self.build("2030-06-01", "2030-06-05")
        it = iter(plan)
        self.assertEqual(next(it).date, datetime.date(2030, 6, 1))
        self.assertEqual([d.day_number for d in plan], [1, 2, 3, 4, 5])
        self.assertEqual(next(it).date, datetime.date(2030, 6, 2))

    def test_cadences_anchor_to_start_across_year_end(self):
        plan = self.build("2030-12-30", "2031-01-02", growth_interval_days="2", output_interval_days="3",
                          checkpoint_interval_days="4")
        days = list(plan)
        self.assertEqual([d.date.isoformat() for d in days], ["2030-12-30", "2030-12-31", "2031-01-01", "2031-01-02"])
        self.assertEqual([d.growth_due for d in days], [False, True, False, True])
        self.assertEqual([d.output_due for d in days], [False, False, True, False])
        self.assertEqual([d.checkpoint_due for d in days], [False, False, False, True])
        self.assertEqual([d.is_calendar_year_end for d in days], [False, True, False, False])
        self.assertEqual([d.is_calendar_year_start for d in days], [False, False, True, False])

    def test_no_implicit_final_flush_and_long_intervals(self):
        plan = self.build("2030-06-01", "2030-06-05", growth_interval_days="7", output_interval_days="100",
                          checkpoint_interval_days="100")
        days = list(plan)
        self.assertFalse(any(d.growth_due or d.output_due or d.checkpoint_due for d in days))
        self.assertTrue(days[-1].is_final_day)

    def test_growth_disabled_never_due(self):
        plan = self.build("2030-06-01", "2030-06-08", growth_enabled="false", growth_interval_days="2")
        self.assertFalse(any(d.growth_due for d in plan))
        self.assertTrue(all(not d.growth_enabled for d in plan))

    def test_process_flags_follow_settings(self):
        plan = self.build("2030-06-01", "2030-06-02", evaporation_enabled="false", transpiration_enabled="true")
        for d in plan:
            self.assertEqual((d.evaporation_enabled, d.transpiration_enabled), (False, True))

    def test_leap_day_and_full_years(self):
        plan = self.build("2028-02-27", "2028-03-01")
        self.assertEqual([(d.date.isoformat(), d.is_leap_day) for d in plan],
                         [("2028-02-27", False), ("2028-02-28", False), ("2028-02-29", True), ("2028-03-01", False)])
        summary = self.build("2028-01-01", "2029-12-31").summary()
        self.assertEqual(summary["counts"]["days"], 366 + 365)
        self.assertEqual(summary["counts"]["leap_days"], 1)
        self.assertEqual(summary["counts"]["calendar_year_end_days"], 2)
        self.assertEqual(summary["calendar_years_touched"], [2028, 2029])
        self.assertEqual(self.build("2029-01-01", "2029-12-31").summary()["counts"]["leap_days"], 0)
        self.assertEqual(list(self.build("2030-06-01", "2030-06-01"))[0].is_final_day, True)

    def test_record_is_json_and_independent(self):
        plan = self.build("2030-06-01", "2030-06-02", rain={1: 6})
        day = list(plan)[1]
        record = day.record()
        json.dumps(record, allow_nan=False)
        self.assertEqual(record["precipitation_selection"], "full_storm_required")
        self.assertEqual(record["nominal_wet_period_s"], 2160.0)  # 6 mm at 10 mm/h = 0.6 h
        self.assertEqual(record["calendar"]["day_of_year"], 153)  # 151 days through May 31, so 2030-06-02 is day 153
        record["weather"]["tmean_c"] = -1.0
        self.assertEqual(day.weather.tmean_c, 25.0)
        self.assertIsInstance(day.winds, tuple)

    def test_summary_does_not_expose_shared_clock_metadata(self):
        plan = self.build("2030-06-01", "2030-06-02")
        record = plan.summary()
        record["clock"]["per_day_order"].clear()
        self.assertEqual(len(plan.summary()["clock"]["per_day_order"]), 4)

    def test_invalid_plan_input(self):
        with self.assertRaises(DailyCsvError):
            DailyPlan(object())

    @needs_yaml
    def test_hand_built_unsorted_or_uncovered_scenario_refused(self):
        plan = self.build("2030-06-01", "2030-06-03",
                          events=(("2030-06-01", "wind/a.yaml"), ("2030-06-03", "wind/a.yaml")))
        with self.assertRaises(DailyCsvError) as ctx:
            DailyPlan(dataclasses.replace(plan.scenario, occurrences=tuple(reversed(plan.scenario.occurrences))))
        self.assertEqual(ctx.exception.code, "event_dates_out_of_order")
        settings = dataclasses.replace(plan.scenario.settings, end_date=datetime.date(2030, 6, 4))
        with self.assertRaises(DailyCsvError) as ctx:
            DailyPlan(dataclasses.replace(plan.scenario, settings=settings))
        self.assertEqual(ctx.exception.code, "weather_coverage")


@needs_yaml
class ExamplePlanTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_daily_scenario(EXAMPLE)
        self.days = list(build_daily_plan(self.scenario))

    def test_example_days(self):
        self.assertEqual([d.date.isoformat() for d in self.days], [f"2030-06-0{i}" for i in range(1, 8)])
        self.assertEqual([len(d.winds) for d in self.days], [0, 1, 0, 2, 0, 1, 0])
        self.assertEqual([d.precipitation_selection for d in self.days],
                         ["none", "recharge_attempt", "recharge_attempt", "full_storm_required",
                          "none", "recharge_attempt", "full_storm_required"])

    def test_same_date_wind_order_is_row_order(self):
        day4 = self.days[3]
        self.assertEqual([w.reference for w in day4.winds], ["wind/wind_b.yaml", "wind/wind_a.yaml"])
        self.assertEqual([w.index for w in day4.winds], [1, 2])
        self.assertNotEqual(day4.winds[0].occurrence_id, self.days[5].winds[0].occurrence_id)
        # Same file used on days 2, 4 and 6: three distinct identities, one set of bytes.
        a = [w for d in self.days for w in d.winds if w.reference.endswith("wind_a.yaml")]
        self.assertEqual(len(a), 3)
        self.assertEqual(len({w.occurrence_id for w in a}), 3)
        self.assertEqual(len({w.blocks for w in a}), 1)

    def test_example_cadence_flags(self):
        self.assertEqual([d.growth_due for d in self.days], [False] * 6 + [True])
        self.assertEqual([d.output_due for d in self.days], [True] * 7)
        self.assertEqual([d.checkpoint_due for d in self.days], [False] * 6 + [True])
        self.assertEqual([d.event_boundary_output_requested for d in self.days],
                         [False, True, True, True, False, True, True])

    def test_example_summary(self):
        s = build_daily_plan(self.scenario).summary()
        self.assertEqual(s["precipitation_selection_counts"], {"none": 2, "recharge_attempt": 3, "full_storm_required": 2})
        c = s["counts"]
        self.assertEqual((c["days"], c["days_with_wind"], c["wind_occurrences"]), (7, 3, 4))
        self.assertEqual((c["growth_due_days"], c["output_due_days"], c["checkpoint_due_days"]), (1, 7, 1))
        self.assertEqual(c["event_boundary_output_days"], 5)


class CliTests(PlanCase):
    def run_cli(self, *argv):
        out = io.StringIO()
        code = daily_cli.main(list(argv), stdout=out)
        return code, json.loads(out.getvalue())

    @needs_yaml
    def test_inspect_example(self):
        code, doc = self.run_cli("inspect", str(EXAMPLE))
        self.assertEqual(code, 0)
        self.assertEqual(doc["status"], "inspected")
        self.assertEqual(doc["settings"]["scenario_id"], "example_week")
        self.assertEqual(doc["wind_events"]["count"], 4)
        self.assertFalse(doc["wind_events"]["truncated"])
        self.assertEqual(doc["sample_days"], [])
        self.assertEqual(doc["weather"]["days"], 7)
        self.assertEqual(doc["weather"]["precipitation_threshold_mm"], 5.0)
        self.assertEqual(doc["plan"]["counts"]["days"], 7)
        self.assertEqual(len(doc["identity_sha256"]), 64)
        for key in ("executed", "native_models_imported", "physical_transport_occurred",
                    "leaf_adapter_available", "execution_ready"):
            self.assertIs(doc["execution"][key], False)
        text = " ".join(doc["limitations"])
        self.assertIn("no simulation was run", text)
        self.assertIn("LEAF", text)
        self.assertNotIn(str(ROOT), json.dumps(doc))

    @needs_yaml
    def test_sample_days_bounded(self):
        _, doc = self.run_cli("inspect", str(EXAMPLE), "--sample-days", "2")
        self.assertEqual([d["date"] for d in doc["sample_days"]], ["2030-06-01", "2030-06-02"])
        code, doc = self.run_cli("inspect", str(EXAMPLE), "--sample-days", "367")
        self.assertEqual((code, doc["error_code"]), (2, "invalid_arguments"))

    def test_long_scenario_output_is_bounded(self):
        self.build("2028-01-01", "2030-12-31")
        out = io.StringIO()
        self.assertEqual(daily_cli.main(["inspect", str(self.dir / "scenario.csv")], stdout=out), 0)
        self.assertLess(len(out.getvalue()), 12000)
        self.assertNotIn("2029-06-15", out.getvalue())
        self.assertEqual(json.loads(out.getvalue())["plan"]["counts"]["days"], 366 + 365 + 365)

    @needs_yaml
    def test_event_listing_is_capped(self):
        events = tuple(("2030-06-02", "wind/a.yaml") for _ in range(70))
        self.build("2030-06-01", "2030-06-03", events=events)
        _, doc = self.run_cli("inspect", str(self.dir / "scenario.csv"))
        self.assertEqual(doc["wind_events"]["count"], 70)
        self.assertEqual(len(doc["wind_events"]["listed"]), 64)
        self.assertTrue(doc["wind_events"]["truncated"])
        _, sampled = self.run_cli("inspect", str(self.dir / "scenario.csv"), "--sample-days", "3")
        day = sampled["sample_days"][1]
        self.assertEqual(day["wind_occurrence_count"], 70)
        self.assertEqual(len(day["wind_occurrences"]), 64)
        self.assertTrue(day["wind_occurrences_truncated"])

    def test_refusals_have_stable_codes(self):
        self.build("2030-06-01", "2030-06-03")
        path = self.dir / "scenario.csv"
        path.write_text(path.read_text().replace("# growth_interval_days: 7", "# growth_interval_days: 9"))
        code, doc = self.run_cli("inspect", str(path))
        self.assertEqual((code, doc["status"], doc["error_code"]), (2, "refused", "growth_interval_out_of_range"))
        self.assertIs(doc["execution"]["executed"], False)
        code, doc = self.run_cli("inspect", str(self.dir / "missing.csv"))
        self.assertEqual((code, doc["error_code"]), (2, "document_missing"))
        code, doc = self.run_cli("inspect", str(path), "--max-file-bytes", "10")
        self.assertEqual((code, doc["error_code"]), (2, "document_too_large"))
        code, doc = self.run_cli("inspect", str(path), "--max-files", "0")
        self.assertEqual((code, doc["error_code"]), (2, "invalid_arguments"))
        code, doc = self.run_cli()
        self.assertEqual((code, doc["error_code"]), (2, "invalid_arguments"))
        code, doc = self.run_cli("run", str(path))
        self.assertEqual((code, doc["error_code"]), (2, "invalid_arguments"))

    @needs_yaml
    def test_launcher_requires_isolation_and_inspects(self):
        proc = subprocess.run([sys.executable, "-I", "-B", str(LAUNCHER), "inspect", str(EXAMPLE)],
                              capture_output=True, text=True, cwd=tempfile.gettempdir())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        doc = json.loads(proc.stdout)
        self.assertEqual(doc["status"], "inspected")
        self.assertEqual(doc["plan"]["counts"]["wind_occurrences"], 4)
        bad = subprocess.run([sys.executable, str(LAUNCHER), "inspect", str(EXAMPLE)], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertEqual(bad.stdout, "")

    def test_launcher_refusal_exit_code(self):
        proc = subprocess.run([sys.executable, "-I", "-B", str(LAUNCHER), "inspect", str(self.dir / "nope.csv")],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["error_code"], "document_missing")


if __name__ == "__main__":
    unittest.main()
