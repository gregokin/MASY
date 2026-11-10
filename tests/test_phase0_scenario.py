"""Phase 0: strict document I/O, dated calendar, event files and scenario binding (stdlib only)."""

from __future__ import annotations

import datetime as dt
import json
import os
import unittest
from fractions import Fraction
from pathlib import Path

import _phase0_support as ph
from _support import TempDirCase

from masy.calendar import CalendarError, CalendarStart, noleap_day_index
from masy.event_files import (
    EventFileError, IntereventFile, RainEventFile, WindEventFile, parse_event_file, process_gate,
)
from masy.scenario import ScenarioError, load_scenario
from masy.scenario_io import IOLimits, ScenarioIOError, parse_strict_json, parse_strict_yaml, read_bounded, relative_reference


class StrictDocumentTests(TempDirCase):
    def assertIO(self, code: str, call, *args, **kwargs) -> None:
        with self.assertRaises(ScenarioIOError) as ctx:
            call(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)

    def test_strict_json_refusals(self) -> None:
        limits = IOLimits(max_depth=4, max_entries=64, max_text_chars=32)
        self.assertEqual(parse_strict_json(b'{"a": [1, 2.5, true, null, "x"]}', limits), {"a": [1, 2.5, True, None, "x"]})
        self.assertIO("duplicate_key", parse_strict_json, b'{"a": 1, "a": 2}', limits)
        self.assertIO("nonfinite_number", parse_strict_json, b'{"a": NaN}', limits)
        self.assertIO("nonfinite_number", parse_strict_json, b'[Infinity]', limits)
        self.assertIO("depth_quota_exceeded", parse_strict_json, b'[[[[[0]]]]]', limits)
        self.assertIO("text_quota_exceeded", parse_strict_json, b'{"k": "' + b"x" * 40 + b'"}', limits)
        self.assertIO("entry_quota_exceeded", parse_strict_json, b"[" + b"0," * 70 + b"0]", limits)
        self.assertIO("invalid_json", parse_strict_json, b'{"a": 1} trailing', limits)
        self.assertIO("document_too_large", parse_strict_json, b"[" + b"0," * 100 + b"0]", IOLimits(max_file_bytes=16))

    def test_malformed_giant_numbers_refuse_cleanly(self) -> None:
        limits = IOLimits()
        self.assertIO("invalid_json", parse_strict_json, b'{"a": ' + b"9" * 5000 + b"}", limits)  # beyond the int digit limit
        self.assertIO("nonfinite_number", parse_strict_json, b'{"a": 1e400}', limits)  # overflows to inf
        self.assertIO("invalid_yaml", parse_strict_yaml, b"a: " + b"9" * 5000 + b"\n", limits)
        self.assertIO("nonfinite_number", parse_strict_yaml, b"a: 1.0e+400\n", limits)  # YAML float form; overflows to inf
        self.assertEqual(parse_strict_yaml(b"a: 1e400\n", limits), {"a": "1e400"})  # not a YAML float: a plain string

    def test_yaml_depth_and_entry_quotas_apply_during_composition(self) -> None:
        # the review reproducer: 400 nested flow sequences must refuse with the configured code, never RecursionError
        deep = ("[" * 400 + "0" + "]" * 400).encode()
        self.assertIO("depth_quota_exceeded", parse_strict_yaml, deep, IOLimits(max_depth=16))
        self.assertIO("depth_quota_exceeded", parse_strict_yaml, deep, IOLimits())  # default 32 too
        block = b"".join(b" " * i + b"- \n" for i in range(0)) or b"a:\n b:\n  c:\n   d: [[[1]]]\n"
        self.assertIO("depth_quota_exceeded", parse_strict_yaml, block, IOLimits(max_depth=5))
        self.assertEqual(parse_strict_yaml(block, IOLimits(max_depth=7)), {"a": {"b": {"c": {"d": [[[1]]]}}}})
        self.assertIO("entry_quota_exceeded", parse_strict_yaml, b"[" + b"0, " * 100 + b"0]", IOLimits(max_entries=50))
        self.assertIO("entry_quota_exceeded", parse_strict_yaml, b"a: [" + b"{x: 1}, " * 40 + b"{x: 1}]\n", IOLimits(max_entries=60))
        self.assertIO("text_quota_exceeded", parse_strict_yaml, b"a: " + b"x" * 100 + b"\n", IOLimits(max_text_chars=50))

    def test_strict_yaml_is_a_json_subset(self) -> None:
        limits = IOLimits()
        self.assertEqual(parse_strict_yaml(b"a: 1\nb: [1, 2.5, true, null, x]\n", limits), {"a": 1, "b": [1, 2.5, True, None, "x"]})
        self.assertIO("duplicate_key", parse_strict_yaml, b"a: 1\na: 2\n", limits)
        self.assertIO("non_string_key", parse_strict_yaml, b"1: x\n", limits)
        self.assertIO("yaml_alias_refused", parse_strict_yaml, b"a: &x 1\nb: *x\n", limits)
        self.assertIO("nonfinite_number", parse_strict_yaml, b"a: .nan\n", limits)
        self.assertIO("unsupported_leaf", parse_strict_yaml, b"a: 2026-03-01\n", limits)  # YAML date is not JSON
        self.assertIO("unsupported_leaf", parse_strict_yaml, b"a: !!binary aGk=\n", limits)

    def test_read_bounded_refuses_symlink_oversize_missing(self) -> None:
        target = self.tmp / "doc.json"
        target.write_bytes(b"{}")
        link = self.tmp / "link.json"
        os.symlink(target, link)
        self.assertIO("document_symlink", read_bounded, link, 1024)
        self.assertIO("document_too_large", read_bounded, target, 1)
        self.assertIO("document_missing", read_bounded, self.tmp / "absent.json", 1024)
        data, pin = read_bounded(target, 1024)
        self.assertEqual((data, pin["size"], pin["sha256"]), (b"{}", 2, ph.sha(b"{}")))

    def test_relative_reference_policy(self) -> None:
        limits = IOLimits(max_path_chars=64)
        (self.tmp / "events").mkdir()
        self.assertEqual(relative_reference("events/a.json", self.tmp, limits, "ref"), self.tmp / "events" / "a.json")
        for bad, code in (("../a.json", "invalid_reference"), ("/abs/a.json", "invalid_reference"), ("", "invalid_reference"),
                          ("a\nb.json", "invalid_reference"), ("events//a.json", "invalid_reference"),
                          ("x" * 70, "invalid_reference"), (5, "invalid_reference")):
            with self.subTest(bad=bad):
                self.assertIO(code, relative_reference, bad, self.tmp, limits, "ref")
        outside = self.tmp.parent / f"{self.tmp.name}-outside"
        outside.mkdir()
        try:
            os.symlink(outside, self.tmp / "escape")
            self.assertIO("reference_escapes_base", relative_reference, "escape/a.json", self.tmp, limits, "ref")
        finally:
            outside.rmdir()


class CalendarTests(unittest.TestCase):
    def assertCal(self, code: str, call, *args, **kwargs) -> None:
        with self.assertRaises(CalendarError) as ctx:
            call(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)

    def test_start_parsing_is_strict(self) -> None:
        start = CalendarStart.parse("2028-02-28T23:00:00-07:00")
        self.assertEqual(start.offset_text, "-07:00")
        self.assertEqual(start.record()["start"], "2028-02-28T23:00:00-07:00")
        for bad in ("2028-02-28T23:00:00Z", "2028-02-28 23:00:00+00:00", "2028-02-30T00:00:00+00:00",
                    "2028-02-28T23:00:00.5+00:00", "2028-02-28T23:00:00-00:00", "2028-02-28T23:00:00+24:00", 5):
            with self.subTest(bad=bad):
                with self.assertRaises(CalendarError):
                    CalendarStart.parse(bad)
        for table, code in (({"start": "2028-01-01T00:00:00+00:00", "convention": "julian", "leap_seconds": "none"},
                             "unsupported_calendar_convention"),
                            ({"start": "2028-01-01T00:00:00+00:00", "convention": "proleptic_gregorian", "leap_seconds": "utc"},
                             "unsupported_leap_seconds"),
                            ({"start": "2028-01-01T00:00:00+00:00"}, "invalid_calendar")):
            with self.subTest(code=code):
                self.assertCal(code, CalendarStart.from_input, table)

    def test_exact_instants_cross_leap_day_and_year_boundary(self) -> None:
        start = CalendarStart.parse("2027-12-31T23:59:59+00:00")
        new_year = start.instant(Fraction(1))
        self.assertEqual((new_year.record()["iso"], new_year.day_of_year, new_year.year), ("2028-01-01T00:00:00+00:00", 1, 2028))
        leap = start.instant(Fraction(1 + 59 * 86400))
        self.assertEqual((leap.record()["date"], leap.day_of_year, leap.record()["days_in_year"]), ("2028-02-29", 60, 366))
        third = start.instant(Fraction(1, 3))
        self.assertEqual((third.record()["iso"], third.record()["subsecond"]), ("2027-12-31T23:59:59+00:00", "1/3"))
        self.assertFalse(third.at_midnight)
        self.assertTrue(new_year.at_midnight)
        self.assertCal("negative_elapsed", start.instant, Fraction(-1))
        self.assertCal("invalid_elapsed", start.instant, 1.0)
        # 2028 is a leap year: 366 days after 2027-12-31 is still 2028-12-31; 2029 is not: 365 days after 2028-12-31 is 2029-12-31
        self.assertEqual(CalendarStart.parse("2027-12-31T00:00:00+00:00").instant(Fraction(366 * 86400)).record()["date"], "2028-12-31")
        self.assertEqual(CalendarStart.parse("2027-12-31T00:00:00+00:00").instant(Fraction(367 * 86400)).record()["date"], "2029-01-01")
        self.assertEqual(CalendarStart.parse("2028-12-31T00:00:00+00:00").instant(Fraction(365 * 86400)).record()["date"], "2029-12-31")
        self.assertEqual(CalendarStart.parse("2028-12-31T00:00:00+00:00").instant(Fraction(366 * 86400)).record()["date"], "2030-01-01")

    def test_dates_touched_and_whole_days(self) -> None:
        start = CalendarStart.parse("2028-02-28T00:00:00+00:00")
        day = Fraction(86400)
        self.assertEqual(start.dates_touched(Fraction(0), 2 * day), (dt.date(2028, 2, 28), dt.date(2028, 2, 29)))
        self.assertEqual(start.dates_touched(Fraction(0), 2 * day + 1), (dt.date(2028, 2, 28), dt.date(2028, 2, 29), dt.date(2028, 3, 1)))
        self.assertEqual(start.dates_touched(day + 3600, day + 7200), (dt.date(2028, 2, 29),))
        self.assertEqual(start.dates_touched(day, day), ())
        self.assertEqual(start.whole_days(Fraction(0), 2 * day), 2)
        self.assertIsNone(start.whole_days(Fraction(3600), 3600 + day))
        self.assertIsNone(start.whole_days(Fraction(0), day + 1))
        self.assertCal("reversed_interval", start.dates_touched, day, Fraction(0))

    def test_elapsed_of_requires_same_offset_and_no_past(self) -> None:
        start = CalendarStart.parse("2028-01-01T00:00:00+01:00")
        self.assertEqual(start.elapsed_of("2028-01-02T00:00:00+01:00", "x"), Fraction(86400))
        self.assertCal("offset_mismatch", start.elapsed_of, "2028-01-02T00:00:00+00:00", "x")
        self.assertCal("before_calendar_start", start.elapsed_of, "2027-12-31T00:00:00+01:00", "x")

    def test_noleap_day_index_mapping(self) -> None:
        self.assertEqual(noleap_day_index(dt.date(2027, 1, 1)), 0)
        self.assertEqual(noleap_day_index(dt.date(2027, 12, 31)), 364)
        self.assertEqual(noleap_day_index(dt.date(2028, 2, 28)), 58)
        self.assertEqual(noleap_day_index(dt.date(2028, 3, 1)), 59)
        self.assertEqual(noleap_day_index(dt.date(2028, 12, 31)), 364)
        self.assertCal("noleap_leap_day", noleap_day_index, dt.date(2028, 2, 29))


class EventFileTests(unittest.TestCase):
    def assertEvent(self, code: str, call, *args) -> None:
        with self.assertRaises(EventFileError) as ctx:
            call(*args)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)

    def test_wind_keeps_complete_native_record(self) -> None:
        event = parse_event_file(ph.wind())
        self.assertIs(type(event), WindEventFile)
        self.assertEqual([b.index for b in event.blocks], [0, 1])
        offsets, roles = event.planned_edges()
        self.assertEqual(offsets, [Fraction(0), Fraction(1800), Fraction(3600)])
        self.assertEqual(roles, ["event_start", "original_block_0_end", "wind_end"])
        self.assertEqual(event.minimum_duration_s, event.maximum_duration_s)
        record = event.record()
        self.assertEqual(record["wind_forcing"]["blocks"][1]["direction_deg"]["hex"], (250.0).hex())
        for bad in (ph.wind(blocks=()), ph.wind(blocks=((0.4, 360.0, 10.0),)), ph.wind(blocks=((0.4, 90.0, 0.0),)),
                    ph.wind(blocks=((-0.0, 90.0, 10.0),)), ph.wind(blocks=((True, 90.0, 10.0),))):
            with self.subTest(bad=bad):
                with self.assertRaises(EventFileError):
                    parse_event_file(bad)
        extra = ph.wind()
        extra["wind_forcing"]["blocks"][0]["start_time_s"] = 0.0  # MAPLE refuses authored times; so does this schema
        self.assertEvent("invalid_keys", parse_event_file, extra)
        split = ph.wind()
        split["first_block"] = 0
        self.assertEvent("invalid_keys", parse_event_file, split)

    def test_rain_reuses_native_schedule_validation(self) -> None:
        event = parse_event_file(ph.rain())
        self.assertIs(type(event), RainEventFile)
        offsets, roles = event.planned_edges()
        self.assertEqual(offsets, [Fraction(0), Fraction(600), Fraction(1800), Fraction(9000)])
        self.assertEqual(roles[-2:], ["rainfall_end", "recession_guard_end"])
        self.assertEqual((event.minimum_duration_s, event.maximum_duration_s), (Fraction(1800), Fraction(9000)))
        self.assertEvent("invalid_rain", parse_event_file, ph.rain(intervals=((600.0, 0.0),)))  # dry rain refuses
        self.assertEvent("out_of_domain", parse_event_file, ph.rain(guard=0.0))
        self.assertEvent("negative_zero", parse_event_file, ph.rain(guard=-0.0))
        self.assertEvent("out_of_domain", parse_event_file, ph.rain(intervals=((0.0, 5.0),)))
        self.assertEvent("nonfinite_number", parse_event_file, ph.rain(intervals=((600.0, float("nan")),)))
        self.assertEvent("invalid_number", parse_event_file, ph.rain(intervals=((600.0, True),)))
        bad = ph.rain()
        bad["water_profile_id"] = ""
        self.assertEvent("invalid_water_profile", parse_event_file, bad)

    def test_interevent_gate_is_closed_and_explicit(self) -> None:
        event = parse_event_file(ph.interevent(86400, growth=False))
        self.assertIs(type(event), IntereventFile)
        self.assertEqual(event.planned_edges(), ([Fraction(0), Fraction(86400)], ["event_start", "interevent_end"]))
        self.assertEqual(process_gate(event), {"evaporation": True, "transpiration": True, "growth": False})
        self.assertEqual(process_gate(parse_event_file(ph.wind())), {"evaporation": False, "transpiration": False, "growth": False})
        self.assertEqual(process_gate(parse_event_file(ph.rain())), {"evaporation": False, "transpiration": False, "growth": False})
        missing = ph.interevent()
        del missing["processes"]["growth"]
        self.assertEvent("invalid_keys", parse_event_file, missing)
        truthy = ph.interevent()
        truthy["processes"]["growth"] = 1
        self.assertEvent("invalid_process_gate", parse_event_file, truthy)
        self.assertEvent("out_of_domain", parse_event_file, ph.interevent(duration_s=0))
        self.assertEvent("out_of_domain", parse_event_file, ph.interevent(duration_s=-5))
        self.assertEvent("invalid_schema", parse_event_file, {"schema": "masy.other.v1"})
        self.assertEvent("invalid_event_id", parse_event_file, ph.interevent(event_id="Bad Id"))

    def test_digests_are_lossless_and_content_bound(self) -> None:
        a, b = parse_event_file(ph.interevent()), parse_event_file(ph.interevent(duration_s=172800.0))
        self.assertEqual(a.digest, b.digest)  # 172800 and 172800.0 are the same binary64
        self.assertNotEqual(a.digest, parse_event_file(ph.interevent(growth=False)).digest)
        self.assertNotEqual(parse_event_file(ph.wind()).digest, parse_event_file(ph.wind(blocks=((0.42, 270.0, 1800.0),))).digest)


class LibraryFreeScenarioTests(TempDirCase):
    def test_weather_free_scenario_and_reuse(self) -> None:
        events = ph.quiet_events() + [("q2", "events/quiet.json", ph.interevent(3600.0, False, False, False, event_id="quiet"), None)]
        scenario = load_scenario(ph.build_tree(self.tmp, events, weather_doc=None))
        self.assertIsNone(scenario.weather)
        self.assertEqual([o.kind for o in scenario.occurrences], ["wind", "interevent", "interevent"])
        self.assertEqual(scenario.occurrences[1].file_sha256, scenario.occurrences[2].file_sha256)
        self.assertEqual(scenario.occurrences[1].event.digest, scenario.occurrences[2].event.digest)
        with self.assertRaises(ScenarioError) as ctx:
            load_scenario(ph.build_tree(self.tmp / "needs", ph.standard_events(), weather_doc=None))
        self.assertEqual(ctx.exception.code, "weather_required")


@ph.requires_library
class ScenarioTests(TempDirCase):
    def assertScenario(self, code: str, path: Path, **kwargs) -> None:
        with self.assertRaises(ScenarioError) as ctx:
            load_scenario(path, **kwargs)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)

    def test_reference_reuse_with_distinct_occurrences(self) -> None:
        scenario = load_scenario(ph.build_tree(self.tmp, ph.standard_events(), weather_doc=ph.weather(ph.WEEK)))
        self.assertEqual([o.kind for o in scenario.occurrences], ["interevent", "wind", "interevent", "rain", "interevent", "wind"])
        self.assertEqual([o.occurrence_id for o in scenario.occurrences][:2], ["interevent1", "wind1"])
        first, third = scenario.occurrences[0], scenario.occurrences[4]
        self.assertEqual((first.file, first.file_sha256, first.event.digest), (third.file, third.file_sha256, third.event.digest))
        self.assertEqual(scenario.kinds_present, frozenset({"interevent", "wind", "rain"}))
        self.assertEqual(scenario.record()["events"][1]["event"]["wind_forcing"]["blocks"][0]["duration_s"]["hex"], (1800.0).hex())
        self.assertEqual(len(scenario.digest), 64)

    def test_source_hash_binds_every_file_deterministically(self) -> None:
        path = ph.build_tree(self.tmp, ph.standard_events(), weather_doc=ph.weather(ph.WEEK))
        before = load_scenario(path).digest
        self.assertEqual(load_scenario(path).digest, before)  # deterministic
        wind_path = self.tmp / "events" / "wind.json"
        original = wind_path.read_bytes()
        wind_path.write_bytes(original.replace(b"1800.0", b"1801.0", 1))
        self.assertScenario("event_source_hash_mismatch", path)
        wind_path.write_bytes(original + b"\n")  # same parsed content, different bytes: still a mismatch
        self.assertScenario("event_source_hash_mismatch", path)
        wind_path.write_bytes(original)
        self.assertEqual(load_scenario(path).digest, before)
        # re-declaring the new hash yields a different scenario digest
        changed = original.replace(b"1800.0", b"1801.0", 1)
        wind_path.write_bytes(changed)
        doc = json.loads(path.read_text())
        for row in doc["events"]:
            if row["file"] == "events/wind.json":
                row["sha256"] = ph.sha(changed)
        path.write_text(json.dumps(doc))
        self.assertNotEqual(load_scenario(path).digest, before)
        # weather bytes are bound too
        weather_path = self.tmp / "weather.json"
        weather_path.write_bytes(weather_path.read_bytes() + b"\n")
        self.assertScenario("weather_source_hash_mismatch", path)

    def test_strict_scenario_schema(self) -> None:
        base = ph.standard_events()
        self.assertScenario("duplicate_occurrence_id", ph.build_tree(self.tmp / "a", base[:1] + base[:1], weather_doc=ph.weather(ph.WEEK)))
        self.assertScenario("invalid_keys", ph.build_tree(self.tmp / "b", base, weather_doc=ph.weather(ph.WEEK), extra={"latitude_deg": 32.0}))
        self.assertScenario("invalid_gap_policy", ph.build_tree(self.tmp / "c", base, weather_doc=ph.weather(ph.WEEK), gap_policy="shift"))
        self.assertScenario("weather_required", ph.build_tree(self.tmp / "d", base, weather_doc=None))
        self.assertScenario("weather_offset_mismatch", ph.build_tree(self.tmp / "e", base, weather_doc=ph.weather(ph.WEEK, offset="+01:00")))
        self.assertScenario("weather_starts_after_scenario", ph.build_tree(self.tmp / "f", base, weather_doc=ph.weather(ph.WEEK[3:])))
        self.assertScenario("event_limit", ph.build_tree(self.tmp / "g", base, weather_doc=ph.weather(ph.WEEK)), max_events=2)
        self.assertScenario("invalid_native_epoch", ph.build_tree(self.tmp / "h", base, weather_doc=ph.weather(ph.WEEK), native_epoch_s=-1.0))
        self.assertScenario("invalid_reference", ph.build_tree(self.tmp / "i", [("w", "../wind.json", ph.wind(), None)], weather_doc=None))
        no_weather_needed = [("w1", "events/wind.json", ph.wind(), None), ("q", "events/quiet.json", ph.interevent(86400, False, False, False), None)]
        scenario = load_scenario(ph.build_tree(self.tmp / "j", no_weather_needed, weather_doc=None))
        self.assertIsNone(scenario.weather)
        bad_digest = ph.build_tree(self.tmp / "k", base, weather_doc=ph.weather(ph.WEEK))
        doc = json.loads(bad_digest.read_text())
        doc["events"][0]["sha256"] = "zz" * 32
        bad_digest.write_text(json.dumps(doc))
        self.assertScenario("invalid_digest", bad_digest)

    def test_scheduled_start_static_rules(self) -> None:
        events = ph.standard_events()
        # minimum end before interevent3: 2 d + 1 h + 23 h + 30 min rainfall = 261000 s after 2027-12-30
        events[4] = (events[4][0], events[4][1], events[4][2], "2028-01-02T00:00:00+00:00")
        self.assertScenario("scheduled_start_before_minimum_end", ph.build_tree(self.tmp / "a", events, weather_doc=ph.weather(ph.WEEK)))
        events[4] = (events[4][0], events[4][1], events[4][2], "2028-01-02T00:30:00+00:00")  # exactly the minimum
        scenario = load_scenario(ph.build_tree(self.tmp / "b", events, weather_doc=ph.weather(ph.WEEK)))
        self.assertEqual(scenario.occurrences[4].scheduled_start_s, Fraction(261000))
        events[5] = (events[5][0], events[5][1], events[5][2], "2028-01-02T00:00:00+00:00")  # earlier than the previous one
        self.assertScenario("scheduled_start_before_minimum_end", ph.build_tree(self.tmp / "c", events, weather_doc=ph.weather(ph.WEEK)))
        events = ph.standard_events()
        events[0] = (events[0][0], events[0][1], events[0][2], "2027-12-29T00:00:00+00:00")
        self.assertScenario("before_calendar_start", ph.build_tree(self.tmp / "d", events, weather_doc=ph.weather(ph.WEEK)))
        events[0] = (events[0][0], events[0][1], events[0][2], "2027-12-30T00:00:00+01:00")
        self.assertScenario("offset_mismatch", ph.build_tree(self.tmp / "e", events, weather_doc=ph.weather(ph.WEEK)))


if __name__ == "__main__":
    unittest.main()
