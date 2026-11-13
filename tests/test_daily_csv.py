"""Daily CSV reader tests: strict parsing, identity, immutability, unsafe/quota refusals.

Expectations are written out independently of the implementation. Wind YAML needs
PyYAML (lazy, optional); those tests skip when it is absent.
"""
import dataclasses
import datetime
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from masy import daily_csv  # noqa: E402
from masy.daily_csv import DailyCsvError, DailyCsvLimits, load_daily_scenario, parse_weather_csv  # noqa: E402

EXAMPLES = ROOT / "examples" / "daily_scenario"
HAVE_YAML = importlib.util.find_spec("yaml") is not None
needs_yaml = unittest.skipUnless(HAVE_YAML, "PyYAML not installed")

WEATHER_HEAD = ("# format: masy.daily_weather.v1\n# pet_method: ecohyd_hargreaves\n"
                "# precipitation_threshold_mm: 5.0\n")
HEADER = "date,tmin_c,tmean_c,tmax_c,precipitation_mm,precipitation_intensity_mm_h\n"
ROW = "2030-06-01,18,25,32,0,0\n"
WIND = "wind_forcing:\n  blocks:\n    - shear_velocity_m_s: 0.45\n      direction_deg: 90.0\n      duration_s: 600.0\n"
WIND2 = "wind_forcing:\n  blocks:\n    - shear_velocity_m_s: 0.35\n      direction_deg: 270.0\n      duration_s: 900.0\n"


def weather_text(start="2030-06-01", days=7, head=WEATHER_HEAD):
    first = datetime.date.fromisoformat(start)
    rows = "".join(f"{first + datetime.timedelta(days=i)},18,25,32,0,0\n" for i in range(days))
    return head + HEADER + rows


def scenario_text(start="2030-06-01", end="2030-06-07", events=(), **over):
    keys = {"format": "masy.daily_scenario.v1", "scenario_id": "t1", "weather_file": "weather.csv",
            "start_date": start, "end_date": end, "evaporation_enabled": "true", "transpiration_enabled": "true",
            "growth_enabled": "true", "growth_interval_days": "7", "output_interval_days": "1",
            "checkpoint_interval_days": "7", "output_at_event_boundaries": "true"}
    keys.update(over)
    head = "".join(f"# {k}: {v}\n" for k, v in keys.items() if v is not None)
    return head + "date,event_type,event_file\n" + "".join(f"{d},wind,{f}\n" for d, f in events)


class Case(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, rel, text, base=None):
        path = (base or self.dir) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)
        return path

    def make(self, events=(("2030-06-02", "wind/a.yaml"),), base=None, weather=None, **over):
        base = base or self.dir
        self.write("weather.csv", weather if weather is not None else weather_text(), base)
        self.write("wind/a.yaml", WIND, base)
        self.write("wind/b.yaml", WIND2, base)
        return self.write("scenario.csv", scenario_text(events=events, **over), base)

    def code(self, fn, *args, **kwargs):
        with self.assertRaises(DailyCsvError) as ctx:
            fn(*args, **kwargs)
        return ctx.exception.code


class WeatherTests(Case):
    def test_shipped_hargreaves_example(self):
        table = parse_weather_csv((EXAMPLES / "weather.csv").read_bytes())
        self.assertEqual(table.pet_method, "ecohyd_hargreaves")
        self.assertEqual(table.precipitation_threshold_mm, 5.0)
        self.assertEqual(table.source, "Illustrative synthetic weather")
        self.assertEqual(len(table.days), 7)
        d4 = table.days[3]
        self.assertEqual((d4.date, d4.tmin_c, d4.tmean_c, d4.tmax_c), (datetime.date(2030, 6, 4), 19.0, 24.0, 29.0))
        self.assertEqual((d4.precipitation_mm, d4.precipitation_intensity_mm_h), (12.0, 24.0))
        self.assertIsNone(d4.pet_mm_day)
        self.assertEqual(table.days[5].wet_duration_s, 60.0)  # 1 mm at 60 mm/h
        self.assertIsNone(table.days[0].wet_duration_s)

    def test_shipped_explicit_example_same_dates(self):
        a = parse_weather_csv((EXAMPLES / "weather.csv").read_bytes())
        b = parse_weather_csv((EXAMPLES / "weather_explicit.csv").read_bytes())
        self.assertEqual(b.pet_method, "explicit")
        self.assertEqual([d.date for d in a.days], [d.date for d in b.days])
        self.assertEqual([d.precipitation_mm for d in a.days], [d.precipitation_mm for d in b.days])
        self.assertEqual(b.days[0].pet_mm_day, 5.1)
        self.assertIsNone(b.days[0].tmin_c)

    def test_binary64_preserved(self):
        text = weather_text(days=1).replace("2030-06-01,18,25,32,0,0", "2030-06-01,0,0.1,32,1e-3,3.5")
        day = parse_weather_csv(text.encode()).days[0]
        self.assertEqual(day.tmean_c.hex(), (0.1).hex())
        self.assertEqual(day.precipitation_mm, 0.001)

    def test_description_optional_and_crlf(self):
        text = WEATHER_HEAD + "# description: two words\n" + HEADER + ROW
        self.assertEqual(parse_weather_csv(text.encode()).description, "two words")
        self.assertEqual(len(parse_weather_csv(text.replace("\n", "\r\n").encode()).days), 1)

    def bad(self, text, code):
        data = text.encode() if isinstance(text, str) else text
        self.assertEqual(self.code(parse_weather_csv, data), code)

    def test_preamble_refusals(self):
        self.bad(WEATHER_HEAD.replace("# precipitation_threshold_mm: 5.0\n", "") + HEADER + ROW, "missing_metadata")
        self.bad(WEATHER_HEAD.replace("# pet_method: ecohyd_hargreaves\n", "") + HEADER + ROW, "missing_metadata")
        self.bad(WEATHER_HEAD.replace("# format: masy.daily_weather.v1\n", "") + HEADER + ROW, "missing_metadata")
        self.bad(WEATHER_HEAD + "# format: masy.daily_weather.v1\n" + HEADER + ROW, "duplicate_metadata")
        self.bad(WEATHER_HEAD + "# latitude: 30\n" + HEADER + ROW, "unknown_metadata")
        self.bad(WEATHER_HEAD.replace("ecohyd_hargreaves", "penman") + HEADER + ROW, "unsupported_pet_method")
        self.bad(WEATHER_HEAD.replace("weather.v1", "weather.v2") + HEADER + ROW, "invalid_format")
        self.bad(WEATHER_HEAD + "#nokey\n" + HEADER + ROW, "invalid_preamble")
        self.bad(WEATHER_HEAD.replace("5.0", "-1") + HEADER + ROW, "negative_value")
        self.bad(WEATHER_HEAD.replace("5.0", "nan") + HEADER + ROW, "invalid_number")

    def test_encoding_and_framing_refusals(self):
        self.bad(b"\xef\xbb\xbf" + (WEATHER_HEAD + HEADER + ROW).encode(), "invalid_encoding")
        self.bad(b"\xff\xfe", "invalid_encoding")
        self.bad(b"", "empty_document")
        self.bad(WEATHER_HEAD + HEADER + ROW + "\n", "malformed_row")  # trailing blank line
        self.bad(WEATHER_HEAD + HEADER + ROW.replace(",", "\t", 1), "invalid_control_character")
        self.bad(WEATHER_HEAD, "missing_table_header")
        self.bad(WEATHER_HEAD + HEADER, "weather_empty")

    def test_header_refusals(self):
        self.bad(WEATHER_HEAD + "date,tmean_c,tmin_c,tmax_c,precipitation_mm,precipitation_intensity_mm_h\n" + ROW, "invalid_header")
        self.bad(WEATHER_HEAD + HEADER.strip() + ",humidity\n" + ROW.strip() + ",1\n", "invalid_header")
        self.bad(WEATHER_HEAD + HEADER.replace("tmax_c", "tmean_c") + ROW, "invalid_header")
        explicit = WEATHER_HEAD.replace("ecohyd_hargreaves", "explicit")
        self.bad(explicit + HEADER + ROW, "invalid_header")  # hargreaves columns under explicit

    def test_row_refusals(self):
        self.bad(WEATHER_HEAD + HEADER + "2030-06-01,18,25,32,0\n", "malformed_row")
        self.bad(WEATHER_HEAD + HEADER + ROW.strip() + ",1\n", "malformed_row")
        self.bad(WEATHER_HEAD + HEADER + '2030-06-01,18,25,"32,0,0\n', "malformed_row")
        self.bad(WEATHER_HEAD + HEADER + "2030-06-01,18, 25,32,0,0\n", "malformed_row")
        for cell, code in (("nan", "invalid_number"), ("inf", "invalid_number"), ("1e999", "nonfinite_number"),
                           ("", "invalid_number"), ("0x10", "invalid_number"), ("1,5", "malformed_row"),
                           ("1e-999", "number_underflow"), ("-0", "negative_zero")):
            self.bad(WEATHER_HEAD + HEADER + f"2030-06-01,18,25,32,{cell},0\n", code)
        self.bad(WEATHER_HEAD + HEADER + "2030-06-01,18,25,32,-1,0\n", "negative_value")
        self.bad(WEATHER_HEAD + HEADER + "2030-06-01,18,25,32,0,-1\n", "negative_value")
        self.bad(WEATHER_HEAD + HEADER + "2030-06-01,18,25,32,1,0\n", "rain_without_intensity")
        self.bad(WEATHER_HEAD + HEADER + "2030-06-01,18,25,32,0,4\n", "intensity_without_rain")
        self.bad(WEATHER_HEAD + HEADER + "2030-06-01,18,25,32,1e300,1e-300\n", "rain_duration_not_representable")

    def test_explicit_pet_refusals(self):
        explicit = WEATHER_HEAD.replace("ecohyd_hargreaves", "explicit")
        head = "date,tmean_c,pet_mm_day,precipitation_mm,precipitation_intensity_mm_h\n"
        self.assertEqual(len(parse_weather_csv((explicit + head + "2030-06-01,25,0,0,0\n").encode()).days), 1)
        self.bad(explicit + head + "2030-06-01,25,-0.1,0,0\n", "negative_value")

    def test_temperature_order(self):
        self.bad(WEATHER_HEAD + HEADER + "2030-06-01,26,25,32,0,0\n", "temperature_order")
        self.bad(WEATHER_HEAD + HEADER + "2030-06-01,18,33,32,0,0\n", "temperature_order")
        ok = WEATHER_HEAD + HEADER + "2030-06-01,-5,-5,-5,0,0\n"
        self.assertEqual(parse_weather_csv(ok.encode()).days[0].tmin_c, -5.0)

    def test_date_refusals(self):
        for text, code in (("2030-6-1", "invalid_date"), ("2030/06/01", "invalid_date"), ("2100-02-29", "invalid_date"),
                           ("2030-13-01", "invalid_date"), ("20300601", "invalid_date"), ("２０３０-06-01", "invalid_date")):
            self.bad(WEATHER_HEAD + HEADER + f"{text},18,25,32,0,0\n", code)
        base = WEATHER_HEAD + HEADER
        self.bad(base + ROW + ROW, "weather_date_duplicate")
        self.bad(base + "2030-06-02,18,25,32,0,0\n2030-06-01,18,25,32,0,0\n", "weather_date_out_of_order")
        self.bad(base + ROW + "2030-06-03,18,25,32,0,0\n", "weather_date_missing")

    def test_leap_and_year_dates_accepted(self):
        table = parse_weather_csv(weather_text("2028-02-27", 4).encode())
        self.assertEqual([d.date.isoformat() for d in table.days],
                         ["2028-02-27", "2028-02-28", "2028-02-29", "2028-03-01"])
        table = parse_weather_csv(weather_text("2030-12-30", 4).encode())
        self.assertEqual(table.last_date, datetime.date(2031, 1, 2))

    def test_row_and_header_quotas(self):
        small = DailyCsvLimits(max_weather_rows=3)
        data = weather_text(days=4).encode()
        self.assertEqual(self.code(parse_weather_csv, data, small), "row_quota_exceeded")
        self.assertEqual(len(parse_weather_csv(weather_text(days=3).encode(), small).days), 3)
        self.assertEqual(self.code(parse_weather_csv, data, DailyCsvLimits(max_file_bytes=50)), "document_too_large")
        self.assertEqual(self.code(parse_weather_csv, data, DailyCsvLimits(max_header_lines=2)), "header_quota_exceeded")
        self.assertEqual(self.code(parse_weather_csv, data, DailyCsvLimits(max_header_chars=10)), "header_quota_exceeded")
        self.assertEqual(self.code(parse_weather_csv, data, DailyCsvLimits(max_cell_chars=5)), "malformed_row")
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(DailyCsvError):
                DailyCsvLimits(max_files=bad)


@needs_yaml
class ScenarioTests(Case):
    def test_shipped_example(self):
        s = load_daily_scenario(EXAMPLES / "scenario.csv")
        st = s.settings
        self.assertEqual((st.scenario_id, st.start_date.isoformat(), st.end_date.isoformat(), st.day_count),
                         ("example_week", "2030-06-01", "2030-06-07", 7))
        self.assertEqual((st.evaporation_enabled, st.transpiration_enabled, st.growth_enabled), (True, True, True))
        self.assertEqual((st.growth_interval_days, st.output_interval_days, st.checkpoint_interval_days), (7, 1, 7))
        self.assertTrue(st.output_at_event_boundaries)
        self.assertEqual([(o.date.isoformat(), o.reference) for o in s.occurrences],
                         [("2030-06-02", "wind/wind_a.yaml"), ("2030-06-04", "wind/wind_b.yaml"),
                          ("2030-06-04", "wind/wind_a.yaml"), ("2030-06-06", "wind/wind_a.yaml")])
        # Repeated filename: distinct occurrence identities, identical file bytes.
        ids = [o.occurrence_id for o in s.occurrences]
        self.assertEqual(len(set(ids)), 4)
        self.assertEqual(len({o.occurrence_sha256 for o in s.occurrences}), 4)
        a = [o for o in s.occurrences if o.reference.endswith("wind_a.yaml")]
        self.assertEqual(len({o.sha256 for o in a}), 1)
        self.assertEqual(len(s.wind_files), 2)
        blocks = s.occurrences[0].blocks
        self.assertEqual([(b.shear_velocity_m_s, b.direction_deg, b.duration_s) for b in blocks],
                         [(0.45, 90.0, 600.0), (0.6, 120.0, 300.0)])
        self.assertEqual(s.occurrences[0].declared_duration_s, 900.0)
        self.assertEqual(s.occurrences[1].blocks[0].direction_deg, 270.0)

    def test_header_only_event_table_is_valid(self):
        path = self.make(events=())
        s = load_daily_scenario(path)
        self.assertEqual(s.occurrences, ())

    def test_events_on_scenario_bounds_accepted(self):
        s = load_daily_scenario(self.make(events=(("2030-06-01", "wind/a.yaml"), ("2030-06-07", "wind/a.yaml"))))
        self.assertEqual(len(s.occurrences), 2)

    def test_scenario_refusals(self):
        cases = [
            (dict(growth_interval_days="8"), "growth_interval_out_of_range"),
            (dict(growth_interval_days="0"), "invalid_interval"),
            (dict(growth_interval_days="-1"), "invalid_interval"),
            (dict(growth_interval_days="1.0"), "invalid_interval"),
            (dict(output_interval_days="0"), "invalid_interval"),
            (dict(checkpoint_interval_days="x"), "invalid_interval"),
            (dict(growth_enabled="True"), "invalid_boolean"),
            (dict(output_at_event_boundaries="yes"), "invalid_boolean"),
            (dict(growth_enabled=None), "missing_metadata"),
            (dict(scenario_id="Bad Id"), "invalid_scenario_id"),
            (dict(format="masy.daily_scenario.v2"), "invalid_format"),
            (dict(start_date="2030-6-1"), "invalid_date"),
            (dict(end_date="2030-05-31"), "invalid_date_range"),
            (dict(end_date="2030-06-08"), "weather_coverage"),
            (dict(start_date="2030-05-31"), "weather_coverage"),
        ]
        for over, code in cases:
            with self.subTest(over=over):
                self.assertEqual(self.code(load_daily_scenario, self.make(events=(), **over)), code)

    def test_growth_range_applies_only_when_enabled_and_long_cadences_allowed(self):
        s = load_daily_scenario(self.make(events=(), growth_enabled="false", growth_interval_days="30",
                                          output_interval_days="900", checkpoint_interval_days="1000"))
        self.assertEqual(s.settings.growth_interval_days, 30)
        self.assertEqual(s.settings.output_interval_days, 900)
        s = load_daily_scenario(self.make(events=(), growth_interval_days="1"))
        self.assertEqual(s.settings.growth_interval_days, 1)

    def test_unknown_and_duplicate_scenario_keys(self):
        path = self.make(events=())
        text = path.read_text()
        path.write_text("# extra: 1\n" + text)
        self.assertEqual(self.code(load_daily_scenario, path), "unknown_metadata")
        path.write_text("# scenario_id: t2\n" + text)
        self.assertEqual(self.code(load_daily_scenario, path), "duplicate_metadata")

    def test_event_row_refusals(self):
        self.assertEqual(self.code(load_daily_scenario, self.make(events=(("2030-06-03", "wind/a.yaml"), ("2030-06-02", "wind/a.yaml")))),
                         "event_dates_out_of_order")
        self.assertEqual(self.code(load_daily_scenario, self.make(events=(("2030-06-08", "wind/a.yaml"),))),
                         "event_date_out_of_range")
        self.assertEqual(self.code(load_daily_scenario, self.make(events=(("2030-05-31", "wind/a.yaml"),))),
                         "event_date_out_of_range")
        path = self.make(events=())
        preamble = path.read_text().split("date,event_type")[0]
        for text, code in (("date,event_type,event_file\n2030-06-02,rain,wind/a.yaml\n", "unsupported_event_type"),
                           ("date,event_file,event_type\n", "invalid_header"),
                           ("date,event_type,event_file,x\n", "invalid_header"),
                           ("date,event_type,event_file\n2030-06-02,wind\n", "malformed_row"),
                           ("date,event_type,event_file\n2030-06-02,wind,wind/a.json\n", "unsupported_wind_suffix"),
                           ("date,event_type,event_file\n2030-06-02,wind,wind/missing.yaml\n", "document_missing"),
                           ("", "missing_table_header")):
            body = preamble + text
            path.write_text(body)
            self.assertEqual(self.code(load_daily_scenario, path), code, text)

    def test_unsafe_references(self):
        for ref in ("../outside.yaml", "/etc/passwd.yaml", "wind/../wind/a.yaml", "./wind/a.yaml", "wind//a.yaml",
                    "wind/a.yaml/"):
            with self.subTest(ref=ref):
                self.assertIn(self.code(load_daily_scenario, self.make(events=(("2030-06-02", ref),))),
                              {"invalid_reference", "unsupported_wind_suffix"})
        self.assertEqual(self.code(load_daily_scenario, self.make(events=(), weather_file="../weather.csv")), "invalid_reference")

    def test_symlinks_refused(self):
        path = self.make(events=(("2030-06-02", "wind/a.yaml"),))
        outside = self.write("outside_dir/a.yaml", WIND, self.dir.parent / (self.dir.name + "_out"))
        self.addCleanup(shutil.rmtree, outside.parent.parent, True)
        try:
            os.symlink(outside.parent, self.dir / "linkdir")
            os.symlink(outside, self.dir / "wind" / "link.yaml")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        text = path.read_text()
        for ref, codes in (("linkdir/a.yaml", {"reference_escapes_base"}), ("wind/link.yaml", {"reference_escapes_base", "reference_not_canonical", "document_symlink"})):
            path.write_text(text.replace("wind/a.yaml", ref))
            self.assertIn(self.code(load_daily_scenario, path), codes)
        # A symlinked directory that stays inside the root is still refused (not canonical).
        os.symlink(self.dir / "wind", self.dir / "inner")
        path.write_text(text.replace("wind/a.yaml", "inner/a.yaml"))
        self.assertEqual(self.code(load_daily_scenario, path), "reference_not_canonical")
        # Symlinked scenario file and symlinked weather file.
        os.symlink(path, self.dir / "s2.csv")
        self.assertEqual(self.code(load_daily_scenario, self.dir / "s2.csv"), "document_symlink")
        (self.dir / "weather.csv").rename(self.dir / "w_real.csv")
        os.symlink(self.dir / "w_real.csv", self.dir / "weather.csv")
        path.write_text(text)
        self.assertIn(self.code(load_daily_scenario, path), {"reference_not_canonical", "document_symlink"})

    def test_wind_content_refusals(self):
        bad = {
            "unknown top key": WIND + "extra: 1\n",
            "no blocks": "wind_forcing:\n  blocks: []\n",
            "extra block key": WIND + "      extra: 1\n",
            "zero duration": WIND.replace("600.0", "0.0"),
            "negative shear": WIND.replace("0.45", "-0.45"),
            "direction 360": WIND.replace("90.0", "360.0"),
            "string number": WIND.replace("600.0", '"600.0"'),
            "bool number": WIND.replace("600.0", "true"),
            "nan": WIND.replace("600.0", ".nan"),
            "not a mapping": "- a\n- b\n",
            "duplicate key": WIND.replace("direction_deg: 90.0\n", "direction_deg: 90.0\n      direction_deg: 91.0\n"),
        }
        for name, text in bad.items():
            with self.subTest(name):
                path = self.make(events=(("2030-06-02", "wind/a.yaml"),))
                self.write("wind/a.yaml", text)
                with self.assertRaises(DailyCsvError):
                    load_daily_scenario(path)
        path = self.make(events=(("2030-06-02", "wind/a.yaml"),))
        self.write("wind/a.yaml", WIND.replace("600.0", "0.0"))
        self.assertEqual(self.code(load_daily_scenario, path), "invalid_wind_record")

    def test_file_and_aggregate_quotas(self):
        path = self.make(events=(("2030-06-02", "wind/a.yaml"), ("2030-06-03", "wind/b.yaml")))
        load_daily_scenario(path, DailyCsvLimits(max_files=4))
        self.assertEqual(self.code(load_daily_scenario, path, DailyCsvLimits(max_files=3)), "file_quota_exceeded")
        self.assertEqual(self.code(load_daily_scenario, path, DailyCsvLimits(max_file_bytes=100)), "document_too_large")
        total = sum(p.stat().st_size for p in (path, self.dir / "weather.csv", self.dir / "wind/a.yaml", self.dir / "wind/b.yaml"))
        self.assertEqual(self.code(load_daily_scenario, path, DailyCsvLimits(max_total_bytes=total - 1)), "aggregate_quota_exceeded")
        load_daily_scenario(path, DailyCsvLimits(max_total_bytes=total))
        # Repeated references count once.
        rep = self.make(events=tuple(("2030-06-02", "wind/a.yaml") for _ in range(10)))
        load_daily_scenario(rep, DailyCsvLimits(max_files=3))
        dry = self.make(events=())
        load_daily_scenario(dry, DailyCsvLimits(max_files=2))
        self.assertEqual(self.code(load_daily_scenario, dry, DailyCsvLimits(max_files=1)), "file_quota_exceeded")

    def test_total_wind_duration_must_fit_native_float(self):
        enormous = WIND.replace("600.0", "1.0e+308")
        enormous += enormous.split("  blocks:\n", 1)[1]
        self.assertEqual(self.code(daily_csv.parse_wind_yaml, enormous.encode()), "invalid_wind_duration")

    def test_scenario_row_quota(self):
        path = self.make(events=tuple(("2030-06-02", "wind/a.yaml") for _ in range(3)))
        self.assertEqual(self.code(load_daily_scenario, path, DailyCsvLimits(max_scenario_rows=2)), "row_quota_exceeded")
        self.assertEqual(len(load_daily_scenario(path, DailyCsvLimits(max_scenario_rows=3)).occurrences), 3)

    def test_concurrent_modification_refused(self):
        path = self.make(events=(("2030-06-02", "wind/a.yaml"), ("2030-06-03", "wind/a.yaml")))
        wind = self.dir / "wind" / "a.yaml"
        real = daily_csv.read_bounded
        reads = []

        def racing(target, limit):
            result = real(target, limit)
            if Path(target) == Path(os.path.realpath(wind)):
                reads.append(1)
                if len(reads) == 1:
                    wind.write_text(WIND2)
            return result

        with mock.patch.object(daily_csv, "read_bounded", racing):
            self.assertEqual(self.code(load_daily_scenario, path), "document_changed")
        self.assertEqual(len(reads), 2)  # repeated reference is read once, then re-checked once

    def test_nonexistent_scenario(self):
        self.assertEqual(self.code(load_daily_scenario, self.dir / "none.csv"), "document_missing")


@needs_yaml
class IdentityTests(Case):
    def ident(self, path):
        return load_daily_scenario(path).identity_sha256

    def test_identity_stable_across_directory_copies_and_cwd(self):
        path = self.make(events=(("2030-06-02", "wind/a.yaml"), ("2030-06-02", "wind/b.yaml")))
        other = Path(tempfile.mkdtemp(prefix="elsewhere_"))
        self.addCleanup(shutil.rmtree, other, True)
        copy = other / "deep" / "copy"
        shutil.copytree(self.dir, copy)
        self.assertEqual(self.ident(path), self.ident(copy / "scenario.csv"))
        old = os.getcwd()
        self.addCleanup(os.chdir, old)
        os.chdir(other)
        self.assertEqual(self.ident(path), self.ident(copy / "scenario.csv"))
        record = json.dumps(load_daily_scenario(path).identity_record())
        self.assertNotIn(str(self.dir), record)
        self.assertNotIn(str(other), record)

    def test_identity_changes(self):
        events = (("2030-06-02", "wind/a.yaml"), ("2030-06-02", "wind/b.yaml"))
        base = self.ident(self.make(events=events))
        # threshold
        self.write("weather.csv", weather_text().replace("5.0", "5.5"))
        self.assertNotEqual(self.ident(self.dir / "scenario.csv"), base)
        # weather bytes (source text) and weather value
        self.write("weather.csv", weather_text().replace("# pet_method", "# source: other\n# pet_method"))
        self.assertNotEqual(self.ident(self.dir / "scenario.csv"), base)
        self.write("weather.csv", weather_text().replace("2030-06-03,18,25,32,0,0", "2030-06-03,18,25,33,0,0"))
        self.assertNotEqual(self.ident(self.dir / "scenario.csv"), base)
        self.write("weather.csv", weather_text())
        self.assertEqual(self.ident(self.dir / "scenario.csv"), base)
        # wind bytes
        self.write("wind/a.yaml", WIND.replace("600.0", "601.0"))
        self.assertNotEqual(self.ident(self.dir / "scenario.csv"), base)
        self.write("wind/a.yaml", WIND)
        # same-date row order
        swapped = self.make(events=tuple(reversed(events)))
        self.assertNotEqual(self.ident(swapped), base)
        # controls
        for over in (dict(output_interval_days="2"), dict(growth_interval_days="3"), dict(growth_enabled="false"),
                     dict(evaporation_enabled="false"), dict(output_at_event_boundaries="false"),
                     dict(checkpoint_interval_days="5"), dict(transpiration_enabled="false"),
                     dict(end_date="2030-06-06"), dict(scenario_id="t2")):
            with self.subTest(over=over):
                self.assertNotEqual(self.ident(self.make(events=events, **over)), base)

    def test_identity_unchanged_by_unrelated_sibling_files(self):
        path = self.make()
        base = self.ident(path)
        self.write("notes.txt", "x")
        self.write("wind/unused.yaml", WIND2)
        self.assertEqual(self.ident(path), base)

    def test_explicit_and_hargreaves_differ(self):
        base = self.ident(self.make())
        explicit = ("# format: masy.daily_weather.v1\n# pet_method: explicit\n# precipitation_threshold_mm: 5.0\n"
                    "date,tmean_c,pet_mm_day,precipitation_mm,precipitation_intensity_mm_h\n"
                    + "".join(f"2030-06-0{i},25,4,0,0\n" for i in range(1, 8)))
        self.assertNotEqual(self.ident(self.make(weather=explicit)), base)

    def test_immutability_and_independent_records(self):
        s = load_daily_scenario(self.make(events=(("2030-06-02", "wind/a.yaml"),)))
        for target, name in ((s, "weather_reference"), (s.settings, "growth_enabled"), (s.weather, "pet_method"),
                             (s.weather.days[0], "tmean_c"), (s.occurrences[0], "reference"),
                             (s.occurrences[0].blocks[0], "duration_s")):
            with self.assertRaises(dataclasses.FrozenInstanceError):
                setattr(target, name, 1)
        self.assertIsInstance(s.occurrences, tuple)
        self.assertIsInstance(s.weather.days, tuple)
        record = s.identity_record()
        record["settings"]["growth_enabled"] = False
        record["events"].clear()
        again = s.identity_record()
        self.assertTrue(again["settings"]["growth_enabled"])
        self.assertEqual(len(again["events"]), 1)
        self.assertEqual(s.identity_sha256, load_daily_scenario(self.dir / "scenario.csv").identity_sha256)
        occ = s.occurrences[0].record()
        occ["blocks"].append("junk")
        self.assertEqual(len(s.occurrences[0].record()["blocks"]), 1)
        day = s.weather.days[0].record()
        day["tmean_c"] = -99.0
        self.assertEqual(s.weather.days[0].tmean_c, 25.0)
        s.summary()["settings"]["scenario_id"] = "x"
        self.assertEqual(s.settings.scenario_id, "t1")

    def test_identity_json_is_canonical_and_lossless(self):
        s = load_daily_scenario(self.make(events=(("2030-06-02", "wind/a.yaml"),)))
        record = s.identity_record()
        self.assertEqual(record["schema"], "masy.daily_source_identity.v1")
        self.assertEqual(record["weather"]["precipitation_threshold_mm"]["hex"], (5.0).hex())
        self.assertEqual(record["wind_files"]["wind/a.yaml"]["blocks"][0]["duration_s"]["fraction"], "600/1")
        self.assertEqual(record["events"][0]["reference"], "wind/a.yaml")
        self.assertEqual(record["weather"]["reference"], "weather.csv")

    def test_reusable_wind_identity_is_bounded_by_sources_and_occurrences(self):
        path = self.make(events=tuple(("2030-06-02", "wind/a.yaml") for _ in range(200)))
        block = WIND.split("  blocks:\n", 1)[1]
        self.write("wind/a.yaml", "wind_forcing:\n  blocks:\n" + block * 200)
        result = load_daily_scenario(path)
        self.assertEqual(len(result.occurrences), 200)
        self.assertLess(len(result.identity_json), 200000)


class ImportTests(unittest.TestCase):
    def test_new_modules_do_not_import_heavy_packages(self):
        code = ("import sys; sys.path.insert(0, %r); import masy.daily_csv, masy.daily_schedule, masy.daily_cli\n"
                "bad = [m for m in sys.modules if m.partition('.')[0] in "
                "('numpy','yaml','maple','maple_syrup','syrup','leaf','numba','scipy','jax','cupy')]\n"
                "assert not bad, bad\n" % str(ROOT / "src"))
        proc = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
