"""Ordered sequential Dense schedule: schema, lossless records, resolution evidence (stdlib only)."""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
import unittest
from fractions import Fraction

import _support  # noqa: F401 - puts MASY src on sys.path

from masy.clock import ClockError, decode_time
from masy.dense_schedule import (
    SCHEDULE_SCHEMA,
    DenseSchedule,
    DenseScheduleError,
    RainEvent,
    RainInterval,
    WindEvent,
    case_schedule_digest,
    check_resolved_event,
    resolve_event,
)

T0 = 1234.5678  # nonzero native epoch


def wind(duration=600.1, shear=0.45, direction=270.0) -> dict:
    return {"kind": "wind", "duration_s": duration, "shear_velocity_m_s": shear,
            "direction_deg_cw_from_north": direction}


def rain(intervals=((0.1, 30.0), (0.2, 0.0), (900, 12.5)), guard=7200.0) -> dict:
    return {"kind": "rain", "recession_guard_s": guard,
            "intervals": [{"duration_s": d, "intensity_mm_h": i} for d, i in intervals]}


def raw(*events) -> dict:
    return {"schema": SCHEDULE_SCHEMA, "events": list(events)}


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


class ScheduleSchemaTests(unittest.TestCase):
    def assertCode(self, code: str, call, *args, **kwargs) -> None:
        with self.assertRaises(DenseScheduleError) as ctx:
            call(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)

    def test_sequential_list_need_not_alternate(self) -> None:
        schedule = DenseSchedule.from_input(raw(wind(), wind(30, 0.0, 0.0), rain(), rain(), wind()))
        self.assertEqual([e.kind for e in schedule.events], ["wind", "wind", "rain", "rain", "wind"])
        self.assertEqual(schedule.events[1].shear_velocity_m_s, 0.0)
        self.assertIs(type(schedule.events[0].duration_s), float)

    def test_input_and_record_round_trips_are_lossless(self) -> None:
        schedule = DenseSchedule.from_input(raw(wind(5e-324), rain(), wind(2 ** 52 + 1)))
        self.assertEqual(DenseSchedule.from_input(json.loads(json.dumps(schedule.as_input()))), schedule)
        record = json.loads(json.dumps(schedule.as_record()))
        rebuilt = DenseSchedule.from_record(record)
        self.assertEqual(rebuilt, schedule)
        self.assertEqual(rebuilt.events[0].duration_s.hex(), (5e-324).hex())
        self.assertEqual(record["events"][1]["rainfall_duration_s"],
                         fraction_text(Fraction(0.1) + Fraction(0.2) + Fraction(900)))

    def test_digest_is_stable_and_sensitive_to_one_ulp(self) -> None:
        a = DenseSchedule.from_input(raw(wind(), rain()))
        b = DenseSchedule.from_input(json.loads(json.dumps(raw(wind(), rain()))))
        self.assertEqual(a.digest, b.digest)
        self.assertRegex(a.digest, r"^[0-9a-f]{64}$")
        c = DenseSchedule.from_input(raw(wind(math.nextafter(600.1, 1e9)), rain()))
        self.assertNotEqual(a.digest, c.digest)
        d = DenseSchedule.from_input(raw(rain(), wind()))
        self.assertNotEqual(a.digest, d.digest)  # order is part of the identity

    def test_records_are_immutable(self) -> None:
        schedule = DenseSchedule.from_input(raw(wind(), rain()))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            schedule.events[0].duration_s = 1.0  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            schedule.events = ()  # type: ignore[misc]
        self.assertIsInstance(schedule.events, tuple)
        self.assertIsInstance(schedule.events[1].intervals, tuple)
        record = schedule.as_record()
        record["events"][0]["duration_s"]["hex"] = "0x1.0p+0"
        self.assertEqual(schedule.as_record()["events"][0]["duration_s"]["hex"], (600.1).hex())

    def test_direct_construction_is_validated(self) -> None:
        self.assertCode("nonpositive_duration", WindEvent, 0.0, 1.0, 0.0)
        self.assertCode("invalid_rain_intervals", RainEvent, [RainInterval(1.0, 1.0)], 10.0)
        self.assertCode("empty_schedule", DenseSchedule, ())
        self.assertCode("invalid_event", DenseSchedule, ({"kind": "wind"},))

    def test_absolute_and_calendar_fields_refuse(self) -> None:
        for event in ({**wind(), "native_start_s": T0}, {**wind(), "native_end_s": T0 + 1},
                      {**wind(), "start": "2026-10-18T00:00:00Z"}, {**rain(), "calendar_start": "2026-10-18"},
                      {**rain(), "end_time_s": 5.0}, {**wind(), "timestamp": 1}):
            with self.subTest(keys=sorted(event)):
                self.assertCode("absolute_time_refused", DenseSchedule.from_input, raw(event))
        self.assertCode("absolute_time_refused", DenseSchedule.from_input, {**raw(wind()), "native_epoch_s": T0})
        interval = rain()
        interval["intervals"][0]["start_s"] = 0.0
        self.assertCode("absolute_time_refused", DenseSchedule.from_input, raw(interval))

    def test_unknown_missing_and_kind_errors(self) -> None:
        self.assertCode("unknown_key", DenseSchedule.from_input, raw({**wind(), "gust_factor": 1.2}))
        self.assertCode("unknown_key", DenseSchedule.from_input, {**raw(wind()), "note": "x"})
        missing = wind()
        missing.pop("direction_deg_cw_from_north")
        self.assertCode("missing_key", DenseSchedule.from_input, raw(missing))
        self.assertCode("invalid_event_kind", DenseSchedule.from_input, raw({**wind(), "kind": "calm"}))
        self.assertCode("invalid_event_kind", DenseSchedule.from_input, raw({**wind(), "kind": "idle"}))
        self.assertCode("empty_schedule", DenseSchedule.from_input, raw())
        self.assertCode("unsupported_schema", DenseSchedule.from_input, {"schema": "masy.dense_schedule.v2",
                                                                         "events": [wind()]})
        self.assertCode("malformed_schedule", DenseSchedule.from_input, [wind()])

    def test_invalid_durations_refuse(self) -> None:
        cases = [
            (True, "invalid_number"), (False, "invalid_number"), ("600", "invalid_number"), (None, "invalid_number"),
            (2 ** 60 + 1, "invalid_number"), (float("nan"), "nonfinite_value"), (float("inf"), "nonfinite_value"),
            (float("-inf"), "nonfinite_value"), (-0.0, "negative_zero"), (0.0, "nonpositive_duration"),
            (0, "nonpositive_duration"), (-1.0, "nonpositive_duration"),
        ]
        for value, code in cases:
            with self.subTest(value=value):
                self.assertCode(code, DenseSchedule.from_input, raw(wind(duration=value)))
                self.assertCode(code, DenseSchedule.from_input, raw(rain(intervals=((value, 1.0),))))
                self.assertCode(code, DenseSchedule.from_input, raw(rain(guard=value)))

    def test_invalid_forcing_refuses(self) -> None:
        self.assertCode("negative_zero", DenseSchedule.from_input, raw(wind(shear=-0.0)))
        self.assertCode("negative_value", DenseSchedule.from_input, raw(wind(shear=-0.1)))
        self.assertCode("invalid_number", DenseSchedule.from_input, raw(wind(shear=True)))
        self.assertCode("invalid_direction", DenseSchedule.from_input, raw(wind(direction=360.0)))
        self.assertCode("invalid_direction", DenseSchedule.from_input, raw(wind(direction=-1.0)))
        self.assertCode("negative_zero", DenseSchedule.from_input, raw(wind(direction=-0.0)))
        self.assertCode("negative_value", DenseSchedule.from_input, raw(rain(intervals=((10.0, -1.0),))))
        self.assertCode("dry_rain_event", DenseSchedule.from_input, raw(rain(intervals=((10.0, 0.0),))))
        self.assertCode("invalid_rain_intervals", DenseSchedule.from_input, raw(rain(intervals=())))

    def test_tampered_records_refuse(self) -> None:
        good = DenseSchedule.from_input(raw(wind(), rain())).as_record()

        def mutated(change):
            record = json.loads(json.dumps(good))
            change(record)
            return record

        cases = [
            ("inconsistent_schedule_record", mutated(lambda r: r["events"][1].__setitem__("rainfall_duration_s", "1/1"))),
            ("inconsistent_schedule_record", mutated(lambda r: r["events"][0].__setitem__("index", True))),
            ("inconsistent_schedule_record", mutated(lambda r: r["events"][1].__setitem__("index", 0))),
            ("unknown_key", mutated(lambda r: r["events"][0].__setitem__("extra", 1))),
            ("absolute_time_refused", mutated(lambda r: r["events"][0].__setitem__("native_start", r["events"][0]
                                                                                    ["duration_s"]))),
            ("unsupported_schema", mutated(lambda r: r.__setitem__("schema", "masy.dense_schedule.v0"))),
            ("malformed_schedule", mutated(lambda r: r["events"][0]["duration_s"].__setitem__("hex", "0x1.0p+0"))),
            ("negative_zero", mutated(lambda r: r["events"][0].__setitem__(
                "shear_velocity_m_s", {"hex": "-0x0.0p+0", "fraction": "0/1"}))),
        ]
        for code, record in cases:
            with self.subTest(code=code):
                self.assertCode(code, DenseSchedule.from_record, record)


class ResolutionTests(unittest.TestCase):
    def test_wind_edges_keep_intended_and_native_separately(self) -> None:
        schedule = DenseSchedule.from_input(raw(wind(), rain()))
        record = resolve_event(schedule, 0, T0)
        self.assertEqual(record["intended"], schedule.as_record()["events"][0])
        self.assertEqual(record["edge_roles"], ["event_start", "wind_end"])
        edges = record["resolved_edges"]["edges"]
        self.assertEqual(decode_time(record["resolved_edges"]["native_start"]).hex(), T0.hex())
        self.assertEqual(decode_time(edges[1]["native"]).hex(), (T0 + 600.1).hex())
        self.assertEqual(edges[1]["intended_fraction"], fraction_text(Fraction(600.1)))
        offset = Fraction(T0 + 600.1) - (Fraction(T0) + Fraction(600.1))
        self.assertEqual(edges[1]["native_minus_intended"], fraction_text(offset))
        self.assertEqual(record["schedule_sha256"], schedule.digest)

    def test_rain_edges_are_exact_cumulative_offsets(self) -> None:
        schedule = DenseSchedule.from_input(raw(rain()))
        record = resolve_event(schedule, 0, T0)
        self.assertEqual(record["edge_roles"], ["event_start", "rain_interval_0_end", "rain_interval_1_end",
                                                "rainfall_end", "recession_guard_end"])
        edges = record["resolved_edges"]["edges"]
        exact = Fraction(0.1) + Fraction(0.2)  # not the float sum 0.30000000000000004
        self.assertEqual(edges[2]["intended_fraction"], fraction_text(exact))
        self.assertEqual(decode_time(edges[2]["native"]).hex(), (T0 + float(exact)).hex())
        rainfall = exact + 900
        self.assertEqual(edges[4]["intended_fraction"], fraction_text(rainfall + Fraction(7200)))

    def test_later_event_resolves_from_its_realized_native_start(self) -> None:
        schedule = DenseSchedule.from_input(raw(wind(), rain(), wind(10.0)))
        realized_rain_end = T0 + 600.1 + 4321.75  # e.g. returned by a native water completion
        record = resolve_event(schedule, 2, realized_rain_end)
        self.assertEqual(decode_time(record["resolved_edges"]["native_start"]).hex(), realized_rain_end.hex())
        self.assertEqual(decode_time(record["resolved_edges"]["edges"][1]["native"]).hex(),
                         (realized_rain_end + 10.0).hex())

    def test_natively_collapsed_duration_refuses(self) -> None:
        schedule = DenseSchedule.from_input(raw(wind(0.5)))
        with self.assertRaises(ClockError) as ctx:
            resolve_event(schedule, 0, float(2 ** 53))
        self.assertEqual(ctx.exception.code, "collapsed_forcing_edges")
        lossy = DenseSchedule.from_input(raw(rain(intervals=((1.0, 5.0), (1.0, 5.0)))))
        with self.assertRaises(ClockError):
            resolve_event(lossy, 0, 1e17)

    def test_invalid_native_start_and_index_refuse(self) -> None:
        schedule = DenseSchedule.from_input(raw(wind()))
        for start, code in ((-0.0, "negative_zero_time"), (-1.0, "negative_time_value"),
                            (float("nan"), "invalid_time_value"), (True, "invalid_time_value")):
            with self.subTest(start=start):
                with self.assertRaises(ClockError) as ctx:
                    resolve_event(schedule, 0, start)
                self.assertEqual(ctx.exception.code, code)
        with self.assertRaises(DenseScheduleError):
            resolve_event(schedule, 1, T0)
        with self.assertRaises(DenseScheduleError):
            resolve_event(schedule, True, T0)

    def test_resolved_record_round_trip_and_tamper(self) -> None:
        schedule = DenseSchedule.from_input(raw(wind(), rain()))
        record = json.loads(json.dumps(resolve_event(schedule, 1, T0 + 600.1)))
        self.assertEqual(check_resolved_event(record, schedule), record)
        tampered = json.loads(json.dumps(record))
        tampered["resolved_edges"]["edges"][1]["native"]["hex"] = (T0 + 600.1).hex()
        with self.assertRaises((DenseScheduleError, ClockError)):
            check_resolved_event(tampered, schedule)
        other = DenseSchedule.from_input(raw(wind(), rain(guard=60.0)))
        with self.assertRaises(DenseScheduleError) as ctx:
            check_resolved_event(record, other)
        self.assertEqual(ctx.exception.code, "foreign_resolved_event")

    def test_case_schedule_digest(self) -> None:
        schedule = DenseSchedule.from_input(raw(wind(), rain()))
        case = "a" * 64
        self.assertEqual(case_schedule_digest(case, schedule), case_schedule_digest(case, schedule))
        self.assertNotEqual(case_schedule_digest(case, schedule), case_schedule_digest("b" * 64, schedule))
        for bad in ("A" * 64, "a" * 63, 1):
            with self.subTest(bad=bad):
                with self.assertRaises(DenseScheduleError):
                    case_schedule_digest(bad, schedule)  # type: ignore[arg-type]


class ImportIsolationTests(unittest.TestCase):
    def test_modules_import_without_numpy_or_native_models(self) -> None:
        code = (
            "import sys; sys.path.insert(0, %r); import masy.dense_schedule, masy.dense_bed; "
            "bad = sorted(m for m in ('numpy', 'maple', 'maple_syrup') if m in sys.modules); "
            "print(','.join(bad))" % str(_support.SRC)
        )
        completed = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
