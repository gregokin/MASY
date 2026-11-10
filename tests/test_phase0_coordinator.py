"""Phase 0: mechanical coordinator protocol, dispatch integrity, process gate, atomic advance and cursor restart.

No native model code runs; adapters are mechanical fakes. Tests whose scenario carries weather need the owning
PET library (`maple_syrup.weather_pet`) for row validation and are skipped explicitly without it; the
`LibraryFreeTests` class covers the protocol with weather-free scenarios.
"""

from __future__ import annotations

import json
import unittest
from fractions import Fraction

import _phase0_support as ph
from _support import TempDirCase

from masy.clock import decode_time, encode_time
from masy.coordinator import AdapterCapabilities, AdapterOutcome, Coordinator, CoordinatorError
from masy.scenario import load_scenario

DAY = 86400.0


class CoordinatorCase(TempDirCase):
    def scenario(self, events=None, *, weather=True, name="s", **kwargs):
        return load_scenario(ph.build_tree(self.tmp / name, events or ph.standard_events(),
                                           weather_doc=ph.weather(ph.WEEK) if weather else None, **kwargs))

    def assertRefusal(self, code: str, call, *args, **kwargs) -> CoordinatorError:
        with self.assertRaises(CoordinatorError) as ctx:
            call(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)
        return ctx.exception

    def snapshot(self, coordinator: Coordinator) -> str:
        return json.dumps(coordinator.as_record(), sort_keys=True)


class LibraryFreeTests(CoordinatorCase):
    """Protocol checks on weather-free scenarios (no PET library needed)."""

    def test_capability_preflight_runs_regardless_of_process_gates(self) -> None:
        # a 1 s interevent with every process disabled still may not reach a whole-day-only adapter
        scenario = self.scenario(ph.quiet_events(1.0), weather=False, name="q1")
        daily = {"wind": ph.FakeAdapter("wind"),
                 "interevent": ph.FakeAdapter("interevent", duration_policy="whole_days_86400_only")}
        coordinator = Coordinator(scenario, daily)
        coordinator.step()
        before = self.snapshot(coordinator)
        error = self.assertRefusal("adapter_duration_policy_violation", coordinator.step)
        self.assertIn("not truncated", error.detail)
        self.assertEqual(self.snapshot(coordinator), before)
        self.assertEqual(daily["interevent"].dispatches, [])  # never called
        # the same policy also applies to wind and rain adapters
        windy = {"wind": ph.FakeAdapter("wind", duration_policy="whole_days_86400_only"), "interevent": ph.FakeAdapter("interevent")}
        self.assertRefusal("adapter_duration_policy_violation", Coordinator(scenario, windy).step)
        # a no-leap adapter cannot receive 29 February even with all gates false
        calendar = {"start": "2028-02-28T00:00:00+00:00", "convention": "proleptic_gregorian", "leap_seconds": "none"}
        leap = load_scenario(ph.build_tree(self.tmp / "leap", [("q", "events/q.json", ph.interevent(2 * DAY, False, False, False), None)],
                                           weather_doc=None, calendar=calendar))
        noleap = {"interevent": ph.FakeAdapter("interevent", calendar_support="noleap_365_day_index")}
        self.assertRefusal("noleap_adapter_leap_day", Coordinator(leap, noleap).step)
        self.assertEqual(noleap["interevent"].dispatches, [])
        gregorian = {"interevent": ph.FakeAdapter("interevent")}
        realized = Coordinator(leap, gregorian).run()
        self.assertEqual(realized[0]["calendar_end"]["date"], "2028-03-01")
        self.assertEqual(gregorian["interevent"].dispatches[0]["calendar"]["dates_touched"], ["2028-02-28", "2028-02-29"])
        self.assertIsNone(gregorian["interevent"].dispatches[0]["weather"])

    def test_adapter_must_declare_native_staging(self) -> None:
        scenario = self.scenario(ph.quiet_events(), weather=False, name="stage")
        unstaged = {"wind": ph.FakeAdapter("wind", staging="in_place_unstaged"), "interevent": ph.FakeAdapter("interevent")}
        error = self.assertRefusal("adapter_staging_required", Coordinator, scenario, unstaged)
        self.assertIn("cannot roll back", error.detail)
        with self.assertRaises(TypeError):  # no default: nothing is implied by omission
            AdapterCapabilities(kind="rain", adapter_id="dry_reset", source_identity="y", continuous_soil_moisture=False,
                                duration_policy="any_positive_duration", calendar_support="proleptic_gregorian_dates",
                                native_state_checkpoint="not_provided", notes="")
        with self.assertRaises(CoordinatorError):
            AdapterCapabilities("wind", "x", "y", False, "any_positive_duration", "proleptic_gregorian_dates", "not_provided",
                                "eventually", "")

    def test_mutated_dispatch_is_refused_and_never_validated(self) -> None:
        scenario = self.scenario(ph.quiet_events(1.0), weather=False, name="mut")

        def stretch(dispatch, end):  # the review reproducer: declare 1 s, claim 2 s through the adapter's copy
            dispatch["planned_edges"]["edges"][-1]["native"] = encode_time(2.0 + DAY * 0 + decode_time(dispatch["native_start"]))
            return decode_time(dispatch["planned_edges"]["edges"][-1]["native"])

        def gate(dispatch, end):
            dispatch["processes"]["growth"] = True
            return end

        def identity(dispatch, end):
            dispatch["occurrence_id"] = "other"
            return end

        def weather(dispatch, end):
            dispatch["weather"] = {"dates": ["2027-12-30"]}
            return end

        def extra_key(dispatch, end):
            dispatch["note"] = "x"
            return end

        for name, mutate in (("edges", stretch), ("gate", gate), ("identity", identity), ("weather", weather), ("key", extra_key)):
            with self.subTest(mutation=name):
                fakes = {"wind": ph.FakeAdapter("wind"), "interevent": ph.FakeAdapter("interevent", mutate=mutate)}
                coordinator = Coordinator(scenario, fakes)
                coordinator.step()
                before = self.snapshot(coordinator)
                ledger_before = coordinator.ledger.as_record()
                self.assertRefusal("dispatch_mutated", coordinator.step)
                self.assertEqual(self.snapshot(coordinator), before)
                self.assertEqual(coordinator.ledger.as_record(), ledger_before)
                self.assertEqual(coordinator.next_index, 1)
                self.assertEqual(coordinator.last_refusal["code"], "dispatch_mutated")
        # even an adapter that restores its copy cannot pass a wrong end: validation re-derives the intended event
        fakes = {"wind": ph.FakeAdapter("wind"), "interevent": ph.FakeAdapter("interevent", end_override=3600.0 + 2.0)}
        coordinator = Coordinator(scenario, fakes)
        coordinator.step()
        self.assertRefusal("realized_end_mismatch", coordinator.step)
        # a cursor accepted by the fixed coordinator always resumes
        fakes = {"wind": ph.FakeAdapter("wind"), "interevent": ph.FakeAdapter("interevent")}
        coordinator = Coordinator(scenario, fakes)
        coordinator.run()
        record = json.loads(json.dumps(coordinator.as_record()))
        self.assertEqual(Coordinator.resume(scenario, record, {"wind": ph.FakeAdapter("wind"), "interevent": ph.FakeAdapter("interevent")}).as_record(), record)

    def test_public_records_are_independent_copies(self) -> None:
        scenario = self.scenario(ph.quiet_events(), weather=False, name="copies")
        coordinator = Coordinator(scenario, {"wind": ph.FakeAdapter("wind"), "interevent": ph.FakeAdapter("interevent")})
        dispatch = coordinator.dispatch()
        dispatch["planned_edges"]["edges"][-1]["native"] = encode_time(1.0)
        dispatch["processes"]["growth"] = True
        self.assertNotEqual(coordinator.dispatch(), dispatch)
        realized = coordinator.step()
        before = self.snapshot(coordinator)
        realized["native_end"]["hex"] = "0x1.0p+0"
        realized["processes"]["growth"] = True
        public = coordinator.realized[0]
        public["calendar_end"]["iso"] = "1999-01-01T00:00:00+00:00"
        record = coordinator.as_record()
        record["ledger"]["entries"][0]["native_end"]["hex"] = "0x0.0p+0"
        record["occurrences"][0]["processes"]["evaporation"] = True
        record["adapters"]["wind"]["continuous_soil_moisture"] = True
        ledger = coordinator.ledger
        ledger.append("idle", native_start=3600.0, native_end=3601.0, commit_before=3600.0, commit_after=3600.0)
        self.assertEqual(self.snapshot(coordinator), before)
        self.assertEqual(len(coordinator.ledger.records()), 1)
        resumed = Coordinator.resume(scenario, json.loads(before), {"wind": ph.FakeAdapter("wind"), "interevent": ph.FakeAdapter("interevent")})
        self.assertEqual(self.snapshot(resumed), before)

    def test_dispatch_is_pure_and_carries_contracts(self) -> None:
        scenario = self.scenario(ph.quiet_events(), weather=False, name="pure")
        coordinator = Coordinator(scenario, {"wind": ph.FakeAdapter("wind"), "interevent": ph.FakeAdapter("interevent")})
        before = self.snapshot(coordinator)
        first = coordinator.dispatch()
        self.assertEqual(coordinator.dispatch(), first)
        self.assertEqual(self.snapshot(coordinator), before)
        self.assertEqual(first["native_start"], encode_time(0.0))
        self.assertEqual(first["commit_before_expected"], encode_time(0.0))
        self.assertEqual(first["adapter"]["native_staging"], "staged_until_accepted")
        self.assertIn("read-only", first["dispatch_contract"])
        self.assertEqual(first["calendar"]["capability_preflight"]["checked_before_dispatch"], True)
        coordinator.run()
        self.assertRefusal("scenario_finished", coordinator.dispatch)


@ph.requires_library
class ProtocolTests(CoordinatorCase):
    def test_sequence_realizes_exact_durations_recession_and_dates(self) -> None:
        scenario = self.scenario()
        fakes = ph.adapters(rain=ph.FakeAdapter("rain", recession_s=1234.5))
        coordinator = Coordinator(scenario, fakes, latitude=ph.lat())
        realized = coordinator.run()
        self.assertTrue(coordinator.finished)
        self.assertEqual([r["occurrence_id"] for r in realized], [o.occurrence_id for o in scenario.occurrences])
        ends = [decode_time(r["native_end"]) for r in realized]
        self.assertEqual(ends[0], 2 * DAY)
        self.assertEqual(ends[1], 2 * DAY + 3600.0)
        self.assertEqual(ends[2], 3 * DAY)
        self.assertEqual(ends[3], 3 * DAY + 1800.0 + 1234.5)  # realized rain end includes recession
        delta = Fraction(1234.5 - 7200.0)
        self.assertEqual(realized[3]["realized_minus_planned_end_s"], f"{delta.numerator}/{delta.denominator}")
        self.assertEqual(decode_time(realized[4]["native_start"]), ends[3])  # next starts at the REALIZED end
        self.assertEqual(ends[4], ends[3] + 2 * DAY)
        self.assertEqual(realized[0]["calendar_start"]["iso"], "2027-12-30T00:00:00+00:00")
        self.assertEqual(realized[0]["calendar_end"]["iso"], "2028-01-01T00:00:00+00:00")  # year boundary crossed
        self.assertEqual(realized[0]["weather_dates"], ["2027-12-30", "2027-12-31"])
        self.assertEqual(realized[2]["weather_dates"], ["2028-01-01"])  # 01:00 -> 24:00, one touched day
        self.assertEqual(realized[4]["calendar_start"]["iso"], "2028-01-02T00:50:34+00:00")
        self.assertEqual(realized[4]["calendar_start"]["subsecond"], "1/2")
        self.assertEqual(realized[4]["weather_dates"], ["2028-01-02", "2028-01-03", "2028-01-04"])
        self.assertEqual([e["kind"] for e in coordinator.ledger.records()], ["event"] * 6)
        self.assertFalse(coordinator.soil_moisture_continuity()["continuous_soil_moisture_across_events"])

    def test_process_gate_and_no_retroactive_weather(self) -> None:
        scenario = self.scenario()
        fakes = ph.adapters()
        Coordinator(scenario, fakes, latitude=ph.lat()).run()
        for adapter in (fakes["wind"], fakes["rain"]):
            for dispatch in adapter.dispatches:
                self.assertEqual(dispatch["processes"], {"evaporation": False, "transpiration": False, "growth": False})
                self.assertIsNone(dispatch["weather"])
                self.assertIn("calendar", dispatch)  # calendar date still carried
        first, second, third = fakes["interevent"].dispatches
        self.assertEqual(first["processes"], {"evaporation": True, "transpiration": True, "growth": True})
        self.assertEqual(second["processes"]["growth"], False)
        self.assertEqual(first["weather"]["dates"], ["2027-12-30", "2027-12-31"])
        self.assertEqual(first["calendar"]["whole_days"], 2)
        self.assertIsNone(second["calendar"]["whole_days"])  # starts 01:00: not whole days
        self.assertEqual(third["weather"]["dates"], ["2028-01-02", "2028-01-03", "2028-01-04"])
        self.assertEqual([p["date"] for p in third["weather"]["daily_pet"]], third["weather"]["dates"])
        self.assertEqual(third["weather"]["daily_pet"][0]["method"], "ecohyd_hargreaves")
        self.assertEqual(third["weather"]["library"]["module"], "maple_syrup.weather_pet")
        rain_dispatch = fakes["rain"].dispatches[0]
        self.assertEqual(rain_dispatch["calendar"]["rainfall_end"]["iso"], "2028-01-02T00:30:00+00:00")
        self.assertEqual(rain_dispatch["edge_roles"][-2:], ["rainfall_end", "recession_guard_end"])

    def test_missing_unused_and_invalid_adapters_refuse(self) -> None:
        scenario = self.scenario()
        self.assertRefusal("adapter_missing", Coordinator, scenario, {"wind": ph.FakeAdapter("wind")}, latitude=ph.lat())
        self.assertRefusal("adapter_missing", Coordinator, scenario, None, latitude=ph.lat())
        extra = ph.adapters()
        extra["other"] = ph.FakeAdapter("wind")
        self.assertRefusal("adapter_unused", Coordinator, scenario, extra, latitude=ph.lat())
        self.assertRefusal("invalid_capability", Coordinator, scenario, ph.adapters(rain=ph.FakeAdapter("wind")), latitude=ph.lat())
        self.assertRefusal("invalid_adapter", Coordinator, scenario, ph.adapters(rain=object()), latitude=ph.lat())
        self.assertRefusal("latitude_required", Coordinator, scenario, ph.adapters())
        with self.assertRaises(CoordinatorError):
            AdapterCapabilities("wind", "x", "y", "no", "any_positive_duration", "proleptic_gregorian_dates", "not_provided",
                                "staged_until_accepted", "")

    def test_dry_reset_adapters_cannot_satisfy_continuous_requirement(self) -> None:
        scenario = self.scenario(requires_continuous=True)
        self.assertRefusal("continuous_soil_moisture_unavailable", Coordinator, scenario, ph.adapters(), latitude=ph.lat())
        all_continuous = ph.adapters(wind=ph.FakeAdapter("wind", continuous=True), rain=ph.FakeAdapter("rain", continuous=True))
        self.assertTrue(Coordinator(scenario, all_continuous, latitude=ph.lat()).soil_moisture_continuity()["continuous_soil_moisture_across_events"])

    def test_adapter_readiness_limits_with_weather(self) -> None:
        scenario = self.scenario()
        daily = ph.adapters(interevent=ph.FakeAdapter("interevent", continuous=True, duration_policy="whole_days_86400_only"))
        coordinator = Coordinator(scenario, daily, latitude=ph.lat())
        coordinator.step()  # interevent1: 2 whole days from midnight
        coordinator.step()  # wind1
        before = self.snapshot(coordinator)
        self.assertRefusal("adapter_duration_policy_violation", coordinator.step)  # interevent2 starts at 01:00
        self.assertEqual(self.snapshot(coordinator), before)
        self.assertEqual(daily["interevent"].dispatches[-1]["occurrence_id"], "interevent1")  # never dispatched
        short = [("ie", "events/ie.json", ph.interevent(DAY), None)]
        calendar = {"start": "2028-03-01T00:00:00+00:00", "convention": "proleptic_gregorian", "leap_seconds": "none"}
        weather = ph.weather(("2028-02-27", "2028-02-28", "2028-02-29", "2028-03-01"))
        scenario = load_scenario(ph.build_tree(self.tmp / "noleap_ok", short, weather_doc=weather, calendar=calendar))
        noleap = {"interevent": ph.FakeAdapter("interevent", continuous=True, calendar_support="noleap_365_day_index")}
        Coordinator(scenario, noleap, latitude=ph.lat()).run()
        self.assertEqual(noleap["interevent"].dispatches[0]["calendar"]["noleap_day_index"], [59])

    def test_weather_coverage_missing_refuses_before_dispatch(self) -> None:
        events = ph.standard_events() + [("interevent4", "events/ie2d.json", ph.interevent(), None),
                                         ("interevent5", "events/ie2d.json", ph.interevent(), None)]
        coordinator = Coordinator(self.scenario(events), ph.adapters(), latitude=ph.lat())
        for _ in range(7):
            coordinator.step()
        before = self.snapshot(coordinator)
        self.assertRefusal("weather_coverage_missing", coordinator.step)  # 2028-01-07 has no weather row
        self.assertEqual(self.snapshot(coordinator), before)


@ph.requires_library
class FailureAndBoundaryTests(CoordinatorCase):
    def test_incomplete_outcome_leaves_cursor_unchanged(self) -> None:
        scenario = self.scenario()
        coordinator = Coordinator(scenario, ph.adapters(wind=ph.FakeAdapter("wind", status="incomplete")), latitude=ph.lat())
        coordinator.step()
        before = self.snapshot(coordinator)
        error = self.assertRefusal("event_incomplete", coordinator.step)
        self.assertIn("not completion", error.detail)
        self.assertEqual(self.snapshot(coordinator), before)
        self.assertEqual(coordinator.next_index, 1)
        self.assertIs(type(coordinator.active_evidence), AdapterOutcome)
        self.assertRefusal("event_incomplete", coordinator.step)
        self.assertEqual(self.snapshot(coordinator), before)

    def test_adapter_exception_and_bad_ends_are_refused_atomically(self) -> None:
        scenario = self.scenario()
        cases = (
            (ph.adapters(interevent=ph.FakeAdapter("interevent", continuous=True, raise_exc=RuntimeError("boom"))), "adapter_failure", 0),
            (ph.adapters(interevent=ph.FakeAdapter("interevent", continuous=True, end_override=2 * DAY + 1.0)), "realized_end_mismatch", 0),
            (ph.adapters(wind=ph.FakeAdapter("wind", end_override=2 * DAY + 3599.0)), "realized_end_mismatch", 1),
            (ph.adapters(rain=ph.FakeAdapter("rain", recession_s=7200.1)), "rain_end_outside_guard", 3),
            (ph.adapters(rain=ph.FakeAdapter("rain", recession_s=-1.0)), "rain_end_outside_guard", 3),
            (ph.adapters(rain=ph.FakeAdapter("rain", end_override=float("nan"))), "rain_end_outside_guard", 3),
            (ph.adapters(wind=ph.FakeAdapter("wind", end_override=None)), "invalid_outcome", 1),
        )
        for fakes, code, failing_index in cases:
            with self.subTest(code=code):
                coordinator = Coordinator(scenario, fakes, latitude=ph.lat())
                for _ in range(failing_index):
                    coordinator.step()
                before = self.snapshot(coordinator)
                self.assertRefusal(code, coordinator.step)
                self.assertEqual(self.snapshot(coordinator), before)
                self.assertEqual(coordinator.next_index, failing_index)
                self.assertEqual(len(coordinator.ledger.records()), failing_index)

    def test_interevent_cannot_commit_and_commit_chain_is_checked(self) -> None:
        scenario = self.scenario()

        class CommittingInterevent(ph.FakeAdapter):
            def run(self, dispatch):
                out = super().run(dispatch)
                return AdapterOutcome("complete", out.native_end_s, None, out.native_end_s)

        self.assertRefusal("interevent_commit_refused",
                           Coordinator(scenario, ph.adapters(interevent=CommittingInterevent("interevent", continuous=True)), latitude=ph.lat()).step)

        class FutureCommit(ph.FakeAdapter):
            def run(self, dispatch):
                out = super().run(dispatch)
                return AdapterOutcome("complete", out.native_end_s, None, out.native_end_s + 1.0)

        coordinator = Coordinator(scenario, ph.adapters(wind=FutureCommit("wind")), latitude=ph.lat())
        coordinator.step()
        self.assertRefusal("clock_refused", coordinator.step)
        with self.assertRaises(CoordinatorError):
            Coordinator(scenario, ph.adapters(), latitude=ph.lat(), initial_commit_s=5.0)
        self.assertRefusal("invalid_initial_commit", Coordinator, scenario, ph.adapters(), latitude=ph.lat(), initial_commit_s=-0.0)

    def test_scheduled_start_overlap_refused_gap_policy(self) -> None:
        events = ph.standard_events()
        events[4] = (events[4][0], events[4][1], events[4][2], "2028-01-02T00:40:00+00:00")  # 261000 + 600 s
        scenario = self.scenario(events, name="refuse")
        fakes = ph.adapters(rain=ph.FakeAdapter("rain", recession_s=700.0))  # realized end 261700 > scheduled 261600
        coordinator = Coordinator(scenario, fakes, latitude=ph.lat())
        for _ in range(4):
            coordinator.step()
        before = self.snapshot(coordinator)
        error = self.assertRefusal("scheduled_start_overlap", coordinator.step)
        self.assertIn("never truncated", error.detail)
        self.assertEqual(self.snapshot(coordinator), before)
        self.assertEqual(fakes["interevent"].dispatches[-1]["occurrence_id"], "interevent2")
        coordinator = Coordinator(scenario, ph.adapters(rain=ph.FakeAdapter("rain", recession_s=100.0)), latitude=ph.lat())
        for _ in range(4):
            coordinator.step()
        self.assertRefusal("uncovered_gap_refused", coordinator.step)  # gap under the default policy
        realized = Coordinator(scenario, ph.adapters(rain=ph.FakeAdapter("rain", recession_s=600.0)), latitude=ph.lat()).run()
        self.assertEqual(decode_time(realized[4]["native_start"]), 261600.0)  # exact coincidence
        self.assertIsNone(realized[4]["gap_ledger_entry_index"])

    def test_gap_policy_idle_records_explicit_no_process_interval(self) -> None:
        events = ph.standard_events()
        events[4] = (events[4][0], events[4][1], events[4][2], "2028-01-02T03:00:00+00:00")
        scenario = self.scenario(events, gap_policy="idle", name="idle")
        fakes = ph.adapters(rain=ph.FakeAdapter("rain", recession_s=100.0))
        coordinator = Coordinator(scenario, fakes, latitude=ph.lat())
        realized = coordinator.run()
        entries = coordinator.ledger.records()
        self.assertEqual([e["kind"] for e in entries], ["event"] * 4 + ["idle", "event", "event"])
        self.assertEqual((realized[4]["gap_ledger_entry_index"], realized[4]["ledger_entry_index"]), (4, 5))
        self.assertEqual((decode_time(entries[4]["native_start"]), decode_time(entries[4]["native_end"])), (261100.0, 270000.0))
        self.assertEqual(entries[4]["commit_before"], entries[4]["commit_after"])
        self.assertEqual(realized[4]["calendar_start"]["iso"], "2028-01-02T03:00:00+00:00")
        gap = fakes["interevent"].dispatches[2]["gap_entry"]
        self.assertEqual(gap["processes"], {"evaporation": False, "transpiration": False, "growth": False})
        self.assertEqual(realized[4]["weather_dates"], ["2028-01-02", "2028-01-03", "2028-01-04"])  # own interval only


@ph.requires_library
class CursorRestartTests(CoordinatorCase):
    def test_cursor_round_trip_identity_at_every_boundary(self) -> None:
        scenario = self.scenario()
        coordinator = Coordinator(scenario, ph.adapters(rain=ph.FakeAdapter("rain", recession_s=250.25)), latitude=ph.lat())
        while True:
            record = json.loads(json.dumps(coordinator.as_record()))
            resumed = Coordinator.resume(scenario, record, ph.adapters(rain=ph.FakeAdapter("rain", recession_s=250.25)), latitude=ph.lat())
            self.assertEqual(resumed.as_record(), record)
            self.assertEqual(resumed.digest, coordinator.digest)
            if coordinator.finished:
                break
            coordinator.step()
            resumed.step()  # continuation from the resumed cursor equals the uninterrupted one
            self.assertEqual(resumed.as_record(), coordinator.as_record())
        self.assertIn("cannot roll back a native model", record["state_scope"])
        self.assertFalse(record["soil_moisture_continuity"]["continuous_soil_moisture_across_events"])

    def test_source_hash_mismatch_on_resume(self) -> None:
        scenario = self.scenario()
        coordinator = Coordinator(scenario, ph.adapters(), latitude=ph.lat())
        coordinator.step()
        record = coordinator.as_record()
        path = self.tmp / "s" / "events" / "wind.json"
        changed = path.read_bytes().replace(b"1800.0", b"1801.0", 1)
        path.write_bytes(changed)
        doc = json.loads((self.tmp / "s" / "scenario.json").read_text())
        for row in doc["events"]:
            if row["file"] == "events/wind.json":
                row["sha256"] = ph.sha(changed)
        (self.tmp / "s" / "scenario.json").write_text(json.dumps(doc))
        reloaded = load_scenario(self.tmp / "s" / "scenario.json")
        self.assertRefusal("scenario_source_mismatch", Coordinator.resume, reloaded, record, ph.adapters(), latitude=ph.lat())

    def test_identity_and_tamper_detection_on_resume(self) -> None:
        scenario = self.scenario()
        coordinator = Coordinator(scenario, ph.adapters(), latitude=ph.lat())
        coordinator.step()
        coordinator.step()
        good = coordinator.as_record()
        self.assertRefusal("adapter_identity_mismatch", Coordinator.resume, scenario, good,
                           ph.adapters(wind=ph.FakeAdapter("wind", adapter_id="other_wind")), latitude=ph.lat())
        self.assertRefusal("adapter_identity_mismatch", Coordinator.resume, scenario, good,
                           ph.adapters(rain=ph.FakeAdapter("rain", continuous=True)), latitude=ph.lat())
        self.assertRefusal("latitude_mismatch", Coordinator.resume, scenario, good, ph.adapters(), latitude=ph.lat(31.0))

        def mutated(change):
            record = json.loads(json.dumps(good))
            change(record)
            return record

        tampered = [
            ("cursor_inconsistent", mutated(lambda r: r.__setitem__("next_index", 1))),
            ("cursor_inconsistent", mutated(lambda r: r["occurrences"][0].__setitem__("calendar_end", r["occurrences"][1]["calendar_end"]))),
            ("cursor_inconsistent", mutated(lambda r: r["occurrences"][0].__setitem__("weather_dates", ["2027-12-30"]))),
            ("cursor_inconsistent", mutated(lambda r: r["occurrences"][1].__setitem__("ledger_entry_index", 0))),
            ("cursor_inconsistent", mutated(lambda r: r["occurrences"].pop())),
            ("cursor_inconsistent", mutated(lambda r: r["occurrences"][0].__setitem__("processes", {"evaporation": False, "transpiration": True, "growth": True}))),
            ("cursor_inconsistent", mutated(lambda r: r["occurrences"][0].__setitem__("event_digest", "0" * 64))),
            ("malformed_cursor", mutated(lambda r: r["ledger"]["entries"][0].__setitem__("global_end", "1/1"))),
            ("malformed_cursor", mutated(lambda r: r.pop("ledger"))),
            ("unsupported_cursor_schema", mutated(lambda r: r.__setitem__("schema", "masy.coordinator_cursor.v2"))),
            ("cursor_inconsistent", mutated(lambda r: r.__setitem__("finished", True))),
        ]
        for code, record in tampered:
            with self.subTest(code=code):
                self.assertRefusal(code, Coordinator.resume, scenario, record, ph.adapters(), latitude=ph.lat())
        extra = json.loads(json.dumps(good))
        last = extra["ledger"]["entries"][-1]
        end = decode_time(last["native_end"])
        nxt = dict(last)
        nxt.update(index=2, kind="idle", native_start=last["native_end"], native_end=encode_time(end + 1.0),
                   global_start=last["global_end"], global_end=f"{Fraction(end + 1.0).numerator}/{Fraction(end + 1.0).denominator}",
                   endpoint_elapsed="1/1", commit_before=last["commit_after"], returned_elapsed=None, endpoint_discrepancy=None)
        extra["ledger"]["entries"].append(nxt)
        self.assertRefusal("cursor_inconsistent", Coordinator.resume, scenario, extra, ph.adapters(), latitude=ph.lat())


if __name__ == "__main__":
    unittest.main()
