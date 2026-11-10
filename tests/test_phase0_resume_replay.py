"""Phase 0 correction 02: resume replays the accepted prefix through the live schedule rules; records are owned.

Every tampered cursor below is internally consistent as a clock ledger (ClockLedger.from_record accepts it); the
refusal must come from the replayed scheduling/gap/capability/commit rules, not from ledger syntax. Scenarios are
weather-free unless noted, so no PET library is needed for most cases.
"""

from __future__ import annotations

import json
import unittest
from fractions import Fraction

import _phase0_support as ph
from _support import TempDirCase

from masy.clock import ClockLedger, decode_time, encode_time
from masy.coordinator import AdapterOutcome, Coordinator, CoordinatorError
from masy.scenario import load_scenario

QUIET = "events/quiet.json"


def quiet(duration_s: float, event_id: str = "quiet") -> dict:
    return ph.interevent(duration_s, False, False, False, event_id=event_id)


def regenerate_ledger(record: dict) -> dict:
    """Re-serialize the ledger from a list of (kind, start, end, before, after, returned) so the derived fields
    (global times, elapsed, discrepancy, index) stay internally consistent: the result passes ClockLedger.from_record."""
    ledger = ClockLedger(decode_time(record["ledger"]["native_epoch"]))
    for entry in record["ledger"]["entries"]:
        ledger.append(entry["kind"], native_start=decode_time(entry["native_start"]), native_end=decode_time(entry["native_end"]),
                      commit_before=decode_time(entry["commit_before"]), commit_after=decode_time(entry["commit_after"]),
                      returned_elapsed=None if entry["returned_elapsed"] is None else decode_time(entry["returned_elapsed"]))
    record["ledger"] = ledger.as_record()
    ClockLedger.from_record(record["ledger"])  # the tampered ledger itself is valid
    return record


class ReplayCase(TempDirCase):
    def adapters(self, scenario=None, **overrides):
        kinds = ("wind", "interevent") if scenario is None else sorted(scenario.kinds_present)
        base = {kind: ph.FakeAdapter(kind) for kind in kinds}
        base.update(overrides)
        return base

    def scenario(self, name: str, events, **kwargs):
        return load_scenario(ph.build_tree(self.tmp / name, events, weather_doc=None, **kwargs))

    def assertResumeRefused(self, scenario, record, inner: str, adapters=None) -> None:
        with self.assertRaises(CoordinatorError) as ctx:
            Coordinator.resume(scenario, record, adapters or self.adapters())
        self.assertEqual(ctx.exception.code, "cursor_inconsistent", ctx.exception.detail)
        self.assertIn(inner, ctx.exception.detail)

    def genuine(self, scenario, adapters=None) -> dict:
        coordinator = Coordinator(scenario, adapters or self.adapters())
        coordinator.run()
        return json.loads(json.dumps(coordinator.as_record()))


class ScheduledStartReplayTests(ReplayCase):
    def test_review_reproducer_scheduled_start_mismatch_is_refused(self) -> None:
        # Scenario A: first quiet 10 s event scheduled at +20 s under gap_policy idle; B identical but +30 s.
        a = self.scenario("a", [("q", QUIET, quiet(10.0), "2027-12-30T00:00:20+00:00")], gap_policy="idle")
        b = self.scenario("b", [("q", QUIET, quiet(10.0), "2027-12-30T00:00:30+00:00")], gap_policy="idle")
        record_b = self.genuine(b, self.adapters(b))
        self.assertEqual(decode_time(record_b["occurrences"][0]["native_start"]), 30.0)
        forged = dict(record_b, scenario_sha256=a.digest)
        self.assertResumeRefused(a, forged, "replay of the accepted prefix refused", self.adapters(a))  # replay derives +20 s
        resumed = Coordinator.resume(a, self.genuine(a, self.adapters(a)), self.adapters(a))  # the genuine A cursor still resumes
        self.assertEqual(decode_time(resumed.realized[0]["native_start"]), 20.0)

    def test_scheduled_start_early_and_late_refused_on_replay(self) -> None:
        # A: wind 3600 s, then quiet 10 s scheduled exactly at the wind end (no gap, consistent)
        a = self.scenario("exact", [("w", "events/wind.json", ph.wind(), None),
                                    ("q", QUIET, quiet(10.0), "2027-12-30T01:00:00+00:00")])
        good = self.genuine(a)
        Coordinator.resume(a, good, self.adapters())
        # early: the same ledger claims the second occurrence started 1 s before the realized wind end
        early = json.loads(json.dumps(good))
        entry = early["ledger"]["entries"][1]
        entry["native_start"] = encode_time(3599.0)
        early["ledger"]["entries"][0]["native_end"] = encode_time(3599.0)  # keep contiguity, so only scheduling differs
        early["ledger"]["entries"][0]["commit_after"] = encode_time(3599.0)
        entry["commit_before"] = encode_time(3599.0)
        entry["commit_after"] = encode_time(3599.0)
        entry["native_end"] = encode_time(3609.0)
        early["occurrences"][0]["native_end"] = encode_time(3599.0)
        early["occurrences"][1]["native_start"] = encode_time(3599.0)
        early["occurrences"][1]["native_end"] = encode_time(3609.0)
        regenerate_ledger(early)
        self.assertResumeRefused(a, early, "realized_end_mismatch")  # wind must end at 3600: replay refuses first
        # late under gap_policy refuse: a record carrying an idle gap before the scheduled start is refused
        late = json.loads(json.dumps(good))
        late["ledger"]["entries"].insert(1, {**late["ledger"]["entries"][0], "kind": "idle", "native_start": encode_time(3600.0),
                                             "native_end": encode_time(3610.0), "commit_before": encode_time(3600.0),
                                             "commit_after": encode_time(3600.0), "returned_elapsed": None})
        late["ledger"]["entries"][2]["native_start"] = encode_time(3610.0)
        late["ledger"]["entries"][2]["native_end"] = encode_time(3620.0)
        late["occurrences"][1].update(gap_ledger_entry_index=1, ledger_entry_index=2, native_start=encode_time(3610.0),
                                      native_end=encode_time(3620.0))
        regenerate_ledger(late)
        self.assertResumeRefused(a, late, "replay of the accepted prefix refused")

    def test_spurious_idle_and_idle_under_refuse_policy(self) -> None:
        # unscheduled sequence: an idle entry can never appear, regardless of policy
        for policy in ("refuse", "idle"):
            with self.subTest(policy=policy):
                scenario = self.scenario(f"seq_{policy}", [("w", "events/wind.json", ph.wind(), None), ("q", QUIET, quiet(10.0), None)],
                                         gap_policy=policy)
                good = self.genuine(scenario)
                spurious = json.loads(json.dumps(good))
                spurious["ledger"]["entries"].insert(1, {**spurious["ledger"]["entries"][0], "kind": "idle",
                                                         "native_start": encode_time(3600.0), "native_end": encode_time(3605.0),
                                                         "commit_before": encode_time(3600.0), "commit_after": encode_time(3600.0),
                                                         "returned_elapsed": None})
                spurious["ledger"]["entries"][2]["native_start"] = encode_time(3605.0)
                spurious["ledger"]["entries"][2]["native_end"] = encode_time(3615.0)
                spurious["occurrences"][1].update(gap_ledger_entry_index=1, ledger_entry_index=2, native_start=encode_time(3605.0),
                                                  native_end=encode_time(3615.0))
                regenerate_ledger(spurious)
                # replay dispatches without a gap, so the recorded idle/start shift is refused by the live rules
                self.assertResumeRefused(scenario, spurious, "replay of the accepted prefix refused")

    def test_valid_rain_recession_determines_the_next_start_on_replay(self) -> None:
        events = [("r", "events/rain.json", ph.rain(), None), ("q", QUIET, quiet(10.0), None)]
        scenario = self.scenario("rain", events)
        for recession in (0.0, 123.456, 7200.0):
            with self.subTest(recession=recession):
                adapters = {"rain": ph.FakeAdapter("rain", recession_s=recession), "interevent": ph.FakeAdapter("interevent")}
                record = self.genuine(scenario, adapters)
                resumed = Coordinator.resume(scenario, record, adapters)
                self.assertEqual(decode_time(resumed.realized[1]["native_start"]), 1800.0 + recession)
                self.assertEqual(resumed.as_record(), record)
        # a realized rain end outside the guard is refused even though the ledger is self-consistent
        adapters = {"rain": ph.FakeAdapter("rain", recession_s=10.0), "interevent": ph.FakeAdapter("interevent")}
        record = self.genuine(scenario, adapters)
        record["ledger"]["entries"][0]["native_end"] = encode_time(1800.0 + 7200.0 + 1.0)
        record["ledger"]["entries"][0]["commit_after"] = encode_time(1800.0 + 7200.0 + 1.0)
        record["ledger"]["entries"][1]["native_start"] = encode_time(1800.0 + 7200.0 + 1.0)
        record["ledger"]["entries"][1]["commit_before"] = encode_time(1800.0 + 7200.0 + 1.0)
        record["ledger"]["entries"][1]["commit_after"] = encode_time(1800.0 + 7200.0 + 1.0)
        record["ledger"]["entries"][1]["native_end"] = encode_time(1800.0 + 7200.0 + 11.0)
        for occ, start, end in ((0, 0.0, 9001.0), (1, 9001.0, 9011.0)):
            record["occurrences"][occ]["native_start"] = encode_time(start)
            record["occurrences"][occ]["native_end"] = encode_time(end)
        regenerate_ledger(record)
        self.assertResumeRefused(scenario, record, "rain_end_outside_guard", adapters)


class CommitAndCapabilityReplayTests(ReplayCase):
    def test_interevent_commit_change_is_refused_on_replay(self) -> None:
        scenario = self.scenario("ie", [("w", "events/wind.json", ph.wind(), None), ("q", QUIET, quiet(10.0), None)])
        good = self.genuine(scenario)
        changed = json.loads(json.dumps(good))
        changed["ledger"]["entries"][1]["commit_after"] = encode_time(3610.0)  # an interevent must not commit the bed
        regenerate_ledger(changed)
        self.assertResumeRefused(scenario, changed, "interevent_commit_refused")

    def test_wrong_first_commit_is_refused_on_replay_and_live(self) -> None:
        scenario = self.scenario("commit", [("w", "events/wind.json", ph.wind(), None)], native_epoch_s=100.0)
        adapters = self.adapters(scenario)
        coordinator = Coordinator(scenario, adapters, initial_commit_s=50.0)
        coordinator.run()
        good = json.loads(json.dumps(coordinator.as_record()))
        self.assertEqual(decode_time(good["ledger"]["entries"][0]["commit_before"]), 50.0)
        self.assertEqual(Coordinator.resume(scenario, good, self.adapters(scenario)).as_record(), good)
        tampered = json.loads(json.dumps(good))
        tampered["ledger"]["entries"][0]["commit_before"] = encode_time(40.0)  # valid for ClockLedger (<= start) but off chain
        regenerate_ledger(tampered)
        self.assertResumeRefused(scenario, tampered, "commit_chain_refused", self.adapters(scenario))
        other_initial = json.loads(json.dumps(good))
        other_initial["initial_commit"] = encode_time(40.0)  # declared initial commit the first entry does not continue
        self.assertResumeRefused(scenario, other_initial, "commit_chain_refused", self.adapters(scenario))

        class OffChain(ph.FakeAdapter):
            def run(self, dispatch):
                out = super().run(dispatch)
                return AdapterOutcome("complete", out.native_end_s, 40.0, out.native_end_s)

        live = Coordinator(scenario, {"wind": OffChain("wind")}, initial_commit_s=50.0)
        with self.assertRaises(CoordinatorError) as ctx:  # live: a commit_before off the accepted chain is refused
            live.step()
        self.assertEqual(ctx.exception.code, "commit_chain_refused")
        self.assertEqual(live.next_index, 0)

    def test_capability_preflight_applies_on_replay(self) -> None:
        scenario = self.scenario("cap", [("w", "events/wind.json", ph.wind(), None), ("q", QUIET, quiet(10.0), None)])
        good = self.genuine(scenario)
        daily = {"wind": ph.FakeAdapter("wind"), "interevent": ph.FakeAdapter("interevent", duration_policy="whole_days_86400_only")}
        with self.assertRaises(CoordinatorError) as ctx:  # capability identity differs first
            Coordinator.resume(scenario, good, daily)
        self.assertEqual(ctx.exception.code, "adapter_identity_mismatch")
        forged = json.loads(json.dumps(good))
        forged["adapters"]["interevent"]["duration_policy"] = "whole_days_86400_only"
        self.assertResumeRefused(scenario, forged, "adapter_duration_policy_violation", daily)


@ph.requires_library
class WeatherCoverageReplayTests(ReplayCase):
    def test_weather_coverage_is_rechecked_on_replay(self) -> None:
        events = [("ie", "events/ie1d.json", ph.interevent(86400.0, event_id="ie1d"), None)]
        full = load_scenario(ph.build_tree(self.tmp / "full", events, weather_doc=ph.weather(ph.WEEK)))
        adapters = {"interevent": ph.FakeAdapter("interevent", continuous=True)}
        coordinator = Coordinator(full, adapters, latitude=ph.lat())
        coordinator.run()
        record = json.loads(json.dumps(coordinator.as_record()))
        self.assertEqual(Coordinator.resume(full, record, adapters, latitude=ph.lat()).as_record(), record)
        short = load_scenario(ph.build_tree(self.tmp / "short", events, weather_doc=ph.weather(ph.WEEK[:2])))  # ends 12-29
        with self.assertRaises(CoordinatorError):  # digest differs: scenario_source_mismatch before replay
            Coordinator.resume(short, record, adapters, latitude=ph.lat())
        forged = dict(record, scenario_sha256=short.digest)
        with self.assertRaises(CoordinatorError) as ctx:
            Coordinator.resume(short, forged, adapters, latitude=ph.lat())
        self.assertEqual(ctx.exception.code, "cursor_inconsistent")
        self.assertIn("weather_coverage_missing", ctx.exception.detail)


class OwnershipTests(ReplayCase):
    def test_accepted_source_bound_records_are_read_only(self) -> None:
        scenario = self.scenario("own", [("w", "events/wind.json", ph.wind(), None), ("q", QUIET, quiet(10.0), None)])
        with self.assertRaises(TypeError):
            scenario.occurrences[1].event.processes["growth"] = True
        binding = ph.lat()
        with self.assertRaises(TypeError):
            binding.source["sha256"] = "1" * 64
        record = binding.record()
        record["source"]["sha256"] = "1" * 64
        self.assertEqual(binding.source["sha256"], "0" * 64)

    @ph.requires_library
    def test_weather_rows_and_coverage_are_owned(self) -> None:
        from masy.weather import WeatherSeries
        series = WeatherSeries.from_input(ph.weather(ph.WEEK))
        with self.assertRaises(TypeError):
            series.rows[0]["tmean_c"] = 99.0
        with self.assertRaises(TypeError):
            series.rows[0]["derived"]["rh_fraction"] = 0.0
        with self.assertRaises(TypeError):
            series.provenance["kind"] = "observed"
        import datetime as dt
        digest = series.digest
        rows = series.coverage((dt.date(2027, 12, 28),))
        rows[0]["tmean_c"] = 99.0
        rows[0]["derived"]["rh_fraction"] = 0.0
        self.assertEqual(series.coverage((dt.date(2027, 12, 28),))[0]["tmean_c"], 8.0)
        rec = series.record()
        rec["rows"][0]["tmean_c"]["hex"] = "0x0.0p+0"
        self.assertEqual(series.digest, digest)
        scenario = load_scenario(ph.build_tree(self.tmp / "wref", ph.standard_events(), weather_doc=ph.weather(ph.WEEK)))
        with self.assertRaises(TypeError):
            scenario.weather_ref["sha256"] = "1" * 64

    def test_scenario_changed_after_binding_is_refused_before_dispatch_and_publication(self) -> None:
        scenario = self.scenario("bind", [("w", "events/wind.json", ph.wind(), None), ("q", QUIET, quiet(10.0), None)])
        coordinator = Coordinator(scenario, self.adapters())
        expected = coordinator.dispatch()

        class Tamper(ph.FakeAdapter):
            def run(self, dispatch):
                # a closure with a reference to the ORIGINAL scenario object bypasses the frozen dataclass
                object.__setattr__(scenario.occurrences[1], "scheduled_start_s", Fraction(7200))
                object.__setattr__(scenario.occurrences[1], "scheduled_start", "2027-12-30T02:00:00+00:00")
                return super().run(dispatch)

        tampering = Coordinator(scenario, {"wind": Tamper("wind"), "interevent": ph.FakeAdapter("interevent")})
        with self.assertRaises(CoordinatorError) as ctx:
            tampering.step()  # refuse before accepting a boundary against a changed source scenario
        self.assertEqual(ctx.exception.code, "scenario_changed")
        self.assertEqual(tampering.next_index, 0)
        self.assertEqual(tampering.ledger.records(), [])
        with self.assertRaises(CoordinatorError) as ctx:
            coordinator.dispatch()
        self.assertEqual(ctx.exception.code, "scenario_changed")
        self.assertNotEqual(coordinator.scenario.digest, expected["scenario_sha256"])
        with self.assertRaises(CoordinatorError) as ctx:
            Coordinator.resume(scenario, {"schema": "masy.coordinator_cursor.v1"}, self.adapters())
        self.assertEqual(ctx.exception.code, "malformed_cursor")
        # a scenario mutated before binding is simply a different scenario: its own fresh coordinator runs
        fresh = Coordinator(scenario, self.adapters())
        fresh.step()
        with self.assertRaises(CoordinatorError) as ctx:
            fresh.step()
        self.assertEqual(ctx.exception.code, "uncovered_gap_refused")


if __name__ == "__main__":
    unittest.main()
