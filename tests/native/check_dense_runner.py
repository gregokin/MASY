"""Explicit native checks of masy.dense_runner on the authored reference fixture.

NOT collected automatically (no `test_` prefix, no package `__init__`). Run it
explicitly with an interpreter whose import paths resolve the locked MAPLE /
SYRUP sources and the MASY candidate `src`, e.g.

    python -I tests/native/check_dense_runner.py

Every positive check runs genuine MAPLE / SYRUP physics through unpatched
native bindings and compares the runner with an independently executed
DIRECT native control (`direct_control`: the call sequence of the verified
source-only reference, written against native APIs only, on a separately built
bed). Negative checks may patch only the MASY delegates
`dense_runner._advance_wind_block` / `dense_runner._complete_event`, always
calling through to (or replacing only the call of) the real native function.
No native type, class, provenance or resolver is faked.
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[2] / "src"
_HERE = Path(__file__).resolve().parent
for _path in (_SRC, _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402
from maple.aeolian.cohorts.dispatch import mobile_state_is_genuinely_empty  # noqa: E402
from maple.aeolian.scheduler.finalization import finalize_wind_event
from maple.aeolian.scheduler.block import advance_wind_block  # noqa: E402
from maple.aeolian.scheduler.state import initial_scheduler_state  # noqa: E402
from maple.core.parameters.avalanching import AvalanchingSpec  # noqa: E402
from maple.core.parameters.grain_classes import GrainClassSet  # noqa: E402
from maple.core.parameters.topographic_commit import TopographicCommitSpec  # noqa: E402
from maple.core.parameters.wind_forcing import WindBlock, resolve_wind_forcing_record  # noqa: E402
from maple.core.types.topographic_commit import committed_elevation_m  # noqa: E402
from maple_syrup.complete_event import DryHandoff, complete_event  # noqa: E402
from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule, constant_rainfall  # noqa: E402
from maple_syrup.sediment_bed import commit_bed, routing_graph_from_elevation  # noqa: E402
from maple_syrup.sediment_event import initial_event_state  # noqa: E402
from maple_syrup.sediment_physics import physics_grid_from_graph  # noqa: E402
from maple_syrup.sediment_transport import transport_network  # noqa: E402
from maple_syrup.storm import StormControl, initial_state  # noqa: E402

import fixture_dense_alternation as fx  # noqa: E402
from masy import dense_bed as adapter  # noqa: E402
from masy import dense_runner  # noqa: E402
from masy.clock import decode_time  # noqa: E402
from masy.dense_bed import SHARED_FIELDS  # noqa: E402
from masy.dense_runner import DenseRunnerError  # noqa: E402


# --- independent direct native control --------------------------------------------------------------------------------

def direct_control(bed, config, events, *, first_graph="reference", water=None, control=None):
    """The verified reference sequence, generalized to an event list, with native calls only.

    `events`: ("wind", duration, shear, direction) / ("rain", ((duration, mm_h), ...), guard).
    `first_graph`: "reference" builds the first graph around the bed at the first rain entry exactly as the
    reference does; "terrain" builds it natively from the initial terrain reference and the current committed
    elevation (`routing_graph_from_elevation`)."""
    ctx = bed.context
    water = water or fx.water_inputs(bed)
    control = control or fx.sediment_control()
    state, pool, t = bed.state, fx.empty_pool(), bed.time_s
    held, soil, log = None, np.zeros((fx.NY, fx.NX)), []
    for event in events:
        if event[0] == "wind":
            _, duration, shear, direction = event
            scheduler = initial_scheduler_state(
                state.voxel_column, state.active_layer, pool, state.ledger, state.committed_topography,
                bed.threshold_modifier_state, time_s=t, vegetation=bed.vegetation,
                sediment_availability=state.sediment_availability, water=state.water)
            result = advance_wind_block(scheduler, WindBlock(shear, direction, duration, t, t + duration), config)
            assert result.conservation_ok
            # Independently call the native completion API on the genuine completed block.
            assert result.state.block_index == 0 and result.state.block_offset_s >= duration - 1e-9
            record = resolve_wind_forcing_record({"blocks": [{"duration_s": duration,
                        "shear_velocity_m_s": shear, "direction_deg": direction}]})
            completed = dataclasses.replace(result.state, block_index=1, block_offset_s=0.0)
            final = finalize_wind_event(completed, config, record, record_origin_s=t)
            assert final.completed and mobile_state_is_genuinely_empty(final.state.physical.mobile_state)
            assert final.state.physical.time_s.hex() == result.state.physical.time_s.hex()
            current = final.state.physical
            state = dataclasses.replace(state, voxel_column=current.voxel_column, active_layer=current.active_layer,
                                        ledger=current.ledger, committed_topography=current.committed_topography,
                                        sediment_availability=current.sediment_availability, water=current.water)
            log.append({"kind": "wind", "start": t, "end": current.time_s,
                        "export": (result.accounting.boundary_export_by_class_kg
                                   + final.boundary_export_mass_by_class_kg).tolist(),
                        "final_deposition": final.deposited_mass_by_class_kg.tolist(),
                        "final_export": final.boundary_export_mass_by_class_kg.tolist(),
                        "detachment": result.accounting.spatial.detachment_by_cell_class_kg.sum(axis=(0, 1)).tolist()})
            pool, t = current.mobile_state, current.time_s
            continue
        _, intervals, guard = event
        if held is None:
            if first_graph == "reference":
                graph, terrain = fx.routing_for(state)
            else:
                terrain = water.terrain
                graph = routing_graph_from_elevation(terrain, committed_elevation_m(state.committed_topography), xp=np)
            held = initial_event_state(graph, terrain, state, ctx, soil, control=control)
        boundary = commit_bed(state, ctx, held.graph, held.terrain, t, force=True,
                              routing_source_m=held.routing_source_elevation_m)
        g = boundary.graph
        entry = dataclasses.replace(
            held, bed=boundary.state, graph=g, routing_source_elevation_m=boundary.routing_source_m,
            network=transport_network(g) if boundary.rerouted else held.network,
            grid=physics_grid_from_graph(g) if boundary.rerouted else held.grid,
            storm=initial_state(g, boundary.state.water.depth_m, soil, t_s=t))
        offsets = np.cumsum([d for d, _ in intervals]).tolist()
        if len(intervals) == 1:
            rainfall = constant_rainfall(t, t + offsets[0], intervals[0][1])
        else:
            rainfall = RainfallSchedule(edges_s=[t] + [t + o for o in offsets],
                                        intensity_mm_per_h=[i for _, i in intervals],
                                        provenance=RainfallProvenance(kind="constant"))
        wet = complete_event(entry, ctx, water.column, water.field, rainfall, water.vegetation_cover_fraction,
                             water.sediment, max_end_s=t + (offsets[-1] + guard), control=control,
                             policy=fx.policy(), report_every_s=fx.REPORT_EVERY_S)
        assert isinstance(wet, DryHandoff)
        result = wet.progress.result
        log.append({"kind": "rain", "start": t, "end": wet.dry_state.t_s, "rerouted": boundary.rerouted,
                    "graph_retained": boundary.graph is held.graph,
                    "export": np.asarray(result.by_class["export_actual"]).tolist(),
                    "pickup": np.asarray(result.by_class["actual_pickup"]).tolist(),
                    "water": wet.accounting["storm_water_budget"]})
        held, state, t = wet.dry_state, wet.dry_state.bed, wet.dry_state.t_s
        soil = held.storm.soil_water_m
    return state, t, log


def wind_event(duration=fx.WIND_S):
    return ("wind", duration, fx.SHEAR, fx.DIRECTION)


def rain_event(intervals=((fx.RAIN_S, fx.RAIN_MM_H),), guard=fx.GUARD_S):
    return ("rain", intervals, guard)


def schedule_of(events):
    return fx.schedule(*(fx.wind(e[1]) if e[0] == "wind" else fx.rain(e[1], e[2]) for e in events))


def field_digests(state) -> dict:
    return {name: adapter._canonical_static(getattr(state, name), name) for name in SHARED_FIELDS}


def available_fraction(state):
    active = state.active_layer.mass_kg
    return state.sediment_availability.available_mass_kg[active > 0] / active[active > 0]


class _RunnerCase(unittest.TestCase):
    def assertSameFields(self, runner_state, control_state) -> None:
        mine, theirs = field_digests(runner_state), field_digests(control_state)
        for name in SHARED_FIELDS:
            self.assertEqual(mine[name], theirs[name], f"persistent field {name} differs from the direct control")

    def assertClocks(self, runner, log) -> None:
        entries = runner.ledger_record()["entries"]
        self.assertEqual(len(entries), len(log))
        for entry, item in zip(entries, log, strict=True):
            self.assertEqual(entry["native_start"]["hex"], float(item["start"]).hex())
            self.assertEqual(entry["native_end"]["hex"], float(item["end"]).hex())

    def snapshot(self, runner):
        bed = runner._bed
        return {"state": bed.state, "time": bed.time_s.hex(), "identity": bed.identity_sha256(),
                "accepted": runner.accepted, "ledger": runner.ledger_record(), "pool": runner.mobile_state,
                "soil": runner.soil_water_m, "routing": runner._routing, "receipts": runner.receipts}

    def assertUnchanged(self, runner, before) -> None:
        after = self.snapshot(runner)
        for key in ("state", "pool", "soil", "routing"):
            self.assertIs(after[key], before[key], key)
        for key in ("time", "identity", "accepted", "ledger", "receipts"):
            self.assertEqual(after[key], before[key], key)
        with self.assertRaises(DenseRunnerError) as ctx:
            runner.step()
        self.assertEqual(ctx.exception.code, "runner_halted")

    def assertStopped(self, receipt, status, code) -> None:
        self.assertEqual((receipt["status"], receipt["code"]), (status, code), receipt["detail"])


class AlternationParity(_RunnerCase):
    """(1) + (4): nonzero six-class wind -> rain/recession -> wind against the direct native control."""

    def test_reference_alternation_matches_direct_native_control(self) -> None:
        bed = fx.build_bed()
        runner = fx.make_runner(bed, fx.reference_schedule())
        initial_identity = bed.identity_sha256()
        fractions = []
        for _ in range(3):
            receipt = runner.step()
            self.assertEqual(receipt["status"], "accepted", receipt.get("detail"))
            fractions.append(available_fraction(bed.state))
        report = runner.report()
        self.assertEqual(report["status"], "complete")
        self.assertNotEqual(bed.identity_sha256(), initial_identity)

        control_bed = fx.build_bed()
        events = [wind_event(), rain_event(), wind_event()]
        state, t_end, log = direct_control(control_bed, fx.wind_config(control_bed), events)
        self.assertSameFields(bed.state, state)
        self.assertClocks(runner, log)
        self.assertEqual(bed.time_s.hex(), float(t_end).hex())

        wind1, rain, wind2 = runner.receipts
        for receipt, item in ((wind1, log[0]), (wind2, log[2])):
            self.assertGreater(sum(receipt["classes"]["detachment_kg"]), 0.0)
            self.assertGreater(sum(receipt["classes"]["boundary_export_kg"]), 0.0)
            self.assertEqual(receipt["classes"]["boundary_export_kg"], item["export"])
            self.assertEqual(receipt["classes"]["detachment_kg"], item["detachment"])
            self.assertEqual(receipt["wind_end_pool"], "genuinely_empty")
            # Declared block metadata kept beside the actual physical clock.
            self.assertEqual(receipt["native"]["physical_time_s"]["hex"], float(item["end"]).hex())
        self.assertGreater(sum(rain["classes"]["actual_pickup_kg"]), 0.0)
        self.assertGreater(sum(rain["classes"]["export_kg"]), 0.0)
        self.assertEqual(rain["classes"]["export_kg"], log[1]["export"])
        self.assertEqual(rain["classes"]["actual_pickup_kg"], log[1]["pickup"])
        self.assertEqual(rain["water"]["rain_m3"], float(log[1]["water"]["rain_m3"]))
        self.assertGreater(rain["water"]["rain_m3"], 0.0)
        self.assertEqual(len(rain["classes"]["export_kg"]), 6)
        self.assertTrue(rain["routing"]["first_graph_construction"])
        self.assertEqual(rain["native"]["phase"], None)
        self.assertFalse(bool(np.any(runner.soil_water_m)))  # carried native dry soil (zero after the reset)
        # Availability survives evolution by native rules: never reset to 100% or to the initial 25%.
        for fraction in fractions:
            self.assertFalse(bool(np.all(fraction == 1.0)))
            self.assertTrue(bool(np.any(fraction != fx.AVAILABLE)))
        # Aggregate per-class closure published only within accumulated native bounds.
        aggregate = runner.receipts[-1]["aggregate"]
        self.assertTrue(all(abs(r) <= b for r, b in zip(aggregate["joint_residual_kg"], aggregate["joint_bound_kg"],
                                                        strict=True)))

    def test_initial_35_60_availability_evolves_natively(self) -> None:
        available = np.full((fx.NY, fx.NX, 6), 0.6)
        available[..., 0] = 0.35
        bed = fx.build_bed(available=available.copy())
        runner = fx.make_runner(bed, fx.schedule(fx.wind()))
        self.assertEqual(runner.step()["status"], "accepted")
        control_bed = fx.build_bed(available=available.copy())
        state, _, log = direct_control(control_bed, fx.wind_config(control_bed), [wind_event()])
        self.assertSameFields(bed.state, state)
        self.assertClocks(runner, log)
        initial = np.broadcast_to(available, bed.state.active_layer.mass_kg.shape)[bed.state.active_layer.mass_kg > 0]
        fraction = available_fraction(bed.state)
        self.assertFalse(bool(np.all(fraction == 1.0)))
        self.assertTrue(bool(np.any(fraction != initial)))


class RainReentry(_RunnerCase):
    """(2): second rain on the wind-modified bed; held routing below the native trigger, reroute above it."""

    def run_case(self, commit_spec):
        events = [wind_event(), rain_event(), wind_event(), rain_event()]
        bed = fx.build_bed(commit_spec=commit_spec)
        runner = fx.make_runner(bed, schedule_of(events))
        self.assertEqual(runner.run()["status"], "complete", runner.stop_receipt)
        control_bed = fx.build_bed(commit_spec=commit_spec)
        state, _, log = direct_control(control_bed, fx.wind_config(control_bed), events, first_graph="terrain")
        self.assertSameFields(bed.state, state)
        self.assertClocks(runner, log)
        second = runner.receipts[3]
        self.assertFalse(second["routing"]["first_graph_construction"])
        self.assertEqual(second["routing"]["entry_rerouted"], log[3]["rerouted"])
        self.assertEqual(second["routing"]["graph_retained_by_identity"], log[3]["graph_retained"])
        self.assertEqual(second["classes"]["export_kg"], log[3]["export"])
        # Totals / counters prepared before adoption accumulate exactly over the accepted rain receipts.
        aggregate = runner.report()["aggregate"]
        rains = [receipt for receipt in runner.receipts if receipt["kind"] == "rain"]
        self.assertEqual(aggregate["water"]["rain_m3"], rains[0]["water"]["rain_m3"] + rains[1]["water"]["rain_m3"])
        self.assertEqual(aggregate["n_entry_reroutes"], sum(int(r["routing"]["entry_rerouted"]) for r in rains))
        self.assertEqual(aggregate["n_event_reroutes"], sum(r["routing"]["n_reroutes_in_event"] for r in rains))
        return second

    def test_below_trigger_retains_graph_and_source_by_identity(self) -> None:
        second = self.run_case(None)
        self.assertFalse(second["routing"]["entry_rerouted"])
        self.assertTrue(second["routing"]["graph_retained_by_identity"])

    def test_above_trigger_reroutes_natively(self) -> None:
        spec = dataclasses.replace(TopographicCommitSpec(), pending_elevation_trigger_m=1e-9)
        second = self.run_case(spec)
        self.assertTrue(second["routing"]["entry_rerouted"])
        self.assertFalse(second["routing"]["graph_retained_by_identity"])


class PiecewiseRain(_RunnerCase):
    """(3): a positive two-interval rain equals the direct native RainfallSchedule control."""

    def test_two_interval_rain(self) -> None:
        intervals = ((30.0, 72.0), (30.0, 36.0))
        events = [wind_event(), rain_event(intervals)]
        bed = fx.build_bed()
        runner = fx.make_runner(bed, schedule_of(events))
        self.assertEqual(runner.run()["status"], "complete", runner.stop_receipt)
        control_bed = fx.build_bed()
        state, _, log = direct_control(control_bed, fx.wind_config(control_bed), events)
        self.assertSameFields(bed.state, state)
        self.assertClocks(runner, log)
        rain = runner.receipts[1]
        self.assertEqual(rain["native"]["intensity_mm_h"], [72.0, 36.0])
        self.assertEqual(len(rain["native"]["rainfall_edges_s"]), 3)
        self.assertEqual(rain["water"]["rain_m3"], float(log[1]["water"]["rain_m3"]))
        self.assertGreater(rain["water"]["rain_m3"], 0.0)
        self.assertEqual(rain["classes"]["export_kg"], log[1]["export"])


class WindEndLoad(_RunnerCase):
    """Genuine nonempty fine-cadence winds finalize before genuine water recession."""

    def test_nonempty_wind_water_wind_water_matches_direct_native(self) -> None:
        events = [wind_event(), rain_event(), wind_event(), rain_event()]
        bed = fx.build_bed()
        runner = fx.make_runner(bed, schedule_of(events), config=fx.wind_config(bed, outer_timestep_s=0.025))
        report = runner.run()
        self.assertEqual(report["status"], "complete", report)
        control_bed = fx.build_bed()
        state, t_end, log = direct_control(control_bed, fx.wind_config(control_bed, outer_timestep_s=0.025), events)
        self.assertEqual(field_digests(bed.state), field_digests(state))
        self.assertEqual(bed.time_s.hex(), float(t_end).hex())
        deposited = np.zeros(6)
        for i, receipt in enumerate(runner.receipts):
            self.assertTrue(all(abs(r) <= b for r, b in zip(receipt["aggregate"]["joint_residual_kg"],
                                                          receipt["aggregate"]["joint_bound_kg"], strict=True)))
            if receipt["kind"] == "wind":
                final = receipt["finalization"]
                self.assertGreater(sum(final["airborne_mass_before_by_class_kg"]), 0.0)
                self.assertGreater(sum(final["deposited_mass_by_class_kg"])
                                   + sum(final["boundary_export_mass_by_class_kg"]), 0.0)
                self.assertEqual(receipt["classes"]["boundary_export_kg"], log[i]["export"])
                self.assertEqual(final["deposited_mass_by_class_kg"], log[i]["final_deposition"])
                self.assertEqual(final["boundary_export_mass_by_class_kg"], log[i]["final_export"])
                self.assertEqual(final["physical_time_s"], receipt["native"]["physical_time_s"])
                self.assertEqual(final["completed_block_index"], 1)
                self.assertEqual(final["completed_block_offset_s"], 0.0)
                self.assertFalse(any(receipt["classes"]["mobile_after_kg"]))
                deposited += np.array(final["deposited_mass_by_class_kg"])
            else:
                self.assertEqual(receipt["classes"]["export_kg"], log[i]["export"])
                deposited += np.array(receipt["classes"]["internal_terminal_deposition_kg"])
        self.assertTrue(mobile_state_is_genuinely_empty(runner.mobile_state))
        self.assertGreater(bed.time_s, 2 * (fx.WIND_S + fx.RAIN_S))
        np.testing.assert_array_equal(report["aggregate"]["cumulative_internal_terminal_deposition_kg"], deposited)
        self.assertGreater(decode_time(runner.receipts[2]["finalization"]["record_origin_s"]), 0.0)

    def test_finalizer_exception_or_incomplete_preserves_last_boundary(self) -> None:
        for mode in ("exception", "incomplete"):
            with self.subTest(mode=mode):
                bed = fx.build_bed()
                runner = fx.make_runner(bed, fx.schedule(fx.wind(), fx.rain()),
                                        config=fx.wind_config(bed, outer_timestep_s=0.025))
                before = self.snapshot(runner)
                actual = dense_runner._finalize_wind_event
                def fail(*args, **kwargs):
                    if mode == "exception":
                        raise RuntimeError("illustrative finalizer failure")
                    result = actual(*args, **kwargs)
                    return dataclasses.replace(result, completed=False, reasons=("illustrative incomplete",))
                with mock.patch.object(dense_runner, "_finalize_wind_event", side_effect=fail), \
                     mock.patch.object(dense_runner, "_complete_event", wraps=dense_runner._complete_event) as rain:
                    receipt = runner.step()
                    self.assertStopped(receipt, "native_error" if mode == "exception" else "incomplete",
                                       "wind_finalization_failure" if mode == "exception" else "wind_finalization_incomplete")
                    self.assertUnchanged(runner, before)
                    rain.assert_not_called()
                self.assertFalse(mobile_state_is_genuinely_empty(runner.active_evidence["transport"].state.physical.mobile_state))
                self.assertEqual(runner.ledger_record()["entries"], [])


    def test_failed_second_wind_finalizer_keeps_accepted_water_boundary(self) -> None:
        bed = fx.build_bed()
        runner = fx.make_runner(bed, fx.schedule(fx.wind(), fx.rain(), fx.wind(), fx.rain()),
                                config=fx.wind_config(bed, outer_timestep_s=0.025))
        self.assertEqual(runner.step()["status"], "accepted")
        self.assertEqual(runner.step()["status"], "accepted")
        before = self.snapshot(runner)
        def fail(state, *args, **kwargs):
            # Simulate a native fault after touching staged bed and pool arrays.
            state.physical.active_layer.mass_kg[0, 0, 0] += 1.0
            state.physical.mobile_state.mass_kg[0] += 1.0
            raise RuntimeError("illustrative staged finalizer mutation")
        with mock.patch.object(dense_runner, "_finalize_wind_event", side_effect=fail), \
             mock.patch.object(dense_runner, "_complete_event", wraps=dense_runner._complete_event) as rain:
            receipt = runner.step()
            self.assertStopped(receipt, "native_error", "wind_finalization_failure")
            self.assertUnchanged(runner, before)
            rain.assert_not_called()
        self.assertEqual(len(runner.ledger_record()["entries"]), 2)


class WindDurationCoverage(_RunnerCase):
    """C-R3: the native step plan must cover the requested wind duration; FP accumulation is kept, not rewritten."""

    def test_dropped_native_remainder_is_refused_before_native_transport(self) -> None:
        cases = [("sub_epsilon", fx.wind(duration_s=1e-12, shear=0.0)),
                 ("dropped_tail", fx.wind(duration_s=2.0000000001))]
        for name, event in cases:
            with self.subTest(case=name):
                bed = fx.build_bed()
                runner = fx.make_runner(bed, fx.schedule(event))
                before = self.snapshot(runner)
                with mock.patch.object(dense_runner, "_advance_wind_block") as wind:
                    receipt = runner.step()
                    wind.assert_not_called()
                self.assertStopped(receipt, "refused", "unsupported_wind_duration")
                self.assertUnchanged(runner, before)
                self.assertEqual(runner.ledger_record()["entries"], [])

    def test_ordinary_fractional_native_steps_match_direct_control(self) -> None:
        # (duration, outer): 0.3 s at 0.1 s has a nonzero FP plan discrepancy; 2.5 s at 1 s an exact 0.5 s tail.
        for duration, outer in ((0.3, 0.1), (2.5, 1.0)):
            with self.subTest(duration=duration, outer=outer):
                bed = fx.build_bed()
                config = fx.wind_config(bed, outer_timestep_s=outer)
                runner = fx.make_runner(bed, fx.schedule(fx.wind(duration_s=duration, shear=0.0)), config=config)
                with mock.patch.object(dense_runner, "_advance_wind_block",
                                       wraps=dense_runner._advance_wind_block) as wind:
                    receipt = runner.step()
                    wind.assert_called_once()
                self.assertEqual(receipt["status"], "accepted", receipt.get("detail"))
                control_bed = fx.build_bed()
                state, t_end, log = direct_control(control_bed, fx.wind_config(control_bed, outer_timestep_s=outer),
                                                   [("wind", duration, 0.0, fx.DIRECTION)])
                self.assertSameFields(bed.state, state)
                self.assertClocks(runner, log)
                self.assertEqual(bed.time_s.hex(), float(t_end).hex())  # native endpoint, never rewritten
                coverage = receipt["coverage"]
                self.assertEqual(coverage["requested_s"]["hex"], float(duration).hex())
                if outer == 0.1:
                    self.assertNotEqual(coverage["plan_discrepancy_s"], "0/1")  # recorded, within FP rounding
                else:
                    self.assertEqual(coverage["plan_discrepancy_s"], "0/1")
                    self.assertEqual(coverage["native_plan_steps"], 3)


class GuardsAndRefusals(_RunnerCase):
    """(6): native guards, typed refusals, staged-identity / static mutation, receipt cap, injected failures."""

    def test_too_short_recession_is_native_incomplete(self) -> None:
        bed = fx.build_bed()
        runner = fx.make_runner(bed, fx.schedule(fx.wind(), fx.rain(guard_s=5.0)))
        self.assertEqual(runner.step()["status"], "accepted")
        before = self.snapshot(runner)
        receipt = runner.step()
        self.assertStopped(receipt, "incomplete", "rain_incomplete_native_limit")
        self.assertIsNone(receipt["unaccepted_elapsed_s"])
        self.assertUnchanged(runner, before)
        self.assertIsNone(runner._routing)

    def test_native_storm_step_guard(self) -> None:
        bed = fx.build_bed()
        control = fx.runner_control(sediment_control=fx.sediment_control(max_steps=125))
        runner = fx.make_runner(bed, fx.schedule(fx.wind(), fx.rain()), control=control)
        self.assertEqual(runner.step()["status"], "accepted")
        before = self.snapshot(runner)
        receipt = runner.step()
        self.assertStopped(receipt, "incomplete", "rain_incomplete_native_limit")
        self.assertIn("max_steps", receipt["detail"])
        self.assertUnchanged(runner, before)

    def test_typed_construction_refusals_mutate_nothing(self) -> None:
        bed = fx.build_bed()
        identity, state = bed.identity_sha256(), bed.state
        base = fx.wind_config(bed)
        reversed_classes = GrainClassSet(classes=tuple(reversed(bed.context.grain_classes.classes)))
        sc = fx.sediment_control()
        cases = [
            ("incompatible_wind_config", dict(config=dataclasses.replace(
                base, geometry=dataclasses.replace(bed.context.geometry)))),
            ("incompatible_wind_config", dict(config=dataclasses.replace(base, grain_classes=reversed_classes))),
            ("incompatible_wind_config", dict(config=dataclasses.replace(
                base, initial_available_fraction=np.full((fx.NY, fx.NX, 6), 0.5)))),
            ("incompatible_wind_config", dict(config=dataclasses.replace(
                base, avalanching_spec=AvalanchingSpec(enabled=True)))),
            ("unsupported_transport_scheme", dict(control=fx.runner_control(
                sediment_control=dataclasses.replace(sc, transport_scheme="characteristic")))),
            ("unsupported_hydraulic_implementation", dict(control=fx.runner_control(
                sediment_control=dataclasses.replace(sc, storm=StormControl(max_dt_s=1.0, implementation="numba"))))),
            ("vegetation_mismatch", dict(water=fx.water_inputs(
                bed, vegetation_cover_fraction=np.full((fx.NY, fx.NX), 0.5)))),
            ("storm_step_limit", dict(limit=fx.limits(max_storm_steps=100))),
            ("event_limit", dict(limit=fx.limits(max_events=2))),
        ]
        for code, kwargs in cases:
            with self.subTest(code=code):
                with self.assertRaises(DenseRunnerError) as ctx:
                    fx.make_runner(bed, fx.reference_schedule(), **kwargs)
                self.assertEqual(ctx.exception.code, code, ctx.exception.detail)
                self.assertIs(bed.state, state)
                self.assertEqual(bed.identity_sha256(), identity)

    def test_preflight_limits_stop_before_any_native_call(self) -> None:
        cases = [("work_bytes_cap", dict(max_work_bytes=1), 0),
                 ("wind_step_limit", dict(max_wind_accounting_steps=2), 0),
                 ("report_rows_limit", dict(max_report_rows=5), 1)]
        for code, limit, accepted in cases:
            with self.subTest(code=code):
                bed = fx.build_bed()
                runner = fx.make_runner(bed, fx.schedule(fx.wind(), fx.rain()), limit=fx.limits(**limit))
                for _ in range(accepted):
                    self.assertEqual(runner.step()["status"], "accepted")
                before = self.snapshot(runner)
                with mock.patch.object(dense_runner, "_advance_wind_block") as wind, \
                        mock.patch.object(dense_runner, "_complete_event") as rain:
                    receipt = runner.step()
                    wind.assert_not_called()
                    rain.assert_not_called()
                self.assertStopped(receipt, "refused", code)
                self.assertUnchanged(runner, before)

    def test_receipt_cap_refuses_before_adoption(self) -> None:
        bed = fx.build_bed()
        runner = fx.make_runner(bed, fx.reference_schedule(), limit=fx.limits(max_receipt_bytes=256))
        before = self.snapshot(runner)
        self.assertStopped(runner.step(), "refused", "receipt_cap_exceeded")
        self.assertUnchanged(runner, before)

    def test_staged_identity_and_static_mutation_refuse_publication(self) -> None:
        real = dense_runner._advance_wind_block
        for code in ("accepted_state_changed", "case_input_mutated", "static_input_mutated"):
            with self.subTest(code=code):
                bed = fx.build_bed()
                water = fx.water_inputs(bed)
                runner = fx.make_runner(bed, fx.reference_schedule(), water=water)
                before = self.snapshot(runner)
                depth = bed.state.water.depth_m
                flux = bed.context.geometry.boundary_x.inflow_flux_kg_m_s
                cover = water.vegetation_cover_fraction

                def mutate_during_native(state, block, config):
                    result = real(state, block, config)
                    if code == "accepted_state_changed":
                        depth[0, 0] = 0.001
                    elif code == "case_input_mutated":
                        flux["phi_1"] = 0.1
                    else:
                        cover[0, 0] = 0.5
                    return result

                try:
                    with mock.patch.object(dense_runner, "_advance_wind_block", side_effect=mutate_during_native):
                        receipt = runner.step()
                finally:
                    depth[0, 0], flux["phi_1"], cover[0, 0] = 0.0, 0.0, 0.0
                self.assertStopped(receipt, "refused", code)
                self.assertUnchanged(runner, before)

    def test_injected_exception_after_mutating_work_input_leaves_accepted_state(self) -> None:
        bed = fx.build_bed()
        runner = fx.make_runner(bed, fx.reference_schedule())
        before = self.snapshot(runner)

        def failing_wind(state, block, config):
            state.physical.voxel_column.mass_kg[...] += 1.0  # the staged WORK copy only
            raise RuntimeError("injected wind failure")

        with mock.patch.object(dense_runner, "_advance_wind_block", side_effect=failing_wind):
            receipt = runner.step()
        self.assertStopped(receipt, "native_error", "wind_native_failure")
        self.assertEqual(receipt["exception_type"], "RuntimeError")
        self.assertIsNone(receipt["unaccepted_elapsed_s"])
        self.assertIsNone(runner.active_evidence)
        self.assertUnchanged(runner, before)

        bed = fx.build_bed()
        runner = fx.make_runner(bed, fx.reference_schedule())
        self.assertEqual(runner.step()["status"], "accepted")
        before = self.snapshot(runner)

        def failing_rain(state, *args, **kwargs):
            # The entry's storm arrays are native copies made for this attempt (WORK input only).
            state.storm.soil_water_m[...] += 1.0
            state.storm.depth_m[...] += 1.0
            raise RuntimeError("injected rain failure")

        with mock.patch.object(dense_runner, "_complete_event", side_effect=failing_rain):
            receipt = runner.step()
        self.assertStopped(receipt, "native_error", "rain_native_failure")
        self.assertIsNone(receipt["unaccepted_elapsed_s"])
        self.assertUnchanged(runner, before)
        self.assertFalse(bool(np.any(runner.soil_water_m)))

    def test_failed_second_rain_leaves_retained_velocity_memory_intact(self) -> None:
        """C-R1: the second rain's entry memory is a work copy; a native write to it never reaches the template."""
        bed = fx.build_bed()
        runner = fx.make_runner(bed, schedule_of([wind_event(), rain_event(), wind_event(), rain_event()]))
        for _ in range(3):
            self.assertEqual(runner.step()["status"], "accepted")
        before = self.snapshot(runner)
        velocity = runner._routing.template.sediment_velocity_m_s
        values = np.array(velocity, copy=True)
        seen = []

        def failing_second_rain(entry, *args, **kwargs):
            seen.append(entry.sediment_velocity_m_s)
            entry.sediment_velocity_m_s[...] = 1.0
            raise RuntimeError("injected failure after writing the WORK velocity")

        with mock.patch.object(dense_runner, "_complete_event", side_effect=failing_second_rain):
            receipt = runner.step()
        self.assertStopped(receipt, "native_error", "rain_native_failure")
        self.assertEqual(len(seen), 1)
        self.assertIsNot(seen[0], velocity)
        self.assertTrue(bool(np.all(seen[0] == 1.0)))  # the work copy really was written
        self.assertIs(runner._routing.template.sediment_velocity_m_s, velocity)
        self.assertTrue(bool(np.array_equal(velocity, values)))  # contents, not only identity
        self.assertFalse(bool(np.any(velocity)))  # the native dry reset value is retained
        self.assertUnchanged(runner, before)

    def test_changed_static_inputs_refused_before_any_native_call(self) -> None:
        """C-R2: a controlling dict / array changed after construction never reaches a native delegate."""
        for case in ("hop_distance_dict", "vegetation_array"):
            with self.subTest(case=case):
                bed = fx.build_bed()
                water = fx.water_inputs(bed)
                runner = fx.make_runner(bed, fx.schedule(fx.wind(), fx.rain()), water=water)
                if case == "vegetation_array":
                    self.assertEqual(runner.step()["status"], "accepted")
                before = self.snapshot(runner)
                hops = runner._wind.config.particle_transport.hop_distance_by_class
                key = next(iter(hops))
                old = hops[key]
                changed = dataclasses.replace(old, median_hop_distance_m=old.median_hop_distance_m * 1.1)
                cover = water.vegetation_cover_fraction
                try:
                    if case == "hop_distance_dict":
                        hops[key] = changed
                    else:
                        cover[0, 0] = 0.5
                    with mock.patch.object(dense_runner, "_advance_wind_block") as wind, \
                            mock.patch.object(dense_runner, "_complete_event") as rain:
                        receipt = runner.step()
                        wind.assert_not_called()
                        rain.assert_not_called()
                    # The caller's input is reported, never repaired.
                    if case == "hop_distance_dict":
                        self.assertIs(hops[key], changed)
                    else:
                        self.assertEqual(float(cover[0, 0]), 0.5)
                finally:
                    hops[key] = old
                    cover[0, 0] = 0.0
                self.assertStopped(receipt, "refused", "static_input_mutated_before_event")
                self.assertIn("OUTSIDE", receipt["guarantee"])
                self.assertUnchanged(runner, before)

    def test_external_change_of_accepted_bed_is_reported_not_hidden(self) -> None:
        bed = fx.build_bed()
        runner = fx.make_runner(bed, fx.reference_schedule())
        before = self.snapshot(runner)
        depth = bed.state.water.depth_m
        try:
            depth[0, 0] = 0.001
            with mock.patch.object(dense_runner, "_advance_wind_block") as wind:
                receipt = runner.step()
                wind.assert_not_called()
            self.assertIs(bed.state, before["state"])  # the only accepted state is neither replaced nor repaired
            self.assertEqual(float(depth[0, 0]), 0.001)
            self.assertNotEqual(receipt["accepted"]["bed_identity_sha256"], before["identity"])
        finally:
            depth[0, 0] = 0.0
        self.assertStopped(receipt, "refused", "accepted_state_externally_mutated")
        self.assertIn("OUTSIDE", receipt["guarantee"])
        self.assertUnchanged(runner, before)


if __name__ == "__main__":
    unittest.main()
