"""Actual current-triplet CPU qualification on explicitly SYNTHETIC inputs.

No field calibration or source guard is mocked. Configured native execution is
a failure if unusable; the pure resource/clock tests can run without upstreams.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from masy.daily_triplet import DailyTripletSession, TripletError, TripletLimits


def limits():
    return TripletLimits(100.0, 12, 8, 50000, 1 << 22, 1 << 16, 1 << 27, 1 << 27, 1 << 28,
                         1 << 27, 1 << 24, 1 << 20, 1 << 20)


def native_fixture(root):
    import numpy as np
    from leaf import growth, plant_session as ps, state, traits
    from leaf import floating_lifecycle_bridge as fb, water_vegetation as lw
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple_syrup import native_case_profile as ncp, soil_evaporation as se, weather_pet as wp
    from maple_syrup import sediment_physics as sp, water_vegetation_binding as vb
    from maple_syrup.pond_event import PondEvent, PondEventControl, PondCompletionPolicy
    from maple_syrup.experimental_storm import ExperimentalControl
    from maple_syrup.rainfall import constant_rainfall
    from masy import daily_triplet_wind as tw
    from masy.daily_csv import load_daily_scenario
    from masy.daily_schedule import DailyPlan
    from masy.daily_wind_handoff import WindBridgeLimits
    from wind_handoff_support import make_plan, make_native, handoff_limits, wind_limits

    root = Path(root)
    make_plan(root / "plan", precipitation=0.01, threshold=0.005)
    scenario = root / "plan/scenario.csv"
    raw = scenario.read_text().replace("end_date: 2030-06-02", "end_date: 2030-06-08")
    raw = raw.replace("transpiration_enabled: false", "transpiration_enabled: true")
    raw = raw.replace("growth_enabled: false", "growth_enabled: true")
    scenario.write_text(raw)
    weather = root / "plan/weather.csv"
    weather.write_text(weather.read_text() + "".join(f"2030-06-{day:02d},25,2.592,0,0\n" for day in range(3, 9)))
    plan = DailyPlan(load_daily_scenario(scenario))
    fixture = make_native(root / "native", profile_name=ncp.CURRENT_PROFILE_NAME,
        source_digest=tw.SYRUP_PACKAGE_SHA256, vegetation_wake=True, fixture_nx=40)
    g = fixture.context.geometry
    leaf_root = Path(os.environ["MASY_TRIPLET_LEAF"])
    doc = json.loads((leaf_root / "examples/phase1/species.json").read_text())
    doc["schema_version"] = 2
    doc["root_layers"] = 3
    for record in doc["species"].values():
        record["roots"]["layer_fractions"] = [1 / 3] * 3
        record["hydrology"] = {traits.RECEIVER_WEIGHTS_KEY: [0.0, 1.0, 1.0]}
    # Labelled one-site engineering fixture, using existing LEAF laws.
    grass = doc["species"]["grass"]
    grass["growth"]["traits"].update(mature_biomass_kg=1.0, intrinsic_rate_per_year=0.5,
        biomass_support_kg_year_per_m3=10.0, live_fraction=0.5, extra_growth_rate_per_year=0.2,
        monthly_activity=[1.0] * 12, temperature_min_c=0.0, temperature_opt_c=20.0, temperature_max_c=40.0)
    grass["water_use"]["traits"]["transpiration_coefficient"] = 0.5
    catalog = traits.SpeciesCatalog(doc)
    domain = state.Domain(g.nx, g.ny, g.dx_m, dy_m=g.dy_m, periodic=(True, True))
    surface = committed_elevation_m(fixture.bed.committed_topography)
    population = state.PlantState.from_records(catalog, domain, [dict(plant_id=11, species="grass",
        x_m=2.5 * g.dx_m, y_m=1.5 * g.dy_m, establishment_ground_m=float(surface[1, 2]),
        agb_dry_kg=0.1, seasonal_agb_dry_kg=0.0, age_days=100.0, status="live")])
    config = ps.SessionConfig(catalog, growth.compile_species(catalog, "numpy", 1 << 28), "numpy", 1 << 16,
        1 << 28, 7, reserve_bytes=1 << 24, builder="numpy", canopy={"max_edges": 1 << 16, "max_bytes": 1 << 27, "builder": "numpy"})
    source = fb.bind_source(fixture.source, expected_syrup_digest=tw.SYRUP_PACKAGE_SHA256,
        expected_syrup_commit=os.environ["MASY_TRIPLET_SYRUP_COMMIT"], expected_maple_commit=tw.MAPLE_COMMIT,
        max_bytes=1 << 27)
    leaf_package = lw.package_source_record()["digest"].removeprefix("sha256:")
    helper = leaf_root / "benchmarks/spatial_mapping.py"
    helpers = [{"module": "benchmarks.spatial_mapping", "sha256": hashlib.sha256(helper.read_bytes()).hexdigest(), "bytes": helper.stat().st_size}]
    wind = tw.prepare_triplet_wind(plan, fixture.source, fixture.admission, fixture.context,
        wind_limits=wind_limits(), handoff_limits=handoff_limits(), bridge_limits=WindBridgeLimits(1 << 20, 1 << 20),
        available_bytes_fn=fixture.available, run_inputs_sha256=hashlib.sha256(b"labelled-synthetic-triplet-fixture-v1").hexdigest())
    pick = lambda a: np.asarray(a)[[0, 2, 5]]
    sediment = sp.sediment_physics_parameters(diameter_m=np.array([c.diameter_m for c in fixture.context.grain_classes.classes]),
        raindrop_a=pick(sp.PLOT1_RAINDROP_A), raindrop_b=pick(sp.PLOT1_RAINDROP_B), raindrop_c=pick(sp.PLOT1_RAINDROP_C),
        raindrop_max_depth_mm=pick(sp.PLOT1_RAINDROP_MAX_MM),
        raindrop_depth_attenuation_per_cm=pick(sp.LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM),
        particle_density_g_cm3=sp.PLOT1_PARTICLE_DENSITY_G_CM3, active_layer_sensitivity_mm=sp.PLOT1_ACTIVE_LAYER_SENSITIVITY_MM,
        ke_model=sp.PLOT1_KE_MODEL, xp=np)
    control = PondEventControl(backend="numpy", time=ExperimentalControl(max_dt_s=0.2, min_dt_s=1 / 1024, max_retries=12, max_steps=50000))
    completion = PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-4, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True)

    def builder(vertical, topology, cover, schedule):
        return PondEvent(fixture.context, None, sediment, constant_rainfall(0, 1, 0) if schedule is None else schedule,
            topology, control=control, policy=completion, floating_vertical=vertical, vegetation_cover_fraction=cover)

    site = vb.site_controls(a1=0.1, a2=1.0, a3=1.0, max_fraction_per_hour=0.5, max_intake_rate_m_s=1e-5,
        background_receiver_weights=[0.0, 1.0, 1.0], n_layers=3, n_cells=g.nx * g.ny, max_bytes=1 << 20)
    terrain = wp.terrain_factor(np.zeros(g.ny * g.nx, dtype=np.int32), np.zeros(g.ny * g.nx), 35.0)
    session = DailyTripletSession(plan, wind=wind, owner=fixture.owner, bed=fixture.bed, initial_event_factory=fixture.factory,
        plant_state=population, leaf_config=config, anchor=dt.datetime(2030, 6, 1, tzinfo=dt.timezone.utc), leaf_source=source,
        terrain=terrain, partition=se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75), site_controls=site,
        active_mask=np.ones((g.ny, g.nx), dtype=bool), event_builder=builder,
        day_control=ExperimentalControl(max_dt_s=3600.0, min_dt_s=1 / 1024, max_retries=12, max_steps=50000),
        wind_vegetation_policy=tw.WindVegetationPolicy(("live", "dead_standing"), "exclude", 1.0), limits=limits(),
        expected_leaf_package_sha256=leaf_package, expected_external_helpers=helpers, available_bytes_fn=fixture.available)
    return session, fixture


class Policy(unittest.TestCase):
    def test_nonfinite_or_boolean_resource_controls_refuse(self):
        for change in ({"event_chunk_s": float("nan")}, {"max_event_calls": True}, {"reserve_bytes": -1}):
            with self.assertRaises(TripletError):
                dataclasses.replace(limits(), **change)

    def test_host_reader_prices_leaf_transients_not_only_retained_arrays(self):
        from types import SimpleNamespace
        session = object.__new__(DailyTripletSession)
        session._limits = SimpleNamespace(reserve_bytes=10)
        session._native_bytes = 100
        session.plants = SimpleNamespace(config=SimpleNamespace(reserve_bytes=400, max_bytes=1000),
            resident_bytes=lambda: 20, candidate_bytes=lambda: 10)
        session._available = lambda: 709
        with self.assertRaisesRegex(TripletError, "declared stage ceiling"):
            session._leaf_admission(0)
        session._available = lambda: 710
        session._leaf_admission(0)


@unittest.skipUnless(os.environ.get("MASY_TRIPLET_NATIVE") == "1", "explicit current native runtime required")
class NativeTriplet(unittest.TestCase):
    def test_actual_wind_storm_T_growth_and_next_day_continuity(self):
        from maple_syrup import vegetation_lifecycle as vl
        import numpy as np

        with tempfile.TemporaryDirectory() as root:
            session, fixture = native_fixture(root)
            initial_biomass = session.plants.snapshot_state().agb_dry_kg.copy()
            initial_recipe = vl.current_record(session.lineage.head).recipe_id
            records = [session.advance().record() for _ in range(8)]
            self.assertEqual(session.accepted_days, 8)
            self.assertEqual(session.wind.next_wind_index, 2)
            self.assertEqual(session.lineage.head.cursor.ordinal, 8)
            self.assertEqual(session.plants.cursor.date(), dt.date(2030, 6, 9))
            self.assertEqual(session.plants.counters()["growth_updates"], 1)
            self.assertGreater(float(session.plants.snapshot_state().agb_dry_kg.sum()), float(initial_biomass.sum()))
            ledger = dict(session.lineage.head.ledger)
            self.assertGreater(ledger["rain_m3"], 0)
            self.assertGreater(ledger["transpiration_m3"], 0)
            self.assertGreater(ledger["evaporation_m3"], 0)
            self.assertLessEqual(abs(ledger["residual_m3"]), ledger["residual_bound_m3"])
            self.assertNotEqual(vl.current_record(session.lineage.head).recipe_id, initial_recipe)
            self.assertTrue(all(r["profile"] == "masy.daily_triplet.dense_cpu.v1" for r in records))
            self.assertTrue(records[0]["full_storm_completed"])
            self.assertTrue(all(not r["full_storm_completed"] for r in records[1:]))
            self.assertTrue(all(a["physical_end_s_hex"] == b["physical_start_s_hex"]
                for a, b in zip(records, records[1:])))
            self.assertGreater(session.lineage.head.t_s, 8 * 86400 + 4)
            np.testing.assert_array_equal(session.lineage.head.bed.water.mobile_mass_by_cell_class_kg, 0)
            with self.assertRaises(StopIteration):
                session.advance()
            with self.assertRaisesRegex(TripletError, "checkpoint_unqualified"):
                session.save_checkpoint("unused")


if __name__ == "__main__":
    unittest.main()
