"""Authored tiny native integration fixtures; no field calibration.

Water parameters are explicit fixture inputs from upstream test tables. Product
code does not choose a texture mapping, a completion policy or a sediment law.
"""
from pathlib import Path
import dataclasses
import os
import shutil


def make_plan(root, *, same_day=False, threshold=5.0, precipitation=0.0):
    from masy.daily_csv import load_daily_scenario
    from masy.daily_schedule import DailyPlan
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "wind.yaml").write_text("wind_forcing:\n  blocks:\n    - shear_velocity_m_s: 0.6\n      direction_deg: 90.0\n      duration_s: 2.0\n")
    (root / "weather.csv").write_text(
        "# format: masy.daily_weather.v1\n# pet_method: explicit\n"
        f"# precipitation_threshold_mm: {threshold}\n"
        "date,tmean_c,pet_mm_day,precipitation_mm,precipitation_intensity_mm_h\n"
        f"2030-06-01,25,2.592,{precipitation},{100.0 if precipitation else 0.0}\n"
        "2030-06-02,25,2.592,0,0\n")
    (root / "scenario.csv").write_text(
        "# format: masy.daily_scenario.v1\n# scenario_id: wind_handoff_fixture\n"
        "# weather_file: weather.csv\n# start_date: 2030-06-01\n# end_date: 2030-06-02\n"
        "# evaporation_enabled: true\n# transpiration_enabled: false\n# growth_enabled: false\n"
        "# growth_interval_days: 7\n# output_interval_days: 1\n# checkpoint_interval_days: 1\n"
        "# output_at_event_boundaries: true\ndate,event_type,event_file\n"
        "2030-06-01,wind,wind.yaml\n"
        f"{'2030-06-01' if same_day else '2030-06-02'},wind,wind.yaml\n")
    return DailyPlan(load_daily_scenario(root / "scenario.csv"))


def make_native(root, *, depth=0.0, layers=(0.03, 0.07, 0.10), saturation=0.5):
    import numpy as np
    from maple_syrup import native_case_profile as ncp, native_dense_profile as nd
    from maple_syrup import native_floating_soil as nfs
    from maple_syrup.experimental_storm import ExperimentalControl
    from maple_syrup.pond_event import PondCompletionPolicy, PondEvent, PondEventControl
    from maple_syrup.pond_network import build_pond_topology
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup import sediment_physics as sp
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple_syrup.subsurface import default_host_available_bytes
    from masy import daily_wind_handoff as mh
    from types import SimpleNamespace

    bundle = Path(os.environ["MASY_DAILY_NATIVE_BUNDLE"])
    syrup = Path(os.environ["MASY_DAILY_NATIVE_SYRUP"])
    source = ncp.verify_source_profile(bundle, expected_syrup_digest=mh.SYRUP_PACKAGE_SHA256)
    case = Path(root) / "case"
    case.mkdir(parents=True)
    for name in ("case.yaml", "model.yaml", "wind.yaml"):
        shutil.copyfile(syrup / "docs/manuals/examples/native_leaf_calendar_cli/case" / name, case / name)
    admission = ncp.admit_package_case(source, case)
    prepared = nd.prepare_native_dense_case(admission, caller_reserve_bytes=1 << 20)
    bed, context = prepared.bed, prepared.context
    if depth:
        bed = dataclasses.replace(bed, water=dataclasses.replace(bed.water, depth_m=np.full_like(bed.water.depth_m, depth)))
    g = context.geometry
    shape = (g.ny, g.nx)
    active = np.ones(shape, dtype=bool)
    # The published native wind engineering control's explicit clay-rich map,
    # within the provider's supported density domain at this fixture's packing.
    # All classes share a texture: this control tests actual depth feedback;
    # separate native tests cover composition-dependent property changes.
    texture = np.tile(np.array([0.05, 0.45, 0.50]), (3, 1))
    cfg = nfs.native_floating_config(layer_thickness_m=layers, grain_texture=texture,
        organic_matter_percent=np.full(3, 3.0), bulk_density_kg_m3=g.bulk_density_kg_m3,
        latent_theta_field_capacity=np.full(3, 0.25), latent_theta_saturation=np.full(3, 0.42),
        theta_residual=0.05, theta_evaporation_stress=0.2, diffusivity_m2_s=np.full(2, 1e-8),
        bottom_drainage="open", matrix_conductivity_m_s=2e-6, matrix_suction_m=0.1,
        matrix_vegetation_factor=0.7, initial_wetting_m=0.01, initial_cumulative_intake_m=0.02,
        backend="numpy")
    nb = nfs.NativeBed(bed.voxel_column, bed.active_layer, g, context.grain_classes, context.mass_resolution_kg, active)
    owner = nfs.initialize_native_floating_soil(nb, cfg, initial_relative_saturation=saturation, available_bytes_fn=default_host_available_bytes)
    z = np.asarray(committed_elevation_m(bed.committed_topography))
    full = np.pad(z, 1, mode="edge")
    outlets = [(r, g.nx - 1, 0, 1) for r in range(g.ny)]
    for r, c, dr, dc in outlets:
        full[r + 1 + dr, c + 1 + dc] = z[r, c] - 0.01
    topology = build_pond_topology(full, active, np.full(shape, 5.0), g.dx_m, outlets)
    return SimpleNamespace(source=source, admission=admission, context=context, bed=bed, owner=owner,
                           factory=native_factory(context, topology), original_topology=topology, available=default_host_available_bytes)


def native_factory(context, topology):
    """Static recipe only: no terrain, water owner or soil provider initialization."""
    import numpy as np
    from maple_syrup import sediment_physics as sp
    from maple_syrup.experimental_storm import ExperimentalControl
    from maple_syrup.pond_event import PondEventControl, PondCompletionPolicy, PondEvent
    from maple_syrup.rainfall import constant_rainfall
    # Explicit fixture subset of the native table; no interpolation/calibration.
    indices = [0, 2, 5]
    pick = lambda a: np.asarray(a)[indices]
    sediment = sp.sediment_physics_parameters(
        diameter_m=np.array([c.diameter_m for c in context.grain_classes.classes]),
        raindrop_a=pick(sp.PLOT1_RAINDROP_A), raindrop_b=pick(sp.PLOT1_RAINDROP_B),
        raindrop_c=pick(sp.PLOT1_RAINDROP_C), raindrop_max_depth_mm=pick(sp.PLOT1_RAINDROP_MAX_MM),
        raindrop_depth_attenuation_per_cm=pick(sp.LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM),
        particle_density_g_cm3=sp.PLOT1_PARTICLE_DENSITY_G_CM3,
        active_layer_sensitivity_mm=sp.PLOT1_ACTIVE_LAYER_SENSITIVITY_MM, ke_model=sp.PLOT1_KE_MODEL, xp=np)
    control = PondEventControl(backend="numpy", time=ExperimentalControl(max_dt_s=0.2, min_dt_s=1.0 / 1024,
                                max_retries=12, max_steps=50000))
    policy = PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-4, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True)
    initial_schedule = constant_rainfall(0.0, 0.36, 100.0)

    def factory(vertical, carried_topology=None, schedule=None):
        return PondEvent(context, None, sediment, initial_schedule if schedule is None else schedule,
                         topology if carried_topology is None else carried_topology, control=control,
                         policy=policy, floating_vertical=vertical)

    return factory


def wind_limits():
    from masy.daily_wind import DailyWindLimits
    return DailyWindLimits(max_cells=1024, max_voxels=64, max_cohorts=16384, max_blocks=8,
        max_accounting_steps=4096, max_stage_bytes=1 << 24, max_work_bytes=1 << 28, max_receipt_bytes=1 << 20)


def handoff_limits():
    from maple_syrup.floating_wind_handoff import WindHandoffLimits
    return WindHandoffLimits(max_cells=1024, max_voxels=64, max_classes=16, max_cohorts=16384, max_batch_cells=64,
                            max_scan_bytes=1 << 26, max_candidate_bytes=1 << 27, max_receipt_bytes=1 << 20)


def physical_record(boundary):
    import hashlib
    from masy import daily_wind as dw
    leaves = {"water": boundary.root.water_state.water_m, "wetting": boundary.root.history.wetting_m,
              "intake": boundary.root.history.cumulative_intake_m, "surface": boundary.surface_depth_m,
              "qx": boundary.qx_m2_s, "qy": boundary.qy_m2_s, "routing_source": boundary.routing[1]}
    return {"bed_sha256": dw._bed_digest(boundary.bed), "column_sha256": boundary.root.column.sha256,
            "arrays": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in leaves.items()},
            "t_s_hex": boundary.t_s.hex(), "ledger": dict(boundary.ledger),
            "ordinal": None if boundary.cursor is None else boundary.cursor.ordinal}


def complete_native(occurrence):
    """Bounded orchestration of the genuine solver and retaining completion."""
    state = occurrence[1]
    event = occurrence[0].event
    for _ in range(8):
        result = event.evolve(state, float(state.t_s) + 100.0, stop_when_quiet=True)
        state = result.state
        if result.quiet_ready:
            handoff = event.settle_retaining_water(state)
            state = handoff.state
            if handoff.complete:
                return handoff, event.audit(state)
    raise AssertionError("native storm/recession is incomplete within the explicit test resource guard")


def run_native_day(bridge, fixture, day):
    """Test driver only: real planned rain/recession then one native daily close."""
    import numpy as np
    from maple_syrup import floating_event_lifecycle as lc, floating_event_daily_bridge as br
    from maple_syrup import floating_daily as fd, recharge as rc, soil_evaporation as se
    from maple_syrup.rainfall import constant_rainfall
    part = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
    native = br.open_day(bridge.lineage, bridge.lineage.head, partition=part,
                        run_digest=bridge.run_digest, available_bytes_fn=fixture.available)
    rain = rc.plan_daily_rain(day.weather.precipitation_mm, day.weather.precipitation_intensity_mm_h,
        bridge._plan.scenario.weather.precipitation_threshold_mm, run_digest=bridge.run_digest, ordinal=native.cursor.ordinal)
    pet = np.full(native.ctx.n_cells, day.weather.pet_mm_day * 1e-3 / 86400.0)
    cover = np.zeros(native.ctx.n_cells)
    result = br.day_action(native, plan=rain, pet_terrain_m_s=pet, cover=cover,
                           event_factory=fixture.factory, available_bytes_fn=fixture.available)
    audits = []
    if result.outcome == "requires_full_storm":
        pending = br.begin_storm_from_day(bridge.lineage, native, rain, pet_terrain_m_s=pet,
            cover=cover, event_factory=fixture.factory, available_bytes_fn=fixture.available)
        schedule = br.storm_schedule(pending.pending_forcing)
        occurrence = lc.begin_occurrence(bridge.lineage, pending,
            lambda v, t: fixture.factory(v, t, schedule), available_bytes_fn=fixture.available)
        handoff, audit = complete_native(occurrence)
        audits.append(audit)
        boundary = lc.close_occurrence(bridge.lineage, occurrence[0],
            occurrence[0].event.completion_capability(handoff, handoff.state), handoff.state, available_bytes_fn=fixture.available)
        for _ in range(4):
            if boundary.eligible_for_interevent:
                break
            recession = constant_rainfall(boundary.t_s, boundary.t_s + 1.0, 0.0)
            occurrence = lc.begin_occurrence(bridge.lineage, boundary,
                lambda v, t: fixture.factory(v, t, recession), available_bytes_fn=fixture.available)
            handoff, audit = complete_native(occurrence)
            audits.append(audit)
            boundary = lc.close_occurrence(bridge.lineage, occurrence[0],
                occurrence[0].event.completion_capability(handoff, handoff.state), handoff.state, available_bytes_fn=fixture.available)
        assert boundary.eligible_for_interevent
        native = br.open_day(bridge.lineage, boundary, partition=part, available_bytes_fn=fixture.available)
        result = br.day_after_storm(native, plan=rain, pet_terrain_m_s=pet, cover=cover,
                                   event_factory=fixture.factory, available_bytes_fn=fixture.available)
    assert result.outcome == "interevent"
    closed = fd.close_floating_day(result.state, expected_parent=fd.expect_parent(result.state))
    boundary = br.hand_back(bridge.lineage, closed.state)
    return boundary, audits
