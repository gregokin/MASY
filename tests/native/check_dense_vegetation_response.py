"""Task70 authored synthetic native-default wake interface, NOT calibration.

This separately named fixture changes only authored vegetation inputs. Original
StageB fixtures stay unchanged. All model physics/configuration/guards are native.
No transport is authorized merely by importing or collecting this source.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from unittest import mock

import check_dense_vegetation as old
import fixture_dense_production as production
import numpy as np
import pytest
from maple.aeolian.scheduler import baseline
from maple.aeolian.scheduler.vegetation_cache import (
    compute_vegetation_wind_cache_key,
    resolve_vegetation_wind_modifier,
    validate_vegetation_wind_cache,
)
from maple.aeolian.vegetation.apply import compute_vegetation_wind_modifier
from maple.core.types.cohort import zeros_mobile_cohort_state
from maple.core.types.vegetation import (
    DEFAULT_VEGETATION_DRAG_COEFFICIENT,
    DEFAULT_VEGETATION_POROSITY,
    vegetation_content_hash,
)
from maple_syrup.complete_event import DryHandoff

from masy import dense_bed as db
from masy import dense_runner as dr
from masy import landscape_vegetation as veg
from masy.clock import ClockLedger, decode_time
from masy.dense_production import DenseProductionLimits, schedule_envelope
from masy.dense_wind_profile import (
    SYNTHETIC_WAKE_AUDIT_SCHEMA,
    DenseSyntheticVegetationWakeProfile,
    build_dense_synthetic_vegetation_wake_config,
)

SHAPE = (4, 4)
CELLS = 16
FIXTURE_LABEL = (
    "MASY70 authored synthetic native-default vegetation wake interface; "
    "true heights0.08/0.04*cellspacing stored cover-weighted; NOT MEASURED; "
    "existing native synthetic100um reduced and wet Plot1-at100um fixture"
)


def response_runner(*, vegetation_limits=None):
    """Complete shared-bed fixture; no patched native builder or transport."""
    events = production.four_events()
    base = production.model(shape=SHAPE)
    cfg = dataclasses.replace(base, vegetation_wake=dataclasses.replace(base.vegetation_wake, enabled=True))
    spacing = min(cfg.geometry.dx_m, cfg.geometry.dy_m)
    # Native raster height_m is cover-weighted, not the true plant height.
    # True heights0.08/0.04*spacing yield wake lengths2/1cells, strictly
    # shorter than the4cell periodic repeat, with default half-cell samples.
    # Native ladders include0: v1wouldhave5wake/1windward samples but iswateronly.
    # Actual wind witnesses barev0(0/0) andv2(3wake/1windward).
    def replacement(cover, height, frontal):
        return veg.CellReplacement((0, 0), cover, cover * height * spacing,
            DEFAULT_VEGETATION_POROSITY, DEFAULT_VEGETATION_DRAG_COEFFICIENT, frontal)
    updates = (
        veg.PlannedDelta("wind", 0, "masy70_authored_increase", 0,
            (replacement(0.5, 0.08, 0.5),), "prescribed_increase"),
        veg.PlannedDelta("water", 0, "masy70_authored_decrease", 1,
            (replacement(0.25, 0.04, 0.25),), "prescribed_decrease"),
    )
    plan = veg.build_vegetation_plan(updates, shape=SHAPE, n_classes=1, n_voxels=6,
        event_kinds=tuple(event.kind for event in events.events), schedule_sha256=events.digest,
        limits=vegetation_limits)
    with mock.patch.object(production, "build_dense_bed", functools.partial(db.build_dense_bed, vegetation_profile=plan)):
        bed = production.build_bed(cfg, mound=False)
    profile = DenseSyntheticVegetationWakeProfile(cfg, "synthetic_test_fixture", FIXTURE_LABEL)
    envelope, _ = schedule_envelope(events)
    config = build_dense_synthetic_vegetation_wake_config(profile,
        wind_envelope_record=envelope, initial_available_fraction=bed.context.initial_available_fraction)
    wind = dr.DenseWindInputs(config, zeros_mobile_cohort_state(cfg.particle_transport.max_active_cohorts))
    water = production.water_inputs(bed)
    runner = dr.DenseRunner(bed, events, wind=wind, water=water,
        control=production.legacy.runner_control(), limits=production.legacy.limits(),
        wind_profile=profile, production_limits=DenseProductionLimits(), vegetation_plan=plan)
    return runner, cfg


def field_bytes(state):
    return {name: getattr(state.raster, name).tobytes() for name in veg.RASTER_FIELDS}


def same_modifier(actual, expected):
    assert type(actual) is type(expected)
    for field in dataclasses.fields(expected):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if type(b) is np.ndarray:
            np.testing.assert_array_equal(a, b, strict=True)
            assert a.tobytes() == b.tobytes()  # includes exact native NaN placement
        else:
            assert type(a) is type(b) and a == b, field.name


def accepted_step(runner):
    receipt = runner.step()
    assert receipt["status"] == "accepted", (receipt.get("code"), receipt.get("detail"))
    return receipt


def test_authored_fixture_binds_exact_synthetic_admission_and_native_defaults():
    runner, cfg = response_runner()
    audit = runner._static_parts["production"]["native_config_audit"]
    assert audit["schema"] == SYNTHETIC_WAKE_AUDIT_SCHEMA and audit["transport_ready"] is False
    assert audit["synthetic_interface"]["native_default_coefficients_only"] is True
    assert cfg.vegetation_wake.enabled is True
    assert dataclasses.replace(cfg.vegetation_wake, enabled=False) == type(cfg.vegetation_wake)()
    assert runner._wind.config.vegetation_wake_spec is cfg.vegetation_wake
    assert runner._bed.context.geometry is cfg.geometry
    assert runner._bed.context.initial_available_fraction is runner._wind.config.initial_available_fraction
    assert runner._vegetation_plan.cells == CELLS and len(runner._vegetation_plan.deltas_at(0)) == 1
    assert runner._bed._dynamic.version == 0 and runner._next == 0
    # The original factory fixture remains disabled; it was not relabelled.
    assert not production.model(shape=SHAPE).vegetation_wake.enabled


def test_four_events_apply_current_native_wake_and_owned_water_cover(tmp_path):
    runner, _cfg = response_runner()
    bed = runner._bed
    advance, complete, finalizer = dr._advance_wind_block, dr._complete_event, dr._finalize_wind_event
    reduced, prepare = baseline.advance_reduced_transport_step, bed._prepare_dynamic
    wind_seen, water_seen, candidates, starts, ends, returned = [], [], [], {}, {}, {}
    current = {}
    native_transport = {}
    actual_samples = {}
    prior_cache = []  # at most one retained prior-event cache

    def transport(*args, **kwargs):
        modifier = kwargs["vegetation_wind_modifier"]
        same_modifier(modifier, current["direct"])
        version = bed._dynamic.version
        stats = native_transport.setdefault(version, {"calls": 0, "nontrivial": False,
            "wake_nontrivial": False, "windward_nontrivial": False})
        stats["calls"] += 1
        stats["nontrivial"] |= bool(np.any(modifier.shear_velocity_ratio < 1.0))
        stats["wake_nontrivial"] |= bool(np.any(modifier.wake_shear_velocity_ratio < 1.0))
        stats["windward_nontrivial"] |= bool(np.any(modifier.windward_shear_velocity_ratio < 1.0))
        return reduced(*args, **kwargs)  # exact original arguments/result

    def wind(state, block, config):
        env, index = bed._dynamic, runner._next
        native = env._native_borrow()
        assert state.physical.vegetation is native and state.vegetation_wind_cache is None
        before = field_bytes(native)
        assert vegetation_content_hash(native) == env.native_content_sha256
        starts[index] = state.physical.time_s
        direct = compute_vegetation_wind_modifier(native, config.geometry,
            block.direction_deg, config.vegetation_wake_spec)
        current["direct"] = direct
        out = advance(state, block, config)
        assert out.conservation_ok and before == field_bytes(native)
        cache = out.state.vegetation_wind_cache
        assert cache is not None and cache.modifier is not None
        sample_counts = (cache.modifier.n_wake_samples, cache.modifier.n_windward_samples)
        assert sample_counts == ((0, 0) if env.version == 0 else (3, 1))
        actual_samples[env.version] = {"wake": sample_counts[0], "windward": sample_counts[1]}
        same_modifier(cache.modifier, direct)
        validate_vegetation_wind_cache(cache, *SHAPE,
            expected_vegetation=native, expected_wind_direction_deg=block.direction_deg)
        assert cache.cache_key == compute_vegetation_wind_cache_key(state.physical, block.direction_deg, config)
        reused, modifier = resolve_vegetation_wind_modifier(cache, out.state.physical, block.direction_deg, config)
        assert reused is cache and modifier is cache.modifier
        assert out.accounting.n_vegetation_recomputations >= 1
        if prior_cache:
            assert prior_cache[0].cache_key != cache.cache_key
            rebuilt, replacement = resolve_vegetation_wind_modifier(prior_cache[0],
                state.physical, block.direction_deg, config)
            assert rebuilt is not prior_cache[0] and rebuilt.cache_key == cache.cache_key
            same_modifier(replacement, direct)
        else:
            prior_cache.append(cache)
        wind_seen.append((env.version, env.native_content_sha256, before))
        returned[index] = out.advanced_duration_s
        return out

    def finish_wind(state, config, record, **kwargs):
        out = finalizer(state, config, record, **kwargs)
        assert out.completed and out.water_handoff_ready
        ends[runner._next] = out.state.physical.time_s
        return out

    def water(state, context, column, field, schedule, cover, sediment, **kwargs):
        env = bed._dynamic
        native = env._native_borrow()
        assert cover.flags.owndata and cover.flags.writeable
        assert not np.shares_memory(cover, native.raster.cover_fraction)
        np.testing.assert_array_equal(cover, native.raster.cover_fraction, strict=True)
        before = field_bytes(native)
        starts[runner._next] = state.t_s
        out = complete(state, context, column, field, schedule, cover, sediment, **kwargs)
        assert type(out) is DryHandoff and out.status == "complete"
        assert bool(np.any(out.pre_reset_soil_m > 0.0)) and out.accounting["soil_removed_m3"] > 0.0
        assert bool(np.all(out.dry_state.storm.soil_water_m == 0.0))
        assert before == field_bytes(native)
        ends[runner._next] = out.dry_state.t_s
        current["native_soil"] = out.dry_state.storm.soil_water_m
        water_seen.append((env.version, env.native_content_sha256, before, cover[0, 0]))
        return out

    def prepared(source, **kwargs):
        candidate = kwargs["prepared_vegetation"]
        expected_soil = runner._soil if kwargs["kind"] == "wind" else current["native_soil"]
        assert candidate.resource.soil is expected_soil and candidate.resource.water_exchange.channels == ()
        candidates.append(candidate.resource.soil)
        return prepare(source, **kwargs)

    with (mock.patch.object(dr, "_advance_wind_block", wind),
          mock.patch.object(dr, "_finalize_wind_event", finish_wind),
          mock.patch.object(dr, "_complete_event", water),
          mock.patch.object(baseline, "advance_reduced_transport_step", transport),
          mock.patch.object(bed, "_prepare_dynamic", prepared)):
        for index in range(4):
            accepted_step(runner)
            assert runner._soil is candidates[-1] and bed._dynamic.cursor["next_event_index"] == index + 1
    assert [item[0] for item in wind_seen] == [0, 2]
    assert [item[0] for item in water_seen] == [1, 2]
    assert wind_seen[1][1:3] == water_seen[1][1:3]  # all5 fields/native hash same version
    assert [item[3] for item in water_seen] == [0.5, 0.25]
    assert set(native_transport) == {0, 2} and all(row["calls"] > 0 for row in native_transport.values())
    assert not native_transport[0]["nontrivial"] and native_transport[2]["nontrivial"]
    assert native_transport[2]["wake_nontrivial"] and native_transport[2]["windward_nontrivial"]
    assert runner.report()["status"] == "complete" and bed._dynamic.version == 2
    assert [r["vegetation"]["input_version"] for r in runner.receipts] == [0, 1, 2, 2]
    assert [r["vegetation"]["delta_copy_bytes"] for r in runner.receipts] == [384, 384, 0, 0]
    assert [r["vegetation"]["required_snapshot_bytes"] for r in runner.receipts] == [640, 768, 640, 768]
    assert runner._vegetation_copied_bytes == 4352 and bool(np.all(runner._soil == 0.0))
    ledger = runner.ledger_record()
    assert ClockLedger.from_record(ledger).as_record() == ledger and len(ledger["entries"]) == 4
    for index, entry in enumerate(ledger["entries"]):
        assert decode_time(entry["native_start"]).hex() == float(starts[index]).hex()
        assert decode_time(entry["native_end"]).hex() == float(ends[index]).hex()
        if index in returned:
            assert decode_time(entry["returned_elapsed"]).hex() == float(returned[index]).hex()
        else:
            assert entry["returned_elapsed"] is None
    summary = {"schema": "masy70.derived_interface_witness.v1", "fixture": FIXTURE_LABEL,
        "wind_versions": [item[0] for item in wind_seen], "water_versions": [item[0] for item in water_seen],
        "shared_native_fields": list(veg.RASTER_FIELDS), "full_modifier_direct_native_equal": True,
        "native_cache_identity_reuse_and_stale_recompute": True, "actual_wind_samples": actual_samples,
        "actual_reduced_consumer": native_transport,
        "native_soil_zero_resets": 2, "soil_candidate_identity_preserved": True,
        "accepted_events": 4, "accepted_copy_bytes": runner._vegetation_copied_bytes,
        "clock_record_sha256": hashlib.sha256(json.dumps(ledger, sort_keys=True).encode()).hexdigest(),
        "scope": "small labelled synthetic CPU interface; no calibration/GPU/windowed/restart/scale qualification"}
    (tmp_path / "derived_response_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


@pytest.mark.parametrize("failure", ["native_wind", "late_prepare"])
def test_late_following_wind_failure_keeps_last_accepted_boundary(failure):
    runner, _ = response_runner()
    accepted_step(runner)
    accepted_step(runner)
    before = old.accepted_snapshot(runner)
    advance, seen = dr._advance_wind_block, []

    def completed(*args, **kwargs):
        out = advance(*args, **kwargs)
        assert out.conservation_ok and bool(np.any(out.state.vegetation_wind_cache.modifier.shear_velocity_ratio < 1.0))
        seen.append(True)
        if failure == "native_wind":
            raise ValueError("authored late actual following-wind failure")
        return out

    with mock.patch.object(dr, "_advance_wind_block", completed):
        if failure == "late_prepare":
            with mock.patch.object(runner._bed, "_verify_prepared",
                    side_effect=veg.VegetationError("authored_late_prepare", "no publication")):
                receipt = runner.step()
        else:
            receipt = runner.step()
    assert seen and receipt["status"] != "accepted"
    assert receipt["code"] == ("wind_native_failure" if failure == "native_wind" else "authored_late_prepare")
    assert old.accepted_snapshot(runner) == before and runner._next == runner._bed._dynamic.version == 2
    if failure == "native_wind":
        assert runner.active_evidence is None  # raising native call exposed no result to runner


@pytest.mark.parametrize("deficit", [1, 0])
def test_following_wind_copy_admission_at_exact_live_boundary(deficit):
    # Required runner traffic: initial6fields + firstwind(5snapshot+3delta)
    # + firstrain(6snapshot+3delta) =23cellfields, then followingwind5.
    before_bytes, following_bytes = 23 * 8 * CELLS, 5 * 8 * CELLS
    limits = dataclasses.replace(veg.VegetationLimits(), max_run_copy_bytes=before_bytes + following_bytes - deficit)
    runner, _ = response_runner(vegetation_limits=limits)
    accepted_step(runner)
    accepted_step(runner)
    assert runner._vegetation_copied_bytes == before_bytes
    assert runner._vegetation_event_copy_bound("wind") == following_bytes
    before = old.accepted_snapshot(runner)
    if deficit:
        with mock.patch.object(db.DenseBed, "stage_copy", side_effect=AssertionError("staging reached before copy admission")):
            receipt = runner.step()
        assert receipt["code"] == "vegetation_run_copy_cap" and old.accepted_snapshot(runner) == before
    else:
        accepted_step(runner)  # live positive control at the exact cap
        assert runner._vegetation_copied_bytes == before_bytes + following_bytes and runner._next == 3


def test_native_periodic_support_guard_still_refuses_unsupported_height():
    runner, cfg = response_runner()
    initial = runner._bed._dynamic._native_borrow()
    raster = initial.raster
    # Equality at25*trueheight ==1m repeat is rejected by the actual native API.
    repeat = cfg.geometry.nx * cfg.geometry.dx_m
    cover = np.full(SHAPE, 0.5)
    high = dataclasses.replace(initial, raster=dataclasses.replace(raster,
        cover_fraction=cover, height_m=cover * repeat / cfg.vegetation_wake.max_wake_length_plant_heights,
        frontal_area_density=np.full(SHAPE, 0.5)))
    before = old.accepted_snapshot(runner)
    with pytest.raises(ValueError, match="must be STRICTLY shorter than the periodic repeat"):
        compute_vegetation_wind_modifier(high, cfg.geometry, 270.0, cfg.vegetation_wake)
    assert old.accepted_snapshot(runner) == before and runner._next == 0
