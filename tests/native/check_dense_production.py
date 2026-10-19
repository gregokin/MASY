"""Bounded synthetic CPU interface checks; execute only in the task-owned OS-capped worker."""
from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import json
import resource
import sys

import fixture_dense_alternation as legacy
import fixture_dense_production as fixture
import numpy as np
import pytest

from masy import dense_production as dp
from masy import dense_runner as dr
from masy.dense_bed import SHARED_FIELDS


def emit(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    frame = b"\nMASY_FRAME " + str(len(payload)).encode() + b" " + hashlib.sha256(payload).hexdigest().encode() + b" " + base64.b64encode(payload) + b"\n"
    sys.__stdout__.buffer.write(frame)
    sys.__stdout__.buffer.flush()


@pytest.mark.parametrize("field,value", [("address_space_bytes", True), ("max_cells", 65), ("max_classes", 2),
                                          ("max_voxels", 17), ("cpu_soft_s", 121), ("cpu_hard_s", -1)])
def test_limit_types_and_scope(field, value):
    with pytest.raises(dp.DenseProductionError, match="invalid_production_limit"):
        dp.DenseProductionLimits(**{field: value})


@pytest.mark.parametrize("which,value", [(resource.RLIMIT_AS, (-1, -1)),
    (resource.RLIMIT_AS, (1 << 30, 4 << 30)), (resource.RLIMIT_CPU, (-1, -1)),
    (resource.RLIMIT_CORE, (1, 1)), (resource.RLIMIT_FSIZE, (-1, -1))])
def test_process_refuses_wrong_limits(monkeypatch, which, value):
    get = resource.getrlimit
    monkeypatch.setattr(resource, "getrlimit", lambda kind: value if kind == which else get(kind))
    with pytest.raises(dp.DenseProductionError, match="resource_limit_mismatch"):
        dp.check_process(dp.DenseProductionLimits())


def test_process_identity_environment_headroom(monkeypatch):
    limits = dp.DenseProductionLimits()
    current = dp.check_process(limits)
    assert current["limits"]["RLIMIT_AS"] == [4 << 30] * 2
    assert current["native_peak_estimate"] is None
    bad = {**current["process_identity"], "start_ticks": -1}
    with pytest.raises(dp.DenseProductionError, match="worker_identity_changed"):
        dp.check_process(limits, bad)
    with pytest.raises(dp.DenseProductionError, match="address_space_headroom"):
        dp.check_process(limits, upcoming_wrapper_bytes=4 << 30)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(dp.DenseProductionError, match="worker_environment_mismatch"):
        dp.check_process(limits)


def test_limits_forced_mutation_still_refuses():
    limits = dp.DenseProductionLimits()
    object.__setattr__(limits, "address_space_bytes", 8 << 30)
    with pytest.raises(dp.DenseProductionError, match="invalid_production_limit"):
        dp.check_process(limits)


def test_config_binding_and_context_identity():
    runner = fixture.make_runner()
    record = runner.static_record
    assert record["profile"]["scope"] == dp.SCOPE
    production = runner._static_parts["production"]
    assert production["schedule_binding"]["schedule_sha256"] == runner._schedule.digest
    assert production["schedule_binding"]["source_event_indices"] == [0, 2]
    assert production["schedule_binding"]["role"] == "forcing_bound_only_not_execution_schedule"
    assert len(runner._schedule.events) == 4  # rain has not been dropped from execution
    assert production["native_config_audit"]["calibration_declaration"]["status"] == "synthetic_test_fixture"
    cfg = runner._wind.config
    assert cfg.initial_available_fraction is runner._bed.context.initial_available_fraction
    assert cfg.prescribed_detachment is not None and cfg.transport_accuracy is not None
    assert cfg.reduced_detachment.out_of_envelope_policy == "reject"
    assert cfg.topographic_wind_spec.enabled and cfg.flow_separation_spec.enabled
    assert cfg.surface_creep_spec.enabled and cfg.avalanching_spec.enabled
    assert cfg.water_coupling_spec.depth_update_rule == runner._bed.context.depth_update_rule == "constant_depth"


def test_wrong_shared_context_and_water_rule_refuse():
    bed, schedule, wind, water, profile = fixture.components()
    kwargs = {"wind": wind, "water": water, "control": legacy.runner_control(), "limits": legacy.limits(),
                  "wind_profile": profile, "production_limits": dp.DenseProductionLimits()}
    wrong = dataclasses.replace(wind, config=dataclasses.replace(wind.config, geometry=copy.deepcopy(wind.config.geometry)))
    with pytest.raises(dr.DenseRunnerError, match="incompatible_wind_config"):
        dr.DenseRunner(bed, schedule, **{**kwargs, "wind": wrong})
    # BedContext is frozen; a different declared native model rule is refused without changing it.
    model = profile.model_config
    profile = dataclasses.replace(profile, model_config=dataclasses.replace(model,
        water_coupling=dataclasses.replace(model.water_coupling, depth_update_rule="constant_free_surface")))
    from masy.dense_wind_profile import build_dense_production_wind_config
    envelope, _ = dp.schedule_envelope(schedule)
    cfg = build_dense_production_wind_config(profile, wind_envelope_record=envelope,
                                             initial_available_fraction=bed.context.initial_available_fraction)
    with pytest.raises(dr.DenseRunnerError, match="incompatible_water_rule"):
        dr.DenseRunner(bed, schedule, **{**kwargs, "wind_profile": profile, "wind": dr.DenseWindInputs(cfg, wind.initial_mobile_state)})


def test_cap_change_at_entry_preserves_boundary(monkeypatch):
    runner = fixture.make_runner()
    before = runner.accepted
    get = resource.getrlimit
    monkeypatch.setattr(resource, "getrlimit", lambda kind: (-1, -1) if kind == resource.RLIMIT_AS else get(kind))
    receipt = runner.step()
    assert receipt["status"] == "refused" and receipt["code"] == "resource_limit_mismatch"
    assert runner.accepted == before


def test_production_memory_error_classification_without_transport(monkeypatch):
    runner = fixture.make_runner()
    before = runner.accepted
    def fail():
        raise MemoryError("authored failure injection before staging")
    monkeypatch.setattr(runner, "_precheck", fail)
    receipt = runner.step()
    assert receipt["status"] == "runner_error" and receipt["code"] == "address_space_exhausted"
    assert runner.accepted == before


def test_production_requires_explicit_opt_in_and_default_still_refuses():
    bed, events, wind, water, _ = fixture.components()
    with pytest.raises(dr.DenseRunnerError, match="unsupported_avalanching"):
        dr.DenseRunner(bed, events, wind=wind, water=water, control=legacy.runner_control(), limits=legacy.limits())
    with pytest.raises(dr.DenseRunnerError, match="production_limits requires"):
        dr.DenseRunner(bed, events, wind=wind, water=water, control=legacy.runner_control(), limits=legacy.limits(),
                       production_limits=dp.DenseProductionLimits())


def test_dimension_guard_and_exact_availability_identity():
    bed, events, wind, water, profile = fixture.components()
    kwargs = {"wind": wind, "water": water, "control": legacy.runner_control(), "limits": legacy.limits(),
                  "wind_profile": profile, "production_limits": dp.DenseProductionLimits(max_cells=1)}
    with pytest.raises(dr.DenseRunnerError, match="production_dimension_limit"):
        dr.DenseRunner(bed, events, **kwargs)
    wind = dataclasses.replace(wind, config=dataclasses.replace(wind.config,
        initial_available_fraction=np.array(bed.context.initial_available_fraction, copy=True)))
    with pytest.raises(dr.DenseRunnerError, match="incompatible_wind_config"):
        dr.DenseRunner(bed, events, **{**kwargs, "wind": wind, "production_limits": dp.DenseProductionLimits()})


def test_live_calibration_declaration_mutation_refuses(monkeypatch):
    runner = fixture.make_runner()
    before = runner.accepted
    object.__setattr__(runner._wind_profile, "calibration_reference", "changed external declaration")
    def forbidden(*args):
        raise AssertionError("native transport reached after provenance mutation")
    monkeypatch.setattr(dr, "_advance_wind_block", forbidden)
    receipt = runner.step()
    assert receipt["status"] == "refused" and receipt["code"] == "static_input_mutated_before_event", receipt
    assert runner.accepted == before


def test_actual_four_event_synthetic_interface():
    runner = fixture.make_runner(mound=False)
    initial_availability = runner._bed.state.sediment_availability.available_mass_kg.copy()
    receipts = []
    held_after_water = None
    for index in range(4):
        receipt = runner.step()
        emit({"test": "four_event_synthetic_interface", "receipt": receipt})
        assert receipt["status"] == "accepted", receipt
        receipts.append(receipt)
        if index == 1:
            held_after_water = runner._routing
        if index == 2:
            assert runner._routing is held_after_water
            assert runner._routing.graph is held_after_water.graph
            assert runner._routing.source is held_after_water.source
    report = runner.report()
    assert report["status"] == "complete"
    assert [r["kind"] for r in receipts] == ["wind", "rain", "wind", "rain"]
    assert all(r["clock"]["native_end"] == receipts[i+1]["clock"]["native_start"] for i, r in enumerate(receipts[:-1]))
    assert all(r["production"]["worker"]["limits"]["RLIMIT_AS"] == [4 << 30] * 2 for r in receipts)
    assert any(sum(r["production"]["transport_launched_mass_by_class_kg"]) > 0 for r in receipts if r["kind"] == "wind")
    assert any(sum(r["classes"]["actual_pickup_kg"]) > 0 for r in receipts if r["kind"] == "rain")
    assert not np.array_equal(initial_availability, runner._bed.state.sediment_availability.available_mass_kg)
    assert np.all(runner._bed.state.water.depth_m == 0)
    assert np.all(runner._bed.state.water.mobile_mass_by_cell_class_kg == 0)
    assert runner._routing is not None
    assert receipts[1]["routing"]["first_graph_construction"]
    assert receipts[1]["routing"]["graph_retained_by_identity"]
    assert not receipts[1]["routing"]["entry_rerouted"]
    assert not receipts[3]["routing"]["first_graph_construction"]
    assert receipts[3]["routing"]["entry_rerouted"]
    emit({"test": "four_event_synthetic_interface", "report": report})


def test_actual_four_event_direct_native_parity():
    from masy.dense_bed import _canonical_static
    runner = fixture.make_runner(mound=False)
    bed, events, wind, water, _ = fixture.components(mound=False)
    direct = fixture.direct_control(bed, wind.config, events, water)
    for state, time_s, native in direct:
        receipt = runner.step()
        assert receipt["status"] == "accepted", receipt
        assert runner._bed.time_s.hex() == float(time_s).hex()
        assert {name: _canonical_static(getattr(state, name), name) for name in SHARED_FIELDS} == {
            name: _canonical_static(getattr(runner._bed.state, name), name) for name in SHARED_FIELDS}
        assert receipt["clock"]["native_end"]["hex"] == float(time_s).hex()
        if native["kind"] == "wind":
            a, final = native["result"].accounting, native["final"]
            assert receipt["classes"]["boundary_export_kg"] == (
                a.boundary_export_by_class_kg + final.boundary_export_mass_by_class_kg).tolist()
            assert receipt["production"]["transport_launched_mass_by_class_kg"] == a.launched_mass_by_class_kg.tolist()
            assert receipt["finalization"]["deposited_mass_by_class_kg"] == final.deposited_mass_by_class_kg.tolist()
        else:
            wet = native["wet"]
            assert receipt["classes"]["export_kg"] == wet.progress.result.by_class["export_actual"].tolist()
            assert receipt["classes"]["actual_pickup_kg"] == wet.progress.result.by_class["actual_pickup"].tolist()
            assert receipt["water"]["rain_m3"] == wet.accounting["storm_water_budget"]["rain_m3"]
    assert runner.report()["status"] == "complete"
    emit({"test": "direct_native_parity", "shared_fields": list(SHARED_FIELDS), "native_clocks_and_budgets_equal": True})


def test_actual_native_separation_wind_avalanche_and_creep_effect():
    events = legacy.schedule(legacy.wind(1.0))
    on = fixture.make_runner(events=events)
    off = fixture.make_runner(events=events, cfg=fixture.model(creep=False))
    on_receipt, off_receipt = on.step(), off.step()
    assert on_receipt["status"] == off_receipt["status"] == "accepted", (on_receipt, off_receipt)
    actual = on_receipt["production"]
    assert actual["topographic_wind"]["separation_bubble_cells"] > 0
    assert actual["latest_wind_commit"]["avalanche_applied"]
    assert sum(actual["latest_wind_commit"]["avalanche_diagnostics"]["gross_moved_by_class_kg"]) > 0
    # Genuine enabled/disabled native outputs, not the conservative signed domain net.
    a, b = on._bed.state, off._bed.state
    active_l1 = float(np.abs(a.active_layer.mass_kg - b.active_layer.mass_kg).sum())
    available_l1 = float(np.abs(a.sediment_availability.available_mass_kg - b.sediment_availability.available_mass_kg).sum())
    assert max(active_l1, available_l1) > on._bed.context.mass_resolution_kg
    emit({"test": "native_feature_witness", "topographic_wind": actual["topographic_wind"],
          "latest_wind_commit": actual["latest_wind_commit"], "paired_native_creep_output_differs": True,
          "creep_pair_active_layer_l1_kg": active_l1, "creep_pair_available_l1_kg": available_l1})


def test_actual_native_envelope_refusal_preserves_boundary():
    runner = fixture.make_runner(events=legacy.schedule(legacy.wind(1.0, shear=0.7)), mound=False)
    before = runner.accepted
    receipt = runner.step()
    assert receipt["status"] == "native_refused" and receipt["code"] == "wind_native_failure", receipt
    assert "fitted shear-velocity interval" in receipt["detail"]
    assert runner.accepted == before


def test_actual_cap_change_immediately_before_publication(monkeypatch):
    runner = fixture.make_runner(events=legacy.schedule(legacy.wind(1.0)), mound=False)
    before = runner.accepted
    check, seen = dp.check_process, []
    def changing(*args, **kwargs):
        seen.append(kwargs.get("checkpoint"))
        if kwargs.get("checkpoint") == "before_publication":
            raise dp.DenseProductionError("resource_limit_mismatch", "authored final-check failure injection")
        return check(*args, **kwargs)
    monkeypatch.setattr(dp, "check_process", changing)
    receipt = runner.step()
    assert "before_publication" in seen
    assert receipt["code"] == "resource_limit_mismatch" and receipt["status"] == "refused", receipt
    assert receipt["unaccepted_elapsed_s"] is not None
    assert runner.active_evidence is not None
    assert runner.accepted == before


@pytest.mark.parametrize("checkpoint", ["after_native_wind", "after_native_rain"])
def test_actual_post_native_resource_refusal_keeps_elapsed_and_evidence(monkeypatch, checkpoint):
    events = (legacy.schedule(legacy.wind(1.0)) if checkpoint == "after_native_wind" else
              legacy.schedule(legacy.rain(), legacy.wind(1.0)))
    runner = fixture.make_runner(events=events, mound=False)
    before = runner.accepted
    check = dp.check_process
    def changing(*args, **kwargs):
        if kwargs.get("checkpoint") == checkpoint:
            raise dp.DenseProductionError("resource_limit_mismatch", "authored post-native failure injection")
        return check(*args, **kwargs)
    monkeypatch.setattr(dp, "check_process", changing)
    receipt = runner.step()
    assert receipt["status"] == "refused" and receipt["code"] == "resource_limit_mismatch"
    assert receipt["unaccepted_elapsed_s"] is not None and runner.active_evidence is not None
    assert runner.accepted == before


def test_construction_memory_error_keeps_original_bed(monkeypatch):
    bed, events, wind, water, profile = fixture.components(mound=False)
    before = bed.identity_sha256()
    def fail(*args, **kwargs):
        raise MemoryError("authored constructor guard allocation failure")
    monkeypatch.setattr(dp, "check_process", fail)
    with pytest.raises(MemoryError):
        dr.DenseRunner(bed, events, wind=wind, water=water, control=legacy.runner_control(), limits=legacy.limits(),
                       wind_profile=profile, production_limits=dp.DenseProductionLimits())
    assert bed.identity_sha256() == before


@pytest.mark.parametrize("checkpoint", ["wrapper_work", "after_native_wind", "after_native_rain", "before_publication"])
def test_actual_memory_error_at_process_checkpoint(monkeypatch, checkpoint):
    events = (legacy.schedule(legacy.rain(), legacy.wind(1.0)) if checkpoint == "after_native_rain" else
              legacy.schedule(legacy.wind(1.0)))
    runner = fixture.make_runner(events=events, mound=False)
    before = runner.accepted
    check = dp.check_process
    def fail(*args, **kwargs):
        if kwargs.get("checkpoint") == checkpoint:
            raise MemoryError("authored process checkpoint allocation failure")
        return check(*args, **kwargs)
    monkeypatch.setattr(dp, "check_process", fail)
    receipt = runner.step()
    assert receipt["status"] == "runner_error" and receipt["code"] == "address_space_exhausted"
    assert runner.accepted == before
    if checkpoint != "wrapper_work":
        assert receipt["unaccepted_elapsed_s"] is not None and runner.active_evidence is not None


def test_actual_unexpected_final_resource_error_keeps_context(monkeypatch):
    runner = fixture.make_runner(events=legacy.schedule(legacy.wind(1.0)), mound=False)
    before = runner.accepted
    check = dp.check_process
    def fail(*args, **kwargs):
        if kwargs.get("checkpoint") == "before_publication":
            raise OSError("authored unexpected final observation error")
        return check(*args, **kwargs)
    monkeypatch.setattr(dp, "check_process", fail)
    receipt = runner.step()
    assert receipt["status"] == "runner_error" and receipt["code"] == "production_resource_observation_failure"
    assert receipt["unaccepted_elapsed_s"] is not None and runner.active_evidence is not None
    assert runner.accepted == before


@pytest.mark.parametrize("kind", ["wind", "rain"])
@pytest.mark.parametrize("error", [MemoryError, TypeError])
def test_actual_post_native_diagnostic_failure_keeps_context(monkeypatch, kind, error):
    events = (legacy.schedule(legacy.rain(), legacy.wind(1.0)) if kind == "rain" else
              legacy.schedule(legacy.wind(1.0)))
    runner = fixture.make_runner(events=events, mound=False)
    before = runner.accepted
    def fail(*args, **kwargs):
        raise error("authored retained-owner diagnostic failure")
    monkeypatch.setattr(dp, "retained_wrapper_owners", fail)
    receipt = runner.step()
    expected = "address_space_exhausted" if error is MemoryError else "production_evidence_failure"
    assert receipt["status"] == "runner_error" and receipt["code"] == expected
    assert receipt["unaccepted_elapsed_s"] is not None and runner.active_evidence is not None
    assert runner.accepted == before


def test_actual_mound_sink_failure_keeps_accepted_wind():
    runner = fixture.make_runner()
    assert runner.step()["status"] == "accepted"
    before = runner.accepted
    receipt = runner.step()
    assert receipt["code"] == "rain_entry_failure" and receipt["status"] == "native_refused", receipt
    assert "unsupported sinks" in receipt["detail"]
    assert runner.accepted == before
    with pytest.raises(dr.DenseRunnerError, match="runner_halted"):
        runner.step()


def test_actual_failure_after_water_does_not_publish(monkeypatch):
    runner = fixture.make_runner(mound=False)
    assert runner.step()["status"] == "accepted"
    assert runner.step()["status"] == "accepted"
    before = runner.accepted
    routing = runner._routing
    def fail(*args):
        raise MemoryError("authored native failure after accepted water")
    monkeypatch.setattr(dr, "_advance_wind_block", fail)
    receipt = runner.step()
    assert receipt["code"] == "address_space_exhausted" and receipt["status"] == "runner_error"
    assert runner.accepted == before and runner._routing is routing


def test_actual_native_water_entry_avalanche_with_open_outlets():
    events = legacy.schedule(legacy.rain(), legacy.wind(1.0))
    bed, events, wind, _, profile = fixture.components(events=events, cfg=fixture.model(shape=(3, 3)))
    water = fixture.water_inputs(bed, all_edges_open=True)
    runner = dr.DenseRunner(bed, events, wind=wind, water=water, control=legacy.runner_control(),
        limits=legacy.limits(), wind_profile=profile, production_limits=dp.DenseProductionLimits())
    receipt = runner.step()
    emit({"test": "native_water_entry_avalanche_attempt", "receipt": receipt})
    assert receipt["status"] == "accepted", receipt.get("detail")
    commit = receipt["production"]["entry_commit"]
    assert commit["avalanche_applied"]
    assert sum(commit["avalanche_diagnostics"]["gross_moved_by_class_kg"]) > 0
    emit({"test": "native_water_entry_avalanche", "scope": "authored3x3_all_edges_open_separate_witness",
          "entry_commit": commit, "accepted_time_s": bed.time_s.hex()})


def test_actual_airborne_finalization_before_and_after_water():
    from masy.dense_bed import _canonical_static
    events = legacy.schedule(legacy.wind(0.025), legacy.rain(), legacy.wind(0.025))
    cfg = fixture.model(outer=0.025)
    runner = fixture.make_runner(events=events, cfg=cfg, mound=False)
    bed, events, wind, water, _ = fixture.components(events=events, cfg=fixture.model(outer=0.025), mound=False)
    winds = []
    for state, time_s, native in fixture.direct_control(bed, wind.config, events, water):
        receipt = runner.step()
        assert receipt["status"] == "accepted", receipt
        assert runner._bed.time_s.hex() == float(time_s).hex()
        assert {name: _canonical_static(getattr(state, name), name) for name in SHARED_FIELDS} == {
            name: _canonical_static(getattr(runner._bed.state, name), name) for name in SHARED_FIELDS}
        if native["kind"] == "wind":
            assert sum(receipt["finalization"]["airborne_mass_before_by_class_kg"]) > 0
            assert sum(receipt["finalization"]["deposited_mass_by_class_kg"]) > 0
            assert receipt["wind_end_pool"] == "genuinely_empty"
            winds.append(receipt["finalization"])
    assert len(winds) == 2 and winds[1]["record_origin_s"]["hex"] != "0x0.0p+0"
    emit({"test": "nonempty_native_finalization", "events": 3,
          "before_and_after_water": True, "shared_fields_and_native_clocks_equal": True,
          "airborne_before_kg": [w["airborne_mass_before_by_class_kg"] for w in winds]})
