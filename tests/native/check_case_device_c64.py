"""Root-only imported NPY C64 interface checks, not production calibration.

Root installs the original one600s/all-events+oracle controls and supplies a
protected device_water_c64_case_budget(binding) fixture. This file issues no
authority, changes no allocator/limit, and contains no transport physics.
"""
from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import fixture_windowed_device_c64_alternation as f
import check_windowed_device_c64_alternation as parity
from tests.phase22b.test_reduced_detachment_config import _MODEL_RAW
from tests.phase22b.test_reduced_integration import SYNTHETIC_FIXTURE_CALIBRATION
from tests.windowed_wind_gpu.test_device_k1_dimension import _dimension_limits
from tests.windowed_wind_gpu.test_device_reduced_step import cupy  # noqa: F401
from maple.core.backend import read_transfer_counters, to_host
from maple.core.types.topographic_commit import committed_elevation_m
from maple.surface.active_layer.bed_exchange_device import DeviceWindowedK1DimensionProfile
from maple_syrup import sediment_physics as wet
from maple_syrup import windowed_shared_bed_device as shared

from masy.case_device_c64 import (
    CaseDeviceC64Error, CaseDeviceC64PlacementLimits,
    device_c64_request_binding, prepare_device_c64_case,
)
from masy.case_water import CaseWaterCopyLimits, CaseWaterSpec, CaseWetClass
from masy.case_windowed import CaseWindowedLimits, adopt_windowed_case, prepare_case_windowed_water
from masy.case_workflow import CaseInputLimits, resolve_case_inputs
from masy.dense_device_resources import DeviceResourceLimits
from masy.dense_wind_profile import DenseProductionWindProfile, build_dense_production_wind_config
from masy.windowed_device_c64_bed import DeviceWindowedC64BedLimits, DeviceWindowedC64BedOwner

pytestmark = pytest.mark.gpu


def _package(root):
    """Normal three-YAML case and supplied NPY; no Native genesis factory."""
    root.mkdir()
    (root / "source").mkdir()
    rows, cols = np.indices((8, 8), dtype=np.float64)
    dem = np.ascontiguousarray(.31 + .001 * rows + .0005 * cols)
    np.save(root / "source/dem.npy", dem, allow_pickle=False)
    raw = copy.deepcopy(_MODEL_RAW)
    geo = copy.deepcopy(raw["geometry"])
    geo.update(nx=8, ny=8)
    grain = copy.deepcopy(raw["grain_classes"][0])
    grain["diameter_m"] = SYNTHETIC_FIXTURE_CALIBRATION["calibrated_grains"][0]["diameter_m"]
    case = dict(run_name="synthetic_imported_C64_interface", seed=9281,
        output_dir="outputs/imported_c64", geometry=geo, grain_classes=[grain],
        topography={"base": "imported", "nz": 20,
                    "perturbation": {"relief_m": 0., "correlation_length_m": 1.}},
        **{"import": {"elevation": {"source": {"path": "source/dem.npy", "format": "npy",
            "row_order": "south_to_north", "value_kind": "length", "value_units": "m"}, "transforms": []}}},
        stratigraphy={"layers": [{"fractions": {grain["class_id"]: 1.}},
                                  {"thickness_m": .07, "fractions": {grain["class_id"]: 1.}}]},
        vegetation={"source": "raster", "raster": {"height_m": .5, "cover_fraction": .25,
            "porosity": .5, "drag_coefficient": 1., "frontal_area_density": .1}},
        sediment_availability={"global_available_fraction": .25},
        requested_diagnostics={"spatial_snapshots": False})
    for name in ("run_name", "seed", "output_dir", "geometry", "grain_classes", "topography",
                 "grain_mixture", "stratigraphy", "vegetation", "sediment_availability"):
        raw.pop(name, None)
    calibration = copy.deepcopy(SYNTHETIC_FIXTURE_CALIBRATION)
    calibration["out_of_envelope_policy"] = "reject"  # Existing labelled C64 interface setting.
    raw.update(detachment_source={"mode": "reduced"}, reduced_detachment=calibration,
        particle_transport={"quantile_bins": 8, "max_active_cohorts": 4000,
            "mobile_state_representation": "flat", "hop_distance_by_class": {grain["class_id"]: {
                "median_hop_distance_m": .3, "geometric_stddev": 2., "representative_flight_speed_m_s": 3.}}},
        time_and_forcing={"outer_timestep_s": 1.}, topographic_wind={"enabled": True},
        flow_separation={"enabled": True}, surface_creep={"enabled": True}, avalanching={"enabled": True})
    raw.setdefault("water_coupling", {}).update(enabled=False, depth_update_rule="constant_depth")
    raw.setdefault("topographic_commit", {}).update(pending_elevation_trigger_m=1e-9)
    wind = {"wind_forcing": {"blocks": [dict(duration_s=1., shear_velocity_m_s=.5, direction_deg=90.)
                                        for _ in range(2)]}}
    for name, content in (("case.yaml", case), ("model.yaml", raw), ("wind.yaml", wind)):
        (root / name).write_text(yaml.safe_dump(content, sort_keys=False))
    return root


def _host(root):
    io = CaseInputLimits(1 << 20, 256, 8 << 20, 16 << 20, 16 << 20, 128 << 20)
    compiled = resolve_case_inputs(_package(root / "input_case"), mode="compile", input_limits=io)
    host = adopt_windowed_case(compiled, store_dir=root / "deep", datum_offset_m=0.,
        window_voxels=3, chunk_columns=64, rows_per_band=2, with_prefix_packets=True,
        limits=CaseWindowedLimits(64, 1, 20, 128 << 20, 4 << 20, 64 << 20))
    surface = committed_elevation_m(host.state.committed_topography)
    ring = np.zeros((10, 10), dtype=np.float64)
    ring[1:-1, 1:-1] = surface
    exports = np.ones((10, 10), dtype=bool)
    exports[1:-1, 1:-1] = False
    column = f.legacy.column((8, 8))
    scalar = wet.sediment_physics_parameters(diameter_m=[host.config.grain_classes.classes[0].diameter_m],
        raindrop_a=wet.PLOT1_RAINDROP_A[:1], raindrop_b=wet.PLOT1_RAINDROP_B[:1],
        raindrop_c=wet.PLOT1_RAINDROP_C[:1], raindrop_max_depth_mm=wet.PLOT1_RAINDROP_MAX_MM[:1],
        raindrop_depth_attenuation_per_cm=wet.LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM[:1],
        particle_density_g_cm3=wet.PLOT1_PARTICLE_DENSITY_G_CM3,
        active_layer_sensitivity_mm=wet.PLOT1_ACTIVE_LAYER_SENSITIVITY_MM, ke_model=wet.PLOT1_KE_MODEL)
    wc = CaseWetClass(host.config.grain_classes.classes[0].class_id,
        float(wet.PLOT1_RAINDROP_A[0]), float(wet.PLOT1_RAINDROP_B[0]), float(wet.PLOT1_RAINDROP_C[0]),
        float(wet.PLOT1_RAINDROP_MAX_MM[0]), float(wet.LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM[0]))
    spec = CaseWaterSpec("synthetic_imported_c64", "authored imported C64 interface; NOT MEASURED",
        ring, exports, np.ones((8, 8), dtype=bool), np.ones((8, 8)), False, None, True, False,
        column.model, column.ksat_m_per_s, column.suction_m, column.drainage_parameter, column.theta_sat,
        column.soil_thickness_m, None, np.zeros((8, 8)), None, np.ones((8, 8)), (wc,),
        wet.PLOT1_ACTIVE_LAYER_SENSITIVITY_MM, scalar.ke_model, scalar.kinematic_viscosity_m2_s,
        scalar.reference_interval_s, scalar.recession_factor_per_reference_s, scalar.ke_vegetation_form,
        scalar.raindrop_composition_scaling, scalar.distance_convention, scalar.dstar_convention, scalar.bagnold_depth_units)
    host_control = f.legacy.runner_control(sediment_control=f.legacy.sediment_control(root_solver="bisection"))
    host_control = dataclasses.replace(host_control,
        completion_policy=dataclasses.replace(host_control.completion_policy, terminal_storage_water=True))
    prepared = prepare_case_windowed_water(host, spec, host_control,
        limits=CaseWaterCopyLimits(64, 1, 128 << 20, 4 << 20))
    cuda = dataclasses.replace(host_control, sediment_control=dataclasses.replace(host_control.sediment_control,
        storm=dataclasses.replace(host_control.sediment_control.storm, implementation="cuda")))
    source_record = host.static_inputs["resolved"]["wind_record"]
    cfg = build_dense_production_wind_config(DenseProductionWindProfile(host.config, f.CALIBRATION, f.LABEL),
        wind_envelope_record=source_record, initial_available_fraction=host.context.initial_available_fraction)
    limits = _dimension_limits(DeviceWindowedK1DimensionProfile(8, 8, 20, 3, 1, 8), cfg)
    dense_events = f.events().as_input()["events"]
    schedule = {"schema": "masy.case_schedule.v1", "events": [
        {"kind": "wind", "first_block": 0, "stop_block": 1},
        dict(dense_events[1], water_profile_id=prepared.profile_id),
        {"kind": "wind", "first_block": 1, "stop_block": 2},
        dict(dense_events[3], water_profile_id=prepared.profile_id)]}
    kwargs = dict(schedule_input=schedule, cuda_control=cuda, record_limits=limits,
        bridge_limits=f.bridge_limits(root),
        bed_limits=DeviceWindowedC64BedLimits(1 << 24, 5 << 30, 2 << 30, 16 << 30),
        resource_limits=DeviceResourceLimits(64 << 20, 64 << 20, 2 << 30, 4 << 20),
        runner_limits=f.legacy.limits(), placement_limits=CaseDeviceC64PlacementLimits(5 << 30, 2 << 30, 4 << 20),
        calibration=f.CALIBRATION, label="synthetic imported NPY C64 interface; NOT MEASURED")
    return host, prepared, kwargs


def _prepare(root, issuer):
    host, water, kwargs = _host(root)
    binding = device_c64_request_binding(host, water, **kwargs)
    budget = issuer(binding)  # Protected Root fixture; no test-owned issuance/fallback.
    assert type(budget) is shared.DeviceWaterC64CaseBudget
    return host, water, kwargs, binding, budget


def test_imported_case_HOST_request_binding_is_complete(tmp_path):
    """Real compiler/adoption/wet preparation, with no grant or placement."""
    host, water, kwargs = _host(tmp_path)
    before = dataclasses.asdict(read_transfer_counters())
    identity = host.identity_sha256
    binding = device_c64_request_binding(host, water, **kwargs)
    record = host.identity_record()
    assert binding["schema"] == "masy.device_c64_case_request.v1"
    assert binding["windowed_identity_sha256"] == identity
    assert binding["native_case_identity_sha256"] == record["native_case_identity_sha256"]
    assert binding["water_identity_sha256"] == water.identity_sha256
    assert binding["authored_file_pins"] == record["file_pins"]
    assert binding["storage_choices"] == record["storage_choices"]
    assert binding["native_committed_regrouping"] == record["native_committed_regrouping"]
    assert binding["native_committed_regrouping"]["exact_height_checked"] is True
    assert binding["storage_choices"]["trusted_packet"] is False
    grouped = binding["case_schedule"]
    assert grouped["case_sha256"] == identity
    assert grouped["water_identity_sha256"] == water.identity_sha256
    assert len(grouped["original_blocks"]) == 2
    assert [(row["first_block"], row["stop_block"], row["local_to_original"])
            for row in grouped["events"] if row["kind"] == "wind"] == [(0, 1, [0]), (1, 2, [1])]
    assert [row["kind"] for row in binding["projected_schedule"]["events"]] == ["wind", "rain", "wind", "rain"]
    assert binding["backend_changes"] == ["sediment_control.storm.implementation"]
    assert binding["derived_cuda"] == {"kind": "Native read-only property", "host": False, "device": True}
    assert water.control.sediment_control.cuda is False
    assert kwargs["cuda_control"].sediment_control.cuda is True
    assert kwargs["cuda_control"].sediment_control.storm.root_solver == water.control.sediment_control.storm.root_solver
    for name in ("record_limits", "bridge_limits", "bed_limits", "resource_limits", "runner_limits", "placement_limits"):
        assert binding[name]
    assert device_c64_request_binding(host, water, **kwargs) == binding
    assert host.identity_sha256 == identity
    assert host.verify() is host and water.verify(host) is water
    assert dataclasses.asdict(read_transfer_counters()) == before


def test_imported_case_four_events_match_native(tmp_path, cupy, device_water_c64_case_budget):
    host, water, kwargs, binding, budget = _prepare(tmp_path, device_water_c64_case_budget)
    original_identity = host.identity_sha256
    a = prepare_device_c64_case(host, water, qualification_budget=budget, request_binding=binding, **kwargs)
    # Independent device owners start the SAME immutable actual HOST genesis.
    b = prepare_device_c64_case(host, water, qualification_budget=budget, request_binding=binding, **kwargs)
    guarded = a.build_runner()
    assert guarded is a and a.build_runner() is a
    assert not hasattr(a, "runner")
    original_prefix = a.report()["known_prefix"]
    changed_report = a.report()
    changed_report["known_prefix"]["known_host_bytes"] = 0
    changed_report["known_prefix"]["known_device_bytes"] = 0
    changed_report["known_prefix"]["terms"]["actual_H2D_payload_bytes"] = 0
    assert a.report()["known_prefix"] == original_prefix
    # Genuine state/context and genuine budget, but no producer capability.
    before_unbound = dataclasses.asdict(read_transfer_counters())
    with pytest.raises(CaseDeviceC64Error, match="case_admission"):
        DeviceWindowedC64BedOwner.from_device_bed(b.bed._accepted.state, b.bed.context,
            case_record=b.bed.case_record, time_s=b.bed.time_s, limits=kwargs["bed_limits"],
            vegetation=b.bed.vegetation, threshold_modifier_state=b.bed.threshold_modifier_state,
            trajectory_semantics=b.bed._semantics)
    assert dataclasses.asdict(read_transfer_counters()) == before_unbound
    initial_version = host.state.voxel_column.deep_store
    for prepared in (a, b):
        assert prepared.bed._accepted.state.voxel_column.deep_store is initial_version
        assert initial_version.header.class_ids == tuple(c.class_id for c in host.config.grain_classes.classes)
        assert prepared.bed.context.geometry is host.config.geometry
        assert prepared.bed.context.grain_classes is host.config.grain_classes
        assert prepared.bed.context.topographic_commit_spec is host.config.topographic_commit
        assert prepared.bed.context.avalanching_spec is host.config.avalanching
        assert prepared.bed.time_s.hex() == host.time_s.hex()
        assert prepared._runner._water.vegetation_cover_fraction is prepared.bed.vegetation.raster.cover_fraction
        for name in f.VEGETATION_FIELDS:
            np.testing.assert_array_equal(to_host(getattr(prepared.bed.vegetation.raster, name)),
                                          getattr(host.vegetation.raster, name))
        parity._same(to_host(prepared.water.initial_soil_water_m), water.water.initial_soil_water_m, "initial soil")
        parity._same(to_host(prepared.bed._accepted.state.sediment_availability.available_mass_kg),
                     host.state.sediment_availability.available_mass_kg, "initial actual availability")
        prefix = prepared.report()["known_prefix"]
        before = prefix["actual_placement_transfer_before"]
        after = prefix["actual_placement_transfer_after_placement"]
        assert after["host_to_device_bytes"] - before["host_to_device_bytes"] == prefix["terms"]["actual_H2D_payload_bytes"]
    oracle = SimpleNamespace(bed=b.bed, wind=b.wind, water=b.water, control=b.control,
        schedule=b.schedule, profile=b.profile, record_limits=b.record_limits,
        physical=SimpleNamespace(vegetation=b.bed.vegetation,
                                 threshold_modifier_state=b.bed.threshold_modifier_state))
    # Oracle is created before a evolves; its own accepted state is untouched.
    direct = f.direct_events(oracle, cupy)
    for index in range(4):
        expected = next(direct)
        receipt = guarded.step()
        assert receipt["status"] == receipt["code"] == "accepted"
        assert guarded.stop_receipt is None
        assert guarded.accepted.next_index == index + 1
        parity._state_equal(a.bed._accepted.state, expected.state)
        assert a.bed.time_s.hex() == expected.time_s.hex()
        if expected.kind == "wind":
            assert expected.result.record.all_blocks_conserved
            assert expected.owner.finalizer_call_count == 1
            assert expected.result.finalization.completed
            assert expected.result.finalization.water_handoff_ready
            assert receipt["native"]["record"]["finalizer_calls"] == 1
            assert not bool(cupy.any(expected.pool.mass_kg))
        if index in (1, 3):
            wet_result = expected.wet
            water_budget = wet_result.accounting["storm_water_budget"]
            assert wet_result.accounting["sediment_closure"]["closed"]
            assert water_budget["closed"]
            assert wet_result.progress.result.n_accepted_steps > 0
            assert wet_result.progress.result.n_maple_water_calls > 0
            assert float(to_host(wet_result.progress.result.cumulative_rain_m.sum())) > 0.0
            for name in ("rain_m3", "surface_initial_m3", "soil_initial_m3", "surface_final_m3",
                         "soil_final_m3", "drainage_m3", "export_m3", "residual_m3", "tolerance_m3"):
                assert receipt["water"][name] == water_budget[name]
            for out, native in (("dry_surface_removed_m3", "surface_removed_m3"),
                                ("dry_soil_removed_m3", "soil_removed_m3")):
                assert receipt["water"][out] == wet_result.accounting[native]
            accounting = wet_result.accounting
            metrics = accounting["completion_after_terminal"]
            assert wet_result.progress.policy.terminal_storage_water is True
            assert wet_result.progress.policy.deposit_mobile_at_flow_end is True
            if "terminal_surface_volume_m3" in metrics:
                assert accounting["terminal_storage_removed_m3"] == metrics["terminal_surface_volume_m3"]
                assert accounting["moving_surface_removed_m3"] == (
                    accounting["surface_removed_m3"] - accounting["terminal_storage_removed_m3"])
            else:
                # Native quiet_metrics keeps its strict schema when this grid
                # has no terminal mask; it does not synthesize ponding fields.
                assert "terminal_storage_removed_m3" not in accounting
                assert "moving_surface_removed_m3" not in accounting
            assert not bool(cupy.any(a.bed._accepted.state.water.depth_m))
            assert not bool(cupy.any(a.bed._accepted.state.water.mobile_mass_by_cell_class_kg))
            assert not bool(cupy.any(expected.soil))
            assert expected.wet.dry_state.phase is None
    with pytest.raises(StopIteration):
        next(direct)
    assert a.report()["runner"]["status"] == "complete"
    for receipt in a.report()["runner"]["receipts"]:
        assert np.all(np.abs(receipt["aggregate"]["joint_residual_kg"]) <=
                      np.asarray(receipt["aggregate"]["joint_bound_kg"]))
    assert host.identity_sha256 == original_identity
    host.verify(); water.verify(host)
    # Caller custody is intentional: public step must catch later mutation
    # before transport, and restoring inputs must not clear the stop latch.
    unused = b.build_runner()
    kwargs["schedule_input"]["events"][0]["stop_block"] = 2
    before_guarded = dataclasses.asdict(read_transfer_counters())
    with pytest.raises(ValueError, match="wind_partition"):
        unused.step()
    assert dataclasses.asdict(read_transfer_counters()) == before_guarded
    assert unused.accepted.next_index == 0 and unused.stop_receipt is None
    assert unused.report()["input_guard_failed"] is True
    kwargs["schedule_input"]["events"][0]["stop_block"] = 1
    before_stopped = dataclasses.asdict(read_transfer_counters())
    with pytest.raises(CaseDeviceC64Error, match="case_stopped"):
        unused.step()
    assert dataclasses.asdict(read_transfer_counters()) == before_stopped


def test_imported_case_old_budget_refuses_before_placement(tmp_path, device_water_c64_case_budget):
    host, water, kwargs, binding, budget = _prepare(tmp_path, device_water_c64_case_budget)
    old = shared.DeviceWaterC64QualificationBudget(budget.native_source_identity, budget.current_runtime_ref,
        budget.producer_source, budget.grant, budget.source_bindings, budget.gpu_uuid,
        budget.source_manifest_sha256)
    before = dataclasses.asdict(read_transfer_counters())
    with pytest.raises(CaseDeviceC64Error, match="case_budget_required"):
        prepare_device_c64_case(host, water, qualification_budget=old, request_binding=binding, **kwargs)
    assert dataclasses.asdict(read_transfer_counters()) == before


def test_imported_case_changed_source_refuses_before_placement(tmp_path, device_water_c64_case_budget):
    host, water, kwargs, binding, budget = _prepare(tmp_path, device_water_c64_case_budget)
    path = host._case_dir / "wind.yaml"
    path.write_bytes(path.read_bytes() + b"\n# changed after binding\n")
    before = dataclasses.asdict(read_transfer_counters())
    with pytest.raises(ValueError, match="source_changed"):
        prepare_device_c64_case(host, water, qualification_budget=budget, request_binding=binding, **kwargs)
    assert dataclasses.asdict(read_transfer_counters()) == before


def test_imported_case_control_and_schedule_refuse_before_placement(tmp_path, device_water_c64_case_budget):
    host, water, kwargs, binding, budget = _prepare(tmp_path, device_water_c64_case_budget)
    before = dataclasses.asdict(read_transfer_counters())
    changed = dataclasses.replace(kwargs["cuda_control"], report_every_s=11.)
    with pytest.raises(CaseDeviceC64Error, match="water_control_changed"):
        prepare_device_c64_case(host, water, qualification_budget=budget, request_binding=binding,
                               **dict(kwargs, cuda_control=changed))
    schedule = copy.deepcopy(kwargs["schedule_input"])
    schedule["events"][0]["stop_block"] = 2
    with pytest.raises(ValueError, match="wind_partition"):
        prepare_device_c64_case(host, water, qualification_budget=budget, request_binding=binding,
                               **dict(kwargs, schedule_input=schedule))
    unsupported = dataclasses.replace(kwargs["record_limits"], max_blocks=2, max_total_outer_steps=2)
    with pytest.raises(ValueError, match="incompatible_dimension_policy"):
        prepare_device_c64_case(host, water, qualification_budget=budget, request_binding=binding,
                               **dict(kwargs, record_limits=unsupported))
    changed_binding = copy.deepcopy(binding)
    changed_binding["label"] += " changed without Root issuance"
    with pytest.raises(CaseDeviceC64Error, match="request_binding"):
        prepare_device_c64_case(host, water, qualification_budget=budget, request_binding=changed_binding, **kwargs)
    assert dataclasses.asdict(read_transfer_counters()) == before
