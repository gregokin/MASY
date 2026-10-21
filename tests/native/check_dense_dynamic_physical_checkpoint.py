"""Authored tiny dynamic physical artifacts; component checks perform no events.

Actual source/environment identities are supplied by the separately gated worker.
These fixtures use native constructors/default coefficients, never calibration.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from unittest import mock

import fixture_dense_alternation as f
import numpy as np
import pytest
from maple.aeolian.scheduler import physical_checkpoint as cp

from masy import dense_bed as db
from masy import dense_checkpoint as bridge
from masy import dense_original_inputs as oi
from masy import dense_runner as dr
from masy import dense_runner_checkpoint as boundary
from masy import landscape_vegetation as lv
from masy.clock import ClockLedger

SOURCES = json.loads(os.environ["MASY139_SOURCE_IDS"])
ENVIRONMENT = os.environ["MASY139_ENV_SHA"]


def dynamic_genesis(*, deltas=None):
    """Same labelled Tiny6 native fixture with explicit manual update intent."""
    schedule = f.schedule(f.wind(), f.rain(), f.wind(), f.rain())
    native = lv._native()
    bare = lv.CellReplacement((0, 0), 0.0, 0.0, native.DEFAULT_VEGETATION_POROSITY,
        native.DEFAULT_VEGETATION_DRAG_COEFFICIENT, 0.0)
    raised = lv.CellReplacement((0, 0), 0.25, 0.05, native.DEFAULT_VEGETATION_POROSITY,
        native.DEFAULT_VEGETATION_DRAG_COEFFICIENT, 0.125)
    future = dataclasses.replace(raised, cover_fraction=0.125, height_m=0.025,
        frontal_area_density=0.0625)
    if deltas is None:
        deltas = (lv.PlannedDelta("wind", 0, "authored-increase", 0, (raised,), "prescribed_increase"),
            lv.PlannedDelta("water", 0, "authored-decrease", 1, (bare,), "prescribed_decrease"),
            lv.PlannedDelta("prescribed", 0, "authored-future", 3, (future,), "prescribed_increase"))
    plan = lv.build_vegetation_plan(deltas, shape=(f.NY, f.NX), n_classes=len(f.CLASS_IDS),
        n_voxels=f.N_VOXELS, event_kinds=tuple(event.kind for event in schedule.events),
        schedule_sha256=schedule.digest)
    classes = f.grain_classes()
    fractions = np.array(np.broadcast_to(np.asarray(f.FRACTIONS, dtype=np.float64),
        (f.NY, f.NX, len(f.CLASS_IDS))))
    bed = db.build_dense_bed(geometry=f.geometry(classes), grain_classes=classes,
        mass_resolution_kg=float(f.DEFAULT_MASS_RESOLUTION_KG), n_voxels=f.N_VOXELS,
        initial_bed_elevation_m=f.chain_elevation(), initial_class_fractions=fractions,
        initial_available_fraction=np.full((f.NY, f.NX, len(f.CLASS_IDS)), f.AVAILABLE),
        initial_water_depth_m=np.zeros((f.NY, f.NX)),
        vegetation=f.empty_vegetation_state(f.NY, f.NX),
        threshold_modifier_state=f.neutral_threshold_modifier_state(f.NY, f.NX),
        topographic_commit_spec=f.TopographicCommitSpec(),
        avalanching_spec=f.AvalanchingSpec(enabled=False), active_mask=np.ones((f.NY, f.NX), dtype=bool),
        initial_time_s=0.0, limits=db.DenseBedLimits(host_memory_cap_bytes=1 << 30,
            max_voxels_per_column=64, stage_copy_max_bytes=1 << 24), vegetation_profile=plan)
    runner = dr.DenseRunner(bed, schedule, wind=dr.DenseWindInputs(f.wind_config(bed), f.empty_pool()),
        water=f.water_inputs(bed), control=f.runner_control(), limits=f.limits(), vegetation_plan=plan)
    return runner


def resources(runner, capsule=None):
    capsule = bridge.DenseBoundaryInputsV2.capture(runner._bed) if capsule is None else capsule
    record = runner._bed._dynamic.as_record()
    return bridge.shared_resources_v2(capsule, runner._vegetation_plan, record,
        metadata_sha256=record["metadata_sha256"], soil_identity=db._array_digest(runner._soil),
        accepted_run_copy_bytes=runner._vegetation_copied_bytes)


def save_native(path, runner, capsule, shared):
    caller = {"schema": boundary._NATIVE_CALLER_SCHEMA_V2, "source_identity": SOURCES,
        "environment_sha256": ENVIRONMENT, "static_sha256": runner.static_sha256}
    packed = bridge.pack_boundary_physical_v2(runner._bed, runner._pool, runner._wind.config,
        caller, inputs=capsule, shared_resources=shared, max_bytes=64 << 20)
    cp.save_exact_physical_checkpoint(path, packed.physical, runner._wind.config,
        packed.identity, max_bytes=64 << 20)
    return packed.identity


def reference(path, fmt, identity):
    if fmt == oi.SCHEMA_V2:
        filename = "manifest.json"
    elif fmt == cp.SCHEMA:
        filename = "checkpoint.json"
    else:
        raise AssertionError("test reference format is closed")
    return {"format": fmt, "relative_path": Path(path).name,
        "manifest_file_sha256": hashlib.sha256((Path(path) / filename).read_bytes()).hexdigest(),
        "identity": identity}


def signature(value):
    if type(value) is np.ndarray:
        return value.dtype.str, value.shape, value.tobytes(), value.flags.writeable
    if dataclasses.is_dataclass(value):
        return tuple((field.name, signature(getattr(value, field.name))) for field in dataclasses.fields(value))
    if type(value) in (tuple, list):
        return tuple(signature(item) for item in value)
    if type(value) is dict:
        return tuple((key, signature(item)) for key, item in value.items())
    return value.hex() if type(value) is float else value


@pytest.fixture(scope="module")
def genesis():
    return dynamic_genesis()


def saved_genesis(tmp_path, runner):
    capsule = bridge.DenseBoundaryInputsV2.capture(runner._bed)
    shared = resources(runner, capsule)
    identity = save_native(tmp_path / "native", runner, capsule, shared)
    scope = bridge._new_dynamic_decode_scope(runner._vegetation_plan, capsule)
    request = bridge._make_dynamic_restore_request(scope, role="genesis",
        ledger_record=runner.ledger_record(), expected_metadata_sha256=shared["vegetation_metadata_sha256"])
    return capsule, shared, identity, scope, request


def load_genesis(tmp_path, runner, capsule, shared, identity, request):
    return bridge.load_boundary_physical_v2(tmp_path / "native", runner._wind.config,
        identity, inputs=capsule, request=request,
        genesis_soil_identity=shared["soil_identity"], max_bytes=64 << 20)


def test_genesis_load_restores_fifteen_field_owner_and_native_borrows(tmp_path, genesis):
    capsule, shared, identity, _, request = saved_genesis(tmp_path, genesis)
    before = signature(genesis._bed.state), signature(genesis._pool)
    out = load_genesis(tmp_path, genesis, capsule, shared, identity, request)
    assert set(vars(out.bed)) == bridge._BED_INSTANCE_V2
    assert out.bed.context is capsule.context
    assert out.bed._dynamic.plan is genesis._vegetation_plan
    assert out.bed._dynamic._native_borrow() is out.initial_ticket.vegetation
    assert out.bed._dynamic._native_borrow(initial=True) is out.initial_ticket.vegetation
    assert out.bed._dynamic.as_record() == shared["vegetation"]
    assert out.bed.identity_sha256() == genesis._bed.identity_sha256()
    assert before == (signature(genesis._bed.state), signature(genesis._pool))
    out.bed.verify_case_inputs()


def test_restore_calls_real_loader_once_without_initializer_or_delta_replay(tmp_path, genesis):
    capsule, shared, identity, _, request = saved_genesis(tmp_path, genesis)
    forbidden = AssertionError("restore replayed an operation")
    with (mock.patch.object(cp, "load_exact_physical_checkpoint", wraps=cp.load_exact_physical_checkpoint) as actual,
          mock.patch.object(db, "build_dense_bed", side_effect=forbidden),
          mock.patch.object(lv, "prepare_vegetation", side_effect=forbidden),
          mock.patch.object(lv.LandscapeVegetation, "from_native", side_effect=forbidden),
          mock.patch.object(dr, "_advance_wind_block", side_effect=forbidden),
          mock.patch.object(dr, "_finalize_wind_event", side_effect=forbidden),
          mock.patch.object(dr, "_complete_event", side_effect=forbidden)):
        load_genesis(tmp_path, genesis, capsule, shared, identity, request)
    assert actual.call_count == 1


def test_foreign_decode_ticket_refuses_before_second_native_load(tmp_path, genesis):
    capsule, shared, identity, _, request = saved_genesis(tmp_path, genesis)
    out = load_genesis(tmp_path, genesis, capsule, shared, identity, request)
    foreign = bridge._new_dynamic_decode_scope(genesis._vegetation_plan, capsule)
    with mock.patch.object(cp, "load_exact_physical_checkpoint", side_effect=AssertionError("early load")), pytest.raises(db.DenseBedError):
        bridge._make_dynamic_restore_request(foreign, role="current", ledger_record=genesis.ledger_record(),
            envelope_record=shared["vegetation"], expected_metadata_sha256=shared["vegetation_metadata_sha256"],
            initial_ticket=out.initial_ticket)


@pytest.mark.parametrize("change", ["metadata", "soil", "case", "limit", "exchange", "schema",
    "soil_shape_kind", "envelope_version_kind", "empty_cursor_kind"])
def test_malformed_controls_refuse_and_preserve_source_owner(tmp_path, genesis, change):
    capsule, shared, identity, _, request = saved_genesis(tmp_path, genesis)
    before = signature(genesis._bed.state), signature(genesis._pool), genesis._bed._dynamic.as_record()
    if change == "metadata":
        request = dataclasses.replace(request, expected_metadata_sha256="a" * 64)
    elif change == "soil":
        shared = {**shared, "soil_identity": {"array": {**shared["soil_identity"]["array"], "dtype": "<f4"}}}
    elif change == "case":
        identity = {**identity, "case_sha256": "a" * 64}
    elif change == "soil_shape_kind":
        forged = {"array": {**shared["soil_identity"]["array"], "shape": [f.NY, True]}}
        with pytest.raises(db.DenseBedError, match="current soil digest layout differs"):
            bridge._soil_identity_v2(forged, (f.NY, f.NX))
        return
    elif change in ("envelope_version_kind", "empty_cursor_kind"):
        if change == "envelope_version_kind":
            shared["vegetation"]["version"] = False
            message = "dynamic resource identities differ"
        else:
            shared["vegetation"]["cursor"]["next_event_index"] = False
            message = "shared resources do not reproduce the actual bed envelope"
        # Omit the optional prior resource hash so it cannot mask the targeted
        # type-exact metadata guard; the real packer must derive the new seal.
        caller = {key: value for key, value in identity["caller"].items()
            if key != "shared_resource_record_sha256"}
        with pytest.raises(db.DenseBedError, match=message):
            bridge.pack_boundary_physical_v2(genesis._bed, genesis._pool, genesis._wind.config,
                caller, inputs=capsule, shared_resources=shared, max_bytes=64 << 20)
        assert before == (signature(genesis._bed.state), signature(genesis._pool), genesis._bed._dynamic.as_record())
        return
    elif change == "limit":
        with pytest.raises(db.DenseBedError):
            bridge.load_boundary_physical_v2(tmp_path / "native", genesis._wind.config, identity,
                inputs=capsule, request=request, genesis_soil_identity=shared["soil_identity"], max_bytes=1)
        assert before == (signature(genesis._bed.state), signature(genesis._pool), genesis._bed._dynamic.as_record())
        return
    elif change == "exchange":
        with pytest.raises(db.DenseBedError):
            bridge._resources_v2({**shared, "water_exchange": [["invented", 0.0]]}, capsule,
                genesis._vegetation_plan, cp, 64 << 20)
        return
    else:
        identity = {**identity, "schema": bridge.SCHEMA}
    with pytest.raises((db.DenseBedError, cp.PhysicalCheckpointError, lv.VegetationError)):
        load_genesis(tmp_path, genesis, capsule, shared, identity, request)
    assert before == (signature(genesis._bed.state), signature(genesis._pool), genesis._bed._dynamic.as_record())


def test_current_requires_real_genesis_ticket(genesis):
    capsule = bridge.DenseBoundaryInputsV2.capture(genesis._bed)
    shared = resources(genesis, capsule)
    scope = bridge._new_dynamic_decode_scope(genesis._vegetation_plan, capsule)
    with pytest.raises(cp.PhysicalCheckpointError, match=r"^unsupported exact type/field inventory: _InitialVegetationTicket$"):
        bridge._make_dynamic_restore_request(scope, role="current", ledger_record=ClockLedger(0.0).as_record(),
            envelope_record=shared["vegetation"], expected_metadata_sha256=shared["vegetation_metadata_sha256"])


def test_v1_capsule_capture_still_refuses_dynamic_owner(genesis):
    with pytest.raises(db.DenseBedError):
        bridge.DenseBoundaryInputs.capture(genesis._bed)
