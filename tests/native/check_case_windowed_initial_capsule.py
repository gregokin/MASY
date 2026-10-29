"""Root-only HOST initial imported C64 capsule; no placement or events.

Compile and load are separate genuine subjects. Load precedes FIRST adoption.
The existing C64 wet/request fixture supplies labelled interface parameters,
not calibration or any Root budget/authority. No GPU fixture is consumed.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import json
import os
import zipfile
from types import FunctionType, MethodType

import numpy as np
import pytest

import check_case_device_c64 as interface
from check_case_device_c64 import (
    f, wet, CaseWetClass, CaseWaterSpec, CaseWaterCopyLimits,
    DenseProductionWindProfile, build_dense_production_wind_config,
    _dimension_limits, DeviceWindowedK1DimensionProfile,
    DeviceWindowedC64BedLimits, DeviceResourceLimits, CaseDeviceC64PlacementLimits,
    committed_elevation_m,
)
from masy import case_windowed as capsule
from masy import case_workflow as cw
from masy import dense_bed as db
from masy.case_device_c64 import device_c64_request_binding, _bytes as request_bytes
from masy.case_windowed import (
    CaseWindowedError, CaseWindowedLimits, adopt_windowed_case,
    prepare_case_windowed_water, save_initial_windowed_case, reopen_initial_windowed_case,
)
from masy.case_workflow import CaseInputLimits, resolve_case_inputs

IO = CaseInputLimits(1 << 20, 256, 8 << 20, 16 << 20, 16 << 20, 128 << 20)
LIMITS = CaseWindowedLimits(64, 1, 20, 128 << 20, 4 << 20, 64 << 20)
MAX_CAPSULE = 4 << 20


def _loaded(root):
    case = interface._package(root / "input_case")
    compiled = resolve_case_inputs(case, mode="compile", input_limits=IO)
    compile_record = capsule._bytes(cw._input_record(compiled.resolved, compiled.compiled, compiled.file_pins), LIMITS.max_metadata_bytes)
    loaded = resolve_case_inputs(case, mode="load", input_limits=IO)
    load_record = capsule._bytes(cw._input_record(loaded.resolved, loaded.compiled, loaded.file_pins), LIMITS.max_metadata_bytes)
    assert compiled.compiled.transform_logs and loaded.compiled.transform_logs == ()
    assert type(compiled.compiled.import_sources[0]) is not dict
    assert type(loaded.compiled.import_sources[0]) is dict
    assert compiled.identity_sha256 != loaded.identity_sha256
    assert compile_record != load_record  # Subjects are retained separately, never retagged.
    host = adopt_windowed_case(loaded, store_dir=root / "deep", datum_offset_m=0.0,
        window_voxels=3, chunk_columns=64, rows_per_band=2, with_prefix_packets=True, limits=LIMITS)
    return loaded, host, compile_record, load_record


def _reopen(inputs, directory, sha):
    return reopen_initial_windowed_case(inputs, capsule_dir=directory,
        expected_capsule_sha256=sha, limits=LIMITS, max_capsule_bytes=MAX_CAPSULE)


def _compact(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    assert type(value) not in (cw.CaseInputs, cw.AdoptedCase, db.DenseBed, FunctionType, MethodType)
    assert type(value).__name__ != "CompiledCase"
    if type(value) is np.ndarray:
        assert value.shape != (8, 8, 20, 1)
        assert not value.flags.writeable
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _compact(getattr(value, field.name), seen)
    elif type(value) in (tuple, list):
        for item in value:
            _compact(item, seen)
    elif type(value) is dict:
        for item in value.values():
            _compact(item, seen)


def _water_request(host, root):
    # Same genuine constructors/explicit fields as the accepted imported HOST
    # fixture, after receiving this test's independently owned adopted state.
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
    return prepared, kwargs


def test_loaded_initial_capsule_reopens_exact_owner(tmp_path):
    from maple.surface.voxels.windowed_inventory import gather_full_columns
    original, host, compile_raw, load_raw = _loaded(tmp_path)
    original_held = capsule._bytes(capsule._held_record(host), LIMITS.max_metadata_bytes)
    flags = [(array, bool(array.flags.writeable)) for _path, array in db._require_host_arrays(
        (original.resolved, original.compiled), "caller")]
    saved = save_initial_windowed_case(original, host, capsule_dir=tmp_path / "initial", max_capsule_bytes=MAX_CAPSULE)
    fresh = resolve_case_inputs(original.case_dir, mode="load", input_limits=IO)
    assert fresh is not original and fresh.compiled is not original.compiled
    reopened, receipt = _reopen(fresh, tmp_path / "initial", saved["capsule"]["sha256"])
    assert reopened is not host and reopened._record_bytes == host._record_bytes
    assert capsule._bytes(capsule._held_record(reopened), LIMITS.max_metadata_bytes) == original_held
    assert capsule._bytes(cw._input_record(fresh.resolved, fresh.compiled, fresh.file_pins), LIMITS.max_metadata_bytes) == load_raw
    assert load_raw != compile_raw
    assert receipt["current_known_plan"]["current_known_total_bytes"] <= LIMITS.max_known_host_bytes
    assert receipt["actual_adapter_io"]["hash_calls"] > 0
    assert receipt["carried_historical_fields"] == ["native_committed_regrouping", "construction_plan", "storage_choices.rows_per_band"]
    assert set(path.name for path in (tmp_path / "initial").iterdir()) == {*capsule._INITIAL_MEMBERS, "CAPSULE.json"}
    col, old = reopened.state.voxel_column, host.state.voxel_column
    assert col.deep_store is not old.deep_store and col.deep_store.store_dir == old.deep_store.store_dir
    assert col.generation == old.generation == 0 and col.manifest_digest == old.manifest_digest
    assert col.deep_store.mapping_cache_chunks == old.deep_store.mapping_cache_chunks == 8
    assert col.pending_deep is None and col.window_cache is None
    for field in ("window_kg", "base", "deep_by_class_kg", "deep_total_kg", "deep_room_kg"):
        assert getattr(col, field).tobytes() == getattr(old, field).tobytes()
    assert col.height_packet.packet.tobytes() == old.height_packet.packet.tobytes()
    assert col.height_packet.binding.encoded_base.tobytes() == old.height_packet.binding.encoded_base.tobytes()
    ids = np.arange(64, dtype=np.int64)
    assert gather_full_columns(col, ids, max_columns=64).tobytes() == gather_full_columns(old, ids, max_columns=64).tobytes()
    assert reopened.context.geometry is reopened.config.geometry
    assert reopened.context.grain_classes is reopened.config.grain_classes
    assert reopened.context.topographic_commit_spec is reopened.config.topographic_commit
    assert reopened.context.avalanching_spec is reopened.config.avalanching
    statics = reopened.static_inputs["compiled"]
    assert statics["coordinates"] is statics["topography_result"].coordinates
    assert capsule._initial_aliases(reopened, LIMITS.max_metadata_bytes) == capsule._initial_aliases(host, LIMITS.max_metadata_bytes)
    _compact(reopened)
    assert not col.height_packet.packet.flags.writeable and not col.height_packet.binding.encoded_base.flags.writeable
    assert all(bool(array.flags.writeable) == flag for array, flag in flags)
    assert capsule._bytes(cw._input_record(original.resolved, original.compiled, original.file_pins), LIMITS.max_metadata_bytes) == load_raw


def test_loaded_initial_capsule_reprepares_exact_water_and_request(tmp_path):
    original, host, _compile_raw, _load_raw = _loaded(tmp_path)
    water, kwargs = _water_request(host, tmp_path)
    raw = tuple(getattr(water, name) for name in ("_spec_bytes", "_water_bytes", "_control_bytes", "_record_bytes"))
    binding = request_bytes(device_c64_request_binding(host, water, **kwargs), kwargs["placement_limits"].max_metadata_bytes)
    saved = save_initial_windowed_case(original, host, capsule_dir=tmp_path / "initial", max_capsule_bytes=MAX_CAPSULE)
    fresh = resolve_case_inputs(original.case_dir, mode="load", input_limits=IO)
    reopened, _receipt = _reopen(fresh, tmp_path / "initial", saved["capsule"]["sha256"])
    prepared, fresh_kwargs = _water_request(reopened, tmp_path)
    fresh_raw = tuple(getattr(prepared, name) for name in ("_spec_bytes", "_water_bytes", "_control_bytes", "_record_bytes"))
    assert fresh_raw == raw
    assert prepared.water is not water.water and prepared.control is not water.control
    assert prepared.water.vegetation_cover_fraction is reopened.vegetation.raster.cover_fraction
    actual = device_c64_request_binding(reopened, prepared, **fresh_kwargs)
    assert request_bytes(actual, fresh_kwargs["placement_limits"].max_metadata_bytes) == binding
    assert actual["bridge_limits"] == device_c64_request_binding(host, water, **kwargs)["bridge_limits"]
    assert reopened.identity_sha256 == host.identity_sha256
    assert prepared.identity_sha256 == water.identity_sha256


def _replace_file(path, raw):
    os.chmod(path, 0o644)  # This test's one explicit generated file only.
    path.write_bytes(raw)
    os.chmod(path, 0o444)


def _retag(directory, name, raw):
    _replace_file(directory / name, raw)
    data = json.loads((directory / "CAPSULE.json").read_bytes())
    data["members"][name] = {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    final = capsule._bytes(data, LIMITS.max_metadata_bytes)
    _replace_file(directory / "CAPSULE.json", final)
    return hashlib.sha256(final).hexdigest()


def test_initial_capsule_refusals_preserve_original_and_publish_last(tmp_path, monkeypatch):
    from maple.surface.voxels import windowed_snapshot
    original, host, _compile_raw, load_raw = _loaded(tmp_path)
    before = host._record_bytes
    with pytest.raises(CaseWindowedError, match="capsule_file_limit"):
        save_initial_windowed_case(original, host, capsule_dir=tmp_path / "tiny", max_capsule_bytes=1)
    assert not (tmp_path / "tiny").exists()
    # Values alone do not prove aliases. Breaking this real static alias must
    # prevent final publication although the original value image still agrees.
    broken = copy.deepcopy(host.static_inputs)
    broken["compiled"]["coordinates"] = copy.deepcopy(broken["compiled"]["coordinates"])
    capsule._freeze(broken)
    changed = dataclasses.replace(host, static_inputs=broken)
    assert capsule._held_record(changed) == capsule._held_record(host)
    with pytest.raises(CaseWindowedError, match="capsule_identity"):
        save_initial_windowed_case(original, changed, capsule_dir=tmp_path / "bad_alias", max_capsule_bytes=MAX_CAPSULE)
    assert not (tmp_path / "bad_alias/CAPSULE.json").exists()
    saved = save_initial_windowed_case(original, host, capsule_dir=tmp_path / "initial", max_capsule_bytes=MAX_CAPSULE)
    directory = tmp_path / "initial"
    sha = saved["capsule"]["sha256"]
    fresh = resolve_case_inputs(original.case_dir, mode="load", input_limits=IO)
    manifest = host.state.voxel_column.deep_store.store_dir / "versions" / (host.state.voxel_column.manifest_digest + ".json")
    manifest_raw = manifest.read_bytes()
    _replace_file(manifest, b"x" * (LIMITS.max_metadata_bytes + 1))
    with pytest.raises(CaseWindowedError, match="capsule_file"):
        _reopen(fresh, directory, sha)
    _replace_file(manifest, manifest_raw)
    moved = manifest.with_name("test-owned-original.json")
    manifest.rename(moved)
    manifest.symlink_to(moved)
    with pytest.raises(CaseWindowedError, match="capsule_file"):
        _reopen(fresh, directory, sha)
    manifest.unlink(); moved.rename(manifest)  # Restore this exact test-owned original, not a store rollback.
    final_raw = (directory / "CAPSULE.json").read_bytes()
    data = json.loads(final_raw)
    data["source_refs"][0]["sha256"] = "0" * 64
    _replace_file(directory / "CAPSULE.json", capsule._bytes(data, LIMITS.max_metadata_bytes))
    with pytest.raises(CaseWindowedError, match="capsule_source"):
        _reopen(fresh, directory, hashlib.sha256((directory / "CAPSULE.json").read_bytes()).hexdigest())
    _replace_file(directory / "CAPSULE.json", final_raw)
    native_raw = (directory / "native.npz").read_bytes()
    altered = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(native_raw)) as archive, zipfile.ZipFile(altered, "w") as output:
        for info in archive.infolist():
            content = archive.read(info.filename)
            if info.filename == "window_kg.npy":
                content = content.replace(b"'<f8'", b"'<i8'", 1)
            output.writestr(info.filename, content, compress_type=zipfile.ZIP_STORED)
    bad_sha = _retag(directory, "native.npz", altered.getvalue())
    with pytest.raises(CaseWindowedError, match="capsule_header"):
        _reopen(fresh, directory, bad_sha)
    _replace_file(directory / "native.npz", native_raw)
    _replace_file(directory / "CAPSULE.json", final_raw)
    genuine = windowed_snapshot.load_windowed_state_snapshot
    def changed_after_read(path, **kwargs):
        result = genuine(path, **kwargs)
        info = path.stat()
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000))
        return result
    with monkeypatch.context() as patch:
        patch.setattr(windowed_snapshot, "load_windowed_state_snapshot", changed_after_read)
        with pytest.raises(CaseWindowedError, match="capsule_changed"):
            _reopen(fresh, directory, sha)
    def sync_refusal(_directory):
        raise OSError("test-owned directory-fsync failure")
    with monkeypatch.context() as patch:
        patch.setattr(capsule, "_initial_sync", sync_refusal)
        with pytest.raises(OSError, match="directory-fsync failure"):
            save_initial_windowed_case(original, host, capsule_dir=tmp_path / "unsynced", max_capsule_bytes=MAX_CAPSULE)
    assert not (tmp_path / "unsynced/CAPSULE.json").exists()
    assert (tmp_path / "unsynced/native.npz").is_file()  # Failure evidence is retained.
    assert host._record_bytes == before
    assert capsule._bytes(cw._input_record(original.resolved, original.compiled, original.file_pins), LIMITS.max_metadata_bytes) == load_raw
