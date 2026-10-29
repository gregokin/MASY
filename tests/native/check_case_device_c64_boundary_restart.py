"""UNISSUED imported C64 Native gate and four SERIAL Root-owned GPU roles.

This module issues no grants, starts no child and has no factory/budget/loader
fallback. Protected fixtures defer ALL setup until the selected test calls
them inside its original600s. Actual model/proof/oracle files stay Root-local.
The proposed1203 save/reopen API must be accepted/composed before execution.
"""
from __future__ import annotations

import copy
import dataclasses
import enum
import hashlib
import json
import math
import os
from contextlib import closing
from pathlib import Path
from types import FunctionType, ModuleType

import numpy as np
import pytest
import check_case_device_c64 as imported
import fixture_windowed_device_c64_alternation as f
from tests.windowed_wind_gpu.test_device_k1_dimension import _dimension_limits
from tests.windowed_wind_gpu.test_device_reduced_step import cupy  # noqa: F401
from maple.core.backend import to_host
from maple.core.types.topographic_commit import committed_elevation_m
from maple.surface.active_layer.bed_exchange_device import DeviceWindowedK1DimensionProfile
from maple.surface.topographic_commit.windowed_device import _bed_leaves
from maple.surface.voxels.windowed_inventory import gather_full_columns
from maple_syrup import sediment_physics as wet
from maple_syrup import windowed_shared_bed_device as shared

from masy import case_device_c64 as producer
from masy import case_windowed as windowed
from masy import case_workflow as caseio
from masy import dense_boundary_store as store
from masy import windowed_device_c64_checkpoint as checkpoint
from masy.case_device_c64 import CaseDeviceC64PlacementLimits
from masy.case_water import CaseWaterCopyLimits, CaseWaterSpec, CaseWetClass
from masy.case_windowed import CaseWindowedLimits, prepare_case_windowed_water
from masy.case_workflow import CaseInputLimits, resolve_case_inputs
from masy.dense_device_resources import STATIC_NAMES, DeviceResourceLimits
from masy.dense_wind_profile import DenseProductionWindProfile, build_dense_production_wind_config
from masy.windowed_device_c64_bed import DeviceWindowedC64BedLimits
from masy.windowed_device_c64_runner import DeviceWindowedC64Runner, _combine_water_counters

pytestmark = pytest.mark.gpu
ROLE_SCHEMA = "masy.imported_device_c64_restart_test_role.v1"
ROLES = ("continuous", "write-W1", "recover-R1", "recover-finish")
REFERENCE_SCHEMA = "masy.imported_device_c64_restart_oracles.v1"
CONTROL_SCHEMA = "masy.imported_device_c64_restart_control.v1"
PROOF_SCHEMA = "masy.initial_windowed_original_input_proof.v1"
RECORDS = ("compile_case_record", "loaded_case_record", "host_record", "water_spec",
           "water_inputs", "water_control", "water_record", "request_projection", "public_request")
SUBJECT = ("compiled_case_directory", "store_directory", "native_case_identity_sha256",
           "windowed_identity_sha256", "water_identity_sha256", "initial_manifest_digest",
           "source_manifest_sha256")
_CAP = 4 << 20
_PUBLIC_REQUEST_CAP = 2 << 20

def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _role(descriptor):
    """Validate data only; Root's fixture owns real authority and UUID checks."""
    checkpoint._keys(descriptor, ("schema", "role", "shared_root",
        "expected_reference_index_sha256", "expected_store_file_sha256",
        "expected_control_sha256", "storage_contract"), "protected role descriptor")
    assert descriptor["schema"] == ROLE_SCHEMA and descriptor["role"] in ROLES
    root = Path(checkpoint._path(descriptor["shared_root"]))
    held = checkpoint.oi._open_dir(root)
    os.close(held)
    contract = store.StorageContract(**descriptor["storage_contract"])
    for name in ("expected_reference_index_sha256", "expected_store_file_sha256", "expected_control_sha256"):
        needed = descriptor["role"] != "continuous" if name == "expected_reference_index_sha256" else descriptor["role"].startswith("recover-")
        if needed:
            checkpoint._digest(descriptor[name], name)
        else:
            assert descriptor[name] is None, "unissued future evidence must stay absent"
    return descriptor["role"], root, contract


def _write(path, value):
    raw = checkpoint._json(value)
    assert len(raw) <= _CAP
    with path.open("xb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    path.chmod(0o400)
    fd = checkpoint.oi._open_dir(path.parent)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(size=len(raw), sha256=_digest(raw))


def _read(path, expected):
    raw = checkpoint.codec._read_file(path, _CAP)
    assert _digest(raw) == expected, "Root-pinned immutable oracle/control bytes changed"
    return checkpoint.oi._strict_json(raw, checkpoint._DEFAULT.image)


def _scientific(value, cp):
    """Finite test-only byte signatures, independent of codec object aliases.

    No value/address of a live handle is encoded. Inactive array NaN bytes are
    preserved. Array hashes cover the complete selected tiny oracle, not a
    surface sample; the product restore never uses this full-column gather.
    """
    used, nodes, rows = [0], [0], []
    active = set()
    def walk(item, path=(), depth=0):
        nodes[0] += 1
        assert nodes[0] <= 100000 and depth <= 24
        kind = type(item)
        def emit(**row):
            rows.append(dict(path=list(path), **row))
        if kind in (np.ndarray, cp.ndarray):
            assert item.dtype.kind in "fbiu" and item.ndim <= 4
            used[0] += int(item.nbytes)
            assert used[0] <= 64 << 20
            host = np.ascontiguousarray(to_host(item))
            emit(kind="array", dtype=host.dtype.str, shape=list(host.shape),
                 bytes=host.nbytes, sha256=_digest(host.tobytes(order="C")))
            return
        if item is None or kind in (bool, str, int):
            assert kind is not int or item.bit_length() <= 128
            assert kind is not str or len(item) <= 4096
            emit(kind=kind.__name__, value=item); return
        if kind is float:
            assert math.isfinite(item)
            emit(kind="float", hex=item.hex()); return
        if isinstance(item, np.generic):
            emit(kind="numpy_scalar", dtype=item.dtype.str)
            walk(item.item(), (*path, ["value", "scalar"]), depth + 1); return
        if isinstance(item, enum.Enum):
            emit(kind="enum", name=kind.__module__ + ":" + kind.__qualname__)
            walk(item.value, (*path, ["value", "enum"]), depth + 1); return
        if kind is ModuleType:
            assert item in (np, cp)
            emit(kind="backend", value="numpy" if item is np else "cupy"); return
        assert id(item) not in active, "unexpected cycle in scientific state"
        active.add(id(item))
        try:
            if dataclasses.is_dataclass(item) and not isinstance(item, type):
                fields = dataclasses.fields(item)
                emit(kind=kind.__module__ + ":" + kind.__qualname__, fields=[field.name for field in fields])
                for field in fields:
                    walk(getattr(item, field.name), (*path, ["field", field.name]), depth + 1)
                return
            if kind in (tuple, list):
                emit(kind=kind.__name__, length=len(item))
                for index, x in enumerate(item):
                    walk(x, (*path, ["index", index]), depth + 1)
                return
            assert kind is dict and all(type(key) is str for key in item)
            emit(kind="dict", keys=sorted(item))
            for key in sorted(item):
                walk(item[key], (*path, ["key", key]), depth + 1)
        finally:
            active.remove(id(item))
    walk(value)
    # Flat rows keep the oracle JSON below the original metadata depth cap;
    # nested scientific holders do not multiply JSON wrapper nesting.
    return dict(layout="flat_Source_fields_v1", rows=rows, numeric_bytes_hashed=used[0])


def _water_and_request(host, root):
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


def _initial_apis():
    """Fail closed BEFORE setup until the real owning1206 API is composed."""
    ns = vars(windowed)
    result = []
    for name in ("save_initial_windowed_case", "reopen_initial_windowed_case"):
        function = dict.get(ns, name)
        assert (type(function) is FunctionType and function.__globals__ is ns
            and function.__code__.co_filename == windowed.__file__
            and function.__code__.co_name == name), "accepted1206 initial capsule API is required"
        result.append(function)
    return tuple(result)


def _raw_write(path, raw):
    assert type(raw) is bytes and 0 < len(raw) <= _CAP
    with path.open("xb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    path.chmod(0o400)
    fd = checkpoint.oi._open_dir(path.parent)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(path), size=len(raw), sha256=_digest(raw))


def _capture_records(host, water, kwargs):
    """Verify/capture/reverify actual owning bytes; no caller digest input."""
    assert type(host) is windowed.CaseWindowedInputs
    assert type(water) is windowed.PreparedCaseWindowedWater
    host.verify(); water.verify(host)
    binding = producer.device_c64_request_binding(host, water, **kwargs)
    raw = dict(host_record=host._record_bytes, water_spec=water._spec_bytes,
        water_inputs=water._water_bytes, water_control=water._control_bytes,
        water_record=water._record_bytes,
        request_projection=producer._bytes(binding, kwargs["placement_limits"].max_metadata_bytes))
    assert all(type(value) is bytes and 0 < len(value) <= _CAP for value in raw.values())
    assert _digest(raw["host_record"]) == host.identity_sha256
    assert _digest(raw["water_record"]) == water.identity_sha256
    host.verify(); water.verify(host)
    again = (host._record_bytes, water._spec_bytes, water._water_bytes,
        water._control_bytes, water._record_bytes,
        producer._bytes(producer.device_c64_request_binding(host, water, **kwargs),
                        kwargs["placement_limits"].max_metadata_bytes))
    assert tuple(raw.values()) == again
    return binding, raw


def _proof_record(raw):
    assert type(raw) is bytes and 0 < len(raw) <= _CAP
    return dict(size=len(raw), sha256=_digest(raw), hex=raw.hex())


def _proof_bytes(entry, name):
    checkpoint._keys(entry, ("size", "sha256", "hex"), "original raw proof")
    maximum = _PUBLIC_REQUEST_CAP if name == "public_request" else _CAP
    assert type(entry["size"]) is int and 0 < entry["size"] <= maximum
    checkpoint._digest(entry["sha256"], "original raw digest")
    text = entry["hex"]
    assert type(text) is str and len(text) == 2*entry["size"]
    assert all(char in "0123456789abcdef" for char in text)
    raw = bytes.fromhex(text)
    assert len(raw) == entry["size"] and _digest(raw) == entry["sha256"]
    return raw


def _proof(root, reference):
    """Fixed nine-row Root metadata only; authenticate BEFORE parse/hex."""
    checkpoint._keys(reference, ("path", "size", "sha256"), "Root original proof ref")
    assert reference["path"] == str(root / "original-input-proof.json")
    checkpoint._check_file(reference, _CAP)
    raw = checkpoint.codec._read_file(reference["path"], _CAP)
    assert len(raw) == reference["size"] and _digest(raw) == reference["sha256"]
    # This selected proof is narrower than the design's64MiB IO ceiling.
    # Its fixed JSON has only dictionaries, strings and positive size ints.
    # Budget tokens/depth/characters BEFORE JSON pending expansion; reject
    # every other lexical family, then let the real parser validate syntax.
    assert raw.isascii()
    index = depth = tokens = chars = 0
    while index < len(raw):
        char = raw[index]
        if char == 34:
            start = index
            index += 1
            while index < len(raw):
                if raw[index] == 92:
                    index += 2
                elif raw[index] == 34:
                    index += 1
                    break
                else:
                    index += 1
            else:
                raise AssertionError("unterminated original proof string")
            chars += index-start
            tokens += 1
        elif char == 123:
            depth += 1; tokens += 1; index += 1
        elif char == 125:
            depth -= 1; index += 1
        elif 48 <= char <= 57:
            start = index
            while index < len(raw) and 48 <= raw[index] <= 57:
                index += 1
            assert index-start <= 19
            tokens += 1
        else:
            assert char in (9, 10, 13, 32, 44, 58), "closed original proof JSON tokens"
            index += 1
        assert 0 <= depth <= 3 and tokens <= 128 and chars <= _CAP
    assert depth == 0
    def pairs(items):
        assert len(items) <= 9
        value = {}
        for key, item in items:
            assert type(key) is str and key not in value
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=pairs)
    assert checkpoint._json(value) == raw
    checkpoint._keys(value, ("schema", "subject", "originals"), "original proof")
    assert value["schema"] == PROOF_SCHEMA
    checkpoint._keys(value["subject"], SUBJECT, "original subject")
    for key in SUBJECT[:2]:
        assert type(value["subject"][key]) is str
        checkpoint._path(value["subject"][key])
    for key in SUBJECT[2:]:
        checkpoint._digest(value["subject"][key], key)
    checkpoint._keys(value["originals"], RECORDS, "nine original owning byte records")
    for name in RECORDS:
        row = value["originals"][name]
        checkpoint._keys(row, ("size", "sha256", "hex"), "original row before hex allocation")
        maximum = _PUBLIC_REQUEST_CAP if name == "public_request" else _CAP
        assert type(row["size"]) is int and 0 < row["size"] <= maximum
    assert sum(value["originals"][name]["size"] for name in RECORDS) <= _CAP
    for name in RECORDS:
        _proof_bytes(value["originals"][name], name)
    checkpoint._check_file(reference, _CAP)
    return value


def _public_request(root, reference):
    """Root provides the real caller file FULLref; never derive its bytes."""
    checkpoint._keys(reference, ("path", "size", "sha256"), "actual public request ref")
    assert Path(reference["path"]).is_relative_to(root.parent)
    checkpoint._check_file(reference, _PUBLIC_REQUEST_CAP)
    raw = checkpoint.codec._read_file(reference["path"], _PUBLIC_REQUEST_CAP)
    assert len(raw) == reference["size"] and _digest(raw) == reference["sha256"]
    checkpoint._check_file(reference, _PUBLIC_REQUEST_CAP)
    return raw


def _subject(inputs, host, water, source_manifest_sha256):
    checkpoint._digest(source_manifest_sha256, "Root actual Source manifest")
    return dict(compiled_case_directory=str(inputs.case_dir),
        store_directory=str(host.state.voxel_column.deep_store.store_dir),
        native_case_identity_sha256=inputs.identity_sha256,
        windowed_identity_sha256=host.identity_sha256, water_identity_sha256=water.identity_sha256,
        initial_manifest_digest=host.state.voxel_column.manifest_digest,
        source_manifest_sha256=source_manifest_sha256)


def _prepare_initial_case(root, *, mode, issuer, public_request_reference,
        source_manifest_sha256, expected_capsule_sha256=None, original_proof_reference=None):
    """Root's deferred fixture calls real setup ONCE under its same600s.

    Root authenticates its original INDEX FULLref BEFORE parsing/passing the
    original proof FULLref. Root supplies genuine public request bytes/ref and
    actual Source manifest; no derived request/caller-digest fallback exists.
    """
    save, reopen = _initial_apis()
    assert mode in ("create", "reopen") and callable(issuer)
    root = Path(checkpoint._path(str(root)))
    public_raw = _public_request(root, public_request_reference)
    checkpoint._digest(source_manifest_sha256, "Root actual Source manifest")
    io = CaseInputLimits(1 << 20, 256, 8 << 20, 16 << 20, 16 << 20, 128 << 20)
    limits = CaseWindowedLimits(64, 1, 20, 128 << 20, 4 << 20, 64 << 20)
    if mode == "create":
        assert expected_capsule_sha256 is original_proof_reference is None
        root.mkdir()
        compiled = resolve_case_inputs(imported._package(root / "input_case"), mode="compile", input_limits=io)
        compiled.verify()
        compile_raw = windowed._bytes(caseio._input_record(
            compiled.resolved, compiled.compiled, compiled.file_pins), limits.max_metadata_bytes)
        _raw_write(root / "compile-input-record.json", compile_raw)
        # Genuine LOAD, not the prior compiler owner, defines first adoption.
        inputs = resolve_case_inputs(root / "input_case", mode="load", input_limits=io)
        inputs.verify()
        loaded_raw = windowed._bytes(caseio._input_record(
            inputs.resolved, inputs.compiled, inputs.file_pins), limits.max_metadata_bytes)
        _raw_write(root / "loaded-input-record.json", loaded_raw)
        host = windowed.adopt_windowed_case(inputs, store_dir=root / "deep", datum_offset_m=0.,
            window_voxels=3, chunk_columns=64, rows_per_band=2, with_prefix_packets=True, limits=limits)
        receipt = save(inputs, host, capsule_dir=root / "initial-capsule", max_capsule_bytes=64 << 20)
        assert type(receipt) is dict
        _write(root / "initial-save-receipt.json", receipt)
    else:
        checkpoint._digest(expected_capsule_sha256, "Root expected initial capsule")
        original = _proof(root, original_proof_reference)
        capsule_ref = checkpoint._file_row(root / "initial-capsule" / "CAPSULE.json", _CAP)
        assert capsule_ref["sha256"] == expected_capsule_sha256
        inputs = resolve_case_inputs(root / "input_case", mode="load", input_limits=io)
        inputs.verify()
        loaded_raw = windowed._bytes(caseio._input_record(
            inputs.resolved, inputs.compiled, inputs.file_pins), limits.max_metadata_bytes)
        _raw_write(root / "writer-loaded-input-record.json", loaded_raw)
        # This prior compiler observation is carried historical evidence, not
        # a newly executed compiler or a relabelled load-mode observation.
        compile_raw = _proof_bytes(original["originals"]["compile_case_record"], "compile_case_record")
        host, receipt = reopen(inputs, capsule_dir=root / "initial-capsule",
            expected_capsule_sha256=expected_capsule_sha256, limits=limits, max_capsule_bytes=64 << 20)
        assert type(receipt) is dict
        _write(root / "initial-reopen-receipt.json", receipt)
    host, water, kwargs = _water_and_request(host, root)
    binding, actual = _capture_records(host, water, kwargs)
    records = dict(compile_case_record=compile_raw, loaded_case_record=loaded_raw,
        **actual, public_request=public_raw)
    assert set(records) == set(RECORDS)
    subject = _subject(inputs, host, water, source_manifest_sha256)
    # Capsule's actual original loaded row is verbatim, not a saved-dict oracle.
    capsule_input = checkpoint.codec._read_file(root / "initial-capsule" / "case-input-record.json", _CAP)
    assert capsule_input == loaded_raw
    if mode == "create":
        original = dict(schema=PROOF_SCHEMA, subject=subject,
            originals={name: _proof_record(records[name]) for name in RECORDS})
        _write(root / "original-input-proof.json", original)
    else:
        assert original["subject"] == subject
        assert all(records[name] == _proof_bytes(original["originals"][name], name) for name in RECORDS)
    proof_ref = checkpoint._file_row(root / "original-input-proof.json", _CAP)
    _proof(root, proof_ref)
    # Root issuer independently checks actual owners/projection/public file,
    # authenticates/publishes its original index BEFORE returning authority.
    budget = issuer(host, water, request_kwargs=kwargs, original_proof_reference=proof_ref,
        public_request_reference=public_request_reference)
    assert type(budget) is shared.DeviceWaterC64CaseBudget
    budget.validated()
    assert budget.source_manifest_sha256 == source_manifest_sha256
    assert budget.case_request_binding_sha256 == _digest(records["request_projection"])
    binding2, actual2 = _capture_records(host, water, kwargs)
    assert binding2 == binding and actual2 == actual
    assert _public_request(root, public_request_reference) == public_raw
    return producer.prepare_device_c64_case(host, water, qualification_budget=budget,
        request_binding=binding, **kwargs)




def _prepared(value, origin):
    assert type(value) is producer.PreparedDeviceC64Case
    assert value.verify() is value
    assert type(value.qualification_budget) is shared.DeviceWaterC64CaseBudget
    assert value.bed.context.runtime_bounds is value.qualification_budget
    assert producer._bridge_image(value._request_kwargs["bridge_limits"]) == producer._bridge_image(f.bridge_limits(origin))
    return value


def _check_origin(prepared, origin, proof_ref):
    checkpoint._keys(proof_ref, ("path", "size", "sha256"), "Root original proof ref")
    assert proof_ref["path"] == str(origin / "original-input-proof.json")
    checkpoint._check_file(proof_ref, _CAP)
    original = _proof(origin, proof_ref)
    _binding, actual = _capture_records(prepared._host, prepared._prepared_host_water, prepared._request_kwargs)
    assert all(actual[name] == _proof_bytes(original["originals"][name], name) for name in actual)
    assert _digest(actual["request_projection"]) == prepared.qualification_budget.case_request_binding_sha256
    host = prepared._host
    assert original["subject"] == dict(compiled_case_directory=str(host._case_dir),
        store_directory=str(host.state.voxel_column.deep_store.store_dir),
        native_case_identity_sha256=host.identity_record()["native_case_identity_sha256"],
        windowed_identity_sha256=host.identity_sha256,
        water_identity_sha256=prepared._prepared_host_water.identity_sha256,
        initial_manifest_digest=host.state.voxel_column.manifest_digest,
        source_manifest_sha256=prepared.qualification_budget.source_manifest_sha256)
    return original


def _budget_identity(budget):
    assert type(budget) is shared.DeviceWaterC64CaseBudget
    return checkpoint._budget_identity(shared.c64_case_budget_record(budget))


def _raw_accounting(runner):
    """All nine real counters/raw prefixes stay; future role maxima may differ."""
    fold = dict.fromkeys(checkpoint._COUNTERS, 0)
    for receipt in runner._receipts:
        if receipt["kind"] != "wind":
            raw = receipt["device_water_qualification"]
            fold = _combine_water_counters(fold, raw["entry"], raw["event"], runner._bed.context.bridge_limits)
            assert fold == raw["cumulative"]
    assert fold == runner._water_counters
    assert tuple(fold) == checkpoint._COUNTERS
    return copy.deepcopy(fold)


def _observation(runner, cp, *, prepared_case=None):
    if prepared_case is None:
        checkpoint.runner_admission(runner, checkpoint._DEFAULT.image)
    else:
        # Continuous has no checkpoint scope/store. Genuine guarded producer
        # and existing process/precheck gates are used, never a forged scope.
        assert type(prepared_case) is producer.PreparedDeviceC64Case
        assert prepared_case.verify() is prepared_case and prepared_case._runner is runner
        runner._precheck()
    _raw_accounting(runner)
    assert runner._water.vegetation_cover_fraction is runner._device_resources.static("vegetation").raster.cover_fraction
    state = runner._bed._accepted.state
    material = {name: getattr(state, name) for name in checkpoint._BED if name != "voxel_column"}
    column = state.voxel_column
    material["resident"] = dict(_bed_leaves(column, state.active_layer))
    material["tiny_full_columns"] = gather_full_columns(column, np.arange(64, dtype=np.int64), max_columns=64)
    material["native_column_scalars"] = (column.n_voxels, column.generation, column.pending_deep)
    material["pool"] = runner._device_resources._accepted.pool
    material["soil"] = runner._device_resources._accepted.soil
    material["statics"] = {name: runner._device_resources.static(name) for name in STATIC_NAMES}
    material["terrain"] = runner._water.terrain
    route = runner._routing
    if route is not None:
        assert route.terrain is runner._water.terrain
        runner._device_own._held_routing(route, runner._water)
    material["routing"] = None if route is None else dict(
        graph=route.graph, network=route.network, grid=route.grid,
        source=route.source, velocity=route.velocity, phase=route.phase, generation=route.generation)
    # Worker/PID/stream/copy diagnostics and resource predicates are not a
    # scientific equality oracle. Their complete original bytes and debits
    # remain in the actual persisted receipts/book; no product field is cut.
    receipts = [{key: value for key, value in row.items()
                 if key not in ("production", "device_water_qualification")} for row in runner._receipts]
    material["accounting"] = dict(ledger=runner._ledger.as_record(), totals=runner._totals,
        receipts=receipts, inventory0=runner._inventory0, native_time=runner._bed.time_s,
        cursor=runner._next, n_accepted=runner._n_accepted,
        bed_generation=runner._bed.generation, resource_generation=runner._device_resources.generation,
        routing_issue_counter=runner._device_own._routing_generation)
    return dict(schema="masy.windowed_device_c64_restart_scientific_oracle.v1", cursor=runner._next,
                scientific=_scientific(material, cp))


def _reference(root, expected_index_sha):
    index = _read(root / "reference" / "index.json", expected_index_sha)
    checkpoint._keys(index, ("schema", "boundaries", "original_proof", "case_budget_identity"), "reference index")
    assert index["schema"] == REFERENCE_SCHEMA
    assert type(index["boundaries"]) is list and len(index["boundaries"]) == 5
    for cursor, row in enumerate(index["boundaries"]):
        checkpoint._keys(row, ("cursor", "name", "size", "sha256"), "reference row")
        assert type(row["cursor"]) is int and row["cursor"] == cursor
        assert row["name"] == f"b{cursor:04d}.json"
        checkpoint._integer(row["size"], _CAP, "reference bytes", 1)
        checkpoint._digest(row["sha256"], "reference hash")
    proof = index["original_proof"]
    checkpoint._keys(proof, ("path", "size", "sha256"), "reference original proof")
    assert proof["path"] == str(root / "origin" / "original-input-proof.json")
    checkpoint._check_file(proof, _CAP)
    _proof(root / "origin", proof)
    assert type(index["case_budget_identity"]) is dict
    return index


def _matches(root, index, runner, cp):
    row = index["boundaries"][runner._next]
    expected = _read(root / "reference" / row["name"], row["sha256"])
    actual = _observation(runner, cp)
    assert checkpoint._json(actual) == checkpoint._json(expected), f"complete scientific boundary {runner._next} differs"
    return actual


def _control(root, name, handle, captured, *, original_parent, original_prefix):
    report = handle.report()
    value = dict(schema=CONTROL_SCHEMA, cursor=report["accepted"]["accepted"]["next_index"],
        store_file_sha256=report["store_file_sha256"], head_token=handle.head_token(),
        original_parent_token=original_parent, original_prefix=original_prefix,
        receipt_sha256=None if handle._state.ack.receipt_bytes is None else _digest(handle._state.ack.receipt_bytes),
        water_counters=captured["water_counters"], complete_receipts=captured["receipts"],
        deep_files=captured["deep_files"], restore_diagnostics=copy.deepcopy(handle._restore_diagnostics), pid=os.getpid())
    return value, _write(root / "control" / name, value)


def _check_old_files(rows):
    for row in rows:
        checkpoint._check_file(row, 64 << 20)


def _observe_current(root, index, cp, captured, handle):
    """One additional genuine checked readback, never a replaced owning API.

    The private prepared owner lives only in this call; the test retains plain
    receipts/counters/file references only. Its extra real load/copies/queries
    use the original positive readback banks and the same outer role deadline.
    """
    with checkpoint._writer_scope(handle._authority):
        head, _raw = handle._head()
        prepared, _chain = handle._load(head)
        runner = checkpoint._owner(prepared)
        _matches(root, index, runner, cp)
        captured.update(cursor=runner._next, water_counters=copy.deepcopy(runner._water_counters),
            receipts=copy.deepcopy(runner._receipts),
            deep_files=checkpoint._deep_files(runner._bed._accepted.state.voxel_column.deep_store,
                                               checkpoint._DEFAULT.image))
        del runner, prepared


def _duplicate_never_transports(handle, parent, expected_receipt_sha):
    """Real once-only path: same immutable files/receipt and Native counters."""
    from maple.core.backend.arrays import read_transfer_counters
    before = (handle._root / "HEAD").read_bytes()
    transfers = read_transfer_counters()
    diagnostics = copy.deepcopy(handle._restore_diagnostics)
    ack = handle.step(parent)
    assert _digest(ack.receipt_bytes) == expected_receipt_sha
    assert (handle._root / "HEAD").read_bytes() == before
    assert read_transfer_counters() == transfers
    assert handle._restore_diagnostics == diagnostics, "duplicate performed a restore/transport"
    return ack


def test_imported_device_c64_boundary_restart_role(windowed_device_c64_restart_role, cupy,
        device_c64_checkpoint_initial_case, device_c64_checkpoint_recovery_budget,
        windowed_device_c64_native_source_refs):
    """One selector, four SERIAL fresh Root roles; no authority/fallback here."""
    descriptor = windowed_device_c64_restart_role
    role, root, contract = _role(descriptor)
    cp, native_refs = cupy, windowed_device_c64_native_source_refs
    origin = root / "origin"
    assert callable(device_c64_checkpoint_initial_case)
    assert callable(device_c64_checkpoint_recovery_budget)
    if role == "continuous":
        reference = root / "reference"; reference.mkdir()
        prepared = _prepared(device_c64_checkpoint_initial_case(), origin)
        runner = prepared._runner
        original_ref = checkpoint._file_row(origin / "original-input-proof.json", _CAP)
        _check_origin(prepared, origin, original_ref)
        identity = _budget_identity(prepared.qualification_budget)
        rows = []
        for cursor in range(5):
            assert runner._next == cursor
            row = _write(reference / f"b{cursor:04d}.json",
                _observation(runner, cp, prepared_case=prepared))
            rows.append(dict(cursor=cursor, name=f"b{cursor:04d}.json", **row))
            if cursor < 4:
                receipt = prepared.step()  # Genuine guarded public input checks.
                assert receipt["status"] == receipt["code"] == "accepted", receipt
        assert prepared.report()["runner"]["status"] == "complete"
        _check_origin(prepared, origin, original_ref)
        _write(reference / "index.json", dict(schema=REFERENCE_SCHEMA, boundaries=rows,
            original_proof=original_ref, case_budget_identity=identity))
        return
    index = _reference(root, descriptor["expected_reference_index_sha256"])
    path, captured = origin / "accepted", {}
    if role == "write-W1":
        (root / "control").mkdir()
        prepared = _prepared(device_c64_checkpoint_initial_case(), origin)
        budget = prepared.qualification_budget
        _check_origin(prepared, origin, index["original_proof"])
        assert checkpoint._json(_budget_identity(budget)) == checkpoint._json(index["case_budget_identity"])
        runner = prepared._runner
        expected = _read(root / "reference" / index["boundaries"][0]["name"], index["boundaries"][0]["sha256"])
        assert checkpoint._json(_observation(runner, cp, prepared_case=prepared)) == checkpoint._json(expected)
        with closing(producer.create_device_c64_case_boundary_store(path, prepared_case=prepared,
                qualification_budget=budget, bridge_limits=f.bridge_limits(origin),
                native_source_refs=native_refs, storage_contract=contract)) as handle:
            # The old prepared stepping capability is permanently handed off.
            before = copy.deepcopy(handle._restore_diagnostics)
            with pytest.raises(producer.CaseDeviceC64Error, match="case_handed_off"):
                prepared.verify()
            assert handle._restore_diagnostics == before
            parent = handle.head_token()
            _observe_current(root, index, cp, captured, handle)
            assert captured["cursor"] == 0
            ack = handle.step(parent)
            _observe_current(root, index, cp, captured, handle)
            assert ack.sequence == captured["cursor"] == 1
            _control(root, "wind.json", handle, captured,
                     original_parent=parent, original_prefix=captured["receipts"])
        return
    name = "wind.json" if role == "recover-R1" else "water.json"
    prior = _read(root / "control" / name, descriptor["expected_control_sha256"])
    checkpoint._keys(prior, ("schema", "cursor", "store_file_sha256", "head_token",
        "original_parent_token", "original_prefix", "receipt_sha256", "water_counters",
        "complete_receipts", "deep_files", "restore_diagnostics", "pid"), "accepted previous role")
    assert prior["schema"] == CONTROL_SCHEMA
    assert prior["cursor"] == (1 if role == "recover-R1" else 2)
    assert prior["store_file_sha256"] == descriptor["expected_store_file_sha256"]
    assert type(prior["pid"]) is int  # Actual birth identity remains Root-owned.
    # This callable accepts NO digest and performs no HOST compile/load/adopt.
    budget = device_c64_checkpoint_recovery_budget()
    assert type(budget) is shared.DeviceWaterC64CaseBudget
    budget.validated()
    original = _proof(origin, index["original_proof"])
    assert budget.source_manifest_sha256 == original["subject"]["source_manifest_sha256"]
    assert budget.case_request_binding_sha256 == _digest(_proof_bytes(original["originals"]["request_projection"], "request_projection"))
    assert checkpoint._json(_budget_identity(budget)) == checkpoint._json(index["case_budget_identity"])
    with closing(producer.recover_device_c64_case_boundary_store(path,
              expected_store_file_sha256=descriptor["expected_store_file_sha256"],
              qualification_budget=budget, bridge_limits=f.bridge_limits(origin),
              native_source_refs=native_refs)) as handle:
        _observe_current(root, index, cp, captured, handle)
        assert captured["cursor"] == prior["cursor"]
        assert handle.head_token() == prior["head_token"]
        assert captured["receipts"] == prior["complete_receipts"]
        assert captured["water_counters"] == prior["water_counters"]
        _check_old_files(prior["deep_files"])
        if role == "recover-R1":
            ack = handle.step(handle.head_token())
            _observe_current(root, index, cp, captured, handle)
            assert ack.sequence == captured["cursor"] == 2
            assert captured["receipts"][:1] == prior["original_prefix"]
            _check_old_files(prior["deep_files"])
            _control(root, "water.json", handle, captured,
                original_parent=prior["head_token"], original_prefix=prior["original_prefix"])
        else:
            counters, receipts = copy.deepcopy(captured["water_counters"]), copy.deepcopy(captured["receipts"])
            ack = _duplicate_never_transports(handle, prior["original_parent_token"], prior["receipt_sha256"])
            assert ack.sequence == 2 and ack.head_token == prior["head_token"]
            _observe_current(root, index, cp, captured, handle)
            assert captured["water_counters"] == counters and captured["receipts"] == receipts
            for cursor in (3, 4):
                ack = handle.step(handle.head_token())
                _observe_current(root, index, cp, captured, handle)
                assert ack.sequence == captured["cursor"] == cursor
                assert captured["receipts"][:2] == prior["complete_receipts"]
            _check_old_files(prior["deep_files"])
            assert handle.report()["accepted"]["status"] == "complete"
            _control(root, "finish.json", handle, captured,
                original_parent=prior["original_parent_token"], original_prefix=prior["original_prefix"])


def test_imported_c64_checkpoint_type_graph(cupy, device_c64_checkpoint_initial_case,
                                          windowed_device_c64_native_source_refs):
    """Separate genuine imported genesis/create/readback/one-wind gate role.

    This requests no fake budget/class row and cannot reuse a factory gate.
    Root binds a separate origin and deferred real create preparation, all in
    the original600s and same full Native/OS/pool/source/currentness controls.
    """
    prepared = device_c64_checkpoint_initial_case()
    assert type(prepared) is producer.PreparedDeviceC64Case
    root = prepared._request_kwargs["bridge_limits"].run_dir
    _prepared(prepared, root)
    budget = prepared.qualification_budget
    proof_ref = checkpoint._file_row(root / "original-input-proof.json", _CAP)
    _check_origin(prepared, root, proof_ref)
    contract = store.StorageContract(operator_declares_local=True, operator_declares_fsync_durable=True)
    required = {module+":"+name for module, name, _fields in checkpoint._C64_TYPES}
    with closing(producer.create_device_c64_case_boundary_store(root / "accepted", prepared_case=prepared,
            qualification_budget=budget, bridge_limits=f.bridge_limits(root),
            native_source_refs=windowed_device_c64_native_source_refs, storage_contract=contract)) as handle:
        with checkpoint._writer_scope(handle._authority):
            ref = handle._header["originals"]
            assert ref["format"] == checkpoint.CASE_ORIGINAL_SCHEMA
            original, _payload, _raw = checkpoint._read_wire(handle._root/ref["relative_path"],
                expected_file_sha=ref["manifest_file_sha256"], expected_seal=ref["identity"],
                schema=checkpoint.CASE_ORIGINAL_SCHEMA, limits=handle._limits.image, expand=False)
            checkpoint._original_preflight(original, handle._header["source_identity"],
                handle._header["environment_sha256"], handle._limits.image)
            tags = set()
            def walk(node):
                if node["t"] in ("class", "enum"):
                    tags.add(node["c"])
                if node["t"] in ("class", "dict"):
                    for _key, child in node["v"]:
                        walk(child)
                elif node["t"] in ("tuple", "list"):
                    for child in node["v"]:
                        walk(child)
                elif node["t"] in ("enum", "numpy_scalar"):
                    walk(node["v"])
            walk(original["wire"]["root"])
            assert required <= tags, "real imported graph omitted a declared C64 owner"
            assert tags <= set(checkpoint._class_table(handle._limits.image))
            assert handle._header["runtime_reference"]["schema"] == checkpoint.CASE_RUNTIME_SCHEMA
        ack = handle.step(handle.head_token())
        assert ack.sequence == 1
        with checkpoint._writer_scope(handle._authority):
            head, _raw = handle._head()
            restored, chain = handle._load(head)
            owner = checkpoint._owner(restored)
            assert owner._next == 1 and owner._bed.generation == owner._device_resources.generation == 1
            assert owner._stop_receipt is owner._active_evidence is owner._pending_water_counters is None
            assert owner._bed.context.runtime_bounds is budget
            _raw_accounting(owner)
            image = chain[0][1]["image"]
            manifest, _raw = checkpoint._image_metadata(handle._root, image, handle._limits)
            checkpoint._preflight(manifest, handle._header["source_identity"],
                handle._header["environment_sha256"], handle._limits.image)
            assert manifest["state"]["next_index"] == 1
            del owner, restored
