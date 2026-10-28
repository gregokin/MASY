"""Real codec metadata/functions and local files, without transport or GPU.

These deliberately illustrative records are not Native state/capabilities.
Root runs them with the existing CPU Native metadata bootstrap. The separate
genuine WindowedDeltaBudgets registry regression remains unchanged.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib

import numpy as np
import pytest

from masy import dense_boundary_store as store
from masy import dense_original_inputs as oi
from masy import windowed_device_checkpoint as c
from masy.clock import ClockLedger, encode_time
from masy.dense_runner_checkpoint import BoundaryImageLimits


LIMITS = BoundaryImageLimits(max_cells=16, max_classes=1, max_voxels=6,
                             max_cohorts=4000, max_events=4)
SOURCE = {"illustrative_Source_identity": "a" * 64}
ENVIRONMENT = "b" * 64


def _chain(depth, *, named):
    value = None
    for _ in range(depth):
        value = {"x": value} if named else [value]
    return value


def _wire(value, limits=LIMITS):
    # Genuine NumPy is the HOST-only codec namespace, not a fake provider.
    encoder = c._Encoder(limits, np)
    root = encoder.node(value)
    return dict(root=root, arrays=encoder.rows, numeric_bytes=encoder.numeric)


def _sequence_wire(depth):
    node = {"t": "scalar", "v": None}
    for index in range(depth - 1, -1, -1):
        node = {"t": "list", "i": index, "v": [node]}
    return dict(root=node, arrays=[], numeric_bytes=0)


def _json_extent(value):
    """Independent raw JSON value count/depth; dictionary keys are not nodes."""
    children = value.values() if type(value) is dict else value if type(value) is list else ()
    extents = [_json_extent(child) for child in children]
    return (1 + sum(n for n, _ in extents),
            0 if not extents else 1 + max(d for _, d in extents))


def _reference(schema, relative):
    return dict(format=schema, relative_path=relative,
                manifest_file_sha256="c" * 64, identity="d" * 64)


def _record(kind, *, depth=24):
    """Closed metadata layout only; None fields are not physical inputs."""
    if kind == "original":
        inputs = dict.fromkeys(c._INPUT_FIELDS)
        inputs["case"] = _chain(depth - 2, named=True)
        material = dict(inputs=inputs, initial_pool=None, initial_soil=None, inventory0=None)
        record = dict(schema=c.ORIGINAL_SCHEMA, profile=c.PROFILE,
                      source_identity=copy.deepcopy(SOURCE), environment_sha256=ENVIRONMENT)
    else:
        material = dict.fromkeys(c._CURRENT_FIELDS)
        material["auxiliary"] = _chain(depth - 1, named=True)
        ledger = ClockLedger(7.25)
        record = dict(schema=c.SCHEMA, profile=c.PROFILE,
            source_identity=copy.deepcopy(SOURCE), environment_sha256=ENVIRONMENT,
            originals=_reference(c.ORIGINAL_SCHEMA, "originals/o-illustrative"),
            genesis=_reference(c.PHYSICAL_SCHEMA, "generations/g-0000/physical"),
            physical=_reference(c.PHYSICAL_SCHEMA, "generations/g-0000/physical"),
            case_sha256="e" * 64, schedule_sha256="f" * 64,
            state=dict(next_index=0, n_accepted=0, native_time=encode_time(7.25),
                       accepted_bed_label="illustrative-generation-0"),
            routing=None, receipts=[], ledger=ledger.as_record(),
            water_counters=dict.fromkeys(c._COUNTERS, 0))
    record.update(wire=_wire(material), payload=dict(size=128, sha256="e" * 64))
    record["manifest_sha256"] = c._sha(record)
    return record


def _preflight(record, kind, limits=LIMITS):
    function = c._original_preflight if kind == "original" else c._preflight
    return function(record, SOURCE, ENVIRONMENT, limits)


def _container(directory, record):
    """A real empty numeric NPY plus a sealed illustrative metadata record."""
    directory.mkdir()
    payload = directory / "a_0000.npy"
    with payload.open("xb") as handle:
        np.save(handle, np.empty(0, dtype=np.uint8), allow_pickle=False)
    body = copy.deepcopy(record)
    body.pop("manifest_sha256")
    raw_payload = payload.read_bytes()
    body["payload"] = dict(size=len(raw_payload), sha256=hashlib.sha256(raw_payload).hexdigest())
    body["manifest_sha256"] = c._sha(body)
    raw = c._json(body)
    (directory / "manifest.json").write_bytes(raw)
    return body, raw


def test_private_structural_views_preserve_original_limits_and_byte_cap():
    before = dataclasses.asdict(LIMITS)
    wire = c._wire_json_limits(LIMITS)
    record = c._record_json_limits(LIMITS)
    assert (wire.max_nodes, wire.max_depth) == (1600001, 74)
    assert (record.max_nodes, record.max_depth) == (1700000, 75)
    assert wire.max_metadata_bytes == record.max_metadata_bytes == LIMITS.max_metadata_bytes == 4 << 20
    assert dataclasses.asdict(LIMITS) == before


@pytest.mark.parametrize("named", [False, True])
def test_typed_depth24_is_admitted_despite_extra_json_wrappers(named):
    value = _wire(_chain(LIMITS.max_depth, named=named))
    _, depth = _json_extent(value)
    assert depth == (74 if named else 50)
    with pytest.raises(store.BoundaryStoreError, match="node/depth cap"):
        c._plain(value, LIMITS)
    assert c._wire_preflight(value, LIMITS) == 0


def test_typed_depth25_is_still_refused_inside_structurally_admitted_wire():
    value = _sequence_wire(LIMITS.max_depth + 1)
    assert _json_extent(value)[1] <= c._wire_json_limits(LIMITS).max_depth
    c._plain_wire(value, LIMITS)
    with pytest.raises(c.WindowedDeviceCheckpointError, match="typed tree"):
        c._wire_preflight(value, LIMITS)
    with pytest.raises(c.WindowedDeviceCheckpointError, match="typed input nodes/depth"):
        _wire(_chain(LIMITS.max_depth + 1, named=False))


def test_original_typed100000_node_boundary_and_next_node_refusal():
    value = _wire([None] * (LIMITS.max_nodes - 1))
    assert _json_extent(value)[0] > LIMITS.max_nodes
    assert c._wire_preflight(value, LIMITS) == 0
    value["root"]["v"].append({"t": "scalar", "v": None})
    with pytest.raises(c.WindowedDeviceCheckpointError, match="typed tree"):
        c._wire_preflight(value, LIMITS)
    with pytest.raises(c.WindowedDeviceCheckpointError, match="typed input nodes/depth"):
        _wire([None] * LIMITS.max_nodes)


def test_rank4_descriptor_and_numpy_scalar_are_in_the_closed_wire_bound():
    value = _wire((np.empty((0, 1, 1, 1), dtype=np.uint8), np.int64(7)))
    assert value["arrays"] == [dict(dtype="|u1", shape=[0, 1, 1, 1], offset=0,
                                   bytes=0, backend="numpy")]
    assert c._wire_preflight(value, LIMITS) == 0
    n, depth = _json_extent(value)
    view = c._wire_json_limits(LIMITS)
    assert n <= view.max_nodes and depth <= view.max_depth


def test_nonwire_node_budget_is_not_multiplied_by_record_encoding():
    limits = dataclasses.replace(LIMITS, max_nodes=8, max_depth=2)
    value = dict(wire=_wire(None), metadata=[None] * 5)
    assert _json_extent(dict(value, wire=None))[0] == 8
    assert c._plain_record(value, limits) is value
    value["metadata"].append(None)
    with pytest.raises(store.BoundaryStoreError, match="node/depth cap"):
        c._plain_record(value, limits)


def test_nonwire_depth_budget_is_not_multiplied_by_record_encoding():
    limits = dataclasses.replace(LIMITS, max_depth=2)
    value = dict(wire=_wire(None), metadata=_chain(1, named=False))
    assert c._plain_record(value, limits) is value
    value["metadata"] = _chain(2, named=False)
    with pytest.raises(store.BoundaryStoreError, match="node/depth cap"):
        c._plain_record(value, limits)


def test_complete_record_aggregate_byte_cap_includes_wire_and_seal():
    limits = dataclasses.replace(LIMITS, max_metadata_bytes=512)
    value = dict(wire=_wire("x" * 100), metadata="y" * 350, manifest_sha256="a" * 64)
    c._plain(dict(value, wire=None), limits)
    c._plain_wire(value["wire"], limits)
    with pytest.raises(store.BoundaryStoreError, match="aggregate wire size"):
        c._plain_record(value, limits)
    # The seal must be in the final guarded skeleton, not added after admission.
    limits = dataclasses.replace(LIMITS, max_nodes=2)
    unsealed = dict(wire=_wire(None))
    assert c._plain_record(unsealed, limits) is unsealed
    with pytest.raises(store.BoundaryStoreError, match="node/depth cap"):
        c._plain_record(dict(unsealed, manifest_sha256="a" * 64), limits)


@pytest.mark.parametrize("kind", ["original", "current"])
def test_original_and_outer_preflights_admit_valid_typed_depth_without_mutation(kind):
    value = _record(kind)
    before = c._json(value)
    assert _json_extent(value)[1] == 75
    with pytest.raises(store.BoundaryStoreError, match="node/depth cap"):
        c._plain(value, LIMITS)
    assert _preflight(value, kind) is None
    assert c._json(value) == before


@pytest.mark.parametrize("kind", ["original", "current"])
def test_owning_preflights_keep_closed_keys_and_exact_root_order(kind):
    value = _record(kind)
    bad = copy.deepcopy(value); bad["unexpected"] = None
    with pytest.raises(c.WindowedDeviceCheckpointError, match="closed_record"):
        _preflight(bad, kind)
    bad = copy.deepcopy(value); bad["wire"]["root"]["v"].reverse()
    with pytest.raises(c.WindowedDeviceCheckpointError, match="closed_record"):
        _preflight(bad, kind)
    bad = copy.deepcopy(value); bad["wire"]["unexpected"] = None
    with pytest.raises(c.WindowedDeviceCheckpointError, match="closed_record"):
        _preflight(bad, kind)


@pytest.mark.parametrize("kind", ["original", "current"])
def test_read_wire_roundtrip_checks_same_canonical_bytes_seal_and_owning_layout(tmp_path, kind):
    directory = tmp_path / kind
    record, raw = _container(directory, _record(kind))
    with pytest.raises(oi.OriginalInputsError, match="node/depth cap"):
        oi._strict_json(raw, LIMITS)
    loaded, payload, read_raw = c._read_wire(directory,
        expected_file_sha=hashlib.sha256(raw).hexdigest(), expected_seal=record["manifest_sha256"],
        schema=record["schema"], limits=LIMITS, expand=False)
    assert payload is None and read_raw == raw and c._json(loaded) == raw
    assert loaded == record and _preflight(loaded, kind) is None


@pytest.mark.parametrize("change,code", [("duplicate", "metadata_duplicate"), ("nonfinite", "metadata_nonfinite")])
def test_structural_read_keeps_duplicate_and_nonfinite_json_refusals(tmp_path, change, code):
    directory = tmp_path / "raw"
    record, raw = _container(directory, _record("original"))
    if change == "duplicate":
        raw = raw.replace(b'"schema":', b'"schema":"duplicate","schema":', 1)
    else:
        raw = raw.replace(b'"numeric_bytes":0', b'"numeric_bytes":NaN', 1)
    (directory / "manifest.json").write_bytes(raw)
    with pytest.raises(oi.OriginalInputsError) as refusal:
        c._read_wire(directory, schema=record["schema"], limits=LIMITS, expand=False)
    assert refusal.value.code == code


def test_read_preserves_raw_byte_cap_before_json_or_payload(tmp_path):
    limits = dataclasses.replace(LIMITS, max_metadata_bytes=512)
    directory = tmp_path / "raw"; directory.mkdir()
    (directory / "manifest.json").write_bytes(b" " * 513)
    with pytest.raises(store.BoundaryStoreError, match="raw cap before hash/JSON"):
        c._read_wire(directory, schema=c.SCHEMA, limits=limits, expand=False)
    with pytest.raises(oi.OriginalInputsError, match="raw byte cap"):
        oi._strict_json(b" " * 513, c._record_json_limits(limits))


def test_record_admission_refuses_nonjson_cycles_and_missing_wire():
    with pytest.raises(c.WindowedDeviceCheckpointError, match="closed_record"):
        c._plain_record({}, LIMITS)
    value = dict(wire=_wire(None), metadata=[])
    value["metadata"].append(value["metadata"])
    with pytest.raises(store.BoundaryStoreError, match="cyclic metadata"):
        c._plain_record(value, LIMITS)
    with pytest.raises(store.BoundaryStoreError, match="non-JSON value"):
        c._plain_record(dict(wire=_wire(None), metadata=object()), LIMITS)


@pytest.mark.parametrize("dimension", ["nodes", "depth"])
def test_derived_structural_bounds_also_refuse_malformed_json_expansion(dimension):
    limits = dataclasses.replace(LIMITS, max_nodes=2, max_depth=1)
    value = _wire(None)
    value["root"] = [None] * 34 if dimension == "nodes" else _chain(6, named=False)
    with pytest.raises(store.BoundaryStoreError, match="node/depth cap"):
        c._wire_preflight(value, limits)
    raw = c._json(dict(wire=value))
    with pytest.raises(oi.OriginalInputsError, match="node/depth cap"):
        oi._strict_json(raw, c._record_json_limits(limits))


def test_rank4_named_arrays_nearly_reach_wire_and_record_node_bounds():
    k = 33
    limits = dataclasses.replace(LIMITS, max_nodes=k, max_depth=2)
    # Each distinct real empty array contributes4 node values +10 descriptor
    # values; each named edge contributes its pair-list and string key.
    material = {f"array-{index}": np.empty((0, 1, 1, 1), dtype=np.uint8)
                for index in range(k - 1)}
    value = _wire(material, limits)
    expected_nodes = 3 + 4 + (2 + 14) * (k - 1)
    assert len(value["arrays"]) == k - 1
    assert all(row["shape"] == [0, 1, 1, 1] and row["bytes"] == 0
               for row in value["arrays"])
    assert _json_extent(value)[0] == expected_nodes == 16 * k - 9
    assert c._wire_preflight(value, limits) == 0
    record = dict(wire=value, metadata=[None] * (k - 3))
    assert _json_extent(dict(record, wire=None))[0] == k
    assert _json_extent(record)[0] == k - 1 + expected_nodes == 17 * k - 10
    assert c._plain_record(record, limits) is record

    # One fewer allowed typed occurrence gives a structurally smaller bound.
    # The same near-tight wire must fail its raw counter, before typed visit.
    smaller = dataclasses.replace(limits, max_nodes=k - 1)
    with pytest.raises(store.BoundaryStoreError, match="node/depth cap"):
        c._wire_preflight(value, smaller)
    # Fit the nonwire skeleton to that smaller budget so only the complete
    # record counter, rather than its skeleton, causes this refusal.
    smaller_record = dict(wire=value, metadata=[None] * (k - 4))
    assert _json_extent(dict(smaller_record, wire=None))[0] == k - 1
    c._plain(dict(smaller_record, wire=None), smaller)
    with pytest.raises(store.BoundaryStoreError, match="node/depth cap"):
        c._plain_record(smaller_record, smaller)


@pytest.mark.parametrize("change,code", [("extra", "closed_record"), ("forward_ref", "reference"),
                                      ("unknown_class", "class_fields"), ("unreferenced", "array_layout")])
def test_structural_wire_admission_does_not_weaken_closed_typed_refusals(change, code):
    value = _wire(None)
    if change == "extra":
        value["root"]["extra"] = None
    elif change == "forward_ref":
        value["root"] = {"t": "ref", "v": 0}
    elif change == "unknown_class":
        value["root"] = {"t": "class", "i": 0, "c": "os:system", "v": []}
    else:
        value["arrays"] = [dict(dtype="|u1", shape=[0], offset=0, bytes=0, backend="numpy")]
    with pytest.raises(c.WindowedDeviceCheckpointError) as refusal:
        c._wire_preflight(value, LIMITS)
    assert refusal.value.code == code


def test_record_reader_retains_manifest_byte_hash_and_semantic_seal(tmp_path):
    directory = tmp_path / "raw"
    record, raw = _container(directory, _record("original"))
    with pytest.raises(c.WindowedDeviceCheckpointError, match="raw manifest hash"):
        c._read_wire(directory, expected_file_sha="0" * 64,
                     schema=c.ORIGINAL_SCHEMA, limits=LIMITS, expand=False)
    record["wire"]["root"]["v"][-1][1] = {"t": "scalar", "v": 1}
    (directory / "manifest.json").write_bytes(c._json(record))
    with pytest.raises(c.WindowedDeviceCheckpointError, match="manifest seal"):
        c._read_wire(directory, schema=c.ORIGINAL_SCHEMA, limits=LIMITS, expand=False)


def test_unrelated_generic_json_guards_and_typed_limits_stay_closed():
    # Neither global parser nor limit class is widened by the private views.
    value = _chain(25, named=False)
    with pytest.raises(oi.OriginalInputsError, match="node/depth cap"):
        oi._strict_json(c._json(value), LIMITS)
    with pytest.raises(store.BoundaryStoreError, match="node/depth cap"):
        c._plain(value, LIMITS)
    for field, value in (("max_nodes", 100001), ("max_depth", 25), ("max_metadata_bytes", (4 << 20) + 1)):
        with pytest.raises(ValueError, match="limit_cap"):
            dataclasses.replace(LIMITS, **{field: value})


def _synthetic_owning_qualification_record():
    """Real metadata constructor; refs label nonexistent synthetic files only.

    No file is read and no authority/runtime/provider is constructed. The
    fixed factory metadata satisfies the genuine descriptor's Source guard;
    these references could not pass its real readonly-file/runtime gate.
    """
    from maple_syrup import windowed_shared_bed_device as shared

    def ref(name, size=1, digest="1" * 64):
        return shared.SourceReference("/synthetic-budget-metadata-only/" + name, size, digest)

    budget = shared.DeviceWaterQualificationBudget(
        native_source_identity="2" * 64,
        current_runtime_ref=ref("current_runtime.json"),
        factory_source=ref("test_windowed_shared_device_physical.py",
            shared._QUALIFICATION_FACTORY_SIZE, shared._QUALIFICATION_FACTORY_SHA256),
        grant=ref("product_budget.json"),
        source_bindings=tuple((name, ref(name.replace(".", "/") + ".py"))
                              for name in shared._SOURCE_MODULES),
        gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
        source_manifest_sha256="3" * 64)
    assert type(budget) is shared.DeviceWaterQualificationBudget
    return shared, budget, shared.qualification_budget_record(budget)


def test_genuine_budget_opaque_tuple_projection_is_strict_json_without_changing_bytes():
    shared, budget, emitted = _synthetic_owning_qualification_record()
    before = copy.deepcopy(emitted)
    budget_before = dataclasses.asdict(budget)
    projected = c._budget_identity(emitted)
    for name, roster in (("opaque_external_bounds", shared._EXTERNAL_ROLES),
                         ("opaque_primitive_units", shared._UNIT_NAMES)):
        assert type(emitted[name]) is tuple and all(type(row) is tuple for row in emitted[name])
        assert emitted[name] == tuple((label, None) for label in roster)
        assert type(projected[name]) is list and all(type(row) is list for row in projected[name])
        assert projected[name] == [[label, None] for label in roster]
    assert len(projected["opaque_external_bounds"]) == 11
    assert len(projected["opaque_primitive_units"]) == 8
    assert projected["opaque_constructor_accounting"] is None

    # Reinsert only the genuine producer's two old tuple fields to reproduce
    # the former strict-walker refusal, with no copied projection algorithm.
    old_tuple_form = copy.deepcopy(projected)
    for name in ("opaque_external_bounds", "opaque_primitive_units"):
        old_tuple_form[name] = emitted[name]
    assert c._json(old_tuple_form) == c._json(projected)
    assert c._sha(old_tuple_form) == c._sha(projected)
    record = _record("original")
    record["source_identity"] = projected
    body = dict(record); body.pop("manifest_sha256")
    record["manifest_sha256"] = c._sha(body)
    raw_before = c._json(record)
    assert c._plain_record(record, LIMITS) is record
    assert c._original_preflight(record, projected, ENVIRONMENT, LIMITS) is None
    assert c._json(record) == raw_before
    bad = copy.deepcopy(record); bad["source_identity"] = old_tuple_form
    with pytest.raises(store.BoundaryStoreError, match="non-JSON value"):
        c._plain_record(bad, LIMITS)
    bad = copy.deepcopy(record); bad["source_identity"]["unrelated_tuple"] = (("x", None),)
    with pytest.raises(store.BoundaryStoreError, match="non-JSON value"):
        c._plain_record(bad, LIMITS)
    assert emitted == before and dataclasses.asdict(budget) == budget_before


def test_genuine_budget_already_json_rosters_keep_names_none_order_and_input():
    import json

    shared, budget, emitted = _synthetic_owning_qualification_record()
    # The actual owning canonical encoder is the independent JSON-list path.
    json_record = json.loads(shared.qualification_budget_bytes(budget))
    before = copy.deepcopy(json_record)
    projected = c._budget_identity(json_record)
    assert projected == c._budget_identity(emitted)
    assert c._json(projected) == c._json(c._budget_identity(emitted))
    for name in ("opaque_external_bounds", "opaque_primitive_units"):
        assert projected[name] == json_record[name]
        assert all(type(row) is list and row[1] is None for row in projected[name])
    assert projected["opaque_constructor_accounting"] is None
    assert json_record == before


def _genuine_native_small_snapshot(tmp_path):
    """Native snapshot fixture values, in the closed 4x4/K1/V6/W3 shape.

    Root supplies the actual Native tests/shared_storage Source path for
    test_deep_store, exactly as Native's own snapshot tests do. No transport,
    fake holder or provider is used; all partition/storage work is Native.
    """
    from test_deep_store import MRES, packed_columns

    from maple.core.boundaries import AxisBoundary, BoundaryKind
    from maple.core.parameters.geometry import GeometrySpec
    from maple.core.parameters.grain_classes import GrainClass, GrainClassSet
    from maple.core.types.sediment_availability import SedimentAvailabilityState
    from maple.core.types.voxel import VoxelColumnState
    from maple.core.types.water import WaterState
    from maple.surface.active_layer.initialization import initialize_active_layer_from_voxels
    from maple.surface.voxels import windowed_snapshot as native
    from maple.surface.voxels.diagnostics import check_voxel_column_state
    from maple.surface.voxels.windowed_inventory import windowed_column_from_dense

    geometry = GeometrySpec(
        nx=4, ny=4, dx_m=1.0, dy_m=1.0,
        boundary_x=AxisBoundary(kind=BoundaryKind.PERIODIC),
        boundary_y=AxisBoundary(kind=BoundaryKind.PERIODIC),
        voxel_dz_m=0.1, bulk_density_kg_m3=1250.0, active_layer_thickness_m=0.05)
    classes = GrainClassSet(classes=(GrainClass(
        class_id="c0", diameter_m=1.0e-4, particle_density_kg_m3=2650.0,
        is_aggregate=False),))
    column = VoxelColumnState(mass_kg=packed_columns("random", 4, 4, 6, 1, seed=3))
    check_voxel_column_state(column, geometry, MRES)
    column, active, _ = initialize_active_layer_from_voxels(column, geometry, MRES)
    state = windowed_column_from_dense(
        column, geometry, MRES, store_dir=tmp_path / "native-deep",
        class_ids=tuple(grain.class_id for grain in classes.classes),
        max_store_bytes=LIMITS.native_max_bytes, window_voxels=3,
        chunk_columns=64, with_prefix_packets=True)
    availability = SedimentAvailabilityState(
        available_mass_kg=active.mass_kg.copy(), bound_mass_kg=np.zeros_like(active.mass_kg))
    water = WaterState(depth_m=np.zeros((4, 4), dtype=np.float64),
                       mobile_mass_by_cell_class_kg=np.zeros((4, 4, 1), dtype=np.float64))
    path = tmp_path / "native_state.npz"
    provenance = {"test_scope": "genuine Native CPU snapshot format; no events"}
    native.save_windowed_state_snapshot(
        state, active, 12.5, path, step=3, geometry=geometry,
        mass_resolution_kg=MRES, sediment_availability=availability,
        water=water, provenance=provenance)
    return native, geometry, classes, MRES, state, path, provenance


def test_genuine_native_int32_snapshot_matches_checkpoint_preflight(tmp_path):
    import json
    import zipfile

    native, geometry, classes, mres, state, path, provenance = _genuine_native_small_snapshot(tmp_path)
    before = path.read_bytes()
    assert state.base.dtype == np.dtype(np.int32)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"metadata", *native.COLUMN_ARRAYS, *native.OTHER_ARRAYS}
        actual_metadata = json.loads(str(archive["metadata"].item()))
        actual_numeric = sum(archive[name].nbytes for name in (*native.COLUMN_ARRAYS, *native.OTHER_ARRAYS))
        assert archive["base"].dtype.str == "<i4"
    with zipfile.ZipFile(path) as archive:
        assert len(archive.infolist()) == 12
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
    metadata, numeric = c._snapshot_preflight(path, LIMITS)
    assert metadata == actual_metadata
    assert numeric == actual_numeric
    assert metadata["provenance"] == provenance
    assert (metadata["n_voxels"], metadata["time_s"], metadata["step"]) == (6, 12.5, 3)
    restored = native.load_windowed_state_snapshot(
        path, geometry=geometry, grain_classes=classes,
        mass_resolution_kg=mres, max_bytes=LIMITS.native_max_bytes)
    assert restored["validated"] and restored["restartable"] and not restored["is_diagnostic"]
    assert restored["voxel_column"].base.dtype == np.dtype(np.int32)
    for name in native.COLUMN_ARRAYS:
        assert getattr(restored["voxel_column"], name).tobytes() == getattr(state, name).tobytes(), name
    assert restored["voxel_column"].height_packet.packet.tobytes() == state.height_packet.packet.tobytes()
    assert restored["provenance"] == provenance
    assert path.read_bytes() == before


def test_genuine_native_snapshot_with_int64_base_remains_refused(tmp_path):
    native, geometry, classes, mres, state, path, _ = _genuine_native_small_snapshot(tmp_path)
    before = path.read_bytes()
    base_before = state.base.tobytes()
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True, order="C") for name in archive.files}
    assert arrays["base"].dtype == np.dtype(np.int32)
    original_members = {name: (value.dtype.str, value.shape, value.tobytes())
                        for name, value in arrays.items()}
    arrays["base"] = arrays["base"].astype(np.int64)
    bad = tmp_path / "int64_base.npz"
    np.savez(bad, **arrays)
    with np.load(bad, allow_pickle=False) as archive:
        for name, original in original_members.items():
            if name != "base":
                assert (archive[name].dtype.str, archive[name].shape, archive[name].tobytes()) == original
        assert archive["base"].dtype == np.dtype(np.int64)
        assert np.array_equal(archive["base"], state.base)
    with pytest.raises(c.WindowedDeviceCheckpointError) as refusal:
        c._snapshot_preflight(bad, LIMITS)
    assert (refusal.value.code, refusal.value.detail) == ("native_header", "base")
    with pytest.raises(ValueError, match="base must be int32"):
        native.load_windowed_state_snapshot(
            bad, geometry=geometry, grain_classes=classes,
            mass_resolution_kg=mres, max_bytes=LIMITS.native_max_bytes)
    assert path.read_bytes() == before
    assert state.base.dtype == np.dtype(np.int32) and state.base.tobytes() == base_before
