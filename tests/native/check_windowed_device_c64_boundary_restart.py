"""One Root-only selector, invoked in four SERIAL fresh protected GPU roles.

Root supplies ``windowed_device_c64_restart_role`` and the existing genuine
``device_water_c64_qualification_budget`` fixture. This module starts no child,
issues no authority, invents no runtime facts and contains no grant fallback.
All oracle/control files are actual Root-local runtime artifacts. Source-only
reviewers receive this Source, never those files or model payloads.
"""
from __future__ import annotations

import copy
import dataclasses
import enum
import hashlib
import math
import os
from contextlib import closing
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import fixture_windowed_device_c64_alternation as f
from tests.windowed_wind_gpu.test_device_reduced_step import cupy  # noqa: F401
from maple.core.backend import to_host
from maple.surface.topographic_commit.windowed_device import _bed_leaves
from maple.surface.voxels.windowed_inventory import gather_full_columns

from masy import dense_boundary_store as store
from masy import windowed_device_c64_checkpoint as checkpoint
from masy.dense_device_resources import STATIC_NAMES
from masy.windowed_device_c64_runner import DeviceWindowedC64Runner

pytestmark = pytest.mark.gpu
ROLE_SCHEMA = "masy.windowed_device_c64_restart_test_role.v1"
ROLES = ("continuous", "write-W1", "recover-R1", "recover-finish")
_CAP = 4 << 20


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


def _observation(runner, cp):
    checkpoint.runner_admission(runner, checkpoint._DEFAULT.image)
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
    checkpoint._keys(index, ("schema", "boundaries"), "reference index")
    assert index["schema"] == "masy.windowed_device_c64_restart_oracles.v1"
    assert type(index["boundaries"]) is list and len(index["boundaries"]) == 5
    for cursor, row in enumerate(index["boundaries"]):
        checkpoint._keys(row, ("cursor", "name", "size", "sha256"), "reference row")
        assert type(row["cursor"]) is int and row["cursor"] == cursor
        assert row["name"] == f"b{cursor:04d}.json"
        checkpoint._integer(row["size"], _CAP, "reference bytes", 1)
        checkpoint._digest(row["sha256"], "reference hash")
    return index


def _matches(root, index, runner, cp):
    row = index["boundaries"][runner._next]
    expected = _read(root / "reference" / row["name"], row["sha256"])
    actual = _observation(runner, cp)
    assert checkpoint._json(actual) == checkpoint._json(expected), f"complete scientific boundary {runner._next} differs"
    return actual


def _control(root, name, handle, captured, *, original_parent, original_prefix):
    report = handle.report()
    value = dict(schema="masy.windowed_device_c64_restart_control.v1", cursor=report["accepted"]["accepted"]["next_index"],
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


def test_windowed_device_c64_boundary_restart_role(windowed_device_c64_restart_role, cupy,
                                               device_water_c64_qualification_budget,
                                               windowed_device_c64_native_source_refs):
    """Root invokes this SAME selector once per fresh serial protected role."""
    descriptor = windowed_device_c64_restart_role
    role, root, contract = _role(descriptor)
    cp, budget = cupy, device_water_c64_qualification_budget
    native_refs = windowed_device_c64_native_source_refs
    assert budget.gpu_uuid  # Genuine UUID/runtime/birth verification is Root-owned.
    if role == "continuous":
        reference = root / "reference"; reference.mkdir()
        case = f.components(root / "continuous", cp, budget)
        runner = f.make_runner(case)
        rows = []
        for cursor in range(5):
            assert runner._next == cursor
            row = _write(reference / f"b{cursor:04d}.json", _observation(runner, cp))
            rows.append(dict(cursor=cursor, name=f"b{cursor:04d}.json", **row))
            if cursor < 4:
                receipt = runner.step()
                assert receipt["status"] == receipt["code"] == "accepted", receipt
        assert runner.report()["status"] == "complete"
        _write(reference / "index.json", dict(schema="masy.windowed_device_c64_restart_oracles.v1", boundaries=rows))
        return
    index = _reference(root, descriptor["expected_reference_index_sha256"])
    path, captured = root / "case" / "accepted", {}
    if role == "write-W1":
        (root / "control").mkdir()
        case = f.components(root / "case", cp, budget)
        runner = f.make_runner(case)
        _matches(root, index, runner, cp)
        with closing(checkpoint.create_windowed_device_c64_boundary_store(path, runner=runner,
                qualification_budget=budget, bridge_limits=f.bridge_limits(root / "case"),
                native_source_refs=native_refs, storage_contract=contract)) as handle:
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
    assert prior["schema"] == "masy.windowed_device_c64_restart_control.v1"
    assert prior["cursor"] == (1 if role == "recover-R1" else 2)
    assert prior["store_file_sha256"] == descriptor["expected_store_file_sha256"]
    # A genuine new role is required; PID reuse is checked by Root's birth
    # receipts, not inferred from this diagnostic integer alone.
    assert type(prior["pid"]) is int
    with closing(checkpoint.recover_windowed_device_c64_boundary_store(path,
              expected_store_file_sha256=descriptor["expected_store_file_sha256"],
              qualification_budget=budget, bridge_limits=f.bridge_limits(root / "case"),
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


def test_c64_factory_checkpoint_type_graph(tmp_path, cupy,
        device_water_c64_qualification_budget, windowed_device_c64_native_source_refs):
    """Actual genesis and one actual accepted wind, all closed typed paths.

    Root must issue a genuine separate controlled gate role; this selector
    does not fabricate a budget or broaden any of the four restart roles.
    """
    root = tmp_path / "case"
    case = f.components(root, cupy, device_water_c64_qualification_budget)
    runner = f.make_runner(case)
    contract = store.StorageContract(operator_declares_local=True, operator_declares_fsync_durable=True)
    required = {module+":"+name for module, name, _fields in checkpoint._C64_TYPES}
    with closing(checkpoint.create_windowed_device_c64_boundary_store(root / "accepted", runner=runner,
            qualification_budget=device_water_c64_qualification_budget, bridge_limits=f.bridge_limits(root),
            native_source_refs=windowed_device_c64_native_source_refs, storage_contract=contract)) as handle:
        with checkpoint._writer_scope(handle._authority):
            ref = handle._header["originals"]
            original, _payload, _raw = checkpoint._read_wire(handle._root/ref["relative_path"],
                expected_file_sha=ref["manifest_file_sha256"], expected_seal=ref["identity"],
                schema=checkpoint.ORIGINAL_SCHEMA, limits=handle._limits.image, expand=False)
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
            assert required <= tags, "complete real factory graph omitted a declared C64 owner"
            assert tags <= set(checkpoint._class_table(handle._limits.image))
        ack = handle.step(handle.head_token())
        assert ack.sequence == 1
        with checkpoint._writer_scope(handle._authority):
            head, _raw = handle._head()
            prepared, chain = handle._load(head)
            owner = checkpoint._owner(prepared)
            assert owner._next == 1 and owner._bed.generation == owner._device_resources.generation == 1
            assert owner._stop_receipt is owner._active_evidence is owner._pending_water_counters is None
            assert owner._bed.context.runtime_bounds is device_water_c64_qualification_budget
            image = chain[0][1]["image"]
            manifest, _raw = checkpoint._image_metadata(handle._root, image, handle._limits)
            checkpoint._preflight(manifest, handle._header["source_identity"],
                handle._header["environment_sha256"], handle._limits.image)
            assert manifest["state"]["next_index"] == 1
            del owner, prepared


def test_c64_case_budget_is_refused_before_reference_or_gpu_work(tmp_path):
    """Genuine owning DTO, explicitly synthetic data, no physical authority."""
    from maple_syrup.windowed_shared_bed_device import DeviceWaterC64CaseBudget, SourceReference
    ref = SourceReference(str(tmp_path / "explicitly-synthetic-unissued-source.py"), 1, "0"*64)
    budget = DeviceWaterC64CaseBudget(native_source_identity="0"*64, current_runtime_ref=ref,
        producer_module="masy.case_device_c64", producer_source=ref, grant=ref, source_bindings=(), gpu_uuid="GPU-explicitly-synthetic-unissued",
        source_manifest_sha256="0"*64, case_request_binding_sha256="0"*64)
    # No referenced file or store exists. Authority's exact class check must
    # refuse before validation, read_reference, process/provider query or open.
    with pytest.raises(checkpoint.codec.WindowedDeviceCheckpointError, match="CaseBudget unsupported"):
        checkpoint.recover_windowed_device_c64_boundary_store(tmp_path/"never-created-store",
            expected_store_file_sha256="0"*64, qualification_budget=budget,
            bridge_limits=None, native_source_refs=())
    assert not (tmp_path/"never-created-store").exists()
    assert not (tmp_path/"explicitly-synthetic-unissued-source.py").exists()
