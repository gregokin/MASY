"""One Root-only selector, invoked in four SERIAL fresh protected GPU roles.

Root supplies ``windowed_device_restart_role`` and the existing genuine
``device_water_qualification_budget`` fixture. This module starts no child,
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
from unittest import mock

import numpy as np
import pytest
import fixture_windowed_device_alternation as f
from tests.windowed_wind_gpu.test_device_reduced_step import cupy  # noqa: F401
from maple.core.backend import to_host
from maple.surface.topographic_commit.windowed_device import _bed_leaves
from maple.surface.voxels.windowed_inventory import gather_full_columns

from masy import dense_boundary_store as store
from masy import windowed_device_checkpoint as checkpoint
from masy.dense_device_resources import STATIC_NAMES
from masy.windowed_device_runner import DeviceWindowedRunner

pytestmark = pytest.mark.gpu
ROLE_SCHEMA = "masy.windowed_device_restart_test_role.v1"
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
    raw = checkpoint._read_file(path, _CAP)
    assert _digest(raw) == expected, "Root-pinned immutable oracle/control bytes changed"
    return checkpoint.oi._strict_json(raw, checkpoint._GPU_DEFAULT.image)


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
    checkpoint.runner_admission(runner, checkpoint._GPU_DEFAULT.image)
    assert runner._water.vegetation_cover_fraction is runner._device_resources.static("vegetation").raster.cover_fraction
    state = runner._bed._accepted.state
    material = {name: getattr(state, name) for name in checkpoint._BED if name != "voxel_column"}
    column = state.voxel_column
    material["resident"] = dict(_bed_leaves(column, state.active_layer))
    material["tiny_full_columns"] = gather_full_columns(column, np.arange(16, dtype=np.int64), max_columns=16)
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
    return dict(schema="masy.windowed_device_restart_scientific_oracle.v1", cursor=runner._next,
                scientific=_scientific(material, cp))


def _reference(root, expected_index_sha):
    index = _read(root / "reference" / "index.json", expected_index_sha)
    checkpoint._keys(index, ("schema", "boundaries"), "reference index")
    assert index["schema"] == "masy.windowed_device_restart_oracles.v1"
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
    value = dict(schema="masy.windowed_device_restart_control.v1", cursor=report["accepted"]["accepted"]["next_index"],
        store_file_sha256=report["store_file_sha256"], head_token=handle.head_token(),
        original_parent_token=original_parent, original_prefix=original_prefix,
        receipt_sha256=None if handle._state.ack.receipt_bytes is None else _digest(handle._state.ack.receipt_bytes),
        water_counters=captured["water_counters"], complete_receipts=captured["receipts"],
        deep_files=captured["deep_files"], pid=os.getpid())
    return value, _write(root / "control" / name, value)


def _check_old_files(rows):
    for row in rows:
        checkpoint._check_file(row, 64 << 20)


def _observe_preparations(root, index, cp, captured):
    actual = checkpoint._prepare
    def observe(*args, **kwargs):
        prepared = actual(*args, **kwargs)
        runner = checkpoint._owner(prepared)
        _matches(root, index, runner, cp)
        captured.update(cursor=runner._next, water_counters=copy.deepcopy(runner._water_counters),
            receipts=copy.deepcopy(runner._receipts),
            deep_files=checkpoint._deep_files(runner._bed._accepted.state.voxel_column.deep_store,
                                               checkpoint._GPU_DEFAULT.image))
        return prepared
    return mock.patch.object(checkpoint, "_prepare", new=observe)


def _duplicate_never_transports(handle, parent, expected_receipt_sha):
    before = (handle._root / "HEAD").read_bytes()
    fail = AssertionError("duplicate token replayed loading, transport or finalization")
    with (mock.patch.object(checkpoint, "_prepare", side_effect=fail),
          mock.patch.object(DeviceWindowedRunner, "step", side_effect=fail),
          mock.patch.object(f.native_wind, "prepare_device_windowed_small_feature_wind_record", side_effect=fail),
          mock.patch.object(f.device_water, "commit_qualification_device_entry", side_effect=fail),
          mock.patch.object(f.device_water, "complete_qualification_device_event", side_effect=fail)):
        ack = handle.step(parent)
    assert _digest(ack.receipt_bytes) == expected_receipt_sha
    assert (handle._root / "HEAD").read_bytes() == before
    return ack


def test_windowed_device_boundary_restart_role(windowed_device_restart_role, cupy,
                                               device_water_qualification_budget):
    """Root invokes this SAME selector once per fresh serial protected role."""
    descriptor = windowed_device_restart_role
    role, root, contract = _role(descriptor)
    cp, budget = cupy, device_water_qualification_budget
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
        _write(reference / "index.json", dict(schema="masy.windowed_device_restart_oracles.v1", boundaries=rows))
        return
    index = _reference(root, descriptor["expected_reference_index_sha256"])
    path, captured = root / "case" / "accepted", {}
    with _observe_preparations(root, index, cp, captured):
        if role == "write-W1":
            (root / "control").mkdir()
            case = f.components(root / "case", cp, budget)
            runner = f.make_runner(case)
            _matches(root, index, runner, cp)
            with closing(store.create_windowed_device_boundary_store(path, runner=runner,
                    qualification_budget=budget, storage_contract=contract)) as handle:
                parent = handle.head_token()
                assert captured["cursor"] == 0
                ack = handle.step(parent)
                assert ack.sequence == captured["cursor"] == 1
                _control(root, "wind.json", handle, captured,
                         original_parent=parent, original_prefix=captured["receipts"])
            return
        name = "wind.json" if role == "recover-R1" else "water.json"
        prior = _read(root / "control" / name, descriptor["expected_control_sha256"])
        checkpoint._keys(prior, ("schema", "cursor", "store_file_sha256", "head_token",
            "original_parent_token", "original_prefix", "receipt_sha256", "water_counters",
            "complete_receipts", "deep_files", "pid"), "accepted previous role")
        assert prior["schema"] == "masy.windowed_device_restart_control.v1"
        assert prior["cursor"] == (1 if role == "recover-R1" else 2)
        assert prior["store_file_sha256"] == descriptor["expected_store_file_sha256"]
        # A genuine new role is required; PID reuse is checked by Root's birth
        # receipts, not inferred from this diagnostic integer alone.
        assert type(prior["pid"]) is int
        with (mock.patch.object(f, "components", side_effect=AssertionError("restore invoked genesis")),
              closing(store.recover_windowed_device_boundary_store(path,
                  expected_store_file_sha256=descriptor["expected_store_file_sha256"],
                  qualification_budget=budget)) as handle):
            assert captured["cursor"] == prior["cursor"]
            assert handle.head_token() == prior["head_token"]
            assert captured["receipts"] == prior["complete_receipts"]
            assert captured["water_counters"] == prior["water_counters"]
            _check_old_files(prior["deep_files"])
            if role == "recover-R1":
                ack = handle.step(handle.head_token())
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
                    assert ack.sequence == captured["cursor"] == cursor
                    assert captured["receipts"][:2] == prior["complete_receipts"]
                _check_old_files(prior["deep_files"])
                assert handle.report()["accepted"]["status"] == "complete"
                _control(root, "finish.json", handle, captured,
                    original_parent=prior["original_parent_token"], original_prefix=prior["original_prefix"])
