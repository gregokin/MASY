"""Focused metadata/control checks; these issue no Native/GPU authority."""
from __future__ import annotations

import copy
import dataclasses
import os
import sys
from importlib.machinery import ModuleSpec, SourceFileLoader
from pathlib import Path
from types import FunctionType, MappingProxyType, ModuleType, SimpleNamespace

import pytest

from masy import dense_boundary_store as store
from masy import dense_runner_checkpoint as boundary
from masy import windowed_device_checkpoint as small
from masy import windowed_device_c64_checkpoint as c64
from masy.windowed_device_c64_runner import _combine_water_counters

Error = small.WindowedDeviceCheckpointError


def _wire():
    return dict(root=dict(t="dict", i=0, v=[["array", dict(t="array", i=1, v=0)]]),
        arrays=[dict(dtype="<f8", shape=[8, 8, 20, 1], offset=0, bytes=10240, backend="cupy")],
        numeric_bytes=10240)


def test_distinct_limits_keep_original_validators_and_defaults():
    image, limits = c64.C64BoundaryImageLimits(), c64.C64BoundaryStoreLimits()
    assert type(image) is not boundary.BoundaryImageLimits
    assert type(limits) is not store.BoundaryStoreLimits
    assert (image.max_cells, image.max_classes, image.max_voxels, image.max_events) == (64, 1, 20, 4)
    assert limits.max_accepted == 5 and limits.max_store_bytes == 512 << 20
    assert (limits.free_reserve_bytes, limits.free_reserve_inodes) == (1 << 30, 256)
    with pytest.raises(store.BoundaryStoreError):
        store.BoundaryStoreLimits.__post_init__(limits)
    with pytest.raises(Error):
        c64.C64BoundaryStoreLimits.__post_init__(store._DEFAULT)
    assert small._GPU_DEFAULT.image.max_cells == 16 and small._GPU_DEFAULT.image.max_voxels == 6
    assert store._limits_record(store._DEFAULT)["image"]["max_voxels"] == store._DEFAULT.image.max_voxels


@pytest.mark.parametrize("field,value", [("max_cells", 16), ("max_voxels", 6),
    ("max_classes", 2), ("max_events", 3), ("max_cohorts", 3999), ("max_nodes", True),
    ("max_metadata_bytes", (4 << 20)+1), ("native_max_bytes", (64 << 20)+1)])
def test_c64_limits_refuse_wrong_dimensions_bool_and_cap_growth(field, value):
    with pytest.raises(Error):
        dataclasses.replace(c64.C64BoundaryImageLimits(), **{field: value})


def test_five_disjoint_rows_do_not_contaminate_original_table():
    original = small._class_table()
    table = small._class_table_for(c64._DEFAULT.image)
    additions = {module+":"+name: (fields, False) for module, name, fields in c64._C64_TYPES}
    assert len(additions) == 5 and not (set(original) & set(additions))
    assert table == dict(original, **additions)
    assert small._class_table_for(boundary.BoundaryImageLimits()) == original
    assert small._class_table() == original
    with pytest.raises(TypeError):
        small._class_table(None)  # Old unsupported-argument behavior stays exact.


def test_real_c64_shape_wire_keeps_same_numeric_and_structural_caps():
    wire = _wire()
    before = copy.deepcopy(wire)
    assert c64._wire_preflight(wire, c64._DEFAULT.image) == 10240
    record = dict(wire=wire, schema="explicitly-synthetic-metadata", manifest_sha256="0"*64)
    assert c64._plain_record(record, c64._DEFAULT.image) is record
    assert wire == before
    assert c64._wire_json_limits(c64._DEFAULT.image).max_metadata_bytes == 4 << 20
    assert c64._record_json_limits(c64._DEFAULT.image).max_metadata_bytes == 4 << 20


@pytest.mark.parametrize("change", ["bool_extent", "bad_rank", "unknown_dtype", "overflow_payload", "duplicate_key", "unknown_class"])
def test_typed_wire_rejects_malformed_closed_metadata(change):
    wire = _wire()
    if change == "bool_extent":
        wire["arrays"][0]["shape"][0] = True
    elif change == "bad_rank":
        wire["arrays"][0]["shape"] = [8, 8, 20, 1, 1]
    elif change == "unknown_dtype":
        wire["arrays"][0]["dtype"] = "|O"
    elif change == "overflow_payload":
        wire["numeric_bytes"] = (1 << 20)+1
    elif change == "duplicate_key":
        wire["root"]["v"].append(copy.deepcopy(wire["root"]["v"][0]))
    else:
        wire["root"] = dict(t="class", i=0, c="foreign:Undeclared", v=[])
    with pytest.raises(Error):
        c64._wire_preflight(wire, c64._DEFAULT.image)


def test_new_class_construction_requires_genuine_issued_scope():
    with pytest.raises(Error, match="actual current C64"):
        c64._class(c64._C64_TYPES[0][0]+":"+c64._C64_TYPES[0][1], c64._DEFAULT.image)


def test_path_rejects_foreign_conversion_before_calling_it():
    class Foreign:
        def __str__(self):
            raise AssertionError("foreign __str__ executed")
    with pytest.raises(Error):
        c64._path(Foreign())
    assert c64._path(Path("/tmp/explicitly-synthetic-c64-metadata")) == "/tmp/explicitly-synthetic-c64-metadata"
    with pytest.raises(Error):
        c64._path("/tmp/../escape")


def test_baseexception_poisons_one_attempt_and_preserves_primary():
    ticket = c64._RestoreTicket(c64._KEY)
    failure = KeyboardInterrupt("synthetic control interruption; no Native work")
    with pytest.raises(KeyboardInterrupt) as caught:
        with c64._attempt(ticket):
            assert c64._checked_ticket(ticket) is ticket
            raise failure
    assert caught.value is failure and ticket.phase == "poisoned"
    assert c64._ACTIVE_SCOPE is None
    with pytest.raises(Error):
        with c64._attempt(ticket):
            pytest.fail("poisoned ticket reused")


@pytest.mark.parametrize("field,value", [("pid", -1), ("thread", -1), ("phase", "consumed"), ("key", None)])
def test_ticket_scope_identity_refuses_before_construction(field, value):
    ticket = c64._RestoreTicket(c64._KEY)
    setattr(ticket, field, value)
    with pytest.raises(Error):
        with c64._attempt(ticket):
            pytest.fail("invalid ticket entered")
    assert c64._ACTIVE_SCOPE is None


def test_nested_restore_refuses_and_unconsumed_attempt_poisoned():
    ticket = c64._RestoreTicket(c64._KEY)
    with pytest.raises(Error):
        with c64._attempt(ticket):
            with pytest.raises(Error):
                with c64._attempt(c64._RestoreTicket(c64._KEY)):
                    pytest.fail("nested restore entered")
    assert ticket.phase == "poisoned" and c64._ACTIVE_SCOPE is None


def test_foreign_cached_module_never_calls_foreign_mapping_get(monkeypatch):
    name = c64.__name__
    fake = ModuleType(name)
    origin = c64.__file__
    fake.__dict__.update(__file__=origin, __spec__=ModuleSpec(name, SourceFileLoader(name, origin), origin=origin),
                         __loader__=SourceFileLoader(name, origin), _SHARED_STORE=store)
    fake.__dict__["_check_module_image"] = FunctionType(c64._check_module_image.__code__, fake.__dict__)
    class Foreign(dict):
        def get(self, key, default=None):
            raise AssertionError("foreign mapping get executed")
    descriptor = fake.__dict__["_check_module_image"]
    fake.__dict__["_SOURCE_BINDINGS"] = ((MappingProxyType(Foreign()), "foreign", descriptor, descriptor.__code__),)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, name, fake)
        assert store._c64_module_if_loaded() is None
        with pytest.raises(store.BoundaryStoreError):
            store._limits_admit(c64._C64GPUStore, c64._DEFAULT)
    assert sys.modules[name] is c64


def test_original_cpu_and_small_profile_pairs_remain_old_only():
    small_header = dict(schema=store.SCHEMA_WINDOWED_GPU, profile=store.PROFILE_WINDOWED_GPU)
    c64_header = dict(schema=c64.STORE_SCHEMA, profile=c64.PROFILE)
    assert store._windowed_gpu_store(small_header) and not store._c64_gpu_store(small_header)
    assert store._c64_gpu_store(c64_header) and not store._windowed_gpu_store(c64_header)
    with pytest.raises(store.BoundaryStoreError):
        store._store_schema(dict(schema=c64.STORE_SCHEMA, profile=store.PROFILE_WINDOWED_GPU))
    with pytest.raises(store.BoundaryStoreError):
        store._limits_admit(store.DenseBoundaryStore, c64._DEFAULT, c64_header)
    with pytest.raises(store.BoundaryStoreError):
        store._limits_admit(c64._C64GPUStore, store._DEFAULT, small_header)


@pytest.mark.parametrize("counter,ceiling", [("logical_work_debited", "max_work"),
    ("logical_read_bytes_debited", "max_logical_read_bytes"), ("versions_reserved", "max_new_versions"),
    ("maximum_host_predicate", "max_host_bytes"), ("maximum_device_predicate", "max_device_bytes")])
def test_unchanged_counter_fold_enforces_each_of_five_true_ceilings(counter, ceiling):
    previous = dict.fromkeys(c64._COUNTERS, 0)
    entry, event = dict(previous), dict(previous)
    entry[counter] = 3
    limits = SimpleNamespace(max_work=100, max_logical_read_bytes=100, max_new_versions=100,
                            max_host_bytes=100, max_device_bytes=100)
    assert _combine_water_counters(previous, entry, event, limits)[counter] == 3
    setattr(limits, ceiling, 2)
    from masy.dense_runner import DenseRunnerError
    with pytest.raises(DenseRunnerError, match="water_cumulative_ceiling"):
        _combine_water_counters(previous, entry, event, limits)
    assert not any(previous.values()) and entry[counter] == 3


def test_read_frontier_prices_all_named_families_and_three_banks():
    plan = c64._read_plan(c64._DEFAULT.image, 5, 1024)
    assert plan["host_bytes"] == 3*sum(plan["families"].values())
    assert plan["families"]["native_archives"] == 6*(64 << 20)
    assert plan["families"]["deep_metadata"] == 12*(4 << 20)
    assert plan["constructor_allocation"] is plan["external_runtime"] is None
    assert plan["banks"] == ("current", "next", "first_error")
