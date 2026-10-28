"""Codec/data refusals only: no Native constructor, provider or GPU fixture.

The small wire records here are illustrative codec units, never a purported
physical checkpoint, qualified budget, Native owner or successful model run.
"""
from __future__ import annotations

import copy
import hashlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from masy import windowed_device_checkpoint as c
from masy.dense_runner_checkpoint import BoundaryImageLimits


LIMITS = BoundaryImageLimits(max_cells=16, max_classes=1, max_voxels=6,
                             max_cohorts=4000, max_events=4)


def scalar(value):
    return {"t": "scalar", "v": value}


def wire(root=None, arrays=None, numeric=0):
    return {"root": scalar(None) if root is None else root,
            "arrays": [] if arrays is None else arrays, "numeric_bytes": numeric}


def array_wire():
    return wire({"t": "array", "i": 0, "v": 0},
        [{"dtype": "<f8", "shape": [2, 2], "offset": 0,
          "bytes": 32, "backend": "cupy"}], 32)


def refused(value, match):
    with pytest.raises(c.WindowedDeviceCheckpointError, match=match):
        c._wire_preflight(value, LIMITS)


def test_array_metadata_is_priced_without_import_or_placement(monkeypatch):
    def no_import(*args, **kwargs):
        raise AssertionError("a wire preflight must not import a class/provider")
    monkeypatch.setattr(c.importlib, "import_module", no_import)
    assert c._wire_preflight(array_wire(), LIMITS) == 32
    assert c._wire_preflight(wire(), LIMITS) == 0


@pytest.mark.parametrize("field,value", [
    ("dtype", "|O"), ("dtype", ">f8"), ("backend", "jax"),
    ("shape", [2, True]), ("shape", [4001]), ("shape", [1]*5),
    ("offset", 1), ("offset", False), ("bytes", 31), ("bytes", True),
])
def test_bad_leaf_layout_refuses_before_expansion(field, value):
    value_wire = array_wire()
    value_wire["arrays"][0][field] = value
    refused(value_wire, "array_layout|integer")


def test_aggregate_numeric_and_unreferenced_rows_are_not_hidden():
    value = array_wire(); value["numeric_bytes"] = 31
    refused(value, "numeric_cap")
    value = array_wire(); value["root"] = scalar(None)
    refused(value, "array_layout")
    value = wire(numeric=LIMITS.max_numeric_bytes + 1)
    refused(value, "integer")


def test_alias_is_backward_only_and_array_definitions_are_unique():
    good = array_wire()
    good["root"] = {"t": "tuple", "i": 0, "v": [
        {"t": "array", "i": 1, "v": 0}, {"t": "ref", "v": 1}]}
    assert c._wire_preflight(good, LIMITS) == 32
    bad = copy.deepcopy(good); bad["root"]["v"].reverse()
    refused(bad, "reference")
    bad = copy.deepcopy(good); bad["root"]["v"][1] = {"t": "array", "i": 2, "v": 0}
    refused(bad, "array_layout")
    refused(wire({"t": "tuple", "i": 0, "v": [{"t": "ref", "v": 0}]}), "reference")


def test_unknown_executable_class_and_module_are_refused_before_import(monkeypatch):
    monkeypatch.setattr(c.importlib, "import_module", lambda *a: pytest.fail("unexpected import"))
    refused(wire({"t": "class", "i": 0, "c": "os:system", "v": []}), "class_fields")
    refused(wire({"t": "module", "v": "builtins"}), "module")
    with pytest.raises(c.WindowedDeviceCheckpointError, match="class"):
        c._class("os:system")


def test_frozen_class_field_order_and_duplicate_dictionary_keys_refuse():
    tag = "masy.windowed_device_bed:DeviceWindowedBedLimits"
    names = c._class_table()[tag][0]
    root = {"t": "class", "i": 0, "c": tag,
            "v": [[name, scalar(1)] for name in names]}
    assert c._wire_preflight(wire(root), LIMITS) == 0
    root["v"].reverse()
    refused(wire(root), "class_fields")
    refused(wire({"t": "dict", "i": 0, "v": [["x", scalar(1)], ["x", scalar(2)]]}), "class_fields")


@pytest.mark.parametrize("value", ["nan", "inf", "0x1p+0", "0X1.0000000000000P+0", "not-a-float"])
def test_lossy_or_noncanonical_scalar_floats_refuse(value):
    refused(wire({"t": "float", "v": value}), "float")


def test_scalar_bits_and_integer_domain_remain_distinct():
    assert c._finite_hex((-0.0).hex()).hex() == (-0.0).hex()
    assert c._finite_hex((0.1).hex()).hex() == (0.1).hex()
    assert c._wire_preflight(wire(scalar((1 << 128) - 1)), LIMITS) == 0
    refused(wire(scalar(1 << 128)), "scalar")
    refused(wire(scalar(0.1)), "scalar")


@pytest.mark.parametrize("dtype,value", [
    ("<i4", scalar(True)), ("<i4", scalar(1 << 31)),
    ("<i8", {"t": "float", "v": (1.0).hex()}),
    ("|b1", scalar(1)), ("<f8", scalar(1)),
    ("<f8", {"t": "list", "i": 0, "v": []}),
])
def test_numpy_scalar_tag_cannot_hide_a_holder_or_change_kind(dtype, value):
    refused(wire({"t": "numpy_scalar", "dtype": dtype, "v": value}), "scalar")


def test_host_encoder_preserves_aliases_and_refuses_a_cycle():
    # A non-provider sentinel: the test exercises only genuine NumPy inputs.
    not_a_provider = SimpleNamespace(ndarray=type("NeverAnArray", (), {}))
    encoder = c._Encoder(LIMITS, not_a_provider)
    array = np.array([0.1, -0.0], dtype=np.float64)
    root = encoder.node((array, array))
    assert encoder.numeric == array.nbytes and len(encoder.rows) == 1
    assert root["v"][1] == {"t": "ref", "v": root["v"][0]["i"]}
    assert c._wire_preflight(wire(root, encoder.rows, encoder.numeric), LIMITS) == 16
    cycle = []; cycle.append(cycle)
    with pytest.raises(c.WindowedDeviceCheckpointError, match="cycle"):
        c._Encoder(LIMITS, not_a_provider).node(cycle)


def _raw_container(path, *, header=None, payload=b"", raw_override=None):
    """Container-format unit only; no typed model/budget is asserted."""
    path.mkdir()
    if raw_override is None:
        buffer = io.BytesIO()
        if header is None:
            np.save(buffer, np.frombuffer(payload, dtype=np.uint8), allow_pickle=False)
        else:
            np.lib.format.write_array_header_1_0(buffer, header)
            buffer.write(payload)
        raw = buffer.getvalue()
    else:
        raw = raw_override
    (path / "a_0000.npy").write_bytes(raw)
    record = {"schema": c.SCHEMA, "wire": wire(),
              "payload": {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
    record["manifest_sha256"] = c._sha(record)
    (path / "manifest.json").write_bytes(c._json(record))
    return record


def test_raw_payload_hash_is_checked_before_any_numpy_expansion(tmp_path, monkeypatch):
    path = tmp_path / "raw"; _raw_container(path)
    old = (path / "a_0000.npy").read_bytes()
    (path / "a_0000.npy").write_bytes(old[:-1] + bytes([old[-1] ^ 1]))
    monkeypatch.setattr(np, "fromfile", lambda *a, **k: pytest.fail("expanded corrupt bytes"))
    with pytest.raises(c.WindowedDeviceCheckpointError, match="file_changed"):
        c._read_wire(path, schema=c.SCHEMA, limits=LIMITS, expand=True)


@pytest.mark.parametrize("header", [
    {"descr": "|u1", "fortran_order": False, "shape": (1 << 30,)},
    {"descr": "|O", "fortran_order": False, "shape": (0,)},
    {"descr": "|u1", "fortran_order": True, "shape": (0,)},
])
def test_authenticated_bad_npy_header_is_refused_before_allocation(tmp_path, monkeypatch, header):
    path = tmp_path / "raw"; _raw_container(path, header=header)
    monkeypatch.setattr(np, "fromfile", lambda *a, **k: pytest.fail("expanded invalid header"))
    with pytest.raises(ValueError):
        c._read_wire(path, schema=c.SCHEMA, limits=LIMITS, expand=True)


def test_manifest_byte_hash_and_semantic_seal_are_independent(tmp_path):
    path = tmp_path / "raw"; record = _raw_container(path)
    with pytest.raises(c.WindowedDeviceCheckpointError, match="manifest_hash"):
        c._read_wire(path, expected_file_sha="0" * 64, schema=c.SCHEMA, limits=LIMITS, expand=False)
    record["wire"]["root"] = scalar("changed")
    (path / "manifest.json").write_bytes(c._json(record))
    with pytest.raises(c.WindowedDeviceCheckpointError, match="manifest_hash"):
        c._read_wire(path, schema=c.SCHEMA, limits=LIMITS, expand=False)


def test_reference_and_ticket_cannot_decode_an_execution_capability():
    ref = dict(format=c.PHYSICAL_SCHEMA, relative_path="../outside",
               manifest_file_sha256="0" * 64, identity="1" * 64)
    with pytest.raises(c.WindowedDeviceCheckpointError, match="artifact_path"):
        c._reference(ref, c.PHYSICAL_SCHEMA)
    with pytest.raises(c.WindowedDeviceCheckpointError, match="ticket"):
        c._RestoreTicket(object())
    with pytest.raises(c.WindowedDeviceCheckpointError, match="ticket"):
        c._checked_ticket(SimpleNamespace(key=c._KEY, consumed=False))
