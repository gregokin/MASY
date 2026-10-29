"""Imported checkpoint metadata/control checks; these issue no execution facts.

Primitive wire/identity examples below are labelled synthetic data, not Native
owners, budgets, grants or a fallback for the protected physical fixtures.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from masy import case_device_c64 as producer
from masy import windowed_device_c64_checkpoint as checkpoint


@pytest.mark.parametrize("raw", [b"{}", b"\x00\xff\x01", bytes(range(256))*8,
    bytes(range(256))*8+b"tail", bytes(range(256))*17])
def test_raw_capsule_round_trips_exact_bytes_with_existing_scalar_limit(raw):
    entry = producer._checkpoint_pack_record(raw, 4 << 20)
    before = copy.deepcopy(entry)
    assert producer._checkpoint_unpack_record(entry, 4 << 20) == raw
    assert entry == before
    assert all(0 < len(chunk) <= 4096 for chunk in entry["hex_chunks"])
    assert all(len(chunk) == 4096 for chunk in entry["hex_chunks"][:-1])


@pytest.mark.parametrize("change", ["bool_size", "wrong_digest", "extra_key", "short_chunk",
    "uppercase", "foreign_sequence", "missing_chunk", "padding"])
def test_raw_capsule_refuses_noncanonical_or_drifted_bytes(change):
    entry = producer._checkpoint_pack_record(bytes(range(256))*9, 4 << 20)
    if change == "bool_size": entry["size"] = True
    elif change == "wrong_digest": entry["sha256"] = "0"*64
    elif change == "extra_key": entry["unused"] = None
    elif change == "short_chunk": entry["hex_chunks"] = (entry["hex_chunks"][0][:-2], *entry["hex_chunks"][1:])
    elif change == "uppercase": entry["hex_chunks"] = tuple(x.upper() for x in entry["hex_chunks"])
    elif change == "foreign_sequence": entry["hex_chunks"] = list(entry["hex_chunks"])
    elif change == "missing_chunk": entry["hex_chunks"] = entry["hex_chunks"][:-1]
    else: entry["hex_chunks"] = (*entry["hex_chunks"][:-1], entry["hex_chunks"][-1]+"00")
    with pytest.raises(producer.CaseDeviceC64Error, match="checkpoint_origin"):
        producer._checkpoint_unpack_record(entry, 4 << 20)


@pytest.mark.parametrize("raw", [b"", bytearray(b"{}"), b"a"*1025])
def test_raw_record_admission_precedes_any_encoding_or_expansion(raw):
    with pytest.raises(producer.CaseDeviceC64Error, match="checkpoint_origin"):
        producer._checkpoint_pack_record(raw, 1024)


def _origin():
    return dict(schema=producer._CHECKPOINT_ORIGIN_SCHEMA,
        **{name: producer._checkpoint_pack_record(b'{"kind":"'+name.encode()+b'"}', 4 << 20)
           for name in ("request", "host", "water")})


def test_capsule_has_a_positive_prospective_frontier_and_keeps_input_unchanged():
    origin = _origin()
    before = copy.deepcopy(origin)
    actual = producer._checkpoint_origin_charge(origin, 4 << 20)
    prospective = producer._checkpoint_restore_charge(4 << 20)
    assert 0 < sum(row["size"] for name, row in origin.items() if name != "schema") < actual <= prospective
    assert producer._metadata_charge(4 << 20) < actual
    assert origin == before
    with pytest.raises(producer.CaseDeviceC64Error, match="unchanged metadata"):
        producer._checkpoint_restore_charge((4 << 20)+1)
    with pytest.raises(producer.CaseDeviceC64Error, match="hex raw capsule"):
        producer._checkpoint_origin_charge(origin, 32)


def test_original_raw_json_is_not_replaced_by_a_reserialized_value():
    raw = b'{"a":-0.0,"b":false}'
    value = producer._checkpoint_json(raw, 4096)
    assert value["a"].hex() == "-0x0.0p+0" and value["b"] is False
    assert producer._bytes(value, 4096) == raw
    for malformed in (b'{ "a":1}', b'{"a":1,"a":2}', b'{"a":NaN}',
                      b'{"a":9223372036854775808}', b'{"a":"\xff"}', b'[]'):
        with pytest.raises(producer.CaseDeviceC64Error, match="checkpoint_origin"):
            producer._checkpoint_json(malformed, 4096)


def _original(schema, source, origin):
    # Explicitly synthetic primitive wire only: no file/provider/owner/budget.
    material = dict(inputs=dict.fromkeys(checkpoint._INPUT_FIELDS), initial_pool=None,
                    initial_soil=None, inventory0=None)
    if origin is not None:
        material["case_origin"] = origin
    encoder = checkpoint._Encoder(checkpoint._DEFAULT.image, np)
    wire = dict(root=encoder.node(material), arrays=encoder.rows, numeric_bytes=encoder.numeric)
    return dict(schema=schema, wire=wire, payload=dict(size=0, sha256="0"*64),
        profile=checkpoint.PROFILE, source_identity=source, environment_sha256="0"*64,
        manifest_sha256="0"*64)


def test_imported_original_root_is_explicit_and_factory_root_stays_exact():
    source = dict(schema=checkpoint.CASE_BUDGET_SCHEMA, synthetic_metadata=True)
    record = _original(checkpoint.CASE_ORIGINAL_SCHEMA, source, _origin())
    checkpoint._original_preflight(record, source, "0"*64, checkpoint._DEFAULT.image)
    factory_source = dict(schema="explicitly_synthetic_factory_metadata")
    factory = _original(checkpoint.ORIGINAL_SCHEMA, factory_source, None)
    checkpoint._original_preflight(factory, factory_source, "0"*64, checkpoint._DEFAULT.image)
    for bad, expected in ((record, factory_source), (factory, source),
                          (_original(checkpoint.CASE_ORIGINAL_SCHEMA, source, None), source)):
        with pytest.raises(checkpoint.codec.WindowedDeviceCheckpointError):
            checkpoint._original_preflight(bad, expected, "0"*64, checkpoint._DEFAULT.image)
    assert checkpoint._ORIGINAL_FIELDS == ("inputs", "initial_pool", "initial_soil", "inventory0")
    assert checkpoint._CASE_ORIGINAL_FIELDS == (*checkpoint._ORIGINAL_FIELDS, "case_origin")


def test_both_origin_kinds_refuse_cross_direction_before_any_loader():
    for imported in (False, True):
        header = dict(runtime_reference=dict(schema=checkpoint.CASE_RUNTIME_SCHEMA if imported else checkpoint.RUNTIME_SCHEMA),
            originals=dict(format=checkpoint.CASE_ORIGINAL_SCHEMA if imported else checkpoint.ORIGINAL_SCHEMA),
            source_identity=dict(schema=checkpoint.CASE_BUDGET_SCHEMA if imported else "synthetic_factory"))
        checkpoint._check_store_kind(header, imported=imported)
        with pytest.raises(checkpoint.codec.WindowedDeviceCheckpointError, match="store_kind"):
            checkpoint._check_store_kind(header, imported=not imported)
        header["originals"]["format"] = "unissued_unknown_schema"
        with pytest.raises(checkpoint.codec.WindowedDeviceCheckpointError, match="store_kind"):
            checkpoint._check_store_kind(header, imported=imported)


def test_imported_identity_changes_only_role_paths_and_preserves_request_caps_and_none():
    # Projection example only. It deliberately is not a complete owning DTO.
    record = dict(schema=checkpoint.CASE_BUDGET_SCHEMA, current_runtime_ref=dict(path="/old", size=1, sha256="1"*64),
        producer_source=dict(path="/old/producer", size=42, sha256="2"*64), producer_module="masy.case_device_c64",
        source_bindings=[dict(module="explicit.example", path="/old/source", size=43, sha256="3"*64)],
        native_source_identity="4"*64, source_manifest_sha256="5"*64, gpu_uuid="explicit_metadata_uuid",
        case_request_binding_sha256="6"*64, execution_budgets=dict(pool_bytes=2 << 30),
        opaque_external_bounds=(("explicit_metadata_role", None),), opaque_primitive_units=(("explicit_metadata_unit", None),),
        opaque_constructor_accounting=None)
    original = copy.deepcopy(record)
    identity = checkpoint._budget_identity(record)
    moved = copy.deepcopy(record)
    moved["current_runtime_ref"] = dict(path="/fresh", size=9, sha256="9"*64)
    moved["producer_source"]["path"] = "/fresh/producer"
    moved["source_bindings"][0]["path"] = "/fresh/source"
    assert checkpoint._budget_identity(moved) == identity
    for field in ("native_source_identity", "source_manifest_sha256", "case_request_binding_sha256", "gpu_uuid"):
        changed = copy.deepcopy(moved); changed[field] += "x"
        assert checkpoint._budget_identity(changed) != identity
    changed = copy.deepcopy(moved); changed["execution_budgets"]["pool_bytes"] += 1
    assert checkpoint._budget_identity(changed) != identity
    assert identity["opaque_constructor_accounting"] is None
    assert identity["opaque_external_bounds"] == [["explicit_metadata_role", None]]
    assert record == original


def test_handoff_latch_is_distinct_from_guard_failure_and_refuses_before_owner_use():
    # Incomplete MASY metadata shell only; no Native state or budget is made.
    kwargs = {f.name: None for f in dataclasses.fields(producer.PreparedDeviceC64Case)
              if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING}
    guarded = producer.PreparedDeviceC64Case(**kwargs, _checkpoint_handed_off=True)
    assert guarded._failed is False
    with pytest.raises(producer.CaseDeviceC64Error, match="case_handed_off"):
        guarded.verify()
    with pytest.raises(producer.CaseDeviceC64Error, match="case_handed_off"):
        guarded.step()
    assert guarded._failed is False


def test_restore_mint_refuses_an_unissued_ticket_before_native_lookup():
    with pytest.raises(checkpoint.codec.WindowedDeviceCheckpointError, match="ticket"):
        producer._mint_checkpoint_admission(object(), object(), object(), object(), {})


def test_codec_decode_retains_ordered_capsule_and_tuple_chunks_without_native_owner():
    origin = _origin()
    raw = producer._bytes({"text": "a"*4097}, 4 << 20)
    origin["request"] = producer._checkpoint_pack_record(raw, 4 << 20)
    encoder = checkpoint._Encoder(checkpoint._DEFAULT.image, np)
    wire = dict(root=encoder.node(origin), arrays=encoder.rows, numeric_bytes=encoder.numeric)
    assert encoder.numeric == 0 and encoder.arrays == []
    decoded = checkpoint._decode(wire, np.empty(0, dtype=np.uint8),
        checkpoint._DEFAULT.image, np, host=True)
    assert decoded == origin
    assert tuple(decoded) == ("schema", "request", "host", "water")
    for name in ("request", "host", "water"):
        assert type(decoded[name]["hex_chunks"]) is tuple
        assert producer._checkpoint_unpack_record(decoded[name], 4 << 20) == producer._checkpoint_unpack_record(origin[name], 4 << 20)
    assert producer._checkpoint_unpack_record(decoded["request"], 4 << 20) == raw
    assert producer._checkpoint_origin_charge(decoded, 4 << 20) > 0


def test_public_create_refuses_absent_prepared_case_before_any_authority_or_path_use():
    with pytest.raises(producer.CaseDeviceC64Error, match="checkpoint_handoff"):
        producer.create_device_c64_case_boundary_store(None, prepared_case=None,
            qualification_budget=None, bridge_limits=None, native_source_refs=None,
            storage_contract=None)


def test_current_producer_binding_check_is_metadata_only_and_origin_closed():
    assert producer._checkpoint_source_check(producer.__file__) > 0
    with pytest.raises(producer.CaseDeviceC64Error, match="checkpoint_source"):
        producer._checkpoint_source_check(producer.__file__+".unissued")
