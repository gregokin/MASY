"""Five-way dispatch, immutable metadata and exactly-once receipt checks.

No budget, Native bed, CUDA result or physical restore is fabricated here.
Illustrative records exercise validation before any owning constructor.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from masy import dense_boundary_store as s
from masy import windowed_device_checkpoint as c
from masy.clock import ClockLedger, encode_time
from masy.dense_schedule import DenseSchedule, resolve_event


L = c._GPU_DEFAULT.image
SOURCE = {"illustrative_Source_identity": "a" * 64}
ENVIRONMENT = "b" * 64


def _schedule():
    wind = dict(kind="wind", duration_s=1.0, shear_velocity_m_s=0.5,
                direction_deg_cw_from_north=90.0)
    rain = dict(kind="rain", recession_guard_s=180.0,
                intervals=[dict(duration_s=60.0, intensity_mm_h=72.0)])
    return DenseSchedule.from_input(dict(schema="masy.dense_schedule.v1",
                                         events=[wind, rain, wind, rain]))


def _root(names):
    return {"root": {"t": "dict", "i": 0,
        "v": [[name, {"t": "scalar", "v": None}] for name in names]},
        "arrays": [], "numeric_bytes": 0}


def _reference(fmt, relative):
    return dict(format=fmt, relative_path=relative,
                manifest_file_sha256="c" * 64, identity="d" * 64)


def _manifest(k=0):
    ledger = ClockLedger(7.25)
    receipts = []
    for i in range(k):
        clock = ledger.append("event", native_start=7.25 + i, native_end=8.25 + i,
            commit_before=7.25 + i, commit_after=8.25 + i, returned_elapsed=1.0)
        body = dict(schema="masy.windowed_device_event_receipt.v1", index=i,
            kind="wind" if i % 2 == 0 else "rain", status="accepted", code="accepted",
            clock=clock, resolved=resolve_event(_schedule(), i, 7.25 + i))
        raw = c._json(body)
        receipts.append(dict(body=body, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    return dict(schema=c.SCHEMA, profile=c.PROFILE, wire=_root(c._CURRENT_FIELDS),
        payload=dict(size=128, sha256="e" * 64), source_identity=copy.deepcopy(SOURCE),
        environment_sha256=ENVIRONMENT,
        originals=_reference(c.ORIGINAL_SCHEMA, "originals/o-" + "0" * 32),
        genesis=_reference(c.PHYSICAL_SCHEMA, "generations/g-0000-" + "1" * 32 + "/physical"),
        physical=_reference(c.PHYSICAL_SCHEMA, f"generations/g-{k:04d}-" + "1" * 32 + "/physical"),
        case_sha256="f" * 64, schedule_sha256=_schedule().digest,
        state=dict(next_index=k, n_accepted=k, native_time=encode_time(7.25 + k),
                   accepted_bed_label=f"illustrative-generation-{k}"),
        routing=None, receipts=receipts, ledger=ledger.as_record(),
        water_counters=dict.fromkeys(c._COUNTERS, 0), manifest_sha256="a" * 64)


def _preflight(value):
    return c._preflight(value, SOURCE, ENVIRONMENT, L)


@pytest.mark.parametrize("schema,profile", [
    (s.SCHEMA, s.PROFILE), (s.SCHEMA_V2, s.PROFILE_V2),
    (s.SCHEMA_ACTIVE_WATER, s.PROFILE_ACTIVE_WATER),
    (s.SCHEMA_ACTIVE_PROCESS, s.PROFILE_ACTIVE_PROCESS),
    (c.STORE_SCHEMA, c.PROFILE),
])
def test_exact_old_four_and_new_fifth_pairs_dispatch(schema, profile):
    assert s._store_schema(dict(schema=schema, profile=profile)) == schema


def test_all_mixed_pairs_and_unknown_profiles_refuse():
    pairs = [(s.SCHEMA, s.PROFILE), (s.SCHEMA_V2, s.PROFILE_V2),
        (s.SCHEMA_ACTIVE_WATER, s.PROFILE_ACTIVE_WATER),
        (s.SCHEMA_ACTIVE_PROCESS, s.PROFILE_ACTIVE_PROCESS), (c.STORE_SCHEMA, c.PROFILE)]
    for i, (schema, _) in enumerate(pairs):
        for j, (_, profile) in enumerate(pairs):
            if i != j:
                with pytest.raises(s.BoundaryStoreError, match="store_profile"):
                    s._store_schema(dict(schema=schema, profile=profile))
    with pytest.raises(s.BoundaryStoreError, match="store_profile"):
        s._store_schema(dict(schema=c.STORE_SCHEMA, profile="any-gpu"))


def test_gpu_does_not_fall_through_to_dense_preparation_without_authority(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("a missing authority must refuse before decoding/Native preparation")
    monkeypatch.setattr(c, "_prepare", denied)
    with pytest.raises(s.BoundaryStoreError, match="fresh_role_authority"):
        s._prepare_for(dict(schema=c.STORE_SCHEMA, profile=c.PROFILE))


def test_active_borrow_and_checkpoint_receivers_stay_cpu_only():
    metadata_only = object.__new__(c._GPUStore)
    metadata_only._header = dict(schema=c.STORE_SCHEMA, profile=c.PROFILE)
    with pytest.raises(s.BoundaryStoreError, match="water_parent"):
        s.borrow_water_parent(metadata_only, expected_parent="1" * 64)
    for function in (metadata_only.checkpoint_water, metadata_only.checkpoint_wind):
        with pytest.raises(s.BoundaryStoreError, match="store_profile"):
            function("1" * 64, boundary_ordinal=0)
    with pytest.raises(s.BoundaryStoreError, match="store_profile"):
        metadata_only.acknowledge("1" * 64, request_kind="event")


def test_valid_metadata_keeps_generation_label_distinct_from_content_sha():
    manifest = _manifest(1)
    assert _preflight(manifest) is None
    summary = s._summary(manifest)
    assert summary["accepted_bed_label"] == "illustrative-generation-1"
    assert "bed_sha256" not in summary


@pytest.mark.parametrize("mutate,code", [
    (lambda m: m.update(extra=1), "closed_record"),
    (lambda m: m.update(profile="default_dense_numpy_accepted_v1"), "image_identity"),
    (lambda m: m.update(source_identity={"different": "Source"}), "image_identity"),
    (lambda m: m.update(environment_sha256="0" * 64), "image_identity"),
    (lambda m: m["state"].update(next_index=True), "integer"),
    (lambda m: m["state"].update(n_accepted=False), "cursor"),
    (lambda m: m["state"].update(native_time=encode_time(8.5)), "clock"),
    (lambda m: m["receipts"][0].update(bytes=1), "receipt"),
    (lambda m: m["receipts"][0].update(sha256="0" * 64), "receipt"),
    (lambda m: m["water_counters"].update(gate_count=True), "integer"),
    (lambda m: m["wire"]["root"]["v"].pop(), "closed_record"),
])
def test_corrupt_identity_cursor_receipt_and_owner_roster_refuse(mutate, code):
    manifest = _manifest(1); mutate(manifest)
    with pytest.raises(ValueError, match=code):
        _preflight(manifest)


def test_resealed_receipt_cannot_change_its_clock_kind_or_original_schedule():
    for field, value in (("clock", {}), ("kind", "rain"), ("index", True)):
        manifest = _manifest(1)
        row = manifest["receipts"][0]; row["body"][field] = value
        raw = c._json(row["body"]); row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        with pytest.raises(ValueError, match="receipt"):
            _preflight(manifest)
    manifest = _manifest(1)
    row = manifest["receipts"][0]; row["body"]["resolved"]["schedule_sha256"] = "0" * 64
    raw = c._json(row["body"]); row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="receipt"):
        _preflight(manifest)


@pytest.mark.parametrize("mutate,code", [
    (lambda body: body.pop("resolved"), "receipt"),
    (lambda body: body.update(resolved=None), "receipt"),
    (lambda body: body.update(resolved=[]), "receipt"),
    (lambda body: body.update(resolved="not a record"), "receipt"),
    (lambda body: body.update(resolved=1), "receipt"),
    (lambda body: body.update(resolved=True), "receipt"),
    (lambda body: body.update(resolved={}), "identity"),
    (lambda body: body["resolved"].pop("schedule_sha256"), "identity"),
])
def test_resealed_malformed_resolved_record_has_closed_refusal(mutate, code):
    manifest = _manifest(1)
    assert _preflight(manifest) is None
    row = manifest["receipts"][0]
    mutate(row["body"])
    raw = c._json(row["body"])
    row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(c.WindowedDeviceCheckpointError) as refusal:
        _preflight(manifest)
    assert refusal.value.code == code


@pytest.mark.parametrize("index,start", [(0, 7.5), (2, 7.25), (1, 7.25)])
def test_resealed_valid_resolution_must_join_actual_event_and_clock(index, start):
    # Each replacement is a valid independently re-derived schedule record;
    # it is nevertheless not the actual accepted event/ClockLedger row.
    manifest = _manifest(1)
    row = manifest["receipts"][0]
    row["body"]["resolved"] = resolve_event(_schedule(), index, start)
    raw = c._json(row["body"])
    row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="receipt"):
        _preflight(manifest)


def test_original_capsule_requires_all_inputs_and_same_immutable_identity():
    original = dict(schema=c.ORIGINAL_SCHEMA, profile=c.PROFILE, wire=_root(c._ORIGINAL_FIELDS),
        payload=dict(size=128, sha256="e" * 64), source_identity=copy.deepcopy(SOURCE),
        environment_sha256=ENVIRONMENT, manifest_sha256="f" * 64)
    original["wire"]["root"]["v"][0][1] = _root(c._INPUT_FIELDS)["root"]
    original["wire"]["root"]["v"][0][1]["i"] = 1
    c._original_preflight(original, SOURCE, ENVIRONMENT, L)
    bad = copy.deepcopy(original); bad["wire"]["root"]["v"][0][1]["v"].pop()
    with pytest.raises(ValueError, match="original_inputs"):
        c._original_preflight(bad, SOURCE, ENVIRONMENT, L)
    bad = copy.deepcopy(original); bad["source_identity"] = {"new": "source"}
    with pytest.raises(ValueError, match="original_identity"):
        c._original_preflight(bad, SOURCE, ENVIRONMENT, L)


def test_role_path_changes_leave_stable_science_but_source_or_caps_changes_do_not():
    # Plain metadata projection only; _Authority still requires actual budget,
    # real references, loaded origins and Root's separately consumed launch.
    old = dict(current_runtime_ref={"path": "/old/role", "size": 1, "sha256": "1" * 64},
        factory_source={"path": "/old/factory", "size": 17, "sha256": "2" * 64},
        source_bindings=[dict(module="selected.module", path="/old/module", size=19, sha256="3" * 64)],
        gpu_uuid="GPU-selected", fixed_budgets={"active_pool_limit_bytes": 2 << 30},
        native_source_identity="4" * 64)
    fresh = copy.deepcopy(old); fresh["current_runtime_ref"]["path"] = "/new/role"
    fresh["factory_source"]["path"] = "/new/factory"
    fresh["source_bindings"][0]["path"] = "/new/module"
    assert c._budget_identity(old) == c._budget_identity(fresh)
    for field in ("gpu_uuid", "native_source_identity"):
        bad = copy.deepcopy(fresh); bad[field] += "changed"
        assert c._budget_identity(old) != c._budget_identity(bad)
    fresh["fixed_budgets"]["active_pool_limit_bytes"] += 1
    assert c._budget_identity(old) != c._budget_identity(fresh)


def test_duplicate_parent_returns_exact_accepted_ack_without_loader_or_event(monkeypatch):
    monkeypatch.setattr(c, "_prepare", lambda *a, **k: pytest.fail("duplicate must not load"))
    manifest = _manifest(1)
    store_id = "3" * 32
    parent = dict(store_id=store_id, head_sha256="1" * 64, sequence=0)
    child = dict(store_id=store_id, head_sha256="2" * 64, sequence=1)
    operation = s._operation(store_id, parent, 0, manifest["schedule_sha256"])
    generation = dict(parent_head_record=parent, operation_id=operation)
    chain = [(child, generation, manifest), (parent, {}, _manifest())]
    metadata_only = SimpleNamespace(_header=dict(schema=c.STORE_SCHEMA, profile=c.PROFILE, store_id=store_id))
    ack = s.DenseBoundaryStore._duplicate(metadata_only, s._token(parent), chain)
    assert ack.operation_id == operation and ack.head_token == s._token(child)
    assert ack.receipt_bytes == c._json(manifest["receipts"][0]["body"])


def test_boundaries_use_existing_caps_and_keep_opaque_costs_unknown():
    reservation = c._reservation(c._GPU_DEFAULT, 1 << 28)
    assert 0 < reservation["disk_bytes"] <= 512 << 20
    assert reservation["disk_bytes"] == (216 << 20) + (224 << 10)
    assert reservation["disk_bytes"] + (72 << 20) == (288 << 20) + (224 << 10)
    assert reservation["disk_bytes"] + (72 << 20) <= 512 << 20
    assert 0 < reservation["memory_bytes"] <= 5 << 30
    assert reservation["constructor_allocation"] is reservation["external_runtime"] is None
    with pytest.raises(ValueError, match="memory_cap"):
        c._reservation(c._GPU_DEFAULT, 5 << 30)
    assert c._GPU_DEFAULT.free_reserve_bytes == s._DEFAULT.free_reserve_bytes
    assert c._GPU_DEFAULT.free_reserve_inodes == s._DEFAULT.free_reserve_inodes


def _bookkeeping_unit():
    manifest = _manifest()
    inputs = dict.fromkeys(c._INPUT_FIELDS)
    inputs.update(schedule=_schedule().as_record(), context={"bridge_limits": SimpleNamespace(
        max_work=100, max_logical_read_bytes=100, max_new_versions=100)})
    totals = {name: np.zeros(1, dtype=np.float64) for name in
              ("export", "input", "bound", "internal_terminal_deposition")}
    totals.update(water=dict.fromkeys(("rain_m3", "drainage_m3", "storm_export_m3",
        "dry_surface_removed_m3", "dry_soil_removed_m3"), 0.0), n_entry_reroutes=0, n_event_reroutes=0)
    book = dict(next_index=0, n_accepted=0, native_time=encode_time(7.25), ledger=manifest["ledger"],
        inventory0=np.zeros(1), totals=totals, receipts=[], water_counters=dict.fromkeys(c._COUNTERS, 0),
        water_record=None, bed_generation=0, resource_generation=0, routing_issue_counter=0,
        accepted_bed_label=manifest["state"]["accepted_bed_label"], accepted_private_label="illustrative-only")
    current = dict(auxiliary=dict(ledger=None, committed_topography=SimpleNamespace(last_commit_time_s=7.25),
        reconciliation_baseline=None), pool=None, soil=None, bookkeeping=book, routing=None)
    return inputs, current, manifest, dict(time_s=7.25, step=0)


def test_logical_generations_and_native_commit_time_must_match_cursor():
    assert c._validate_book(*_bookkeeping_unit()).digest == _schedule().digest
    for field in ("bed_generation", "resource_generation", "n_accepted"):
        inputs, current, manifest, snapshot = _bookkeeping_unit()
        current["bookkeeping"][field] = 1
        with pytest.raises(ValueError, match="bookkeeping"):
            c._validate_book(inputs, current, manifest, snapshot)
    inputs, current, manifest, snapshot = _bookkeeping_unit()
    current["auxiliary"]["committed_topography"].last_commit_time_s = np.nextafter(7.25, 8.0)
    with pytest.raises(ValueError, match="clock"):
        c._validate_book(inputs, current, manifest, snapshot)


def test_original_work_ceiling_and_cumulative_state_are_not_reset_or_widened():
    inputs, current, manifest, snapshot = _bookkeeping_unit()
    current["bookkeeping"]["water_counters"]["logical_work_debited"] = 1
    manifest["water_counters"]["logical_work_debited"] = 1
    with pytest.raises(ValueError, match="totals"):
        c._validate_book(inputs, current, manifest, snapshot)
    inputs, current, manifest, snapshot = _bookkeeping_unit()
    current["bookkeeping"]["routing_issue_counter"] = 1
    with pytest.raises(ValueError, match="routing"):
        c._validate_book(inputs, current, manifest, snapshot)


@pytest.mark.parametrize("counter,ceiling", [
    ("logical_work_debited", "max_work"),
    ("logical_read_bytes_debited", "max_logical_read_bytes"),
    ("versions_reserved", "max_new_versions"),
])
def test_consistent_receipt_fold_preserves_each_original_cumulative_ceiling(counter, ceiling):
    # Illustrative metadata, no Native result: one wind and one rain receipt
    # must fold exactly before the owning cumulative-ceiling comparison runs.
    def accepted_book(amount):
        inputs, current, _manifest0, _snapshot0 = _bookkeeping_unit()
        manifest = _manifest(2)
        receipts = [copy.deepcopy(row["body"]) for row in manifest["receipts"]]
        zero = [0.0]
        receipts[0].update(classes=dict(boundary_export_kg=zero, boundary_input_kg=zero,
            inventory_bound_kg=zero), finalization=dict(deposited_mass_by_class_kg=zero))
        entry, event = dict.fromkeys(c._COUNTERS, 0), dict.fromkeys(c._COUNTERS, 0)
        entry[counter], event[counter] = 40, amount-40
        cumulative = dict.fromkeys(c._COUNTERS, 0)
        cumulative[counter] = amount
        water_record = dict(entry=entry, event=event, cumulative=cumulative)
        receipts[1].update(classes=dict(export_kg=zero, import_kg=zero, inventory_bound_kg=zero,
            internal_terminal_deposition_kg=zero),
            water=dict(rain_m3=0.0, drainage_m3=0.0, export_m3=0.0,
                dry_surface_removed_m3=0.0, dry_soil_removed_m3=0.0),
            routing=dict(entry_rerouted=False, n_reroutes_in_event=0),
            device_water_qualification=water_record)
        for row, body in zip(manifest["receipts"], receipts, strict=True):
            raw = c._json(body)
            row.update(body=body, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        manifest["water_counters"] = copy.deepcopy(cumulative)
        manifest["routing"] = dict(source_sha256="a" * 64)
        book = current["bookkeeping"]
        book.update(next_index=2, n_accepted=2, bed_generation=2, resource_generation=2,
            native_time=manifest["state"]["native_time"], ledger=manifest["ledger"],
            receipts=receipts, water_counters=copy.deepcopy(cumulative), water_record=water_record,
            routing_issue_counter=1, accepted_bed_label=manifest["state"]["accepted_bed_label"])
        current["auxiliary"]["committed_topography"].last_commit_time_s = 9.25
        current["routing"] = dict(source=np.zeros((4, 4)), velocity=np.zeros((4, 4, 1)),
            phase=None, generation=1, terrain_sha256="a" * 64, cache_fingerprints=[])
        assert getattr(inputs["context"]["bridge_limits"], ceiling) == 100
        assert _preflight(manifest) is None
        return inputs, current, manifest, dict(time_s=9.25, step=2)

    # At-limit acceptance proves none of the earlier joins masks this branch.
    assert c._validate_book(*accepted_book(100)).digest == _schedule().digest
    with pytest.raises(c.WindowedDeviceCheckpointError) as refusal:
        c._validate_book(*accepted_book(101))
    assert refusal.value.code == "water_cumulative_ceiling"
    assert refusal.value.detail == counter
