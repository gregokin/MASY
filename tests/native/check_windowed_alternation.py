"""Affected Source-authored checks; unexecuted by author, Root runs serially.

Run the ownership/refusal cases first, then the two-event gate, then four
alternating events only after the first physical gate is independently accepted.
The oracle invokes actual Native APIs directly on an independent case. This is
same-backend CPU interface evidence, not GPU/restart/scale/calibration evidence.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path

import fixture_windowed_alternation as fixture
import numpy as np
import pytest
from maple.core.types.cohort import zeros_mobile_cohort_state
from maple.surface.voxels.deep_store import DeepJournal
from maple.surface.voxels.windowed_inventory import gather_full_columns, validate_windowed_column_full
from maple.surface.voxels.windowed_pending import attach_pending_journal, validate_pending_binding
from maple.surface.voxels.windowed_prefix import rebind_prefix_packets
from maple_syrup.sediment_bed import BedIntegrationError
from maple_syrup.windowed_shared_bed import WindowedSharedBedState

from masy import windowed_bed as holder
from masy import windowed_runner as adapter
from masy.dense_bed import _leaves
from masy.dense_runner import DenseRunnerError

_TIMING = ("first_step_wall_s", "first_step_cpu_s", "remaining_wall_s", "remaining_cpu_s", "commit_wall_s")


def same(a, b, path="value"):
    """Complete exact same-backend typed values; no user epsilon or broad filter."""
    assert type(a) is type(b), path
    if type(a) is np.ndarray:
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert a.tobytes(order="C") == b.tobytes(order="C"), path
    elif dataclasses.is_dataclass(a) and not isinstance(a, type):
        for field in dataclasses.fields(a):
            same(getattr(a, field.name), getattr(b, field.name), path + "." + field.name)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            same(a[key], b[key], path + "." + str(key))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            same(x, y, path + "[" + str(i) + "]")
    elif type(a) is float:
        assert a.hex() == b.hex(), path
    else:
        assert a == b, path


def files(root):
    """Bounded test-only old-file integrity; no deep-store snapshot/rollback."""
    root = Path(root)
    rows = {}
    total = 0
    for path in sorted(root.rglob("*")):
        assert not path.is_symlink()
        if path.is_file():
            assert len(rows) < 4096
            size = path.stat().st_size
            total += size
            assert total <= 64 << 20
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(65536):
                    hasher.update(block)
            rows[str(path)] = (size, hasher.hexdigest())
    return rows


def retained_files(rows):
    for name, expected in rows.items():
        path = Path(name)
        assert path.exists() and path.stat().st_size == expected[0]
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(65536):
                hasher.update(block)
        assert hasher.hexdigest() == expected[1]


def arrays(value):
    leaves = []
    _leaves(value, "", leaves)
    return [a for _, a in leaves]


def same_bed(a, b, case, oracle):
    assert type(a) is type(b) is WindowedSharedBedState
    cells = np.arange(16, dtype=np.int64)
    # Tiny acceptance-time read only; product/transition never materializes depth.
    same(gather_full_columns(a.voxel_column, cells), gather_full_columns(b.voxel_column, cells), "logical columns")
    for name in holder.WINDOWED_FIELDS[1:]:
        same(getattr(a, name), getattr(b, name), "bed." + name)
    for name in ("window_kg", "base", "deep_by_class_kg", "deep_total_kg", "deep_room_kg"):
        same(getattr(a.voxel_column, name), getattr(b.voxel_column, name), "column." + name)
    assert (a.voxel_column.pending_deep is None) is (b.voxel_column.pending_deep is None)
    if a.voxel_column.pending_deep is not None:
        same(a.voxel_column.pending_deep.journal, b.voxel_column.pending_deep.journal, "pending")
        for column in (a.voxel_column, b.voxel_column):
            # Exact owning binding fields: journal, version and the five
            # resident references; version paths differ between role stores.
            validate_pending_binding(column, operation="MASY850 complete comparison")
            assert column.pending_deep.version is column.deep_store
            assert len(column.pending_deep.arrays) == 5
            for name, value in zip(holder._COLUMN_ARRAYS, column.pending_deep.arrays, strict=True):
                assert getattr(column, name) is value
    # Distinct Root-owned stores are expected; versions remain actual Native
    # results. A no-op publication need not advance a version (832 Source law).
    assert a.voxel_column.deep_store.store_dir != b.voxel_column.deep_store.store_dir
    for state, c in ((a, case), (b, oracle)):
        validate_windowed_column_full(state.voxel_column, c.bed.context.geometry,
                                     c.bed.context.mass_resolution_kg, class_ids=c.fx.ids)


def test_closed_constructors_and_complete_inherited_protocol():
    with pytest.raises(holder.WindowedBedError, match="private_constructor"):
        holder.WindowedBed()
    with pytest.raises(DenseRunnerError, match="private_constructor"):
        adapter.WindowedRunner()
    adapter._require_complete_protocol()
    assert adapter.WindowedRunner._finish is adapter.DenseRunner._finish
    assert adapter.WindowedRunner._precheck is adapter.DenseRunner._precheck
    assert adapter.WindowedOwnership.prepare_bed is adapter._HostOwnership.prepare_bed


def test_metadata_reservation_is_positive_and_keeps_work_separate():
    terms = dict(adapter._metadata_terms(np, ({"key": (1, 2.0, "native"), "array": np.zeros((1,))},), 1 << 20, 4))
    assert all(type(value) is int and value > 0 for value in terms.values())
    assert terms["held_current_encoded_parsed_and_returned_receipts"] >= 5 * (1 << 20)
    assert terms["own_control_code_and_raw_buffers"] > 0
    assert terms["finite_sort_and_projection_visits"] > 0
    # A large comparison count is a work counter, never retained payload.
    payload = sum(value for name, value in terms.items() if name != "finite_sort_and_projection_visits")
    assert payload < sum(terms.values())


def test_metadata_unknown_cycle_and_oversized_text_refuse_before_canonicalization(monkeypatch):
    monkeypatch.setattr(adapter, "_canonical_input", lambda *a, **k: pytest.fail("canonicalization reached"))
    cycle = []
    cycle.append(cycle)
    for value in (object(), cycle, "x" * 4097):
        with pytest.raises(DenseRunnerError, match="windowed_metadata_domain"):
            adapter._metadata_terms(np, (value,), 1 << 20, 4)


def test_own_module_image_rejects_unknown_holder_and_counts_constant_content(monkeypatch):
    before = dict(adapter._own_image_terms())
    monkeypatch.setattr(adapter, "_source_test_constant", {"retained": ["native" * 8]}, raising=False)
    after = dict(adapter._own_image_terms())
    assert after["own_control_text_encode_hash_error_buffers"] > before["own_control_text_encode_hash_error_buffers"]
    assert after["own_control_reference_and_frame_words"] > before["own_control_reference_and_frame_words"]
    monkeypatch.setattr(adapter, "_source_test_unknown", object(), raising=False)
    with pytest.raises(DenseRunnerError, match="windowed_image_domain"):
        adapter._own_image_terms()


def test_factory_canonical_vegetation_origin_and_complete_state(tmp_path):
    c = fixture.components(tmp_path)
    r = fixture.make_runner(c)
    assert r.accepted.next_index == 0 and r.accepted.n_accepted_events == 0
    assert c.bed.time_s.hex() == fixture.ORIGIN_S.hex()
    assert c.bed.state.committed_topography.last_commit_time_s.hex() == fixture.ORIGIN_S.hex()
    assert c.water.vegetation_cover_fraction is c.bed.vegetation.raster.cover_fraction
    assert np.all(c.water.vegetation_cover_fraction == 0.25)
    assert c.bed.vegetation.source == "raster"
    stage = c.bed.stage_copy()
    state = c.bed.stage_scheduler_state(stage, zeros_mobile_cohort_state(4000), time_s=fixture.ORIGIN_S)
    assert tuple(f.name for f in dataclasses.fields(state.physical)) == holder.PHYSICAL_FIELDS
    assert state.physical.water is stage.state.water
    assert state.physical.ledger_reconciliation_baseline is stage.state.reconciliation_baseline
    assert state.physical.vegetation is c.bed.vegetation
    assert state.block_index == 0 and state.block_offset_s.hex() == 0.0.hex()
    assert state.coupled is state.topographic_wind_cache is state.vegetation_wind_cache is None
    assert state.active_layer_composition_cache is None


def test_stage_copies_all_residents_and_never_writes_deep_store(tmp_path):
    c = fixture.components(tmp_path)
    bed = c.bed
    initial, identity = bed.state, bed.identity_sha256()
    old = files(initial.voxel_column.deep_store.store_dir)
    stage = bed.stage_copy()
    assert bed.identity_sha256() == identity == stage.identity_sha256
    assert stage.state is not initial and stage.state.voxel_column is not initial.voxel_column
    assert stage.state.voxel_column.deep_store is initial.voxel_column.deep_store
    for name in holder.WINDOWED_FIELDS[1:]:
        if getattr(initial, name) is not None:
            same(getattr(stage.state, name), getattr(initial, name), name)
            for source in arrays(getattr(initial, name)):
                assert not any(np.shares_memory(source, target) for target in arrays(getattr(stage.state, name)))
    for name in ("window_kg", "base", "deep_by_class_kg", "deep_total_kg", "deep_room_kg"):
        assert not np.shares_memory(getattr(stage.state.voxel_column, name), getattr(initial.voxel_column, name))
    assert files(initial.voxel_column.deep_store.store_dir) == old
    retained_files(old)


def test_pending_identity_journal_is_copied_and_content_is_controlling(tmp_path):
    c = fixture.components(tmp_path)
    original = c.bed.state.voxel_column
    d = int(original.base.reshape(-1)[0]) - 1
    assert d >= 0  # actual deep voxel, not a fabricated reservoir
    mass = c.fx.dense_state.physical.voxel_column.mass_kg.reshape((16, 6, 1))[0, d].copy()
    journal = DeepJournal.from_records(np.array([0], dtype=np.int64), np.array([d], dtype=np.int64),
        mass.reshape((1, 1)), header=original.deep_store.header, max_records=1)
    column = attach_pending_journal(rebind_prefix_packets(original, original), journal)
    pending_state = dataclasses.replace(c.bed.state, voxel_column=column)
    c.bed.adopt_bed_state(pending_state, time_s=fixture.ORIGIN_S)
    stage = c.bed.stage_copy()
    assert stage.pending_records_in == stage.pending_records_out == 1
    same(stage.state.voxel_column.pending_deep.journal, column.pending_deep.journal)
    for name in ("cell_flat", "voxel", "values_kg"):
        assert not np.shares_memory(getattr(stage.state.voxel_column.pending_deep.journal, name),
                                   getattr(column.pending_deep.journal, name))
    before = c.bed.identity_sha256()
    values = column.pending_deep.journal.values_kg
    values.flags.writeable = True  # external mutation of read-only exposed state is detected
    values[0, 0] = np.nextafter(values[0, 0], math.inf)
    assert c.bed.identity_sha256() != before


def test_mutated_scheduler_rebuild_refuses_before_first_transport(tmp_path, monkeypatch):
    c = fixture.components(tmp_path)
    # Same enabled feature flags, but a genuine scheduler field differs from
    # the central Native builder. Flag-only validation would miss this.
    bad = dataclasses.replace(c.wind.config, outer_timestep_s=0.5)
    wind = dataclasses.replace(c.wind, config=bad)
    profile = dataclasses.replace(c.profile, wind_config=bad)
    monkeypatch.setattr(adapter, "_prepare_record", lambda *a, **k: pytest.fail("transport reached"))
    with pytest.raises(DenseRunnerError, match="scheduler_config_mismatch"):
        fixture.make_runner(c, wind=wind, profile=profile)


def test_failed_wind_keeps_accepted_boundary_and_all_old_versions(tmp_path, monkeypatch):
    c = fixture.components(tmp_path)
    runner = fixture.make_runner(c)
    accepted = runner.accepted.as_record()
    old = files(c.bed.state.voxel_column.deep_store.store_dir)
    calls = []
    def refused(*args, **kwargs):
        calls.append("prepare")
        raise ValueError("authored native preflight refusal")
    monkeypatch.setattr(adapter, "_prepare_record", refused)
    monkeypatch.setattr(adapter, "_advance_record", lambda *a: pytest.fail("advance reached"))
    receipt = runner.step()
    assert calls == ["prepare"] and receipt["status"] == "native_refused"
    assert runner.accepted.as_record() == accepted
    assert runner.receipts == [] and runner.stop_receipt is not None
    retained_files(old)


def _physical_case(tmp_path, monkeypatch, count):
    process_profile = {2: "two_events", 4: "four_events"}[count]
    actual = fixture.components(tmp_path / "wrapper", count=count, process_profile=process_profile)
    oracle = fixture.components(tmp_path / "native", count=count, process_profile=process_profile)
    runner = fixture.make_runner(actual)
    old_files = files(actual.bed.state.voxel_column.deep_store.store_dir)
    oracle_old_files = files(oracle.bed.state.voxel_column.deep_store.store_dir)
    original_bed = actual.bed.state
    # Observe exact genuine return objects without changing any operation or
    # source binding. These are test-only spies, not fabricated Native results.
    calls = []
    prepare, advance, commit, complete = (adapter._prepare_record, adapter._advance_record,
                                          adapter._commit_entry, adapter._complete_positive)
    def observed_prepare(*args, **kwargs):
        owner = prepare(*args, **kwargs)
        calls.append(("prepare", owner))
        return owner
    def observed_advance(owner):
        value = advance(owner)
        calls.append(("wind", value))
        return value
    def observed_commit(*args, **kwargs):
        value = commit(*args, **kwargs)
        calls.append(("entry", value))
        return value
    def observed_complete(*args, **kwargs):
        value = complete(*args, **kwargs)
        calls.append(("rain", value))
        return value
    monkeypatch.setattr(adapter, "_prepare_record", observed_prepare)
    monkeypatch.setattr(adapter, "_advance_record", observed_advance)
    monkeypatch.setattr(adapter, "_commit_entry", observed_commit)
    monkeypatch.setattr(adapter, "_complete_positive", observed_complete)
    previous_time = fixture.ORIGIN_S
    snapshots = []
    for i, expected in enumerate(fixture.direct_events(oracle)):
        before_calls = len(calls)
        receipt = runner.step()
        assert receipt["status"] == receipt["code"] == "accepted", receipt
        assert receipt["index"] == i
        assert actual.bed.time_s.hex() == expected.time_s.hex() and expected.time_s > previous_time
        same_bed(actual.bed.state, expected.state, actual, oracle)
        same(runner.mobile_state, expected.pool, "mobile pool")
        same(runner.soil_water_m, expected.soil, "soil")
        for field in fixture.VEGETATION_FIELDS:
            same(getattr(actual.bed.vegetation.raster, field), getattr(oracle.bed.vegetation.raster, field), field)
        if expected.kind == "wind":
            assert fixture.mobile_state_is_genuinely_empty(expected.pool)
            assert fixture.mobile_state_is_genuinely_empty(runner.mobile_state)
            assert [name for name, _ in calls[before_calls:]] == ["prepare", "wind"]
            got = calls[-1][1]
            same(got.record.aggregate_accounting, expected.result.record.aggregate_accounting, "wind accounting")
            for f in dataclasses.fields(got.finalization):
                if f.name != "state":
                    same(getattr(got.finalization, f.name), getattr(expected.result.finalization, f.name), f.name)
            assert receipt["native"]["record"]["prepare_calls"] == 1
            assert receipt["native"]["record"]["advance_calls"] == 1
            assert receipt["native"]["record"]["finalizer_calls"] == 1
        else:
            assert [name for name, _ in calls[before_calls:]] == ["entry", "rain"]
            entry_commit, entry_summary = calls[-2][1]
            for name in ("did_commit", "rerouted", "net_displacement_m", "terminal_storage_change"):
                same(getattr(entry_commit, name), getattr(expected.entry_commit, name), "entry." + name)
            same(entry_commit.routing_source_m, expected.entry_commit.routing_source_m, "entry.routing_source")
            same(entry_commit.graph, expected.entry_commit.graph, "entry.graph")
            assert receipt["windowed_water"]["entry_admission"] == dataclasses.asdict(entry_summary)
            got, summary = calls[-1][1]
            same(got.accounting, expected.wet.accounting, "dry accounting")
            same(got.pre_reset_surface_m, expected.wet.pre_reset_surface_m, "pre-reset surface")
            same(got.pre_reset_soil_m, expected.wet.pre_reset_soil_m, "pre-reset soil")
            wr, er = got.progress.result, expected.wet.progress.result
            for f in dataclasses.fields(wr):
                if f.name in _TIMING:
                    assert all(type(v) is float and math.isfinite(v) and v >= 0.0
                               for v in (getattr(wr, f.name), getattr(er, f.name)))
                elif f.name != "state":
                    same(getattr(wr, f.name), getattr(er, f.name), "water result." + f.name)
            same(runner._routing.source, expected.routing.routing_source_elevation_m, "routing source")
            for name in ("graph", "terrain", "network", "grid"):
                same(getattr(runner._routing, name), getattr(expected.routing, name), "routing." + name)
            same(runner._routing.template.storm, expected.routing.storm, "retained storm")
            assert not np.any(actual.bed.state.water.depth_m)
            assert not np.any(actual.bed.state.water.mobile_mass_by_cell_class_kg)
            # Admission work/read/storage counters belong to their actual
            # invocation, including staged/cache owner metadata. They are
            # genuine positive receipts, not cross-role physical quantities.
            assert receipt["windowed_water"]["event_admission"] == dataclasses.asdict(summary)
            for value in dataclasses.asdict(summary).values():
                assert type(value) is int and value >= 0
            assert receipt["windowed_water"]["report_rows_declared"] == 4096
        # Old in-memory boundaries and every referenced immutable file remain
        # intact. All versions, including legitimate unchanged versions, stay.
        snapshots.append((actual.bed.state, holder._field_identity(actual.bed.state)))
        for state, identity in snapshots:
            assert holder._field_identity(state) == identity
        retained_files(old_files)
        retained_files(oracle_old_files)
        previous_time = expected.time_s
    assert runner.accepted.next_index == runner.accepted.n_accepted_events == count
    assert len(runner.receipts) == count and runner.stop_receipt is None
    assert runner.report()["status"] == "complete"
    assert original_bed is not actual.bed.state
    assert actual.bed.state.voxel_column.deep_store.store_dir == original_bed.voxel_column.deep_store.store_dir


def test_first_two_events_match_direct_native_and_preserve_old_versions(tmp_path, monkeypatch):
    _physical_case(tmp_path, monkeypatch, 2)


def test_four_events_match_direct_native_and_preserve_full_trajectory(tmp_path, monkeypatch):
    _physical_case(tmp_path, monkeypatch, 4)
