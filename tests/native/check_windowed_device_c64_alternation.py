"""Root-only one-budget four-event Native GPU interface qualification.

No saved simulation payload is shared with Source reviewers. This bounded C64 complete
readback is an oracle only; the product adapter never gathers full columns.
"""
from __future__ import annotations

import dataclasses
import hashlib
import stat

import numpy as np
import pytest
import fixture_windowed_device_c64_alternation as f
from tests.windowed_wind_gpu.test_device_reduced_step import cupy  # noqa: F401
from maple.core.backend import to_host
from maple.surface.topographic_commit.windowed_device import _bed_leaves
from maple.surface.voxels.windowed_inventory import gather_full_columns
from maple_syrup import windowed_shared_bed_device as device_water
from masy.clock import decode_time
from masy.dense_runner import DenseRunnerError

pytestmark = pytest.mark.gpu


def _same(left, right, label):
    cp = __import__("sys").modules["cupy"]
    if type(left) in (np.ndarray, cp.ndarray):
        assert type(right) is type(left), label
        assert left.dtype == right.dtype and left.shape == right.shape, label
        assert np.ascontiguousarray(to_host(left)).tobytes() == np.ascontiguousarray(to_host(right)).tobytes(), label
    elif dataclasses.is_dataclass(left) and not isinstance(left, type):
        assert type(left) is type(right), label
        assert tuple(vars(left)) == tuple(vars(right)), label
        for field in dataclasses.fields(left):
            _same(getattr(left, field.name), getattr(right, field.name), label+"."+field.name)
    elif type(left) in (tuple, list):
        assert type(left) is type(right) and len(left) == len(right), label
        for i, (a, b) in enumerate(zip(left, right)):
            _same(a, b, label+"["+str(i)+"]")
    elif type(left) is dict:
        assert type(right) is dict and left.keys() == right.keys(), label
        for key in left:
            _same(left[key], right[key], label+"."+str(key))
    elif type(left) is float:
        assert type(right) is float and left.hex() == right.hex(), label
    else:
        assert type(left) is type(right) and left == right, label


def _state_equal(actual, expected):
    for name in ("active_layer", "water", "ledger", "committed_topography",
                 "sediment_availability", "reconciliation_baseline"):
        _same(getattr(actual, name), getattr(expected, name), name)
    cells = np.arange(64, dtype=np.int64)
    _same(gather_full_columns(actual.voxel_column, cells, max_columns=64),
          gather_full_columns(expected.voxel_column, cells, max_columns=64), "bounded C64 complete columns")
    for name in ("window_kg", "base", "deep_by_class_kg", "deep_total_kg", "deep_room_kg"):
        _same(getattr(actual.voxel_column, name), getattr(expected.voxel_column, name), "resident."+name)
    assert actual.voxel_column.pending_deep is expected.voxel_column.pending_deep is None
    assert actual.voxel_column.n_voxels == expected.voxel_column.n_voxels == 20
    assert actual.voxel_column.generation == expected.voxel_column.generation


def _snapshot(case):
    """Read exactly the retained original fixture leaves under a finite ceiling."""
    cp = __import__("sys").modules["cupy"]
    rows, seen = [], set()
    def walk(value):
        if type(value) in (np.ndarray, cp.ndarray):
            assert len(rows) < 4096 and sum(len(row[3]) for row in rows)+value.nbytes <= 64 << 20
            rows.append((value, value.dtype.str, value.shape, np.ascontiguousarray(to_host(value)).tobytes()))
        elif id(value) not in seen:
            seen.add(id(value))
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                for field in dataclasses.fields(value):
                    walk(getattr(value, field.name))
            elif type(value) in (tuple, list):
                for item in value:
                    walk(item)
            elif type(value) is dict:
                for item in value.values():
                    walk(item)
    state = case.initial
    for name in device_water._BED_FIELDS:
        if name != "voxel_column":
            walk(getattr(state, name))
    for _name, array in _bed_leaves(state.voxel_column, state.active_layer):
        walk(array)
    accepted = case.bed._accepted.state
    for name in device_water._BED_FIELDS:
        if name != "voxel_column":
            walk(getattr(accepted, name))
    for _name, array in _bed_leaves(accepted.voxel_column, accepted.active_layer):
        walk(array)
    walk((case.physical.vegetation, case.physical.threshold_modifier_state, case.wind,
          case.water, case.control, case.schedule, case.host.state))
    return rows


def _unchanged(rows):
    for array, dtype, shape, raw in rows:
        assert (array.dtype.str, array.shape, np.ascontiguousarray(to_host(array)).tobytes()) == (dtype, shape, raw)


def _files(root):
    result, count, size = {}, 0, 0
    for path in root.rglob("*"):
        count += 1
        assert count <= 4096
        metadata = path.lstat()
        if stat.S_ISREG(metadata.st_mode):
            assert size+metadata.st_size <= 64 << 20
            raw = path.read_bytes()
            assert len(raw) == metadata.st_size
            size += len(raw)
            result[str(path.relative_to(root))] = (len(raw), hashlib.sha256(raw).hexdigest())
        else:
            assert stat.S_ISDIR(metadata.st_mode)
    return result


def _old_files_unchanged(root, before):
    current = _files(root)
    for name, identity in before.items():
        assert current[name] == identity  # later immutable versions may coexist


def test_four_c64_device_events_match_direct_native(tmp_path, cupy, device_water_c64_qualification_budget):
    """Two genuine independent initializations; one fixed whole-attempt budget."""
    cp = cupy
    case = f.components(tmp_path/"wrapper", cp, device_water_c64_qualification_budget)
    oracle = f.components(tmp_path/"oracle", cp, device_water_c64_qualification_budget)
    assert case.control.completion_policy is not oracle.control.completion_policy
    assert dataclasses.asdict(case.control.completion_policy) == dataclasses.asdict(oracle.control.completion_policy)
    assert case.control.completion_policy.terminal_storage_water is True
    assert case.control.completion_policy.deposit_mobile_at_flow_end is True
    assert case.bed.case_record["water_completion"] == oracle.bed.case_record["water_completion"]
    assert case.bed.case_record["water_completion"]["schema"] == "masy.interim_accounted_ponded_water.v1"
    inputs = _snapshot(case)+_snapshot(oracle)
    old = case.initial.voxel_column.deep_store
    other_old = oracle.initial.voxel_column.deep_store
    files, other_files = _files(old.store_dir), _files(other_old.store_dir)
    runner = f.make_runner(case)
    counts = dict(case.bed._counts)
    with pytest.raises(DenseRunnerError, match="bed_owner_consumed"):
        f.make_runner(case)  # Refuses before any Native receiver or copy.
    assert case.bed._counts == counts and case.bed._runner_consumed is True
    assert case.bed._zero_op_copy_roles is None
    canonical = runner._device_resources.static("vegetation")
    original_vegetation = tuple(np.ascontiguousarray(to_host(getattr(canonical.raster, name))).tobytes()
                                for name in f.VEGETATION_FIELDS)
    assert runner._water.vegetation_cover_fraction is canonical.raster.cover_fraction
    accepted, ponded_removed = [], []
    for index, direct in enumerate(f.direct_events(oracle, cp)):
        # A yield retains the actual upstream outcome. No oracle success is
        # manufactured from the wrapper's receipt or result fields.
        receipt = runner.step()
        assert receipt["status"] == receipt["code"] == "accepted", receipt
        assert receipt["index"] == index and receipt["kind"] == direct.kind
        assert runner.accepted.next_index == index+1
        assert case.bed._zero_op_copy_roles is None
        assert case.bed.time_s.hex() == float(direct.time_s).hex()
        _state_equal(case.bed._accepted.state, direct.state)
        accepted.append(runner.accepted)
        assert tuple(np.ascontiguousarray(to_host(getattr(canonical.raster, name))).tobytes()
                     for name in f.VEGETATION_FIELDS) == original_vegetation
        _same(canonical, oracle.physical.vegetation, "complete prescribed vegetation")
        if direct.kind == "wind":
            final = direct.result.finalization
            assert final.completed and final.water_handoff_ready
            assert direct.owner.finalizer_call_count == 1 and direct.result.record.all_blocks_conserved
            assert receipt["finalization"]["status"] == final.status
            assert receipt["native"]["record"]["finalizer_calls"] == 1
            assert decode_time(receipt["native"]["record"]["record_origin_s"]).hex() == (
                f.ORIGIN_S if index == 0 else accepted[index-1].time_s).hex()
            accounting = direct.result.record.aggregate_accounting
            assert accounting.n_topographic_wind_recomputations > 0
            assert accounting.n_active_layer_composition_recomputations > 0
            assert direct.result.state.topographic_wind_cache is None
            assert direct.result.state.active_layer_composition_cache is None
            assert not cp.any(direct.pool.mass_kg)
        else:
            wet = direct.wet
            result = wet.progress.result
            assert result.n_accepted_steps > 0 and result.n_maple_water_calls > 0
            assert float(to_host(result.cumulative_rain_m.sum())) > 0.0
            assert float(to_host(result.by_class["actual_pickup"].sum())) > 0.0
            assert wet.accounting["sediment_closure"]["closed"]
            assert wet.accounting["storm_water_budget"]["closed"]
            accounting, budget = wet.accounting, wet.accounting["storm_water_budget"]
            after = accounting["completion_after_terminal"]
            assert wet.progress.policy is oracle.control.completion_policy
            assert accounting["policy"] == "external dry-again removal; no evapotranspiration simulated"
            assert accounting["terminal_storage_removed_m3"] == after["terminal_surface_volume_m3"]
            assert accounting["terminal_storage_cells"] == after["terminal_storage_cells"]
            assert accounting["terminal_storage_removed_m3"] >= 0.0
            assert accounting["moving_surface_removed_m3"] == (
                accounting["surface_removed_m3"] - accounting["terminal_storage_removed_m3"])
            assert accounting["surface_removed_m3"] == budget["surface_final_m3"]
            assert accounting["soil_removed_m3"] == budget["soil_final_m3"] >= 0.0
            assert abs(accounting["combined_water_residual_m3"]) <= budget["tolerance_m3"]
            _same(wet.pre_reset_surface_m, result.state.storm.depth_m, "actual pre-reset surface water")
            _same(wet.pre_reset_soil_m, result.state.storm.soil_water_m, "actual pre-reset soil water")
            ponded_removed.append(accounting["terminal_storage_removed_m3"])
            for name in ("rain_m3", "surface_initial_m3", "soil_initial_m3", "surface_final_m3", "soil_final_m3",
                         "drainage_m3", "export_m3", "residual_m3", "tolerance_m3"):
                assert receipt["water"][name] == budget[name]
            for receipt_name, native_name in (("dry_surface_removed_m3", "surface_removed_m3"),
                    ("dry_soil_removed_m3", "soil_removed_m3"), ("water_added_m3", "water_added_m3"),
                    ("combined_water_residual_m3", "combined_water_residual_m3")):
                assert receipt["water"][receipt_name] == accounting[native_name]
            assert receipt["water"]["water_added_m3"] == 0.0
            assert float(to_host(result.peak_depth_m.max())) > 0.0
            for array in (direct.state.water.depth_m, direct.state.water.mobile_mass_by_cell_class_kg,
                          direct.soil, wet.dry_state.storm.depth_m, wet.dry_state.storm.soil_water_m,
                          wet.dry_state.sediment_velocity_m_s):
                assert not cp.any(array)
            assert wet.dry_state.phase is None
            assert runner._routing.graph is not None
            _same(runner._routing.source, wet.dry_state.routing_source_elevation_m, "retained routing source")
            assert receipt["routing"]["entry_rerouted"] is direct.entry_commit.rerouted
            assert receipt["routing"]["first_graph_construction"] is (index == 1)
            for summary in (direct.entry_summary, direct.event_summary):
                assert summary.known_subset_only is True and all(value is None for _name, value in summary.opaque_terms)
            actual = receipt["device_water_qualification"]
            for kind, summary in (("entry", direct.entry_summary), ("event", direct.event_summary)):
                assert actual[kind]["gate_count"] == summary.gate_count
                assert actual[kind]["logical_work_debited"] == summary.logical_work_debited
            assert device_water._OWNER is None
        _unchanged(inputs)
        _old_files_unchanged(old.store_dir, files)
        _old_files_unchanged(other_old.store_dir, other_files)
    assert len(accepted) == 4 and case.bed.generation == 4
    assert len(ponded_removed) == 2 and sum(ponded_removed) > 0.0
    report = runner.report()
    assert report["status"] == "complete" and len(report["receipts"]) == 4
    for row in report["receipts"]:
        assert np.all(np.abs(row["aggregate"]["joint_residual_kg"]) <= np.asarray(row["aggregate"]["joint_bound_kg"]))
    assert report["qualification"]["allocation_or_fit_proof"] is None
    assert report["qualification"]["restart_supported"] is False
    assert all(row["qualification"]["known_subset_only"] is True for row in report["receipts"])
    assert runner._water_counters["gate_count"] == sum(row["device_water_qualification"][kind]["gate_count"]
        for row in report["receipts"] if row["kind"] == "rain" for kind in ("entry", "event"))


def test_c64_device_failure_keeps_last_accepted_boundary(tmp_path, cupy, device_water_c64_qualification_budget):
    """A real original static-input guard after a genuine accepted wind prefix.

    This is an adversarial metadata mutation, not a manufactured Native result
    or a claim that an unproved forced resource quota separates two events.
    """
    case = f.components(tmp_path/"refusal", cupy, device_water_c64_qualification_budget)
    runner = f.make_runner(case)
    assert runner.step()["status"] == "accepted"
    accepted, ledger, receipts = runner.accepted, runner.ledger_record(), runner.receipts
    snapshot = _snapshot(case)
    # No subsequent transport receives this changed value: original _precheck
    # hashes every controlling field and refuses before work staging.
    object.__setattr__(case.control, "report_every_s", 20.0)
    stop = runner.step()
    assert stop["status"] == "refused" and stop["code"] == "static_input_mutated_before_event"
    assert runner.accepted == accepted and runner.ledger_record() == ledger
    assert runner.receipts == receipts and runner.accepted.next_index == 1
    assert case.bed.generation == 1 and case.bed._active is None
    assert runner._device_resources._active is None and device_water._OWNER is None
    _unchanged(snapshot)
    with pytest.raises(ValueError):
        runner.step()  # stopped runner never silently skips to wind or another rain
