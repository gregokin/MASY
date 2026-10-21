"""Inert dynamic B components plus one separately gated actual CPU continuation.

Tests named test_transport_* require the later explicit four-event gate. They
are excluded from the initial constructor-only component execution plan.
"""
from __future__ import annotations

import json
import shutil
from unittest import mock

import check_dense_dynamic_physical_checkpoint as f
import numpy as np
import pytest

from masy import dense_original_inputs as oi
from masy import dense_runner_checkpoint as b
from masy import landscape_vegetation as lv


def setup_artifacts(root, runner=None):
    runner = f.dynamic_genesis() if runner is None else runner
    originals = oi.capture_original_inputs_v2(runner, source_identity=f.SOURCES)
    oi.save_original_inputs_v2(root / "original", originals)
    oref = f.reference(root / "original", oi.SCHEMA_V2, originals.content_sha256)
    shared = f.resources(runner, originals.capsule)
    identity = f.save_native(root / "genesis", runner, originals.capsule, shared)
    pref = f.reference(root / "genesis", f.cp.SCHEMA, identity)
    seal = b.save_passive_boundary_v2(root / "boundary", runner, artifact_root=root,
        originals=oref, genesis=pref, physical=pref, source_identity=f.SOURCES,
        environment_sha256=f.ENVIRONMENT)
    return runner, originals, oref, pref, seal


@pytest.fixture(scope="module")
def sealed(tmp_path_factory):
    root = tmp_path_factory.mktemp("dynamic-boundary-components")
    return root, setup_artifacts(root)


def prepare(root, seal, **kwargs):
    return b.prepare_passive_boundary_v2(root / "boundary", artifact_root=root,
        expected_manifest_sha256=seal, expected_source_identity=f.SOURCES,
        expected_environment_sha256=f.ENVIRONMENT, **kwargs)


def reseal(path, record):
    record["manifest_sha256"] = b._sha({key: value for key, value in record.items()
        if key != "manifest_sha256"})
    (path / "manifest.json").chmod(0o600)
    (path / "manifest.json").write_bytes(oi._json(record))
    return record["manifest_sha256"]


def mutated(tmp_path, sealed, change):
    old_root, _ = sealed
    root = tmp_path / "copy"
    shutil.copytree(old_root, root)
    record = json.loads((root / "boundary/manifest.json").read_bytes())
    change(record)
    return root, reseal(root / "boundary", record)


def test_genesis_inert_twenty_eight_field_owner_aliases_and_pending_future(sealed):
    root, (runner, _, _, _, seal) = sealed
    prepared = prepare(root, seal)
    assert type(prepared) is b._PreparedBoundaryV2 and type(prepared) is not b._PreparedBoundary
    assert not hasattr(prepared, "owner") and not hasattr(prepared, "step")
    out = prepared._PreparedBoundaryV2__owner
    assert set(vars(out)) == set(b.RUNNER_FIELDS_V2)
    assert out._vegetation_plan is out._bed._dynamic.plan
    assert out._wind.config.geometry is out._bed.context.geometry
    assert out._wind.config.initial_available_fraction is out._bed.context.initial_available_fraction
    assert out._water.vegetation_cover_fraction is not out._bed._dynamic._native_borrow().raster.cover_fraction
    assert out._vegetation_plan.plan_record() == runner._vegetation_plan.plan_record()
    assert out._bed._dynamic.cursor["next_event_index"] == 0
    assert out._vegetation_copied_bytes == 48 * out._vegetation_plan.cells
    assert out._bed._dynamic.as_record() == runner._bed._dynamic.as_record()
    assert out.accepted.as_record() == runner.accepted.as_record()
    assert out.ledger_record() == runner.ledger_record()
    assert not np.shares_memory(out._soil, runner._soil)
    assert not np.shares_memory(out._inventory0, runner._inventory0)


def test_genesis_native_load_once_and_no_constructor_transport_update_or_reset(sealed):
    root, (_, _, _, _, seal) = sealed
    forbidden = AssertionError("restore replays physical/update work")
    with (mock.patch.object(f.dr.DenseRunner, "__init__", side_effect=forbidden),
          mock.patch.object(f.db, "build_dense_bed", side_effect=forbidden),
          mock.patch.object(lv, "prepare_vegetation", side_effect=forbidden),
          mock.patch.object(lv.LandscapeVegetation, "from_native", side_effect=forbidden),
          mock.patch.object(f.dr, "_advance_wind_block", side_effect=forbidden),
          mock.patch.object(f.dr, "_finalize_wind_event", side_effect=forbidden),
          mock.patch.object(f.dr, "_complete_event", side_effect=forbidden),
          mock.patch("maple_syrup.sediment_bed.commit_bed", side_effect=forbidden),
          mock.patch.object(f.bridge, "load_boundary_physical_v2", wraps=f.bridge.load_boundary_physical_v2) as native,
          mock.patch.object(oi, "load_original_inputs_v2", wraps=oi.load_original_inputs_v2) as originals,
          mock.patch.object(lv, "vegetation_plan_from_record", wraps=lv.vegetation_plan_from_record) as plans):
        prepare(root, seal)
    assert native.call_count == originals.call_count == plans.call_count == 1


@pytest.mark.parametrize("change", ["schema", "source", "exchange", "metadata", "count", "counter", "soil"])
def test_closed_metadata_forgery_refuses_and_preserves_existing_accepted_owner(tmp_path, sealed, change):
    _, (runner, _, _, _, _) = sealed
    before = f.signature(runner._bed.state), f.signature(runner._pool), f.signature(runner._soil), runner._bed._dynamic.as_record()
    def alter(record):
        if change == "schema":
            record["schema"] = b.SCHEMA
        elif change == "source":
            record["source_identity"]["masy"] = "a" * 64
        elif change == "exchange":
            record["shared_resources"]["water_exchange"] = [["invented", 0.0]]
        elif change == "metadata":
            record["shared_resources"]["vegetation_metadata_sha256"] = "a" * 64
        elif change == "count":
            record["state"]["n_accepted"] = 1
        elif change == "counter":
            record["shared_resources"]["accepted_run_copy_bytes"] += 1
        else:
            record["shared_resources"]["soil_identity"]["array"]["dtype"] = "<f4"
    root, seal = mutated(tmp_path, sealed, alter)
    with pytest.raises((b.BoundaryCheckpointError, f.db.DenseBedError, lv.VegetationError, f.cp.PhysicalCheckpointError)):
        prepare(root, seal)
    assert before == (f.signature(runner._bed.state), f.signature(runner._pool), f.signature(runner._soil), runner._bed._dynamic.as_record())


def test_conflicting_empty_boundary_clock_is_not_accepted(tmp_path, sealed):
    root, seal = mutated(tmp_path, sealed,
        lambda record: record["ledger"].__setitem__("native_epoch", f.bridge.encode_time(1.0)))
    with pytest.raises((b.BoundaryCheckpointError, f.db.DenseBedError, ValueError)):
        prepare(root, seal)


def test_global_child_cap_refuses_before_hash_or_native_load(sealed):
    root, (_, _, _, _, seal) = sealed
    with (mock.patch.object(f.bridge, "load_boundary_physical_v2", side_effect=AssertionError("native before cap")),
          mock.patch("masy.runtime._hash_path", side_effect=AssertionError("hash before cap")),
          pytest.raises(b.BoundaryCheckpointError, match="serialized_cap")):
        prepare(root, seal, limits=b.BoundaryImageLimits(max_global_raw_bytes=1))


def test_late_loader_failure_preserves_accepted_source_owner(sealed):
    root, (runner, _, _, _, seal) = sealed
    before = f.signature(runner._bed.state), f.signature(runner._pool), runner._bed._dynamic.as_record()
    with mock.patch.object(f.bridge, "load_boundary_physical_v2", side_effect=OSError("authored late load failure")), pytest.raises(OSError):
        prepare(root, seal)
    assert before == (f.signature(runner._bed.state), f.signature(runner._pool), runner._bed._dynamic.as_record())


def test_v1_entrypoint_never_accepts_dynamic_carrier_or_image(sealed):
    root, (runner, _, oref, pref, seal) = sealed
    with pytest.raises(b.BoundaryCheckpointError):
        b.prepare_passive_boundary(root / "boundary", artifact_root=root,
            expected_manifest_sha256=seal, expected_source_identity=f.SOURCES,
            expected_environment_sha256=f.ENVIRONMENT)
    with pytest.raises(b.BoundaryCheckpointError):
        b.save_passive_boundary(root / "wrong-v1", runner, artifact_root=root,
            originals=oref, genesis=pref, physical=pref, source_identity=f.SOURCES,
            environment_sha256=f.ENVIRONMENT)
    assert not (root / "wrong-v1").exists()


def test_transport_real_four_events_resume_future_delta_exactly_once_and_late_second_load(tmp_path):
    """Separate explicit transport gate: actual native wind/rain, no clock fakes."""
    runner, originals, oref, genesis, _ = setup_artifacts(tmp_path)
    for index in range(2):
        receipt = runner.step()
        assert receipt["status"] == "accepted" and receipt["index"] == index
    assert runner._bed._dynamic.version == 2
    assert runner._vegetation_plan.deltas[-1].after_event_index == 3
    shared = f.resources(runner, originals.capsule)
    identity = f.save_native(tmp_path / "current2", runner, originals.capsule, shared)
    current = f.reference(tmp_path / "current2", f.cp.SCHEMA, identity)
    seal = b.save_passive_boundary_v2(tmp_path / "boundary2", runner, artifact_root=tmp_path,
        originals=oref, genesis=genesis, physical=current, source_identity=f.SOURCES,
        environment_sha256=f.ENVIRONMENT)
    before = f.signature(runner._bed.state), f.signature(runner._pool), f.signature(runner._soil), runner.report()
    actual = f.bridge.load_boundary_physical_v2
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("authored second native load failure")
        return actual(*args, **kwargs)
    with mock.patch.object(f.bridge, "load_boundary_physical_v2", side_effect=fail_second), pytest.raises(OSError):
        b.prepare_passive_boundary_v2(tmp_path / "boundary2", artifact_root=tmp_path,
            expected_manifest_sha256=seal, expected_source_identity=f.SOURCES,
            expected_environment_sha256=f.ENVIRONMENT)
    assert len(calls) == 2
    assert before == (f.signature(runner._bed.state), f.signature(runner._pool), f.signature(runner._soil), runner.report())
    prepared = b.prepare_passive_boundary_v2(tmp_path / "boundary2", artifact_root=tmp_path,
        expected_manifest_sha256=seal, expected_source_identity=f.SOURCES,
        expected_environment_sha256=f.ENVIRONMENT)
    out = prepared._PreparedBoundaryV2__owner  # test-only access, never C adoption
    assert not out._soil.flags.writeable
    assert out._water.vegetation_cover_fraction.tobytes() == originals.water.vegetation_cover_fraction.tobytes()
    assert out._bed._dynamic.version == 2 and out._bed._dynamic.cursor["next_event_index"] == 2
    for index in range(2, 4):
        left, right = runner.step(), out.step()
        assert left["status"] == right["status"] == "accepted"
        assert f.dr._canonical_json(left) == f.dr._canonical_json(right)
        assert f.signature(runner._bed.state) == f.signature(out._bed.state)
        assert f.signature(runner._pool) == f.signature(out._pool)
        assert f.signature(runner._soil)[:3] == f.signature(out._soil)[:3]
        assert not np.shares_memory(runner._soil, out._soil)
    assert runner.ledger_record() == out.ledger_record()
    assert out._bed._dynamic.version == 3 and len(out._bed._dynamic.receipts) == 3
    assert out._bed._dynamic.as_record() == runner._bed._dynamic.as_record()
    assert out._vegetation_copied_bytes == runner._vegetation_copied_bytes
    assert out._vegetation_copied_bytes == 48 * out._vegetation_plan.cells + 16 * out._vegetation_plan.cells + sum(row["copied_bytes"] for row in out._bed._dynamic.receipts)
    assert np.count_nonzero(out._soil) == 0
