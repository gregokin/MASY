"""C-v2 failure controls plus one separately selected real four-event case."""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
from unittest import mock

import numpy as np
import pytest

import check_dense_dynamic_boundary_store as store
from check_dense_dynamic_boundary_store import environment, genesis
from masy import dense_boundary_store as c
from masy import dense_original_inputs as oi
from masy import dense_runner as dr
from masy import dense_runner_checkpoint as b
from masy import landscape_vegetation as lv


def test_failed_private_event_preserves_head_and_recovery_never_replays(genesis):
    path, sha, token = genesis
    before = (path / "HEAD").read_bytes()
    with contextlib.closing(store.recover(path, sha)) as handle:
        original = handle.report()["accepted"]
        with (mock.patch.object(dr.DenseRunner, "step", side_effect=RuntimeError("authored pre-transport failure")) as call,
              pytest.raises(RuntimeError, match="pre-transport")):
            handle.step(token)
        assert call.call_count == 1
        assert handle.report()["phase"] == "halted_before_head"
        assert handle.report()["accepted"] == original
        assert (path / "HEAD").read_bytes() == before
        assert handle._failed_owner is not None
        with pytest.raises(c.BoundaryStoreError, match="store_halted"):
            handle.step(token)
    fail = AssertionError("recovery replayed transport")
    with (mock.patch.object(dr, "_advance_wind_block", side_effect=fail),
          mock.patch.object(dr, "_finalize_wind_event", side_effect=fail),
          mock.patch.object(dr, "_complete_event", side_effect=fail),
          contextlib.closing(store.recover(path, sha)) as restored):
        assert restored.head_token() == token
        assert restored.report()["accepted"] == original


def test_late_second_native_load_failure_has_no_accepted_head(tmp_path, environment):
    path = tmp_path / "late-native"
    runner = environment[0]
    before = (store.fixture.signature(runner._bed.state), store.fixture.signature(runner._pool),
        store.fixture.signature(runner._soil), runner.report())
    actual = c.bridge.load_boundary_physical_v2
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("authored second native restore failure")
        return actual(*args, **kwargs)

    with (mock.patch.object(c.bridge, "load_boundary_physical_v2", side_effect=fail_second),
          pytest.raises(OSError, match="second native")):
        store.create(path, environment)
    assert len(calls) == 2 and not (path / "HEAD").exists()
    assert before == (store.fixture.signature(runner._bed.state), store.fixture.signature(runner._pool),
        store.fixture.signature(runner._soil), runner.report())
    assert (path / "STORE.json").exists()
    with pytest.raises(FileNotFoundError):
        store.recover(path, store.digest((path / "STORE.json").read_bytes()))


def test_outer_generation_write_failure_never_publishes(tmp_path, environment):
    path = tmp_path / "generation"
    actual = c._sealed_write

    def fail_generation(root, name, *args, **kwargs):
        if name == "generation.json":
            raise OSError("authored generation write failure")
        return actual(root, name, *args, **kwargs)

    with (mock.patch.object(c, "_sealed_write", side_effect=fail_generation),
          pytest.raises(OSError, match="generation write")):
        store.create(path, environment)
    assert not (path / "HEAD").exists()
    assert (path / "STORE.json").exists()


def test_genesis_postreplace_fsync_failure_is_indeterminate_and_recoverable(tmp_path, environment):
    path = tmp_path / "post-replace"
    real_open = c.DenseBoundaryStore._open.__func__
    real_fsync = c.os.fsync
    captured = []

    def opened(cls, *args, **kwargs):
        handle = real_open(cls, *args, **kwargs)
        captured.append(handle)
        return handle

    def fail_final(fd):
        if (captured and fd == captured[0]._rootfd
                and captured[0]._state.phase == "publication_possible"):
            raise OSError("authored post-replace root fsync failure")
        return real_fsync(fd)

    with (mock.patch.object(c.DenseBoundaryStore, "_open", classmethod(opened)),
          mock.patch.object(c.os, "fsync", side_effect=fail_final),
          pytest.raises(OSError, match="post-replace")):
        store.create(path, environment)
    assert captured[0]._closed
    assert captured[0]._state.phase == "publication_indeterminate"
    assert (path / "HEAD").is_file()
    sha = store.digest((path / "STORE.json").read_bytes())
    with contextlib.closing(store.recover(path, sha)) as handle:
        assert handle.report()["accepted"]["accepted"]["next_index"] == 0
        assert handle.head_token() == captured[0]._state.intended


def test_stale_or_malformed_token_cannot_create_intent(genesis):
    path, sha, _ = genesis
    with contextlib.closing(store.recover(path, sha)) as handle:
        before = tuple(sorted(str(x.relative_to(path)) for x in path.rglob("*")))
        with (mock.patch.object(dr.DenseRunner, "step", side_effect=AssertionError("event on stale token")),
              pytest.raises(c.BoundaryStoreError, match="metadata_digest")):
            handle.step(True)
        with (mock.patch.object(dr.DenseRunner, "step", side_effect=AssertionError("event on stale token")),
              pytest.raises(c.BoundaryStoreError, match="stale_parent")):
            handle.step("0" * 64)
        assert before == tuple(sorted(str(x.relative_to(path)) for x in path.rglob("*")))
        assert handle.report()["phase"] == "idle"


def test_original_future_plan_mismatch_refuses_before_directory(tmp_path, environment):
    runner, _, reference = environment
    other = store.fixture.dynamic_genesis(deltas=())
    originals = oi.capture_original_inputs_v2(other, source_identity=store.old.SOURCES)
    with pytest.raises(c.BoundaryStoreError, match="original_identity"):
        c.create_dynamic_boundary_store(tmp_path / "mismatched", runner=runner,
            originals=originals, runtime_reference=reference,
            storage_contract=c.StorageContract(True, True))
    assert not (tmp_path / "mismatched").exists()


def _prefix_corruptions(handle, chain):
    """Probe every non-seal ancestor envelope field against real selected history."""
    assert len(chain) >= 3
    selected = chain[0][2]["shared_resources"]["vegetation"]
    ancestor = chain[1][2]["shared_resources"]["vegetation"]
    changed = {
        "schema": "unsupported", "profile": "unsupported", "version": ancestor["version"] + 1,
        "version_sha256": "f" * 64, "plan_sha256": "f" * 64,
        "native_content_sha256": "f" * 64, "initial_native_content_sha256": "f" * 64,
        "source": "none" if ancestor["source"] != "none" else "raster",
        "source_transition": None if ancestor["source_transition"] is not None else "none->raster",
        "cursor": {**ancestor["cursor"], "next_event_index": 0},
        "receipts": [], "bindings": [], "last_boundary_binding": None,
    }
    assert set(changed) == set(lv._ENVELOPE_KEYS) - {"metadata_sha256"}
    for key, value in changed.items():
        assert value != ancestor[key], key
        bad = copy.deepcopy(chain)
        bad[1][2]["shared_resources"]["vegetation"][key] = value
        with pytest.raises(c.BoundaryStoreError, match="vegetation_history"):
            c._history_v2(handle._root, handle._header, bad, handle._limits)
    for section in ("ledger", "receipts"):
        bad = copy.deepcopy(chain)
        if section == "ledger":
            bad[1][2][section]["entries"] = []
        else:
            bad[1][2][section] = []
        with pytest.raises(c.BoundaryStoreError, match="vegetation_history"):
            c._history_v2(handle._root, handle._header, bad, handle._limits)
    bad = copy.deepcopy(chain)
    bad[1][2]["shared_resources"]["accepted_run_copy_bytes"] += 8
    with pytest.raises(c.BoundaryStoreError, match="vegetation_history"):
        c._history_v2(handle._root, handle._header, bad, handle._limits)
    # The seal's value is header/ref-bound provenance on intermediate ancestors;
    # its independent native/B relationship still refuses a cross-header value.
    bad = copy.deepcopy(chain[1][2])
    bad["shared_resources"]["vegetation"]["metadata_sha256"] = selected["metadata_sha256"]
    assert bad["shared_resources"]["vegetation"]["metadata_sha256"] != ancestor["metadata_sha256"]
    with pytest.raises(b.BoundaryCheckpointError, match="resource_identity"):
        b._preflight_metadata_v2(bad, store.old.SOURCES, store.fixture.ENVIRONMENT, handle._limits.image)
    for key in ("physical", "copy", "binding"):
        bad = copy.deepcopy(chain)
        if key == "physical":
            bad[-1][2]["physical"] = copy.deepcopy(chain[0][2]["physical"])
        elif key == "copy":
            bad[-1][2]["shared_resources"]["accepted_run_copy_bytes"] += 8
        else:
            bad[-1][2]["shared_resources"]["vegetation"]["last_boundary_binding"] = copy.deepcopy(selected["last_boundary_binding"])
        with pytest.raises(c.BoundaryStoreError, match="vegetation_history"):
            c._history_v2(handle._root, handle._header, bad, handle._limits)


def _assert_exact_readonly_restore(restored, reference):
    """Physical bits match; restored arrays are independent and read-only."""
    assert type(restored) is type(reference)
    if type(restored) is np.ndarray:
        assert restored.dtype.str == reference.dtype.str
        assert restored.shape == reference.shape
        assert restored.tobytes() == reference.tobytes()
        assert not restored.flags.writeable
        assert not np.shares_memory(restored, reference, max_work=4096)
    elif dataclasses.is_dataclass(restored) and not isinstance(restored, type):
        for field in dataclasses.fields(restored):
            _assert_exact_readonly_restore(getattr(restored, field.name), getattr(reference, field.name))
    elif type(restored) in (tuple, list):
        assert len(restored) == len(reference)
        for left, right in zip(restored, reference, strict=True):
            _assert_exact_readonly_restore(left, right)
    elif type(restored) is dict:
        assert tuple(restored) == tuple(reference)
        for key in restored:
            _assert_exact_readonly_restore(restored[key], reference[key])
    else:
        assert store.fixture.signature(restored) == store.fixture.signature(reference)


def test_transport_saved_four_events_future_delta_exactly_once_and_history_corruption(tmp_path, environment):
    """Separate transport gate: genuine native alternation and persisted recovery."""
    path = tmp_path / "real-four"
    reference = store.fixture.dynamic_genesis()
    runner = store.fixture.dynamic_genesis()
    initial_copies = runner._vegetation_copied_bytes
    originals = oi.capture_original_inputs_v2(runner, source_identity=store.old.SOURCES)
    controlling = (runner, originals, environment[2])
    handle = store.create(path, controlling)
    first_parent = handle.head_token()
    try:
        for index in range(2):
            expected = reference.step()
            assert expected["status"] == "accepted" and expected["index"] == index
            ack = handle.step(handle.head_token())
            assert ack.receipt_bytes == dr._canonical_json(expected).encode()
        sha = handle.report()["store_file_sha256"]
        current_token = handle.head_token()
    finally:
        handle.close()
    fail = AssertionError("recovery or duplicate replayed native work/update")
    with (mock.patch.object(dr, "_advance_wind_block", side_effect=fail),
          mock.patch.object(dr, "_finalize_wind_event", side_effect=fail),
          mock.patch.object(dr, "_complete_event", side_effect=fail),
          mock.patch.object(lv, "prepare_vegetation", side_effect=fail)):
        handle = store.recover(path, sha)
        try:
            assert handle.head_token() == current_token
            # Historical parent acknowledgement is exact direct-child only;
            # neither the pending later prescribed delta nor transport replays.
            duplicate = handle.step(first_parent)
            assert duplicate.sequence == 1
            prepared, _ = handle._load(handle._state.head)
            owner = c._owner_v2(prepared)
            assert owner._bed._dynamic.version == 2
            assert owner._vegetation_plan.deltas[-1].after_event_index == 3
            assert not owner._soil.flags.writeable
            del prepared, owner
        except BaseException:
            handle.close()
            raise
    try:
        for index in range(2, 4):
            expected = reference.step()
            assert expected["status"] == "accepted" and expected["index"] == index
            ack = handle.step(handle.head_token())
            assert ack.receipt_bytes == dr._canonical_json(expected).encode()
        prepared, chain = handle._load(handle._state.head)
        owner = c._owner_v2(prepared)
        assert owner._next == owner._n_accepted == 4
        assert owner.ledger_record() == reference.ledger_record()
        assert owner._bed._dynamic.as_record() == reference._bed._dynamic.as_record()
        assert owner._bed._dynamic.version == 3 and len(owner._bed._dynamic.receipts) == 3
        assert owner._vegetation_plan.plan_record() == reference._vegetation_plan.plan_record()
        assert owner._vegetation_copied_bytes == reference._vegetation_copied_bytes
        assert owner._vegetation_copied_bytes == initial_copies + sum(
            row["vegetation"]["required_snapshot_bytes"] + row["vegetation"]["delta_copy_bytes"]
            for row in reference._receipts)
        assert owner._water.vegetation_cover_fraction.tobytes() == originals.water.vegetation_cover_fraction.tobytes()
        _assert_exact_readonly_restore(owner._bed.state, reference._bed.state)
        _assert_exact_readonly_restore(owner._pool, reference._pool)
        assert store.fixture.signature(owner._soil)[:3] == store.fixture.signature(reference._soil)[:3]
        assert not np.shares_memory(owner._soil, reference._soil)
        assert not owner._soil.flags.writeable and np.count_nonzero(owner._soil) == 0
        assert handle.report()["accepted"]["status"] == "complete"
        _prefix_corruptions(handle, chain)
        for _head, generation, manifest in chain:
            for role in ("genesis", "physical"):
                assert (path / manifest[role]["relative_path"]).is_dir()
        final_head = (path / "HEAD").read_bytes()
        final_parent = chain[1][0]
        duplicate_token = c._token(final_parent)
        with mock.patch.object(dr.DenseRunner, "step", side_effect=fail):
            duplicate = handle.step(duplicate_token)
        assert duplicate.sequence == 4 and (path / "HEAD").read_bytes() == final_head
        with (mock.patch.object(dr.DenseRunner, "step", side_effect=fail),
              pytest.raises(c.BoundaryStoreError, match="schedule_complete")):
            handle.step(handle.head_token())
    finally:
        handle.close()
