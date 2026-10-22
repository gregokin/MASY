"""Source-authored recovery/IO fences; no nested worker or execution grant.

Python exception fences are distinct from Root's later sequential actual
process-death qualification. Saved expectations stay local to Root's workers.
"""
from __future__ import annotations

import contextlib
import copy
import gc
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import check_dense_water_pause as q
import check_dense_active_boundary_store as fixture
from masy import dense_active_checkpoint as a
from masy import dense_active_water_store as aw
from masy import dense_boundary_store as c
from masy import dense_original_inputs as oi
from masy import dense_runner as dr


def recover(root, sha):
    return c.recover_active_water_boundary_store(root, expected_store_file_sha256=sha)


def write_fresh_paused_fixture(directory):
    """Genuine writer-only role; Root reaps and pins expected.json before reader."""
    root = Path(directory)
    root.mkdir(mode=0o700, exist_ok=False)
    with fixture.store_at(root / "store") as (handle, _runtime):
        parent = handle.head_token()
        first = handle.checkpoint_water(parent, boundary_ordinal=1)
        _head, generation = fixture.selected(handle)
        expected = {"schema": aw.SCHEMA + ".fresh_fixture", "store": str(handle._root),
            "store_file_sha256": handle.report()["store_file_sha256"],
            "parent_token": parent, "selected_token": first.head_token,
            "ack_hex": first.ack_bytes.hex(), "accepted": handle.report()["accepted"],
            "first_child": generation["water_ref"], "files": fixture.attempt_files(handle, generation)}
    expected["uninterrupted"] = fixture.uninterrupted()
    path = root / "expected.json"
    c._write(path, oi._json(expected))
    path.chmod(0o444)
    return {"expectation_path": str(path),
            "expectation_sha256": c.runtime._hash_path(path, budget=1 << 20)[0],
            "role": "writer_only; no nested or implicit reader"}


class ActiveRecovery(unittest.TestCase):
    def test_recovery_loads_without_transport_and_reuses_same_selected_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            with fixture.store_at(root) as (handle, _runtime):
                ack = handle.checkpoint_water(handle.head_token(), boundary_ordinal=1)
                sha = handle.report()["store_file_sha256"]
                accepted = copy.deepcopy(handle.report()["accepted"])
                before = (root / "HEAD").read_bytes()
            gc.collect()
            with mock.patch.object(dr, "_complete_event", side_effect=AssertionError("recovery transport")), \
                    mock.patch.object(dr, "_rain_enter", side_effect=AssertionError("recovery entry")), \
                    contextlib.closing(recover(root, sha)) as handle:
                self.assertEqual(handle.head_token(), ack.head_token)
                self.assertEqual(handle.report()["accepted"], accepted)
                self.assertEqual((root / "HEAD").read_bytes(), before)
                self.assertFalse(handle._entered)
                self.assertIsNone(handle._failed_owner)

    def test_real_intent_fence_does_not_adopt_orphan_or_replay_unpublished_ack(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            with fixture.store_at(root) as (handle, _runtime):
                parent, sha = handle.head_token(), handle.report()["store_file_sha256"]
                before = (root / "HEAD").read_bytes()
                actual = aw._reserve_attempt
                def intent_then_interrupt(*args, **kwargs):
                    actual(*args, **kwargs)  # real directory + intent fsync
                    raise InterruptedError("authored pre-Native intent fence")
                with mock.patch.object(aw, "_reserve_attempt", side_effect=intent_then_interrupt), \
                        mock.patch.object(dr, "_complete_event", side_effect=AssertionError("Native")):
                    with self.assertRaises(InterruptedError):
                        handle.checkpoint_water(parent, boundary_ordinal=1)
                self.assertEqual((root / "HEAD").read_bytes(), before)
                self.assertEqual(handle.report()["phase"], "halted_before_head")
            with contextlib.closing(recover(root, sha)) as handle:
                self.assertEqual(handle.head_token(), parent)
                with mock.patch.object(a, "load_water_pause", side_effect=AssertionError("orphan LOAD")), \
                        mock.patch.object(dr, "_complete_event", side_effect=AssertionError("orphan transport")):
                    with self.assertRaises(c.BoundaryStoreError) as caught:
                        handle.acknowledge(parent, request_kind="checkpoint_water", boundary_ordinal=1)
                self.assertEqual(caught.exception.code, "request_not_published")
                ack = handle.checkpoint_water(parent, boundary_ordinal=1)
                self.assertEqual(ack.revision, 1)
                self.assertEqual(len(aw._inventory(handle)["generations"]), 3)

    def test_native_child_before_head_fence_leaves_old_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            with fixture.store_at(root) as (handle, _runtime):
                parent, sha = handle.head_token(), handle.report()["store_file_sha256"]
                before = (root / "HEAD").read_bytes()
                def before_publish(*_args, **_kwargs):
                    raise InterruptedError("authored after child/P2/readback/close/sync before HEAD")
                with mock.patch.object(c.DenseBoundaryStore, "_publish", side_effect=before_publish):
                    with self.assertRaises(InterruptedError):
                        handle.checkpoint_water(parent, boundary_ordinal=1)
                self.assertEqual((root / "HEAD").read_bytes(), before)
                self.assertFalse(handle._entered)
            with contextlib.closing(recover(root, sha)) as handle:
                self.assertEqual(handle.head_token(), parent)
                self.assertEqual(handle.report()["accepted"]["accepted"]["n_accepted_events"], 0)

    def test_lost_ack_after_real_publish_is_metadata_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary, fixture.store_at(Path(temporary) / "store") as (handle, _runtime):
            parent, captured = handle.head_token(), []
            actual = c.DenseBoundaryStore._publish
            def publish_then_lose_ack(self, *args):
                answer = actual(self, *args)  # real replace/root fsync, exact returned ACK
                captured.append(answer)
                raise BrokenPipeError("authored external ACK delivery loss")
            with mock.patch.object(c.DenseBoundaryStore, "_publish", new=publish_then_lose_ack):
                with self.assertRaises(BrokenPipeError):
                    handle.checkpoint_water(parent, boundary_ordinal=1)
            self.assertEqual(len(captured), 1)
            self.assertEqual(handle.report()["phase"], "idle")
            with mock.patch.object(a, "load_water_pause", side_effect=AssertionError("ACK LOAD")), \
                    mock.patch.object(dr, "_complete_event", side_effect=AssertionError("ACK transport")):
                self.assertEqual(handle.acknowledge(parent, request_kind="checkpoint_water", boundary_ordinal=1), captured[0])

    def test_root_fsync_failure_is_indeterminate_and_requires_explicit_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            with fixture.store_at(root) as (handle, _runtime):
                parent, sha = handle.head_token(), handle.report()["store_file_sha256"]
                actual = os.fsync
                def fail_root(fd):
                    if fd == handle._rootfd:
                        raise OSError("authored HEAD root-fsync failure")
                    return actual(fd)
                with mock.patch.object(c.os, "fsync", side_effect=fail_root):
                    with self.assertRaises(OSError):
                        handle.checkpoint_water(parent, boundary_ordinal=1)
                self.assertEqual(handle.report()["phase"], "publication_indeterminate")
                with self.assertRaises(c.BoundaryStoreError):
                    handle.step(parent)
            with contextlib.closing(recover(root, sha)) as handle:
                # The actual replace was done, but this does not prove physical
                # durability under power failure; recovery validates selected R.
                self.assertEqual(handle._state.head["kind"], "water_paused")
                selected_before = handle.head_token()
                raw_before = (root / "HEAD").read_bytes()
                with mock.patch.object(a, "load_water_pause", side_effect=AssertionError("duplicate LOAD")), \
                        mock.patch.object(dr, "_complete_event", side_effect=AssertionError("duplicate transport")):
                    self.assertEqual(handle.acknowledge(parent, request_kind="checkpoint_water", boundary_ordinal=1).head_token,
                                     selected_before)
                self.assertEqual(handle.head_token(), selected_before)
                self.assertEqual((root / "HEAD").read_bytes(), raw_before)

    def test_changed_raw_child_refuses_before_native_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            with fixture.store_at(root) as (handle, _runtime):
                handle.checkpoint_water(handle.head_token(), boundary_ordinal=1)
                sha = handle.report()["store_file_sha256"]
                _head, generation = fixture.selected(handle)
                target = aw._managed(handle) / generation["water_ref"]["attempt_name"] / "native/continuation.npz"
                target.chmod(0o600)  # deliberate custody corruption of this test's new child only
                with target.open("r+b") as output:
                    output.seek(-1, 2)
                    bit = output.read(1)
                    output.seek(-1, 2)
                    output.write(bytes((bit[0] ^ 1,)))
                target.chmod(0o444)
            with mock.patch.object(a, "load_water_pause", side_effect=AssertionError("changed child LOAD")), \
                    mock.patch.object(dr, "_complete_event", side_effect=AssertionError("transport")):
                with self.assertRaises(dr.DenseRunnerError) as caught:
                    recover(root, sha)
                self.assertEqual(caught.exception.code, "active_raw_changed")

    def test_unknown_sibling_entry_refuses_before_hash_or_parser(self):
        with tempfile.TemporaryDirectory() as temporary, fixture.store_at(Path(temporary) / "store") as (handle, _runtime):
            (aw._managed(handle) / "unknown").write_bytes(b"authored metadata negative")
            with mock.patch.object(c, "_read", side_effect=AssertionError("hash/parser before inventory")):
                with self.assertRaises(c.BoundaryStoreError):
                    handle._stable()


class FreshActiveReader(unittest.TestCase):
    def test_fresh_recovery_reexport_finish(self):
        """Root separately reaps writer and externally pins its actual bytes."""
        import check_dense_active_checkpoint as stagea
        expected = stagea.fresh_metadata(os.environ["MASY388_FRESH_EXPECTED"],
                                         os.environ["MASY388_FRESH_EXPECTED_SHA256"])
        if expected["schema"] != aw.SCHEMA + ".fresh_fixture":
            raise AssertionError("exact actual writer expectation required")
        with contextlib.closing(recover(Path(expected["store"]), expected["store_file_sha256"])) as handle:
            self.assertEqual(handle.head_token(), expected["selected_token"])
            self.assertEqual(oi._json(handle.report()["accepted"]), oi._json(expected["accepted"]))
            with mock.patch.object(a, "load_water_pause", side_effect=AssertionError("duplicate LOAD")), \
                    mock.patch.object(dr, "_complete_event", side_effect=AssertionError("duplicate transport")):
                ack = handle.acknowledge(expected["parent_token"], request_kind="checkpoint_water", boundary_ordinal=1)
                self.assertEqual(ack.ack_bytes.hex(), expected["ack_hex"])
            _head, first = fixture.selected(handle)
            self.assertEqual(oi._json(first["water_ref"]), oi._json(expected["first_child"]))
            self.assertEqual(fixture.attempt_files(handle, first), expected["files"])
            second = handle.checkpoint_water(handle.head_token(), boundary_ordinal=3)
            self.assertEqual(second.accepted_sequence, 0)
            self.assertEqual(handle._state.head["pause_count"], 2)
            finish = handle.step(second.head_token)
            self.assertEqual(finish.accepted_sequence, 1)
            self.assertEqual(oi._json(fixture.accepted_snapshot(handle)), oi._json(expected["uninterrupted"]))
            self.assertEqual(fixture.attempt_files(handle, first), expected["files"])


if __name__ == "__main__":
    unittest.main()
