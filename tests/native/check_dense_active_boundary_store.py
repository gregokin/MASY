"""Source-authored opt-in active-water checks; Root alone grants execution.

Positive paths use genuine C/Native owners and the original Tiny water fixture.
Synthetic scalar records test metadata identity only, never Native authority.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import check_dense_water_pause as q
from masy import dense_active_checkpoint as a
from masy import dense_active_water_store as aw
from masy import dense_boundary_store as c
from masy import dense_original_inputs as oi
from masy import dense_runner as dr
from masy import dense_water_pause as pause


@contextlib.contextmanager
def store_at(path, *, events=None):
    runtime = q.reference()
    runner = q.new_runner(events=events)
    source, _environment = c._runtime_binding(runtime, c._DEFAULT)
    originals = oi.capture_original_inputs(runner, source_identity=source)
    handle = c.create_active_water_boundary_store(path, runner=runner, originals=originals,
        runtime_reference=runtime, storage_contract=c.StorageContract(True, True))
    del runner, originals
    try:
        yield handle, runtime
    finally:
        handle.close()


def selected(handle):
    head, _raw = handle._head()
    chain = aw._generation_chain(handle._root, handle._header, c._sha(handle._store_raw),
                                 head, handle._limits)
    return head, chain[0][1]


def accepted_snapshot(handle):
    head, _raw = handle._head()
    prepared, chain = handle._load(head)
    owner = c._owner_for(handle._header, prepared)
    result = q.snapshot(owner)
    del prepared, owner, chain
    return result


def attempt_files(handle, generation):
    """Only the actual selected tiny Native child, after existing raw admission."""
    aw._inventory(handle)
    root = aw._managed(handle) / generation["water_ref"]["attempt_name"]
    names = ("export.json", "entry/capsule.json", "native/checkpoint.json",
             "native/maple_state.npz", "native/continuation.npz")
    return {name: c.runtime._hash_path(root / name, budget=22 << 20)[0] for name in names}


def uninterrupted():
    owner = q.new_runner()
    answer = owner.step()
    if answer["status"] != "accepted":
        raise AssertionError("genuine uninterrupted Native dry completion required")
    result = q.snapshot(owner)
    del owner
    return result


def metadata_head(*, revision=0, kind="genesis", sequence=0, anchor=None,
                  operation="d" * 64, ordinal=None, count=None):
    """Authored scalar identity arithmetic only; no Native holder/certificate."""
    name = f"g-{revision:04d}-" + "a" * 32
    return c._seal({"schema": aw.SCHEMA + ".head", "store_id": "b" * 32,
        "store_file_sha256": "c" * 64, "revision": revision,
        "accepted_sequence": sequence, "kind": kind, "generation_name": name,
        "generation_file_sha256": "d" * 64, "generation_sha256": "e" * 64,
        "anchor_generation": name if anchor is None else anchor,
        "event_operation_id": operation, "request_id": "f" * 64,
        "pause_ordinal": ordinal, "pause_count": count}, "head_sha256")


class ActiveMetadata(unittest.TestCase):
    def test_request_types_and_ordinal_are_distinct_from_pause_count(self):
        self.assertEqual(aw._intent("checkpoint_water", 17)["boundary_ordinal"], 17)
        aw._pause_fields("water_paused", 17, 2)
        for bad in (True, False, 0, -1, 4097, 1.0, "1"):
            with self.subTest(value=bad), self.assertRaises(c.BoundaryStoreError):
                aw._intent("checkpoint_water", bad)
        with self.assertRaises(c.BoundaryStoreError):
            aw._pause_fields("water_paused", 17, 3)
        with self.assertRaises(c.BoundaryStoreError):
            aw._intent("finish", 1)

    def test_schema_profile_are_closed_and_opt_in(self):
        self.assertFalse(c._active_water_store(None))
        self.assertFalse(c._active_water_store({"schema": c.SCHEMA, "profile": c.PROFILE}))
        self.assertFalse(c._active_water_store({"schema": aw.SCHEMA, "profile": c.PROFILE}))
        self.assertTrue(c._active_water_store({"schema": aw.SCHEMA, "profile": aw.PROFILE}))

    def test_revision_cursor_and_head_inventory_refuse(self):
        valid = metadata_head()
        aw._head_fields(valid, c._DEFAULT)
        for key, value in (("revision", True), ("revision", 25),
                           ("accepted_sequence", 1), ("kind", "dry_but_unaccepted")):
            changed = {**valid, key: value}
            changed.pop("head_sha256")
            with self.subTest(key=key), self.assertRaises(c.BoundaryStoreError):
                aw._head_fields(c._seal(changed, "head_sha256"), c._DEFAULT)
        with self.assertRaises(c.BoundaryStoreError):
            aw._head_fields({**valid, "unknown": None}, c._DEFAULT)

    def test_request_keeps_event_operation_and_binds_selected_revision(self):
        header = {"store_id": "b" * 32, "limits": c._limits_record(c._DEFAULT)}
        anchor, schedule = metadata_head(), "1" * 64
        finish, checkpoint = aw._intent("finish"), aw._intent("checkpoint_water", 1)
        operation, request = aw._request_id(header, anchor, anchor, oi._json(anchor), finish, schedule)
        operation2, request2 = aw._request_id(header, anchor, anchor, oi._json(anchor), checkpoint, schedule)
        self.assertEqual(operation, operation2)
        self.assertNotEqual(request, request2)
        paused = metadata_head(revision=1, kind="water_paused", anchor=anchor["generation_name"],
                               operation=operation, ordinal=1, count=1)
        operation3, request3 = aw._request_id(header, anchor, paused, oi._json(paused), finish, schedule)
        self.assertEqual(operation3, operation)
        self.assertNotEqual(request3, request)
        with self.assertRaises(c.BoundaryStoreError):
            aw._request_id(header, anchor, paused, oi._json(anchor), finish, schedule)

    def test_plain_metadata_admission_precedes_encoding(self):
        with mock.patch.object(c, "_seal", side_effect=AssertionError("seal before bound")):
            with self.assertRaises(c.BoundaryStoreError):
                aw._seal_control({"x": "x" * 4096}, "intent_sha256", 1024, "authored negative")
        with mock.patch.object(oi, "_json", side_effect=AssertionError("encode before bound")):
            with self.assertRaises(c.BoundaryStoreError):
                aw._metadata_bytes({"x": "x" * 4096}, 1024, "authored negative")

    def test_unknown_request_and_foreign_parent_refuse(self):
        with self.assertRaises(c.BoundaryStoreError):
            aw._intent("resume_automatically", None)
        with self.assertRaises(c.BoundaryStoreError):
            aw._check_intent({"kind": "finish", "boundary_ordinal": None, "other": 0})
        with self.assertRaises(c.BoundaryStoreError):
            aw._selection(object())

    def test_whole_reservation_refuses_before_copy_or_transport(self):
        limits = dataclasses.replace(c._DEFAULT, max_memory_bytes=1)
        with mock.patch.object(oi, "capture_original_inputs", side_effect=AssertionError("copy")), \
                mock.patch.object(dr, "_complete_event", side_effect=AssertionError("transport")):
            with self.assertRaises(c.BoundaryStoreError) as caught:
                aw._reservation(limits, 1 << 28)
        self.assertEqual(caught.exception.code, "memory_cap")


class GenuineActiveWater(unittest.TestCase):
    def test_first_genuine_active_pause_publication(self):
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, _runtime):
            parent = handle.head_token()
            accepted, before = copy.deepcopy(handle.report()["accepted"]), accepted_snapshot(handle)
            ack = handle.checkpoint_water(parent, boundary_ordinal=1)
            head, generation = selected(handle)
            self.assertIs(type(ack), aw.ActiveWaterAcknowledgement)
            self.assertEqual((head["kind"], head["revision"], head["accepted_sequence"]), ("water_paused", 1, 0))
            self.assertEqual((head["pause_ordinal"], head["pause_count"]), (1, 1))
            self.assertEqual(handle.report()["accepted"], accepted)
            self.assertEqual(accepted_snapshot(handle), before)
            self.assertEqual(oi._strict_json(ack.ack_bytes, c._JSON)["returned_kind"], "water_paused")
            self.assertEqual(len(attempt_files(handle, generation)), 5)
            self.assertFalse(handle._entered)
            self.assertIsNone(handle._failed_owner)

    def test_cold_reexport_and_finish_match_native_control(self):
        expected = uninterrupted()
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, _runtime):
            first = handle.checkpoint_water(handle.head_token(), boundary_ordinal=1)
            _head, generation = selected(handle)
            old_files = attempt_files(handle, generation)
            second = handle.checkpoint_water(first.head_token, boundary_ordinal=3)
            head, current = selected(handle)
            self.assertEqual((head["revision"], head["accepted_sequence"], head["pause_ordinal"], head["pause_count"]),
                             (2, 0, 3, 2))
            self.assertNotEqual(current["water_ref"]["attempt_name"], generation["water_ref"]["attempt_name"])
            self.assertEqual(attempt_files(handle, generation), old_files)
            ack = handle.step(second.head_token)
            self.assertEqual(oi._strict_json(ack.ack_bytes, c._JSON)["returned_kind"], "event_accepted")
            self.assertEqual(accepted_snapshot(handle), expected)
            self.assertEqual(attempt_files(handle, generation), old_files)

    def test_third_pause_refuses_before_native_and_retains_selected_head(self):
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, _runtime):
            first = handle.checkpoint_water(handle.head_token(), boundary_ordinal=1)
            second = handle.checkpoint_water(first.head_token, boundary_ordinal=3)
            before = (handle._root / "HEAD").read_bytes()
            with mock.patch.object(pause, "resume_water", side_effect=AssertionError("third consume")), \
                    mock.patch.object(a, "load_water_pause", side_effect=AssertionError("third LOAD")), \
                    mock.patch.object(c, "_prepare_for", side_effect=AssertionError("third B load")), \
                    mock.patch.object(dr, "_complete_event", side_effect=AssertionError("third transport")):
                with self.assertRaises(c.BoundaryStoreError) as caught:
                    handle.checkpoint_water(second.head_token, boundary_ordinal=4)
            self.assertEqual(caught.exception.code, "water_pause_limit")
            self.assertEqual((handle._root / "HEAD").read_bytes(), before)
            self.assertEqual(handle.report()["phase"], "idle")
            self.assertEqual(handle.head_token(), second.head_token)

    def test_duplicate_ack_and_conflict_have_zero_load_or_transport(self):
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, _runtime):
            parent = handle.head_token()
            first = handle.checkpoint_water(parent, boundary_ordinal=1)
            with mock.patch.object(a, "load_water_pause", side_effect=AssertionError("LOAD")), \
                    mock.patch.object(dr, "_complete_event", side_effect=AssertionError("transport")), \
                    mock.patch.object(c, "_prepare_for", side_effect=AssertionError("B load")):
                self.assertEqual(handle.acknowledge(parent, request_kind="checkpoint_water", boundary_ordinal=1), first)
                self.assertEqual(handle.checkpoint_water(parent, boundary_ordinal=1), first)
                with self.assertRaises(c.BoundaryStoreError) as caught:
                    handle.acknowledge(parent, request_kind="finish")
                self.assertEqual(caught.exception.code, "request_conflict")

    def test_actual_exclusive_lease_and_public_borrow_refuse(self):
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, _runtime):
            with self.assertRaises(c.BoundaryStoreError) as caught:
                c.recover_active_water_boundary_store(handle._root,
                    expected_store_file_sha256=handle.report()["store_file_sha256"])
            self.assertEqual(caught.exception.code, "lease_busy")
            with self.assertRaises(c.BoundaryStoreError):
                c.borrow_water_parent(handle, expected_parent=handle.head_token())

    def test_completion_wins_for_original_checkpoint_request(self):
        expected = uninterrupted()
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, _runtime):
            first = handle.checkpoint_water(handle.head_token(), boundary_ordinal=1)
            _head, generation = selected(handle)
            path = aw._managed(handle) / generation["water_ref"]["attempt_name"] / "entry/capsule.json"
            capsule = a._decode(c._read(path, 1 << 20), 1 << 20)
            ordinal = len(capsule["plan"]) - 1
            self.assertGreater(ordinal, 1)
            ack = handle.checkpoint_water(first.head_token, boundary_ordinal=ordinal)
            self.assertEqual(oi._strict_json(ack.ack_bytes, c._JSON)["returned_kind"], "event_accepted")
            self.assertEqual(accepted_snapshot(handle), expected)
            with mock.patch.object(a, "load_water_pause", side_effect=AssertionError("replay")):
                self.assertEqual(handle.acknowledge(first.head_token, request_kind="checkpoint_water",
                                                  boundary_ordinal=ordinal), ack)


if __name__ == "__main__":
    unittest.main()
