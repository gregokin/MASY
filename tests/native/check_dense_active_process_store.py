"""Finite Source-authored combined store checks; execution remains Root-gated.

One real Native owner per process. Metadata-only cases mint no physical proof.
The fixed writer never spawns; a separately reaped fresh reader is Root-owned.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import check_dense_active_process_checkpoint as q
from masy import dense_active_water_store as store
from masy import dense_active_checkpoint as water_checkpoint
from masy import dense_active_wind_checkpoint as wind_checkpoint
from masy import dense_boundary_store as c
from masy import dense_original_inputs as oi
from masy import dense_runner as dr
from masy import dense_water_pause as water
from masy import dense_wind_pause as wind


def selected(handle):
    head, _raw = handle._head()
    chain = store._generation_chain(handle._root, handle._header, c._sha(handle._store_raw),
                                    head, handle._limits)
    return head, chain[0][1]


def accepted_snapshot(handle):
    head, _raw = handle._head()
    prepared, chain = handle._load(head)
    owner = c._owner_for(handle._header, prepared)
    answer = q.snapshot(owner)
    del owner, prepared, chain
    return answer


def retained_child(handle, generation):
    store._inventory(handle)
    ref = generation["active_ref"]
    root = store._managed(handle) / ref["attempt_name"]
    names = (("export.json", "entry/capsule.json", "native/continuation.bin")
             if ref["process_kind"] == "wind" else
             ("export.json", "entry/capsule.json", "native/checkpoint.json",
              "native/maple_state.npz", "native/continuation.npz"))
    return {name: c.runtime._hash_path(root / name, budget=22 << 20)[0] for name in names}


def metadata_head(*, combined=True, revision=0, sequence=0, kind="genesis", ordinal=None, count=None, anchor=None):
    """Synthetic scalar arithmetic only, never a C handle or Native authority."""
    schema = store.SCHEMA_PROCESS if combined else store.SCHEMA
    name = f"g-{revision:04d}-" + "a" * 32
    return c._seal({"schema": schema + ".head", "store_id": "b" * 32,
        "store_file_sha256": "c" * 64, "revision": revision, "accepted_sequence": sequence,
        "kind": kind, "generation_name": name, "generation_file_sha256": "d" * 64,
        "generation_sha256": "e" * 64, "anchor_generation": name if anchor is None else anchor,
        "event_operation_id": "f" * 64, "request_id": "1" * 64,
        "pause_ordinal": ordinal, "pause_count": count}, "head_sha256")


class ActiveProcessMetadata(unittest.TestCase):
    def test_closed_opt_in_format_never_changes_default_or_water_profile(self):
        old = {"schema": store.SCHEMA, "profile": store.PROFILE}
        new = {"schema": store.SCHEMA_PROCESS, "profile": store.PROFILE_PROCESS}
        self.assertTrue(c._active_water_store(old))
        self.assertFalse(c._active_process_store(old))
        self.assertTrue(c._active_process_store(new))
        self.assertFalse(c._active_water_store(new))
        self.assertEqual(c._store_schema({"schema": c.SCHEMA, "profile": c.PROFILE}), c.SCHEMA)
        with self.assertRaises(c.BoundaryStoreError):
            c._store_schema({"schema": store.SCHEMA_PROCESS, "profile": store.PROFILE})
        self.assertEqual(store._ref_key(old), "water_ref")
        self.assertEqual(store._ref_key(new), "active_ref")

    def test_both_record_format_directions_refuse(self):
        old, new = metadata_head(combined=False), metadata_head()
        for header, value in ((old, new), (new, old)):
            with self.subTest(schema=header["schema"]), self.assertRaises(c.BoundaryStoreError):
                store._head_fields(value, c._DEFAULT, header=header)
            with self.assertRaises(c.BoundaryStoreError):
                store._same_format(header, value)

    def test_distinct_closed_process_intents_preserve_original_ordinal_rules(self):
        header = {"schema": store.SCHEMA_PROCESS}
        self.assertEqual(store._intent_for("checkpoint_wind", 17, header),
                         {"kind": "checkpoint_wind", "boundary_ordinal": 17})
        self.assertEqual(store._intent_for("checkpoint_water", 17, header), store._intent("checkpoint_water", 17))
        for bad in (True, False, 0, -1, 4097, 1.0, "1"):
            with self.subTest(value=bad), self.assertRaises(c.BoundaryStoreError):
                store._intent_for("checkpoint_wind", bad, header)
        with self.assertRaises(c.BoundaryStoreError):
            store._intent_for("checkpoint_wind", 1, {"schema": store.SCHEMA})
        with self.assertRaises(c.BoundaryStoreError):
            store._intent_for("checkpoint_other", 1, header)

    def test_wind_pause_revision_is_not_accepted_event_cursor(self):
        anchor = metadata_head()["generation_name"]
        paused = metadata_head(revision=1, kind="wind_paused", ordinal=17, count=1, anchor=anchor)
        store._head_fields(paused, c._DEFAULT, header={"schema": store.SCHEMA_PROCESS})
        self.assertEqual(paused["accepted_sequence"], 0)
        changed = {**paused, "pause_count": 3}
        changed.pop("head_sha256")
        with self.assertRaises(c.BoundaryStoreError):
            store._head_fields(c._seal(changed, "head_sha256"), c._DEFAULT,
                               header={"schema": store.SCHEMA_PROCESS})

    def test_plain_metadata_and_foreign_parent_refuse_before_encoder_or_codec(self):
        with mock.patch.object(oi, "_json", side_effect=AssertionError("unbounded encode")):
            with self.assertRaises(c.BoundaryStoreError):
                store._metadata_bytes({"x": "x" * 4096}, 1024, "authored negative")
        with self.assertRaises(c.BoundaryStoreError):
            store._selection(object())


class GenuineActiveProcess(unittest.TestCase):
    def test_first_genuine_active_wind_publication(self):
        with tempfile.TemporaryDirectory() as temporary, q.store_at(Path(temporary) / "store", combined=True) as (handle, _runtime):
            before, accepted = accepted_snapshot(handle), copy.deepcopy(handle.report()["accepted"])
            parent = handle.head_token()
            answer = handle.checkpoint_wind(parent, boundary_ordinal=1)
            head, generation = selected(handle)
            self.assertEqual((head["kind"], head["revision"], head["accepted_sequence"]), ("wind_paused", 1, 0))
            self.assertEqual((head["pause_ordinal"], head["pause_count"]), (1, 1))
            self.assertEqual(generation["active_ref"]["process_kind"], "wind")
            self.assertEqual(len(retained_child(handle, generation)), 3)
            self.assertEqual(handle.report()["accepted"], accepted)
            self.assertEqual(accepted_snapshot(handle), before)
            self.assertEqual(oi._strict_json(answer.ack_bytes, c._JSON)["returned_kind"], "wind_paused")
            self.assertFalse(handle._entered)
            self.assertIsNone(handle._failed_owner)

    def test_two_cold_wind_pauses_finish_and_duplicate_ack_without_replay(self):
        expected = q.ordinary()
        with tempfile.TemporaryDirectory() as temporary, q.store_at(Path(temporary) / "store", combined=True) as (handle, _runtime):
            parent = handle.head_token()
            first = handle.checkpoint_wind(parent, boundary_ordinal=1)
            _head, first_generation = selected(handle)
            original_files = retained_child(handle, first_generation)
            second = handle.checkpoint_wind(first.head_token, boundary_ordinal=2)
            self.assertEqual((second.revision, second.accepted_sequence), (2, 0))
            self.assertEqual(retained_child(handle, first_generation), original_files)
            accepted = handle.step(second.head_token)
            self.assertEqual((accepted.revision, accepted.accepted_sequence), (3, 1))
            self.assertEqual(accepted_snapshot(handle), expected)
            with mock.patch.object(wind_checkpoint, "load_wind_pause", side_effect=AssertionError("duplicate LOAD")), \
                    mock.patch.object(wind, "resume_wind", side_effect=AssertionError("duplicate consume")), \
                    mock.patch.object(c, "_prepare_for", side_effect=AssertionError("duplicate parent LOAD")):
                self.assertEqual(handle.acknowledge(parent, request_kind="checkpoint_wind", boundary_ordinal=1), first)
                self.assertEqual(handle.checkpoint_wind(first.head_token, boundary_ordinal=2), second)
                self.assertEqual(handle.step(second.head_token), accepted)
                with self.assertRaises(c.BoundaryStoreError) as caught:
                    handle.acknowledge(parent, request_kind="finish")
                self.assertEqual(caught.exception.code, "request_conflict")
            self.assertEqual(retained_child(handle, first_generation), original_files)

    def test_third_wind_pause_and_wrong_process_refuse_before_load(self):
        with tempfile.TemporaryDirectory() as temporary, q.store_at(Path(temporary) / "store", combined=True) as (handle, _runtime):
            first = handle.checkpoint_wind(handle.head_token(), boundary_ordinal=1)
            second = handle.checkpoint_wind(first.head_token, boundary_ordinal=2)
            head = (handle._root / "HEAD").read_bytes()
            with mock.patch.object(wind_checkpoint, "load_wind_pause", side_effect=AssertionError("LOAD")), \
                    mock.patch.object(c, "_prepare_for", side_effect=AssertionError("parent LOAD")):
                with self.assertRaises(c.BoundaryStoreError) as caught:
                    handle.checkpoint_wind(second.head_token, boundary_ordinal=3)
                self.assertEqual(caught.exception.code, "wind_pause_limit")
                with self.assertRaises(c.BoundaryStoreError) as caught:
                    handle.checkpoint_water(second.head_token, boundary_ordinal=3)
                self.assertEqual(caught.exception.code, "active_process_kind")
            self.assertEqual((handle._root / "HEAD").read_bytes(), head)
            self.assertEqual(handle.report()["phase"], "idle")

    def test_selected_wind_recovery_loads_actual_codec_and_advances_no_transport(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            with q.store_at(root, combined=True) as (handle, _runtime):
                ack = handle.checkpoint_wind(handle.head_token(), boundary_ordinal=1)
                sha = handle.report()["store_file_sha256"]
                _head, generation = selected(handle)
                retained = retained_child(handle, generation)
            from maple.aeolian.scheduler import block
            with mock.patch.object(block, "advance_wind_block_pausable", side_effect=AssertionError("begin in recovery")), \
                    mock.patch.object(block, "resume_wind_block", side_effect=AssertionError("advance in recovery")), \
                    mock.patch.object(dr, "_finalize_wind_event", side_effect=AssertionError("finalize in recovery")):
                recovered = c.recover_active_process_boundary_store(root, expected_store_file_sha256=sha)
            try:
                self.assertEqual(recovered.head_token(), ack.head_token)
                self.assertEqual(retained_child(recovered, generation), retained)
                self.assertFalse(recovered._entered)
            finally:
                recovered.close()

    def test_wind_rain_wind_rain_preserves_same_native_trajectory(self):
        events = q.f.schedule(q.f.wind(duration_s=1.0), q.f.rain(), q.f.wind(duration_s=1.0), q.f.rain())
        expected = q.ordinary(events=events)
        with tempfile.TemporaryDirectory() as temporary, q.store_at(Path(temporary) / "store", combined=True, events=events) as (handle, _runtime):
            for kind in ("wind", "water", "wind", "water"):
                parent = handle.head_token()
                paused = (handle.checkpoint_wind(parent, boundary_ordinal=1) if kind == "wind"
                          else handle.checkpoint_water(parent, boundary_ordinal=1))
                _head, generation = selected(handle)
                self.assertEqual(generation["active_ref"]["process_kind"], kind)
                self.assertEqual(handle.step(paused.head_token).accepted_sequence, handle._state.sequence)
            head, _generation = selected(handle)
            self.assertEqual((head["revision"], head["accepted_sequence"]), (8, 4))
            self.assertEqual(accepted_snapshot(handle), expected)

    def test_old_active_water_store_keeps_its_schema_and_native603_path(self):
        runtime = q.reference()
        runner = q.new_runner(events=q.f.schedule(q.f.rain()))
        source, _environment = c._runtime_binding(runtime, c._DEFAULT)
        originals = oi.capture_original_inputs(runner, source_identity=source)
        with tempfile.TemporaryDirectory() as temporary:
            handle = c.create_active_water_boundary_store(Path(temporary) / "store", runner=runner,
                originals=originals, runtime_reference=runtime, storage_contract=c.StorageContract(True, True))
            try:
                self.assertEqual(handle._header["schema"], store.SCHEMA)
                self.assertIn("managed_water", handle._header)
                self.assertNotIn("managed_active", handle._header)
                with self.assertRaises(c.BoundaryStoreError):
                    handle.checkpoint_wind(handle.head_token(), boundary_ordinal=1)
                pause = handle.checkpoint_water(handle.head_token(), boundary_ordinal=1)
                head, generation = store._head(handle)[0], None
                chain = store._generation_chain(handle._root, handle._header, c._sha(handle._store_raw), head, handle._limits)
                generation = chain[0][1]
                self.assertIn("water_ref", generation)
                self.assertNotIn("active_ref", generation)
                self.assertEqual(handle.step(pause.head_token).accepted_sequence, 1)
            finally:
                handle.close()

    def test_failed_actual_restore_never_publishes_or_adopts_an_orphan(self):
        with tempfile.TemporaryDirectory() as temporary, q.store_at(Path(temporary) / "store", combined=True) as (handle, _runtime):
            before = (handle._root / "HEAD").read_bytes()
            actual = wind_checkpoint.load_wind_pause
            def refuse(*args, **kwargs):
                # Real writer ran; this explicit negative never claims LOAD.
                raise dr.DenseRunnerError("authored_readback_refusal", "negative second-parent proof")
            with mock.patch.object(wind_checkpoint, "load_wind_pause", side_effect=refuse):
                with self.assertRaises(dr.DenseRunnerError):
                    handle.checkpoint_wind(handle.head_token(), boundary_ordinal=1)
            self.assertEqual((handle._root / "HEAD").read_bytes(), before)
            self.assertEqual(handle.report()["phase"], "halted_before_head")
            self.assertFalse(handle._entered)
            self.assertIs(actual, wind_checkpoint.load_wind_pause)


def write_fresh_process_fixture(directory):
    """Root fixed writer only; externally reap/pin before a new reader actor."""
    root = Path(directory)
    root.mkdir(mode=0o700, exist_ok=False)
    expected = q.ordinary()
    with q.store_at(root / "store", combined=True) as (handle, _runtime):
        pause = handle.checkpoint_wind(handle.head_token(), boundary_ordinal=1)
        value = {"schema": "masy423.Root.fresh_active_process.v1", "store": str(root / "store"),
            "store_file_sha256": handle.report()["store_file_sha256"], "head_token": pause.head_token,
            "expected": expected}
    gc.collect()
    file = root / "expected.json"
    with file.open("xb") as output:
        output.write(oi._json(value))
    file.chmod(0o444)
    return {"expectation_path": str(file), "expectation_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "protocol": "Root_fixed_writer_reaped_before_fresh_reader_no_nested_process"}


class FreshReader(unittest.TestCase):
    def test_fresh_process_recovers_genuine_wind_child_and_finishes_once(self):
        path = Path(os.environ["MASY423_FRESH_EXPECTATION"])
        expected_sha = os.environ["MASY423_FRESH_EXPECTATION_SHA256"]
        c._digest(expected_sha, "Root external fresh expectation")
        before = path.lstat()
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_mode & 0o222
                or not 1 <= before.st_size <= 1 << 20):
            raise AssertionError("readonly bounded external expectation required")
        raw = path.read_bytes()
        if len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected_sha:
            raise AssertionError("whole fresh expectation differs")
        after = path.lstat()
        if c._identity(before) != c._identity(after) or before.st_mtime_ns != after.st_mtime_ns or before.st_ctime_ns != after.st_ctime_ns:
            raise AssertionError("fresh expectation changed")
        value = json.loads(raw)
        self.assertEqual(value["schema"], "masy423.Root.fresh_active_process.v1")
        handle = c.recover_active_process_boundary_store(value["store"],
            expected_store_file_sha256=value["store_file_sha256"])
        try:
            self.assertEqual(handle.head_token(), value["head_token"])
            ack = handle.step(value["head_token"])
            self.assertEqual((ack.revision, ack.accepted_sequence), (2, 1))
            self.assertEqual(accepted_snapshot(handle), value["expected"])
            with mock.patch.object(wind, "resume_wind", side_effect=AssertionError("replayed resume")), \
                    mock.patch.object(wind_checkpoint, "load_wind_pause", side_effect=AssertionError("replayed LOAD")):
                self.assertEqual(handle.step(value["head_token"]), ack)
        finally:
            handle.close()


if __name__ == "__main__":
    unittest.main()
