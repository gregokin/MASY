"""SOURCE-authored StageA checks; Root alone admits collection/execution.

The unchanged owning water fixture supplies real Native constructors and an
externally pinned runtime reference. A historical committed lock cannot certify
this modified MASY Source. No Native child process is launched by this module.
"""
from __future__ import annotations

import contextlib
import dataclasses
import gc
import hashlib
import json
import os
import stat
import tempfile
import threading
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import check_dense_water_pause as q
import numpy as np
from masy import dense_active_checkpoint as a
from masy import dense_bed as db
from masy import dense_boundary_store as c
from masy import dense_original_inputs as oi
from masy import dense_runner as dr
from masy import dense_water_pause as pause


@contextlib.contextmanager
def store_at(path):
    """Real newly captured original inputs, C creation and accepted parent."""
    runtime = q.reference()
    runner = q.new_runner()
    source, _environment = c._runtime_binding(runtime, c._DEFAULT)
    originals = oi.capture_original_inputs(runner, source_identity=source)
    handle = c.create_boundary_store(path, runner=runner, originals=originals,
        runtime_reference=runtime, storage_contract=c.StorageContract(True, True))
    del runner, originals
    try:
        yield handle, runtime
    finally:
        handle.close()


@contextlib.contextmanager
def borrowed(handle):
    with c.borrow_water_parent(handle, expected_parent=handle.report()["head_token"]) as parent:
        runner = parent.take_runner()
        yield parent, runner


def accepted_reference():
    runner = q.new_runner()
    complete, evidence = dr._complete_event, []
    def observe(*args, **kwargs):
        result = complete(*args, **kwargs)
        evidence.append(q.native_evidence(result))
        return result
    with mock.patch.object(dr, "_complete_event", side_effect=observe):
        receipt = runner.step()
    if receipt["status"] != "accepted" or len(evidence) != 1:
        raise AssertionError("real uninterrupted Native water completion required")
    return {"snapshot": q.snapshot(runner), "native": evidence[0]}


def finish_cold(runner, token):
    complete, evidence = dr._complete_event, []
    def observe(*args, **kwargs):
        if kwargs.get("continuation") is None:
            raise AssertionError("cold continuation must never invoke fresh event entry")
        result = complete(*args, **kwargs)
        evidence.append(q.native_evidence(result))
        return result
    with mock.patch.object(dr, "_complete_event", side_effect=observe), \
            mock.patch.object(dr, "_rain_enter", side_effect=AssertionError("fresh MASY entry")):
        receipt = pause.resume_water(runner, token)
    if receipt["status"] != "accepted" or len(evidence) != 1:
        raise AssertionError("genuine cold Native dry completion required")
    return {"snapshot": q.snapshot(runner), "native": evidence[0]}


def export_in_context(parent, runner, runtime, attempt):
    token = pause.pause_water(runner, runtime_reference=runtime, boundary_ordinal=1)
    if type(token) is not pause._WaterToken:
        raise AssertionError("genuine unfinished original-boundary wet token required")
    lease = runner._active_evidence
    before = q.snapshot(runner)
    original_progress = lease.progress
    sha = a.export_water_pause(parent, runner, token, attempt=attempt)
    if (runner._active_evidence is not lease or lease.token is not token or lease.phase != "paused"
            or lease.progress is not original_progress or q.snapshot(runner) != before):
        raise AssertionError("non-consuming export changed accepted or private owner")
    return sha, token


def fresh_metadata(path, expected):
    path = Path(path)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_mode & 0o222
            or not 1 <= before.st_size <= 1 << 20):
        raise AssertionError("Root-owned readonly bounded fresh-stage expectation required")
    raw = path.read_bytes()
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_nlink")
    if hashlib.sha256(raw).hexdigest() != expected or any(getattr(before, k) != getattr(after, k) for k in fields):
        raise AssertionError("whole external expectation SHA/stat drift")
    return json.loads(raw)


def write_fresh_process_fixture(directory):
    """One separately gated Root writer; return metadata, never spawn a reader.

    Root must reap this worker, retain all files, make the expectation readonly,
    and independently pin its actual whole SHA before a new reader worker.
    """
    root = Path(directory)
    root.mkdir(mode=0o700, exist_ok=False)
    with store_at(root / "store") as (handle, runtime):
        parent_token = handle.report()["head_token"]
        store_sha = handle.report()["store_file_sha256"]
        with borrowed(handle) as (parent, runner):
            export_sha, _token = export_in_context(parent, runner, runtime, root / "attempt")
        del runner, parent, _token
        if handle.report()["head_token"] != parent_token:
            raise AssertionError("StageA writer published an active HEAD")
    gc.collect()
    reference = accepted_reference()  # private writer state has been released
    expectation = {"schema": "masy362.Root.fresh_stage.v1", "store": str(root / "store"),
                   "store_file_sha256": store_sha, "parent_token": parent_token,
                   "attempt": str(root / "attempt"), "export_sha256": export_sha,
                   "expected": reference}
    path = root / "expected.json"
    with path.open("xb") as target:
        target.write(oi._json(expectation))
    path.chmod(0o444)
    return {"expectation_path": str(path), "expectation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "protocol": "writer_only_no_nested_Native_process", "active_HEAD_published": False}


class EntryMetadataProtocols(unittest.TestCase):
    def test_external_digest_and_canonical_float_refusals(self):
        for value in ("nan", "0x0p+0", True, "0x1p+999999", "x" * 65):
            with self.subTest(value=value), self.assertRaises(dr.DenseRunnerError):
                a._float(value)
        self.assertNotEqual(a._float("-0x0.0p+0").hex(), a._float("0x0.0p+0").hex())

    def test_integer_bool_and_bitlength_refuse(self):
        for value in (True, -1, 1 << 300):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(dr.DenseRunnerError):
                a._integer(value, 10, "authored integer")

    def test_foreign_parent_is_not_authority(self):
        with self.assertRaises(dr.DenseRunnerError) as caught:
            a._parent(SimpleNamespace(check=lambda _runner: None), object())
        self.assertEqual(caught.exception.code, "active_parent")

    def test_cold_witness_is_not_a_stage_or_public_constructor(self):
        with self.assertRaises(dr.DenseRunnerError):
            dr._ColdRainStage()
        self.assertFalse(issubclass(dr._ColdRainStage, db.DenseStage))

    def test_capsule_metadata_bounds_precede_serialization(self):
        with mock.patch.object(oi, "_json", side_effect=AssertionError("serialize before admission")):
            with self.assertRaises(c.BoundaryStoreError):
                a._metadata({"authored": "x" * 1024}, 32, "bounded negative")

    def test_empty_array_and_signed_zero_raw_projection(self):
        empty = np.empty((0, 3), dtype=np.float64)
        self.assertEqual(a._array_hash(empty, np)["sha256"], hashlib.sha256(b"").hexdigest())
        self.assertNotEqual(a._array_hash(np.array([0.0]), np)["sha256"],
                            a._array_hash(np.array([-0.0]), np)["sha256"])


class GenuineColdWater(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = accepted_reference()

    def test_first_genuine_wet_save_audit_load_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with store_at(root / "store") as (handle, runtime):
                head, before = handle.report()["head_token"], (root / "store/HEAD").read_bytes()
                with borrowed(handle) as (parent, runner):
                    sha, _token = export_in_context(parent, runner, runtime, root / "attempt")
                del runner, parent, _token
                gc.collect()
                with borrowed(handle) as (parent, runner):
                    unchanged = q.snapshot(runner)
                    with mock.patch.object(dr, "_rain_enter", side_effect=AssertionError("LOAD fresh entry")), \
                            mock.patch.object(dr, "_complete_event", side_effect=AssertionError("LOAD transport")):
                        token = a.load_water_pause(parent, runner, attempt=root / "attempt",
                            expected_export_sha256=sha, runtime_reference=runtime)
                    self.assertEqual(q.snapshot(runner), unchanged)
                    self.assertIs(type(runner._active_evidence.prepared), a._ColdPrepared)
                    self.assertIsNot(type(runner._active_evidence.prepared.stage), db.DenseStage)
                    self.assertEqual(finish_cold(runner, token), self.expected)
                    with self.assertRaises(dr.DenseRunnerError):
                        pause.resume_water(runner, token)
                self.assertEqual(handle.report()["head_token"], head)
                self.assertEqual((root / "store/HEAD").read_bytes(), before)

    def test_actual_parent_reentrancy_and_one_shot_transfer(self):
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, _runtime):
            with borrowed(handle) as (parent, runner):
                with self.assertRaises(c.BoundaryStoreError):
                    parent.take_runner()
                with self.assertRaises(c.BoundaryStoreError):
                    c.borrow_water_parent(handle, expected_parent=handle.report()["head_token"])
                with self.assertRaises(c.BoundaryStoreError):
                    handle.close()
                parent.check(runner)
            self.assertFalse(handle._entered)

    def test_borrow_context_keeps_no_transferred_runner(self):
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, _runtime):
            with c.borrow_water_parent(handle, expected_parent=handle.report()["head_token"]) as parent:
                runner = parent.take_runner()
                reference = weakref.ref(runner)
                del runner
                gc.collect()
                self.assertIsNone(reference())
            self.assertEqual(parent.cleanup_record[0], "taken_collected")

    def test_busy_export_close_refuses_before_releasing_parent(self):
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, runtime):
            with borrowed(handle) as (parent, runner):
                token = pause.pause_water(runner, runtime_reference=runtime, boundary_ordinal=1)
                self.assertIs(type(token), pause._WaterToken)
                lease = runner._active_evidence
                for phase in ("busy", "finishing", "exporting"):
                    with self.subTest(phase=phase):
                        lease.phase = phase  # deliberate private-metadata negative
                        try:
                            with self.assertRaises(c.BoundaryStoreError):
                                parent.close()
                            self.assertTrue(handle._entered)
                            self.assertEqual(parent.cleanup_record[1][1], "water_busy")
                        finally:
                            lease.phase = "paused"

    def test_wrong_thread_parent_refuses_before_head_reader(self):
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, _runtime):
            with borrowed(handle) as (parent, runner):
                errors = []
                def foreign():
                    try:
                        parent.check(runner)
                    except BaseException as error:
                        errors.append(getattr(error, "code", None))
                worker = threading.Thread(target=foreign)
                worker.start()
                worker.join()
                self.assertEqual(errors, ["handle_binding"])
                parent.check(runner)

    def test_native_failure_retains_attempt_and_valid_nonconsumed_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with store_at(root / "store") as (handle, runtime), borrowed(handle) as (parent, runner):
                token = pause.pause_water(runner, runtime_reference=runtime, boundary_ordinal=1)
                self.assertIs(type(token), pause._WaterToken)
                checkpoint = a._declare()[1]
                with mock.patch.object(checkpoint, "save_checkpoint", side_effect=OSError("authored SAVE failure")):
                    with self.assertRaisesRegex(OSError, "authored SAVE failure"):
                        a.export_water_pause(parent, runner, token, attempt=root / "failed")
                self.assertTrue((root / "failed/entry/capsule.json").is_file())
                self.assertIs(pause._lease(runner, token).token, token)

    def test_mutating_native_failure_halts_and_preserves_primary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with store_at(root / "store") as (handle, runtime), borrowed(handle) as (parent, runner):
                token = pause.pause_water(runner, runtime_reference=runtime, boundary_ordinal=1)
                self.assertIs(type(token), pause._WaterToken)
                checkpoint = a._declare()[1]
                def bad_save(*_args, **_kwargs):
                    result = runner._active_evidence.progress.result
                    object.__setattr__(result, "n_accepted_steps", result.n_accepted_steps + 1)
                    raise OSError("authored mutating SAVE failure")
                with mock.patch.object(checkpoint, "save_checkpoint", side_effect=bad_save):
                    with self.assertRaisesRegex(OSError, "authored mutating SAVE failure"):
                        a.export_water_pause(parent, runner, token, attempt=root / "failed")
                self.assertIsNotNone(runner._stop_receipt)
                with self.assertRaises(dr.DenseRunnerError):
                    pause.resume_water(runner, token)

    def test_space_refusal_precedes_attempt_creation_and_native_save(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with store_at(root / "store") as (handle, runtime), borrowed(handle) as (parent, runner):
                token = pause.pause_water(runner, runtime_reference=runtime, boundary_ordinal=1)
                checkpoint = a._declare()[1]
                with mock.patch.object(os, "fstatvfs", return_value=SimpleNamespace(f_bavail=0, f_frsize=4096, f_favail=0)), \
                        mock.patch.object(checkpoint, "save_checkpoint", side_effect=AssertionError("SAVE after cap")):
                    with self.assertRaises(dr.DenseRunnerError) as caught:
                        a.export_water_pause(parent, runner, token, attempt=root / "denied")
                self.assertEqual(caught.exception.code, "active_disk_space")
                self.assertFalse((root / "denied").exists())
                pause._lease(runner, token)

    def test_source_or_body_refusal_precedes_child_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with store_at(root / "store") as (handle, runtime):
                with borrowed(handle) as (parent, runner):
                    sha, _token = export_in_context(parent, runner, runtime, root / "attempt")
                with borrowed(handle) as (parent, runner):
                    checkpoint = a._declare()[1]
                    with mock.patch.object(checkpoint, "load_checkpoint", side_effect=AssertionError("Native LOAD too early")):
                        with self.assertRaises(c.BoundaryStoreError):
                            a.load_water_pause(parent, runner, attempt=root / "attempt",
                                expected_export_sha256=sha, runtime_reference={})
                        with self.assertRaises(dr.DenseRunnerError):
                            a.load_water_pause(parent, runner, attempt=root / "attempt",
                                expected_export_sha256="0" * 64, runtime_reference=runtime)
                    self.assertIsNone(runner._active_evidence)

    def test_native_raw_file_tamper_refuses_without_mint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with store_at(root / "store") as (handle, runtime):
                with borrowed(handle) as (parent, runner):
                    sha, _token = export_in_context(parent, runner, runtime, root / "attempt")
                target = root / "attempt/native/checkpoint.json"
                target.chmod(0o644)
                with target.open("ab") as stream:
                    stream.write(b" ")
                target.chmod(0o444)
                with borrowed(handle) as (parent, runner):
                    checkpoint = a._declare()[1]
                    with mock.patch.object(checkpoint, "load_checkpoint", side_effect=AssertionError("LOAD tampered child")):
                        with self.assertRaises(dr.DenseRunnerError):
                            a.load_water_pause(parent, runner, attempt=root / "attempt",
                                expected_export_sha256=sha, runtime_reference=runtime)
                    self.assertIsNone(runner._active_evidence)

    def test_capsule_parent_bool_alias_is_not_integer_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with store_at(root / "store") as (handle, runtime), borrowed(handle) as (parent, runner):
                _sha, _token = export_in_context(parent, runner, runtime, root / "attempt")
                capsule = json.loads((root / "attempt/entry/capsule.json").read_bytes())
                self.assertIs(type(capsule["parent"]["accepted_sequence"]), int)
                for alias in (False, 0.0):
                    with self.subTest(alias_type=type(alias).__name__):
                        capsule["parent"]["accepted_sequence"] = alias
                        with mock.patch.object(pause, "_plan", side_effect=AssertionError("plan before typed parent")):
                            with self.assertRaises(dr.DenseRunnerError) as caught:
                                a._admit_capsule(parent, runner, capsule, a._declare()[0], c)
                        self.assertEqual(caught.exception.code, "active_parent_changed")

    def test_cold_invalid_ordinal_is_nonconsuming(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with store_at(root / "store") as (handle, runtime):
                with borrowed(handle) as (parent, runner):
                    sha, _token = export_in_context(parent, runner, runtime, root / "attempt")
                with borrowed(handle) as (parent, runner):
                    token = a.load_water_pause(parent, runner, attempt=root / "attempt",
                        expected_export_sha256=sha, runtime_reference=runtime)
                    with self.assertRaises(dr.DenseRunnerError):
                        pause.resume_water(runner, token, boundary_ordinal=True)
                    pause._lease(runner, token)
                    self.assertEqual(finish_cold(runner, token), self.expected)

    def test_actual_cold_entry_change_is_nonconsuming(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with store_at(root / "store") as (handle, runtime):
                with borrowed(handle) as (parent, runner):
                    sha, _token = export_in_context(parent, runner, runtime, root / "attempt")
                with borrowed(handle) as (parent, runner):
                    token = a.load_water_pause(parent, runner, attempt=root / "attempt",
                        expected_export_sha256=sha, runtime_reference=runtime)
                    facts = runner._active_evidence.prepared.facts
                    original = facts["entry"]["rerouted"]
                    facts["entry"]["rerouted"] = not original
                    try:
                        with self.assertRaises(dr.DenseRunnerError):
                            pause.resume_water(runner, token)
                        pause._lease(runner, token)
                    finally:
                        facts["entry"]["rerouted"] = original
                    self.assertEqual(finish_cold(runner, token), self.expected)

    def test_postconsume_cold_accessor_failure_halts_truthfully(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with store_at(root / "store") as (handle, runtime):
                head, head_bytes = handle.report()["head_token"], (root / "store/HEAD").read_bytes()
                with borrowed(handle) as (parent, runner):
                    sha, _token = export_in_context(parent, runner, runtime, root / "attempt")
                with borrowed(handle) as (parent, runner):
                    token = a.load_water_pause(parent, runner, attempt=root / "attempt",
                        expected_export_sha256=sha, runtime_reference=runtime)
                    lease = pause._lease(runner, token)
                    unchanged = q.snapshot(runner)
                    failure = dr.DenseRunnerError("authored_accessor", "failure")
                    with mock.patch.object(a, "_native_fields", side_effect=failure), \
                            mock.patch.object(dr, "_complete_event", side_effect=AssertionError("Native after accessor failure")) as complete:
                        receipt = pause.resume_water(runner, token)
                    complete.assert_not_called()
                    self.assertEqual(receipt, {
                        "schema": dr.RECEIPT_SCHEMA, "index": lease.index, "kind": "rain",
                        "status": "runner_error", "code": "water_pause_failure",
                        "detail": dr._detail(failure), "exception_type": "DenseRunnerError",
                        "native_start": dr.encode_time(lease.origin), "unaccepted_elapsed_s": None,
                        "accepted": unchanged["report"]["accepted"], "guarantee": dr._UNCHANGED})
                    self.assertEqual(lease.phase, "halted")
                    self.assertIsNone(lease.token)
                    self.assertIsNone(lease.prepared)
                    self.assertIsNone(lease.progress)
                    self.assertIsNone(runner._active_evidence)
                    self.assertEqual(runner.stop_receipt, receipt)
                    with self.assertRaises(dr.DenseRunnerError) as caught:
                        pause.resume_water(runner, token)
                    self.assertEqual(caught.exception.code, "water_token")
                    # A truthful stop changes only report status/stop; physical state,
                    # accepted cursor, all clocks, receipts and totals remain exact.
                    unchanged["report"]["status"] = "stopped"
                    unchanged["report"]["stop"] = receipt
                    self.assertEqual(q.snapshot(runner), unchanged)
                self.assertEqual(handle.report()["head_token"], head)
                self.assertEqual((root / "store/HEAD").read_bytes(), head_bytes)


class FreshProcessReader(unittest.TestCase):
    def test_Root_reaped_writer_then_fresh_native_reader(self):
        # This group is dormant until Root supplies BOTH externally pinned
        # expectation metadata and the separately admitted genuine runtime.
        row = fresh_metadata(os.environ["MASY362_FRESH"], os.environ["MASY362_FRESH_SHA256"])
        self.assertEqual(row["schema"], "masy362.Root.fresh_stage.v1")
        with contextlib.closing(c.recover_boundary_store(row["store"], expected_store_file_sha256=row["store_file_sha256"])) as handle:
            self.assertEqual(handle.report()["head_token"], row["parent_token"])
            with borrowed(handle) as (parent, runner):
                token = a.load_water_pause(parent, runner, attempt=row["attempt"],
                    expected_export_sha256=row["export_sha256"], runtime_reference=q.reference())
                self.assertEqual(finish_cold(runner, token), row["expected"])
            self.assertEqual(handle.report()["head_token"], row["parent_token"])
