"""SOURCE-authored genuine Native cold-wind checks; Root alone may execute.

All fixtures are synthetic interface cases. No historical runtime is a default;
MASY423_RUNTIME must name a fresh qualified source_manifest_prototype composition.
No child process, store clone, codec reimplementation or mocked positive exists.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fixture_dense_alternation as f
import check_dense_water_pause as water_fixture
from maple.aeolian.scheduler import block
from masy import dense_active_wind_checkpoint as wind_checkpoint
from masy import dense_boundary_store as c
from masy import dense_original_inputs as oi
from masy import dense_runner as dr
from masy import dense_wind_pause as wind


def reference():
    path = Path(os.environ["MASY423_RUNTIME"])
    expected = os.environ["MASY423_RUNTIME_SHA256"]
    c._digest(expected, "Root qualification runtime reference")
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_mode & 0o222
            or not 1 <= before.st_size <= 32768):
        raise AssertionError("readonly bounded externally pinned runtime reference required")
    raw = path.read_bytes()
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
    if (len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected
            or any(getattr(before, x) != getattr(after, x) for x in fields)):
        raise AssertionError("whole runtime reference changed")
    value = json.loads(raw)
    if value.get("kind") != "source_manifest_prototype":
        raise AssertionError("genuine new wind development composition required, no lock fallback")
    return value


def new_runner(*, events=None, commit=None):
    """Original Tiny C6/V3/K6 Native constructors; explicit approved OFF option."""
    actual = f.build_dense_bed
    def at_origin(**kwargs):
        return actual(**dict(kwargs, initial_time_s=7.25))
    with mock.patch.object(f, "build_dense_bed", side_effect=at_origin):
        bed = f.build_bed(commit_spec=commit)
    config = f.wind_config(bed, outer_timestep_s=0.025)
    config = dataclasses.replace(config,
        dormancy_spec=dataclasses.replace(config.dormancy_spec, enabled=False))
    control = f.runner_control(sediment_control=dataclasses.replace(f.sediment_control(),
                                                                   force_final_commit=False))
    return f.make_runner(bed, events or f.schedule(f.wind(duration_s=1.0)), config=config, control=control)


def snapshot(runner):
    return water_fixture.snapshot(runner)


@contextlib.contextmanager
def store_at(path, *, events=None, combined=False, commit=None):
    runtime = reference()
    runner = new_runner(events=events, commit=commit)
    source, _environment = c._runtime_binding(runtime, c._DEFAULT)
    originals = oi.capture_original_inputs(runner, source_identity=source)
    create = c.create_active_process_boundary_store if combined else c.create_boundary_store
    handle = create(path, runner=runner, originals=originals,
                    runtime_reference=runtime, storage_contract=c.StorageContract(True, True))
    del runner, originals
    try:
        yield handle, runtime
    finally:
        handle.close()


@contextlib.contextmanager
def borrowed(handle):
    with c.borrow_water_parent(handle, expected_parent=handle.head_token()) as parent:
        runner = parent.take_runner()
        yield parent, runner


def exported(parent, runner, runtime, attempt):
    before = snapshot(runner)
    token = wind.pause_wind(runner, runtime_reference=runtime, boundary_ordinal=1)
    if type(token) is not wind._WindToken:
        raise AssertionError("genuine unfinished wind token required")
    lease, progress = runner._active_evidence, runner._active_evidence.progress
    sha = wind_checkpoint.export_wind_pause(parent, runner, token, attempt=attempt)
    if (runner._active_evidence is not lease or lease.token is not token
            or lease.phase != "paused" or lease.progress is not progress or snapshot(runner) != before):
        raise AssertionError("export consumed or changed the actual paused owner/accepted boundary")
    return sha, token


def original_child(path):
    names = ("export.json", "entry/capsule.json", "native/continuation.bin")
    return {name: c.runtime._hash_path(Path(path) / name, budget=8 << 20)[0] for name in names}


def ordinary(*, events=None, commit=None):
    runner = new_runner(events=events, commit=commit)
    report = runner.run()
    if report["status"] != "complete":
        raise AssertionError("genuine original native events must complete")
    answer = snapshot(runner)
    del runner
    return answer


class GenuineColdWind(unittest.TestCase):
    def test_first_genuine_codec_cold_resume_exactly_matches_native(self):
        expected = ordinary()
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, runtime):
            root = Path(temporary)
            head = (handle._root / "HEAD").read_bytes()
            with borrowed(handle) as (parent, runner):
                sha, token = exported(parent, runner, runtime, root / "attempt")
            del parent, runner, token
            retained = original_child(root / "attempt")
            with borrowed(handle) as (parent, runner):
                before = snapshot(runner)
                # These are negative transport tripwires during LOAD only.
                # The genuine codec and its native validators remain real.
                with mock.patch.object(block, "advance_wind_block_pausable", side_effect=AssertionError("fresh begin")), \
                        mock.patch.object(block, "resume_wind_block", side_effect=AssertionError("advance on LOAD")), \
                        mock.patch.object(dr, "_finalize_wind_event", side_effect=AssertionError("finalize on LOAD")):
                    token = wind_checkpoint.load_wind_pause(parent, runner, attempt=root / "attempt",
                        expected_export_sha256=sha, runtime_reference=runtime)
                self.assertEqual(snapshot(runner), before)
                self.assertIsNotNone(runner._active_evidence.cold_binding)
                with mock.patch.object(dr, "_wind_enter", side_effect=AssertionError("fresh MASY entry")):
                    result = wind.resume_wind(runner, token)
                self.assertEqual(result["status"], "accepted")
                self.assertEqual(snapshot(runner), expected)
                with self.assertRaises(dr.DenseRunnerError):
                    wind.resume_wind(runner, token)
            self.assertEqual((handle._root / "HEAD").read_bytes(), head)
            self.assertEqual(original_child(root / "attempt"), retained)

    def test_cold_reexport_keeps_original_plan_and_original_files(self):
        expected = ordinary()
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, runtime):
            root = Path(temporary)
            with borrowed(handle) as (parent, runner):
                first, _token = exported(parent, runner, runtime, root / "first")
            before = original_child(root / "first")
            with borrowed(handle) as (parent, runner):
                token = wind_checkpoint.load_wind_pause(parent, runner, attempt=root / "first",
                    expected_export_sha256=first, runtime_reference=runtime)
                second_token = wind.resume_wind(runner, token, boundary_ordinal=2)
                self.assertIs(type(second_token), wind._WindToken)
                self.assertEqual((runner._active_evidence.ordinal, runner._active_evidence.pauses), (2, 2))
                second = wind_checkpoint.export_wind_pause(parent, runner, second_token, attempt=root / "second")
            self.assertEqual(original_child(root / "first"), before)
            with borrowed(handle) as (parent, runner):
                token = wind_checkpoint.load_wind_pause(parent, runner, attempt=root / "second",
                    expected_export_sha256=second, runtime_reference=runtime)
                self.assertEqual(wind.resume_wind(runner, token)["status"], "accepted")
                self.assertEqual(snapshot(runner), expected)
            self.assertEqual(original_child(root / "first"), before)

    def test_completed_binding_changes_exactly_two_equal_control_references(self):
        expected = ordinary()
        actual_bind = wind_checkpoint.bind_completed_result
        seen = []
        def observe(owner, lease, result):
            physical = result.state.physical
            self.assertIsNot(physical.vegetation, owner._bed.vegetation)
            self.assertIsNot(physical.threshold_modifier_state, owner._bed.threshold_modifier_state)
            # A forged value is negative DATA, never accepted as a Native run.
            threshold = physical.threshold_modifier_state
            changed = threshold.clay_fraction.copy()
            changed.flat[0] += 0.125
            forged = dataclasses.replace(result, state=dataclasses.replace(result.state,
                physical=dataclasses.replace(physical, threshold_modifier_state=dataclasses.replace(
                    threshold, clay_fraction=changed))))
            with self.assertRaises(dr.DenseRunnerError):
                actual_bind(owner, lease, forged)
            answer = actual_bind(owner, lease, result)
            self.assertIs(answer.state.physical.vegetation, owner._bed.vegetation)
            self.assertIs(answer.state.physical.threshold_modifier_state, owner._bed.threshold_modifier_state)
            for field in dataclasses.fields(physical):
                if field.name not in ("vegetation", "threshold_modifier_state"):
                    self.assertIs(getattr(answer.state.physical, field.name), getattr(physical, field.name))
            seen.append(True)
            return answer
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, runtime):
            root = Path(temporary)
            with borrowed(handle) as (parent, runner):
                sha, _token = exported(parent, runner, runtime, root / "attempt")
            with borrowed(handle) as (parent, runner):
                token = wind_checkpoint.load_wind_pause(parent, runner, attempt=root / "attempt",
                    expected_export_sha256=sha, runtime_reference=runtime)
                with mock.patch.object(wind_checkpoint, "bind_completed_result", side_effect=observe):
                    result = wind.resume_wind(runner, token)
                self.assertEqual(result["status"], "accepted")
                self.assertEqual(snapshot(runner), expected)
        self.assertEqual(seen, [True])

    def test_raw_truncation_refuses_before_native_restore_and_preserves_c_head(self):
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, runtime):
            root = Path(temporary)
            with borrowed(handle) as (parent, runner):
                sha, _token = exported(parent, runner, runtime, root / "attempt")
            head = (handle._root / "HEAD").read_bytes()
            file = root / "attempt/native/continuation.bin"
            file.chmod(0o600)
            with file.open("r+b") as output:
                output.truncate(1)
            file.chmod(0o444)
            from maple.aeolian.scheduler import baseline_continuation_codec as codec
            with borrowed(handle) as (parent, runner), \
                    mock.patch.object(codec, "restore_paused_continuation", side_effect=AssertionError("unadmitted restore")) as restore:
                with self.assertRaises(dr.DenseRunnerError):
                    wind_checkpoint.load_wind_pause(parent, runner, attempt=root / "attempt",
                        expected_export_sha256=sha, runtime_reference=runtime)
                restore.assert_not_called()
                self.assertIsNone(runner._active_evidence)
            self.assertEqual((handle._root / "HEAD").read_bytes(), head)

    def test_completed_original_boundary_wins_without_wind_export(self):
        expected = ordinary()
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, runtime):
            with borrowed(handle) as (_parent, runner), \
                    mock.patch.object(wind_checkpoint, "export_wind_pause", side_effect=AssertionError("no paused child")):
                result = wind.pause_wind(runner, runtime_reference=runtime, boundary_ordinal=4096)
                self.assertEqual(result["status"], "accepted")
                self.assertEqual(snapshot(runner), expected)

    def test_natural_commit_cold_resume_preserves_native_cadence(self):
        commit = f.TopographicCommitSpec(commit_interval_s=0.05)
        expected = ordinary(commit=commit)
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store", commit=commit) as (handle, runtime):
            root = Path(temporary)
            with borrowed(handle) as (parent, runner):
                sha, _token = exported(parent, runner, runtime, root / "attempt")
            with borrowed(handle) as (parent, runner):
                token = wind_checkpoint.load_wind_pause(parent, runner, attempt=root / "attempt",
                    expected_export_sha256=sha, runtime_reference=runtime)
                answer = wind.resume_wind(runner, token)
                self.assertEqual(answer["status"], "accepted")
                self.assertGreater(answer["native"]["n_commits_this_block"], 0)
                self.assertEqual(snapshot(runner), expected)

    def test_cold_active_owner_blocks_other_transport_and_busy_parent_close(self):
        actual_resume = block.resume_wind_block
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, runtime):
            root = Path(temporary)
            with borrowed(handle) as (parent, runner):
                sha, _token = exported(parent, runner, runtime, root / "attempt")
            with borrowed(handle) as (parent, runner):
                token = wind_checkpoint.load_wind_pause(parent, runner, attempt=root / "attempt",
                    expected_export_sha256=sha, runtime_reference=runtime)
                lease, before = runner._active_evidence, snapshot(runner)
                from masy import dense_water_pause as water
                with self.assertRaises(dr.DenseRunnerError):
                    runner.step()
                with self.assertRaises(dr.DenseRunnerError):
                    water.pause_water(runner, runtime_reference=runtime, boundary_ordinal=1)
                self.assertIs(runner._active_evidence, lease)
                self.assertIs(lease.token, token)
                self.assertEqual(snapshot(runner), before)
                def observe(*args, **kwargs):
                    with self.assertRaises(c.BoundaryStoreError) as caught:
                        parent.close()
                    self.assertEqual(caught.exception.code, "wind_busy")
                    self.assertEqual(parent._phase, "entered")
                    self.assertTrue(handle._entered)
                    return actual_resume(*args, **kwargs)
                with mock.patch.object(block, "resume_wind_block", side_effect=observe) as native:
                    self.assertEqual(wind.resume_wind(runner, token)["status"], "accepted")
                self.assertEqual(native.call_count, 1)

    def test_forged_attempt_binding_refuses_before_native_restore(self):
        # A rewritten capsule/descriptor below is explicitly forged negative
        # DATA. It is never used as a positive original-writer authority.
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, runtime):
            root = Path(temporary)
            with borrowed(handle) as (parent, runner):
                _sha, _token = exported(parent, runner, runtime, root / "attempt")
            from masy import dense_active_checkpoint as water_checkpoint
            capsule = root / "attempt/entry/capsule.json"
            facts = water_checkpoint._decode(c._read(capsule, 1 << 20), 1 << 20)
            facts["attempt"] = "f" * 64
            raw = oi._json(facts)
            capsule.chmod(0o600); capsule.write_bytes(raw); capsule.chmod(0o444)
            descriptor = root / "attempt/export.json"
            facts = water_checkpoint._decode(c._read(descriptor, 1 << 20), 1 << 20)
            facts["capsule_sha256"], facts["capsule_bytes"] = hashlib.sha256(raw).hexdigest(), len(raw)
            raw = oi._json(facts)
            descriptor.chmod(0o600); descriptor.write_bytes(raw); descriptor.chmod(0o444)
            expected_sha = hashlib.sha256(raw).hexdigest()
            from maple.aeolian.scheduler import baseline_continuation_codec as codec
            head = (handle._root / "HEAD").read_bytes()
            with borrowed(handle) as (parent, runner), \
                    mock.patch.object(codec, "restore_paused_continuation", side_effect=AssertionError("forged restore")) as restore:
                with self.assertRaises(dr.DenseRunnerError) as caught:
                    wind_checkpoint.load_wind_pause(parent, runner, attempt=root / "attempt",
                        expected_export_sha256=expected_sha, runtime_reference=runtime)
                self.assertEqual(caught.exception.code, "wind_capsule_path")
                restore.assert_not_called()
                self.assertIsNone(runner._active_evidence)
            self.assertEqual((handle._root / "HEAD").read_bytes(), head)

    def test_actual_borrow_close_abandons_paused_native_and_stales_wrapper(self):
        with tempfile.TemporaryDirectory() as temporary, store_at(Path(temporary) / "store") as (handle, runtime):
            with borrowed(handle) as (_parent, runner):
                token = wind.pause_wind(runner, runtime_reference=runtime, boundary_ordinal=1)
                lease = runner._active_evidence
                self.assertIs(type(token), wind._WindToken)
            self.assertEqual(lease.phase, "halted")
            self.assertIsNone(lease.native_token)
            self.assertFalse(handle._entered)
            with self.assertRaises(dr.DenseRunnerError):
                wind.resume_wind(runner, token)


if __name__ == "__main__":
    unittest.main()
