"""Finite M1 metadata/refusal controls plus genuine P2-gated ownership checks.

Inert receivers below test only early metadata refusals; they are never native
success, a runtime grant or a replacement for genuine source binding.
"""
from __future__ import annotations

import dataclasses
import gc
import os
import threading
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

from masy import dense_boundary_store as store, dense_runner as dr, dense_water_pause as water
from masy import dense_wind_pause as pause
from maple.aeolian.scheduler import block
from check_dense_wind_pause import new_runner, reference, snapshot


def nominal():
    runner = object.__new__(dr.DenseRunner)
    runner._stop_receipt, runner._active_evidence, runner._next = None, None, 0
    return runner


def lease_for(runner):
    lease = pause._WindLease(weakref.ref(runner), weakref.ref(threading.current_thread()), os.getpid(),
                              0, None, 7.25, (), {}, phase='paused')
    lease.token = pause._WindToken()
    runner._active_evidence = lease
    return lease


class WindPauseMetadata(unittest.TestCase):
    def test_opaque_wrapper_has_no_owner_or_native_payload(self):
        token = pause._WindToken()
        self.assertEqual(pause._WindToken.__slots__, ())
        with self.assertRaises(TypeError): vars(token)
        with self.assertRaises(AttributeError): token.owner = object()
        with self.assertRaises(AttributeError): token.native_token = object()

    def test_callback_owns_only_two_plain_counters(self):
        callback = pause._callback(1, 3)
        self.assertEqual(tuple(type(c.cell_contents) for c in callback.__closure__), (int, int))
        self.assertIs(callback(), False)
        self.assertIs(callback(), True)
        self.assertIs(callback(), False)

    def test_closed_ordinal_types_and_increasing_original_counter(self):
        for value in (True, False, None, 0, -1, 5, 1.0, '1'):
            with self.subTest(value=value), self.assertRaises(dr.DenseRunnerError):
                pause._ordinal(value, 4)
        pause._ordinal(1, 4)
        pause._ordinal(4, 4, 3)
        with self.assertRaises(dr.DenseRunnerError): pause._ordinal(2, 4, 2)

    def test_active_dispatch_precedes_native_and_stop_in_step_run_property(self):
        runner = nominal()
        lease = lease_for(runner)
        with mock.patch.object(dr, '_native', side_effect=AssertionError('native')) as native, \
                mock.patch.object(dr.DenseRunner, '_stop', side_effect=AssertionError('stop')) as stop:
            for action in (runner.step, runner.run, lambda: runner.active_evidence):
                with self.subTest(action=action), self.assertRaises(dr.DenseRunnerError): action()
                self.assertIs(runner._active_evidence, lease)
                self.assertEqual(lease.phase, 'paused')
        native.assert_not_called(); stop.assert_not_called()

    def test_native_declaration_never_runs_for_wrong_wrapper_token(self):
        runner = nominal()
        lease = lease_for(runner)
        with mock.patch.object(pause, '_declare', side_effect=AssertionError('declaration')) as declare:
            for token in (None, object(), pause._WindToken()):
                with self.subTest(token=type(token)), self.assertRaises(dr.DenseRunnerError):
                    pause.resume_wind(runner, token)
        declare.assert_not_called()
        self.assertEqual(lease.phase, 'paused')

    def test_foreign_owner_pid_and_busy_metadata_preserve_token(self):
        runner, other = nominal(), nominal()
        lease = lease_for(runner)
        with self.assertRaises(dr.DenseRunnerError): pause._lease(other, lease.token)
        lease.pid += 1
        with self.assertRaises(dr.DenseRunnerError): pause._lease(runner, lease.token)
        lease.pid -= 1
        for phase in ('busy', 'finishing', 'halted', 'closed'):
            lease.phase = phase
            with self.subTest(phase=phase), self.assertRaises(dr.DenseRunnerError): pause._lease(runner, lease.token)
        lease.phase = 'paused'
        self.assertIs(pause._lease(runner, lease.token), lease)

    def test_foreign_thread_metadata_refuses_without_consume(self):
        runner = nominal()
        lease = lease_for(runner)
        errors = []
        def foreign():
            try: pause._lease(runner, lease.token)
            except dr.DenseRunnerError as exc: errors.append(exc.code)
        thread = threading.Thread(target=foreign)
        thread.start(); thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ['wind_token'])
        self.assertEqual(lease.phase, 'paused')
        self.assertIs(pause._lease(runner, lease.token), lease)

    def test_weak_runner_binding_and_empty_public_token_do_not_keep_runner(self):
        runner = nominal()
        lease = lease_for(runner)
        token, ref = lease.token, weakref.ref(runner)
        del runner
        gc.collect()
        self.assertIsNone(ref())
        self.assertIsNone(lease.owner())
        self.assertIs(type(token), pause._WindToken)

    def test_old_committed_reference_is_not_prototype_activation(self):
        api = SimpleNamespace(store=store)
        with mock.patch.object(store, '_runtime_binding', side_effect=AssertionError('old binding')) as bind:
            with self.assertRaises(dr.DenseRunnerError):
                pause._runtime(api, {'kind': 'cpu_committed_runtime_v1'})
        bind.assert_not_called()

    def test_water_cross_token_refuses_without_owning_native_declaration(self):
        runner = nominal()
        lease = lease_for(runner)
        with mock.patch.object(water, '_declare', side_effect=AssertionError('wet declaration')) as declare:
            with self.assertRaises(dr.DenseRunnerError): water.resume_water(runner, lease.token)
        declare.assert_not_called()
        self.assertIs(runner._active_evidence, lease)


class WindPauseNativeAdmission(unittest.TestCase):
    def paused(self):
        runner = new_runner()
        token = pause.pause_wind(runner, runtime_reference=reference(), boundary_ordinal=1)
        self.assertIs(type(token), pause._WindToken)
        return runner, token

    def test_pause_limit_and_old_token_reuse_refuse_without_consuming_current(self):
        runner, old = self.paused()
        current = pause.resume_wind(runner, old, boundary_ordinal=2)
        lease = runner._active_evidence
        self.assertIs(type(current), pause._WindToken)
        with self.assertRaises(dr.DenseRunnerError): pause.resume_wind(runner, old)
        with self.assertRaises(dr.DenseRunnerError): pause.resume_wind(runner, current, boundary_ordinal=3)
        self.assertIs(lease.token, current)
        self.assertEqual(lease.phase, 'paused')
        self.assertEqual(pause.resume_wind(runner, current)['status'], 'accepted')

    def test_existing_capture_profiles_refuse_active_before_payload_work(self):
        runner, token = self.paused()
        from masy import dense_runner_checkpoint as checkpoint, dense_original_inputs as oi
        with self.assertRaises(checkpoint.BoundaryCheckpointError) as b:
            checkpoint._runner_admission(runner, checkpoint.BoundaryImageLimits())
        self.assertEqual(b.exception.code, 'not_quiescent')
        with self.assertRaises(oi.OriginalInputsError) as original:
            oi.capture_original_inputs(runner, source_identity=None)
        self.assertEqual(original.exception.code, 'not_genesis')
        with mock.patch.object(store.os, 'mkdir', side_effect=AssertionError('write')) as mkdir:
            with self.assertRaises(checkpoint.BoundaryCheckpointError) as c:
                store.create_boundary_store('must_not_be_created', runner=runner, originals=None,
                    runtime_reference=None, storage_contract=store.StorageContract(True, True))
        self.assertEqual(c.exception.code, 'not_quiescent')
        mkdir.assert_not_called()
        self.assertIs(runner._active_evidence.token, token)
        self.assertEqual(runner._active_evidence.phase, 'paused')

    def test_changed_original_entry_metadata_halts_before_native_resume(self):
        runner, token = self.paused()
        lease = runner._active_evidence
        lease.prepared.upper += 1
        with mock.patch.object(block, 'resume_wind_block', side_effect=AssertionError('native')) as native:
            result = pause.resume_wind(runner, token)
        self.assertNotEqual(result['status'], 'accepted')
        native.assert_not_called()
        self.assertTrue(runner.active_evidence['unknown_current_endpoint'])

    def test_external_accepted_mutation_keeps_external_guarantee(self):
        runner, token = self.paused()
        leaf = runner._soil
        old, writable = float(leaf.flat[0]), leaf.flags.writeable
        leaf.flags.writeable = True
        try:
            leaf.flat[0] = old + 0.001
            result = pause.resume_wind(runner, token)
        finally:
            leaf.flat[0] = old
            leaf.flags.writeable = writable
        self.assertEqual(result['guarantee'], dr._EXTERNAL)
        self.assertNotEqual(result['status'], 'accepted')
        self.assertEqual(runner._next, 0)

    def test_low_adapter_reservation_refuses_before_original_stage(self):
        runner = new_runner(work=1 << 20)
        with mock.patch.object(dr, '_wind_enter', side_effect=AssertionError('stage')) as enter:
            result = pause.pause_wind(runner, runtime_reference=reference(), boundary_ordinal=1)
        enter.assert_not_called()
        self.assertEqual(result['code'], 'work_bytes_cap')
        self.assertEqual(runner._next, 0)

    def test_production_and_dynamic_profiles_refuse_before_stage(self):
        for attribute in ('_production', '_vegetation_plan'):
            runner = new_runner()
            setattr(runner, attribute, object())
            with self.subTest(attribute=attribute), \
                    mock.patch.object(dr, '_wind_enter', side_effect=AssertionError('stage')) as enter, \
                    self.assertRaises(dr.DenseRunnerError):
                pause.pause_wind(runner, runtime_reference=reference(), boundary_ordinal=1)
            enter.assert_not_called()
            self.assertIsNone(runner._active_evidence)

    def test_binding_reference_drift_halts_before_native_resume(self):
        runner, token = self.paused()
        lease = runner._active_evidence
        lease.runtime_bytes = b'{}'
        with mock.patch.object(block, 'resume_wind_block', side_effect=AssertionError('native')) as native:
            result = pause.resume_wind(runner, token)
        native.assert_not_called()
        self.assertNotEqual(result['status'], 'accepted')
        self.assertTrue(runner.active_evidence['unknown_current_endpoint'])


if __name__ == '__main__':
    unittest.main()
