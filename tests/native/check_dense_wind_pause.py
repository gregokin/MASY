"""Synthetic native M1 checks; SOURCE_UNACTIVATED until Root's genuine P2 gate.

Root must provide a new exact three-package reference and raw SHA; historical
7f0/9eab and a source-table hash alone are not positive authority. No saved
model inputs/results are loaded. All positive native wrappers forward exact
arguments and return values to the original real functions.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import stat
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import fixture_dense_alternation as f
from maple.aeolian.scheduler import baseline, baseline_continuation as continuation, block
from maple.core.parameters.topographic_commit import TopographicCommitSpec
from masy import dense_bed as db, dense_runner as dr, dense_runner_checkpoint as boundary
from masy import dense_wind_pause as pause


def reference():
    path = Path(os.environ['MASY252_RUNTIME'])
    expected = os.environ['MASY252_RUNTIME_SHA256']
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise AssertionError('external Root P2 runtime SHA required')
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or s.st_mode & 0o222 or s.st_size > 32768:
        raise AssertionError('readonly bounded Root runtime metadata required')
    raw = path.read_bytes()
    if len(raw) != s.st_size or hashlib.sha256(raw).hexdigest() != expected:
        raise AssertionError('Root runtime reference changed')
    return json.loads(raw)


def new_runner(*, events=None, origin=7.25, duration=1.0, outer=0.025, commit=None, work=None):
    """Labelled synthetic interface fixture; no production calibration claim."""
    actual = f.build_dense_bed
    def at_origin(**kwargs):
        return actual(**dict(kwargs, initial_time_s=float(origin)))
    with mock.patch.object(f, 'build_dense_bed', side_effect=at_origin):
        bed = f.build_bed(commit_spec=commit)
    config = f.wind_config(bed, outer_timestep_s=float(outer))
    return f.make_runner(bed, events or f.schedule(f.wind(duration_s=duration)), config=config,
                         limit=f.limits() if work is None else f.limits(max_work_bytes=work))


def digest(value):
    return db._sha256(dr._canonical_input(np, value, 'exact_native'))


def snapshot(runner):
    held = runner._routing
    routing = None if held is None else boundary._cache_fingerprint({
        name: getattr(held, name) for name in ('graph', 'network', 'grid', 'terrain', 'source', 'template')})
    return {'bed': runner._bed.identity_sha256(), 'state': digest(runner._bed.state),
            'pool': digest(runner._pool), 'soil': digest(runner._soil), 'totals': digest(runner._totals),
            'routing': routing, 'report': runner.report()}


@contextlib.contextmanager
def counters():
    """Metadata-only, scoped observers; no arguments, returns or physics changed."""
    values = {name: 0 for name in ('begin', 'enter', 'accept', 'progress', 'resume', 'finalize')}
    values['progress_scalars'] = []  # at most two six-scalar observations, no native payload
    owners = {'begin': (baseline, '_begin_baseline_block'), 'enter': (dr, '_wind_enter'),
              'accept': (dr, '_wind_accept'), 'progress': (block, 'wind_block_progress'),
              'resume': (block, 'resume_wind_block'), 'finalize': (dr, '_finalize_wind_event')}
    with contextlib.ExitStack() as stack:
        for key, (owner, name) in owners.items():
            actual = getattr(owner, name)
            def observe(*args, _key=key, _actual=actual, **kwargs):
                values[_key] += 1
                result = _actual(*args, **kwargs)
                if _key == 'progress':
                    values['progress_scalars'].append(tuple(getattr(result, field) for field in pause._PROGRESS_FIELDS))
                return result
            stack.enter_context(mock.patch.object(owner, name, side_effect=observe))
        yield values


class WindPauseNativeParity(unittest.TestCase):
    def run_pair(self, *, ordinal=1, second=None, **kwargs):
        original, paused = new_runner(**kwargs), new_runner(**kwargs)
        with counters() as ordinary_calls:
            expected = original.step()
        self.assertEqual(expected['status'], 'accepted')
        before = snapshot(paused)
        with counters() as calls:
            token = pause.pause_wind(paused, runtime_reference=reference(), boundary_ordinal=ordinal)
            self.assertIs(type(token), pause._WindToken)
            self.assertEqual(snapshot(paused), before)
            lease = paused._active_evidence
            self.assertEqual(lease.ordinal, ordinal)
            self.assertEqual(lease.progress['next_step_index'], ordinal)
            self.assertGreater(lease.progress['n_steps'], ordinal)
            self.assertEqual(calls['finalize'], 0)
            self.assertEqual(calls['accept'], 0)
            self.assertEqual(calls['progress'], 1)
            observed = calls['progress_scalars'][0]
            self.assertEqual(lease.progress['native_time']['hex'], lease.prepared.ops.scalar(observed[0]).hex())
            self.assertEqual(lease.progress['next_step_index'], observed[1])
            self.assertEqual(lease.progress['n_steps'], observed[2])
            self.assertEqual(lease.progress['advanced_hex'], [lease.prepared.ops.scalar(v).hex() for v in observed[3:]])
            if second is not None:
                old = token
                token = pause.resume_wind(paused, token, boundary_ordinal=second)
                self.assertIs(type(token), pause._WindToken)
                self.assertIsNot(token, old)
                self.assertEqual(snapshot(paused), before)
                self.assertEqual(calls['progress'], 2)
                self.assertEqual(paused._active_evidence.ordinal, second)
            final = pause.resume_wind(paused, token)
        self.assertEqual(final, expected)
        self.assertEqual(snapshot(paused), snapshot(original))
        self.assertIsNone(paused._active_evidence)
        self.assertEqual(paused._next, 1)
        self.assertEqual(len(paused._ledger.records()), 1)
        self.assertEqual(calls['begin'], 1)
        self.assertEqual(calls['enter'], 1)
        self.assertEqual(calls['accept'], 1)
        self.assertEqual(calls['finalize'], 1)
        self.assertEqual(calls['progress'], 1 + (second is not None))
        self.assertEqual(calls['resume'], 1 + (second is not None))
        self.assertEqual(ordinary_calls['begin'], 1)
        self.assertEqual(ordinary_calls['finalize'], 1)
        return paused, final

    def test_first_genuine_wind_pause_exact_completion(self):
        _, result = self.run_pair()
        self.assertGreater(sum(result["classes"]["detachment_kg"]), 0.)
        self.assertGreater(sum(result["finalization"]["airborne_mass_before_by_class_kg"]), 0.)
        self.assertTrue(all(v == 0. for v in result["finalization"]["airborne_mass_after_by_class_kg"]))

    def test_two_original_boundaries_and_nonzero_epoch(self):
        self.run_pair(ordinal=1, second=2, origin=7.25)

    def test_short_native_remainder_keeps_actual_clocks(self):
        self.run_pair(duration=0.0625, outer=0.025, origin=11.75)

    def test_native_natural_commit_cadence_is_not_pause_commit(self):
        # Existing native commit law, explicitly selected synthetic cadence;
        # not a forced commit at a wrapper pause.
        _, result = self.run_pair(commit=TopographicCommitSpec(commit_interval_s=0.05))
        self.assertGreater(result['native']['n_commits_this_block'], 0)

    def test_completed_native_block_wins_without_getter(self):
        original, paused = new_runner(), new_runner()
        expected = original.step()
        with counters() as calls:
            actual = pause.pause_wind(paused, runtime_reference=reference(), boundary_ordinal=4096)
        self.assertEqual(actual, expected)
        self.assertEqual(snapshot(paused), snapshot(original))
        self.assertEqual(calls['progress'], 0)
        self.assertEqual(calls['finalize'], 1)
        self.assertEqual(calls['resume'], 0)

    def test_wind_rain_wind_rain_one_bed_and_retained_routing(self):
        events = f.schedule(f.wind(duration_s=1.), f.rain(), f.wind(duration_s=1.), f.rain())
        original, paused = new_runner(events=events), new_runner(events=events)
        expected = original.run()
        for _ in range(2):
            held = paused._routing
            source = None if held is None else held.source
            token = pause.pause_wind(paused, runtime_reference=reference(), boundary_ordinal=1)
            self.assertIs(type(token), pause._WindToken)
            self.assertEqual(pause.resume_wind(paused, token)['status'], 'accepted')
            self.assertIs(paused._routing, held)
            if held is not None:
                self.assertIs(paused._routing.source, source)
            self.assertEqual(paused.step()['status'], 'accepted')
        self.assertEqual(paused.report(), expected)
        self.assertEqual(snapshot(paused), snapshot(original))
        self.assertEqual(paused._n_accepted, 4)


class WindPauseNativeFailure(unittest.TestCase):
    def paused(self):
        runner = new_runner()
        token = pause.pause_wind(runner, runtime_reference=reference(), boundary_ordinal=1)
        self.assertIs(type(token), pause._WindToken)
        return runner, token

    def test_getter_is_never_called_on_resume_preflight(self):
        runner, token = self.paused()
        with mock.patch.object(block, 'wind_block_progress', side_effect=AssertionError('extra P1')) as getter:
            result = pause.resume_wind(runner, token)
        self.assertEqual(result['status'], 'accepted')
        getter.assert_not_called()

    def test_passive_bad_request_does_not_consume_real_lease(self):
        runner, token = self.paused()
        lease, before = runner._active_evidence, snapshot(runner)
        for bad in (True, False, 0, 1, -1, 1.0, '2'):
            with self.subTest(value=bad), self.assertRaises(dr.DenseRunnerError):
                pause.resume_wind(runner, token, boundary_ordinal=bad)
            self.assertIs(runner._active_evidence, lease)
            self.assertIs(lease.token, token)
            self.assertEqual(lease.phase, 'paused')
            self.assertEqual(snapshot(runner), before)
        self.assertEqual(pause.resume_wind(runner, token)['status'], 'accepted')

    def test_mutated_wrapper_scalar_progress_halts_without_native_resume(self):
        runner, token = self.paused()
        before, lease = snapshot(runner), runner._active_evidence
        lease.progress['next_step_index'] += 1
        with mock.patch.object(block, 'resume_wind_block', side_effect=AssertionError('native consumed')) as native:
            result = pause.resume_wind(runner, token)
        native.assert_not_called()
        self.assertEqual(result['status'], 'runner_error')
        self.assertTrue(runner.active_evidence['unknown_current_endpoint'])
        self.assertEqual(runner.active_evidence['last_validated']['next_step_index'], 1)
        self.assertEqual(snapshot(runner)['state'], before['state'])
        self.assertIsNone(lease.prepared)
        self.assertIsNone(lease.native_token)

    def test_owned_native_valueerror_preserves_category_and_stops(self):
        runner, token = self.paused()
        before = snapshot(runner)
        with mock.patch.object(block, 'resume_wind_block', side_effect=ValueError('source-labelled native refusal')):
            result = pause.resume_wind(runner, token)
        self.assertEqual(result['status'], 'native_refused')
        self.assertEqual(result['code'], 'wind_native_failure')
        self.assertTrue(runner.active_evidence['unknown_current_endpoint'])
        self.assertEqual(snapshot(runner)['state'], before['state'])
        with self.assertRaises(dr.DenseRunnerError): runner.step()

    def test_interrupt_after_owned_call_halts_and_reraises(self):
        runner, token = self.paused()
        lease, before = runner._active_evidence, snapshot(runner)
        with mock.patch.object(block, 'resume_wind_block', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): pause.resume_wind(runner, token)
        self.assertEqual(lease.phase, 'halted')
        self.assertIsNone(lease.prepared)
        self.assertIsNone(lease.native_token)
        self.assertEqual(snapshot(runner)['state'], before['state'])
        with self.assertRaises(dr.DenseRunnerError): pause.resume_wind(runner, token)

    def test_finalizer_delegate_failure_never_publishes(self):
        runner, token = self.paused()
        before = snapshot(runner)
        with mock.patch.object(dr, '_finalize_wind_event', side_effect=RuntimeError('source-labelled finalizer fault')):
            result = pause.resume_wind(runner, token)
        self.assertEqual(result['code'], 'wind_finalization_failure')
        self.assertEqual(runner._next, 0)
        self.assertEqual(snapshot(runner)['state'], before['state'])
        self.assertTrue(runner.active_evidence['unknown_current_endpoint'])

    def test_lost_ack_after_real_acceptance_is_not_stop_or_replay(self):
        runner, token = self.paused()
        actual = dr._wind_accept
        def lost(*args, **kwargs):
            actual(*args, **kwargs)
            raise RuntimeError('source-labelled lost acknowledgement')
        with mock.patch.object(dr, '_wind_accept', side_effect=lost):
            with self.assertRaisesRegex(RuntimeError, 'lost acknowledgement'):
                pause.resume_wind(runner, token)
        self.assertEqual(runner._next, 1)
        self.assertIsNone(runner._active_evidence)
        self.assertIsNone(runner._stop_receipt)
        self.assertEqual(len(runner._receipts), 1)
        with self.assertRaises(dr.DenseRunnerError): pause.resume_wind(runner, token)

    def test_failure_after_real_outer_step_keeps_last_accepted_boundary(self):
        runner = new_runner()
        before, calls = snapshot(runner), []
        actual = baseline.advance_baseline_outer_step
        def after_real(*args, **kwargs):
            actual(*args, **kwargs)
            calls.append(1)
            raise RuntimeError('source-labelled failure after real physical step')
        with mock.patch.object(baseline, 'advance_baseline_outer_step', side_effect=after_real):
            result = pause.pause_wind(runner, runtime_reference=reference(), boundary_ordinal=1)
        self.assertEqual(calls, [1])
        self.assertEqual(result['status'], 'native_error')
        self.assertEqual(snapshot(runner)['state'], before['state'])
        self.assertEqual(runner._next, 0)
        self.assertTrue(runner.active_evidence['unknown_current_endpoint'])

    def test_callback_failure_halts_real_native_attempt(self):
        runner = new_runner()
        before, calls = snapshot(runner), []
        def callback():
            calls.append(1)
            raise RuntimeError('source-labelled callback failure')
        with mock.patch.object(pause, '_callback', return_value=callback):
            result = pause.pause_wind(runner, runtime_reference=reference(), boundary_ordinal=1)
        self.assertEqual(calls, [1])
        self.assertEqual(result['status'], 'native_error')
        self.assertEqual(result['exception_type'], 'RuntimeError')
        self.assertIn('source-labelled callback failure', result['detail'])
        self.assertEqual(snapshot(runner)['state'], before['state'])
        self.assertTrue(runner.active_evidence['unknown_current_endpoint'])

    def test_wrapper_token_allocation_failure_drops_real_paused_owner(self):
        runner = new_runner()
        before = snapshot(runner)
        with mock.patch.object(pause, '_WindToken', side_effect=MemoryError('source-labelled token allocation')):
            result = pause.pause_wind(runner, runtime_reference=reference(), boundary_ordinal=1)
        self.assertEqual(result['status'], 'runner_error')
        self.assertEqual(result['exception_type'], 'MemoryError')
        self.assertEqual(snapshot(runner)['state'], before['state'])
        self.assertTrue(runner.active_evidence['unknown_current_endpoint'])

    def test_partial_adoption_interrupt_is_halted_without_replay(self):
        runner, token = self.paused()
        lease = runner._active_evidence
        real = dr._HostOwnership.commit
        def interrupted(owner, prepared):
            real(owner, prepared)
            raise KeyboardInterrupt('source-labelled interruption after real bed adoption')
        with mock.patch.object(dr._HostOwnership, 'commit', new=interrupted):
            with self.assertRaises(KeyboardInterrupt): pause.resume_wind(runner, token)
        self.assertEqual(lease.phase, 'halted')
        self.assertEqual(runner._next, 0)
        self.assertIn('may be partial', runner.stop_receipt['guarantee'])
        self.assertIsNone(lease.prepared)
        with self.assertRaises(dr.DenseRunnerError): runner.step()

    def test_explicit_abandon_is_incomplete_and_drops_only_private_work(self):
        runner, token = self.paused()
        before, lease = snapshot(runner), runner._active_evidence
        result = pause.abandon_wind(runner, token)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['code'], 'wind_pause_abandoned')
        self.assertEqual(runner._next, 0)
        self.assertEqual(snapshot(runner)['state'], before['state'])
        self.assertIsNone(lease.prepared)
        self.assertIsNone(lease.native_token)
        with self.assertRaises(dr.DenseRunnerError): runner.step()


if __name__ == '__main__':
    unittest.main()
