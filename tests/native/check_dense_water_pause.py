"""Opt-in SOURCE-gated CPU checks for the private in-memory native water lease.

This file is not an execution grant. The numerical groups require a Root-owned
runtime-reference metadata file and its external raw SHA via MASY212_RUNTIME /
MASY212_RUNTIME_SHA256. The product's ACTUAL _runtime_binding is never replaced
on a positive path. A separately declared development component can exercise
this changed-source candidate; the historical7f0 lock is not positive authority.
No saved model arrays are read or written by these checks.
"""
from __future__ import annotations

import contextlib
import dataclasses
import gc
import hashlib
import json
import os
import stat
import threading
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import fixture_dense_alternation as f
from masy import dense_bed as db
from masy import dense_boundary_store as store
from masy import dense_runner as dr
from masy import dense_runner_checkpoint as boundary
from masy import dense_water_pause as pause


def reference():
    path = Path(os.environ['MASY212_RUNTIME'])
    expected = os.environ['MASY212_RUNTIME_SHA256']
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise AssertionError('external Root runtime-reference SHA missing')
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_size > 32768 or st.st_mode & 0o222:
        raise AssertionError('bounded readonly runtime-reference metadata required')
    raw = path.read_bytes()
    if len(raw) != st.st_size or hashlib.sha256(raw).hexdigest() != expected:
        raise AssertionError('runtime-reference changed')
    return json.loads(raw)


def new_runner(*, events=None, origin=7.25, max_steps=None, work_cap=None):
    actual = f.build_dense_bed
    def at_origin(**kw):
        return actual(**dict(kw, initial_time_s=float(origin)))
    with mock.patch.object(f, 'build_dense_bed', side_effect=at_origin):
        bed = f.build_bed()
    sc = dataclasses.replace(f.sediment_control(), force_final_commit=False)
    if max_steps is not None:
        sc = dataclasses.replace(sc, storm=dataclasses.replace(sc.storm, max_steps=max_steps))
    return f.make_runner(bed, events or f.schedule(f.rain()),
                         control=f.runner_control(sediment_control=sc),
                         limit=f.limits() if work_cap is None else f.limits(max_work_bytes=work_cap))


def native_evidence(item):
    """Equal-run physical oracle excludes ONLY the five native wall/CPU timers.

    The native return itself is untouched. All other71result fields, including
    full accumulated report rows, native counters and exact clocks, are hashed.
    This digest is test-local metadata, not a checkpoint or copied physics.
    """
    r = item.progress.result if type(item) is dr._native().DryHandoff else item.result
    timers = ('first_step_wall_s', 'first_step_cpu_s', 'remaining_wall_s', 'remaining_cpu_s', 'commit_wall_s')
    physical = {field.name:getattr(r, field.name) for field in dataclasses.fields(r) if field.name not in timers}
    terminal = (int(any(v != 0 for v in item.accounting['flow_end_settlement']['mobile_before_settlement_by_class_kg']))
                if type(item) is dr._native().DryHandoff else 0)
    return {'sha256':db._sha256(dr._canonical_input(np, physical, 'native_result')),
            'terminal_call':terminal,
            'end':float(r.state.t_s).hex(), 'steps':r.n_accepted_steps,
            'calls':r.n_maple_water_calls, 'commits':r.n_commits,
            'forced_commits':r.n_forced_commits, 'reroutes':r.n_reroutes,
            'rows':len(r.boundaries)}


def snapshot(runner):
    held = runner._routing
    routing = None if held is None else boundary._cache_fingerprint({
        name:getattr(held, name) for name in ('graph','network','grid','terrain','source','template')})
    return {'bed':runner._bed.identity_sha256(),
            'physical':db._sha256(dr._canonical_input(np,runner._bed.state,'physical')),
            'pool':db._sha256(dr._canonical_input(np,runner._pool,'pool')),
            'soil':db._sha256(dr._canonical_input(np,runner._soil,'soil')),
            'totals':db._sha256(dr._canonical_input(np,runner._totals,'totals')),
            'routing':routing, 'report':runner.report()}


def nominal_runner():
    """Inert nominal receiver ONLY for early-refusal protocols; no fake native success."""
    r = object.__new__(dr.DenseRunner)
    r._stop_receipt, r._active_evidence, r._next = None, None, 0
    return r


def nominal_lease(runner):
    item = pause._WaterLease(weakref.ref(runner), weakref.ref(threading.current_thread()), os.getpid(),
                             runner._next, None, 0.0, phase='paused')
    item.token = pause._WaterToken()
    runner._active_evidence = item
    return item


class LeaseProtocols(unittest.TestCase):
    """No model success is synthesized; only private metadata/refusal behavior."""

    def test_callback_keeps_only_scalar_target(self):
        class Progress:
            pass
        item = Progress()
        item.result = SimpleNamespace(state=SimpleNamespace(t_s=3.0))
        ref = weakref.ref(item)
        callback = pause._callback(3.0)
        self.assertIs(callback(item), True)
        item.result.state.t_s = 4.0
        self.assertIs(callback(item), False)
        self.assertEqual(tuple(c.cell_contents for c in callback.__closure__), (3.0,))
        del item
        gc.collect()
        self.assertIsNone(ref())

    def test_opaque_token_carries_no_native_payload(self):
        token = pause._WaterToken()
        with self.assertRaises(TypeError):vars(token)
        with self.assertRaises(AttributeError):token.progress = object()

    def test_ordinal_closed_types_and_original_guard(self):
        for value in (None, False, True, 0, -1, 3, 4, 1.0, '1'):
            with self.subTest(value=value), self.assertRaises(dr.DenseRunnerError):
                pause._ordinal(value, (10.,20.,30.))
        pause._ordinal(1,(10.,20.,30.))
        pause._ordinal(2,(10.,20.,30.),1)
        with self.assertRaises(dr.DenseRunnerError):pause._ordinal(1,(10.,20.,30.),1)

    def test_step_run_and_public_evidence_refuse_before_native_or_stop(self):
        runner = nominal_runner()
        lease = nominal_lease(runner)
        for action in (runner.step, runner.run, lambda:runner.active_evidence):
            with self.subTest(action=str(action)), mock.patch.object(dr,'_native',side_effect=AssertionError('native')) as native, \
                    mock.patch.object(dr.DenseRunner,'_stop',side_effect=AssertionError('stop')) as stop:
                with self.assertRaises(dr.DenseRunnerError):action()
                self.assertIs(runner._active_evidence,lease)
                self.assertEqual(lease.phase,'paused')
                native.assert_not_called();stop.assert_not_called()

    def test_foreign_stale_and_busy_tokens_do_not_consume(self):
        runner = nominal_runner();lease = nominal_lease(runner);token = lease.token
        foreign = nominal_runner();foreign._active_evidence = lease
        for receiver, supplied in ((foreign,token),(runner,pause._WaterToken()),(runner,None)):
            with self.subTest(receiver=receiver is runner), self.assertRaises(dr.DenseRunnerError):
                pause._lease(receiver,supplied)
            self.assertIs(lease.token,token)
        lease.phase = 'busy'
        with self.assertRaises(dr.DenseRunnerError):pause._lease(runner,token)
        lease.phase = 'paused'
        self.assertIs(pause._lease(runner,token),lease)

    def test_process_thread_and_cursor_bindings_refuse(self):
        runner = nominal_runner();lease = nominal_lease(runner);token = lease.token
        with mock.patch.object(pause.os,'getpid',return_value=lease.pid+1), self.assertRaises(dr.DenseRunnerError):
            pause._lease(runner,token)
        errors = []
        def other():
            try:pause._lease(runner,token)
            except dr.DenseRunnerError as e:errors.append(e.code)
        thread = threading.Thread(target=other);thread.start();thread.join(timeout=2)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,['water_token'])
        runner._next += 1
        with self.assertRaises(dr.DenseRunnerError):pause._lease(runner,token)
        runner._next -= 1
        self.assertIs(pause._lease(runner,token),lease)

    def test_weak_owner_and_no_global_history(self):
        runner = nominal_runner();lease = nominal_lease(runner);owner = weakref.ref(runner)
        del runner;gc.collect()
        self.assertIsNone(owner());self.assertIsNone(lease.owner())
        self.assertFalse(hasattr(lease.token,'__dict__'))

    def test_stopped_runner_refusal_precedes_active_slot(self):
        runner = nominal_runner();runner._stop_receipt = {};runner._active_evidence = object()
        with self.assertRaises(dr.DenseRunnerError) as error:
            pause.pause_water(runner,runtime_reference={},boundary_ordinal=1)
        self.assertEqual(error.exception.code,'runner_halted')

    def test_foreign_runner_refuses_before_declaration(self):
        with mock.patch.object(pause,'_declare',side_effect=AssertionError('declare')) as declare:
            with self.assertRaises(dr.DenseRunnerError):pause.pause_water(object(),runtime_reference={},boundary_ordinal=1)
            declare.assert_not_called()

    def test_halt_drops_private_references_even_if_receipt_allocation_fails(self):
        runner = nominal_runner();lease = nominal_lease(runner)
        lease.prepared, lease.progress = object(), object()
        with mock.patch.object(dr.DenseRunner,'_stop',side_effect=MemoryError):
            with self.assertRaises(MemoryError):pause._halt(runner,lease,dr._Stop('runner_error','test','authored failure'))
        self.assertEqual(lease.phase,'halted');self.assertIsNone(lease.token)
        self.assertIsNone(lease.prepared);self.assertIsNone(lease.progress)
        with self.assertRaises(dr.DenseRunnerError):runner.step()

    def test_partial_adoption_exception_is_never_called_unchanged(self):
        runner = nominal_runner();lease = nominal_lease(runner)
        def stopping(self,index,kind,start,stop):return {'guarantee':stop.guarantee,'code':stop.code}
        def failed(*args,**kw):lease.phase='finishing';raise MemoryError('authored adoption failure')
        with mock.patch.object(dr,'_rain_native',return_value=None), mock.patch.object(pause,'_outcome',side_effect=failed), \
                mock.patch.object(dr.DenseRunner,'_stop',autospec=True,side_effect=stopping):
            result = pause._execute(runner,lease,None,None,None,None,{},None)
        self.assertIn('partial',result['guarantee']);self.assertNotEqual(result['guarantee'],dr._UNCHANGED)
        self.assertEqual(lease.phase,'halted')

    def test_partial_adoption_baseexception_blocks_and_reraises(self):
        runner = nominal_runner();lease = nominal_lease(runner)
        def failed(*args,**kw):lease.phase='finishing';raise KeyboardInterrupt()
        stops=[]
        def stopping(self,index,kind,start,stop):stops.append(stop);return {}
        with mock.patch.object(dr,'_rain_native',return_value=None), mock.patch.object(pause,'_outcome',side_effect=failed), \
                mock.patch.object(dr.DenseRunner,'_stop',autospec=True,side_effect=stopping):
            with self.assertRaises(KeyboardInterrupt):pause._execute(runner,lease,None,None,None,None,{},None)
        self.assertIn('partial',stops[0].guarantee);self.assertEqual(lease.phase,'halted')

    def test_actual_closed_native_layout_and_bit_digest(self):
        n,checkpoint,schemas,unused = pause._declare()
        bounds={'array_cap':4096,'raw_cap':4096,'sequence_cap':128}
        a=np.array([0.0,-0.0],dtype=np.float64)
        before=pause._scan(a,n,schemas,bounds)['sha256']
        a[1]=0.0
        self.assertNotEqual(before,pause._scan(a,n,schemas,bounds)['sha256'])
        self.assertNotEqual(pause._scan(np.float32(1),n,schemas,bounds)['sha256'],
                            pause._scan(np.float64(1),n,schemas,bounds)['sha256'])
        self.assertEqual(len(schemas),21);self.assertEqual(len(pause._FIELDS),21)
        pause._scan(np.empty((0,2),dtype=np.float64),n,schemas,bounds)
        from maple_syrup.storm import plan_boundaries
        edges=[0.,1.,1.+2.**-50,2.]
        schedule=n.RainfallSchedule(edges_s=edges,intensity_mm_per_h=[1.,2.,3.],
                                    provenance=n.RainfallProvenance(kind='constant'))
        plan=plan_boundaries(schedule,0.,3.,.5,max_report_rows=32)
        # Native merger preserves BOTH exact knots even within its coincidence
        # tolerance. Thus original intervals cannot outnumber the plan rows.
        self.assertTrue(all(edge in plan for edge in edges[1:]))
        self.assertGreaterEqual(len(plan),len(edges)-1)

    def test_evidence_walk_refuses_object_strides_caps_unknown_and_cycles(self):
        n,checkpoint,schemas,unused=pause._declare()
        bounds={'array_cap':16,'raw_cap':16,'sequence_cap':2}
        invalid=(np.array([object()],dtype=object),np.arange(8.)[::2],np.arange(3.),
                 np.array([1.],dtype='>f8'),np.array([1.],dtype=np.float16),
                 object(),[1,2,3],{'bad':object()},{'\ud800':0})
        for value in invalid:
            with self.subTest(kind=type(value).__name__),self.assertRaises(dr.DenseRunnerError):pause._scan(value,n,schemas,bounds)
        cycle=[];cycle.append(cycle)
        with self.assertRaises(dr.DenseRunnerError):pause._scan(cycle,n,schemas,bounds)

    def test_array_raw_and_leaf_caps_precede_payload_hash(self):
        n,checkpoint,schemas,unused=pause._declare()
        actual=pause.hashlib.sha256
        reads=[]
        class Hash:
            def __init__(self):self.inner=actual()
            def update(self,value):
                if isinstance(value,memoryview):reads.append(len(value))
                self.inner.update(value)
            def hexdigest(self):return self.inner.hexdigest()
        a=np.zeros(1,dtype=np.float64);b=np.ones(1,dtype=np.float64)
        with mock.patch.object(pause.hashlib,'sha256',side_effect=Hash):
            with self.assertRaises(dr.DenseRunnerError):
                pause._scan([a,b],n,schemas,{'array_cap':8,'raw_cap':8,'sequence_cap':2048})
        self.assertEqual(reads,[8])
        reads.clear()
        with mock.patch.object(pause.hashlib,'sha256',side_effect=Hash):
            pause._scan([a,b],n,schemas,{'array_cap':8,'raw_cap':16,'sequence_cap':2048})
        self.assertEqual(reads,[8,8]);reads.clear()
        leaves=[np.zeros(1,dtype=np.bool_) for _ in range(pause._MAX_ARRAYS+1)]
        with mock.patch.object(pause.hashlib,'sha256',side_effect=Hash):
            pause._scan(leaves[:-1],n,schemas,{'array_cap':8,'raw_cap':4096,'sequence_cap':2048})
        self.assertEqual(reads,[1]*pause._MAX_ARRAYS);reads.clear()
        with mock.patch.object(pause.hashlib,'sha256',side_effect=Hash):
            with self.assertRaises(dr.DenseRunnerError):
                pause._scan(leaves,n,schemas,{'array_cap':8,'raw_cap':4096,'sequence_cap':2048})
        self.assertEqual(len(reads),pause._MAX_ARRAYS)

    def test_alias_array_data_hashed_once_with_identity_and_flag_markers(self):
        n,checkpoint,schemas,unused=pause._declare()
        limits={'array_cap':8,'raw_cap':8,'sequence_cap':128}
        a=np.zeros(1,dtype=np.float64)
        actual=pause.hashlib.sha256;reads=[]
        class Hash:
            def __init__(self):self.inner=actual()
            def update(self,value):
                if isinstance(value,memoryview):reads.append(len(value))
                self.inner.update(value)
            def hexdigest(self):return self.inner.hexdigest()
        with mock.patch.object(pause.hashlib,'sha256',side_effect=Hash):
            result=pause._scan([a,a],n,schemas,limits)
        self.assertEqual(reads,[8])
        self.assertEqual(result['raw_bytes'],8)
        self.assertEqual(result['array_data_hashed_bytes'],8)
        self.assertEqual(result['array_alias_visits'],1)
        self.assertEqual(len(result['arrays']),1)
        self.assertEqual(pause._scan([a,a],n,schemas,limits)['sha256'],result['sha256'])
        a.flags.writeable=False
        changed=pause._scan([a,a],n,schemas,limits)
        self.assertNotEqual(changed['sha256'],result['sha256'])

    def test_string_and_integer_caps_precede_repr_allocation(self):
        n,checkpoint,schemas,unused=pause._declare()
        limits={'array_cap':8,'raw_cap':8,'sequence_cap':128}
        for value in ('x'*(pause._MAX_STRING+1),1<<(pause._MAX_INT_BITS+1)):
            with self.subTest(kind=type(value).__name__),mock.patch.object(pause,'repr',create=True,side_effect=AssertionError('repr')) as represent:
                with self.assertRaises(dr.DenseRunnerError):pause._scan(value,n,schemas,limits)
                represent.assert_not_called()
        for value in ('x'*pause._MAX_STRING,(1<<pause._MAX_INT_BITS)-1):
            with self.subTest(admitted=type(value).__name__),mock.patch.object(pause,'repr',create=True,side_effect=AssertionError('repr')):
                with self.assertRaisesRegex(AssertionError,'repr'):pause._scan(value,n,schemas,limits)

    def test_interrupt_survives_failed_stop_evidence_allocation(self):
        runner=nominal_runner();lease=nominal_lease(runner)
        lease.prepared,lease.progress=object(),object()
        with mock.patch.object(dr,'_rain_native',side_effect=KeyboardInterrupt), \
                mock.patch.object(dr.DenseRunner,'_stop',side_effect=MemoryError):
            with self.assertRaises(KeyboardInterrupt):pause._execute(runner,lease,None,None,None,None,{},None)
        self.assertEqual(lease.phase,'halted');self.assertIsNone(lease.token)
        self.assertIsNone(lease.prepared);self.assertIsNone(lease.progress)

    def test_actual_native_defaulted_field_omission_is_refused(self):
        n,checkpoint,schemas,unused=pause._declare()
        from maple_syrup.complete_event import QuietProgress
        item=QuietProgress()
        object.__delattr__(item,'ready')
        with self.assertRaises(dr.DenseRunnerError):pause._scan(item,n,schemas,{'array_cap':1024,'raw_cap':1024,'sequence_cap':128})


class NativeParity(unittest.TestCase):
    """Unmodified native solves and actual owning runtime binding on every API path."""
    @classmethod
    def setUpClass(cls):
        pause._declare()
        cls.reference=reference()
        runner=new_runner()
        results=[]
        actual=dr._complete_event
        def record(*a,**kw):
            item=actual(*a,**kw);results.append(native_evidence(item));return item
        with mock.patch.object(dr,'_complete_event',side_effect=record):
            receipt=runner.step()
        if receipt['status']!='accepted':raise AssertionError('genuine uninterrupted reference did not accept')
        cls.reference_snapshot=snapshot(runner)
        cls.reference_native=results[-1]
        cls.actual_dry_time=float.fromhex(results[-1]['end'])

    def complete_with_pauses(self,count):
        runner=new_runner();accepted=snapshot(runner)
        bed,pool,soil=runner._bed.state,runner._pool,runner._soil
        actual=dr._complete_event;results=[]
        def record(*a,**kw):
            item=actual(*a,**kw);results.append(native_evidence(item));return item
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(dr,'_complete_event',side_effect=record))
            bindings=stack.enter_context(mock.patch.object(store,'_runtime_binding',wraps=store._runtime_binding))
            entry=stack.enter_context(mock.patch.object(dr,'_rain_enter',wraps=dr._rain_enter))
            stage=stack.enter_context(mock.patch.object(dr.DenseRunner,'_stage',autospec=True,side_effect=dr.DenseRunner._stage))
            accepted_call=stack.enter_context(mock.patch.object(dr,'_rain_accept',wraps=dr._rain_accept))
            token=pause.pause_water(runner,runtime_reference=self.reference,boundary_ordinal=1)
            self.assertIs(type(token),pause._WaterToken);self.assertEqual(snapshot(runner),accepted)
            lease=runner._active_evidence
            self.assertIs(runner._bed.state,bed);self.assertIs(runner._pool,pool);self.assertIs(runner._soil,soil)
            if count==2:
                old=token;token=pause.resume_water(runner,token,boundary_ordinal=2)
                self.assertIs(type(token),pause._WaterToken);self.assertIsNot(token,old)
                with self.assertRaises(dr.DenseRunnerError):pause.resume_water(runner,old)
                self.assertEqual(snapshot(runner),accepted)
                with self.assertRaises(dr.DenseRunnerError) as limited:
                    pause.resume_water(runner,token,boundary_ordinal=3)
                self.assertEqual(limited.exception.code,'water_pause_limit')
                self.assertIs(lease.token,token)
            result=pause.resume_water(runner,token)
            self.assertEqual(result['status'],'accepted')
            self.assertEqual(entry.call_count,1);self.assertEqual(stage.call_count,1);self.assertEqual(accepted_call.call_count,1)
            self.assertEqual(bindings.call_count,2+2*count)
        self.assertIsNone(runner._active_evidence)
        self.assertIsNone(lease.prepared);self.assertIsNone(lease.progress);self.assertIsNone(lease.token)
        with self.assertRaises(dr.DenseRunnerError):pause.resume_water(runner,token)
        self.assertEqual(snapshot(runner),self.reference_snapshot)
        self.assertEqual(results[-1],self.reference_native)
        self.assertEqual(results[-1]['calls'],2*results[-1]['steps']+results[-1]['terminal_call'])
        self.assertGreater(float.fromhex(results[-1]['end']),7.25)

    def test_one_pause_nonzero_origin_exact_native_and_wrapper_parity(self):
        self.complete_with_pauses(1)

    def test_two_pauses_preserve_global_counters_and_single_acceptance(self):
        self.complete_with_pauses(2)

    def test_genuine_native_step_guard_after_resume_is_incomplete(self):
        # Derive from THIS same-case uninterrupted solve, never an old endpoint.
        cap=self.reference_native['steps']-1
        minimum=int(np.ceil((f.RAIN_S+f.policy().hold_s)/f.sediment_control().storm.max_dt_s))
        self.assertGreaterEqual(cap,minimum)
        runner=new_runner(max_steps=cap);before=snapshot(runner)
        token=pause.pause_water(runner,runtime_reference=self.reference,boundary_ordinal=1)
        self.assertIs(type(token),pause._WaterToken)
        result=pause.resume_water(runner,token)
        self.assertEqual((result['status'],result['code']),('incomplete','rain_incomplete_native_limit'))
        self.assertEqual(snapshot(runner)['physical'],before['physical'])
        self.assertEqual(runner._next,0)
        with self.assertRaises(dr.DenseRunnerError):pause.resume_water(runner,token)

    def test_native_ready_dry_precedes_requested_later_pause(self):
        runner=new_runner()
        n,unused,unused2,unused3=pause._declare()
        resolved,plan=pause._plan(runner,n)
        target=next(i+1 for i,t in enumerate(plan[:-1]) if t>self.actual_dry_time)
        result=pause.pause_water(runner,runtime_reference=self.reference,boundary_ordinal=target)
        self.assertIs(type(result),dict);self.assertEqual(result['status'],'accepted')
        self.assertIsNone(runner._active_evidence)
        self.assertEqual(snapshot(runner),self.reference_snapshot)


class NativeOwnership(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pause._declare();cls.reference=reference()

    def wet(self):
        runner=new_runner();before=snapshot(runner)
        token=pause.pause_water(runner,runtime_reference=self.reference,boundary_ordinal=1)
        self.assertIs(type(token),pause._WaterToken)
        return runner,token,before

    def test_paused_native_accepted_state_and_passive_guard(self):
        runner,token,before=self.wet();lease=runner._active_evidence
        self.assertFalse(lease.prepared.stage.consumed)
        for _,wet_array in db._require_host_arrays((lease.progress.result.state.bed,
                lease.progress.result.state.storm.soil_water_m,lease.progress.result.state.sediment_velocity_m_s),'wet'):
            for _,accepted_array in db._require_host_arrays((runner._bed.state,runner._pool,runner._soil),'accepted'):
                self.assertFalse(np.shares_memory(wet_array,accepted_array))
        with mock.patch.object(dr,'_complete_event',side_effect=AssertionError('native')) as native:
            for action in (runner.step,runner.run,lambda:runner.active_evidence):
                with self.assertRaises(dr.DenseRunnerError):action()
            with self.assertRaises(boundary.BoundaryCheckpointError):
                boundary._runner_admission(runner,boundary.BoundaryImageLimits())
            native.assert_not_called()
        self.assertIs(runner._active_evidence,lease);self.assertIs(lease.token,token)
        self.assertEqual(snapshot(runner),before)
        result=pause.abandon_water(runner,token)
        self.assertEqual(result['status'],'incomplete');self.assertEqual(result['code'],'water_pause_abandoned')
        self.assertIsNone(lease.prepared);self.assertIsNone(lease.progress)
        self.assertEqual(snapshot(runner)['physical'],before['physical'])

    def test_invalid_next_ordinal_and_runtime_drift_do_not_consume(self):
        runner,token,before=self.wet();lease=runner._active_evidence
        for value in (True,0,1,len(lease.plan)):
            with self.subTest(value=value),self.assertRaises(dr.DenseRunnerError):pause.resume_water(runner,token,boundary_ordinal=value)
            self.assertIs(lease.token,token);self.assertEqual(lease.phase,'paused')
        with mock.patch.object(store,'_runtime_binding',side_effect=store.BoundaryStoreError('authored_refusal','no source success')):
            with self.assertRaises(store.BoundaryStoreError):pause.resume_water(runner,token)
        self.assertIs(lease.token,token);self.assertEqual(snapshot(runner),before)
        pause.abandon_water(runner,token)

    def test_mutated_private_progress_refuses_without_native_consumption(self):
        from maple_syrup.sediment_event import SedimentEventError
        runner,token,before=self.wet();lease=runner._active_evidence
        leaf=lease.progress.result.cumulative_rain_m
        old=float(leaf.flat[0]);leaf.flat[0]=old+1.0
        with mock.patch.object(dr,'_complete_event',side_effect=AssertionError('native')) as native:
            with self.assertRaises(SedimentEventError):pause.resume_water(runner,token)
            native.assert_not_called()
        self.assertIs(lease.token,token);self.assertEqual(lease.phase,'paused')
        self.assertEqual(snapshot(runner),before)
        known=dr._elapsed(lease.origin,lease.plan[lease.ordinal-1])
        storm=lease.progress.result.state.storm
        object.__setattr__(storm,'t_s',float(storm.t_s)+999.)
        result=pause.abandon_water(runner,token)
        self.assertEqual(result['unaccepted_elapsed_s'],known)

    def test_original_forcing_receipt_and_routing_holder_cannot_be_rebound(self):
        runner,token,before=self.wet();lease=runner._active_evidence
        rate=lease.prepared.schedule.rate_m_per_s;was=rate.flags.writeable;old=float(rate[0])
        rate.flags.writeable=True
        try:
            rate[0]=old*2
            with mock.patch.object(dr,'_complete_event',side_effect=AssertionError('native')) as native:
                with self.assertRaises(dr.DenseRunnerError):pause.resume_water(runner,token)
                native.assert_not_called()
        finally:rate[0]=old;rate.flags.writeable=was
        old=lease.resolved['kind'];lease.resolved['kind']='wind'
        try:
            with self.assertRaises(dr.DenseRunnerError):pause.resume_water(runner,token)
        finally:lease.resolved['kind']=old
        held=lease.prepared.held;lease.prepared.held=None
        try:
            with self.assertRaises(dr.DenseRunnerError) as error:pause.resume_water(runner,token)
            self.assertEqual(error.exception.code,'water_routing_changed')
        finally:lease.prepared.held=held
        self.assertIsNone(runner._routing)
        runner._routing=lease.prepared.held
        try:
            with self.assertRaises(dr.DenseRunnerError) as error:pause.resume_water(runner,token)
            self.assertEqual(error.exception.code,'water_routing_changed')
        finally:runner._routing=None
        self.assertIs(lease.token,token);self.assertEqual(snapshot(runner),before)
        pause.abandon_water(runner,token)

    def test_external_accepted_mutation_halts_with_truthful_external_receipt(self):
        runner,token,before=self.wet()
        leaf=runner._soil;old=float(leaf.flat[0]);was=leaf.flags.writeable
        leaf.flags.writeable=True
        try:
            leaf.flat[0]=old+0.001
            result=pause.resume_water(runner,token)
        finally:
            leaf.flat[0]=old;leaf.flags.writeable=was
        self.assertEqual(result['status'],'refused')
        self.assertEqual(result['code'],'accepted_state_externally_mutated')
        self.assertIn('OUTSIDE',result['guarantee'])
        with self.assertRaises(dr.DenseRunnerError):pause.resume_water(runner,token)

    def test_initial_external_mutation_is_receipted_without_private_stop_escape(self):
        runner=new_runner();leaf=runner._soil;old=float(leaf.flat[0]);was=leaf.flags.writeable
        leaf.flags.writeable=True
        try:
            leaf.flat[0]=old+0.001
            with mock.patch.object(dr.DenseRunner,'_stage',side_effect=AssertionError('stage')) as stage:
                result=pause.pause_water(runner,runtime_reference=self.reference,boundary_ordinal=1)
                stage.assert_not_called()
        finally:
            leaf.flat[0]=old;leaf.flags.writeable=was
        self.assertEqual(result['code'],'accepted_state_externally_mutated')
        self.assertIn('OUTSIDE',result['guarantee'])
        self.assertIsNotNone(runner.stop_receipt)

    def test_entry_plan_drift_halts_before_native_complete(self):
        runner=new_runner();before=snapshot(runner);actual=dr._rain_enter
        def changed(*args,**kw):
            prepared=actual(*args,**kw);prepared.every*=2;return prepared
        with mock.patch.object(dr,'_rain_enter',side_effect=changed),mock.patch.object(dr,'_complete_event',side_effect=AssertionError('native')) as native:
            result=pause.pause_water(runner,runtime_reference=self.reference,boundary_ordinal=1)
            native.assert_not_called()
        self.assertEqual(result['code'],'water_entry_failure')
        self.assertEqual(snapshot(runner)['physical'],before['physical'])

    def test_consumed_native_error_halts_without_dry_reset(self):
        runner,token,before=self.wet();lease=runner._active_evidence
        with mock.patch.object(dr,'_complete_event',side_effect=ValueError('authored native refusal')):
            result=pause.resume_water(runner,token)
        self.assertEqual((result['status'],result['code']),('native_refused','rain_native_failure'))
        self.assertIsNone(lease.token);self.assertIsNone(lease.progress);self.assertIsNone(lease.prepared)
        self.assertEqual(snapshot(runner)['physical'],before['physical'])
        self.assertEqual(runner._next,0);self.assertEqual(runner.ledger_record(),before['report']['clock_ledger'])

    def test_interrupt_after_consume_halts_and_releases_own_progress(self):
        runner,token,before=self.wet();lease=runner._active_evidence
        progress_ref=weakref.ref(lease.progress)
        with mock.patch.object(dr,'_complete_event',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):pause.resume_water(runner,token)
        gc.collect()
        self.assertIsNone(lease.progress);self.assertIsNone(lease.prepared)
        self.assertIsNone(progress_ref());self.assertIsNone(lease.token)
        self.assertIsNotNone(runner.stop_receipt)
        self.assertEqual(snapshot(runner)['physical'],before['physical'])

    def test_low_resource_and_unsupported_profiles_refuse_before_stage(self):
        runner=new_runner(work_cap=1);before=snapshot(runner)
        with mock.patch.object(dr.DenseRunner,'_stage',side_effect=AssertionError('stage')) as stage:
            result=pause.pause_water(runner,runtime_reference=self.reference,boundary_ordinal=1)
            stage.assert_not_called()
        self.assertEqual(result['code'],'work_bytes_cap');self.assertEqual(snapshot(runner)['physical'],before['physical'])
        runner=new_runner()
        bad_storm=dataclasses.replace(runner._control.sediment_control.storm,max_steps='authored-invalid')
        from maple_syrup.sediment_event import SedimentEventError
        from maple_syrup.storm import StormError
        with self.assertRaises(StormError):bad_storm.validated()
        runner._control=dataclasses.replace(runner._control,
            sediment_control=dataclasses.replace(runner._control.sediment_control,storm=bad_storm))
        with mock.patch.object(dr.DenseRunner,'_stage',side_effect=AssertionError('stage')) as stage:
            with self.assertRaises(dr.DenseRunnerError) as error:pause.pause_water(runner,runtime_reference=self.reference,boundary_ordinal=1)
            self.assertEqual(error.exception.code,'water_pause_control')
            self.assertTrue(error.exception.detail.startswith('invalid native control/runner limits: '))
            stage.assert_not_called()
        self.assertIsNone(runner._active_evidence)
        for feature in ('production','dynamic','characteristic','CSR','CSR_capacity','windowed_control'):
            runner=new_runner()
            if feature=='production':runner._production=object()
            elif feature=='dynamic':
                # Actual dynamic constructor adds these two fields. This is an
                # explicit CLOSED-FIELD capability refusal, not a fake dynamic
                # model success or growth/restore fixture.
                runner._vegetation_plan=object();runner._vegetation_copied_bytes=0
            else:
                overrides=({'transport_scheme':'characteristic'} if feature=='characteristic' else
                           {'phase_storage':'csr'} if feature=='CSR' else
                           {'phase_max_nnz':1} if feature=='CSR_capacity' else
                           {'transport_scheme':'characteristic','phase_storage':'csr',
                            'transport_implementation':'auto','deep_publication':'boundary',
                            'deep_publication_high_water_records':1})
                # Actual native validation itself refuses upwind+CSR and dense
                # phase capacity. Do not claim an impossible independently
                # valid upwind CSR profile. The valid windowed control metadata
                # is refused by this narrower wrapper before any native state.
                sc=dataclasses.replace(runner._control.sediment_control,**overrides)
                if feature in ('CSR','CSR_capacity'):
                    with self.assertRaises(SedimentEventError):sc.validated()
                else:
                    self.assertIs(sc.validated(),sc)
                runner._control=dataclasses.replace(runner._control,sediment_control=sc)
            with self.subTest(feature=feature),mock.patch.object(dr.DenseRunner,'_stage',side_effect=AssertionError('stage')) as stage:
                with self.assertRaises(dr.DenseRunnerError) as error:pause.pause_water(runner,runtime_reference=self.reference,boundary_ordinal=1)
                self.assertEqual(error.exception.code,'water_pause_profile' if feature in ('production','dynamic') else 'water_pause_control')
                if feature in ('CSR','CSR_capacity'):
                    self.assertTrue(error.exception.detail.startswith('invalid native control/runner limits: '))
                elif feature in ('characteristic','windowed_control'):
                    self.assertEqual(error.exception.detail,'original checkpoint-compatible array/upwind/bisection controls required')
                stage.assert_not_called()
            self.assertIsNone(runner._active_evidence)
        from maple.core.types.voxel import WindowedVoxelColumnState
        runner=new_runner();original=runner._bed._state
        # A real nominal native WINDOWED type with deliberately absent fields:
        # no deepstore/array/window constructor is invoked. Any premature
        # .mass_kg/window-data access would raise AttributeError and fail this
        # classified capability refusal. No fake dense holder is involved.
        windowed=object.__new__(WindowedVoxelColumnState)
        runner._bed._state=dataclasses.replace(original,voxel_column=windowed)
        try:
            with mock.patch.object(dr.DenseRunner,'_stage',side_effect=AssertionError('stage')) as stage:
                with self.assertRaises(dr.DenseRunnerError) as error:pause.pause_water(runner,runtime_reference=self.reference,boundary_ordinal=1)
                self.assertEqual(error.exception.code,'water_pause_profile');stage.assert_not_called()
        finally:runner._bed._state=original
        self.assertIsNone(runner._active_evidence)


class HeldRoutingParity(unittest.TestCase):
    def test_genuine_previous_rain_source_survives_wind_and_resume(self):
        pause._declare();ref=reference();events=f.schedule(f.rain(),f.wind(),f.rain())
        direct=new_runner(events=events);resumed=new_runner(events=events)
        for runner in (direct,resumed):
            for expected in ('rain','wind'):
                receipt=runner.step();self.assertEqual((receipt['status'],receipt['kind']),('accepted',expected))
        source=resumed._routing.source
        displacement=dr._native().committed_elevation_m(resumed._bed.state.committed_topography)-source
        self.assertGreater(float(np.max(np.abs(displacement))),0.)
        expected=direct.step();self.assertEqual(expected['status'],'accepted')
        actual_entry=dr._rain_enter;entries=[]
        def retained(*a,**kw):
            item=actual_entry(*a,**kw);entries.append((item.first,item.held.source));return item
        with mock.patch.object(dr,'_rain_enter',side_effect=retained):
            token=pause.pause_water(resumed,runtime_reference=ref,boundary_ordinal=1)
            self.assertIs(type(token),pause._WaterToken)
            result=pause.resume_water(resumed,token)
        self.assertEqual(result['status'],'accepted');self.assertEqual(len(entries),1)
        self.assertIs(entries[0][0],False);self.assertIs(entries[0][1],source)
        self.assertEqual(snapshot(resumed),snapshot(direct))


class ResourceOwnerByteAccounting(unittest.TestCase):
    """Synthetic interface fixtures; no transport or ecological physics."""

    def test_numpy_backend_is_not_a_resident_array(self):
        import dataclasses
        import numpy as np
        from masy import dense_bed

        @dataclasses.dataclass
        class Holdings:
            xp: object
            mass: object
            alias: object

        mass = np.arange(6, dtype=np.float64)
        holder = Holdings(np, mass, mass)
        self.assertEqual(dense_bed._tree_nbytes(holder, mass), mass.nbytes)
        self.assertEqual([value for _, value in dense_bed._require_host_arrays(holder, "holdings")], [mass, mass])

    def test_backend_namespace_is_not_probed_as_a_device_array(self):
        from types import ModuleType
        import numpy as np
        from masy import dense_bed

        backend = ModuleType("synthetic_backend_namespace")
        backend.shape, backend.dtype = np.shape, np.dtype
        def reject_probe(name):
            raise AssertionError("namespace must not be probed: " + name)
        backend.__getattr__ = reject_probe
        mass = np.zeros((2, 3), dtype=np.float64)
        self.assertEqual(dense_bed._tree_nbytes((backend, mass), {"backend": backend}), mass.nbytes)

    def test_device_array_refusal_and_unique_array_accounting_remain(self):
        import numpy as np
        from masy import dense_bed

        class SyntheticDeviceArray:
            shape = (2, 3)
            dtype = np.dtype("float64")
            nbytes = 48

        device = SyntheticDeviceArray()
        with self.assertRaises(dense_bed.DenseBedError):
            dense_bed._require_host_arrays((device,), "synthetic_device")
        left, right = np.zeros(2), np.zeros(3)
        self.assertEqual(dense_bed._tree_nbytes((left, right, left), {"right": right}), left.nbytes + right.nbytes)


if __name__=='__main__':
    unittest.main()


