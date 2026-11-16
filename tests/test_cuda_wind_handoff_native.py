"""Actual resident CUDA wind/water authority controls; explicitly selected GPU only."""
import dataclasses
import os
import tempfile
import unittest
from pathlib import Path

@unittest.skipUnless(os.environ.get('MASY_CUDA_NATIVE')=='1','Root bounded CUDA native runtime required')
class NativeWindHandoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cupy as cp
        from maple_syrup import floating_wind_handoff_cuda as wh, floating_event_lifecycle as lc
        from maple_syrup import floating_event_lifecycle_cuda as flc, floating_matrix_infiltration as fm
        from maple_syrup import vegetation_lifecycle as vl
        from maple.core.backend import read_transfer_counters
        from cuda_triplet_support import fixture
        cls.cp,cls.wh,cls.lc,cls.flc,cls.fm,cls.vl,cls.transfers=cp,wh,lc,flc,fm,vl,staticmethod(read_transfer_counters)
        cls.tmp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.tmp.cleanup)
        cls.session,cls.fixture=fixture(Path(cls.tmp.name),initial_surface_depth=0.001,saturation=0.9999999,bottom='closed')
        cls.parent=cls.session.lineage.head
        cls.occurrence=next(iter(cls.session.plan)).winds[0]
        cls.vegetation=cls.session._wind_vegetation()
        cls.old=cls.fingerprint(cls.parent)
        cls.completed=cls.session.wind._wind.execute_finalized(cls.parent.bed,cls.occurrence,float(cls.parent.t_s),vegetation=cls.vegetation)
        assert cls.old==cls.fingerprint(cls.parent),'actual native wind mutated accepted parent'

    @classmethod
    def fingerprint(cls,b):
        from maple_syrup.floating_wind_handoff_cuda import _integrity
        return _integrity(cls.cp,b)

    @classmethod
    def kwargs(cls):
        s=cls.session;return dict(context=s.wind._binding.context,occurrence_id=cls.occurrence.occurrence_id,occurrence_sha256=cls.occurrence.occurrence_sha256,
            start_s=float(cls.parent.t_s),limits=s.wind._limits,batch_cells=s.wind._batch,available_bytes_fn=s.policy.device_available_bytes_fn,
            reserve_bytes=s.policy.device_reserve_bytes,host_available_bytes_fn=s.policy.host_available_bytes_fn,host_reserve_bytes=s.policy.host_reserve_bytes,
            memory_budget_bytes=s.policy.max_device_bytes,host_memory_budget_bytes=s.policy.max_host_bytes)

    @classmethod
    def stage(cls,**changes):
        return cls.wh.stage_wind_handoff(cls.session.lineage,cls.parent,cls.completed.finalization,cls.completed.candidate.state,**{**cls.kwargs(),**changes})

    def assert_parent(self):
        self.assertIs(self.session.lineage.head,self.parent);self.assertEqual(self.old,self.fingerprint(self.parent))

    def test_01_surface_and_persistent_history_survive_actual_wind(self):
        c=self.stage();p=c._cell['parts'];root=self.parent.root
        self.assertEqual(c.receipt()['water']['surface_leaves_bitwise_equal'],True)
        self.assertTrue(self.cp.array_equal(self.parent.bed.water.depth_m,self.completed.candidate.state.water.depth_m))
        self.assertTrue(self.cp.array_equal(self.parent.bed.water.mobile_mass_by_cell_class_kg,self.completed.candidate.state.water.mobile_mass_by_cell_class_kg))
        self.assertTrue(self.cp.array_equal(root.history.wetting_m,p.base.history.wetting_m))
        self.assertTrue(self.cp.array_equal(root.history.cumulative_intake_m,p.base.history.cumulative_intake_m))
        self.assertIs(p.base.routing,self.parent.routing);self.assertIs(p.vegetation,root.vegetation)
        self.assertGreater(c.receipt()['properties']['n_dirty'],0)
        self.assertGreater(c.receipt()['properties']['n_changed'],0)
        self.assertGreater(c.receipt()['water']['zero_time_reconcile']['surface_return_m3'],0)
        self.assertEqual(c.role,'pending_recession');self.assert_parent()

    def test_02_host_and_device_refuse_separately_before_arrays(self):
        for key in ('host_memory_budget_bytes','memory_budget_bytes'):
            before=self.transfers()
            with self.subTest(key=key),self.assertRaises(MemoryError):self.stage(**{key:1})
            delta=self.transfers().delta(before)
            self.assertEqual(delta.host_to_device_bytes+delta.device_to_host_bytes,0)
            self.assert_parent()

    def test_03_callback_failure_and_candidate_mutation_publish_nothing(self):
        c=self.stage();q=self.kwargs();q={k:v for k,v in q.items() if k in ('available_bytes_fn','reserve_bytes','host_available_bytes_fn','host_reserve_bytes','memory_budget_bytes','host_memory_budget_bytes')}
        def fail():raise RuntimeError('labelled failure before publication')
        with self.assertRaisesRegex(RuntimeError,'labelled failure'):self.wh.adopt_wind_handoff(self.session.lineage,self.parent,c,before_publish=fail,**q)
        self.assertFalse(c.consumed);self.assert_parent()
        a=c._cell['parts'].base.water_state.water_m;old=a[0,0].copy();a[0,0]+=1e-8
        try:
            with self.assertRaisesRegex(self.wh.WindHandoffError,'integrity'):self.wh.adopt_wind_handoff(self.session.lineage,self.parent,c,**q)
        finally:a[0,0]=old
        self.assert_parent()

    def test_04_wrong_physical_device_refuses_before_readers(self):
        p=self.fixture.policy;called=[]
        bad=dataclasses.replace(p,physical_uuid='GPU-ffffffff-0000-0000-0000-000000000000',host_available_bytes_fn=lambda:called.append('h'),device_available_bytes_fn=lambda:called.append('d'))
        with self.assertRaisesRegex(ValueError,'UUID'):bad.admit(0,0,'negative selected UUID')
        self.assertEqual(called,[]);self.assert_parent()

    def test_05_foreign_host_pool_and_incomplete_finalizer_refuse(self):
        from maple.core.backend import to_host_tree
        final=self.completed.finalization;phys=final.state.physical
        foreign=dataclasses.replace(final,state=dataclasses.replace(final.state,physical=dataclasses.replace(phys,mobile_state=to_host_tree(phys.mobile_state))))
        with self.assertRaisesRegex(self.wh.WindHandoffError,'host|foreign|mixed'):
            self.wh.stage_wind_handoff(self.session.lineage,self.parent,foreign,self.completed.candidate.state,**self.kwargs())
        bad=dataclasses.replace(final,completed=False)
        with self.assertRaises(self.wh.WindHandoffError):self.wh.stage_wind_handoff(self.session.lineage,self.parent,bad,self.completed.candidate.state,**self.kwargs())
        self.assert_parent()

    def test_99_adopt_once_requires_genuine_recession_and_exact_water(self):
        c=self.stage();q=self.kwargs();q={k:v for k,v in q.items() if k in ('available_bytes_fn','reserve_bytes','host_available_bytes_fn','host_reserve_bytes','memory_budget_bytes','host_memory_budget_bytes')}
        b=self.wh.adopt_wind_handoff(self.session.lineage,self.parent,c,**q)
        self.assertTrue(c.consumed);self.assertIs(c.accepted_boundary_ref(),b);self.assertIs(self.session.lineage.head,b)
        self.assertIs(b.routing,self.parent.routing);self.assertEqual(b.role,'pending_recession')
        self.assertFalse(b.eligible_for_interevent);self.assertEqual(float(b.root.water_state.t_s).hex(),float(b.t_s).hex())
        self.assertLessEqual(abs(b.ledger['residual_m3']),b.ledger['residual_bound_m3'])
        with self.assertRaises(self.lc.LifecycleError):self.wh.adopt_wind_handoff(self.session.lineage,self.parent,c,**q)
        with self.assertRaises(self.lc.LifecycleError):self.stage()
        with self.assertRaises(self.wh.WindHandoffError):self.wh.restore_wind_context(b,None,None)

        # This adverse near-saturated closed-bottom case is deliberately a
        # bounded continuation control. Full recession exceeded Root's CPU
        # guard in the preserved prior attempt; no quiet state is fabricated.
        plant_sequence=self.session.plants.sequence
        occ,state=self.lc.begin_occurrence(self.session.lineage,b,self.session._factory(b),**self.session._quota(budgets=True))
        self.assertIs(self.session.lineage.head,occ)
        evolved=occ.event.evolve(state,float(state.t_s)+0.2,stop_when_quiet=True,max_steps=1)
        self.assertGreater(float(evolved.state.t_s),float(state.t_s))
        self.assertIs(self.session.lineage.head,occ)
        from masy.daily_wind_handoff import WindHandoffError
        with self.assertRaises(WindHandoffError):self.session.wind.execute(self.occurrence,vegetation=self.vegetation)
        self.assertEqual(self.session.plants.sequence,plant_sequence)
        self.assertEqual(occ.ledger.get('evaporation_m3',0.0),0.0)
        self.assertEqual(occ.ledger.get('transpiration_m3',0.0),0.0)


@unittest.skipUnless(os.environ.get('MASY_CUDA_NATIVE')=='1','Root bounded CUDA native runtime required')
class NativeReducedRain(unittest.TestCase):
    def test_actual_daily_recharge_at_five_mm_equality_uses_reduced_branch(self):
        from cuda_triplet_support import fixture
        with tempfile.TemporaryDirectory() as tmp:
            session,f=fixture(Path(tmp),precipitation=5.0,threshold=5.0)
            record=session.advance().record()
            self.assertFalse(record['full_storm_completed'])
            self.assertEqual(session.accepted_days,1)
            self.assertEqual(session.plants.counters()['withdrawals_accepted'],1)
            self.assertGreater(session.lineage.head.ledger['transpiration_m3'],0.0)
            led=session.lineage.head.ledger
            self.assertAlmostEqual(led['rain_m3'],5e-3*f.context.geometry.nx*f.context.geometry.ny*f.context.geometry.dx_m*f.context.geometry.dy_m,places=14)
            self.assertLessEqual(abs(led['residual_m3']),led['residual_bound_m3'])
