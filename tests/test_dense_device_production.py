"""Finite CPU metadata/adapter controls only; no genuine device or transport qualification.

Native holder TYPES and native to_host are used; manufactured holders are explicitly observer/unit templates,
not accepted physical histories. No fixture/runner model, GPU query, native transport or child is started.
"""
from __future__ import annotations

import gc
import inspect
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from masy import dense_bed as bed_module
from masy import dense_device_production as production
from masy import dense_device_runner as device


def case_record():
    # Authored primitive metadata template; not a physical initialization or restart payload.
    record = dict.fromkeys(production._CASE_ROOT_KEYS)
    record.update(schema=bed_module.DENSE_CASE_SCHEMA, backend="numpy", dtype="float64",
                  representation="dense_voxel_column", n_voxels=6, active_mask="full",
                  depth_update_rule="constant_depth")
    return record


def context():
    from maple_syrup.sediment_bed import BedContext

    return BedContext(geometry=SimpleNamespace(ny=4, nx=4, dx_m=1.0, dy_m=1.0),
                      grain_classes=SimpleNamespace(classes=(object(),)), mass_resolution_kg=1.0,
                      topographic_commit_spec=object(), avalanching_spec=object(),
                      initial_available_fraction=np.full((4, 4, 1), 0.25))


def view():
    record = case_record()
    return production._HostBedView.create(None, context(), record, bed_module._sha256(record))


def limits(**changes):
    args = dict(fft_workspace_reserve_bytes=1 << 20, jit_driver_reserve_bytes=1 << 20,
                native_transient_reserve_bytes=1 << 20, diagnostic_scratch_bytes=4096)
    args.update(changes)
    return production.DeviceProductionLimits(**args)


def commit(diagnostics=None, applied=False):
    from maple.surface.topographic_commit.commit import CommitResult

    # Exact native holder TYPE; only observer fields are populated. No native commit is fabricated/run.
    value = object.__new__(CommitResult)
    object.__setattr__(value, "avalanche_applied", applied)
    object.__setattr__(value, "avalanche_diagnostics", diagnostics)
    return value


class CompactOwner:
    def __init__(self, water):
        self.water = water
        self.calls = []
        self.generation = 0

    def _compact(self, fn, *, max_values):
        value = fn(SimpleNamespace(water=self.water))
        assert value.ndim <= 1 and value.size <= max_values
        self.calls.append((max_values, int(value.size)))
        return np.array(value, dtype=np.float64, copy=True)


def owned_water():
    return SimpleNamespace(depth_m=np.zeros((4, 4)), mobile_mass_by_cell_class_kg=np.zeros((4, 4, 1)))


def adapter():
    owner = CompactOwner(owned_water())
    ops = SimpleNamespace(xp=np, _np=np, scalar=lambda a: float(np.asarray(a).reshape(-1)[0]),
                          float_values=lambda a: [float(x) for x in np.asarray(a).reshape(-1)])
    # Trusted seam template. It is NOT a DeviceDenseBedOwner and no GPU is claimed.
    own = object.__new__(device._DeviceOwnership)
    own._owner, own._ops, own._nc = owner, ops, 1
    return own


def native_observer_templates(own):
    water = SimpleNamespace(depth_m=own._owner.water.depth_m.copy(),
                            mobile_mass_by_cell_class_kg=own._owner.water.mobile_mass_by_cell_class_kg.copy())
    physical = SimpleNamespace(water=water, ledger_reconciliation_baseline=None)
    result = SimpleNamespace(state=SimpleNamespace(physical=physical, topographic_wind_cache=None),
        accounting=SimpleNamespace(water_clamp_added_volume_m3=0.0, water_displaced_volume_m3=None,
                                   launched_mass_by_class_kg=np.zeros(1)), last_scheduler_commit_result=None)
    final = SimpleNamespace(state=SimpleNamespace(physical=physical, topographic_wind_cache=None,
                                                  active_layer_composition_cache=None))
    return result, final


class MetadataAndLimits(unittest.TestCase):
    def test_original_dense_bed_and_volume_released(self):
        original = object.__new__(bed_module.DenseBed)
        original._context, original._case_record = context(), case_record()
        original._case_sha256 = bed_module._sha256(original._case_record)
        original._state = np.zeros((4, 4, 6, 1))  # tiny synthetic retention sentinel, not native physical state
        bed_ref, volume_ref = weakref.ref(original), weakref.ref(original._state)
        held = production._HostBedView.create(None, original.context, original._case_record, original._case_sha256)
        self.assertFalse(hasattr(held, "state"))
        self.assertEqual(held.__slots__, ("context", "case_record", "case_sha256"))
        del original
        gc.collect()
        self.assertIsNone(bed_ref())
        self.assertIsNone(volume_ref())
        held.verify()

    def test_context_extra_source_reference_refused(self):
        ctx, record = context(), case_record()
        object.__setattr__(ctx, "source_bed", object())
        with self.assertRaisesRegex(production.DenseProductionError, "declared fields"):
            production._HostBedView.create(None, ctx, record, bed_module._sha256(record))

    def test_record_copy_detached_and_mutation_refused(self):
        record, ctx = case_record(), context()
        record["geometry"] = {"synthetic": [1, 2]}
        held = production._HostBedView.create(None, ctx, record, bed_module._sha256(record))
        record["geometry"]["synthetic"][0] = 7
        held.verify()
        held.case_record["n_voxels"] = 5
        with self.assertRaisesRegex(production.DenseProductionError, "case_sha256"):
            held.verify()

    def test_metadata_cap_before_hash_or_copy(self):
        for value in (1 << 1024, "x" * 4097, object(), np.zeros(1), {"x": [0] * 129}):
            record = case_record()
            record["geometry"] = value
            with self.subTest(kind=type(value).__name__):
                with patch.object(bed_module, "_sha256", side_effect=AssertionError("hash must not run")):
                    with self.assertRaises(production.DenseProductionError):
                        production._HostBedView.create(None, context(), record, "0" * 64)

    def test_record_sha_and_root_shape_refused(self):
        for record, sha in ((case_record(), "x" * 64), ({**case_record(), "extra": None}, "0" * 64)):
            with self.assertRaises(production.DenseProductionError):
                production._HostBedView.create(None, context(), record, sha)

    def test_metadata_binding_checks_actual_context_and_case(self):
        held = view()
        # Exact private holder TYPE, metadata-only template. This does not construct/qualify device owners.
        metadata = object.__new__(device._HostMetadataBed)
        metadata._host_context, metadata.case_record, metadata.case_sha256 = held.context, held.case_record, held.case_sha256
        held.bind_metadata(metadata)
        duck = SimpleNamespace(host_context=held.context, case_record=held.case_record, case_sha256=held.case_sha256)
        with self.assertRaisesRegex(production.DenseProductionError, "exact private"):
            held.bind_metadata(duck)
        metadata._host_context = context()
        with self.assertRaises(production.DenseProductionError):
            held.bind_metadata(metadata)

    def test_controlling_record_delegates_original_gpu_profile_and_guards_drift(self):
        from masy.dense_wind_profile import DenseProductionGpuWindProfile
        held = view()
        ctx = held.context
        class SchedulerTemplate:
            pass
        class PoolTemplate:
            pass
        cfg = SchedulerTemplate()
        for name in ("geometry", "grain_classes", "avalanching_spec", "topographic_commit_spec",
                     "initial_available_fraction"):
            setattr(cfg, name, getattr(ctx, name))
        cfg.mass_resolution_kg, cfg.outer_timestep_s = 1.0, 1.0
        cfg.particle_transport = SimpleNamespace(mobile_state_representation="flat", max_active_cohorts=1,
                                                 quantile_bins=8)
        cfg.water_coupling_spec = SimpleNamespace(enabled=False, depth_update_rule="constant_depth")
        model = SimpleNamespace(geometry=ctx.geometry, grain_classes=ctx.grain_classes,
            avalanching=ctx.avalanching_spec, topographic_commit=ctx.topographic_commit_spec,
            water_coupling=cfg.water_coupling_spec)
        profile = DenseProductionGpuWindProfile(model, "synthetic_test_fixture", "unit delegate template")
        wind = SimpleNamespace(config=cfg, initial_mobile_state=PoolTemplate())
        native = SimpleNamespace(np=np, SchedulerPhysicsConfig=SchedulerTemplate, MobileCohortState=PoolTemplate,
                                  mobile_state_is_genuinely_empty=lambda value: True)
        schedule = SimpleNamespace(events=(None,) * 4)
        # Deliberate delegate spies; this is not a genuine ModelConfig or positive native GPU audit.
        audit = {"audit_sha256": "authored_metadata_witness", "metadata": "complete_audit_retained"}
        with patch("masy.dense_production.schedule_envelope", return_value=("wind_envelope", {"index": [0, 2]})), \
             patch("masy.dense_wind_profile.audit_dense_production_gpu_wind_config", return_value=audit) as called, \
             patch.object(bed_module, "_require_host_arrays"), \
             patch("maple_syrup.sediment_bed.reroute_threshold_m", return_value=0.5) as threshold:
            record = production.controlling_record(native, held, wind, schedule, profile, limits())
            called.assert_called_once_with(profile, cfg, wind_envelope_record="wind_envelope",
                                            initial_available_fraction=ctx.initial_available_fraction)
            threshold.assert_called_once_with(ctx)
            self.assertEqual(record["native_config_audit"], audit)
            cfg.geometry = object()
            with self.assertRaises(production.DenseProductionError):
                production.controlling_record(native, held, wind, schedule, profile, limits())
            self.assertEqual(called.call_count, 1)

    def test_finite_profile_admission_does_not_hash_copy_or_native_audit(self):
        from masy.dense_schedule import DenseSchedule, RainEvent, RainInterval, WindEvent
        from masy.dense_wind_profile import DenseProductionGpuWindProfile

        class SchedulerTemplate:
            pass
        class PoolTemplate:
            pass
        cfg = SchedulerTemplate()
        cfg.particle_transport = SimpleNamespace(max_active_cohorts=1, quantile_bins=1, mobile_state_representation="flat")
        wind = SimpleNamespace(config=cfg, initial_mobile_state=PoolTemplate())
        n = SimpleNamespace(np=np, SchedulerPhysicsConfig=SchedulerTemplate, MobileCohortState=PoolTemplate)
        schedule = DenseSchedule((WindEvent(1.0, 0.4, 270.0), RainEvent((RainInterval(1.0, 1.0),), 2.0),
                                  WindEvent(1.0, 0.4, 270.0), RainEvent((RainInterval(1.0, 1.0),), 2.0)))
        profile = DenseProductionGpuWindProfile(None, "synthetic_test_fixture", "explicit unit metadata delegate")
        # Native slot-layout delegate is authored metadata here; no ModelConfig or physical pool is created.
        with patch("masy.dense_device_resources._closed_metadata", return_value=((), [np.zeros(1)], 0)), \
             patch.object(bed_module, "_sha256", side_effect=AssertionError("early case hash")), \
             patch.object(production, "controlling_record", side_effect=AssertionError("early native audit")):
            record = production.profile_admission(n, context(), case_record(), "0" * 64, wind, schedule, profile,
                                                  limits(), SimpleNamespace(max_wind_accounting_steps=64))
            self.assertIs(record["case_copied_or_hashed"], False)
            bad = case_record()
            bad["n_voxels"] = 1 << 1024
            with self.assertRaises(production.DenseProductionError):
                production.profile_admission(n, context(), bad, "0" * 64, wind, schedule, profile,
                                             limits(), SimpleNamespace(max_wind_accounting_steps=64))

    def test_runner_production_coherence_refuses_both_mismatched_directions(self):
        # Private runner-shell metadata only. No owner, constructor or event is invoked.
        for runner_module, view_module in ((production, None), (None, production)):
            runner = object.__new__(device._DeviceDenseRunner)
            runner._production = runner_module
            runner._host_view = SimpleNamespace(_production=view_module)
            runner._device_resources = SimpleNamespace(verify_static=lambda: None)
            with patch.object(device, "_bind_device_inputs", side_effect=AssertionError("bind must not run")):
                with self.assertRaisesRegex(device.DenseRunnerError, "production module differs"):
                    runner._static_parts_record(SimpleNamespace(np=np))

    def test_native_scalar_admission_and_cap_refusal_without_copy(self):
        from maple.core.parameters.flow_separation import FlowSeparationSpec
        from masy.dense_schedule import DenseSchedule, RainEvent, RainInterval, WindEvent
        events = DenseSchedule((WindEvent(1.0, 0.4, 270.0), RainEvent((RainInterval(1.0, 1.0),), 2.0),
                                WindEvent(1.0, 0.4, 270.0), RainEvent((RainInterval(1.0, 1.0),), 2.0)))
        wind = SimpleNamespace(config=SimpleNamespace(flow_separation_spec=FlowSeparationSpec(
            enabled=True, max_bubble_length_m=1.9)))
        device_limits = SimpleNamespace(max_device_bytes=8 << 30)
        with patch.object(production, "_loaded_cupy", side_effect=AssertionError("no device read")):
            record = production.construction_admission(None, view(), wind, events, limits(), 1, device_limits)
            self.assertEqual(record["flow_separation"]["bubble_samples"], 3)
            self.assertEqual(record["full_production_peak"], "unqualified_not_estimated")
            with self.assertRaises(production.DenseProductionError):
                production.construction_admission(None, view(), wind, events,
                    limits(max_separation_samples=2), 1, device_limits)
            with self.assertRaises(production.DenseProductionError):
                production.construction_admission(None, view(), wind, events, limits(), 8 << 30, device_limits)

    def test_limits_closed_and_revalidate_forced_mutation(self):
        value = limits()
        self.assertEqual(value.as_record()["full_production_peak"], "unqualified_not_estimated")
        self.assertNotIn("pid", value.as_record())
        object.__setattr__(value, "max_events", True)
        with self.assertRaises(production.DenseProductionError):
            value.as_record()
        for bad in (0, True, 1 << 80):
            with self.subTest(value=bad), self.assertRaises(production.DenseProductionError):
                limits(fft_workspace_reserve_bytes=bad)

    def test_process_request_refuses_before_device_observation(self):
        with patch.object(production, "_loaded_cupy", side_effect=AssertionError("device read")):
            for kwargs in ({"upcoming_wrapper_bytes": -1}, {"upcoming_wrapper_bytes": True},
                           {"checkpoint": "after_native_invented"}, {"checkpoint": "unknown"}):
                with self.subTest(kwargs=kwargs), self.assertRaises(production.DenseProductionError):
                    production.check_process(limits(), {}, **kwargs)

    def test_process_metadata_contract_and_budget_failures(self):
        import os
        import resource

        policy = limits()
        identity = {"pid": 1, "start_ticks": 2, "boot_id": "authored_unit_identity", "device_id": 0,
                    "stream_ptr": 0, "cuda_visible_devices": "authored_unit_uuid"}
        state = {"free": 12 << 30, "total": 24 << 30, "used": 1024, "held": 2048, "cap": 8 << 30}
        pool = SimpleNamespace(used_bytes=lambda: state["used"], total_bytes=lambda: state["held"],
                               get_limit=lambda: state["cap"])
        cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(
            memGetInfo=lambda: (state["free"], state["total"]))), get_default_memory_pool=lambda: pool)
        enforced = {resource.RLIMIT_CPU: (policy.cpu_soft_s, policy.cpu_hard_s), resource.RLIMIT_CORE: (0, 0),
                    resource.RLIMIT_FSIZE: (policy.file_size_bytes, policy.file_size_bytes),
                    resource.RLIMIT_NOFILE: (policy.nofile, policy.nofile)}
        # All observations are explicit metadata delegates. No genuine device/process budget success is claimed.
        # The outer worker separately proves real -I/-B flags; this unit isolates the library predicate itself.
        flags = SimpleNamespace(isolated=1, no_user_site=1, dont_write_bytecode=1, optimize=0)
        process_sys = SimpleNamespace(platform="linux", flags=flags)
        with patch.object(production, "sys", process_sys), \
             patch.dict(os.environ, {"JAX_PLATFORMS": "cpu", "NUMBA_DISABLE_CUDA": "1"}, clear=True), \
             patch.object(resource, "getrlimit", side_effect=lambda key: enforced[key]), \
             patch.object(resource, "getrusage", return_value=SimpleNamespace(ru_utime=0.25, ru_stime=0.5)), \
             patch.object(production, "_read_bounded", return_value=b"100 5"), \
             patch.object(production, "_host_mem_available_bytes", return_value=2 << 30), \
             patch.object(production, "process_identity", return_value=dict(identity)), \
             patch.object(production, "_loaded_cupy", return_value=cp):
            record = production.check_process(policy, identity, upcoming_wrapper_bytes=policy.reserve_total_bytes,
                                               checkpoint="constructing")
            self.assertEqual(record["cpu_used_s"], 0.75)
            self.assertIn("unbounded_by_environment", record["thread_arena_policy"])
            self.assertEqual(record["device_free_floor"], "checked_at_construction")
            for name, bad in (("isolated", 0), ("no_user_site", 0), ("dont_write_bytecode", 0), ("optimize", 1)):
                good = getattr(flags, name)
                setattr(flags, name, bad)
                try:
                    with self.subTest(flag=name), self.assertRaisesRegex(production.DenseProductionError, "Linux with -I -B"):
                        production.check_process(policy, identity)
                finally:
                    setattr(flags, name, good)
            process_sys.platform = "win32"
            with self.assertRaisesRegex(production.DenseProductionError, "Linux with -I -B"):
                production.check_process(policy, identity)
            process_sys.platform = "linux"
            with patch.dict(os.environ, {"OMP_NUM_THREADS": "1"}):
                with self.assertRaisesRegex(production.DenseProductionError, "overrides are absent"):
                    production.check_process(policy, identity)
            with self.assertRaisesRegex(production.DenseProductionError, "pid differs"):
                production.check_process(policy, {**identity, "pid": 2})
            state["free"] = 0
            self.assertEqual(production.check_process(policy, identity, checkpoint="after_native_wind")
                             ["device_free_floor"], "not_applied_during_work")
            with self.assertRaisesRegex(production.DenseProductionError, "current driver free"):
                production.check_process(policy, identity, upcoming_wrapper_bytes=1, checkpoint="wrapper_work")
            state["free"] = 9 << 30
            with self.assertRaisesRegex(production.DenseProductionError, "min_device_free"):
                production.check_process(policy, identity, upcoming_wrapper_bytes=policy.reserve_total_bytes,
                                         checkpoint="constructing")
            state["free"] = 12 << 30
            state["used"] = state["held"] = 8 << 30
            with self.assertRaisesRegex(production.DenseProductionError, "used plus upcoming"):
                production.check_process(policy, identity, upcoming_wrapper_bytes=1, checkpoint="wrapper_work")
            state["held"] += 1
            with self.assertRaisesRegex(production.DenseProductionError, "held bytes"):
                production.check_process(policy, identity)

    def test_no_discovery_or_optional_accelerator_import(self):
        self.assertNotIn("cupy", sys.modules)
        with self.assertRaises(production.DenseProductionError):
            production._loaded_cupy()
        self.assertNotIn("cupy", sys.modules)


class DiagnosticAndOwnership(unittest.TestCase):
    def test_signed_zero_and_none_preserved_without_array_escape(self):
        original = np.array([-0.0])
        result = production.commit_evidence(np, commit({production._DIAGNOSTIC_NAMES[0]: original}))
        values = result["avalanche_diagnostics"]
        self.assertEqual(math.copysign(1.0, values[production._DIAGNOSTIC_NAMES[0]][0]), -1.0)
        self.assertIsNone(values[production._DIAGNOSTIC_NAMES[1]])
        original[0] = 4.0
        self.assertEqual(values[production._DIAGNOSTIC_NAMES[0]], [-0.0])
        self.assertEqual(production.commit_evidence(np, None)["status"], "no_commit")

    def test_native_flag_and_vector_refusals(self):
        with self.assertRaises(production.DenseProductionError):
            production.commit_evidence(np, commit(applied=np.bool_(True)))
        for value in (np.zeros(2), np.zeros(1, dtype=np.float32), np.array([float("nan")]), [0.0]):
            with self.subTest(kind=type(value).__name__), self.assertRaises(production.DenseProductionError):
                production.commit_evidence(np, commit({production._DIAGNOSTIC_NAMES[0]: value}))

    def test_wet_and_unknown_module_refused_without_publication(self):
        own = adapter()
        own.production_require_dry(production, np)
        own._owner.water.depth_m[0, 0] = 0.1
        with self.assertRaises(production.DenseProductionError):
            own.production_require_dry(production, np)
        with self.assertRaises(production.DenseProductionError):
            own.production_require_dry(SimpleNamespace(), np)
        self.assertEqual(own._owner.generation, 0)

    def test_final_cache_refusal_does_not_discard_result_witness(self):
        own = adapter()
        result, final = native_observer_templates(own)
        result.state.topographic_wind_cache = SimpleNamespace(modifier=SimpleNamespace(
            algorithm_version="synthetic_unit_template", wind_direction_deg=270.0,
            separation=SimpleNamespace(bubble_mask=np.zeros((4, 4), dtype=bool))))
        evidence = own.production_wind_evidence(production, SimpleNamespace(np=np), result, final)
        self.assertEqual(evidence["topographic_wind"]["separation_bubble_cells"], 0)
        final.state.active_layer_composition_cache = object()
        with self.assertRaises(production.DenseProductionError):
            own.production_wind_evidence(production, SimpleNamespace(np=np), result, final)
        self.assertEqual(own._owner.generation, 0)

    def test_water_byte_change_and_exchange_refused(self):
        own = adapter()
        result, final = native_observer_templates(own)
        final.state.physical.water.depth_m[0, 0] = -0.0
        with self.assertRaises(production.DenseProductionError):
            own.production_wind_evidence(production, SimpleNamespace(np=np), result, final)
        self.assertEqual(own._owner.generation, 0)
        result, final = native_observer_templates(own)
        result.accounting.water_displaced_volume_m3 = 1e-12
        with self.assertRaises(production.DenseProductionError):
            own.production_wind_evidence(production, SimpleNamespace(np=np), result, final)

    def test_bad_water_layout_refused_before_comparison(self):
        own = adapter()
        result, final = native_observer_templates(own)
        result.state.physical.water.depth_m = np.zeros((4, 4), dtype=np.float32)
        with self.assertRaises(production.DenseProductionError):
            own.production_wind_evidence(production, SimpleNamespace(np=np), result, final)

    def test_retained_owner_metadata_and_host_table_refusals(self):
        own = adapter()
        calls = []
        own._owner._scope = lambda: calls.append("bed_scope")
        own._owner.evidence = lambda: {"scope": "unit_owner_metadata_only"}
        own._res = SimpleNamespace(_scope=lambda: calls.append("private_scope"),
            evidence=lambda: {"scope": "unit_private_metadata_only"}, pool_bytes=lambda: 64, soil_bytes=lambda: 128)
        policy = np.full((4, 4, 1), 0.25)
        own._metadata = SimpleNamespace(host_context=SimpleNamespace(initial_available_fraction=policy))
        hop = SimpleNamespace(hop_distance_m=np.ones((1, 8)), weight=np.full((1, 8), 0.125))
        wind = SimpleNamespace(config=SimpleNamespace(hop_bins=hop))
        # Scope/evidence are deliberate metadata delegates; no real device owner, resource or routing exists.
        result = own.production_retained_owners(production, wind, None, None, None, None, None, None, 0)
        self.assertEqual(calls, ["bed_scope", "private_scope"])
        self.assertEqual(result["host_reference_bytes"], {"hop_tables": 128, "initial_host_policy": 128})
        self.assertEqual(result["owned_device_bytes"], {"private_pool": 64, "private_soil": 128})
        self.assertEqual(result["routing"], {"status": "absent"})
        self.assertEqual(own._owner.generation, 0)
        for bad in (True, 9):
            with self.subTest(receipts=bad), self.assertRaises(production.DenseProductionError):
                own.production_retained_owners(production, wind, None, None, None, None, None, None, bad)
        hop.weight = np.zeros((1, 8), dtype=np.float32)
        with self.assertRaisesRegex(production.DenseProductionError, "host float64"):
            own.production_retained_owners(production, wind, None, None, None, None, None, None, 0)
        hop.weight = np.zeros((1, 8))
        own._metadata.host_context.initial_available_fraction = np.zeros((4, 4), dtype=np.float64)
        with self.assertRaisesRegex(production.DenseProductionError, "host initial-availability"):
            own.production_retained_owners(production, wind, None, None, None, None, None, None, 0)

    def test_default_factory_signature_and_native_delegates_remain(self):
        signature = inspect.signature(device._create_dense_device_runner)
        self.assertEqual(tuple(signature.parameters), ("bed", "schedule", "wind", "water", "control", "limits",
            "device_limits", "bed_limits", "resource_limits", "device_id"))
        self.assertNotIn("wind_profile", signature.parameters)
        self.assertTrue(all(callable(getattr(device._runner, name)) for name in
                            ("_advance_wind_block", "_finalize_wind_event", "_complete_event")))

    def test_new_factory_failure_clears_only_candidate_made(self):
        made_seen = []
        def fail(*args, **kwargs):
            made = args[-1]
            made["synthetic_candidate"] = object()
            made_seen.append(made)
            raise production.DenseProductionError("synthetic_refusal", "unit delegate failure")
        with patch.object(device, "_build_impl", side_effect=fail):
            with self.assertRaisesRegex(device.DenseRunnerError, "synthetic_refusal"):
                device._create_dense_production_device_runner(None, None, wind=None, water=None, control=None,
                    limits=None, device_limits=None, bed_limits=None, resource_limits=None, device_id=0,
                    wind_profile=None, production_limits=None, worker_identity={})
        self.assertEqual(made_seen, [{}])


class ExplicitConstructionFreeFloor(unittest.TestCase):
    """Actual pure policy predicates over synthetic scalar counter inputs.

    No provider/device/process observation, physical owner or model succeeds
    here. Root must separately qualify a new explicit named 9 GiB profile.
    """

    def _observe(self, policy, **changes):
        counters = {"free": 9 << 30, "total": 24 << 30, "used": 0, "held": 0,
                    "cap": 8 << 30, "upcoming": policy.reserve_total_bytes,
                    "checkpoint": "constructing"}
        counters.update(changes)
        return production._check_observed_device_memory(policy, **counters)

    def test_explicit_nine_default_ten_and_original_upper_bound(self):
        nine, default = limits(min_device_free_bytes=9 << 30), limits()
        self.assertEqual(nine.min_device_free_bytes, 9 << 30)
        self.assertEqual(default.min_device_free_bytes, 10 << 30)
        self.assertEqual(nine.as_record()["policy"]["min_device_free_bytes"], 9 << 30)
        self.assertEqual(default.as_record()["policy"]["min_device_free_bytes"], 10 << 30)
        self.assertEqual(limits(min_device_free_bytes=1 << 40).min_device_free_bytes, 1 << 40)
        for value in ((9 << 30) - 1, (1 << 40) + 1):
            with self.subTest(value=value), self.assertRaises(production.DenseProductionError) as caught:
                limits(min_device_free_bytes=value)
            self.assertEqual(caught.exception.code, "invalid_limits")

    def test_nine_constructing_equality_uses_original_scalar_predicate(self):
        policy = limits(min_device_free_bytes=9 << 30)
        with patch.object(production, "_loaded_cupy", side_effect=AssertionError("no provider query")):
            self.assertIsNone(self._observe(policy, free=9 << 30))

    def test_below_nine_preserves_original_eight_counter_refusal(self):
        policy = limits(min_device_free_bytes=9 << 30)
        free = (9 << 30) - 1
        with self.assertRaises(production.DenseProductionError) as caught:
            self._observe(policy, free=free)
        self.assertEqual(caught.exception.code, "device_free_floor")
        expected = (free, 24 << 30, 0, 0, 8 << 30, policy.reserve_total_bytes, 9 << 30, 8 << 30)
        self.assertEqual(caught.exception._device_memory_refusal, expected)
        self.assertEqual(production._copy_device_memory_refusal(caught.exception), expected)

    def test_default_ten_still_refuses_below_ten_and_accepts_equality(self):
        default = limits()
        for free in (9 << 30, (10 << 30) - 1):
            with self.subTest(free=free), self.assertRaises(production.DenseProductionError) as caught:
                self._observe(default, free=free)
            self.assertEqual(caught.exception.code, "device_free_floor")
            self.assertEqual(caught.exception._device_memory_refusal[6], 10 << 30)
        self.assertIsNone(self._observe(default, free=10 << 30))

    def test_nonconstructing_checkpoints_do_not_reapply_either_floor(self):
        for policy in (limits(), limits(min_device_free_bytes=9 << 30)):
            for checkpoint in ("event_entry", "wrapper_work", "after_native_wind",
                               "after_native_rain", "before_publication"):
                with self.subTest(minimum=policy.min_device_free_bytes, checkpoint=checkpoint):
                    self.assertIsNone(self._observe(policy, free=0, upcoming=0, checkpoint=checkpoint))

    def test_original_pool_and_upcoming_guards_still_refuse_with_nine(self):
        policy = limits(min_device_free_bytes=9 << 30)
        cases = (({"held": (8 << 30) + 1}, "device_pool_limit"),
                 ({"cap": (8 << 30) - 1}, "device_pool_limit"),
                 ({"used": 8 << 30, "held": 8 << 30, "upcoming": 1}, "device_budget"),
                 ({"free": 1, "upcoming": 2, "checkpoint": "wrapper_work"}, "device_budget"))
        for changes, code in cases:
            with self.subTest(changes=changes), self.assertRaises(production.DenseProductionError) as caught:
                self._observe(policy, **changes)
            self.assertEqual(caught.exception.code, code)

    def test_underreserve_constructing_refuses_before_any_observation(self):
        policy = limits(min_device_free_bytes=9 << 30)
        with patch.object(production, "_loaded_cupy", side_effect=AssertionError("no provider query")), \
             patch.object(production, "process_identity", side_effect=AssertionError("no process query")), \
             patch.object(production, "_read_bounded", side_effect=AssertionError("no host query")):
            with self.assertRaises(production.DenseProductionError) as caught:
                production.check_process(policy, {}, upcoming_wrapper_bytes=policy.reserve_total_bytes - 1,
                                         checkpoint="constructing")
        self.assertEqual(caught.exception.code, "invalid_process_request")


if __name__ == "__main__":
    unittest.main()
