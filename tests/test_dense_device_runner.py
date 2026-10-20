"""CPU semantic / non-transport tests of the private Dense device binding.

Run explicitly with the task source-origin harness. Exact native holders and the authored six-class
alternation fixture are used; the two private owners receive the SAME NumPy `_ops` semantic adapter as
`test_dense_device_bed.py` through their private `_create` seams. The public factory stays genuine-CuPy only
and is exercised here ONLY up to its pre-copy refusals (CuPy absent). The retained-routing checks call the
actual native SYRUP first-rain constructors (graph, `initial_event_state`, entry `commit_bed`, `initial_state`)
over the NumPy owners. Nothing here runs a native wind step, finalizer, rain event, CUDA hydraulics or any
solver: these tests prove guards, ownership, labels and reference release, never GPU physics or CPU-GPU
agreement.
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import importlib.util
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from maple.core.backend import arrays_share_memory, read_transfer_counters
from maple_syrup.event_backend import host_packet

from masy import dense_bed, dense_runner
from masy import dense_device_bed as dbed
from masy import dense_device_resources as dres
from masy import dense_device_runner as drun

_NATIVE_TESTS = Path(__file__).resolve().parent / "native"
sys.path.insert(0, str(_NATIVE_TESTS))
import fixture_dense_alternation as fx  # noqa: E402

NC = len(fx.CLASS_IDS)


class CpuOps:
    """NumPy semantic adapter (same shape as `dense_device_bed._CupyOps`); never a GPU substitute."""

    array_type = np.ndarray
    xp = np
    host_np = np

    def __init__(self):
        self.device_id, self.stream_id = 0, 0
        self.clone_calls, self.packet_calls = 0, 0

    def current_device(self):
        return self.device_id

    def current_stream(self):
        return self.stream_id

    def device(self, array):
        return 0

    def clone(self, array):
        self.clone_calls += 1
        return array.copy(order="C")

    def overlap(self, a, b):
        return arrays_share_memory(a, b)

    def raw_equal(self, a, b):
        return np.all(a.reshape(-1).view(np.uint8) == b.reshape(-1).view(np.uint8))

    def packet(self, flags):
        self.packet_calls += 1
        return host_packet(np, flags)

    def vector_packet(self, vector):
        return host_packet(np, [], vector)

    def counters(self):
        return dataclasses.asdict(read_transfer_counters())


def device_limits(**overrides):
    values = dict(max_device_bytes=1 << 34, hydraulics_max_log=64, wind_workspace_bytes=1 << 20,
                  routing_workspace_bytes=1 << 20)
    values.update(overrides)
    return drun.DeviceRunnerLimits(**values)


def bed_limits():
    return dbed.DeviceBedLimits(1 << 26, 1 << 30, 1 << 20, 1 << 20)


def resource_limits():
    return dres.DeviceResourceLimits(1 << 26, 1 << 26, 1 << 30, 1 << 20)


def inputs():
    bed = fx.build_bed()
    wind = dense_runner.DenseWindInputs(fx.wind_config(bed), fx.empty_pool())
    water = fx.water_inputs(bed)
    control = fx.runner_control(sediment_control=fx.sediment_control(implementation="cuda"))
    return bed, wind, water, control


def statics_of(bed, wind, water):
    return {"vegetation": bed._vegetation, "threshold_modifier_state": bed._threshold_modifier_state,
            "initial_available_fraction": wind.config.initial_available_fraction, "column": water.column,
            "field": water.field, "sediment": water.sediment}


def owners(ops=None):
    """Both private owners over host copies through their private seams (the factory's genuine conversion is
    replaced ONLY by NumPy copies of the same source objects)."""
    bed, wind, water, control = inputs()
    ops = CpuOps() if ops is None else ops
    ctx = bed.context
    owner = dbed.DeviceDenseBedOwner._create(
        bed.state, geometry=ctx.geometry, mass_resolution_kg=ctx.mass_resolution_kg,
        n_voxels=bed.case_record["n_voxels"], n_classes=NC, case_sha256=bed.case_sha256, time_s=bed.time_s,
        limits=bed_limits(), _ops=ops)
    resources = dres.DeviceDenseResources._create(
        pool=wind.initial_mobile_state, soil=np.zeros((fx.NY, fx.NX)), statics=statics_of(bed, wind, water),
        shape=(fx.NY, fx.NX), limits=resource_limits(), _ops=ops)
    return SimpleNamespace(bed=bed, wind=wind, water=water, control=control, owner=owner, resources=resources,
                           ops=ops)


def ownership(env):
    metadata = drun._HostMetadataBed(env.owner, env.bed, env.resources)
    array_ops = drun._DeviceArrayOps(np, NC)  # NumPy namespace: the semantic seam, not CuPy
    own = drun._DeviceOwnership(env.owner, env.resources, array_ops, NC, 1 << 20, metadata)
    return own, metadata


def device_inputs(env, terrain=None):
    """The factory's shadow device inputs, built exactly as `_build` builds them, over the owned statics. An
    explicit `terrain` lets a second adapter share another run's caller-owned TerrainReference."""
    res = env.resources
    wind = dense_runner.DenseWindInputs(
        config=dataclasses.replace(env.wind.config,
                                   initial_available_fraction=res.static("initial_available_fraction")),
        initial_mobile_state=None)
    water = dense_runner.DenseWaterInputs(
        terrain=env.water.terrain if terrain is None else terrain, column=res.static("column"),
        field=res.static("field"),
        vegetation_cover_fraction=res.static("vegetation").raster.cover_fraction, sediment=res.static("sediment"),
        initial_soil_water_m=None)
    return wind, water


def factory(env, **overrides):
    args = dict(wind=env.wind, water=env.water, control=env.control,
                limits=fx.limits(max_wind_accounting_steps=64), device_limits=device_limits(),
                bed_limits=bed_limits(), resource_limits=resource_limits(), device_id=0)
    args.update(overrides)
    return drun._create_dense_device_runner(env.bed, fx.reference_schedule(), **args)


def writable(array):
    if not array.flags.writeable:
        array.flags.writeable = True  # owned private copy: test-only fault injection
    return array


class PreCopyAdmission(unittest.TestCase):
    """Refusals happen before ANY host->device conversion, and without CuPy."""

    def env(self):
        bed, wind, water, control = inputs()
        return SimpleNamespace(bed=bed, wind=wind, water=water, control=control)

    def refused(self, env, code, **overrides):
        with (mock.patch.dict(sys.modules, {"cupy": None}),
              mock.patch("maple.core.backend.to_device_tree", side_effect=AssertionError("copied")),
              self.assertRaises(dense_runner.DenseRunnerError) as caught):
            factory(env, **overrides)
        self.assertEqual(caught.exception.code, code)
        self.assertFalse(any(k == "cupy" or k.startswith("cupy.") for k in sys.modules if sys.modules[k]))

    def test_unbounded_accounting_steps_refuse_before_backend(self):
        self.refused(self.env(), "unsupported_profile", limits=fx.limits())  # fixture default 4096 > 64

    def test_cohorts_above_first_profile_refuse_before_backend(self):
        env = self.env()
        particle = copy.copy(env.wind.config.particle_transport)  # bypass native validation: a cap test only
        object.__setattr__(particle, "max_active_cohorts", drun._FIRST_PROFILE_CAPS["max_flat_cohorts"] + 1)
        big = copy.copy(env.wind.config)
        object.__setattr__(big, "particle_transport", particle)
        self.refused(env, "unsupported_profile", wind=dense_runner.DenseWindInputs(big, env.wind.initial_mobile_state))

    def test_pool_leaf_not_slot_shaped_refuses(self):
        env = self.env()
        pool = copy.copy(env.wind.initial_mobile_state)
        object.__setattr__(pool, "mass_kg", np.zeros(3))
        self.refused(env, "unsupported_profile", wind=dense_runner.DenseWindInputs(env.wind.config, pool))

    def test_undeclared_bed_attribute_refuses_before_backend(self):
        env = self.env()
        object.__setattr__(env.bed.state.active_layer, "hidden_payload", np.ones(1))
        self.refused(env, "unsupported_schema")

    def test_dynamic_vegetation_bed_refuses_before_native_or_backend(self):
        env = self.env()
        env.bed._dynamic = object()  # test-only marker: the factory dispatch guard reads its presence only
        # A native-namespace or CuPy access would surface as `device_construction_failure`, not this code.
        with (mock.patch.object(dense_runner, "_native", side_effect=AssertionError("native")),
              mock.patch.object(drun, "_genuine_cupy", side_effect=AssertionError("cupy"))):
            self.refused(env, "unsupported_profile")

    def test_valid_first_profile_reaches_only_the_genuine_backend_refusal(self):
        self.refused(self.env(), "unsupported_backend")

    def test_numpy_module_is_not_accepted_as_cupy(self):
        from types import ModuleType

        fake = ModuleType("cupy")
        fake.ndarray = np.ndarray
        with mock.patch.dict(sys.modules, {"cupy": fake}), self.assertRaises(dense_runner.DenseRunnerError) as c:
            drun._genuine_cupy()
        self.assertEqual(c.exception.code, "unsupported_backend")

    def test_owner_caps_refuse_before_conversion(self):
        env = self.env()
        soil = np.zeros((fx.NY, fx.NX))
        g = env.bed.context.geometry
        device, _ = drun._water_parameter_sources(np, env.water.column, env.water.sediment, (g.ny, g.nx), NC)
        for small_bed, small_res in ((dbed.DeviceBedLimits(1, 1, 1, 1), resource_limits()),
                                     (bed_limits(), dres.DeviceResourceLimits(1, 1, 1, 1))):
            with self.subTest(bed=small_bed), self.assertRaises(dense_runner.DenseRunnerError) as caught:
                drun._host_reservations(env.bed, env.wind, env.water, soil, device, small_bed, small_res,
                                        device_limits())
            self.assertEqual(caught.exception.code, "work_bytes_cap")

    def test_tampered_frozen_limits_refuse(self):
        env = self.env()
        limits = bed_limits()
        object.__setattr__(limits, "max_work_bytes", True)
        soil = np.zeros((fx.NY, fx.NX))
        with self.assertRaises(dense_runner.DenseRunnerError) as caught:
            drun._host_reservations(env.bed, env.wind, env.water, soil, [], limits, resource_limits(),
                                    device_limits())
        self.assertEqual(caught.exception.code, "invalid_limit")

    def test_declared_allowances_cannot_be_zero(self):
        for name in ("wind_workspace_bytes", "routing_workspace_bytes", "max_device_bytes"):
            with self.subTest(name=name), self.assertRaises(dense_runner.DenseRunnerError):
                device_limits(**{name: 0})
        with self.assertRaises(dense_runner.DenseRunnerError):
            device_limits(cuda_mode="numba")

    def test_private_runner_cannot_be_constructed_publicly(self):
        with self.assertRaises(dense_runner.DenseRunnerError):
            drun._DeviceDenseRunner()
        self.assertEqual(drun.__all__, [])
        imports = [line.strip() for line in Path(dense_runner.__file__).read_text().splitlines()
                   if line.strip().startswith(("import ", "from "))]
        self.assertFalse([line for line in imports if "dense_device" in line or "cupy" in line])


class WaterParameterBridge(unittest.TestCase):
    def test_closed_native_sources_split_device_and_host_leaves(self):
        bed, _, water, _ = inputs()
        device, host = drun._water_parameter_sources(np, water.column, water.sediment, (fx.NY, fx.NX), NC)
        names = [name for name, _ in device]
        self.assertTrue(all(n.startswith("column.") or n.startswith("sediment.d_") for n in names))
        self.assertEqual([n for n, _ in host], [f"sediment.{n}" for n in drun._SEDIMENT_HOST])

    def test_bridge_keeps_every_native_value_and_switches_only_xp_and_device_fields(self):
        _, _, water, _ = inputs()
        column, sediment = water.column, water.sediment
        new_column, new_sediment = drun._device_water_parameters(np, column, sediment)
        self.assertIs(type(new_column), type(column))
        self.assertIs(type(new_sediment), type(sediment))
        self.assertEqual(set(vars(new_sediment)), set(vars(sediment)))
        for name, value in vars(sediment).items():
            if name not in drun._SEDIMENT_DEVICE and name != "xp":
                self.assertIs(getattr(new_sediment, name), value, name)  # host summaries/scalars untouched
        self.assertEqual(new_column.model, column.model)
        for name in drun._COLUMN_ARRAYS:
            old, new = getattr(column, name), getattr(new_column, name)
            if old is not None:
                self.assertEqual(old.tobytes(), new.tobytes())

    def test_undeclared_payload_and_wrong_namespace_refuse(self):
        _, _, water, _ = inputs()
        object.__setattr__(water.sediment, "hidden_payload", object())
        with self.assertRaises(dense_runner.DenseRunnerError) as caught:
            drun._water_parameter_sources(np, water.column, water.sediment, (fx.NY, fx.NX), NC)
        self.assertEqual(caught.exception.code, "unsupported_schema")
        _, _, water, _ = inputs()
        with self.assertRaises(dense_runner.DenseRunnerError):
            drun._water_parameter_sources(np, water.column, dataclasses.replace(water.sediment, xp=object()),
                                          (fx.NY, fx.NX), NC)
        with self.assertRaises(dense_runner.DenseRunnerError):
            drun._water_parameter_sources(np, water.column, water.sediment, (fx.NY, fx.NX), NC - 1)


class ClosedStaticMetadata(unittest.TestCase):
    def test_records_bind_scalars_models_signed_zero_and_refuse_extras(self):
        _, _, water, _ = inputs()
        record = dres._closed_metadata(water.sediment, (np.ndarray,), (np,))[0]
        positive = dres._closed_metadata(dataclasses.replace(water.sediment, sigma=0.0), (np.ndarray,), (np,))[0]
        negative = dres._closed_metadata(dataclasses.replace(water.sediment, sigma=-0.0), (np.ndarray,), (np,))[0]
        self.assertNotEqual(positive, negative)
        self.assertNotEqual(record, positive)
        other_model = dataclasses.replace(water.column, model="other")
        self.assertNotEqual(dres._closed_metadata(water.column, (np.ndarray,), (np,))[0],
                            dres._closed_metadata(other_model, (np.ndarray,), (np,))[0])
        with self.assertRaises(dres.DeviceResourceError):
            dres._closed_metadata(water.column, (np.ndarray,), ())  # undeclared xp module
        object.__setattr__(water.column, "hidden", 1)
        with self.assertRaises(dres.DeviceResourceError) as caught:
            dres._closed_metadata(water.column, (np.ndarray,), (np,))
        self.assertEqual(caught.exception.code, "unsupported_schema")

    def test_owned_static_scalar_model_and_raw_signed_zero_changes_are_detected(self):
        for mutate, code in (
                (lambda r: object.__setattr__(r.static("sediment"), "particle_density_kg_m3", 2651.0),
                 "static_changed"),
                (lambda r: object.__setattr__(r.static("column"), "model", "other"), "static_changed"),
                (lambda r: object.__setattr__(r.static("column"), "xp", None), "static_changed"),
                (lambda r: writable(r.static("column").suction_m).__setitem__((0, 0), -0.0), "integrity_failed"),
                (lambda r: writable(r.static("initial_available_fraction")).__setitem__((0, 0, 0), 0.5),
                 "integrity_failed")):
            with self.subTest(code=code):
                env = owners()
                env.resources.verify_static()
                mutate(env.resources)
                with self.assertRaises(dres.DeviceResourceError) as caught:
                    env.resources.verify_static()
                self.assertEqual(caught.exception.code, code)


class DeviceContextBridge(unittest.TestCase):
    def test_context_is_native_replace_bound_to_the_owned_availability(self):
        env = owners()
        _, metadata = ownership(env)
        n = dense_bed._native()
        host, device = metadata.host_context, metadata.context
        self.assertIs(host, env.bed.context)
        self.assertIs(type(device), n.BedContext)
        self.assertIsNot(device, host)
        for field in dataclasses.fields(n.BedContext):
            if field.name != "initial_available_fraction":
                self.assertIs(getattr(device, field.name), getattr(host, field.name), field.name)
        avail = env.resources.static("initial_available_fraction")
        self.assertIs(device.initial_available_fraction, avail)
        self.assertIsNot(avail, host.initial_available_fraction)
        self.assertEqual(avail.tobytes(), host.initial_available_fraction.tobytes())
        wind, _ = device_inputs(env)
        self.assertIs(wind.config.initial_available_fraction, device.initial_available_fraction)
        metadata.verify(n)

    def test_device_context_drift_is_refused(self):
        n = dense_bed._native()
        for mutate in (
                lambda c, a: object.__setattr__(c, "initial_available_fraction", a.copy()),
                lambda c, a: object.__setattr__(c, "depth_update_rule", "other"),
                lambda c, a: object.__setattr__(c, "geometry", dataclasses.replace(c.geometry)),
                lambda c, a: object.__setattr__(c, "hidden", 1)):
            env = owners()
            _, metadata = ownership(env)
            mutate(metadata.context, env.resources.static("initial_available_fraction"))
            with self.assertRaises(dense_bed.DenseBedError) as caught:
                metadata.verify(n)
            self.assertEqual(caught.exception.code, "case_input_mutated")

    def test_runner_native_namespace_serves_the_build_metadata_verify(self):
        # `_build` passes `dense_runner._native()` (not `dense_bed._native()`) to `metadata.verify`.
        n = dense_runner._native()
        env = owners()
        _, metadata = ownership(env)
        metadata.verify(n)
        self.assertIs(n.BedContext, dense_bed._native().BedContext)
        self.assertIs(type(metadata.context), n.BedContext)
        object.__setattr__(metadata.context, "depth_update_rule", "other")
        with self.assertRaises(dense_bed.DenseBedError) as caught:
            metadata.verify(n)
        self.assertEqual(caught.exception.code, "case_input_mutated")

    def test_host_context_object_replacement_is_refused(self):
        env = owners()
        _, metadata = ownership(env)
        object.__setattr__(metadata.host_context, "grain_classes", dataclasses.replace(
            metadata.host_context.grain_classes))
        with self.assertRaises(dense_bed.DenseBedError):
            metadata.verify(dense_bed._native())

    def test_shadow_inputs_are_bound_to_host_and_owned_statics(self):
        env = owners()
        wind, water = device_inputs(env)
        drun._bind_device_inputs(np, np, env.wind, env.water, wind, water, env.resources)
        diverged = dense_runner.DenseWindInputs(dataclasses.replace(
            wind.config, initial_available_fraction=wind.config.initial_available_fraction.copy()), None)
        with self.assertRaises(dense_runner.DenseRunnerError) as caught:
            drun._bind_device_inputs(np, np, env.wind, env.water, diverged, water, env.resources)
        self.assertEqual(caught.exception.code, "device_config_diverged")
        object.__setattr__(env.resources.static("column"), "model", "other")
        with self.assertRaises(dense_runner.DenseRunnerError):
            drun._bind_device_inputs(np, np, env.wind, env.water, wind, water, env.resources)


class CompactReads(unittest.TestCase):
    def test_class_vectors_scalars_and_signed_zero_bytes(self):
        ops = drun._DeviceArrayOps(np, NC)
        vector = ops.class_vector(np.array([-0.0] + [1.0] * (NC - 1)))
        self.assertTrue(np.signbit(vector[0]))
        self.assertEqual(ops.scalar(np.array(-0.0)).hex(), (-0.0).hex())
        self.assertEqual(ops.scalar(np.array([2.5])), 2.5)  # native length-1 tolerance shape
        for spatial in (np.zeros((fx.NY, fx.NX)), np.zeros(NC + 1)):
            with self.assertRaises(dense_runner.DenseRunnerError) as caught:
                ops.class_vector(spatial)
            self.assertEqual(caught.exception.code, "spatial_payload")
        with self.assertRaises(dense_runner.DenseRunnerError):
            ops.scalar(np.zeros(2))
        with self.assertRaises(dense_runner.DenseRunnerError):
            ops.class_vector("not a value")


    def test_native_host_lists_keep_values_order_and_signed_zero_without_device_read(self):
        ops = drun._DeviceArrayOps(np, NC)
        ops._packet = mock.Mock(side_effect=AssertionError("host list used device packet"))
        expected = np.array([-0.0, 0.0, 1.25, 2.5, 3.75, 4.0], dtype=np.float64)
        self.assertEqual(expected.size, NC)
        for values in (expected.tolist(), [-0.0, 0, 1.25, 2.5, 3.75, 4]):
            with self.subTest(values_type="native .tolist or mixed plain numbers"):
                got = ops.class_vector(values)
                self.assertEqual(got.shape, (NC,))
                self.assertEqual(got.dtype, np.dtype(np.float64))
                self.assertEqual(got.tobytes(), expected.tobytes())
                self.assertEqual([v.hex() for v in ops.float_values(values)],
                                 [float(v).hex() for v in expected])
        self.assertEqual(ops.scalar([-0.0]).hex(), (-0.0).hex())
        self.assertEqual(ops.class_vector([]).shape, (0,))
        ops._packet.assert_not_called()

    def test_host_list_size_refuses_before_inspection_or_allocation(self):
        ops = drun._DeviceArrayOps(np, NC)
        ops._packet = mock.Mock(side_effect=AssertionError("host list used device packet"))
        with mock.patch.object(np, "asarray", side_effect=AssertionError("premature allocation")) as convert:
            with self.assertRaises(dense_runner.DenseRunnerError) as caught:
                ops.class_vector([object()] * (NC + 1))
            self.assertEqual(caught.exception.code, "spatial_payload")
            convert.assert_not_called()
        ops._packet.assert_not_called()

    def test_non_plain_host_list_items_and_sequences_refuse_before_allocation(self):
        class ListSubclass(list):
            pass

        ops = drun._DeviceArrayOps(np, NC)
        ops._packet = mock.Mock(side_effect=AssertionError("host list used device packet"))
        values = ([[1.0]], [True], [np.float64(1)], [np.int64(1)], [np.array(1.0)],
                  [object()], ListSubclass([1.0]), (1.0,))
        with mock.patch.object(np, "asarray", side_effect=AssertionError("premature allocation")) as convert:
            for value in values:
                with self.subTest(value_type=type(value).__name__):
                    with self.assertRaises(dense_runner.DenseRunnerError) as caught:
                        ops.class_vector(value)
                    self.assertEqual(caught.exception.code, "unsupported_value")
            convert.assert_not_called()
        ops._packet.assert_not_called()



class TypedPrivateTransactions(unittest.TestCase):
    def test_lifecycle_disjointness_and_ticket_refusals(self):
        env, other = owners(), owners()
        res = env.resources
        stage = res.begin_stage()
        with self.assertRaises(dres.DeviceResourceError) as caught:
            res.begin_stage()
        self.assertEqual(caught.exception.code, "transaction_active")
        with self.assertRaises(dres.DeviceResourceError) as caught:
            other.resources.work_soil(stage)
        self.assertEqual(caught.exception.code, "foreign_ticket")
        soil, pool = res.work_soil(stage), res.work_pool(stage)
        self.assertIs(res.work_soil(stage), soil)  # at most one work copy per stage
        for accepted in (res._accepted.soil, res._accepted.soil_mirror):
            self.assertFalse(np.shares_memory(soil, accepted))
        self.assertFalse(np.shares_memory(pool.mass_kg, res._accepted.pool.mass_kg))
        with self.assertRaises(dres.DeviceResourceError) as caught:
            res.prepare_publish(stage, soil=res._accepted.soil)
        self.assertEqual(caught.exception.code, "private_alias")
        soil[0, 0] = -0.0  # native-admissible signed zero is preserved, not refused
        prepared = res.prepare_publish(stage, soil=soil)
        with self.assertRaises(dres.DeviceResourceError) as caught:
            res.prepare_publish(stage, soil=soil)
        self.assertEqual(caught.exception.code, "stale_ticket")
        res._check_publish(prepared)
        res._swap(prepared)
        self.assertEqual(res.generation, 1)
        self.assertTrue(np.signbit(res._accepted.soil[0, 0]))
        res.verify()
        with self.assertRaises(dres.DeviceResourceError) as caught:
            res.discard(prepared)
        self.assertEqual(caught.exception.code, "ticket_consumed")

    def test_negative_or_nonfinite_soil_candidate_refuses_without_publication(self):
        env = owners()
        res = env.resources
        for bad in (-1.0, float("nan")):
            stage = res.begin_stage()
            soil = res.work_soil(stage)
            soil[0, 0] = bad
            with self.assertRaises(dres.DeviceResourceError):
                res.prepare_publish(stage, soil=soil)
            res.discard(stage)
            self.assertEqual(res.generation, 0)
            res.verify()


class DualOwnerPublication(unittest.TestCase):
    def publish_once(self, own, env, dt=1.0):
        stage = own.begin()
        soil = own.work_soil(np, None)
        t = env.owner.time_s + dt
        prepared = own.prepare_bed(own.stage_state(stage), t, soil=soil)
        predicted = own.candidate_label(own.stage_state(stage), t)
        before = own.stage_label(stage)
        own.retained_private(None, None)
        own.commit(prepared)
        return before, predicted

    def test_predicted_label_equals_accepted_and_second_event_continues(self):
        env = owners()
        own, _ = ownership(env)
        first_before = own.accepted_label()
        before, predicted = self.publish_once(own, env)
        self.assertEqual(before, first_before)
        self.assertEqual(own.accepted_label(), predicted)
        self.assertEqual((env.owner.generation, env.resources.generation), (1, 1))
        stage = own.begin()  # the engine's _finish compares accepted_label with this stage label
        self.assertEqual(own.stage_label(stage), own.accepted_label())
        own.abandon()
        second_before, second = self.publish_once(own, env)
        self.assertEqual(second_before, predicted)
        self.assertEqual(own.accepted_label(), second)
        self.assertNotIn("sha256", second)
        self.assertIn("mirror-verified", second)

    def test_failed_event_does_not_advance_labels_or_generations(self):
        env = owners()
        own, _ = ownership(env)
        label = own.accepted_label()
        stage = own.begin()
        own.prepare_bed(own.stage_state(stage), env.owner.time_s + 1.0, soil=own.work_soil(np, None))
        own.abandon()
        self.assertEqual(own.accepted_label(), label)
        self.assertEqual((env.owner.generation, env.resources.generation), (0, 0))
        self.assertFalse(env.owner._active is not None or env.resources.active)
        self.assertEqual(own.cleanup_failures, 0)

    def test_publication_checks_both_owners_before_any_assignment(self):
        env = owners()
        own, _ = ownership(env)
        stage = own.begin()
        prepared = own.prepare_bed(own.stage_state(stage), env.owner.time_s + 1.0,
                                   soil=own.work_soil(np, None))
        bed_record, res_record = env.owner._accepted, env.resources._accepted
        writable(res_record.soil_mirror)[0, 0] = 0.75  # corrupt the SECOND owner's accepted mirror
        with self.assertRaises(dense_bed.DenseBedError) as caught:
            own.commit(prepared)
        self.assertEqual(caught.exception.code, "integrity_failed")
        self.assertIs(env.owner._accepted, bed_record)  # first owner NOT swapped
        self.assertIs(env.resources._accepted, res_record)
        own.abandon()
        self.assertEqual((env.owner.generation, env.resources.generation), (0, 0))

    def test_foreign_or_stale_publication_refused(self):
        env = owners()
        own, _ = ownership(env)
        with self.assertRaises(TypeError):
            own.commit(object())
        with self.assertRaises(dense_bed.DenseBedError):
            own.work_soil(np, None)  # no active stage

    def test_private_labels_distinguish_stale_and_foreign_references(self):
        env = owners()
        own, _ = ownership(env)
        ref = drun._PrivateRef(env.resources.generation)
        self.assertEqual(own.private_label(np, ref, ref), "device-private:mirror-verified:generation=0")
        self.assertEqual(own.private_label(np, ref, drun._PrivateRef(0)), "device-private:not-an-owner-reference")
        self.assertEqual(own.private_label(np, env.wind.initial_mobile_state, np.zeros(1)),
                         "device-private:not-an-owner-reference")
        stale = drun._PrivateRef(7)
        self.assertEqual(own.private_label(np, stale, stale), "device-private:stale-generation")


def routing_seed(env, terrain=None):
    """The engine's first-rain routing construction (`DenseRunner._rain_event`) with the actual native SYRUP calls
    over the staged owned bed, the owned-availability context and the owner work soil. No storm step is run."""
    n = dense_runner._native()
    own, metadata = ownership(env)
    _, water = device_inputs(env, terrain)
    stage = own.begin()
    bed, soil, t = own.stage_state(stage), own.work_soil(np, None), env.owner.time_s
    graph = n.routing_graph_from_elevation(water.terrain, n.committed_elevation_m(bed.committed_topography), xp=np)
    seed = n.initial_event_state(graph, water.terrain, bed, metadata.context, soil, t_s=t,
                                 control=env.control.sediment_control)
    return SimpleNamespace(n=n, own=own, ctx=metadata.context, water=water, bed=bed, soil=soil, t=t, seed=seed,
                           held=own.retain_routing(seed), shape=(fx.NY, fx.NX))


def staged_seed(n, own, ctx, water, env):
    """One more event of the SAME adapter: a new owner stage and the actual native first-rain seed over it."""
    stage = own.begin()
    bed, soil, t = own.stage_state(stage), own.work_soil(np, None), env.owner.time_s
    graph = n.routing_graph_from_elevation(water.terrain, n.committed_elevation_m(bed.committed_topography), xp=np)
    seed = n.initial_event_state(graph, water.terrain, bed, ctx, soil, t_s=t, control=env.control.sediment_control)
    return SimpleNamespace(stage=stage, bed=bed, soil=soil, t=t, seed=seed)


def assert_same_native(test, actual, expected, where):
    if type(expected) is np.ndarray:
        test.assertIs(type(actual), np.ndarray, where)
        test.assertEqual((actual.dtype, actual.shape), (expected.dtype, expected.shape), where)
        test.assertEqual(actual.tobytes(), expected.tobytes(), where)  # raw bytes: signed zero included
    elif type(expected) is float:
        test.assertEqual(actual.hex(), expected.hex(), where)
    else:
        test.assertEqual(actual, expected, where)


class RetainedDeviceRouting(unittest.TestCase):
    """The engine's three closed retained-routing calls over the device namespace (NumPy semantic seam)."""

    def seeded(self):
        r = routing_seed(owners())
        self.addCleanup(r.own.abandon)
        return r

    def test_memory_guard_and_entry_reconstruct_the_native_event(self):
        r = self.seeded()
        n, own, held, seed = r.n, r.own, r.held, r.seed
        self.assertEqual(set(vars(held)), drun._HELD_ROUTING_FIELDS)
        for name, field in (("graph", "graph"), ("terrain", "terrain"), ("network", "network"), ("grid", "grid"),
                            ("source", "routing_source_elevation_m")):
            self.assertIs(getattr(held, name), getattr(seed, field), name)
        self.assertIs(held.terrain, r.water.terrain)
        self.assertIsNone(held.phase)
        self.assertEqual(held.generation, 1)
        self.assertFalse(np.shares_memory(held.velocity, seed.sediment_velocity_m_s))
        assert_same_native(self, held.velocity, np.zeros_like(seed.sediment_velocity_m_s), "held velocity")
        self.assertEqual(own.routing_label(held), "device-routing:held-source:generation=1")

        names, nbytes = own.routing_memory(np, held, r.water, r.shape, NC)
        self.assertEqual(names, ("sediment_velocity_m_s",))
        self.assertEqual(nbytes, fx.NY * fx.NX * NC * 8)
        self.assertIsNone(own.routing_guard(np, held, r.water, r.shape, NC))

        boundary = n.commit_bed(r.bed, r.ctx, held.graph, held.terrain, r.t, force=True, routing_source_m=held.source)
        graph = boundary.graph
        if boundary.rerouted:
            network, grid = n.transport_network(graph), n.physics_grid_from_graph(graph)
        else:
            network, grid = held.network, held.grid
        entry = own.routing_entry(n, held, boundary, graph, network, grid, r.soil, r.t, names, water=r.water,
                                  shape=r.shape, nc=NC)
        # The CPU template path's exact reconstruction (`_HostOwnership.routing_entry`) over the same native objects.
        expected = dataclasses.replace(
            seed, bed=boundary.state, graph=graph, network=network, grid=grid, terrain=held.terrain,
            routing_source_elevation_m=boundary.routing_source_m, phase=None,
            storm=n.initial_state(graph, boundary.state.water.depth_m, r.soil, t_s=r.t))
        self.assertIs(type(entry), type(expected))
        for name in ("bed", "graph", "network", "grid", "terrain", "routing_source_elevation_m"):
            self.assertIs(getattr(entry, name), getattr(expected, name), name)
        self.assertIsNone(entry.phase)
        self.assertIs(type(entry.storm), type(expected.storm))
        for field in dataclasses.fields(expected.storm):
            assert_same_native(self, getattr(entry.storm, field.name), getattr(expected.storm, field.name),
                               f"storm.{field.name}")
        assert_same_native(self, entry.sediment_velocity_m_s, expected.sediment_velocity_m_s, "velocity")
        # Work copy, never the held memory: a native write to the entry cannot reach the retained namespace.
        self.assertFalse(np.shares_memory(entry.sediment_velocity_m_s, held.velocity))
        entry.sediment_velocity_m_s[...] = 1.0
        self.assertFalse(held.velocity.any())
        self.assertIsNone(own.routing_guard(np, held, r.water, r.shape, NC))
        self.assertIs(held.graph, seed.graph)
        self.assertIs(held.source, seed.routing_source_elevation_m)
        self.assertEqual(held.generation, own._routing_generation)
        # First rain (no retained routing in the engine): no memory names, the per-event held zero is handed over.
        first = own.routing_entry(n, held, boundary, graph, network, grid, r.soil, r.t, (), water=r.water,
                                  shape=r.shape, nc=NC)
        self.assertIs(first.sediment_velocity_m_s, held.velocity)

    def refused(self, call, code, guarantee):
        with self.assertRaises(dense_runner._Stop) as caught:
            call()
        self.assertEqual((caught.exception.status, caught.exception.code), ("refused", code))
        self.assertEqual(caught.exception.guarantee, guarantee)

    def test_foreign_or_malformed_namespace_refuses_before_copy_or_native_entry(self):
        r = self.seeded()
        held = r.held

        def variant(**changes):
            return SimpleNamespace(**{**vars(held), **changes})

        extra = variant()
        extra.template = r.seed
        fresh, _ = ownership(owners())  # an owner that never retained routing: generation 1 is not its own
        cases = {
            "cpu_template_namespace": (r.own, dense_runner._HostOwnership(None).retain_routing(r.seed)),
            "extra_attribute": (r.own, extra),
            "future_generation": (r.own, variant(generation=held.generation + 1)),
            "zero_generation": (r.own, variant(generation=0)),
            "bool_generation": (r.own, variant(generation=True)),
            "foreign_terrain": (r.own, variant(terrain=copy.copy(held.terrain))),
            "foreign_owner": (fresh, held),
            "plain_object": (r.own, object()),
            "missing_stamp": (r.own, SimpleNamespace(**{k: v for k, v in vars(held).items() if k != "stamp"})),
            "foreign_stamp": (r.own, variant(stamp=object())),
            "adapter_as_stamp": (r.own, variant(stamp=r.own)),
        }
        for label, (own, bad) in cases.items():
            native = mock.Mock()
            calls = (lambda: own.routing_memory(np, bad, r.water, r.shape, NC),
                     lambda: own.routing_guard(np, bad, r.water, r.shape, NC),
                     lambda: own.routing_entry(native, bad, native.boundary, native.graph, native.network,
                                               native.grid, r.soil, r.t, ("sediment_velocity_m_s",),
                                               water=r.water, shape=r.shape, nc=NC))
            with self.subTest(case=label), mock.patch.object(own, "work_memory",
                                                             side_effect=AssertionError("copied")) as work:
                for call in calls:
                    self.refused(call, "restored_routing_layout_changed", dense_runner._EXTERNAL)
                work.assert_not_called()
                self.assertEqual(native.mock_calls, [])
        self.assertFalse(held.velocity.any())

    def refused_before_copy(self, own, bad, water, soil, t, shape):
        """All three hooks refuse `bad` before any work copy, event-state construction or native call."""
        native = mock.Mock()
        calls = (lambda: own.routing_memory(np, bad, water, shape, NC),
                 lambda: own.routing_guard(np, bad, water, shape, NC),
                 lambda: own.routing_entry(native, bad, native.boundary, native.graph, native.network, native.grid,
                                           soil, t, ("sediment_velocity_m_s",), water=water, shape=shape, nc=NC))
        with (mock.patch.object(own, "work_memory", side_effect=AssertionError("copied")) as work,
              mock.patch.object(own, "entry_state", side_effect=AssertionError("entered")) as entry):
            for call in calls:
                self.refused(call, "restored_routing_layout_changed", dense_runner._EXTERNAL)
            work.assert_not_called()
            entry.assert_not_called()
        self.assertEqual(native.mock_calls, [])

    def test_same_generation_same_terrain_foreign_adapter_refuses_real_held_namespace(self):
        r = self.seeded()
        other = routing_seed(owners(), terrain=r.water.terrain)  # a second real adapter on the SAME terrain
        self.addCleanup(other.own.abandon)
        # Every pre-stamp check passes across adapters: identical layout, same issued generation, same terrain.
        self.assertEqual(set(vars(other.held)), set(vars(r.held)))
        self.assertEqual((r.held.generation, other.held.generation), (1, 1))
        self.assertEqual(other.own._routing_generation, r.own._routing_generation)
        self.assertIs(other.water.terrain, r.water.terrain)
        self.assertIs(other.held.terrain, r.held.terrain)
        # The stamp is a bare opaque object: no adapter, owner, metadata or native bed is held through it.
        self.assertIs(type(r.held.stamp), object)
        self.assertIsNot(r.held.stamp, other.held.stamp)
        forbidden = (r.own, other.own, r.own._owner, r.own._res, r.own._metadata, r.bed)
        self.assertFalse([k for k, v in vars(r.held).items() if any(v is f for f in forbidden)])
        for own, mine, foreign, run in ((r.own, r.held, other.held, r), (other.own, other.held, r.held, other)):
            self.assertIsNone(own.routing_guard(np, mine, r.water, r.shape, NC))  # own namespace: accepted
            self.refused_before_copy(own, foreign, r.water, run.soil, run.t, r.shape)
        self.assertFalse(r.held.velocity.any() or other.held.velocity.any())

    def test_accepted_older_routing_survives_a_later_abandoned_preparation(self):
        env = owners()
        n = dense_runner._native()
        own, metadata = ownership(env)
        self.addCleanup(own.abandon)
        _, water = device_inputs(env)
        # Event 1 (engine order): prepare the bed publication, retain routing, publish.
        first = staged_seed(n, own, metadata.context, water, env)
        prepared = own.prepare_bed(first.bed, first.t + 1.0, soil=first.soil)
        accepted = own.retain_routing(first.seed)
        own.commit(prepared)
        self.assertEqual((accepted.generation, env.owner.generation, env.resources.generation), (1, 1, 1))
        # Event 2: prepared and a NEWER routing issued, then an ordinary failure before publication.
        second = staged_seed(n, own, metadata.context, water, env)
        own.prepare_bed(second.bed, second.t + 1.0, soil=second.soil)
        abandoned = own.retain_routing(second.seed)
        own.abandon()
        self.assertEqual(abandoned.generation, 2)
        self.assertEqual(own._routing_generation, 2)  # the issue counter is NOT rolled back by abandon
        self.assertEqual((env.owner.generation, env.resources.generation), (1, 1))
        self.assertFalse(env.owner._active is not None or env.resources.active)
        self.assertEqual(own.cleanup_failures, 0)
        self.assertLess(accepted.generation, own._routing_generation)  # older than the latest ISSUED routing
        # Event 3 retries from the still-accepted boundary with the older held routing: all three hooks accept it.
        third = staged_seed(n, own, metadata.context, water, env)
        shape = (fx.NY, fx.NX)
        names, nbytes = own.routing_memory(np, accepted, water, shape, NC)
        self.assertEqual((names, nbytes), (("sediment_velocity_m_s",), fx.NY * fx.NX * NC * 8))
        self.assertIsNone(own.routing_guard(np, accepted, water, shape, NC))
        boundary = n.commit_bed(third.bed, metadata.context, accepted.graph, accepted.terrain, third.t, force=True,
                                routing_source_m=accepted.source)
        graph = boundary.graph
        if boundary.rerouted:
            network, grid = n.transport_network(graph), n.physics_grid_from_graph(graph)
        else:
            network, grid = accepted.network, accepted.grid
        entry = own.routing_entry(n, accepted, boundary, graph, network, grid, third.soil, third.t, names,
                                  water=water, shape=shape, nc=NC)
        self.assertIs(type(entry), type(third.seed))
        self.assertIs(entry.terrain, accepted.terrain)
        self.assertFalse(np.shares_memory(entry.sediment_velocity_m_s, accepted.velocity))
        # The next stage stays usable: it publishes and issues the next routing generation.
        prepared = own.prepare_bed(third.bed, third.t + 1.0, soil=third.soil)
        newest = own.retain_routing(entry)
        own.commit(prepared)
        self.assertEqual((newest.generation, env.owner.generation, env.resources.generation), (3, 2, 2))
        self.assertIsNone(own.routing_guard(np, newest, water, shape, NC))
        # A foreign adapter on the same terrain whose issue counter covers generation 1 still refuses it.
        other = routing_seed(owners(), terrain=water.terrain)
        self.addCleanup(other.own.abandon)
        other.own.retain_routing(other.seed)
        self.assertGreaterEqual(other.own._routing_generation, accepted.generation)
        self.assertIs(other.held.terrain, accepted.terrain)
        self.refused_before_copy(other.own, accepted, water, other.soil, other.t, shape)
        self.assertFalse(accepted.velocity.any())

    def test_wrong_velocity_holder_refuses_before_work_copy(self):
        r = self.seeded()
        for label, velocity in (("missing_class", np.zeros((fx.NY, fx.NX, NC - 1))),
                                ("flattened_cells", np.zeros((fx.NY * fx.NX, NC))),
                                ("float32", np.zeros((fx.NY, fx.NX, NC), dtype=np.float32)),
                                ("not_an_array", [[[0.0] * NC] * fx.NX] * fx.NY)):
            bad = SimpleNamespace(**{**vars(r.held), "velocity": velocity})
            with self.subTest(case=label), mock.patch.object(r.own, "work_memory",
                                                             side_effect=AssertionError("copied")) as work:
                self.refused(lambda: r.own.routing_memory(np, bad, r.water, r.shape, NC),
                             "transport_memory_unsupported", dense_runner._UNCHANGED)
                work.assert_not_called()

    def test_nonempty_phase_or_velocity_blocks_entry(self):
        r = self.seeded()
        host = dense_runner._HostOwnership(None)
        phased = SimpleNamespace(**{**vars(r.held), "phase": object()})
        self.refused(lambda: r.own.routing_guard(np, phased, r.water, r.shape, NC), "transport_memory_not_empty",
                     dense_runner._UNCHANGED)
        for label, value, blocked in (("positive", 5e-324, True), ("negative", -1.0, True),
                                      ("signed_zero", -0.0, False)):
            velocity = np.zeros_like(r.held.velocity)
            velocity[0, 0, NC - 1] = value
            device = SimpleNamespace(**{**vars(r.held), "velocity": velocity})
            # The CPU template guard over the genuine native event type with the same velocity bytes.
            template = SimpleNamespace(template=dataclasses.replace(r.seed, sediment_velocity_m_s=velocity.copy()))
            with self.subTest(case=label):
                for own, held in ((r.own, device), (host, template)):
                    if blocked:
                        self.refused(lambda: own.routing_guard(np, held, r.water, r.shape, NC),
                                     "transport_memory_not_empty", dense_runner._UNCHANGED)
                    else:
                        self.assertIsNone(own.routing_guard(np, held, r.water, r.shape, NC))


class BoundedEvidenceAndRelease(unittest.TestCase):
    def test_summary_is_bounded_scalar_metadata(self):
        self.assertEqual(drun._bounded_summary(["log"]), {"summary_type": "list"})
        record = drun._bounded_summary({"mode": "fused", "n": 3, "t": float("inf"), "log": [1, 2],
                                        "nested": {"a": 1.5, "obj": object()}})
        self.assertEqual(record, {"mode": "fused", "n": 3, "t": "inf", "nested": {"a": 1.5}})
        self.assertEqual(drun._bounded_summary({"x": "y" * (drun._SUMMARY_CHARS + 1)}),
                         {"summary_truncated": True})

    def runner(self, env):
        own, _ = ownership(env)
        runner = object.__new__(drun._DeviceDenseRunner)
        runner._device_own, runner._hydraulics, runner._hydraulics_summary = own, None, None
        return runner, own

    def test_rain_lifecycle_references_are_dropped_after_summary_or_failure(self):
        class Lifecycle:
            def __init__(self, fail):
                self.fail, self.context = fail, np.zeros(1024)

            def summary(self):
                if self.fail:
                    raise RuntimeError("native summary failure")
                return {"mode": "split", "log": ["x"] * 10}

        for fail in (False, True):
            runner, _ = self.runner(owners())
            lifecycle = Lifecycle(fail)
            ref = weakref.ref(lifecycle)
            runner._hydraulics, lifecycle = lifecycle, None
            runner._rain_native_release()
            gc.collect()
            self.assertIsNone(ref())
            self.assertIsNone(runner._hydraulics)
            expected = {"summary_error": "RuntimeError"} if fail else {"mode": "split"}
            self.assertEqual(runner._hydraulics_summary, expected)

    def test_stop_evidence_retains_no_native_tree_or_exception(self):
        class NativeResult:
            def __init__(self):
                self.candidate = np.zeros(4096)

        class NativeFault(RuntimeError):
            pass  # Exception subclasses support weakrefs; the builtin RuntimeError does not.

        runner, _ = self.runner(owners())
        evidence, exc = NativeResult(), NativeFault("native")
        refs = (weakref.ref(evidence), weakref.ref(exc))
        stop = dense_runner._Stop("incomplete", "rain_not_dry", "d", exception=exc, elapsed="0x1p+0",
                                  evidence=evidence)
        kept = runner._stop_evidence(stop)
        self.assertEqual(kept["evidence_type"], "NativeResult")
        self.assertFalse(kept["evidence_retained"])
        self.assertEqual(kept["exception_type"], "NativeFault")
        self.assertEqual(kept["native_unaccepted_elapsed_s"], "0x1p+0")
        del stop, evidence, exc
        gc.collect()
        self.assertTrue(all(r() is None for r in refs))

    def test_ordinary_failure_cleanup_releases_stage_and_candidate_copies(self):
        env = owners()
        runner, own = self.runner(env)
        stage = own.begin()
        soil = own.work_soil(np, None)
        prepared = own.prepare_bed(own.stage_state(stage), env.owner.time_s + 1.0, soil=soil)
        refs = [weakref.ref(soil), weakref.ref(prepared), weakref.ref(stage),
                weakref.ref(prepared.resources._accepted.soil), weakref.ref(own.stage_state(stage).water.depth_m)]
        del stage, soil, prepared
        runner._abandon()
        gc.collect()
        self.assertEqual([r() for r in refs], [None] * len(refs))
        self.assertFalse(env.owner._active is not None or env.resources.active)
        own.verify_case_inputs()


class WorkspaceMapping(unittest.TestCase):
    def test_named_terms_follow_traced_native_shapes_and_allowances_are_added(self):
        bed, wind, _, _ = inputs()
        limits = device_limits(wind_workspace_bytes=11, routing_workspace_bytes=13)
        soil = np.zeros((fx.NY, fx.NX))
        mapping = drun._workspace_mapping(None, bed, wind, soil, {"required_bytes": 5}, limits)
        known, cells = mapping["named_device_bytes"], fx.NY * fx.NX
        self.assertEqual(known["routing_held_zero_velocity_bytes"], 2 * cells * NC * 8)  # (ny, nx, nc) float64
        self.assertEqual(known["routing_held_source_elevation_bytes"], 2 * cells * 8)  # (ny, nx), no ring
        self.assertEqual(mapping["declared_total_bytes"], 24)
        self.assertEqual(mapping["named_total_bytes"], sum(known.values()))
        positive = {k: v for k, v in known.items() if k != "finalizer_bed_candidate_bytes"}
        self.assertTrue(all(type(v) is int and v > 0 for v in positive.values()), positive)
        self.assertFalse(mapping["native_flat_estimate"]["gated_by_preflight"])
        self.assertIn("NOT bounded", mapping["native_coefficients"]["scope"])


class DormantGate(unittest.TestCase):
    def test_gated_file_has_no_import_side_effect_and_refuses_without_scope(self):
        path = Path(__file__).resolve().parent / "gpu/test_dense_device_runner_cupy.py"
        spec = importlib.util.spec_from_file_location("device_runner_gate_source", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(RuntimeError):
            module.load_tests(None, None, None)
        from masy.runtime_worker import WorkerRefusal

        with self.assertRaises(WorkerRefusal):
            module._scope({}, {}, drun.__file__)
        host = module._host_inputs()
        self.assertEqual(len(host["bed"].context.grain_classes.classes), NC)
        self.assertEqual(host["control"].sediment_control.storm.implementation, "cuda")
        self.assertFalse(any(k == "cupy" or k.startswith("cupy.") for k in sys.modules if sys.modules[k]))


if __name__ == "__main__":
    unittest.main()
