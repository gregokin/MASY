"""CPU semantic tests using exact native holders/validators, never a GPU substitute.

Run explicitly with the task source-origin harness. The public constructor still
requires actual CuPy; only private _create receives this NumPy testing adapter.
No solver/event/transport/Numba kernel or accelerator import is executed.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from maple.core.backend import array_leaves, arrays_share_memory, read_transfer_counters
from maple.core.backend.arrays import array_leaf_inventory
from maple_syrup.event_backend import host_packet

from masy import backend, dense_bed
from masy import dense_device_bed as d

_NATIVE_TESTS = Path(__file__).resolve().parent / "native"
sys.path.insert(0, str(_NATIVE_TESTS))
from check_dense_bed import fixture


class CpuOps:
    array_type = np.ndarray
    xp = np

    def __init__(self):
        self.device_id, self.stream_id = 0, 0
        self.wrong_leaf, self.clone_calls, self.packet_calls = None, 0, 0
        self.fail_clone, self.fail_packet = None, None

    def current_device(self):
        return self.device_id

    def current_stream(self):
        return self.stream_id

    def device(self, array):
        return 1 if array is self.wrong_leaf else 0

    def clone(self, array):
        self.clone_calls += 1
        if self.clone_calls == self.fail_clone:
            raise MemoryError("labelled injected D2D-copy allocation failure")
        return array.copy(order="C")

    def overlap(self, a, b):
        return arrays_share_memory(a, b)

    def raw_equal(self, a, b):
        return np.all(a.reshape(-1).view(np.uint8) == b.reshape(-1).view(np.uint8))

    def packet(self, flags):
        self.packet_calls += 1
        if self.packet_calls == self.fail_packet:
            raise RuntimeError("labelled injected compact packet failure")
        if len(flags) > 69:
            raise AssertionError("unbounded comparison packet")
        return host_packet(np, flags)

    def counters(self):
        return dataclasses.asdict(read_transfer_counters())


def create(state=None, *, ops=None, limits=None, **kwargs):
    inputs = fixture(2)
    bed = dense_bed.build_dense_bed(**inputs)
    state = bed.state if state is None else state
    ops = CpuOps() if ops is None else ops
    args = {
        "geometry": inputs["geometry"],
        "mass_resolution_kg": inputs["mass_resolution_kg"],
        "n_voxels": inputs["n_voxels"],
        "n_classes": 2,
        "case_sha256": bed.case_sha256,
        "time_s": bed.time_s,
        "limits": limits or d.DeviceBedLimits(1 << 24, 1 << 29, 1 << 20, 1 << 20),
    }
    args.update(kwargs)
    return d.DeviceDenseBedOwner._create(state, _ops=ops, **args), bed, ops


def exact_record(owner):
    # Test-only internal observation, not a production accepted-state accessor.
    record = owner._accepted
    return (
        record,
        owner.time_s.hex(),
        owner.generation,
        tuple(a.tobytes() for a in array_leaves(record.state)),
    )


class OwnershipChecks(unittest.TestCase):
    def test_public_constructor_refuses_numpy_and_fake_module_without_cupy(self):
        _, bed, _ = create()
        with mock.patch.dict(sys.modules, {"cupy": None}):
            with self.assertRaises(d.DeviceBedError) as caught:
                d.DeviceDenseBedOwner.from_device_bed(
                    bed.state,
                    geometry=None,
                    mass_resolution_kg=1.0,
                    n_voxels=1,
                    n_classes=1,
                    case_sha256="a" * 64,
                    time_s=0.0,
                    limits=None,
                )
            self.assertEqual(caught.exception.code, "unsupported_backend")
        from types import ModuleType

        fake = ModuleType("cupy")
        fake.ndarray = np.ndarray
        with (
            mock.patch.dict(sys.modules, {"cupy": fake}),
            self.assertRaises(d.DeviceBedError),
        ):
            d.DeviceDenseBedOwner.from_device_bed(
                None,
                geometry=None,
                mass_resolution_kg=1.0,
                n_voxels=1,
                n_classes=1,
                case_sha256="a" * 64,
                time_s=0.0,
                limits=None,
            )
        self.assertFalse(any(k == "cupy" or k.startswith("cupy.") for k in sys.modules))

    def test_native_schema_and_inventory_parity(self):
        owner, bed, _ = create()
        metadata = backend._native()
        for cls, fields in owner._schemas.items():
            self.assertEqual(tuple(fields), tuple(metadata.schemas[cls]))
            self.assertTrue(all(f.init for f in dataclasses.fields(cls)))
            self.assertNotIn("__post_init__", vars(cls))
        self.assertEqual(len(array_leaves(bed.state)), 23)
        ev = owner.evidence()
        self.assertEqual(
            ev["native_placement_inventory"], array_leaf_inventory(bed.state)
        )
        inv = ev["native_placement_inventory"]
        self.assertEqual(
            ev["required_structural_work_bytes"],
            6 * inv["transfer_bytes"] + ev["comparison_reservation_bytes"] + (1 << 20),
        )
        self.assertFalse(ev["whole_worker_memory_fit"])
        self.assertFalse(ev["restart_supported"])
        self.assertNotIn("content_sha256", ev)

    def test_source_stage_and_postpublication_candidate_are_independent(self):
        owner, bed, ops = create()
        bed.state.water.depth_m[:] = 0.4
        owner.verify()
        stage = owner.begin_stage()
        candidate = stage.state
        candidate.water.depth_m[:] = 0.0
        old = exact_record(owner)
        prepared = owner.prepare_publish(stage, candidate, time_s=owner.time_s + 1.0)
        self.assertIs(owner._accepted, old[0])
        self.assertEqual(owner.generation, 0)
        owner.publish(prepared)
        self.assertTrue(stage.consumed)
        self.assertEqual(owner.generation, 1)
        candidate.water.depth_m[:] = 0.7
        owner.verify()
        self.assertTrue(np.all(owner._accepted.state.water.depth_m == 0.0))
        self.assertGreater(ops.clone_calls, 4 * 23)
        with self.assertRaises(d.DeviceBedError):
            _ = stage.state

    def test_strong_memo_preserves_exact_aliases_without_native_permission(self):
        owner, bed, ops = create()
        ledger = dataclasses.replace(
            bed.state.ledger,
            pending_compensation_kg=bed.state.ledger.pending_bed_mass_change_kg,
        )
        aliased = dataclasses.replace(bed.state, ledger=ledger)
        inv = array_leaf_inventory(aliased)
        before = ops.clone_calls
        cloned = owner._clone(aliased)
        self.assertIs(
            cloned.ledger.pending_compensation_kg,
            cloned.ledger.pending_bed_mass_change_kg,
        )
        self.assertIsNot(
            cloned.ledger.pending_bed_mass_change_kg, ledger.pending_bed_mass_change_kg
        )
        self.assertEqual(ops.clone_calls - before, inv["n_unique_leaves"])
        self.assertLess(inv["unique_bytes"], inv["transfer_bytes"])
        stage = owner.begin_stage()
        with self.assertRaises(d.DeviceBedError) as caught:
            owner.prepare_publish(stage, aliased, time_s=owner.time_s)
        self.assertEqual(caught.exception.code, "invalid_native_state")
        owner.discard(stage)

    def test_zero_dimensional_raw_byte_compare_and_signed_zero(self):
        ops = CpuOps()
        self.assertFalse(bool(ops.raw_equal(np.array(0.0), np.array(-0.0))))
        owner, _, _ = create()
        owner._accepted.state.water.depth_m[0, 0] = -0.0
        owner._accepted.mirror.water.depth_m[0, 0] = 0.0
        self.assertTrue(
            np.array_equal(
                owner._accepted.state.water.depth_m,
                owner._accepted.mirror.water.depth_m,
            )
        )
        with self.assertRaises(d.DeviceBedError) as caught:
            owner.verify()
        self.assertEqual(caught.exception.code, "integrity_failed")

    def test_scalar_integrity_is_exact_and_time_checks_native(self):
        owner, _, _ = create()
        object.__setattr__(owner._accepted.state.ledger, "largest_residual_kg", -0.0)
        with self.assertRaises(d.DeviceBedError):
            owner.verify()
        owner, _, _ = create()
        stage = owner.begin_stage()
        for time_s in (-0.0, True, float("nan"), owner.time_s - 1.0):
            with self.subTest(time=time_s), self.assertRaises(d.DeviceBedError):
                owner.prepare_publish(stage, stage.state, time_s=time_s)
        future = dataclasses.replace(
            stage.state.committed_topography, last_commit_time_s=owner.time_s + 5.0
        )
        candidate = dataclasses.replace(stage.state, committed_topography=future)
        with self.assertRaises(d.DeviceBedError) as caught:
            owner.prepare_publish(stage, candidate, time_s=owner.time_s)
        self.assertEqual(caught.exception.code, "future_commit")
        owner.discard(stage)

    def test_clone_and_packet_failures_preserve_accepted_state_and_ticket(self):
        for where in ("begin_copy", "prepare_copy", "prepare_packet", "publish_packet"):
            with self.subTest(where=where):
                owner, _, ops = create()
                old = exact_record(owner)
                if where == "begin_copy":
                    ops.fail_clone = ops.clone_calls + 1
                    with self.assertRaises(MemoryError):
                        owner.begin_stage()
                    self.assertIsNone(owner._active)
                else:
                    stage = owner.begin_stage()
                    if where == "prepare_copy":
                        ops.fail_clone = ops.clone_calls + 3
                        with self.assertRaises(MemoryError):
                            owner.prepare_publish(
                                stage, stage.state, time_s=owner.time_s
                            )
                        self.assertIs(owner._active, stage)
                    elif where == "prepare_packet":
                        ops.fail_packet = ops.packet_calls + 1
                        with self.assertRaises(RuntimeError):
                            owner.prepare_publish(
                                stage, stage.state, time_s=owner.time_s
                            )
                        self.assertIs(owner._active, stage)
                    else:
                        prepared = owner.prepare_publish(
                            stage, stage.state, time_s=owner.time_s
                        )
                        ops.fail_packet = ops.packet_calls + 1
                        with self.assertRaises(RuntimeError):
                            owner.publish(prepared)
                        self.assertIs(owner._active, prepared)
                    ops.fail_clone = ops.fail_packet = None
                    owner.discard(owner._active)
                self.assertEqual(exact_record(owner), old)
                owner.verify()

    def test_native_validation_failure_does_not_consume_work(self):
        owner, _, _ = create()
        old = exact_record(owner)
        stage = owner.begin_stage()
        stage.state.voxel_column.mass_kg[0, 0, 0, 0] = -1.0
        with self.assertRaises(d.DeviceBedError) as caught:
            owner.prepare_publish(stage, stage.state, time_s=owner.time_s)
        self.assertEqual(caught.exception.code, "invalid_native_state")
        self.assertEqual(exact_record(owner), old)
        self.assertFalse(stage.consumed)
        owner.discard(stage)

    def test_foreign_consumed_stale_and_parallel_tickets(self):
        owner, _, _ = create()
        other, _, _ = create()
        stage = owner.begin_stage()
        with self.assertRaises(d.DeviceBedError):
            owner.begin_stage()
        with self.assertRaises(d.DeviceBedError):
            other.discard(stage)
        prepared = owner.prepare_publish(stage, stage.state, time_s=owner.time_s)
        with self.assertRaises(d.DeviceBedError):
            owner.prepare_publish(stage, stage.state, time_s=owner.time_s)
        owner.discard(prepared)
        with self.assertRaises(d.DeviceBedError):
            owner.publish(prepared)
        with self.assertRaises(d.DeviceBedError):
            owner.discard(stage)
        new = owner.begin_stage()
        owner.discard(new)

    def test_device_stream_thread_refusals_precede_copies(self):
        owner, _, ops = create()
        before = ops.clone_calls
        ops.device_id = 1
        with self.assertRaises(d.DeviceBedError):
            owner.begin_stage()
        ops.device_id = 0
        ops.stream_id = 1
        with self.assertRaises(d.DeviceBedError):
            owner.begin_stage()
        ops.stream_id = 0
        errors = []

        def other_thread():
            try:
                owner.begin_stage()
            except d.DeviceBedError as exc:
                errors.append(exc.code)

        t = threading.Thread(target=other_thread)
        t.start()
        t.join(timeout=5)
        self.assertFalse(t.is_alive())
        self.assertEqual(errors, ["wrong_thread"])
        self.assertEqual(before, ops.clone_calls)

    def test_publication_has_no_fallible_copy_after_reference_swap(self):
        owner, _, ops = create()
        stage = owner.begin_stage()
        prepared = owner.prepare_publish(stage, stage.state, time_s=owner.time_s)
        clones = ops.clone_calls
        ops.fail_clone = clones + 1
        owner.publish(prepared)
        self.assertEqual(clones, ops.clone_calls)
        self.assertEqual(owner.generation, 1)
        self.assertIsNone(owner._active)

    def test_prepared_copy_does_not_follow_later_work_mutation(self):
        owner, _, _ = create()
        stage = owner.begin_stage()
        candidate = stage.state
        candidate.water.depth_m.fill(0.0)
        prepared = owner.prepare_publish(stage, candidate, time_s=owner.time_s)
        candidate.water.depth_m.fill(0.3)
        owner.publish(prepared)
        owner.verify()
        self.assertTrue(np.all(owner._accepted.state.water.depth_m == 0.0))

    def test_private_corruption_before_publish_refuses_without_generation_change(self):
        owner, _, _ = create()
        stage = owner.begin_stage()
        prepared = owner.prepare_publish(stage, stage.state, time_s=owner.time_s)
        old = owner._accepted
        value = float(old.state.water.depth_m[0, 0])
        old.state.water.depth_m[0, 0] = 0.4
        with self.assertRaises(d.DeviceBedError):
            owner.publish(prepared)
        self.assertIs(owner._accepted, old)
        self.assertEqual(owner.generation, 0)
        old.state.water.depth_m[0, 0] = value
        owner.discard(prepared)
        owner.verify()

    def test_disjoint_contiguous_view_is_severed_into_owned_copy(self):
        _, bed, _ = create()
        source = np.empty((bed.context.geometry.ny + 1, bed.context.geometry.nx))
        view = source[:-1]
        view[:] = bed.state.water.depth_m
        state = dataclasses.replace(
            bed.state, water=dataclasses.replace(bed.state.water, depth_m=view)
        )
        owner, _, _ = create(state)
        self.assertIsNone(owner._accepted.state.water.depth_m.base)
        source.fill(0.2)
        owner.verify()

    def test_native_integer_touch_dtype_is_preserved(self):
        _, bed, _ = create()
        state = dataclasses.replace(
            bed.state,
            ledger=dataclasses.replace(
                bed.state.ledger,
                n_physical_touches=bed.state.ledger.n_physical_touches.astype(np.int32),
            ),
        )
        owner, _, _ = create(state)
        self.assertEqual(
            owner._accepted.state.ledger.n_physical_touches.dtype, np.int32
        )

    def test_plain_integer_limit_change_and_boolean_ticket_refuse(self):
        owner, _, _ = create()
        stage = owner.begin_stage()
        stage._generation = False
        with self.assertRaises(d.DeviceBedError):
            owner.discard(stage)
        stage._generation = 0
        owner.discard(stage)
        object.__setattr__(
            owner._limits, "max_work_bytes", owner._limits.max_work_bytes + 1
        )
        with self.assertRaises(d.DeviceBedError) as caught:
            owner.begin_stage()
        self.assertEqual(caught.exception.code, "invalid_limit")


class AdmissionChecks(unittest.TestCase):
    def test_clock_subclass_cannot_run_hooks_between_geometry_check_and_copy(self):
        class NeverConvert(float):
            def __float__(self):
                raise AssertionError("clock subclass conversion hook executed")

        with self.assertRaises(d.DeviceBedError) as caught:
            create(time_s=NeverConvert(1.0))
        self.assertEqual(caught.exception.code, "invalid_time")
        owner, _, _ = create()
        stage = owner.begin_stage()
        with self.assertRaises(d.DeviceBedError):
            owner.prepare_publish(stage, stage.state, time_s=NeverConvert(owner.time_s))
        owner.discard(stage)

    def test_hidden_geometry_resources_refuse_before_deepcopy(self):
        class NeverCopy:
            def __deepcopy__(self, memo):
                raise AssertionError("undeclared geometry resource was copied")

        for where in ("geometry", "boundary", "growth"):
            with self.subTest(where=where):
                inputs = fixture(2)
                geometry = inputs["geometry"]
                holder = (
                    geometry
                    if where == "geometry"
                    else (
                        geometry.boundary_x
                        if where == "boundary"
                        else copy.deepcopy(geometry.voxel_growth)
                    )
                )
                if where == "growth":
                    geometry = dataclasses.replace(geometry, voxel_growth=holder)
                object.__setattr__(holder, "hidden_payload", NeverCopy())
                ops = CpuOps()
                with self.assertRaises(d.DeviceBedError) as caught:
                    create(geometry=geometry, ops=ops)
                self.assertEqual(caught.exception.code, "invalid_geometry")
                self.assertEqual(ops.clone_calls, 0)

    def test_host_boundary_metadata_size_is_bounded_before_copy(self):
        geometry = fixture(2)["geometry"]
        boundary = dataclasses.replace(
            geometry.boundary_x, inflow_flux_kg_m_s={"x" * (64 << 10): 0.0}
        )
        geometry = dataclasses.replace(geometry, boundary_x=boundary)
        with self.assertRaises(d.DeviceBedError) as caught:
            create(geometry=geometry)
        self.assertEqual(caught.exception.code, "invalid_geometry")

    def assert_early(self, mutate, code=None, **kwargs):
        _, bed, _ = create()
        state = mutate(bed.state)
        ops = CpuOps()
        with (
            mock.patch.object(
                d.DeviceDenseBedOwner,
                "_validate",
                side_effect=AssertionError("validator too early"),
            ),
            self.assertRaises(d.DeviceBedError) as caught,
        ):
            create(state, ops=ops, **kwargs)
        if code is not None:
            self.assertEqual(caught.exception.code, code)
        self.assertEqual(ops.clone_calls, 0)

    def test_huge_virtual_shape_refuses_without_reading_or_allocating_it(self):
        def huge(s):
            array = np.lib.stride_tricks.as_strided(
                s.voxel_column.mass_kg,
                shape=(3, 4, 10**12, 2),
                strides=(64 * 10**12, 16 * 10**12, 16, 8),
            )
            return dataclasses.replace(
                s, voxel_column=dataclasses.replace(s.voxel_column, mass_kg=array)
            )

        self.assert_early(huge, "invalid_shape")

    def test_unknown_holder_and_extra_attribute_refuse(self):
        self.assert_early(
            lambda s: dataclasses.replace(s, voxel_column=object()),
            "unsupported_holder",
        )

        def extra(s):
            object.__setattr__(s.active_layer, "hidden", np.ones(1))
            return s

        self.assert_early(extra, "unsupported_schema")

    def test_shape_dtype_and_noncontiguous_refuse_before_native_validator(self):
        self.assert_early(
            lambda s: dataclasses.replace(
                s,
                voxel_column=dataclasses.replace(
                    s.voxel_column, mass_kg=np.empty((1, 1, 1, 1))
                ),
            ),
            "invalid_shape",
        )
        self.assert_early(
            lambda s: dataclasses.replace(
                s,
                active_layer=dataclasses.replace(
                    s.active_layer, mass_kg=s.active_layer.mass_kg.astype(object)
                ),
            ),
            "unsupported_layout",
        )
        self.assert_early(
            lambda s: dataclasses.replace(
                s, water=dataclasses.replace(s.water, depth_m=s.water.depth_m[:, ::-1])
            ),
            "unsupported_layout",
        )
        self.assert_early(
            lambda s: dataclasses.replace(
                s,
                ledger=dataclasses.replace(
                    s.ledger,
                    n_physical_touches=s.ledger.n_physical_touches.astype(bool),
                ),
            ),
            "invalid_dtype",
        )

    def test_overlap_refuses_before_native_validator(self):
        self.assert_early(
            lambda s: dataclasses.replace(
                s,
                water=dataclasses.replace(
                    s.water, depth_m=s.committed_topography.reference_elevation_m.view()
                ),
            ),
            "overlapping_views",
        )

    def test_budget_and_tampered_limits_refuse_before_native_validator(self):
        self.assert_early(
            lambda s: s, "work_bytes_cap", limits=d.DeviceBedLimits(1, 100, 1, 1)
        )
        limits = d.DeviceBedLimits(1 << 24, 1 << 29, 1, 1)
        object.__setattr__(limits, "native_validation_scratch_bytes", True)
        self.assert_early(lambda s: s, "invalid_limit", limits=limits)
        owner, _, _ = create()
        object.__setattr__(owner._limits, "comparison_reduction_scratch_bytes", False)
        with self.assertRaises(d.DeviceBedError):
            owner.begin_stage()

    def test_scratch_is_admitted_in_addition_to_six_roots_before_native_values(self):
        _, bed, _ = create()
        size = array_leaf_inventory(bed.state)["transfer_bytes"]
        self.assert_early(
            lambda state: state,
            "work_bytes_cap",
            limits=d.DeviceBedLimits(size, 6 * size, 1, 1),
        )

    def test_wrong_device_scalar_type_and_init_false_refuse_before_copy(self):
        _, bed, _ = create()
        ops = CpuOps()
        ops.wrong_leaf = bed.state.water.depth_m
        with self.assertRaises(d.DeviceBedError) as caught:
            create(bed.state, ops=ops)
        self.assertEqual(caught.exception.code, "wrong_device")
        self.assertEqual(ops.clone_calls, 0)
        self.assert_early(
            lambda s: dataclasses.replace(
                s,
                ledger=dataclasses.replace(
                    s.ledger, largest_residual_kg=np.float64(0.0)
                ),
            ),
            "invalid_scalar",
        )
        real_fields = dataclasses.fields

        def altered(value):
            fields = real_fields(value)
            if type(value) is type(bed.state):
                first = copy.copy(fields[0])
                first.init = False
                return (first, *fields[1:])
            return fields

        with (
            mock.patch.object(d.dataclasses, "fields", side_effect=altered),
            self.assertRaises(d.DeviceBedError) as caught,
        ):
            create(bed.state)
        self.assertEqual(caught.exception.code, "unsupported_schema")

    def test_native_validator_sequence_matches_unchanged_cpu_adapter(self):
        owner, bed, _ = create()
        expected, actual = [], []
        names = [
            "is_windowed_bed",
            "check_voxel_column_geometry_shape",
            "check_voxel_column_state",
            "check_active_layer_voxel_partition",
            "check_sediment_availability_state",
            "validate_water_state",
            "validate_ledger_state",
            "validate_committed_topography_state",
        ]

        def observe(n, target):
            stack = __import__("contextlib").ExitStack()
            for name in names:
                fn = getattr(n, name)

                def call(*args, _name=name, _fn=fn, **kw):
                    target.append(_name)
                    return _fn(*args, **kw)

                stack.enter_context(mock.patch.object(n, name, side_effect=call))
            return stack

        cpu = dense_bed._native()
        with observe(cpu, expected):
            bed._validate_shared(cpu, bed.state)
        with observe(owner._n, actual):
            owner._validate(bed.state, owner.time_s)
        self.assertEqual(actual, expected)


class DormantGateChecks(unittest.TestCase):
    def test_gated_file_has_no_import_side_effect_and_builds_native_host_fixture(self):
        path = Path(__file__).resolve().parent / "gpu/test_dense_device_bed_cupy.py"
        spec = importlib.util.spec_from_file_location("device_owner_gate_source", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(RuntimeError):
            module.load_tests(None, None, None)
        from masy.runtime_worker import WorkerRefusal

        with self.assertRaises(WorkerRefusal):
            module._scope({}, {}, d.__file__)
        fixture_bed = module._fixture()
        self.assertEqual(fixture_bed.state.voxel_column.mass_kg.shape, (2, 2, 3, 2))
        self.assertIs(type(fixture_bed.state), dense_bed._native().BedState)
        self.assertFalse(any(k == "cupy" or k.startswith("cupy.") for k in sys.modules))


if __name__ == "__main__":
    unittest.main()
