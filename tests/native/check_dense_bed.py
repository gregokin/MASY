"""Explicit native checks of masy.dense_bed on real authored CPU fixtures.

NOT collected automatically (no `test_` prefix, no package `__init__`). Run it
explicitly with an interpreter whose import paths resolve the locked MAPLE /
SYRUP sources and the MASY candidate `src`, e.g.

    python -I tests/native/check_dense_bed.py

Every object here is built with actual MAPLE/SYRUP constructors; no native
type, binding, physics, provenance or resolver is faked or patched. The only
instrumentation is in `ResourceChecks`: two MASY allocation helpers
(`_owned_array`, `_probe_per_cell_bytes`) are wrapped call-through, inside a
`with` block, to observe call order. The water exchange is MAPLE's genuine water
API through SYRUP's actual `apply_bed_demand`. No wind stepping, wind
finalization, water event or restart is exercised (out of chunk scope).
"""

from __future__ import annotations

import copy
import dataclasses
import sys
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
from maple.aeolian.scheduler.state import SchedulerPhysicalState, SchedulerState  # noqa: E402
from maple.aeolian.trajectories.drag import TRAJECTORY_ALGORITHM_VERSION  # noqa: E402
from maple.core.boundaries import AxisBoundary, BoundaryKind  # noqa: E402
from maple.core.parameters.avalanching import AvalanchingSpec  # noqa: E402
from maple.core.parameters.geometry import GeometrySpec  # noqa: E402
from maple.core.parameters.grain_classes import GrainClass, GrainClassSet  # noqa: E402
from maple.core.parameters.topographic_commit import TopographicCommitSpec  # noqa: E402
from maple.core.parameters.water_coupling import MAHLERAN_1_2_1_CLASS_DIAMETERS_M  # noqa: E402
from maple.core.types.cohort import zeros_mobile_cohort_state  # noqa: E402
from maple.core.types.threshold_modifiers import neutral_threshold_modifier_state  # noqa: E402
from maple.core.types.vegetation import uniform_vegetation_raster  # noqa: E402
from maple.core.types.voxel import VoxelColumnState  # noqa: E402
from maple.water.interfaces import WaterProcessDemand  # noqa: E402
from maple_syrup.sediment_bed import BedContext, BedState, apply_bed_demand  # noqa: E402

from masy import dense_bed as adapter  # noqa: E402
from masy.dense_bed import (  # noqa: E402
    SHARED_FIELDS,
    DenseBedError,
    DenseBedLimits,
    build_dense_bed,
    estimate_initialization_bytes,
)

MRES = 1e-10
T0 = 1234.5678
NY, NX, NV = 3, 4, 4
VOXEL_DZ = 0.05  # capacity 0.2 m: the fixture elevations span several voxels (stratigraphy)


def grain_classes(n_classes: int) -> GrainClassSet:
    if n_classes == 6:
        diameters = MAHLERAN_1_2_1_CLASS_DIAMETERS_M
        ids = tuple(f"phi_{k}" for k in range(1, 7))
    else:
        diameters = (1.0e-4, 3.0e-4)[:n_classes]
        ids = ("fine", "coarse")[:n_classes]
    return GrainClassSet(classes=tuple(
        GrainClass(class_id=c, diameter_m=float(d), particle_density_kg_m3=2650.0, is_aggregate=False)
        for c, d in zip(ids, diameters, strict=True)))


def geometry(classes: GrainClassSet, *, dx=0.5, dy=0.5) -> GeometrySpec:
    def boundary():
        return AxisBoundary(kind=BoundaryKind.PRESCRIBED, inflow_flux_kg_m_s={c.class_id: 0.0 for c in classes.classes})

    return GeometrySpec(nx=NX, ny=NY, dx_m=dx, dy_m=dy, boundary_x=boundary(), boundary_y=boundary(),
                        voxel_dz_m=VOXEL_DZ, bulk_density_kg_m3=1250.0, active_layer_thickness_m=0.002)


def fixture(n_classes: int, **overrides) -> dict:
    classes = grain_classes(n_classes)
    rows, cols = np.indices((NY, NX)).astype(np.float64)
    elevation = 0.12 + 0.01 * rows + 0.005 * cols
    if n_classes == 1:
        fractions = np.ones((NY, NX, 1))
    else:
        base = np.linspace(1.0, 2.0, n_classes)[None, None, :] * (1.0 + 0.25 * (cols % 2))[:, :, None]
        fractions = base / base.sum(axis=-1, keepdims=True)
    available = np.full((NY, NX, n_classes), 0.6)
    available[..., 0] = 0.35
    threshold = dataclasses.replace(neutral_threshold_modifier_state(NY, NX),
                                    physical_crust_multiplier=np.full((NY, NX), 1.5))
    inputs = dict(
        geometry=geometry(classes), grain_classes=classes, mass_resolution_kg=MRES, n_voxels=NV,
        initial_bed_elevation_m=elevation, initial_class_fractions=fractions,
        initial_available_fraction=available, initial_water_depth_m=np.full((NY, NX), 0.002),
        vegetation=uniform_vegetation_raster(NY, NX, height_m=0.3, cover_fraction=0.25),
        threshold_modifier_state=threshold, topographic_commit_spec=TopographicCommitSpec(),
        avalanching_spec=AvalanchingSpec(enabled=False), active_mask=np.ones((NY, NX), dtype=bool),
        initial_time_s=T0,
        limits=DenseBedLimits(host_memory_cap_bytes=1 << 30, max_voxels_per_column=64, stage_copy_max_bytes=1 << 24),
    )
    inputs.update(overrides)
    return inputs


def per_class_total(state) -> np.ndarray:
    return state.voxel_column.mass_kg.sum(axis=(0, 1, 2)) + state.active_layer.mass_kg.sum(axis=(0, 1))


class InitializationChecks(unittest.TestCase):
    def test_one_and_two_class_beds_are_genuine_native_with_stratigraphy_and_partial_availability(self) -> None:
        for n_classes in (1, 2):
            with self.subTest(n_classes=n_classes):
                inputs = fixture(n_classes)
                bed = build_dense_bed(**inputs)
                state = bed.state
                self.assertIs(type(state), BedState)
                self.assertIs(type(bed.context), BedContext)
                self.assertIs(type(state.voxel_column), VoxelColumnState)
                self.assertIs(bed.context.geometry, inputs["geometry"])
                self.assertIs(bed.context.grain_classes, inputs["grain_classes"])
                self.assertEqual(state.voxel_column.mass_kg.shape, (NY, NX, NV, n_classes))
                occupied_levels = (state.voxel_column.mass_kg.sum(axis=-1) > 0).sum(axis=2)
                self.assertGreaterEqual(int(occupied_levels.max()), 2)  # real buried stratigraphy
                active = state.active_layer.mass_kg
                self.assertTrue(bool((active > 0).all()))
                avail = state.sediment_availability
                np.testing.assert_allclose(avail.available_mass_kg / active, inputs["initial_available_fraction"],
                                           rtol=1e-12)
                self.assertTrue(bool((avail.bound_mass_kg > 0).all()))  # not 100% available
                g = inputs["geometry"]
                requested = (inputs["initial_bed_elevation_m"][:, :, None] * g.bulk_density_kg_m3 * g.dx_m * g.dy_m
                             * inputs["initial_class_fractions"]).sum(axis=(0, 1))
                np.testing.assert_allclose(per_class_total(state), requested, rtol=1e-12)
                self.assertEqual(bed.time_s.hex(), T0.hex())
                self.assertEqual(bed.case_record["class_order"], [c.class_id for c in inputs["grain_classes"].classes])

    def test_case_hash_is_stable_static_and_separate_from_evolving_identity(self) -> None:
        inputs = fixture(2)
        a, b = build_dense_bed(**inputs), build_dense_bed(**fixture(2))
        self.assertEqual(a.case_sha256, b.case_sha256)
        self.assertEqual(a.identity_sha256(), b.identity_sha256())
        changed = fixture(2)
        changed["initial_available_fraction"][0, 0, 1] = np.nextafter(0.6, 1.0)
        self.assertNotEqual(build_dense_bed(**changed).case_sha256, a.case_sha256)
        crust = fixture(2)
        crust["threshold_modifier_state"] = dataclasses.replace(crust["threshold_modifier_state"],
                                                                physical_crust_multiplier=np.full((NY, NX), 1.6))
        self.assertNotEqual(build_dense_bed(**crust).case_sha256, a.case_sha256)
        # Caller mutation of an input array after construction cannot redefine the case.
        inputs["initial_bed_elevation_m"][:] = 0.0
        inputs["initial_available_fraction"][:] = 1.0
        a.verify_case_inputs()
        self.assertEqual(a.case_sha256, b.case_sha256)
        held = a.context.initial_available_fraction
        self.assertFalse(held.flags.writeable)
        with self.assertRaises(ValueError):
            held[0, 0, 0] = 1.0


class MappingChecks(unittest.TestCase):
    def test_shared_field_identity_in_both_directions_and_fresh_native_caches(self) -> None:
        inputs = fixture(2)
        bed = build_dense_bed(**inputs)
        mobile = zeros_mobile_cohort_state(64)
        scheduler = bed.fresh_scheduler_state(mobile, time_s=bed.time_s)
        self.assertIs(type(scheduler), SchedulerState)
        physical = scheduler.physical
        for name in SHARED_FIELDS:
            self.assertIs(getattr(physical, name), getattr(bed.state, name), name)
        self.assertIs(physical.vegetation, inputs["vegetation"])
        self.assertIs(physical.threshold_modifier_state, inputs["threshold_modifier_state"])
        self.assertIs(physical.mobile_state, mobile)
        self.assertEqual(physical.time_s.hex(), T0.hex())
        # Native fresh-entry behaviour: no coupled state, block 0, every derived wind cache uninitialized.
        self.assertIsNone(scheduler.coupled)
        self.assertEqual((scheduler.block_index, scheduler.block_offset_s), (0, 0.0))
        self.assertIsNone(scheduler.last_resolved_wind)
        self.assertIsNone(scheduler.topographic_wind_cache)
        self.assertIsNone(scheduler.vegetation_wind_cache)
        self.assertIsNone(scheduler.active_layer_composition_cache)
        self.assertEqual(physical.coupled_trajectory_semantics, TRAJECTORY_ALGORITHM_VERSION)
        # Scheduler -> bed by reference.
        later = dataclasses.replace(physical, time_s=T0 + 5.0)
        returned = bed.adopt_scheduler_physical(later)
        self.assertIs(returned, bed.state)
        for name in SHARED_FIELDS:
            self.assertIs(getattr(bed.state, name), getattr(later, name), name)
        self.assertEqual(bed.time_s.hex(), (T0 + 5.0).hex())
        self.assertIs(bed.vegetation_cover_fraction(), inputs["vegetation"].raster.cover_fraction)

    def test_real_native_water_exchange_is_carried_through_the_adapter(self) -> None:
        inputs = fixture(6)  # the genuine six-class MAHLERAN set
        bed = build_dense_bed(**inputs)
        original = build_dense_bed(**fixture(6))  # an original-case build, for comparison only
        start = bed.state
        before_total = per_class_total(start) + start.water.mobile_mass_by_cell_class_kg.sum(axis=(0, 1))
        removal = 0.25 * start.sediment_availability.available_mass_kg
        new_state, _ = apply_bed_demand(start, bed.context, WaterProcessDemand(removal, np.zeros_like(removal)))
        bed.adopt_bed_state(new_state, time_s=T0 + 60.0)
        self.assertIs(bed.state, new_state)
        mobile = new_state.water.mobile_mass_by_cell_class_kg
        self.assertTrue(bool((mobile > 0).any()))
        self.assertTrue(bool((new_state.sediment_availability.available_mass_kg
                              < start.sediment_availability.available_mass_kg).any()))
        np.testing.assert_allclose(per_class_total(new_state) + mobile.sum(axis=(0, 1)), before_total, rtol=1e-12)
        with self.assertRaises(DenseBedError) as ctx:
            bed.fresh_scheduler_state(zeros_mobile_cohort_state(8), time_s=bed.time_s)
        self.assertEqual(ctx.exception.code, "water_mobile_not_empty")
        self.assertIs(bed.state, new_state)

        back, _ = apply_bed_demand(new_state, bed.context, WaterProcessDemand(np.zeros_like(mobile), mobile.copy()))
        bed.adopt_bed_state(back, time_s=T0 + 120.0)
        self.assertFalse(bool(back.water.mobile_mass_by_cell_class_kg.any()))
        np.testing.assert_allclose(per_class_total(back), before_total, rtol=1e-12)
        self.assertTrue(bool(np.any(back.ledger.actual_mass_positive_kg != 0)))  # native gross activity retained
        self.assertNotEqual(bed.identity_record()["fields"], original.identity_record()["fields"])
        self.assertEqual(bed.case_sha256, original.case_sha256)
        bed.verify_case_inputs()
        scheduler = bed.fresh_scheduler_state(zeros_mobile_cohort_state(8), time_s=T0 + 120.0)
        self.assertIs(scheduler.physical.sediment_availability, back.sediment_availability)
        self.assertIs(scheduler.physical.ledger, back.ledger)
        self.assertIs(scheduler.physical.water, back.water)
        self.assertIs(scheduler.physical.vegetation, inputs["vegetation"])
        self.assertIs(scheduler.physical.threshold_modifier_state, inputs["threshold_modifier_state"])


class RefusalChecks(unittest.TestCase):
    def assertRefused(self, code: str, call, *args, **kwargs) -> None:
        with self.assertRaises(DenseBedError) as ctx:
            call(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)

    def test_invalid_static_inputs_refuse_before_any_mutation(self) -> None:
        def with_array(name, change):
            inputs = fixture(2)
            array = inputs[name].copy()
            change(array)
            inputs[name] = array
            return inputs

        classes = grain_classes(2)
        too_high = fixture(2)
        too_high["initial_bed_elevation_m"] = np.full((NY, NX), NV * VOXEL_DZ + 0.01)
        cases = [
            ("unsupported_backend", fixture(2, backend="cupy")),
            ("non_square_grid", fixture(2, geometry=geometry(classes, dx=0.5, dy=0.25))),
            ("voxel_capacity_exceeded", fixture(2, n_voxels=65)),
            ("voxel_capacity_exceeded", too_high),
            ("invalid_static_input", fixture(2, initial_bed_elevation_m=np.full((NY, NX), 0.15, dtype=np.float32))),
            ("invalid_static_input", fixture(2, initial_bed_elevation_m=np.full((NY + 1, NX), 0.15))),
            ("nonfinite_static_input", with_array("initial_bed_elevation_m", lambda a: a.__setitem__((0, 0), np.nan))),
            ("invalid_static_input", with_array("initial_class_fractions", lambda a: a.__setitem__((0, 0, 0), 0.9))),
            ("invalid_static_input", with_array("initial_available_fraction", lambda a: a.__setitem__((0, 0, 0), 1.5))),
            ("invalid_static_input", with_array("initial_water_depth_m", lambda a: a.__setitem__((0, 0), -0.0))),
            ("insufficient_bed", fixture(2, initial_bed_elevation_m=np.full((NY, NX), 0.001))),
            ("unsupported_active_mask", with_array("active_mask", lambda a: a.__setitem__((0, 0), False))),
            ("unsupported_type", fixture(2, geometry=dataclasses.asdict(geometry(classes)))),
            ("unsupported_type", fixture(2, vegetation=None)),
            ("invalid_vegetation", fixture(2, vegetation=uniform_vegetation_raster(NY + 1, NX, height_m=0.3,
                                                                                    cover_fraction=0.25))),
            ("invalid_time", fixture(2, initial_time_s=-0.0)),
            ("invalid_time", fixture(2, initial_time_s=float("nan"))),
            ("host_memory_cap_exceeded", fixture(2, limits=DenseBedLimits(host_memory_cap_bytes=1024,
                                                                          max_voxels_per_column=64,
                                                                          stage_copy_max_bytes=1 << 24))),
        ]
        for code, inputs in cases:
            with self.subTest(code=code):
                snapshot = {k: v.copy() for k, v in inputs.items() if isinstance(v, np.ndarray)}
                self.assertRefused(code, build_dense_bed, **inputs)
                for key, value in snapshot.items():
                    np.testing.assert_array_equal(inputs[key], value)

    def test_invalid_adoptions_leave_the_bed_unchanged(self) -> None:
        inputs = fixture(2)
        bed = build_dense_bed(**inputs)
        state, identity = bed.state, bed.identity_sha256()
        self.assertRefused("missing_shared_field", bed.adopt_bed_state, dataclasses.replace(state, water=None),
                           time_s=T0)
        self.assertRefused("unsupported_type", bed.adopt_bed_state,
                           dataclasses.replace(state, voxel_column=state.active_layer), time_s=T0)
        self.assertRefused("time_reversed", bed.adopt_bed_state, state, time_s=T0 - 1.0)
        self.assertRefused("invalid_time", bed.adopt_bed_state, state, time_s=-0.0)
        self.assertRefused("time_mismatch", bed.fresh_scheduler_state, zeros_mobile_cohort_state(4), time_s=T0 + 1.0)
        physical = bed.fresh_scheduler_state(zeros_mobile_cohort_state(4), time_s=T0).physical
        self.assertRefused("prescribed_field_replaced", bed.adopt_scheduler_physical,
                           dataclasses.replace(physical, vegetation=copy.deepcopy(inputs["vegetation"])))
        self.assertRefused("prescribed_field_replaced", bed.adopt_scheduler_physical,
                           dataclasses.replace(physical, vegetation=None))
        self.assertRefused("unsupported_type", bed.adopt_scheduler_physical, state)
        self.assertIs(bed.state, state)
        self.assertEqual(bed.identity_sha256(), identity)
        # An in-place change of a prescribed field is detected, never silently adopted.
        # (If the native raster is itself read-only, the in-place write is refused natively instead.)
        cover = inputs["vegetation"].raster.cover_fraction
        if not cover.flags.writeable:
            with self.assertRaises(ValueError):
                cover[0, 0] = 0.5
        else:
            saved = cover.copy()
            cover[0, 0] = 0.5
            try:
                self.assertRefused("prescribed_field_changed", bed.fresh_scheduler_state,
                                   zeros_mobile_cohort_state(4), time_s=T0)
            finally:
                cover[:] = saved
        bed.verify_case_inputs()


class StagingChecks(unittest.TestCase):
    def test_bounded_copy_is_isolated_and_restorable_once(self) -> None:
        bed = build_dense_bed(**fixture(2))
        live = bed.state
        stage = bed.stage_copy()
        self.assertLessEqual(stage.nbytes, bed.limits.stage_copy_max_bytes)
        self.assertEqual(stage.identity_sha256, bed.identity_sha256())
        for name in SHARED_FIELDS:
            self.assertIs(type(getattr(stage.state, name)), type(getattr(live, name)))
            self.assertIsNot(getattr(stage.state, name), getattr(live, name))
        self.assertFalse(np.shares_memory(stage.state.voxel_column.mass_kg, live.voxel_column.mass_kg))
        self.assertFalse(np.shares_memory(stage.state.active_layer.mass_kg, live.active_layer.mass_kg))
        live.water.depth_m[0, 0] = 0.25  # a live in-place mutation
        self.assertEqual(float(stage.state.water.depth_m[0, 0]), 0.002)
        self.assertNotEqual(bed.identity_sha256(), stage.identity_sha256)
        bed.restore_stage(stage)
        self.assertIs(bed.state, stage.state)
        self.assertEqual(bed.identity_sha256(), stage.identity_sha256)
        with self.assertRaises(DenseBedError) as ctx:
            bed.restore_stage(stage)
        self.assertEqual(ctx.exception.code, "stage_consumed")

    def test_copy_cap_refuses(self) -> None:
        bed = build_dense_bed(**fixture(2, limits=DenseBedLimits(host_memory_cap_bytes=1 << 30,
                                                                 max_voxels_per_column=64, stage_copy_max_bytes=64)))
        with self.assertRaises(DenseBedError) as ctx:
            bed.stage_copy()
        self.assertEqual(ctx.exception.code, "stage_copy_cap_exceeded")
        foreign = build_dense_bed(**fixture(1)).stage_copy()
        with self.assertRaises(DenseBedError) as ctx:
            bed.restore_stage(foreign)
        self.assertEqual(ctx.exception.code, "foreign_stage")


class ReviewRegressionChecks(unittest.TestCase):
    """Regressions for the independently reproduced component findings B1-B3."""

    def assertUnchanged(self, bed, state, time_s, identity, case_sha256) -> None:
        self.assertIs(bed.state, state)
        self.assertEqual(bed.time_s.hex(), time_s.hex())
        self.assertEqual(bed.identity_sha256(), identity)
        self.assertEqual(bed.case_sha256, case_sha256)

    def test_b1_live_boundary_flux_mutation_refuses_every_transition_without_mutation(self) -> None:
        inputs = fixture(1)
        bed = build_dense_bed(**inputs)
        state, time_s, identity, case_sha256 = bed.state, bed.time_s, bed.identity_sha256(), bed.case_sha256
        physical = bed.fresh_scheduler_state(zeros_mobile_cohort_state(4), time_s=T0).physical
        stage = bed.stage_copy()
        flux_x = inputs["geometry"].boundary_x.inflow_flux_kg_m_s
        flux_y = inputs["geometry"].boundary_y.inflow_flux_kg_m_s
        self.assertIs(bed.context.geometry.boundary_x.inflow_flux_kg_m_s, flux_x)  # the live native dict
        mutations = (
            ("changed_value", lambda: flux_x.__setitem__("fine", 0.1), lambda: flux_x.__setitem__("fine", 0.0)),
            ("added_key", lambda: flux_y.__setitem__("extra", 0.0), lambda: flux_y.pop("extra")),
        )
        operations = {
            "verify_case_inputs": lambda: bed.verify_case_inputs(),
            "fresh_scheduler_state": lambda: bed.fresh_scheduler_state(zeros_mobile_cohort_state(4), time_s=T0),
            "adopt_bed_state": lambda: bed.adopt_bed_state(state, time_s=T0),
            "adopt_scheduler_physical": lambda: bed.adopt_scheduler_physical(physical),
            "stage_copy": lambda: bed.stage_copy(),
            "restore_stage": lambda: bed.restore_stage(stage),
        }
        for label, change, undo in mutations:
            change()
            try:
                for name, call in operations.items():
                    with self.subTest(mutation=label, operation=name):
                        with self.assertRaises(DenseBedError) as ctx:
                            call()
                        self.assertEqual(ctx.exception.code, "case_input_mutated", ctx.exception.detail)
                        self.assertUnchanged(bed, state, time_s, identity, case_sha256)
                        self.assertFalse(stage.consumed)
            finally:
                undo()
        # Nothing was reset or reconstructed by the refusals: the original inputs verify and the stage still works.
        bed.verify_case_inputs()
        self.assertEqual(bed.case_sha256, build_dense_bed(**fixture(1)).case_sha256)
        bed.restore_stage(stage)
        self.assertIs(bed.state, stage.state)

    def test_b2_future_native_commit_stamp_refuses_without_state_or_time_change(self) -> None:
        bed = build_dense_bed(**fixture(1))
        state, identity, case_sha256 = bed.state, bed.identity_sha256(), bed.case_sha256
        committed = state.committed_topography
        self.assertEqual(float(committed.last_commit_time_s).hex(), T0.hex())
        future_committed = dataclasses.replace(committed, last_commit_time_s=T0 + 1.0)
        future = dataclasses.replace(state, committed_topography=future_committed)
        physical = bed.fresh_scheduler_state(zeros_mobile_cohort_state(4), time_s=T0).physical
        future_physical = dataclasses.replace(physical, committed_topography=future_committed)
        stage = bed.stage_copy()
        staged = copy.deepcopy(stage.state)
        forged = dataclasses.replace(stage, state=dataclasses.replace(
            staged, committed_topography=dataclasses.replace(staged.committed_topography,
                                                             last_commit_time_s=T0 + 1.0)))
        # Same owner and ticket, consistent identity: only the future stamp is wrong.
        forged = dataclasses.replace(forged, identity_sha256=adapter._sha256(bed._stage_identity(forged)))
        one_ulp_early = float(np.nextafter(T0 + 1.0, 0.0))
        refusals = {
            "adopt_bed_state": lambda: bed.adopt_bed_state(future, time_s=T0),
            "adopt_bed_state_one_ulp_early": lambda: bed.adopt_bed_state(future, time_s=one_ulp_early),
            "adopt_scheduler_physical": lambda: bed.adopt_scheduler_physical(future_physical),
            "restore_stage": lambda: bed.restore_stage(forged),
        }
        for name, call in refusals.items():
            with self.subTest(operation=name):
                with self.assertRaises(DenseBedError) as ctx:
                    call()
                self.assertEqual(ctx.exception.code, "future_commit", ctx.exception.detail)
                self.assertUnchanged(bed, state, T0, identity, case_sha256)
                self.assertFalse(stage.consumed)
                self.assertEqual(float(bed.state.committed_topography.last_commit_time_s).hex(), T0.hex())
        # A stamp equal to the native time is not after it (exact rule, no tolerance, no rewrite).
        bed.adopt_bed_state(future, time_s=T0 + 1.0)
        self.assertIs(bed.state, future)
        self.assertEqual(bed.time_s.hex(), (T0 + 1.0).hex())
        self.assertEqual(float(bed.state.committed_topography.last_commit_time_s).hex(), (T0 + 1.0).hex())

    def test_b3_stage_is_bound_to_its_owning_bed_instance(self) -> None:
        first, second = build_dense_bed(**fixture(1)), build_dense_bed(**fixture(1))
        self.assertEqual(first.case_sha256, second.case_sha256)
        self.assertEqual(first.identity_sha256(), second.identity_sha256())
        stage = first.stage_copy()
        state, identity = second.state, second.identity_sha256()
        with self.assertRaises(DenseBedError) as ctx:
            second.restore_stage(stage)
        self.assertEqual(ctx.exception.code, "foreign_stage")
        self.assertFalse(stage.consumed)
        self.assertUnchanged(second, state, T0, identity, second.case_sha256)
        first.restore_stage(stage)  # the owner restores it, exactly once
        self.assertIs(first.state, stage.state)
        self.assertTrue(stage.consumed)
        with self.assertRaises(DenseBedError) as ctx:
            first.restore_stage(stage)
        self.assertEqual(ctx.exception.code, "stage_consumed")
        with self.assertRaises(DenseBedError) as ctx:
            second.restore_stage(stage)
        self.assertEqual(ctx.exception.code, "foreign_stage")
        self.assertIs(second.state, state)


class StageSchedulerChecks(unittest.TestCase):
    """`stage_scheduler_state`: a genuine native scheduler over the independent stage, never the accepted bed."""

    def assertRefused(self, code: str, call, *args, **kwargs) -> None:
        with self.assertRaises(DenseBedError) as ctx:
            call(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)

    def test_work_aliases_the_stage_and_accepted_fields_are_untouched(self) -> None:
        inputs = fixture(2)
        bed = build_dense_bed(**inputs)
        accepted, identity = bed.state, bed.identity_sha256()
        stage = bed.stage_copy()
        mobile = zeros_mobile_cohort_state(64)
        scheduler = bed.stage_scheduler_state(stage, mobile, time_s=T0)
        self.assertIs(type(scheduler), SchedulerState)
        physical = scheduler.physical
        for name in SHARED_FIELDS:
            self.assertIs(getattr(physical, name), getattr(stage.state, name), name)
            self.assertIsNot(getattr(physical, name), getattr(accepted, name), name)
        self.assertIs(physical.vegetation, inputs["vegetation"])
        self.assertIs(physical.threshold_modifier_state, inputs["threshold_modifier_state"])
        self.assertIs(physical.mobile_state, mobile)
        self.assertIsNone(scheduler.coupled)
        self.assertEqual((scheduler.block_index, scheduler.block_offset_s), (0, 0.0))
        self.assertEqual(physical.coupled_trajectory_semantics, TRAJECTORY_ALGORITHM_VERSION)
        physical.voxel_column.mass_kg[0, 0, 0, 0] += 1.0  # work mutation reaches only the stage
        self.assertIs(bed.state, accepted)
        self.assertEqual(bed.identity_sha256(), identity)
        self.assertFalse(stage.consumed)
        # The existing accepted mapping is unchanged: it still aliases the accepted objects.
        fresh = bed.fresh_scheduler_state(zeros_mobile_cohort_state(64), time_s=T0).physical
        for name in SHARED_FIELDS:
            self.assertIs(getattr(fresh, name), getattr(accepted, name), name)

    def test_stale_mutated_foreign_consumed_and_mistimed_stages_refuse(self) -> None:
        bed = build_dense_bed(**fixture(2))
        mobile = zeros_mobile_cohort_state(8)
        mutated = bed.stage_copy()
        mutated.state.water.depth_m[0, 0] = 0.25
        self.assertRefused("stage_mutated", bed.stage_scheduler_state, mutated, mobile, time_s=T0)
        foreign = build_dense_bed(**fixture(2)).stage_copy()
        self.assertRefused("foreign_stage", bed.stage_scheduler_state, foreign, mobile, time_s=T0)
        self.assertRefused("foreign_stage", bed.stage_scheduler_state, bed.state, mobile, time_s=T0)
        stage = bed.stage_copy()
        self.assertRefused("time_mismatch", bed.stage_scheduler_state, stage, mobile, time_s=T0 + 1.0)
        consumed = bed.stage_copy()
        bed.restore_stage(consumed)
        self.assertRefused("stage_consumed", bed.stage_scheduler_state, consumed, mobile, time_s=T0)
        stale = bed.stage_copy()
        physical = bed.fresh_scheduler_state(zeros_mobile_cohort_state(8), time_s=T0).physical
        bed.adopt_scheduler_physical(dataclasses.replace(physical, time_s=T0 + 5.0))
        state, identity = bed.state, bed.identity_sha256()
        self.assertRefused("stale_stage", bed.stage_scheduler_state, stale, mobile, time_s=T0)
        self.assertIs(bed.state, state)
        self.assertEqual(bed.identity_sha256(), identity)
        current = bed.stage_copy()
        self.assertRefused("time_mismatch", bed.stage_scheduler_state, current, mobile, time_s=T0)
        bed.stage_scheduler_state(current, mobile, time_s=T0 + 5.0)

    def test_staged_water_load_refuses(self) -> None:
        bed = build_dense_bed(**fixture(6))
        start = bed.state
        removal = 0.25 * start.sediment_availability.available_mass_kg
        loaded, _ = apply_bed_demand(start, bed.context, WaterProcessDemand(removal, np.zeros_like(removal)))
        bed.adopt_bed_state(loaded, time_s=T0 + 60.0)
        stage = bed.stage_copy()
        self.assertRefused("water_mobile_not_empty", bed.stage_scheduler_state, stage,
                           zeros_mobile_cohort_state(8), time_s=T0 + 60.0)
        self.assertIs(bed.state, loaded)


class ResourceChecks(unittest.TestCase):
    """B4: the host-memory cap refuses from shapes alone before any domain copy or native probe."""

    def observed_build(self, inputs):
        events: list[tuple[str, ...]] = []
        owned, probe = adapter._owned_array, adapter._probe_per_cell_bytes

        def observe_owned(*args, **kwargs):
            events.append(("owned", args[-1]))
            return owned(*args, **kwargs)

        def observe_probe(*args, **kwargs):
            events.append(("probe",))
            return probe(*args, **kwargs)

        bed, error = None, None
        with mock.patch.object(adapter, "_owned_array", side_effect=observe_owned), \
                mock.patch.object(adapter, "_probe_per_cell_bytes", side_effect=observe_probe):
            try:
                bed = build_dense_bed(**inputs)
            except DenseBedError as exc:
                error = exc
        self.assertIs(adapter._owned_array, owned)
        self.assertIs(adapter._probe_per_cell_bytes, probe)
        return bed, error, events

    def test_b4_cap_refuses_before_domain_copies_and_probe_and_normal_build_remains_valid(self) -> None:
        lower = estimate_initialization_bytes(NY, NX, NV, 2, 0)
        for label, cap, expected_events in (("tiny", 1, []), ("at_shape_lower_bound", lower, [("probe",)])):
            with self.subTest(cap=label):
                inputs = fixture(2)
                inputs["limits"] = dataclasses.replace(inputs["limits"], host_memory_cap_bytes=cap)
                snapshot = {k: v.copy() for k, v in inputs.items() if isinstance(v, np.ndarray)}
                bed, error, events = self.observed_build(inputs)
                self.assertIsNone(bed)
                self.assertIsNotNone(error)
                self.assertEqual(error.code, "host_memory_cap_exceeded", error.detail)
                self.assertEqual(events, expected_events)
                for key, value in snapshot.items():
                    np.testing.assert_array_equal(inputs[key], value)
        bed, error, events = self.observed_build(fixture(2))
        self.assertIsNone(error)
        self.assertEqual(events[0], ("probe",))
        self.assertEqual(events[1:], [("owned", "initial_bed_elevation_m"), ("owned", "initial_class_fractions"),
                                      ("owned", "initial_available_fraction"), ("owned", "initial_water_depth_m")])
        receipt = bed.initialization_receipt
        self.assertEqual(receipt["host_memory_lower_bound_bytes"], lower)
        self.assertGreater(receipt["estimated_initialization_bytes"], lower)
        self.assertEqual(receipt["estimated_initialization_bytes"],
                         estimate_initialization_bytes(NY, NX, NV, 2,
                                                       receipt["measured_per_cell_ledger_committed_bytes"]))
        bed.verify_case_inputs()
        self.assertEqual(bed.case_sha256, build_dense_bed(**fixture(2)).case_sha256)


if __name__ == "__main__":
    unittest.main()
