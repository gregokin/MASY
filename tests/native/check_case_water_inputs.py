"""Focused actual-Native CPU interface checks; run only by the run owner.

All numeric arrays and coefficients below are labelled synthetic interface
fixtures, not scientific defaults or a production calibration. No event is
stepped. The evolving-bed checks use genuine Native pickup/ledger accounting.
No Native function, class or physics is replaced or patched.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
import sys
import unittest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from maple.core.boundaries import AxisBoundary, BoundaryKind
from maple.core.parameters.avalanching import AvalanchingSpec
from maple.core.parameters.geometry import GeometrySpec
from maple.core.parameters.grain_classes import GrainClass, GrainClassSet
from maple.core.parameters.topographic_commit import TopographicCommitSpec
from maple.core.types.threshold_modifiers import neutral_threshold_modifier_state
from maple.core.types.vegetation import uniform_vegetation_raster
from maple.water.interfaces import WaterProcessDemand
from maple_syrup.complete_event import CompletionPolicy
from maple_syrup.infiltration import column_parameters, initial_soil_water_m
from maple_syrup.rainfall import rainfall_field
from maple_syrup.routing import build_routing_graph
from maple_syrup.sediment_bed import apply_bed_demand, terrain_reference
from maple_syrup.sediment_event import SedimentEventControl
from maple_syrup.sediment_physics import sediment_physics_parameters
from maple_syrup.storm import StormControl

from masy.case_water import (CaseWaterCopyLimits, CaseWaterError, CaseWaterSpec,
                             CaseWetClass, prepare_case_water)
from masy.dense_bed import DenseBedLimits, SHARED_FIELDS, build_dense_bed
from masy.dense_runner import DenseRunnerControl

NY, NX, NV, K = 3, 5, 4, 2
T0 = 1234.5678
LIMITS = CaseWaterCopyLimits(max_cells=15, max_classes=2,
    max_known_host_bytes=8 << 20, max_metadata_bytes=1 << 20)


def _bed(densities=(2650.0, 2650.0)):
    """Non4x4, stratified, partly available, actual two-class Native bed."""
    classes = GrainClassSet(classes=tuple(GrainClass(class_id=key,
        diameter_m=diameter, particle_density_kg_m3=density, is_aggregate=False)
        for key, diameter, density in zip(("fine", "coarse"), (1e-4, 3e-4), densities, strict=True)))
    boundary = AxisBoundary(kind=BoundaryKind.PRESCRIBED,
        inflow_flux_kg_m_s={c.class_id: 0.0 for c in classes.classes})
    geometry = GeometrySpec(nx=NX, ny=NY, dx_m=0.5, dy_m=0.5,
        boundary_x=boundary, boundary_y=boundary, voxel_dz_m=0.05,
        bulk_density_kg_m3=1250.0, active_layer_thickness_m=0.002)
    rows, cols = np.indices((NY, NX)).astype(np.float64)
    fractions = np.empty((NY, NX, K), dtype=np.float64)
    fractions[..., 0] = 0.3 + 0.1 * (cols % 2)
    fractions[..., 1] = 1.0 - fractions[..., 0]
    available = np.empty_like(fractions)
    available[..., 0], available[..., 1] = 0.35, 0.6
    return build_dense_bed(geometry=geometry, grain_classes=classes,
        mass_resolution_kg=1e-10, n_voxels=NV,
        initial_bed_elevation_m=0.12 + 0.01 * rows + 0.005 * cols,
        initial_class_fractions=fractions, initial_available_fraction=available,
        initial_water_depth_m=np.full((NY, NX), 0.002, dtype=np.float64),
        vegetation=uniform_vegetation_raster(NY, NX, height_m=0.3, cover_fraction=0.25),
        threshold_modifier_state=neutral_threshold_modifier_state(NY, NX),
        topographic_commit_spec=TopographicCommitSpec(),
        avalanching_spec=AvalanchingSpec(enabled=False),
        active_mask=np.ones((NY, NX), dtype=bool), initial_time_s=T0,
        limits=DenseBedLimits(host_memory_cap_bytes=1 << 30,
            max_voxels_per_column=64, stage_copy_max_bytes=1 << 24))


def _spec(bed, *, masked=False):
    """Explicit synthetic south outlet; no values are user/case defaults."""
    rows, cols = np.indices((NY, NX)).astype(np.float64)
    z = np.full((NY + 2, NX + 2), 1.0, dtype=np.float64)
    z[1:-1, 1:-1] = 0.12 + 0.01 * rows + 0.005 * cols
    z[0, 1:-1] = z[1, 1:-1] - 0.02
    exports = np.zeros(z.shape, dtype=bool)
    exports[0, 1:-1] = True
    active = np.ones((NY, NX), dtype=bool)
    if masked:
        active[-1, -1] = False  # no active cell drains into this high corner
    def grid(value):
        return np.full((NY, NX), value, dtype=np.float64)
    return CaseWaterSpec(profile_id="synthetic-interface-v1",
        source_reference="authored:test_case_water_inputs;not-production-calibration",
        elevation_full_m=z, export_receiver_full=exports, active_mask=active,
        friction_factor=grid(1.0), allow_masked_nodata=False, nodata_value=None,
        allow_pit_storage=False, allow_flat_storage=False,
        soil_model="fixed_ksat", ksat_m_per_s=grid(1e-5), suction_m=grid(0.01),
        drainage_parameter=grid(0.1), theta_sat=grid(0.4), soil_thickness_m=grid(0.2),
        pavement_cover_fraction=None, initial_theta=grid(0.1), retained_soil_water_m=None,
        rainfall_multiplier=np.ascontiguousarray(1.0 + 0.1 * rows),
        # Reverse authoring order deliberately. Native bed order is unchanged.
        wet_classes=(CaseWetClass("coarse", 2.0, 1.1, 0.5, 0.02, 0.7),
                     CaseWetClass("fine", 1.0, 1.2, 0.4, 0.01, 0.6)),
        active_layer_sensitivity_mm=1.0, ke_model="verstraeten_exp",
        kinematic_viscosity_m2_s=1.003e-6, reference_interval_s=1.0,
        recession_factor_per_reference_s=0.9, ke_vegetation_form="legacy_literal",
        raindrop_composition_scaling="legacy_none", distance_convention="legacy_literal",
        dstar_convention="legacy_sigma_minus_one", bagnold_depth_units="legacy_mm")


def _control():
    return DenseRunnerControl(sediment_control=SedimentEventControl(
        storm=StormControl(max_steps=10_000, implementation="array"),
        transport_scheme="upwind", transport_implementation="array"),
        completion_policy=CompletionPolicy(deposit_mobile_at_flow_end=True), report_every_s=10.0)


def _caller_snapshot(spec):
    return {f.name: (value.copy(), bool(value.flags.writeable))
        for f in dataclasses.fields(spec) if isinstance(value := getattr(spec, f.name), np.ndarray)}


def _direct_native(bed, spec):
    """Independent direct public factory calls, no wrapper helper/formula."""
    graph = build_routing_graph(spec.elevation_full_m.copy(), spec.export_receiver_full.copy(),
        spec.friction_factor.copy(), bed.context.geometry.dx_m,
        dy_m=bed.context.geometry.dy_m, active_mask=spec.active_mask.copy(), xp=np,
        nodata_value=spec.nodata_value, allow_masked_nodata=spec.allow_masked_nodata,
        allow_pit_storage=spec.allow_pit_storage, allow_flat_storage=spec.allow_flat_storage)
    terrain = terrain_reference(spec.elevation_full_m, spec.export_receiver_full,
        bed.state, graph, nodata_value=spec.nodata_value)
    column = column_parameters(model=spec.soil_model, ksat_m_per_s=spec.ksat_m_per_s,
        suction_m=spec.suction_m, drainage_parameter=spec.drainage_parameter,
        theta_sat=spec.theta_sat, soil_thickness_m=spec.soil_thickness_m,
        pavement_cover_fraction=spec.pavement_cover_fraction, active_mask=spec.active_mask)
    rain = rainfall_field(NY, NX, scale=spec.rainfall_multiplier, active_mask=spec.active_mask, xp=np)
    lookup = {row.class_id: row for row in spec.wet_classes}
    names = ("raindrop_a", "raindrop_b", "raindrop_c", "raindrop_max_depth_mm",
             "raindrop_depth_attenuation_per_cm")
    vectors = {name: [getattr(lookup[c.class_id], name) for c in bed.context.grain_classes.classes]
               for name in names}
    sediment = sediment_physics_parameters(
        diameter_m=[c.diameter_m for c in bed.context.grain_classes.classes], **vectors,
        particle_density_g_cm3=bed.context.grain_classes.classes[0].particle_density_kg_m3 / 1000.0,
        active_layer_sensitivity_mm=spec.active_layer_sensitivity_mm, ke_model=spec.ke_model,
        kinematic_viscosity_m2_s=spec.kinematic_viscosity_m2_s,
        reference_interval_s=spec.reference_interval_s,
        recession_factor_per_reference_s=spec.recession_factor_per_reference_s,
        ke_vegetation_form=spec.ke_vegetation_form,
        raindrop_composition_scaling=spec.raindrop_composition_scaling,
        distance_convention=spec.distance_convention, dstar_convention=spec.dstar_convention,
        bagnold_depth_units=spec.bagnold_depth_units, xp=np)
    soil = (initial_soil_water_m(column, spec.initial_theta) if spec.initial_theta is not None
            else spec.retained_soil_water_m.copy())
    return terrain, column, rain, sediment, soil


class CaseWaterChecks(unittest.TestCase):
    def assert_native_equal(self, actual, expected):
        self.assertIs(type(actual), type(expected))
        for field in dataclasses.fields(expected):
            a, b = getattr(actual, field.name), getattr(expected, field.name)
            if isinstance(b, np.ndarray):
                np.testing.assert_array_equal(a, b, err_msg=field.name)
            else:
                self.assertEqual(a, b, field.name)

    def assert_callers_unchanged(self, spec, snapshot):
        for name, (value, writable) in snapshot.items():
            actual = getattr(spec, name)
            np.testing.assert_array_equal(actual, value, err_msg=name)
            self.assertEqual(bool(actual.flags.writeable), writable, name)

    def assert_refused_without_bed_change(self, bed, spec, code, *, limits=LIMITS):
        before, state = bed.identity_record(), bed.state
        callers = _caller_snapshot(spec)
        with self.assertRaises(CaseWaterError) as caught:
            prepare_case_water(bed, spec, _control(), limits=limits)
        self.assertEqual(caught.exception.code, code)
        self.assertIs(bed.state, state)
        self.assertEqual(bed.identity_record(), before)
        self.assert_callers_unchanged(spec, callers)

    def test_non4x4_two_class_matches_direct_native_without_reset(self):
        bed = _bed()
        removal = 0.1 * bed.state.sediment_availability.available_mass_kg
        evolved, result = apply_bed_demand(bed.state, bed.context,
            WaterProcessDemand(removal, np.zeros_like(removal)))
        self.assertTrue(bool((result.actual_removal_by_cell_class_kg > 0).any()))
        bed.adopt_bed_state(evolved, time_s=T0 + 2.0)
        spec, control = _spec(bed), _control()
        before, held, context, vegetation = bed.identity_record(), bed.state, bed.context, bed.vegetation
        callers = _caller_snapshot(spec)
        prepared = prepare_case_water(bed, spec, control, limits=LIMITS)
        direct = _direct_native(bed, spec)
        for actual, expected in zip((prepared.water.terrain, prepared.water.column,
                                    prepared.water.field, prepared.water.sediment), direct[:4], strict=True):
            self.assert_native_equal(actual, expected)
        np.testing.assert_array_equal(prepared.water.initial_soil_water_m, direct[4])
        self.assertIs(prepared.control, control)
        self.assertIs(prepared.verify(bed), prepared)
        self.assertIs(bed.state, held)
        self.assertIs(bed.context, context)
        self.assertIs(bed.vegetation, vegetation)
        self.assertEqual(bed.time_s.hex(), (T0 + 2.0).hex())
        for name in SHARED_FIELDS:
            self.assertIs(getattr(bed.state, name), getattr(held, name), name)
        self.assertEqual(bed.identity_record(), before)
        self.assert_callers_unchanged(spec, callers)
        self.assertIs(prepared.water.vegetation_cover_fraction, vegetation.raster.cover_fraction)
        np.testing.assert_array_equal(prepared.water.sediment.raindrop_a, [1.0, 2.0])
        np.testing.assert_array_equal(prepared.water.sediment.raindrop_depth_attenuation_per_cm, [0.6, 0.7])
        record = prepared.identity_record()
        self.assertEqual(record["native_graph_calls_in_preparation"], 2)
        self.assertEqual(record["known_array_plan"]["shared_bed_copied_bytes"], 0)
        self.assertFalse(record["whole_process_fit_claim"])
        record["profile_id"] = "detached-record-edit"
        self.assertEqual(prepared.identity_record()["profile_id"], spec.profile_id)

    def test_mask_is_shared_and_rain_is_zero_outside(self):
        bed = _bed()
        spec = _spec(bed, masked=True)
        prepared = prepare_case_water(bed, spec, _control(), limits=LIMITS)
        np.testing.assert_array_equal(prepared.water.column.active_mask, spec.active_mask)
        np.testing.assert_array_equal(prepared.water.terrain.active_mask, spec.active_mask)
        np.testing.assert_array_equal(prepared.water.field.multiplier[~spec.active_mask], [0.0])
        np.testing.assert_array_equal(prepared.water.initial_soil_water_m,
                                     initial_soil_water_m(prepared.water.column, spec.initial_theta))

    def test_retained_soil_is_explicit_owned_and_not_reset(self):
        bed = _bed()
        retained = np.linspace(0.01, 0.03, NY * NX, dtype=np.float64).reshape(NY, NX)
        spec = dataclasses.replace(_spec(bed), initial_theta=None, retained_soil_water_m=retained)
        prepared = prepare_case_water(bed, spec, _control(), limits=LIMITS)
        np.testing.assert_array_equal(prepared.water.initial_soil_water_m, retained)
        self.assertFalse(np.shares_memory(prepared.water.initial_soil_water_m, retained))
        self.assertFalse(prepared.water.initial_soil_water_m.flags.writeable)

    def test_missing_extra_duplicate_class_ids_are_refused(self):
        bed = _bed()
        spec = _spec(bed)
        for rows in (spec.wet_classes[:1],
                     (dataclasses.replace(spec.wet_classes[0], class_id="extra"), spec.wet_classes[1]),
                     (spec.wet_classes[0], spec.wet_classes[0])):
            with self.subTest(rows=rows):
                self.assert_refused_without_bed_change(bed,
                    dataclasses.replace(spec, wet_classes=rows), "class_mismatch")

    def test_common_density_and_exact_unit_roundtrip_are_required(self):
        for densities in ((2650.0, 2700.0), (math.nextafter(2000.0, math.inf),) * 2):
            with self.subTest(densities=densities):
                bed = _bed(densities)
                self.assert_refused_without_bed_change(bed, _spec(bed), "density_mismatch")

    def test_inactive_native_mobile_inventory_is_refused_without_discard(self):
        bed = _bed()
        removal = np.zeros((NY, NX, K), dtype=np.float64)
        removal[-1, -1] = 0.1 * bed.state.sediment_availability.available_mass_kg[-1, -1]
        evolved, result = apply_bed_demand(bed.state, bed.context,
            WaterProcessDemand(removal, np.zeros_like(removal)))
        bed.adopt_bed_state(evolved, time_s=T0 + 1.0)
        self.assertTrue(bool((result.actual_removal_by_cell_class_kg[-1, -1] > 0).any()))
        self.assert_refused_without_bed_change(bed, _spec(bed, masked=True), "inactive_mobile_mass")

    def test_mutated_caller_grid_and_authored_metadata_are_detected(self):
        bed = _bed()
        spec = _spec(bed)
        prepared = prepare_case_water(bed, spec, _control(), limits=LIMITS)
        spec.rainfall_multiplier[0, 0] = 2.0
        with self.assertRaisesRegex(CaseWaterError, "controlling_input_changed"):
            prepared.verify()
        fresh_spec = _spec(bed)
        fresh = prepare_case_water(bed, fresh_spec, _control(), limits=LIMITS)
        object.__setattr__(fresh_spec, "source_reference", "changed-authored-reference")
        with self.assertRaisesRegex(CaseWaterError, "controlling_input_changed"):
            fresh.verify()

    def test_mutated_native_returned_array_is_detected(self):
        bed = _bed()
        prepared = prepare_case_water(bed, _spec(bed), _control(), limits=LIMITS)
        array = prepared.water.column.ksat_m_per_s
        array.flags.writeable = True
        array[0, 0] = 2e-5
        with self.assertRaisesRegex(CaseWaterError, "controlling_input_changed"):
            prepared.verify()

    def test_limits_and_ambiguous_soil_refuse_without_changes(self):
        bed = _bed()
        spec = _spec(bed)
        self.assert_refused_without_bed_change(bed, spec, "known_copy_limit",
            limits=dataclasses.replace(LIMITS, max_known_host_bytes=1))
        self.assert_refused_without_bed_change(bed, spec, "shape_limit",
            limits=dataclasses.replace(LIMITS, max_cells=NY * NX - 1))
        self.assert_refused_without_bed_change(bed,
            dataclasses.replace(spec, retained_soil_water_m=np.zeros((NY, NX))), "ambiguous_initial_soil")
        self.assert_refused_without_bed_change(bed,
            dataclasses.replace(spec, initial_theta=None), "ambiguous_initial_soil")
        self.assert_refused_without_bed_change(bed, spec, "metadata_limit",
            limits=dataclasses.replace(LIMITS, max_metadata_bytes=64))
        before = bed.identity_record()
        control = _control()
        for invalid in (
            dataclasses.replace(control, completion_policy=CompletionPolicy(hold_s=0.0)),
            dataclasses.replace(control, sediment_control=dataclasses.replace(
                control.sediment_control, storm=dataclasses.replace(control.sediment_control.storm, max_steps=0))),
            dataclasses.replace(control, sediment_control=dataclasses.replace(control.sediment_control, storm=None)),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(CaseWaterError) as caught:
                prepare_case_water(bed, spec, invalid, limits=LIMITS)
            self.assertEqual(caught.exception.code, "invalid_control")
            self.assertEqual(bed.identity_record(), before)

    def test_native_invalid_soil_refusal_preserves_callers(self):
        bed = _bed()
        spec = dataclasses.replace(_spec(bed), initial_theta=None,
            retained_soil_water_m=np.full((NY, NX), 0.5, dtype=np.float64))
        self.assert_refused_without_bed_change(bed, spec, "invalid_initial_soil")


if __name__ == "__main__":
    unittest.main()
