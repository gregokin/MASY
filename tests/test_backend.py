"""Native metadata-only fixtures. No terrain input, transport, device or simulation."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _support  # noqa: F401 - candidate MASY, never a live implementation tree

from masy import backend


def native():
    """The test driver supplies the exact locked native CPU sources on sys.path."""
    return backend._native()


def bed_fixture():
    n = native()
    from maple.core.types.sediment_availability import zeros_sediment_availability_state
    from maple.core.types.sediment_ledger import zeros_sediment_ledger_state

    z = n.np.zeros
    # Deliberately no native initialization/validation/physical bed operation:
    # metadata admission must also describe a placeholder without accepting it
    # as a physically valid landscape.
    return n.BedState(n.VoxelColumnState(z((1, 2, 3, 2))), n.ActiveLayerState(z((1, 2, 2))),
                      n.WaterState(z((1, 2)), z((1, 2, 2))), zeros_sediment_ledger_state(1, 2, 2),
                      n.CommittedTopographyState(z((1, 2)), z((1, 2)), z((1, 2, 2)), z((1, 2, 2)), 0.0, 0),
                      zeros_sediment_availability_state(1, 2, 2))


def terrain_fixture():
    n = native()
    return n.TerrainReference(n.np.zeros((3, 4)), n.np.zeros((1, 2)), n.np.zeros((3, 4), dtype=bool),
                              n.np.ones((1, 2), dtype=bool), n.np.ones((1, 2)), 1.0)


class BackendMetadataTests(unittest.TestCase):
    def code(self, code, call, *args, **kwargs):
        with self.assertRaises(backend.BackendMetadataError) as ctx:
            call(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code)

    def test_import_is_lazy_in_a_fresh_isolated_process(self):
        script = ("import json,sys; sys.path.insert(0, sys.argv[1]); import masy.backend; "
                  "print(json.dumps([n for n in sys.modules if n.split('.')[0] "
                  "in ('numpy','maple','maple_syrup','cupy','pynvml')]))")
        r = subprocess.run([sys.executable, "-I", "-B", "-c", script,
                            str(Path(backend.__file__).parents[1])], capture_output=True, text=True, timeout=20,
                           check=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout), [])

    def test_unsupported_requests_refuse_before_native_import(self):
        with mock.patch.object(backend, "_native", side_effect=AssertionError("native import")):
            for value in ("cupy", "cuda", "auto", None, True):
                with self.subTest(value=value):
                    self.code("unsupported_backend", backend.inspect_dense_metadata, object(), backend=value)
            for value in ("windowed", "compact", None, True):
                with self.subTest(value=value):
                    self.code("unsupported_representation", backend.inspect_dense_metadata, object(),
                              representation=value)

    def test_native_bed_metadata_has_exact_walk_and_inventory_without_transfers(self):
        n, bed = native(), bed_fixture()
        from maple.core.backend import read_transfer_counters

        before, refs = read_transfer_counters(), tuple(n.array_leaves(bed))
        with mock.patch("maple.core.backend.arrays.to_host", side_effect=AssertionError("to_host")), \
             mock.patch("maple.core.backend.arrays.to_device", side_effect=AssertionError("to_device")), \
             mock.patch("copy.deepcopy", side_effect=AssertionError("deepcopy")):
            result = backend.inspect_dense_metadata(bed)
        self.assertEqual(len(result.arrays), len(refs))
        self.assertEqual(dict(result.placement_inventory), n.array_leaf_inventory((bed,)))
        self.assertTrue(all(a is b for a, b in zip(refs, n.array_leaves(bed), strict=True)))
        self.assertTrue(all(v == 0 for v in dataclasses.asdict(read_transfer_counters().delta(before)).values()))
        self.assertFalse(result.as_record()["transport_ready"])
        self.assertFalse(result.as_record()["values_validated"])
        self.assertNotIn("cupy", sys.modules)

    def test_optional_native_fields_and_host_evidence_are_classified_without_values(self):
        n, bed, terrain = native(), bed_fixture(), terrain_fixture()
        from maple.core.types.threshold_modifiers import (
            neutral_threshold_modifier_state,
        )
        from maple.core.types.vegetation import empty_vegetation_state

        veg, thresholds, soil = empty_vegetation_state(1, 2), neutral_threshold_modifier_state(1, 2), n.np.zeros((1, 2))
        result = backend.inspect_dense_metadata(bed, vegetation=veg, threshold_modifier_state=thresholds,
                                                soil_water_m=soil, terrain=terrain)
        self.assertEqual(sum(r[0] == "resident" for r in result.arrays),
                         len(n.array_leaves(bed, veg, thresholds, soil)))
        self.assertEqual(sum(r[0] == "host_evidence" for r in result.arrays), len(n.array_leaves(terrain)))
        self.assertEqual(dict(result.placement_inventory), n.array_leaf_inventory((bed, veg, thresholds, soil, terrain)))

    def test_exact_array_aliases_are_native_placement_visits(self):
        n, bed = native(), bed_fixture()
        bed = dataclasses.replace(bed, water=dataclasses.replace(bed.water,
                                  mobile_mass_by_cell_class_kg=bed.active_layer.mass_kg))
        result = backend.inspect_dense_metadata(bed)
        inventory = dict(result.placement_inventory)
        self.assertEqual(inventory, n.array_leaf_inventory((bed,)))
        self.assertGreater(inventory["n_aliased_leaves"], 0)
        self.assertGreater(inventory["transfer_bytes"], inventory["unique_bytes"])
        self.assertEqual(len(result.arrays), len(n.array_leaves(bed)))

    def test_one_array_cannot_be_classified_as_both_state_and_evidence(self):
        bed = bed_fixture()
        terrain = dataclasses.replace(terrain_fixture(), initial_committed_elevation_m=bed.water.depth_m)
        self.code("ambiguous_role", backend.inspect_dense_metadata, bed, terrain=terrain)

    def test_foreign_root_and_unknown_array_holders_refuse(self):
        n = native()
        self.code("unsupported_holder", backend.inspect_dense_metadata, SimpleNamespace(voxel_column=None))
        bed = bed_fixture()
        object.__setattr__(bed.active_layer, "mass_kg", SimpleNamespace(shape=(1, 2, 2), dtype=n.np.dtype("f8")))
        self.code("unsupported_holder", backend.inspect_dense_metadata, bed)

    def test_array_subclasses_and_object_arrays_refuse(self):
        n = native()

        class ArraySubclass(n.np.ndarray):
            pass

        for value, code in ((n.np.zeros((1, 2, 2)).view(ArraySubclass), "unsupported_holder"),
                            (n.np.empty((1, 2, 2), dtype=object), "unsupported_array")):
            with self.subTest(code=code):
                bed = bed_fixture()
                object.__setattr__(bed.active_layer, "mass_kg", value)
                self.code(code, backend.inspect_dense_metadata, bed)

    def test_windowed_and_compact_native_types_refuse_without_reading_them(self):
        from maple.core.types.sediment_ledger import CompactSedimentLedgerState
        from maple.core.types.voxel import WindowedVoxelColumnState

        for name, cls in (("voxel_column", WindowedVoxelColumnState), ("ledger", CompactSedimentLedgerState)):
            with self.subTest(name=name):
                bed = bed_fixture()
                object.__setattr__(bed, name, object.__new__(cls))
                self.code("unsupported_holder", backend.inspect_dense_metadata, bed)

    def test_missing_extra_and_array_valued_scalar_fields_refuse(self):
        n, bed = native(), bed_fixture()
        object.__delattr__(bed, "active_layer")
        self.code("missing_field", backend.inspect_dense_metadata, bed)
        bed = bed_fixture()
        object.__setattr__(bed.voxel_column, "unexpected", n.np.zeros(1))
        self.code("unsupported_schema", backend.inspect_dense_metadata, bed)
        bed = bed_fixture()
        object.__setattr__(bed.committed_topography, "last_commit_time_s", n.np.zeros(1))
        self.code("unsupported_holder", backend.inspect_dense_metadata, bed)

    def test_deleted_defaulted_instance_fields_do_not_fall_back_to_class_defaults(self):
        terrain = terrain_fixture()
        object.__delattr__(terrain, "allow_pit_storage")
        self.code("missing_field", backend.inspect_dense_metadata, bed_fixture(), terrain=terrain)

    def test_numpy_scalar_metadata_is_outside_the_exact_python_scalar_schema(self):
        n, bed = native(), bed_fixture()
        object.__setattr__(bed.ledger, "largest_residual_kg", n.np.float64(0.0))
        self.code("unsupported_holder", backend.inspect_dense_metadata, bed)

    def test_native_schema_init_flag_change_refuses_before_inventory(self):
        n, bed, fields = native(), bed_fixture(), dataclasses.fields

        def altered(obj):
            actual = fields(obj)
            if type(obj) is n.VoxelColumnState:
                return (*actual, SimpleNamespace(name="new_cache", init=False))
            return actual

        with mock.patch.object(backend.dataclasses, "fields", side_effect=altered), \
             mock.patch("maple.core.backend.arrays.array_leaf_inventory", side_effect=AssertionError("inventory")):
            self.code("unsupported_schema", backend.inspect_dense_metadata, bed)

    def test_native_walkers_init_false_difference_is_explicit_not_a_new_copy_rule(self):
        n = native()

        @dataclasses.dataclass(frozen=True)
        class Fixture:
            first: object
            second: object
            hidden: object = dataclasses.field(default=None, init=False)

        a = n.np.zeros(3)
        obj = Fixture(a, a)
        object.__setattr__(obj, "hidden", n.np.zeros(2))
        self.assertEqual(len(n.array_leaves(obj)), 3)
        self.assertEqual(n.array_leaf_inventory(obj), {"n_leaves": 2, "transfer_bytes": 48,
                          "n_unique_leaves": 1, "unique_bytes": 24, "n_aliased_leaves": 1, "non_memoizing": True})
        self.code("unsupported_holder", backend.inspect_dense_metadata, obj)

    def test_no_numerical_validation_or_value_scan_is_claimed(self):
        n, bed = native(), bed_fixture()
        bed.active_layer.mass_kg.fill(float("nan"))
        with mock.patch.object(n.np, "asarray", side_effect=AssertionError("conversion")), \
             mock.patch.object(n.np, "isfinite", side_effect=AssertionError("value scan")):
            record = backend.inspect_dense_metadata(bed).as_record()
        self.assertFalse(record["values_validated"])
        self.assertFalse(record["transport_ready"])

    def test_unsupported_optional_roots_and_objects_vegetation_refuse(self):
        bed = bed_fixture()
        for name in ("terrain", "soil_water_m", "vegetation", "threshold_modifier_state"):
            with self.subTest(name=name):
                self.code("unsupported_holder", backend.inspect_dense_metadata, bed, **{name: object()})
        from maple.core.types.vegetation import empty_vegetation_state
        veg = dataclasses.replace(empty_vegetation_state(1, 2), source="objects")
        self.code("unsupported_vegetation", backend.inspect_dense_metadata, bed, vegetation=veg)

    def test_provenance_is_unverified_frozen_caller_metadata(self):
        p = backend.BackendProvenance(source_lock_id="caller-label", interpreter="caller-label",
                                     physical_device_uuid="unverified", visible_device_ordinal=7, visible_devices="7")
        result = backend.inspect_dense_metadata(bed_fixture(), provenance=p)
        record = result.as_record()
        self.assertEqual(record["provenance"]["physical_device_uuid"], "unverified")
        self.assertFalse(record["source_binding_verified"] or record["device_qualified"])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.provenance = backend.BackendProvenance()
        record["transport_ready"] = True
        self.assertFalse(result.as_record()["transport_ready"])
        self.code("invalid_provenance", backend.BackendProvenance, visible_device_ordinal=True)
        self.code("invalid_provenance", backend.inspect_dense_metadata, bed_fixture(), provenance={})

    def test_tampered_provenance_cannot_smuggle_arrays_into_record_copying(self):
        n, bed = native(), bed_fixture()
        p = backend.BackendProvenance(source_lock_id="unverified")
        object.__setattr__(p, "source_lock_id", n.np.zeros(1))
        self.code("invalid_provenance", backend.inspect_dense_metadata, bed, provenance=p)
        clean = backend.inspect_dense_metadata(bed)
        object.__setattr__(clean.provenance, "unexpected", n.np.zeros(1))
        with mock.patch("copy.deepcopy", side_effect=AssertionError("array copied")):
            self.code("invalid_provenance", clean.as_record)


if __name__ == "__main__":
    unittest.main()
