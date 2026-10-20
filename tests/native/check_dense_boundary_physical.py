"""Independent actual-native tests of the bounded Dense physical bridge.

Run only through the task's explicit -I -B origin-verified CPU prototype
harness. No source/package/provenance guard is faked. Mocked calls below are
only sentinels proving refusal order and forbidden reconstruction paths.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import pkgutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fixture_dense_alternation import empty_pool, wind_config
from fixture_dense_boundary_physical import (
    bed_exact,
    exact,
    initial_boundary,
    leaves,
    pending_boundary,
)
from maple.aeolian.scheduler import physical_checkpoint as native
from maple.core.types.ledger_reconciliation import (
    ledger_reconciliation_baseline_from_committed_state,
)

from masy import dense_bed
from masy import dense_checkpoint as bridge
from masy.dense_bed import DenseBed

CAP = 32 << 20
CALLER = {"profile": "source-manifest prototype only", "run": "authored-native-bridge"}


class PhysicalBridgeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bed, cls.pool, cls.config, cls.receipt = pending_boundary()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "physical"
        self.inputs = bridge.DenseBoundaryInputs.capture(self.bed)

    def packed(self, bed=None, pool=None, config=None, inputs=None, cap=CAP):
        return bridge.pack_boundary_physical(
            bed or self.bed, self.pool if pool is None else pool, config or self.config,
            CALLER, inputs=self.inputs if inputs is None else inputs, max_bytes=cap)

    def save(self):
        packed = self.packed()
        native.save_exact_physical_checkpoint(self.path, packed.physical, self.config,
                                              packed.identity, max_bytes=CAP)
        return packed

    def load(self, identity, inputs=None, cap=CAP):
        return bridge.load_boundary_physical(self.path, self.config, identity,
            inputs=self.inputs if inputs is None else inputs, max_bytes=cap)

    def native_variant(self, physical, identity, name):
        path = Path(self.temp.name) / name
        native.save_exact_physical_checkpoint(path, physical, self.config, identity, max_bytes=CAP)
        return path

    def test_real_pending_ledger_roundtrip_is_exact_and_initial_metadata_stays_initial(self):
        before = bed_exact(self.bed, self.pool)
        packed = self.save()
        self.assertIsNone(packed.physical.ledger_reconciliation_baseline)
        self.assertIsNone(packed.physical.coupled_trajectory_semantics)
        for name in ("voxel_column", "active_layer", "water", "ledger", "committed_topography", "sediment_availability"):
            self.assertIs(getattr(packed.physical, name), getattr(self.bed.state, name))
        restored = self.load(packed.identity)
        self.assertEqual(bed_exact(restored.bed, restored.mobile_state), before)
        self.assertEqual(bed_exact(self.bed, self.pool), before)
        self.assertLess(restored.bed.state.committed_topography.last_commit_time_s, restored.bed.time_s)
        self.assertNotEqual(restored.bed.time_s.hex(), 0.0.hex())
        self.assertEqual(restored.bed.case_record["initial_time_s"], self.bed.case_record["initial_time_s"])
        self.assertEqual(restored.bed.initialization_receipt, self.bed.initialization_receipt)
        self.assertIsNot(restored.bed._owner, self.bed._owner)
        for leaf in leaves(restored.bed.state).values():
            if type(leaf) is np.ndarray:
                self.assertFalse(leaf.flags.writeable)

    def test_initial_boundary_roundtrip_is_exact(self):
        bed, pool, config = initial_boundary()
        inputs = bridge.DenseBoundaryInputs.capture(bed)
        packed = self.packed(bed, pool, config, inputs)
        native.save_exact_physical_checkpoint(self.path, packed.physical, config, packed.identity, max_bytes=CAP)
        restored = bridge.load_boundary_physical(self.path, config, packed.identity, inputs=inputs, max_bytes=CAP)
        self.assertEqual(bed_exact(restored.bed, restored.mobile_state), bed_exact(bed, pool))

    def test_nonzero_awkward_epoch_and_prescribed_wet_initial_fields_are_not_reset(self):
        from check_dense_bed import fixture

        from masy.dense_bed import build_dense_bed
        bed = build_dense_bed(**fixture(2))
        pool, config = empty_pool(), wind_config(bed)
        inputs = bridge.DenseBoundaryInputs.capture(bed)
        packed = self.packed(bed, pool, config, inputs)
        native.save_exact_physical_checkpoint(self.path, packed.physical, config, packed.identity, max_bytes=CAP)
        restored = bridge.load_boundary_physical(self.path, config, packed.identity, inputs=inputs, max_bytes=CAP)
        self.assertEqual(bed_exact(restored.bed, restored.mobile_state), bed_exact(bed, pool))
        self.assertGreater(restored.bed.time_s, 0.0)
        self.assertTrue(bool(np.any(restored.bed.state.water.depth_m > 0.0)))
        self.assertTrue(bool(np.any(restored.bed.vegetation.raster.cover_fraction > 0.0)))
        self.assertFalse(bool(np.all(restored.bed.threshold_modifier_state.physical_crust_multiplier == 1.0)))

    def test_restore_never_reconstructs_commits_finalizes_resets_or_transports(self):
        packed = self.save()
        forbidden = (
            "masy.dense_bed.build_dense_bed", "masy.dense_runner.DenseRunner.__init__",
            "maple.core.types.voxel.zeros_voxel_column_state",
            "maple.core.types.sediment_ledger.zeros_sediment_ledger_state",
            "maple.surface.voxels.transfer.deposit_surface_mixture_batch",
            "maple.surface.active_layer.initialization.initialize_active_layer_from_voxels",
            "maple.surface.topographic_commit.initialization.initial_committed_topography_state_from_physical_state",
            "maple.surface.topographic_commit.commit.commit_topography",
            "maple.surface.topographic_commit.commit.force_commit_before_checkpoint",
            "maple.aeolian.scheduler.state.initial_scheduler_state",
            "maple.aeolian.scheduler.finalization.finalize_wind_event",
            "maple.aeolian.scheduler.block.advance_wind_block",
            "maple_syrup.complete_event.complete_event", "maple_syrup.complete_event.finalize_event",
            "maple_syrup.sediment_bed.commit_bed", "maple_syrup.sediment_bed.routing_graph_from_elevation",
        )
        originals = {target: pkgutil.resolve_name(target) for target in forbidden}
        original_ids = {id(value) for value in originals.values()}
        aliases = [(module, name, value) for module in tuple(sys.modules.values())
                   if module is not None and getattr(module, "__name__", "").startswith(("maple", "masy"))
                   for name, value in tuple(vars(module).items()) if id(value) in original_ids]
        with contextlib.ExitStack() as stack:
            sentinels, replacements = {}, {}
            for target in forbidden:
                replacement = stack.enter_context(
                    mock.patch(target, side_effect=AssertionError("forbidden " + target)))
                sentinels[target.rsplit(".", 1)[1]] = replacement
                replacements[id(originals[target])] = replacement
            for module, name, original in aliases:
                replacement = replacements[id(original)]
                if vars(module)[name] is original:
                    stack.enter_context(mock.patch.object(module, name, replacement))
                self.assertIs(vars(module)[name], replacement)
            # DenseBed imports its function namespace afresh on every call:
            # verify sentinels reach those actual import sites as well.
            namespace = dense_bed._native()
            for name in ("zeros_voxel_column_state", "zeros_sediment_ledger_state",
                         "deposit_surface_mixture_batch", "initialize_active_layer_from_voxels",
                         "initial_committed_topography_state_from_physical_state", "initial_scheduler_state"):
                self.assertIs(getattr(namespace, name), sentinels[name])
            restored = self.load(packed.identity)
        self.assertEqual(restored.bed.identity_sha256(), self.bed.identity_sha256())

    def test_readonly_restore_can_stage_without_changing_the_restored_bed(self):
        packed = self.save()
        restored = self.load(packed.identity)
        before = bed_exact(restored.bed, restored.mobile_state)
        stage = restored.bed.stage_copy()
        self.assertEqual(stage.identity_sha256, restored.bed.identity_sha256())
        self.assertIsNot(stage.state.voxel_column.mass_kg, restored.bed.state.voxel_column.mass_kg)
        self.assertEqual(bed_exact(restored.bed, restored.mobile_state), before)

    def test_all_original_context_fields_are_rehashed_before_native_load(self):
        packed = self.save()
        for name, value in (("depth_update_rule", "constant_free_surface"), ("record_exchange_history", True),
                            ("window_max_journal_records", 1), ("deep_chunk_encoding", "overlay"),
                            ("deep_overlay_max_bytes", 128)):
            with self.subTest(field=name):
                bad = dataclasses.replace(self.inputs, context=dataclasses.replace(self.inputs.context, **{name: value}))
                with mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("native expanded")), self.assertRaises(ValueError):
                    self.load(packed.identity, inputs=bad)

    def test_original_case_and_initialization_metadata_changes_refuse(self):
        packed = self.save()
        for field in ("case_record", "initialization_receipt", "prescribed_digests"):
            with self.subTest(field=field):
                bad = copy.copy(self.inputs)
                data = copy.deepcopy(getattr(bad, field))
                data["unsupported_new_field"] = 0
                object.__setattr__(bad, field, data)
                with mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("native expanded")), self.assertRaises(ValueError):
                    self.load(packed.identity, inputs=bad)

    def test_original_array_alias_content_is_rehashed_before_native_load(self):
        packed = self.save()
        bad = copy.copy(self.inputs)
        held = dict(bad.held_inputs)
        key = "initial_bed_elevation_m"
        changed = held[key].copy()
        changed.flat[0] = np.nextafter(changed.flat[0], np.inf)
        changed.flags.writeable = False
        held[key] = changed
        object.__setattr__(bad, "held_inputs", held)
        before = exact(bad)
        with mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("native expanded")), self.assertRaises(ValueError):
            self.load(packed.identity, inputs=bad)
        self.assertEqual(exact(bad), before)

    def test_extra_capsule_and_deleted_default_context_instance_fields_refuse(self):
        packed = self.save()
        for kind in ("extra_capsule", "extra_context", "deleted_default"):
            with self.subTest(kind=kind):
                bad = copy.copy(self.inputs)
                if kind == "extra_capsule":
                    object.__setattr__(bad, "unexpected", 1)
                else:
                    ctx = dataclasses.replace(self.inputs.context)
                    if kind == "extra_context":
                        object.__setattr__(ctx, "unexpected", 1)
                    else:
                        object.__delattr__(ctx, "depth_update_rule")
                    object.__setattr__(bad, "context", ctx)
                with self.assertRaises(ValueError):
                    self.load(packed.identity, inputs=bad)

    def test_native_loaded_baseline_trajectory_and_water_mobile_are_refused(self):
        packed = self.packed()
        original = packed.physical
        baseline = ledger_reconciliation_baseline_from_committed_state(original.committed_topography)
        for kind, bad, message in (
            ("baseline", dataclasses.replace(original, ledger_reconciliation_baseline=baseline), "boundary profile"),
            ("trajectory", dataclasses.replace(original, coupled_trajectory_semantics="active-history"), "boundary profile"),
            ("water_mobile", dataclasses.replace(original, water=dataclasses.replace(original.water,
                mobile_mass_by_cell_class_kg=np.ones_like(original.water.mobile_mass_by_cell_class_kg))),
                "unresolved wind/water mobile load"),
        ):
            with self.subTest(kind=kind):
                path = self.native_variant(bad, packed.identity, kind)
                with self.assertRaisesRegex(ValueError, message):
                    bridge.load_boundary_physical(path, self.config, packed.identity,
                                                  inputs=self.inputs, max_bytes=CAP)

    def test_identity_configuration_and_payload_corruption_refuse_without_mutation(self):
        packed = self.save()
        before = bed_exact(self.bed, self.pool)
        changed = copy.deepcopy(packed.identity)
        changed["case_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.load(changed)
        with self.assertRaises(ValueError):
            bridge.load_boundary_physical(self.path, dataclasses.replace(self.config, outer_timestep_s=0.5),
                                          packed.identity, inputs=self.inputs, max_bytes=CAP)
        with (self.path / "native_state.npz").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaises(ValueError):
            self.load(packed.identity)
        self.assertEqual(bed_exact(self.bed, self.pool), before)

    def test_cap_refusal_precedes_native_validation_or_loading(self):
        packed = self.save()
        for cap in (0, True, 1.0, 32):
            with self.subTest(cap=cap):
                with self.assertRaises(ValueError):
                    self.packed(cap=cap)
                with mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("native expanded")), self.assertRaises(ValueError):
                    self.load(packed.identity, cap=cap)

    def test_loaded_fields_recompute_identity_and_refuse_missing_required_state(self):
        packed = self.packed()
        for name in ("water", "sediment_availability", "vegetation"):
            with self.subTest(missing=name):
                path = self.native_variant(dataclasses.replace(packed.physical, **{name: None}),
                                           packed.identity, "missing_" + name)
                with self.assertRaisesRegex(ValueError, "missing required shared/prescribed state"):
                    bridge.load_boundary_physical(path, self.config, packed.identity,
                                                  inputs=self.inputs, max_bytes=CAP)
        for name in ("case_sha256", "bed_identity_sha256", "original_inputs_sha256"):
            identity = {**packed.identity, name: "0" * 64}
            with self.subTest(identity=name):
                path = self.native_variant(packed.physical, identity, "wrong_" + name)
                message = "loaded native fields" if name == "bed_identity_sha256" else "boundary identity/original inputs"
                if name == "bed_identity_sha256":
                    with self.assertRaisesRegex(ValueError, message):
                        bridge.load_boundary_physical(path, self.config, identity, inputs=self.inputs, max_bytes=CAP)
                else:
                    # These are expected-identity preflights, before file loading.
                    with mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("native expanded")), self.assertRaisesRegex(ValueError, message):
                        bridge.load_boundary_physical(path, self.config, identity, inputs=self.inputs, max_bytes=CAP)

    def test_valid_native_nonempty_wind_pool_is_not_discarded(self):
        from maple.aeolian.flux.validation import validate_mobile_state
        pool = copy.deepcopy(self.pool)
        pool.mass_kg[0] = 1.0e-6
        pool.active_mask[0] = True
        pool.velocity_m_s[0] = 1.0
        pool.hop_distance_m[0] = 0.1
        pool.launch_time_s[0] = self.bed.time_s
        pool.impact_time_s[0] = self.bed.time_s + 1.0
        validate_mobile_state(pool)
        before = exact(pool)
        with self.assertRaises(ValueError):
            self.packed(pool=pool)
        self.assertEqual(exact(pool), before)

    def test_byte_preflight_counts_array_leaves_inside_mappings_and_sequences(self):
        value = {"nested": [np.zeros(1024, dtype=np.float64)]}
        native._bound(131072 + 4096)
        self.assertLess(bridge._budget(native, 131072 + 4096, {"nested": [None]}), 131072 + 4096)
        with self.assertRaisesRegex(ValueError, "physical/static data or metadata preflight"):
            bridge._budget(native, 131072 + 4096, value)

    def test_oversized_original_metadata_refuses_before_fingerprinting(self):
        packed = self.save()
        bad = copy.copy(self.inputs)
        receipt = copy.deepcopy(bad.initialization_receipt)
        receipt["native_sequence"] = ["X" * (2 << 20)]
        object.__setattr__(bad, "initialization_receipt", receipt)
        with mock.patch.object(native, "_fingerprint", side_effect=AssertionError("oversized metadata hashed")), self.assertRaises(ValueError):
            self.load(packed.identity, inputs=bad)

    def test_original_capture_refuses_oversized_metadata_before_bed_hashing(self):
        bed = copy.copy(self.bed)
        receipt = copy.deepcopy(bed._initialization_receipt)
        receipt["native_sequence"] = ["X" * (2 << 20)]
        bed._initialization_receipt = receipt
        with mock.patch.object(DenseBed, "verify_case_inputs", side_effect=AssertionError("oversized bed hashed")), self.assertRaises(ValueError):
            bridge.DenseBoundaryInputs.capture(bed)

    def test_static_reservation_precedes_native_decode_and_reduces_its_budget(self):
        packed = self.save()
        with mock.patch.object(native, "load_exact_physical_checkpoint", wraps=native.load_exact_physical_checkpoint) as load:
            restored = self.load(packed.identity)
        self.assertGreater(load.call_args.kwargs["max_bytes"], 0)
        self.assertLess(load.call_args.kwargs["max_bytes"], CAP)
        self.assertEqual(bed_exact(restored.bed, restored.mobile_state), bed_exact(self.bed, self.pool))

    def test_oversized_caller_identity_refuses_before_native_identity_copy(self):
        packed = self.save()
        bad = {**packed.identity, "caller": {"huge": "X" * (2 << 20)}}
        with mock.patch.object(native, "_identity", side_effect=AssertionError("oversized identity copied")), self.assertRaises(ValueError):
            self.load(bad)

    def test_cyclic_original_metadata_refuses_before_fingerprinting(self):
        packed = self.save()
        bad = copy.copy(self.inputs)
        receipt = copy.deepcopy(bad.initialization_receipt)
        cycle = []
        cycle.append(cycle)
        receipt["native_sequence"] = cycle
        object.__setattr__(bad, "initialization_receipt", receipt)
        with mock.patch.object(native, "_fingerprint", side_effect=AssertionError("cyclic metadata hashed")), self.assertRaises(ValueError):
            self.load(packed.identity, inputs=bad)

    def test_rebound_source_static_arrays_refuse_before_source_bed_hashing(self):
        oversized = np.empty(CAP // 8 + 1024, dtype=np.float64)
        oversized.flags.writeable = False
        for field in ("held", "context", "extra_held"):
            with self.subTest(field=field):
                bed = copy.copy(self.bed)
                if field == "context":
                    bed._context = dataclasses.replace(bed.context, initial_available_fraction=oversized)
                else:
                    bed._held_inputs = dict(bed._held_inputs)
                    bed._held_inputs["extra" if field == "extra_held" else "initial_bed_elevation_m"] = oversized
                with mock.patch.object(DenseBed, "verify_case_inputs", side_effect=AssertionError("unbounded source bed hashed")), self.assertRaises(ValueError):
                    self.packed(bed=bed)

    def test_writeable_context_availability_refuses_before_native_load(self):
        packed = self.save()
        writable = self.inputs.context.initial_available_fraction.copy()
        bad = dataclasses.replace(self.inputs, context=dataclasses.replace(self.inputs.context,
                                                                          initial_available_fraction=writable))
        with mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("mutable original context adopted")), self.assertRaises(ValueError):
            self.load(packed.identity, inputs=bad)

    def test_source_bed_limits_must_match_original_capsule(self):
        bed = copy.copy(self.bed)
        bed._limits = dataclasses.replace(bed.limits, stage_copy_max_bytes=bed.limits.stage_copy_max_bytes - 1)
        with self.assertRaises(ValueError):
            self.packed(bed=bed)

    def test_pack_and_native_load_do_not_change_source_flags_or_owner(self):
        def ownership():
            trees = (self.bed.state, self.pool, self.bed.context, self.bed._held_inputs,
                     self.bed.vegetation, self.bed.threshold_modifier_state)
            arrays = {(i, name): (id(value), value.flags.writeable, value.flags.owndata)
                      for i, tree in enumerate(trees) for name, value in leaves(tree).items()
                      if type(value) is np.ndarray}
            return arrays, id(self.bed._owner), {k: id(v) for k, v in self.bed._context_refs.items()}
        before = ownership()
        packed = self.packed()
        self.assertEqual(ownership(), before)
        self.assertFalse(hasattr(DenseBed, "from_boundary_physical"))
        with self.assertRaisesRegex(ValueError, "_NativePhysicalLoad"):
            DenseBed._from_native_boundary_load(packed.physical, self.config, inputs=self.inputs,
                                               expected_identity=packed.identity, max_bytes=CAP)
        native.save_exact_physical_checkpoint(self.path, packed.physical, self.config,
                                              packed.identity, max_bytes=CAP)
        with mock.patch.object(DenseBed, "_from_native_boundary_load", wraps=DenseBed._from_native_boundary_load) as constructor:
            restored = self.load(packed.identity)
        constructor.assert_called_once()
        loaded = constructor.call_args.args[0].physical
        self.assertEqual(ownership(), before)
        self.assertIsNot(restored.bed._owner, self.bed._owner)
        self.assertEqual(bed_exact(restored.bed, restored.mobile_state), bed_exact(self.bed, self.pool))
        original = leaves(packed.physical)
        for name, array in leaves(loaded).items():
            if type(array) is np.ndarray:
                self.assertFalse(array.flags.writeable)
                self.assertFalse(np.shares_memory(array, original[name]))

    def test_readonly_views_over_writable_source_arrays_are_not_restore_owners(self):
        packed = self.packed()

        def views(value):
            if dataclasses.is_dataclass(value):
                return dataclasses.replace(value, **{f.name: views(getattr(value, f.name))
                                                     for f in dataclasses.fields(value)})
            if type(value) is np.ndarray:
                view = value.view()
                view.flags.writeable = False
                return view
            return value
        carrier = views(packed.physical)
        self.assertFalse(carrier.voxel_column.mass_kg.flags.writeable)
        self.assertTrue(np.shares_memory(carrier.voxel_column.mass_kg, self.bed.state.voxel_column.mass_kg))
        self.assertTrue(self.bed.state.voxel_column.mass_kg.flags.writeable)
        with self.assertRaisesRegex(ValueError, "_NativePhysicalLoad"):
            DenseBed._from_native_boundary_load(carrier, self.config, inputs=self.inputs,
                                               expected_identity=packed.identity, max_bytes=CAP)

    def test_private_constructor_refuses_wrong_ticket_authority(self):
        packed = self.packed()
        ticket = bridge._NativePhysicalLoad(packed.physical, object())
        with mock.patch.object(DenseBed, "__init__", side_effect=AssertionError("holder created")), self.assertRaisesRegex(ValueError, "private immediate native-loader handoff"):
            DenseBed._from_native_boundary_load(ticket, self.config, inputs=self.inputs,
                                               expected_identity=packed.identity, max_bytes=CAP)


if __name__ == "__main__":
    unittest.main(verbosity=2)
