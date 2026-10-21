"""CPU genesis-only contracts for the private cold HOST physical precursor.

The synthetic DEVICE-IDENTITY FORMAT below authenticates actual CPU physical
bytes for format tests. It asserts no CuPy origin, accepted GPU event, device
restoration, executable runner or combined restart. No transport fixture runs.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest import mock

import numpy as np
from maple.aeolian.scheduler import physical_checkpoint as native
from maple.core.backend import array_leaves

from fixture_dense_boundary_physical import exact, initial_boundary
from check_dense_device_checkpoint import cpu_origin_export
from masy import dense_bed as bed_module
from masy import dense_checkpoint as bridge
from masy import dense_device_checkpoint as checkpoint
from masy import dense_device_restore as target
from masy import dense_device_runner as device


def synthetic_device_format(bed, pool, config):
    """Private test seam; valid native CPU data and explicitly authored metadata."""
    partial, inputs, limits = cpu_origin_export(bed, pool, config)
    # A valid native signed-zero water depth makes the bit-preservation assertion
    # observable without running wind/water or changing any sediment transfer.
    depth = np.array(partial.physical.water.depth_m, copy=True)
    depth.flat[0] = -0.0
    depth.setflags(write=False)
    physical = dataclasses.replace(partial.physical,
        water=dataclasses.replace(partial.physical.water, depth_m=depth))
    label = "SYNTHETIC DEVICE-IDENTITY FORMAT; actual CPU genesis; no GPU provenance"
    static = {
        "schema": device.DEVICE_STATIC_SCHEMA, "profile": dict(device.DEVICE_PROFILE),
        "case_sha256": inputs.case_sha256,
        "schedule_sha256": hashlib.sha256(b"authored format schedule label").hexdigest(),
        "case_schedule_sha256": hashlib.sha256(b"authored case/schedule label").hexdigest(),
        "controlling_inputs_sha256": hashlib.sha256(label.encode()).hexdigest(),
        "limits": dict(max_events=8, max_wind_accounting_steps=64, max_storm_steps=10000,
                       max_report_rows=100, max_work_bytes=1 << 28, max_receipt_bytes=1 << 20),
        "device_limits": dict(max_device_bytes=1 << 28, hydraulics_max_log=10000,
                              wind_workspace_bytes=1 << 20, routing_workspace_bytes=1 << 20,
                              cuda_mode="auto"),
        "bed_limits": dict(max_bed_bytes=1 << 24, max_work_bytes=1 << 28,
                           native_validation_scratch_bytes=1 << 20,
                           comparison_reduction_scratch_bytes=1 << 20),
        "resource_limits": dict(max_private_bytes=1 << 24, max_static_bytes=1 << 24,
                                max_work_bytes=1 << 28, comparison_reduction_scratch_bytes=1 << 20),
        "initial_evolving": {
            "mobile_state_sha256": native._fingerprint(pool),
            "soil_water_m": bed_module._array_digest(np.zeros(depth.shape, dtype=np.float64)),
        },
        "source": {"bed_identity_sha256": bed.identity_sha256(), "conversion": label},
    }
    identity = {**partial.identity, "caller": {"scope": label},
                "static_sha256": bed_module._sha256(static), "native_time_hex": physical.time_s.hex(),
                "physical_sha256": native._fingerprint(physical),
                "next_event": 0, "bed_generation": 0, "resource_generation": 0,
                "scope": target._IDENTITY_SCOPE}
    exported = checkpoint._issue_export(
        physical, identity, identity["physical_sha256"],
        checkpoint._json_seal(native, bridge, limits, identity), partial.evidence,
        partial.evidence_sha256, partial.config_sha256, partial.capsule_sha256)
    return exported, inputs, limits, static


class ColdPhysicalCPUContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bed, cls.pool, cls.config = initial_boundary()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.paths("")
        self.export, self.inputs, self.limits, self.static = synthetic_device_format(
            self.bed, self.pool, self.config)

    def paths(self, suffix):
        self.bundle = self.root / ("physical" + suffix)
        self.record = self.root / ("record" + suffix + ".json")

    def save(self):
        checkpoint.save_device_boundary_physical(
            self.bundle, self.export, self.config, inputs=self.inputs, limits=self.limits)

    def seal(self):
        return target.seal_device_boundary_restore(
            self.bundle, self.record, self.export, self.config,
            inputs=self.inputs, static_record=self.static, limits=self.limits)

    def prepare(self):
        self.save()
        self.digest = self.seal()

    def cold(self, **changes):
        args = dict(bundle=self.bundle, record_path=self.record, config=self.config,
                    inputs=self.inputs, limits=self.limits, expected_record_sha256=self.digest)
        args.update(changes)
        return target.load_device_boundary_cold(**args)

    def altered_record(self, change):
        row = json.loads(self.record.read_bytes())
        change(row)
        raw = native._json(row).encode()
        self.record.write_bytes(raw)
        self.digest = hashlib.sha256(raw).hexdigest()

    def test_actual_native_cold_load_after_export_release_preserves_bits_and_originals(self):
        self.save()
        expected, source = exact(self.export.physical), exact(self.bed.state)
        export_ref = weakref.ref(self.export)
        with mock.patch.object(bed_module.DenseBed, "__init__", side_effect=AssertionError("constructor")), \
                mock.patch.object(bed_module, "build_dense_bed", side_effect=AssertionError("fresh bed")):
            self.digest = self.seal()
            del self.export
            self.assertIsNone(export_ref())
            result = self.cold()
        self.assertEqual(exact(result._physical), expected)
        self.assertEqual(exact(self.bed.state), source)
        self.assertEqual(result._physical.water.depth_m.flat[0].tobytes(), np.float64(-0.0).tobytes())
        self.assertEqual(result._physical.time_s.hex(), self.bed.time_s.hex())
        self.assertIs(result._inputs, self.inputs)
        self.assertIs(result._config, self.config)
        self.assertEqual(result._record["static_record"], self.static)
        self.assertEqual(result._physical.ledger_reconciliation_baseline, None)
        for a in array_leaves(result._physical):
            self.assertFalse(a.flags.writeable)
            for original in array_leaves(self.bed.state):
                self.assertFalse(np.shares_memory(a, original))
        info = result.diagnostics()
        for flag in ("device_restore", "combined_restart", "gpu_qualified", "memory_fit", "restart_supported"):
            self.assertIs(info[flag], False)
        info["identity"]["next_event"] = 8
        self.assertEqual(result.diagnostics()["identity"]["next_event"], 0)
        result._release(); result._release()
        self.assertTrue(result.diagnostics()["released"])
        self.assertIsNone(result._physical)
        self.assertIsNone(result._inputs)

    def test_direct_carrier_creation_and_unminted_object_refuse(self):
        with self.assertRaises(checkpoint.DeviceCheckpointError):
            target._ColdPhysical()
        forged = object.__new__(target._ColdPhysical)
        with self.assertRaises(checkpoint.DeviceCheckpointError):
            forged.diagnostics()
        self.assertFalse(hasattr(target._ColdPhysical, "admit_device"))
        self.assertEqual(target.__all__, [])

    def test_cpu_origin_partial_identity_is_not_device_format(self):
        self.export, self.inputs, self.limits = cpu_origin_export(self.bed, self.pool, self.config)
        self.save()
        with self.assertRaises(checkpoint.DeviceCheckpointError), \
                mock.patch.object(checkpoint, "verify_device_boundary_roundtrip", side_effect=AssertionError("load")):
            self.seal()
        self.assertFalse(self.record.exists())

    def test_record_raw_hash_size_and_symlink_fail_before_native_load(self):
        self.prepare()
        with mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("load")), \
                mock.patch.object(native, "_audit_archive", side_effect=AssertionError("ZIP")):
            with self.assertRaises(checkpoint.DeviceCheckpointError):
                self.cold(expected_record_sha256="0" * 64)
            alias = self.root / "alias.json"; alias.symlink_to(self.record)
            with self.assertRaises(checkpoint.DeviceCheckpointError):
                self.cold(record_path=alias)
            self.record.write_bytes(b" " * (self.limits.max_metadata_bytes + 1))
            with self.assertRaises(checkpoint.DeviceCheckpointError):
                self.cold()

    def test_closed_schema_identity_static_layout_reservation_before_zip_or_load(self):
        self.prepare()
        original = self.record.read_bytes()
        mutations = {
            "extra": lambda r: r.update(unknown=True),
            "identity": lambda r: r["identity"].update(config_sha256="0" * 64),
            "static": lambda r: r["static_record"]["source"].update(conversion="different"),
            "layout": lambda r: r["layout"]["host"].update(numeric_bytes=self.limits.max_data_bytes + 1),
            "reservation": lambda r: r["reservation"].update(required_host_data_bytes=1),
            "shape": lambda r: next(iter(r["layout"]["archives"][native.PAYLOADS[0]].values()))
                            .update(shape=[self.limits.max_data_bytes + 1]),
            "objects": lambda r: next(iter(r["layout"]["archives"][native.PAYLOADS[0]].values()))
                              .update(dtype="|O8"),
        }
        for label, change in mutations.items():
            self.record.write_bytes(original)
            self.altered_record(change)
            with self.subTest(label=label), \
                    mock.patch.object(native, "_audit_archive", side_effect=AssertionError("ZIP")), \
                    mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("load")):
                with self.assertRaises((checkpoint.DeviceCheckpointError, bed_module.DenseBedError)):
                    self.cold()

    def test_native_raw_size_cap_precedes_all_payload_hashing(self):
        self.bundle.mkdir()
        (self.bundle / native.MANIFEST).write_bytes(b"{}")
        for name in native.PAYLOADS:
            (self.bundle / name).write_bytes(b"a" * 2048)
        tiny = dataclasses.replace(self.limits, max_native_bytes=1)
        # ZIP slack is owning native policy, so a modest sparse file tests the
        # actual serialized ceiling without exceeding the worker's FSIZE cap.
        with (self.bundle / native.PAYLOADS[0]).open("wb") as stream:
            stream.truncate(native._ZIP_METADATA_SLACK + 2)
        with mock.patch.object(target.hashlib, "sha256", side_effect=AssertionError("hash")), \
                mock.patch.object(native, "_audit_archive", side_effect=AssertionError("ZIP")):
            with self.assertRaises(checkpoint.DeviceCheckpointError):
                target._snapshot(self.bundle, native, tiny)

    def test_every_native_literal_including_manifest_is_raw_byte_bound(self):
        for i, name in enumerate((native.MANIFEST, *native.PAYLOADS)):
            self.paths(str(i)); self.prepare()
            with (self.bundle / name).open("ab") as stream:
                stream.write(b" ")
            with self.subTest(name=name), \
                    mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("load")), \
                    mock.patch.object(native, "_audit_archive", side_effect=AssertionError("ZIP")):
                with self.assertRaises(checkpoint.DeviceCheckpointError):
                    self.cold()

    def test_native_samefile_manifest_leftover_is_preserved_and_bound(self):
        self.save()
        leftover = self.bundle / (".manifest-" + "a" * 32)
        os.link(self.bundle / native.MANIFEST, leftover)
        self.digest = self.seal()
        result = self.cold()
        self.assertIn(leftover.name, result._record["files"])
        self.assertTrue(os.path.samefile(leftover, self.bundle / native.MANIFEST))
        self.assertEqual(exact(result._physical), exact(self.export.physical))

    def test_extra_symlink_or_false_leftover_native_namespace_refuses(self):
        for i, kind in enumerate(("fifth", "payload_symlink", "false_leftover")):
            self.paths(str(i)); self.save()
            if kind == "fifth":
                os.link(self.bundle / native.MANIFEST, self.bundle / (".manifest-" + "b" * 32))
                (self.bundle / "extra").write_bytes(b"x")
            elif kind == "payload_symlink":
                payload = self.bundle / native.PAYLOADS[0]
                outside = self.root / ("outside" + str(i))
                payload.rename(outside); payload.symlink_to(outside)
            else:
                (self.bundle / (".manifest-" + "c" * 32)).write_bytes(b"wrong")
            with self.subTest(kind=kind), \
                    mock.patch.object(checkpoint, "verify_device_boundary_roundtrip", side_effect=AssertionError("load")):
                with self.assertRaises(checkpoint.DeviceCheckpointError):
                    self.seal()
            self.assertFalse(self.record.exists())

    def test_original_case_capsule_and_config_drift_refuse(self):
        self.prepare()
        bad_config = dataclasses.replace(self.config, execution_backend="cupy")
        changed = dataclasses.replace(self.inputs, case_sha256="0" * 64)
        missing_case = {k: v for k, v in self.inputs.case_record.items() if k != "n_voxels"}
        missing = dataclasses.replace(self.inputs, case_record=missing_case)
        for name, delta in (("config", {"config": bad_config}), ("capsule", {"inputs": changed}),
                            ("missing_voxels", {"inputs": missing})):
            with self.subTest(name=name), \
                    mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("load")):
                with self.assertRaises((checkpoint.DeviceCheckpointError, bed_module.DenseBedError)):
                    self.cold(**delta)
        self.inputs.case_record["initial_time_s"] = {"float_hex": "0x1.0p+0"}
        with self.assertRaises((checkpoint.DeviceCheckpointError, bed_module.DenseBedError)):
            self.cold()

    def test_native_readonly_fingerprint_and_original_prescribed_stage_guards(self):
        self.prepare()
        loader = native.load_exact_physical_checkpoint
        for kind in ("writable", "clock", "prescribed"):
            def altered(*args, **kwargs):
                physical = loader(*args, **kwargs)
                if kind == "writable":
                    physical.active_layer.mass_kg.setflags(write=True)
                elif kind == "clock":
                    physical = dataclasses.replace(physical, time_s=1.0)
                else:
                    a = np.array(physical.threshold_modifier_state.gravimetric_moisture_fraction, copy=True)
                    a.flat[0] = 0.01; a.setflags(write=False)
                    physical = dataclasses.replace(physical, threshold_modifier_state=dataclasses.replace(
                        physical.threshold_modifier_state, gravimetric_moisture_fraction=a))
                return physical
            with self.subTest(kind=kind), mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=altered):
                with self.assertRaises((checkpoint.DeviceCheckpointError, bed_module.DenseBedError)):
                    self.cold()

    def test_native_genuinely_empty_mobile_predicate_is_required(self):
        self.prepare()
        with mock.patch("maple.aeolian.cohorts.dispatch.mobile_state_is_genuinely_empty", return_value=False) as check:
            with self.assertRaises(checkpoint.DeviceCheckpointError) as error:
                self.cold()
        self.assertEqual(error.exception.code, "cold_mobile")
        check.assert_called_once()

    def test_native_header_mismatch_refuses_before_physical_allocation(self):
        self.prepare()
        audit = native._audit_archive
        def altered(path, remaining):
            rows, size = audit(path, remaining)
            rows = copy.deepcopy(rows)
            next(iter(rows.values()))["npy_bytes"] += 1
            return rows, size
        with mock.patch.object(native, "_audit_archive", side_effect=altered), \
                mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("load")):
            with self.assertRaises(checkpoint.DeviceCheckpointError) as error:
                self.cold()
        self.assertEqual(error.exception.code, "cold_layout")

    def test_artifact_or_original_mutation_during_load_cannot_mint_ticket(self):
        self.prepare()
        loader = native.load_exact_physical_checkpoint
        def changed(*args, **kwargs):
            physical = loader(*args, **kwargs)
            with (self.bundle / native.MANIFEST).open("ab") as stream:
                stream.write(b" ")
            return physical
        with mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=changed):
            with self.assertRaises(checkpoint.DeviceCheckpointError) as error:
                self.cold()
        self.assertEqual(error.exception.code, "cold_file")
        self.assertTrue(self.bundle.exists())

    def test_companion_is_exclusive_outside_bundle_and_failure_retains_evidence(self):
        self.prepare()
        original = self.record.read_bytes()
        for path in (self.record, self.bundle / "record.json"):
            with self.subTest(path=path), self.assertRaises(checkpoint.DeviceCheckpointError):
                target.seal_device_boundary_restore(self.bundle, path, self.export, self.config,
                    inputs=self.inputs, static_record=self.static, limits=self.limits)
        self.assertEqual(self.record.read_bytes(), original)
        fresh = self.root / "uncertain.json"
        with mock.patch.object(native, "_fsync", side_effect=OSError("authored directory fsync failure")):
            with self.assertRaises(target.ColdRecordPublicationUncertain) as error:
                target.seal_device_boundary_restore(self.bundle, fresh, self.export, self.config,
                    inputs=self.inputs, static_record=self.static, limits=self.limits)
        self.assertTrue(fresh.is_file())
        self.assertEqual(hashlib.sha256(fresh.read_bytes()).hexdigest(), error.exception.record_sha256)
        self.assertEqual(self.record.read_bytes(), original)
        postcheck = self.root / "postcheck_failed.json"
        with mock.patch.object(target, "_read", side_effect=OSError("authored postpublication read failure")):
            with self.assertRaises(target.ColdRecordPublicationUncertain) as caught:
                target.seal_device_boundary_restore(self.bundle, postcheck, self.export, self.config,
                    inputs=self.inputs, static_record=self.static, limits=self.limits)
        self.assertTrue(postcheck.is_file())
        self.assertEqual(hashlib.sha256(postcheck.read_bytes()).hexdigest(), caught.exception.record_sha256)
        native_uncertain = native.CheckpointPublicationUncertain("authored", "authored")
        with mock.patch.object(checkpoint, "verify_device_boundary_roundtrip", side_effect=native_uncertain):
            with self.assertRaises(native.CheckpointPublicationUncertain) as caught:
                target.seal_device_boundary_restore(self.bundle, self.root / "native_uncertain.json",
                    self.export, self.config, inputs=self.inputs, static_record=self.static, limits=self.limits)
        self.assertIs(caught.exception, native_uncertain)
        self.assertFalse((self.root / "native_uncertain.json").exists())


if __name__ == "__main__":
    unittest.main()
# New private admission preflight/interface controls. The original15 methods
# above remain exact. These are CPU FORMAT fixtures, never genuine CuPy owners.
class ColdBedAdmissionCPUContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bed, cls.pool, cls.config = initial_boundary()

    def setUp(self):
        self.cold = None
        self.addCleanup(self._drop_fixture_references)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.bundle, self.record = self.root / "native", self.root / "cold.json"
        # cpu_origin_export copies the packed source bed fields bit-for-bit;
        # refuse a nonzero native genesis source BEFORE the old format helper
        # changes +0 to -0. Existing helper and all original15 tests stay exact.
        self.assertEqual(self.bed.state.water.depth_m.flat[0], 0.0)
        export, self.inputs, self.limits, static = synthetic_device_format(self.bed, self.pool, self.config)
        checkpoint.save_device_boundary_physical(self.bundle, export, self.config,
                                                  inputs=self.inputs, limits=self.limits)
        digest = target.seal_device_boundary_restore(self.bundle, self.record, export, self.config,
                                                     inputs=self.inputs, static_record=static, limits=self.limits)
        self.cold = target.load_device_boundary_cold(self.bundle, self.record, self.config,
            inputs=self.inputs, limits=self.limits, expected_record_sha256=digest)
        from masy.dense_device_bed import DeviceBedLimits
        self.bed_limits = DeviceBedLimits(**static["bed_limits"])

    def _drop_fixture_references(self):
        if self.cold is not None:
            self.cold._release()
        self.cold = self.inputs = None

    def plan(self, **changes):
        return target._prepare_bed_admission(changes.get("cold", self.cold),
                                              changes.get("bed_limits", self.bed_limits))

    def test_real_cold_preflight_preserves_complete_native_fields_without_constructors(self):
        physical = self.cold._physical
        before = exact(physical)
        flags = tuple(a.flags.writeable for a in array_leaves(physical))
        with mock.patch.object(bed_module.DenseBed, "__init__", side_effect=AssertionError("fresh bed")), \
                mock.patch.object(bed_module, "build_dense_bed", side_effect=AssertionError("fresh bed")), \
                mock.patch.object(device, "_create_dense_device_runner", side_effect=AssertionError("runner")), \
                mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("reload")):
            plan = self.plan()
        self.assertEqual(tuple(target._RESIDUAL_FIELDS), ("mobile_state", "threshold_modifier_state", "time_s",
            "vegetation", "coupled_trajectory_semantics", "ledger_reconciliation_baseline"))
        for name, item in zip(target._RESIDUAL_FIELDS, plan["residual"], strict=True):
            self.assertIs(item, getattr(physical, name))
        for name in bed_module.SHARED_FIELDS:
            self.assertIs(getattr(plan["shallow"], name), getattr(physical, name))
        self.assertEqual(plan["binding"]["bed_leaves"], 23)
        self.assertEqual(plan["binding"]["native_time_hex"], physical.time_s.hex())
        self.assertEqual(plan["binding"]["saved_bed_generation"], self.cold._record["identity"]["bed_generation"])
        self.assertEqual(plan["binding"]["native_commit_count"], physical.committed_topography.commit_count)
        self.assertTrue(np.signbit(physical.water.depth_m.flat[0]))
        self.assertEqual(exact(physical), before)
        self.assertEqual(tuple(a.flags.writeable for a in array_leaves(physical)), flags)
        self.assertIs(self.cold._physical, physical)

    def test_binding_is_detached_plain_metadata_and_host_reservation_is_not_rss_or_fit(self):
        plan = self.plan()
        encoded = json.dumps(plan["binding"], sort_keys=True, allow_nan=False)
        self.assertEqual(json.loads(encoded)["bed_leaves"], 23)
        self.assertNotIn("owner_local_generation", plan["binding"])
        reservation = plan["binding"]["reservation"]
        self.assertFalse(reservation["memory_fit"])
        self.assertEqual(reservation["scratch_status"], "DECLARED_UNMEASURED")
        self.assertLessEqual(reservation["required_host_data_bytes"], self.limits.max_host_data_bytes)
        self.assertEqual(reservation["host_data_terms"]["native_stream_hash_buffers"], 2 << 20)
        plan["binding"]["source_identity"]["caller"]["modified_locally"] = True
        self.assertNotIn("modified_locally", self.cold._record["identity"]["caller"])

    def test_mismatched_and_small_host_limits_fail_before_source_fingerprints_or_hash_reads(self):
        for mutation in ("bed", "host"):
            with self.subTest(mutation=mutation):
                original_limits = self.cold._limits
                limits = dataclasses.replace(self.bed_limits, max_work_bytes=self.bed_limits.max_work_bytes + 1)
                if mutation == "host":
                    self.cold._limits = dataclasses.replace(self.limits, max_host_data_bytes=1 << 20)
                    limits = self.bed_limits
                try:
                    with mock.patch.object(native, "_fingerprint", side_effect=AssertionError("hash")), \
                            mock.patch.object(target, "_read", side_effect=AssertionError("read")), \
                            mock.patch.object(target, "_snapshot", side_effect=AssertionError("snapshot")):
                        with self.assertRaises(checkpoint.DeviceCheckpointError) as caught:
                            self.plan(bed_limits=limits)
                    self.assertEqual(caught.exception.code, "admission_limit" if mutation == "bed" else "host_bytes_cap")
                finally:
                    self.cold._limits = original_limits

    def test_allocation_root_bytes_and_disjoint_slices_are_not_lifetime_disjoint(self):
        allocation = np.arange(32, dtype=np.float64)
        left, right = allocation[:8], allocation[24:]
        self.assertFalse(np.shares_memory(left, right))
        plan = target._admission_roots((("left", left),), np, native, self.limits)
        self.assertEqual(plan["visible_bytes"], left.nbytes)
        self.assertEqual(plan["root_bytes"], allocation.nbytes)
        other = target._admission_roots((("right", right),), np, native, self.limits)
        with self.assertRaises(checkpoint.DeviceCheckpointError):
            target._admission_root_disjoint(plan["roots"], other["roots"], np)
        small = dataclasses.replace(self.limits, max_host_data_bytes=left.nbytes)
        with self.assertRaises(checkpoint.DeviceCheckpointError):
            target._admission_roots((("left", left),), np, native, small)

    def test_unknown_backing_subclass_module_and_arbitrary_object_refuse(self):
        import types
        class ForeignArray(np.ndarray):
            pass
        values = {"bytes_backing": np.frombuffer(bytes(64), dtype=np.float64),
                  "subclass": np.zeros(2).view(ForeignArray),
                  "module": types.ModuleType("authored_unknown_backend"), "object": object()}
        for name, value in values.items():
            with self.subTest(case=name), self.assertRaises(checkpoint.DeviceCheckpointError):
                target._admission_roots((("unknown", value),), np, native, self.limits)

    def test_alias_visits_charge_every_visit_but_one_terminal_root_and_cycles_refuse(self):
        array = np.zeros(8, dtype=np.float64)
        plan = target._admission_roots((("repeated", [array, array]),), np, native, self.limits)
        self.assertEqual(len(plan["arrays"]), 2)
        self.assertEqual(len(plan["roots"]), 1)
        self.assertEqual(plan["visible_bytes"], 2 * array.nbytes)
        self.assertEqual(plan["root_bytes"], array.nbytes)
        cycle = []; cycle.append(cycle)
        with self.assertRaises(checkpoint.DeviceCheckpointError):
            target._admission_roots((("cyclic", cycle),), np, native, self.limits)

    def test_forbidden_bed_holder_and_real_physical_residual_shared_base_refuse(self):
        physical = self.cold._physical
        with self.assertRaises(checkpoint.DeviceCheckpointError):
            target._admission_roots((("holder", physical.active_layer),), np, native, self.limits,
                                     forbidden_holders=(id(physical.active_layer),))
        depth, threshold = physical.water.depth_m, physical.threshold_modifier_state.clay_fraction
        base = np.empty(depth.size + threshold.size, dtype=np.float64)
        first = base[:depth.size].reshape(depth.shape); first[...] = depth
        second = base[depth.size:].reshape(threshold.shape); second[...] = threshold
        first.setflags(write=False); second.setflags(write=False)
        self.assertFalse(np.shares_memory(first, second))
        changed = dataclasses.replace(physical, water=dataclasses.replace(physical.water, depth_m=first),
            threshold_modifier_state=dataclasses.replace(physical.threshold_modifier_state, clay_fraction=second))
        self.assertEqual(exact(changed), exact(physical))  # Same values; unsupported shared lifetime.
        self.assertEqual(tuple(a.flags.writeable for a in array_leaves(changed)),
                         tuple(a.flags.writeable for a in array_leaves(physical)))
        self.cold._physical = changed
        with mock.patch.object(native, "_fingerprint", side_effect=AssertionError("hash")):
            with self.assertRaises(checkpoint.DeviceCheckpointError) as caught:
                self.plan()
        self.assertEqual(caught.exception.code, "admission_retention")
        self.assertIs(self.cold._physical, changed)

    def test_missing_and_extra_holder_attributes_fail_before_hash_not_class_default_fallback(self):
        for name in ("extra", "missing"):
            with self.subTest(case=name):
                physical = dataclasses.replace(self.cold._physical)
                if name == "extra":
                    object.__setattr__(physical, "unsupported", "authored metadata")
                else:
                    object.__delattr__(physical, "ledger_reconciliation_baseline")
                old = self.cold._physical
                self.cold._physical = physical
                try:
                    with mock.patch.object(native, "_fingerprint", side_effect=AssertionError("hash")):
                        with self.assertRaises(native.PhysicalCheckpointError) as caught:
                            self.plan()
                    self.assertIn("unsupported exact instance layout", str(caught.exception))
                finally:
                    self.cold._physical = old

    def test_saved_touch_dtype_and_native_byteorder_are_refused_without_cast(self):
        physical = self.cold._physical
        for dtype in (np.dtype("int32"), np.dtype("int64").newbyteorder("S")):
            with self.subTest(dtype=dtype.str):
                touches = np.array(physical.ledger.n_physical_touches, dtype=dtype)
                touches.setflags(write=False)
                changed = dataclasses.replace(physical, ledger=dataclasses.replace(physical.ledger,
                                                    n_physical_touches=touches))
                self.cold._physical = changed
                try:
                    with mock.patch.object(native, "_fingerprint", side_effect=AssertionError("hash")):
                        with self.assertRaises(checkpoint.DeviceCheckpointError):
                            self.plan()
                    self.assertEqual(touches.dtype, dtype)
                    self.assertFalse(touches.flags.writeable)
                finally:
                    self.cold._physical = physical

    def test_readonly_drift_and_unsupported_residual_semantics_do_not_reset_source(self):
        physical = self.cold._physical
        writable = np.array(physical.water.depth_m, copy=True)
        changes = (dataclasses.replace(physical, water=dataclasses.replace(physical.water, depth_m=writable)),
                   dataclasses.replace(physical, coupled_trajectory_semantics="unsupported authored marker"))
        for changed in changes:
            with self.subTest(marker=changed.coupled_trajectory_semantics):
                self.cold._physical = changed
                if changed.coupled_trajectory_semantics is None:
                    with self.assertRaises(bed_module.DenseBedError) as caught:
                        self.plan()
                    self.assertEqual(caught.exception.code, "invalid_boundary_physical")
                    self.assertIn("writeable", str(caught.exception))
                else:
                    with self.assertRaises(checkpoint.DeviceCheckpointError) as caught:
                        self.plan()
                    self.assertEqual(caught.exception.code, "admission_profile")
                self.assertIs(self.cold._physical, changed)
        self.cold._physical = physical

    def test_forged_and_released_carriers_fail_before_backend(self):
        forged = object.__new__(target._ColdPhysical)
        with mock.patch.object(target, "_admission_backend", side_effect=AssertionError("backend")) as backend:
            for value in (object(), forged):
                with self.subTest(holder=type(value).__name__), self.assertRaises(target.DeviceBedAdmissionError) as caught:
                    target.admit_device_bed_from_cold(value, bed_limits=self.bed_limits)
                self.assertEqual(caught.exception.code, "admission_cold")
            self.cold._release()
            with self.assertRaises(target.DeviceBedAdmissionError) as caught:
                target.admit_device_bed_from_cold(self.cold, bed_limits=self.bed_limits)
            self.assertEqual(caught.exception.code, "admission_cold")
            backend.assert_not_called()

    def test_absent_backend_cannot_reach_native_preparation_or_mint_authority(self):
        import sys
        with mock.patch.dict(sys.modules, {"cupy": None}), \
                mock.patch.object(target, "_prepare_bed_admission", side_effect=AssertionError("preparation")):
            with self.assertRaises(target.DeviceBedAdmissionError) as caught:
                target.admit_device_bed_from_cold(self.cold, bed_limits=self.bed_limits)
        self.assertEqual(caught.exception.code, "admission_backend")
        self.assertEqual(caught.exception.evidence["native_snapshots"], [])
        self.assertIsNone(caught.exception.evidence["owner_private_counts"])
        self.assertIsNotNone(self.cold._physical)
        self.assertEqual(target.__all__, [])
        self.assertFalse(hasattr(target._DeviceBedPhysical, "step"))

    def test_detached_error_does_not_retain_source_exception_frame_owner(self):
        references = []
        class InterfaceOwner:
            pass
        def fail(cold, bed_limits, evidence):
            owner = InterfaceOwner()
            references.append(weakref.ref(owner))
            evidence["phase"] = "authored_interface_failure_no_device"
            raise RuntimeError("authored interface refusal")
        with mock.patch.object(target, "_convert_cold_bed", side_effect=fail):
            with self.assertRaises(target.DeviceBedAdmissionError) as caught:
                target.admit_device_bed_from_cold(self.cold, bed_limits=self.bed_limits)
        self.assertIsNone(caught.exception.__context__)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(references[0]())
        self.assertFalse(caught.exception.evidence["owner_returned"])
        self.assertEqual(caught.exception.code, "admission_failure")
        self.assertEqual(caught.exception.evidence["phase"], "authored_interface_failure_no_device")
        self.assertIsNotNone(self.cold._physical)

    def test_keyboard_interrupt_and_system_exit_preserve_cancellation_contract(self):
        for exception in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception=exception.__name__), \
                    mock.patch.object(target, "_convert_cold_bed", side_effect=exception):
                with self.assertRaises(exception):
                    target.admit_device_bed_from_cold(self.cold, bed_limits=self.bed_limits)

    def test_native_counter_schema_and_phase_expectations_are_closed(self):
        from maple.core.backend.arrays import TransferCounters
        before = target._admission_counter_record(TransferCounters())
        after = dict(before, host_to_device=23, host_to_device_bytes=4096)
        target._admission_expected_delta(before, after, host_to_device=23, host_to_device_bytes=4096)
        self.assertEqual(target._admission_counter_delta(before, after)["device_to_host"], 0)
        with self.assertRaises(checkpoint.DeviceCheckpointError):
            target._admission_expected_delta(before, after, host_to_device=23)  # Missing actual byte transfer.
        with self.assertRaises(checkpoint.DeviceCheckpointError):
            target._admission_counter_record(dict(before))  # Wrong holder, even with plausible fields.
        for case in ("bool", "missing", "negative", "unknown_expectation"):
            with self.subTest(case=case), self.assertRaises(checkpoint.DeviceCheckpointError):
                altered = dict(after)
                if case == "bool": altered["host_to_device"] = True
                elif case == "missing": altered.pop("scalar_reads")
                elif case == "negative": altered["scalar_reads"] = -1
                if case == "unknown_expectation":
                    target._admission_expected_delta(before, before, not_a_native_counter=1)
                else:
                    target._admission_counter_delta(before, altered)

    def test_physical_reference_swap_during_validation_is_refused_not_silently_adopted(self):
        physical = self.cold._physical
        replacement = dataclasses.replace(physical)
        actual = target._physical
        def validate_then_swap(*args, **kwargs):
            actual(*args, **kwargs)
            self.cold._physical = replacement
        with mock.patch.object(target, "_physical", side_effect=validate_then_swap):
            with self.assertRaises(checkpoint.DeviceCheckpointError) as caught:
                self.plan()
        self.assertEqual(caught.exception.code, "admission_source")
        self.assertIs(self.cold._physical, replacement)  # No assumed rollback.
        self.assertEqual(exact(replacement), exact(physical))

    def test_complete_bed_residual_partition_drift_refuses_before_walk_or_hash(self):
        for fields in (target._RESIDUAL_FIELDS[:-1], (*target._RESIDUAL_FIELDS[:-1], "water"),
                       (*target._RESIDUAL_FIELDS[:-1], "not_a_native_field")):
            with self.subTest(fields=fields), mock.patch.object(target, "_RESIDUAL_FIELDS", fields), \
                    mock.patch.object(target, "_admission_roots", side_effect=AssertionError("walk")), \
                    mock.patch.object(native, "_fingerprint", side_effect=AssertionError("hash")):
                with self.assertRaises(checkpoint.DeviceCheckpointError) as caught:
                    self.plan()
                self.assertEqual(caught.exception.code, "admission_layout")

    def test_extra_backing_allocation_hits_new_host_cap_after_inherited_reservation_admits(self):
        plan = self.plan()
        inherited = plan["binding"]["reservation"]["inherited_cold_data_bytes"]
        self.assertLessEqual(inherited, self.limits.max_host_data_bytes)
        physical = self.cold._physical
        depth = physical.water.depth_m
        bytes_needed = self.limits.max_host_data_bytes - inherited + depth.nbytes + (1 << 20)
        # Explicit CPU fixture bound BEFORE allocation; no disk or model-size change.
        self.assertLessEqual(bytes_needed, 32 << 20)
        backing = np.empty((bytes_needed + 7) // 8, dtype=np.float64)
        view = backing[:depth.size].reshape(depth.shape); view[...] = depth
        view.setflags(write=False)
        changed = dataclasses.replace(physical, water=dataclasses.replace(physical.water, depth_m=view))
        self.assertEqual(exact(changed), exact(physical))
        self.cold._physical = changed
        with mock.patch.object(native, "_fingerprint", side_effect=AssertionError("hash")), \
                mock.patch.object(target, "_read", side_effect=AssertionError("read")), \
                mock.patch.object(target, "_snapshot", side_effect=AssertionError("snapshot")):
            with self.assertRaises(checkpoint.DeviceCheckpointError) as caught:
                self.plan()
        self.assertEqual(caught.exception.code, "admission_host_cap")
        self.assertIs(self.cold._physical, changed)
