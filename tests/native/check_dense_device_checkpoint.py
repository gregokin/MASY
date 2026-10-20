"""CPU-origin physical-codec contracts; NOT a positive GPU export test.

Explicit origin-verified CPU worker only. One test obtains a legitimate
nonzero-time/pending-ledger boundary through the existing native wind fixture.
No CuPy namespace is synthesized and no device is selected or queried.
"""
from __future__ import annotations

import dataclasses
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from fixture_dense_boundary_physical import exact, initial_boundary, pending_boundary
from maple.aeolian.scheduler import physical_checkpoint as native
from maple.core.backend import array_leaves
from maple.core.backend.arrays import _map_tree

from masy import dense_checkpoint as bridge
from masy import dense_device_checkpoint as target
from masy.dense_bed import DenseBed, DenseBedError


def cpu_origin_export(bed, pool, config):
    """Private test seam: genuine CPU carrier, no device-export claim."""
    limits = target.DeviceCheckpointLimits()
    inputs = bridge.DenseBoundaryInputs.capture(bed)
    packed = bridge.pack_boundary_physical(
        bed, pool, config, {"scope": "CPU-origin codec contract only"},
        inputs=inputs, max_bytes=limits.max_host_data_bytes)
    def own(a):
        result = np.array(a, copy=True, order="C")
        result.setflags(write=False)
        return result
    physical = _map_tree(packed.physical, own)
    digest = native._fingerprint(physical)
    config_sha = native._fingerprint(config)
    identity = {
        "schema": target.SCHEMA, "caller": {"scope": "CPU-origin codec contract only"},
        "capsule_sha256": inputs.binding_sha256, "case_sha256": inputs.case_sha256,
        "config_sha256": config_sha, "physical_sha256": digest,
        "native_time_hex": physical.time_s.hex(),
        "limits": dataclasses.asdict(limits),
    }
    layout = target._array_plan((physical,), np.ndarray, np, limits)
    static = bridge._budget(native, limits.max_host_data_bytes, inputs, config)
    evidence = {"scope": "CPU-origin codec contract only", "gpu_qualified": False,
                "combined_restart": False, "device_restore": False,
                "reservation": target._reserve(layout, static, limits)}
    exported = target._issue_export(
        physical, identity, digest,
        target._json_seal(native, bridge, limits, identity), evidence,
        target._json_seal(native, bridge, limits, evidence), config_sha,
        inputs.binding_sha256)
    return exported, inputs, limits


class DeviceCheckpointCPUContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bed, cls.pool, cls.config = initial_boundary()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "physical"
        self.export, self.inputs, self.limits = cpu_origin_export(self.bed, self.pool, self.config)

    def save(self):
        return target.save_device_boundary_physical(
            self.path, self.export, self.config, inputs=self.inputs, limits=self.limits)

    def load(self, **changes):
        values = {"path": self.path, "export": self.export, "config": self.config,
                  "inputs": self.inputs, "limits": self.limits}
        values.update(changes)
        return target.verify_device_boundary_roundtrip(**values)

    def reissue_cpu_evidence(self, evidence):
        """CPU-origin failure injection; never an owner or device-export proof."""
        original = self.export
        return target._issue_export(original.physical, original.identity,
            original.expected_physical_sha256, original.identity_sha256, evidence,
            target._json_seal(native, bridge, self.limits, evidence), original.config_sha256,
            original.capsule_sha256)

    def test_real_cpu_origin_native_roundtrip_preserves_all_fields(self):
        before = exact(self.export.physical)
        source = exact(self.bed.state)
        self.save()
        restored = self.load()
        self.assertIs(type(restored), type(self.export.physical))
        self.assertEqual(exact(restored), before)
        self.assertEqual(exact(self.bed.state), source)
        for a in array_leaves(restored):
            self.assertFalse(a.flags.writeable)
        self.assertFalse(self.export.evidence["gpu_qualified"])

    def test_readback_never_constructs_or_reinitializes_a_dense_bed(self):
        self.save()
        with mock.patch.object(DenseBed, "__init__", side_effect=AssertionError("bed constructor")), \
                mock.patch("masy.dense_bed.build_dense_bed", side_effect=AssertionError("case reload")):
            self.assertEqual(exact(self.load()), exact(self.export.physical))

    def test_pending_native_wind_ledger_and_nonzero_time_roundtrip(self):
        bed, pool, config, _ = pending_boundary()
        exported, inputs, limits = cpu_origin_export(bed, pool, config)
        self.assertGreater(exported.physical.time_s, 0.0)
        before = exact(exported.physical)
        target.save_device_boundary_physical(self.path, exported, config, inputs=inputs, limits=limits)
        loaded = target.verify_device_boundary_roundtrip(
            self.path, exported, config, inputs=inputs, limits=limits)
        self.assertEqual(exact(loaded), before)
        self.assertEqual(loaded.time_s.hex(), bed.time_s.hex())
        self.assertTrue(np.any(loaded.ledger.pending_bed_mass_change_kg != 0.0))

    def test_host_copy_is_owned_preserves_bits_and_borrowed_flags(self):
        for writable in (True, False):
            with self.subTest(writable=writable):
                a = np.array([1.0, -0.0, 2.0])
                a.setflags(write=writable)
                counts = dict(device_arrays=0, device_bytes=0, host_arrays=0, host_bytes=0)
                copied = target._copy_tree(a, np, np, counts)
                self.assertIsNot(copied, a)
                self.assertFalse(np.shares_memory(a, copied))
                self.assertFalse(copied.flags.writeable)
                self.assertEqual(a.flags.writeable, writable)
                self.assertEqual(copied.tobytes(), a.tobytes())
                self.assertEqual(counts, dict(device_arrays=0, device_bytes=0, host_arrays=1, host_bytes=24))

    def test_mirror_comparison_detects_one_bit_including_signed_zero(self):
        a = np.array([-0.0])
        target._verify_mirror(a, a.copy(), np, np)
        b = np.array([0.0])
        with self.assertRaises(target.DeviceCheckpointError) as error:
            target._verify_mirror(a, b, np, np)
        self.assertEqual(error.exception.code, "copy_mismatch")

    def test_mirror_layout_refuses_before_any_download(self):
        cases = ((np.ones(1), np.ones(2)), (np.ones(1), np.ones(1, dtype=np.float32)),
                 (np.ones(2), np.ones(4)[::2]))
        for a, b in cases:
            with self.subTest(shape=b.shape, dtype=str(b.dtype)), \
                    mock.patch("maple.core.backend.to_host", side_effect=AssertionError("download")):
                with self.assertRaises(target.DeviceCheckpointError) as error:
                    target._verify_mirror(a, b, np, np)
                self.assertEqual(error.exception.code, "copy_mismatch")

    def test_native_copy_walk_agrees_with_plan_and_repeated_leaves_refuse(self):
        a, b = np.ones(3), np.ones(2, dtype=np.int64)
        plan = target._array_plan(((a, {"b": b}),), np.ndarray, np, self.limits)
        self.assertEqual(plan, dict(numeric_bytes=40, largest_leaf_bytes=24, leaves=2,
                                    device_bytes=0, device_arrays=0))
        with self.assertRaises(target.DeviceCheckpointError) as error:
            target._array_plan(((a, a),), np.ndarray, np, self.limits)
        self.assertEqual(error.exception.code, "array_layout")

    def test_native_walk_disagreement_refuses(self):
        a = np.ones(3)
        inventory = dict(n_leaves=1, transfer_bytes=24, n_aliased_leaves=0)
        for field in inventory:
            changed = {**inventory, field: inventory[field] + 1}
            with self.subTest(field=field), \
                    mock.patch("maple.core.backend.arrays.array_leaf_inventory", return_value=changed):
                with self.assertRaises(target.DeviceCheckpointError) as error:
                    target._array_plan((a,), np.ndarray, np, self.limits)
                self.assertEqual(error.exception.code, "array_layout")

    def test_copy_accounting_accepts_exact_counts_and_refuses_mismatch(self):
        layout = dict(numeric_bytes=40, largest_leaf_bytes=24, leaves=2,
                      device_bytes=24, device_arrays=1)
        counts = dict(device_arrays=1, device_bytes=24, host_arrays=1, host_bytes=16)
        delta = dict(device_to_host=1, device_to_host_bytes=24,
                     host_to_device=0, host_to_device_bytes=0)
        target._check_copies(layout, counts, delta)
        for field in counts:
            changed = {**counts, field: counts[field] + 1}
            with self.subTest(copy_field=field), self.assertRaises(target.DeviceCheckpointError) as error:
                target._check_copies(layout, changed, delta)
            self.assertEqual(error.exception.code, "copy_accounting")
        for field in delta:
            changed = {**delta, field: delta[field] + 1}
            with self.subTest(counter_field=field), self.assertRaises(target.DeviceCheckpointError) as error:
                target._check_copies(layout, counts, changed)
            self.assertEqual(error.exception.code, "copy_accounting")

    def test_limits_reject_bool_zero_negative_and_above_ceiling(self):
        for name, ceiling in target._CEILINGS.items():
            for bad in (True, 0, -1, ceiling + 1):
                with self.subTest(name=name, bad=bad), self.assertRaises(target.DeviceCheckpointError):
                    target.DeviceCheckpointLimits(**{name: bad})

    def test_extra_limit_attribute_is_refused(self):
        limits = target.DeviceCheckpointLimits()
        object.__setattr__(limits, "unlisted", 1)
        with self.assertRaises(target.DeviceCheckpointError):
            target._limits(limits)

    def test_non_device_runner_refuses_before_native_bridge(self):
        with mock.patch.object(target, "_native", side_effect=AssertionError("native imports")):
            with self.assertRaises(target.DeviceCheckpointError) as error:
                target.export_device_boundary_physical(
                    None, self.config, {}, inputs=self.inputs, limits=self.limits)
        self.assertEqual(error.exception.code, "runner_layout")

    def test_unissued_export_refuses_before_native_save(self):
        for bad in (dataclasses.replace(self.export), copy.copy(self.export)):
            with self.subTest(copy_type=type(bad._authority).__name__), \
                    mock.patch.object(native, "save_exact_physical_checkpoint", side_effect=AssertionError("native save")):
                with self.assertRaises(target.DeviceCheckpointError) as error:
                    target.save_device_boundary_physical(
                        self.path, bad, self.config, inputs=self.inputs, limits=self.limits)
                self.assertEqual(error.exception.code, "export_ticket")
        self.assertFalse(self.path.exists())

    def test_changed_identity_and_evidence_refuse_before_native_save(self):
        for field in ("identity", "evidence"):
            bad, _, _ = cpu_origin_export(self.bed, self.pool, self.config)
            getattr(bad, field)["unexpected"] = True
            with self.subTest(field=field), self.assertRaises(target.DeviceCheckpointError) as error:
                target.save_device_boundary_physical(
                    self.path, bad, self.config, inputs=self.inputs, limits=self.limits)
            self.assertEqual(error.exception.code, "export_identity")
            self.assertFalse(self.path.exists())

    def test_changed_admitted_limits_refuse_before_save_and_load(self):
        changed = dataclasses.replace(self.limits, max_native_bytes=4 << 20)
        with mock.patch.object(native, "save_exact_physical_checkpoint", side_effect=AssertionError("save")), \
                mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("load")):
            for action in (lambda: target.save_device_boundary_physical(
                    self.path, self.export, self.config, inputs=self.inputs, limits=changed),
                    lambda: self.load(limits=changed)):
                with self.subTest(action=action), self.assertRaises(target.DeviceCheckpointError) as error:
                    action()
                self.assertEqual(error.exception.code, "export_identity")
        self.assertFalse(self.path.exists())

    def test_resealed_modified_ticket_still_refuses(self):
        identity = {**self.export.identity, "caller": {"scope": "copied ticket"}}
        bad = dataclasses.replace(self.export, identity=identity,
                                  identity_sha256=target._json_seal(native, bridge, self.limits, identity))
        with self.assertRaises(target.DeviceCheckpointError) as error:
            target.save_device_boundary_physical(
                self.path, bad, self.config, inputs=self.inputs, limits=self.limits)
        self.assertEqual(error.exception.code, "export_ticket")

    def test_in_place_resealed_issued_ticket_refuses(self):
        self.export.identity["caller"] = {"scope": "changed issued object"}
        object.__setattr__(self.export, "identity_sha256",
                           target._json_seal(native, bridge, self.limits, self.export.identity))
        with mock.patch.object(native, "save_exact_physical_checkpoint", side_effect=AssertionError("save")):
            with self.assertRaises(target.DeviceCheckpointError) as error:
                self.save()
        self.assertEqual(error.exception.code, "export_ticket")

    def test_inconsistent_resealed_cpu_reservation_refuses_before_io(self):
        for kind in ("layout", "terms"):
            reservation = copy.deepcopy(self.export.evidence["reservation"])
            if kind == "layout":
                reservation["numeric_bytes"] += 8
            else:
                reservation["terms"]["static_inputs_bound"] += 1
                reservation["required_host_data_bytes"] += 1
            evidence = {**self.export.evidence, "reservation": reservation}
            issued = self.reissue_cpu_evidence(evidence)
            with self.subTest(kind=kind), \
                    mock.patch.object(native, "save_exact_physical_checkpoint", side_effect=AssertionError("save")), \
                    mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("load")):
                for action in (lambda: target.save_device_boundary_physical(
                        self.path, issued, self.config, inputs=self.inputs, limits=self.limits),
                        lambda: self.load(export=issued)):
                    with self.assertRaises(target.DeviceCheckpointError) as error:
                        action()
                    self.assertEqual(error.exception.code, "host_bytes_cap")
            self.assertFalse(self.path.exists())

    def test_unbounded_config_refuses_before_fingerprint_or_io(self):
        bad = []
        cursor = bad
        for _ in range(30):
            child = []
            cursor.append(child)
            cursor = child
        with mock.patch.object(native, "save_exact_physical_checkpoint", side_effect=AssertionError("save")):
            with self.assertRaises(DenseBedError) as error:
                target.save_device_boundary_physical(
                    self.path, self.export, bad, inputs=self.inputs, limits=self.limits)
        self.assertEqual(error.exception.code, "invalid_boundary_physical")

    def test_mismatched_archive_layout_refuses_before_native_load(self):
        self.save()
        payload = self.path / native.PAYLOADS[0]
        with payload.open("wb") as stream:
            np.savez(stream, incompatible=np.zeros(100))
        with mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("load")):
            with self.assertRaises(target.DeviceCheckpointError) as error:
                self.load()
        self.assertEqual(error.exception.code, "readback_layout")
        self.assertTrue(payload.is_file())

    def test_extra_array_or_oversized_scalar_refuses_before_native_load(self):
        self.save()
        payload = self.path / native.PAYLOADS[0]
        with np.load(payload, allow_pickle=False) as archive:
            members = {name: archive[name] for name in archive.files}
        additions = (self.export.physical.active_layer.mass_kg,
                     np.array("x" * 300_000))
        self.assertGreater(additions[1].nbytes, self.limits.max_metadata_bytes)
        for extra in additions:
            changed = {**members, "reason": extra}
            with payload.open("wb") as stream:
                np.savez(stream, **changed)
            self.assertLess(sum((self.path / name).stat().st_size for name in native.PAYLOADS),
                            self.limits.max_native_bytes)
            with self.subTest(rank=extra.ndim), \
                    mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("load")):
                with self.assertRaises(target.DeviceCheckpointError) as error:
                    self.load()
                self.assertEqual(error.exception.code, "readback_layout")
                expected = ("native scalar payload bytes exceed metadata cap" if extra.ndim == 0 else
                            "native array headers differ from admitted physical layout")
                self.assertEqual(error.exception.detail, expected)

    def test_optional_native_fields_refuse_before_native_load(self):
        self.save()
        cases = ((native.PAYLOADS[0], "vegetation_objects_center_x_m"),
                 (native.PAYLOADS[1], "baseline__active_layer_mass_by_class_kg"))
        for name, extra_name in cases:
            payload = self.path / name
            original = payload.read_bytes()
            with np.load(payload, allow_pickle=False) as archive:
                members = {key: archive[key] for key in archive.files}
            with payload.open("wb") as stream:
                np.savez(stream, **members, **{extra_name: np.zeros(0)})
            try:
                with self.subTest(payload=name), \
                        mock.patch.object(native, "load_exact_physical_checkpoint", side_effect=AssertionError("load")):
                    with self.assertRaises(target.DeviceCheckpointError) as error:
                        self.load()
                    self.assertEqual(error.exception.code, "readback_layout")
                    self.assertEqual(error.exception.detail,
                                     "native field inventory differs from admitted physical state")
            finally:
                payload.write_bytes(original)

    def test_tampered_owned_physical_bytes_refuse_before_directory_creation(self):
        a = self.export.physical.active_layer.mass_kg
        a.setflags(write=True)
        a.flat[0] = np.nextafter(a.flat[0], np.inf)
        a.setflags(write=False)
        with self.assertRaises(target.DeviceCheckpointError) as error:
            self.save()
        self.assertEqual(error.exception.code, "physical_identity")
        self.assertFalse(self.path.exists())

    def test_writable_physical_payload_is_refused(self):
        self.export.physical.active_layer.mass_kg.setflags(write=True)
        with self.assertRaises(DenseBedError) as error:
            self.save()
        self.assertEqual(error.exception.code, "invalid_boundary_physical")
        self.assertFalse(self.path.exists())

    def test_existing_target_is_preserved_and_not_retried(self):
        self.save()
        manifest = (self.path / native.MANIFEST).read_bytes()
        with self.assertRaises(native.PhysicalCheckpointError):
            self.save()
        self.assertEqual((self.path / native.MANIFEST).read_bytes(), manifest)

    def test_native_indeterminate_publication_exception_propagates(self):
        error = native.CheckpointPublicationUncertain("test-path", "test-seal")
        with mock.patch.object(native, "save_exact_physical_checkpoint", side_effect=error):
            with self.assertRaises(native.CheckpointPublicationUncertain) as caught:
                self.save()
        self.assertIs(caught.exception, error)

    def test_zero_free_inodes_refuses_before_native_save(self):
        actual = target.os.statvfs(self.path.parent)
        usage = SimpleNamespace(free=1 << 40)
        probe = mock.Mock(f_favail=0, f_bsize=actual.f_bsize)
        with mock.patch.object(target.os, "statvfs", return_value=probe), \
                mock.patch.object(target.shutil, "disk_usage", return_value=usage), \
                mock.patch.object(native, "save_exact_physical_checkpoint", side_effect=AssertionError("save")):
            with self.assertRaises(target.DeviceCheckpointError) as error:
                self.save()
        self.assertEqual(error.exception.code, "disk_admission")
        self.assertFalse(self.path.exists())

    def test_corrupted_archive_is_refused_and_retained(self):
        self.save()
        path = self.path / native.PAYLOADS[1]
        raw = path.read_bytes()
        path.write_bytes(raw[:-1])
        with self.assertRaises(native.PhysicalCheckpointError):
            self.load()
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
