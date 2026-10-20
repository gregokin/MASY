"""Bounded synthetic component checks only: no native event/transport runs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fixture_dense_alternation as fixture
import numpy as np

from masy import dense_original_inputs as codec

SOURCES = {
    "kind": "source_manifest_prototype",
    "masy": hashlib.sha256(b"synthetic masy source reference").hexdigest(),
    "maple": hashlib.sha256(b"synthetic maple source reference").hexdigest(),
    "syrup": hashlib.sha256(b"synthetic syrup source reference").hexdigest(),
}


def seal(path, manifest):
    value = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(codec._json(value)).hexdigest()
    (path / "manifest.json").chmod(384)
    (path / "manifest.json").write_bytes(codec._json(manifest))
    return manifest["manifest_sha256"]


def wire_rows(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from wire_rows(value)
    elif isinstance(node, list):
        for value in node:
            yield from wire_rows(value)


def exact(value):
    """Independent content/type signature, with no codec wire/helper dependency."""
    from enum import Enum

    if type(value) is np.ndarray:
        return (
            "array",
            value.dtype.str,
            value.shape,
            hashlib.sha256(value.tobytes()).hexdigest(),
        )
    if isinstance(value, np.generic):
        return ("scalar", value.dtype.str, value.tobytes().hex())
    if value is np:
        return ("module", "numpy")
    if isinstance(value, Enum):
        return (type(value).__module__, type(value).__qualname__, exact(value.value))
    if dataclasses.is_dataclass(value):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(((k, exact(v)) for k, v in sorted(vars(value).items()))),
        )
    if type(value) is dict:
        return ("dict", tuple(((k, exact(v)) for k, v in value.items())))
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(exact(v) for v in value))
    return (type(value).__name__, value.hex() if type(value) is float else value)


class OriginalPackChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bed = fixture.build_bed()
        cls.runner = fixture.make_runner(
            cls.bed,
            fixture.schedule(
                fixture.wind(), fixture.rain(), fixture.wind(), fixture.rain()
            ),
        )
        cls.pack = codec.capture_original_inputs(cls.runner, source_identity=SOURCES)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def saved(self):
        path = self.root / "pack"
        identity = codec.save_original_inputs(path, self.pack)
        return (path, identity)

    def test_01_capture_no_transport_or_owner_change(self):
        before = self.bed.identity_sha256()
        native = codec._native()
        with (
            mock.patch.object(
                native.runner.DenseRunner,
                "step",
                side_effect=AssertionError("transport forbidden"),
            ),
            mock.patch.object(
                native.runner,
                "_advance_wind_block",
                side_effect=AssertionError("transport forbidden"),
            ),
            mock.patch.object(
                native.runner,
                "_finalize_wind_event",
                side_effect=AssertionError("finalization forbidden"),
            ),
            mock.patch.object(
                native.runner,
                "_complete_event",
                side_effect=AssertionError("water forbidden"),
            ),
            mock.patch.object(
                native.bed,
                "build_dense_bed",
                side_effect=AssertionError("bed construction forbidden"),
            ),
        ):
            captured = codec.capture_original_inputs(
                self.runner, source_identity=SOURCES
            )
        self.assertEqual(captured.content_sha256, self.pack.content_sha256)
        self.assertEqual(
            codec.validate_original_inputs(self.pack), self.pack.content_sha256
        )
        self.assertEqual(self.bed.identity_sha256(), before)
        self.assertIs(self.pack.capsule.context, self.bed.context)
        self.assertIs(self.pack.wind.initial_mobile_state, self.runner._pool)

    def test_02_complete_native_roundtrip(self):
        path, identity = self.saved()
        with (
            mock.patch.object(
                codec._native().bed,
                "build_dense_bed",
                side_effect=AssertionError("no bed construction"),
            ),
            mock.patch.object(
                codec._native().runner.DenseRunner,
                "__init__",
                side_effect=AssertionError("no runner initialization"),
            ),
        ):
            restored = codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
            )
        self.assertEqual(exact(self.pack), exact(restored))
        for name in (
            "geometry",
            "grain_classes",
            "topographic_commit_spec",
            "avalanching_spec",
        ):
            self.assertIs(
                getattr(restored.wind.config, name),
                getattr(restored.capsule.context, name),
            )
        self.assertIs(
            restored.wind.config.initial_available_fraction,
            restored.capsule.context.initial_available_fraction,
        )
        self.assertEqual(
            restored.capsule.binding_sha256, self.pack.capsule.binding_sha256
        )
        self.assertEqual(restored.static_record, self.pack.static_record)
        self.assertEqual(restored.schedule.digest, self.pack.schedule.digest)
        self.assertFalse(restored.initial_inventory_kg.flags.writeable)
        self.assertFalse(
            np.shares_memory(
                restored.wind.initial_mobile_state.mass_kg,
                self.pack.wind.initial_mobile_state.mass_kg,
            )
        )
        self.assertFalse(hasattr(restored, "bed"))

    def test_03_oversized_raw_stops_before_payload_hash_or_header(self):
        path, identity = self.saved()
        with (
            mock.patch.object(
                codec, "_hash_fd", side_effect=AssertionError("must not hash")
            ),
            mock.patch.object(
                np.lib.format,
                "read_magic",
                side_effect=AssertionError("must not parse"),
            ),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
                limits=dataclasses.replace(
                    codec.OriginalPackLimits(), max_serialized_bytes=1
                ),
            )
        self.assertEqual(error.exception.code, "serialized_cap")

    def test_04_expanded_aggregate_stops_before_first_allocation_or_hash(self):
        path, identity = self.saved()
        with (
            mock.patch.object(
                codec, "_hash_fd", side_effect=AssertionError("must not hash")
            ),
            mock.patch.object(
                np, "fromfile", side_effect=AssertionError("must not expand")
            ),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
                limits=dataclasses.replace(
                    codec.OriginalPackLimits(), max_numeric_bytes=1
                ),
            )
        self.assertEqual(error.exception.code, "array_cap")

    def test_05_source_binding_before_payload_hash(self):
        path, identity = self.saved()
        with (
            mock.patch.object(
                codec, "_hash_fd", side_effect=AssertionError("must not hash")
            ),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity={**SOURCES, "maple": "0" * 64},
            )
        self.assertEqual(error.exception.code, "source_identity")

    def test_06_no_clobber(self):
        path, identity = self.saved()
        before = (path / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            codec.save_original_inputs(path, self.pack)
        self.assertEqual((path / "manifest.json").read_bytes(), before)
        self.assertEqual(json.loads(before)["manifest_sha256"], identity)

    def test_07_extra_instance_field_before_hash(self):
        cfg = dataclasses.replace(self.pack.wind.config)
        object.__setattr__(cfg, "unknown_future_state", 7)
        changed = dataclasses.replace(
            self.pack, wind=dataclasses.replace(self.pack.wind, config=cfg)
        )
        with (
            mock.patch.object(
                codec.hashlib,
                "sha256",
                side_effect=AssertionError("hash before closure"),
            ),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.validate_original_inputs(changed)
        self.assertEqual(error.exception.code, "unsupported_layout")

    def test_08_deleted_default_field_before_hash(self):
        cfg = dataclasses.replace(self.pack.wind.config)
        del vars(cfg)["transport_accuracy"]
        changed = dataclasses.replace(
            self.pack, wind=dataclasses.replace(self.pack.wind, config=cfg)
        )
        with (
            mock.patch.object(
                codec.hashlib,
                "sha256",
                side_effect=AssertionError("hash before closure"),
            ),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.validate_original_inputs(changed)
        self.assertEqual(error.exception.code, "unsupported_layout")

    def test_09_capture_tiny_budget_before_source_hash(self):
        with (
            mock.patch.object(
                codec.hashlib, "sha256", side_effect=AssertionError("hash before cap")
            ),
            self.assertRaises(codec.OriginalInputsError),
        ):
            codec.capture_original_inputs(
                self.runner,
                source_identity=SOURCES,
                limits=dataclasses.replace(
                    codec.OriginalPackLimits(), max_numeric_bytes=1
                ),
            )

    def test_10_changed_derived_column_refuses(self):
        c = self.pack.water.column
        changed = dataclasses.replace(
            c, storage_max_m=np.nextafter(c.storage_max_m, np.inf)
        )
        pack = dataclasses.replace(
            self.pack, water=dataclasses.replace(self.pack.water, column=changed)
        )
        with self.assertRaises(codec.OriginalInputsError) as error:
            codec.validate_original_inputs(pack)
        self.assertEqual(error.exception.code, "recipe_mismatch")

    def test_11_raw_hop_float32_not_normalized(self):
        bins = self.pack.wind.config.hop_bins
        changed = dataclasses.replace(
            bins, hop_distance_m=bins.hop_distance_m.astype(np.float32)
        )
        cfg = dataclasses.replace(self.pack.wind.config, hop_bins=changed)
        pack = dataclasses.replace(
            self.pack, wind=dataclasses.replace(self.pack.wind, config=cfg)
        )
        with self.assertRaises(codec.OriginalInputsError) as error:
            codec.validate_original_inputs(pack)
        self.assertEqual(error.exception.code, "recipe_mismatch")

    def test_12_unsupported_recipe(self):
        changed = dataclasses.replace(
            self.pack, recipes={**self.pack.recipes, "sediment": "custom_inverse_SI"}
        )
        with self.assertRaises(codec.OriginalInputsError) as error:
            codec.validate_original_inputs(changed)
        self.assertEqual(error.exception.code, "recipe_unsupported")

    def test_13_symlink_directory_and_ancestor(self):
        path, identity = self.saved()
        link = self.root / "alias"
        link.symlink_to(path, target_is_directory=True)
        with self.assertRaises(OSError):
            codec.load_original_inputs(
                link,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
            )
        parentlink = self.root / "ancestor"
        parentlink.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(OSError):
            codec.save_original_inputs(parentlink / "new", self.pack)
        self.assertFalse((self.root / "new").exists())

    def test_14_symlink_numeric_file(self):
        path, identity = self.saved()
        array = next(path.glob("a_*.npy"))
        outside = self.root / "outside"
        outside.write_bytes(b"kept local")
        array.unlink()
        array.symlink_to(outside)
        with self.assertRaises(codec.OriginalInputsError) as error:
            codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
            )
        self.assertEqual(error.exception.code, "file_type")

    def test_15_fifo_is_nonblocking_refusal(self):
        path, identity = self.saved()
        array = next(path.glob("a_*.npy"))
        array.unlink()
        os.mkfifo(array)
        with self.assertRaises(codec.OriginalInputsError) as error:
            codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
            )
        self.assertEqual(error.exception.code, "file_type")

    def test_16_foreign_hardlink(self):
        path, identity = self.saved()
        os.link(next(path.glob("a_*.npy")), self.root / "foreign")
        with self.assertRaises(codec.OriginalInputsError) as error:
            codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
            )
        self.assertEqual(error.exception.code, "file_type")

    def test_17_manifest_corruption(self):
        path, _identity = self.saved()
        m = json.loads((path / "manifest.json").read_bytes())
        m["profile"] = "GPU"
        changed = seal(path, m)
        with self.assertRaises(codec.OriginalInputsError) as error:
            codec.load_original_inputs(
                path, expected_manifest_sha256=changed, expected_source_identity=SOURCES
            )
        self.assertEqual(error.exception.code, "manifest_identity")

    def test_18_unhashable_dtype_typed_refusal(self):
        path, _ = self.saved()
        m = json.loads((path / "manifest.json").read_bytes())
        m["files"][0]["dtype"] = []
        changed = seal(path, m)
        with self.assertRaises(codec.OriginalInputsError) as error:
            codec.load_original_inputs(
                path, expected_manifest_sha256=changed, expected_source_identity=SOURCES
            )
        self.assertEqual(error.exception.code, "array_layout")

    def test_19_path_traversal_before_payload_hash(self):
        path, _ = self.saved()
        m = json.loads((path / "manifest.json").read_bytes())
        m["files"][0]["name"] = "../outside.npy"
        changed = seal(path, m)
        with (
            mock.patch.object(
                codec, "_hash_fd", side_effect=AssertionError("hash after unsafe path")
            ),
            self.assertRaises(codec.OriginalInputsError),
        ):
            codec.load_original_inputs(
                path, expected_manifest_sha256=changed, expected_source_identity=SOURCES
            )

    def test_20_corrupt_numeric_byte(self):
        path, identity = self.saved()
        array = next(path.glob("a_*.npy"))
        raw = bytearray(array.read_bytes())
        raw[-1] ^= 1
        array.chmod(384)
        array.write_bytes(raw)
        with self.assertRaises(codec.OriginalInputsError) as error:
            codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
            )
        self.assertEqual(error.exception.code, "file_hash")

    def test_21_overflow_json_and_duplicate_error_codes(self):
        for raw, code in (
            (b'{"x":1e999}', "metadata_nonfinite"),
            (b'{"x":1,"x":2}', "metadata_duplicate"),
        ):
            with (
                self.subTest(code=code),
                self.assertRaises(codec.OriginalInputsError) as error,
            ):
                codec._strict_json(raw, codec.OriginalPackLimits())
            self.assertEqual(error.exception.code, code)

    def test_22_transient_written_corruption_never_seals(self):
        path = self.root / "pack"
        original = np.lib.format.write_array

        def corrupt(stream, array, **kwargs):
            original(stream, array, **kwargs)
            stream.seek(-1, os.SEEK_END)
            stream.write(b"\xa5")

        with (
            mock.patch.object(np.lib.format, "write_array", side_effect=corrupt),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.save_original_inputs(path, self.pack)
        self.assertEqual(error.exception.code, "array_hash")
        self.assertFalse((path / "manifest.json").exists())
        self.assertEqual(
            codec.validate_original_inputs(self.pack), self.pack.content_sha256
        )

    def test_23_seal_unlink_ambiguity_and_exact_leftover_recovery(self):
        path = self.root / "pack"
        unlink = os.unlink

        def fail(name, *args, **kwargs):
            if name == "manifest.tmp":
                raise OSError("injected after published seal")
            return unlink(name, *args, **kwargs)

        with (
            mock.patch.object(codec.os, "unlink", side_effect=fail),
            self.assertRaises(codec.OriginalPackPublicationUncertain),
        ):
            codec.save_original_inputs(path, self.pack)
        self.assertEqual(
            (path / "manifest.tmp").stat().st_ino,
            (path / "manifest.json").stat().st_ino,
        )
        identity = json.loads((path / "manifest.json").read_bytes())["manifest_sha256"]
        restored = codec.load_original_inputs(
            path, expected_manifest_sha256=identity, expected_source_identity=SOURCES
        )
        self.assertEqual(exact(restored), exact(self.pack))
        self.assertTrue((path / "manifest.tmp").exists())

    def test_24_preseal_fsync_failure_keeps_unsealed_evidence(self):
        path = self.root / "pack"
        with (
            mock.patch.object(
                codec.os, "fsync", side_effect=OSError("injected before seal")
            ),
            self.assertRaises(OSError),
        ):
            codec.save_original_inputs(path, self.pack)
        self.assertFalse((path / "manifest.json").exists())
        with self.assertRaises(codec.OriginalInputsError):
            codec.load_original_inputs(
                path,
                expected_manifest_sha256="0" * 64,
                expected_source_identity=SOURCES,
            )

    def test_25_closed_scalars_tuple_list_none_signed_zero(self):
        enc = codec._Encoder(
            codec.OriginalPackLimits(), codec._registry(), hash_arrays=False
        )
        value = (None, [False, 0, 0.0, -0.0, np.float32(-0.0), np.int32(0)])
        wire = codec._flat_wire({"anchors": [], "body": enc.node(value)})
        decoder = codec._Decoder(
            [], codec._registry(), codec.OriginalPackLimits(), wire["nodes"]
        )
        self.assertEqual(exact(decoder.node(wire["body"])), exact(value))

    def test_26_object_header_before_expansion(self):
        path, _ = self.saved()
        m = json.loads((path / "manifest.json").read_bytes())
        row = m["files"][0]
        array = path / row["name"]
        array.chmod(384)
        with array.open("wb") as stream:
            np.lib.format.write_array(
                stream, np.array([None], dtype=object), allow_pickle=True
            )
        row["file_bytes"] = array.stat().st_size
        row["sha256"] = hashlib.sha256(array.read_bytes()).hexdigest()
        identity = seal(path, m)
        with (
            mock.patch.object(
                np, "fromfile", side_effect=AssertionError("object expansion forbidden")
            ),
            self.assertRaises(codec.OriginalInputsError),
        ):
            codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
            )

    def test_27_defaulted_event_field_deletion_refuses(self):
        import copy

        schedule = copy.deepcopy(self.pack.schedule)
        del vars(schedule.events[0])["duration_s"]
        with self.assertRaises(codec.OriginalInputsError):
            codec.validate_original_inputs(
                dataclasses.replace(self.pack, schedule=schedule)
            )

    def test_28_save_serialized_cap_before_directory_creation(self):
        path = self.root / "pack"
        with self.assertRaises(codec.OriginalInputsError) as error:
            codec.save_original_inputs(
                path,
                self.pack,
                limits=dataclasses.replace(
                    codec.OriginalPackLimits(), max_serialized_bytes=1
                ),
            )
        self.assertEqual(error.exception.code, "serialized_cap")
        self.assertFalse(path.exists())

    def test_29_bool_cursor_refused(self):
        original = self.runner._next
        try:
            self.runner._next = False
            with self.assertRaises(codec.OriginalInputsError):
                codec.capture_original_inputs(self.runner, source_identity=SOURCES)
        finally:
            self.runner._next = original

    def test_30_changed_zero_totals_refused_without_event(self):
        original = self.runner._totals["n_event_reroutes"]
        try:
            self.runner._totals["n_event_reroutes"] = 1
            with self.assertRaises(codec.OriginalInputsError):
                codec.capture_original_inputs(self.runner, source_identity=SOURCES)
        finally:
            self.runner._totals["n_event_reroutes"] = original

    def test_31_masked_resolved_rainfall_and_signed_zero(self):
        from maple_syrup.rainfall import rainfall_field

        scale = np.linspace(0.25, 2.0, fixture.NY, dtype=np.float64)[:, None]
        scale[0, 0] = -0.0
        active = np.array([True, False, True, True, False, True])[:, None]
        field = rainfall_field(
            fixture.NY, fixture.NX, scale=scale, active_mask=active, xp=np
        )
        water = dataclasses.replace(self.pack.water, field=field)
        runner = fixture.make_runner(self.bed, self.pack.schedule, water=water)
        captured = codec.capture_original_inputs(runner, source_identity=SOURCES)
        path = self.root / "masked"
        identity = codec.save_original_inputs(path, captured)
        restored = codec.load_original_inputs(
            path, expected_manifest_sha256=identity, expected_source_identity=SOURCES
        )
        self.assertEqual(exact(restored.water.field), exact(field))
        self.assertTrue(np.signbit(restored.water.field.multiplier[0, 0]))
        self.assertTrue(np.all(restored.water.field.multiplier[~active] == 0.0))
        self.assertFalse(restored.water.field.multiplier.flags.writeable)

    def test_32_original_pool_signed_zero_and_no_source_flag_change(self):
        pool = fixture.empty_pool()
        pool.position_x_m[0] = -0.0
        runner = fixture.make_runner(self.bed, self.pack.schedule, pool=pool)
        captured = codec.capture_original_inputs(runner, source_identity=SOURCES)
        flags = {
            k: v.flags.writeable for k, v in vars(pool).items() if type(v) is np.ndarray
        }
        path = self.root / "signed"
        identity = codec.save_original_inputs(path, captured)
        restored = codec.load_original_inputs(
            path, expected_manifest_sha256=identity, expected_source_identity=SOURCES
        )
        self.assertEqual(exact(restored.wind.initial_mobile_state), exact(pool))
        self.assertTrue(np.signbit(restored.wind.initial_mobile_state.position_x_m[0]))
        self.assertEqual(
            {
                k: v.flags.writeable
                for k, v in vars(pool).items()
                if type(v) is np.ndarray
            },
            flags,
        )

    def test_33_resealed_raw_float32_hop_cache_refuses_normalization(self):
        path, _ = self.saved()
        m = json.loads((path / "manifest.json").read_bytes())
        table = m["wire"]["nodes"]
        hop = next(
            node
            for node in table
            if node.get("t") == "class" and node["c"].endswith(":HopQuantileBins")
        )
        array_node = table[dict(hop["v"])["hop_distance_m"]]
        row = m["files"][array_node["v"]]
        array = path / row["name"]
        with array.open("rb") as stream:
            raw = np.lib.format.read_array(stream, allow_pickle=False)
        changed = raw.astype(np.float32)
        array.chmod(0o600)
        with array.open("wb") as stream:
            np.lib.format.write_array(stream, changed, allow_pickle=False)
        row.update(
            dtype=changed.dtype.str,
            data_bytes=changed.nbytes,
            data_sha256=hashlib.sha256(changed.tobytes()).hexdigest(),
            file_bytes=array.stat().st_size,
            sha256=hashlib.sha256(array.read_bytes()).hexdigest(),
        )
        identity = seal(path, m)
        with self.assertRaises(codec.OriginalInputsError) as error:
            codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
            )
        self.assertEqual(error.exception.code, "recipe_mismatch")

    def test_34_declared_factory_bins_cap_before_native_allocation(self):
        particle = dataclasses.replace(
            self.pack.wind.config.particle_transport, quantile_bins=1 << 28
        )
        config = dataclasses.replace(self.pack.wind.config, particle_transport=particle)
        changed = dataclasses.replace(
            self.pack, wind=dataclasses.replace(self.pack.wind, config=config)
        )
        with (
            mock.patch(
                "maple.aeolian.trajectories.quantile_bins.build_hop_quantile_bins",
                side_effect=AssertionError("unbounded factory"),
            ),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.validate_original_inputs(changed)
        self.assertEqual(error.exception.code, "profile_cap")

    def test_35_opened_fd_closed_when_fstat_fails(self):
        path, identity = self.saved()
        before = len(list(Path("/proc/self/fd").iterdir()))
        with (
            mock.patch.object(
                codec.os, "fstat", side_effect=OSError("injected fstat refusal")
            ),
            self.assertRaises(OSError),
        ):
            codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
            )
        self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), before)

    def test_36_header_cap_before_expansion(self):
        import struct

        path, _ = self.saved()
        m = json.loads((path / "manifest.json").read_bytes())
        row = m["files"][0]
        array = path / row["name"]
        with array.open("rb") as stream:
            np.lib.format.read_magic(stream)
            np.lib.format.read_array_header_1_0(stream)
            data = stream.read()
        header = repr(
            {
                "descr": row["dtype"],
                "fortran_order": False,
                "shape": tuple(row["shape"]),
            }
        ).encode()
        padded = header + b" " * (8192 - len(header) - 1) + b"\n"
        array.chmod(0o600)
        array.write_bytes(
            b"\x93NUMPY\x01\x00" + struct.pack("<H", 8192) + padded + data
        )
        row.update(
            file_bytes=array.stat().st_size,
            sha256=hashlib.sha256(array.read_bytes()).hexdigest(),
        )
        identity = seal(path, m)
        with (
            mock.patch.object(
                np.lib.format,
                "read_array_header_1_0",
                side_effect=AssertionError("v1 parser must not read oversized header"),
            ),
            mock.patch.object(
                np, "fromfile", side_effect=AssertionError("oversize header expansion")
            ),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
            )
        self.assertEqual(error.exception.code, "header_cap")

    def test_37_measure_reference_pack(self):
        path, _ = self.saved()
        m = json.loads((path / "manifest.json").read_bytes())
        metric = {
            "component": "synthetic_original_pack",
            "numeric_files": len(m["files"]),
            "numeric_data_bytes": sum(r["data_bytes"] for r in m["files"]),
            "serialized_bytes": sum(f.stat().st_size for f in path.iterdir()),
            "metadata_bytes": (path / "manifest.json").stat().st_size,
        }
        self.assertLessEqual(metric["numeric_files"], 256)
        self.assertLessEqual(metric["numeric_data_bytes"], 64 << 20)
        print(json.dumps(metric, sort_keys=True), flush=True)

    def test_38_wrong_registered_root_refuses_before_hash(self):
        changed = dataclasses.replace(self.pack, wind=self.pack.control)
        with (
            mock.patch.object(
                codec.hashlib, "sha256", side_effect=AssertionError("early hash")
            ),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.validate_original_inputs(changed)
        self.assertEqual(error.exception.code, "unsupported_layout")

    def test_39_scalar_and_unshared_availability_anchors_refuse(self):
        for scalar in (True, False):
            available = (
                None
                if scalar
                else self.pack.capsule.context.initial_available_fraction.copy()
            )
            config = dataclasses.replace(
                self.pack.wind.config, initial_available_fraction=available
            )
            changed = dataclasses.replace(
                self.pack, wind=dataclasses.replace(self.pack.wind, config=config)
            )
            with (
                self.subTest(scalar=scalar),
                mock.patch.object(
                    codec.hashlib, "sha256", side_effect=AssertionError("early hash")
                ),
                self.assertRaises(codec.OriginalInputsError) as error,
            ):
                codec.validate_original_inputs(changed)
            self.assertEqual(error.exception.code, "anchor_layout")

    def test_40_aggregate_hint_traversal_is_bounded_before_hash(self):
        original = self.pack.wind.config.particle_transport
        spec = next(iter(original.hop_distance_by_class.values()))
        particle = dataclasses.replace(
            original,
            hop_distance_by_class={f"synthetic_{i}": spec for i in range(10000)},
        )
        config = dataclasses.replace(self.pack.wind.config, particle_transport=particle)
        changed = dataclasses.replace(
            self.pack, wind=dataclasses.replace(self.pack.wind, config=config)
        )
        with (
            mock.patch.object(codec, "_hint", wraps=codec._hint) as audit,
            mock.patch.object(
                codec.hashlib, "sha256", side_effect=AssertionError("early hash")
            ),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.validate_original_inputs(
                changed,
                limits=dataclasses.replace(codec.OriginalPackLimits(), max_nodes=1000),
            )
        self.assertEqual(error.exception.code, "metadata_cap")
        self.assertLessEqual(audit.call_count, 1001)

    def test_41_genuine_label_refuses_before_hash(self):
        changed = dataclasses.replace(
            self.pack,
            source_identity={**SOURCES, "kind": "genuine_locked_runtime"},
        )
        with (
            mock.patch.object(
                codec.hashlib, "sha256", side_effect=AssertionError("early hash")
            ),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.validate_original_inputs(changed)
        self.assertEqual(error.exception.code, "source_identity")

    def test_42_mutated_schedule_scalar_kind_refuses_before_hash(self):
        event = dataclasses.replace(self.pack.schedule.events[0])
        object.__setattr__(event, "duration_s", 1)
        schedule = dataclasses.replace(
            self.pack.schedule, events=(event, *self.pack.schedule.events[1:])
        )
        changed = dataclasses.replace(self.pack, schedule=schedule)
        with (
            mock.patch.object(
                codec.hashlib, "sha256", side_effect=AssertionError("early hash")
            ),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.validate_original_inputs(changed)
        self.assertEqual(error.exception.code, "schedule_invalid")

    def test_43_actual_native_terrain_validation_refuses_invalid_friction(self):
        from maple_syrup import sediment_bed

        terrain = dataclasses.replace(
            self.pack.water.terrain,
            friction_factor=-np.ones_like(self.pack.water.terrain.friction_factor),
        )
        changed = dataclasses.replace(
            self.pack, water=dataclasses.replace(self.pack.water, terrain=terrain)
        )
        with (
            mock.patch.object(
                sediment_bed,
                "routing_graph_from_elevation",
                wraps=sediment_bed.routing_graph_from_elevation,
            ) as audit,
            self.assertRaisesRegex(ValueError, "friction_factor"),
        ):
            codec.validate_original_inputs(changed)
        audit.assert_called_once()

    def test_44_every_source_array_and_owner_stays_unchanged(self):
        arrays, visited = {}, set()

        def visit(value):
            if id(value) in visited:
                return
            visited.add(id(value))
            if type(value) is np.ndarray:
                arrays[id(value)] = value
            elif dataclasses.is_dataclass(value):
                for item in vars(value).values():
                    visit(item)
            elif type(value) is dict:
                for item in value.values():
                    visit(item)
            elif type(value) in (list, tuple):
                for item in value:
                    visit(item)

        visit(self.pack)
        before = {
            key: (exact(value), value.flags.writeable) for key, value in arrays.items()
        }
        owner = self.bed.identity_sha256()
        codec.validate_original_inputs(self.pack)
        self.saved()
        self.assertEqual(
            before,
            {
                key: (exact(value), value.flags.writeable)
                for key, value in arrays.items()
            },
        )
        self.assertEqual(owner, self.bed.identity_sha256())

    def test_45_seal_syscall_return_gap_is_truthfully_uncertain(self):
        real_link = os.link

        def linked_then_failed(*args, **kwargs):
            real_link(*args, **kwargs)
            raise OSError("injected exception after actual seal publication")

        path = self.root / "gap"
        with (
            mock.patch.object(codec.os, "link", side_effect=linked_then_failed),
            self.assertRaises(codec.OriginalPackPublicationUncertain),
        ):
            codec.save_original_inputs(path, self.pack)
        manifest = json.loads((path / "manifest.json").read_bytes())
        self.assertEqual(
            (path / "manifest.tmp").stat().st_ino,
            (path / "manifest.json").stat().st_ino,
        )
        restored = codec.load_original_inputs(
            path,
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_source_identity=SOURCES,
        )
        self.assertEqual(exact(restored), exact(self.pack))
        self.assertTrue((path / "manifest.tmp").exists())

    def test_46_declared_header_length_cap_before_native_parser_or_large_read(self):
        path, _ = self.saved()
        manifest = json.loads((path / "manifest.json").read_bytes())
        row = manifest["files"][0]
        array = path / row["name"]
        array.chmod(0o600)
        array.write_bytes(b"\x93NUMPY\x02\x00\xff\xff\xff\xff")
        row.update(
            file_bytes=array.stat().st_size,
            sha256=hashlib.sha256(array.read_bytes()).hexdigest(),
        )
        identity = seal(path, manifest)
        with (
            mock.patch.object(
                np.lib.format,
                "read_array_header_2_0",
                side_effect=AssertionError("native parser must not read a huge header"),
            ),
            mock.patch.object(
                np, "fromfile", side_effect=AssertionError("expansion forbidden")
            ),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
            )
        self.assertEqual(error.exception.code, "header_cap")

    def test_47_mutated_genesis_clock_kind_bits_and_container_refuse(self):
        import copy

        for field, changed_value in (
            ("epoch", False),
            ("epoch", -0.0),
            ("_entries", ()),
        ):
            runner = copy.copy(self.runner)
            runner._ledger = copy.copy(runner._ledger)
            setattr(runner._ledger, field, changed_value)
            with (
                self.subTest(field=field, value=repr(changed_value)),
                self.assertRaises(codec.OriginalInputsError) as error,
            ):
                codec.capture_original_inputs(runner, source_identity=SOURCES)
            self.assertEqual(error.exception.code, "not_genesis")

    def test_48_untyped_source_kind_refuses_before_hash(self):
        changed = dataclasses.replace(
            self.pack, source_identity={**SOURCES, "kind": np.array([1, 2])}
        )
        with (
            mock.patch.object(
                codec.hashlib, "sha256", side_effect=AssertionError("early hash")
            ),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.validate_original_inputs(changed)
        self.assertEqual(error.exception.code, "source_identity")

    def test_49_leftover_manifest_hardlink_charged_once_at_exact_raw_cap(self):
        path, identity = self.saved()
        unique_bytes = sum(file.stat().st_size for file in path.iterdir())
        os.link(path / "manifest.json", path / "manifest.tmp")
        restored = codec.load_original_inputs(
            path,
            expected_manifest_sha256=identity,
            expected_source_identity=SOURCES,
            limits=dataclasses.replace(
                codec.OriginalPackLimits(), max_serialized_bytes=unique_bytes
            ),
        )
        self.assertEqual(exact(restored), exact(self.pack))
        self.assertTrue((path / "manifest.tmp").exists())

    def test_50_unreferenced_numeric_file_refuses(self):
        path, _ = self.saved()
        manifest = json.loads((path / "manifest.json").read_bytes())
        value = np.zeros(1, dtype=np.float64)
        name = f"a_{len(manifest['files']):04d}.npy"
        with (path / name).open("wb") as stream:
            np.lib.format.write_array(stream, value, allow_pickle=False)
        manifest["files"].append(
            {
                "name": name,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "data_bytes": value.nbytes,
                "data_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                "file_bytes": (path / name).stat().st_size,
                "sha256": hashlib.sha256((path / name).read_bytes()).hexdigest(),
            }
        )
        identity = seal(path, manifest)
        with self.assertRaises(codec.OriginalInputsError) as error:
            codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
            )
        self.assertEqual(error.exception.code, "wire_invalid")
        self.assertIn("unreferenced numerical", str(error.exception))

    def test_51_new_directory_fd_closes_when_scandir_refuses(self):
        before = len(list(Path("/proc/self/fd").iterdir()))
        with (
            mock.patch.object(codec, "_names", side_effect=OSError("scandir refused")),
            self.assertRaises(OSError),
        ):
            codec.save_original_inputs(self.root / "unsealed", self.pack)
        self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), before)
        self.assertFalse((self.root / "unsealed/manifest.json").exists())

    def test_52_nonregular_file_refuses_before_any_file_open(self):
        os.mkfifo(self.root / "a_0000.npy")
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with (
                mock.patch.object(
                    codec.os, "open", side_effect=AssertionError("no device/FIFO open")
                ),
                self.assertRaises(codec.OriginalInputsError) as error,
            ):
                codec._open_file(directory, "a_0000.npy")
            self.assertEqual(error.exception.code, "file_type")
        finally:
            os.close(directory)

    def test_53_foreign_limits_never_dispatches_override(self):
        class ForeignLimits(codec.OriginalPackLimits):
            def __post_init__(self):
                raise AssertionError("foreign validator must never execute")

        limits = object.__new__(ForeignLimits)
        calls = (
            lambda: codec.capture_original_inputs(
                self.runner, source_identity=SOURCES, limits=limits
            ),
            lambda: codec.validate_original_inputs(self.pack, limits=limits),
            lambda: codec.save_original_inputs(
                self.root / "new", self.pack, limits=limits
            ),
            lambda: codec.load_original_inputs(
                self.root / "missing",
                expected_manifest_sha256="0" * 64,
                expected_source_identity=SOURCES,
                limits=limits,
            ),
        )
        for index, call in enumerate(calls):
            with (
                self.subTest(api=index),
                self.assertRaises(codec.OriginalInputsError) as error,
            ):
                call()
            self.assertEqual(error.exception.code, "unsupported_layout")
        self.assertFalse((self.root / "new").exists())

    def test_54_actual_written_manifest_checked_before_seal(self):
        real_fsync = os.fsync
        path = self.root / "bad_manifest"

        def corrupt_after_flush(fd):
            real_fsync(fd)
            actual = Path(os.readlink(f"/proc/self/fd/{fd}"))
            if actual.name == "manifest.tmp":
                actual.chmod(0o600)
                with actual.open("r+b") as stream:
                    stream.write(b"!")

        with (
            mock.patch.object(codec.os, "fsync", side_effect=corrupt_after_flush),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.save_original_inputs(path, self.pack)
        self.assertEqual(error.exception.code, "manifest_identity")
        self.assertFalse((path / "manifest.json").exists())

    def test_55_native_schedule_construction_cannot_precede_event_cap(self):
        path, _ = self.saved()
        manifest = json.loads((path / "manifest.json").read_bytes())
        table = manifest["wire"]["nodes"]
        body = table[manifest["wire"]["body"]]
        schedule = table[dict(body["v"])["schedule"]]
        events = table[dict(schedule["v"])["events"]]
        events["v"] *= 3
        identity = seal(path, manifest)
        with (
            mock.patch.object(
                codec._native().schedule,
                "from_record",
                side_effect=AssertionError("construct after event cap only"),
            ),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
            )
        self.assertEqual(error.exception.code, "profile_cap")

    def test_56_full_written_file_identity_checked_before_seal(self):
        real_verify = codec._verify_written
        changed = False

        def corrupt_padding(directory, row, limits):
            nonlocal changed
            if not changed:
                changed = True
                fd = os.open(row["name"], os.O_RDONLY, dir_fd=directory)
                try:
                    actual = Path(os.readlink(f"/proc/self/fd/{fd}"))
                finally:
                    os.close(fd)
                actual.chmod(0o600)
                with actual.open("r+b") as stream:
                    np.lib.format.read_magic(stream)
                    np.lib.format.read_array_header_1_0(stream)
                    stream.seek(stream.tell() - 2)
                    stream.write(b"\t")
            return real_verify(directory, row, limits)

        path = self.root / "bad_padding"
        with (
            mock.patch.object(codec, "_verify_written", side_effect=corrupt_padding),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.save_original_inputs(path, self.pack)
        self.assertEqual(error.exception.code, "file_hash")
        self.assertFalse((path / "manifest.json").exists())

    def test_57_prescribed_raster_modifiers_and_nonzero_initial_soil(self):
        from maple.core.types.vegetation import uniform_vegetation_raster

        vegetation = uniform_vegetation_raster(
            fixture.NY, fixture.NX, height_m=0.3, cover_fraction=0.25
        )
        modifiers = dataclasses.replace(
            fixture.neutral_threshold_modifier_state(fixture.NY, fixture.NX),
            physical_crust_multiplier=np.full((fixture.NY, fixture.NX), 1.5),
        )
        # One-time authored component initialization; never a restore path.
        with (
            mock.patch.object(
                fixture, "empty_vegetation_state", return_value=vegetation
            ),
            mock.patch.object(
                fixture, "neutral_threshold_modifier_state", return_value=modifiers
            ),
        ):
            bed = fixture.build_bed()
        water = fixture.water_inputs(
            bed,
            vegetation_cover_fraction=bed.vegetation_cover_fraction(),
            initial_soil_water_m=np.linspace(0.001, 0.006, fixture.NY)[:, None],
        )
        runner = fixture.make_runner(bed, self.pack.schedule, water=water)
        pack = codec.capture_original_inputs(runner, source_identity=SOURCES)
        owner = bed.identity_sha256()
        path = self.root / "prescribed"
        identity = codec.save_original_inputs(path, pack)
        with mock.patch.object(
            codec._native().bed,
            "build_dense_bed",
            side_effect=AssertionError("never rebuild a bed to decode originals"),
        ):
            restored = codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
            )
        self.assertEqual(exact(pack), exact(restored))
        self.assertTrue(np.all(restored.water.initial_soil_water_m > 0.0))
        self.assertTrue(np.all(restored.water.vegetation_cover_fraction > 0.0))
        self.assertEqual(restored.capsule.prescribed_digests, bed._prescribed_digests)
        self.assertFalse(restored.water.initial_soil_water_m.flags.writeable)
        self.assertEqual(owner, bed.identity_sha256())

    def test_58_save_refuses_new_foreign_numeric_hardlink_before_seal(self):
        real_verify = codec._verify_written
        changed = False

        def add_link(directory, row, limits):
            nonlocal changed
            if not changed:
                changed = True
                os.link(row["name"], self.root / "foreign", src_dir_fd=directory)
            return real_verify(directory, row, limits)

        path = self.root / "hardlink"
        with (
            mock.patch.object(codec, "_verify_written", side_effect=add_link),
            self.assertRaises(codec.OriginalInputsError) as error,
        ):
            codec.save_original_inputs(path, self.pack)
        self.assertEqual(error.exception.code, "file_type")
        self.assertFalse((path / "manifest.json").exists())

    def test_59_prechecked_inode_change_refuses_and_closes_fd(self):
        from types import SimpleNamespace

        path, _ = self.saved()
        directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        actual = os.stat("a_0000.npy", dir_fd=directory, follow_symlinks=False)
        stale = SimpleNamespace(
            st_mode=actual.st_mode, st_dev=actual.st_dev, st_ino=actual.st_ino + 1
        )
        before = len(list(Path("/proc/self/fd").iterdir()))
        try:
            with (
                mock.patch.object(codec.os, "stat", return_value=stale),
                self.assertRaises(codec.OriginalInputsError) as error,
            ):
                codec._open_file(directory, "a_0000.npy")
            self.assertEqual(error.exception.code, "file_changed")
            self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), before)
        finally:
            os.close(directory)

    def test_60_static_record_scalar_kind_mismatch_refuses_validate_and_load(self):
        import copy

        record = copy.deepcopy(self.pack.static_record)
        record["limits"]["max_events"] = float(record["limits"]["max_events"])
        changed = dataclasses.replace(self.pack, static_record=record)
        with self.assertRaises(codec.OriginalInputsError) as error:
            codec.validate_original_inputs(changed)
        self.assertEqual(error.exception.code, "static_mismatch")
        path, _ = self.saved()
        manifest = json.loads((path / "manifest.json").read_bytes())
        table = manifest["wire"]["nodes"]
        body = table[manifest["wire"]["body"]]
        record_node = table[dict(body["v"])["static_record"]]
        limits = table[dict(record_node["v"])["limits"]]
        count = table[dict(limits["v"])["max_events"]]
        count.update(t="float", v=float(count["v"]).hex())
        identity = seal(path, manifest)
        with self.assertRaises(codec.OriginalInputsError) as error:
            codec.load_original_inputs(
                path,
                expected_manifest_sha256=identity,
                expected_source_identity=SOURCES,
            )
        self.assertEqual(error.exception.code, "static_mismatch")

    def test_61_static_parts_boolean_integer_substitution_refuses(self):
        import copy

        parts = copy.deepcopy(self.pack.static_parts)

        def alter(value):
            if type(value) is dict:
                for key, item in value.items():
                    if type(item) is int and item == 1:
                        value[key] = True
                        return True
                    if alter(item):
                        return True
            elif type(value) is list:
                for index, item in enumerate(value):
                    if type(item) is int and item == 1:
                        value[index] = True
                        return True
                    if alter(item):
                        return True
            return False

        self.assertTrue(alter(parts))
        changed = dataclasses.replace(self.pack, static_parts=parts)
        with self.assertRaises(codec.OriginalInputsError) as error:
            codec.validate_original_inputs(changed)
        self.assertEqual(error.exception.code, "static_mismatch")

    def test_62_native_mapping_insertion_order_survives_full_roundtrip(self):
        original = self.pack.wind.config.particle_transport
        reversed_hops = dict(reversed(list(original.hop_distance_by_class.items())))
        particle = dataclasses.replace(original, hop_distance_by_class=reversed_hops)
        config = dataclasses.replace(self.pack.wind.config, particle_transport=particle)
        runner = fixture.make_runner(self.bed, self.pack.schedule, config=config)
        pack = codec.capture_original_inputs(runner, source_identity=SOURCES)
        path = self.root / "ordered"
        identity = codec.save_original_inputs(path, pack)
        restored = codec.load_original_inputs(
            path,
            expected_manifest_sha256=identity,
            expected_source_identity=dict(reversed(list(SOURCES.items()))),
        )
        self.assertEqual(exact(restored), exact(pack))
        self.assertEqual(
            list(restored.wind.config.particle_transport.hop_distance_by_class),
            list(reversed_hops),
        )

    def test_63_external_source_mapping_exact_types_precede_open_or_comparison(self):
        class PoisonDict(dict):
            def __eq__(self, other):
                raise AssertionError("foreign dictionary comparison")

            def __ne__(self, other):
                raise AssertionError("foreign dictionary comparison")

        class PoisonString(str):
            def __eq__(self, other):
                raise AssertionError("foreign string comparison")

            def __ne__(self, other):
                raise AssertionError("foreign string comparison")

            __hash__ = str.__hash__

        class EqualAnything:
            def __eq__(self, other):
                raise AssertionError("foreign identity comparison")

        cases = (
            PoisonDict(SOURCES),
            {**SOURCES, "masy": PoisonString(SOURCES["masy"])},
            {**SOURCES, "masy": EqualAnything()},
            {k: v for k, v in SOURCES.items() if k != "maple"},
            {**SOURCES, "extra": "unsupported"},
            {PoisonString(k): v for k, v in SOURCES.items()},
        )
        for index, sources in enumerate(cases):
            with (
                self.subTest(case=index),
                mock.patch.object(
                    codec, "_open_dir", side_effect=AssertionError("validate before IO")
                ),
                self.assertRaises(codec.OriginalInputsError) as error,
            ):
                codec.load_original_inputs(
                    self.root / "missing",
                    expected_manifest_sha256="0" * 64,
                    expected_source_identity=sources,
                )
            self.assertIn(error.exception.code, ("metadata_fields", "source_identity"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
