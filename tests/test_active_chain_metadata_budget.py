"""Synthetic stat/clock-format fixtures, never Native capsules or store authority."""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import _support  # noqa: F401 - selects this checkout's MASY source

from masy import dense_active_water_store as active
from masy.clock import encode_time


class ActiveChainMetadataBudget(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        outer = Path(self.temp.name)
        root = outer / "store"
        root.mkdir()
        (root / "generations").mkdir()
        self.root = root
        self.inventory = {"water_attempts": set()}

    def handle(self, combined=True):
        store_id = "0" * 32
        managed = self.root.parent / ((".masy-active-" if combined else ".masy-water-") + store_id)
        managed.mkdir(exist_ok=True)
        sibling, parent = managed.stat(), managed.parent.stat()
        fs_id = os.statvfs(managed).f_fsid
        header = {
            "schema": active.SCHEMA_PROCESS if combined else active.SCHEMA,
            "store_id": store_id,
            "storage_contract": {"observed_fs_id": fs_id},
            "managed_active" if combined else "managed_water": {
                "path": str(managed), "dev": sibling.st_dev, "ino": sibling.st_ino,
                "parent_path": str(managed.parent), "parent_dev": parent.st_dev,
                "parent_ino": parent.st_ino, "fs_id": fs_id,
            },
        }
        return SimpleNamespace(_root=self.root, _header=header, _limits=active.c._DEFAULT,
                               _root_identity=(self.root.stat().st_dev, self.root.stat().st_ino))

    def capsule(self, handle, size, export_size=0):
        name = "synthetic-stat-" + str(len(self.inventory["water_attempts"]))
        path = active._managed(handle) / name
        (path / "entry").mkdir(parents=True)
        # Sparse raw extents intentionally are NOT parsed or used as capsules.
        with (path / "entry/capsule.json").open("wb") as stream:
            stream.truncate(size)
        if export_size:
            with (path / "export.json").open("wb") as stream:
                stream.truncate(export_size)
        self.inventory["water_attempts"].add(name)
        return path / "entry/capsule.json"

    def charge(self, handle):
        return active._metadata_stat_charge(handle, self.inventory)

    def test_empty_has_no_prospective_constructor(self):
        self.assertEqual(self.charge(self.handle()), 0)

    def test_more_smaller_capsules_keep_current_occurrences_without_extra_future(self):
        handle = self.handle()
        self.capsule(handle, 1000)
        first = self.charge(handle)
        self.capsule(handle, 200)
        self.capsule(handle, 200)
        # Both identical CURRENT graphs stay charged; future selection is still
        # the original larger candidate, not another capture per old child.
        self.assertEqual(self.charge(handle) - first, 6 * active._parse_charge(200))

    def test_new_largest_capsule_expands_future_once(self):
        handle = self.handle()
        self.capsule(handle, 7)
        first = self.charge(handle)
        self.capsule(handle, 91)
        self.assertEqual(self.charge(handle) - first,
                         3 * active._parse_charge(91) + 6 * active._parse_charge(91 - 7))

    def test_histories_descriptors_and_repeated_image_receivers_remain_additive(self):
        handle = self.handle()
        generation = self.root / "generations" / ("g-0000-" + "1" * 32)
        (generation / "image").mkdir(parents=True)
        (generation / "intent.json").write_bytes(b"x" * 11)
        (generation / "generation.json").write_bytes(b"x" * 17)
        (generation / "image/manifest.json").write_bytes(b"x" * 19)
        (self.root / "generations" / ("g-0001-" + "2" * 32)).mkdir()
        self.capsule(handle, 23, export_size=29)
        # Two B receiver occurrences pay the largest image separately even
        # though only one such file is physically present in this stat fixture.
        current = sum(active._parse_charge(n) for n in (11, 17, 19, 19, 23, 29))
        self.assertEqual(self.charge(handle), 3 * current + 6 * active._parse_charge(23 + 2933))

    def test_legacy_water_reservation_is_unchanged(self):
        handle = self.handle(combined=False)
        sizes = (1, 17, 11, 1 << 20)
        for size in sizes:
            self.capsule(handle, size)
        expected = sum(6 * active._parse_charge(min(1 << 20, n + 3)) for n in sizes)
        self.assertEqual(self.charge(handle), expected)

    def test_wire_cap_exact_and_future_constructor_not_clamped(self):
        handle = self.handle()
        path = self.capsule(handle, 1 << 20)
        expected = (3 * active._parse_charge(1 << 20)
                    + 6 * active._parse_charge((1 << 20) + 2933))
        self.assertEqual(self.charge(handle), expected)
        with path.open("r+b") as stream:
            stream.truncate((1 << 20) + 1)
        with self.assertRaises(active.c.BoundaryStoreError) as caught:
            self.charge(handle)
        self.assertEqual(caught.exception.code, "active_memory")

    def test_nonregular_capsule_still_refuses_before_parse(self):
        handle = self.handle()
        path = self.capsule(handle, 3)
        target = path.with_name("synthetic-target")
        path.rename(target)
        path.symlink_to(target.name)
        with self.assertRaises(active.c.BoundaryStoreError) as caught:
            self.charge(handle)
        self.assertEqual(caught.exception.code, "active_memory")

    def test_growth_bound_covers_independent_binary64_fraction_formats(self):
        # General signed differences challenge the 2099-bit proof; native_time
        # itself remains nonnegative, as required by the actual wind clock.
        values = (0.0, math.ulp(0.0), -math.ulp(0.0), sys.float_info.min,
                  1.0, -1.0, sys.float_info.max, -sys.float_info.max)
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode("ascii")
        rational = lambda value: f"{value.numerator}/{value.denominator}"
        original = {"ordinal": 1, "pauses": 1, "progress": {
            "native_time": encode_time(0.0), "elapsed_local_s": "0/1",
            "global_elapsed_s": "0/1", "next_step_index": 0,
            "advanced_hex": [0.0.hex(), 1.0.hex(), 2.0.hex()], "n_steps": 4096}}
        for left, right in itertools.product(values, repeat=2):
            difference = Fraction(left) - Fraction(right)
            self.assertLessEqual(len(rational(difference)), 958)
            updated = {"ordinal": 4096, "pauses": 2, "progress": {
                **original["progress"], "native_time": encode_time(abs(left)),
                "elapsed_local_s": rational(difference),
                "global_elapsed_s": rational(-difference), "next_step_index": 4096}}
            with self.subTest(left=left.hex(), right=right.hex()):
                self.assertLessEqual(len(canonical(updated)) - len(canonical(original)),
                                     active._COMBINED_CAPSULE_GROWTH_BYTES)
        self.assertEqual(active._COMBINED_CAPSULE_GROWTH_BYTES, 2933)


if __name__ == "__main__":
    unittest.main()
