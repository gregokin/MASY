"""Task-111 namespace-local mount ID checks.

Exercises the actual DenseBoundaryStore._storage, _record, hashing and fstat on a
tiny temporary root with an authored probe receipt. Only _mount_observation is
replaced. No model, bed, fixture, native runtime, store or genesis is created.
"""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from masy import dense_boundary_store as s
from masy import dense_original_inputs as oi

SAVED = {
    "mount_id": 654,
    "mount_path": "/home/okin/MASY",
    "major_minor": "8:48",
    "fs_type": "ext4",
}


def _mount(**change):
    value = dict(SAVED)
    value.update(change)
    return value


def _without(name):
    value = dict(SAVED)
    del value[name]
    return value


class StorageIdentity(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(os.path.realpath(temporary.name))
        self.rootfd = os.open(
            self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        self.addCleanup(os.close, self.rootfd)
        os.mkdir(self.root / ".storage_probe", 0o700)

    def receipt(self, **change):
        info = os.fstat(self.rootfd)
        value = {
            "schema": s.SCHEMA + ".probe",
            "observation": dict(SAVED),
            "f_fsid": os.fstatvfs(self.rootfd).f_fsid,
            "root_dev": info.st_dev,
            "root_ino": info.st_ino,
            "root_path": str(self.root),
            "primitives": ["exclusive_mkdir", "exclusive_file", "hardlink"],
            "scope": "authored task-111 unit receipt",
        }
        value.update(change)
        return value

    def storage(self, current, receipt=None, **contract):
        raw = oi._json(self.receipt() if receipt is None else receipt)
        path = self.root / ".storage_probe" / "result.json"
        if path.exists():
            path.unlink()
        path.write_bytes(raw)
        info = os.fstat(self.rootfd)
        record = {
            "schema": s.SCHEMA + ".storage",
            "operator_declares_local": True,
            "operator_declares_fsync_durable": True,
            "observed_fs_type": "ext4",
            "observed_fs_id": os.fstatvfs(self.rootfd).f_fsid,
            "root_path": str(self.root),
            "root_dev": info.st_dev,
            "root_ino": info.st_ino,
            "probe_file_sha256": s._sha(raw),
        }
        record.update(contract)
        store = s.DenseBoundaryStore.__new__(s.DenseBoundaryStore)
        store._root, store._rootfd = self.root, self.rootfd
        store._header = {"storage_contract": record}

        def observe(root, device):
            self.assertEqual((root, device), (self.root, info.st_dev))
            return copy.deepcopy(current)

        try:
            with mock.patch.object(s, "_mount_observation", observe):
                store._storage()
        finally:
            self.assertEqual(os.listdir(self.root), [".storage_probe"])
            self.assertEqual(os.listdir(self.root / ".storage_probe"), ["result.json"])

    def refuses(self, current, receipt=None, **contract):
        with self.assertRaises(s.BoundaryStoreError):
            self.storage(current, receipt, **contract)

    def test_unchanged_mount_id_accepted(self):
        self.storage(dict(SAVED))

    def test_fresh_namespace_mount_id_accepted(self):
        self.storage(_mount(mount_id=585))
        self.storage(_mount(mount_id=1), self.receipt(observation=_mount(mount_id=7)))

    def test_durable_field_mismatch_refused_with_equal_id(self):
        for name, value in (
            ("mount_path", "/home/okin"),
            ("major_minor", "8:49"),
            ("fs_type", "xfs"),
        ):
            with self.subTest(name=name):
                self.refuses(_mount(**{name: value}))
                self.refuses(dict(SAVED), self.receipt(observation=_mount(**{name: value})))

    def test_malformed_observations_refused(self):
        bad = [
            _without("mount_id"),
            _without("fs_type"),
            _mount(extra=1),
            _mount(mount_id=True),
            _mount(mount_id=0),
            _mount(mount_id=-585),
            _mount(mount_id="585"),
            _mount(mount_id=None),
            _mount(mount_path=None),
            _mount(major_minor=848),
            _mount(fs_type=["ext4"]),
            None,
            [],
        ]
        for index, value in enumerate(bad):
            with self.subTest(saved=index):
                self.refuses(_mount(mount_id=585), self.receipt(observation=value))
            with self.subTest(current=index):
                self.refuses(value)
        for value in (585.0, 1 << 63):
            with self.subTest(current=value):
                self.refuses(_mount(mount_id=value))
        with self.subTest(saved="absent observation"):
            receipt = self.receipt()
            del receipt["observation"]
            self.refuses(_mount(mount_id=585), receipt)

    def test_probe_raw_hash_refused(self):
        self.refuses(_mount(mount_id=585), probe_file_sha256="0" * 64)

    def test_root_and_filesystem_guards_retained(self):
        info = os.fstat(self.rootfd)
        fsid = os.fstatvfs(self.rootfd).f_fsid
        for name, value in (
            ("root_path", str(self.root) + "-other"),
            ("root_dev", info.st_dev + 1),
            ("root_ino", info.st_ino + 1),
            ("observed_fs_type", "xfs"),
            ("observed_fs_id", fsid + 1),
        ):
            with self.subTest(contract=name):
                self.refuses(_mount(mount_id=585), **{name: value})
        for name, value in (
            ("f_fsid", fsid + 1),
            ("root_dev", info.st_dev + 1),
            ("root_ino", info.st_ino + 1),
            ("root_path", str(self.root) + "-other"),
        ):
            with self.subTest(probe=name):
                self.refuses(_mount(mount_id=585), self.receipt(**{name: value}))
        with self.subTest(unsupported="btrfs"):
            receipt = self.receipt(observation=_mount(fs_type="btrfs"))
            self.refuses(
                _mount(mount_id=585, fs_type="btrfs"), receipt, observed_fs_type="btrfs"
            )

    def test_parser_ids_differ_only_by_namespace(self):
        line = b" 1 8:48 / /home/okin/MASY rw,relatime shared:1 - ext4 /dev/sdd rw\n"
        device = os.makedev(8, 48)
        root = Path("/home/okin/MASY/store")
        old = s._mount_observation(root, device, raw=b"654" + line)
        new = s._mount_observation(root, device, raw=b"585" + line)
        self.assertEqual((old["mount_id"], new["mount_id"]), (654, 585))
        self.assertEqual(old, SAVED)
        self.assertEqual(
            s._durable_mount(old, "old"), s._durable_mount(new, "new")
        )
        self.assertEqual(s._durable_mount(new, "new"), ("/home/okin/MASY", "8:48", "ext4"))


if __name__ == "__main__":
    unittest.main()
