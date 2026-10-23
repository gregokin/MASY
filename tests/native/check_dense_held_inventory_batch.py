"""Genuine small filesystem checks for the quiescent held-inventory batch.

These tests author their own tiny files. They do not load a runtime lock, run
models, import Native providers explicitly, or claim hostile-writer atomicity.
Instrumentation at race boundaries always delegates actual filesystem checks.
"""

from __future__ import annotations

import errno
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from masy import dense_boundary_store as c


class HeldInventoryBatchChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.parent = self.root / "owned"
        self.parent.mkdir()
        self.first = self.parent / "alpha"
        self.second = self.parent / "beta"
        self.write(self.first, b"one")
        self.write(self.second, b"two")
        self.inventories = []

    def tearDown(self):
        for held in self.inventories:
            held.close()
        self.temporary.cleanup()

    @staticmethod
    def write(path, raw):
        path.write_bytes(raw)
        path.chmod(0o444)

    def inventory(self, paths=None):
        paths = (self.first, self.second) if paths is None else paths
        held = c._HeldInventory(32, max_parents=4, max_files=8)
        self.inventories.append(held)
        for path in paths:
            held.admit(str(path), path.stat().st_size)
        return held

    def refusal(self, code, call):
        with self.assertRaises(c.BoundaryStoreError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def replace_file(self, path, raw=b"new"):
        replacement = self.parent / "replacement"
        self.write(replacement, raw)
        os.replace(replacement, path)

    def replace_parent(self):
        retained = self.root / "retained"
        self.parent.rename(retained)
        self.parent.mkdir()
        self.write(self.first, b"one")
        self.write(self.second, b"two")
        return retained

    def test_genuine_held_descriptors_and_two_parent_brackets(self):
        other = self.root / "other"
        other.mkdir()
        third = other / "gamma"
        self.write(third, b"six")
        held = self.inventory((self.first, self.second, third))
        parent_check, lstat = held._check_parent, held._lstat
        observed = []

        def parent(path):
            descriptor = parent_check(path)
            self.assertEqual(c._identity(os.fstat(descriptor)), held._parents[path][1])
            observed.append(("parent", path))
            return descriptor

        def file(parent_path, name, descriptor):
            self.assertEqual(descriptor, held._parents[parent_path][0])
            info = lstat(parent_path, name, descriptor)
            self.assertEqual(info.st_ino, os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_ino)
            observed.append(("file", str(Path(parent_path) / name)))
            return info

        with mock.patch.object(held, "_check_parent", side_effect=parent), \
             mock.patch.object(held, "_lstat", side_effect=file):
            held.check()
        parents = [("parent", str(p)) for p in (self.parent, other)]
        files = [("file", str(p)) for p in (self.first, self.second, third)]
        self.assertEqual(observed, parents + files + parents)
        self.assertEqual(held._bytes, 9)

    def test_repeated_batch_rechecks_actual_file_without_result_cache(self):
        held = self.inventory()
        held.check()
        self.replace_file(self.second)
        self.refusal("identity_drift", held.check)

    def test_empty_batch_and_closed_batch(self):
        held = self.inventory(())
        held.check()
        held.close()
        self.refusal("inventory_closed", held.check)

    def test_same_size_inode_replacement_refuses(self):
        held = self.inventory()
        before = self.second.stat()
        self.replace_file(self.second, b"two")
        self.assertNotEqual(before.st_ino, self.second.stat().st_ino)
        self.refusal("identity_drift", held.check)

    def test_in_place_size_change_refuses(self):
        held = self.inventory()
        self.second.chmod(0o644)
        self.second.write_bytes(b"longer")
        self.second.chmod(0o444)
        self.refusal("identity_drift", held.check)

    def test_actual_timestamp_change_refuses(self):
        held = self.inventory()
        before = self.second.stat()
        os.utime(self.second, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        self.assertNotEqual(before.st_mtime_ns, self.second.stat().st_mtime_ns)
        self.refusal("identity_drift", held.check)

    def test_readonly_mode_change_refuses(self):
        held = self.inventory()
        self.second.chmod(0o400)
        self.refusal("identity_drift", held.check)

    def test_writable_mode_change_refuses(self):
        held = self.inventory()
        self.second.chmod(0o644)
        self.refusal("identity_drift", held.check)

    def test_added_hard_link_refuses(self):
        held = self.inventory()
        os.link(self.second, self.parent / "extra-link")
        self.assertEqual(self.second.stat().st_nlink, 2)
        self.refusal("identity_drift", held.check)

    def test_removed_file_refuses(self):
        held = self.inventory()
        self.second.unlink()
        self.refusal("inventory_file", held.check)

    def test_file_replaced_by_symlink_is_not_followed(self):
        held = self.inventory()
        self.second.unlink()
        self.second.symlink_to(self.first)
        self.assertTrue(self.second.is_symlink())
        self.refusal("identity_drift", held.check)

    def test_file_replaced_by_directory_refuses(self):
        held = self.inventory()
        self.second.unlink()
        self.second.mkdir()
        self.refusal("identity_drift", held.check)

    def test_persistent_parent_replacement_refuses_with_real_held_fd(self):
        held = self.inventory()
        descriptor, identity = held._parents[str(self.parent)]
        retained = self.replace_parent()
        self.assertEqual(c._identity(os.fstat(descriptor)), identity)
        self.assertEqual(os.stat("alpha", dir_fd=descriptor, follow_symlinks=False).st_ino,
                         (retained / "alpha").stat().st_ino)
        self.refusal("identity_drift", held.check)

    def test_parent_symlink_refuses_without_following_it(self):
        held = self.inventory()
        retained = self.root / "retained"
        self.parent.rename(retained)
        self.parent.symlink_to(retained, target_is_directory=True)
        with self.assertRaises(OSError):
            held.check()

    def test_actual_file_drift_after_prepass_refuses(self):
        held = self.inventory()
        parent_check = held._check_parent
        calls = 0

        def check(parent):
            nonlocal calls
            descriptor = parent_check(parent)
            calls += 1
            if calls == 1:
                self.replace_file(self.first)
            return descriptor

        with mock.patch.object(held, "_check_parent", side_effect=check):
            self.refusal("identity_drift", held.check)
        self.assertEqual(calls, 1)

    def test_mutation_between_file_checks_refuses(self):
        held = self.inventory()
        lstat = held._lstat
        observed = []

        def check(parent, name, descriptor):
            info = lstat(parent, name, descriptor)
            observed.append(name)
            if name == "alpha":
                self.replace_file(self.second)
            return info

        with mock.patch.object(held, "_lstat", side_effect=check):
            self.refusal("identity_drift", held.check)
        self.assertEqual(observed, ["alpha", "beta"])

    def test_parent_replaced_after_prepass_uses_held_fd_then_refuses(self):
        held = self.inventory()
        parent_check, lstat = held._check_parent, held._lstat
        calls, files = 0, []

        def parent(path):
            nonlocal calls
            descriptor = parent_check(path)
            calls += 1
            if calls == 1:
                self.replace_parent()
            return descriptor

        def file(parent_path, name, descriptor):
            info = lstat(parent_path, name, descriptor)
            self.assertEqual(info.st_ino, (self.root / "retained" / name).stat().st_ino)
            files.append(name)
            return info

        with mock.patch.object(held, "_check_parent", side_effect=parent), \
             mock.patch.object(held, "_lstat", side_effect=file):
            self.refusal("identity_drift", held.check)
        self.assertEqual(files, ["alpha", "beta"])
        self.assertEqual(calls, 1)  # the failing postpass does not return

    def test_parent_drift_after_final_file_is_caught_by_endpass(self):
        held = self.inventory()
        lstat = held._lstat
        files = []

        def file(parent_path, name, descriptor):
            info = lstat(parent_path, name, descriptor)
            files.append(name)
            if name == "beta":
                self.replace_parent()
            return info

        with mock.patch.object(held, "_lstat", side_effect=file):
            self.refusal("identity_drift", held.check)
        self.assertEqual(files, ["alpha", "beta"])

    def test_refusal_keeps_fd_owned_until_explicit_close(self):
        held = self.inventory()
        descriptor, identity = held._parents[str(self.parent)]
        self.second.unlink()
        self.refusal("inventory_file", held.check)
        self.assertEqual(c._identity(os.fstat(descriptor)), identity)
        held.close()
        with self.assertRaises(OSError) as caught:
            os.fstat(descriptor)
        self.assertEqual(caught.exception.errno, errno.EBADF)
        self.assertFalse(held._parents)
        self.assertFalse(held._files)
        self.refusal("inventory_closed", held.check)

    def test_standalone_read_rechecks_file_identity(self):
        held = self.inventory()
        self.assertEqual(held.read(str(self.first), 3), b"one")
        self.replace_file(self.first)
        with mock.patch.object(held, "_check_held_file", side_effect=AssertionError("batch-only helper")):
            self.refusal("identity_drift", lambda: held.read(str(self.first), 3))

    def test_readmission_rechecks_existing_file_identity(self):
        held = self.inventory()
        self.replace_file(self.second)
        with mock.patch.object(held, "_check_held_file", side_effect=AssertionError("batch-only helper")):
            self.refusal("identity_drift", lambda: held.admit(str(self.second), 3))


if __name__ == "__main__":
    unittest.main()
