"""Module origins and distribution metadata (synthetic packages only)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from _support import TempDirCase, write

from masy import importcheck
from masy.importcheck import OriginError, distribution_records, import_upstreams, module_origin_violations


def module(name: str, file: str | None, path: list[str] | None = None) -> types.ModuleType:
    mod = types.ModuleType(name)
    if file is not None:
        mod.__file__ = file
    if path is not None:
        mod.__path__ = path
    return mod


class OriginTests(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        self.pkg = str(self.tmp / "MAPLE" / "src" / "maple")
        write(Path(self.pkg) / "__init__.py", "")
        write(Path(self.pkg) / "core.py", "")
        self.expected = {"maple": self.pkg}

    def test_correct_origins_pass(self) -> None:
        modules = {
            "maple": module("maple", f"{self.pkg}/__init__.py", [self.pkg]),
            "maple.core": module("maple.core", f"{self.pkg}/core.py"),
            "maple_syrup": module("maple_syrup", "/elsewhere/maple_syrup/__init__.py", ["/elsewhere"]),
        }
        self.assertEqual(module_origin_violations(modules, self.expected), [])

    def test_shadow_package_refuses(self) -> None:
        other = str(self.tmp / "shadow" / "maple")
        modules = {"maple": module("maple", f"{other}/__init__.py", [other])}
        self.assertEqual(len(module_origin_violations(modules, self.expected)), 1)

    def test_sibling_prefix_directory_is_not_inside(self) -> None:
        sibling = str(self.tmp / "MAPLE" / "src" / "maple_evil" / "x.py")
        modules = {"maple.x": module("maple.x", sibling)}
        self.assertEqual(module_origin_violations(modules, self.expected)[0]["reason"],
                         "module file outside the observed package directory")

    def test_loose_ancestor_is_not_accepted(self) -> None:
        ancestor = str(self.tmp / "MAPLE" / "src" / "x.py")
        self.assertEqual(len(module_origin_violations({"maple.x": module("maple.x", ancestor)}, self.expected)), 1)

    def test_extra_package_path_entry_refuses(self) -> None:
        modules = {"maple": module("maple", f"{self.pkg}/__init__.py", [self.pkg, str(self.tmp)])}
        self.assertEqual(len(module_origin_violations(modules, self.expected)), 1)

    def test_namespace_or_synthetic_module_refuses(self) -> None:
        self.assertEqual(len(module_origin_violations({"maple.ns": module("maple.ns", None)}, self.expected)), 1)

    def test_symlinked_module_file_resolving_outside_refuses(self) -> None:
        outside = write(self.tmp / "outside.py", "")
        os.symlink(outside, Path(self.pkg) / "linked.py")
        modules = {"maple.linked": module("maple.linked", f"{self.pkg}/linked.py")}
        self.assertEqual(len(module_origin_violations(modules, self.expected)), 1)


class ImportUpstreamsTests(TempDirCase):
    PACKAGES = ("masyfakeup",)

    def setUp(self) -> None:
        super().setUp()
        self._path = list(sys.path)
        self._modules = set(sys.modules)

    def tearDown(self) -> None:
        sys.path[:] = self._path
        for name in set(sys.modules) - self._modules:
            del sys.modules[name]
        super().tearDown()

    def test_wrong_origin_from_import_path_refuses(self) -> None:
        selected = self.tmp / "selected" / "src" / "masyfakeup"
        write(selected / "__init__.py", "")
        shadow_src = self.tmp / "shadow" / "src"
        write(shadow_src / "masyfakeup" / "__init__.py", "")
        with self.assertRaises(OriginError) as ctx:
            import_upstreams([str(shadow_src)], {"masyfakeup": str(selected)})
        self.assertEqual(ctx.exception.code, "module_origin_mismatch")

    def test_selected_origin_passes(self) -> None:
        src = self.tmp / "selected" / "src"
        write(src / "masyfakeup" / "__init__.py", "")
        record = import_upstreams([str(src)], {"masyfakeup": str(src / "masyfakeup")})
        self.assertEqual(record["sys_path_inserted"], [str(src)])
        self.assertEqual(record["packages"]["masyfakeup"]["file"], str(src / "masyfakeup" / "__init__.py"))

    def test_preimported_upstream_refuses(self) -> None:
        sys.modules["masyfakeup"] = types.ModuleType("masyfakeup")
        with self.assertRaises(OriginError) as ctx:
            import_upstreams([str(self.tmp)], {"masyfakeup": str(self.tmp / "masyfakeup")})
        self.assertEqual(ctx.exception.code, "upstream_preimported")

    def test_import_failure_refuses(self) -> None:
        src = self.tmp / "src"
        write(src / "masyfakeup" / "__init__.py", "raise RuntimeError('broken upstream')\n")
        with self.assertRaises(OriginError) as ctx:
            import_upstreams([str(src)], {"masyfakeup": str(src / "masyfakeup")})
        self.assertEqual(ctx.exception.code, "upstream_import_failed")


class BoundedHashTests(TempDirCase):
    def test_unchanged_file_is_hashed(self) -> None:
        target = self.tmp / "core.so"
        target.write_bytes(b"12345678")
        record = importcheck._bounded_sha256(str(target))
        self.assertEqual((record["size"], record["sha256"]), (8, hashlib.sha256(b"12345678").hexdigest()))

    def test_file_over_bound_is_not_hashed(self) -> None:
        target = self.tmp / "core.so"
        target.write_bytes(b"x" * 32)
        with mock.patch.object(importcheck, "MAX_HASH_BYTES", 16):
            record = importcheck._bounded_sha256(str(target))
        self.assertIsNone(record["sha256"])
        self.assertEqual(record["reason"], "larger than hash bound")

    def test_growth_after_stat_reports_no_digest(self) -> None:
        target = self.tmp / "core.so"
        target.write_bytes(b"x" * 8)
        real_stat = os.stat
        grown = []

        def growing_stat(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            if not grown and os.fspath(path) == str(target):
                grown.append(True)
                with open(target, "ab") as handle:
                    handle.write(b"y" * 64)
            return result

        with mock.patch.object(importcheck, "MAX_HASH_BYTES", 16), mock.patch("os.stat", growing_stat):
            record = importcheck._bounded_sha256(str(target))
        self.assertEqual(target.stat().st_size, 72)
        self.assertIsNone(record["sha256"])
        self.assertIn("changed", record["reason"])

    def test_growth_after_open_reports_no_digest(self) -> None:
        target = self.tmp / "core.so"
        target.write_bytes(b"x" * 8)
        real_fstat = os.fstat
        grown = []

        def growing_fstat(fd):
            result = real_fstat(fd)
            if not grown:
                grown.append(True)
                with open(target, "ab") as handle:
                    handle.write(b"y" * 64)
            return result

        with mock.patch.object(importcheck, "MAX_HASH_BYTES", 16), mock.patch("os.fstat", growing_fstat):
            record = importcheck._bounded_sha256(str(target))
        self.assertIsNone(record["sha256"])
        self.assertIn("changed during read", record["reason"])

    def test_non_regular_file_is_not_hashed(self) -> None:
        self.assertIsNone(importcheck._bounded_sha256(str(self.tmp))["sha256"])


class DistributionMetadataTests(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        self._path = list(sys.path)

    def tearDown(self) -> None:
        sys.path[:] = self._path
        super().tearDown()

    def install_metadata(self, name: str, direct_url: dict | None) -> None:
        site = self.tmp / "site"
        info = site / f"{name}-0.dist-info"
        write(info / "METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: 0\n")
        if direct_url is not None:
            write(info / "direct_url.json", json.dumps(direct_url))
        sys.path.insert(0, str(site))

    def test_absent_metadata_is_info(self) -> None:
        _, findings = distribution_records({"masy-test-absent-dist": None})
        self.assertEqual([(f["code"], f["severity"]) for f in findings], [("distribution_metadata_absent", "info")])

    def test_editable_metadata_for_selected_root_matches(self) -> None:
        root = str(self.tmp / "root")
        self.install_metadata("masytestdist", {"url": Path(root).as_uri(), "dir_info": {"editable": True}})
        records, findings = distribution_records({"masytestdist": root})
        self.assertEqual(findings, [])
        self.assertEqual(records["masytestdist"]["distributions"][0]["direct_url"]["dir_info"], {"editable": True})

    def test_metadata_for_other_root_is_reported_not_repaired(self) -> None:
        self.install_metadata("masytestdist", {"url": "file:///somewhere/else", "dir_info": {"editable": True}})
        metadata = self.tmp / "site" / "masytestdist-0.dist-info" / "direct_url.json"
        before = metadata.read_bytes()
        _, findings = distribution_records({"masytestdist": str(self.tmp / "root")})
        self.assertEqual([f["code"] for f in findings], ["distribution_metadata_unverified"])
        self.assertEqual(metadata.read_bytes(), before)

    def test_non_editable_metadata_is_unverified(self) -> None:
        self.install_metadata("masytestdist", None)
        _, findings = distribution_records({"masytestdist": str(self.tmp / "root")})
        self.assertEqual([f["severity"] for f in findings], ["blocking"])


if __name__ == "__main__":
    unittest.main()
