"""Focused unit engineering controls of `native_water_calendar._check_delivery_manifest` (the cold delivery-bytes check). The function reads only `fence.native_target` and `fence.delivery`, so a minimal
stand-in carrying those two members is used (engineering control of the I/O contract; the genuine native/fence positives are in test_delivery_water.py). Needs the native stack (run via the fenced runner)."""
import hashlib
import os
import signal
import stat
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from masy import native_water_calendar as nwc  # noqa: E402

BIG = 1 << 40


class _Buf(bytearray):
    """bytearray subclass that supports weak references (a var-sized bytes subclass cannot) to observe buffer lifetime."""


class DeliveryManifestUnit(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="dvm_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))
        self.path = self.dir / "bundle-manifest.json"
        self.data = os.urandom(3 * (1 << 20) + 123)  # > 3 chunks
        self.path.write_bytes(self.data)
        self.fence = self.fence_for(self.data)

    def fence_for(self, data):
        return SimpleNamespace(native_target=str(self.dir), delivery={"manifest_sha256": hashlib.sha256(data).hexdigest()})

    def code(self, fn):
        try:
            fn()
            return "ok"
        except Exception as exc:  # noqa: BLE001
            return getattr(exc, "code", type(exc).__name__)

    def test_exact_bytes_pass_and_changed_bytes_refuse(self):
        self.assertEqual(self.code(lambda: nwc._check_delivery_manifest(self.fence, 1 << 20, lambda: BIG)), "ok")
        self.assertEqual(self.code(lambda: nwc._check_delivery_manifest(self.fence_for(b"other"), 1 << 20, lambda: BIG)), "source_fence_mismatch")

    def test_no_previous_chunk_is_live_when_the_next_read_allocates(self):
        real_read, refs, live_at_next = os.read, [], []

        def spy(fd, n):
            if refs:
                live_at_next.append(refs[-1]() is not None)  # is the PREVIOUS buffer still alive at the next allocation?
            out = real_read(fd, n)
            if out:
                buf = _Buf(out)
                refs.append(weakref.ref(buf))
                return buf
            return out
        with mock.patch.object(os, "read", side_effect=spy):
            self.assertEqual(self.code(lambda: nwc._check_delivery_manifest(self.fence, 1 << 20, lambda: BIG)), "ok")
        self.assertGreaterEqual(len(refs), 4)
        self.assertEqual(live_at_next, [False] * len(live_at_next))  # one chunk at a time

    def test_low_or_invalid_availability_opens_nothing(self):
        opened = []
        real_open = os.open
        with mock.patch.object(os, "open", side_effect=lambda p, *a, **k: (opened.append(str(p)), real_open(p, *a, **k))[1]):
            self.assertEqual(self.code(lambda: nwc._check_delivery_manifest(self.fence, 1 << 20, lambda: (1 << 20) + (1 << 20) - 1)), "memory_refused")
            self.assertEqual(self.code(lambda: nwc._check_delivery_manifest(self.fence, 0, lambda: 0)), "memory_refused")
            self.assertEqual(self.code(lambda: nwc._check_delivery_manifest(self.fence, 0, lambda: True)), "invalid_availability")
            self.assertEqual(self.code(lambda: nwc._check_delivery_manifest(self.fence, 0, lambda: -1)), "invalid_availability")
        self.assertEqual([p for p in opened if "bundle-manifest.json" in p], [])
        self.assertEqual(self.code(lambda: nwc._check_delivery_manifest(self.fence, 0, lambda: 1 << 20)), "ok")  # exactly reserve + one chunk is enough

    def test_special_and_non_regular_manifests_are_refused_without_opening_or_blocking(self):
        opened = []
        real_open = os.open
        signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(AssertionError("blocked")))
        for kind in ("fifo", "symlink", "directory"):
            if os.path.lexists(self.path):
                self.path.unlink() if not self.path.is_dir() or self.path.is_symlink() else self.path.rmdir()
            if kind == "fifo":
                os.mkfifo(self.path)
            elif kind == "symlink":
                other = self.dir / "real.json"
                other.write_bytes(self.data)
                os.symlink(other, self.path)
            else:
                os.mkdir(self.path)
            signal.alarm(5)
            try:
                with mock.patch.object(os, "open", side_effect=lambda p, *a, **k: (opened.append(str(p)), real_open(p, *a, **k))[1]):
                    self.assertEqual(self.code(lambda: nwc._check_delivery_manifest(self.fence, 0, lambda: BIG)), "source_fence_mismatch", kind)
            finally:
                signal.alarm(0)
            self.assertEqual([p for p in opened if "bundle-manifest.json" in p], [], kind)  # refused from lstat alone: never opened

    def test_file_replaced_between_examination_and_open_is_refused(self):
        real_open = os.open

        def swap(p, *a, **k):
            if str(p).endswith("bundle-manifest.json"):
                os.unlink(p)
                Path(p).write_bytes(self.data + b"x")  # same name, different size/inode
            return real_open(p, *a, **k)
        with mock.patch.object(os, "open", side_effect=swap):
            self.assertEqual(self.code(lambda: nwc._check_delivery_manifest(self.fence, 0, lambda: BIG)), "source_fence_mismatch")

    def test_oversized_manifest_refused_from_metadata(self):
        with mock.patch.object(os, "lstat", return_value=SimpleNamespace(st_mode=stat.S_IFREG | 0o444, st_size=(16 << 20) + 1, st_dev=1, st_ino=1)):
            with mock.patch.object(os, "open", side_effect=AssertionError("opened")):
                self.assertEqual(self.code(lambda: nwc._check_delivery_manifest(self.fence, 0, lambda: BIG)), "source_fence_mismatch")


if __name__ == "__main__":
    unittest.main()
