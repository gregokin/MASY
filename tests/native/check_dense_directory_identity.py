"""Source-defined filesystem qualification; mocked dispatch and genuine cases are separate.

Root execution is separately gated. These fixtures contain no model data, no
Native transport and no successful syscall substitute in genuine cases.
"""

import errno
import os
import threading
import unittest
import ctypes
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from masy import dense_boundary_store as c


class _Stop(BaseException):
    """Private non-Exception signal used only by mocked dispatch tests."""


FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC


def _fake_os(**overrides):
    ns = SimpleNamespace(
        O_RDONLY=os.O_RDONLY,
        O_DIRECTORY=os.O_DIRECTORY,
        O_CLOEXEC=os.O_CLOEXEC,
        fsencode=os.fsencode,
        open=mock.Mock(side_effect=AssertionError("unexpected os.open")),
        close=mock.Mock(),
        fstat=mock.Mock(side_effect=AssertionError("unexpected os.fstat")),
    )
    ns.__dict__.update(overrides)
    return ns


def _fake_ctypes(cdll, handle=0x5150):
    return SimpleNamespace(
        CDLL=cdll,
        pythonapi=SimpleNamespace(_handle=handle),
        c_long=ctypes.c_long,
        c_char_p=ctypes.c_char_p,
        c_void_p=ctypes.c_void_p,
        c_size_t=ctypes.c_size_t,
        byref=ctypes.byref,
    )


class MockedDirectoryDispatch(unittest.TestCase):
    """Mocked metadata/dispatch checks only; these prove no genuine syscall."""

    def setUp(self):
        self.walker = self.enterContext(
            mock.patch.object(c.oi, "_open_dir", return_value="FALLBACK")
        )
        self.profile = self.enterContext(
            mock.patch.object(c, "_abi_profile", return_value=True)
        )

    def _arm(self, fn, opened=(7,), close=None, fstat=None):
        self.os = _fake_os(open=mock.Mock(side_effect=list(opened)))
        if close is not None:
            self.os.close = close
        if fstat is not None:
            self.os.fstat = fstat
        self.enterContext(mock.patch.object(c, "os", self.os))
        self.enterContext(mock.patch.object(c, "_ABI", (object(), object(), fn)))
        return self.os

    def test_profile_mismatch_is_evaluated_fresh_and_delegates(self):
        self.profile.return_value = False
        setup = self.enterContext(mock.patch.object(c, "_abi_setup"))
        fake = self.enterContext(mock.patch.object(c, "os", _fake_os()))
        self.assertEqual(c._open_current_directory("/a/b"), "FALLBACK")
        self.assertEqual(c._open_current_directory("/a/b"), "FALLBACK")
        self.assertEqual(self.profile.call_count, 2)
        setup.assert_not_called()
        fake.open.assert_not_called()
        self.assertEqual(self.walker.call_args_list, [mock.call(Path("/a/b"))] * 2)

    def test_refused_paths_delegate_without_setup_or_open(self):
        setup = self.enterContext(mock.patch.object(c, "_abi_setup"))
        fake = self.enterContext(mock.patch.object(c, "os", _fake_os()))
        refused = [
            "relative/x",
            "/",
            "/a/../b",
            "//a/b",
            "/a\0b",
            "/" + "a" * 4095,
        ]
        for path in refused:
            with self.subTest(path=path[:24]):
                self.walker.reset_mock()
                self.assertEqual(c._open_current_directory(path), "FALLBACK")
                self.walker.assert_called_once_with(Path(path))
        fake.fsencode = mock.Mock(
            side_effect=UnicodeEncodeError("utf-8", "x", 0, 1, "mock")
        )
        self.walker.reset_mock()
        self.assertEqual(c._open_current_directory("/a/b"), "FALLBACK")
        self.walker.assert_called_once_with(Path("/a/b"))
        setup.assert_not_called()
        fake.open.assert_not_called()

    def test_boundary_length_4095_is_not_refused(self):
        fn = mock.Mock(return_value=-1)
        fake = self._arm(fn)
        path = "/" + "a" * 4094
        self.assertEqual(c._open_current_directory(path), "FALLBACK")
        fn.assert_called_once()
        fake.close.assert_called_once_with(7)

    def test_setup_failure_retries_and_walker_error_has_no_context(self):
        self.enterContext(mock.patch.object(c, "_ABI", None))
        setup = self.enterContext(
            mock.patch.object(c, "_abi_setup", side_effect=RuntimeError("setup"))
        )
        fake = self.enterContext(mock.patch.object(c, "os", _fake_os()))
        self.walker.side_effect = OSError(errno.ENOENT, "walker")
        for _ in range(2):
            with self.assertRaises(OSError) as caught:
                c._open_current_directory("/a/b")
            self.assertEqual(caught.exception.errno, errno.ENOENT)
            self.assertIsNone(caught.exception.__context__)
            self.assertIsNone(caught.exception.__cause__)
        self.assertEqual(setup.call_count, 2)
        fake.open.assert_not_called()

    def test_setup_indexes_borrowed_handle_cdll_with_exact_abi(self):
        fn = SimpleNamespace()
        lib = mock.MagicMock()
        lib.__getitem__.return_value = fn
        cdll = mock.Mock(return_value=lib)
        fake_ct = _fake_ctypes(cdll)
        self.enterContext(mock.patch.object(c, "ctypes", fake_ct))
        self.enterContext(mock.patch.object(c, "_ABI", None))
        self.addCleanup(ctypes.set_errno, ctypes.get_errno())
        ctypes.set_errno(77)
        abi = c._abi_setup()
        self.assertEqual(ctypes.get_errno(), 77)
        cdll.assert_called_once_with(None, handle=0x5150, use_errno=False)
        lib.__getitem__.assert_called_once_with("syscall")
        self.assertIs(abi[0], fake_ct.pythonapi)
        self.assertIs(abi[1], lib)
        self.assertIs(abi[2], fn)
        self.assertEqual(
            fn.argtypes,
            (ctypes.c_long, ctypes.c_long, ctypes.c_char_p,
             ctypes.c_void_p, ctypes.c_size_t),
        )
        self.assertIs(fn.restype, ctypes.c_long)
        self.assertIs(c._ABI, abi)
        self.assertIs(c._abi_setup(), abi)
        cdll.assert_called_once()

    def test_concurrent_setup_runs_once_without_path_io_under_lock(self):
        observed = []
        fn = mock.Mock(return_value=-1)
        lib = mock.MagicMock()
        lib.__getitem__.return_value = fn

        def cdll(*args, **kwargs):
            observed.append(
                (c._ABI_LOCK.locked(), fake.open.call_count, self.walker.call_count)
            )
            return lib

        fake = _fake_os(open=mock.Mock(side_effect=[7, 9]))
        self.enterContext(mock.patch.object(c, "os", fake))
        self.enterContext(mock.patch.object(c, "ctypes", _fake_ctypes(
            mock.Mock(side_effect=cdll))))
        self.enterContext(mock.patch.object(c, "_ABI", None))
        barrier = threading.Barrier(2)
        results = []

        def run():
            barrier.wait()
            results.append(c._open_current_directory("/a/b"))

        threads = [threading.Thread(target=run) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        self.assertEqual(results, ["FALLBACK", "FALLBACK"])
        self.assertEqual(observed, [(True, 0, 0)])
        self.assertEqual(fn.call_count, 2)
        self.assertEqual(sorted(a.args[0] for a in fake.close.call_args_list), [7, 9])

    def test_negative_results_fall_back_fresh_each_time(self):
        fn = mock.Mock(return_value=-1)
        fake = self._arm(fn, opened=(7, 9))
        self.assertEqual(c._open_current_directory("/a/b"), "FALLBACK")
        self.assertEqual(c._open_current_directory("/a/b"), "FALLBACK")
        self.assertEqual(fn.call_count, 2)
        self.assertEqual(fake.open.call_count, 2)
        self.assertEqual(fake.close.call_args_list, [mock.call(7), mock.call(9)])
        self.assertEqual(self.walker.call_args_list, [mock.call(Path("/a/b"))] * 2)

    def test_distinct_reserve_and_result_closed_once_via_same_directory(self):
        fn = mock.Mock(return_value=0)
        stats = {0: SimpleNamespace(ident="A"), 5: SimpleNamespace(ident="A")}
        fake = self._arm(fn, fstat=mock.Mock(side_effect=stats.__getitem__))
        self.enterContext(mock.patch.object(c, "_identity", lambda st: st.ident))
        fail = self.enterContext(mock.patch.object(c, "_fail"))
        c._same_directory("/a/b", 5)
        fail.assert_not_called()
        fake.open.assert_called_once_with("/", FLAGS)
        self.assertEqual(fake.close.call_args_list, [mock.call(7), mock.call(0)])
        self.walker.assert_not_called()
        nr, dirfd, encoded, how_ref, size = fn.call_args.args
        self.assertEqual((nr, dirfd, encoded, size), (437, -100, b"/a/b", 24))
        how = how_ref._obj
        self.assertIsInstance(how, c._OpenHow)
        self.assertEqual((how.flags, how.mode, how.resolve), (FLAGS, 0, 0x04))

    def test_reserve_close_error_cleans_result_once_and_aggregates(self):
        def close(fd):
            if fd == 7:
                raise OSError(errno.EBADF, "reserve")
            raise _Stop("result")

        fake = self._arm(mock.Mock(return_value=11), close=mock.Mock(side_effect=close))
        with self.assertRaises(BaseExceptionGroup) as caught:
            c._open_current_directory("/a/b")
        kinds = [type(e) for e in caught.exception.exceptions]
        self.assertEqual(kinds, [OSError, _Stop])
        self.assertEqual(fake.close.call_args_list, [mock.call(7), mock.call(11)])
        self.walker.assert_not_called()

    def test_reserve_close_error_alone_propagates_after_result_close(self):
        close = mock.Mock(side_effect=[OSError(errno.EIO, "reserve"), None])
        fake = self._arm(mock.Mock(return_value=11), close=close)
        with self.assertRaises(OSError) as caught:
            c._open_current_directory("/a/b")
        self.assertEqual(caught.exception.errno, errno.EIO)
        self.assertEqual(fake.close.call_args_list, [mock.call(7), mock.call(11)])

    def test_syscall_interruption_closes_reserve_and_propagates(self):
        fake = self._arm(mock.Mock(side_effect=_Stop("interrupt")))
        with self.assertRaises(_Stop):
            c._open_current_directory("/a/b")
        fake.close.assert_called_once_with(7)
        self.walker.assert_not_called()

    def test_how_construction_failure_closes_reserve(self):
        fn = mock.Mock()
        fake = self._arm(fn)
        self.enterContext(
            mock.patch.object(c, "_OpenHow", side_effect=_Stop("how"))
        )
        with self.assertRaises(_Stop):
            c._open_current_directory("/a/b")
        fn.assert_not_called()
        fake.close.assert_called_once_with(7)

    def test_stat_failure_after_transfer_closes_result_in_finally(self):
        fstat = mock.Mock(side_effect=OSError(errno.EIO, "stat"))
        fake = self._arm(mock.Mock(return_value=0), fstat=fstat)
        fail = self.enterContext(mock.patch.object(c, "_fail"))
        with self.assertRaises(OSError) as caught:
            c._same_directory("/a/b", 5)
        self.assertEqual(caught.exception.errno, errno.EIO)
        fstat.assert_called_once_with(0)
        fail.assert_not_called()
        self.assertEqual(fake.close.call_args_list, [mock.call(7), mock.call(0)])

    def test_real_setup_failure_publishes_no_partial_tuple_and_retries(self):
        fn = mock.Mock(return_value=-1)
        lib = mock.MagicMock()
        lib.__getitem__.return_value = fn
        cdll = mock.Mock(side_effect=[AttributeError("missing symbol owner"), lib])
        fake = _fake_os(open=mock.Mock(return_value=7))
        self.enterContext(mock.patch.object(c, "ctypes", _fake_ctypes(cdll)))
        self.enterContext(mock.patch.object(c, "os", fake))
        self.enterContext(mock.patch.object(c, "_ABI", None))
        self.assertEqual(c._open_current_directory("/a/b"), "FALLBACK")
        self.assertIsNone(c._ABI)
        self.assertEqual(c._open_current_directory("/a/b"), "FALLBACK")
        self.assertEqual(cdll.call_count, 2)
        self.assertIs(c._ABI[2], fn)
        fake.close.assert_called_once_with(7)

    def test_reserve_open_failure_falls_back_outside_context_and_retries(self):
        fn = mock.Mock()
        fake = self._arm(fn)
        fake.open.side_effect = OSError(errno.EMFILE, "reserve")
        self.walker.side_effect = OSError(errno.ENOENT, "walker")
        for _ in range(2):
            with self.assertRaises(OSError) as caught:
                c._open_current_directory("/a/b")
            self.assertEqual(caught.exception.errno, errno.ENOENT)
            self.assertIsNone(caught.exception.__context__)
        self.assertEqual(fake.open.call_count, 2)
        fn.assert_not_called()
        fake.close.assert_not_called()


import errno
import fcntl
import os
import tempfile
import unittest
from pathlib import Path

from masy import dense_boundary_store as c


def _old_error(path):
    try:
        fd = c.oi._open_dir(path)
    except OSError as exc:
        return type(exc), exc.errno
    os.close(fd)
    raise AssertionError(f"original walker unexpectedly opened {path}")


def _new_error(path):
    try:
        fd = c._open_current_directory(str(path))
    except OSError as exc:
        return type(exc), exc.errno
    os.close(fd)
    raise AssertionError(f"currentness helper unexpectedly opened {path}")


class GenuineDirectoryIdentity(unittest.TestCase):
    def setUp(self):
        if not c._abi_profile():
            self.fail("supported ABI profile unmet; genuine helper path unqualified")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(os.path.realpath(self._tmp.name)) / "base"
        self.base.mkdir(mode=0o700)

    def test_execute_only_intermediate_is_fast_positive(self):
        if os.geteuid() == 0:
            self.fail("non-root required; execute-only proof unqualified")
        a = self.base / "a"
        final = a / "b" / "final"
        final.mkdir(parents=True, mode=0o700)
        held = c.oi._open_dir(final)
        try:
            os.chmod(a, 0o111)
            try:
                try:
                    os.close(c.oi._open_dir(final))
                except PermissionError as exc:
                    self.assertEqual(exc.errno, errno.EACCES)
                else:
                    self.fail("original walker read intermediate 0o111; unqualified")
                with c._HeldInventory(0) as inventory:
                    with self.assertRaises(c.BoundaryStoreError) as refused:
                        inventory._hold(str(final))
                self.assertEqual(refused.exception.code, "inventory_parent")
                cause = refused.exception.__cause__
                self.assertIsInstance(cause, PermissionError)
                self.assertEqual(cause.errno, errno.EACCES)
                c._same_directory(str(final), held)
            finally:
                os.chmod(a, 0o700)
            os.chmod(final, 0o311)
            try:
                with self.assertRaises(PermissionError) as unreadable:
                    os.close(c._open_current_directory(str(final)))
                self.assertEqual(unreadable.exception.errno, errno.EACCES)
            finally:
                os.chmod(final, 0o700)
        finally:
            os.close(held)

    def test_root_and_depths_share_identity_with_cloexec(self):
        deep = self.base / "x" / "y" / "z"
        deep.mkdir(parents=True)
        for path in (Path("/"), self.base, deep):
            with self.subTest(path=path):
                fd = c._open_current_directory(str(path))
                try:
                    flags = fcntl.fcntl(fd, fcntl.F_GETFD)
                    self.assertTrue(flags & fcntl.FD_CLOEXEC)
                    got, want = os.fstat(fd), os.stat(path)
                    self.assertEqual(
                        (got.st_dev, got.st_ino), (want.st_dev, want.st_ino)
                    )
                    self.assertEqual(c._identity(got), c._identity(want))
                finally:
                    os.close(fd)
                held = c.oi._open_dir(path)
                try:
                    c._same_directory(str(path), held)
                finally:
                    os.close(held)

    def test_refusals_match_original_walker(self):
        (self.base / "file").write_bytes(b"x")
        real = self.base / "real"
        (real / "inner").mkdir(parents=True)
        (self.base / "link").symlink_to(real, target_is_directory=True)
        cases = (
            self.base / "missing",
            self.base / "missing" / "child",
            self.base / "file",
            self.base / "file" / "child",
            self.base / "link",
            self.base / "link" / "inner",
        )
        for path in cases:
            with self.subTest(path=path):
                self.assertEqual(_new_error(path), _old_error(path))

    def test_replaced_final_drifts_then_missing_matches(self):
        final = self.base / "final"
        final.mkdir()
        held = c.oi._open_dir(final)
        try:
            final.rename(self.base / "moved")
            final.mkdir()
            with self.assertRaises(c.BoundaryStoreError) as drift:
                c._same_directory(str(final), held)
            self.assertEqual(drift.exception.code, "identity_drift")
            final.rmdir()
            expected = _old_error(final)
            with self.assertRaises(OSError) as gone:
                c._same_directory(str(final), held)
            self.assertEqual(
                (type(gone.exception), gone.exception.errno), expected
            )
        finally:
            os.close(held)

    def test_admitted_readonly_file_write_or_replace_drifts(self):
        for change in ("write", "replace"):
            with self.subTest(change=change):
                path = self.base / f"{change}.bin"
                path.write_bytes(b"tiny")
                path.chmod(0o444)
                with c._HeldInventory(64) as inventory:
                    inventory.admit(str(path), 4)
                    if change == "write":
                        path.chmod(0o644)
                        path.write_bytes(b"TINY")
                    else:
                        spare = self.base / f"{change}.new"
                        spare.write_bytes(b"tiny")
                        spare.chmod(0o444)
                        os.replace(spare, path)
                    with self.assertRaises(c.BoundaryStoreError) as drift:
                        inventory.check()
                    self.assertEqual(drift.exception.code, "identity_drift")

import resource


class GenuineDirectoryDescriptors(unittest.TestCase):
    # Real observations only: wrappers forward to the actual os calls and the
    # actual ctypes syscall function, recording their genuine return values.

    def _actual_abi(self):
        self.assertTrue(c._abi_profile(), "unsupported ABI profile")
        abi = c._abi_setup()
        self.assertIsNotNone(abi)
        return abi

    def _observer(self, abi, log, before=None):
        real = abi[2]

        def fn(*args):
            if before is not None:
                before()
            raw = real(*args)
            log.append(raw)
            return raw

        return fn

    @staticmethod
    def _proc_count():
        return len(os.listdir("/proc/self/fd"))

    def test_actual_peak_reserve_result_and_cloexec(self):
        owner, lib, fn = self._actual_abi()
        real_open, real_close, real_fstat = os.open, os.close, os.fstat
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp).resolve()
            held = c.oi._open_dir(target)
            try:
                opened, closed, fstats, raws, live = [], [], [], [], set()
                peak = [0]
                cloexec = {}

                def w_open(*a, **k):
                    fd = real_open(*a, **k)
                    opened.append(fd)
                    live.add(fd)
                    peak[0] = max(peak[0], len(live))
                    return fd

                def w_close(fd):
                    if fd in raws:
                        cloexec[fd] = fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
                    closed.append(fd)
                    live.discard(fd)
                    return real_close(fd)

                def w_fstat(fd, *a, **k):
                    fstats.append(fd)
                    return real_fstat(fd, *a, **k)

                observer_base = self._observer((owner, lib, fn), raws)

                def observer(*a):
                    raw = observer_base(*a)
                    if raw >= 0:
                        live.add(raw)
                        peak[0] = max(peak[0], len(live))
                    return raw

                before = self._proc_count()
                with mock.patch.object(c, "_ABI", (owner, lib, observer)), \
                        mock.patch.object(os, "open", w_open), \
                        mock.patch.object(os, "close", w_close), \
                        mock.patch.object(os, "fstat", w_fstat):
                    c._same_directory(target, held)
                after = self._proc_count()

                self.assertEqual(len(raws), 1, "actual syscall not reached")
                result = raws[0]
                self.assertGreaterEqual(result, 0, "actual openat2 failed")
                self.assertEqual(len(opened), 1)
                reserve = opened[0]
                self.assertNotEqual(reserve, result)
                self.assertEqual(peak[0], 2)
                self.assertEqual(closed, [reserve, result])
                self.assertEqual(len(fstats), 2)
                self.assertIn(result, fstats)
                self.assertIn(held, fstats)
                self.assertEqual(cloexec, {result: fcntl.FD_CLOEXEC})
                self.assertEqual(live, set())
                self.assertEqual(before, after)
            finally:
                os.close(held)

    def test_one_free_and_two_free_margin(self):
        owner, lib, fn = self._actual_abi()
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        self.assertNotEqual(soft, resource.RLIM_INFINITY)
        self.assertLessEqual(soft, 1024)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp).resolve()
            before = self._proc_count()
            held = c.oi._open_dir(target)
            fillers = []
            try:
                for _ in range(soft + 1):
                    try:
                        fillers.append(os.open("/", os.O_RDONLY | os.O_CLOEXEC))
                    except OSError as error:
                        self.assertEqual(error.errno, errno.EMFILE)
                        break
                else:
                    self.fail("EMFILE not reached within bound")

                os.close(fillers.pop())
                with self.assertRaises(OSError) as walker_err:
                    c.oi._open_dir(target)
                self.assertEqual(walker_err.exception.errno, errno.EMFILE)
                raws = []
                with mock.patch.object(c, "_ABI", (owner, lib, self._observer((owner, lib, fn), raws))):
                    with self.assertRaises(OSError) as helper_err:
                        c._open_current_directory(target)
                self.assertEqual(helper_err.exception.errno, errno.EMFILE)

                os.close(fillers.pop())
                raws = []
                with mock.patch.object(c, "_ABI", (owner, lib, self._observer((owner, lib, fn), raws))):
                    fd = c._open_current_directory(target)
                os.close(fd)
                self.assertEqual(len(raws), 1)
                self.assertGreaterEqual(raws[0], 0, "helper did not use actual syscall")
                self.assertEqual(raws[0], fd)
                fd = c.oi._open_dir(target)
                os.close(fd)
            finally:
                while fillers:
                    os.close(fillers.pop())
                os.close(held)
            self.assertEqual(before, self._proc_count())

    def test_actual_fd0_result_ownership(self):
        owner, lib, fn = self._actual_abi()
        real_open, real_close = os.open, os.close
        stdin_inheritable = os.get_inheritable(0)
        saved = os.dup(0)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp).resolve()
                held = c.oi._open_dir(target)
                try:
                    opened, closes0, observer_closes, raws = [], [], [], []

                    def w_open(*a, **k):
                        fd = real_open(*a, **k)
                        opened.append(fd)
                        return fd

                    def w_close(fd):
                        if fd == 0:
                            closes0.append("generation")
                        return real_close(fd)

                    def close_old_stdin():
                        real_close(0)
                        observer_closes.append(0)

                    observer = self._observer((owner, lib, fn), raws, close_old_stdin)
                    with mock.patch.object(c, "_ABI", (owner, lib, observer)), \
                            mock.patch.object(os, "open", w_open), \
                            mock.patch.object(os, "close", w_close):
                        c._same_directory(target, held)

                    self.assertEqual(observer_closes, [0])
                    self.assertEqual(raws, [0], "actual syscall did not return 0")
                    self.assertEqual(len(opened), 1)
                    self.assertNotEqual(opened[0], 0)
                    self.assertEqual(closes0, ["generation"])
                finally:
                    os.close(held)
        finally:
            os.dup2(saved, 0, inheritable=stdin_inheritable)
            os.close(saved)


if __name__ == "__main__":
    unittest.main()
