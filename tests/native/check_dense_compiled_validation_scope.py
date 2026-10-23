"""Source-authored compiled-scope controls; Root alone executes them.

Temporary byte files are filesystem/hash interface fixtures, not Native modules
or model positives. The separate loaded-runtime case never mocks a provider.
"""
import hashlib
import importlib.machinery
import os
import pickle
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from masy import runtime_worker as rw


class CompiledFileScope(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.maps = self.root / 'maps.txt'
        self.maps.write_text('')
        self.path = self.root / ('fixture' + importlib.machinery.EXTENSION_SUFFIXES[-1])
        self.path.write_bytes(b'authored-metadata-interface\n' * 8)
        self.modules = {'fixture': types.SimpleNamespace(__file__=str(self.path))}
        self.token = object()

    def scope(self, count=2, size=4096):
        return rw._new_compiled_reuse(self.token, max_count=count, max_bytes=size)

    def scan(self, scope, modules=None, count=2, size=4096):
        return rw._compiled_identities_scoped(
            scope, self.token, self.modules if modules is None else modules,
            max_count=count, max_bytes=size, maps_path=str(self.maps), library_roots=[])

    def full(self, modules=None, count=2, size=4096):
        return rw.compiled_identities(
            self.modules if modules is None else modules, max_count=count,
            max_bytes=size, maps_path=str(self.maps), library_roots=[])

    def test_real_first_hit_final_records_and_payload_reads(self):
        expected = self.full()
        # Transparent observations cover actual filesystem reads; no Native API.
        with mock.patch.object(rw.os, 'read', wraps=os.read) as read:
            scope = self.scope()
            self.assertEqual(self.scan(scope), expected)
            self.assertGreater(read.call_count, 0)
            read.reset_mock()
            self.assertEqual(self.scan(scope), expected)
            self.assertEqual(read.call_count, 0)
            scope._force_full = True
            self.assertEqual(self.scan(scope), expected)
            self.assertGreater(read.call_count, 0)

    def test_hit_still_opens_and_checks_current_descriptor(self):
        with mock.patch.object(rw.os, 'open', wraps=os.open) as opened, \
                mock.patch.object(rw.os, 'fstat', wraps=os.fstat) as fstat:
            scope = self.scope()
            self.scan(scope)
            opened.reset_mock(); fstat.reset_mock()
            self.scan(scope)
            self.assertEqual(opened.call_count, 1)
            self.assertEqual(fstat.call_count, 2)

    def test_same_size_write_restored_mtime_uses_real_changed_hash(self):
        scope = self.scope(); before = self.scan(scope)
        info = self.path.stat(); old_stamp = rw._file_identity(info)
        self.path.write_bytes(b'X' * info.st_size)
        os.utime(self.path, ns=(info.st_atime_ns, info.st_mtime_ns))
        self.assertNotEqual(rw._file_identity(self.path.stat()), old_stamp,
                            'this control requires actual ordinary metadata drift')
        actual = self.scan(scope)
        self.assertEqual(actual, self.full())
        self.assertNotEqual(actual[0]['sha256'], before[0]['sha256'])

    def test_same_bytes_replacement_preserves_original_acceptance(self):
        scope = self.scope(); expected = self.scan(scope)
        other = self.root / 'replacement'
        other.write_bytes(self.path.read_bytes())
        os.replace(other, self.path)
        self.assertEqual(self.scan(scope), self.full())
        self.assertEqual(self.scan(scope), expected)

    def test_missing_file_keeps_original_refusal(self):
        scope = self.scope(); self.scan(scope); self.path.unlink()
        with self.assertRaises(rw.WorkerRefusal) as scoped:
            self.scan(scope)
        with self.assertRaises(rw.WorkerRefusal) as full:
            self.full()
        self.assertEqual(scoped.exception.code, full.exception.code)

    def test_candidate_symlink_is_not_a_regular_hit(self):
        scope = self.scope(); self.scan(scope)
        target = self.root / 'target'; target.write_bytes(b'x')
        self.path.unlink(); self.path.symlink_to(target)
        # Direct private hit-ladder fixture, without the public realpath step.
        scope._begin_scan(1)
        with self.assertRaises(rw.WorkerRefusal) as error:
            scope._hash(str(self.path), 4096)
        self.assertEqual(error.exception.code, 'compiled_identity_unreadable')
        scope._discard_scan()

    def test_hit_permission_error_is_not_retried_into_success(self):
        original = os.open
        state = {'fail': False}
        def observed(path, *args, **kwargs):
            if state['fail'] and str(path) == str(self.path):
                raise PermissionError('authored open refusal')
            return original(path, *args, **kwargs)
        with mock.patch.object(rw.os, 'open', side_effect=observed):
            scope = self.scope(); self.scan(scope); state['fail'] = True
            with self.assertRaises(rw.WorkerRefusal) as error:
                self.scan(scope)
            self.assertEqual(error.exception.code, 'compiled_identity_unreadable')
            self.assertIsNone(scope._next)

    def test_hit_raw_close_failure_preserves_original_error(self):
        original = os.close
        state = {'fail': False}
        def observed(fd):
            original(fd)
            if state['fail']:
                raise OSError('authored close failure after actual close')
        with mock.patch.object(rw.os, 'close', side_effect=observed):
            scope = self.scope(); self.scan(scope); state['fail'] = True
            try:
                with self.assertRaisesRegex(OSError, 'authored close failure'):
                    self.scan(scope)
            finally:
                state['fail'] = False
            self.assertIsNone(scope._next)

    def test_hit_raw_fstat_failure_closes_actual_descriptor(self):
        original = os.fstat
        state = {'fail': False, 'fd': None}
        def observed(fd):
            if state['fail']:
                state['fd'] = fd
                raise OSError('authored fstat refusal')
            return original(fd)
        with mock.patch.object(rw.os, 'fstat', side_effect=observed):
            scope = self.scope(); self.scan(scope); state['fail'] = True
            with self.assertRaisesRegex(OSError, 'authored fstat refusal'):
                self.scan(scope)
            state['fail'] = False
            self.assertIsNotNone(state['fd'])
            with self.assertRaises(OSError):
                original(state['fd'])

    def test_cache_hits_charge_the_current_full_byte_budget(self):
        size = self.path.stat().st_size
        scope = self.scope(size=size + 8); self.scan(scope, size=size + 8)
        extra = self.root / ('second' + importlib.machinery.EXTENSION_SUFFIXES[-1])
        extra.write_bytes(b'Y' * 9)
        modules = {**self.modules, 'second': types.SimpleNamespace(__file__=str(extra))}
        # The unchanged first file is a real hit; the second file fits alone.
        # Only their complete logical byte sum exhausts the current budget.
        self.assertLess(9, size + 8)
        with self.assertRaises(rw.WorkerRefusal) as error:
            self.scan(scope, modules, size=size + 8)
        self.assertEqual(error.exception.code, 'compiled_identity_bound_exceeded')

    def test_new_and_removed_roster_does_not_accumulate_union(self):
        scope = self.scope(); self.scan(scope)
        path = self.root / ('second' + importlib.machinery.EXTENSION_SUFFIXES[-1])
        path.write_bytes(b'new actual bytes')
        modules = {'second': types.SimpleNamespace(__file__=str(path))}
        records = self.scan(scope, modules)
        self.assertEqual(records, self.full(modules))
        self.assertEqual(len(scope._rows), 1)
        self.assertEqual(scope._rows[0].path, str(path))

    def test_partial_scan_never_commits_a_success_roster(self):
        path = self.root / ('second' + importlib.machinery.EXTENSION_SUFFIXES[-1])
        path.write_bytes(b'Z' * 1024)
        modules = {**self.modules, 'second': types.SimpleNamespace(__file__=str(path))}
        scope = self.scope(size=512)
        with self.assertRaises(rw.WorkerRefusal):
            self.scan(scope, modules, size=512)
        self.assertIsNone(scope._rows)
        self.assertIsNone(scope._next)

    def test_actual_mapped_addition_is_discovered_without_a_cached_roster(self):
        scope = self.scope()
        options = dict(max_count=2, max_bytes=4096, maps_path=str(self.maps),
                       library_roots=[str(self.root)])
        rw._compiled_identities_scoped(scope, self.token, self.modules, **options)
        extra = self.root / 'mapped.so'; extra.write_bytes(b'authored mapped interface')
        self.maps.write_text('0 0 0 0 0 ' + str(extra) + '\n')
        actual = rw._compiled_identities_scoped(scope, self.token, self.modules, **options)
        self.assertEqual(actual, rw.compiled_identities(self.modules, **options))
        self.assertEqual({r['kind'] for r in actual}, {'extension_module', 'mapped_library'})

    def test_count_limit_is_checked_before_new_row_allocation(self):
        scope = self.scope(count=1)
        extra = self.root / ('second' + importlib.machinery.EXTENSION_SUFFIXES[-1])
        extra.write_bytes(b'authored second interface')
        modules = {**self.modules, 'second': types.SimpleNamespace(__file__=str(extra))}
        with self.assertRaises(rw.WorkerRefusal) as error:
            self.scan(scope, modules, count=1)
        self.assertEqual(error.exception.code, 'compiled_identity_bound_exceeded')
        self.assertIsNone(scope._next)

    def test_changed_hasher_uses_current_global_full_path(self):
        scope = self.scope(); self.scan(scope)
        original = rw._sha256_file
        with mock.patch.object(rw, '_sha256_file', wraps=original) as hashed:
            self.assertEqual(self.scan(scope), self.full())
            self.assertGreaterEqual(hashed.call_count, 2)
            self.assertTrue(scope._disabled)

    def test_foreign_closed_thread_scope_cannot_reuse(self):
        scope = self.scope(); self.scan(scope)
        with self.assertRaises(rw.WorkerRefusal):
            rw._compiled_identities_scoped(scope, object(), self.modules,
                                          max_count=2, max_bytes=4096,
                                          maps_path=str(self.maps), library_roots=[])
        errors = []
        def other_thread():
            try:
                self.scan(scope)
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=other_thread); thread.start(); thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], rw.WorkerRefusal)
        scope._closed = True
        with self.assertRaises(rw.WorkerRefusal):
            self.scan(scope)

    def test_nested_scan_and_immutable_rows_refuse(self):
        scope = self.scope(); self.scan(scope)
        with self.assertRaises(AttributeError):
            scope._rows[0].digest = '0' * 64
        scope._begin_scan(1)
        with self.assertRaises(rw.WorkerRefusal):
            self.scan(scope)
        self.assertIsNone(scope._next)

    def test_success_rows_and_scope_refuse_serialization(self):
        scope = self.scope(); self.scan(scope)
        for value in (scope, scope._rows[0]):
            with self.subTest(type=type(value).__name__):
                with self.assertRaises(TypeError):
                    pickle.dumps(value)
        with self.assertRaises(AttributeError):
            del scope._rows[0].identity

    def test_fd_zero_is_a_real_descriptor_not_false(self):
        scope = self.scope(); expected = self.scan(scope)
        saved = os.dup(0)
        os.close(0)
        try:
            self.assertEqual(self.scan(scope), expected)
        finally:
            os.dup2(saved, 0); os.close(saved)

    def test_successful_rows_hash_actual_bytes(self):
        scope = self.scope(); records = self.scan(scope)
        self.assertEqual(records[0]['sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())


class CompiledFunctionImage(unittest.TestCase):
    def test_original_refusal_class_cell_has_a_complete_live_image(self):
        image = rw._compiled_reuse_image()
        self.assertIsNotNone(image)
        record = next(row for row in image[0] if row[0] is rw.WorkerRefusal.__init__)
        closure = rw.WorkerRefusal.__init__.__closure__
        self.assertIs(record[6], closure)
        self.assertEqual(len(record[7]), 1)
        self.assertIs(record[7][0][0], closure[0])
        self.assertIs(record[7][0][1], rw.WorkerRefusal)
        self.assertTrue(rw._same_compiled_image(image, rw._compiled_reuse_image()))
        self.assertIsNotNone(rw._compiled_reuse_charge())

    def test_unknown_callable_closure_and_changed_owned_cell_are_ineligible(self):
        def outer():
            value = object()
            def foreign():
                return value
            return foreign
        for function in (mock.MagicMock(wraps=rw._sha256_file), object(), outer()):
            with self.subTest(type=type(function).__name__):
                self.assertIsNone(rw._compiled_function_image(function))
        function = rw.WorkerRefusal.__init__
        cell, original = function.__closure__[0], rw.WorkerRefusal
        before = rw._compiled_reuse_image()
        try:
            cell.cell_contents = object()
            self.assertIsNone(rw._compiled_function_image(function))
            self.assertFalse(rw._same_compiled_image(before, rw._compiled_reuse_image()))
            del cell.cell_contents
            self.assertIsNone(rw._compiled_function_image(function))
        finally:
            cell.cell_contents = original
        self.assertTrue(rw._same_compiled_image(before, rw._compiled_reuse_image()))


class GenuineLoadedRuntimeMetadata(unittest.TestCase):
    def test_actual_loaded_full_scoped_full_records_agree(self):
        # Root must first establish the genuine immutable runtime/origin guards.
        # This observes loaded files, never imports a Native/model/provider here.
        modules = dict(sys.modules)
        expected = rw.compiled_identities(modules, max_count=4000, max_bytes=4 << 30)
        token = object()
        scope = rw._new_compiled_reuse(token, max_count=4000, max_bytes=4 << 30)
        actual = rw._compiled_identities_scoped(scope, token, modules,
                                               max_count=4000, max_bytes=4 << 30)
        self.assertEqual(actual, expected)
        self.assertEqual(rw._compiled_identities_scoped(scope, token, modules,
                                                       max_count=4000, max_bytes=4 << 30), expected)
        scope._force_full = True
        self.assertEqual(rw._compiled_identities_scoped(scope, token, modules,
                                                       max_count=4000, max_bytes=4 << 30), expected)


if __name__ == '__main__':
    unittest.main()
