"""Affected stdlib-only eligibility tests; no Native/provider/model imports."""
from __future__ import annotations

import hashlib
import tempfile
import typing
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from masy import runtime_worker as worker


class RuntimePreexpansionTests(unittest.TestCase):
    def test_unicode_line_population_matches_the_actual_decoder(self):
        endings = (b'\n', b'\r', b'\r\n', b'\v', b'\f', b'\x1c', b'\x1d',
                   b'\x1e', b'\xc2\x85', b'\xe2\x80\xa8', b'\xe2\x80\xa9')
        examples = [b'', b'word', b'\x85', b'\xc2', b'\xe2\x80', b'\xff\xc2\x85\xff']
        for ending in endings:
            examples.extend((ending, b'a' + ending, ending + b'a',
                             b'a' + ending + b'b' + ending, b'\xff' + ending + b'\x85'))
        examples.append(b''.join(endings))
        for raw in examples:
            self.assertEqual(worker._compiled_maps_rows(raw),
                             len(raw.decode('utf-8', 'surrogateescape').splitlines()))

    def test_oversized_frontier_refuses_before_sort_or_path_expansion(self):
        modules = {str(i): None for i in range(worker._COMPILED_FRONTIER_LIMIT // 7 + 1)}
        for collect in (worker.compiled_identities, worker._compiled_identities_body):
            with (
                patch.object(worker, 'sorted', side_effect=AssertionError('early sort'), create=True),
                patch.object(worker.os.path, 'realpath', side_effect=AssertionError('early realpath')),
                self.assertRaises(worker.WorkerRefusal) as caught,
            ):
                collect(modules, max_count=10, max_bytes=10, library_roots=[])
            self.assertEqual(caught.exception.code, 'compiled_preexpansion_unsupported')
        with tempfile.TemporaryDirectory() as directory:
            maps = Path(directory) / 'maps'
            # Unicode NEL, not LF: the raw byte cap alone would admit this.
            maps.write_bytes(b'\xc2\x85' * 100001)
            with self.assertRaises(worker.WorkerRefusal):
                worker._mapped_libraries(str(maps), [])

    def test_custom_getters_descriptors_and_foreign_keys_are_not_invoked(self):
        calls = []

        class Hooked(ModuleType):
            def __getattribute__(self, name):
                calls.append(name)
                raise AssertionError('custom getter executed')

        class Descriptor:
            def __get__(self, *args):
                calls.append('descriptor')
                raise AssertionError('descriptor executed')

        class WithDescriptor(ModuleType):
            __file__ = Descriptor()

        class ForeignKey:
            def __hash__(self):
                return hash('__file__')

            def __eq__(self, other):
                calls.append('foreign equality')
                raise AssertionError('foreign equality executed')

        foreign = ModuleType('foreign')
        foreign.__dict__[ForeignKey()] = None
        bad_file = ModuleType('bad_file')
        bad_file.__file__ = object()
        for holder in (Hooked('hooked'), WithDescriptor('descriptor'), foreign, bad_file, object()):
            with self.assertRaises(worker.WorkerRefusal):
                worker._compiled_static_file(holder, worker._COMPILED_FRONTIER_LIMIT)
        self.assertEqual(calls, [])

    def test_original_aliases_maps_hashes_and_empty_holder_semantics_survive(self):
        class Lazy(ModuleType):
            def __getattr__(self, name):
                raise AssertionError('file was present; no fallback is needed')

        class Plain:
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suffix = worker.importlib.machinery.EXTENSION_SUFFIXES[0]
            extension = root / ('extension' + suffix)
            shared = root / 'library.so'
            extension.write_bytes(b'extension Source fixture')
            shared.write_bytes(b'mapped Source fixture')
            maps = root / 'maps'
            maps.write_text(f'1-2 r-xp 0 0:0 0 {shared}\n'
                            f'2-3 r-xp 0 0:0 0 {shared} (deleted)\n')
            module = Lazy('extension')
            module.__file__ = str(extension)
            modules = {'z.alias': module, 'a.alias': module, 'none': None, 'class': Plain}
            expected = [
                {'path': str(extension), 'kind': 'extension_module',
                 'modules': ['a.alias', 'z.alias'], 'size': extension.stat().st_size,
                 'sha256': hashlib.sha256(extension.read_bytes()).hexdigest()},
                {'path': str(shared), 'kind': 'mapped_library', 'modules': [],
                 'size': shared.stat().st_size, 'sha256': hashlib.sha256(shared.read_bytes()).hexdigest()},
            ]
            for collect in (worker.compiled_identities, worker._compiled_identities_body):
                self.assertEqual(collect(modules, max_count=10, max_bytes=1024,
                                         maps_path=str(maps), library_roots=[str(root)]), expected)

    def test_same_raw_maps_buffer_and_admitted_references_are_retained(self):
        module = ModuleType('module')
        module.__file__ = None
        modules = {'module': module}
        with patch.object(worker, '_compiled_maps_data', return_value=b'') as read:
            self.assertEqual(worker.compiled_identities(modules, max_count=1,
                             max_bytes=1, library_roots=[]), [])
        self.assertEqual(read.call_count, 1)
        admitted, _data, _roots = worker._compiled_preexpansion(modules, '/does-not-exist/971', [])
        modules['module'] = ModuleType('replacement')
        with self.assertRaises(worker.WorkerRefusal) as caught:
            worker._compiled_admitted_file(modules, 'module', admitted)
        self.assertEqual(caught.exception.code, 'compiled_preexpansion_changed')
        modules['module'] = module
        module.__file__ = 'changed'
        with self.assertRaises(worker.WorkerRefusal):
            worker._compiled_admitted_file(modules, 'module', admitted)
        del modules['module']
        modules['renamed'] = module
        with self.assertRaises(worker.WorkerRefusal) as caught:
            worker._compiled_admitted_file(modules, 'renamed', admitted)
        self.assertEqual(caught.exception.code, 'compiled_preexpansion_changed')

    def test_genuine_typing_holders_and_exact_custom_getter_refusal(self):
        holders = {'typing.io': typing.io, 'typing.re': typing.re}
        self.assertEqual(worker.compiled_identities(holders, max_count=1, max_bytes=1,
                         maps_path='/does-not-exist/971', library_roots=[]), [])
        meta = type(typing.io)
        impostor = meta('io', (), {'__module__': 'typing'})
        with self.assertRaises(worker.WorkerRefusal):
            worker._compiled_static_file(impostor, 100000)
        original = meta.__dict__['__getattribute__']
        calls = []

        def changed(cls, name):
            calls.append(name)
            raise AssertionError('modified typing getter executed')

        try:
            type.__setattr__(meta, '__getattribute__', changed)
            with self.assertRaises(worker.WorkerRefusal):
                worker._compiled_static_file(typing.io, 100000)
            self.assertEqual(calls, [])
        finally:
            type.__setattr__(meta, '__getattribute__', original)
        class ForeignPath:
            def __ne__(self, other):
                calls.append('foreign path comparison')
                raise AssertionError('foreign value compared before type admission')

        with patch.dict(typing.__dict__, {'__file__': ForeignPath()}), self.assertRaises(worker.WorkerRefusal):
            worker._compiled_static_file(typing.io, 100000)
        self.assertEqual(calls, [])

        class ForeignTypingKey:
            def __hash__(self):
                return hash('__file__')

            def __eq__(self, other):
                calls.append('foreign typing key equality')
                raise AssertionError('typing namespace looked up before key admission')

        key = ForeignTypingKey()
        original_path = typing.__dict__.pop('__file__')
        typing.__dict__[key] = original_path
        try:
            with self.assertRaises(worker.WorkerRefusal):
                worker._compiled_static_file(typing.io, 100000)
            self.assertEqual(calls, [])
        finally:
            del typing.__dict__[key]
            typing.__dict__['__file__'] = original_path

    def test_scoped_failure_discards_pending_scan_and_images_bind_new_owners(self):
        token = object()
        scope = worker._new_compiled_reuse(token, max_count=10, max_bytes=1024)
        with self.assertRaises(worker.WorkerRefusal):
            worker._compiled_identities_scoped(scope, token, {'bad': object()},
                    max_count=10, max_bytes=1024, maps_path='/does-not-exist/971', library_roots=[])
        self.assertIsNone(scope._next)
        self.assertIsNone(scope._rows)
        image = worker._compiled_reuse_image()
        self.assertIsNotNone(image)
        self.assertTrue(worker._same_compiled_image(image, worker._compiled_reuse_image()))
        charge = worker._compiled_reuse_charge()
        self.assertEqual(set(charge), {'rows', 'fixed', 'bytes', 'max_count',
                         'max_path_chars', 'row_bytes', 'image_functions',
                         'image_operators', 'image_constants'})
        self.assertEqual(charge['bytes'], charge['rows'] + charge['fixed'])
        self.assertGreater(charge['fixed'], 0)
        with patch.object(worker, '_COMPILED_LINE_ENDINGS', (b'\n',)):
            self.assertFalse(worker._same_compiled_image(image, worker._compiled_reuse_image()))
        with patch.object(worker, '_COMPILED_SYS_MODULES', {}):
            self.assertFalse(worker._same_compiled_image(image, worker._compiled_reuse_image()))
        with patch.object(worker, '_COMPILED_TYPE_MRO', object()):
            # Descriptor identity is image-bound even if no module is scanned.
            self.assertFalse(worker._same_compiled_image(image, worker._compiled_reuse_image()))


if __name__ == '__main__':
    unittest.main()
