"""Synthetic code-identity DTO checks, never a Native/runtime qualification.

Compile the exact owning pure function bodies and their real dependencies.
No model module, extension binary or generated controller is imported/executed.
"""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.machinery
import json
import os
import stat
import sysconfig
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def extract(path, names, namespace):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    selected = []
    for name in names:
        matches = [node for node in tree.body
                   if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
                   or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                                                           and target.id == name for target in node.targets))]
        if len(matches) != 1:
            raise AssertionError('exact Source symbol required: ' + name)
        selected.append(matches[0])
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)


class TestCaseCPUCode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Product layout first; task-only Source copy is for archive execution.
        schedule = ROOT / 'src/masy/case_schedule.py'
        if not schedule.is_file():
            schedule = ROOT.parent / 'Source_inputs/case_schedule.py'
        cli = {'__name__': 'synthetic_case_code_protocol', 'Path': Path,
               'json': json, 'hashlib': hashlib, 'os': os, 'stat': stat}
        extract(schedule, ('CaseScheduleError', 'closed', 'integer'), cli)
        extract(ROOT / 'src/masy/case_cli.py',
                ('canonical', 'sha', 'strict_json', 'read_pin', 'path_value',
                 '_CPU_EXTENSION_OWNERS', 'cpu_extensions_from'), cli)
        worker = {'__name__': 'synthetic_case_code_worker_protocol', 'Path': Path, 'json': json}
        extract(ROOT / 'src/masy/case_worker.py',
                ('_require_extension_paths', '_compiled_identity_error', '_compiled_extension_record'), worker)
        cls.cli, cls.worker = cli, worker

    def test_exact_five_owner_parser_rejects_missing_unknown_duplicate_and_changed(self):
        owners = self.cli['_CPU_EXTENSION_OWNERS']
        self.assertEqual(owners, ('jaxlib_mlir_hlo', 'numpy_fft', 'stdlib_readline',
                                  'stdlib_resource', 'stdlib_mmap'))
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for owner in owners:
                path = Path(directory).resolve() / (owner + '.synthetic')
                data = ('synthetic DTO only: ' + owner).encode()
                path.write_bytes(data)
                rows.append({'owner': owner, 'path': str(path), 'size': len(data),
                             'sha256': hashlib.sha256(data).hexdigest()})
            value = {'schema': 'masy.case_cpu_extensions.v1', 'lock_id': 'synthetic lock',
                     'source_heads': {}, 'packages': {}, 'interpreter_identity': {},
                     'extensions': rows, 'qualification_origin': 'synthetic protocol; no runtime authority'}
            limits = {'max_compiled_files': 5, 'max_compiled_bytes': sum(row['size'] for row in rows)}
            def parse(item, selected_limits=limits):
                return self.cli['cpu_extensions_from'](self.cli['canonical'](item),
                    lock_id=value['lock_id'], source_heads={}, packages={},
                    interpreter_identity={}, limits=selected_limits)
            self.assertEqual(parse(value), value)
            missing = copy.deepcopy(value); missing['extensions'].pop()
            unknown = copy.deepcopy(value); unknown['extensions'][-1]['owner'] = 'arbitrary_library'
            duplicate = copy.deepcopy(value); duplicate['extensions'][-1] = dict(duplicate['extensions'][0])
            changed = copy.deepcopy(value); changed['extensions'][-1]['sha256'] = '0' * 64
            for bad in (missing, unknown, duplicate, changed):
                with self.assertRaises(ValueError):
                    parse(bad)
            for small in ({**limits, 'max_compiled_files': 4},
                          {**limits, 'max_compiled_bytes': limits['max_compiled_bytes'] - 1}):
                with self.assertRaises(ValueError):
                    parse(value, small)

    def test_mmap_path_requires_selected_dynload_directory_and_real_suffix(self):
        # Harmless metadata paths only: no extension is read or imported.
        dyn = Path(sysconfig.get_config_var('DESTSHARED')).resolve(strict=True)
        site = Path(sysconfig.get_path('platlib')).resolve(strict=True)
        suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
        check = self.worker['_require_extension_paths']
        check({'extensions': [{'owner': 'stdlib_mmap', 'path': str(dyn / ('mmap' + suffix))}]})
        for path in (site / ('mmap' + suffix), dyn / 'mmap.unqualified', dyn / ('resource' + suffix)):
            with self.assertRaises(ValueError):
                check({'extensions': [{'owner': 'stdlib_mmap', 'path': str(path)}]})
        with self.assertRaises(ValueError):
            check({'extensions': [{'owner': 'unknown', 'path': str(dyn / ('mmap' + suffix))}]})

    def test_original_pins_remain_mandatory_and_failure_detail_is_bounded(self):
        def row(path, digest='a' * 64):
            return {'path': path, 'kind': 'extension_module', 'modules': ['synthetic'],
                    'size': 10, 'sha256': digest}
        baseline = [row('/synthetic/original.so')]
        extensions = [{'owner': owner, 'path': '/synthetic/' + owner + '.so',
                       'size': 10, 'sha256': 'b' * 64} for owner in self.cli['_CPU_EXTENSION_OWNERS']]
        contract = {'extensions': extensions, 'qualification_origin': 'synthetic DTO only'}
        actual = [row(baseline[0]['path']), row(extensions[-1]['path'], 'b' * 64)]
        guard = self.worker['_compiled_extension_record']
        limits = {'max_compiled_files': 6, 'max_compiled_bytes': 60}
        out = guard(actual, baseline, contract, limits)
        self.assertIs(out['actual'], actual); self.assertIs(out['baseline'], baseline)
        for bad in (actual[1:], [row(baseline[0]['path'], 'c' * 64)],
                    actual + [row('/synthetic/unknown.so')]):
            with self.assertRaises(ValueError) as caught:
                guard(bad, baseline, contract, limits)
            text = str(caught.exception)
            self.assertLessEqual(len(text.encode('ascii')), 65536)
            detail = json.loads(text.partition('; compiled_identity=')[2])
            self.assertEqual(set(detail), {'path', 'path_truncated', 'expected', 'actual'})
            self.assertFalse(detail['path_truncated'])
        for small in ({**limits, 'max_compiled_files': 5}, {**limits, 'max_compiled_bytes': 59}):
            with self.assertRaises(ValueError):
                guard(actual, baseline, contract, small)
        error = self.worker['_compiled_identity_error']('original refusal', '/' + '\U0001f332' * 5000,
                                                        (10, 'a' * 64), None)
        self.assertLessEqual(len(str(error).encode('ascii')), 65536)
        detail = json.loads(str(error).partition('; compiled_identity=')[2])
        self.assertTrue(detail['path_truncated']); self.assertEqual(len(detail['path']), 4096)
        fallback = self.worker['_compiled_identity_error']('original refusal', '/synthetic/x', object(), None)
        self.assertEqual(str(fallback), 'original refusal; compiled_identity_detail_unavailable')


if __name__ == '__main__':
    unittest.main()
