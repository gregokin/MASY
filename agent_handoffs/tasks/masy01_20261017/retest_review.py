"""Independent Codex regression probes for the concrete first-review defects.

Only scalar/synthetic source fixtures; no actual model state or transport.
"""
from pathlib import Path
import argparse
import dataclasses
from fractions import Fraction
import json
import os
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

TASK = Path(__file__).resolve().parent
PYTHON = '/home/okin/MAPLE/.venv/bin/python'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = args.snapshot.resolve()
    sys.pycache_prefix = str(TASK / 'independent_retest_cache')
    sys.path.insert(0, str(source / 'src'))
    from masy.clock import ClockLedger, ClockError, resolve_forcing_edges, encode_time, decode_time
    from masy.config import ConfigError, parse_config
    from masy.contracts import MASY_REQUIRED_API, Requirement, check_masy_api
    from masy.report import publish_report, ReportTooLarge
    from masy import sourceobs, importcheck
    results = {}

    def refusal(name, expected, call):
        try:
            value = call()
        except expected as exc:
            results[name] = {'pass': True, 'exception': type(exc).__name__, 'message': str(exc)}
        except Exception as exc:
            results[name] = {'pass': False, 'exception': type(exc).__name__, 'message': str(exc)}
        else:
            results[name] = {'pass': False, 'unexpected_acceptance': str(value)[:200]}

    refusal('first_positive_forcing_edge_collapse', ClockError,
            lambda: resolve_forcing_edges(float(2**53), ['1', '2']))
    ordinary = resolve_forcing_edges(0.1, ['0', '0.2', '0.4'])
    results['ordinary_decimal_unchanged'] = {
        'pass': ordinary['edges'][1]['native']['hex'] == (0.1 + 0.2).hex()}
    refusal('negative_epoch', ClockError, lambda: ClockLedger(-2.0))
    config = {'schema': 'masy.preflight.v1', 'sources': {'maple': {'root': '/fake/MAPLE'},
                                                       'syrup': {'root': '/fake/SYRUP'}},
              'clock': {'native_epoch_s': -2.0}}
    refusal('negative_config_epoch', ConfigError, lambda: parse_config(config, masy_root=str(source)))
    ledger = ClockLedger(100.0)
    ledger.append('event', native_start=100., native_end=101., commit_before=100., commit_after=101.)
    record = ledger.as_record()
    record['global_elapsed_definition'] = 'native_t * 1000'
    refusal('changed_serialized_definition', ClockError, lambda: ClockLedger.from_record(record))

    with tempfile.TemporaryDirectory(prefix='masy-independent-', dir=TASK) as directory:
        root = Path(directory)
        fake = root / 'fake_git.py'
        fake.write_text('#!' + PYTHON + '\nimport sys\nsys.stderr.write("x"*4096)\nprint("ok")\n')
        fake.chmod(0o700)
        with patch.object(sourceobs, 'GIT_EXECUTABLE', str(fake)):
            refusal('git_combined_output_cap', sourceobs.SourceRefusal,
                    lambda: sourceobs.run_git(root, [], sourceobs.Limits(git_timeout_s=3.,
                                                                        git_max_output_bytes=64)))

        reportdir = root / 'report'
        reportdir.mkdir()
        refusal('report_hard_byte_cap', ReportTooLarge,
                lambda: publish_report(str(reportdir), {'verdict': 'not_transport_ready', 'body': 'x'*200},
                                       max_report_bytes=64))
        results['report_hard_byte_cap']['no_files_written'] = not any(reportdir.iterdir())
        results['report_hard_byte_cap']['pass'] &= not any(reportdir.iterdir())

        binary = root / 'identity.bin'
        binary.write_bytes(b'a' * 8)
        original_stat = importcheck.os.stat
        def grow(path, *positional, **kwargs):
            value = original_stat(path, *positional, **kwargs)
            if str(path) == str(binary):
                with open(binary, 'ab') as stream:
                    stream.write(b'b' * 64)
            return value
        with patch.object(importcheck, 'MAX_HASH_BYTES', 16), patch.object(importcheck.os, 'stat', side_effect=grow):
            identity = importcheck._bounded_sha256(str(binary))
        results['changed_binary_identity'] = {'pass': identity.get('sha256') is None and bool(identity.get('reason')),
                                               'record': identity}

        maple, syrup = root / 'MAPLE', root / 'SYRUP'
        (maple / 'src' / 'maple').mkdir(parents=True)
        syrup.mkdir()
        cfg = root / 'config.toml'
        cfg.write_text('schema="masy.preflight.v1"\n[sources.maple]\nroot=' + json.dumps(str(maple)) +
                       '\n[sources.syrup]\nroot=' + json.dumps(str(syrup)) + '\n')
        launcher = source / 'tools' / 'masy_preflight.py'
        before = sorted(str(x.relative_to(maple)) for x in maple.rglob('*'))
        result = subprocess.run([PYTHON, '-B', '-I', str(launcher), '--config', str(cfg), '--run-dir',
                                 str(maple / 'src' / 'maple' / 'audit_output')],
                                capture_output=True, text=True, timeout=20)
        after = sorted(str(x.relative_to(maple)) for x in maple.rglob('*'))
        results['source_output_overlap'] = {'pass': result.returncode == 2 and before == after,
                                            'exit_code': result.returncode, 'unchanged': before == after}
        result = subprocess.run([PYTHON, '-B', '-I', str(launcher), '--help'], cwd=root,
                                capture_output=True, text=True, timeout=15)
        results['informational_help'] = {'pass': result.returncode == 0 and 'usage' in result.stdout.lower(),
                                          'exit_code': result.returncode}

    req = next(item for item in MASY_REQUIRED_API if item.attribute == 'DryHandoff')
    fake = types.ModuleType(req.module)
    fake.DryHandoff = None
    with patch.dict(sys.modules, {req.module: fake}):
        inspection = check_masy_api((req,))
        results['none_handoff_refused'] = {'pass': inspection[0]['status'] == 'fail', 'result': inspection}
        fake.DryHandoff = dataclasses.make_dataclass('DryHandoff', [(name, object) for name in req.fields])
        results['declared_handoff_fields_pass'] = {'pass': check_masy_api((req,))[0]['status'] == 'pass'}
        fake.DryHandoff = dataclasses.make_dataclass('DryHandoff', [(name, object) for name in req.fields[:-1]])
        results['missing_handoff_field_refused'] = {'pass': check_masy_api((req,))[0]['status'] == 'fail'}
    for attribute in ('advance_wind_block', 'reroute_threshold_m', 'validate_event_boundary',
                      'mobile_state_mass_is_finite', 'mobile_state_is_genuinely_empty'):
        results['required_' + attribute] = {'pass': any(x.attribute == attribute for x in MASY_REQUIRED_API)}
    results['dense_dependency_declared'] = {
        'pass': any(x.attribute == 'WindowedVoxelColumnState' and x.kind == 'class' and x.required_by
                    for x in MASY_REQUIRED_API),
        'windowed_type_requirement_present': any(x.attribute == 'WindowedVoxelColumnState' for x in MASY_REQUIRED_API),
        'note': 'New source finding after review_01; required by current SYRUP Dense classifier. See dense_compatibility_finding.md.'}

    conversion = ClockLedger(100.0)
    refusal('conversion_negative_native', ClockError, lambda: conversion.global_elapsed(-1.0))
    refusal('conversion_signed_zero_native', ClockError, lambda: conversion.global_elapsed(-0.0))
    refusal('conversion_negative_native_inverse', ClockError, lambda: conversion.native_from_global(Fraction(-101)))
    for native in (0.0, 25.0, 100.0, 101.25):
        global_offset = conversion.global_elapsed(native)
        results['conversion_roundtrip_' + native.hex()] = {
            'pass': conversion.native_from_global(global_offset).hex() == native.hex(),
            'global_offset': str(global_offset), 'native_hex': native.hex()}
    results['raw_codec_signed_zero_preserved'] = {
        'pass': decode_time(encode_time(-0.0)).hex() == (-0.0).hex()}

    export = types.ModuleType('masy_reexport_fixture')
    export.PublicColumn = type('InternalColumn', (), {'__module__': 'native_implementation'})
    public_requirement = Requirement(export.__name__, 'PublicColumn', kind='class')
    with patch.dict(sys.modules, {export.__name__: export}):
        inspection = check_masy_api((public_requirement,))
        results['legitimate_class_reexport'] = {'pass': inspection[0]['status'] == 'pass', 'result': inspection}
        for label, value in (('none', None), ('function', lambda: None), ('instance', object())):
            export.PublicColumn = value
            inspection = check_masy_api((public_requirement,))
            results['nonclass_' + label] = {'pass': inspection[0]['status'] == 'fail', 'result': inspection}
        del export.PublicColumn
        inspection = check_masy_api((public_requirement,))
        results['missing_class_binding'] = {'pass': inspection[0]['status'] == 'fail', 'result': inspection}
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))
    return 0 if all(item.get('pass', True) for item in results.values()) else 1

if __name__ == '__main__':
    raise SystemExit(main())
