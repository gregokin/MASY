"""Bounded, explicit JSON case workflow and parent-side original-engine launch.

No Native/provider module is imported by this parent. The worker owns Native
compilation, adoption and transport under the separately declared case policy.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any

from masy.case_schedule import CaseScheduleError, closed, integer, number
from masy.case_runner import CaseCPUProcessPolicy

RUN_SCHEMA = 'masy.case_cpu_run.v1'
OUTER_SCHEMA = 'masy.case_cpu_worker_request.v1'
RESULT_SCHEMA = 'masy.case_cpu_result.v1'
_ARRAYS = ('elevation_full_m', 'export_receiver_full', 'active_mask', 'friction_factor', 'ksat_m_per_s',
           'suction_m', 'drainage_parameter', 'theta_sat', 'soil_thickness_m', 'pavement_cover_fraction',
           'initial_theta', 'retained_soil_water_m', 'rainfall_multiplier')
_NULL_ARRAYS = ('pavement_cover_fraction', 'initial_theta', 'retained_soil_water_m')
_BOOL_ARRAYS = ('export_receiver_full', 'active_mask')
_SPEC_FIELDS = ('profile_id', 'source_reference', *_ARRAYS[:4], 'allow_masked_nodata', 'nodata_value',
                'allow_pit_storage', 'allow_flat_storage', 'soil_model', *_ARRAYS[4:12], 'rainfall_multiplier',
                'wet_classes', 'active_layer_sensitivity_mm', 'ke_model', 'kinematic_viscosity_m2_s',
                'reference_interval_s', 'recession_factor_per_reference_s', 'ke_vegetation_form',
                'raindrop_composition_scaling', 'distance_convention', 'dstar_convention', 'bagnold_depth_units')
_STORM = ('max_dt_s', 'min_dt_s', 'max_retries', 'max_steps', 'courant_max', 'bisection_iterations',
          'root_tolerance_m', 'implementation', 'root_solver', 'newton_max_iterations')
_SEDIMENT = ('storm', 'sediment_courant_max', 'max_transport_substeps', 'commit', 'force_final_commit',
             'max_commit_log', 'max_rejections_recorded', 'transport_scheme', 'phase_bins', 'transport_implementation',
             'phase_storage', 'phase_max_nnz', 'record_exchange_history', 'deep_publication', 'deep_publication_high_water_records')
_COMPLETION = ('max_depth_m', 'surface_volume_m3', 'max_unit_discharge_m2_s', 'outlet_discharge_m3_s',
               'hold_s', 'mobile_per_cell_class_kg', 'terminal_storage_water', 'deposit_mobile_at_flow_end')


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data, *, maximum, entries, text, depth=32):
    if type(data) is not bytes or len(data) > maximum:
        raise ValueError('JSON byte quota exceeded')
    # Bound nesting before constructing the JSON graph. Strings/escapes do not
    # contribute brackets; this is a structural prepass, not a second parser.
    level = 0; quoted = False; escaped = False
    for ch in data:
        if quoted:
            if escaped: escaped = False
            elif ch == 92: escaped = True
            elif ch == 34: quoted = False
        elif ch == 34: quoted = True
        elif ch in (91, 123):
            level += 1
            if level > depth: raise ValueError('JSON nesting quota exceeded')
        elif ch in (93, 125): level -= 1
    def pairs(rows):
        out = {}
        for key, value in rows:
            if key in out: raise ValueError('duplicate JSON key')
            out[key] = value
        return out
    def nonfinite(value): raise ValueError('nonfinite JSON number: ' + value)
    value = json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=nonfinite)
    todo = [value]; count = chars = 0
    while todo:
        item = todo.pop(); count += 1
        if count > entries: raise ValueError('JSON entry quota exceeded')
        if type(item) is dict:
            todo.extend(item.values()); todo.extend(item.keys())
        elif type(item) is list: todo.extend(item)
        elif type(item) is str:
            chars += len(item)
            if chars > text: raise ValueError('JSON text quota exceeded')
        elif type(item) is float:
            import math
            if not math.isfinite(item): raise ValueError('nonfinite numeric conversion')
        elif item is not None and type(item) not in (int, bool): raise ValueError('unsupported JSON leaf')
    return value


def read_pin(path, maximum):
    path = Path(path)
    st = path.lstat()
    if (not stat.S_ISREG(st.st_mode) or path.is_symlink() or path.resolve() != path or st.st_size > maximum):
        raise ValueError('noncanonical or oversized regular input: ' + str(path))
    def stamp(s): return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    with path.open('rb') as stream:
        data = stream.read(maximum + 1); held = os.fstat(stream.fileno())
    if len(data) > maximum or stamp(st) != stamp(held) or stamp(st) != stamp(path.lstat()):
        raise ValueError('input changed during bounded read: ' + str(path))
    return data, {'path': str(path), 'size': len(data), 'sha256': sha(data)}


def path_value(value, maximum, *, absolute):
    if type(value) is not str or not 0 < len(value) <= maximum or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError('invalid bounded path')
    p = Path(value)
    if p.is_absolute() is not absolute or '..' in p.parts or str(p) != value:
        raise ValueError('path must be canonical and have the declared absolute/relative domain')
    return p


def policy_from(value):
    fields = tuple(f.name for f in dataclasses.fields(CaseCPUProcessPolicy))
    closed(value, fields, 'process_policy')
    normalized = dict(value); normalized['wall_s'] = number(value['wall_s'], 'wall_s', positive=True)
    return CaseCPUProcessPolicy(**normalized)


def request_from(data):
    value = strict_json(data, maximum=1 << 20, entries=131072, text=1 << 20)
    closed(value, ('schema', 'case', 'schedule', 'water_profile', 'input_limits', 'bed_limits',
                   'water_copy_limits', 'runner_limits', 'process_policy'), 'case request')
    if value['schema'] != RUN_SCHEMA: raise ValueError('unsupported case request schema')
    p = policy_from(value['process_policy'])
    if len(data) > p.max_request_bytes: raise ValueError('request exceeds declared policy')
    # Apply declared graph/text domains before touching any Native input.
    strict_json(data, maximum=p.max_request_bytes, entries=p.max_input_entries, text=p.max_request_bytes)
    closed(value['case'], ('directory', 'authored_source_sha256', 'copy_mode'), 'case')
    c = value['case']; path_value(c['directory'], p.max_path_chars, absolute=True)
    if c['copy_mode'] != 'authored_source_then_compile': raise ValueError('unsupported case copy mode')
    if type(c['authored_source_sha256']) is not str or len(c['authored_source_sha256']) != 64 or any(
            ch not in '0123456789abcdef' for ch in c['authored_source_sha256']): raise ValueError('invalid authored source SHA256')
    from masy.case_workflow import CaseInputLimits
    from masy.dense_bed import DenseBedLimits
    from masy.case_water import CaseWaterCopyLimits
    from masy.dense_runner import DenseRunnerLimits
    limits = {}
    for name, cls in (('input_limits', CaseInputLimits), ('bed_limits', DenseBedLimits),
                      ('water_copy_limits', CaseWaterCopyLimits), ('runner_limits', DenseRunnerLimits)):
        closed(value[name], tuple(f.name for f in dataclasses.fields(cls)), name)
        limits[name] = cls(**value[name]); limits[name].__post_init__()
    closed(value['schedule'], ('schema', 'events'), 'schedule')
    if value['schedule']['schema'] != 'masy.case_schedule.v1' or type(value['schedule']['events']) is not list:
        raise ValueError('unsupported schedule')
    events = value['schedule']['events']
    if not 0 < len(events) <= limits['runner_limits'].max_events: raise ValueError('event quota exceeded')
    for event in events:
        if type(event) is not dict: raise ValueError('event must be a closed object')
        if event.get('kind') == 'wind':
            closed(event, ('kind', 'first_block', 'stop_block'), 'wind group')
            integer(event['first_block'], 'first block', minimum=0); integer(event['stop_block'], 'stop block')
        elif event.get('kind') == 'rain':
            closed(event, ('kind', 'water_profile_id', 'intervals', 'recession_guard_s'), 'rain group')
            number(event['recession_guard_s'], 'recession guard', positive=True)
            if type(event['intervals']) is not list or not event['intervals']: raise ValueError('rain intervals required')
            for interval in event['intervals']:
                closed(interval, ('duration_s', 'intensity_mm_h'), 'rain interval')
                number(interval['duration_s'], 'rain duration', positive=True); number(interval['intensity_mm_h'], 'rain intensity')
        else: raise ValueError('unsupported event kind/calendar')
    closed(value['water_profile'], ('spec', 'control'), 'water_profile')
    spec, control = value['water_profile']['spec'], value['water_profile']['control']
    closed(spec, _SPEC_FIELDS, 'water spec')
    closed(control, ('sediment_control', 'completion_policy', 'report_every_s'), 'water control')
    closed(control['sediment_control'], _SEDIMENT, 'sediment control')
    closed(control['sediment_control']['storm'], _STORM, 'storm control')
    closed(control['completion_policy'], _COMPLETION, 'completion policy')
    for key in _ARRAYS:
        descriptor = spec[key]
        if descriptor is None and key in _NULL_ARRAYS: continue
        closed(descriptor, ('kind', 'path', 'size', 'sha256', 'dtype', 'shape'), key)
        if descriptor['kind'] != 'npy' or descriptor['dtype'] != ('bool' if key in _BOOL_ARRAYS else 'float64'):
            raise ValueError('first reader lane requires explicit bool/float64 NPY arrays')
        path_value(descriptor['path'], p.max_path_chars, absolute=False)
        integer(descriptor['size'], key + ' file bytes')
        if descriptor['size'] > p.max_water_input_file_bytes: raise ValueError('water input file quota exceeded')
        if type(descriptor['sha256']) is not str or len(descriptor['sha256']) != 64 or any(ch not in '0123456789abcdef' for ch in descriptor['sha256']):
            raise ValueError('invalid water input SHA256')
        if type(descriptor['shape']) is not list or len(descriptor['shape']) != 2:
            raise ValueError('water input requires an explicit two-dimensional shape')
        for dim in descriptor['shape']: integer(dim, 'array dimension')
    return value, p, limits


def tree_census(root, *, maximum_entries, maximum_bytes):
    root = Path(root); todo = [root]; count = total = 0
    while todo:
        folder = todo.pop()
        if folder.is_symlink() or folder.resolve() != folder: raise ValueError('run symlink/directory changed')
        with os.scandir(folder) as entries:
            for entry in entries:
                count += 1
                if count > maximum_entries: raise ValueError('run entry quota exceeded')
                if entry.is_symlink(): raise ValueError('run symlinks refused')
                if entry.is_dir(follow_symlinks=False): todo.append(folder / entry.name)
                elif entry.is_file(follow_symlinks=False): total += entry.stat(follow_symlinks=False).st_size
                else: raise ValueError('nonordinary run file refused')
                if total > maximum_bytes: raise ValueError('whole-run phase census exceeded declared quota')
    return {'entries': count, 'bytes': total}


def reserve_disk(root, extra, policy):
    integer(extra, 'prospective disk bytes', minimum=0)
    census = tree_census(root, maximum_entries=policy.max_input_entries, maximum_bytes=policy.max_run_bytes)
    if extra > policy.max_run_bytes - census['bytes'] or extra > shutil.disk_usage(root).free:
        raise ValueError('prospective output/copy overlap cannot fit whole-run/free-filesystem quota')
    return {**census, 'reserved_extra_bytes': extra, 'scope': 'declared Source-derived phase reservation; no filesystem aggregate hard limit'}


def inspect_case(directory, limits):
    from masy.case_workflow import _inventory
    root = Path(directory).resolve(strict=True)
    rows = _inventory(root, limits, compiled=False)
    return {'schema': 'masy.case_authored_inventory.v1', 'directory': str(root), 'files': rows,
            'authored_source_sha256': sha(canonical(rows))}


def _sources_current(rt, lock, limits):
    rows = {}
    for kind, source in lock['sources'].items():
        got = rt.observe_source(kind, source['root'], expected_revision=source['head'], limits=limits)
        if got['head'] != source['head'] or got['tree'] != source['tree']:
            raise ValueError('current committed source no longer matches selected runtime: ' + kind)
        rows[kind] = got
    return rows



_CPU_EXTENSION_OWNERS = ('jaxlib_mlir_hlo', 'numpy_fft', 'stdlib_readline', 'stdlib_resource', 'stdlib_mmap')


def cpu_extensions_from(data, *, lock_id, source_heads, packages, interpreter_identity, limits):
    """Explicit Root-issued code identities; no provider import or model claim."""
    value = strict_json(data, maximum=1 << 20, entries=8192, text=1 << 20)
    closed(value, ('schema', 'lock_id', 'source_heads', 'packages', 'interpreter_identity',
                   'extensions', 'qualification_origin'), 'CPU code identity contract')
    if (value['schema'] != 'masy.case_cpu_extensions.v1' or value['lock_id'] != lock_id
            or value['source_heads'] != source_heads or value['packages'] != packages
            or value['interpreter_identity'] != interpreter_identity):
        raise ValueError('CPU code qualification does not bind this exact lock/interpreter/source triple')
    if (type(value['qualification_origin']) is not str or not 0 < len(value['qualification_origin']) <= 4096
            or type(value['extensions']) is not list or len(value['extensions']) != 5):
        raise ValueError('explicit qualified five-owner contract required')
    owners = []; paths = set(); total = 0
    if len(value['extensions']) > limits['max_compiled_files']: raise ValueError('compiled count quota')
    for row in value['extensions']:
        closed(row, ('owner', 'path', 'size', 'sha256'), 'qualified CPU extension')
        if row['owner'] not in _CPU_EXTENSION_OWNERS: raise ValueError('unknown CPU extension owner')
        path_value(row['path'], 4096, absolute=True)
        integer(row['size'], 'extension bytes')
        if row['path'] in paths: raise ValueError('duplicate qualified extension path')
        if type(row['sha256']) is not str or len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256']):
            raise ValueError('invalid extension SHA256')
        total += row['size']
        if total > limits['max_compiled_bytes']: raise ValueError('qualified extension byte quota')
        _, pin = read_pin(Path(row['path']), row['size'])
        if (pin['size'], pin['sha256']) != (row['size'], row['sha256']):
            raise ValueError('qualified extension bytes changed')
        paths.add(row['path']); owners.append(row['owner'])
    if sorted(owners) != sorted(_CPU_EXTENSION_OWNERS): raise ValueError('exact five CPU extension owners required')
    return value


def run_parent(args, launch):
    from masy import runtime as rt
    lock, lock_path = rt.load_lock_ref(args.store, args.lock, 4 << 20)
    rt.validate_lock(lock, store=args.store, lock_name=lock_path.name)
    limits = rt.RuntimeLimits.from_mapping(lock['limits'], 'lock.limits')
    raw, request_pin = read_pin(Path(args.request).resolve(strict=True), 1 << 20)
    request, policy, _ = request_from(raw)
    if policy.max_output_bytes > limits.worker_max_output_bytes or policy.wall_s > limits.worker_timeout_s:
        raise ValueError('case output/wall quota exceeds the explicitly selected original runtime ceiling')
    if launch['root'] != lock['sources']['masy']['root']:
        raise ValueError('case launcher must come from the selected current MASY repository')
    before = _sources_current(rt, lock, limits)
    # A real separate script pin, never passed to observe_launcher() as the old script.
    rel = 'tools/masy_case.py'
    from masy.sourceobs import run_git
    head_bytes = run_git(launch['root'], ['show', lock['sources']['masy']['head'] + ':' + rel], limits.source_limits())
    if (sha(head_bytes) != launch['sha256'] or len(head_bytes) != launch['size']
            or read_pin(Path(launch['path']), 1 << 20)[1]['sha256'] != launch['sha256']):
        raise ValueError('new case launcher differs from its own committed Source blob')
    for kind, obj in lock['objects']['sources'].items():
        rt.verify_source_object(obj['path'], obj['object_id'], limits)
    run = rt._Run(args.store, args.run_id, 'case_cpu').create()
    # Original launcher and engine are actual owners of the runtime verify.
    old_launcher = Path(launch['root']) / 'tools' / 'masy_runtime.py'
    run.stage('runtime_verify', 'started', launcher=str(old_launcher))
    try:
        verify = rt.run_bounded([lock['runtime']['python']['spelling'], '-I', '-B', str(old_launcher), 'verify',
            '--store', args.store, '--lock', str(lock_path)], env=dict(os.environ), timeout_s=limits.worker_timeout_s,
            max_output_bytes=limits.worker_max_output_bytes)
    except rt.RuntimeRefusal as error:
        for name, data in zip(('runtime_verify.stdout', 'runtime_verify.stderr'), getattr(error, 'partial', (b'', b'')), strict=True):
            run.write(name, data)
        run.stage('runtime_verify', 'failed', code=error.code, evidence=error.evidence)
        raise
    run.write('runtime_verify.stdout', verify.stdout); run.write('runtime_verify.stderr', verify.stderr)
    run.stage('runtime_verify', 'complete' if verify.returncode == 0 else 'failed', returncode=verify.returncode, pid=verify.pid, elapsed_s=verify.elapsed_s)
    if verify.returncode != 0: raise ValueError('original runtime launcher verify refused; raw evidence retained')
    nested = {'schema': rt._worker.REQUEST_SCHEMA, 'operation': 'runtime', **rt._runtime_request(lock, run, limits)}
    extension_raw, extension_pin = read_pin(Path(args.cpu_extensions).resolve(strict=True), 1 << 20)
    source_heads = {kind: source['head'] for kind, source in lock['sources'].items()}
    extensions = cpu_extensions_from(extension_raw, lock_id=lock['lock_id'], source_heads=source_heads,
        packages=nested['packages'], interpreter_identity=lock['verification']['interpreter_identity'], limits=nested['limits'])
    outer = {'schema': OUTER_SCHEMA, 'runtime': nested, 'expected_stable': lock['verification'],
             'case_request': request_pin, 'process_policy': request['process_policy'], 'launcher': launch,
             'lock_id': lock['lock_id'], 'source_heads': source_heads,
             'cpu_extensions': {'input_pin': extension_pin, 'contract': extensions}}
    request_path = run.write('case_worker_request.json', canonical(outer))
    if request_path.stat().st_size > policy.max_request_bytes: raise ValueError('closed outer worker request exceeds quota')
    worker_source = Path(lock['objects']['sources']['masy']['path']) / 'source/src/masy/case_worker.py'
    tree = ast.parse(worker_source.read_bytes())
    matches = [n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BOOTSTRAP' for t in n.targets)]
    if len(matches) != 1 or not isinstance(matches[0], ast.Constant) or type(matches[0].value) is not str:
        raise ValueError('sealed cold case bootstrap is not one literal')
    argv = rt.worker_command(lock['runtime']['python']['spelling'], matches[0].value, run, request_path)
    run.stage('case_worker', 'started', argv=argv, declared_policy=policy.as_record())
    try:
        child = rt.run_bounded(argv, env=rt.worker_env(run.dir), timeout_s=policy.wall_s,
                               max_output_bytes=policy.max_output_bytes)
    except rt.RuntimeRefusal as error:
        for name, data in zip(('case.stdout', 'case.stderr'), getattr(error, 'partial', (b'', b'')), strict=True):
            run.write(name, data)
        run.stage('case_worker', 'failed', code=error.code, evidence=error.evidence)
        raise
    run.write('case.stdout', child.stdout); run.write('case.stderr', child.stderr)
    try:
        if child.returncode != 0:
            raise ValueError('case worker nonzero/incomplete; original raw evidence retained')
        if read_pin(Path(extension_pin['path']), extension_pin['size'])[1] != extension_pin:
            raise ValueError('CPU code qualification input changed during execution')
        after = _sources_current(rt, lock, limits)
        if before != after: raise ValueError('source changed during case process')
        result = strict_json(child.stdout, maximum=policy.max_output_bytes, entries=policy.max_input_entries,
                             text=policy.max_output_bytes)
        if (type(result) is not dict or result.get('schema') != RESULT_SCHEMA
                or result.get('status') != 'complete' or result.get('lock_id') != lock['lock_id']):
            raise ValueError('case worker incomplete/malformed; original raw evidence retained')
    except Exception as error:
        run.stage('case_worker', 'failed', returncode=child.returncode, pid=child.pid, elapsed_s=child.elapsed_s,
                  reason=type(error).__name__ + ': ' + str(error)[:2000])
        raise
    run.stage('case_worker', 'complete', pid=child.pid, elapsed_s=child.elapsed_s)
    return {'schema': RESULT_SCHEMA, 'status': 'complete', 'run_dir': str(run.dir), 'lock_id': lock['lock_id'],
            'case_report': str(run.dir / 'case_report.json'), 'worker_pid': child.pid, 'elapsed_s': child.elapsed_s,
            'qualification': 'this execution only; no combined restart/large-domain/GPU/calibration claim'}


def main(argv=None, *, launch=None):
    parser = argparse.ArgumentParser(prog='masy_case', allow_abbrev=False)
    sub = parser.add_subparsers(dest='command', required=True)
    inspect_p = sub.add_parser('inspect', allow_abbrev=False)
    inspect_p.add_argument('--case', required=True); inspect_p.add_argument('--input-limits', required=True)
    run_p = sub.add_parser('run', allow_abbrev=False)
    for name in ('store', 'lock', 'request', 'run-id', 'cpu-extensions'): run_p.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'inspect':
            from masy.case_workflow import CaseInputLimits
            data, _ = read_pin(Path(args.input_limits).resolve(strict=True), 1 << 20)
            raw = strict_json(data, maximum=1 << 20, entries=131072, text=1 << 20)
            closed(raw, tuple(f.name for f in dataclasses.fields(CaseInputLimits)), 'input limits')
            result = inspect_case(args.case, CaseInputLimits(**raw))
        else:
            if launch is None: raise ValueError('run requires the real pinned tools/masy_case.py entry')
            result = run_parent(args, launch)
        sys.stdout.buffer.write(canonical(result)); return 0
    except Exception as error:
        sys.stderr.write(type(error).__name__ + ': ' + str(error)[:4000] + '\n'); return 2
