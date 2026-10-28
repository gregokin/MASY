#!/usr/bin/env python3
"""Public cold imported-case GPU launcher. No import/transport on --help."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time

def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def _read(path, maximum, *, immutable=False):
    path = Path(path); st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or path.is_symlink() or path.resolve() != path or st.st_size > maximum:
        raise ValueError('noncanonical or oversized bootstrap Source/lock')
    if immutable and st.st_mode & 0o222: raise ValueError('writable immutable Source file')
    with path.open('rb') as stream:
        data = stream.read(maximum + 1); held = os.fstat(stream.fileno())
    def stamp(s): return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if len(data) > maximum or stamp(st) != stamp(held) or stamp(st) != stamp(path.lstat()):
        raise ValueError('bootstrap Source/lock changed during read')
    return data


def _pairs(rows):
    out = {}
    for key, value in rows:
        if key in out: raise ValueError('duplicate bootstrap JSON key')
        out[key] = value
    return out


def _sealed_prefix(store, ref):
    store = Path(store)
    if not store.is_absolute() or store.resolve() != store: raise ValueError('canonical absolute store required')
    if type(ref) is not str or len(ref)!=64 or any(c not in '0123456789abcdef' for c in ref): raise ValueError('explicit profile ID required')
    profile_path = store / 'profiles' / (ref + '.json')
    raw = _read(profile_path, 32 << 20, immutable=True)
    lock = json.loads(raw, object_pairs_hook=_pairs)
    body = dict(lock); identity = body.pop('profile_id')
    if identity != ref or hashlib.sha256(_canonical(body)).hexdigest() != ref or lock['store'] != str(store):
        raise ValueError('profile identity/path mismatch')
    obj = lock['objects']['masy']; ident = obj['object_id']
    if type(ident) is not str or len(ident) != 64 or any(ch not in '0123456789abcdef' for ch in ident):
        raise ValueError('malformed MASY Source object identity')
    root = store / 'objects' / 'sources' / ident
    if str(root) != obj['path']: raise ValueError('MASY Source object path not derived from lock')
    manifest_raw = _read(root / 'manifest.json', 4 << 20, immutable=True)
    if hashlib.sha256(manifest_raw).hexdigest() != obj['manifest_sha256']: raise ValueError('Source manifest pin mismatch')
    manifest = json.loads(manifest_raw, object_pairs_hook=_pairs); identity = manifest['identity']
    if (manifest['object_id'] != ident or hashlib.sha256(_canonical(identity)).hexdigest() != ident
            or identity['kind'] != 'masy' or identity['package'] != 'masy'):
        raise ValueError('MASY Source identity mismatch')
    limits = lock['limits']; rows = identity['files']
    if type(rows) is not list or len(rows) > limits['max_files']: raise ValueError('Source file count bound')
    total = 0; names = set()
    for row in rows:
        if type(row) is not list or len(row) != 4: raise ValueError('Source row schema mismatch')
        rel, size, sha, blob = row
        if (type(rel) is not str or Path(rel).is_absolute() or '..' in Path(rel).parts or str(Path(rel)) != rel
                or rel in names or type(size) is not int or size < 0): raise ValueError('invalid Source row')
        names.add(rel); total += size
        if total > limits['max_bytes']: raise ValueError('Source byte budget')
        data = _read(root / 'source' / rel, size, immutable=True)
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha: raise ValueError('Source bytes changed')
        if blob is not None:
            hasher = hashlib.new(identity['object_format']); hasher.update(b'blob %d\0' % size); hasher.update(data)
            if hasher.hexdigest() != blob: raise ValueError('Source Git blob mismatch')
    expected_dirs = {''}
    for name in names:
        parent = Path(name).parent
        while str(parent) != '.':
            expected_dirs.add(str(parent)); parent = parent.parent
    found_files = set(); found_dirs = {''}; todo = [(root / 'source', '')]
    while todo:
        directory, rel = todo.pop(); info = directory.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o222 or directory.resolve() != directory):
            raise ValueError('Source directory is writable, aliased or nonordinary before import')
        with os.scandir(directory) as entries:
            for entry in entries:
                name = (rel + '/' if rel else '') + entry.name
                if entry.is_symlink(): raise ValueError('Source symlink before import')
                if entry.is_dir(follow_symlinks=False):
                    if name not in expected_dirs: raise ValueError('unexpected Source directory before import')
                    found_dirs.add(name); todo.append((directory / entry.name, name))
                elif entry.is_file(follow_symlinks=False):
                    if name not in names: raise ValueError('unexpected Source file/bytecode before import')
                    found_files.add(name)
                else: raise ValueError('nonordinary Source entry before import')
    root_info = root.lstat()
    if root.resolve() != root or not stat.S_ISDIR(root_info.st_mode) or root_info.st_mode & 0o222:
        raise ValueError('immutable Source object root custody refused')
    if found_files != names or found_dirs != expected_dirs: raise ValueError('incomplete Source tree before import')
    required = {'src/masy/'+name+'.py' for name in ('__init__','case_gpu_cli','case_gpu_worker','case_device_c64','windowed_device_c64_bed','windowed_device_c64_runner','gpu_runtime','gpu_runtime_worker','runtime','case_cli','case_worker')}
    if not required <= names: raise ValueError('case modules absent from locked MASY Source')
    # The complete original source-object verifier runs immediately in the
    # authenticated parent, including exact directory/overlay/read-only checks.
    return str(root / 'source' / 'src')

def main(argv):
    started = time.monotonic()
    if not argv or argv[0] in ('-h','--help'):
        print('python -I -B tools/masy_case_gpu.py run --store STORE --profile ID --authority FILE --authority-size N --authority-sha256 SHA --request REQUEST.json --run-id NAME\nPrepare/verify a profile with the existing masy_gpu_runtime.py command. Only synthetic C64/K1/V20/W3 W/R/W/R is qualified by a later Root run; no general capacity or restart claim.')
        return 0
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode and sys.flags.safe_path) or sys.flags.optimize:raise ValueError('use python -I -B without optimization')
    if any(n.partition('.')[0] in ('masy','maple','maple_syrup','numpy','numba','jax','cupy','scipy') for n in sys.modules):raise ValueError('preloaded product/provider')
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=('run',))
    for key in ('store','profile','authority','authority-size','authority-sha256','request','run-id'):p.add_argument('--'+key,required=True)
    a=p.parse_args(argv)
    prior=signal.getitimer(signal.ITIMER_REAL)
    if prior!=(0.0,0.0):raise ValueError('pre-existing caller timer refused without changing it')
    launch=None
    def deadline(_signum,_frame):
        if launch is not None and launch.get('_published_result') is not None:return
        raise TimeoutError('whole600s deadline from launcher entry')
    def interrupted(_signum,_frame):
        if launch is not None and launch.get('_published_result') is not None:return
        raise InterruptedError('public case operation terminated; reap owned child')
    old=signal.signal(signal.SIGALRM,deadline)
    old_term=signal.signal(signal.SIGTERM,interrupted)
    signal.setitimer(signal.ITIMER_REAL,max(0.001,600-(time.monotonic()-started)))
    try:
        prefix=_sealed_prefix(a.store,a.profile)
        import re
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',a.run_id):raise ValueError('unsafe run ID')
        run=Path(a.store)/'runs'/a.run_id
        if run.exists() or run.is_symlink():raise ValueError('operation directory already exists')
        sys.pycache_prefix=str(run/'launcher-pycache')
        here=Path(__file__).resolve();data=_read(here,1<<20)
        launch=dict(path=str(here),root=str(here.parent.parent),size=len(data),sha256=hashlib.sha256(data).hexdigest(),started_monotonic=started)
        sys.path.insert(0,prefix)
        from masy.case_gpu_cli import main as run_main
        return run_main(argv,launch=launch)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old)
        signal.signal(signal.SIGTERM,old_term)


if __name__=='__main__':
    try:raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc)[:2000],file=sys.stderr);raise SystemExit(2)
