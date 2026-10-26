#!/usr/bin/env python3
"""Real case entry. prepare/verify delegate the unchanged runtime launcher.

run authenticates the selected immutable MASY Sources before importing its
sealed case_cli; the new script has its own committed Source pin there.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


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
    lock_path = Path(ref) if Path(ref).is_absolute() else store / 'locks' / (ref + '.json')
    raw = _read(lock_path, 4 << 20); lock = json.loads(raw, object_pairs_hook=_pairs)
    body = dict(lock); digest = body.pop('lock_id')
    if (hashlib.sha256(_canonical(body)).hexdigest() != digest or lock_path != store / 'locks' / (digest + '.json')
            or lock['store'] != str(store)):
        raise ValueError('bootstrap lock self identity/path mismatch')
    obj = lock['objects']['sources']['masy']; ident = obj['object_id']
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
    required = {'src/masy/__init__.py' , 'src/masy/case_cli.py', 'src/masy/case_worker.py', 'src/masy/runtime.py'}
    if not required <= names: raise ValueError('case modules absent from locked MASY Source')
    # The complete original source-object verifier runs immediately in the
    # authenticated parent, including exact directory/overlay/read-only checks.
    return str(root / 'source' / 'src')


def main(argv):
    root = Path(__file__).resolve().parent.parent
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode): raise ValueError('use python -I -B')
    forbidden = ('masy', 'maple', 'maple_syrup', 'numpy', 'numba', 'jax', 'cupy', 'scipy')
    if any(name.partition('.')[0] in forbidden for name in sys.modules): raise ValueError('preloaded package/provider')
    if argv and argv[0] in ('prepare', 'verify'):
        # Actual existing launcher path and argv, not a forged launch record.
        os.execv(sys.executable, [sys.executable, '-I', '-B', str(root / 'tools/masy_runtime.py'), *argv])
    if not argv or argv[0] in ('-h', '--help'):
        print('python -I -B tools/masy_case.py prepare --config RUNTIME.toml\n'
              'python -I -B tools/masy_case.py verify --store STORE --lock LOCK\n'
              'python -I -B tools/masy_case.py inspect --case CASE --input-limits LIMITS.json\n'
              'python -I -B tools/masy_case.py run --store STORE --lock LOCK --request RUN.json --cpu-extensions CPU_CODE.json --run-id NAME')
        return 0
    if argv[0] == 'run':
        p = argparse.ArgumentParser(allow_abbrev=False)
        p.add_argument('command', choices=('run',))
        for key in ('store', 'lock', 'request', 'run-id', 'cpu-extensions'): p.add_argument('--' + key, required=True)
        args = p.parse_args(argv)
        prefix = _sealed_prefix(args.store, args.lock)
        import re
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', args.run_id): raise ValueError('unsafe run name')
        fresh = Path(args.store) / 'runs' / args.run_id
        if fresh.exists() or fresh.is_symlink(): raise ValueError('new run directory must be absent before parent import')
        sys.pycache_prefix = str(fresh / 'cache' / 'pycache')
    elif argv[0] == 'inspect':
        prefix = str(root / 'src')  # readonly metadata usability; no qualification/model authority
    else: raise ValueError('unknown case command')
    data = _read(Path(__file__).resolve(), 1 << 20)
    launch = {'path': str(Path(__file__).resolve()), 'root': str(root), 'size': len(data),
              'sha256': hashlib.sha256(data).hexdigest()}
    sys.path.insert(0, prefix)
    from masy.case_cli import main as case_main
    return case_main(argv, launch=launch)


if __name__ == '__main__':
    try: sys.exit(main(sys.argv[1:]))
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error)[:4000], file=sys.stderr); sys.exit(2)
