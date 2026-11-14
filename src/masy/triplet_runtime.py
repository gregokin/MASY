"""Cold CPU verification/bootstrap seam for the opt-in four-object lock.

``bootstrap_spec`` returns a command/environment, never starts it. Root applies
its existing bounded supervisor. The target is a closed qualified MASY entry
point, not an arbitrary module, script, pickle or evaluated configuration.
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import sysconfig
from types import ModuleType

from masy import triplet_lock as tl

BOOTSTRAP = r'''
import hashlib, json, os, pathlib, stat, sys, sysconfig
families = ('maple','maple_syrup','leaf','masy','benchmarks','numpy','numba','scipy','jax','cupy','cupy_backends')
if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
    raise RuntimeError('triplet requires -I -B')
if any(n.partition('.')[0] in families for n in sys.modules):
    raise RuntimeError('triplet refuses preloaded native/provider/helper modules')
p=pathlib.Path(sys.argv[1])
if not p.is_absolute() or p.resolve()!=p:
    raise RuntimeError('noncanonical lock path')
st=p.lstat()
if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_size>8388608:
    raise RuntimeError('lock file domain')
fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
with os.fdopen(fd,'rb') as f:
    raw=f.read(8388609)
if len(raw)>8388608 or hashlib.sha256(raw).hexdigest()!=sys.argv[2]:
    raise RuntimeError('external lock SHA differs')
lock=json.loads(raw)
cache=pathlib.Path(sys.argv[5])
if not cache.is_absolute() or cache.resolve()!=cache or sys.pycache_prefix!=str(cache) or not cache.is_dir() or any(cache.iterdir()):
    raise RuntimeError('fresh empty run-owned -X pycache_prefix required')
source=pathlib.Path(lock['objects']['masy']['path'])/'source'
if source.resolve()!=source:
    raise RuntimeError('noncanonical MASY object')
rows=lock['qualification']['sources']['masy']['files']
expected={r[0]:(r[1],r[2]) for r in rows}
seen=set()
for parent,dirs,files in os.walk(source,followlinks=False):
    for name in dirs:
        d=pathlib.Path(parent)/name
        if d.is_symlink() or d.stat().st_mode & 0o222:
            raise RuntimeError('mutable/symlink MASY source directory')
    for name in files:
        f=pathlib.Path(parent)/name
        rel=f.relative_to(source).as_posix()
        st=f.lstat()
        if rel not in expected or not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_mode & 0o222 or st.st_size!=expected[rel][0]:
            raise RuntimeError('MASY source file set differs')
        h=hashlib.sha256()
        with open(f,'rb') as stream:
            while True:
                chunk=stream.read(1048576)
                if not chunk: break
                h.update(chunk)
        if h.hexdigest()!=expected[rel][1]:
            raise RuntimeError('MASY source bytes differ')
        seen.add(rel)
if seen!=set(expected):
    raise RuntimeError('MASY source files missing')
# Only independently hashed MASY is visible until its stdlib verifier closes
# all other objects. Native/metadata roots cannot shadow standard-library code.
stdlib={sysconfig.get_path('stdlib'),sysconfig.get_path('platstdlib'),
        os.path.join(sys.base_exec_prefix,'lib','python'+sysconfig.get_python_version(),'lib-dynload'),
        os.path.join(sys.base_prefix,'lib','python'+str(sys.version_info.major)+str(sys.version_info.minor)+'.zip')}
first=[x for x in sys.path if x in stdlib]
tail=[x for x in sys.path if x not in stdlib]
if not first or not all(type(x) is str for x in tail):
    raise RuntimeError('unexpected interpreter path')
sys.path[:]=first+[str(source/'src')]+tail
from masy.triplet_runtime import worker_main
sys.exit(worker_main(sys.argv[1:]))
'''


def _standard_paths(path):
    roots = {sysconfig.get_path('stdlib'), sysconfig.get_path('platstdlib'),
             os.path.join(sys.base_exec_prefix, 'lib', 'python' + sysconfig.get_python_version(), 'lib-dynload'),
             os.path.join(sys.base_prefix, 'lib', 'python' + str(sys.version_info.major) + str(sys.version_info.minor) + '.zip')}
    return [p for p in path if p in roots]


def install_prefix(prefix):
    'After tl.verify only: sealed roots follow stdlib and precede site-packages.'
    standard = _standard_paths(sys.path)
    tl._need(standard and all(type(p) is str for p in sys.path), 'import_prefix', 'plain interpreter path required')
    tail = [p for p in sys.path if p not in standard and p not in prefix]
    sys.path[:] = standard + list(prefix) + tail


def _roots(lock):
    roots = {tl.PACKAGE[n]: Path(lock["objects"][n]["path"]) / "source" /
             ("leaf" if n == "leaf" else "src/" + tl.PACKAGE[n]) for n in tl.MODELS}
    roots["maple"] = Path(lock["qualification"]["maple_bundle"]["target"]) / "source/src/maple"
    return roots


def helper_record(lock):
    row = next(r for r in lock["qualification"]["sources"]["leaf"]["files"]
               if r[0] == "benchmarks/spatial_mapping.py")
    return {"module": "benchmarks.spatial_mapping", "sha256": row[2], "bytes": row[1]}


def _namespace(lock, path):
    expected = Path(lock["objects"]["leaf"]["path"]) / "source/benchmarks"
    # The LEAF package digest alone omits this namespace/helper. No unrelated
    # benchmarks root, concrete package or alternate helper may participate.
    candidates = []
    for entry in path:
        if type(entry) is not str:
            raise tl.TripletLockError("helper_namespace", "plain import path required")
        p = Path(entry or os.getcwd()) / "benchmarks"
        if os.path.lexists(p):
            tl._path(str(p), directory=True)
            candidates.append(p)
    tl._need(candidates == [expected], "helper_namespace", "ambiguous or missing benchmarks namespace")
    spec = importlib.machinery.PathFinder.find_spec("benchmarks", path)
    tl._need(spec is not None and spec.origin is None and list(spec.submodule_search_locations) == [str(expected)],
             "helper_namespace", "namespace must resolve only to sealed LEAF helper directory")


def check_import_sources(lock, modules):
    """Check actual loaded project/helper files without invoking foreign getters."""
    roots, found = _roots(lock), {}
    limits = lock["qualification"]["limits"]
    tables = {}
    for model in tl.MODELS:
        prefix = "leaf/" if model == "leaf" else "src/" + tl.PACKAGE[model] + "/"
        tables[tl.PACKAGE[model]] = {r[0][len(prefix):]: (r[1], r[2])
                                    for r in lock["qualification"]["sources"][model]["files"] if r[0].startswith(prefix)}
    expected_helper = Path(lock["objects"]["leaf"]["path"]) / "source/benchmarks/spatial_mapping.py"
    tl._need(len(modules) <= 100000, "module_limit", "loaded module census")
    for name, module in modules.items():
        if type(name) is not str:
            raise tl.TripletLockError("module_origin", "plain module names required")
        family = name.partition(".")[0]
        tl._need(family not in ("cupy", "cupy_backends"), "backend", "CPU profile refuses CuPy")
        if family not in tl.FAMILIES and family != "benchmarks":
            continue
        if module is None:
            continue
        tl._need(type(module) is ModuleType, "module_origin", name)
        namespace = vars(module)
        if name == "benchmarks":
            expected = str(expected_helper.parent)
            tl._need(namespace.get("__file__") is None and list(namespace.get("__path__", ())) == [expected], "helper_namespace", name)
            continue
        if family == "benchmarks":
            tl._need(name == "benchmarks.spatial_mapping", "helper_namespace", name)
            expected, info = expected_helper, helper_record(lock)
            size, sha = info["bytes"], info["sha256"]
        else:
            origin = namespace.get("__file__")
            tl._need(type(origin) is str, "module_origin", name)
            p = Path(origin)
            tl._need(p.is_relative_to(roots[family]), "module_origin", name)
            rel = p.relative_to(roots[family]).as_posix()
            tl._need(rel in tables[family], "module_origin", name + ": foreign/unlocked source")
            expected, (size, sha) = roots[family] / rel, tables[family][rel]
        spec = namespace.get("__spec__")
        tl._need(namespace.get("__file__") == str(expected) and spec is not None and spec.origin == str(expected), "module_origin", name)
        _, n, actual = tl._read(expected, limits["max_file_bytes"], keep=False)
        tl._need((n, actual) == (size, sha), "module_source", name)
        found[name] = {"path": str(expected), "size": n, "sha256": actual}
    return found


def _distributions(lock):
    out = {}
    prefix = tl.import_prefix(lock)
    for model in tl.MODELS:
        distribution = importlib.metadata.distribution(tl.PROJECT[model])
        parent = (Path(lock["qualification"]["maple_bundle"]["target"]) / "editable" if model == "maple"
                  else Path(lock["objects"][model]["path"]) / "metadata")
        actual = Path(distribution._path)
        tl._need(actual.parent == parent and parent.as_posix() in prefix, "distribution_origin", model)
        out[tl.PROJECT[model]] = {"path": str(actual), "version": distribution.version,
                                "kind": "genuine_MAPLE_bundle" if model == "maple" else "labelled_metadata_overlay"}
    return out


def verify_imports(lock):
    """Call genuine owning bundle/profile checks and census the actual imports."""
    tl._need(sys.flags.isolated and sys.flags.dont_write_bytecode, "interpreter", "-I -B required")
    q = lock["qualification"]
    tl._need(list(sys.version_info[:3]) == q["interpreter"]["version"]
             and sys.executable == q["interpreter"]["path"]
             and Path(sys.prefix) == Path(tl.PYTHON).parent.parent, "interpreter", "selected interpreter differs")
    tl._need(all(os.environ.get(k) == v for k, v in {"NUMBA_DISABLE_CUDA": "1", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "GIT_OPTIONAL_LOCKS": "0"}.items())
             and "PYTHONPATH" not in os.environ, "environment", "controlled CPU environment required")
    prefix = tl.import_prefix(lock)
    offset = len(_standard_paths(sys.path))
    tl._need(sys.path[:offset] == _standard_paths(sys.path) and sys.path[offset:offset + len(prefix)] == prefix,
             "import_prefix", "stdlib then exact sealed roots required")
    check_import_sources(lock, sys.modules)
    _namespace(lock, sys.path)
    bundle = importlib.import_module("maple_syrup.dependency_bundle")
    actual_bundle = bundle.check_runtime(q["maple_bundle"]["target"])
    tl._need(actual_bundle["package_source_digest"] == q["sources"]["maple"]["package_sha256"], "bundle", "native runtime differs")
    for name in tl.FAMILIES:
        importlib.import_module(name)
    importlib.import_module("benchmarks.spatial_mapping")
    ncp = importlib.import_module("maple_syrup.native_case_profile")
    source = ncp.verify_source_profile(q["maple_bundle"]["target"], expected_syrup_digest=q["sources"]["syrup"]["package_sha256"], profile=q["native_profile"])
    source.recheck()
    # Verify with SYRUP's owning digest implementation as well as committed rows.
    provenance = importlib.import_module("maple_syrup.provenance")
    roots = _roots(lock)
    for name, root in roots.items():
        model = next(n for n in tl.MODELS if tl.PACKAGE[n] == name)
        actual = provenance.source_tree_digest(root, max_files=q["limits"]["max_files"], max_bytes=q["limits"]["max_source_bytes"])
        tl._need(actual.digest_sha256 == q["sources"][model]["package_sha256"], "package_digest", name)
    origins = check_import_sources(lock, sys.modules)
    return source, {"schema": "masy.triplet_imports.v1", "profile": tl.PROFILE, "origins": origins,
                    "distributions": _distributions(lock), "external_helpers": [helper_record(lock)],
                    "bundle": actual_bundle, "python": sys.version, "backend": "numpy",
                    "native_compatibility_checked": True, "model_executed": False}


@dataclasses.dataclass(frozen=True)
class RuntimeContext:
    """A verified source capability supplied only by the cold worker.

    It contains no plant, water, landscape or physics configuration. The owning
    daily adapter must join its LEAF binding helper pins to external_helpers.
    """
    lock_path: str
    lock_sha256: str
    source_profile: object

    def record(self):
        lock = tl.load(self.lock_path, expected_sha256=self.lock_sha256)
        return {"profile": tl.PROFILE, "lock_sha256": self.lock_sha256,
                "sources": {n: {k: lock["qualification"]["sources"][n][k] for k in ("commit", "tree", "package_sha256")}
                            for n in tl.MODELS}, "external_helpers": [helper_record(lock)]}

    def recheck(self):
        # No Git subprocess during transport: code files and owning source
        # capability are rechecked; the parent checks branch currency at launch.
        lock = tl.verify(self.lock_path, expected_sha256=self.lock_sha256, current_refs=False)
        self.source_profile.recheck()
        return check_import_sources(lock, sys.modules)


def bootstrap_spec(lock_path, *, expected_sha256, run_dir, entrypoint=None, argv=()):
    """Return argv/env for Root's bounded supervisor; never spawn a model.

    Entry points take ``main(argv, *, runtime=RuntimeContext)``. With None this
    only performs native/source metadata verification, never event execution.
    """
    lock = tl.verify(lock_path, expected_sha256=expected_sha256)
    tl._path(run_dir, directory=True)
    mode = "verify" if entrypoint is None else entrypoint
    tl._need(entrypoint is None or entrypoint in lock["qualification"]["entrypoints"], "entrypoint", "not in externally qualified roster")
    tl._need(type(argv) in (list, tuple) and len(argv) <= 256 and all(type(a) is str and "\x00" not in a for a in argv), "arguments", "bounded plain arguments")
    encoded = tl.canonical(list(argv)).decode("ascii")
    tl._need(len(encoded) <= 1 << 20, "arguments", "argument bytes")
    base = Path(run_dir)
    cache = base / "pycache"
    tl._need(not os.path.lexists(cache), "pycache", "fresh pycache path required; no repair/reuse")
    cache.mkdir(mode=0o700)
    env = {"PATH": str(Path(tl.PYTHON).parent) + ":/usr/bin:/bin", "LC_ALL": "C.UTF-8",
           "HOME": str(base / "home"), "TMPDIR": str(base / "tmp"), "XDG_CACHE_HOME": str(base / "cache"),
           "NUMBA_CACHE_DIR": str(base / "cache/numba"), "MPLCONFIGDIR": str(base / "cache/matplotlib"),
           "NUMBA_DISABLE_CUDA": "1", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
           "GIT_OPTIONAL_LOCKS": "0"}
    return {"argv": [tl.PYTHON, "-I", "-B", "-X", "pycache_prefix=" + str(cache), "-c", BOOTSTRAP, str(lock_path), expected_sha256, mode, encoded, str(cache)],
            "env": env, "scope": "Root must impose actual process/resource bounds; no process has been launched"}


def worker_main(argv):
    path, sha, mode, args_json, cache = argv
    tl._need(sys.pycache_prefix == cache and not any(tl._path(cache, directory=True).iterdir()), "pycache", "fresh empty redirected bytecode path required")
    lock = tl.verify(path, expected_sha256=sha, current_refs=True)
    install_prefix(tl.import_prefix(lock))
    source, receipt = verify_imports(lock)
    if mode == "verify":
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
        return 0
    tl._need(mode in lock["qualification"]["entrypoints"], "entrypoint", "unknown run target")
    module, name = mode.split(":")
    target = getattr(importlib.import_module(module), name)
    check_import_sources(lock, sys.modules)
    runtime = RuntimeContext(path, sha, source)
    result = target(tl._json(args_json), runtime=runtime)
    runtime.recheck()
    tl._need(result is None or type(result) is int, "entrypoint", "CLI entry point must return int or None")
    return 0 if result is None else result
