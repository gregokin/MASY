"""Sealed child-side checks of the MASY locked CPU runtime.

Imported ONLY inside the bootstrap child that `masy.runtime` launches as

    <python> -I -B -X pycache_prefix=<run>/cache/pycache -c BOOTSTRAP <run>/request-<n>-<op>.json

with an environment built from nothing (`masy.runtime.worker_env`). `BOOTSTRAP` (below; its exact bytes are read
from the SEALED MASY object and hashed into the lock) refuses if any `maple`, `maple_syrup` or `masy` module is
already loaded (site `.pth` hooks still run under -I), inserts the request's explicit sealed `sys.path` prefix (-I
ignores PYTHONPATH, so upstream `runtime_environment()` cannot be used as is) and only then imports this module from
the sealed MASY source object. Once imported, this module's own sealed `masy` modules are expected; their origins
are checked like those of `maple` and `maple_syrup`.

Nothing here monkeypatches modules, classes, resolvers or distribution discovery. Every upstream function is called
unmodified from the sealed copies; the native checkpoint registries are class-dictionary operations only (no state,
array or solver step).

Operations (one bounded JSON result line on stdout; exit 0 ok, 2 refused, 3 incompatible, 1 internal error):

- `self-check`: interpreter flags, exact sys.path prefix, environment and sealed MASY origin only.
- `export`: SYRUP's `dependency_bundle.export_bundle` of the sealed MAPLE scope (MAPLE is not imported).
- `materialize`: `verify_bundle`, `materialize_bundle`, `verify_materialized` (MAPLE is not imported).
- `runtime`: `verify_bundle`, `check_runtime` (the unmodified native resolver must select the frozen cache),
  `masy.contracts.run_native_audit`, both native checkpoint registries, module origins of all three package
  families, selected distribution metadata, package digests, environment versions and compiled identities.

Results separate `stable` identity (compared strictly against a lock), `operational` per-run values (validated by
the parent against its own run directory, never compared with an older run) and an informational `receipt`
(shadowed unselected metadata, sys.path tail, platform string).
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
import platform
import site
import stat
import sys
import threading
from types import FunctionType
import traceback
from typing import Any, Mapping

__all__ = [
    "BOOTSTRAP",
    "FAMILIES",
    "OPERATIONAL_ENV_KEYS",
    "OPERATIONS",
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "STABLE_ENV_KEYS",
    "WorkerRefusal",
    "compiled_identities",
    "main",
    "preload_violations",
    "registry_probe",
    "selected_distributions",
]

REQUEST_SCHEMA = "masy.runtime_request.v1"
RESULT_SCHEMA = "masy.runtime_result.v1"
FAMILIES = ("maple", "maple_syrup", "masy")
OPERATIONS = ("self-check", "export", "materialize", "runtime")
STATUS_EXIT = {"ok": 0, "refused": 2, "incompatible": 3}
# Fixed, CPU-only values: part of the stable identity.
STABLE_ENV_KEYS = ("PATH", "LC_ALL", "NUMBA_DISABLE_CUDA", "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS")
# Per-run paths: validated against the owning run directory, never compared across runs.
OPERATIONAL_ENV_KEYS = ("HOME", "TMPDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR", "MPLCONFIGDIR")
# Variables CPython may set itself (PEP 538 locale coercion); reported, not refused.
INTERPRETER_SET_ENV = frozenset({"LC_CTYPE"})
# Versions recorded from installed metadata of the EXISTING environment (identified, not bundled).
ENVIRONMENT_DISTRIBUTIONS = (
    "numpy", "jax", "jaxlib", "ml_dtypes", "opt_einsum", "scipy", "pyyaml", "xarray", "zarr", "dask", "tifffile",
    "packaging", "numba", "llvmlite", "cupy-cuda12x",
)
MAX_MESSAGE = 4000
MAX_METADATA_FILE_BYTES = 1 << 20
MAX_MAPS_BYTES = 8 << 20
_CHUNK = 1 << 20

_COMMON_KEYS = frozenset({"schema", "operation", "python", "run", "sys_path_prefix", "packages", "limits"})
_OPERATION_KEYS = {
    "self-check": frozenset(),
    "export": frozenset({"export", "tool_module_sha256"}),
    "materialize": frozenset({"bundle"}),
    "runtime": frozenset({"bundle", "distributions"}),
}
_LIMIT_KEYS = frozenset({"max_files", "max_bytes", "max_compiled_files", "max_compiled_bytes"})

# Launched with `-c`; reads the request with a fixed bound and imports nothing from the three package families
# before its preload check. Keep it free of f-strings/format braces so its bytes are easy to audit.
BOOTSTRAP = """\
import sys


def _masy_bootstrap():
    import json

    def emit(code, detail, **evidence):
        record = {"schema": "masy.runtime_result.v1", "operation": "bootstrap", "status": "refused",
                  "code": code, "detail": detail, "evidence": evidence}
        sys.stdout.write(json.dumps(record, sort_keys=True) + "\\n")
        sys.stdout.flush()
        return 2

    families = ("maple", "maple_syrup", "masy")
    preloaded = sorted(name for name in list(sys.modules) if name.partition(".")[0] in families)
    if preloaded:
        return emit("worker_preloaded_modules", "package modules were loaded before the sealed worker",
                    modules=preloaded[:50])
    if len(sys.argv) != 2:
        return emit("worker_request_invalid", "expected exactly one request path argument")
    with open(sys.argv[1], "rb") as handle:
        data = handle.read(1048577)
    if len(data) > 1048576:
        return emit("worker_request_invalid", "request is larger than 1048576 bytes")
    try:
        request = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        return emit("worker_request_invalid", "request is not UTF-8 JSON: " + str(exc)[:500])
    prefix = request.get("sys_path_prefix") if isinstance(request, dict) else None
    if not isinstance(prefix, list) or not prefix or \\
            not all(isinstance(entry, str) and entry.startswith("/") for entry in prefix):
        return emit("worker_request_invalid", "sys_path_prefix must be a non-empty list of absolute paths")
    base_sys_path = list(sys.path)
    sys.path[0:0] = prefix
    import masy.runtime_worker as worker
    return worker.main(request, base_sys_path)


sys.exit(_masy_bootstrap())
"""


class WorkerRefusal(Exception):
    """A child-side check failed. `status` is `refused` or `incompatible`; `evidence` is JSON-safe."""

    def __init__(self, code: str, detail: str, *, status: str = "refused", **evidence: Any) -> None:
        super().__init__(f"{code}: {detail}")
        if status not in ("refused", "incompatible"):
            raise ValueError(f"invalid refusal status {status!r}")
        self.code = code
        self.detail = detail[:MAX_MESSAGE]
        self.status = status
        self.evidence = evidence


def _message(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:MAX_MESSAGE]


def _family(name: str) -> str:
    return name.partition(".")[0]


# --------------------------------------------------------------------------------------------- pure helpers


def preload_violations(modules: Mapping[str, Any], families: tuple[str, ...] = FAMILIES) -> list[str]:
    """Loaded module names belonging to any package family (the bootstrap's preload rule, as a testable helper)."""
    return sorted(name for name in modules if _family(name) in families)


_IDENTITY_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return tuple(getattr(info, name) for name in _IDENTITY_FIELDS)


def _sha256_file(path: str, budget: int, *, code: str = "compiled_identity_unreadable",
                 bound_code: str = "compiled_identity_bound_exceeded") -> tuple[str, int]:
    """sha256 and size of one regular file (no symlink follow), reading at most `budget` bytes in chunks.

    The path lstat before open, the descriptor fstat after open and after the last read, and the path lstat after
    the read must carry the same device, inode, mode, size, mtime_ns and ctime_ns: replacement or a same-size
    in-place write during the read refuses with `code` (as does a missing or unreadable path at any point)."""
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise WorkerRefusal(code, f"cannot stat {path}: {_message(exc)}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise WorkerRefusal(code, f"{path} is not a regular file")
    if info.st_size > budget:
        raise WorkerRefusal(bound_code, f"{path} ({info.st_size} bytes) exceeds the remaining byte budget {budget}")
    identity = _file_identity(info)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        raise WorkerRefusal(code, f"cannot open {path}: {_message(exc)}") from exc
    try:
        if _file_identity(os.fstat(fd)) != identity:
            raise WorkerRefusal(code, f"{path} changed between lstat and open")
        hasher, read = hashlib.sha256(), 0
        while True:
            chunk = os.read(fd, min(_CHUNK, info.st_size + 1 - read))
            if not chunk:
                break
            read += len(chunk)
            if read > info.st_size:
                raise WorkerRefusal(code, f"{path} grew while it was read")
            hasher.update(chunk)
        if read != info.st_size or _file_identity(os.fstat(fd)) != identity:
            raise WorkerRefusal(code, f"{path} was modified while it was read")
    finally:
        os.close(fd)
    try:
        final = os.lstat(path)
    except OSError as exc:
        raise WorkerRefusal(code, f"{path} is missing or unreadable after it was read: {_message(exc)}") from exc
    if _file_identity(final) != identity:
        raise WorkerRefusal(code, f"{path} was replaced or modified while it was read")
    return hasher.hexdigest(), read


def _mapped_libraries(maps_path: str, roots: list[str]) -> list[str]:
    """Shared libraries mapped into this process that live under one of `roots` (bounded /proc/self/maps read)."""
    try:
        with open(maps_path, "rb") as handle:
            data = handle.read(MAX_MAPS_BYTES + 1)
    except OSError:
        return []
    if len(data) > MAX_MAPS_BYTES:
        raise WorkerRefusal("compiled_identity_bound_exceeded", f"{maps_path} is larger than {MAX_MAPS_BYTES} bytes")
    found = set()
    for line in data.decode("utf-8", "surrogateescape").splitlines():
        fields = line.split(None, 5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        path = fields[5].strip()
        base = os.path.basename(path)
        if ".so" not in base or path.endswith(" (deleted)"):
            continue
        real = os.path.realpath(path)
        if any(real == root or real.startswith(root + os.sep) for root in roots):
            found.add(real)
    return sorted(found)


def compiled_identities(modules: Mapping[str, Any], *, max_count: int, max_bytes: int,
                        maps_path: str = "/proc/self/maps", library_roots: list[str] | None = None) -> list[dict]:
    """sha256 of every loaded extension module plus every shared library mapped from the environment's site
    directories (e.g. NumPy's bundled BLAS), bounded in count and total bytes. Exceeding a bound refuses; nothing
    is silently truncated."""
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    entries: dict[str, dict[str, Any]] = {}
    for name in sorted(modules):
        module = modules[name]
        file = getattr(module, "__file__", None) if module is not None else None
        if isinstance(file, str) and file.endswith(suffixes):
            entry = entries.setdefault(os.path.realpath(file), {"kind": "extension_module", "modules": []})
            if len(entry["modules"]) < 20:
                entry["modules"].append(name)
    if library_roots is None:
        library_roots = sorted({os.path.realpath(p) for p in site.getsitepackages()})
    for path in _mapped_libraries(maps_path, library_roots):
        entries.setdefault(path, {"kind": "mapped_library", "modules": []})
    if len(entries) > max_count:
        raise WorkerRefusal("compiled_identity_bound_exceeded",
                            f"{len(entries)} compiled files exceed max_compiled_files={max_count}")
    records, total = [], 0
    for path in sorted(entries):
        digest, size = _sha256_file(path, max_bytes - total)
        total += size
        records.append({"path": path, **entries[path], "size": size, "sha256": digest})
    return records


def selected_distributions(specs: list[Mapping[str, Any]], *, path: list[str] | None = None
                           ) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """The distribution `importlib.metadata` selects for each name must be the sealed overlay at `spec["dir"]`
    with exactly `spec["files"]` (RECORD set and sha256 of the actual bytes). Every further match on the search
    path is reported as shadowed, not hidden and not a failure. `path=None` searches sys.path, as the resolver
    does; discovery is never patched."""
    selected: dict[str, Any] = {}
    shadowed: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        name, expected_dir, files = spec["name"], spec["dir"], spec["files"]
        kwargs: dict[str, Any] = {"name": name}
        if path is not None:
            kwargs["path"] = path
        found = list(importlib.metadata.distributions(**kwargs))
        locations = [os.path.realpath(str(dist.locate_file(""))) for dist in found]
        if not found:
            raise WorkerRefusal("selected_distribution_missing", f"no {name!r} distribution metadata is visible")
        if locations[0] != expected_dir:
            raise WorkerRefusal("selected_distribution_mismatch",
                                f"{name!r} metadata selected from {locations[0]}, expected the sealed {expected_dir}",
                                locations=locations[:20])
        dist = found[0]
        if path is None:
            singular = os.path.realpath(str(importlib.metadata.distribution(name).locate_file("")))
            if singular != expected_dir:
                raise WorkerRefusal("selected_distribution_mismatch",
                                    f"importlib.metadata.distribution({name!r}) located {singular}")
        listed = dist.files
        names = sorted(str(item) for item in listed) if listed is not None else None
        if names != sorted(files):
            raise WorkerRefusal("selected_distribution_files_mismatch",
                                f"{name!r} RECORD lists {names if names is None else names[:50]}, expected "
                                f"{sorted(files)[:50]}")
        for rel in sorted(files):
            target = str(dist.locate_file(rel))
            digest, _ = _sha256_file(target, MAX_METADATA_FILE_BYTES, code="selected_distribution_files_mismatch",
                                     bound_code="selected_distribution_files_mismatch")
            if digest != files[rel]:
                raise WorkerRefusal("selected_distribution_files_mismatch", f"{target} bytes differ from the lock")
        selected[name] = {"location": locations[0], "name": dist.metadata["Name"], "version": dist.version,
                          "files": dict(sorted(files.items()))}
        shadowed[name] = [{"location": loc, "name": other.metadata["Name"], "version": other.version}
                          for loc, other in zip(locations[1:], found[1:])][:50]
    return selected, shadowed


def registry_probe(*, checkpoint: str = "maple_syrup.checkpoint", pond: str = "maple_syrup.pond_checkpoint",
                   ledger: str = "maple.core.types.sediment_ledger",
                   topography: str = "maple.core.types.topographic_commit",
                   water: str = "maple.core.types.water") -> dict[str, Any]:
    """Call SYRUP's two native checkpoint registries unmodified and require the canonical public MAPLE classes
    under the registries' own keying (`{t.__name__: t}`), by object identity. The compact ledger is expected
    exactly when it is literally bound in the ledger module namespace. Class-dictionary operations only."""
    try:
        modules = {label: importlib.import_module(name) for label, name in (
            ("checkpoint", checkpoint), ("pond", pond), ("ledger", ledger), ("topography", topography),
            ("water", water))}
    except Exception as exc:  # noqa: BLE001 - recorded verbatim
        raise WorkerRefusal("checkpoint_registry_unavailable", f"registry module import failed ({_message(exc)})",
                            status="incompatible") from exc
    problems: list[str] = []

    def public_class(module, name: str, *, optional: bool = False):
        namespace = vars(module)
        if name not in namespace:
            if not optional:
                problems.append(f"{module.__name__} lacks {name}")
            return None
        value = namespace[name]
        if not isinstance(value, type):
            problems.append(f"{module.__name__}.{name} is bound to a non-class {type(value).__name__}")
            return None
        return value

    dense = public_class(modules["ledger"], "SedimentLedgerState")
    compact = public_class(modules["ledger"], "CompactSedimentLedgerState", optional=True)
    compact_bound = "CompactSedimentLedgerState" in vars(modules["ledger"])
    committed = public_class(modules["topography"], "CommittedTopographyState")
    water_state = public_class(modules["water"], "WaterState")
    native = [cls for cls in (dense, compact) if cls is not None]
    record: dict[str, Any] = {"compact_ledger_bound": compact_bound}
    calls = (("checkpoint._registry", modules["checkpoint"], "_registry", [*native, committed, water_state]),
             ("pond_checkpoint._native_registry", modules["pond"], "_native_registry", [*native, committed]))
    for label, module, attribute, expected in calls:
        function = vars(module).get(attribute)
        if not callable(function):
            problems.append(f"{module.__name__}.{attribute} is not callable")
            continue
        try:
            registry = function()
        except Exception as exc:  # noqa: BLE001 - e.g. the unpatched 2c60c73 unconditional compact import
            problems.append(f"{label}() raised {_message(exc)}")
            continue
        if not isinstance(registry, dict):
            problems.append(f"{label}() returned {type(registry).__name__}, not a dict")
            continue
        for cls in expected:
            if cls is not None and registry.get(cls.__name__) is not cls:
                problems.append(f"{label}() does not map {cls.__name__!r} to the public {cls.__module__} class")
        if not compact_bound and "CompactSedimentLedgerState" in registry:
            problems.append(f"{label}() has a CompactSedimentLedgerState entry the ledger module does not bind")
        record[label] = {"keys": sorted(str(key) for key in registry),
                         "native": {cls.__name__: f"{cls.__module__}.{cls.__qualname__}"
                                    for cls in expected if cls is not None}}
    if problems:
        raise WorkerRefusal("checkpoint_registry_unavailable", "; ".join(problems), status="incompatible",
                            record=record)
    return record


# ------------------------------------------------------------------------------------------- process checks


def _check_request(request: Any) -> str:
    if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
        raise WorkerRefusal("worker_request_invalid", f"request schema is not {REQUEST_SCHEMA!r} (the parent and the "
                            "sealed worker must come from compatible MASY versions)")
    operation = request.get("operation")
    if operation not in OPERATIONS:
        raise WorkerRefusal("worker_request_invalid", f"unknown operation {operation!r}")
    expected = _COMMON_KEYS | _OPERATION_KEYS[operation]
    if set(request) != expected:
        raise WorkerRefusal("worker_request_invalid", f"request keys {sorted(request)} != {sorted(expected)}")
    if not isinstance(request["limits"], dict) or set(request["limits"]) != _LIMIT_KEYS or not all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in request["limits"].values()):
        raise WorkerRefusal("worker_request_invalid", "request limits are malformed")
    return operation


def check_process(request: dict[str, Any], base_sys_path: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Flags, pycache prefix, exact sys.path prefix, environment and interpreter spelling of THIS process."""
    run, python, prefix = request["run"], request["python"], request["sys_path_prefix"]
    flags = {"isolated": sys.flags.isolated, "dont_write_bytecode": sys.flags.dont_write_bytecode,
             "ignore_environment": sys.flags.ignore_environment, "no_user_site": sys.flags.no_user_site,
             "safe_path": getattr(sys.flags, "safe_path", None), "utf8_mode": sys.flags.utf8_mode}
    problems = []
    for key in ("isolated", "dont_write_bytecode", "ignore_environment", "no_user_site"):
        if flags[key] != 1:
            problems.append(f"sys.flags.{key}={flags[key]}")
    if flags["safe_path"] is not True:
        problems.append(f"sys.flags.safe_path={flags['safe_path']}")
    if sys.pycache_prefix != run["pycache_prefix"]:
        problems.append(f"sys.pycache_prefix={sys.pycache_prefix!r}, expected {run['pycache_prefix']!r}")
    if sys.path[:len(prefix)] != prefix:
        problems.append("sys.path does not start with the exact sealed prefix")
    for entry in prefix:
        if os.path.realpath(entry) != entry or not os.path.isdir(entry):
            problems.append(f"sealed sys.path entry {entry} is not a canonical directory")
    env = run["env"]
    for key, value in env.items():
        if os.environ.get(key) != value:
            problems.append(f"environment {key}={os.environ.get(key)!r}, expected {value!r}")
    extra = sorted(set(os.environ) - set(env) - INTERPRETER_SET_ENV)
    if extra:
        problems.append(f"unexpected environment variables {extra[:20]}")
    if sys.executable != python["spelling"]:
        problems.append(f"sys.executable={sys.executable!r}, expected the configured spelling {python['spelling']!r}")
    if os.path.realpath(sys.executable) != python["realpath"]:
        problems.append(f"interpreter realpath {os.path.realpath(sys.executable)} != {python['realpath']}")
    if python["venv_dir"] is not None and os.path.realpath(sys.prefix) != python["venv_dir"]:
        problems.append(f"sys.prefix={sys.prefix} is not the configured environment {python['venv_dir']}")
    if problems:
        raise WorkerRefusal("worker_process_mismatch", "; ".join(problems)[:MAX_MESSAGE])
    stable = {"flags": flags, "sys_path_prefix": list(prefix), "executable": sys.executable,
              "executable_realpath": os.path.realpath(sys.executable), "prefix": sys.prefix,
              "base_prefix": sys.base_prefix, "version": sys.version, "implementation": sys.implementation.name,
              "machine": platform.machine(), "sys_platform": sys.platform,
              "environment": {key: os.environ.get(key) for key in STABLE_ENV_KEYS}}
    operational = {"pycache_prefix": sys.pycache_prefix, "pid": os.getpid(),
                   "environment": {key: os.environ.get(key) for key in OPERATIONAL_ENV_KEYS},
                   "interpreter_set_environment": {key: os.environ[key] for key in sorted(INTERPRETER_SET_ENV)
                                                   if key in os.environ}}
    receipt = {"sys_path_tail": sys.path[len(prefix):], "base_sys_path": list(base_sys_path),
               "platform": platform.platform()}
    return {"stable": {"process": stable}, "operational": operational, "receipt": receipt}, request["packages"]


def _origins(packages: Mapping[str, Mapping[str, str]], *, forbid: tuple[str, ...] = ()) -> None:
    from masy.importcheck import module_origin_violations

    violations = module_origin_violations(dict(sys.modules), {name: spec["dir"] for name, spec in packages.items()})
    loaded_forbidden = sorted(name for name in sys.modules if _family(name) in forbid)
    if violations:
        raise WorkerRefusal("module_origin_mismatch", f"{len(violations)} module(s) with a foreign origin",
                            violations=violations[:50])
    if loaded_forbidden:
        raise WorkerRefusal("module_origin_mismatch", f"operation must not load {forbid}: {loaded_forbidden[:20]}")


def _package_digests(packages: Mapping[str, Mapping[str, str]], limits: Mapping[str, int]) -> dict[str, Any]:
    provenance = importlib.import_module("maple_syrup.provenance")
    records = {}
    for name in sorted(packages):
        spec = packages[name]
        digest = provenance.source_tree_digest(spec["dir"], max_files=limits["max_files"],
                                               max_bytes=limits["max_bytes"])
        if digest.digest_sha256 != spec["digest"]:
            raise WorkerRefusal("package_digest_mismatch", f"imported {name} at {spec['dir']} digests to "
                                f"{digest.digest_sha256}, the lock/source manifest says {spec['digest']}")
        records[name] = {"dir": spec["dir"], "digest_sha256": digest.digest_sha256, "file_count": digest.file_count,
                         "total_bytes": digest.total_bytes, "algorithm": digest.algorithm}
    return records


def _bundle_tool(expected_sha256: str, syrup_dir: str):
    try:
        bundle = importlib.import_module("maple_syrup.dependency_bundle")
    except Exception as exc:  # noqa: BLE001
        raise WorkerRefusal("upstream_import_failed",
                            f"import maple_syrup.dependency_bundle failed ({_message(exc)})") from exc
    file = os.path.realpath(bundle.__file__)
    if file != os.path.join(syrup_dir, "dependency_bundle.py"):
        raise WorkerRefusal("module_origin_mismatch", f"dependency_bundle loaded from {file}")
    digest, _ = _sha256_file(file, MAX_METADATA_FILE_BYTES * 16, code="bundle_tool_mismatch",
                             bound_code="bundle_tool_mismatch")
    if digest != expected_sha256:
        raise WorkerRefusal("bundle_tool_mismatch", f"sealed dependency_bundle.py sha256 {digest} != {expected_sha256}")
    return bundle, digest


def _upstream_call(code: str, call, *args):
    try:
        return call(*args)
    except WorkerRefusal:
        raise
    except Exception as exc:  # noqa: BLE001 - upstream BundleError / MapleDependencyError recorded verbatim
        raise WorkerRefusal(code, _message(exc), error_type=type(exc).__name__) from exc


# ------------------------------------------------------------------------------------------------ operations


def _op_self_check(request: dict[str, Any], record: dict[str, Any], packages) -> dict[str, Any]:
    _origins({"masy": packages["masy"]}, forbid=("maple", "maple_syrup"))
    return record


def _op_export(request: dict[str, Any], record: dict[str, Any], packages) -> dict[str, Any]:
    bundle, tool_sha = _bundle_tool(request["tool_module_sha256"], packages["maple_syrup"]["dir"])
    spec = request["export"]
    manifest = _upstream_call("upstream_bundle_error", bundle.export_bundle, spec["project"], spec["distribution"],
                              spec["origin"], spec["output"])
    _origins(packages, forbid=("maple",))
    record["stable"]["bundle"] = {"bundle_id": manifest["bundle_id"], "tool_module_sha256": tool_sha}
    return record


def _op_materialize(request: dict[str, Any], record: dict[str, Any], packages) -> dict[str, Any]:
    spec = request["bundle"]
    bundle, tool_sha = _bundle_tool(spec["tool_module_sha256"], packages["maple_syrup"]["dir"])
    manifest = _upstream_call("upstream_bundle_error", bundle.verify_bundle, spec["dir"])
    if manifest["bundle_id"] != spec["bundle_id"]:
        raise WorkerRefusal("bundle_identity_mismatch", f"bundle at {spec['dir']} is {manifest['bundle_id']}")
    target = _upstream_call("upstream_bundle_error", bundle.materialize_bundle, spec["dir"], spec["cache"])
    verified = _upstream_call("upstream_bundle_error", bundle.verify_materialized, target)
    if str(target) != spec["target"] or verified["bundle_id"] != spec["bundle_id"]:
        raise WorkerRefusal("bundle_identity_mismatch", f"materialized {target} ({verified['bundle_id']}), expected "
                            f"{spec['target']}")
    _origins(packages, forbid=("maple",))
    record["stable"]["bundle"] = {"bundle_id": spec["bundle_id"], "tool_module_sha256": tool_sha,
                                  "target": str(target)}
    return record


def _op_runtime(request: dict[str, Any], record: dict[str, Any], packages) -> dict[str, Any]:
    from masy.contracts import run_native_audit
    from masy.importcheck import environment_record

    spec, limits = request["bundle"], request["limits"]
    stable = record["stable"]
    bundle, tool_sha = _bundle_tool(spec["tool_module_sha256"], packages["maple_syrup"]["dir"])
    manifest = _upstream_call("upstream_bundle_error", bundle.verify_bundle, spec["dir"])
    if manifest["bundle_id"] != spec["bundle_id"]:
        raise WorkerRefusal("bundle_identity_mismatch", f"bundle at {spec['dir']} is {manifest['bundle_id']}")
    stable["bundle"] = {"bundle_id": manifest["bundle_id"], "tool_module_sha256": tool_sha, "target": spec["target"]}
    stable["check_runtime"] = _upstream_call("native_check_runtime_failed", bundle.check_runtime, spec["target"])
    dependency = importlib.import_module("maple_syrup.dependency")
    audit, findings = run_native_audit(dependency, os.path.join(spec["target"], "source"))
    stable["native_audit"], stable["native_findings"] = audit, findings
    registry_failure = None
    try:
        stable["registry_probe"] = registry_probe()
    except WorkerRefusal as exc:
        stable["registry_probe"] = {"status": "failed", "code": exc.code, "detail": exc.detail, **exc.evidence}
        registry_failure = exc
    _origins(packages)
    stable["package_digests"] = _package_digests(packages, limits)
    stable["selected_distributions"], shadowed = selected_distributions(request["distributions"])
    env = environment_record()
    stable["interpreter_identity"] = env["executable_identity"]
    stable["numpy"] = env["numpy"]
    stable["optional_distribution_versions"] = env["optional_distribution_versions"]
    stable["optional_modules_loaded"] = env["optional_modules_loaded_by_upstream"]
    versions = {}
    for name in ENVIRONMENT_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    stable["distribution_versions"] = versions
    stable["compiled"] = compiled_identities(dict(sys.modules), max_count=limits["max_compiled_files"],
                                             max_bytes=limits["max_compiled_bytes"])
    stable["gpu_queried"] = False
    record["operational"]["environment_record"] = {"pycache_prefix": env["pycache_prefix"],
                                                   "environment_overrides": env["environment_overrides"]}
    record["receipt"]["shadowed_distributions"] = shadowed
    _origins(packages)  # final sweep after every import above

    severities = {finding["severity"] for finding in findings}
    if "refusal" in severities:
        raise WorkerRefusal("native_audit_refused", "; ".join(
            f"{f['code']}: {f['detail']}" for f in findings if f["severity"] == "refusal"), result=record)
    if "incompatible" in severities:
        raise WorkerRefusal("native_audit_incompatible", "; ".join(
            f"{f['code']}: {f['detail']}" for f in findings if f["severity"] == "incompatible"),
            status="incompatible", result=record)
    if registry_failure is not None:
        raise WorkerRefusal(registry_failure.code, registry_failure.detail, status="incompatible", result=record)
    return record


_OPERATIONS = {"self-check": _op_self_check, "export": _op_export, "materialize": _op_materialize,
               "runtime": _op_runtime}


def _emit(record: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
                     + "\n")
    sys.stdout.flush()


def main(request: Any, base_sys_path: list[str]) -> int:
    """Entered only from BOOTSTRAP. Prints exactly one result line; unexpected errors exit 1 with a traceback on
    stderr and no result line (the parent never treats that as success)."""
    operation = request.get("operation") if isinstance(request, dict) else None
    try:
        operation = _check_request(request)
        try:
            record, packages = check_process(request, base_sys_path)
        except (KeyError, TypeError, AttributeError) as exc:
            raise WorkerRefusal("worker_request_invalid", f"malformed request field ({_message(exc)})") from exc
        _origins({"masy": packages["masy"]}, forbid=("maple", "maple_syrup"))
        record = _OPERATIONS[operation](request, record, packages)
        result = {"schema": RESULT_SCHEMA, "operation": operation, "status": "ok", **record}
    except WorkerRefusal as exc:
        result = {"schema": RESULT_SCHEMA, "operation": operation, "status": exc.status, "code": exc.code,
                  "detail": exc.detail, "evidence": exc.evidence}
    except Exception:  # noqa: BLE001 - internal error: no result line
        traceback.print_exc()
        return 1
    try:
        _emit(result)
    except (TypeError, ValueError):
        traceback.print_exc()
        return 1
    return STATUS_EXIT[result["status"]]

def _compiled_identities_body(modules: Mapping[str, Any], *, max_count: int, max_bytes: int,
                        maps_path: str = "/proc/self/maps", library_roots: list[str] | None = None, _reuse=None) -> list[dict]:
    """sha256 of every loaded extension module plus every shared library mapped from the environment's site
    directories (e.g. NumPy's bundled BLAS), bounded in count and total bytes. Exceeding a bound refuses; nothing
    is silently truncated."""
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    entries: dict[str, dict[str, Any]] = {}
    for name in sorted(modules):
        module = modules[name]
        file = getattr(module, "__file__", None) if module is not None else None
        if isinstance(file, str) and file.endswith(suffixes):
            entry = entries.setdefault(os.path.realpath(file), {"kind": "extension_module", "modules": []})
            if len(entry["modules"]) < 20:
                entry["modules"].append(name)
    if library_roots is None:
        library_roots = sorted({os.path.realpath(p) for p in site.getsitepackages()})
    for path in _mapped_libraries(maps_path, library_roots):
        entries.setdefault(path, {"kind": "mapped_library", "modules": []})
    if len(entries) > max_count:
        raise WorkerRefusal("compiled_identity_bound_exceeded",
                            f"{len(entries)} compiled files exceed max_compiled_files={max_count}")
    if _reuse is not None:
        _reuse._begin_scan(len(entries))
    records, total = [], 0
    for path in sorted(entries):
        if _reuse is None:
            digest, size = _sha256_file(path, max_bytes - total)
        else:
            digest, size = _reuse._hash(path, max_bytes - total)
        total += size
        records.append({"path": path, **entries[path], "size": size, "sha256": digest})
    if _reuse is not None:
        _reuse._complete_scan()
    return records


# Private operation-scoped optimization; no runtime or Native authority.
_COMPILED_REUSE_KEY = object()
_COMPILED_REUSE_MAX_COUNT = 4000
_COMPILED_REUSE_MAX_BYTES = 4 << 30
_COMPILED_REUSE_PATH_CHARS = 4096
_COMPILED_REUSE_SCALAR = (1 << 64) - 1
# Only the Source-owned zero-argument-super constructor may retain a closure.
_COMPILED_REUSE_REFUSAL_IMAGE = (
    WorkerRefusal, WorkerRefusal.__init__, WorkerRefusal.__init__.__code__,
    WorkerRefusal.__init__.__closure__, WorkerRefusal.__init__.__closure__[0])


class _CompiledRow:
    __slots__ = ('path', 'size', 'digest', 'identity')

    def __init__(self, path, size, digest, identity):
        object.__setattr__(self, 'path', path)
        object.__setattr__(self, 'size', size)
        object.__setattr__(self, 'digest', digest)
        object.__setattr__(self, 'identity', identity)

    def __setattr__(self, name, value):
        raise AttributeError('compiled success row is immutable')

    def __delattr__(self, name):
        raise AttributeError('compiled success row is immutable')

    def __reduce_ex__(self, protocol):
        raise TypeError('compiled success row is process-local and nonserializable')


def _compiled_function_image(function):
    """Fixed Python function metadata only, not an arbitrary object graph."""
    if type(function) is not FunctionType:
        return None
    closure, cells = function.__closure__, ()
    if closure is not None:
        owner = _COMPILED_REUSE_REFUSAL_IMAGE
        if (type(owner) is not tuple or len(owner) != 5
                or WorkerRefusal is not owner[0] or function is not owner[1]
                or function is not WorkerRefusal.__init__ or function.__code__ is not owner[2]
                or function.__code__.co_freevars != ('__class__',)
                or closure is not owner[3] or len(closure) != 1 or closure[0] is not owner[4]):
            return None
        try:
            contents = closure[0].cell_contents
        except ValueError:  # An emptied real cell is ineligible, never a cache hit.
            return None
        if contents is not owner[0]:
            return None
        cells = ((closure[0], contents),)
    kw = function.__kwdefaults__
    if kw is not None and (type(kw) is not dict or len(kw) > 4
                           or any(type(name) is not str for name in kw)):
        return None
    return (function, function.__code__, function.__defaults__, kw,
            tuple(sorted(kw.items())) if kw is not None else (), function.__globals__,
            closure, cells)


def _compiled_reuse_image():
    functions = (WorkerRefusal.__init__, _message, _file_identity, _sha256_file, _mapped_libraries,
                 compiled_identities, _compiled_identities_body,
                 _compiled_identities_scoped, _new_compiled_reuse,
                 _compiled_function_image, _compiled_reuse_image,
                 _same_compiled_image, _compiled_reuse_charge, _compiled_frame_charge,
                 _compiled_image_charge,
                 _CompiledRow.__init__, _CompiledRow.__setattr__, _CompiledRow.__delattr__,
                 _CompiledRow.__reduce_ex__,
                 _CompiledReuse.__init__, _CompiledReuse.__reduce_ex__,
                 _CompiledReuse._check, _CompiledReuse._disable,
                 _CompiledReuse._begin_scan, _CompiledReuse._candidate,
                 _CompiledReuse._eligible, _CompiledReuse._remember,
                 _CompiledReuse._full, _CompiledReuse._hash,
                 _CompiledReuse._complete_scan, _CompiledReuse._discard_scan)
    images = tuple(_compiled_function_image(f) for f in functions)
    if any(x is None for x in images):
        return None
    operators = (os, os.path, hashlib, stat, site, importlib.machinery,
                 os.lstat, os.open, os.fstat, os.read, os.close, os.path.realpath,
                 os.getpid, threading.get_ident, hashlib.sha256, stat.S_ISREG,
                 site.getsitepackages, sys.getsizeof, sys._getframe, os.stat_result,
                 WorkerRefusal, _CompiledRow, _CompiledReuse,
                 FunctionType, _COMPILED_REUSE_REFUSAL_IMAGE)
    constants = (_IDENTITY_FIELDS, _CHUNK, MAX_MAPS_BYTES, MAX_MESSAGE,
                 tuple(importlib.machinery.EXTENSION_SUFFIXES),
                 _COMPILED_REUSE_MAX_COUNT, _COMPILED_REUSE_MAX_BYTES,
                 _COMPILED_REUSE_PATH_CHARS, _COMPILED_REUSE_SCALAR,
                 os.O_RDONLY, getattr(os, 'O_NOFOLLOW', 0), getattr(os, 'O_NONBLOCK', 0))
    return (images, operators, constants)


def _same_compiled_image(before, after):
    if before is None or after is None:
        return False
    if len(before[0]) != len(after[0]) or len(before[1]) != len(after[1]):
        return False
    for a, b in zip(before[0], after[0]):
        if any(a[i] is not b[i] for i in (0, 1, 2, 3, 5, 6)) or len(a[4]) != len(b[4]):
            return False
        if any(ka != kb or va is not vb for (ka, va), (kb, vb) in zip(a[4], b[4])):
            return False
        if len(a[7]) != len(b[7]) or any(ca is not cb or va is not vb
                for (ca, va), (cb, vb) in zip(a[7], b[7])):
            return False
    return all(a is b for a, b in zip(before[1], after[1])) and before[2] == after[2]


def _compiled_frame_charge(functions):
    """Charge every named/new nested code frame once; no alias subtraction."""
    pointer = sys.getsizeof((None,)) - sys.getsizeof(())
    header = sys.getsizeof(sys._getframe())
    # Actual input tuple, bounded traversal containers/scalars, and generator
    # object headers are separate from the logical plain-frame projection.
    generator = (None for _ in ())
    try:
        generator_header = sys.getsizeof(generator)
    finally:
        generator.close()
    total = sys.getsizeof(functions) + generator_header
    for function in functions:
        pending = [function.__code__]
        code_count = 0
        while pending:
            total += sys.getsizeof(pending)
            code = pending.pop()
            slots = (code.co_nlocals + code.co_stacksize
                     + len(code.co_cellvars) + len(code.co_freevars))
            total += header + slots * (pointer + sys.getsizeof(_COMPILED_REUSE_SCALAR))
            if code.co_flags & 0x20:  # CPython CO_GENERATOR; full object header additive.
                total += generator_header
            code_count += 1
            pending.extend(x for x in code.co_consts if type(x) is type(code))
        # Each named finite Source-code tree contributes a traversal list and
        # one pointer per visit without sharing/peak subtraction.
        total += code_count * sys.getsizeof(_COMPILED_REUSE_SCALAR)
    return total


def _compiled_image_charge(image):
    """Exact fixed image containers plus bounded constructor temporaries."""
    pointer = sys.getsizeof((None,)) - sys.getsizeof(())
    total = sys.getsizeof(image) + sum(sys.getsizeof(x) for x in image)
    for record in image[0]:
        total += sys.getsizeof(record) + sys.getsizeof(record[4])
        total += sum(sys.getsizeof(pair) for pair in record[4])
        # At most one exact owned cell; charge every retained tuple/pair/cell
        # occurrence without traversing or materializing its class contents.
        total += sys.getsizeof(record[7])
        if record[6] is not None:
            total += sys.getsizeof(record[6])
            total += sum(sys.getsizeof(pair) + sys.getsizeof(pair[0]) for pair in record[7])
        # Each real bounded kwdict is already admitted by the image builder.
        # Count its items view, sorted list and independent item-pair roles.
        kw = record[3]
        if kw is not None:
            ordered = sorted(kw.items())
            total += sys.getsizeof(kw.items()) + sys.getsizeof(ordered)
            total += sum(sys.getsizeof(pair) for pair in ordered)
    for operator in image[1]:
        if operator is _COMPILED_REUSE_REFUSAL_IMAGE:
            total += sys.getsizeof(operator)
    # Source-fixed function tuple exists during image construction.
    total += sys.getsizeof(()) + len(image[0]) * pointer
    for value in image[2]:
        if type(value) is tuple:
            total += sys.getsizeof(value)
    return total


def _compiled_reuse_charge():
    """Additive logical metadata projection, never a whole-heap fit claim."""
    n, length = _COMPILED_REUSE_MAX_COUNT, _COMPILED_REUSE_PATH_CHARS
    pointer = sys.getsizeof((None,)) - sys.getsizeof(())
    row = (sys.getsizeof(_CompiledRow(None, None, None, None))
           + sys.getsizeof((None,) * 6) + 7 * sys.getsizeof(_COMPILED_REUSE_SCALAR)
           + sys.getsizeof('\U00010000') + 4 * length + sys.getsizeof('0' * 64))
    rows = 2 * (sys.getsizeof([]) + n * pointer) + (2 * n + 1) * row
    # Two complete finite image occurrences, plus independently charged slots.
    image = _compiled_reuse_image()
    if image is None:
        return None
    image_bytes = _compiled_image_charge(image)
    # Referenced code/functions/modules are retained, not copied or materialized.
    fixed = (sys.getsizeof(object.__new__(_CompiledReuse))
             + len(_CompiledReuse.__slots__) * (pointer + sys.getsizeof(_COMPILED_REUSE_SCALAR)))
    stat_role = (sys.getsizeof(os.stat_result((0,) * 10))
                 + (os.stat_result.n_fields - os.stat_result.n_sequence_fields) * pointer
                 + os.stat_result.n_fields * (sys.getsizeof(_COMPILED_REUSE_SCALAR)
                                              + sys.getsizeof(0.0)))
    # Raw os.open only: two FS-encoded path roles, no buffered file object.
    fixed += 2 * (sys.getsizeof(b'') + 4 * length + 1)
    # Actual nine-field worker projection result; caller adds its own record.
    fixed += sys.getsizeof(dict.fromkeys(('rows', 'fixed', 'bytes', 'max_count',
             'max_path_chars', 'row_bytes', 'image_functions', 'image_operators',
             'image_constants'))) + 9 * sys.getsizeof(_COMPILED_REUSE_SCALAR)
    comparisons = 4 * (sys.getsizeof((None,) * 6)
                       + 6 * sys.getsizeof(_COMPILED_REUSE_SCALAR))
    fixed += 2 * stat_role + comparisons
    # All named functions and their finite nested code objects are additive.
    frames = _compiled_frame_charge(tuple(record[0] for record in image[0]))
    return {'rows': rows, 'fixed': fixed + 2 * image_bytes + frames,
            'bytes': rows + fixed + 2 * image_bytes + frames,
            'max_count': n, 'max_path_chars': length,
            'row_bytes': row, 'image_functions': len(image[0]),
            'image_operators': len(image[1]), 'image_constants': len(image[2])}


class _CompiledReuse:
    __slots__ = ('_key', '_token', '_pid', '_thread', '_max_count', '_max_bytes',
                 '_rows', '_next', '_at', '_cursor', '_image', '_image_builder',
                 '_closed', '_disabled', '_force_full')

    def __init__(self, key, token, max_count, max_bytes):
        if key is not _COMPILED_REUSE_KEY or type(token) is not object or type(max_count) is not int or not 0 <= max_count <= 4000 \
                or type(max_bytes) is not int or not 0 <= max_bytes <= 4 << 30:
            raise WorkerRefusal('compiled_scope', 'private bounded compiled scope required')
        self._key, self._token = key, token
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._max_count, self._max_bytes = max_count, max_bytes
        self._rows, self._next = None, None
        self._at = self._cursor = 0
        self._closed = self._disabled = self._force_full = False
        self._image_builder = _compiled_reuse_image
        self._image = _compiled_reuse_image()
        if self._image is None:
            self._disabled = True

    def __reduce_ex__(self, protocol):
        raise TypeError('compiled scope is process-local and nonserializable')

    def _check(self, token):
        if self._key is not _COMPILED_REUSE_KEY or self._token is not token or self._closed \
                or os.getpid() != self._pid or threading.get_ident() != self._thread:
            raise WorkerRefusal('compiled_scope', 'closed/foreign/fork/thread compiled scope')
        if self._image_builder is not _compiled_reuse_image or not _same_compiled_image(
                self._image, _compiled_reuse_image()):
            self._disable()

    def _disable(self):
        self._disabled = True
        self._rows = self._next = None

    def _begin_scan(self, count):
        if self._next is not None:
            raise WorkerRefusal('compiled_scope', 'nested compiled scan is unsupported')
        if not self._disabled:
            self._next = [None] * count
            self._at = self._cursor = 0

    def _candidate(self, path):
        rows = self._rows
        if rows is None:
            return None
        while self._cursor < len(rows):
            row = rows[self._cursor]
            if row is None or row.path < path:
                self._cursor += 1
            elif row.path == path:
                return row
            else:
                break
        return None

    def _eligible(self, path, info):
        ident = _file_identity(info)
        return (type(path) is str and len(path) <= _COMPILED_REUSE_PATH_CHARS
                and all(type(x) is int and abs(x) <= _COMPILED_REUSE_SCALAR for x in ident))

    def _remember(self, row):
        if self._disabled:
            return
        if self._next is None or self._at >= len(self._next):
            raise WorkerRefusal('compiled_scope', 'compiled roster occurrence differs')
        self._next[self._at] = row
        self._at += 1

    def _full(self, path, budget):
        pre = None
        if not self._disabled:
            try:
                pre = os.lstat(path)
            except OSError:
                pass
        digest, size = _sha256_file(path, budget)
        row = None
        if pre is not None and not self._disabled:
            try:
                post = os.lstat(path)
            except OSError:
                post = None
            if post is not None and self._eligible(path, post) and pre.st_size == size \
                    and _file_identity(pre) == _file_identity(post):
                row = _CompiledRow(path, size, digest, _file_identity(post))
        self._remember(row)
        return digest, size

    def _hash(self, path, budget):
        self._check(self._token)
        candidate = None if self._disabled or self._force_full else self._candidate(path)
        if candidate is None:
            return self._full(path, budget)
        code, bound_code = 'compiled_identity_unreadable', 'compiled_identity_bound_exceeded'
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise WorkerRefusal(code, f'cannot stat {path}: {_message(exc)}') from exc
        if not stat.S_ISREG(info.st_mode):
            raise WorkerRefusal(code, f'{path} is not a regular file')
        if info.st_size > budget:
            raise WorkerRefusal(bound_code, f'{path} ({info.st_size} bytes) exceeds the remaining byte budget {budget}')
        identity = _file_identity(info)
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
        except OSError as exc:
            raise WorkerRefusal(code, f'cannot open {path}: {_message(exc)}') from exc
        stale = False
        try:
            if _file_identity(os.fstat(fd)) != identity:
                raise WorkerRefusal(code, f'{path} changed between lstat and open')
            if candidate.identity != identity:
                stale = True
            elif _file_identity(os.fstat(fd)) != identity:
                raise WorkerRefusal(code, f'{path} was modified while it was read')
        finally:
            os.close(fd)
        if stale:
            return self._full(path, budget)
        try:
            final = os.lstat(path)
        except OSError as exc:
            raise WorkerRefusal(code, f'{path} is missing or unreadable after it was read: {_message(exc)}') from exc
        if _file_identity(final) != identity:
            raise WorkerRefusal(code, f'{path} was replaced or modified while it was read')
        self._remember(candidate)
        return candidate.digest, info.st_size

    def _complete_scan(self):
        if self._disabled:
            return
        if self._next is None or self._at != len(self._next):
            raise WorkerRefusal('compiled_scope', 'partial compiled roster cannot commit')
        self._rows, self._next = self._next, None

    def _discard_scan(self):
        self._disable()


def _new_compiled_reuse(token, *, max_count, max_bytes):
    return _CompiledReuse(_COMPILED_REUSE_KEY, token, max_count, max_bytes)


def _compiled_identities_scoped(scope, token, modules, *, max_count, max_bytes,
                                maps_path='/proc/self/maps', library_roots=None):
    if type(scope) is not _CompiledReuse:
        raise WorkerRefusal('compiled_scope', 'actual private compiled scope required')
    scope._check(token)
    if (max_count, max_bytes) != (scope._max_count, scope._max_bytes):
        raise WorkerRefusal('compiled_scope', 'compiled scope limits changed')
    try:
        return _compiled_identities_body(modules, max_count=max_count, max_bytes=max_bytes,
                                         maps_path=maps_path, library_roots=library_roots,
                                         _reuse=scope)
    except BaseException:
        scope._discard_scan()
        raise
