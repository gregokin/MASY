"""Upstream import, module-origin, distribution-metadata and environment records.

Called only after the stdlib source observation passed. Origins are checked
against the concrete observed package directories (`<root>/src/<package>`),
never a loose ancestor. Distribution metadata is read with
`importlib.metadata` and reported, never rewritten. Optional compiled
backends (numba, llvmlite, CUDA packages) are never imported here; only their
installed metadata versions are read, and their origins recorded if an
upstream import already loaded them. No device is queried.

These are LIVE imports of a working tree: they are not immutable-runtime
evidence (`immutable_import_unverified` stays blocking).
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

__all__ = [
    "OPTIONAL_DISTRIBUTIONS",
    "OriginError",
    "distribution_records",
    "environment_record",
    "import_upstreams",
    "module_origin_violations",
]

OPTIONAL_DISTRIBUTIONS = (
    "numba", "llvmlite", "jax", "jaxlib", "cupy-cuda12x", "cupy-cuda11x", "cuda-python",
    "nvidia-cuda-runtime-cu12",
)
OPTIONAL_MODULES = ("numba", "llvmlite", "jax", "jaxlib", "cupy")
MAX_HASH_BYTES = 512 * 1024 * 1024
MAX_DIRECT_URL_BYTES = 64 * 1024


class OriginError(Exception):
    def __init__(self, code: str, detail: str, **evidence: Any) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.evidence = evidence


def _belongs(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


def _within(path: str, directory: str) -> bool:
    return Path(os.path.realpath(path)).is_relative_to(Path(directory))


def module_origin_violations(modules: Mapping[str, Any], expected_dirs: Mapping[str, str]) -> list[dict[str, Any]]:
    """Every loaded module of each expected package must come from its
    concrete package directory; the package itself from `<dir>/__init__.py`
    with exactly that directory as `__path__`."""
    violations: list[dict[str, Any]] = []
    for name in sorted(modules):
        module = modules[name]
        package = next((pkg for pkg in expected_dirs if _belongs(name, pkg)), None)
        if package is None or module is None:
            continue
        directory = expected_dirs[package]
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            violations.append({"module": name, "reason": "no __file__ (namespace or synthetic module)"})
            continue
        if name == package:
            if os.path.realpath(module_file) != os.path.join(directory, "__init__.py"):
                violations.append({"module": name, "file": module_file, "expected_dir": directory,
                                   "reason": "package is not <dir>/__init__.py"})
                continue
            paths = [os.path.realpath(p) for p in getattr(module, "__path__", [])]
            if paths != [directory]:
                violations.append({"module": name, "path": paths, "expected_dir": directory,
                                   "reason": "package __path__ is not exactly the observed directory"})
            continue
        if not _within(module_file, directory):
            violations.append({"module": name, "file": module_file, "expected_dir": directory,
                               "reason": "module file outside the observed package directory"})
            continue
        for entry in getattr(module, "__path__", None) or []:
            if not _within(entry, directory):
                violations.append({"module": name, "path_entry": entry, "expected_dir": directory,
                                   "reason": "subpackage __path__ outside the observed package directory"})
    return violations


def import_upstreams(upstream_src: list[str], expected_dirs: Mapping[str, str]) -> dict[str, Any]:
    """Insert upstream src roots after MASY's src and import the packages.

    Refuses (OriginError) if an upstream package was already imported, an
    import fails, or any loaded module has a foreign origin.
    """
    upstream_names = [name for name in expected_dirs if name != "masy"]
    preloaded = sorted(n for n in sys.modules if any(_belongs(n, pkg) for pkg in upstream_names))
    if preloaded:
        raise OriginError("upstream_preimported", f"upstream modules already imported: {preloaded[:20]}")
    path_before = list(sys.path)
    sys.path[1:1] = list(upstream_src)
    importlib.invalidate_caches()
    record: dict[str, Any] = {
        "sys_path_before": path_before,
        "sys_path_inserted": list(upstream_src),
        "sys_path_after": list(sys.path),
        "packages": {},
    }
    for name in upstream_names:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - reported verbatim as a refusal
            raise OriginError(
                "upstream_import_failed", f"import {name} failed: {type(exc).__name__}: {exc}"[:4000],
                record=record,
            ) from exc
        record["packages"][name] = {"file": getattr(module, "__file__", None), "expected_dir": expected_dirs[name]}
    violations = module_origin_violations(dict(sys.modules), expected_dirs)
    if violations:
        raise OriginError("module_origin_mismatch", f"{len(violations)} module(s) with a foreign origin",
                          violations=violations[:50], record=record)
    return record


def _identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _bounded_sha256(path: str) -> dict[str, Any]:
    """sha256 of one regular file, read in bounded chunks never past
    min(size, MAX_HASH_BYTES) + 1 bytes. The path stat, the open descriptor
    and a final path stat must agree on device/inode/size/mtime, and the
    bytes read must equal the size; otherwise no digest is reported."""
    real = os.path.realpath(path)
    bound = MAX_HASH_BYTES
    try:
        before = os.stat(real)
        if not stat.S_ISREG(before.st_mode):
            return {"path": real, "sha256": None, "reason": "not a regular file"}
        if before.st_size > bound:
            return {"path": real, "size": before.st_size, "sha256": None, "reason": "larger than hash bound"}
        fd = os.open(real, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
                return {"path": real, "sha256": None, "reason": "file changed between stat and open"}
            hasher = hashlib.sha256()
            limit = before.st_size + 1
            read = 0
            while read < limit:
                chunk = os.read(fd, min(1 << 20, limit - read))
                if not chunk:
                    break
                read += len(chunk)
                hasher.update(chunk)
            after_fd = os.fstat(fd)
        finally:
            os.close(fd)
        after_path = os.stat(real)
    except OSError as exc:
        return {"path": real, "sha256": None, "reason": f"{type(exc).__name__}: {exc}"}
    if read != before.st_size or _identity(after_fd) != _identity(before) or \
            _identity(after_path) != _identity(before):
        return {"path": real, "size": before.st_size, "sha256": None,
                "reason": f"file changed during read (read {read} bytes of {before.st_size}, or identity changed)"}
    return {"path": real, "size": before.st_size, "sha256": hasher.hexdigest()}


def distribution_records(names: dict[str, str | None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Installed metadata for each distribution vs its selected source root.

    `names` maps distribution name -> selected source root (or None).
    Returns (records, findings). Never rewrites metadata.
    """
    records: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    for dist_name, selected_root in names.items():
        dists = list(importlib.metadata.distributions(name=dist_name))
        entries = []
        for dist in dists:
            raw = dist.read_text("direct_url.json")
            direct_url: Any = None
            if raw is not None:
                if len(raw) > MAX_DIRECT_URL_BYTES:
                    direct_url = {"unparsed": "larger than bound"}
                else:
                    try:
                        direct_url = json.loads(raw)
                    except json.JSONDecodeError:
                        direct_url = {"unparsed": raw[:1000]}
            entries.append({
                "name": dist.metadata["Name"],
                "version": dist.version,
                "location": str(dist.locate_file("")),
                "direct_url": direct_url,
            })
        records[dist_name] = {"selected_root": selected_root, "distributions": entries}
        if not entries:
            findings.append({"code": "distribution_metadata_absent", "severity": "info",
                             "detail": f"no installed {dist_name!r} distribution metadata"})
            continue
        matches = []
        for entry in entries:
            direct = entry["direct_url"] if isinstance(entry["direct_url"], dict) else {}
            url = direct.get("url")
            parsed = urlparse(url) if isinstance(url, str) else None
            dir_info = direct.get("dir_info")
            editable = isinstance(dir_info, dict) and dir_info.get("editable") is True
            matches.append(bool(
                editable and selected_root and parsed is not None and parsed.scheme == "file"
                and os.path.realpath(unquote(parsed.path)) == selected_root
            ))
        if len(entries) != 1 or not all(matches):
            findings.append({
                "code": "distribution_metadata_unverified", "severity": "blocking",
                "detail": f"{dist_name!r} metadata ({len(entries)} distribution(s)) does not identify the "
                          f"selected root {selected_root} as an editable source; reported, not repaired",
            })
    return records, findings


def environment_record() -> dict[str, Any]:
    """Interpreter, NumPy compiled core and optional backend identities."""
    record: dict[str, Any] = {
        "python_version": platform.python_version(),
        "implementation": sys.implementation.name,
        "executable": sys.executable,
        "executable_identity": _bounded_sha256(sys.executable),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "platform": platform.platform(),
        "flags": {"isolated": sys.flags.isolated, "dont_write_bytecode": sys.flags.dont_write_bytecode,
                  "safe_path": getattr(sys.flags, "safe_path", None)},
        "pycache_prefix": sys.pycache_prefix,
        "environment_overrides": {key: os.environ.get(key) for key in
                                  ("NUMBA_CACHE_DIR", "NUMBA_DISABLE_CUDA", "CUDA_VISIBLE_DEVICES",
                                   "JAX_PLATFORMS", "MPLCONFIGDIR")},
    }
    # NumPy is recorded only if upstream imports loaded it; it is not imported here.
    numpy_module: ModuleType | None = sys.modules.get("numpy")
    numpy_record: dict[str, Any] = {"loaded": numpy_module is not None}
    if numpy_module is not None:
        numpy_record["version"] = getattr(numpy_module, "__version__", None)
        numpy_record["file"] = getattr(numpy_module, "__file__", None)
        core = sys.modules.get("numpy._core._multiarray_umath") or sys.modules.get("numpy.core._multiarray_umath")
        numpy_record["compiled_core"] = (
            {"module": core.__name__, **_bounded_sha256(core.__file__)} if core is not None
            and getattr(core, "__file__", None) else None
        )
    record["numpy"] = numpy_record
    versions: dict[str, str | None] = {}
    for name in OPTIONAL_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    record["optional_distribution_versions"] = versions
    record["optional_modules_loaded_by_upstream"] = {
        name: getattr(sys.modules[name], "__file__", None) for name in OPTIONAL_MODULES if name in sys.modules
    }
    record["gpu_queried"] = False
    return record
