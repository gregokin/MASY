"""Closure-safe, stdlib-only verification of an immutable LEAF source capsule (`masy.leaf_capsule.v1`).

Imports only the standard library and `masy.native_source_gate` (never `masy.runtime`, `masy.sourceobs`, NumPy, LEAF, SYRUP or MAPLE), so the parent and worker
closures may load it. The operator-side sealing lives in `masy.leaf_capsule_seal`.

Layout (a namespace OUTSIDE any V1 runtime store)::

    <ns>/objects/<capsule_id>/            0555, id = sha256(canonical identity)
        manifest.json                     canonical {schema, object_id, identity, provenance}
        source-manifest.json              gate envelope {"files": {rel: sha256}, "note": ...}
        source/pyproject.toml, source/leaf/**   0444 files, 0555 directories

The identity hashes file bytes (path, size, sha256, Git blob id), the object format, the project identity and two DERIVED digests; it never hashes a path, commit or
revision. Provenance (actual commit/tree) lives only in manifest.json. The capsule is a byte-consistency record: it neither renames nor qualifies any LEAF source
profile (the literal 8da manifest/package check stays in `leaf_source_profile`), and it makes no claim of independent scientific review.

Modelled byte estimates admit BEFORE each body read/parse; they are not whole-process RSS and exclude interpreter/import costs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tomllib
from typing import Any, Callable

from masy.native_source_gate import GateLimits, GateRefusal, _tree_digest as _size_tree_digest, read_manifest, verify_snapshot

__all__ = ["CAPSULE_SCHEMA", "CAPSULE_KIND", "SOURCE_DIGEST_ALGORITHM", "canonical_bytes", "package_digest", "tree_digest", "validate_identity", "validate_namespace", "verify_capsule",
           "verify_capsule_dir", "MAX_NAMESPACE_OBJECTS"]

CAPSULE_SCHEMA = "masy.leaf_capsule.v1"
CAPSULE_KIND = "leaf_capsule"
SOURCE_DIGEST_ALGORITHM = "sha256(sorted relpath NUL file_sha256 LF), excl __pycache__/*.pyc/*.pyo; v1"
PACKAGE_DIR = "leaf"
TOP_ENTRIES = frozenset({"manifest.json", "source-manifest.json", "source"})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_STAGING = re.compile(r"^\.staging-[0-9A-Za-z._-]{1,128}-leaf$")
_SLACK = 16384
MAX_NAMESPACE_OBJECTS = 4096  # explicit bound on capsule + staging directories one namespace may hold (a namespace is not a source tree)
_MAX_NESTING = 16
Admit = Callable[[int, str], None]


def _no_admit(_n: int, _what: str) -> None:
    return None


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def tree_digest(entries: list[tuple[str, str]]) -> str:
    """The `sourceobs.DIGEST_ALGORITHM` / `maple_syrup.provenance` source-digest algorithm over (relative path, sha256) pairs."""
    hasher = hashlib.sha256()
    for relative, digest in sorted(entries):
        hasher.update(relative.encode("utf-8", "surrogateescape") + b"\0" + digest.encode("ascii") + b"\n")
    return hasher.hexdigest()


def package_digest(files: list[list[Any]]) -> str:
    """Digest of the `leaf/` package directory alone (paths relative to it): the quantity a LEAF package literal names, DISTINCT from the full-tree digest."""
    prefix = PACKAGE_DIR + "/"
    return tree_digest([(f[0][len(prefix):], f[2]) for f in files if f[0].startswith(prefix)])


def _refuse(code: str, detail: str) -> GateRefusal:
    return GateRefusal(code, detail)


def _check_path(path: Any, limits: GateLimits) -> str:
    parts = path.split("/") if type(path) is str else []
    if (type(path) is not str or not path or len(path) > limits.max_path_chars or "\x00" in path or "\\" in path or path.startswith("/") or any(p in ("", ".", "..") for p in parts)
            or len(parts) > limits.max_depth):
        raise _refuse("capsule_manifest_invalid", f"unsafe capsule path {path!r}")
    if ".git" in parts or "__pycache__" in parts or path.endswith((".pyc", ".pyo")):
        raise _refuse("capsule_manifest_invalid", f"{path!r}: VCS metadata or bytecode is not capsule source")
    if not (path == "pyproject.toml" or path.startswith(PACKAGE_DIR + "/")):
        raise _refuse("capsule_manifest_invalid", f"{path!r} is outside the capsule scope (pyproject.toml and {PACKAGE_DIR}/)")
    return path


def _nesting_ok(data: bytes) -> bool:
    depth, in_str, esc = 0, False, False
    for b in data:
        if in_str:
            if esc:
                esc = False
            elif b == 0x5C:
                esc = True
            elif b == 0x22:
                in_str = False
        elif b == 0x22:
            in_str = True
        elif b in (0x7B, 0x5B):
            depth += 1
            if depth > _MAX_NESTING:
                return False
        elif b in (0x7D, 0x5D):
            depth -= 1
    return True


def _file_join(st: os.stat_result) -> tuple[int, int, int, int, int]:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _read_file(path: str, cap: int, admit: Admit, what: str, code: str) -> bytes:
    """No-follow bounded read of one small regular file: size cap and admission BEFORE the body; every read is at most min(chunk, remaining + 1) bytes; the path lstat, the open
    descriptor and the final path lstat must carry the same device/inode/size/mtime/ctime. Modelled bytes (raw + one copy + slack), not RSS."""
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise _refuse(code, f"{what}: {exc}") from None
    if not stat.S_ISREG(before.st_mode):
        raise _refuse(code, f"{what} is not a regular file")
    if before.st_size > cap:
        raise _refuse(code, f"{what} is {before.st_size} B, over the {cap}-B cap (checked before the body)")
    admit(2 * int(before.st_size) + _SLACK, f"reading {what}")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except OSError as exc:
        raise _refuse(code, f"{what}: {exc}") from None
    try:
        if _file_join(os.fstat(fd)) != _file_join(before):
            raise _refuse("capsule_changed", f"{what} was replaced while opened")
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, min(1 << 20, before.st_size - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > before.st_size:
                raise _refuse("capsule_changed", f"{what} grew while read")
            chunks.append(chunk)
        if total != before.st_size or _file_join(os.fstat(fd)) != _file_join(before):
            raise _refuse("capsule_changed", f"{what} changed while read")
    finally:
        os.close(fd)
    try:
        final = os.lstat(path)
    except OSError as exc:
        raise _refuse("capsule_changed", f"{what} vanished after it was read: {exc}") from None
    if _file_join(final) != _file_join(before):
        raise _refuse("capsule_changed", f"{what} was replaced or modified while it was read")
    return b"".join(chunks)


def _scan_dir(path: str, cap: int, limits: GateLimits, admit: Admit, what: str, code: str) -> list[tuple[str, os.stat_result]]:
    """Bounded descriptor-relative no-follow enumeration of one directory: admission BEFORE the directory is opened and before each entry is retained; the cap is enforced while
    iterating (the (cap+1)-th entry refuses, so at most `cap` entries are ever retained, and nothing is sorted before the cap holds). Entries are lstat'ed relative to the descriptor."""
    admit(limits.entry_model_bytes, what)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise _refuse("capsule_unsafe", f"{path} cannot be opened as a real directory: {exc}") from None
    out: list[tuple[str, os.stat_result]] = []
    try:
        with os.scandir(fd) as it:
            for entry in it:
                if len(out) >= cap:
                    raise _refuse(code, f"{path} holds more than {cap} entries (checked while iterating)")
                admit(limits.entry_model_bytes, what)
                name = entry.name
                if len(name) > limits.max_path_chars or "\x00" in name:
                    raise _refuse("capsule_unsafe", f"unsafe or over-long entry name {name[:80]!r} in {path}")
                try:
                    out.append((name, os.stat(name, dir_fd=fd, follow_symlinks=False)))
                except OSError as exc:
                    raise _refuse("capsule_changed", f"{path}/{name} vanished during enumeration: {exc}") from None
    finally:
        os.close(fd)
    out.sort(key=lambda item: item[0])
    return out


def _strict_json(data: bytes, limits: GateLimits, admit: Admit, what: str) -> Any:
    if not _nesting_ok(data):
        raise _refuse("capsule_manifest_invalid", f"{what}: nesting deeper than {_MAX_NESTING}")
    nodes = data.count(b":") + data.count(b",") + 1
    admit(2 * len(data) + limits.node_model_bytes * nodes + _SLACK, f"parsing {what}")

    def pairs(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(items)

    def refuse_constant(name):
        raise ValueError(f"non-finite constant {name}")
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=refuse_constant)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise _refuse("capsule_manifest_invalid", f"{what}: {type(exc).__name__}: {exc}") from None


def _closed(value: Any, keys: set[str], what: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise _refuse("capsule_manifest_invalid", f"{what} keys {got} != {sorted(keys)}")
    return value


def _hex(value: Any, what: str, pattern=_HEX64) -> str:
    if type(value) is not str or not pattern.match(value):
        raise _refuse("capsule_manifest_invalid", f"{what} is not a lowercase hex digest")
    return value


def validate_identity(identity: Any, limits: GateLimits) -> dict[str, Any]:
    """Closed identity validation, including the recomputed derived digests."""
    ident = _closed(identity, {"schema", "kind", "object_format", "package", "project", "files", "tree_digest", "package_digest"}, "identity")
    if ident["schema"] != CAPSULE_SCHEMA or ident["kind"] != CAPSULE_KIND or ident["package"] != PACKAGE_DIR or ident["object_format"] not in ("sha1", "sha256"):
        raise _refuse("capsule_manifest_invalid", "identity schema/kind/package/object_format is invalid")
    project = _closed(ident["project"], {"name", "version", "pyproject_sha256"}, "identity.project")
    if project["name"] != "leaf" or type(project["version"]) is not str or not project["version"]:
        raise _refuse("capsule_manifest_invalid", "identity.project must name project 'leaf' with a version")
    _hex(project["pyproject_sha256"], "identity.project.pyproject_sha256")
    files = ident["files"]
    if not isinstance(files, list) or not 1 <= len(files) <= limits.max_files:
        raise _refuse("capsule_manifest_invalid", f"identity.files needs 1..{limits.max_files} entries")
    previous, total = None, 0
    blob_len = 40 if ident["object_format"] == "sha1" else 64
    for item in files:
        if not (isinstance(item, list) and len(item) == 4 and type(item[1]) is int and item[1] >= 0):
            raise _refuse("capsule_manifest_invalid", f"identity file entry {item!r} is malformed")
        _check_path(item[0], limits)
        _hex(item[2], f"{item[0]} sha256")
        _hex(item[3], f"{item[0]} blob", re.compile(f"^[0-9a-f]{{{blob_len}}}$"))
        if item[1] > limits.max_file_bytes:
            raise _refuse("capsule_manifest_invalid", f"{item[0]} is over the {limits.max_file_bytes}-B file cap")
        if previous is not None and item[0] <= previous:
            raise _refuse("capsule_manifest_invalid", "identity files are not strictly sorted")
        previous, total = item[0], total + item[1]
    if total > limits.max_total_bytes:
        raise _refuse("capsule_manifest_invalid", f"identity files exceed {limits.max_total_bytes} B in total")
    paths = {f[0] for f in files}
    if "pyproject.toml" not in paths or "leaf/__init__.py" not in paths:
        raise _refuse("capsule_manifest_invalid", "identity lacks pyproject.toml or leaf/__init__.py")
    if next(f for f in files if f[0] == "pyproject.toml")[2] != project["pyproject_sha256"]:
        raise _refuse("capsule_manifest_invalid", "identity pyproject sha256 differs from its file entry")
    if ident["tree_digest"] != tree_digest([(f[0], f[2]) for f in files]) or ident["package_digest"] != package_digest(files):
        raise _refuse("capsule_manifest_invalid", "identity derived digests do not match its file entries")
    return ident


def _expected_dirs(files: list[list[Any]]) -> set[str]:
    dirs: set[str] = set()
    for f in files:
        parent = os.path.dirname(f[0])
        while parent:
            dirs.add(parent)
            parent = os.path.dirname(parent)
    return dirs


def _dir_census(root: str, expected: set[str], limits: GateLimits, admit: Admit) -> None:
    """Exact directory set of `root` (extra/missing directories refuse; `verify_snapshot` compares the file set only). Bounded per directory and in total, no-follow."""
    found: set[str] = set()
    pending, entries, total_cap = [("", root)], 0, limits.max_files + len(expected) + 1
    while pending:
        rel, path = pending.pop()
        for name, st in _scan_dir(path, min(limits.max_dir_entries, total_cap - entries), limits, admit, "capsule directory census", "capsule_entries"):
            entries += 1
            sub = f"{rel}/{name}" if rel else name
            if stat.S_ISDIR(st.st_mode):
                found.add(sub)
                pending.append((sub, os.path.join(path, name)))
    if found != expected:
        raise _refuse("capsule_entries", f"directory set differs: extra={sorted(found - expected)[:5]} missing={sorted(expected - found)[:5]}")


def _check_top(location: str, limits: GateLimits, admit: Admit) -> None:
    try:
        st = os.lstat(location)
    except OSError as exc:
        raise _refuse("capsule_missing", f"{location}: {exc}") from None
    if not stat.S_ISDIR(st.st_mode):
        raise _refuse("capsule_unsafe", f"{location} is not a real directory (symlink or special refused)")
    if st.st_mode & 0o222:
        raise _refuse("capsule_writable", f"{location} has write bits; a sealed capsule is read-only")
    entries = _scan_dir(location, len(TOP_ENTRIES), limits, admit, "capsule top-level census", "capsule_entries")
    names = {name for name, _st in entries}
    if names != TOP_ENTRIES:
        raise _refuse("capsule_entries", f"top-level entries differ: extra={sorted(names - TOP_ENTRIES)} missing={sorted(TOP_ENTRIES - names)}")
    for name, est in entries:
        want = stat.S_ISDIR if name == "source" else stat.S_ISREG
        if not want(est.st_mode):
            raise _refuse("capsule_unsafe", f"{name} has the wrong file type (symlink/special refused)")
        if est.st_mode & 0o222:
            raise _refuse("capsule_writable", f"{name} has write bits")


def verify_capsule_dir(location: str, capsule_id: str, limits: GateLimits | None = None, admit: Admit = _no_admit, *, name_dir: str | None = None,
                       expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Independent read-only verification of the capsule at `location` whose final identity directory is `name_dir` (default `location`). Everything is derived from bytes:
    exact entry/directory/file census, closed manifests, id = sha256(canonical identity) = directory name, source-manifest == identity files, in-place `verify_snapshot`
    (read-only modes, exact set, every byte). Returns the verified record; `source_root` is the directory to use as a LEAF root. Git blob ids are not re-derived here."""
    limits = limits or GateLimits()
    _hex(capsule_id, "capsule id")
    if not os.path.isabs(location) or os.path.normpath(location) != location:
        raise _refuse("capsule_path_invalid", f"{location!r} must be an absolute normalized path")
    if os.path.basename(name_dir or location) != capsule_id:
        raise _refuse("capsule_id_mismatch", f"directory name is not the capsule id {capsule_id}")
    _check_top(location, limits, admit)
    manifest_bytes = _read_file(os.path.join(location, "manifest.json"), limits.max_manifest_bytes, admit, "capsule manifest.json", "capsule_manifest_invalid")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise _refuse("capsule_manifest_sha_mismatch", "manifest.json sha256 differs from the expected value")
    manifest = _closed(_strict_json(manifest_bytes, limits, admit, "capsule manifest.json"), {"schema", "object_id", "identity", "provenance"}, "manifest")
    if manifest["schema"] != CAPSULE_SCHEMA or canonical_bytes(manifest) != manifest_bytes:
        raise _refuse("capsule_manifest_invalid", "manifest.json is not this schema's canonical encoding")
    identity = validate_identity(manifest["identity"], limits)
    if manifest["object_id"] != capsule_id or hashlib.sha256(canonical_bytes(identity)).hexdigest() != capsule_id:
        raise _refuse("capsule_id_mismatch", "the identity digest is not the capsule id")
    prov = _closed(manifest["provenance"], {"head", "tree", "scope_pathspecs", "git_status", "source_digest_algorithm", "source_manifest_sha256"}, "provenance")
    _hex(prov["head"], "provenance.head", _OBJECT_ID)
    _hex(prov["tree"], "provenance.tree", _OBJECT_ID)
    if (prov["scope_pathspecs"] != ["leaf", "pyproject.toml"] or prov["git_status"] != "clean" or prov["source_digest_algorithm"] != SOURCE_DIGEST_ALGORITHM):
        raise _refuse("capsule_manifest_invalid", "provenance scope/status/algorithm record is not the closed value")
    files, source_sha = read_manifest(os.path.join(location, "source-manifest.json"), limits, admit)
    if source_sha != _hex(prov["source_manifest_sha256"], "provenance.source_manifest_sha256"):
        raise _refuse("capsule_manifest_sha_mismatch", "source-manifest.json sha256 differs from the provenance record")
    if files != {f[0]: f[2] for f in identity["files"]}:
        raise _refuse("capsule_manifest_invalid", "source-manifest.json differs from the identity files")
    source = os.path.join(location, "source")
    _dir_census(source, _expected_dirs(identity["files"]), limits, admit)
    snap = verify_snapshot(source, files, limits, admit, manifest_sha256=source_sha)
    # `verify_snapshot`'s tree digest hashes (path, SIZE, sha256) per file; the identity rows must derive the same digest, so every per-file size agrees (not just the total)
    if snap["tree_digest"] != _size_tree_digest({f[0]: (f[1], f[2]) for f in identity["files"]}):
        raise _refuse("snapshot_differs", "per-file source sizes differ from the identity sizes")
    pyproject = _read_file(os.path.join(source, "pyproject.toml"), limits.max_file_bytes, admit, "source/pyproject.toml", "capsule_manifest_invalid")
    try:
        project = tomllib.loads(pyproject.decode("utf-8")).get("project")
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise _refuse("capsule_manifest_invalid", f"source/pyproject.toml: {exc}") from None
    if not isinstance(project, dict) or {"name": project.get("name"), "version": project.get("version")} != {k: identity["project"][k] for k in ("name", "version")}:
        raise _refuse("capsule_manifest_invalid", "source/pyproject.toml disagrees with the identity project")
    return {"object_id": capsule_id, "location": location, "source_root": source, "manifest_sha256": manifest_sha, "source_manifest_sha256": source_sha,
            "tree_digest": identity["tree_digest"], "snapshot_tree_digest": snap["tree_digest"], "package_digest": identity["package_digest"], "file_count": len(files),
            "total_bytes": snap["total_bytes"], "object_format": identity["object_format"], "project": dict(identity["project"]), "provenance": dict(prov),
            "files": {f[0]: f[2] for f in identity["files"]}, "sizes": {f[0]: f[1] for f in identity["files"]}, "blobs": {f[0]: f[3] for f in identity["files"]}, "root_identity": snap["root_identity"]}


def validate_namespace(namespace: str, limits: GateLimits | None = None, admit: Admit = _no_admit, *, max_objects: int = MAX_NAMESPACE_OBJECTS) -> None:
    """Read-only layout check of a capsule namespace: only `objects/`; objects holds only 64-hex capsule directories and `.staging-<run>-leaf` directories (at most `max_objects`,
    enforced while iterating); no symlinks/specials, owned by this user. Unknown content refuses; nothing is repaired."""
    limits = limits or GateLimits()
    if type(max_objects) is not int or max_objects < 1:
        raise _refuse("namespace_invalid", "max_objects must be a positive int")
    if not os.path.isabs(namespace) or os.path.normpath(namespace) != namespace or os.path.realpath(namespace) != namespace:
        raise _refuse("namespace_invalid", f"{namespace!r} must be an absolute, normalized, symlink-free path")
    try:
        st = os.lstat(namespace)
    except OSError as exc:
        raise _refuse("namespace_invalid", f"{namespace}: {exc}") from None
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise _refuse("namespace_invalid", f"{namespace} is not a real directory owned by this user")
    top = _scan_dir(namespace, 1, limits, admit, "capsule namespace census", "namespace_unexpected_entry")
    if not top:
        return
    if top[0][0] != "objects":
        raise _refuse("namespace_unexpected_entry", f"{namespace} has unexpected entry {top[0][0]!r}")
    if not stat.S_ISDIR(top[0][1].st_mode) or top[0][1].st_uid != os.getuid():
        raise _refuse("namespace_invalid", f"{namespace}/objects is not a real directory owned by this user")
    for name, est in _scan_dir(os.path.join(namespace, "objects"), max_objects, limits, admit, "capsule namespace census", "namespace_too_many_objects"):
        if not (_HEX64.match(name) or _STAGING.match(name)) or not stat.S_ISDIR(est.st_mode):
            raise _refuse("namespace_unexpected_entry", f"{namespace}/objects/{name} is not a valid capsule or staging directory")


def verify_capsule(namespace: str, capsule_id: str, limits: GateLimits | None = None, admit: Admit = _no_admit, *, expected_manifest_sha256: str | None = None,
                   max_objects: int = MAX_NAMESPACE_OBJECTS) -> dict[str, Any]:
    """Verify `<namespace>/objects/<capsule_id>` (namespace layout first, then the capsule)."""
    limits = limits or GateLimits()
    _hex(capsule_id, "capsule id")
    validate_namespace(namespace, limits, admit, max_objects=max_objects)
    return verify_capsule_dir(os.path.join(namespace, "objects", capsule_id), capsule_id, limits, admit, expected_manifest_sha256=expected_manifest_sha256)
