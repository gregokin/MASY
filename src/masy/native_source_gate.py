"""Stdlib-only source bootstrap gate of the configured native calendar run (`masy.native_water_run`).

Runs in the PARENT before the SYRUP helper child or the worker exists, and again (same code) in the worker. It imports neither NumPy, SYRUP, MAPLE nor
`masy.native_water_calendar`. It verifies caller-provided READ-ONLY source snapshots in place (no copies): bounded streaming, descriptor-relative, no-follow traversal with
metadata caps enforced before any body is read, fd metadata joins against replacement races, and an exact file-set/hash match with a bounded closed manifest.

What this is NOT: manifest integrity is not scientific acceptance (acceptance stays with the literal package digests, the real census and the cold recheck in the worker);
a development snapshot is never `production_qualified`; origin text is the operator's declaration, separate from verified content and from any real captured repository
metadata (nothing is inferred from an ancestor `.git`). Ancestors of the snapshot root are not required read-only: a trusted operator who can write an ancestor can rename the
root; the recorded root `(st_dev, st_ino)` is re-checked, but a trusted owner's `chmod`, the root user and OS-level sandboxing are outside this contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

__all__ = ["BOOTSTRAP_SCHEMA", "GateLimits", "GateRefusal", "PARENT_CLOSURE", "PRODUCTION_PARENT_CLOSURE", "PRODUCTION_WORKER_ALLOWED", "PRODUCTION_WORKER_REQUIRED", "WORKER_ALLOWED", "WORKER_REQUIRED", "loaded_masy_files", "PRODUCTION_RECIPES", "bootstrap_identity_sha", "build_identity", "check_loaded_closure", "derive_identity", "interpreter_identity",
           "parent_code_record", "production_gate", "read_manifest", "record_manifest", "verify_snapshot"]

BOOTSTRAP_SCHEMA = "masy.native_water_run_bootstrap.v1"
# The CLOSED set of `masy` modules the running parent loads before any helper exists (measured: the real parent CLI loads exactly these five). `__init__` is the package.
PARENT_CLOSURE = ("__init__", "scenario_io", "native_water_run", "native_water_run_config", "native_source_gate")
WORKER_ALLOWED = PARENT_CLOSURE + ("native_water_worker",)
WORKER_REQUIRED = ("__init__", "scenario_io", "native_water_run_config", "native_source_gate", "native_water_worker")
# v2 production closures (explicit siblings; the development tuples above are unchanged). `production_runtime` and `leaf_capsule` are imported lazily by the production branch only.
PRODUCTION_PARENT_CLOSURE = PARENT_CLOSURE + ("production_runtime", "leaf_capsule")
PRODUCTION_WORKER_ALLOWED = PRODUCTION_PARENT_CLOSURE + ("native_water_worker",)
PRODUCTION_WORKER_REQUIRED = WORKER_REQUIRED + ("production_runtime", "leaf_capsule")
_FORBIDDEN_NAMES = ("__pycache__", ".git")
_HEX64 = frozenset("0123456789abcdef")
PRODUCTION_RECIPES: Any = MappingProxyType({})  # profile -> published clean-committed recipe record. EMPTY: 95ef is a historical uncommitted accepted snapshot, not a recipe.


class GateRefusal(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclass(frozen=True)
class GateLimits:
    """Defaults equal the worker census limits (`CensusLimits`) so the gate never admits a tree the child census would refuse."""

    max_files: int = 1024
    max_depth: int = 12
    max_path_chars: int = 200
    max_file_bytes: int = 4 << 20
    max_total_bytes: int = 32 << 20
    max_dir_entries: int = 4096
    chunk_bytes: int = 1 << 20
    max_manifest_bytes: int = 16 << 20
    max_interpreter_bytes: int = 64 << 20
    entry_model_bytes: int = 512  # modelled metadata per traversed entry (names, stat tuple, identities)
    node_model_bytes: int = 256  # modelled retained bytes per parsed JSON node (dict slot + str object)
    max_string_token_bytes: int = 1024
    max_note_chars: int = 512
    max_whitespace_run: int = 1024
    max_whitespace_total: int = 64 * 1024

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if type(value) is not int or value < 1:
                raise GateRefusal("invalid_limits", f"{name} must be a positive int")


Admit = Callable[[int, str], None]


def _no_admit(_n: int, _what: str) -> None:
    return None


def _ident(st: os.stat_result) -> tuple[int, int, int, int]:
    return (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_nlink)


def _valid_rel(rel: Any, limits: GateLimits, where: str) -> str:
    if (type(rel) is not str or not rel or len(rel) > limits.max_path_chars or rel.startswith("/") or "\x00" in rel or "\\" in rel
            or any(p in ("", ".", "..") for p in rel.split("/"))):
        raise GateRefusal("manifest_invalid", f"{where}: unsafe relative path {rel!r}")
    return rel


_SLACK = 16384  # fixed headroom for the scan, exception/traceback objects and the small closed validation structures
_WS = frozenset(b" \t\r\n")
_STRUCT = frozenset(b'{}:,"[]')
_HEXDIGITS = frozenset(b"0123456789abcdefABCDEF")
_ESCAPES = frozenset(b'"\\/bfnrt')
_MAX_BARE = 64


def _preflight(data: bytes, limits: GateLimits, where: str) -> tuple[int, int]:
    """Constant-memory bounded LEXICAL preflight BEFORE `json.loads`: a byte-at-a-time state machine that keeps only integers (no regex, slices, match objects or token
    materialization). String length, whitespace runs/total and bare-token length are enforced WHILE traversing (the first violation stops the scan); arrays and stray bytes refuse;
    nesting depth <= 2; token/colon counts capped from the entry limit. Returns (node count, string bytes) for the exact parse admission. UTF-8 validity is checked by the strict
    decode that runs only after this scan and its admission."""
    n, pos, depth, tokens, colons, nodes, strbytes, ws_total = len(data), 0, 0, 0, 0, 0, 0, 0
    max_tokens, max_str, max_run, max_ws = 4 * limits.max_files + 32, limits.max_string_token_bytes, limits.max_whitespace_run, limits.max_whitespace_total
    ws, struct, hexd, esc = _WS, _STRUCT, _HEXDIGITS, _ESCAPES

    def refuse(why: str) -> None:
        raise GateRefusal("manifest_invalid", f"{where}: {why} (byte {pos})")
    while pos < n:
        c = data[pos]
        if c in ws:
            start = pos
            while pos < n and data[pos] in ws:
                pos += 1
                if pos - start > max_run:
                    refuse(f"a whitespace run longer than {max_run} B")
            ws_total += pos - start
            if ws_total > max_ws:
                refuse(f"more than {max_ws} B of whitespace in total")
            continue
        tokens += 1
        if tokens > max_tokens:
            refuse(f"more than {max_tokens} tokens (entry limit {limits.max_files}); refused before parsing")
        if c == 0x22:  # string: length checked on EVERY byte, before the closing quote is sought
            pos += 1
            length = 0
            while True:
                if pos >= n:
                    refuse("unterminated string")
                c = data[pos]
                if c == 0x22:
                    pos += 1
                    break
                length += 1
                if length > max_str:
                    refuse(f"a string longer than {max_str} B")
                if c < 0x20:
                    refuse("control character inside a string")
                if c == 0x5C:
                    pos += 1
                    if pos >= n:
                        refuse("unterminated escape")
                    e = data[pos]
                    length += 1
                    if e == 0x75:
                        for k in range(1, 5):
                            if pos + k >= n or data[pos + k] not in hexd:
                                refuse("invalid \\u escape")
                        pos += 4
                        length += 4
                    elif e not in esc:
                        refuse("invalid escape")
                pos += 1
            nodes += 1
            strbytes += length + 2
        elif c == 0x7B:
            depth += 1
            nodes += 1
            pos += 1
            if depth > 2:
                refuse("nesting deeper than 2 (envelope -> files)")
        elif c == 0x7D:
            depth -= 1
            pos += 1
            if depth < 0:
                refuse("unbalanced braces")
        elif c == 0x3A:
            colons += 1
            pos += 1
            if colons > limits.max_files + 2:
                refuse(f"more than {limits.max_files + 2} members; refused before parsing")
        elif c == 0x2C:
            pos += 1
        elif c in struct:  # `[` or `]`
            refuse("arrays are not allowed")
        else:  # bare token (number/literal): run capped while scanned
            start = pos
            while pos < n and data[pos] not in ws and data[pos] not in struct:
                pos += 1
                if pos - start > _MAX_BARE:
                    refuse("an over-long bare token")
            nodes += 1
    if depth != 0:
        refuse("unbalanced braces")
    return nodes, strbytes


def read_manifest(path: str, limits: GateLimits, admit: Admit = _no_admit) -> tuple[dict[str, str], str]:
    """Bounded closed manifest read: `lstat` size cap (hard 16 MiB, never raised) BEFORE anything, raw-read admission, no-follow chunked read with an fd join, a bounded LEXICAL
    preflight, the exact parse admission (raw + nodes + string bytes), then strict `json.loads` (no duplicate keys/NaN) and closed structure validation. Accepted shapes: the
    emitted envelope `{"files": {rel: sha}, "note": <short string>}` and a bare `{rel: sha}` map (a top-level `files` whose value is a string is an ordinary file name).
    Returns (files, sha256 of the file bytes)."""
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise GateRefusal("manifest_unreadable", f"{path}: {exc}") from None
    if not stat.S_ISREG(before.st_mode):
        raise GateRefusal("manifest_unreadable", f"{path} is not a regular file")
    if before.st_size > limits.max_manifest_bytes:
        raise GateRefusal("manifest_too_large", f"{path} is {before.st_size} B, over the hard {limits.max_manifest_bytes}-B cap (checked before the body)")
    admit(2 * int(before.st_size) + _SLACK, f"reading the manifest {os.path.basename(path)}")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise GateRefusal("manifest_unreadable", f"{path}: {exc}") from None
    try:
        held = os.fstat(fd)
        if _ident(held)[:2] != _ident(before)[:2] or held.st_size != before.st_size:
            raise GateRefusal("manifest_changed", f"{path} changed while opening")
        data = bytearray(before.st_size)  # ONE raw buffer, filled in place chunk by chunk (no chunk list, no join copy)
        view, total = memoryview(data), 0
        while total < before.st_size:
            got = os.readv(fd, [view[total:total + limits.chunk_bytes]])
            if got == 0:
                break
            total += got
        view.release()
        if total != before.st_size or os.read(fd, 1) != b"" or _ident(os.fstat(fd)) != _ident(held):
            raise GateRefusal("manifest_changed", f"{path} changed while read")
    finally:
        os.close(fd)
    where = str(path)
    nodes, strbytes = _preflight(data, limits, where)
    admit(2 * len(data) + limits.node_model_bytes * nodes + 2 * strbytes + _SLACK, f"parsing the manifest {os.path.basename(path)}")  # raw + decoded text + nodes

    def pairs(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(items)

    def refuse_constant(name):
        raise ValueError(f"non-finite constant {name}")
    try:
        doc = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=refuse_constant)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise GateRefusal("manifest_invalid", f"{where}: {type(exc).__name__}: {exc}") from None
    if not isinstance(doc, dict):
        raise GateRefusal("manifest_invalid", f"{where}: the manifest must be an object")
    if isinstance(doc.get("files"), dict):
        extra = sorted(set(doc) - {"files", "note"})
        if extra:
            raise GateRefusal("manifest_invalid", f"{where}: unknown envelope keys {extra}")
        note = doc.get("note")
        if "note" in doc and (type(note) is not str or len(note) > limits.max_note_chars):
            raise GateRefusal("manifest_invalid", f"{where}: `note` must be a string of at most {limits.max_note_chars} characters")
        files = doc["files"]
    else:
        files = doc
    if not files or len(files) > limits.max_files:
        raise GateRefusal("manifest_invalid", f"{where}: needs 1..{limits.max_files} entries")
    for rel, sha in files.items():
        _valid_rel(rel, limits, where)
        if type(sha) is not str or len(sha) != 64 or not set(sha) <= _HEX64:
            raise GateRefusal("manifest_invalid", f"{where}: {rel!r} is not a lowercase 64-hex sha256")
    return dict(sorted(files.items())), hashlib.sha256(data).hexdigest()


def _walk(root: str, limits: GateLimits, admit: Admit, *, require_readonly: bool) -> tuple[list, dict, tuple]:
    """Pass 1 (metadata only, caps before any body): returns (files [(rel, size)], dir idents {rel: ident}, root ident)."""
    try:
        rootfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise GateRefusal("root_unsafe", f"{root} must be a real directory (not a symlink): {exc}") from None
    files: list[tuple[str, int]] = []
    dirs: dict[str, tuple] = {}
    state = {"total": 0, "entries": 0}

    def visit(fd: int, rel: str, depth: int) -> None:
        st = os.fstat(fd)
        if require_readonly and st.st_mode & 0o222:
            raise GateRefusal("writable_directory", f"{rel or '.'} has write bits (mode {oct(stat.S_IMODE(st.st_mode))}); a read-only sealed snapshot is required")
        dirs[rel] = _ident(st)
        with os.scandir(fd) as it:
            names = sorted(e.name for e in it)
        if len(names) > limits.max_dir_entries:
            raise GateRefusal("too_many_entries", f"{rel or '.'} holds {len(names)} entries (> {limits.max_dir_entries})")
        for name in names:
            sub = f"{rel}/{name}" if rel else name
            if "\x00" in name or "\\" in name or len(sub) > limits.max_path_chars:
                raise GateRefusal("unsafe_name", f"unsafe or over-long entry name {sub[:80]!r}")
            if name in _FORBIDDEN_NAMES or name.endswith(".pyc"):
                raise GateRefusal("cache_artifact", f"{sub}: caches/VCS metadata are not part of a source snapshot (refused, never recursed)")
            state["entries"] += 1
            admit(limits.entry_model_bytes, "source traversal metadata")
            est = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISLNK(est.st_mode):
                raise GateRefusal("symlink", f"{sub} is a symlink; symlinks are refused")
            if stat.S_ISDIR(est.st_mode):
                if depth + 1 > limits.max_depth:
                    raise GateRefusal("too_deep", f"{sub} is deeper than {limits.max_depth}")
                try:
                    cfd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                except OSError as exc:
                    raise GateRefusal("source_changed", f"{sub} cannot be opened as a directory: {exc}") from None
                try:
                    if (os.fstat(cfd).st_dev, os.fstat(cfd).st_ino) != (est.st_dev, est.st_ino):
                        raise GateRefusal("source_changed", f"{sub} was replaced while traversed")
                    visit(cfd, sub, depth + 1)
                finally:
                    os.close(cfd)
            elif stat.S_ISREG(est.st_mode):
                if require_readonly and est.st_mode & 0o222:
                    raise GateRefusal("writable_file", f"{sub} has write bits (mode {oct(stat.S_IMODE(est.st_mode))})")
                if est.st_size > limits.max_file_bytes:
                    raise GateRefusal("file_too_large", f"{sub} is {est.st_size} B (> {limits.max_file_bytes})")
                state["total"] += est.st_size
                if len(files) + 1 > limits.max_files or state["total"] > limits.max_total_bytes:
                    raise GateRefusal("too_many_entries", f"more than {limits.max_files} files or {limits.max_total_bytes} B in total; nothing was read")
                files.append((sub, int(est.st_size)))
            else:
                raise GateRefusal("special_file", f"{sub} is neither a regular file nor a directory")

    try:
        root_ident = _ident(os.fstat(rootfd))
        visit(rootfd, "", 0)
    except OSError as exc:
        raise GateRefusal("source_changed", f"traversal failed: {exc}") from None
    finally:
        os.close(rootfd)
    return files, dirs, root_ident


def _open_member(rootfd: int, rel: str, dirs: dict) -> int:
    """Open `rel` descriptor-relative, every directory component with O_NOFOLLOW and compared with the pass-1 identity (dev, ino)."""
    parts = rel.split("/")
    fd = os.dup(rootfd)
    try:
        cur = ""
        for part in parts[:-1]:
            cur = f"{cur}/{part}" if cur else part
            nfd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = nfd
            if (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != tuple(dirs[cur][:2]):
                raise GateRefusal("source_changed", f"{cur} was replaced after traversal")
        out = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=fd)
        return out
    finally:
        os.close(fd)


def _hash_all(root: str, files: list, dirs: dict, root_ident: tuple, limits: GateLimits, admit: Admit) -> dict[str, tuple[int, str]]:
    admit(2 * limits.chunk_bytes, "source hash buffers")
    rootfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    got: dict[str, tuple[int, str]] = {}
    try:
        if _ident(os.fstat(rootfd))[:2] != root_ident[:2]:
            raise GateRefusal("source_changed", "the snapshot root was replaced after traversal")
        for rel, size in files:
            try:
                fd = _open_member(rootfd, rel, dirs)
            except OSError as exc:
                raise GateRefusal("source_changed", f"{rel}: {exc}") from None
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_size != size:
                    raise GateRefusal("source_changed", f"{rel} changed after traversal")
                digest, total = hashlib.sha256(), 0
                while True:
                    chunk = os.read(fd, limits.chunk_bytes)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > size:
                        raise GateRefusal("source_changed", f"{rel} grew while hashed")
                    digest.update(chunk)
                after = os.fstat(fd)
                if total != size or _ident(after)[:3] != _ident(before)[:3] or after.st_size != size:
                    raise GateRefusal("source_changed", f"{rel} changed while hashed")
            finally:
                os.close(fd)
            got[rel] = (size, digest.hexdigest())
        # every directory re-checked: entries added/removed/replaced during the hashing change mtime/nlink/ino
        for rel, ident in dirs.items():
            fd = rootfd if rel == "" else _open_dir(rootfd, rel, dirs)
            try:
                if _ident(os.fstat(fd)) != tuple(ident):
                    raise GateRefusal("source_changed", f"{rel or '.'} changed while the snapshot was verified")
            finally:
                if rel != "":
                    os.close(fd)
    finally:
        os.close(rootfd)
    try:
        after_root = os.lstat(root)
    except OSError as exc:
        raise GateRefusal("source_changed", f"the snapshot root vanished: {exc}") from None
    if (after_root.st_dev, after_root.st_ino) != tuple(root_ident[:2]):
        raise GateRefusal("source_changed", "the snapshot root path now names a different directory")
    return got


def _open_dir(rootfd: int, rel: str, dirs: dict) -> int:
    fd = os.dup(rootfd)
    cur = ""
    try:
        for part in rel.split("/"):
            cur = f"{cur}/{part}" if cur else part
            nfd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = nfd
        return fd
    except OSError as exc:
        os.close(fd)
        raise GateRefusal("source_changed", f"{rel}: {exc}") from None


def _tree_digest(entries: dict[str, tuple[int, str]]) -> str:
    h = hashlib.sha256()
    for rel in sorted(entries):
        size, sha = entries[rel]
        h.update(f"{rel}\x00{size}\x00{sha}\n".encode())
    return h.hexdigest()


def verify_snapshot(root: str, manifest: dict[str, str], limits: GateLimits, admit: Admit = _no_admit, *, manifest_sha256: str | None = None) -> dict[str, Any]:
    """Verify the read-only snapshot `root` against the bounded manifest, in place. Refuses before any body for metadata violations; the file set must equal the manifest exactly."""
    admit(limits.entry_model_bytes * (len(manifest) + 1), "source manifest records")
    files, dirs, root_ident = _walk(root, limits, admit, require_readonly=True)
    found = {rel for rel, _ in files}
    extra, missing = sorted(found - set(manifest)), sorted(set(manifest) - found)
    if extra or missing:
        raise GateRefusal("snapshot_differs", f"file set differs from the manifest: extra={extra[:5]} missing={missing[:5]}")
    got = _hash_all(root, files, dirs, root_ident, limits, admit)
    changed = sorted(rel for rel, (_s, sha) in got.items() if sha != manifest[rel])
    if changed:
        raise GateRefusal("snapshot_differs", f"content differs from the manifest: {changed[:5]}")
    return {"root": root, "root_identity": [root_ident[0], root_ident[1]], "tree_digest": _tree_digest(got), "file_count": len(got), "total_bytes": sum(s for s, _ in got.values()),
            "manifest_sha256": manifest_sha256, "files": {rel: sha for rel, (_s, sha) in got.items()}}


def record_manifest(root: str, limits: GateLimits | None = None, admit: Admit = _no_admit) -> dict[str, Any]:
    """DEVELOPMENT/operator helper (replaces the old whole-file `os.walk` recorder): the same bounded no-follow streaming traversal, but writable trees are allowed (it is
    run before sealing). It records the tree as found; the result is declarative and never an acceptance authority."""
    limits = limits or GateLimits()
    files, dirs, root_ident = _walk(root, limits, admit, require_readonly=False)
    got = _hash_all(root, files, dirs, root_ident, limits, admit)
    return {"files": {rel: sha for rel, (_s, sha) in sorted(got.items())}, "note": "development snapshot manifest: a declarative expectation recorded from the tree as found; NOT a qualification authority"}


def interpreter_identity(limits: GateLimits, path: str | None = None, admit: Admit = _no_admit) -> dict[str, Any]:
    real = os.path.realpath(path or sys.executable)
    try:
        st = os.lstat(real)
        if not stat.S_ISREG(st.st_mode) or st.st_size > limits.max_interpreter_bytes:
            raise GateRefusal("interpreter_unsafe", f"{real} is not a regular file within {limits.max_interpreter_bytes} B")
        admit(2 * limits.chunk_bytes, "interpreter hash buffers")
        digest, total = hashlib.sha256(), 0
        fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            while True:
                chunk = os.read(fd, limits.chunk_bytes)
                if not chunk:
                    break
                total += len(chunk)
                if total > limits.max_interpreter_bytes:
                    raise GateRefusal("interpreter_unsafe", "interpreter grew while hashed")
                digest.update(chunk)
        finally:
            os.close(fd)
    except OSError as exc:
        raise GateRefusal("interpreter_unsafe", f"{real}: {exc}") from None
    return {"realpath": real, "sha256": digest.hexdigest(), "size": total}


def parent_code_record(masy_manifest: dict[str, str]) -> dict[str, Any]:
    """The sealed sha256 of the CLOSED parent-side closure (`PARENT_CLOSURE`, from the VERIFIED manifest). Parent and worker both derive this same record."""
    sealed = {}
    for name in PARENT_CLOSURE:
        key = _module_key(name)
        if key not in masy_manifest:
            raise GateRefusal("parent_code_mismatch", f"the sealed MASY snapshot has no {key}")
        sealed[name] = masy_manifest[key]
    digest = hashlib.sha256("".join(f"{n}\x00{sealed[n]}\n" for n in PARENT_CLOSURE).encode()).hexdigest()
    return {"modules": sealed, "digest": digest}


def _module_key(name: str) -> str:
    return "src/masy/__init__.py" if name == "__init__" else f"src/masy/{name}.py"


def loaded_masy_files() -> dict[str, str]:
    """Every module this process actually has loaded under `masy` (`masy`, `masy.*`, and `__main__` when run as `-m masy.x`): {sealed-manifest key: loaded file path}."""
    out: dict[str, str] = {}
    main = sys.modules.get("__main__")
    main_name = getattr(getattr(main, "__spec__", None), "name", None)
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if name == "__main__" and main_name and (main_name == "masy" or main_name.startswith("masy.")):
            name = main_name
        if name != "masy" and not name.startswith("masy."):
            continue
        path = getattr(mod, "__file__", None)
        if not path:
            raise GateRefusal("parent_code_mismatch", f"{name} is loaded without a source file")
        rel = name.split(".")[1:]
        if not rel:
            key = "src/masy/__init__.py"
        elif hasattr(mod, "__path__"):
            key = "src/masy/" + "/".join(rel) + "/__init__.py"
        else:
            key = "src/masy/" + "/".join(rel) + ".py"
        out[key] = path
    return out


def _stream_hash(path: str, limits: GateLimits, admit: Admit, what: str) -> str:
    """Bounded streamed SHA-256 of one loaded source file: fresh admission BEFORE the first body read, no-follow open, size cap, chunked reads with an fd metadata join."""
    admit(2 * limits.chunk_bytes, f"hashing {what}")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise GateRefusal("parent_code_mismatch", f"{path}: {exc}") from None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limits.max_file_bytes:
            raise GateRefusal("parent_code_mismatch", f"{path} is not a regular file within {limits.max_file_bytes} B")
        digest, total = hashlib.sha256(), 0
        while True:
            chunk = os.read(fd, limits.chunk_bytes)
            if not chunk:
                break
            total += len(chunk)
            if total > before.st_size:
                raise GateRefusal("parent_code_mismatch", f"{path} grew while hashed")
            digest.update(chunk)
        after = os.fstat(fd)
        if total != before.st_size or _ident(after)[:3] != _ident(before)[:3]:
            raise GateRefusal("parent_code_mismatch", f"{path} changed while hashed")
    finally:
        os.close(fd)
    return digest.hexdigest()


def check_loaded_closure(masy_manifest: dict[str, str], limits: GateLimits, admit: Admit, *, allowed: tuple[str, ...], required: tuple[str, ...]) -> dict[str, str]:
    """The running process's loaded `masy` modules must be inside the CLOSED `allowed` set (`parent_code_unexpected_module`), include every `required` module, and every loaded
    file's streamed hash must equal the verified sealed manifest entry (`parent_code_mismatch`). Returns {manifest key: sha}."""
    files = loaded_masy_files()
    allowed_keys = {_module_key(n) for n in allowed}
    unexpected = sorted(set(files) - allowed_keys)
    if unexpected:
        raise GateRefusal("parent_code_unexpected_module", f"masy modules outside the closed pre-helper closure are loaded: {unexpected[:5]}")
    missing = sorted(_module_key(n) for n in required if _module_key(n) not in files)
    if missing:
        raise GateRefusal("parent_code_mismatch", f"required masy modules are not loaded from a source file: {missing[:5]}")
    got = {}
    for key in sorted(files):
        got[key] = _stream_hash(files[key], limits, admit, key)
        if masy_manifest.get(key) != got[key]:
            raise GateRefusal("parent_code_mismatch", f"the running {key} differs from the selected sealed MASY snapshot")
    return got


def build_identity(*, mode: str, config_sha256: str, profile: str, native_target: str, origin: dict[str, Any], masy: dict[str, Any], syrup: dict[str, Any],
                   interpreter: dict[str, Any], parent_code: dict[str, Any]) -> dict[str, Any]:
    """The CLOSED bootstrap identity (every field required; canonical JSON hashed by `bootstrap_identity_sha`)."""
    def tree(rec: dict[str, Any]) -> dict[str, Any]:
        return {k: rec[k] for k in ("tree_digest", "manifest_sha256", "file_count", "total_bytes", "root_identity")}
    return {"schema": BOOTSTRAP_SCHEMA, "mode": mode, "production_qualified": False, "config_sha256": config_sha256, "profile": profile, "native_target": native_target,
            "origin": {"declared": dict(origin), "verified_repository_metadata": None, "note": "declared by the operator; no repository identity is inferred"},
            "parent_code": parent_code, "interpreter": interpreter, "masy": tree(masy), "syrup": tree(syrup)}


def bootstrap_identity_sha(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def derive_identity(source: dict[str, Any], config_sha256: str, limits: GateLimits, admit: Admit, *, allowed: tuple[str, ...] = PARENT_CLOSURE,
                    required: tuple[str, ...] = PARENT_CLOSURE) -> dict[str, Any]:
    """Verify both development snapshots and build the closed bootstrap identity from THIS process's own reads. `allowed`/`required` define the CLOSED set of `masy` modules this process
    may have loaded; every loaded file's bytes must equal the sealed MASY manifest entry (parent: PARENT_CLOSURE; worker: WORKER_ALLOWED/WORKER_REQUIRED)."""
    masy_files, masy_sha = read_manifest(source["masy_manifest"], limits, admit)
    syrup_files, syrup_sha = read_manifest(source["syrup_manifest"], limits, admit)
    if "src/masy/__init__.py" not in masy_files or any(k.startswith("src/maple_syrup/") for k in masy_files):
        raise GateRefusal("wrong_source_family", "source.masy_manifest does not describe a MASY source snapshot")
    if "src/maple_syrup/__init__.py" not in syrup_files or any(k.startswith("src/masy/") for k in syrup_files):
        raise GateRefusal("wrong_source_family", "source.syrup_manifest does not describe a SYRUP source snapshot")
    masy = verify_snapshot(source["masy_root"], masy_files, limits, admit, manifest_sha256=masy_sha)
    syrup = verify_snapshot(source["syrup_root"], syrup_files, limits, admit, manifest_sha256=syrup_sha)
    check_loaded_closure(masy_files, limits, admit, allowed=allowed, required=required)
    admit(4096, "bootstrap identity record")
    return build_identity(mode=source["mode"], config_sha256=config_sha256, profile=source["profile"], native_target=source["native_target"], origin=source["origin"], masy=masy,
                          syrup=syrup, interpreter=interpreter_identity(limits, admit=admit), parent_code=parent_code_record(masy_files))


def production_gate(profile: str, store: str | None = None, lock: str | None = None) -> None:
    """`production_committed`: no published clean-committed recipe exists for any profile (95ef is a historical uncommitted accepted snapshot; the final Phase 1 source will get
    a NEW recipe after its qualification), so this refuses typed BEFORE any helper or worker and NEVER falls back to a development snapshot. Once a recipe is published its
    positive integration must consume the existing `masy.runtime` verification through its `-I -B` launcher (`tools/masy_runtime.py` + a store run directory); the existing
    `runtime.verify` cannot be called in-process from this CLI, which is deferred until matching final bytes are published."""
    if profile not in PRODUCTION_RECIPES:
        raise GateRefusal("production_recipe_unavailable", f"no published clean-committed production recipe exists for profile {profile!r}; production_committed never falls back to a development snapshot")
    raise GateRefusal("production_verify_not_integrated", "a recipe exists but the positive production verification path is not integrated in this launcher")
