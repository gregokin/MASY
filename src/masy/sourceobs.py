"""Stdlib-only observation of the selected source pair and of MASY itself.

Nothing here imports MAPLE or SYRUP code. The preflight inserts upstream
`src` roots on `sys.path` only after `observe_sources` has passed.

For each upstream (`root` must be the repository top level):

- HEAD is the current local committed revision (`rev-parse --verify
  HEAD^{commit}`); an optional `expected_revision` refuses on mismatch.
- `git status --porcelain=v1 -z --untracked-files=all -- src pyproject.toml`
  must print nothing (tracked edits, deletions, untracked files).
- The HEAD tree entries in that scope (`ls-tree -r -z HEAD`) must all be
  regular blobs (no symlinks or submodules), and every committed file must be
  present on disk with the same git blob id. This ties the observed content to
  the committed revision independently of the index.
- The ACTUAL filesystem under `src/` is enumerated with `lstat`. Git's
  tracked list alone omits ignored files, which Python would still import.
  Only `__pycache__` directories are excluded (the launcher redirects
  bytecode reads/writes to a run-owned `sys.pycache_prefix`, so these are
  never read). Any other file that is not committed refuses, except
  generated distribution metadata directly under `src/` (`*.egg-info`,
  `*.dist-info`), which is hashed and recorded, never repaired. Symlinks and
  special files refuse. File-count and byte bounds apply during enumeration
  and before each file is opened.
- `pyproject.toml` is hashed separately and must name the expected project.

Every git call disables optional locks and fsmonitor, drops inherited
`GIT_*` environment variables, and is bounded in wall time and in combined
stdout+stderr bytes.
Nothing here writes to any repository.

MASY has no Git repository and none is fabricated: its `src/masy/` files,
the launcher and `pyproject.toml` (if present) are hashed as an observation,
guarded only by an explicit lock (`--expect-lock`).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import stat
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "CACHE_DIR_NAMES",
    "DIGEST_ALGORITHM",
    "SCOPE_PATHSPECS",
    "Limits",
    "SourceRefusal",
    "canonical_json",
    "compare_lock",
    "diff_observations",
    "load_lock",
    "observe_masy",
    "observe_sources",
    "observe_upstream",
    "run_git",
]

GIT_EXECUTABLE = "git"
# Identical to maple_syrup.provenance.SOURCE_DIGEST_ALGORITHM so the package
# digest can be cross-checked against the upstream helper after import.
DIGEST_ALGORITHM = "sha256(sorted relpath NUL file_sha256 LF), excl __pycache__/*.pyc/*.pyo; v1"
SCOPE_PATHSPECS = ("src", "pyproject.toml")
CACHE_DIR_NAMES = frozenset({"__pycache__"})
GENERATED_METADATA_SUFFIXES = (".egg-info", ".dist-info")
MAX_REPORTED_ENTRIES = 50
_READ_CHUNK = 1 << 20
_STDERR_KEEP = 2000
_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REGULAR_MODES = frozenset({"100644", "100755"})


class SourceRefusal(Exception):
    """A source guard failed. `code` is stable; `evidence` is JSON-safe."""

    def __init__(self, code: str, detail: str, **evidence: Any) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.evidence = evidence


@dataclass(frozen=True)
class Limits:
    git_timeout_s: float = 30.0
    git_max_output_bytes: int = 8 * 1024 * 1024
    max_files: int = 5_000
    max_bytes: int = 64 * 1024 * 1024
    max_lock_bytes: int = 4 * 1024 * 1024


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


# --------------------------------------------------------------------------- git


def run_git(root: str | os.PathLike[str], args: list[str], limits: Limits) -> bytes:
    """Run one read-only git command; return stdout or raise SourceRefusal.

    stdout and stderr are both pipes read while the child runs, and
    `git_max_output_bytes` is a hard cap on their COMBINED size (stderr-only
    output counts). On timeout or cap refusal the child is killed and reaped.
    """
    command = [GIT_EXECUTABLE, "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(root), *args]
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0", LC_ALL="C")
    deadline = time.monotonic() + limits.git_timeout_s
    try:
        proc = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        )
    except OSError as exc:
        raise SourceRefusal(
            "git_unavailable", f"cannot run {GIT_EXECUTABLE!r}: {type(exc).__name__}: {exc}",
            command=command,
        ) from exc
    out_chunks: list[bytes] = []
    err_kept = b""
    total = 0
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
            selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SourceRefusal(
                        "git_timeout", f"git exceeded {limits.git_timeout_s} s", command=command,
                    )
                for key, _ in selector.select(remaining):
                    chunk = os.read(key.fileobj.fileno(), _READ_CHUNK)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    total += len(chunk)
                    if total > limits.git_max_output_bytes:
                        raise SourceRefusal(
                            "git_output_exceeded",
                            f"git printed more than {limits.git_max_output_bytes} bytes (stdout+stderr)",
                            command=command,
                        )
                    if key.data == "stdout":
                        out_chunks.append(chunk)
                    elif len(err_kept) < _STDERR_KEEP:
                        err_kept += chunk[:_STDERR_KEEP - len(err_kept)]
        try:
            returncode = proc.wait(timeout=max(deadline - time.monotonic(), 0.001))
        except subprocess.TimeoutExpired as exc:
            raise SourceRefusal(
                "git_timeout", f"git exceeded {limits.git_timeout_s} s", command=command,
            ) from exc
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        proc.stderr.close()
    if returncode != 0:
        reason = err_kept.decode("utf-8", errors="replace").strip()
        raise SourceRefusal(
            "git_failed", f"git exited {returncode}: {reason}", command=command,
        )
    return b"".join(out_chunks)


def _parse_porcelain_z(output: bytes) -> list[dict[str, str]]:
    tokens = output.decode("utf-8", errors="surrogateescape").split("\0")
    entries: list[dict[str, str]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        entry = {"code": token[:2], "path": token[3:]}
        if token[0] in "RC" or token[1] in "RC":
            entry["original_path"] = tokens[index] if index < len(tokens) else ""
            index += 1
        entries.append(entry)
    return entries


def _parse_ls_tree_z(output: bytes) -> dict[str, tuple[str, str, str]]:
    tracked: dict[str, tuple[str, str, str]] = {}
    for record in output.decode("utf-8", errors="surrogateescape").split("\0"):
        if not record:
            continue
        meta, sep, path = record.partition("\t")
        fields = meta.split(" ")
        if not sep or len(fields) != 3:
            raise SourceRefusal("git_output_unparseable", f"unexpected ls-tree record {record[:200]!r}")
        tracked[path] = (fields[0], fields[1], fields[2])
    return tracked


# ------------------------------------------------------------------ filesystem


class _Budget:
    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        self.files = 0
        self.bytes = 0

    def admit(self, path: str, size: int) -> int:
        if self.files >= self.limits.max_files:
            raise SourceRefusal(
                "max_files_exceeded", f"more than max_files={self.limits.max_files} files in scope (at {path})",
            )
        remaining = self.limits.max_bytes - self.bytes
        if size > remaining:
            raise SourceRefusal(
                "max_bytes_exceeded",
                f"more than max_bytes={self.limits.max_bytes} bytes in scope (next file {path} is {size} bytes)",
            )
        return remaining

    def consume(self, size: int) -> None:
        self.files += 1
        self.bytes += size


def _hash_file(path: Path, expected_size: int, byte_budget: int, blob_algorithm: str | None) -> tuple[str, str | None]:
    """sha256 (and optional git blob id) of one regular file, without following
    a symlink, re-checked on the open descriptor, reading at most the budget."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SourceRefusal("unreadable_source_file", f"cannot open {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise SourceRefusal("non_regular_file", f"{path} is not a regular file")
        sha = hashlib.sha256()
        blob = hashlib.new(blob_algorithm) if blob_algorithm else None
        if blob is not None:
            blob.update(b"blob %d\0" % expected_size)
        read = 0
        while True:
            chunk = os.read(fd, min(_READ_CHUNK, byte_budget - read + 1))
            if not chunk:
                break
            read += len(chunk)
            if read > byte_budget:
                raise SourceRefusal("max_bytes_exceeded", f"{path} grew beyond the byte budget while read")
            sha.update(chunk)
            if blob is not None:
                blob.update(chunk)
    finally:
        os.close(fd)
    if read != expected_size:
        raise SourceRefusal(
            "source_changed_during_read", f"{path} changed size during read ({expected_size} -> {read})",
        )
    return sha.hexdigest(), (blob.hexdigest() if blob is not None else None)


def _lstat(path: Path, missing_code: str) -> os.stat_result:
    try:
        return os.lstat(path)
    except FileNotFoundError as exc:
        raise SourceRefusal(missing_code, f"{path} does not exist") from exc
    except OSError as exc:
        raise SourceRefusal("unreadable_source_file", f"cannot stat {path}: {exc}") from exc


def _walk_scope(root: Path, relative_dir: str, limits: Limits) -> list[tuple[str, os.stat_result]]:
    """Every file under root/relative_dir except pruned cache directories.

    Symlinks (file or directory) and special files refuse. Enumeration stops
    with a refusal as soon as the file count exceeds `max_files`.
    """
    base = root / relative_dir
    info = _lstat(base, "scope_missing")
    if stat.S_ISLNK(info.st_mode):
        raise SourceRefusal("symlink_in_scope", f"{base} is a symlink")
    if not stat.S_ISDIR(info.st_mode):
        raise SourceRefusal("scope_missing", f"{base} is not a directory")
    found: list[tuple[str, os.stat_result]] = []
    pending = [base]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as iterator:
                names = sorted(entry.name for entry in iterator)
        except OSError as exc:
            raise SourceRefusal("unreadable_source_file", f"cannot list {current}: {exc}") from exc
        for name in names:
            path = current / name
            entry_info = _lstat(path, "source_changed_during_read")
            if stat.S_ISLNK(entry_info.st_mode):
                raise SourceRefusal("symlink_in_scope", f"{path} is a symlink; symlinks are refused in scope")
            if stat.S_ISDIR(entry_info.st_mode):
                if name not in CACHE_DIR_NAMES:
                    pending.append(path)
                continue
            if not stat.S_ISREG(entry_info.st_mode):
                raise SourceRefusal("non_regular_file", f"{path} is not a regular file")
            found.append((path.relative_to(root).as_posix(), entry_info))
            if len(found) > limits.max_files:
                raise SourceRefusal(
                    "max_files_exceeded", f"more than max_files={limits.max_files} files under {base}",
                )
    found.sort(key=lambda item: item[0])
    return found


def _read_regular_bytes(path: Path, max_bytes: int, missing_code: str) -> bytes:
    info = _lstat(path, missing_code)
    if stat.S_ISLNK(info.st_mode):
        raise SourceRefusal("symlink_in_scope", f"{path} is a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise SourceRefusal("non_regular_file", f"{path} is not a regular file")
    if info.st_size > max_bytes:
        raise SourceRefusal("max_bytes_exceeded", f"{path} is larger than {max_bytes} bytes")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SourceRefusal("unreadable_source_file", f"cannot open {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise SourceRefusal("non_regular_file", f"{path} is not a regular file")
        data = b""
        while len(data) <= max_bytes:
            chunk = os.read(fd, min(_READ_CHUNK, max_bytes - len(data) + 1))
            if not chunk:
                break
            data += chunk
    finally:
        os.close(fd)
    if len(data) > max_bytes:
        raise SourceRefusal("max_bytes_exceeded", f"{path} grew beyond {max_bytes} bytes while read")
    if len(data) != info.st_size:
        raise SourceRefusal("source_changed_during_read", f"{path} changed size during read")
    return data


def _tree_digest(entries: list[tuple[str, str]]) -> str:
    hasher = hashlib.sha256()
    for relative, file_hash in sorted(entries):
        hasher.update(relative.encode("utf-8", errors="surrogateescape"))
        hasher.update(b"\0")
        hasher.update(file_hash.encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _canonical_root(root: str) -> Path:
    if not os.path.isabs(root):
        raise SourceRefusal("source_root_not_absolute", f"{root!r} is not an absolute path")
    real = os.path.realpath(root)
    if real != os.path.normpath(root):
        raise SourceRefusal("source_root_not_canonical", f"{root!r} resolves through a symlink to {real!r}")
    if not os.path.isdir(real):
        raise SourceRefusal("source_root_missing", f"{root!r} is not a directory")
    return Path(real)


def _is_generated_metadata(relative: str) -> bool:
    parts = relative.split("/")
    return len(parts) >= 3 and parts[0] == "src" and parts[1].endswith(GENERATED_METADATA_SUFFIXES)


def _pyproject_record(data: bytes, path: Path) -> dict[str, Any]:
    try:
        project = tomllib.loads(data.decode("utf-8")).get("project", {})
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SourceRefusal("pyproject_unreadable", f"{path}: {exc}") from exc
    if not isinstance(project, dict):
        raise SourceRefusal("pyproject_unreadable", f"{path}: [project] is not a table")
    return {
        "path": "pyproject.toml",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "project_name": project.get("name"),
        "project_version": project.get("version"),
    }


# -------------------------------------------------------------------- observers


def observe_upstream(
    name: str,
    root: str,
    *,
    package: str,
    project_name: str,
    expected_revision: str | None,
    limits: Limits,
) -> dict[str, Any]:
    """Observe one upstream checkout; raise SourceRefusal on any guard failure."""
    real = _canonical_root(root)
    toplevel = run_git(real, ["rev-parse", "--show-toplevel"], limits).decode("utf-8", "surrogateescape").strip()
    if os.path.realpath(toplevel) != str(real):
        raise SourceRefusal(
            "not_repository_root", f"{real} is not its repository top level ({toplevel})",
        )
    head = run_git(real, ["rev-parse", "--verify", "HEAD^{commit}"], limits).decode("ascii", "replace").strip()
    if not _OBJECT_ID.match(head):
        raise SourceRefusal("head_unreadable", f"HEAD did not resolve to an object id: {head[:100]!r}")
    if expected_revision is not None and head != expected_revision:
        raise SourceRefusal(
            "expected_revision_mismatch",
            f"current HEAD {head} differs from expected_revision {expected_revision}",
            head=head, expected_revision=expected_revision,
        )

    status = _parse_porcelain_z(run_git(
        real,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none", "--",
         *SCOPE_PATHSPECS],
        limits,
    ))
    if status:
        raise SourceRefusal(
            "dirty_scope", f"{len(status)} uncommitted change(s) in {list(SCOPE_PATHSPECS)}",
            entry_count=len(status), entries=status[:MAX_REPORTED_ENTRIES],
        )

    tracked = _parse_ls_tree_z(run_git(real, ["ls-tree", "-r", "-z", "HEAD", "--", *SCOPE_PATHSPECS], limits))
    for path, (mode, _kind, _oid) in tracked.items():
        if mode == "120000":
            raise SourceRefusal("symlink_in_scope", f"committed {path} is a symlink")
        if mode not in _REGULAR_MODES:
            raise SourceRefusal("unsupported_tracked_entry", f"committed {path} has mode {mode}")
    if "pyproject.toml" not in tracked:
        raise SourceRefusal("pyproject_not_committed", f"{real}/pyproject.toml is not in HEAD")
    if f"src/{package}/__init__.py" not in tracked:
        raise SourceRefusal("package_not_committed", f"src/{package}/__init__.py is not in HEAD")
    blob_algorithm = "sha1" if len(head) == 40 else "sha256"

    budget = _Budget(limits)
    pyproject_path = real / "pyproject.toml"
    pyproject_bytes = _read_regular_bytes(pyproject_path, limits.max_bytes, "pyproject_missing")
    budget.admit("pyproject.toml", len(pyproject_bytes))
    budget.consume(len(pyproject_bytes))
    pyproject = _pyproject_record(pyproject_bytes, pyproject_path)
    blob = hashlib.new(blob_algorithm, b"blob %d\0" % len(pyproject_bytes) + pyproject_bytes).hexdigest()
    if blob != tracked["pyproject.toml"][2]:
        raise SourceRefusal("content_differs_from_head", "pyproject.toml differs from HEAD")
    pyproject["git_blob"] = blob
    if pyproject["project_name"] != project_name:
        raise SourceRefusal(
            "pyproject_project_mismatch",
            f"pyproject.toml names project {pyproject['project_name']!r}, expected {project_name!r}",
        )

    hashed: dict[str, str] = {"pyproject.toml": pyproject["sha256"]}
    sizes: dict[str, int] = {"pyproject.toml": len(pyproject_bytes)}
    generated: list[dict[str, Any]] = []
    for relative, info in _walk_scope(real, "src", limits):
        remaining = budget.admit(relative, info.st_size)
        in_head = relative in tracked
        file_hash, file_blob = _hash_file(
            real / relative, info.st_size, remaining, blob_algorithm if in_head else None,
        )
        budget.consume(info.st_size)
        if in_head:
            if file_blob != tracked[relative][2]:
                raise SourceRefusal("content_differs_from_head", f"{relative} differs from HEAD")
            hashed[relative] = file_hash
            sizes[relative] = info.st_size
        elif _is_generated_metadata(relative):
            generated.append({"path": relative, "sha256": file_hash, "size": info.st_size})
        else:
            raise SourceRefusal(
                "untracked_runtime_input",
                f"{relative} is inside the runtime scope but not committed (ignored or untracked)",
                path=relative,
            )
    missing = sorted(set(tracked) - set(hashed))
    if missing:
        raise SourceRefusal(
            "tracked_file_missing", f"{len(missing)} committed file(s) not observed on disk",
            paths=missing[:MAX_REPORTED_ENTRIES],
        )

    prefix = f"src/{package}/"
    package_entries = [(rel[len(prefix):], digest) for rel, digest in hashed.items() if rel.startswith(prefix)]
    return {
        "name": name,
        "root": str(real),
        "selection": "current_local_committed_head",
        "head": head,
        "object_format": blob_algorithm,
        "expected_revision": expected_revision,
        "scope_pathspecs": list(SCOPE_PATHSPECS),
        "git_status": "clean",
        "cache_exclusions": {"directory_names": sorted(CACHE_DIR_NAMES)},
        "pyproject": pyproject,
        "package": {
            "name": package,
            "dir": str(real / "src" / package),
            "algorithm": DIGEST_ALGORITHM,
            "digest_sha256": _tree_digest(package_entries),
            "file_count": len(package_entries),
            "total_bytes": sum(sizes[prefix + rel] for rel, _ in package_entries),
        },
        "tracked_scope": {
            "algorithm": DIGEST_ALGORITHM,
            "digest_sha256": _tree_digest(list(hashed.items())),
            "file_count": len(hashed),
            "total_bytes": sum(sizes.values()),
        },
        "generated_metadata": {
            "policy": "recorded_not_repaired",
            "entries": generated,
            "digest_sha256": _tree_digest([(item["path"], item["sha256"]) for item in generated]),
        },
    }


def observe_masy(masy_root: str, launcher: str, limits: Limits) -> dict[str, Any]:
    """Hash MASY's own runtime files; there is no Git repository to consult."""
    real = _canonical_root(masy_root)
    launcher_real = Path(os.path.realpath(launcher))
    if launcher_real != real / "tools" / "masy_preflight.py":
        raise SourceRefusal(
            "launcher_outside_masy_root", f"launcher {launcher_real} is not {real}/tools/masy_preflight.py",
        )
    budget = _Budget(limits)
    files = _walk_scope(real, "src/masy", limits)
    files.append(("tools/masy_preflight.py", _lstat(launcher_real, "launcher_missing")))
    pyproject_present = os.path.lexists(real / "pyproject.toml")
    if pyproject_present:
        files.append(("pyproject.toml", _lstat(real / "pyproject.toml", "pyproject_missing")))
    hashed: list[tuple[str, str]] = []
    total = 0
    for relative, info in files:
        if stat.S_ISLNK(info.st_mode):
            raise SourceRefusal("symlink_in_scope", f"{real / relative} is a symlink")
        remaining = budget.admit(relative, info.st_size)
        file_hash, _ = _hash_file(real / relative, info.st_size, remaining, None)
        budget.consume(info.st_size)
        hashed.append((relative, file_hash))
        total += info.st_size
    by_path = dict(hashed)
    prefix = "src/masy/"
    package_entries = [(rel[len(prefix):], digest) for rel, digest in hashed if rel.startswith(prefix)]
    return {
        "name": "masy",
        "root": str(real),
        "selection": "working_files_no_git_repository",
        "guard": "explicit_lock_only",
        "cache_exclusions": {"directory_names": sorted(CACHE_DIR_NAMES)},
        "launcher": {"path": "tools/masy_preflight.py", "sha256": by_path["tools/masy_preflight.py"]},
        "pyproject": {"path": "pyproject.toml", "sha256": by_path["pyproject.toml"]} if pyproject_present else None,
        "package": {
            "name": "masy",
            "dir": str(real / "src" / "masy"),
            "algorithm": DIGEST_ALGORITHM,
            "digest_sha256": _tree_digest(package_entries),
            "file_count": len(package_entries),
        },
        "files_digest": {
            "algorithm": DIGEST_ALGORITHM,
            "digest_sha256": _tree_digest(hashed),
            "file_count": len(hashed),
            "total_bytes": total,
        },
    }


def observe_sources(config: Any, launcher: str) -> dict[str, Any]:
    """Observe MASY, MAPLE and SYRUP in that order; first refusal propagates."""
    limits: Limits = config.limits
    observation: dict[str, Any] = {}
    steps = [("masy", lambda: observe_masy(config.masy_root, launcher, limits))]
    for source in config.upstreams:
        steps.append((source.name, lambda source=source: observe_upstream(
            source.name, source.root, package=source.package, project_name=source.project_name,
            expected_revision=source.expected_revision, limits=limits,
        )))
    for name, step in steps:
        try:
            observation[name] = step()
        except SourceRefusal as exc:
            exc.evidence.setdefault("source", name)
            raise
    return observation


def diff_observations(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Dotted keys (two levels) whose canonical JSON differs."""
    differences: list[str] = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if canonical_json(old) == canonical_json(new):
            continue
        if isinstance(old, dict) and isinstance(new, dict):
            for sub in sorted(set(old) | set(new)):
                if canonical_json(old.get(sub)) != canonical_json(new.get(sub)):
                    differences.append(f"{key}.{sub}")
        else:
            differences.append(key)
    return differences


def load_lock(path: str, limits: Limits) -> dict[str, Any]:
    """A lock is exactly a previously reported `source_observation` block."""
    try:
        data = _read_regular_bytes(Path(path), limits.max_lock_bytes, "lock_unreadable")
        lock = json.loads(data.decode("utf-8"))
    except SourceRefusal as exc:
        raise SourceRefusal("lock_unreadable", exc.detail) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceRefusal("lock_unreadable", f"{path}: {exc}") from exc
    if not isinstance(lock, dict) or set(lock) != {"masy", "maple", "syrup"}:
        raise SourceRefusal(
            "lock_unreadable", f"{path} is not a source_observation block with exactly masy/maple/syrup",
        )
    return lock


def compare_lock(lock: dict[str, Any], observation: dict[str, Any]) -> None:
    differences = diff_observations(lock, observation)
    if differences:
        raise SourceRefusal(
            "lock_mismatch", f"current source observation differs from the lock at {differences}",
            differences=differences,
        )
