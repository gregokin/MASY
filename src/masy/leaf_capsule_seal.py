"""Operator-side observation and sealing of a clean committed LEAF source (`leaf/` + `pyproject.toml`) into an immutable capsule.

NOT closure-safe: imports `masy.runtime` and `masy.sourceobs` (generic Git/hash/copy/publish helpers, reused unchanged). Never imported by the run parent or worker; the run-time
verifier is `masy.leaf_capsule`. Reuse, not a new framework: `runtime.ls_scope/_git/head_tree` (Git, bounded), `sourceobs._walk_scope` (no-follow enumeration), `runtime._hash_path`
(descriptor-joined streaming hash + Git blob id), `_copy_verified`, `_freeze`, `_rename_noreplace`, `_fsync_dir`, `_write_new` (no-replace atomic publication) and
`check_store_location` (namespace separation). Existing objects are verified, never repaired; failed staging is retained. Nothing here writes to a repository.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Callable

from masy import leaf_capsule as cap
from masy.native_source_gate import GateLimits, GateRefusal
from masy.runtime import (RuntimeLimits, RuntimeRefusal, _copy_verified, _freeze, _fsync_dir, _from_source_refusal, _git, _hash_path, _rename_noreplace,
                          _require_real_dir, _walk_object, _write_new, check_store_location, head_tree, ls_scope, project_identity)
from masy.sourceobs import SourceRefusal, _canonical_root, _walk_scope

__all__ = ["observe_leaf_scope", "seal_leaf_capsule", "verify_leaf_capsule"]

PATHSPECS = ["leaf", "pyproject.toml"]
_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_RUN_ID = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]{0,127}$")
_CHUNK = 1 << 20
Admit = Callable[[int, str], None]


def _no_admit(_n: int, _what: str) -> None:
    return None


def _gate_to_runtime(exc: GateRefusal) -> RuntimeRefusal:
    return RuntimeRefusal(exc.code, exc.detail)


def observe_leaf_scope(root: str, *, expected_revision: str, limits: RuntimeLimits | None = None, gate_limits: GateLimits | None = None, admit: Admit = _no_admit) -> dict[str, Any]:
    """Observe one clean committed LEAF checkout. `expected_revision` is REQUIRED and must equal HEAD exactly. Refuses a dirty/untracked/ignored/symlinked/special scope, any file
    whose bytes differ from the commit's blob, and any bound violation (the capsule bounds equal the run-time verifier's `GateLimits`, so every sealed capsule can be verified)."""
    limits, gl = limits or RuntimeLimits(), gate_limits or GateLimits()
    if type(expected_revision) is not str or not _OBJECT_ID.match(expected_revision):
        raise RuntimeRefusal("expected_revision_required", "expected_revision must be a full lowercase hex commit id")
    try:
        real = _canonical_root(root)
    except SourceRefusal as exc:
        raise _from_source_refusal(exc) from exc
    real_s = str(real)
    sl = limits.source_limits()
    admit(2 * _CHUNK + gl.entry_model_bytes * gl.max_files, "LEAF observation buffers")
    top = _git(real_s, ["rev-parse", "--show-toplevel"], limits).decode("utf-8", "surrogateescape").strip()
    if os.path.realpath(top) != real_s:
        raise RuntimeRefusal("not_repository_root", f"{real_s} is not its repository top level ({top})")
    head = _git(real_s, ["rev-parse", "--verify", "HEAD^{commit}"], limits).decode("ascii", "replace").strip()
    if not _OBJECT_ID.match(head):
        raise RuntimeRefusal("head_unreadable", f"HEAD did not resolve to an object id: {head[:100]!r}")
    if head != expected_revision:
        raise RuntimeRefusal("expected_revision_mismatch", f"current HEAD {head} differs from expected_revision {expected_revision}", head=head, expected_revision=expected_revision)
    status = _git(real_s, ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none", "--", *PATHSPECS], limits)
    if status:
        raise RuntimeRefusal("dirty_scope", f"uncommitted change(s) in {PATHSPECS}", entries=status.decode("utf-8", "replace").split("\0")[:20])
    entries = ls_scope(real_s, head, PATHSPECS, limits)
    object_format = "sha1" if len(head) == 40 else "sha256"
    if len(entries) > gl.max_files:
        raise RuntimeRefusal("max_files_exceeded", f"more than {gl.max_files} committed files in scope")
    committed = {path for path, _m, _o in entries}
    try:
        on_disk = _walk_scope(Path(real_s), "leaf", sl)
    except SourceRefusal as exc:
        raise _from_source_refusal(exc) from exc
    stray = sorted(rel for rel, _info in on_disk if rel not in committed)
    if stray:  # ignored or untracked runtime source (caches under __pycache__ are the only pruned entries)
        raise RuntimeRefusal("untracked_runtime_input", f"{len(stray)} file(s) under leaf/ are not committed (ignored or untracked): {stray[:5]}", paths=stray[:20])
    files, remaining, checked, pyproject = [], gl.max_total_bytes, set(), None
    for path, _mode, oid in entries:
        try:
            cap._check_path(path, gl)
        except GateRefusal as exc:
            raise RuntimeRefusal("unsupported_tracked_entry", exc.detail) from None
        parent = os.path.dirname(path)
        while parent and parent not in checked:
            _require_real_dir(os.path.join(real_s, parent), "symlink_in_scope")
            checked.add(parent)
            parent = os.path.dirname(parent)
        admit(2 * _CHUNK, f"hashing {path}")
        sha, blob, size, data = _hash_path(os.path.join(real_s, path), budget=min(remaining, gl.max_file_bytes), blob_algorithm=object_format, keep=(path == "pyproject.toml"))
        if blob != oid:
            raise RuntimeRefusal("content_differs_from_head", f"{path} differs from commit {head}")
        remaining -= size
        files.append([path, size, sha, blob])
        if data is not None:
            pyproject = data
    if pyproject is None:
        raise RuntimeRefusal("pyproject_not_committed", "pyproject.toml is not in HEAD")
    if not any(f[0] == "leaf/__init__.py" for f in files):
        raise RuntimeRefusal("package_not_committed", "leaf/__init__.py is not in HEAD")
    project = project_identity(pyproject, "leaf", "pyproject.toml")
    return {"root": real_s, "head": head, "tree": head_tree(real_s, head, limits), "object_format": object_format, "files": files, "project": project}


def _identity(observation: dict[str, Any]) -> dict[str, Any]:
    files = sorted([list(f) for f in observation["files"]])
    return {"schema": cap.CAPSULE_SCHEMA, "kind": cap.CAPSULE_KIND, "object_format": observation["object_format"], "package": cap.PACKAGE_DIR, "project": dict(observation["project"]),
            "files": files, "tree_digest": cap.tree_digest([(f[0], f[2]) for f in files]), "package_digest": cap.package_digest(files)}


def verify_leaf_capsule(location: str, capsule_id: str, *, limits: GateLimits | None = None, admit: Admit = _no_admit, name_dir: str | None = None,
                        expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Deep operator verification: the closure verifier, then `_walk_object` (no symlink/special/writable anywhere) and every file's sha256, size AND Git blob id."""
    limits = limits or GateLimits()
    try:
        record = cap.verify_capsule_dir(location, capsule_id, limits, admit, name_dir=name_dir, expected_manifest_sha256=expected_manifest_sha256)
    except GateRefusal as exc:
        raise _gate_to_runtime(exc) from None
    n_files = len(record["files"])
    found, dirs = _walk_object(Path(location), 2 * n_files + n_files + 16)
    expected = {"manifest.json", "source-manifest.json"} | {f"source/{p}" for p in record["files"]}
    if set(found) != expected:
        raise RuntimeRefusal("object_tamper", f"{location}: file set differs from the manifest")
    for path, sha in record["files"].items():  # identity data comes from the already bounded, validated closure verification (no second manifest read)
        size, blob = record["sizes"][path], record["blobs"][path]
        admit(2 * _CHUNK, f"hashing {path}")
        try:
            got_sha, got_blob, got_size, _d = _hash_path(os.path.join(location, "source", path), budget=size + 1, blob_algorithm=record["object_format"])
        except RuntimeRefusal as exc:
            raise RuntimeRefusal("object_tamper", exc.detail) from exc
        if (got_sha, got_size, got_blob) != (sha, size, blob):
            raise RuntimeRefusal("object_tamper", f"{path} differs from the capsule identity (sha256/size/Git blob)")
    return record


def _ensure_namespace(namespace: str, limits: GateLimits) -> None:
    if not os.path.lexists(namespace):
        _require_real_dir(os.path.dirname(namespace), "namespace_parent_missing")
        try:
            os.mkdir(namespace, 0o755)
        except FileExistsError:
            pass
    try:
        os.mkdir(os.path.join(namespace, "objects"), 0o755)
    except FileExistsError:
        pass
    try:
        cap.validate_namespace(namespace, limits)
    except GateRefusal as exc:
        raise _gate_to_runtime(exc) from None


def seal_leaf_capsule(namespace: str, observation: dict[str, Any], *, run_id: str, wrapper_roots: list[str], protected: dict[str, str], limits: RuntimeLimits | None = None,
                      gate_limits: GateLimits | None = None, admit: Admit = _no_admit, _after_copy: Callable[[], None] | None = None) -> dict[str, Any]:
    """Seal `observation` (from `observe_leaf_scope`) into `<namespace>/objects/<id>`: staged copy with per-file hash re-checks, frozen read-only, deeply verified against its final
    identity, then published with renameat2(RENAME_NOREPLACE). An existing object is deeply verified and reused (a corrupt one refuses, untouched). Failed staging is retained.
    `namespace` must be disjoint from the V1 store, the wrapper roots and every protected root (the observed LEAF root is always protected)."""
    limits, gl = limits or RuntimeLimits(), gate_limits or GateLimits()
    if type(run_id) is not str or not _RUN_ID.match(run_id):
        raise RuntimeRefusal("run_id_invalid", f"run id {run_id!r} is not a safe single component")
    check_store_location(namespace, wrapper_roots=wrapper_roots, protected={**protected, "leaf source root": observation["root"]})
    identity = _identity(observation)
    try:
        cap.validate_identity(identity, gl)
    except GateRefusal as exc:
        raise _gate_to_runtime(exc) from None
    capsule_id = hashlib_sha(cap.canonical_bytes(identity))
    _ensure_namespace(namespace, gl)
    objects = os.path.join(namespace, "objects")
    final = os.path.join(objects, capsule_id)
    if os.path.lexists(final):
        return {**verify_leaf_capsule(final, capsule_id, limits=gl, admit=admit), "reused": True}
    staging = Path(objects) / f".staging-{run_id}-leaf"
    try:
        os.mkdir(staging, 0o700)
    except FileExistsError as exc:
        raise RuntimeRefusal("staging_exists", f"{staging} already exists") from exc
    root = Path(observation["root"])
    try:
        for path, size, sha, _blob in identity["files"]:
            admit(2 * _CHUNK, f"copying {path}")
            _copy_verified(root / path, staging / "source" / path, size, sha)
        if _after_copy is not None:
            _after_copy()
        try:  # the SAME bounded observation as before the copy (includes ignored/untracked runtime input): files, commit and tree must be identical
            after = observe_leaf_scope(str(root), expected_revision=observation["head"], limits=limits, gate_limits=gl, admit=admit)
        except RuntimeRefusal as exc:
            raise RuntimeRefusal("source_changed_during_seal", f"re-observation after the copy refused ({exc.code}): {exc.detail}") from exc
        if after["files"] != observation["files"] or after["tree"] != observation["tree"] or after["head"] != observation["head"]:
            raise RuntimeRefusal("source_changed_during_seal", "the LEAF scope differs from the observation after the copy")
        source_manifest = cap.canonical_bytes({"files": {f[0]: f[2] for f in identity["files"]}, "note": "LEAF source capsule: byte-consistency record; not a qualification authority"})
        provenance = {"head": observation["head"], "tree": observation["tree"], "scope_pathspecs": list(PATHSPECS), "git_status": "clean",
                      "source_digest_algorithm": cap.SOURCE_DIGEST_ALGORITHM, "source_manifest_sha256": hashlib_sha(source_manifest)}
        _write_new(staging / "source-manifest.json", source_manifest)
        _write_new(staging / "manifest.json", cap.canonical_bytes({"schema": cap.CAPSULE_SCHEMA, "object_id": capsule_id, "identity": identity, "provenance": provenance}))
        _freeze(staging)
        verify_leaf_capsule(str(staging), capsule_id, limits=gl, admit=admit, name_dir=final)
        try:
            _rename_noreplace(staging, Path(final))
        except FileExistsError:
            return {**verify_leaf_capsule(final, capsule_id, limits=gl, admit=admit), "reused": True, "retained_staging": str(staging)}
    except RuntimeRefusal as exc:
        exc.evidence.setdefault("retained_staging", str(staging))
        raise
    except OSError as exc:
        raise RuntimeRefusal("object_publication_failed", f"leaf capsule: {exc}", retained_staging=str(staging)) from exc
    _fsync_dir(objects)
    return {**verify_leaf_capsule(final, capsule_id, limits=gl, admit=admit), "reused": False}


def hashlib_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
