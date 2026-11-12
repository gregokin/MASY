"""Opt-in committed production runtime resolution of the configured joint (water + LEAF) native calendar run (run config v2, `source.mode = production_committed`).

Closure-safe: stdlib + `masy.native_source_gate` + `masy.leaf_capsule` only (imported lazily by the parent/worker production branch, so the development closures are unchanged).
It never imports `masy.runtime`, `masy.sourceobs`, NumPy, SYRUP, MAPLE or LEAF. The existing V1 three-source runtime is REUSED, not re-implemented: the parent genuinely launches
`tools/masy_runtime.py verify` of the release MASY worktree in a bounded child and cross-checks its closed success output; the stdlib code here additionally re-derives, byte by
byte and independently of that child, the sealed MASY/SYRUP objects, the immutable LEAF capsule, the lock, the release worktree, the interpreter and the running parent/worker modules.

Trust model (declared, accidental-drift; no signing): hashes, object ids and commit ids prove BYTE CONSISTENCY with what the recipe names. Independent scientific acceptance is only
the published Root-reviewed release receipt the operator selects by sha256; this module checks the receipt/recipe join and status but cannot prove independence. A self-generated
recipe + receipt + shas is consistent, not independently reviewed. `production_qualified` therefore stays false in every identity/summary.

Modelled byte estimates admit BEFORE each read/hash/parse/spawn; they are not whole-process RSS. The verify child imports the native numerical stack (fixed cost, excluded).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

from masy import leaf_capsule as cap
from masy import native_source_gate as gate
from masy.native_source_gate import GateLimits, GateRefusal

__all__ = ["DELIVERY_SCHEMA", "DELIVERY_CORE_KEYS", "delivery_core", "validate_delivery", "IDENTITY_SCHEMA", "LOCK_SCHEMA", "RECIPE_SCHEMA", "RECEIPT_SCHEMA", "ProductionLimits", "Resolved", "canonical_json", "parse_receipt", "parse_recipe", "production_identity", "resolve"]

RECIPE_SCHEMA = "masy.production_recipe.v1"
RECEIPT_SCHEMA = "masy.production_release_receipt.v1"
LOCK_SCHEMA = "masy.runtime_lock.v1"
SOURCE_OBJECT_SCHEMA = "masy.source_object.v1"
OUTPUT_SCHEMA = "masy.runtime_output.v1"
IDENTITY_SCHEMA = "masy.native_water_run_bootstrap.v2"
DELIVERY_SCHEMA = "masy.native_delivery.v1"
DELIVERY_CORE_KEYS = ("schema", "bundle_id", "manifest_sha256", "archive_sha256", "device", "leaf_profile", "leaf_variant")  # the path-free, run-independent part: compared at restart
_DELIVERY_KEYS = {"schema", "bundle_id", "manifest_sha256", "archive_sha256", "target", "leaf_profile", "device", "leaf_variant", "leaf_root", "syrup_root", "masy_root", "lock_id",
                  "production_identity_sha256"}
LEAF_PROFILES = MappingProxyType({"cpu": "leaf23_8da_water109_native9f_cpu_v1", "cuda": "leaf23_8da_water109_native9f_cuda_v1"})  # pinned equal to leaf_source_profile by a test
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SLACK = 16384
_KINDS = {"masy": "masy", "syrup": "maple_syrup"}  # sealed V1 source kinds this module verifies -> package name (MAPLE is verified by the V1 child + the worker's cold bundle check)
Admit = Callable[[int, str], None]


def _no_admit(_n: int, _what: str) -> None:
    return None


@dataclass(frozen=True)
class ProductionLimits:
    max_recipe_bytes: int = 1 << 20
    max_receipt_bytes: int = 1 << 20
    max_lock_bytes: int = 4 << 20
    verify_timeout_s: float = 900.0
    verify_max_output_bytes: int = 1 << 20

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise GateRefusal("invalid_limits", f"{name} must be positive")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _refuse(code: str, detail: str) -> GateRefusal:
    return GateRefusal(code, detail)


def _hex(value: Any, what: str, pattern=_HEX64) -> str:
    if type(value) is not str or not pattern.match(value):
        raise _refuse("production_record_invalid", f"{what} is not a lowercase hex digest")
    return value


def _closed(value: Any, keys: set[str], what: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise _refuse("production_record_invalid", f"{what} keys {sorted(value) if isinstance(value, dict) else type(value).__name__} != {sorted(keys)}")
    return value


def _text(value: Any, what: str, limit: int = 512) -> str:
    if type(value) is not str or not value or len(value) > limit or any(ord(c) < 0x20 for c in value):
        raise _refuse("production_record_invalid", f"{what} must be a short printable string")
    return value


def _json_file(path: str, cap_bytes: int, limits: GateLimits, admit: Admit, what: str, code: str) -> tuple[Any, bytes, str]:
    data = cap._read_file(path, cap_bytes, admit, what, code)
    return cap._strict_json(data, limits, admit, what), data, _sha(data)


def validate_delivery(record: Any) -> dict[str, Any]:
    """Closed validation of the verified-delivery record handed to `leaf_source_profile.verify_profile_delivery` (shape only; the ACTUAL native verification must then match it)."""
    if type(record) is MappingProxyType:  # exactly the immutable type `resolve` produces (ownership stays immutable); no other Mapping is accepted
        record = dict(record)
    if type(record) is not dict:  # exact types only: plain dict or the immutable proxy above (no dict subclass/other Mapping)
        raise _refuse("production_record_invalid", f"delivery must be a plain dict or the immutable record `resolve` produces, got {type(record).__name__}")
    r = _closed(record, _DELIVERY_KEYS, "delivery")
    if r["schema"] != DELIVERY_SCHEMA:
        raise _refuse("production_record_invalid", f"delivery schema must be {DELIVERY_SCHEMA!r}")
    for k in ("bundle_id", "manifest_sha256", "archive_sha256", "lock_id", "production_identity_sha256"):
        _hex(r[k], f"delivery.{k}")
    if r["device"] not in LEAF_PROFILES or r["leaf_profile"] != LEAF_PROFILES[r["device"]] or r["leaf_variant"] != "8da":
        raise _refuse("production_record_invalid", "delivery device/leaf_profile/leaf_variant must be the literal 8da pair")
    for k in ("target", "leaf_root", "syrup_root", "masy_root"):
        if type(r[k]) is not str or not os.path.isabs(r[k]) or os.path.normpath(r[k]) != r[k]:
            raise _refuse("production_record_invalid", f"delivery.{k} must be an absolute normalized path")
    return dict(r)


def delivery_core(record: Any) -> dict[str, Any]:
    """Noncircular projection of a validated delivery record: only the run-independent identity facts (no paths, lock id or production digest). It is what the fence/identity/checkpoint compare."""
    r = validate_delivery(record)
    return {k: r[k] for k in DELIVERY_CORE_KEYS}


# ---------------------------------------------------------------------------------------------- recipe / receipt


def parse_recipe(doc: Any) -> dict[str, Any]:
    r = _closed(doc, {"schema", "name", "profiles", "sources"}, "recipe")
    if r["schema"] != RECIPE_SCHEMA:
        raise _refuse("production_record_invalid", f"recipe schema must be {RECIPE_SCHEMA!r}")
    _text(r["name"], "recipe.name", 128)
    if not isinstance(r["profiles"], dict) or not 1 <= len(r["profiles"]) <= 8:
        raise _refuse("production_record_invalid", "recipe.profiles needs 1..8 entries")
    for name, p in r["profiles"].items():
        _text(name, "recipe profile name", 128)
        p = _closed(p, {"device", "leaf_profile", "leaf_variant"}, f"recipe.profiles.{name}")
        if p["device"] not in LEAF_PROFILES or p["leaf_variant"] != "8da" or p["leaf_profile"] != LEAF_PROFILES[p["device"]]:
            raise _refuse("production_record_invalid", f"recipe profile {name!r}: device/leaf_variant/leaf_profile must be the literal 8da pair")
    s = _closed(r["sources"], {"maple", "syrup", "masy", "leaf"}, "recipe.sources")
    for kind, keys in (("maple", {"commit", "object_id", "package_digest", "bundle_id", "bundle_manifest_sha256", "archive_sha256"}), ("syrup", {"commit", "object_id", "package_digest"}),
                       ("masy", {"commit", "object_id", "package_digest", "launcher_sha256"}), ("leaf", {"commit", "capsule_id", "package_digest", "tree_digest"})):
        rec = _closed(s[kind], keys, f"recipe.sources.{kind}")
        _hex(rec["commit"], f"{kind}.commit", _OBJECT_ID)
        for k in keys - {"commit"}:
            _hex(rec[k], f"{kind}.{k}")
    return r


def parse_receipt(doc: Any, recipe_sha256: str) -> dict[str, Any]:
    r = _closed(doc, {"schema", "status", "independent", "recipe_sha256", "scope", "evidence", "limits"}, "receipt")
    if r["schema"] != RECEIPT_SCHEMA:
        raise _refuse("production_record_invalid", f"receipt schema must be {RECEIPT_SCHEMA!r}")
    if r["status"] != "root_reviewed" or r["independent"] is not True:
        raise _refuse("recipe_unreviewed", f"the release receipt status {r['status']!r}/independent={r['independent']!r} is not a Root-reviewed attestation: draft/non-reviewed status is refused. Independence is attested by the published Root receipt plus the operator's pins, not inferred from these JSON fields (accidental-drift model)")
    if r["recipe_sha256"] != recipe_sha256:
        raise _refuse("receipt_recipe_mismatch", "the receipt names a different recipe sha256")
    _text(r["scope"], "receipt.scope", 2048)
    for key in ("evidence", "limits"):
        if not isinstance(r[key], list) or not 1 <= len(r[key]) <= 64 or not all(isinstance(x, str) and 0 < len(x) <= 512 for x in r[key]):
            raise _refuse("production_record_invalid", f"receipt.{key} must be 1..64 short strings")
    return r


# ---------------------------------------------------------------------------------------------- lock / sealed objects


def _read_lock(store: str, lock_id: str, plim: ProductionLimits, limits: GateLimits, admit: Admit) -> tuple[dict[str, Any], str]:
    if type(lock_id) is not str or not _HEX64.match(lock_id):
        raise _refuse("lock_ref_invalid", "production_lock must be a 64-hex lock id")
    path = os.path.join(store, "locks", f"{lock_id}.json")
    lock, data, sha = _json_file(path, plim.max_lock_bytes, limits, admit, "runtime lock", "lock_unreadable")
    if not isinstance(lock, dict) or lock.get("schema") != LOCK_SCHEMA:
        raise _refuse("lock_invalid", f"lock schema is not {LOCK_SCHEMA!r}")
    if data != (canonical_json(lock) + "\n").encode("ascii"):
        raise _refuse("lock_invalid", "the lock file is not canonical lock JSON")
    body = {k: v for k, v in lock.items() if k != "lock_id"}
    if lock.get("lock_id") != lock_id or _sha(canonical_json(body).encode("ascii")) != lock_id or lock.get("store") != store:
        raise _refuse("lock_invalid", "the lock id is not the sha256 of its canonical body, or its store differs")
    return lock, sha


def _lock_get(lock: dict[str, Any], *path: str) -> Any:
    cur: Any = lock
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise _refuse("lock_invalid", f"the lock lacks {'.'.join(path)}")
        cur = cur[key]
    return cur


def _rows(identity: dict[str, Any], kind: str) -> list[list[Any]]:
    keys = {"schema", "kind", "object_format", "package", "project", "files", "overlay"}
    ident = _closed(identity, keys, "source object identity")
    if ident["schema"] != SOURCE_OBJECT_SCHEMA or ident["kind"] != kind or ident["package"] != _KINDS[kind] or ident["object_format"] not in ("sha1", "sha256"):
        raise _refuse("object_tamper", "source object identity kind/package/schema/format differs")
    rows, previous = ident["files"], None
    if not isinstance(rows, list) or not 1 <= len(rows) <= GateLimits().max_files:
        raise _refuse("object_tamper", "source object identity has no/too many files")
    for row in rows:
        if not (isinstance(row, list) and len(row) == 4 and type(row[0]) is str and type(row[1]) is int and row[1] >= 0):
            raise _refuse("object_tamper", f"malformed identity row {row!r}")
        _hex(row[2], f"{row[0]} sha256")
        _hex(row[3], f"{row[0]} blob", _OBJECT_ID)
        path = row[0]
        parts = path.split("/")
        if (not path or path.startswith("/") or "\\" in path or any(p in ("", ".", "..", ".git", "__pycache__") for p in parts)
                or not (path == "pyproject.toml" or path.startswith("src/") or (kind == "masy" and path == "tools/masy_runtime.py"))):
            raise _refuse("object_tamper", f"identity path {path!r} is outside the sealed scope")
        if previous is not None and path <= previous:
            raise _refuse("object_tamper", "identity files are not strictly sorted")
        previous = path
    return rows


def _verify_sealed_source(store: str, kind: str, expected_id: str, limits: GateLimits, admit: Admit) -> dict[str, Any]:
    """Independent closure verification of one sealed V1 source object against the recipe-pinned id: exact entries, canonical manifest, id = sha256(canonical identity) = directory name,
    exact directory/file census, read-only modes, every byte (`verify_snapshot`) and every per-file size (size-inclusive tree digest)."""
    _hex(expected_id, f"{kind} object id")
    location = os.path.join(store, "objects", "sources", expected_id)
    try:
        st = os.lstat(location)
    except OSError as exc:
        raise _refuse("object_missing", f"{kind} object: {exc}") from None
    if not stat.S_ISDIR(st.st_mode):
        raise _refuse("object_tamper", f"{location} is not a real directory")
    if st.st_mode & 0o222:
        raise _refuse("object_tamper", f"{location} has write bits; a sealed V1 object is frozen read-only")
    entries = cap._scan_dir(location, 3, limits, admit, "sealed object top-level census", "object_tamper")
    top = {n for n, _s in entries}
    if not {"manifest.json", "source"} <= top or top - {"manifest.json", "source", "editable"}:
        raise _refuse("object_tamper", f"{location} entries {sorted(top)} differ from the V1 sealed layout")
    for name, est in entries:  # type + read-only mode of every top-level member (symlinks/specials/writable refuse)
        want = stat.S_ISREG if name == "manifest.json" else stat.S_ISDIR
        if not want(est.st_mode) or est.st_mode & 0o222:
            raise _refuse("object_tamper", f"{location}/{name} is not a read-only {'file' if name == 'manifest.json' else 'directory'}")
    data = cap._read_file(os.path.join(location, "manifest.json"), limits.max_manifest_bytes, admit, f"{kind} object manifest.json", "object_tamper")
    manifest = cap._strict_json(data, limits, admit, f"{kind} object manifest.json")
    manifest = _closed(manifest, {"schema", "object_id", "identity", "overlay_files"}, "source object manifest")
    if manifest["schema"] != SOURCE_OBJECT_SCHEMA or data != (canonical_json(manifest) + "\n").encode("ascii") or manifest["object_id"] != expected_id:
        raise _refuse("object_tamper", f"{kind} object manifest is not canonical or names another object")
    if _sha(canonical_json(manifest["identity"]).encode("ascii")) != expected_id:
        raise _refuse("object_tamper", f"{kind} object identity digest is not the recipe-pinned id")
    rows = _rows(manifest["identity"], kind)
    files = {r[0]: r[2] for r in rows}
    source = os.path.join(location, "source")
    dirs: set[str] = set()
    for r in rows:
        parent = os.path.dirname(r[0])
        while parent:
            dirs.add(parent)
            parent = os.path.dirname(parent)
    cap._dir_census(source, dirs, limits, admit)
    snap = verify_sealed_snapshot(source, rows, limits, admit)
    overlay = manifest["overlay_files"]  # the V1 metadata overlay subtree: present exactly when the manifest lists it, then every byte/entry/mode is checked like the source
    if (overlay is None) == ("editable" in top):
        raise _refuse("object_tamper", f"{location}: the editable overlay presence does not match the manifest")
    if overlay is not None:
        if not isinstance(overlay, list) or not 1 <= len(overlay) <= limits.max_files:
            raise _refuse("object_tamper", "malformed overlay file list")
        orows = []
        for item in overlay:
            if not (isinstance(item, list) and len(item) == 3 and type(item[0]) is str and type(item[1]) is int and item[1] >= 0):
                raise _refuse("object_tamper", f"malformed overlay row {item!r}")
            _hex(item[2], f"overlay {item[0]} sha256")
            if not item[0] or item[0].startswith("/") or "\\" in item[0] or any(p in ("", ".", "..") for p in item[0].split("/")):
                raise _refuse("object_tamper", f"unsafe overlay path {item[0]!r}")
            orows.append([item[0], item[1], item[2], ""])
        odirs: set[str] = set()
        for r in orows:
            parent = os.path.dirname(r[0])
            while parent:
                odirs.add(parent)
                parent = os.path.dirname(parent)
        cap._dir_census(os.path.join(location, "editable"), odirs, limits, admit)
        verify_sealed_snapshot(os.path.join(location, "editable"), orows, limits, admit)
    pkg = _KINDS[kind]
    prefix = f"src/{pkg}/"
    return {"object_id": expected_id, "location": location, "source_root": source, "files": files, "tree_digest": snap["tree_digest"], "manifest_sha256": _sha(data),
            "package_digest": cap.tree_digest([(p[len(prefix):], s) for p, s in files.items() if p.startswith(prefix)]), "object_format": manifest["identity"]["object_format"]}


def verify_sealed_snapshot(source: str, rows: list[list[Any]], limits: GateLimits, admit: Admit) -> dict[str, Any]:
    files = {r[0]: r[2] for r in rows}
    snap = gate.verify_snapshot(source, files, limits, admit)
    if snap["tree_digest"] != gate._tree_digest({r[0]: (r[1], r[2]) for r in rows}):
        raise _refuse("snapshot_differs", "per-file source sizes differ from the object identity")
    return snap


# ---------------------------------------------------------------------------------------------- interpreter (V1 record semantics)

_PY_KEYS = ("spelling", "is_symlink", "link_target", "realpath", "size", "sha256", "venv")


def _interpreter_record(limits: GateLimits, admit: Admit) -> dict[str, Any]:
    """The running interpreter's record in the SAME shape/semantics as the V1 lock's `runtime.python` (`runtime.interpreter_identity` / `_python_stable`): the spelling as invoked
    (`sys.executable`; a venv `bin/python` keeps its environment), only the final component may be a symlink, the realpath bytes hashed, and the venv directory + `pyvenv.cfg` hash.
    The base binary is NOT an identity of the environment: two venvs over one /usr/bin/python3.12 differ in spelling and venv."""
    spelling = sys.executable
    if not spelling or not os.path.isabs(spelling) or os.path.normpath(spelling) != spelling:
        raise _refuse("interpreter_mismatch", f"sys.executable {spelling!r} is not an absolute normalized path")
    parent = os.path.dirname(spelling)
    if os.path.realpath(parent) != parent:
        raise _refuse("interpreter_mismatch", f"{parent} resolves through a symlink; only the interpreter's own final path component may be a symlink")
    try:
        st = os.lstat(spelling)
    except OSError as exc:
        raise _refuse("interpreter_mismatch", f"{spelling}: {exc}") from None
    is_link = stat.S_ISLNK(st.st_mode)
    if not is_link and not stat.S_ISREG(st.st_mode):
        raise _refuse("interpreter_mismatch", f"{spelling} is neither a regular file nor a symlink")
    ident = gate.interpreter_identity(limits, spelling, admit)
    venv_dir = os.path.dirname(parent)
    cfg_path = os.path.join(venv_dir, "pyvenv.cfg")
    venv = None
    if os.path.lexists(cfg_path):
        if os.path.realpath(venv_dir) != venv_dir:
            raise _refuse("interpreter_mismatch", f"environment {venv_dir} is not canonical")
        venv = {"dir": venv_dir, "pyvenv_cfg_sha256": _sha(cap._read_file(cfg_path, 1 << 20, admit, "pyvenv.cfg", "interpreter_mismatch"))}
    return {"spelling": spelling, "is_symlink": is_link, "link_target": os.readlink(spelling) if is_link else None, "realpath": ident["realpath"], "size": ident["size"],
            "sha256": ident["sha256"], "venv": venv}


# ---------------------------------------------------------------------------------------------- release root / V1 verify child


def _check_release_root(release: str, masy_files: dict[str, str], recipe_launcher: str, lock: dict[str, Any], limits: GateLimits, admit: Admit) -> None:
    if not os.path.isabs(release) or os.path.normpath(release) != release or os.path.realpath(release) != release:
        raise _refuse("release_root_invalid", f"{release!r} must be an absolute, normalized, symlink-free path")
    if _lock_get(lock, "sources", "masy", "root") != release:
        raise _refuse("release_root_mismatch", "release_masy_root is not the MASY root the runtime lock was prepared from")
    for sub in ("src", "tools"):
        st = os.lstat(os.path.join(release, sub))
        if not stat.S_ISDIR(st.st_mode):
            raise _refuse("release_root_invalid", f"{release}/{sub} is not a real directory")
    launcher = cap._read_file(os.path.join(release, "tools", "masy_runtime.py"), 1 << 20, admit, "release launcher", "release_root_invalid")
    if not (_sha(launcher) == recipe_launcher == masy_files["tools/masy_runtime.py"] == _lock_get(lock, "sources", "masy", "launcher", "sha256")):
        raise _refuse("release_root_differs", "the release launcher bytes differ from the recipe/lock/sealed object")
    pyproject = cap._read_file(os.path.join(release, "pyproject.toml"), 1 << 20, admit, "release pyproject.toml", "release_root_invalid")
    if _sha(pyproject) != masy_files["pyproject.toml"]:
        raise _refuse("release_root_differs", "release pyproject.toml differs from the sealed MASY object")
    found = gate.record_manifest(os.path.join(release, "src"), limits, admit)["files"]
    want = {p[len("src/"):]: s for p, s in masy_files.items() if p.startswith("src/")}
    if found != want:
        extra, missing = sorted(set(found) - set(want)), sorted(set(want) - set(found))
        changed = sorted(p for p in set(found) & set(want) if found[p] != want[p])
        raise _refuse("release_root_differs", f"release src differs from the sealed MASY object: extra={extra[:5]} missing={missing[:5]} changed={changed[:5]}")


def _child_env(capability: str, cuda_visible_devices: str | None) -> dict[str, str]:
    """Narrow explicit child environment: CPU hides every GPU; CUDA forwards ONLY the caller's authorized `CUDA_VISIBLE_DEVICES` binding (never unset = all devices)."""
    env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    if capability == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif capability == "cuda":
        if type(cuda_visible_devices) is not str or not cuda_visible_devices.strip() or any(ord(c) < 0x20 for c in cuda_visible_devices):
            raise _refuse("cuda_visibility_unbound", "a CUDA production run requires an explicit non-empty CUDA_VISIBLE_DEVICES binding in the caller environment (unset would expose every device)")
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    else:
        raise _refuse("production_device_mismatch", f"unknown capability {capability!r}")
    return env


def _run_verify_child(python: str, launcher: str, store: str, lock_id: str, plim: ProductionLimits, admit: Admit, *, capability: str = "cpu", cuda_visible_devices: str | None = None) -> dict[str, Any]:
    """Genuinely run the EXISTING V1 `masy_runtime verify` and cross-check its closed success output. Bounded: wall time, combined stdout+stderr bytes, no shell, own process group."""
    admit(2 * plim.verify_max_output_bytes + _SLACK, "V1 runtime verify child output buffers")
    argv = [python, "-I", "-B", launcher, "verify", "--store", store, "--lock", lock_id]
    env = _child_env(capability, cuda_visible_devices)
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd="/", start_new_session=True)
    except OSError as exc:
        raise _refuse("verify_child_failed", f"cannot start the V1 verify child: {exc}") from None
    out: list[bytes] = []
    total, deadline, err_total = 0, time.monotonic() + plim.verify_timeout_s, 0
    try:
        with selectors.DefaultSelector() as sel:
            sel.register(proc.stdout, selectors.EVENT_READ, "o")
            sel.register(proc.stderr, selectors.EVENT_READ, "e")
            while sel.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _refuse("verify_child_timeout", f"the V1 verify child exceeded {plim.verify_timeout_s} s")
                for key, _ in sel.select(timeout=min(remaining, 1.0)):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        sel.unregister(key.fileobj)
                        continue
                    if key.data == "o":
                        total += len(chunk)
                        out.append(chunk)
                    else:
                        err_total += len(chunk)
                    if total + err_total > plim.verify_max_output_bytes:
                        raise _refuse("verify_child_output_too_large", f"the V1 verify child wrote more than {plim.verify_max_output_bytes} B")
        proc.wait(timeout=max(deadline - time.monotonic(), 1.0))
    except subprocess.TimeoutExpired:
        raise _refuse("verify_child_timeout", "the V1 verify child did not exit") from None
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait()
        for stream in (proc.stdout, proc.stderr):
            stream.close()
    lines = b"".join(out).decode("utf-8", "replace").strip().splitlines()
    try:
        result = json.loads(lines[-1]) if lines else None
    except ValueError:
        result = None
    if proc.returncode != 0 or not isinstance(result, dict):
        raise _refuse("verify_child_failed", f"the V1 verify child exited {proc.returncode} without a closed JSON result ({lines[-1][:200] if lines else 'no output'})")
    ok = (result.get("schema") == OUTPUT_SCHEMA and result.get("verdict") == "runtime_verified" and result.get("exit_code") == 0 and result.get("lock_id") == lock_id
          and result.get("selection") == "explicit_lock" and result.get("asserts_current") is False and result.get("transport_ready") is False)
    if not ok:
        raise _refuse("verify_child_failed", f"the V1 verify child did not report runtime_verified for this lock: {sorted(result)[:12]}")
    return {"verdict": result["verdict"], "lock_id": result["lock_id"], "selection": result["selection"]}


# ---------------------------------------------------------------------------------------------- resolution


@dataclass(frozen=True)
class Resolved:
    identity: dict[str, Any]
    identity_sha256: str
    source: Any
    manifests: Any
    leaf_root: str
    verify: Any
    delivery: Any = None


def production_identity(**parts: Any) -> dict[str, Any]:
    return {"schema": IDENTITY_SCHEMA, "mode": "production_committed", "production_qualified": False, **parts}


def _parent_code(masy_files: dict[str, str], names: tuple[str, ...]) -> dict[str, Any]:
    sealed = {}
    for name in names:
        key = "src/masy/__init__.py" if name == "__init__" else f"src/masy/{name}.py"
        if key not in masy_files:
            raise _refuse("parent_code_mismatch", f"the sealed MASY object has no {key}")
        sealed[name] = masy_files[key]
    return {"modules": sealed, "digest": _sha("".join(f"{n}\x00{sealed[n]}\n" for n in names).encode())}


def resolve(source: Any, config_sha256: str, *, capability: str, role: str, limits: GateLimits | None = None, plim: ProductionLimits | None = None, admit: Admit = _no_admit,
            run_verify: bool = False, check_running_code: bool = True) -> Resolved:
    """Resolve one v2 production source block into verified sealed roots + the production identity. `role` is 'parent' or 'worker' (closure sets); only the parent runs the V1 verify child.
    Everything here precedes the helper child/worker/model. `check_running_code=False` exists for the pure-layer unit tests only; production callers never pass it."""
    limits, plim = limits or GateLimits(), plim or ProductionLimits()
    if role not in ("parent", "worker") or (run_verify and role != "parent"):
        raise _refuse("production_role_invalid", "role must be parent or worker; only the parent runs the V1 verify child")
    profile, store = source["profile"], source["production_store"]
    # 1. recipe + receipt (bounded reads, strict JSON, operator-selected sha pins, join)
    recipe_doc, _rb, recipe_sha = _json_file(source["recipe"], plim.max_recipe_bytes, limits, admit, "production recipe", "recipe_unreadable")
    if recipe_sha != source["recipe_sha256"]:
        raise _refuse("recipe_sha_mismatch", "the recipe bytes differ from source.recipe_sha256 (operator pin: consistency only)")
    recipe = parse_recipe(recipe_doc)
    receipt_doc, _tb, receipt_sha = _json_file(source["receipt"], plim.max_receipt_bytes, limits, admit, "release receipt", "receipt_unreadable")
    if receipt_sha != source["receipt_sha256"]:
        raise _refuse("receipt_sha_mismatch", "the receipt bytes differ from source.receipt_sha256 (operator pin: consistency only)")
    parse_receipt(receipt_doc, recipe_sha)
    entry = recipe["profiles"].get(profile)
    if entry is None:
        raise _refuse("production_recipe_unavailable", f"the recipe names no production profile {profile!r}")
    if entry["device"] != capability:
        raise _refuse("production_device_mismatch", f"the recipe profile device {entry['device']!r} is not the configured profile capability {capability!r}")
    rs = recipe["sources"]
    # 2. lock (portable ids vs per-machine lock)
    lock, lock_file_sha = _read_lock(store, source["production_lock"], plim, limits, admit)
    for kind in ("maple", "syrup", "masy"):
        rec = rs[kind]
        if (_lock_get(lock, "sources", kind, "head") != rec["commit"] or _lock_get(lock, "objects", "sources", kind, "object_id") != rec["object_id"]
                or _lock_get(lock, "objects", "sources", kind, "package_digest") != rec["package_digest"]):
            raise _refuse("lock_recipe_mismatch", f"the runtime lock's {kind} head/object id/package digest differs from the recipe")
    if _lock_get(lock, "objects", "maple_bundle", "bundle_id") != rs["maple"]["bundle_id"]:
        raise _refuse("lock_recipe_mismatch", "the runtime lock's MAPLE bundle id differs from the recipe")
    target = _lock_get(lock, "objects", "maple_materialized", "path")
    if target != os.path.join(store, "objects", "maple-materialized", rs["maple"]["bundle_id"]):
        raise _refuse("lock_invalid", "the lock's materialized native path is not derived from the bundle id")
    if not (_lock_get(lock, "objects", "maple_bundle", "manifest_sha256") == _lock_get(lock, "objects", "maple_materialized", "manifest_sha256") == rs["maple"]["bundle_manifest_sha256"]
            and _lock_get(lock, "objects", "maple_bundle", "archive_sha256") == rs["maple"]["archive_sha256"]):
        raise _refuse("lock_recipe_mismatch", "the lock's bundle manifest/archive sha256 differ from the recipe's delivery pins")
    manifest_bytes = cap._read_file(os.path.join(target, "bundle-manifest.json"), 16 << 20, admit, "materialized native bundle manifest", "delivery_manifest_unreadable")
    if _sha(manifest_bytes) != rs["maple"]["bundle_manifest_sha256"]:
        raise _refuse("delivery_manifest_mismatch", "the bytes of the materialized bundle-manifest.json differ from the recipe/lock delivery manifest pin")
    # 3. sealed MASY / SYRUP objects and the immutable LEAF capsule (independent of the V1 child)
    sealed = {kind: _verify_sealed_source(store, kind, rs[kind]["object_id"], limits, admit) for kind in ("masy", "syrup")}
    for kind in sealed:
        if sealed[kind]["package_digest"] != rs[kind]["package_digest"]:
            raise _refuse("lock_recipe_mismatch", f"the sealed {kind} package digest differs from the recipe")
    capsule = cap.verify_capsule(source["leaf_capsule_namespace"], rs["leaf"]["capsule_id"], limits, admit)
    if (capsule["package_digest"] != rs["leaf"]["package_digest"] or capsule["tree_digest"] != rs["leaf"]["tree_digest"] or capsule["provenance"]["head"] != rs["leaf"]["commit"]):
        raise _refuse("capsule_recipe_mismatch", "the LEAF capsule package/tree digest or provenance commit differs from the recipe")
    # 4. interpreter, release root, running code
    interp = _interpreter_record(limits, admit)
    py = _lock_get(lock, "runtime", "python")
    if not isinstance(py, dict) or {k: py.get(k) for k in _PY_KEYS} != interp:
        raise _refuse("interpreter_mismatch", "the running interpreter (spelling, symlink, realpath, size, sha256, venv dir + pyvenv.cfg) is not the interpreter record the runtime lock was prepared with")
    _check_release_root(source["release_masy_root"], sealed["masy"]["files"], rs["masy"]["launcher_sha256"], lock, limits, admit)
    closure = (gate.PRODUCTION_PARENT_CLOSURE, gate.PRODUCTION_PARENT_CLOSURE) if role == "parent" else (gate.PRODUCTION_WORKER_ALLOWED, gate.PRODUCTION_WORKER_REQUIRED)
    if check_running_code:
        gate.check_loaded_closure(sealed["masy"]["files"], limits, admit, allowed=closure[0], required=closure[1])
    # 5. the genuine V1 verify (parent only)
    verify = None
    if run_verify:
        verify = _run_verify_child(py["spelling"], os.path.join(source["release_masy_root"], "tools", "masy_runtime.py"), store, source["production_lock"], plim, admit,
                                   capability=capability, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    # NONCIRCULAR delivery algorithm: (1) CORE from the recipe-pinned/lock-checked/actually-read facts only; (2) the production identity includes the CORE; (3) identity_sha = hash(identity);
    # (4) the outer delivery record = CORE + paths + lock_id + identity_sha. The identity never contains the outer record, so no fixed point/placeholder exists.
    core = {"schema": DELIVERY_SCHEMA, "bundle_id": rs["maple"]["bundle_id"], "manifest_sha256": rs["maple"]["bundle_manifest_sha256"], "archive_sha256": rs["maple"]["archive_sha256"],
            "device": capability, "leaf_profile": entry["leaf_profile"], "leaf_variant": entry["leaf_variant"]}
    identity = production_identity(
        delivery_core=core, config_sha256=config_sha256, profile=profile, leaf_profile=entry["leaf_profile"], device=capability,
        recipe={"name": recipe["name"], "sha256": recipe_sha}, receipt={"sha256": receipt_sha},
        lock={"lock_id": source["production_lock"], "file_sha256": lock_file_sha, "store": store},
        sources={"masy": {"commit": rs["masy"]["commit"], "object_id": sealed["masy"]["object_id"], "package_digest": sealed["masy"]["package_digest"], "tree_digest": sealed["masy"]["tree_digest"],
                          "launcher_sha256": rs["masy"]["launcher_sha256"]},
                 "syrup": {"commit": rs["syrup"]["commit"], "object_id": sealed["syrup"]["object_id"], "package_digest": sealed["syrup"]["package_digest"], "tree_digest": sealed["syrup"]["tree_digest"]},
                 "maple": {"commit": rs["maple"]["commit"], "object_id": rs["maple"]["object_id"], "package_digest": rs["maple"]["package_digest"], "bundle_id": rs["maple"]["bundle_id"],
                           "bundle_manifest_sha256": rs["maple"]["bundle_manifest_sha256"], "archive_sha256": rs["maple"]["archive_sha256"]},
                 "leaf": {"commit": rs["leaf"]["commit"], "capsule_id": capsule["object_id"], "manifest_sha256": capsule["manifest_sha256"], "source_manifest_sha256": capsule["source_manifest_sha256"],
                          "tree_digest": capsule["tree_digest"], "package_digest": capsule["package_digest"]}},
        interpreter=dict(interp), release_masy_root=source["release_masy_root"],
        parent_code=_parent_code(sealed["masy"]["files"], gate.PRODUCTION_PARENT_CLOSURE))
    resolved_source = MappingProxyType({
        "mode": "production_committed", "profile": profile, "native_target": _lock_get(lock, "objects", "maple_materialized", "path"),
        "masy_root": sealed["masy"]["source_root"], "syrup_root": sealed["syrup"]["source_root"], "masy_manifest": "", "syrup_manifest": "",
        "origin": MappingProxyType({"label": "production_committed recipe " + recipe["name"], "declared_by": "release receipt " + receipt_sha})})
    identity_sha = gate.bootstrap_identity_sha(identity)
    delivery = {**core, "target": target, "leaf_root": capsule["source_root"], "syrup_root": os.path.join(sealed["syrup"]["source_root"], "src", "maple_syrup"),
                "masy_root": os.path.join(sealed["masy"]["source_root"], "src", "masy"), "lock_id": source["production_lock"], "production_identity_sha256": identity_sha}
    validate_delivery(delivery)
    return Resolved(identity, identity_sha, resolved_source,
                    MappingProxyType({"masy": MappingProxyType(dict(sealed["masy"]["files"])), "syrup": MappingProxyType(dict(sealed["syrup"]["files"]))}), capsule["source_root"],
                    None if verify is None else MappingProxyType(verify), MappingProxyType(delivery))
