"""Opt-in committed MAPLE/SYRUP/LEAF + MASY source locks (stdlib only).

An externally authenticated qualification is authority; observations never are.
No native import, installation, worktree mutation, model or child run occurs here.
The old three-source runtime and its locks are independent and unchanged.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shutil
import stat
import subprocess
import time
import tomllib
from urllib.parse import quote

QUALIFICATION_SCHEMA = "masy.triplet_qualification.v1"
LOCK_SCHEMA = "masy.triplet_lock.v1"
PROFILE = "masy.daily_wind_water_growth.cpu.v1"
FAMILIES = ("maple", "maple_syrup", "leaf", "masy")
MODELS = ("maple", "syrup", "leaf", "masy")
PACKAGE = {"maple": "maple", "syrup": "maple_syrup", "leaf": "leaf", "masy": "masy"}
PROJECT = {"maple": "maple", "syrup": "maple-syrup", "leaf": "leaf", "masy": "masy"}
SCOPES = {"maple": ("src", "pyproject.toml"), "syrup": ("src", "pyproject.toml"),
          "leaf": ("leaf", "pyproject.toml", "benchmarks/spatial_mapping.py", "benchmarks/__init__.py"),
          "masy": ("src", "pyproject.toml", "tools/masy_triplet.py")}
PYTHON = "/home/okin/MAPLE/.venv/bin/python"
DEFAULT_LIMITS = {"max_files": 10000, "max_source_bytes": 256 << 20, "max_file_bytes": 64 << 20,
                  "max_manifest_bytes": 8 << 20, "max_disk_bytes": 512 << 20,
                  "git_timeout_s": 30, "max_git_output_bytes": 16 << 20}
HEX = re.compile(r"[0-9a-f]{64}\Z")
OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
REF = re.compile(r"refs/heads/[A-Za-z0-9_./-]+\Z")
ENTRY = re.compile(r"masy\.[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*\Z")
CHUNK = 1 << 20


@dataclasses.dataclass(frozen=True)
class _Policy:
    """Private per-profile constants; ``_CPU`` reproduces the public v1 behavior exactly.

    A separate explicit profile module constructs its own frozen instance and
    passes it to the private helpers. Nothing here is mutated, defaulted from
    the environment, or discovered at runtime.
    """
    label: str
    qualification_schema: str
    lock_schema: str
    profile: str
    python: str
    interpretation: tuple
    generator_files: tuple
    extra_keys: tuple = ()
    extra_check: object = None


_CPU = _Policy("CPU", QUALIFICATION_SCHEMA, LOCK_SCHEMA, PROFILE, PYTHON,
               (("code_only", True), ("transport_qualified_by_lock", False), ("combined_restart", False),
                ("backend", "numpy"), ("immutable_upgrade", False)),
               ("src/masy/triplet_lock.py", "src/masy/triplet_runtime.py", "tools/masy_triplet.py"))


class TripletLockError(ValueError):
    def __init__(self, code, detail):
        self.code, self.detail = code, str(detail)[:1000]
        super().__init__(f"{code}: {self.detail}")


def _need(ok, code, detail):
    if not ok:
        raise TripletLockError(code, detail)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode("ascii")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _keys(value, keys, what):
    _need(type(value) is dict and set(value) == set(keys), "schema", f"closed {what} fields required")
    return value


def _sha(value):
    _need(type(value) is str and HEX.fullmatch(value) and value != "0" * 64,
          "unqualified_profile", "an external nonzero SHA-256 is required")
    return value


def _rel(value):
    _need(type(value) is str and value and "\\" not in value and "\x00" not in value,
          "path", "plain POSIX relative path required")
    p = PurePosixPath(value)
    _need(not p.is_absolute() and all(x not in ("", ".", "..") for x in value.split("/"))
          and ".git" not in p.parts, "path", f"unsafe relative path {value!r}")
    return value


def _path(value, *, directory=False):
    _need(type(value) is str and os.path.isabs(value) and os.path.normpath(value) == value,
          "path", "canonical absolute path required")
    p = Path(value)
    for q in (p, *p.parents):
        st = q.lstat()
        _need(not stat.S_ISLNK(st.st_mode), "symlink", str(q))
    st = p.lstat()
    _need(stat.S_ISDIR(st.st_mode) if directory else stat.S_ISREG(st.st_mode), "path", str(p))
    if not directory:
        _need(st.st_nlink == 1, "hardlink", str(p))
    return p


def _stamp(st):
    return (st.st_dev, st.st_ino, st.st_mode, st.st_nlink, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _read(path, cap, *, keep=True):
    p = _path(str(path))
    before = p.lstat()
    _need(before.st_size <= cap, "byte_limit", str(p))
    h, pieces, n = hashlib.sha256(), [], 0
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        _need(_stamp(os.fstat(fd)) == _stamp(before), "source_changed", str(p))
        while True:
            data = os.read(fd, min(CHUNK, cap - n + 1))
            if not data:
                break
            n += len(data)
            _need(n <= cap, "byte_limit", str(p))
            h.update(data)
            if keep:
                pieces.append(data)
        _need(n == before.st_size and _stamp(os.fstat(fd)) == _stamp(before)
              and _stamp(p.lstat()) == _stamp(before), "source_changed", str(p))
    finally:
        os.close(fd)
    return (b"".join(pieces) if keep else None), n, h.hexdigest()


def _json(data):
    def pairs(rows):
        out = {}
        for k, v in rows:
            _need(k not in out, "schema", "duplicate JSON key")
            out[k] = v
        return out
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise TripletLockError("schema", "bounded plain JSON required") from exc


def _reference(ref, cap):
    _keys(ref, ("path", "size", "sha256"), "file reference")
    _sha(ref["sha256"])
    _need(type(ref["size"]) is int and 0 <= ref["size"] <= cap, "byte_limit", "reference size")
    data, n, sha = _read(ref["path"], cap)
    _need(n == ref["size"] and sha == ref["sha256"], "reference_changed", ref["path"])
    return data


def file_reference(path, cap):
    _, n, sha = _read(str(path), cap, keep=False)
    return {"path": str(path), "size": n, "sha256": sha}


def _limits(values):
    _keys(values, DEFAULT_LIMITS, "limits")
    for k, upper in DEFAULT_LIMITS.items():
        _need(type(values[k]) is int and 0 < values[k] <= upper, "limits", k)
    return values


def _git(repo, args, limits, *, cap=None):
    """Bound both pipe streams before appending; no shell, optional locks or external diff."""
    repo = str(_path(repo, directory=True))
    cap = limits["max_git_output_bytes"] if cap is None else cap
    env = {"PATH": os.defpath, "LC_ALL": "C", "GIT_OPTIONAL_LOCKS": "0",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_TERMINAL_PROMPT": "0", "GIT_NO_REPLACE_OBJECTS": "1", "GIT_NO_LAZY_FETCH": "1"}
    proc = subprocess.Popen(["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", repo, *args], env=env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    streams, total = {proc.stdout: bytearray(), proc.stderr: bytearray()}, 0
    until = time.monotonic() + limits["git_timeout_s"]
    try:
        with selectors.DefaultSelector() as sel:
            for stream in streams:
                sel.register(stream, selectors.EVENT_READ)
            while sel.get_map():
                _need(time.monotonic() < until, "git_timeout", args[0])
                for key, _ in sel.select(min(0.2, max(0, until - time.monotonic()))):
                    part = os.read(key.fileobj.fileno(), min(65536, cap - total + 1))
                    if not part:
                        sel.unregister(key.fileobj)
                        continue
                    total += len(part)
                    _need(total <= cap, "git_output_limit", args[0])
                    streams[key.fileobj].extend(part)
        proc.wait(timeout=max(0.01, until - time.monotonic()))
        _need(proc.returncode == 0, "git", bytes(streams[proc.stderr])[:500].decode("utf-8", "replace"))
        return bytes(streams[proc.stdout])
    except subprocess.TimeoutExpired as exc:
        raise TripletLockError("git_timeout", args[0]) from exc
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdout.close()
        proc.stderr.close()


def _selected(spec, limits):
    repo, ref = spec["repo"], spec["ref"]
    _need(type(ref) is str and REF.fullmatch(ref) and ".." not in ref and not ref.endswith("/"), "ref", "explicit local branch ref required")
    root = _git(repo, ["rev-parse", "--show-toplevel"], limits).decode().strip()
    _need(root == repo, "path", "repository top level required")
    commit = _git(repo, ["rev-parse", "--verify", ref + "^{commit}"], limits).decode().strip()
    tree = _git(repo, ["rev-parse", "--verify", commit + "^{tree}"], limits).decode().strip()
    _need(OID.fullmatch(commit) and OID.fullmatch(tree), "ref", "plain commit/tree ids required")
    return commit, tree


def _tree_rows(name, spec, limits):
    commit, tree = _selected(spec, limits)
    raw = _git(spec["repo"], ["ls-tree", "-r", "-l", "-z", commit, "--", *SCOPES[name]], limits)
    rows, total = [], 0
    for entry in raw.split(b"\x00"):
        if not entry:
            continue
        fields, path = entry.split(b"\t", 1)
        mode, kind, blob, size = fields.split()
        rel = _rel(path.decode("utf-8", "strict"))
        _need(mode in (b"100644", b"100755") and kind == b"blob", "source_mode", rel)
        n = int(size)
        _need(n <= limits["max_file_bytes"], "byte_limit", rel)
        total += n
        _need(len(rows) < limits["max_files"] and total <= limits["max_source_bytes"], "source_limit", name)
        data = _git(spec["repo"], ["cat-file", "blob", blob.decode()], limits, cap=n + 4096)
        _need(len(data) == n, "source_changed", rel)
        # Git's own selected object is independently checked; no worktree read.
        algorithm = "sha1" if len(blob) == 40 else "sha256"
        _need(hashlib.new(algorithm, b"blob %d\0" % n + data).hexdigest() == blob.decode(), "git_blob", rel)
        rows.append([rel, n, digest(data), blob.decode()])
    rows.sort()
    package = "leaf/" if name == "leaf" else f"src/{PACKAGE[name]}/"
    _need(any(r[0] == package + "__init__.py" for r in rows) and any(r[0] == "pyproject.toml" for r in rows), "source_scope", name)
    if name == "leaf":
        _need(any(r[0] == "benchmarks/spatial_mapping.py" for r in rows)
              and not any(r[0] == "benchmarks/__init__.py" for r in rows), "helper_namespace", "exact namespace helper required; package initializer unsupported")
    else:
        _need(all(not r[0].startswith("src/") or r[0].startswith(package) for r in rows), "source_scope", "only the owning package may occupy src")
    return commit, tree, rows


def _package_digest(name, rows):
    prefix = "leaf/" if name == "leaf" else f"src/{PACKAGE[name]}/"
    h = hashlib.sha256()
    for rel, _, sha, _ in rows:
        if rel.startswith(prefix):
            h.update((rel[len(prefix):] + "\0" + sha + "\n").encode())
    return h.hexdigest()


def _active(name, spec, limits):
    """Observe excluded active bytes directly, without Git clean/filter drivers.

    ls-tree/ls-files are index/object reads; fsmonitor is disabled in _git.
    No status/diff, attributes conversion, clean process, or textconv runs.
    """
    repo = spec["repo"]
    head = _git(repo, ["rev-parse", "--verify", "HEAD^{commit}"], limits).decode().strip()
    raw = _git(repo, ["ls-tree", "-r", "-l", "-z", head, "--", *SCOPES[name]], limits)
    rows, total = [], 0
    for entry in raw.split(b"\x00"):
        if not entry:
            continue
        _need(len(rows) < limits["max_files"], "source_limit", "active tracked Source")
        fields, path = entry.split(b"\t", 1)
        mode, kind, blob, size = fields.split()
        rel = _rel(path.decode("utf-8", "strict"))
        _need(kind == b"blob", "source_mode", "active scope must contain ordinary Source entries")
        p = Path(repo) / rel
        if not os.path.lexists(p):
            rows.append([rel, "missing", None, None, blob.decode()])
            continue
        st = p.lstat()
        if stat.S_ISLNK(st.st_mode):
            rows.append([rel, "symlink_excluded", st.st_size, digest(os.readlink(p).encode()), blob.decode()])
            continue
        _need(stat.S_ISREG(st.st_mode), "source_mode", rel)
        _need(len(rows) < limits["max_files"] and st.st_size <= limits["max_file_bytes"], "source_limit", "active tracked Source")
        total += st.st_size
        _need(total <= limits["max_source_bytes"], "source_limit", "active tracked Source")
        _, n, sha = _read(p, limits["max_file_bytes"], keep=False)
        rows.append([rel, "regular", n, sha, blob.decode(), st.st_mode])
    index = _git(repo, ["ls-files", "--stage", "-z", "--", *SCOPES[name]], limits)
    untracked = _git(repo, ["ls-files", "--others", "--exclude-standard", "-z", "--", *SCOPES[name]], limits)
    return {"active_head": head, "tracked_worktree": rows,
            "index_sha256": digest(index), "untracked_paths_z": untracked.decode("utf-8", "surrogateescape"),
            "excluded_from_committed_snapshot": True, "untracked_contents_read": False,
            "git_status_or_clean_filters_used": False}


def resolve(sources, *, limits=None):
    """Return UNQUALIFIED observations of explicit refs; never generate authority."""
    limits = _limits(dict(DEFAULT_LIMITS) if limits is None else limits)
    _keys(sources, MODELS, "source selection")
    out = {}
    for name in MODELS:
        spec = _keys(sources[name], ("repo", "ref"), "source proposal")
        commit, tree, rows = _tree_rows(name, spec, limits)
        _need(_selected(spec, limits) == (commit, tree), "ref_advanced", name)
        out[name] = {**spec, "commit": commit, "tree": tree, "files": rows,
                     "package_sha256": _package_digest(name, rows), "active_excluded": _active(name, spec, limits)}
    return {"status": "UNQUALIFIED_OBSERVATION", "sources": out}



def draft_qualification(sources, *, interpreter, maple_bundle, native_profile, entrypoints=(), limits=None):
    """Construct an UNQUALIFIED descriptor for independent Root review.

    This does not turn observed hashes into authority. prepare refuses its
    status until Root independently qualifies the source frontier and supplies
    the final descriptor's external SHA.
    """
    policy = dict(DEFAULT_LIMITS) if limits is None else limits
    observations = resolve(sources, limits=policy)
    rows = {n: {k: v for k, v in s.items() if k != "active_excluded"}
            for n, s in observations["sources"].items()}
    descriptor = {"schema": QUALIFICATION_SCHEMA, "profile": PROFILE, "status": "unqualified",
                  "sources": rows, "interpreter": interpreter, "maple_bundle": maple_bundle,
                  "native_profile": native_profile, "limits": policy, "entrypoints": list(entrypoints)}
    return {"qualification": descriptor, "excluded_active_work": {
        n: s["active_excluded"] for n, s in observations["sources"].items()}}


def _source_spec(name, spec, limits):
    _keys(spec, ("repo", "ref", "commit", "tree", "files", "package_sha256"), f"{name} source")
    _path(spec["repo"], directory=True)
    _need(type(spec["commit"]) is str and OID.fullmatch(spec["commit"]) and int(spec["commit"], 16)
          and type(spec["tree"]) is str and OID.fullmatch(spec["tree"]) and int(spec["tree"], 16), "unqualified_profile", name)
    _need(type(spec["ref"]) is str and REF.fullmatch(spec["ref"]) and ".." not in spec["ref"] and not spec["ref"].endswith("/"), "ref", "explicit local branch required")
    _sha(spec["package_sha256"])
    rows = spec["files"]
    _need(type(rows) is list and 0 < len(rows) <= limits["max_files"], "source_limit", name)
    total, paths = 0, []
    for row in rows:
        _need(type(row) is list and len(row) == 4, "schema", "source row = path,size,sha,blob")
        rel, size, sha, blob = row
        _rel(rel); _sha(sha)
        _need(type(size) is int and 0 <= size <= limits["max_file_bytes"] and type(blob) is str and OID.fullmatch(blob), "schema", "source row")
        total += size
        paths.append(rel)
    _need(total <= limits["max_source_bytes"] and paths == sorted(set(paths)), "source_limit", "unique sorted rows")
    _need(_package_digest(name, rows) == spec["package_sha256"], "source_digest", name)


def _validate_qualification(q, policy=_CPU):
    _keys(q, ("schema", "profile", "status", "sources", "interpreter", "maple_bundle", "limits", "entrypoints", "native_profile", *policy.extra_keys), "qualification")
    _need(q["schema"] == policy.qualification_schema and q["profile"] == policy.profile and q["status"] == "qualified", "unqualified_profile", f"independently qualified {policy.label} descriptor required")
    _need(type(q["native_profile"]) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", q["native_profile"]), "unqualified_profile", "explicit qualified native source profile required")
    limits = _limits(q["limits"])
    _keys(q["sources"], MODELS, "sources")
    for name in MODELS:
        _source_spec(name, q["sources"][name], limits)
    _need(sum(len(s["files"]) for s in q["sources"].values()) <= limits["max_files"]
          and sum(r[1] for s in q["sources"].values() for r in s["files"]) <= limits["max_source_bytes"], "source_limit", "complete four-object frontier")
    _keys(q["interpreter"], ("path", "sha256", "version"), "interpreter")
    _need(q["interpreter"]["path"] == policy.python and type(q["interpreter"]["version"]) is list
          and len(q["interpreter"]["version"]) == 3 and all(type(x) is int and x >= 0 for x in q["interpreter"]["version"]), "interpreter", f"existing MAPLE {policy.label} interpreter required")
    _sha(q["interpreter"]["sha256"])
    _keys(q["maple_bundle"], ("target", "manifest"), "genuine MAPLE materialization")
    _need(type(q["entrypoints"]) is list and len(q["entrypoints"]) <= 16
          and len(set(q["entrypoints"])) == len(q["entrypoints"])
          and all(type(x) is str and ENTRY.fullmatch(x) for x in q["entrypoints"]), "entrypoint", "closed MASY entrypoint roster")
    if policy.extra_check is not None:
        policy.extra_check(q)
    return q


def _load_qualification(path, expected_sha256, policy):
    _sha(expected_sha256)
    data, _, sha = _read(str(path), DEFAULT_LIMITS["max_manifest_bytes"])
    _need(sha == expected_sha256, "qualification_sha256", "external qualification SHA differs")
    q = _validate_qualification(_json(data), policy)
    return q, {"path": str(path), "size": len(data), "sha256": sha}


def load_qualification(path, *, expected_sha256):
    return _load_qualification(path, expected_sha256, _CPU)


def _current(q):
    limits = q["limits"]
    active = {}
    for name in MODELS:
        spec = q["sources"][name]
        commit, tree, rows = _tree_rows(name, spec, limits)
        _need((commit, tree) == (spec["commit"], spec["tree"]), "ref_advanced", name + ": new qualification required; no fallback")
        _need(rows == spec["files"], "source_changed", name)
        active[name] = _active(name, spec, limits)
    return active


def _exact_tree(root, expected, limits):
    """Close every import-root file AND directory, including bytecode/empty extras."""
    root = _path(str(root), directory=True)
    dirs = {p for rel in expected for p in list(PurePosixPath(rel).parents)[:-1]}
    pending, seen, entries = [root], set(), 0
    while pending:
        parent = pending.pop()
        with os.scandir(parent) as scan:
            for item in scan:
                entries += 1
                _need(entries <= limits["max_files"] * 20, "source_limit", str(root))
                p = Path(item.path)
                rel = p.relative_to(root).as_posix()
                st = item.stat(follow_symlinks=False)
                _need(not stat.S_ISLNK(st.st_mode), "symlink", str(p))
                if stat.S_ISDIR(st.st_mode):
                    _need(PurePosixPath(rel) in dirs, "bundle_extra", rel)
                    pending.append(p)
                else:
                    _need(rel in expected, "bundle_extra", rel)
                    _, n, sha = _read(p, limits["max_file_bytes"], keep=False)
                    _need((n, sha) == expected[rel], "reference_changed", str(p))
                    seen.add(rel)
    _need(seen == set(expected), "bundle", "missing import-root files")


def _native_overlay(record, target, distribution):
    """Expected bytes of owning dependency_bundle._overlay_files, not a writer.

    This finite metadata comparison reproduces that public bundle recipe. The
    genuine owning check_runtime/verify_materialized still runs before MAPLE import.
    """
    project = record["project"]
    _need(project.get("name") == "maple" and type(project.get("version")) is str
          and re.fullmatch(r"[A-Za-z0-9_.+!-]{1,128}", project["version"]), "bundle", "MAPLE static metadata")
    version = project["version"]
    info = "maple-" + version + ".dist-info"
    files = {info + "/" + n: b for n, b in distribution.items()}
    files[info + "/INSTALLER"] = b"maple_syrup.dependency_bundle\n"
    files[info + "/direct_url.json"] = json.dumps(
        {"dir_info": {"editable": True}, "url": "file://" + quote(str(target / "source"))}, sort_keys=True).encode()
    files["__editable__.maple-" + version + ".pth"] = (str(target / "source/src") + "\n").encode()
    rows = [f"{k},sha256={base64.urlsafe_b64encode(hashlib.sha256(v).digest()).rstrip(b'=').decode()},{len(v)}" for k, v in sorted(files.items())]
    files[info + "/RECORD"] = ("\n".join(rows + [info + "/RECORD,,"]) + "\n").encode()
    return files


def _bundle(q):
    spec, limits = q["maple_bundle"], q["limits"]
    target = _path(spec["target"], directory=True)
    _need(spec["manifest"]["path"] == str(target / "bundle-manifest.json"), "bundle", "materialized manifest must be inside target")
    record = _json(_reference(spec["manifest"], limits["max_manifest_bytes"]))
    _need(type(record) is dict and record.get("schema") == "maple_syrup.dependency_bundle.v1" and target.name == record.get("bundle_id"), "bundle", "genuine materialized bundle required")
    origin = record.get("origin")
    _need(type(origin) is dict and type(origin.get("original_source")) is dict
          and origin["original_source"].get("revision") == q["sources"]["maple"]["commit"]
          and origin.get("patch_chain") == [], "bundle", "selected committed MAPLE without patches required")
    expected = {r[0]: (r[1], r[2]) for r in q["sources"]["maple"]["files"]}
    actual, distribution = {}, {}
    _need(type(record.get("files")) is list and len(record["files"]) <= limits["max_files"], "bundle", "bounded bundle table")
    total, seen = 0, set()
    for row in record["files"]:
        _keys(row, ("path", "size", "sha256"), "bundle row")
        rel = _rel(row["path"])
        _need(rel not in seen and rel.startswith(("source/", "distribution/")) and type(row["size"]) is int
              and 0 <= row["size"] <= limits["max_file_bytes"], "bundle", rel)
        seen.add(rel); total += row["size"]
        _need(total <= limits["max_source_bytes"], "bundle", "bundle byte bound")
        if rel.startswith("source/"):
            actual[rel[7:]] = (row["size"], row["sha256"])
        else:
            name = rel[len("distribution/"):]
            _need(name in ("METADATA", "WHEEL", "top_level.txt", "entry_points.txt"), "bundle", "closed native distribution files")
            distribution[name] = _reference({"path": str(target / rel), "size": row["size"], "sha256": row["sha256"]}, limits["max_file_bytes"])
    _need(actual == expected and {"METADATA", "WHEEL", "top_level.txt"} <= set(distribution), "bundle", "complete committed MAPLE / genuine metadata rows required")
    project = record.get("project")
    _need(type(project) is dict and type(project.get("package_source_digest")) is dict
          and project["package_source_digest"].get("digest_sha256") == q["sources"]["maple"]["package_sha256"], "bundle", "package digest")
    _exact_tree(target / "source", expected, limits)
    _exact_tree(target / "distribution", {n: (len(b), digest(b)) for n, b in distribution.items()}, limits)
    overlay = _native_overlay(record, target, distribution)
    _exact_tree(target / "editable", {n: (len(b), digest(b)) for n, b in overlay.items()}, limits)
    top = set()
    with os.scandir(target) as entries:
        for item in entries:
            top.add(item.name)
            _need(len(top) <= 5, "bundle_extra", "materialized top-level entries")
    _need(top == {"source", "distribution", "editable", "bundle-manifest.json", "materialization.json"}, "bundle_extra", "materialized top-level entries")
    materialization = canonical({"schema": "maple_syrup.dependency_materialization.v1", "bundle_id": record["bundle_id"], "target": str(target),
                                "layout": {"source": "source/", "distribution": "distribution/", "overlay": "editable/", "manifest": "bundle-manifest.json"}})
    _need(_read(target / "materialization.json", 1 << 16)[0] == materialization, "bundle", "materialization record differs")
    return record


def _interpreter(q):
    spec = q["interpreter"]
    # The existing venv interpreter may be a symlink; identity is the real binary.
    real = str(Path(spec["path"]).resolve(strict=True))
    _, _, sha = _read(real, DEFAULT_LIMITS["max_source_bytes"], keep=False)
    _need(sha == spec["sha256"], "interpreter", "interpreter binary changed")


def _new(path, data, *, mode=0o444):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
    except BaseException:
        raise
    os.chmod(path, mode)


def _fsync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _metadata(name, source, object_id):
    """Labelled static non-MAPLE overlay, never a wheel or installation."""
    p = tomllib.loads(_read(source / "pyproject.toml", 1 << 20)[0].decode("utf-8"))["project"]
    _need(p.get("name") == PROJECT[name] and "dynamic" not in p, "metadata", name)
    version = p.get("version")
    _need(type(version) is str and re.fullmatch(r"[A-Za-z0-9_.+!-]+", version), "metadata", "static safe version required")
    def line(s):
        _need(type(s) is str and "\n" not in s and "\r" not in s, "metadata", "one-line static field")
        return s
    lines = ["Metadata-Version: 2.1", "Name: " + PROJECT[name], "Version: " + version]
    for key, field in (("description", "Summary"), ("requires-python", "Requires-Python")):
        if key in p:
            lines.append(field + ": " + line(p[key]))
    for dep in p.get("dependencies", []):
        lines.append("Requires-Dist: " + line(dep))
    for extra, deps in p.get("optional-dependencies", {}).items():
        _need(re.fullmatch(r"[A-Za-z0-9_-]+", extra), "metadata", "extra name")
        lines.append("Provides-Extra: " + extra)
        for dep in deps:
            req, sep, marker = line(dep).partition(";")
            suffix = f'({marker.strip()}) and extra == "{extra}"' if sep else f'extra == "{extra}"'
            lines.append("Requires-Dist: " + req.strip() + "; " + suffix)
    info = PROJECT[name].replace("-", "_") + "-" + version + ".dist-info/"
    files = {info + "METADATA": ("\n".join(lines) + "\n").encode(),
             info + "top_level.txt": (PACKAGE[name] + "\n").encode(),
             info + "INSTALLER": b"masy.triplet_lock (metadata overlay; not installed)\n",
             info + "direct_url.json": canonical({"url": source.as_uri(), "dir_info": {"editable": True}}),
             info + "MASY_TRIPLET_OVERLAY.json": canonical({"schema": "masy.triplet_metadata_overlay.v1",
                 "source_object": object_id, "distribution": PROJECT[name], "installed": False, "wheel_built": False})}
    if p.get("scripts"):
        files[info + "entry_points.txt"] = ("[console_scripts]\n" + "".join(line(k) + " = " + line(v) + "\n" for k, v in sorted(p["scripts"].items()))).encode()
    rows = [f"{k},sha256={base64.urlsafe_b64encode(hashlib.sha256(v).digest()).rstrip(b'=').decode()},{len(v)}" for k, v in sorted(files.items())]
    files[info + "RECORD"] = ("\n".join(rows + [info + "RECORD,,"]) + "\n").encode()
    return files


def _snapshot(name, spec, objects, limits):
    identity = {k: spec[k] for k in ("commit", "tree", "files", "package_sha256")}
    oid = digest(canonical({"model": name, **identity}))
    obj = objects / oid
    obj.mkdir(mode=0o700)
    source = obj / "source"
    source.mkdir()
    for rel, n, sha, blob in spec["files"]:
        data = _git(spec["repo"], ["cat-file", "blob", blob], limits, cap=n + 4096)
        _need(len(data) == n and digest(data) == sha, "source_changed", rel)
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        _new(target, data)
    overlay = {}
    if name != "maple":
        overlay = _metadata(name, source, oid)
        for rel, data in overlay.items():
            target = obj / "metadata" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            _new(target, data)
    _new(obj / "identity.json", canonical({"model": name, **identity}))
    # Only this just-created object is frozen; no Root/status/artifact traversal.
    for root, dirs, _ in os.walk(obj, topdown=False):
        for child in dirs:
            os.chmod(Path(root) / child, 0o555)
        _fsync(root)
    os.chmod(obj, 0o555)
    return {"object_id": oid, "path": str(obj), "metadata_files": [[k, len(v), digest(v)] for k, v in sorted(overlay.items())]}


def import_prefix(lock):
    objects, bundle = lock["objects"], lock["qualification"]["maple_bundle"]["target"]
    return [str(Path(bundle) / "editable"), *[str(Path(objects[n]["path"]) / "metadata") for n in ("syrup", "leaf", "masy")],
            str(Path(objects["masy"]["path"]) / "source/src"), str(Path(objects["syrup"]["path"]) / "source/src"),
            str(Path(objects["leaf"]["path"]) / "source"), str(Path(bundle) / "source/src")]



def _generator(q, policy=_CPU):
    root = Path(__file__).absolute().parents[2]
    table = {r[0]: (r[1], r[2]) for r in q["sources"]["masy"]["files"]}
    for rel in policy.generator_files:
        _need(rel in table, "generator", "qualification must include actual provisioning code")
        _, n, sha = _read(root / rel, q["limits"]["max_file_bytes"], keep=False)
        _need((n, sha) == table[rel], "generator", "running provisioner differs from qualified committed code")


def _prepare(qualification_path, expected_sha256, destination, policy):
    q, qref = _load_qualification(qualification_path, expected_sha256, policy)
    _generator(q, policy); _interpreter(q); _bundle(q)
    before = _current(q)
    dest = Path(destination)
    _need(dest.is_absolute() and str(dest) == os.path.normpath(str(dest)), "path", "absolute new destination")
    _path(str(dest.parent), directory=True)
    _need(not os.path.lexists(dest), "output_exists", str(dest))
    # Source payload + bounded manifests/overlays + simultaneous read buffer. No model-fit claim.
    source_bytes = sum(r[1] for s in q["sources"].values() for r in s["files"])
    source_files = sum(len(s["files"]) for s in q["sources"].values())
    known_disk = source_bytes + 8 * q["limits"]["max_manifest_bytes"] + source_files * 4096
    _need(source_files <= q["limits"]["max_files"] and known_disk <= q["limits"]["max_disk_bytes"]
          and shutil.disk_usage(dest.parent).free >= known_disk, "disk_limit", "known code/metadata frontier not admitted")
    dest.mkdir(mode=0o700)
    objects = dest / "objects"
    objects.mkdir()
    try:
        sealed = {n: _snapshot(n, q["sources"][n], objects, q["limits"]) for n in MODELS}
        after = _current(q)
        _need(before == after, "source_changed", "refs or excluded active delta changed during provisioning")
        for name in MODELS:
            _verify_object(name, q["sources"][name], sealed[name], q["limits"])
        _bundle(q); _interpreter(q)
        _need(_json(_reference(qref, q["limits"]["max_manifest_bytes"])) == q, "qualification_sha256", "qualification changed during preparation")
        lock = {"schema": policy.lock_schema, "profile": policy.profile, "qualification": q, "qualification_ref": qref,
                "root": str(dest), "objects": sealed, "excluded_active_work": before,
                "interpretation": dict(policy.interpretation)}
        lock["sys_path_prefix"] = import_prefix(lock)
        path = dest / "triplet_lock.json"
        _new(path, canonical(lock))
        _fsync(objects); _fsync(dest); _fsync(dest.parent)
        return {"lock": file_reference(path, q["limits"]["max_manifest_bytes"]), "profile": policy.profile,
                "scope": "source provisioning only; no model or native compatibility check executed"}
    except BaseException:
        # Keep truthful partial output; never remove or repair a previous object.
        _fsync(dest)
        raise


def prepare(qualification_path, *, expected_sha256, destination):
    """No-clobber snapshot from qualified committed objects; no native/runtime run."""
    return _prepare(qualification_path, expected_sha256, destination, _CPU)


def _load(path, expected_sha256, policy):
    _sha(expected_sha256)
    raw, _, sha = _read(str(path), DEFAULT_LIMITS["max_manifest_bytes"])
    _need(sha == expected_sha256, "lock_sha256", "external lock SHA differs")
    lock = _json(raw)
    _keys(lock, ("schema", "profile", "qualification", "qualification_ref", "root", "objects", "excluded_active_work", "interpretation", "sys_path_prefix"), "lock")
    _need(lock["schema"] == policy.lock_schema and lock["profile"] == policy.profile, "schema", "triplet lock required")
    q = _validate_qualification(lock["qualification"], policy)
    _need(_json(_reference(lock["qualification_ref"], q["limits"]["max_manifest_bytes"])) == q, "qualification_sha256", "qualification differs from trusted original")
    _need(str(_path(lock["root"], directory=True) / "triplet_lock.json") == str(path), "path", "lock must remain in its original root")
    _keys(lock["objects"], MODELS, "objects")
    _keys(lock["excluded_active_work"], MODELS, "excluded active work")
    _need(lock["interpretation"] == dict(policy.interpretation), "schema", "scope differs")
    for name in MODELS:
        obj = _keys(lock["objects"][name], ("object_id", "path", "metadata_files"), "source object")
        identity = {k: q["sources"][name][k] for k in ("commit", "tree", "files", "package_sha256")}
        oid = digest(canonical({"model": name, **identity}))
        _need(obj["object_id"] == oid and obj["path"] == str(Path(lock["root"]) / "objects" / oid), "path", "content-addressed source object differs")
    _need(lock["sys_path_prefix"] == import_prefix(lock), "path", "sealed prefix differs")
    return lock


def load(path, *, expected_sha256):
    return _load(path, expected_sha256, _CPU)


def _verify_object(name, spec, obj, limits):
    root = _path(obj["path"], directory=True)
    _need(not root.lstat().st_mode & 0o222, "mutable_source", str(root))
    expected = {"source/" + r[0]: (r[1], r[2]) for r in spec["files"]}
    identity = canonical({"model": name, **{k: spec[k] for k in ("commit", "tree", "files", "package_sha256")}})
    expected["identity.json"] = (len(identity), digest(identity))
    overlay = {} if name == "maple" else _metadata(name, root / "source", obj["object_id"])
    _need(obj["metadata_files"] == [[r, len(b), digest(b)] for r, b in sorted(overlay.items())], "metadata", name)
    expected.update({"metadata/" + r: (len(b), digest(b)) for r, b in overlay.items()})
    actual = {}
    for parent, dirs, files in os.walk(root, followlinks=False):
        for dirname in dirs:
            d = _path(str(Path(parent) / dirname), directory=True)
            _need(not d.lstat().st_mode & 0o222, "mutable_source", str(d))
        for filename in files:
            p = Path(parent) / filename
            rel = p.relative_to(root).as_posix()
            _need(rel in expected and not p.lstat().st_mode & 0o222, "source_changed", rel)
            _, n, sha = _read(p, limits["max_file_bytes"], keep=False)
            actual[rel] = (n, sha)
            _need(len(actual) <= limits["max_files"] + 128, "source_limit", name)
    _need(actual == expected, "source_changed", name + ": missing/extra/changed snapshot file")


def _verify(path, expected_sha256, current_refs, policy):
    lock = _load(path, expected_sha256, policy)
    q = lock["qualification"]
    if current_refs:
        _current(q)
    for name in MODELS:
        _verify_object(name, q["sources"][name], lock["objects"][name], q["limits"])
    _bundle(q); _interpreter(q)
    return lock


def verify(path, *, expected_sha256, current_refs=True):
    """Verify code/bundle/interpreter; default also refuses advanced selected refs."""
    return _verify(path, expected_sha256, current_refs, _CPU)
