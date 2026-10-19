"""Immutable CuPy operational profiles, separate from the CPU v1 lock path.

Source preparation never imports the native stack in this process or CuPy in
workers. The optional identity probe is gated, bounded, and performs no model
calculation or kernel compilation. Existing CPU runtime files remain unchanged.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import gpu_runtime_worker as worker
from . import runtime as cpu

CONFIG_SCHEMA = "masy.gpu_runtime_config.v1"
PROFILE_SCHEMA = "masy.gpu_runtime_profile.v1"
PROBE_SCHEMA = "masy.gpu_runtime_probe.v1"
DEFAULT_STORE = "/home/okin/MASY/runs/gpu-runtime-store"
LAUNCHER = "tools/masy_gpu_runtime.py"
NEW_MODULES = {"src/masy/gpu_runtime.py", "src/masy/gpu_runtime_worker.py"}
_PROFILE_KEYS = {"schema", "profile_id", "store", "sources", "objects", "bundle", "metadata",
    "runtime", "device", "limits", "audit", "interpretation", "preparation_receipt"}
Refusal = cpu.RuntimeRefusal
MODULE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _bytes(value):
    return (cpu.canonical_json(value) + "\n").encode("ascii")


def _id(value, field):
    return hashlib.sha256(_bytes({k: v for k, v in value.items() if k != field})).hexdigest()


def _closed(value, keys, where):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise Refusal("gpu_schema_invalid", f"{where} has an invalid closed key set")
    return value


def _path(value, where):
    value = cpu._path_value(value, where)
    if os.path.realpath(value) != value:
        raise Refusal("gpu_path_invalid", f"{where} has a symlink or noncanonical parent")
    return value


def _read_json(path, bound=4 << 20):
    return cpu._strict_json(cpu._read_bounded(Path(path), bound, "gpu_input_unreadable"),
        "gpu_input_invalid", str(path))


def _file(path, bound=64 << 20):
    sha, _blob, size, _data = cpu._hash_path(Path(path), budget=bound)
    return {"path": str(path), "size": size, "sha256": sha}


def _native_loaded():
    families = {"maple", "maple_syrup", "numpy", "numba", "llvmlite", "cupy", "cupyx", "cupy_backends", "cuda"}
    return sorted(n for n in sys.modules if n.partition(".")[0] in families)


@dataclass(frozen=True)
class GPUConfig:
    native: cpu.RuntimeConfig
    masy: dict
    device: dict
    metadata_receipt: dict | None
    input_identity: dict | None = None


def parse_gpu_config(raw, *, input_identity=None):
    _closed(raw, {"schema", "sources", "runtime", "device", "limits", "metadata_receipt"}, "config")
    if raw["schema"] != CONFIG_SCHEMA:
        raise Refusal("gpu_schema_invalid", "GPU config schema differs")
    _closed(raw["sources"], {"maple", "syrup", "masy"}, "sources")
    masy = raw["sources"]["masy"]
    _closed(masy, {"mode", "root", "base_repository", "base_revision"}, "sources.masy")
    if masy["mode"] not in ("committed", "development_snapshot"):
        raise Refusal("gpu_schema_invalid", "explicit committed or development_snapshot selection required")
    masy = {**masy, "root": _path(masy["root"], "masy.root"),
        "base_repository": _path(masy["base_repository"], "masy.base_repository")}
    if cpu._revision(masy["base_revision"], "masy.base_revision") is None:
        raise Refusal("gpu_schema_invalid", "full base revision required")
    native_raw = {"schema": cpu.CONFIG_SCHEMA,
        "sources": {"maple": raw["sources"]["maple"], "syrup": raw["sources"]["syrup"]},
        "runtime": raw["runtime"], "limits": raw["limits"]}
    native = cpu.parse_runtime_config(native_raw)
    if native.python != worker.PYTHON:
        raise Refusal("gpu_interpreter_invalid", "only the existing MAPLE .venv-cupy interpreter is supported")
    _path(native.store, "runtime.store")
    if native.store == cpu.DEFAULT_STORE:
        raise Refusal("gpu_store_invalid", "GPU profiles must not use the CPU runtime store")
    receipt = raw["metadata_receipt"]
    if receipt is not None:
        _closed(receipt, {"path", "sha256"}, "metadata_receipt")
        _path(receipt["path"], "metadata_receipt.path")
        if not isinstance(receipt["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", receipt["sha256"]):
            raise Refusal("gpu_schema_invalid", "metadata receipt hash invalid")
    try:
        device = worker.check_device(raw["device"])
    except worker.WorkerRefusal as exc:
        raise Refusal(exc.code, exc.detail) from exc
    return GPUConfig(native, masy, device, receipt, input_identity)


def load_gpu_config(path):
    return parse_gpu_config(_read_json(path, 1 << 20), input_identity=_file(path, 1 << 20))


def _check_launch(launch, store):
    _closed(launch, {"masy_root", "launcher", "launcher_sha256", "run_id", "store", "argv"}, "launch")
    root = _path(launch["masy_root"], "launch.masy_root")
    if store != launch["store"] or not re.fullmatch(r"[0-9A-Za-z._-]{1,120}", launch["run_id"]):
        raise Refusal("gpu_launch_invalid", "store/run identity differs")
    if Path(__file__).resolve() != Path(root) / "src/masy/gpu_runtime.py" or \
            launch["launcher"] != str(Path(root) / LAUNCHER) or \
            _file(launch["launcher"])["sha256"] != launch["launcher_sha256"]:
        raise Refusal("gpu_launch_invalid", "running complete MASY source/launcher identity differs")
    if _file(__file__)["sha256"] != MODULE_SHA256 or _file(worker.__file__)["sha256"] != worker.MODULE_SHA256:
        raise Refusal("source_changed_during_prepare", "running GPU module source changed after import")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1 or _native_loaded():
        raise Refusal("gpu_launch_invalid", "parent must be -I -B and native-stack free")
    if sys.executable not in ("/home/okin/MAPLE/.venv/bin/python", worker.PYTHON):
        raise Refusal("gpu_launch_invalid", "parent interpreter is not an allowed existing runtime")


def _git(root, args, limits):
    return cpu._git(root, args, limits)


def observe_masy(config, limits):
    """Complete source scope; an explicit snapshot is never treated as Git HEAD."""
    root, repo, base = config["root"], config["base_repository"], config["base_revision"]
    launcher = Path(root) / LAUNCHER
    launcher_record = _file(launcher, cpu.MAX_SMALL_FILE_BYTES)
    if config["mode"] == "committed":
        observation = cpu.observe_source("masy", repo, expected_revision=base, limits=limits)
        baseline, _captured = cpu.hash_scope(observation, limits)
        if root != repo:
            raise Refusal("snapshot_not_selected", "committed mode root differs from repository")
        status = _git(repo, ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--", LAUNCHER], limits)
        committed = _git(repo, ["show", f"{base}:{LAUNCHER}"], limits)
        if status or hashlib.sha256(committed).hexdigest() != launcher_record["sha256"]:
            raise Refusal("dirty_scope", "GPU launcher is not the current committed source")
        return {**observation, "source_status": "committed", "launcher_gpu": launcher_record}, baseline, _captured
    # Explicit development lineage reads immutable base blobs, never the live
    # repository's possibly newer runtime files. Its current HEAD is observed
    # truthfully and compared before/after; the baseline is not relabelled HEAD.
    object_format = _git(repo, ["rev-parse", "--show-object-format"], limits).decode().strip()
    current_head = _git(repo, ["rev-parse", "HEAD"], limits).decode().strip()
    base_tree = cpu.head_tree(repo, base, limits)
    entries = cpu.ls_scope(repo, base, [*cpu.SCOPE_PATHSPECS, cpu.LAUNCHER_RELPATH], limits)
    baseline = []
    remaining = limits.max_bytes
    for relative, _mode, blob in entries:
        cpu._check_relpath(relative, launcher_allowed=True)
        data = _git(repo, ["show", f"{base}:{relative}"], limits)
        remaining -= len(data)
        if remaining < 0:
            raise Refusal("max_bytes_exceeded", "base source exceeds snapshot bound")
        baseline.append([relative, len(data), hashlib.sha256(data).hexdigest(), blob])
    if NEW_MODULES & {row[0] for row in baseline}:
        raise Refusal("snapshot_invalid", "this qualified snapshot mode requires GPU modules to be additions")
    # The candidate is a complete physical tree; no sparse package roots or
    # import stitching. Runtime additions are closed for this qualified v1.
    allowed = {row[0] for row in baseline} | NEW_MODULES
    found = set()
    for directory, dirs, names in os.walk(Path(root) / "src", followlinks=False):
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise Refusal("snapshot_invalid", "symlink directory in candidate")
        for name in names:
            found.add(str((Path(directory) / name).relative_to(root)))
    found |= {"pyproject.toml", cpu.LAUNCHER_RELPATH}
    if found != allowed:
        raise Refusal("snapshot_invalid", "complete runtime snapshot file set differs", extra=sorted(found - allowed),
            missing=sorted(allowed - found))
    before_files, captured = [], {}
    remaining = limits.max_bytes
    base_files = {row[0]: row for row in baseline}
    for relative in sorted(found):
        path = Path(root) / relative
        if os.path.realpath(path) != str(path):
            raise Refusal("snapshot_invalid", f"symlink in {relative}")
        keep = relative in {"pyproject.toml", "src/masy/__init__.py"}
        sha, blob, size, data = cpu._hash_path(path, budget=remaining, blob_algorithm=object_format, keep=keep)
        remaining -= size
        entry = [relative, size, sha, blob]
        if relative in base_files and entry != base_files[relative]:
            raise Refusal("snapshot_invalid", f"baseline runtime changed: {relative}")
        before_files.append(entry)
        if data is not None:
            captured[relative] = data
    if len(before_files) > limits.max_files:
        raise Refusal("max_files_exceeded", "snapshot file count bound exceeded")
    record = {"name": "masy", "root": root, "object_format": object_format,
        "source_status": "development_snapshot", "committed": False, "base_repository": repo,
        "base_revision": base, "base_tree": base_tree, "base_repository_head_observed": current_head,
        "files": before_files, "delta": {"added": sorted(NEW_MODULES), "modified": [], "deleted": []},
        "snapshot_sha256": hashlib.sha256(_bytes(before_files)).hexdigest(), "launcher_gpu": launcher_record,
        "generated_metadata": "unchanged runtime.py generator; new complete-source object/path binding"}
    return record, before_files, captured


def _metadata(config, maple, files):
    dist = cpu.distribution_metadata_record(config.native.maple_distribution_metadata)
    if config.metadata_receipt is None:
        if not dist["path"].startswith(worker.VENV + "/lib/") or not dist["path"].endswith(".dist-info"):
            raise Refusal("metadata_origin_unverified", "non-installed metadata needs an exact build receipt")
        return {"origin": "installed", "distribution": dist, "build_receipt": None}
    selector = config.metadata_receipt
    actual = _file(selector["path"])
    if actual["sha256"] != selector["sha256"]:
        raise Refusal("metadata_origin_unverified", "build receipt hash differs")
    receipt = _read_json(selector["path"])
    inputs = Path(selector["path"]).parent / "input_manifest.json"
    manifest = _read_json(inputs, 16 << 20)
    if receipt.get("committed_head") != maple["head"] or receipt.get("metadata_path") != dist["path"] or \
            receipt.get("metadata_files") != dist["files"] or \
            receipt.get("input_manifest_sha256") != _file(inputs)["sha256"] or manifest.get("head") != maple["head"]:
        raise Refusal("metadata_origin_unverified", "build/source/metadata binding differs")
    expected = {rel: {"size": size, "sha256": sha} for rel, size, sha, _blob in files}
    if manifest.get("files") != expected:
        raise Refusal("metadata_origin_unverified", "build input manifest differs from complete selected native source")
    artifact = _file(_path(receipt.get("artifact"), "build artifact"), config.native.limits.max_bytes)
    if artifact["sha256"] != receipt.get("artifact_sha256"):
        raise Refusal("metadata_origin_unverified", "built artifact hash differs")
    return {"origin": "wheel_build_receipt", "distribution": dist,
        "build_receipt": actual, "input_manifest": _file(inputs), "artifact": artifact}


def _observe(config):
    limits = config.native.limits
    observations, hashes = {}, {}
    for kind in ("maple", "syrup"):
        spec = getattr(config.native, kind)
        observations[kind] = cpu.observe_source(kind, spec.root, expected_revision=spec.expected_revision, limits=limits)
        hashes[kind] = cpu.hash_scope(observations[kind], limits)
    observations["masy"], files, captured = observe_masy(config.masy, limits)
    hashes["masy"] = (files, captured)
    metadata = _metadata(config, observations["maple"], hashes["maple"][0])
    if config.input_identity is not None and _file(config.input_identity["path"]) != config.input_identity:
        raise Refusal("source_changed_during_prepare", "config bytes changed")
    return observations, hashes, metadata


_LAYOUT = {"objects", "profiles", "probes", "runs"}


def validate_store(store):
    cpu.check_store_path(store)
    if set(os.listdir(store)) != _LAYOUT:
        raise Refusal("gpu_store_invalid", "GPU store top-level entries differ")
    if set(os.listdir(Path(store) / "objects")) != set(cpu.OBJECT_SUBDIRS):
        raise Refusal("gpu_store_invalid", "GPU object areas differ")
    for area in ("profiles", "probes", "runs", *(f"objects/{a}" for a in cpu.OBJECT_SUBDIRS)):
        root = Path(store) / area
        cpu._require_real_dir(root)
        for child in root.iterdir():
            if child.is_symlink():
                raise Refusal("gpu_store_invalid", "symlink store entry")
            if area in ("profiles", "probes"):
                valid = bool(re.fullmatch(r"(?:[0-9a-f]{64}|\.staging-[0-9A-Za-z._-]+)\.json", child.name)) and child.is_file()
            elif area == "runs":
                valid = bool(re.fullmatch(r"[0-9A-Za-z._-]{1,120}", child.name)) and child.is_dir()
            else:
                rules = cpu._STORE_ENTRY_RULES[area.rsplit("/", 1)[-1]]
                kind = "dir" if child.is_dir() else "file" if child.is_file() else "other"
                valid = any(pattern.fullmatch(child.name) and kind == expected for pattern, expected in rules)
            if not valid:
                raise Refusal("gpu_store_invalid", f"unknown store entry {child}")


def _ensure_store(store):
    path = Path(store)
    if not path.exists():
        cpu._require_real_dir(path.parent)
        path.mkdir(mode=0o700)
        for area in ("objects", "profiles", "probes", "runs"):
            (path / area).mkdir(mode=0o700)
        for area in cpu.OBJECT_SUBDIRS:
            (path / "objects" / area).mkdir(mode=0o700)
    validate_store(store)


class _Run:
    def __init__(self, store, run_id):
        self.dir = Path(store) / "runs" / run_id
        self.dir.mkdir(mode=0o700)
        self.run_id, self.receipts = run_id, []
        (self.dir / "cache").mkdir(mode=0o700)
        for relative in worker._CACHE_PATHS.values():
            (self.dir / relative).mkdir(mode=0o700)
        self.pycache = str(self.dir / "cache/pycache")

    def write(self, name, value, *, raw=False):
        path = self.dir / name
        cpu._write_new(path, value if raw else _bytes(value))
        return path

    def stage(self, operation, status, **values):
        # A fresh file per stage preserves failures without an append/overwrite API.
        index = len(list(self.dir.glob("stage-*.json"))) + 1
        self.write(f"stage-{index:03d}.json", {"operation": operation, "status": status,
            "unix_s": time.time(), **values})


def _bootstrap(masy_object):
    path = Path(masy_object["path"]) / "source/src/masy/gpu_runtime_worker.py"
    tree = ast.parse(cpu._read_bounded(path, 1 << 20, "bootstrap_unavailable"))
    values = [node.value.value for node in tree.body if isinstance(node, ast.Assign) and
        len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "BOOTSTRAP" and
        isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
    if len(values) != 1:
        raise Refusal("bootstrap_unavailable", "sealed worker has no unique literal bootstrap")
    return values[0]


def _request(run, python, device, prefix, packages, limits):
    return {"python": cpu._python_request(python), "run": {"run_dir": str(run.dir),
        "pycache_prefix": run.pycache, "env": worker.worker_env(run.dir, device["uuid"])},
        "sys_path_prefix": prefix, "packages": packages, "device": device,
        "limits": {key: getattr(limits, key) for key in ("max_files", "max_bytes", "max_compiled_files", "max_compiled_bytes")}}


def _call(run, operation, body, bootstrap, limits):
    request = {"schema": worker.REQUEST_SCHEMA, "operation": operation, **body}
    index = len(run.receipts) + 1
    path = run.write(f"request-{index}-{operation}.json", request)
    argv = [worker.PYTHON, "-I", "-B", "-X", f"pycache_prefix={run.pycache}", "-c", bootstrap, str(path)]
    receipt = {"operation": operation, "request": _file(path), "argv": argv, "cwd": str(run.dir),
        "env": body["run"]["env"], "limits": limits.record(), "started_unix_s": time.time()}
    run.receipts.append(receipt)
    run.stage(operation, "started")
    try:
        result = cpu.run_bounded(argv, env=body["run"]["env"], timeout_s=limits.worker_timeout_s,
            max_output_bytes=limits.worker_max_output_bytes, cwd=str(run.dir))
    except Refusal as exc:
        for suffix, data in zip(("stdout", "stderr"), getattr(exc, "partial", (b"", b""))):
            receipt[suffix] = _file(run.write(f"worker-{index}.{suffix}", data, raw=True))
        receipt.update(status="failed", code=exc.code, reaped=True, evidence=exc.evidence)
        run.write(f"worker-{index}-receipt.json", receipt)
        raise
    for suffix, data in (("stdout", result.stdout), ("stderr", result.stderr)):
        receipt[suffix] = _file(run.write(f"worker-{index}.{suffix}", data, raw=True))
    receipt.update(pid=result.pid, exit_code=result.returncode, reaped=True, elapsed_s=result.elapsed_s)
    try:
        output = cpu._strict_json(result.stdout, "worker_failed", "GPU worker result")
        if not isinstance(output, dict) or output.get("schema") != worker.RESULT_SCHEMA or \
                output.get("operation") != operation or output.get("status") not in ("ok", "refused") or \
                result.returncode != (0 if output["status"] == "ok" else 2):
            raise Refusal("worker_failed", "GPU worker did not return one compatible result")
        receipt["process"] = output.get("process")
        if output["status"] != "ok":
            raise Refusal(output.get("code", "worker_refused"), output.get("detail", ""),
                worker_evidence=output.get("evidence"), run_dir=str(run.dir))
        _closed(output, {"schema", "operation", "status", "result", "process", "interpretation"}, "worker result")
        process = output["process"]
        if process["pid"] != result.pid or process["cwd"] != str(run.dir) or \
                process["pycache_prefix"] != run.pycache or process["sys_path_prefix"] != body["sys_path_prefix"] or \
                any(process["environment"].get(k) != v for k, v in body["run"]["env"].items()):
            raise Refusal("worker_failed", "worker operational identity differs from launched child")
        expected_interpretation = dict(worker.INTERPRETATION)
        if operation == "probe":
            expected_interpretation.update(device_identity_observed=True, identity_probe_passed=True)
        if output["interpretation"] != expected_interpretation:
            raise Refusal("worker_failed", "unexpected readiness claims")
        receipt["status"] = "ok"
        return output
    finally:
        run.write(f"worker-{index}-receipt.json", receipt)


def _publish(store, area, value, field, run_id):
    identity = _id(value, field)
    value = {**value, field: identity}
    staging = Path(store) / area / f".staging-{run_id}.json"
    final = Path(store) / area / f"{identity}.json"
    cpu._write_new(staging, _bytes(value))
    staging.chmod(0o444)
    try:
        cpu._rename_noreplace(staging, final)
    except FileExistsError:
        if cpu._read_bounded(final, 32 << 20, "profile_collision") != _bytes(value):
            raise Refusal("profile_collision", "existing immutable object differs; staging retained")
    cpu._fsync_dir(final.parent)
    return value, final


def _verify_preparation_receipt(store, record, limits):
    """A valid environment cannot excuse missing or changed launch evidence."""
    _closed(record, {"path", "size", "sha256"}, "preparation receipt")
    path = Path(_path(record["path"], "preparation receipt path"))
    if path.name != "prepare-receipt.json" or path.parent.parent != Path(store) / "runs":
        raise Refusal("profile_tamper", "preparation receipt is outside this store's run")
    if _file(path, limits.max_evidence_bytes) != record:
        raise Refusal("profile_tamper", "preparation receipt content differs")
    receipt = _read_json(path, limits.max_evidence_bytes)
    _closed(receipt, {"launch", "config", "before_after_unchanged", "workers", "disk_preflight"}, "preparation receipt content")
    if receipt["before_after_unchanged"] is not True or not isinstance(receipt["workers"], list) or \
            [r.get("operation") for r in receipt["workers"]] != ["export", "materialize", "audit", "audit"]:
        raise Refusal("profile_tamper", "incomplete preparation worker evidence")
    remaining = limits.max_evidence_bytes - record["size"]
    for index, child in enumerate(receipt["workers"], 1):
        if child.get("status") != "ok" or child.get("exit_code") != 0 or child.get("reaped") is not True or \
                not isinstance(child.get("process"), dict) or child["process"].get("pid") != child.get("pid") or \
                not str(child["process"].get("start_ticks", "")).isdigit():
            raise Refusal("profile_tamper", "missing accepted child identity/exit evidence")
        if cpu._read_bounded(path.parent / f"worker-{index}-receipt.json", remaining, "profile_tamper") != _bytes(child):
            raise Refusal("profile_tamper", "per-worker receipt differs from accepted preparation")
        for key in ("request", "stdout", "stderr"):
            selector = child.get(key)
            _closed(selector, {"path", "size", "sha256"}, f"worker {key}")
            selected = Path(_path(selector["path"], f"worker {key} path"))
            if selected.parent != path.parent or type(selector["size"]) is not int or selector["size"] < 0 or \
                    _file(selected, remaining) != selector:
                raise Refusal("profile_tamper", "worker request/raw stream content differs or escapes its run")
            remaining -= selector["size"]
    return receipt


def _profile_body(profile, run, python):
    objects, bundle = profile["objects"], profile["bundle"]
    paths = {kind: objects[kind]["path"] for kind in cpu.SOURCES}
    paths.update(bundle=bundle["dir"], materialized=bundle["target"])
    prefix = cpu.runtime_sys_path(paths)
    if prefix != profile["runtime"]["sys_path_prefix"]:
        raise Refusal("profile_tamper", "profile prefix differs from exact source objects")
    cpu.check_import_roots(prefix)
    packages = {"masy": {"dir": str(Path(paths["masy"]) / "source/src/masy"), "digest": objects["masy"]["package_digest"]},
        "maple_syrup": {"dir": str(Path(paths["syrup"]) / "source/src/maple_syrup"), "digest": objects["syrup"]["package_digest"]},
        "maple": {"dir": str(Path(paths["materialized"]) / "source/src/maple"), "digest": objects["maple"]["package_digest"]}}
    distributions = [{"name": "maple", "dir": str(Path(paths["materialized"]) / "editable"),
        "files": cpu._editable_files(Path(paths["materialized"]) / "editable")}]
    for kind in ("syrup", "masy"):
        distributions.append({"name": cpu.SOURCES[kind][1], "dir": str(Path(paths[kind]) / "editable"),
            "files": objects[kind]["overlay_files"]})
    limits = cpu.RuntimeLimits.from_mapping(profile["limits"])
    return {**_request(run, python, profile["device"], prefix, packages, limits),
        "bundle": {key: bundle[key] for key in ("dir", "bundle_id", "target", "tool_module_sha256")},
        "distributions": distributions}


def _protected(store, masy_root, native_roots, python, dist=None, base_repository=None):
    protected = {"interpreter environment": worker.VENV, "interpreter binary": python["realpath"]}
    if dist is not None:
        protected["native metadata"] = dist
    for root in native_roots:
        protected[f"native {root}"] = root
    if base_repository:
        # A Git base used only for immutable blob lookup is not the running
        # wrapper. Protect its actual inputs, while permitting the separately
        # authorized isolated task store outside its src/tools/.git paths.
        protected.update({f"base {relative}": str(Path(base_repository) / relative)
            for relative in ("src", "tools", "pyproject.toml", ".git")})
    cpu.check_store_location(store, wrapper_roots=[masy_root], protected=protected)


def prepare_profile(config, *, launch):
    limits, store = config.native.limits, config.native.store
    _check_launch(launch, store)
    if config.masy["root"] != launch["masy_root"]:
        raise Refusal("gpu_launch_invalid", "snapshot is not the complete running MASY source")
    python = cpu.interpreter_identity(worker.PYTHON, limits)
    _protected(store, config.masy["root"], [config.native.maple.root, config.native.syrup.root],
        python, config.native.maple_distribution_metadata, config.masy["base_repository"])
    before, hashed, metadata = _observe(config)
    estimate = sum(sum(row[1] for row in value[0]) for value in hashed.values()) * 5 + \
        8 * limits.worker_max_output_bytes + limits.max_evidence_bytes
    disk = cpu.disk_preflight(store, estimate, limits)
    _ensure_store(store)
    run = _Run(store, launch["run_id"])
    run.stage("prepare", "started", disk=disk)
    try:
        objects = {}
        for kind in ("maple", "syrup", "masy"):
            files, captured = hashed[kind]
            objects[kind] = cpu.seal_source(store, kind, before[kind], files, captured,
                run_id=run.run_id, limits=limits)
        bootstrap = _bootstrap(objects["masy"])
        if bootstrap != worker.BOOTSTRAP:
            raise Refusal("source_changed_during_prepare", "sealed worker bootstrap differs from running source")
        paths = {kind: obj["path"] for kind, obj in objects.items()}
        prefix = cpu.bundle_sys_path(paths)
        cpu.check_import_roots(prefix)
        packages = {"masy": {"dir": str(Path(paths["masy"]) / "source/src/masy"), "digest": objects["masy"]["package_digest"]},
            "maple_syrup": {"dir": str(Path(paths["syrup"]) / "source/src/maple_syrup"), "digest": objects["syrup"]["package_digest"]}}
        tool_sha = next(row[2] for row in hashed["syrup"][0] if row[0] == "src/maple_syrup/dependency_bundle.py")
        origin = cpu._origin_record(before["maple"], objects["maple"]["object_id"])
        origin_path = run.write("origin.json", origin)
        export = Path(store) / "objects/maple-bundles" / f".export-{run.run_id}"
        body = _request(run, python, config.device, prefix, packages, limits)
        body.update(export={"project": str(Path(paths["maple"]) / "source"),
            "distribution": config.native.maple_distribution_metadata, "origin": str(origin_path), "output": str(export)},
            tool_module_sha256=tool_sha)
        exported = _call(run, "export", body, bootstrap, limits)["result"]["manifest"]
        bundle_id = exported["bundle_id"]
        data = cpu._read_bounded(export / "bundle-manifest.json", cpu.MAX_BUNDLE_MANIFEST_BYTES, "bundle_invalid")
        cpu.check_bundle_manifest(data, bundle_id=bundle_id, tool_sha256=tool_sha, maple_object=objects["maple"],
            maple_files=hashed["maple"][0], distribution=metadata["distribution"], origin=origin,
            package_digest=before["maple"]["package"]["digest_sha256"])
        final_bundle = export.parent / bundle_id
        try:
            cpu._rename_noreplace(export, final_bundle)
        except FileExistsError:
            if cpu._read_bounded(final_bundle / "bundle-manifest.json", cpu.MAX_BUNDLE_MANIFEST_BYTES, "bundle_collision") != data:
                raise Refusal("bundle_collision", "existing bundle differs; export retained")
        cpu._fsync_dir(final_bundle.parent)
        target = Path(store) / "objects/maple-materialized" / bundle_id
        body = _request(run, python, config.device, prefix, packages, limits)
        body["bundle"] = {"dir": str(final_bundle), "bundle_id": bundle_id, "cache": str(target.parent),
            "target": str(target), "tool_module_sha256": tool_sha}
        _call(run, "materialize", body, bootstrap, limits)
        paths.update(materialized=str(target), bundle=str(final_bundle))
        prefix = cpu.runtime_sys_path(paths)
        profile = {"schema": PROFILE_SCHEMA, "profile_id": "", "store": store, "sources": before,
            "objects": objects, "bundle": {"dir": str(final_bundle), "bundle_id": bundle_id, "target": str(target),
                "tool_module_sha256": tool_sha, "manifest_sha256": hashlib.sha256(data).hexdigest()},
            "metadata": metadata, "runtime": {"python": cpu._python_stable(python),
                "bootstrap_sha256": hashlib.sha256(bootstrap.encode()).hexdigest(), "sys_path_prefix": prefix,
                "stable_environment": {**worker._STABLE_ENV, "CUDA_VISIBLE_DEVICES": config.device["uuid"]},
                "cache_policy": dict(worker._CACHE_PATHS), "home": "absent", "launcher": before["masy"]["launcher_gpu"]},
            "device": config.device, "limits": limits.record(), "audit": {},
            "interpretation": dict(worker.INTERPRETATION), "preparation_receipt": {}}
        first = _call(run, "audit", _profile_body(profile, run, python), bootstrap, limits)
        after, _again, metadata_after = _observe(config)
        if before != after or metadata != metadata_after or cpu._python_stable(cpu.interpreter_identity(worker.PYTHON, limits)) != \
                profile["runtime"]["python"]:
            raise Refusal("source_changed_during_prepare", "source/metadata/interpreter changed")
        second = _call(run, "audit", _profile_body(profile, run, python), bootstrap, limits)
        if first["result"] != second["result"]:
            raise Refusal("environment_changed", "independent audit workers observed different source/environment/binaries")
        profile["audit"] = second["result"]
        receipt = run.write("prepare-receipt.json", {"launch": launch, "config": config.input_identity,
            "before_after_unchanged": True, "workers": run.receipts, "disk_preflight": disk})
        profile["preparation_receipt"] = _file(receipt, limits.max_evidence_bytes)
        profile, path = _publish(store, "profiles", profile, "profile_id", run.run_id)
        run.stage("prepare", "accepted", profile_id=profile["profile_id"])
        return {"verdict": "gpu_source_environment_profile", "profile_id": profile["profile_id"],
            "profile_path": str(path), "run_dir": str(run.dir), "interpretation": profile["interpretation"]}
    except Exception as exc:
        run.stage("prepare", "failed", error_type=type(exc).__name__, detail=str(exc)[:4000])
        raise


def _load_profile(store, ref):
    validate_store(store)
    identity = Path(ref).stem if os.path.isabs(ref) else ref
    if not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise Refusal("profile_ref_invalid", "explicit profile ID required")
    path = Path(store) / "profiles" / f"{identity}.json"
    if os.path.isabs(ref) and str(path) != ref:
        raise Refusal("profile_ref_invalid", "profile path differs from store")
    profile = _read_json(path, 32 << 20)
    _closed(profile, _PROFILE_KEYS, "profile")
    if profile["schema"] != PROFILE_SCHEMA or profile["store"] != store or profile["profile_id"] != identity or \
            _id(profile, "profile_id") != identity or profile["interpretation"] != worker.INTERPRETATION:
        raise Refusal("profile_tamper", "profile content identity/schema/readiness differs")
    worker.check_device(profile["device"])
    limits = cpu.RuntimeLimits.from_mapping(profile["limits"])
    _verify_preparation_receipt(store, profile["preparation_receipt"], limits)
    _closed(profile["objects"], set(cpu.SOURCES), "objects")
    for kind, record in profile["objects"].items():
        expected_path = str(Path(store) / "objects/sources" / record["object_id"])
        if record["path"] != expected_path:
            raise Refusal("profile_tamper", "source object path differs from identity")
        actual = cpu.verify_source_object(expected_path, record["object_id"], limits)
        if any(actual[key] != record[key] for key in actual):
            raise Refusal("profile_tamper", f"{kind} source object differs")
    bundle = profile["bundle"]
    if not re.fullmatch(r"[0-9a-f]{64}", bundle["bundle_id"]):
        raise Refusal("profile_tamper", "invalid native bundle ID")
    for key, area in (("dir", "maple-bundles"), ("target", "maple-materialized")):
        target = Path(store) / "objects" / area / bundle["bundle_id"]
        if bundle[key] != str(target) or _file(target / "bundle-manifest.json")["sha256"] != bundle["manifest_sha256"]:
            raise Refusal("profile_tamper", "native bundle path/manifest differs")
    bootstrap = _bootstrap(profile["objects"]["masy"])
    if hashlib.sha256(bootstrap.encode()).hexdigest() != profile["runtime"]["bootstrap_sha256"]:
        raise Refusal("profile_tamper", "bootstrap differs")
    return profile, limits, bootstrap


def verify_profile(store, profile_ref, *, launch):
    _check_launch(launch, store)
    profile, limits, bootstrap = _load_profile(store, profile_ref)
    python = cpu.interpreter_identity(worker.PYTHON, limits)
    if cpu._python_stable(python) != profile["runtime"]["python"]:
        raise Refusal("environment_changed", "interpreter identity differs from profile")
    _protected(store, launch["masy_root"], [], python)
    cpu.disk_preflight(store, 3 * limits.worker_max_output_bytes + limits.max_evidence_bytes, limits)
    run = _Run(store, launch["run_id"])
    audited = _call(run, "audit", _profile_body(profile, run, python), bootstrap, limits)
    if audited["result"] != profile["audit"]:
        raise Refusal("environment_changed", "source/environment/compiled identities differ from retained profile")
    _load_profile(store, profile_ref)  # refuse concurrent object mutations
    if cpu._python_stable(cpu.interpreter_identity(worker.PYTHON, limits)) != profile["runtime"]["python"]:
        raise Refusal("environment_changed", "interpreter changed during verify")
    receipt = run.write("verify-receipt.json", {"profile_id": profile["profile_id"], "asserts_current": False,
        "launch": launch, "workers": run.receipts, "interpretation": dict(worker.INTERPRETATION)})
    return {"verdict": "gpu_source_environment_verified", "profile_id": profile["profile_id"],
        "run_dir": str(run.dir), "receipt": _file(receipt, limits.max_evidence_bytes)}


def resource_audit(*, limits):
    """Read-only observations, never reservations. A failed direct query refuses."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        raise Refusal("resource_unavailable", "nvidia-smi not found")
    binary = os.path.realpath(binary)
    before = _file(binary)
    queries = ["--query-gpu=uuid,pci.bus_id,driver_version,memory.total,memory.used,utilization.gpu",
        "--query-compute-apps=pid,gpu_uuid,used_memory"]
    receipts, outputs = [], []
    for query in queries:
        argv = [binary, query, "--format=csv,noheader,nounits"]
        result = cpu.run_bounded(argv, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"}, timeout_s=min(15, limits.worker_timeout_s),
            max_output_bytes=min(1 << 20, limits.worker_max_output_bytes))
        receipts.append({"argv": argv, "pid": result.pid, "exit_code": result.returncode, "reaped": True,
            "stdout": result.stdout.decode("utf-8", "strict"), "stderr": result.stderr.decode("utf-8", "replace")})
        if result.returncode != 0:
            raise Refusal("resource_unavailable", "direct nvidia-smi query failed", receipts=receipts)
        outputs.append(result.stdout.decode("utf-8", "strict"))
    if _file(binary) != before:
        raise Refusal("resource_changed", "resource-query executable changed")
    gpus, processes = [], []
    for line in outputs[0].splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            raise Refusal("resource_invalid", "GPU query columns differ")
        try:
            gpus.append({"uuid": worker.physical_uuid(parts[0]), "pci_bus_id": worker.pci_id(parts[1]),
                "driver_version": parts[2], "memory_total_mib": int(parts[3]), "memory_used_mib": int(parts[4]),
                "utilization_percent": int(parts[5])})
        except (ValueError, worker.WorkerRefusal) as exc:
            raise Refusal("resource_invalid", "GPU observation missing/unsupported") from exc
    for line in outputs[1].splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            raise Refusal("resource_invalid", "process query columns differ")
        try:
            processes.append({"pid": int(parts[0]), "uuid": worker.physical_uuid(parts[1]), "memory_mib": parts[2]})
        except (ValueError, worker.WorkerRefusal) as exc:
            raise Refusal("resource_invalid", "process observation unsupported") from exc
    return {"schema": "masy.gpu_resource_observation.v1", "observed_unix_s": time.time(),
        "gpus": gpus, "processes": processes, "command_receipts": {"binary": before, "queries": receipts}}


def probe_profile(store, profile_ref, *, launch, authorization):
    """Gated identity only; callers must provide the separately accepted scope."""
    verified = verify_profile(store, profile_ref, launch=launch)
    profile, limits, bootstrap = _load_profile(store, profile_ref)
    # verify used its own run; probe gets a distinct never-overwritten run.
    run = _Run(store, launch["run_id"] + "-probe")
    resource = resource_audit(limits=limits)
    gate = {"authorization": authorization, "resource": resource,
        "resource_sha256": hashlib.sha256(json.dumps(resource, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode()).hexdigest()}
    try:
        worker.check_probe_gate(gate, profile["profile_id"], profile["device"])
    except worker.WorkerRefusal as exc:
        run.write("refused-gate.json", {"gate": gate, "code": exc.code, "detail": exc.detail})
        raise Refusal(exc.code, exc.detail) from exc
    python = cpu.interpreter_identity(worker.PYTHON, limits)
    body = _profile_body(profile, run, python)
    body.update(profile_id=profile["profile_id"], probe_gate=gate)
    result = _call(run, "probe", body, bootstrap, limits)
    for key in ("native", "declared_kernels", "environment"):
        if result["result"][key] != profile["audit"][key]:
            raise Refusal("environment_changed", f"identity probe changed bound {key}")
    observed_compiled = {r["path"]: r for r in result["result"]["compiled"]["files"]}
    for previous in profile["audit"]["compiled"]["files"]:
        if observed_compiled.get(previous["path"]) != previous:
            raise Refusal("environment_changed", "bound source-stage binary changed during probe")
    total = 0
    for record in observed_compiled.values():
        actual = _file(record["path"], limits.max_compiled_bytes - total)
        total += actual["size"]
        if actual["sha256"] != record["sha256"] or actual["size"] != record["size"]:
            raise Refusal("environment_changed", "loaded probe binary changed after child exit")
    after = resource_audit(limits=limits)
    after_gate = {"authorization": authorization, "resource": after,
        "resource_sha256": hashlib.sha256(json.dumps(after, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode()).hexdigest()}
    try:
        worker.check_probe_gate(after_gate, profile["profile_id"], profile["device"])
    except worker.WorkerRefusal as exc:
        run.write("refused-post-resource.json", {"resource": after, "code": exc.code, "detail": exc.detail})
        raise Refusal(exc.code, exc.detail) from exc
    post = _call(run, "audit", _profile_body(profile, run, python), bootstrap, limits)
    if post["result"] != profile["audit"]:
        raise Refusal("environment_changed", "source/environment changed after identity probe")
    _load_profile(store, profile_ref)
    if cpu._python_stable(cpu.interpreter_identity(worker.PYTHON, limits)) != profile["runtime"]["python"]:
        raise Refusal("environment_changed", "interpreter changed during identity probe")
    receipt = run.write("probe-receipt.json", {"authorization": authorization, "resource_before": resource,
        "resource_after": after, "source_verify": verified, "workers": run.receipts})
    probe = {"schema": PROBE_SCHEMA, "probe_id": "", "profile_id": profile["profile_id"],
        "device": profile["device"], "result": result["result"], "interpretation": result["interpretation"],
        "receipt": _file(receipt, limits.max_evidence_bytes)}
    probe, path = _publish(store, "probes", probe, "probe_id", run.run_id)
    return {"verdict": "gpu_identity_observed", "profile_id": profile["profile_id"], "probe_id": probe["probe_id"],
        "probe_path": str(path), "interpretation": probe["interpretation"]}
