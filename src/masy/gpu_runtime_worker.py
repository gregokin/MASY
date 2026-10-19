"""Sealed GPU-profile worker: source audit or explicitly gated identity queries.

No transport, numerical probe, kernel compilation, or automatic device fallback.
Importing this module imports only the standard library and the unchanged CPU
worker's source/provenance helpers. Source audit blocks accelerator imports.
"""
from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import json
import math
import os
import re
import stat
import sys
import time
import uuid
from pathlib import Path

from . import runtime_worker as native

REQUEST_SCHEMA = "masy.gpu_runtime_request.v1"
RESULT_SCHEMA = "masy.gpu_runtime_result.v1"
PYTHON = "/home/okin/MAPLE/.venv-cupy/bin/python"
VENV = "/home/okin/MAPLE/.venv-cupy"
MAX_REQUEST = 4 << 20
MAX_MAPS = 8 << 20
MAX_METADATA = 64 << 20
MAX_RECORD_ENTRIES = 300_000
INTERPRETATION = {
    "device_identity_observed": False, "identity_probe_passed": False,
    "runtime_smoke": False, "kernels_compiled": False,
    "transport_ready": False, "gpu_qualified": False,
}
_CACHE_PATHS = {"TMPDIR": "tmp", "XDG_CACHE_HOME": "cache/xdg",
    "NUMBA_CACHE_DIR": "cache/numba", "MPLCONFIGDIR": "cache/matplotlib",
    "CUPY_CACHE_DIR": "cache/cupy", "CUDA_CACHE_PATH": "cache/cuda",
    "CUPY_CUDA_LIB_PATH": "cache/cupy-cuda-lib"}
_STABLE_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "NUMBA_DISABLE_CUDA": "1",
    "JAX_PLATFORMS": "cpu", "CUPY_CACHE_IN_MEMORY": "0", "CUPY_NVRTC_USE_PCH": "0"}
_COMMON = {"schema", "operation", "python", "run", "sys_path_prefix", "packages", "limits", "device"}
_EXTRA = {"export": {"export", "tool_module_sha256"}, "materialize": {"bundle"},
    "audit": {"bundle", "distributions"},
    "probe": {"bundle", "distributions", "profile_id", "probe_gate"}}
WorkerRefusal = native.WorkerRefusal
MODULE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# Literal is read from the sealed source object by the parent, never a live
# installed module. JSON duplicate/non-finite rejection happens before imports.
BOOTSTRAP = r'''
import json, os, sys
def refuse(detail):
    print(json.dumps({"schema":"masy.gpu_runtime_result.v1","operation":"bootstrap","status":"refused","code":"bootstrap_refused","detail":detail}), flush=True)
    raise SystemExit(2)
if any(n.partition(".")[0] in ("masy","maple","maple_syrup","cupy","cupyx","cupy_backends","cuda") for n in sys.modules):
    refuse("native or accelerator package preloaded")
def pairs(items):
    result = {}
    for key, value in items:
        if key in result: refuse("duplicate JSON key")
        result[key] = value
    return result
with open(sys.argv[1], "rb") as source:
    data = source.read((4 << 20) + 1)
if len(data) > 4 << 20: refuse("request exceeds bound")
request = json.loads(data, object_pairs_hook=pairs, parse_constant=lambda x: refuse("nonfinite JSON"))
prefix = request.get("sys_path_prefix") if isinstance(request, dict) else None
if not isinstance(prefix,list) or not prefix or not all(isinstance(p,str) and os.path.isabs(p) and os.path.realpath(p)==p and os.path.isdir(p) for p in prefix):
    refuse("invalid sealed prefix")
base = list(sys.path)
sys.path[:0] = prefix
from masy.gpu_runtime_worker import main
raise SystemExit(main(request, base))
'''


def closed(value, keys, where):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise WorkerRefusal("gpu_schema_invalid", f"{where} has an invalid closed key set")
    return value


def physical_uuid(value):
    if not isinstance(value, str) or not re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value):
        raise WorkerRefusal("device_identity_invalid", "expected a canonical full physical GPU UUID")
    return "GPU-" + str(uuid.UUID(value[4:]))


def pci_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", value):
        raise WorkerRefusal("device_identity_invalid", "missing or malformed PCI identity")
    domain, bus, rest = value.lower().split(":")
    device, function = rest.split(".")
    if int(device, 16) > 31:
        raise WorkerRefusal("device_identity_invalid", "PCI device exceeds31")
    return f"{int(domain, 16):08x}:{int(bus, 16):02x}:{int(device, 16):02x}.{function}"


def check_device(value):
    closed(value, {"uuid", "pci_bus_id", "visible_ordinal"}, "device")
    if type(value["visible_ordinal"]) is not int or value["visible_ordinal"] != 0:
        raise WorkerRefusal("device_identity_invalid", "v1 supports exactly visible ordinal0")
    return {"uuid": physical_uuid(value["uuid"]), "pci_bus_id": pci_id(value["pci_bus_id"]), "visible_ordinal": 0}


def worker_env(run_dir, gpu_uuid):
    return {**_STABLE_ENV, "CUDA_VISIBLE_DEVICES": physical_uuid(gpu_uuid),
        **{key: str(Path(run_dir) / relative) for key, relative in _CACHE_PATHS.items()}}


class AcceleratorBlocker(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in {"cupy", "cupyx", "cupy_backends", "cuda", "pynvml"} or \
                fullname == "numba.cuda" or fullname.startswith("numba.cuda."):
            self.attempts.append(fullname)
            raise WorkerRefusal("accelerator_import_refused", f"no-context stage tried to import {fullname}")


def check_request(request):
    if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA or request.get("operation") not in _EXTRA:
        raise WorkerRefusal("gpu_schema_invalid", "unsupported worker schema/operation")
    operation = request["operation"]
    closed(request, _COMMON | _EXTRA[operation], "request")
    closed(request["limits"], {"max_files", "max_bytes", "max_compiled_files", "max_compiled_bytes"}, "limits")
    if not all(type(v) is int and v > 0 for v in request["limits"].values()):
        raise WorkerRefusal("gpu_schema_invalid", "limits must be positive integers")
    request["device"] = check_device(request["device"])
    return operation


def check_process(request, base_sys_path):
    closed(request["python"], {"spelling", "realpath", "venv_dir"}, "python")
    closed(request["run"], {"run_dir", "pycache_prefix", "env"}, "run")
    run = request["run"]
    expected_env = worker_env(run["run_dir"], request["device"]["uuid"])
    if run["env"] != expected_env:
        raise WorkerRefusal("worker_process_mismatch", "request environment differs from the profile policy")
    extra = set(os.environ) - set(expected_env)
    if extra - {"LC_CTYPE"} or any(os.environ.get(k) != v for k, v in expected_env.items()):
        raise WorkerRefusal("worker_process_mismatch", "actual environment differs from exact policy")
    if "LC_CTYPE" in os.environ and os.environ["LC_CTYPE"] not in ("C.UTF-8", "C.utf8"):
        raise WorkerRefusal("worker_process_mismatch", "unexpected interpreter locale adjustment")
    py = request["python"]
    if py["spelling"] != PYTHON or py["venv_dir"] != VENV or sys.executable != PYTHON or \
            os.path.realpath(sys.executable) != py["realpath"] or os.path.realpath(sys.prefix) != VENV:
        raise WorkerRefusal("worker_process_mismatch", "interpreter/venv differs from explicit .venv-cupy profile")
    if any(getattr(sys.flags, flag) != 1 for flag in ("isolated", "dont_write_bytecode", "ignore_environment", "no_user_site")) or \
            sys.flags.safe_path is not True:
        raise WorkerRefusal("worker_process_mismatch", "isolated/source-only bytecode flags missing")
    expected_pycache = str(Path(run["run_dir"]) / "cache" / "pycache")
    if run["pycache_prefix"] != expected_pycache or sys.pycache_prefix != expected_pycache or os.path.lexists(expected_pycache):
        raise WorkerRefusal("worker_process_mismatch", "pycache prefix differs or already exists")
    prefix = request["sys_path_prefix"]
    if sys.path[:len(prefix)] != prefix or len(prefix) != len(set(prefix)):
        raise WorkerRefusal("worker_process_mismatch", "sealed prefix differs or repeats")
    for path in [run["run_dir"], *prefix, *[expected_env[k] for k in _CACHE_PATHS]]:
        if not isinstance(path, str) or os.path.realpath(path) != path or not os.path.isdir(path):
            raise WorkerRefusal("worker_process_mismatch", f"noncanonical/missing directory {path!r}")
    # The environment and argv contain no arbitrary action supplied by a caller.
    ticks = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
    return {"pid": os.getpid(), "start_ticks": ticks, "cwd": os.getcwd(),
        "executable": sys.executable, "executable_realpath": os.path.realpath(sys.executable),
        "prefix": sys.prefix, "flags": {k: getattr(sys.flags, k) for k in
            ("isolated", "dont_write_bytecode", "ignore_environment", "no_user_site", "safe_path")},
        "environment": dict(os.environ), "pycache_prefix": sys.pycache_prefix,
        "sys_path_prefix": list(prefix), "base_sys_path": list(base_sys_path), "limits": dict(request["limits"])}


def mapped_files(maps_path="/proc/self/maps"):
    try:
        with open(maps_path, "rb") as handle:
            data = handle.read(MAX_MAPS + 1)
    except OSError as exc:
        raise WorkerRefusal("compiled_maps_unavailable", str(exc)) from exc
    if len(data) > MAX_MAPS:
        raise WorkerRefusal("compiled_identity_bound_exceeded", "maps exceeds bound")
    found = {}
    for line in data.decode("utf-8", "surrogateescape").splitlines():
        fields = line.split(None, 5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        path = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[5])
        executable = "x" in fields[1] or ".so" in os.path.basename(path)
        if path.endswith(" (deleted)"):
            if executable:
                raise WorkerRefusal("compiled_mapping_deleted", path)
            found[path] = {"path": path, "kind": "deleted_nonbinary_mapping"}
            continue
        real = os.path.realpath(path)
        try:
            info = os.stat(real)
        except OSError as exc:
            if executable:
                raise WorkerRefusal("compiled_mapping_unreadable", path) from exc
            found[real] = {"path": real, "kind": "unreadable_nonbinary_mapping"}
            continue
        kind = "binary" if executable and stat.S_ISREG(info.st_mode) else \
            "regular_nonbinary_mapping" if stat.S_ISREG(info.st_mode) else "special_mapping"
        previous = found.get(real)
        if previous is None or kind == "binary":
            found[real] = {"path": real, "kind": kind, "mode": stat.S_IFMT(info.st_mode)}
    return [found[key] for key in sorted(found)]


def environment_metadata(*, path=None):
    """Metadata/RECORD identities, not a claim that every declared file loaded."""
    kwargs = {} if path is None else {"path": path}
    records, binary_records, total, entries = [], {}, 0, 0
    for dist in importlib.metadata.distributions(**kwargs):
        if len(records) >= 4000:
            raise WorkerRefusal("environment_metadata_bound", "more than4000 distributions")
        files = dist.files
        if files is None:
            files = []
        metadata_paths = []
        declared = []
        for item in files:
            entries += 1
            if entries > MAX_RECORD_ENTRIES:
                raise WorkerRefusal("environment_metadata_bound", "RECORD entry bound exceeded")
            rel = str(item)
            actual = os.path.realpath(dist.locate_file(item))
            if ".so" in os.path.basename(rel):
                declared.append({"path": actual, "record_hash": str(item.hash) if item.hash else None,
                    "record_size": item.size})
                if item.hash is not None:
                    binary_records.setdefault(actual, []).append({"distribution": dist.metadata.get("Name"),
                        "mode": item.hash.mode, "value": item.hash.value, "size": item.size})
            if ".dist-info/" in rel and rel.rsplit("/", 1)[-1] in \
                    {"METADATA", "WHEEL", "RECORD", "direct_url.json", "INSTALLER"}:
                digest, size = native._sha256_file(actual, MAX_METADATA - total)
                total += size
                metadata_paths.append({"path": actual, "sha256": digest, "size": size})
        records.append({"name": dist.metadata.get("Name"), "version": dist.version,
            "location": os.path.realpath(str(dist.locate_file(""))),
            "metadata_files": sorted(metadata_paths, key=lambda x: x["path"]),
            "declared_native_files": sorted(declared, key=lambda x: x["path"])})
    return sorted(records, key=lambda x: (x["name"] or "", x["location"])), binary_records


def compiled_inventory(limits, binary_records, maps_path="/proc/self/maps"):
    mappings = mapped_files(maps_path)
    paths = {record["path"] for record in mappings if record["kind"] == "binary"}
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if isinstance(path, str) and path.endswith(suffixes):
            paths.add(os.path.realpath(path))
    if len(paths) > limits["max_compiled_files"]:
        raise WorkerRefusal("compiled_identity_bound_exceeded", "compiled file count exceeded")
    compiled, total = [], 0
    for path in sorted(paths):
        digest, size = native._sha256_file(path, limits["max_compiled_bytes"] - total)
        total += size
        owners = binary_records.get(path, [])
        for owner in owners:
            actual = base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=").decode()
            if owner["mode"] != "sha256" or owner["value"] != actual or owner["size"] != size:
                raise WorkerRefusal("compiled_record_mismatch", f"{path} differs from installed RECORD")
        if path.startswith(VENV + "/lib/") and ".so" in os.path.basename(path) and not owners:
            raise WorkerRefusal("compiled_record_missing", f"no RECORD hash for environment binary {path}")
        compiled.append({"path": path, "sha256": digest, "size": size, "record_owners": owners})
    return {"files": compiled, "mapped_files": mappings, "total_bytes": total,
        "scope": "actual loaded regular binaries; special mappings have metadata only; lazy/unmapped binaries unobserved"}


def declared_kernels():
    routing = importlib.import_module("maple_syrup.routing_cuda")
    hydro = importlib.import_module("maple_syrup.hydrology_cuda")
    newton = importlib.import_module("maple_syrup.routing_newton_cuda")
    result = {"scope": "source declarations only; no kernel compiled or executed", "routing": {}, "hydrology": {}}
    for label, module, names in (("routing", routing, ("_KERNEL_SOURCE", "_BLOCK_SOURCE")),
            ("hydrology", hydro, ("_SOURCE", "_SOURCE_NEWTON"))):
        for name in names:
            value = getattr(module, name)
            if not isinstance(value, str):
                raise WorkerRefusal("native_kernel_schema_changed", f"{label}.{name} is not source text")
            result[label][name + "_sha256"] = hashlib.sha256(value.encode()).hexdigest()
    result["compile_options"] = list(routing.COMPILE_OPTIONS)
    result["backend"] = "nvrtc"
    result["newton_device_sha256"] = hashlib.sha256(newton.newton_device_source().encode()).hexdigest()
    for module_name, function in (("maple.core.backend.cell_reductions", "cell_reduction_provenance"),
            ("maple.core.backend.prefix_cell_reductions", "prefix_reduction_provenance")):
        result[module_name] = getattr(importlib.import_module(module_name), function)()
    return result


def native_audit(request):
    from masy.contracts import run_native_audit

    packages, spec = request["packages"], request["bundle"]
    bundle, tool_sha = native._bundle_tool(spec["tool_module_sha256"], packages["maple_syrup"]["dir"])
    manifest = native._upstream_call("upstream_bundle_error", bundle.verify_bundle, spec["dir"])
    if manifest["bundle_id"] != spec["bundle_id"]:
        raise WorkerRefusal("bundle_identity_mismatch", "verified bundle ID differs")
    runtime = native._upstream_call("native_check_runtime_failed", bundle.check_runtime, spec["target"])
    dependency = importlib.import_module("maple_syrup.dependency")
    audit, findings = run_native_audit(dependency, os.path.join(spec["target"], "source"))
    if any(f["severity"] in ("refusal", "incompatible") for f in findings):
        raise WorkerRefusal("native_audit_failed", "native API audit did not pass", findings=findings)
    registry = native.registry_probe()
    native._origins(packages)
    digests = native._package_digests(packages, request["limits"])
    distributions, shadowed = native.selected_distributions(request["distributions"])
    return {"bundle_id": manifest["bundle_id"], "tool_module_sha256": tool_sha,
        "check_runtime": runtime, "native_audit": audit, "native_findings": findings,
        "registry_probe": registry, "package_digests": digests,
        "selected_distributions": distributions, "shadowed_distributions": shadowed}


def check_probe_gate(gate, profile_id, device, *, now=None):
    closed(gate, {"authorization", "resource", "resource_sha256"}, "probe_gate")
    auth, resource = gate["authorization"], gate["resource"]
    closed(auth, {"schema", "authorized", "operation", "profile_id", "uuid", "expires_unix_s",
        "max_resource_age_s", "max_memory_used_mib", "max_utilization_percent"}, "authorization")
    now = time.time() if now is None else now
    numbers = (auth["expires_unix_s"], auth["max_resource_age_s"])
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in numbers) or \
            any(type(auth[k]) is not int or auth[k] < 0 for k in ("max_memory_used_mib", "max_utilization_percent")):
        raise WorkerRefusal("probe_not_authorized", "invalid authorization bounds")
    if auth["schema"] != "masy.gpu_identity_authorization.v1" or auth["authorized"] is not True or \
            auth["operation"] != "identity_probe" or auth["profile_id"] != profile_id or \
            auth["uuid"] != device["uuid"] or now > auth["expires_unix_s"]:
        raise WorkerRefusal("probe_not_authorized", "scope/profile/UUID/expiry mismatch")
    encoded = json.dumps(resource, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != gate["resource_sha256"]:
        raise WorkerRefusal("probe_not_authorized", "resource receipt hash differs")
    closed(resource, {"schema", "observed_unix_s", "gpus", "processes", "command_receipts"}, "resource")
    if resource["schema"] != "masy.gpu_resource_observation.v1":
        raise WorkerRefusal("resource_invalid", "unsupported resource evidence schema")
    observed = resource["observed_unix_s"]
    if type(observed) not in (int, float) or not math.isfinite(observed) or not 0 <= now - observed <= auth["max_resource_age_s"]:
        raise WorkerRefusal("resource_stale", "allocation evidence not fresh")
    rows = resource["gpus"]
    if not isinstance(rows, list) or not isinstance(resource["processes"], list):
        raise WorkerRefusal("resource_invalid", "GPU/process evidence malformed")
    for process in resource["processes"]:
        closed(process, {"uuid", "pid", "memory_mib"}, "resource process")
        physical_uuid(process["uuid"])
        if type(process["pid"]) is not int or process["pid"] <= 0 or \
                not isinstance(process["memory_mib"], str) or not process["memory_mib"].strip():
            raise WorkerRefusal("resource_invalid", "malformed process identity or memory evidence")
    matches = [r for r in rows if isinstance(r, dict) and r.get("uuid") == device["uuid"]]
    same_pci = [r for r in rows if isinstance(r, dict) and r.get("pci_bus_id") == device["pci_bus_id"]]
    if len(matches) != 1 or same_pci != matches:
        raise WorkerRefusal("resource_invalid", "UUID/PCI physical mapping missing or ambiguous")
    row = matches[0]
    for key in ("memory_used_mib", "memory_total_mib", "utilization_percent"):
        if type(row.get(key)) is not int or row[key] < 0:
            raise WorkerRefusal("resource_invalid", "unreadable memory/utilization observation")
    if row["memory_used_mib"] > auth["max_memory_used_mib"] or \
            row["utilization_percent"] > auth["max_utilization_percent"] or \
            any(p["uuid"] == device["uuid"] for p in resource["processes"]):
        raise WorkerRefusal("resource_busy", "selected device exceeds authorized idle/allocation scope")
    return row


def check_observed_device(device, properties, pci):
    raw = properties.get("uuid")
    if not isinstance(raw, bytes) or len(raw) != 16:
        raise WorkerRefusal("device_identity_unverified", "native UUID must be exactly16 bytes")
    observed_uuid = "GPU-" + str(uuid.UUID(bytes=raw))
    observed_pci = pci_id(pci)
    if observed_uuid != device["uuid"] or observed_pci != device["pci_bus_id"]:
        raise WorkerRefusal("device_identity_mismatch", "actual CUDA UUID/PCI differs from physical audit")
    # Independently supplied PCI properties may never contradict the API string.
    keys = ("pciDomainID", "pciBusID", "pciDeviceID")
    if any(k in properties for k in keys):
        if any(type(properties.get(k)) is not int for k in keys):
            raise WorkerRefusal("device_identity_unverified", "malformed native PCI properties")
        props_pci = f"{properties[keys[0]]:08x}:{properties[keys[1]]:02x}:{properties[keys[2]]:02x}.0"
        if pci_id(props_pci) != observed_pci:
            raise WorkerRefusal("device_identity_mismatch", "native PCI properties contradict PCI API")
    return {"uuid": observed_uuid, "pci_bus_id": observed_pci, "visible_ordinal": 0}


def identity_probe(request):
    check_probe_gate(request["probe_gate"], request["profile_id"], request["device"])
    import cupy as cp
    from maple.core.backend.selection import resolve_backend

    if cp.cuda.runtime.getDeviceCount() != 1:
        raise WorkerRefusal("device_identity_unverified", "expected exactly one visible device")
    observed = check_observed_device(request["device"], cp.cuda.runtime.getDeviceProperties(0),
        cp.cuda.runtime.deviceGetPCIBusId(0))
    backend = resolve_backend("cupy", device_id=0)
    with backend.device_scope():
        if cp.cuda.runtime.getDevice() != 0:
            raise WorkerRefusal("device_identity_mismatch", "native device scope is not ordinal0")
        versions = {"cupy": cp.__version__, "driver": int(cp.cuda.runtime.driverGetVersion()),
            "runtime": int(cp.cuda.runtime.runtimeGetVersion()), "local_runtime": int(cp.cuda.get_local_runtime_version()),
            "nvrtc": list(cp.cuda.nvrtc.getVersion())}
        if any(value is None for value in backend.fingerprint.values()):
            raise WorkerRefusal("device_provenance_incomplete", "native backend fingerprint is incomplete")
        routing = importlib.import_module("maple_syrup.routing_cuda").kernel_provenance()
        hydro = importlib.import_module("maple_syrup.hydrology_cuda").kernel_provenance()
        if any(value.get("provenance_incomplete") or value.get("nvrtc") is None for value in (routing, hydro)):
            raise WorkerRefusal("device_provenance_incomplete", "native CUDA provenance is incomplete")
    return {"device": observed, "versions": versions, "native_backend": backend.fingerprint,
        "native_kernel_declarations": {"routing": routing, "hydrology": hydro},
        "deterministic_scatter": backend.deterministic_scatter}


def main(request, base_sys_path):
    operation, blocker, process = "bootstrap", None, None
    try:
        operation = check_request(request)
        process = check_process(request, base_sys_path)
        if operation != "probe":
            blocker = AcceleratorBlocker()
            sys.meta_path.insert(0, blocker)
        else:
            check_probe_gate(request["probe_gate"], request["profile_id"], request["device"])
        packages = request["packages"]
        if operation == "export":
            tool, tool_sha = native._bundle_tool(request["tool_module_sha256"], packages["maple_syrup"]["dir"])
            spec = request["export"]
            result = {"manifest": native._upstream_call("upstream_bundle_error", tool.export_bundle,
                spec["project"], spec["distribution"], spec["origin"], spec["output"]), "tool_sha256": tool_sha}
            native._origins(packages, forbid=("maple",))
        elif operation == "materialize":
            spec = request["bundle"]
            tool, _sha = native._bundle_tool(spec["tool_module_sha256"], packages["maple_syrup"]["dir"])
            target = native._upstream_call("upstream_bundle_error", tool.materialize_bundle, spec["dir"], spec["cache"])
            verified = native._upstream_call("upstream_bundle_error", tool.verify_materialized, target)
            if str(target) != spec["target"] or verified["bundle_id"] != spec["bundle_id"]:
                raise WorkerRefusal("bundle_identity_mismatch", "materialized native identity differs")
            result = {"target": str(target), "bundle_id": verified["bundle_id"]}
            native._origins(packages, forbid=("maple",))
        else:
            result = {"native": native_audit(request), "declared_kernels": declared_kernels()}
            if operation == "probe":
                result["probe"] = identity_probe(request)
            distributions, record_index = environment_metadata()
            if not any(d["name"] == "cupy-cuda12x" for d in distributions):
                raise WorkerRefusal("environment_incompatible", "cupy-cuda12x distribution metadata absent")
            result["environment"] = distributions
            result["compiled"] = compiled_inventory(request["limits"], record_index)
            native._origins(packages)
        if blocker is not None and (blocker.attempts or any(n.partition(".")[0] in
                {"cupy", "cupyx", "cupy_backends", "cuda", "pynvml"} for n in sys.modules)):
            raise WorkerRefusal("accelerator_import_refused", "no-context accelerator import attempt recorded",
                attempts=blocker.attempts)
        interpretation = dict(INTERPRETATION)
        if operation == "probe":
            interpretation.update(device_identity_observed=True, identity_probe_passed=True)
        output = {"schema": RESULT_SCHEMA, "operation": operation, "status": "ok", "result": result,
            "process": process, "interpretation": interpretation}
    except WorkerRefusal as exc:
        output = {"schema": RESULT_SCHEMA, "operation": operation, "status": "refused", "code": exc.code,
            "detail": exc.detail, "evidence": exc.evidence, "process": process}
    finally:
        if blocker is not None:
            sys.meta_path.remove(blocker)
    print(json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False), flush=True)
    return 0 if output["status"] == "ok" else 2
