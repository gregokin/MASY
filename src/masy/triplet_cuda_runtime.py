"""Explicit opt-in CUDA source profile for the four-object triplet lock.

``masy.daily_wind_water_growth.cuda.v1`` locks the same immutable frontier as
the CPU profile (the four selected committed sources, the genuine MAPLE
bundle, the LEAF ``benchmarks.spatial_mapping`` helper and the labelled project
distribution metadata) under the existing ``.venv-cupy`` interpreter, backend
``cuda`` and the device-runtime native source profile. It reuses the CPU lock's
bounded Git archive/blob reads, scoped dirty-work exclusion, copy, exact-tree
verification, cold import, metadata and helper guards through private policy
parameters. The CPU constants, descriptors, locks and bootstrap are unchanged
and still refuse CuPy; nothing here patches them.

Source provisioning grants no backend acceptance. Physical device availability,
permission, allocation and finite time/memory execution control remain the
Root parent's responsibility: ``bootstrap_spec`` constructs argv/env and
launches nothing, the worker selects or switches no device, and nothing here
reads or sends model data. Combined persisted restart stays false. One actual
backend per interpreter: CuPy may load only after source verification, only
from the selected interpreter's own site-packages, and is never swapped in
``sys.modules``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
from importlib.machinery import EXTENSION_SUFFIXES
import json
import os
from pathlib import Path
import re
import stat
import sys
import sysconfig
from types import ModuleType

from masy import triplet_lock as tl
from masy import triplet_runtime as tr

QUALIFICATION_SCHEMA = "masy.triplet_cuda_qualification.v1"
LOCK_SCHEMA = "masy.triplet_cuda_lock.v1"
PROFILE = "masy.daily_wind_water_growth.cuda.v1"
PYTHON = "/home/okin/MAPLE/.venv-cupy/bin/python"
BACKEND = "cuda"
PROVIDER = "cupy"
#: The owning SYRUP registry decides which literal names carry device-runtime
#: capability; this MASY profile version binds exactly this one and still
#: requires the sealed record's own ``device_runtime_capability`` at runtime.
NATIVE_PROFILE = "native_maple_1dca_case_latitude_cuda_v1"
DEVICE_FAMILIES = ("cupy", "cupyx", "cupy_backends")
GENERATOR_FILES = ("src/masy/triplet_lock.py", "src/masy/triplet_runtime.py", "src/masy/triplet_cuda_runtime.py")
INTERPRETATION = (("code_only", True), ("transport_qualified_by_lock", False), ("combined_restart", False),
                  ("backend", BACKEND), ("immutable_upgrade", False), ("device_qualified_by_lock", False))
ENVIRONMENT = (("NUMBA_DISABLE_CUDA", "1"), ("JAX_PLATFORMS", "cpu"), ("GIT_OPTIONAL_LOCKS", "0"))
CACHES = (*tr._CPU_CACHES, ("CUPY_CACHE_DIR", "cache/cupy"), ("CUDA_CACHE_PATH", "cache/cuda"))
UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
REFUSED_DETAIL = "device provider must not load before source verification and the qualified GPU entry point"
MAX_REPORT_BYTES = 1 << 16
MAX_REPORTS = 64
MAX_DEVICE_MODULES = 4096
MAX_EXTENSION_BYTES = 128 << 20  # metadata-only binary census; separate from source-body caps

# Same stdlib-only head as the CPU profile (lock SHA, sealed MASY bytes, fresh
# pycache, stdlib-first path); this tail refuses a non-CUDA lock or a process
# whose CUDA_VISIBLE_DEVICES is not the locked physical UUID before importing.
BOOTSTRAP = tr._BOOTSTRAP_HEAD + (
    "if lock.get('schema')!=%s or lock.get('profile')!=%s:\n"
    "    raise RuntimeError('explicit CUDA triplet lock required')\n"
    "device=lock.get('qualification',{}).get('device',{}).get('physical_uuid')\n"
    "if type(device) is not str or os.environ.get('CUDA_VISIBLE_DEVICES')!=device:\n"
    "    raise RuntimeError('selected physical device differs from the lock')\n"
    "from masy.triplet_cuda_runtime import worker_main\n"
    "sys.exit(worker_main(sys.argv[1:]))\n") % (json.dumps(LOCK_SCHEMA), json.dumps(PROFILE))


def _device(value):
    tl._keys(value, ("physical_uuid", "logical_ordinal"), "device")
    tl._need(type(value["physical_uuid"]) is str and UUID.fullmatch(value["physical_uuid"]), "device",
             "canonical lowercase physical GPU UUID required")
    tl._need(type(value["logical_ordinal"]) is int and value["logical_ordinal"] == 0, "device",
             "v1 selects exactly logical device 0 of the visible set")
    return {"physical_uuid": value["physical_uuid"], "logical_ordinal": 0}


def _extension(q):
    """CUDA-only descriptor fields, checked after every shared CPU-equivalent check."""
    backend = tl._keys(q["backend"], ("name", "provider"), "backend")
    tl._need(backend["name"] == BACKEND and backend["provider"] == PROVIDER, "backend", "explicit cuda/cupy backend required")
    _device(q["device"])
    tl._need(q["native_profile"] == NATIVE_PROFILE, "unqualified_profile", "explicit device-runtime native source profile required")


_POLICY = tl._Policy("CUDA", QUALIFICATION_SCHEMA, LOCK_SCHEMA, PROFILE, PYTHON, INTERPRETATION, GENERATOR_FILES,
                     ("backend", "device"), _extension)


def device_facts(q):
    """Identity labels from the qualified descriptor; not an observation of hardware."""
    device = _device(q["device"])
    return {"backend": BACKEND, "provider": PROVIDER, "physical_uuid": device["physical_uuid"], "logical_ordinal": 0,
            "visible_devices": device["physical_uuid"], "interpreter": q["interpreter"]["path"],
            "native_profile": q["native_profile"], "physical_device_verified": False}


def _environment(lock):
    return {**dict(ENVIRONMENT), "CUDA_VISIBLE_DEVICES": _device(lock["qualification"]["device"])["physical_uuid"]}


def draft_qualification(sources, *, interpreter, maple_bundle, device, entrypoints=(), limits=None):
    """Construct an UNQUALIFIED CUDA descriptor for independent Root review.

    Observed hashes never become authority: ``prepare`` refuses this status
    until Root independently qualifies the frontier, freezes the final SYRUP/
    LEAF/package hashes and supplies the descriptor's external SHA.
    """
    selected = _device(device)
    draft = tl.draft_qualification(sources, interpreter=interpreter, maple_bundle=maple_bundle,
                                   native_profile=NATIVE_PROFILE, entrypoints=entrypoints, limits=limits)
    draft["qualification"] = {**draft["qualification"], "schema": QUALIFICATION_SCHEMA, "profile": PROFILE,
                              "backend": {"name": BACKEND, "provider": PROVIDER}, "device": selected}
    return draft


def load_qualification(path, *, expected_sha256):
    return tl._load_qualification(path, expected_sha256, _POLICY)


def prepare(qualification_path, *, expected_sha256, destination):
    """No-clobber snapshot from qualified committed objects; no native, device or runtime run."""
    return tl._prepare(qualification_path, expected_sha256, destination, _POLICY)


def load(path, *, expected_sha256):
    return tl._load(path, expected_sha256, _POLICY)


def verify(path, *, expected_sha256, current_refs=True):
    """Verify sealed code/bundle/interpreter; default also refuses advanced selected refs."""
    return tl._verify(path, expected_sha256, current_refs, _POLICY)


def check_import_sources(lock, modules):
    """Pre-entry census: exact project/helper origins and no device provider loaded yet."""
    return tr._check_import_sources(lock, modules, DEVICE_FAMILIES, REFUSED_DETAIL)


def _origin(path, cap):
    tl._need(type(path) is str and os.path.isabs(path) and os.path.normpath(path) == path, "backend",
             "canonical device provider origin required")
    p = Path(path)
    for q in (p, *p.parents):
        tl._need(not stat.S_ISLNK(q.lstat().st_mode), "backend", f"symlinked device provider path {q}")
    st = p.lstat()
    tl._need(stat.S_ISREG(st.st_mode) and st.st_size <= cap, "backend", f"device provider origin domain {path}")
    return st


def _hash(p, st, cap):
    h, n = hashlib.sha256(), 0
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        tl._need(tl._stamp(os.fstat(fd)) == tl._stamp(st), "source_changed", str(p))
        while True:
            data = os.read(fd, min(tl.CHUNK, cap - n + 1))
            if not data:
                break
            n += len(data)
            tl._need(n <= cap, "byte_limit", str(p))
            h.update(data)
    finally:
        os.close(fd)
    tl._need(n == st.st_size, "source_changed", str(p))
    return n, h.hexdigest()


def device_runtime_census(modules, *, provider_root=None, limits=None):
    """Record loaded device-provider module origins; authenticity is origin, not a measurement.

    Every ``cupy``/``cupyx``/``cupy_backends`` entry must be an owning module (including stdlib lazy modules) whose
    origin is its own file inside the selected interpreter's site-packages
    family tree, with a consistent spec and package path. CuPy14 FFT config is
    admitted only as the exact canonical singleton of its authenticated owner. Python sources are
    hashed; compiled extension origins are recorded by path and size only
    (``hashed`` false), so Root's compiled inventory stays the identity
    authority for binaries. No device, context or array is touched.
    """
    if provider_root is None:
        root = Path(sysconfig.get_path("purelib"))
        tl._need(root.is_relative_to(Path(PYTHON).parent.parent), "backend",
                 "device provider root must be inside the selected interpreter environment")
    else:
        root = Path(provider_root)
    tl._need(root.is_absolute() and str(root) == os.path.normpath(str(root)), "backend", "canonical provider root required")
    cap = (tl.DEFAULT_LIMITS if limits is None else tl._limits(limits))["max_file_bytes"]
    tl._need(len(modules) <= 100000, "module_limit", "loaded module census")
    found, aliases = {}, []
    module_types = (ModuleType, importlib.util._LazyModule)
    for name, module in list(modules.items()):
        if type(name) is not str:
            raise tl.TripletLockError("module_origin", "plain module names required")
        family = name.partition(".")[0]
        if family not in DEVICE_FAMILIES or module is None:
            continue
        tl._need(len(found) < MAX_DEVICE_MODULES, "module_limit", "device provider module census")
        if name == "cupy.fft.config":
            aliases.append((name, module))
            continue
        tl._need(type(module) in module_types, "backend", name + ": device provider entry is not an owning module")
        # Bypass stdlib LazyModule.__getattribute__: census must not execute
        # optional modules or create arrays just to inspect their namespace.
        namespace = ModuleType.__getattribute__(module, "__dict__")
        origin, spec = namespace.get("__file__"), namespace.get("__spec__")
        tl._need(type(origin) is str and spec is not None and spec.name == name and spec.origin == origin, "backend",
                 name + ": inconsistent device provider origin")
        p, parts = Path(origin), name.split(".")
        tl._need(p.is_relative_to(root / family), "backend", name + ": device provider outside the selected interpreter site-packages")
        if "__path__" in namespace:
            tl._need(p.parent == root.joinpath(*parts) and p.name.partition(".")[0] == "__init__"
                     and list(namespace["__path__"]) == [str(p.parent)], "backend", name + ": device provider package layout differs")
        else:
            tl._need(p.parent == root.joinpath(*parts[:-1]) and p.name.partition(".")[0] == parts[-1], "backend",
                     name + ": device provider module layout differs")
        tl._need(p.suffix == ".py" or any(p.name.endswith(x) for x in EXTENSION_SUFFIXES), "backend",
                 name + ": only source or native extension origins are admitted; no sourceless bytecode")
        st = _origin(origin, cap if p.suffix == ".py" else MAX_EXTENSION_BYTES)
        if p.suffix == ".py":
            n, sha = _hash(p, st, cap)
            found[name] = {"path": origin, "size": n, "sha256": sha, "kind": "source", "hashed": True}
        else:
            found[name] = {"path": origin, "size": st.st_size, "sha256": None, "kind": "extension", "hashed": False}
    for name, alias in aliases:
        owner = modules.get("cupy.fft._config")
        tl._need(type(owner) in module_types and "cupy.fft._config" in found, "backend",
                 name + ": canonical FFT config owner is absent")
        ns = ModuleType.__getattribute__(owner, "__dict__")
        cls = ns.get("_FFTConfig")
        tl._need(type(cls) is type and cls.__module__ == "cupy.fft._config" and cls.__name__ == "_FFTConfig"
                 and type(alias) is cls and alias is ns.get("config"), "backend",
                 name + ": not the genuine owner's FFT config singleton")
        tl._need(len(found) < MAX_DEVICE_MODULES, "module_limit", "device provider alias census")
        found[name] = {**found["cupy.fft._config"], "kind": "config_alias", "owner": "cupy.fft._config"}
    return found


def check_runtime_sources(lock, modules, *, provider_root=None):
    """Post-entry census: exact project/helper origins plus the authenticated loaded device provider."""
    return {"origins": tr._check_import_sources(lock, modules, (), ""),
            "device_runtime": device_runtime_census(modules, provider_root=provider_root, limits=lock["qualification"]["limits"])}


def _compare_census(previous, current):
    """Earlier provider entries must be unchanged; growth is permitted because the fresh census authenticated it."""
    for name, row in previous.items():
        tl._need(current.get(name) == row, "backend", name + ": loaded device provider module changed")


def _source_hook(source, lock):
    """After the owning SYRUP verification: the sealed record itself must grant device capability."""
    q = lock["qualification"]
    tl._need(source.profile == q["native_profile"] == NATIVE_PROFILE, "unqualified_profile",
             "owning source profile differs from the qualified device-runtime profile")
    tl._need(source.device_runtime_capability is True, "backend", "owning native source profile grants no device runtime capability")
    tl._need(not any(type(n) is str and n.partition(".")[0] in DEVICE_FAMILIES for n in sys.modules), "backend", REFUSED_DETAIL)


def _receipt_extra(lock, source):
    return {"device": device_facts(lock["qualification"]), "device_runtime_capability": True,
            "device_provider_loaded": False, "device_executed": False, "compiled_identities_hashed": False}


@dataclasses.dataclass(frozen=True)
class CudaRuntimeContext:
    """A verified CUDA source capability supplied only by the cold worker.

    It carries identity labels (lock, device selection, native profile), never
    a device handle, stream, array or physics configuration. Adapters must join
    their expected LEAF/SYRUP source and helper values to ``record()``. Device
    observations enter only through ``record_device_runtime`` and are stored
    verbatim as unverified entry-point reports.
    """
    profile = PROFILE
    _load = staticmethod(load)
    _verify = staticmethod(verify)
    lock_path: str
    lock_sha256: str
    source_profile: object
    device: dict
    _ledger: dict = dataclasses.field(default_factory=dict, repr=False, compare=False)

    def record(self):
        lock = self._load(self.lock_path, expected_sha256=self.lock_sha256)
        q = lock["qualification"]
        return {"profile": self.profile, "lock_sha256": self.lock_sha256, "backend": BACKEND, "native_profile": q["native_profile"],
                "device": _device(q["device"]),
                "sources": {n: {k: q["sources"][n][k] for k in ("commit", "tree", "package_sha256")} for n in tl.MODELS},
                "external_helpers": [tr.helper_record(lock)]}

    def record_device_runtime(self, facts):
        """Store an entry point's own device/compiled-dependency observations; nothing is measured or verified here."""
        tl._need(type(facts) is dict and all(type(k) is str for k in facts), "device_record", "plain string-keyed mapping required")
        try:
            data = tl.canonical(facts)
        except (TypeError, ValueError) as exc:
            raise tl.TripletLockError("device_record", "finite JSON-representable facts required") from exc
        tl._need(len(data) <= MAX_REPORT_BYTES, "device_record", "bounded device report")
        reports = self._ledger.setdefault("reported", [])
        tl._need(len(reports) < MAX_REPORTS, "device_record", "device report count")
        reports.append({"facts": tl._json(data), "verified_by_runtime": False})

    def device_runtime_reports(self):
        return [dict(r) for r in self._ledger.get("reported", ())]

    def recheck(self):
        # No Git subprocess during transport: sealed files, the owning source
        # capability, the device selection and the loaded provider are
        # rechecked; the parent checks branch currency at launch.
        lock = self._verify(self.lock_path, expected_sha256=self.lock_sha256, current_refs=False)
        q = lock["qualification"]
        tl._need(dict(self.device) == _device(q["device"]), "device", "runtime device record differs from the lock")
        tl._need(os.environ.get("CUDA_VISIBLE_DEVICES") == q["device"]["physical_uuid"], "device", "selected physical device changed")
        self.source_profile.recheck()
        tl._need(self.source_profile.device_runtime_capability is True, "backend",
                 "owning native source profile grants no device runtime capability")
        found = check_runtime_sources(lock, sys.modules)
        identities = {n: id(m) for n, m in sys.modules.items() if type(n) is str and n.partition(".")[0] in DEVICE_FAMILIES}
        if "census" in self._ledger:
            _compare_census(self._ledger["census"], found["device_runtime"])
            for name, ident in self._ledger["identities"].items():
                tl._need(identities.get(name) == ident, "backend", name + ": device provider module object was replaced")
        self._ledger["census"], self._ledger["identities"] = found["device_runtime"], identities
        return found


def _context(lock_path, lock_sha256, source, lock):
    return CudaRuntimeContext(lock_path, lock_sha256, source, _device(lock["qualification"]["device"]))


def verify_imports(lock):
    """Genuine owning bundle/profile checks under the CUDA policy; CuPy must not be loaded yet."""
    return tr._verify_imports(lock, _RUNTIME)


def bootstrap_spec(lock_path, *, expected_sha256, run_dir, entrypoint=None, argv=()):
    """Return argv/env selecting the locked physical device; never spawn anything.

    ``CUDA_VISIBLE_DEVICES`` is the locked physical UUID, so logical device 0
    is that device. Root's supervisor owns launch, time, memory and GPU
    permission; this grants no transport authority.
    """
    lock = verify(lock_path, expected_sha256=expected_sha256)
    spec = tr._bootstrap_spec(lock, lock_path, expected_sha256, run_dir=run_dir, entrypoint=entrypoint, argv=argv, policy=_RUNTIME)
    spec["device"] = device_facts(lock["qualification"])
    spec["scope"] = "Root must impose actual process/resource/device bounds; no process has been launched and no device was opened"
    return spec


def worker_main(argv):
    return tr._worker_main(argv, _RUNTIME)


_RUNTIME = tr._RuntimePolicy("CUDA", PROFILE, PYTHON, BACKEND, BOOTSTRAP, DEVICE_FAMILIES, REFUSED_DETAIL, CACHES,
                             _environment, verify, _context, _source_hook, _receipt_extra, False)
