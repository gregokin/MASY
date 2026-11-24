"""Separate immutable four-source profile for current irregular terrain CUDA.

Shares the reviewed stdlib bootstrap and actual CuPy provenance census. Old
CPU/CUDA profiles retain their exact schemas, native source names and defaults.
"""
from __future__ import annotations

import json
import sys

from masy import triplet_lock as tl
from masy import triplet_runtime as tr
from masy import triplet_cuda_runtime as cr

PROFILE = "masy.daily_wind_water_growth.terrain_cuda.v1"
QUALIFICATION_SCHEMA = "masy.triplet_terrain_cuda.qualification.v1"
LOCK_SCHEMA = "masy.triplet_terrain_cuda.lock.v1"
NATIVE_PROFILE = "native_maple_terrain_20261123_case_latitude_cuda_v1"
PYTHON = cr.PYTHON
GENERATOR_FILES = (*cr.GENERATOR_FILES, "src/masy/triplet_terrain_runtime.py")
BOOTSTRAP = tr._BOOTSTRAP_HEAD + (
    "if lock.get('schema')!=%s or lock.get('profile')!=%s:\n"
    "    raise RuntimeError('explicit terrain CUDA triplet lock required')\n"
    "device=lock.get('qualification',{}).get('device',{}).get('physical_uuid')\n"
    "if type(device) is not str or os.environ.get('CUDA_VISIBLE_DEVICES')!=device:\n"
    "    raise RuntimeError('selected physical device differs from the lock')\n"
    "from masy.triplet_terrain_runtime import worker_main\n"
    "sys.exit(worker_main(sys.argv[1:]))\n") % (json.dumps(LOCK_SCHEMA), json.dumps(PROFILE))


def _extension(q):
    backend = tl._keys(q["backend"], ("name", "provider"), "backend")
    tl._need(backend == {"name": "cuda", "provider": "cupy"}, "backend", "explicit cuda/cupy backend required")
    cr._device(q["device"])
    tl._need(q["native_profile"] == NATIVE_PROFILE, "unqualified_profile", "exact native terrain source profile required")


_POLICY = tl._Policy("terrain CUDA", QUALIFICATION_SCHEMA, LOCK_SCHEMA, PROFILE,
    PYTHON, cr.INTERPRETATION, GENERATOR_FILES, ("backend", "device"), _extension)


def draft_qualification(sources, *, interpreter, maple_bundle, device, entrypoints=(), limits=None):
    result = tl.draft_qualification(sources, interpreter=interpreter,
        maple_bundle=maple_bundle, native_profile=NATIVE_PROFILE, entrypoints=entrypoints, limits=limits)
    result["qualification"] = {**result["qualification"], "schema": QUALIFICATION_SCHEMA,
        "profile": PROFILE, "backend": {"name": "cuda", "provider": "cupy"}, "device": cr._device(device)}
    return result


def load_qualification(path, *, expected_sha256):
    return tl._load_qualification(path, expected_sha256, _POLICY)


def prepare(qualification_path, *, expected_sha256, destination):
    return tl._prepare(qualification_path, expected_sha256, destination, _POLICY)


def load(path, *, expected_sha256):
    return tl._load(path, expected_sha256, _POLICY)


def verify(path, *, expected_sha256, current_refs=True):
    return tl._verify(path, expected_sha256, current_refs, _POLICY)


def _source_hook(source, lock):
    tl._need(source.profile == lock["qualification"]["native_profile"] == NATIVE_PROFILE,
        "unqualified_profile", "owning source profile differs from the exact terrain profile")
    tl._need(source.device_runtime_capability is True, "backend", "native terrain source grants no CUDA capability")
    tl._need(not any(type(n) is str and n.partition(".")[0] in cr.DEVICE_FAMILIES for n in sys.modules),
        "backend", cr.REFUSED_DETAIL)


class TerrainCudaRuntimeContext(cr.CudaRuntimeContext):
    profile = PROFILE
    _load = staticmethod(load)
    _verify = staticmethod(verify)


def _context(lock_path, lock_sha256, source, lock):
    return TerrainCudaRuntimeContext(lock_path, lock_sha256, source, cr._device(lock["qualification"]["device"]))


def verify_imports(lock):
    return tr._verify_imports(lock, _RUNTIME)


def bootstrap_spec(lock_path, *, expected_sha256, run_dir, entrypoint=None, argv=()):
    lock = verify(lock_path, expected_sha256=expected_sha256)
    return tr._bootstrap_spec(lock, lock_path, expected_sha256, run_dir=run_dir,
        entrypoint=entrypoint, argv=argv, policy=_RUNTIME)


def worker_main(argv):
    return tr._worker_main(argv, _RUNTIME)


_RUNTIME = tr._RuntimePolicy("terrain CUDA", PROFILE, PYTHON, "cuda", BOOTSTRAP,
    cr.DEVICE_FAMILIES, cr.REFUSED_DETAIL, cr.CACHES, cr._environment, verify,
    _context, _source_hook, cr._receipt_extra, False)
