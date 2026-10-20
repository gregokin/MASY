"""Explicit parent-gated clone qualification; never collected/run by default.

No module-level CuPy import or hardware query. A future parent controller must
launch this callable under a NEW genuine sealed GPU profile containing the
owner, exact worker environment and bounded run_bounded lifecycle. It supplies
both the native identity/allocation gate and the additional copy-test scope.
Missing scope is a refusal, never a skip or permission inferred from collection.
This file is source-only in task39; it has NOT executed on hardware.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import time
from pathlib import Path

REQUIRED_CHECKS = frozenset(
    (
        "native_validation",
        "copy_disjointness",
        "rollback",
        "candidate_isolation",
        "signed_zero_integrity",
        "overlap_refusal",
        "counters",
    )
)
_SCOPE_KEYS = {
    "schema",
    "authorized",
    "operation",
    "profile_id",
    "store",
    "uuid",
    "expires_unix_s",
    "test_sha256",
    "module_sha256",
    "max_bed_bytes",
    "max_work_bytes",
    "native_validation_scratch_bytes",
    "comparison_reduction_scratch_bytes",
    "max_pool_total_bytes",
    "max_driver_delta_bytes",
    "max_native_d2h_bytes",
    "max_elapsed_s",
}


def _scope(request, scope, module_path):
    from masy import gpu_runtime_worker as guard

    guard.closed(scope, _SCOPE_KEYS, "device bed test scope")
    if (
        scope["schema"] != "masy.device_bed_test_scope.v1"
        or scope["authorized"] is not True
        or scope["operation"] != "dense_device_owner"
        or scope["profile_id"] != request["profile_id"]
        or scope["uuid"] != request["device"]["uuid"]
    ):
        raise ValueError(
            "explicit parent device-owner copy-test scope/profile/UUID required"
        )
    for name in _SCOPE_KEYS - {
        "schema",
        "authorized",
        "operation",
        "profile_id",
        "store",
        "uuid",
        "expires_unix_s",
        "test_sha256",
        "module_sha256",
    }:
        if type(scope[name]) is not int or scope[name] <= 0:
            raise ValueError("positive plain resource/test bounds required")
    expiry = scope["expires_unix_s"]
    if (
        type(expiry) not in (int, float)
        or not math.isfinite(expiry)
        or time.time() > expiry
    ):
        raise ValueError("copy-test authorization expired or malformed")
    for path, key in (
        (Path(__file__), "test_sha256"),
        (Path(module_path), "module_sha256"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != scope[key]:
            raise ValueError("exact test/module source differs from parent scope")
    if request["operation"] != "probe":
        raise ValueError("parent must supply the genuine sealed probe-shaped request")
    from types import SimpleNamespace

    from masy import gpu_runtime

    # Reuse exact read-only source/profile verification and native request builder.
    profile, profile_limits, _ = gpu_runtime._load_profile(
        scope["store"], request["profile_id"]
    )
    run = SimpleNamespace(
        dir=Path(request["run"]["run_dir"]), pycache=request["run"]["pycache_prefix"]
    )
    expected = gpu_runtime._profile_body(profile, run, profile["runtime"]["python"])
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError("request does not bind the exact immutable source profile")
    if scope["max_elapsed_s"] > profile_limits.worker_timeout_s:
        raise ValueError("test window exceeds the immutable operational profile")
    # Native physical freshness/UUID/PCI/process validation, no invented resource engine.
    guard.check_probe_gate(
        request["probe_gate"], request["profile_id"], request["device"]
    )


def _fixture():
    """Tiny authored two-class bed, no event or calibrated response law."""
    import numpy as np
    from maple.core.boundaries import AxisBoundary, BoundaryKind
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.parameters.geometry import GeometrySpec
    from maple.core.parameters.grain_classes import GrainClass, GrainClassSet
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.types.threshold_modifiers import neutral_threshold_modifier_state
    from maple.core.types.vegetation import empty_vegetation_state

    from masy.dense_bed import DenseBedLimits, build_dense_bed

    boundary = AxisBoundary(BoundaryKind.PERIODIC)
    geometry = GeometrySpec(
        nx=2, ny=2, dx_m=0.5, dy_m=0.5, boundary_x=boundary, boundary_y=boundary
    )
    classes = GrainClassSet(
        classes=tuple(
            GrainClass(
                class_id=key,
                diameter_m=diameter,
                particle_density_kg_m3=2650.0,
                is_aggregate=False,
            )
            for key, diameter in (("fine", 1e-4), ("coarse", 3e-4))
        )
    )
    return build_dense_bed(
        geometry=geometry,
        grain_classes=classes,
        mass_resolution_kg=1e-10,
        n_voxels=3,
        initial_bed_elevation_m=np.full((2, 2), 0.12),
        initial_class_fractions=np.full((2, 2, 2), 0.5),
        initial_available_fraction=np.full((2, 2, 2), 0.25),
        initial_water_depth_m=np.zeros((2, 2)),
        vegetation=empty_vegetation_state(2, 2),
        threshold_modifier_state=neutral_threshold_modifier_state(2, 2),
        topographic_commit_spec=TopographicCommitSpec(),
        avalanching_spec=AvalanchingSpec(enabled=False),
        active_mask=np.ones((2, 2), dtype=bool),
        initial_time_s=0.0,
        limits=DenseBedLimits(1 << 26, 3, 1 << 24),
    )


def run_authorized_checks(request, scope, *, base_sys_path):
    """Called only by a separately reviewed parent-owned bounded sealed worker.

    Returns bounded metadata and required checkset. No full fields or datasets.
    Runtime/source/profile preparation and actual allocation query are external.
    """
    from masy import gpu_runtime_worker as guard

    started = time.monotonic()
    guard.check_request(request)
    process = guard.check_process(request, base_sys_path)
    module_path = Path(request["packages"]["masy"]["dir"]) / "dense_device_bed.py"
    _scope(request, scope, module_path)
    native = guard.native_audit(
        request
    )  # Genuine bundle, metadata, origins, source digests.
    from masy import dense_device_bed as device

    if Path(device.__file__).resolve() != module_path.resolve():
        raise ValueError("owner import differs from the source-verified profile path")
    host = _fixture()
    from maple.core.backend import array_leaves
    from maple.core.backend.arrays import array_leaf_inventory

    inventory = array_leaf_inventory(host.state)
    limits = device.DeviceBedLimits(
        *(
            scope[key]
            for key in (
                "max_bed_bytes",
                "max_work_bytes",
                "native_validation_scratch_bytes",
                "comparison_reduction_scratch_bytes",
            )
        )
    )
    _, required = device._work_reservation(
        inventory, max(a.nbytes for a in array_leaves(host.state)), limits
    )
    if (
        inventory["transfer_bytes"] > limits.max_bed_bytes
        or required > limits.max_work_bytes
    ):
        raise ValueError(
            "fixture plus comparison/native scratch exceeds scope before context"
        )
    if (
        time.time() > scope["expires_unix_s"]
        or time.monotonic() - started > scope["max_elapsed_s"]
    ):
        raise ValueError("operation authority/window expired before context")
    guard.check_probe_gate(
        request["probe_gate"], request["profile_id"], request["device"]
    )
    import cupy as cp  # All explicit authority/source/process checks already passed.
    from maple.core.backend import array_leaves, arrays_share_memory, to_device_tree
    from maple.core.backend.selection import resolve_backend

    if cp.cuda.runtime.getDeviceCount() != 1:
        raise ValueError("exactly one physically scoped visible device required")
    observed = guard.check_observed_device(
        request["device"],
        cp.cuda.runtime.getDeviceProperties(0),
        cp.cuda.runtime.deviceGetPCIBusId(0),
    )
    backend = resolve_backend("cupy", device_id=0)
    completed = set()
    with backend.device_scope():
        pool = cp.get_default_memory_pool()
        old_limit = pool.get_limit()
        if pool.total_bytes() > scope["max_pool_total_bytes"]:
            raise ValueError("existing pool already exceeds test scope")
        pool.set_limit(size=scope["max_pool_total_bytes"])
        free_before, _ = cp.cuda.runtime.memGetInfo()
        try:
            state = to_device_tree(
                host.state, cp
            )  # One explicit initial upload, outside owner counters.
            owner = device.DeviceDenseBedOwner.from_device_bed(
                state,
                geometry=host.context.geometry,
                mass_resolution_kg=host.context.mass_resolution_kg,
                n_voxels=3,
                n_classes=2,
                case_sha256=host.case_sha256,
                time_s=host.time_s,
                limits=limits,
            )
            owner.verify()
            completed.add("native_validation")
            # Test-only internal pointer observations; no accepted buffer is a public API result.
            for a in array_leaves(owner._accepted.state):
                for b in array_leaves(state):
                    if arrays_share_memory(a, b):
                        raise AssertionError("initial copy aliases input")
            completed.add("copy_disjointness")
            stage = owner.begin_stage()
            stage.state.water.depth_m.fill(0.01)
            owner.discard(stage)
            owner.verify()
            if owner.generation != 0:
                raise AssertionError("discard published work")
            completed.add("rollback")
            stage = owner.begin_stage()
            candidate = stage.state
            prepared = owner.prepare_publish(stage, candidate, time_s=owner.time_s)
            owner.publish(prepared)
            candidate.water.depth_m.fill(0.02)
            owner.verify()
            if owner.generation != 1 or arrays_share_memory(
                owner._accepted.state.water.depth_m, candidate.water.depth_m
            ):
                raise AssertionError(
                    "publication did not advance to an independent accepted copy"
                )
            completed.add("candidate_isolation")
            owner._accepted.state.water.depth_m[0, 0] = -0.0
            try:
                owner.verify()
            except device.DeviceBedError as exc:
                if exc.code != "integrity_failed":
                    raise
            else:
                raise AssertionError("raw signed-zero corruption not detected")
            owner._accepted.state.water.depth_m[0, 0] = 0.0
            owner.verify()
            completed.add("signed_zero_integrity")
            stage = owner.begin_stage()
            bad = dataclasses.replace(
                stage.state,
                water=dataclasses.replace(
                    stage.state.water,
                    depth_m=stage.state.committed_topography.reference_elevation_m.view(),
                ),
            )
            try:
                owner.prepare_publish(stage, bad, time_s=owner.time_s)
            except device.DeviceBedError as exc:
                if exc.code != "overlapping_views":
                    raise
            else:
                raise AssertionError("distinct overlapping view accepted")
            owner.discard(stage)
            owner.verify()
            completed.add("overlap_refusal")
            evidence = owner.evidence()
            delta = evidence["native_transfer_delta"]
            if (
                delta["host_to_device_bytes"] != 0
                or delta["device_to_host_bytes"] > scope["max_native_d2h_bytes"]
            ):
                raise AssertionError("owner helper transfer scope exceeded")
            if evidence["copy_and_packet_counters"]["d2d_enqueued_arrays"] <= 0:
                raise AssertionError("no genuine device copy was exercised")
            completed.add("counters")
            cp.cuda.get_current_stream().synchronize()
            free_after, _ = cp.cuda.runtime.memGetInfo()
            memory = {
                "pool_used_bytes": pool.used_bytes(),
                "pool_total_bytes": pool.total_bytes(),
                "sampled_driver_delta_bytes": max(0, free_before - free_after),
                "driver_peak_proven": False,
            }
            if (
                memory["pool_total_bytes"] > scope["max_pool_total_bytes"]
                or memory["sampled_driver_delta_bytes"]
                > scope["max_driver_delta_bytes"]
            ):
                raise AssertionError("pool/sampled driver test cap exceeded")
        finally:
            pool.set_limit(size=old_limit)
    if (
        completed != REQUIRED_CHECKS
        or time.monotonic() - started > scope["max_elapsed_s"]
    ):
        raise AssertionError("required callset missing or operation window exceeded")
    _, binaries = guard.environment_metadata()
    compiled = guard.compiled_inventory(request["limits"], binaries)
    if (
        time.monotonic() - started > scope["max_elapsed_s"]
        or time.time() > scope["expires_unix_s"]
    ):
        raise AssertionError("complete test/compiled-evidence window exceeded")
    return {
        "scope": "genuine tiny clone/validation only, no solver or transport qualification",
        "completed_checks": sorted(completed),
        "required_checks": sorted(REQUIRED_CHECKS),
        "skips": 0,
        "process": process,
        "device": observed,
        "profile_id": request["profile_id"],
        "native_audit": native,
        "owner": evidence,
        "memory": memory,
        "compiled": compiled,
        "cupy_version": cp.__version__,
        "elapsed_s": time.monotonic() - started,
        "counter_limits": "native helper counters; library-internal transfers/driver peak not proven",
        "controller_contract": "parent must own fresh allocation, exact source/profile, wall/CPU/output bounds",
    }


if __name__ == "__main__":
    raise SystemExit(
        "refused: invoke only from the reviewed parent bounded sealed worker with explicit scope"
    )


def load_tests(loader, tests, pattern):
    raise RuntimeError(
        "no automatic/unittest execution: parent must run_authorized_checks with exact callset"
    )
