"""Explicit parent-gated construction check of the private Dense device runner; never collected/run by default.

No module-level CuPy import or hardware query. A future parent controller must launch
`run_authorized_checks` under a NEW genuine sealed GPU profile (exact owner/worker environment, bounded
run_bounded lifecycle) and supply the native identity/allocation gate plus this file's own construction scope.
Missing scope is a refusal, never a skip or permission inferred from collection.

Scope of this first genuine case: construct the runner from the authored six-class host fixture through the
private factory (one native `to_device_tree` per source object), verify owner/static/device-context binding and
SYRUP namespace agreement, detect a raw static change, and release -- WITHOUT any wind step, rain event,
CUDA hydraulics or transport. The four-event device run and any CPU-GPU comparison are separately authorized
later launches; nothing here qualifies them. Only explicit parent run receipts determine whether and how this
source executed.
"""

from __future__ import annotations

import gc
import hashlib
import math
import sys
import time
import weakref
from pathlib import Path

REQUIRED_CHECKS = frozenset(
    (
        "construction",
        "no_transport",
        "device_context_namespace",
        "shared_availability_binding",
        "water_parameter_namespace",
        "static_verification",
        "raw_static_change_detected",
        "private_labels",
        "release",
    )
)
_BOUND_KEYS = (
    "max_device_bytes",
    "hydraulics_max_log",
    "wind_workspace_bytes",
    "routing_workspace_bytes",
    "max_bed_bytes",
    "max_bed_work_bytes",
    "native_validation_scratch_bytes",
    "bed_comparison_scratch_bytes",
    "max_private_bytes",
    "max_static_bytes",
    "max_resource_work_bytes",
    "resource_comparison_scratch_bytes",
    "max_pool_total_bytes",
    "max_driver_delta_bytes",
    "max_native_d2h_bytes",
    "max_elapsed_s",
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
    *_BOUND_KEYS,
}


def _scope(request, scope, module_path):
    from masy import gpu_runtime_worker as guard

    guard.closed(scope, _SCOPE_KEYS, "device runner test scope")
    if (
        scope["schema"] != "masy.device_runner_test_scope.v1"
        or scope["authorized"] is not True
        or scope["operation"] != "dense_device_runner_construction"
        or scope["profile_id"] != request["profile_id"]
        or scope["uuid"] != request["device"]["uuid"]
    ):
        raise ValueError("explicit parent device-runner construction scope/profile/UUID required")
    for name in _BOUND_KEYS:
        if type(scope[name]) is not int or scope[name] <= 0:
            raise ValueError("positive plain resource/test bounds required")
    expiry = scope["expires_unix_s"]
    if type(expiry) not in (int, float) or not math.isfinite(expiry) or time.time() > expiry:
        raise ValueError("construction-test authorization expired or malformed")
    for path, key in ((Path(__file__), "test_sha256"), (Path(module_path), "module_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != scope[key]:
            raise ValueError("exact test/module source differs from parent scope")
    if request["operation"] != "probe":
        raise ValueError("parent must supply the genuine sealed probe-shaped request")
    from types import SimpleNamespace

    from masy import gpu_runtime

    profile, profile_limits, _ = gpu_runtime._load_profile(scope["store"], request["profile_id"])
    run = SimpleNamespace(dir=Path(request["run"]["run_dir"]), pycache=request["run"]["pycache_prefix"])
    expected = gpu_runtime._profile_body(profile, run, profile["runtime"]["python"])
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError("request does not bind the exact immutable source profile")
    if scope["max_elapsed_s"] > profile_limits.worker_timeout_s:
        raise ValueError("test window exceeds the immutable operational profile")
    guard.check_probe_gate(request["probe_gate"], request["profile_id"], request["device"])


def _host_inputs():
    """The authored six-class alternation fixture on the HOST with the first device profile's controls (native
    `cuda` hydraulic implementation, bounded accounting steps). No device work; importable without CuPy."""
    native_tests = str(Path(__file__).resolve().parents[1] / "native")
    if native_tests not in sys.path:
        sys.path.insert(0, native_tests)
    import fixture_dense_alternation as fx

    from masy.dense_runner import DenseWindInputs

    bed = fx.build_bed()
    return {
        "bed": bed,
        "schedule": fx.reference_schedule(),
        "wind": DenseWindInputs(fx.wind_config(bed), fx.empty_pool()),
        "water": fx.water_inputs(bed),
        "control": fx.runner_control(sediment_control=fx.sediment_control(implementation="cuda")),
        "limits": fx.limits(max_wind_accounting_steps=64),
    }


def run_authorized_checks(request, scope, *, base_sys_path):
    """Called only by a separately reviewed parent-owned bounded sealed worker. Returns bounded metadata."""
    from masy import gpu_runtime_worker as guard

    started = time.monotonic()
    guard.check_request(request)
    process = guard.check_process(request, base_sys_path)
    module_path = Path(request["packages"]["masy"]["dir"]) / "dense_device_runner.py"
    _scope(request, scope, module_path)
    native = guard.native_audit(request)
    from masy import dense_device_bed as dbed
    from masy import dense_device_resources as dres
    from masy import dense_device_runner as drun

    if Path(drun.__file__).resolve() != module_path.resolve():
        raise ValueError("runner import differs from the source-verified profile path")
    host = _host_inputs()
    device_limits = drun.DeviceRunnerLimits(
        max_device_bytes=scope["max_device_bytes"], hydraulics_max_log=scope["hydraulics_max_log"],
        wind_workspace_bytes=scope["wind_workspace_bytes"], routing_workspace_bytes=scope["routing_workspace_bytes"])
    bed_limits = dbed.DeviceBedLimits(scope["max_bed_bytes"], scope["max_bed_work_bytes"],
                                      scope["native_validation_scratch_bytes"], scope["bed_comparison_scratch_bytes"])
    resource_limits = dres.DeviceResourceLimits(scope["max_private_bytes"], scope["max_static_bytes"],
                                                scope["max_resource_work_bytes"],
                                                scope["resource_comparison_scratch_bytes"])
    if time.time() > scope["expires_unix_s"] or time.monotonic() - started > scope["max_elapsed_s"]:
        raise ValueError("operation authority/window expired before context")
    guard.check_probe_gate(request["probe_gate"], request["profile_id"], request["device"])
    import cupy as cp  # All explicit authority/source/process checks already passed.
    from maple.core.backend import resolve_operand_namespace
    from maple.core.backend.selection import resolve_backend
    from maple_syrup.event_backend import host_packet

    if cp.cuda.runtime.getDeviceCount() != 1:
        raise ValueError("exactly one physically scoped visible device required")
    observed = guard.check_observed_device(request["device"], cp.cuda.runtime.getDeviceProperties(0),
                                           cp.cuda.runtime.deviceGetPCIBusId(0))
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
            runner = drun._create_dense_device_runner(
                host["bed"], host["schedule"], wind=host["wind"], water=host["water"], control=host["control"],
                limits=host["limits"], device_limits=device_limits, bed_limits=bed_limits,
                resource_limits=resource_limits, device_id=0)
            construction = runner.construction_evidence
            if construction["source_copy_verification"]["verified_device_leaves"] <= 0:
                raise AssertionError("no converted device leaf was verified against its host source")
            completed.add("construction")
            if runner._next != 0 or runner._receipts or runner._hydraulics is not None:
                raise AssertionError("construction must not run an event or create a hydraulics lifecycle")
            completed.add("no_transport")
            # Test-only internal observations (no accepted buffer is a public result).
            context, resources = runner._bed.context, runner._device_resources
            if resolve_operand_namespace(runner._device_bed._accepted.state, context.initial_available_fraction) is not cp:
                raise AssertionError("device BedContext availability is not in the bed namespace")
            if type(runner._bed.host_context.initial_available_fraction) is cp.ndarray:
                raise AssertionError("the host context was converted instead of bridged")
            completed.add("device_context_namespace")
            avail = resources.static("initial_available_fraction")
            if not (context.initial_available_fraction is avail
                    and runner._wind.config.initial_available_fraction is avail):
                raise AssertionError("wind config and water context do not share the ONE owned availability map")
            completed.add("shared_availability_binding")
            water = runner._water
            if not (water.column.xp is cp and water.sediment.xp is cp and water.field.xp is cp):
                raise AssertionError("native water parameters are not in the CuPy graph namespace")
            if type(water.sediment.diameter_m) is cp.ndarray or type(water.sediment.d_diameter) is not cp.ndarray:
                raise AssertionError("sediment host/device leaf contract not preserved")
            completed.add("water_parameter_namespace")
            runner._device_own.verify_case_inputs()
            if runner._static_parts_record(runner_native()) != runner._static_parts:
                raise AssertionError("controlling static record changed across construction")
            completed.add("static_verification")
            bits = avail.view(cp.uint64)
            original = bits[0, 0, 0].copy()
            bits[0, 0, 0] = cp.uint64(0x8000000000000000)  # IEEE binary64 -0.0
            try:
                resources.verify_static()
            except dres.DeviceResourceError as exc:
                if exc.code != "integrity_failed":
                    raise
            else:
                raise AssertionError("raw static availability change not detected")
            bits[0, 0, 0] = original
            if not all(float(v) == 1.0 for v in host_packet(cp, [bits[0, 0, 0] == original])):
                raise AssertionError("original availability bits not restored")
            resources.verify_static()
            completed.add("raw_static_change_detected")
            if (runner._accepted_bed != runner._device_own.accepted_label()
                    or "sha256" in runner._accepted_bed or "mirror-verified" not in runner._accepted_private):
                raise AssertionError("labels are not mirror/generation labels")
            completed.add("private_labels")
            evidence = runner.device_evidence()
            owner_refs = (weakref.ref(runner._device_bed), weakref.ref(runner._device_resources),
                          weakref.ref(runner))
            del runner, context, resources, avail, bits, original, water
            gc.collect()
            if any(ref() is not None for ref in owner_refs):
                raise AssertionError("runner/owner references survive release")
            completed.add("release")
            cp.cuda.get_current_stream().synchronize()
            free_after, _ = cp.cuda.runtime.memGetInfo()
            memory = {"pool_used_bytes": pool.used_bytes(), "pool_total_bytes": pool.total_bytes(),
                      "sampled_driver_delta_bytes": max(0, free_before - free_after), "driver_peak_proven": False,
                      "deallocation_asserted": False}
            if (memory["pool_total_bytes"] > scope["max_pool_total_bytes"]
                    or memory["sampled_driver_delta_bytes"] > scope["max_driver_delta_bytes"]):
                raise AssertionError("pool/sampled driver test cap exceeded")
            d2h = construction["source_copy_verification"]["downloaded_bytes"]
            if d2h > scope["max_native_d2h_bytes"]:
                raise AssertionError("construction verification downloads exceed scope")
        finally:
            pool.set_limit(size=old_limit)
    if completed != REQUIRED_CHECKS or time.monotonic() - started > scope["max_elapsed_s"]:
        raise AssertionError("required callset missing or operation window exceeded")
    _, binaries = guard.environment_metadata()
    compiled = guard.compiled_inventory(request["limits"], binaries)
    if time.monotonic() - started > scope["max_elapsed_s"] or time.time() > scope["expires_unix_s"]:
        raise AssertionError("complete test/compiled-evidence window exceeded")
    return {
        "scope": "genuine construction/binding/release only; no wind, rain, hydraulics or transport qualification",
        "completed_checks": sorted(completed),
        "required_checks": sorted(REQUIRED_CHECKS),
        "skips": 0,
        "process": process,
        "device": observed,
        "profile_id": request["profile_id"],
        "native_audit": native,
        "construction": {key: construction[key] for key in ("required_device_bytes", "owner_reservations",
                                                            "source_copy_verification", "first_profile_caps")},
        "device_evidence": evidence,
        "memory": memory,
        "compiled": compiled,
        "cupy_version": cp.__version__,
        "elapsed_s": time.monotonic() - started,
        "controller_contract": "parent must own fresh allocation, exact source/profile, wall/CPU/output bounds",
    }


def runner_native():
    from masy.dense_runner import _native

    return _native()


if __name__ == "__main__":
    raise SystemExit("refused: invoke only from the reviewed parent bounded sealed worker with explicit scope")


def load_tests(loader, tests, pattern):
    raise RuntimeError("no automatic/unittest execution: parent must run_authorized_checks with exact callset")
