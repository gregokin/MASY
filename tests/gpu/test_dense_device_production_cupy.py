"""Dormant production-owner construction witness; automatic test runners execute no case.

The reviewed protected parent must supply a NEW genuine committed source profile, independent native-probe
and construction grants, all fixture/tool bytes, selected UUID/PCI allocation, source/environment/bootstrap and
resource observers. This function inherits that controller authority; hashes inside scope are not an independent
external root. Full original four-event schedule is retained, ZERO steps. No production/calibration/peak/restart
qualification follows from construction. No module-level native/CuPy import or hardware observation.
"""
from __future__ import annotations

import dataclasses
import gc
import importlib.machinery
import math
import os
import stat
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import weakref

REQUIRED_CHECKS = frozenset(("construction", "zero_transport", "original_schedule", "host_bed_release",
    "shared_policy", "host_hop_tables", "controlling_binding", "private_state", "release"))
_FIXTURES = frozenset(("fixture_dense_production", "fixture_dense_alternation", "tests", "tests.phase22b",
    "tests.phase22b.test_reduced_detachment_config", "tests.phase22b.test_reduced_integration",
    "tests.unit._scheduler_helpers"))
_BOUND_KEYS = frozenset(("max_device_bytes", "hydraulics_max_log", "wind_workspace_bytes", "routing_workspace_bytes",
    "max_bed_bytes", "max_bed_work_bytes", "native_validation_scratch_bytes", "bed_comparison_scratch_bytes",
    "max_private_bytes", "max_static_bytes", "max_resource_work_bytes", "resource_comparison_scratch_bytes",
    "max_pool_total_bytes", "max_driver_delta_bytes", "max_native_d2h_bytes", "max_elapsed_s"))
_LIMIT_KEYS = frozenset(("fft_workspace_reserve_bytes", "jit_driver_reserve_bytes", "native_transient_reserve_bytes",
    "diagnostic_scratch_bytes", "cpu_soft_s", "cpu_hard_s", "file_size_bytes", "nofile", "max_host_rss_bytes",
    "min_host_free_bytes", "max_device_bytes", "min_device_free_bytes", "max_separation_samples",
    "max_separation_owners", "max_events"))
_SCOPE_KEYS = frozenset(("schema", "authorized", "operation", "profile_id", "store", "uuid", "expires_unix_s",
    "test_sha256", "module_sha256", "production_module_sha256", "fixture_tools", "production_limits")) | _BOUND_KEYS


def _hash(path):
    from masy.runtime import _hash_path
    return _hash_path(Path(path), budget=1 << 20)[0]



def _fixture_roots(request, tools):
    """Closed seven-file run-owned source copies; reject any unbound executable before fixture import."""
    run = Path(request["run"]["run_dir"])
    paths = {name: Path(ref["path"]) for name, ref in tools.items()}
    masy_root = paths["fixture_dense_production"].parent
    native_root = paths["tests.phase22b.test_reduced_detachment_config"].parents[2]
    expected = {"fixture_dense_production": masy_root / "fixture_dense_production.py",
        "fixture_dense_alternation": masy_root / "fixture_dense_alternation.py",
        "tests": native_root / "tests/__init__.py", "tests.phase22b": native_root / "tests/phase22b/__init__.py",
        "tests.phase22b.test_reduced_detachment_config": native_root / "tests/phase22b/test_reduced_detachment_config.py",
        "tests.phase22b.test_reduced_integration": native_root / "tests/phase22b/test_reduced_integration.py",
        "tests.unit._scheduler_helpers": native_root / "tests/unit/_scheduler_helpers.py"}
    roots = (masy_root, native_root)
    if paths != expected or any(root == run or not root.is_relative_to(run) or str(root.resolve()) != str(root) for root in roots):
        raise ValueError("exact canonical run-owned copied fixture layout required")
    if masy_root.is_relative_to(native_root) or native_root.is_relative_to(masy_root):
        raise ValueError("fixture source roots must be disjoint")
    wanted = set(paths.values())
    directories = set(roots)
    for path in wanted:
        root = next(root for root in roots if path.is_relative_to(root))
        directories.update(parent for parent in path.parents if parent.is_relative_to(root))
    pending, observed = list(roots), set()
    while pending:
        folder = pending.pop()
        info = folder.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o222:
            raise ValueError("copied fixture directory must be readonly and unlinked")
        with os.scandir(folder) as entries:
            for entry in entries:
                path = folder / entry.name
                info = entry.stat(follow_symlinks=False)
                if path in directories and stat.S_ISDIR(info.st_mode) and not info.st_mode & 0o222:
                    pending.append(path)
                elif (path in wanted and stat.S_ISREG(info.st_mode) and not info.st_mode & 0o222
                      and info.st_nlink == 1 and 0 <= info.st_size <= 1 << 20):
                    observed.add(path)
                else:
                    raise ValueError("unexpected linked/writable/unbound copied fixture entry")
    if observed != wanted:
        raise ValueError("complete seven copied fixture source inventory required")
    return roots


def _scope(request, scope, base_sys_path):
    from masy import gpu_runtime, gpu_runtime_worker as guard
    guard.closed(scope, _SCOPE_KEYS, "production construction scope")
    if (scope["schema"] != "masy.production_device_construction_scope.v1" or scope["authorized"] is not True
            or scope["operation"] != "production_device_construction" or request["operation"] != "probe"
            or scope["profile_id"] != request["profile_id"] or scope["uuid"] != request["device"]["uuid"]):
        raise ValueError("explicit cross-bound production construction scope required")
    for name in _BOUND_KEYS:
        if type(scope[name]) is not int or not 0 < scope[name] <= 1 << 40:
            raise ValueError("bounded positive plain resource declarations required")
    expiry = scope["expires_unix_s"]
    if type(expiry) not in (int, float) or not math.isfinite(expiry) or time.time() > expiry:
        raise ValueError("construction scope expired")
    guard.closed(scope["production_limits"], _LIMIT_KEYS, "production limits")
    tools = scope["fixture_tools"]
    guard.closed(tools, _FIXTURES, "seven exact fixture source modules")
    for ref in tools.values():
        guard.closed(ref, {"path", "sha256"}, "fixture source reference")
        if type(ref["path"]) is not str or not Path(ref["path"]).is_absolute() or _hash(ref["path"]) != ref["sha256"]:
            raise ValueError("fixture source differs from controller binding")
    _fixture_roots(request, tools)
    module = Path(request["packages"]["masy"]["dir"])
    for path, key in ((Path(__file__), "test_sha256"), (module / "dense_device_runner.py", "module_sha256"),
                      (module / "dense_device_production.py", "production_module_sha256")):
        if _hash(path) != scope[key]:
            raise ValueError("exact production constructor/test source differs")
    profile, profile_limits, _ = gpu_runtime._load_profile(scope["store"], request["profile_id"])
    if profile["sources"]["masy"].get("source_status") != "committed":
        raise ValueError("a genuine new committed MASY source profile is required")
    run = SimpleNamespace(dir=Path(request["run"]["run_dir"]), pycache=request["run"]["pycache_prefix"])
    expected = gpu_runtime._profile_body(profile, run, profile["runtime"]["python"])
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError("request differs from complete sealed committed profile")
    if scope["max_elapsed_s"] > profile_limits.worker_timeout_s:
        raise ValueError("construction window exceeds sealed profile")
    # The native helper owns bootstrap, prefix, process and independent probe-gate validation.
    guard.check_probe_gate(request["probe_gate"], request["profile_id"], request["device"])
    return module



def _post_process_check(request, base_sys_path, process):
    """Actual115 entry/post cadence: record the two known JAX additions, never reset the environment.

    Entry uses unchanged native check_process BEFORE native imports. This post adapter inherits reviewed
    parent authority; it does not widen native worker_env, thread/arena settings, or profile selection.
    """
    flags = {key: getattr(sys.flags, key) for key in process["flags"]}
    if (os.getpid() != process["pid"]
            or Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19] != process["start_ticks"]
            or os.getcwd() != process["cwd"] or sys.executable != process["executable"]
            or os.path.realpath(sys.executable) != process["executable_realpath"] or sys.prefix != process["prefix"]
            or flags != process["flags"] or sys.flags.optimize or sys.pycache_prefix != process["pycache_prefix"]
            or sys.path != [*request["sys_path_prefix"], *base_sys_path]):
        raise ValueError("post-construction process/path/flags differ")
    before, after = process["environment"], dict(os.environ)
    delta = {"extra": {k: after[k] for k in sorted(after.keys() - before.keys())},
             "missing": sorted(before.keys() - after.keys()),
             "changed": {k: after[k] for k in sorted(before.keys() & after.keys()) if after[k] != before[k]}}
    if delta != {"extra": {"TF_CPP_MIN_LOG_LEVEL": "1", "TPU_SKIP_MDS_QUERY": "1"}, "missing": [], "changed": {}}:
        raise ValueError("unexpected post-native-import environment delta")
    return {"complete_sys_path_restored": True, "flags_unchanged": True, "environment_delta": delta,
            "thread_arena_policy": "absent_unbounded_by_environment_external_whole_worker_caps_required"}


def _host_inputs(request, scope):
    # Keep the ENTIRE immutable runtime prefix first; tool copies follow it before unrelated base paths.
    tools = scope["fixture_tools"]
    roots = _fixture_roots(request, tools)
    if any(name in sys.modules for name in (*_FIXTURES, "tests.unit")):
        raise ValueError("fixture namespaces must be fresh before their bound source imports")
    modules_before = set(sys.modules)
    point = len(request["sys_path_prefix"])
    sys.path[point:point] = [str(root) for root in roots]
    import fixture_dense_production as fixture
    from maple.core.parameters import validate_config
    from maple.core.types.cohort import zeros_mobile_cohort_state
    from masy.dense_production import schedule_envelope
    from masy.dense_runner import DenseWindInputs
    from masy.dense_wind_profile import DenseProductionGpuWindProfile, build_dense_production_gpu_wind_config
    for name, ref in tools.items():
        loaded = sys.modules.get(name)
        if loaded is None or Path(loaded.__file__).resolve() != Path(ref["path"]).resolve() or _hash(ref["path"]) != ref["sha256"]:
            raise ValueError("fixture module import origin/source differs")
    unit = sys.modules.get("tests.unit")
    # Python's actual namespace loader gives __file__=None, not an executable __init__.py.
    if (unit is None or getattr(unit, "__file__", None) is not None or unit.__spec__.origin is not None
            or type(unit.__loader__) is not importlib.machinery.NamespaceLoader
            or unit.__spec__.loader is not unit.__loader__
            or list(unit.__path__) != [str(Path(tools["tests.unit._scheduler_helpers"]["path"]).parent)]):
        raise ValueError("exact native helper namespace required; no unit initializer executes")
    imported = set()
    for name in set(sys.modules) - modules_before:
        loaded = sys.modules[name]
        if loaded is None:
            continue
        origin = getattr(getattr(loaded, "__spec__", None), "origin", None)
        locations = [value for value in (getattr(loaded, "__file__", None), origin)
                     if type(value) is str and Path(value).is_absolute()]
        locations.extend(str(value) for value in getattr(loaded, "__path__", ()))
        if any(Path(value).is_relative_to(root) for value in locations for root in roots):
            imported.add(name)
    if imported != set(_FIXTURES) | {"tests.unit"}:
        raise ValueError("fixture imports escaped the seven bound sources and one empty namespace")
    authored = fixture.model()
    # Author a NEW original GPU ModelConfig before bed/build/audit. Do not relabel an existing scheduler/run.
    model = dataclasses.replace(authored, execution=dataclasses.replace(authored.execution, backend="cupy",
                                                                       deterministic_scatter=True))
    validate_config(model)
    schedule = fixture.four_events()
    bed = fixture.build_bed(model)
    profile = DenseProductionGpuWindProfile(model, "synthetic_test_fixture",
        "native authored synthetic100um interface; wet Plot1class1 at100um; production calibration pending")
    envelope, _ = schedule_envelope(schedule)
    config = build_dense_production_gpu_wind_config(profile, wind_envelope_record=envelope,
                                                     initial_available_fraction=bed.context.initial_available_fraction)
    control = fixture.legacy.runner_control(sediment_control=fixture.legacy.sediment_control(implementation="cuda"))
    return bed, schedule, DenseWindInputs(config, zeros_mobile_cohort_state(model.particle_transport.max_active_cohorts)), \
        fixture.water_inputs(bed), control, fixture.legacy.limits(max_wind_accounting_steps=64), profile


def run_authorized_checks(request, scope, *, base_sys_path):
    from masy import gpu_runtime_worker as guard
    started = time.monotonic()
    guard.check_request(request)
    process = guard.check_process(request, base_sys_path)
    module = _scope(request, scope, base_sys_path)
    native = guard.native_audit(request)  # actual source/environment check BEFORE any device observation
    from masy import dense_device_bed as dbed, dense_device_resources as dres
    from masy import dense_device_runner as drun, dense_device_production as prod
    if Path(drun.__file__).resolve() != (module / "dense_device_runner.py").resolve() or \
            Path(prod.__file__).resolve() != (module / "dense_device_production.py").resolve():
        raise ValueError("production imports differ from genuine profile")
    policy = prod.DeviceProductionLimits(**scope["production_limits"])
    if scope["max_pool_total_bytes"] != policy.max_device_bytes:
        raise ValueError("worker pool policy differs from production policy")
    counts = {name: 0 for name in ("_advance_wind_block", "_finalize_wind_event", "_complete_event")}
    original = {name: getattr(drun._runner, name) for name in counts}
    def forbidden(name):
        def call(*args, **kwargs):
            counts[name] += 1
            raise AssertionError("transport is forbidden during construction")
        return call
    runner = None
    completed = set()
    try:
        for name in counts:
            setattr(drun._runner, name, forbidden(name))
        path_before = list(sys.path)
        try:
            host = _host_inputs(request, scope)
        finally:
            sys.path[:] = path_before
        bed, schedule, wind, water, control, limits, profile = host
        if [event.kind for event in schedule.events] != ["wind", "rain", "wind", "rain"]:
            raise AssertionError("original full four-event schedule missing")
        source_before = bed.identity_sha256()
        if time.time() > scope["expires_unix_s"] or time.monotonic() - started > scope["max_elapsed_s"]:
            raise ValueError("authority/window expired before device")
        guard.check_probe_gate(request["probe_gate"], request["profile_id"], request["device"])
        import cupy as cp  # All genuine source/process/dual-parent scopes passed. No automatic backend fallback.
        from maple.core.backend.selection import resolve_backend
        if cp.cuda.runtime.getDeviceCount() != 1:
            raise ValueError("exactly one scoped visible GPU required")
        observed = guard.check_observed_device(request["device"], cp.cuda.runtime.getDeviceProperties(0),
                                               cp.cuda.runtime.deviceGetPCIBusId(0))
        with resolve_backend("cupy", device_id=0).device_scope():
            pool = cp.get_default_memory_pool()
            previous_limit = pool.get_limit()
            if pool.total_bytes() > scope["max_pool_total_bytes"]:
                raise ValueError("existing pool exceeds construction cap")
            pool.set_limit(size=scope["max_pool_total_bytes"])
            free_before, _ = cp.cuda.runtime.memGetInfo()
            try:
                runner = drun._create_dense_production_device_runner(bed, schedule, wind=wind, water=water,
                    control=control, limits=limits, device_id=0,
                    device_limits=drun.DeviceRunnerLimits(scope["max_device_bytes"], scope["hydraulics_max_log"],
                        scope["wind_workspace_bytes"], scope["routing_workspace_bytes"]),
                    bed_limits=dbed.DeviceBedLimits(scope["max_bed_bytes"], scope["max_bed_work_bytes"],
                        scope["native_validation_scratch_bytes"], scope["bed_comparison_scratch_bytes"]),
                    resource_limits=dres.DeviceResourceLimits(scope["max_private_bytes"], scope["max_static_bytes"],
                        scope["max_resource_work_bytes"], scope["resource_comparison_scratch_bytes"]),
                    wind_profile=profile, production_limits=policy, worker_identity=prod.process_identity())
                construction = runner.construction_evidence
                if bed.identity_sha256() != source_before or construction["source_copy_verification"]["verified_device_leaves"] <= 0:
                    raise AssertionError("source changed or no genuine copy verified")
                completed.add("construction")
                if runner._next != 0 or runner._receipts or runner._hydraulics is not None or any(counts.values()):
                    raise AssertionError("construction performed native transport")
                completed.update(("zero_transport", "original_schedule"))
                availability = runner._device_resources.static("initial_available_fraction")
                if (runner._bed.context.initial_available_fraction is not availability
                        or runner._wind.config.initial_available_fraction is not availability):
                    raise AssertionError("one owned policy is not shared")
                completed.add("shared_policy")
                if (runner._wind.config.hop_bins is not wind.config.hop_bins
                        or type(wind.config.hop_bins.weight) is cp.ndarray):
                    raise AssertionError("host-at-rest hop identity lost")
                completed.add("host_hop_tables")
                runner._device_own.verify_case_inputs()
                if runner._static_parts_record(drun._runner._native()) != runner._static_parts:
                    raise AssertionError("controlling record differs after conversion")
                completed.add("controlling_binding")
                try:
                    runner._device_own.accepted_state()
                except drun.DenseRunnerError as exc:
                    if exc.code != "device_private":
                        raise
                else:
                    raise AssertionError("accepted device state escaped")
                completed.add("private_state")
                source_ref = weakref.ref(bed)
                del bed, host
                gc.collect()
                if source_ref() is not None:
                    raise AssertionError("original full CPU bed retained after construction")
                completed.add("host_bed_release")
                evidence = runner.device_evidence()
                refs = (weakref.ref(runner), weakref.ref(runner._device_bed), weakref.ref(runner._device_resources))
                runner = None
                del availability
                gc.collect()
                if any(ref() is not None for ref in refs):
                    raise AssertionError("runner/device owners retained after release")
                completed.add("release")
                cp.cuda.get_current_stream().synchronize()
                free_after, _ = cp.cuda.runtime.memGetInfo()
                memory = {"pool_used_bytes": pool.used_bytes(), "pool_total_bytes": pool.total_bytes(),
                          "sampled_driver_delta_bytes": max(0, free_before - free_after),
                          "driver_peak_proven": False, "deallocation_asserted": False}
                if memory["pool_total_bytes"] > scope["max_pool_total_bytes"] or \
                        memory["sampled_driver_delta_bytes"] > scope["max_driver_delta_bytes"] or \
                        construction["source_copy_verification"]["downloaded_bytes"] > scope["max_native_d2h_bytes"]:
                    raise AssertionError("construction resource/copy cap exceeded")
            finally:
                runner = None
                pool.set_limit(size=previous_limit)
    finally:
        for name, function in original.items():
            setattr(drun._runner, name, function)
    if completed != REQUIRED_CHECKS or any(counts.values()) or time.time() > scope["expires_unix_s"] or \
            time.monotonic() - started > scope["max_elapsed_s"]:
        raise AssertionError("missing callset, native transport, or construction window exceeded")
    _scope(request, scope, base_sys_path)  # every exact fixture/module/test ref read again; no mutation accepted
    _, binaries = guard.environment_metadata()
    compiled = guard.compiled_inventory(request["limits"], binaries)
    post_process = _post_process_check(request, base_sys_path, process)
    if time.time() > scope["expires_unix_s"] or time.monotonic() - started > scope["max_elapsed_s"]:
        raise ValueError("construction scope/window expired during post-verification")
    return {"scope": "production construction only, zero events/transport; no calibration/peak/restart qualification",
            "completed_checks": sorted(completed), "required_checks": sorted(REQUIRED_CHECKS), "skips": 0,
            "native_transport_counts": counts, "process": process, "post_process": post_process, "native_audit": native, "device": observed,
            "profile_id": request["profile_id"], "construction": construction, "device_evidence": evidence,
            "memory": memory, "compiled": compiled, "elapsed_s": time.monotonic() - started,
            "authority": "inherited reviewed parent; no independently external root authenticated by this module"}


def load_tests(loader, tests, pattern):
    raise RuntimeError("unittest discovery refuses; pytest has no executable cases; protected parent must call run_authorized_checks")
