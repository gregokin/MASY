"""Private, detached physical checkpoints of the default Dense device owner.

This is a storage conversion, not a device-runner restore or a combined
checkpoint. Source/runtime/device admission belongs to the protected caller.
Native imports are lazy; no default runner or CLI selects this component.
"""
from __future__ import annotations

import dataclasses
from collections import Counter
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any
import weakref

__all__: list[str] = []
SCHEMA = "masy.dense_device_boundary_physical.v1"
_RUNNER_FIELDS = frozenset({
    "_accepted_bed", "_accepted_private", "_active_evidence", "_bed",
    "_construction_evidence", "_control", "_cp", "_device_bed", "_device_limits",
    "_device_ops", "_device_own", "_device_resources", "_host_view", "_hydraulics",
    "_hydraulics_summary", "_inventory0", "_inventory_elements", "_ledger", "_limits",
    "_n_accepted", "_n_classes", "_next", "_pool", "_production", "_receipts",
    "_routing", "_schedule", "_soil", "_static_parts", "_static_parts_sha256",
    "_static_record", "_static_sha256", "_stop_receipt", "_totals", "_water", "_wind",
})
_CEILINGS = {
    "max_data_bytes": 16 << 20, "max_host_data_bytes": 128 << 20,
    "max_native_bytes": 32 << 20, "max_metadata_bytes": 1 << 20,
    "max_leaves": 256, "max_events": 8,
}


class DeviceCheckpointError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code, self.detail = code, detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> None:
    raise DeviceCheckpointError(code, detail)


@dataclasses.dataclass(frozen=True)
class DeviceCheckpointLimits:
    max_data_bytes: int = 16 << 20
    max_host_data_bytes: int = 128 << 20
    max_native_bytes: int = 32 << 20
    max_metadata_bytes: int = 1 << 20
    max_leaves: int = 256
    max_events: int = 8

    def __post_init__(self):
        for name, ceiling in _CEILINGS.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                _fail("invalid_limit", name)


@dataclasses.dataclass(frozen=True)
class DeviceBoundaryExport:
    physical: Any
    identity: dict
    expected_physical_sha256: str
    identity_sha256: str
    evidence: dict
    evidence_sha256: str
    config_sha256: str
    capsule_sha256: str
    _authority: object = dataclasses.field(default=None, init=False, repr=False, compare=False)


@dataclasses.dataclass(frozen=True)
class _ExportAuthority:
    owner: weakref.ReferenceType
    seals: tuple


def _seals(export):
    return (export.expected_physical_sha256, export.identity_sha256,
            export.evidence_sha256, export.config_sha256, export.capsule_sha256)


def _issue_export(*args):
    export = DeviceBoundaryExport(*args)
    object.__setattr__(export, "_authority", _ExportAuthority(weakref.ref(export), _seals(export)))
    return export


def _native():
    from masy import dense_bed as bed
    from masy import dense_checkpoint as bridge
    from masy import dense_runner as runner
    n, codec = bridge._native()
    if (codec.SCHEMA != "maple-exact-physical-checkpoint/1"
            or codec.MAX_METADATA_BYTES != _CEILINGS["max_metadata_bytes"]
            or len(codec.PAYLOADS) != 2):
        _fail("native_storage_profile", "native codec schema/metadata/payload bounds differ")
    return bed, bridge, n, codec, runner._native()


def _limits(value):
    if type(value) is not DeviceCheckpointLimits or set(vars(value)) != set(_CEILINGS):
        _fail("invalid_limit", "exact declared limits required")
    value.__post_init__()


def _plain(value, low, high, what):
    if type(value) is not int or not low <= value <= high:
        _fail("unsupported_profile", what)
    return value


def _json_seal(codec, bridge, limits, value):
    bridge._budget(codec, limits.max_metadata_bytes, value, plain_json=True)
    raw = codec._json(value).encode()
    if len(raw) > limits.max_metadata_bytes:
        _fail("metadata_bytes_cap", "canonical metadata exceeds cap")
    return hashlib.sha256(raw).hexdigest()


def _inspect(runner, config, inputs, limits):
    from masy import dense_device_runner as device
    from masy.dense_device_bed import DeviceDenseBedOwner
    from masy.dense_device_resources import DeviceDenseResources, STATIC_NAMES

    _limits(limits)
    if type(runner) is not device._DeviceDenseRunner:
        _fail("runner_layout", "exact default device runner required")
    if set(vars(runner)) != _RUNNER_FIELDS:
        _fail("runner_layout", "default 36-field device layout required")
    if runner._production is not None:
        _fail("unsupported_profile", "production export is not qualified")
    cp = device._genuine_cupy()
    if runner._cp is not cp:
        _fail("unsupported_backend", "the already-imported genuine CuPy namespace is required")
    owner, resources, own = runner._device_bed, runner._device_resources, runner._device_own
    if (type(owner) is not DeviceDenseBedOwner or type(resources) is not DeviceDenseResources
            or type(own) is not device._DeviceOwnership
            or type(runner._bed) is not device._HostMetadataBed
            or own._owner is not owner or own._res is not resources or own._metadata is not runner._bed
            or runner._bed._owner is not owner or runner._bed._resources is not resources):
        _fail("owner_layout", "owner and metadata links differ")
    if owner._ops.xp is not cp or resources._ops.xp is not cp:
        _fail("unsupported_backend", "CPU semantic owners cannot authorize device export")
    if (owner._active is not None or resources._active is not None or own._stage is not None
            or own._prepared is not None or runner._hydraulics is not None
            or runner._active_evidence is not None or runner._stop_receipt is not None):
        _fail("not_quiescent", "accepted unstopped event boundary required")
    if (type(runner._pool) is not device._PrivateRef or runner._pool is not runner._soil
            or runner._pool.generation != resources.generation):
        _fail("private_identity", "accepted private generation differs")
    g = owner._geometry
    _plain(g.nx, 1, 64, "nx"); _plain(g.ny, 1, 64, "ny")
    _plain(g.nx * g.ny, 1, 64, "cells"); _plain(owner._nv, 1, 64, "voxels")
    if owner._nc != 6 or runner._n_classes != 6:
        _fail("unsupported_profile", "six native classes required")
    _plain(runner._next, 0, limits.max_events, "event cursor")
    if (type(runner._receipts) is not list or len(runner._receipts) != runner._next
            or runner._n_accepted != runner._next
            or len(runner._schedule.events) > limits.max_events):
        _fail("cursor_identity", "accepted event counts differ")
    if tuple(resources._statics) != STATIC_NAMES or tuple(resources._static_mirrors) != STATIC_NAMES:
        _fail("owner_layout", "unknown static fields")
    if resources._statics["vegetation"].objects is not None:
        _fail("unsupported_profile", "this projection supports native raster vegetation only")
    owner._scope(); resources._scope()
    bed, bridge, n, codec, transport = _native()
    bridge._validate_inputs(inputs, n, codec, limits.max_host_data_bytes)
    bridge._budget(codec, limits.max_metadata_bytes, runner._bed.case_record,
                   runner._static_record, plain_json=True)
    if (runner._bed.host_context is not inputs.context
            or runner._bed.case_sha256 != inputs.case_sha256
            or runner._static_record["case_sha256"] != inputs.case_sha256
            or bed._sha256(runner._bed.case_record) != inputs.case_sha256
            or codec._fingerprint(runner._bed.case_record) != codec._fingerprint(inputs.case_record)):
        _fail("source_identity", "original metadata/capsule differs")
    if (config is not runner._host_view._wind.config or config is runner._wind.config
            or type(config) is not transport.SchedulerPhysicsConfig or config.execution_backend != "numpy"):
        _fail("storage_config", "unchanged original host storage config required")
    _plain(config.particle_transport.max_active_cohorts, 1, 16000, "cohorts")
    if config.particle_transport.mobile_state_representation != "flat":
        _fail("unsupported_profile", "flat cohorts required")
    if bed._sha256(runner._static_record) != runner._static_sha256:
        _fail("static_identity", "original static record changed")
    return bed, bridge, n, codec, transport, owner, resources


def _stamp(runner, codec, bridge, limits):
    owner, resources = runner._device_bed, runner._device_resources
    clocks = runner._ledger.as_record()
    bridge._budget(codec, limits.max_host_data_bytes, clocks, runner._receipts,
                   runner._totals, runner._inventory0)
    return (
        tuple(id(x) for x in (owner._accepted, owner._accepted.state, owner._accepted.mirror,
              resources._accepted, resources._accepted.pool, resources._accepted.pool_mirror,
              resources._accepted.soil, resources._accepted.soil_mirror, runner._pool, runner._soil,
              runner._routing, runner._ledger, runner._control, runner._limits, runner._inventory0)),
        owner.generation, resources.generation, owner.time_s.hex(), runner._next, runner._n_accepted,
        tuple(id(x) for x in runner._receipts), runner._accepted_bed, runner._accepted_private,
        codec._fingerprint(clocks), codec._fingerprint(runner._receipts), codec._fingerprint(runner._totals),
        codec._fingerprint(runner._inventory0),
    )


def _array_plan(sources, array_type, np, limits):
    from maple.core.backend import array_leaves
    from maple.core.backend.arrays import array_leaf_inventory

    arrays = [a for tree in sources for a in array_leaves(tree)]
    if len(arrays) > limits.max_leaves or len({id(a) for a in arrays}) != len(arrays):
        _fail("array_layout", "too many or repeated source leaves")
    numeric, largest, device_bytes, device_arrays = 0, 0, 0, 0
    for a in arrays:
        if type(a) not in (array_type, np.ndarray):
            _fail("array_layout", "unknown source namespace")
        if (not a.flags.c_contiguous or a.dtype.hasobject or a.dtype.kind not in "fiub"
                or not 1 <= a.ndim <= 4):
            _fail("array_layout", "unsupported source dtype/layout")
        numeric += int(a.nbytes); largest = max(largest, int(a.nbytes))
        if type(a) is not np.ndarray:
            device_bytes += int(a.nbytes)
            device_arrays += 1
    if numeric > limits.max_data_bytes:
        _fail("data_bytes_cap", "physical projection exceeds cap")
    inventories = [array_leaf_inventory(tree) for tree in sources]
    if (sum(i["n_leaves"] for i in inventories) != len(arrays)
            or any(i["n_aliased_leaves"] for i in inventories)
            or sum(i["transfer_bytes"] for i in inventories) != numeric):
        _fail("array_layout", "planned leaves differ from native copy traversal")
    return {"numeric_bytes": numeric, "largest_leaf_bytes": largest, "leaves": len(arrays),
            "device_bytes": device_bytes, "device_arrays": device_arrays}


def _reserve(layout, static, limits):
    # Conservative simultaneous DATA allowance. CPython/RSS, JIT and device
    # allocator peaks remain the protected caller's independent responsibility.
    terms = {"seven_payload_lifetimes": 7 * layout["numeric_bytes"],
             "leaf_copy_compare_hash_scratch": 4 * layout["largest_leaf_bytes"],
             "native_raw_read_and_write_allowance": 2 * limits.max_native_bytes,
             "numpy_zip_iterator_buffers": 32 << 20,
             "native_availability_fallback_and_digest": 2 * 64 * 6 * 8,
             "metadata_lifetimes": 8 * _CEILINGS["max_metadata_bytes"],
             "static_inputs_bound": static}
    required = sum(terms.values())
    if required > limits.max_host_data_bytes:
        _fail("host_bytes_cap", "simultaneous data reservation exceeds cap")
    return {**layout, "terms": terms, "required_host_data_bytes": required,
            "kind": "conservative_data_reservation_not_RSS_or_device_fit"}


def _plan(sources, owner, bridge, codec, limits, inputs, config):
    layout = _array_plan(sources, owner._ops.array_type, owner._ops.host_np, limits)
    static = bridge._budget(codec, limits.max_host_data_bytes, inputs, config)
    return _reserve(layout, static, limits)


def _check_copies(reservation, copies, delta):
    if (copies["device_arrays"] != reservation["device_arrays"]
            or copies["device_bytes"] != reservation["device_bytes"]
            or copies["device_arrays"] + copies["host_arrays"] != reservation["leaves"]
            or copies["device_bytes"] + copies["host_bytes"] != reservation["numeric_bytes"]
            or delta["device_to_host"] != copies["device_arrays"]
            or delta["device_to_host_bytes"] != copies["device_bytes"]
            or delta["host_to_device"] != 0 or delta["host_to_device_bytes"] != 0):
        _fail("copy_accounting", "native transfer/copy counts differ from admitted leaves")


def _copy_tree(tree, xp, np, copies):
    from maple.core.backend import to_host
    from maple.core.backend.arrays import _map_tree

    def copy(a):
        if type(a) is np.ndarray:
            out = np.array(a, copy=True, order="C")
            copies["host_arrays"] += 1; copies["host_bytes"] += int(a.nbytes)
        elif type(a) is xp.ndarray:
            out = to_host(a)
            copies["device_arrays"] += 1; copies["device_bytes"] += int(a.nbytes)
        else:
            _fail("array_layout", "unknown leaf during copy")
        if type(out) is not np.ndarray or out.dtype != a.dtype or out.shape != a.shape:
            _fail("copy_mismatch", "lossless host conversion failed")
        out.setflags(write=False)
        return out
    return _map_tree(tree, copy)


def _verify_mirror(host, mirror, xp, np):
    from maple.core.backend import array_leaves, to_host

    left, right = list(array_leaves(host)), list(array_leaves(mirror))
    if len(left) != len(right):
        _fail("copy_mismatch", "mirror leaf count differs")
    for a, b in zip(left, right, strict=True):
        if (type(a) is not np.ndarray or type(b) not in (xp.ndarray, np.ndarray)
                or a.dtype != b.dtype or a.shape != b.shape or not b.flags.c_contiguous):
            _fail("copy_mismatch", "mirror leaf layout differs before readback")
        back = to_host(b) if type(b) is xp.ndarray else b
        if (type(back) is not np.ndarray or a.dtype != back.dtype or a.shape != back.shape
                or not np.array_equal(a.reshape(-1).view(np.uint8), back.reshape(-1).view(np.uint8))):
            _fail("copy_mismatch", "mirror raw bytes differ")
        back = None


def export_device_boundary_physical(runner, config, caller, *, inputs, limits):
    bed, bridge, n, codec, transport, owner, resources = _inspect(runner, config, inputs, limits)
    from maple.core.backend import read_transfer_counters

    bridge._budget(codec, limits.max_metadata_bytes, caller, plain_json=True)
    source = (owner._accepted.state, resources._accepted.pool,
              resources._statics["vegetation"], resources._statics["threshold_modifier_state"])
    mirrors = (owner._accepted.mirror, resources._accepted.pool_mirror,
               resources._static_mirrors["vegetation"], resources._static_mirrors["threshold_modifier_state"])
    owner._describe(source[0]); resources._metadata(source[1], "pool")
    for name in ("vegetation", "threshold_modifier_state"):
        resources._metadata(resources._statics[name], name, allow_host=True)
    reservation = _plan(source, owner, bridge, codec, limits, inputs, config)
    deltas = {}
    def phase(name, fn):
        before = read_transfer_counters()
        value = fn()
        deltas[name] = dataclasses.asdict(read_transfer_counters().delta(before))
        return value
    start = read_transfer_counters()
    phase("precheck", runner._precheck)
    stamp = _stamp(runner, codec, bridge, limits)
    config_sha = codec._fingerprint(config)
    copies = dict(device_arrays=0, device_bytes=0, host_arrays=0, host_bytes=0)
    projected = phase("payload_copy", lambda: tuple(_copy_tree(x, runner._cp, n.np, copies) for x in source))
    _check_copies(reservation, copies, deltas["payload_copy"])
    phase("mirror_readback", lambda: [_verify_mirror(a, b, runner._cp, n.np)
                                      for a, b in zip(projected, mirrors, strict=True)])
    if (deltas["mirror_readback"]["host_to_device"] != 0
            or deltas["mirror_readback"]["host_to_device_bytes"] != 0):
        _fail("copy_accounting", "unexpected mirror upload")
    state, pool, vegetation, threshold = projected
    actual_prescribed = {
        "vegetation": bed._sha256(bed._canonical_static(vegetation, "vegetation")),
        "native_vegetation_content_sha256": n.vegetation_content_hash(vegetation),
        "threshold_modifier_state": bed._sha256(bed._canonical_static(threshold, "threshold_modifier_state")),
    }
    if actual_prescribed != inputs.prescribed_digests:
        _fail("prescribed_identity", "accepted native statics differ from original capsule")
    if not phase("empty_pool_check", lambda: transport.mobile_state_is_genuinely_empty(pool)):
        _fail("mobile_not_empty", "native wind material remains")
    physical = n.SchedulerPhysicalState(
        **{name: getattr(state, name) for name in bed.SHARED_FIELDS}, mobile_state=pool,
        vegetation=vegetation, threshold_modifier_state=threshold, time_s=owner.time_s,
        coupled_trajectory_semantics=None, ledger_reconciliation_baseline=None)
    bridge._validate_physical(physical, config, inputs, n, codec, limits.max_host_data_bytes)
    bed._check_commit_stamp(state, owner.time_s)
    digest = codec._fingerprint(physical)
    phase("postcheck", runner._precheck)
    if stamp != _stamp(runner, codec, bridge, limits) or config_sha != codec._fingerprint(config):
        _fail("source_changed", "accepted source changed during export")
    bridge._validate_inputs(inputs, n, codec, limits.max_host_data_bytes)
    identity = {"schema": SCHEMA, "caller": codec._identity(caller),
                "capsule_sha256": inputs.binding_sha256, "case_sha256": inputs.case_sha256,
                "static_sha256": runner._static_sha256, "config_sha256": config_sha,
                "physical_sha256": digest, "native_time_hex": owner.time_s.hex(),
                "next_event": runner._next, "bed_generation": owner.generation,
                "resource_generation": resources.generation,
                "limits": dataclasses.asdict(limits),
                "scope": "physical_only;soil_and_routing_payloads_excluded;no_restore_authority"}
    codec._identity(identity)
    identity_sha = _json_seal(codec, bridge, limits, identity)
    evidence = {"reservation": reservation, "copies": copies, "transfer_categories": deltas,
                "total_transfer_delta": dataclasses.asdict(read_transfer_counters().delta(start)),
                "source_unchanged": True, "routing_retained": runner._routing is not None,
                "clock_ledger": runner._ledger.as_record(), "gpu_qualified": False,
                "combined_restart": False, "device_restore": False}
    evidence_sha = _json_seal(codec, bridge, limits, evidence)
    return _issue_export(physical, identity, digest, identity_sha, evidence, evidence_sha, config_sha,
                         inputs.binding_sha256)


def _validate_export(export, config, inputs, limits):
    _limits(limits)
    if (type(export) is not DeviceBoundaryExport or type(export._authority) is not _ExportAuthority
            or export._authority.owner() is not export or export._authority.seals != _seals(export)
            or type(export.identity) is not dict or type(export.evidence) is not dict):
        _fail("export_ticket", "immediate private export required")
    bed, bridge, n, codec, _ = _native()
    bridge._validate_inputs(inputs, n, codec, limits.max_host_data_bytes)
    static = bridge._budget(codec, limits.max_host_data_bytes, inputs, config)
    if (inputs.binding_sha256 != export.capsule_sha256
            or _json_seal(codec, bridge, limits, export.identity) != export.identity_sha256
            or export.identity.get("schema") != SCHEMA
            or export.identity.get("physical_sha256") != export.expected_physical_sha256
            or export.identity.get("config_sha256") != export.config_sha256
            or export.identity.get("capsule_sha256") != export.capsule_sha256
            or export.identity.get("case_sha256") != inputs.case_sha256
            or export.identity.get("limits") != dataclasses.asdict(limits)
            or _json_seal(codec, bridge, limits, export.evidence) != export.evidence_sha256
            or codec._fingerprint(config) != export.config_sha256):
        _fail("export_identity", "capsule/config/independent identity differs")
    bridge._budget(codec, limits.max_host_data_bytes, export.physical, require_readonly=True)
    if (type(export.physical) is not n.SchedulerPhysicalState
            or export.physical.vegetation is None or export.physical.vegetation.objects is not None):
        _fail("unsupported_profile", "exact raster-vegetation physical carrier required")
    if codec._fingerprint(export.physical) != export.expected_physical_sha256:
        _fail("physical_identity", "detached physical bytes changed")
    bridge._validate_physical(export.physical, config, inputs, n, codec, limits.max_host_data_bytes)
    codec._identity(export.identity)
    actual = _array_plan((export.physical,), n.np.ndarray, n.np, limits)
    reservation = export.evidence.get("reservation")
    if type(reservation) is not dict:
        _fail("host_bytes_cap", "missing export reservation")
    layout = {key: reservation.get(key) for key in actual}
    if (any(type(value) is not int or value < 0 for value in layout.values())
            or any(actual[key] != layout[key] for key in ("numeric_bytes", "largest_leaf_bytes", "leaves"))
            or layout["device_bytes"] > layout["numeric_bytes"]
            or layout["device_arrays"] > layout["leaves"]):
        _fail("host_bytes_cap", "detached physical differs from planned data")
    if _reserve(layout, static, limits) != reservation:
        _fail("host_bytes_cap", "reservation differs under current limits")
    return bridge, n, codec


def _readback_layout(path, export, codec, np, limits):
    """Audit a caller-private quiescent artifact before native allocation.

    The protected caller owns the artifact's exclusion from concurrent writers
    across both this preflight and native loading. This is not a file-lock API.
    """
    root = Path(path).absolute()
    if root.is_symlink() or not root.is_dir():
        _fail("readback_layout", "checkpoint directory missing or symlink")
    for name in codec.PAYLOADS:
        payload = root / name
        if payload.is_symlink() or not payload.is_file():
            _fail("readback_layout", "native payload is not a regular non-symlink file")
    codec._serialized_budget(root, limits.max_native_bytes)
    arrays = list(codec._arrays(export.physical))
    reservation = export.evidence["reservation"]
    if (len(arrays) != reservation["leaves"]
            or sum(int(a.nbytes) for a in arrays) != reservation["numeric_bytes"]
            or max((int(a.nbytes) for a in arrays), default=0) != reservation["largest_leaf_bytes"]):
        _fail("readback_layout", "native serialization walk differs from admitted leaves")
    expected = Counter((a.dtype.str, tuple(a.shape)) for a in arrays)
    expected_names = {codec.PAYLOADS[0]: codec._expected_base(export.physical),
                      codec.PAYLOADS[1]: set(codec._pending(export.physical))}
    actual, used, scalar_bytes = Counter(), 0, 0
    for name in codec.PAYLOADS:
        members, size = codec._audit_archive(root / name, limits.max_native_bytes - used)
        used += size
        if set(members) != expected_names[name]:
            _fail("readback_layout", "native field inventory differs from admitted physical state")
        for spec in members.values():
            shape = tuple(spec["shape"])
            dtype = np.dtype(spec["dtype"])
            if shape:
                actual[(dtype.str, shape)] += 1
            else:
                scalar_bytes += dtype.itemsize
    if actual != expected:
        _fail("readback_layout", "native array headers differ from admitted physical layout")
    if scalar_bytes > limits.max_metadata_bytes:
        _fail("readback_layout", "native scalar payload bytes exceed metadata cap")


def save_device_boundary_physical(path, export, config, *, inputs, limits):
    _, _, codec = _validate_export(export, config, inputs, limits)
    target = Path(path).absolute()
    usage = shutil.disk_usage(target.parent)
    fs = os.statvfs(target.parent)
    disk = (limits.max_native_bytes + len(codec.PAYLOADS) * codec._ZIP_METADATA_SLACK
            + 2 * codec.MAX_METADATA_BYTES + 8 * fs.f_bsize)
    if usage.free < disk or fs.f_favail < 8:
        _fail("disk_admission", "checkpoint bundle reserve does not fit")
    return codec.save_exact_physical_checkpoint(target, export.physical, config, export.identity,
                                                max_bytes=limits.max_native_bytes)


def verify_device_boundary_roundtrip(path, export, config, *, inputs, limits):
    bridge, n, codec = _validate_export(export, config, inputs, limits)
    _readback_layout(path, export, codec, n.np, limits)
    physical = codec.load_exact_physical_checkpoint(path, config, export.identity,
                                                    max_bytes=limits.max_native_bytes)
    bridge._budget(codec, limits.max_host_data_bytes, physical, export.physical,
                   require_readonly=True)
    bridge._budget(codec, limits.max_host_data_bytes, physical, export.physical, inputs, config)
    bridge._validate_physical(physical, config, inputs, n, codec, limits.max_host_data_bytes)
    if (codec._fingerprint(physical) != export.expected_physical_sha256
            or physical.time_s.hex() != export.identity["native_time_hex"]):
        _fail("physical_identity", "native readback differs from independent export seal")
    _validate_export(export, config, inputs, limits)
    return physical
