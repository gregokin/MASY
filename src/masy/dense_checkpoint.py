"""Source-bound Dense physical bridge; no runner or publication authority.

The caller saves the returned native carrier with MAPLE's exact codec and
retains the original typed input capsule separately. This is not a portable
static-input decoder, combined checkpoint, or active-process continuation.
Native imports are lazy. Private MAPLE15 validation/fingerprinting helpers
are deliberate dependencies of this separately source-qualified prototype.
"""
from __future__ import annotations

import copy
import dataclasses
import math
from enum import Enum
from typing import Any

from masy import dense_bed as _bed
from masy.clock import _native_clock, decode_time, encode_time

SCHEMA = "masy.dense_boundary_physical.v1"
_CASE = frozenset({"schema", "backend", "dtype", "representation", "geometry", "grain_classes",
    "class_order", "mass_resolution_kg", "n_voxels", "initial_time_s", "initial_bed_elevation_m",
    "initial_class_fractions", "initial_available_fraction", "initial_availability_policy",
    "initial_water_depth_m", "active_mask", "vegetation", "native_vegetation_content_sha256",
    "threshold_modifier_state", "topographic_commit_spec", "avalanching_spec", "depth_update_rule"})
_INITIAL = frozenset({"native_sequence", "requested_by_class_kg", "deposit_numerical_residual_total_kg",
    "init_numerical_residual_total_kg", "bed_inventory_by_class_kg", "host_memory_lower_bound_bytes",
    "estimated_initialization_bytes", "measured_per_cell_ledger_committed_bytes", "limits"})
_HELD = ("initial_bed_elevation_m", "initial_class_fractions", "initial_available_fraction", "initial_water_depth_m")
_PRESCRIBED = ("vegetation", "native_vegetation_content_sha256", "threshold_modifier_state")
_CONTEXT = ("geometry", "grain_classes", "mass_resolution_kg", "topographic_commit_spec", "avalanching_spec",
    "initial_available_fraction", "depth_update_rule", "record_exchange_history", "window_max_journal_records",
    "deep_chunk_encoding", "deep_overlay_max_bytes")
_INPUTS = ("case_record", "held_inputs", "context", "limits", "initialization_receipt", "prescribed_digests",
           "case_sha256", "binding_sha256")
_IDENTITY = ("schema", "caller", "original_inputs_sha256", "case_sha256", "bed_identity_sha256")
_BED_INSTANCE = {"_case_record", "_case_sha256", "_held_inputs", "_context", "_state", "_vegetation",
    "_threshold_modifier_state", "_time_s", "_limits", "_initialization_receipt", "_prescribed_digests",
    "_context_refs", "_owner"}
_NATIVE_LOAD_TOKEN = object()


def _native():
    from maple.aeolian.scheduler import physical_checkpoint
    return _bed._native(), physical_checkpoint


def _fail(message):
    raise _bed.DenseBedError("invalid_boundary_physical", message)


def _keys(value, names, where):
    if type(value) is not dict or len(value) != len(names) or set(value) != set(names):
        _fail(f"{where}: unknown or missing fields")


def _budget(cp, cap, *values, plain_json=False, require_readonly=False):
    """Bound nested arrays and metadata before copying, hashing or decoding.

    Aliases count each time they occur. This is a conservative data budget,
    not a claim about Python/NumPy peak RSS or a second archive parser.
    """
    cp._bound(cap)
    used, metadata, nodes = 131072, 0, 0

    def charge(size, *, data=False):
        nonlocal used, metadata
        used += size
        if not data:
            metadata += size
        if used > cap or metadata > cp.MAX_METADATA_BYTES:
            _fail("physical/static data or metadata preflight exceeds bound")

    def text(value):
        # Worst-case ASCII JSON escaping, including astral Unicode pairs.
        charge(12 * len(value) + 2)

    def visit(value, depth=0):
        nonlocal nodes
        nodes += 1
        if depth > 24 or nodes > 100_000:
            _fail("physical/static tree exceeds depth/node bound")
        if type(value) is dict:
            if len(value) > 100_000 - nodes:
                _fail("physical/static mapping exceeds node bound")
            charge(2 + 2 * len(value))
            for key, item in value.items():
                if type(key) is not str:
                    _fail("physical/static mapping keys must be strings")
                text(key)
                visit(item, depth + 1)
        elif type(value) is list or (not plain_json and type(value) is tuple):
            if len(value) > 100_000 - nodes:
                _fail("physical/static sequence exceeds node bound")
            charge(2 + len(value))
            for item in value:
                visit(item, depth + 1)
        elif not plain_json and dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields = cp._instance(value)
            charge(2 + 2 * len(fields))
            for name, item in fields.items():
                text(name)
                visit(item, depth + 1)
        elif not plain_json and isinstance(value, Enum):
            text(type(value).__module__ + "." + type(value).__qualname__)
            visit(value.value, depth + 1)
        elif not plain_json and type(value) is cp.np.ndarray:
            # Reuse native layout/type checks without reading array contents.
            for array in cp._arrays(value):
                if require_readonly and array.flags.writeable:
                    _fail("native loader returned a writeable physical array")
                charge(array.nbytes, data=True)
                charge(256 + 16 * array.ndim)
        elif value is None or type(value) is bool:
            charge(5)
        elif type(value) is str:
            text(value)
        elif type(value) is int:
            # Avoid converting an unbounded integer to a decimal string.
            charge(value.bit_length() + 3)
        elif type(value) is float and math.isfinite(value):
            charge(32)
        else:
            _fail("unsupported physical/static metadata or array type")

    for value in values:
        visit(value)
    if used > cap:
        _fail("metadata allowance exceeds max_bytes")
    return used


def _original_metadata(cp, cap, case, receipt, prescribed):
    _budget(cp, cap, case, receipt, prescribed, plain_json=True)


@dataclasses.dataclass(frozen=True)
class DenseBoundaryInputs:
    """Original static inputs; a binding digest never replaces revalidation.

    Arrays/context may be shared with the original immutable case. Callers
    must keep them quiescent; every bridge operation checks their contents.
    Construction from original artifact objects is supported by the explicit
    fields; capture is convenience for a bed already initialized once.
    """
    case_record: dict
    held_inputs: dict
    context: Any
    limits: _bed.DenseBedLimits
    initialization_receipt: dict
    prescribed_digests: dict
    case_sha256: str
    binding_sha256: str

    @classmethod
    def capture(cls, bed):
        if cls is not DenseBoundaryInputs:
            _fail("unsupported capsule subclass")
        n, cp = _native()
        _check_bed(bed)
        cp._fields(bed.limits, _bed.DenseBedLimits,
                   ("host_memory_cap_bytes", "max_voxels_per_column", "stage_copy_max_bytes"))
        cap = bed.limits.host_memory_cap_bytes
        _original_metadata(cp, cap, bed._case_record, bed._initialization_receipt, bed._prescribed_digests)
        _budget(cp, cap, bed.context, bed._held_inputs, bed._case_record,
                bed._initialization_receipt, bed._prescribed_digests, bed.limits)
        bed.verify_case_inputs()
        candidate = cls(copy.deepcopy(bed._case_record), dict(bed._held_inputs), bed.context, bed.limits,
                        copy.deepcopy(bed._initialization_receipt), dict(bed._prescribed_digests),
                        bed.case_sha256, "")
        candidate = dataclasses.replace(candidate, binding_sha256=cp._fingerprint(_material(candidate)))
        _validate_inputs(candidate, n, cp, bed.limits.host_memory_cap_bytes)
        return candidate


@dataclasses.dataclass(frozen=True)
class PackedBoundary:
    physical: Any
    identity: dict


@dataclasses.dataclass(frozen=True)
class RestoredBoundary:
    bed: _bed.DenseBed
    mobile_state: Any


@dataclasses.dataclass(frozen=True)
class _NativePhysicalLoad:
    """Immediate internal loader handoff; never published or returned.

    Nominal Python ownership protocol, not security or provenance proof.
    """
    physical: Any
    _authority: object


def _material(inputs):
    return {name: getattr(inputs, name) for name in _INPUTS if name != "binding_sha256"}


def _check_bed(bed):
    if (type(bed) is not _bed.DenseBed or len(vars(bed)) != len(_BED_INSTANCE)
            or set(vars(bed)) != _BED_INSTANCE):
        _fail("unsupported DenseBed type/instance layout")


def _validate_inputs(inputs, n, cp, cap):
    cp._bound(cap)
    cp._fields(inputs, DenseBoundaryInputs, _INPUTS)
    _keys(inputs.case_record, _CASE, "original case")
    _keys(inputs.held_inputs, _HELD, "held original inputs")
    _keys(inputs.initialization_receipt, _INITIAL, "original initialization receipt")
    _keys(inputs.prescribed_digests, _PRESCRIBED, "prescribed identities")
    cp._fields(inputs.context, n.BedContext, _CONTEXT)
    cp._fields(inputs.limits, _bed.DenseBedLimits,
               ("host_memory_cap_bytes", "max_voxels_per_column", "stage_copy_max_bytes"))
    _original_metadata(cp, cap, inputs.case_record, inputs.initialization_receipt, inputs.prescribed_digests)
    _budget(cp, cap, inputs)
    inputs.limits.__post_init__()
    r, ctx, np = inputs.case_record, inputs.context, n.np
    for name, cls in (("geometry", n.GeometrySpec), ("grain_classes", n.GrainClassSet),
                      ("topographic_commit_spec", n.TopographicCommitSpec), ("avalanching_spec", n.AvalanchingSpec)):
        if type(getattr(ctx, name)) is not cls:
            _fail(f"unsupported native original context type: {name}")
    n.validate_grain_classes(ctx.grain_classes)
    n.validate_geometry(ctx.geometry, {c.class_id for c in ctx.grain_classes.classes})
    n.validate_mass_resolution_kg(ctx.mass_resolution_kg)
    if ctx.geometry.dx_m != ctx.geometry.dy_m or ctx.geometry.voxel_growth != n.DISABLED_VOXEL_GROWTH:
        _fail("unsupported original geometry/growth policy")
    if (r["schema"] != _bed.DENSE_CASE_SCHEMA or r["backend"] != "numpy" or r["dtype"] != "float64"
            or r["representation"] != "dense_voxel_column" or r["active_mask"] != "full"
            or r["initial_availability_policy"] != _bed.INITIAL_AVAILABILITY_POLICY
            or r["depth_update_rule"] != "constant_depth"):
        _fail("unsupported original Dense profile")
    if (ctx.depth_update_rule != "constant_depth" or ctx.record_exchange_history is not False
            or ctx.window_max_journal_records is not None or ctx.deep_chunk_encoding != "full"
            or ctx.deep_overlay_max_bytes is not None):
        _fail("unsupported original context options")
    live = {name: _bed._canonical_static(getattr(ctx, name), name) for name in
            ("geometry", "grain_classes", "topographic_commit_spec", "avalanching_spec")}
    live.update(class_order=[c.class_id for c in ctx.grain_classes.classes],
                mass_resolution_kg=encode_time(ctx.mass_resolution_kg))
    if any(r[name] != value for name, value in live.items()):
        _fail("original context differs from original case record")
    if inputs.prescribed_digests != {
        "vegetation": _bed._sha256(r["vegetation"]),
        "native_vegetation_content_sha256": r["native_vegetation_content_sha256"],
        "threshold_modifier_state": _bed._sha256(r["threshold_modifier_state"]),
    }:
        _fail("prescribed identities differ from original case")
    if type(inputs.case_sha256) is not str or _bed._sha256(r) != inputs.case_sha256:
        _fail("original case identity differs")
    if cp._fingerprint(_material(inputs)) != inputs.binding_sha256:
        _fail("original input capsule content changed")
    g, nc = ctx.geometry, len(ctx.grain_classes.classes)
    for name, array in (*inputs.held_inputs.items(),
                        ("context.initial_available_fraction", ctx.initial_available_fraction)):
        key = "initial_available_fraction" if name.startswith("context.") else name
        shape = (g.ny, g.nx, nc) if key in ("initial_class_fractions", "initial_available_fraction") else (g.ny, g.nx)
        if (type(array) is not np.ndarray or array.dtype != np.float64 or array.shape != shape
                or not array.flags.c_contiguous or array.flags.writeable
                or not bool(np.isfinite(array).all()) or _bed._array_digest(array) != r[key]):
            _fail(f"original {name} values/layout/identity changed")
    if inputs.initialization_receipt["limits"] != inputs.limits.as_record():
        _fail("original initialization limits differ")
    decode_time(r["initial_time_s"])
    return inputs.binding_sha256


def _validate_identity(identity, inputs, cp):
    _budget(cp, cp.MAX_METADATA_BYTES, identity, plain_json=True)
    identity = cp._identity(identity)
    _keys(identity, _IDENTITY, "boundary identity")
    if (identity["schema"] != SCHEMA or identity["original_inputs_sha256"] != inputs.binding_sha256
            or identity["case_sha256"] != inputs.case_sha256):
        _fail("boundary identity/original inputs mismatch")
    cp._identity(identity["caller"])
    if type(identity["bed_identity_sha256"]) is not str:
        _fail("invalid evolving bed identity")
    return identity


def _validate_physical(physical, config, inputs, n, cp, cap):
    _budget(cp, cap, physical, inputs, config)
    cp._validate(physical, config)
    if any(getattr(physical, name) is None for name in (*_bed.SHARED_FIELDS, "vegetation", "threshold_modifier_state")):
        _fail("missing required shared/prescribed state")
    if physical.ledger_reconciliation_baseline is not None or physical.coupled_trajectory_semantics is not None:
        _fail("boundary profile has no accounting baseline or active trajectory continuation")
    if (n.carries_coupled_trajectory_history(physical, None)
            or bool(n.np.any(physical.water.mobile_mass_by_cell_class_kg != 0.0))):
        _fail("unresolved wind/water mobile load")
    _native_clock(physical.time_s, "boundary time")
    if physical.time_s < decode_time(inputs.case_record["initial_time_s"]):
        _fail("boundary time precedes original epoch")
    for name in ("geometry", "grain_classes", "mass_resolution_kg", "topographic_commit_spec", "avalanching_spec"):
        if cp._fingerprint(getattr(config, name)) != cp._fingerprint(getattr(inputs.context, name)):
            _fail(f"native configuration/context mismatch: {name}")
    if _bed._array_digest(config.resolved_initial_available_fraction()) != _bed._array_digest(
            inputs.context.initial_available_fraction):
        _fail("native configuration availability policy differs")


def pack_boundary_physical(bed, mobile_state, config, identity, *, inputs, max_bytes):
    """Return native carrier + bound identity; caller uses MAPLE15's saver."""
    if type(max_bytes) is not int or max_bytes < 1:
        _fail("max_bytes must be a positive plain int")
    n, cp = _native()
    _validate_inputs(inputs, n, cp, max_bytes)
    _check_bed(bed)
    _keys(bed._held_inputs, _HELD, "source bed held original inputs")
    cp._fields(bed._context, n.BedContext, _CONTEXT)
    cp._fields(bed._limits, _bed.DenseBedLimits,
               ("host_memory_cap_bytes", "max_voxels_per_column", "stage_copy_max_bytes"))
    _original_metadata(cp, max_bytes, bed._case_record, bed._initialization_receipt, bed._prescribed_digests)
    _budget(cp, max_bytes, bed.state, mobile_state, bed.vegetation, bed.threshold_modifier_state, config,
            bed._context, bed._held_inputs, bed._limits)
    if (type(bed._context.initial_available_fraction) is not n.np.ndarray
            or bed._context.initial_available_fraction.flags.writeable):
        _fail("source context availability must remain an original readonly array")
    if any(type(value) is not n.np.ndarray for value in bed._held_inputs.values()):
        _fail("source held original inputs must be native NumPy arrays")
    bed.verify_case_inputs()
    if (bed.case_sha256 != inputs.case_sha256 or cp._fingerprint(bed.context) != cp._fingerprint(inputs.context)
            or cp._fingerprint(bed._held_inputs) != cp._fingerprint(inputs.held_inputs)
            or cp._fingerprint(bed._initialization_receipt) != cp._fingerprint(inputs.initialization_receipt)
            or cp._fingerprint(bed._limits) != cp._fingerprint(inputs.limits)
            or cp._fingerprint(bed._prescribed_digests) != cp._fingerprint(inputs.prescribed_digests)):
        _fail("bed does not belong to original input capsule")
    physical = n.SchedulerPhysicalState(**{name: getattr(bed.state, name) for name in _bed.SHARED_FIELDS},
        mobile_state=mobile_state, vegetation=bed.vegetation, threshold_modifier_state=bed.threshold_modifier_state,
        time_s=bed.time_s, coupled_trajectory_semantics=None, ledger_reconciliation_baseline=None)
    _validate_physical(physical, config, inputs, n, cp, max_bytes)
    bed._validate_shared(n, bed.state)
    _bed._check_commit_stamp(bed.state, bed.time_s)
    _budget(cp, max_bytes, identity, plain_json=True)
    bound = {"schema": SCHEMA, "caller": cp._identity(identity), "original_inputs_sha256": inputs.binding_sha256,
             "case_sha256": bed.case_sha256, "bed_identity_sha256": bed.identity_sha256()}
    # Pack never creates a second owner over this bed's writable arrays.
    _validate_identity(bound, inputs, cp)
    _budget(cp, max_bytes, physical, inputs, config, bound)
    _validate_inputs(inputs, n, cp, max_bytes)
    return PackedBoundary(physical, bound)


def _restore_dense_bed(cls, loaded, config, *, inputs, expected_identity, cap):
    if cls is not _bed.DenseBed:
        _fail("unsupported DenseBed subclass")
    n, cp = _native()
    cp._fields(loaded, _NativePhysicalLoad, ("physical", "_authority"))
    if loaded._authority is not _NATIVE_LOAD_TOKEN:
        _fail("restore requires the private immediate native-loader handoff")
    physical = loaded.physical
    _validate_inputs(inputs, n, cp, cap)
    identity = _validate_identity(expected_identity, inputs, cp)
    _validate_physical(physical, config, inputs, n, cp, cap)
    _budget(cp, cap, physical, inputs, config, identity)
    _budget(cp, cap, physical, require_readonly=True)
    state = n.BedState(**{name: getattr(physical, name) for name in _bed.SHARED_FIELDS})
    bed = cls(_bed._TOKEN, case_record=copy.deepcopy(inputs.case_record), held_inputs=dict(inputs.held_inputs),
              context=inputs.context, state=state, vegetation=physical.vegetation,
              threshold_modifier_state=physical.threshold_modifier_state, time_s=physical.time_s,
              limits=inputs.limits, initialization_receipt=copy.deepcopy(inputs.initialization_receipt),
              prescribed_digests=dict(inputs.prescribed_digests))
    bed.verify_case_inputs()
    bed._validate_shared(n, state)
    _bed._check_commit_stamp(state, physical.time_s)
    if bed.identity_sha256() != identity["bed_identity_sha256"]:
        _fail("loaded native fields do not reproduce evolving bed identity")
    _validate_inputs(inputs, n, cp, cap)
    return bed


def load_boundary_physical(path, config, expected_identity, *, inputs, max_bytes):
    """Use actual native loader, validate privately, return bed + empty pool."""
    if type(max_bytes) is not int or max_bytes < 1:
        _fail("max_bytes must be a positive plain int")
    n, cp = _native()
    _validate_inputs(inputs, n, cp, max_bytes)
    identity = _validate_identity(expected_identity, inputs, cp)
    static_bytes = _budget(cp, max_bytes, inputs, config, identity)
    remaining = max_bytes - static_bytes
    if remaining < 1:
        _fail("no physical payload capacity remains after static reservation")
    physical = cp.load_exact_physical_checkpoint(path, config, identity, max_bytes=remaining)
    # Only this immediate actual-loader return can enter the private constructor.
    # No caller-visible raw-carrier constructor, registry or boolean bypass.
    loaded = _NativePhysicalLoad(physical, _NATIVE_LOAD_TOKEN)
    bed = _bed.DenseBed._from_native_boundary_load(loaded, config, inputs=inputs,
                                                  expected_identity=identity, max_bytes=max_bytes)
    return RestoredBoundary(bed, physical.mobile_state)
