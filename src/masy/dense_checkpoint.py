"""Source-bound Dense physical bridge; no runner or publication authority.

The caller saves the returned native carrier with MAPLE's exact codec and
retains the original typed input capsule separately. This is not a portable
static-input decoder, combined checkpoint, or active-process continuation.
Native imports are lazy. Private MAPLE15 validation/fingerprinting helpers
are deliberate dependencies of this separately source-qualified prototype.

The named physical-v2 path (`pack_boundary_physical_v2` /
`load_boundary_physical_v2`) is a separate closed layout for the dynamic
manual-vegetation Dense bed. It reuses the same native checkpoint/1 codec;
v1 entrypoints, schemas and refusals are unchanged and refuse dynamic beds.
"""
from __future__ import annotations

import copy
import dataclasses
import json
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

# Separate closed v2 layout (dynamic manual vegetation); the v1 sets above stay exact.
SCHEMA_V2 = "masy.dense_boundary_physical.v2"
CAPSULE_SCHEMA_V2 = "masy.dense_boundary_inputs.v2"
_CASE_V2 = _CASE | {"vegetation_profile"}
_INITIAL_V2 = _INITIAL | {"vegetation_initial_owned_copy_bytes"}
_VEGETATION_CASE = frozenset({"profile", "plan_sha256", "initial_version_sha256", "initial_native_content_sha256"})
_IDENTITY_V2 = (*_IDENTITY, "plan_sha256", "vegetation_metadata_sha256")
_BED_INSTANCE_V2 = _BED_INSTANCE | {"_dynamic", "_veg_initial"}
_ROLES = frozenset({"genesis", "current"})
_REQUEST = ("scope", "role", "plan", "envelope_json", "ledger_json", "expected_metadata_sha256",
            "initial_ticket", "_authority")
_TICKET = ("scope", "plan", "vegetation", "original_inputs_sha256", "case_sha256", "genesis_identity",
           "_authority")
_DYNAMIC_LOAD_TOKEN = object()
_DYNAMIC_SCOPE_TOKEN = object()
_DYNAMIC_REQUEST_TOKEN = object()
_INITIAL_TICKET_TOKEN = object()


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
    return _validate_capsule(inputs, n, cp, cap, cls=DenseBoundaryInputs, case_keys=_CASE,
                             initial_keys=_INITIAL, case_schema=_bed.DENSE_CASE_SCHEMA, material=_material)


def _validate_capsule(inputs, n, cp, cap, *, cls, case_keys, initial_keys, case_schema, material):
    """Shared checked original-capsule invariants; each layout passes its own closed constants."""
    cp._bound(cap)
    cp._fields(inputs, cls, _INPUTS)
    _keys(inputs.case_record, case_keys, "original case")
    _keys(inputs.held_inputs, _HELD, "held original inputs")
    _keys(inputs.initialization_receipt, initial_keys, "original initialization receipt")
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
    if (r["schema"] != case_schema or r["backend"] != "numpy" or r["dtype"] != "float64"
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
    if cp._fingerprint(material(inputs)) != inputs.binding_sha256:
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


# Explicit dynamic artifact path. These APIs do not grant store publication.
RESOURCE_SCHEMA_V2 = "masy.dense_shared_resources.v2"
DYNAMIC_PROFILE_V2 = "dynamic_manual_dense_numpy_accepted_v1"
_RESOURCE_KEYS_V2 = frozenset({"schema", "profile", "version", "prescribed_identity_sha256",
    "vegetation", "vegetation_metadata_sha256", "soil_array", "soil_identity",
    "accepted_run_copy_bytes", "water_exchange"})
_NATIVE_LIMIT_V2 = 64 * 1024 * 1024


def _vegetation():
    from masy import landscape_vegetation
    return landscape_vegetation


def _v2_cap(value):
    if type(value) is not int or not 1 <= value <= _NATIVE_LIMIT_V2:
        _fail("dynamic physical max_bytes must be a plain integer within 64MiB")
    return value


def _digest_v2(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        _fail("dynamic identity must be a lowercase SHA256")
    return value


def _validate_inputs_v2(inputs, n, cp, cap):
    lv = _vegetation()
    _v2_cap(cap)
    _validate_capsule(inputs, n, cp, cap, cls=DenseBoundaryInputsV2, case_keys=_CASE_V2,
                      initial_keys=_INITIAL_V2, case_schema=lv.CASE_SCHEMA_V2, material=_material)
    g = inputs.context.geometry
    nv = inputs.case_record["n_voxels"]
    if (not 1 <= g.ny * g.nx <= 64 or not 1 <= len(inputs.context.grain_classes.classes) <= 6
            or type(nv) is not int or not 1 <= nv <= 16):
        _fail("dynamic physical dimension profile exceeded")
    profile = inputs.case_record["vegetation_profile"]
    _keys(profile, _VEGETATION_CASE, "original vegetation profile")
    if profile["profile"] != lv.VEGETATION_PROFILE:
        _fail("unsupported original vegetation profile")
    for key in _VEGETATION_CASE - {"profile"}:
        _digest_v2(profile[key])
    receipt = inputs.initialization_receipt
    cells = g.ny * g.nx
    if (type(receipt["estimated_initialization_bytes"]) is not int
            or receipt["estimated_initialization_bytes"] < 0
            or type(receipt["vegetation_initial_owned_copy_bytes"]) is not int
            or receipt["vegetation_initial_owned_copy_bytes"] != 40 * cells
            or receipt["estimated_initialization_bytes"] + 40 * cells > inputs.limits.host_memory_cap_bytes):
        _fail("original dynamic initialization copy admission differs")
    return inputs.binding_sha256


def _plan_v2(plan, inputs, n, cp, cap):
    lv = _vegetation()
    _validate_inputs_v2(inputs, n, cp, cap)
    if type(plan) is not lv.VegetationPlan:
        _fail("exact admitted VegetationPlan required")
    plan.verify()
    g = inputs.context.geometry
    if (plan.shape != (g.ny, g.nx) or plan.n_classes != len(inputs.context.grain_classes.classes)
            or plan.n_voxels != inputs.case_record["n_voxels"]
            or plan.plan_sha256 != inputs.case_record["vegetation_profile"]["plan_sha256"]
            or 48 * plan.cells > plan.limits.max_run_copy_bytes):
        _fail("admitted plan/original dynamic case/copy cap differs")


def _check_bed_v2(bed):
    if type(bed) is not _bed.DenseBed or set(vars(bed)) != _BED_INSTANCE_V2:
        _fail("exact dynamic DenseBed layout required")


@dataclasses.dataclass(frozen=True)
class DenseBoundaryInputsV2:
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
        if cls is not DenseBoundaryInputsV2:
            _fail("unsupported dynamic capsule subclass")
        n, cp = _native()
        _check_bed_v2(bed)
        cap = min(bed.limits.host_memory_cap_bytes, _NATIVE_LIMIT_V2)
        _original_metadata(cp, cap, bed._case_record, bed._initialization_receipt, bed._prescribed_digests)
        _budget(cp, cap, bed.context, bed._held_inputs, bed.limits)
        bed.verify_case_inputs()
        candidate = cls(copy.deepcopy(bed._case_record), dict(bed._held_inputs), bed.context, bed.limits,
                        copy.deepcopy(bed._initialization_receipt), dict(bed._prescribed_digests),
                        bed.case_sha256, "")
        candidate = dataclasses.replace(candidate, binding_sha256=cp._fingerprint(_material(candidate)))
        _plan_v2(bed._dynamic.plan, candidate, n, cp, cap)
        return candidate


@dataclasses.dataclass(frozen=True)
class _DynamicDecodeScope:
    plan: Any
    inputs: DenseBoundaryInputsV2
    _authority: object


@dataclasses.dataclass(frozen=True)
class _VegetationRestoreRequest:
    scope: Any
    role: str
    plan: Any
    envelope_json: Any
    ledger_json: str
    expected_metadata_sha256: str
    initial_ticket: Any
    _authority: object


@dataclasses.dataclass(frozen=True)
class _InitialVegetationTicket:
    scope: Any
    plan: Any
    vegetation: Any
    original_inputs_sha256: str
    case_sha256: str
    genesis_identity: dict
    _authority: object


@dataclasses.dataclass(frozen=True)
class _NativePhysicalLoadV2:
    physical: Any
    request: Any
    shared_resources: dict
    _authority: object


@dataclasses.dataclass(frozen=True)
class RestoredBoundaryV2:
    bed: _bed.DenseBed
    mobile_state: Any
    initial_ticket: Any


def _new_dynamic_decode_scope(plan, inputs, *, max_bytes=_NATIVE_LIMIT_V2):
    n, cp = _native()
    _plan_v2(plan, inputs, n, cp, _v2_cap(max_bytes))
    return _DynamicDecodeScope(plan, inputs, _DYNAMIC_SCOPE_TOKEN)


def _scope_v2(scope, inputs, n, cp, cap):
    cp._fields(scope, _DynamicDecodeScope, ("plan", "inputs", "_authority"))
    if scope._authority is not _DYNAMIC_SCOPE_TOKEN or scope.inputs is not inputs:
        _fail("dynamic restore requires the same owning original-input decode")
    _plan_v2(scope.plan, inputs, n, cp, cap)


def _ticket_v2(ticket, scope, n, cp):
    cp._fields(ticket, _InitialVegetationTicket, _TICKET)
    if (ticket._authority is not _INITIAL_TICKET_TOKEN or ticket.scope is not scope
            or ticket.plan is not scope.plan or ticket.original_inputs_sha256 != scope.inputs.binding_sha256
            or ticket.case_sha256 != scope.inputs.case_sha256):
        _fail("foreign initial-vegetation ticket")
    initial = _vegetation()._vegetation_initial_record(scope.plan, ticket.vegetation)
    identity = _validate_identity_v2(ticket.genesis_identity, scope.inputs, cp)
    profile = scope.inputs.case_record["vegetation_profile"]
    if (initial["metadata_sha256"] != identity["vegetation_metadata_sha256"]
            or initial["initial_native_content_sha256"] != profile["initial_native_content_sha256"]
            or initial["version_sha256"] != profile["initial_version_sha256"]):
        _fail("initial native ticket identity changed")
    return ticket.vegetation


def _make_dynamic_restore_request(scope, *, role, ledger_record, expected_metadata_sha256,
                                  envelope_record=None, initial_ticket=None, max_bytes=_NATIVE_LIMIT_V2):
    n, cp = _native()
    cap, lv = _v2_cap(max_bytes), _vegetation()
    cp._fields(scope, _DynamicDecodeScope, ("plan", "inputs", "_authority"))
    _scope_v2(scope, scope.inputs, n, cp, cap)
    if type(role) is not str or role not in _ROLES:
        _fail("unsupported dynamic physical load role")
    _digest_v2(expected_metadata_sha256)
    lv._json_admit(ledger_record, max_bytes=1024 * 1024, max_nodes=1024)
    if role == "genesis":
        if envelope_record is not None or initial_ticket is not None:
            _fail("genesis envelope is derived only from the actual native load")
        envelope_json = None
    else:
        _ticket_v2(initial_ticket, scope, n, cp)
        lv._json_admit(envelope_record, max_bytes=scope.plan.limits.max_metadata_bytes, max_nodes=3551)
        envelope_json = lv._canonical_json(envelope_record)
    return _VegetationRestoreRequest(scope, role, scope.plan, envelope_json,
        lv._canonical_json(ledger_record), expected_metadata_sha256, initial_ticket, _DYNAMIC_REQUEST_TOKEN)


def _request_v2(request, inputs, n, cp, cap):
    cp._fields(request, _VegetationRestoreRequest, _REQUEST)
    _scope_v2(request.scope, inputs, n, cp, cap)
    if (request._authority is not _DYNAMIC_REQUEST_TOKEN or request.plan is not request.scope.plan
            or type(request.role) is not str or request.role not in _ROLES):
        _fail("foreign dynamic restore request")
    _digest_v2(request.expected_metadata_sha256)
    if type(request.ledger_json) is not str or len(request.ledger_json) > 1024 * 1024:
        _fail("full clock raw limit")
    _budget(cp, cap, request.ledger_json, request.envelope_json, plain_json=True)
    ledger = json.loads(request.ledger_json)
    lv = _vegetation()
    lv._json_admit(ledger, max_bytes=1024 * 1024, max_nodes=1024)
    if request.role == "genesis":
        if request.initial_ticket is not None or request.envelope_json is not None:
            _fail("genesis request cannot carry an initial array or envelope")
        envelope = None
    else:
        _ticket_v2(request.initial_ticket, request.scope, n, cp)
        if type(request.envelope_json) is not str or len(request.envelope_json) > request.plan.limits.max_metadata_bytes:
            _fail("envelope raw limit")
        envelope = json.loads(request.envelope_json)
        lv._json_admit(envelope, max_bytes=request.plan.limits.max_metadata_bytes, max_nodes=3551)
    return envelope, ledger


def _soil_identity_v2(identity, shape):
    _keys(identity, ("array",), "soil digest")
    value = identity["array"]
    _keys(value, ("dtype", "shape", "order", "nbytes", "sha256"), "soil array digest")
    if (value["dtype"] != "<f8" or type(value["shape"]) is not list
            or any(type(size) is not int for size in value["shape"])
            or value["shape"] != list(shape) or value["order"] != "C"
            or type(value["nbytes"]) is not int or value["nbytes"] != 8 * shape[0] * shape[1]):
        _fail("current soil digest layout differs")
    _digest_v2(value["sha256"])


def shared_resources_v2(inputs, plan, envelope_record, *, metadata_sha256, soil_identity,
                        accepted_run_copy_bytes, max_bytes=_NATIVE_LIMIT_V2):
    n, cp = _native()
    cap, lv = _v2_cap(max_bytes), _vegetation()
    _plan_v2(plan, inputs, n, cp, cap)
    lv._json_admit(envelope_record, max_bytes=plan.limits.max_metadata_bytes, max_nodes=3551)
    _keys(envelope_record, lv._ENVELOPE_KEYS, "vegetation envelope")
    _budget(cp, cap, soil_identity, plain_json=True)
    _soil_identity_v2(soil_identity, plan.shape)
    _digest_v2(metadata_sha256)
    if (type(accepted_run_copy_bytes) is not int
            or not 48 * plan.cells <= accepted_run_copy_bytes <= plan.limits.max_run_copy_bytes
            or envelope_record.get("metadata_sha256") != metadata_sha256):
        _fail("shared vegetation seal/copy admission differs")
    value = {"schema": RESOURCE_SCHEMA_V2, "profile": DYNAMIC_PROFILE_V2,
        "version": envelope_record.get("version"),
        "prescribed_identity_sha256": _bed._sha256(inputs.prescribed_digests),
        "vegetation": envelope_record, "vegetation_metadata_sha256": metadata_sha256,
        "soil_array": "current_soil", "soil_identity": soil_identity,
        "accepted_run_copy_bytes": accepted_run_copy_bytes, "water_exchange": []}
    _budget(cp, cap, value, plain_json=True)
    return json.loads(lv._canonical_json(value))


def _resources_v2(value, inputs, plan, cp, cap):
    _budget(cp, cap, value, plain_json=True)
    _keys(value, _RESOURCE_KEYS_V2, "dynamic shared resources")
    if (value["schema"] != RESOURCE_SCHEMA_V2 or value["profile"] != DYNAMIC_PROFILE_V2
            or value["soil_array"] != "current_soil" or type(value["water_exchange"]) is not list
            or value["water_exchange"] != [] or type(value["version"]) is not int
            or not 0 <= value["version"] <= plan.limits.max_total_updates):
        _fail("unsupported dynamic shared resource profile")
    made = shared_resources_v2(inputs, plan, value["vegetation"],
        metadata_sha256=value["vegetation_metadata_sha256"], soil_identity=value["soil_identity"],
        accepted_run_copy_bytes=value["accepted_run_copy_bytes"], max_bytes=cap)
    if (value["version"] != made["version"]
            or _vegetation()._canonical_json(value) != _vegetation()._canonical_json(made)):
        _fail("dynamic resource identities differ")
    return made


def _validate_identity_v2(identity, inputs, cp):
    _budget(cp, cp.MAX_METADATA_BYTES, identity, plain_json=True)
    identity = cp._identity(identity)
    _keys(identity, _IDENTITY_V2, "dynamic physical identity")
    if (identity["schema"] != SCHEMA_V2 or identity["original_inputs_sha256"] != inputs.binding_sha256
            or identity["case_sha256"] != inputs.case_sha256
            or identity["plan_sha256"] != inputs.case_record["vegetation_profile"]["plan_sha256"]):
        _fail("dynamic physical/original/plan identity differs")
    for key in ("bed_identity_sha256", "vegetation_metadata_sha256"):
        _digest_v2(identity[key])
    caller = cp._identity(identity["caller"])
    _digest_v2(caller.get("shared_resource_record_sha256"))
    return identity


def _empty_physical_v2(physical, config, inputs, n, cp, cap):
    from maple.aeolian.cohorts.dispatch import mobile_state_is_genuinely_empty
    _validate_physical(physical, config, inputs, n, cp, cap)
    if (type(physical.mobile_state) is not n.MobileCohortState
            or physical.mobile_state.mass_kg.size > 16000
            or not mobile_state_is_genuinely_empty(physical.mobile_state)):
        _fail("dynamic physical boundary requires genuinely empty flat native cohorts")


def _config_context_v2(config, inputs, cp, cap):
    from maple.aeolian.scheduler.state import SchedulerPhysicsConfig
    if type(config) is not SchedulerPhysicsConfig:
        _fail("exact native scheduler configuration required")
    _budget(cp, cap, config)
    for name in ("geometry", "grain_classes", "mass_resolution_kg", "topographic_commit_spec", "avalanching_spec"):
        if cp._fingerprint(getattr(config, name)) != cp._fingerprint(getattr(inputs.context, name)):
            _fail("native configuration differs from the admitted original context")
    # Geometry/classes match the admitted <=64-cell/6-class context BEFORE
    # this owning native resolver can construct its optional fraction grid.
    if _bed._array_digest(config.resolved_initial_available_fraction()) != _bed._array_digest(
            inputs.context.initial_available_fraction):
        _fail("native configuration availability differs")


def pack_boundary_physical_v2(bed, mobile_state, config, identity, *, inputs, shared_resources,
                              max_bytes):
    n, cp = _native()
    cap = _v2_cap(max_bytes)
    _check_bed_v2(bed)
    _plan_v2(bed._dynamic.plan, inputs, n, cp, cap)
    _config_context_v2(config, inputs, cp, cap)
    bed.verify_case_inputs()
    _budget(cp, cap, bed.state, mobile_state, bed._dynamic._native_borrow(),
            bed.threshold_modifier_state, config, inputs)
    if (bed.case_sha256 != inputs.case_sha256 or cp._fingerprint(bed.context) != cp._fingerprint(inputs.context)
            or cp._fingerprint(bed._held_inputs) != cp._fingerprint(inputs.held_inputs)
            or cp._fingerprint(bed._initialization_receipt) != cp._fingerprint(inputs.initialization_receipt)
            or cp._fingerprint(bed._prescribed_digests) != cp._fingerprint(inputs.prescribed_digests)):
        _fail("dynamic bed differs from original capsule")
    resources = _resources_v2(shared_resources, inputs, bed._dynamic.plan, cp, cap)
    if (_vegetation()._canonical_json(resources["vegetation"])
            != _vegetation()._canonical_json(bed._dynamic.as_record())):
        _fail("shared resources do not reproduce the actual bed envelope")
    caller = cp._identity(identity)
    resource_sha = _bed._sha256(resources)
    if "shared_resource_record_sha256" in caller and caller["shared_resource_record_sha256"] != resource_sha:
        _fail("caller resource identity differs")
    caller["shared_resource_record_sha256"] = resource_sha
    physical = n.SchedulerPhysicalState(**{name: getattr(bed.state, name) for name in _bed.SHARED_FIELDS},
        mobile_state=mobile_state, vegetation=bed._dynamic._native_borrow(),
        threshold_modifier_state=bed.threshold_modifier_state, time_s=bed.time_s,
        coupled_trajectory_semantics=None, ledger_reconciliation_baseline=None)
    _empty_physical_v2(physical, config, inputs, n, cp, cap)
    bed._validate_shared(n, bed.state)
    _bed._check_commit_stamp(bed.state, bed.time_s)
    bound = {"schema": SCHEMA_V2, "caller": caller, "original_inputs_sha256": inputs.binding_sha256,
        "case_sha256": bed.case_sha256, "bed_identity_sha256": bed.identity_sha256(),
        "plan_sha256": bed._dynamic.plan.plan_sha256,
        "vegetation_metadata_sha256": resources["vegetation_metadata_sha256"]}
    _validate_identity_v2(bound, inputs, cp)
    _budget(cp, cap, physical, inputs, config, bound, resources)
    return PackedBoundary(physical, bound)


def _restore_dense_bed_v2(cls, loaded, config, *, inputs, expected_identity, cap):
    from masy.clock import ClockLedger
    n, cp = _native()
    if cls is not _bed.DenseBed:
        _fail("unsupported DenseBed subclass")
    cp._fields(loaded, _NativePhysicalLoadV2, ("physical", "request", "shared_resources", "_authority"))
    if loaded._authority is not _DYNAMIC_LOAD_TOKEN:
        _fail("restore requires the immediate dynamic native-loader ticket")
    request, physical, lv = loaded.request, loaded.physical, _vegetation()
    record, ledger_record = _request_v2(request, inputs, n, cp, cap)
    identity = _validate_identity_v2(expected_identity, inputs, cp)
    _config_context_v2(config, inputs, cp, cap)
    _empty_physical_v2(physical, config, inputs, n, cp, cap)
    _budget(cp, cap, physical, require_readonly=True)
    ledger = ClockLedger.from_record(ledger_record)
    endpoint = ledger.epoch if not ledger._entries else ledger._entries[-1].native_end
    if ledger.epoch.hex() != decode_time(inputs.case_record["initial_time_s"]).hex() or endpoint.hex() != physical.time_s.hex():
        _fail("native physical endpoint differs from the complete accepted ledger")
    if request.role == "genesis":
        if ledger._entries:
            _fail("genesis physical role has accepted clock entries")
        initial = physical.vegetation
        record = lv._vegetation_initial_record(request.plan, initial)
    else:
        initial = _ticket_v2(request.initial_ticket, request.scope, n, cp)
    if (record["metadata_sha256"] != request.expected_metadata_sha256
            or identity["vegetation_metadata_sha256"] != request.expected_metadata_sha256):
        _fail("independently expected vegetation metadata seal differs")
    envelope = lv._restore_vegetation_envelope(request.plan, record, ledger_record=ledger_record,
        initial=initial, current=physical.vegetation, expected_metadata_sha256=request.expected_metadata_sha256)
    original = inputs.case_record["vegetation_profile"]
    initial_record = lv._vegetation_initial_record(request.plan, initial)
    if (initial_record["version_sha256"] != original["initial_version_sha256"]
            or envelope.initial_native_content_sha256 != original["initial_native_content_sha256"]):
        _fail("native initial vegetation differs from the original case")
    resources = _resources_v2(loaded.shared_resources, inputs, request.plan, cp, cap)
    if (_vegetation()._canonical_json(resources["vegetation"]) != _vegetation()._canonical_json(record)
            or _bed._sha256(resources) != identity["caller"]["shared_resource_record_sha256"]):
        _fail("native caller does not bind the selected shared resources")
    state = n.BedState(**{name: getattr(physical, name) for name in _bed.SHARED_FIELDS})
    bed = cls(_bed._TOKEN, case_record=copy.deepcopy(inputs.case_record), held_inputs=dict(inputs.held_inputs),
        context=inputs.context, state=state, vegetation=physical.vegetation,
        threshold_modifier_state=physical.threshold_modifier_state, time_s=physical.time_s,
        limits=inputs.limits, initialization_receipt=copy.deepcopy(inputs.initialization_receipt),
        prescribed_digests=dict(inputs.prescribed_digests))
    bed._veg_initial, bed._dynamic = initial, envelope
    if (bed._dynamic.plan is not request.plan or bed._veg_initial is not initial
            or bed._vegetation is not bed._dynamic._native_borrow()):
        _fail("dynamic loaded native ownership aliases differ")
    bed.verify_case_inputs()
    bed._validate_shared(n, state)
    _bed._check_commit_stamp(state, physical.time_s)
    if bed.identity_sha256() != identity["bed_identity_sha256"]:
        _fail("loaded dynamic native fields do not reproduce accepted bed identity")
    _validate_inputs_v2(inputs, n, cp, cap)
    return bed


def load_boundary_physical_v2(path, config, expected_identity, *, inputs, request, max_bytes,
                              shared_resources=None, genesis_soil_identity=None):
    """Load physical authority; derive genesis resources only after its real load.

    The expected native identity already seals the v0 resource hash. Original
    soil digest comes from the owning admitted OI decode; B checks that role.
    No unavailable genesis array/record is reconstructed before loading it.
    """
    n, cp = _native()
    cap = _v2_cap(max_bytes)
    record, ledger = _request_v2(request, inputs, n, cp, cap)
    identity = _validate_identity_v2(expected_identity, inputs, cp)
    _config_context_v2(config, inputs, cp, cap)
    if identity["vegetation_metadata_sha256"] != request.expected_metadata_sha256:
        _fail("request and independently expected native metadata seal differ")
    if request.role == "genesis":
        if shared_resources is not None:
            _fail("genesis resource envelope must be derived from the native load")
        _budget(cp, cap, genesis_soil_identity, plain_json=True)
        _soil_identity_v2(genesis_soil_identity, request.plan.shape)
        resource_static = genesis_soil_identity
    else:
        if genesis_soil_identity is not None:
            _fail("current physical role cannot replace original soil identity")
        shared_resources = _resources_v2(shared_resources, inputs, request.plan, cp, cap)
        if (_vegetation()._canonical_json(shared_resources["vegetation"]) != _vegetation()._canonical_json(record)
                or _bed._sha256(shared_resources) != identity["caller"]["shared_resource_record_sha256"]):
            _fail("selected B resources differ from the independently sealed native reference")
        resource_static = shared_resources
    static = _budget(cp, cap, inputs, config, identity, request.envelope_json,
                     request.ledger_json, resource_static)
    if request.role == "genesis":
        # No genesis native array/record is invented to preflight metadata.
        # The accepted component supplies its source bound for v0 encoding.
        # Outside that component the call path below emits three v0 records
        # and two resource roundtrips (encoded + expanded): seven encodings.
        # The resource's exact comparison adds two frame+v0 encodings and the
        # selected-envelope exact comparison adds two v0 encodings: 11 v0 and
        # six frame encodings in total. Scratch and
        # Python heap are separately counted by the execution allocation gate.
        lv = _vegetation()
        v0_bound = lv._envelope_admit(request.plan.limits, receipts_bytes=2,
            bindings_bytes=2, last_binding_bytes=0)
        frame = {"schema": RESOURCE_SCHEMA_V2, "profile": DYNAMIC_PROFILE_V2, "version": 0,
            "prescribed_identity_sha256": _bed._sha256(inputs.prescribed_digests),
            "vegetation": None, "vegetation_metadata_sha256": request.expected_metadata_sha256,
            "soil_array": "current_soil", "soil_identity": genesis_soil_identity,
            "accepted_run_copy_bytes": 48 * request.plan.cells, "water_exchange": []}
        static += 11 * v0_bound + 6 * _budget(cp, cap, frame, plain_json=True)
    if cap - static < 1:
        _fail("no dynamic physical payload capacity remains")
    physical = cp.load_exact_physical_checkpoint(path, config, identity, max_bytes=cap - static)
    if request.role == "genesis":
        _budget(cp, cap, physical, require_readonly=True)
        initial_record = _vegetation()._vegetation_initial_record(request.plan, physical.vegetation)
        shared_resources = shared_resources_v2(inputs, request.plan, initial_record,
            metadata_sha256=request.expected_metadata_sha256, soil_identity=genesis_soil_identity,
            accepted_run_copy_bytes=48 * request.plan.cells, max_bytes=cap)
    loaded = _NativePhysicalLoadV2(physical, request, shared_resources, _DYNAMIC_LOAD_TOKEN)
    bed = _bed.DenseBed._from_native_boundary_load_v2(loaded, config, inputs=inputs,
        expected_identity=identity, max_bytes=cap)
    ticket = (request.initial_ticket if request.role == "current" else _InitialVegetationTicket(
        request.scope, request.plan, physical.vegetation, inputs.binding_sha256,
        inputs.case_sha256, identity, _INITIAL_TICKET_TOKEN))
    return RestoredBoundaryV2(bed, physical.mobile_state, ticket)
