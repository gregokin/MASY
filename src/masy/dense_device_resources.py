"""Private Dense device owner of the wind pool, soil water and static device inputs.

Companion of `DeviceDenseBedOwner` (which owns ONLY the six bed holders). Same
rules: accepted arrays never escape, native callers receive work copies, a
publication copies the candidate into TWO new private buffers, and raw byte
equality is checked only at ownership boundaries with ONE compact host packet
per check. Statics are owned copies plus mirrors and are never replaced.

Byte reservation is structural (pool/soil x {accepted, mirror, work, new,
new-mirror}, statics x {owned, mirror}, comparison scratch). It is not a
driver/pool fit proof. No run/source/allocation authority is held here.
"""

from __future__ import annotations

import dataclasses
import threading
from typing import Any

__all__ = ["DeviceResourceError", "DeviceResourceLimits"]

STATIC_NAMES = ("vegetation", "threshold_modifier_state", "initial_available_fraction", "column", "field",
                "sediment")
_PRIVATE_COPIES = 5  # accepted, mirror, work, new, new-mirror
_STATIC_COPIES = 2  # owned, mirror


class DeviceResourceError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise DeviceResourceError("invalid_limit", f"{name} must be plain int >= 1")
    return value


@dataclasses.dataclass(frozen=True)
class DeviceResourceLimits:
    """Structural reservations only; scratch is declared, not measured."""

    max_private_bytes: int
    max_static_bytes: int
    max_work_bytes: int
    comparison_reduction_scratch_bytes: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _integer(getattr(self, field.name), field.name)


_META_MAX_ENTRIES = 4096
_META_MAX_BYTES = 64 << 10  # host metadata scope cap (strings/keys/holder names), not a physics decision
_META_MAX_DEPTH = 8
_META_MAX_ITEMS = 256  # per tuple/list/dict
_ARRAY_KINDS = "fbiu"  # native float/bool/int leaves; complex/object/str/void refused
_ARRAY_MAX_NDIM = 4


def _closed_metadata(value, array_types, xp_modules):
    """Bounded closed walk of a native holder tree, BEFORE any copy.

    Admits only native (`maple*`) dataclass holders with exactly their declared
    fields (no extra attributes), arrays of `array_types` with native dtypes,
    plain host scalars (floats bound bit-exactly via hex, so signed zero is
    kept), enums, bounded tuples/lists/str-key dicts and the exact `xp_modules`.
    Repeated references (legitimate native internal aliases) are recorded as
    aliases, not refused. Returns (record, arrays, modules); `record` carries no
    object identity, so an owned copy reproduces its source's record exactly.
    """
    import enum
    from types import ModuleType

    record, arrays, modules, seen = [], [], [], {}
    size = [0]

    def add(entry, nbytes=0):
        record.append(entry)
        size[0] += nbytes + 32
        if len(record) > _META_MAX_ENTRIES or size[0] > _META_MAX_BYTES:
            raise DeviceResourceError("oversized_metadata", "holder metadata exceeds the bounded scope")

    def walk(obj, path, depth):
        if depth > _META_MAX_DEPTH:
            raise DeviceResourceError("oversized_metadata", f"{path}: holder nesting too deep")
        kind = type(obj)
        if obj is None or kind in (bool, int, str):
            if kind is int and obj.bit_length() > 63:
                raise DeviceResourceError("unsupported_schema", f"{path}: unbounded int")
            add((path, kind.__name__, obj), len(obj.encode("utf-8")) if kind is str else 0)
            return
        if kind is float:
            add((path, "float", obj.hex()))
            return
        if isinstance(obj, enum.Enum):
            add((path, "enum", f"{kind.__module__}.{kind.__qualname__}", repr(obj.value)), 64)
            return
        if kind is ModuleType:
            if not any(obj is m for m in xp_modules):
                raise DeviceResourceError("unsupported_backend", f"{path}: undeclared module namespace")
            modules.append(obj)
            add((path, "module", obj.__name__.split(".")[0]))
            return
        key = id(obj)
        if key in seen:
            add((path, "alias", seen[key]))
            return
        seen[key] = path
        if kind in array_types:
            if (obj.dtype.hasobject or obj.dtype.kind not in _ARRAY_KINDS or obj.ndim > _ARRAY_MAX_NDIM
                    or not obj.flags.c_contiguous):
                raise DeviceResourceError("unsupported_layout", f"{path}: unsupported dtype/shape/layout")
            arrays.append(obj)
            add((path, "array", kind.__module__.split(".")[0], obj.dtype.str, tuple(obj.shape)))
            return
        if kind in (tuple, list):
            if len(obj) > _META_MAX_ITEMS:
                raise DeviceResourceError("oversized_metadata", f"{path}: sequence too long")
            add((path, kind.__name__, len(obj)))
            for i, item in enumerate(obj):
                walk(item, f"{path}[{i}]", depth + 1)
            return
        if kind is dict:
            if len(obj) > _META_MAX_ITEMS or any(type(k) is not str for k in obj):
                raise DeviceResourceError("unsupported_schema", f"{path}: bounded str-key mapping required")
            add((path, "dict", tuple(obj)), sum(len(k.encode("utf-8")) for k in obj))
            for k, item in obj.items():
                walk(item, f"{path}[{k!r}]", depth + 1)
            return
        if dataclasses.is_dataclass(obj) and kind.__module__.split(".")[0] in ("maple", "maple_syrup"):
            names = tuple(f.name for f in dataclasses.fields(obj))
            if hasattr(obj, "__dict__") and set(vars(obj)) != set(names):
                raise DeviceResourceError("unsupported_schema", f"{path}: undeclared/missing holder attributes")
            add((path, "holder", f"{kind.__module__}.{kind.__qualname__}", names),
                len(kind.__qualname__) + sum(len(k) for k in names))
            for name in names:
                walk(getattr(obj, name), f"{path}.{name}", depth + 1)
            return
        raise DeviceResourceError("unsupported_schema", f"{path}: undeclared payload {kind.__name__} refused")

    walk(value, "", 0)
    return tuple(record), arrays, modules


def _structure(record):
    """Holder/array/module structure of a metadata record; scalar values dropped (evolving pool scalars)."""
    return tuple(e if e[1] in ("holder", "array", "module", "tuple", "list", "dict", "alias") else e[:2]
                 for e in record)


def resource_reservation(private_bytes: int, static_bytes: int, n_leaves: int, largest_leaf_bytes: int,
                         limits: DeviceResourceLimits) -> tuple[int, int]:
    """(comparison, required) structural bytes; shared with the factory's preflight."""
    compare = int(largest_leaf_bytes) + 8 * (n_leaves + 16) + limits.comparison_reduction_scratch_bytes
    return int(compare), int(_PRIVATE_COPIES * private_bytes + _STATIC_COPIES * static_bytes + compare)


@dataclasses.dataclass(frozen=True)
class _Private:
    pool: Any
    pool_mirror: Any
    soil: Any
    soil_mirror: Any
    generation: int


class ResourceStage:
    """One work borrow of pool + soil (copies made on demand, at most once each)."""

    __slots__ = ("_consumed", "_generation", "_owner", "_pool", "_soil")

    def __init__(self, owner, generation):
        self._owner, self._generation = owner, generation
        self._pool = self._soil = None
        self._consumed = False

    @property
    def consumed(self):
        return self._consumed


class _ResourcePrepared:
    __slots__ = ("_accepted", "_consumed", "_generation", "_owner", "_stage")

    def __init__(self, owner, generation, stage, accepted):
        self._owner, self._generation = owner, generation
        self._stage, self._accepted, self._consumed = stage, accepted, False


class DeviceDenseResources:
    """Single-thread/current-stream, same-device private owner of pool, soil and statics.

    `_create` is the only constructor; `_ops` is the same adapter shape as
    `dense_device_bed._CupyOps` (genuine CuPy in production, a NumPy semantic
    seam in CPU tests). Leaves are discovered by MAPLE's native `array_leaves`.
    """

    def __init__(self):
        raise DeviceResourceError("constructor", "private factory only")

    @classmethod
    def _create(cls, *, pool, soil, statics, shape, limits, _ops):
        from maple.core.backend import array_leaves
        from maple.core.backend.arrays import _map_tree

        if cls is not DeviceDenseResources:
            raise DeviceResourceError("unsupported_holder", "owner subclasses unsupported")
        if type(limits) is not DeviceResourceLimits or set(vars(limits)) != {
                f.name for f in dataclasses.fields(limits)}:
            raise DeviceResourceError("invalid_limit", "exact declared DeviceResourceLimits required")
        limits.__post_init__()
        if type(statics) is not dict or tuple(statics) != STATIC_NAMES:
            raise DeviceResourceError("invalid_static", f"statics must be exactly {STATIC_NAMES} in order")
        owner = object.__new__(cls)
        owner._ops, owner._leaves, owner._map = _ops, array_leaves, _map_tree
        owner._limits = limits
        owner._limit_record = tuple(getattr(limits, f.name) for f in dataclasses.fields(limits))
        owner._shape = tuple(shape)
        owner._token, owner._active = object(), None
        owner._thread = threading.get_ident()
        owner._device, owner._stream = _ops.current_device(), _ops.current_stream()
        owner._counts = {"d2d_enqueued_arrays": 0, "d2d_enqueued_bytes": 0, "clone_attempts": 0,
                         "packet_attempts": 0, "packets_completed": 0}
        owner._pool_type = type(pool)
        # Closed bounded holder metadata of every input BEFORE any clone: exact types/fields/attributes,
        # scalars, enums, module identity and array layouts; undeclared payloads are refused, not ignored.
        pool_record, _ = owner._metadata(pool, "pool")
        owner._pool_structure = _structure(pool_record)
        pool_leaves = owner._device_leaves(pool, "pool")
        owner._pool_layout = owner._layout(pool_leaves)
        owner._soil_check(soil)
        owner._static_meta = {name: owner._metadata(value, name, allow_host=True) for name, value in statics.items()}
        static_leaves = {name: owner._device_leaves(value, name, allow_host=True)
                         for name, value in statics.items()}
        private_bytes = sum(a.nbytes for _, a in pool_leaves) + int(soil.nbytes)
        device_static = [(p, a) for leaves in static_leaves.values() for p, a in leaves
                         if type(a) is _ops.array_type]
        static_bytes = sum(a.nbytes for _, a in device_static)
        all_leaves = [*pool_leaves, ("soil", soil), *device_static]
        compare, required = resource_reservation(private_bytes, static_bytes, len(all_leaves),
                                                 max(a.nbytes for _, a in all_leaves), limits)
        if (private_bytes > limits.max_private_bytes or static_bytes > limits.max_static_bytes
                or required > limits.max_work_bytes):
            raise DeviceResourceError("work_bytes_cap", "pool/soil/static copies plus comparison exceed reservation")
        owner._private_bytes, owner._static_bytes = private_bytes, static_bytes
        owner._compare_bytes, owner._required_bytes = compare, required
        # Owned statics and mirrors: independent copies of every leaf (device on device, host on host).
        owned = {name: owner._clone(value) for name, value in statics.items()}
        mirrors = {name: owner._clone(value) for name, value in owned.items()}
        owner._statics, owner._static_mirrors = owned, mirrors
        owner._static_layout = {name: owner._layout(owner._device_leaves(value, name, allow_host=True))
                                for name, value in owned.items()}
        pool_a, pool_m = owner._clone(pool), owner._clone(pool)
        soil_a, soil_m = owner._clone(soil), owner._clone(soil)
        owner._disjoint(pool, soil, *statics.values())  # distinct native inputs must not alias each other
        for name in STATIC_NAMES:
            owner._disjoint(owned[name], statics[name], mirrors[name], pool, soil)
            owner._disjoint(mirrors[name], statics[name], pool, soil)
        owner._disjoint(pool_a, pool, pool_m, soil, soil_a, soil_m)
        owner._disjoint(pool_m, pool, soil, soil_a, soil_m)
        owner._disjoint(soil_a, soil, soil_m)
        owner._disjoint(soil_m, soil)
        owner._compare((pool, pool_a), (pool_a, pool_m), (soil, soil_a), (soil_a, soil_m),
                       *((statics[name], owned[name]) for name in STATIC_NAMES),
                       *((owned[name], mirrors[name]) for name in STATIC_NAMES))
        for name in STATIC_NAMES:  # owned copies and mirrors reproduce the source metadata exactly
            owner._check_static_meta(name, owned[name])
            owner._check_static_meta(name, mirrors[name])
        for value in (pool_a, pool_m):
            owner._check_pool(value)
        owner._accepted = _Private(pool_a, pool_m, soil_a, soil_m, 0)
        return owner

    # --- structure -------------------------------------------------------------------------------------------------
    def _scope(self):
        if threading.get_ident() != self._thread:
            raise DeviceResourceError("wrong_thread", "owner is single-threaded")
        if self._ops.current_device() != self._device or self._ops.current_stream() != self._stream:
            raise DeviceResourceError("device_stream_changed", "current device/stream must remain pinned")
        if tuple(getattr(self._limits, f.name) for f in dataclasses.fields(self._limits)) != self._limit_record:
            raise DeviceResourceError("invalid_limit", "resource reservation changed after ownership began")

    def _device_leaves(self, value, path, *, allow_host=False):
        """Native leaf discovery; every leaf contiguous, non-object, on the pinned device (host only if allowed)."""
        np = self._ops.host_np
        leaves = []
        for i, array in enumerate(self._leaves(value)):
            where = f"{path}[{i}]"
            if type(array) is self._ops.array_type:
                if self._ops.device(array) != self._device:
                    raise DeviceResourceError("wrong_device", f"{where}: leaf is not on pinned device")
            elif not (allow_host and type(array) is np.ndarray):
                raise DeviceResourceError("unsupported_backend", f"{where}: exact resident array required")
            if not array.flags.c_contiguous or array.dtype.hasobject:
                raise DeviceResourceError("unsupported_layout", f"{where}: contiguous non-object array required")
            leaves.append((where, array))
        return leaves

    def _metadata(self, value, path, *, allow_host=False):
        """(record, modules) of the closed walk; its arrays must be exactly the native `array_leaves`."""
        types = (self._ops.array_type, self._ops.host_np.ndarray) if allow_host else (self._ops.array_type,)
        record, arrays, modules = _closed_metadata(value, types, (self._ops.xp,))
        if {id(a) for a in arrays} != {id(a) for a in self._leaves(value)}:
            raise DeviceResourceError("unsupported_schema", f"{path}: native leaves differ from the closed schema")
        return record, tuple(modules)

    def _check_static_meta(self, name, value):
        record, modules = self._metadata(value, name, allow_host=True)
        held_record, held_modules = self._static_meta[name]
        if record != held_record or len(modules) != len(held_modules) or any(
                a is not b for a, b in zip(modules, held_modules, strict=True)):
            raise DeviceResourceError("static_changed", f"{name}: owned static holder metadata changed")

    @staticmethod
    def _layout(leaves):
        return tuple((path, type(a).__module__.split(".")[0], a.dtype.str, tuple(a.shape)) for path, a in leaves)

    def _soil_check(self, soil):
        if (type(soil) is not self._ops.array_type or tuple(soil.shape) != self._shape
                or soil.dtype != self._ops.host_np.float64 or not soil.flags.c_contiguous
                or self._ops.device(soil) != self._device):
            raise DeviceResourceError("invalid_soil", "soil water must be a contiguous (ny, nx) float64 device array")

    def _check_pool(self, pool):
        if type(pool) is not self._pool_type:
            raise DeviceResourceError("unsupported_holder", "pool must keep the native holder type")
        if _structure(self._metadata(pool, "pool")[0]) != self._pool_structure:
            raise DeviceResourceError("unsupported_schema", "pool holder/field/array structure changed")
        if self._layout(self._device_leaves(pool, "pool")) != self._pool_layout:
            raise DeviceResourceError("invalid_shape", "pool leaf layout changed")

    def _clone(self, value):
        memo = {}
        np = self._ops.host_np

        def leaf(array):
            found = memo.get(id(array))
            if found is not None:
                return found[1]
            self._counts["clone_attempts"] += 1
            if type(array) is np.ndarray and type(array) is not self._ops.array_type:
                moved = np.array(array, order="C", copy=True)
                moved.flags.writeable = array.flags.writeable
            else:
                moved = self._ops.clone(array)
                self._counts["d2d_enqueued_arrays"] += 1
                self._counts["d2d_enqueued_bytes"] += int(array.nbytes)
            memo[id(array)] = (array, moved)
            return moved

        return self._map(value, leaf)

    def _disjoint(self, value, *others):
        """No device leaf of `value` shares memory with any device leaf of `others` (the bed owner's native
        `arrays_share_memory` predicate, contiguity checked first). Host static leaves use NumPy's exact
        `shares_memory` (views are caught, not only identity). Aliases WITHIN one native holder are its own
        legitimate relationships and are never compared here."""
        np = self._ops.host_np
        left = [a for a in self._leaves(value)]
        for other in others:
            for b in self._leaves(other):
                for a in left:
                    if type(a) is np.ndarray and type(a) is not self._ops.array_type:
                        if type(b) is np.ndarray and bool(np.shares_memory(a, b)):
                            raise DeviceResourceError("private_alias", "host static leaf memory is shared")
                        continue
                    if type(b) is not self._ops.array_type:
                        continue
                    if not a.flags.c_contiguous or not b.flags.c_contiguous:
                        raise DeviceResourceError("unsupported_layout", "contiguity required before overlap predicate")
                    if self._ops.overlap(a, b):
                        raise DeviceResourceError("private_alias", "work/candidate/private copy allocations overlap")

    def _compare(self, *pairs, extra_flags=()):
        np = self._ops.host_np
        flags, host_equal = list(extra_flags), True
        for left, right in pairs:
            a, b = self._leaves(left), self._leaves(right)
            if len(a) != len(b):
                raise DeviceResourceError("integrity_failed", "leaf count differs")
            for x, y in zip(a, b, strict=True):
                if type(x) is not type(y) or x.dtype != y.dtype or tuple(x.shape) != tuple(y.shape):
                    raise DeviceResourceError("integrity_failed", "leaf type/dtype/shape differs")
                if type(x) is self._ops.array_type:
                    flags.append(self._ops.raw_equal(x, y))
                else:  # host static leaf: raw bytes on the host, no transfer
                    host_equal &= bool(np.array_equal(x.reshape(-1).view(np.uint8), y.reshape(-1).view(np.uint8)))
        if not host_equal:
            raise DeviceResourceError("integrity_failed", "raw host static bytes differ")
        if not flags:
            return
        self._counts["packet_attempts"] += 1
        packet = self._ops.packet(flags)
        self._counts["packets_completed"] += 1
        if packet.shape != (len(flags),) or not all(float(v) == 1.0 for v in packet):
            raise DeviceResourceError("integrity_failed", "raw device bytes differ")

    # --- verification ----------------------------------------------------------------------------------------------
    def verify(self):
        """Accepted pool/soil equal their mirrors (one packet)."""
        self._scope()
        acc = self._accepted
        self._check_pool(acc.pool)
        self._soil_check(acc.soil)
        self._compare((acc.pool, acc.pool_mirror), (acc.soil, acc.soil_mirror))

    def verify_static(self):
        """Owned statics and mirrors keep their closed holder metadata (types/fields/scalars/enums/xp identity/
        array layouts) and layout, and are raw-equal (one packet). No download or hash of spatial content."""
        self._scope()
        for name in STATIC_NAMES:
            layout = self._layout(self._device_leaves(self._statics[name], name, allow_host=True))
            if layout != self._static_layout[name]:
                raise DeviceResourceError("static_changed", f"{name}: owned static layout changed")
            self._check_static_meta(name, self._statics[name])
            self._check_static_meta(name, self._static_mirrors[name])
        self._compare(*((self._statics[name], self._static_mirrors[name]) for name in STATIC_NAMES))

    def static(self, name):
        """Owned static object (native inputs are read-only by native contract; mirrors detect writes)."""
        return self._statics[name]

    # --- transactions ----------------------------------------------------------------------------------------------
    def begin_stage(self):
        self._scope()
        if self._active is not None:
            raise DeviceResourceError("transaction_active", "one active work transaction allowed")
        self.verify()
        stage = ResourceStage(self._token, self._accepted.generation)
        self._active = stage
        return stage

    def _ticket(self, ticket, kind):
        self._scope()
        if type(ticket) is not kind or ticket._owner is not self._token:
            raise DeviceResourceError("foreign_ticket", "ticket belongs to a different owner/type")
        if ticket._consumed:
            raise DeviceResourceError("ticket_consumed", "ticket was consumed")
        if self._active is not ticket or ticket._generation != self._accepted.generation:
            raise DeviceResourceError("stale_ticket", "ticket is not current active generation")

    def work_pool(self, stage):
        self._ticket(stage, ResourceStage)
        if stage._pool is None:
            acc = self._accepted
            work = self._clone(acc.pool)
            self._disjoint(work, acc.pool, acc.pool_mirror, acc.soil, acc.soil_mirror)
            self._compare((acc.pool, work))
            stage._pool = work
        return stage._pool

    def work_soil(self, stage):
        self._ticket(stage, ResourceStage)
        if stage._soil is None:
            acc = self._accepted
            work = self._clone(acc.soil)
            self._disjoint(work, acc.pool, acc.pool_mirror, acc.soil, acc.soil_mirror)
            self._compare((acc.soil, work))
            stage._soil = work
        return stage._soil

    def prepare_publish(self, stage, *, pool=None, soil=None):
        """Candidate pool and/or soil (None keeps the accepted pair) into new private pairs."""
        self._ticket(stage, ResourceStage)
        acc = self._accepted
        accepted = (acc.pool, acc.pool_mirror, acc.soil, acc.soil_mirror)
        # All admission (type/layout/alias) precedes any clone.
        if pool is not None:
            self._check_pool(pool)
            self._disjoint(pool, *accepted)
        if soil is not None:
            self._soil_check(soil)
            self._disjoint(soil, *accepted)
        if pool is None:
            new_pool, pool_mirror = acc.pool, acc.pool_mirror
        else:
            new_pool, pool_mirror = self._clone(pool), self._clone(pool)
            self._disjoint(new_pool, pool, pool_mirror, *accepted)
            self._disjoint(pool_mirror, pool, *accepted)
        if soil is None:
            new_soil, soil_mirror = acc.soil, acc.soil_mirror
        else:
            new_soil, soil_mirror = self._clone(soil), self._clone(soil)
            self._disjoint(new_soil, soil, soil_mirror, *accepted)
            self._disjoint(soil_mirror, soil, *accepted)
        pairs = [(acc.pool, acc.pool_mirror), (acc.soil, acc.soil_mirror)]
        extra = []
        if pool is not None:
            pairs += [(pool, new_pool), (new_pool, pool_mirror)]
        if soil is not None:
            pairs += [(soil, new_soil), (new_soil, soil_mirror)]
            # Soil water finite and >= 0 in the SAME compact packet (no extra download). Signed zero is NOT
            # refused here: that stricter rule belongs to construction admission (`_check_water`) only.
            xp = self._ops.xp
            extra = [xp.all(xp.isfinite(soil) & (soil >= 0.0))]
        self._compare(*pairs, extra_flags=extra)
        prepared = _ResourcePrepared(self._token, acc.generation, stage,
                                     _Private(new_pool, pool_mirror, new_soil, soil_mirror, acc.generation + 1))
        self._active = prepared
        return prepared

    def _check_publish(self, prepared):
        self._ticket(prepared, _ResourcePrepared)
        self.verify()

    def _swap(self, prepared):
        """Assignment only; callers ran `_check_publish`."""
        stage = prepared._stage
        self._accepted = prepared._accepted
        stage._pool, stage._soil, stage._consumed = None, None, True
        prepared._accepted, prepared._stage, prepared._consumed = None, None, True
        self._active = None

    def discard(self, ticket):
        kind = ResourceStage if type(ticket) is ResourceStage else _ResourcePrepared
        self._ticket(ticket, kind)
        if kind is _ResourcePrepared:
            stage = ticket._stage
            stage._pool, stage._soil, stage._consumed = None, None, True
            ticket._stage, ticket._accepted = None, None
        else:
            ticket._pool, ticket._soil = None, None
        ticket._consumed, self._active = True, None

    def _discard_active(self):
        if self._active is not None:
            self.discard(self._active)

    @property
    def generation(self):
        return self._accepted.generation

    @property
    def active(self):
        return self._active is not None

    def pool_bytes(self):
        return sum(a.nbytes for a in self._leaves(self._accepted.pool))

    def soil_bytes(self):
        return int(self._accepted.soil.nbytes)

    def evidence(self):
        return {
            "schema": "masy.device_resources_integrity.v1",
            "generation": self.generation,
            "visible_device_ordinal": self._device,
            "stream_pointer": self._stream,
            "private_bytes": self._private_bytes,
            "static_device_bytes": self._static_bytes,
            "private_copy_count": _PRIVATE_COPIES,
            "static_copy_count": _STATIC_COPIES,
            "comparison_reservation_bytes": self._compare_bytes,
            "required_structural_work_bytes": self._required_bytes,
            "copy_and_packet_counters": dict(self._counts),
            "limits": dataclasses.asdict(self._limits),
            "scope": "private pool/soil/static ownership only; driver, pool allocator and native peak unqualified",
        }

    @classmethod
    def _from_restored_boundary(cls, ticket, *, pool, soil, statics, shape, limits, _ops, generation):
        """Genuine private/static construction followed by MASY label rebind.

        The existing constructor, copies, mirrors, admission and all native
        arrays are unchanged. No Native private counter or old ticket enters.
        """
        from .windowed_device_checkpoint import _checked_ticket, _integer
        _checked_ticket(ticket)
        _integer(generation, 4, "saved MASY resource generation")
        if generation != ticket.current["bookkeeping"]["next_index"]:
            raise DeviceResourceError("restore_generation", "resource/cursor mismatch")
        owner = cls._create(pool=pool, soil=soil, statics=statics, shape=shape, limits=limits, _ops=_ops)
        if owner.generation != 0 or owner.active:
            raise DeviceResourceError("restore_generation", "fresh idle resources required")
        owner._accepted = dataclasses.replace(owner._accepted, generation=generation)
        owner.verify()
        return owner
