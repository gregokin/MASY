"""Bounded default Dense CPU accepted-boundary store (source-manifest prototype).

The durable authority is HEAD. Native payloads belong to the unchanged original,
passive-wrapper and MAPLE codecs. This module owns only finite filesystem control
metadata and private orchestration; it never publishes a live runner.
"""

from __future__ import annotations

import contextlib
import copy
import ctypes
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import math
import os
import platform
import re
import stat
import sys
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

from masy import dense_checkpoint as bridge
from masy import dense_original_inputs as oi
from masy import dense_runner as dr
from masy import dense_runner_checkpoint as b
from masy import runtime
from masy import runtime_worker as rw

SCHEMA = "masy.dense_boundary_store.v1"
PROFILE = "default_dense_numpy_accepted_v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_NONCE = re.compile(r"[0-9a-f]{32}\Z")
_GEN = re.compile(r"g-([0-9]{4})-([0-9a-f]{32})\Z")
_TEMP = re.compile(r"HEAD\.([0-9a-f]{32})\.tmp\Z")
_STORE = (
    "schema",
    "store_id",
    "profile",
    "source_identity",
    "runtime_reference",
    "environment_sha256",
    "storage_contract",
    "limits",
    "originals",
    "genesis_physical",
    "store_sha256",
)
_INTENT = (
    "schema",
    "store_id",
    "attempt_id",
    "sequence",
    "expected_parent_head_file_sha256",
    "expected_parent_generation",
    "operation_id",
    "source_identity",
    "environment_sha256",
    "intent_sha256",
)
_GENERATION = (
    "schema",
    "store_id",
    "store_file_sha256",
    "attempt_id",
    "sequence",
    "intent_file_sha256",
    "parent_head_record",
    "parent_head_file_sha256",
    "parent_generation",
    "operation_id",
    "image",
    "accepted_summary",
    "receipt_bytes_sha256",
    "generation_sha256",
)
_HEAD = (
    "schema",
    "store_id",
    "store_file_sha256",
    "sequence",
    "generation_name",
    "generation_file_sha256",
    "generation_sha256",
    "operation_id",
    "head_sha256",
)
_RUNTIME = (
    "schema",
    "kind",
    "source_manifest_path",
    "source_manifest_file_sha256",
    "native_bundle_manifest_path",
    "native_bundle_manifest_file_sha256",
    "module_roots",
    "cpu_store",
    "cpu_lock_id",
    "cpu_lock_file_sha256",
)
_CONTRACT = (
    "schema",
    "operator_declares_local",
    "operator_declares_fsync_durable",
    "observed_fs_type",
    "observed_fs_id",
    "root_path",
    "root_dev",
    "root_ino",
    "probe_file_sha256",
)
_FLAGS = os.O_NOFOLLOW | os.O_CLOEXEC
_JSON = SimpleNamespace(max_metadata_bytes=128 << 10, max_nodes=100000, max_depth=24)


class BoundaryStoreError(ValueError):
    def __init__(self, code, detail):
        self.code, self.detail = code, str(detail)[:2048]
        super().__init__(f"{code}: {self.detail}")


class BoundaryStoreIndeterminate(OSError):
    """HEAD visibility/durability is uncertain; close and explicitly recover."""


def _fail(code, detail):
    raise BoundaryStoreError(code, detail)


@dataclasses.dataclass(frozen=True)
class StorageContract:
    """Calling controller's POSIX assumption, not a hardware power-loss proof."""

    operator_declares_local: bool
    operator_declares_fsync_durable: bool

    def __post_init__(self):
        if (
            type(self) is not StorageContract
            or set(vars(self))
            != {"operator_declares_local", "operator_declares_fsync_durable"}
            or self.operator_declares_local is not True
            or self.operator_declares_fsync_durable is not True
        ):
            _fail(
                "storage_contract",
                "two explicit plain true POSIX/fsync declarations required",
            )


@dataclasses.dataclass(frozen=True)
class BoundaryStoreLimits:
    max_store_bytes: int = 1 << 30
    max_entries: int = 4096
    max_accepted: int = 9
    max_orphans: int = 4
    free_reserve_bytes: int = 1 << 30
    free_reserve_inodes: int = 256
    max_memory_bytes: int = 6 << 30
    max_head_bytes: int = 32 << 10
    max_generation_bytes: int = 128 << 10
    max_intent_bytes: int = 32 << 10
    max_runtime_metadata_bytes: int = 8 << 20
    image: b.BoundaryImageLimits = dataclasses.field(
        default_factory=b.BoundaryImageLimits
    )

    def __post_init__(self):
        names = tuple(f.name for f in dataclasses.fields(BoundaryStoreLimits))
        if type(self) is not BoundaryStoreLimits or set(vars(self)) != set(names):
            _fail("limit_layout", "exact closed BoundaryStoreLimits required")
        caps = (
            1 << 30,
            4096,
            9,
            4,
            1 << 30,
            256,
            6 << 30,
            32 << 10,
            128 << 10,
            32 << 10,
            8 << 20,
        )
        for name, cap in zip(names[:-1], caps, strict=True):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= cap:
                _fail("limit_cap", name)
        if self.free_reserve_bytes != 1 << 30 or self.free_reserve_inodes != 256:
            _fail("limit_cap", "v1 retains the full1GiB/256inode free reserve")
        b.BoundaryImageLimits.__post_init__(self.image)


_DEFAULT = BoundaryStoreLimits()


def _keys(value, names, where):
    if type(value) is not dict or set(value) != set(names):
        _fail("metadata_layout", where)


def _digest(value, where):
    if type(value) is not str or not _SHA.fullmatch(value):
        _fail("metadata_digest", where)
    return value


def _integer(value, cap, where):
    if type(value) is not int or not 0 <= value <= cap:
        _fail("metadata_integer", where)
    return value


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _seal(value, field):
    value = dict(value)
    value[field] = _sha(oi._json(value))
    return value


def _check_seal(value, names, field, schema):
    _keys(value, names, schema)
    if value["schema"] != schema:
        _fail("metadata_schema", schema)
    _digest(value[field], field)
    if _sha(oi._json({k: v for k, v in value.items() if k != field})) != value[field]:
        _fail("metadata_seal", field)


def _control_fields(value, *, intent=False, generation=False):
    """Check wire kinds before equality (bool must never equal an integer)."""
    if type(value["store_id"]) is not str or not _NONCE.fullmatch(value["store_id"]):
        _fail("metadata_nonce", "store id")
    _integer(value["sequence"], 8, "sequence")
    _digest(value["operation_id"], "operation id")
    if (intent or generation) and (
        type(value["attempt_id"]) is not str
        or not _NONCE.fullmatch(value["attempt_id"])
    ):
        _fail("metadata_nonce", "attempt id")
    if not intent:
        _digest(value["store_file_sha256"], "store raw hash")
    if generation:
        _digest(value["intent_file_sha256"], "intent raw hash")
        parent_hash, parent_name = (
            value["parent_head_file_sha256"],
            value["parent_generation"],
        )
        if value["receipt_bytes_sha256"] is not None:
            _digest(value["receipt_bytes_sha256"], "receipt raw hash")
    elif intent:
        oi._source_identity(value["source_identity"])
        _digest(value["environment_sha256"], "environment")
        parent_hash, parent_name = (
            value["expected_parent_head_file_sha256"],
            value["expected_parent_generation"],
        )
    else:
        _digest(value["generation_file_sha256"], "generation raw hash")
        _digest(value["generation_sha256"], "generation internal seal")
        return
    if parent_hash is not None:
        _digest(parent_hash, "parent raw hash")
    if parent_name is not None and (
        type(parent_name) is not str or not _GEN.fullmatch(parent_name)
    ):
        _fail("metadata_nonce", "parent generation")


def _token(head):
    return _sha(
        oi._json(
            {"store_id": head["store_id"], "head_file_sha256": _sha(oi._json(head))}
        )
    )


def _operation(store_id, parent, index, schedule_sha):
    return _sha(
        oi._json(
            [
                store_id,
                None if parent is None else _sha(oi._json(parent)),
                index,
                schedule_sha,
            ]
        )
    )


def _identity(st):
    return st.st_dev, st.st_ino



# Fresh directory-currentness resolution; original acquisition walkers stay intact.
class _OpenHow(ctypes.Structure):
    _fields_ = (
        ("flags", ctypes.c_uint64),
        ("mode", ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    )


_ABI_LOCK = threading.Lock()
_ABI = None


def _abi_profile():
    return (
        sys.platform == "linux"
        and getattr(sys.implementation, "_multiarch", None) == "x86_64-linux-gnu"
        and ctypes.sizeof(ctypes.c_long) == ctypes.sizeof(ctypes.c_void_p) == 8
        and ctypes.sizeof(_OpenHow) == 24
    )


def _abi_setup():
    global _ABI
    with _ABI_LOCK:
        if _ABI is None:
            owner = ctypes.pythonapi
            lib = ctypes.CDLL(None, handle=owner._handle, use_errno=False)
            fn = lib["syscall"]
            fn.argtypes = (
                ctypes.c_long,
                ctypes.c_long,
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            )
            fn.restype = ctypes.c_long
            _ABI = (owner, lib, fn)
        return _ABI


def _open_current_directory(path):
    p = Path(path)
    if not (
        _abi_profile()
        and p.is_absolute()
        and p.anchor == "/"
        and len(p.parts) > 1
        and ".." not in p.parts
    ):
        return oi._open_dir(p)
    try:
        encoded = os.fsencode(str(p))
    except Exception:
        encoded = None
    if encoded is None or b"\0" in encoded or len(encoded) > 4095:
        return oi._open_dir(p)
    abi = _ABI
    if abi is None:
        try:
            abi = _abi_setup()
        except Exception:
            abi = None
        if abi is None:
            return oi._open_dir(p)
    try:
        reserve = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except Exception:
        reserve = None
    if reserve is None:
        return oi._open_dir(p)
    result = None
    failure = None
    try:
        how = _OpenHow(os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC, 0, 0x04)
        raw = abi[2](437, -100, encoded, ctypes.byref(how), 24)
        if raw >= 0:
            result = raw
        held, reserve = reserve, None
        os.close(held)
    except BaseException as error:
        failure = error
    if failure is not None:
        errors = [failure]
        held, reserve = reserve, None
        if held is not None:
            try:
                os.close(held)
            except BaseException as error:
                errors.append(error)
        held, result = result, None
        if held is not None:
            try:
                os.close(held)
            except BaseException as error:
                errors.append(error)
        if len(errors) == 1:
            raise failure
        raise BaseExceptionGroup("directory currentness cleanup failed", errors)
    if result is None:
        return oi._open_dir(p)
    descriptor, result = result, None
    return descriptor


def _same_directory(path, descriptor):
    other = _open_current_directory(path)
    try:
        if _identity(os.fstat(other)) != _identity(os.fstat(descriptor)):
            _fail("identity_drift", "held ancestor no longer names directory")
    finally:
        os.close(other)


def _read(path, cap, *, links=1, _directory=None, _with_identity=False):
    """Finite C metadata only; runtime owns no-follow/stability byte reading."""
    path = Path(path)
    held = oi._open_dir(path.parent) if _directory is None else os.dup(_directory)
    try:
        info = os.stat(path.name, dir_fd=held, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != links:
            _fail("metadata_file", "regular metadata with exact link count required")
        if info.st_size > cap:
            _fail("metadata_cap", "raw cap before hash/JSON")
        _same_directory(path.parent, held)
        raw = runtime._read_bounded(path, cap, "boundary_metadata_read")
        _same_directory(path.parent, held)
        post = os.stat(path.name, dir_fd=held, follow_symlinks=False)
        if (
            oi._stat_id(post) != oi._stat_id(info)
            or post.st_nlink != links
            or not stat.S_ISREG(post.st_mode)
        ):
            _fail("identity_drift", "metadata changed around delegated read")
        return (raw, oi._stat_id(post)) if _with_identity else raw
    except runtime.RuntimeRefusal as exc:
        raise BoundaryStoreError(exc.code, exc.detail) from exc
    finally:
        os.close(held)


def _record(path, cap, *, links=1):
    raw = _read(path, cap, links=links)
    view = SimpleNamespace(max_metadata_bytes=cap, max_nodes=100000, max_depth=24)
    value = oi._strict_json(raw, view)
    if raw != oi._json(value):
        _fail("metadata_noncanonical", "canonical bytes without newline required")
    return value, raw


def _write(path, raw, *, _directory=None):
    parent = (
        oi._open_dir(Path(path).parent) if _directory is None else os.dup(_directory)
    )
    try:
        _same_directory(Path(path).parent, parent)
        fd = os.open(
            Path(path).name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FLAGS,
            0o600,
            dir_fd=parent,
        )
        try:
            view = memoryview(raw)
            while view:
                n = os.write(fd, view)
                if n <= 0:
                    raise OSError("short metadata write")
                view = view[n:]
            os.fsync(fd)
        finally:
            os.close(fd)
        _same_directory(Path(path).parent, parent)
    finally:
        os.close(parent)


def _sealed_write(directory, name, value, cap):
    raw = oi._json(value)
    if len(raw) > cap:
        _fail("metadata_cap", "writer metadata cap")
    temporary = name.removesuffix(".json") + ".tmp"
    held = oi._open_dir(directory)
    try:
        _write(directory / temporary, raw, _directory=held)
        readback, read_identity = _read(
            directory / temporary, cap, _directory=held, _with_identity=True
        )
        if readback != raw:
            _fail("metadata_readback", name)
        verified = os.stat(temporary, dir_fd=held, follow_symlinks=False)
        if oi._stat_id(verified) != read_identity:
            _fail("identity_drift", "seal inode changed after exact readback")
        _same_directory(directory, held)
        os.link(
            temporary, name, src_dir_fd=held, dst_dir_fd=held, follow_symlinks=False
        )
        a = os.stat(temporary, dir_fd=held, follow_symlinks=False)
        z = os.stat(name, dir_fd=held, follow_symlinks=False)
        if (
            _identity(a) != _identity(verified)
            or _identity(a) != _identity(z)
            or not stat.S_ISREG(a.st_mode)
            or not stat.S_ISREG(z.st_mode)
            or a.st_nlink != 2
            or z.st_nlink != 2
            or a.st_size != len(raw)
            or z.st_size != len(raw)
            or a.st_mtime_ns != verified.st_mtime_ns
            or z.st_mtime_ns != verified.st_mtime_ns
        ):
            _fail("identity_drift", "seal link differs from verified inode")
        os.fsync(held)
        os.unlink(temporary, dir_fd=held)
        os.fsync(held)
        _same_directory(directory, held)
    finally:
        os.close(held)
    return _sha(raw)


def _published_links(directory, name):
    temporary = name.removesuffix(".json") + ".tmp"
    held = oi._open_dir(directory)
    try:
        try:
            z = os.stat(temporary, dir_fd=held, follow_symlinks=False)
        except FileNotFoundError:
            return 1
        try:
            a = os.stat(name, dir_fd=held, follow_symlinks=False)
        except FileNotFoundError:
            _fail(
                "manifest_unsealed",
                "only pre-link temp exists; never promote or retry destination",
            )
        if (
            not stat.S_ISREG(a.st_mode)
            or not stat.S_ISREG(z.st_mode)
            or _identity(a) != _identity(z)
            or a.st_nlink != 2
            or z.st_nlink != 2
        ):
            _fail(
                "manifest_leftover", "only exact own sealed manifest hardlink allowed"
            )
        _same_directory(directory, held)
        return 2
    finally:
        os.close(held)


def _sync_tree(root):
    """Already admitted finite tree, child-to-parent; no deletion or model parsing."""
    held = oi._open_dir(root)
    try:
        with os.scandir(held) as iterator:
            for entry in iterator:
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    _sync_tree(root / entry.name)
                elif stat.S_ISREG(info.st_mode):
                    fd = os.open(
                        entry.name, os.O_RDONLY | _FLAGS | os.O_NONBLOCK, dir_fd=held
                    )
                    try:
                        if oi._stat_id(os.fstat(fd)) != oi._stat_id(info):
                            _fail("identity_drift", "sync file changed")
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                else:
                    _fail("path_refused", "nonregular tree entry")
        _same_directory(root, held)
        os.fsync(held)
    finally:
        os.close(held)


def _durable_mount(value, where):
    """Closed typed observation; namespace-local mount_id is diagnosis only."""
    _keys(value, ("mount_id", "mount_path", "major_minor", "fs_type"), where)
    if (
        type(value["mount_id"]) is not int
        or not 1 <= value["mount_id"] < 1 << 63
        or any(
            type(value[name]) is not str
            for name in ("mount_path", "major_minor", "fs_type")
        )
    ):
        _fail("storage_observation", where)
    return value["mount_path"], value["major_minor"], value["fs_type"]


def _mount_unescape(value):
    if re.search(r"\\(?!040|011|012|134)", value):
        _fail("storage_observation", "unsupported mountinfo escape")
    return re.sub(r"\\(040|011|012|134)", lambda m: chr(int(m[1], 8)), value)


def _mount_observation(path, device, raw=None):
    if not Path(path).is_absolute() or os.path.normpath(str(path)) != str(path):
        _fail("storage_observation", "canonical absolute root required")
    if raw is None:
        fd = os.open("/proc/self/mountinfo", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            chunks, used = [], 0
            while True:
                part = os.read(fd, min(65536, (2 << 20) + 1 - used))
                if not part:
                    break
                chunks.append(part)
                used += len(part)
                if used > 2 << 20:
                    _fail("storage_observation", "mountinfo raw cap")
            raw = b"".join(chunks)
        finally:
            os.close(fd)
    if type(raw) is not bytes or len(raw) > 2 << 20:
        _fail("storage_observation", "mountinfo raw cap/type")
    lines = raw.splitlines()
    if not lines or len(lines) > 8192 or any(len(line) > 8192 for line in lines):
        _fail("storage_observation", "mountinfo line cap")
    candidates = []
    for line in lines:
        try:
            left, right = line.decode("utf-8", "strict").split(" - ")
            a, z = left.split(), right.split()
            if (
                len(a) < 6
                or len(z) != 3
                or not a[0].isdigit()
                or not re.fullmatch(r"[0-9]+:[0-9]+", a[2])
            ):
                raise ValueError("fields")
            mount = _mount_unescape(a[4])
            if not mount.startswith("/"):
                raise ValueError("mount path")
            if str(path) == mount or str(path).startswith(mount.rstrip("/") + "/"):
                candidates.append(
                    (
                        len(mount),
                        {
                            "mount_id": int(a[0]),
                            "mount_path": mount,
                            "major_minor": a[2],
                            "fs_type": z[0],
                        },
                    )
                )
        except (ValueError, UnicodeError) as exc:
            raise BoundaryStoreError(
                "storage_observation", "malformed mountinfo"
            ) from exc
    if not candidates:
        _fail("storage_observation", "no enclosing mount")
    longest = max(x[0] for x in candidates)
    matches = [x[1] for x in candidates if x[0] == longest]
    if (
        len(matches) != 1
        or matches[0]["major_minor"] != f"{os.major(device)}:{os.minor(device)}"
    ):
        _fail("storage_observation", "ambiguous mount/device mismatch")
    return matches[0]


def _probe(root, rootfd):
    observation = _mount_observation(root, os.fstat(rootfd).st_dev)
    if observation["fs_type"] not in ("ext4", "xfs"):
        _fail("storage_unsupported", observation["fs_type"])
    probe = root / ".storage_probe"
    _same_directory(root, rootfd)
    os.mkdir(".storage_probe", 0o700, dir_fd=rootfd)
    fd = oi._open_dir(probe)
    try:
        _write(probe / "original", b"bounded-posix-probe-v1")
        os.link(
            "original", "linked", src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False
        )
        a = os.stat("original", dir_fd=fd, follow_symlinks=False)
        z = os.stat("linked", dir_fd=fd, follow_symlinks=False)
        if (
            _identity(a) != _identity(z)
            or a.st_nlink != 2
            or z.st_nlink != 2
            or _read(probe / "linked", 1024, links=2) != b"bounded-posix-probe-v1"
        ):
            _fail("storage_probe", "hardlink inode/bytes/count not observed")
        _write(probe / "replacement", b"bounded-replace-probe-v1")
        os.replace("replacement", "linked", src_dir_fd=fd, dst_dir_fd=fd)
        if (
            _read(probe / "original", 1024) != b"bounded-posix-probe-v1"
            or _read(probe / "linked", 1024) != b"bounded-replace-probe-v1"
        ):
            _fail("storage_probe", "replace/readback not observed")
        os.fsync(fd)
        info = os.fstatvfs(rootfd)
        receipt = {
            "schema": SCHEMA + ".probe",
            "observation": observation,
            "f_fsid": info.f_fsid,
            "root_dev": os.fstat(rootfd).st_dev,
            "root_ino": os.fstat(rootfd).st_ino,
            "root_path": str(root),
            "primitives": [
                "exclusive_mkdir",
                "exclusive_file",
                "hardlink",
                "replace",
                "file_fsync",
                "directory_fsync",
            ],
            "scope": "observed POSIX calls; no hardware power-loss proof",
        }
        _write(probe / "result.json", oi._json(receipt))
        if _read(probe / "result.json", 32 << 10) != oi._json(receipt):
            _fail("storage_probe", "result readback differs")
        _same_directory(probe, fd)
        _same_directory(root, rootfd)
        os.fsync(fd)
        os.fsync(rootfd)
        return receipt, _sha(oi._json(receipt))
    finally:
        os.close(fd)


def _inventory(root, limits, *, expected_root=None):
    entries, used, generations, temps = 0, 0, set(), set()

    def walk(path, relative, expected=None, depth=0):
        nonlocal entries, used
        if depth > 3:
            _fail("path_refused", "finite store directory depth")
        held = oi._open_dir(path)
        try:
            if expected is not None and _identity(os.fstat(held)) != _identity(
                expected
            ):
                _fail("identity_drift", "inventory child differs from admitted entry")
            local = {}
            with os.scandir(held) as iterator:
                for entry in iterator:
                    entries += 1
                    if entries > limits.max_entries:
                        _fail("entry_cap", "store inventory before hashing")
                    name, info = entry.name, entry.stat(follow_symlinks=False)
                    local[name] = info
                    want_directory, cap = False, limits.max_store_bytes
                    if relative == "":
                        match = _TEMP.fullmatch(name)
                        if (
                            name
                            not in {
                                "STORE.json",
                                "STORE.tmp",
                                ".writer.lock",
                                "HEAD",
                                "originals",
                                "generations",
                                ".storage_probe",
                            }
                            and not match
                        ):
                            _fail("path_refused", "unexpected store entry")
                        if match:
                            temps.add(match[1])
                        want_directory = name in {
                            "originals",
                            "generations",
                            ".storage_probe",
                        }
                        cap = (
                            0
                            if name == ".writer.lock"
                            else (
                                128 << 10
                                if name.startswith("STORE.")
                                else limits.max_head_bytes
                            )
                        )
                    elif relative == "generations":
                        match = _GEN.fullmatch(name)
                        if (
                            not match
                            or not stat.S_ISDIR(info.st_mode)
                            or int(match[1]) > 8
                        ):
                            _fail("path_refused", "generation name/type")
                        generations.add(name)
                        want_directory = True
                    elif re.fullmatch(r"generations/g-[0-9]{4}-[0-9a-f]{32}", relative):
                        if name not in {
                            "intent.json",
                            "physical",
                            "image",
                            "generation.json",
                            "generation.tmp",
                            "failure.json",
                        }:
                            _fail("path_refused", "unexpected generation entry")
                        want_directory = name in {"physical", "image"}
                        cap = (
                            limits.max_generation_bytes
                            if name.startswith("generation.")
                            else limits.max_intent_bytes
                        )
                    elif relative == "originals":
                        if not re.fullmatch(
                            r"o-[0-9a-f]{32}", name
                        ) or not stat.S_ISDIR(info.st_mode):
                            _fail("path_refused", "original pack name/type")
                        want_directory = True
                    elif relative == ".storage_probe":
                        if name not in {"original", "linked", "result.json"}:
                            _fail("path_refused", "probe inventory")
                        cap = 32 << 10
                    elif re.fullmatch(r"originals/o-[0-9a-f]{32}", relative):
                        if name not in {
                            "manifest.json",
                            "manifest.tmp",
                        } and not re.fullmatch(r"a_[0-9]{4}\.npy", name):
                            _fail("path_refused", "original child inventory")
                        cap = (
                            4 << 20
                            if name.startswith("manifest.")
                            else (64 << 20) + 4096
                        )
                    elif re.fullmatch(
                        r"generations/g-[0-9]{4}-[0-9a-f]{32}/image", relative
                    ):
                        if name not in {
                            "manifest.json",
                            "manifest.tmp",
                        } and not re.fullmatch(r"a_000[0-6]\.npy", name):
                            _fail("path_refused", "wrapper child inventory")
                        cap = (
                            limits.image.max_metadata_bytes
                            if name.startswith("manifest.")
                            else limits.image.max_numeric_bytes
                            + limits.image.max_header_bytes
                        )
                    elif re.fullmatch(
                        r"generations/g-[0-9]{4}-[0-9a-f]{32}/physical", relative
                    ):
                        if name not in {
                            "checkpoint.json",
                            "native_state.npz",
                            "pending_state.npz",
                        } and not re.fullmatch(r"\.manifest-[0-9a-f]{32}", name):
                            _fail("path_refused", "native child inventory")
                        cap = (
                            (1 << 20)
                            if not name.endswith(".npz")
                            else _native_reservation(limits)
                        )
                    else:
                        _fail("path_refused", "unknown store directory role")
                    if want_directory:
                        if not stat.S_ISDIR(info.st_mode):
                            _fail("path_refused", "known directory has wrong type")
                        walk(
                            path / name,
                            name if not relative else relative + "/" + name,
                            info,
                            depth + 1,
                        )
                    elif stat.S_ISREG(info.st_mode):
                        if info.st_nlink not in (1, 2):
                            _fail(
                                "file_links", "foreign hardlink count in retained store"
                            )
                        if info.st_size > cap:
                            _fail("metadata_cap", "role raw size cap before hash/parse")
                        used += info.st_size
                        if used > limits.max_store_bytes:
                            _fail(
                                "store_raw_cap",
                                "aggregate serialized cap before hashing",
                            )
                    else:
                        _fail("path_refused", "symlink/special entry")
            for name, info in local.items():
                if not stat.S_ISREG(info.st_mode):
                    continue
                partner = {
                    "STORE.json": "STORE.tmp",
                    "STORE.tmp": "STORE.json",
                    "generation.json": "generation.tmp",
                    "generation.tmp": "generation.json",
                    "manifest.json": "manifest.tmp",
                    "manifest.tmp": "manifest.json",
                }.get(name)
                if name == "checkpoint.json":
                    matches = [
                        key
                        for key in local
                        if re.fullmatch(r"\.manifest-[0-9a-f]{32}", key)
                    ]
                    partner = matches[0] if len(matches) == 1 else None
                elif re.fullmatch(r"\.manifest-[0-9a-f]{32}", name):
                    partner = "checkpoint.json"
                other = local.get(partner)
                if (
                    info.st_nlink == 2 or (other is not None and partner is not None)
                ) and (
                    other is None
                    or not stat.S_ISREG(other.st_mode)
                    or info.st_nlink != 2
                    or other.st_nlink != 2
                    or _identity(info) != _identity(other)
                ):
                    _fail(
                        "manifest_leftover",
                        "only exact own manifest hardlink pair allowed",
                    )
            _same_directory(path, held)
        finally:
            os.close(held)

    walk(root, "", expected_root)
    nonces = {name.rsplit("-", 1)[1] for name in generations}
    if not temps <= nonces:
        _fail("path_refused", "HEAD temp lacks matching retained attempt")
    return {
        "entries": entries,
        "bytes": used,
        "generations": generations,
        "temps": temps,
    }


_CL_KIND = "committed_cpu_lock"
_CL_OBJECT_BYTES = 1 << 20
_CL_LOCK_BYTES = min(_DEFAULT.max_runtime_metadata_bytes, 4 << 20)
_CL_SOURCE_FILES = 2000
_CL_PATH_DEPTH = 16
_CL_SOURCE_FILE_BYTES = 16 << 20
_CL_OVERLAY_FILES = 64
_CL_OVERLAY_FILE_BYTES = 1 << 20
_CL_AGGREGATE_BYTES = 128 << 20
_CL_INTERPRETER_BYTES = 256 << 20
_CL_NATIVE_MANIFEST = "bundle-manifest.json"
# Outer source-object manifest layout; identity semantics stay runtime-owned.
_CL_OBJECT_KEYS = ("schema", "object_id", "identity", "overlay_files")


def _json_string_bound(text):
    """Upper bound of one quoted JSON string under any escape/UTF-8 policy."""
    if text.isascii() and text.isprintable():
        return len(text) + text.count('"') + text.count("\\") + 2
    total = 2
    for char in text:
        point = ord(char)
        if point > 0xFFFF:
            total += 12  # surrogate-pair escape; raw UTF-8 is at most 4
        elif point < 0x20 or point >= 0x7F:
            total += 6  # \uXXXX (also lone surrogates); raw UTF-8 is at most 3
        elif char == '"' or char == "\\":
            total += 2
        else:
            total += 1
    return total


def _plain_metadata(value, max_bytes, where, *, max_nodes=100000, max_depth=24):
    """Type/size admission of caller objects before owning validators; not a parser."""
    nodes, wire, active = 0, 0, set()

    def visit(item, depth):
        nonlocal nodes, wire
        nodes += 1
        if nodes > max_nodes or depth > max_depth:
            _fail("metadata_cap", f"{where}: node/depth cap")
        kind = type(item)
        if kind is dict or kind is list:
            if id(item) in active:
                _fail("metadata_invalid", f"{where}: cyclic metadata")
            active.add(id(item))
            wire += 2
            if kind is dict:
                for key, child in item.items():
                    if type(key) is not str or len(key) > 4096:
                        _fail("metadata_invalid", f"{where}: invalid key")
                    # key, ": " and ", " under either compact or default separators
                    wire += _json_string_bound(key) + 4
                    if wire > max_bytes:
                        _fail("metadata_cap", f"{where}: aggregate wire size")
                    visit(child, depth + 1)
            else:
                for child in item:
                    wire += 2
                    visit(child, depth + 1)
            active.discard(id(item))
        elif kind is str:
            if len(item) > 4096:
                _fail("metadata_cap", f"{where}: string cap")
            wire += _json_string_bound(item)
        elif kind is int:
            if item.bit_length() > 4096:
                _fail("metadata_cap", f"{where}: integer cap")
            # decimal digits <= floor(bits * log10(2)) + 1, plus sign and slack
            wire += item.bit_length() * 1233 // 4096 + 3
        elif kind is float:
            if not math.isfinite(item):
                _fail("metadata_nonfinite", f"{where}: nonfinite number")
            wire += 24  # longest float repr, e.g. -2.2250738585072014e-308
        elif item is not None and kind is not bool:
            _fail("metadata_invalid", f"{where}: non-JSON value")
        else:
            wire += 5
        # Conservative upper wire bound (no indentation); exact canonical length
        # is still checked afterwards by the owning serializer.
        if wire > max_bytes:
            _fail("metadata_cap", f"{where}: aggregate wire size")

    visit(value, 0)
    return value


def _canonical_path(value, where):
    if (
        type(value) is not str
        or not value
        or len(value) > 4096
        or "\x00" in value
        or not os.path.isabs(value)
        or os.path.normpath(value) != value
    ):
        _fail("runtime_reference", f"{where}: canonical absolute lexical path required")
    return value


def committed_runtime_reference(*, cpu_store, cpu_lock_id, cpu_lock_file_sha256, lock):
    """Pure committed-lock runtime reference; performs no filesystem or native IO."""
    _canonical_path(cpu_store, "cpu store")
    _digest(cpu_lock_id, "cpu lock id")
    _digest(cpu_lock_file_sha256, "cpu lock raw hash")
    _plain_metadata(lock, _CL_LOCK_BYTES, "cpu lock")
    try:
        runtime.validate_lock(lock, store=cpu_store, lock_name=cpu_lock_id + ".json")
        raw = runtime._canonical_bytes(lock)
    except runtime.RuntimeRefusal as exc:
        raise BoundaryStoreError(exc.code, exc.detail) from exc
    if lock["lock_id"] != cpu_lock_id:
        _fail("runtime_reference", "lock id differs from declared cpu lock id")
    if len(raw) > _CL_LOCK_BYTES:
        _fail("metadata_cap", "canonical cpu lock bytes")
    if _sha(raw) != cpu_lock_file_sha256:
        _fail("runtime_reference", "canonical cpu lock raw hash differs")
    prefix = lock["runtime"]["sys_path_prefix"]
    if type(prefix) is not list or len(prefix) < 6:
        _fail("runtime_reference", "sealed sys.path prefix lacks module roots")
    roots = {
        "masy": _canonical_path(prefix[3], "masy module root"),
        "maple_syrup": _canonical_path(prefix[4], "syrup module root"),
        "maple": _canonical_path(prefix[5], "maple module root"),
    }
    if len(set(roots.values())) != 3:
        _fail("runtime_reference", "module roots must be distinct")
    bundle = lock["objects"]["maple_bundle"]
    native_path = os.path.join(
        _canonical_path(bundle["path"], "native bundle"), _CL_NATIVE_MANIFEST
    )
    reference = {
        "schema": SCHEMA + ".runtime",
        "kind": _CL_KIND,
        "source_manifest_path": None,
        "source_manifest_file_sha256": None,
        "native_bundle_manifest_path": native_path,
        "native_bundle_manifest_file_sha256": _digest(
            bundle["manifest_sha256"], "native bundle manifest hash"
        ),
        "module_roots": roots,
        "cpu_store": cpu_store,
        "cpu_lock_id": cpu_lock_id,
        "cpu_lock_file_sha256": cpu_lock_file_sha256,
    }
    _keys(reference, _RUNTIME, "runtime reference")
    return reference


def _overlay_path(path, dist_info):
    if (
        type(path) is not str
        or not path
        or "\\" in path
        or path.startswith("/")
        or any(ord(c) < 32 or ord(c) == 127 for c in path)
        or any(0xD800 <= ord(c) <= 0xDFFF for c in path)
        or len(path.encode("utf-8", "surrogateescape")) > 1024
    ):
        _fail("source_object", "unsafe overlay path")
    parts = path.split("/")
    if (
        len(parts) < 2
        or len(parts) > _CL_PATH_DEPTH
        or parts[0] != dist_info
        or any(p in ("", ".", "..") for p in parts)
    ):
        _fail("source_object", "overlay path escapes its dist-info namespace")
    return path


def _admit_source_object(raw, *, kind, object_id, max_bytes=_CL_OBJECT_BYTES):
    """Bounded raw source-object manifest admission; no file walk/hash/read."""
    _digest(object_id, "source object id")
    if kind not in runtime.SOURCES:
        _fail("source_object", "unknown source object kind")
    cap = min(max_bytes, _CL_OBJECT_BYTES)
    if type(raw) is not bytes or len(raw) > cap:
        _fail("metadata_cap", "source object manifest raw cap")
    try:
        manifest = oi._strict_json(
            raw, SimpleNamespace(max_metadata_bytes=cap, max_nodes=20000, max_depth=8)
        )
    except oi.OriginalInputsError as exc:
        raise BoundaryStoreError(exc.code, exc.detail) from exc
    try:
        if raw != runtime._canonical_bytes(manifest):
            _fail("metadata_noncanonical", "source object manifest bytes")
        _keys(manifest, _CL_OBJECT_KEYS, "source object manifest")
        if manifest["schema"] != runtime.SOURCE_OBJECT_SCHEMA:
            _fail("source_object", "source object manifest schema")
        identity = runtime._validate_identity(manifest["identity"])
        if (
            manifest["object_id"] != object_id
            or runtime.identity_id(identity) != object_id
        ):
            _fail("source_object", "source object id differs")
    except runtime.RuntimeRefusal as exc:
        raise BoundaryStoreError(exc.code, exc.detail) from exc
    if identity["kind"] != kind:
        _fail("source_object", "source object kind differs")
    files, declared = identity["files"], 0
    if len(files) > _CL_SOURCE_FILES:
        _fail("source_cap", "finite source object file count")
    for rel, size, _sha256, _blob in files:
        if len(rel.split("/")) > _CL_PATH_DEPTH:
            _fail("source_cap", "source path depth")
        _integer(size, _CL_SOURCE_FILE_BYTES, "source file bytes")
        declared += size
        if declared > _CL_AGGREGATE_BYTES:
            _fail("source_cap", "source object declared bytes")
    overlay = manifest["overlay_files"]
    # Runtime authority: overlay_files is None exactly when identity.overlay is
    # None (MAPLE); generated sources carry a genuine finite list. The manifest
    # and raw bytes stay untouched; only the returned rows normalize to [].
    if identity["overlay"] is None:
        if overlay is not None:
            _fail("source_object", "overlay files present without overlay identity")
        overlay = []
    elif type(overlay) is not list or len(overlay) > _CL_OVERLAY_FILES:
        _fail("source_object", "finite overlay file list required")
    elif not overlay:
        _fail("source_object", "overlay files do not match overlay presence")
    previous = None
    for row in overlay:
        if type(row) is not list or len(row) != 3:
            _fail("source_object", "overlay row layout")
        path = _overlay_path(row[0], identity["overlay"]["dist_info"])
        if previous is not None and path <= previous:
            _fail("source_object", "overlay rows must be strictly sorted/unique")
        previous = path
        _integer(row[1], _CL_OVERLAY_FILE_BYTES, "overlay file bytes")
        _digest(row[2], "overlay file SHA")
        declared += row[1]
        if declared > _CL_AGGREGATE_BYTES:
            _fail("source_cap", "source object declared bytes")
    return manifest, files, overlay, declared


_INV_PARENTS = 512
_INV_FILES = 10000


class _HeldInventory:
    """Ephemeral held-parent inventory of owned source/native artifact files.

    Quiescent owned-source protocol, not a descriptor-atomic hostile-writer
    proof: parents are held no-follow, delegated reads stay path based, and
    every admission is rechecked by full stat identity. Never writes or fsyncs.
    Not for interpreter symlinks or compiled environment files.
    """

    def __init__(self, byte_budget, *, max_parents=_INV_PARENTS, max_files=_INV_FILES):
        for value, cap, where in (
            (byte_budget, None, "byte budget"),
            (max_parents, _INV_PARENTS, "parent cap"),
            (max_files, _INV_FILES, "file cap"),
        ):
            if type(value) is not int or value < 0 or (cap is not None and value > cap):
                _fail("inventory_invalid", f"held inventory {where}")
        self._budget, self._max_parents, self._max_files = (
            byte_budget,
            max_parents,
            max_files,
        )
        self._parents = {}  # parent path -> (descriptor, directory identity)
        self._files = {}  # path -> (parent, name, size, stat id, nlink, mode)
        self._bytes = 0
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False

    def close(self):
        self._closed = True
        parents, self._parents = self._parents, {}
        self._files = {}
        for descriptor, _identity_ in parents.values():
            with contextlib.suppress(OSError):
                os.close(descriptor)

    def _live(self):
        if self._closed:
            _fail("inventory_closed", "held inventory is closed")

    def _hold(self, parent):
        held = self._parents.get(parent)
        if held is not None:
            return held[0]
        if len(self._parents) >= self._max_parents:
            _fail("inventory_cap", "held parent directory cap")
        try:
            descriptor = oi._open_dir(parent)
        except oi.OriginalInputsError as exc:
            raise BoundaryStoreError(exc.code, exc.detail) from exc
        except OSError as exc:
            raise BoundaryStoreError(
                "inventory_parent", f"{parent}: {exc.strerror or exc}"
            ) from exc
        try:
            self._parents[parent] = (descriptor, _identity(os.fstat(descriptor)))
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    def _lstat(self, parent, name, descriptor):
        try:
            return os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise BoundaryStoreError(
                "inventory_file", f"{os.path.join(parent, name)}: {exc.strerror or exc}"
            ) from exc

    def _check_parent(self, parent):
        descriptor, identity = self._parents[parent]
        if _identity(os.fstat(descriptor)) != identity:
            _fail("identity_drift", "held parent descriptor changed")
        _same_directory(parent, descriptor)
        return descriptor

    def _check_file(self, path):
        parent, name, _size, ident, links, mode = self._files[path]
        descriptor = self._check_parent(parent)
        info = self._lstat(parent, name, descriptor)
        if (
            oi._stat_id(info) != ident
            or info.st_nlink != links
            or info.st_mode != mode
        ):
            _fail("identity_drift", f"{path}: admitted file changed")
        _same_directory(parent, descriptor)

    def admit(self, path, declared_size):
        """Admit one read-only single-link regular file of exact declared size."""
        self._live()
        path = _canonical_path(path, "held inventory path")
        if type(declared_size) is not int or declared_size < 0:
            _fail("inventory_invalid", f"{path}: plain non-negative size required")
        known = self._files.get(path)
        if known is not None:
            if known[2] != declared_size:
                _fail("inventory_conflict", f"{path}: roles disagree on size")
            self._check_file(path)
            return
        if len(self._files) >= self._max_files:
            _fail("inventory_cap", "held inventory file cap")
        if declared_size > self._budget - self._bytes:
            _fail("inventory_cap", f"{path}: byte budget exceeded")
        parent, name = os.path.split(path)
        if not name:
            _fail("runtime_reference", "held inventory path names root")
        descriptor = self._hold(parent)
        _same_directory(parent, descriptor)
        info = self._lstat(parent, name, descriptor)
        if not stat.S_ISREG(info.st_mode):
            _fail("inventory_file", f"{path}: regular file required")
        if info.st_mode & 0o222:
            _fail("inventory_file", f"{path}: read-only file required")
        if info.st_nlink != 1:
            _fail("inventory_file", f"{path}: single link required")
        if info.st_size != declared_size:
            _fail("inventory_file", f"{path}: size differs from declaration")
        _same_directory(parent, descriptor)
        self._files[path] = (
            parent,
            name,
            declared_size,
            oi._stat_id(info),
            info.st_nlink,
            info.st_mode,
        )
        self._bytes += declared_size

    def _check_held_file(self, path):
        """Recheck one file through its held parent between check() passes."""
        parent, name, _size, ident, links, mode = self._files[path]
        descriptor = self._parents[parent][0]
        info = self._lstat(parent, name, descriptor)
        if (
            oi._stat_id(info) != ident
            or info.st_nlink != links
            or info.st_mode != mode
        ):
            _fail("identity_drift", f"{path}: admitted file changed")

    def check(self):
        """Recheck parents, held-file stats, then current parent paths again.

        This is a quiescent owned-source batch, not an atomic hostile-writer
        snapshot. Ordinary per-file read/admit checks retain their own guards.
        """
        self._live()
        for parent in self._parents:
            self._check_parent(parent)
        for path in self._files:
            self._check_held_file(path)
        for parent in self._parents:
            self._check_parent(parent)

    def read(self, path, cap, declared_size=None):
        """Bounded read of an admitted (or now admitted control) file."""
        self._live()
        path = _canonical_path(path, "held inventory path")
        if type(cap) is not int or cap < 0:
            _fail("inventory_invalid", f"{path}: plain read cap required")
        if declared_size is not None and (
            type(declared_size) is not int or declared_size < 0
        ):
            _fail("inventory_invalid", f"{path}: plain non-negative size required")
        known = self._files.get(path)
        if known is None:
            if declared_size is None:
                parent, name = os.path.split(path)
                if not name:
                    _fail("runtime_reference", "held inventory path names root")
                descriptor = self._hold(parent)
                declared_size = self._lstat(parent, name, descriptor).st_size
            if declared_size > cap:
                _fail("metadata_cap", f"{path}: raw cap before admission")
            self.admit(path, declared_size)
            known = self._files[path]
        elif declared_size is not None and declared_size != known[2]:
            _fail("inventory_conflict", f"{path}: roles disagree on size")
        if known[2] > cap:
            _fail("metadata_cap", f"{path}: raw cap before read")
        descriptor = self._check_parent(known[0])
        raw = _read(path, cap, links=1, _directory=descriptor)
        self._check_file(path)
        if len(raw) != known[2]:
            _fail("identity_drift", f"{path}: read length differs from admission")
        return raw


def _committed_json(raw, cap, where):
    """Reuse the bounded original-input decoder for control metadata only."""
    try:
        value = oi._strict_json(
            raw,
            SimpleNamespace(max_metadata_bytes=cap, max_nodes=100000, max_depth=24),
        )
    except oi.OriginalInputsError as exc:
        raise BoundaryStoreError(exc.code, exc.detail) from exc
    _plain_metadata(value, cap, where)
    return value


def _committed_cpu_flags():
    # In particular refuse before importing a native delegate when -B is absent.
    flags = {
        "isolated": sys.flags.isolated,
        "dont_write_bytecode": sys.flags.dont_write_bytecode,
        "ignore_environment": sys.flags.ignore_environment,
        "no_user_site": sys.flags.no_user_site,
        "safe_path": getattr(sys.flags, "safe_path", None),
        "utf8_mode": sys.flags.utf8_mode,
    }
    if any(flags[k] != 1 for k in (
        "isolated", "dont_write_bytecode", "ignore_environment", "no_user_site"
    )) or flags["safe_path"] is not True:
        _fail("committed_runtime", "current process requires isolated -I -B flags")
    expected = {"CUDA_VISIBLE_DEVICES": "", "NUMBA_DISABLE_CUDA": "1", "JAX_PLATFORMS": "cpu"}
    if any(os.environ.get(k) != v for k, v in expected.items()):
        _fail("committed_runtime", "current process must explicitly hide accelerators")
    return flags


def _committed_sources(lock, held, cap):
    """Admit complete source/overlay inventories before any full-file delegate."""
    admitted, controls, total = {}, {}, 0
    for kind in ("maple", "syrup", "masy"):
        spec = lock["objects"]["sources"][kind]
        path = Path(spec["path"])
        filename = str(path / "manifest.json")
        raw = held.read(filename, min(cap, _CL_OBJECT_BYTES))
        if _sha(raw) != spec.get("manifest_sha256"):
            _fail("source_identity", f"{kind} source manifest differs from lock")
        manifest, files, overlay, declared = _admit_source_object(
            raw, kind=kind, object_id=spec["object_id"], max_bytes=cap
        )
        total += declared
        if total > _CL_AGGREGATE_BYTES:
            _fail("source_cap", "three-source plus overlay aggregate byte cap")
        for rel, size, _digest_, _blob in files:
            held.admit(str(path / "source" / rel), size)
        for rel, size, _digest_ in overlay:
            held.admit(str(path / "editable" / rel), size)
        admitted[kind] = (manifest, files, overlay)
        controls[filename] = raw
    return admitted, controls


def _committed_native(lock, reference, admitted, held, cap, native):
    """Bounded native metadata admission; native owns every schema/overlay rule."""
    bundle, materialized = (
        lock["objects"]["maple_bundle"], lock["objects"]["maple_materialized"]
    )
    bundle_path, target = Path(bundle["path"]), Path(materialized["path"])
    limits = native.Limits(
        max_files=4000, max_file_bytes=16 << 20, max_total_bytes=64 << 20,
        max_manifest_bytes=cap, max_origin_bytes=64 << 10,
    )
    filename = str(bundle_path / native.MANIFEST_NAME)
    mfilename = str(target / native.MANIFEST_NAME)
    raw, mraw = held.read(filename, cap), held.read(mfilename, cap)
    if (
        _sha(raw) != bundle["manifest_sha256"]
        or _sha(raw) != reference["native_bundle_manifest_file_sha256"]
        or _sha(mraw) != materialized.get("manifest_sha256")
        or mraw != raw
    ):
        _fail("native_binding", "native bundle/materialized manifest pins differ")
    record = _committed_json(raw, cap, "native bundle manifest")
    table = native._validate_manifest(record, limits)
    if native._canonical(record) + b"\n" != raw:
        _fail("native_binding", "noncanonical native manifest")
    archive = record["archive"]
    held.admit(str(bundle_path / native.ARCHIVE_NAME), archive["size"])
    for rel, row in table.items():
        if len(rel.split("/")) - 1 > _CL_PATH_DEPTH:
            _fail("source_cap", "native materialized path depth")
        held.admit(str(target / rel), row["size"])
    held.read(str(target / native.MATERIALIZATION_NAME), 64 << 10)
    # Only the native's small distribution metadata is read here. Its owning
    # pure generator supplies the actual path-dependent MAPLE overlay bytes.
    dist = {}
    for name in record["distribution"]["files"]:
        row = table["distribution/" + name]
        data = held.read(str(target / "distribution" / name), 1 << 20, row["size"])
        if _sha(data) != row["sha256"]:
            _fail("native_binding", "native distribution metadata bytes differ")
        dist[name] = data
    overlay = native._overlay_files(record, target, dist)
    if type(overlay) is not dict or len(overlay) > _CL_OVERLAY_FILES:
        _fail("source_cap", "native editable file cap")
    overlay_map = {}
    for rel, data in overlay.items():
        # Names and bytes come only from the actual pure native generator.
        if type(data) is not bytes or len(data) > _CL_OVERLAY_FILE_BYTES:
            _fail("source_cap", "native editable byte cap")
        held.admit(str(target / "editable" / rel), len(data))
        overlay_map[rel] = _sha(data)
    if overlay_map != lock["distributions"]["maple"]["files"]:
        _fail("native_binding", "native generated MAPLE overlay differs from lock")
    expected_source = [[p, s, h] for p, s, h, _blob in admitted["maple"][1]]
    native_source = [
        [p[len("source/"):], row["size"], row["sha256"]]
        for p, row in table.items() if p.startswith("source/")
    ]
    tool_rows = {
        p: digest for p, _s, digest, _b in admitted["syrup"][1]
    }
    tool_sha = tool_rows.get("src/maple_syrup/dependency_bundle.py")
    if (
        sorted(native_source) != sorted(expected_source)
        or record["bundle_id"] != bundle["bundle_id"]
        or record["project"]["exported_from"]
        != str(Path(lock["objects"]["sources"]["maple"]["path"]) / "source")
        or record["tool"]["module_sha256"] != bundle["tool_module_sha256"]
        or tool_sha != bundle["tool_module_sha256"]
        or archive["sha256"] != bundle["archive_sha256"]
        or record["project"]["package_source_digest"]["digest_sha256"]
        != bundle["package_digest"]
        or bundle["package_digest"]
        != lock["objects"]["sources"]["maple"]["package_digest"]
    ):
        _fail("native_binding", "native source/tool/archive/package cross-links differ")
    return record, limits, {filename: raw, mfilename: mraw}


def _committed_verify_sources(lock, admitted, held, cap):
    summaries = {}
    for kind in ("maple", "syrup", "masy"):
        spec = lock["objects"]["sources"][kind]
        held.check()
        summary = runtime.verify_source_object(
            spec["path"], spec["object_id"], runtime.RuntimeLimits(max_lock_bytes=cap)
        )
        if any(summary.get(key) != spec.get(key) for key in (
            "file_count", "total_bytes", "manifest_sha256", "package_digest"
        )):
            _fail("source_identity", f"{kind}: four source summary fields differ")
        manifest, _files, overlay = admitted[kind]
        if summary["manifest_sha256"] != _sha(runtime._canonical_bytes(manifest)):
            _fail("source_identity", f"{kind}: source raw SHA differs")
        expected_overlay = {p: sha for p, _s, sha in overlay} if kind != "maple" else None
        if summary.get("overlay_files") != expected_overlay:
            _fail("source_identity", f"{kind}: verified overlay map differs")
        if kind != "maple":
            dist_name = "maple-syrup" if kind == "syrup" else "masy"
            if expected_overlay != lock["distributions"][dist_name]["files"]:
                _fail("source_identity", f"{kind}: selected overlay map differs")
        summaries[kind] = summary
        held.check()
    return summaries


def _committed_process(lock, flags):
    """The stable record's exact field set is runtime_worker.check_process's."""
    prefix = lock["runtime"]["sys_path_prefix"]
    if sys.path[:len(prefix)] != prefix:
        _fail("committed_runtime", "current sys.path lacks the sealed prefix")
    value = {
        "flags": flags, "sys_path_prefix": list(prefix), "executable": sys.executable,
        "executable_realpath": os.path.realpath(sys.executable), "prefix": sys.prefix,
        "base_prefix": sys.base_prefix, "version": sys.version,
        "implementation": sys.implementation.name, "machine": platform.machine(),
        "sys_platform": sys.platform,
        "environment": {key: os.environ.get(key) for key in rw.STABLE_ENV_KEYS},
    }
    if (
        value != lock["verification"]["process"]
        or value["environment"] != lock["runtime"].get("stable_environment")
    ):
        _fail("committed_runtime", "current stable process record differs from lock")
    return value


def _committed_identity(lock, admitted, roots, flags):
    """Current interpreter/bootstrap/resolver and C environment; no child/write."""
    return _committed_identity_common(lock, admitted, roots, flags)


def _committed_binding(reference, limits):
    """Read-only opt-in committed provenance for the existing v1 digest tuple."""
    return _committed_binding_common(reference, limits)


def _environment_identity():
    """Reuse operational/compiled observers; do not select or change a runtime."""
    return _environment_identity_common()


def _runtime_binding(reference, limits):
    return _runtime_binding_common(reference, limits)


def _limits_record(limits):
    return dataclasses.asdict(limits)


def _limits_load(record):
    _keys(
        record,
        (f.name for f in dataclasses.fields(BoundaryStoreLimits)),
        "store limits",
    )
    values = dict(record)
    _keys(
        values["image"],
        (f.name for f in dataclasses.fields(b.BoundaryImageLimits)),
        "image limits",
    )
    values["image"] = b.BoundaryImageLimits(**values["image"])
    return BoundaryStoreLimits(**values)


def _native_reservation(limits):
    from maple.aeolian.scheduler import physical_checkpoint as cp

    # Native raw cap is codec-defined: two archives, each with metadata slack.
    return limits.image.native_max_bytes + 2 * cp._ZIP_METADATA_SLACK + (1 << 20)


def _reservation(limits, max_work_bytes):
    if type(max_work_bytes) is not int or not 0 <= max_work_bytes <= 6 << 30:
        _fail("memory_cap", "finite plain native work reservation required")
    native = _native_reservation(limits)
    disk = (
        2 * native
        + limits.image.max_serialized_bytes
        + limits.max_generation_bytes
        + 3 * limits.max_intent_bytes
        + limits.max_head_bytes
    )
    # Simultaneous work/readback, native/genesis validations, original capsule,
    # graph factory scratch and native event staging are independent charges.
    memory = (
        2 * limits.image.max_global_raw_bytes
        + 8 * limits.image.native_max_bytes
        + 5 * (64 << 20)
        + 12 * limits.image.max_cache_bytes
        + max_work_bytes
        + (64 << 20)
    )
    if memory > limits.max_memory_bytes:
        _fail("memory_cap", "computed simultaneous owner/native/readback reservation")
    return {
        "disk_bytes": disk,
        "entries": 300,
        "memory_bytes": memory,
        "native_event_work_bytes": max_work_bytes,
        "max_original_decodes": 5,
        "max_native_loads": 8,
        "max_saved_source_graph_builds": 7,
    }


def _reference(directory, fmt, identity, root):
    filename = "manifest.json" if fmt == oi.SCHEMA else "checkpoint.json"
    cap = (4 << 20) if fmt == oi.SCHEMA else (1 << 20)
    digest = runtime._hash_path(directory / filename, budget=cap)[0]
    return {
        "format": fmt,
        "relative_path": str(directory.relative_to(root)),
        "manifest_file_sha256": digest,
        "identity": copy.deepcopy(identity),
    }


def _owner(prepared):
    """Only actual43 loader results, immediately internal; nominal discipline."""
    if type(prepared) is not b._PreparedBoundary:
        _fail("private_carrier", "actual inert passive loader result required")
    owner = prepared._PreparedBoundary__owner
    if type(owner) is not dr.DenseRunner or set(vars(owner)) != set(b.RUNNER_FIELDS):
        _fail("private_carrier", "exact private26-field owner required")
    return owner


def _receipt(manifest):
    if not manifest["receipts"]:
        return None
    row = manifest["receipts"][-1]
    raw = dr._canonical_json(row["body"]).encode("utf-8")
    if len(raw) != row["bytes"] or _sha(raw) != row["sha256"]:
        _fail("receipt_identity", "canonical native wrapper receipt bytes differ")
    return raw


def _head_record(header, store_file_sha, name, generation, generation_file_sha):
    return _seal(
        {
            "schema": _store_schema(header) + ".head",
            "store_id": header["store_id"],
            "store_file_sha256": store_file_sha,
            "sequence": generation["sequence"],
            "generation_name": name,
            "generation_file_sha256": generation_file_sha,
            "generation_sha256": generation["generation_sha256"],
            "operation_id": generation["operation_id"],
        },
        "head_sha256",
    )


def _summary(manifest):
    state, route = manifest["state"], manifest["routing"]
    return {
        "next_index": state["next_index"],
        "time_s": state["native_time"],
        "bed_identity_sha256": state["accepted_bed_sha256"],
        "n_accepted_events": state["n_accepted"],
        "routing_held": route is not None,
        "routing_source_sha256": None if route is None else route["source_sha256"],
    }


def _image_metadata(root, image, limits):
    _keys(
        image,
        ("format", "relative_path", "manifest_file_sha256", "manifest_sha256"),
        "image reference",
    )
    if image["format"] != b.SCHEMA:
        _fail("image_reference", "format")
    _digest(image["manifest_file_sha256"], "image raw hash")
    _digest(image["manifest_sha256"], "image seal")
    path = root / image["relative_path"]
    directory = oi._open_dir(path)
    try:
        manifest, _arrays, raw = b._read_wrapper(
            path, image["manifest_sha256"], limits.image, metadata_only=True
        )
        # _read_wrapper raw includes all wrapper files; its own cap precedes hash.
        if (
            runtime._hash_path(
                path / "manifest.json", budget=limits.image.max_metadata_bytes
            )[0]
            != image["manifest_file_sha256"]
        ):
            _fail("image_reference", "raw manifest hash differs")
        # B metadata admission has checked ALL actual5/7 raw sizes/types/names
        # before these byte hashes. Ancestor arrays need no expansion.
        for row in manifest["arrays"]:
            actual, _blob, size, _data = runtime._hash_path(
                path / row["name"], budget=row["file_bytes"]
            )
            if actual != row["sha256"] or size != row["file_bytes"]:
                _fail("image_reference", "retained wrapper bytes differ")
        _same_directory(path, directory)
    except runtime.RuntimeRefusal as exc:
        raise BoundaryStoreError(exc.code, exc.detail) from exc
    finally:
        os.close(directory)
    return manifest, raw


def _generation_chain(root, header, store_sha, head, limits):
    """Metadata/byte-bound ancestry; selected full native preparation is separate."""
    if _active_water_store(header):
        return _active_module()._generation_chain(root, header, store_sha, head, limits)
    _inventory(
        root,
        limits,
        expected_root=SimpleNamespace(
            st_dev=header["storage_contract"]["root_dev"],
            st_ino=header["storage_contract"]["root_ino"],
        ),
    )
    if _dynamic_store(header):
        _reservation_v2(limits, 0)
    chain, seen, child_receipts = [], set(), None
    native_config = None
    current = head
    while True:
        _check_seal(current, _HEAD, "head_sha256", _store_schema(header) + ".head")
        _control_fields(current)
        sequence = _integer(
            current["sequence"], min(8, limits.max_accepted - 1), "HEAD sequence"
        )
        name = current["generation_name"]
        match = _GEN.fullmatch(name) if type(name) is str else None
        if not match or int(match[1]) != sequence or name in seen:
            _fail("ancestry", "generation name/sequence/cycle")
        seen.add(name)
        if (
            current["store_id"] != header["store_id"]
            or current["store_file_sha256"] != store_sha
        ):
            _fail("ancestry", "foreign HEAD store")
        path = root / "generations" / name
        generation, raw = _record(
            path / "generation.json",
            limits.max_generation_bytes,
            links=_published_links(path, "generation.json"),
        )
        _check_seal(
            generation, _GENERATION, "generation_sha256", _store_schema(header) + ".generation"
        )
        _control_fields(generation, generation=True)
        if (
            _sha(raw) != current["generation_file_sha256"]
            or generation["generation_sha256"] != current["generation_sha256"]
        ):
            _fail("ancestry", "byte-bound generation differs")
        if (
            any(
                generation[k] != current[k]
                for k in ("store_id", "store_file_sha256", "sequence", "operation_id")
            )
            or generation["attempt_id"] != match[2]
        ):
            _fail("ancestry", "generation/HEAD binding")
        intent, intent_raw = _record(path / "intent.json", limits.max_intent_bytes)
        _check_seal(intent, _INTENT, "intent_sha256", _store_schema(header) + ".intent")
        _control_fields(intent, intent=True)
        if _sha(intent_raw) != generation["intent_file_sha256"]:
            _fail("ancestry", "intent raw hash differs")
        for key in ("store_id", "attempt_id", "sequence", "operation_id"):
            if intent[key] != generation[key]:
                _fail("ancestry", "intent/generation identity")
        if (
            oi._json(intent["source_identity"]) != oi._json(header["source_identity"])
            or intent["environment_sha256"] != header["environment_sha256"]
        ):
            _fail("ancestry", "intent runtime identity")
        if (
            intent["expected_parent_head_file_sha256"]
            != generation["parent_head_file_sha256"]
            or intent["expected_parent_generation"] != generation["parent_generation"]
        ):
            _fail("ancestry", "intent parent identity")
        image = generation["image"]
        if (
            type(image) is not dict
            or image.get("relative_path") != f"generations/{name}/image"
        ):
            _fail("image_reference", "image must belong to own generation")
        manifest, wrapper_raw = _image_for(header, root, image, limits)
        _preflight_for(
            header, manifest,
            header["source_identity"],
            header["environment_sha256"],
            limits.image,
        )
        if oi._json(manifest["originals"]) != oi._json(header["originals"]) or oi._json(
            manifest["genesis"]
        ) != oi._json(header["genesis_physical"]):
            _fail("ancestry", "original/genesis references differ")
        if manifest["physical"]["relative_path"] != f"generations/{name}/physical":
            _fail("ancestry", "physical child must belong to own generation")
        if sequence == 0 and oi._json(manifest["physical"]) != oi._json(
            header["genesis_physical"]
        ):
            _fail("ancestry", "genesis physical differs")
        artifacts = _artifacts_for(
            header, root,
            manifest["originals"],
            manifest["genesis"],
            manifest["physical"],
            limits.image,
            wrapper_raw,
        )
        try:
            artifact_hashes = {}
            for ref in artifacts.refs:
                artifact_hashes[ref["relative_path"]] = artifacts.hashed(ref)
            if _dynamic_store(header):
                config = _native_headers_v2(artifacts, manifest, limits, artifact_hashes)
                if native_config is not None and config != native_config:
                    _fail("native_identity", "native configuration branches across ancestry")
                native_config = config
            artifacts.check()
        finally:
            artifacts.close()
        if (
            manifest["state"]["next_index"] != sequence
            or manifest["state"]["n_accepted"] != sequence
            or len(manifest["receipts"]) != sequence
        ):
            _fail("ancestry", "sequence does not equal accepted receipt cursor")
        if oi._json(generation["accepted_summary"]) != oi._json(_summary(manifest)):
            _fail("ancestry", "accepted summary differs from B image")
        receipt = _receipt(manifest)
        if generation["receipt_bytes_sha256"] != (
            None if receipt is None else _sha(receipt)
        ):
            _fail("receipt_identity", "generation receipt identity differs")
        if child_receipts is not None and oi._json(child_receipts[:-1]) != oi._json(
            manifest["receipts"]
        ):
            _fail("ancestry", "receipt prefix branches")
        chain.append((current, generation, manifest))
        child_receipts = manifest["receipts"]
        parent = generation["parent_head_record"]
        if sequence == 0:
            if any(
                generation[k] is not None
                for k in (
                    "parent_head_record",
                    "parent_head_file_sha256",
                    "parent_generation",
                )
            ):
                _fail("ancestry", "genesis has parent")
            break
        if (
            type(parent) is not dict
            or parent.get("sequence") != sequence - 1
            or generation["parent_generation"] != parent.get("generation_name")
            or _sha(oi._json(parent)) != generation["parent_head_file_sha256"]
        ):
            _fail("ancestry", "byte-exact parent record binding")
        if generation["operation_id"] != _operation(
            header["store_id"],
            parent,
            sequence - 1,
            manifest["receipts"][-1]["body"]["resolved"]["schedule_sha256"],
        ):
            _fail("ancestry", "event operation identity differs")
        current = parent
    if _dynamic_store(header):
        _history_v2(root, header, chain, limits)
    return chain


@dataclasses.dataclass(frozen=True)
class _State:
    phase: str
    head: dict | None
    token: str | None
    report: dict | None
    diagnostics: dict | None
    ack: dict | None
    attempt_id: str | None = None
    sequence: int = 0
    intended: str | None = None
    max_work_bytes: int = 0
    schedule_sha256: str | None = None
    identity_halt: object = None


@dataclasses.dataclass(frozen=True)
class BoundaryAcknowledgement:
    schema: str
    store_id: str
    operation_id: str
    sequence: int
    head_token: str
    receipt_bytes: bytes | None


class DenseBoundaryStore:
    """Metadata-only facade. Construct using create/recover, never directly."""

    __slots__ = (
        "_attempt_halt",
        "_boot",
        "_closed",
        "_entered",
        "_evidence_write_error",
        "_failed_owner",
        "_header",
        "_limits",
        "_lock_identity",
        "_lockfd",
        "_pid",
        "_publication_uncertain",
        "_reservation_record",
        "_runtime_validation",
        "_root",
        "_root_identity",
        "_rootfd",
        "_start",
        "_state",
        "_store_raw",
        "_thread",
    )

    def __init__(self):
        _fail("private_constructor", "use create_boundary_store/recover_boundary_store")

    @classmethod
    def _open(cls, root, limits, *, create):
        BoundaryStoreLimits.__post_init__(limits)
        root = Path(root).absolute()
        if str(root) != os.path.realpath(root) or root.name in ("", ".", ".."):
            _fail("path_refused", "canonical no-symlink root required")
        parent = oi._open_dir(root.parent)
        try:
            if create:
                os.mkdir(root.name, 0o700, dir_fd=parent)
                os.fsync(parent)
            rootfd = oi._open_dir(root)
        finally:
            os.close(parent)
        lockfd = None
        try:
            lockfd = os.open(
                ".writer.lock",
                os.O_RDWR | _FLAGS | (os.O_CREAT | os.O_EXCL if create else 0),
                0o600,
                dir_fd=rootfd,
            )
            info = os.fstat(lockfd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_size != 0
            ):
                _fail(
                    "lease_identity",
                    "permanent empty regular single-link lock required",
                )
            try:
                fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BoundaryStoreError(
                    "lease_busy", "another open store owns the lifetime lease"
                ) from exc
            os.fsync(rootfd)
            handle = object.__new__(cls)
            handle._root, handle._rootfd, handle._lockfd = root, rootfd, lockfd
            handle._lock_identity, handle._root_identity = (
                _identity(info),
                _identity(os.fstat(rootfd)),
            )
            handle._pid, handle._thread = os.getpid(), threading.get_ident()
            handle._start = (
                Path(f"/proc/{handle._pid}/stat")
                .read_text()
                .rsplit(")", 1)[1]
                .split()[19]
            )
            handle._boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            handle._header, handle._store_raw = None, None
            handle._limits, handle._state = (
                limits,
                _State("initializing", None, None, None, None, None),
            )
            handle._entered, handle._failed_owner, handle._closed = False, None, False
            handle._reservation_record = None
            handle._runtime_validation = None
            handle._publication_uncertain = None
            handle._attempt_halt = None
            handle._evidence_write_error = None
            handle._binding()
            return handle
        except BaseException:
            if lockfd is not None:
                os.close(lockfd)
            os.close(rootfd)
            raise

    def _binding(self):
        try:
            self._check_binding()
        except Exception as exc:
            drift = isinstance(exc, OSError) or getattr(exc, "code", None) in {
                "identity_drift",
                "lease_identity",
                "limit_identity",
                "limit_cap",
                "limit_layout",
            }
            if (
                drift
                and self._state.phase == "idle"
                and self._state.identity_halt is not None
            ):
                self._state = self._state.identity_halt
            raise

    def _check_binding(self):
        if self._closed:
            _fail("store_closed", "handle is closed")
        if os.getpid() != self._pid or threading.get_ident() != self._thread:
            _fail("handle_binding", "fork/thread reuse is unsupported")
        BoundaryStoreLimits.__post_init__(self._limits)
        if self._header is not None and self._header.get("limits") != _limits_record(
            self._limits
        ):
            _fail("limit_identity", "live limits differ from immutable store contract")
        if (
            Path(f"/proc/{self._pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
            != self._start
            or Path("/proc/sys/kernel/random/boot_id").read_text().strip() != self._boot
        ):
            _fail("handle_binding", "PID/start/boot binding differs")
        _same_directory(self._root, self._rootfd)
        info = os.stat(".writer.lock", dir_fd=self._rootfd, follow_symlinks=False)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size != 0
            or _identity(info) != self._lock_identity
            or _identity(os.fstat(self._lockfd)) != self._lock_identity
        ):
            _fail("lease_identity", "permanent lease inode/path changed")
        if _identity(os.fstat(self._rootfd)) != self._root_identity:
            _fail("identity_drift", "store root changed")

    def _storage(self):
        record = self._header["storage_contract"]
        _keys(record, _CONTRACT, "storage contract")
        info = os.fstat(self._rootfd)
        if (
            record["schema"] != _store_schema(self._header) + ".storage"
            or record["operator_declares_local"] is not True
            or record["operator_declares_fsync_durable"] is not True
            or (record["root_dev"], record["root_ino"]) != _identity(info)
            or record["root_path"] != str(self._root)
        ):
            _fail("storage_contract", "persisted root/declarations differ")
        observed = _mount_observation(self._root, info.st_dev)
        durable = _durable_mount(observed, "current mount observation")
        if (
            observed["fs_type"] not in ("ext4", "xfs")
            or observed["fs_type"] != record["observed_fs_type"]
            or os.fstatvfs(self._rootfd).f_fsid != record["observed_fs_id"]
        ):
            _fail("storage_observation", "filesystem identity/type changed")
        probe, raw = _record(self._root / ".storage_probe/result.json", 32 << 10)
        if (
            _sha(raw) != record["probe_file_sha256"]
            or _durable_mount(probe.get("observation"), "saved mount observation")
            != durable
            or probe.get("f_fsid") != record["observed_fs_id"]
            or probe.get("root_dev") != info.st_dev
            or probe.get("root_ino") != info.st_ino
            or probe.get("root_path") != str(self._root)
        ):
            _fail("storage_observation", "bound probe receipt differs")

    def _stable(self):
        if _active_water_store(self._header):
            return _active_module()._stable(self)
        self._binding()
        inventory = _inventory(
            self._root, self._limits, expected_root=os.fstat(self._rootfd)
        )
        raw = _read(
            self._root / "STORE.json",
            128 << 10,
            links=_published_links(self._root, "STORE.json"),
        )
        if raw != self._store_raw:
            _fail("identity_drift", "immutable STORE bytes changed")
        self._temporary_heads(inventory)
        self._storage()
        source, environment = _runtime_binding(
            self._header["runtime_reference"], self._limits
        )
        if (
            source != self._header["source_identity"]
            or environment != self._header["environment_sha256"]
        ):
            _fail("identity_drift", "retained source/environment differs")

    def _temporary_heads(self, inventory):
        # Aggregate raw admission has completed before parsing any retained temp.
        if _active_water_store(self._header):
            return _active_module()._temporary_heads(self, inventory)
        for nonce in inventory["temps"]:
            head, _raw = _record(
                self._root / f"HEAD.{nonce}.tmp", self._limits.max_head_bytes
            )
            _check_seal(head, _HEAD, "head_sha256", _store_schema(self._header) + ".head")
            _control_fields(head)
            match = (
                _GEN.fullmatch(head["generation_name"])
                if type(head["generation_name"]) is str
                else None
            )
            if (
                not match
                or match[2] != nonce
                or head["generation_name"] not in inventory["generations"]
                or head["store_id"] != self._header["store_id"]
                or head["store_file_sha256"] != _sha(self._store_raw)
            ):
                _fail("head_temporary", "temp does not bind its retained attempt/store")
            path = self._root / "generations" / head["generation_name"]
            generation, raw = _record(
                path / "generation.json",
                self._limits.max_generation_bytes,
                links=_published_links(path, "generation.json"),
            )
            _check_seal(
                generation, _GENERATION, "generation_sha256", _store_schema(self._header) + ".generation"
            )
            if (
                _sha(raw) != head["generation_file_sha256"]
                or generation["generation_sha256"] != head["generation_sha256"]
            ):
                _fail("head_temporary", "temp generation raw identity differs")

    def _head(self):
        if _active_water_store(self._header):
            return _active_module()._head(self)
        record, raw = _record(self._root / "HEAD", self._limits.max_head_bytes)
        _check_seal(record, _HEAD, "head_sha256", _store_schema(self._header) + ".head")
        _control_fields(record)
        return record, raw

    def _admit(self, accepted_names, max_work_bytes, *, initial=False):
        if _active_water_store(self._header):
            return _active_module()._admit(self, accepted_names, max_work_bytes, initial=initial)
        if not initial and len(accepted_names) >= self._limits.max_accepted:
            _fail(
                "accepted_cap",
                "next boundary exceeds retained accepted-generation allowance",
            )
        inventory = _inventory(
            self._root, self._limits, expected_root=os.fstat(self._rootfd)
        )
        orphans = {
            name.rsplit("-", 1)[1]
            for name in inventory["generations"] - set(accepted_names)
        } | inventory["temps"]
        if len(orphans) + (0 if initial else 1) > self._limits.max_orphans:
            _fail("orphan_cap", "retained failed attempts exhaust allowance")
        if (
            self._limits.max_head_bytes,
            self._limits.max_generation_bytes,
            self._limits.max_intent_bytes,
        ) != (32 << 10, 128 << 10, 32 << 10):
            _fail(
                "metadata_cap",
                "v1 writes reserve the complete finite control envelope before work",
            )
        reservation = _reservation_for(self._header, self._limits, max_work_bytes)
        if initial:
            reservation = {
                **reservation,
                "disk_bytes": reservation["disk_bytes"] + (72 << 20),
            }
        if (
            inventory["bytes"] + reservation["disk_bytes"]
            > self._limits.max_store_bytes
            or inventory["entries"] + reservation["entries"] > self._limits.max_entries
        ):
            _fail("store_cap", "complete attempt cannot fit retained store budget")
        space = os.fstatvfs(self._rootfd)
        if (
            space.f_bavail * space.f_frsize
            < self._limits.free_reserve_bytes + reservation["disk_bytes"]
            or space.f_favail
            < self._limits.free_reserve_inodes + reservation["entries"]
        ):
            _fail("storage_cap", "free disk/inodes plus full attempt headroom")
        self._reservation_record = reservation

    def _load(self, head):
        if _active_water_store(self._header):
            return _active_module()._load(self, head)
        chain = _generation_chain(
            self._root, self._header, _sha(self._store_raw), head, self._limits
        )
        _reservation_for(self._header, self._limits, 0)
        image = chain[0][1]["image"]
        prepared = _prepare_for(self._header,
            self._root / image["relative_path"],
            artifact_root=self._root,
            expected_manifest_sha256=image["manifest_sha256"],
            expected_source_identity=self._header["source_identity"],
            expected_environment_sha256=self._header["environment_sha256"],
            limits=self._limits.image,
        )
        owner = _owner_for(self._header, prepared)
        diagnostics = prepared.diagnostics()
        if oi._json(diagnostics["accepted"]) != oi._json(
            chain[0][1]["accepted_summary"]
        ):
            _fail("readback_identity", "full selected native accepted state differs")
        for parent, generation, _manifest in chain:
            origin_parent = generation["parent_head_record"]
            index = 0 if parent["sequence"] == 0 else parent["sequence"] - 1
            if generation["operation_id"] != _operation(
                self._header["store_id"], origin_parent, index, owner._schedule.digest
            ):
                _fail("operation_identity", "original schedule does not bind operation")
        return prepared, chain

    def _idle_state(self, head, prepared, generation):
        if _active_water_store(self._header):
            return _active_module()._idle_state(self, head, prepared, generation)
        owner = _owner_for(self._header, prepared)
        report, diagnostics = owner.report(), prepared.diagnostics()
        receipt = (
            None
            if not owner._receipts
            else dr._canonical_json(owner._receipts[-1]).encode("utf-8")
        )
        if (None if receipt is None else _sha(receipt)) != generation[
            "receipt_bytes_sha256"
        ]:
            _fail("receipt_identity", "loaded receipt differs from generation")
        token = _token(head)
        ack = BoundaryAcknowledgement(
            _store_schema(self._header) + ".ack",
            self._header["store_id"],
            generation["operation_id"],
            head["sequence"],
            token,
            receipt,
        )
        base = _State(
            "idle",
            copy.deepcopy(head),
            token,
            report,
            diagnostics,
            ack,
            sequence=head["sequence"],
            intended=token,
            max_work_bytes=owner._limits.max_work_bytes,
            schedule_sha256=owner._schedule.digest,
        )
        halt = dataclasses.replace(base, phase="halted_before_head", intended=None)
        return dataclasses.replace(base, identity_halt=halt)

    def _reserve(self, sequence, parent, schedule):
        attempt = uuid.uuid4().hex
        operation = _operation(
            self._header["store_id"], parent, max(0, sequence - 1), schedule
        )
        name = f"g-{sequence:04d}-{attempt}"
        old = self._state
        busy = _State(
            "busy_before_head",
            old.head,
            old.token,
            old.report,
            old.diagnostics,
            old.ack,
            attempt,
            sequence,
            None,
        )
        halt = dataclasses.replace(busy, phase="halted_before_head")
        self._attempt_halt = halt
        self._state = busy
        try:
            path = self._root / "generations" / name
            directory = oi._open_dir(self._root / "generations")
            try:
                os.mkdir(name, 0o700, dir_fd=directory)
                os.fsync(directory)
            finally:
                os.close(directory)
            intent = _seal(
                {
                    "schema": _store_schema(self._header) + ".intent",
                    "store_id": self._header["store_id"],
                    "attempt_id": attempt,
                    "sequence": sequence,
                    "expected_parent_head_file_sha256": None
                    if parent is None
                    else _sha(oi._json(parent)),
                    "expected_parent_generation": None
                    if parent is None
                    else parent["generation_name"],
                    "operation_id": operation,
                    "source_identity": self._header["source_identity"],
                    "environment_sha256": self._header["environment_sha256"],
                },
                "intent_sha256",
            )
            _write(path / "intent.json", oi._json(intent))
            fd = oi._open_dir(path)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            return path, intent, halt
        except BaseException:
            self._state = halt
            raise

    def _extra_original(self, manifest, wrapper_raw):
        if _dynamic_store(self._header):
            return _extra_original_v2(self, manifest, wrapper_raw)
        artifacts = b._Artifacts(
            self._root,
            manifest["originals"],
            manifest["genesis"],
            manifest["physical"],
            self._limits.image,
            wrapper_raw,
        )
        try:
            original = manifest["originals"]
            artifacts.hashed(original)
            seal = b._original_manifest_seal(artifacts, original, self._limits.image)
            pack = oi.load_original_inputs(
                self._root / original["relative_path"],
                expected_manifest_sha256=seal,
                expected_source_identity=self._header["source_identity"],
                limits=oi.OriginalPackLimits(
                    max_cells=self._limits.image.max_cells,
                    max_classes=self._limits.image.max_classes,
                    max_voxels=self._limits.image.max_voxels,
                    max_cohorts=self._limits.image.max_cohorts,
                    max_events=self._limits.image.max_events,
                ),
            )
            if pack.content_sha256 != original["identity"]:
                _fail(
                    "original_identity",
                    "independently decoded original content differs",
                )
            artifacts.check()
            artifacts.hashed(original)
            return pack
        finally:
            artifacts.close()

    def _save_physical(self, path, owner, pack):
        if _dynamic_store(self._header):
            return _save_physical_v2(self, path, owner, pack)
        if (
            pack.content_sha256 != self._header["originals"]["identity"]
            or pack.static_sha256 != owner.static_sha256
        ):
            _fail("original_identity", "work/original controlling identity differs")
        caller = {
            "schema": "masy.dense_runner_boundary_native.v1",
            "source_identity": self._header["source_identity"],
            "environment_sha256": self._header["environment_sha256"],
            "static_sha256": owner.static_sha256,
        }
        packed = bridge.pack_boundary_physical(
            owner._bed,
            owner._pool,
            owner._wind.config,
            caller,
            inputs=pack.capsule,
            max_bytes=self._limits.image.native_max_bytes,
        )
        from maple.aeolian.scheduler import physical_checkpoint as cp

        cp.save_exact_physical_checkpoint(
            path,
            packed.physical,
            owner._wind.config,
            packed.identity,
            max_bytes=self._limits.image.native_max_bytes,
        )
        return _reference(path, cp.SCHEMA, packed.identity, self._root)

    def _save_image(self, directory, owner, physical):
        if _dynamic_store(self._header):
            return _save_image_v2(self, directory, owner, physical)
        seal = b.save_passive_boundary(
            directory,
            owner,
            artifact_root=self._root,
            originals=self._header["originals"],
            genesis=self._header["genesis_physical"],
            physical=physical,
            source_identity=self._header["source_identity"],
            environment_sha256=self._header["environment_sha256"],
            limits=self._limits.image,
        )
        return {
            "format": b.SCHEMA,
            "relative_path": str(directory.relative_to(self._root)),
            "manifest_file_sha256": runtime._hash_path(
                directory / "manifest.json",
                budget=self._limits.image.max_metadata_bytes,
            )[0],
            "manifest_sha256": seal,
        }

    def _seal_generation(self, directory, intent, parent, image):
        manifest, _raw = _image_for(self._header, self._root, image, self._limits)
        receipt = _receipt(manifest)
        generation = _seal(
            {
                "schema": _store_schema(self._header) + ".generation",
                "store_id": self._header["store_id"],
                "store_file_sha256": _sha(self._store_raw),
                "attempt_id": intent["attempt_id"],
                "sequence": intent["sequence"],
                "intent_file_sha256": _sha(oi._json(intent)),
                "parent_head_record": copy.deepcopy(parent),
                "parent_head_file_sha256": None
                if parent is None
                else _sha(oi._json(parent)),
                "parent_generation": None
                if parent is None
                else parent["generation_name"],
                "operation_id": intent["operation_id"],
                "image": image,
                "accepted_summary": _summary(manifest),
                "receipt_bytes_sha256": None if receipt is None else _sha(receipt),
            },
            "generation_sha256",
        )
        file_sha = _sealed_write(
            directory, "generation.json", generation, self._limits.max_generation_bytes
        )
        _inventory(self._root, self._limits, expected_root=os.fstat(self._rootfd))
        _sync_tree(self._root)
        return _head_record(
            self._header, _sha(self._store_raw), directory.name, generation, file_sha
        )

    def _publish(self, head, prepared_state, old_head, halt):
        """No model/validation/allocation after required root fsync."""
        raw = oi._json(head)
        name = f"HEAD.{head['generation_name'].rsplit('-', 1)[1]}.tmp"
        possible = dataclasses.replace(
            halt, phase="publication_possible", intended=prepared_state.token
        )
        uncertain = dataclasses.replace(possible, phase="publication_indeterminate")
        self._publication_uncertain = uncertain
        _write(self._root / name, raw)
        if _read(self._root / name, self._limits.max_head_bytes) != raw:
            _fail("metadata_readback", "HEAD temp differs")
        self._stable()
        _generation_chain(
            self._root, self._header, _sha(self._store_raw), head, self._limits
        )
        if old_head is None:
            if os.path.lexists(self._root / "HEAD"):
                _fail("parent_drift", "genesis HEAD unexpectedly exists")
        elif self._head()[1] != oi._json(old_head):
            _fail("parent_drift", "exact parent HEAD changed before replace")
        # Last fallible preparation completed. Conservative phase precedes syscall.
        try:
            self._state = possible
            os.replace(name, "HEAD", src_dir_fd=self._rootfd, dst_dir_fd=self._rootfd)
            os.fsync(self._rootfd)
            self._state = prepared_state
            return prepared_state.ack
        except BaseException:
            self._state = uncertain
            raise

    def head_token(self):
        self._binding()
        if self._state.phase != "idle":
            _fail("store_halted", self._state.phase)
        return self._state.token

    def report(self):
        if _active_water_store(self._header):
            return _active_module().report(self)
        self._binding()
        return {
            "schema": _store_schema(self._header) + ".report",
            "phase": self._state.phase,
            "head_token": self._state.token,
            "intended_token": self._state.intended,
            "store_file_sha256": None
            if self._store_raw is None
            else _sha(self._store_raw),
            "accepted": copy.deepcopy(self._state.report),
            "diagnostics": copy.deepcopy(self._state.diagnostics),
            "reservation": copy.deepcopy(self._reservation_record),
            "failure_evidence_error": self._evidence_write_error,
            "scope": "bounded source-manifest prototype; active-event/production/GPU/windowed excluded",
        }

    def step(self, expected_parent):
        if _active_water_store(self._header):
            return _active_module().step(self, expected_parent)
        self._binding()
        if self._entered:
            _fail("handle_reentrant", "one store operation at a time")
        if self._state.phase != "idle":
            _fail("store_halted", self._state.phase)
        _digest(expected_parent, "expected parent token")
        old = self._state
        early_halt = dataclasses.replace(old, phase="halted_before_head", intended=None)
        reserved, halt = False, early_halt
        try:
            self._entered = True
            self._publication_uncertain = None
            self._attempt_halt = None
            try:
                self._stable()
                head, _raw = self._head()
                if head != old.head:
                    _fail("parent_drift", "durable HEAD differs from facade")
            except BaseException:
                self._state = early_halt
                raise
            chain = _generation_chain(
                self._root, self._header, _sha(self._store_raw), head, self._limits
            )
            if expected_parent != old.token:
                return self._duplicate(expected_parent, chain)
            if (
                head["sequence"] >= self._limits.image.max_events
                or old.report["status"] == "complete"
            ):
                _fail("schedule_complete", "all events already accepted")
            # Parent B metadata is not a fresh bed constructor and does not expand arrays.
            manifest, wrapper_raw = _image_for(
                self._header, self._root, chain[0][1]["image"], self._limits
            )
            # Original native work guard is persisted in original pack, checked below.
            self._admit(
                [row[0]["generation_name"] for row in chain], old.max_work_bytes
            )
            schedule = old.schedule_sha256
            directory, intent, halt = self._reserve(
                head["sequence"] + 1, head, schedule
            )
            reserved = True
            prepared, _parent_chain = self._load(head)
            work = _owner_for(self._header, prepared)
            self._failed_owner = work
            result = work.step()
            if (
                type(result) is not dict
                or result.get("status") != "accepted"
                or work._next != head["sequence"] + 1
            ):
                _fail(
                    "event_incomplete",
                    "private event did not accept; native evidence retained until close",
                )
            _admit_owner_for(self._header, work, self._limits.image)
            pack = self._extra_original(manifest, wrapper_raw)
            physical = self._save_physical(directory / "physical", work, pack)
            del pack
            image = self._save_image(directory / "image", work, physical)
            next_head = self._seal_generation(directory, intent, head, image)
            readback, selected = self._load(next_head)
            read_owner = _owner_for(self._header, readback)
            _compare_for(self._header, work, read_owner, self._limits.image)
            state = self._idle_state(next_head, readback, selected[0][1])
            del work, read_owner, prepared, readback
            self._failed_owner = None
            return self._publish(next_head, state, head, halt)
        except BaseException as exc:
            if (
                reserved
                and self._publication_uncertain is not None
                and self._state.token != old.token
            ):
                self._state = self._publication_uncertain
            elif (
                self._state.phase == "busy_before_head"
                and self._attempt_halt is not None
            ):
                self._state = self._attempt_halt
            elif reserved and self._state.phase not in (
                "publication_possible",
                "publication_indeterminate",
            ):
                self._state = halt
            if (
                reserved
                and isinstance(exc, Exception)
                and self._state.phase == "halted_before_head"
            ):
                self._failure(directory, exc)
            raise
        finally:
            self._entered = False

    def checkpoint_water(self, expected_parent, *, boundary_ordinal):
        if not _active_water_store(self._header):
            _fail("store_profile", "explicit active-water format required")
        return _active_module().checkpoint_water(self, expected_parent, boundary_ordinal=boundary_ordinal)

    def acknowledge(self, expected_parent, *, request_kind, boundary_ordinal=None):
        if not _active_water_store(self._header):
            _fail("store_profile", "explicit active-water format required")
        return _active_module().acknowledge(self, expected_parent, request_kind=request_kind,
                                            boundary_ordinal=boundary_ordinal)

    def _failure(self, directory, error):
        try:
            record = {
                "schema": _store_schema(self._header) + ".failure",
                "phase": self._state.phase,
                "attempt_id": self._state.attempt_id,
                "intended_token": self._state.intended,
                "exception_type": type(error).__name__,
                "code": getattr(error, "code", "error"),
                "detail": str(error)[:2048],
                "scope": "unaccepted private attempt; no active resume",
            }
            raw = oi._json(record)
            if len(raw) <= self._limits.max_intent_bytes:
                _write(directory / "failure.json", raw)
        except Exception:  # noqa: BLE001 - optional evidence must not replace primary failure
            # Primary error and its live unknown evidence remain authoritative.
            self._evidence_write_error = (
                "failure metadata write unconfirmed; primary error retained"
            )

    def _duplicate(self, expected_parent, chain):
        for index, (head, _gen, _manifest) in enumerate(chain):
            if _token(head) == expected_parent:
                if index == 0:
                    _fail(
                        "stale_parent", "current parent has no accepted child operation"
                    )
                accepted_head, generation, manifest = chain[index - 1]
                receipt = _receipt(manifest)
                if (
                    receipt is None
                    or oi._json(generation["parent_head_record"]) != oi._json(head)
                    or generation["operation_id"]
                    != _operation(
                        self._header["store_id"],
                        head,
                        head["sequence"],
                        manifest["receipts"][-1]["body"]["resolved"]["schedule_sha256"],
                    )
                ):
                    _fail("stale_parent", "no matching direct-child operation")
                return BoundaryAcknowledgement(
                    _store_schema(self._header) + ".ack",
                    self._header["store_id"],
                    generation["operation_id"],
                    accepted_head["sequence"],
                    _token(accepted_head),
                    receipt,
                )
        _fail("stale_parent", "token is not in selected ancestry")

    def close(self):
        if self._closed:
            return
        if os.getpid() != self._pid or threading.get_ident() != self._thread:
            _fail("handle_binding", "close in creating PID/thread only")
        if self._entered:
            _fail("handle_reentrant", "close during an operation is unsupported")
        self._failed_owner = None
        lock, root = self._lockfd, self._rootfd
        self._lockfd = self._rootfd = -1
        self._closed = True
        try:
            os.close(lock)
        finally:
            os.close(root)


def _compare_owners(work, readback, limits):
    """Exact passive readback comparison without another native graph rebuild."""
    for name in (
        "_inventory0",
        "_pool",
        "_soil",
        "_totals",
        "_static_parts",
        "_static_record",
    ):
        b._same(getattr(work, name), getattr(readback, name), limits)
    for a, z in (
        (work._bed.state, readback._bed.state),
        (work._bed.context, readback._bed.context),
        (work._bed.vegetation, readback._bed.vegetation),
        (work._bed.threshold_modifier_state, readback._bed.threshold_modifier_state),
    ):
        b._same(a, z, limits)
    for name in (
        "_accepted_bed",
        "_accepted_private",
        "_n_accepted",
        "_next",
        "_n_classes",
        "_inventory_elements",
        "_static_parts_sha256",
        "_static_sha256",
    ):
        if getattr(work, name) != getattr(readback, name):
            _fail("readback_identity", name)
    if (
        oi._json(work.ledger_record()) != oi._json(readback.ledger_record())
        or dr._canonical_json(work._receipts) != dr._canonical_json(readback._receipts)
        or oi._json(work.accepted.as_record())
        != oi._json(readback.accepted.as_record())
    ):
        _fail("readback_identity", "clocks/receipts/accepted state")
    if (work._routing is None) != (readback._routing is None):
        _fail("readback_identity", "routing presence")
    if work._routing is not None:
        geometry = work._bed.context.geometry
        shape = (geometry.ny, geometry.nx)
        a = b._routing_record(
            work._routing, work._water, shape, work._n_classes, limits
        )
        z = b._routing_record(
            readback._routing, readback._water, shape, readback._n_classes, limits
        )
        if oi._json(a) != oi._json(z):
            _fail("readback_identity", "retained native source/cache identity")


def create_boundary_store(
    path, *, runner, originals, runtime_reference, limits=_DEFAULT, storage_contract
):
    """Create a NEW genesis store; never import an arbitrary progressed runner."""
    BoundaryStoreLimits.__post_init__(limits)
    StorageContract.__post_init__(storage_contract)
    b._runner_admission(runner, limits.image)
    if (
        runner._next != 0
        or runner._n_accepted != 0
        or runner._receipts
        or runner._stop_receipt is not None
        or runner._active_evidence is not None
    ):
        _fail(
            "fresh_runner_required",
            "only untouched default genesis may initialize a store",
        )
    if type(originals) is not oi.OriginalInputs:
        _fail("original_identity", "actual newly captured OriginalInputs required")
    source, environment = _runtime_binding(runtime_reference, limits)
    if originals.source_identity != source:
        _fail("original_identity", "captured source differs from complete runtime")
    # The original codec preserves input mapping order; never silently reorder
    # its captured source dictionary merely because control JSON sorts keys.
    captured = oi.capture_original_inputs(
        runner, source_identity=originals.source_identity
    )
    if (
        captured.content_sha256 != originals.content_sha256
        or captured.static_sha256 != originals.static_sha256
    ):
        _fail(
            "original_identity",
            "independently recaptured original does not match runner",
        )
    del captured
    handle = DenseBoundaryStore._open(path, limits, create=True)
    try:
        probe, probe_sha = _probe(handle._root, handle._rootfd)
        os.mkdir(handle._root / "originals", 0o700)
        os.mkdir(handle._root / "generations", 0o700)
        handle._header = {
            "schema": SCHEMA,
            "store_id": uuid.uuid4().hex,
            "profile": PROFILE,
            "source_identity": source,
            "runtime_reference": copy.deepcopy(runtime_reference),
            "environment_sha256": environment,
            "storage_contract": {
                "schema": SCHEMA + ".storage",
                "operator_declares_local": True,
                "operator_declares_fsync_durable": True,
                "observed_fs_type": probe["observation"]["fs_type"],
                "observed_fs_id": probe["f_fsid"],
                "root_path": str(handle._root),
                "root_dev": probe["root_dev"],
                "root_ino": probe["root_ino"],
                "probe_file_sha256": probe_sha,
            },
            "limits": _limits_record(limits),
            "originals": None,
            "genesis_physical": None,
        }
        handle._admit([], runner._limits.max_work_bytes, initial=True)
        original_path = handle._root / "originals" / ("o-" + uuid.uuid4().hex)
        oi.save_original_inputs(
            original_path,
            originals,
            limits=oi.OriginalPackLimits(
                max_cells=limits.image.max_cells,
                max_classes=limits.image.max_classes,
                max_voxels=limits.image.max_voxels,
                max_cohorts=limits.image.max_cohorts,
                max_events=limits.image.max_events,
            ),
        )
        handle._header["originals"] = _reference(
            original_path, oi.SCHEMA, originals.content_sha256, handle._root
        )
        directory, intent, halt = handle._reserve(0, None, runner._schedule.digest)
        physical = handle._save_physical(directory / "physical", runner, originals)
        handle._header["genesis_physical"] = physical
        handle._header = _seal(handle._header, "store_sha256")
        _sealed_write(handle._root, "STORE.json", handle._header, 128 << 10)
        handle._store_raw = oi._json(handle._header)
        image = handle._save_image(directory / "image", runner, physical)
        head = handle._seal_generation(directory, intent, None, image)
        prepared, chain = handle._load(head)
        _compare_owners(runner, _owner(prepared), limits.image)
        state = handle._idle_state(head, prepared, chain[0][1])
        del prepared
        handle._publish(head, state, None, halt)
        return handle
    except BaseException:
        handle.close()
        raise


def recover_boundary_store(path, *, expected_store_file_sha256, limits=_DEFAULT):
    """Recover ONLY selected HEAD with retained originals/native sources; no fallback."""
    BoundaryStoreLimits.__post_init__(limits)
    _digest(expected_store_file_sha256, "expected STORE file hash")
    handle = DenseBoundaryStore._open(path, limits, create=False)
    try:
        _inventory(handle._root, limits, expected_root=os.fstat(handle._rootfd))
        header, raw = _record(
            handle._root / "STORE.json",
            128 << 10,
            links=_published_links(handle._root, "STORE.json"),
        )
        _check_seal(header, _STORE, "store_sha256", SCHEMA)
        if (
            _sha(raw) != expected_store_file_sha256
            or header["profile"] != PROFILE
            or type(header["store_id"]) is not str
            or not _NONCE.fullmatch(header["store_id"])
        ):
            _fail("store_identity", "store hash/id/profile mismatch")
        stored_limits = _limits_load(header["limits"])
        if _limits_record(stored_limits) != _limits_record(limits):
            _fail("limit_identity", "restore must use exact persisted limits")
        b._reference(header["originals"], oi.SCHEMA)
        b._reference(header["genesis_physical"], "maple-exact-physical-checkpoint/1")
        if not re.fullmatch(
            r"originals/o-[0-9a-f]{32}", header["originals"]["relative_path"]
        ) or not re.fullmatch(
            r"generations/g-0000-[0-9a-f]{32}/physical",
            header["genesis_physical"]["relative_path"],
        ):
            _fail("store_identity", "original/genesis roles")
        handle._header, handle._store_raw = header, raw
        handle._stable()
        head, head_raw = handle._head()
        prepared, chain = handle._load(head)
        state = handle._idle_state(head, prepared, chain[0][1])
        del prepared
        _sync_tree(handle._root)
        handle._stable()
        if handle._head()[1] != head_raw:
            _fail("parent_drift", "HEAD changed during recovery")
        # Final fsync, then only the preprepared metadata state assignment.
        os.fsync(handle._rootfd)
        handle._state = state
        return handle
    except BaseException:
        handle.close()
        raise


# Closed manual-vegetation boundary codec; no additional publication engine.
SCHEMA_V2 = "masy.dense_boundary_store.v2"
PROFILE_V2 = "dynamic_manual_dense_numpy_accepted_v1"


def _store_schema(header):
    """Closed code-owned v1/v2 selector; partial headers allowed before store_sha256."""
    if type(header) is not dict:
        _fail("store_profile", "exact header mapping required")
    schema = header.get("schema")
    profile = header.get("profile")
    if type(schema) is not str or type(profile) is not str:
        _fail("store_profile", "exact plain str schema/profile required")
    if schema == SCHEMA and profile == PROFILE:
        return SCHEMA
    if schema == SCHEMA_V2 and profile == PROFILE_V2:
        return SCHEMA_V2
    if schema == SCHEMA_ACTIVE_WATER and profile == PROFILE_ACTIVE_WATER:
        return SCHEMA_ACTIVE_WATER
    _fail("store_profile", "unknown or mixed schema/profile pair")


def _dynamic_store(header):
    return _store_schema(header) == SCHEMA_V2


def _owner_v2(prepared):
    """Only actual v2 loader results, immediately internal; nominal discipline."""
    if type(prepared) is not b._PreparedBoundaryV2:
        _fail("private_carrier", "actual inert passive v2 loader result required")
    owner = prepared._PreparedBoundaryV2__owner
    fields = b.RUNNER_FIELDS_V2
    if (
        type(owner) is not dr.DenseRunner
        or len(set(fields)) != 28
        or set(vars(owner)) != set(fields)
    ):
        _fail("private_carrier", "exact private28-field owner required")
    try:
        plan = owner._vegetation_plan
        aligned = plan is not None and plan is owner._bed._dynamic.plan
    except AttributeError:
        aligned = False
    if not aligned:
        _fail("private_carrier", "vegetation plan identity mismatch")
    return owner


def _reference_v2(directory, fmt, identity, root):
    if type(fmt) is str and fmt == oi.SCHEMA_V2:
        filename, cap = "manifest.json", 4 << 20
    elif type(fmt) is str and fmt == "maple-exact-physical-checkpoint/1":
        filename, cap = "checkpoint.json", 1 << 20
    else:
        _fail("reference_format", "closed v2 reference format required")
    digest = runtime._hash_path(directory / filename, budget=cap)[0]
    result = {
        "format": fmt,
        "relative_path": str(directory.relative_to(root)),
        "manifest_file_sha256": digest,
        "identity": copy.deepcopy(identity),
    }
    b._reference_v2(result, fmt)
    return result


def _owner_for(header, prepared):
    return _owner_v2(prepared) if _dynamic_store(header) else _owner(prepared)


def _preflight_for(header, *args):
    function = b._preflight_metadata_v2 if _dynamic_store(header) else b._preflight_metadata
    return function(*args)


def _artifacts_for(header, *args):
    kind = b._ArtifactsV2 if _dynamic_store(header) else b._Artifacts
    return kind(*args)


def _prepare_for(header, *args, **kwargs):
    function = b.prepare_passive_boundary_v2 if _dynamic_store(header) else b.prepare_passive_boundary
    return function(*args, **kwargs)


def _admit_owner_for(header, owner, limits):
    function = b._runner_admission_v2 if _dynamic_store(header) else b._runner_admission
    return function(owner, limits)


def _compare_for(header, owner, readback, limits):
    function = _compare_owners_v2 if _dynamic_store(header) else _compare_owners
    return function(owner, readback, limits)


def _image_for(header, root, image, limits):
    dynamic = _dynamic_store(header)
    if type(image) is not dict or image.get("format") != (b.SCHEMA_V2 if dynamic else b.SCHEMA):
        _fail("image_reference", "image format must match the admitted store codec")
    return _image_metadata_v2(root, image, limits) if dynamic else _image_metadata(root, image, limits)


def _reservation_for(header, limits, max_work_bytes):
    return _reservation_v2(limits, max_work_bytes) if _dynamic_store(header) else _reservation(limits, max_work_bytes)


def _extra_original_v2(handle, manifest, wrapper_raw):
    artifacts = b._ArtifactsV2(handle._root, manifest["originals"], manifest["genesis"],
        manifest["physical"], handle._limits.image, wrapper_raw)
    try:
        original = manifest["originals"]
        artifacts.hashed(original)
        seal = b._original_manifest_seal(artifacts, original, handle._limits.image)
        decoded = oi.load_original_inputs_v2(handle._root / original["relative_path"],
            expected_manifest_sha256=seal,
            expected_source_identity=handle._header["source_identity"],
            limits=oi.OriginalPackLimits(max_cells=handle._limits.image.max_cells,
                max_classes=handle._limits.image.max_classes, max_voxels=handle._limits.image.max_voxels,
                max_cohorts=handle._limits.image.max_cohorts, max_events=handle._limits.image.max_events))
        if (type(decoded) is not oi._DecodedOriginalInputsV2
                or decoded._authority is not oi._DECODED_V2_TOKEN
                or type(decoded.originals) is not oi.OriginalInputsV2
                or decoded.originals.content_sha256 != original["identity"]):
            _fail("original_identity", "exact v2 original decoder authority/content required")
        # Retain the decoded plan together with its original capsule until save.
        artifacts.check()
        artifacts.hashed(original)
        return decoded
    finally:
        artifacts.close()


def _save_physical_v2(handle, path, owner, decoded):
    if (type(decoded) is oi.OriginalInputsV2
            and handle._header["genesis_physical"] is None and owner._next == 0):
        # A virgin owner supplies the genuine, independently recaptured pack.
        pack, plan = decoded, owner._vegetation_plan
    elif (type(decoded) is oi._DecodedOriginalInputsV2
            and decoded._authority is oi._DECODED_V2_TOKEN):
        pack, plan = decoded.originals, decoded.plan
    else:
        _fail("original_identity", "genuine genesis pack or exact v2 decoder authority required")
    if (pack.content_sha256 != handle._header["originals"]["identity"]
            or pack.static_sha256 != owner.static_sha256
            or plan.plan_sha256 != owner._vegetation_plan.plan_sha256):
        _fail("original_identity", "work/original/plan controlling identity differs")
    resources = b._resources_from_runner_v2(owner, handle._limits.image)
    caller = {"schema": b._NATIVE_CALLER_SCHEMA_V2,
        "source_identity": handle._header["source_identity"],
        "environment_sha256": handle._header["environment_sha256"],
        "static_sha256": owner.static_sha256}
    packed = bridge.pack_boundary_physical_v2(owner._bed, owner._pool, owner._wind.config,
        caller, inputs=pack.capsule, shared_resources=resources,
        max_bytes=handle._limits.image.native_max_bytes)
    from maple.aeolian.scheduler import physical_checkpoint as cp

    cp.save_exact_physical_checkpoint(path, packed.physical, owner._wind.config, packed.identity,
        max_bytes=handle._limits.image.native_max_bytes)
    return _reference_v2(path, cp.SCHEMA, packed.identity, handle._root)


def _save_image_v2(handle, directory, owner, physical):
    seal = b.save_passive_boundary_v2(directory, owner, artifact_root=handle._root,
        originals=handle._header["originals"], genesis=handle._header["genesis_physical"],
        physical=physical, source_identity=handle._header["source_identity"],
        environment_sha256=handle._header["environment_sha256"], limits=handle._limits.image)
    return {"format": b.SCHEMA_V2, "relative_path": str(directory.relative_to(handle._root)),
        "manifest_file_sha256": runtime._hash_path(directory / "manifest.json",
            budget=handle._limits.image.max_metadata_bytes)[0], "manifest_sha256": seal}


def _image_metadata_v2(root, image, limits):
    _keys(image, ("format", "relative_path", "manifest_file_sha256", "manifest_sha256"), "image reference")
    if image["format"] != b.SCHEMA_V2:
        _fail("image_reference", "v2 format")
    _digest(image["manifest_file_sha256"], "image raw hash")
    _digest(image["manifest_sha256"], "image seal")
    path = root / image["relative_path"]
    directory = oi._open_dir(path)
    try:
        manifest, _arrays, raw = b._read_wrapper_v2(path, image["manifest_sha256"],
            limits.image, metadata_only=True)
        # B already required its exact read bytes == oi._json(manifest) and
        # checked every held file identity. Bind that admitted canonical image;
        # `raw` below is B's aggregate byte COUNT, not its manifest bytes.
        if _sha(oi._json(manifest)) != image["manifest_file_sha256"]:
            _fail("image_reference", "raw manifest hash differs")
        for row in manifest["arrays"]:
            actual, _blob, size, _data = runtime._hash_path(path / row["name"], budget=row["file_bytes"])
            if actual != row["sha256"] or size != row["file_bytes"]:
                _fail("image_reference", "retained wrapper bytes differ")
        _same_directory(path, directory)
    except runtime.RuntimeRefusal as exc:
        raise BoundaryStoreError(exc.code, exc.detail) from exc
    finally:
        os.close(directory)
    return manifest, raw


def _native_headers_v2(artifacts, manifest, limits, hashes):
    """Native header consistency only; selected B/native loads own physical validation."""
    from maple.aeolian.scheduler import physical_checkpoint as cp

    expected_config = None
    seen = {}
    for name in ("genesis", "physical"):
        reference = manifest[name]
        relative = reference["relative_path"]
        if relative in seen:
            if oi._json(seen[relative]) != oi._json(reference):
                _fail("native_identity", "one native path requires one complete identity")
            continue
        seen[relative] = reference
        path, _fd, _rows, _fmt, sizes = artifacts.files[relative]
        if (type(sizes["checkpoint.json"]) is not int
                or not 1 <= sizes["checkpoint.json"] <= min(cp.MAX_METADATA_BYTES, 1 << 20)):
            _fail("native_identity", "bounded native header required before parse")
        artifacts.check()
        raw = runtime._read_bounded(path / "checkpoint.json", sizes["checkpoint.json"], "native_manifest_read")
        if len(raw) != sizes["checkpoint.json"] or _sha(raw) != reference["manifest_file_sha256"]:
            _fail("native_identity", "native admitted header bytes differ")
        native = cp._strict_json(raw)
        cp._keys(native, cp._MANIFEST_KEYS, "manifest")
        identity = cp._identity(native["identity"])
        if (native["schema"] != cp.SCHEMA
                or cp._json(identity) != cp._json(cp._identity(reference["identity"]))):
            _fail("native_identity", "native schema/identity differs")
        # The immutable raw header is ref-bound. The actual selected/genesis
        # native loaders validate their internal seals; no copied seal framing.
        _digest(native["manifest_sha256"], "native manifest seal")
        config = _digest(native["config_sha256"], "native configuration")
        _digest(native["physical_sha256"], "native physical metadata")
        if expected_config is not None and config != expected_config:
            _fail("native_identity", "genesis/current native configurations differ")
        expected_config = config
        cp._keys(native["files"], cp.PAYLOADS, "payload list")
        for payload in cp.PAYLOADS:
            row = native["files"][payload]
            cp._keys(row, ("sha256", "size_bytes", "members"), "payload record")
            if (type(row["size_bytes"]) is not int or row["size_bytes"] != sizes[payload]
                    or _digest(row["sha256"], "native payload") != hashes[relative][payload]
                    or type(row["members"]) is not dict or not 1 <= len(row["members"]) <= 256):
                _fail("native_identity", "admitted native payload metadata differs")
            for member, description in row["members"].items():
                if type(member) is not str or len(member) > 200:
                    _fail("native_identity", "bounded native member name")
                cp._keys(description, ("dtype", "shape", "npy_bytes"), "native member")
                if (type(description["dtype"]) is not str or len(description["dtype"]) > 128
                        or type(description["npy_bytes"]) is not int
                        or not 0 <= description["npy_bytes"] <= limits.image.native_max_bytes
                        or type(description["shape"]) is not list or len(description["shape"]) > 5
                        or any(type(v) is not int or not 0 <= v <= max(
                            limits.image.max_cohorts, limits.image.max_cells + 2) for v in description["shape"])):
                    _fail("native_identity", "bounded native raw description before expansion")
        artifacts.check()
        # Header parse/canonical/identity outputs never enter ancestry records.
        # Drop them before the next header; only admitted complete refs remain.
        del raw, native, identity, row, description
    # Also check the current role when genesis/current are the same complete ref.
    identity = manifest["physical"]["identity"]
    resources = manifest["shared_resources"]
    if (identity["bed_identity_sha256"] != manifest["state"]["accepted_bed_sha256"]
            or identity["vegetation_metadata_sha256"] != resources["vegetation_metadata_sha256"]
            or identity["plan_sha256"] != resources["vegetation"]["plan_sha256"]
            or identity["caller"]["shared_resource_record_sha256"] != bridge._bed._sha256(resources)):
        _fail("native_identity", "native current/vegetation/resource roles differ")
    for field in ("original_inputs_sha256", "case_sha256", "plan_sha256"):
        if manifest["genesis"]["identity"][field] != manifest["physical"]["identity"][field]:
            _fail("native_identity", "fixed original/case/plan native identity differs")
    if (manifest["genesis"]["identity"]["caller"]["static_sha256"]
            != manifest["physical"]["identity"]["caller"]["static_sha256"]):
        _fail("native_identity", "fixed native static identity differs")
    return expected_config


def _history_v2(root, header, chain, limits):
    """Exact accepted metadata prefixes, no original/native array expansion or update replay."""
    from masy import landscape_vegetation as lv

    selected = chain[0][2]
    endpoint = selected["shared_resources"]["vegetation"]
    _keys(endpoint, lv._ENVELOPE_KEYS, "selected vegetation envelope")
    if type(endpoint["source"]) is not str or endpoint["source"] not in ("none", "raster"):
        _fail("vegetation_history", "only admitted native none/raster sources")
    rows = endpoint["receipts"]
    if type(rows) is not list or len(rows) > lv._CEILINGS["max_total_updates"]:
        _fail("vegetation_history", "bounded plain delta receipts required")
    for row in rows:
        _keys(row, lv._RECEIPT_KEYS, "accepted vegetation receipt")
        _integer(row["event_index"], limits.image.max_events - 1, "delta event")
        _integer(row["sequence"], lv._CEILINGS["max_total_updates"], "delta sequence")
        _integer(row["copied_bytes"], lv._CEILINGS["max_run_copy_bytes"], "delta copies")
        if type(row["proposer"]) is not str or row["proposer"] not in lv.PROPOSERS:
            _fail("vegetation_history", "exact prescribed proposer required")
    bindings = endpoint["bindings"]
    if type(bindings) is not list or len(bindings) > limits.image.max_events:
        _fail("vegetation_history", "bounded plain boundary bindings required")
    initial_source = ("none" if any(row["source_transition"] == "none->raster" for row in rows)
        else endpoint["source"])
    initial_version_sha = (selected["receipts"][0]["body"]["vegetation"]["input_version_sha256"]
        if selected["receipts"] else endpoint["version_sha256"])
    for _head, generation, manifest in chain:
        s = generation["sequence"]
        shared = manifest["shared_resources"]
        env = shared["vegetation"]
        _keys(env, lv._ENVELOPE_KEYS, "vegetation envelope")
        cursor = env["cursor"]
        _keys(cursor, lv._CURSOR_KEYS, "vegetation cursor")
        prefix = [row for row in rows if row["event_index"] < s]
        last = {}
        for row in prefix:
            last[row["proposer"]] = row["sequence"]
        expected_cursor = {"next_event_index": s, "last_sequence": last,
            "copied_bytes": sum(row["copied_bytes"] for row in prefix)}
        expected_ledger = {**selected["ledger"], "entries": selected["ledger"]["entries"][:s]}
        if (oi._json(env["receipts"]) != oi._json(prefix)
                or oi._json(cursor) != oi._json(expected_cursor)
                or oi._json(env["bindings"]) != oi._json(bindings[:s])
                or oi._json(manifest["ledger"]) != oi._json(expected_ledger)
                or oi._json(manifest["receipts"]) != oi._json(selected["receipts"][:s])
                or env["schema"] != endpoint["schema"] or env["profile"] != endpoint["profile"]
                or env["plan_sha256"] != endpoint["plan_sha256"]
                or env["initial_native_content_sha256"] != endpoint["initial_native_content_sha256"]
                or shared["prescribed_identity_sha256"] != selected["shared_resources"]["prescribed_identity_sha256"]):
            _fail("vegetation_history", "accepted ledger/binding/delta prefix differs")
        if s:
            body = selected["receipts"][s - 1]["body"]["vegetation"]
            expected_version = body["output_version"]
            expected_version_sha = body["output_version_sha256"]
            expected_native = body["output_native_content_sha256"]
            copies = body["accepted_run_copy_bytes"]
            binding_row = bindings[s - 1]
            binding = lv.BoundaryBinding.from_ledger_record(manifest["ledger"],
                event_index=s - 1, event_kind=binding_row["event_kind"],
                schedule_sha256=binding_row["schedule_sha256"])
            if (oi._json(binding.record()) != oi._json(env["last_boundary_binding"])
                    or binding.binding_sha256 != binding_row["binding_sha256"]):
                _fail("vegetation_history", "complete last native boundary binding differs")
        else:
            expected_version, expected_version_sha, expected_native = (
                0, initial_version_sha, endpoint["initial_native_content_sha256"])
            # Genuine genesis native restore validates its exact initial copy
            # counter/resource hash. Do not duplicate that construction here.
            copies = None
            if (oi._json(manifest["physical"]) != oi._json(selected["genesis"])
                    or bridge._bed._sha256(shared) != selected["genesis"]["identity"]["caller"]["shared_resource_record_sha256"]):
                _fail("vegetation_history", "genuine genesis ref/resource anchor differs")
            if env["last_boundary_binding"] is not None:
                _fail("vegetation_history", "genesis alone has no boundary binding")
        if (type(env["version"]) is not int or env["version"] != expected_version
                or env["version_sha256"] != expected_version_sha
                or env["native_content_sha256"] != expected_native
                or type(shared["accepted_run_copy_bytes"]) is not int
                or (copies is not None and shared["accepted_run_copy_bytes"] != copies)
                or env["source"] != ("raster" if prefix else initial_source)
                or env["source_transition"] != ("none->raster" if any(
                    row["source_transition"] == "none->raster" for row in prefix) else None)):
            _fail("vegetation_history", "accepted endpoint version/native/copies/source differs")
        descriptions = manifest["arrays"]
        soil = descriptions[b._ROLES.index("current_soil")]
        identity = shared["soil_identity"]["array"]
        expected_soil = {"dtype": soil["dtype"], "shape": soil["shape"], "order": "C",
            "nbytes": soil["data_bytes"], "sha256": soil["data_sha256"]}
        if oi._json(identity) != oi._json(expected_soil):
            _fail("vegetation_history", "current soil identity differs from admitted raw row")
        path = root / generation["image"]["relative_path"]
        directory = oi._open_dir(path)
        try:
            oi._verify_written(directory, soil, limits.image)
            _same_directory(path, directory)
        finally:
            os.close(directory)


def _reservation_v2(limits, max_work_bytes):
    """Add the pinned v2 owner/metadata/factory envelope before any recapture.

    This is a logical requested-payload reservation, with a separate whole-worker
    OS address-space ceiling required by the qualification controller. It is not
    a Python heap/RSS or opaque NumPy allocator peak estimate.
    """
    base = _reservation(limits, max_work_bytes)
    image = limits.image
    cells = image.max_cells
    metadata = image.max_metadata_bytes
    # ny*nx <= C implies (ny+2)*(nx+2) <= 3*C+6. Donors have
    # 4*C entries; every other factory array has <=C+1 or five entries.
    q = max(3 * cells + 6, 4 * cells, cells + 1, 5)
    # Exact source census over the pinned D4 factory closure: every expression,
    # including scalars/views, charged as a fresh Q-element float64 allocation;
    # four-direction, <=C and <=C+1 loops and local helpers expanded at C=64.
    graph_bytes = 25130 * q * 8
    increments = {
        # Sum all eight v2 native restore metadata/validation budgets, even
        # though writer preparation/readback releases earlier temporary owners.
        "native_v2_validation_bytes": 8 * image.native_max_bytes,
        # Four ancestry passes can remain referenced at publication. Also
        # retain four standalone writer/parent B records; no ancestor arrays.
        "retained_wrapper_metadata_bytes": (4 * limits.max_accepted + 4) * metadata,
        "retained_control_metadata_bytes": 4 * limits.max_accepted * (
            limits.max_generation_bytes + limits.max_intent_bytes + limits.max_head_bytes),
        # One owning full typed plan and its encoded original record per decode.
        "typed_original_metadata_bytes": 5 * ((4 << 20) + (1 << 20)),
        # Max-live bounded raw/parsed/identity/canonical header slots. Current
        # native headers stream one at a time; all-chain records above stay held.
        "native_header_frame_bytes": 8 << 20,
        "canonical_frame_bytes": 4 * metadata,
        "native_clock_frame_bytes": 4 * metadata,
        # Complete native D4 closure, including transitive scalar-cell loops;
        # no new routing decision or cached graph is installed by this census.
        "graph_requested_payload_bytes": 16 * graph_bytes,
        # Sum eight five-field initial snapshots plus all four three-grid
        # native probes. Each actual probe totals24*C bytes, with no alias credit.
        "additional_dynamic_array_bytes": (8 * 40 + 4 * 24) * cells,
        # Source hash loops can hold old and new 1MiB chunks at assignment.
        "additional_hash_buffer_bytes": 2 << 20,
    }
    memory = base["memory_bytes"] + sum(increments.values())
    if memory > limits.max_memory_bytes:
        _fail("memory_cap", "computed dynamic owner/history/native/factory reservation")
    return {**base, "memory_bytes": memory, "v2_increments": increments,
        "max_original_terrain_graph_builds": 10,
        "max_passive_runner_constructions": 4,
        "max_step_original_terrain_graph_builds": 9,
        "max_total_passive_graph_builds": 16,
        "max_ancestry_passes": 4,
        "max_native_header_parses": 8 * limits.max_accepted,
        "max_streamed_soil_checks": 4 * limits.max_accepted,
        "scope": "bounded default-flat CPU logical reservation; whole-worker OS cap separate"}


def _compare_owners_v2(work, readback, limits):
    """Exact passive readback comparison without another native graph rebuild."""
    for name in (
        "_inventory0",
        "_pool",
        "_soil",
        "_totals",
        "_static_parts",
        "_static_record",
    ):
        b._same(getattr(work, name), getattr(readback, name), limits)
    for a, z in (
        (work._bed.state, readback._bed.state),
        (work._bed.context, readback._bed.context),
        (work._bed.threshold_modifier_state, readback._bed.threshold_modifier_state),
    ):
        b._same(a, z, limits)
    for name in (
        "_accepted_bed",
        "_accepted_private",
        "_n_accepted",
        "_next",
        "_n_classes",
        "_inventory_elements",
        "_static_parts_sha256",
        "_static_sha256",
    ):
        if getattr(work, name) != getattr(readback, name):
            _fail("readback_identity", name)
    if (
        oi._json(work.ledger_record()) != oi._json(readback.ledger_record())
        or dr._canonical_json(work._receipts) != dr._canonical_json(readback._receipts)
        or oi._json(work.accepted.as_record())
        != oi._json(readback.accepted.as_record())
    ):
        _fail("readback_identity", "clocks/receipts/accepted state")
    if (work._routing is None) != (readback._routing is None):
        _fail("readback_identity", "routing presence")
    if work._routing is not None:
        geometry = work._bed.context.geometry
        shape = (geometry.ny, geometry.nx)
        a = b._routing_record(
            work._routing, work._water, shape, work._n_classes, limits
        )
        z = b._routing_record(
            readback._routing, readback._water, shape, readback._n_classes, limits
        )
        if oi._json(a) != oi._json(z):
            _fail("readback_identity", "retained native source/cache identity")
    from masy import landscape_vegetation as lv

    for owner in (work, readback):
        env = owner._bed._dynamic
        if (type(owner._vegetation_plan) is not lv.VegetationPlan
                or env.plan is not owner._vegetation_plan
                or env._native_borrow() is not owner._bed._vegetation):
            _fail("readback_identity", "shared plan/current native vegetation aliases")
        # Fresh native construction deliberately owns a separate initial copy.
        # Exact type/layout/bytes bind it to the static prescribed input; only
        # the current native state and owning typed plan require object aliases.
        b._same(env._native_borrow(initial=True), owner._bed._veg_initial, limits)
    a, z = work._bed._dynamic, readback._bed._dynamic
    for left, right in ((a._native_borrow(initial=True), z._native_borrow(initial=True)),
            (a._native_borrow(), z._native_borrow())):
        b._same(left, right, limits)
    if (oi._json(work._vegetation_plan.plan_record()) != oi._json(readback._vegetation_plan.plan_record())
            or oi._json(work._vegetation_plan.limits.as_record()) != oi._json(readback._vegetation_plan.limits.as_record())
            or oi._json(a.as_record()) != oi._json(z.as_record())
            or type(work._vegetation_copied_bytes) is not int
            or type(readback._vegetation_copied_bytes) is not int
            or work._vegetation_copied_bytes != readback._vegetation_copied_bytes):
        _fail("readback_identity", "complete typed plan/envelope/copy identity")


def create_dynamic_boundary_store(
    path, *, runner, originals, runtime_reference, limits=_DEFAULT, storage_contract
):
    """Create a NEW genesis store; never import an arbitrary progressed runner."""
    BoundaryStoreLimits.__post_init__(limits)
    StorageContract.__post_init__(storage_contract)
    # Admit the complete v2 metadata/recapture envelope before native/plan work.
    # Event work is separately read only after the actual closed runner admission.
    _reservation_v2(limits, 0)
    b._runner_admission_v2(runner, limits.image)
    _reservation_v2(limits, runner._limits.max_work_bytes)
    if (
        runner._next != 0
        or runner._n_accepted != 0
        or runner._receipts
        or runner._stop_receipt is not None
        or runner._active_evidence is not None
    ):
        _fail(
            "fresh_runner_required",
            "only untouched dynamic manual genesis may initialize a store",
        )
    if type(originals) is not oi.OriginalInputsV2:
        _fail("original_identity", "actual newly captured OriginalInputs required")
    env = runner._bed._dynamic
    record = env.as_record()
    if (record["version"] != 0 or record["cursor"]["next_event_index"] != 0
            or record["receipts"] or record["bindings"] or record["last_boundary_binding"] is not None
            or type(runner._vegetation_copied_bytes) is not int
            or runner._vegetation_copied_bytes != 48 * runner._vegetation_plan.cells):
        _fail("fresh_runner_required", "dynamic virgin envelope and exact initial copy counter required")
    source, environment = _runtime_binding(runtime_reference, limits)
    if originals.source_identity != source:
        _fail("original_identity", "captured source differs from complete runtime")
    # The original codec preserves input mapping order; never silently reorder
    # its captured source dictionary merely because control JSON sorts keys.
    captured = oi.capture_original_inputs_v2(
        runner, source_identity=originals.source_identity
    )
    if (
        captured.content_sha256 != originals.content_sha256
        or captured.static_sha256 != originals.static_sha256
    ):
        _fail(
            "original_identity",
            "independently recaptured original does not match runner",
        )
    del captured
    handle = DenseBoundaryStore._open(path, limits, create=True)
    try:
        probe, probe_sha = _probe(handle._root, handle._rootfd)
        os.mkdir(handle._root / "originals", 0o700)
        os.mkdir(handle._root / "generations", 0o700)
        handle._header = {
            "schema": SCHEMA_V2,
            "store_id": uuid.uuid4().hex,
            "profile": PROFILE_V2,
            "source_identity": source,
            "runtime_reference": copy.deepcopy(runtime_reference),
            "environment_sha256": environment,
            "storage_contract": {
                "schema": SCHEMA_V2 + ".storage",
                "operator_declares_local": True,
                "operator_declares_fsync_durable": True,
                "observed_fs_type": probe["observation"]["fs_type"],
                "observed_fs_id": probe["f_fsid"],
                "root_path": str(handle._root),
                "root_dev": probe["root_dev"],
                "root_ino": probe["root_ino"],
                "probe_file_sha256": probe_sha,
            },
            "limits": _limits_record(limits),
            "originals": None,
            "genesis_physical": None,
        }
        handle._admit([], runner._limits.max_work_bytes, initial=True)
        original_path = handle._root / "originals" / ("o-" + uuid.uuid4().hex)
        oi.save_original_inputs_v2(
            original_path,
            originals,
            limits=oi.OriginalPackLimits(
                max_cells=limits.image.max_cells,
                max_classes=limits.image.max_classes,
                max_voxels=limits.image.max_voxels,
                max_cohorts=limits.image.max_cohorts,
                max_events=limits.image.max_events,
            ),
        )
        handle._header["originals"] = _reference_v2(
            original_path, oi.SCHEMA_V2, originals.content_sha256, handle._root
        )
        directory, intent, halt = handle._reserve(0, None, runner._schedule.digest)
        physical = handle._save_physical(directory / "physical", runner, originals)
        handle._header["genesis_physical"] = physical
        handle._header = _seal(handle._header, "store_sha256")
        _sealed_write(handle._root, "STORE.json", handle._header, 128 << 10)
        handle._store_raw = oi._json(handle._header)
        image = handle._save_image(directory / "image", runner, physical)
        head = handle._seal_generation(directory, intent, None, image)
        prepared, chain = handle._load(head)
        _compare_owners_v2(runner, _owner_v2(prepared), limits.image)
        state = handle._idle_state(head, prepared, chain[0][1])
        del prepared
        handle._publish(head, state, None, halt)
        return handle
    except BaseException:
        handle.close()
        raise

def recover_dynamic_boundary_store(path, *, expected_store_file_sha256, limits=_DEFAULT):
    """Recover ONLY selected HEAD with retained originals/native sources; no fallback."""
    BoundaryStoreLimits.__post_init__(limits)
    _digest(expected_store_file_sha256, "expected STORE file hash")
    handle = DenseBoundaryStore._open(path, limits, create=False)
    try:
        _inventory(handle._root, limits, expected_root=os.fstat(handle._rootfd))
        header, raw = _record(
            handle._root / "STORE.json",
            128 << 10,
            links=_published_links(handle._root, "STORE.json"),
        )
        _check_seal(header, _STORE, "store_sha256", SCHEMA_V2)
        if (
            _sha(raw) != expected_store_file_sha256
            or header["profile"] != PROFILE_V2
            or type(header["store_id"]) is not str
            or not _NONCE.fullmatch(header["store_id"])
        ):
            _fail("store_identity", "store hash/id/profile mismatch")
        stored_limits = _limits_load(header["limits"])
        if _limits_record(stored_limits) != _limits_record(limits):
            _fail("limit_identity", "restore must use exact persisted limits")
        b._reference_v2(header["originals"], oi.SCHEMA_V2)
        b._reference_v2(header["genesis_physical"], "maple-exact-physical-checkpoint/1")
        if not re.fullmatch(
            r"originals/o-[0-9a-f]{32}", header["originals"]["relative_path"]
        ) or not re.fullmatch(
            r"generations/g-0000-[0-9a-f]{32}/physical",
            header["genesis_physical"]["relative_path"],
        ):
            _fail("store_identity", "original/genesis roles")
        if not _dynamic_store(header):
            _fail("store_profile", "named dynamic recovery requires v2")
        handle._header, handle._store_raw = header, raw
        handle._stable()
        head, head_raw = handle._head()
        prepared, chain = handle._load(head)
        state = handle._idle_state(head, prepared, chain[0][1])
        del prepared
        _sync_tree(handle._root)
        handle._stable()
        if handle._head()[1] != head_raw:
            _fail("parent_drift", "HEAD changed during recovery")
        # Final fsync, then only the preprepared metadata state assignment.
        os.fsync(handle._rootfd)
        handle._state = state
        return handle
    except BaseException:
        handle.close()
        raise


# --- StageA private water-parent borrow (static stores only; no HEAD/publication promise) ---
_WATER_PARENT_KEY = object()


def _water_parent_ready(handle, parent):
    if type(handle) is not DenseBoundaryStore:
        _fail("water_parent", "exact DenseBoundaryStore required")
    handle._binding()
    if _active_water_store(handle._header):
        _fail("water_parent", "active-water stores issue their own reserved parent context")
    state = handle._state
    if handle._entered is not False:
        _fail("handle_reentrant", "store operation already in progress")
    if type(state) is not _State or state.phase != "idle" or type(state.head) is not dict:
        _fail("water_parent", "idle store with retained parent head required")
    if _dynamic_store(handle._header):
        _fail("water_parent", "static store only")
    if type(state.token) is not str or parent != state.token:
        _fail("water_parent", "expected parent differs from current store token")
    return state


def _abandon_issued(runner):
    from masy import dense_water_pause as pause
    lease = runner._active_evidence
    try:
        if type(lease) is not pause._WaterLease:
            return "released"
        if lease.owner() is not runner:
            _fail("water_cleanup", "issued runner carries a foreign water lease")
        if lease.phase == "busy":
            _fail("water_busy", "issued runner has busy water work; refusing to claim idle")
        if lease.phase == "paused":
            if lease.token is None:
                _fail("water_cleanup", "paused water lease has no token")
            pause.abandon_water(runner, lease.token)
            return "abandoned"
        if lease.phase not in ("closed", "halted"):
            _fail("water_cleanup", "unknown water lease phase")
        return "released"
    finally:
        lease = runner = None


class _WaterParent:
    """One-shot private borrow of the unchanged accepted parent; StageA only."""

    __slots__ = ("_handle", "_pid", "_thread", "_phase", "_owner", "_issued", "_head",
                 "_token", "_source", "_environment", "_outcome", "_cleanup", "_image", "_selection", "__weakref__")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("water parent context is final")

    def __init__(self, key, handle, token, head, source, environment):
        if key is not _WATER_PARENT_KEY:
            _fail("private_carrier", "water parent contexts come only from borrow_water_parent")
        self._handle, self._pid, self._thread = handle, os.getpid(), threading.get_ident()
        self._phase, self._owner, self._issued = "new", None, None
        self._head, self._token = head, token
        self._source, self._environment = source, environment
        self._outcome, self._cleanup, self._image = None, None, None
        self._selection = None

    def __reduce_ex__(self, protocol):
        _fail("private_carrier", "water parent context cannot be copied or pickled")

    @property
    def cleanup_record(self):
        return self._outcome, self._cleanup

    def _own(self):
        if type(self) is not _WaterParent or os.getpid() != self._pid or threading.get_ident() != self._thread:
            _fail("handle_binding", "water parent used outside creating PID/thread")

    def _light(self, h):
        if self._selection is not None:
            return _active_module()._parent_light(self, h)
        h._binding()
        state = h._state
        if (h._entered is not True or type(state) is not _State or state.phase != "idle"
                or state.token != self._token or type(state.head) is not dict
                or oi._json(state.head) != self._head or _dynamic_store(h._header)
                or oi._json(h._header["source_identity"]) != self._source
                or h._header["environment_sha256"] != self._environment):
            _fail("water_parent", "parent store state or metadata drifted")
        record, _raw = h._head()
        if oi._json(record) != self._head:
            _fail("water_parent", "HEAD differs from borrowed parent")

    def _release(self):
        if self._selection is not None:
            return _active_module()._release_parent(self)
        h, self._handle = self._handle, None
        self._owner = self._issued = None
        if self._phase == "entered" and h is not None:
            h._entered = False
        self._phase = "closed"

    def __enter__(self):
        if self._selection is not None:
            return _active_module()._enter_parent(self)
        self._own()
        if self._phase != "new":
            _fail("water_parent", "one-shot context already entered or closed")
        h = self._handle
        _water_parent_ready(h, self._token)
        self._phase, h._entered = "entered", True
        prepared = owner = chain = None
        try:
            if oi._json(h._state.head) != self._head:
                _fail("water_parent", "store parent changed before entry")
            h._stable()
            record, _raw = h._head()
            if oi._json(record) != self._head:
                _fail("water_parent", "HEAD differs from borrowed parent")
            prepared, chain = h._load(record)
            self._image = oi._json(chain[0][1]["image"])
            chain = None
            owner = _owner_for(h._header, prepared)
            prepared = None
            self._owner, owner = owner, None
        except BaseException:
            prepared = owner = chain = None
            self._image = None
            self._outcome = "entry_failed"
            self._release()
            raise
        return self

    def take_runner(self):
        import weakref
        self._own()
        if self._phase != "entered" or self._issued is not None or self._owner is None:
            _fail("water_parent", "runner already taken, not entered, or closed")
        self._light(self._handle)
        runner = self._owner
        issued = weakref.ref(runner)
        self._owner, self._issued = None, issued
        return runner

    def check(self, runner):
        self._own()
        if (self._phase != "entered" or self._issued is None or runner is None
                or self._issued() is not runner):
            _fail("water_parent", "foreign, unissued or closed water parent runner")
        self._light(self._handle)

    def close(self):
        self._own()
        if self._phase == "entered" and self._issued is not None:
            from masy import dense_water_pause as pause
            live = self._issued()
            active = None if live is None else live._active_evidence
            if type(active) is pause._WaterLease and active.phase in ("busy", "finishing", "exporting"):
                self._cleanup = ("BoundaryStoreError", "water_busy")
                _fail("water_busy", "active close is non-consuming; parent guard remains held")
        if self._phase == "closed":
            return
        error = runner = None
        try:
            self._owner = None
            if self._phase != "entered":
                self._outcome = "never_entered"
            else:
                issued = self._issued
                runner = None if issued is None else issued()
                self._issued = None
                self._outcome = (("never_taken" if issued is None else "taken_collected")
                                 if runner is None else _abandon_issued(runner))
                runner = None
                h = self._handle
                h._stable()
                self._light(h)
        except BaseException as exc:
            code = getattr(exc, "code", None)
            self._cleanup = (type(exc).__name__[:64],
                             code if type(code) is str and len(code) <= 64 else None)
            error = exc
        finally:
            runner = None
            self._release()
        if error is not None:
            try:
                raise error
            finally:
                error = None

    def __exit__(self, exc_type, exc, tb):
        exc = tb = None
        try:
            self.close()
        except BaseException:
            if exc_type is None:
                raise
        return False


def borrow_water_parent(handle, *, expected_parent):
    """Private one-shot borrow of the unchanged accepted parent; no store writes or receipt."""
    _digest(expected_parent, "water parent token")
    state = _water_parent_ready(handle, expected_parent)
    return _WaterParent(_WATER_PARENT_KEY, handle, state.token, oi._json(state.head),
                        oi._json(handle._header["source_identity"]),
                        handle._header["environment_sha256"])


# New opt-in durable active-water schema; default callers never import its owner.
SCHEMA_ACTIVE_WATER = "masy.dense_boundary_store.activewater.v1"
PROFILE_ACTIVE_WATER = "default_dense_numpy_activewater_v1"


def _active_water_store(header):
    return type(header) is dict and header.get("schema") == SCHEMA_ACTIVE_WATER and header.get("profile") == PROFILE_ACTIVE_WATER


def _active_module():
    from masy import dense_active_water_store
    return dense_active_water_store


def create_active_water_boundary_store(path, *, runner, originals, runtime_reference,
                                       limits=_DEFAULT, storage_contract):
    return _active_module().create_active_water_boundary_store(path, runner=runner, originals=originals,
        runtime_reference=runtime_reference, limits=limits, storage_contract=storage_contract)


def recover_active_water_boundary_store(path, *, expected_store_file_sha256, limits=_DEFAULT):
    return _active_module().recover_active_water_boundary_store(path,
        expected_store_file_sha256=expected_store_file_sha256, limits=limits)


def _committed_binding_common(reference, limits, *, _reuse=None):
    """Read-only opt-in committed provenance for the existing v1 digest tuple."""
    cap = min(limits.max_runtime_metadata_bytes, _CL_LOCK_BYTES)
    for key in ("source_manifest_path", "source_manifest_file_sha256"):
        if reference[key] is not None:
            _fail("runtime_reference", "committed source-manifest slots must be null")
    store = _canonical_path(reference["cpu_store"], "cpu store")
    lock_id = _digest(reference["cpu_lock_id"], "cpu lock id")
    _digest(reference["cpu_lock_file_sha256"], "cpu lock raw hash")
    _keys(reference["module_roots"], ("masy", "maple", "maple_syrup"), "module roots")
    for value in reference["module_roots"].values():
        _canonical_path(value, "module root")
    _canonical_path(reference["native_bundle_manifest_path"], "native manifest")
    _digest(reference["native_bundle_manifest_file_sha256"], "native manifest SHA")
    flags = _committed_cpu_flags()
    # This fixed cap covers referenced sealed artifacts, not a new allocation or
    # materialization. The unchanged environment observer has its separate caps.
    with _HeldInventory(512 << 20) as held:
        lock_path = os.path.join(store, "locks", lock_id + ".json")
        raw = held.read(lock_path, cap)
        if _sha(raw) != reference["cpu_lock_file_sha256"]:
            _fail("committed_lock", "retained CPU lock raw SHA differs")
        _committed_json(raw, cap, "cpu lock")
        try:
            lock, loaded_path = runtime.load_lock_ref(store, lock_id, cap)
            if str(loaded_path) != lock_path or runtime._canonical_bytes(lock) != raw:
                _fail("committed_lock", "actual strict lock loader returned different bytes")
            if committed_runtime_reference(
                cpu_store=store, cpu_lock_id=lock_id,
                cpu_lock_file_sha256=reference["cpu_lock_file_sha256"], lock=lock,
            ) != reference:
                _fail("runtime_reference", "reference does not derive exactly from committed lock")
            _committed_process(lock, flags)
            admitted, controls = _committed_sources(lock, held, cap)
            from maple_syrup import dependency_bundle as native

            record, native_limits, native_controls = _committed_native(
                lock, reference, admitted, held, cap, native
            )
            controls.update(native_controls)
            controls[lock_path] = raw
            held.check()
            _committed_verify_sources(lock, admitted, held, min(cap, _CL_OBJECT_BYTES))
            checked = native.verify_bundle(
                lock["objects"]["maple_bundle"]["path"], limits=native_limits
            )
            if native._canonical(checked) + b"\n" != native_controls[
                reference["native_bundle_manifest_path"]
            ]:
                _fail("native_binding", "actual verified native manifest differs")
            held.check()
            binding = native.check_runtime(
                lock["objects"]["maple_materialized"]["path"], limits=native_limits
            )
            if (
                binding["bundle_id"] != record["bundle_id"]
                or binding["package_source_digest"]
                != lock["objects"]["maple_bundle"]["package_digest"]
            ):
                _fail("native_binding", "actual native in-process binding differs")
            held.check()
            result = (_committed_identity(lock, admitted, reference["module_roots"], flags)
                      if _reuse is None else _committed_identity_common(
                          lock, admitted, reference["module_roots"], flags, _reuse=_reuse))
            for filename, original in controls.items():
                if held.read(filename, cap) != original:
                    _fail("identity_drift", "closing control metadata differs")
            held.check()
            return copy.deepcopy(result[0]), result[1]
        except (runtime.RuntimeRefusal, rw.WorkerRefusal) as exc:
            raise BoundaryStoreError(exc.code, exc.detail) from exc
        except oi.OriginalInputsError as exc:
            raise BoundaryStoreError(exc.code, exc.detail) from exc
        except (KeyError, TypeError, ValueError) as exc:
            # Native BundleError is a ValueError with no code/detail attributes.
            # Keep our already bounded component refusals intact.
            if isinstance(exc, BoundaryStoreError):
                raise
            raise BoundaryStoreError("committed_binding", str(exc)) from exc


def _committed_identity_common(lock, admitted, roots, flags, *, _reuse=None):
    """Current interpreter/bootstrap/resolver and C environment; no child/write."""
    _committed_process(lock, flags)
    python = runtime.interpreter_identity(
        sys.executable, runtime.RuntimeLimits(max_compiled_bytes=_CL_INTERPRETER_BYTES)
    )
    if (
        runtime._python_stable(python) != lock["runtime"]["python"]
        or python["venv"] is not None
        and os.path.realpath(sys.prefix) != python["venv"]["dir"]
    ):
        _fail("committed_runtime", "actual interpreter/venv differs from lock")
    bootstrap = runtime.sealed_bootstrap(lock["objects"]["sources"]["masy"]["path"])
    if _sha(bootstrap.encode("utf-8")) != lock["runtime"]["bootstrap_sha256"]:
        _fail("committed_runtime", "sealed bootstrap SHA differs from lock")
    runtime.check_import_roots(lock["runtime"]["sys_path_prefix"])
    rw.selected_distributions([
        {"name": name, "dir": lock["distributions"][name]["overlay_dir"],
         "files": lock["distributions"][name]["files"]}
        for name in ("maple", "maple-syrup", "masy")
    ], path=None)
    environment = (_environment_identity() if _reuse is None
                   else _environment_identity_common(_reuse=_reuse))
    executable = environment["interpreter_identity"]
    if (
        executable["path"] != python["realpath"]
        or executable["size"] != python["size"]
        or executable["sha256"] != python["sha256"]
    ):
        _fail("environment_identity", "C executable observer differs from actual interpreter")
    tables = {}
    for package, kind in (("masy", "masy"), ("maple_syrup", "syrup"), ("maple", "maple")):
        tables[package] = {
            rel: {"size_bytes": size, "sha256": digest}
            for rel, size, digest, _blob in admitted[kind][1]
            if rel == "pyproject.toml" or rel.startswith("src/")
        }
    # Observers and native delegates may have imported additional modules.
    for name, module in tuple(sys.modules.items()):
        package = name.split(".")[0]
        filename = getattr(module, "__file__", None)
        if package in roots and filename:
            path, expected = Path(filename), Path(roots[package])
            if not path.is_absolute() or not path.is_relative_to(expected):
                _fail("module_origin", name)
            if str(path.relative_to(expected.parent)) not in tables[package]:
                _fail("module_origin", "loaded module absent from committed source table")
    ids = {
        "kind": "source_manifest_prototype",
        "masy": _sha(oi._json(tables["masy"])),
        "syrup": _sha(oi._json(tables["maple_syrup"])),
        "maple": lock["objects"]["maple_bundle"]["bundle_id"],
    }
    oi._source_identity(ids)
    return ids, _sha(oi._json(environment))


def _environment_identity_common(*, _reuse=None):
    """Reuse operational/compiled observers; do not select or change a runtime."""
    from masy.importcheck import environment_record
    from masy.runtime_worker import ENVIRONMENT_DISTRIBUTIONS, compiled_identities

    observed = environment_record()
    versions = {}
    for name in ENVIRONMENT_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "interpreter_identity": observed["executable_identity"],
        "python_version": observed["python_version"],
        "implementation": observed["implementation"],
        "numpy": observed["numpy"],
        "optional_distribution_versions": observed["optional_distribution_versions"],
        "distribution_versions": versions,
        "compiled": (compiled_identities(
            dict(sys.modules), max_count=4000, max_bytes=4 << 30
        ) if _reuse is None else rw._compiled_identities_scoped(
            _reuse._worker, _reuse._nonce, dict(sys.modules), max_count=4000, max_bytes=4 << 30)),
    }


def _runtime_binding_common(reference, limits, *, _reuse=None):
    _keys(reference, _RUNTIME, "runtime reference")
    if reference["schema"] != SCHEMA + ".runtime":
        _fail("runtime_reference", "schema")
    if reference["kind"] == _CL_KIND:
        return (_committed_binding(reference, limits) if _reuse is None
                else _committed_binding_common(reference, limits, _reuse=_reuse))
    if reference["kind"] != "source_manifest_prototype":
        _fail("runtime_kind_unsupported_v1", "v1 has no committed-lock success branch")
    if any(
        reference[k] is not None
        for k in ("cpu_store", "cpu_lock_id", "cpu_lock_file_sha256")
    ):
        _fail("runtime_reference", "prototype CPU lock slots must be null")
    roots = reference["module_roots"]
    _keys(roots, ("masy", "maple", "maple_syrup"), "module roots")
    for value in [
        *roots.values(),
        reference["source_manifest_path"],
        reference["native_bundle_manifest_path"],
    ]:
        if (
            type(value) is not str
            or not Path(value).is_absolute()
            or str(Path(value)) != value
            or os.path.normpath(value) != value
        ):
            _fail(
                "runtime_reference", "canonical absolute retained runtime path required"
            )
    for field in ("source_manifest_file_sha256", "native_bundle_manifest_file_sha256"):
        _digest(reference[field], field)
    # Complete source metadata is finite and raw-admitted before parsing.
    raw = _read(reference["source_manifest_path"], limits.max_runtime_metadata_bytes)
    if _sha(raw) != reference["source_manifest_file_sha256"]:
        _fail("source_identity", "retained source manifest changed")
    manifest = oi._strict_json(
        raw,
        SimpleNamespace(
            max_metadata_bytes=limits.max_runtime_metadata_bytes,
            max_nodes=100000,
            max_depth=24,
        ),
    )
    _keys(
        manifest,
        ("schema", "source_identity", "module_roots", "files", "environment"),
        "source manifest",
    )
    if manifest["schema"] != SCHEMA + ".sources" or manifest["module_roots"] != roots:
        _fail("source_identity", "source schema/root mismatch")
    oi._source_identity(manifest["source_identity"])
    _keys(manifest["files"], roots, "complete package tables")
    admitted, total = [], 0
    for package, table in manifest["files"].items():
        if type(table) is not dict or not 1 <= len(table) <= 2000:
            _fail("source_cap", "finite source file count")
        base = Path(roots[package]).parent
        observed = set()
        # Reuse existing source scope selection; no second export/selection engine.
        scoped, _directories = runtime._walk_object(base / "src", max_entries=4000)
        observed.update("src/" + rel for rel in scoped)
        observed.add("pyproject.toml")
        if observed != set(table):
            _fail("source_identity", "complete scoped runtime file set differs")
        for rel, row in table.items():
            _keys(row, ("size_bytes", "sha256"), "source file")
            if type(rel) is not str or not (
                rel == "pyproject.toml" or rel.startswith("src/")
            ):
                _fail("source_identity", "outside source scope")
            runtime._check_relpath(rel, launcher_allowed=False)
            _integer(row["size_bytes"], 64 << 20, "source bytes")
            _digest(row["sha256"], "source SHA")
            path = base / rel
            parent = oi._open_dir(path.parent)
            try:
                info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_size != row["size_bytes"]
                    or info.st_nlink != 1
                    or info.st_mode & 0o222
                ):
                    _fail("source_identity", "source type/size differs")
            finally:
                os.close(parent)
            total += row["size_bytes"]
            if total > 128 << 20:
                _fail("source_cap", "complete source serialized budget")
            admitted.append((path, row))
    for path, row in admitted:
        parent = oi._open_dir(path.parent)
        try:
            actual, _blob, size, _data = runtime._hash_path(
                path, budget=row["size_bytes"]
            )
            _same_directory(path.parent, parent)
            if actual != row["sha256"] or size != row["size_bytes"]:
                _fail("source_identity", "source content differs")
        except runtime.RuntimeRefusal as exc:
            raise BoundaryStoreError(exc.code, exc.detail) from exc
        finally:
            os.close(parent)
    from maple_syrup import dependency_bundle

    if (
        Path(reference["native_bundle_manifest_path"]).name
        != dependency_bundle.MANIFEST_NAME
    ):
        _fail("native_binding", "literal native bundle manifest required")
    native_raw = _read(
        reference["native_bundle_manifest_path"], limits.max_runtime_metadata_bytes
    )
    if _sha(native_raw) != reference["native_bundle_manifest_file_sha256"]:
        _fail("native_binding", "retained native bundle manifest changed")
    native = dependency_bundle.verify_bundle(
        Path(reference["native_bundle_manifest_path"]).parent
    )
    binding = dependency_bundle.check_runtime(Path(roots["maple"]).parent.parent)
    if (
        binding["bundle_id"] != native["bundle_id"]
        or _read(
            reference["native_bundle_manifest_path"], limits.max_runtime_metadata_bytes
        )
        != native_raw
        or dependency_bundle._canonical(native) + b"\n" != native_raw
    ):
        _fail(
            "native_binding", "actual checked bundle/materialized/manifest bytes differ"
        )
    native_table = {
        row["path"].removeprefix("source/"): {
            "size_bytes": row["size"],
            "sha256": row["sha256"],
        }
        for row in native["files"]
        if row["path"].startswith("source/")
    }
    if native_table != manifest["files"]["maple"]:
        _fail("native_binding", "MAPLE source table is not exact native bundle source")
    ids = manifest["source_identity"]
    if ids["maple"] != native["bundle_id"]:
        _fail("native_binding", "native source identity is not actual bundle")
    for package, slot in (("masy", "masy"), ("maple_syrup", "syrup")):
        if _sha(oi._json(manifest["files"][package])) != ids[slot]:
            _fail("source_identity", "source table digest mismatch")
    environment = _environment_identity()
    if oi._json(environment) != oi._json(manifest["environment"]):
        _fail(
            "environment_identity",
            "retained interpreter/distribution/compiled identity differs",
        )
    # Import observers/native helpers before the final module-origin sweep.
    for name, module in tuple(sys.modules.items()):
        package = name.split(".")[0]
        filename = getattr(module, "__file__", None)
        if package in roots and filename:
            path = Path(filename)
            expected = Path(roots[package])
            if not path.is_absolute() or not path.is_relative_to(expected):
                _fail("module_origin", name)
            rel = str(path.relative_to(expected.parent))
            if rel not in manifest["files"][package]:
                _fail("module_origin", "loaded module absent from frozen manifest")
    return copy.deepcopy(ids), _sha(oi._json(environment))


# Exact operation owner; never stores a binding/environment/Native result.
_RUNTIME_VALIDATION_KEY = object()
_RUNTIME_REUSE_REFERENCE_BYTES = 128 << 10


class _RuntimeValidationOwner:
    __slots__ = ('_key', '_handle', '_nonce', '_pid', '_thread', '_start', '_boot',
                 '_reference_raw', '_limits', '_worker', '_phase', '_image', '_charge')

    def __init__(self, key, handle):
        if key is not _RUNTIME_VALIDATION_KEY:
            _fail('runtime_scope', 'private operation owner required')
        self._key, self._handle, self._nonce = key, handle, object()
        self._pid, self._thread, self._start, self._boot = (
            handle._pid, handle._thread, handle._start, handle._boot)
        self._reference_raw, self._worker, self._charge = None, None, None
        self._limits, self._phase = handle._limits, 'pending'
        self._image = _runtime_scope_image()

    def __reduce_ex__(self, protocol):
        raise TypeError('runtime validation owner is operation-local')


def _runtime_scope_image():
    functions = (_runtime_binding, _runtime_binding_common,
                 _committed_binding, _committed_binding_common,
                 _committed_identity, _committed_identity_common,
                 _environment_identity, _environment_identity_common,
                 _RuntimeValidationOwner.__init__, _RuntimeValidationOwner.__reduce_ex__,
                 _new_runtime_validation, _checked_runtime_validation,
                 _drop_runtime_validation, _final_runtime_validation,
                 _runtime_validation_charge, _operation_runtime_binding,
                 _runtime_scope_image)
    images = tuple(rw._compiled_function_image(f) for f in functions)
    if any(x is None for x in images):
        return None
    return (images, (rw, rw._compiled_function_image, rw._same_compiled_image,
                     rw._compiled_reuse_charge, rw._compiled_image_charge, rw._new_compiled_reuse,
                     rw._compiled_identities_scoped, _RuntimeValidationOwner,
                     _RUNTIME_VALIDATION_KEY, os.getpid, threading.get_ident),
            (_RUNTIME_REUSE_REFERENCE_BYTES,))


def _new_runtime_validation(handle):
    if type(handle) is not DenseBoundaryStore:
        _fail('runtime_scope', 'actual entered active-water handle required')
    if handle._runtime_validation is not None:
        _fail('runtime_scope', 'nested runtime validation owner')
    # No optimization for a prototype, default C, dynamic C or other profile.
    if (not _active_water_store(handle._header)
            or handle._header['runtime_reference']['kind'] != _CL_KIND):
        return
    if type(handle) is not DenseBoundaryStore or not handle._entered:
        _fail('runtime_scope', 'actual entered active-water handle required')
    handle._binding()
    owner = _RuntimeValidationOwner(_RUNTIME_VALIDATION_KEY, handle)
    if owner._image is not None:
        handle._runtime_validation = owner


def _drop_runtime_validation(handle):
    owner, handle._runtime_validation = handle._runtime_validation, None
    if type(owner) is _RuntimeValidationOwner:
        owner._phase = 'closed'
        worker = owner._worker
        if type(worker) is rw._CompiledReuse:
            worker._closed = True
            worker._rows = worker._next = worker._image = None
        owner._worker = owner._reference_raw = owner._image = owner._charge = None
        owner._handle = owner._limits = None


def _checked_runtime_validation(handle):
    owner = handle._runtime_validation
    if owner is None:
        return None
    if (type(owner) is not _RuntimeValidationOwner or owner._key is not _RUNTIME_VALIDATION_KEY
            or owner._handle is not handle or owner._limits is not handle._limits
            or owner._phase not in ('pending', 'live', 'final') or not handle._entered
            or os.getpid() != owner._pid or threading.get_ident() != owner._thread
            or handle._start != owner._start or handle._boot != owner._boot):
        _fail('runtime_scope', 'foreign/closed/fork/thread runtime validation owner')
    if not rw._same_compiled_image(owner._image, _runtime_scope_image()):
        _drop_runtime_validation(handle)
        return None
    return owner


def _final_runtime_validation(handle):
    owner = _checked_runtime_validation(handle)
    if owner is not None:
        owner._phase = 'final'
        if owner._worker is not None:
            owner._worker._force_full = True


def _runtime_validation_charge():
    worker = rw._compiled_reuse_charge()
    image = _runtime_scope_image()
    if worker is None or image is None:
        return None
    pointer = sys.getsizeof((None,)) - sys.getsizeof(())
    image_bytes = rw._compiled_image_charge(image)
    fixed = (sys.getsizeof(object.__new__(_RuntimeValidationOwner))
             + len(_RuntimeValidationOwner.__slots__) * (pointer + sys.getsizeof((1 << 64) - 1))
             + sys.getsizeof(object()) + 2 * image_bytes
             + 2 * (_RUNTIME_REUSE_REFERENCE_BYTES + sys.getsizeof(b''))
             + rw._compiled_frame_charge(tuple(record[0] for record in image[0])))
    # Worker nine-field dict and this actual twelve-field expansion can overlap.
    fixed += sys.getsizeof(dict.fromkeys(('rows', 'fixed', 'bytes', 'max_count',
             'max_path_chars', 'row_bytes', 'image_functions', 'image_operators',
             'image_constants', 'owner_fixed_bytes', 'owner_image_functions',
             'owner_image_operators'))) + 12 * sys.getsizeof((1 << 64) - 1)
    return {**worker, 'owner_fixed_bytes': fixed,
            'bytes': worker['bytes'] + fixed,
            'owner_image_functions': len(image[0]),
            'owner_image_operators': len(image[1])}


def _operation_runtime_binding(handle):
    owner = _checked_runtime_validation(handle)
    reference, limits = handle._header['runtime_reference'], handle._limits
    if owner is None:
        return _runtime_binding(reference, limits)
    if owner._charge is None:
        _drop_runtime_validation(handle)
        return _runtime_binding(reference, limits)
    try:
        _plain_metadata(reference, _RUNTIME_REUSE_REFERENCE_BYTES, 'runtime reuse reference')
    except BoundaryStoreError:
        # Eligibility failure cannot make the physical/runtime domain wider.
        _drop_runtime_validation(handle)
        return _runtime_binding(reference, limits)
    current = oi._json(reference)
    if owner._reference_raw is None:
        owner._reference_raw = current
        owner._worker = rw._new_compiled_reuse(owner._nonce, max_count=4000, max_bytes=4 << 30)
        if owner._phase == 'final':
            owner._worker._force_full = True
        else:
            owner._phase = 'live'
    elif owner._reference_raw != current:
        _drop_runtime_validation(handle)
        return _runtime_binding(reference, limits)
    return _runtime_binding_common(reference, limits, _reuse=owner)
