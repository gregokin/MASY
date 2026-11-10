"""Bounded, strict document reading for scenario, event and weather files (stdlib + optional PyYAML).

Shared by the Phase 0 scheduler modules (`masy.scenario`, `masy.event_files`,
`masy.weather`). Rules:

- A document is a regular, non-symlink file whose size is within the declared
  bound; it is read once with a size re-check and its SHA-256 is recorded.
- JSON is parsed strictly: duplicate keys, NaN/Infinity, nesting deeper than
  the bound, more entries or more text than the bound all refuse. Floats are
  accepted only when finite; bool stays bool (never silently an int).
- YAML is a JSON subset here: when PyYAML is importable, `yaml.SafeLoader` is
  used with duplicate-key, non-string-key, alias and non-JSON-type refusal;
  `.nan`/`.inf` refuse like JSON. Without PyYAML a `.yaml`/`.yml` document
  refuses with `yaml_unavailable` (nothing is installed). No tags beyond the
  plain JSON scalars/containers are accepted.
- References between files are relative, canonical, control-character free,
  contain no `..`, and resolve inside the referencing document's directory.

Importing this module imports neither NumPy nor any native model; PyYAML is
imported lazily and only for `.yaml`/`.yml` inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DOCUMENT_SUFFIXES",
    "IOLimits",
    "ScenarioIOError",
    "canonical_json",
    "load_document",
    "parse_strict_json",
    "parse_strict_yaml",
    "read_bounded",
    "relative_reference",
    "sha256_hex",
    "walk_plain_json",
]

DOCUMENT_SUFFIXES = (".json", ".yaml", ".yml")
_READ_CHUNK = 1 << 20


class ScenarioIOError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class IOLimits:
    """Byte/graph bounds applied before any document graph is built."""

    max_file_bytes: int = 1 << 20
    max_entries: int = 65536
    max_text_chars: int = 1 << 20
    max_depth: int = 32
    max_path_chars: int = 4096
    max_files: int = 256

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if type(value) is not int or value < 1:
                raise ScenarioIOError("invalid_limit", f"IOLimits.{name} must be a positive plain int")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stamp(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def read_bounded(path: Path, max_bytes: int) -> tuple[bytes, dict[str, Any]]:
    """Read one regular file within `max_bytes`; refuse symlinks, specials,
    oversize and concurrent change. Returns (bytes, pin) with the SHA-256."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ScenarioIOError("invalid_limit", "max_bytes must be a positive plain int")
    path = Path(path)
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ScenarioIOError("document_missing", f"{path} does not exist") from exc
    except OSError as exc:
        raise ScenarioIOError("document_unreadable", f"cannot stat {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode):
        raise ScenarioIOError("document_symlink", f"{path} is a symlink; symlinks are refused")
    if not stat.S_ISREG(before.st_mode):
        raise ScenarioIOError("document_not_regular", f"{path} is not a regular file")
    if before.st_size > max_bytes:
        raise ScenarioIOError("document_too_large", f"{path} is {before.st_size} bytes > {max_bytes}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ScenarioIOError("document_unreadable", f"cannot open {path}: {exc}") from exc
    try:
        held = os.fstat(fd)
        if not stat.S_ISREG(held.st_mode):
            raise ScenarioIOError("document_not_regular", f"{path} is not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(_READ_CHUNK, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ScenarioIOError("document_too_large", f"{path} grew beyond {max_bytes} bytes while read")
            chunks.append(chunk)
    finally:
        os.close(fd)
    data = b"".join(chunks)
    if _stamp(before) != _stamp(held) or _stamp(before) != _stamp(path.lstat()) or len(data) != before.st_size:
        raise ScenarioIOError("document_changed", f"{path} changed while being read")
    return data, {"path": str(path), "size": len(data), "sha256": sha256_hex(data)}


def walk_plain_json(value: Any, limits: IOLimits, where: str = "document") -> None:
    """Refuse non-JSON leaves, non-string keys, non-finite floats and graphs
    beyond the entry/text/depth bounds. Iterative; never recursive."""
    todo: list[tuple[Any, int]] = [(value, 1)]
    count = chars = 0
    while todo:
        item, depth = todo.pop()
        count += 1
        if count > limits.max_entries:
            raise ScenarioIOError("entry_quota_exceeded", f"{where} has more than {limits.max_entries} entries")
        if type(item) is dict:
            if depth > limits.max_depth:
                raise ScenarioIOError("depth_quota_exceeded", f"{where} nests deeper than {limits.max_depth}")
            for key, sub in item.items():
                if type(key) is not str:
                    raise ScenarioIOError("non_string_key", f"{where} has a non-string key {key!r:.80}")
                chars += len(key)
                if chars > limits.max_text_chars:
                    raise ScenarioIOError("text_quota_exceeded", f"{where} has more than {limits.max_text_chars} text chars")
                todo.append((sub, depth + 1))
        elif type(item) is list:
            if depth > limits.max_depth:
                raise ScenarioIOError("depth_quota_exceeded", f"{where} nests deeper than {limits.max_depth}")
            todo.extend((sub, depth + 1) for sub in item)
        elif type(item) is str:
            chars += len(item)
            if chars > limits.max_text_chars:
                raise ScenarioIOError("text_quota_exceeded", f"{where} has more than {limits.max_text_chars} text chars")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ScenarioIOError("nonfinite_number", f"{where} contains a non-finite number")
        elif item is not None and type(item) not in (int, bool):
            raise ScenarioIOError("unsupported_leaf", f"{where} contains an unsupported value of type {type(item).__name__}")


def parse_strict_json(data: bytes, limits: IOLimits, where: str = "document") -> Any:
    """Strict JSON: duplicate keys, NaN/Infinity, depth/entry/text bounds refuse.
    The structural depth prepass runs before the JSON graph is built."""
    if type(data) is not bytes:
        raise ScenarioIOError("invalid_input", "parse_strict_json requires bytes")
    if len(data) > limits.max_file_bytes:
        raise ScenarioIOError("document_too_large", f"{where} exceeds {limits.max_file_bytes} bytes")
    level = 0
    quoted = escaped = False
    for byte in data:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            level += 1
            if level > limits.max_depth:
                raise ScenarioIOError("depth_quota_exceeded", f"{where} nests deeper than {limits.max_depth}")
        elif byte in (93, 125):
            level -= 1

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, item in rows:
            if key in out:
                raise ScenarioIOError("duplicate_key", f"{where} repeats key {key!r:.80}")
            out[key] = item
        return out

    def nonfinite(token: str) -> Any:
        raise ScenarioIOError("nonfinite_number", f"{where} contains {token}")

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)
    except UnicodeDecodeError as exc:
        raise ScenarioIOError("invalid_encoding", f"{where} is not UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ScenarioIOError("invalid_json", f"{where}: {exc}") from exc
    except RecursionError as exc:  # defence only: the byte prepass above bounds the nesting first
        raise ScenarioIOError("depth_quota_exceeded", f"{where} nests too deeply for the JSON decoder") from exc
    except ScenarioIOError:
        raise
    except ValueError as exc:  # e.g. an integer literal beyond the interpreter's digit limit
        raise ScenarioIOError("invalid_json", f"{where}: {str(exc)[:200]}") from exc
    walk_plain_json(value, limits, where)
    return value


def parse_strict_yaml(data: bytes, limits: IOLimits, where: str = "document") -> Any:
    """YAML restricted to the JSON data model via `yaml.SafeLoader`; duplicate
    keys, non-string keys, aliases/anchors, `.nan`/`.inf` and non-JSON scalar
    types (dates, binary, sets, ...) refuse. Refuses `yaml_unavailable` when
    PyYAML cannot be imported; nothing is installed."""
    if type(data) is not bytes:
        raise ScenarioIOError("invalid_input", "parse_strict_yaml requires bytes")
    if len(data) > limits.max_file_bytes:
        raise ScenarioIOError("document_too_large", f"{where} exceeds {limits.max_file_bytes} bytes")
    try:
        import yaml  # noqa: PLC0415 - optional, lazily imported
    except ImportError as exc:
        raise ScenarioIOError("yaml_unavailable", f"{where}: PyYAML is not importable; use JSON") from exc

    class _Loader(yaml.SafeLoader):
        """Depth and entry quotas are enforced DURING composition (before any
        nested node is expanded), so a deeply nested or huge document refuses
        with the configured code instead of recursing or allocating first."""

        _depth = 0
        _nodes = 0

        def compose_node(self, parent: Any, index: Any) -> Any:
            if self.check_event(yaml.AliasEvent):
                raise ScenarioIOError("yaml_alias_refused", f"{where} uses a YAML anchor/alias; refused")
            self._nodes += 1
            if self._nodes > limits.max_entries:
                raise ScenarioIOError("entry_quota_exceeded", f"{where} has more than {limits.max_entries} entries")
            return super().compose_node(parent, index)

        def _enter(self) -> None:
            self._depth += 1
            if self._depth > limits.max_depth:
                raise ScenarioIOError("depth_quota_exceeded", f"{where} nests deeper than {limits.max_depth}")

        def compose_sequence_node(self, anchor: Any) -> Any:
            self._enter()
            try:
                return super().compose_sequence_node(anchor)
            finally:
                self._depth -= 1

        def compose_mapping_node(self, anchor: Any) -> Any:
            self._enter()
            try:
                return super().compose_mapping_node(anchor)
            finally:
                self._depth -= 1

        def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if type(key) is not str:
                    raise ScenarioIOError("non_string_key", f"{where} has a non-string YAML key {key!r:.80}")
                if key in out:
                    raise ScenarioIOError("duplicate_key", f"{where} repeats key {key!r:.80}")
                out[key] = self.construct_object(value_node, deep=deep)
            return out

    _Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                            lambda loader, node: loader.construct_mapping(node, deep=True))
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ScenarioIOError("invalid_encoding", f"{where} is not UTF-8: {exc}") from exc
    try:
        value = yaml.load(text, Loader=_Loader)  # noqa: S506 - restricted SafeLoader subclass
    except ScenarioIOError:
        raise
    except yaml.YAMLError as exc:
        raise ScenarioIOError("invalid_yaml", f"{where}: {str(exc)[:400]}") from exc
    except RecursionError as exc:  # defence only: the composition quotas above refuse first
        raise ScenarioIOError("depth_quota_exceeded", f"{where} nests too deeply for the YAML composer") from exc
    except ValueError as exc:  # e.g. an integer scalar beyond the interpreter's digit limit
        raise ScenarioIOError("invalid_yaml", f"{where}: {str(exc)[:200]}") from exc
    walk_plain_json(value, limits, where)
    return value


def load_document(path: Path, limits: IOLimits, where: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Read and strictly parse one `.json`/`.yaml`/`.yml` document; returns (value, pin)."""
    path = Path(path)
    label = where or str(path)
    suffix = path.suffix.lower()
    if suffix not in DOCUMENT_SUFFIXES:
        raise ScenarioIOError("unsupported_document_suffix", f"{label}: suffix {suffix!r} is not one of {DOCUMENT_SUFFIXES}")
    data, pin = read_bounded(path, limits.max_file_bytes)
    if suffix == ".json":
        value = parse_strict_json(data, limits, label)
    else:
        value = parse_strict_yaml(data, limits, label)
    return value, pin


def relative_reference(value: Any, base_dir: Path, limits: IOLimits, where: str) -> Path:
    """A canonical relative reference resolving inside `base_dir` (real path)."""
    if type(value) is not str or not 0 < len(value) <= limits.max_path_chars:
        raise ScenarioIOError("invalid_reference", f"{where} must be a non-empty string of at most {limits.max_path_chars} chars")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ScenarioIOError("invalid_reference", f"{where} contains control characters")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or str(candidate) != value or not candidate.parts:
        raise ScenarioIOError("invalid_reference", f"{where} must be a canonical relative path without '..'")
    base_real = Path(os.path.realpath(base_dir))
    resolved = Path(os.path.realpath(base_real / candidate))
    try:
        resolved.relative_to(base_real)
    except ValueError as exc:
        raise ScenarioIOError("reference_escapes_base", f"{where} resolves outside {base_real}") from exc
    if resolved != (base_real / candidate):
        raise ScenarioIOError("reference_not_canonical", f"{where} resolves through a symlink; refused")
    return resolved
