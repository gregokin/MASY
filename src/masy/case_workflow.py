"""Adopt a genuine MAPLE case package without rebuilding its sediment bed.

This is CPU input/adoption plumbing, not an event runner or a restart codec.
Native compilation/loading and every scientific validator remain upstream.
Known copies are admitted separately from the whole worker's OS limits.
"""
from __future__ import annotations

import copy
import dataclasses
import enum
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
from types import SimpleNamespace
from typing import Any

from masy import dense_bed as _bed

__all__ = ["CaseWorkflowError", "CaseInputLimits", "CaseInspection", "CaseInputs", "AdoptedCase",
           "resolve_case_inputs", "adopt_case_inputs"]

CASE_SCHEMA = "masy.maple_case_initial.v1"
_INPUT_TOKEN = object()
_AUTHORED = ("case.yaml", "model.yaml", "wind.yaml")


class CaseWorkflowError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclasses.dataclass(frozen=True)
class CaseInputLimits:
    """Input I/O eligibility, never a decoder-heap bound or a science parameter."""
    max_authored_file_bytes: int
    max_source_files: int
    max_source_file_bytes: int
    max_source_bytes: int
    max_metadata_bytes: int
    max_compiled_bytes: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 1:
                raise CaseWorkflowError("invalid_input_limit", f"{field.name} must be a positive plain int")


def _native() -> SimpleNamespace:
    import numpy as np
    from maple.case_tools.compilers.case_compiler import CompiledCase, load_compiled_case
    from maple.case_tools.generators.availability import build_availability_builder_maps
    from maple.case_tools.importers.resolve import ImportSourceEntry
    from maple.case_tools.transforms.log import TransformLog
    from maple.case_tools.validators.compiled_case import validate_compiled_case, validate_seeded_case_state
    from maple.config import config_to_dict, validate_config
    from maple.core.parameters.case_initialization import resolve_initial_available_fraction_array
    from maple.experiment.orchestrator import compile_case_package
    from maple.experiment.schema import ResolvedCasePackage, resolve_case_package
    return SimpleNamespace(**{name: value for name, value in locals().items()})


def _canonical(value: Any) -> Any:
    """Lossless metadata/array identity, including Native path-bearing specs."""
    import numpy as np
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        # Metadata diagnostics can legitimately contain NaN/inf. Preserve the
        # exact binary64 image; physical/config validators still own their
        # finite-value requirements, and no numerical tolerance is invented.
        return {"binary64": struct.pack(">d", value).hex()}
    if isinstance(value, Path):
        return {"path": str(value)}
    if type(value) is np.ndarray:
        return _bed._array_digest(value)
    if isinstance(value, np.generic):
        return {"numpy_scalar": value.dtype.str, "value": _canonical(value.item())}
    if isinstance(value, enum.Enum):
        return {"enum": f"{type(value).__module__}.{type(value).__qualname__}", "value": _canonical(value.value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {field.name: _canonical(getattr(value, field.name)) for field in dataclasses.fields(value)}}
    if type(value) in (tuple, list):
        return {"sequence": type(value).__name__, "items": [_canonical(item) for item in value]}
    if type(value) is dict:
        pairs = [[_canonical(key), _canonical(item)] for key, item in value.items()]
        return {"mapping": sorted(pairs, key=lambda pair: _bed._canonical_json(pair[0]))}
    if type(value) in (set, frozenset):
        return {"set": sorted((_canonical(item) for item in value), key=_bed._canonical_json)}
    raise CaseWorkflowError("unsupported_metadata", f"unsupported case metadata {type(value).__module__}.{type(value).__qualname__}")


def _record(resolved: Any, compiled: Any) -> dict[str, Any]:
    # Every CompiledCase field is included, including masks, mobile, expected
    # inventories, provenance, import_sources and transform logs. No live data
    # array is converted to a JSON list.
    return {"resolved": _canonical(resolved), "compiled": _canonical(compiled)}


def _input_record(resolved: Any, compiled: Any, file_pins: Any) -> dict[str, Any]:
    return {"case": _record(resolved, compiled), "files": _canonical(file_pins)}


def _stamp(info: Any) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _pin(path: Path, root: Path, ceiling: int) -> dict[str, Any]:
    try:
        before = path.lstat()
    except OSError as error:
        raise CaseWorkflowError("input_file_refused", f"cannot inspect required input: {path}") from error
    if (not stat.S_ISREG(before.st_mode) or path.is_symlink() or path.resolve() != path
            or before.st_size > ceiling):
        raise CaseWorkflowError("input_file_refused", f"not a canonical bounded regular input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        size = 0
        while True:
            block = stream.read(min(1 << 20, ceiling - size + 1))
            if not block:
                break
            size += len(block)
            if size > ceiling:
                raise CaseWorkflowError("input_file_limit", f"input exceeds its declared byte limit: {path}")
            digest.update(block)
        held = os.fstat(stream.fileno())
    if _stamp(before) != _stamp(held) or _stamp(before) != _stamp(path.lstat()) or size != before.st_size:
        raise CaseWorkflowError("input_changed", f"input changed while hashing: {path}")
    return {"path": str(path.relative_to(root)), "size": size, "sha256": digest.hexdigest()}


def _inventory(root: Path, limits: CaseInputLimits, *, compiled: bool) -> list[dict[str, Any]]:
    rows = [_pin(root / name, root, limits.max_authored_file_bytes) for name in _AUTHORED]
    pending = [root / "source"] if (root / "source").exists() else []
    count, total = 0, 0
    while pending:
        directory = pending.pop()
        if directory.is_symlink() or directory.resolve() != directory or not directory.is_dir():
            raise CaseWorkflowError("input_directory_refused", "case source directories must be canonical ordinary directories")
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                if count > limits.max_source_files:
                    raise CaseWorkflowError("source_count_limit", "source file/directory count exceeds the input policy")
                path = directory / entry.name
                if entry.is_symlink():
                    raise CaseWorkflowError("input_symlink", "source symlinks are outside this input profile")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                else:
                    row = _pin(path, root, limits.max_source_file_bytes)
                    total += row["size"]
                    if total > limits.max_source_bytes:
                        raise CaseWorkflowError("source_byte_limit", "case source bytes exceed the input policy")
                    rows.append(row)
    if compiled:
        rows.append(_pin(root / "provenance.yaml", root, limits.max_metadata_bytes))
        pending = [root / name for name in ("processed", "masks", "diagnostics") if (root / name).exists()]
        generated = 0
        while pending:
            directory = pending.pop()
            if directory.is_symlink() or directory.resolve() != directory or not directory.is_dir():
                raise CaseWorkflowError("compiled_directory_refused", "compiled directories must be canonical")
            with os.scandir(directory) as entries:
                for entry in entries:
                    count += 1
                    if count > limits.max_source_files:
                        raise CaseWorkflowError("input_count_limit", "complete input/artifact entry count exceeds input policy")
                    path = directory / entry.name
                    if entry.is_symlink():
                        raise CaseWorkflowError("compiled_symlink", "compiled input symlinks refuse")
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    else:
                        row = _pin(path, root, limits.max_compiled_bytes)
                        generated += row["size"]
                        if generated > limits.max_compiled_bytes:
                            raise CaseWorkflowError("compiled_byte_limit", "compiled input bytes exceed policy")
                        rows.append(row)
    return sorted(rows, key=lambda row: row["path"])


def _check_import_bindings(resolved: Any, compiled: Any, root: Path, pins: Any) -> None:
    """Join Native's whole import inventory to the already authenticated files."""
    n = _native()
    by_path = {row["path"]: row for row in pins}
    inventories = [resolved.import_sources]
    if compiled is not None:
        inventories.append(compiled.import_sources)
    images = []
    for inventory in inventories:
        rows = []
        for entry in inventory:
            if type(entry) is n.ImportSourceEntry:
                row = entry.to_dict()
            elif type(entry) is dict:
                row = entry
            else:
                raise CaseWorkflowError("unsupported_import_record", "Native import inventory has an unsupported entry")
            path = Path(row["authored_path"])
            pin = by_path.get(str(path))
            if (path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "source"
                    or Path(row["resolved_path"]) != root / path or pin is None
                    or pin["sha256"] != row["sha256"] or pin["size"] != row["bytes"]):
                raise CaseWorkflowError("case_import_mismatch", "Native imported source differs from the held source inventory")
            rows.append(row)
        images.append(_canonical(rows))
    if compiled is not None and images[0] != images[1]:
        raise CaseWorkflowError("case_import_mismatch", "resolved and compiled Native import inventories differ")


@dataclasses.dataclass(frozen=True)
class CaseInspection:
    case_dir: Path
    resolved: Any
    file_pins: tuple[dict[str, Any], ...]
    input_limits: CaseInputLimits


@dataclasses.dataclass(frozen=True)
class CaseInputs:
    case_dir: Path
    resolved: Any
    compiled: Any
    file_pins: tuple[dict[str, Any], ...]
    input_limits: CaseInputLimits
    identity_sha256: str
    _token: Any = dataclasses.field(repr=False, compare=False)

    def verify(self) -> None:
        if self._token is not _INPUT_TOKEN:
            raise CaseWorkflowError("foreign_case_inputs", "CaseInputs must come from resolve_case_inputs")
        if type(self.input_limits) is not CaseInputLimits:
            raise CaseWorkflowError("invalid_input_limit", "held CaseInputLimits type changed")
        self.input_limits.__post_init__()
        _check_pair(self.resolved, self.compiled)
        _check_import_bindings(self.resolved, self.compiled, self.case_dir, self.file_pins)
        if (self.resolved.case_yaml_path.parent.resolve() != self.case_dir
                or self.compiled.case_path.resolve() != self.case_dir):
            raise CaseWorkflowError("case_root_mismatch", "Native package/compiled root differs from the held case root")
        if _bed._sha256(_input_record(self.resolved, self.compiled, self.file_pins)) != self.identity_sha256:
            raise CaseWorkflowError("case_input_mutated", "held complete Native case metadata or arrays changed")
        if _inventory(self.case_dir, self.input_limits, compiled=True) != list(self.file_pins):
            raise CaseWorkflowError("case_source_changed", "authored/imported/compiled case files changed")


@dataclasses.dataclass(frozen=True)
class AdoptedCase:
    bed: Any
    inputs: CaseInputs
    copy_plan: dict[str, int]

    @property
    def initial_mobile_state(self) -> Any:
        return self.inputs.compiled.initial_mobile_state


def _check_pair(resolved: Any, compiled: Any) -> None:
    n = _native()
    if type(resolved) is not n.ResolvedCasePackage or type(compiled) is not n.CompiledCase:
        raise CaseWorkflowError("invalid_native_case", "exact Native ResolvedCasePackage and CompiledCase required")
    n.validate_config(resolved.config)
    n.validate_config(compiled.config)
    if _canonical(n.config_to_dict(resolved.config)) != _canonical(n.config_to_dict(compiled.config)):
        raise CaseWorkflowError("case_config_mismatch", "resolved and compiled Native configs differ")
    if getattr(compiled, "ledger_reconciliation_baseline", None) is not None:
        raise CaseWorkflowError("unsupported_reconciliation_baseline", "initial nonNone baseline is not supported")
    if _canonical(resolved.stratigraphy) != _canonical(compiled.stratigraphy):
        raise CaseWorkflowError("case_stratigraphy_mismatch", "compiled layers differ from the authored resolved layers")
    # For layered/imported cases compiled.grain_mixture is a derived Native
    # summary, not the authored mixture (which legitimately can be None).
    if resolved.grain_mixture is not None and _canonical(resolved.grain_mixture) != _canonical(compiled.grain_mixture):
        raise CaseWorkflowError("case_mixture_mismatch", "compiled uniform mixture differs from authored mixture")
    for name, array in _bed._require_host_arrays((resolved, compiled), "Native case"):
        if not array.flags.c_contiguous or (array.dtype.kind == "f" and array.dtype != n.np.float64):
            raise CaseWorkflowError("unsupported_case_array", f"{name} must be C-contiguous native float64 when floating")


def resolve_case_inputs(case_dir: str | Path, *, mode: str, input_limits: CaseInputLimits) -> CaseInspection | CaseInputs:
    """Resolve the three-file package; validate, compile once, or load natively.

    Compile is non-overwriting and transactional in MAPLE. Load still requires
    the complete authored package and Native provenance. No transport runs.
    """
    if type(input_limits) is not CaseInputLimits:
        raise CaseWorkflowError("invalid_input_limit", "input_limits must be CaseInputLimits")
    input_limits.__post_init__()
    if mode not in ("validate", "compile", "load") or type(mode) is not str:
        raise CaseWorkflowError("invalid_mode", "mode must be validate, compile or load")
    root = Path(case_dir).resolve(strict=True)
    if not root.is_dir():
        raise CaseWorkflowError("invalid_case_directory", "case_dir must be a directory")
    before = _inventory(root, input_limits, compiled=mode == "load")
    n = _native()
    resolved = n.resolve_case_package(root)
    _check_import_bindings(resolved, None, root, before)
    if mode == "validate":
        if _inventory(root, input_limits, compiled=False) != before:
            raise CaseWorkflowError("case_source_changed", "case changed while resolving")
        return CaseInspection(root, resolved, tuple(before), input_limits)
    if mode == "compile":
        resolved, compiled = n.compile_case_package(root, overwrite=False)
    else:
        compiled = n.load_compiled_case(root)
    _check_pair(resolved, compiled)
    after = _inventory(root, input_limits, compiled=True)
    by_path = {row["path"]: row for row in after}
    authored_after = _inventory(root, input_limits, compiled=False)
    original_authored = [row for row in before if row["path"] in _AUTHORED or row["path"].startswith("source/")]
    if authored_after != original_authored or any(by_path.get(row["path"]) != row for row in before):
        raise CaseWorkflowError("case_source_changed", "case changed while compiling/loading")
    _check_import_bindings(resolved, compiled, root, after)
    record = _input_record(resolved, compiled, tuple(after))
    if len(_bed._canonical_json(record).encode()) > input_limits.max_metadata_bytes:
        raise CaseWorkflowError("case_metadata_limit", "complete Native case identity exceeds metadata policy")
    return CaseInputs(root, resolved, compiled, tuple(after), input_limits, _bed._sha256(record), _INPUT_TOKEN)


def _array_cost(value: Any) -> tuple[int, int]:
    leaves = _bed._require_host_arrays(value, "case copies")
    copied, roots = 0, 0
    for _, array in leaves:
        copied += int(array.nbytes)
        root = array
        while isinstance(root.base, type(array)):
            root = root.base
        roots += int(root.nbytes)
    return copied, roots  # Per occurrence; no identity/alias credit.


def _fallback(resolved: Any, compiled: Any, root: Path) -> Any:
    n = _native()
    logs = []  # Private; NEVER merged into either caller's transform logs.
    maps, all_map = n.build_availability_builder_maps(
        resolved.sediment_availability, resolved.config.geometry, compiled.config.seed,
        case_dir=root, log_sink=logs)
    prefix = "sediment_availability.builder"
    prior = [log for log in compiled.transform_logs
             if type(log) is n.TransformLog and (log.field_label == prefix or log.field_label.startswith(prefix + "_by_class."))]
    if (_canonical([log.to_dict() for log in logs]) != _canonical([log.to_dict() for log in prior])):
        raise CaseWorkflowError("availability_transform_mismatch", "fresh Native availability transform logs differ or are unavailable")
    source_rows = [entry.to_dict() if type(entry) is n.ImportSourceEntry else entry for entry in compiled.import_sources]
    for log in logs:
        if type(log) is not n.TransformLog or not any(
                row["authored_path"] == log.source["authored_path"] and row["sha256"] == log.source["sha256"]
                and row["format"] == log.source["format"] for row in source_rows):
            raise CaseWorkflowError("availability_source_mismatch", "fresh Native availability source is not in compiled provenance")
    array = n.resolve_initial_available_fraction_array(
        resolved.sediment_availability, resolved.config.geometry,
        [grain.class_id for grain in resolved.config.grain_classes.classes],
        builder_maps_by_class=maps, all_class_builder_map=all_map)
    shape = (resolved.config.geometry.ny, resolved.config.geometry.nx, len(resolved.config.grain_classes.classes))
    if type(array) is not n.np.ndarray or array.dtype != n.np.float64 or array.shape != shape:
        raise CaseWorkflowError("invalid_fallback", "Native availability fallback has incompatible shape/dtype")
    violations = _bed._native().validate_available_fraction_array(array, shape)
    if violations:
        raise CaseWorkflowError("invalid_fallback", "; ".join(violations))
    return array


def _validate_compiled(compiled: Any, resolved: Any, fallback: Any) -> None:
    n = _native()
    cfg = resolved.config
    if compiled.initial_mobile_state is not None:
        n.validate_seeded_case_state(compiled.voxel_column, compiled.active_layer, compiled.committed_topography,
            compiled.sediment_availability, compiled.initial_mobile_state, compiled.expected_total_mass_by_class_kg,
            cfg.geometry, cfg.grain_classes, cfg.numerics.mass_resolution_kg)
    else:
        n.validate_compiled_case(compiled.voxel_column, compiled.active_layer, compiled.committed_topography,
            compiled.sediment_ledger, cfg.geometry, cfg.grain_classes, cfg.numerics.mass_resolution_kg,
            None, compiled.expected_total_mass_by_class_kg, vegetation=compiled.vegetation,
            sediment_availability=compiled.sediment_availability, threshold_modifier_state=compiled.threshold_modifier_state,
            expected_mass_by_cell_class_kg=compiled.expected_mass_by_cell_class_kg,
            expected_mass_by_level_class_kg=compiled.expected_mass_by_level_class_kg,
            n_deposit_passes=max(len(compiled.stratigraphy.layers), 1) if compiled.stratigraphy is not None else 1)
        expected = _bed._native().sediment_availability_state_from_active_layer(compiled.active_layer.mass_kg, fallback)
        if _canonical(expected) != _canonical(compiled.sediment_availability):
            raise CaseWorkflowError("availability_policy_mismatch", "Native initial holdings disagree with the rebuilt fallback")


def adopt_case_inputs(inputs: CaseInputs, *, bed_limits: _bed.DenseBedLimits) -> AdoptedCase:
    """Make an owned working bed and a hash-authenticated complete initial case.

    Initial array leaves are read-only; mutable metadata is checked on use.
    Initial mobile is retained here, not an allocated working scheduler pool.
    """
    if type(inputs) is not CaseInputs or type(bed_limits) is not _bed.DenseBedLimits:
        raise CaseWorkflowError("invalid_adoption_input", "CaseInputs and DenseBedLimits required")
    bed_limits.__post_init__()
    inputs.verify()
    n = _native()
    cfg, compiled = inputs.resolved.config, inputs.compiled
    g = cfg.geometry
    shape = compiled.voxel_column.mass_kg.shape
    if len(shape) != 4 or shape[:2] != (g.ny, g.nx) or shape[3] != len(cfg.grain_classes.classes):
        raise CaseWorkflowError("case_geometry_mismatch", "Native dense column shape differs from case geometry/classes")
    if shape[2] > bed_limits.max_voxels_per_column:
        raise CaseWorkflowError("voxel_capacity_exceeded", "compiled depth exceeds max_voxels_per_column")
    if g.dx_m != g.dy_m or g.voxel_growth != _bed._native().DISABLED_VOXEL_GROWTH:
        raise CaseWorkflowError("unsupported_case_geometry", "first case adoption requires square cells and fixed voxel capacity")
    copied, roots = _array_cost((inputs.resolved, compiled))
    state_bytes, _ = _array_cost((compiled.voxel_column, compiled.active_layer, compiled.water,
        compiled.sediment_ledger, compiled.committed_topography, compiled.sediment_availability))
    cells, classes = g.ny * g.nx, len(cfg.grain_classes.classes)
    # This known payload/copy plan is NOT a Python/native heap estimate.
    # The three encoded metadata payload owners are input identity, owned
    # identity and the new bed record; object/decoder overhead remains under
    # the separately enforced original whole-worker AS limit.
    # Known original roots + complete owned initial envelope + SIX working
    # physical holders + possible next/error six-holder state. Vegetation,
    # thresholds and initial mobile are retained in the owned initial forest,
    # not copied into another mutable working pool by this factory.
    # Native fallback/maps/forward-check arrays get a separate allowance;
    # their internal solver/decoder scratch stays under the whole-worker cap.
    metadata = len(_bed._canonical_json(_input_record(inputs.resolved, compiled, inputs.file_pins)).encode())
    fallback_bytes = 8 * cells * (6 * classes + 2)
    plan = {"original_array_roots_bytes": roots, "owned_initial_array_bytes": copied,
            "working_state_bytes": state_bytes, "next_or_error_state_bytes": state_bytes,
            "fallback_and_comparison_allowance_bytes": fallback_bytes, "metadata_copy_bytes": 3 * metadata}
    required = sum(plan.values())
    if required > bed_limits.host_memory_cap_bytes:
        raise CaseWorkflowError("host_memory_cap_exceeded", "known original/current/next/case copy owners exceed host_memory_cap_bytes")
    if state_bytes > bed_limits.stage_copy_max_bytes:
        raise CaseWorkflowError("stage_copy_cap_exceeded", "actual complete bed copy exceeds stage_copy_max_bytes")
    # No owned domain copy or rebuilt fallback before the checks above.
    fallback = _fallback(inputs.resolved, compiled, inputs.case_dir)
    inputs.verify()  # Reauthenticate imported builder inputs after Native reads.
    _validate_compiled(compiled, inputs.resolved, fallback)
    owned_resolved, owned_compiled = copy.deepcopy((inputs.resolved, compiled))
    for _, array in _bed._require_host_arrays((owned_resolved, owned_compiled), "held initial case"):
        array.flags.writeable = False
    owned = CaseInputs(inputs.case_dir, owned_resolved, owned_compiled, copy.deepcopy(inputs.file_pins), inputs.input_limits,
                       inputs.identity_sha256, _INPUT_TOKEN)
    # The private owning factory takes independent working copies, never a DEM.
    bed = _bed._adopt_case_inputs(owned, fallback, bed_limits, plan)
    inputs.verify()
    owned.verify()
    return AdoptedCase(bed, owned, plan)


def _verify_adopted_case(bed: Any) -> None:
    inputs = bed._adopted_case_inputs
    inputs.verify()
    if inputs.identity_sha256 != bed._case_record["native_case_identity_sha256"]:
        raise _bed.DenseBedError("case_input_mutated", "complete Native initial case identity changed")
