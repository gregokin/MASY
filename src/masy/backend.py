"""Unwired, metadata-only description of explicitly supported native Dense holders.

This is not physical validation, allocation admission, solver selection, or a
transport/GPU qualification. No array values are read. The native MAPLE walkers
provide leaf discovery and placement accounting; the small closed schemas below
only prevent unknown holders from disappearing through those walkers. Native
numerical validators and source guards remain the caller's responsibility.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

__all__ = ["BackendMetadataError", "BackendProvenance", "DenseBackendMetadata", "inspect_dense_metadata"]


class BackendMetadataError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclasses.dataclass(frozen=True)
class BackendProvenance:
    """Caller-supplied labels only; this module verifies no source or device identity."""

    source_lock_id: str | None = None
    interpreter: str | None = None
    physical_device_uuid: str | None = None
    visible_device_ordinal: int | None = None
    visible_devices: str | None = None

    def __post_init__(self) -> None:
        BackendProvenance.as_record(self)

    def as_record(self) -> dict[str, str | int | None]:
        fields = dataclasses.fields(BackendProvenance)
        try:
            attributes = vars(self)
        except TypeError as exc:
            raise BackendMetadataError("invalid_provenance", "expected declared provenance attributes") from exc
        if type(self) is not BackendProvenance or set(attributes) != {f.name for f in fields}:
            raise BackendMetadataError("invalid_provenance", "expected only the declared provenance labels")
        record = {}
        for field in fields:
            value = getattr(self, field.name)
            expected = int if field.name == "visible_device_ordinal" else str
            if value is not None and type(value) is not expected:
                raise BackendMetadataError("invalid_provenance", f"{field.name} must be {expected.__name__} or None")
            record[field.name] = value
        return record


@dataclasses.dataclass(frozen=True)
class DenseBackendMetadata:
    """Inspector results are immutable metadata, with no authoritative arrays/native state."""

    # Each entry is (role, native leaf ordinal, shape, dtype, bytes).
    arrays: tuple[tuple[str, int, tuple[int, ...], str, int], ...]
    placement_inventory: tuple[tuple[str, int | bool], ...]
    provenance: BackendProvenance

    def as_record(self) -> dict[str, Any]:
        return {"schema": "masy.dense_backend_metadata.v1", "backend": "numpy", "representation": "dense",
                "qualification": "metadata_only", "transport_ready": False, "device_qualified": False,
                "source_binding_verified": False, "provenance": BackendProvenance.as_record(self.provenance),
                "arrays": [{"role": role, "ordinal": i, "shape": list(shape), "dtype": dtype, "nbytes": size}
                           for role, i, shape, dtype, size in self.arrays],
                "native_placement_inventory": dict(self.placement_inventory),
                "inventory_scope": "native generic tree placement visits; not actual transfers, resident bytes, "
                                   "overlapping-view allocation or a whole-driver peak",
                "init_false_policy": "unsupported schema; never silently omitted",
                "values_validated": False}


def _native() -> SimpleNamespace:
    # Lazy CPU imports only. No backend resolver, optional dependency probe,
    # device/context selection, numerical validator, or solver is invoked.
    import numpy as np
    from maple.core.backend import array_leaves, resolve_operand_namespace
    from maple.core.backend.arrays import array_leaf_inventory
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.sediment_availability import SedimentAvailabilityState
    from maple.core.types.sediment_ledger import SedimentLedgerState
    from maple.core.types.threshold_modifiers import ThresholdModifierState
    from maple.core.types.topographic_commit import CommittedTopographyState
    from maple.core.types.vegetation import VegetationRasterState, VegetationState
    from maple.core.types.voxel import VoxelColumnState
    from maple.core.types.water import WaterState
    from maple_syrup.sediment_bed import BedState, TerrainReference

    # Field names and init flags are versioned admission boundaries, not inferred
    # permission to accept arbitrary/new dataclasses or opaque resource holders.
    ledger_fields = ("pending_bed_mass_change_kg", "pending_compensation_kg", "process_totals_kg",
                     "process_totals_compensation_kg", "numerical_residual_positive_kg",
                     "numerical_residual_positive_compensation_kg", "numerical_residual_negative_kg",
                     "numerical_residual_negative_compensation_kg", "actual_mass_positive_kg",
                     "actual_mass_positive_compensation_kg", "actual_mass_negative_kg",
                     "actual_mass_negative_compensation_kg", "largest_residual_kg", "n_physical_touches")
    schemas = {
        BedState: {"voxel_column": VoxelColumnState, "active_layer": ActiveLayerState, "water": WaterState,
                   "ledger": SedimentLedgerState, "committed_topography": CommittedTopographyState,
                   "sediment_availability": SedimentAvailabilityState},
        VoxelColumnState: {"mass_kg": np.ndarray}, ActiveLayerState: {"mass_kg": np.ndarray},
        WaterState: {"depth_m": np.ndarray, "mobile_mass_by_cell_class_kg": np.ndarray},
        SedimentLedgerState: {name: float if name == "largest_residual_kg" else np.ndarray
                             for name in ledger_fields},
        SedimentAvailabilityState: {"available_mass_kg": np.ndarray, "bound_mass_kg": np.ndarray},
        CommittedTopographyState: {"reference_elevation_m": np.ndarray, "elevation_offset_m": np.ndarray,
                                  "committed_active_layer_mass_by_class_kg": np.ndarray,
                                  "committed_voxel_mass_by_class_kg": np.ndarray,
                                  "last_commit_time_s": float, "commit_count": int},
        VegetationState: {"source": str, "raster": VegetationRasterState, "objects": type(None)},
        VegetationRasterState: dict.fromkeys(("height_m", "cover_fraction", "porosity", "drag_coefficient",
                                             "frontal_area_density"), np.ndarray),
        ThresholdModifierState: dict.fromkeys(("gravimetric_moisture_fraction", "clay_fraction",
                                              "physical_crust_multiplier", "chemical_cementation_multiplier",
                                              "biological_binding_multiplier"), np.ndarray),
        TerrainReference: {"initial_full_elevation_m": np.ndarray, "initial_committed_elevation_m": np.ndarray,
                           "export_receiver_full": np.ndarray, "active_mask": np.ndarray,
                           "friction_factor": np.ndarray, "dx_m": float, "allow_masked_nodata": bool,
                           "nodata_value": (float, type(None)), "allow_pit_storage": bool,
                           "allow_flat_storage": bool},
    }
    return SimpleNamespace(np=np, schemas=schemas, array_leaves=array_leaves,
                           resolve_operand_namespace=resolve_operand_namespace, array_leaf_inventory=array_leaf_inventory,
                           BedState=BedState, ActiveLayerState=ActiveLayerState, VoxelColumnState=VoxelColumnState,
                           WaterState=WaterState, CommittedTopographyState=CommittedTopographyState,
                           VegetationState=VegetationState, ThresholdModifierState=ThresholdModifierState,
                           TerrainReference=TerrainReference)


def _check_holder(n: SimpleNamespace, value: Any, expected: Any, where: str) -> None:
    """Check only known holder/type metadata before a native walker can ignore it."""
    allowed = expected if type(expected) is tuple else (expected,)
    if type(value) not in allowed:
        raise BackendMetadataError("unsupported_holder", f"{where}: expected exact native/schema type")
    if expected is n.np.ndarray:
        # Object arrays could hide unknown holders. Numerical shape/dtype/value
        # requirements belong to the native validators, not this description.
        if value.dtype.hasobject:
            raise BackendMetadataError("unsupported_array", f"{where}: object arrays are outside this schema")
        return
    schema = n.schemas.get(type(value))
    if schema is None:  # an explicitly declared scalar, not arbitrary metadata
        return
    if tuple((f.name, f.init) for f in dataclasses.fields(value)) != tuple((name, True) for name in schema):
        raise BackendMetadataError("unsupported_schema", f"{where}: native field/init schema changed")
    try:
        attributes = vars(value)
    except TypeError as exc:
        raise BackendMetadataError("unsupported_schema", f"{where}: native attribute layout changed") from exc
    if set(schema) - set(attributes):
        raise BackendMetadataError("missing_field", f"{where}: missing declared instance fields")
    if set(attributes) - set(schema):
        raise BackendMetadataError("unsupported_schema", f"{where}: undeclared attributes")
    for name, kind in schema.items():
        try:
            child = getattr(value, name)
        except AttributeError as exc:
            raise BackendMetadataError("missing_field", f"{where}.{name}") from exc
        _check_holder(n, child, kind, f"{where}.{name}")
    if type(value) is n.VegetationState and value.source not in ("none", "raster"):
        raise BackendMetadataError("unsupported_vegetation", "only prescribed none/raster holders are described")


def inspect_dense_metadata(bed: Any, *, vegetation: Any = None, threshold_modifier_state: Any = None,
                           soil_water_m: Any = None, terrain: Any = None, backend: str = "numpy",
                           representation: str = "dense", provenance: BackendProvenance | None = None
                           ) -> DenseBackendMetadata:
    """Describe supported holders without mutation, copying, conversion or value scans.

    Required: exact native SYRUP BedState with all six Dense fields. Optional:
    native prescribed none/raster VegetationState, ThresholdModifierState, one
    NumPy soil-water array, and native host TerrainReference. None omits only an
    optional root. RoutingGraph, event/cohort/config holders, objects vegetation,
    compact/windowed storage and CuPy are deliberately unsupported. Native bed
    correctness, units, inventory/shape/dtype contracts and source binding are
    NOT established here. No caller or transport is wired to this function.
    """
    if type(backend) is not str or backend != "numpy":
        raise BackendMetadataError("unsupported_backend", "only NumPy metadata is described; GPU is unqualified")
    if type(representation) is not str or representation != "dense":
        raise BackendMetadataError("unsupported_representation", "only explicit Dense metadata is described")
    provenance = BackendProvenance() if provenance is None else provenance
    if type(provenance) is not BackendProvenance:
        raise BackendMetadataError("invalid_provenance", "expected BackendProvenance labels")
    # Revalidate even forcibly altered frozen inputs, then retain owned scalar
    # labels. Serialization must never deepcopy a smuggled array/holder.
    provenance = BackendProvenance(**BackendProvenance.as_record(provenance))
    n = _native()
    resident = [bed]
    _check_holder(n, bed, n.BedState, "bed")
    for name, value, kind in (("vegetation", vegetation, n.VegetationState),
                              ("threshold_modifier_state", threshold_modifier_state, n.ThresholdModifierState),
                              ("soil_water_m", soil_water_m, n.np.ndarray)):
        if value is not None:
            _check_holder(n, value, kind, name)
            resident.append(value)
    host = []
    if terrain is not None:
        _check_holder(n, terrain, n.TerrainReference, "terrain")
        host.append(terrain)
    resident_leaves, host_leaves = n.array_leaves(*resident), n.array_leaves(*host)
    # The native discovery/copy semantics remain authoritative: repeated array
    # refs are repeated leaves; containers have a cycle guard. Inventory below
    # uses the native non-memoizing placement traversal (including duplicates).
    if n.resolve_operand_namespace(*resident) is not n.np or n.resolve_operand_namespace(*host) is not n.np:
        raise BackendMetadataError("unsupported_backend", "all described fields must remain NumPy")
    if {id(a) for a in resident_leaves} & {id(a) for a in host_leaves}:
        raise BackendMetadataError("ambiguous_role", "one exact array object has both roles; views are not checked")
    rows = tuple((role, i, tuple(a.shape), a.dtype.str, int(a.nbytes))
                 for role, leaves in (("resident", resident_leaves), ("host_evidence", host_leaves))
                 for i, a in enumerate(leaves))
    inventory = n.array_leaf_inventory(tuple(resident + host))
    return DenseBackendMetadata(rows, tuple(inventory.items()), provenance)
