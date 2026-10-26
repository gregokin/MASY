"""Explicit CPU water-input preparation on an existing, verified DenseBed.

This module supplies no scientific defaults and runs no event.  Native SYRUP
constructs the parameters and terrain evidence; the existing runner owns
transport, routing reuse, completion and publication.  Import is model-free.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any

from masy.dense_bed import DenseBed, DenseBedError
from masy.dense_runner import DenseRunnerControl, DenseWaterInputs, DenseRunnerError, _canonical_input

__all__ = ["CaseWaterError", "CaseWetClass", "CaseWaterSpec", "CaseWaterCopyLimits",
           "PreparedCaseWater", "prepare_case_water"]


class CaseWaterError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code, self.detail = code, detail
        super().__init__(f"{code}: {detail}")


@dataclasses.dataclass(frozen=True)
class CaseWetClass:
    """Authored coefficients keyed by an existing MAPLE class ID."""
    class_id: str
    raindrop_a: float
    raindrop_b: float
    raindrop_c: float
    raindrop_max_depth_mm: float
    raindrop_depth_attenuation_per_cm: float


@dataclasses.dataclass(frozen=True)
class CaseWaterSpec:
    """Every value is explicit. Arrays are caller-owned NumPy C-contiguous
    grids; preparation copies them. This is neither a calibration certificate
    nor a file reader. `source_reference` identifies caller provenance.
    """
    profile_id: str
    source_reference: str
    elevation_full_m: Any
    export_receiver_full: Any
    active_mask: Any
    friction_factor: Any
    allow_masked_nodata: bool
    nodata_value: float | None
    allow_pit_storage: bool
    allow_flat_storage: bool
    soil_model: str
    ksat_m_per_s: Any
    suction_m: Any
    drainage_parameter: Any
    theta_sat: Any
    soil_thickness_m: Any
    pavement_cover_fraction: Any | None
    initial_theta: Any | None
    retained_soil_water_m: Any | None
    rainfall_multiplier: Any
    wet_classes: tuple[CaseWetClass, ...]
    active_layer_sensitivity_mm: float
    ke_model: str
    kinematic_viscosity_m2_s: float
    reference_interval_s: float
    recession_factor_per_reference_s: float
    ke_vegetation_form: str
    raindrop_composition_scaling: str
    distance_convention: str
    dstar_convention: str
    bagnold_depth_units: str


@dataclasses.dataclass(frozen=True)
class CaseWaterCopyLimits:
    """Finite preparation limits, independent of Native/run process caps.

    Known array bytes cover the water inputs and the Source-visible native
    graph/parameter array envelope, including current/candidate/error overlap.
    Serialized metadata bytes are a separate ceiling, not a Python heap bound.
    Existing bed/process limits and actual admission remain mandatory.
    """
    max_cells: int
    max_classes: int
    max_known_host_bytes: int
    max_metadata_bytes: int

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise CaseWaterError("invalid_copy_limit", f"{field.name} must be a positive plain int")


_ARRAY_FIELDS = ("elevation_full_m", "export_receiver_full", "active_mask", "friction_factor",
                 "ksat_m_per_s", "suction_m", "drainage_parameter", "theta_sat", "soil_thickness_m",
                 "pavement_cover_fraction", "initial_theta", "retained_soil_water_m", "rainfall_multiplier")
_BOOLEAN_FIELDS = ("export_receiver_full", "active_mask")
_SCALAR_FIELDS = ("active_layer_sensitivity_mm", "kinematic_viscosity_m2_s", "reference_interval_s",
                  "recession_factor_per_reference_s")
_TEXT_FIELDS = ("profile_id", "source_reference", "soil_model", "ke_model", "ke_vegetation_form",
                "raindrop_composition_scaling", "distance_convention", "dstar_convention", "bagnold_depth_units")


def _json_bytes(record, maximum):
    """Bound the encoded identity while writing; no model arrays serialize."""
    out = bytearray()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    for text in encoder.iterencode(record):
        chunk = text.encode("utf-8")
        if len(out) + len(chunk) > maximum:
            raise CaseWaterError("metadata_limit", f"identity exceeds {maximum} serialized bytes")
        out.extend(chunk)
    return bytes(out)


def _identity(np, value, maximum):
    try:
        canonical = _canonical_input(np, value, "case_water")
    except (DenseRunnerError, DenseBedError) as exc:
        raise CaseWaterError("controlling_input_unhashable", exc.detail) from exc
    return _json_bytes(canonical, maximum)


def _verify_case(bed):
    # The existing dynamic profile exposes defensive snapshots instead of
    # the stable prescribed cover owner this narrow factory binds. Refuse
    # explicitly; never select a second vegetation authority or stale value.
    if hasattr(bed, "_dynamic"):
        raise CaseWaterError("unsupported_dynamic_vegetation",
                             "dynamic vegetation needs a separately qualified water preparation")
    try:
        bed.verify_case_inputs()
    except DenseBedError as exc:
        raise CaseWaterError("case_input_changed", exc.detail) from exc


def _array(np, value, name, shape):
    dtype = np.bool_ if name in _BOOLEAN_FIELDS else np.float64
    if (type(value) is not np.ndarray or value.dtype != dtype or value.shape != shape
            or not value.flags.c_contiguous):
        raise CaseWaterError("invalid_array", f"{name} must be a C-contiguous host {dtype} array of shape {shape}")
    return value


def _copy_plan(spec, ny, nx, nc):
    """Known array-role envelope, not a whole-process or allocator guarantee.

    Native routing.py306-556 uses scalar/D4 grids, sorted indices and at most
    four donors per cell. The deliberately conservative slots below include
    all retained, expression, sort and old/new arrays in those named roles.
    No voxel-depth term exists: the shared bed is referenced, never copied.
    """
    cells, ring = ny * nx, (ny + 2) * (nx + 2)
    caller = sum(getattr(spec, name).nbytes for name in _ARRAY_FIELDS if getattr(spec, name) is not None)
    graph = cells * (32 * 8 + 32 * 8 + 16 * 4 + 32 + 8) + ring * (2 * 8 + 2)
    native_parameters = cells * (16 * 8 + 4) + nc * (32 * 8)
    # The original shared mobile inventory is only read. Boolean indexing
    # may own a full cell/class copy plus its comparison and inverted mask.
    mobile_mask_check = cells * nc * (8 + 1) + cells
    return {"caller_water_array_bytes": int(caller), "owned_input_copy_bytes": int(caller),
            "native_graph_array_envelope_one": graph,
            "graph_current_candidate_error_bytes": 3 * graph,
            "native_input_current_candidate_error_bytes": 3 * native_parameters,
            # Ordinary first-rain construction plus native graph/source validation
            # stays in the runner. Reserve this separately; no bypass or new call.
            "first_rain_graph_creation_validation_allowance_bytes": 3 * graph,
            "inactive_mobile_check_bytes": mobile_mask_check,
            "shared_bed_copied_bytes": 0,
            "known_host_bytes": int(2 * caller + 6 * graph + 3 * native_parameters + mobile_mask_check)}


@dataclasses.dataclass(frozen=True)
class PreparedCaseWater:
    water: DenseWaterInputs
    control: DenseRunnerControl
    profile_id: str
    identity_sha256: str
    _bed: Any = dataclasses.field(repr=False, compare=False)
    _spec: Any = dataclasses.field(repr=False, compare=False)
    _limits: Any = dataclasses.field(repr=False, compare=False)
    _spec_bytes: bytes = dataclasses.field(repr=False, compare=False)
    _water_bytes: bytes = dataclasses.field(repr=False, compare=False)
    _control_bytes: bytes = dataclasses.field(repr=False, compare=False)
    _record_bytes: bytes = dataclasses.field(repr=False, compare=False)

    def identity_record(self):
        if hashlib.sha256(self._record_bytes).hexdigest() != self.identity_sha256:
            raise CaseWaterError("prepared_identity_changed", "prepared identity bytes changed")
        return json.loads(self._record_bytes)

    def verify(self, bed=None):
        """Verify controlling inputs, allowing ordinary accepted bed evolution.

        This does not replace runner event/current-state verification or admit a
        new vegetation version. Changed canonical cover requires explicit setup.
        """
        import numpy as np

        held = self._bed if bed is None else bed
        if type(held) is not DenseBed or held is not self._bed:
            raise CaseWaterError("bed_identity_mismatch", "prepared water belongs to a different DenseBed")
        _verify_case(held)
        record = self.identity_record()
        if held.case_sha256 != record["case_sha256"] or self.profile_id != record["profile_id"]:
            raise CaseWaterError("case_input_changed", "prepared case/profile identity changed")
        if (type(self._limits) is not CaseWaterCopyLimits
                or dataclasses.asdict(self._limits) != record["copy_limits"]):
            raise CaseWaterError("copy_limit_changed", "prepared copy/metadata limits changed")
        maximum = self._limits.max_metadata_bytes
        if (_identity(np, self._spec, maximum) != self._spec_bytes
                or _identity(np, self.water, maximum) != self._water_bytes
                or _identity(np, self.control, maximum) != self._control_bytes):
            raise CaseWaterError("controlling_input_changed", "caller spec, native water inputs or control changed")
        if self.water.vegetation_cover_fraction is not held.vegetation.raster.cover_fraction:
            raise CaseWaterError("vegetation_changed", "prepared cover is no longer the actual shared canonical cover")
        return self


def prepare_case_water(bed: DenseBed, spec: CaseWaterSpec, control: DenseRunnerControl,
                       *, limits: CaseWaterCopyLimits) -> PreparedCaseWater:
    """Prepare real Native CPU inputs; never reload, reset or transport a bed."""
    if type(bed) is not DenseBed or type(spec) is not CaseWaterSpec or type(limits) is not CaseWaterCopyLimits:
        raise CaseWaterError("invalid_input_type", "requires actual DenseBed, CaseWaterSpec and CaseWaterCopyLimits")
    if type(control) is not DenseRunnerControl:
        raise CaseWaterError("invalid_control", "requires existing DenseRunnerControl")
    import numpy as np
    from maple_syrup.complete_event import CompletionPolicy
    from maple_syrup.infiltration import column_parameters, initial_soil_water_m
    from maple_syrup.rainfall import rainfall_field
    from maple_syrup.routing import build_routing_graph
    from maple_syrup.sediment_bed import terrain_reference
    from maple_syrup.sediment_event import SedimentEventControl, SedimentEventError
    from maple_syrup.sediment_physics import sediment_physics_parameters
    from maple_syrup.storm import StormControl, StormError

    _verify_case(bed)
    state, context, native_vegetation, clock = bed.state, bed.context, bed.vegetation, bed.time_s
    case_sha256 = bed.case_sha256
    g = context.geometry
    ny, nx = g.ny, g.nx
    classes = context.grain_classes.classes
    nc = len(classes)
    if ny * nx > limits.max_cells or nc > limits.max_classes:
        raise CaseWaterError("shape_limit", "selected case exceeds the declared cell/class preparation limits")
    if g.dx_m != g.dy_m or context.depth_update_rule != "constant_depth":
        raise CaseWaterError("unsupported_geometry_or_depth", "requires square cells and native constant_depth")
    if (type(control.sediment_control) is not SedimentEventControl
            or type(control.completion_policy) is not CompletionPolicy
            or type(control.sediment_control.storm) is not StormControl
            or control.sediment_control.storm.implementation != "array"
            or control.sediment_control.transport_scheme != "upwind"):
        raise CaseWaterError("invalid_control", "requires genuine Native array/upwind controls accepted by DenseRunner")
    try:
        control.sediment_control.validated()
        control.completion_policy.validated()
    except (ValueError, SedimentEventError, StormError) as exc:
        raise CaseWaterError("invalid_control", str(exc)) from exc
    if (control.sediment_control.commit is not True
            or control.sediment_control.record_exchange_history is not False
            or control.sediment_control.deep_publication != "step"):
        raise CaseWaterError("invalid_control", "requires DenseRunner's existing commit/history/publication controls")
    if (type(control.report_every_s) is not float or not math.isfinite(control.report_every_s)
            or control.report_every_s <= 0):
        raise CaseWaterError("invalid_control", "report_every_s must be a finite positive plain float")
    for name in _TEXT_FIELDS:
        value = getattr(spec, name)
        if type(value) is not str or not value or len(value) > limits.max_metadata_bytes:
            raise CaseWaterError("invalid_spec", f"{name} must be explicit nonempty bounded text")
    for name in ("allow_masked_nodata", "allow_pit_storage", "allow_flat_storage"):
        if type(getattr(spec, name)) is not bool:
            raise CaseWaterError("invalid_spec", f"{name} must be an explicit bool")
    for name in _SCALAR_FIELDS:
        if type(getattr(spec, name)) is not float or not math.isfinite(getattr(spec, name)):
            raise CaseWaterError("invalid_spec", f"{name} must be an explicit finite plain float")
    if spec.nodata_value is not None and (type(spec.nodata_value) is not float or not math.isfinite(spec.nodata_value)):
        raise CaseWaterError("invalid_spec", "nodata_value must be None or an explicit finite float")
    if (spec.initial_theta is None) == (spec.retained_soil_water_m is None):
        raise CaseWaterError("ambiguous_initial_soil", "supply exactly one initial_theta or retained_soil_water_m")
    for name in _ARRAY_FIELDS:
        value = getattr(spec, name)
        if value is None:
            if name not in ("pavement_cover_fraction", "initial_theta", "retained_soil_water_m"):
                raise CaseWaterError("invalid_array", f"{name} is required")
            continue
        shape = (ny + 2, nx + 2) if name in ("elevation_full_m", "export_receiver_full") else (ny, nx)
        _array(np, value, name, shape)
    if type(spec.wet_classes) is not tuple or len(spec.wet_classes) != nc:
        raise CaseWaterError("class_mismatch", "wet_classes must explicitly name exactly the bed's classes")
    rows = {}
    for row in spec.wet_classes:
        if type(row) is not CaseWetClass or type(row.class_id) is not str or row.class_id in rows:
            raise CaseWaterError("class_mismatch", "class coefficients must have unique explicit class IDs")
        for field in dataclasses.fields(row):
            if field.name != "class_id":
                value = getattr(row, field.name)
                if type(value) is not float or not math.isfinite(value):
                    raise CaseWaterError("invalid_spec", f"{row.class_id}.{field.name} must be a finite plain float")
        rows[row.class_id] = row
    ids = tuple(c.class_id for c in classes)
    if set(rows) != set(ids):
        raise CaseWaterError("class_mismatch", "missing or extra wet coefficient class IDs")
    densities = tuple(float(c.particle_density_kg_m3) for c in classes)
    density = densities[0]
    if any(d != density for d in densities) or (density / 1000.0) * 1000.0 != density:
        raise CaseWaterError("density_mismatch", "requires one common density with an exact native unit round-trip")
    cover = native_vegetation.raster.cover_fraction
    _array(np, cover, "vegetation_cover_fraction", (ny, nx))
    _array(np, state.water.mobile_mass_by_cell_class_kg, "water_mobile", (ny, nx, nc))
    plan = _copy_plan(spec, ny, nx, nc)
    if plan["known_host_bytes"] > limits.max_known_host_bytes:
        raise CaseWaterError("known_copy_limit", f"known water array envelope {plan['known_host_bytes']} exceeds {limits.max_known_host_bytes}")
    spec_bytes = _identity(np, spec, limits.max_metadata_bytes)
    control_bytes = _identity(np, control, limits.max_metadata_bytes)
    # All caller grids are copied before Native code can freeze/retain them.
    owned = {name: np.array(getattr(spec, name), copy=True, order="C")
             for name in _ARRAY_FIELDS if getattr(spec, name) is not None}
    try:
        active = owned["active_mask"]
        if bool(np.any(state.water.mobile_mass_by_cell_class_kg[~active] != 0.0)):
            raise CaseWaterError("inactive_mobile_mass", "native water mobile inventory exists on a water-inactive cell")
        graph = build_routing_graph(owned["elevation_full_m"], owned["export_receiver_full"],
            owned["friction_factor"], g.dx_m, dy_m=g.dy_m, active_mask=active, xp=np,
            nodata_value=spec.nodata_value, allow_masked_nodata=spec.allow_masked_nodata,
            allow_pit_storage=spec.allow_pit_storage, allow_flat_storage=spec.allow_flat_storage)
        terrain = terrain_reference(owned["elevation_full_m"], owned["export_receiver_full"],
                                    state, graph, nodata_value=spec.nodata_value)
        column = column_parameters(model=spec.soil_model,
            **{name: owned[name] for name in ("ksat_m_per_s", "suction_m", "drainage_parameter", "theta_sat", "soil_thickness_m")},
            pavement_cover_fraction=owned.get("pavement_cover_fraction"), active_mask=active)
        field = rainfall_field(ny, nx, scale=owned["rainfall_multiplier"], active_mask=active, xp=np)
        soil = (initial_soil_water_m(column, owned["initial_theta"]) if spec.initial_theta is not None
                else owned["retained_soil_water_m"])
        if (not bool(np.isfinite(soil).all()) or bool((soil < 0).any()) or bool(np.signbit(soil).any())
                or bool((soil > column.storage_max_m).any())):
            raise CaseWaterError("invalid_initial_soil", "stored soil water must be finite in [0,native storage_max_m]")
        soil.flags.writeable = False
        vectors = {name: [getattr(rows[class_id], name) for class_id in ids]
                   for name in ("raindrop_a", "raindrop_b", "raindrop_c", "raindrop_max_depth_mm", "raindrop_depth_attenuation_per_cm")}
        diameters = np.array([c.diameter_m for c in classes], dtype=np.float64)
        sediment = sediment_physics_parameters(diameter_m=diameters, **vectors,
            particle_density_g_cm3=density / 1000.0, active_layer_sensitivity_mm=spec.active_layer_sensitivity_mm,
            ke_model=spec.ke_model, kinematic_viscosity_m2_s=spec.kinematic_viscosity_m2_s,
            reference_interval_s=spec.reference_interval_s, recession_factor_per_reference_s=spec.recession_factor_per_reference_s,
            ke_vegetation_form=spec.ke_vegetation_form, raindrop_composition_scaling=spec.raindrop_composition_scaling,
            distance_convention=spec.distance_convention, dstar_convention=spec.dstar_convention,
            bagnold_depth_units=spec.bagnold_depth_units, xp=np)
        if (sediment.n_classes != nc or not np.array_equal(sediment.diameter_m, diameters)
                or sediment.particle_density_kg_m3 != density):
            raise CaseWaterError("class_mismatch", "actual native parameters differ from the bed's exact metadata")
        if (not np.array_equal(graph.active, active) or not np.array_equal(column.active_mask, active)
                or not np.array_equal(terrain.active_mask, active) or bool((field.multiplier[~active] != 0).any())):
            raise CaseWaterError("mask_mismatch", "actual native water masks differ")
        water = DenseWaterInputs(terrain, column, field, cover, sediment, soil)
        water_bytes = _identity(np, water, limits.max_metadata_bytes)
        record = {"schema":"masy.case_water_inputs.v1", "profile_id":spec.profile_id,
            "case_sha256":case_sha256, "source_reference":spec.source_reference,
            "spec_sha256":hashlib.sha256(spec_bytes).hexdigest(),
            "water_sha256":hashlib.sha256(water_bytes).hexdigest(),
            "control_sha256":hashlib.sha256(control_bytes).hexdigest(),
            "copy_limits":dataclasses.asdict(limits), "known_array_plan":plan,
            "native_graph_calls_in_preparation":2,
            "first_rain_routing":"ordinary runner construction/validation/commit; native reroute decisions unchanged",
            "backend":"numpy", "scientific_default_fallback":False,
            "transport_or_completion_executed":False, "whole_process_fit_claim":False}
        record_bytes = _json_bytes(record, limits.max_metadata_bytes)
        _verify_case(bed)
        if (bed.state is not state or bed.context is not context or bed.vegetation is not native_vegetation
                or bed.time_s != clock or bed.case_sha256 != case_sha256):
            raise CaseWaterError("bed_changed_during_preparation", "current bed/case/vegetation/clock changed")
        if (_identity(np, spec, limits.max_metadata_bytes) != spec_bytes
                or _identity(np, control, limits.max_metadata_bytes) != control_bytes):
            raise CaseWaterError("controlling_input_changed", "caller spec/control changed during preparation")
        prepared = PreparedCaseWater(water, control, spec.profile_id, hashlib.sha256(record_bytes).hexdigest(),
            bed, spec, limits, spec_bytes, water_bytes, control_bytes, record_bytes)
        return prepared.verify()
    except CaseWaterError:
        raise
    except Exception as exc:
        raise CaseWaterError("native_input_refused", f"{type(exc).__name__}: {exc}") from exc
