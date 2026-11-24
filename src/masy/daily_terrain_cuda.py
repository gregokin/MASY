"""Source-bound Dense CUDA terrain handoff; all physical operations are upstream.

Separate from the historical baseline profile. The fixed base, physical footprint,
frozen regional plane and native resource census follow MAPLE's actual APIs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from masy import daily_wind as dw
from masy import daily_wind_cuda as gw
from masy import daily_wind_handoff as mh
from masy import daily_triplet_cuda as gt
from masy.daily_schedule import DailyPlan
from masy.gpu_contract import GpuPolicy, footprint

PROFILE = "masy.daily_triplet_wind.terrain_cuda.v1"
MAPLE_COMMIT = "c4b9343706240d3f98e9abb9c7274958a45438cc"
MAPLE_PACKAGE_SHA256 = "f4e4dc694dff1bbf126ddbb69a8cd0bc089d8006ebb5f4ed2ce1ac30e8e5b8bc"
# Filled only after independent review of the exact owning source candidates.
SYRUP_PACKAGE_SHA256 = "92f3658276b8f432c12006e42eb73235e19606cbf1feea5455517fa90363fa15"
LEAF_PACKAGE_SHA256 = "069f711aade205c4db3a2b044020361dfb289613bad9434e97140037be1daf25"


def _sources():
    root = Path(__file__).resolve().parent
    return {**gw._sources(), **{name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ("daily_terrain_cuda.py", "terrain_boundary.py")}}


def _source(source, admission):
    from maple_syrup import native_case_profile as ncp
    if type(source) is not ncp.NativeSourceProfile or type(admission) is not ncp.NativeCaseAdmission:
        raise dw.DailyWindError("native_source", "genuine terrain source and case admission required")
    source.recheck()
    admission.recheck()
    if (SYRUP_PACKAGE_SHA256 == "0" * 64 or not source.device_runtime_capability
            or source.profile != ncp.TERRAIN_CUDA_PROFILE_NAME
            or admission.profile_name != source.profile or source.maple_commit != MAPLE_COMMIT
            or source.package_digest != MAPLE_PACKAGE_SHA256
            or source.syrup_package_digest != SYRUP_PACKAGE_SHA256
            or admission.bundle_id != source.bundle_id or admission.package_digest != source.package_digest
            or admission.syrup_package_digest != source.syrup_package_digest):
        raise dw.DailyWindError("native_source", "the exact separately qualified terrain CUDA source pair is required")
    return {"profile": source.profile, "maple_commit": source.maple_commit,
        "maple_package": source.package_digest, "syrup_package": source.syrup_package_digest,
        "bundle_id": source.bundle_id, "case_sha256": admission.case_identity_sha256,
        "latitude": admission.latitude_binding()}


def _terrain_record(context, compiled):
    import numpy as np
    from maple.core.types.domain import validate_domain_state, domain_identity
    from maple.core.types.fixed_base import validate_fixed_base_state, fixed_base_identity
    g = context.geometry
    result = {}
    for name, validator, identity in (("domain", validate_domain_state, domain_identity),
                                      ("fixed_base", validate_fixed_base_state, fixed_base_identity)):
        owner = getattr(context, name, None)
        original = getattr(compiled, name, None)
        if (owner is None) != (original is None):
            raise dw.DailyWindError("terrain", f"context {name} does not match the admitted landscape")
        if owner is not None:
            validator(owner, g.ny, g.nx, verify_content=True)
            if identity(owner) != identity(original):
                raise dw.DailyWindError("terrain", f"context {name} identity differs from admission")
            result[name] = identity(owner)
    if compiled.regional_slope.enabled:
        result['regional_slope_sha256'] = dw._sha(dw._canonical(np,
            (compiled.regional_slope, compiled.regional_slope_result), 'frozen_regional_plane'))
    return result


class TerrainCudaWindExecutor(gw.CudaWindExecutor):
    profile = PROFILE
    _source_identity = staticmethod(_source)
    _wrapper_identity = staticmethod(_sources)

    def _recheck(self):
        super()._recheck()
        cells = self._context.geometry.ny * self._context.geometry.nx
        self.policy.admit(1024 * cells, 128 * cells, 'cold terrain identity checks')
        if _terrain_record(self._context, self._admission.compiled) != self.identity()["terrain"]:
            raise dw.DailyWindError("terrain", "immutable terrain identity changed")

    def _scheduler_config(self, n, record):
        from maple.core.backend import to_device_tree
        from maple.aeolian.regional_slope.runtime_admission import (
            plan_regional_slope_runtime_admission, build_regional_slope_runtime_context,
        )
        compiled = self._admission.compiled
        cfg = compiled.config
        runtime = None
        if compiled.regional_slope.enabled:
            plan = plan_regional_slope_runtime_admission(
                compiled.regional_slope, compiled.regional_slope_result, cfg, record,
                context=PROFILE, execution_backend="cupy", device=True,
                footprint=self._context.domain is not None,
                available_memory_bytes_fn=self.policy.host_available_bytes_fn)
            runtime = build_regional_slope_runtime_context(plan, compiled.regional_slope_result)
            h, d = footprint(runtime)
            self.policy.admit(h + (4 << 20), d + h, "native frozen regional wind context")
            runtime = to_device_tree(runtime, self.policy.gate())
        effective = dw._effective_wind_config(cfg, self._context)
        result = n.build_scheduler_physics_config(effective,
            initial_available_fraction=self._context.initial_available_fraction, wind_record=record,
            regional_slope_runtime=runtime, fixed_base=self._context.fixed_base,
            domain=self._context.domain, external_water_carry_enabled=True)
        self._terrain_execution = {'regional_wind': None if runtime is None else runtime.identity_payload(),
            'surface_edge_policy': result.geometry.surface_edge_policy,
            'wind_boundary_x': result.geometry.boundary_x.kind.value,
            'wind_boundary_y': result.geometry.boundary_y.kind.value,
            'moisture_feedback': result.fluid_threshold_spec.modifiers.moisture_enabled}
        self._validate_terrain(result, None)
        return result

    def _additional_work_bytes(self, config):
        from maple.aeolian.scheduler.domain_runtime import domain_runtime_owner_census, admit_domain_run_memory
        from maple.aeolian.regional_slope.runtime_admission import prospective_regional_slope_runtime_peak
        rows = {}
        if config.domain is not None:
            census = domain_runtime_owner_census(config, device=True)
            admit_domain_run_memory(census, self.policy.gate(), context=PROFILE,
                available_memory_bytes_fn=self.policy.host_available_bytes_fn)
            rows["footprint"] = int(census["total_bytes"])
        if config.regional_slope_runtime is not None:
            census = prospective_regional_slope_runtime_peak(config.regional_slope_runtime,
                config.geometry, config.topographic_wind_spec, config.flow_separation_spec,
                device=True, available_memory_bytes_fn=self.policy.device_available_bytes_fn)
            rows["regional_wind"] = int(census["peak_bytes"])
        self._last_terrain_census = rows
        return sum(rows.values())

    def _validate_terrain(self, config, physical):
        from maple.aeolian.scheduler.domain_runtime import require_domain_scheduler_profile, validate_scheduler_domain_boundary
        from maple.core.backend import to_bool
        require_domain_scheduler_profile(config, path=PROFILE)
        if physical is None:
            return
        validate_scheduler_domain_boundary(config, physical, path=PROFILE)
        if config.domain is not None:
            cp = self.policy.gate()
            invalid = ~config.domain.valid_mask
            if (to_bool(cp.any(cp.where(invalid, physical.water.depth_m, 0.0) != 0))
                    or to_bool(cp.any(cp.where(invalid[..., None], physical.water.mobile_mass_by_cell_class_kg, 0.0) != 0))):
                raise dw.DailyWindError("mask", "water appeared outside the physical footprint")

    def _publish_boundary(self, n, config, bed, final, end, *, work, external_reserve_bytes):
        """Terrain boundary: the authored case retains the accounting reference through the
        native finalizer, so a present baseline is published through MAPLE's public run-ending
        force commit at the unchanged end clock (see ``masy.terrain_boundary``). An absent
        baseline passes through exactly as in the base profile."""
        from masy import terrain_boundary as tb
        return tb.publish_terrain_boundary(self, n, config, bed, final, end,
                                           work=work, external_reserve_bytes=external_reserve_bytes)


def prepare_terrain_wind_cuda(source, admission, context, *, limits, policy):
    import numpy as np
    from maple.core.backend import to_device_tree
    if type(limits) is not dw.DailyWindLimits or type(policy) is not GpuPolicy:
        raise dw.DailyWindError("limits", "explicit native wind limits and two-pool CUDA policy required")
    cp = policy.gate()
    facts = _source(source, admission)
    n = dw._native()
    if type(context) is not n.BedContext:
        raise dw.DailyWindError("context", "a genuine native terrain BedContext is required")
    cfg = admission.compiled.config
    for mine, theirs in ((context.geometry, cfg.geometry), (context.grain_classes, cfg.grain_classes),
                          (context.avalanching_spec, cfg.avalanching),
                          (context.topographic_commit_spec, cfg.topographic_commit)):
        if not dw._same(mine, theirs):
            raise dw.DailyWindError("context", "native context differs from admitted case")
    g, fraction = context.geometry, context.initial_available_fraction
    if (cfg.execution.backend not in ("cupy", "cuda") or cfg.water_coupling.enabled
            or cfg.prescribed_external_shear.enabled or cfg.aeolian_process_mode.mode != dw.SUPPORTED_WIND_MODE
            or context.depth_update_rule != "constant_depth" or context.mass_resolution_kg != cfg.numerics.mass_resolution_kg
            or g.nx * g.ny > limits.max_cells or type(fraction) is not cp.ndarray
            or fraction.dtype != cp.float64 or fraction.shape != (g.ny, g.nx, len(context.grain_classes.classes))
            or not bool(cp.isfinite(fraction).all())):
        raise dw.DailyWindError("unsupported_profile", "case/backend/context exceeds the explicit terrain wind profile")
    terrain = _terrain_record(context, admission.compiled)
    h, d = footprint(admission.compiled.threshold_modifier_state)
    policy.admit(h + (4 << 20), d + h + fraction.nbytes, "terrain wind static placement")
    threshold = to_device_tree(admission.compiled.threshold_modifier_state, cp)
    identity = {"profile": PROFILE, "source": facts, "wrapper_sources": _sources(), "limits": limits.as_record(),
        "policy": policy.record(), "config": dw._sha(dw._canonical(np, cfg, "config")),
        "prescribed_sha256": dw._prescribed_digest(admission.compiled), "terrain": terrain}
    return TerrainCudaWindExecutor(source, admission, context, limits, policy, threshold, fraction.copy(), identity)


def prepare_terrain_triplet_wind_cuda(plan, source, admission, context, *, wind_limits, handoff_limits,
                                     bridge_limits, policy, run_inputs_sha256, terrain_source, batch_cells=64):
    from maple_syrup import floating_wind_handoff_cuda as wh
    if (type(plan) is not DailyPlan or type(policy) is not GpuPolicy
            or type(bridge_limits) is not mh.WindBridgeLimits or type(handoff_limits) is not wh.WindHandoffLimits
            or type(batch_cells) is not int or not 1 <= batch_cells <= handoff_limits.max_batch_cells
            or handoff_limits.max_cohorts < admission.compiled.config.particle_transport.max_active_cohorts):
        raise mh.WindHandoffError("limits", "matching native terrain plan, policies, batch and cohort caps required")
    wind = prepare_terrain_wind_cuda(source, admission, context, limits=wind_limits, policy=policy)
    result = gt.CudaTripletWind(plan, wind, context, handoff_limits, bridge_limits,
        policy.device_available_bytes_fn, policy.device_reserve_bytes, batch_cells)
    result.policy, result._inputs_sha256 = policy, mh._hex(run_inputs_sha256, "run_inputs_sha256")
    result._terrain_source = terrain_source
    result._host_extra = 0
    return result


class DailyTerrainCudaTripletSession(gt.DailyCudaTripletSession):
    """New exact source frontier, retaining the same native daily transaction order."""
    profile = 'masy.daily_triplet.terrain_cuda.v1'

    def __init__(self, *args, terrain_builder=None, minimum_soil_depth_m=None, **kwargs):
        if terrain_builder is not None and not callable(terrain_builder):
            raise gt.TripletError('terrain', 'an explicit native terrain-input builder is required')
        self._terrain_builder = terrain_builder
        if minimum_soil_depth_m is not None:
            import math
            if type(minimum_soil_depth_m) not in (int, float) or not math.isfinite(minimum_soil_depth_m) or minimum_soil_depth_m <= 0:
                raise gt.TripletError('terrain', 'finite positive minimum soil depth required')
        self._minimum_soil_depth_m = minimum_soil_depth_m
        self._weather_surface = None
        super().__init__(*args, **kwargs)

    def _observe_surface(self, surface):
        if self._terrain_builder is not None:
            surface.flags.writeable = False
            self._weather_surface = surface

    def _held(self, extra=0):
        h, d = super()._held(extra)
        return h + (0 if self._weather_surface is None else self._weather_surface.nbytes), d

    def _prepare_weather_terrain(self):
        if self._minimum_soil_depth_m is not None:
            import numpy as np
            host_depth = self.lineage.head.root.host_depth
            if type(host_depth) is not np.ndarray or host_depth.shape != (self._mask.size,):
                raise gt.TripletError('terrain', 'native sealed host depth required for the soil-depth guard')
            if np.any(host_depth[self._mask.reshape(-1)] < self._minimum_soil_depth_m):
                raise gt.TripletError('terrain', 'soil became shallower than the admitted equal-layer diffusion mapping')
        if self._terrain_builder is None:
            return
        from maple_syrup import weather_pet as wp
        n = self._mask.size
        h, d = self._held()
        self.policy.admit(h + 256*n, d + 64*n, 'current-surface daily terrain factor')
        factor = self._terrain_builder(self._weather_surface)
        if (type(factor) is not wp.TerrainFactor or factor.backend != 'cuda'
                or factor.shape != (n,) or factor.latitude_deg != self._terrain.latitude_deg
                or not factor.verify_identity()):
            raise gt.TripletError('terrain', 'native current-surface terrain factor is inconsistent')
        self._terrain = factor

    def _prepare_after_storm_weather(self):
        if self._terrain_builder is None:
            return False
        from maple.core.backend import to_host
        from maple.core.types.topographic_commit import committed_elevation_m
        h, d = self._held()
        self.policy.admit(h + 8*self._mask.size, d + 8*self._mask.size,
            'current post-storm world surface for daily weather')
        self._observe_surface(to_host(committed_elevation_m(self.lineage.head.bed.committed_topography)))
        self._prepare_weather_terrain()
        return True

    @staticmethod
    def _validate_leaf_source(expected_leaf_package_sha256, expected_external_helpers):
        if (LEAF_PACKAGE_SHA256 == "0" * 64
                or expected_leaf_package_sha256.removeprefix("sha256:") != LEAF_PACKAGE_SHA256
                or tuple(expected_external_helpers) != gt.LEAF_EXTERNAL_HELPERS):
            raise gt.TripletError("leaf_source", "the exact terrain LEAF source and helper are required")
