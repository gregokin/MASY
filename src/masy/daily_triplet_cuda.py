"""Explicit Dense CUDA daily composition using actual MAPLE/SYRUP/LEAF APIs.

Resident water and bed stay native. Cold vegetation mapping uses the existing
host-facing contracts with bounded surface/plant metadata, never soil/voxel
mirrors. No transport, soil, uptake or growth law is copied into this module.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import hashlib
import json
import math
from fractions import Fraction
from types import SimpleNamespace

from masy import daily_wind as dw, daily_wind_handoff as mh
from masy import daily_wind_cuda as gw
from masy.daily_schedule import DailyPlan
from masy.daily_triplet import TripletError, TripletLimits, AcceptedTripletDay
from masy.daily_triplet_wind import WindVegetationPolicy
from masy.gpu_contract import GpuPolicy, footprint, device_leaves, host_vector

PROFILE = 'masy.daily_triplet.dense_cuda.v1'
LEAF_PACKAGE_SHA256 = 'b3e96519e28ad411d198658c6a8b84b5b418b74ec6d0653c3895fc6e7503a550'  # explicit frozen development/qualification identity
LEAF_EXTERNAL_HELPERS = ({'module': 'benchmarks.spatial_mapping', 'bytes': 47260,
    'sha256': '333921bb645d984fa8e7449619d03114262ee333c7fea61a22493a878fd9ccc5'},)


class CudaTripletWind(mh.WindHandoffExecutor):
    """The shared exactly-once scalar cursor, with genuine CUDA native hooks."""
    def _profile(self):
        return self._wind.profile

    def _native(self):
        from maple_syrup import floating_event_lifecycle as lc, floating_wind_handoff_cuda as wh
        q = dict(host_available_bytes_fn=self.policy.host_available_bytes_fn,
                 host_reserve_bytes=self.policy.host_reserve_bytes + self._host_extra,
                 memory_budget_bytes=self.policy.max_device_bytes,
                 host_memory_budget_bytes=self.policy.max_host_bytes)
        if getattr(self, '_terrain_source', None) is not None:
            q['terrain_source'] = self._terrain_source
        # Thin signature-preserving delegates. They return only genuine owning
        # records; the namespace is not a physical state/capability substitute.
        return lc, SimpleNamespace(start_wind_boundary=functools.partial(wh.start_wind_boundary, **q),
            stage_wind_handoff=functools.partial(wh.stage_wind_handoff, **q),
            adopt_wind_handoff=functools.partial(wh.adopt_wind_handoff, **q))

    def _head(self):
        lc, _wh = self._native()
        try:
            kind, head = lc.saveable_head(self.lineage, vegetation=True)
        except lc.LifecycleError as exc:
            raise mh.WindHandoffError('not_ready', str(exc)) from None
        if kind != 'boundary':
            raise mh.WindHandoffError('not_ready', 'CUDA wind requires a completed native boundary')
        return head

    def _policy(self):
        return {**super()._policy(), 'cuda': self.policy.record(), 'triplet_inputs_sha256': self._inputs_sha256}

    def _new_binding(self, lineage, boundary, context):
        from maple_syrup.sediment_bed import bed_inventory
        self._wind._recheck()
        bind = mh._Binding(lineage, context, self._digest(context.record()))
        inv = tuple(float(v).hex() for v in host_vector(bed_inventory(boundary.bed), size=len(self._context.grain_classes.classes)))
        zero = (Fraction(0),) * len(inv)
        self._pending_start = (bind, boundary, mh._Cursor(initial_inventory_hex=inv, boundary_export=zero, boundary_input=zero))

    def execute(self, occurrence, *, vegetation, external_reserve_bytes=0, external_host_bytes=0):
        if type(external_host_bytes) is not int or external_host_bytes < 0:
            raise mh.WindHandoffError('limits', 'explicit nonnegative foreign host bytes required')
        self._host_extra = external_host_bytes
        try:
            return super().execute(occurrence, vegetation=vegetation, external_reserve_bytes=external_reserve_bytes)
        finally:
            self._host_extra = 0

    def checkpoint_record(self, **kwargs):
        raise mh.WindHandoffError('checkpoint_unqualified', 'combined CUDA plant/wind/water cursor persistence is separate')

    def restore(self, *args, **kwargs):
        raise mh.WindHandoffError('checkpoint_unqualified', 'combined CUDA plant/wind/water restart is separate')


def prepare_triplet_wind_cuda(plan, source, admission, context, *, wind_limits, handoff_limits, bridge_limits,
                              policy, run_inputs_sha256, batch_cells=64):
    from maple_syrup import floating_wind_handoff_cuda as wh
    if type(plan) is not DailyPlan or type(policy) is not GpuPolicy or type(bridge_limits) is not mh.WindBridgeLimits:
        raise mh.WindHandoffError('plan', 'one parsed plan and explicit CUDA/bridge resource policy required')
    if type(handoff_limits) is not wh.WindHandoffLimits or type(batch_cells) is not int or batch_cells < 1:
        raise mh.WindHandoffError('limits', 'genuine native CUDA handoff limits and admitted batch required')
    if batch_cells > handoff_limits.max_batch_cells or handoff_limits.max_cohorts < admission.compiled.config.particle_transport.max_active_cohorts:
        raise mh.WindHandoffError('limits', 'handoff policy must admit batch and actual configured wind cohort pool')
    wind = gw.prepare_daily_wind_cuda(source, admission, context, limits=wind_limits, policy=policy)
    result = CudaTripletWind(plan, wind, context, handoff_limits, bridge_limits,
        policy.device_available_bytes_fn, policy.device_reserve_bytes, batch_cells)
    result.policy, result._inputs_sha256 = policy, mh._hex(run_inputs_sha256, 'run_inputs_sha256')
    result._host_extra = 0
    return result


class DailyCudaTripletSession:
    """One real water lineage and plant generation, fail-stop daily composition.

    No whole-day atomicity or persisted combined restart is asserted. Each native
    transaction remains visible as genuine accepted/active evidence on failure.
    """
    profile = PROFILE
    @staticmethod
    def _validate_leaf_source(expected_leaf_package_sha256, expected_external_helpers):
        if (LEAF_PACKAGE_SHA256 == '0' * 64 or expected_leaf_package_sha256.removeprefix('sha256:') != LEAF_PACKAGE_SHA256
                or tuple(expected_external_helpers) != LEAF_EXTERNAL_HELPERS):
            raise TripletError('leaf_source', 'the exact separately qualified CUDA LEAF package and helper are required')

    def __init__(self, plan, *, wind, owner, bed, initial_event_factory, plant_state, leaf_config, anchor,
                 leaf_source, terrain, partition, site_controls, active_mask, event_builder,
                 day_control, wind_vegetation_policy, limits, expected_leaf_package_sha256, expected_external_helpers,
                 policy, preferential=None, require_receiver_traits=True):
        import numpy as np
        from leaf import floating_lifecycle_bridge_cuda as fb, plant_session as ps
        from maple_syrup import weather_pet as wp, soil_evaporation as se, water_vegetation_binding as vb
        from maple_syrup.experimental_storm import ExperimentalControl
        if type(plan) is not DailyPlan or type(wind) is not CudaTripletWind or wind._plan is not plan:
            raise TripletError('plan', 'matching parsed plan and genuine CUDA triplet wind required')
        if type(policy) is not GpuPolicy or wind.policy is not policy or type(limits) is not TripletLimits:
            raise TripletError('limits', 'one shared two-pool CUDA policy and finite triplet limits required')
        if type(require_receiver_traits) is not bool or (not require_receiver_traits and preferential is not None):
            raise TripletError('hydrology', 'missing receiving traits may only be selected for a matrix-only run')
        self._require_receiver_traits = require_receiver_traits
        cp = policy.gate()
        self._validate_leaf_source(expected_leaf_package_sha256, expected_external_helpers)
        if type(leaf_config) is not ps.SessionConfig or leaf_config.backend != 'cuda':
            raise TripletError('backend', 'a genuine CUDA PlantSession configuration is required')
        if type(wind_vegetation_policy) is not WindVegetationPolicy or not callable(event_builder) or not callable(initial_event_factory):
            raise TripletError('factory', 'explicit native factories and wind vegetation policy required')
        if type(partition) is not se.EvaporationPartition or type(site_controls) is not vb.SiteControls or type(day_control) is not ExperimentalControl:
            raise TripletError('controls', 'genuine explicit native E/vegetation/day controls required')
        vb.verify_site_controls(site_controls)
        day_control.validated()
        if (type(anchor) is not dt.datetime or anchor.tzinfo is None or anchor.utcoffset() is None
                or anchor.time().replace(tzinfo=None) != dt.time() or anchor.date() != plan.scenario.settings.start_date
                or leaf_config.cadence_days != plan.scenario.settings.growth_interval_days):
            raise TripletError('calendar', 'nominal midnight/offset and LEAF growth cadence must match scenario')
        g, domain = wind._context.geometry, plant_state.domain
        shape = (g.ny, g.nx)
        if ((domain.ny, domain.nx) != shape or domain.dx_m != g.dx_m or domain.dy_m != g.dy_m
                or domain.origin_x_m != 0 or domain.origin_y_m != 0
                or tuple(domain.periodic) != (g.boundary_x.kind == 'periodic', g.boundary_y.kind == 'periodic')):
            raise TripletError('geometry', 'plant domain and actual native landscape mapping differ')
        if type(active_mask) is not np.ndarray or active_mask.dtype != np.bool_ or active_mask.shape != shape or not active_mask.flags.c_contiguous:
            raise TripletError('mask', 'explicit canonical host boolean active mask required by native GPU vegetation binding')
        footprint = getattr(wind._context, 'domain', None)
        if footprint is not None:
            from maple.core.backend import to_host
            if not np.array_equal(active_mask, to_host(footprint.valid_mask)):
                raise TripletError('mask', 'SYRUP/LEAF active mask differs from MAPLE physical footprint')
        if (type(terrain) is not wp.TerrainFactor or terrain.backend != 'cuda' or terrain.shape != (g.ny * g.nx,)
                or terrain.latitude_deg != wind._wind._admission.latitude_binding()['value'] or not terrain.verify_identity()):
            raise TripletError('terrain', 'genuine CUDA terrain PET mapping at the admitted latitude required')
        self.plan, self.wind, self.policy, self._limits = plan, wind, policy, limits
        self._terrain, self._partition, self._controls = terrain, partition, site_controls
        self._mask, self._builder, self._wind_policy, self._day_control = active_mask, event_builder, wind_vegetation_policy, day_control
        self._leaf_package, self._helpers, self._anchor = expected_leaf_package_sha256, expected_external_helpers, anchor
        self._failed, self._ordinal, self._last_receipt = None, 0, None
        leaf_source.recheck(max_bytes=limits.leaf_bridge_bytes, reserve_bytes=policy.host_reserve_bytes)
        boundary = wind.start(owner, bed, initial_event_factory, preferential=preferential)
        if boundary.root.column.backend != 'cuda' or not np.array_equal(active_mask.reshape(-1), boundary.root.host_active):
            raise TripletError('mask', 'initial native CUDA mask/column differs')
        self._native = self._native_footprint(boundary)
        foreign = sum(self._native) + policy.host_reserve_bytes + policy.device_reserve_bytes
        if leaf_config.reserve_bytes < foreign:
            raise TripletError('leaf_reserve', 'LEAF aggregate caller reserve must cover both native payload pools')
        # LEAF's existing max_bytes is an aggregate ledger. Charging its entire
        # remaining ceiling to EACH pool is conservative, never pooling free RAM.
        ceiling = leaf_config.max_bytes - leaf_config.reserve_bytes
        policy.admit(self._native[0] + ceiling, self._native[1] + ceiling, 'initial CUDA plants')
        day_binding = ps.DayBinding(wind.run_digest, anchor.date().isoformat())
        self._connection = fb.connect(leaf_source, wind.lineage, boundary, day_binding=day_binding)
        self.plants = ps.PlantSession.start(plant_state, leaf_config, anchor=anchor, day_binding=day_binding,
                                            initial_resource=self._connection.resource_cursor)
        self._days = iter(plan)
        self._apply_vegetation()

    @property
    def lineage(self):
        return self.wind.lineage

    @property
    def accepted_days(self):
        return self._ordinal

    @property
    def failure(self):
        return None if self._failed is None else dict(self._failed)

    @staticmethod
    def _native_footprint(boundary):
        r = boundary.root
        h, d = footprint(boundary.bed, r.water_state, r.history, r.depth_m, r.ks_m_s,
                         boundary.surface_depth_m, boundary.qx_m2_s, boundary.qy_m2_s, boundary.routing, r.config)
        # Owning estimates include opaque operator maps not found by traversal.
        d += int(r.column.estimate.retained_bytes) + int(r.matrix_op.retained_bytes)
        d += 0 if r.pref_op is None else int(r.pref_op.retained_bytes)
        if r.vegetation is not None:
            h += int(r.vegetation.owned_bytes) + int(r.vegetation.borrowed_bytes)
            d += int(r.vegetation.device_bytes)
        return h, d

    def _held(self, extra=0):
        # Conservative per-pool plant aggregate pricing, without fake host W.
        plant = self.plants.resident_bytes() + self.plants.candidate_bytes()
        return self._native[0] + plant + int(extra), self._native[1] + plant + int(extra)

    def _observe_surface(self, surface):
        return None

    def _prepare_weather_terrain(self):
        return None

    def _prepare_after_storm_weather(self):
        return False

    def _observe_water_progress(self, state, result):
        return None

    def _quota(self, extra=0, *, budgets=False):
        h, d = self._held(extra)
        q = dict(available_bytes_fn=self.policy.device_available_bytes_fn,
                 reserve_bytes=self.policy.device_reserve_bytes + d,
                 host_available_bytes_fn=self.policy.host_available_bytes_fn,
                 host_reserve_bytes=self.policy.host_reserve_bytes + h)
        if budgets:
            q.update(memory_budget_bytes=self.policy.max_device_bytes, host_memory_budget_bytes=self.policy.max_host_bytes)
        return q

    def _bridge_pools(self, extra):
        from leaf.floating_lifecycle_bridge_cuda import CudaPools
        h, d = self._held(extra)
        return CudaPools(min(self._limits.leaf_bridge_bytes, self.policy.max_device_bytes),
            self.policy.device_available_bytes_fn,
            min(self._limits.leaf_bridge_bytes, self.policy.max_host_bytes),
            self.policy.host_available_bytes_fn,
            device_reserve_bytes=self.policy.device_reserve_bytes + d,
            host_reserve_bytes=self.policy.host_reserve_bytes + h)

    def _admit_leaf(self, extra):
        h, d = self._held(extra)
        required = h + d + self.policy.host_reserve_bytes + self.policy.device_reserve_bytes
        if self.plants.config.reserve_bytes < required:
            raise TripletError('leaf_reserve', 'LEAF aggregate reserve does not cover the actual coexisting frontier')
        ceiling = self.plants.config.max_bytes - self.plants.config.reserve_bytes
        self.policy.admit(h + ceiling, d + ceiling, 'CUDA plant operation')

    def _apply_vegetation(self):
        from leaf import water_vegetation as lw
        from maple.core.backend import to_host
        from maple.core.types.topographic_commit import committed_elevation_m
        from maple_syrup import vegetation_lifecycle as vl, water_vegetation_binding as vb
        parent = self.lineage.head
        self._native = self._native_footprint(parent)
        h, d = self._held()
        n = self._mask.size
        self.policy.admit(h + self._limits.projection_bytes + 8 * n, d + self._limits.projection_bytes, 'cold vegetation projection')
        # Existing native generation/binding API explicitly hashes host surface
        # and mask. Only the derived N-cell surface crosses this cold boundary.
        surface = to_host(committed_elevation_m(parent.bed.committed_topography))
        self._observe_surface(surface)
        projection = lw.project_from_plant_session(self.plants, surface, active_mask=self._mask,
            max_edges=self._limits.projection_edges, max_bytes=self._limits.projection_bytes,
            reserve_bytes=self.policy.host_reserve_bytes + h, backend='cuda', device=self.policy.device_id,
            available_bytes_fn=self.policy.host_available_bytes_fn, max_device_bytes=self.policy.max_device_bytes,
            device_reserve_bytes=self.policy.device_reserve_bytes + d, available_device_bytes_fn=self.policy.device_available_bytes_fn,
            **({} if self._require_receiver_traits else {'require_receiver_traits': False}))
        ident = projection.identity_record()
        binding = vb.bind_projection(projection, self._controls, expected_leaf_package_sha256=self._leaf_package,
            expected_projection_sha256=ident['content_sha256'], expected_source={k: v for k, v in ident['source'].items() if k != 'note'},
            expected_external_helpers=self._helpers, active_mask=self._mask, current_surface=surface, session=self.plants,
            max_bytes=self._limits.binding_bytes, reserve_bytes=self.policy.host_reserve_bytes + h,
            available_bytes_fn=self.policy.host_available_bytes_fn)
        boundary = vl.apply_vegetation_generation(self.lineage, parent, binding,
            memory_budget_bytes=self.policy.max_device_bytes, host_memory_budget_bytes=self.policy.max_host_bytes,
            **self._quota())
        self._native = self._native_footprint(boundary)
        return boundary

    def _wind_vegetation(self):
        from leaf import canopy
        from maple.case_tools.importers.leaf_vegetation import vegetation_from_leaf_view
        from maple.core.types.topographic_commit import committed_elevation_m
        from maple.core.backend import to_host, to_device_tree
        h, d = self._held()
        g = self.wind._context.geometry
        need = 4096 * int(self.plants.base.n) + 1024 * g.ny * g.nx + (4 << 20)
        if need + h > self._limits.wind_vegetation_bytes:
            raise TripletError('memory', 'cold wind vegetation mapping exceeds its explicit cap')
        self.policy.admit(h + need, d + 64 * g.ny * g.nx, 'wind vegetation mapping')
        st = self.plants.snapshot_state()
        surface = to_host(committed_elevation_m(self.lineage.head.bed.committed_topography))
        view = canopy.maple_vegetation_view(st, self.plants.config.catalog, surface,
            statuses=self._wind_policy.statuses, height_mode='exposed', unsupported_policy=self._wind_policy.unsupported_policy,
            drag_coefficient=self._wind_policy.drag_coefficient)
        mapped = vegetation_from_leaf_view(view, g, population=st, catalog=self.plants.config.catalog,
            attest_view_from_population=True,
            **({} if getattr(self.wind._context, 'domain', None) is None else {'valid_mask': self._mask}))
        return to_device_tree(mapped.vegetation, self.policy.gate())

    def _factory(self, boundary, schedule=None):
        from maple_syrup import vegetation_lifecycle as vl
        return vl.event_factory_for(boundary, lambda v, t, c: self._builder(v, t, c, schedule))

    def _complete(self, boundary, schedule=None, extra=0):
        from maple_syrup import floating_event_lifecycle as lc
        occurrence, state = lc.begin_occurrence(self.lineage, boundary, self._factory(boundary, schedule), **self._quota(extra, budgets=True))
        event = occurrence.event
        for _ in range(self._limits.max_event_calls):
            end = float(state.t_s) + self._limits.event_chunk_s
            if not math.isfinite(end) or end <= state.t_s:
                raise TripletError('clock_limit', 'native event cannot advance', active_state=state)
            result = event.evolve(state, end, stop_when_quiet=True, max_steps=self._limits.max_native_steps_per_call)
            state = result.state
            self._observe_water_progress(state, result)
            if result.quiet_ready:
                retained = event.settle_retaining_water(state)
                state = retained.state
                if retained.complete:
                    audit = event.audit(state)
                    boundary = lc.close_occurrence(self.lineage, occurrence,
                        event.completion_capability(retained, state), state, **self._quota(extra, budgets=True))
                    self._native = self._native_footprint(boundary)
                    return boundary, audit
            elif not result.reached_target:
                raise TripletError('water_incomplete', result.stop_reason, active_state=state)
        raise TripletError('water_incomplete', 'native event execution limit is not completion', active_state=state)

    def _recession(self, extra=0):
        for _ in range(self._limits.max_recession_occurrences):
            if self.lineage.head.eligible_for_interevent:
                return
            self._complete(self.lineage.head, extra=extra)
        raise TripletError('water_incomplete', 'native recession did not finish', active_state=self.lineage.head)

    def advance(self):
        from leaf import floating_lifecycle_bridge_cuda as fb, plant_session as ps
        from maple_syrup import floating_daily as fd, floating_event_daily_bridge as br, recharge as rc, weather_pet as wp
        cp = self.policy.gate()
        if self._failed is not None:
            raise TripletError('session_stopped', 'inspect the earlier native/plant failure before continuing')
        if self._ordinal == self.plan.day_count:
            raise StopIteration
        day = next(self._days)
        start = float(self.lineage.head.t_s)
        active, stage, storm = None, 'wind', False
        try:
            for occurrence in day.winds:
                self._recession()
                self._apply_vegetation()
                vegetation = self._wind_vegetation()
                h, d = self._held(footprint(vegetation)[1])
                self.wind.execute(occurrence, vegetation=vegetation, external_reserve_bytes=d, external_host_bytes=h)
                del vegetation
                self._native = self._native_footprint(self.lineage.head)
            self._recession()
            self._apply_vegetation()
            stage = 'weather'
            self._prepare_weather_terrain()
            h, d = self._held(40 * self._terrain.shape[0])
            self.policy.admit(h, d, 'CUDA daily PET')
            if d > self._limits.pet_bytes:
                raise TripletError('memory', 'PET workspace exceeds explicit cap')
            w = day.weather
            native_weather = wp.DailyWeather(w.date, w.tmin_c, w.tmean_c, w.tmax_c, pet_mm_day=w.pet_mm_day)
            demand = wp.potential_demand(native_weather, self._terrain.latitude_deg, method=self.plan.scenario.weather.pet_method)
            applied = wp.apply_terrain(demand, self._terrain)
            reference = applied.pet_mm_day_cells.reshape(-1)
            pet = wp.pet_m_s(reference)
            if not day.evaporation_enabled:
                pet = cp.zeros_like(pet)
            pet_live = int(pet.nbytes) + int(reference.nbytes)
            stage = 'daily'
            boundary = self.lineage.head
            opening = br.open_day(self.lineage, boundary, partition=self._partition, run_digest=self.wind.run_digest,
                                  **self._quota(pet_live, budgets=True))
            rain = rc.plan_daily_rain(w.precipitation_mm, w.precipitation_intensity_mm_h,
                self.plan.scenario.weather.precipitation_threshold_mm, run_digest=self.wind.run_digest,
                ordinal=opening.cursor.ordinal, label=day.date.isoformat())
            result = br.day_action(opening, plan=rain, pet_terrain_m_s=pet, cover=None, event_factory=self._factory(boundary),
                                   day_control=self._day_control, **self._quota(pet_live))
            if result.outcome == 'requires_full_storm':
                stage = 'storm'
                pending = br.begin_storm_from_day(self.lineage, opening, rain, pet_terrain_m_s=pet, cover=None,
                    event_factory=self._factory(boundary), **self._quota(pet_live))
                result = opening = None
                self._complete(pending, br.storm_schedule(pending.pending_forcing), extra=pet_live)
                self._recession(pet_live)
                storm = True
                if self._prepare_after_storm_weather():
                    # The new terrain route may have a changed physical bed.
                    # Only rebuild derived PET inputs; the storm did no E/T.
                    h, d = self._held(pet_live + 40 * self._terrain.shape[0])
                    self.policy.admit(h, d, 'post-storm CUDA daily PET')
                    if d > self._limits.pet_bytes:
                        raise TripletError('memory', 'post-storm PET workspace exceeds explicit cap')
                    applied = wp.apply_terrain(demand, self._terrain)
                    reference = applied.pet_mm_day_cells.reshape(-1)
                    pet = wp.pet_m_s(reference) if day.evaporation_enabled else cp.zeros_like(reference)
                    pet_live = int(pet.nbytes) + int(reference.nbytes)
                boundary = self.lineage.head
                opening = br.open_day(self.lineage, boundary, partition=self._partition, **self._quota(pet_live, budgets=True))
                result = br.day_after_storm(opening, plan=rain, pet_terrain_m_s=pet, cover=None,
                    event_factory=self._factory(boundary), day_control=self._day_control, **self._quota(pet_live))
            if result.outcome != 'interevent' or result.state is None:
                raise TripletError('daily_incomplete', result.outcome, active_state=opening)
            fb.begin_day(self._connection, opening, leaf_resource_cursor=self.plants.resource_cursor)
            active, stage = result.state, 'plants'
            live = pet_live + (4 * active.ctx.n_layers + 12) * active.ctx.n_cells * 8 + self.plants.n_live * 8
            self._admit_leaf(live)
            nominal = self._anchor + dt.timedelta(days=self._ordinal)
            plan = self.plants.prepare_daily(ps.DayContext(nominal, w.tmean_c, day.transpiration_enabled, day.growth_enabled), terrain_pet=reference)
            if day.transpiration_enabled:
                read = fb.read_availability(self._connection, active, leaf_resource_cursor=self.plants.resource_cursor,
                    pools=self._bridge_pools(live))
                live += int(read.supply_m3.nbytes)
                self._admit_leaf(live)
                proposal = self.plants.propose_withdrawal(plan, read)
                live += int(proposal.retained_feedback_bytes) + 2 * int(proposal.proposed_m3.nbytes) + 8 * self.plants.n_live + 4096
                depth = fb.request_depth_m(read, proposal.proposed_m3, pools=self._bridge_pools(live))
                live += int(depth.nbytes)
                outcome = fd.request_floating_withdrawal(active, request_m=depth,
                    **fb.request_args(read, self._connection, active), **self._quota(live))
                live += sum(int(a.nbytes) for a in (outcome.actual_m, outcome.unmet_m, outcome.scale, outcome.state.water.water_m))
                accepted = fb.accepted_withdrawal(self._connection, proposal, outcome,
                    pools=self._bridge_pools(live))
                live += int(accepted.actual_m3.nbytes) + int(accepted.unmet_m3.nbytes)
                self._admit_leaf(live)
                staged = self.plants.stage_day(proposal, accepted)
                active = outcome.state
            else:
                staged = self.plants.stage_dry_day(plan)
            stage = 'close'
            closed = fd.close_floating_day(active, expected_parent=fd.expect_parent(active), **self._quota(live))
            self.policy.admit(*self._held(live), 'native CUDA handback')
            boundary = br.hand_back(self.lineage, closed.state, **self._quota(live))
            self._native = self._native_footprint(boundary)
            self.plants = self.plants.publish(staged)
            fb.commit_day(self._connection, active, closed.state, boundary, session=self.plants)
            self._ordinal += 1
            stage = 'receipt'
            record = dict(profile=self.profile, date=day.date.isoformat(), accepted_days=self._ordinal,
                scenario_sha256=self.plan.scenario.identity_sha256, run_digest=self.wind.run_digest,
                wind_occurrences=len(day.winds), next_wind_index=self.wind.next_wind_index,
                physical_start_s_hex=start.hex(), physical_end_s_hex=float(boundary.t_s).hex(),
                nominal_plant_cursor=self.plants.cursor.isoformat(), plant_sequence=self.plants.sequence,
                plant_geometry_sha256=self.plants.geometry_sha256, water_ledger=dict(boundary.ledger),
                full_storm_completed=storm, reference_pet_mm_day=demand.pet_mm_day,
                transpiration_enabled=day.transpiration_enabled, growth_enabled=day.growth_enabled,
                output_requested=day.output_due, checkpoint_requested=day.checkpoint_due, combined_checkpoint_supported=False,
                cuda=self.policy.record())
            raw = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
            if len(raw.encode()) > self._limits.receipt_bytes:
                raise TripletError('receipt_limit', 'accepted scalar receipt exceeds cap', active_state=boundary)
        except BaseException as exc:
            self._failed = dict(stage=stage, code=getattr(exc, 'code', 'native_refused'), detail=str(exc)[:500])
            if isinstance(exc, TripletError) or not isinstance(exc, Exception):
                raise
            raise TripletError(self._failed['code'], f'{stage}: {exc}', active_state=active) from exc
        self._last_receipt = AcceptedTripletDay(raw)
        return self._last_receipt

    def save_checkpoint(self, *args, **kwargs):
        raise TripletError('checkpoint_unqualified', 'combined CUDA plant/wind/water checkpoint is not qualified')
