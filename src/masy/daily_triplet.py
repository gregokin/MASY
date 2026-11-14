"""Bounded Dense CPU composition of actual MAPLE, SYRUP and LEAF APIs.

The caller supplies native factories and scientific parameters. MASY supplies
order, nominal dates, physical-clock accounting and fail-stop coordination.
There is one live native water lineage and one accepted plant generation.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math

from masy.daily_schedule import DailyPlan
from masy.daily_triplet_wind import TripletWindExecutor, WindVegetationPolicy, vegetation_from_session

PROFILE = "masy.daily_triplet.dense_cpu.v1"
LEAF_PACKAGE_SHA256 = "98ae6e933f97f5e4df6e3c2cab757c240996b963a804bada81c38360bc7fc18a"
LEAF_EXTERNAL_HELPERS = ({"module": "benchmarks.spatial_mapping", "bytes": 47260,
    "sha256": "333921bb645d984fa8e7449619d03114262ee333c7fea61a22493a878fd9ccc5"},)


class TripletError(RuntimeError):
    """Failure evidence is local; a partial operation is never an accepted day."""

    def __init__(self, code, detail, *, active_state=None):
        self.code, self.detail, self.active_state = code, str(detail)[:500], active_state
        super().__init__(f"{code}: {self.detail}")


@dataclasses.dataclass(frozen=True)
class TripletLimits:
    event_chunk_s: float
    max_event_calls: int
    max_recession_occurrences: int
    max_native_steps_per_call: int
    pet_bytes: int
    projection_edges: int
    projection_bytes: int
    binding_bytes: int
    vegetation_law_bytes: int
    leaf_bridge_bytes: int
    wind_vegetation_bytes: int
    receipt_bytes: int
    reserve_bytes: int

    def __post_init__(self):
        if type(self.event_chunk_s) not in (float, int) or not math.isfinite(self.event_chunk_s) or self.event_chunk_s <= 0:
            raise TripletError("limits", "finite positive event_chunk_s required")
        for f in dataclasses.fields(self):
            if f.name == "event_chunk_s":
                continue
            value = getattr(self, f.name)
            minimum = 0 if f.name == "reserve_bytes" else 1
            if type(value) is not int or value < minimum:
                raise TripletError("limits", f"plain integer {f.name} >= {minimum} required")


@dataclasses.dataclass(frozen=True)
class AcceptedTripletDay:
    json_record: str

    def record(self):
        return json.loads(self.json_record)


class DailyTripletSession:
    """Explicit latest-source CPU run; combined persisted restart is refused.

    ``event_builder(vertical, topology, splash_cover, schedule)`` constructs a
    genuine PondEvent. ``schedule`` is the native absolute storm schedule, or
    None for a caller-authored zero-rain runtime. Native completion, routing,
    water budgets and sediment transfer checks remain authoritative.

    On failure this session stops. ``lineage.head``, ``plants`` and the error's
    active_state expose truthful progress. Earlier native event transactions
    may already have published; a whole day is NOT advertised as atomic.
    """

    def __init__(self, plan, *, wind, owner, bed, initial_event_factory, plant_state, leaf_config, anchor,
                 leaf_source, terrain, partition, site_controls, active_mask, event_builder,
                 day_control, wind_vegetation_policy, limits, expected_leaf_package_sha256, expected_external_helpers,
                 available_bytes_fn, preferential=None):
        import numpy as np
        from leaf import floating_lifecycle_bridge as fb, plant_session as ps
        from maple_syrup import weather_pet as wp
        from maple_syrup.experimental_storm import ExperimentalControl
        from maple_syrup import soil_evaporation as se, water_vegetation_binding as vb

        if type(plan) is not DailyPlan or type(wind) is not TripletWindExecutor or wind._plan is not plan:
            raise TripletError("plan", "one matching parsed plan and prepared TripletWindExecutor required")
        if type(limits) is not TripletLimits or type(wind_vegetation_policy) is not WindVegetationPolicy:
            raise TripletError("limits", "explicit triplet limits and wind vegetation policy required")
        if not callable(event_builder) or not callable(available_bytes_fn):
            raise TripletError("factory", "explicit native event builder and host-memory reader required")
        if (LEAF_PACKAGE_SHA256 == "0" * 64 or type(expected_leaf_package_sha256) is not str
                or expected_leaf_package_sha256.removeprefix("sha256:") != LEAF_PACKAGE_SHA256):
            raise TripletError("leaf_source", "the independently qualified complete LEAF package is required")
        if (type(expected_external_helpers) not in (list, tuple)
                or tuple(expected_external_helpers) != LEAF_EXTERNAL_HELPERS):
            raise TripletError("leaf_source", "the independently qualified external LEAF helper is required")
        if type(partition) is not se.EvaporationPartition or type(site_controls) is not vb.SiteControls:
            raise TripletError("controls", "genuine explicit native evaporation partition and vegetation site controls required")
        vb.verify_site_controls(site_controls)
        if type(day_control) is not ExperimentalControl:
            raise TripletError("control", "explicit native wet-interevent timestep control required")
        day_control.validated()
        if (type(anchor) is not dt.datetime or anchor.tzinfo is None or anchor.utcoffset() is None
                or anchor.time().replace(tzinfo=None) != dt.time() or anchor.date() != plan.scenario.settings.start_date):
            raise TripletError("calendar", "nominal start-date midnight with explicit fixed offset required")
        if type(leaf_config) is not ps.SessionConfig or leaf_config.backend != "numpy":
            raise TripletError("leaf_config", "explicit NumPy LEAF SessionConfig required")
        if leaf_config.cadence_days != plan.scenario.settings.growth_interval_days:
            raise TripletError("calendar", "LEAF growth cadence must match the resolved daily plan")
        shape = (wind._context.geometry.ny, wind._context.geometry.nx)
        geom, domain = wind._context.geometry, plant_state.domain
        if ((domain.ny, domain.nx) != shape or domain.dx_m != geom.dx_m or domain.dy_m != geom.dy_m
                or domain.origin_x_m != 0.0 or domain.origin_y_m != 0.0
                or tuple(domain.periodic) != (geom.boundary_x.kind == "periodic", geom.boundary_y.kind == "periodic")):
            raise TripletError("geometry", "plant domain must match the actual MAPLE grid, origin and per-axis periodic mapping")
        if type(active_mask) is not np.ndarray or active_mask.dtype != np.bool_ or active_mask.shape != shape:
            raise TripletError("mask", "explicit native-grid boolean active mask required")
        if (type(terrain) is not wp.TerrainFactor or terrain.backend != "numpy" or terrain.shape != (shape[0] * shape[1],)
                or terrain.latitude_deg != wind._wind._admission.latitude_binding()["value"] or not terrain.verify_identity()):
            raise TripletError("terrain", "genuine terrain-applied PET mapping at the admitted case latitude required")
        if 40 * shape[0] * shape[1] + limits.reserve_bytes > limits.pet_bytes:
            raise TripletError("limits", "PET quota must admit the per-cell reference, E forcing and native workspace")
        self.plan, self.wind, self._limits, self._available = plan, wind, limits, available_bytes_fn
        self._terrain, self._partition, self._controls = terrain, partition, site_controls
        self._mask, self._builder, self._wind_policy = active_mask, event_builder, wind_vegetation_policy
        self._day_control = day_control
        self._leaf_package, self._helpers = expected_leaf_package_sha256, expected_external_helpers
        self._anchor, self._failed, self._ordinal = anchor, None, 0
        self._last_receipt = None
        leaf_source.recheck(max_bytes=limits.leaf_bridge_bytes, reserve_bytes=limits.reserve_bytes)
        boundary = wind.start(owner, bed, initial_event_factory, preferential=preferential)
        self._native_bytes = self._native_footprint(boundary)
        if not np.array_equal(active_mask.reshape(-1), boundary.root.column._maps["active"]):
            raise TripletError("mask", "LEAF projection mask differs from the actual native soil mask")
        binding = ps.DayBinding(wind.run_digest, anchor.date().isoformat())
        self._connection = fb.connect(leaf_source, wind.lineage, boundary, day_binding=binding)
        if leaf_config.reserve_bytes < limits.reserve_bytes + self._native_bytes:
            raise TripletError("leaf_reserve", "LEAF reserve must include the held native landscape/water arrays")
        if available_bytes_fn() < limits.reserve_bytes + self._native_bytes + leaf_config.max_bytes - leaf_config.reserve_bytes:
            raise TripletError("memory", "host memory does not admit LEAF's declared construction ceiling")
        self.plants = ps.PlantSession.start(plant_state, leaf_config, anchor=anchor, day_binding=binding,
                                            initial_resource=self._connection.resource_cursor)
        self._days = iter(plan)
        self._apply_vegetation()

    @property
    def lineage(self):
        return self.wind.lineage

    @property
    def failure(self):
        return None if self._failed is None else dict(self._failed)

    @property
    def accepted_days(self):
        return self._ordinal

    @staticmethod
    def _native_footprint(boundary):
        """Positive structural inventory from actual owning array metadata.

        Aliases may be counted twice across owning estimates. No array values,
        bed columns or deep storage are scanned or materialized here.
        """
        from masy.daily_wind import _bed_helpers

        root = boundary.root
        total = _bed_helpers().nbytes(boundary.bed, root.water_state, root.history, root.depth_m, root.ks_m_s,
            boundary.surface_depth_m, boundary.qx_m2_s, boundary.qy_m2_s, boundary.routing, root.config)
        total += int(root.column.estimate.retained_bytes) + int(root.matrix_op.retained_bytes)
        total += 0 if root.pref_op is None else int(root.pref_op.retained_bytes)
        if root.vegetation is not None:
            total += int(root.vegetation.owned_bytes) + int(root.vegetation.borrowed_bytes)
        return total

    def _held(self, extra=0):
        return self._native_bytes + self.plants.resident_bytes() + self.plants.candidate_bytes() + int(extra)

    def _reserve(self, extra=0):
        return self._limits.reserve_bytes + self._held(extra)

    def _leaf_admission(self, extra):
        # PlantSession prices its own parent/candidate. Its immutable caller
        # reserve must cover the actual foreign native/bridge holdings too.
        foreign = self._limits.reserve_bytes + self._native_bytes + int(extra)
        if foreign > self.plants.config.reserve_bytes:
            raise TripletError("leaf_reserve", "LEAF caller reserve does not admit coexisting native/bridge holdings")
        if self._available() < foreign + self.plants.config.max_bytes - self.plants.config.reserve_bytes:
            raise TripletError("memory", "available host memory does not admit LEAF's declared stage ceiling")

    def _apply_vegetation(self):
        from leaf import water_vegetation as lw
        from maple.core.types.topographic_commit import committed_elevation_m
        from maple_syrup import vegetation_lifecycle as vl, water_vegetation_binding as vb

        parent = self.lineage.head
        self._native_bytes = self._native_footprint(parent)
        surface = committed_elevation_m(parent.bed.committed_topography)
        lim = self._limits
        projection = lw.project_from_plant_session(self.plants, surface, active_mask=self._mask,
            max_edges=lim.projection_edges, max_bytes=lim.projection_bytes, reserve_bytes=self._reserve(),
            backend="numpy", available_bytes_fn=self._available)
        identity = projection.identity_record()
        binding = vb.bind_projection(projection, self._controls,
            expected_leaf_package_sha256=self._leaf_package, expected_projection_sha256=identity["content_sha256"],
            expected_source={k: v for k, v in identity["source"].items() if k != "note"}, expected_external_helpers=self._helpers,
            active_mask=self._mask, current_surface=surface, session=self.plants,
            max_bytes=lim.binding_bytes, reserve_bytes=self._reserve(), available_bytes_fn=self._available)
        boundary = vl.apply_vegetation_generation(self.lineage, parent, binding,
            memory_budget_bytes=lim.vegetation_law_bytes, reserve_bytes=self._reserve(), available_bytes_fn=self._available)
        self._native_bytes = self._native_footprint(boundary)
        return boundary

    def _factory(self, boundary, schedule=None):
        from maple_syrup import vegetation_lifecycle as vl

        return vl.event_factory_for(boundary, lambda v, t, c: self._builder(v, t, c, schedule))

    def _complete_occurrence(self, boundary, schedule=None, *, extra=0):
        from maple_syrup import floating_event_lifecycle as lc

        lim = self._limits
        occurrence, state = lc.begin_occurrence(self.lineage, boundary, self._factory(boundary, schedule),
            available_bytes_fn=self._available, reserve_bytes=self._reserve(extra))
        event = occurrence.event
        for _ in range(lim.max_event_calls):
            end = float(state.t_s) + lim.event_chunk_s
            if not math.isfinite(end) or end <= state.t_s:
                raise TripletError("clock_limit", "event guard cannot advance native time", active_state=state)
            result = event.evolve(state, end, stop_when_quiet=True, max_steps=lim.max_native_steps_per_call)
            state = result.state
            if result.quiet_ready:
                retained = event.settle_retaining_water(state)
                state = retained.state
                if retained.complete:
                    audit = event.audit(state)
                    accepted = lc.close_occurrence(self.lineage, occurrence, event.completion_capability(retained, state), state,
                        available_bytes_fn=self._available, reserve_bytes=self._reserve(extra))
                    self._native_bytes = self._native_footprint(accepted)
                    return accepted, audit
            elif not result.reached_target:
                raise TripletError("water_incomplete", result.stop_reason, active_state=state)
        raise TripletError("water_incomplete", "native event exceeded its explicit execution guard", active_state=state)

    def _recession(self, *, extra=0):
        for _ in range(self._limits.max_recession_occurrences):
            if self.lineage.head.eligible_for_interevent:
                return
            self._complete_occurrence(self.lineage.head, extra=extra)
        if not self.lineage.head.eligible_for_interevent:
            raise TripletError("water_incomplete", "native boundary still requires recession", active_state=self.lineage.head)

    def advance(self):
        import numpy as np
        from leaf import floating_lifecycle_bridge as fb, plant_session as ps
        from maple_syrup import floating_daily as fd, floating_event_daily_bridge as br
        from maple_syrup import recharge as rc, weather_pet as wp
        from masy.daily_wind import _bed_helpers

        if self._failed is not None:
            raise TripletError("session_stopped", "inspect the earlier failure and native progress before a new run")
        if self._ordinal == self.plan.day_count:
            raise StopIteration
        day = next(self._days)
        start_s = float(self.lineage.head.t_s)
        active, stage, storm_completed = None, "wind", False
        try:
            for occurrence in day.winds:
                self._recession()
                self._apply_vegetation()
                mapped = vegetation_from_session(self.plants, self.lineage.head.bed, self.wind._context.geometry, self._wind_policy,
                    max_bytes=self._limits.wind_vegetation_bytes, reserve_bytes=self._reserve(),
                    available_bytes_fn=self._available)
                vegetation = mapped.vegetation
                del mapped  # the archive/population are not held through wind
                self.wind.execute(occurrence, vegetation=vegetation,
                    external_reserve_bytes=self._held(_bed_helpers().nbytes(vegetation)))
                del vegetation
                self._native_bytes = self._native_footprint(self.lineage.head)
            self._recession()
            self._apply_vegetation()
            stage = "weather"
            lim = self._limits
            pet_need = self._reserve(40 * self._terrain.shape[0])
            if pet_need > lim.pet_bytes or self._available() < pet_need:
                raise TripletError("memory", "available host memory does not admit PET workspace")
            w = day.weather
            native_weather = wp.DailyWeather(w.date, w.tmin_c, w.tmean_c, w.tmax_c, pet_mm_day=w.pet_mm_day)
            demand = wp.potential_demand(native_weather, self._terrain.latitude_deg, method=self.plan.scenario.weather.pet_method)
            applied = wp.apply_terrain(demand, self._terrain)
            reference = applied.pet_mm_day_cells.reshape(-1)
            pet = wp.pet_m_s(reference)
            if not day.evaporation_enabled:
                pet = np.zeros_like(pet)  # original plant PET remains intact
            pet_live = int(reference.nbytes) + int(pet.nbytes)
            stage = "daily_water"
            boundary = self.lineage.head
            active = br.open_day(self.lineage, boundary, partition=self._partition, run_digest=self.wind.run_digest,
                available_bytes_fn=self._available, reserve_bytes=self._reserve(pet_live))
            rain = rc.plan_daily_rain(w.precipitation_mm, w.precipitation_intensity_mm_h,
                self.plan.scenario.weather.precipitation_threshold_mm, run_digest=self.wind.run_digest,
                ordinal=active.cursor.ordinal, label=day.date.isoformat())
            result = br.day_action(active, plan=rain, pet_terrain_m_s=pet, cover=None, event_factory=self._factory(boundary),
                day_control=self._day_control, available_bytes_fn=self._available, reserve_bytes=self._reserve(pet_live))
            if result.outcome == "requires_full_storm":
                stage = "storm"
                pending = br.begin_storm_from_day(self.lineage, active, rain, pet_terrain_m_s=pet, cover=None,
                    event_factory=self._factory(boundary), available_bytes_fn=self._available, reserve_bytes=self._reserve(pet_live))
                result = active = None  # release the abandoned opening/day banks
                self._complete_occurrence(pending, br.storm_schedule(pending.pending_forcing), extra=pet_live)
                self._recession(extra=pet_live)
                storm_completed = True
                boundary = self.lineage.head
                active = br.open_day(self.lineage, boundary, partition=self._partition,
                    available_bytes_fn=self._available, reserve_bytes=self._reserve(pet_live))
                result = br.day_after_storm(active, plan=rain, pet_terrain_m_s=pet, cover=None,
                    event_factory=self._factory(boundary), day_control=self._day_control,
                    available_bytes_fn=self._available, reserve_bytes=self._reserve(pet_live))
            if result.outcome != "interevent" or result.state is None:
                raise TripletError("daily_incomplete", result.outcome, active_state=active)
            fb.begin_day(self._connection, active, leaf_resource_cursor=self.plants.resource_cursor)
            active = result.state
            stage = "plants"
            # Opening, result and bridge retain current daily W/J/I/surface
            # banks beside the accepted root. Reserve every bank occurrence.
            day_live = pet_live + (4 * active.ctx.n_layers + 12) * active.ctx.n_cells * 8 + self.plants.n_live * 8
            self._leaf_admission(day_live)
            nominal = self._anchor + dt.timedelta(days=self._ordinal)
            plant_plan = self.plants.prepare_daily(ps.DayContext(nominal, w.tmean_c, day.transpiration_enabled, day.growth_enabled),
                                                    terrain_pet=reference)
            if day.transpiration_enabled:
                read = fb.read_availability(self._connection, active, leaf_resource_cursor=self.plants.resource_cursor,
                    max_bytes=lim.leaf_bridge_bytes, reserve_bytes=self._reserve(day_live), available_bytes_fn=self._available)
                day_live += int(read.supply_m3.nbytes)
                self._leaf_admission(day_live)
                proposal = self.plants.propose_withdrawal(plant_plan, read)
                # The actual uptake plan owns demand/request/proposed/fraction
                # and a packet; retained_feedback_bytes names the coexisting
                # aggregate feedback banks reserved by LEAF's own contract.
                day_live += int(proposal.retained_feedback_bytes) + 2 * int(proposal.proposed_m3.nbytes) + 8 * self.plants.n_live + 4096
                depth = fb.request_depth_m(read, proposal.proposed_m3, max_bytes=lim.leaf_bridge_bytes, reserve_bytes=self._reserve(day_live))
                day_live += int(depth.nbytes)
                outcome = fd.request_floating_withdrawal(active, request_m=depth,
                    **fb.request_args(read, self._connection, active), available_bytes_fn=self._available, reserve_bytes=self._reserve(day_live))
                day_live += sum(int(a.nbytes) for a in (outcome.actual_m, outcome.unmet_m, outcome.scale, outcome.state.water.water_m))
                accepted = fb.accepted_withdrawal(self._connection, proposal, outcome,
                    max_bytes=lim.leaf_bridge_bytes, reserve_bytes=self._reserve(day_live))
                day_live += int(accepted.actual_m3.nbytes) + int(accepted.unmet_m3.nbytes)
                self._leaf_admission(day_live)
                staged = self.plants.stage_day(proposal, accepted)
                active = outcome.state
            else:
                staged = self.plants.stage_dry_day(plant_plan)
            stage = "close"
            closed = fd.close_floating_day(active, expected_parent=fd.expect_parent(active),
                available_bytes_fn=self._available, reserve_bytes=self._reserve(day_live))
            if self._available() < self._reserve(day_live):
                raise TripletError("memory", "handback publication cannot fit the held native/plant frontier")
            boundary = br.hand_back(self.lineage, closed.state)
            self._native_bytes = self._native_footprint(boundary)
            self.plants = self.plants.publish(staged)
            fb.commit_day(self._connection, active, closed.state, boundary, session=self.plants)
            self._ordinal += 1  # truthful even if the subsequent receipt cannot fit
            stage = "receipt"
            record = {"profile": PROFILE, "date": day.date.isoformat(), "accepted_days": day.day_number,
                "scenario_sha256": self.plan.scenario.identity_sha256, "run_digest": self.wind.run_digest,
                "wind_occurrences": len(day.winds), "next_wind_index": self.wind.next_wind_index,
                "physical_start_s_hex": start_s.hex(), "physical_end_s_hex": float(boundary.t_s).hex(),
                "nominal_plant_cursor": self.plants.cursor.isoformat(), "plant_sequence": self.plants.sequence,
                "plant_geometry_sha256": self.plants.geometry_sha256, "leaf_package_sha256": self._leaf_package,
                "water_ledger": dict(boundary.ledger), "reference_pet_mm_day": demand.pet_mm_day,
                "evaporation_enabled": day.evaporation_enabled, "transpiration_enabled": day.transpiration_enabled,
                "growth_enabled": day.growth_enabled, "output_requested": day.output_due,
                "full_storm_completed": storm_completed,
                "checkpoint_requested": day.checkpoint_due, "combined_checkpoint_supported": False}
            raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if len(raw.encode()) > lim.receipt_bytes:
                raise TripletError("receipt_limit", "accepted physical state exceeds requested scalar receipt cap", active_state=boundary)
        except BaseException as exc:
            self._failed = {"stage": stage, "code": getattr(exc, "code", "native_refused"), "detail": str(exc)[:500]}
            if isinstance(exc, TripletError) or not isinstance(exc, Exception):
                raise
            raise TripletError(self._failed["code"], f"{stage}: {exc}", active_state=active) from exc
        self._last_receipt = AcceptedTripletDay(raw)
        return self._last_receipt

    def save_checkpoint(self, *args, **kwargs):
        raise TripletError("checkpoint_unqualified", "combined LEAF/wind/water persisted restart is not qualified for this profile")
