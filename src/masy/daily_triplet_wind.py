"""Explicit Phase 2.5 CPU wind profile with current shared LEAF vegetation.

Only source selection and native adapter calls live here. The older wind and
water profiles keep their own exact source pins and refusal policies.
"""
from __future__ import annotations

import dataclasses

from masy import daily_wind as dw
from masy import daily_wind_handoff as mh
from masy.daily_schedule import DailyPlan

PROFILE = dw._TRIPLET_PROFILE
MAPLE_COMMIT = "1dca8539b87bf80e2b4bb72d4c6637d02013a305"
MAPLE_PACKAGE_SHA256 = "191416153d3eca03e0b5e0817720d1365055de5b0b37391ac95454321aee7d68"
# Filled after independent review of the owning compatibility patch. A new
# source digest is never learned from whatever package happens to import.
SYRUP_PACKAGE_SHA256 = "0a9be299a81bea385575bc681933eeee07cdfbeea23743dd2cb97d9f968953e6"


def check_source(source, admission):
    from maple_syrup import native_case_profile as ncp

    facts = dw._source_facts(source, admission, expected_profile=ncp.CURRENT_PROFILE_NAME,
                             expected_maple_commit=MAPLE_COMMIT, expected_maple_package=MAPLE_PACKAGE_SHA256)
    if SYRUP_PACKAGE_SHA256 == "0" * 64 or facts["syrup_package_digest"] != SYRUP_PACKAGE_SHA256:
        raise dw.DailyWindError("native_source", "the independently qualified current triplet source is required")
    return facts


def validate_vegetation(vegetation, geometry):
    from maple.core.types.vegetation import VegetationState, validate_vegetation_state

    if type(vegetation) is not VegetationState:
        raise dw.DailyWindError("vegetation", "current canonical MAPLE VegetationState from LEAF is required")
    validate_vegetation_state(vegetation, int(geometry.ny), int(geometry.nx))


@dataclasses.dataclass(frozen=True)
class WindVegetationPolicy:
    """Caller-selected existing LEAF export controls; no response law."""

    statuses: tuple[str, ...]
    unsupported_policy: str
    drag_coefficient: float


def vegetation_from_session(session, bed, geometry, policy: WindVegetationPolicy, *, max_bytes, reserve_bytes, available_bytes_fn):
    """Project the accepted plants against this bed through the owning APIs."""
    from leaf import canopy
    from leaf.plant_session import PlantSession
    from maple.case_tools.importers.leaf_vegetation import vegetation_from_leaf_view
    from maple.core.types.topographic_commit import committed_elevation_m

    if type(policy) is not WindVegetationPolicy or type(session) is not PlantSession:
        raise dw.DailyWindError("vegetation", "genuine PlantSession and explicit export policy required")
    # Bounded Dense CPU frontend plan, before the session snapshot or native
    # rasterization. Includes all P/N array occurrences of the two public
    # adapters, with conservative Python/Source-check allowance; no P*N term.
    # Native wind's separate workspace is admitted by its own executor later.
    need = 4096 * int(session.base.n) + 1024 * int(geometry.ny) * int(geometry.nx) + (4 << 20) + reserve_bytes
    if type(max_bytes) is not int or max_bytes < 1 or type(reserve_bytes) is not int or reserve_bytes < 0:
        raise dw.DailyWindError("limits", "explicit wind vegetation byte cap and reserve required")
    if need > max_bytes or not callable(available_bytes_fn) or available_bytes_fn() < need:
        raise dw.DailyWindError("vegetation_memory", "wind vegetation frontend exceeds its explicit memory admission")
    state = session.snapshot_state()
    surface = committed_elevation_m(bed.committed_topography)
    view = canopy.maple_vegetation_view(state, session.config.catalog, surface, statuses=policy.statuses,
                                        height_mode="exposed", unsupported_policy=policy.unsupported_policy,
                                        drag_coefficient=policy.drag_coefficient)
    result = vegetation_from_leaf_view(view, geometry, population=state, catalog=session.config.catalog,
                                      attest_view_from_population=True)
    validate_vegetation(result.vegetation, geometry)
    return result


class TripletWindExecutor(mh.WindHandoffExecutor):
    """Native vegetated handoff with the same bounded scalar wind cursor."""

    def _native(self):
        from maple_syrup import floating_event_lifecycle as lc
        from maple_syrup import floating_wind_handoff_vegetated as wh

        return lc, wh

    def _digest(self, context_record):
        return mh._sha({"wind": super()._digest(context_record), "triplet_inputs_sha256": self._triplet_inputs})

    def execute(self, occurrence, *, vegetation, external_reserve_bytes=0):
        return super().execute(occurrence, vegetation=vegetation, external_reserve_bytes=external_reserve_bytes)

    def _head(self):
        lc, _wh = self._native()
        kind, head = lc.saveable_head(self.lineage, vegetation=True)
        if kind != "boundary":
            raise mh.WindHandoffError("not_ready", "a completed native Boundary is required")
        return head

    def checkpoint_record(self, **kwargs):
        raise mh.WindHandoffError("checkpoint_unqualified", "new triplet checkpoint needs the LEAF bridge cursor codec")

    def restore(self, *args, **kwargs):
        raise mh.WindHandoffError("checkpoint_unqualified", "new triplet restart has not been qualified")


def prepare_triplet_wind(plan, source, admission, context, *, wind_limits, handoff_limits, bridge_limits,
                         available_bytes_fn, run_inputs_sha256, reserve_bytes=0, batch_cells=64):
    """Opt in explicitly; no source fallback, bed construction or transport."""
    from maple_syrup import floating_wind_handoff_vegetated as wh

    if type(plan) is not DailyPlan or type(bridge_limits) is not mh.WindBridgeLimits:
        raise mh.WindHandoffError("plan", "parsed DailyPlan and explicit bridge limits required")
    if type(reserve_bytes) is not int or reserve_bytes < 0 or type(batch_cells) is not int or batch_cells < 1:
        raise mh.WindHandoffError("limits", "nonnegative reserve and positive batch size required")
    if not callable(available_bytes_fn) or type(handoff_limits) is not wh.WindHandoffLimits:
        raise mh.WindHandoffError("limits", "native handoff limits and memory reader required")
    if batch_cells > handoff_limits.max_batch_cells:
        raise mh.WindHandoffError("limits", "batch exceeds admitted handoff cap")
    if handoff_limits.max_cohorts < admission.compiled.config.particle_transport.max_active_cohorts:
        raise mh.WindHandoffError("limits", "handoff cap must admit native wind pool")
    wind = dw._prepare_daily_wind(source, admission, context, limits=wind_limits, profile=PROFILE)
    result = TripletWindExecutor(plan, wind, context, handoff_limits, bridge_limits, available_bytes_fn,
                                  reserve_bytes, batch_cells)
    result._triplet_inputs = mh._hex(run_inputs_sha256, "run_inputs_sha256")
    return result
