# Daily wind to Phase 2 water handoff

The [native wind/water interface](daily_wind_handoff_interface.md) now implements this additional connection on its
explicit Dense NumPy CPU source profile. See its [qualification](daily_wind_handoff_qualification_20261113.md).
The requirements and original published-Phase-2 observations below remain the design record; Phase 2 component
publication alone did not provide this external-wind operation.

SYRUP Phase 2 is now published on main at
`02388adb70b7f1f22b33b6b3d5f1249e223842e5`. Its soil refresh, event/day bridge
and native water checkpoints are implemented and qualified at their documented
scope. The remaining external-wind adoption described here is an additional
integration interface; see the
[published-source readiness check](daily_phase2_readiness_20261113.md).

The user has resolved the Phase 2 scientific choices. SYRUP's
`bedrock_refresh_decisions_20261113/user_decisions.json` and
historical `phase2_completion_architecture.md` record the selected rules and
original implementation slices. No new scientific answer is requested here.

MASY's wind component executes actual MAPLE wind from a parsed CSV occurrence
and returns a staged current bed, actual physical end time and accounting.
Neither the historical N15 lifecycle nor the published Phase 2 lifecycle exposes
a public operation to adopt this wind-altered bed/time into a live water lineage.
Their factory-only boundary states cannot be edited or retimed by MASY. The
external-wind lifecycle operation is the remaining dependency for a runnable
coupled daily loop. This page specifies the MASY-facing requirements; it does
not claim a function with these arguments already exists or dictate the owning
developer's API name.

## State and time crossing the boundary

Inputs must identify the actual accepted water parent, expected parent token,
complete wind-returned native bed, wind completion/accounting evidence, actual
wind start/end seconds, and unique scenario/occurrence identity. Shared storage,
class order, geometry/mask/boundary/source compatibility and mobile-load
completion must be checked. The original landscape is never reinitialized.

The native operation returns a genuine updated lifecycle operator/state plus
an accounted handoff receipt. It preserves the daily run digest, nominal day
ordinal, water budget anchors, accepted history and request counters. The
combined physical clock counts wind without running a soil/E/T/growth interval
for that duration. Keep an exact reversible mapping to native elapsed values;
any native clock transition must come from the owning API and remain consistent
with shared-bed commit timestamps. Any genuinely required water recession adds its actual time
and must finish before a later process starts. No fake positive timestep or
manual rewriting of a minted lifecycle state is permitted.

Native source/profile/checkpoint versions must identify the new operation and
property generations explicitly. Historical N15/C1 pins are not rebound. MASY
must qualify the actual new source pair and interfaces before enabling this
path; its existing water-only guard is preserved until then.

## Approved Phase 2 treatment to reuse

At each transport-event handoff, refresh affected columns; properties remain
fixed within the event. A composition change matters even when elevation is
unchanged. Fixed nominal soil layers follow the current surface and retain
water per layer through pure reassignment before accounted fluxes. Preserve
Green-Ampt history/cumulative intake. No erosion-only water sink is introduced.

Only actual soil portions above the authoritative flat, impermeable FULL MAPLE
voxel-grid bottom provide pore storage and root water. Partial/zero-soil layers
must use Phase 2's real support; a resident-window floor is not bedrock. Roots
keep their fixed fractions without plasticity or implicit renormalization.

The staged new-capacity transaction uses the approved field-capacity cascade
once: open-profile bottom export is counted once and applies only above rock;
a closed/rock bottom uses deeper storage before returning unavoidable excess
to the surface. There is no transmission through rock, artificial substrate,
water erasure, second drainage/E/T/rain/day operation or extra interval. The
native owner determines actual required flow completion while keeping stored
surface water truthful.

Hydraulic routing retains the actual graph and last routing-source surface.
Only the owning net-elevation/compatibility rule decides its refresh. An event
boundary or property update alone must not force rerouting.

## Scheduler order and acceptance

For a nominal date: each wind row in file order, its native Phase 2 handoff,
then the next wind row; after all winds, that date's precipitation, actual rain
and recession as required, and one daily soil/E update. Plant transactions
remain separately connected to LEAF. Wind uses the documented dry-soil
threshold approximation while the real water remains with SYRUP.

A failure must retain the last accepted combined boundary. Each wind export,
property generation, drainage and surface return is accounted once. Restart
must either retain the accepted per-occurrence cursor with the physical state,
or publish the whole day's staged result atomically at a day boundary. It must
never restore old water and then skip an already published wind, nor replay an
export or water exchange. Active-event restart is a separate qualification
from a closed daily boundary.

## Joint acceptance once the native hook exists

Run real wind → native property/capacity handoff → reduced/full rain → daily
water → wind, including retained nonzero water and nonzero sediment movement.
Compare actual state and fluxes with independent owning-model controls; verify
per-class sediment and total surface+soil water budgets, repeated same-day
files, exact physical clocks, no biological catch-up, failed staged handoffs
and fresh-process restart. Include composition-only changes, open-above-rock
drainage and partial/zero soil at impermeable bedrock through the qualified
Phase 2 implementation. CPU/GPU/storage paths keep separate acceptance.
