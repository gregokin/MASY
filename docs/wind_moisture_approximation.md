# Wind moisture approximation and Phase 2 handoff

User decision, November 12, 2026: wind treats the soil as dry even when SYRUP's
actual soil or surface water is present. The planned daily wind path must not
require a physically dry surface and must not introduce a soil-moisture
correction to the existing dry-soil entrainment threshold.

This is a known temporary limitation: wetness and ponding do not suppress wind
entrainment in this approximation. Other configured native wind controls remain
their owning model's responsibility. This decision changes the planned coupling
contract; it does not claim that the new wind handoff is implemented or tested.

## Water remains real

SYRUP retains authority over the actual water inventories, persistent wetting
history, cumulative intake and water accounting. Treating soil as dry for wind
must not zero soil water or ponded water, drain the profile, reset history, count
an artificial water loss, or substitute a newly initialized water state.
Rainfall, drainage, evaporation and transpiration continue to use actual water
when their authorized native operations run.

Wind and water transport remain sequential. A previous water event must
genuinely finish rainfall and recession; a guard stop is incomplete. Remaining
quiet stored soil and surface water can be retained after genuine recession
completion. Wind completion must resolve airborne
sediment through actual MAPLE behavior. There is no extra daily hydrology, E/T
or growth during wind and no catch-up over wind duration. MASY must preserve
actual physical elapsed time and the nominal daily occurrence cursor.

## What the latest Phase 2 decisions resolve

The selected Phase 2 treatment provides the core rules for
wind-driven terrain change and persistent soil water:

- Fixed-thickness layers follow the current surface. Pure geometry/property
  reassignment retains each layer's water mass/equivalent volume before real
  accounted fluxes. Implicit relocation and layer averaging are accepted.
- Roots move with the soil, retaining the current fixed depth/cell fractions.
- Floating geometry introduces no erosion-only water sink. This supersedes the
  earlier disappearance proposal for this treatment.
- Updated properties use the approved field-capacity gravity cascade in a
  staged transaction before final capacity validation. Open-bottom drainage is
  external and counted once; a closed bottom first uses deeper storage, then
  returns only unavoidable excess to the surface as an internal transfer.
- Wetting history and cumulative intake persist without automatic resets.
  Property changes must not create duplicate drainage, E/T, rain, daily time or
  artificial elapsed time.

These decisions address the scientific core of the changed-terrain/retained-soil
water problem. They do not establish implementation or coupled acceptance.
The user has since resolved the two choices previously listed as open:
properties refresh at transport-event handoffs and remain fixed during an
event; the full MAPLE voxel-grid bottom is fixed, flat and impermeable. Only
real soil portions store/supply water; partial/zero-soil layers at rock are
allowed, and open-profile drainage applies only above rock. These selections
are recorded in `bedrock_refresh_decisions_20261113/user_decisions.json` in the
same SYRUP task. The native Phase 2 implementation is now independently
qualified and published on main at
`02388adb70b7f1f22b33b6b3d5f1249e223842e5`. The remaining external-wind adoption
interface is described in the [readiness check](daily_phase2_readiness_20261113.md).

The native release qualifies retained Green-Ampt history through its property
refresh operations. Preservation through the complete wind/water scheduler
still needs qualification; it is not authority to reset history.

The original design provenance is SYRUP's
`agent_handoffs/tasks/phase2_evolving_soil_20261112/syrup_candidate/docs/subsurface_hydrology/phase2_floating_profile_design.md`,
with the `floating_profile_decisions_20261113/user_decisions.json` and
`floating_capacity_decision_20261113/user_decisions.json` receipts in the same
task. Receipt timestamps are November 13 UTC. Exact inspected bytes/hashes are
retained in MASY's `agent_handoffs/tasks/wind_dry_approximation_20261112/`.
Those records describe the design reviewed at that time. Current publication
and source checks are in the [Phase 2 readiness record](daily_phase2_readiness_20261113.md);
the older Phase 1.5 package retains its original scope.

## Integration still required

Phase 2 is published. The remaining native external-wind adoption operation
must connect actual MAPLE wind execution to the live water lineage on the same
evolving bed, preserving water, clocks, ledgers and occurrence identity. Hydraulic
properties and routing follow their owning update rules; routing continues to
use the native configured net-elevation trigger, rather than rerouting simply
because an event boundary was crossed. Boundary publication and restart must
remain atomic and count each exchange once.

The current daily API still refuses wind. Existing wind profiles retain their
previous guards and qualification, including the legacy dry/reset paths; this
documentation does not remove those checks. The new persistent-water bridge
must implement the user-approved dry-wind approximation and prove preservation
of nonzero soil/surface water with wind, terrain updates and restart. The
[native wind component](daily_wind_executor.md) now executes individual CSV wind
occurrences while preserving surface water, with an explicit dry-wind
configuration view. Its [Phase 2 lifecycle handoff](daily_wind_phase2_handoff.md)
still needs the actual native operation. That work
can proceed independently of LEAF's plant-demand/growth connection.
