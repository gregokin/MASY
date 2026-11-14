# Published Phase 2 and the daily wind handoff

This is the earlier published-source audit. The additional
[native wind/water interface](daily_wind_handoff_interface.md) now addresses the specific adoption gap described below;
its [separate qualification](daily_wind_handoff_qualification_20261113.md) records the new sources and checks. This audit's
Phase 2 source169 evidence is retained unchanged and is not a qualification of the added module.

SYRUP Phase 2 is complete and published at its bounded native API scope. MASY
verified that both `origin/main` and `origin/feature/subsurface-evolving-soil`
point to `02388adb70b7f1f22b33b6b3d5f1249e223842e5`. The source implementation
is `71bb8fe0bb0ed2d17ee302e4502ba0511032d416`, followed by manuals and main
integration. The clean main checkout is separate from the unrelated dirty
legacy development checkout. No further Phase 2 commit or push was necessary.

The published implementation supplies evolving soil properties, conservative
capacity reconciliation, event/day handoffs and native water checkpoints. Its
release records report 111 fresh CPU/host checks and 39 actual GPU checks at
their documented scope. MASY did not rerun those release suites or extend their
acceptance to a scheduled combined model.

## Fresh source and interface checks

Codex independently compared all 169 package files with both the committed Git
blobs and the published manifest. The package SHA-256 is
`3917dba4f59f662650bdf6fb08c10e4fece9d3de601c4b42ad41c9a796382c82`.
Its qualified MAPLE dependency remains commit
`9f466d2cfcd52991b24496cf67f4562d6268afa9`, package
`fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e`.

A fresh isolated CPU process used that exact Phase 2 source snapshot and the
actual immutable MAPLE bundle. Native source verification, genuine case
admission and Dense context preparation succeeded. The existing MASY wind
factory refused this source with `native_source`, as required by its explicit
N15 pin. No transport was run. Historical wind/water profiles and locks remain
unchanged; replacing a package hash is not sufficient integration qualification.

## What still needs connecting

The published public lifecycle does not accept an external wind bed into its
live water lineage. This is a specific remaining integration interface, not
unfinished Phase 2 soil physics:

* `floating_event_lifecycle.begin_occurrence` reuses `boundary.bed`.
  `floating_event_daily_bridge.hand_back` preserves the held bed, replacing its
  water depth as needed; neither takes a wind-altered bed.
* `FloatingPondVertical.check_initial` requires the accepted boundary's actual
  voxel and active-layer identities. A changed bed is explicitly refused.
* `native_floating_soil.stage_native_refresh` requires a current native owner.
  Opening the event or daily lineage permanently retires that owner, so using
  its old initial water after a day or storm is forbidden.
* `floating_event_refresh.stage_completed_event_refresh` requires a genuine
  completed Pond event and derives affected columns from its actual sediment
  ledger. A wind result cannot be passed off as such an event.

Codex inspected these callers; an independent source-only Claude audit reached
the same conclusion. The audit is not a new physics or GPU qualification.

## Required native extension

The [wind handoff specification](daily_wind_phase2_handoff.md) remains the
connection contract. The owning lifecycle needs a staged transaction that
accepts its current completed boundary and the actual completed wind result,
refreshes properties against current layer water/history, accounts for any
native reconciliation return/export once, and publishes the new accepted head
last. Fresh initialization must also support a first wind without inventing a
storm or consuming a nominal day.

The transaction must preserve the last real hydraulic routing source, the daily
cursor and lifetime accounting. A native capacity return that requires renewed
flow must remain pending until genuine native completion. Existing native
helpers provide the soil calculations; MASY must not reconstruct a fresh water
owner, mint private lifecycle records or fabricate completion to supply this
connection.

Exact global/native time mapping remains coordinator work. An explicit offset
can represent wind duration without running water physics, but it does not
authorize replacing the lineage bed or mixing incompatible shared-bed commit
timestamps. Preserve actual native elapsed values and use a genuine owning API
for any required state-clock transition; never invent a hydrology timestep.

After this operation exists, explicitly qualify its actual source pair, the
wind adapter and the combined wind → refresh → rain → daily-water sequence,
including failure recovery and exactly-once restart. Phase 2 component
publication alone does not establish those results. LEAF projection/coupling
and GPU/windowed scheduler acceptance remain separately scoped.

## Evidence

Local records are in `agent_handoffs/tasks/phase2_scheduler_readiness_20261113/`:
remote/source verification, complete submitted source hashes, exact prompts,
separate raw Claude output, PID/start identities and reaping, and the isolated
compatibility probe. The first Claude attempt failed to reach its API; a bounded
network-enabled retry completed the source-only audit. No saved simulation
payload was shared, no upstream worktree was changed, and no environment was
installed or updated.
