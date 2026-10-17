# Wrapper architecture and handoff contract

This is the initial design contract, not a statement that integration is implemented.

## Components

| Component | Authority | Wrapper responsibility |
|---|---|---|
| MAPLE | Wind transport, sediment voxels/active layer, availability, wind caches and shared storage | Invoke actual scheduler; translate forcing and boundary state |
| SYRUP | Rain/infiltration, hydraulics, wet sediment transport and storm completion | Construct event state from the current bed; invoke conservative event APIs |
| MASY | Event order/global clock, adapters, dependency pair, aggregate audit and joint restart | Coordinate, validate and persist; no physics duplication |

Prefer one Python process with one qualified MAPLE package and thin `masy.adapters` modules; separate-process execution is an option only after a lossless portable state contract is demonstrated. Do not install two incompatible `maple` modules into one interpreter. Future names describe modules to implement, not existing APIs.

## Scheduling

An ordered list defines wind and water events. Wind forcing uses native MAPLE shear velocity, direction and durations. Rainfall uses native SYRUP semantics and exact units/interval integration. The wrapper derives global intervals from actual elapsed durations; maps between local and global clocks must be invertible and persisted.

A rain series finishing does not end its water event: recession and terminal sediment handling count as elapsed time. A wind block finishing does not authorize deleting airborne cohorts. First integration must establish native supported end-of-wind handling and report any extra physical settlement time. A rejected/incomplete event stops before the next process starts, with a truthful restart boundary.

Sequential examples are not fixed-calendar forecasts. Future prescribed absolute start times must refuse overlap with ongoing flow/transport unless a separately specified physical policy exists. No concurrent transport, forcing truncation, hidden time dilation, repeated storm rain or rain dropped during wind.

## Landscape identity

One class-resolved MAPLE bed authority is shared. A handoff envelope must identify:

- Geometry, datum, grain identities/order, bulk density/porosity and mask; each process's explicit boundary mapping.
- Complete voxel/deep-storage state and active-layer masses; availability/bound mass and any substrate constraints.
- Current physical surface, committed terrain, pending ledger/journal and immutable store/version references. Resolve pending accounting exactly once before a boundary; retain every referenced version.
- Prescribed vegetation/pavement/surface modifiers and process-specific handling, with no implicit reinitialization.
- Backend/device ownership, dtype, clocks, event identity, provenance, exports/imports and parent state identity.

Keep native representations when both support them. If wind currently needs dense columns, start with a small dense case; do not materialize Chastre's full bed to claim windowed compatibility. Storage conversion and CPU/GPU transfer are separately measured and validated paths.

## Process-private state and event readiness

Wind owns cohorts, trajectory/equilibrium state and wind-derived caches. Water owns soil/surface water, discharge, transport phase/load, routing graph/source and hydraulic context. Persist each during its active event. Retain the SYRUP routing graph and its last-build source surface in the combined handoff envelope during subsequent wind events and dry intervals; they are not discarded by the dry reset. At a complete water boundary require genuine SYRUP `DryHandoff`, account for final deposition and external water reset, and pass its resulting bed to wind. At a wind boundary establish an actual conservative cohort disposition before water pickup begins.

Preserve MAPLE availability rules: water pickup permission can differ from wind pickup permission without resetting inventories. After water changes, refresh/invalidate actual MAPLE wind caches and continuation compatibility; after wind changes, compare physical elevation to the retained water routing source. Build a water graph initially; thereafter respect net displacement `max_active(abs(z_current-z_last_routing)) >= configured_threshold`. Checkpoints, accounting and switching processes alone are not hydraulic reroute triggers. Do not claim this rule covers independently changed masks/boundaries/options; incompatible case geometry requires explicit new initialization/qualification.

## Dependency policy

Resolve current committed local HEAD of each configured upstream at new-run start. Refuse dirty runtime/config sources unless the user explicitly selects a labelled development snapshot; record every included delta and hash source before/after snapshot creation. Unrelated untracked cases are outside the runtime scope. Record actual package paths, revisions, scoped dirty changes and deterministic file hashes. Reuse existing upstream provenance/bundle mechanisms and qualify the pair before mutation. Freeze run sources to immutable snapshots or verified bundles; upstream development can continue independently.

A shared MAPLE runtime must meet both the current wind interface and SYRUP's requirements. SYRUP's exact-source storage checks remain active until a reviewed compatibility update replaces them. If newest sources do not qualify, fail preflight with the concrete requirement and proposed remedy; never pretend an older snapshot is newest. An explicit prior compatible lock may be selected for a reproducible run, visibly.

Restarts use the original source pair and schedule. New upstream versions are for new runs. Upgrades of existing state need explicit, separately qualified migration. Do not auto-pull or alter active environments/files.

## Accounting and persistence

At every event boundary audit per-class `bed + wind_mobile + water_mobile + cumulative_export - cumulative_import` against its initial total using actual upstream tolerance definitions. Count internal bed/mobile transfers once. Separately audit rainfall, infiltration/drainage, stored water, export and documented dry reset. Use model outputs, not invented correction terms.

Persist an atomic combined manifest that references native checkpoints, compatible MAPLE state/store versions, source lock, resolved forcing and current event/global-local time mapping. Do not invent a portable NPZ that omits deep stores. Validate before adopting a handoff; document input mutation and stage a recoverable boundary so upstream exceptions cannot corrupt the accepted state. Crash/resume within wind, within wet recession and at both transitions must equal uninterrupted continuation within declared contracts.

## Initial limits

The combined runner is not implemented at setup. Current SYRUP legacy replay is unsuitable for coupling; conservative event mode is required. SYRUP's no-spill/closed-boundary Chastre limitation remains. Direct water splash, growth, full ecohydrology/ET/nutrients and overlapping transport are excluded. Small dense CPU success does not establish windowed/GPU/full-catchment support.

## Bootstrap review refinements

The import resolver must verify that the actual `maple`, `maple_syrup` and `masy` module origins and distribution metadata match immutable run sources, even when the chosen existing interpreter has editable installs. Reuse upstream dependency-bundle metadata mechanisms; never silently use a live checkout through site-packages. Record environment/package versions and relevant compiled-extension identities, disable upstream bytecode writes, and keep caches/output in run-owned paths.

Wind settlement must have a finite explicit guard and stop incomplete with preserved load if it cannot finish. Never park airborne cohorts across a water event. All process case parameters, masks and prescribed surface mappings are hashed alongside forcing. Shared vegetation is prescribed landscape content; process-specific modifiers retain declared units and time basis. Water reset belongs only in water accounting unless an actual documented operation also exchanges sediment.

Global-clock representation is a Phase-1 contract: retain native solver clocks and returned elapsed values without timestep rounding, using exact accumulation/anchors at event boundaries so restart maps do not drift. Input schedule and realized intervals are distinct records. Fixed-calendar start fields are rejected in the first sequential schema.

Rollback should use versioned deep references and bounded journals/touched-window staging, not a full deep-volume copy; the tiny dense milestone can stage a bounded copy. Preflight free space and declared host/device/store/thread budgets. Keep every reference needed by a retained checkpoint and do not add automatic GC.
