# Wrapper architecture and handoff contract

This is the initial design contract, not a statement that integration is implemented. Dated implementation status is appended at the end.

## Components

| Component | Authority | Wrapper responsibility |
|---|---|---|
| MAPLE | Wind transport, sediment voxels/active layer, availability, wind caches and shared storage | Invoke actual scheduler; translate forcing and boundary state |
| SYRUP | Rain/infiltration, hydraulics, wet sediment transport and storm completion | Construct event state from the current bed; invoke conservative event APIs |
| MASY | Event order/global clock, adapters, dependency pair, aggregate audit and joint restart | Coordinate, validate and persist; no physics duplication |

Prefer one Python process with one qualified MAPLE package and thin `masy.adapters` modules; separate-process execution is an option only after a lossless portable state contract is demonstrated. Do not install two incompatible `maple` modules into one interpreter. Future names describe modules to implement, not existing APIs.

## Scheduling

An ordered list defines wind and water events. Wind forcing uses native MAPLE shear velocity, direction and durations. Rainfall uses native SYRUP semantics and exact units/interval integration. The wrapper derives global intervals from actual elapsed durations; maps between local and global clocks must be invertible and persisted.

A rain series finishing does not end its water event: recession and terminal sediment handling count as elapsed time. At an intentional wind-event end, the user's October 18 rule assumes wind becomes zero and native MAPLE settles every remaining airborne cohort conservatively at the unchanged event-end clock. Pauses, checkpoints and intermediate forcing blocks retain ordinary continuation semantics. MASY calls and validates native finalization with an explicit event origin, adopts its returned bed/ledger and counts boundary export once. The native origin extension is committed and included in MAPLE `39bcf79`; the actual Dense CPU sequence passes with SYRUP `b4ee14e` under a fresh immutable lock (see [qualification](committed_pair_qualification.md)). Older APIs refuse before transport. A rejected event stops before the next process starts.

Sequential examples are not fixed-calendar forecasts. Future prescribed absolute start times must refuse overlap with ongoing flow/transport unless a separately specified physical policy exists. No concurrent transport, forcing truncation, hidden time dilation, repeated storm rain or rain dropped during wind.

## Landscape identity

One class-resolved MAPLE bed authority is shared. A handoff envelope must identify:

- Geometry, datum, grain identities/order, bulk density/porosity and mask; each process's explicit boundary mapping.
- Complete voxel/deep-storage state and active-layer masses; availability/bound mass and any substrate constraints.
- Current physical surface, committed terrain, pending ledger/journal and immutable store/version references. Resolve pending accounting exactly once before a boundary; retain every referenced version.
- Shared vegetation, pavement and surface modifiers with process-specific handling and no implicit reinitialization. The current executable profile prescribes vegetation; the planned evolving profile binds accepted native vegetation and soil-water versions and transactional resource changes.
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

Wind finalization must have finite time/resource guards and stop incomplete with preserved load if it fails. Under the instantaneous event-end assumption it adds no physical elapsed time; no calm coasting substitutes for that native operation. Never park airborne cohorts across a water event. All process case parameters, masks and prescribed surface mappings are hashed alongside forcing. Shared vegetation is prescribed landscape content in the currently qualified profile; process-specific modifiers retain declared units and time basis. The [vegetation information plan](vegetation_information_plan.md) specifies a future versioned shared state and atomic vegetation/soil-water updates between events. SYRUP retains authority over water fields, readiness and named budget exchanges; a future retained-subsurface-water profile cannot silently reinterpret an existing dry reset. Response laws and soil-water policy require the user's scientific direction. Water reset belongs only in water accounting unless an actual documented operation also exchanges sediment.

Global-clock representation is a Phase-1 contract: retain native solver clocks and returned elapsed values without timestep rounding, using exact accumulation/anchors at event boundaries so restart maps do not drift. Input schedule and realized intervals are distinct records. Fixed-calendar start fields are rejected in the first sequential schema.

Rollback should use versioned deep references and bounded journals/touched-window staging, not a full deep-volume copy; the tiny dense milestone can stage a bounded copy. Preflight free space and declared host/device/store/thread budgets. Keep every reference needed by a retained checkpoint and do not add automatic GC.

## MASY-04 implementation status

The [Dense Python runner](dense_runner.md) now implements and verifies the restricted in-process transition path on
an explicit frozen CPU pair. It stages small Dense bed copies and mutable private wind/water memory, checks event
and aggregate budgets and prepares all publication references before adoption. It retains the latest accepted bed,
private state, routing pair, cursor, receipts and exact clock ledger. This is an in-memory transaction boundary;
there is no persistent combined checkpoint or active-event resume yet. The broader storage/backend/restart contracts
above remain requirements for later qualification.

## October 19 current pair and shared vegetation plan

The fresh [MASY-09 qualification](source_pair_20261019.md) accepts MAPLE `39bcf79` with newly committed SYRUP `d05d8c6`, under lock `43976b9a…`. Earlier pair references above remain their historical acceptance. Bounded synthetic [Dense CPU production-feature coupling](dense_production_wind.md) is published; production calibration, windowed/GPU execution and combined persisted restart remain separate work. The [vegetation information plan](vegetation_information_plan.md) binds future accepted native vegetation and soil-water references, payload-bound updates and exactly-once receipts; physics questions require the user’s decision before dependent implementation.

## October 19 coherent CPU candidate status

Accepted CPU components now share one engine, and the default runner's results are unchanged:

- **Vegetation.** [Manual versioned vegetation](vegetation_information_plan.md#current-accepted-cpu-component-status-october-19)
  holds all five native fields. A native wake response reads the latest accepted cover. The SYRUP-owned soil object
  passes unchanged after the existing native dry reset, and the joint soil exchange seam refuses any nonempty
  exchange.
- **Persistence.** The [accepted-boundary store](dense_boundary_store.md) is a first bounded slice of the combined
  manifest above. It persists immutable originals, native checkpoints and passive images under an atomic `HEAD`,
  locked to source manifests. It restarts only from accepted boundaries.
- **Not yet met.** Active-event resume, power-loss recovery, committed-lock binding, dynamic-vegetation persistence,
  GPU and windowed storage are not covered. The restart and backend requirements above still stand for these.
- **Device.** The [private device runner](dense_device_runner.md) reuses the same engine. It is qualified only for
  CPU semantics.
