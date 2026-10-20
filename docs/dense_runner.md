# Dense runner (`masy.dense_runner`): ordered wind / rain events on one native bed

Status: **Dense CPU integration independently accepted on the committed pair (MASY-09).**
Genuine nonempty wind → rain/recession → nonempty wind → second rain/recession matches direct native execution
on one six-class evolving bed, including exact persistent fields, partial availability, native clocks, exports,
class/water budgets, routing and failure isolation. This is a Python API without production CLI or combined restart.
Scientific calibration, coupled saltation, GPU/windowed support and large-case performance are unqualified.

Current qualified runtime inputs: MAPLE `39bcf79e1e67b6ed960f78a923ddfedf1d173e68`, SYRUP
`d05d8c6c66d9fd2969c007681916254e8d1c2c7c`, MASY `22aa6ab65ad73a58fb372ffe09e3f7e0ca9893a1`.
Reference lock: `43976b9a6c3aa5384d0cdfdcf7c61f90d6a2f0ced45a0890e92dad5fb6a5cd33`.
The native `record_origin_s` extension is committed upstream. All earlier locks remain unchanged, including
MASY-07's preceding committed `b3e8756` qualification and MASY-08's `39bcf79`/`b4ee14e` qualification.
See [source-pair qualification](source_pair_20261019.md)
for actual checks, source-derived metadata, reproducibility and limits. MASY-06's uncommitted development pair
`3eed7f55c5d474531a4dcc38d5f4492a546249845e9a494b1c1348f9d1203320` is historical evidence only.

The later opt-in [production-feature profile](dense_production_wind.md) is
separately checked on bounded synthetic CPU fixtures. Its calibration and
resource limits do not broaden this reference profile's qualification.

On the coherent CPU candidate this runner is the shared engine. Its public constructor, its 26 instance attributes
and its ordinary result structure are preserved. Default and synthetic production-feature results, including
late wind/water fault operands, are byte-identical to the original runner. Two opt-in components build on it:
[manual vegetation updates](vegetation_information_plan.md#current-accepted-cpu-component-status-october-19) and
the [accepted-boundary store](dense_boundary_store.md). The runner itself remains in-memory only.

The caller must establish locked import paths, matching distribution metadata and actual module origins before
transport. The runner does not select or qualify dependencies itself. The existing runtime command qualifies source
identity; it does not launch this API or make an uncommitted wrapper a committed production source.

## API

```python
from masy.dense_runner import (DenseRunner, DenseRunnerControl, DenseRunnerLimits,
                               DenseWaterInputs, DenseWindInputs)

runner = DenseRunner(
    bed,                      # masy.dense_bed.DenseBed (build_dense_bed)
    schedule,                 # masy.dense_schedule.DenseSchedule (relative durations only)
    wind=DenseWindInputs(config=SchedulerPhysicsConfig(...),       # native
                         initial_mobile_state=zeros_mobile_cohort_state(...)),
    water=DenseWaterInputs(terrain=TerrainReference, column=..., field=RainfallField,
                           vegetation_cover_fraction=..., sediment=..., initial_soil_water_m=...),
    control=DenseRunnerControl(sediment_control=SedimentEventControl(...),
                               completion_policy=CompletionPolicy(...), report_every_s=10.0),
    limits=DenseRunnerLimits(max_events=..., max_wind_accounting_steps=..., max_storm_steps=...,
                             max_report_rows=..., max_work_bytes=..., max_receipt_bytes=...))
receipt = runner.step()      # one event, transactional; accepted receipt or the stop receipt
report = runner.run()        # step until the schedule is done or the first stop
runner.accepted              # AcceptedBoundary: next index, native time, bed identity, counts, routing source digest
runner.receipts              # accepted event receipts (plain JSON copies)
runner.stop_receipt          # None, or the stop receipt
runner.active_evidence       # latest unaccepted native output of a stopped event (never adopted)
runner.ledger_record()       # masy.clock.ClockLedger record of every accepted interval
```

The caller builds every native input. Product code never imports a test helper or a case CLI.
`tests/native/fixture_dense_alternation.py` is a source-only example that uses the actual constructors.

## Supported profile (refused before any mutation otherwise)

| Area | Requirement | Refusal code |
|---|---|---|
| Bed | `DenseBed`: NumPy float64 Dense, square, full mask, fixed voxel capacity, explicit boundaries | (bed codes) |
| Wind config | `geometry`, `grain_classes`, `avalanching_spec`, `topographic_commit_spec` are the bed context objects; equal mass-resolution bits; `initial_available_fraction` bitwise the bed policy | `incompatible_wind_config` |
| Wind mode | `detachment_transport_deposition`; `water_coupling_spec.enabled` False (no MAPLE water pass in wind); topographic wind off; no prescribed/reduced detachment, external shear, surface creep, transport-accuracy or cohort calendar | `unsupported_wind_mode` |
| Avalanching | explicitly disabled | `unsupported_avalanching` |
| Wind pool | flat native `MobileCohortState`, host arrays, genuinely empty | `unsupported_mobile_state`, `mobile_state_not_empty` |
| Water | `transport_scheme='upwind'` (phase None); `storm.implementation='array'`; `commit=True`; no exchange history; `deep_publication='step'` | `unsupported_transport_scheme`, `unsupported_hydraulic_implementation`, `unsupported_water_control` |
| Vegetation | the water vegetation array is bitwise the bed's canonical cover FRACTION (no percent conversion, no second field) | `vegetation_mismatch` |
| Inputs | `TerrainReference` on the bed grid; NumPy `RainfallField` on the bed grid; finite non-negative soil water | `invalid_water_inputs`, `incompatible_water_inputs` |
| Limits | events ≤ `max_events`; native `storm.max_steps` ≤ `max_storm_steps` | `event_limit`, `storm_step_limit` |

## Event sequence (only actual native calls)

**Every event** first runs the pre-event check, before any work copy, allocation or native call. It covers
`bed.verify_case_inputs()`, the live digest of every controlling input, and the accepted bed identity plus the host
arrays of the exposed pool and soil water. A difference can only come from outside the runner. It is refused as
`case_input_mutated_before_event`, `static_input_mutated_before_event` or `accepted_state_externally_mutated`, and
nothing is repaired or replaced. The stop receipt's `guarantee` then states that the change was made OUTSIDE the runner,
and its `accepted` record is the live (changed) identity.

**Wind** (`WindEvent`):
1. `resolve_event` gives the native edges from the accepted native time.
2. Preflight:
   - an exact upper bound `ceil(duration / outer_timestep_s) + 1` on native accounting steps is checked against
     `max_wind_accounting_steps`.
   - Only then is the native `accounting_step_durations_s(duration, outer_timestep_s)` plan built. It has at most
     `upper` floats. Its exact (rational) sum must equal the requested duration within
     `2·2⁻⁵³·max(requested, covered)`, which is the rounding of the split's own two operations (`n·outer` and
     `duration − n·outer`).
   - A dropped or extended interval is refused as `refused` / `unsupported_wind_duration` before any model work. This
     covers an empty plan for a positive duration, e.g. 1e-12 s, and a sub-epsilon tail dropped by native, e.g.
     2.0000000001 s at 1 s. Native time and physics are never rewritten, there is no absolute time floor, and there is
     no copied wind stepping.
   - Work bytes include four bed-sized stages/outputs, three pool-sized allocations, a bounded native deposition request and the native accounting plan; checked before transport against `max_work_bytes`.
3. `bed.stage_copy()`, a deep copy of the retained empty pool, then `bed.stage_scheduler_state(stage, pool, time_s)`.
   This is a fresh native factory over the stage: block 0 and no stale equilibrium, trajectories or caches.
4. `WindBlock(shear_velocity_m_s, direction_deg, duration_s, start_time_s=accepted t, end_time_s=resolved end)`,
   called by keyword, then `advance_wind_block`.
5. Required before acceptance:
   - the `baseline_direct` path, with `conservation_ok`;
   - zero water-borne accounting;
   - the native step count within the bound;
   - native physical time strictly advanced (`wind_no_elapsed` otherwise; `time_reversed` if it went back);
   - after the native block budget passes, the native consumed-block cursor maps to the completed one-block record;
   - `finalize_wind_event(completed_state, config, local_record, record_origin_s=accepted_start)` must complete;
     exceptions, incomplete status, open native budgets, changed clock/cursor or any remaining airborne material stop
     before water. This also validates already-empty events. Pauses and intermediate blocks never call this adapter;
   - the per-class native identity `bed_change + Δmobile − input + export` within
     `summation_error_bound_kg(max(n_steps, 1), reservoir_arithmetic_scale_kg(...))`;
   - the same closure on actual `bed_inventory` values, within the native bound plus the summation bound of the actual
     inventory reductions.
6. Adoption uses the finalizer's returned physical bed/availability/ledger directly with
   `adopt_scheduler_physical(physical)`. The returned empty pool is retained. Finalizer deposition is internal and
   is never applied again; finalizer boundary export is added once to transport export. Both native class-budget
   bounds enter event/aggregate closure. The receipt preserves the local record/hash, absolute origin, unchanged
   native end clock/cursor, deposited/export vectors and native transfer/reservoir identities.
   See [wind finalization](wind_finalization.md).

**Rain** (`RainEvent`):
1. Preflight:
   - a lower bound on storm steps to reach the rainfall end plus the native `hold_s` at `max_dt_s`, checked against
     native `max_steps`;
   - a report-row bound against `max_report_rows`;
   - work bytes (stage, committed candidate and dry bed, storm/soil/velocity grids, two report tables, plus the exact
     bytes of the retained template memory copied in step 5).
2. `RainfallSchedule(edges_s=resolved native edges, intensity_mm_per_h=intervals, provenance=RainfallProvenance(kind="constant"))`.
   Piecewise rain is supported; the original values are passed unchanged.
3. Routing:
   - On the first rain, the graph is built natively with `routing_graph_from_elevation(terrain, committed elevation
     of the stage)`, and the seed comes from `initial_event_state(..., t_s=t, control)`.
   - On later rains, the held graph, terrain, network, grid, routing source and template come from the previous
     `DryHandoff.dry_state`.
4. `commit_bed(stage bed, ctx, held graph, held terrain, t, force=True, routing_source_m=held source)`. Only the
   returned `rerouted` flag decides:
   - if True, `transport_network` and `physics_grid_from_graph` are rebuilt;
   - if False, the held graph, source, network and grid are reused by identity (`routing_identity_changed` otherwise).
5. The entry is `replace(template, bed, graph, network, grid, terrain, routing_source, phase=None,
   storm=initial_state(graph, bed water depth, retained soil, t_s=t))`. The template's transport memory must already
   be zero.
   - On later rains, every other array field of the retained template is passed as an independent copy with the
     same values, so native reset semantics are kept. This includes `sediment_velocity_m_s`, which must be a host
     array (`transport_memory_unsupported` otherwise).
   - Writes to this work transport memory cannot reach the accepted template's memory. The native solver reads the
     retained graph, network, grid and terrain; these are shared, and native rerouting constructs a new routing pair.
     Arbitrary external writes to their arrays are outside this staging guarantee.
6. `complete_event(..., max_end_s=recession guard end, control, policy, report_every_s, max_report_rows)`.
   - An `IncompleteEventError` gives `incomplete` / `rain_incomplete_native_limit`.
   - Any non-`DryHandoff` return gives `rain_not_dry`. Wet progress is kept as evidence only, with no combined resume.
7. Required before acceptance:
   - native `sediment_closure` closed with the request reconciled;
   - the storm water budget closed, and the combined storm plus dry-reset water residual within its native tolerance;
   - the dry state has zero water-borne load, zero depth, zero soil, phase None and zero transport memory;
   - the terrain is unchanged;
   - the dry time lies within [rainfall end, guard end];
   - the actual inventory closure `inventory_after − inventory_before + export` is within the native closure
     tolerance plus the terminal tolerance plus the reduction bound.
8. Adoption is `adopt_bed_state(dry_state.bed, time_s=dry_state.t_s)`. The dry graph, source, network, grid and
   soil (zero after the native reset) are carried forward. The original case soil and bed are never reloaded.

## Budgets and receipts

Each accepted receipt (schema `masy.dense_runner_event.v1`) contains:
- the resolved event (intended input record, native edges);
- the `ClockLedger` entry;
- bed identities before and after;
- native counts and paths;
- per-class vectors:
  - wind: bed change, boundary input and export, mobile before and after, detachment, native residual and bound,
    inventories;
  - rain: actual pickup, actual deposition, export, imports (zero), **internal** terminal deposition, flow-end mobile
    before settlement, native closure residual and tolerance, inventories;
- the water budget: rain, surface and soil initial and final, drainage, storm export, residual and tolerance, and the
  explicit dry-reset surface and soil removal;
- the aggregate: the joint per-class residual
  `inventory − initial inventory + Σexport − Σimport`, checked against
  `Σ per-event native bounds + summation_error_bound_kg(2·inventory elements + 2·accepted events, scale)`.
  The term count scales with the actual events. There is no fixed count, no tolerance floor and no correction.

Receipts contain no state histories: only scalars, class vectors and hashes. The latest state is held only by the bed
and the runner's accepted references.

## Clocks

- Native floats are recorded unchanged with `ClockLedger` (`native_start` = accepted time, `native_end` = actual
  native end, commit stamps before and after).
- Wind stores `advanced_duration_s` as `returned_elapsed`, so its exact discrepancy from the endpoint difference is
  recorded. Block start, end and duration metadata are kept beside `physical.time_s`.
- The wind receipt's `coverage` records the native plan's step count and exact sum, its discrepancy from the request,
  the rounding bound, and the exact elapsed discrepancy. Legitimate FP accumulation (e.g. 0.3 s at a 0.1 s cadence)
  is accepted with the native endpoint unchanged.
- Rain intervals end at the native dry time, after physical recession. That end is never predicted or rewritten
  to the forcing edges.
- The reference control's 0 → 2 → 169 → 171 s is a property of that control on the frozen pair. It is not used by
  the code.

## Transaction and failure guarantees

All model work is staged. Before the single adoption the runner prepares, in order:
1. the complete candidate totals as a new mapping: export, import and bound, internal terminal deposition, water
   (including dry-reset removal), and reroute counters. The accepted mapping is never modified in place.
2. the aggregate budget;
3. a candidate ledger (round trip through `ClockLedger.from_record`, then `append`);
4. the candidate bed identity;
5. the receipt, which must serialize as canonical JSON within `max_receipt_bytes`;
6. the new receipt list, the detached returned receipt, the next cursor and count, the candidate routing namespace,
   soil and pool references, and the candidate pool / soil identity;
7. rechecks for changes made during the call: `bed.verify_case_inputs()`, the live digest of every controlling runner
   input, the accepted bed identity equal to the stage's parent identity, and the accepted pool / soil identity. A
   refusal here names the change as external (`guarantee`).

The adoption itself validates fully before its one replacement. After it, the runner only assigns prepared references
to attributes. There is no computation, allocation, copying or validation after adoption.

On any refusal, native exception, limit or incomplete outcome:
- the step returns a stop receipt with status `refused` / `incomplete` / `native_refused` / `native_error` /
  `runner_error`, a code,
  a truncated detail, the exception type and the native start;
- `unaccepted_elapsed_s` is null when unknown;
- the bed, clock ledger, counters, routing, soil, pool and cursor are unchanged by the runner (an external change
  detected by a recheck is reported as such, not as an unchanged boundary);
- the runner halts (`runner_halted`) and no next event runs.

## Hashes

- `static_record` covers the profile, case digest, schedule digest, `case_schedule_digest` and limits.
- `controlling_inputs_sha256` covers the canonical wind config, terrain, column, rainfall field, vegetation fraction,
  sediment parameters, sediment control, completion policy, report cadence and schedule record.
- The array namespace must be the actual `numpy` module, recorded as `{"array_namespace": "numpy", "numpy_version"}`.
  Other modules refuse, and unknown leaf types refuse.
- The changing graph, routing source, soil and pool are evolving state. They are not static inputs; only their
  initial values are recorded.

## Private host array and ownership operations

- `_HostArrayOps` and `_HostOwnership` are the private host bodies of the engine's array-op and ownership hooks.
  They are not exported, and the public `DenseRunner` constructor, default profile and passive checkpoint
  behavior always use them; nothing public selects or injects another body. Each method body is the former inline
  host expression, so values, shapes and reduction order are unchanged.
- Native class and accounting vectors cross one explicit class-vector seam (`class_vector`). The receipt
  serialization of raw native vectors in the wind and rain events uses `float_values`, which keeps the existing
  host `_floats` result. A native scalar still becomes a one-element list. Native spatial sums stay native and are
  converted only afterwards. The native `flow_end_settlement` list is already a plain host list and is copied
  unchanged. Compact class-vector arithmetic on the host stays in the engine.
- Publication records are explicit. A wind event prepares the full native `SchedulerPhysicalState`
  (`_SchedulerPublication`). A rain event prepares the native `BedState` and its native time (`_BedPublication`).
  Every `DenseBed` validation still runs at commit, in the same late adoption position as before.
- The only other bodies belong to the private explicit factory `masy.dense_device_runner._create_dense_device_runner`
  ([device runner](dense_device_runner.md)). It builds the SAME engine (same `step` / wind / rain / finish code)
  with private device ownership, compact array-op and per-rain `CudaHydraulics` hook bodies, including the three
  retained-routing calls. `dense_runner` does not import it, and it is not the default. It is an unqualified
  prototype: device construction, events, peak memory fit and restart are not qualified, and nothing here makes
  or implies a GPU-ready claim.
- CPU evolving vegetation (an explicit `vegetation_plan` over a dynamic bed; see the
  [vegetation plan](vegetation_information_plan.md)) runs only through this CPU engine. The
  device factory refuses a dynamic-vegetation bed (`unsupported_profile`) before any native, CuPy or copy work.

## Limitations

- MAPLE must provide the explicit `record_origin_s` finalizer API. Construction refuses older APIs with
  `unsupported_native_finalizer`; no clock reset or forcing prefix substitutes for the extension.
- Only the exact [qualified committed pair](source_pair_20261019.md) is covered. A newer upstream revision
  requires fresh qualification; existing locks are never rebound. Shared-storage libraries do not qualify windowed wind.
- The work-byte cap bounds MASY-side staged copies with a documented estimate. It does not bound native internal
  temporaries.
- In-process only: the runner itself has no checkpoint, persistence or combined restart, and it never resumes wet
  progress. The separate opt-in [boundary store](dense_boundary_store.md) persists accepted boundaries only; it
  has no active-event resume.
- The duration-coverage guard checks the native split plan. Actual native accumulation is preserved and recorded
  (`coverage.elapsed_discrepancy_s`); the finalizer additionally checks the mapped completed-record endpoint using
  its unchanged native clock tolerance. Support for sub-epsilon durations or changed tolerances belongs upstream.
- The pre-event external-change check covers the bed identity and the pool / soil host arrays. Non-array pool fields
  and the private retained routing template are not digested; the template is never exposed, and its memory is
  work-copied.
- Only the profile above. The mobile-state flat pool and the `upwind`/`array` water path are the qualified-first
  combination; everything else refuses.

## Completion-clock and diagnostic limits

The native end-clock tolerance remains unchanged. At sufficiently large absolute origins or after long floating-point
step accumulation, native completion may refuse even a physically completed block. This fails before publication;
changing the native tolerance requires separate native work. No clock is rebased to evade it.

`active_evidence` is local native diagnostic state, not a serialized restart schema. Before finalization a wind stop
holds its native `WindBlockResult`; during/after finalization it holds a mapping containing `transport` and, once
returned, `finalization`. Inspect the stop code before consuming that evidence. A finalizer exception is reported as
`native_error`; either native error or refusal blocks the next process and retains the last accepted boundary.

In wind class receipts, `native_residual_kg` / `native_bound_kg` describe the transport block. The combined
`bed_change_kg`, final `mobile_after_kg` and total `boundary_export_kg` use `event_residual_kg` / `event_bound_kg`;
original transport bed/export/mobile values remain separately named. Aggregate internal terminal deposition includes
both native wind settlement and SYRUP's terminal deposition, each counted once as internal accounting.
