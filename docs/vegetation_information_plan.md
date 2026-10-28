# Shared Landscape Vegetation: Information Plan (MASY-09)

**Status:** This began as a design only. Accepted CPU components now exist; see [current status](#current-accepted-cpu-component-status-october-19). The original design text below is kept as written.
- **Phase 1 publication** means writing this file to `docs/vegetation_information_plan.md` in the MASY working tree. The user subsequently authorized commit/push of accepted milestones; this design may be published with its scope intact.
- **Out of scope for this task:** code, data, response laws and soil-water policy.
- **Parallel work:** it runs alongside the SYRUP candidate review and commit and the new Dense pair qualification, and blocks neither.

## 1. Current status (from sources)

**MAPLE `core/types/vegetation.py`**
- `VegetationState(source ∈ {none, raster, objects})` holds one canonical `(ny, nx)` FP64 raster:
  - `height_m`: cover-weighted, in m. It is >0 exactly where cover >0.
  - `cover_fraction`: in [0,1].
  - `porosity`: flow-through fraction in [0,1], default 0.20.
  - `drag_coefficient` and `frontal_area_density`: native morphology metadata; the current vegetation wind modifier does not use them to introduce new response physics.
- Objects are structure-of-arrays with no plant IDs.
- Rasterization is one-way, uses a host loop, and is documented as "not a hot path".
- `verify_rasterization` is an opt-in bitwise check.
- The module states that vegetation is prescribed, affects wind stress only, and is in no ledger.

**MAPLE `aeolian/scheduler/vegetation_cache.py`**
- The cache key includes the full `vegetation_content_hash`, recomputed on every resolve.
- Topographic commits do not invalidate the cache, and the cache is never persisted.
- A cache hit cross-checks the modifier's retained raster copies against the current raster.

**MASY `dense_bed.py`**
- Vegetation is held by reference, outside `SHARED_FIELDS`, and is part of `case_record`.
- Two digests are re-verified at every transition.
- Replacing the vegetation object is refused (`prescribed_field_replaced`).
- Only host NumPy arrays are accepted.

**MASY `dense_runner.py`**
- The water path's cover must be bitwise `bed.vegetation_cover_fraction()`, and that cover is a controlling static input.
- A rain event is accepted only as a `DryHandoff` with:
  - zero surface depth, soil water and mobile load;
  - phase `None`;
  - zero sediment velocity.
- Accepted soil water (`self._soil`) is held by the runner and carried unchanged through wind events.
- Each wind event builds a fresh scheduler, so the wind cache is rebuilt on every re-entry.
- `ClockLedger` records the native endpoints and commit stamps.

**SYRUP**
- `dry_raindrop_pickup_demand` takes a cover fraction and converts it to percent internally.
- `PondEvent` makes a host copy of the cover and folds it into `config_sha256`.
- `pond_checkpoint` binds vegetation as static caller configuration.
- These sources do not establish whether `complete_event` copies the cover.

**Qualification**
- Current Dense qualification exercises nonempty transport with baseline/prescribed vegetation.
- It does **not** qualify evolving vegetation, full vegetation-physics coverage, or full production wind settings.

## 2. Authority (proposed)

There is one logical authority per landscape: `LandscapeVegetation`. It is a MASY envelope (`src/masy/landscape_vegetation.py`) that owns exactly one native MAPLE `VegetationState`. The native container stays the only physical-field container. The envelope adds identity and transaction metadata, never a second field.

**Write protection**
- Arrays the envelope publishes are envelope-owned and read-only (`writeable=False`).
- Proposers receive views, never write access.

**Object-or-raster authority**
- Dual authorities are refused, as `validate_vegetation_state` already does.
- First-milestone dynamic support covers:
  - `source="raster"`;
  - `source="none"`, through a declared transition to raster.
- Dynamic updates of `source="objects"` are refused until stable plant IDs exist. An array index is never treated as plant identity.
- Converting objects to raster is a declared, irreversible, versioned delta and never happens implicitly.
- Editing the raster while objects are retained is always refused.

**Canonical form**
- An authored height is converted to cover-weighted form exactly once, as `uniform_vegetation_raster` does.
- A delta that reaches cover 0 sets `height_m` and `frontal_area_density` to exactly 0. It sets porosity and drag to the declared rasterizer defaults, which preserves the native invariant.

**Masks:** validate whatever mask semantics the configured profile defines (currently full mask only). No inactive-cell value rule is invented.

**Water reads**
- Water reads cover derived from the accepted version.
- Adapters check fraction semantics (FP64, shape, [0,1]). SYRUP keeps its own percent conversion.
- Because `PondEvent` copies today, the contract is a bound, immutable, per-event native view or copy, carried with its content hash and version. Zero-copy is not claimed.
- Future same-backend native APIs should accept references so the cover is not copied repeatedly.

**Optional layers (conditional follow-ups)**
- Candidate layers are functional type, live/dead standing, and litter.
- Each would be a separate native container with explicit overlap, height and porosity aggregation semantics and provenance. Covers are never summed blindly.
- A request for an unsupported layer or physics is refused with a declared code, not silently ignored.
- Feeding drag or frontal metadata into kernels requires a separate physics specification.

## 3. Identity and integrity

**Static profile (default)**
- All current inputs and digests stay bit-identical: `case_record`, the prescribed digests, the cover in `_static_parts_record`, and `PondEvent.config_sha256`.
- Existing checkpoints remain valid.

**Dynamic profile (explicitly selected)**
- It uses new schemas, e.g. `masy.dense_case.v2` and `masy.landscape_vegetation.v1`. Migration is explicit and qualified, never silent.
- The case record binds the version-0 content.
- Version identity = sha256(schema, parent identity, accepted delta receipt digest, native `vegetation_content_hash` of the result).
- A version number is a label, not proof. No caller token bypasses the native validators or the `validate_vegetation_wind_cache` cross-checks.

**Milestone verification**
- Run the full native validator and content hash at each accepted update. That is once per boundary, not per step.
- Run the same full checks at checkpoint save and load.

**Conditional follow-up, gated on measured cost**
- Per-tile digests under an authenticated root. At an update, validate the touched tiles plus the parent root; at checkpoint, run full verification.
- Any change to the native cache-key API is upstream MAPLE work.
- Benchmark GPU hashing before choosing its algorithm.

## 4. Transactions

**Delta contents (`VegetationDelta`)**

| Field | Content |
|---|---|
| `schema` | Delta schema version |
| `proposer` | wind, water, or prescribed |
| Sequence | Per-proposer sequence number |
| Base | Base version identity and base content hash |
| Interval | The exact accepted interval (see below) |
| Changes | Affected cells and new canonical values, increases or decreases |
| Water exchange | Optional (§5) |
| `reason` | Reason code |
| Payload digest | Digest of the delta payload |

- The accepted interval is identified by:
  - the `ClockLedger` entry index;
  - the native start and end as float hex;
  - the commit stamps;
  - any recorded offsets or anchors.
- There is no single global float, and native time is never changed.
- The delta ID hashes every field above, so two distinct updates proposed at one boundary cannot collide.

**Timing**
- Vegetation is constant within a native event.
- A delta derived from event *k*'s accepted outputs is prepared inside event *k*'s `_finish` and published with it.
- Prescribed deltas bind to a schedule boundary index.
- Both kinds take effect from event *k+1*, so changes land between wind and water events.

**Validation**
- The candidate is built by copy-on-write of the changed arrays, then checked with the native validators and bounds.
- Any failure stops the event and publishes nothing. The delta is kept only as evidence.

**Atomic publication**
- The bed, vegetation version, soil-water reference, clock ledger, receipts and cursor are all prepared, then published in one adoption.
- `DenseBed` gains a dynamic-profile adopt method that accepts a vegetation candidate.

**Exactly once**
- Exactly-once application comes from retained atomic receipts plus the accepted cursor (next index and last sequence per proposer). The version chain alone is not sufficient.
- A delta ID already present in the receipts is recognized as published and is not reapplied (idempotent).
- A delta whose base does not match the current version is refused.
- A pause or pond checkpoint mid-event completes no ecological event; accumulators persist in the checkpoint.

**Reserved schema slots:** the envelope reserves `process_state` (per-process native accumulators, versioned by their owner) and `update_cursor`.

## 5. Water state and resource exchange

**Default historical profile**
- The current qualified dry-again reset is unchanged, and its `DryHandoff`/reset tests are preserved exactly.
- Soil water is zero at the current water-event acceptance boundary; any requested debit exceeding the actual stored water is refused as unavailable. No credit can silently restore reset water.
- Removed water is never re-created.

**Future retained-subsurface handoff (SYRUP-native, separately versioned, explicitly selected)**
- It is accepted when water-borne surface transport and cohorts are genuinely complete: zero mobile load, phase `None`, zero velocity, and surface state satisfying a SYRUP readiness policy that is still to be specified.
- Subsurface soil water is retained.
- SYRUP owns the soil-water fields, units, layer and storage definitions, water budget, and readiness.
- MASY carries only the accepted reference. It never holds a second copy and never retains water after an existing native reset.

**Exchange**
- A delta's optional `water_exchange` gives per-cell debits and credits against SYRUP soil water, per layer once SYRUP defines layers.
- A SYRUP native validator books them to named budget channels:
  - `vegetation_uptake`, a sink that applies even though biomass storage is deferred;
  - any other transfer or source, each with explicit provenance.
- Insufficient storage or an out-of-bounds value refuses the whole transaction; values are never clipped.
- An update may be a water no-op.

**Wind phase:** wind has zero water transport. It carries soil water unchanged, as the runner already does. No evapotranspiration or wetness coupling is invented.

**Exposure summaries (opt-in)**
- *Water:* a SYRUP accumulator captures exposure before the reset.
  - Values are weighted by actual accepted dt; rejected attempts are excluded.
  - The response contract selects and caps the fields, for example wet duration, ∫depth dt, and ∫soil-water dt.
  - Reset amounts are labelled as removal, not as available water.
- *Wind:* summaries use actual native outputs in native units, such as accounting detachment and committed-topography change. No shear-times-modifier formula is invented.

**Drivers, not laws**
- Either process may drive positive or negative contributions.
- Wind exposure and burial damage are drivers, not automatic mortality.
- Burial or erosion relative to plants needs a reference elevation, plus root depth only if a specified response requires it.

## 6. Caches, routing, memory, restart, backend

**Caches and routing**
- A content change alters the native hash, so the wind cache invalidates automatically (current behavior).
- Water coefficients derived from cover are keyed by version identity with provenance, and are refreshed separately from the routing graph.
- A vegetation-only change does not trigger a reroute. `commit_bed` keeps its native net-displacement and geometry rule.

**Memory**
- The raster costs 40 B/cell; each selected summary field costs 8 B/cell.
- A candidate copies only the changed arrays (tiles in the follow-up).
- Caps are checked before allocation.
- Add no per-step full-state history or scans in the new adapter. Native wind-cache content hashing remains an existing per-resolve cost until a measured, independently qualified upstream replacement exists.

**Restart**
- The dynamic profile persists, beside the MAPLE snapshot under one combined manifest:
  - the envelope record;
  - the current payload;
  - receipts and cursor;
  - process state.
- `pond_checkpoint` gains a vegetation version binding, distinct from static configuration, under a new schema.
- Same-backend restarts are exact wherever the native contract supports it.
- CPU and GPU results agree within native tolerances; universal bit equality is not required.

**GPU**
- The intended design is device-resident native arrays with lazy CPU views.
- Native raster validators already accept CuPy. Current raster construction/rasterization uses host paths, and content hashing canonicalizes device arrays through counted host transfers. MASY currently refuses non-NumPy arrays. Device validation alone does not qualify resident GPU vegetation exchange or its performance.

## 7. Staged plan

| Phase | Scope | Owner / files |
|---|---|---|
| 1 (design accepted) | Publish this plan and its architecture status; executable update hooks remain a later task | MASY docs |
| 2 | Envelope, delta, receipts, cursor; dynamic profile; atomic adopt; prescribed schedule entries; raster only | MASY `landscape_vegetation.py`, `dense_bed.py`, `dense_runner.py`, `dense_schedule.py` |
| 3 | Per-event bound cover view with version and hash; checkpoint version binding; opt-in pre-reset exposure accumulator | SYRUP `complete_event`, `pond_event.py`, `pond_checkpoint.py` |
| **Milestone** | Prescribed ± deltas consumed end to end by native wind and water; interface verification only, no science claim | MASY + tests |
| 4 (conditional) | Retained-subsurface handoff and water-exchange validator, after the physics answers in §9 | SYRUP; MASY carries references |
| 5 (conditional) | Stable plant IDs, tile rasterizer and digests, optional layers, measured cache-hash API | MAPLE `core/types/vegetation.py`, `vegetation_cache.py` |
| 6 | Response laws in separate specifications | Physics owners |

## 8. Acceptance tests

1. **Shared read:** a nonzero heterogeneous raster is read by wind (modifier fields) and water (values plus version and hash). Object identity is asserted only where an API promises reference transfer.
2. **Ordering:** prescribed positive and negative deltas from each proposer change the *following* process only.
3. **Rollback:** an injected native failure or invalid delta leaves the bed, version, soil water, clock, receipts and cursor unchanged.
4. **Restart:** an interrupted run matches an uninterrupted one on the same backend. A duplicate delta is recognized and not reapplied; a base mismatch is refused.
5. **No default regression:** static-profile receipts, digests and checkpoints are bitwise unchanged, and the reset tests are untouched.
6. **Objects/raster consistency:**
   - a raster edit on an objects-sourced state is refused;
   - objects-sourced dynamic updates are refused until IDs exist;
   - the bare-cell invariant holds at cover 0.
7. **No reroute:** a vegetation-only delta gives `rerouted False`.
8. **Water exchange (phase 4):**
   - an over-debit is refused;
   - a debit under the reset profile is refused;
   - the budget closes with a named sink.
9. **Memory:** CPU peak and candidate bytes stay within caps. GPU is tested only once a backend exists.

## 9. Decisions and limits

**Physics questions to ask the user before implementing any law or soil-water policy:**
- Which summaries drive growth, mortality or damage, with what laws and coefficients?
- What retained-handoff readiness is required (surface water state), and how are soil layers and storage defined?
- Are there uptake limits beyond storage, and is evapotranspiration needed?
- Do dead standing material or litter affect wind or rain?
- How are the reference elevation and root depth defined?
- Should drag or frontal metadata enter the kernels?

**Routine, authorized implementation choices:** schemas, delta format, caps, digest layout, tile size, checkpoint layout.

**Limits:**
- No sub-event vegetation change.
- No calendar ticks.
- No standalone growth model.
- Dense host profile only, until the backend work lands.

---

### Review risks (source-specific)

- **Per-resolve hashing.** `compute_vegetation_wind_cache_key` hashes the full vegetation state on every resolve. Measure its cost before proposing any upstream change, and keep the raster cross-checks in `validate_vegetation_wind_cache`.
- **Static digests.** `_static_parts_record` includes `vegetation_cover_fraction`, so a dynamic profile needs a new schema rather than changes to the current digests.
- **PondEvent config hash.** `PondEvent.config_sha256` includes `veg`, so the version binding must be separated from static configuration, or every update invalidates pond checkpoints.
- **Pre-reset exposure.** The data exists only inside SYRUP. Once the `DryHandoff` is returned, depth and soil water are already zero.
- **Tile bit-identity.** A tile re-rasterization must reproduce the `np.add.at` order and the cap rescale exactly.
- **Snapshot coverage.** These sources do not show whether MAPLE `save_state_snapshot` stores vegetation, so combined-restart coverage must be confirmed.

## Isolated dynamic dense CPU interface prototype (task51 StageB)

This opt-in prototype layers the accepted vegetation component onto the explicit
ec2 + task43 source-manifest prototype with native MAPLE89/SYRUPd05 from immutable
lock851. It is not a current committed runtime or production calibration. The
original static runner keeps its 26 fields, schemas, input checks and artifacts.

Author a `VegetationPlan` before bed construction with the exact schedule digest,
event kinds, grid, class count and voxel count. Pass that same object as
`build_dense_bed(..., vegetation_profile=plan)` and
`DenseRunner(..., vegetation_plan=plan)`. Omitting both selects the unchanged
static profile; selecting only one refuses. Dynamic case, bed identity, static
record, receipts and report use explicit v2 schemas. Updates run after accepted
native event completion and before the next event starts. The binding contains
the complete native `ClockLedger` record, including returned elapsed time and any
realized clock discrepancy. Native rain schedule kind `rain` maps explicitly to
the prescribed `water` proposer. Empty boundaries still advance their binding
cursor, and repeated/conflicting updates use the component's exactly-once rules.

All five native canonical raster fields are retained in one versioned envelope.
Native wind receives its private read-only authority by reference through the
actual scheduler factory. The actual SYRUP completion API accepts only cover
fraction, so it receives an owned current cover array, bound in each receipt to
the shared version/native-content identity. The other four fields remain in the
same shared envelope; this prototype does not add an unimplemented SYRUP API.
Public `DenseBed.vegetation` and cover snapshots are independent caller-owned
copies in this profile. Its original prescribed vegetation is retained separately
and checked against the original case digest. Wind admission retains every native
physical/mobile/threshold/reconciliation/clock guard.

The joint-resource candidate carries the runner's actual event-candidate soil
object unchanged. Any named water exchange refuses, even a zero-valued channel.
The existing native dry reset still runs before the vegetation hook, and no water
or soil reset is replayed or credited by a vegetation update. New growth,
mortality, uptake, evapotranspiration or retention laws are absent.

Publication first prepares vegetation, complete native bed adoption, aggregate
budgets, full clocks, soil/pool/routing references, canonical receipts and all
accepted identities. It then checks the prepared native/source lineage and makes
only reference assignments. A late ordinary exception leaves the last accepted
bed, vegetation, soil, clocks, routing and totals unchanged. An exception from a
native call may expose no partial result; the stop receipt makes no invented
completion, elapsed-time or recovery claim. This is ordinary exception atomicity,
not protection against process death and not active-event restart.

The component's finite logical array/metadata/copy bounds apply before its
allocations. Runner-required constructor cover, production diagnostic snapshots,
water covers and update copies are charged to finite event/run copy limits;
additional component/retained-owner/metadata reservations also enter the existing
`max_work_bytes` check before event staging. Public user-requested snapshots are
caller-owned and are outside accepted runner traffic. These logical reservations
are not a whole-native peak estimator. Qualification uses a dedicated CPU worker
with enforced 4GiB address space, CPU 120/135s and 128MiB file ceiling, a 180s wall
limit, bounded streams and fresh host/disk admission. Native FFT, physics and
conservation tolerances are unchanged.

The clearly labelled four-event synthetic fixture uses the existing planar
100um test configuration, topographic wind/separation, creep and native configured
avalanching with periodic wind/open water boundaries. It prescribes a positive
update after the first wind and a negative update after the first rain, and checks
both following models' actual inputs. It is not measured calibration or evidence
for a scientific vegetation response. The existing production-profile foundation
still refuses enabled vegetation wake; separate task63 must qualify an explicit
synthetic-only admission and actual native modifier witness before claiming an
applied following-wind response. No disabled-wake result is labelled such evidence.

The original task43 passive codec refuses the dynamic v0 layout before writing.
Dynamic boundary/active-event restart, scalable/windowed storage and GPU events
remain separate work. No wrapper reroute decision or new vegetation trigger is
introduced: any hydraulic reroute remains the existing native physical-surface
rule with its actual configured threshold.


## Task70 separate authored native wake response qualifier

A fresh isolated full StageB prototype with the exact accepted task63 synthetic
factory/adapter overlay has a separately named response qualifier. It retains
all original/default fixtures and native source. Authored cover-weighted plant
heights meet the unchanged native periodic-support guard; only the native wake
master enable differs from its default spec. Wake/windward coefficients are not
recalibrated. Native sampling ladders include zero distance. Actual wind observes barev0
(no sampling) andv2 with3wake samples(0,0.125,0.25m) and1windward sample(0m).
The authoredv1 height would yield5wake/1windward samples but is observed by
water cover alone, so it establishes no wind response. The native windward
witness covers only the occupied-cell zero-distance response, with no off-cell
windward support qualification. Both native kernel fields and their combined
modifier must match the direct native result.

The qualifier compares all five current landscape fields/version at wind entry,
the actual reduced transport modifier against direct native output, native cache
key/integrity/reuse, and owned current cover at water entry. Native SYRUP currently
uses cover alone; the other fields remain in the shared envelope. It also checks
actual native soil reset/candidate identity, full clocks/accounting, late failure
atomicity and exact copy admission. Source alone proves no response: an accepted
actual bounded run and independent review are required. No growth/mortality/uptake
law, tolerance, production calibration, GPU/windowed/scale or dynamic restart
qualification is inferred. The separate accepted common-engine prototype still
requires an explicitly coherent integration.

## Current accepted CPU component status (October 19)

This status supersedes the pending notes in the two sections above.

**What is accepted.** On the coherent shared CPU engine, typed versioned updates
of all five native fields (StageA/B, task 51) are accepted. These are manual
boundary updates. The native wake response (tasks 63/70) is also accepted: the
direct CPU wind response uses the latest accepted cover. In the source fixture,
wind sees v0 and then v2, and water sees v1 and then v2. The wind response has
three native wake samples and one windward zero-distance sample. Source-only
defaults are unchanged.

**Soil water.** The SYRUP-owned soil object passes unchanged after the current
native explicit dry reset, so soil is zero at each accepted water boundary. A
typed future joint soil/vegetation exchange seam exists. It refuses any nonempty
exchange until the exchange is specified scientifically.

**Not included.** There is no growth, mortality, uptake or ET law, and no
retained-soil policy. Dynamic-vegetation persisted restart, GPU and windowed
runs are refused. The [boundary store](dense_boundary_store.md) persists only
the static profile.

## Interim ponded-water accounting (October 28, 2026)

The user's interim choice is to remove and account for remaining ponded water
before wind, pending a future ET-capable model. Water events retain Native
pit ponding. The explicitly selected profile uses existing
`terminal_storage_water=True` and `deposit_mobile_at_flow_end=True`: depth and
surface-volume completion checks apply to moving cells, excluding recognized
terminal cells. Rain-off, discharge, hold, finite/nonnegative validation, real
Native final sediment settlement and conservation still apply.

Native finalization externally removes remaining surface water, including its
terminal-storage share, and performs the existing soil reset before returning
an accepted `DryHandoff`. MASY forwards the returned bed and records Native
accounting without reapplying either reset. Surface and soil removal are separate
accounted totals; dry surface,
soil and mobile state remain zero before wind. This is not ET, plant uptake or
water retained for vegetation. An ET-capable or retained-soil-water profile
needs explicit scientific specification and qualification; no automatic ET
route or growth response is added to the vegetation plan.

SYRUP `919c82c` and MASY `5cab827` are Source-accepted and pushed on
qualification branches. At this qualification stage, the C64 water event with positive ponded-water
removal matched its independent Dense GPU oracle, the direct Native entry
comparison passed, and the full C64 four-event GPU wind/water sequence matched direct Native
calls in 406.51 seconds under the original limits. This changes no ordinary
case defaults and grants no general arbitrary-landscape GPU/windowed support
or production calibration. The separate failure-guard check also passed;
persisted restart is not covered by this C64 result.
Earlier dated status above remains historical.
