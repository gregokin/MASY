# Dense integration, chunk 01: schedule and complete-bed adapter

Status: **schedule and Dense bed mappings independently accepted, October 18, 2026.** This document covers those two
modules. The separate [event runner](dense_runner.md) has passed actual wind → rain/recession → wind checks on the
frozen CPU pair. The bed/bootstrap qualification on this page precedes the [reviewed finalizer integration](wind_finalization.md). There is no production transport CLI or combined restart; the existing
preflight/runtime tools still report `transport_ready=false`.

Modules: `masy.dense_schedule` (data only) and `masy.dense_bed` (thin native adapter). Importing either one imports
neither numpy nor a native model. `dense_bed` imports native code lazily, inside each call.

## 1. Schedule (`masy.dense_schedule`)

Schema `masy.dense_schedule.v1`. The input is an ordered list of events:

```json
{"schema": "masy.dense_schedule.v1",
 "events": [
   {"kind": "wind", "duration_s": 600.0, "shear_velocity_m_s": 0.45, "direction_deg_cw_from_north": 270.0},
   {"kind": "rain", "recession_guard_s": 7200.0,
    "intervals": [{"duration_s": 900.0, "intensity_mm_h": 30.0}, {"duration_s": 600.0, "intensity_mm_h": 5.0}]},
   {"kind": "wind", "duration_s": 300.0, "shear_velocity_m_s": 0.6, "direction_deg_cw_from_north": 0.0}]}
```

Rules:
- Any order of kinds is valid. Nothing requires alternation.
- A wind event has one constant forcing. Shear velocity is `>= 0` m/s. Direction is in `[0, 360)` degrees clockwise
  from north.
- A rain event has one or more piecewise intervals. Each interval duration is `> 0` and each intensity is `>= 0` mm/h,
  with at least one intensity positive. The rain event also has a finite positive `recession_guard_s`, counted from the
  end of rainfall. If native completion is not reached by the guard, the event is incomplete, never complete.
- Values must be plain JSON `int`/`float` (an int must be exactly representable as binary64). The following are
  refused: bool, NaN, ±inf, `-0.0`, zero or negative durations, strings and numpy scalars.
- Key sets are closed. Any key naming an absolute, calendar or native time is refused with `absolute_time_refused`.
  This includes `native_start_s`, `native_end_s`, `start`, `calendar_start`, epochs and timestamps. All other unknown
  keys are refused with `unknown_key`.
- Every class is a frozen dataclass and validates its own inputs (`WindEvent`, `RainInterval`, `RainEvent`,
  `DenseSchedule`).

Records and digests:
- `as_record()` is lossless. Every value is stored as binary64 hex plus its exact fraction (`masy.clock.encode_time`).
  A rain event also stores its exact `rainfall_duration_s`.
- `from_record()` rebuilds the schedule and requires the canonical JSON to match exactly. A tampered derived field,
  bool-for-int, an extra key or a non-canonical hex is refused.
- `digest` is the SHA-256 of the canonical record. Event order is part of it.
- `case_schedule_digest(case_sha256, schedule)` combines the static case digest (§3) with the schedule. Evolving bed
  state is never part of it.

Clock semantics:
- The input holds intended durations only. `resolve_event(schedule, i, native_start_s)` produces execution evidence for
  one event, starting from its actual native start. For event 0 that is the native epoch. For a later event it is the
  realized native end of the preceding event, exactly as the native code returned it.
- Rain recession length is physical, so later starts are never predicted.
- Edges are exact cumulative offsets: `event_start`, interval ends, `rainfall_end` and `recession_guard_end` for rain;
  `event_start` and `wind_end` for wind.
- Edges are passed as exact decimals to the unmodified `masy.clock.resolve_forcing_edges`. Its rule is
  `float(native_start + float(Fraction(edge)))`, and it records the native-minus-intended offset.
- The intended record and the resolved edges are returned side by side. Nothing is retimed, rescaled or rounded away.
- An edge that collapses natively (for example `2**53 + 0.5`) raises `ClockError("collapsed_forcing_edges")`.
- `check_resolved_event` re-derives a serialized record and refuses any drift, or a record from a different schedule.

## 2. Bed adapter (`masy.dense_bed`)

### Construction (exactly once)

```python
bed = build_dense_bed(
    geometry=GeometrySpec(...), grain_classes=GrainClassSet(...), mass_resolution_kg=1e-10, n_voxels=4,
    initial_bed_elevation_m=z,          # (ny, nx) f64: sediment surface above the voxel-column base (datum 0)
    initial_class_fractions=f,          # (ny, nx, nc) f64: mass composition, native class order, sums to 1
    initial_available_fraction=a,       # (ny, nx, nc) f64 in [0, 1]: explicit initial availability policy
    initial_water_depth_m=h,            # (ny, nx) f64 >= +0: SYRUP-owned depth
    vegetation=VegetationState, threshold_modifier_state=ThresholdModifierState,
    topographic_commit_spec=TopographicCommitSpec(...), avalanching_spec=AvalanchingSpec(...),
    active_mask=np.ones((ny, nx), bool), initial_time_s=T0,
    limits=DenseBedLimits(host_memory_cap_bytes=..., max_voxels_per_column=..., stage_copy_max_bytes=...))
```

The native sequence, using the frozen MAPLE/SYRUP paths:
1. `zeros_voxel_column_state`
2. `deposit_surface_mixture_batch`. A resolved shortfall is refused.
3. `initialize_active_layer_from_voxels`. An underfilled layer is refused.
4. `zeros_sediment_ledger_state`
5. `initial_committed_topography_state_from_physical_state(..., initial_time_s=T0)`
6. `sediment_availability_state_from_active_layer(active, a)`
7. `water_state_from_depth`

The result is packaged as SYRUP's actual `BedState` plus one `BedContext(geometry, grain_classes, mass_resolution,
commit_spec, avalanching_spec, a)`. The caller's own geometry and grain-class objects are reused, never reconstructed.
Every process must take geometry, classes, mass resolution and the fallback fraction from `bed.context`. In particular,
a future wind config must use `bed.context.initial_available_fraction` (the same read-only array).

Availability policy: `a` is used for both the initial available/bound split and the native replenishment fallback.
That is the meaning `SchedulerPhysicsConfig.initial_available_fraction` and SYRUP's `BedContext` already give it.
Full availability is never inferred.

### Preflight, before any domain-sized native allocation

- **Exact native types:** `GeometrySpec`, `GrainClassSet`, `AxisBoundary`, `VegetationState`,
  `ThresholdModifierState`, `TopographicCommitSpec`, `AvalanchingSpec`. Subclasses and duck types refuse. Nothing
  emulates a Windowed or Compact class.
- **Native validators:**
  - `validate_grain_classes`
  - `validate_geometry` (boundaries checked against the class ids)
  - `validate_mass_resolution_kg`
  - `validate_vegetation_state(geometry=...)`
  - `validate_threshold_modifier_state`
  - `validate_available_fraction_array`
- **Scope checks:**
  - backend `"numpy"` only
  - host `numpy.ndarray` leaves only
  - square grid (`dx_m == dy_m`)
  - `voxel_growth == DISABLED_VOXEL_GROWTH`
  - a full bool active mask
- **Arrays:** exact shape and dtype, finite values. Fractions must lie in `[0, 1]` and sum to 1 per cell, within
  `4·nc·eps`. Depth must be `>= +0`.
- **Capacity:**
  - elevation `>=` the active-layer thickness
  - per-cell mass `<= n_voxels · voxel_dz · bulk · dx · dy`
  - `n_voxels <= max_voxels_per_column`
- **Host memory:** `estimate_initialization_bytes` must not exceed `host_memory_cap_bytes`. The estimate is
  `8·(4·cells·nv·nc + 8·cells·nc + 4·cells) + cells·(ledger + committed bytes per cell)`. It is checked in two
  steps, and both run right after the geometry, class, mass-resolution, `n_voxels` and time checks:
  1. A cheap lower bound, the same formula with 0 per-cell native bytes, computed from validated shapes and counts
     only. It runs before the vegetation and threshold validators, the active-mask scan, any owned copy
     (`_owned_array`), any value-scan temporary and the probe. A tiny cap therefore refuses before any of them.
  2. Then the per-cell bytes are measured on a bounded 1×1 empty native probe, built with the same native
     constructors, and the full estimate is checked before any domain-sized copy or native domain allocation.

  Both values are recorded in the receipt (`host_memory_lower_bound_bytes`, `estimated_initialization_bytes`). This
  is a conservative bound, not a measured peak RSS. The cap is explicit; this chunk does not query the machine's
  available memory.

After construction, every shared field passes the actual validators:
- `check_voxel_column_state`
- `check_active_layer_voxel_partition`
- `check_sediment_availability_state`
- `validate_water_state`
- `validate_ledger_state`
- `validate_committed_topography_state`

### Ownership and mappings (always by reference)

| Owner | Fields |
|---|---|
| Shared bed (`bed.state`, one object set) | `voxel_column`, `active_layer`, `water`, `ledger`, `committed_topography`, `sediment_availability` |
| Shared prescribed landscape (held by reference, verified unchanged by digest) | `vegetation`, `threshold_modifier_state` |
| Wind-private (caller-owned, never stored) | mobile cohorts, coupled state, block cursor, derived caches |
| Water-private (caller-owned, never stored) | graph, routing source, terrain reference, storm, velocity/phase, soil water |

**`bed.fresh_scheduler_state(mobile_state, *, time_s)`** builds a genuine native `initial_scheduler_state` over the
shared objects, with explicit `vegetation`, `sediment_availability` and `water` (never None). It passes no `coupled`
state and no semantics keyword. The native factory therefore sets block 0, `last_resolved_wind=None` and every derived
wind cache to None, and derives the current trajectory semantics from the empty content. Conditions:
- `time_s` must equal `bed.time_s` bit for bit.
- `mobile_state` must be an exact `MobileCohortState` and genuinely empty, according to the native
  `carries_coupled_trajectory_history`.
- `water.mobile_mass_by_cell_class_kg` must be exactly zero.
- The prescribed fields must be unchanged.

Standing water depth is not checked here; it is the future runner's decision.

**`bed.stage_scheduler_state(stage, mobile_state, *, time_s)`** (added for the runner, see docs/dense_runner.md) builds
the same genuine native scheduler through the same private factory (`_scheduler_over`), but over the INDEPENDENT
shared objects of `stage.state`, never over the accepted `bed.state`. Before the factory runs it requires:
- the stage was made by this bed instance (`foreign_stage`) and is unconsumed (`stage_consumed`);
- the stage's recorded identity equals the current accepted identity (`stale_stage`: the accepted bed moved on);
- the staged content still reproduces that identity (`stage_mutated`);
- `time_s` equals the stage time and the bed time bit for bit (`time_mismatch`);
- the native shared-field validators, the commit-stamp rule, a genuinely empty pool and a zero staged water-borne load,
  exactly as for `fresh_scheduler_state`.

The stage is not consumed and nothing on the accepted bed changes. `fresh_scheduler_state` keeps its behaviour and
check order; only its tail moved into the shared factory.

**`bed.adopt_scheduler_physical(physical)`** takes the six shared fields back by reference, at `physical.time_s`.
Conditions:
- `physical.vegetation` and `physical.threshold_modifier_state` must be the held objects.
- The wind pool must be empty. Airborne load is never dropped or settled here.
- `ledger_reconciliation_baseline` must be None, because `BedState` has no slot for it and the adapter refuses to drop
  it.

**`bed.adopt_bed_state(state, *, time_s)`** takes an actual SYRUP `BedState` by reference, at an explicit native time.
An example is `DryHandoff.dry_state.bed`. Native exchange legitimately returns updated availability, ledger, water and
so on, and they are adopted as returned.

Every adoption:
- validates first and mutates nothing on refusal
- re-verifies the live static inputs first (see below)
- requires native time to be non-decreasing
- refuses `future_commit` if `committed_topography.last_commit_time_s` is after the adopted native time. This is
  ClockLedger's rule (`commit_after <= native_end`): the stamp must be a native clock value (`invalid_commit_time`
  otherwise), the comparison is exact with no tolerance, and neither the stamp nor the time is rewritten.
  `fresh_scheduler_state` and `restore_stage` apply the same rule, against the bed time and the stage time
- never rebuilds a bed from a DEM, fractions or the original case
- never passes None for water or availability
- never resets availability

`bed.vegetation_cover_fraction()` returns the canonical cover-fraction array, the same object, unconverted. SYRUP's
`sediment_physics_step` converts it to percent internally, so it must not be converted twice.

### Hash scope (§3) and staging

- **Static case** (`case_record` / `case_sha256`, schema `masy.dense_case.v1`). It contains:
  - backend, dtype, representation
  - the canonical native geometry, including boundaries, inflow maps and voxel growth
  - the grain-class set and class order
  - mass resolution, `n_voxels`, initial time
  - the initial-availability policy name, active mask policy and depth update rule
  - digests (dtype, shape, C order, nbytes, SHA-256 of bytes) of owned read-only copies of the four initial arrays
  - the prescribed vegetation (canonical form plus native `vegetation_content_hash`) and threshold modifiers
  - the commit and avalanching specs

  Unknown static types fail closed (`unhashable_static_input`). Resource limits are recorded in the initialization
  receipt, not in the case digest.

  Because the held initial copies are read-only, a later caller mutation cannot redefine the case.
  `case_sha256` is computed once and never recomputed from live objects.

  `verify_case_inputs()` recomputes every live static input against the original record. It covers the `BedContext`
  type, and each context field (geometry, grain classes, mass resolution, commit and avalanching specs, fallback
  fraction) must still be the original object. It also checks the canonical geometry, including both boundaries'
  inflow-flux dicts, the class set and class order, the mass resolution bits, the specs, the fallback-fraction
  digest, the held read-only arrays and the prescribed fields. A static change raises `case_input_mutated`; a
  prescribed-field change raises `prescribed_field_changed`. The same verification runs first, before anything
  changes, in `fresh_scheduler_state`, both `adopt_*` methods, `stage_copy` and `restore_stage`, and once at the end
  of construction. A refusal never resets or rebuilds the evolving bed. Restoring the original input values makes
  the bed usable again.
- **Evolving identity** (`identity_record()` / `identity_sha256()`, schema `masy.dense_bed_identity.v1`). It contains
  the case digest, the bed time, per-array digests of all six shared fields and the prescribed digests. This identity
  changes whenever native exchange changes the bed. Such a change is never treated as case mutation.
- **`stage_copy()`.** The total nbytes of the six shared fields is checked against `stage_copy_max_bytes` before
  copying, then the fields are deep-copied with their native types preserved. The copy's identity must reproduce the
  live identity. `restore_stage(stage)` is a single-use rollback.

  Each stage carries a private owner token of the bed instance that made it. The token is never part of any hash.
  A stage from another instance is refused as `foreign_stage` before it is consumed, even when the case and identity
  digests are identical. Under standard `copy` semantics, a deep copy of a stage gets a new plain `object()` token,
  so it is foreign too. This is not covered by a check. This is in-process rollback only, not restart: the token
  does not survive serialization.

  The bed keeps no stores, deep-store versions or history. Retention is the caller's choice, and only boundary copies
  are intended.

## 3. Deliberate limitations of this chunk

- Dense `VoxelColumnState` only, on the NumPy CPU host with float64 arrays. Windowed (developed on a separate MAPLE
  branch, not touched), Compact, CuPy/GPU and voxel growth all refuse.
- Square grid, full active mask, flat-pool `MobileCohortState` only. A segmented cohort calendar refuses.
- No standing-water rule at wind entry. No moisture, growth or hydrology law. Prescribed fields are only verified.
- The memory bound is a documented conservative estimate, not a measured RSS.

## 4. Pending (not in this chunk)

- **Native wind event-end finalizer.** The adapter is independently reviewed and verified on an explicit development
  pair using actual native settlement. Its [absolute-origin patch](../agent_handoffs/maple_record_origin_handoff.md)
  still needs an upstream commit and new current-pair qualification. `adopt_scheduler_physical` continues to refuse
  a nonempty pool; the runner supplies only native-finalized genuinely empty state.

- **Runner:** the restricted `masy.dense_runner` Python API is independently accepted; see
  [runner qualification](dense_runner.md). A documented collaborator launcher and combined restart remain later work.
- Combined restart and persistence; GPU, windowed and performance qualification.
