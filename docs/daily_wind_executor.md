# Native daily wind executor (`masy.daily_wind`)

## Scope

A thin, opt-in, bounded Dense NumPy CPU executor of **one complete wind occurrence** over an actual current MAPLE/SYRUP
`BedState`. It is a native wind *component*. It is **not** a wind-water daily run.

It does not: adopt or mint any SYRUP lifecycle state, own a day cursor, advance weather/PET, update water (no drainage, E/T,
moisture reset, sink or coupling), introduce moisture-dependent thresholds, or change native numerical formulas or tolerances. The wind-only configuration view
explicitly disables moisture feedback and selects the existing constant-depth bed-update rule; the authored case remains unchanged.
`daily_native.py` and its wind guard are untouched; the legacy dry-wind/reset paths and old source locks are unchanged.

## API

```python
executor = prepare_daily_wind(source, admission, context, limits=DailyWindLimits(...))
candidate = executor.execute(bed, occurrence, start_s)   # DailyWindCandidate
```

* `source`: genuine `NativeSourceProfile`; `admission`: genuine `NativeCaseAdmission`; `context`: native `BedContext`.
* `bed`: the current native `BedState` (may be later than initialization). `occurrence`: `daily_csv.WindOccurrence`.
  `start_s`: actual native physical start seconds (any finite float >= +0.0).
* `candidate.state` (native `BedState` from the returned complete physical fields), `candidate.end_s` (actual native end
  clock), `candidate.receipt()` (bounded scalar JSON; no arrays). Nothing is adopted; the caller owns acceptance.

## Source and association

Exact qualified guards stay explicit: MAPLE commit `9f466d2cfcd52991b24496cf67f4562d6268afa9`, MAPLE package
`fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e`, SYRUP N15 CPU package
`82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107`, CPU profile (no CUDA capability). `source.recheck()` and
`admission.recheck()` run at preparation, at every `execute` entry, and again before a candidate is returned; the source and
admission association and the prescribed vegetation/threshold digests must be unchanged. Newer or dirty sources are refused,
never rebound.

## Physics reused (no copies)

* `build_scheduler_physics_config(effective_wind_config, initial_available_fraction=context.initial_available_fraction,
  wind_record=<this occurrence's native record>)`: the forcing envelope covers the complete record; availability is never
  replaced by 100 %.
* Water coupling, `dry_aeolian_commit_separation`, a prescribed external shear field, a non-`detachment_transport_deposition`
  mode and non-NumPy backends are **refused**, not changed. The effective configuration overrides only `fluid_threshold.modifiers.moisture_enabled=false` and
  `water_coupling.depth_update_rule=constant_depth` (the existing native `BedContext` policy). Both original and effective
  choices are reported. A native `constant_free_surface` commit can alter water volume even when water coupling is disabled;
  that would violate SYRUP water authority during wind. Vegetation and threshold fields are the admitted compiled
  prescribed objects, passed by reference to `initial_scheduler_state`.
* A fresh native empty pool (`zeros_mobile_cohort_state(config.particle_transport.max_active_cohorts)`).
* `advance_wind_block` per block in exact row order. Native `advance_wind_record` enforces time relative to 0, so blocks are
  driven one at a time at the real physical origin (as `dense_runner` does); after each block the native record loop's logical
  cursor mapping (`block_index+1`, `block_offset_s=0.0`) is applied. No physical clock is rewritten and no timestep is dropped
  or rounded; each block's nominal end is `start + duration` in binary64 and the realized native end is recorded beside it.
* The native accounting split (`accounting_step_durations_s`) is checked against the configured step cap and its own rounding
  before any copy; substeps are never manufactured.
* `finalize_wind_event(..., record_origin_s=start_s)` is required. Settled mass is internal to the returned bed (counted once);
  only the finalizer's boundary export is added to the external export. An incomplete, unfinished or failing finalization is a
  refusal; mobile cohorts are never discarded.

## Dry wind and water

The effective native configuration disables moisture threshold feedback; the original source case and threshold arrays
remain unchanged. Wind therefore assumes dry regardless of soil or surface water. Nonzero ponded depth is allowed and is carried through **bit-identically**
(every water leaf is compared with the parent). Water-borne sediment must be exactly zero (finite, valid water state) at entry
and after every block. Note the native finalizer preserves carried water by identity and does not demand dryness.

## Bounds and transaction

`DailyWindLimits` (explicit plain ints): `max_cells`, `max_voxels`, `max_cohorts`, `max_blocks`, `max_accounting_steps`,
`max_stage_bytes`, `max_work_bytes`, `max_receipt_bytes`. Cell, voxel, block, step, cohort, stage-byte and work-byte caps are
checked before the transactional `deepcopy` of the six shared bed fields and before physics. Windowed beds and GPU arrays are refused.
Large-volume use is unqualified; explicit caller caps are admission limits, not scale qualification. The staged copy carries the parent's identity; afterwards the parent bed and
prescribed-field digests are re-verified. A refusal raises `DailyWindError(code, detail, evidence=...)` with scalar-only
evidence; no exception tracebacks or per-block domain states are retained by the executor. The sealed admission may retain
its initial compiled case; this is an initial reference, not a series of evolving beds. The workspace checks cover the
declared dense staging/native pool work; they are not a large-domain memory or performance qualification.

## Conservation

Per block: native `check_block_conservation` result, the native per-class reservoir identity against
`summation_error_bound_kg(...)` over `reservoir_arithmetic_scale_kg`, and the water-exchange fields. Per occurrence: the
finalizer's own transfer/reservoir identities and bounds; the combined class identity (transport + finalizer); and a whole
current-bed plus boundary-export per-class inventory closure (`bed_inventory` before/after). Aggregate bounds are composed only
from the native helper (`summation_error_bound_kg` with 6 terms per block, which equals `dense_runner`'s formula for one block)
plus the native reported bounds; no new epsilon or tolerance is introduced for conservation bounds.

## Receipt

Binds the source pair, case/config identity (`config_sha256`, `record_sha256`, executor identity with context, availability
digest and prescribed digest), occurrence id/hash/file hash, parent and candidate bed digests, expected/realized clocks per block
(lossless hex and exact fractions), native per-class input/export/bed deltas, settlement terms, and the check results.

## Pending (external dependency)

SYRUP Phase 2 is published, but its public lifecycle has no external-wind bed-adoption operation; see the
[published-source check](daily_phase2_readiness_20261113.md). Until that integration operation exists, a candidate from this
component cannot be adopted into a native lifecycle state, and there is no combined wind-water daily run. Connecting them is an
external dependency, not a scientific-choice question; this module provides no stand-in.
See [the native Phase 2 handoff requirements](daily_wind_phase2_handoff.md) for state, clock, accounting and restart acceptance.

## Verification status

Codex independently ran all 15 wind tests successfully, including the explicit wind-policy case. Final review and
exact-source verification are recorded in the [qualification note](daily_wind_qualification_20261112.md). Separate direct-native controls at a nonzero physical
origin compare the entire final bed bit-for-bit for dry and ponded cases, including an admitted moisture-enabled flag
that the wind view explicitly disables. These are synthetic 4 by 6, three-class component controls, not calibration.

Metadata/import/limits tests run with no native runtime; the genuine native CPU tests in
`tests/test_daily_wind.py` require `MASY_DAILY_NATIVE_BUNDLE` and `MASY_DAILY_NATIVE_SYRUP`. Root runs the verification.
