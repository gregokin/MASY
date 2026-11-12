# Native water calendar adapter (`masy.native_water_calendar`)

Status: bounded CPU development candidate, NumPy and Numba, actual native9f Dense, fixed material. Qualified-development
protocol only (explicit frozen manifest); **not** a production source lock, CLI/default adoption, restart, wind handoff or full Phase 1.

## What it is
MASY stays temporal-only. `NativeWaterSession` owns an accepted pair `(Coordinator, LifecycleState)` and one bounded
unaccepted progress slot, and supplies the two genuine protocol adapters (`rain`, `interevent`) required by
`Coordinator`. All physics is the actual SYRUP lifecycle (`begin_storm`/`evolve_storm`, `enter_interevent_phase`,
`enter_interevent_from_boundary`, `evolve_interevent`, `close_interevent`), the actual `RainfallSchedule`/`rainfall_field`, the actual
`weather_pet` factories (`DailyWeather`, `potential_demand`, `terrain_factor`, `apply_terrain`, `pet_m_s`) and the actual
native MAPLE bed carried by the lifecycle (its `water.depth_m` aliases the authoritative surface H). No wind/water/PET formula is copied.

Capabilities: `continuous_soil_moisture=True`, `any_positive_duration`, `proleptic_gregorian_dates`,
`native_state_checkpoint='not_provided'`, `native_staging='staged_until_accepted'`.

## correction01 API changes (supersede author01 where they differ)
* `create(..., preparation=<genuine native_dense_profile.NativeDensePreparation>, fence=SourceFence, ...)` replaces the plain `source` dict. The preparation is validated
  cold (pinned literals, sealed record, `prep.recheck` under whole-call admission on the still-initial bed, `state.payload.bed is prep.bed`, `lop.context is prep.context`) and is **not retained**.
  `SourceFence` is a declarative expected record holding the real MASY (168) and SYRUP (308) manifests; `source_census` is now a bounded no-follow descriptor census
  (`CensusLimits`), `namespace_census` checks every loaded `masy.*`/`maple_syrup.*`/`maple.*` module against its manifest/root, `distribution_records(fence)` requires the `maple`
  distribution in the bundle `editable/` overlay. `module_origins` was removed.
* `NativeCalendarLimits` gains the required `temporal_reserve_bytes` (explicit caller reserve for coordinator Python metadata). `create(..., retained_native_bytes=)` is stored and charged.
* The session is immutable (`AttributeError` on assignment) and re-verifies its sealed config token before every step; `required_call_bytes()` returns the named whole-call requirement;
  a typed terminal failure is sticky for its frame (`terminal_summary()`); known memory failures are `NativeWaterCalendarMemoryError`.
* Row orientation: the native importer canonicalises row 0 = SOUTH (`maple/case_tools/importers/readers.py`), so `+row` is north, as `steepest_d4_descent` assumes.

## correction02 API/behaviour changes
* `NativeWaterSession.plan_create(...)` (metadata only) composes named cold phases (`nd.plan_preparation` recheck figure, lifecycle audit, census staging, terrain staging) + retained/temporal/caller
  reserves; `create` admits it (typed `NativeWaterCalendarMemoryError`) **before** any census/hash/read/recheck. `source_facts`/`preparation_plan_bytes` are the pure helpers.
* `required_call_bytes()` for rain uses `event_extra_bytes_formula` / `begin_storm_bytes_formula` (pure arithmetic mirrors of two SYRUP functions; tests compare them on real schedules and pin the real
  source digests) plus `rain_forcing_bytes`; the real `RainfallSchedule` exists only after whole-call admission.
* The census scanner streams `scandir` entries (limits per entry, at most one extra entry discovered) and holds no pending descriptors (directories are opened on pop and closed in `finally`).
* `session.exports` is a freshly detached plain copy; the retained record is deeply immutable; fence record and scenario digest are cached so the per-step config check reads no files.
* Nested native recheck quota failures are `NativeDenseProfileError` (not `MemoryError`); they are typed by comparing the plan figure with availability, never by message matching.

## correction03 changes
* `plan_create` keeps the common lifecycle resident `B = storm resident + owned fixed + existing reserves` live through the cold scratch phases:
  `max(B+native_recheck, B+census, whole_lifecycle_audit(=B+audit scratch), B+terrain_staging) + retained + temporal + caller`; the nested native recheck receives `nested_reserve = B + extras`.
  The native Dense helper excludes the retained bed/compile (exported separately); native H/mobile/material stay represented once by the existing native reserve. Excluded: native compile/config/coords/forcing,
  caller-input containers, process/JIT/allocator overhead; not a peak measurement.
* The genuine `prep.recheck` receives a fresh-reading availability wrapper: insufficient -> `NativeWaterCalendarMemoryError`, otherwise the actual reading is returned (readings are not reservations; nothing cached; native errors are not re-categorised).
* Every step plan carries `extras['persistent_terrain'] = 8N` while a terrain factor exists (E-on, E-off and rain); PET scratch / rain forcing remain separate (`pet_scratch_or_rain_forcing`).

## Contract
* Create (`NativeWaterSession.create`) refuses before any physics: wind; `gap_policy!='refuse'` (positive idle gaps); interevent
  transpiration/growth true; rain `water_profile_id` mismatch; occurrence/memory limits; source fence mismatch (bundle,
  native package, SYRUP package, module origins under the fenced roots, CuPy imported); latitude join mismatch (path, sha256, field,
  value between MASY's `case.yaml` binding and the native sealed admission); lifecycle state not a boundary at exactly the native epoch.
  Windowed/CUDA/remapped beds are refused by the actual `prepare_lifecycle`.
* `step()` is atomic: a **trial** `Coordinator.resume` is built from the bounded accepted record (the authoritative coordinator is never
  stepped, because `Coordinator.step` advances its cursor before deep-copying its return); the real adapter stages a pure physical child;
  the new pair and evidence are built; then ONE reference assignment publishes. Any earlier failure (including after real physics
  and after the trial coordinator's own internal advance) leaves the accepted pair object, record and state identical.
* Rain: exact-`Fraction` interval edges converted once; every `origin+edge` and the actual deadline `(origin+rain_end)+guard` are bit-compared with
  the sealed dispatch before the storm is begun (the 1-ULP origin=.1/duration=.1/guard=100 case is refused, nothing snapped). Realized end includes
  recession. E/T/growth are zero during rain and recession and never caught up. Consecutive rain uses the genuine zero-time
  `enter_interevent_phase`+`close_interevent` of the completed storm (tested inert).
* Interevent: from a completed storm or from a closed/initial boundary (explicit quiet policy). Split at exact calendar midnights; per date the real PET
  (explicit or Hargreaves with the dated latitude), terrain applied once, current date only. E off → explicit zero array; drainage/intake/routing still evolve.
  `interevent_max_dt_s/min_dt_s/retries` are **required** limits separate from the storm control.
* Incomplete (`call_step_budget` exhausted) returns status `incomplete`; the accepted pair is unchanged, one progress slot keyed by
  (pair serial, parent digest, occurrence, dispatch token, config token) continues the already-folded candidate. Cumulative per-occurrence cap cannot be reset by re-calling;
  recession-limit/retry/min-dt/cap are terminal and drop progress.
* Terrain: `terrain_factor_from_topology` uses `pond_bed.steepest_d4_descent` (atan of drop/dx), D4 codes 1/2/3/4 → N/E/S/W (code 1 = `+row` neighbour), factory
  strict `<5°` flat rule and factory south-hemisphere reflection.

## Resources (scope)
Whole-call admission (before any temporal record/hash/array/operator) against the column's `available_bytes_fn`: `max` over the named disjoint lifecycle phases of the transition
(enter/begin/evolve/audit/close, from `wl.required_*` and the lifecycle's own metadata estimators) + old accepted parent beside progress + PET scratch + caller-retained native bytes + `temporal_reserve_bytes` + caller reserve. Create adds the lifecycle audit, terrain staging and cold census staging. Excluded: Python/JIT/import/allocator overhead,
the coordinator's O(realized) replay in `resume` (bounded by `max_occurrences`), whole-process fit. Fixtures ≤4×6 cells, seconds of physics.

## How to run the tests (important)
Use the existing MAPLE CPU `.venv`, put the frozen native9f bundle (`editable`, `source/src`), the frozen 308-file SYRUP source and this candidate's `src` on
`PYTHONPATH`, and **disable the live SYRUP pytest configuration** (`-c /dev/null --noconftest`): the live `/home/okin/SYRUP/pyproject.toml` otherwise
prepends `/home/okin/SYRUP/src` and silently shadows the frozen `maple_syrup` (the tests then skip, they never import the live source).
```
. env (see report.md) ; $PY -m pytest tests/native_calendar -c /dev/null --noconftest --rootdir=$PWD -p no:cacheprovider
```

## Limits
No wind, positive gaps, T/growth, Windowed/CUDA/remap, stage/backwater, durable combined restart (current native39 codecs cannot be used for native9f or the boundary
origin), CLI/defaults. No dt-sensitivity or day/season efficiency claim; no GPU. Interevent `evolve_interevent` per call admits per-call bytes by the lifecycle, not by this module.
