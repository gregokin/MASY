# Shared scheduler Phase 0A: dated reusable event schedule and coordinator contract

Status: candidate implementation, source-only, after correction 01 (2026-11-10) against MASY
`68936bfa` as an additive layer. Nothing here runs a model, constructs native MAPLE/SYRUP/LEAF
state or changes an existing schedule, runner, checkpoint or source guard. The existing
`masy.dense_schedule`, `masy.case_schedule`, `masy.clock`, `masy.dense_runner` and their refusals
are untouched; the new modules import the clock/schedule validators and never redefine them.
Independent Codex review and the native adapters are later work; the stubs used in the tests are
mechanical and do not constitute acceptance of a three-model simulation. This is the Phase 0A
foundation, not whole-Phase-0 or coupled native simulation acceptance.

## 1. Scope and ownership

The coordinator (temporary name MASY) owns TEMPORAL coordination only: event order, the dated
calendar over the exact native clock, realized-time accounting, the process gate, dispatch to
explicitly supplied adapters and the temporal cursor. It owns no physics and no weather physics:

| Concern | Owner | Where the coordinator stops |
|---|---|---|
| Landscape initialization, including MAHLERAN imports; wind transport; bed/voxel authority | MAPLE (`masy.case_workflow` already calls the native case compiler) | A wind dispatch carries the complete native wind record and the exact native start; MAPLE runs it. |
| Rainfall, infiltration, routing, wet transport, storm completion and recession; all physical water | SYRUP | A rain dispatch carries the native SYRUP interval/guard semantics; the realized end (after recession) comes back from the adapter. |
| Weather physics: saturation vapour pressure, RH/VPD derivation, Hargreaves PET, polar/cold rules | SYRUP `maple_syrup.weather_pet` (Phase 0B candidate) | MASY validates the schema and binding, then delegates every evaluation through `masy.pet`; no formula is duplicated. |
| Vegetation initialization, roots, plant withdrawal requests, growth | LEAF | An interevent dispatch carries the interval, the process gate, the covered weather rows and daily PET. |

## 2. Files and schemas (all strict, closed-key, source-bound)

Documents are JSON (strict: duplicate keys, NaN/Infinity, nesting/entry/text quotas refuse; integer
literals beyond the interpreter's digit limit refuse `invalid_json`) or YAML restricted to the JSON
data model when PyYAML is importable (`yaml.SafeLoader` subclass: duplicate keys, non-string keys,
anchors/aliases, dates/binary refuse; the depth and entry quotas are enforced DURING composition,
before any nested node is expanded, so `[`×400 refuses `depth_quota_exceeded` instead of raising
`RecursionError`). Nothing is installed; a `.yaml` file without PyYAML refuses `yaml_unavailable`.
Files are regular, non-symlink and bounded (`IOLimits`: 1 MiB per file, 65536 entries, 32 levels,
256 distinct files by default). Module: `src/masy/scenario_io.py`.

### 2.1 Event files (`src/masy/event_files.py`)

| Schema | Content | Owning semantics reused |
|---|---|---|
| `masy.wind_event.v1` | `event_id`, `wind_forcing: {blocks: [{shear_velocity_m_s, direction_deg, duration_s}]}` | The complete MAPLE `wind_forcing` record form (`maple/core/parameters/wind_forcing.py`); scalars validated by the existing `masy.case_schedule.WindDeclaration`. Every occurrence runs the COMPLETE record in order; selection, splitting, retiming and authored start/end times refuse. |
| `masy.rain_event.v1` | `event_id`, `water_profile_id`, `intervals: [{duration_s, intensity_mm_h}]`, `recession_guard_s` | Existing `masy.dense_schedule.RainEvent`/`RainInterval` (positive intervals, at least one wet interval, positive finite guard). Rainfall end and recession guard end are planned edges; the realized end is never predicted. |
| `masy.interevent.v1` | `event_id`, `duration_s`, `processes: {evaporation, transpiration, growth}` (bools, exactly these three) | New. The only place the three processes can be enabled. |

Durations/forcing are plain JSON numbers (binary64; bool, NaN, ±inf, −0.0 and non-representable
ints refuse). Records are lossless (`masy.clock.encode_time`) and each event has a digest of its
canonical record; the scenario additionally binds the file's byte SHA-256.

### 2.2 Scenario (`src/masy/scenario.py`, schema `masy.scenario.v1`)

```json
{"schema": "masy.scenario.v1", "scenario_id": "...",
 "calendar": {"start": "2027-12-30T00:00:00+00:00", "convention": "proleptic_gregorian", "leap_seconds": "none"},
 "native_epoch_s": 0.0, "gap_policy": "refuse", "requires_continuous_soil_moisture": false,
 "weather": {"file": "weather.json", "sha256": "..."} ,
 "events": [{"occurrence_id": "interevent1", "file": "events/interevent_2d.json", "sha256": "...", "scheduled_start": null},
            {"occurrence_id": "wind1", "file": "events/wind_a.json", "sha256": "...", "scheduled_start": null}, ...]}
```

- `occurrence_id` is unique per use; the same file may be referenced many times (reuse). Files
  are relative, canonical, inside the scenario directory, and their bytes must hash to the declared
  `sha256` (`event_source_hash_mismatch` / `weather_source_hash_mismatch`).
- Any order of kinds is accepted; nothing requires alternation.
- `scenario.digest` covers the calendar, epoch, policies, weather hash and digest, and every
  occurrence's file hash plus lossless event record. Changing one byte of any referenced file
  changes it; re-declaring the hash still yields a different digest.
- Static checks: `weather_required` when an interevent enables any process; weather offset equals
  the calendar offset; weather starts no later than the scenario; scheduled starts are
  non-decreasing and never precede the minimum possible end of everything before them
  (`scheduled_start_before_minimum_end`: wind and interevent durations exact, rain contributes its
  rainfall duration only).
- The illustrative tree is `examples/phase0_scenario/` (interevent1, wind1, interevent2, rain1,
  interevent3 with a scheduled start under `gap_policy: "idle"`, wind2). It is a template, not a
  simulation input; its weather is declared `prescribed` (invented values).

### 2.3 Calendar (`src/masy/calendar.py`)

Explicit dated start `YYYY-MM-DDTHH:MM:SS±HH:MM` (fixed UTC offset only; `Z`, named zones,
daylight-saving rules, fractional seconds refuse). Convention `proleptic_gregorian`; `leap_seconds:
"none"` (every day is 86400 SI seconds). The dated layer sits ON TOP of the existing exact mapping
`G(t) = Fraction(t) − Fraction(T0)` from `masy.clock`: an instant is `start + floor(G)` seconds
plus the exact sub-second remainder as a fraction, so nothing is rounded. `datetime` handles leap
years and year boundaries. Helpers: `dates_touched(start, end)` (half-open interval, dates in the
scenario offset), `whole_days` (midnight-aligned integer days), `noleap_day_index` (the 365-day
index LEAF's `leaf/stepwat.py:37` documents; 29 February has no image and refuses).

### 2.4 Shared weather (`src/masy/weather.py`, schema `masy.weather.v1`)

One weather document is the ONLY weather stream of a scenario, read by both SYRUP (evaporative
demand) and LEAF (temperature/humidity/radiation responses). Row fields use the owning library's
names so one record maps 1:1 onto `maple_syrup.weather_pet.DailyWeather`: `date`, `tmean_c`,
optional `tmin_c`/`tmax_c` (together), optional `relative_humidity_fraction` (0..1, never
percent), `vpd_kpa`, `radiation_mj_m2_day`, `pet_mm_day`. Declarations are closed:

- `units`: exactly `degC`, RH `fraction`, VPD `kPa`, radiation `MJ_m2_day`, PET `mm_day`; other
  spellings refuse instead of being converted.
- `temperature.extrema` (bool) says whether rows carry `tmin_c`/`tmax_c`. `temperature.tmean` is
  `supplied` or `arithmetic_mean_of_tmin_tmax` (a declared MASY convention, flagged in `derived`,
  allowed only with extrema). `ecohyd_hargreaves` and any humidity field require extrema
  (`extrema_required`); `explicit` PET with mean temperature works without extrema or latitude.
- `humidity.supplied`: `none` / `rh` / `vpd` / `both`; `humidity.derivation` must name the owning
  library (`maple_syrup.weather_pet`), which derives the missing quantity once and checks a
  supplied pair for consistency; the relation it applied is copied from its record per row. With
  `both`, an optional `consistency_tolerance_kpa` is passed through only when the library's
  `DailyWeather` exposes a tolerance parameter (currently `humidity_consistency_kpa`, default
  1e-6); otherwise the library default applies and a declared tolerance refuses
  (`humidity_tolerance_unsupported`). A disagreement refuses `weather_library_refused` with the
  library's message.
- `radiation.supplied` bool; `pet.method` from the `masy.pet` registry.
- `provenance.kind`: the library's label `observed` / `prescribed` / `estimated` with
  description/source; simplified forcing declares `prescribed` or `estimated`, never `observed`.
- Rows are contiguous calendar days (missing-data policy: contiguous, no nulls); dates `YYYY-MM-DD`
  in the declared `utc_offset`, which must equal the scenario calendar offset.
- `WeatherSeries.library` records the actual module file and SHA-256 that validated the rows
  (development identity; not a source-locked runtime).

### 2.5 Latitude (`LatitudeBinding`, `latitude_binding_from_case_yaml`)

Latitude is NOT a weather or scenario field. It is MAPLE's existing `case.yaml`
`location.latitude_deg` (`/home/okin/MAPLE/src/maple/core/parameters/location.py`: descriptive
metadata, degrees north positive, NO default when omitted, null refused, not part of the scientific
case identity). `latitude_binding_from_case_yaml(path)` reads that section with the same closed
rules (`latitude_unspecified`, `latitude_null`, `latitude_invalid`) and records the file path and
SHA-256; the coordinator refuses `latitude_required` when the weather's PET method needs it and no
binding was supplied. This is the first physical use of the field; MAPLE's own resolver stays the
authority at compile time, and the binding's SHA-256 is meant to be cross-checked against the case
pins (`masy.case_workflow._inventory`) when a native adapter is wired. No MAPLE change was made.

### 2.6 PET registry and delegate (`src/masy/pet.py`)

MASY holds no PET equation. `PET_METHODS` is a declarative registry
(`ecohyd_hargreaves`: needs `tmean_c`, `tmin_c`, `tmax_c` and the bound latitude; `explicit`: needs
`tmean_c` and `pet_mm_day`, no extrema, no latitude; `penman_monteith`: declared hook, refuses
`pet_method_unimplemented`). `weather_pet_library()` lazily imports the ACTUAL owning library
`maple_syrup.weather_pet`, checks the documented Phase 0B names (`DailyWeather`,
`potential_demand`, `METHODS`, `SOURCE_IDENTITY`, error classes) and refuses
`pet_library_unavailable` / `pet_library_incompatible` with an actionable message; nothing is
installed, substituted or adapted silently. `library_origin()` records the imported file and its
SHA-256 plus the library's `SOURCE_IDENTITY` (EcoHyD coefficients 15.392 / 0.4093 / 1.405, dated
Gregorian J over 365, analytic polar handling, zero-clamped cold demand are the library's documented
choices, not MASY's). That origin is a development identity: real transport still goes through the
existing source-qualified MASY runtime, whose origin and checkpoint guards are unchanged. Daily PET
returned to adapters is the library's `PotentialDemand.provenance()` (method, date, J, latitude,
full weather record, PET, Ra equivalent, clamp flag, source identity) next to MASY's latitude
source; the partition into soil evaporation and plant transpiration belongs to SYRUP/LEAF and is
not spent twice.

## 3. Coordinator protocol (`src/masy/coordinator.py`)

```python
scenario = load_scenario(path)                       # strict, source-bound
latitude = latitude_binding_from_case_yaml(case_yaml, IOLimits())   # only when PET needs it
coordinator = Coordinator(scenario, {"wind": wind_adapter, "rain": rain_adapter, "interevent": ie_adapter}, latitude=latitude)
while not coordinator.finished:
    realized = coordinator.step()                    # sealed dispatch -> adapter.run(copy) -> validate -> atomic advance
record = coordinator.as_record()                     # temporal cursor for restart
```

Adapters are required for every kind present (`adapter_missing`; extra kinds `adapter_unused`;
no stub is substituted). Each provides:

- `capabilities() -> AdapterCapabilities(kind, adapter_id, source_identity, continuous_soil_moisture,
  duration_policy, calendar_support, native_state_checkpoint, native_staging, notes)`. No field has a
  default. `continuous_soil_moisture=False` is what the existing dry-reset rain handoff must
  declare; `requires_continuous_soil_moisture: true` refuses any `False`
  (`continuous_soil_moisture_unavailable`), and the cursor states the conjunction.
  `native_staging` must be `staged_until_accepted` (`adapter_staging_required` otherwise): the
  coordinator's atomicity covers its own cursor only and cannot roll back a native model mutated in
  place, so an adapter must stage native changes (bed, soil water, plants, process-private state)
  and publish them only after acceptance or discard them on refusal, as the existing `DenseRunner`
  stages bed copies. `duration_policy ∈ {any_positive_duration, whole_days_86400_only}` and
  `calendar_support ∈ {proleptic_gregorian_dates, noleap_365_day_index}` expose LEAF's current
  limits (§5).
- `run(dispatch) -> AdapterOutcome(status, native_end_s, commit_before_s, commit_after_s,
  returned_elapsed_s, evidence)`.

Capability preflight runs for EVERY occurrence of the adapter's kind BEFORE the adapter is called
and independently of weather or process gates: a whole-day-only adapter never receives a 1 s
interevent even when every process is disabled (`adapter_duration_policy_violation`; the duration is
never truncated), and a no-leap adapter never receives 29 February (`noleap_adapter_leap_day`).
The dispatch calendar carries `dates_touched`, `whole_days`, `noleap_day_index` and the preflight
record for all kinds.

Dispatch integrity. The coordinator seals the dispatch (`masy.dispatch.v1`: occurrence identity and
hashes, lossless event record, exact `native_start`, planned edges from the existing
`resolve_forcing_edges`, edge roles, calendar instants, process gate, adapter capabilities, expected
commit stamp, and for interevent with any process enabled the weather slice with daily PET) and
hands the adapter an independent deep copy. After `run`, the copy is compared with the sealed frame;
any change (edges, gate, identity, weather, extra keys) refuses `dispatch_mutated` with the cursor
unchanged. Validation never reads the adapter-visible object: the planned edges are re-derived from
the scenario's own event file and the sealed native start. Public accessors (`dispatch()`,
`realized`, `ledger`, `as_record()`, the value returned by `step()`) are independent deep copies;
mutating them cannot change accepted state (tested).

Bound identity and record ownership (correction 02). At construction the coordinator binds the
scenario digest (every event/weather record and file hash) and the latitude record; before every
dispatch, acceptance, publication (`as_record()`) and resume it recomputes them and refuses
`scenario_changed` / `latitude_changed` if the loaded objects were altered meanwhile, e.g. by an
adapter closure holding the original `Scenario` (bounded metadata only; no native state is hashed).
Accepted source-bound records are read-only: `IntereventFile.processes`, `Scenario.weather_ref`,
`LatitudeBinding.source`, `WeatherSeries.rows` / `provenance` / `library` are frozen mapping views
(assignment raises `TypeError`), while `WeatherSeries.coverage()`, `record()` and every coordinator
accessor return independent copies.

Commit chain. `commit_before` of an outcome must equal the accepted chain stamp (the declared
initial commit for the first occurrence) exactly (`commit_chain_refused`); the ledger's own
`commit_before <= native_start` rule alone would accept an off-chain first stamp.

Realized-time rules, each validated on a trial copy of the existing `ClockLedger` before anything
is published:

- Relative sequential dispatch starts the next occurrence at the REALIZED native end of the
  previous one (rain includes recession). Wind and interevent must end exactly at their planned end
  (`realized_end_mismatch`); rain must end within `[rainfall_end, recession_guard_end]`
  (`rain_end_outside_guard`). A status other than `"complete"` is not completion
  (`event_incomplete`); the output is kept only as `active_evidence`.
- An explicit `scheduled_start` earlier than the realized previous end refuses
  (`scheduled_start_overlap`); forcing is never truncated or shifted. A later start leaves a gap:
  `gap_policy: "refuse"` (default) refuses it (`uncovered_gap_refused`); `"idle"` records one
  explicit `idle` ledger entry with an all-false process gate.
- Commit stamps chain through the ledger; an interevent occurrence performs no bed commit
  (`interevent_commit_refused`); stamps from the future or out of order refuse via the ledger
  (`clock_refused`).
- Atomic advance: ledger, realized list and cursor are replaced together only after every check;
  any refusal (adapter exception included) leaves `next_index`, the ledger and `as_record()`
  unchanged. There is no resume of a stopped occurrence.

Process gate: `evaporation`, `transpiration`, `growth` are dispatched only inside interevent
occurrences, exactly over their own interval; wind and rain dispatches (recession included) carry
an all-false gate; calendar time advances through every kind; nothing is applied retroactively over
an excluded duration. Native infiltration, drainage/redistribution required by storm hydrology,
routing and sediment exchange remain storm processes inside the rain adapter.

## 4. Restart contract (temporal cursor only)

`Coordinator.as_record()` (`masy.coordinator_cursor.v1`) holds the scenario digest and id, native
epoch and initial commit stamp, the complete `ClockLedger` record, the realized occurrence records
(ledger indices, native/global/calendar start and end, planned end, realized−planned, process gate,
weather dates, status, evidence presence), `next_index`, adapter capability records, the latitude
binding record and the soil-moisture continuity statement. `Coordinator.resume(scenario, record,
adapters, latitude=…)` refuses `scenario_source_mismatch` (any changed file), `adapter_identity_mismatch`,
`latitude_mismatch` and malformed records (`malformed_cursor`), then REPLAYS the accepted prefix
through the same sealed-dispatch and validation path as a live run without calling any adapter: for
each recorded occurrence it builds the sealed dispatch (scheduled start, gap policy, capability
preflight, weather coverage; daily PET is skipped because no adapter receives it), feeds the recorded
ledger entry back as the outcome (`native_end`, commit stamps, returned elapsed, evidence presence)
and validates it (planned/realized ends, interevent no-commit, commit chain). The derived realized
record and the appended ledger entries must equal the recorded ones canonically, the rebuilt ledger
must equal the serialized ledger, and `as_record()` must round-trip identically; any difference is
`cursor_inconsistent` with the inner refusal in the detail. This is input consistency with the
authored schedule (e.g. a cursor realized under a +30 s scheduled start is refused against a
scenario authored at +20 s even though its ledger is self-consistent), not cryptographic
authenticity. Tests show that continuing from a resumed cursor equals the uninterrupted run at every
boundary, and that scheduled-start early/late, spurious idle, idle under `refuse`, interevent commit
change, wrong first commit, capability policy and weather coverage are refused on replay while a
valid rain recession correctly determines the next start.

This is explicitly NOT a native state checkpoint or transaction. Bed, soil water, plant and
process-private state (routing graph/source, cohorts, column state) remain adapter-owned
(`native_state_checkpoint` and `native_staging` capabilities, `state_scope` in the record), and the
cursor's atomicity cannot undo an in-place native mutation. A combined restart that binds this
cursor to native checkpoints (`masy.dense_runner_checkpoint`, SYRUP/LEAF state) is a later task
and is not claimed.

## 5. Current callers and integration gaps (read-only inspection)

| Source | Observation | Gap for continuous three-model operation |
|---|---|---|
| `src/masy/case_schedule.py:117–152` | `CaseSchedule` accepts only `CaseWindGroup`/`CaseRainGroup` and requires the complete native wind record partition. | Unchanged. The new wind event file keeps the complete-record rule per occurrence; a bridge that maps a scenario to the existing case CLI request is not written. |
| `src/masy/clock.py:139` | Exact global/native elapsed ledger. | Reused as is; the calendar is a layer over it. |
| `src/masy/dense_runner.py:1261–1281` | Rain acceptance requires a genuine native `DryHandoff` with zero surface depth, zero soil water, empty mobile load and phase `None`; bed changes are staged copies adopted after checks. | The existing profile is a dry-reset handoff: its adapter must declare `continuous_soil_moisture=False` (and `native_staging="staged_until_accepted"`, which its staging already satisfies in spirit). A continuous-water boundary profile needs a new explicitly qualified SYRUP completion/handoff, not a weakened guard. |
| `src/masy/case_workflow.py:54–65, 286–323` | Native MAPLE case compile/load with pinned authored files. | Landscape initialization stays there. The latitude binding's `case.yaml` SHA-256 should be checked against these pins when a native adapter is built. |
| `/home/okin/MAPLE/src/maple/core/parameters/location.py` | `location.latitude_deg` is optional descriptive metadata, no default, null refused, outside the scientific identity. | First physical use is bound here with source identity; MAPLE unchanged. Whether latitude should enter restart compatibility is an open design question. |
| SYRUP `agent_handoffs/tasks/shared_weather_pet_phase0_20261110/syrup_candidate/src/maple_syrup/weather_pet.py` | Owning weather/PET library (Phase 0B candidate, under its own correction cycle): `DailyWeather`, `potential_demand`, `ecohyd_hargreaves`/`explicit`/`penman_monteith`, `humidity_consistency_kpa`. | MASY binds it lazily by documented names and records the actual origin; the library is not yet part of a source-locked MASY runtime and its own acceptance is separate. |
| `/home/okin/leaf/leaf/coupling.py:28, 59–70, 227–248, 323–345` | `DAY_S = 86400`, `check_phase` rejects wind/rain/water phases, `step_day`/`step_days` are daily-only. | A LEAF interevent adapter must declare `duration_policy="whole_days_86400_only"`; the coordinator refuses non-midnight or fractional-day intervals for it for every occurrence, gates or not. Variable-duration interevent support in LEAF is not inferred from the variable-dt SYRUP column. |
| `/home/okin/leaf/leaf/stepwat.py:37, 103–114` | 365-day no-leap calendar, day 0 = 1 January, advanced by accepted biology days. | `calendar_support="noleap_365_day_index"`: the coordinator supplies `noleap_day_index` per date and refuses 29 February for such an adapter. Advancing LEAF's development clock through wind/rain without catch-up E/T/growth needs an explicit LEAF-side mapping and restart tests. |
| `/home/okin/leaf/benchmarks/daily_water_growth.py:312–330` | Root layer fractions must be strictly positive. | The agreed contract allows zero roots in some layers; a new LEAF mode with non-negative fractions and positive normalized total is required (recorded in the interface audit; not changed here). |

## 6. Plant water contract and related user decisions (documentation only)

Agreed root-uptake contract (see SYRUP `docs/subsurface_hydrology/development_phases_20261109.md`),
per coordinator timestep, in physical water volume with the explicit cell-area conversion done once:

1. LEAF: plant p requests `D_p` and distributes `R_pcl = D_p · w_pcl` over its actually rooted cells
   c and layers l, with `Σ_cl w_pcl = 1` (species traits and actual spatial roots set the weights).
2. SYRUP: reports withdrawable `A_cl ≥ 0` per cell/layer within its declared bounds.
3. Per cell/layer: `Q_cl = Σ_p R_pcl`; for `Q_cl > 0`, `f_cl = min(1, A_cl / Q_cl)`; each plant rooted
   there receives `U_pcl = R_pcl · f_cl`. For `Q_cl = 0` nothing is withdrawn. Same factor for every
   root in the location; no priority, no order dependence, no plasticity, no reallocation of unmet
   demand to wetter locations.
4. SYRUP validates, debits `Σ_p U_pcl ≤ A_cl` exactly once and reports accepted amounts and
   shortfalls; LEAF uses accepted amounts and keeps no second soil-water store.
5. Water-supported growth fraction for `D_p > 0`: `S_p = Σ_cl U_pcl / D_p = Σ_cl w_pcl f_cl`, applied to
   otherwise planned aboveground growth under its other constraints. It is NOT the minimum location
   satisfaction. `D_p = 0` is explicit (no withdrawal, no invented growth factor).

User example: `D = 100`, weights 0.5/0.5, available 25/200 → requests 50/50, accepted 25/50,
remaining 0/150, `S = 0.75`; water closes independently of the dimensionless factor. The exact-
rational reference arithmetic lives OUTSIDE the production namespace, in
`tests/_root_uptake_reference.py` (used by `tests/test_root_uptake_contract.py`: user example,
two-plant/two-cell case from the task's NumPy/Numba probe, zero demand, violations). The production
coordinator is not a plant-water solver: LEAF computes withdrawals, SYRUP owns accounting.

Separate user decision (2026-11-10), recorded for later phases and NOT implemented here:
preferential-infiltration receiving weights are LEAF species traits distinct from root-uptake
fractions. They are nonnegative relative weights that need not sum to one; a positive vector is
normalized for allocation, an all-zero vector disables that species' preferential pathway, and
`[0, 1, 1]` over three layers divides the species' share equally between layers 2 and 3. SYRUP
keeps one aggregate supply/capacity-limited water budget across all contributions and never
assumes every shrub participates. Phase 1 uses explicitly supplied hydraulic properties while the
actual MAPLE voxel texture is computed; a texture-to-hydraulic-property method is REQUIRED in
Phase 2. No Phase 1 physics is implemented in this candidate.

## 7. Verification executed (candidate, CPU, `/home/okin/MAPLE/.venv` Python 3.12.3, no installs)

Development import path for the owning PET library (not a qualified runtime); correction 02 used the
immutable archived PET correction-01 snapshot
`.../shared_weather_pet_phase0_20261110/author_handoff_correction01/candidate/src` (`weather_pet.py`
sha256 `a992cc67…`) while the PET candidate's next correction is active. Tests report the actual
module origin and skip with an explicit reason when it is absent.

```
cd <candidate> && PYTHONPATH=<both roots> python -B -m pytest -q -p no:cacheprovider --rootdir=. --confcutdir=. -c /dev/null \
    tests/test_phase0_scenario.py tests/test_phase0_weather_pet.py tests/test_phase0_coordinator.py tests/test_phase0_resume_replay.py \
    tests/test_root_uptake_contract.py tests/test_clock.py tests/test_case_schedule.py tests/test_dense_schedule.py tests/test_contracts.py \
    tests/test_sourceobs.py tests/test_config.py
# 223 passed, 217 subtests passed
PYTHONPATH=src python -B -m pytest ... tests/test_phase0_scenario.py tests/test_phase0_weather_pet.py tests/test_phase0_coordinator.py tests/test_phase0_resume_replay.py
# 32 passed, 26 skipped (explicit: owning PET library unavailable)
```

The new tests cover: reference reuse with distinct occurrences; calendar strictness, leap day and
year boundaries; exact durations and realized recession; overlap, gap and unfinished refusal;
strict metadata schemas including YAML composition quotas and giant numbers; deterministic
source-hash changes; refusal leaving the cursor and ledger unchanged; dispatch mutation (edges,
gate, identity, weather, keys) refused; public record tampering inert; capability preflight
independent of gates; staging declaration required; cursor round trip and resumed continuation
identity; resume replay refusing scheduled-start/idle/commit/capability/coverage inconsistencies on
otherwise valid ledgers; read-only accepted records and bound scenario identity; weather validation
through the owning library, latitude semantics and the PET delegate.
All 196 baseline candidate files except the two README notes are byte-identical to `baseline.json`.
Importing every new module imports neither NumPy, PyYAML nor `maple_syrup`.

## 8. Not claimed / open items

- No native wind, rain or interevent adapter exists yet; the test adapters are mechanical. The
  existing `DenseRunner` is not wired to a scenario; it remains the qualified dry-reset profile.
- No continuous soil-moisture handoff, no LEAF variable-duration driver, no combined native restart,
  no native transaction: the cursor cannot undo in-place native mutation (hence `native_staging`).
- PET physics, humidity derivation, polar/cold rules and their qualification belong to the SYRUP
  library and its own review; MASY only binds and records them. The library's import here is a
  development origin, not a source-locked qualified runtime. Terrain adjustment, Penman–Monteith and
  the soil-evaporation/transpiration partition are outside MASY.
- Whether `latitude_deg` should enter scientific restart identity is open. Uncovered gaps default to
  refusal because silently idle time would be unaccounted evaporation.
