# Daily CSV scheduler

This stage adds NEW, opt-in input formats, an immutable source-bound daily plan
and a read-only `inspect` command. **It runs no simulation.** There is no native
execution, physics or PET computation in the inspector. The separate
[native water connection](daily_native_water.md) delegates these operations to
SYRUP; LEAF execution remains gated. Existing
MASY APIs, `masy.clock` and every older scenario meaning are unchanged.

Use the existing MAPLE CPU environment's Python; no installation is required.

```
python -I -B tools/masy_daily.py inspect examples/daily_scenario/scenario.csv
```

`inspect` prints bounded JSON (identity, settings, weather summary, a capped
wind-occurrence list, plan counts, limitations). `--sample-days N` (0..366) adds
the first N plan days with at most 64 wind occurrence IDs per day; no full wind
block records or full weather table are printed. Input quotas can be
lowered or raised with `--max-file-bytes`, `--max-weather-rows`,
`--max-scenario-rows`, `--max-files`, `--max-total-bytes`, `--max-header-lines`,
`--max-header-chars`, `--max-line-chars`, `--max-cell-chars`, `--max-path-chars`
(positive integers; `max_files` counts the scenario, weather and distinct wind
files). Exit codes: 0 inspected, 2 refused (JSON with a stable
`error_code`), 1 internal error or a launcher not run with `-I -B`.

## Weather CSV (`masy.daily_weather.v1`)

UTF-8, `# key: value` preamble, then one ordinary CSV table. Required preamble
keys: `format`, `pet_method`, `precipitation_threshold_mm` (no implicit default).
Optional: `source`, `description`. Duplicate or unknown keys refuse.

| `pet_method` | exact columns |
|---|---|
| `ecohyd_hargreaves` | `date,tmin_c,tmean_c,tmax_c,precipitation_mm,precipitation_intensity_mm_h` |
| `explicit` | `date,tmean_c,pet_mm_day,precipitation_mm,precipitation_intensity_mm_h` |

Units: temperatures in degrees C, PET in mm/day, precipitation in mm per day,
intensity in mm/h. There is no humidity/radiation extension and no other method
or column. MASY contains no PET equation: the native connection delegates PET to
the actual SYRUP, and the Hargreaves latitude stays in the native
MAPLE case, not in the weather file. Columns must match exactly (no reorder,
duplicates or extras).

Rules: dates are canonical Gregorian `YYYY-MM-DD`, consecutive, strictly
increasing (duplicate, missing/gap and out-of-order dates refuse). Numbers are
plain decimals that must be finite in binary64 (no `nan`/`inf`, hex, overflow,
silent underflow or `-0`). Precipitation, PET, intensity and the threshold must
be non-negative; temperatures need `tmin <= tmean <= tmax`. Positive
precipitation with zero intensity refuses; zero precipitation with positive
intensity also refuses (inconsistent). Nothing is defaulted, derived or rescaled.
Blank lines, tabs, a BOM and malformed CSV refuse; one final newline is allowed.

**Wet period.** A day with `P > 0` rains at constant intensity `I` for `P / I`
hours (`P / I * 3600` seconds, shown as `nominal_wet_period_s` in plan records).
This is the nominal rainfall duration only; it excludes any recession.

## Scenario CSV (`masy.daily_scenario.v1`)

All preamble keys are required: `format`, `scenario_id`, `weather_file`,
`start_date`, `end_date` (inclusive; both covered by the weather file),
`evaporation_enabled`, `transpiration_enabled`, `growth_enabled`,
`growth_interval_days`, `output_interval_days`, `checkpoint_interval_days`,
`output_at_event_boundaries`. Booleans are the literals `true`/`false`; cadences
are strict positive integers (at most 9 digits). When growth is enabled the
growth interval must lie in LEAF's existing qualified 1..7 day range; it is never
widened. Output and checkpoint intervals may exceed the run length.

Event table header: `date,event_type,event_file`. Only `event_type` `wind` is
supported. A header-only table is valid. Dates must be nondecreasing and within
the scenario dates. Same-date row order is authoritative. Each row is its own
occurrence with a derived identity (`wind-000000`, ... plus a SHA-256 over the
scenario id, row index, date, reference and file hash), even when a file name
repeats. `event_file` and `weather_file` are canonical relative paths resolved
against the scenario directory; absolute paths, `..`, control characters,
symlinks (files or directories, even ones staying inside the root), special files
and non-`.yaml`/`.yml` wind files refuse. File bytes are hashed automatically;
every distinct file is read once, re-read at the end, and refused with
`document_changed` if it moved.

## Wind YAML

The native MAPLE format: only `wind_forcing: {blocks: [...]}`, each block having
`shear_velocity_m_s`, `direction_deg`, `duration_s`. Scalars are validated by the
existing `masy.event_files.WindEventFile` / `masy.case_schedule.WindDeclaration`
rules and the strict YAML reader; the complete record is retained unchanged per
occurrence (never truncated or rescaled). PyYAML is imported lazily; if it is
absent the read refuses with `yaml_unavailable` (nothing is installed). Native
MAPLE wind runtime validation and execution are a separate adapter qualification.

## Identity

`DailyScenario.identity_sha256` hashes a canonical record of: scenario file
hash/size, normalized settings, weather reference (scenario-relative), weather
file hash/size, preamble values (threshold as lossless binary64), a hash over all
parsed weather values, each distinct wind file's complete declarations, and
every wind occurrence (reference and hash) in row order. Reusing a wind file
does not duplicate its block declarations in the identity. No absolute path or working directory enters it, so
copies of a scenario directory share an identity. Changing the threshold, any
control, weather or wind bytes, or the order of wind rows changes it. Parsed
records are frozen dataclasses/tuples; `identity_record()`, `record()` and
`summary()` return fresh JSON-compatible copies.

## Daily plan and the two clocks

`masy.daily_schedule.build_daily_plan` iterates every nominal date from start to
end (dry and wind-free days included, leap days and year crossings included).
Each day carries weather, wind occurrences in row order, process flags,
`growth_due` / `output_due` / `checkpoint_due`, `event_boundary_output_requested`
and calendar boundary information (`is_calendar_year_end`, `is_leap_day`, ...).
Cadence `N` is due after `N` accepted days, anchored at the scenario start. There
is no implicit final growth flush and no biological year-end rule.

*Nominal clock:* Gregorian days. *Physical clock:* native elapsed seconds. Each
day is wind(s), then precipitation (including full storm recession if required),
then ONE native daily soil/E update, then LEAF transactions when available. There
is no subdaily inter-event soil/T/growth and no biological catch-up over
transport durations. Wind and storm time can make a nominal year physically
longer than 365/366 days. This is separate from the existing `masy.clock`.

**Precipitation selection (intended only):** `P = 0` no rain; `0 < P <= T`
attempts recharge (equality `P == T` is recharge); `P > T` requires a native full
storm. A recharge attempt may still fall back to a full storm when the
closed-column storage capacity is insufficient. Water above field capacity
moves downward through the layers; saturation limits storage. An open bottom
allows drainage. If any active closed column still cannot hold the rain, SYRUP
discards the entire recharge trial and applies the full original amount as a
storm from the unchanged parent. Other refusals do not authorize fallback. MASY does
not predict soil capacity, recession, physical duration, budgets or acceptance.
Native SYRUP Phase 1.5 remains the authority.

## Not delivered here

Native solver settings stay in the owner configs. Enabled transpiration or growth
require the future LEAF connection; no no-op model is substituted. This stage
delivers input parsing, identity and a nominal plan. The separately qualified
water-only API is described in [daily_native_water.md](daily_native_water.md).
