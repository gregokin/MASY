# Native LEAF + water calendar CLI (`masy.native_water_run.v2`)

Status: opt-in v2 CLI; bounded C1 qualification is recorded in `phase1_release_qualification.md`. Same command surface as `docs/native_calendar_run.md`: `run`, `manifest`, pause (`--pause-after-occurrences N`), `--resume DIR --expected-seal-sha256 HEX`, seal. The only difference is the
configuration schema: `schema: masy.native_water_run.v2` adds ONE closed `leaf` block to the v1 keys (plus the optional `cuda` block, unchanged). v1 files, records, artifacts and the parent's five-module closure are unchanged;
a v1 file containing `leaf` is refused as an unknown key, a v2 file without `leaf` is refused, and `v2` never silently runs water-only.

## Closed `leaf` block (mode-specific `leaf_root`; other keys required; unknown keys, Booleans-as-integers, strings-as-numbers, relative `leaf_root` and escaping paths are refused)
| key | meaning |
|---|---|
| `leaf_variant` | `"8da"` only (the d4fe codec-v2 family is not accepted by the CLI) |
| `leaf_root` | Development only; production must omit it (derived from the verified capsule; a supplied value is refused). In development, use an absolute path of the clean LEAF 8da reference capsule (23 modules + pyproject); verified by the literal source profile BEFORE any LEAF import |
| `catalog`, `plants` | relative files; `plants` is the plain LEAF CSV, its companion `<stem>.metadata.json` must sit beside it (all three are pinned by bytes) |
| `cadence_days` | integer 1..7, the growth grid |
| `geometry` | `max_edges`, `max_bytes`, `canopy_max_edges`, `canopy_max_bytes` (positive ints), `builder` in `numpy` / `compiled` |
| `budgets` | `leaf_budget_bytes`; `checkpoint{max_host_bytes, max_device_bytes, max_disk_bytes, max_biology_bytes}` (device cap 0 on CPU) |
| `io` | `max_catalog_bytes`, `max_plants_bytes`, `max_plant_records` (checked by a streamed row count against the metadata `n_plants` before any plant array exists) |

Rules: `source.profile` must be `native9f_layered_factory_afca_{cpu,cuda}_v1` (the CUDA profile needs the `cuda` block as before); `output.physical_checkpoint` is `none` or `combined_terminal` (the water-only
`terminal_standalone` bundle would drop the biology and is refused); there is no valid-cell mask key: the known active mask of the water setup is used (see below). Plants outside the grid, a grid/cell-area/layer mismatch with the catalog, an inconsistent CSV/metadata pair
and an unresolved SURFACE anchor whose cell is inactive/unknown are typed refusals (exit 2, no physical step). SURFACE anchors (initial plant files) are resolved on BOTH backends with LEAF's public `resolve_establishment` over the retained HOST initial bed (`PondTopology.source_bed_m`, generation 0, finite, anchors in active cells; no device-to-host copy, no second bed, nothing filled or invented).

## Order (every refusal before the first physical step exits 2)
parent: closed bounded config -> cold zero-memory refusal -> closed bootstrap identity -> child environment (runtime + MASY `src` + `leaf_root`). worker: identity -> run directory -> input pins (incl. the three LEAF files)
-> MASY/SYRUP census -> LEAF source census (`verify_profile` (development) or `verify_profile_delivery` (production)) -> plant-file control (caps) -> case/layered preparation -> public LEAF loaders and joins -> `NativeWaterSession.create_leaf(_cuda)` or
`resume_leaf(_cuda)` (plants are NOT read at resume; the biology comes from the joint artifact) -> scheduled run. `limits.storm.max_steps` is the per-call native work budget; the whole occurrence is bounded by `limits.max_accepted_steps_per_occurrence` and `call_step_budget` (the accepted C5 worker continues incomplete native calls of the same occurrence). A step-budget exhaustion ends exit 3 with no checkpoint; a partial storm is never checkpointed.

## CSV record count (I/O control only)
The pre-array control counts plant records with the stdlib `csv.reader(strict=True)` over a bounded UTF-8 text wrapper (the same record semantics as LEAF's parser: a quoted multi-line field is ONE record, blank records are skipped, a malformed quote is a typed refusal), without a line list or row history, and stops at `max_plant_records`. LEAF's parser stays the authority on every value, the header and SURFACE tokens.

## Cold HOST/DEVICE admission (before any native LEAF object)
Stage records: `cold_plan_lower_bound` (before the catalog is parsed: table with S=1, mask, parse, geometry, population), `cold_plan_remaining` (actual S, ALL Q base records), `plant_parse`, `geometry`, `valid_cell_mask`, `population`; resume: catalog/table/mask only. Each stage takes a fresh strict reading of the lifecycle operator's REAL caller readers (CUDA: the lifecycle's own host reader and the device reader kept separate) and refuses when the stage's estimated NEW bytes + the explicit reserve (`resources.caller_reserve_bytes`; CUDA device: `cuda.device_reserve_bytes`) exceed it. Held input files and earlier results are not charged again; the component `max_bytes` values are only the declared size caps used in the estimates, never availability. The declared stdlib-side models are the named constants in `native_water_leaf_run.py`; LEAF's own constants/`table_memory` are reused. Records appear in `summary.leaf_create.cold_admission` (with `n_base_initial`, `resolved_anchors`, `masked`, `canopy_valid_area_m2`) and `resolved.json`. Interpreter, JIT, CUDA context, pool rounding and RSS are excluded (a reading is not a reservation).

## Shared valid-cell mask
The KNOWN active mask of the water setup (`PondTopology.active`, host, flattened once) feeds BOTH the initial canopy build and the interval operator geometry (so every growth rebuild and the resume operator use it): inactive cells carry no canopy area and LEAF still owns the roots. A fully active domain passes None (whole domain valid) which is exactly equivalent. Changing the mask at resume is refused (input pins / static identity).

## Resume
The artifact is the joint `masy.native_water_leaf_checkpoint.v1` (`physical/` + `biology/` + `temporal.json` + sealed manifest). Resume re-pins the catalog, plants+metadata, cadence, geometry and limits; any change is an
`inputs_mismatch`. A water-only artifact is refused by a v2 resume and a joint artifact by a v1 resume; missing, extra, truncated, symlinked or wrong-seal members are refused. A resume always takes a NEW run directory.

## Outputs
`summary.leaf` is the bounded biology summary (comparable between an uninterrupted run and a paused+resumed one). Create/resume-time, run-specific facts are in `summary.leaf_create`: `cold_admission` (fresh free-memory readings, so volatile), and on create `n_base_initial`, `n_live_initial`, `resolved_anchors`, `masked`, `canopy_valid_area_m2` (host float on CPU; `None` on CUDA, which never exports a canopy map just for a scalar).
`RUN_STATUS.json`/`MASY_RUN_RESULT` summaries gain bounded `leaf_source`, `leaf_inputs` and `leaf` entries (generation, plant counts, calendar cursor, pending growth days, participant id). Nothing unbounded (plant rows) is written to status files.

## Limits
CUDA file-CLI tests remain explicitly opt-in. Actual committed-runtime CPU/CUDA run and restart evidence is in `phase1_release_qualification.md`; this is a small fixed-geometry qualification, not a capacity or field claim.
`production_committed`: the v2 C1 path now has actual bounded CPU and CUDA
run/restart qualification; see [native_production.md](native_production.md)
and [the exact code/evidence record](phase1_release_qualification.md).
v1 production still refuses. Dated-clock limit: event/interevent boundary
instants must be exactly representable in the microsecond LEAF clock;
unsupported instants are refused rather than rounded. No wind, Phase 1.5
recharge, moving-soil, field or scaling qualification is implied. Example:
`examples/native_leaf_calendar_cli/`.
