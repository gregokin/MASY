# Configured water-only native calendar run (`masy.native_water_run`)

Opt-in command that runs a real MASY scenario (interevent and rain occurrences on a calendar) over a real MAPLE case with SYRUP's layered water
owner. It is separate from the read-only preflight CLI. MASY stays temporal-only: SYRUP owns all water, MAPLE owns the landscape and the latitude,
LEAF is not involved (no transpiration demand, no growth).

    PYTHONPATH=<masy_root>/src /home/okin/MAPLE/.venv/bin/python -B -m masy.native_water_run run CONFIG.yaml
    PYTHONPATH=<masy_root>/src /home/okin/MAPLE/.venv/bin/python -B -m masy.native_water_run manifest <tree> <out.json>   # development helper

A worked input set is in `examples/native_layered_calendar/` (its README gives the exact steps).

## What is reused (nothing is copied)
`masy.scenario.load_scenario`, `masy.weather` latitude binding, the Coordinator/`NativeWaterSession`, MASY's source census and weather/PET hooks;
SYRUP `native_case_profile`, `native_dense_profile` and `layered_case` (the accepted layered factory) and, optionally, the Stage-B native9f
lifecycle checkpoint codec. The parent process only reads the configuration and launches a fresh `-B` child whose environment comes from the
genuine SYRUP `dependency_bundle.runtime_environment`; no `sys.modules` swapping, no editable import counted as qualification.

## Source profile
`SourceFence.profile` selects a named SYRUP package literal from the closed registry `SOURCE_PROFILES`:

| profile | SYRUP package digest | use |
|---|---|---|
| `c3_literal308` (default) | `d00ad390…` | historical C3 adapter tests; record and identity strings unchanged |
| `native9f_layered_factory_95ef_cpu_v1` | `95efc6dc…0796d` | this run command (layered factory, NumPy/Numba CPU) |

A manifest or lock in the config is a declarative expectation; the literal digest, the real source census, the namespace/distribution check and
the cold recheck remain the authority. `manifest` records what a tree currently contains and is never qualification.

## Configuration (`masy.native_water_run.v1`, closed schema)
References only: `maple_case_dir`, `syrup_parameters`, `scenario`, `profile_id`; `source` (`profile`, `native_target`, `masy_root`, `masy_manifest`,
`syrup_root`, `syrup_manifest`); `execution.quiet_policy`; `limits` (interevent dt/retries, `storm` control, occurrence/date/step caps, record bytes);
`resources` (reserves and an optional `host_available_cap_bytes`, availability = min(reading, cap)); `output` (`run_dir`, `step_cadence`, byte/line
caps, `physical_checkpoint: none|terminal_standalone`, and the explicit `compile_output_reserve_bytes`). Unknown or missing keys, wrong types and symlinked paths refuse. There are no implicit
physical defaults, no resampling, no water reset. The backend (NumPy or Numba) is chosen in the SYRUP parameter file; CUDA, windowed storage,
wind, transpiration and growth are refused by the actual profile fences. Native epoch must be 0 (a nonzero epoch is refused, never restamped).
Latitude comes only from the case's `case.yaml` `location.latitude_deg`; omitted/null/out-of-range latitude refuses before physical work.

## Output budget (C1)
`max_output_bytes` and `max_file_bytes` are hard and all-inclusive and are never raised. `STATUS_RESERVE_BYTES` (8192) of both is reserved up front for the
terminal `RUN_STATUS.json` (minima: `max_output_bytes >= 32768`, `max_file_bytes >= 8192`); every other write — input copies, records, step lines, native compile
outputs, the standalone checkpoint — is admitted against what remains BEFORE it is written. A status that cannot fit is compacted (exact core ledger values) and, if
still too large, replaced by a minimal `failed` status; a run is never reported complete unless the persisted status says so. Free disk of the output parent must cover
`max_output_bytes` before the run directory is created.
The native case compile has no per-member output planner. Its outputs are therefore held against the budget up front (`compile_output_reserve_bytes`), every write of
the process is capped to `reserve // 16` (`RLIMIT_FSIZE`, `SIGXFSZ` ignored, restored afterwards) while it runs, and the real size is measured afterwards. Practical
restriction: a case whose compiled member exceeds `reserve // 16` refuses (`native_case_refused`); the tiny example needs about 610 KB of reserve, hence the 8 MiB
setting. The standalone checkpoint is planned with the codec's own `estimate_lifecycle_budget_v2` (serialized bytes + manifest cap) and refused before any write when it does
not fit the remaining budget (this plan is conservative: several times the bytes finally written).

## Input snapshots (C1)
The configuration is read once; its stored copy is the parsed bytes. Case files, layered parameter file, scenario, every event/weather file and every referenced `.npy` map
are pinned (SHA-256 of the bytes actually read; maps at configuration time) and compared again when copied; a changed input refuses `input_changed_after_pin` (or the
scenario/layered loader's own typed refusal) before physical adoption. Stored map copies are the factory's own immutable content (`inputs/syrup_maps/`). `resolved.json` records
the stored hashes, the original and run-owned case paths and the agreeing scenario/parameter/map pins. `RunConfig` is sealed (`dataclasses.replace` refuses).
The `import` case route stays unsupported by the native profile.

## Bootstrap ordering and per-file cap (C2)
The whole worker runs under `RLIMIT_FSIZE = max_file_bytes` (nested `min(compile reserve // 16, max_file_bytes)` during the native compile), so no writer — native compile, partial
temp files, checkpoint — can create a file over the per-file cap at any stage; settlements charge every untracked regular file under the run directory. A cap smaller than a
native compiled member (e.g. 16384 for the 38112-B sediment ledger) refuses before physical work. The compile's 16-member bound is justified by the ordinary package compile
(5 NPZ + report + provenance = 7 files, doubled for temp+publish = 14) and the case-feature fences of the native profile.
Configuration loading validates every value first without opening any reference. The parent (`references=False`) reads only the configuration (<= 1 MiB, the one declared-excluded
read) and refuses `cold_memory` before it runs any helper if `caller_reserve + 4096 B` is not available; the worker then pins each reference (case files, the parameter file once,
scenario, map streams in 1 MiB chunks, at most 32 maps) only after a fresh availability reading covers `caller_reserve + buffer model` (2x size per read, 12x per structured parse,
2 MiB per map hash, 64 KiB pin metadata). The 16 MiB manifest cap is enforced from the file size before the body. The parent passes the sha of the bytes it parsed
(`--expected-config-sha256`); a configuration that changed before the worker read it refuses before any output or physical work. The source-bootstrap policy (which source roots
may be executed) is NOT part of this and is specified only as a design in `bootstrap_design.md` of the task.

The per-file limit never raises an existing stricter process limit: the effective `RLIMIT_FSIZE` is `min(requested, finite existing soft, finite existing hard)` (a soft 0 stays 0), restored on every path.

## Source modes and the bootstrap gate (C3)
`source.mode` is required: `development_snapshot` or `production_committed` (no implicit default; each mode has its own closed field set).
* `development_snapshot`: integrity metadata only, never acceptance; every output carries `production_qualified: false`, the mode and the operator's declared `origin`
  (separate from the verified content; no repository identity is inferred from an ancestor `.git`). `masy.native_source_gate` (stdlib only, no NumPy/SYRUP/calendar import)
  verifies both source roots IN PLACE before the SYRUP helper child or the worker exists: real-directory root, every traversed directory and file without write bits, no
  symlinks/special files/`__pycache__`/`.git`, descriptor-relative no-follow traversal, count/size/depth/entry caps from metadata before any body, fd metadata joins against
  replacement races, exact file-set and hash match with a bounded manifest (16 MiB cap checked from size first). Traversal metadata, hash buffers and manifest parse are admitted
  from the fresh availability reading + caller reserve (a zero/too-low cap refuses before the helper). The parent's actually loaded `native_water_run*`/gate modules must equal the
  sealed MASY manifest bytes. Parent and worker each derive the same CLOSED bootstrap identity (mode, config sha, parent code, interpreter realpath+sha256, both tree digests and
  manifest shas, root identities, profile, native target, declared origin); the worker refuses `bootstrap_identity_mismatch` before any output. The in-worker census/namespace/
  distribution/cold recheck is unchanged. Ancestors of a snapshot root are not required read-only (the recorded `(st_dev, st_ino)` is re-checked); a trusted owner's `chmod`, root
  user and OS sandboxing are outside the contract. The native bundle is verified by the existing in-worker profile check, not re-walked by the gate.
* `production_committed`: refuses `production_recipe_unavailable` before any helper/worker for every profile and never falls back to development: 95ef is a historical
  uncommitted accepted snapshot, not a clean committed recipe. The positive path must use the existing `masy.runtime` verification through its `-I -B` launcher once a recipe for the
  final qualified source is published; that integration is deferred.

## Closed source closure and manifest memory (C4)
The parent's pre-helper `masy` code is a CLOSED set: `__init__`, `scenario_io`, `native_water_run`, `native_water_run_config`, `native_source_gate` (the real parent CLI loads
exactly these). The gate enumerates every `masy` module actually loaded, refuses any outside the allowed set (`parent_code_unexpected_module`) and any whose streamed, fresh-admitted
SHA-256 differs from the verified sealed MASY manifest (`parent_code_mismatch`); the worker verifies its own closure (the five plus `native_water_worker`) and re-derives the same
five-module parent record from the verified snapshot. Manifests use ONE closed parser: size cap (16 MiB, never raised) before the body, a bounded lexical preflight before
`json.loads` (no arrays, depth <= 2, token/entry caps, string caps), an exact parse admission (raw + decoded text + nodes + strings) and a closed structure: the emitted
`{"files": {path: sha256}, "note": "<short string>"}` envelope or a bare `{path: sha256}` map; descriptive extra keys, nested notes and non-string descriptions refuse
(`manifest_invalid`), so Root-style rich receipt manifests must be reduced to their `files` map. Harness note: importing Numba from READ-ONLY source trees needs a writable
`NUMBA_CACHE_DIR`. (C5) The public CLI now provides it: when the child environment has no non-empty `NUMBA_CACHE_DIR`, the parent creates `<private TMPDIR>/numba_cache` (0700) before the worker starts and removes it with the TMPDIR on success and failure; a caller-supplied value is honoured exactly and never created, altered or removed. The JIT cache is a JIT/context exclusion (not model output, not counted in the output caps, no disk or algorithmic-memory bound claimed).
The manifest preflight is a constant-memory byte lexer (C5; no regex): string length, whitespace runs/total (1 KiB / 64 KiB) and bare tokens (64 B) are enforced while scanning, so malformed input is refused with scratch independent of its size.

## Memory (C1)
After `NativeWaterSession.create` only the lifecycle operator (with its texture) and the session's current state are kept; the preparation, admission, layered configuration
and maps, initial setup/cover and initial state are released (checked by weakref: after create, after the first step and at the end; reported in the status summary). The
configuration's manifests are read by the worker only after the availability cap is known; every new cold buffer (manifest bodies, census staging, scenario/event/weather/case/parameter
reads, copies) is guarded by a fresh availability reading plus the caller reserve and refuses `cold_memory` before it is read. Declared exclusions: the parent bootstrap reads the
bounded (<= 1 MiB) configuration before the cap is known; native case compile/JIT/import whole-process footprint is not protected or modelled. Each run child gets a unique private
`TMPDIR`. `retained_native_bytes` still counts the maps conservatively (they are alive during `create`; they are released afterwards).

## Behaviour
Order: source verification → MAPLE case admission → layered factory → scenario/latitude → `NativeWaterSession.create` → release of the preparation →
occurrence steps. The case is copied into the run-owned `inputs/case` and admitted from there (the native compile writes `processed/` beside the
case files; the user's case directory is never touched) and the copies are checked against the digests pinned at configuration time. Interevents
run evaporation (explicit PET or dated Hargreaves from the scenario weather) only; rain occurrences run storm infiltration/routing and the
recession guard with no E/T/growth. Occurrences never overlap and no process is applied retroactively.

Outputs (exclusive creation, no symlinks, byte caps): `inputs/`, `resolved.json`, `steps.jsonl` (bounded line per accepted step), optionally
`physical_terminal_checkpoint_standalone/`, and `RUN_STATUS.json` written last. The standalone checkpoint is the Stage-B native9f lifecycle
checkpoint of the final state. It is NOT a combined calendar+physical resume; there is no `--resume`.

Exit codes: 0 complete; 2 refused before physical work; 3 failed or incomplete (last accepted boundary kept pure); 1 internal. The last stdout line
is `MASY_RUN_RESULT:<json>`.

## Limits
CPU only, single process, fixed layered material from the factory; not a GPU, windowed, wind, LEAF or evolving-soil qualification and no scientific
field validation or performance claim. The old C3 literal308 tests keep qualifying only their own frozen 308 source.
