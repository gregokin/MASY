# Configured CPU calendar run: combined checkpoint, pause and `--resume` (R3)

Status: bounded CPU **development** feature on top of the R1–R2 artifact (`masy.native_water_checkpoint`, `NativeWaterSession.checkpoint/resume`). Same source profile
(`native9f_layered_factory_95ef_cpu_v1`, genuine native 9f), `development_snapshot` mode only (`production_committed` still refuses typed). Not GPU, wind, LEAF,
mid-occurrence restart or production acceptance. Old `none` / `terminal_standalone` runs and the default (no options) invocation are unchanged.

## Surface
```
python -m masy.native_water_run run CONFIG                                      # as before
python -m masy.native_water_run run CONFIG --pause-after-occurrences N          # needs output.physical_checkpoint: combined_terminal; exit 4 when paused
python -m masy.native_water_run run CONFIG --resume CKPT --expected-seal-sha256 SHA [--pause-after-occurrences N]
```
* `output.physical_checkpoint: combined_terminal` (new, explicit) writes `<run_dir>/combined_checkpoint/` after the invocation's last accepted occurrence: the sealed
  calendar+physical artifact (`physical/` SYRUP native9f bundle + `temporal.json` cursor + `combined_checkpoint.json` seal). It is NOT the standalone physical bundle
  (`terminal_standalone`, unchanged) and NOT a cursor-only record. A run writes one kind or the other.
* `--resume` is explicit, pinned by the manifest seal (`--expected-seal-sha256`, required), consumes the artifact read-only and always runs in a FRESH `output.run_dir`
  (an existing run directory is never overwritten). There is no latest-checkpoint lookup, scanning or fallback. The configuration may change only resource reserves
  (`caller_reserve_bytes`, `temporal_reserve_bytes`, `max_record_bytes`); they are freshly admitted and disclosed in `resolved.json.invocation.resume_receipt`. Every
  behavioural limit, the scenario/events/weather/calendar, case/latitude/inputs, source fence and quiet policy must equal the writer's or the resume is refused (exit 2).
* `--pause-after-occurrences N` stops after the N-th accepted occurrence IN TOTAL (resume position included; it never resets accepted-occurrence, per-occurrence step or retry
  budgets). `0 <= N < scenario length`; with `--resume` it must exceed the resumed cursor position. A pause happens only between accepted whole occurrences and only with its
  checkpoint written; a requested pause whose checkpoint fails is `failed`, never `paused`.

## Status and exit codes
`0` complete · `1` internal · `2` refused before physical work · `3` failed or incomplete (an incomplete/cumulative-budget failure writes NO checkpoint) · **`4` paused**
(`status:"paused"`). `RUN_STATUS.json` carries `summary.combined_checkpoint` (written/path/kind/seal/cursor/state/physical digests/planned vs written bytes/lineage, or the
truthful failure incl. the publisher's `cleanup.removed/left`) and `summary.invocation` (resumed, seal, pause_after, start/end cursor index, receipt). Resuming an already
FINISHED checkpoint is a verified no-step resume: complete, `accepted_steps: 0`, `invocation.resume_of_finished: true`. `resumed_from.json` records the consumed artifact (path,
seal, cursor, role, prior lineage). `resolved.json` gains `invocation` only when an option or the combined mode is used.

## Authentication and order
Parent (stdlib, exact 5-module closure): configuration, production gate, zero/low-budget refusal before any helper, bootstrap identity, then metadata-only option checks; it
passes the SHA-256 identity of the controls (`--expected-options-sha256`) which the worker recomputes before any output or physical work. Worker: fresh output directory →
source census/namespace/distribution → scenario → resume artifact pre-authentication (no-follow bounded open, admitted header, seal, cursor, pause bound) BEFORE the reference
build → input snapshots → layered factory (the caller-built reference) → `NativeWaterSession.resume` (never `create`, never restamp/replay; a completed storm stays a storm) →
reference released → steps → optional combined checkpoint planned by `plan_checkpoint()` and reserved in the output budget before any write, under the same per-file cap.

## Limits
CPU only, fixed material, accepted occurrence boundaries, tiny fixtures; admission figures exclude process/JIT/allocator overhead; each resume rebuilds the reference (cold native
compile). Artifacts resume only under the exact MASY/SYRUP sources that wrote them (R2). Unsealed partials left by an unprovable failed publication are never loadable.
