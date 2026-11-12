# Combined calendar + physical-water checkpoint (`masy.native_water_checkpoint`, `NativeWaterSession.checkpoint/resume`)

Status: bounded CPU **development candidate** (R1 pure artifact layer + R2 session API). NumPy and Numba, actual native9f Dense, fixed material, SYRUP source
profile `native9f_layered_factory_95ef_cpu_v1` only. Not a production source lock, not CLI/config/worker integration (R3), not Phase 1 acceptance.
`PHYSICAL_CODECS` has exactly one entry; no GPU codec, wind, LEAF, Windowed, remap or mid-occurrence restart exists here.

## What it does
`session.checkpoint(path, checkpoint_limits=…)` seals the CURRENT accepted occurrence boundary (coordinator cursor + the one physical lifecycle role) into a NEW
directory. `NativeWaterSession.resume(…)` returns a NEW, UNADOPTED session restored from it. Resume never calls `create`, `begin_storm`, `enter_*`, `evolve_*` or
`close_interevent`, never restamps a clock, never replays rain/E/T and never zeroes a ledger: the cursor comes from the adapter-free replay of `Coordinator.resume`
and the physical state from the unchanged SYRUP codec (`lifecycle_checkpoint_v2`, profile `native9f_dense_lifecycle_cpu_v2`), joined by an exact path-free identity.

## Artifact (`<ckpt>/`)
`physical/` (EXACTLY the SYRUP bundle: `lifecycle_checkpoint.json`, `maple_state.npz`, `lifecycle_state.npz`), `temporal.json` (canonical coordinator cursor record,
≤ `limits.max_record_bytes`), `combined_checkpoint.json` (sealed manifest `masy.native_water_combined_checkpoint.v1`, linked LAST). Nothing else is accepted.
The manifest is closed (every nested record has a closed key set), ASCII canonical JSON, ≤ 512 KiB (derived: 6 × the declared string caps), depth ≤ 6, ≤ 4096
nodes, parsed after a constant-memory byte scan. It pins: cursor digest/bytes/sha, boundary role/clock/commits, scenario/occurrence/weather/calendar/epoch/gap policy,
latitude (historical record + path-free identity), fenced source identity (MASY and SYRUP manifests, literals, census), case/route/profile/lop/topology/quiet
policy, optional input pins, behavioural + resource limits, adapter capabilities, the physical bundle (manifest sha, `state_digest`, `physical_digest`, revision,
clock, identity) and the writer backend/runtime.

## Accepted role contract (closed)
| cursor | required physical state |
|---|---|
| no occurrence accepted | closed `boundary` at the native epoch (resumes through the same code, never `create`) |
| last accepted `interevent` | closed `boundary` at that entry's `native_end` |
| last accepted `rain` | the **completed `storm`** at that entry's `native_end` (never closed or restamped: closing would change the next interevent's source kind) |
| anything else | refused (`role_not_accepted`): `interevent` role, incomplete storm, wind, windowed/CUDA/remapped bed |
`initial_commit`/`last_commit` are the record's actual values. ANY pending `incomplete` progress or sticky terminal failure refuses (`pending_progress`,
`terminal_failure`); a partial occurrence is never checkpointed, abandoned, restarted or have its cumulative step budget reset.

## Joins at resume (all exact; before the heavy work unless they need the cold recheck)
Scenario bytes/occurrences/weather/calendar/epoch, fence identity + both manifests + census, case identity/route/profile/lop/topology/quiet policy, latitude
**path-free** (kind, field, sha256, value) — the historical record (with its original absolute case path) is restored for the cursor while the live copy is joined
to the native admission, so a resume in a different run directory keeps every future cursor digest identical to an uninterrupted run — adapter capabilities,
writer backend/runtime, optional `inputs_sha256`, behavioural limits (everything except the three resource limits `caller_reserve_bytes`, `temporal_reserve_bytes`,
`max_record_bytes`, which may change only under fresh admission and are disclosed in `session.resume_receipt`).

## Resources and admission
Whole-call figures are metadata-only and precede serialization, record creation, body hashing, census and codec work (`plan_checkpoint()`, `plan_resume(…)`); only the small
manifest header is a separately admitted first stage. Availability callables are strict (non-Boolean integer ≥ 0; `False`, `None`, negatives, non-callables are typed
`invalid_availability`; 0 is a valid reading; a valid callable that is falsy is still used; every call is a fresh reading so dynamic drops are honoured).
The unchanged conservative codec admission is used: the codec's whole figure counts the live/reference owner as `resident_old_bytes` and is compared with a plain free
reading, so it over-states the incremental need by at most that term (reported as `conservatism_upper_bound_bytes`; no "borrowed" credit is taken, because that term is a
modelled estimate that cannot be shown exact). Temporal/caller/retained reserves are passed once through the codec's `extra_reserve_bytes`. The reference
preparation/setup are caller-built borrowed objects (their factories keep their own planners); native compile, JIT, import and allocator overhead are excluded and no
whole-process bound is claimed.

## Publication (C1: ownership-proof rollback)
Exclusive `mkdir` → the codec writes `physical/` → `temporal.json` (O_EXCL, fsync) → manifest temp + `os.link` (atomic no-clobber seal) → fsync. Before the seal a
failure removes ONLY objects whose generation this call proves with the ORIGINAL descriptors it still HOLDS (C2): an open descriptor keeps an unlinked inode allocated, so a
replacement can never carry its number (a recorded `(dev, ino)` alone is not a proof: unlink + recreate reuses numbers). Each created file's creating fd is retained
(`_write_new(..., hold)`); `physical/` (its directory fd and read-only no-follow fds of the ≤3 closed members) is held only after the codec callback genuinely returned; the live
`fstat` of the held fd is compared with `lstat` of the path before every unlink/rmdir (never ctime/mtime/nlink, never `rmtree`). At most `MAX_HELD_FDS = 8` descriptors
(parent, outer, `physical/`, ≤3 members, temporal, temp manifest) are held and all are closed before `publish` returns or raises. A replaced file/directory, a foreign extra entry, or a `physical/` whose ownership is unknown because the codec callback raised is LEFT
(an unsealed, unloadable partial, no manifest) and reported in `exc.cleanup` (`removed`/`left`); a replaced outer directory is not touched at all. Failed-write cleanup inside the
codec stays the codec's. After the seal a failure is `CheckpointIndeterminate` and nothing is deleted. Not every failure path leaves an empty directory. Directory metadata is bounded: the artifact directory and `physical/` are checked by a descriptor `scandir` against closed name sets (stop at the first foreign name or after
closed-count+1 entries; one truncated name in the message, never a listing), and `Artifact.physical_path` runs the closed `physical/` preflight before the old codec opens it. Reads are
descriptor-relative with no-follow and metadata/size joins; the codec itself takes a path for its own bundle, so the outer directory identity is re-checked
immediately before and after the codec call (residual window = the codec's path handling). The loader never writes to the artifact.

## Reference ownership
`resume` takes the caller-built INITIAL `preparation`/`lop`/`reference_state` as the codec's expected components and for the cold source recheck (the only place
`prep.recheck` runs; never on a restored owner). They are never mutated or adopted. After the caller drops them the preparation, reference bed, initial state and setup
are released (tested with weak references); the loaded operator legitimately keeps the native context as its shared fixed component. The session holds exactly one bed.

## Source identity (important)
Adapter identity includes the MASY and SYRUP manifest digests, so an artifact resumes only under the exact sources that wrote it (95ef SYRUP package; the codec also
refuses another `maple_syrup`/`maple` pair). No migration. A future final SYRUP source needs a NEW literal profile; the 95ef and 308 pins are untouched. The capability
string `native_state_checkpoint` stays `"not_provided"` on purpose (cursor/adapters records stay byte-identical to C5).

## Tests
`tests/native_calendar_checkpoint/test_checkpoint_pure.py` (labelled test doubles; artifact only) and `test_checkpoint_real.py` (+ `ck_driver.py`/`ck_scenarios.py`, fresh
children with the genuine `runtime_environment`, read-only sealed roots): uninterrupted-vs-resumed identity of arrays, J, intake, lifetime, state/physical digests, cursor,
ledger, receipts and clocks at initial/after interevent/after completed rain and the remaining mixed trajectory (NumPy and Numba, rain- and interevent-first,
rain→rain, leap day + variable daily weather, chained save→resume→step→save→resume), tripwires, release, ~100 tamper/splice/caller-mismatch refusals, pending/terminal,
forged role/time, exact/−1/False/dynamic-drop admission, publication failure/indeterminate/source drift.

## Limits
CPU only; fixed material; accepted occurrence boundaries only; no wind, positive gaps, T/growth, Windowed/CUDA/remap, GPU, LEAF; artifacts are trusted-writer
integrity (hashes, not signatures); no cross-version/cross-backend migration; no production recipe (`production_committed` still refuses); no CLI/worker/`--resume`.
