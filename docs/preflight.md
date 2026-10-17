# MASY-01 read-only current-source preflight

The preflight observes the current MAPLE/SYRUP pair, imports it under guards, records native resolver and API evidence, and writes a truthful JSON report. It never runs transport and has no success verdict. Signatures are interface evidence only. They do not qualify physics, restart or alternating events.

## Invocation

```
/home/okin/MAPLE/.venv/bin/python -B -I /home/okin/MASY/tools/masy_preflight.py \
    --config /home/okin/MASY/examples/preflight.example.toml \
    --run-dir /tmp/masy-preflight-<new-id> [--expect-lock LOCK.json]
```

The parent of `--run-dir` must already exist; only the leaf is created (exclusively). `/tmp` always exists. A fresh MASY tree has no `runs/` directory: to keep runs there, create the parent once yourself (`mkdir -p /home/okin/MASY/runs`) and then pass `--run-dir /home/okin/MASY/runs/preflight-<new-id>`. The launcher never creates parents.

`--help` (or `-h`) prints usage and exits 0 without creating anything or importing MASY/upstream code. It is informational, not an audit and not a transport-ready verdict.

| Exit | Verdict | Meaning |
|---|---|---|
| 0 | (none) | `--help` only. No audit is run |
| 2 | `refused` | A guard failed: launcher flags, run-dir/input overlap, unreadable config, config, source, dirty scope, lock, concurrent change, module origin, native resolver, provenance cross-check, or an oversized report. Launcher-level refusals print to stderr and publish no report |
| 3 | `incompatible` | A required API is missing or mismatched, and no refusal occurred |
| 4 | `not_transport_ready` | The best achievable verdict. The standing blocking findings always apply |
| 1 | (none) | Internal error. No verdict is published |

Precedence is `refused` > `incompatible` > `not_transport_ready`. `transport_ready` is always `false`.

**Live results (observed baselines, not a prediction for every future run).**

- On the initially observed clean pair (MAPLE HEAD `f1555e09e7d6e49dadb2413b36a1700d4f3d0ea5` with a clean scope), Codex independently reproduced `refused` (exit 2) at the native resolver: SYRUP's own `resolve_maple_dependency('/home/okin/MAPLE')` fails because the installed MAPLE metadata yields `source_root=None`. MASY does not bypass, patch or repair that guard. The resolver error stays in `native_audit.native_resolver`, with SYRUP's `check_required_api()` result and the MASY API inspection beside it.
- During MASY-01 a separate upstream edit then appeared in MAPLE `src/maple/core/parameters/flow_separation.py` (HEAD unchanged, file dirty), outside MASY's scope. While it was uncommitted, the live audit correctly refused earlier, with `dirty_scope`, before any upstream import. MASY did not adopt, revert, copy or work around it, and did not fall back to the earlier clean observation.
- That work was later committed independently, outside MASY: MAPLE HEAD `5b1a052e1b9a9edfed0760152aac2a743813e79a` with a clean `src`/`pyproject.toml` scope; SYRUP stayed at `9fabdca60bd9cd7c5b358d5cbf620dbd3209d93a`. Codex's read-only audit of this pair again reaches the native resolver and reproduces the same `source_root=None` refusal. SYRUP's own `check_required_api()` passes there, but its list is incomplete: the MASY list fails on the Dense-path type dependency below (finding 11). Both pieces of evidence are kept in the report; the verdict stays `refused`.
- Which refusal a live run reports depends on the actual upstream state at that moment. No API pass, alone or together, proves Dense compatibility or qualifies anything physically.

## Launcher (`tools/masy_preflight.py`, stdlib only)

The launcher runs these steps in order:

1. `--help`/`-h`: prints usage and exits 0 (nothing created or imported).
2. Refuses unless `sys.flags.isolated` and `sys.flags.dont_write_bytecode` are set (`-I -B`). Then parses arguments strictly (no abbreviations); a usage error exits 2 before anything is created.
3. Sets `sys.pycache_prefix = RUN/cache/pycache`, computed from the resolved future run directory. The directory is not created: with `-B` nothing is written, and because the prefix tree is empty, no stale bytecode from source `__pycache__` directories can be read.
4. **Input/output separation, before any write.** Reads the config's `[sources.*].root` strings with a bounded stdlib TOML read. The run directory is refused (stderr only, nothing created) if, in either its lexical absolute form or its symlink-resolved form (symlinked parents included), it equals, lies inside or contains any configured source root (lexical and resolved forms) or MASY's `src/` or `tools/`. Configured roots are protected even when they are otherwise invalid (missing, not a repository) and even when the rest of the config is invalid. A config that cannot be read or parsed refuses here too, because the inputs to protect are unknown. Ordinary run directories elsewhere (including `<MASY>/runs/...` and siblings such as `MAPLE-run`) are unaffected.
5. Creates `--run-dir` with `os.mkdir` (the parent must exist). If the directory already exists, the launcher refuses and leaves it untouched. The resolved location is re-checked after creation; if it changed or now overlaps an input, the empty directory is removed and the launcher refuses.
6. Creates `RUN/cache/numba` and `RUN/cache/matplotlib` and sets:
   - `NUMBA_CACHE_DIR` and `MPLCONFIGDIR` to them;
   - `NUMBA_DISABLE_CUDA=1`, `CUDA_VISIBLE_DEVICES=""` and `JAX_PLATFORMS=cpu`. `maple/__init__.py` imports JAX, and these values stop imports from probing devices.

   They are recorded under `launch.environment_overrides`; the protected inputs under `launch.protected_inputs`.
7. Inserts `<MASY>/src` at `sys.path[0]`, then calls `masy.cli.main`.

Limit: the overlap check protects the roots the config names and MASY's runtime directories. It cannot protect an unrelated tree that the config does not name; a parent swapped by another process between the check and `mkdir` is caught only by the post-creation re-check.

## Stages and gating (`src/masy/cli.py`)

1. **Strict config** (`config.py`). This is a closed TOML schema `masy.preflight.v1`:
   - Unknown keys refuse. That includes `update`; no such field exists.
   - Calendar and absolute-time keys refuse by exact name at any nesting level with `unsupported_calendar_field`: `start_datetime`, `start_date`, `calendar`, `timezone`, `epoch_utc`, `absolute_start_s`, `fixed_calendar`.
   - IDs and paths that contain dates are valid.
2. **MASY module origin.** Every loaded `masy*` module must come from `<MASY>/src/masy`.
3. **Source observation** (`sourceobs.py`, stdlib only, no upstream import). MASY is observed first, then MAPLE and SYRUP. The optional `--expect-lock` is checked next. Any failure skips all upstream imports.
4. **Upstream imports.** The `src` roots of MAPLE and SYRUP are inserted after MASY's `src`, and the exact `sys.path` before and after is recorded. Then `maple` and `maple_syrup` are imported. Any loaded module outside its concrete observed package directory (`<root>/src/<package>`) refuses. Each package's `__path__` must be exactly that directory. Sibling-prefix directories and ancestors are not accepted.
5. **Native audit** (`contracts.py`). This stage runs SYRUP's unmodified `resolve_maple_dependency(<maple root>)`, then SYRUP's `check_required_api()`, then the MASY API list below, then the optional modules. Each part runs regardless of the others.
6. **Provenance cross-check.** SYRUP's `maple_syrup.provenance.source_tree_digest` and `scoped_git_status` must agree with the stdlib observation. After that comes a final origin sweep over every loaded `maple*`, `maple_syrup*` and `masy*` module. Last, installed distribution metadata for `maple`, `maple-syrup` and `masy` is read and reported, never rewritten.
7. **Environment record** (`importcheck.environment_record`). It records:
   - the Python version, the realpath of the executable and its bounded sha256, the prefixes and the flags;
   - the NumPy version and its compiled core (`numpy._core._multiarray_umath`), with path and hash, only if upstream imports loaded NumPy;
   - metadata versions of numba, llvmlite, jax, jaxlib and the CUDA packages. These are never imported here. If an upstream import already loaded them, their origins are recorded.

   `gpu_queried` is `false`.
8. **Post-audit re-observation.** This runs whenever the first observation succeeded, including after a lock, import, resolver or API failure. Any difference refuses with `source_changed_during_preflight`. A post-observation guard failure refuses with `post_observation_refused`.
9. **Declarations.** Clock and forcing data are resolved with data-only helpers (see [clock_contract.md](clock_contract.md)). Budgets are recorded as `declared_not_enforced`. Case inputs are recorded as `absent` or `declared_unverified`, because they are never read.

## Source observation details

The scope for each upstream is `src/` plus `pyproject.toml`. `root` must be canonical (no symlink) and must be the repository's top level.

- **HEAD.** HEAD is `git rev-parse --verify HEAD^{commit}`, the current local committed revision. If the optional `expected_revision` differs, the observation refuses. There is no historical fallback.
- **Dirty scope.** `git status --porcelain=v1 -z --untracked-files=all --ignore-submodules=none -- src pyproject.toml` must print nothing. Untracked files outside the scope, such as cases and notes, are ignored.
- **HEAD tree check.** Every HEAD tree entry in the scope must be a regular blob. Mode 120000 (symlink) and submodules refuse. Each committed file must exist on disk with the same git blob id (sha1 or sha256 object format). This catches edits hidden from `status`, for example `--assume-unchanged`.
- **Actual filesystem.** The real tree under `src/` is enumerated with `lstat`, because Git's tracked list omits ignored files that Python would still import.
  - Only `__pycache__` directories are pruned. They are never read, because of `pycache_prefix`.
  - Any other uncommitted file refuses with `untracked_runtime_input`. That includes ignored `*.so` files and sourceless `*.pyc` files outside `__pycache__`.
  - The one exception is generated distribution metadata directly under `src/` (`*.egg-info`, `*.dist-info`). It is hashed and listed in `generated_metadata` with the policy `recorded_not_repaired`.
  - Symlinks (file or directory) and special files refuse.
  - Files are opened with `O_NOFOLLOW` and re-checked with `fstat`. A size change during the read refuses.
- **pyproject.toml.** It is hashed (sha256 and git blob) and parsed with `tomllib`. `project.name` must be `maple` or `maple-syrup`.
- **Limits.** `max_files` and `max_bytes` apply during enumeration and before each file is opened.
- **Git calls.** Every git call uses `--no-optional-locks -c core.fsmonitor=false`, drops inherited `GIT_*` variables, and is bounded by `git_timeout_s` and by `git_max_output_bytes`, a hard cap on combined stdout+stderr read while the child runs (stderr-only output counts). On timeout or cap refusal the child is killed and reaped; at most 2000 stderr bytes are kept for the `git_failed` message. Nothing writes to any repository.
- **Live checkouts only.** The observation requires live Git checkout roots (HEAD, status and the HEAD tree are read from `.git`). It cannot simply be pointed at a materialized bundle, which has no `.git`. A future bounded locked-runtime verification mode would have to verify a bundle against the upstream bundle/origin manifests instead; MASY-01 does not implement it and does not weaken the checkout guards to accept one.
- **Recorded digests.** `package.digest_sha256` uses the same algorithm as `maple_syrup.provenance.SOURCE_DIGEST_ALGORITHM`. The tracked scope digest and the generated-metadata digest are recorded separately.

MASY has no Git repository, and none is fabricated. Its observation hashes `src/masy/**`, `tools/masy_preflight.py` and `pyproject.toml` (if present). This observation is guarded only by an explicit lock.

**Lock.** A lock file is the JSON of a previous report's `source_observation` block, copied unchanged, with exactly the keys `masy`, `maple` and `syrup`. Any difference refuses with `lock_mismatch` and lists the differing keys.

## Source-backed API surface (MASY list, `contracts.MASY_REQUIRED_API`)

Only names are checked. A function requirement must be a non-class callable whose signature has the listed parameters. A dataclass requirement must be a dataclass type declaring the listed fields (so `None`, a plain class or an instance fails). A class requirement must bind the listed public module attribute to a class, whatever name the class was defined under: re-exports and aliases of an implementation class pass, while `None`, a function, an instance or a missing attribute fails. That is what a valid `isinstance` target needs. Module-origin and source-root checks are separate and unchanged. Nothing is called, constructed or instance-checked, and no configured value is read: the reroute threshold is required as the `reroute_threshold_m(context)` API and the `TopographicCommitSpec.pending_elevation_trigger_m` field, never as a MASY constant. Each result carries `required_by` (the observed caller, where useful), and a failure's finding detail names it.

Lines cite the archived clean `source_context` (MAPLE `f1555e0`, SYRUP `9fabdca`). Codex confirmed that MAPLE `core/types/voxel.py` and SYRUP `sediment_bed.py` are byte-identical in the current pair (MAPLE `5b1a052`).

| Symbol | Kind and required names | Source (observed clean context) |
|---|---|---|
| `maple.aeolian.scheduler.block.advance_wind_block` | function `state, block, config` | MAPLE `aeolian/scheduler/block.py:426` |
| `maple.aeolian.scheduler.state.initial_scheduler_state` | function: the six bed objects, `time_s`, `coupled`, `vegetation`, `sediment_availability`, `water`, `coupled_trajectory_semantics` | MAPLE `aeolian/scheduler/state.py:1125-1138` |
| `maple.aeolian.scheduler.state.SchedulerPhysicalState` | dataclass: `voxel_column, active_layer, mobile_state, ledger, committed_topography, threshold_modifier_state, time_s, vegetation, sediment_availability, water, coupled_trajectory_semantics` | MAPLE `aeolian/scheduler/state.py:300-377` |
| `maple.aeolian.scheduler.state_validation.validate_event_boundary` | function `physical, coupled, fluvial, geometry, grain_classes` | MAPLE `aeolian/scheduler/state_validation.py:961` |
| `maple.aeolian.cohorts.dispatch.mobile_state_mass_is_finite`, `mobile_state_is_genuinely_empty` | functions `mobile_state` | MAPLE `aeolian/cohorts/dispatch.py:491,508` |
| `maple.surface.topographic_commit.commit.commit_topography` | function `current_time_s, water_callback, routing_callback` | MAPLE `surface/topographic_commit/commit.py:1101` |
| `maple.core.parameters.topographic_commit.TopographicCommitSpec` | dataclass `pending_elevation_trigger_m` | MAPLE `core/parameters/topographic_commit.py:37,54` |
| `maple_syrup.sediment_bed.BedState` | dataclass `voxel_column, active_layer, water, ledger, committed_topography, sediment_availability` | SYRUP `sediment_bed.py:28-35` |
| `maple_syrup.sediment_bed.BedContext` | dataclass `geometry, grain_classes, mass_resolution_kg, topographic_commit_spec, initial_available_fraction` | SYRUP `sediment_bed.py:38-45` |
| `maple.core.types.voxel.VoxelColumnState` | class (the genuine Dense voxel column) | MAPLE `core/types/voxel.py:13-14` |
| `maple.core.types.voxel.WindowedVoxelColumnState` | class, required by the current SYRUP Dense path (see below) | imported at SYRUP `sediment_bed.py:610-613`; **not defined** in the observed MAPLE `core/types/voxel.py` |
| `maple_syrup.sediment_bed.is_windowed_bed`, `bed_inventory`, `bed_by_cell_class`, `bed_voxel_count` | functions `state` | SYRUP `sediment_bed.py:610,264,616,624` |
| `maple_syrup.sediment_bed.apply_bed_demand` | function `state, context, demand` | SYRUP `sediment_bed.py:186` |
| `maple_syrup.sediment_bed.bed_already_committed` | function `state, context, time_s` | SYRUP `sediment_bed.py:628` |
| `maple.aeolian.flux.step.bed_inventory_by_class_kg` | function `voxel_column, active_layer` (Dense branch of `bed_inventory`) | MAPLE `aeolian/flux/step.py:325` |
| `maple.water.apply_water_process_demand` | function: the four bed objects, `demand, geometry, grain_classes, mass_resolution_kg, sediment_availability, initial_available_fraction, adapter_name, detachment_integration` (Dense branch of `apply_bed_demand`) | MAPLE `water/step.py:155-169`, re-exported `water/__init__.py:40` |
| `maple.surface.topographic_commit.commit.is_already_committed` | function `ledger, committed_state, voxel_column, active_layer, geometry, mass_resolution_kg, current_time_s` (Dense branch of `bed_already_committed`) | MAPLE `surface/topographic_commit/commit.py:795-805` |
| `maple_syrup.sediment_bed.reroute_threshold_m` | function `context` | SYRUP `sediment_bed.py:391` |
| `maple_syrup.sediment_bed.current_routing_surface_m`, `net_routing_displacement_m`, `commit_bed` | functions | SYRUP `sediment_bed.py:401,430,495` |
| `maple_syrup.sediment_event.initial_event_state` | function, takes a nonzero `t_s` | SYRUP `sediment_event.py:746` |
| `maple_syrup.complete_event.DryHandoff` | dataclass `progress, dry_state, pre_reset_surface_m, pre_reset_soil_m, terminal_deposition_kg, accounting, status` | SYRUP `complete_event.py:206-215` |
| `maple_syrup.complete_event.complete_event` | function `state, context, max_end_s, control, continuation` | SYRUP `complete_event.py:218-221` |
| `maple_syrup.complete_event.finalize_event` | function `progress, context, column` | SYRUP `complete_event.py:339` |

Passing these checks says only that the names exist. The scientific mapping and ownership of the bed between processes, the native adapters, bed exchange, availability and prescribed inputs, binding of the reroute threshold, mutation/staging boundaries, restart and conservation are all still unqualified.

The optional module `maple.water.windowed_transaction` is the windowed paired transaction behind SYRUP `sediment_bed.py:214-231`. It is located with `find_spec` and not imported. If it is absent, an `info` finding is recorded and the dense audit does not fail.

### Dense-path type dependency (finding 11)

The observed SYRUP `is_windowed_bed` (`sediment_bed.py:610-613`) runs `from maple.core.types.voxel import WindowedVoxelColumnState` unconditionally, before it looks at the column. `bed_inventory`, `apply_bed_demand`, `bed_by_cell_class`, `bed_voxel_count` and `bed_already_committed` call it first, so **Dense** beds reach that import too. The observed MAPLE voxel module defines only `VoxelColumnState`, so those Dense paths raise `ImportError` before classification. Codex reproduced this on the archived, explicitly identified source copies with a plain object; no state, array or transport was loaded.

Despite its name, this type is therefore a required interface of the currently observed SYRUP caller, and MASY lists it as a required class. Its absence is reported as `masy_required_api_missing` (`incompatible`), with the caller named in the detail. That differs from the optional paired-transaction module above, whose absence alone stays `info`. Requiring the type does **not** make full Windowed storage required or supported for Dense, and nothing here exercises it. On the current pair the overall verdict is still `refused`, because the native metadata-origin guard also fails; both findings stay in the report. SYRUP's native `check_required_api()` can still pass, because its upstream list does not cover this import.

The requirement is tied to the observed SYRUP caller. An owning-SYRUP change, reviewed independently, may remove the import: the accepted design proposal recognizes the genuine Dense MAPLE class without importing an absent optional Windowed class, refuses unknown/foreign representations, and leaves the deep-storage/source/transaction guards unchanged. That proposal is design-only: no upstream code is changed in MASY-01 and its classifier is not copied into MASY. If such a change lands, a newly qualified MASY requirement update reflects the actual source change. Until then, MASY does not synthesize the class, patch modules or relax any private/deep-storage/source guard.

## Standing blocking findings (always present)

Each of these findings is present in every report:

- `immutable_import_unverified`: the imports are live working-tree imports. MASY-01 implements no bundle, snapshot or metadata overlay.
- `transport_not_implemented`: no simulation entry point exists.
- `alternating_physics_unqualified`: no alternating event has been executed or validated.
- `wind_mobile_completion_unqualified`: the candidate is zero-input-wind native stepping under finite budgets. It has not been probed.
- `routing_reentry_unqualified`: reuse of the retained SYRUP graph and routing source after wind needs a validated adapter.
- `cache_invalidation_unqualified`: no adapter yet invalidates MAPLE scheduler caches after water.
- `native_restart_unqualified`: no combined restart exists.
- `forcing_adapter_unqualified`: forcing edge resolution is data-only.

## Report publication

`report.json` is canonical JSON: sorted keys, compact separators, ASCII, and no NaN or Infinity. `report.json.sha256` holds its digest.

Each file is written to an `O_EXCL` staging file and then hard-linked into place. A link never overwrites an existing file. `max_report_bytes` is a hard bound on `report.json`. If the report would exceed it, a minimal `refused` report (`report_oversized`) that names the original verdict is published instead. If even that minimal report does not fit (roughly 300 bytes), nothing is written, the refusal is printed to stderr and the exit is 2. The fixed-size `.sha256` sidecar is not counted.

Compiled identities in the environment record (interpreter, NumPy core) are hashed in bounded chunks. The path stat, the open descriptor and a final stat must agree on device, inode, size and mtime, and the bytes read must equal the size. Otherwise `sha256` is `null` with a reason; no digest is reported for content that changed during the read.

## Next bounded milestone (MASY-02 proposal, not implemented)

1. Build an immutable runtime with the upstream `maple_syrup.dependency_bundle` mechanisms. This means a scoped export of the MAPLE project (no `.git`, no datasets) and a matching SYRUP bundle. The bundle's distribution overlay must truthfully identify the source root, so the unmodified resolver can pass.
2. If an unmodified bundle cannot express the required metadata, a narrow upstream delivery extension is required. That extension must be shown necessary, then reviewed.
3. Verify the bundle with `check-runtime` and the upstream bundle/origin manifests. The current checkout-based observation cannot verify a bundle (no `.git`); that needs a separate, bounded locked-runtime verification mode, designed and reviewed on its own. Only after that, probe the wind-completion, routing-reentry and cache-invalidation adapters, one at a time.
