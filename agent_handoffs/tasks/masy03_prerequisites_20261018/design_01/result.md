# MASY-03 prerequisites: design (no edits, no execution)

I read all the files you named, plus SYRUP's `dependency_bundle.py`, `dependency.py`, `provenance.py`, the pond checkpoint and windowed paths, the existing tests, MAPLE's `sediment_ledger.py`/pyproject, and the actual installed MAPLE dist-info. Nothing was written and nothing was run, so the claims below about blockers are from reading the code only. Codex needs to reproduce them.

---

## Prerequisite A: SYRUP Dense checkpoints on current MAPLE

### Production change (registry logic only; codecs and solvers untouched)

**`checkpoint.py`.** Add one helper. It reuses the classifier's already-accepted binding rule `sediment_bed._maple_voxel_class(module, name, required=...)` instead of writing a second absence rule:

```python
def _native_ledger_types():
    """The ledger classes MAPLE's public module binds: SedimentLedgerState required; CompactSedimentLedgerState only when
    literally present in vars(module) (no module __getattr__), and then required to be a class."""
    import importlib
    from maple_syrup.sediment_bed import _maple_voxel_class
    ledger = importlib.import_module('maple.core.types.sediment_ledger')      # import failure propagates
    dense = _maple_voxel_class(ledger, 'SedimentLedgerState', required=True)  # absent: AttributeError; non-class: BedIntegrationError
    compact = _maple_voxel_class(ledger, 'CompactSedimentLedgerState', required=False)
    return (dense,) if compact is None else (dense, compact)
```

- `_registry()` uses `*_native_ledger_types()` in place of the two-name import.
- The `CommittedTopographyState`/`WaterState` imports and the `{t.__name__: t}` keying stay as they are.

**`pond_checkpoint.py`.** `_native_registry()` builds `(*_native_ledger_types(), CommittedTopographyState)`, imported from `maple_syrup.checkpoint`. The module already imports from there (line 96).

How each case behaves:
- **Compact really absent:** neither registry has the key.
  - Encode: a foreign object named `CompactSedimentLedgerState` is refused by the existing `registry.get(name) is not type(value)` check.
  - Decode: a `{"type": "CompactSedimentLedgerState"}` node is refused by the existing "unknown checkpoint/native type".
- **Compact bound to a non-class:** `BedIntegrationError`, raised before anything is staged.
- **Import failures:** a missing or broken module raises from `import_module` and propagates.
- **Dense class missing:** this changes from `ImportError` to `AttributeError`. That matches the classifier's accepted semantics, but it is a small behaviour change.

There is no broad `except ImportError`, no fake type, no fallback by name or duck type, and no schema change.

**Possible further blockers on current MAPLE.** I found none on the Dense path. The `WindowedVoxelColumnState`, `windowed_*` and `numpy_pairwise_order` imports in `pond_checkpoint.py` (lines 1309, 1315, 1426, 1461) only run on windowed paths. Codex should confirm by running `tests/phase6/test_checkpoint.py` and `tests/pond_checkpoint` on current MAPLE.

### Tests

**`tests/masy_compat/test_pond_windowed_compat.py`.** Remove the `ImportError` branch. The test always does a real Dense save → load → `assert_tree_equal` → continuation: advance both the paused and the reloaded state N steps and compare trees. It asserts the ledger type `is vars(sediment_ledger)['SedimentLedgerState']` and updates the docstring.

**New `tests/checkpoint_compat/`:**
- `conftest.py` binds this tree's `src`, the same way `masy_compat` does, and records which MAPLE was bound and whether it has the compact class.
- `test_registries.py`:
  - Both registries succeed. The Dense entry is the identical object to the public binding. The compact key is present exactly when `'CompactSedimentLedgerState' in vars(module)`. Both branches assert; neither is skipped.
  - Compact bound to `3`, a function, or an instance → `BedIntegrationError` from both registries. `save_checkpoint` and `save_pond_checkpoint` write nothing.
  - `sys.modules[...] = None` → `ImportError` propagates. Patched `importlib.import_module` raising `RuntimeError("broken")` → propagates unchanged.
  - Deleting the Dense class → `AttributeError`.
- `test_current_dense_restart.py`:
  - **Legacy:** `tests.phase6.test_complete_event.fixture` → wet `save_checkpoint` → `load_checkpoint` → native state equality (bed, ledger, committed topography, water), then continue both to `complete_event` and check equality.
  - **Pond:** same pattern with `pond_fixture`.
  - **Negatives:** in a legacy `checkpoint.json` payload, rename the ledger node type to `CompactSedimentLedgerState`; in a pond checkpoint, do the same and re-seal with `test_refusals.rewrite`-style helpers. Both are refused on either MAPLE: current MAPLE says unknown type; reference MAPLE says field mismatch. Also: a malformed ledger node (missing field, extra field) and a source-pair mismatch.
- `test_reference_compact.py`: a genuine compact ledger from reference MAPLE's own factory goes round trip through `_native_encode`/`_native_decode` with type identity kept. It skips explicitly when the bound MAPLE lacks the class. The existing `tests/pond_windowed_checkpoint`, `phase6` and `pond_checkpoint` suites stay the reference regressions, run by Codex on the content-cache reference root.

**New doc** `docs/current_maple_checkpoint_compatibility.md`: the binding table, error types, what changed from 2c60c73, and limits (Dense only on current MAPLE; existing checkpoints are not migrated, because the SYRUP digest changes).

`checkpoint_compat` is not in SYRUP's `testpaths`, because `pyproject.toml` is outside the writable scope. Codex runs it explicitly.

---

## Prerequisite B: MASY locked runtime prepare/verify

### What is reused vs. what MASY adds

**Reused from SYRUP's upstream bundle tool, unmodified:**
- `maple_syrup.dependency_bundle`: `export_bundle`, `verify_bundle`, `materialize_bundle` (it handles no-clobber and re-verifies an existing target), `verify_materialized`, `check_runtime`.
- `maple_syrup.dependency.resolve_maple_dependency` and `check_required_api`.
- `provenance.source_tree_digest`.
- These run only in child processes that import the sealed SYRUP copy, never the live checkout. The parent then checks that the bundle manifest's `tool.module_sha256` equals the sealed `src/maple_syrup/dependency_bundle.py` entry.

**Reused from MASY, unmodified:**
- `sourceobs`: `observe_upstream`, `run_git`, `Limits`, `SourceRefusal`, `canonical_json`, `diff_observations`.
- `importcheck`: `module_origin_violations`, `environment_record`.
- `contracts`: `run_native_audit`, `check_masy_api`.

**New MASY code (orchestration only, no physics):**
- a copy of exactly the committed `src/` + `pyproject.toml` (plus the MASY launcher) into content-addressed objects;
- labelled metadata overlays for `maple-syrup` and `masy`;
- isolated child processes and the lock;
- verification of an existing lock.

### Files

- **`pyproject.toml`** (new): `name="masy"`, `version="0.0.3.dev0"`, description, `requires-python=">=3.11"`, `dependencies=[]`. There are no `[project.scripts]`, so no CLI name can collide. maple/maple-syrup are deliberately not declared, for the same reason SYRUP gives.
- **`src/masy/__init__.py`**: `__version__ = "0.0.3.dev0"`. `0.0.0.masy01` is not a valid PEP 440 version. Prepare checks `__version__` (read from the AST, not imported) against the pyproject value.
- **`src/masy/runtime.py`** (parent; stdlib + `packaging`; never imports maple or maple_syrup):
  - Constants: `CONFIG_SCHEMA="masy.runtime.v1"`, `LOCK_SCHEMA="masy.runtime_lock.v1"`, `SOURCE_OBJECT_SCHEMA="masy.source_object.v1"`, `OVERLAY_SCHEMA="masy.metadata_overlay.v1"`.
  - Types: `RuntimeRefusal(code, detail, severity, **evidence)`, `RuntimeLimits`, `RuntimeConfig`.
  - Functions: `load_runtime_config`, `prepare(config, *, launch)`, `verify(store, lock_ref, *, launch)`, `seal_source`, `verify_source_object`, `generate_overlay(project, package, object_root) -> dict[str, bytes]`, `validate_lock`, `publish_lock`, `run_bounded(argv, *, env, timeout_s, max_output_bytes)`, `worker_sys_path(lock)`, `worker_env(run_dir)`, `main(argv, *, launch)`.
- **`src/masy/runtime_worker.py`** (child; imported only from the sealed MASY):
  - `main(argv)` with subcommands `bundle-export`, `bundle-verify`, `bundle-materialize`, `verify-runtime`. It reads a bounded JSON request file and prints one bounded JSON result.
  - Pure, testable helpers: `selected_distributions(names, path)`, `registry_probe()`, `compiled_extensions(modules, max_count, max_bytes)`, `preload_violations()`.
- **`tools/masy_runtime.py`**: modelled on `masy_preflight.py`. It requires `-I -B`, refuses if `masy` is already imported, does a bounded stdlib read of `runtime.store`, sets `sys.pycache_prefix` to `<store>/runs/<run_id>/cache/pycache`, puts the live `src` first, and calls `masy.runtime.main`.
- **`src/masy/contracts.py`** (minimal change):
  - `CompactSedimentLedgerState` becomes `kind="optional_class"`, with `required_by` naming the accepted SYRUP `_native_ledger_types` caller (Codex fills in the commit at adoption).
  - Add `SedimentLedgerState` as a required `class`.
  - Add a `checkpoint_registries` record to `run_native_audit` that calls the real SYRUP `checkpoint._registry()` and `pond_checkpoint._native_registry()`. It requires success and that the Dense entry is the public binding. It reports `checkpoint_registry_unavailable` (incompatible).
  - The probe is needed because otherwise the checkout preflight, now stricter, would wrongly pass against old SYRUP 2c60c73, where the registry still fails. It reads private names; that is documented.
- **Docs and tests:** `tests/test_runtime.py`, `tests/test_contracts.py`, `docs/runtime.md`, `examples/runtime.example.toml`, plus short pointer updates in `README.md`, `docs/preflight.md`, `docs/upstream_interface_audit.md` and `tests/README.md`.

### Config (closed schema; unknown keys refuse)

```toml
schema = "masy.runtime.v1"
[sources.maple]
root = "/home/okin/MAPLE"
distribution_metadata = "/home/okin/MAPLE/.venv/lib/python3.12/site-packages/maple-0.0.1.dist-info"
# expected_revision = "<full id>"     (optional; refuses on mismatch, never selects)
[sources.syrup]
root = "/home/okin/SYRUP"
[runtime]
python = "/home/okin/MAPLE/.venv/bin/python"
store  = "/home/okin/MASY-runtime-store"   # absolute, canonical, outside every repo/venv
[limits]   # defaults shown
git_timeout_s = 30.0
git_max_output_bytes = 8388608
max_files = 5000
max_bytes = 67108864
worker_timeout_s = 300.0
worker_max_output_bytes = 4194304
max_lock_bytes = 4194304
store_max_new_bytes = 1073741824
disk_reserve_bytes = 1073741824
```

- The MASY root is the launcher's own repository root. It is not configurable.
- The store must not equal, contain or sit inside any source root, MASY's `src`/`tools`, the venv prefix, or the dist-info directory.

### Store layout (content-addressed; nothing is overwritten or garbage-collected)

```
<store>/objects/sources/<sha256>/{manifest.json, source/{pyproject.toml,src/...}, editable/<dist>-<ver>.dist-info/...}
<store>/objects/maple-bundles/<bundle_id>/{bundle-manifest.json, maple-project.tar}     # upstream export output
<store>/objects/maple-materialized/<bundle_id>/...                                        # upstream materialize
<store>/locks/<lock_id>.json
<store>/runs/<run_id>/{stages.jsonl, evidence.json, request-*.json, worker-*.stdout|.stderr, cache/{pycache,numba,matplotlib}, home/}
```

**Source objects:**
- The object id is `sha256(canonical{schema, overlay_schema, kind, project name/version, pyproject sha, [(path, size, sha256)…]})`. It has no revision, so identical content is deduplicated. Revisions live in the lock.
- Files are 0444 and directories 0555.
- The MAPLE object has no overlay; it is the scoped project handed to `export_bundle`.
- Because its `source/` path is deterministic, `exported_from`, and therefore the `bundle_id`, are stable across runs.
- The MASY object also carries `tools/masy_runtime.py` and appears as `masy` in the lock.

**Overlays for `maple-syrup` and `masy`:**
- Byte-deterministic files generated only from the archived pyproject: `METADATA` (2.1; Name, Version, Summary, Requires-Python, Requires-Dist with extras as `; extra == "x"`, Provides-Extra), `top_level.txt`, `entry_points.txt` from `[project.scripts]`, `INSTALLER="masy.runtime"`, `direct_url.json` (editable, file URL of `<object>/source`, mirroring upstream `_overlay_files`), `MASY_OVERLAY.json` (`kind: run_owned_metadata_overlay`, `not_an_installation: true`, pyproject sha, overlay schema), and `RECORD`.
- Refused: `dynamic`, or any `[project]` key outside {name, version, description, requires-python, dependencies, optional-dependencies, scripts}; a non-canonical version; a `__version__` mismatch.
- Verification regenerates the overlay for the object's final path and compares bytes exactly.

### Prepare: stage order, errors and publication

| Stage | Action | On failure |
|---|---|---|
| P0 | launcher guards; parse config | refuse; nothing created |
| P1 | validate the store (only `objects`, `locks`, `runs` plus `.staging-*` entries); create `runs/<run_id>` exclusively; create caches | refuse |
| P2 | **before** observation: `observe_upstream` for maple and syrup (clean scope, blob = HEAD, no untracked or ignored runtime input, no symlinks or special files). MASY: the same with `package="masy"`, plus `tools/masy_runtime.py` (status clean, blob = HEAD). Also record `HEAD^{tree}` and the dist-info file hashes. Check that the live MASY modules loaded in the parent hash-match their committed files | `dirty_scope`, `untracked_runtime_input`, `symlink_in_scope`, … |
| P3 | disk preflight: estimate ≈ 3×MAPLE scope + SYRUP + MASY + overlays + margin; require `statvfs` free ≥ estimate + reserve, and estimate ≤ `store_max_new_bytes` | `disk_quota_insufficient`, before any object is written |
| P4 | seal syrup, masy and maple-scope: copy only the `ls-tree` entries, computing the git blob id and sha256 from the same bytes that are written; any blob ≠ HEAD refuses. Stage in `objects/sources/.staging-<run_id>-<kind>`, freeze, verify fully, `renameat2(RENAME_NOREPLACE)` into the same parent. If the id exists, run the full `verify_source_object` and reuse it; a mismatch refuses and the existing object is never touched | `object_tamper`, `source_changed_during_read`; failed staging is retained |
| P5 | child A (sealed SYRUP on `sys.path`, no MAPLE): `export_bundle(maple-object/source, dist-info, runs/<id>/origin.json, objects/maple-bundles/.staging-<run_id>)` → `verify_bundle`. Parent checks the tool hash, then renames to `<bundle_id>` or verifies the existing one. Child A then calls `materialize_bundle(..., cache_root=objects/maple-materialized)`. The origin record is `kind="git_commit_scoped_copy"`, `revision=HEAD`, with an explicit reason for the empty patch chain | upstream `BundleError` recorded verbatim |
| P6 | child B `verify-runtime` (see below) | refused or incompatible; no lock |
| P7 | **after** observation must be identical (`diff_observations`), and the dist-info files are re-hashed | `source_changed_during_prepare` |
| P8 | build the lock; `lock_id = sha256(canonical body)`; run the same `validate_lock` that verify uses; write `locks/.staging-<run_id>.json` (`O_EXCL`, fsync, 0444), then `renameat2` NOREPLACE to `<lock_id>.json` and fsync the directory. If the name exists with identical bytes it is reused; different bytes refuse (`lock_collision`) | |
| P9 | read back and compare bytes; write `evidence.json`; print `{verdict:"runtime_locked", lock_id, lock_path, transport_ready:false, remaining_adapter_work:[…]}` | |

Every stage appends a record to `stages.jsonl` as it goes, so a failed run keeps its evidence. Exit codes: 0 `runtime_locked`/`runtime_verified`; 2 refused; 3 incompatible; 1 internal error.

### The child process (both prepare P6 and verify)

- **Command:** `<python> -I -B -X pycache_prefix=<run>/cache/pycache -c <BOOTSTRAP> <sys_path_json> verify-runtime <request>`.
  - `BOOTSTRAP` is a constant string whose hash is in the lock.
  - The environment is built from nothing: `PATH=/usr/bin:/bin`, `LC_ALL=C`, `HOME=<run>/home`, `NUMBA_CACHE_DIR`, `MPLCONFIGDIR`, `NUMBA_DISABLE_CUDA=1`, `CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu`.
  - stdout and stderr together are capped by size, with a timeout; the child is killed and reaped on either.
- **Exact `sys.path` prefix:** `[maple-materialized/editable, syrup/editable, masy/editable, masy/source/src, syrup/source/src, maple-materialized/source/src]`, followed by the interpreter's own entries.
  - Upstream's `runtime_environment()` passes `PYTHONPATH`, which `-I` ignores. That is why the ordering is built explicitly; this is documented.
  - Top-level names in the three `src` directories must not collide, and the `editable/` directories may hold only dist-info and `.pth` files.
- **Checks, in order:**
  1. flags isolated / dont_write_bytecode / pycache_prefix are set; the `sys.path` head is exactly as above; no `maple*` or `masy*` module is preloaded (site `.pth` hooks run under `-I`);
  2. `check_runtime(materialized target)`, which internally calls the unmodified `resolve_maple_dependency`;
  3. `run_native_audit(dependency, str(target/"source"))`, covering the resolver, SYRUP's `check_required_api`, the updated MASY API and the registry probe;
  4. `module_origin_violations` across every loaded `maple`, `maple_syrup` and `masy` module, against the sealed package directories;
  5. `selected_distributions`: `importlib.metadata.distribution(n)` for maple, maple-syrup and masy must sit in its overlay directory with byte hashes matching the lock. All other `distributions(name=n)`, such as the venv's `maple-0.0.1.dist-info`, are reported as `shadowed_installed_metadata`, not hidden;
  6. `source_tree_digest` of each imported package matches its object manifest;
  7. `environment_record()`, plus the sha256 of each loaded compiled extension (`.so`), bounded in count and size.
- Nothing is monkeypatched.

### Verify an explicit lock (`tools/masy_runtime.py verify --store S --lock <id|path>`)

1. **V1:** bounded read; strict schema; `lock_id` = the filename = `sha256(body)`.
2. **V2:** fully re-verify every referenced object. Source objects are checked by MASY; the bundle and the materialization are checked by upstream `verify_bundle`/`verify_materialized` in child A, which uses the lock's sealed SYRUP.
3. **V3:** the interpreter's realpath sha, version and prefix still match the lock.
4. **V4:** child B runs with the lock's `sys.path` and environment.
5. **V5:** the results must equal the lock's verification record. Any drift in packages, origins, selected distribution, numpy core or extensions refuses (`lock_runtime_mismatch` or `environment_changed`).

Verify never runs git against the configured upstreams. It reports `selection:"explicit_lock"` and `asserts_current:false`. Old locks and objects are kept.

### Lock contents (`masy.runtime_lock.v1`)

- `selection`: `{policy: current_local_committed_head, development_snapshot: false}`.
- Per source: root, head, tree, object_format, pyproject {sha, name, version}, tracked-scope and package digests, source object id. MASY also records the launcher blob.
- `objects`: the bundle id, archive sha and tool sha; the materialized path.
- `distributions`: for each, the overlay directory, file hashes and origin label. For MAPLE: "upstream dependency_bundle overlay of installed dist-info <path, file shas>".
- `runtime`: python identity, flags, exact `sys.path`, environment, bootstrap sha, pycache policy.
- `verification`: the full child B record.
- `limits`.
- `interpretation`: `{transport_ready: false, not_restart_physics_performance_gpu_qualified: true, remaining: [wind/water adapters, schedule runner, transition accounting, combined restart, routing re-entry, cache invalidation, wind mobile completion, case-input hashing, windowed/GPU/.venv-cupy qualification, performance]}`.

No simulation arrays or state go into a lock or object.

### MASY tests (`tests/test_runtime.py`, stdlib unittest; synthetic git repos made in a temp dir with authored commits)

- **Config:** unknown keys; relative paths; the store overlapping a root, `src`/`tools`, the venv or the dist-info.
- **Snapshot:** a clean repo seals to exactly the HEAD blobs; untracked `cases/` and `.git`/`.venv` are never copied. Refusals for: a tracked edit, an untracked file in `src`, an ignored `.so` in `src`, a committed symlink, a FIFO.
- **Concurrency:** an injected hook commits between before and after → `source_changed_during_prepare`, with no lock and evidence kept. A file edited mid-copy → blob refusal.
- **Objects:** reuse after verification; refusals for a tampered byte, an extra file, a symlink inside an object, a foreign object at the id name (left untouched), a failed staging (retained and ignored).
- **Overlay:** deterministic; METADATA parses back to the pyproject; refusals for `dynamic`, an unknown key, `0.0.0.masy01`, a `__version__` mismatch, a tampered METADATA; the overlay is bound to its path.
- **Lock:** self-hash; tamper; id ≠ filename; oversize; unknown key; identical reuse; a collision refuses with the original kept. Two commits give two locks, and after the synthetic repo is deleted the old lock still verifies, which shows verify never rebinds.
- **`run_bounded`:** timeout with kill and reap; output cap; bounded stderr on non-zero exit.
- **Disk preflight** with an injected `statvfs` refuses before any write.
- **Worker helpers:** overlay-first selection with a shadow reported; a missing overlay refuses; origins with fake modules; preload detection.
- **Child bootstrap** on a sealed copy of real MASY with no MAPLE gives structured refusal JSON. It records `isolated=1`, the pycache prefix and the exact `sys.path`, and writes no bytecode into objects.
- **Gated real-roots test** (`MASY_RUNTIME_REAL_CONFIG=<toml>`): prepare + verify of the actual pair. This is Codex's acceptance run.
- **`test_contracts`:** compact is `optional_class` (absent passes, non-class fails); Dense is required; registry probe pass, fail and error.

Expected cost: hashing about 3× the MAPLE `src` size plus small SYRUP/MASY copies, and two Python child processes importing numpy and MAPLE — seconds to tens of seconds. This is CPU and `.venv` only.

---

## Bootstrap circularity: how final qualification stays truthful

1. **Prepare refuses its own development tree.** It only accepts MASY at clean committed HEAD, so the uncommitted `runtime.py` is correctly `dirty_scope`. Development tests use synthetic repos.
2. **Pre-acceptance candidates.** Codex clones MASY and SYRUP into a temp directory, applies the exact diffs, makes a labelled local candidate commit (no push), and runs the clone's launcher with `sources.syrup.root` set to the SYRUP clone. The lock truthfully records the clone roots and candidate commits, so nothing is labelled as current main.
3. **After adoption.** Committing the same files gives the same `HEAD^{tree}` ids, which the lock records, so equivalence can be proven. A second prepare on `/home/okin/MASY` and `/home/okin/SYRUP` then produces the final lock, reusing the verified content-addressed objects.
4. **The bundle tool is part of SYRUP, the package being sealed.** It always runs from the sealed SYRUP copy, and its hash is cross-checked, so live code never seals itself.

## Disagreements and open issues

- **SYRUP commit authority.** Final qualification against "current" SYRUP needs Prerequisite A committed in SYRUP. SYRUP's AGENTS.md records that recent requests did not authorize SYRUP commits, and your commit/push instruction was about MASY-02B. Codex should confirm SYRUP commit scope. Without it, the final lock can only name the isolated candidate clone.
- **The existing preflight now mislabels MASY.** `sourceobs.observe_masy` reports `working_files_no_git_repository`, but MASY has been a Git repository since 204622d. I left it unchanged as scoped and noted it in `docs/preflight.md`. It needs a separate small fix.
- **Stale SYRUP docs.** `masy_dense_compatibility.md` and `masy02a_integration.md` still say checkpoints are blocked. They are outside the writable scope; the new doc supersedes them, and adding a pointer would need scope.
- **Resolver label.** The resolver will report `source_kind="editable_working_tree"` for the read-only materialized cache. That is upstream's term; the MASY report labels the cache `immutable_materialized_cache` next to it.
- **Read-only modes are not a security boundary.** 0444/0555 can be undone by the owner. The real protection is the full hash verification before every reuse or verify.
- **Behaviour change in A.** A missing Dense ledger class now raises `AttributeError` instead of `ImportError`, matching the classifier.
- **Registry probe uses private names.** It calls SYRUP's private `_registry` and `_native_registry`. If Codex prefers, SYRUP could expose a public wrapper later, as separate scope.
