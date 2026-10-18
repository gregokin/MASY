# MASY-03 locked CPU runtime (`prepare` / `verify`)

**Status: implemented but not yet tested or accepted.** The code below has not been executed by its author. Synthetic tests (`tests/test_runtime.py`) are still to be written, and no actual prepare or verify has run. Nothing here is a transport, restart, science, GPU or performance result.

The runtime tool seals the current committed MAPLE, SYRUP and MASY sources into a content-addressed store. It produces a frozen MAPLE bundle with SYRUP's own `maple_syrup.dependency_bundle` and checks the native interfaces in a fresh isolated child. It then publishes a lock. `verify` re-checks one explicit lock later. The output is CPU runtime identity and native interface delivery only. Every output says `transport_ready: false`, has `transport_entry_point: null` and lists the unimplemented pieces. No transport, adapter or event runner exists.

## Invocation

Use the existing MAPLE CPU interpreter. The configured spelling is kept as written (see [Interpreter](#interpreter)).

```
/home/okin/MAPLE/.venv/bin/python -I -B /home/okin/MASY/tools/masy_runtime.py \
    prepare --config /home/okin/MASY/examples/runtime.example.toml

/home/okin/MAPLE/.venv/bin/python -I -B /home/okin/MASY/tools/masy_runtime.py \
    verify --store /home/okin/MASY/runs/runtime-store --lock <lock_id> [--max-lock-bytes N]
```

- `-I -B` are mandatory, and the launcher refuses without them. `--help` is informational: it exits 0 and creates or imports nothing.
- `--lock` takes either a lock id or `<store>/locks/<lock_id>.json`.
- The store's parent must already exist. The store itself and its fixed layout are created by `prepare`, but parents never are. For the default store, run `mkdir -p /home/okin/MASY/runs` once (`runs/` is Git-ignored).
- One canonical JSON object is printed on stdout (schema `masy.runtime_output.v1`).

| Exit | Verdict | Meaning |
|---|---|---|
| 0 | `runtime_locked` / `runtime_verified` | A lock was published or reused, or the explicit lock re-verified. `transport_ready` is still `false` |
| 2 | `refused` | A guard failed. Launcher-level refusals print to stderr only, before anything is created |
| 3 | `incompatible` | A native interface, registry or API requirement failed |
| 1 | (none) | Internal error. No verdict is trusted |

## Launcher (`tools/masy_runtime.py`)

The launcher uses only the stdlib until it hands over. In order, it:

1. refuses unless the interpreter runs with `-I -B`, and unless no `masy`, `maple`, `maple_syrup` or native-stack module (`numpy`, `numba`, `llvmlite`, `jax`, `jaxlib`, `scipy`, `cupy`) is loaded yet;
2. finds the store: `verify --store`, or the config's `[runtime].store` for `prepare` (default `/home/okin/MASY/runs/runtime-store`). The store must be an absolute, normalized path. `masy.runtime` re-parses the config and refuses if its store differs;
3. creates a fresh run id (`<command>-<UTC stamp>-<pid>-<random>`) and sets `sys.pycache_prefix` to the future `<store>/runs/<run_id>/cache/pycache`. With `-B` that directory is never created, nothing is written and no stale bytecode is read;
4. records the launch identity and hands over to `masy.runtime.main`. The record holds the launcher realpath and sha256, the MASY root, store, run id, argv, `sys.executable` as invoked, `sys.prefix`/`base_prefix`, flags and the `sys.path` insertion of this tree's `src`. `masy.runtime._check_launch` then re-checks the flags, the pycache prefix, MASY module origins and that no upstream module is loaded.

The parent never imports `maple`, `maple_syrup` or native packages. That is re-checked after `main` returns, and a violation turns the exit into `1`. Every upstream call runs in a bounded child.

## Selection and sealing (`prepare`)

- **Selection.** The current local committed HEAD of MAPLE and SYRUP (configured roots) and of the MASY tree that holds the launcher. An optional `expected_revision` pins each one. There is no development snapshot and no fallback. Uncommitted, untracked, ignored-in-scope, symlinked or special files in the scope refuse, through the existing `sourceobs.observe_upstream` guards. That includes MASY's own `src/`, `pyproject.toml` and `tools/masy_runtime.py`, so a default prepare refuses uncommitted runtime code, the wrapper included.
- **Scope.** Exactly the committed `src/` and `pyproject.toml` blobs (plus MASY's launcher), hashed from disk and checked against the Git blob ids. `.git`, `.venv`, cases, outputs and live `*.egg-info`/`*.dist-info` are never copied. Live generated metadata is recorded in the observation only.
- **Before/after.** The source observation and the installed MAPLE dist-info record are taken before anything is created and again after the last child. Any difference refuses, and no lock is published.
- **Disk preflight.** A conservative estimate must fit `store_max_new_bytes`, and the free space must cover the estimate plus `disk_reserve_bytes`. Both are checked before the store, a run directory or any object is created. Per-user quotas invisible to `statvfs` are not detected.

## Metadata provenance (no installation)

- **MAPLE**: SYRUP's unmodified `export_bundle` / `verify_bundle` / `materialize_bundle` / `verify_materialized` / `check_runtime` run from the sealed SYRUP copy, with the installed MAPLE dist-info as input. The materialized cache's editable overlay is what the native resolver selects.
- **SYRUP and MASY**: run-owned metadata overlays (`<object>/editable/<name>-<version>.dist-info`) generated from the archived static `[project]` table. Each carries `MASY_OVERLAY.json`, which labels it `not_an_installation` and binds the object id, generator sha256 and pyproject sha256. Only `name`, `version`, `description`, `requires-python`, `dependencies`, `optional-dependencies` and `scripts` are accepted; `dynamic` or any other key refuses. The version must be canonical PEP 440 and equal the package's literal `__version__`.
- MASY's own [pyproject.toml](../pyproject.toml) is static: `masy` version `0.0.3.dev0`, with no dependencies (a `maple` requirement would name an unrelated PyPI project) and no console scripts. Its `[build-system]`/`[tool.setuptools]` tables are declarative only. setuptools is not present in the CPU `.venv`, and nothing is built or installed.
- Nothing is pip-installed, built, downloaded or garbage-collected.

## Store layout

```
<store>/
  objects/sources/<object_id>/      read-only: manifest.json, source/ (scoped copy), editable/ (SYRUP/MASY overlay)
  objects/maple-bundles/<bundle_id>/       upstream export (bundle-manifest.json + archive)
  objects/maple-materialized/<bundle_id>/  upstream materialized cache (+ editable overlay)
  locks/<lock_id>.json              canonical lock (stable identity only)
  runs/<run_id>/                    stages.jsonl, evidence.json, requests, raw worker outputs, cache/, home/, tmp/
```

- Object ids bind the source bytes, the metadata generator identity and the overlay schema.
- Objects are staged, fully verified, frozen and published with `renameat2(RENAME_NOREPLACE)`.
- An existing object is fully re-verified and reused, never repaired, overwritten or deleted. Locks are published with `O_EXCL` plus link: identical bytes are reused and different bytes refuse.
- Old locks, objects and runs are retained, with no automatic GC.
- **Location rules.** The store may not overlap the MAPLE/SYRUP roots (ancestors and descendants included), the installed dist-info, the interpreter or its venv; symlink aliases are checked. Inside the MASY tree the store must lie strictly under `runs/`, outside `src/`, `tools/`, `pyproject.toml` and `.git`. It may not equal the MASY root or one of its ancestors, and symlinked store components refuse.

## Verify (old-lock re-verification)

`verify` reads only the explicit lock and the store. It never reads current upstream repositories and never selects anything new (`asserts_current: false`). It:

1. reads the lock with a bound and validates it strictly;
2. fully re-verifies every referenced object and the bundle manifests;
3. checks the sealed bootstrap hash and the interpreter identity;
4. runs one fresh `runtime` child from the lock's own sealed code;
5. compares the closed stable identity strictly and re-verifies the objects again.

Per-run operational values (run directory, request path, pycache prefix, pid, `HOME`, `TMPDIR`, `XDG_CACHE_HOME`, `NUMBA_CACHE_DIR`, `MPLCONFIGDIR`) are validated against the new run directory and saved in that run's receipts. They are never compared with an older run. Shadowed, unselected installed metadata is recorded, but a missing live upstream egg-info does not invalidate an old lock. Source, package, selected metadata, interpreter, dependency-version and compiled identities stay strict.

Two fresh `verify` runs of one lock are expected to succeed independently. That has **not** been demonstrated yet.

## Worker child

`<python> -I -B -X pycache_prefix=<run>/cache/pycache -c BOOTSTRAP <request>` runs with an environment built from nothing: `PATH=/usr/bin:/bin`, `LC_ALL=C`, run-owned home/tmp/caches, `NUMBA_DISABLE_CUDA=1`, `CUDA_VISIBLE_DEVICES=""` and `JAX_PLATFORMS=cpu`.

- The bootstrap refuses preloaded `maple`/`maple_syrup`/`masy` modules, inserts the explicit sealed `sys.path` prefix, then imports the sealed worker.
- The worker checks module origins for all three package families, the selected `importlib.metadata` distributions, package digests and the bundle tool hash.
- It runs the upstream `check_runtime` on the frozen cache and MASY's `contracts.run_native_audit` (unchanged).
- It calls both native checkpoint registries (`checkpoint._registry`, `pond_checkpoint._native_registry`) as class-dictionary probes only, with no state, arrays or solver steps.
- It records environment versions and compiled identities of the existing environment with count and byte bounds.
- Output, wall time and process groups are bounded. Process groups are killed and reaped on timeout, overflow or error. Incomplete output is never treated as success.

File hashes in parent and worker require the lstat before open, the fstat after open, the fstat after the last read and the final lstat to agree on device, inode, mode, size, `mtime_ns` and `ctime_ns`. A replaced file or a same-size in-place write during the read refuses. Residual limit: this relies on the filesystem updating timestamps on write. A writer that preserves both timestamps, or a filesystem with coarse timestamps hit within one tick, is not detected.

## Failure evidence

- Every stage is appended to `<store>/runs/<run_id>/stages.jsonl` as it happens.
- A refusal keeps the run directory, private staging, raw worker stdout/stderr and `evidence.json`. The printed output names `code`, `detail`, bounded `evidence`, `run_dir` and `evidence_path`.
- If even evidence cannot be written (for example `ENOSPC`), the failure is listed in `log_failures` and printed to stderr.
- Launcher refusals happen before any directory exists and print to stderr only.

## Interpreter

The configured spelling (`/home/okin/MAPLE/.venv/bin/python`) is used verbatim to launch children. It is a legitimate symlink to the base executable, and resolving it would lose the venv. Its realpath, size, sha256 and venv (`pyvenv.cfg`) are recorded and checked. Other symlinks in sources and the store still refuse. Only the CPU `.venv` is in scope. `.venv-cupy` and GPU use are not qualified.

## Standing limitations

- `transport_ready` is always `false`, and there is no transport entry point. Not implemented: wind adapter, water adapter, event-schedule runner, transition accounting, combined restart, routing-reentry adapter, cache-invalidation adapter, wind mobile completion, case input hashing, windowed storage qualification, GPU/`.venv-cupy` qualification, performance qualification and physical science qualification.
- The native checkpoint registries are probed by class identity only. Dense checkpoint save/load is not exercised here.
- Third-party wheels and native libraries come from the existing environment. They are identified by version and hash, not bundled, and not independent of that environment.
- Existing run or checkpoint sources are never rebound or upgraded.
- The checkout-only preflight ([preflight.md](preflight.md)) is separate and unchanged.
