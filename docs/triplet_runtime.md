# Committed daily triplet runtime

This new, opt-in source runtime keeps the historical three-source runtime and
all previously qualified source profiles unchanged. It provisions code only.
Dense NumPy CPU is the sole profile; this lock alone grants no transport,
calibration, GPU, windowed or combined-restart acceptance.

Root supplies an independently qualified descriptor and its SHA-256. There is
no default descriptor, zero-pin fallback, automatic hash blessing, installation,
fetch, pull, checkout change or in-place upgrade. The descriptor chooses exact
local branch refs; it does not mean the active checkout HEAD or search for a
commit by date. If a selected ref advances, a new run refuses until the new
source frontier has been qualified.

The descriptor schema is `masy.triplet_qualification.v1`, with exactly:

- `schema`, `profile=masy.daily_wind_water_growth.cpu.v1`, `status=qualified`;
- `sources`: exactly `maple`, `syrup`, `leaf`, `masy`; each has canonical `repo`,
  explicit `refs/heads/...` `ref`, `commit`, `tree`, sorted complete `files`
  rows `[relative_path, bytes, sha256, git_blob]`, and `package_sha256`;
- `interpreter`: `path=/home/okin/MAPLE/.venv/bin/python`, real-binary `sha256`,
  exact three-integer `version`;
- `maple_bundle`: canonical `target` and authenticated materialized `manifest`
  reference `{path,size,sha256}`; `native_profile` names the independently
  qualified owning SYRUP source profile (the current additive spelling is
  `native_maple_1dca_case_latitude_cpu_v1`; it remains an explicit operand);
- `limits`: all keys of `triplet_lock.DEFAULT_LIMITS`, positive and no larger
  than those ceilings; `entrypoints`: at most 16 qualified `masy.module:main`
  targets. An empty roster enables source verification only.

`triplet_lock.resolve` returns explicitly **UNQUALIFIED** observations for
Root's review, never authority. `draft_qualification` assembles those into a
complete descriptor with `status=unqualified`, which prepare refuses. Root
must independently qualify the final source frontier and supply its external SHA. Prepare requires the independently supplied
external descriptor SHA. It compares every committed Git blob, hashes selected
bytes before/after copying, and checks refs/tree/active tracked-byte observations
again. Git fsmonitor is disabled; status/diff/clean-filter commands are not
used. Active tracked files are hashed directly, index state is recorded through
ls-files, and lazy fetch is disabled. Dirty active work is visibly recorded and
excluded under the explicit committed-source selection; it is neither overwritten nor imported. Untracked
paths are recorded in the observation, but their contents are not read.

The fixed scopes are MAPLE/SYRUP `src` and `pyproject.toml`; MASY also includes
`tools/masy_triplet.py`; LEAF uses flat `leaf`, `pyproject.toml` and exactly
`benchmarks/spatial_mapping.py`. A concrete `benchmarks/__init__.py`, another
participating namespace root or another helper is refused. The helper is
separately recorded because it is outside LEAF's package digest.

Prepare creates a fresh destination and four read-only content-addressed source
objects. File/byte/read/manifest/Git-output and known disk frontiers are bounded
before mutation. An incomplete failed preparation remains evidence; it is not
repaired or reused. The genuine MAPLE materialized bundle is reused explicitly:
its full Source table must equal the selected committed MAPLE scope. No custom
MAPLE distribution is generated. SYRUP/LEAF/MASY get labelled static metadata
overlays, not wheels/installations. Their project versions and dependency/script
metadata come from the selected committed `pyproject.toml`; the complete source
file remains pinned even when a descriptive readme is not copied into METADATA.

```
/home/okin/MAPLE/.venv/bin/python -I -B tools/masy_triplet.py prepare \
  --qualification QUALIFIED.json --qualification-sha256 EXTERNAL_SHA \
  --destination NEW_CODE_ROOT
/home/okin/MAPLE/.venv/bin/python -I -B tools/masy_triplet.py verify \
  --lock NEW_CODE_ROOT/triplet_lock.json --lock-sha256 EXTERNAL_LOCK_SHA
```

These commands do not import native models. `verify` verifies code, bundle rows,
interpreter and current selected refs; its output says native runtime has not
been checked. Root must subsequently use `triplet_runtime.bootstrap_spec` under
its existing bounded controller. This returns argv/environment and launches
nothing. The cold worker uses `-I -B -X pycache_prefix=NEW_EMPTY_DIRECTORY`, forbids
preloaded project/provider/helper modules and CuPy, and first authenticates
only MASY. Its stdlib verifier closes all other Source and metadata objects
before adding them after stdlib and before site-packages. MAPLE source,
distribution and editable trees have exact file/directory sets: extra modules
and bytecode refuse. The comparison uses the existing owning bundle metadata
recipe without creating/replacing a MAPLE distribution. The genuine
`dependency_bundle.check_runtime` runs before the explicit MAPLE import; then
`native_case_profile.verify_source_profile` runs. The worker then verifies four package digests,
actual selected distribution origins, loaded module Source hashes and helper.

A qualified run entrypoint has `main(argv, *, runtime)` and receives a
`RuntimeContext` with the genuine `source_profile`, scalar `record()` and
`recheck()`. The daily adapter must join its actual LEAF vegetation binding's
external-helper record to `runtime.record()['external_helpers']`. Rechecks during
transport use immutable code/native checks without Git subprocesses; the parent
and cold worker check ref currency before native startup. No run configuration, physics or serialization
format is invented here. Root remains responsible for actual process limits,
current Native/LEAF compatibility and model qualification.

Each bootstrap_spec call reserves a fresh empty run_dir/pycache directory;
reuse refuses. It still launches nothing. Root uses the returned command once
under its existing bounded controller. Source-only prepare/verify may be
repeated without a native/model run; actual Native bootstrap remains a
separate required Root check.
