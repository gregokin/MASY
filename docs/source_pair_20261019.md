# Current Dense CPU pair: October 19, 2026

MASY-09 independently accepted the new committed source pair. Claude reviewed the SYRUP candidate and the
qualification code; Codex inspected actual changes, executed the checks and verified the final evidence.

| Component | Committed revision |
|---|---|
| MAPLE | `39bcf79e1e67b6ed960f78a923ddfedf1d173e68` |
| SYRUP | `d05d8c6c66d9fd2969c007681916254e8d1c2c7c` |
| MASY runtime | `22aa6ab65ad73a58fb372ffe09e3f7e0ca9893a1` |

Immutable lock: `43976b9a6c3aa5384d0cdfdcf7c61f90d6a2f0ced45a0890e92dad5fb6a5cd33`.
The [pinned workspace configuration](../config/dense_cpu_20261019_syrup_shared.toml) records exact upstream pins,
existing interpreter, retained source-built metadata and runtime limits. Its bytes are hashed; it is a resolver
configuration, separate from the physical case inputs captured by the runner. No environment was installed or changed.

## Verified scope

| Independent check | Passed | Additional subtests |
|---|---:|---:|
| SYRUP candidate shared-storage compatibility, before commit | 96 | 0 |
| Native finalization/origin and unsupported-storage guards on the new lock | 113 | 0 |
| Complete bed, availability and bounded staging | 15 | 38 |
| Alternating Dense runner and independent qualification extensions | 22 | 23 |
| Native SYRUP wet checkpoint continuations | 3 | 0 |

No failures or skips in these selected checks. The new-lock total is 153 tests and 61 subtests; the 96 focused
component checks are separate. This does not claim a new full upstream CPU/GPU suite run.

The nonempty wind → water/recession → wind → water/recession case matches direct native execution, including
persistent bed fields, exact native clocks, second wind origin at 169 seconds, native final settlement, class budgets,
water accounting and retained routing. An injected second-wind failure leaves the last accepted water boundary intact
and blocks the next water event; a fresh bounded native stage demonstrates recoverability. The stopped wrapper remains
halted. The three native water restart checks establish selected upstream continuation, not combined persisted restart.

## Committed candidate and provenance

The user-authorized SYRUP commit contains exactly the reviewed 26 files; every committed file matches the accepted
candidate, and all 65 runtime/config hashes remain unchanged. Existing production functions changed only at the named
exact-source allowlist in `check_deep_storage_source`; the new memory-admission helper is opt-in. Physics, tolerances,
rerouting rules and defaults are unchanged. SYRUP's historical qualification lock/docs still describe the original
uncommitted acceptance; they are retained as historical records. This publication and its new MASY lock establish the
committed identity. The commit was pushed to `origin/main` on October 19; GitHub HEAD was verified as the exact accepted revision.

All 361 captured test/helper files match exact committed Git blobs and the previously accepted selected-source hashes.
Expected test-name arrays are unchanged from MASY-08. Actual ancestor conftest contents and dispositions are compared
against literal reviewed artifact hashes; the intentional SYRUP pytest testpath addition is explicit. Collection and call
lists match, with their harness, source, interpreter, lock, lifecycle and metadata identities verified.

The genuine MAPLE metadata built for the unchanged `39bcf79` source in MASY-08 is reused through the unmodified native
bundle mechanism. A fresh audit verifies all eight base requirements, enabled `dask[array]` dependencies, selected
metadata, and the native editable-overlay bytes. The existing installed MAPLE metadata remains stale for the optional
windowed extra; the qualification binds actual source-built metadata, not that editable installation.

All seven preceding locks, including MASY-08 `ee190c81…`, remain unchanged. Normal preparation selected current clean
local committed HEADs; source objects, native source guard, module origins and selected metadata were checked before
and after transport. Fresh explicit-lock verification passed. Later development uses separate candidate snapshots and
cannot silently rebind this run. Any new commit, amend, rebase or squash requires its own provenance/qualification.

```bash
/home/okin/MAPLE/.venv/bin/python -I -B /home/okin/MASY/tools/masy_runtime.py \
    prepare --config /home/okin/MASY/config/dense_cpu_20261019_syrup_shared.toml
/home/okin/MAPLE/.venv/bin/python -I -B /home/okin/MASY/tools/masy_runtime.py \
    verify --store /home/okin/MASY/runs/runtime-store \
    --lock 43976b9a6c3aa5384d0cdfdcf7c61f90d6a2f0ced45a0890e92dad5fb6a5cd33
```

These commands resolve/verify sources; transport qualification comes from the separate Python API checks.
`prepare` observes the then-current committed MASY HEAD. A later documentation or implementation commit therefore
produces a distinct lock; it does not rebind the accepted MASY `22aa6ab` runtime above. The explicit `verify` command
retains that original accepted lock.
Complete receipts, exact commands, raw separate streams, source-only review hashes, lifecycle/reaping, failed attempts,
finding dispositions and final acceptance are in
[`masy09_syrup_commit_qualify_20261019`](../agent_handoffs/tasks/masy09_syrup_commit_qualify_20261019/verification.md).
The [previous MASY-08 qualification](committed_pair_qualification.md) remains historical evidence.

## Remaining work

This accepts the small Dense NumPy CPU baseline wind/conservative upwind-array water profile. Production reduced wind,
FFT/separation, creep/avalanching, GPU/windowed coupling, combined persisted restart, ecological response physics,
large-case performance and a collaborator transport CLI have separate acceptance work. Eleven owner-reported older
native restart/unrelated assertion failures remain explicitly excluded, with no overlap in selected test files;
MASY-09 did not claim to reproduce or fix them.

Current parallel tasks and the user's physics-question requirement are recorded in
[project authorization](../agent_handoffs/project_authorization_20261019.md). The
[shared vegetation plan](vegetation_information_plan.md) reserves native vegetation and soil-water updates without
selecting a growth/mortality/uptake law or changing the current dry-reset policy.
