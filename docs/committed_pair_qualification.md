# Committed Dense CPU source-pair qualification

MASY-08 independently accepted on October 18, 2026 (October 19 UTC). Claude accepted the corrected source-only
qualification, and Codex independently verified every execution and evidence condition; no blocking findings remain.
The qualification covers the existing `masy.dense_runner` Python API on the small authored Dense NumPy fixture.

| Component | Exact committed revision |
|---|---|
| MAPLE | `39bcf79e1e67b6ed960f78a923ddfedf1d173e68` |
| SYRUP | `b4ee14e5bf37814efb0d754de17b7ea604b1cd74` |
| MASY runtime | `22aa6ab65ad73a58fb372ffe09e3f7e0ca9893a1` |

New immutable lock: `ee190c813ae347a1f5c11e7ab5fad595dada898f3a13dddc775fca4488762a70`.
Preparation selected the actual clean local committed HEADs. Execution uses their immutable source objects and
matching distribution metadata; it does not import the live editable source trees. The six preceding locks,
including the qualified `b3e8756` pair, remain unchanged. Subsequent upstream revisions require fresh qualification.

## Actual checks

| Check | Passed | Additional subtests |
|---|---:|---:|
| Native wind finalization/origins and unsupported-storage guards | 113 | 0 |
| Complete evolving bed, availability, mapping and bounded staging | 15 | 38 |
| Dense event runner and two task-local qualification extensions | 22 | 23 |
| Existing SYRUP wet checkpoint/save/load/continuation checks | 3 | 0 |

No failures or skips. These are independently executed MASY checks, separate from the upstream owner's larger CPU
and GPU matrices. The 264-test MASY stdlib suite passed for the unchanged runtime implementation in MASY-06;
it was not rerun for this documentation/configuration task.

The genuine nonempty wind → water/recession → nonempty wind → water/recession sequence matches direct native
execution on an independently constructed bed. Assertions cover all persistent shared fields, exact native start/end
clock hex values, class budgets and exports, native finalizer deposition counted once, genuinely empty mobile state,
all nine native water-budget values, documented dry-water removal, and retained routing graph/source behavior.
The second wind's actual origin is 169 seconds. Native settlement tests also cover sub-resolution cohorts,
repeat-zero transfer, invalid origins and incomplete-event refusals. The native time tolerance is unchanged.

A deliberate second-wind failure proves writes occurred only in staged bed and mobile arrays. The accepted bed,
pool, soil and routing source retain their complete contents and references, time/cursor/ledger/receipts stay intact,
and the next water call never runs. A fresh bounded native stage from that accepted boundary successfully advances
and finalizes wind, matching an independent three-event replay. The stopped wrapper remains halted: this is evidence
that the accepted state is recoverable, not an implemented combined checkpoint/resume facility.

The three selected SYRUP tests exercise genuine wet legacy and pond checkpoint continuations, with exact persistent
state and resumed results. Native dense snapshot writers and the wind scheduler continue to refuse unsupported
windowed columns and compact wind ledgers. Constructing windowed fixtures for refusal tests does not qualify
windowed transport.

## Source and metadata preparation

The existing installed MAPLE metadata lacks the new optional `windowed` extra. Normal preparation correctly refused
it with `upstream_bundle_error`; this failed attempt is retained. No installed environment was updated.
A real offline wheel was then built from 352 exact committed source inputs (351 Python modules and `pyproject.toml`)
using the declared `setuptools.build_meta` backend, existing system setuptools 68.1.2 and a task-local, hash-verified
wheel 0.45.1 build-tool archive. The tool archive was downloaded from PyPI; it was not installed. All 351 modules in
the built wheel match the committed inputs byte for byte, and input hashes remain unchanged. Only its actual
`dist-info` is supplied to the normal native bundle mechanism, which validates requirements and extras unchanged.
The build interpreter is separate from the existing MAPLE CPU interpreter used for all transport checks.
An independent metadata audit confirms all other metadata fields and base requirements are unchanged; all eight
base requirements and the enabled `dask[array]` dependencies satisfy the actual CPU environment. The installed
MAPLE metadata remains stale: acceptance covers the immutable source and selected build metadata, rather than
certifying the editable installation as a deployable package.

The local [pinned configuration](../config/dense_cpu_20261019.toml) names that retained task-owned metadata artifact.
It is a configuration for this workspace; another checkout needs its own matching metadata and source store.
Fresh preparation still requires the pinned commits to be the current clean local HEADs and refuses any mismatch.
It never checks out or upgrades an upstream project. Existing explicit-lock verification is independent of current HEAD.

```bash
/home/okin/MAPLE/.venv/bin/python -I -B /home/okin/MASY/tools/masy_runtime.py \
    prepare --config /home/okin/MASY/config/dense_cpu_20261019.toml
/home/okin/MAPLE/.venv/bin/python -I -B /home/okin/MASY/tools/masy_runtime.py \
    verify --store /home/okin/MASY/runs/runtime-store \
    --lock ee190c813ae347a1f5c11e7ab5fad595dada898f3a13dddc775fca4488762a70
```

Preparation returned `runtime_locked`; a fresh post-test verification returned `runtime_verified`.
These commands qualify source delivery and do not run transport. Their `transport_ready=false` field reflects
that the runtime tool has no transport entry point; the separate Python API checks above establish the bounded
transport qualification. Tests use an explicit hashed pytest configuration with conftest discovery and plugin autoload disabled.
Pinned repository settings and relevant conftests are audited: no warning/error or strictness policy is omitted,
and the declared GPU marker is retained. Locked import paths replace live-tree path setup. The native autouse
source-location assertion is replaced by the equivalent explicit assertion against the locked package root.
Exact expected test names are collected and frozen before the rerun; both collection and executed call lists must match.
Every captured test/helper file (361 total) is verified against its exact committed Git blob. Before and after
transport, receipts verify immutable source objects, metadata, the native bundle guard and all imported package
origins. Test caches and outputs remain task-owned; no bytecode is written into source trees.

## Evidence and remaining work

Complete local evidence is under `agent_handoffs/tasks/masy08_storage_pair_20261019/`: baseline and manifests,
metadata build inputs/tool/artifact hashes, separate raw streams, commands, selected compiled/runtime identities,
PID/start/exit/reaping receipts, actual test results, independent review and finding dispositions. Source-only review
receipts identify every submitted file/content hash and allowed tools. Saved model payloads stay local.
MASY-07's preceding committed-origin qualification is retained under `tasks/masy07_committed_pair_20261018/`, lock
`b2cf01e16e80ca575a0f4351e94c170740fdfda2006965ef254f7bee511f2370`; MAPLE subsequently advanced to this task's merge.

This accepts neither GPU nor windowed wind/water execution, large-case performance, scientific calibration,
coupled saltation, combined persisted restart or a production collaborator CLI. MAPLE's merge provides shared-storage
libraries; dense wind remains dense. The owner reports three pre-existing native snapshot-restart failures and eight
stale unrelated assertions; their exact test identifiers are recorded and none overlap the selected tests or files.
MASY did not independently reproduce or repair them in this bounded task.
Combined restart and a documented collaborator workflow remain the next development milestone, with native restart
prerequisites resolved in their owning projects before acceptance. No upstream edits, environment installations,
commit or push are part of this qualification.
