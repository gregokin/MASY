# MAPLE-SYRUP (MASY)

MAPLE-SYRUP is designed to alternate actual MAPLE wind events and actual SYRUP water events over one evolving MAPLE sediment landscape. Wind physics stays in `~/MAPLE` and water physics stays in `~/SYRUP`. SYRUP's existing `maple_syrup` package name remains unchanged, and the combined wrapper uses `masy`. The wrapper contains no copied transport physics, no re-created sediment bed between events and no simultaneous wind/water transport.

**Current status:** the restricted Dense CPU Python integration passes on committed MAPLE `39bcf79` + SYRUP
`d05d8c6`, using MASY runtime `22aa6ab`. Claude accepted the source-only qualification review, followed by Codex
verification of all execution and evidence conditions.
Actual nonempty wind → rain/recession → nonempty wind → second rain/recession passes on one evolving bed,
including a second wind at 169 seconds, final settlement, class budgets, routing and staged-failure isolation.
See [the exact committed pair, lock and checks](docs/source_pair_20261019.md).

Later published MASY additions provide [bounded Dense CPU production-feature
coupling](docs/dense_production_wind.md), including reduced wind, topographic
separation, creep and avalanching on labelled synthetic interface fixtures.
Production calibration remains pending. The [GPU runtime](docs/gpu_runtime.md)
provides separate immutable source profiles and an identity-only probe;
numerical GPU transport remains unqualified. These later sources require their
own new runtime identities and never change the reference lock above.

**Coherent CPU components (October 19).** The coherent CPU publication candidate is the immutable source-manifest
prototype `77f9c077…`, built over live published MASY `72a7d1d`, MAPLE `89a2775` and SYRUP `d05d8c6`. It uses one
shared CPU engine whose ordinary default results stay byte-identical to the original runner. Accepted components on
that engine are [manual versioned vegetation updates with a native wake
response](docs/vegetation_information_plan.md#current-accepted-cpu-component-status-october-19) and an opt-in
[accepted-boundary store prototype](docs/dense_boundary_store.md) bound to immutable source manifests. Neither
component is tied to a committed runtime lock yet. The earlier store refusal at exact environment identity was
measured as exactly two missing compiled child extensions (`readline`, `resource`); after a test-fixture-only
correction, the fresh targeted store, routing and schema checks and two native controls passed on this candidate
(see [Tests](#tests)). Publication and a fresh committed CPU lock are still pending. No production calibration, GPU,
windowed or full-scale support is claimed. The project goal of full alternating operation remains active.

The [schedule and bed adapter](docs/dense_integration.md), [Python event runner](docs/dense_runner.md), and
[native finalization adapter](docs/wind_finalization.md) reuse the actual upstream physics. MAPLE's clock-origin
extension is committed and included in this merge. Shared-storage libraries are present; windowed wind execution
remains unqualified. GPU events ([private device runner](docs/dense_device_runner.md)), large cases, general
combined restart and a collaborator transport CLI require later work.
The [runtime tools](docs/runtime.md) still report `transport_ready=false` because they have no transport entry point;
source delivery and the separately tested Python API have distinct acceptance evidence.

The [shared vegetation and soil-water plan](docs/vegetation_information_plan.md) defines versioned updates between
events, read by both models. Manual CPU updates of the five native fields now exist. Joint soil-water exchange is
still refused until it is scientifically specified. Growth, mortality and uptake laws remain separate scientific
decisions.

MASY now has a static, declarative [pyproject.toml](pyproject.toml) (`masy` `0.0.3.dev0`, equal to `masy.__version__`). It is never installed or built. The runtime uses it only to generate a labelled, run-owned metadata overlay that is not an installation.

*Historical (MASY-01/02B): MASY then had no `pyproject.toml` and only the preflight existed.*

## Preflight

Use MAPLE's existing CPU interpreter. Nothing needs to be installed, and no new environment is created.

```
/home/okin/MAPLE/.venv/bin/python -B -I /home/okin/MASY/tools/masy_preflight.py \
    --config /home/okin/MASY/examples/preflight.example.toml \
    --run-dir /tmp/masy-preflight-$(date +%Y%m%dT%H%M%S)
```

- `-B -I` are mandatory for an audit, and the launcher refuses without them. `--help` is informational, exits 0 and creates nothing.
- `--run-dir` must not exist yet, and its parent must exist (the launcher never creates parents; a fresh MASY tree has no `runs/`, so either use `/tmp` or run `mkdir -p /home/okin/MASY/runs` first). The launcher creates it exclusively, and it receives `report.json`, `report.json.sha256` and the run-owned caches.
- `--run-dir` may not equal, contain or lie inside a configured source root or MASY's `src/`/`tools/` (symlinked aliases included). That is refused before anything is created.
- The preflight reads live Git checkouts only. Use the separate runtime tool below to verify a materialized bundle and explicit lock.
- To pin a previously observed state, copy the report's `source_observation` block to a file and pass `--expect-lock FILE`.

Exit codes are `2` refused, `3` incompatible, `4` not_transport_ready (the best achievable result) and `1` internal error. There is no audit success exit.

Observed live results, in order:

1. On the initially clean pair (MAPLE `f1555e0`, SYRUP `9fabdca`) the run was `refused` at SYRUP's own resolver, which rejects the installed MAPLE metadata (`source_root=None`). Its API checks are reported next to that refusal, and MASY does not bypass or repair the guard.
2. An unrelated upstream MAPLE edit then appeared (`src/maple/core/parameters/flow_separation.py` dirty, HEAD unchanged), and the live audit correctly refused earlier with `dirty_scope`, before any upstream import. MASY did not adopt, revert or work around it, and did not fall back to the earlier clean observation.
3. *Historical (SYRUP `9fabdca`).* That work was committed externally as MAPLE `5b1a052` (clean scope; SYRUP unchanged at `9fabdca`). The audit again refused at the native resolver. SYRUP's own API list passed, but MASY's list reported a concrete Dense-path incompatibility: that SYRUP's `is_windowed_bed` imported `maple.core.types.voxel.WindowedVoxelColumnState` on every bed path, including Dense ones, and the observed MAPLE does not define it (finding 11, see [docs/preflight.md](docs/preflight.md#dense-path-type-dependency-finding-11-historical)). This did not make Windowed storage required or supported.
4. *Historical committed pair (MASY-02B).* MAPLE `5b1a052` with SYRUP `2c60c73` (committed and pushed after `1f70a87`). SYRUP's classifier now treats Windowed as an optional binding: absence permits Dense, while a present non-class binding is refused. MASY's list mirrors that rule, so missing Windowed is no longer an incompatibility. MASY's list now also requires `maple.core.types.sediment_ledger.CompactSedimentLedgerState`. Both SYRUP checkpoint registries import it unconditionally, and current MAPLE lacks it ([checkpoint ledger dependency](docs/preflight.md#checkpoint-ledger-dependency-masy-02b)). Codex ran this MASY candidate's preflight against the actual clean current committed sources. The result was `refused` (exit 2). The MAPLE `5b1a052e` and SYRUP `2c60c73` observations were unchanged, and there were no wrong module origins. SYRUP's own API list passed (36 items in that receipt). The optional Windowed requirement passed, with the class genuinely absent. The only failed MASY item was `CompactSedimentLedgerState` (attribute missing, both checkpoint callers named). The optional transaction module stayed `info`. Separately, the native resolver still refused (`source_root=None`), and `transport_ready` was `false`. The final adopted-root preflight confirmed the same result; receipts are in [agent_handoffs/verification.md](agent_handoffs/verification.md). Nothing here accepts coupled operation, restart, a backend or performance.

A later run reports whatever the upstream state is at that moment. Under [AGENTS.md](AGENTS.md), a new run selects the current local committed HEAD of each configured upstream, subject to compatibility and immutable-runtime qualification. A development snapshot requires an explicit selection, with its dirty deltas recorded. The preflight never infers adoption of a dirty tree or snapshot. MASY-04's bounded native checks used the explicit prior frozen pair while MAPLE main was being developed independently.

See [docs/preflight.md](docs/preflight.md) for stages, guards and report fields, and [docs/clock_contract.md](docs/clock_contract.md) for the exact clock and forcing representation.

The MASY requirement table has changed since the receipt above. For the owning-SYRUP checkpoint candidate it now requires `SedimentLedgerState` and SYRUP's shared `checkpoint._native_ledger_types` helper, and treats `CompactSedimentLedgerState` as an optional class. An unpatched SYRUP `2c60c73` therefore still fails, now on the missing helper. That observation above is historical and has not been re-run with the new table.

## Locked CPU runtime (MASY-03)

```
mkdir -p /home/okin/MASY/runs   # once: the store's parent must exist
/home/okin/MAPLE/.venv/bin/python -I -B /home/okin/MASY/tools/masy_runtime.py \
    prepare --config /home/okin/MASY/examples/runtime.example.toml
/home/okin/MAPLE/.venv/bin/python -I -B /home/okin/MASY/tools/masy_runtime.py \
    verify --store /home/okin/MASY/runs/runtime-store --lock <lock_id>
```

- `prepare` selects the current local committed HEADs: no development snapshot and no uncommitted runtime code, the wrapper included. It seals them into a content-addressed store and freezes MAPLE with SYRUP's own `dependency_bundle`. It probes the native resolver, API and both checkpoint registries in a fresh `-I -B` CPU child, then publishes a lock.
- `verify` re-checks one explicit old lock without reading current upstream repositories.
- Nothing is installed, built, downloaded or garbage-collected. Old locks and objects are retained.
- Exit `0` means runtime identity only, never transport readiness.

Layout, provenance, refusal evidence and limitations are in [docs/runtime.md](docs/runtime.md).

Codex verified committed-source preparation, repeated fresh verification, verification with both configured upstream source roots absent, and retention of an older lock after the wrapper changed. Three native wet Dense checkpoint/continuation checks also passed on the locked imports. The owning-SYRUP checkpoint fix `b4ee14e` is committed and pushed; GitHub's matching head was verified on October 18. An unpatched SYRUP still refuses qualification. See [verification](agent_handoffs/verification.md).

## Tests

The default suite uses stdlib `unittest`, with synthetic temporary Git repositories, fake upstream modules and data-only schedule checks. It does not import real MAPLE/SYRUP code or run transport. Native integration checks live separately under `tests/native/` and require locked upstream imports.

```
cd /home/okin/MASY && /home/okin/MAPLE/.venv/bin/python -B -m unittest discover -s tests -v
```

On the task-80 source prototype, 12 completed groups passed 543 tests plus 150 subtests, with byte-exact whole default and synthetic production-feature parity. Its store component failed, and the task-82 narrow parent/child import-symmetry attempt also refused at exact environment identity before its kill point; both failed attempts are preserved. Task 86 measured the cause as exactly the missing compiled `readline`/`resource` child extensions, with every other environment operand exact. Task 87 appended those two symbols symmetrically to the test `cpu_declarations` tuple only (no guard, physics, tolerance or runtime change; native runtime `4139f62a…` unchanged). Its fresh targeted run then passed 17 methods plus 50 subtests (store 2, routing 5+6, schema 10+44) and two native controls, with no failures or skips. Publication and a fresh committed lock remain pending. At MASY `bf46354`, 307 stdlib tests and 46 separate native configuration checks pass with no skips. The MASY-09 reference pair passed 153 tests plus 61 subtests and 96 separate component checks. MASY-06’s 264-test result and MASY-08’s 113/15/22/3 native results are historical. GPU source-profile provenance tests also inspect local immutable historical source/build receipts and the retained CPU lock; they launch no accelerator or model. Fresh-checkout test-fixture packaging remains collaborator-workflow work. See [tests/README.md](tests/README.md) and [verification](agent_handoffs/verification.md) for exact sources, receipts and limits.

## Project rules

Read [AGENTS.md](AGENTS.md) and the [scientific brief](docs/scientific_brief.md) before agent work. Then read the [implementation phases](docs/implementation_plan.md) and the [observed upstream interfaces](docs/upstream_interface_audit.md). The current task and its status live in `agent_handoffs/`. No model, GPU job, package installation, Git initialization, commit or push is performed by the preflight or the runtime tool. MASY now has a Git baseline, but the preflight's own MASY observation still uses file manifests and an explicit `--expect-lock`, not MASY's Git state.
