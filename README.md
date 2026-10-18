# MAPLE-SYRUP (MASY)

MAPLE-SYRUP is designed to alternate actual MAPLE wind events and actual SYRUP water events over one evolving MAPLE sediment landscape. Wind physics stays in `~/MAPLE` and water physics stays in `~/SYRUP`. SYRUP's existing `maple_syrup` package name remains unchanged, and the combined wrapper uses `masy`. The wrapper contains no copied transport physics, no re-created sediment bed between events and no simultaneous wind/water transport.

**Current status:** there are two executable pieces.

- The MASY-01 read-only current-source compatibility preflight.
- The MASY-03 locked CPU runtime `prepare`/`verify` ([docs/runtime.md](docs/runtime.md)). It seals the committed MAPLE/SYRUP/MASY sources and checks native interfaces. CPU source and interface delivery is independently qualified for MAPLE `5b1a052e` with local SYRUP `b4ee14e`.

No transport entry point, adapter or alternating-event validation exists. Neither tool reports a pair as ready to run, and `transport_ready` is always `false`.

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

A later run reports whatever the upstream state is at that moment. Under [AGENTS.md](AGENTS.md), a new run selects the current local committed HEAD of each configured upstream, subject to compatibility and immutable-runtime qualification. A development snapshot requires an explicit selection, with its dirty deltas recorded. No coupled simulation has been launched; the preflight never infers adoption of a dirty tree or snapshot.

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

Codex verified committed-source preparation, repeated fresh verification, verification with both configured upstream source roots absent, and retention of an older lock after the wrapper changed. Three native wet Dense checkpoint/continuation checks also passed on the locked imports. The owning-SYRUP checkpoint fix is committed on local main as `b4ee14e`; it has not been pushed to the SYRUP remote. An unpatched SYRUP still refuses qualification. See [verification](agent_handoffs/verification.md).

## Tests

Tests use only stdlib `unittest`, with synthetic temporary Git repositories and fake upstream modules. They do not import real MAPLE/SYRUP code, construct beds or solvers, run transport or read datasets.

```
cd /home/okin/MASY && /home/okin/MAPLE/.venv/bin/python -B -m unittest discover -s tests -v
```

All 245 synthetic stdlib tests pass, including the 70 runtime regressions in `tests/test_runtime.py`. Native checkpoint tests are separate from this suite; see [tests/README.md](tests/README.md) and the verification receipt above.

## Project rules

Read [AGENTS.md](AGENTS.md) and the [scientific brief](docs/scientific_brief.md) before agent work. Then read the [implementation phases](docs/implementation_plan.md) and the [observed upstream interfaces](docs/upstream_interface_audit.md). The current task and its status live in `agent_handoffs/`. No model, GPU job, package installation, Git initialization, commit or push is performed by the preflight or the runtime tool. MASY now has a Git baseline, but the preflight's own MASY observation still uses file manifests and an explicit `--expect-lock`, not MASY's Git state.
