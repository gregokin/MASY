# MAPLE-SYRUP (MASY)

MAPLE-SYRUP is designed to alternate actual MAPLE wind events and actual SYRUP water events over one evolving MAPLE sediment landscape. Wind physics stays in `~/MAPLE` and water physics stays in `~/SYRUP`. SYRUP's existing `maple_syrup` package name remains unchanged, and the combined wrapper uses `masy`. The wrapper contains no copied transport physics, no re-created sediment bed between events and no simultaneous wind/water transport.

**Current status (MASY-01):** the only executable piece is a read-only current-source compatibility preflight. No transport entry point, adapter, immutable runtime or alternating-event validation exists yet. MASY is not an installed package and has no `pyproject.toml`. The preflight never reports a pair as ready to run.

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
- The preflight reads live Git checkouts only. It cannot be pointed at a materialized bundle (no `.git`); verifying a locked runtime needs a separate future mode based on the upstream bundle/origin manifests.
- To pin a previously observed state, copy the report's `source_observation` block to a file and pass `--expect-lock FILE`.

Exit codes are `2` refused, `3` incompatible, `4` not_transport_ready (the best achievable result) and `1` internal error. There is no audit success exit.

Observed live results, in order:

1. On the initially clean pair (MAPLE `f1555e0`, SYRUP `9fabdca`) the run was `refused` at SYRUP's own resolver, which rejects the installed MAPLE metadata (`source_root=None`). Its API checks are reported next to that refusal, and MASY does not bypass or repair the guard.
2. An unrelated upstream MAPLE edit then appeared (`src/maple/core/parameters/flow_separation.py` dirty, HEAD unchanged), and the live audit correctly refused earlier with `dirty_scope`, before any upstream import. MASY did not adopt, revert or work around it, and did not fall back to the earlier clean observation.
3. That work was committed externally as MAPLE `5b1a052` (clean scope; SYRUP unchanged). The audit again refuses at the native resolver. SYRUP's own API list passes, but MASY's list reports a concrete Dense-path incompatibility: the current SYRUP `is_windowed_bed` imports `maple.core.types.voxel.WindowedVoxelColumnState` on every bed path, including Dense ones, and the observed MAPLE does not define it (finding 11, see [docs/preflight.md](docs/preflight.md#dense-path-type-dependency-finding-11)). This does not make Windowed storage required or supported.

A later run reports whatever the upstream state is at that moment. Which sources a later actual run should use (committed sources, or explicitly labelled development snapshots) is pending with the user; the preflight never infers adoption of a dirty tree or snapshot.

See [docs/preflight.md](docs/preflight.md) for stages, guards and report fields, and [docs/clock_contract.md](docs/clock_contract.md) for the exact clock and forcing representation.

## Tests

Tests use only stdlib `unittest`, with synthetic temporary Git repositories and fake upstream modules. They do not import real MAPLE/SYRUP code, construct beds or solvers, run transport or read datasets.

```
cd /home/okin/MASY && /home/okin/MAPLE/.venv/bin/python -B -m unittest discover -s tests -v
```

## Project rules

Read [AGENTS.md](AGENTS.md) and the [scientific brief](docs/scientific_brief.md) before agent work. Then read the [implementation phases](docs/implementation_plan.md) and the [observed upstream interfaces](docs/upstream_interface_audit.md). The current task and its status live in `agent_handoffs/`. No model, GPU job, package installation, Git initialization, commit or push is performed by the preflight.
