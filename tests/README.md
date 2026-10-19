# MASY tests

The default stdlib suite uses synthetic temporary Git repositories, fake upstream modules, `/tmp` fixtures and
data-only schedules. It never imports real MAPLE/SYRUP code, runs transport, reads datasets or installs anything.
The separately invoked checks under `tests/native/` construct small native beds and execute actual CPU transport.

```
cd /home/okin/MASY && /home/okin/MAPLE/.venv/bin/python -B -m unittest discover -s tests -v
```

| File | Covers |
|---|---|
| `test_config.py`, `test_clock.py` | preflight config schema; exact clock/forcing contract |
| `test_dense_schedule.py` | relative sequential forcing, exact records/hashes, calendar and malformed-input refusal |
| `test_sourceobs.py` | committed-scope source observation and lock guards |
| `test_importcheck.py` | module origins and environment record |
| `test_contracts.py` | native audit classification and the `MASY_REQUIRED_API` table against synthetic modules |
| `test_report.py`, `test_launch.py` | report publication; preflight launcher |
| `test_runtime.py` | 70 runtime regressions: source capture, immutable objects, metadata, locks, input/output separation, resource bounds, process cleanup and bootstrap guards |

`test_contracts.py` changes in MASY-03 (table contracts only):

- the checkpoint-ledger rows now expect a required `SedimentLedgerState`, an `optional_class` `CompactSedimentLedgerState` and a required `checkpoint._native_ledger_types` callable;
- the old-pair test now fails on the missing helper;
- added: literal compact absence passes, `__getattr__` cannot manufacture the compact binding, a missing dense ledger fails, a missing or class/None helper fails, and a candidate-shaped surface passes.

The generic class/optional-class tests and the Dense/helper requirements are unchanged.

Codex ran 245 tests with no failures, errors or skips: the updated 175-test preflight suite plus 70 runtime tests. Source files stayed unchanged, and all loaded MASY modules came from the candidate. The first expanded run exposed a missing-version TypeError; Claude corrected the type guard and the complete suite then passed. Historical MASY-02B had 170 tests.

MASY-04's complete stdlib suite passes **263 tests**, including 18 new schedule tests, with no failures, errors or
skips. The first full candidate capture omitted an existing example config; Codex restored both unchanged committed
example templates to the test snapshot and the full rerun passed. No product change was needed for that capture error.

Independent native qualification is separate: current-source checkpoint suites passed 291 tests (8 skips), and genuine reference compact/windowed suites passed 404 (6 skips). Three further wet Dense save/load/continuation checks passed using only locked native imports and selected metadata. These tiny CPU fixtures do not qualify combined restart, alternating transport, windowed coupling, GPU or performance. See [verification](../agent_handoffs/verification.md).

MASY-04 adds explicit native checks, which are excluded from ordinary stdlib discovery:

| File | Verified scope |
|---|---|
| `native/check_dense_bed.py` | 15 tests and 38 subtests: genuine bed construction, complete shared-state mappings, partial availability, bounded staging, stage-owned wind entry and four independently reproduced regressions |
| `native/check_dense_runner.py` | 18 tests and 21 subtests: nonzero wind/rain/recession/wind versus direct native control, persistent fields and clocks, routing re-entry, piecewise rainfall, native incomplete outcomes, staging failures, static-input and duration guards |
| `native/fixture_dense_alternation.py` | Authored six-class 6×1-cell fixture using actual native constructors; no external terrain or forcing payloads |

Run these only after a caller establishes the explicit immutable native pair, MASY source snapshot, matching
distribution metadata and module origins. An editable environment alone is insufficient; setting `PYTHONPATH` does
not override `-I`. Codex's bounded verification workers record exact source/candidate hashes, isolated import paths,
compiled environment identities, CPU limits, output paths and child exit/reaping. Payloads stay local. The complete
receipts and independent review dispositions are linked from [verification](../agent_handoffs/verification.md).

This qualifies only the [restricted Dense CPU runner](../docs/dense_runner.md). Combined restart, coupled saltation,
windowed/GPU operation and large-case performance require separate work. The origin extension is now committed upstream.

MASY-06 adds native finalizer integration: 92 native finalizer checks (including explicit absolute origin, unchanged
placement, tiny inventory and repeat idempotency), 20 runner checks plus 23 subtests, 15 bed checks plus 38 subtests,
and 264 stdlib tests pass on proportional sealed snapshots. The genuine nonempty four-event sequence matches direct
native execution. Finalizer failure/mutation retains the accepted water boundary and blocks the next process.
See [verification](../agent_handoffs/verification.md) for exact development IDs, source guards and lifecycle.

Historical (MASY-08): this task requalified the merged committed MAPLE `39bcf79` + SYRUP `b4ee14e` pair with unchanged MASY `22aa6ab` runtime.
Passed: 113 native settlement/origin/unsupported-storage checks; 15 bed +38 subtests; 22 runner +23 subtests;
three existing genuine wet SYRUP checkpoint continuations. No skips. The runner count includes 20 committed tests
and two task-local extensions adding exact clock/routing/water-budget and staged-content/recovery assertions.
All 361 captured native test/helper files are verified against exact committed Git blobs. Independent review status,
source/metadata preparation and limits are in [committed-pair qualification](../docs/committed_pair_qualification.md).
Historical (MASY-08): no runtime code or committed product tests changed; the stdlib suite was not rerun for that documentation task.

Current published wrapper milestone `bf46354` passed 307 stdlib checks and 46 native configuration checks, with no skips. MASY-09’s reference pair passed 153 tests plus 61 subtests and 96 separate component checks. That reference retains its original locked runtime; later wrapper commits have separate identities. GPU source-profile tests use bounded immutable Git fixtures. Two provenance/retention tests also inspect the existing local historical source object, task08 source-built metadata receipt and retained CPU lock. These source-only checks run no accelerator or model. Packaging fixtures for a fresh checkout remains collaborator-workflow work.
