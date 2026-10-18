# MASY tests

Stdlib `unittest` only. The tests use synthetic temporary Git repositories, fake upstream modules and `/tmp` fixtures. They never import real MAPLE/SYRUP code, build beds or solvers, run transport, read datasets, install anything or touch `/home/okin/MASY/runs`.

```
cd /home/okin/MASY && /home/okin/MAPLE/.venv/bin/python -B -m unittest discover -s tests -v
```

| File | Covers |
|---|---|
| `test_config.py`, `test_clock.py` | preflight config schema; exact clock/forcing contract |
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

Independent native qualification is separate: current-source checkpoint suites passed 291 tests (8 skips), and genuine reference compact/windowed suites passed 404 (6 skips). Three further wet Dense save/load/continuation checks passed using only locked native imports and selected metadata. These tiny CPU fixtures do not qualify combined restart, alternating transport, windowed coupling, GPU or performance. See [verification](../agent_handoffs/verification.md).
