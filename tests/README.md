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
| `test_runtime.py` | **not written yet**: runtime prepare/verify synthetic tests (planned next stage) |

`test_contracts.py` changes in MASY-03 (table contracts only):

- the checkpoint-ledger rows now expect a required `SedimentLedgerState`, an `optional_class` `CompactSedimentLedgerState` and a required `checkpoint._native_ledger_types` callable;
- the old-pair test now fails on the missing helper;
- added: literal compact absence passes, `__getattr__` cannot manufacture the compact binding, a missing dense ledger fails, a missing or class/None helper fails, and a candidate-shaped surface passes.

The generic class/optional-class tests and the Dense/helper requirements are unchanged.

Historical count: 170 tests passed in Codex's MASY-02B receipt. These edits remove one test and add six, so the expected count is 175 before `test_runtime.py` exists. That count has not been run.
