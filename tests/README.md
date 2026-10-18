# Tests

MASY-01 has stdlib `unittest` tests for the read-only preflight only: config schema, source observation and Git output bounds, import origins and bounded hashing, native resolver/API classification, clock/forcing records, report publication and the launcher's input/output separation. They use synthetic temporary Git repositories and fake upstream modules. They do not import real MAPLE/SYRUP code, construct beds or solvers, run transport or read datasets.

The API classification tests in `test_contracts.py` cover:

- **Generic kinds.** Class, dataclass and callable requirements, including re-exports and aliases.
- **Optional Windowed binding (MASY-02B).**
  - A literally absent binding passes.
  - Present non-class bindings fail.
  - Module import failures fail.
  - A module `__getattr__` cannot manufacture or mask the binding.
  - Invalid `optional_class` parameters or fields are refused.
- **Checkpoint ledger.** The required `CompactSedimentLedgerState` fails when it is missing or not a class, and the finding names both checkpoint callers.
- **Dense requirement.** The required `VoxelColumnState` still fails when missing.
- **Current-pair-shaped audit.** Resolver refusal, the separate compact-ledger incompatibility, the optional transaction as `info`, and a dynamic SYRUP requirement count.

One explicitly labelled historical test keeps the SYRUP `9fabdca` unconditional-Windowed contract as a custom requirement. It is not the current table.

Codex ran the full suite on the MASY-02B candidate: 170 tests passed (exit 0). These are synthetic stdlib tests, not physical, transport or GPU tests.

```
/home/okin/MAPLE/.venv/bin/python -B -m unittest discover -s tests -v
```

Still planned, none implemented: same-bed identity and class budgets; nonzero wind→water→wind exchange; conservative airborne/water-mobile boundary disposition; actual upstream single-process controls; wet/wind/boundary restart; CPU/GPU and windowed qualification; resource/transfer measurements.
