# Tests

MASY-01 has stdlib `unittest` tests for the read-only preflight only: config schema, source observation and Git output bounds, import origins and bounded hashing, native resolver/API classification (including the Dense-path type requirement), clock/forcing records, report publication and the launcher's input/output separation. They use synthetic temporary Git repositories and fake upstream modules. They do not import real MAPLE/SYRUP code, construct beds or solvers, run transport or read datasets.

```
/home/okin/MAPLE/.venv/bin/python -B -m unittest discover -s tests -v
```

Still planned, none implemented: same-bed identity and class budgets; nonzero wind→water→wind exchange; conservative airborne/water-mobile boundary disposition; actual upstream single-process controls; wet/wind/boundary restart; CPU/GPU and windowed qualification; resource/transfer measurements.
