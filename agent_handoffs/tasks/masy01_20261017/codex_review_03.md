# Codex review of correction_02 — two narrow corrections required

Exact reviewed bytes: correction_02_manifest.json and correction_02_snapshot/. Claude correction_02 exited 0/reaped. Codex verification_03: 149 tests and 106 subtests pass; syntax and isolated CLI help pass. Finding 11 now detects the actual required Dense-path type import, with precise required_by evidence and optional transaction absence kept informational. Findings 1–10 retain their original verified fixes; the residual conversion-method gap below extends finding 4.

## Finding 12 — Medium, OPEN: conversion methods violate the native clock domain

Independent scalar-only reproduction: working_reproductions_05.json (also earlier clock-only working_reproductions_04.json). With ClockLedger(100.0), global_elapsed(-1.0) returns -101 and native_from_global(-101) returns -1.0. global_elapsed(-0.0) returns -100 and the inverse returns +0.0, losing the sign despite the stated bit-exact accepted-domain policy. Constructors, append and forcing/config already reject these native values; the public conversion methods still do not.

Required correction: use the existing _native_clock validation for the native input of global_elapsed and the native result of native_from_global, preserving exactly representable nonnegative native values. Do not reject all negative GLOBAL offsets: with epoch 100, offset -75 correctly represents earlier native time 25 (e.g. an earlier commit anchor). Reject a result below zero. Add focused tests for negative native input, negative zero native input, negative native inverse, valid earlier nonnegative native roundtrip, and positive-zero bits with a nonzero epoch. Leave the generic lossless encode_time/decode_time codec unchanged; invalid-value diagnostics may use it. No normalization, rounded timing or solver changes.

## Finding 13 — Medium, OPEN: new class check imposes unsupported implementation names

contracts.py newly requires obj.__module__ and obj.__qualname__ to exactly match the public module/attribute requirement. That rejects genuine classes re-exported or aliased by the verified public module. Independent scalar synthetic module repro in working_reproductions_05.json binds PublicColumn to an actual class defined under native_implementation.InternalColumn; it fails only because of the naming rule.

Required correction: for kind="class", validate that the exported public binding is a real class (isinstance(obj, type)). Do not impose private definition-name strings. Existing module-origin/source-root validation remains mandatory and unchanged. None, function, instance and missing attributes must fail. Replace tests asserting renamed/re-exported genuine classes fail with meaningful passing aliases/re-exports; retain nonclass failures and origin/shadow-module checks. Update docstrings/docs that state the unsupported exact-definition-name rule. This is the same canonical public binding issue already resolved in the accepted owning-SYRUP DESIGN proposal; do not copy its classifier into runtime.

## Scope and acceptance

Only these targeted changes in implementation/src/masy/{clock,contracts}.py, focused tests, matching documentation and implementation report. Preserve actual missing WindowedVoxelColumnState detection, required_by evidence, native guards, output/source overlap protections, bounded I/O and current-source policies. No upstream edits, physical runs, environment/packaging, immutable-run builder or source adoption. Current observed clean pair remains MAPLE 5b1a052e1b9a9edfed0760152aac2a743813e79a / SYRUP 9fabdca60bd9cd7c5b358d5cbf620dbd3209d93a, unqualified for transport. Codex runs independent checks; Claude states tests NOT RUN.
