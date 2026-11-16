# Dense CUDA integration — accepted

Actual MAPLE wind, native SYRUP rain/recession/daily water and LEAF demand/feedback/weekly growth now share one resident Dense CUDA landscape. The current code sources are in ACCEPTANCE.json; source/runtime qualification is separate from physical evidence.

The final sealed committed-source8day run passes on physicalGPU0, with2wind events, full rain/recession,8daily E/T updates and1weekly growth update. Final fields match the CPU control under owning tolerances.16native CUDA controls cover positive5mm equality recharge, wet-surface wind, capacity return, bounded real recession, native feedback, source/device/memory/stale/replay and publication-purity checks.

Known limits: small Dense baseline-direct profile; no Windowed/large-domain speed claim, full CUDA scenario CLI or combined persisted restart. Wind assumes dry soil and preserves actual water. Near-saturated closed-bottom full recession exceeded its external300sCPUguard; that attempt is incomplete, with bounded continuation/wind exclusion qualified. No physics/tolerances were altered to complete it.

Claude source-only implementation/review receipts, all findings/dispositions, independent verification and preserved failed attempts are in this task archive. Saved simulation arrays stay local.
