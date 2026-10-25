# Bounded six-class windowed CPU wind record — October 25, 2026

A labelled synthetic 3 × 3 landscape completed a two-block wind record using MAPLE’s native vertical-windowed bed. Three of six sediment voxels per column were resident; the bed had six sediment classes. The native dense oracle ran independently from the same initial physical state. The full-record test passed its original physical-state, class-budget, clock, final-settlement and storage-version assertions.

This is wind-only qualification on this exact immutable source selection:

| Component | Git revision |
| --- | --- |
| MASY qualification runtime | `edfa8dc0ce82c5fbf81be773c36a7723cadfe943` |
| MAPLE | `b4fdada0ad0af4d4f903fadffafe63372b9dbeeb` |
| SYRUP runtime dependency | `5e6a589f8b4c6b37422e7f0d728115582450eef0` |
| CPU runtime lock | `448552a6ec7a15caa023a707a0468148e346720c65166cf1269defa148d91574` |

SYRUP did not run a water event in this test. This source selection is separate from the [bounded Dense GPU alternating-event qualification](dense_gpu_qualification_20261025.md); neither result qualifies the other selection or an arbitrary later checkout.

## Physical and execution checks

The record starts at 7.25 seconds and runs two two-second wind blocks, then deliberately invokes native wind-event finalization once. The oracle and windowed run both execute native reduced wind transport. Bed commits, four accounting steps, retained initial airborne material, final deposition, genuinely empty terminal airborne state, exact clock/cursor checks, conservative class budgets, immutable old deep-store versions and the single-use record contract all pass. Sediment-transfer checks and existing native numerical bounds are unchanged.

This K6 fixture uses attached flow; separation, vegetation wake and dormancy are explicitly disabled, and creep and topographic wind are not enabled in this full-record case. It is not the separately developed windowed production-feature fixture. Synthetic calibration remains labelled; production calibration remains pending.

The selected test was `tests/windowed_wind/test_w8_multiclass_windowed_record.py::test_k6_record_explicit_8GiB_matches_native_dense`. Only this affected case ran: one test passed in 13.87 seconds, with setup, call and teardown all accepted. The original bounded controller and worker exited successfully, were reaped and observed gone, and the immutable runtime verified again afterward. Previously passed metadata and private-storage checks were retained through explicit committed-source continuity rather than replayed.

## Memory allowance and limits

This is the separately named **8 GiB address-space qualification profile**, with at least 10 GiB fresh host headroom before launch. The worker retains the original 120-second CPU, 180-second wall, 32 MiB per-file, 4 MiB output, 1,024-descriptor and 64 MiB bed-store limits. Native component, metadata, current/prior-state, work and finalizer admission checks remain active. No core physics or comparison tolerance changed.

The original 4 GiB profile remains unchanged and its full-record attempt remains failed before the full record ran. The 8 GiB allowance is an engineering ceiling; this pass is not a measurement of 8 GiB use, a peak-memory estimator, a performance benchmark, a general capacity bound or evidence of efficient scaling.

The local independent acceptance receipt is `agent_handoffs/tasks/masy802_K6_explicit_8GiB_committed_CPU_qualification_20261025/Root_data/ROOT_affected_CPU_acceptance_01.json`, SHA-256 `3f8fd9669fa83314f70bcbfdcd87f02fe7d5873e837923f6ffc37e097def8e82`. Detailed simulation evidence stays local.

Windowed wind–water coupling, complete windowed production features, windowed GPU execution, persisted combined restart and larger-landscape memory/performance qualification remain open. This result adds no collaborator transport CLI or automatic model launch.
