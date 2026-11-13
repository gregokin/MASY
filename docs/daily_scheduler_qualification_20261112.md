# Daily CSV scheduler and native water connection qualification

This record covers the new opt-in daily CSV inputs, nominal plan and inspector,
plus a fixed-geometry CPU connection to actual SYRUP Phase 1.5 water operations.
It does not qualify a complete wind–water–LEAF run. The illustrated full scenario
is readable now; wind and enabled transpiration/growth refuse native execution
before the first daily operation.

## Verified behavior

The weather header chooses minimal Hargreaves or explicit-PET columns and owns
the required precipitation threshold. The shipped threshold is 5 mm;
`0 < P <= threshold` requests reduced recharge, including equality. Scenario
headers own calendar/process/cadence controls; wind files can be reused, with
same-day row order preserved before rain. Input hashes, complete source wind
declarations and distinct occurrence identities are generated automatically.
The inspector prints bounded metadata without importing native models.

The water API delegates PET, recharge, downward movement above field capacity,
open-bottom drainage, closed retention, full storm/recession, daily evaporation
and water budgets to genuine native APIs. Only a minted native capacity failure
authorizes full original-rain fallback from the unchanged parent. Refusals and
interruptions preserve the last accepted boundary. No hidden substeps, dry reset,
physics copies or relaxed solver/conservation thresholds were added.

Each accepted nominal day contains exactly one native daily soil/E update.
Native storm/recession seconds are counted separately, so a nominal year can
have more than 365/366 days of physical elapsed time. Native V3 checkpoints
contain actual water state and daily cursor together; recovery selects the next
date and does not replay rain, evaporation or drainage. This is a water-only
boundary checkpoint, with exact source-byte identity.

## Evidence

The existing MAPLE CPU interpreter, Python 3.12.3 and NumPy 2.5.3 were used,
with `-I -B`, run-owned caches and no installation or GPU use. These MASY
physical checks used the NumPy backend; other backends require separate
acceptance.

| Check | Result |
|---|---|
| New daily suite without a native source pair | 78 discovered: 64 passed, 14 opt-in native cases skipped |
| Explicit real native CPU suite | 17 passed: 14 physical/failure cases and 3 preflight cases |
| Distinct new tests exercised across both runtimes | 78; the 3 preflight tests overlap |
| Original whole-suite baseline | 442 tests; 1 stale SHA failure, 32 missing-native errors, 26 skips |
| Final whole-suite regression comparison | 520 tests; same 33 failing test IDs as baseline, 40 skips; no new failures |

Physical checks use a synthetic native MAPLE 4 by 6 Dense case with two soil
layers. They exercise 5 mm equality, dry/leap-day continuation, gravitational
drainage, closed retention, full original-rain fallback and a direct full storm,
native Hargreaves from case latitude, disabled E, actual V3 recovery and exact
receipt comparison with uninterrupted continuation. All native lifetime water
audits remain within their original bounds. Bounded/incomplete storms, retained
pond CFL refusal, interruption, source drift, low memory, unrepresentable future
rain and receipt failure retain the accepted boundary. These are mechanics and
coupling checks, not field validation or scale benchmarks.

Codex inspected the actual source, native callers and diffs and ran the checks.
Claude implemented the initial input layer; independent read-only Claude
reviews used separate sessions and frozen source snapshots. The first review
found no blocking defect in its inspected code and identified several low
severity API edges. Interrupt handling, host-memory admission, native rain-plan
preflight, source drift handling and hand-built plan checks were corrected and
covered by regressions. Follow-up source review found no blocking defect in
the final delta; native execution separately verified the exception types and
controls that the reviewer could not check. The task archive records each finding and disposition,
exact submitted source hashes, prompts, raw separate process output and reaping.

## Exact sources

MASY baseline: `fa8f2667679b7fcac21f85065e65856807743da3`. The change is additive
to runtime code; old schemas, clock meanings, source guards, defaults and locks
remain byte-identical. Unrelated pre-existing dirty work was preserved.

| Qualified MASY module | SHA-256 |
|---|---|
| `daily_csv.py` | `7d4bc1a515ea35827280666f7adbba07be4f0d5bd432723b7ebe9f9900b247ec` |
| `daily_schedule.py` | `bbf26a16ae77c9ccbd813800104108d090016c8acc170efb24952319be3dfb75` |
| `daily_native.py` | `e010000545b67aa27077b7c8a2c1100eda0c61df965e7cf91ba3a560a29ea423` |
| `daily_cli.py` | `509b66fbd613c7f024a2a71e70e4f401e42b55a45f0f1186ddbd8d7879ae1f18` |

SYRUP uses the explicit published Phase 1.5 package
`82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107`
(114 package files; source commit `06952dbc2f984af8fb75dbf09a9c9f7fdbba23fc`,
documentation `a5b8b18358e0d014fe13c9f48d00eb582710655a`). The committed source
snapshot was hash-verified against its published manifest. MAPLE uses the actual
materialized historical `9f466d2cfcd52991b24496cf67f4562d6268afa9` bundle
`5d850789f9e82dc456227cb1eb4dfcc255b3ee4dfb0ed74883813c93e98d0ab3`, package
`fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e`.
Native source, module-origin, metadata, case and latitude guards were exercised.
These are explicit qualification sources; no current live HEAD was silently
rebound or historical runtime lock changed.

The complete local evidence is under
`agent_handoffs/tasks/daily_scheduler_csv_20261112/`: baseline and preservation
manifests, original Claude candidate, review snapshots/manifests, corrected
product manifest, daily/native check logs and whole-suite comparison. Failed
attempts are retained. No saved simulation payload was submitted to Claude.

## Remaining integration

Wind exchange with persistent layered water, the LEAF plant-only bridge and
combined physical restart remain open in this profile. LEAF must compute demand
and consume actual uptake/growth feedback while SYRUP alone withdraws water;
calling the older combined LEAF water driver would double count physical
updates. Output/checkpoint cadence flags are exposed requests, not automatic
output publication by this Python API. General runtime provisioning, active
storm recovery through MASY, CUDA/windowed/large-scale operation and scientific
field validation require their own acceptance. Native retained-pond
`router_step_rejected` remains an explicit supported refusal, with no hidden
daily subdivision.

Usage: [daily inputs and inspector](daily_scheduler.md),
[native water API and restart](daily_native_water.md).
