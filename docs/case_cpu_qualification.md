# Case-driven CPU workflow qualification — October 26, 2026

The real `tools/masy_case.py run` entry completed a bounded, labelled synthetic
MAPLE case through wind → rain plus physical recession → wind, using the same
evolving Native bed. Root independently accepted coldCLI04 (acceptance digest below). This is a
CPU case-workflow qualification on the exact executable sources below.

For a user case, start with the [practical guide](user-guide/case-cpu-workflow.md),
[MAPLE case adoption](user-guide/maple-case-adoption.md),
[explicit water inputs](user-guide/case-water-inputs.md) and
[immutable runtime](runtime.md). Supply the real DEM and authored scientific
inputs; no real user terrain or default calibration is provided by this test.

## Executed source and identity

| Subject | Executed identity |
| --- | --- |
| MASY | `84fba92d5623d7762aaa7d78e99f8761880e6717` |
| MAPLE | `6b2233cdac49096593a7d405c0151251a9207cce` |
| SYRUP | `6c8dc9137ba957def43241400a69089bf29b097d` |
| Immutable runtime lock | `f8d1bd0f69df1c2a4f7fd9191675eeaffa3bdea8193ec00833c77dc60e50139f` |
| Independent coldCLI04 acceptance | `72440e0a2933de62423453199cfe1388aaa69d9deef3fb4cbeed6e6ccea5511e` |

These are feature-source identities, not claims that MAPLE or SYRUP main has
advanced. Other machines must select and verify their actual reviewed source
checkouts. Later documentation-only publication changes
the MASY HEAD; it does not rename the executed `84fba92` runtime or create a new
run result. Executable-source continuity and any fresh lock remain explicit.

## What passed

The fixture imported a 9 × 8 float64 NPY DEM (72 cells), with 2 sediment classes and
20 voxels per column. Its declared first wind event contained 2 Native forcing
blocks and one event-end finalizer. A 60-second rain forcing was followed by
physical recession and genuine water completion. The final wind event contained
1 block and its one finalizer. Three event boundaries were accepted; actual
Native elapsed time ended at 177.75 seconds. The Native worker took
10.079492 seconds of measured elapsed runtime; this is one observation, not a
performance or fit guarantee for another case.

Checks accepted positive wind and water transport, Native per-class sediment
and water budgets, genuine final settlement with empty airborne inventory,
all five prescribed vegetation fields, unchanged original case/water inputs,
and final Source/current-binary identity. The final analysis archive was read
back against the owning final-state projection: all 36 NPY members matched
exactly (16 arrays and 20 scalars). Six required HOST role joins and complete
process cleanup were independently accepted. Full payloads and raw logs remain
local; no simulation arrays are reproduced here.

The existing 15 schedule cases, 2 failure/workcap checks and 19 Native input checks
remain earlier component evidence, without relabelling them as fresh runs on
this pair. The 3 new code-owner metadata checks passed separately. A separate direct
Native positive oracle on the same exact new pair also passed: one case and
three pytest phases in 12.589302 seconds, with exact direct Native endpoints.
Root reported its independent oracle receipt digest as
`1a85d43436ceb422b4da3e48b490038d0ff3a2312e2d0681d2058d492a69d486`.
An earlier attempt passed its test assertions but failed the outer evidence-path
join and was retained as failed. The accepted repeat changed only that outer
caller evidence path and used fresh run directories. This oracle acceptance is
distinct from the complete cold CLI acceptance above.

## Boundaries of this result

The workflow retains the actual MAPLE bed, layers, availability and water;
SYRUP runs conservative water events. It uses the explicit bounded CPU process
policy, caller-selected singleton CPU affinity, immutable source/runtime and
five-owner machine code contract in the guide. Original Native numerical
bounds, forcing, completion and engineering caps were retained. Passing this
fixture does not establish resource fit at other dimensions or certify a new
case's masks, hydraulic boundary or coefficients. Unsupported inputs and
unknown code identities continue to refuse.

`analysis_landscape.npz` remains a diagnostic analysis artifact:
`is_diagnostic=True`, `validated=False`, `restartable=False`. It omits the
ledger, committed/reconciliation state and process-private continuation needed
for restart. Exact array readback does not change those flags.

This record does not qualify CPU–GPU equivalence, full windowed coupling,
large cases, production calibration, arbitrary imported formats, combined
restart or automatic future-source compatibility. Those retain their separate
qualification requirements.
