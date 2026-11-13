# Daily CSV native wind component qualification

This milestone qualifies execution of one parsed daily CSV wind occurrence
through actual MAPLE physics over the current shared bed. It returns a staged
bed, native elapsed time and accounting. It does not yet qualify the combined
daily wind–water loop: SYRUP's Phase 2 changed-bed lifecycle operation remains
under development. The existing water-only daily executor still refuses wind.

## Accepted behavior

The executor drives the native wind blocks in row order at the actual physical
origin and requires native wind finalization. Repeated uses of one forcing file
have distinct occurrence identities. It preserves the complete evolving bed,
active layer, sediment availability and prescribed vegetation instead of
reinitializing from a DEM. Native mobile inventory must be resolved; a guard or
failure is not event completion.

Wind assumes dry soil as directed by the user, including over ponded cells.
An explicit wind-only configuration clone disables moisture feedback and uses
MAPLE's existing `constant_depth` bed-update rule. The original case/configuration
is unchanged, and both choices are recorded in the receipt. This mapping is
necessary because native `constant_free_surface` can change water depth on a
bed commit even when water coupling is disabled. No water array is zeroed or
rewritten to permit wind. Carried surface-water fields are bitwise preserved;
water-borne sediment must be empty. Layered soil water stays outside this wind
component and belongs to the pending native SYRUP handoff.

All transport, finalization and conservation helpers come from MAPLE. Per-block,
event and whole-bed class budgets retain native bounds. Explicit resource caps
precede dense staging and the configured mobile-pool allocation. Failure keeps
the accepted input bed unchanged and returns bounded scalar evidence. The
returned candidate does not publish a lifecycle, advance a day cursor or provide
a combined restart.

## Verification

Codex ran the final main-tree checks with the existing MAPLE CPU interpreter,
Python 3.12.3 and NumPy 2.5.3, isolated imports, disabled source bytecode writes,
run-owned caches and no installation or GPU use.

| Check | Result |
|---|---|
| All daily tests with explicit genuine native source pair | 93 passed, no skips |
| Wind tests within that run | 15 passed: 5 metadata/import checks and 10 native cases |
| Separate direct-native controls | 3 complete final beds matched bit-for-bit |
| Whole-repository check without native source selection | 535 tests; same 1 stale SHA failure and 32 missing-native errors as the pre-existing baseline, 50 skips; no new failing test IDs |

The wind checks use the authored native 4 by 6 Dense NumPy case with three grain
classes and restricted, nonuniform sediment availability. They exercise actual
nonzero transport, preserved ponded water, multiple blocks, repeated files,
nonzero physical origins, the explicit dry-wind configuration, resource limits,
water-mobile/configuration refusal and injected native/finalizer failure.
They do not exhaust every post-physics refusal code.

Codex also wrote a separate direct MAPLE block/finalizer control before reviewing
the candidate executor. Using the same genuinely admitted source and starting
bed, the control and the parsed-CSV executor produced bitwise-identical complete
beds for dry cells, 0.001 m ponding, and 0.001 m ponding with an admitted
moisture-enabled flag disabled by the explicit wind view. These controls start
at 86400 seconds and include two forcing blocks. They are external task evidence,
not additional repository unit tests. N15's restriction to neutral prescribed
threshold fields remains intact; these controls do not qualify non-neutral
moisture fields or a moisture response law.

Claude authored the initial component in a bounded source-only session. Codex
inspected native callers, reproduced the water-depth/configuration issue,
corrected the implementation and fixtures, and ran the numerical checks. A
separate read-only Claude review of the final numerical source found no blocking
defect. Its observations about external test evidence and conservation wording
are reflected here and in the executor guide. The accounting-split clock
rounding check matches the existing `dense_runner`; it does not relax physical
or conservation tolerances. Reviewers did not run the tests themselves.

## Sources and retained evidence

MASY baseline: `9d358e5f92b85c0fb8d655f328cc6c93b3039a33`.
The qualified `src/masy/daily_wind.py` SHA-256 is
`557a41b8a7f67b79c3b6b5e3e030b63804343820a49bdf4f2a64613c86c61aa5`.
Existing runtime files, locks and defaults are unchanged. Only the README and
wind-moisture documentation change among existing tracked files. Pre-existing
dirty work is preserved.

The explicit component qualification pair matches the published daily water
connection:

| Source | Identity |
|---|---|
| MAPLE commit | `9f466d2cfcd52991b24496cf67f4562d6268afa9` |
| MAPLE package SHA-256 | `fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e` |
| MAPLE immutable bundle | `5d850789f9e82dc456227cb1eb4dfcc255b3ee4dfb0ed74883813c93e98d0ab3` |
| SYRUP N15 CPU package SHA-256 | `82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107` |
| SYRUP source commit | `06952dbc2f984af8fb75dbf09a9c9f7fdbba23fc` |

These are explicit historical qualification sources, not a claim to use newer
live upstream HEADs. Source, case, wrapper, configuration and module-origin
guards were retained and checked. The actual new Phase 2 pairing will require
its own qualification, without rebinding old profiles.

Local evidence is retained under
`agent_handoffs/tasks/daily_wind_integration_20261112/`: exact baseline and source
manifests, author/reviewer prompts and sessions, submitted-content hashes,
separate raw stdout/stderr, PID/start identity and reaping records, failed
attempts, direct controls, final native tests and the whole-suite comparison.
Only relevant source/design files and authored illustrative templates were
submitted to Claude; saved simulation payloads stayed local.

## Remaining acceptance

The [Phase 2 handoff requirements](daily_wind_phase2_handoff.md) define the native
operation needed to adopt each completed wind bed and actual elapsed time while
retaining soil-water lineage and accounting. After that implementation is
qualified, MASY can connect wind → changed-bed refresh → reduced/full rain →
daily water, with failure recovery and exactly-once combined restart.

No GPU, windowed storage, large-domain performance, full coupled daily run,
LEAF transaction, scientific calibration or field validation is qualified by
this component milestone.

Usage: [native wind API](daily_wind_executor.md),
[daily inputs and inspector](daily_scheduler.md).
