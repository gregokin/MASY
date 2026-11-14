# Daily wind/live-water interface qualification — November 13, 2026

This qualification covers the additional public SYRUP wind-handoff operation and the opt-in MASY coordinator
`masy.daily_wind_handoff.cpu.v1`. Actual MAPLE wind now enters the current native Phase 2 water lineage with
its changed bed and actual physical clock. The [interface guide](daily_wind_handoff_interface.md) gives the
API, authority, resource and checkpoint-publication contracts.

## Exact source scope

The native extension starts from published SYRUP main `02388adb70b7f1f22b33b6b3d5f1249e223842e5`.
The interface is published in SYRUP commit
`953e6fdf85b73e3fdf5a1ac7b0d97dea4df7dab1` on main. All 169 existing package files remain byte-identical to the Phase 2 base. The single new module is
`maple_syrup.floating_wind_handoff`; its tests/docs are additive. The new 170-file package digest is
`0bc7dc17c7eaa0e7445896759654bb3fff0649a14bcf43e79115b0cee2afed76`, recorded in the owning
`docs/subsurface_hydrology/wind_handoff_source_manifest.json`. This does not rewrite the Phase 2 release manifests.

The explicitly selected immutable MAPLE dependency is commit `9f466d2cfcd52991b24496cf67f4562d6268afa9`,
412-file package digest `fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e`.
It is the existing source-qualified native dependency, not the newer live MAPLE main. Exact source/runtime/module
origin guards remain enabled. Historical N15 factories retain their original literal digest and refuse this native
package. The water-only runner and old locks/defaults are unchanged.

MASY changes are one new coordinator, a small wind-component extension exposing the actual native finalization
result under explicit request, profile-specific source checks, tests and documentation. No numerical kernel,
conservation tolerance, wind/water response law, soil-property law or case default was changed.

## Executed checks

| Check | Result and scope |
| --- | --- |
| Native wind handoff suite | 68 passed on the final source; actual native bed/ledger exchange fixtures, soil refresh/return/export, integrity, admission, failure and interruption controls. Mechanical native-result fixtures are not claimed as real wind execution. |
| New MASY interface | 15 passed on the final source; actual CSV/MAPLE wind, retained 1 mm ponding, live W/J/I and routing, same-file same-day order, reduced rain/day and full storm/recession/day between winds, native summed sediment bounds, failure/retry and exactly-once cursor. |
| Native file plus MASY cursor restart | Included above: restored next actual wind matches continuous execution in bed, W/J/I, ledger, time and exact wind exchanges, with no replay. |
| Fresh native process | Included above: a separate interpreter restores the native file/context and completes the next daily update identically; original terrain/bed, water-owner and soil-property initialization are disabled. |
| Existing native lifecycle/bridge/checkpoint | 28 passed; eight Numba parameter cases explicitly deselected. These regressions do not qualify the new seam on Numba/windowed/GPU. |
| Historical daily source profile | 97 passed, including the original daily CSV/water/wind paths and new metadata checks. Eleven new-profile native cases deliberately skipped in that interpreter and executed separately on the correct new source above. |

The CPU interpreter was `/home/okin/MAPLE/.venv/bin/python`, Python 3.12.3, NumPy 2.5.3,
SciPy 1.18.1, Numba 0.67.0, llvmlite 0.49.0 and PyYAML 6.0.3. Native imports came from the immutable MAPLE
bundle and isolated exact-source SYRUP worktree. Bytecode/GPU access were disabled and caches were task-owned;
no environment was installed or updated. Source and loaded compiled-extension hashes are archived locally.

The small 4 × 6, three-class control uses explicit authored test parameters, partial availability, actual native
sediment exchange and real soil history. One full-storm control uses an explicitly smaller threshold to keep a bounded
engineering test; the shipped weather example remains 5 mm with equality attempting reduced recharge. These tests
establish interface mechanics, not calibrated erosion, vegetation or field-scale scientific validity.

## Review and retained evidence

Claude authored the native seam and a bounded correction; Codex reproduced defects, independently inspected callers,
added transaction controls and implemented the MASY coordinator. Separate source-only Claude review sessions cover the
combined implementation and final memory-admission correction. The full independent review found no high/medium defects and four low findings. Codex reproduced and fixed them;
the final independent review accepted every disposition and found no new defects. The memory correction was also
independently reviewed. Exact sessions, source hashes, checks and dispositions are recorded in the
[source/acceptance receipt](daily_wind_handoff_sources_20261113.json).

Preserved failed attempts include an invalid authored ledger fixture, a test that incorrectly compared independently
minted native state tokens, actual API/texture/table fixture errors, and an isolated test snapshot missing the unchanged
CLI launcher. All were corrected without relaxing native source guards, model tolerances or physical completion.
A final resource probe exposed a 1 MiB strided hash copy reserved as 64 KiB; the final source reserves that bounded
chunk in scan, selected-candidate and adoption admission and has a direct regression.

Local exact prompts, submitted file/content hashes, original/corrected source snapshots, raw separate Claude output,
process PID/start identities, exits/reaping and test supervisors are retained in
`agent_handoffs/tasks/daily_wind_handoff_20261113/`. Actual simulation payloads stayed local. Unrelated dirty MASY,
live SYRUP and Phase 2.5 work were preserved; no jobs were altered.

## Limits

This acceptance is an interface on Dense NumPy CPU. The caller drives real rain/recession/daily APIs on the same lineage
and atomically publishes the physical native checkpoint with its externally sealed coordinator record. The adapter
neither supplies an entire scenario execution CLI nor a complete fresh-process wind-runtime provisioner. It never
runs a no-op LEAF model. LEAF bridges, active-event/combined atomic publisher, Numba/GPU/windowed scheduler acceptance,
large-domain performance and scientific calibration remain separately scoped.
