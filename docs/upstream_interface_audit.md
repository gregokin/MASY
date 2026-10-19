# Initial upstream interface observations — October 16, 2026

**Current qualification:** [MASY-09 committed Dense CPU pair](source_pair_20261019.md), MAPLE `39bcf79`
+ SYRUP `d05d8c6`, MASY runtime `22aa6ab`. The dated sections below preserve their original observations;
present-tense blockers in those sections describe those historical revisions.

Read-only source inspection during setup; API candidates and blockers, not runtime compatibility certification. Full observed source manifests/revisions are archived in `agent_handoffs/tasks/bootstrap_20261016/upstream_baseline.json`. Live checkouts can change after this observation.

| Area | Observed source | Implication |
|---|---|---|
| Native wind dispatcher | MAPLE `src/maple/aeolian/scheduler/block.py`, `advance_wind_block`; `record.py` for whole records | Call actual scheduler; select supported upstream process mode |
| Wind forcing | MAPLE `src/maple/core/parameters/wind_forcing.py`, `WindBlock` / resolver | Shear velocity m/s, clockwise-from-north direction, duration-derived times; ordinary wind speed is not interchangeable |
| Wind physical state | MAPLE `src/maple/aeolian/scheduler/state.py`, `SchedulerPhysicalState` | Carries voxel column, active layer, mobile cohorts, ledger, committed topography, availability and optional water/vegetation; annotations alone do not prove resident-window support |
| Wind restart | MAPLE `src/maple/aeolian/scheduler/snapshot.py`, save/load scheduler snapshot | Preserve genuine scheduler continuation and process caches; do not assume arbitrary water state can enter loader |
| Water bed | SYRUP `src/maple_syrup/sediment_bed.py`, `BedState` / `BedContext` | References actual MAPLE column/active layer/water/ledger/topography/availability, supports distinct dense and windowed paths |
| Water lifecycle | SYRUP `src/maple_syrup/complete_event.py`, `complete_event`, `DryHandoff` | Can return wet pause; only actual complete dry handoff permits wind; duration failure retains wet evidence |
| Water restart | SYRUP `src/maple_syrup/checkpoint.py` | Native checkpoint binds identities, wet/dry state and referenced window storage |
| Specialized storage dependency | SYRUP `sediment_bed.py`: paired transaction and deep-storage source checks | Uses private `maple.water.windowed_transaction._apply_windowed_pickup_transport_deposit` and exact MAPLE digest guards; current latest MAPLE is not automatically qualified |
| Existing MAPLE water coupling | MAPLE `src/maple/coupling/water_event.py` | Shared sediment-demand application is not SYRUP's full storm solver; avoid duplicate water passes during wind |
| Package naming | SYRUP `pyproject.toml`: distribution `maple-syrup`, import `maple_syrup` | New combined distribution/namespace must be distinct (`masy`); do not install a conflicting second maple-syrup distribution |

A concrete storage gap was confirmed by file existence checks in the current live MAPLE checkout: `water/windowed_transaction.py`, `water/windowed_step.py`, `surface/voxels/windowed_snapshot.py` and `surface/voxels/deep_store.py` are absent. They exist in SYRUP's separately qualified MAPLE dependency lineage. Consequently current live MAPLE is not a drop-in replacement for those windowed SYRUP paths. This observation does not establish failure of the small dense water/wind path, which still needs runtime qualification. Any shared windowed capability should be developed/adopted in actual MAPLE through a scoped task, not copied into MASY or enabled by disabling guards.

Unresolved first-task questions:

- Can current committed MAPLE plus current SYRUP operate on the same supported MAPLE source with their configured features? Determine concrete failed contracts rather than bypassing pins.
- Which actual public initialization/update APIs accept an existing evolving bed in each process, without reloading original case state? Standalone case CLIs are not evidence of this.
- Can current wind execute on SYRUP's resident-window storage? If not, prove the small dense path first and scope the missing upstream interface separately.
- How does native MAPLE finish an event with mobile airborne cohorts, and how much physical time does that require? State disposition must be conservative and recorded.
- Which availability, cache, committed-surface and checkpoint fields must cross each boundary? Which remain process-local? How will failed upstream mutation be isolated?
- How should local rainfall clocks and native wind cumulative record clocks map to one global elapsed timeline, including recession and settlement?

Known carry-over limitations: conservative SYRUP is separate from its default legacy replay; no accepted wind handoff exists yet. Chastre's zero-outlet/no-spill storage case is not a physically realistic catchment drainage demo. Event-end water removal is an explicit accounted model assumption, not simulated inter-event ET.

---

## MASY-01 current-source finding — October 17, 2026 (historical: observed at SYRUP `9fabdca`)

*Historical.* Everything in this section, including its "current pair", "33 items" and present-tense wording, describes MAPLE `5b1a052` + SYRUP `9fabdca`. The Windowed import mismatch was resolved by owning SYRUP at `2c60c73`; see the MASY-02B section below.

Appended by MASY-01; the October 16 observations above are kept unchanged as history and are not restated as current acceptance. Read-only source and interface observation only; nothing here is runtime, physical or transport qualification.

**Observed sequence.** MASY-01 first captured a clean pair: MAPLE `f1555e09e7d6e49dadb2413b36a1700d4f3d0ea5`, SYRUP `9fabdca60bd9cd7c5b358d5cbf620dbd3209d93a`. An unrelated MAPLE edit (`src/maple/core/parameters/flow_separation.py`) then made the MAPLE scope dirty, and the MASY preflight correctly refused with `dirty_scope` before importing anything. That work was later committed independently, outside MASY: current MAPLE is committed `5b1a052e1b9a9edfed0760152aac2a743813e79a` with a clean `src`/`pyproject.toml` scope. SYRUP is unchanged at `9fabdca`. Codex's audit saw identical source observations before and after.

**Native provenance.** SYRUP's unmodified `resolve_maple_dependency` still refuses the current pair: the installed MAPLE distribution metadata yields `source_root=None`. MASY reports this and does not bypass or repair it. SYRUP's native `check_required_api()` (33 items) passes.

**Confirmed Dense type-import mismatch.** SYRUP `sediment_bed.py:610-613`:

```python
def is_windowed_bed(state: BedState) -> bool:
    from maple.core.types.voxel import WindowedVoxelColumnState

    return isinstance(state.voxel_column, WindowedVoxelColumnState)
```

The import runs before the column is inspected. `bed_inventory`, `apply_bed_demand`, `bed_by_cell_class`, `bed_voxel_count` and `bed_already_committed` call it first, so Dense beds hit it too. The committed MAPLE `core/types/voxel.py` defines only `VoxelColumnState`. The voxel and `sediment_bed.py` source hashes are identical in the initially archived pair and the current pair, so the mismatch holds for both. Codex reproduced the actual `ImportError` on explicitly identified archived source copies with a plain object, before any bed access; no model state, arrays or transport were loaded.

- SYRUP's native API list does not cover this import, which is why it passes. The MASY required list now includes `VoxelColumnState` and `WindowedVoxelColumnState` as genuine-class requirements, plus the exact Dense helper signatures, and reports the missing type as `masy_required_api_missing` (`incompatible`), naming the caller. Overall the audit still refuses, because of the native provenance failure. Both findings are retained.
- This is distinct from the absent optional paired-transaction module `maple.water.windowed_transaction` (still `info`). Requiring the type does **not** make full Windowed storage required or supported for Dense. The October 16 storage-gap paragraph still applies to the windowed paths.
- The requirement is tied to the observed SYRUP caller. An accepted owning-SYRUP design proposal (design-only, under independent review; no upstream file is changed by MASY-01) would recognize the genuine Dense MAPLE class without importing an unavailable optional Windowed class, refuse unknown/foreign representations, and keep the deep-storage/source/transaction guards unchanged. If such a change is committed, a newly qualified MASY requirement update reflects that actual source. This is not permission to fake the class or to bypass private, deep-storage or source guards.

**Still open.** Which sources a later actual run uses (committed sources, or explicitly labelled development snapshots) is pending with the user under AGENTS.md. Physical qualification needs a declared qualified source pair, an immutable runtime, real bed/clock/cache/routing adapters and actual conservative tests. No code is physically qualified.

---

## MASY-02B committed adoption — October 17, 2026

Appended by MASY-02B. The sections above are history and are not restated as current acceptance. This is source and interface observation plus a MASY requirement-table update. It is not runtime, physical, restart, backend, performance or transport qualification.

**Committed pair.** MAPLE `5b1a052e1b9a9edfed0760152aac2a743813e79a` (package `maple`, 334 files, digest `97b3df72336e6313e8d89d0b7b20784b9a0f001e30a4fc3035360e6de6068612`) and SYRUP `2c60c73b01fe1c0cef576987d52010e84058c568` (package `maple_syrup`, 63 files, digest `fc789e240f7b0061e9ac80bc9078b7fe42d302fc8309ee5746073455f7f9f6bb`). Both are clean upstream commits; SYRUP's is committed and pushed, following `1f70a87`. The SYRUP package digest equals the final MASY-02A integration digest, and Codex verified that the committed tree matches the MASY-02A acceptance file hashes. MASY changed no upstream file.

**Adopted Windowed classifier.** SYRUP `sediment_bed.py` (SHA256 `0015f7fe…`, lines 613–644) now imports `maple.core.types.voxel` normally and reads the literal module namespace. A truly absent `WindowedVoxelColumnState` permits genuine Dense. `VoxelColumnState` is required. A present non-class binding of either name raises `BedIntegrationError` on every bed path, and an import failure of the module propagates. Exact class identity is required, so subclasses and foreign objects are refused. MASY now lists Windowed as an `optional_class` requirement (absent, or bound to a class) instead of a required class. The MASY-01 missing-Windowed incompatibility therefore no longer applies to this pair. MASY applies no classifier and constructs no column.

**Compact-ledger checkpoint blocker.** SYRUP `checkpoint.py:179-190` (`_registry`) and `pond_checkpoint.py:694-698` (`_native_registry`) both import `maple.core.types.sediment_ledger.CompactSedimentLedgerState` unconditionally, so both native restart codecs need it, Dense beds included. Current MAPLE `5b1a052` lacks it, and Codex independently reproduced the failure in both registries without transport or arrays. SYRUP's own `REQUIRED_MAPLE_API` does not list it. The class is now a required class in the MASY table, naming both callers. Its absence is `masy_required_api_missing` (`incompatible`). The native codec is not made optional or bypassed.

**Native metadata refusal still unresolved.** SYRUP's unmodified `resolve_maple_dependency` still refuses the live current MAPLE (`source_root=None`); MASY reports this and does not repair it. No combined runtime or adapter exists. MASY reads the actual length of SYRUP's required list and hardcodes no count; 36 is the value observed in Codex's receipt.

**Verified candidate preflight.** Codex ran this MASY candidate against the actual clean current committed sources. The final adopted-root preflight confirmed the same result; receipts are in [agent_handoffs/verification.md](../agent_handoffs/verification.md). Observed:

- verdict `refused` (exit 2);
- MAPLE `5b1a052e` and SYRUP `2c60c73` observations unchanged, no wrong module origins;
- SYRUP's own API list (36 items) passes;
- the optional Windowed requirement passes, with the class genuinely absent;
- the only failed MASY item is `CompactSedimentLedgerState` (attribute missing, both checkpoint callers named);
- optional transaction `info`;
- native resolver refusal (`source_root=None`), independent of the above;
- `transport_ready=false`, with all standing blocking findings.

Codex's full stdlib candidate suite also passed (170 tests, exit 0). Those are synthetic unit tests, not physical, transport or GPU tests. No transport or restart was qualified.

No pair is reported ready. Under AGENTS.md, a new run selects the current local committed HEAD of each configured upstream, subject to compatibility and immutable-runtime qualification; a development snapshot requires an explicit selection with recorded dirty deltas. No run has been selected or launched. MASY now has a Git baseline (`d0c2b71`), but its source observer still uses file manifests and an explicit lock, not Git binding.

---

## MASY-03 checkpoint helper and runtime prerequisite — October 17, 2026

Appended by MASY-03. The sections above remain historical. Codex independently qualified the CPU source/interface delivery and native Dense checkpoint prerequisite below. This does not qualify combined restart, alternating transport, GPU, windowed coupling or performance.

**Owning-SYRUP checkpoint helper.** The narrow owning-SYRUP patch adds `maple_syrup.checkpoint._native_ledger_types`, shared by `checkpoint._registry` and `pond_checkpoint._native_registry`. It requires `SedimentLedgerState`; `CompactSedimentLedgerState` is included only when literally bound to a class. Invalid present bindings and broken imports still fail. The patch was implemented and reviewed in an isolated SYRUP branch, then adopted on local main as `b4ee14e5bf37814efb0d754de17b7ea604b1cd74`; the SYRUP remote remains unchanged. No physics, codec/schema body or source/storage guard changed. Current MAPLE `5b1a052e` can now use both Dense checkpoint registries; genuine reference compact/windowed behavior remains covered separately.

**MASY table.** `SedimentLedgerState` is now a required class, `CompactSedimentLedgerState` an `optional_class`, and `checkpoint._native_ledger_types` a required callable. An unpatched SYRUP `2c60c73` lacks the helper and stays `masy_required_api_missing` by name, with nothing called. The checkout audit (`contracts.run_native_audit`), its report schema, statuses and filter are unchanged. The historical receipts above (the compact class as the only failed item) were not re-run with this table.

**Runtime prerequisite.** The separate [runtime tool](runtime.md) seals all three committed source scopes, reuses the unmodified upstream `dependency_bundle` and `check_runtime`, and selects matching metadata from the frozen objects. The native resolver, actual API lists, both registry probes and all package origins passed. MAPLE package digest is `97b3df72336e6313e8d89d0b7b20784b9a0f001e30a4fc3035360e6de6068612`; SYRUP is `6973938042b878449a7dce5744bef7aecc72cc73adda2dc73672b8ef149dc517`. Explicit old locks verified twice while both configured source roots were absent, and an older lock still verified after the wrapper generator changed. All old versions are retained.

**Validation.** All 245 MASY stdlib tests pass. Native checkpoint regressions: current 291 passed/8 skipped; genuine reference compact/windowed 404 passed/6 skipped. Three additional genuine wet Dense checkpoint save/load/continuation checks passed on the locked source/metadata paths, with exact persistent native state and continuation assertions, unchanged source objects and no unexpected native origins. These are bounded CPU fixtures. Full receipts, findings and limitations are in [verification](../agent_handoffs/verification.md).

---

## MASY-04 Dense coupling qualification — October 18, 2026

The original task baseline was MAPLE `5b1a052e` and SYRUP `b4ee14e`, both matching GitHub main at capture. SYRUP's
checkpoint fix is now pushed. Subsequent user-owned MAPLE main finalizer edits are outside this evidence; windowed
storage is being developed on another branch. MASY modified neither upstream tree or environment.

The explicit immutable CPU lock `172d9916c03c47ce44ab7c324b6e8dfe165329bffe567dd0eb5e574f6f9b0c28` supplies both native
packages and their actual metadata. MASY's new code is a separately sealed, explicitly uncommitted development test
snapshot. Actual package/submodule origins and native source/metadata identity are checked before and after the final
native checks. This is not a new-current-main or production-runtime claim.

Confirmed actual paths: MAPLE `initial_scheduler_state` and `advance_wind_block`; SYRUP `BedState` / `BedContext`,
`commit_bed`, `RainfallSchedule`, `complete_event` and genuine `DryHandoff`. One complete native Dense bed passes
through actual nonzero baseline aerodynamic wind, conservative upwind water plus physical recession, and another
wind. Native persistent fields, clocks, class exports and inventories match direct upstream execution. Partial
availability evolves natively. The retained water graph/source survive dry reset and wind; the actual `commit_bed`
decision retains their identities below its configured gate and reroutes above it. Piecewise rainfall also passes.

The old pair has no qualified instantaneous wind finalizer. The output lifecycle helper `commit_and_end_event` does
not receive mobile cohorts and does not supply that contract. MASY accepts a wind boundary only when the native pool
is genuinely empty; any residual, including a sub-resolution value, stops incomplete with the next process blocked.
The user's native MAPLE main fix must be committed and qualified before MASY calls its actual API. Calm coasting and
wrapper sediment deletion/deposition are not substitutes.

Accepted scope and limits are in [dense_runner.md](dense_runner.md). Combined restart, alternate wind/transport
modes, GPU/windowed storage and performance remain separate qualifications; the native hydraulic/scientific
limitations above remain in force. Full review findings and receipts are in
[verification](../agent_handoffs/verification.md).

## MASY-05 native finalizer qualification — October 18, 2026

MAPLE main `7b0e2539f6e57b5b0182280efeadc06488fde37a` is locally committed, clean in the runtime scope and unpushed.
With SYRUP `b4ee14e`, it passes a new immutable CPU runtime qualification, lock
`d29d5740b9633aae346eba53fef05cfeb8bd6376a55b8cb7d33a8dc4417f4eef`. Previous locks remain unchanged. The existing
Dense runner regressions pass 18 tests and 21 subtests on this pair. This does not claim that MASY calls the new API.

An independent native check confirms actual nonempty wind settlement at zero origin: class budgets close, mobile
state becomes genuinely empty, event-end clock/cursor are preserved, and a repeated call transfers nothing. An
actual subsequent wind after SYRUP recession exposes a completed-record clock limitation: it starts at 169 s and
ends natively at 171.00000000000045 s. The finalizer rejects its relative 2 s record because the native clock is not
2 s; putting its actual absolute block into a record fails the record validator's zero-based cumulative check.
Both failures preserve input arrays. No class, guard, timestamp or native binding was patched for this proof.

Full MASY finalizer integration therefore requires a narrow owning-MAPLE completed-event marker extension for an
explicit nonzero absolute origin. Placement must keep actual physical/cohort timestamps; no clock resetting or
fabricated prefix wind history is acceptable. The concrete contract and reproduction are in
[the MAPLE handoff requirement](../agent_handoffs/maple_finalizer_clock_requirement.md). MASY's accepted physics
code remains unchanged pending that API. Windowed storage stays outside this qualification.

Claude's source-only design launch hit its reported session limit, resetting at 6:50 p.m. America/Los_Angeles on
October 18 (01:50 UTC October 19). No completed Claude design or implementation review is claimed, and no automatic
retry/watcher was created. Exact new-pair receipts and independent native evidence are retained under
`agent_handoffs/tasks/masy05_native_finalizer_20261018/`.


## MASY-06 finalizer adapter and explicit origin (October 18, 2026)

The F1 completed-record origin gap is resolved by a narrow independently reviewed native extension proposal,
`finalize_wind_event(..., record_origin_s=0.0)`. It preserves local forcing validation, terminal cursor/final wind,
all physical/trajectory/class-budget/fluvial guards and all settling physics. It maps only completed-record end
bookkeeping onto the unchanged absolute native clock. Its reviewed patch passes `git apply --check` against current
MAPLE `7b0e2539`; the live upstream source and separate windowed work remain untouched. See
[owning-MAPLE handoff](../agent_handoffs/maple_record_origin_handoff.md).

MASY's actual adapter is adopted, uncommitted. It uses genuine native completion, preserves returned evolving
bed/availability/ledger, adds finalizer boundary export once and reports internal deposition without applying it
again. Native errors/incomplete outcomes preserve the accepted staged boundary and block water. The required API
now includes the finalizer's explicit origin signature/result. Actual unextended `7b0e2539` is refused before
transport; no older lock is rebound or silently selected. Native clock tolerance remains unchanged and can fail
closed at large origins/long accumulated runs.

The exact development pair `3eed7f55c5d474531a4dcc38d5f4492a546249845e9a494b1c1348f9d1203320` is explicitly uncommitted,
using genuine SYRUP bundle provenance and unchanged SYRUP `b4ee14e`. Proportional checks pass: 92 native finalizer,
20 runner + 23 subtests, 15 bed + 38 subtests and 264 stdlib. Genuine nonempty wind-water/recession-wind-second-water
matches direct native execution, including all persistent fields and clocks. Claude accepted the source patch
without blocking findings, followed by Codex's independent source/caller/real execution acceptance. Full review
notes/dispositions and exact lifecycle are in the MASY-06 task archive. This does not establish a normal committed
source pair, production CLI, combined restart, scientific calibration, GPU/windowed or large-case readiness.
Next is upstream origin-patch adoption/commit and new current committed pair qualification.

## MASY-07 / MASY-08 committed qualification — October 18, 2026 (October 19 UTC)

MASY-07 qualified clean committed `b3e8756` / `b4ee14e` / `22aa6ab` under lock
`b2cf01e16e80ca575a0f4351e94c170740fdfda2006965ef254f7bee511f2370`. During those checks MAPLE advanced to `39bcf79`;
its identity was recorded rather than silently rebound. The user then explicitly requested qualification of that merge.

MASY-08 selected current clean committed MAPLE `39bcf79e1e67b6ed960f78a923ddfedf1d173e68`, unchanged committed
SYRUP `b4ee14e5bf37814efb0d754de17b7ea604b1cd74`, MASY runtime `22aa6ab65ad73a58fb372ffe09e3f7e0ca9893a1`.
Fresh immutable lock: `ee190c813ae347a1f5c11e7ab5fad595dada898f3a13dddc775fca4488762a70`.
The old installed metadata correctly refused because it lacks the newly declared optional `windowed` extra;
actual source-built distribution metadata passes all unchanged native bundle guards. No install or upstream edit.

Execution passes 113 native settlement/origin/unsupported-storage tests, 15 bed+38 subtests, 22 runner+23 subtests
and three genuine SYRUP wet checkpoint continuations, with no skips. Exact nonempty four-event parity includes the
second wind at169seconds, unchanged native clocks, class/water budgets, final mobile emptiness, routing and failure
isolation/recoverable accepted state. Fresh explicit-lock verification passes and all six old locks remain unchanged.
Independent review status, actual metadata preparation, full evidence and bounded limits are in
[committed-pair qualification](committed_pair_qualification.md). No new transport physics or product tests changed.
Shared-storage libraries do not qualify windowed wind/water, GPU, large cases or combined restart.
