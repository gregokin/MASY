# Initial upstream interface observations — October 16, 2026

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
