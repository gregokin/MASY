# Native LEAF-calendar participant (candidate; unqualified; CPU only)

Task `phase1_leaf_calendar_impl_20261112`. Design: `../Design.md` (task directory). Status: **partial implementation, not accepted, Phase 1 not complete.** Read the limits first.

## What exists (and is tested on CPU in a fresh genuine child)

| module | role |
|---|---|
| `masy.leaf_source_profile` | NEW named profiles `leaf23_d4fe_water109_native9f_cpu_v1` / `..._cuda_v1` (same bytes, distinct capability). LEAF = git base `1e468e14` **plus the qualified scoped delta** (package digest `d4fe6cd2…`, 23 modules + `pyproject.toml`, 24 literal hashes); SYRUP = water109 `afcad536…`; MAPLE = native9f bundle through the genuine `native_case_profile`. Strict cold order (MASY manifest census → LEAF census before any LEAF import → import → association → native profile → tree digests), `check_hot` per call. The old 21-module reference fails the literal census. |
| `masy.leaf_calendar` | Pure wrapper of ONE `leaf.biocalendar.Schedule`: `plan_open` (midnight/grid/new-year splitting + the actual `OpenStep`), `transport`, `flush_required`/`flushed`, `exact_datetime` (refuses non-microsecond instants, no rounding). |
| `masy.leaf_water_interval` | ONE hydrology+E advance then ONE actual T withdrawal per eligible fragment (real `wl.evolve_interevent`, `la.layer_availability_m`, `up.propose`, `wl.request_withdrawal`, `up.accept`), per-plant **canopy-area-weighted terrain PET** (`plant_pet.canopy_pet`, m/s→mm/day once) → `growth.demand_per_plant`, credit only when growth is enabled, season turnover / boundary growth / flush with LEAF's `rebuild_geometry` afterwards, `bind_head`/`release_head`/`carry_boundary` (no `make_head` after the first boundary), compare-and-adopt. |
| `masy.native_water_participant` | `run_interevent` (enter → fragments → look-ahead flush → close → release), `carry_through_rain`; `WATER_ONLY`/`LEAF_WATER` capability constants. Bounded step budget **refuses** the occurrence (no partial T, no replay). |
| `masy.leaf_biology_codec` | Biology-only codec (`biology.json`, `base_plants.csv`, `arrays.bin`), closed 3-member layout with derived limits; decode rebuilds roots/canopy with LEAF's `rebuild_geometry` and requires identical digests, re-validates the `Schedule`, joins the water **state digest + continuity mark**, request **anchor + delta**, optional accepted-cursor join. |

Physics preserved (user decisions, not re-asked): prescribed hydrology cover/species maps (feedback is Phase 2.5); canopy-weighted terrain PET per plant now; growth disabled ⇒ T still withdraws but **no** credit and prior credit is preserved; no E/T/growth in rain/recession/wind/catch-up; equal access per cell/layer, no root compensation (the 100/0.5/0.5/25/200 ⇒ 75 example is reproduced exactly with E off; after a REAL storm the withdrawal is larger, not wrong).

## Interfaces (new; nothing old widened)

`prepare_interval_operator(receipt, lop, *, table, leaf_budget_bytes, catalog, geometry)`; `new_boundary(op, water, ...)`; `bind_head(op, leaf, water, *, parent=None)`; `run_interval(op, head, PlanItem, *, pet_terrain_m_s, interval_token, transpiration, growth_enabled, evaporation, ...)`; `apply_growth`; `adopt`; `release_head`; `carry_boundary`; `codec.encode(op, leaf)` / `codec.decode(op, members, water, *, calendar_cls, expected_cursor=None)`. The old accepted modules/records/defaults are byte-identical (all 246 baseline files unchanged; only new files added). The codec stages its payload and the CSR host export in full host memory, admitted before allocation (no chunked D2H claim)..

## Refused / unsupported in this candidate (typed)

growth enabled with T off (no water-limited law; water-free growth is never invented); mortality/recruitment/disturbance laws; wind (the water-only wind refusal of the session stays); non-microsecond-exact instants; rebuild budgets that do not fit; a hydrology budget too small for the occurrence.

## NOT implemented (proposed remaining engineering — Root decides)

1. **Session wiring**: optional participant on `NativeWaterSession.create/resume` (pair becomes `(coordinator, state, LeafBoundary)`, `_Progress` carrying the working head, hooks in `_run_interevent/_run_rain`, `check_scenario` accepting T/growth gates only with a participant, capability/config digest only when present). Design.md §5; the pure orchestration it calls exists and is tested.
2. **Joint artifact layout**: generalising `native_water_checkpoint._record_physical/_rollback_physical/_publish_engine` to a second closed member directory `biology/` under the same held-FD/no-clobber/rollback engine, manifest `biology` section, closed fd caps. Only the biology codec (bytes in/out, derived limits) exists; the V2 physical restart is exercised through the actual unchanged `lifecycle_checkpoint_v2` API, **not** through `NativeWaterSession.checkpoint`.
3. **Config/CLI**: optional closed `leaf` block in `native_water_run_config`/`native_water_worker`.
4. CUDA: the CUDA receipt/profile id exists and the CUDA code paths of the interval adapter are authored (device plans, host staging) but **never run**; no GPU test exists. The base C4 GPU size-only schedule defect belongs to the C5 task.
5. Known measured GPU cost (reference adapter): one `(L,N)` request SHA D2H per biology transaction (≤1 MiB stream); no "zero full-field transfer" claim.

## Verification (existing MAPLE `.venv`, CPU, `CUDA_VISIBLE_DEVICES=''`, fresh child per run: `../run_child.py`)

See `../report.md`.
