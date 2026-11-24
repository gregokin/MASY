# Moab/Jornada computational readiness — 2026-11-24 UTC

**Status: integration interfaces connected; the full target scenario is not computationally qualified.** The actual sealed CUDA runner accepted two days, including wind, ET and the exactly-threshold reduced rain. Its full storm was still in recession when the one-hour CPU guard stopped it. A limit is not water-event completion. The scheduler correctly withheld the subsequent wind and daily processes.

## Tested sources and case

These integration revisions include the current local `main` revisions checked for all three upstream projects. Existing active upstream checkouts and unrelated work were preserved. Publication is on integration branches; upstream `main` was not moved.

| Project | Tested integration commit | Package SHA-256 |
| --- | --- | --- |
| MAPLE | `c4b9343706240d3f98e9abb9c7274958a45438cc` | `f4e4dc694dff1bbf126ddbb69a8cd0bc089d8006ebb5f4ed2ce1ac30e8e5b8bc` |
| SYRUP | `241c14fb8cb04bb74bce7775c090aeb04844424f` | `d20889144c5955ca6ad3b6eb064baedcf66d3958eadea1169266c3be6bdb06b8` |
| LEAF | `7b744951c42518e0f09bbf30304faa64a0ec5cd3` | `069f711aade205c4db3a2b044020361dfb289613bad9434e97140037be1daf25` |
| MASY | `672fb2f70a5ed0212bad1a918129454a19ed8ab0` | `a5ae5600dfa8288406a5a72281ad96079c0c2daf359235897bb93d9adeddd9ed` |

MAPLE uses branch `integration/masy-persistent-terrain-20261123`; the other projects use `integration/masy-irregular-20261123`. Later documentation commits do not change the model package bytes tested here and are not substituted into the physical-run evidence.

The fixture is an exact 128 × 128 crop of MAPLE's Moab Stream 3 DEM and base at 0.25 m spacing: 32 m across, 13,125 active cells, a real observed exterior ring and an irregular physical footprint. The wind follows the native frozen regional-plane and local FFT path with native completion and 16 m padding. The selected regional route is `flat_reference_residual` / `open_support`; it does not apply a separate regional speed correction. Periodic wind boundaries and closed sediment surface edges remain distinct native settings.

Native LEAF initialization generated 10,146 plants, targeting 10% grass and 10% shrubs over the active footprint. Realized cover was about 10.002% grass and 9.822% shrubs, within the explicit tolerance. Roots use the user-approved remapping of the documented curves to three 10 cm layers, normalized within the modeled 30 cm. Soil properties come from the selected Jornada preset and explicit native supplied-property maps.

The eight nominal days contain two 0.5 s wind events, a 5 mm reduced-rain day and a 6 mm full storm at 360 mm/h, with daily supplied PET and a weekly growth cadence. These are illustrative computational inputs, not calibrated scientific forcing.

## Actual bounded run

The normal run used the genuine cold, sealed `tools/masy_triplet.py run-terrain` path, with exact source/module checks and run-owned caches. It used the existing MAPLE CUDA environment; no environment was installed or replaced.

- Lock SHA-256: `2c868b24662d6c5aeec4b55028a51d34372b10b4b6cb55136692f4f64f988945`.
- Input manifest SHA-256: `c9a31da51d8d1e53a76b41ab2dc6d2e32c0917c8483c96154f53be32694434ee`.
- Selected physical device: GPU 3, UUID `GPU-fcbcc375-a471-a9c3-ab1c-eed25d898ef0`, freshly idle before launch.
- CPU guard: 3,600 s; wall guard: 3,660 s; host RSS guard: 3 GiB; driver device-memory guard: 5 GiB; CuPy pool cap: 4 GiB.
- Result: `SIGXCPU` (`-24`) after 3,634.44 s wall time. The owned process was reaped and the GPU allocation released.
- Accepted days: **2**. Last accepted boundary time: 172,800.5 s.
- Active storm time at stop: 173,489.5 s — **689 s of event time**, comprising 60 s rainfall and 629 s recession, through **13,633 accepted native steps**. The native completion flag remained false.
- Peak host RSS: 1,670,316,032 bytes; peak driver device memory: 984,612,864 bytes, both within their guards.

The first native wind, finalization, accounting publication, terrain handoff, daily water/LEAF feedback and ET, and the exactly-5 mm reduced rain were accepted. Native rainfall and recession ran without a reported solver, source or conservation error before the CPU guard. This does **not** establish a completed storm budget, the subsequent wind, the weekly growth update or the full eight-day result.

Short source-identical diagnostics located substantial work in native sediment-ledger, transport and soil-depth operations. A separate Python profiling-hook experiment crashed and was excluded from physical acceptance. The final normal run used ordinary scalar progress reporting, with no profiling hook. No solver law or completion threshold was relaxed to obtain a pass.

## Reviewed changes and limits

Codex and Claude implementation/review cycles covered the native terrain source descriptor, passive-water wind carry, the LEAF active-mask initializer, explicit supplied soil properties, native boundary publication, outlet closure and the thin runner. The final native outlet contracts passed five CPU tests; wrapper input/runtime regressions passed 19 tests and 38 subtests; cold lock tests passed 12 tests. An actual native CUDA boundary-component check preserved real water bitwise and the event clock. A native chunk comparison verified exact end time, rainfall accounting and water/sediment closure across the smaller progress targets. Additional upstream qualification and dispositions are recorded in the task archive.

The user-approved `close_uphill_at_water_boundary` approximation preserves the observed ring and closes unsupported outlet faces before water processing. It conserves water and sediment inventories and preserves pending-recession status. Its native full-event completion remains authoritative. The **SYRUP developer follow-up is open** at `docs/subsurface_hydrology/masy_boundary_followup_20261124.md` on the SYRUP integration branch: evolving stage/backwater boundaries, reopening, in-event changes, flux treatment and restart need scientific decisions and separate qualification.

The dry-wind moisture approximation remains explicit: wind receives dry-soil entrainment behavior over real wet or ponded cells without erasing water. Missing moisture suppression is a known issue.

Before operational use of this target case, investigate native water-event runtime and the completion metrics, then finish the water–wind–ET–growth scenario and its independent output budgets. Annual vegetation orchestration, large/windowed capacity, combined restart and scientific validation are outside this result. No model process or automatic resumption is left running.

Detailed source/process/review receipts and local-only model payloads are under `agent_handoffs/tasks/current_triplet_irregular_20261123/`. The final partial result is `runtime_07/physical_01/readiness_result.json`. Terrain, forcing payloads, arrays and raw logs remain local.
