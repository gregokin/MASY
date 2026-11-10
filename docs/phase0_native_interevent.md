# Phase 0 native interevent qualification driver — limits guide

Status: independently qualified on NumPy, Numba CPU and GPU0 (67/67 checks each, 2026-11-10).
This is a qualification DRIVER, not a production adapter or whole-Phase-0 acceptance.
File: `tests/native/check_phase0_subsurface.py`
(SHA-256 `b89ff311939501e6b757b9efa04c676909668cd3b00dd75166a5e8af9db1775e`, 987 lines).

## 1. What the driver demonstrates

One actual dated MASY `Coordinator` (scheduler Phase 0A candidate) drives the EXISTING SYRUP soil column
(`maple_syrup.subsurface`) through interevent occurrences, with daily PET from the owning SYRUP weather library
(`maple_syrup.weather_pet`, Phase 0B candidate) reached through the actual `masy.pet` delegate. The column is small
but genuine:

| Item | Value |
|---|---|
| Layers | 3, depth edges 0 / 0.125 / 0.375 / 1.0 m |
| Hydraulic values (explicit, no texture rule) | theta residual 0.0625, field capacity 0.25, saturation 0.5, evaporation stress 0.25, D = 2^-22 m2/s per interface, evaporation factor 1.2, gravity drainage on |
| Cells | 4 flat cells, cell 2 inactive, cell area 6.25 m2 |
| Initial theta per cell | (0.375, 0.3125, 0.3125), (0.125, 0.25, 0.125), inactive (0.1, 0.1, 0.1), (0.0625 + 2^-10, 0.0625, 0.0625) |
| Intake | zero everywhere (no rain coupling) |
| Calendar | start 2028-02-28T18:00:00+00:00, native epoch 86400 s, three interevents: 12 h, 36 h, the same 12 h file again |
| Weather | scenario A explicit PET + mean temperature; scenario B `ecohyd_hargreaves` from tmin/tmean/tmax with latitude 32.6 bound from a MAPLE `case.yaml` location EXCERPT |

The first occurrence runs 18:00 → 06:00 across the leap-day midnight (2028-02-29), so every occurrence is a
partial-day interval split into exact calendar-day chunks; the three occurrences share one persistent column state
with no reset between them.

Per run the driver performs 67 checks (all passed by the author on numpy and numba):

* source isolation: `maple` resolves to the IMMUTABLE MAPLE39 bundle source, `masy` to the scheduler candidate,
  `maple_syrup` to the hash-verified overlay (see §3);
* exact clocks: every realized end equals the planned end bit for bit, the column time equals the native end, the
  ledger `endpoint_discrepancy` is `0/1`, and the exact rational chunk seconds sum to the ledger's exact elapsed;
  a non-representable exact duration (`Fraction(1, 3)` s) is refused before any native work;
* per-step native budget residual within the existing MAPLE-derived FP64 bound (`max |residual| / bound` 0.0055
  or less), aggregate closure over the whole sequence within `conservation.volume_roundoff_bound_m3` for
  `4 L N n_steps + 7` terms (residuals of order 1e-16 m3 against a bound of 8.8e-13 m3);
* positive evaporation, positive UNMET evaporation (the near-residual cell), positive percolation and bottom
  drainage (0.439453125 m3 in the first 6 h step, exact), non-zero diffusive redistribution, EXACTLY zero surface
  return and zero added water, zero flux on the inactive cell and bitwise identity of its state;
* no soil reset between occurrences: the accepted state entering occurrence k is bitwise the state published by
  occurrence k-1 and it advances;
* coordinator-driven result == independently initialized direct-native control for every occurrence: same exact
  chunks (dates and rational seconds, derived by the control with the standard library), same step count, exactly
  equal budgets, bitwise identical final state and storage, the control's own `initialize_profile` closure within its
  bound, and the owning library's `integrate_interevent_pet_chunks` reference depth consistent with the sum of chunk
  rates x durations;
* refusal injections after staged computation (second occurrence of each scenario):
  * a wrong reported end → `realized_end_mismatch`; the stage existed and was discarded; accepted state array
    (SHA-256), column time, coordinator cursor and ledger unchanged, `next_index` unchanged;
  * non-zero potential evaporation on the inactive cell → the native `SubsurfaceStepError` ("potential_evaporation_m_s
    must be 0 on inactive cells") surfaces as `adapter_failure`; nothing was staged; everything unchanged;
* a transpiration-enabled interevent (LEAF process) → the adapter returns a non-complete status, the coordinator
  refuses `event_incomplete`, cursor unchanged, no native work;
* scenario files byte-unchanged after the run; the initial water array unchanged.

## 2. Staging and publication (read this before reusing the helper)

`NativeIntereventTestAdapter` is TEST-LOCAL. Its `run(dispatch)` never writes the accepted state or any input array
(`subsurface.step` is pure and returns new arrays); it holds the advanced state as a stage. The DRIVER calls
`publish()` only after `Coordinator.step()` returned acceptance and `discard()` on any `CoordinatorError`. The
Coordinator has NO native commit hook, no combined transaction and no native checkpoint; its cursor atomicity covers
its own ledger only. Do not advertise this helper as a production-safe coupled adapter.

The adapter declares `continuous_soil_moisture=True`. In this interevent-only scenario that means exactly: consecutive
interevent occurrences share one persistent column state. No wind or rain adapter exists here, and the native rain
handoff (`DryHandoff`, `complete_event.finalize_event`) still resets soil water; the coordinator's
`continuous_soil_moisture_across_events: True` for these scenarios is therefore NOT a statement about rain or wind.

The direct control is an independent test reference (own column, own state, own chunking, own PET records,
`initialize_profile`), not a second simulation inventory.

## 3. Source arrangement — development overlay, not a runtime

`maple_syrup.subsurface` (live SYRUP) and `maple_syrup.weather_pet` (Phase 0B candidate) must import from ONE package.
Two arrangements are supported; both are development identities, neither is a locked production runtime, and
neither bypasses a MASY/SYRUP guard:

1. **Self-built overlay (default).** The driver writes `<work>/overlay/maple_syrup/` with byte-identical copies
   (SHA-256 verified against their origins) of `__init__.py`, `conservation.py`, `subsurface.py`,
   `subsurface_numba.py`, `subsurface_cuda.py` and the import-only closure `routing_cuda.py` / `routing.py` /
   `routing_newton.py` (the CUDA soil backend imports its NVRTC compile options from `routing_cuda`; no routing code is
   executed) from the current SYRUP `src`, plus `weather_pet.py` from the PET candidate `src`. Rebuilt on every run.
2. **Root-prepared overlay (`--overlay-src <dir> [--overlay-identity <manifest.json>]`).** An existing same-package
   root (e.g. the task's `runtime_overlay/src`, 72 current SYRUP modules plus the weather module) is used AS-IS: never
   edited, never rebuilt; every `maple_syrup/*.py` file is hashed and, when the manifest is given, must match its
   `source_files[...].sha256` entry or the driver refuses (exit 3).

The three roots (immutable MAPLE39 `source/src`, MASY candidate `src`, overlay) are prepended to `sys.path` in a
fresh process; after import every module's `__file__` must lie inside its declared root or the driver exits 3.
There is no `sys.modules` replacement, no fake module or class, no installation, and no source lock is created,
mutated or rebound. The driver sets `sys.dont_write_bytecode = True` so no `__pycache__` is written into the
read-only dependency trees or the prepared overlay. The receipt records each origin path and SHA-256.

Because `masy.weather` records the owning library's import origin (file path + hash) in the weather record, the
SCENARIO DIGEST includes the overlay path and therefore differs between work directories and between the two
arrangements, even though the scenario, event and weather FILES are byte-identical across runs (verified). This is
the documented MASY development identity, not a change in forcing.

Default roots come from the task `baseline.json` (`dependencies_read_only`); each can be overridden
(`--masy-src`, `--pet-src`, `--syrup-src`, `--maple-src`).

## 4. Running

Use explicit source roots outside the archived task directory. Set `MAPLE_SOURCE`
to the immutable MAPLE39 source root named in the qualification record; this
test does not install, qualify or silently select a different production runtime.
From the MASY repository:

```bash
/home/okin/MAPLE/.venv/bin/python -I -B tests/native/check_phase0_subsurface.py \
  --backend numba --output /tmp/phase0-column-numba.json \
  --masy-src /home/okin/MASY/src --syrup-src /home/okin/SYRUP/src \
  --pet-src /home/okin/SYRUP/src --maple-src "$MAPLE_SOURCE/src"
```

Use `--backend numpy` for the reference path. For CUDA select an allocated,
currently idle GPU, use MAPLE's `.venv-cupy`, and pass `--backend cuda`. Refuse
a failed availability query or a busy GPU; do not fall back to another device.
The optional `--overlay-src` and `--overlay-identity` select an explicitly
prepared development source overlay. No production source guard is disabled.

Scenario trees (and the self-built overlay) are written under `<output>.work/` (or `--work-dir`); an existing
non-empty work dir is refused unless `--reuse-work-dir` is given; an unwritable work dir is a setup refusal. Exit
codes: 0 passed, 1 checks failed (receipt written), 2 selected backend unavailable (clean refusal receipt; never a
fallback), 3 setup / source-isolation refusal.

Memory: the admitted requirement is the actual `prepare_column` estimate plus an explicit reserve declared BEFORE
`prepare_column` for the live coordinator / scenario / weather / dispatch records and the staged, control and input
arrays (about 25 kB here). The reading uses the real host (`sysconf`) or device (driver + pool) query, nothing faked.
Working sets are O(L N): no per-step arrays are retained, only scalar summaries (device-resident accumulators are
read once per occurrence).

## 5. Author results (CPU, 2026-11-10)

| Backend | Overlay | Interpreter | Result | Wall |
|---|---|---|---|---|
| numpy | self-built | `/home/okin/MAPLE/.venv` (Python 3.12.3) | 67/67 passed | 0.72 s |
| numpy | Root-prepared, 72 files manifest-verified | same | 67/67 passed | 0.72 s |
| numba 0.67.0 / llvmlite 0.49.0 | self-built | same | 67/67 passed | 2.10 s (incl. ~1.4 s compile) |
| numba | Root-prepared | same | 67/67 passed | 2.15 s |
| cuda | self-built | `/home/okin/MAPLE/.venv` (no CuPy) | refused cleanly, exit 2: "requires CuPy, which is not usable" | — |
| cuda | Root-prepared | `/home/okin/MAPLE/.venv-cupy`, `CUDA_VISIBLE_DEVICES=""` | refused cleanly, exit 2: "requires a visible CUDA device (device count 0)" | — |

Final soil states are bitwise identical across all four passing runs (numpy / numba x self-built / Root overlay).
Physical totals over the 2.5-day sequence: storage 3.369903564453125 → 2.8311694407776296 m3 (explicit PET
4.0 / 4.5 / 3.5 / 5.0 mm/day) and → 2.846694737606726 m3 (Hargreaves PET 3.4284 / 3.5643 / 3.0773 / 3.7313 mm/day
at 32.6 deg); drainage 0.439453125 m3 (first step only, exact), evaporation 0.0993 / 0.0838 m3, unmet evaporation
0.00409 / 0.00325 m3, percolation 0.046875 m summed over cells (exact), zero surface return.

The two overlays differ in ONE source byte-range: the live PET candidate `weather_pet.py`
(`abd04bae…`) carries `init=False` on the `_derived` dataclass field, the Root snapshot (`5711d54b…`, the
correction02 report hash) does not. Both produce identical PET values and identical soil states here; Root decides
which pair is the final one.

The actual GPU execution was NOT run by the author (CPU-only authoring; GPU 0 is the user allocation). The CUDA path
is supported by the driver (`require_cuda`, device-side arrays, one packet read per step, device accumulators) and
awaits Root's run on a free GPU 0.

## 6. Not accepted / not claimed

* Persistent rain coupling: the native rain acceptor requires `DryHandoff` with zero soil storage and the storm
  completion zeros soil water. Layered persistent infiltration is Phase 1 work.
* Plant transpiration and growth: no LEAF adapter; transpiration-gated dispatches refuse. The agreed root-uptake
  contract is not exercised.
* Soil remapping after erosion / deposition; evolving geometry (Phase 2).
* Combined native restart: the coordinator cursor is a temporal record only; there is no native soil checkpoint and
  no transaction binding cursor and column.
* Wind / rain integration or mixed-event gate proofs: those remain the MECHANICAL protocol tests of the scheduler
  candidate and are not re-labelled by this driver.
* A production coupled adapter, a source-locked runtime, calibration or observational validation.

## Independent execution qualification

Codex independently passed all 67 checks on NumPy, Numba and physical GPU0.
Final water arrays, times and scalar budgets are bit-for-bit identical across
these three backends for both scenarios. A guarded host availability query
confirmed GPU0 used zero memory and zero utilization immediately before the
accepted repeat. See `shared_scheduler_phase0_qualification.md` for exact
source identities and delivery limits. The old Root-prepared overlay above
is retained historical evidence; the accepted current weather source includes
the humidity-cache fix (`abd04bae…`).
