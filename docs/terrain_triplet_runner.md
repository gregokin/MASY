# Irregular terrain triplet runner

The separately selected `masy.daily_wind_water_growth.terrain_cuda.v1` profile connects actual MAPLE, SYRUP and LEAF on a Dense CUDA bed with a fixed base and irregular physical footprint. MAPLE owns the frozen regional plane, local FFT completion/padding, wind, creep, avalanching and sediment. SYRUP owns every water update. LEAF supplies daily demand/allocation/feedback and weekly growth. Wind assumes dry soil, including over wet/ponded cells, while preserving real water. Moisture suppression remains a known approximation.

This bounded runner accepts one200um class, <=16384cells, an explicit matrix-only Jornada soil configuration and same-year daily scenarios. The scientific parameters and forcing are illustrative; annual vegetation orchestration, combined restart, windowed/large capacity and scientific validation are outside this qualification. Native solver substeps retain their own clocks; weather and plant-demand cadence is daily. Extra wind/rain/recession duration is added to realized elapsed time without daily E/T/growth catch-up.

`tools/prepare_moab_hybrid.py` uses explicit `--maple-case`, `--leaf-case`, `--syrup-root` and a new `--output`. It writes an exact128x128crop of MAPLE's Moab Stream3 DEM/base, its real exterior ring, synthetic10%grass+10%shrubs and8days of forcing. The approved root curves are remapped and normalized over0–10,10–20,20–30cm. Trial seed/patch initialization settings and Jornada defaults are copied from the selected inputs, with hashes. No transport runs during preparation.

Runtime preparation requires an independently accepted four-source qualification and genuine MAPLE bundle. Code qualification is distinct from physical execution. Existing profiles and locks keep their own sources and names.

```
/home/okin/MAPLE/.venv/bin/python -I -B tools/masy_triplet.py prepare --runtime terrain-cuda --qualification QUALIFICATION --qualification-sha256 SHA --destination CODE
/home/okin/MAPLE/.venv/bin/python -I -B tools/masy_triplet.py verify --runtime terrain-cuda --lock LOCK --lock-sha256 SHA
/home/okin/MAPLE/.venv/bin/python -I -B tools/masy_triplet.py run-terrain --lock LOCK --lock-sha256 SHA --inputs INPUTS --inputs-sha256 INPUT_MANIFEST_SHA --output NEW_OUTPUT --run-dir RUN_DIR
```

The caller supplies an existing private run directory, its home/tmp/cache subdirectories and finite process/RSS/device budgets. The CLI replaces itself with the genuine cold worker so one owned PID/process group can be supervised. Its device UUID comes from the explicit lock; it never chooses another device automatically. All simulation payloads and raw logs remain local.

The input manifest and every referenced input are checked, then copied into a verified run-owned snapshot. Wind occurs before daily rain. Rain at exactly5mm takes SYRUP's reduced attempt; above-threshold rain uses native runoff and genuine recession. A failed reduced attempt can request the full storm without applying rain twice. After a storm, daily PET uses the actual current world terrain. Both plants and soil use that same field.

Outputs include accepted-day receipts, generation details, native routing/water/plant ledgers, exact source/device identities and a bounded actual compiled-library/cache census. `physical_result.json` survives a later verification failure. Diagnostic arrays are explicitly not restart files. No completed combined checkpoint is implied by successful day receipts.
