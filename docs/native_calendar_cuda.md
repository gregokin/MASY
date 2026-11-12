# Scheduled native layered water on the qualified 109-module source — CPU109 and CUDA109 (water-only)

Status: DEVELOPMENT candidate. CPU109 has real fresh-process CLI evidence; every CUDA path is WRITTEN, NOT RUN (no hardware by the author) and unqualified until Root's hardware matrix.
See the task `Design.md` for the interface reading behind each seam.

## What was added (production: 5 files changed, 1 new; all other 62 `.py` files byte-exact incl. `native_source_gate.py` and `native_water_run.py`)
* `native_water_backend.py`: second sealed module-literal policy `CUDA_V1` next to `CPU_V1` (same private class, exact-object selection; names/mappings/copies refused); closed `PROFILE_CAPABILITY`
  table; `policy_for_profile`. New seams: `prep_type/prep_profile_name/prep_errors/plan_prep_bytes`, `device_available_fn`, `pet_array`, `estimate_save/estimate_load`, `codec_kw`,
  `runtime_matches_early/late`, `indeterminate_error`; `runtime_record(lop, receipt)`. CPU methods reproduce the old inline statements. `__all__`, `cuda_capability()` (the S1a reserved
  refusal) and the CPU surface are unchanged.
* `native_water_cuda.py` (NEW, imports no CuPy/SYRUP at import): `CudaPlacement` (explicit device id, caller device reserve, declared device cap, native reserve; no defaults), pure CUDA
  preparation plans, the two device packets (one `(N,)` PET packet per date fragment; the storm rainfall field built with a device mask), the budget ADAPTER over the actual planner budgets,
  sizes-only schedule metadata for `cuda_entry_plan`.
* `native_water_checkpoint.py`: `CudaNative9fV2Codec` registered in a SEPARATE closed registry `CUDA_CODECS` (`cuda_codec_for`; the CPU `PHYSICAL_CODECS`/`codec_for` contract is untouched) —
  id `cuda_native9f_dense_lifecycle_v2`, profile `native9f_dense_lifecycle_cuda_v2`, members `{lifecycle_checkpoint_cuda_v2.json, maple_state.npz, lifecycle_state.npz}` (equal to the actual V2
  policy's `allowed_names`); it wraps the accepted V2 save/load engines and the accepted budget facade. the public `publish(path, *, build_physical, temporal_bytes, make_manifest)` and `open_artifact` keep their EXACT historical signatures. ONE private engine `_publish_engine(backend, …)` (the held-descriptor / rollback / hardlink / fsync body, not cloned or rewritten) is selected by the EXACT sealed policy object (`CPU_V1`/`CUDA_V1`; None, names, copies and imposters are refused before any path, descriptor or callback) and derives the closed member names from it; the explicit siblings are `publish_cuda` and `open_artifact_cuda`. No member name is ever a public argument.
* `native_water_calendar.py`: two literal profiles `native9f_layered_factory_afca_{cpu,cuda}_v1` (same digest AFCA); profile↔policy join in `_source_facts`; policy-bound `_source_facts/_validate_source/
  _storm_parts/_namespace_census/_preparation_plan_bytes` everywhere (the rain run, the resume loader and `_fresh_census` previously called the CPU-bound public names); `create_cuda`, `resume_cuda`,
  `plan_resume_cuda` around the shared private engines (CPU `create/resume/plan_resume` unchanged and CPU-only); CUDA create/step/checkpoint/resume admission with SEPARATE host and device figures
  (the lifecycle's own `cuda_entry_plan`, i.e. the CURRENT phase only — later stages of a call re-admit themselves freshly at run time; the actual CUDA facade/codec figures are retained as returned and every outside reserve (host reserve, extra reserve, device reserve) is counted exactly ONCE, never added to a reading; host reader `ss.default_host_available_bytes`, device reader `column.available_bytes_fn`; never summed, no synthetic free value, outside reserves once).
* `native_water_run_config.py`: optional top-level `cuda` block (REQUIRED for the CUDA profile, REFUSED for CPU profiles); the capability table lives here because the parent closure is exactly
  five masy modules. CPU records/digests are unchanged (no `cuda` key).
* `native_water_worker.py`: capability-selected branches: CUDA loader (`parse_layered_cuda_config`) vs CPU loader (the water file's `compute.backend` stays the only backend choice),
  `verify_source_profile(expected_syrup_digest=AFCA, profile=CUDA)` → `admit_package_case` → pure `plan_layered_cuda_setup_from_admission` (+ declared cap) → `prepare_native_dense_cuda_case` →
  `prepare_layered_cuda_case` → `create_cuda`/`resume_cuda`; closed CUDA member names for the resume artifact; cold products released as in the CPU path.

## Contracts kept
CPU public signatures/defaults/records/artifacts, 308/95ef identities, the empty `production_committed` registry, source/census/namespace/manifest guards, the held-descriptor publisher semantics, one water
authority, no create(non-initial)/replay/restamp/dry-reset, completed storm retained as a completed storm until its true STORM_END, wind refusal, no E/T during rain/recession.
## Not done / limits
No CUDA execution by the author; budget exactness of the CUDA admission against the real engines, CPU-vs-GPU physical equivalence, device-counter behaviour, the native39 refusal (separate actual run) and
all device negatives are authored in `tests/native_calendar_cuda/test_gpu_cli.py` (and the accepted facade tests). No LEAF, no wind, no mid-occurrence resume, no whole-process memory bound, no production recipe.
The CUDA preadmission of a rain step is a lower-level check from `begin_storm`/`evolve_storm` figures; each lifecycle entry re-admits itself at run time (typed refusal, parent untouched).

## Transfers
The per-fragment PET upload is GRID-sized (`(N,)` float64, 8N bytes, once per dated fragment — it scales with the grid; never per hydraulic step or per voxel).

## C2 corrections (metadata-only resume planning, staged device admission)
* `plan_resume_cuda` and the resume whole-call plan are METADATA ONLY: no rainfall field, `StormSpec`, device array, upload or hash exists at that point. The completed-storm load budget comes from
  `native_water_cuda.estimate_load_metadata`, which runs the accepted facade's fences in its order and then the SAME `_gather_v2` + `estimate_load_budget_cuda_v2` (the storm terms depend only on
  the schedule's three array SIZES and the lifecycle's own `cuda_entry_plan`; a sizes-only schedule record carries them). The real `StormSpec` is built ONLY after the strict host/device admission
  and the temporal/source/identity joins, and is validated normally by the actual V2 loader (no loader fence is bypassed). Equality with the facade/loader is a GPU test (`test_gpu_planning.py`).
* Staged DEVICE admission on resume (each with a fresh strict reading, the cap and `max_device_bytes`): whole-call largest stage → source recheck (reference prep plan + reserve) → rainfall-field
  build peak **17N** (bool mask N + base 8N + result 8N, read from the actual `rainfall_field`) + reserve → codec load stage-prefix peak **beside the borrowed persistent field 8N** (the codec figure
  already holds the reserve once). The host plan is separate. Rain steps charge the same field-build peak and the borrowed field beside `begin_storm`; the GRID-sized PET upload (8N) is charged beside
  `evolve_interevent`. CUDA memcpy staging/allocator rounding/pool growth are outside the figures.
* `nck.plan_load_cuda` (new; CPU `plan_load` byte-identical, pinned by a test) is the HOST-only normalization: the CPU plan's `common_bytes`/`audit_scratch_bytes` are lifecycle resident/scratch figures that on CUDA are DEVICE quantities and are not host terms; the codec host figure (`host_required_with_reserves`) already holds the codec's host peak, the wrapper's retained host bytes and the outside reserves/`extra` exactly once; the CUDA figures are incremental (no resident-owner top-up); the codec's own `restore_audit` stage covers the restored-state audit prefix and the cold coexistence of the reference and new beds is the codec's device stage prefixes beside the borrowed reference bed; the calendar's later `audit_lifecycle` is a lifecycle entry that re-admits itself.
* CUDA `create`/`step` host readings use a STRICT reader (non-Boolean integer >= 0, no coercion, typed `invalid_availability`); the CPU path keeps its historical `int(reader())`.
* Scope honesty: `required_call_bytes` / `_requirements_cuda` is CURRENT-ENTRY pre-admission (its record says so); every later lifecycle entry freshly re-admits host and device.
