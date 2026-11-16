# Explicit CUDA triplet profile (`masy.triplet_cuda_runtime`)

Profile `masy.daily_wind_water_growth.cuda.v1` is a separate, opt-in source lock.
It is not a GPU qualification, a transport authority, a restart capability or a
change to the CPU profile `masy.daily_wind_water_growth.cpu.v1`.

## What is locked

* The four selected committed sources (MAPLE, SYRUP, LEAF, MASY) by explicit
  local branch ref, commit, tree and per-file blob rows (Git object reads only;
  dirty worktree bytes are observed, recorded and excluded).
* The genuine materialized MAPLE bundle and its distribution metadata.
* The LEAF external helper `benchmarks/spatial_mapping.py` (namespace helper,
  no package initializer).
* Labelled metadata overlays for SYRUP/LEAF/MASY (not installations).
* Interpreter: exactly `/home/okin/MAPLE/.venv-cupy/bin/python` plus its binary SHA.
* Backend `{"name": "cuda", "provider": "cupy"}` and device
  `{"physical_uuid": "GPU-…", "logical_ordinal": 0}`.
* Native source profile name `native_maple_1dca_case_latitude_cuda_v1`; the
  worker still requires the sealed SYRUP record's own `device_runtime_capability`.

## Schemas

* Qualification: `masy.triplet_cuda_qualification.v1` = CPU fields plus
  `backend` and `device`; `status` must be `qualified` and the file's external
  SHA-256 must be supplied by Root. No hash is hardcoded.
* Lock: `masy.triplet_cuda_lock.v1`, interpretation
  `code_only=True, transport_qualified_by_lock=False, combined_restart=False,
  backend="cuda", immutable_upgrade=False, device_qualified_by_lock=False`.

## Cold order in the worker

1. stdlib-only bootstrap: lock SHA, sealed MASY bytes, fresh pycache, stdlib-first path,
   then the CUDA tail refuses a non-CUDA lock or a `CUDA_VISIBLE_DEVICES` that is not the locked UUID;
2. strict lock verification (objects, bundle, interpreter, current refs);
3. import census refusing any preloaded `cupy`/`cupyx`/`cupy_backends`;
4. `maple_syrup.dependency_bundle.check_runtime(target)`;
5. `native_case_profile.verify_source_profile(target, expected_syrup_digest, profile=<CUDA profile>)`
   then `recheck()`, then `device_runtime_capability is True`;
6. entry point runs with a `CudaRuntimeContext`; CuPy may load only now and only
   from the interpreter's own site-packages;
7. `runtime.recheck()`: no Git, sealed files, owning source recheck, device env,
   provider origins, module identities unchanged.

Root owns launch, time, memory, GPU permission and the physical device.

The provider census authenticates loaded Python modules by content and compiled
modules by origin/size. The run supervisor must additionally record the actual
compiled environment. Root's qualification inventories hashes of loaded shared
libraries and the run-owned CUDA kernel cache, plus CuPy/NumPy/Python and CUDA
runtime/driver versions. A source lock is separate from that run evidence.
