# Immutable GPU runtime profiles

`tools/masy_gpu_runtime.py` seals an explicit MAPLE, SYRUP and MASY source
selection for the existing `/home/okin/MAPLE/.venv-cupy` interpreter. It records
package origins, native bundle checks, distribution metadata and loaded binary
hashes. GPU profiles have their own store and schema; existing CPU runtime locks
retain their original identities.

Preparation and verification block accelerator imports. A separate identity
probe checks an explicitly selected physical GPU UUID and PCI address through
the native MAPLE backend. It records CUDA, driver, CuPy and native kernel
declarations. Numerical kernels and model transport require later qualification.

Run the launcher with an existing allowed interpreter and isolated imports:

```sh
/home/okin/MAPLE/.venv/bin/python -I -B tools/masy_gpu_runtime.py prepare --config /absolute/path/gpu-config.json
/home/okin/MAPLE/.venv/bin/python -I -B tools/masy_gpu_runtime.py verify --store /absolute/path/gpu-store --profile PROFILE_ID
```

The closed JSON configuration uses schema `masy.gpu_runtime_config.v1` and
requires these fields:

| Field | Meaning |
| --- | --- |
| `sources.maple`, `sources.syrup` | Existing CPU runtime source specifications, with exact committed upstream revisions and source-bound distribution metadata when required. |
| `sources.masy` | `mode`, `root`, `base_repository` and full `base_revision`. Use `committed` for normal committed runs. `development_snapshot` explicitly records complete scoped deltas and the observed repository HEAD. |
| `runtime` | Existing `.venv-cupy` interpreter and a separate absolute GPU store path. |
| `device` | Physical `uuid`, normalized `pci_bus_id` and `visible_ordinal: 0`. |
| `limits` | Existing runtime work, disk, output and worker-time bounds. |
| `metadata_receipt` | `null`, or the exact source-built metadata receipt path and SHA-256. Native source guards remain enforced. |

Source changes require a fresh profile. Verification never upgrades an existing
profile, changes a CPU lock, installs packages or updates the interpreter.
Caches and output belong to the run. Unreadable or changed binary provenance
causes refusal.

For a device identity probe, supply a closed authorization record with schema
`masy.gpu_identity_authorization.v1`, `authorized: true`,
`operation: "identity_probe"`, the accepted `profile_id`, physical `uuid`,
`expires_unix_s`, `max_resource_age_s`, `max_memory_used_mib` and
`max_utilization_percent`. The live gate checks allocation before and after the
worker and refuses a selected device with a compute process, stale evidence or
usage outside the recorded bounds. This is an allocation observation, not a
reservation. The caller must have authorization for the selected device.

```sh
/home/okin/MAPLE/.venv/bin/python -I -B tools/masy_gpu_runtime.py probe --store /absolute/path/gpu-store --profile PROFILE_ID --authorization /absolute/path/identity-authorization.json
```

On October 19, source preparation, verification and a bounded identity probe
passed for MAPLE `39bcf79`, SYRUP `d05d8c6` and an explicitly labelled MASY
development snapshot based on `22aa6ab`. Profile
`25beb28af8569f2802311443f03d2ef79c3e902223eddf61d31748e4242695be`
and probe
`2a50c5595cdccaacc01c30cd2c252f971fb87d3cd4a27a1bbd42e0f400d77d76`
retain that exact selection. The observed device was physical UUID
`GPU-5c8bd3d4-2885-82c6-a0f0-ad2b7f87b871`, PCI `00000000:17:00.0`, a
GeForce GTX 1080 Ti, with CuPy 14.2.0 and CUDA/NVRTC 12.9. The worker exited and
was reaped, with the selected device idle at both allocation gates.

Only `device_identity_observed` and `identity_probe_passed` become true after
this probe. `kernels_compiled`, `runtime_smoke`, `transport_ready` and
`gpu_qualified` remain false. This evidence does not qualify numerical GPU
execution, coupled GPU transport, windowed wind, performance or persisted
combined restart. A profile prepared from a later committed MASY revision has
its own identity and verification evidence.
