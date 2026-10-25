# Private Dense device runner (task 79/81 prototype)

The latest bounded qualification, for its exact source selection only, is recorded in the [October 25 Dense GPU milestone](dense_gpu_qualification_20261025.md): four labelled synthetic GPU events completed with exact agreement for the captured boundary records and clocks against a separate native GPU control. CPU–GPU field comparison, windowed coupling, persisted restart, scale and production calibration remain open. The sections below describe the earlier task-79/81 prototype and its historical qualification.

Historical task-81 status: **private prototype, qualified for CPU semantics only.** The task-67 device hook seam is merged into the
coherent engine. The task-81 runner uses the SAME shared engine, with typed bed/pool/soil/static owners, an exact
native backend parameter bridge and the per-owner routing stamp described below.

The task-81 private CPU qualification ran 68 unique NumPy semantic and native-constructor methods. It produced 107
passed report rows, including subtests, with no failures or skips, no native transport attempt and no accelerator import. Root accepted it as
`9a10c51a`.

There is no actual CUDA construction, native GPU transport, full GPU event or memory-fit acceptance. Other GPU
evidence is separate. Only the existing genuine task-58 tiny GPU bed copy passed (7/7). The task-40 cold/warm native
primitive CPU/GPU smoke tests were accepted independently, but they are not coupled GPU transport. The task-83
construction harness has source and CPU-metadata acceptance (21 protocol methods) using labelled syntactic stubs;
it has constructed no native or GPU fixture and queried no hardware, so its real profile, allocation and binding
remain absent. All six readiness flags are false, and the profile is a genuine source profile only.

## Scope

- `masy.dense_device_runner._create_dense_device_runner` is a private, same-engine, bounded **new-run** factory
  for the default six-class Dense profile. It is not exported, not used by the CLI, and has no profile or
  readiness flag. Importing `masy.dense_runner` does not import this module or CuPy.
- It reuses the existing `DenseRunner` engine (same step/wind/rain/finish code) through private hooks; no native
  wind/water formulas are copied. Strict default Dense physics, native `upwind` transport and the native `cuda`
  hydraulic implementation are required; every other enabled feature is refused.
- The native context/availability and the `ColumnParameters` / `SedimentPhysicsParameters` namespace bridge
  switch only the native `xp` namespace and place exactly the fields the native builders place on the device;
  nothing is recomputed.
- A bed carrying evolving vegetation (`_dynamic`) is refused as `unsupported_profile` before the native namespace,
  CuPy or any copy is touched. CPU evolving vegetation remains CPU-only; there is no device vegetation growth.

## Retained routing

The engine calls the same three closed retained-routing hooks as on the CPU (`routing_memory`, `routing_guard`,
`routing_entry`). `_DeviceOwnership` implements them over its own narrow held namespace, with no native template:

- `retain_routing` keeps the native graph, terrain, network, grid and routing source by reference, a private zero
  float64 sediment velocity, phase `None`, an owner routing generation and the adapter's private `stamp`.
- **Owner-bound.** Each `_DeviceOwnership` creates one opaque `object()` stamp in `__init__` and puts it in every
  namespace it issues; `_held_routing` requires the held stamp to be that object (`is`). Generation, layout and
  terrain alone cannot tell two adapters apart: a second adapter at the same issued generation over the same
  terrain reference would otherwise accept the first adapter's routing. The stamp is a bare object, so the
  namespace holds no reference to the adapter, its owners or the native bed. It exists only in this process and
  is never serialized (there is no persisted GPU restart).
- A different holder type or field layout, a missing or foreign stamp, a generation outside this adapter's issued
  range, or a different terrain reference is refused as `restored_routing_layout_changed` with the
  external-change guarantee before any work copy or native entry. CPU templates and restored CPU projections stay
  CPU-only. Generation labels describe the namespace lifetime and are not portable ownership tokens.
- **Why an older generation stays valid.** The generation check is `0 < generation <= issued`, not "equal to the
  latest". The shared engine prepares the bed publication and calls `retain_routing` (incrementing the issue
  counter) before `_finish` runs its remaining fallible checks and publishes. If an ordinary failure then abandons
  the event, `abandon` discards the stage and prepared tickets but does not roll the counter back. The routing
  the engine still holds from the last accepted event is therefore older than the latest issued one, and the
  retry from the accepted boundary must be able to use it. A single latest-issued pointer would break that retry;
  no history of issued namespaces is kept. The trade-off: an issued but never-accepted namespace of the same
  adapter also passes this guard. The engine only hands back the routing it accepted, so the guard does not
  separate accepted from issued.
- `routing_memory` requires the held velocity to be a `(ny, nx, nc)` float64 array of the owner namespace
  (`transport_memory_unsupported` otherwise). `routing_guard` requires phase `None` and zero velocity
  (`transport_memory_not_empty`).
- `routing_entry` builds the exact nine-field native `SedimentEventState` (a native field change is refused). On
  later rains the velocity is an independent work copy, so native writes cannot reach the held namespace.

## Ownership

- `DeviceDenseBedOwner` holds the six bed roots. It is distinct from `DeviceDenseResources`, which owns the
  evolving pool/soil and the static holders (vegetation, threshold modifier state, availability, column,
  rainfall field, sediment).
- Construction checks: raw host->device boundary copy comparison including signed zero, compact native checks,
  atomic pair publication (all fallible checks before assignment-only publication), and typed tickets.
- Each rain uses a fresh native `CudaHydraulics`; references are dropped afterwards (the native class has no
  `close` API). Failure metadata records what actually happened; no success is synthesized.
- Before any source hash or copy, `_water_parameter_sources` closes the exact native parameter holders and
  `_closed_host_sources` runs the resource owner's bounded `_closed_metadata` walk over the actual CPU pool and
  static sources, requiring exact coverage of the native `array_leaves` the reservations count. Unknown extra
  holder attributes, payloads, dtypes and layouts are refused; legitimate native aliases are kept. This is an
  engineering guard, not a physics or value policy.

## Workspace admission (partial)

The construction admission is the native CUDA water preflight plus owner reservations, construction terms,
**named partial lower reservations** and **declared unmeasured allowances**. It is NOT a complete full-step
gate and NOT a peak fit:

- MAPLE's `flat_pool_unbudgeted_bytes` is a partial estimate with no preflight gate.
- Flat wind row terms use `max_active_cohorts` rows as a lower bound; flat transient rows can exceed stored
  cohorts, so pool capacity does not bound the row count.
- `scatter_owner_bytes` is called with 2 index components, a minimum; the class-indexed face/deposition
  scatter ranks have not been audited against the native ops.
- A full source-derived peak/transport admission is future qualification work.

## Not covered

CPU-GPU event comparison has **not been run**. Its policy is settled: each model's existing native CPU/GPU
comparison bounds, never relaxed; copies, handoffs and discrete decisions exact where required; native
conservation unchanged. `device_evidence()` keeps `gpu_transport_qualified`, `memory_fit_qualified` and
`restart_supported` False. The workspace admission stays partial, the declared allowances stay unmeasured, and
qualification still needs the full protected GPU worker. Production GPU, windowed, dynamic vegetation, persisted
restart and scale qualification remain separate work.

## Tests

- `tests/test_dense_device_runner.py`: CPU semantic/guard tests through the NumPy owner seam. The retained-routing
  tests call the actual native first-rain constructors (graph, `initial_event_state`, entry `commit_bed`,
  `initial_state`), but no native wind step, finalizer, rain event or CUDA model. They include a second real
  adapter at the same generation on the same terrain (refused by stamp), missing or foreign stamps, and an
  accepted older routing that stays usable after a later prepared and abandoned event (and stays refused by
  another adapter).
- `tests/gpu/test_dense_device_runner_cupy.py`: dormant by default. A genuine construction test needs a fresh
  sealed source pair/profile, a selected device UUID, fresh cross grants and protected resource caps from the
  parent authority. It covers first non-transport construction only.

The CPU file passed in the task-81 qualification above. The CuPy file has not performed an actual CUDA
construction. Existing default CPU contracts passed the task-80 source-prototype broad run, and the coherent
candidate's targeted store/routing/schema checks and native controls passed in task 87.
