# Production GPU configuration component

`masy.dense_wind_profile.DenseProductionGpuWindProfile` is a distinct, exact frozen
declaration with `model_config`, `calibration_status` and `calibration_reference`.
The named `build_dense_production_gpu_wind_config` maps the original native MAPLE
`ModelConfig` through the owning `validate_config` and
`build_scheduler_physics_config`. `audit_dense_production_gpu_wind_config`
independently rebuilds and compares every scheduler field, retaining the complete
model, availability, forcing envelope, calibration declaration and code-owned
factory policy in the audit identity. Native supported-shear derivation remains
mandatory. The caller model is never rewritten to NumPy.

This first component is configuration-only: `transport_ready=False`, with no
environment/source-lock, device allocation, owner construction or execution
authority. The native execution declaration is explicitly `cupy` with
`deterministic_scatter=True`; configuration arrays remain host NumPy arrays at
rest. The actual mapper can use CPU JAX for its small hop-bin calculation.
Nothing here selects or queries a device, creates a bed/pool/runner/graph/cache,
or advances an event. A new committed source pair, CPU qualification, genuine GPU
SOURCE profile and separately admitted ownership/resource/event checks remain
necessary before any future GPU integration.

The existing CPU and synthetic vegetation-wake APIs retain their exact type
selection, NumPy requirement, schemas, audit bytes and refusal behavior. They
refuse the new declaration type. No default or CLI path selects this component.

The supported recipe is an **authored synthetic interface**, with production
calibration pending. It pins the existing native synthetic100um sand calibration,
its coefficients, uncertainty/range and composition checks, retaining the
existing authored CPU fixture's strict `reject` envelope policy. It requires
`detachment_transport_deposition`/`reduced`, native enabled topographic FFT,
flow separation, surface creep and avalanching, flat mobile storage and explicit
`constant_depth`. Wind water transport, wake, prescribed external shear,
dry-commit separation, prescribed sediment/water sources, physical-window
admission, calendar storage/admission overrides and voxel growth refuse. No
coefficient, tolerance, physical rule or caller input is changed to fit admission.

Native stationary and dormancy declarations remain preserved and audited. In
this required noncoupled reduced mode, native stationary continuation is
unreachable and the native dormancy proof explicitly refuses reduced sources;
the default enabled metadata therefore does not enable either shortcut. This
component adds no acceleration policy or ecological response.

Before native validation, mapping or canonical expansion, a closed source-bound
table checks all supported native holder symbols and their complete declared and
present instance fields. Unknown holders, subclasses, missing/extra fields,
cycles, unsupported scalars/arrays and unbounded metadata refuse. Limits are:

| Input | First component bound |
| --- | --- |
| Geometry and grain classes | 1–64 cells, exactly one native class |
| Flat cohorts and hop bins | 1–4000 cohorts, 1–8 bins |
| Wind forcing envelope | 1–8 exact native blocks |
| Each raw tree | depth 16, 4096 nodes, 32 entries per container |
| Text | 4096 UTF-8 bytes per string, 32768 bytes per tree |
| Integers | plain integers with absolute value below 2^63 |
| Availability | exact contiguous host NumPy float64 `(ny,nx,1)`, at most 512 bytes |
| Supplied scheduler audit arrays | same availability object plus two native float64 `(1,bins)` arrays, at most 64 bytes each |

All ModelConfig/forcing arrays are refused in this first path. Availability is
checked for finite values in `[0,1]` and retained by identity, including readonly
inputs; the factory does not copy it or change its flags. A supplied audit
scheduler is admitted before native rebuilding, including native array-field
presence, dtype, shape and contiguous layout. Raw limits are engineering
refusals, not claims of a full native/JAX heap bound or future physical workspace
fit. Configuration allocates only bounded metadata and actual native hop-bin
configuration arrays; hard process limits remain a separate qualification gate.

The GPU audit uses distinct v1 schema/factory literals and recomputes the whole
record SHA. Scheduler-affecting drift refuses an inconsistent supplied result.
Valid metadata-only edits may keep the scheduler equivalent, but change the full
audit identity. No hidden registry stores an earlier declaration or silently
rebinds an old audit.

The focused native check module uses actual config/mapper constructors, authored
small availability, source-exact old CPU/wake parity and meaningful pre-expansion
negative controls. Its qualified worker supplies the old module source through
`MASY130_CPU_BASELINE_PROFILE` and immutable native fixture inputs, forbids
accelerator/device/bed/owner/runner/transport operations, and binds exact origins
and compiled dependencies under fixed CPU-only limits. Source acceptance and
execution results are separate receipts; adding these checks alone establishes
no device or event qualification.
