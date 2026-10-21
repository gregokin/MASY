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


## Private production device owner

`masy.dense_device_runner._create_dense_production_device_runner` is an explicit
private opt-in factory. The default device factory, default six-class profile,
shared event engine and native wind/water APIs retain their existing behavior.
This owner component is source-only until its separate source, allocation and
protected-worker gates pass. Adding its dormant check does not qualify production
construction, events, restart, calibration or a memory peak.

The first owner profile is exactly one class on a 4 by 4 dense grid with six
voxels, flat cohorts up to 4000 slots, up to eight hop bins and the original
wind/rain/wind/rain schedule. Separation uses the actual native sampling-step,
crest-lookahead and sample-count resolvers. It refuses more than 128 samples or
16 owners before allocating a sampling workspace; it never clips a native value.
Wind remains periodic, water retains its native open-boundary mapping, shared
water depth uses `constant_depth`, and actual native wind and water commit
avalanching remains enabled. The native reduced, FFT topographic wind, separation
and creep algorithms are not copied into the wrapper. The synthetic100um
interface declaration retains production calibration pending.

The new `dense_device_production` module supplies the shared engine's production
record, process checks and bounded commit diagnostics. Three device ownership
hooks require dry accepted water, verify exact water bytes against result and
final native water, and report compact native wind/commit and retained-owner
evidence. A result's native topographic modifier is a diagnostic; finalizer cache
invalidation remains required. A zero signed domain creep total is not evidence
of zero gross creep. Five selected native avalanche class vectors preserve
`None`, values and signed zero through the owning native `to_host` transfer;
these bounded observer lists never become a second sediment authority.

Original host ModelConfig, hop tables, availability policy and shared geometry,
class and commit anchors remain host references. Only the existing device static
leaves are converted. Both device BedContext and the device scheduler use the
same owned availability static. Native hop/reduced kernels may copy their small
host-at-rest tables at point of use; these native transfers must be counted in
later event qualification. No claim of zero host/device transfers is made.

A closed metadata-only host-bed view retains original host BedContext, bounded
primitive case metadata and the initial case identity. It contains no physical
state, voxel store or reference back to the original full CPU DenseBed. The
existing device metadata checks bind that view to the same context and case;
source identity is checked before and after conversion. The full CPU bed is a
construction input that can be released afterward. A unit weak-reference test
uses a clearly authored retention sentinel, while the dormant construction check
requires release of the actual original DenseBed and private device owners.

`DeviceProductionLimits` requires four explicit positive
`DECLARED_UNMEASURED` reserves: FFT workspace, driver/JIT, native transients and
diagnostic scratch. Each is at most 1 GiB and their sum is at most 4 GiB. Existing
bed, private-resource, routing and native water-preflight terms remain separate.
Per-array FFT payloads and declared separation-shape allowances are identified as
such; they do not estimate a whole FFT/JIT/native peak. The read-only library
checks the already-enforced CPU, core, file and descriptor limits, worker
PID/start/boot, current device/stream, host RSS/global MemAvailable, actual pool
cap and prospective device work. Before the first domain copy it rechecks
post-context free memory against the admitted bytes and the 10 GiB floor.
Subsequent checkpoints apply incremental-work/current-free checks without
reapplying the entry free floor. Device-driver virtual mappings are not treated
as the CPU address-space budget. No resource setter is called by the library.

The GPU worker's existing closed environment remains authoritative. It includes
`JAX_PLATFORMS=cpu` and `NUMBA_DISABLE_CUDA=1`; the six CPU-style thread/arena
overrides are absent. Thread and arena counts are therefore **unbounded by that
environment**, and independent whole-worker CPU/RSS/wall/cache/output/device
monitoring is required. The dormant source follows the accepted protected-worker
cadence: unchanged native process check before native imports, then exact
recorded installed-JAX additions `TF_CPP_MIN_LOG_LEVEL=1` and
`TPU_SKIP_MDS_QUERY=1`, with all other environment/process/path drift refused.
This adapter does not change the native profile or worker guard. A separately
reviewed concrete process recipe remains necessary; no memory-fit claim follows.

### Optional construction-refusal diagnostic

At the existing `device_free_floor` refusal, the original guard may attach a
private `_device_memory_refusal` tuple. Its eight plain integers are, in order:
the same observation's driver free/total and pool used/held/limit bytes, the
derived upcoming-wrapper bytes, and the policy minimum-free/maximum-device
bytes. The last three values are declared or derived values, not measurements.
The production factory carries only this immutable tuple through its existing
cleanup and fresh `DenseRunnerError`; it retains no original exception,
traceback, frame, driver, pool or unaccepted owner through this data.

Absent, malformed or foreign data means `diagnostic_unavailable`. It never
means zero free bytes, successful admission or permission to change a guard.
The original error code/detail, 10 GiB entry floor, 8 GiB maximum pool cap,
query/call order, cancellation and success records remain unchanged. Optional
metadata failures preserve the primary refusal and cleanup. CPU scalar/cleanup
tests do not establish genuine device-counter collection. New runtime source
requires a new accepted source object/profile; old locks, profiles and failed
evidence remain immutable. A protected evidence consumer and any actual device
attempt require their own source and execution gates.

The dormant `tests/gpu/test_dense_device_production_cupy.py` refuses ordinary
automatic collection. A reviewed protected parent must provide a new genuine
committed source profile, both distinct grants, exact test/module/fixture bytes,
fresh selected UUID/PCI allocation and complete source/bootstrap/environment and
resource checks. Its four-event construction executes zero steps and requires
all three native transport tripwires to remain zero. Hashes inside its supplied
scope are inherited controller source bindings, not an independently external
root. Later paired production events need separate native comparisons, complete
transfer/memory observations and ROOT execution admission. CPU metadata tests
alone do not satisfy any of those device or transport gates.
