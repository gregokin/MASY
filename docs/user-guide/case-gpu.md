# Run a supplied case with the closed GPU profile

`tools/masy_case_gpu.py` is a separate public command for the labelled synthetic
8×8/K1/V20/W3 imported-case interface. It uses the actual MAPLE compiler, adopts
its complete bed into immutable windowed storage, places that owner on the
selected GPU, and runs wind/rain/wind/rain through the guarded MASY case API.
Wind and water transport remain upstream. The ordinary CPU command and its
schemas are unchanged.

This initial profile requires two complete authored one-second wind blocks,
one class, 20 voxels per column, a three-voxel window and height-prefix packets.
The supplied case uses the existing labelled synthetic calibration. Production
calibration, larger capacity and C64 persisted restart are not qualified by
this interface. A successful process alone is not an allocator theorem.

## Invocation

Use the existing `/home/okin/MAPLE/.venv-cupy/bin/python` interpreter. Prepare
and verify the immutable source/environment profile with the existing
`tools/masy_gpu_runtime.py` command. Device identity probing establishes
identity; it does not authorize transport.

An operator must separately issue the closed, pinned case authority described
below after checking the current committed source pair, actual runtime,
compiled files and available physical device. Actual values are never filled
from an example, a test factory, an older profile or guessed image counts.

```bash
/home/okin/MAPLE/.venv-cupy/bin/python -I -B tools/masy_case_gpu.py run \
  --store /absolute/runtime-store --profile PROFILE_SHA256 \
  --authority /absolute/case-authority.json --authority-size AUTHORITY_BYTES \
  --authority-sha256 AUTHORITY_SHA256 \
  --request /absolute/request.json --run-id unique-case-run
```

The command prints one result JSON. Successful results identify
`store/runs/<run-id>/result.json`, the detailed case report and the Native
physical diagnostic. A run ID is consumed by creating that exclusive
directory; rerunning the same ID refuses. Choose a fresh authorized operation
after a failure. The command does not replay, repair, delete old deep versions
or resume an active event.

`--help` imports no model or accelerator. Execution requires an isolated,
unoptimized interpreter (`-I -B`, without `-O`).

## Authored request

The exact JSON schema is `masy.case_gpu_request.v1`. Duplicate keys,
nonfinite numbers, excessive depth, unsupported integer widths, missing or
extra keys refuse. The complete top-level fields are:

| Field | Required content |
|---|---|
| `schema`, `label`, `calibration` | Schema above, explicit bounded label, and `synthetic_test_fixture` |
| `case` | Absolute directory, actual `authored_source_sha256`, and `copy_mode="authored_source_then_compile"` |
| `input_limits` | Every field of genuine `CaseInputLimits` |
| `compile_bed_limits` | Every field of genuine `DenseBedLimits`, used only for the original compiler disk plan |
| `windowed_storage` | Explicit `datum_offset_m=0.0`, `window_voxels=3`, `chunk_columns` 64…1024, `rows_per_band` 1…8, `with_prefix_packets=true` |
| `windowed_limits` | Every field of genuine `CaseWindowedLimits` |
| `water_profile` | Complete ordinary `CaseWaterSpec` projection and complete explicit sediment/completion/report controls |
| `water_copy_limits` | Every field of genuine `CaseWaterCopyLimits` |
| `native_limits` | Exact profile object and complete nested record policies below |
| `bridge_limits` | All `PositiveWindowedDeviceWaterLimits` fields except `run_dir`, which this operation owns |
| `bed_limits` | Every `DeviceWindowedC64BedLimits` field |
| `resource_limits` | Every `DeviceResourceLimits` field |
| `runner_limits` | Every `DenseRunnerLimits` field |
| `placement_limits` | Every `CaseDeviceC64PlacementLimits` field |
| `io_policy` | Seven explicit IO fields below |
| `schedule` | `masy.case_schedule.v1`, exactly wind/rain/wind/rain, with one original complete wind block per wind event |
| `backend_change` | Exactly `{ "field":"sediment_control.storm.implementation", "from":"array", "to":"cuda" }` |

The case is the ordinary `case.yaml`, `model.yaml`, `wind.yaml` and supplied
NPY elevation source. This first compiler lane supports the genuine simple
NPY DEM path, no user transforms, prescribed raster/absent vegetation and no
initial airborne seed. Unsupported import branches refuse before compilation.
The original Native compiler writer guards and output reservation apply.
The command compiles once, adopts that actual full initial bed, and retains
the compiled inputs, initial HOST owner, working owner and all deep references.
It never reconstructs a transition bed from a DEM or surface fractions.

The `native_limits` root has exactly `profile` and `record_limits`.
`profile` has all six explicit fields of
`DeviceWindowedK1DimensionProfile`: `ny=8`, `nx=8`, `n_voxels=20`,
`window_voxels=3`, `n_classes=1`, `hop_bins=8`.
The decoder constructs the actual upstream classes, with every policy field
explicit: `WindowedWindRecordLimits` →
`DeviceWindowedK1DimensionWindBlockLimits` → step/solve/validation/commit;
step uses `DeviceWindowedK1DimensionStepLimits`, its impact policy uses
`DeviceK1DimensionImpactLimits`, and its creep/solve/validation nonimpact
policies retain `DeviceBedExchangeLimits` with R256. Nested delta budgets are
actual `WindowedDeltaBudgets`. One genuine profile object is shared by the
dimension policies. A request cannot select a class by string, inject a
different profile, omit constructor fields or substitute boolean integers.
All original Native validators, byte ceilings and safety fraction remain.

Water grid inputs are relative, pinned `.npy` descriptors with exactly
`kind`, `path`, `size`, `sha256`, `dtype`, `shape`. Full elevation and export
ring shapes are 10×10; interior arrays are 8×8. Masks are exact boolean;
numeric arrays are exact float64 and C order. Public NumPy header readers
check rank, shape, dtype, object status and exact file extent before mapping.
The genuine GPU process admission and prospective raw/header/map/current,
next and error charges precede those reads. Only owned read-only NPY mappings
are explicitly closed after the working copy; no Native deep-store or GC
credit is taken. Inputs are checked again afterwards.

Supply every ordinary storm/sediment/completion field and every wet-class
coefficient. The CLI does not choose pit storage, terminal storage, deposition,
soil/reset policy, forcing, tolerances or calibration for the author. It
prepares the actual HOST water inputs and derives only
`sediment_control.storm.implementation="cuda"`; the read-only CUDA property
and all other controls retain their genuine owning semantics. Completion
must be Native accepted completion with its explicit dry accounting.

`io_policy` contains `max_path_chars`, `max_water_input_bytes`,
`max_water_input_file_bytes`, `max_report_bytes`, `max_output_bytes`,
`max_input_entries`, `max_request_bytes`, each an explicit positive plain
integer. Their ceilings are respectively 4096, 64 MiB, 64 MiB, 4 MiB, 4 MiB,
4096 and 2 MiB. Choosing smaller values narrows admission; it does not enlarge
the original operation budgets.

## Fresh authority and immutable provenance

The pinned read-only authority uses `masy.case_gpu_authority.v1`, operation
`masy.case_gpu.run` and lane `C64_K1_V20_W3_complete_device_water_v1`.
It binds the actual store/profile file, fresh run ID, new committed launcher,
request file, current committed MASY/MAPLE/SYRUP root/HEAD/tree triples,
physical UUID and PCI identity, and ordered issue/expiry timestamps.
Every top-level authority field is required; no vendor or path wildcard grants
lazy binary loads. The authority also includes:

- The exact original twelve execution-budget fields.
- The actual complete Native image/count operands required by the cold
  bootstrap. Four genuine depth/class orders `(6,1)`, `(6,6)`, `(12,3)`,
  `(20,1)` are initialized before FIRST captures; images/anchors/operators
  are checked again after work. No imported image is fabricated or padded.
- A canonical path/size/SHA roster `compiled_additions` for any actually
  approved extra mapped binaries. All retained audit binaries stay exact;
  new files outside that closed roster refuse. Environment, distribution
  RECORD and actual module-origin checks remain in force.
- `case_budget_delegation`: the exact immutable worker issuer SourceRef,
  Native source identity, current runtime SourceRef, real case producer
  SourceRef, ordered22 Source bindings, actual source-manifest SHA and the
  exclusive `runs/<run-id>/case/case-budget.json` path.

At initial issuance the authority must be current and unexpired. Later
immutable replay checks preserve its bytes, ordered timestamps and joins;
expiration while a previously authorized operation runs does not mutate the
grant or reauthorize a second launch.

The child compiles/adopts/prepares the real supplied inputs under that
preauthorization, computes their actual HOST request binding, then issues
one genuine `DeviceWaterC64CaseBudget`. The temporary metadata-only grant
operand is the actual preauthorization reference; the owning encoder excludes
that self-reference. Canonical budget bytes are published once, flushed,
read-only, then a NEW final budget uses the actual published reference.
Only the final owning budget reaches the guarded case API. A synthetic C64
factory budget is not imported-case authority.

## One bounded operation

One deadline starts at launcher entry and covers sealed-source authentication,
currentness metadata, genuine cold audit, compilation/adoption, device
placement, all four events and final checks. The serial child sequence is
`metadata`, genuine runtime `audit`, `case`, `metadata-after`. It uses the
actual profile/environment/bootstrap protocol under this command's monitor;
it never forges the old runtime launcher or calls its detached launch helper.
Everything belongs below the one operation root, including all caches,
compiler output, deep storage, budget, stage streams, receipts and reports.

Original ceilings remain CPU 480/540 s, whole wall 600 s, active pool 2 GiB,
sampled leader RSS 5 GiB, sampled driver growth 9 GiB, whole owned storage
512 MiB, each file 64 MiB, combined stage/query streams 4 MiB, FD1024,
core0, and Native safety fraction 0.5. Before launch the actual authorized
device must be free with the original 9728 MiB headroom, and HOST available
memory must meet the original 6 GiB precondition. Deadline, storage, stream,
CPU, RSS, driver or actual upstream refusals stop the operation. The monitor
records actual PID/start, pidfd exit, reaping and sampled peaks. TERM/5 s/KILL
cleanup can occur after the deadline and is labelled; it never adds model
time. Storage disappearance retries at most three whole walks and discards
partial totals; symlinks, special files and other walk errors still refuse.
These are bounded observations plus named prospective admission, not hard
aggregate filesystem, allocator or universal vendor-heap proofs.

Initial process/environment checks retain the exact original worker policy.
After Native import, two fixed selected JAX Source files and their actual
module origins are checked. Only their documented logging default and the
conditional observed `TPU_SKIP_MDS_QUERY="1"` assignment are reconciled;
the actual cloud-TPU runtime flag must be exactly false. This records which
environment outcome occurred and makes no claim about a TPU hardware count.
All other environment changes refuse. The checked execution environment is
frozen for subsequent admissions and rechecked after work.

## Inspecting success and failure

The detailed report retains the genuine accepted event receipts, clocks,
class conservation bounds, Native completion/reset accounting, original input
pins, actual HOST projection, final image identities and actual binary/source
checks. All eleven external provider terms, eight primitive units and
constructor accounting stay opaque (`None` in Python, `null` in JSON).
The guarded case API and its original before/after verification preserve the
five prescribed vegetation fields and real Native resource ownership. The
public result does not claim a final array readback of those five fields.

The trusted internal output step verifies the exact idle bed/case before and
after calling genuine Native `save_windowed_state_snapshot` on that accepted
state. Its source identity, resident/current-next-error buffers, Unicode/NPY/ZIP
format upper, IO overlap and original bounded chunk validator are admitted
first. The existing per-chunk validator may gather the bounded full 64×20×1
column block (10,240 bytes), reencode and validate its prefix; that work stays
charged. No additional Dense authoritative bed or deep-store copy is created.
The diagnostic references the retained immutable deep store and contains the
Native physical resident/active/availability/water fields. It is **not** a
complete MASY checkpoint: no private soil/pool/five-vegetation companion or
C64 restart capability is implied.

On failure the command preserves its consumed run root, stage/lifecycle
evidence and any emitted accepted receipts. A failed transport is not
completion and does not silently resume. Partial compiler/store output may
remain; inputs are not restored by deleting or rewriting deep versions.
Root must independently accept source, controls and the bounded physical
qualification before publication claims extend beyond this labelled run.
