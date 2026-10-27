# A MAPLE case through ordered wind and rain events

This CPU workflow compiles a run-owned copy of a real MAPLE case package and
adopts its complete Native initial bed. MAPLE transports wind and SYRUP
transports water on that evolving bed. It does not reconstruct the landscape
from the DEM at event transitions. This is a new case policy; the historical
Small and Dense qualification profiles are unchanged.

The [October 26 qualification](../case_cpu_qualification.md) exercised the real
cold CLI on one labelled synthetic imported 9 × 8, two-class case with 20 voxels per column. Its
reference values are interface-test inputs, not defaults for another landscape.
No user DEM has been selected or run by that qualification.

## Prepare your inputs

1. Author the normal MAPLE package: `case.yaml`, `model.yaml`, `wind.yaml`,
   and the original DEM under `source/`. The first qualified reader accepts a
   C-order float64 NPY DEM. Keep MAPLE's explicit `import.elevation` units,
   row order, masks, classes, layers and physical choices; see
   [complete case adoption](maple-case-adoption.md).
2. Supply the complete hydraulic ring/outlet masks, soil/wet-class parameters,
   water controls, rain intervals and recession guard. The
   [water-input guide](case-water-inputs.md) lists the required fields and units.
   A DEM alone does not provide these scientific inputs; the synthetic fixture
   is not a calibration or a source of user-case defaults.
3. Write explicit input/copy/runner/process quotas and the closed `run.json`
   described below. Use the digest returned by `inspect`. Wind ranges refer to
   the authored MAPLE record and must partition it in order; they do not settle
   each forcing block separately.
4. Configure `runtime.toml` from the [runtime example](../../examples/runtime.example.toml) for the reviewed
   Source checkouts and existing CPU interpreter. The accepted reference used
   feature sources listed in the qualification record; unrelated current main
   branches are not that reference. A new run selects current committed heads
   and must pass their compatibility/currentness checks.
5. Obtain a five-owner `CPU_CODE.json` issued for that exact new lock, Source
   selection and interpreter. `prepare` does not mint this machine qualification
   contract. Its identities must be authenticated by the runtime qualification
   owner; copying another machine's file or changing its HEAD label is invalid.
6. Select one actually allowed CPU and apply singleton affinity before launching
   the cold CLI. Keep `LC_ALL=C` before interpreter exec so the real startup
   flags match the locked profile. Changing the environment after startup does
   not repair an incompatible interpreter flag.

Use the actual scripts from the reviewed MASY checkout with an existing MAPLE
CPU interpreter. Commands below show the qualified host's interpreter spelling;
substitute your own qualified existing MAPLE interpreter and issue its own lock
and machine contract. There is no installed `masy` console command:

```sh
LC_ALL=C /home/okin/MAPLE/.venv/bin/python -I -B tools/masy_case.py prepare --config runtime.toml
LC_ALL=C /home/okin/MAPLE/.venv/bin/python -I -B tools/masy_case.py verify --store /absolute/runtime-store --lock LOCK_ID
LC_ALL=C /home/okin/MAPLE/.venv/bin/python -I -B tools/masy_case.py inspect --case /absolute/maple-case --input-limits input_limits.json
```

`prepare` and `verify` execute the unchanged `tools/masy_runtime.py` launcher.
They keep its genuine current committed Source selection, immutable bundles,
metadata, interpreter, origin and compiled-library checks. A runtime lock alone
is not a physical qualification. Commit the reviewed MASY case Sources and the
real new script before preparing a new lock. Never relabel an older lock.

`inspect` is bounded and read-only. Its `authored_source_sha256` hashes the
canonical complete inventory of `case.yaml`, `model.yaml`, `wind.yaml` and all
ordinary files beneath `source/`. The same inventory and hash are required by
`run`. Symlinks, changing files and inputs outside the Native package rules
refuse. Native field imports must join that inventory. The original case is
never compiled or overwritten: only those authored/source bytes are copied to
`<store>/runs/NAME/input_case`, then the genuine Native package compiler runs
there once. Its original/current provenance and complete initial artifacts
are retained. The known current Native load-only config drift is not bypassed.

The first file-reader qualification is a full C-order float64 `.npy` DEM on the
case's own `(ny,nx)` grid. Author it through MAPLE's real `import.elevation`
recipe, including its required row order, value units/kind and any nodata
policy. Native owns orientation, declared unit conversion, terrain validation,
stratigraphy and all physical initialization. The current sizing lane accepts
no explicit additional transform and no other imported family; it refuses
those branches before compilation with a specific reason. The original bytes
and Native unit/orientation log remain in provenance. Object-generated
vegetation, nonempty initial airborne seeds, non-NumPy/calendar/coupled modes
and nonNone reconciliation baselines need separate qualification. Prescribed
none/raster vegetation is retained completely. Unknown eligibility is refused,
not converted into another scientific profile.

A run request is closed JSON, schema `masy.case_cpu_run.v1`, with exactly:

- `case`: `directory`, the inspected `authored_source_sha256`, and
  `copy_mode="authored_source_then_compile"`.
- `schedule`: schema `masy.case_schedule.v1`, plus an ordered `events` list.
  A wind event has `kind`, `first_block`, `stop_block`; its positive half-open
  range must partition the entire original MAPLE wind record in order.
  A rain event has `kind`, `water_profile_id`, explicit `intervals` containing
  `duration_s` and `intensity_mm_h`, and positive `recession_guard_s`.
- `water_profile`: complete explicit `CaseWaterSpec` under `spec`, plus
  complete Native sediment/storm/completion controls and reporting cadence
  under `control`. No scientific value is inferred from a synthetic test.
- `input_limits`: all six `CaseInputLimits` fields used by `inspect`.
- `bed_limits`: all three existing `DenseBedLimits` fields.
- `water_copy_limits`: all four `CaseWaterCopyLimits` fields.
- `runner_limits`: all six existing `DenseRunnerLimits` fields.
- `process_policy`: all twelve `CaseCPUProcessPolicy` fields listed below.

`water_profile.spec` uses every public `CaseWaterSpec` field. Each array value
is an exact descriptor `{kind:"npy",path,size,sha256,dtype,shape}`; paths are
canonical relative paths from the request directory, dtype is `float64` or
`bool` as required, and shape is the complete declared two-dimensional extent.
The elevation/export arrays include the full hydraulic boundary ring. The
other arrays use the bed grid. The three optional array fields preserve their
explicit null semantics. Exactly one initial-theta/retained-soil input is
required by the water factory. Header version, rank/order/dtype, exact payload
extent, pin and total byte/copy quotas are checked before mapping or copying.
No pickle, NPZ guessing, invented outlet ring or DEM-to-soil mapping occurs.

The public water factory requires authored class-specific coefficients,
compatible class ordering/common density, square cells, active-mask policy,
soil parameters, vegetation interpretation and boundary ring. Its complete
validation remains authoritative. Missing or incompatible science inputs
refuse; they are never filled with Plot1/legacy calibration. The separately
recorded effective wind config changes only the previously agreed
water-disabled/`constant_depth` interface. All authored boundary, avalanche,
separation, timestep, tolerance and detachment choices remain unchanged;
unsupported choices are reported explicitly.

Forcing blocks are not wind events. All blocks in one declared range run on
one staged bed and one real Native accounting accumulator. The finalizer runs
once at the deliberate group end, then checks genuine empty airborne inventory.
An intermediate block failure retains actual partial evidence and abandons the
stage; no earlier block in that event is published. The next event starts at
the actual preceding Native endpoint, including rain recession. Native clocks
are never snapped or retimed. The original zero-local Native public result
validator cannot validate absolute post-rain blocks; the lane retains genuine
Native entry/state/accounting/conservation checks, explicit actual absolute
forcing/advanced-interval/cursor checks, original MASY budgets and finalizer
checks. It does not fabricate an invalid absolute forcing record for validation.

The process policy fields are `address_space_bytes` (exactly 4294967296),
`cpu_soft_s`, `cpu_hard_s`, `wall_s`, `max_output_bytes`, `max_run_bytes`,
`max_request_bytes`, `max_input_entries`, `max_path_chars`,
`max_water_input_file_bytes`, `max_water_input_bytes`, `max_report_bytes`.
All quotas are explicit positive engineering inputs. Request/outer bootstrap is
at most the original 1MiB ceiling; output and wall cannot exceed the selected
original runtime's own ceilings. The dedicated cold worker sets AS, CPU,
`FSIZE=max_run_bytes` and core0 before any Native/provider import. An inherited
hard ceiling that cannot meet the declared policy refuses before limits change.
`FSIZE` is a per-file limit derived from the declared total quota: it is not an
aggregate filesystem limiter. Source-derived Native serialization/copy/temp/
normalization/publication/error overlap is reserved against the remaining run
quota and free disk before compilation. Phase censuses catch additional actual
bytes and entries, but are not advertised as continuous aggregate enforcement.
Native/provider workspace peaks remain protected by the whole-worker OS limit;
known wrapper copy/work estimates are not a complete heap theorem.

The first qualification recipe requires explicit caller-selected singleton CPU
affinity (choose one CPU from the caller's actual allowed mask); the worker
records `os.sched_getaffinity(0)`. This avoids pretending that a many-CPU JAX
code-generation startup is qualified under the small whole-worker limit. No
thread/JAX environment override or automatic affinity selection is added.
After selecting the CPU, the command is:

```sh
LC_ALL=C taskset -c SELECTED_ALLOWED_CPU /home/okin/MAPLE/.venv/bin/python -I -B tools/masy_case.py run \
  --store /absolute/runtime-store --lock LOCK_ID --request /absolute/run.json --cpu-extensions /absolute/CPU_CODE.json --run-id case-example
```

`--cpu-extensions` is a separate machine qualification file, not a scientific
case option. The runtime owner issues `masy.case_cpu_extensions.v1` after
joining the exact selected lock, interpreter, Source triple and package paths/
digests to the same-environment code identities. It contains exactly five
possible additional owners: resource, readline, mmap, NumPy pocketfft and JAX MLIR
HLO. The worker need not load every optional owner: readline was observed in
the input-test harness, not established as a CLI prerequisite. Every original
compiled file identity remains mandatory; unknown additions and changed bytes
refuse. All other original stable fields remain exact. Full actual rows and
aliases are retained unchanged; alias/module-load-set equality is not used.
The first run needs a file issued by the runtime qualification owner for its
new lock, not an old receipt with a relabelled HEAD. The owner supplies the
closed schema with the contract. Its fields are `schema`, `lock_id`,
`source_heads`, `packages`, `interpreter_identity`, `extensions` and
`qualification_origin`; every extension row has `owner`, `path`, `size` and
`sha256`. The exact owner IDs are `jaxlib_mlir_hlo`, `numpy_fft`,
`stdlib_readline`, `stdlib_resource` and `stdlib_mmap`. The
[owning parser](../../src/masy/case_cli.py) enforces that schema.

The parent imports no Native/provider package. It authenticates the readonly
MASY Source tree before importing the sealed parent and pins this new real
script separately against its current committed Git blob. The original bounded
process engine drains actual stdout/stderr, enforces wall/output limits and
kills/reaps the one process group. Failures keep bounded raw streams and
truthful stages. The original runtime environment is retained; only the exact
Source-authenticated two JAX import bookkeeping additions are accepted, with
full original/delta/actual provenance. Unknown or changed environment keys
refuse; no value is set, hidden or restored by that guard.

`case_report.json` records complete resolved event grouping and original block
maps, actual clocks, input/pair/policy identities, effective config mapping,
Native class/water accounting, the retained routing behavior and disk/copy
admissions. The report is bounded separately and written only after final
identity/currentness checks. A stopped event is incomplete and has nonzero
worker/parent status. Native initial compiler artifacts remain initial-state
snapshots. `analysis_landscape.npz` is a genuine Native snapshot of the final
accepted voxel/active/availability/threshold/prescribed vegetation/water arrays
and actual time. It is separately reserved before compilation and again before
writing, content-hashed, published without overwrite and made readonly. The
report holds the accepted bed identity, artifact pin and actual omitted-field
roster. Native diagnostic flags stay `validated=False,restartable=False`; the
report separately records the genuine shared validation performed before that
writer. No extra topographic commit or initial-bed reload occurs. The archive
omits the ledger, committed/reconciliation and MASY/private continuation state,
so it is explicitly an analysis artifact, not a Native or combined restart. No GPU, large/windowed,
calibration, general provider memory or future API compatibility claim follows
from this bounded CPU workflow.

## Explicit mapped-NPY code identity

Read-only water NPY descriptors use the existing NumPy memory-map path. NumPy
imports the standard-library `mmap` extension lazily in its actual memmap
constructor. The machine contract therefore includes the exact fifth owner
`stdlib_mmap`, selected only from `DESTSHARED` with a genuine platform
`EXTENSION_SUFFIXES` spelling. This does not change mapping, copying, closing,
transport, Native physics or the process policy. All original locked compiled
identities remain mandatory; full locked-plus-qualified count/byte limits and
unknown/changed-file refusal remain. Preparation checks its actual loaded
subset against the same complete authority without inventing loaded modules.

The owner declaration is a Source-level compatibility change. A new committed
Source/lock and a genuinely issued five-owner code contract are required for a
run. The earlier generic unknown-file refusal still does not identify its
unnamed offending file. The executed result is recorded separately in the
[qualification record](../case_cpu_qualification.md); it does not pin identities
for other machines or locks. On future product-worker compiled-identity refusal,
the existing error/stderr includes bounded expected/actual file pins and a path
(with a truthful truncation flag if necessary); secondary diagnostic failure
retains the original refusal. This adds no generic future-import or peak/fit
claim.
