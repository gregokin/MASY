# Adopt a MAPLE case without rebuilding the bed

The input is the normal MAPLE case directory: `case.yaml`, `model.yaml`,
`wind.yaml`, and original imported files under `source/`. MAPLE continues to
own parsing, units, transformations, stratigraphy, calibration validation and
compilation. This interface adopts the resulting complete Native case.

It is an input/ownership interface. It does not run alternating events, create
a combined restart, select a calibration, or certify an arbitrary-size case.
The case-driven CPU event runner, water-input factory and command-line entry
are separate work. There is currently no installed `masy case` command.

```python
from masy.case_workflow import CaseInputLimits, resolve_case_inputs, adopt_case_inputs
from masy.dense_bed import DenseBedLimits

# Explicit engineering declarations chosen for the supplied case and worker.
# Their values are not scientific case parameters or site-calibration evidence.
inputs = resolve_case_inputs(case_dir, mode="compile", input_limits=input_limits)
adopted = adopt_case_inputs(inputs, bed_limits=bed_limits)
bed = adopted.bed
native_config_authority = adopted.inputs.resolved.config
wind_record = adopted.inputs.resolved.wind_record
```

`input_limits` must be an exact `CaseInputLimits` with positive integer fields
`max_authored_file_bytes`, `max_source_files`, `max_source_file_bytes`,
`max_source_bytes`, `max_metadata_bytes`, and `max_compiled_bytes`. The source
entry count includes directories and the complete generated-input inventory.
Source and compiled symlinks are outside this initial input profile. These are
I/O and metadata eligibility bounds, not a compressed-array decoder heap bound.
Use a genuinely bounded immutable CPU runtime/worker; the interface does not
set or weaken its OS limits.

Native's shared import inventory covers DEM/sediment/vegetation/forcing,
threshold fields, availability, stratigraphy and prescribed shear sources.
Their authored references must remain under `source/`; external paths are
not silently followed or omitted. Reduced-detachment calibration coefficients
and provenance labels are authored values, not a calibration-file loader.
Current Native imported prescribed-shear construction lacks the required
case-directory argument and refuses that input; this interface does not
bypass the Native refusal or claim that profile is supported.

`bed_limits` is the existing exact `DenseBedLimits`. Its
`max_voxels_per_column` applies before owned copies;
`host_memory_cap_bytes` checks the positive known original-array, initial,
working, next/error, static-fallback and encoded-metadata payload plan;
`stage_copy_max_bytes` bounds the complete working-bed stage. Python/native
allocator and solver scratch remain subject to the original worker/Native
admissions and OS ceiling. A plan passing is not a whole-heap fit promise.
Limits are recorded separately from the scientific initial-case digest.

Modes:

- `validate` resolves all three authored files and returns `CaseInspection`;
  it does not fabricate a compiled state or write `processed/`.
- `compile` calls Native `compile_case_package(..., overwrite=False)`. MAPLE
  owns its transactional outputs. Existing compiled output is refused. A
  later CLI can stage a run-owned case copy; this API never silently overwrites
  user-authored files.
- `load` resolves the full package and calls Native `load_compiled_case`. It
  requires all original/imported/compiled files and Native provenance. Current
  MAPLE main's loader re-resolves `case.yaml` alone; if this loses authored
  `model.yaml` values, this adapter reports `case_config_mismatch` and refuses.
  It does not rebind a mismatched Native config or repair the upstream loader.
  Imported availability also requires the original Native transform-log
  records; a loader that does not restore them refuses that fallback rather
  than inventing its provenance.
  Compile mode supplies the correct resolved/compiled pair for the first path.

The adoption retains the actual voxel column, active layer, water, sediment
ledger, committed topography, current available/bound holdings, prescribed
vegetation, thresholds, initial mobile pool, expected inventories, masks,
stratigraphy, imported-source inventory, transform logs and provenance.
Caller inputs remain unchanged. `adopted.inputs` owns a complete independent
initial envelope, with read-only initial arrays; `bed.state` owns separate
working Native physical fields. New identities cover every initial descendant
and all file pins, while ordinary bed identities describe evolving state.
Canonical metadata preserves even diagnostic NaN/inf by exact binary64 image;
Native validators retain all physical/config finite-value requirements.
The retained initial mobile pool is also read-only; an event runner must
preflight and make its own working pool rather than updating that snapshot.

The ordinary uniform initializer `build_dense_bed` remains unchanged. It must
not be used to adopt a layered case: it fills a new uniform bed from elevation
and fractions. The new private factory never deposits or extracts an active
layer again. Derived depth-integrated mixture summaries do not replace the
case's layer information.

The single configuration authority for future scheduler/context construction
is `adopted.inputs.resolved.config`. Native compilation parses a separate,
equal config object. MASY validates their full Native values, then uses the
owned resolved geometry/classes/specs by identity. No case boundary,
avalanche flag, dry-separation setting, physical coefficient or timestep is
automatically changed. Adoption does not establish their later coupling
eligibility.

Availability's current holdings remain exactly compiled. Only the static
replenishment fallback is rebuilt through the actual Native builder/resolver,
using the case's authenticated original root, seed and declared precedence.
Its private reconstruction log is never merged into the original transform
log. The complete fresh availability log subset must match the compiled
Native logs by exact typed metadata identity, including diagnostic float bits,
and each imported source must match the authenticated Native source inventory.
Source files are checked before and after rebuilding. Unseeded initial
holdings are compared with the Native forward initializer; seeded holdings
are checked by Native seeded-state validation, never fraction-times-current-
active or a ratio. Seeded mass remains in the retained mobile pool. Expected
per-level/per-cell arrays retain their Native pre-seed-deposition meaning;
Native total reconciliation includes the post-seed bed plus mobile mass.

First adoption is dense host NumPy/float64, square cells, fixed voxel capacity,
with a genuinely absent reconciliation baseline. Existing Native and DenseBed
validators remain in force. Non-None returned baselines are refused before
publication, not discarded. Nonempty initial seeds can be adopted and
reconciled here; the first event runner may explicitly refuse their Native
trajectory family before transport. Water is retained as the actual Native
WaterState, including exactly dry tracked state; it is never replaced by
untracked `None` because wind-water transport is disabled.

Original source/artifact changes or mutation of the complete owned initial
metadata are detected by ordinary bed input verification/stage operations.
This initial correctness path hashes the complete initial array forest and
files at each verification, then retains the original DenseBed checks (which
also hash held initial arrays). It avoids extra materialization but is not a
constant-cost verifier for large cases.
Failure produces no adopted case or accepted model boundary. Native compiled
outputs written before a later adapter refusal remain Native outputs; they
are not relabelled as a successful MASY run.

A later alternating run also needs explicit hydraulic boundary/ring, soil,
rain and wet-transport inputs. MAPLE's DEM or wind moisture fields do not
supply those laws. Preserve Native class ordering/density and actual soil/
routing policies; unsupported combinations require an actionable decision.
No combined restart or arbitrary-windowed/GPU claim follows from adoption.
