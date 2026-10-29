# Native case inputs with windowed storage

Start with the normal MAPLE package: `case.yaml`, `model.yaml`, `wind.yaml`
and the supplied landscape under `source/`. MAPLE resolves and compiles that
package, including the imported DEM, stratigraphy, class order, vegetation,
availability holdings, masks and provenance. MASY then converts the actual
compiled column through Native `windowed_column_from_dense`; it does not
initialize a replacement bed from the DEM or surface fractions.

`masy.case_windowed` supplies a HOST input/adoption API. It does not grant an
event-runner profile. The existing qualified 4×4/K1 windowed wind/water
profiles retain their own shape and configuration guards. General-domain
transport and GPU placement need separate qualification.

Within a locked, bounded CPU worker, use a fresh run-owned case copy and a
fresh store outside that case directory. The existing case workflow owns
input discovery and Native compilation; its compile mode publishes Native
processed files, so select the run-owned copy explicitly. On the separately
qualified MAPLE `a88f71cc` revision, the compiled-case loader authenticates and
reloads the authored package model configuration in load mode; it does not
silently substitute default model settings. Compilation and loading still have distinct provenance. The initial
capsule workflow below requires a genuine load before its first adoption.

```python
from pathlib import Path
from masy.case_workflow import CaseInputLimits, resolve_case_inputs
from masy.case_windowed import CaseWindowedLimits, adopt_windowed_case

# These are explicit example engineering ceilings for the labelled small
# adoption qualification. Select actual limits in the owning bounded worker.
io = CaseInputLimits(1 << 20, 256, 8 << 20, 16 << 20, 16 << 20, 128 << 20)
limits = CaseWindowedLimits(
    max_cells=72, max_classes=2, max_voxels=20,
    max_known_host_bytes=128 << 20,
    max_metadata_bytes=4 << 20, max_store_bytes=16 << 20,
)
inputs = resolve_case_inputs(Path("/run/owned-case"), mode="compile", input_limits=io)
windowed = adopt_windowed_case(
    inputs, store_dir=Path("/run/deep-store"),
    datum_offset_m=0.0, window_voxels=3, chunk_columns=64,
    rows_per_band=2, with_prefix_packets=True, limits=limits,
)
windowed.verify()
```

All storage choices are explicit. Native `DeepStoreHeader` requires
`64 <= chunk_columns <= 1024`; an unsupported positive value raises Native
`ValueError` before the store builder creates artifacts. The example's
72 cells form two chunks of 64 and 8 columns. The two-row streaming band is
independent of the chunk width; it does not require a separate chunk per band.

Datum `+0.0` is MAPLE's existing dense
physical frame; the adapter does not shift elevations. Native committed
regrouping runs with `require_exact_height=True, trusted_packet=False`,
records its genuine receipt and rejects incompatible packet heights. The
adapter rejects enabled authored voxel-capacity growth before creating a
store and before shape-limit checks, rather than disabling it. The current API requires square cells,
initial airborne state `None` and reconciliation baseline `None`; it retains
those explicit compatibility limits. Authored configuration, boundaries,
availability and scientific coefficients are not replaced.

Initial clock and context defaults match the existing Dense case-adoption
path: time is `0.0`, and the native `BedContext` defaults supply
`depth_update_rule='constant_depth'`, exchange-history false and the default
deep-storage options. This API does not infer a new clock from forcing or
change a native water policy.

The result holds the genuine Native windowed column and six-field SYRUP
`BedState`, complete active layer, water, ledger, committed topography and
availability, plus vegetation, threshold modifiers, Native config and copied
noncolumn statics. The seventh landscape field is explicitly retained as
`ledger_reconciliation_baseline=None`. Source identities and the complete
original input/provenance image are retained as bounded canonical metadata;
no `CaseInputs`, `CompiledCase`, `DenseBed`, dense-column array, bound method
or closure is retained by the result. Its readonly initial arrays and
content identity are checked by `verify()`, which also authenticates every
deep file and uses Native streamed partition/packet validation. This is an
input boundary check, not work to repeat on every transport step.

Native compilation still materializes the initial dense column. The caller
can retain `inputs` after adoption. The construction record charges those
original array roots, candidate noncolumn arrays, resident window and prefix
packets, Native regrouping/fallback owners, a declared band/chunk payload
and metadata text. The retained context availability-fraction copy has its
own positive `candidate_available_fraction_bytes` term alongside the native
fallback/map envelope. This named known-owner plan does not bound every Native
temporary, Python/provider allocation or mapped page, prove whole-process
fit, or promise that dropping a reference releases memory. The owning worker
must retain its actual AS/CPU/wall/file/output policy. This API provides no
out-of-core compilation or automatic store collection.

Pre-construction refusals leave no store when the input/storage preflight
stops before Native construction. Once that construction starts, a later
refusal can leave partial chunk files or a complete published genesis even
though no adopted owner was returned. Regrouping, final source checks and
full owner-record metadata checks occur after Native writes. This boundary
does not roll those writes back or delete any store/version. Retain the
failure evidence and use a fresh store path for a retry; reusing the same
existing path is refused. Do not remove storage referenced by another owner
or checkpoint. No generic cleanup or garbage collector is supplied.

For explicit water inputs, pass the same complete authored `CaseWaterSpec`
and existing `DenseRunnerControl` used by `prepare_case_water`, with
`CaseWaterCopyLimits`, to:

```python
from masy.case_windowed import prepare_case_windowed_water
prepared = prepare_case_windowed_water(windowed, spec, control, limits=water_limits)
prepared.verify(windowed)
```

The shared private factory calls actual SYRUP routing, terrain-reference,
infiltration, rainfall and class-specific wet-parameter constructors. The
prepared owner is distinct from the old Dense owner. It keeps the actual
canonical vegetation cover, explicit hydraulic boundary ring, masks,
friction, soil state, rain scaling, coefficients and conventions; it infers
none of these from periodic wind boundaries or a sample case. Original
Dense `prepare_case_water` guards, defaults and returned Native values remain
on their existing public path. Adapter/ownership failures use
`CaseWindowedError`; the shared wet-input factory retains `CaseWaterError`,
including invalid Dense control types and invalid wet parameters. Prepared
identity verification binds the spec/water/control byte fields to the hashes
inside its hash-bound identity record, so replacing both a controlling object
and its cached bytes cannot keep the old identity. Preparing these parameters performs no rain,
transport, finalization or completion.

Actual Root qualification is required before this new input boundary is
called by a collaborator run. The authored check covers an imported
9×8/K2/V20 synthetic case, complete logical values and noncolumn fields,
compact ownership, original-input immutability, explicit refusals and Native
water inputs. Those values are interface fixtures, not calibration.

## Reopen an initial imported C64 adoption

The additional `save_initial_windowed_case` and `reopen_initial_windowed_case`
APIs are an implemented bounded initial-input capability with separate Source
and genuine HOST validation. They provide no event restart, GPU placement,
imported CaseBudget restoration or production calibration approval. The
[imported GPU saved-boundary API](case-device-c64-restart.md) has its own
physical qualification. This capsule alone grants no execution authority or
physical qualification.

This first profile is an imported, passive HOST 8×8/K1/V20/W3 bed, published
full-v1/gen0 with one chunk, E1/P8, fixed capacity, datum `+0.0`, no pending
journal/window cache, no initial seed/baseline and a mapping-cache capacity of
8. The original authored geometry, class ordering, availability, vegetation,
thresholds and numerical settings remain unchanged. Other profiles refuse.

Compile once and keep its provenance as its own subject. Then genuinely use
`resolve_case_inputs(mode="load")` **before the first adoption**, water
preparation, request or reference. Load and compile are different identities;
never retag an existing adopted owner. This loaded-subject path requires the
actual availability builder to produce zero TransformLogs, because load does
not restore those logs. Imported elevation logs do not authorize dropping
availability provenance. A genuine loaded adoption must pass the existing
Native/MASY checks.

```python
from masy.case_windowed import (
    save_initial_windowed_case, reopen_initial_windowed_case,
)

# inputs is already a genuine loaded CaseInputs; windowed is its one adoption.
# Use explicit caller-owned paths/limits in the original bounded worker.
saved = save_initial_windowed_case(
    inputs, windowed, capsule_dir=initial_capsule,
    max_capsule_bytes=4 << 20,
)
# A later caller may already own a separately genuine mode='load' CaseInputs.
reopened, receipt = reopen_initial_windowed_case(
    freshly_loaded_inputs, capsule_dir=initial_capsule,
    expected_capsule_sha256=saved["capsule"]["sha256"],
    limits=windowed_limits, max_capsule_bytes=4 << 20,
)
```

Reopen itself calls no Dense loader, compiler, adopter, DEM reconstruction or
committed regrouping. The genuine Native snapshot supplies the column,
active layer, availability and water and selects the **original manifest
digest** at the same canonical store path. Two saved arrays supply the
original Native committed voxel baseline and context available fraction.
Authenticated matching caller inputs supply the ledger, other committed
fields, config, noncolumn statics, vegetation and thresholds. The full
original HOST record stays verbatim. Actual reconstructed values and saved
object-alias positions must match before an owner is returned; fresh water
preparation must keep the canonical vegetation-cover object.

The six files are `native.npz`, `committed-voxel.npy`,
`initial-available-fraction.npy`, `host-record.json`,
`case-input-record.json` and final `CAPSULE.json`. The caller externally pins
that final file's size/SHA. Files, headers, original deep descriptor/manifest
and chunk are bounded and authenticated before Native reads. The Native
archive is stamped and hashed before and after its genuine loader. Exclusive
writes and directory fsync precede and follow the final publication marker;
use an ordinary hard-link/fsync-capable filesystem, not drvfs. Failure keeps
truthful partial files and returns no usable new owner. Existing artifacts
are never replaced or deleted.

Reopen preserves the original regrouping receipt, construction plan and
band-size choice as **historical** fields. It does not repeat that arithmetic
or present them as current measurements. Its separate receipt prices actual
caller/copy/metadata/read occurrences and reports its own IO. Native runtime
LRU contents/locks and global diagnostic counters are not historical physical
state; they are not reset. Limits on capsule bytes are IO eligibility only,
not an allocator/RSS/whole-process fit theorem. Opaque service costs and the
original worker's actual limits still apply.

Root's reference/writer protocol also retains a separately authenticated
`original-input-proof.json`: full original compile/load/HOST records, all four
raw prepared-water byte sequences, original own-encoded device request and
public request. Reprepare water and invoke the real projector on the reopened
owner; encode the **fresh** dict with that producer's encoder and compare
bytes exactly to the original before fresh budget issuance. The product
capsule issues no such budget and contains none of that external authority.

Both serial roles require the same newly accepted immutable Source profile.
Continuous must finish and all owned processes must be gone before writer
reopens or publishes. Native content-addressed versions permit serial
same-depth siblings from gen0; retain every branch/version/orphan under the
same quotas. Never reset a HEAD, delete later files, copy the full store or
assume an existing content filename was independently rehashed by its writer.

The initial caller's existing genuine load still materializes its original
Dense inputs, which may remain retained and positively counted. This is an
explicit small C64 setup limitation, not scalable loading or per-event
materialization. The reopened physical bed is exclusively the Native
windowed snapshot plus its retained immutable deep references.
