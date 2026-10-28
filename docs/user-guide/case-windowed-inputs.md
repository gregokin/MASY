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
processed files, so select the run-owned copy explicitly. The native loader's
existing case-configuration restrictions still apply in load mode.

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
