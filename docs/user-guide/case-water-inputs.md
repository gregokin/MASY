# Explicit water inputs for an existing MAPLE case

`masy.case_water.prepare_case_water` prepares CPU SYRUP inputs on an actual
verified `DenseBed`. Complete MAPLE case adoption belongs to the bed adapter;
this factory keeps that bed, its availability holdings, ledger, water,
committed topography and shared vegetation. It reads no original case and
initializes no replacement landscape. It runs no transport or completion.

The call is:

```python
prepared = prepare_case_water(bed, authored_spec, existing_control,
                              limits=declared_copy_limits)
prepared.verify(bed)
water = prepared.water       # existing DenseWaterInputs
control = prepared.control   # original DenseRunnerControl
profile_id = prepared.profile_id
identity = prepared.identity_record()
```

`authored_spec` must be a `CaseWaterSpec`, `declared_copy_limits` a
`CaseWaterCopyLimits`. No values for a user landscape are supplied by this
example. Preparing inputs does not qualify a schedule, a source pair, a GPU
path, restart or the fit of a complete event. Use the independently admitted
runner with immutable source provenance and bounded process controls.
The control must meet the existing CPU runner's array/upwind,
`commit=True`, `record_exchange_history=False`, `deep_publication="step"`
requirements. Native validation still checks all remaining control fields.

## Required authoring

Every `CaseWaterSpec` field is required. Scientific grids must be NumPy
float64 C-contiguous arrays; masks must be NumPy bool C-contiguous arrays.
Interior grids have the bed's `(ny, nx)` shape. Full hydraulic elevation and
export masks have `(ny+2, nx+2)` shape, including the authored boundary ring.
Native MAPLE/SYRUP row zero is south. Geometry, class metadata and density
come from the actual bed. This factory currently requires square cells and
the existing `constant_depth` bed context.

| Input group | Explicit fields |
| --- | --- |
| Provenance | `profile_id`, `source_reference`; these identify inputs and do not certify calibration |
| Hydraulic terrain | `elevation_full_m`, `export_receiver_full`, `active_mask`, `friction_factor` |
| Routing policy | `allow_masked_nodata`, `nodata_value`, `allow_pit_storage`, `allow_flat_storage` |
| Native soil | `soil_model`, `ksat_m_per_s`, `suction_m`, `drainage_parameter`, `theta_sat`, `soil_thickness_m`, `pavement_cover_fraction` |
| Initial retained soil water | Exactly one of `initial_theta` and `retained_soil_water_m`; explicitly set the other to `None` |
| Rain distribution | `rainfall_multiplier` |
| Wet class coefficients | One `CaseWetClass` for every actual bed class ID: `raindrop_a`, `raindrop_b`, `raindrop_c`, `raindrop_max_depth_mm`, `raindrop_depth_attenuation_per_cm` |
| Wet response configuration | `active_layer_sensitivity_mm`, `ke_model`, `kinematic_viscosity_m2_s`, `reference_interval_s`, `recession_factor_per_reference_s` |
| Native conventions | `ke_vegetation_form`, `raindrop_composition_scaling`, `distance_convention`, `dstar_convention`, `bagnold_depth_units` |

Units follow the named Native arguments: elevation and stored soil water in
metres; conductivity in m/s; moisture and cover as fractions; explicit
raindrop depth and active-layer sensitivity in mm; attenuation per cm;
viscosity in m2/s; reference interval in seconds. The existing native
constructors validate the selected response laws and ranges. For
`pavement_hawkins`, the caller must supply pavement cover; for `fixed_ksat`
it must be `None`. This does not derive pavement from vegetation.

The factory uses Native `build_routing_graph`, `terrain_reference`,
`column_parameters`, `initial_soil_water_m` (when theta is provided),
`rainfall_field` and `sediment_physics_parameters`. Terrain evidence must
match the current committed bed within Native's existing constant-datum
check. It does not invent ring heights, fill pits or choose drainage policy.

Class coefficients are mapped by ID into the unchanged bed order. Missing,
extra or duplicate classes refuse. Native increasing-diameter requirements
remain in force. SYRUP requires one common particle density; MASY refuses
mixed densities and a kg/m3-to-g/cm3-to-kg/m3 conversion that cannot return
the exact original float. It compares the actual resulting Native count,
diameters and density before returning. Native event-entry checks remain.

The same authored active mask goes to the routing, column and rain
constructors. Their actual copies are checked, rain is zero outside the
mask, and existing water-mobile sediment on an inactive cell causes a
refusal. No mobile mass is discarded or moved to make the mask fit.

Water receives the actual current shared canonical
`bed.vegetation.raster.cover_fraction` object. All other shared vegetation
arrays remain with the bed. MASY performs no cover-percent conversion and
adds no growth, mortality, uptake, ET, soil-water or friction law. An
accepted future vegetation version requires explicit new preparation.
The existing dynamic vegetation profile is refused explicitly: its
defensive snapshot interface needs separately qualified preparation.

## Identity and mutation

Preparation copies caller water arrays before Native factories may freeze
them. Caller array bytes and writeability remain unchanged. The returned
column, rain, soil, sediment and terrain contain actual Native owned
objects; no event state is constructed. The original control is retained.

`PreparedCaseWater.verify()` checks the same bed's genuine case verification,
case and profile, original spec, actual returned water inputs, control and
declared copy limits. It detects changes to caller arrays or authored
metadata and tampering with returned parameter arrays. Ordinary accepted
bed evolution is allowed; this method does not freeze the physical bed at
preparation. The runner retains responsibility for event-entry validation,
current terrain/routing, clock mapping, completion and atomic publication.

`identity_record()` returns a detached JSON-safe record. It binds the input
and returned-object digests, profile/reference, case, limits and known array
plan. It is not a calibration certificate or source-lock replacement.

## Declared preparation bounds

All four `CaseWaterCopyLimits` values are positive plain integers:
`max_cells`, `max_classes`, `max_known_host_bytes`, `max_metadata_bytes`.
The gate precedes the factory's caller-array copies and all Native factory
calls. Serialized metadata is bounded while encoding; that ceiling is not
a Python heap estimate. The existing bed is referenced and never copied.

For `C=ny*nx`, `R=(ny+2)*(nx+2)`, `K=n_classes`, and `A` the bytes of all
supplied caller grids, the Source-known array ledger is:

| Role | Bytes charged |
| --- | --- |
| Caller water arrays plus owned preparation copies | `2*A` |
| One conservative graph array envelope `G` | `616*C + 18*R` |
| Preparation graph current/candidate/error overlap | `3*G` |
| Ordinary first-rain graph construction/validation allowance | `3*G` |
| Column, rain, soil, wet vectors and their expression envelope `P` | `132*C + 256*K` |
| Native input current/candidate/error overlap | `3*P` |
| Inactive mobile inventory read-copy, comparison and inverted mask | `9*C*K + C` |
| Existing bed copied by this preparation | `0` |

`G` reserves 32 float64 and 32 int64 cell slots, 16 int32 slots, 32 byte
mask/aspect slots, one additional 8-byte cell slot, and two float64 plus
two byte full-ring slots. These conservatively group routing's retained
scalar grids, D4 donor grids, indices, sorts, masks, expression temporaries,
hash byte copies and terrain-reference copies. `P` groups 16 float64 and
four byte cell slots plus 32 float64 class slots for column/rain/soil and
the host/broadcast wet vectors and construction temporaries. The overlap
multipliers reserve original/candidate/error owners without relying on
allocator reuse or early deallocation.

Preparation genuinely calls the graph constructor twice: once directly and
once inside `terrain_reference` for its normal evidence validation. The
runner still performs its ordinary first-rain graph construction and Native
validation/commit; this factory neither bypasses nor replaces those calls.
The ledger declares the separate first-rain allowance and is retained in
the identity. Later event arrays, transport, Native commit/reroute resource
admission, interpreter/provider memory, metadata object overhead and error
traceback machinery remain subject to the existing run controls. This
array ledger is deliberately limited to named Source-visible water-input
roles and does not assert future event or whole-process fit.

## Focused checks

`tests/native/check_case_water_inputs.py` contains explicit checks on a
labelled synthetic 3-by-5, two-class Native bed, including a genuine evolved
bed with nonempty accounted water-mobile sediment. It compares actual
factory outputs with independent direct Native factory calls and checks
no-reset identity, shared cover, retained soil, masks, class/density and
conversion refusals, mutation detection and finite-budget refusals. The
fixture supplies its own south export ring and wet coefficients. Those
values are interface fixtures and must not become user-case defaults.

## Explicit interim ponded-water policy (October 28, 2026)

The case workflow continues to require authored routing flags and a supplied
control; preparing inputs does not select a new default. For the user-approved
interim choice, explicitly select Native pit storage and the existing control
flags `terminal_storage_water=True` and `deposit_mobile_at_flow_end=True`. The
C64 qualification keeps flat storage off. During water events pits retain
water. Only depth and surface-volume completion criteria exclude recognized
terminal cells; rain-off, discharge, hold, finite/nonnegative checks, final
Native sediment settlement and conservation must still pass.

SYRUP Native finalization removes and accounts for remaining ponded surface
water and performs the existing soil reset before returning an accepted dry
handoff. MASY forwards the returned bed and records Native accounting without
applying either reset again.
The terminal share is included in total surface removal; the existing soil
reset has a separate removal total. Surface, soil and mobile
state must be dry before wind. This is external interevent removal, not ET.
Future ET-capable or retained-soil-water behavior needs an explicitly qualified
profile and does not automatically enable uptake or growth.

SYRUP `919c82c` and MASY `5cab827` are Source-accepted and pushed on
qualification branches. At this qualification stage, the C64 water event with positive ponded-water
removal matched its independent Dense GPU oracle, the direct Native entry
comparison passed, and the full C64 four-event GPU wind/water sequence matched direct Native
calls in 406.51 seconds under the original limits. This changes no ordinary
case defaults and grants no general arbitrary-landscape GPU/windowed support
or production calibration. The separate failure-guard check also passed;
persisted restart is not covered by this C64 result.
