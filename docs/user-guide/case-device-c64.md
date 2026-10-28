# Imported MAPLE cases on the bounded C64 GPU profile

The imported-case API passed a bounded native comparison on October 28, 2026:
an authored synthetic NPY landscape completed wind → water → wind → water,
matching independent MAPLE/SYRUP execution after every event. The accepted
profile is exactly 8×8 cells, one class, 20 logical voxels and three resident
voxels. The ordinary `tools/masy_case.py run` command runs Dense CPU cases.
The public GPU command and C64 persisted restart remain separate work;
production calibration and general landscape capacity remain pending.

The actual run used MASY `9e7af84aecf46d4b7f6de7d0218534c03e8230c2`,
MAPLE `40d1a598b09929b2d0f831513fb6d5c015d88772` and SYRUP
`61dcf61de14ba302148a4fb1bfdbe08ddf4df10c`. Its separately committed test
source was `fc39319ef5d04156eec037bbeaf71108b8c572ba`; runtime source and
test source keep their distinct provenance. Both initializations, four wrapper
events and four native comparison events shared one original 600-second
allowance and finished in 387.636 seconds. Root independently checked source,
compiled files, accounting and process/device cleanup. Final acceptance receipt
SHA-256: `f054cd53b1f52710281a487a4230f0d34ab438bf0a0e6b3c2aef5228bd44ecad`.

Start with the normal MAPLE `case.yaml`, `model.yaml`, `wind.yaml` and supplied
NPY landscape under `source/`, together with complete explicit hydraulic,
soil, rain and wet-class inputs. Existing `resolve_case_inputs` and
`adopt_windowed_case` compile and adopt the actual Native landscape. They
preserve stratigraphy, active layer, availability holdings, water, ledger,
committed topography, prescribed vegetation, threshold modifiers and provenance.
The adapter never creates a replacement bed from a DEM or surface fractions.

The first device profile requires exactly 8×8 cells, one class, logical 20
voxels, resident window 3, full-v1 storage with one chunk and Native E1/P8
packets. Square cells, fixed capacity, a published clean initial version,
initial baselineNone and initial unseeded/dry wind state are required. No
cropping, resampling, class conversion, water reset or silent config rewrite
is supplied. Existing HOST adoption's explicit storage/datum and compatibility
guards remain unchanged.

Prepare genuine HOST water inputs with `prepare_case_windowed_water`. Supply
a separate genuine `cuda_control`: only its Native
`sediment_control.storm.implementation` field changes from `array` to `cuda`.
Native's read-only `sediment_control.cuda` property must consequently change
from `False` to `True`; it is not a constructor field. Every other field, including root
solver, completion gates, tolerances and report interval, must match exactly.
The existing public HOST preparation APIs still reject CUDA controls.

Use an explicitly authored four-event W/R/W/R case schedule. Each wind group
must contain one original one-second Native wind block, consuming exactly the
two complete blocks of `wind.yaml`. Rain forcing and physical recession remain
separate Native intervals. Actual Native settlement/recession advances the
global clock; no forcing is shifted or truncated. Other schedules/durations
are refused rather than split into extra finalization events.

The caller supplies complete genuine Native record/step/solve/validation/commit
limits, existing MASY bed/resource/runner limits, bridge limits and explicit
`CaseDeviceC64PlacementLimits`. All old C64 caps remain in force. The Native
dimension-profile object is shared by all relevant limits and context owners;
no test helper or test factory is imported by product code.

`device_c64_request_binding(...)` projects the actual verified HOST input,
authored file pins, windowed/water identity, storage choices, full original
wind mapping, CUDA control, calibration/label and all supplied limits. Root
uses that exact bounded projection to issue a genuine
`DeviceWaterC64CaseBudget` for the selected runtime, Source pair and physical
GPU UUID. It is distinct from the synthetic fixture's qualification budget.
The producer verifies actual Source/grant/origin currentness before placement;
an arbitrary caller tag, synthetic factory reference or invented descriptor
does not authorize transport. Actual Root operands are not bundled here.
The new-kind bed receiver also requires the producer's private admission for
that actual placed genesis, context, budget and request. The runner checks the
same admitted owner and binding. A case budget alone cannot admit another bed.

```python
from masy.case_device_c64 import prepare_device_c64_case

# windowed/prepared_host_water are the genuine existing HOST owners.
# kwargs holds explicit schedule/control/Native and MASY limits/calibration.
# binding and qualification_budget must join Root's actual issuance exactly.
case = prepare_device_c64_case(
    windowed, prepared_host_water,
    qualification_budget=qualification_budget,
    request_binding=binding, **kwargs,
)
runner = case.build_runner()  # Returns the guarded case over its one private runner.
while runner.stop_receipt is None and runner.accepted.next_index < 4:
    runner.step()  # Always checks the case inputs and binding around the event.
report = case.report()
```

Native placement moves resident/noncolumn arrays and explicitly binds copied
prefix packets while keeping the same immutable deep-store handle/version.
The deep column is not reconstructed or uploaded. Every transferred leaf is
checked byte-exactly at placement. The runner owns its canonical device
vegetation/cover and private soil/mobile state; exposed `case.water` and
`case.wind` are the retained actual initial inputs, useful for an independent
Native comparison. The underlying runner is private. `accepted` exposes its
genuine frozen boundary; `stop_receipt` returns its copied stop metadata.
Reports copy the known-prefix data; mutating a returned report cannot change
later admission checks.

The positive construction record counts caller-retained dense input roots,
the HOST windowed/water/config state, actual alias-counted transfers, HOST
empty-pool construction, packets/encoded base, verification/error/current-next
copies, metadata/Source reads and IO/store overlaps. Existing downstream bed,
resource and Native admissions remain additive. Caller originals may remain
alive; there is no deduplication, release, automatic garbage collection or
whole-process allocation proof. Constructor/provider/JIT terms remain opaque.
Prospective Native wind-config leaves and intermediate canonical/encoded
metadata have positive reservations before those allocations. The actual
mapped wind-config array inventory is checked against its Source envelope.

Root must install and monitor the original complete attempt: CPU480/540s,
one600s wall scope,2GiB active pool,5GiB sampled RSS,9GiB driver growth,
512MiB whole run/store,64MiB files and4MiB streams. This includes preparation,
all events, retained owners and any independent oracle, without resetting a
timer. The producer reads real OS/pool/Native-image/headroom state before
placement and preserves the actual worker/process/device identity.

Preserve authored routing/boundary choices. The supported terrain has no
masked/nodata or flat-storage lane; pit storage is an explicit Boolean input.
For this first imported interface profile, the already approved terminal
surface removal and mobile deposition flags must be explicitlyTrue. Native
finalization settles sediment and removes/accounts surface and soil water
before returning accepted DryHandoff. MASY records/forwards it and applies no
second reset. This is external dry accounting, not simulated ET.

All model/calibration checks still apply. The authored reduced model must
already satisfy the existing topographic-wind/separation/creep/avalanche and
constant-depth coupling profile. This first Source supports only the labelled
`synthetic_test_fixture` interface declaration. It supplies no measured
calibration or implicit response coefficients for a user's production case.
An unsupported scientific input receives a refusal; changing it requires a
separate explicit decision and qualification.

Input/currentness checks surround each whole event. A failure permanently
stops this case wrapper; the existing runner retains its genuine last accepted
boundary and active failure evidence. No deep rollback or deletion is supplied.
Authored caller objects are retained, so later caller changes fail closed;
the adapter does not silently copy them into a different scientific input.
Retain all referenced immutable versions and failure artifacts, then use a
fresh admitted attempt rather than replaying the same owner.
