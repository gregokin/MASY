# Dense CUDA daily integration

[Accepted sources and executed qualification](dense_triplet_cuda_qualification.md).

`masy.daily_triplet_cuda.DailyCudaTripletSession` connects actual MAPLE wind,
SYRUP rainfall/recession/daily water and LEAF demand, feedback and growth on one
resident Dense CUDA landscape. The explicit profile is
`masy.daily_triplet.dense_cuda.v1`. Use the separate source/runtime lock in
[triplet_cuda_runtime.md](triplet_cuda_runtime.md).

The first qualification is a small synthetic 40 × 4 landscape, three sediment
classes and three soil layers. It covers two wind occurrences, full rainfall
and recession, eight daily E/T updates, one weekly growth update and a following
wind. Independent controls cover standing water during wind, near-saturated
closed-bottom capacity return and genuine recession, native stale/replay
refusals, candidate mutation, publication callback failure, distinct host/device
budget refusals and the selected physical GPU UUID. This qualifies the stated
Dense API composition; Windowed storage, large-domain throughput, additional
wind integration paths and combined persisted restart need separate evidence.

## Native ownership and clocks

MAPLE owns the bed and wind transport. SYRUP owns every water update, including
LEAF withdrawal. LEAF proposes demand, consumes actual accepted feedback and
updates plants. No physical law is implemented in MASY's CUDA modules.

`daily_wind_cuda` clones the admitted Dense bed on the same device, invokes the
native scheduler/finalizer and checks native class budgets. A wind duration or
resource limit is never completion. The native CUDA water handoff validates
the real finalized bed and carries current W/J/I, surface water, routing source
and the applied vegetation generation. Changed columns alone use SYRUP's
existing property/capacity reconciliation. Returned water joins the surface
once; an actual `pending_recession` boundary must finish native recession before
another process begins. Open-bottom export is accounted separately.

Wind treats the soil as dry. It has no soil-moisture threshold feedback and
preserves actual soil and surface water. This is an explicit approximation.

Wind, rainfall/recession and interevent periods do not overlap. E/T/growth are
interevent operations. Wet interevent hydraulics and E use the existing native
substeps; plant demand/withdrawal and biological feedback occur once per nominal
day. No catch-up E/T/growth is added for event time. Native physical clocks and
nominal calendar days remain distinct; annual realized elapsed time can exceed
365 days. Same-date wind rows retain file order and precede that day's rain.

## Construction

Construct genuine inputs under the sealed CUDA runtime:

1. Verify the exact qualified source pair using
   `native_case_profile.CURRENT_CUDA_PROFILE_NAME`; admit the authored native
   case. Its MAPLE execution backend must explicitly be `cupy`.
2. Use `native_dense_cuda_profile.prepare_native_dense_cuda_case` to place the
   complete bed and availability on the selected device once. Initialize the
   native floating soil configuration with backend `cuda` and explicit host
   and device caps/readers/reserves. Supply the actual native static event factory.
3. Bind `leaf.floating_lifecycle_bridge_cuda.bind_source` to that genuine source
   capability. Build a CUDA `SessionConfig`, explicit catalog/plant population,
   nominal anchor, native site controls, E partition and day/event limits.
4. Create one `GpuPolicy`: current logical device, actual physical UUID,
   independent host/device caps, reserves and truthful available-byte readers.
   Pass it to `prepare_triplet_wind_cuda` and `DailyCudaTripletSession`.
5. Call `advance()` once per nominal day. It returns a bounded accepted scalar
   receipt. Keep any raised `TripletError.active_state`, the lineage's actual
   accepted/active head, plant frontier and `session.failure` as failure evidence.

The caller supplies the genuine terrain PET factor at the case latitude;
PET is applied per cell to both E and plant demand. The qualified control uses
flat terrain factors. Automatic regeneration of slope/aspect PET after bed
changes is not introduced by this API.

There is no new full-scenario CUDA command in this milestone. The committed
synthetic control definitions in `tests/cuda_triplet_support.py` show concrete
construction; they are engineering fixtures, not scientific parameter advice.

## Resources and transfers

No per-step host mirror of soil W/J/I or Dense voxel storage is made. Wind uses
resident staging and native physics. Dirty-column comparison is a bounded cold
device scan; only selected compact soil-property tiles use the existing host
provider. Cold post-stage/adoption integrity uses native chunked SHA256 streams.
These boundary streams are explicit costs, not evidence of large-domain speed.

The existing vegetation binding APIs require host canonical mask/surface and
plant mapping metadata. The adapter explicitly downloads the current surface
at vegetation-generation boundaries, uses native host projection/import APIs,
then places native vegetation fields/covers on the device. Root-map demand,
resource availability, proposed withdrawal and accepted feedback remain resident.
LEAF's existing aggregate memory ledger is conservatively charged to each
independent pool; free host and device memory are never pooled.

All device calls require the same current device. `GpuPolicy` checks its actual
physical UUID and never silently switches devices. The external supervisor must
check availability/allocation and impose process, RSS, driver-memory, output and
time limits. Payload estimates exclude driver/JIT/allocator overhead; record
actual sampled peaks and native transfer counters separately. Source locks alone
confer no device authorization or physical qualification.

## Failure and restart

Native transactions publish only after validation; the wind handoff guards the
expected head, staged content and one-shot adoption. A failed day stops that
session. The entire wind/storm/day/plant sequence is not one rollback transaction.
Do not retry it from a partial head or replay a debit.

`save_checkpoint`, wind cursor checkpoint/restore and combined CUDA context
restore explicitly refuse this unqualified combined restart. Old CPU profiles,
locks and defaults retain their own exact source identities. Selecting these
new upstream GPU commits for a CPU run requires its own source qualification;
there is no implicit rebinding of an old lock.

The adverse near-saturated closed-bottom standing-water control verifies real
recession initialization, one accepted native step and wind exclusion while
water is active. Full recession for that control exceeded the external 300 s
CPU guard; that attempt remains incomplete. No tolerance or physical law was
changed to complete it. Long wet recession performance is not qualified here.
