# Small same-backend GPU accepted-boundary restart

This additive candidate uses the existing MASY atomic boundary store for the
closed C16/K1/V6/W3 four-event GPU profile. A boundary is accepted only after a
genuine complete wind or water event, independent Native/companion readback,
all existing accounting checks, file/directory synchronization and atomic HEAD
publication. This Source does not assert that the new persisted GPU restart
has passed a physical check. Root supplies current immutable runtime facts,
fresh authority and physical verification separately.

The existing four store profiles and the original event implementations keep
their behavior. The new exact schema/profile pair is
`masy.windowed_device_boundary_store.v1` /
`small_C16_K1_V6_W3_gpu_accepted_boundary_v1`. Mixed pairs refuse. An active
event, stopped event, pending deep publication or nonempty airborne pool is
not an accepted boundary. There is no fallback to a previous generation,
active-event resume, backend migration, relocation or automatic cleanup.

## Owning save and restore

The two entry points in `masy.dense_boundary_store` are:

```python
create_windowed_device_boundary_store(
    path, *, runner, qualification_budget, limits=..., storage_contract=...)
recover_windowed_device_boundary_store(
    path, *, expected_store_file_sha256, qualification_budget, limits=...)
```

Both return the existing metadata-only boundary-store facade. Close the
handle in its creating process/thread, for example with `contextlib.closing`.
Call `handle.step(handle.head_token())` for one accepted event transaction.
A previously accepted parent token returns its exact child acknowledgement
and receipt without another load or transport. This uses the existing
operation/receipt/parent-chain machinery; the new codec is not a second
publication protocol. Callers cannot obtain an accepted bed or GPU capability
from the facade. The old active-process borrow/checkpoint APIs reject this
new profile.

Creation accepts only an untouched genuine selected-profile runner at cursor
zero. Recovery requires the externally pinned STORE-file hash and the same
retained absolute deep-store location. The physical GPU UUID, scientific
Source images, Native source identity, numerical controls, fixed execution
budgets and compatible immutable runtime/environment must match. A different
Source image requires a separate qualification, not an automatic rebind.

Native `save_windowed_state_snapshot` and `load_windowed_state_snapshot` own
the resident column, immutable deep version, active layer, availability and
water. The companion supplies the complete missing ledger and compensation
fields, committed topography, optional reconciliation baseline, actual empty
pool, current soil, prescribed vegetation, controls, statics, routing and
MASY bookkeeping. No bed is rebuilt from a DEM, surface composition, original
case or full-depth dense forest. Native streaming deep validation is kept.

The companion admits a closed Source-owned Native type roster, exact field
names/order, dtype/shape/order/backend metadata, bounded numeric byte vector,
canonical finite float hex and exact integers. There is no pickle, file-
directed Python class import, callback or serialized device address. Every
Native NPZ member header, raw member size and exact selected shape/dtype is
checked before Native expansion. Whole-file hashes and semantic seals bind
the original capsule, physical snapshot, image, controls and ancestry.

Restoring invokes genuine Native placement and the existing MASY owning bed
and resource constructors with the fresh genuine budget. It installs only
saved **MASY** bed/resource logical generations and the MASY routing issue
counter, after checking their accepted cursor. It never writes a Native
private counter, changes a Native version/commit/reconciliation counter,
retimes a physical state, calls genesis or replays an event. New actual
prefix bindings, process/thread/device/stream ownership, mirrors, references
and the new routing stamp come from the owning factories.

Native snapshot step, lossless time, committed stamp, ClockLedger endpoint,
receipt count and cursor must agree exactly. Old receipt bytes and the entire
clock prefix are retained. All cumulative water counters survive restart.
The existing three ceilings remain work, logical reads and new versions;
H2D, D2H, scalar reads and gate counts remain guarded nonnegative int63
diagnostics, with no invented new ceiling or counter reset.

Routing is rebuilt with the actual Native factories from the saved **last
hydraulic source**, then compared against all saved graph/network/grid
fingerprints. That cache construction does not increment physical rerouting
or commit accounting. Exactly one decoded terrain object belongs to both the
water input and held routing namespace. Water cover aliases the exact owned
prescribed vegetation raster. Actual Native reroute and cache invalidation
rules remain the event engine's rules.

## New role authority and retained evidence

Root's protected worker consumes one fresh, unexpired, one-use launch/device
grant for each actual OS role before importing the scientific subjects. The
store is not a launch issuer. It takes that role's genuine
`DeviceWaterQualificationBudget`; its parent/work and readback owners share
the same budget object and unchanged runtime/source checks. A later role
needs a new genuine budget and owning resources.

Saved old budget/current-runtime/grant SourceReferences remain immutable
provenance evidence. Their role paths and nonce bytes may differ from the
new role. The new receiver authenticates the actual new references, source
origins and complete budget bytes and compares the same Source images,
profile, caps, physical UUID and runtime/environment invariants. It neither
reuses an expired launch authority nor overwrites the old evidence to make it
look fresh. Root independently joins the actual compatible runtime/binary
profile, cold constructor witness, selected UUID and process birth.

## Bounded boundary operation

The original Root controls stay 2 GiB active pool, 5 GiB leader RSS, 9 GiB
driver growth, 512 MiB whole-run storage, 64 MiB per file, 4 MiB combined
output, CPU 480/540 seconds and 600 wall seconds. Native safety fraction 0.5
and all original reserves stay in force. There is no pool reset, extra
per-event allowance or prewarm exception. Parent/work/readback and test
oracles must be included in the one role budget before actual construction.

Companion numeric bytes are at most the original 1 MiB; Native snapshot raw
bytes remain at most 64 MiB. Original image metadata/header/serialized/raw
caps stay intact. The new profile tightens the common store to 512 MiB, HOST
reservation to 5 GiB, five accepted generations and four events. The original
free disk/inode reserves, orphan bounds and writer lease remain unchanged.
The actual Native whole-run inventory includes the new store reservation,
retained deep files, partials and orphans before boundary writes. Native event
writers keep their own deep-write preflights. Boundary serialization creates
no deep version and never deletes or rewrites an old one.

The boundary projection keeps positive caller/input, accepted/mirror,
readback, current/next and first-error payload occurrences. The inherited
Native bed/resource constructors also retain their complete existing
positive planners and original headroom checks. All new boundary IO/copies
are additional work. Retaining the original Native ceilings is deliberate;
no deallocation, garbage collection or allocator reuse is assumed to make a
run fit. The eleven provider roles, eight primitive units and constructor
allocation remain opaque `None`. This is bounded physical qualification of
known payloads, not an allocator or arbitrary-scale fit theorem. The old
runner report's `restart_supported=False` describes its standalone interface;
the separately qualified new boundary-store profile owns this capability.

The first profile assumes one quiescent owner of the immutable store/deep
files and the lifetime MASY writer lease. Native load reopens paths. Files
are bounded and authenticated, with path/inode/mtime/size and byte identities
checked before and after delegation. This is not descriptor-atomic loading
against a hostile concurrent writer, hardware power-loss proof or permission
to modify artifacts used by an active job.

## Focused acceptance and four serial roles

Root first runs only the two new metadata modules:

* `tests/unit/test_windowed_device_checkpoint.py`
* `tests/unit/test_windowed_device_boundary_store.py`

They check malformed dtype/shape/header/offset/alias/class/scalar records,
authenticated corruption, closed rosters, all four legacy profile pairs,
the new exact fifth pair, no missing-authority Dense fallback, clock and
logical-generation joins, positive original ceilings, and exact duplicate
receipt behavior. They do not construct a Native model or a provider.

The one physical selector is
`tests/native/check_windowed_device_boundary_restart.py::test_windowed_device_boundary_restart_role`.
Root invokes it four times in serial, each in a distinct fresh protected
worker/controller with one new genuine grant and the original limits:

| Role | Genuine operation |
| --- | --- |
| `continuous` | Independent Native fixture, one continuous W/R/W/R run; write five complete tiny scientific boundary oracles. |
| `write-W1` | Independent untouched fixture, create the real store, accept W1 and close at cursor 1. Compare actual loaded/readback boundaries with the continuous oracle. |
| `recover-R1` | Load the selected persisted W1 in a new process, verify exact old receipts/debits and immutable files, accept R1, close at cursor 2. No genesis. |
| `recover-finish` | Load selected R1 in a new process, test duplicate W1-parent acknowledgement without loader/event calls or HEAD change, then accept W2/R2 and compare all remaining scientific boundaries. No genesis. |

The protected fixture `windowed_device_restart_role` is Root-owned and must
provide exactly `schema`, `role`, `shared_root`,
`expected_reference_index_sha256`, `expected_store_file_sha256`,
`expected_control_sha256`, and `storage_contract`. Schema is
`masy.windowed_device_restart_test_role.v1`. The root is an existing canonical
owned absolute directory; role names are the four literals above. Root
authenticates predecessor exit/reaping, new process birth, one physical UUID,
fresh authority, actual Source/currentness and all explicit expected hashes.
There is no in-worker coordinator, nested child or inherited grant. A missing
fixture or unissued hash fails closed, without a synthetic success fallback.

Continuous role has all three expected hashes `None`. Write-W1 receives
Root's actual `reference/index.json` SHA; its other two hashes stay `None`.
The two recover roles receive the same reference-index SHA, actual STORE SHA,
and actual preceding `control/wind.json` or `control/water.json` SHA. These
are facts from actual prior roles, never placeholders. `storage_contract`
contains the original explicit local/fsync declarations. Root retains the
whole shared directory and every restart-referenced deep version.

The oracles stream complete tiny array byte signatures, including resident
and 16-column test-only full readback, all seven bed fields, pool/soil,
statics, vegetation, routing, clocks, class/water totals and scientific
receipt fields. Worker/PID/stream/copy diagnostics and resource predicates
are compared through the separate persisted exact receipt/debit checks,
rather than being mistaken for cross-process physical equality. Those full
original product fields are not removed or rewritten. All oracle/control
payloads stay local under Root's combined budgets and are excluded from
Source-only review packets. Physical failure is truthful incomplete work;
the unchanged last accepted HEAD and actual primary evidence remain the
authority.

The closed Native metadata registry additionally includes the canonical
`maple.surface.voxels.windowed_delta:WindowedDeltaBudgets` and its exact six
required fields. The original167 canonical entries, wire schema, class/field
refusals and genuine Native constructor checks remain unchanged. This fixes
the specific original-input encoding omission; it adds no arbitrary class
import, alias fallback, budget change or storage/recovery qualification.
Root's focused CPU metadata selector is
`tests/native/check_windowed_device_checkpoint_registry.py::test_native_windowed_delta_budget_registry_roundtrip_and_strict_refusals`.
It uses the real Native budget type and checks nested encoder/preflight/host
decode identity, the exact old omission, malformed/unknown-class refusals and
the unchanged Native constructor bound. It performs no transport or GPU work.
The author has not executed it. Actual fresh-write and recovery roles still
require independent review, a coherent runtime and Root qualification.

## Typed values and their JSON representation

The companion retains its original logical limits: at most100000 typed
occurrences and typed depth24. A typed dictionary field is encoded as a JSON
pair list containing a name and another typed node. These representation
wrappers are not additional admitted scientific values. Private closed-wire
checks therefore use derived JSON bounds16N+1 nodes/max(3D+2,4) depth; an
original or current record with one direct wire member uses17N nodes/
max(3D+3,5) depth. At the existing limits these are1600001/74 and1700000/75.

Every record's nonwire validation skeleton, with its one wire member replaced
by None in a temporary shallow copy, still passes the original N/D admission.
The complete wire and record still pass the original conservative aggregate
4MiB metadata bound. Exact serialized4MiB and all numeric, header, file,
storage and memory bounds remain unchanged. The writer admits the final
record including its seal; the reader applies the private structural view
before the unchanged typed, class, reference, array and owning record checks.
No global JSON parser or other snapshot/physical metadata domain is widened.

The focused metadata module is
`tests/native/check_windowed_device_checkpoint_wire_metadata.py`. It covers
the original typed node/depth boundaries, additional JSON wrappers, both
original/current envelopes, capped nonwire metadata, sealed canonical bytes,
real local empty NPY files, and closed refusals. Its illustrative metadata
does not stand for a Native holder or successful physical restore. The real
Native class regression above stays unchanged. Root owns actual execution,
independent Source review, current runtime/source evidence and the separate
fresh write/recovery qualification; this correction alone issues none.

The stable budget identity projects the two owning opaque metadata rosters
to JSON lists, preserving every name, position and `None`. The owning record
API emits tuples; JSON encoding already represents those tuples as lists, so
canonical bytes and identity hashes remain exact. Strict metadata validation
still rejects tuples elsewhere. The focused metadata checks use the genuine
owning descriptor/record API with explicitly synthetic, unread references;
they issue no execution authority or physical checkpoint acceptance.

The Small Native snapshot preflight expects the owning Native `base` array
as signed int32 (`<i4` on the selected runtime). The Native writer preserves
that dtype; its column validator and device validator both require int32.
The former int64 header expectation rejected a genuine saved snapshot before
the first wind event. This correction changes only that expected dtype.
The twelve-member roster, shapes, all other dtypes, exact payload lengths,
uncompressed format, metadata checks and original byte/resource caps remain.

Two focused checks use the actual Native CPU snapshot writer and reader in
the closed 4x4/K1/V6/W3 shape, with the existing Native packed-column fixture.
They check exact Native header/preflight agreement and refusal of an int64
replacement of only `base`. They execute no wind or water event and establish
no GPU save/recovery acceptance. Root owns their execution and the separate
physical restart qualification on freshly bound Sources and runtime.


## GPU intent Source identity validation

The private intent wire gate uses the existing exact GPU store schema/profile
selector only when supplied the actual header by generation-chain validation.
It requires an exact dictionary and complete canonical Source-identity byte
equality with that already-authority-validated header. Unsupported canonical
values and every missing, additional or changed Source operand refuse; no
identity is normalized here. The existing subsequent intent/header environment
and ancestry comparison remains. CPU intents retain the original strict
four-field source-manifest-prototype validator, including its default call
without a header. This dispatch changes no launch authority, source-currentness
validation, schema, cap, physics or checkpoint payload. Synthetic metadata
regressions establish this private check only; actual GPU restart remains a
separate Root qualification.
