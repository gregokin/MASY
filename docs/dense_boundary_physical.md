# Dense physical bridge prototype

`masy.dense_checkpoint` connects an existing accepted Dense bed to the
parent-accepted native MAPLE exact physical codec. It preserves the actual
pending ledger, original committed arrays/clock/count, fractional availability,
water, prescribed vegetation/modifiers, empty native wind pool and exact time.
There is one current sediment authority. No bed is reconstructed from its DEM,
initial composition or original case.

```python
from masy.dense_checkpoint import (
    DenseBoundaryInputs, pack_boundary_physical, load_boundary_physical,
)
from maple.aeolian.scheduler.physical_checkpoint import save_exact_physical_checkpoint

original = DenseBoundaryInputs.capture(bed)
packed = pack_boundary_physical(
    bed, empty_native_pool, native_wind_config, caller_identity,
    inputs=original, max_bytes=physical_byte_cap,
)
save_exact_physical_checkpoint(
    new_directory, packed.physical, native_wind_config, packed.identity,
    max_bytes=physical_byte_cap,
)
restored = load_boundary_physical(
    new_directory, native_wind_config, packed.identity,
    inputs=original, max_bytes=physical_byte_cap,
)
bed, pool = restored.bed, restored.mobile_state
```

Saving remains the native codec's responsibility; this bridge adds no file,
manifest, HEAD or publication authority. The caller holds quiescent ownership
across pack/save and retains the returned identity and complete original input
capsule. Capture copies metadata mappings but retains original readonly initial
arrays and native context objects. Every operation independently rehashes the
complete capsule, including all eleven actual `BedContext` fields, and checks
closed declared/instance inventories. Cached digests are not substitutes for
content validation. Changed aliases, missing defaults, extra fields and changed
case/configuration identities refuse. Native source binding is integrity and
compatibility evidence, not an authenticity signature against a hostile writer.

The capsule can be supplied from the same typed original artifact objects; no
generic static-input decoder is provided. It includes the original case,
held initial input arrays, native context, initialization receipt, bed limits and
prescribed identities. Initialization metadata remains initial even when the
saved bed has evolved. A self-contained combined checkpoint must separately
retain those original immutable artifacts; the native physical payload does not
serialize this capsule or the full configuration.

There is no public raw-carrier restore constructor. The bridge immediately
hands the actual native loader's fresh return to the private
`DenseBed._from_native_boundary_load` constructor in a private exact-layout
typed ticket. The ticket is neither exposed by the public API nor persisted;
no registry, boolean bypass or new persistence format is added. Raw carriers,
including readonly views over a live source bed, are unsupported and refuse.
This is an internal Python ownership protocol, not security or provenance proof
against access to private module internals, function replacement or reflection.

The private constructor checks every physical array's readonly top flag through
the complete bounded traversal. Native15 can return readonly views over newly
allocated private writable base arrays; owning-only or base-chain-readonly
requirements incorrectly reject those genuine native loads. The bridge changes
no flags and copies no physical arrays. Callers keep loaded buffers quiescent,
never re-enable writes and use independent staging for subsequent mutation.
The constructor uses the existing assignment-only private holder, then checks
original static inputs and native state before returning. It never invokes
fresh-bed/scheduler/runner initialization, deposition, commit, avalanche, reset,
finalization, rerouting or transport. Loaded native arrays remain readonly;
the normal bounded stage-copy path produces independent work storage.

The supported profile is exact canonical Dense NumPy and the native original
context defaults (`constant_depth`, no exchange-history/window options). Config
and context geometry/classes/specs/mass resolution/availability values must
match. Non-None accounting baselines and trajectory-continuation claims,
nonempty wind history or water-borne sediment, foreign representations/backends,
and missing prescribed/shared fields refuse. The native exact codec can support
states outside this bridge profile; none are silently dropped to fit `BedState`.

A positive plain-int `max_bytes` bounds a conservative physical/static array
footprint plus metadata allowance before native validation/expansion. Nested
arrays in mappings and sequences count, and aliases count each occurrence.
Metadata has explicit depth, node and conservative encoded-size ceilings before
copying or hashing. Restore reserves static inputs/configuration/identity first
and gives only the remaining budget to the actual native loader. The native
codec separately bounds its serialized archive and metadata. This does not
establish peak-RSS or large/windowed/GPU readiness. Private native validation and
fingerprint helpers are explicit source-bound dependencies and require separate
qualification if the native source changes.

This task uses a complete immutable scoped development manifest: committed
MASY22aa plus this explicit candidate delta, MAPLE39 plus the accepted exact
codec module, and committed SYRUPd05. Actual SYRUP bundle export/materialization,
genuine previously source-built metadata and `-I -B` import-origin checks are
used. The initially stale installed metadata was rejected and left unchanged.
This is a **source-manifest prototype**, not a genuine MASY source lock or a new
committed-pair qualification. Fresh whole-pair qualification follows parent
publication before combined integration acceptance.

Tests use tiny authored native fixtures, including real completed wind
finalization with a pending ledger. A separate initial-state test preserves a
nonzero awkward epoch, prescribed cover/crust and surface water without a reset.
That demonstrates physical preservation, not certification of water-event
completion or permission for a following event. No cursor, totals, clock ledger,
routing graph, soil resource, receipts, active process or event-acceptance HEAD is
restored here. Persistent boundary transactions and active-event continuation
remain separate milestones. No ecological law or dry-reset policy is changed.
