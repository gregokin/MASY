# Explicit default CPU wind pause and resume

**Status: qualified for a bounded scope only.** Root independently accepted
qualification of this component for task 393 on this runtime composition:

- MASY `b9d8ae51016c49e074cd93526bcc84245d5e9a3e`
- MAPLE `fa2d5ef2733f84ef88005c7b06c119415ccba1df`
- SYRUP `603f8aeb0f2a87fafd556819c9c41fde98bd41b9`
- immutable CPU lock
  `84ed1f93bf084a50e0c0b0fe758573dd29761fcbd2abbf3442c3be7ca0105af0`
- Root aggregate
  `d7ae954c0d49801c32176c7b15df512157841a9449bc8c64d738629f52ca843f`

The qualification ran 37 unique tests: 35 original wind behaviour tests and 2
patched historical Source-census refusals. With the first and natural focus
reruns, this was 39 executions and 27 subtests. All 8 workers exited 0, were
reaped, and their actual host processes were confirmed gone. No test failed or
was skipped. Later documentation-only descendants do not change the qualified
runtime HEAD `b9d8ae5`.

The qualified tests cover:

- genuine pause and resume
- native commit cadence
- a four-event wind-water-wind-water sequence
- retained routing
- nonzero origins
- final settlement
- in-memory failure admission

Native bounds, clocks, physics, Source guards and resource caps are unchanged.
The two Source-fingerprint tests first execute the real, unpatched C binding
and then patch one TABLE literal. They establish refusal behaviour only. No
native compatibility is inferred from them.

This result covers the bounded static default Dense NumPy profile only.
Production reduced/topographic/creep/avalanching wind, GPU, windowed storage,
dynamic vegetation, persisted active restart, heap peaks, scale and performance
remain outside this qualification. This component has no dedicated pause CLI.

The actual committed C binding derives a supported `source_manifest_prototype`
compatibility view through the existing C Source-manifest mechanism. That view
alone is not a lock and grants no execution authority. The `committed_cpu_lock`
reference kind is still intentionally refused with `wind_runtime_kind`, and no
lock is relabelled. Ordinary `DenseRunner` calls never select pause
automatically. A matching digest does not establish Source, default, resource
or caller equivalence, or model support. The Source census names MAPLE
`fa2d5ef` (370 rows) and SYRUP `603f8ae` (65 rows). The adapter hashes complete Source tables for these revisions; those hashes
are distinct from the native bundle ID.

Only the exact static default Dense NumPy runner is supported: aerodynamic,
flat hop-distance transport, dormancy off and the original native profile.
Production, GPU/device, reduced/topographic/wake wind, windowed storage,
dynamic vegetation, persisted active work and active-process recovery are
outside M1. Water-pause qualification is separate. The previously qualified
Stage A water result (58 unique tests, 51 subtests) remains on MASY
`038ed538`/MAPLE `fa2d5ef`/SYRUP `603f8ae`; see
[dense active checkpoint](../dense_active_checkpoint.md). It is not
relabelled to `b9d8ae5`, and this wind qualification does not extend it.
Ordinary wind and water retain their original source and compatibility
requirements.

Within the qualified scope above, and under an explicit Root execution gate,
use:

```python
from masy.dense_wind_pause import pause_wind, resume_wind, abandon_wind

token_or_receipt = pause_wind(runner, runtime_reference=reference,
                             boundary_ordinal=1)
# A genuine opaque token differs from a normal accepted/stop receipt.
receipt_or_new_token = resume_wind(runner, token_or_receipt,
                                  boundary_ordinal=2)
# Resume a returned token without another ordinal to finish this event.
```

`boundary_ordinal` counts completed ORIGINAL native outer steps, not a time.
At most two unfinished boundaries may pause. Completion wins if no unfinished
boundary reaches the request. The native duration list is never regenerated,
sliced or summed as partial blocks. No pause accepts or finalizes partial
accounting. The original finalizer must genuinely settle native airborne mass
before water starts. The native six-scalar getter runs once while publishing
each pause, never on resume admission. Clocks use its real values and the
existing exact ledger/global inverse, with no forced nominal=physical end.
Only original final acceptance advances the accepted bed, event cursor,
exports, receipts and ledger.

The public single-use token has no fields, owner or native payload. Active
step/run/active-evidence inspection, competing wind/water, B/OI/C capture and store entry refuse.
Malformed passive requests preserve the current token. Once its owned native
resume is attempted, failure halts this MASY attempt; native consumption is
not guessed from exception text and work is never retried. Integrity failures
halt too. Primary categories and interrupts are preserved. A completed visible
publication wins over a lost acknowledgement. Interrupted partial publication
permanently halts the runner; this API provides no in-place recovery. An
independently accepted persistent boundary remains the recovery authority.

`abandon_wind(runner, token)` discards only private work and reports incomplete.
It never adopts airborne mass, resets water or starts the next event. Failure
retains bounded scalar last-validation/unknown-endpoint evidence, never a
native token/carry or serializable active restart.

The ordinary prologue and completion suffix are shared by mechanical
extraction, retaining duration, native conservation, commit cadence,
finalizer, error order, publication and the original26 constructor fields.
No native equation, tolerance, default or source guard changes. Full inverse
proof is required, not only matching method names.

Resource admission preserves original MASY/native limits:64cells,6classes,
16levels,16000cohort slots,4096steps and2pauses. Native64MiB backing and128MiB
read cap apply per native scan, not whole-worker peaks. A successful R-pause
block performs5+9R native scans and2+3R source passes, R<=2. Genuine C runtime
binding runs2+2R times. Accepted-state checks, bounded class-vector/metadata
and scalar-status hashing, parsed metadata, original/current owners, natural
commit and validation scratch are separately charged. No new per-step payload
observer, full-volume rollback/history or device transfer is introduced. The
reservation is admission DATA, never measured fit. Separately bounded Root
process/AS/RSS/CPU/wall/output/disk/source/compiled/cleanup controls remain
mandatory. Authored fixtures are synthetic interface checks, not production
calibration, GPU/windowed/restart/performance/scale qualification.
