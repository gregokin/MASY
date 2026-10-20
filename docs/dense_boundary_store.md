# Bounded Dense accepted-boundary store prototype

This opt-in source-manifest component keeps immutable originals/native physical
checkpoints/passive images and an atomic `HEAD`. Only default small Dense NumPy
accepted boundaries are supported. The existing runner and native physics are
unchanged. Structural component checks are not actual event/restart qualification;
the actual qualification status is below.

## Status and scope (October 19)

**Accepted components.** The following are accepted: original inputs (task 34),
the passive physical MAPLE API, passive images (B43) and the C53 store. Also
accepted are actual length-1 native scalar receipts (task 64), read-only routing
capture (task 68) and three genuine restored-routing readers (task 77).

**Actual qualification.** The full task-78 qualification passed on the exact
task-77 source-manifest prototype. It ran four ordinary events with native
boundaries at 2, 169, 171 and 338 s: wind 2 s, a 167 s water event including
rain and recession, wind 2 s, and a second 167 s water event including rain and
recession. These are water-event durations, not rainfall-forcing durations. It
covered:

- cold restoration at each accepted boundary;
- re-saving held routing;
- SIGKILL before `HEAD` replacement, which kept the old parent and retried once;
- a lost acknowledgement after replacement or before acknowledgement, which
  stayed exactly-once even for later events;
- a native incomplete guard that blocked the next event, followed by cold recovery.

On restore, native clocks and cursors are preserved and the original physical
baselines are retained. Pending ledgers are not reinitialized.

**Persistent versus in-memory.** Only accepted boundaries persist: immutable
originals, native physical checkpoints, passive images, receipts and the selected
`HEAD`. Anything inside an event is in-memory and is lost on a crash. This
includes native work, staged candidates and unaccepted evidence. Recovery
restarts from the last `HEAD`. "Exactly-once" covers accepted accounting and
acknowledgement only. It does not mean CPU work is never repeated.

**Limits.**

- v1 has only the `source_manifest_prototype` runtime branch. There is no
  committed-lock branch and no default CLI checkpoint API.
- Use is explicit, bounded and local to POSIX, with locked source manifests. The
  native incomplete guard stays in place.
- Not qualified: active-event resume, power loss, GPU, windowed, dynamic
  vegetation and production schedules.
- Persisted qualification for the newest committed pair awaits publication and a
  fresh lock.
- On the coherent CPU candidate (source manifest `9aa41aaf…`, native runtime
  `4139f62a…`), the earlier fresh-child refusal at environment identity was
  measured as exactly the missing compiled `readline`/`resource` extensions.
  After a test-fixture-only `cpu_declarations` correction, the targeted store
  (2), routing (5+6) and schema (10+44) reruns and both native controls passed
  with no failures or skips, including an actual kill -9 at the intended `HEAD`
  point followed by successful recovery. The full four-event task-78 result
  above remains task-77 source-manifest evidence only.

`create_boundary_store(path, runner=..., originals=..., runtime_reference=...,
limits=BoundaryStoreLimits(), storage_contract=StorageContract(True, True))`
exclusively creates a new genesis. The calling controller declares the ordinary
local POSIX fsync assumption; observed filesystem/root/probe fields are derived.
`recover_boundary_store(path, expected_store_file_sha256=..., limits=...)` uses
the retained original inputs and runtime, with no caller case arrays. Both return
a metadata-only facade with `head_token`, `report`, `step(expected_parent)`, `close`.
There is no runner getter or implicit run. Close handles explicitly.

The runtime reference has exactly `schema`, `kind`, `source_manifest_path`,
`source_manifest_file_sha256`, `native_bundle_manifest_path`,
`native_bundle_manifest_file_sha256`, `module_roots`, `cpu_store`, `cpu_lock_id`,
`cpu_lock_file_sha256`. Schema is `masy.dense_boundary_store.v1.runtime`, kind is
`source_manifest_prototype`, and the three CPU slots are null. Module roots are
the absolute retained `src` directories for `masy`, `maple`, `maple_syrup`.
The source manifest has exactly `schema`, `source_identity`, `module_roots`,
`files`, `environment`; its schema ends in `.sources`. Files map each package to
the COMPLETE `src/`+`pyproject.toml` table of `{size_bytes,sha256}`. MASY/SYRUP
identity digests hash canonical tables; MAPLE identity is the actual native
bundle id, additionally cross-bound to that bundle's complete source file table
and `check_runtime` result. Native canonical manifest bytes are rechecked. The
retained operational/compiled environment is conservatively exact: an additional
loaded extension can refuse rather than silently broaden the runtime.

The lifetime stable-inode flock excludes cooperating writers. Each attempt runs
on an actual private passive load. Volatile runner acceptance is never reported
as durable acceptance. Native15, bridge22, original34 and passive43 remain the
only data codecs. An additional independently validated original decode supplies
the physical pack capsule without rebinding the work owner's context. Final full
readback and metadata preparation finish before HEAD replacement; no live owner
is retained in idle state. Exactly-once means selected accepted accounting and
acknowledgement, not CPU computation across crashes.

Raw/entry/disk/inode and independent memory reservations precede staging.
Maximum scope is64cells/6classes/16voxels/16000cohorts/8events, data store1GiB,
9accepted generations+4unreferenced attempts/4096entries. No old versions or
orphans are deleted or promoted. Independent maxima need not fit simultaneously.
The conservative memory reservation is at most6GiB under a separately enforced
8GiB-AS/120CPU/180wall worker. Library code does not alter caller environment.
Native event scratch/RSS and repeated-load cost require separately authorized Q.
V1 writes require the complete32KiB HEAD/128KiB generation/32KiB intent/failure
control envelope and the full1GiB/256inode free reserve. Lower read budgets fail
closed before hashing; they do not authorize an event with an incomplete write
reservation. Actual genesis call counts are three original decodes/three native
physical loads/zero saved-source graph factories for the None-routing fixture.
A non-genesis attempt reserves five decodes/eight native loads/seven graph builds;
these are bounded source-derived costs, not measured event performance.

Observed ext4/xfs and an explicit fsync declaration are required. Metadata-only
primitive probes and child-to-parent fsync establish an engineering contract;
they do not prove hardware/controller power-loss honesty. Any exception from
publication-possible onward leaves an indeterminate handle. No further step is
allowed; close/recover selected HEAD, never infer the latest directory. Missing
or corrupt HEAD refuses. Failed evidence and exact allowed hardlink leftovers
remain retained. Source/runtime/static/lease drift refuses without repair.

Source-manifest prototype is the sole runtime kind in v1. Committed binding and
public/new-run acceptance require separate qualification with new
originals/genesis. The actual four-event continuation/crash/exactly-once checks
are reported in the status section above.
No source upgrade, old artifact rebinding, active-event/production/GPU/windowed
restart, vegetation law, new reset or tolerance is introduced. Resource-v0 keeps
static prescribed vegetation, original/genesis/current soil anchors and the
unchanged native dry-reset accounting distinct; update cursor/receipts stay empty.

Original34 refuses retained routing during capture and passive43 requires it iff
accepted rain exists. Therefore structural C15 proves actual unstepped re-save
with None routing and refusal of a fabricated held genesis. The actual held-rain
re-save was later established by the task-78 run on the task-77 prototype (see
the status section). No receipt
history, physical fingerprint or native validator is forged to fill that gap.
