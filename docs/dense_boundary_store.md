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

- Runtime selection is explicit. The isolated task-91 candidate adds an opt-in
  `committed_cpu_lock` branch; its new committed positive path is not qualified.
  There is no default CLI checkpoint API.
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

Committed binding and public/new-run acceptance require separate qualification with new
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

## Opt-in committed CPU binding (task-91 candidate)

`committed_runtime_reference(cpu_store=..., cpu_lock_id=...,
cpu_lock_file_sha256=..., lock=...)` is a pure metadata constructor. It validates
the existing CPU lock, its id and canonical raw SHA, and derives the six-entry
sealed import prefix, module roots and native bundle manifest. It performs no
filesystem observation, hashing of files, latest lookup or lock publication.
The returned reference retains exactly the ten v1 keys listed above, sets kind
to `committed_cpu_lock`, fills all three CPU slots and sets both
`source_manifest_*` slots to null. The reference must be retained unchanged.

The new branch reads that exact immutable lock and reuses
`runtime.load_lock_ref`, `runtime.verify_source_object`, the native
`dependency_bundle.verify_bundle` / `check_runtime`, and the real selected
distribution resolver. The four source summary fields and all three selected
overlay maps/bytes are bound to the lock. Native source rows, package digest,
tool SHA, archive SHA and both manifest SHAs are compared to their distinct lock
slots. The current interpreter/venv, sealed bootstrap literal, import prefix,
isolation flags and stable process record must match the lock. Accelerators
must be explicitly hidden. The unchanged C environment observer additionally
persists the current compiled-extension context; the lock's earlier verification
extension list is not silently treated as the current process's complete list.

The A/B `source_identity` kind remains the legacy `source_manifest_prototype`
**digest format**, with exactly four fields. In the committed branch, provenance
comes independently from the retained `committed_cpu_lock` reference. MASY and
SYRUP tuple values still hash the complete `src/` plus `pyproject.toml` tables;
MAPLE remains the exact native bundle id. The MASY launcher is verified as part
of the complete source object but is excluded from the existing tuple table.
No old original inputs, genesis, store or reference is relabelled or migrated.

Admission precedes large file verification. Lock/native control files are at
most min(the caller's runtime metadata budget, 4 MiB); each source manifest is
at most 1 MiB. Each source has at most 2,000 files, depth 16, and 16 MiB per
file. Generated source overlays have at most 64 files and 1 MiB per file;
the three-source-plus-overlay total is 128 MiB. Native limits are explicitly
4,000 files, 16 MiB per file, 64 MiB total and 64 KiB origin metadata, with the
native archive ceiling derived by its owning helper. Its small distribution
metadata supplies the owning helper's path-dependent editable overlay. Every
referenced sealed file must be a read-only, single-link regular file of the
exact declared size. The held inventory has fixed ceilings of 10,000 files,
512 parent descriptors and 512 MiB of distinct referenced artifact bytes.
Interpreter and compiled libraries retain their separate owning contracts,
including the interpreter's allowed final symlink; they are not sealed sources.

Binding creates no files, child processes, materializations, caches or pycache.
Existing referenced storage is not new disk consumption. It does repeatedly
read source, archive, materialized package, overlays and environment binaries:
the native archive has separate hash/member passes, and materialized verification
and the final native resolver repeat package reads. The unchanged environment
observer can scan 4,000 compiled entries and 4 GiB, plus executable and NumPy
observations. Completed create/recover/accepted-step paths each bind twice;
duplicate acknowledgement, incomplete event and completed schedule paths bind
once. These are IO costs, not peak-heap estimates. A separately admitted bounded
controller must establish memory, read/wall, disk/inode and fresh headroom
budgets before execution; the library does not raise process resource limits.

Held no-follow parents, full file-stat identities, and closing control rereads
detect ordinary persistent drift under the quiescent immutable-owner contract.
Delegated walks/reads remain path-based. This is not whole-tree atomic security
against a hostile concurrent writer, and closing six manifests alone does not
prove that all source or overlay races were impossible. No repair or fallback
is attempted after refusal.

Task-91 protocol tests cover metadata construction, admission, refusals and
explicitly labelled unit delegate wiring without a bed/model/child/device run.
They cannot establish genuine positive provenance for the new C module against
the old committed lock: its origin must refuse. Root source acceptance and
bounded protocols precede publication; a new committed module/lock, fresh CPU
qualification and fresh original/genesis/store identities are required before
cold positive component and four-event durable-restart qualification. The
historical task-78 result above remains evidence for its exact prototype only.

## Namespace-local mount ID (task-111 correction)

Linux mount IDs are local to a process's mount namespace, so a genuine store
could refuse cold recovery in a fresh namespace although its filesystem was
unchanged. Saved and current mount observations must still be exactly
`{mount_id, mount_path, major_minor, fs_type}` with a positive plain-integer
`mount_id` and string fields. The ID is retained in the probe receipt only as
diagnosis. Recovery compares `mount_path`, `major_minor` and `fs_type`. The
persisted probe raw SHA-256, root path/dev/ino, `f_fsid`, ext4/xfs, the
longest-enclosing-mount/device check and every other guard are unchanged. This
correction does not qualify cross-process restart and does not migrate old
artifacts. That still needs new committed source, a new lock and fresh
originals/genesis. There is still no power-loss guarantee.
