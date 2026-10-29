# Imported C64 GPU accepted-boundary checkpoint API

The implemented imported-case API saves and recovers accepted event boundaries
for the closed 8×8/K1/V20/W3 GPU profile. Its separate four-role physical
qualification passed on October 29, 2026, for the exact subject below. It adds
no public save/resume command; active-event recovery, migration, broader
capacity and production calibration remain outside this interface.

`masy.case_device_c64.create_device_c64_case_boundary_store` accepts `path` and
the keyword arguments `prepared_case`, `qualification_budget`, `bridge_limits`,
`native_source_refs`, `limits` and `storage_contract`. `prepared_case` must be
the genuine, verified result of `prepare_device_c64_case`, before its first
event. The exact CaseBudget must also be the prepared case's budget. The store
must remain inside that budget's genuine bridge run directory. Creation
captures the actual owning request, HOST and prepared-water raw records and
hands the guarded stepping capability to the store. A handed-off prepared case
cannot verify or step again; that latch is distinct from its input-failure flag.

`recover_device_c64_case_boundary_store` accepts `path` and the keyword arguments
`expected_store_file_sha256`, `qualification_budget`, `bridge_limits`,
`native_source_refs` and `limits`. It recovers the selected accepted HEAD using
the existing store handle and its parent token, `step`, report and close
protocol. It does not compile the case, adopt another bed, replay genesis or
reconstruct the evolving bed from the original DEM. The genuine Native snapshot
loader and existing bounded validators restore the saved seven-field bed and
referenced immutable deep versions. Native soil state, all five prescribed
vegetation fields, routing source/cache state, clocks, raw receipts and the
existing nine logical water counters retain their original restore checks.

Root must provide fresh execution-role authority, a genuine CaseBudget, the
ordered 22 SYRUP Source references and the separate ordered two Native E1/prefix
references. Role paths and evidence may change; original request/case/water,
Source size/SHA identities, physical UUID/backend, exact scientific controls,
full bridge limits including run directory, and all budget ceilings must match.
Equal descriptor bytes do not prove fresh launch authority. Old receipts and
original role evidence are never rebound to the new role. Duplicate parent
tokens retain the existing return-before-transport behavior.

The original three raw records are preserved as size/SHA-bound canonical hex
chunks using the existing scalar string domain. Recovery validates their full
identity joins before the Native snapshot loader. A checked same-PID/thread
restore ticket authorizes one producer admission after the real restored state
and context exist and before the real bed constructor. Failed attempts poison
their ticket; a consumed admission cannot be minted again.

The factory-only API remains separate. Both entry directions refuse the other
origin kind before loading an owner. Existing saved stores retain their exact
old Source locks and cannot be upgraded in place. A newly edited runtime needs
its own factory and imported-path verification; earlier acceptance is evidence
for the original locked runtime only.

All existing metadata, serialized-byte, logical-work, Native memory and storage
limits remain in force. The original caller HOST forest, original placed
owners, work/readback owners, raw/hex/parser views, Source code and first-error
banks are additive. Derived comparison DOMs and paired encoding images retain
the existing positive metadata charge; repacked hex and canonical re-encoding
buffers are explicit additional occurrences. No release, garbage-collection
or alias credit is taken.
These are positive known-subset charges; constructor and external-provider
allocation terms remain opaque. The original 600-second operation deadline,
480/540 CPU limits, 2 GiB pool, 5 GiB RSS, 9 GiB driver growth, 512 MiB owned
storage, 64 MiB files, 4 MiB streams and Native 0.5 safety fraction are unchanged.

## Initial adoption and fresh roles

The initial-input owner now supplies `save_initial_windowed_case` and
`reopen_initial_windowed_case`; see [the HOST contract](case-windowed-inputs.md#reopen-an-initial-imported-c64-adoption).
Compile once, then genuinely load the authenticated compiled package before
the first adoption, water preparation and original request. Compiler and load
provenance are different subjects and remain separately recorded. Native's
package-configuration reload correction preserves the authenticated authored
model settings when loading; it does not replace them with defaults or change
numerical laws.

Saving the passive initial HOST owner publishes a bounded six-file capsule.
A later writer already owns a genuine fresh load of the same compiled case;
reopen itself neither loads Dense inputs nor compiles/adopts/regroups. It
selects the same retained initial Native manifest at the same canonical deep
and bridge paths and verifies all saved fields and aliases. The original raw
HOST, four prepared-water records and own-encoded device request must match
before fresh owning CaseBudget issuance. This capsule reopens the initial
input only. Evolving accepted-state recovery uses the checkpoint store above.
Retain every referenced old deep version; do not reset HEAD or delete branches.

The corrected retained read-only occurrence inventory remains part of the
original restore validation and positive known-owner accounting. It does not
relax read-only/currentness/alias checks or grant release credit. Constructor,
allocator and external-provider terms remain opaque.

## Qualification status and exact subject

The imported type-graph gate and all four serial GPU roles passed on MASY
`5784565605074c7d08490588e12181a49c36d897`, MAPLE
`a88f71cc924466440a4633c24c487ac631308f4e` and SYRUP
`61dcf61de14ba302148a4fb1bfdbe08ddf4df10c`, under immutable runtime profile
`2d3117dc14a370fff7ba5a6568cff2aaa411daf207375829541cc214deb67632`.

| Role | Worker elapsed time |
| --- | ---: |
| Uninterrupted four-event reference | 237.997 s |
| Save after first wind | 122.165 s |
| Recover and complete first water | 186.181 s |
| Recover and finish second wind/water | 290.046 s |

All five accepted scientific boundaries matched the uninterrupted reference
exactly, including resident/deep sediment, prescribed vegetation, soil,
routing, clocks and accounting. Complete saved receipt prefixes and all nine
water counters survived recovery. A duplicate acknowledgement changed neither
HEAD, receipt, transfer counters nor restore diagnostics; it replayed no
transport or event accounting. Referenced old deep versions remain retained.
Root independently checked Source/runtime currentness and owned-process
closure; the selected GPU returned idle. The original execution limits and
Native numerical checks stayed unchanged. Sampled resource measurements do
not establish allocator peaks or capacity for another landscape.

Final acceptance receipt SHA-256:
`51fca7ebf5b0c9090f21b3d58e9b34f61083ca9d080c95c6b50c4ee822d16b23`.

This accepts the Python store API for same-Source/backend/physical-UUID
recovery at accepted event boundaries. The historical factory and public
command results retain their original locked Sources. Public CLI resume,
active-event recovery, backend/device/Source migration, broader dimensions or
class counts and production calibration remain separate work.
