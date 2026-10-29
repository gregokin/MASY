# Imported C64 GPU accepted-boundary checkpoint API

This candidate adds an API for the existing imported 8×8/K1/V20/W3 GPU case at
accepted event boundaries. It adds no public save/resume command. Physical
qualification, general capacity, calibration, active-event recovery and backend
or device migration are not established by this Source change.

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

The separate initial adopted-HOST persist/reopen contract and protected fresh
role fixtures are still pending. They must preserve identical original raw
bindings without a second adoption or compiler replay. No fallback budget,
test factory or assumed loader is supplied by this API.
