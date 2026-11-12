# Committed production runtime (opt-in, run config v2) — C4B

`source.mode: production_committed` under `schema: masy.native_water_run.v2` (the LEAF joint run). v1 production stays a typed refusal (`production_recipe_unavailable`); development snapshots, v1 parser/identity/closures,
literal science profiles (AFCA / Leaf 8da / native9f) and every physics module are unchanged. Not a qualification: this records what the code does and what still needs real C1 sources.

## What runs (parent, in this order; every read/hash/parse/spawn is admitted first)
1. bounded config read; closed v2 validation (no reference opened); availability vs `caller_reserve + buffer` (a zero cap refuses here).
2. `masy.production_runtime.resolve(role="parent")`: recipe + receipt (sha pins, receipt join, `root_reviewed`/`independent`), V1 lock (canonical bytes, id), recipe↔lock (heads, object ids, package digests, bundle id, launcher),
   sealed MASY/SYRUP objects (exact entries/dirs/modes/bytes/sizes), the immutable LEAF capsule (`leaf_capsule.verify_capsule`), interpreter == lock interpreter, release MASY worktree bytes == sealed MASY object,
   running parent modules == sealed MASY files, then the **genuine existing V1 `tools/masy_runtime.py verify`** in a bounded child (wall time, output cap), whose closed success output is cross-checked.
3. the worker re-derives all of the above itself (no second V1 child) and must reach the parent's production identity hash; then `resolved_config` completes the sealed config (sealed roots, capsule-derived `leaf_root`)
   and the unchanged worker flow runs its cold census/namespace/distribution/native-bundle/`verify_profile` checks against the sealed roots.

## Records (closed, versioned)
Recipe `masy.production_recipe.v1` (portable ids only, no lock id/path/own commit), receipt `masy.production_release_receipt.v1`, identity `masy.native_water_run_bootstrap.v2`; see `plan.md`. Config keys:
`recipe, recipe_sha256, receipt, receipt_sha256, production_store, production_lock, release_masy_root, leaf_capsule_namespace`; the `leaf` block has no `leaf_root`.

## Trust model
Hashes/ids/commits = byte consistency. Independent acceptance = the published Root-reviewed receipt the operator selects by sha256; code checks its join/status, not its independence. A self-generated recipe+receipt+shas is
consistent, not reviewed. Summaries keep `production_qualified: false` and add `production_runtime_verified: true`.

## Publication / Root steps (nothing here hard-codes a commit)
C1 commits (Root, scoped) -> real V1 `prepare` at the exact commits -> `tools/masy_leaf_capsule.py seal` -> `tools/masy_production.py draft-recipe` (derived from the real lock/capsule) -> Root review/CPU+GPU
positive+negative checks -> receipt -> C2 metadata commit pinning the actual ids -> `render-config` -> run.

## Limits
Memory admissions are modelled estimates, not RSS; the V1 verify child imports the native stack (excluded fixed cost). LEAF is verified as an immutable capsule, MAPLE through the V1 child plus the worker's cold bundle check
(the stdlib parent does not re-hash the MAPLE object). The real end-to-end positive (real prepare/verify/worker, CPU and CUDA) is deferred to Root on actual C1 sources.

## Verified delivery (C4C)
A fresh `prepare`/export of the SAME pinned 9f source yields a different bundle id (it also hashes the export path/tool) while archive/project/package/distribution content is identical (Root's probe: reference `55d1485a…` vs `ca688738…`). The historical `leaf_source_profile.verify_profile` keeps its mandatory
reference-id gate EXACTLY (it refuses the new delivery, `native_profile`). Production uses the explicit sibling `verify_profile_delivery(delivery=…)`: the same cold engine, with the bundle gate replaced by a closed `masy.native_delivery.v1` record built by `production_runtime.resolve` from the verified V1 lock and the
Root-selected recipe (new keys `bundle_manifest_sha256`, `archive_sha256`) and bound into the production identity; the ACTUAL native verification (genuine `ncp.verify_source_profile`) must match its bundle id, the bytes of the materialized `bundle-manifest.json`, the literal archive sha256 and every root/profile/device/variant. No flag disables the old gate,
the sibling refuses a missing delivery, development refuses a delivery, production requires one. Water side (Root-approved extension): `native_water_calendar.SourceFence` gains an explicit closed `delivery` member (default None keeps the historical constructor/record/identity and the reference-bundle literal EXACTLY). With a delivery the fence validates and joins the record
(bundle id, target, roots, device), `_source_facts` replaces the reference literal by the literal archive sha256, the ACTUAL materialized bundle-manifest bytes and the genuine preparation's bundle id, `record()/identity()` add the path-free delivery CORE, and the combined checkpoint's
`source` section carries the core as its only optional member: an old (reference) checkpoint cannot be resumed under a delivery fence and a delivery checkpoint cannot be resumed under a reference/different-delivery fence (`source_mismatch`).
Noncircular algorithm: (1) CORE {schema, bundle_id, manifest_sha256, archive_sha256, device, leaf_profile, leaf_variant}; (2) the production identity includes the CORE; (3) identity_sha = hash(identity); (4) the outer record = CORE + paths + lock id + identity_sha. Only the CORE is compared at restart.
`native_water_leaf_run.verify_receipt` keeps its four-argument API (refuses production); production uses `verify_receipt_production`.
