# Committed native water–LEAF runtime (opt-in v2)

The actual committed C1 runtime completed the public fixed-bed example on
NumPy, Numba and CUDA, including two chained same-device restarts. See
[the bounded qualification](phase1_release_qualification.md) for code commits,
case, checks and limits. This is not qualification of wind-with-persistent-soil,
moving beds, field calibration or large-domain capacity.

Use `schema: masy.native_water_run.v2` and `source.mode: production_committed`.
The v1 water-only schema retains its explicit production refusal. Existing
development snapshots and historical reference-bundle guards remain separate.

## Resolution and source checks

The parent admits and reads the closed configuration, recipe and selected
review receipt, then checks their hashes and join. It verifies the canonical
V1 runtime lock, committed source/object/package identities, actual native
bundle manifest, sealed MASY/SYRUP objects, immutable LEAF capsule, exact
interpreter/virtual-environment identity, release tree and running parent code.
It invokes the unchanged V1 `tools/masy_runtime.py verify` in a bounded child.
The worker independently re-derives the same binding before physical work.

The genuine native source profile continues to check MAPLE commit, archive,
project, package, distribution metadata, runtime, imported modules and actual
classes. A fresh export can have a new delivery id because operational origin
and tool/path metadata are part of that identity. Production therefore uses
the explicit `verify_profile_delivery` and `delivery_source_fence` interfaces.
Neither is an arbitrary id override or a fallback. The original
`verify_profile` and `SourceFence` constructor retain their reference behavior.

The versioned delivery record joins the lock, recipe, bundle manifest, roots,
device and LEAF profile. Its path-free core is included in the parent identity
before that identity is hashed; the outer record then carries the resulting
digest. There is no self-hash cycle. The source fence and restart compare the
core; each invocation separately verifies its current operational binding.
Old artifacts cannot be silently rebound to a new delivery.

Metadata-only source facts perform no file reads. Actual delivery-manifest
bytes are verified during cold source admission using a single hash chunk;
malformed/special files, changed bytes and insufficient memory are refused.
No cached manifest becomes a substitute for verification.

## Inputs and workflow

The source block contains `profile`, `recipe`, `recipe_sha256`, `receipt`,
`receipt_sha256`, `production_store`, `production_lock`, `release_masy_root`
and `leaf_capsule_namespace`. The LEAF root comes from its verified capsule;
an additional caller-supplied `leaf_root` is refused in production.

1. Select the exact qualified code commits and a supported MAPLE environment. Create clean detached worktrees at the exact code commits: SYRUP `e40a1b8`, LEAF `52f39fe`, MASY `a704002`, MAPLE `9f466d2`. After documentation commits, the branch tip is not the code `expected_revision`. Run the launcher from that detached C1 MASY worktree.
2. Use `python -I -B tools/masy_runtime.py prepare --config RUNTIME.toml` to
   create an actual runtime lock. Point its distribution metadata at the exact
   qualified native9f metadata: the companion manual archive's `examples/native9f_metadata/distribution/` folder, kept outside the detached code trees; do not assume
   newly generated pip metadata is byte-identical. The source guards remain.
3. Seal the clean committed LEAF source with `tools/masy_leaf_capsule.py`.
4. Derive a recipe with `tools/masy_production.py draft-recipe` from the actual
   lock and capsule, then select its independently reviewed receipt by hash.
5. Use `render-config` to copy the public example into a fresh run directory.
   Match the source profile, water-file backend and explicit CUDA controls.
6. Run `python -B -m masy.native_water_run run CONFIG`. Use the documented
   occurrence-boundary pause/resume options for combined water/biology restart.

The helper's `draft-receipt` produces an unreviewed draft that is refused.
The published qualification recipes describe the actual experiments; a new
machine/path can produce different operational ids and requires its own
qualified recipe selection. Do not edit or rebind the old records to fit it.

## Interpretation

Hashes and commits establish byte consistency. Independent scientific review
is attested by the selected published review record; the code cannot establish
review independence from a JSON Boolean. Summaries retain
`production_qualified: false` and report `production_runtime_verified: true`.
The unchanged generic V1 runtime tool still makes no transport-readiness claim;
the separate opt-in v2 experiment is qualified only to the scope above.

Named memory admissions exclude interpreter/JIT/allocator overhead and are not
RSS measurements. The microsecond LEAF dated-clock limit remains; Phase 1.5
will introduce a separate nominal-calendar capability, not migration of these
locks. Wind integration, evolving-soil remapping and observational validation
retain their separate development requirements.
