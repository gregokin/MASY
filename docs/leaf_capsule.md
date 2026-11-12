# Immutable LEAF source capsule (C4A helpers)

Bounded helpers that observe a clean committed LEAF checkout (`leaf/` + `pyproject.toml`), seal it into an immutable capsule in a namespace OUTSIDE any V1 runtime store,
and verify it independently. Not a production gate, recipe, composite, run-config shape or scientific qualification; it neither renames nor qualifies the LEAF 8da profile
(the literal 24-file manifest / 23-file package digest check remains in `leaf_source_profile`) and a self-sealed capsule claims no independent review.

* `masy.leaf_capsule` (stdlib + `native_source_gate` only; closure-safe): `verify_capsule(ns, id, limits, admit, expected_manifest_sha256=)`, `verify_capsule_dir`, `validate_namespace`.
  Checks: exact top-level/directory/file census, read-only modes, closed strict-JSON manifests (no duplicates/NaN/unknown keys, nesting cap), id = sha256(canonical identity) = directory
  name, `source-manifest.json` == identity, in-place `verify_snapshot`, pyproject vs identity. Admission precedes each body read/parse (modelled bytes, not RSS; interpreter/import costs excluded).
  Git blob ids are NOT re-derived here.
* `masy.leaf_capsule_seal` (operator side; imports `masy.runtime`/`masy.sourceobs`): `observe_leaf_scope` (required exact `expected_revision`, clean scope, no ignored/untracked/symlink/special,
  bytes == committed blobs, bounds equal the verifier's `GateLimits`), `seal_leaf_capsule` (staged copy with hash re-check, freeze, deep verify, `renameat2` NOREPLACE; existing objects verified and
  reused; failed staging retained; no repair/cleanup), `verify_leaf_capsule` (closure verifier + `_walk_object` + sha256/size/Git blob of every file).
* `tools/masy_leaf_capsule.py`: `seal` / `verify` CLI (`python -I -B`), explicit `--reserve-bytes`.

Identity: file (path, size, sha256, blob), object format, project, `tree_digest` (full tree, 24 files for 8da) and `package_digest` (`leaf/` only, 23 files; same algorithm as
`maple_syrup.provenance`). No path/commit/revision is hashed; the actual commit/tree are retained in `manifest.json` provenance.
Layout: `<ns>/objects/<id>/{manifest.json,source-manifest.json,source/}` (dirs 0555, files 0444); staging `.staging-<run>-leaf`. The namespace must satisfy `check_store_location`
(e.g. `<store>-leaf-capsules`; nested under the store, a wrapper root or an input is refused).
Tests: `tests/leaf_capsule/` (synthetic temporary Git repos; the 8da test copies the frozen reference tree into a synthetic repo and is NOT original-commit provenance).
