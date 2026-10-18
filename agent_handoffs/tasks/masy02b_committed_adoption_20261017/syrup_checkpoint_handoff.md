# Follow-up for owning SYRUP: current-MAPLE Dense checkpoints

Source-backed proposal, not an implemented or accepted fix. Do not change active upstream files or jobs based on this note alone; follow the owning repository's scoped development workflow and existing authorizations.

Verified source pair: SYRUP `2c60c73b01fe1c0cef576987d52010e84058c568`, MAPLE `5b1a052e1b9a9edfed0760152aac2a743813e79a`. SYRUP package digest `fc789e240f7b0061e9ac80bc9078b7fe42d302fc8309ee5746073455f7f9f6bb`, MAPLE `97b3df72336e6313e8d89d0b7b20784b9a0f001e30a4fc3035360e6de6068612`. Owning-SYRUP acceptance already records this limitation. MASY independently reproduced it without model execution or physical arrays.

Both native registry functions fail with:

```
ImportError: cannot import name 'CompactSedimentLedgerState' from 'maple.core.types.sediment_ledger'
```

Actual callers:

- `/home/okin/SYRUP/src/maple_syrup/checkpoint.py:179`, `_registry`, imports the compact class unconditionally alongside `SedimentLedgerState`.
- `/home/okin/SYRUP/src/maple_syrup/pond_checkpoint.py:694`, `_native_registry`, does the same.
- Current MAPLE has `SedimentLedgerState` but no compact binding. Thus the codec fails even if the supplied bed/ledger is genuinely Dense and needs no compact representation.

An owning-SYRUP follow-up should determine whether a narrow registry compatibility change can support genuine current-MAPLE Dense round trips while preserving genuine compact/windowed behavior on a capable source. Do not synthesize a missing class, coerce an unknown ledger, bypass source/storage checks, loosen codec whitelists or alter physics. Unknown/invalid representations must still refuse; actual missing-module/broken-import errors must remain visible. Preserve complete native state and all checkpoint/source identities.

Qualification should include actual Dense save/load/continuation using current committed MAPLE and existing interpreters, matching native-state/accounting evidence, invalid-binding/unknown-node refusal and genuine compact/windowed reference regressions. Capability-refusal assertions alone are not successful Dense checkpoint qualification. Source-sharing reviews remain source-only; actual model datasets/stores stay local. No installation, GPU/full-storm launch, commit or push is implied by this proposal.

Native immutable-runtime metadata qualification is a separate remaining MASY task. A source-root metadata fix does not manufacture the compact class; a codec fix does not prove provenance. After any owning-SYRUP change is committed and independently accepted, MASY must re-derive the transitive requirement from the actual new caller and qualify the new source pair. No existing checkpoint/run is automatically upgraded.

Evidence in this task: `independent_source_checks/result.json`, `adopted_syrup_cpu/{result.json,junit.xml}`, `baseline.json`, and the two owning-source excerpts in `claude_workspace/source_context/syrup_source_excerpts.md`.
