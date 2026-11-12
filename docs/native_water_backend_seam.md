# Private backend policy seam (`masy.native_water_backend`) — S1a

Status: bounded CPU-preserving refactor; **no GPU/CuPy execution, no new source profile or pin**. The old 308/95ef profiles, CPU public signatures/defaults/records/artifacts,
physical ordering, publisher and restart contracts are unchanged. Purpose: a later CUDA sibling can reuse the current calendar implementation without cloning it.

* `CPU_V1`: the single sealed module-literal policy (private final class; no public constructor; copy/deepcopy/pickle/subclass/mutation refused). `policy_of(None)` is
  `CPU_V1`; everything that is not that exact object (copies, imposters with every method, mappings, strings such as `"cpu"`/`"cuda"`) is a typed `BackendPolicyError`
  (`policy_forged`/`policy_sealed`) raised BEFORE any I/O or use of the other arguments. No caller-selected formulas or hooks. `cuda_capability()` is an explicit
  `cuda_unimplemented` refusal.
* The session keeps the policy in the separate private handle `_backend`; `session.policy` stays the storm quiet policy; the CPU config digest, records and artifacts do not
  contain it.
* Seams (only where a backend-specific statement sat inline): operator backend tuple, imported-CuPy refusal of the namespace census, dense-preparation module/type/profile/errors
  and cold recheck, host availability reader, rainfall field namespace, physical codec selection + manifest codec profile, codec error family, codec budget mapping, writer runtime
  record, closed physical member names on the READ path (`Artifact.physical_members`, defaulted to the old tuple), codec `StormSpec`, checkpoint-limits type.
* Public `create/resume/checkpoint/plan_resume/plan_checkpoint`, `validate_source/source_facts/preparation_plan_bytes/namespace_census/storm_parts`, `nck.open_artifact/publish/
  codec_for/...` keep their exact signatures (verified by AST comparison with the frozen R3C1 baseline; the only difference is the defaulted `Artifact.physical_members` field) and
  delegate to private engines (`_create_engine`, `_resume_engine`, `_plan_resume_engine`, `_validate_source`, `_source_facts`, `_namespace_census`, `_storm_parts`, `_open_artifact`)
  that take the policy first.

## Limits / deliberately not done
* The census/namespace/distribution helpers are still called through their public CPU-bound module names inside `_validate_source` (the existing tests patch those names); a CUDA
  sibling will add its own engine call.
* The publisher body (`publish`, `_record_physical`, `_rollback*`) is untouched, so its closed physical member names are still the CPU tuple; a CUDA codec needs those names threaded
  through the descriptor-held publisher later (bounded issue for Root, not done here).
* No CUDA preparation type, device availability reader or CUDA codec exists; no new SYRUP profile (108/109) is registered.
