# Verification — MASY-03 prerequisites

Independent Codex acceptance, October17,2026 America/Los_Angeles (receipts use UTC). Source code and tests were authored by Claude with source-only tools; Codex executed checks using the existing MAPLE CPU .venv. No environment/install, GPU or collaborator-case launch occurred.

| Check | Result | Receipt under tasks/masy03_prerequisites_20261018/ |
|---|---|---|
| Initial requested MASY publication |204622d pushed; GitHub HEAD verified | ../masy02b_publication_20261017/push_receipt.json |
| Current owning-SYRUP checkpoint suites |291 passed,8 skipped | checkpoint_current_02/result.json and junit.xml |
| Genuine reference compact/windowed suites |404 passed,6 skipped | checkpoint_reference_02/result.json and junit.xml |
| Updated preflight suite before runtime tests |175 passed | runtime_units_01/verification.json |
| Expanded suite first attempt |245 run, one missing-version error; preserved | runtime_units_02/verification.json |
| Corrected complete MASY suite |245 passed, no skips/errors/failures, sources unchanged, no wrong origins | runtime_units_03/verification.json |
| Independent hashing/object/process probes | capture/seal/reuse, tamper refusal, time/output bounds, six mutation refusals | codex_runtime_source_probe_02.json; hash_integrity_probe_02.json |
| Committed corrected candidate preparation |runtime_locked, exit0 | runtime_prepare_candidate_02/stdout.json |
| Two fresh corrected-candidate verifies |runtime_verified, exit0 each | runtime_verify_candidate_02a/ and runtime_verify_candidate_02b/ |
| Both original configured upstream roots absent | two fresh verifications passed; worktrees restored | offline_runtime_acceptance.json; offline_sources_restore.json |
| Retained old lock after generator/code change |runtime_verified, no rebinding | runtime_verify_old_lock_new_parent/stdout.json |
| Locked native Dense wet checkpoints |3 genuine save/load/continuation checks passed, exact persistent-state assertions, unchanged sources/metadata/native guard, no wrong origins | locked_dense_checkpoints_01/result.json |
| Adopted actual-root preparation |runtime_locked using local MAPLE/SYRUP/main commits | runtime_prepare_main_01/stdout.json |

The current/reference suites include pure registry/refusal tests and tiny genuine native fixtures. The reference source is only a regression oracle, never a current-runtime fallback. Skips include unavailable native representation/backend paths; current Dense restart checks do not skip. The three locked checks cover legacy wet state plus pond pending and committed/rerouted states; they compare native continuation, inventories/accounting, counters and clocks with the native assertions. Their authored checkpoint arrays and raw logs remain local and are excluded from review submissions/publication.

Exact source pair: MAPLE5b1a052e1b9a9edfed0760152aac2a743813e79a (package97b3df72336e6313e8d89d0b7b20784b9a0f001e30a4fc3035360e6de6068612) and SYRUPb4ee14e5bf37814efb0d754de17b7ea604b1cd74 (package6973938042b878449a7dce5744bef7aecc72cc73adda2dc73672b8ef149dc517). The latter is committed/adopted on local SYRUP main and NOT pushed. MASY implementation commits446b2af/c8e23aa and documentation328b597 were adopted by fast-forward. All accepted product hashes match. Full source/API/native metadata/compiled identities are in the local retained locks and execution receipts;45 imported compiled/mapped files totaling401877381 bytes were recorded in the initial native qualification.

Retained locks: initiale13050d98fda6617336c0222e070c2615464ac1f4c8fd2c0524137d95cf830b6; corrected candidate98a739d53d40e89c85c1759febc80c16e82cc717605088cf7cae03827e12d5a0; adopted rootf577f90885872ae8cec1216cb01191a6ba11439f9f95201b59387fd1d58af7a9. Fresh run caches/pids remain operational receipts; source/package/selected metadata/interpreter/dependency/compiled identity stays strict. Every referenced object/version remains retained.

Final publication procedure: these handoffs are committed before the last current-HEAD checks to avoid inventing a self-referential Git revision. Read tasks/masy03_prerequisites_20261018/final_runtime_summary.json for actual final-HEAD prepare/two verifies and tasks/masy03_prerequisites_20261018/publication_receipt.json for final MASY GitHub HEAD. Those final receipts remain local/ignored; if absent or unsuccessful the publication step is incomplete. Product Python, test and pyproject bytes are unchanged after the245-test pass. The final source lock must select the current local committed pair without fallback.

Limits: runtime source/interface delivery and bounded native CPU Dense checkpoint behavior only. No alternating wind→water→wind run, shared-bed adapters, airborne completion, combined restart, GPU, coupled windowed storage, full-catchment or performance acceptance. Both CLI tools retain transport_ready=false. Source-bound storage guards and upstream physics/tolerances remain intact.
