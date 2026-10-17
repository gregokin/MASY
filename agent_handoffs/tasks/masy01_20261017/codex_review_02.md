# Codex review of correction_01 — original findings closed; one new source finding

Exact corrected version: `correction_01_manifest.json`, read-only correction_01_snapshot/. Claude stage exited 0/reaped. Codex ran **143 tests + 101 subtests (pass)**, syntax and isolated --help (pass), and the independent first-review regression script (all original cases pass). See verification_02/ and `retest_review.py`. Original findings 1–10 are CLOSED, subject to preserving them in this narrow final change.

Current-source read-only audit observes MAPLE **5b1a052e1b9a9edfed0760152aac2a743813e79a** and SYRUP **9fabdca60bd9cd7c5b358d5cbf620dbd3209d93a**, with unchanged source observations before/after. Independent MAPLE work committed its previously dirty changes; the package/pyproject scope is clean. No physical run was upgraded or rebound (none exists). The audit currently reproduces the unmodified native source_root=None refusal while the native 33-item API list and the current MASY list pass. That pass misses the dependency below.

## Finding 11 — Medium, OPEN: required Dense type dependency missing from MASY audit

Actual source: SYRUP sediment_bed.py:610 imports `maple.core.types.voxel.WindowedVoxelColumnState` unconditionally inside `is_windowed_bed`, before touching the column. The committed MAPLE voxel module does not define it. This affects Dense inventory, demand and already-committed paths too. The relevant voxel and sediment_bed source hashes are identical between the initially archived pair and the current clean pair, so the source finding applies to both. Codex reproduced the actual ImportError on explicitly identified archived source copies with a plain object, before any bed access; no model state, arrays or transport loaded. Exact source origins/hashes/error: `independent_dense_type_import_probe.json` and `dense_compatibility_finding.md`.

This is different from the OPTIONAL paired-windowed TRANSACTION module. Its absence stays informational for Dense. The type imported by all existing bed paths is a required interface of the currently observed SYRUP implementation, despite its name.

Required change (scope is MASY only):

1. Add the genuine MAPLE Dense class and the currently required Windowed type to the explicit interface audit with actual CLASS/type validation (a callable function is not a valid isinstance target). Extend Requirement minimally if needed; do not construct/synthesize a missing class, invoke transport, monkeypatch real modules, or weaken any guard. Add the actual Dense inventory/already-committed helper dependencies if absent, using precise existing source signatures, not guesses.
2. Record the missing Windowed type as concrete API incompatibility and explain its Dense-path caller. The overall current audit still refuses because the native metadata-origin guard also fails; retain BOTH pieces of evidence. Native SYRUP check_required_api may still pass because its upstream list is incomplete. Do not label the full Windowed storage path required or supported.
3. Document that this transitive type dependency is tied to the observed current SYRUP caller. A later independently reviewed owning-SYRUP fix may remove it; a newly qualified MASY requirement update then reflects that actual source change. This is not permission to bypass private/deep-storage/source guards.
4. Add focused synthetic API tests: missing type, present None/function (invalid type), genuine class, and distinction from missing optional transaction module. No actual bed/solver construction.
5. Update the source-backed audit and proposal documentation. The accepted owning-SYRUP design proposal is in `upstream_proposal_03/result.md`, with Codex review/acceptance in `upstream_proposal_acceptance.md`. It is only a proposal; no upstream code or environment changes are in this task. Do not copy proposed classifier code into MASY's runtime.

## Small documentation consistency changes

- Add updated staged `src/masy/README.md` and `tests/README.md` in the already allowed directories; the live placeholders say nothing executable exists and will be stale on publication. Preserve their accurate physical/restart/performance limits.
- Stage the original `docs/upstream_interface_audit.md` from source_context and append an explicitly dated current MASY-01 finding: new committed MAPLE HEAD, unchanged current SYRUP, clean current runtime, native provenance failure and confirmed Dense type-import mismatch. Preserve the original Oct 16 observations/history; do not rewrite them as current acceptance.
- Existing docs should state the observed sequence accurately: initially clean pair → unrelated dirty edit correctly refused → external commit 5b1a052 and clean scope. Avoid evergreen claims that the live tree is necessarily dirty or that all API checks prove Dense compatibility.
- Source choice for a later actual run is pending with the user under AGENTS.md (committed sources vs explicitly labelled development snapshots). Continue the read-only task independently; never infer dirty/patch adoption or physical launch from this review.

All original corrections remain required and independently verified. Update the implementation report with finding 11 and intended commands, state tests NOT RUN by you. No new packager, state adapter, model/GPU run, installation, upstream edit or commit. The final API list/report must tell the truth about the source pair; standalone preflight remains not transport-ready.
