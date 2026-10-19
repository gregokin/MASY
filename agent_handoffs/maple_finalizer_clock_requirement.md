# Owning-MAPLE requirement: explicit absolute origin for a completed wind event

**Historical MASY-06 handoff; fulfilled upstream.** MAPLE committed origin support at `b3e8756` and retained it
in merged `39bcf79`. Historical MASY-08 checks passed on `39bcf79` + SYRUP `b4ee14e`, lock `ee190c81…`;
see [that qualification](../docs/committed_pair_qualification.md). The current accepted reference is
[MASY-09](../docs/source_pair_20261019.md): `39bcf79` + `d05d8c6` + `22aa6ab`, lock `43976b9a…`. The original
patch/request below is provenance, not an instruction to reapply it to current MAPLE.

Verified on committed MAPLE `7b0e2539f6e57b5b0182280efeadc06488fde37a`, actual SYRUP
`b4ee14e5bf37814efb0d754de17b7ea604b1cd74`, new immutable CPU lock
`d29d5740b9633aae346eba53fef05cfeb8bd6376a55b8cb7d33a8dc4417f4eef`.
This is a concrete source-backed integration requirement, not a proposed change to settling physics.

The native finalizer works at zero origin. A genuine superthreshold wind record leaves airborne material;
`finalize_wind_event` settles it with native class closure, genuine emptiness, unchanged event-end clock/cursor,
and no repeated transfer on another call. The actual nonzero-time schedule then exposes the limitation below.

## Reproduction

1. Actual MASY/MAPLE wind from0 to2 s.
2. Actual conservative SYRUP rain plus physical recession, returning DryHandoff at169 s.
3. A real absolute native WindBlock starts at169 s and runs for2 s with0.025 s outer cadence. It leaves a real
   airborne pool; native physical.time_s is171.00000000000045 s. The caller truthfully moves its one fully consumed
   block cursor to block_index1/block_offset0, as the native record loop does. No physical timestamp is changed.
4. Supplying the real2 s local record to the finalizer raises WindEventFinalizationError:
   physical.time_s171.00000000000045 is not record end2.0.
5. Supplying the actual absolute169..171 s block inside WindForcingRecordSpec raises ValueError, because
   validate_wind_forcing_record reconstructs every record cumulatively from0.

Both failures leave the input arrays unchanged. Every native binding/class/validator is genuine and unpatched.
Exact local source/lifecycle/receipt: native_finalizer_02/ under this task. Source-only reproducer is
native_finalizer_probe.py; its actual saved outputs stay local and are excluded from Claude review.

## Required contract for the MAPLE development thread

Please extend the completed-event marker accepted by finalize_wind_event to express a nonzero absolute event origin
explicitly. One narrow option is an optional keyword such as record_origin_s=0.0; MAPLE owns the final API name/design.
The record can keep its existing valid0-based relative block times. Validate actual event completion against the
explicit origin plus the record's relative end, using the existing native completion-clock semantics, while retaining
all existing cursor, forcing, physical, mobile, trajectory, budget and fluvial-boundary guards.

The origin describes the REAL absolute WindBlocks advanced by the caller. Never synthesize earlier zero-wind blocks,
pretend the rain interval was native wind history, reset/rebase physical.time_s or cohort launch/impact times, alter
committed-topography timestamps, subclass native records to change their meaning, or bypass record validation.
Placement must continue using the actual absolute physical/cohort timestamps. Successful finalization must preserve
that actual native end time bit-for-bit (including ordinary floating accumulation), cursor and last wind.

MASY will bind/hash the actual event forcing and origin, attest its real absolute-block execution and add the native
finalizer's boundary export once to the whole-event budget. Native deposited mass/ledger accumulation is already in
the returned evolving bed and must never be applied again. Exceptions/completedFalse remain blocked water handoffs.

## Acceptance for the native extension

- Existing default-zero API/tests retain their behavior; no physics/storage defaults change.
- A genuine wind -> SYRUP rain/recession -> wind sequence finalizes the second nonempty wind at its nonzero absolute
  clock, preserving all original timestamps/cursors and closing native per-class transfer/reservoir identities.
- Wrong/nonfinite/invalid origins, incomplete records/blocks, pauses and incompatible last wind still fail before
  publication. Ordinary native end-time FP discrepancies retain existing semantics; no clock is rewritten.
- Real sub-resolution airborne material is resolved; no threshold filtering, double deposition or duplicate export.
- Failure/incomplete leaves input unchanged; repeated successful finalization transfers nothing again.
- The resulting returned bed passes the actual fluvial-boundary validator and conservative SYRUP re-entry.

Neither upstream tree or branch was modified by MASY. Dense CPU only; separate windowed storage remains unqualified.
Claude's source-only design attempt hit its reported session limit (reset6:50pm America/Los_Angeles on October18), so
this requirement is independently verified by Codex but is not represented as a completed Claude design/review.

## Reviewed implementation available (MASY-06)

The narrow explicit `record_origin_s=0.0` extension and MASY adapter are implemented, independently reviewed and
verified on an isolated development pair. The live native tree is unchanged. See the
[reviewed patch and ownership handoff](maple_record_origin_handoff.md). The owning MAPLE thread must adopt/commit
that patch, followed by a new current committed source-pair qualification. No forcing prefix or clock reset is used.
