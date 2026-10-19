# MAPLE explicit completed-wind origin patch (independently reviewed)

**Historical MASY-06 handoff; fulfilled upstream.** MAPLE committed origin support at `b3e8756` and retained it
in merged `39bcf79`. Historical MASY-08 checks passed on `39bcf79` + SYRUP `b4ee14e`, lock `ee190c81…`;
see [that qualification](../docs/committed_pair_qualification.md). The current accepted reference is
[MASY-09](../docs/source_pair_20261019.md): `39bcf79` + `d05d8c6` + `22aa6ab`, lock `43976b9a…`. The original
patch/request below is provenance, not an instruction to reapply it to current MAPLE.

Baseline is local committed MAPLE main
`7b0e2539f6e57b5b0182280efeadc06488fde37a`. The live tree is unchanged.
The patch is [maple_record_origin.patch](maple_record_origin.patch) next to this handoff and passes
read-only `git apply --check` against that baseline. Its exact hash and check
receipt are in [the task archive](tasks/masy06_origin_integration_20261018/maple_patch_receipt.json). Do not apply a changed-baseline patch
without inspecting and qualifying the change.

The patch adds one keyword-only finalizer argument:

```python
result = finalize_wind_event(
    completed_state, scheduler_config, local_completed_record,
    record_origin_s=wind_event_absolute_start_s,
)
```

The ordinary forcing record stays local to zero. Completed-record validation
compares the unchanged physical end time with `origin + total_duration` using
the existing native completion epsilon. It still requires a completed native
cursor, zero offset and the record's true final wind. Invalid/nonfinite/negative
origins and unrepresentable positive intervals fail before mutation. Existing
zero-origin calls preserve their API and behavior. No deposition, kinematics,
availability, ledger, budget, boundary or fluvial validation physics changes.

MASY advances the genuine absolute block and applies the native consumed-block
cursor mapping only after that block actually completes. It calls the finalizer
once at deliberate event end and retains all original physical/cohort/commit
clock values. It adopts the returned evolving bed/availability/ledger and counts
native export once. MASY receipts retain the local record/hash and absolute
origin. Pauses, checkpoints and intermediate forcing blocks never finalize.

Independent local CPU tests of the isolated patch: 92 focused native finalizer
checks pass, including zero compatibility, absolute-origin placement, native
budgets, sub-resolution cohorts, invalid origins/cursors and zero repeated
transfer. MASY genuine nonempty wind-water/recession-wind-second-water matches a
separate direct-native sequence, with exact clocks and persistent fields.
The finalizer exception/incomplete and staged mutation tests block water while
retaining the last accepted boundary. Full native CPU/GPU suites were not rerun.

Claude independently accepted the bounded patch with no blocking findings; Codex verified the actual callers, native budgets, failure staging and exact saved code. All nonblocking review notes have dispositions in the MASY-06 task receipt.
After adoption the owning MAPLE thread should commit its source changes using
its own authorized workflow. MASY must then qualify the new current committed
MAPLE/SYRUP pairing as a new immutable lock. The isolated development bundle is
explicitly uncommitted and must not be relabelled as current committed HEAD.
This patch establishes neither full native absolute-record snapshot/restart nor
MASY combined restart. Dense CPU only; the separate windowed branch is excluded.

Reviewed patch SHA256: `7adbb4a578ae4074f7ba85e9e09da97933524ba0eb8a9c77753e5184985c7d35`.
