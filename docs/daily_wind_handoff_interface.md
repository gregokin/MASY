# Wind into the live native water lifecycle

`masy.daily_wind_handoff.prepare_wind_handoff` connects a parsed daily scenario's wind rows to SYRUP's current
floating-water lineage. MAPLE runs the actual wind; SYRUP stages and adopts the changed bed, refreshes soil properties,
performs its approved capacity reconciliation and carries current layer water and Green–Ampt history forward.
MASY owns the ordered wind cursor and exact accumulated wind exports/time. No water physics is copied into MASY.

This is an explicit **Dense NumPy CPU API profile**, `masy.daily_wind_handoff.cpu.v1`. It uses MAPLE commit
`9f466d2cfcd52991b24496cf67f4562d6268afa9` and the separately qualified SYRUP Phase 2 plus wind-handoff source.
The factory checks the literal package digests in `daily_wind.py` and `daily_wind_handoff.py`; it never adopts a newer
package automatically. This qualification deliberately uses that immutable MAPLE pair, not the newer live MAPLE main.
Historical N15 factories, the water-only CLI's wind refusal, locks and default behavior retain their existing contracts.

## Calls and authority

```python
bridge = prepare_wind_handoff(
    daily_plan, verified_source, admitted_case, bed_context,
    wind_limits=wind_limits,
    handoff_limits=native_wind_handoff_limits,
    bridge_limits=WindBridgeLimits(max_receipt_bytes=..., max_record_bytes=...),
    available_bytes_fn=actual_host_available_bytes,
)
boundary = bridge.start(current_water_owner, complete_bed, native_event_factory)
boundary = bridge.execute(daily_plan.scenario.occurrences[0])
receipt = bridge.last_receipt()
```

All limits are explicit and typed. The source/admission/context and the static event factory are real owning-model
objects, supplied by the caller. The initializer creates a real accepted boundary before the first wind without a
fictional storm or daily update. It retires the original water owner only after native and coordinator preparation.
The bridge retains one native lineage; `bridge.lineage.head` is always the authoritative current water boundary.

For each nominal date, execute its wind rows in CSV order before rain and the interevent day. Reusing one file on
multiple rows is supported; occurrences have distinct identities. Between wind calls the caller advances that same
lineage through public SYRUP rain/recession/day APIs. `tests/wind_handoff_support.py::run_native_day` is an executable
test driver showing those calls. The adapter itself does not implement daily water or plant demand/growth.

Wind accepts a completed boundary with quiet stored surface water, using the documented dry-soil wind threshold
approximation regardless of real moisture. Surface water and current W/J/I remain with SYRUP. Wind runs no E/T,
growth or subsurface time step; the native continuation constructor carries the physical clock to the actual MAPLE
wind end. Nominal day progress does not change. Actual event/recession time is additional to nominal calendar days,
so accumulated physical time can exceed the calendar span. There is no biological catch-up.

The native handoff preserves the actual routing graph and source. It compares all active bed rows in bounded batches,
including composition-only changes, and reuses Phase 2's selected soil refresh and reconciliation. A surface return
marks pending recession. Real native water completion must then occur before another wind or daily update; neither
the adapter nor a time/resource limit can declare completion. The owning routing rule governs later graph refresh.

## Transaction and restart interface

The wind executes on a bounded staged Dense bed. All source, accounting, water-carry, native validity, integrity and
resource checks complete before adoption. Failure retains the accepted parent and unchanged wind cursor. If an
interrupt arrives immediately after native publication, the coordinator recovers its prepared cursor/receipt from the
candidate's weak accepted-boundary reference. No wind workspace, mobile pool or series of old beds remains retained.

`wind_accounting()` accumulates actual native per-class wind exports/inputs and inventory bounds as exact fractions.
The water coordinator must add the separate genuine water sediment receipts to evaluate the combined class budget.
This wind-only aggregate is not a substitute for water accounting. Only the latest scalar wind receipt is held.

In a combined receipt, `wind.classes.boundary_export_hex` is the complete occurrence export (transport plus final
settlement). `handoff.classes.boundary_export_hex` reports only the native finalizer's settlement export, which the
wind total already includes. Accounting uses the `wind` total once; the handoff value is completion evidence, not an
additional export to add or a replacement for the occurrence total.

At an accepted native boundary, save the genuine native lifecycle checkpoint, then obtain:

```python
record = bridge.checkpoint_record(native_manifest_sha256=native_file.manifest_sha256)
```

The caller must publish that scalar record, its external SHA and the verified native file as one atomic accepted
checkpoint. On restart, load the actual native file under its expected identity and manifest SHA, prepare the same
static wind runtime, then use a fresh bridge's `restore(lineage, boundary, record, expected_record_sha256=...,
native_manifest_sha256=...)`. The record binds source/case/policy/schedule, native head/context, next occurrence and
exact accumulated time/exports. It cannot replay an already accepted wind row or skip an earlier unexecuted row.

This API supplies the native-file/cursor connection. It does not publish files atomically itself or provide a full
combined scenario CLI/runtime capsule. Active-event restart, LEAF bridges and GPU/windowed qualification remain
separate work. A caller must not treat an arbitrary supplied manifest string as proof that a native file was saved.

## Verification scope

The real CPU controls cover ponded-water carry, W/J/I and routing preservation, two same-day rows using one file,
reduced rain/day and full storm/recession/day between wind events, exact combined sediment budgets using the native
bounds, quota refusal/retry, interrupted adoption, initial-retirement failure and wind-workspace release.
Native checkpoint plus coordinator cursor resumes to the same next-wind state as uninterrupted execution.
The fresh native restart control uses a separate interpreter with original-bed/owner/property initialization disabled;
full combined runtime provisioning remains outside this interface's claim.

See the owning SYRUP `docs/subsurface_hydrology/wind_handoff.md` for native validation, dirty-column reconciliation,
resource-model terms and the native test suite. The dated qualification records the final source hashes, checks and
independent Claude findings. These are engineering controls, not scientific vegetation or erosion calibration.
