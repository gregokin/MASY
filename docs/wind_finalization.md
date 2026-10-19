# Dense wind-event finalization

MASY implements native instantaneous event-end settling for the restricted Dense CPU runner.
The original MASY-06 adapter passed independent review; MAPLE committed its absolute-origin extension at
`b3e8756`, now included in merged `39bcf79`. Historical MASY-08 qualified `39bcf79` + SYRUP `b4ee14e`
under its [retained immutable lock](committed_pair_qualification.md). The current accepted reference is
[MASY-09](source_pair_20261019.md): MAPLE `39bcf79` + SYRUP `d05d8c6` + MASY `22aa6ab`, lock `43976b9a…`.
Those reports record exact sources, commands, review status and limits. No native physics is copied.

A scheduled wind event currently contains one forcing block. MASY advances the
actual absolute block on its staged evolving bed, verifies native transport
budgets, and maps its consumed cursor to the terminal record cursor using the
native record loop's configured completion epsilon. Its local forcing record
contains that event's actual forcing only. MASY passes the accepted absolute
start as `record_origin_s`. Native validation still requires the correct final
wind, complete cursor and physical end within the original native tolerance.
Physical, cohort, commit and water clocks are unchanged. Settlement is
instantaneous and adds no elapsed interval. Actual native accumulated wind and
water/recession intervals remain in the clock ledger.

The finalizer must report `completed` and `water_handoff_ready`, preserve the
end clock/cursor, pass its native class identities and return genuinely empty
airborne state, including sub-resolution material. An exception or incomplete
result retains the accepted boundary and local active-event evidence and
blocks the next process. There is no retry/resume of the stopped event here.
Staged input/output memory and cohort capacity bound this small CPU operation;
the test worker separately bounds host address space, CPU time and wall time.
No calm coasting or parked mobile state substitutes for settling.

The returned bed, availability and ledger are adopted directly. Native deposited
mass is an internal transfer already present in that returned state. It is
reported in the event and aggregate internal terminal deposition totals but is
never deposited or accumulated in the native ledger a second time. Native
finalizer boundary export is external and is added once to wind-block export.
Event inventory checks include both native transport and finalizer bounds plus
the reduction bound for the actual bed inventory sums. Water reset/deposition
remains documented by SYRUP's genuine dry handoff.

Each wind receipt includes the actual local record and hash, absolute origin,
completed native cursor, physical clock, original transport export/bed change
and post-transport mobile inventory, native finalizer deposition/export and its
transfer/reservoir gap and bound. The retained water routing graph/source and
native reroute decision remain unchanged in ownership; subsequent water entry
compares the evolved physical bed against that retained source.

The committed pair uses actual SYRUP dependency-bundle export/materialization and matching metadata built from
the committed MAPLE source. Every execution verifies source hashes, real metadata, native source guards and imported
module origins before and after. The earlier uncommitted development pair remains historical evidence; old locks,
installed environments, upstream trees and unrelated jobs are unchanged. GPU/windowed execution, large-case
performance, scientific calibration and combined checkpoint/restart remain unqualified.

The [original reviewed native patch](../agent_handoffs/maple_record_origin_handoff.md) is retained for provenance;
it has been implemented upstream and must not be reapplied to the merged tree. MAPLE's existing absolute-clock completion tolerance
is retained and may refuse large origins/long accumulated runs; such a refusal blocks publication. It does not
establish completion. Exact review notes and their dispositions are in the task's acceptance receipt.
