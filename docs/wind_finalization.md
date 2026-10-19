# Dense wind-event finalization

MASY-06 implements native instantaneous event-end settling for the restricted
Dense CPU runner. The implementation has passed independent review and actual isolated CPU checks on an explicit
uncommitted development pair. This is not production committed source-pair acceptance. MAPLE `7b0e2539` provides the
finalizer; the isolated native extension adds keyword-only `record_origin_s=0.0`.
SYRUP remains the exact frozen `b4ee14e` source. No native physics is copied.

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

The candidate uses actual SYRUP dependency-bundle export/materialization,
recording the native original revision and isolated patch chain. Its explicit
development manifest distinguishes uncommitted patched MAPLE and MASY from the
parent committed-pair lock. Every execution verifies source hashes, real
metadata, native source guards and imported module origins before and after.
No existing lock, environment, live upstream tree or running job is changed.
GPU, windowed storage, large-case performance, scientific calibration and
combined checkpoint/restart remain unqualified.

Claude accepted the bounded source patch with no blocking findings. The reviewed native patch and ownership handoff
are [here](../agent_handoffs/maple_record_origin_handoff.md). MAPLE's existing absolute-clock completion tolerance
is retained and may refuse large origins/long accumulated runs; such a refusal blocks publication. It does not
establish completion. Exact review notes and their dispositions are in the task's acceptance receipt.
