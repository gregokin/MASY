# MASY clock contract (data-only, `src/masy/clock.py`)

## Source facts

These facts come from the upstream sources:

- MAPLE `surface/topographic_commit/commit.py:1258` rejects `current_time_s < committed_state.last_commit_time_s`. The commit chronology of the shared bed must therefore be monotone across processes.
- Both native event entries accept a nonzero start time:
  - MAPLE `initial_scheduler_state(..., time_s=)`, at `aeolian/scheduler/state.py:1133`;
  - SYRUP `initial_event_state(..., t_s=)`, at `sediment_event.py:747`.

  SYRUP rainfall can also have nonzero interval edges.

As a result, MASY keeps one monotone native epoch across the shared bed. It never resets a process clock to zero and never rewrites a commit timestamp.

## Representation

The representation rests on a few rules:

- **Native times.** A native time is the binary64 value an upstream returned or accepted. It is stored unchanged as `{"hex": float.hex(t), "fraction": "p/q"}`. On decode the record must have exactly those two string keys, the hex must be the canonical finite `float.hex` form, and the fraction must equal it (`malformed_time_record` / `inconsistent_time_record`).
- **Non-negative.** Native clocks and commit stamps are non-negative, as the native solvers require. Negative values refuse with `negative_time_value`.
- **Signed zero.** A rational value cannot distinguish `+0.0` from `-0.0`, so `G(t)` alone cannot invert a negative zero. MASY therefore refuses `-0.0` as an epoch, time, commit stamp or duration (`negative_zero_time`) rather than silently rewriting it to `+0.0`; serialized records keep the hex next to the fraction so the original bits are never inferred from the fraction.
- **Epoch.** `T0` is the native time of the first ledger entry. It may be nonzero.
- **Global elapsed time.** It is defined as `G(t) = Fraction(t) - Fraction(T0)`. This is exact, with `G(T0) = 0`. For every accepted `t` (finite, non-negative, not `-0.0`), `native_from_global(G(t))` returns `t` bit for bit. `global_elapsed` applies the same native-value policy to its input (`negative_time_value` / `negative_zero_time`), and `native_from_global` applies it to its result: a negative global offset is accepted when it maps to a valid earlier native time (epoch 100, offset -75 gives 25.0), but a result below zero refuses. A global value with no binary64 image refuses (`not_a_native_value`). Nothing is rounded; the generic `encode_time`/`decode_time` codec still records any finite binary64, including -0.0, losslessly.
- **Separate fields.** Each entry keeps the realized `native_start` and `native_end` apart from the input schedule. Any `returned_elapsed` reported by the upstream is stored separately. Its exact discrepancy from `Fraction(end) - Fraction(start)` is recorded in `endpoint_discrepancy`. It is never used to correct the interval.

## Ledger invariants (`ClockLedger.append`)

Each item below is enforced, and a violation raises `ClockError` with the given code:

1. The kind is `event` or `idle` (`invalid_entry_kind`).
2. Every value is a finite float, or an int that is exactly representable. It is never a bool (`invalid_time_value`). The epoch, both endpoints, both commit stamps and any `returned_elapsed` are non-negative (`negative_time_value`) and not `-0.0` (`negative_zero_time`). The config's `clock.native_epoch_s` has the same rule.
3. The first `native_start` equals `T0` bit for bit. Each next `native_start` equals the preceding `native_end` bit for bit. A one-ulp difference refuses. Gaps exist only as explicit `idle` entries (`non_contiguous_interval`).
4. `native_end >= native_start` (`reversed_interval`).
5. `commit_before <= native_start` and `commit_after <= native_end`. No commit stamp may come from the future (`future_commit`).
6. Commit stamps are monotone. `commit_after >= commit_before`, and each `commit_before` equals the preceding `commit_after` (`non_monotone_commit`). The first entry's `commit_before` may predate `T0` (but is never negative), for example when the stamp comes from loaded state.
7. An `idle` entry performs no commit, so its two stamps are equal (`idle_commit`).

`ClockLedger.as_record` writes `schema = "masy.clock_ledger.v1"` and `global_elapsed_definition = "Fraction(native_t) - Fraction(native_epoch)"`. `ClockLedger.from_record` accepts only that schema (`unsupported_ledger_schema`) and exactly that definition (`unsupported_global_elapsed_definition`), so a record cannot redefine its own meaning. Key sets of the ledger and of each entry are closed (`malformed_ledger_record`). The ledger is rebuilt through `append`, and the whole recomputed record must equal the input in canonical JSON, so every redundant field (index, global times, endpoint elapsed, discrepancy) is checked and a bool cannot stand in for an int (`inconsistent_ledger_record`). Every malformed input raises `ClockError`.

## Forcing edges (`resolve_forcing_edges`)

Each edge offset is given relative to the event's own start, as a plain non-negative decimal string. The helper keeps two records for every edge:

- **Intended:** the decimal string and its exact `Fraction`.
- **Native:** `float(native_start + float(Fraction(edge)))`, which is the binary64 value a native clock produces. It is stored with its exact offset from the intended value (`native_minus_intended`).

Inexact sums such as `0.2 + 0.1 = 0.30000000000000004` are accepted and recorded. Nothing is retimed or rescaled.

These edges refuse:

- non-decimal input, including exponents, `nan`, `inf` and negative values;
- edges with a non-finite native value;
- reversed edges;
- collapsed edges, meaning equal successive intended values or native values. For example, `1e17 + 1 == 1e17` collapses;
- a positive first edge whose native value equals the origin, which would lose the origin-to-first-edge interval. For example, start `2**53` with first edge `"1"` collapses. A first edge of `"0"` at the origin is not a collapse;
- a negative or `-0.0` native start.

## Unsupported

Calendar and absolute-time configuration fields are refused by exact key name: `start_datetime`, `start_date`, `calendar`, `timezone`, `epoch_utc`, `absolute_start_s` and `fixed_calendar`. Fixed-calendar scheduling, including collision detection between a next event and unfinished transport, is future work. It is never shifted silently.

## Not qualified

These helpers validate data only:

- They do not show that any solver adapter maps these intervals or boundaries correctly. That item is reported as `forcing_adapter_unqualified`.
- They do not show that a native solver realizes the edges.
- They do not show that recession and wind settlement are counted in practice.

Each of those needs adapter-level runtime qualification in a later milestone.
