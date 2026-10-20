The passive Dense receipt schema preserves the current native terminal sediment
bound as a one-element list of plain finite floats. SYRUP's
`conservation.reservoir_bound_kg` returns a scalar; `complete_event` retains that
scalar, and unchanged `DenseRunner._floats` encodes it as `[float]`. Its length is
one for both one-class and multi-class inputs.

Only the rain `classes.terminal_sediment_tolerance_kg` shape check changes.
Class-resolved closure tolerances, inventories, transfers and bounds keep their
existing class count. No value, broadcast, tolerance, budget, native policy or
transport behavior changes, and validation does not normalize the stored list.
Malformed kinds, nonfinite values, missing/unknown keys and other class-vector
shapes still refuse.

The focused component check generates the unchanged six-class wind/rain fixture,
reproduces the old schema refusal, checks corrected full runner admission,
and uses the actual receipt locally for negative controls. Its one-class and
signed-zero bodies are labelled synthetic schema fixtures; they are not physical
one-class runs or calibration evidence. Actual receipts and model payloads are
kept local and excluded from source-only review packets.

This isolated correction does not establish persisted four-event continuation.
That qualification requires independent acceptance, a new complete source
identity and fresh original/genesis state. Prior failed task60 data, old source
snapshots and source locks remain unchanged; no migration or rebinding occurs.
