# Current wind, water and vegetation growth

`masy.daily_triplet.DailyTripletSession` composes the actual MAPLE wind model,
SYRUP Phase 2.5 water model and LEAF plant session on one evolving landscape.
This is an explicit dense NumPy CPU profile. Its source runtime is described in
[triplet_runtime.md](triplet_runtime.md). Historical two-model workflows retain
their separately qualified source pairs; use this profile for vegetation growth.

The initial source frontier is MAPLE `1dca853`, SYRUP `cd07f46` plus the additive
current-MAPLE/vegetated-handoff compatibility patch, and LEAF `ff127d2` plus its
native floating-soil lifecycle bridge. The accepted descriptor records the full
resulting commits, complete package hashes, LEAF's external mapping helper and
MAsy's own source. LEAF's Phase 2.5 traits are on
`feature/syrup-hydrology-traits`, ahead of its older local `main`.

New runs resolve the explicitly configured committed source refs. Advancing a
ref requires requalification; a running case stays on its immutable four-source
lock. Dirty active developer checkouts are recorded and excluded from this
explicit committed-source selection. They are never silently imported, reset or
updated. No package installation or automatic Git pull is involved.

The caller supplies the parsed daily scenario/plan, native case admission,
landscape, water event factory and controls, LEAF species/population/session
configuration, terrain PET mapping, evaporation partition, vegetation site
controls, wind vegetation export policy and finite memory/execution limits.
Those are scientific case inputs, not parameters inferred by the wrapper.
`prepare_triplet_wind` binds the complete caller input identity through its
required `run_inputs_sha256`; applications must hash their actual resolved
inputs. A source qualification never supplies field calibration.

For a verified runtime, use its genuine `runtime.source_profile` to admit the
native case and bind `leaf.floating_lifecycle_bridge.bind_source`, using source
commits/package hashes from `runtime.record()`. Pass that same prepared case and
source to `prepare_triplet_wind`. Build `DailyTripletSession` with that executor,
the native initial event factory and the LEAF configuration. Join
`expected_external_helpers` to `runtime.record()['external_helpers']`, and use
the qualified LEAF package hash. Each `session.advance()` completes one nominal
day and returns a bounded scalar receipt; `StopIteration` ends the schedule.
The existing MAPLE case workflow remains the landscape/case authority.

Within a day, scheduled wind finishes through MAPLE's native finalizer before
water transport can start. SYRUP completes storm transport and recession, keeps
the real soil and surface water, and performs its native daily water and
evaporation action. LEAF computes plant demand; SYRUP performs the actual
soil-water debit; LEAF receives exactly that feedback and applies its existing
growth laws. The current accepted plants are projected into both models before
their next transport operation. Native vegetation-dependent soil properties and
water displacement remain SYRUP's responsibility.

Nominal Gregorian weather/plant dates and native physical time are distinct.
Each nominal day includes one full native daily water action. Wind, storm and
recession add their actual physical seconds without extra plant days or shifted
weather rows. Receipts preserve the native start/end clocks in exact hexadecimal
float notation. Wind preserves water depth and uses the approved dry-wind
approximation; it does not erase the retained soil or ponded water.

Failures stop the session and expose the last native lineage head, accepted
plants and active-state evidence. A whole day is not atomic: earlier native
subtransactions may already be accepted. Limits never count as completion.
The new combined persisted restart is explicitly refused. Receipt output and
checkpoint flags report requests; they do not write files automatically.

Qualification uses labelled synthetic interface fixtures with real native
providers and laws, including wind, storm/recession, daily evaporation,
transpiration, growth and following-day continuity. It does not establish field
calibration, arbitrary memory capacity, GPU/windowed execution, annual plant
transitions or a persisted three-model restart. Those require separate cases
and acceptance.
