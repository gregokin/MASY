# Daily native water connection

`masy.daily_native.open_water_session` connects a parsed daily CSV scenario to
SYRUP Phase 1.5's actual daily water lifecycle. MAPLE initializes the landscape;
SYRUP owns the surface water, layered soil, infiltration, drainage, evaporation,
storm recession and every water ledger. MASY reads weather, requests one native
occurrence per nominal date, and adopts a boundary after native acceptance.
There is no copied hydrology or PET equation.

This is a bounded, fixed-geometry CPU Python API. It accepts only wind-free
scenarios with `transpiration_enabled: false` and `growth_enabled: false`.
The whole scenario is checked before starting a native daily lineage. A wind
later in the file also refuses immediately. The full illustrated scenario can
be inspected now, but needs the wind and LEAF bridges for physical execution.

The user has now specified the [wind moisture approximation](wind_moisture_approximation.md):
the planned wind bridge treats soil as dry for wind physics while retaining
SYRUP's actual soil/surface water. It must not require physical dryness or add
a moisture-dependent entrainment threshold. This is an explicit known
limitation. The policy is documented; the current water-only API and historical
wind guards have not been changed by this decision.

## Explicit source and case association

There is no implicit upstream selection or installation. The caller supplies a
genuine `maple_syrup.native_case_profile.NativeSourceProfile`, its genuine
`NativeCaseAdmission`, and `LayeredCaseSetup` produced by the native factories:

1. `verify_source_profile(bundle, expected_syrup_digest=SYRUP_PACKAGE_SHA256)`
2. `admit_package_case(profile, case_dir)`
3. `prepare_native_dense_case(admission, caller_reserve_bytes=...)`
4. `load_layered_config(layered_water_yaml)` and `prepare_layered_case(...)`

The connection requires MAPLE `9f466d2cfcd52991b24496cf67f4562d6268afa9`, MAPLE
package `fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e`,
and the published Phase 1.5 SYRUP package
`82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107`.
SYRUP source commit `06952dbc2f984af8fb75dbf09a9c9f7fdbba23fc`, documentation
commit `a5b8b18358e0d014fe13c9f48d00eb582710655a`, records the owning qualification.
These are an explicit historical qualification pair, not a claim about the
current live upstream HEADs. The native verifier rechecks actual module origins,
bundle metadata, dependencies and source manifests. Old MASY locks and guards
are not changed. A new source pair requires separate qualification.

Latitude comes from the sealed admission's `location.latitude_deg` in native
MAPLE `case.yaml`; it must match the setup's latitude binding. Both PET methods
use the actual SYRUP weather factory. The caller supplies a genuine native
`weather_pet.TerrainFactor`, with one value per cell, NumPy backend and the same
latitude. Terrain science stays with native factories: MASY does not derive
aspect conventions or assume a flat landscape. It applies that factor once.

## Calling the connection

After constructing the native objects above in an explicitly source-bound,
isolated CPU runtime:

```python
from masy.daily_csv import load_daily_scenario
from masy.daily_native import open_water_session

scenario = load_daily_scenario("scenario.csv")
session = open_water_session(
    scenario, setup=setup, admission=admission, source=profile,
    terrain=terrain, control=native_storm_control,
    quiet_policy=native_quiet_policy,
    storm_limit_s=storm_guard_seconds, pet_max_bytes=pet_workspace_quota,
)
receipt = session.advance().record()
accepted_boundary = session.state
```

The caller explicitly supplies native storm controls and quiet policy. The
finite storm guard and PET workspace bound are execution/resource admission
limits, not a replacement for the scenario's calendar controls or native solver
configuration. The PET bound declares 32 bytes per cell of temporary CPU weather
workspace and is compared with the host-memory reading before each day. The
reader defaults to SYRUP's actual OS reader; this is admission, not an OS memory
reservation. Native factories separately admit solver memory. No large-domain
memory or performance qualification follows from the tiny CPU tests.

For each date MASY builds a native `DailyWeather` and calls native
`potential_demand`, `apply_terrain` and `pet_m_s`. With evaporation disabled the
applied E request is zero, while weather validation and identity remain active.
It then calls `next_plan` with that row's P/I and the weather header's threshold.
All rows also pass through the native rain-plan validator before starting the
lineage, so an unrepresentable amount fails before the first daily update.
Binding forcing to a realized physical start remains a per-occurrence check.

For no rain or reduced recharge, `daily_water_action` performs one native
86400-second update. Only its minted `CapacityFailure` authorizes
`begin_daily_storm` from the original parent and the full original rainfall.
Above threshold, the same native storm entry is used directly. `evolve_storm`
must genuinely complete rain and recession within the explicit bounds before
`finish_daily_storm` performs one daily update. `close_interevent` and the native
lifetime audit precede adoption. No artificial dry reset, subdaily soil/E/T,
biological catch-up or relaxed completion/CFL tolerance is introduced.

`AcceptedWaterDay` contains scalar native receipts, their digest, requested
output/checkpoint cadence flags, and the nominal date. Actual native start/end
seconds are also preserved as lossless float hex. Wind/storm/recession duration
adds physical time between nominal daily steps; a calendar year may therefore
contain more than 365/366 days of physical elapsed time.

`advance` does not write output files or automatically checkpoint. Cadence flags
are requests to a future full scenario runner. At schedule end it raises
`StopIteration`. On refusal it keeps the last accepted boundary, exposes scalar
failure metadata, and stops. The exception may carry one active native state
for local inspection; this is incomplete evidence, never an accepted day.
Exception tracebacks are not retained in the session. A guard stop does not
advance the nominal date or trigger E/T/growth.
Interruptions stop the session as well. Receipt serialization failure preserves
the previous boundary and exposes the closed candidate as unadopted evidence.

## Physical boundary restart

`session.save_checkpoint(path, limits=..., host_available_bytes_fn=...)` delegates
to SYRUP's native V3 atomic writer with explicit quotas and a new destination.
The checkpoint binds the source pair, MASY connection sources, full scenario
identity, native setup/configuration, terrain, native controls and run digest.
The native daily cursor persists the next ordinal and pending status along with
physical state and water accounting. There is no second temporal file that can
get ahead of the native physical checkpoint.
Exact source bytes are intentional: even a comment edit in one of the three
bound MASY modules invalidates its old checkpoints. Execution requires readable
source files and checks them again before and after a candidate day; source
edits during a session refuse adoption. An installed or zipped runtime is not
part of this source-only profile.

Pass `checkpoint_path`, `checkpoint_limits` and `host_available_bytes_fn` to
`open_water_session` with the same factory products/identities to restore a
closed boundary. The returned native operator is adopted along with its loaded
state. The cursor selects the next nominal date without replaying rain, E or
drainage. Changed weather, threshold, controls, sources or terrain refuse.
This is a water-only boundary restart; combined bed/wind/LEAF restart and active
storm recovery through this wrapper are not qualified.

## Verification and limits

The opt-in tests in `tests/test_daily_native.py` use genuine native MAPLE case
compilation, Dense preparation, layered setup, daily operations and V3 codec.
They include equality at 5 mm, a dry leap day, downward gravity/open-bottom
drainage, closed retention, full original-rain fallback, actual full storms,
native Hargreaves, E disabled, restart identity refusal and exactly-once daily
continuation. They also require a bounded storm stop and native daily router
refusal to preserve the accepted boundary.

The fixture is synthetic, 4 by 6 cells and two soil layers. Its explicit flat PET
terrain and prescribed weather are illustrative, not calibrated field forcing.
Native Phase 1.5 can refuse a single daily step when a retained pond violates
the existing router's CFL. Such a `router_step_rejected` is propagated with no
hidden substeps, changed PET, changed tolerance or unauthorized storm fallback.

LEAF demand/allocation/feedback/growth must connect as plant-only operations:
one demand, one SYRUP lifecycle withdrawal, then actual uptake feedback. The
older combined LEAF water driver must not be called as well. Wind exchange with
persistent layered state, biological transactions, combined physical restart,
CUDA/windowed/scaled operation and scientific field validation remain separate
work. Importing the connection alone loads no native model or NumPy.
