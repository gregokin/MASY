# Bounded Dense GPU coupling — October 25, 2026

A labelled synthetic landscape completed **wind → water/recession → wind → water/recession** on the GPU, using the actual MAPLE and SYRUP solvers over one evolving Dense bed. A separate native-API control run used identical fixture inputs and initial bed and completed the same sequence. The captured boundary records and clock records matched exactly at all four event boundaries.

This acceptance belongs to this immutable source selection:

| Component | Git revision |
| --- | --- |
| MASY | `42b872a3a210e0f40c9021d500d289f179a9ef90` |
| MAPLE | `952047f73af93cf6f1fcf8e11b19e488f84b3899` |
| SYRUP | `5e6a589f8b4c6b37422e7f0d728115582450eef0` |
| GPU runtime profile | `62873a266bb9a0a052c2b4d4feaff67081bb3abc83349ae092f175db2357c0a7` |

This new GPU source profile is distinct from the October 19 CPU reference lock. Later repository changes require their own compatibility checks and immutable runtime selection. This result does not qualify the current HEAD of an arbitrary checkout.

## Fixture and checks

The authored interface fixture has a 4 × 4 horizontal grid, six Dense voxels per column and one 100 μm sediment class. It starts at 7 seconds; each wind event lasts 0.05 seconds. Each water event has 60 seconds of rain at 72 mm/h followed by native physical recession, with a 180-second recession guard. The guard is a completion limit, not an imposed recession duration. Native reduced transport, topographic wind, separation, creep and avalanching are enabled. Prescribed shared vegetation has height 0.5 m and cover fraction 0.25. Wind boundaries are periodic and water has explicit open drainage. Wind holds surface-water depth constant (dry) through `constant_depth`.

The run uses labelled synthetic wind coefficients and wet coefficients reused at the shared class diameter. Production calibration remains pending. Enabled features and this small case do not establish calibrated erosion magnitude or large-landscape performance.

The two roles ran in separate, serial, monitored workers. Each completed successfully and was reaped before Codex independently accepted its saved record. The control invokes upstream transport, finalization and water completion APIs directly. The comparison checks the complete boundary records captured by the qualification observer, including exact clocks; it does not infer equality for uncaptured fields. Each role passed native class-budget checks, accepted completion for its two water events, and settlement and empty-airborne-inventory checks for its two wind finalizations. A separate wind → water prefix passed on the same source selection and monitoring controls.

Observed usage stayed within these worker limits: 8 GiB for the CuPy pool, 9 GiB for sampled driver-memory increase, 5 GiB for sampled host RSS and 600 seconds wall time per role. These controls and the accepted small run are not a complete peak-memory estimator or a landscape capacity measurement. Allocation checks require the selected physical GPU to be free of other compute jobs before launch and idle after worker reaping. Other physical GPUs and unrelated jobs are left untouched.

The local acceptance record is `agent_handoffs/tasks/masy694_GPU_monitored_functional_runner_Source_implementation_20261024/Root_data/ROOT_dense_GPU_four_event_milestone_01.json`, SHA-256 `da250655498e4332d6538523ee1288c4e84fb2e703340f462a477fcd4741ff78`. Detailed simulation evidence stays local. The qualification harness records one complete result per role and validates its length and hash before comparison.

## Remaining qualification

Full CPU-versus-GPU field comparison on identical inputs remains open. The result above compares GPU wrapper execution with GPU native execution. This run did not test numerical bounds across devices. Existing upstream CPU/GPU comparison policies remain unchanged; exact copies, handoffs and discrete decisions retain their own exact checks.

Separate [bounded CPU windowed wind records](windowed_wind_qualification_20261025.md) have passed their native dense oracles: a six-class attached-flow case and a one-class case with topographic wind, separation, creep and avalanching enabled. Windowed wind–water coupling, multiclass windowed feature qualification, complete windowed production features, windowed GPU execution, persisted GPU/combined restart and larger-landscape memory/performance qualification remain open. The vegetation in the GPU fixture is prescribed. Growth, mortality, uptake and nonempty joint soil-water exchanges have no new physics implementation or acceptance.

The private production-device factory and protected qualification harness are the execution path used here. This milestone does not add a collaborator transport CLI, automatically run a model or change the runtime tool's readiness flags. The [GPU runtime tools](gpu_runtime.md) continue to establish source/runtime identity; the [device runner](dense_device_runner.md) and [bounded reduced-wind interface](dense_production_wind.md) describe the underlying APIs and historical restrictions.
