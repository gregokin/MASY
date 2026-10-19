# Opt-in reduced wind: bounded CPU interface candidate

`DenseRunner(..., wind_profile=profile, production_limits=DenseProductionLimits())`
accepts the actual native config produced by `build_dense_production_wind_config`.
Use the exact bed context availability array and shared native model geometry,
classes, avalanche and commit specs. The runner rebuilds/audits through MAPLE's
native mapper and binds the full coupled schedule, wind source indices, envelope
record and caller calibration declaration. The derived wind record only computes
the forcing bound; all water/recession intervals and native event clocks remain
in the original schedule. Old calls retain the strict aerodynamic profile.
The opt-in requires at least one actual wind event; rain-only schedules refuse
an empty envelope. Static rechecks may allocate small audit records before work
staging; they do not advance native transport.
The production audit reads the hydraulic routing trigger through native
`reroute_threshold_m`; the wrapper never duplicates the decision. The authored
activation fixture sets the existing native pending-elevation trigger to1e-9m
and records it exactly. This is an explicit numerical interface fixture, not
a claim about the default0.00075m gate or production rerouting frequency.

The present path is **small CPU interface qualification only**. Production
calibration remains pending. Native reduced envelope/composition rejection and
tolerances are unchanged. No wind/water physics is copied. Topographic wind,
separation, creep, commit avalanche and finalization are native configuration and
execution. Wind envelopes never become physical terrain. Fresh native scheduler
caches are constructed over the accepted bed at each wind entry; the finalizer's
topographic/composition invalidation and empty mobile pool are verified.

The user explicitly accepted periodic wind with open water drainage, native
availability-aware avalanche at both process commits, and dry surface water
preserved during wind. ModelConfig, scheduler and SYRUP context must explicitly
agree on `constant_depth`, with wind water transport disabled. Nothing silently
overrides the native `constant_free_surface` model default. Native commit selects
the water callback whenever physical water is present, even with transport off.
Dry water arrays and native wind volume-exchange accounting are checked.

The library requires an **already enforced dedicated Linux worker** with finite
AS soft=hard <=4GiB, CPU soft<=120s/hard<=135s, CORE soft=hard0 and
FSIZE soft=hard128MiB. The configured
and observed limits must match exactly. The default ceiling is4GiB/120/135s.
Case bounds are <=64cells,1class,16fixedvoxels. PID/start ticks/bootid and the
explicit single-thread CPU/allocator environment must remain unchanged. Limits
are read before construction, each entry, known wrapper work and immediately
before publication; the library never sets process globals. Existing wrapper
stage/work/step/report/receipt guards also apply.

The task harness sets AS/CPU/CORE/FSIZE before model imports and CPU/thread/arena
environment before exec, preflights fresh host/cgroup available memory>=AS+1GiB
and disk>=128MiB+1GiB, enforces180s wall,2MiB per stream and128MiB run output,
drains pipes continuously and reaps the exact process group. Each accepted
receipt is emitted as a complete length+SHA frame before the next event.
Actual retained wrapper owners and process virtual size are diagnostics. There
is **no complete native peak estimator**: FFT/separation, flat transport,
avalanche/native scratch and output lifetimes are covered by the whole-worker OS
ceiling, not invented partial sums or an FFT multiplier. AS is virtual memory,
not RSS; this does not establish arbitrary-size or production resource admission.

Python refusals, MemoryError, native error, incomplete recession and finalizer
failure leave the last accepted boundary unchanged and block the next event.
The production branch identifies MemoryError as `address_space_exhausted`.
An OS/runtime termination may prevent even a stop receipt; the harness reports
worker failure and only complete emitted accepted frames. No persisted resumable
state is implied. Combined restart, GPU/windowed/large runs remain unqualified.

The authored fixtures reuse the explicitly synthetic native single100um reduced
coefficients/range/composition allowance unchanged and explicitly tighten that
fixture's warn policy to reject. Wet Plot1 class1 coefficients are
reused at that same100um diameter and explicitly labelled changed/synthetic;
they are not the unchanged62.5um wet fixture or measured production calibration.
The gentle4x4 fixture exercises real four-event alternation/direct native parity.
Separate mound wind witnesses show native separation, active wind avalanche and
paired creep effects. A separate3x3 open-edge rain-entry witness exercises active
water avalanche on an authored unstable initial mound. Latest-commit vectors are not whole-event totals or a solver
convergence history. Conservative signed domain nets do not prove gross motion.
A mound-induced native hydraulic sink is retained as a truthful failed transition;
no sink policy, filling/carving, calibration allowance or tolerance is altered.
