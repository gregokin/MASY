# Initial MAPLE-SYRUP implementation phases

These are MASY milestones, independent of upstream phase numbers. User priority is connection first, with hydraulic refinements continuing in SYRUP.

1. **Dependency and shared-state audit.** Resolve current upstream source paths/manifests, inspect actual scheduler/water APIs, establish compatible shared MAPLE runtime, storage representation, forcing clocks and airborne-sediment boundary behavior. Specify/verify immutable import origins, dirty-source snapshot policy, bounded settlement/resource guards, process case hashes and exact global/local clocks. Deliver executable read-only preflight and an exact compatibility report before changing physical state. Propose narrow upstream interface changes if necessary; never waive existing source guards.
2. **Sequential schedule and thin adapters.** Validate wind/rain definitions and units, reject unsupported calendar overlap, reuse MAPLE scheduler and SYRUP conservative-event entry points. Initialize the bed once; implement actual state conversion, clock mapping and explicit event readiness. Initially allow a small dense NumPy case if proven compatible. Report unresolved airborne load/incomplete water events instead of pretending a handoff succeeded.
3. **First real alternating case.** Run wind → rain/recession → wind on a small supported terrain, with genuine nonzero transport in both processes. Show that water pickup sees wind-altered inventory/topography and the second wind sees water-altered state. Verify class budgets, boundary export and recorded dry reset; compare wind-only and water-only controls to upstream execution. Obtain independent Claude/Codex review.
4. **Combined restart and collaborator workflow.** Prove interruption/resume inside wind, inside water and at transitions; package a documented example with forcing, source lock, diagnostics and reproducible environment invocation. No installation into existing environments unless explicitly scoped. Add simple event-scale maps and inventory/export summaries rather than giant full-state histories.
5. **GPU and windowed extension.** Establish actual supported storage/backend handoff, bounded state ownership, native deep references, device compatibility and CPU/GPU physical agreement. Measure transfer/transition costs and full-event memory/runtime. Any upstream missing windowed wind capability is explicit work in MAPLE, not a copied MASY wind implementation.
6. **Continuous upstream compatibility.** Test new current MAPLE/SYRUP pairs against the alternating case and restart contracts. Adopt updates per new run with recorded manifests; retain explicit previous locks for reproducibility. Hydraulic/pit/outlet refinements remain in SYRUP and reach new wrapper runs through their qualified dependency pair.

First development assignment: Phase 1, a bounded audit/preflight task. No simulation/GPU work or upstream edits were launched by project setup. At setup, commit/push required its own request. On October19 the user explicitly authorized publication of independently accepted milestones as needed.

## Status on October 18, 2026 (October 19 UTC)

The restricted Dense schedule, complete-bed adapter, native finalizer adapter and Python event runner pass on
committed MAPLE `39bcf79` + SYRUP `b4ee14e`, MASY runtime `22aa6ab`, under a fresh genuine immutable lock.
The real nonempty four-event sequence includes native recession and a second wind starting at 169 seconds.
Exact persistent fields/clocks, class and water budgets, routing, final settlement and failure isolation pass.
Three existing genuine SYRUP wet checkpoint continuations also pass; they do not establish combined restart.
See [committed-pair qualification](committed_pair_qualification.md); Claude accepted its source-only review and Codex verified all execution/evidence conditions.

The origin patch is committed upstream and the shared-storage libraries are merged. Windowed wind remains
unsupported. Next is phase 4: combined restart and a documented collaborator workflow, addressing the known native
restart prerequisites in their owning projects. No GPU/windowed execution, large-case performance, scientific
calibration or production transport CLI is accepted. New upstream revisions require fresh qualification; old locks
remain unchanged. This task leaves documentation/configuration changes uncommitted.

## Active parallel work on October 19, 2026

MASY-09 committed the reviewed SYRUP storage candidate as `d05d8c6` and accepted its fresh Dense CPU pairing with MAPLE `39bcf79`; see [current qualification](source_pair_20261019.md). Automatic Codex–Claude work is authorized across all three repositories. Independent tasks cover native production reduced-wind configuration, MAPLE vertical-windowed wind foundations, passive exact native restart prerequisites and GPU runtime/backend contracts. One writer per implementation tree; current physical-model choices are not changed by interface work. The user authorized labelled synthetic interface fixtures while leaving production calibration pending, periodic wind with open water drainage, and native avalanching at both wind and water commits. Native envelope rejection remains intact; any new physical choice is asked separately before dependent qualification.

The [vegetation information plan](vegetation_information_plan.md) adds versioned shared vegetation and future joint native soil-water updates to the design. Current static vegetation and dry reset stay qualified as before; ecological laws and retained-water readiness are separate scientific decisions.

Current published CPU additions are described in [bounded production wind](dense_production_wind.md). Native windowed helper release and combined persisted restart remain separate gates. The [GPU runtime](gpu_runtime.md) provides immutable preparation and an identity-only probe; numerical GPU transport remains unqualified. The local orchestration state records exact current task gates and preserved failed attempts.

## Coherent CPU components (October 19, later)

The full goal is still active. Its CPU parts now exist as accepted components on one shared engine. That engine
keeps the original constructor, attributes and byte-identical default results.

- **Phase 3 extensions:** bounded synthetic production-feature coupling, manual versioned vegetation over the five
  native fields, and a native wake response that reads the latest accepted cover
  ([vegetation status](vegetation_information_plan.md#current-accepted-cpu-component-status-october-19)).
  Production calibration still waits for the user's choice.
- **Phase 4 (partial):** an opt-in [accepted-boundary store](dense_boundary_store.md) restarts at accepted
  boundaries. Its actual four-event crash/retry/exactly-once qualification passed on the exact task-77 source-manifest
  prototype. Its targeted store/routing/schema reruns and native controls have since passed on the coherent
  candidate after a test-fixture-only environment-declaration correction (task 87). Still open: publication with a
  fresh committed lock, a committed-lock binding, active-event resume, a collaborator CLI and a packaged example.
- **Phase 5:** a [private device runner](dense_device_runner.md) passes CPU semantic checks only. Windowed native
  wind remains source/design work. No GPU event, windowed run, memory fit or large-catchment result is accepted.
