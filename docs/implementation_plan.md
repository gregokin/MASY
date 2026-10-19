# Initial MAPLE-SYRUP implementation phases

These are MASY milestones, independent of upstream phase numbers. User priority is connection first, with hydraulic refinements continuing in SYRUP.

1. **Dependency and shared-state audit.** Resolve current upstream source paths/manifests, inspect actual scheduler/water APIs, establish compatible shared MAPLE runtime, storage representation, forcing clocks and airborne-sediment boundary behavior. Specify/verify immutable import origins, dirty-source snapshot policy, bounded settlement/resource guards, process case hashes and exact global/local clocks. Deliver executable read-only preflight and an exact compatibility report before changing physical state. Propose narrow upstream interface changes if necessary; never waive existing source guards.
2. **Sequential schedule and thin adapters.** Validate wind/rain definitions and units, reject unsupported calendar overlap, reuse MAPLE scheduler and SYRUP conservative-event entry points. Initialize the bed once; implement actual state conversion, clock mapping and explicit event readiness. Initially allow a small dense NumPy case if proven compatible. Report unresolved airborne load/incomplete water events instead of pretending a handoff succeeded.
3. **First real alternating case.** Run wind → rain/recession → wind on a small supported terrain, with genuine nonzero transport in both processes. Show that water pickup sees wind-altered inventory/topography and the second wind sees water-altered state. Verify class budgets, boundary export and recorded dry reset; compare wind-only and water-only controls to upstream execution. Obtain independent Claude/Codex review.
4. **Combined restart and collaborator workflow.** Prove interruption/resume inside wind, inside water and at transitions; package a documented example with forcing, source lock, diagnostics and reproducible environment invocation. No installation into existing environments unless explicitly scoped. Add simple event-scale maps and inventory/export summaries rather than giant full-state histories.
5. **GPU and windowed extension.** Establish actual supported storage/backend handoff, bounded state ownership, native deep references, device compatibility and CPU/GPU physical agreement. Measure transfer/transition costs and full-event memory/runtime. Any upstream missing windowed wind capability is explicit work in MAPLE, not a copied MASY wind implementation.
6. **Continuous upstream compatibility.** Test new current MAPLE/SYRUP pairs against the alternating case and restart contracts. Adopt updates per new run with recorded manifests; retain explicit previous locks for reproducibility. Hydraulic/pit/outlet refinements remain in SYRUP and reach new wrapper runs through their qualified dependency pair.

First development assignment: Phase 1, a bounded audit/preflight task. No simulation/GPU work or upstream edits were launched by project setup. Commit/push requires its own request for MASY.

## Status on October 18, 2026

The CPU source/runtime prerequisites and the restricted Dense schedule, bed adapter and Python event runner are
independently accepted. Actual MAPLE wind → conservative SYRUP rain/recession → MAPLE wind passes on one evolving
six-class bed; native controls also cover partial availability, a second rain with routing reuse/rerouting and
piecewise rainfall. This evidence uses a labelled frozen source pair and uncommitted MASY test snapshot, rather than
the user's changing MAPLE main. See [runner qualification](dense_runner.md) and [verification](../agent_handoffs/verification.md).

The native finalizer adapter and narrow absolute-origin extension have now passed independent review and real
nonempty wind → water/recession → nonempty wind → second water checks on an explicit uncommitted development pair.
Next, the owning MAPLE thread adopts/commits the [reviewed origin patch](../agent_handoffs/maple_record_origin_handoff.md);
MASY qualifies that new current committed pair. Windowed storage stays on its independent upstream branch. Then
prove combined restart and provide the collaborator workflow in phase 4. Current acceptance does not cover coupled saltation, GPU/windowed handoff, large-case performance,
scientific calibration or a production transport CLI. Leave this task's changes uncommitted until separately asked.
