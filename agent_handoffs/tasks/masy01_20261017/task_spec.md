# MASY-01: current-source compatibility preflight

User authorization (2026-10-17): proceed with MASY development using automatic orchestration: Claude designs; Codex evaluates; Claude implements; Codex independently reviews and returns corrections until accepted. This task is the first dependency/state audit milestone; subsequent adapters and transport require a qualified result, not an invented success.

## Baseline and scope

Masy has no usable Git repository. `baseline.json` records the existing scoped files, current MAPLE/SYRUP HEADs and full package-source hashes. Upstream runtime/configuration files are read-only. Unrelated untracked cases are not runtime dirtiness. Source hashes are checked again after capture and verification.

Claude receives only source code, design documentation, illustrative configuration and their exact manifest. No actual terrain, forcing, saved simulation arrays, NPZ/Zarr/deep stores or raw data-bearing run logs. Claude tools are restricted to the task workspace; no Bash, execution, network, agents or MCP. Design is read-only. Implementation writes only `implementation/`; Codex publishes reviewed files into MASY after verification. Source-context copies are evidence, never a runtime dependency or copied physics implementation.

Allowed eventual MASY files: `src/masy/`, `tests/`, `tools/masy_preflight.py` (the evaluated isolated stdlib launcher), a minimal `pyproject.toml` if justified, explicit preflight example configuration, README and documentation updates. Codex alone maintains orchestration records. No upstream changes, package installation, environment creation, Git init/commit/push, GPU work or transport simulation in MASY-01.

## Required deliverables

1. Source-backed design and compatibility audit covering the real current pair: native injection of the complete bed, dense/windowed limits, availability and prescribed surfaces, wind-mobile completion, water DryHandoff, caches/routing lifetime, mutations/staging, native restart and conservation contracts. Distinguish observed APIs, runtime probes, proposed interfaces and unqualified behavior.
2. An executable read-only preflight using the existing MAPLE CPU interpreter, current-local-committed-HEAD policy, explicit scoped dirty refusal, reproducible source/package manifests and before/after consistency checks. Reuse upstream provenance/delivery tools where suitable; preserve source guards. A concrete incompatibility is a valid truthful result; signatures alone cannot qualify transport.
3. Immutable-source import verification plan or implementation: source pair, `maple`, `maple_syrup`, `masy` origins and distribution metadata; environment/compiled-extension identities; bytecode disabled and run-owned caches. Never claim live editable imports are immutable. No historical source fallback.
4. Exact global/native clock contract, keeping native solver values unchanged and reversible mappings; reject fixed-calendar inputs. Finite settlement and resource budgets and explicit hashed case/static-input requirements before execution.
5. Focused tests of meaningful refusal/identity/clock contracts and a reproducible local report. No model or GPU launch. Explain remaining upstream extensions and next bounded milestone.

## Acceptance

Codex reads actual implementation and callers, verifies source facts against live unchanged baseline, exercises preflight on the real local pair and meaningful negative/fixture cases, checks upstream manifests unchanged, and records every finding/disposition. Unresolved high correctness findings block acceptance. Tests cannot manufacture physical compatibility or waive source guards. Keep any failed attempt and correction round. Leave changes uncommitted.

## Execution

Each Claude stage has a fresh explicit session ID, verified local CLI help, exact submitted source manifest, command, PID/start ticks, separate raw stdout/stderr, timeout and exit/reaping receipt. No ambiguous latest-session resumption. Codex remains active, gives regular progress updates, and stops this milestone truthfully if a required interface is missing. No unattended scheduler promise.
