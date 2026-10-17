# Automatic Codex-Claude handoffs

Read `orchestration_state.md`, `current_prompt.md`, governing specifications and actual files/processes before resuming. `claude_report.md`, `codex_review.md` and `verification.md` are current pointers, not proof by themselves. Immutable task-specific prompts, baseline manifests, source diffs, raw logs, exact commands/session/process identities, reports, corrections and acceptance belong under `tasks/<task_id>/`.

Use `templates/task_prompt.md` and `templates/task_report.md`. Codex assigns bounded work and independently reviews; Claude leaves changes uncommitted. One writer per tree. Ordinary permissions, scoped tools/directories, separate stdout/stderr and bounded waits are mandatory. Relevant source sharing is authorized; saved simulation payloads remain local without separate permission. Do not inherit old live PIDs or start jobs just because a state file says automatic mode.
