"""In-environment worker of the configured water-only native calendar run (`python -m masy.native_water_run run CONFIG` launches it in a fresh child).

Runs INSIDE the genuine immutable-bundle environment (native 9f `editable`/`source/src` + the declared SYRUP source root + this MASY root). It composes only
existing, qualified components: SYRUP `native_case_profile`/`native_dense_profile` (the actual MAPLE case), SYRUP `layered_case` (the configured soil layers, retained
texture, explicit hydraulic/matrix/preferential/evaporation/cover/initial parameters and domain), MASY `load_scenario`/`latitude_binding_from_case_yaml` and
`NativeWaterSession` (temporal coordination of the real SYRUP lifecycle). No landscape, hydrology, terrain, PET, rainfall or scheduling equation is copied here.

Order (every refusal before the first physical step is exit 2 and leaves only the run's own status file): run directory exclusively created -> bounded input pins ->
source census + namespace + distribution of the declared named profile -> scenario + latitude (actual MAPLE case.yaml) -> actual case admission/preparation ->
layered factory -> `NativeWaterSession.create` (its own cold admission/census/recheck; the preparation and compiled case are then RELEASED) -> steps. Physical failure
or an incomplete event is exit 3 with the last accepted boundary summary; nothing is retroactively applied, no E/T/growth during rain or recession, no forced dry reset.
A terminal physical checkpoint (optional) is a STANDALONE Stage-B native9f lifecycle bundle, not a combined calendar+physical resume.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import hashlib
import json
import os
import signal
import stat
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from masy import native_source_gate as gate
from masy.native_water_run_config import MANIFEST_MAX_BYTES, PROFILE_CAPABILITY, STATUS_RESERVE_BYTES, RunConfig, RunConfigError, load_manifest, load_run_config, make_availability
from masy.scenario_io import IOLimits, ScenarioIOError, canonical_json, read_bounded

__all__ = ["EXIT_PAUSED", "Invocation", "RunOutcome", "RunOutputs", "RunOutputError", "main", "run_configured", "run_configured_production"]

EXIT_OK, EXIT_INTERNAL, EXIT_REFUSED, EXIT_FAILED = 0, 1, 2, 3
EXIT_PAUSED = 4  # a clean pause after an accepted whole occurrence, with its combined checkpoint written (distinct from complete and from every failure)
STATUS_NAME = "RUN_STATUS.json"
RESULT_PREFIX = "MASY_RUN_RESULT:"
COMPILE_MAX_MEMBERS = 16  # native compile members: 5 processed + diagnostics + provenance + masks, doubled for a temp+publish copy (derived from the native writer's output set)
MANIFEST_PARSE_FACTOR = 12  # declared working-memory model of a strictly parsed JSON/YAML text: 12 x its bytes (text + token/node graph), conservative
_COMPILED_NAMES = ("processed", "diagnostics", "masks", "provenance.yaml")


class RunOutputError(RuntimeError):
    """Output collision, unsafe parent, symlink, budget or partial-write failure (nothing existing is ever replaced)."""


class RunRefused(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclasses.dataclass(frozen=True)
class Invocation:
    """The per-invocation controls (NOT configuration): an explicit pinned resume and/or a pause count. `options_sha256` is the parent's identity of them."""

    resume: str | None = None
    expected_seal_sha256: str | None = None
    pause_after: int | None = None
    options_sha256: str | None = None


@dataclasses.dataclass(frozen=True)
class RunOutcome:
    status: str  # complete | refused | failed | incomplete | paused
    exit_code: int
    code: str | None
    detail: str | None
    summary: dict[str, Any]


class RunOutputs:
    """Run-owned output directory: created EXCLUSIVELY under a real (non-symlink) parent, every file `O_CREAT|O_EXCL|O_NOFOLLOW` relative to a directory descriptor.

    HARD budgets, never raised: `max_total` and `max_file` are all-inclusive. `status_reserve` bytes of BOTH are reserved from the start for the terminal status, so every other
    write (input copies, records, step lines, compiled/native outputs, the standalone checkpoint) is admitted against `max_total - status_reserve` (and `max_file`) BEFORE it is written,
    and the status itself against `status_reserve`. `reserve()`/`settle()` hold bytes for writers that cannot be planned per member (the native compile). Nothing existing is
    overwritten or deleted; a failure leaves the partial files and a status naming the failure (the existing bounded-publication convention of `masy.report`)."""

    def __init__(self, run_dir: str, max_total: int, max_file: int, status_reserve: int = STATUS_RESERVE_BYTES, *, disk_free_fn: Callable[[str], int] | None = None) -> None:
        self.max_total, self.max_file, self.status_reserve = int(max_total), int(max_file), int(status_reserve)
        self.total, self.held = 0, 0
        if self.status_reserve < 1 or self.status_reserve > self.max_file or self.status_reserve >= self.max_total:
            raise RunOutputError(f"the {self.status_reserve}-B terminal-status reserve does not fit inside max_file_bytes={self.max_file} / max_output_bytes={self.max_total}")
        self.path = Path(run_dir)
        parent = self.path.parent
        try:
            plst = os.lstat(parent)
        except OSError as exc:
            raise RunOutputError(f"output parent {parent} cannot be examined: {exc}") from None
        if not stat.S_ISDIR(plst.st_mode) or stat.S_ISLNK(plst.st_mode):
            raise RunOutputError(f"output parent {parent} must be a real directory (not a symlink)")
        free = (disk_free_fn or _disk_free)(str(parent))
        if free < self.max_total:
            raise RunOutputError(f"output parent {parent} has {free} B free, less than the declared {self.max_total}-B output cap")
        try:
            os.mkdir(self.path, 0o700)
        except FileExistsError:
            raise RunOutputError(f"run directory {self.path} already exists (no overwrite)") from None
        except OSError as exc:
            raise RunOutputError(f"run directory {self.path} cannot be created: {exc}") from None
        self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.dirs: dict[str, int] = {"": self.fd}
        self.streams: dict[str, int] = {}
        self.written: dict[str, int] = {}

    @property
    def budget(self) -> int:
        """Bytes still admissible for non-status output."""
        return self.max_total - self.status_reserve - self.total - self.held

    def _dir(self, rel_dir: str) -> int:
        if rel_dir in self.dirs:
            return self.dirs[rel_dir]
        parent = self._dir(rel_dir.rsplit("/", 1)[0] if "/" in rel_dir else "")
        name = rel_dir.rsplit("/", 1)[-1]
        os.mkdir(name, 0o700, dir_fd=parent)
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        self.dirs[rel_dir] = fd
        return fd

    @staticmethod
    def _check_name(rel: str) -> tuple[str, str]:
        parts = rel.split("/")
        if not rel or any(p in ("", ".", "..") or "\x00" in p or "\\" in p for p in parts) or len(rel) > 240:
            raise RunOutputError(f"unsafe output name {rel!r}")
        return "/".join(parts[:-1]), parts[-1]

    def check(self, rel: str, add: int, *, status: bool = False) -> None:
        """Admission of `add` more bytes into `rel` (raises before anything is written)."""
        if add + self.written.get(rel, 0) > self.max_file:
            raise RunOutputError(f"{rel}: {self.written.get(rel, 0) + add} B exceeds the {self.max_file}-B per-file output cap")
        if status:
            if add > self.status_reserve or self.total + add > self.max_total:
                raise RunOutputError(f"terminal status {add} B exceeds its {self.status_reserve}-B reservation / the {self.max_total}-B run output cap")
        elif add > self.budget:
            raise RunOutputError(f"output {rel} (+{add} B) exceeds the {self.max_total}-B run output cap ({self.status_reserve} B reserved for the terminal status, "
                                 f"{self.total} B written, {self.held} B held)")

    def write(self, rel: str, data: bytes, *, status: bool = False) -> None:
        d, name = self._check_name(rel)
        self.check(rel, len(data), status=status)
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self._dir(d))
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        self.written[rel] = len(data)
        self.total += len(data)

    def append(self, rel: str, data: bytes) -> None:
        d, name = self._check_name(rel)
        self.check(rel, len(data))
        if rel not in self.streams:
            self.streams[rel] = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_APPEND, 0o600, dir_fd=self._dir(d))
        view = memoryview(data)
        while view:
            view = view[os.write(self.streams[rel], view):]
        self.written[rel] = self.written.get(rel, 0) + len(data)
        self.total += len(data)

    def reserve(self, nbytes: int, what: str) -> None:
        """Hold `nbytes` of the budget for a writer that cannot be planned per member (refuses BEFORE that writer runs)."""
        if nbytes > self.budget:
            raise RunOutputError(f"{what} needs a {nbytes}-B output reservation but only {self.budget} B remain")
        self.held += nbytes

    def settle(self, reserved: int, actual: int, what: str) -> None:
        """Replace a reservation by the measured size; exceeding the reservation is a failure (never silently absorbed)."""
        self.held -= reserved
        if actual > reserved:
            self.total += actual
            raise RunOutputError(f"{what} wrote {actual} B, more than its {reserved}-B reservation")
        self.total += actual

    def measure(self, names: tuple[str, ...], under: str = "") -> int:
        """Real size of the named entries (files/trees) under `under`; special files and symlinks are refused."""
        size = 0
        root = self.path / under
        for name in names:
            top = root / name
            if not os.path.lexists(top):
                continue
            for dirpath, _dirs, files in os.walk(top):
                for f in files:
                    st = os.lstat(os.path.join(dirpath, f))
                    if not stat.S_ISREG(st.st_mode):
                        raise RunOutputError(f"{os.path.join(dirpath, f)} is not a regular file")
                    size += st.st_size
            if os.path.isfile(top):
                size += os.lstat(top).st_size
        return size

    def untracked_bytes(self) -> int:
        """Bytes of every regular file under the run directory that no recorded write accounts for (native compile members, partial temp files, checkpoint members): a
        non-regular entry refuses. This is what settlements charge, so nothing written by a foreign writer escapes the budget."""
        size = 0
        for dirpath, _dirs, files in os.walk(self.path):
            for f in files:
                st = os.lstat(os.path.join(dirpath, f))
                if not stat.S_ISREG(st.st_mode):
                    raise RunOutputError(f"{os.path.join(dirpath, f)} is not a regular file")
                size += st.st_size
        return max(size - self.total, 0)

    def account_tree(self, rel: str) -> int:
        """Charge an externally written bundle AFTER the fact (its size was bounded BEFORE the write by the caller); over budget is a RunOutputError."""
        size = self.measure((rel,))
        self.total += size
        if self.total + self.held > self.max_total - self.status_reserve:
            raise RunOutputError(f"the {rel} bundle ({size} B) exceeds the run output budget")
        return size

    def close(self) -> None:
        for fd in self.streams.values():
            os.close(fd)
        for key, fd in sorted(self.dirs.items(), key=lambda kv: -len(kv[0])):
            os.close(fd)
        self.streams.clear()
        self.dirs.clear()


def _disk_free(parent: str) -> int:
    st = os.statvfs(parent)
    return int(st.f_bavail) * int(st.f_frsize)


@contextlib.contextmanager
def _file_size_limit(nbytes: int):
    """Process-wide RLIMIT_FSIZE for the duration: ANY writer (including the native compile that has no per-member planner) fails with EFBIG instead of overspending disk.
    The effective cap is `min(nbytes, finite existing soft, finite existing hard)` (`RLIM_INFINITY` = no ceiling): an existing stricter limit is NEVER raised and a soft limit of 0
    stays 0. SIGXFSZ is ignored meanwhile so the write raises `OSError` rather than killing the process; soft/hard and the handler are restored on every path, including when
    entering `setrlimit` itself raises."""
    import resource

    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
        raise ValueError(f"the file-size cap must be a non-Boolean integer >= 0, got {nbytes!r}")
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    effective = int(nbytes)
    for ceiling in (old_soft, old_hard):
        if ceiling != resource.RLIM_INFINITY:
            effective = min(effective, int(ceiling))
    old_handler = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (effective, old_hard))
    except BaseException:
        signal.signal(signal.SIGXFSZ, old_handler)
        raise
    try:
        yield
    finally:
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (old_soft, old_hard))
        finally:
            signal.signal(signal.SIGXFSZ, old_handler)


class _ColdGuard:
    """Fresh availability reading (bounded by the user's cap) against `reserve + buffer model` BEFORE every new cold buffer is read/hashed/allocated."""

    def __init__(self, avail: Callable[[], int], reserve: int) -> None:
        self.avail, self.reserve = avail, int(reserve)

    def require(self, nbytes: int, what: str) -> None:
        reading = int(self.avail())
        if reading < self.reserve + int(nbytes):
            raise RunRefused("cold_memory", f"{what} needs {int(nbytes)} B + the {self.reserve}-B reserve but only {reading} B are available")


def _read_input(path: Path, cap: int, guard: _ColdGuard, what: str, outputs: RunOutputs | None = None, rel: str | None = None) -> tuple[bytes, str]:
    """Bounded no-follow read of one input: size from `lstat` checked against the cap, the remaining output budget and the cold guard BEFORE the body is read; growth/change during
    the read refuses; the returned SHA-256 is of exactly the returned bytes."""
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise RunRefused("input_unreadable", f"{what} {path}: {exc}") from None
    if not stat.S_ISREG(st.st_mode):
        raise RunRefused("input_unreadable", f"{what} {path} is not a regular file")
    if st.st_size > cap:
        raise RunRefused("input_too_large", f"{what} {path} is {st.st_size} B, over its {cap}-B cap")
    if outputs is not None and rel is not None:
        try:
            outputs.check(rel, int(st.st_size))
        except RunOutputError as exc:
            raise RunRefused("output_budget", str(exc)) from None
    guard.require(2 * int(st.st_size) + 4096, f"reading {what}")
    try:
        data, pin = read_bounded(path, max(int(st.st_size), 1))
    except ScenarioIOError as exc:
        raise RunRefused("input_changed_after_pin", f"{what} {path} changed or grew while being read: {exc}") from None
    return data, pin["sha256"]


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False) + "\n").encode("ascii")


def _tree_nbytes(obj: Any) -> int:
    import numpy as np

    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return sum(_tree_nbytes(getattr(obj, f.name)) for f in dataclasses.fields(obj))
    return 0


def _availability(cap: int | None, override: Callable[[], int] | None) -> Callable[[], int]:
    """The run's ONE availability reading (strict non-Boolean integer >= 0 every call): the injected callable (tests), else the stdlib host helper equal to SYRUP's default,
    in every case bounded by the user's declared cap."""
    try:
        return make_availability(cap, override)
    except RunConfigError as exc:
        raise RunRefused(exc.code, exc.detail) from None


def _device_availability(dev: Any):
    """The ONE device availability reading of the cold CUDA factory: SYRUP's own default device reader for the declared device (created lazily on the first reading, so building
    this callable queries nothing), validated as a non-Boolean integer >= 0 and bounded by the declared `device_cap_bytes`. No synthetic value, no other device."""
    state: list = []

    def reading() -> int:
        if not state:
            from maple_syrup import subsurface_cuda

            cp = subsurface_cuda.require_cuda()
            state.append(subsurface_cuda.default_device_available_bytes_fn(cp, dev.device_id))
        value = state[0]()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RunRefused("invalid_availability", f"the device availability reading must be a non-Boolean integer >= 0, got {value!r}")
        return min(value, dev.device_cap_bytes)
    return reading


def _build_fence(cfg: RunConfig, masy_manifest: Any, syrup_manifest: Any, delivery: Any = None):
    from masy import native_water_calendar as nwc

    s = cfg.source
    try:
        base = nwc.SourceFence(native_commit=nwc.NATIVE_COMMIT, native_package_digest=nwc.NATIVE_PACKAGE_DIGEST, bundle_id=nwc.NATIVE_BUNDLE_ID,
                               syrup_package_digest=nwc.SOURCE_PROFILES.get(s["profile"], ""), masy_root=s["masy_root"], syrup_root=s["syrup_root"],
                               native_target=s["native_target"], masy_manifest=dict(masy_manifest), syrup_manifest=dict(syrup_manifest), profile=s["profile"])
        return base if delivery is None else nwc.delivery_source_fence(base, delivery)  # explicit sibling: the historical constructor call above is unchanged
    except nwc.NativeWaterCalendarError as exc:
        raise RunRefused(exc.code, exc.detail) from None


def _status_bytes(outcome: RunOutcome, reserve: int) -> tuple[bytes, RunOutcome]:
    """The terminal status, bounded by its reservation. Full when it fits; else compacted (exact core ledger values, `summary_compacted`); else a minimal FAILED status. A
    complete outcome is never persisted unless the persisted document says complete."""
    def doc(o: RunOutcome, summary: dict) -> bytes:
        return _json({"schema": "masy.native_water_run_status.v1", "status": o.status, "exit_code": o.exit_code, "code": o.code, "detail": o.detail, "summary": summary})
    raw = doc(outcome, outcome.summary)
    if len(raw) <= reserve:
        return raw, outcome
    core = {k: outcome.summary[k] for k in ("run_id", "config_sha256", "accepted_steps", "final_role", "final_t_s_hex", "final_state_digest", "lifetime", "combined_checkpoint", "invocation") if k in outcome.summary}
    core["summary_compacted"] = True
    detail = None if outcome.detail is None else str(outcome.detail)[:512]
    compact = RunOutcome(outcome.status, outcome.exit_code, outcome.code, detail, core)
    raw = doc(compact, core)
    if len(raw) <= reserve:
        return raw, compact
    failed = RunOutcome("failed", EXIT_FAILED, "status_overflow", f"the terminal status did not fit its {reserve}-B reservation (was {outcome.status})", {"run_id": outcome.summary.get("run_id")})
    return doc(failed, failed.summary), failed


def options_identity(resume: str | None, expected_seal: str | None, pause_after: int | None) -> str:
    """Same canonical identity the parent computes for the invocation controls (kept local so the worker does not import the parent module)."""
    doc = {"resume": resume, "expected_seal_sha256": expected_seal, "pause_after": pause_after}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def _preauthenticate(cfg: RunConfig, outputs: RunOutputs, scenario: Any, inv: Invocation | None, guard: "_ColdGuard", cuda: bool = False) -> dict[str, Any]:
    """Invocation controls and the resume artifact, authenticated BEFORE the reference build or any physical read: pause preconditions, then (resume) the no-follow
    bounded artifact open with its admitted header, the pinned seal, the cursor position and the pause bound. The artifact is only ever read."""
    info: dict[str, Any] = {"resumed": False, "start_next_index": 0, "lineage": None, "seal": None}
    if inv is None:
        return info
    n = len(scenario.occurrences)
    if inv.pause_after is not None:
        if cfg.output["physical_checkpoint"] != "combined_terminal":
            raise RunRefused("pause_requires_combined_terminal", "a pause needs output.physical_checkpoint: combined_terminal (a pause without a resumable checkpoint is refused)")
        if inv.pause_after >= n:
            raise RunRefused("pause_beyond_scenario", f"--pause-after-occurrences {inv.pause_after} is not below the {n} scenario occurrences (a pause at the end would be a completion)")
    if inv.resume is None:
        return info
    from masy import native_water_checkpoint as nck

    if inv.expected_seal_sha256 is None:
        raise RunRefused("seal_required", "resuming needs the explicit expected seal; there is no implicit or latest checkpoint")
    run_dir, ckpt = str(outputs.path), os.path.abspath(inv.resume)
    if run_dir == ckpt or run_dir.startswith(ckpt + os.sep) or ckpt.startswith(run_dir + os.sep):
        raise RunRefused("output_inside_checkpoint", "the fresh run directory and the resumed checkpoint must be disjoint")
    try:
        _admit = lambda nbytes: guard.require(3 * int(nbytes) + 4096, "the combined manifest header")  # noqa: E731
        if cfg.leaf is not None:  # a v2 run resumes ONLY a joint artifact (a water-only artifact would silently drop the biology and is refused by the closed layout)
            art = (nck.open_artifact_leaf_cuda if cuda else nck.open_artifact_leaf)(ckpt, admit_header=_admit, temporal_cap=int(cfg.limits["max_record_bytes"]))
        elif not cuda:
            art = nck.open_artifact(ckpt, admit_header=_admit, temporal_cap=int(cfg.limits["max_record_bytes"]))
        else:
            art = nck.open_artifact_cuda(ckpt, admit_header=_admit, temporal_cap=int(cfg.limits["max_record_bytes"]))
    except nck.CheckpointError as exc:
        raise RunRefused(exc.code, exc.detail) from None
    try:
        man = art.manifest
        if man["manifest_sha256"] != inv.expected_seal_sha256:
            raise RunRefused("seal_mismatch", "the checkpoint's manifest seal differs from the expected seal")
        idx = int(man["boundary"]["next_index"])
        if idx > n:
            raise RunRefused("scenario_mismatch", "the sealed cursor position does not fit the loaded scenario")
        if inv.pause_after is not None and inv.pause_after <= idx:
            raise RunRefused("pause_not_after_resume", f"--pause-after-occurrences {inv.pause_after} is not beyond the resumed cursor position {idx} (a pause never resets the accepted-occurrence total)")
        lineage = man["lineage"] if isinstance(man["lineage"], dict) else None
        info.update(resumed=True, start_next_index=idx, lineage=lineage, seal=man["manifest_sha256"], role=man["boundary"]["role"], last_kind=man["boundary"]["last_kind"],
                    finished=bool(man["boundary"]["finished"]), cursor_digest=man["temporal"]["cursor_digest"], checkpoint=ckpt)
    finally:
        art.close()
    return info


def _input_pins(cfg: RunConfig) -> dict[str, str]:
    """The pinned input identities handed to the checkpoint/resume join (case files, layered parameters, maps, scenario). The run configuration's own hash is deliberately NOT
    included: a resumed run has a different run directory and therefore a different configuration file."""
    pins = {"parameters": cfg.syrup_parameters_sha256, "scenario": cfg.scenario_sha256}
    pins.update({f"case/{k}": v for k, v in cfg.case_files.items()})
    for name, digest in cfg.syrup_map_pins.items():
        pins[f"map/{name}" if len(name) <= 120 else "map/" + hashlib.sha256(name.encode()).hexdigest()[:32]] = digest
    if cfg.leaf is not None:  # v2: the LEAF input files and the closed static LEAF settings join the artifact's input pins (a v1 run keeps its exact pin set)
        from masy import native_water_leaf_run as lrun

        pins.update(lrun.static_pins(cfg))
    return dict(sorted(pins.items()))


cfg_boot: list = [None]  # the verified bootstrap identity of the current run (recorded in resolved.json)


def run_configured(cfg: RunConfig, *, available_bytes_fn: Callable[[], int] | None = None, status_reserve_bytes: int = STATUS_RESERVE_BYTES,
                   disk_free_fn: Callable[[str], int] | None = None, bootstrap: dict[str, Any] | None = None, invocation: Invocation | None = None) -> RunOutcome:
    """Execute one configured run. `available_bytes_fn` (tests) replaces the native available-host reading; the config's cap still applies. (Historical public signature, unchanged.)"""
    return _run_configured(cfg, available_bytes_fn=available_bytes_fn, status_reserve_bytes=status_reserve_bytes, disk_free_fn=disk_free_fn, bootstrap=bootstrap, invocation=invocation)


def run_configured_production(cfg: RunConfig, resolved: Any, *, available_bytes_fn: Callable[[], int] | None = None, status_reserve_bytes: int = STATUS_RESERVE_BYTES,
                              disk_free_fn: Callable[[str], int] | None = None, bootstrap: dict[str, Any] | None = None, invocation: Invocation | None = None) -> RunOutcome:
    """Explicit v2 production sibling: `resolved` is the verified `masy.production_runtime.Resolved` (sealed roots, manifests, capsule-derived LEAF root)."""
    if resolved is None:
        return RunOutcome("refused", EXIT_REFUSED, "production_recipe_unavailable", "run_configured_production requires a verified production resolution", {})
    return _run_configured(cfg, available_bytes_fn=available_bytes_fn, status_reserve_bytes=status_reserve_bytes, disk_free_fn=disk_free_fn, bootstrap=bootstrap, invocation=invocation, resolved=resolved)


def _run_configured(cfg: RunConfig, *, available_bytes_fn: Callable[[], int] | None = None, status_reserve_bytes: int = STATUS_RESERVE_BYTES,
                    disk_free_fn: Callable[[str], int] | None = None, bootstrap: dict[str, Any] | None = None, invocation: Invocation | None = None, resolved: Any = None) -> RunOutcome:
    try:
        cfg.revalidate()
    except RunConfigError as exc:
        return RunOutcome("refused", EXIT_REFUSED, exc.code, exc.detail, {})
    if resolved is not None:  # v2 production: the sealed roots/capsule root derived by the verified resolution complete the (still sealed) configuration
        from masy.native_water_run_config import resolved_config

        try:
            cfg = resolved_config(cfg, resolved.source, resolved.leaf_root)
        except RunConfigError as exc:
            return RunOutcome("refused", EXIT_REFUSED, exc.code, exc.detail, {})
    res = cfg.resources
    out = cfg.output
    try:
        outputs = RunOutputs(out["run_dir"], out["max_output_bytes"], out["max_file_bytes"], status_reserve_bytes, disk_free_fn=disk_free_fn)
    except RunOutputError as exc:
        return RunOutcome("refused", EXIT_REFUSED, "output_refused", str(exc), {})
    summary: dict[str, Any] = {"run_id": cfg.run_id, "config_sha256": cfg.config_sha256, "source_mode": cfg.source["mode"], "production_qualified": False}
    if resolved is not None:
        summary["production_runtime_verified"] = True  # byte/lock/capsule consistency of the named release; NOT independent scientific qualification (that is the published Root receipt)
    if bootstrap is not None:
        summary["bootstrap_identity_sha256"] = gate.bootstrap_identity_sha(bootstrap)
    try:
        cfg_boot[0] = bootstrap
        with _file_size_limit(max(int(out["max_file_bytes"]), 1)):  # EVERY writer of the process (native compile, partial temp files, checkpoint) is capped per file from here on
            outcome = _run(cfg, outputs, summary, _availability(res["host_available_cap_bytes"], available_bytes_fn), invocation, resolved)
    except RunRefused as exc:
        outcome = RunOutcome("refused", EXIT_REFUSED, exc.code, exc.detail, summary)
    except RunOutputError as exc:
        outcome = RunOutcome("failed", EXIT_FAILED, "output_failed", str(exc), summary)
    except Exception as exc:  # noqa: BLE001 - classified below; never swallowed silently
        outcome = _classify(exc, summary)
    try:
        raw, outcome = _status_bytes(outcome, outputs.status_reserve)
        outputs.write(STATUS_NAME, raw, status=True)
    except Exception as exc:  # noqa: BLE001 - status could not be written inside the reservation: report truthfully, never claim complete
        code = outcome.exit_code if outcome.exit_code else EXIT_FAILED
        outcome = RunOutcome("failed" if outcome.status == "complete" else outcome.status, code, outcome.code or "status_not_written", f"{outcome.detail}; status not written: {exc}", outcome.summary)
    finally:
        outputs.close()
    return outcome


def _classify(exc: Exception, summary: dict[str, Any]) -> RunOutcome:
    name, mod = type(exc).__name__, type(exc).__module__
    code = getattr(exc, "code", None) or name
    pre = summary.get("physical_steps", 0) == 0 and not summary.get("session_created")
    if pre:
        return RunOutcome("refused", EXIT_REFUSED, str(code), f"{mod}.{name}: {exc}", summary)
    if mod.startswith(("masy", "maple_syrup")) or isinstance(exc, MemoryError):
        return RunOutcome("failed", EXIT_FAILED, str(code), f"{mod}.{name}: {exc}", summary)
    return RunOutcome("failed", EXIT_INTERNAL, str(code), f"{mod}.{name}: {exc}", summary)


def _run(cfg: RunConfig, outputs: RunOutputs, summary: dict[str, Any], avail: Callable[[], int], invocation: Invocation | None = None, resolved: Any = None) -> RunOutcome:
    from maple_syrup import layered_case as lcase
    from maple_syrup import layered_event as ev
    from maple_syrup import native_case_profile as ncp
    from maple_syrup import native_dense_profile as nd

    from masy import native_water_calendar as nwc
    from masy.scenario import ScenarioError, load_scenario
    from masy.weather import WeatherError, latitude_binding_from_case_yaml

    res, lim = cfg.resources, cfg.limits
    if cfg.source["mode"] != "development_snapshot" and resolved is None:
        raise RunRefused("production_recipe_unavailable", "the worker runs only the verified development_snapshot mode or a v2 production resolution; production_committed never reaches it otherwise")
    guard = _ColdGuard(avail, res["caller_reserve_bytes"])
    io = IOLimits()
    shas: dict[str, str] = {}

    def store(rel: str, data: bytes, sha: str) -> None:
        if hashlib.sha256(data).hexdigest() != sha:
            raise RunRefused("input_snapshot_mismatch", f"internal: {rel} bytes do not hash to the recorded pin")
        try:
            outputs.write(f"inputs/{rel}", data)
        except RunOutputError as exc:
            raise RunRefused("output_budget", str(exc)) from None
        shas[rel] = sha

    # 1. declared source profile: manifests (parsed only now, after the cap is known and guarded), census + namespace + distribution BEFORE any physical object exists -----------
    capability = PROFILE_CAPABILITY.get(cfg.source["profile"])
    is_cuda = capability == "cuda"
    if "cupy" in sys.modules and not is_cuda:
        raise RunRefused("gpu_import", "CuPy is imported; this run path is CPU only")
    manifests = {}
    for label in ("masy", "syrup"):
        if resolved is not None:  # manifests derived from the VERIFIED sealed object identities (no caller-declared file)
            manifests[label] = dict(resolved.manifests[label])
            continue
        mpath = Path(cfg.source[f"{label}_manifest"])
        try:
            size = os.lstat(mpath).st_size
        except OSError as exc:
            raise RunRefused("manifest_unreadable", f"{label} manifest: {exc}") from None
        if int(size) > MANIFEST_MAX_BYTES:
            raise RunRefused("manifest_too_large", f"{label} manifest is {int(size)} B, over the hard {MANIFEST_MAX_BYTES}-B cap (checked before the body)")
        try:
            manifests[label] = load_manifest(mpath, f"source.{label}_manifest", max_bytes=MANIFEST_MAX_BYTES, admit=lambda n, w: guard.require(n, w))
        except RunConfigError as exc:
            raise RunRefused(exc.code, exc.detail) from None
    fence = _build_fence(cfg, manifests["masy"], manifests["syrup"], None if resolved is None else resolved.delivery)
    try:
        for label, root, manifest in (("masy", fence.masy_root, fence.masy_manifest), ("syrup", fence.syrup_root, fence.syrup_manifest)):
            guard.require(fence.census_limits.staging_bytes(len(manifest)), f"{label} source census staging")
            report = nwc.source_census(root, manifest, fence.census_limits)
            if not report["clean"]:
                raise RunRefused("source_census_failed", f"{label}: changed={report['changed'][:5]} missing={report['missing'][:5]} unexpected={report['unexpected'][:5]} unsafe={report['unsafe'][:5]}")
        if is_cuda:
            from masy import native_water_backend as nwb

            nwc._namespace_census(nwb.CUDA_V1, fence)
        else:
            nwc.namespace_census(fence)
        nwc.distribution_records(fence)
    except nwc.NativeWaterCalendarError as exc:
        raise RunRefused(exc.code, exc.detail) from None
    summary["source_profile"] = cfg.source["profile"]
    lrun = receipt = None
    if cfg.leaf is not None:  # v2: the genuine LEAF source receipt (MASY census, then the LEAF literal census BEFORE any LEAF import) right after the water-side census
        from masy import native_water_leaf_run as lrun

        try:
            device_kind = "cuda" if is_cuda else "cpu"
            receipt = (lrun.verify_receipt(cfg, manifests["masy"], avail, device_kind) if resolved is None
                       else lrun.verify_receipt_production(cfg, manifests["masy"], avail, device_kind, resolved.delivery))
        except lrun.LeafRunRefused as exc:
            raise RunRefused(exc.code, exc.detail) from None
        summary["leaf_source"] = {"profile": receipt.profile, "variant": receipt.variant, "leaf_package_digest": receipt.digests["leaf"],
                                  **({} if receipt.delivery is None else {"delivery_bundle_id": receipt.delivery["bundle_id"], "delivery_manifest_sha256": receipt.delivery["manifest_sha256"]})}
    if is_cuda and cfg.output["physical_checkpoint"] == "terminal_standalone":
        raise RunRefused("standalone_unsupported_on_cuda", "output.physical_checkpoint 'terminal_standalone' is the CPU Stage-B bundle; the CUDA profile supports none and combined_terminal")
    # 2. scenario (loads event/weather files itself; its pins must be the configuration's) -----------------------------------------------------------------------------------
    guard.require(MANIFEST_PARSE_FACTOR * io.max_file_bytes, "scenario/event/weather loading (one bounded document at a time)")
    try:
        scenario = load_scenario(Path(cfg.scenario))
    except ScenarioError as exc:
        raise RunRefused(getattr(exc, "code", "scenario_refused"), str(exc)) from None
    if scenario.source_sha256 != cfg.scenario_sha256:
        raise RunRefused("input_changed_after_pin", "the scenario file changed after the configuration pinned it")
    inv_info = _preauthenticate(cfg, outputs, scenario, invocation, guard, cuda=is_cuda)  # invocation controls + resume artifact seal/cursor/pause bound BEFORE any reference build
    if inv_info["resumed"]:
        _write_record(outputs, "resumed_from.json", {"schema": "masy.native_water_resumed_from.v1", "checkpoint": inv_info["checkpoint"], "seal_sha256": inv_info["seal"],
                                                     "next_index": inv_info["start_next_index"], "role": inv_info["role"], "last_kind": inv_info["last_kind"], "finished": inv_info["finished"],
                                                     "cursor_digest": inv_info["cursor_digest"], "prior_lineage": inv_info["lineage"], "run_id": cfg.run_id, "config_sha256": cfg.config_sha256,
                                                     "options_sha256": invocation.options_sha256, "note": "the checkpoint is consumed read-only; no copy is made"})
    # 3. byte-exact bounded snapshots of every input; each stored copy is hashed from the bytes actually read and compared with its pin ---------------------------------------------
    store("run_config" + Path(cfg.config_path).suffix, cfg.config_bytes, cfg.config_sha256)
    case_orig = Path(cfg.maple_case_dir)
    for name, pin in sorted(cfg.case_files.items()):
        data, sha = _read_input(case_orig / name, io.max_file_bytes, guard, f"case file {name}", outputs, f"inputs/case/{name}")
        if sha != pin:
            raise RunRefused("case_changed_after_pin", f"maple_case_dir/{name} changed after the configuration was loaded")
        store(f"case/{name}", data, sha)
    data, sha = _read_input(Path(cfg.scenario), io.max_file_bytes, guard, "scenario", outputs, "inputs/scenario/" + Path(cfg.scenario).name)
    if sha != cfg.scenario_sha256:
        raise RunRefused("input_changed_after_pin", "the scenario file changed after the configuration pinned it")
    store("scenario/" + Path(cfg.scenario).name, data, sha)
    base = Path(scenario.base_dir)
    for occ in scenario.occurrences:
        if "scenario/" + occ.file in shas:
            continue
        data, sha = _read_input(base / occ.file, io.max_file_bytes, guard, f"event file {occ.file}", outputs, "inputs/scenario/" + occ.file)
        if sha != occ.file_sha256:
            raise RunRefused("input_changed_after_pin", f"event file {occ.file} changed after the scenario pinned/loaded it")
        store("scenario/" + occ.file, data, sha)
    if scenario.weather_ref is not None:
        wfile = scenario.weather_ref["file"]
        data, sha = _read_input(base / wfile, io.max_file_bytes, guard, f"weather file {wfile}", outputs, "inputs/scenario/" + wfile)
        if sha != scenario.weather_ref["sha256"]:
            raise RunRefused("input_changed_after_pin", f"weather file {wfile} changed after the scenario pinned/loaded it")
        store("scenario/" + wfile, data, sha)
    leaf_bytes = None
    if cfg.leaf is not None:  # the three LEAF inputs, each read once under its OWN declared cap (admitted), snapshotted into the run directory and checked by the stdlib I/O control BEFORE any plant array
        lio = cfg.leaf["io"]
        got = {}
        for key, rel, cap in (("catalog", "leaf/" + Path(cfg.leaf["catalog"]).name, lio["max_catalog_bytes"]), ("plants", "leaf/" + Path(cfg.leaf["plants"]).name, lio["max_plants_bytes"]),
                              ("plants_metadata", "leaf/" + Path(cfg.leaf["plants_metadata"]).name, lio["max_catalog_bytes"])):
            data, sha = _read_input(Path(cfg.leaf[key]), cap, guard, f"leaf {key}", outputs, "inputs/" + rel)
            store(rel, data, sha)
            got[key] = data
        try:
            summary["leaf_inputs"] = lrun.plant_file_control(got["plants"], got["plants_metadata"], lio, cfg.leaf["pins"], cfg.leaf["pins"]["plants_sha256"], cfg.leaf["pins"]["metadata_sha256"])
        except lrun.LeafRunRefused as exc:
            raise RunRefused(exc.code, exc.detail) from None
        if hashlib.sha256(got["catalog"]).hexdigest() != cfg.leaf["pins"]["catalog_sha256"]:
            raise RunRefused("input_changed_after_pin", "the leaf catalog changed after the configuration pinned it")
        leaf_bytes = SimpleNamespace(**got)
    # 4. the SYRUP layered parameter file: read ONCE (bounded), pinned, parsed from those very bytes; every referenced .npy map is its immutable MapRef content -------------------
    params_path = Path(cfg.syrup_parameters)
    limits_l = lcase.LayeredCaseLimits()
    data, sha = _read_input(params_path, limits_l.max_yaml_bytes, guard, "layered parameter file", outputs, "inputs/syrup_parameters" + params_path.suffix)
    if sha != cfg.syrup_parameters_sha256:
        raise RunRefused("input_changed_after_pin", "the layered parameter file changed after the configuration pinned it")
    try:
        _parse = lcase.parse_layered_cuda_config if is_cuda else lcase.parse_layered_config  # the file's compute.backend is the ONLY backend choice; the loader is chosen by the profile capability
        config = _parse(data, base_dir=params_path.parent, limits=limits_l, source_path=str(params_path), available_bytes_fn=avail, reserve_bytes=res["caller_reserve_bytes"])
    except lcase.LayeredCaseError as exc:
        raise RunRefused(type(exc).__name__, str(exc)) from None
    store("syrup_parameters" + params_path.suffix, data, sha)
    del data
    pins = dict(cfg.syrup_map_pins)
    maps = {m.name: m for m in config.maps()}
    if set(maps) != set(pins):
        raise RunRefused("map_pin_mismatch", f"the parameter file references maps {sorted(maps)} but the configuration pinned {sorted(pins)}")
    m = raw = None
    for name in sorted(maps):
        m = maps[name]
        raw = m.array
        while raw is not None and type(raw) is not bytes:  # np.frombuffer(bytes).reshape(): follow the base chain to the immutable file content the factory actually uses
            raw = getattr(raw, "base", None)
        if m.sha256 != pins[name]:
            raise RunRefused("input_changed_after_pin", f"map {name} changed after the configuration pinned it")
        if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != m.sha256:
            raise RunRefused("map_snapshot_unavailable", f"map {name}: the factory's immutable content is not available for a byte-exact copy")
        store(f"syrup_maps/{name}", raw, m.sha256)
    # 5. latitude from the run-owned case copy; the original path is recorded too ---------------------------------------------------------------------------------------------------
    case_dir = str(outputs.path / "inputs" / "case")
    try:
        latitude = latitude_binding_from_case_yaml(Path(case_dir) / "case.yaml", IOLimits())
    except WeatherError as exc:
        raise RunRefused(getattr(exc, "code", "latitude_refused"), str(exc)) from None
    # 6. actual MAPLE case -> sealed Dense preparation. The native compile has NO per-member output planner: its outputs are held against the budget up front
    #    (`compile_output_reserve_bytes`), every write of the process is capped to reserve // COMPILE_MAX_MEMBERS (RLIMIT_FSIZE) meanwhile, and the real size is measured after -----------
    dev = None
    try:
        if is_cuda:  # the CUDA source capability with the SAME literal full-water digest the fence names; a CPU profile never reaches the CUDA factory
            profile = ncp.verify_source_profile(cfg.source["native_target"], expected_syrup_digest=nwc.SOURCE_PROFILES[cfg.source["profile"]], profile=ncp.CUDA_SOURCE_PROFILE_NAME)
        elif cfg.source["profile"] == "native9f_layered_factory_afca_cpu_v1":
            profile = ncp.verify_source_profile(cfg.source["native_target"], expected_syrup_digest=nwc.SOURCE_PROFILES[cfg.source["profile"]])
        else:
            profile = ncp.verify_source_profile(cfg.source["native_target"])  # historical profiles: unchanged call
        reserve = int(cfg.output["compile_output_reserve_bytes"])
        try:
            outputs.reserve(reserve, "the native case compile outputs")
        except RunOutputError as exc:
            raise RunRefused("output_budget", str(exc)) from None
        try:
            with _file_size_limit(max(min(reserve // COMPILE_MAX_MEMBERS, int(cfg.output["max_file_bytes"])), 1)):
                admission = ncp.admit_package_case(profile, case_dir)
        finally:
            actual = outputs.untracked_bytes()
            try:
                outputs.settle(reserve, actual, "the native case compile")
            except RunOutputError as exc:
                raise RunRefused("output_budget", str(exc)) from None
        if is_cuda:
            from masy import native_water_cuda as nwcu
            from maple_syrup import layered_case_cuda as lcc
            from maple_syrup import native_dense_cuda_profile as ndc

            if config.backend != "cuda":
                raise RunRefused("backend_profile_mismatch", f"the water file's compute.backend is {config.backend!r} but the CUDA profile requires 'cuda' (the file is never relabelled)")
            try:
                dev = nwcu.CudaPlacement(**dict(cfg.cuda))
            except nwcu.CudaPlacementError as exc:
                raise RunRefused(exc.code, exc.detail) from None
            dev_avail = _device_availability(dev)
            # PURE metadata plan BEFORE any device transfer (no CuPy, no device query); the declared device cap bounds the planned device requirement
            try:
                cplan = lcc.plan_layered_cuda_setup_from_admission(
                    admission, config, device_id=dev.device_id, caller_host_reserve_bytes=res["caller_reserve_bytes"], caller_device_reserve_bytes=dev.device_reserve_bytes,
                    lifecycle_extra_reserve_bytes=res["lifecycle_extra_reserve_bytes"], native_reserve_bytes=dev.native_reserve_bytes, host_available_bytes_fn=avail)
            except lcase.LayeredCaseError as exc:
                raise RunRefused(type(exc).__name__, str(exc)) from None
            if int(cplan["device_required_bytes"]) + dev.device_reserve_bytes > dev.device_cap_bytes:
                raise RunRefused("device_cap_exceeded", f"the planned device requirement {int(cplan['device_required_bytes'])} B + reserve {dev.device_reserve_bytes} B exceeds cuda.device_cap_bytes {dev.device_cap_bytes}")
            prep = ndc.prepare_native_dense_cuda_case(admission, device_id=dev.device_id, caller_host_reserve_bytes=res["preparation_reserve_bytes"],
                                                      caller_device_reserve_bytes=dev.device_reserve_bytes, host_available_bytes_fn=avail, device_available_bytes_fn=dev_avail)
        else:
            prep = nd.prepare_native_dense_case(admission, caller_reserve_bytes=res["preparation_reserve_bytes"], available_bytes_fn=avail)
    except (ncp.NativeCaseProfileError, nd.NativeDenseProfileError) as exc:
        raise RunRefused("native_case_refused", str(exc)) from None
    except Exception as exc:  # noqa: BLE001 - only the CUDA preparation's own typed error is converted
        if is_cuda:
            from maple_syrup import native_dense_cuda_profile as ndc_

            if isinstance(exc, ndc_.NativeDenseCudaProfileError):
                raise RunRefused("native_case_refused", str(exc)) from None
        raise
    # 7. configured layered-water factory (explicit soil layers/properties/cover/initial/domain; retained texture) ----------------------------------------------------
    try:
        if is_cuda:
            setup = lcc.prepare_layered_cuda_case(prep, config, caller_host_reserve_bytes=res["caller_reserve_bytes"], caller_device_reserve_bytes=dev.device_reserve_bytes,
                                                  lifecycle_extra_reserve_bytes=res["lifecycle_extra_reserve_bytes"], native_reserve_bytes=dev.native_reserve_bytes,
                                                  host_available_bytes_fn=avail, device_available_bytes_fn=dev_avail)
        else:
            setup = lcase.prepare_layered_case(prep, config, caller_reserve_bytes=res["caller_reserve_bytes"],
                                               lifecycle_extra_reserve_bytes=res["lifecycle_extra_reserve_bytes"], available_bytes_fn=avail)
    except lcase.LayeredCaseError as exc:
        raise RunRefused(type(exc).__name__, str(exc)) from None
    if setup.identities["backend"] not in (("cuda",) if is_cuda else ("numpy", "numba")):
        raise RunRefused("unsupported_backend", "the genuine CUDA setup only" if is_cuda else "NumPy and Numba only")
    if is_cuda and setup.identities["device_id"] != dev.device_id:
        raise RunRefused("device_mismatch", "the CUDA setup's device differs from cuda.device_id")
    if setup.identities["latitude_deg"] != latitude.latitude_deg:
        raise RunRefused("latitude_join_mismatch", "the layered factory's native latitude differs from the MASY case.yaml binding")
    layered_cfg = dict(setup.identities["config"])
    if is_cuda:  # host-retained products only (device bytes are never added to the host retained figure): texture + topology + graph static metadata + parameter maps
        act = setup.plan["actual"]
        retained = int(act["texture_host_bytes"]) + int(act["topology_host_bytes"]) + int(act["graph_host_bytes"]) + sum(int(m.array.nbytes) for m in config.maps())
        layer_plan = {"plan_required_bytes": {"host": int(setup.plan["host_required_bytes"]), "device": int(setup.plan["device_required_bytes"])}, "plan_stage_bytes": setup.plan["stage_bytes"],
                      "topology_graph_bytes_actual": {k: int(v) for k, v in act.items()}, "n_cells": int(setup.lifecycle_operator.n_cells), "n_layers": int(setup.lifecycle_operator.n_layers),
                      "n_classes": int(setup.lifecycle_operator.n_classes), "texture_source_sha256": setup.identities["texture_source_sha256"]}
    else:
        retained = int(setup.texture.memory.output_bytes) + int(setup.plan["topology_graph_bytes_actual"]) + sum(int(m.array.nbytes) for m in config.maps())
        layer_plan = {"plan_required_bytes": setup.plan["required_bytes"], "plan_stage_bytes": setup.plan["stage_bytes"], "topology_graph_bytes_actual": setup.plan["topology_graph_bytes_actual"],
                      "n_cells": setup.plan["n_cells"], "n_layers": setup.plan["n_layers"], "n_classes": setup.plan["n_classes"], "texture_source_sha256": setup.identities["texture_source_sha256"]}
    control = ev.LayeredEventControl(max_dt_s=lim["storm"]["max_dt_s"], min_dt_s=lim["storm"]["min_dt_s"], max_retries=lim["storm"]["max_retries"],
                                     max_steps=lim["storm"]["max_steps"], max_recession_s=lim["storm"]["max_recession_s"])
    limits = nwc.NativeCalendarLimits(
        interevent_max_dt_s=lim["interevent_max_dt_s"], interevent_min_dt_s=lim["interevent_min_dt_s"], interevent_max_retries=lim["interevent_max_retries"],
        storm_control=control, caller_reserve_bytes=res["caller_reserve_bytes"], temporal_reserve_bytes=res["temporal_reserve_bytes"],
        max_occurrences=lim["max_occurrences"], max_dates_per_interevent=lim["max_dates_per_interevent"], max_record_bytes=lim["max_record_bytes"],
        max_accepted_steps_per_occurrence=lim["max_accepted_steps_per_occurrence"], call_step_budget=lim["call_step_budget"])
    built = ref_op = None
    cold_log: list = []
    if cfg.leaf is not None:  # v2: the LEAF participant over THE SAME water lifecycle operator (create) / the REFERENCE operator (resume: the initial plants are not read)
        try:
            if inv_info["resumed"]:
                ref_op = lrun.reference_operator(cfg, receipt, leaf_bytes.catalog, setup.lifecycle_operator, dev, cold=cold_log)
            else:
                built = lrun.build_participant(cfg, receipt, leaf_bytes.catalog, leaf_bytes.plants, leaf_bytes.plants_metadata, scenario, prep, setup.lifecycle_operator, setup.state, dev, cold=cold_log)
        except lrun.LeafRunRefused as exc:
            raise RunRefused(exc.code, exc.detail) from None
        del leaf_bytes
    try:
        if cfg.leaf is not None:
            from masy import native_water_backend as nwb
            from masy import native_water_checkpoint as nck
            from maple_syrup import layered_checkpoint as lcp

            if inv_info["resumed"]:
                reading = avail()
                ck_limits = lcp.CheckpointLimits(max_serialized_bytes=max(outputs.budget, 1), max_host_bytes=max(reading, 1), max_device_bytes=dev.device_cap_bytes if is_cuda else 0,
                                                 max_scratch_bytes=max(reading, 1))
                kw = dict(checkpoint_dir=inv_info["checkpoint"], scenario=scenario, preparation=prep, lop=setup.lifecycle_operator, reference_state=setup.state, fence=fence, latitude_live=latitude,
                          limits=limits, profile_id=cfg.profile_id, checkpoint_limits=ck_limits, leaf_op=ref_op, leaf_limits=nwc.LeafCheckpointLimits(**lrun.leaf_limits(cfg, is_cuda)),
                          retained_native_bytes=retained, inputs_sha256=_input_pins(cfg), expected_seal_sha256=inv_info["seal"], host_available_bytes_fn=avail)
                session = nwc.NativeWaterSession.resume_leaf_cuda(placement=dev, **kw) if is_cuda else nwc.NativeWaterSession.resume_leaf(**kw)
            else:
                kw = dict(scenario=scenario, lop=setup.lifecycle_operator, state=setup.state, preparation=prep, fence=fence, latitude=latitude, limits=limits, profile_id=cfg.profile_id,
                          participant=nwb.LEAF_WATER_V1, leaf_op=built.op, leaf_boundary=built.boundary, retained_native_bytes=retained)
                session = nwc.NativeWaterSession.create_leaf_cuda(placement=dev, **kw) if is_cuda else nwc.NativeWaterSession.create_leaf(**kw)
        elif inv_info["resumed"]:
            from masy import native_water_checkpoint as nck
            from maple_syrup import layered_checkpoint as lcp

            reading = avail()
            ck_limits = lcp.CheckpointLimits(max_serialized_bytes=max(outputs.budget, 1), max_host_bytes=max(reading, 1), max_device_bytes=dev.device_cap_bytes if is_cuda else 0,
                                             max_scratch_bytes=max(reading, 1))
            if is_cuda:
                session = nwc.NativeWaterSession.resume_cuda(checkpoint_dir=inv_info["checkpoint"], scenario=scenario, preparation=prep, lop=setup.lifecycle_operator,
                                                             reference_state=setup.state, fence=fence, latitude_live=latitude, limits=limits, profile_id=cfg.profile_id,
                                                             checkpoint_limits=ck_limits, placement=dev, retained_native_bytes=retained, inputs_sha256=_input_pins(cfg),
                                                             expected_seal_sha256=inv_info["seal"], host_available_bytes_fn=avail)
            else:
                session = nwc.NativeWaterSession.resume(checkpoint_dir=inv_info["checkpoint"], scenario=scenario, preparation=prep, lop=setup.lifecycle_operator, reference_state=setup.state,
                                                        fence=fence, latitude_live=latitude, limits=limits, profile_id=cfg.profile_id, checkpoint_limits=ck_limits,
                                                        retained_native_bytes=retained, inputs_sha256=_input_pins(cfg), expected_seal_sha256=inv_info["seal"], host_available_bytes_fn=avail)
        elif is_cuda:
            session = nwc.NativeWaterSession.create_cuda(scenario=scenario, lop=setup.lifecycle_operator, state=setup.state, preparation=prep, fence=fence, latitude=latitude,
                                                         limits=limits, profile_id=cfg.profile_id, placement=dev, retained_native_bytes=retained)
        else:
            session = nwc.NativeWaterSession.create(scenario=scenario, lop=setup.lifecycle_operator, state=setup.state, preparation=prep, fence=fence, latitude=latitude,
                                                    limits=limits, profile_id=cfg.profile_id, retained_native_bytes=retained)
    except nwc.NativeWaterCalendarError as exc:
        raise RunRefused(exc.code, exc.detail) from None
    except Exception as exc:  # noqa: BLE001 - only the typed checkpoint refusals are converted (everything else keeps its classification)
        from masy import native_water_checkpoint as nck_

        if isinstance(exc, nck_.CheckpointError):
            raise RunRefused(exc.code, exc.detail) from None
        raise
    summary["session_created"] = True
    if cfg.leaf is not None:
        summary["leaf"] = lrun.biology_summary(session)
        create = {"cold_admission": list(cold_log)}  # run-specific create/resume-time facts live apart from the bounded biology summary (which is comparable across paused and uninterrupted runs)
        if built is not None:
            create.update(n_base_initial=built.n_base, n_live_initial=built.n_live, resolved_anchors=built.resolved, catalog_hash=built.catalog_hash, masked=built.masked, canopy_valid_area_m2=built.canopy_valid_area_m2)
        summary["leaf_create"] = create
        built = ref_op = None  # the initial objects are released (the accepted generation lives only in the session pair)
    resolved = {"schema": "masy.native_water_run_resolved.v1", "run": cfg.record(), "inputs_sha256": dict(sorted(shas.items())), "fence": fence.record(),
                "fence_identity": fence.identity(), "source_profile": cfg.source["profile"],
                "native_ids": {"native_package": prep.package_digest, "native_bundle": prep.bundle_id, "syrup_package": prep.syrup_package_digest},
                "case_identity_sha256": prep.case_identity_sha256, "case_route": prep.route,
                "case_dir_original": str(case_orig), "case_dir_run_owned": case_dir,
                "latitude": {"deg": latitude.latitude_deg, "source": dict(latitude.source), "native_binding": prep.latitude_binding()},
                "layered": {"identities": {k: v for k, v in dict(setup.identities).items() if k != "config"}, "config": layered_cfg, **layer_plan},
                "snapshot_agreement": {"scenario_sha256": scenario.source_sha256, "syrup_parameters_sha256": cfg.syrup_parameters_sha256,
                                       "map_sha256": {n: maps[n].sha256 for n in sorted(maps)}},
                "retained_native_bytes_declared": retained, "session_exports": session.exports,
                "scenario": {"scenario_id": scenario.scenario_id, "digest": scenario.digest, "n_occurrences": len(scenario.occurrences)},
                "bootstrap": cfg_boot[0], "source_mode": cfg.source["mode"], "production_qualified": False,
                "initial_state_digest": session.state.digest, "limits": limits.record(), "note": "standalone configured water-only CPU run; no combined calendar+physical resume"}
    if cfg.leaf is not None:
        resolved["leaf"] = {"variant": receipt.variant, "profile": receipt.profile, "leaf_package_digest": receipt.digests["leaf"], "pins": dict(cfg.leaf["pins"]), "static_pins": lrun.static_pins(cfg),
                            "cadence_days": cfg.leaf["cadence_days"], "geometry": dict(cfg.leaf["geometry"]), "checkpoint_limits": dict(cfg.leaf["budgets"]["checkpoint"]),
                            "cold_admission": list(cold_log), "note": "LEAF participant (leaf_water_v1): one authoritative SYRUP water budget; biology restores only from the joint artifact"}
        resolved["note"] = "configured LEAF+water run (development snapshot; UNQUALIFIED); joint calendar+physical+biology checkpoint/resume per the invocation record"
    if is_cuda:
        resolved["cuda"] = dict(cfg.cuda)
        resolved["note"] = "configured water-only CUDA run (development snapshot; UNQUALIFIED until the Root hardware matrix); no combined calendar+physical resume unless the invocation record says so"
    if invocation is not None or cfg.output["physical_checkpoint"] == "combined_terminal":
        resolved["note"] = f"configured water-only {'CUDA' if is_cuda else 'CPU'} run; combined calendar+physical checkpoint/resume per the invocation record (see summary.combined_checkpoint)"
    # RELEASE: keep only what the run still needs (the lifecycle operator with its texture and the current state in the session); everything initial/duplicate is dropped
    kept = SimpleNamespace(lop=session.lop, texture=setup.texture)  # session.lop IS setup.lifecycle_operator for a fresh run; after a resume it is the loaded operator (the reference one is dropped)
    watch = {"initial_state": weakref.ref(setup.state), "preparation": weakref.ref(prep), "admission": weakref.ref(admission),
             "setup": weakref.ref(setup), "layered_config": weakref.ref(config)}
    if hasattr(setup, "cover"):  # the CPU setup retains an initial cover array; the CUDA setup has none (its cover lives only in the lifecycle's private device copy)
        watch["initial_cover"] = weakref.ref(setup.cover)
    for n, m in maps.items():
        watch[f"map:{n}"] = weakref.ref(m.array)
    del setup, config, prep, admission, profile, maps, layered_cfg, m, raw
    gc.collect()
    summary["released_after_create"] = {k: v() is None for k, v in watch.items()}
    if invocation is not None:  # old default runs keep a byte-identical resolved.json
        resolved["invocation"] = {"resume": inv_info["resumed"], "resume_seal_sha256": inv_info["seal"], "pause_after": invocation.pause_after, "start_next_index": inv_info["start_next_index"],
                                  "options_sha256": invocation.options_sha256, "resume_receipt": session.resume_receipt}
    _write_record(outputs, "resolved.json", resolved)
    # 8. steps -------------------------------------------------------------------------------------------------------------------------------------------------------------
    cadence, max_lines = cfg.output["step_cadence"], cfg.output["max_step_lines"]
    steps, lines, calls, incomplete = 0, 0, 0, None
    start_idx = int(inv_info["start_next_index"])
    pause = None if invocation is None else invocation.pause_after
    while not session.finished:
        if pause is not None and incomplete is None and start_idx + steps >= pause:
            break  # a clean pause ONLY between accepted whole occurrences (never after an `incomplete` call); the pause count is the cumulative accepted total, never a budget reset
        calls += 1
        if calls > max_lines:
            raise RunOutputError(f"more than {max_lines} step calls (output.max_step_lines bounds the call count and the stream)")
        result = session.step()
        summary["physical_steps"] = summary.get("physical_steps", 0) + 1
        if calls == 1:
            gc.collect()
            summary["released_after_first_step"] = {k: v() is None for k, v in watch.items()}
        if result.status == "incomplete":
            incomplete = result.occurrence_id
            outputs.append("steps.jsonl", _json({"call": calls, "occurrence_id": result.occurrence_id, "status": "incomplete", "evidence": result.evidence}))
            lines += 1
            continue
        steps += 1
        incomplete = None
        if steps % cadence == 0 or session.finished:
            st = session.state
            outputs.append("steps.jsonl", _json({"call": calls, "step": steps, "occurrence_id": result.occurrence_id, "status": "complete", "record": result.record,
                                                 "evidence": result.evidence, "role": st.role, "t_s_hex": float(st.t_s).hex(), "state_digest": st.digest,
                                                 "lifetime": st.lifetime.as_dict()}))
            lines += 1
    if cfg.leaf is not None:
        summary["leaf"] = lrun.biology_summary(session)
    final = session.state
    ledger = final.lifetime.as_dict()
    gc.collect()
    summary.update({"accepted_steps": steps, "calls": calls, "step_lines": lines, "final_role": final.role, "final_t_s_hex": float(final.t_s).hex(),
                    "final_state_digest": final.digest, "lifetime": ledger, "realized_occurrences": len(session.realized),
                    "released_at_end": {k: v() is None for k, v in watch.items()}})
    # 9. optional STANDALONE physical terminal checkpoint (Stage-B native9f codec; never a combined resume). Its size is PLANNED by the codec's own estimator before any write --------
    if cfg.output["physical_checkpoint"] == "terminal_standalone":
        from maple_syrup import layered_checkpoint as lcp
        from maple_syrup import lifecycle_checkpoint_v2 as lk2

        try:
            payload = final.payload
            bed = payload.water.bed if final.role == "storm" else payload.bed
            extra = max(int(final.binding.event.required_bytes) - int(final.binding.op.memory.required_bytes), 0) if final.role == "storm" else 0
            plan = lk2.estimate_lifecycle_budget_v2(kept.lop, final.role, profile=lk2.NATIVE9F_PROFILE_V2, bed=bed, event_extra_bytes=extra)
            del payload, bed
        except Exception as exc:  # noqa: BLE001
            raise RunOutputError(f"standalone physical checkpoint cannot be planned: {type(exc).__name__}: {exc}") from None
        planned = int(plan.serialized_bytes) + int(plan.manifest_cap_bytes)
        if planned > outputs.budget:
            raise RunOutputError(f"the standalone physical checkpoint needs up to {planned} B but only {outputs.budget} B of the output budget remain; nothing was written")
        reading = avail()
        limits_ck = lcp.CheckpointLimits(max_serialized_bytes=max(outputs.budget, 1), max_host_bytes=max(reading, 1), max_device_bytes=0, max_scratch_bytes=max(reading, 1))
        ident = {"kind": "standalone_physical_checkpoint", "run_id": cfg.run_id, "config_sha256": cfg.config_sha256}
        outputs.reserve(planned, "the standalone physical checkpoint")
        try:
            try:
                with _file_size_limit(min(cfg.output["max_file_bytes"], max(planned, 1))):
                    rec = lk2.save_lifecycle_checkpoint_v2(str(outputs.path / "physical_terminal_checkpoint_standalone"), kept.lop, final, ident,
                                                          profile=lk2.NATIVE9F_PROFILE_V2, limits=limits_ck, host_available_bytes_fn=avail)
            except Exception as exc:  # noqa: BLE001
                raise RunOutputError(f"standalone physical checkpoint failed: {type(exc).__name__}: {exc}") from None
        finally:
            actual = outputs.untracked_bytes()
            outputs.settle(planned, actual, "the standalone physical checkpoint")
        summary["standalone_physical_checkpoint"] = {"path": "physical_terminal_checkpoint_standalone", "profile": rec.profile, "role": rec.role,
                                                     "manifest_sha256": rec.manifest_sha256, "planned_bytes": planned, "written_bytes": actual,
                                                     "label": "standalone physical lifecycle checkpoint; NOT a combined calendar+physical resume"}
    paused = pause is not None and incomplete is None and not session.finished
    end_idx = start_idx + steps
    if invocation is not None:
        summary["invocation"] = {"resumed": inv_info["resumed"], "resume_seal_sha256": inv_info["seal"], "pause_after": pause, "start_next_index": start_idx, "end_next_index": end_idx,
                                 "resume_of_finished": bool(inv_info["resumed"] and inv_info.get("finished") and steps == 0), "options_sha256": invocation.options_sha256,
                                 "resume_receipt": session.resume_receipt}
    # 10. optional COMBINED calendar+physical resumable checkpoint (R1-R2 artifact; distinct from the standalone bundle and from any cursor-only record) ----------------------------------
    if cfg.output["physical_checkpoint"] == "combined_terminal":
        _write_combined(cfg, outputs, summary, avail, session, incomplete, invocation, inv_info, end_idx)
    if incomplete is not None:
        return RunOutcome("incomplete", EXIT_FAILED, "event_incomplete", incomplete, summary)
    if paused:
        return RunOutcome("paused", EXIT_PAUSED, None, None, summary)
    return RunOutcome("complete", EXIT_OK, None, None, summary)


def _write_combined(cfg: RunConfig, outputs: RunOutputs, summary: dict[str, Any], avail: Callable[[], int], session: Any, incomplete: str | None, invocation: Invocation | None,
                    inv_info: dict[str, Any], end_idx: int) -> None:
    """Write `<run_dir>/combined_checkpoint/` for the accepted boundary the session holds. Never with a pending incomplete occurrence (no false checkpoint). Planned by the session's
    whole-call `plan_checkpoint()` and RESERVED in the output budget before any write; the real bytes are settled afterwards. Any failure is a typed `RunOutputError` (status failed)
    that records the truth (including the publisher's removed/left report) in the summary."""
    from masy import native_water_checkpoint as nck
    from maple_syrup import layered_checkpoint as lcp

    kind = "calendar_physical_accepted_occurrence_boundary"
    label = "combined calendar+physical resumable checkpoint; NOT the standalone physical bundle and NOT a cursor-only record"
    if incomplete is not None:
        summary["combined_checkpoint"] = {"written": False, "reason": f"occurrence {incomplete!r} is incomplete; a partial occurrence is never checkpointed", "kind": kind}
        return
    if cfg.leaf is not None:
        kind = "calendar_physical_biology_accepted_occurrence_boundary"
        label = "JOINT calendar+physical+biology resumable checkpoint (physical/, biology/, temporal.json, sealed manifest)"
    try:
        plan = session.plan_checkpoint_leaf() if cfg.leaf is not None else session.plan_checkpoint()
    except nck.CheckpointError as exc:
        summary["combined_checkpoint"] = {"written": False, "error": f"{exc.code}: {exc.detail}"[:1024], "kind": kind}
        raise RunOutputError(f"combined checkpoint cannot be planned: {exc.code}: {exc.detail}") from None
    planned = int(plan["disk_bytes"])
    try:
        outputs.reserve(planned, "the combined checkpoint")
    except RunOutputError as exc:
        summary["combined_checkpoint"] = {"written": False, "error": str(exc)[:1024], "planned_bytes": planned, "kind": kind}
        raise
    reading = int(avail())
    limits_ck = lcp.CheckpointLimits(max_serialized_bytes=max(planned, 1), max_host_bytes=max(reading, 1),
                                     max_device_bytes=0 if cfg.cuda is None else int(cfg.cuda["device_cap_bytes"]), max_scratch_bytes=max(reading, 1))
    prior = inv_info.get("lineage") or {}
    depth = (int(prior["chain_depth"]) + 1) if inv_info["resumed"] and type(prior.get("chain_depth")) is int else (1 if inv_info["resumed"] else 0)
    lineage = {"run_id": cfg.run_id, "config_sha256": cfg.config_sha256, "options_sha256": None if invocation is None else invocation.options_sha256,
               "resumed_from_seal": inv_info["seal"], "chain_depth": depth}
    rec = None
    failure: Exception | None = None
    try:
        with _file_size_limit(max(min(int(cfg.output["max_file_bytes"]), planned), 1)):
            if cfg.leaf is not None:
                from masy import native_water_calendar as nwc_
                from masy import native_water_leaf_run as lrun_

                rec = session.checkpoint_leaf(str(outputs.path / "combined_checkpoint"), checkpoint_limits=limits_ck, leaf_limits=nwc_.LeafCheckpointLimits(**lrun_.leaf_limits(cfg, cfg.cuda is not None)),
                                              host_available_bytes_fn=avail, inputs_sha256=_input_pins(cfg), lineage=lineage)
            else:
                rec = session.checkpoint(str(outputs.path / "combined_checkpoint"), checkpoint_limits=limits_ck, host_available_bytes_fn=avail, inputs_sha256=_input_pins(cfg), lineage=lineage)
    except Exception as exc:  # noqa: BLE001 - reported truthfully below
        failure = exc
    finally:
        actual = outputs.untracked_bytes()
        try:
            outputs.settle(planned, actual, "the combined checkpoint")
        except RunOutputError as exc:
            if failure is None:
                failure = exc
    if failure is not None:
        cleanup = getattr(failure, "cleanup", None)
        summary["combined_checkpoint"] = {"written": False, "error": f"{type(failure).__name__}: {failure}"[:1024], "planned_bytes": planned, "left_on_disk_bytes": actual, "cleanup": cleanup, "kind": kind,
                                          "note": "an unsealed partial, if listed under cleanup.left, is NOT loadable (no sealed manifest) and was left because its ownership could not be proved"}
        raise RunOutputError(f"combined checkpoint failed: {type(failure).__name__}: {failure}") from None
    summary["combined_checkpoint"] = {"written": True, "path": "combined_checkpoint", "kind": kind, "label": label, "seal_sha256": rec.manifest_sha256,
                                      "physical_manifest_sha256": rec.physical_manifest_sha256, "cursor_digest": rec.cursor_digest, "state_digest": rec.state_digest,
                                      "physical_digest": rec.physical_digest, "next_index": rec.next_index, "role": rec.role, "planned_bytes": planned, "written_bytes": actual,
                                      "plan_total_host_bytes": int(rec.plan["total"]), "lineage": lineage}


def _write_record(outputs: RunOutputs, rel: str, value: Any) -> None:
    try:
        outputs.write(rel, _json(value))
    except RunOutputError as exc:
        raise RunRefused("output_budget", str(exc)) from None


def _refuse_print(code: str, detail: str) -> int:
    print(RESULT_PREFIX + json.dumps({"status": "refused", "exit_code": EXIT_REFUSED, "code": code, "detail": detail, "summary": {}}))
    return EXIT_REFUSED


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    expected = boot = None
    invocation = None
    if len(argv) == 5 and argv[1] == "--expected-config-sha256" and argv[3] == "--expected-bootstrap-sha256":
        argv, expected, boot = [argv[0]], argv[2], argv[4]
    elif len(argv) == 3 and argv[1] == "--expected-config-sha256":
        argv, expected = [argv[0]], argv[2]
    elif len(argv) > 5 and (len(argv) - 1) % 2 == 0:
        flags = {}
        for flag, value in zip(argv[1::2], argv[2::2]):
            if flag not in ("--expected-config-sha256", "--expected-bootstrap-sha256", "--expected-options-sha256", "--resume", "--expected-seal-sha256", "--pause-after-occurrences") or flag in flags:
                return _refuse_print("usage", "unknown or repeated worker option")
            flags[flag] = value
        if "--expected-options-sha256" not in flags or "--expected-config-sha256" not in flags or "--expected-bootstrap-sha256" not in flags:
            return _refuse_print("options_identity_required", "invocation controls need the parent's options identity together with the config and bootstrap identities")
        pause = None
        if "--pause-after-occurrences" in flags:
            try:
                pause = int(flags["--pause-after-occurrences"])
            except ValueError:
                return _refuse_print("invalid_options", "--pause-after-occurrences must be an integer")
            if pause < 0:
                return _refuse_print("invalid_options", "--pause-after-occurrences must be >= 0")
        resume, seal = flags.get("--resume"), flags.get("--expected-seal-sha256")
        if resume is None and (seal is not None or pause is None):
            return _refuse_print("invalid_options", "invocation controls need --resume (with its seal) and/or --pause-after-occurrences")
        if options_identity(resume, seal, pause) != flags["--expected-options-sha256"]:
            return _refuse_print("options_changed_between_parent_and_worker", "the invocation controls the worker received differ from the ones the parent authenticated; refused before any output or physical work")
        invocation = Invocation(resume, seal, pause, flags["--expected-options-sha256"])
        argv, expected, boot = [argv[0]], flags["--expected-config-sha256"], flags["--expected-bootstrap-sha256"]
    if len(argv) != 1:
        print("usage: python -m masy.native_water_worker CONFIG --expected-config-sha256 SHA --expected-bootstrap-sha256 SHA [--expected-options-sha256 SHA [--resume DIR --expected-seal-sha256 SHA] [--pause-after-occurrences N]]", file=sys.stderr)
        return EXIT_REFUSED
    try:
        cfg = load_run_config(argv[0])  # closed validation first; every reference body is then guarded by a fresh availability reading before it is opened
    except RunConfigError as exc:
        return _refuse_print(exc.code, exc.detail)
    if expected is not None and cfg.config_sha256 != expected:
        return _refuse_print("config_changed_between_parent_and_worker", "the configuration bytes the worker read differ from the bytes the parent parsed; refused before any physical work")
    production = cfg.source["mode"] == "production_committed" and cfg.schema == "masy.native_water_run.v2"
    if cfg.source["mode"] != "development_snapshot" and not production:
        return _refuse_print("production_recipe_unavailable", "the worker runs only the verified development_snapshot mode or a v2 production_committed configuration")
    if boot is None:
        return _refuse_print("bootstrap_identity_required", "the worker is launched only by the parent, which passes the closed bootstrap identity it verified")
    try:
        avail = _availability(cfg.resources["host_available_cap_bytes"], None)
        reserve = cfg.resources["caller_reserve_bytes"]

        def admit(n: int, what: str) -> None:
            r = avail()
            if r < reserve + n:
                raise gate.GateRefusal("cold_memory", f"{what} needs {n} B + the {reserve}-B caller reserve but only {r} B are available")
        if production:  # the worker re-derives every byte/lock/capsule/interpreter/closure join itself (no second V1 child) and must reach the parent's identity
            from masy import production_runtime as prod

            resolved = prod.resolve(cfg.source, cfg.config_sha256, capability=PROFILE_CAPABILITY[cfg.source["profile"]], role="worker", admit=admit)
            identity = resolved.identity
        else:
            resolved = None
            identity = gate.derive_identity(cfg.source, cfg.config_sha256, gate.GateLimits(), admit, allowed=gate.WORKER_ALLOWED, required=gate.WORKER_REQUIRED)
    except (gate.GateRefusal, RunRefused) as exc:
        return _refuse_print(exc.code, exc.detail)
    if gate.bootstrap_identity_sha(identity) != boot:
        return _refuse_print("bootstrap_identity_mismatch", "the worker's own verification of sources/interpreter/code/config differs from the parent's closed bootstrap identity; refused before any output or physical work")
    outcome = (run_configured_production(cfg, resolved, bootstrap=identity, invocation=invocation) if production else run_configured(cfg, bootstrap=identity, invocation=invocation))
    print(RESULT_PREFIX + json.dumps({"status": outcome.status, "exit_code": outcome.exit_code, "code": outcome.code, "detail": outcome.detail, "summary": outcome.summary},
                                     sort_keys=True, default=str))
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
