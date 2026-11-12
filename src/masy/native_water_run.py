"""Documented command of the configured water-only native calendar run (separate from the read-only preflight CLI and its success semantics).

    python -m masy.native_water_run run CONFIG            # refused before physical work: exit 2; physical failure/incomplete: exit 3; complete: exit 0
    python -m masy.native_water_run run CONFIG --pause-after-occurrences N      # (needs output.physical_checkpoint: combined_terminal) paused after N accepted occurrences: exit 4
    python -m masy.native_water_run run CONFIG --resume CKPT --expected-seal-sha256 SHA [--pause-after-occurrences N]   # resume a combined checkpoint into a FRESH run directory
    python -m masy.native_water_run manifest ROOT OUT     # DEVELOPMENT helper: record the expected manifest of a frozen snapshot (declarative; never qualification)

`run` only reads the configuration (stdlib, no native import), obtains the genuine SYRUP `dependency_bundle.runtime_environment(native_target, <syrup_root>/src)` from a
stdlib-only helper child, and launches `masy.native_water_worker` in a FRESH child process with that environment plus the declared MASY root `src` (`-B`; no sys.modules
swapping, no editable import accepted as qualification). The child verifies the named source profile (census, namespace, distribution) before any physical work. There is no
implicit restart: `--resume` is explicit, pinned by the artifact's seal, and the combined artifact (`output.physical_checkpoint: combined_terminal`) is distinct from the
standalone physical bundle (`terminal_standalone`). Invocation controls (`--resume`, `--pause-after-occurrences`) are options, not configuration keys; the parent passes their
SHA-256 identity to the worker, which recomputes it before any output or physical work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from masy import native_source_gate as gate
from masy.native_water_run_config import CONFIG_SCHEMA_V2, PARENT_BUFFER_BYTES, PROFILE_CAPABILITY, RunConfigError, load_run_config, make_availability

__all__ = ["EXIT_PAUSED", "build_manifest", "child_environment", "main", "options_identity"]

RESULT_PREFIX = "MASY_RUN_RESULT:"
EXIT_PAUSED = 4
_HEX64 = frozenset("0123456789abcdef")
_ENV_HELPER = ("import json, sys; sys.path.insert(0, sys.argv[2]); from maple_syrup import dependency_bundle as b; "
               "print(json.dumps(b.runtime_environment(sys.argv[1], sys.argv[2])['env']))")


def child_environment(native_target: str, syrup_root: str, masy_root: str, python: str | None = None, extra_paths: tuple[str, ...] = ()) -> dict[str, str]:
    """Environment of the worker child: the existing SYRUP `runtime_environment` (obtained in a fresh `-I -B` helper child that imports only `dependency_bundle`) + the MASY `src`."""
    python = python or sys.executable
    syrup_src = str(Path(syrup_root) / "src")
    done = subprocess.run([python, "-I", "-B", "-c", _ENV_HELPER, native_target, syrup_src], capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise RunConfigError("environment_unavailable", f"runtime_environment helper failed: {done.stderr.strip()[-400:]}")
    env_part = json.loads(done.stdout.strip().splitlines()[-1])
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env.update(env_part)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(masy_root) / "src"), *map(str, extra_paths), env_part["PYTHONPATH"]])  # extra_paths: the v2 LEAF reference capsule (verified by the worker's census BEFORE any LEAF import)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def options_identity(resume: str | None, expected_seal: str | None, pause_after: int | None) -> str:
    """SHA-256 of the canonical JSON of the invocation controls (parent -> worker authentication; never part of the configuration identity)."""
    doc = {"resume": resume, "expected_seal_sha256": expected_seal, "pause_after": pause_after}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def _check_options(cfg: Any, resume: str | None, seal: str | None, pause: int | None) -> tuple[str | None, str | None, int | None]:
    """Metadata-only validation of the invocation controls (the parent opens no checkpoint body and imports no native module)."""
    if resume is None and seal is not None:
        raise RunConfigError("invalid_options", "--expected-seal-sha256 is only meaningful with --resume")
    if resume is not None:
        if seal is None or type(seal) is not str or len(seal) != 64 or not set(seal) <= _HEX64:
            raise RunConfigError("invalid_options", "--resume REQUIRES --expected-seal-sha256 (64 lowercase hex digits): there is no implicit or latest checkpoint")
        resume = os.path.abspath(resume)
        try:
            st = os.lstat(resume)
        except OSError as exc:
            raise RunConfigError("invalid_options", f"the resume checkpoint cannot be examined: {exc.strerror or exc}") from None
        if not stat.S_ISDIR(st.st_mode):
            raise RunConfigError("invalid_options", "the resume checkpoint must be a real directory (a symlink or file is refused)")
    if pause is not None:
        if type(pause) is not int or pause < 0:
            raise RunConfigError("invalid_options", "--pause-after-occurrences must be an integer >= 0")
        if cfg.output["physical_checkpoint"] != "combined_terminal":
            raise RunConfigError("pause_requires_combined_terminal", "--pause-after-occurrences needs output.physical_checkpoint: combined_terminal (a pause without a resumable checkpoint is refused)")
    return resume, seal, pause


def build_manifest(root: str | os.PathLike) -> dict:
    """Expected manifest {relative path: sha256} of a snapshot tree, recorded by the bounded no-follow streaming traversal of `masy.native_source_gate` (symlinks, special
    files, caches/VCS metadata and over-cap trees refuse; directory symlinks are never followed). A DEVELOPMENT, declarative record: never an acceptance authority."""
    try:
        return gate.record_manifest(str(root), gate.GateLimits(), _manifest_admit())
    except gate.GateRefusal as exc:
        raise RunConfigError(exc.code, exc.detail) from None


def _manifest_admit():
    avail = make_availability(None)

    def admit(n: int, what: str) -> None:
        if avail() < n:
            raise gate.GateRefusal("cold_memory", f"{what} needs {n} B but only {avail()} B are available")
    return admit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m masy.native_water_run", description="configured native calendar run (water-only v1; v2 adds the LEAF participant)", allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="execute a run configuration")
    run.add_argument("config")
    run.add_argument("--resume", default=None, metavar="CHECKPOINT", help="resume this sealed combined checkpoint directory (fresh run directory; needs --expected-seal-sha256)")
    run.add_argument("--expected-seal-sha256", default=None, metavar="SHA", help="the manifest seal the resumed checkpoint must carry")
    run.add_argument("--pause-after-occurrences", type=int, default=None, metavar="N", help="stop cleanly after N accepted occurrences in total (needs combined_terminal); exit 4")
    man = sub.add_parser("manifest", help="development: record an expected manifest of a snapshot tree")
    man.add_argument("root")
    man.add_argument("out")
    args = parser.parse_args(argv)
    if args.command == "manifest":
        out = Path(args.out)
        try:
            data = (json.dumps(build_manifest(args.root), indent=1, sort_keys=True) + "\n").encode()
            fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        except (RunConfigError, OSError) as exc:
            print(f"manifest refused: {exc}", file=sys.stderr)
            return 2
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return 0
    try:
        # the PARENT form: the bounded (<= 1 MiB) configuration is the only file it reads before the availability cap is known (declared excluded); NO reference is opened here
        cfg = load_run_config(args.config, references=False)
        production = cfg.source["mode"] == "production_committed"
        if production and cfg.schema != CONFIG_SCHEMA_V2:
            gate.production_gate(cfg.source["profile"], cfg.source["production_store"], cfg.source["production_lock"])  # v1: refuses typed; never falls back to development
        avail = make_availability(cfg.resources["host_available_cap_bytes"])
        reading = avail()
        need = cfg.resources["caller_reserve_bytes"] + PARENT_BUFFER_BYTES
        if reading < need:
            raise RunConfigError("cold_memory", f"the run needs {need} B (caller reserve + parent buffer) before any helper or worker starts but only {reading} B are available")
        reserve = cfg.resources["caller_reserve_bytes"]
        resume, seal, pause = _check_options(cfg, args.resume, args.expected_seal_sha256, args.pause_after_occurrences)  # metadata only, after the zero/low-budget refusal

        def admit(n: int, what: str) -> None:  # fresh reading + caller reserve for every traversal/hash/manifest buffer BEFORE the helper child or worker exists
            r = avail()
            if r < reserve + n:
                raise gate.GateRefusal("cold_memory", f"{what} needs {n} B + the {reserve}-B caller reserve but only {r} B are available")
        if production:  # explicit v2 positive path: admission above precedes every recipe/lock/object read, hash, parse and the genuine V1 verify child
            from masy import production_runtime as prod

            resolved = prod.resolve(cfg.source, cfg.config_sha256, capability=PROFILE_CAPABILITY[cfg.source["profile"]], role="parent", admit=admit, run_verify=True)
            identity = resolved.identity
            env = child_environment(resolved.source["native_target"], resolved.source["syrup_root"], resolved.source["masy_root"], extra_paths=(resolved.leaf_root,))
        else:
            identity = gate.derive_identity(cfg.source, cfg.config_sha256, gate.GateLimits(), admit, allowed=gate.PARENT_CLOSURE, required=gate.PARENT_CLOSURE)
            env = child_environment(cfg.source["native_target"], cfg.source["syrup_root"], cfg.source["masy_root"], extra_paths=() if cfg.leaf is None else (cfg.leaf["leaf_root"],))
    except (RunConfigError, gate.GateRefusal) as exc:
        print(RESULT_PREFIX + json.dumps({"status": "refused", "exit_code": 2, "code": exc.code, "detail": exc.detail, "summary": {}}))
        return 2
    # a unique private TMPDIR per run process (the native codec stages temporary files there); removed afterwards
    tmp = tempfile.mkdtemp(prefix="masy_run_")
    env["TMPDIR"] = tmp
    try:
        # writable Numba JIT cache for the worker: a non-empty caller NUMBA_CACHE_DIR is honoured EXACTLY (never overridden, created or removed by us); otherwise a cache directory
        # inside OUR private TMPDIR (created before the worker starts, removed with it on success and failure). JIT/context exclusion: not model output, not an output-cap or memory bound.
        if not env.get("NUMBA_CACHE_DIR"):
            cache = os.path.join(tmp, "numba_cache")
            os.mkdir(cache, 0o700)
            env["NUMBA_CACHE_DIR"] = cache
        argv = [sys.executable, "-B", "-m", "masy.native_water_worker", str(Path(args.config).resolve()), "--expected-config-sha256", cfg.config_sha256,
                "--expected-bootstrap-sha256", gate.bootstrap_identity_sha(identity)]
        if resume is not None or pause is not None:  # old invocations keep the exact old argv
            argv += ["--expected-options-sha256", options_identity(resume, seal, pause)]
            if resume is not None:
                argv += ["--resume", resume, "--expected-seal-sha256", seal]
            if pause is not None:
                argv += ["--pause-after-occurrences", str(pause)]
        done = subprocess.run(argv, env=env, cwd="/", capture_output=True, text=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.stdout.write(done.stdout)
    if done.stderr:
        sys.stderr.write(done.stderr)
    return done.returncode


if __name__ == "__main__":
    raise SystemExit(main())
