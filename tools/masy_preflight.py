"""MASY-01 isolated preflight launcher (stdlib only).

    /home/okin/MAPLE/.venv/bin/python -B -I tools/masy_preflight.py \
        --config CONFIG.toml --run-dir NEW_RUN_DIR [--expect-lock LOCK.json]

`--help` prints informational usage and exits 0 without creating anything or
importing MASY/upstream code; it is not an audit and carries no verdict.

Before any MASY or upstream import this launcher, in this order:

1. refuses unless the interpreter runs with -I (isolated) and -B (no
   bytecode writes);
2. points `sys.pycache_prefix` at the future run-owned
   `<run-dir>/cache/pycache` (not created: with -B nothing is written, and a
   missing prefix tree means no stale source-tree bytecode can be read);
3. reads the config's `[sources.*].root` strings (bounded stdlib TOML read)
   and refuses, BEFORE any directory or file is created, if the resolved run
   directory (symlinked parents included) equals, lies inside or contains a
   configured source root or MASY's `src`/`tools` directories. An unreadable
   or unparseable config also refuses here, since the inputs to protect are
   unknown; a parseable but otherwise invalid config is refused by `masy.cli`
   with a published report;
4. creates the run directory exclusively (an existing one refuses), re-checks
   its resolved location, then creates the run-owned NUMBA/matplotlib cache
   directories inside it;
5. sets NUMBA_CACHE_DIR, MPLCONFIGDIR, NUMBA_DISABLE_CUDA=1,
   CUDA_VISIBLE_DEVICES="" and JAX_PLATFORMS=cpu so imports do not probe
   devices; then puts this tree's `src` first on `sys.path` and hands over to
   `masy.cli`.

Exit codes: 0 `--help` only (informational, no audit); 2 refused,
3 incompatible, 4 not_transport_ready (best achievable), 1 internal error.
There is no audit success exit.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

EXIT_HELP = 0
EXIT_REFUSED = 2
EXIT_INTERNAL = 1
MAX_CONFIG_BYTES = 1024 * 1024  # same bound as masy.config.MAX_CONFIG_BYTES
ENV_OVERRIDES = ("NUMBA_CACHE_DIR", "MPLCONFIGDIR", "NUMBA_DISABLE_CUDA", "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS")
HELP_EPILOG = (
    "Informational help only: no audit is run, nothing is created and no verdict is published. "
    "A real audit must run with the interpreter flags -B -I. Audit exit codes: 2 refused, 3 incompatible, "
    "4 not_transport_ready (best achievable), 1 internal error; there is no audit success exit. The run "
    "directory must be new, its parent must exist, and it must not overlap a configured source root or "
    "MASY's src/tools directories."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="masy_preflight", description="MASY-01 read-only source preflight",
                                     epilog=HELP_EPILOG, allow_abbrev=False)
    parser.add_argument("--config", required=True, help="preflight TOML (schema masy.preflight.v1)")
    parser.add_argument("--run-dir", required=True, help="new run directory (created exclusively by the launcher)")
    parser.add_argument("--expect-lock", help="JSON copy of a previous report's source_observation block")
    return parser


def _refuse(message: str) -> int:
    print(f"masy_preflight: refused: {message}", file=sys.stderr)
    return EXIT_REFUSED


def _wants_help(argv: list[str]) -> bool:
    for arg in argv:
        if arg == "--":
            return False
        if arg in ("-h", "--help"):
            return True
    return False


def configured_roots(config_path: str) -> dict[str, str]:
    """`[sources.<name>].root` strings of a parseable config (may be empty).

    Raises ValueError if the config cannot be read or parsed: the inputs to
    protect are then unknown and nothing may be created."""
    import tomllib  # imported after sys.pycache_prefix is set

    try:
        with open(config_path, "rb") as handle:
            data = handle.read(MAX_CONFIG_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"cannot read config {config_path}: {exc}") from exc
    if len(data) > MAX_CONFIG_BYTES:
        raise ValueError(f"config {config_path} is larger than {MAX_CONFIG_BYTES} bytes")
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot parse config {config_path}: {exc}") from exc
    sources = raw.get("sources")
    roots: dict[str, str] = {}
    if isinstance(sources, dict):
        for name, table in sources.items():
            root = table.get("root") if isinstance(table, dict) else None
            if isinstance(root, str) and root and "\0" not in root:
                roots[str(name)] = root
    return roots


def _contains(outer: str, inner: str) -> bool:
    try:
        return os.path.commonpath([outer, inner]) == outer
    except ValueError:
        return False


def overlap_conflicts(run_dir: str, protected: dict[str, str]) -> list[str]:
    """Protected entries that equal, contain or lie inside the run directory,
    comparing both the lexical absolute and the symlink-resolved forms."""
    run_forms = {os.path.abspath(run_dir), os.path.realpath(run_dir)}
    conflicts = []
    for label, path in protected.items():
        forms = {os.path.abspath(path), os.path.realpath(path)}
        if any(_contains(p, r) or _contains(r, p) for p in forms for r in run_forms):
            conflicts.append(f"{label} ({path})")
    return conflicts


def main(argv: list[str]) -> int:
    launcher = os.path.realpath(__file__)
    masy_root = os.path.dirname(os.path.dirname(launcher))
    masy_src = os.path.join(masy_root, "src")
    if _wants_help(argv):
        build_parser().print_help()
        return EXIT_HELP
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        return _refuse("run with the interpreter flags -B -I (isolated, no bytecode writes)")
    if any(name == "masy" or name.startswith("masy.") for name in sys.modules):
        return _refuse("masy is already imported")
    args = build_parser().parse_args(argv)  # usage errors exit 2 before anything is created

    run_dir = os.path.abspath(args.run_dir)
    planned_real = os.path.realpath(run_dir)
    sys.pycache_prefix = os.path.join(planned_real, "cache", "pycache")

    try:
        roots = configured_roots(args.config)
    except ValueError as exc:
        return _refuse(f"{exc}; input/output separation cannot be established, nothing was created")
    protected = {"MASY src": masy_src, "MASY tools": os.path.join(masy_root, "tools")}
    protected.update({f"sources.{name}.root": root for name, root in roots.items()})
    conflicts = overlap_conflicts(run_dir, protected)
    if conflicts:
        return _refuse(f"run directory {run_dir} overlaps read-only input(s) {conflicts}; nothing was created")

    try:
        os.mkdir(run_dir, 0o700)
    except FileExistsError:
        return _refuse(f"run directory {run_dir} already exists; a new run directory is required")
    except OSError as exc:
        return _refuse(f"cannot create run directory {run_dir}: {exc}")
    run_real = os.path.realpath(run_dir)
    late_conflicts = overlap_conflicts(run_real, protected)
    if run_real != planned_real or late_conflicts:
        os.rmdir(run_dir)
        return _refuse(f"run directory {run_dir} resolved to {run_real} after creation "
                       f"(planned {planned_real}; overlaps {late_conflicts}); removed, nothing published")

    caches = {name: os.path.join(run_real, "cache", name) for name in ("numba", "matplotlib")}
    for path in caches.values():
        os.makedirs(path, 0o700)
    os.environ["NUMBA_CACHE_DIR"] = caches["numba"]
    os.environ["MPLCONFIGDIR"] = caches["matplotlib"]
    os.environ["NUMBA_DISABLE_CUDA"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"

    sys_path_before = list(sys.path)
    sys.path.insert(0, masy_src)
    launch = {
        "launcher": launcher,
        "masy_root": masy_root,
        "masy_src": masy_src,
        "run_dir": run_real,
        "argv": list(argv),
        "sys_path_before": sys_path_before,
        "sys_path_inserted": [masy_src],
        "pycache_prefix": sys.pycache_prefix,
        "pycache_prefix_policy": "run-owned, not created (-B: no bytecode writes; no source-tree bytecode read)",
        "protected_inputs": protected,
        "environment_overrides": {name: os.environ[name] for name in ENV_OVERRIDES},
    }
    try:
        from masy.cli import main as cli_main

        return cli_main(argv, launch=launch)
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - internal error: no verdict is published
        traceback.print_exc()
        print("masy_preflight: internal error; no verdict published", file=sys.stderr)
        return EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
