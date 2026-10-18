"""MASY-03 locked CPU runtime launcher (stdlib only before handing over to `masy.runtime`).

    /home/okin/MAPLE/.venv/bin/python -I -B tools/masy_runtime.py prepare --config RUNTIME.toml
    /home/okin/MAPLE/.venv/bin/python -I -B tools/masy_runtime.py verify --store STORE --lock LOCK_ID

`--help` prints informational usage and exits 0 without creating anything or importing MASY/upstream code.

Before any MASY import this launcher, in this order:

1. refuses unless the interpreter runs with -I (isolated) and -B (no bytecode writes), and unless no `masy`,
   upstream (`maple`, `maple_syrup`) or native-stack module is already loaded;
2. determines the store: `verify --store`, or for `prepare` the config's `[runtime].store` (bounded stdlib TOML
   read; default /home/okin/MASY/runs/runtime-store). It must be an absolute, normalized path. `masy.runtime`
   re-parses the full config and refuses if its store differs from this one;
3. generates a fresh run id and points `sys.pycache_prefix` at the future run-owned
   `<store>/runs/<run_id>/cache/pycache` (never created: with -B nothing is written, and a missing prefix tree
   means no stale bytecode is read);
4. records the canonical launch identity (launcher realpath and sha256, MASY root, store, run id, argv, the
   interpreter exactly as invoked plus sys.prefix/base_prefix, flags and the sys.path insertion) and puts this
   tree's `src` first on `sys.path`; then hands over to `masy.runtime.main`, whose `_check_launch` re-validates it.

The interpreter spelling is recorded as invoked (`sys.executable`), never resolved: the configured venv symlink
(/home/okin/MAPLE/.venv/bin/python) keeps its venv. Children use `[runtime].python` from the config (prepare) or
the lock (verify), also unresolved. This process never imports `maple`, `maple_syrup` or native packages; after
`masy.runtime.main` returns that is re-checked and a violation turns the exit into an internal error.

Exit codes: 0 runtime_locked/runtime_verified (always transport_ready=false) or `--help`; 2 refused (including
launcher refusals on stderr, before anything is created); 3 incompatible; 1 internal error.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import traceback

EXIT_HELP = 0
EXIT_REFUSED = 2
EXIT_INTERNAL = 1
MAX_CONFIG_BYTES = 1024 * 1024  # same bound as masy.runtime.MAX_CONFIG_BYTES
MAX_LAUNCHER_BYTES = 1024 * 1024  # same bound as masy.runtime.MAX_SMALL_FILE_BYTES
DEFAULT_STORE = "/home/okin/MASY/runs/runtime-store"  # masy.runtime.DEFAULT_STORE; a mismatch refuses there
# Top-level packages the parent must never load (upstream code and the native numerical stack).
FORBIDDEN_FAMILIES = ("maple", "maple_syrup", "numpy", "numba", "llvmlite", "jax", "jaxlib", "scipy", "cupy")
HELP_EPILOG = (
    "Informational help only: nothing is created and no verdict is published. A real run must use the "
    "interpreter flags -I -B. prepare seals the current committed MAPLE/SYRUP/MASY HEADs and publishes a lock; "
    "verify re-checks one explicit lock with a fresh child and never reads current upstream repositories. "
    "Exit codes: 0 runtime_locked/runtime_verified (transport_ready is always false), 2 refused, "
    "3 incompatible, 1 internal error. Nothing is installed, built, downloaded or garbage-collected."
)


def build_parser() -> argparse.ArgumentParser:
    """Mirror of `masy.runtime.build_parser` used only to find the store before MASY is imported."""
    parser = argparse.ArgumentParser(prog="masy_runtime", description="MASY-03 locked CPU runtime prepare/verify",
                                     epilog=HELP_EPILOG, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", allow_abbrev=False, help="seal the current committed pair and publish a lock")
    p.add_argument("--config", required=True, help="runtime TOML (schema masy.runtime.v1)")
    v = sub.add_parser("verify", allow_abbrev=False, help="re-verify one explicit lock with a fresh child")
    v.add_argument("--store", required=True, help="absolute runtime store path")
    v.add_argument("--lock", required=True, help="lock id or <store>/locks/<id>.json")
    v.add_argument("--max-lock-bytes", type=int, default=None, help="bound for reading the lock file")
    return parser


def _refuse(message: str) -> int:
    print(f"masy_runtime: refused: {message}", file=sys.stderr)
    return EXIT_REFUSED


def _wants_help(argv: list[str]) -> bool:
    for arg in argv:
        if arg == "--":
            return False
        if arg in ("-h", "--help"):
            return True
    return False


def _loaded(families: tuple[str, ...]) -> list[str]:
    return sorted(name for name in list(sys.modules) if name.partition(".")[0] in families)


def configured_store(config_path: str) -> str:
    """`[runtime].store` of a parseable config, or the default. Raises ValueError if the config cannot be read or
    parsed or the value is not a string: the run location is then unknown and nothing may be created."""
    import tomllib  # imported after the preload check; stdlib only

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
    runtime = raw.get("runtime", {})
    store = runtime.get("store", DEFAULT_STORE) if isinstance(runtime, dict) else None
    if not isinstance(store, str):
        raise ValueError(f"config {config_path}: [runtime].store must be a string")
    return store


def _check_store_spelling(store: str) -> str | None:
    if not store or "\0" in store or any(ord(ch) < 0x20 or ord(ch) == 0x7f for ch in store):
        return f"store {store!r} contains control characters or is empty"
    if not os.path.isabs(store) or os.path.normpath(store) != store:
        return f"store {store!r} must be an absolute, normalized path"
    return None


def new_run_id(command: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{command}-{stamp}-{os.getpid()}-{os.urandom(4).hex()}"


def _launcher_sha256(path: str) -> str:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        data = b""
        while len(data) <= MAX_LAUNCHER_BYTES:
            chunk = os.read(fd, MAX_LAUNCHER_BYTES + 1 - len(data))
            if not chunk:
                break
            data += chunk
    finally:
        os.close(fd)
    if len(data) > MAX_LAUNCHER_BYTES:
        raise ValueError(f"launcher {path} is larger than {MAX_LAUNCHER_BYTES} bytes")
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str]) -> int:
    launcher = os.path.realpath(__file__)
    masy_root = os.path.dirname(os.path.dirname(launcher))
    masy_src = os.path.join(masy_root, "src")
    if _wants_help(argv):
        build_parser().print_help()
        return EXIT_HELP
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        return _refuse("run with the interpreter flags -I -B (isolated, no bytecode writes)")
    preloaded = _loaded(("masy", *FORBIDDEN_FAMILIES))
    if preloaded:
        return _refuse(f"modules were loaded before the launcher: {preloaded[:20]}")
    args = build_parser().parse_args(argv)  # usage errors exit 2 before anything is created

    if args.command == "prepare":
        try:
            store = configured_store(args.config)
        except ValueError as exc:
            return _refuse(f"{exc}; the store is unknown, nothing was created")
    else:
        store = args.store
    problem = _check_store_spelling(store)
    if problem:
        return _refuse(f"{problem}; nothing was created")
    try:
        launcher_sha256 = _launcher_sha256(launcher)
    except (OSError, ValueError) as exc:
        return _refuse(f"cannot hash the launcher {launcher}: {exc}; nothing was created")

    run_id = new_run_id(args.command)
    sys.pycache_prefix = os.path.join(store, "runs", run_id, "cache", "pycache")
    sys_path_before = list(sys.path)
    sys.path.insert(0, masy_src)
    launch = {
        "launcher": launcher,
        "launcher_sha256": launcher_sha256,
        "masy_root": masy_root,
        "masy_src": masy_src,
        "store": store,
        "run_id": run_id,
        "command": args.command,
        "argv": list(argv),
        "python": {"executable_as_invoked": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
                   "version": sys.version, "flags": {"isolated": sys.flags.isolated,
                                                     "dont_write_bytecode": sys.flags.dont_write_bytecode}},
        "sys_path_before": sys_path_before,
        "sys_path_inserted": [masy_src],
        "pycache_prefix": sys.pycache_prefix,
        "pycache_prefix_policy": "run-owned, not created (-B: no bytecode writes; no source-tree bytecode read)",
    }
    try:
        from masy.runtime import main as runtime_main

        code = runtime_main(argv, launch=launch)
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - internal error: no verdict is published
        traceback.print_exc()
        print("masy_runtime: internal error; no verdict published", file=sys.stderr)
        return EXIT_INTERNAL
    leaked = _loaded(FORBIDDEN_FAMILIES)
    if leaked:
        print(f"masy_runtime: internal error: the parent imported upstream/native modules {leaked[:20]}; the "
              "printed verdict must not be trusted", file=sys.stderr)
        return EXIT_INTERNAL
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
