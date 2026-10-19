"""Isolated GPU-profile launcher. No model execution; --help is read-only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path


def parser():
    value = argparse.ArgumentParser(prog="masy_gpu_runtime", allow_abbrev=False,
        description="Immutable .venv-cupy source/environment profiles and separately gated identity queries")
    commands = value.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", allow_abbrev=False)
    prepare.add_argument("--config", required=True, help="closed masy.gpu_runtime_config.v1 JSON")
    for name in ("verify", "probe"):
        item = commands.add_parser(name, allow_abbrev=False)
        item.add_argument("--store", required=True)
        item.add_argument("--profile", required=True)
        if name == "probe":
            item.add_argument("--authorization", required=True,
                help="explicit accepted profile/UUID identity_probe scope; does not authorize transport")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    forbidden = {"masy", "maple", "maple_syrup", "numpy", "numba", "llvmlite", "cupy", "cupyx", "cupy_backends", "cuda"}
    try:
        if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1 or \
                any(n.partition(".")[0] in forbidden for n in sys.modules):
            raise ValueError("use -I -B with no native stack preloaded")
        root = Path(__file__).resolve().parents[1]
        if args.command == "prepare":
            with open(args.config, "rb") as handle:
                data = handle.read((1 << 20) + 1)
            if len(data) > 1 << 20:
                raise ValueError("config exceeds1MiB")
            store = json.loads(data)["runtime"]["store"]
        else:
            store = args.store
        if not isinstance(store, str) or not os.path.isabs(store) or os.path.normpath(store) != store:
            raise ValueError("store must be an absolute normalized path")
        run_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:12]
        # Before the first MASY import, route bytecode reads away from the
        # candidate/source objects into a deliberately nonexistent prefix.
        sys.pycache_prefix = str(Path(store) / "runs" / run_id / "cache/pycache")
        sys.path.insert(0, str(root / "src"))
        from masy import gpu_runtime as runtime
        launch = {"masy_root": str(root), "launcher": str(Path(__file__).resolve()),
            "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "run_id": run_id, "store": store, "argv": list(sys.argv)}
        if args.command == "prepare":
            result = runtime.prepare_profile(runtime.load_gpu_config(args.config), launch=launch)
        elif args.command == "verify":
            result = runtime.verify_profile(store, args.profile, launch=launch)
        else:
            result = runtime.probe_profile(store, args.profile, launch=launch,
                authorization=runtime._read_json(args.authorization))
        if any(n.partition(".")[0] in forbidden - {"masy"} for n in sys.modules):
            raise ValueError("parent loaded native stack")
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - unexpected failures are classified as internal_error
        # Source refusal evidence stays in run-owned receipts. An unexpected
        # exception is distinguished rather than presented as qualification.
        record = {"verdict": "refused" if hasattr(exc, "code") or isinstance(exc, ValueError) else "internal_error",
            "code": getattr(exc, "code", type(exc).__name__), "detail": str(exc)[:4000],
            "evidence": getattr(exc, "evidence", {})}
        print(json.dumps(record, sort_keys=True, separators=(",", ":"), default=str), file=sys.stderr)
        return 2 if record["verdict"] == "refused" else 1


if __name__ == "__main__":
    raise SystemExit(main())
