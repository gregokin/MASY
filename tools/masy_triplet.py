#!/usr/bin/env python3
"""Explicit triplet provisioning and separately selected terrain execution."""
from __future__ import annotations

import argparse
import json
import os
import sysconfig
from pathlib import Path
import sys


def main(argv):
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise ValueError("use the existing MAPLE CPU python -I -B")
    forbidden = ("masy", "maple", "maple_syrup", "leaf", "benchmarks", "numpy", "numba", "cupy", "cupy_backends")
    if any(n.partition(".")[0] in forbidden for n in sys.modules):
        raise ValueError("preloaded project/provider/helper package")
    standard = {sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib"),
                os.path.join(sys.base_exec_prefix, "lib", "python" + sysconfig.get_python_version(), "lib-dynload"),
                os.path.join(sys.base_prefix, "lib", "python" + str(sys.version_info.major) + str(sys.version_info.minor) + ".zip")}
    first = [p for p in sys.path if p in standard]
    tail = [p for p in sys.path if p not in standard]
    sys.path[:] = first + [str(Path(__file__).resolve().parent.parent / "src")] + tail
    from masy import triplet_lock as tl
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", allow_abbrev=False)
    p.add_argument("--qualification", required=True)
    p.add_argument("--qualification-sha256", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--runtime", choices=("cpu", "cuda", "terrain-cuda"), default="cpu")
    p = sub.add_parser("verify", allow_abbrev=False)
    p.add_argument("--lock", required=True)
    p.add_argument("--lock-sha256", required=True)
    p.add_argument("--runtime", choices=("cpu", "cuda", "terrain-cuda"), default="cpu")
    p = sub.add_parser("run-terrain", allow_abbrev=False)
    p.add_argument("--lock", required=True)
    p.add_argument("--lock-sha256", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--inputs-sha256", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run-terrain" or args.runtime == "terrain-cuda":
            from masy import triplet_terrain_runtime as selected
        elif args.runtime == "cuda":
            from masy import triplet_cuda_runtime as selected
        else:
            selected = tl
        if args.command == "prepare":
            record = selected.prepare(args.qualification, expected_sha256=args.qualification_sha256, destination=args.destination)
        elif args.command == "verify":
            lock = selected.verify(args.lock, expected_sha256=args.lock_sha256)
            record = {"status": "source_verified", "profile": lock["profile"], "root": lock["root"],
                      "native_runtime_checked": False, "model_executed": False}
        else:
            spec = selected.bootstrap_spec(args.lock, expected_sha256=args.lock_sha256,
                run_dir=args.run_dir, entrypoint="masy.terrain_runner:main",
                argv=("--inputs", args.inputs, "--inputs-sha256", args.inputs_sha256,
                      "--output", args.output, "--run-dir", args.run_dir))
            # Preserve one owned PID/process group for an external resource supervisor.
            os.execve(spec["argv"][0], spec["argv"], spec["env"])
        print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except tl.TripletLockError as exc:
        print(json.dumps({"status": "refused", "code": exc.code, "detail": exc.detail}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
