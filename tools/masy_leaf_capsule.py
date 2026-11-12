"""Operator tool for immutable LEAF source capsules (stdlib until the sealing helpers are imported).

    python -I -B tools/masy_leaf_capsule.py seal --leaf-root ROOT --expected-revision HEAD --namespace NS --run-id ID --wrapper-root MASYROOT --reserve-bytes N [--available-cap-bytes N]
    python -I -B tools/masy_leaf_capsule.py verify --namespace NS --capsule-id ID --reserve-bytes N [--available-cap-bytes N] [--expected-manifest-sha256 SHA]

`verify` imports only `masy.leaf_capsule` (closure-safe). Exit 0 ok, 2 refused, 1 internal error. Admission uses MemAvailable (optionally capped) minus the explicit reserve.
This tool seals/verifies bytes only: it is not a production gate, recipe or independent qualification.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _available(cap):
    with open("/proc/meminfo", "rb") as fh:
        for line in fh:
            if line.startswith(b"MemAvailable:"):
                value = int(line.split()[1]) * 1024
                return value if cap is None else min(value, cap)
    raise SystemExit("MemAvailable unreadable")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="masy_leaf_capsule", allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("seal", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--namespace", required=True)
        p.add_argument("--reserve-bytes", type=int, required=True)
        p.add_argument("--available-cap-bytes", type=int, default=None)
    s = sub.choices["seal"]
    s.add_argument("--leaf-root", required=True)
    s.add_argument("--expected-revision", required=True)
    s.add_argument("--run-id", required=True)
    s.add_argument("--wrapper-root", action="append", required=True)
    v = sub.choices["verify"]
    v.add_argument("--capsule-id", required=True)
    v.add_argument("--expected-manifest-sha256", default=None)
    args = parser.parse_args(argv)
    for flag, value in (("--reserve-bytes", args.reserve_bytes), ("--available-cap-bytes", args.available_cap_bytes)):
        if value is not None and value < 0:  # a negative value would let the reserve add to (or the cap subtract from) the reading; refused before any file/namespace read
            print(json.dumps({"status": "refused", "code": "invalid_argument", "detail": f"{flag} must be an integer >= 0 (zero is valid: a zero cap refuses every admission), got {value}"}))
            return 2
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        print("refused: run with python -I -B", file=sys.stderr)
        return 2
    sys.path.insert(0, SRC)

    def admit(n, what):
        reading = _available(args.available_cap_bytes)
        if reading < args.reserve_bytes + n:
            from masy.native_source_gate import GateRefusal
            raise GateRefusal("cold_memory", f"{what} needs {n} B + the {args.reserve_bytes}-B reserve but only {reading} B are available")
    try:
        if args.command == "verify":
            from masy.leaf_capsule import verify_capsule
            out = verify_capsule(args.namespace, args.capsule_id, admit=admit, expected_manifest_sha256=args.expected_manifest_sha256)
        else:
            from masy.leaf_capsule_seal import observe_leaf_scope, seal_leaf_capsule
            obs = observe_leaf_scope(args.leaf_root, expected_revision=args.expected_revision, admit=admit)
            roots = [os.path.realpath(r) for r in args.wrapper_root]
            protected = {f"wrapper {rel} ({r})": os.path.join(r, rel) for r in roots for rel in ("src", "tools", "pyproject.toml", ".git")}
            out = seal_leaf_capsule(args.namespace, obs, run_id=args.run_id, wrapper_roots=roots, protected=protected, admit=admit)
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "code", None)
        if code is None:
            print(f"internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"status": "refused", "code": code, "detail": getattr(exc, "detail", str(exc))}))
        return 2
    print(json.dumps({"status": "ok", **{k: v for k, v in out.items() if k not in ("files", "blobs")}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
