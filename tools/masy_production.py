"""Operator helpers for the v2 production runtime (stdlib; imports only the closure-safe `masy.production_runtime` / `masy.leaf_capsule`).

    python -I -B tools/masy_production.py draft-recipe --store S --lock LOCKID --capsule-namespace NS --capsule-id ID --name NAME --profile P --device cpu|cuda --reserve-bytes N --out RECIPE.json
    python -I -B tools/masy_production.py draft-receipt --recipe RECIPE.json --out RECEIPT.json
    python -I -B tools/masy_production.py render-config OUTDIR --recipe R --receipt R --store S --lock ID --release-root R --capsule-namespace NS [--profile P]

`draft-recipe` DERIVES every value from the real V1 lock and the real capsule (nothing is typed in, no commit is guessed). `draft-receipt` writes an UNREVIEWED draft the production gate refuses; only Root's
reviewed receipt (status root_reviewed, independent true) is accepted, and its sha256 is what the operator pins. These tools write new files only (O_EXCL) and are not a qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))


def _admit_factory(reserve: int):
    def admit(n, what):
        pages, page = os.sysconf("SC_AVPHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        if pages * page < reserve + n:
            from masy.native_source_gate import GateRefusal
            raise GateRefusal("cold_memory", f"{what} needs {n} B + the {reserve}-B reserve")
    return admit


def _write_new(path: str, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="masy_production", allow_abbrev=False)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft-recipe")
    for k in ("store", "lock", "capsule-namespace", "capsule-id", "name", "profile", "out"):
        d.add_argument(f"--{k}", required=True)
    d.add_argument("--device", choices=("cpu", "cuda"), required=True)
    d.add_argument("--reserve-bytes", type=int, required=True)
    r = sub.add_parser("draft-receipt")
    r.add_argument("--recipe", required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("render-config")
    c.add_argument("out")
    for k in ("recipe", "receipt", "store", "lock", "release-root", "capsule-namespace"):
        c.add_argument(f"--{k}", required=True)
    c.add_argument("--profile", default="native9f_layered_factory_afca_cpu_v1")
    a = p.parse_args(argv)
    try:
        if a.cmd == "draft-recipe":
            if a.reserve_bytes < 0:
                raise SystemExit("--reserve-bytes must be >= 0")
            from masy import leaf_capsule as cap
            from masy import production_runtime as prod
            from masy.native_source_gate import GateLimits
            admit, lim = _admit_factory(a.reserve_bytes), GateLimits()
            lock, _sha = prod._read_lock(a.store, a.lock, prod.ProductionLimits(), lim, admit)
            capsule = cap.verify_capsule(a.capsule_namespace, a.capsule_id, lim, admit)
            g = lambda *k: prod._lock_get(lock, *k)  # noqa: E731
            src = {kind: {"commit": g("sources", kind, "head"), "object_id": g("objects", "sources", kind, "object_id"), "package_digest": g("objects", "sources", kind, "package_digest")} for kind in ("maple", "syrup", "masy")}
            src["maple"]["bundle_id"] = g("objects", "maple_bundle", "bundle_id")
            src["maple"]["bundle_manifest_sha256"] = g("objects", "maple_bundle", "manifest_sha256")
            src["maple"]["archive_sha256"] = g("objects", "maple_bundle", "archive_sha256")
            src["masy"]["launcher_sha256"] = g("sources", "masy", "launcher", "sha256")
            src["leaf"] = {"commit": capsule["provenance"]["head"], "capsule_id": capsule["object_id"], "package_digest": capsule["package_digest"], "tree_digest": capsule["tree_digest"]}
            recipe = {"schema": prod.RECIPE_SCHEMA, "name": a.name, "profiles": {a.profile: {"device": a.device, "leaf_profile": prod.LEAF_PROFILES[a.device], "leaf_variant": "8da"}}, "sources": src}
            prod.parse_recipe(recipe)
            _write_new(a.out, (prod.canonical_json(recipe) + "\n").encode())
        elif a.cmd == "draft-receipt":
            sha = hashlib.sha256(Path(a.recipe).read_bytes()).hexdigest()
            doc = {"schema": "masy.production_release_receipt.v1", "status": "unreviewed_draft", "independent": False, "recipe_sha256": sha, "scope": "DRAFT: not reviewed",
                   "evidence": ["none"], "limits": ["a draft is refused by the production gate; only a Root-reviewed receipt is accepted"]}
            _write_new(a.out, (json.dumps(doc, sort_keys=True, indent=1) + "\n").encode())
        else:
            out = Path(a.out).resolve()
            if out.exists():
                raise SystemExit(f"{out} already exists; nothing written")
            inputs = HERE.parent / "examples" / "native_leaf_calendar_cli"
            out.mkdir(parents=True)
            (out / "out").mkdir()
            for name in ("case", "scenario", "layered_water.yaml", "catalog.json", "plants.csv", "plants.metadata.json"):
                s = inputs / name
                shutil.copytree(s, out / name) if s.is_dir() else shutil.copy2(s, out / name)
            text = (HERE.parent / "examples" / "native_production" / "run.template.yaml").read_text()
            vals = {"RECIPE": a.recipe, "RECEIPT": a.receipt, "PRODUCTION_STORE": a.store, "PRODUCTION_LOCK": a.lock, "RELEASE_MASY_ROOT": a.release_root, "LEAF_CAPSULE_NAMESPACE": a.capsule_namespace}
            vals["RECIPE_SHA256"] = hashlib.sha256(Path(a.recipe).read_bytes()).hexdigest()
            vals["RECEIPT_SHA256"] = hashlib.sha256(Path(a.receipt).read_bytes()).hexdigest()
            for k, v in vals.items():
                text = text.replace(f"@{k}@", str(Path(v).resolve()) if k in ("RECIPE", "RECEIPT", "RELEASE_MASY_ROOT", "LEAF_CAPSULE_NAMESPACE", "PRODUCTION_STORE") else v)
            text = text.replace("native9f_layered_factory_afca_cpu_v1     #", f"{a.profile}     #", 1)
            if "@" in text.replace("@", "", 0) and any(f"@{k}@" in text for k in vals):
                raise SystemExit("unresolved placeholder")
            (out / "run.yaml").write_text(text)
            print(f"wrote {out / 'run.yaml'} (recipe/receipt shas are the operator pins; they are consistency, not review)")
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "code", None)
        if isinstance(exc, SystemExit):
            raise
        print(json.dumps({"status": "refused", "code": code or type(exc).__name__, "detail": getattr(exc, "detail", str(exc))}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
