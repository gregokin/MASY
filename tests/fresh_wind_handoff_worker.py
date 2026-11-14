"""Fresh native restart control. Static recipe + verified native file, no bed genesis.

This is test tooling, not a combined CLI or runtime provisioner. The separately
tested MASY cursor binds to an already prepared static wind runtime.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
from types import SimpleNamespace

root = Path(__file__).resolve().parents[1]
bundle = Path(os.environ["MASY_DAILY_NATIVE_BUNDLE"])
syrup = Path(os.environ["MASY_DAILY_NATIVE_SYRUP"])
sys.path[:0] = [str(bundle / "editable"), str(bundle / "source/src"), str(syrup / "src"), str(root / "src"), str(root / "tests")]

import numpy as np
from maple_syrup import native_case_profile as ncp, native_floating_soil as nfs
from maple_syrup import floating_lifecycle_checkpoint as lcp, floating_wind_handoff as wh
from maple_syrup.sediment_bed import BedContext
from maple_syrup.pond_network import build_pond_topology
from maple_syrup.subsurface import default_host_available_bytes
from maple.experiment import schema, orchestrator
from maple.case_tools.compilers import case_compiler
from maple_syrup import probe
from masy import daily_wind_handoff as mh
from masy.daily_csv import load_daily_scenario
from masy.daily_schedule import DailyPlan
import wind_handoff_support as hs


def forbidden(*args, **kwargs):
    raise AssertionError("fresh native restart must not initialize original terrain, a water owner or soil properties")


def main(work):
    work = Path(work)
    seal = json.loads((work / "fresh_seal.json").read_text())
    ncp.verify_source_profile(bundle, expected_syrup_digest=mh.SYRUP_PACKAGE_SHA256)
    nfs.initialize_native_floating_soil = forbidden
    nfs.stage_native_refresh = forbidden
    nfs._batch_properties = forbidden
    probe.build_minimal_state = forbidden
    orchestrator.compile_case_package = forbidden
    # The public resolver builds parameters only; no compiler/bed constructor.
    case_compiler.compile_case = forbidden
    config = schema.resolve_case_package(work / "case").config
    g = config.geometry
    fraction = np.broadcast_to(np.array([0.5, 0.75, 1.0]), (g.ny, g.nx, 3)).copy()
    context = BedContext(g, config.grain_classes, config.numerics.mass_resolution_kg,
                         config.topographic_commit, config.avalanching, fraction)
    capsule = work / "static_topology.npz"
    assert hashlib.sha256(capsule.read_bytes()).hexdigest() == seal["static_sha256"]
    with np.load(capsule, allow_pickle=False) as z:
        topology = build_pond_topology(z["source_z_full"], z["active"], z["friction"],
                                       g.dx_m, [tuple(map(int, row)) for row in z["outlets"]])
    assert topology.topology_sha256 == seal["topology_sha256"]
    factory = hs.native_factory(context, topology)
    loaded = lcp.load_floating_lifecycle_checkpoint(work / "checkpoint", factory, seal["identity"],
        seal["native_manifest_sha256"], original_topology=topology, available_bytes_fn=default_host_available_bytes)
    assert hs.physical_record(loaded.head) == seal["before"]
    assert wh.restore_wind_context(loaded.head, context, seal["native_context"]).record() == seal["native_context"]
    plan = DailyPlan(load_daily_scenario(work / "inputs/scenario.csv"))
    facade = SimpleNamespace(lineage=loaded.lineage, _plan=plan, run_digest=seal["run_digest"])
    fixture = SimpleNamespace(factory=factory, available=default_host_available_bytes)
    final, audits = hs.run_native_day(facade, fixture, list(plan.days())[1])
    assert audits == []
    (work / "fresh_result.json").write_text(json.dumps(hs.physical_record(final), sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
