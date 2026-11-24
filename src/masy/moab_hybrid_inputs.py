"""Create the bounded Moab/Jornada synthetic integration case from explicit local inputs.

Writes exact real DEM/base samples and an observed exterior ring. No transport
runs. Root profiles follow the user's three10cm-layer normalization. Generated
forcing and numerical controls are illustrative computational checks.
"""
from pathlib import Path
import argparse
import hashlib
import math
import json
import tomllib
import numpy as np
import yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--maple-case', required=True)
    parser.add_argument('--leaf-case', required=True)
    parser.add_argument('--syrup-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    original = Path(args.maple_case).resolve(strict=True)
    leaf_case = Path(args.leaf_case).resolve(strict=True)
    syrup = Path(args.syrup_root).resolve(strict=True)
    target = Path(args.output).absolute()
    required = [original / x for x in ('case.yaml','model.yaml','wind.yaml','slope.yaml',
        'input_receipt.json','source/surface.asc','source/base.asc')]
    required += [leaf_case/'species.json',leaf_case/'generation.json',leaf_case/'case.toml',
        syrup/'cases/subsurface_validation/jorn_defaults.json',
        syrup/'docs/subsurface_hydrology/jorn_fc_sensitivity/data/case_numba_fc10_s1p5.json']
    source_hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in required}
    target.mkdir(exist_ok=False)
    (target / "case/source").mkdir(parents=True)
    (target / "plan/wind").mkdir(parents=True)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def read_asc(path):
        with path.open() as f:
            head = [f.readline().split() for _ in range(6)]
            values = np.loadtxt(f, dtype=np.float64)
        return dict((k.lower(), float(v)) for k, v in head), values

    header, surface = read_asc(original / "source/surface.asc")
    bh, base = read_asc(original / "source/base.asc")
    if header != bh or surface.shape != base.shape:
        raise ValueError("surface/base geometry differs")
    valid = surface != header["nodata_value"]
    thickness = surface-base
    side = 128
    choices = []
    for row in range(1, surface.shape[0] - side - 1, 8):
        for col in range(1, surface.shape[1] - side - 1, 8):
            m = valid[row:row+side, col:col+side]
            frac = float(m.mean())
            if not 0.65 <= frac <= 0.85:
                continue
            z = surface[row:row+side, col:col+side][m]
            if np.ptp(z) < 0.5:
                continue
            if np.min(thickness[row:row+side,col:col+side][m]) <= 0.3:
                continue  # retain all three nominal Jornada layers in this bounded case
            choices.append((abs(frac - 0.8), row, col))
    if not choices:
        raise ValueError("No admitted real crop; do not invent a footprint")
    _, row, col = min(choices)
    cut = np.s_[row:row+side, col:col+side]
    ring = np.s_[row-1:row+side+1, col-1:col+side+1]

    def write_asc(path, values):
        dx = header["cellsize"]
        lines = [f"ncols {side}", f"nrows {side}", f"xllcorner {header['xllcorner'] + col*dx:.17g}",
            f"yllcorner {header['yllcorner'] + (surface.shape[0]-row-side)*dx:.17g}",
            f"cellsize {dx:.17g}", f"NODATA_value {header['nodata_value']:.17g}"]
        with path.open("w") as f:
            f.write("\n".join(lines)+"\n")
            np.savetxt(f, values, fmt="%.17g")

    for name, a in (("surface", surface), ("base", base)):
        write_asc(target / f"case/source/{name}.asc", a[cut])
    # The native Darcy outlets need REAL downslope elevations in the exterior ring.
    # Preserve the observed one-cell ring: no ghost lowering or invented outlet drop.
    np.save(target / "physical_ring_LOCAL_ONLY.npy", np.ascontiguousarray(np.flipud(surface[ring])))

    receipt = json.loads((original / "input_receipt.json").read_text())
    summary_path = Path(receipt["original_dem"]).parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    if sha(receipt["original_dem"]) != receipt["original_sha256"]:
        raise ValueError("original DEM hash differs from MAPLE receipt")
    geo = summary["source"]["dem_geographic_grid"]
    if geo["crs"] != "EPSG:4326" or geo["dx"] <= 0 or geo["dy"] <= 0:
        raise ValueError("expected MAPLE north-origin geographic source grid with positive spacings")
    # A shared site reference, exactly from MAPLE's own original geographic grid.
    # No geospatial dependency/install or new map-projection implementation needed.
    lon = [geo["x0"] + geo["shape"][1] * geo["dx"] / 2]
    lat = [geo["y0"] - geo["shape"][0] * geo["dy"] / 2]
    if not (math.isfinite(lat[0]) and 0 < lat[0] <= 90 and math.isfinite(lon[0]) and -180 <= lon[0] <= 180):
        raise ValueError("invalid northern-hemisphere Moab site reference")
    crs = summary["crs"]

    case = yaml.safe_load((original / "case.yaml").read_text())
    case["run_name"] = "masy_moab_irregular_jornada_small"
    case["geometry"].update(nx=side, ny=side)
    case["location"] = {"latitude_deg": float(lat[0])}
    if case["geometry"]["surface_edge_policy"] != "closed":
        raise ValueError("hybrid requires closed sediment surface edges")
    if not (case["geometry"]["boundary_x"]["kind"] == case["geometry"]["boundary_y"]["kind"] == "periodic"):
        raise ValueError("hybrid requires authored periodic wind boundaries")
    model = yaml.safe_load((original / "model.yaml").read_text())
    model["execution"] = {"backend": "cupy", "device_id": 0}
    model["particle_transport"]["max_active_cohorts"] = 2*side*side*model["particle_transport"]["quantile_bins"]
    model["vegetation_wake"] = {"enabled": True}
    wind = {"wind_forcing": {"blocks": [dict(shear_velocity_m_s=0.6, direction_deg=45.0, duration_s=0.5),
                                          dict(shear_velocity_m_s=0.6, direction_deg=135.0, duration_s=0.5)]}}
    for name, obj in (("case", case), ("model", model), ("wind", wind)):
        (target / f"case/{name}.yaml").write_text(yaml.safe_dump(obj, sort_keys=False))
    (target / "case/slope.yaml").write_bytes((original / "slope.yaml").read_bytes())
    for name, block in zip(("a", "b"), wind["wind_forcing"]["blocks"]):
        (target / f"plan/wind/{name}.yaml").write_text(yaml.safe_dump({"wind_forcing": {"blocks": [block]}}))
    (target / "plan/scenario.csv").write_text(
        "# format: masy.daily_scenario.v1\n# scenario_id: moab_jornada_small\n# weather_file: weather.csv\n"
        "# start_date: 2030-06-01\n# end_date: 2030-06-08\n# evaporation_enabled: true\n"
        "# transpiration_enabled: true\n# growth_enabled: true\n# growth_interval_days: 7\n"
        "# output_interval_days: 1\n# checkpoint_interval_days: 365\n# output_at_event_boundaries: true\n"
        "date,event_type,event_file\n2030-06-01,wind,wind/a.yaml\n2030-06-04,wind,wind/b.yaml\n")
    (target / "plan/weather.csv").write_text(
        "# format: masy.daily_weather.v1\n# pet_method: explicit\n# precipitation_threshold_mm: 5\n"
        "# source: Synthetic mechanical test forcing; no field validation\n"
        "date,tmean_c,pet_mm_day,precipitation_mm,precipitation_intensity_mm_h\n" +
        "".join(f"2030-06-{i:02d},25,2.592,{5 if i==2 else 6 if i==3 else 0},{50 if i==2 else 360 if i==3 else 0}\n" for i in range(1,9)))
    species_path = leaf_case / "species.json"
    catalog = json.loads(species_path.read_text())
    if catalog["root_layers"] != 3:
        raise ValueError("hybrid root remapping requires the documented three-layer source catalog")
    betas = {"BOER": 0.977, "PROGLA": 0.983}
    root_choices = {}
    for sid, beta in betas.items():
        record = catalog["species"][sid]["roots"]
        old = list(record["layer_fractions"])
        weights = [(beta**a - beta**b)/(1-beta**30) for a,b in ((0,10),(10,20),(20,30))]
        record["layer_fractions"] = weights
        record["notes"] += " MASY hybrid user direction2026-11-23: SAME documented beta curve integrated over0,10,20,30cm and normalized within30cm; old1.2m fractions not reused."
        root_choices[sid] = dict(beta=beta, old=old, remapped=weights, conditional_depth_cm=30)
    catalog["catalog_id"] += "_masy_30cm_user_20261123"
    (target / "species.json").write_text(json.dumps(catalog, indent=2)+"\n")
    generation = json.loads((leaf_case / "generation.json").read_text())
    generation["domain"] = dict(nx=side, ny=side, dx_m=header["cellsize"], dy_m=header["cellsize"], periodic=[True,True])
    generation["notes"] = "Bounded hybrid crop. Per-species cover0.10 over active footprint, tolerance0.005; original LEAF sizes/ages/exclusion retained."
    generation["cover_tolerance"] = 0.005
    generation["max_iterations"] = 24
    for sid in betas:
        generation["species"][sid]["target_cover"] = 0.1
    (target / "generation.json").write_text(json.dumps(generation, indent=2)+"\n")
    src = syrup
    for rel, name in (("cases/subsurface_validation/jorn_defaults.json", "jorn_defaults.json"),
        ("docs/subsurface_hydrology/jorn_fc_sensitivity/data/case_numba_fc10_s1p5.json", "jorn_soil_reference.json")):
        (target / name).write_bytes((src / rel).read_bytes())
    leaf_runtime = tomllib.loads((leaf_case/'case.toml').read_text())
    controls = dict(schema="masy.terrain_controls.v1", friction_factor=5.0, soil_property_mode="supplied_final",
        outlet_boundary_policy="close_uphill_at_water_boundary",
        plant_session=dict(rng_seed=leaf_runtime['runtime']['rng_seed'],
            patch_size_m=leaf_runtime['reproduction']['reproduction_patch_size_m']),
        initial_relative_saturation=0.25, matrix_history=dict(initial_wetting_m=0.0, initial_cumulative_intake_m=0.0),
        event_time=dict(max_dt_s=0.2, min_dt_s=1/1024, max_retries=12, max_steps=50000),
        completion=dict(hold_s=1.0, max_depth_change_m=1e-4, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True),
        day_time=dict(max_dt_s=3600.0, min_dt_s=1/1024, max_retries=12, max_steps=50000),
        evaporation_partition=dict(canopy_intercept=1.25, canopy_slope=0.5, transpiration_slope=1.0,
            transpiration_offset=0.125, soil_factor=1.0),
        wind_vegetation=dict(statuses=["live","dead_standing"], unsupported_policy="exclude", drag_coefficient=1.0),
        preferential_enabled=False, sediment_reference="native_syrup_plot1_sand",
        note="Explicit illustrative mechanical controls from the accepted native integration fixtures, E=1 from JORN; not field calibration")
    (target / "coupling_controls.json").write_text(json.dumps(controls, indent=2)+"\n")
    manifest = {str(p.relative_to(target)): sha(p) for p in target.rglob("*") if p.is_file()}
    for name, digest in source_hashes.items():
        if sha(name) != digest:
            raise RuntimeError('source input changed during preparation: '+name)
    record = dict(schema="masy.hybrid_case_inputs.v1", shape=[side,side],
        input_builder_sha256=sha(__file__), selected_source_input_sha256=source_hashes,
        nodata_value=header['nodata_value'],
        source_case=str(original), source_surface_sha256=sha(original / "source/surface.asc"),
        source_base_sha256=sha(original / "source/base.asc"), original_raster_sha256=receipt["original_sha256"],
        crop_top_row=row, crop_left_col=col, n_active=int(valid[cut].sum()),
        physical_samples="Exact decoded original MAPLE samples; no interpolation or elevation/base change",
        exterior_ring="Real adjacent original DEM samples; only observed positive-drop faces may be Darcy outlets",
        location=dict(latitude_deg=float(lat[0]), longitude_deg=float(lon[0]), source_crs=crs,
            method="centre of MAPLE original EPSG4326 source grid, shared site reference for this crop",
            source_summary_sha256=sha(summary_path)),
        roots=root_choices, generation_tolerance=0.005,
        forcing="Synthetic 8days,2 half-second winds,5mm reduced equality and6mm full storm; native recession",
        files=manifest, acceptance=False)
    (target / "input_manifest.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k:record[k] for k in ("shape","n_active","crop_top_row","crop_left_col","location")}))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
