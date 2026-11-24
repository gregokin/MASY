"""Bounded irregular-terrain CUDA runner, entered through the sealed bootstrap.

The input directory binds case, daily CSVs, native vegetation initialization and
an explicit three-layer soil reference. All transport, water and biology run in
their owning packages. Output snapshots are diagnostics, not combined restarts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def validate_inputs(root, expected_sha256):
    root = Path(root).resolve(strict=True)
    manifest = root / 'input_manifest.json'
    if manifest.is_symlink() or manifest.stat().st_size > 1 << 20 or _sha_file(manifest) != expected_sha256:
        raise ValueError('input manifest does not match its external SHA256')
    record = json.loads(manifest.read_text())
    if record.get('schema') != 'masy.hybrid_case_inputs.v1' or type(record.get('files')) is not dict:
        raise ValueError('versioned terrain input manifest required')
    if len(record['files']) > 64:
        raise ValueError('too many input files')
    total = 0
    for rel, sha in record['files'].items():
        p = root / rel
        if (type(rel) is not str or Path(rel).is_absolute() or '..' in Path(rel).parts
                or p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root)):
            raise ValueError('only real input files within the selected directory are accepted')
        total += p.stat().st_size
        if p.stat().st_size > 16 << 20 or total > 64 << 20 or _sha_file(p) != sha:
            raise ValueError('input body changed or exceeds its bounded read budget: ' + rel)
    return root, record, expected_sha256


def _freeze_inputs(selected, destination):
    """Verified small input copies; the model never rereads mutable originals."""
    root, record, expected_sha = selected
    destination.mkdir(exist_ok=False)
    total = 0
    for rel, expected in {**record['files'], 'input_manifest.json': expected_sha}.items():
        source = root / rel
        before = source.stat()
        if source.is_symlink() or before.st_size > 16 << 20:
            raise ValueError('input changed before its bounded snapshot')
        with source.open('rb') as f:
            raw = f.read((16 << 20) + 1)
        after = source.stat()
        if (before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('input changed during its snapshot: ' + rel)
        total += len(raw)
        if len(raw) > 16 << 20 or total > 65 << 20 or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('input copy differs from its external manifest: ' + rel)
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        target.chmod(0o444)
    validate_inputs(root, expected_sha)
    return validate_inputs(destination, expected_sha)


def terrain_maps(full, mask, dx, nodata):
    """Declared geometric data conversion: centred differences, one-sided at NoData.

    Uses real exterior-ring samples where available, with no periodic height
    seam. A missing pair supplies no gradient along that axis; it never fills
    terrain. The owning weather API supplies all PET response arithmetic.
    """
    import numpy as np
    z = full[1:-1, 1:-1]
    known = np.isfinite(full) & (full != nodata)
    known[1:-1, 1:-1] &= mask
    def axis(plus, minus, pk, mk):
        return np.where(pk & mk, (plus-minus)/(2*dx),
            np.where(pk, (plus-z)/dx, np.where(mk, (z-minus)/dx, 0.0)))
    gx = axis(full[1:-1, 2:], full[1:-1, :-2], known[1:-1, 2:], known[1:-1, :-2])
    gy = axis(full[2:, 1:-1], full[:-2, 1:-1], known[2:, 1:-1], known[:-2, 1:-1])
    gx, gy = np.where(mask, gx, 0.0), np.where(mask, gy, 0.0)
    slope = np.arctan(np.hypot(gx, gy))
    downslope_deg = np.degrees(np.arctan2(-gx, -gy)) % 360.0
    aspect = ((np.floor((downslope_deg + 22.5)/45).astype(np.int32) % 8) + 1)
    aspect = np.where((slope == 0) | ~mask, 0, aspect).astype(np.int32)
    return np.ascontiguousarray(aspect.reshape(-1)), np.ascontiguousarray(slope.reshape(-1))


def _outlets(full, active, nodata):
    """Only native D4 export faces with a real positive observed bed drop."""
    import numpy as np
    ny, nx = active.shape
    result = []
    for row, col in zip(*np.nonzero(active)):
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            rr, cc = row+dr, col+dc
            if 0 <= rr < ny and 0 <= cc < nx:
                continue
            outside = full[rr+1, cc+1]
            if np.isfinite(outside) and outside != nodata and full[row+1, col+1] > outside:
                result.append((int(row), int(col), dr, dc))
    if not result:
        raise ValueError('no real downslope outlet; no ghost slope may be introduced')
    return result


def build_session(*, source, sources, inputs, output, policy):
    """Construct actual native owners from explicit local inputs, without advancing."""
    import numpy as np
    from leaf import growth, initialization, plant_session as ps, traits, state
    from leaf import floating_lifecycle_bridge_cuda as fb
    from maple.core.backend import to_host
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple_syrup import native_case_profile as ncp, native_dense_cuda_profile as nd, native_floating_soil as nfs
    from maple_syrup import soil_evaporation as se, weather_pet as wp, sediment_physics as sp
    from maple_syrup import water_vegetation_binding as vb, subsurface as ss
    from maple_syrup import floating_wind_handoff_cuda as wh
    from maple_syrup.experimental_storm import ExperimentalControl
    from maple_syrup.pond_event import PondEvent, PondEventControl, PondCompletionPolicy
    from maple_syrup.pond_network import build_pond_topology
    from maple_syrup.rainfall import constant_rainfall
    from masy import daily_terrain_cuda as terrain
    from masy.daily_csv import load_daily_scenario
    from masy.daily_schedule import DailyPlan
    from masy.daily_triplet import TripletLimits
    from masy.daily_triplet_wind import WindVegetationPolicy
    from masy.daily_wind import DailyWindLimits
    from masy.daily_wind_handoff import WindBridgeLimits

    cp = policy.gate()
    out = Path(output)
    out.mkdir(exist_ok=False)
    root, manifest, _input_sha = _freeze_inputs(inputs, out / 'inputs')
    controls = json.loads((root / 'coupling_controls.json').read_text())
    if (controls.get('schema') != 'masy.terrain_controls.v1' or controls.get('preferential_enabled') is not False
            or controls.get('sediment_reference') != 'native_syrup_plot1_sand'):
        raise ValueError('the bounded runner requires explicit matrix-only controls and the native sediment reference')
    case = out / 'case'
    for rel in manifest['files']:
        if rel.startswith('case/'):
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / rel, dst)
    plan = DailyPlan(load_daily_scenario(root / 'plan/scenario.csv'))
    if plan.scenario.settings.start_date.year != plan.scenario.settings.end_date.year:
        raise ValueError('this bounded runner has no qualified annual vegetation boundary orchestration')
    admission = ncp.admit_package_case(source, case)
    prepared = nd.prepare_native_dense_cuda_case(admission, device_id=policy.device_id,
        caller_host_reserve_bytes=policy.host_reserve_bytes, caller_device_reserve_bytes=policy.device_reserve_bytes,
        host_available_bytes_fn=policy.host_available_bytes_fn, device_available_bytes_fn=policy.device_available_bytes_fn)
    bed, context = prepared.bed, prepared.context
    g = context.geometry
    if (context.domain is None or g.nx*g.ny > 16384
            or len(context.grain_classes.classes) != 1
            or context.grain_classes.classes[0].diameter_m != 0.0002):
        raise ValueError('this bounded hybrid runner requires an irregular footprint <=16384cells and one200um sand class')
    mask = np.ascontiguousarray(to_host(context.domain.valid_mask))
    N = mask.size
    z = to_host(committed_elevation_m(bed.committed_topography))
    full = np.load(root / 'physical_ring_LOCAL_ONLY.npy', allow_pickle=False)
    nodata = float(manifest['nodata_value'])
    if full.shape != (g.ny+2, g.nx+2) or not np.array_equal(full[1:-1, 1:-1][mask], z[mask]):
        raise ValueError('physical ring/interior differ from actual compiled world elevations')
    topology = build_pond_topology(full, mask, np.full(mask.shape, controls['friction_factor']), g.dx_m,
        _outlets(full, mask, nodata), nodata_value=nodata, allow_closed_components=True)
    preset = json.loads((root / 'jorn_defaults.json').read_text())
    reference = json.loads((root / 'jorn_soil_reference.json').read_text())
    if _sha_file(root / 'jorn_soil_reference.json') != preset['source']['case_file_sha256']:
        raise ValueError('soil reference differs from the explicit preset source hash')
    p, layers = preset['parameters'], reference['layers']
    if (preset['schema'] != 'syrup.jorn_defaults/1' or layers['edges_m'] != [0, 0.1, 0.2, 0.3]
            or p['theta_field_capacity_all_layers'] != layers['theta_field_capacity'][0]
            or any(v != layers['theta_field_capacity'][0] for v in layers['theta_field_capacity'])
            or p['conductivity_multiplier'] != 1.0 or p['interface_conductivity_method'] != 'harmonic'
            or p['theta_evaporation_stress'] != layers['theta_evaporation_stress']
            or p['diffusion_coefficient'] != reference['physics']['diffusion_coefficient']
            or p['evaporation_factor'] != reference['physics']['evaporation_factor']):
        raise ValueError('the explicit selected three-layer equal-depth soil preset/reference disagree')
    # Use the owning donor-depth law for the equal nominal-layer constant
    # mapping. Its representational roundoff is checked below; no equation is copied.
    law = ss.soil_layers(layers['edges_m'], theta_residual=layers['theta_residual'],
        theta_field_capacity=layers['theta_field_capacity'], theta_saturation=layers['theta_saturation'],
        theta_evaporation_stress=p['theta_evaporation_stress'], diffusion_mode='tietjen_donor_depth',
        diffusion_coefficient=p['diffusion_coefficient'], interface_conductivity_m_s=layers['interface_conductivity_m_s'])
    if not np.allclose(law.diffusivity_down_m2_s, law.diffusivity_up_m2_s, rtol=8*np.finfo(float).eps, atol=0):
        raise ValueError('unequal-layer directional diffusion is not a constant-diffusivity case')
    ks = [r['Ks_m_s'] for r in reference['provenance']['layer_properties']]
    override = dict(theta_33_override=np.tile(np.array(layers['theta_field_capacity'])[:, None], (1, N)),
        theta_saturation_override=np.tile(np.array(layers['theta_saturation'])[:, None], (1, N)),
        saturated_conductivity_override_m_s=np.full((1,N),ks[0]))
    config = nfs.native_floating_config(layer_thickness_m=(0.1, 0.1, 0.1),
        grain_texture=np.tile(np.array([1.0, 0.0, 0.0]), (len(context.grain_classes.classes), 1)),
        organic_matter_percent=np.zeros(3), bulk_density_kg_m3=g.bulk_density_kg_m3,
        latent_theta_field_capacity=np.array(layers['theta_field_capacity']), latent_theta_saturation=np.array(layers['theta_saturation']),
        theta_residual=float(layers['theta_residual'][0]), theta_evaporation_stress=p['theta_evaporation_stress'],
        evaporation_factor=p['evaporation_factor'], diffusivity_m2_s=np.array(law.diffusivity_down_m2_s),
        bottom_drainage='open', matrix_conductivity_mode='texture_ks',
        latent_saturated_conductivity_m_s=ks[-1], matrix_suction_m=reference['matrix']['suction_m'],
        matrix_vegetation_factor=reference['matrix']['local_conductivity_factor'],
        **controls['matrix_history'], overrides=override, backend='cuda', property_mode=controls['soil_property_mode'])
    quota = dict(available_bytes_fn=policy.device_available_bytes_fn, reserve_bytes=policy.device_reserve_bytes,
        memory_budget_bytes=1 << 29, host_available_bytes_fn=policy.host_available_bytes_fn,
        host_reserve_bytes=policy.host_reserve_bytes, host_memory_budget_bytes=1 << 29)
    owner = nfs.initialize_native_floating_soil(nfs.NativeBed(bed.voxel_column, bed.active_layer, g,
        context.grain_classes, context.mass_resolution_kg, mask), config,
        initial_relative_saturation=controls['initial_relative_saturation'], **quota)
    if np.min(owner.depth_m[mask.reshape(-1)]) < 0.3:
        raise ValueError('the equal nominal-layer diffusion mapping requires the full modeled soil depth')
    sediment = sp.sediment_physics_parameters(
        diameter_m=np.array([c.diameter_m for c in context.grain_classes.classes]),
        raindrop_a=np.array(sp.PLOT1_RAINDROP_A)[[-1]], raindrop_b=np.array(sp.PLOT1_RAINDROP_B)[[-1]],
        raindrop_c=np.array(sp.PLOT1_RAINDROP_C)[[-1]], raindrop_max_depth_mm=np.array(sp.PLOT1_RAINDROP_MAX_MM)[[-1]],
        raindrop_depth_attenuation_per_cm=np.array(sp.LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM)[[-1]],
        particle_density_g_cm3=sp.PLOT1_PARTICLE_DENSITY_G_CM3,
        active_layer_sensitivity_mm=sp.PLOT1_ACTIVE_LAYER_SENSITIVITY_MM, ke_model=sp.PLOT1_KE_MODEL, xp=cp)
    event_control = PondEventControl(backend='cuda', time=ExperimentalControl(**controls['event_time']))
    completion = PondCompletionPolicy(**controls['completion'])
    def event_builder(vertical, carried, cover, schedule):
        return PondEvent(context, None, sediment, constant_rainfall(0, 1, 0) if schedule is None else schedule,
            carried, control=event_control, policy=completion, floating_vertical=vertical, vegetation_cover_fraction=cover)
    def initial(vertical, carried_topology=None, schedule=None):
        return event_builder(vertical, topology if carried_topology is None else carried_topology, None, schedule)
    catalog = traits.SpeciesCatalog(json.loads((root / 'species.json').read_text()))
    population, generation = initialization.generate_population(catalog,
        json.loads((root / 'generation.json').read_text()), surface=z, active_mask=mask)
    state.write_plant_csv(population, out / 'plants_initial.csv', initial=True, provenance={'inputs': manifest['source_surface_sha256']})
    (out / 'generation.json').write_text(json.dumps(generation, indent=2)+'\n')
    plant_config = ps.SessionConfig(catalog, growth.compile_species(catalog, 'cuda', 1 << 29, device=policy.device_id),
        'cuda', 1 << 20, 1 << 29, plan.scenario.settings.growth_interval_days, reserve_bytes=1 << 27,
        builder='numpy', canopy={'max_edges': 1 << 20, 'max_bytes': 1 << 28, 'builder': 'numpy'},
        rng_seed=controls['plant_session']['rng_seed'], patch_size_m=controls['plant_session']['patch_size_m'])
    bridge = fb.bind_source(source, expected_syrup_digest=terrain.SYRUP_PACKAGE_SHA256,
        expected_syrup_commit=sources['syrup']['commit'], expected_maple_commit=terrain.MAPLE_COMMIT, max_bytes=1 << 28)
    cohorts = admission.compiled.config.particle_transport.max_active_cohorts
    wind = terrain.prepare_terrain_triplet_wind_cuda(plan, source, admission, context,
        wind_limits=DailyWindLimits(16384, 140, 524288, 16, 10000, 1 << 28, 1 << 30, 1 << 20),
        handoff_limits=wh.WindHandoffLimits(max_cells=16384, max_voxels=140, max_classes=16,
            max_cohorts=cohorts, max_batch_cells=64, max_scan_bytes=1 << 28,
            max_candidate_bytes=1 << 29, max_receipt_bytes=1 << 20),
        bridge_limits=WindBridgeLimits(1 << 20, 1 << 20), policy=policy,
        run_inputs_sha256=_sha_file(root / 'input_manifest.json'), terrain_source=prepared.source)
    site_controls = vb.site_controls(a1=reference['matrix']['vegetation_a1'], a2=reference['matrix']['vegetation_a2'],
        a3=0.0, max_fraction_per_hour=0.0, max_intake_rate_m_s=0.0, background_receiver_weights=None,
        n_layers=3, n_cells=N, max_bytes=1 << 20)
    aspect, slope = terrain_maps(full, mask, g.dx_m, nodata)
    factor = wp.terrain_factor(cp.asarray(aspect), cp.asarray(slope), admission.require_latitude_for_pet(), backend='cuda')
    def terrain_builder(current):
        observed = full.copy()
        observed[1:-1, 1:-1][mask] = current[mask]
        new_aspect, new_slope = terrain_maps(observed, mask, g.dx_m, nodata)
        return wp.terrain_factor(cp.asarray(new_aspect), cp.asarray(new_slope),
            admission.require_latitude_for_pet(), backend='cuda')
    import math
    if any(day.weather.precipitation_mm > 0 and day.weather.precipitation_intensity_mm_h <= 0 for day in plan):
        raise ValueError('wet days require positive intensity for full storm or reduced-overflow fallback')
    rain_seconds = max((3600*day.weather.precipitation_mm/day.weather.precipitation_intensity_mm_h
        for day in plan if day.weather.precipitation_mm > 0), default=0.)
    event_calls = math.ceil(rain_seconds/100.)+32  # explicit finite3200s recession allowance
    if event_calls > 4096:
        raise ValueError('storm duration exceeds this bounded runner event-call cap')
    limits = TripletLimits(100.0, event_calls, 16, 50000, 1 << 28, 1 << 20, 1 << 29,
        1 << 29, 1 << 29, 1 << 28, 1 << 28, 1 << 20, 1 << 20)
    session = terrain.DailyTerrainCudaTripletSession(plan, wind=wind, owner=owner, bed=bed,
        initial_event_factory=initial, plant_state=population, leaf_config=plant_config,
        anchor=dt.datetime.combine(plan.scenario.settings.start_date, dt.time(), dt.timezone.utc),
        leaf_source=bridge, terrain=factor, partition=se.EvaporationPartition(**controls['evaporation_partition']),
        site_controls=site_controls, active_mask=mask, event_builder=event_builder,
        day_control=ExperimentalControl(**controls['day_time']),
        wind_vegetation_policy=WindVegetationPolicy(tuple(controls['wind_vegetation']['statuses']),
            controls['wind_vegetation']['unsupported_policy'], controls['wind_vegetation']['drag_coefficient']), limits=limits,
        expected_leaf_package_sha256=terrain.LEAF_PACKAGE_SHA256, expected_external_helpers=terrain.gt.LEAF_EXTERNAL_HELPERS,
        policy=policy, require_receiver_traits=False, terrain_builder=terrain_builder,
        minimum_soil_depth_m=0.3)
    return session, dict(source=source, admission=admission, prepared=prepared, context=context,
        mask=mask, generation=generation, soil_diffusion=law.summary(), topology=topology,
        initial_biomass_kg=float(population.agb_dry_kg.sum()), output=out)


def main(argv=None, *, runtime):
    from masy.triplet_terrain_runtime import TerrainCudaRuntimeContext
    if type(runtime) is not TerrainCudaRuntimeContext:
        raise ValueError('this transport entry requires the genuine cold terrain runtime')
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args(argv)
    import os
    run_dir = Path(args.run_dir).resolve(strict=True)
    if Path(os.environ.get('CUPY_CACHE_DIR','')).resolve() != run_dir/'cache/cupy':
        raise ValueError('run directory differs from the cold runtime cache binding')
    selected = validate_inputs(args.inputs, args.inputs_sha256)
    facts = runtime.record()
    runtime.recheck()
    import cupy as cp
    import numpy as np
    from maple.core.backend import to_host
    from maple_syrup.subsurface import default_host_available_bytes
    from maple_syrup.floating_soil_layers_cuda import device_reader
    from masy.gpu_contract import GpuPolicy
    policy = GpuPolicy(0, facts['device']['physical_uuid'], 2 << 30, 2 << 30, 1 << 20, 1 << 20,
        default_host_available_bytes, device_reader(cp))
    policy.gate().get_default_memory_pool().set_limit(size=policy.max_device_bytes)
    session, setup = build_session(source=runtime.source_profile, sources=facts['sources'],
        inputs=selected, output=args.output, policy=policy)
    records = []
    try:
        while True:
            try:
                accepted = session.advance()
            except StopIteration:
                break
            record = accepted.record()
            records.append(record)
            head = session.lineage.head
            inactive = cp.asarray(~setup['mask'].reshape(-1))
            if (bool(cp.any(head.root.water_state.water_m[:, inactive] != 0))
                    or bool(cp.any(head.surface_depth_m.reshape(-1)[inactive] != 0))
                    or bool(cp.any(head.root.history.wetting_m.reshape(-1)[inactive] != 0))
                    or bool(cp.any(head.root.history.cumulative_intake_m.reshape(-1)[inactive] != 0))
                    or bool(cp.any(head.bed.voxel_column.mass_kg[cp.asarray(~setup['mask'])] != 0))
                    or bool(cp.any(head.bed.active_layer.mass_kg[cp.asarray(~setup['mask'])] != 0))):
                raise RuntimeError('nonzero water/history/sediment outside physical footprint')
            validate_inputs(args.inputs, args.inputs_sha256)
            setup['admission'].recheck()
            runtime.recheck()
            (setup['output'] / 'progress.json').write_text(json.dumps(dict(accepted_days=session.accepted_days,
                physical_time_s=float(head.t_s), failure=None), indent=2)+'\n')
            print(json.dumps({'accepted_day': record['date'], 'full_storm': record['full_storm_completed']}), flush=True)
    except BaseException:
        (setup['output'] / 'failure.json').write_text(json.dumps(dict(failure=session.failure,
            accepted_days=session.accepted_days, recoverable_combined_checkpoint=False), indent=2)+'\n')
        raise
    head = session.lineage.head
    arrays = {k: to_host(v) for k,v in dict(W=head.root.water_state.water_m,
        J=head.root.history.wetting_m, I=head.root.history.cumulative_intake_m,
        surface=head.surface_depth_m, active=head.bed.active_layer.mass_kg,
        voxels=head.bed.voxel_column.mass_kg, routing_source=head.routing[1]).items()}
    arrays['biomass'] = session.plants.snapshot_state().agb_dry_kg
    np.savez(setup['output'] / 'diagnostic_arrays_LOCAL_ONLY.npz', **arrays)
    result = dict(profile='masy.terrain_runner.cuda.v1', complete=True, records=records,
        accepted_days=session.accepted_days, winds=session.wind.next_wind_index,
        water_ledger=dict(head.ledger), plants=session.plants.counters(),
        initial_biomass_kg=setup['initial_biomass_kg'], final_biomass_kg=float(arrays['biomass'].sum()),
        input_manifest_sha256=args.inputs_sha256, source_lock=facts,
        wind_assumes_dry=True, water_reset=False, combined_restart_supported=False,
        diagnostic_arrays_are_restart=False)
    result['soil_diffusion'] = setup['soil_diffusion']
    result['generation'] = setup['generation']
    result['wind_runtime'] = dict(native_terrain=getattr(session.wind._wind,'_terrain_execution',None),
        native_memory_census=getattr(session.wind._wind,'_last_terrain_census',None),
        source=session.wind._wind.identity())
    result['physical_complete'] = True
    (setup['output'] / 'physical_result.json').write_text(json.dumps(result, indent=2)+'\n')
    from masy.runtime_binary_inventory import inventory
    try:
        binaries = inventory(run_dir)
        (setup['output'] / 'compiled_runtime.json').write_text(json.dumps(binaries, indent=2)+'\n')
        result['compiled_runtime_sha256'] = _sha_file(setup['output'] / 'compiled_runtime.json')
        runtime.recheck()
    except BaseException as exc:
        result.update(complete=False, verification_failure=type(exc).__name__+': '+str(exc)[:500])
        (setup['output'] / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
        raise
    result['versions'] = dict(numpy=np.__version__, cupy=cp.__version__,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cuda_driver=cp.cuda.runtime.driverGetVersion())
    (setup['output'] / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return 0
