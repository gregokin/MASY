"""Labelled small synthetic Dense CUDA control; existing native laws only."""
from pathlib import Path
import datetime as dt
import dataclasses
import hashlib
import json
import os
import shutil
from types import SimpleNamespace


def fixture(root, *, initial_surface_depth=0.0, saturation=0.5, bottom="open", precipitation=0.01, threshold=0.005):
    import numpy as np
    import cupy as cp
    import yaml
    from leaf import growth,plant_session as ps,state,traits
    from leaf import floating_lifecycle_bridge_cuda as fb,water_vegetation as lw
    from maple.core.backend import to_host
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple_syrup import native_case_profile as ncp,native_dense_cuda_profile as nd,native_floating_soil as nfs
    from maple_syrup import soil_evaporation as se,weather_pet as wp,sediment_physics as sp,water_vegetation_binding as vb
    from maple_syrup.subsurface import default_host_available_bytes
    from maple_syrup.floating_soil_layers_cuda import device_reader
    from maple_syrup.pond_event import PondEvent,PondEventControl,PondCompletionPolicy
    from maple_syrup.experimental_storm import ExperimentalControl
    from maple_syrup.pond_network import build_pond_topology
    from maple_syrup.rainfall import constant_rainfall
    from masy import daily_wind_cuda as gw,daily_triplet_cuda as gt
    from masy.daily_csv import load_daily_scenario
    from masy.daily_schedule import DailyPlan
    from masy.daily_wind_handoff import WindBridgeLimits
    from masy.gpu_contract import GpuPolicy
    from test_daily_triplet import limits
    from wind_handoff_support import make_plan,handoff_limits,wind_limits
    root=Path(root);make_plan(root/'plan',precipitation=precipitation,threshold=threshold)
    f=root/'plan/scenario.csv';s=f.read_text().replace('end_date: 2030-06-02','end_date: 2030-06-08').replace('transpiration_enabled: false','transpiration_enabled: true').replace('growth_enabled: false','growth_enabled: true');f.write_text(s)
    f=root/'plan/weather.csv';f.write_text(f.read_text()+''.join(f'2030-06-{i:02d},25,2.592,0,0\n' for i in range(3,9)))
    plan=DailyPlan(load_daily_scenario(root/'plan/scenario.csv'))
    bundle=Path(os.environ['MASY_DAILY_NATIVE_BUNDLE']); syrup=Path(os.environ['MASY_DAILY_NATIVE_SYRUP']); leaf=Path(os.environ['MASY_TRIPLET_LEAF'])
    policy=GpuPolicy(0,os.environ['CUDA_VISIBLE_DEVICES'],1<<30,1<<30,1<<20,1<<20,default_host_available_bytes,device_reader(cp));policy.gate()
    source=ncp.verify_source_profile(bundle,expected_syrup_digest=gw.SYRUP_PACKAGE_SHA256,profile=ncp.CURRENT_CUDA_PROFILE_NAME)
    case=root/'native/case';case.mkdir(parents=True)
    for name in ('case.yaml','model.yaml','wind.yaml'): shutil.copyfile(syrup/'docs/manuals/examples/native_leaf_calendar_cli/case'/name,case/name)
    f=case/'case.yaml';s=yaml.safe_load(f.read_text());s['geometry']['nx']=40;f.write_text(yaml.safe_dump(s))
    f=case/'model.yaml';s=yaml.safe_load(f.read_text());s['vegetation_wake']={'enabled':True};s.setdefault('execution',{})['backend']='cupy';f.write_text(yaml.safe_dump(s))
    admission=ncp.admit_package_case(source,case)
    prepared=nd.prepare_native_dense_cuda_case(admission,device_id=0,caller_host_reserve_bytes=1<<20,caller_device_reserve_bytes=1<<20,
        host_available_bytes_fn=policy.host_available_bytes_fn,device_available_bytes_fn=policy.device_available_bytes_fn)
    bed,context=prepared.bed,prepared.context;g=context.geometry;shape=(g.ny,g.nx);N=g.ny*g.nx;mask=np.ones(shape,dtype=bool)
    if initial_surface_depth:
        bed=dataclasses.replace(bed,water=dataclasses.replace(bed.water,depth_m=cp.full(shape,initial_surface_depth,dtype=cp.float64)))
    cfg=nfs.native_floating_config(layer_thickness_m=(0.03,0.07,0.10),grain_texture=np.tile(np.array([0.05,0.45,0.50]),(3,1)),
        organic_matter_percent=np.full(3,3.0),bulk_density_kg_m3=g.bulk_density_kg_m3,latent_theta_field_capacity=np.full(3,0.25),latent_theta_saturation=np.full(3,0.42),
        theta_residual=0.05,theta_evaporation_stress=0.2,diffusivity_m2_s=np.full(2,1e-8),bottom_drainage=bottom,matrix_conductivity_m_s=2e-6,
        matrix_suction_m=0.1,matrix_vegetation_factor=0.7,initial_wetting_m=0.01,initial_cumulative_intake_m=0.02,backend='cuda')
    nb=nfs.NativeBed(bed.voxel_column,bed.active_layer,g,context.grain_classes,context.mass_resolution_kg,mask)
    q=dict(available_bytes_fn=policy.device_available_bytes_fn,reserve_bytes=1<<20,memory_budget_bytes=1<<29,host_available_bytes_fn=policy.host_available_bytes_fn,host_reserve_bytes=1<<20,host_memory_budget_bytes=1<<29)
    owner=nfs.initialize_native_floating_soil(nb,cfg,initial_relative_saturation=saturation,**q)
    z=to_host(committed_elevation_m(bed.committed_topography)); full=np.pad(z,1,mode='edge');outlets=[(r,g.nx-1,0,1) for r in range(g.ny)]
    for r,c,dr,dc in outlets: full[r+1+dr,c+1+dc]=z[r,c]-0.01
    topology=build_pond_topology(full,mask,np.full(shape,5.0),g.dx_m,outlets)
    pick=lambda a:np.asarray(a)[[0,2,5]]
    sediment=sp.sediment_physics_parameters(diameter_m=np.array([c.diameter_m for c in context.grain_classes.classes]),raindrop_a=pick(sp.PLOT1_RAINDROP_A),raindrop_b=pick(sp.PLOT1_RAINDROP_B),raindrop_c=pick(sp.PLOT1_RAINDROP_C),raindrop_max_depth_mm=pick(sp.PLOT1_RAINDROP_MAX_MM),raindrop_depth_attenuation_per_cm=pick(sp.LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM),particle_density_g_cm3=sp.PLOT1_PARTICLE_DENSITY_G_CM3,active_layer_sensitivity_mm=sp.PLOT1_ACTIVE_LAYER_SENSITIVITY_MM,ke_model=sp.PLOT1_KE_MODEL,xp=cp)
    control=PondEventControl(backend='cuda',time=ExperimentalControl(max_dt_s=0.2,min_dt_s=1/1024,max_retries=12,max_steps=50000))
    completion=PondCompletionPolicy(hold_s=1.0,max_depth_change_m=1e-4,max_unit_discharge_m2_s=1e-3,deposit_mobile_at_flow_end=True)
    def builder(vertical,topology,cover,schedule):
        return PondEvent(context,None,sediment,constant_rainfall(0,1,0) if schedule is None else schedule,topology,control=control,policy=completion,floating_vertical=vertical,vegetation_cover_fraction=cover)
    def initial(vertical,carried_topology=None,schedule=None): return builder(vertical,topology if carried_topology is None else carried_topology,None,schedule)
    doc=json.loads((leaf/'examples/phase1/species.json').read_text());doc['schema_version']=2;doc['root_layers']=3
    for r in doc['species'].values():r['roots']['layer_fractions']=[1/3]*3;r['hydrology']={traits.RECEIVER_WEIGHTS_KEY:[0.0,1.0,1.0]}
    grass=doc['species']['grass'];grass['growth']['traits'].update(mature_biomass_kg=1.0,intrinsic_rate_per_year=0.5,biomass_support_kg_year_per_m3=10.0,live_fraction=0.5,extra_growth_rate_per_year=0.2,monthly_activity=[1.0]*12,temperature_min_c=0.0,temperature_opt_c=20.0,temperature_max_c=40.0);grass['water_use']['traits']['transpiration_coefficient']=0.5
    catalog=traits.SpeciesCatalog(doc);domain=state.Domain(g.nx,g.ny,g.dx_m,dy_m=g.dy_m,periodic=(True,True))
    population=state.PlantState.from_records(catalog,domain,[dict(plant_id=11,species='grass',x_m=2.5*g.dx_m,y_m=1.5*g.dy_m,establishment_ground_m=float(z[1,2]),agb_dry_kg=0.1,seasonal_agb_dry_kg=0.0,age_days=100.0,status='live')])
    config=ps.SessionConfig(catalog,growth.compile_species(catalog,'cuda',1<<28,device=0),'cuda',1<<16,1<<28,7,reserve_bytes=1<<24,builder='numpy',canopy={'max_edges':1<<16,'max_bytes':1<<27,'builder':'numpy'})
    bridge=fb.bind_source(source,expected_syrup_digest=gw.SYRUP_PACKAGE_SHA256,expected_syrup_commit=os.environ['MASY_TRIPLET_SYRUP_COMMIT'],expected_maple_commit=gw.MAPLE_COMMIT,max_bytes=1<<27)
    package=lw.package_source_record()['digest'].removeprefix('sha256:');helper=leaf/'benchmarks/spatial_mapping.py';helpers=[dict(module='benchmarks.spatial_mapping',bytes=helper.stat().st_size,sha256=hashlib.sha256(helper.read_bytes()).hexdigest())]
    wind=gt.prepare_triplet_wind_cuda(plan,source,admission,context,wind_limits=wind_limits(),handoff_limits=handoff_limits(),bridge_limits=WindBridgeLimits(1<<20,1<<20),policy=policy,run_inputs_sha256=hashlib.sha256(b'labelled-synthetic-triplet-fixture-v1').hexdigest())
    site=vb.site_controls(a1=0.1,a2=1.0,a3=1.0,max_fraction_per_hour=0.5,max_intake_rate_m_s=1e-5,background_receiver_weights=[0.0,1.0,1.0],n_layers=3,n_cells=N,max_bytes=1<<20)
    terrain=wp.terrain_factor(cp.zeros(N,dtype=cp.int32),cp.zeros(N,dtype=cp.float64),35.0,backend='cuda')
    session=gt.DailyCudaTripletSession(plan,wind=wind,owner=owner,bed=bed,initial_event_factory=initial,plant_state=population,leaf_config=config,anchor=dt.datetime(2030,6,1,tzinfo=dt.timezone.utc),leaf_source=bridge,terrain=terrain,partition=se.EvaporationPartition(1.25,0.5,1.0,0.125,0.75),site_controls=site,active_mask=mask,event_builder=builder,day_control=ExperimentalControl(max_dt_s=3600.0,min_dt_s=1/1024,max_retries=12,max_steps=50000),wind_vegetation_policy=gt.WindVegetationPolicy(('live','dead_standing'),'exclude',1.0),limits=limits(),expected_leaf_package_sha256=package,expected_external_helpers=helpers,policy=policy)
    return session,SimpleNamespace(source=source,admission=admission,prepared=prepared,context=context,bed=bed,owner=owner,policy=policy,original_topology=topology)
