"""Cold closed public-case worker; no import-time Native/provider work.

The parent exclusively owns launch authority/deadline/monitoring. Metadata and
physical case modes are separate fresh processes. No pytest or factory tail.
"""
from __future__ import annotations
import copy
import dataclasses
import hashlib
import importlib
import importlib.abc
import io
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import types
from types import SimpleNamespace

_JAX_ENV_SOURCES = (
    ('jax', '/home/okin/MAPLE/.venv-cupy/lib/python3.12/site-packages/jax/__init__.py',
     9024, 'e97031df0826c1ed1399652872281209a5c8d1d64302da6cf8ce158c0bb51f66'),
    ('jax._src.cloud_tpu_init', '/home/okin/MAPLE/.venv-cupy/lib/python3.12/site-packages/jax/_src/cloud_tpu_init.py',
     6441, 'c3372d35eb53d0d762564d25dad6d211768cc50c55b620b6ed7b70e5ff5b5b13'),
)

BOOTSTRAP = r'''
import json,os,sys
if any(n.partition('.')[0] in ('masy','maple','maple_syrup','numpy','cupy','jax','numba') for n in sys.modules):
    raise RuntimeError('case worker must start cold')
if sys.flags.optimize:raise RuntimeError('Native Source assertions require an unoptimized interpreter')
def pairs(items):
    out={}
    for k,v in items:
        if k in out:raise ValueError('duplicate request key')
        out[k]=v
    return out
def bad(x):raise ValueError('nonfinite request token')
with open(sys.argv[1],'rb') as f:raw=f.read((4<<20)+1)
if len(raw)>4<<20:raise ValueError('bounded closed worker request')
r=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
prefix=r['runtime']['sys_path_prefix']
if type(prefix) is not list or not prefix or len(set(prefix))!=len(prefix) or any(type(p)is not str or not os.path.isabs(p) or os.path.realpath(p)!=p or not os.path.isdir(p) for p in prefix):raise ValueError('sealed prefix')
base=list(sys.path);sys.path[:0]=prefix
from masy.case_gpu_worker import main
raise SystemExit(main(r,base))
'''


def _remaining(outer):
    deadline=outer['deadline_monotonic']
    if type(deadline) not in (int,float) or not math.isfinite(deadline) or time.monotonic()>=deadline:
        raise TimeoutError('original whole-operation deadline')


def _jax_environment_sources(*, imported):
    """Two fixed selected Source owners; no generic package/path allowance."""
    from . import case_gpu_cli as cli
    pins=[]
    for name,path,size,sha in _JAX_ENV_SOURCES:
        ref=dict(path=path,size=size,sha256=sha);cli.pinned(ref)
        if imported:
            module=sys.modules.get(name)
            if (type(module)is not types.ModuleType or vars(module).get('__file__')!=path
                    or getattr(vars(module).get('__spec__'),'origin',None)!=path):
                raise ValueError('selected JAX environment owner origin changed')
        pins.append(ref)
    return pins


def _project_native_environment(before,after,cloud_flag):
    """Exactly two Source-proven outcomes; no inferred TPU hardware count."""
    if (type(before)is not dict or type(after)is not dict or cloud_flag is not False
            or any(type(k)is not str or type(v)is not str for d in (before,after) for k,v in d.items())
            or 'TPU_SKIP_MDS_QUERY' in before):
        raise ValueError('genuine pre-Native GPU environment/cloud state required')
    expected=dict(before);expected.setdefault('TF_CPP_MIN_LOG_LEVEL','1')
    branch='no_TPU_skip_assignment_observed'
    if 'TPU_SKIP_MDS_QUERY' in after:
        expected['TPU_SKIP_MDS_QUERY']='1'
        branch='TPU_skip_assignment_observed; chip_count_not_asserted'
    if after!=expected:raise ValueError('unapproved Native environment change')
    return expected,branch


def _reconcile_native_environment(process):
    pins=_jax_environment_sources(imported=True)
    cloud=sys.modules['jax._src.cloud_tpu_init']
    expected,branch=_project_native_environment(process['environment'],dict(os.environ),
        vars(cloud).get('running_in_cloud_tpu_vm'))
    frozen=dict(process);frozen['environment']=expected
    return frozen,dict(Source=pins,observed_branch=branch,cloud_TPU_runtime=False,
        known_Source_raw_current_next_error_bytes=3*sum(p['size'] for p in pins))


def _check_gpu_case_process(identity, *, upcoming=0, checkpoint='entry'):
    from . import case_gpu_cli as cli
    positive=cli.positive
    positive(upcoming,'known HOST prefix',5<<30,0)
    for kind,expected in ((resource.RLIMIT_CPU,(480,540)),(resource.RLIMIT_FSIZE,(64<<20,64<<20)),
                           (resource.RLIMIT_NOFILE,(1024,1024)),(resource.RLIMIT_CORE,(0,0))):
        if resource.getrlimit(kind)!=expected:raise ValueError('original exact process limits required')
    if (os.getpid()!=identity['pid'] or cli.birth(os.getpid())['start_ticks']!=identity['start_ticks']
            or os.getcwd()!=identity['cwd'] or dict(os.environ)!=identity['environment']):
        raise ValueError('same genuine GPU process/environment required')
    from maple.aeolian.cohorts.capacity import default_available_memory_bytes
    available=default_available_memory_bytes()
    if type(available)is not int or 2*upcoming>available:raise ValueError('genuine HOST half-headroom')
    return dict(checkpoint=checkpoint,known_HOST_prefix=upcoming)


def _sources(outer,authority):
    """No Native/provider import; current HEAD, full src blobs and own tool."""
    from . import gpu_runtime as gpu,runtime as rt
    from .sourceobs import run_git
    from . import case_gpu_cli as cli
    _remaining(outer)
    profile,limits,_=gpu._load_profile(authority['store'],authority['profile_id'])
    got={}
    for name,selected in authority['sources'].items():
        _remaining(outer)
        obs=rt.observe_source(name,selected['root'],expected_revision=selected['head'],limits=limits)
        if obs['head']!=selected['head'] or obs['tree']!=selected['tree']:raise ValueError('current committed Source drift: '+name)
        # The profile object is bound to these actual complete source rows.
        obj=profile['objects'][name];verified=rt.verify_source_object(obj['path'],obj['object_id'],limits)
        if obs['package']['digest_sha256']!=verified['package_digest']:raise ValueError('current package differs from selected immutable Source')
        got[name]=obs
    launch=outer['launcher'];selected=authority['sources']['masy']
    if launch['path']!=str(Path(selected['root'])/cli.TOOL):raise ValueError('actual own launcher path')
    committed=run_git(selected['root'],['show',selected['head']+':'+cli.TOOL],limits.source_limits())
    actual=cli.pinned(dict(path=launch['path'],size=launch['size'],sha256=launch['sha256']))
    if committed!=actual:raise ValueError('new GPU case tool differs from its own committed blob')
    rows=rt.ls_scope(selected['root'],selected['head'],['src/masy'],limits)
    manifest=json.loads(rt._read_bounded(Path(profile['objects']['masy']['path'])/'manifest.json',4<<20,'source_identity'))
    by_path={r[0]:r for r in manifest['identity']['files'] if r[0].startswith('src/masy/')}
    if {r[0] for r in rows}!=set(by_path) or any(by_path[rel][3]!=blob for rel,_mode,blob in rows):raise ValueError('all actual committed MASY src blobs required')
    cli.pinned(authority['request']);_remaining(outer)
    return dict(sources=got,launcher=dict(path=launch['path'],size=launch['size'],sha256=launch['sha256']),
                profile_id=profile['profile_id'],request=authority['request'])


def _plain_leaf(value):
    if value is not None and type(value) not in (bool,int,float,str):raise ValueError('Native policy leaf must be plain JSON scalar')
    if type(value)is int and value.bit_length()>63:raise ValueError('Native policy integer domain')
    if type(value)is float and not math.isfinite(value):raise ValueError('finite Native policy value')
    return value


def _record(cls,value,nested=None,profile=None):
    """Code-owned class dispatch, every field explicit; no type tags or eval."""
    from . import case_gpu_cli as cli
    nested={} if nested is None else nested
    fields=dataclasses.fields(cls);expected={f.name for f in fields}
    if profile is not None and 'profile'in expected:expected.remove('profile')
    cli.closed(value,expected,cls.__name__)
    kwargs={}
    for f in fields:
        if f.name=='profile' and profile is not None:kwargs[f.name]=profile
        elif f.name in nested:kwargs[f.name]=nested[f.name](value[f.name])
        else:
            leaf=_plain_leaf(value[f.name]);annotation=f.type
            kind={int:int,float:float,bool:bool,str:str,'int':int,'float':float,'bool':bool,'str':str}.get(annotation)
            if kind is not None and type(leaf)is not kind:raise ValueError(cls.__name__+'.'+f.name+': exact scalar type')
            kwargs[f.name]=leaf
    obj=cls(**kwargs)
    return obj


def decode_native_limits(value):
    from . import case_gpu_cli as cli
    from maple.surface.active_layer.bed_exchange_device import DeviceBedExchangeLimits,DeviceWindowedK1DimensionProfile,DeviceK1DimensionImpactLimits
    from maple.surface.voxels.windowed_delta import WindowedDeltaBudgets
    from maple.surface.avalanche.windowed_device import DeviceWindowedAvalancheLimits,DeviceWindowedAvalancheValidationLimits
    from maple.surface.topographic_commit.windowed_device import DeviceWindowedCommitLimits
    from maple.aeolian.flux.windowed_step_device import DeviceWindowedK1DimensionStepLimits
    from maple.aeolian.scheduler.windowed_device import DeviceWindowedK1DimensionWindBlockLimits
    from maple.aeolian.scheduler.windowed_record import WindowedWindRecordLimits
    cli.closed(value,('profile','record_limits'),'Native policy root')
    profile=_record(DeviceWindowedK1DimensionProfile,value['profile'])
    bed=lambda x:_record(DeviceBedExchangeLimits,x)
    impact=lambda x:_record(DeviceK1DimensionImpactLimits,x,profile=profile)
    delta=lambda x:_record(WindowedDeltaBudgets,x)
    step=lambda x:_record(DeviceWindowedK1DimensionStepLimits,x,dict(bed_limits=impact,creep_limits=bed,delta_budgets=delta),profile)
    solve=lambda x:_record(DeviceWindowedAvalancheLimits,x,dict(bed_limits=bed))
    validation=lambda x:_record(DeviceWindowedAvalancheValidationLimits,x,dict(bed_limits=bed,delta_budgets=delta))
    commit=lambda x:_record(DeviceWindowedCommitLimits,x)
    block=lambda x:_record(DeviceWindowedK1DimensionWindBlockLimits,x,dict(step=step,solve=solve,validation=validation,commit=commit),profile)
    return _record(WindowedWindRecordLimits,value['record_limits'],dict(block=block))


def _decode_case_limits(v,root):
    from .case_workflow import CaseInputLimits
    from .case_windowed import CaseWindowedLimits
    from .case_water import CaseWaterCopyLimits
    from .dense_bed import DenseBedLimits
    from .dense_runner import DenseRunnerLimits
    from .dense_device_resources import DeviceResourceLimits
    from .windowed_device_c64_bed import DeviceWindowedC64BedLimits
    from .case_device_c64 import CaseDeviceC64PlacementLimits
    from maple_syrup.windowed_shared_bed_device import PositiveWindowedDeviceWaterLimits
    result={name:_record(cls,v[name]) for name,cls in (
        ('input_limits',CaseInputLimits),('compile_bed_limits',DenseBedLimits),('windowed_limits',CaseWindowedLimits),
        ('water_copy_limits',CaseWaterCopyLimits),('bed_limits',DeviceWindowedC64BedLimits),
        ('resource_limits',DeviceResourceLimits),('runner_limits',DenseRunnerLimits),('placement_limits',CaseDeviceC64PlacementLimits))}
    from . import case_gpu_cli as cli
    fields={f.name for f in dataclasses.fields(PositiveWindowedDeviceWaterLimits)}
    cli.closed(v['bridge_limits'],fields-{'run_dir'},'full explicit bridge limits')
    kwargs={k:_plain_leaf(x) for k,x in v['bridge_limits'].items()};kwargs['run_dir']=root
    result['bridge_limits']=PositiveWindowedDeviceWaterLimits(**kwargs)
    result['record_limits']=decode_native_limits(v['native_limits'])
    return result


def _bootstrap_native(runtime_root, facts):
    import numpy as np
    runtime_roots = {'maple': str(Path(runtime_root))}
    # CWC captures its original operator image at FIRST import. Do not import,
    # reload, rebind or fabricate that image before its literal owners are loaded.
    import importlib as _operator_importlib
    import types as _operator_types
    _operator_cwc_name = "maple.surface.topographic_commit.closed_windowed_commit"
    _operator_provider_names = ('numpy',
     'numpy._core.numeric',
     'numpy._core.overrides',
     'numpy._core._ufunc_config',
     'json',
     'json.encoder',
     'builtins',
     'weakref',
     'dataclasses',
     'pathlib',
     'posixpath',
     'os',
     'stat',
     'hashlib',
     'io',
     'math',
     'struct',
     'threading')
    _operator_native_names = ('maple.aeolian.cohorts.capacity',
     'maple.core.backend',
     'maple.core.backend.arrays',
     'maple.core.backend.depth_class_reductions',
     'maple.core.backend.namespace',
     'maple.core.backend.numpy_pairwise_order',
     'maple.core.backend.prefix_cell_reductions',
     'maple.core.backend.transforms',
     'maple.core.backend.windowed_domain_reductions',
     'maple.core.boundaries',
     'maple.core.parameters.avalanching',
     'maple.core.parameters.geometry',
     'maple.core.parameters.voxel_growth',
     'maple.core.profiling',
     'maple.core.types.active_layer',
     'maple.core.types.ledger_reconciliation',
     'maple.core.types.sediment_availability',
     'maple.core.types.sediment_ledger',
     'maple.core.types.topographic_commit',
     'maple.core.types.voxel',
     'maple.coupling.sediment_ledger.accumulate',
     'maple.coupling.sediment_ledger.diagnostics',
     'maple.coupling.sediment_ledger.validation',
     'maple.surface.active_layer.bed_exchange',
     'maple.surface.active_layer.capacity',
     'maple.surface.active_layer.diagnostics',
     'maple.surface.active_layer.validation',
     'maple.surface.active_layer.windowed_exchange',
     'maple.surface.active_layer.windowed_session',
     'maple.surface.availability.exchange',
     'maple.surface.availability.validation',
     'maple.surface.avalanche.callback',
     'maple.surface.avalanche.slope',
     'maple.surface.avalanche.solver',
     'maple.surface.avalanche.transfer',
     'maple.surface.avalanche.validation',
     'maple.surface.avalanche.windowed',
     'maple.surface.avalanche.windowed_publication',
     'maple.surface.avalanche.windowed_solver',
     'maple.surface.avalanche.windowed_validation',
     'maple.surface.topographic_commit.commit',
     'maple.surface.topographic_commit.windowed_commit',
     'maple.surface.voxels._numerics',
     'maple.surface.voxels.capacity',
     'maple.surface.voxels.deep_store',
     'maple.surface.voxels.diagnostics',
     'maple.surface.voxels.windowed_admission',
     'maple.surface.voxels.windowed_depth_class',
     'maple.surface.voxels.windowed_evidence',
     'maple.surface.voxels.windowed_inventory',
     'maple.surface.voxels.windowed_native_validation',
     'maple.surface.voxels.windowed_pending',
     'maple.surface.voxels.windowed_prefix',
     'maple.surface.voxels.windowed_publication_check',
     'maple.surface.voxels.windowed_transfer')
    assert _operator_cwc_name not in sys.modules, 'CWC imported before exact owner preamble'
    for _operator_name in (*_operator_provider_names, "numpy.random", *_operator_native_names):
        _operator_module = _operator_importlib.import_module(_operator_name)
        assert type(_operator_module) is _operator_types.ModuleType
        assert sys.modules.get(_operator_name) is _operator_module
        assert vars(_operator_module).get('__name__') == _operator_name
        if _operator_name == 'maple' or _operator_name.startswith('maple.'):
            _operator_root = Path(runtime_roots['maple'])
            _operator_tail = _operator_name.removeprefix('maple.').replace('.', '/')
        elif _operator_name == 'numpy' or _operator_name.startswith('numpy.'):
            _operator_root = Path(np.__file__).resolve().parent
            _operator_tail = _operator_name.removeprefix('numpy.').replace('.', '/') if _operator_name != 'numpy' else ''
        else:
            # These are exact original stdlib provider objects. CWC independently
            # binds/checks their function/code/globals and existing terminal custody
            # verifies actual observed compiled/origin owners; no alternate provider.
            assert _operator_cwc_name not in sys.modules, 'provider imported CWC before owner closure'
            continue
        _operator_file = Path(vars(_operator_module)['__file__'])
        _operator_expected = ((_operator_root / (_operator_tail + '.py')),
                              (_operator_root / _operator_tail / '__init__.py')) if _operator_tail else (_operator_root / '__init__.py',)
        assert _operator_file.is_absolute() and _operator_file.resolve() == _operator_file
        assert _operator_file in _operator_expected, 'operator owner outside exact admitted Source origin'
        assert _operator_cwc_name not in sys.modules, 'native owner imported CWC before owner closure'
    # P3 below performs FIRST CWC after genuine N1; the later unchanged test reuses that exact module.
    # No operator-image getter/constructor, Native operation or token is invoked in this old preamble.

    import time as _qualification_time
    from maple.core.backend import depth_class_reductions as _native_identity
    _native_qualification_orders = []
    for _extent in ((6, 1), (6, 6), (12, 3), (20, 1)):
        _qualified = _native_identity.qualify_depth_class_order(*_extent)
        assert type(_qualified) is _native_identity._DepthClassOrder
        _native_qualification_orders.append(_qualified)

    # P3: FIRST genuine CWC, after the unchanged old preamble and check-only N1.
    _w1_name='maple.aeolian.scheduler.windowed_baseline'
    assert _w1_name not in sys.modules and _operator_cwc_name not in sys.modules
    _held_cwc=_operator_importlib.import_module(_operator_cwc_name)
    assert type(_held_cwc) is _operator_types.ModuleType
    assert sys.modules.get(_operator_cwc_name) is _held_cwc and vars(_held_cwc)['__name__']==_operator_cwc_name
    _cwc_origin=Path(vars(_held_cwc)['__file__'])
    assert _cwc_origin==Path(runtime_roots['maple'])/'surface/topographic_commit/closed_windowed_commit.py'
    assert _cwc_origin.resolve()==_cwc_origin and _w1_name not in sys.modules
    # Genuine current image is constructed by Native; no getter, repair or replacement.
    assert _held_cwc._P_BASE_BINDING_REFERENCE_SLOTS==facts['cwc_base_slots']
    assert _held_cwc._P_K6_BINDING_REFERENCE_SLOTS==facts['cwc_k6_slots']
    assert _held_cwc._P_BINDING_REFERENCE_SLOTS==facts['cwc_slots']
    assert _held_cwc._P_CURRENT_BINDING_REFERENCE_SLOTS==facts['cwc_current_slots']
    assert _held_cwc._P_OPERATOR_ERROR is None and len(_held_cwc._P_OPERATORS)==facts['cwc_operators']
    # P4: actual image19 and direct2 dependencies. No reload or image repair.
    _w1_prior_owners=('maple.aeolian.flux.windowed_step', 'maple.aeolian.flux.windowed_step_admission', 'maple.aeolian.scheduler.baseline', 'maple.aeolian.scheduler.block', 'maple.aeolian.scheduler.commit', 'maple.aeolian.scheduler.accounting', 'maple.aeolian.scheduler.active_layer_composition_cache', 'maple.aeolian.scheduler.wind_modifiers', 'maple.aeolian.scheduler.topographic_wind_cache', 'maple.aeolian.scheduler.invalidation', 'maple.aeolian.scheduler.validation', 'maple.aeolian.scheduler.state_validation', 'maple.aeolian.scheduler.state', 'maple.aeolian.detachment.threshold', 'maple.aeolian.topographic_wind.transfer', 'maple.aeolian.topographic_wind.separation', 'maple.surface.topographic_commit.accounting_reconciliation', 'maple.surface.topographic_commit.initialization', 'maple.surface.topographic_commit.height')
    _w1_direct_owners=('maple.core.backend.leaves', 'maple.core.parameters.wind_forcing')
    for _w1_owner in (*_w1_prior_owners,*_w1_direct_owners):
        _w1_module=_operator_importlib.import_module(_w1_owner)
        assert type(_w1_module) is _operator_types.ModuleType
        assert sys.modules.get(_w1_owner) is _w1_module and vars(_w1_module).get('__name__')==_w1_owner
        _w1_file=Path(vars(_w1_module)['__file__'])
        _w1_tail=_w1_owner.removeprefix('maple.').replace('.','/')
        _w1_root=Path(runtime_roots['maple'])
        assert _w1_file.is_absolute() and _w1_file.resolve()==_w1_file
        assert _w1_file in (_w1_root/(_w1_tail+'.py'),_w1_root/_w1_tail/'__init__.py')
        assert sys.modules.get(_operator_cwc_name) is _held_cwc and _w1_name not in sys.modules
    # Root858 B3 direct Source providers, after genuine FIRST CWC.
    # These ordinary imports mint only their original import-time images;
    # no getter, reload, image repair, model fixture or device query occurs.
    _b3_modules_before = dict(sys.modules)
    _b3_extra_names = (
        'maple.aeolian.scheduler.vegetation_cache',
        'maple.aeolian.vegetation.apply',
        'maple.aeolian.scheduler.windowed_device',
    )
    for _b3_name in _b3_extra_names:
        _b3_module = _operator_importlib.import_module(_b3_name)
        assert type(_b3_module) is _operator_types.ModuleType
        assert sys.modules.get(_b3_name) is _b3_module
        assert _w1_name not in sys.modules
        assert sys.modules.get(_operator_cwc_name) is _held_cwc
    assert all(sys.modules.get(n) is m for n, m in _b3_modules_before.items())
    _b3_added = tuple(sorted(set(sys.modules)-set(_b3_modules_before)))
    del _b3_modules_before
    # Check the genuine device image before FIRST W1; the record image follows W1.
    _b3_device=sys.modules['maple.aeolian.scheduler.windowed_device']
    assert _b3_device._Image.error is None and _b3_device._Image.operators
    _b3_device._check_image()
    # P5: FIRST W1 LAST; all prior module identities remain and no other entry appears.
    _w1_modules_before=dict(sys.modules)
    _w1_module=_operator_importlib.import_module(_w1_name)
    assert type(_w1_module) is _operator_types.ModuleType
    assert sys.modules.get(_w1_name) is _w1_module and vars(_w1_module).get('__name__')==_w1_name
    _w1_file=Path(vars(_w1_module)['__file__'])
    assert _w1_file==Path(runtime_roots['maple'])/'aeolian/scheduler/windowed_baseline.py'
    assert _w1_file.resolve()==_w1_file
    _w1_added=tuple(sorted(set(sys.modules)-set(_w1_modules_before)))
    if _w1_added!=(_w1_name,):
        if len(_w1_added)>4096 or any(type(n)is not str or len(n)>512 for n in _w1_added):
            raise RuntimeError('FIRST W1 unexpected module-name census exceeds bounded diagnostics')
        raise RuntimeError('FIRST W1 expected only '+_w1_name+'; actual added modules='+repr(_w1_added))
    assert all(sys.modules.get(n) is m for n,m in _w1_modules_before.items())
    assert sys.modules.get(_operator_cwc_name) is _held_cwc
    assert _w1_module._Image.error is None and len(_w1_module._Image.operators)==facts['w1_operators']
    _w1_module._check_image()
    _w1_preload_receipt={'old_native_owner_count':len(_operator_native_names),'additional_owner_count':len(_w1_prior_owners),'direct_owner_count':len(_w1_direct_owners),
        'CWC_origin':str(_cwc_origin),'W1_origin':str(_w1_file),'W1_new_module_only':True,
        'prior_module_identities_unchanged':True,'genuine_image_checked_before_fixture':True}
    del _w1_modules_before
    # The shared record's own genuine import-time images bind only after
    # all named Native finalizer providers exist. No image is repaired.
    _b3_finalizer_names = (
        'maple.aeolian.scheduler.finalization',
        'maple.aeolian.scheduler.record',
        'maple.aeolian.cohorts.dispatch', 'maple.aeolian.cohorts.routing',
        'maple.aeolian.feedback.momentum', 'maple.aeolian.feedback.trajectory',
        'maple.aeolian.flux.deposition', 'maple.aeolian.flux.validation',
        'maple.aeolian.launch_direction', 'maple.aeolian.trajectories.drag',
        'maple.core.parameters.impact_rebound_splash',
        'maple.core.types.cohort', 'maple.core.types.cohort_calendar',
        'maple.core.types.regional_wind', 'maple.water.validation',
        'maple.aeolian.scheduler.windowed_finalization',
        'maple.aeolian.scheduler.windowed_record',
    )
    for _b3_name in _b3_finalizer_names:
        _b3_module = _operator_importlib.import_module(_b3_name)
        assert type(_b3_module) is _operator_types.ModuleType
        assert sys.modules.get(_b3_name) is _b3_module
        assert sys.modules.get(_operator_cwc_name) is _held_cwc
        assert sys.modules.get(_w1_name) is _w1_module
    _b3_device = sys.modules['maple.aeolian.scheduler.windowed_device']
    _b3_records = sys.modules['maple.aeolian.scheduler.windowed_record']
    assert _b3_device._Image.error is None and _b3_device._Image.operators
    assert _b3_device._RecordImage.error is None and _b3_device._RecordImage.operators
    _b3_device._check_image()
    _b3_device._check_record_image()
    _w1_preload_receipt['B3_additional_imports_before_W1']=_b3_added
    _w1_preload_receipt['B3_device_image_count']=len(_b3_device._Image.operators)
    _w1_preload_receipt['B3_record_image_count']=len(_b3_device._RecordImage.operators)
    _w1_preload_receipt['B3_finalizer_declarations']=list(_b3_finalizer_names)
    return _native_qualification_orders, _w1_preload_receipt



def _water_npy_header(raw, row, key, g):
    """Original closed numeric header predicate, without loading/mapping data."""
    import numpy as np
    from . import case_cli as cli
    # Native public NPY header readers precede load/mapping. Numeric exact
    # dtype, rank, shape, order and exact file extent are already bounded.
    import io
    stream = io.BytesIO(raw); version = np.lib.format.read_magic(stream)
    if version == (1, 0): shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10000)
    elif version == (2, 0): shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=10000)
    else: raise ValueError('unqualified NPY version')
    expected_shape = (g.ny + 2, g.nx + 2) if key in ('elevation_full_m', 'export_receiver_full') else (g.ny, g.nx)
    expected_dtype = np.dtype('bool') if key in cli._BOOL_ARRAYS else np.dtype('float64')
    payload = math.prod(shape) * dtype.itemsize
    if (shape != expected_shape or list(shape) != row['shape'] or dtype != expected_dtype or fortran
            or dtype.hasobject or stream.tell() + payload != row['size']):
        raise ValueError('water NPY header/layout/payload differs from descriptor/native geometry')
    return payload


def _load_water_arrays_gpu(spec, root, g, policy, copy_limits):
    import numpy as np
    from . import case_cli as cli
    declared = [spec[key] for key in cli._ARRAYS if spec[key] is not None]
    declared_files = sum(row['size'] for row in declared)
    declared_payloads = sum(math.prod(row['shape']) * (1 if row['dtype'] == 'bool' else 8) for row in declared)
    known = 3 * declared_payloads + 2 * declared_files + 10000 * len(declared)
    if declared_files > policy.max_water_input_bytes or known > copy_limits.max_known_host_bytes:
        raise ValueError('prospective NPY raw/header/map/current-next/error owners exceed declared quota')
    _check_gpu_case_process(policy.identity, upcoming=known, checkpoint='before_water_NPY_headers')
    checked = []; total_file = total_data = 0
    for key in cli._ARRAYS:
        row = spec[key]
        if row is None: continue
        path = root / cli.path_value(row['path'], policy.max_path_chars, absolute=False)
        raw, pin = cli.read_pin(path, policy.max_water_input_file_bytes)
        if (pin['size'], pin['sha256']) != (row['size'], row['sha256']): raise ValueError('water NPY input pin changed')
        total_file += pin['size']
        if total_file > policy.max_water_input_bytes: raise ValueError('water NPY file sum exceeds policy')
        payload = _water_npy_header(raw, row, key, g)
        total_data += payload; checked.append((key, path, pin))
    if 3 * total_data > copy_limits.max_known_host_bytes:
        raise ValueError('NPY current/map/owned/error copy reservation exceeds water copy quota')
    result = dict(spec)
    for key, path, pin in checked:
        mapped = np.load(path, allow_pickle=False, mmap_mode='r', max_header_size=10000)
        try:
            result[key] = np.array(mapped, copy=True, order='C', subok=False)
        finally:
            # Explicitly close only this owned read-only mapping. Never GC-credit
            # the working ndarray or any Native deep store.
            mapped._mmap.close()
        if cli.read_pin(path, pin['size'])[1] != pin: raise ValueError('NPY changed during load')
    return result, checked, {'input_file_bytes': total_file, 'owned_array_bytes': total_data,
                              'known_current_candidate_error_bytes': known, 'raw_and_header_bytes': 2 * declared_files + 10000 * len(declared)}



def _prepare_windowed_water(spec, control, bed, limits):
    from masy.case_water import CaseWaterSpec, CaseWetClass
    from masy.case_windowed import prepare_case_windowed_water
    from . import case_cli as cli
    from masy.dense_runner import DenseRunnerControl
    from maple_syrup.storm import StormControl
    from maple_syrup.sediment_event import SedimentEventControl
    from maple_syrup.complete_event import CompletionPolicy
    values = dict(spec)
    if type(values['wet_classes']) is not list or not values['wet_classes']: raise ValueError('explicit wet-class coefficients required')
    rows = []
    for row in values['wet_classes']:
        cli.closed(row, tuple(f.name for f in dataclasses.fields(CaseWetClass)), 'wet class')
        rows.append(CaseWetClass(**row))
    values['wet_classes'] = tuple(rows)
    sc = dict(control['sediment_control']); sc['storm'] = StormControl(**sc['storm']).validated()
    sediment = SedimentEventControl(**sc).validated()
    completion = CompletionPolicy(**control['completion_policy']).validated()
    return prepare_case_windowed_water(bed, CaseWaterSpec(**values),
        DenseRunnerControl(sediment, completion, cli.number(control['report_every_s'], 'report cadence', positive=True)), limits=limits)



def _capture_images(facts):
    closed=sys.modules['maple.surface.topographic_commit.closed_windowed_commit']
    bridge=sys.modules['maple.aeolian.scheduler.windowed_baseline']
    commit=importlib.import_module('maple.surface.topographic_commit.windowed_device')
    device=sys.modules['maple.aeolian.scheduler.windowed_device']
    records=sys.modules['maple.aeolian.scheduler.windowed_record']
    owner=sys.modules['maple.core.backend.depth_class_reductions']
    api_owner,api_anchor=commit._API_OWNER,commit._ORIGINAL_API
    if commit._api()!=api_anchor:raise ValueError('genuine Native device commit anchor')
    modules={m.__name__:m for m in (closed,bridge,commit,device,records,owner)}
    images=tuple((m,n,getattr(m,n),getattr(m,n).operators,getattr(m,n).anchor) for m,n in
        ((closed,'_ClosedNativeImage'),(closed,'_BoundedCPUQualificationImage'),(closed,'_ClosedWaterImage'),
         (bridge,'_Image'),(device,'_Image'),(device,'_RecordImage')))
    counts=(len(closed._ClosedNativeImage.operators),len(closed._BoundedCPUQualificationImage.operators),
        len(closed._ClosedWaterImage.operators),len(api_anchor),len(device._Image.operators),len(device._RecordImage.operators))
    expected=tuple(facts[k] for k in ('closed_native_operators','bounded_cpu_operators','closed_water_operators',
        'device_commit_operators','device_scheduler_operators','device_record_operators'))
    if counts!=expected:raise ValueError('Root actual complete Native image counts required')
    return dict(modules=modules,images=images,p=(closed._P_OPERATORS,closed._P_OPERATORS_ANCHOR),
                api=(api_owner,api_anchor),counts=counts)


def _final_images(first,orders,facts):
    if any(sys.modules.get(k)is not v for k,v in first['modules'].items()):raise ValueError('FIRST Native module identity changed')
    closed=first['modules']['maple.surface.topographic_commit.closed_windowed_commit']
    commit=first['modules']['maple.surface.topographic_commit.windowed_device']
    bridge=first['modules']['maple.aeolian.scheduler.windowed_baseline']
    device=first['modules']['maple.aeolian.scheduler.windowed_device']
    owner=first['modules']['maple.core.backend.depth_class_reductions']
    if (closed._P_OPERATORS is not first['p'][0] or closed._P_OPERATORS_ANCHOR is not first['p'][1]
            or commit._API_OWNER is not first['api'][0] or commit._ORIGINAL_API is not first['api'][1] or commit._api()!=first['api'][1]):
        raise ValueError('FIRST operators/API rebound')
    for m,name,image,operators,anchor in first['images']:
        if getattr(m,name,None)is not image or image.operators is not operators or image.anchor is not anchor or operators is not anchor:
            raise ValueError('genuine Native image identity changed')
    closed._p_check_operators(closed._P_OPERATORS);closed._w_check_image();bridge._check_image();device._check_image();device._check_record_image()
    if len(closed._P_OPERATORS)!=facts['cwc_operators'] or len(bridge._Image.operators)!=facts['w1_operators']:raise ValueError('actual image count changed')
    for order,extent in zip(orders,((6,1),(6,6),(12,3),(20,1)),strict=True):owner._require_order(order,*extent)
    return dict(four_genuine_orders=[[6,1],[6,6],[12,3],[20,1]],FIRST_identities_unchanged=True,actual_image_counts=list(first['counts']))


def _post_runtime(outer,authority):
    from . import gpu_runtime_worker as gw
    from . import case_gpu_cli as cli
    gw.native._origins(outer['runtime']['packages'])
    digests=gw.native._package_digests(outer['runtime']['packages'],outer['runtime']['limits'])
    environment,index=gw.environment_metadata()
    if environment!=outer['expected_audit']['environment']:raise ValueError('environment/RECORD metadata changed')
    if digests!=outer['expected_audit']['native']['package_digests']:raise ValueError('actual Native/package Source digest changed')
    actual=gw.compiled_inventory(outer['runtime']['limits'],index)
    baseline={r['path']:r for r in outer['expected_audit']['compiled']['files']}
    now={r['path']:r for r in actual['files']}
    if any(now.get(k)!=v for k,v in baseline.items()):raise ValueError('retained audit binary changed or disappeared')
    allowed={r['path']:r for r in authority['compiled_additions']}
    additions=[]
    for path,row in now.items():
        if path in baseline:continue
        pin={k:row[k] for k in ('path','size','sha256')}
        if allowed.get(path)!=pin:raise ValueError('new compiled binary lacks exact Root-issued byte identity: '+path)
        additions.append(row)  # Genuine compiled_inventory already streamed/hashed/RECORD-checked it.
    return dict(compiled=actual,actual_additions=additions,environment_and_RECORD_unchanged=True,package_Source_unchanged=True)


def _issue_case_budget(outer,issued,binding,path):
    from . import case_gpu_cli as cli
    from maple_syrup import windowed_shared_bed_device as shared
    d=issued.value['case_budget_delegation']
    if Path(__file__).resolve()!=Path(d['issuer']['path']):raise ValueError('genuine delegated immutable worker required')
    cli.pinned(d['issuer'],immutable=True)
    def ref(r):return shared.SourceReference(r['path'],r['size'],r['sha256'])
    if [r['module'] for r in d['source_bindings']]!=list(shared._SOURCE_MODULES):raise ValueError('exact ordered22 owning Source modules')
    kwargs=dict(native_source_identity=d['native_source_identity'],current_runtime_ref=ref(d['current_runtime_ref']),
        producer_module='masy.case_device_c64',producer_source=ref(d['producer_source']),
        source_bindings=tuple((r['module'],ref(r)) for r in d['source_bindings']),gpu_uuid=issued.value['availability']['gpu_uuid'],
        source_manifest_sha256=d['source_manifest_sha256'],case_request_binding_sha256=hashlib.sha256(binding).hexdigest())
    # Real immutable preauthorization reference, metadata only. No Native/DTO
    # consumer ever receives this temporary object; canonical bytes omit grant.
    temporary=shared.DeviceWaterC64CaseBudget(grant=ref(issued.ref),**kwargs)
    encoded=shared.c64_case_budget_bytes(temporary)
    del temporary
    _remaining(outer);cli.reserve(outer['operation_root'],len(encoded))
    pin=cli.write_new(path,encoded,raw=True,immutable=True)
    final=shared.DeviceWaterC64CaseBudget(grant=ref(pin),**kwargs)
    if shared.c64_case_budget_bytes(final)!=encoded:raise ValueError('owning budget encoder self-reference')
    shared._c64_case_budget_runtime(final,verify_files=True)
    return final,pin


def _native_snapshot_source(outer):
    """Join actual Native module to its whole verified Source object row."""
    from . import gpu_runtime as gpu,runtime as rt,case_gpu_cli as cli
    a=cli.read_authority(outer['authority'],at_issue=False).value
    profile,limits,_=gpu._load_profile(a['store'],a['profile_id'])
    obj=profile['objects']['maple'];manifest_path=Path(obj['path'])/'manifest.json'
    raw=rt._read_bounded(manifest_path,4<<20,'Native_snapshot_Source')
    if hashlib.sha256(raw).hexdigest()!=obj['manifest_sha256']:raise ValueError('Native Source object manifest changed')
    manifest=json.loads(raw);name='src/maple/surface/voxels/windowed_snapshot.py'
    rows=[r for r in manifest['identity']['files'] if r[0]==name]
    if len(rows)!=1:raise ValueError('selected Native snapshot Source absent')
    row=rows[0];module=importlib.import_module('maple.surface.voxels.windowed_snapshot')
    expected=Path(outer['runtime']['packages']['maple']['dir'])/'surface/voxels/windowed_snapshot.py'
    if Path(module.__file__).resolve()!=expected:raise ValueError('actual Native snapshot module origin')
    pin=dict(path=str(expected),size=row[1],sha256=row[2]);cli.pinned(pin)
    return module,pin


def _write_native_diagnostic(case,outer,policy):
    """Root-authorized trusted idle-owner read, never a public raw-runner API."""
    from . import case_gpu_cli as cli
    from .case_device_c64 import PreparedDeviceC64Case
    from .windowed_device_c64_bed import DeviceWindowedC64BedOwner
    from .case_worker import _NPZ_MEMBER_OVERHEAD,_NPZ_NAME_BYTES
    from maple.surface.voxels.windowed_prefix import require_prefix_packets
    import numpy as np
    if type(case)is not PreparedDeviceC64Case or type(case.bed)is not DeviceWindowedC64BedOwner:raise ValueError('exact genuine diagnostic owner')
    case.verify();bed=case.bed;bed.verify()
    runner=case._runner  # Only the explicitly scoped trusted diagnostic read.
    if (bed._active is not None or runner._device_resources.active or runner._active_evidence is not None
            or runner._pending_water_counters is not None):raise ValueError('Native diagnostic requires a verified idle accepted owner')
    state=bed._accepted.state;column=state.voxel_column
    stamp=(id(bed._accepted),bed.generation,bed.time_s.hex(),column.generation,column.manifest_digest)
    native,pin=_native_snapshot_source(outer)
    packets=require_prefix_packets(column,operation='public_case_diagnostic')
    arrays=tuple(getattr(column,k) for k in native.COLUMN_ARRAYS)+(packets.packet,state.active_layer.mass_kg,
        state.sediment_availability.available_mass_kg,state.sediment_availability.bound_mass_kg,
        state.water.depth_m,state.water.mobile_mass_by_cell_class_kg)
    names=(*native.COLUMN_ARRAYS,*native.OTHER_ARRAYS)
    if (len(arrays)!=11 or any(len((n+'.npy').encode('ascii'))>_NPZ_NAME_BYTES for n in (*names,'metadata'))
            or column.n_voxels!=20 or column.window_voxels!=3 or bed.context.geometry.nx!=8 or bed.context.geometry.ny!=8):
        raise ValueError('closed C64 Native diagnostic layout')
    resident=sum(int(a.nbytes) for a in arrays)
    prefix=int(packets.packet.nbytes);full=64*20*8
    provenance=dict(schema='masy.case_gpu_native_diagnostic.v1',authority_sha256=outer['authority']['sha256'],
        case_request_binding_sha256=case.report()['case_request_binding_sha256'],accepted_cursor=case.accepted.next_index)
    metadata=dict(schema=native.SCHEMA,store_dir=str(column.deep_store.store_dir.resolve()),manifest_digest=column.manifest_digest,
        generation=column.generation,header_digest=column.deep_store.header.digest(),n_voxels=20,time_s=float(bed.time_s),
        step=case.accepted.next_index,class_ids=list(column.deep_store.header.class_ids),provenance=provenance)
    text=json.dumps(metadata,sort_keys=True,allow_nan=False)
    unicode_bytes=np.dtype('U'+str(len(text))).itemsize
    # Same positively charged primitive/header/ZIP64 law as the original
    # Native compiler planner. Every fixed member name is checked above.
    archive=resident+unicode_bytes+12*_NPZ_MEMBER_OVERHEAD+22+56+20
    if archive>64<<20 or len(text.encode())>policy.max_report_bytes:raise ValueError('prospective Native diagnostic file/header quota')
    source=Path(native.__file__).read_text();tree=__import__('ast').parse(source)
    selected=[n for n in tree.body if isinstance(n,__import__('ast').FunctionDef) and n.name in ('validate_windowed_partition','_validate_other','save_windowed_state_snapshot','_publish_npz_no_clobber')]
    # Every selected expression receives a full closed family envelope, including
    # branch alternatives. Retain original callee/source/native/allocator charges.
    expressions=sum(sum(isinstance(n,__import__('ast').expr) for n in __import__('ast').walk(fn)) for fn in selected)
    validation=expressions*(full+3*prefix+512+256)
    mapped=bed.context.bridge_limits.max_mapped_bytes
    known=3*(resident+unicode_bytes+archive+validation+mapped+pin['size'])
    _check_gpu_case_process(policy.identity,upcoming=known,checkpoint='before_Native_diagnostic')
    from maple.surface.active_layer.bed_exchange_device import _check_device_admission
    _check_device_admission(3*(resident+validation),device_id=bed._device,safety_fraction=0.5)
    _remaining(outer);cli.reserve(outer['operation_root'],2*archive+policy.max_report_bytes)
    path=Path(outer['runtime']['run']['run_dir'])/'native-diagnostic.npz'
    native.save_windowed_state_snapshot(column,state.active_layer,bed.time_s,path,step=case.accepted.next_index,
        geometry=bed.context.geometry,mass_resolution_kg=bed.context.mass_resolution_kg,
        sediment_availability=state.sediment_availability,water=state.water,provenance=provenance)
    actual=path.stat().st_size
    if actual>archive:raise ValueError('Native diagnostic exceeded actual Source format upper')
    import zipfile
    with zipfile.ZipFile(path) as z:
        members=z.infolist()
        if len(members)!=12 or {i.filename for i in members}!={n+'.npy' for n in (*names,'metadata')} or sum(i.file_size for i in members)>archive or any(i.compress_type!=zipfile.ZIP_STORED or i.flag_bits&1 for i in members):raise ValueError('actual Native diagnostic closed format')
    result=cli.io.read_pin(path,64<<20)[1]
    cli.pinned(pin);bed.verify();case.verify()
    if stamp!=(id(bed._accepted),bed.generation,bed.time_s.hex(),column.generation,column.manifest_digest):raise ValueError('diagnostic changed accepted ownership')
    return dict(file=result,Native_Source=pin,format=native.SCHEMA,deep_store_dir=str(column.deep_store.store_dir),
        manifest_digest=column.manifest_digest,generation=column.generation,accepted_cursor=case.accepted.next_index,
        label='Native physical diagnostic; not complete MASY/C64 restart; no private soil/pool/vegetation companion',
        prospective=dict(resident_bytes=resident,prefix_bytes=prefix,Native_full_chunk_result_bytes=full,
            validation_expression_occurrences=expressions,known_validation_and_IO_bytes=known,archive_upper=archive,
            temporary_and_final_namespace_occurrences=2,constructor_allocation=None,external_runtime=None))


def _case_operation(outer,issued,process):
    from . import case_gpu_cli as cli,gpu_runtime_worker as gw
    authority=issued.value
    _remaining(outer)
    _jax_environment_sources(imported=False)
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()!=1 or cp.cuda.Device().id!=0:raise ValueError('only authorized physical UUID visible as0')
    gw.check_observed_device(outer['runtime']['device'],cp.cuda.runtime.getDeviceProperties(0),cp.cuda.runtime.deviceGetPCIBusId(0))
    pool=cp.cuda.MemoryPool();pool.set_limit(size=2<<30);cp.cuda.set_allocator(pool.malloc)
    qualified,preload=_bootstrap_native(Path(outer['runtime']['packages']['maple']['dir']),authority['native_image_facts'])
    pre_native_process=process
    process,environment_proof=_reconcile_native_environment(pre_native_process)
    first=_capture_images(authority['native_image_facts'])
    from maple_syrup import windowed_shared_bed_device as shared
    for name in shared._SOURCE_MODULES:importlib.import_module(name)
    from . import case_device_c64 as producer
    from . import case_worker as original
    from .case_workflow import resolve_case_inputs
    from .case_windowed import adopt_windowed_case
    _remaining(outer);raw=cli.pinned(authority['request']);v=cli.request_from(raw)
    root=Path(outer['operation_root']);run=Path(outer['runtime']['run']['run_dir'])
    limits=_decode_case_limits(v,root)
    policy=SimpleNamespace(**v['io_policy'],max_run_bytes=512<<20,identity=process)
    inventory=cli.io.inspect_case(v['case']['directory'],limits['input_limits'])
    if inventory['authored_source_sha256']!=v['case']['authored_source_sha256']:raise ValueError('authored case source identity')
    inspection=resolve_case_inputs(v['case']['directory'],mode='validate',input_limits=limits['input_limits'])
    cfg=inspection.resolved.config;voxels=inspection.resolved.topography.nz+inspection.resolved.topography.extra_top_layers
    if (cfg.geometry.nx,cfg.geometry.ny,voxels,len(cfg.grain_classes.classes))!=(8,8,20,1):raise ValueError('C64/K1/V20 eligibility before initial compilation')
    writers=original._require_compile_writers()
    disk=original._native_disk_plan(inspection,inventory,policy,limits['compile_bed_limits'])
    disk['Native_writer_Source']=writers
    reservation=cli.reserve(root,disk['prospective_peak_extra_bytes']+limits['windowed_limits'].max_store_bytes+2*(64<<20)+3*policy.max_report_bytes)
    _check_gpu_case_process(process,upcoming=2*disk['authored_copy_bytes'],checkpoint='before_case_copy')
    owned=run/'input_case';original._owned_case_copy(Path(v['case']['directory']),owned,inventory,policy)
    if cli.io.inspect_case(v['case']['directory'],limits['input_limits'])!=inventory:raise ValueError('source case changed during copy')
    copied=cli.io.inspect_case(owned,limits['input_limits'])
    if copied['files']!=inventory['files']:raise ValueError('owned case source differs')
    _remaining(outer)
    inputs=resolve_case_inputs(owned,mode='compile',input_limits=limits['input_limits'])
    host=adopt_windowed_case(inputs,store_dir=run/'deep_store',limits=limits['windowed_limits'],**v['windowed_storage'])
    spec,pins,npy_plan=_load_water_arrays_gpu(v['water_profile']['spec'],Path(authority['request']['path']).parent,cfg.geometry,policy,limits['water_copy_limits'])
    water=_prepare_windowed_water(spec,v['water_profile']['control'],host,limits['water_copy_limits'])
    cuda=dataclasses.replace(water.control,sediment_control=dataclasses.replace(water.control.sediment_control,
        storm=dataclasses.replace(water.control.sediment_control.storm,implementation='cuda')))
    kwargs=dict(schedule_input=v['schedule'],cuda_control=cuda,record_limits=limits['record_limits'],bridge_limits=limits['bridge_limits'],
        bed_limits=limits['bed_limits'],resource_limits=limits['resource_limits'],runner_limits=limits['runner_limits'],
        placement_limits=limits['placement_limits'],calibration=v['calibration'],label=v['label'])
    binding=producer.device_c64_request_binding(host,water,**kwargs)
    encoded=cli.canonical(binding);cli.reserve(root,len(encoded)+3*policy.max_report_bytes)
    projection=cli.write_new(run/'projection.json',encoded,raw=True,immutable=True)
    budget,budget_pin=_issue_case_budget(outer,issued,encoded,Path(authority['case_budget_delegation']['grant_path']))
    guarded=producer.prepare_device_c64_case(host,water,qualification_budget=budget,request_binding=binding,**kwargs)
    if guarded.build_runner()is not guarded:raise ValueError('guarded owning public runner interface required')
    cli.write_new(run/'accepted-0000.json',dict(cursor=0,accepted=dataclasses.asdict(guarded.accepted)))
    while guarded.stop_receipt is None and guarded.accepted.next_index<4:
        _remaining(outer);cli.reserve(root,3*policy.max_report_bytes)
        receipt=guarded.step()
        if receipt['status']!=receipt['code'] or receipt['status']!='accepted':raise ValueError('actual event did not complete')
        cli.write_new(run/('accepted-%04d.json'%guarded.accepted.next_index),dict(cursor=guarded.accepted.next_index,receipt=receipt))
        cli.census(root)
    report=guarded.report()
    if guarded.stop_receipt is not None or report['runner']['status']!='complete':raise ValueError('actual four-event completion required')
    diagnostic=_write_native_diagnostic(guarded,outer,policy)
    inputs.verify();host.verify();water.verify(host);guarded.verify()
    if cli.io.inspect_case(v['case']['directory'],limits['input_limits'])!=inventory:raise ValueError('original inputs changed during operation')
    for _name,path,pin in pins:
        if cli.io.read_pin(path,pin['size'])[1]!=pin:raise ValueError('water input changed')
    shared._c64_case_budget_runtime(budget,verify_files=True);issued.revalidate();cli.pinned(authority['request'])
    final_images=_final_images(first,qualified,authority['native_image_facts'])
    if getattr(cp.cuda.get_allocator(),'__self__',None)is not pool or pool.get_limit()!=2<<30:raise ValueError('original active pool replaced')
    cp.cuda.Stream.null.synchronize()
    post=_post_runtime(outer,authority)
    final_process,final_environment_proof=_reconcile_native_environment(pre_native_process)
    if final_process!=process or final_environment_proof!=environment_proof:raise ValueError('checked Native environment changed during operation')
    _remaining(outer);cli.census(root)
    body=dict(schema='masy.case_gpu_report.v1',status='complete',case_request=authority['request'],authority=issued.ref,
        projection=projection,case_budget=budget_pin,report=report,diagnostic=diagnostic,Native_preload=preload,final_images=final_images,
        original_inputs=inventory,owned_case=str(owned),input_identity=inputs.identity_sha256,known_disk_plan=disk,disk_admission=reservation,
        NPY_plan=npy_plan,runtime_post=post,Native_environment=environment_proof,execution_budgets=cli.BUDGETS,known_subset_only=True,
        qualification='synthetic imported C64 execution only; acceptance Root-owned; no allocator/general-capacity/restart claim')
    raw=cli.canonical(body)
    if len(raw)>policy.max_report_bytes:raise ValueError('declared report output quota')
    cli.reserve(root,len(raw));report_pin=cli.write_new(run/'case-report.json',raw,raw=True,immutable=True)
    return dict(status='complete',binding_sha256=hashlib.sha256(encoded).hexdigest(),report=report_pin,diagnostic=diagnostic)


def main(outer,base_sys_path):
    from . import case_gpu_cli as cli,gpu_runtime_worker as gw
    mode=outer.get('operation');process=None;safe_stage=None
    try:
        cli.closed(outer,('schema','operation','runtime','authority','case_request','launcher','profile_id','operation_root','deadline_monotonic','expected_audit','consumption'),'new case worker request')
        if outer['schema']!=cli.WORKER_SCHEMA or mode not in ('metadata','metadata-after','case'):raise ValueError('closed new operation mode')
        _remaining(outer);issued=cli.read_authority(outer['authority'],at_issue=False);a=issued.value
        if outer['case_request']!=a['request'] or outer['profile_id']!=a['profile_id'] or Path(outer['operation_root'])!=Path(a['store'])/'runs'/a['run_id']:
            raise ValueError('actual operation/preauthorization join')
        if outer['deadline_monotonic']!=outer['launcher']['started_monotonic']+600:raise ValueError('original whole deadline join')
        stage=Path(outer['operation_root'])/mode
        if Path(outer['runtime']['run']['run_dir'])!=stage or Path.cwd()!=stage:raise ValueError('exact owned stage/cwd')
        if Path(outer['consumption']['path'])!=Path(outer['operation_root'])/'consumption.json':raise ValueError('owned consumption path')
        consumption=cli.parse(cli.pinned(outer['consumption'],immutable=True))
        cli.closed(consumption,('authority','operation','pid','started_monotonic'),'one consumption')
        if consumption['authority']!=issued.ref or consumption['operation']!=cli.OPERATION or consumption['started_monotonic']!=outer['launcher']['started_monotonic']:raise ValueError('exclusive operation consumption identity')
        process=gw.check_process(outer['runtime'],base_sys_path)
        from . import gpu_runtime as gpu,runtime as rt
        profile,limits,_=gpu._load_profile(a['store'],a['profile_id'])
        python=rt.interpreter_identity(gw.PYTHON,limits)
        if rt._python_stable(python)!=profile['runtime']['python']:raise ValueError('actual selected interpreter')
        actual_body=gpu._profile_body(profile,SimpleNamespace(dir=stage,pycache=str(stage/'cache/pycache')),python)
        if actual_body!=outer['runtime'] or profile['audit']!=outer['expected_audit']:raise ValueError('genuine exact runtime/profile process body')
        if Path(__file__).resolve()!=Path(a['case_budget_delegation']['issuer']['path']):raise ValueError('actual selected case worker')
        cli.pinned(a['case_budget_delegation']['issuer'],immutable=True)
        safe_stage=stage
        cli.write_new(stage/'startup.json',dict(process=process,operation=mode,authority=issued.ref,deadline_monotonic=outer['deadline_monotonic']))
        if mode!='case':
            class Blocker(importlib.abc.MetaPathFinder):
                def find_spec(self,fullname,path=None,target=None):
                    if fullname.partition('.')[0] in ('maple','maple_syrup','numpy','numba','jax','cupy','cupyx','cuda','scipy'):
                        raise ValueError('metadata operation cannot import Native/provider')
            block=Blocker();sys.meta_path.insert(0,block)
            try:result=_sources(outer,a)
            finally:sys.meta_path.remove(block)
        else:result=_case_operation(outer,issued,process)
        output=dict(schema=cli.WORKER_RESULT,operation=mode,status='ok',result=result,process=process)
        code=0
    except Exception as exc:
        traceback.print_exc()
        output=dict(schema=cli.WORKER_RESULT,operation=mode,status='refused',code=getattr(exc,'code',type(exc).__name__),detail=str(exc)[:2000],process=process)
        if safe_stage is not None:
            # Bounded truthful receipts remain local; never claim a Native error
            # left an active candidate restartable or silently replay genesis.
            evidence=dict(output,last_accepted_receipts=[p.name for p in sorted(safe_stage.glob('accepted-*.json'))],
                          cleanup_after_deadline_s=max(0,time.monotonic()-outer['deadline_monotonic']))
            try:cli.write_new(safe_stage/'failure.json',evidence)
            except OSError:pass
        code=2
    sys.stdout.buffer.write(cli.canonical(output)+b'\n');sys.stdout.buffer.flush();return code
