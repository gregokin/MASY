"""Sealed cold case CPU worker; entered only through its literal bootstrap.

The original runtime operation remains unchanged and runs before case work.
Artifacts and failures stay under the run-owned directory; no upstream/user
case is compiled in place. No whole-heap or filesystem aggregate theorem is
implied by the known-copy reservations and phase censuses.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

from masy import case_cli as cli
from masy.case_runner import CaseRunner, check_case_process, native_environment_record

BOOTSTRAP = """\
import sys

def _case_bootstrap():
    import json, math, os, resource
    if any(n.partition('.')[0] in ('masy','maple','maple_syrup','numpy','numba','jax','cupy','scipy') for n in sys.modules):
        raise ValueError('cold case worker has preloaded model/provider modules')
    if len(sys.argv) != 2: raise ValueError('one closed outer request required')
    with open(sys.argv[1], 'rb') as stream: raw = stream.read(1048577)
    if len(raw) > 1048576: raise ValueError('outer request exceeds original1MiB bootstrap cap')
    def pairs(rows):
        out = {}
        for k,v in rows:
            if k in out: raise ValueError('duplicate outer request key')
            out[k] = v
        return out
    request = json.loads(raw, object_pairs_hook=pairs)
    if type(request) is not dict or request.get('schema') != 'masy.case_cpu_worker_request.v1': raise ValueError('outer schema')
    policy = request['process_policy']
    expected = ('address_space_bytes','cpu_soft_s','cpu_hard_s','wall_s','max_output_bytes','max_run_bytes','max_request_bytes','max_input_entries','max_path_chars','max_water_input_file_bytes','max_water_input_bytes','max_report_bytes')
    if type(policy) is not dict or set(policy) != set(expected): raise ValueError('closed case process policy required')
    for k in expected:
        v=policy[k]
        if k=='wall_s':
            if type(v) not in (int,float) or not math.isfinite(v) or v<=0: raise ValueError('wall quota')
        elif type(v) is not int or v<=0: raise ValueError('plain positive process quota')
    if policy['address_space_bytes'] != 4294967296 or policy['cpu_hard_s'] < policy['cpu_soft_s']: raise ValueError('case4GiB/CPU policy')
    if policy['max_request_bytes']>1048576 or len(raw)>policy['max_request_bytes']: raise ValueError('outer request quota')
    wanted = {resource.RLIMIT_AS:(4294967296,4294967296),resource.RLIMIT_CPU:(policy['cpu_soft_s'],policy['cpu_hard_s']),resource.RLIMIT_FSIZE:(policy['max_run_bytes'],policy['max_run_bytes']),resource.RLIMIT_CORE:(0,0)}
    for k,pair in wanted.items():
        hard=resource.getrlimit(k)[1]
        if hard!=resource.RLIM_INFINITY and pair[1]>hard: raise ValueError('inherited hard ceiling cannot meet explicit case policy')
    for k,pair in wanted.items(): resource.setrlimit(k,pair)
    runtime=request['runtime']; prefix=runtime['sys_path_prefix']
    if type(prefix) is not list or not prefix or not all(type(p) is str and p.startswith('/') for p in prefix): raise ValueError('sealed prefix')
    base=list(sys.path); sys.path[0:0]=prefix
    import masy.case_worker as worker
    return worker.main(request,base)

sys.exit(_case_bootstrap())
"""


_COMPILE_SOURCE_PINS = {'case_tools/compilers/case_compiler.py': (121363, '2db21adb101f2817e12933c0da1fbb613979f0cc929c9f3c385dc54e91581252'), 'case_tools/report.py': (8160, '095fd008b42ceb26e20537eca42fc1e01bc5149d49759adeb9cf446e028e6c46'), 'case_tools/provenance.py': (50889, '5478bc06cfb0aa7f42646831d3a73b69cf9422c3a3b462d3b1ab3ad264e027bf'), 'io/outputs/snapshot.py': (76370, '1320c4d442aea5536419ecf2a1a0ed9e75cb6458a3dc8f39a0f5bbb412c3a5b3'), 'experiment/orchestrator.py': (210852, 'f4cc9d44a86a68585530ba5fd02f43664c32e5d476dfa596b0e2d57cc11f7ea7'), 'core/types/sediment_ledger.py': (56532, '4e4bd1d2ca8f126ac3224d3233366af8c3d5dee5afeb4b85fd883e3cc8725003'), 'case_tools/generators/vegetation.py': (17470, 'aaf4bc9c06bcfded02528040e0e562f0cf08d902af0de8818d6bba10d1d5553d'), 'case_tools/transforms/log.py': (7097, '45bdf15736cce004fb3bfd8eaa98401c3c26456188626b8bc9df8a595883bd19'), 'case_tools/importers/field.py': (7946, '7469e38563f221c841f0b14dd830f1455b45a8acc0b93f99814f708e121cedb8'), 'case_tools/importers/readers.py': (34883, '4cf74976ccf44fddd08553a1942f672f6251ea5424b9c9132f7e7c0221df11a7'), 'case_tools/transforms/empirical.py': (64162, 'a59608f29bda182d332d8f0eea313b52c47de1d8ba738e36fc13ff2300d452a8'), 'core/parameters/case_import.py': (72592, '29a0363d664c2bb84657dda974df9191f6f24065a0194c85f017958f94c62630')}


def _require_compile_writers():
    """Exact known Native Source sizing contract, not future API compatibility."""
    import maple
    root = Path(maple.__file__).resolve().parent
    receipt = []
    for relative, expected in _COMPILE_SOURCE_PINS.items():
        data, pin = cli.read_pin(root / relative, expected[0])
        if (len(data), pin['sha256']) != expected:
            raise ValueError('Native compiler/output Source changed; refresh the finite sizing qualification: ' + relative)
        receipt.append(pin)
    return receipt


def _owned_case_copy(source, destination, inventory, policy):
    # The caller has reserved the full authored copy plus all compiler overlap.
    # Validate the complete path roster before the first output directory.
    for row in inventory['files']:
        relative = cli.path_value(row['path'], policy.max_path_chars, absolute=False)
        for path in (source / relative, destination / relative):
            cli.path_value(str(path), policy.max_path_chars, absolute=True)
    destination.mkdir(mode=0o700)
    for row in inventory['files']:
        relative = cli.path_value(row['path'], policy.max_path_chars, absolute=False)
        original = source / relative; target = destination / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        data, pin = cli.read_pin(original, row['size'])
        if (pin['size'], pin['sha256']) != (row['size'], row['sha256']): raise ValueError('authored Source changed before copy')
        with target.open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        if cli.read_pin(target, row['size'])[1]['sha256'] != row['sha256']: raise ValueError('owned input copy differs')


def _dem_disk_domain(inspection, inventory, policy):
    """Admit the genuine simple NPY DEM branch without changing its recipe."""
    import numpy as np
    from maple.core.parameters.case_import import ImportedFieldSpec
    r = inspection.resolved; imp = r.case_import; g = r.config.geometry
    if any(getattr(imp, key) is not None for key in ('sediment', 'vegetation', 'forcing')):
        raise ValueError('first case reader qualifies NPY DEM plus authored stratigraphy, not other imported families')
    fields = []
    def visit(value):
        if type(value) is ImportedFieldSpec:
            fields.append(value)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value): visit(getattr(value, field.name))
        elif type(value) in (tuple, list):
            for item in value: visit(item)
        elif type(value) is dict:
            for item in value.values(): visit(item)
    visit(r)
    if imp.elevation is None:
        if fields or r.import_sources: raise ValueError('consumer-side imported fields need their own log/output sizing law')
        return {'members': 0, 'payload': 0, 'archives': 0, 'log_scalar_slots': 0, 'source_bytes': 0}
    dem = imp.elevation
    if (len(fields) != 1 or fields[0] is not dem or dem.source.format != 'npy' or dem.transforms
            or len(r.import_sources) != 1):
        raise ValueError('first DEM lane requires one original C-shaped NPY field, with Native automatic units/orientation only')
    source = cli.path_value(dem.source.path, policy.max_path_chars, absolute=False)
    if not source.parts or source.parts[0] != 'source': raise ValueError('DEM must belong to the authored source snapshot')
    row = next((row for row in inventory['files'] if row['path'] == str(source)), None)
    if row is None: raise ValueError('DEM lacks an authenticated authored Source pin')
    raw, pin = cli.read_pin(inspection.case_dir / source, inspection.input_limits.max_source_file_bytes)
    if (pin['size'], pin['sha256']) != (row['size'], row['sha256']): raise ValueError('DEM changed before header admission')
    import io
    stream = io.BytesIO(raw); version = np.lib.format.read_magic(stream)
    if version == (1, 0): shape, order, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10000)
    elif version == (2, 0): shape, order, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=10000)
    else: raise ValueError('unqualified Native DEM NPY version')
    if (shape != (g.ny, g.nx) or order or dtype != np.dtype('float64') or dtype.hasobject
            or stream.tell() + 8 * g.ny * g.nx != len(raw)):
        raise ValueError('first Native DEM must be a full C-order float64 grid on the authored geometry')
    # All actual value/unit/nodata/orientation checks and arithmetic remain in
    # Native read_source/build_imported_field. This reader never loads a DEM.
    # Compiler: raw1+processed(1+4 masks) NPZ members, and4 standalone masks.
    # TransformLog has one automatic convert_units record (24 fields),
    # its12-field Source record and14-field totals;64 scalar slots dominate
    # compound pairs. All dynamic strings occur in the admitted legacy recipe
    # or the named fixed Native Source text; no user transform/array parameter.
    return {'members': 10, 'payload': 24 * g.ny * g.nx, 'archives': 2,
            'log_scalar_slots': 64, 'source_bytes': len(raw)}



_NPZ_NAME_BYTES = max(len(name.encode('ascii')) + 4 for name in (
    'threshold_modifier_gravimetric_moisture_fraction',
    'threshold_modifier_chemical_cementation_multiplier',
    'threshold_modifier_biological_binding_multiplier',
    'committed_active_layer_mass_by_class_kg',
    'vegetation_raster_frontal_area_density',
    'water_mobile_mass_by_cell_class_kg',
    'numerical_residual_positive_compensation_kg'))
_NPZ_MEMBER_OVERHEAD = 10000 + 30 + 46 + 20 + 28 + 2 * _NPZ_NAME_BYTES
_ANALYSIS_RUN_NAME = 'masy_case_analysis'
_ANALYSIS_REASON = 'final_accepted_boundary_analysis_only'


def _analysis_provenance(case_sha, schedule_sha, bed_sha, lock_id):
    for value in (case_sha, schedule_sha, bed_sha, lock_id):
        if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('analysis provenance requires genuine64-hex identities')
    return {'schema': 'masy.case_analysis.v1', 'case_sha256': case_sha,
            'schedule_sha256': schedule_sha, 'accepted_bed_identity_sha256': bed_sha, 'lock_id': lock_id,
            'purpose': 'analysis_only_not_Native_or_combined_restart'}


def _analysis_disk_upper(cells, voxels, classes, vegetation_source):
    from maple.io.outputs import snapshot
    version = snapshot._EXACT_HISTORY_SCHEMA_VERSION
    if type(version) is not str or len(version) > 10000:
        raise ValueError('unqualified Native history-schema metadata width')
    # Exactly16 base,8 prescribed raster vegetation,6 threshold,3
    # availability and3 water members. Optional absent families only reduce
    # this bound; object vegetation/nonempty cohorts are not this profile.
    members = 36
    payload = 8*cells*voxels*classes + 8*cells*classes + 80*cells + 16*cells*classes + 8*cells*(1+classes)
    provenance = json.dumps(_analysis_provenance(*(['0'*64]*4)), sort_keys=True).encode('utf-8')
    text = len(provenance) + len(version.encode('utf-8')) + len(_ANALYSIS_RUN_NAME) + len(_ANALYSIS_REASON) + 4*len(vegetation_source)
    # At most one8-byte primitive scalar per member; all byte/UCS text is
    # separately positive above. The empty bytes scalars are covered here.
    upper = payload + 8*members + text + members*_NPZ_MEMBER_OVERHEAD + 22 + 56 + 20
    return {'members_upper': members, 'numeric_array_payload_upper': payload,
            'scalar_and_text_upper': 8*members + text, 'archive_bytes_upper': upper,
            'provenance_bytes': len(provenance), 'history_schema_version': version,
            'vegetation_source': vegetation_source,
            'temporary_published_error_occurrences': 2}


def _stream_pin(path, maximum):
    import stat
    path = Path(path); before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or path.is_symlink() or path.resolve() != path
            or before.st_size > maximum): raise ValueError('analysis archive custody/size changed')
    def stamp(v): return (v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
    digest = hashlib.sha256(); size = 0
    with path.open('rb') as stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk: break
            size += len(chunk)
            if size > maximum: raise ValueError('analysis archive exceeded reservation')
            digest.update(chunk)
        held = os.fstat(stream.fileno())
    if size != before.st_size or stamp(before) != stamp(held) or stamp(before) != stamp(path.lstat()):
        raise ValueError('analysis archive changed during bounded streaming hash')
    return {'path': str(path), 'size': size, 'sha256': digest.hexdigest()}


def _save_analysis_snapshot(runner, run_root, policy, prospective, lock_id):
    from masy import dense_bed as bed_owner
    from masy.case_runner import _array_bytes
    from maple.io.outputs.snapshot import save_state_snapshot
    bed = runner._bed; state = bed.state; n = bed_owner._native()
    runner._precheck()
    bed._validate_shared(n, state)
    from masy import dense_runner as runner_owner
    if not runner_owner._native().mobile_state_is_genuinely_empty(runner.mobile_state):
        raise ValueError('analysis boundary has unresolved wind mobile sediment')
    if (bed.vegetation.objects is not None or bed.vegetation.source != prospective['vegetation_source']):
        raise ValueError('vegetation differs from admitted snapshot family')
    if (runner._case_effective.topographic_commit.dry_aeolian_commit_separation_enabled
            or getattr(state, 'aeolian_reconciliation_baseline', None) is not None):
        raise ValueError('nonNone reconciliation is not this analysis profile')
    from maple.io.outputs import snapshot
    if snapshot._EXACT_HISTORY_SCHEMA_VERSION != prospective['history_schema_version']:
        raise ValueError('Native history metadata changed after prospective admission')
    values = (state.voxel_column, state.active_layer, state.sediment_availability,
              state.water, bed.threshold_modifier_state, bed.vegetation.raster)
    leaves = bed_owner._require_host_arrays(values, 'analysis')
    payload = sum(int(array.nbytes) for _, array in leaves)
    if payload > prospective['numeric_array_payload_upper']:
        raise ValueError('current snapshot arrays exceed prospective dimension reservation')
    known = _array_bytes(values) + 2 * (1 << 20) + prospective['scalar_and_text_upper']
    check_case_process(policy, upcoming=known, checkpoint='before_Native_analysis_snapshot')
    partial = run_root / '.analysis_landscape.partial.npz'; final = run_root / 'analysis_landscape.npz'
    for path in (partial, final):
        cli.path_value(str(path), policy.max_path_chars, absolute=True)
        if path.exists() or path.is_symlink(): raise ValueError('analysis output already exists')
    reservation = cli.reserve_disk(run_root, 2*prospective['archive_bytes_upper'] + policy.max_report_bytes + 2*policy.max_output_bytes, policy)
    identity = bed.identity_record(); identity_sha = bed_owner._sha256(identity)
    provenance = _analysis_provenance(bed.case_sha256, runner._schedule.digest, identity_sha, lock_id)
    # Reuse already authenticated canonical digest metadata. No second array
    # copy/scan is made just to construct the analysis oracle projection.
    fields = identity['fields']
    member_identities = {'voxel_mass_kg': fields['voxel_column']['fields']['mass_kg'],
                         'active_mass_kg': fields['active_layer']['fields']['mass_kg']}
    for holder, prefix in (('sediment_availability', 'sediment_availability_'), ('water', 'water_')):
        if fields[holder] is not None:
            for name, value in fields[holder]['fields'].items(): member_identities[prefix + name] = value
    for name, value in bed._case_record['threshold_modifier_state']['fields'].items():
        member_identities['threshold_modifier_' + name] = value
    for name, value in bed._case_record['vegetation']['fields']['raster']['fields'].items():
        member_identities['vegetation_raster_' + name] = value
    raster_fields = ('height_m', 'cover_fraction', 'porosity', 'drag_coefficient', 'frontal_area_density')
    threshold_fields = ('gravimetric_moisture_fraction', 'clay_fraction', 'physical_crust_multiplier',
                        'chemical_cementation_multiplier', 'biological_binding_multiplier')
    if (set(f.name for f in dataclasses.fields(bed.vegetation.raster)) != set(raster_fields)
            or set(f.name for f in dataclasses.fields(bed.threshold_modifier_state)) != set(threshold_fields)):
        raise ValueError('Native snapshot field roster changed')
    provenance_json = json.dumps(provenance, sort_keys=True)
    if len(provenance_json.encode('utf-8')) != prospective['provenance_bytes']:
        raise ValueError('analysis provenance exceeds admitted exact text width')
    expected_scalars = {
        'time_s': {'kind': 'float64', 'hex': bed.time_s.hex()},
        'run_name': {'kind': 'bytes_utf8', 'value': _ANALYSIS_RUN_NAME},
        'seed': {'kind': 'int64', 'value': runner._case_effective.seed},
        'step': {'kind': 'int64', 'value': runner.accepted.n_accepted_events},
        'reason': {'kind': 'bytes_utf8', 'value': _ANALYSIS_REASON},
        'validated': {'kind': 'bool', 'value': False}, 'is_diagnostic': {'kind': 'bool', 'value': True},
        'restartable': {'kind': 'bool', 'value': False},
        'error_type': {'kind': 'bytes_utf8', 'value': ''}, 'error_message': {'kind': 'bytes_utf8', 'value': ''},
        'snapshot_schema_version': {'kind': 'int64', 'value': snapshot.SNAPSHOT_SCHEMA_VERSION},
        'history_schema_version': {'kind': 'bytes_utf8', 'value': prospective['history_schema_version']},
        'provenance_json': {'kind': 'bytes_utf8', 'value': provenance_json},
        'history_store_path': {'kind': 'bytes_utf8', 'value': ''},
        'has_vegetation': {'kind': 'bool', 'value': True},
        'vegetation_source': {'kind': 'unicode', 'value': bed.vegetation.source},
        'vegetation_has_objects': {'kind': 'bool', 'value': False},
        'has_threshold_modifier_state': {'kind': 'bool', 'value': True}}
    if state.sediment_availability is not None: expected_scalars['has_sediment_availability'] = {'kind': 'bool', 'value': True}
    if state.water is not None: expected_scalars['has_water'] = {'kind': 'bool', 'value': True}
    for name in ('seed', 'step', 'snapshot_schema_version'):
        value = expected_scalars[name]['value']
        if type(value) is not int or not -(1 << 63) <= value < (1 << 63): raise ValueError('Native snapshot int64 metadata domain')
    expected_arrays = {'voxel_mass_kg', 'active_mass_kg'}
    expected_arrays.update('vegetation_raster_' + f for f in raster_fields)
    expected_arrays.update('threshold_modifier_' + f for f in threshold_fields)
    if state.sediment_availability is not None:
        expected_arrays.update(('sediment_availability_available_mass_kg', 'sediment_availability_bound_mass_kg'))
    if state.water is not None: expected_arrays.update(('water_depth_m', 'water_mobile_mass_by_cell_class_kg'))
    if set(member_identities) != expected_arrays: raise ValueError('snapshot array identity projection differs from Native member roster')
    expected_members = sorted(expected_arrays | set(expected_scalars))
    if len(expected_members) > prospective['members_upper']:
        raise ValueError('snapshot member count exceeds reservation')
    if any(len((name + '.npy').encode('ascii')) > _NPZ_NAME_BYTES for name in expected_members):
        raise ValueError('snapshot member name exceeds Source name reservation')
    save_state_snapshot(state.voxel_column, state.active_layer, bed.time_s, partial,
        run_name=_ANALYSIS_RUN_NAME, seed=runner._case_effective.seed, step=runner.accepted.n_accepted_events,
        reason=_ANALYSIS_REASON, is_diagnostic=True, geometry=bed.context.geometry,
        mass_resolution_kg=bed.context.mass_resolution_kg, provenance=provenance,
        vegetation=bed.vegetation, threshold_modifier_state=bed.threshold_modifier_state,
        sediment_availability=state.sediment_availability, water=state.water)
    with partial.open('rb') as stream: os.fsync(stream.fileno())
    before = _stream_pin(partial, prospective['archive_bytes_upper'])
    if bed.identity_sha256() != identity_sha: raise ValueError('accepted bed changed during analysis serialization')
    os.chmod(partial, 0o444)
    try:
        os.link(partial, final)  # atomic no-overwrite publication; retain the old partial reference.
        directory = os.open(run_root, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
        pin = _stream_pin(final, prospective['archive_bytes_upper'])
        if (pin['size'], pin['sha256']) != (before['size'], before['sha256']): raise ValueError('analysis publication differs')
    except Exception as error:
        raise ValueError('analysis publication incomplete; actual partial/final remain at ' + str(partial) + ' and ' + str(final)) from error
    return {'schema': 'masy.case_analysis_reference.v1', 'artifact': pin,
            'retained_partial': str(partial), 'accepted_bed_identity': identity,
            'time_s': bed.time_s.hex(), 'member_array_identities': member_identities,
            'expected_members': expected_members, 'expected_scalars': expected_scalars, 'reservation': reservation,
            'known_wrapper_current_array_root_bytes': known,
            'Native_shared_validation_before_writer': True,
            'Native_writer_flags': {'is_diagnostic': True, 'validated': False, 'restartable': False},
            'omitted': ['ledger', 'committed_topography', 'aeolian_reconciliation_baseline',
                        'empty_wind_pool', 'MASY_schedule_cursor_clocks', 'SYRUP_soil_routing_source'],
            'scope': 'final accepted analysis landscape; not a Native/combined restart; no commit/reset/reload'}


def _native_disk_plan(inspection, inventory, policy, bed_limits):
    """Metadata-only prospective roster for the current Native compiler.

    Numeric payload occurrences are explicit. The small text/output portion
    is separately admitted from genuine Native serialization of the resolved
    authoring document; unsupported dynamic text branches refuse here.
    """
    from maple.experiment.orchestrator import _build_legacy_case_yaml_raw
    from maple.core.types.sediment_ledger import ledger_storage_bytes
    import yaml
    resolved = inspection.resolved; cfg = resolved.config; g = cfg.geometry
    cells, classes = g.ny * g.nx, len(cfg.grain_classes.classes)
    voxels = resolved.topography.nz + resolved.topography.extra_top_layers
    for value in (g.ny, g.nx, cells, classes, voxels):
        if type(value) is not int or not 0 < value < 1 << 63: raise ValueError('unsupported compiler dimension domain')
    if resolved.initial_mobile_seed.mass_by_class_kg:
        raise ValueError('case runner has not qualified nonempty initial airborne seed; refuses before compile')
    # Imported object/complex placement counts require an additional output
    # dimension law. Do not turn an authored object profile into a raster.
    if resolved.vegetation.source not in ('none', 'raster'):
        raise ValueError('object vegetation output reservation is not yet qualified; authored profile preserved')
    legacy = yaml.safe_dump(_build_legacy_case_yaml_raw(resolved)).encode('utf-8')
    if len(legacy) > inspection.input_limits.max_metadata_bytes: raise ValueError('resolved authoring metadata quota')
    dem = _dem_disk_domain(inspection, inventory, policy)
    # snapshot: voxel+active, five threshold rasters, two availability
    # grids and optional depth/mobile water. No initial cohort is admitted.
    state = 8 * cells * voxels * classes + 8 * cells * classes + 40 * cells + 16 * cells * classes + 8 * cells * (1 + classes)
    topography = 40 * cells + 10 * 8
    committed = 16 * cells + 16 * cells * classes + 16
    vegetation = 40 * cells + 4 * len(resolved.vegetation.source) + 1
    ledger = ledger_storage_bytes(g.ny, g.nx, classes)['total_bytes'] + 8
    # 28 snapshot,15 topography,6 committed,7 vegetation,14 ledger
    # members; each primitive rank<=4 header is below the Native reader's
    #10000-byte header ceiling. ZIP local+central+ZIP64 plus the exact
    #fixed-name roster upper is charged separately, plus22/archive.
    members = 28 + 15 + 6 + 7 + 14 + dem['members']
    archive = (state + topography + committed + vegetation + ledger + dem['payload']
               + members * _NPZ_MEMBER_OVERHEAD + (5 + dem['archives']) * (22 + 56 + 20))
    # Dynamic report fields: four class-ID occurrences plus numeric row
    # strings. Fixed report/provenance scaffolding is bounded by its own
    # complete Source text; resolved values appear in at most eight named
    # metadata placements (legacy, provenance sections, rebased source).
    ids = sum(len(c.class_id.encode('utf-8')) for c in cfg.grain_classes.classes)
    from maple.case_tools import report, provenance
    from maple.case_tools.importers import readers, field as imported_field
    from maple.case_tools.transforms import log as log_owner, empirical
    fixed = sum(len(Path(owner.__file__).read_bytes()) for owner in
                (report, provenance, readers, imported_field, log_owner, empirical))
    text = (8 * len(legacy) + fixed + 12 * ids + 256 * classes + 32 * policy.max_path_chars
            + 32 * dem['log_scalar_slots'])
    # String fields inside snapshot include authored run_name and fixed
    # provenance/version/reason text. Cover them as an additional text copy.
    archive += text
    published = archive + text
    if published > inspection.input_limits.max_compiled_bytes:
        raise ValueError('Source-derived Native output upper exceeds compiled-input quota')
    authored = sum(row['size'] for row in inventory['files'])
    # Original authored copy + legacy temporary + complete Native temp,
    # normalization replacement and final publication/cross-device copies.
    # Count all occurrences, including possible retained partial/error output.
    analysis = _analysis_disk_upper(cells, bed_limits.max_voxels_per_column, classes, resolved.vegetation.source)
    extra = (authored + len(legacy) + 4 * published + 2*analysis['archive_bytes_upper']
             + policy.max_report_bytes + 2 * policy.max_output_bytes)
    return {'schema': 'masy.case_compile_disk_plan.v1', 'cells': cells, 'voxels': voxels, 'classes': classes,
            'numeric_payload_bytes': state + topography + committed + vegetation + ledger,
            'npz_members_upper': members, 'native_published_bytes_upper': published,
            'authored_copy_bytes': authored, 'legacy_temporary_bytes': len(legacy),
            'native_temp_normalization_publication_error_occurrences': 4,
            'prospective_peak_extra_bytes': extra,
            'analysis_snapshot': analysis, 'dem': dem, 'unsupported': 'non-NPY/explicit transforms/other imported families/object vegetation/nonempty seed',
            'guarantee': 'Source reservation plus phase census/per-file FSIZE; no aggregate filesystem hard bound'}


def _load_water_arrays(spec, root, g, policy, copy_limits):
    import numpy as np
    declared = [spec[key] for key in cli._ARRAYS if spec[key] is not None]
    declared_files = sum(row['size'] for row in declared)
    declared_payloads = sum(math.prod(row['shape']) * (1 if row['dtype'] == 'bool' else 8) for row in declared)
    known = 3 * declared_payloads + 2 * declared_files + 10000 * len(declared)
    if declared_files > policy.max_water_input_bytes or known > copy_limits.max_known_host_bytes:
        raise ValueError('prospective NPY raw/header/map/current-next/error owners exceed declared quota')
    check_case_process(policy, upcoming=known, checkpoint='before_water_NPY_headers')
    checked = []; total_file = total_data = 0
    for key in cli._ARRAYS:
        row = spec[key]
        if row is None: continue
        path = root / cli.path_value(row['path'], policy.max_path_chars, absolute=False)
        raw, pin = cli.read_pin(path, policy.max_water_input_file_bytes)
        if (pin['size'], pin['sha256']) != (row['size'], row['sha256']): raise ValueError('water NPY input pin changed')
        total_file += pin['size']
        if total_file > policy.max_water_input_bytes: raise ValueError('water NPY file sum exceeds policy')
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
                or dtype.hasobject or stream.tell() + payload != pin['size']):
            raise ValueError('water NPY header/layout/payload differs from descriptor/native geometry')
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


def _prepare_water(spec, control, bed, limits):
    from masy.case_water import CaseWaterSpec, CaseWetClass, prepare_case_water
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
    return prepare_case_water(bed, CaseWaterSpec(**values),
        DenseRunnerControl(sediment, completion, cli.number(control['report_every_s'], 'report cadence', positive=True)), limits=limits)



def _require_extension_paths(contract):
    # The contract is finite and phase-independent: readline may never load in
    # the CLI, while resource must load before the original runtime operation.
    # No alias/module-load-set equality is asserted.
    import importlib.machinery
    import sysconfig
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    site = Path(sysconfig.get_path('platlib')).resolve(strict=True)
    dyn = Path(sysconfig.get_config_var('DESTSHARED')).resolve(strict=True)
    for row in contract['extensions']:
        path = Path(row['path']); owner = row['owner']
        if owner == 'jaxlib_mlir_hlo':
            valid = path == site / 'jaxlib/mlir/_mlir_libs/_mlirHlo.so'
        elif owner == 'numpy_fft':
            valid = path.parent == site / 'numpy/fft' and any(path.name == '_pocketfft_umath' + x for x in suffixes)
        elif owner in ('stdlib_readline', 'stdlib_resource'):
            stem = 'readline' if owner == 'stdlib_readline' else 'resource'
            valid = path.parent == dyn and any(path.name == stem + x for x in suffixes)
        else:
            valid = False
        if not valid: raise ValueError('CPU extension does not belong to its exact selected-environment owner')


def _compiled_extension_record(actual, baseline, contract, limits):
    def identities(rows):
        out = {}
        for row in rows:
            if row['path'] in out: raise ValueError('duplicate compiled identity')
            out[row['path']] = (row['size'], row['sha256'])
        return out
    old, current = identities(baseline), identities(actual)
    allowed = {row['path']: (row['size'], row['sha256']) for row in contract['extensions']}
    if any(current.get(path) != pin for path, pin in old.items()):
        raise ValueError('original compiled file missing or changed')
    if any(old.get(path, allowed.get(path)) != pin for path, pin in current.items()):
        raise ValueError('unknown or changed compiled file needs separate CPU qualification')
    # Retain every current row and every possible-next approved owner. Charge
    # metadata/file-hash work under the original compiled count/byte domains.
    prospective = dict(old)
    for path, pin in allowed.items():
        if path in prospective and prospective[path] != pin: raise ValueError('qualified compiled identity conflicts with baseline')
        prospective[path] = pin
    if (len(prospective) > limits['max_compiled_files']
            or sum(pin[0] for pin in prospective.values()) > limits['max_compiled_bytes']):
        raise ValueError('current-plus-qualified CPU files exceed original compiled quota')
    return {'baseline': baseline, 'actual': actual, 'qualified': contract['extensions'],
            'qualification_origin': contract['qualification_origin'],
            'scope': 'same-environment code-file identities only; no alias/load-set or model-outcome claim'}


def _runtime_stable_identity(record, expected, contract, limits):
    if set(record['stable']) != set(expected): raise ValueError('runtime stable field schema changed')
    for key in expected:
        if key != 'compiled' and record['stable'][key] != expected[key]:
            raise ValueError('real runtime stable identity differs from lock: ' + key)
    return _compiled_extension_record(record['stable']['compiled'], expected['compiled'], contract, limits)


def _post_native_identity(request, record, packages, baseline, contract):
    from masy import runtime_worker as rw
    from masy.importcheck import environment_record
    rw._origins(packages)
    if rw._package_digests(packages, request['limits']) != record['stable']['package_digests']:
        raise ValueError('sealed package bytes changed during case work')
    selected, _ = rw.selected_distributions(request['distributions'])
    if selected != record['stable']['selected_distributions']: raise ValueError('selected distribution metadata changed')
    env = environment_record()
    for key in ('interpreter_identity', 'numpy', 'optional_distribution_versions'):
        actual = env['executable_identity'] if key == 'interpreter_identity' else env[key]
        if actual != record['stable'][key]: raise ValueError('postcase environment identity changed: ' + key)
    compiled = rw.compiled_identities(dict(sys.modules), max_count=request['limits']['max_compiled_files'],
                                      max_bytes=request['limits']['max_compiled_bytes'])
    extension_identity = _compiled_extension_record(compiled, baseline['compiled'], contract, request['limits'])
    # Pure Python lazy imports are permitted only under actual sealed origins;
    # module-set equality would reject legitimate readers without evidence.
    return {'compiled': compiled, 'CPU_extension_identity': extension_identity, 'origins': 'all actual family modules verified',
            'selected_distributions': selected, 'optional_modules_loaded': env['optional_modules_loaded_by_upstream']}


def main(outer, base_sys_path):
    policy = cli.policy_from(outer['process_policy']); result = None
    try:
        cli.closed(outer, ('schema', 'runtime', 'expected_stable', 'case_request', 'process_policy', 'launcher', 'lock_id', 'source_heads', 'cpu_extensions'), 'outer worker request')
        if outer['schema'] != cli.OUTER_SCHEMA: raise ValueError('outer worker schema')
        request = outer['runtime']
        from masy import runtime_worker as rw
        if rw._check_request(request) != 'runtime': raise ValueError('original nested runtime operation required')
        cli.closed(outer['cpu_extensions'], ('input_pin', 'contract'), 'CPU qualification binding')
        epin = outer['cpu_extensions']['input_pin']
        eraw, current_epin = cli.read_pin(Path(epin['path']), 1 << 20)
        if current_epin != epin: raise ValueError('CPU qualification changed after parent admission')
        extensions = cli.cpu_extensions_from(eraw, lock_id=outer['lock_id'], source_heads=outer['source_heads'],
            packages=request['packages'], interpreter_identity=outer['expected_stable']['interpreter_identity'], limits=request['limits'])
        if extensions != outer['cpu_extensions']['contract']: raise ValueError('CPU qualification body differs')
        _require_extension_paths(extensions)
        original_environment = dict(os.environ)
        cold_environment = native_environment_record(original_environment, initial=True)
        check_case_process(policy, environment=original_environment, checkpoint='cold_before_native')
        record, packages = rw.check_process(request, base_sys_path)
        rw._origins({'masy': packages['masy']}, forbid=('maple', 'maple_syrup'))
        record = rw._op_runtime(request, record, packages)
        imported_environment = native_environment_record(original_environment)
        runtime_extension_identity = _runtime_stable_identity(record, outer['expected_stable'], extensions, request['limits'])
        raw, pin = cli.read_pin(Path(outer['case_request']['path']), policy.max_request_bytes)
        if pin != outer['case_request']: raise ValueError('case request changed after parent admission')
        value, actual_policy, limits = cli.request_from(raw)
        if actual_policy != policy: raise ValueError('worker policy differs from authenticated case request')
        run_root = Path(request['run']['run_dir'])
        inventory = cli.inspect_case(value['case']['directory'], limits['input_limits'])
        if inventory['authored_source_sha256'] != value['case']['authored_source_sha256']:
            raise ValueError('authored/source case differs from explicit inspected digest')
        from masy.case_workflow import resolve_case_inputs, adopt_case_inputs
        inspection = resolve_case_inputs(value['case']['directory'], mode='validate', input_limits=limits['input_limits'])
        writer_sources = _require_compile_writers()
        disk_plan = _native_disk_plan(inspection, inventory, policy, limits['bed_limits'])
        disk_plan['Native_writer_Source'] = writer_sources
        reservation = cli.reserve_disk(run_root, disk_plan['prospective_peak_extra_bytes'], policy)
        owned = run_root / 'input_case'
        check_case_process(policy, upcoming=2 * disk_plan['authored_copy_bytes'], checkpoint='before_owned_source_copy')
        _owned_case_copy(Path(value['case']['directory']), owned, inventory, policy)
        if cli.inspect_case(value['case']['directory'], limits['input_limits']) != inventory:
            raise ValueError('original case changed during owned source snapshot')
        copied = cli.inspect_case(owned, limits['input_limits'])
        if copied['files'] != inventory['files'] or copied['authored_source_sha256'] != inventory['authored_source_sha256']:
            raise ValueError('owned authored/source bytes differ')
        # First and only compile; source artifacts/temporary Native outputs are
        # already reserved above. Never load the known main-loader config drift.
        inputs = resolve_case_inputs(owned, mode='compile', input_limits=limits['input_limits'])
        cli.tree_census(run_root, maximum_entries=policy.max_input_entries, maximum_bytes=policy.max_run_bytes)
        adopted = adopt_case_inputs(inputs, bed_limits=limits['bed_limits'])
        water_spec, npy_pins, npy_plan = _load_water_arrays(value['water_profile']['spec'], Path(pin['path']).parent,
                                                         inputs.resolved.config.geometry, policy, limits['water_copy_limits'])
        prepared = _prepare_water(water_spec, value['water_profile']['control'], adopted.bed, limits['water_copy_limits'])
        runner = CaseRunner(adopted, prepared, value['schedule'], limits=limits['runner_limits'], process_policy=policy)
        while runner.stop_receipt is None and runner.accepted.next_index < len(runner._schedule.events):
            cli.reserve_disk(run_root, policy.max_report_bytes + 2 * policy.max_output_bytes, policy)
            runner.step()
            cli.tree_census(run_root, maximum_entries=policy.max_input_entries, maximum_bytes=policy.max_run_bytes)
        report = runner.report()
        analysis = _save_analysis_snapshot(runner, run_root, policy, disk_plan['analysis_snapshot'], outer['lock_id'])
        post = _post_native_identity(request, record, packages, outer['expected_stable'], extensions)
        if cli.read_pin(Path(epin['path']), epin['size'])[1] != epin: raise ValueError('CPU qualification input changed')
        final_environment = native_environment_record(original_environment)
        if cli.inspect_case(value['case']['directory'], limits['input_limits']) != inventory:
            raise ValueError('original case inputs changed during execution')
        inputs.verify(); prepared.verify(adopted.bed)
        for _, path, oldpin in npy_pins:
            if cli.read_pin(path, oldpin['size'])[1] != oldpin: raise ValueError('water input changed during execution')
        body = {'schema': 'masy.case_cpu_report.v1', 'lock_id': outer['lock_id'], 'status': report['status'],
                'process_policy': policy.as_record(), 'actual_cpu_affinity': sorted(os.sched_getaffinity(0)),
                'environment': {'cold': cold_environment, 'after_runtime': imported_environment, 'final': final_environment},
                'case_input_identity': inputs.identity_sha256, 'original_inputs': inventory, 'owned_case': str(owned),
                'schedule': runner._schedule.as_record(), 'mapping': runner._case_mapping,
                'water_identity': prepared.identity_record(), 'disk_plan': disk_plan, 'disk_admission': reservation,
                'npy_plan': npy_plan, 'analysis_landscape': analysis, 'report': report, 'runtime_post': post,
                'runtime_original_actual': record, 'runtime_extension_identity': runtime_extension_identity,
                'CPU_qualification_input': outer['cpu_extensions'],
                'restart': 'not a Native or combined restart; final analysis omits ledger/committed/private continuation fields'}
        encoded = cli.canonical(body)
        if len(encoded) > policy.max_report_bytes: raise ValueError('case report exceeds declared output quota')
        cli.reserve_disk(run_root, len(encoded) + 2 * policy.max_output_bytes, policy)
        with (run_root / 'case_report.json').open('xb') as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        result = {'schema': cli.RESULT_SCHEMA, 'status': 'complete' if report['status'] == 'complete' else 'incomplete',
                  'lock_id': outer['lock_id'], 'report_sha256': cli.sha(encoded), 'report_bytes': len(encoded),
                  'accepted': report['accepted'], 'worker_pid': os.getpid()}
        code = 0 if result['status'] == 'complete' else 2
    except Exception as error:
        traceback.print_exc()
        result = {'schema': cli.RESULT_SCHEMA, 'status': 'refused', 'lock_id': outer.get('lock_id'),
                  'code': getattr(error, 'code', type(error).__name__), 'detail': str(error)[:2000]}
        code = 2
    sys.stdout.buffer.write(cli.canonical(result)); sys.stdout.buffer.flush()
    return code
