"""Focused HOST metadata and labelled non-model process checks for the GPU CLI.

Synthetic JSON references/counts below exercise syntax only. They issue no
Source, device, provider or execution authority and never enter a case worker.
Root selects/runs these checks; no GPU fixture or transport is used here.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import io
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from masy import case_gpu_cli as cli
from masy import case_gpu_worker as worker


def _ref(path):
    return dict(path=str(path), size=7, sha256='a' * 64)


def _authority(root):
    """Illustrative closed JSON; deliberately not a usable physical grant."""
    store = root / 'store'
    profile = 'b' * 64
    return dict(schema=cli.AUTH_SCHEMA, operation=cli.OPERATION, lane=cli.LANE,
        store=str(store), profile_id=profile, profile_file=_ref(store / 'profiles' / (profile + '.json')),
        run_id='metadata-only', sources={name: dict(root=str(root / name), head='c' * 40, tree='d' * 40)
            for name in ('masy', 'maple', 'syrup')}, launcher_rel=cli.TOOL,
        request=_ref(root / 'request.json'), nvidia_smi=_ref(root / 'unissued-nvidia-smi'),
        availability=dict(gpu_uuid='GPU-00000000-0000-0000-0000-000000000000',
            pci_bus_id='00000000:01:00.0', issued_unix_ns=100, expires_unix_ns=200),
        native_image_facts={k: 31415 for k in cli.FACT_FIELDS},
        execution_budgets=dict(cli.BUDGETS), compiled_additions=[],
        case_budget_delegation=dict(issuer=_ref(root / 'worker.py'), native_source_identity='e' * 64,
            current_runtime_ref=_ref(root / 'runtime.json'), producer_source=_ref(root / 'producer.py'),
            source_manifest_sha256='f' * 64, grant_path=str(store / 'runs/metadata-only/case/case-budget.json'),
            source_bindings=[dict(module='unissued.source_%02d' % i, **_ref(root / ('source_%02d.py' % i))) for i in range(22)]))


def _syntax_request(root):
    """Only the Native-free envelope is exercised; not scientific parameters."""
    spec = {key: None for key in cli.io._SPEC_FIELDS}
    for key in cli.io._ARRAYS:
        if key in cli.io._NULL_ARRAYS:
            continue
        spec[key] = dict(kind='npy', path=key + '.npy', size=512, sha256='a' * 64,
            dtype='bool' if key in cli.io._BOOL_ARRAYS else 'float64',
            shape=[10, 10] if key in ('elevation_full_m', 'export_receiver_full') else [8, 8])
    control = dict(sediment_control={k: None for k in cli.io._SEDIMENT},
        completion_policy={k: None for k in cli.io._COMPLETION}, report_every_s=1.)
    control['sediment_control']['storm'] = {k: None for k in cli.io._STORM}
    control['sediment_control']['storm']['implementation'] = 'array'
    rain = dict(kind='rain', water_profile_id='syntax-only',
        intervals=[dict(duration_s=1., intensity_mm_h=1.)], recession_guard_s=1.)
    value = dict(schema=cli.REQUEST_SCHEMA, label='synthetic syntax only; NOT MEASURED',
        calibration='synthetic_test_fixture', case=dict(directory=str(root),
            authored_source_sha256='a' * 64, copy_mode='authored_source_then_compile'),
        windowed_storage=dict(datum_offset_m=0., window_voxels=3, chunk_columns=64,
            rows_per_band=2, with_prefix_packets=True), water_profile=dict(spec=spec, control=control),
        native_limits=dict(profile={}, record_limits={}),
        io_policy=dict(zip(cli.IO_FIELDS, (4096, 64 << 20, 64 << 20, 4 << 20, 4 << 20, 4096, 2 << 20), strict=True)),
        backend_change=dict(field='sediment_control.storm.implementation', **{'from': 'array', 'to': 'cuda'}),
        schedule=dict(schema='masy.case_schedule.v1', events=[dict(kind='wind', first_block=0, stop_block=1),
            copy.deepcopy(rain), dict(kind='wind', first_block=1, stop_block=2), copy.deepcopy(rain)]))
    for key in ('input_limits', 'compile_bed_limits', 'windowed_limits', 'water_copy_limits',
                'bridge_limits', 'bed_limits', 'resource_limits', 'runner_limits', 'placement_limits'):
        value[key] = {}
    return value


def test_authority_expiry_is_initial_only_and_immutable_bytes_stay_bound(tmp_path, monkeypatch):
    value = _authority(tmp_path)
    pin = cli.write_new(tmp_path / 'authority.json', value, immutable=True)
    monkeypatch.setattr(cli.time, 'time_ns', lambda: 150)
    issued = cli.read_authority(pin)
    raw = (tmp_path / 'authority.json').read_bytes()
    monkeypatch.setattr(cli.time, 'time_ns', lambda: 250)
    assert issued.revalidate() == value
    assert (tmp_path / 'authority.json').read_bytes() == raw
    with pytest.raises(ValueError, match='not currently issued'):
        cli.read_authority(pin)
    path = Path(pin['path']); path.chmod(0o644)
    path.write_bytes(raw + b' '); path.chmod(0o444)
    with pytest.raises(ValueError):
        issued.revalidate()


@pytest.mark.parametrize('kind', ('missing', 'extra', 'bool_budget', 'timestamps', 'repeated_source', 'other_grant_path'))
def test_authority_closed_joins_refuse_before_any_runtime(tmp_path, kind):
    value = _authority(tmp_path)
    if kind == 'missing': del value['native_image_facts']
    elif kind == 'extra': value['unissued_override'] = True
    elif kind == 'bool_budget': value['execution_budgets']['wall_abort_seconds'] = True
    elif kind == 'timestamps': value['availability']['expires_unix_ns'] = 100
    elif kind == 'repeated_source': value['case_budget_delegation']['source_bindings'][1] = copy.deepcopy(value['case_budget_delegation']['source_bindings'][0])
    else: value['case_budget_delegation']['grant_path'] = str(tmp_path / 'elsewhere.json')
    with pytest.raises(ValueError): cli.validate_authority(value, at_issue=True, now_ns=150)


def test_compiled_additions_are_required_exact_rows_not_vendor_allowance(tmp_path):
    value = _authority(tmp_path)
    row = dict(path=str(tmp_path / 'unissued.so'), size=(64 << 20) + 1, sha256='a' * 64)
    value['compiled_additions'] = [row]  # Immutable input binary, not an owned output file.
    assert cli.validate_authority(value, at_issue=True, now_ns=150) is value
    for bad in (None, [row, row], [dict(row, owner='any_vendor')], [dict(row, size=True)]):
        candidate = copy.deepcopy(value); candidate['compiled_additions'] = bad
        with pytest.raises(ValueError): cli.validate_authority(candidate, at_issue=True, now_ns=150)


@pytest.mark.parametrize('raw', (b'{"a":0,"a":1}', b'{"a":NaN}', b'{"a":1e999}',
                                b'[' * 33 + b'0' + b']' * 33,
                                b'{"a":340282366920938463463374607431768211456}'))
def test_json_refuses_duplicate_nonfinite_depth_and_integer_domain(raw):
    with pytest.raises(ValueError): cli.parse(raw)


@pytest.mark.parametrize('kind', ('extra', 'run_dir', 'backend', 'bool_width', 'npy_dtype', 'schedule'))
def test_request_exact_envelope_refusals(tmp_path, kind):
    request = _syntax_request(tmp_path)
    assert cli.request_from(cli.canonical(request)) == request
    if kind == 'extra': request['source_guard_override'] = True
    elif kind == 'run_dir': request['bridge_limits']['run_dir'] = str(tmp_path)
    elif kind == 'backend': request['backend_change']['to'] = 'array'
    elif kind == 'bool_width': request['windowed_storage']['chunk_columns'] = True
    elif kind == 'npy_dtype': request['water_profile']['spec']['active_mask']['dtype'] = 'float64'
    else: request['schedule']['events'][0]['absolute_start_s'] = 0.
    with pytest.raises(ValueError): cli.request_from(cli.canonical(request))


def _npy(array):
    import numpy as np
    stream = io.BytesIO(); np.save(stream, array, allow_pickle=True)
    return stream.getvalue()


@pytest.mark.parametrize('key,shape,dtype,payload', (
    ('friction_factor', (8, 8), 'float64', 512),
    ('active_mask', (8, 8), 'bool', 64),
    ('elevation_full_m', (10, 10), 'float64', 800),
    ('export_receiver_full', (10, 10), 'bool', 100)))
def test_real_numpy_closed_header_accepts_only_declared_layout(key, shape, dtype, payload):
    import numpy as np
    raw = _npy(np.zeros(shape, dtype=dtype))
    row = dict(size=len(raw), shape=list(shape))
    before = bytes(raw)
    assert worker._water_npy_header(raw, row, key, SimpleNamespace(ny=8, nx=8)) == payload
    assert raw == before


@pytest.mark.parametrize('kind', ('truncated_header', 'truncated_payload', 'trailing', 'object', 'fortran', 'dtype', 'shape', 'rank'))
def test_real_numpy_header_corruption_refuses_without_load_or_process_bypass(kind):
    import numpy as np
    array = np.zeros((8, 8), dtype='float64')
    if kind == 'object': array = np.zeros((8, 8), dtype=object)
    elif kind == 'fortran': array = np.asfortranarray(array)
    elif kind == 'dtype': array = np.zeros((8, 8), dtype='float32')
    elif kind == 'shape': array = np.zeros((7, 8), dtype='float64')
    elif kind == 'rank': array = np.zeros((8, 8, 1), dtype='float64')
    raw = _npy(array)
    if kind == 'truncated_header': raw = raw[:11]
    elif kind == 'truncated_payload': raw = raw[:-1]
    elif kind == 'trailing': raw += b'x'
    row = dict(size=len(raw), shape=[8, 8])
    with pytest.raises((ValueError, EOFError)):
        worker._water_npy_header(raw, row, 'friction_factor', SimpleNamespace(ny=8, nx=8))


def test_native_policy_decoder_uses_genuine_closed_profile_and_impact_classes():
    from maple.surface.active_layer.bed_exchange_device import DeviceWindowedK1DimensionProfile, DeviceK1DimensionImpactLimits
    image = dict(ny=8, nx=8, n_voxels=20, window_voxels=3, n_classes=1, hop_bins=8)
    profile = worker._record(DeviceWindowedK1DimensionProfile, image)
    assert type(profile) is DeviceWindowedK1DimensionProfile
    values = dict(max_operations=2, max_gather_columns=4, max_journal_records=320, max_cells=64,
        max_classes=1, max_voxels=20, max_impact_rows=512, max_new_device_bytes=0,
        max_host_staging_bytes=0, max_logical_deep_read_bytes=0, max_mapped_bytes=0,
        device_reserve_bytes=0, host_reserve_bytes=0, device_safety_fraction=0.5)
    # Constructor metadata only. Zero ceilings issue no storage/transport ability.
    result = worker._record(DeviceK1DimensionImpactLimits, values, profile=profile)
    assert result.profile is profile and result.max_impact_rows == 512
    for changed in (dict(image, ny=True), dict(image, implicit_default=1), {k:v for k,v in image.items() if k != 'hop_bins'}):
        with pytest.raises(ValueError): worker._record(DeviceWindowedK1DimensionProfile, changed)
    for changed in (dict(values, max_impact_rows=True), dict(values, profile=image), dict(values, device_safety_fraction=1.)):
        with pytest.raises(ValueError): worker._record(DeviceK1DimensionImpactLimits, changed, profile=profile)


def test_genuine_case_budget_encoding_is_grant_independent_metadata_only(tmp_path):
    from maple_syrup import windowed_shared_bed_device as shared
    ref = lambda name: shared.SourceReference(str(tmp_path / name), 7, 'a' * 64)
    common = dict(native_source_identity='b' * 64, current_runtime_ref=ref('runtime.json'),
        producer_module='masy.case_device_c64', producer_source=ref('case_device_c64.py'),
        source_bindings=tuple((name, ref('source_%02d.py' % i)) for i,name in enumerate(shared._SOURCE_MODULES)),
        gpu_uuid='GPU-00000000-0000-0000-0000-000000000000', source_manifest_sha256='c' * 64,
        case_request_binding_sha256='d' * 64)
    # Explicitly synthetic references: only the actual owning metadata encoder
    # is called, never runtime validation, publication, Native or delivery.
    a = shared.DeviceWaterC64CaseBudget(grant=ref('preauthorization.json'), **common)
    b = shared.DeviceWaterC64CaseBudget(grant=ref('final-budget.json'), **common)
    raw = shared.c64_case_budget_bytes(a)
    assert raw == shared.c64_case_budget_bytes(b)
    record = cli.parse(raw)
    assert 'grant' not in record and 'preauthorization.json' not in raw.decode()
    assert all(row[1] is None for row in record['opaque_external_bounds'])
    assert all(row[1] is None for row in record['opaque_primitive_units'])
    assert record['opaque_constructor_accounting'] is None


def test_full_genuine_HOST_policy_decode_keeps_nested_profile_and_native_run_path(tmp_path):
    """Real initial HOST preparation; no GPU placement, transport or authority.

    Root explicitly permits the existing labelled HOST helper here. It builds
    the complete genuine policy family instead of fake Native headers/classes.
    The public physical selector still performs its own sole compilation
    inside the command, never through this separate metadata check.
    """
    from check_case_device_c64 import _host
    from masy.case_device_c64 import _bridge_image
    from masy.case_water import CaseWaterCopyLimits
    from masy.dense_bed import DenseBedLimits
    host, prepared, kwargs = _host(tmp_path)
    native = kwargs['record_limits']
    profile = dataclasses.asdict(native.block.profile)
    record = dataclasses.asdict(native)
    for part in (record['block'], record['block']['step'], record['block']['step']['bed_limits']):
        assert part.pop('profile') == profile
    bridge = dataclasses.asdict(kwargs['bridge_limits']); assert bridge.pop('run_dir') == tmp_path
    value = dict(input_limits=dataclasses.asdict(host._input_limits),
        compile_bed_limits=dataclasses.asdict(DenseBedLimits(128 << 20, 20, 64 << 20)),
        windowed_limits=dataclasses.asdict(host._limits),
        water_copy_limits=dataclasses.asdict(CaseWaterCopyLimits(64, 1, 128 << 20, 4 << 20)),
        native_limits=dict(profile=profile, record_limits=record), bridge_limits=bridge)
    for key in ('bed_limits', 'resource_limits', 'runner_limits', 'placement_limits'):
        value[key] = dataclasses.asdict(kwargs[key])
    original = copy.deepcopy(value)
    restored = worker._decode_case_limits(value, tmp_path)
    assert value == original
    assert dataclasses.asdict(restored['record_limits']) == dataclasses.asdict(native)
    block = restored['record_limits'].block
    assert block.profile is block.step.profile is block.step.bed_limits.profile
    assert block.step.bed_limits.max_impact_rows == 512
    assert block.step.creep_limits.max_impact_rows == block.solve.bed_limits.max_impact_rows == block.validation.bed_limits.max_impact_rows == 256
    assert type(restored['bridge_limits']) is type(kwargs['bridge_limits'])
    assert restored['bridge_limits'].run_dir is tmp_path
    assert _bridge_image(restored['bridge_limits']) == _bridge_image(kwargs['bridge_limits'])
    for key in ('input_limits', 'compile_bed_limits', 'windowed_limits', 'water_copy_limits',
                'bed_limits', 'resource_limits', 'runner_limits', 'placement_limits'):
        assert dataclasses.asdict(restored[key]) == value[key]
    drift = copy.deepcopy(value); drift['bridge_limits']['run_dir'] = str(tmp_path)
    with pytest.raises(ValueError): worker._decode_case_limits(drift, tmp_path)
    host.verify(); prepared.verify(host)


def test_native_environment_projection_is_two_closed_observed_outcomes_only():
    baseline={'CUDA_VISIBLE_DEVICES':'synthetic-syntax-only','LC_ALL':'C'}
    before=copy.deepcopy(baseline)
    post=dict(baseline,TF_CPP_MIN_LOG_LEVEL='1')
    projected,branch=worker._project_native_environment(baseline,post,False)
    assert projected==post and branch=='no_TPU_skip_assignment_observed'
    skipped=dict(post,TPU_SKIP_MDS_QUERY='1')
    assert worker._project_native_environment(baseline,skipped,False)[0]==skipped
    preset=dict(baseline,TF_CPP_MIN_LOG_LEVEL='9')
    assert worker._project_native_environment(preset,preset,False)[0]==preset
    assert baseline==before
    for changed in (dict(post,UNKNOWN_JAX_KEY='1'), dict(post,TF_CPP_MIN_LOG_LEVEL='0'),
                    dict(post,TPU_SKIP_MDS_QUERY='0'), {k:v for k,v in post.items() if k!='LC_ALL'}):
        with pytest.raises(ValueError): worker._project_native_environment(baseline,changed,False)
    for flag in (True,0,None):
        with pytest.raises(ValueError): worker._project_native_environment(baseline,post,flag)
    with pytest.raises(ValueError):
        worker._project_native_environment(dict(baseline,TPU_SKIP_MDS_QUERY='1'),skipped,False)


def test_exclusive_readonly_budget_publication_refuses_replay_and_mutation(tmp_path):
    path = tmp_path / 'budget.json'
    raw = b'{"synthetic_metadata_only":true}'
    pin = cli.write_new(path, raw, raw=True, immutable=True)
    assert cli.pinned(pin, immutable=True) == raw
    assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError): cli.write_new(path, raw, raw=True, immutable=True)
    path.chmod(0o644)
    with pytest.raises(ValueError, match='custody'): cli.pinned(pin, immutable=True)


def test_census_discards_partial_walk_and_recovers_real_disappearance(tmp_path, monkeypatch):
    stable = tmp_path / 'stable'; stable.write_bytes(b'stable')
    gone = tmp_path / 'labelled-startup.json.part'; gone.write_bytes(b'partial')
    original = cli.os.scandir; calls = []
    @contextlib.contextmanager
    def race(path):
        calls.append(str(path))
        with original(path) as entries: rows = list(entries)
        if len(calls) == 1: gone.unlink()
        yield iter(rows)
    monkeypatch.setattr(cli.os, 'scandir', race)
    assert cli.census(tmp_path) == dict(bytes=6, entries=1)
    assert len(calls) == 2


def test_census_exhausts_exactly_three_whole_walks_and_does_not_retry_symlinks(tmp_path, monkeypatch):
    original = cli.os.scandir; calls = []
    @contextlib.contextmanager
    def race(path):
        calls.append(str(path)); gone = tmp_path / ('labelled-%d.part' % len(calls)); gone.write_bytes(b'x')
        with original(path) as entries: rows = list(entries)
        gone.unlink(); yield iter(rows)
    with monkeypatch.context() as patch:
        patch.setattr(cli.os, 'scandir', race)
        with pytest.raises(FileNotFoundError): cli.census(tmp_path)
    assert len(calls) == 3
    (tmp_path / 'alias').symlink_to(tmp_path)
    with pytest.raises(ValueError, match='symlink/special'): cli.census(tmp_path)


def test_census_special_file_and_entry_overflow_are_not_credited(tmp_path):
    special = tmp_path / 'fifo'; os.mkfifo(special)
    with pytest.raises(ValueError, match='symlink/special'): cli.census(tmp_path)
    special.unlink()
    for i in range(4097): (tmp_path / ('empty-%04d' % i)).touch()
    with pytest.raises(ValueError, match='entry cap'): cli.census(tmp_path)


def test_one_deadline_is_not_renewed_between_serial_stages(monkeypatch):
    now = [1000.]
    monkeypatch.setattr(cli.time, 'monotonic', lambda: now[0])
    clock = cli.Deadline(1000.)
    now[0] += 590.; assert clock.remaining() == 10.
    now[0] += 9.; assert clock.remaining() == 1.
    now[0] += 1.
    with pytest.raises(TimeoutError): clock.remaining()
    with pytest.raises(ValueError): cli.Deadline(True)


def test_nonmodel_terminated_child_has_real_birth_pidfd_and_reap():
    """Root may run this HOST-only OS check under the existing20/25CPU cap."""
    code = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(60)"
    child = subprocess.Popen([sys.executable, '-I', '-B', '-c', code], start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fd = None
    try:
        assert select.select([child.stdout], [], [], 3)[0]
        assert child.stdout.readline() == b'ready\n'
        identity = cli.birth(child.pid); fd = os.pidfd_open(child.pid)
        assert identity['pid'] == child.pid and identity['start_ticks'].isdigit()
        row = cli.reap_owned(child, fd, terminate=True, deadline=time.monotonic()-1)
        fd = None
        assert row['exit_code'] == -signal.SIGKILL and row['reaped'] and row['pidfd_exited']
        assert row['reap_after_deadline_s'] >= 1 and child.returncode == -signal.SIGKILL
        assert not Path('/proc/%d' % child.pid).exists()
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL); child.wait()
        if fd is not None: os.close(fd)
        child.stdout.close(); child.stderr.close()


def test_nonmodel_already_exited_child_keeps_exact_exit_and_reap():
    child = subprocess.Popen([sys.executable, '-I', '-B', '-c', 'raise SystemExit(7)'], start_new_session=True)
    fd = os.pidfd_open(child.pid)
    try:
        assert select.select([fd], [], [], 3)[0]
        row = cli.reap_owned(child, fd, terminate=False, deadline=time.monotonic()+5)
        fd = None
        assert row['exit_code'] == 7 and row['reaped'] and row['pidfd_exited']
        assert row['reap_after_deadline_s'] == 0
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL); child.wait()
        if fd is not None: os.close(fd)


def test_public_launcher_reads_genuine_GPU_profile(case_gpu_profile_scope):
    """Root supplies actual immutable metadata; no fake profile or run grant.

    Protected fixture fields are store, profile, profile_ref and launcher.
    The genuine owner performs its real Source/profile checks. This calls
    no public main, Native bootstrap, provider, model or GPU operation.
    """
    import importlib.util
    from masy import gpu_runtime as gpu

    scope = case_gpu_profile_scope
    assert type(scope) is dict and set(scope) == {'store', 'profile', 'profile_ref', 'launcher'}
    store = Path(scope['store'])
    assert store.is_absolute() and store.resolve() == store
    profile_id = scope['profile']
    assert type(profile_id) is str and len(profile_id) == 64
    assert all(ch in '0123456789abcdef' for ch in profile_id)
    profile_ref, launcher_ref = scope['profile_ref'], scope['launcher']
    assert Path(profile_ref['path']) == store / 'profiles' / (profile_id + '.json')
    assert Path(launcher_ref['path']).name == 'masy_case_gpu.py'
    profile_raw = cli.pinned(profile_ref, immutable=True)
    launcher_raw = cli.pinned(launcher_ref, immutable=True)
    profile, _limits, _bootstrap = gpu._load_profile(str(store), profile_id)
    assert profile['profile_id'] == profile_id and profile['store'] == str(store)
    spec = importlib.util.spec_from_file_location('_profile_digest_metadata_launcher', launcher_ref['path'])
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    prefix = launcher._sealed_prefix(str(store), profile_id)
    assert prefix == str(Path(profile['objects']['masy']['path']) / 'source' / 'src')
    assert cli.pinned(profile_ref, immutable=True) == profile_raw
    assert cli.pinned(launcher_ref, immutable=True) == launcher_raw
