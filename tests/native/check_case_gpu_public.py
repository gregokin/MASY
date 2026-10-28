"""One real public-command physical check, explicitly Root-controlled.

Root supplies ``case_gpu_public_scope`` with exactly seven fields: python,
launcher, store, profile, request (a pinned JSON reference), run_id and
issue_authority (a protected callable accepting that request reference).
The request/authored YAML/NPY are controlled setup only, using the existing
labelled 8x8/K1/V20/W3 synthetic case. Setup performs no compile, adoption,
genesis or transport. This test never issues authority or constructs a budget.

Root's original whole600s outer timer covers invocation AND readback; the
tool's own timer does not extend it. There is no separate oracle/model run.
The actual native parity of the owning API has its own qualification.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

from masy import case_gpu_cli as cli

pytestmark = pytest.mark.gpu


def test_public_imported_C64_case_runs_four_actual_events(case_gpu_public_scope):
    started = time.monotonic()
    scope = case_gpu_public_scope
    assert type(scope) is dict and set(scope) == {
        'python', 'launcher', 'store', 'profile', 'request', 'run_id', 'issue_authority'}
    request_pin = scope['request']
    original = cli.pinned(request_pin)
    request = cli.request_from(original)
    assert request['calibration'] == 'synthetic_test_fixture'
    assert [e['kind'] for e in request['schedule']['events']] == ['wind', 'rain', 'wind', 'rain']
    authority_pin = scope['issue_authority'](dict(request_pin))
    # The protected issuer supplies real Source/runtime/device/currentness
    # facts, never a synthetic DTO, oracle factory or test-owned fallback.
    authority = cli.read_authority(authority_pin).value
    assert authority['request'] == request_pin
    assert authority['run_id'] == scope['run_id'] and authority['profile_id'] == scope['profile']
    root = Path(scope['store']) / 'runs' / scope['run_id']
    assert not root.exists()
    argv = [scope['python'], '-I', '-B', scope['launcher'], 'run',
        '--store', scope['store'], '--profile', scope['profile'], '--authority', authority_pin['path'],
        '--authority-size', str(authority_pin['size']), '--authority-sha256', authority_pin['sha256'],
        '--request', request_pin['path'], '--run-id', scope['run_id']]
    child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        # No second physical allowance. Root's earlier outer deadline also
        # remains installed and may leave less time than this local ceiling.
        stdout, stderr = child.communicate(timeout=max(.001, 600-(time.monotonic()-started)))
    finally:
        if child.poll() is None:
            # The tool's TERM handler interrupts its monitor and reaps the
            # serial worker. These waits are cleanup, never new model work.
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=7)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        child.stdout.close(); child.stderr.close()
    assert child.returncode == 0, stderr.decode(errors='replace')[-2000:]
    assert len(stdout) + len(stderr) <= 4 << 20
    result = cli.parse(stdout, 4 << 20)
    assert result['schema'] == cli.RESULT_SCHEMA and result['status'] == 'complete'
    assert result['run_dir'] == str(root) and result['authority'] == authority_pin
    assert result['known_subset_only'] and result['restart_qualified'] is False
    assert result['general_domain_qualified'] is False
    assert len(result['children']) == 4
    assert all(c['exit_code'] == 0 and c['reaped'] and c['pidfd_exited'] and c['reason'] is None
               for c in result['children'])
    assert all(not Path('/proc/%d' % c['process']['pid']).exists() for c in result['children'])
    report = cli.parse(cli.pinned(result['report'], immutable=True), 4 << 20)
    assert report['status'] == 'complete'
    actual = report['report']
    assert actual['input_guard_failed'] is False
    assert actual['runner']['status'] == 'complete'
    receipts = actual['runner']['receipts']
    assert len(receipts) == 4 and [r['kind'] for r in receipts] == ['wind', 'rain', 'wind', 'rain']
    for r in receipts:
        assert r['status'] == r['code'] == 'accepted'
        assert all(abs(x) <= bound for x, bound in zip(
            r['aggregate']['joint_residual_kg'], r['aggregate']['joint_bound_kg'], strict=True))
        if r['kind'] == 'wind':
            assert r['native']['record']['finalizer_calls'] == 1
        else:
            water = r['water']
            assert water['rain_m3'] > 0
            assert abs(water['residual_m3']) <= water['tolerance_m3']
            assert water['dry_surface_removed_m3'] >= 0 and water['dry_soil_removed_m3'] >= 0
            assert water['dry_reset'] == 'external dry-again removal (native); soil water is zero after the reset'
    assert report['final_images']['four_genuine_orders'] == [[6, 1], [6, 6], [12, 3], [20, 1]]
    assert report['final_images']['FIRST_identities_unchanged'] is True
    binding = actual['request_binding']
    assert hashlib.sha256(cli.canonical(binding)).hexdigest() == result['binding_sha256']
    assert binding['calibration'] == 'synthetic_test_fixture'
    assert binding['storage_choices']['window_voxels'] == 3
    assert binding['storage_choices']['with_prefix_packets'] is True
    assert binding['projected_schedule']['events']
    # Five prescribed vegetation fields are protected by actual case/bed
    # verification and immutable original static binding. They are NOT a
    # final-array readback from the Native-only diagnostic below.
    assert actual['derived_cuda'] == {'kind': 'Native read-only property', 'host': False, 'device': True}
    assert report['runtime_post']['package_Source_unchanged']
    assert report['runtime_post']['environment_and_RECORD_unchanged']
    diagnostic = result['diagnostic']
    cli.pinned(diagnostic['file'])
    assert diagnostic['accepted_cursor'] == 4
    assert Path(diagnostic['deep_store_dir']).is_relative_to(root)
    assert 'not complete MASY/C64 restart' in diagnostic['label']
    assert diagnostic['prospective']['constructor_allocation'] is None
    assert diagnostic['prospective']['external_runtime'] is None
    for cursor in range(5):
        record = cli.parse((root / 'case' / ('accepted-%04d.json' % cursor)).read_bytes(), 4 << 20)
        assert record['cursor'] == cursor
    assert cli.pinned(request_pin) == original
    assert (root / 'consumption.json').exists() and (root / 'result.json').exists()
