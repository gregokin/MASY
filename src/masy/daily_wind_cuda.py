"""Actual MAPLE wind on one current Dense CUDA bed, under an explicit profile.

Only source checks, staging, execution order and accounting live here. No wind
or water formula is implemented. Old CPU factories are unchanged.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from masy import daily_wind as dw
from masy.clock import _native_clock, encode_time
from masy.daily_csv import WindOccurrence
from masy.gpu_contract import GpuPolicy, GpuContractError, clone_device_tree, device_leaves, footprint, host_vector

PROFILE = 'masy.daily_triplet_wind.cuda.v1'
MAPLE_COMMIT = '1dca8539b87bf80e2b4bb72d4c6637d02013a305'
MAPLE_PACKAGE_SHA256 = '191416153d3eca03e0b5e0817720d1365055de5b0b37391ac95454321aee7d68'
SYRUP_PACKAGE_SHA256 = '96561d67a46cff571f06cbbd0a687354e438edc4a73435d6944fcec3f1f325f8'  # explicit frozen development/qualification identity


def _sources():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
        ('daily_wind_cuda.py', 'gpu_contract.py', 'daily_triplet_cuda.py', 'daily_wind.py',
         'daily_wind_handoff.py', 'daily_triplet.py', 'daily_triplet_wind.py', 'daily_csv.py', 'daily_schedule.py', 'clock.py')}


def _source(source, admission):
    from maple_syrup import native_case_profile as ncp
    if type(source) is not ncp.NativeSourceProfile or type(admission) is not ncp.NativeCaseAdmission:
        raise dw.DailyWindError('native_source', 'genuine CUDA source and case admission required')
    source.recheck()
    admission.recheck()
    if (not source.device_runtime_capability or not admission.device_runtime_capability
            or source.profile != ncp.CURRENT_CUDA_PROFILE_NAME
            or admission.profile_name != source.profile or source.maple_commit != MAPLE_COMMIT
            or source.package_digest != MAPLE_PACKAGE_SHA256
            or source.syrup_package_digest != SYRUP_PACKAGE_SHA256 or SYRUP_PACKAGE_SHA256 == '0' * 64
            or admission.bundle_id != source.bundle_id or admission.package_digest != source.package_digest
            or admission.syrup_package_digest != source.syrup_package_digest):
        raise dw.DailyWindError('native_source', 'the separately qualified exact current CUDA source pair is required')
    return {'profile': source.profile, 'maple_commit': source.maple_commit, 'maple_package': source.package_digest,
            'syrup_package': source.syrup_package_digest, 'bundle_id': source.bundle_id,
            'case_sha256': admission.case_identity_sha256, 'latitude': admission.latitude_binding()}


def prepare_daily_wind_cuda(source, admission, context, *, limits, policy):
    import numpy as np
    if type(limits) is not dw.DailyWindLimits or type(policy) is not GpuPolicy:
        raise dw.DailyWindError('limits', 'explicit wind limits and two-pool CUDA policy required')
    policy.gate()
    facts = _source(source, admission)
    n = dw._native()
    cp = policy.gate()
    n.np = cp
    dw._check_context(n, admission, context)
    cfg = admission.compiled.config
    if (cfg.execution.backend != 'cupy' and cfg.execution.backend != 'cuda'):
        raise dw.DailyWindError('backend', 'the authored native wind case must explicitly select GPU execution')
    if (cfg.water_coupling.enabled or cfg.topographic_commit.dry_aeolian_commit_separation_enabled
            or cfg.prescribed_external_shear.enabled or cfg.aeolian_process_mode.mode != dw.SUPPORTED_WIND_MODE
            or cfg.avalanching.enabled or cfg.surface_creep.enabled
            or getattr(getattr(cfg.geometry, 'voxel_growth', None), 'enabled', False)):
        raise dw.DailyWindError('unsupported_profile', 'current bounded CUDA profile requires the admitted baseline wind controls')
    if cfg.geometry.nx * cfg.geometry.ny > limits.max_cells:
        raise dw.DailyWindError('limits', 'case cells exceed wind cap')
    # The native preparer placed the initial availability with the bed. Static
    # threshold modifiers are a single explicit cold transfer, not a new law.
    h, d = footprint(admission.compiled.threshold_modifier_state)
    policy.admit(h + (4 << 20), d + h + context.initial_available_fraction.nbytes, 'wind static threshold placement')
    from maple.core.backend import to_device_tree
    threshold = to_device_tree(admission.compiled.threshold_modifier_state, cp)
    fraction = context.initial_available_fraction.copy()
    identity = {'profile': PROFILE, 'source': facts, 'wrapper_sources': _sources(), 'limits': limits.as_record(),
                'policy': policy.record(), 'config': dw._sha(dw._canonical(np, cfg, 'config')),
                'prescribed_sha256': dw._prescribed_digest(admission.compiled)}
    return CudaWindExecutor(source, admission, context, limits, policy, threshold, fraction, identity)


class CudaWindExecutor:
    def __init__(self, source, admission, context, limits, policy, threshold, fraction, identity):
        self._source, self._admission, self._context = source, admission, context
        self._limits, self.policy, self._threshold = limits, policy, threshold
        self._fraction = fraction
        self._identity_json = dw._json(identity)
        self.identity_sha256 = hashlib.sha256(self._identity_json.encode()).hexdigest()

    def identity(self):
        return json.loads(self._identity_json)

    def _recheck(self):
        self.policy.gate()
        if _source(self._source, self._admission) != self.identity()['source'] or _sources() != self.identity()['wrapper_sources']:
            raise dw.DailyWindError('native_source', 'source/case/wrapper changed during the run')
        import numpy as np
        if dw._sha(dw._canonical(np, self._admission.compiled.config, 'config')) != self.identity()['config']:
            raise dw.DailyWindError('native_source', 'the admitted case configuration changed')
        if dw._prescribed_digest(self._admission.compiled) != self.identity()['prescribed_sha256']:
            raise dw.DailyWindError('native_source', 'the admitted static source fields changed')
        cp = self.policy.gate()
        if not bool(cp.array_equal(self._context.initial_available_fraction, self._fraction)):
            raise dw.DailyWindError('native_source', 'the admitted initial availability map changed')

    def execute_finalized(self, bed, occurrence, start_s, *, vegetation, external_reserve_bytes=0):
        import numpy as np
        from maple.core.backend import to_device_tree, read_transfer_counters
        self._recheck()
        cp = self.policy.gate()
        n = dw._native()
        n.np = cp
        g, lim = self._context.geometry, self._limits
        if type(occurrence) is not WindOccurrence or not occurrence.blocks or len(occurrence.blocks) > lim.max_blocks:
            raise dw.DailyWindError('occurrence', 'one bounded parsed wind occurrence required')
        start_s = _native_clock(start_s, 'wind start')
        if type(start_s) is not float or type(external_reserve_bytes) is not int or external_reserve_bytes < 0:
            raise dw.DailyWindError('clock', 'actual float native start and explicit nonnegative reserve required')
        if type(bed) is not n.BedState or n.is_windowed_bed(bed):
            raise dw.DailyWindError('bed', 'the new CUDA wind supports actual Dense native beds only')
        rows = device_leaves(bed, self.policy, what='wind bed')
        device_leaves(vegetation, self.policy, what='current vegetation')
        nv, nc = int(bed.voxel_column.mass_kg.shape[2]), len(self._context.grain_classes.classes)
        if nv > lim.max_voxels or bed.voxel_column.mass_kg.shape != (g.ny, g.nx, nv, nc):
            raise dw.DailyWindError('bed', 'native voxel axes or declared depth cap differ')
        staged_bytes = sum(int(a.nbytes) for _, a in rows)
        if staged_bytes > lim.max_stage_bytes:
            raise dw.DailyWindError('limits', 'Dense transaction copy exceeds its explicit byte cap')
        record = n.resolve_wind_forcing_record({'blocks': [dict(shear_velocity_m_s=b.shear_velocity_m_s,
            direction_deg=b.direction_deg, duration_s=b.duration_s) for b in occurrence.blocks]})
        effective = dw._effective_wind_config(self._admission.compiled.config, self._context)
        config = n.build_scheduler_physics_config(effective, initial_available_fraction=self._context.initial_available_fraction,
                                                 wind_record=record)
        plans = dw._plan_blocks(n, record, config.outer_timestep_s, lim)
        capacity = config.particle_transport.max_active_cohorts
        if capacity > lim.max_cohorts:
            raise dw.DailyWindError('limits', 'configured native cohort pool exceeds cap')
        empty = n.zeros_mobile_cohort_state(max_cohorts=0)
        pool_rows = []
        dw._bed_helpers().leaves(empty, 'pool', pool_rows)
        item_bytes = sum(int(a.dtype.itemsize) for _, a in pool_rows)
        pool_bytes = capacity * item_bytes
        work = 4 * staged_bytes + 3 * pool_bytes + 8 * g.nx * g.ny * (nv + 1) * nc + 32 * sum(p['upper'] for p in plans)
        if work + external_reserve_bytes > lim.max_work_bytes:
            raise dw.DailyWindError('limits', 'native wind staging/work frontier exceeds explicit device cap')
        self.policy.admit(pool_bytes + (8 << 20), work, 'native CUDA wind', device_extra=external_reserve_bytes)
        dw._validate_native(n, bed, self._context, start_s)
        from maple.core.types.vegetation import validate_vegetation_state
        validate_vegetation_state(vegetation, g.ny, g.nx)
        transfers = read_transfer_counters()
        before = host_vector(n.bed_inventory(bed), size=nc)
        stage = clone_device_tree(bed, self.policy)
        pool = to_device_tree(n.zeros_mobile_cohort_state(max_cohorts=capacity), cp)
        if not n.mobile_state_is_genuinely_empty(pool):
            raise dw.DailyWindError('mobile', 'actual native fresh pool is not empty')
        state = n.initial_scheduler_state(stage.voxel_column, stage.active_layer, pool, stage.ledger,
            stage.committed_topography, self._threshold, time_s=start_s, vegetation=vegetation,
            sediment_availability=stage.sediment_availability, water=stage.water)
        accumulator = n.BlockAccountingAccumulator(nc)
        bounds, scales, blocks = [], 0.0, []
        for i, rb in enumerate(record.blocks):
            t0 = float(state.physical.time_s)
            block = n.WindBlock(rb.shear_velocity_m_s, rb.direction_deg, rb.duration_s, t0, t0 + rb.duration_s)
            result = dw._advance_wind_block(state, block, config)
            if (not result.conservation_ok or result.process_mode != dw.SUPPORTED_WIND_MODE
                    or result.integration_path != dw.SUPPORTED_PATH
                    or result.accounting.n_accounting_steps > plans[i]['upper']
                    or result.accounting.n_accounting_steps > lim.max_accounting_steps
                    or result.state.block_index != i or result.state.block_offset_s < block.duration_s - n.record_end_epsilon_s):
                raise dw.DailyWindError('incomplete', 'actual native wind block/record/conservation did not complete')
            a = result.accounting
            v = {name: host_vector(getattr(a, name), size=nc) for name in ('bed_change_by_class_kg',
                'mobile_mass_after_by_class_kg', 'mobile_mass_before_by_class_kg', 'boundary_input_by_class_kg', 'boundary_export_by_class_kg')}
            residual = v['bed_change_by_class_kg'] + v['mobile_mass_after_by_class_kg'] - v['mobile_mass_before_by_class_kg'] - v['boundary_input_by_class_kg'] + v['boundary_export_by_class_kg']
            bound = float(n.summation_error_bound_kg(max(a.n_accounting_steps, 1), n.reservoir_arithmetic_scale_kg(a, result.state.physical)))
            if not np.all(np.abs(residual) <= bound):
                raise dw.DailyWindError('accounting', 'native per-class wind block closure exceeds its native bound')
            for name in ('mobile_mass_before_by_class_kg', 'mobile_mass_after_by_class_kg', 'boundary_input_by_class_kg', 'boundary_export_by_class_kg'):
                value = getattr(a, 'water_' + name, None)
                if value is not None and np.any(host_vector(value, size=nc) != 0):
                    raise dw.DailyWindError('water_changed', 'native wind executed water transport')
            if bool(cp.any(result.state.physical.water.mobile_mass_by_cell_class_kg != 0)):
                raise dw.DailyWindError('water_changed', 'water-borne mobile sediment appeared in wind')
            t1 = float(result.state.physical.time_s)
            if not t1 > t0:
                raise dw.DailyWindError('clock', 'native wind did not advance')
            accumulator.add(a)
            bounds.append(bound)
            scales += sum(float(np.abs(v[name]).sum()) for name in ('bed_change_by_class_kg', 'boundary_input_by_class_kg', 'boundary_export_by_class_kg'))
            blocks.append(dict(index=i, start_s=encode_time(t0), end_s=encode_time(t1), native_bound_kg=bound))
            state = dataclasses.replace(result.state, block_index=i + 1, block_offset_s=0.0)
            del result
        total, end = accumulator.total, float(state.physical.time_s)
        final = dw._finalize_wind_event(state, config, record, record_origin_s=start_s)
        if (type(final) is not n.WindEventFinalizationResult or not final.completed or not final.water_handoff_ready
                or float(final.time_s).hex() != end.hex() or float(final.state.physical.time_s).hex() != end.hex()
                or final.state.block_index != len(record.blocks) or final.state.block_offset_s != 0.0
                or not n.mobile_state_is_genuinely_empty(final.state.physical.mobile_state)):
            raise dw.DailyWindError('incomplete', 'native wind finalization did not produce a genuine completed empty state')
        phys = final.state.physical
        candidate = n.BedState(*(getattr(phys, name) for name in dw._SHARED_FIELDS))
        dw._validate_native(n, candidate, self._context, end)
        if phys.vegetation is not vegetation or phys.threshold_modifier_state is not self._threshold or phys.ledger_reconciliation_baseline is not None:
            raise dw.DailyWindError('static_changed', 'native wind replaced its event-held static fields')
        for (_, a), (_, b) in zip(device_leaves(bed.water, self.policy), device_leaves(candidate.water, self.policy), strict=True):
            if a.dtype != b.dtype or a.shape != b.shape or not bool(cp.array_equal(a.view(cp.uint8), b.view(cp.uint8))):
                raise dw.DailyWindError('water_changed', 'wind did not preserve all real surface-water leaves bitwise')
        names = ('airborne_mass_before_by_class_kg', 'airborne_mass_after_by_class_kg', 'deposited_mass_by_class_kg',
                 'boundary_export_mass_by_class_kg', 'bed_change_by_class_kg', 'transfer_identity_gap_by_class_kg', 'reservoir_identity_gap_by_class_kg')
        fv = {name: host_vector(getattr(final, name), size=nc) for name in names}
        if any(not np.isfinite(v).all() for v in fv.values()):
            raise dw.DailyWindError('accounting', 'native finalizer has a nonfinite class vector')
        for prefix in ('transfer', 'reservoir'):
            bound = float(getattr(final, prefix + '_identity_bound_kg'))
            if not math.isfinite(bound) or bound < 0 or not np.all(np.abs(fv[prefix + '_identity_gap_by_class_kg']) <= bound):
                raise dw.DailyWindError('accounting', 'native finalizer class identity failed')
        mobile_before = host_vector(total.mobile_mass_before_by_class_kg, size=nc)
        mobile_last = host_vector(n.mobile_mass_by_class_kg(state.physical.mobile_state, nc), size=nc)
        if (np.any(mobile_before != 0) or np.any(fv['airborne_mass_after_by_class_kg'] != 0)
                or not np.array_equal(mobile_last, fv['airborne_mass_before_by_class_kg'])
                or not np.array_equal(mobile_last, host_vector(total.mobile_mass_after_by_class_kg, size=nc))):
            raise dw.DailyWindError('accounting', 'native airborne endpoint accounting differs')
        inflow = host_vector(total.boundary_input_by_class_kg, size=nc)
        export = host_vector(total.boundary_export_by_class_kg, size=nc) + fv['boundary_export_mass_by_class_kg']
        change = host_vector(total.bed_change_by_class_kg, size=nc) + fv['bed_change_by_class_kg']
        after = host_vector(n.bed_inventory(candidate), size=nc)
        inv_scale = max(float(np.abs(before).sum()), float(np.abs(after).sum()), float(inflow.sum()), float(export.sum()))
        inv_bound = sum(bounds) + float(final.reservoir_identity_bound_kg) + float(n.summation_error_bound_kg(2 * g.ny * g.nx * (nv + 1) + 6 * len(record.blocks), inv_scale))
        if not np.all(np.abs(after - before - inflow + export) <= inv_bound):
            raise dw.DailyWindError('accounting', 'whole native wind class inventories fail closure')
        event_scale = scales + sum(float(np.abs(v).sum()) for v in (fv['bed_change_by_class_kg'], fv['airborne_mass_after_by_class_kg'], mobile_before, fv['boundary_export_mass_by_class_kg']))
        event_bound = sum(bounds) + float(final.reservoir_identity_bound_kg) + float(n.summation_error_bound_kg(6 * len(record.blocks), event_scale))
        if not np.all(np.abs(change - inflow + export) <= event_bound):
            raise dw.DailyWindError('accounting', 'combined wind transport/finalizer identity failed')
        self._recheck()
        transfer = read_transfer_counters().delta(transfers)
        receipt = {'profile': PROFILE, 'executor_identity_sha256': self.identity_sha256, 'occurrence': occurrence.record(),
            'clocks': {'start_s': encode_time(start_s), 'end_s': encode_time(end), 'realized_elapsed_s': dw._fraction_text(Fraction(end) - Fraction(start_s))},
            'blocks': blocks, 'classes': {'boundary_input_hex': [v.hex() for v in inflow], 'boundary_export_hex': [v.hex() for v in export],
                'inventory_before_hex': [v.hex() for v in before], 'inventory_after_hex': [v.hex() for v in after],
                'inventory_bound_kg': inv_bound, 'event_bound_kg': event_bound},
            'checks': {'inventory_closure_with_boundary_export': True, 'native_block_checks': True, 'finalizer_checks': True},
            'wind_policy': {'soil_moisture_feedback_enabled': False, 'water_reset': False},
            'transfers': dataclasses.asdict(transfer), 'work_payload_device_bytes': work,
            'scope': 'bounded Dense GPU native wind; no water physics; scalar/class packets only in adapter'}
        raw = dw._json(receipt)
        if len(raw.encode()) > lim.max_receipt_bytes:
            raise dw.DailyWindError('limits', 'wind scalar receipt exceeds byte cap')
        return dw.FinalizedDailyWind(dw.DailyWindCandidate(candidate, end, raw), final)
