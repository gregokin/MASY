"""Case-driven, dense CPU event loop over actual MAPLE and SYRUP operations.

The old tiny/production profiles are separate. This lane has explicit process
and copy/work quotas, not an arbitrary-size or full-heap admission theorem.
"""
from __future__ import annotations

import dataclasses
from fractions import Fraction
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import resource
import sys
import sysconfig
from typing import Any

from masy import dense_bed as _bed, dense_runner as _dr
from masy.case_schedule import CaseRainGroup, CaseSchedule, CaseScheduleError, CaseWindGroup, WindDeclaration, integer, number

POLICY_SCHEMA = "masy.case_cpu_process.v1"


@dataclasses.dataclass(frozen=True)
class CaseCPUProcessPolicy:
    address_space_bytes: int
    cpu_soft_s: int
    cpu_hard_s: int
    wall_s: float
    max_output_bytes: int
    max_run_bytes: int
    max_request_bytes: int
    max_input_entries: int
    max_path_chars: int
    max_water_input_file_bytes: int
    max_water_input_bytes: int
    max_report_bytes: int

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name == "wall_s":
                if type(value) is not float:
                    raise _dr.DenseRunnerError("invalid_case_policy", "wall_s must be a finite plain float")
                number(value, "wall_s", positive=True)
            else:
                integer(value, field.name)
        if self.address_space_bytes != 4 << 30 or self.cpu_hard_s < self.cpu_soft_s:
            raise _dr.DenseRunnerError("invalid_case_policy", "requires4GiB AS and CPU hard>=soft")
        if self.max_request_bytes > 1 << 20:
            raise _dr.DenseRunnerError("invalid_case_policy", "original bootstrap maximum request is1MiB")
        if self.max_report_bytes > self.max_output_bytes or self.max_output_bytes > self.max_run_bytes:
            raise _dr.DenseRunnerError("invalid_case_policy", "report<=output<=whole-run byte quotas required")

    @property
    def file_size_bytes(self):
        # A per-file upper derived from the explicitly declared TOTAL quota.
        # No runtime source-file budget is mislabelled as a Native output law.
        return self.max_run_bytes

    def as_record(self):
        return {"schema": POLICY_SCHEMA, **dataclasses.asdict(self),
                "derived_file_size_bytes": self.file_size_bytes, "core_bytes": 0,
                "scope": "case_dimensions_explicit_dense_numpy_cpu", "native_peak_estimate": None}


def apply_process_policy(policy: CaseCPUProcessPolicy) -> None:
    """Called in the dedicated cold child BEFORE any Native/provider import."""
    if type(policy) is not CaseCPUProcessPolicy or sys.platform != "linux":
        raise _dr.DenseRunnerError("unsupported_case_process", "exact Linux case policy required")
    policy.__post_init__()
    wanted = {resource.RLIMIT_AS: (policy.address_space_bytes,) * 2,
              resource.RLIMIT_CPU: (policy.cpu_soft_s, policy.cpu_hard_s),
              resource.RLIMIT_FSIZE: (policy.file_size_bytes,) * 2, resource.RLIMIT_CORE: (0, 0)}
    # Refuse all incompatible inherited hard ceilings BEFORE changing any.
    for key, pair in wanted.items():
        hard = resource.getrlimit(key)[1]
        if hard != resource.RLIM_INFINITY and pair[1] > hard:
            raise _dr.DenseRunnerError("inherited_case_limit", "inherited hard ceiling cannot meet declared policy")
    for key, pair in wanted.items():
        resource.setrlimit(key, pair)


def check_case_process(policy: CaseCPUProcessPolicy, *, identity=None, environment=None, upcoming=0, checkpoint="entry"):
    from masy.dense_production import process_identity
    if type(policy) is not CaseCPUProcessPolicy or sys.platform != "linux":
        raise _dr.DenseRunnerError("unsupported_case_process", "exact Linux case policy required")
    policy.__post_init__()
    integer(upcoming, "upcoming", minimum=0)
    wanted = {"RLIMIT_AS": [policy.address_space_bytes] * 2,
              "RLIMIT_CPU": [policy.cpu_soft_s, policy.cpu_hard_s], "RLIMIT_CORE": [0, 0],
              "RLIMIT_FSIZE": [policy.file_size_bytes] * 2}
    observed = {name: list(resource.getrlimit(getattr(resource, name))) for name in wanted}
    if observed != wanted:
        raise _dr.DenseRunnerError("case_process_limits", "actual OS limits differ from declared case policy")
    actual = process_identity()
    if identity is not None and actual != identity:
        raise _dr.DenseRunnerError("case_process_identity", "owned PID/start/boot changed")
    for name, value in (("NUMBA_DISABLE_CUDA", "1"), ("CUDA_VISIBLE_DEVICES", ""), ("JAX_PLATFORMS", "cpu")):
        if os.environ.get(name) != value:
            raise _dr.DenseRunnerError("case_cpu_environment", "original CPU-only runtime environment required")
    environment_receipt = None if environment is None else native_environment_record(environment)
    virtual = int(Path('/proc/self/statm').read_text().split()[0]) * os.sysconf('SC_PAGE_SIZE')
    if virtual >= policy.address_space_bytes or upcoming > policy.address_space_bytes - virtual:
        raise _dr.DenseRunnerError("case_address_space_headroom", "current mappings plus known upcoming copies cannot fit")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"checkpoint": checkpoint, "identity": actual, "limits": observed,
            "virtual_bytes": virtual, "known_upcoming_bytes": upcoming,
            "self_cpu_s": float(usage.ru_utime + usage.ru_stime), "native_peak_estimate": None,
            "environment": environment_receipt}


def _array_bytes(value):
    # Complete owning ndarray backings per occurrence, without alias credit.
    # Unknown external buffers refuse rather than equating view.nbytes to R.
    total = 0
    for _, array in _bed._require_host_arrays(value, "case owners"):
        root = array
        while root.base is not None:
            if type(root.base) is not type(array):
                raise _dr.DenseRunnerError('unsupported_case_backing', 'non-ndarray owning buffer requires its own size law')
            root = root.base
        total += int(root.nbytes)
    return total


def _case_native():
    from maple.aeolian.scheduler.accounting import validate_block_accounting
    from maple.aeolian.scheduler.block import WindBlockResult, check_block_conservation
    from maple.aeolian.scheduler.record import BlockAccountingAccumulator
    from maple.core.parameters import validate_config
    from maple.core.parameters.wind_forcing import validate_wind_forcing_record
    from maple.core.types.cohort import zeros_mobile_cohort_state
    from maple.experiment.scheduler_adapter import build_scheduler_physics_config
    return locals()


class CaseRunner(_dr.DenseRunner):
    def __init__(self, adopted, prepared_water, schedule_input, *, limits: _dr.DenseRunnerLimits,
                 process_policy: CaseCPUProcessPolicy):
        if type(self) is not CaseRunner:
            raise _dr.DenseRunnerError("invalid_case_runner", "case runner subclasses are not an input profile")
        process = check_case_process(process_policy, checkpoint="construction_before_native_import")
        from masy.case_workflow import AdoptedCase
        from masy.case_water import PreparedCaseWater
        if type(adopted) is not AdoptedCase or type(prepared_water) is not PreparedCaseWater or type(limits) is not _dr.DenseRunnerLimits:
            raise _dr.DenseRunnerError("invalid_case_input", "exact adopted/prepared/limits types required")
        limits.__post_init__()
        bed = adopted.bed
        if type(bed) is not _bed.DenseBed or hasattr(bed, '_dynamic'):
            raise _dr.DenseRunnerError("unsupported_case_bed", "prescribed dense case bed required")
        adopted.inputs.verify(); bed.verify_case_inputs(); prepared_water.verify(bed)
        self._case_policy = process_policy
        self._case_worker_identity = process['identity']
        self._case_environment = dict(os.environ)
        self._case_adopted, self._case_water = adopted, prepared_water
        self._production = None  # old production PROFILE is never used as authority
        n, native = _dr._native(), _case_native()
        cfg = adopted.inputs.resolved.config
        if cfg.execution.backend != 'numpy' or cfg.particle_transport.mobile_state_representation != 'flat':
            raise _dr.DenseRunnerError("unsupported_case_backend", "first case lane requires NumPy and flat cohorts")
        if cfg.aeolian_process_mode.mode != 'detachment_transport_deposition' or cfg.detachment_source.mode not in ('aerodynamic', 'reduced'):
            raise _dr.DenseRunnerError("unsupported_case_wind", "authored aerodynamic or reduced baseline wind required")
        if cfg.topographic_commit.dry_aeolian_commit_separation_enabled:
            raise _dr.DenseRunnerError("unsupported_case_baseline", "authored dry separation creates a nonNone baseline; this case carrier refuses it")
        seed = adopted.initial_mobile_state
        if seed is not None and not n.mobile_state_is_genuinely_empty(seed):
            raise _dr.DenseRunnerError("unsupported_initial_seed", "nonempty Native seed is retained by adoption but not yet qualified for this runner")
        effective = dataclasses.replace(cfg, water_coupling=dataclasses.replace(cfg.water_coupling, enabled=False, depth_update_rule='constant_depth'))
        native['validate_config'](effective)
        record = adopted.inputs.resolved.wind_record
        native['validate_wind_forcing_record'](record)
        if not record.blocks:
            raise _dr.DenseRunnerError("empty_case_wind", "a complete positive Native wind record is required")
        # Source cohort.py212–236: ten float64, four int32, one bool per slot.
        pool_bytes = 97 * cfg.particle_transport.max_active_cohorts
        mapper_inputs = _array_bytes((cfg, adopted.inputs.resolved.prescribed_external_shear))
        self._case_constructor_bytes = 3 * pool_bytes + 2 * mapper_inputs + len(_bed._canonical_json(adopted.copy_plan).encode())
        if self._case_constructor_bytes > limits.max_work_bytes:
            raise _dr.DenseRunnerError("case_constructor_cap", "known pool/current-next mapper owners exceed work quota")
        check_case_process(process_policy, identity=self._case_worker_identity, environment=self._case_environment,
                           upcoming=self._case_constructor_bytes, checkpoint='before_mapper_pool')
        wind_cfg = native['build_scheduler_physics_config'](effective,
            spatial_block_diagnostics_enabled=adopted.inputs.resolved.requested_diagnostics.spatial_snapshots,
            initial_available_fraction=bed.context.initial_available_fraction,
            prescribed_external_shear=adopted.inputs.resolved.prescribed_external_shear, wind_record=record)
        for name in ('geometry', 'grain_classes', 'avalanching_spec', 'topographic_commit_spec', 'initial_available_fraction'):
            if getattr(wind_cfg, name) is not getattr(bed.context, name):
                raise _dr.DenseRunnerError('case_context_mismatch', name)
        if wind_cfg.mass_resolution_kg != bed.context.mass_resolution_kg:
            raise _dr.DenseRunnerError('case_context_mismatch', 'mass resolution')
        pool = native['zeros_mobile_cohort_state'](cfg.particle_transport.max_active_cohorts)
        declarations = tuple(WindDeclaration(i, float(b.shear_velocity_m_s), float(b.direction_deg), float(b.duration_s))
                             for i, b in enumerate(record.blocks))
        schedule = CaseSchedule.from_input(schedule_input, blocks=declarations, case_sha256=adopted.inputs.identity_sha256,
            wind_record_sha256=_bed._sha256(_dr._canonical_input(n.np, record, 'original_record')),
            water_profile_id=prepared_water.profile_id, water_identity_sha256=prepared_water.identity_sha256,
            max_events=limits.max_events)
        self._case_effective = effective
        self._case_record = record
        self._case_mapping = {'authored_water': _dr._canonical_input(n.np, cfg.water_coupling, 'authored.water'),
                              'effective_water': _dr._canonical_input(n.np, effective.water_coupling, 'effective.water'),
                              'changes': ['water_coupling.enabled=False', 'water_coupling.depth_update_rule=constant_depth'],
                              'dimensions': {'cells': cfg.geometry.ny * cfg.geometry.nx, 'classes': len(cfg.grain_classes.classes),
                                             'voxels': bed.case_record['n_voxels'], 'cohort_slots': cfg.particle_transport.max_active_cohorts}}
        wind, water, control = _dr.DenseWindInputs(wind_cfg, pool), prepared_water.water, prepared_water.control
        if 'record_origin_s' not in inspect.signature(n.finalize_wind_event).parameters:
            raise _dr.DenseRunnerError('unsupported_native_finalizer', 'explicit origin required')
        soil = self._check_water(n, bed, water, control, limits)
        _dr._initialize_common(self, n, bed, schedule, wind, water, control, limits, soil)
        self._static_record['profile'] = {'scope': 'case_dimensions_explicit_dense_numpy_cpu',
                                          'calibration': 'case_authored_not_qualified_by_synthetic_tests'}
        self._static_record['case_process_policy'] = process_policy.as_record()
        self._static_sha256 = _bed._sha256(self._static_record)

    def _combined_schedule_identity(self, bed_sha256, schedule):
        if type(schedule) is not CaseSchedule:
            raise _dr.DenseRunnerError('invalid_case_schedule', 'exact case schedule required')
        return schedule.combined_digest(bed_sha256)

    def _static_parts_record(self, n):
        self._case_adopted.inputs.verify(); self._case_water.verify(self._bed)
        record = super()._static_parts_record(n)
        record['case_workflow'] = {'original_case': self._case_adopted.inputs.identity_sha256,
                                  'prepared_water': self._case_water.identity_sha256,
                                  'mapping': self._case_mapping, 'policy': self._case_policy.as_record()}
        return record

    def _production_process_check(self, upcoming_wrapper_bytes=0, *, checkpoint='event_entry', elapsed=None, evidence=None):
        try:
            return check_case_process(self._case_policy, identity=self._case_worker_identity,
                environment=self._case_environment, upcoming=upcoming_wrapper_bytes, checkpoint=checkpoint)
        except (_dr.DenseRunnerError, MemoryError) as error:
            raise _dr._Stop('refused', getattr(error, 'code', 'address_space_exhausted'), str(error),
                            exception=error, elapsed=elapsed, evidence=evidence) from error

    def step(self):
        index, event, start = _dr._step_start(self)
        try:
            self._precheck()
            resolved = self._schedule.resolve(index, start)
            self._schedule.check_resolved(resolved)
            if type(event) is CaseWindGroup:
                return self._case_wind(index, event, resolved)
            if type(event) is not CaseRainGroup:
                raise _dr.DenseRunnerError('invalid_case_event', 'foreign event')
            return self._rain_event(index, event.event, resolved)
        except _dr._Stop as stop:
            return _dr._step_stopped(self, index, event, start, stop)
        except Exception as error:
            return _dr._step_failed(self, index, event, start, error)

    def _case_wind(self, index, event, resolved):
        n, native = _dr._native(), _case_native()
        np, cfg, own = n.np, self._wind.config, self._ownership()
        ops, t = self._array_ops(np), self._bed.time_s
        from masy.dense_production import require_dry_wind_water
        require_dry_wind_water(np, self._bed.state.water)
        local_record = n.resolve_wind_forcing_record({'blocks': [
            {'shear_velocity_m_s': b.shear_velocity_m_s, 'direction_deg': b.direction_deg, 'duration_s': b.duration_s}
            for b in event.blocks]})
        native['validate_wind_forcing_record'](local_record)
        plan, requested, covered, rounding, uppers = [], Fraction(0), Fraction(0), Fraction(0), []
        for b in event.blocks:
            q = Fraction(b.duration_s) / Fraction(cfg.outer_timestep_s)
            upper = -(-q.numerator // q.denominator) + 1
            uppers.append(upper)
        upper = sum(uppers)
        if upper > self._limits.max_wind_accounting_steps:
            raise _dr._Stop('refused', 'wind_step_limit', 'group accounting-step bound exceeds declared quota')
        s, p = own.stage_bytes(), own.pool_bytes(self._pool)
        base = 4 * s + 3 * p + 8 * self._inventory_elements * self._n_classes + 32 * upper
        scalar_bytes = (len(json.dumps(self._schedule.as_record(), sort_keys=True).encode())
                        + self._limits.max_receipt_bytes)
        self._work_bytes_check(base + self._case_constructor_bytes + 3 * scalar_bytes)
        for b, maximum in zip(event.blocks, uppers, strict=True):
            pieces = n.accounting_step_durations_s(b.duration_s, cfg.outer_timestep_s)
            want = Fraction(b.duration_s); got = sum((Fraction(dt) for dt in pieces), Fraction(0))
            bound = Fraction(2, 2 ** 53) * max(want, got)
            if len(pieces) > maximum or got <= 0 or abs(got - want) > bound:
                raise _dr._Stop('refused', 'unsupported_wind_duration', 'Native split drops/extends an authored interval')
            plan.extend(pieces); requested += want; covered += got; rounding += bound
        stage = self._stage(); pool = own.work_pool(self._pool)
        state = own.scheduler(stage, pool, t)
        initial = ops.inventory(n, own.stage_state(stage))
        accumulator = native['BlockAccountingAccumulator'](self._n_classes)
        summaries, result, block = [], None, None
        last_actual_end = t
        try:
            for local_index, (b, maximum) in enumerate(zip(event.blocks, uppers, strict=True)):
                if state.block_index != local_index or state.block_offset_s != 0.0:
                    raise _dr._Stop('incomplete', 'case_cursor_mismatch', 'genuine local cursor does not match block')
                current = float(state.physical.time_s)
                block = n.WindBlock(b.shear_velocity_m_s, b.direction_deg, b.duration_s, current, current + b.duration_s)
                retained = 0 if accumulator.n_blocks == 0 else _array_bytes(accumulator.total)
                self._work_bytes_check(base + self._case_constructor_bytes + 4 * retained + 3 * scalar_bytes)
                try:
                    result = _dr._advance_wind_block(state, block, cfg)
                except Exception as error:
                    raise _dr._Stop('native_refused' if isinstance(error, ValueError) else 'native_error',
                        'wind_native_failure', _dr._detail(error), exception=error,
                        evidence={'call_input_state': state, 'known_completed_elapsed_s': _dr._elapsed(t, current), 'completed_blocks': summaries,
                                  'accounting': None if accumulator.n_blocks == 0 else accumulator.total}) from error
                if type(result) is not native['WindBlockResult']:
                    raise _dr._Stop('incomplete', 'invalid_wind_result', 'unexpected Native block result', evidence=result)
                a, physical = result.accounting, result.state.physical
                end = float(physical.time_s)
                if not math.isfinite(end) or end < 0 or math.copysign(1.0, end) < 0:
                    raise _dr._Stop('incomplete', 'invalid_wind_clock', 'Native returned an invalid actual clock', evidence=result)
                last_actual_end = end
                def incomplete(code, detail):
                    return _dr._Stop('incomplete', code, detail, elapsed=_dr._elapsed(t, end), evidence=result)
                if (result.process_mode != cfg.aeolian_process_mode.mode or result.integration_path != 'baseline_direct'
                        or result.block_index != local_index or result.state.block_index != local_index):
                    raise incomplete('unsupported_wind_path', 'wrong Native path/cursor')
                if any(ops.scalar(getattr(result, name)).hex() != ops.scalar(getattr(block, name)).hex()
                       for name in ('start_time_s', 'end_time_s', 'duration_s', 'shear_velocity_m_s', 'direction_deg')):
                    raise incomplete('wind_block_changed', 'Native forcing/bounds differ from actual supplied block')
                # Native block._prepare_aeolian_wind_block derives these
                # exact metadata operands from the genuine zero-offset state.
                # The physical endpoint remains Native's actual returned clock;
                # its discrepancy is recorded, never snapped to a forcing edge.
                if (ops.scalar(result.advanced_start_time_s).hex() != current.hex()
                        or ops.scalar(result.advanced_duration_s).hex() != b.duration_s.hex()
                        or ops.scalar(result.advanced_end_time_s).hex() != (current + b.duration_s).hex()
                        or ops.scalar(result.state.block_offset_s).hex() != (end - current).hex()
                        or result.state.coupled is not None or result.n_accounting_steps != a.n_accounting_steps):
                    raise incomplete('wind_interval_changed', 'actual Native interval/cursor metadata is inconsistent')
                violations = native['validate_block_accounting'](a, self._n_classes,
                    cfg.impact_rebound_splash_spec.impact_state_bins.n_impact_state_bins(), ny=cfg.geometry.ny,
                    nx=cfg.geometry.nx, spatial_enabled=cfg.spatial_block_diagnostics_enabled, coupled_mode=False)
                if violations:
                    raise incomplete('invalid_wind_accounting', '; '.join(violations)[:4000])
                conserved, residual = native['check_block_conservation'](a, final_physical=physical)
                if not conserved or result.conservation_ok is not True:
                    raise incomplete('wind_conservation_failed', 'genuine Native per-block conservation failed')
                if a.n_accounting_steps > maximum or end <= current:
                    raise incomplete('wind_step_limit', 'Native block exceeded step quota or did not advance')
                if any(getattr(a, 'water_' + key, None) is not None and ops.any_nonzero(getattr(a, 'water_' + key))
                       for key in ('mobile_mass_before_by_class_kg', 'mobile_mass_after_by_class_kg',
                                   'boundary_input_by_class_kg', 'boundary_export_by_class_kg')):
                    raise incomplete('wind_water_pass', 'wind reported water transport')
                if ops.any_nonzero(physical.water.mobile_mass_by_cell_class_kg):
                    raise incomplete('water_mobile_not_empty', 'water load appeared during wind')
                bed_change = ops.class_vector(a.bed_change_by_class_kg)
                delta = ops.class_vector(a.mobile_mass_after_by_class_kg) - ops.class_vector(a.mobile_mass_before_by_class_kg)
                inflow, export = ops.class_vector(a.boundary_input_by_class_kg), ops.class_vector(a.boundary_export_by_class_kg)
                class_gap = bed_change + delta - inflow + export
                bound = n.summation_error_bound_kg(max(a.n_accounting_steps, 1), n.reservoir_arithmetic_scale_kg(a, physical))
                if not bool(np.all(np.abs(class_gap) <= bound)):
                    raise incomplete('wind_budget_open', 'original per-class reservoir check failed')
                if result.state.block_offset_s < b.duration_s - n.record_end_epsilon_s:
                    raise incomplete('wind_record_incomplete', 'Native block was not consumed')
                self._work_bytes_check(base + self._case_constructor_bytes + 4 * (_array_bytes(a) + retained) + 3 * scalar_bytes)
                accumulator.add(a)
                state = dataclasses.replace(result.state, block_index=local_index + 1, block_offset_s=0.0)
                summaries.append({'original_index': b.index, 'local_index': local_index,
                    'native_start': float(result.start_time_s).hex(), 'native_end': float(result.end_time_s).hex(),
                    'actual_end': end.hex(), 'advanced_duration': float(result.advanced_duration_s).hex(),
                    'n_accounting_steps': int(a.n_accounting_steps), 'n_commits': int(result.n_commits_this_block),
                    'native_conservation_residual_kg': float(residual)})
                if len(json.dumps(summaries, allow_nan=False).encode()) > self._limits.max_receipt_bytes:
                    raise incomplete('case_summary_cap', 'bounded scalar block summaries exceed the receipt quota')
            if accumulator.n_blocks != len(event.blocks) or accumulator.n_blocks < 1:
                raise _dr._Stop('incomplete', 'case_accumulator_count', 'not every declared block was folded')
            a = accumulator.total
            physical, t_end = state.physical, float(state.physical.time_s)
            conserved, _ = native['check_block_conservation'](a, final_physical=physical)
            if not conserved:
                raise _dr._Stop('incomplete', 'wind_aggregate_conservation_failed', 'Native aggregate refused',
                    elapsed=_dr._elapsed(t, t_end), evidence={'transport': result, 'accounting': a, 'completed_blocks': tuple(summaries)})
            bed_change = ops.class_vector(a.bed_change_by_class_kg)
            delta = ops.class_vector(a.mobile_mass_after_by_class_kg) - ops.class_vector(a.mobile_mass_before_by_class_kg)
            inflow, export = ops.class_vector(a.boundary_input_by_class_kg), ops.class_vector(a.boundary_export_by_class_kg)
            gap = bed_change + delta - inflow + export
            native_bound = float(n.summation_error_bound_kg(max(a.n_accounting_steps, 1), n.reservoir_arithmetic_scale_kg(a, physical)))
            if not bool(np.all(np.abs(gap) <= native_bound)):
                raise _dr._Stop('incomplete', 'wind_budget_open', 'aggregate per-class Native bound failed',
                    elapsed=_dr._elapsed(t, t_end), evidence={'transport': result, 'accounting': a, 'completed_blocks': tuple(summaries)})
            case_summary = {'event_index': index, 'schedule_sha256': self._schedule.digest,
                                      'blocks': summaries, 'n_blocks': accumulator.n_blocks,
                                      'original_range': [event.first_block, event.stop_block],
                                      'origin_s': t.hex(), 'actual_end_s': t_end.hex(),
                                      'n_commits': sum(row['n_commits'] for row in summaries),
                                      'terminal_native_field_scope': 'last_genuine_block_result',
                                      'public_record_validator': 'not applicable to absolute blocks; Native accounting/conservation and original MASY checks retained'}
            prepared = _dr._WindEntry(n, np, self._bed, cfg, ops, own, t, float(block.end_time_s),
                Fraction(event.duration_s) / Fraction(cfg.outer_timestep_s), upper, s, p, plan, requested, covered,
                rounding, stage, pool, state, block, initial)
            return _dr._wind_terminal(self, index, event, resolved, prepared, result, physical, a, t_end, _dr._elapsed(t, t_end),
                bed_change, inflow, export, gap, native_bound, state, local_record, case_summary=case_summary)
        except _dr._Stop as stop:
            if result is not None and stop.evidence is None:
                stop.evidence = {'last_returned_result': result, 'completed_blocks': tuple(summaries),
                    'accounting': None if accumulator.n_blocks == 0 else accumulator.total}
                if stop.elapsed is None: stop.elapsed = _dr._elapsed(t, last_actual_end)
            raise
        except Exception as error:
            evidence = None if result is None else {
                'last_returned_result': result, 'completed_blocks': tuple(summaries),
                'accounting': None if accumulator.n_blocks == 0 else accumulator.total}
            raise _dr._Stop('runner_error', 'address_space_exhausted' if isinstance(error, MemoryError) else 'runner_internal_error',
                _dr._detail(error), exception=error, elapsed=None if result is None else _dr._elapsed(t, last_actual_end),
                evidence=evidence) from error



def _environment_require(condition):
    if not condition:
        raise _dr.DenseRunnerError('case_environment_changed', 'unknown, changed or unbound provider environment delta')


_JAX_ENV_SOURCE = {'jax': {'relative': '__init__.py', 'size': 9024, 'sha256': 'e97031df0826c1ed1399652872281209a5c8d1d64302da6cf8ce158c0bb51f66'}, 'jax._src.cloud_tpu_init': {'relative': '_src/cloud_tpu_init.py', 'size': 6441, 'sha256': 'c3372d35eb53d0d762564d25dad6d211768cc50c55b620b6ed7b70e5ff5b5b13', 'function_firstlineno': 60}}


def native_environment_record(declared, *, initial=False):
    """Read only. Admit two exact Source-owned additions, never changed originals.

    This reports the current permitted Native import delta, not a causal
    execution trace or evidence about TPU hardware. It imports/queries no
    provider and never changes, removes, restores or hides environment values.
    """
    baseline, actual = dict(declared), dict(os.environ)
    _environment_require(all(type(key) is str and type(value) is str for key, value in baseline.items()))
    _environment_require(set(baseline) <= set(actual))
    _environment_require(all(actual[key] == value for key, value in baseline.items()))
    delta = {key: value for key, value in actual.items() if key not in baseline}
    allowed = {'TF_CPP_MIN_LOG_LEVEL': '1', 'TPU_SKIP_MDS_QUERY': '1'}
    _environment_require(set(delta) <= set(allowed))
    _environment_require(all(value == allowed[key] for key, value in delta.items()))
    if initial:
        _environment_require(not delta and actual == baseline)
    provenance = {}
    names = ('jax', 'jax._src.cloud_tpu_init') if 'TPU_SKIP_MDS_QUERY' in delta else ('jax',)
    if delta:
        from masy import runtime_worker
        provider_root = Path(sysconfig.get_path('purelib')) / 'jax'
        _environment_require(provider_root.resolve() == provider_root)
        for name in names:
            module = sys.modules.get(name)
            _environment_require(type(module) is type(sys) and sys.modules.get(name) is module)
            _environment_require(vars(module).get('__name__') == name)
            expected = provider_root / _JAX_ENV_SOURCE[name]['relative']
            _environment_require(Path(vars(module)['__file__']) == expected and expected.resolve() == expected)
            spec = vars(module).get('__spec__')
            _environment_require(spec is not None and spec.name == name and spec.origin == str(expected))
            cached = vars(module).get('__cached__')
            _environment_require(cached is None or not Path(cached).exists())
            digest, size = runtime_worker._sha256_file(str(expected), _JAX_ENV_SOURCE[name]['size'])
            _environment_require((size, digest) == (_JAX_ENV_SOURCE[name]['size'], _JAX_ENV_SOURCE[name]['sha256']))
            entry = {'path': str(expected), 'size': size, 'sha256': digest, 'module': name}
            if name == 'jax._src.cloud_tpu_init':
                function = vars(module).get('cloud_tpu_init')
                _environment_require(type(function) is type(native_environment_record))
                _environment_require(function.__globals__ is vars(module) and function.__module__ == name)
                _environment_require(function.__name__ == 'cloud_tpu_init')
                _environment_require(function.__code__.co_filename == str(expected))
                _environment_require(function.__code__.co_firstlineno == _JAX_ENV_SOURCE[name]['function_firstlineno'])
                entry['function'] = 'cloud_tpu_init'
                entry['firstlineno'] = function.__code__.co_firstlineno
            provenance[name] = entry
    _environment_require(actual == {**baseline, **delta})
    return {'initial': initial, 'baseline': baseline, 'Native_provider_delta': delta,
            'actual': actual, 'Source_owners': provenance,
            'scope': 'two current Source-owned JAX import additions; no backend/hardware query by this guard'}


