"""Thin native boundary publication for the Dense CUDA terrain wind profile.

MAPLE's wind finalizer preserves ``ledger_reconciliation_baseline`` by design.
Under the authored terrain case (``dry_aeolian_commit_separation_enabled``) the
accounting reference therefore survives a completed wind event, while the Dense
six-field ``BedState`` SYRUP receives has no slot for it and SYRUP's own handoff
commit cannot be handed a separate wind baseline. MAPLE's documented contract
for "the run is ending" is the PUBLIC ``force_commit_before_checkpoint``: one
actual commit at the UNCHANGED end clock is exactly the event that makes the
accounting and geometric references coincide again; afterwards an absent
(``None``) baseline IS the native implicit committed baseline
(``resolve_ledger_reconciliation_baseline``).

Only ordering, admission, native calls and audits live here. No wind, water,
avalanche or commit formula is implemented; every validator is MAPLE's own. The
old (non-terrain) CUDA profile never reaches this module: its identity hook in
``daily_wind_cuda`` publishes the finalized state unchanged and its guard is
unchanged.
"""
from __future__ import annotations

from masy import daily_wind as dw
from masy.clock import encode_time
from masy.gpu_contract import host_vector, leaves

BOUNDARY = 'masy.daily_triplet_wind.terrain_cuda.native_boundary.v1'
FORCE_COMMIT_API = 'maple.surface.topographic_commit.commit.force_commit_before_checkpoint'
SCOPE = ('thin native boundary publication: one public MAPLE force commit at the unchanged wind end clock; '
         'no wind resume, no combined restart, no water physics; scalar/class packets only in the adapter')


def _refuse(code, detail):
    return dw.DailyWindError(code, detail)


def _hexes(vector):
    return [float(v).hex() for v in vector]


def scratch_bytes(phys, config):
    """Explicit device admission for ONE native commit on the finalized state.

    ``commit_topography`` copies the voxel column and active layer twice (isolated
    candidate copies for the callback window plus its byte-for-byte snapshots) and
    clears the ledger into a fresh zero ledger in the state's namespace; the
    remaining scratch is per-cell / per-cell-class: the refreshed baseline and
    committed inventories, the surface diagnostics the water stage and refresh
    use, the avalanche transfer accounting (plus directed slopes when avalanching
    is enabled) and a possibly new availability partition. No host copy of any
    volume is made. Returns ``(total_bytes, composition)``.
    """
    voxel = int(phys.voxel_column.mass_kg.nbytes)
    active = int(phys.active_layer.mass_kg.nbytes)
    cell_class = active
    cell = 8 * int(config.geometry.ny) * int(config.geometry.nx)
    composition = {
        'commit_volume_copies': 2 * (voxel + active),
        'fresh_zero_ledger': sum(int(a.nbytes) for _, a in leaves(phys.ledger)),
        'baseline_and_committed_inventories': 4 * cell_class,
        'surface_and_water_depth_diagnostics': 8 * cell,
        'avalanche_transfer_and_slopes': 3 * cell_class + (16 * cell if bool(config.avalanching_spec.enabled) else 0),
        'availability_partition': 2 * cell_class,
    }
    return sum(composition.values()), composition


def publish_terrain_boundary(executor, n, config, bed, final, end, *, work, external_reserve_bytes):
    """Publish the finalized terrain wind state as a Dense boundary SYRUP can adopt.

    Baseline absent: the finalized state is already the implicit committed baseline;
    it is returned unchanged with a passthrough record. Baseline present: the public
    MAPLE force commit runs at ``end`` with the actual availability-aware avalanche
    callback (base/domain) and the actual constant-depth water callback (base); the
    returned voxels, active layer, ledger, committed topography, availability and
    water are adopted verbatim, the refreshed baseline is verified against the
    committed implicit baseline with native helpers, and only then is the
    process-private explicit baseline collapsed to ``None`` inside the republished
    native finalization record. Returns ``(final, record)``.
    """
    import dataclasses
    import numpy as np
    from maple.aeolian.scheduler.commit import water_stage_fixed_base_kwargs
    from maple.aeolian.scheduler.state_validation import validate_event_boundary, validate_scheduler_state
    from maple.core.backend import read_transfer_counters
    from maple.core.types.ledger_reconciliation import (LedgerReconciliationBaseline,
                                                        ledger_reconciliation_baseline_from_committed_state,
                                                        validate_ledger_reconciliation_baseline)
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple.coupling.sediment_ledger.diagnostics import global_pending_mass_by_class_kg
    from maple.surface.avalanche.callback import make_availability_aware_avalanche_callback
    from maple.surface.topographic_commit.commit import (CommitResult, force_commit_before_checkpoint,
                                                         is_already_committed, validate_reconciliation_context)
    from maple.surface.topographic_commit.height import check_committed_reference_consistent_with_fixed_base
    from maple.water.commit_callback import WaterCommitResult, make_water_depth_commit_callback

    policy, context, lim = executor.policy, executor._context, executor._limits
    cp = policy.gate()
    if type(final) is not n.WindEventFinalizationResult or not final.completed:
        raise _refuse('native_commit', 'a genuine completed native wind finalization is required')
    phys = final.state.physical
    g, nc = config.geometry, len(config.grain_classes.classes)
    shape = (g.ny, g.nx, nc)
    baseline = phys.ledger_reconciliation_baseline
    if baseline is None:
        return final, {'profile': BOUNDARY, 'status': 'implicit_committed_baseline', 'commit': None,
                       'clock_s': encode_time(end), 'scope': SCOPE}
    if type(baseline) is not LedgerReconciliationBaseline:
        raise _refuse('native_commit', 'the finalized state carries a foreign reconciliation baseline')
    if not n.mobile_state_is_genuinely_empty(phys.mobile_state) or float(phys.time_s).hex() != end.hex():
        raise _refuse('native_commit', 'boundary publication requires the finalized empty-air state at the end clock')
    if (config.fixed_base is not context.fixed_base or config.domain is not context.domain
            or not dw._same(config.geometry, context.geometry)
            or float(config.mass_resolution_kg).hex() != float(context.mass_resolution_kg).hex()):
        raise _refuse('native_commit', 'scheduler terrain/geometry differ from the admitted native context')
    spec = config.water_coupling_spec
    if (phys.water is None or bool(spec.enabled) or spec.depth_update_rule != 'constant_depth'
            or context.depth_update_rule != 'constant_depth' or not bool(config.external_water_carry_enabled)):
        raise _refuse('native_commit', 'the carried constant_depth water profile is required for a terrain boundary commit')
    try:
        if tuple(validate_ledger_reconciliation_baseline(baseline)) != shape:
            raise ValueError('reconciliation baseline shape differs from the admitted case')
        validate_reconciliation_context(phys.ledger, phys.voxel_column, phys.active_layer, g,
            committed_state=phys.committed_topography, reconciliation_baseline=baseline, current_time_s=end,
            operation=BOUNDARY, baseline_validated=True)
    except (ValueError, TypeError) as exc:
        raise _refuse('native_commit', dw._detail(exc))
    executor._validate_terrain(config, phys)

    # Memory: the commit's volume copies live inside the wind frontier already admitted
    # for this transaction (the pre-wind stage is released by the caller first); the
    # per-cell scratch is admitted explicitly against the actual readers now.
    nv = int(phys.voxel_column.mass_kg.shape[2])
    scratch, composition = scratch_bytes(phys, config)
    if type(work) is not int or type(external_reserve_bytes) is not int or scratch > work or scratch + external_reserve_bytes > lim.max_work_bytes:
        raise _refuse('limits', f'native boundary commit scratch exceeds the admitted wind frontier: '
                      f'scratch={scratch}, work={work}, external={external_reserve_bytes}, limit={lim.max_work_bytes}')
    policy.admit(64 * nc * 8 + (1 << 20), scratch, 'native terrain boundary commit', device_extra=external_reserve_bytes)
    pool = cp.get_default_memory_pool()
    used_before, total_before = int(pool.used_bytes()), int(pool.total_bytes())
    transfers = read_transfer_counters()
    before = host_vector(n.bed_inventory(n.BedState(*(getattr(phys, name) for name in dw._SHARED_FIELDS))), size=nc)

    base_kw = {} if config.fixed_base is None else {'fixed_base': config.fixed_base}
    domain_kw = {} if config.domain is None else {'domain': config.domain}
    try:
        avalanche, avalanche_box = make_availability_aware_avalanche_callback(
            config.avalanching_spec, phys.sediment_availability, config.resolved_initial_available_fraction(),
            **base_kw, **domain_kw)
        water_callback, water_box = make_water_depth_commit_callback(
            phys.water, g, committed_elevation_m(phys.committed_topography),
            depth_update_rule=spec.depth_update_rule, **water_stage_fixed_base_kwargs(config, path=BOUNDARY))
    except (ValueError, TypeError) as exc:
        raise _refuse('native_commit', dw._detail(exc))
    try:
        result = force_commit_before_checkpoint(
            phys.ledger, phys.committed_topography, phys.voxel_column, phys.active_layer, g,
            config.mass_resolution_kg, end, avalanche_callback=avalanche, water_callback=water_callback,
            reconciliation_baseline=baseline, **base_kw)
    except (ValueError, TypeError) as exc:
        water_box.discard()
        raise _refuse('native_commit', dw._detail(exc))
    except BaseException:
        water_box.discard()
        raise
    if type(result) is not CommitResult:
        water_box.discard()
        raise _refuse('native_commit', 'the public force commit returned a foreign result type')
    # The transaction returned, so every stage succeeded: promote the water stage's
    # pending result into the LOCAL box. Nothing is published to a bed until every
    # audit below has passed.
    water_box.publish()

    new_committed, new_ledger, new_baseline = result.new_committed_state, result.new_ledger, result.new_reconciliation_baseline
    availability = avalanche_box.new_sediment_availability
    try:
        if type(new_baseline) is not LedgerReconciliationBaseline:
            raise ValueError('native commit returned no refreshed reconciliation baseline')
        if tuple(validate_ledger_reconciliation_baseline(new_baseline)) != shape:
            raise ValueError('refreshed reconciliation baseline shape differs from the admitted case')
        implicit = ledger_reconciliation_baseline_from_committed_state(new_committed)
        for name in ('active_layer_mass_by_class_kg', 'voxel_mass_by_class_kg'):
            a, b = getattr(new_baseline, name), getattr(implicit, name)
            if (type(a) is not cp.ndarray or a.dtype != b.dtype or a.shape != b.shape
                    or not bool(cp.array_equal(a.view(cp.uint8), b.view(cp.uint8)))):
                raise ValueError(f'refreshed baseline {name} is not bitwise the committed inventory')
        if (float(new_baseline.last_reconciliation_time_s).hex() != end.hex()
                or float(implicit.last_reconciliation_time_s).hex() != end.hex()
                or float(new_committed.last_commit_time_s).hex() != end.hex()
                or int(new_committed.commit_count) != int(phys.committed_topography.commit_count) + 1
                or int(new_baseline.reconciliation_count) != int(baseline.reconciliation_count) + 1):
            raise ValueError('refreshed baseline/committed clocks or counters are not the native post-commit values')
        pending = host_vector(global_pending_mass_by_class_kg(new_ledger), size=nc)
        if np.any(pending != 0.0):
            raise ValueError('native ledger pending mass is not zero after the forced commit')
        if not is_already_committed(new_ledger, new_committed, result.new_voxel_column, result.new_active_layer, g,
                                    config.mass_resolution_kg, end, reconciliation_baseline=new_baseline):
            raise ValueError('the native already-committed predicate refuses the published state')
        validate_reconciliation_context(new_ledger, result.new_voxel_column, result.new_active_layer, g,
            committed_state=new_committed, reconciliation_baseline=new_baseline, current_time_s=end,
            operation=BOUNDARY, baseline_validated=True)
        if config.fixed_base is not None:
            check_committed_reference_consistent_with_fixed_base(new_committed, g, config.fixed_base,
                                                                 n_voxels=nv, operation=BOUNDARY)
        if availability is None:
            raise ValueError('the availability-aware avalanche callback returned no availability state')
        n.check_sediment_availability_state(availability, result.new_active_layer, config.mass_resolution_kg)
        update = water_box.depth_update
        depth, held = water_box.water.depth_m, phys.water.depth_m
        if (type(water_box) is not WaterCommitResult or water_box.ran is not True or update is None
                or update.rule != 'constant_depth' or float(update.displaced_volume_m3) != 0.0
                or float(update.clamp_added_volume_m3) != 0.0 or int(update.clamp_cell_count) != 0
                or water_box.water.mobile_mass_by_cell_class_kg is not phys.water.mobile_mass_by_cell_class_kg
                or type(depth) is not cp.ndarray or depth.dtype != held.dtype or depth.shape != held.shape
                or not bool(cp.array_equal(depth.view(cp.uint8), held.view(cp.uint8)))):
            raise ValueError('the constant_depth water stage did not carry the surface water unchanged')
    except (ValueError, TypeError) as exc:
        raise _refuse('native_commit', dw._detail(exc))

    published = n.BedState(result.new_voxel_column, result.new_active_layer, water_box.water, new_ledger, new_committed, availability)
    after = host_vector(n.bed_inventory(published), size=nc)
    net = host_vector(result.avalanche_diagnostics['global_net_by_class_kg'], size=nc)
    scale = max(float(np.abs(before).sum()), float(np.abs(after).sum()))
    bound = float(n.summation_error_bound_kg(2 * g.ny * g.nx * (nv + 1), scale))
    if not np.isfinite(after).all() or not np.isfinite(net).all() or not np.all(np.abs(after - before - net) <= bound):
        raise _refuse('accounting', 'native commit changed the whole-bed class inventories beyond its native bound')

    new_phys = dataclasses.replace(phys, voxel_column=result.new_voxel_column, active_layer=result.new_active_layer,
        ledger=new_ledger, committed_topography=new_committed, sediment_availability=availability,
        water=water_box.water, ledger_reconciliation_baseline=None)
    new_state = dataclasses.replace(final.state, physical=new_phys)
    try:
        validate_event_boundary(new_phys, new_state.coupled, fluvial=True, geometry=g, grain_classes=config.grain_classes)
        validate_scheduler_state(new_state, nc, config=config)
    except (ValueError, TypeError) as exc:
        raise _refuse('native_commit', dw._detail(exc))
    executor._validate_terrain(config, new_phys)
    new_final = dataclasses.replace(final, state=new_state)

    used_after, total_after = int(pool.used_bytes()), int(pool.total_bytes())
    delta = read_transfer_counters().delta(transfers)
    record = {
        'profile': BOUNDARY, 'status': 'native_force_commit', 'api': FORCE_COMMIT_API, 'clock_s': encode_time(end),
        'baseline_before': {'last_reconciliation_time_s': encode_time(float(baseline.last_reconciliation_time_s)),
                            'reconciliation_count': int(baseline.reconciliation_count)},
        'baseline_after': {'last_reconciliation_time_s': encode_time(float(new_baseline.last_reconciliation_time_s)),
                           'reconciliation_count': int(new_baseline.reconciliation_count),
                           'bitwise_committed_inventory': True,
                           'collapsed_to': 'None (native implicit committed baseline)'},
        'committed': {'commit_count_before': int(phys.committed_topography.commit_count),
                      'commit_count_after': int(new_committed.commit_count), 'last_commit_time_s': encode_time(end)},
        'commit': {'stage_log': list(result.commit_stage_log), 'avalanche_applied': bool(result.avalanche_applied),
                   'pre_commit_pending_mass_hex': _hexes(host_vector(result.pre_commit_pending_mass_by_class_kg, size=nc)),
                   'avalanche_gross_moved_hex': _hexes(host_vector(result.avalanche_diagnostics['gross_moved_by_class_kg'], size=nc)),
                   'avalanche_global_net_hex': _hexes(net),
                   'committed_elevation_change_max_abs_m': float(cp.abs(result.committed_elevation_change_m).max()),
                   'inventory_before_hex': _hexes(before), 'inventory_after_hex': _hexes(after),
                   'inventory_bound_kg': bound, 'ledger_pending_zero': True, 'already_committed_predicate': True},
        'water': {'rule': update.rule, 'ran': True, 'displaced_volume_m3': float(update.displaced_volume_m3),
                  'clamp_added_volume_m3': float(update.clamp_added_volume_m3),
                  'clamp_cell_count': int(update.clamp_cell_count), 'depth_bitwise_preserved': True},
        'terrain': {'fixed_base': config.fixed_base is not None, 'domain': config.domain is not None,
                    'world_reference_checked': config.fixed_base is not None,
                    'dry_aeolian_commit_separation_enabled': bool(config.topographic_commit_spec.dry_aeolian_commit_separation_enabled),
                    'avalanching_enabled': bool(config.avalanching_spec.enabled)},
        'memory': {'admitted_scratch_device_bytes': scratch, 'composition': composition, 'within_wind_work_bytes': work,
                   'pool_used_bytes_before': used_before, 'pool_used_bytes_after': used_after,
                   'pool_total_bytes_before': total_before, 'pool_total_bytes_after': total_after,
                   'transfers': dataclasses.asdict(delta)},
        'scope': SCOPE,
    }
    return new_final, record
