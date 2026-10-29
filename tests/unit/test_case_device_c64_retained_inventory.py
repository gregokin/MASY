"""Read-only retained metadata checks; no transport or execution authority."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from masy import case_device_c64 as producer


def _genuine_k1_limits():
    """Genuine Native engineering metadata, without constructing a bed/event."""
    from maple.aeolian.flux.windowed_step_device import DeviceWindowedK1DimensionStepLimits
    from maple.surface.active_layer.bed_exchange_device import (
        DeviceBedExchangeLimits, DeviceK1DimensionImpactLimits, DeviceWindowedK1DimensionProfile,
    )
    from maple.surface.voxels.windowed_delta import WindowedDeltaBudgets

    profile = DeviceWindowedK1DimensionProfile(8, 8, 20, 3, 1, 8)
    common = dict(max_operations=2, max_gather_columns=4, max_journal_records=320,
        max_cells=64, max_classes=1, max_voxels=20, max_impact_rows=256,
        max_new_device_bytes=64 << 20, max_host_staging_bytes=64 << 20,
        max_logical_deep_read_bytes=4 << 20, max_mapped_bytes=4 << 20,
        device_reserve_bytes=256 << 20, host_reserve_bytes=64 << 20,
        device_safety_fraction=0.5)
    nonimpact = DeviceBedExchangeLimits(**common)
    impact = DeviceK1DimensionImpactLimits(profile=profile, **dict(common, max_impact_rows=512))
    delta = WindowedDeltaBudgets(max_gathered_cells=64, max_gather_columns=4,
        max_journal_records=320, max_deep_read_bytes=2*64*20*8,
        max_gather_temp_bytes=1 << 20, max_resident_temp_bytes=4 << 20)
    return DeviceWindowedK1DimensionStepLimits(profile=profile, bed_limits=impact,
        creep_limits=nonimpact, delta_budgets=delta, max_mobile_slots=4000,
        max_state_leaves=512, max_config_leaves=512, max_leaf_bytes=8 << 20,
        max_snapshot_device_bytes=64 << 20, max_snapshot_host_bytes=16 << 20,
        transport_body_device_bytes=512 << 20, transport_body_host_bytes=256 << 20,
        boundary_host_bytes=16 << 20, max_step_h2d_bytes=16 << 20,
        max_step_d2h_bytes=16 << 20, reserve_host_bytes=64 << 20,
        device_reserve_bytes=256 << 20, device_safety_fraction=0.5)


def test_genuine_k1_shared_profile_is_preserved_without_reconstruction():
    from maple.core.backend.arrays import array_leaf_inventory
    from maple.surface.active_layer.bed_exchange import BedExchangeAdmissionError

    limits = _genuine_k1_limits()
    payload = np.empty(3, dtype=np.float64)
    before = {field.name: getattr(limits, field.name) for field in dataclasses.fields(limits)}
    profile = limits.profile
    assert limits.bed_limits.profile is profile
    # The old placement walk reconstructs the two profile occurrences as
    # different objects and truthfully triggers the unchanged Native guard.
    with pytest.raises(BedExchangeAdmissionError, match="same exact K1 impact profile/limits required"):
        array_leaf_inventory((limits, payload))
    assert producer._checkpoint_retained_array_bytes((limits, payload)) == payload.nbytes
    assert all(getattr(limits, name) is value for name, value in before.items())
    assert limits.profile is limits.bed_limits.profile is profile
    limits.__post_init__()


def test_repeated_container_and_array_occurrences_remain_positive():
    from maple.core.backend.arrays import array_leaf_inventory

    payload = np.empty(6, dtype=np.float64)
    nested = (payload, {"payload": payload})
    forest = (nested, nested, {"alias": nested}, [payload, payload])
    original = array_leaf_inventory(forest)
    assert original["n_leaves"] == 8 and original["n_unique_leaves"] == 1
    assert producer._checkpoint_retained_array_bytes(forest) == original["transfer_bytes"] == 8*payload.nbytes
    assert forest[0] is forest[1] is forest[2]["alias"] is nested
    assert all(value is payload for value in forest[3])


def test_read_only_dataclass_walk_skips_init_false_without_constructor_calls():
    calls = []

    @dataclasses.dataclass(frozen=True, slots=True)
    class MetadataHolder:
        payload: object
        derived: object = dataclasses.field(init=False, default=None)

        def __post_init__(self):
            calls.append(id(self))

    payload, derived = np.empty(5, dtype=np.float64), np.empty(17, dtype=np.int64)
    holder = MetadataHolder(payload)
    object.__setattr__(holder, "derived", derived)
    before = tuple(calls)
    assert producer._checkpoint_retained_array_bytes((holder, holder)) == 2*payload.nbytes
    assert tuple(calls) == before and before == (id(holder),)
    assert holder.payload is payload and holder.derived is derived


def test_cyclic_retained_container_is_refused():
    loop = []
    payload = np.empty(2, dtype=np.float64)
    branch = (payload, loop)
    loop.append(branch)
    with pytest.raises(producer.CaseDeviceC64Error, match="checkpoint_handoff: cyclic retained holder"):
        producer._checkpoint_retained_array_bytes(branch)
    assert loop[0] is branch and branch[0] is payload and branch[1] is loop
