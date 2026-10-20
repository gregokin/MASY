"""Tiny authored component fixtures; actual native vegetation/clock APIs, no transport.

All replacement values are explicit canonical raster values, not response laws.
The complete task51 candidate supplies MASY; immutable851 supplies MAPLE/SYRUP.
"""

from __future__ import annotations

import dataclasses
import json
from fractions import Fraction
from unittest import mock

import numpy as np
import pytest
from maple.core.types.vegetation import (
    DEFAULT_VEGETATION_DRAG_COEFFICIENT,
    DEFAULT_VEGETATION_POROSITY,
    VegetationRasterState,
    VegetationState,
    empty_vegetation_state,
    uniform_vegetation_raster,
    validate_vegetation_state,
    vegetation_content_hash,
)
from masy import landscape_vegetation as v
from masy.clock import ClockLedger

SHAPE = (3, 4)
SCHEDULE = "a" * 64
KINDS = ("wind", "rain", "wind")


def cell(position=(0, 0), *, cover=0.5, height=0.25, frontal=0.5):
    return v.CellReplacement(
        position,
        cover,
        height,
        DEFAULT_VEGETATION_POROSITY,
        DEFAULT_VEGETATION_DRAG_COEFFICIENT,
        frontal,
    )


def delta(
    payload="wind0",
    *,
    proposer="wind",
    sequence=0,
    at=0,
    replacements=None,
    reason="prescribed_replacement",
):
    return v.PlannedDelta(
        proposer,
        sequence,
        payload,
        at,
        (cell(),) if replacements is None else replacements,
        reason,
    )


def plan(deltas=(), *, limits=None, kinds=KINDS, shape=SHAPE):
    return v.build_vegetation_plan(
        deltas,
        shape=shape,
        n_classes=1,
        n_voxels=6,
        event_kinds=kinds,
        schedule_sha256=SCHEDULE,
        limits=limits,
    )


def envelope(deltas=(), *, limits=None, kinds=KINDS, initial=None):
    return v.LandscapeVegetation.from_native(
        empty_vegetation_state(*SHAPE) if initial is None else initial,
        plan=plan(deltas, limits=limits, kinds=kinds),
    )


def ledger(ends=(8.0,), *, epoch=7.0, returned=None):
    result = ClockLedger(epoch)
    start = epoch
    for i, end in enumerate(ends):
        elapsed = end - start if returned is None else returned[i]
        result.append(
            "event",
            native_start=start,
            native_end=end,
            commit_before=start,
            commit_after=end,
            returned_elapsed=elapsed,
        )
        start = end
    return result


def bind(index=0, *, ends=None, kinds=KINDS, schedule=SCHEDULE):
    if ends is None:
        ends = tuple(8.0 + i for i in range(index + 1))
    return v.BoundaryBinding.from_ledger_record(
        ledger(ends).as_record(),
        event_index=index,
        event_kind=kinds[index],
        schedule_sha256=schedule,
    )


def prepare(env, binding=None, *, soil=None, version=None):
    return v.prepare_vegetation(
        env,
        bind() if binding is None else binding,
        resource=v.JointResourceCandidate(soil, vegetation_version_sha256=version),
    )


def native_bytes(env):
    # Tests deliberately inspect the private trusted borrow to verify authority,
    # never publish it as a safe user snapshot.
    state = env._native_borrow()
    return {
        name: (
            getattr(state.raster, name).tobytes(),
            getattr(state.raster, name).flags.writeable,
        )
        for name in v.RASTER_FIELDS
    }


def test_native_content_and_version_identities_are_distinct():
    env = envelope((delta(),))
    assert env.native_content_sha256 == vegetation_content_hash(env.state)
    assert env.version_sha256 != env.native_content_sha256
    prepared = prepare(env)
    out = prepared.envelope
    validate_vegetation_state(out.state, *SHAPE)
    assert out.native_content_sha256 == vegetation_content_hash(out.state)
    assert out.version == 1 and out.version_sha256 != out.native_content_sha256
    assert prepared.new_receipts[0]["base_version_sha256"] == env.version_sha256
    assert prepared.new_receipts[0]["version_sha256"] == out.version_sha256
    assert prepared.new_receipts[0]["source_transition"] == "none->raster"
    assert out.as_record()["source_transition"] == "none->raster"


def test_explicit_increase_then_bare_replacement_native_values():
    first = delta(reason="prescribed_increase")
    second = delta(
        "water0",
        proposer="water",
        at=1,
        replacements=(cell(cover=0.0, height=0.0, frontal=0.0),),
        reason="prescribed_decrease",
    )
    e0 = envelope((first, second))
    e1 = prepare(e0).envelope
    e2 = prepare(e1, bind(1)).envelope
    assert e1.state.raster.cover_fraction[0, 0] == 0.5
    for name in ("cover_fraction", "height_m", "frontal_area_density"):
        assert getattr(e2.state.raster, name)[0, 0] == 0.0
    validate_vegetation_state(e2.state, *SHAPE)
    assert e2.version == 2 and e2.cursor["last_sequence"] == {"wind": 0, "water": 0}


def test_rain_is_actual_schedule_kind_and_maps_to_water_proposer():
    p = plan((delta("water0", proposer="water", at=1),))
    assert p.event_kinds == KINDS
    assert p.plan_record()["event_kinds"] == list(KINDS)
    with pytest.raises(v.VegetationError):
        plan((delta(proposer="water"),))
    with pytest.raises(v.VegetationError):
        plan(kinds=("water",))


def test_event_candidate_soil_is_returned_by_identity_without_exchange():
    soil = np.full(SHAPE, 0.125, dtype=np.float64)
    before = soil.tobytes()
    env = envelope((delta(),))
    out = prepare(env, soil=soil, version=env.version_sha256)
    assert out.resource.soil is soil and soil.tobytes() == before
    assert out.resource.vegetation_version_sha256 == out.envelope.version_sha256
    assert out.resource.water_exchange.channels == ()


@pytest.mark.parametrize(
    "value", [(("uptake_m3", 0.0),), (("deposition", 0.0),), (object(),)]
)
def test_every_named_or_nonempty_exchange_refuses(value):
    with pytest.raises(v.VegetationError, match="water_exchange_unsupported"):
        v.WaterExchange(value)


def test_wrong_non_none_resource_version_refuses_before_copy():
    env = envelope((delta(),))
    before = native_bytes(env)
    with (
        mock.patch.object(np, "array", side_effect=AssertionError("copy must not run")),
        pytest.raises(v.VegetationError, match="resource_version_mismatch"),
    ):
        prepare(env, version="b" * 64)
    assert native_bytes(env) == before and env.version == 0


@pytest.mark.parametrize("name", ["state", "initial_state"])
def test_public_native_snapshots_cannot_reenable_authority_writes(name):
    env = envelope()
    before = native_bytes(env)
    snapshot = getattr(env, name)
    assert snapshot is not env._current
    for field in v.RASTER_FIELDS:
        array = getattr(snapshot.raster, field)
        authority = getattr(env._current.raster, field)
        assert array.base is None and not np.shares_memory(array, authority)
        array.setflags(write=True)
        array.flat[0] = 0.75
    assert native_bytes(env) == before
    env.verify()
    assert env.snapshot_copy_bytes == 5 * 8 * np.prod(SHAPE)


def test_public_cover_snapshot_does_not_share_authority():
    env = envelope()
    array = env.cover_fraction_snapshot()
    array[:] = 0.75
    assert not np.any(env.state.raster.cover_fraction)


def test_initial_native_input_is_not_retained():
    source = empty_vegetation_state(*SHAPE)
    env = envelope(initial=source)
    source.raster.cover_fraction[:] = 0.75
    assert not np.any(env.state.raster.cover_fraction)
    env.verify()


@pytest.mark.parametrize("mutate_value", [False, True])
def test_private_native_borrow_mutation_is_detected(mutate_value):
    env = envelope()
    raw = env._native_borrow()
    raw.raster.cover_fraction.setflags(write=True)
    if mutate_value:
        raw.raster.cover_fraction.flat[0] = 0.75
    with pytest.raises(v.VegetationError, match="accepted_state_externally_mutated"):
        env.verify()


def test_full_native_clock_roundtrip_keeps_real_discrepancy():
    source = ledger((0.1 + 0.2,), epoch=0.1, returned=(0.2,)).as_record()
    binding = v.BoundaryBinding.from_ledger_record(
        source, event_index=0, event_kind="wind", schedule_sha256=SCHEDULE
    )
    assert binding.record()["clock_ledger"] == source
    entry = source["entries"][0]
    assert entry["endpoint_discrepancy"] != "0/1"
    assert Fraction(entry["endpoint_discrepancy"]) == Fraction(0.2) - (
        Fraction(0.1 + 0.2) - Fraction(0.1)
    )
    assert entry["returned_elapsed"]["hex"] == (0.2).hex()
    assert binding.record()["schedule_sha256"] == SCHEDULE


@pytest.mark.parametrize(
    "field,value",
    [
        ("global_end", "42/1"),
        ("global_start", "42/1"),
        ("endpoint_elapsed", "42/1"),
        ("endpoint_discrepancy", "42/1"),
        ("index", True),
        ("kind", "idle"),
        ("returned_elapsed", {"hex": "0x1.0000000000000p+0", "fraction": "42/1"}),
    ],
)
def test_complete_clock_native_parser_rejects_redundant_or_returned_fields(
    field, value
):
    record = ledger().as_record()
    record["entries"][0][field] = value
    with pytest.raises(v.VegetationError, match="invalid_binding"):
        v.BoundaryBinding.from_ledger_record(
            record, event_index=0, event_kind="wind", schedule_sha256=SCHEDULE
        )


def test_binding_uses_full_ledger_and_exact_event_count():
    with pytest.raises(v.VegetationError):
        v.BoundaryBinding.from_ledger_record(
            ledger().records()[0],
            event_index=0,
            event_kind="wind",
            schedule_sha256=SCHEDULE,
        )
    with pytest.raises(v.VegetationError):
        v.BoundaryBinding.from_ledger_record(
            ledger((8.0, 9.0)).as_record(),
            event_index=0,
            event_kind="wind",
            schedule_sha256=SCHEDULE,
        )


def test_wrong_schedule_digest_refuses_even_empty_boundary():
    env = envelope()
    with pytest.raises(v.VegetationError, match="binding_mismatch"):
        prepare(env, bind(schedule="b" * 64))


def test_full_clock_raw_metadata_refuses_before_native_parser():
    record = ledger().as_record()
    record["native_epoch"]["fraction"] = "1" * 4096
    with (
        mock.patch.object(
            ClockLedger,
            "from_record",
            side_effect=AssertionError("native parser reached"),
        ),
        pytest.raises(v.VegetationError, match="metadata_cap_exceeded"),
    ):
        v.BoundaryBinding.from_ledger_record(
            record, event_index=0, event_kind="wind", schedule_sha256=SCHEDULE
        )


def test_empty_boundary_replay_retains_exact_binding_and_returns_current():
    env = envelope()
    first = prepare(env)
    again = prepare(first.envelope)
    assert again.duplicate and again.envelope is first.envelope
    assert again.new_receipts == () and len(first.envelope.bindings) == 1
    assert env.bindings == () and first.envelope.version == 0
    with pytest.raises(v.VegetationError, match="delta_conflict"):
        prepare(first.envelope, bind(ends=(8.5,)))


def test_old_delta_replay_after_later_boundary_returns_current_without_rewind():
    e0 = envelope(
        (delta(), delta("water0", proposer="water", at=1, replacements=(cell((1, 1)),)))
    )
    first = prepare(e0)
    second = prepare(first.envelope, bind(1))
    replay = prepare(second.envelope)
    assert (
        replay.duplicate
        and replay.envelope is second.envelope
        and replay.envelope.version == 2
    )
    assert replay.new_receipts == first.new_receipts
    assert replay.resource.vegetation_version_sha256 == second.envelope.version_sha256


def test_new_boundary_refuses_rewritten_earlier_native_history():
    env = prepare(envelope()).envelope
    before = env.as_record()
    with pytest.raises(v.VegetationError, match="binding_history_conflict"):
        prepare(env, bind(1, ends=(8.5, 9.0)))
    assert env.as_record() == before


def test_boundary_skip_refuses_and_no_accepted_state_changes():
    env = envelope((delta(),))
    before = env.as_record()
    with pytest.raises(v.VegetationError, match="boundary_skipped"):
        prepare(env, bind(1))
    assert env.as_record() == before


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("_cursor_json", "{}"),
        ("_receipts_json", "[{}]"),
        ("_bindings_json", "[{}]"),
        ("_last_binding_json", "{}"),
        ("_version", 1),
        ("_source_transition", "none->raster"),
        ("_version_sha256", "b" * 64),
    ],
)
def test_envelope_seal_detects_metadata_tampering(attribute, value):
    env = envelope()
    object.__setattr__(env, attribute, value)
    with pytest.raises(v.VegetationError, match="accepted_metadata_externally_mutated"):
        env.verify()


def test_actual_frozen_delta_metadata_is_checked_against_held_plan_bytes():
    item = delta()
    p = plan((item,))
    object.__setattr__(item, "reason", "prescribed_increase")
    with pytest.raises(v.VegetationError, match="plan_externally_mutated"):
        p.verify()


def test_plan_record_is_safe_copy_and_modified_held_record_refuses():
    p = plan((delta(),))
    record = p.plan_record()
    record["deltas"][0]["payload"]["reason"] = "changed"
    p.verify()
    object.__setattr__(p, "_record_json", "{}")
    with pytest.raises(v.VegetationError, match="plan_externally_mutated"):
        p.verify()


def test_binding_fields_and_bytes_tamper_refuses():
    binding = bind()
    object.__setattr__(binding, "_event_kind", "rain")
    with pytest.raises(v.VegetationError, match="invalid_binding"):
        binding.verify()


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_plan_record_bytes", 4096),
        ("max_event_copy_bytes", 1),
        ("max_run_copy_bytes", 1),
        ("max_candidate_peak_bytes", 1),
    ],
)
def test_plan_source_cap_admission_precedes_record_native_and_json(field, value):
    limits = dataclasses.replace(v.VegetationLimits(), **{field: value})
    with (
        mock.patch.object(
            v, "_plan_record", side_effect=AssertionError("record allocation reached")
        ),
        mock.patch.object(
            v, "_native", side_effect=AssertionError("native work reached")
        ),
        mock.patch.object(
            v.json, "dumps", side_effect=AssertionError("JSON allocation reached")
        ),
        pytest.raises(v.VegetationError),
    ):
        plan((delta(),), limits=limits)


def test_prospective_metadata_cap_precedes_changed_array_allocation():
    env = envelope(
        (delta(),),
        limits=dataclasses.replace(v.VegetationLimits(), max_metadata_bytes=4096),
    )
    binding = bind()
    before = native_bytes(env)
    with (
        mock.patch.object(
            np,
            "array",
            side_effect=AssertionError("candidate array allocation reached"),
        ),
        pytest.raises(v.VegetationError, match="metadata_cap_exceeded"),
    ):
        prepare(env, binding)
    assert native_bytes(env) == before and env.version == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: cell(cover=10**10000),
        lambda: v.VegetationLimits(max_cells=10**10000),
        lambda: v.CellReplacement((10**10000, 0), 0.5, 0.25, 0.2, 0.5, 0.5),
        lambda: v.candidate_peak_bytes(10**10000),
    ],
)
def test_huge_plain_integer_is_bounded_vegetation_error(factory):
    with pytest.raises(v.VegetationError) as captured:
        factory()
    assert len(str(captured.value)) < 256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cover": -0.0},
        {"cover": float("nan")},
        {"cover": float("inf")},
        {"cover": True},
    ],
)
def test_noncanonical_authored_values_refuse(kwargs):
    with pytest.raises(v.VegetationError):
        cell(**kwargs)


def test_invalid_native_canonical_replacement_refuses_during_static_plan():
    with pytest.raises(v.VegetationError, match="invalid_replacement"):
        plan((delta(replacements=(cell(cover=1.5),)),))
    with pytest.raises(v.VegetationError, match="bare_cell_not_canonical"):
        plan((delta(replacements=(cell(cover=0.0, height=0.0, frontal=1.0),)),))


def test_order_sequence_duplicate_cells_and_payload_ids_refuse():
    cases = [
        (delta(sequence=1),),
        (delta(), delta("wind1", sequence=1, replacements=(cell(), cell()))),
        (delta(), delta(sequence=1)),
        (delta("water0", proposer="water", at=1), delta("wind0")),
    ]
    for values in cases:
        with pytest.raises(v.VegetationError):
            plan(values)


def test_failure_after_valid_first_delta_preserves_accepted_arrays_flags_and_metadata():
    first = delta(proposer="prescribed")
    second = delta(
        "next", proposer="prescribed", sequence=1, replacements=(cell((1, 1)),)
    )
    env = envelope((first, second))
    before = native_bytes(env)
    record = env.as_record()
    native_validate = v._validate_native
    calls = []

    def fail_second(n, state, shape):
        # Native verification checks the unchanged accepted states first;
        # the second staged raster has both explicit authored cells populated.
        if state is not env._current and state.raster.cover_fraction[1, 1] == 0.5:
            calls.append("second_staged_candidate")
            raise v.VegetationError("injected_native_failure", "second candidate")
        calls.append("valid")
        return native_validate(n, state, shape)

    with (
        mock.patch.object(v, "_validate_native", side_effect=fail_second),
        pytest.raises(v.VegetationError, match="injected_native_failure"),
    ):
        prepare(env)
    assert "second_staged_candidate" in calls
    assert native_bytes(env) == before and env.as_record() == record
    assert env.version == 0 and not env.receipts and not env.bindings


def test_only_changed_fields_copied_no_parent_state_chain_and_noop_is_explicit():
    first = delta(proposer="prescribed")
    second = delta("same", proposer="prescribed", sequence=1, at=1)
    e0 = envelope((first, second))
    p1 = prepare(e0)
    p2 = prepare(p1.envelope, bind(1))
    e2 = p2.envelope
    assert p1.new_receipts[0]["changed_fields"] == [
        "height_m",
        "cover_fraction",
        "frontal_area_density",
    ]
    assert p1.new_receipts[0]["copied_bytes"] == 3 * 8 * np.prod(SHAPE)
    assert (
        p2.new_receipts[0]["changed_fields"] == []
        and p2.new_receipts[0]["copied_bytes"] == 0
    )
    assert e2.native_content_sha256 == p1.envelope.native_content_sha256
    assert e2.version_sha256 != p1.envelope.version_sha256
    assert e2._initial is e0._initial
    assert not any(
        isinstance(getattr(e2, name), v.LandscapeVegetation) for name in e2.__slots__
    )
    arrays = {
        id(getattr(state.raster, name))
        for state in (e2._initial, e2._current)
        for name in v.RASTER_FIELDS
    }
    assert len(arrays) <= 10
    assert not any(
        getattr(e2._current.raster, name).flags.writeable for name in v.RASTER_FIELDS
    )


def test_raster_native_source_is_supported_objects_and_unknown_layout_refuse():
    native = uniform_vegetation_raster(*SHAPE, height_m=0.5, cover_fraction=0.25)
    env = envelope(initial=native)
    assert (
        env.state.source == "raster"
        and env.native_content_sha256 == vegetation_content_hash(native)
    )
    bad = VegetationState(source="objects", raster=native.raster, objects=object())
    with pytest.raises(v.VegetationError, match="objects_source_unsupported"):
        envelope(initial=bad)

    class Derived(VegetationRasterState):
        pass

    bad_raster = Derived(
        **{name: getattr(native.raster, name) for name in v.RASTER_FIELDS}
    )
    with pytest.raises(v.VegetationError, match="requalification_required"):
        envelope(
            initial=VegetationState(source="raster", raster=bad_raster, objects=None)
        )


def test_largest_domain_and_finite_caps_are_source_bound():
    p = plan(shape=(8, 8))
    env = v.LandscapeVegetation.from_native(empty_vegetation_state(8, 8), plan=p)
    assert v.candidate_peak_bytes(64) == (4 * 5 + 2 + 4) * 8 * 64 + 16 * 1024
    assert v.candidate_peak_bytes(64) <= p.limits.max_candidate_peak_bytes
    assert env.snapshot_copy_bytes == 5 * 8 * 64
    with pytest.raises(v.VegetationError):
        plan(shape=(9, 8))
    with pytest.raises(v.VegetationError):
        v.VegetationLimits(max_cells=65)
    assert json.dumps(env.as_record(), sort_keys=True)


@pytest.mark.parametrize(
    "value",
    [
        "\x7f",
        "".join(chr(c) for c in range(128)),
        "\x80\u07ff\uffff\U00010000\U0010ffff\ud800",
    ],
)
def test_ascii_json_admission_exact_escape_lengths(value):
    assert v._json_admit(value) == len(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    )


def test_del_amplified_clock_metadata_refuses_before_native_parser():
    record = ledger().as_record()
    record["entries"][0]["index"] = ["\x7f" * 2048] * 100
    with (
        mock.patch.object(
            ClockLedger,
            "from_record",
            side_effect=AssertionError("native parser reached"),
        ),
        pytest.raises(v.VegetationError, match="metadata_cap_exceeded"),
    ):
        v.BoundaryBinding.from_ledger_record(
            record, event_index=0, event_kind="wind", schedule_sha256=SCHEDULE
        )

# StageB: explicit synthetic interface integration. These authored replacements
# prescribe values; no growth/mortality/uptake law or measured calibration.
DYNAMIC_SHAPE = (4, 4)


def dynamic_runner(*, wake=False, deltas=None, vegetation_limits=None, runner_limits=None):
    import functools

    import fixture_dense_production as production
    from masy.dense_bed import build_dense_bed
    from masy.dense_production import DenseProductionLimits
    from masy.dense_runner import DenseRunner

    events = production.four_events()
    cfg = production.model(shape=DYNAMIC_SHAPE)
    if wake:
        # Separate labelled test uses existing native wake unchanged defaults;
        # only master enable is explicit. This is not a calibrated response.
        cfg = dataclasses.replace(cfg, vegetation_wake=dataclasses.replace(cfg.vegetation_wake, enabled=True))
    if deltas is None:
        deltas = (delta("authored_wind_increase", replacements=(cell(cover=0.5),)),
                  delta("authored_water_decrease", proposer="water", at=1,
                        replacements=(cell(cover=0.25, height=0.125, frontal=0.25),)))
    authored = v.build_vegetation_plan(deltas, shape=DYNAMIC_SHAPE, n_classes=1, n_voxels=6,
        event_kinds=tuple(e.kind for e in events.events), schedule_sha256=events.digest, limits=vegetation_limits)
    with mock.patch.object(production, "build_dense_bed", functools.partial(build_dense_bed, vegetation_profile=authored)):
        bed, events, wind, water, profile = production.components(events=events, cfg=cfg, mound=False)
    runner = DenseRunner(bed, events, wind=wind, water=water, control=production.legacy.runner_control(),
        limits=production.legacy.limits() if runner_limits is None else runner_limits,
        wind_profile=profile, production_limits=DenseProductionLimits(), vegetation_plan=authored)
    return runner, cfg


def accepted_snapshot(runner):
    """By-value evidence of the accepted boundary; active failure evidence excluded."""
    return (runner.accepted.as_record(), runner._bed.identity_record(), runner.ledger_record(),
            runner.receipts, runner.vegetation_record, runner._soil.tobytes(),
            runner._vegetation_copied_bytes, id(runner._bed.state), id(runner._bed._dynamic),
            id(runner._soil), id(runner._routing), id(runner._pool), id(runner._totals), id(runner._ledger))


def test_four_event_positive_negative_updates_reach_actual_native_models():
    from masy import dense_runner as dr
    runner, cfg = dynamic_runner()
    bed = runner._bed
    seen_wind, seen_water, candidates = [], [], []
    advance, complete, prepare_bed = dr._advance_wind_block, dr._complete_event, bed._prepare_dynamic

    def wind(state, block, config):
        version = bed._dynamic.version
        assert state.physical.vegetation is bed._dynamic._native_borrow()
        assert state.vegetation_wind_cache is None  # no incompatible prior-event cache
        initial_hash = vegetation_content_hash(state.physical.vegetation)
        before_fields = {f: getattr(state.physical.vegetation.raster, f).tobytes() for f in v.RASTER_FIELDS}
        out = advance(state, block, config)
        assert config.vegetation_wake_spec.enabled is False
        assert out.state.vegetation_wind_cache is None
        assert before_fields == {f: getattr(state.physical.vegetation.raster, f).tobytes() for f in v.RASTER_FIELDS}
        seen_wind.append((version, initial_hash, None, before_fields))
        return out

    def water(state, context, column, field, schedule, cover, sediment, **kwargs):
        env = bed._dynamic
        assert cover.flags.owndata and cover.flags.writeable
        assert not np.shares_memory(cover, env._native_borrow().raster.cover_fraction)
        np.testing.assert_array_equal(cover, env._native_borrow().raster.cover_fraction, strict=True)
        fields = {f: getattr(env._native_borrow().raster, f).tobytes() for f in v.RASTER_FIELDS}
        seen_water.append((env.version, env.native_content_sha256, fields, cover.copy()))
        return complete(state, context, column, field, schedule, cover, sediment, **kwargs)

    def prepared(source, **kwargs):
        candidate = kwargs["prepared_vegetation"]
        assert candidate.resource.water_exchange.channels == ()
        # Exact event candidate, including native rain reset, retained by identity.
        candidates.append((kwargs["kind"], candidate.resource.soil, candidate.resource.soil.tobytes()))
        return prepare_bed(source, **kwargs)

    with mock.patch.object(dr, "_advance_wind_block", wind), mock.patch.object(dr, "_complete_event", water), mock.patch.object(bed, "_prepare_dynamic", prepared):
        for i in range(4):
            receipt = runner.step()
            assert receipt["status"] == "accepted", (receipt.get("code"), receipt.get("detail"))
            assert runner._soil is candidates[-1][1] and runner._soil.tobytes() == candidates[-1][2]
            assert bed._dynamic.cursor["next_event_index"] == i + 1
    assert [w[0] for w in seen_wind] == [0, 2]
    assert [w[0] for w in seen_water] == [1, 2]
    assert seen_wind[1][3] == seen_water[1][2]  # all five fields same shared version
    assert seen_water[0][3][0, 0] == 0.5 and seen_water[1][3][0, 0] == 0.25
    assert runner.report()["status"] == "complete"
    assert runner._bed._dynamic.version == 2
    assert runner._vegetation_copied_bytes == 4352  # three changed fields per delta, not full-five upper
    assert runner.report()["schema"] == v.REPORT_SCHEMA_V2
    assert runner.static_record["schema"] == v.STATIC_SCHEMA_V2
    assert [r["vegetation"]["input_version"] for r in runner.receipts] == [0, 1, 2, 2]
    # Native sediment closure, thresholds and rain reset remain actual authority.
    assert np.all(runner._soil == 0.0)
    assert cfg.vegetation_wake == type(cfg.vegetation_wake)()
    assert not cfg.vegetation_wake.enabled  # actual applied response remains separate task63 qualification


def test_late_second_native_failure_keeps_first_accepted_boundary():
    from masy import dense_runner as dr
    runner, _ = dynamic_runner()
    first = runner.step()
    assert first["status"] == "accepted", (first.get("code"), first.get("detail"))
    before = accepted_snapshot(runner)
    complete, calls = dr._complete_event, []
    def completed_then_failed(*args, **kwargs):
        out = complete(*args, **kwargs)
        calls.append(type(out).__name__)
        raise ValueError("authored late second-native failure after native completion")
    with mock.patch.object(dr, "_complete_event", completed_then_failed):
        receipt = runner.step()
    assert calls and receipt["code"] == "rain_native_failure", (receipt.get("code"), receipt.get("detail"))
    assert receipt["status"] == "native_refused"
    assert accepted_snapshot(runner) == before
    assert runner._next == 1 and runner._bed._dynamic.version == 1
    assert runner.active_evidence is None  # raising native call exposed no partial DryHandoff


def test_late_vegetation_receipt_failure_keeps_native_bed_soil_and_clock():
    runner, _ = dynamic_runner()
    first = runner.step()
    assert first["status"] == "accepted", (first.get("code"), first.get("detail"))
    before = accepted_snapshot(runner)
    # Force a fallible metadata check after the next real native event, before publication.
    with mock.patch.object(runner._bed, "_verify_prepared", side_effect=v.VegetationError("authored_prepare_refusal", "no publication")):
        receipt = runner.step()
    assert receipt["code"] == "authored_prepare_refusal", (receipt.get("code"), receipt.get("detail"))
    assert accepted_snapshot(runner) == before


@pytest.mark.parametrize("field", ["max_event_copy_bytes", "max_run_copy_bytes"])
def test_dynamic_handoff_copy_budget_refuses_before_stage(field):
    from masy import dense_bed as db
    limits = dataclasses.replace(v.VegetationLimits(), **{field: 6 * 8 * int(np.prod(DYNAMIC_SHAPE)) if field == "max_run_copy_bytes" else 1})
    runner, _ = dynamic_runner(deltas=(), vegetation_limits=limits)
    before = accepted_snapshot(runner)
    with mock.patch.object(db.DenseBed, "stage_copy", side_effect=AssertionError("staging must not start")):
        receipt = runner.step()
    assert receipt["code"] == ("vegetation_event_copy_cap" if field == "max_event_copy_bytes" else "vegetation_run_copy_cap"), receipt
    assert accepted_snapshot(runner) == before


def test_dynamic_work_budget_refuses_before_stage():
    import fixture_dense_production as production
    from masy import dense_bed as db
    # Measure the actual unchanged native-wrapper base estimate. A cap equal to
    # this value admits the old estimate; only the additional vegetation overlap
    # can cause work_bytes_cap. The old128byte test was vacuous for this contract.
    probe, _ = dynamic_runner()
    seen, admit = [], probe._vegetation_work_admission
    def observe(estimate):
        seen.append(estimate)
        return admit(estimate)
    with mock.patch.object(probe, "_vegetation_work_admission", observe):
        first = probe.step()
    assert first["status"] == "accepted", (first.get("code"), first.get("detail"))
    assert len(seen) == 1 and seen[0] > 8 * int(np.prod(DYNAMIC_SHAPE))
    runner, _ = dynamic_runner(runner_limits=production.legacy.limits(max_work_bytes=seen[0]))
    before = accepted_snapshot(runner)
    with mock.patch.object(db.DenseBed, "stage_copy", side_effect=AssertionError("staging must not start")):
        receipt = runner.step()
    assert receipt["code"] == "work_bytes_cap", receipt
    assert accepted_snapshot(runner) == before


@pytest.mark.parametrize("mutation", ["vegetation", "threshold", "baseline", "mobile", "clock"])
def test_dynamic_wind_publication_reuses_complete_native_guard(mutation):
    from masy import dense_bed as db
    runner, _ = dynamic_runner(deltas=())
    bed = runner._bed
    physical = bed.fresh_scheduler_state(runner._pool, time_s=bed.time_s).physical
    if mutation == "vegetation":
        physical = dataclasses.replace(physical, vegetation=bed.vegetation)
    elif mutation == "threshold":
        physical = dataclasses.replace(physical, threshold_modifier_state=dataclasses.replace(bed.threshold_modifier_state))
    elif mutation == "baseline":
        physical = dataclasses.replace(physical, ledger_reconciliation_baseline=object())
    elif mutation == "mobile":
        physical = dataclasses.replace(physical, mobile_state=object())
    else:
        physical = dataclasses.replace(physical, time_s=1.0)
    clock = ClockLedger(0.0)
    clock.append("event", native_start=0.0, native_end=0.0, commit_before=0.0, commit_after=0.0, returned_elapsed=0.0)
    binding = v.BoundaryBinding.from_ledger_record(clock.as_record(), event_index=0, event_kind="wind", schedule_sha256=runner._schedule.digest)
    prepared = v.prepare_vegetation(bed._dynamic, binding, resource=v.JointResourceCandidate(runner._soil))
    before = accepted_snapshot(runner)
    with pytest.raises(db.DenseBedError):
        bed._prepare_dynamic(physical, kind="wind", time_s=0.0, prepared_vegetation=prepared)
    assert accepted_snapshot(runner) == before


def test_old_passive_codec_refuses_dynamic_v0_before_filesystem(tmp_path):
    from masy import dense_runner_checkpoint as codec
    runner, _ = dynamic_runner(deltas=())
    target = tmp_path / "must_not_exist"
    sources = {"kind": "source_manifest_prototype", "masy": "a" * 64, "maple": "b" * 64, "syrup": "c" * 64}
    # Authored structurally valid references; no artifact opened because the real
    # codec's layout refusal precedes writes, hashes and child reads.
    native_identity = {"schema": "masy.dense_boundary_physical.v1", "caller": {
        "schema": "masy.dense_runner_boundary_native.v1", "source_identity": sources,
        "environment_sha256": "c" * 64, "static_sha256": runner.static_sha256},
        "original_inputs_sha256": "d" * 64, "case_sha256": runner._bed.case_sha256,
        "bed_identity_sha256": runner._bed.identity_sha256()}
    ref = {"format": codec._NATIVE_FORMAT, "relative_path": "native", "manifest_file_sha256": "d" * 64, "identity": native_identity}
    original = {"format": codec._ORIGINAL_FORMAT, "relative_path": "original", "manifest_file_sha256": "d" * 64, "identity": "e" * 64}
    with pytest.raises(codec.BoundaryCheckpointError, match="runner_layout"):
        codec.save_passive_boundary(target, runner, artifact_root=tmp_path,
            originals=original, genesis=ref, physical=ref,
            source_identity=sources, environment_sha256="c" * 64)
    assert not target.exists() and list(tmp_path.iterdir()) == []


def test_constructor_snapshot_work_cap_precedes_cover_copy():
    from masy import dense_bed as db
    from masy import dense_runner as dr
    runner, _ = dynamic_runner(deltas=())
    with (
        mock.patch.object(db.DenseBed, "vegetation_cover_fraction", side_effect=AssertionError("cover copy must not run")),
        pytest.raises(dr.DenseRunnerError, match="work_bytes_cap"),
    ):
        dr.DenseRunner(runner._bed, runner._schedule, wind=runner._wind, water=runner._water,
                control=runner._control, limits=dataclasses.replace(runner._limits, max_work_bytes=127),
                wind_profile=runner._wind_profile, production_limits=runner._production_limits,
                vegetation_plan=runner._vegetation_plan)


def test_native_wind_carrier_is_rechecked_after_vegetation_preparation():
    runner, _ = dynamic_runner()
    bed, before = runner._bed, accepted_snapshot(runner)
    verify = bed._verify_prepared
    def late_carrier_change(prepared):
        assert prepared.kind == "wind"
        # Change only the staged native result, after initial preparation.
        object.__setattr__(prepared.publication_source, "ledger_reconciliation_baseline", object())
        return verify(prepared)
    with mock.patch.object(bed, "_verify_prepared", late_carrier_change):
        receipt = runner.step()
    assert receipt["code"] == "unsupported_reconciliation_baseline", receipt
    assert accepted_snapshot(runner) == before
