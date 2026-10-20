"""Authored native component controls; no runner, event or transport calls.

Native construction/preparation creates fresh test endpoints. Passive restore
must borrow them unchanged and never replay that preparation.
"""

import copy
import dataclasses
from unittest import mock

import check_dense_vegetation as f
import numpy as np
import pytest
from maple.core.types.vegetation import (
    VegetationState,
    empty_vegetation_state,
    validate_vegetation_state,
)

from masy import landscape_vegetation as v
from masy.clock import ClockLedger


def decode(plan, record=None, **changes):
    args = {
        "expected_plan_sha256": plan.plan_sha256,
        "shape": plan.shape,
        "n_classes": plan.n_classes,
        "n_voxels": plan.n_voxels,
        "event_kinds": plan.event_kinds,
        "schedule_sha256": plan.schedule_sha256,
        "limits": plan.limits,
    }
    args.update(changes)
    return v.vegetation_plan_from_record(
        plan.plan_record() if record is None else record, **args
    )


def prepared(deltas=(), *, accepted=0, initial=None, returned=None):
    env = f.envelope(deltas, initial=initial)
    ledger = ClockLedger(7.0)
    for index in range(accepted):
        start = 7.0 + index
        ledger.append(
            "event",
            native_start=start,
            native_end=start + 1.0,
            commit_before=start,
            commit_after=start + 1.0,
            returned_elapsed=1.0 if returned is None else returned[index],
        )
        binding = v.BoundaryBinding.from_ledger_record(
            ledger.as_record(),
            event_index=index,
            event_kind=f.KINDS[index],
            schedule_sha256=f.SCHEDULE,
        )
        env = v.prepare_vegetation(
            env, binding, resource=v.JointResourceCandidate(None)
        ).envelope
    return env, ledger.as_record()


def native_bytes(*states):
    return tuple(
        tuple(
            (
                getattr(state.raster, name).tobytes(),
                getattr(state.raster, name).flags.writeable,
            )
            for name in v.RASTER_FIELDS
        )
        for state in states
    )


def restore(
    env, ledger, *, record=None, initial=None, current=None, plan=None, expected=None
):
    record = env.as_record() if record is None else record
    return v._restore_vegetation_envelope(
        env.plan if plan is None else plan,
        record,
        ledger_record=ledger,
        initial=env._native_borrow(initial=True) if initial is None else initial,
        current=env._native_borrow() if current is None else current,
        expected_metadata_sha256=record["metadata_sha256"]
        if expected is None
        else expected,
    )


def reseal(record, env):
    """Give a forged plain record a genuine checksum, never fake validation."""
    held = v.LandscapeVegetation(
        v._ENV_TOKEN,
        plan=env.plan,
        initial=env._native_borrow(initial=True),
        current=env._native_borrow(),
        initial_native_sha256=record["initial_native_content_sha256"],
        native_sha256=record["native_content_sha256"],
        version=record["version"],
        version_sha256=record["version_sha256"],
        receipts_json=v._canonical_json(record["receipts"]),
        cursor_json=v._canonical_json(record["cursor"]),
        bindings_json=v._canonical_json(record["bindings"]),
        last_binding_json=None
        if record["last_boundary_binding"] is None
        else v._canonical_json(record["last_boundary_binding"]),
        source_transition=record["source_transition"],
    )
    held._seal()
    record["metadata_sha256"] = held.metadata_sha256
    return record


@pytest.mark.parametrize("deltas", [(), (f.delta(),)])
def test_plan_roundtrip_preserves_exact_canonical_bytes_and_limits(deltas):
    original = f.plan(
        deltas, limits=dataclasses.replace(v.VegetationLimits(), max_cells=32)
    )
    result = decode(original)
    assert result is not original
    assert result.plan_record() == original.plan_record()
    assert result._record_json == original._record_json
    assert result.plan_sha256 == original.plan_sha256
    assert result.limits.as_record() == original.limits.as_record()


@pytest.mark.parametrize(
    "changes",
    [
        {"shape": (4, 3)},
        {"n_classes": 2},
        {"n_voxels": 5},
        {"event_kinds": ("rain", "wind", "wind")},
        {"schedule_sha256": "b" * 64},
        {"limits": dataclasses.replace(v.VegetationLimits(), max_cells=32)},
        {"expected_plan_sha256": "b" * 64},
    ],
)
def test_independent_plan_operands_refuse_before_native_builder(changes):
    p = f.plan((f.delta(),))
    with mock.patch.object(
        v, "_native", side_effect=AssertionError("native builder reached")
    ), pytest.raises(v.VegetationError):
        decode(p, **changes)


@pytest.mark.parametrize(
    "case",
    [
        "extra",
        "bool",
        "hex",
        "negative_zero",
        "infinite",
        "huge_integer",
        "exchange",
        "payload_hash",
        "future",
    ],
)
def test_plan_record_tamper_refuses_before_native_work(case):
    p = f.plan((f.delta("future", at=2),))
    record = p.plan_record()
    payload = record["deltas"][0]["payload"]
    if case == "extra":
        record["extra"] = None
    elif case == "bool":
        record["n_classes"] = True
    elif case == "hex":
        payload["replacements"][0]["height_m"] = "0x1p-2"
    elif case == "negative_zero":
        payload["replacements"][0]["height_m"] = "-0x0.0p+0"
    elif case == "infinite":
        payload["replacements"][0]["height_m"] = "inf"
    elif case == "huge_integer":
        payload["sequence"] = 1 << 10000
    elif case == "exchange":
        payload["water_exchange"] = [["authored_zero", "0x0.0p+0"]]
    elif case == "payload_hash":
        record["deltas"][0]["payload_sha256"] = "b" * 64
    else:
        payload["replacements"][0]["height_m"] = "0x1.0000000000000p-1"
    with mock.patch.object(
        v, "_native", side_effect=AssertionError("native builder reached")
    ), pytest.raises(v.VegetationError) as error:
        decode(p, record)
    assert len(str(error.value)) < 512


def test_initial_record_and_zero_boundary_restore_borrow_without_replay():
    env, ledger = prepared()
    initial = env._native_borrow(initial=True)
    before = native_bytes(initial)
    assert v._vegetation_initial_record(env.plan, initial) == env.as_record()
    restored_plan = decode(env.plan)
    record = env.as_record()
    current = env._native_borrow()
    with (
        mock.patch.object(
            v, "prepare_vegetation", side_effect=AssertionError("replay")
        ),
        mock.patch.object(v, "_copy_native", side_effect=AssertionError("copy")),
        mock.patch.object(
            v.LandscapeVegetation,
            "from_native",
            side_effect=AssertionError("initializer"),
        ),
        mock.patch.object(v, "_validate_native", wraps=v._validate_native) as validator,
    ):
        out = v._restore_vegetation_envelope(
            restored_plan,
            record,
            ledger_record=ledger,
            initial=initial,
            current=current,
            expected_metadata_sha256=record["metadata_sha256"],
        )
    assert validator.call_count > 0
    assert (
        out._native_borrow() is initial and out._native_borrow(initial=True) is initial
    )
    assert out.as_record() == env.as_record() and native_bytes(initial) == before
    snapshot = out.state
    snapshot.raster.height_m.setflags(write=True)
    snapshot.raster.height_m[0, 0] = 10.0
    assert native_bytes(initial) == before


def test_three_proposers_multiple_deltas_future_intent_and_noop():
    deltas = (
        f.delta("wind0"),
        f.delta("p0", proposer="prescribed"),
        f.delta(
            "water0",
            proposer="water",
            at=1,
            replacements=(f.cell(cover=0.25, height=0.125, frontal=0.25),),
        ),
        f.delta(
            "pending",
            proposer="prescribed",
            sequence=1,
            at=2,
            replacements=(f.cell(cover=0.75, height=0.375, frontal=0.75),),
        ),
    )
    env, ledger = prepared(deltas, accepted=2, returned=(0.75, 1.25))
    initial, current = env._native_borrow(initial=True), env._native_borrow()
    before = native_bytes(initial, current)
    with (
        mock.patch.object(
            v, "prepare_vegetation", side_effect=AssertionError("replay")
        ),
        mock.patch.object(v, "_copy_native", side_effect=AssertionError("copy")),
    ):
        out = restore(env, ledger, plan=decode(env.plan))
    assert out.version == 3 and out.cursor["next_event_index"] == 2
    assert out.cursor["last_sequence"] == {"wind": 0, "prescribed": 0, "water": 0}
    assert (
        out.receipts[1]["changed_fields"] == [] and out.receipts[1]["copied_bytes"] == 0
    )
    assert [r["payload_id"] for r in out.receipts] == ["wind0", "p0", "water0"]
    assert out.plan.deltas[-1].payload_id == "pending"
    assert out._native_borrow() is current and native_bytes(initial, current) == before
    assert out.as_record() == env.as_record()
    # A following component boundary applies its pending update once, with no model.
    following = ClockLedger.from_record(ledger)
    following.append(
        "event",
        native_start=9.0,
        native_end=10.0,
        commit_before=9.0,
        commit_after=10.0,
        returned_elapsed=1.0,
    )
    binding = v.BoundaryBinding.from_ledger_record(
        following.as_record(),
        event_index=2,
        event_kind=f.KINDS[2],
        schedule_sha256=f.SCHEDULE,
    )
    a = v.prepare_vegetation(out, binding, resource=v.JointResourceCandidate(None))
    b = v.prepare_vegetation(env, binding, resource=v.JointResourceCandidate(None))
    assert a.envelope.as_record() == b.envelope.as_record()
    duplicate = v.prepare_vegetation(
        a.envelope, binding, resource=v.JointResourceCandidate(None)
    )
    assert duplicate.duplicate and duplicate.envelope is a.envelope


def test_empty_boundary_retains_full_clock_discrepancy_and_conflicting_replay_refuses():
    env, ledger = prepared(accepted=1, returned=(0.25,))
    out = restore(env, ledger)
    assert out.receipts == () and len(out.bindings) == 1
    assert out.as_record()["last_boundary_binding"]["clock_ledger"] == ledger
    altered = f.ledger(returned=(0.75,)).as_record()
    with pytest.raises(v.VegetationError):
        restore(env, altered)


def test_all_five_authored_native_fields_roundtrip_without_response_law():
    # Explicit synthetic surface properties for a codec test, not calibration.
    replacement = v.CellReplacement(
        (0, 0),
        0.5,
        0.25,
        f.DEFAULT_VEGETATION_POROSITY * 0.5,
        f.DEFAULT_VEGETATION_DRAG_COEFFICIENT * 2.0,
        0.5,
    )
    env, ledger = prepared((f.delta(replacements=(replacement,)),), accepted=1)
    assert env.receipts[0]["changed_fields"] == list(v.RASTER_FIELDS)
    assert env.receipts[0]["copied_bytes"] == 5 * 8 * int(np.prod(f.SHAPE))
    out = restore(env, ledger)
    for name in v.RASTER_FIELDS:
        assert (
            getattr(out._native_borrow().raster, name)[0, 0].item().hex()
            == getattr(replacement, name).hex()
        )
    assert out.as_record() == env.as_record()


def test_signed_zero_selected_field_and_native_none_to_raster_noop():
    initial = empty_vegetation_state(*f.SHAPE)
    initial.raster.height_m[0, 0] = -0.0
    initial.raster.frontal_area_density[0, 1] = -0.0
    validate_vegetation_state(initial, *f.SHAPE)
    bare = (f.cell(cover=0.0, height=0.0, frontal=0.0),)
    env, ledger = prepared(
        (
            f.delta("first", proposer="prescribed", replacements=bare),
            f.delta("noop", proposer="prescribed", sequence=1, at=1, replacements=bare),
        ),
        accepted=2,
        initial=initial,
    )
    assert env.receipts[0]["changed_fields"] == ["height_m"]
    assert env.receipts[0]["copied_bytes"] == 8 * int(np.prod(f.SHAPE))
    assert (
        env.receipts[1]["changed_fields"] == [] and env.receipts[1]["copied_bytes"] == 0
    )
    assert env.receipts[0]["source_transition"] == "none->raster"
    out = restore(env, ledger)
    assert out._native_borrow().raster.height_m[0, 0].item().hex() == (0.0).hex()
    assert (
        out._native_borrow().raster.frontal_area_density[0, 1].item().hex()
        == (-0.0).hex()
    )
    assert native_bytes(out._native_borrow()) == native_bytes(env._native_borrow())
    nochange, clock = prepared((f.delta("bare", replacements=bare),), accepted=1)
    assert nochange.receipts[0]["changed_fields"] == [] and nochange.version == 1
    assert nochange.receipts[0]["source_transition"] == "none->raster"
    assert restore(nochange, clock).as_record() == nochange.as_record()


@pytest.mark.parametrize(
    "key",
    [
        "copied_bytes",
        "changed_cells",
        "event_index",
        "ledger_index",
        "sequence",
        "base_version",
        "version",
        "event_kind",
        "payload_id",
        "reason",
        "payload_sha256",
        "base_version_sha256",
        "delta_id",
        "version_sha256",
        "native_content_sha256",
        "changed_fields",
        "source_transition",
        "water_exchange",
        "extra",
    ],
)
def test_freshly_resealed_receipt_forgery_refuses_without_authority_mutation(key):
    env, ledger = prepared((f.delta(),), accepted=1)
    initial, current = env._native_borrow(initial=True), env._native_borrow()
    before = native_bytes(initial, current), env.as_record()
    record = copy.deepcopy(env.as_record())
    receipt = record["receipts"][0]
    if key == "extra":
        receipt[key] = None
    elif key == "changed_fields":
        receipt[key] = list(reversed(receipt[key]))
    elif key == "source_transition":
        receipt[key] = None
    elif key == "water_exchange":
        receipt[key] = [["authored_zero", 0]]
    elif type(receipt[key]) is int:
        receipt[key] += 1
    else:
        receipt[key] = (
            "b" * 64 if key.endswith("sha256") or key == "delta_id" else "wrong"
        )
    reseal(record, env)
    with pytest.raises(v.VegetationError):
        restore(env, ledger, record=record)
    assert (native_bytes(initial, current), env.as_record()) == before


@pytest.mark.parametrize(
    "case",
    [
        "cursor_count",
        "cursor_sequence",
        "cursor_copy",
        "bool_version",
        "binding",
        "last_binding",
        "plan",
        "source",
        "native",
        "initial",
        "version",
        "external_seal",
        "extra",
    ],
)
def test_envelope_binding_forgery_refuses(case):
    env, ledger = prepared((f.delta(),), accepted=1)
    record = copy.deepcopy(env.as_record())
    if case == "cursor_count":
        record["cursor"]["next_event_index"] = 0
    elif case == "cursor_sequence":
        record["cursor"]["last_sequence"]["wind"] = 1
    elif case == "cursor_copy":
        record["cursor"]["copied_bytes"] += 1
    elif case == "bool_version":
        record["version"] = True
    elif case == "binding":
        record["bindings"][0]["binding_sha256"] = "b" * 64
    elif case == "last_binding":
        record["last_boundary_binding"] = None
    elif case == "plan":
        record["plan_sha256"] = "b" * 64
    elif case == "source":
        record["source"] = "none"
    elif case == "native":
        record["native_content_sha256"] = "b" * 64
    elif case == "initial":
        record["initial_native_content_sha256"] = "b" * 64
    elif case == "version":
        record["version_sha256"] = "b" * 64
    elif case == "external_seal":
        record["metadata_sha256"] = "b" * 64
    else:
        record["extra"] = None
    with pytest.raises(v.VegetationError):
        restore(env, ledger, record=record, expected=env.metadata_sha256)


@pytest.mark.parametrize(
    "case", ["foreign", "subclass", "objects", "writable", "strided", "extra", "values"]
)
def test_native_endpoint_substitution_refuses_without_mutating_inputs(case):
    env, ledger = prepared((f.delta(),), accepted=1)
    source = env._native_borrow()
    if case == "foreign":
        bad = object()
    elif case == "subclass":

        class Derived(VegetationState):
            pass

        bad = Derived(source=source.source, raster=source.raster, objects=None)
    elif case == "objects":
        bad = dataclasses.replace(source, source="objects", objects=object())
    elif case == "writable":
        bad = env.state
        bad.raster.height_m.setflags(write=True)
    elif case == "strided":
        base = np.zeros((f.SHAPE[0], f.SHAPE[1] * 2), dtype=np.float64)
        array = base[:, ::2]
        array[:] = source.raster.height_m
        array.setflags(write=False)
        bad = dataclasses.replace(
            source, raster=dataclasses.replace(source.raster, height_m=array)
        )
    elif case == "extra":
        bad = dataclasses.replace(source)
        object.__setattr__(bad, "extra", 1)
    else:
        bad = env.state
        bad.raster.height_m.setflags(write=True)
        bad.raster.height_m[0, 0] += 0.125
        for name in v.RASTER_FIELDS:
            getattr(bad.raster, name).setflags(write=False)
    before = native_bytes(env._native_borrow(initial=True), source), env.as_record()
    with pytest.raises(v.VegetationError):
        restore(env, ledger, current=bad)
    assert (
        native_bytes(env._native_borrow(initial=True), source),
        env.as_record(),
    ) == before


@pytest.mark.parametrize(
    "case", ["string", "huge_integer", "container", "cycle", "ledger_nodes"]
)
def test_raw_metadata_caps_precede_native_work(case):
    env, ledger = prepared((f.delta(),), accepted=1)
    record = copy.deepcopy(env.as_record())
    if case == "string":
        record["source"] = "x" * 2049
    elif case == "huge_integer":
        record["version"] = 1 << 10000
    elif case == "container":
        record["receipts"] = [0] * 4097
    elif case == "cycle":
        record["receipts"].append(record)
    else:
        ledger["entries"][0]["index"] = [0] * 1025
    with mock.patch.object(
        v, "_native", side_effect=AssertionError("native work reached")
    ), pytest.raises(v.VegetationError, match="metadata_cap_exceeded"):
        # Save native refs before the tripwire; this helper's env access
        # would itself verify native state, so call restore entry directly.
        v._restore_vegetation_envelope(
            env.plan,
            record,
            ledger_record=ledger,
            initial=env._initial,
            current=env._current,
            expected_metadata_sha256=env.metadata_sha256,
        )


def test_full_future_plan_hash_cannot_be_rebound_to_old_envelope():
    old = (f.delta("pending", at=2),)
    env, ledger = prepared(old, accepted=1)
    changed = f.plan(
        (
            f.delta(
                "pending",
                at=2,
                replacements=(f.cell(cover=0.25, height=0.125, frontal=0.25),),
            ),
        )
    )
    assert decode(changed).plan_sha256 != env.plan.plan_sha256
    with pytest.raises(v.VegetationError, match="restore_mismatch"):
        restore(env, ledger, plan=changed)
