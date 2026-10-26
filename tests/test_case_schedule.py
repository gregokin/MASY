"""Affected pure metadata tests; declarations here are synthetic, not model facts."""
import copy
import dataclasses

import pytest

from masy.case_schedule import CaseSchedule, CaseScheduleError, CaseWindGroup, WindDeclaration
from masy.clock import decode_time


def blocks():
    return (WindDeclaration(0, 0.4, 90.0, 0.5), WindDeclaration(1, 0.6, 120.0, 0.75),
            WindDeclaration(2, 0.3, 180.0, 0.25))


def declaration():
    return {"schema": "masy.case_schedule.v1", "events": [
        {"kind": "wind", "first_block": 0, "stop_block": 2},
        {"kind": "rain", "water_profile_id": "explicit", "intervals": [
            {"duration_s": 2.0, "intensity_mm_h": 5.0}], "recession_guard_s": 10.0},
        {"kind": "wind", "first_block": 2, "stop_block": 3}]}


def bind(value):
    return CaseSchedule.from_input(value, blocks=blocks(), case_sha256="a" * 64,
        wind_record_sha256="b" * 64, water_profile_id="explicit", water_identity_sha256="c" * 64,
        max_events=8)


def test_grouping_preserves_all_original_forcing():
    schedule = bind(declaration())
    assert type(schedule.events[0]) is CaseWindGroup
    assert schedule.events[0].blocks == blocks()[:2]
    assert [b.index for e in schedule.events if type(e) is CaseWindGroup for b in e.blocks] == [0, 1, 2]
    assert schedule.as_record()["events"][0]["local_to_original"] == [0, 1]


def test_resolution_uses_actual_nonzero_origin():
    schedule = bind(declaration())
    resolved = schedule.resolve(0, 7.25)
    assert [decode_time(e["native"]) for e in resolved["resolved_edges"]["edges"]] == [7.25, 7.75, 8.5]
    assert schedule.check_resolved(resolved) == resolved
    second = schedule.resolve(2, 20.75)  # actual returned prior end supplied by caller
    assert decode_time(second["resolved_edges"]["edges"][-1]["native"]) == 21.0


@pytest.mark.parametrize("first,stop", [(False, 2), (1, 2), (0, 0), (0, 4), (0, True)])
def test_bad_first_range_refuses(first, stop):
    value = declaration(); value["events"][0].update(first_block=first, stop_block=stop)
    with pytest.raises(CaseScheduleError): bind(value)


def test_missing_overlap_and_reordered_blocks_refuse():
    for first, stop in [(1, 3), (3, 3), (0, 1)]:
        value = declaration(); value["events"][-1].update(first_block=first, stop_block=stop)
        with pytest.raises(CaseScheduleError): bind(value)


def test_calendar_and_forcing_overrides_refuse():
    for field, value in [("start_time_s", 0), ("duration_s", 1), ("shear_velocity_m_s", .9)]:
        request = declaration(); request["events"][0][field] = value
        with pytest.raises(CaseScheduleError): bind(request)


def test_explicit_water_profile_and_full_record_required():
    request = declaration(); request["events"][1]["water_profile_id"] = "different"
    with pytest.raises(CaseScheduleError): bind(request)
    request = declaration(); request["events"].pop()
    with pytest.raises(CaseScheduleError): bind(request)


def test_rain_science_not_inferred_and_bad_numbers_refuse():
    for value in [True, float("nan"), float("inf"), -1.0]:
        request = declaration(); request["events"][1]["intervals"][0]["duration_s"] = value
        with pytest.raises(CaseScheduleError): bind(request)
    request = declaration(); del request["events"][1]["recession_guard_s"]
    with pytest.raises(CaseScheduleError): bind(request)


def test_resolution_and_binding_mutation_are_detected():
    schedule = bind(declaration()); value = copy.deepcopy(schedule.resolve(0, 0.0))
    value["intended"]["stop_block"] = 1
    with pytest.raises(CaseScheduleError): schedule.check_resolved(value)
    assert dataclasses.replace(schedule, water_identity_sha256="d" * 64).digest != schedule.digest


def test_negative_zero_dry_rain_and_malformed_resolution_refuse():
    for field in ("shear_velocity_m_s", "direction_deg", "duration_s"):
        values = {"index": 0, "shear_velocity_m_s": .4, "direction_deg": 90., "duration_s": .5}
        values[field] = -0.0
        with pytest.raises(CaseScheduleError): WindDeclaration(**values)
    request = declaration(); request["events"][1]["intervals"][0]["intensity_mm_h"] = 0.0
    with pytest.raises(CaseScheduleError): bind(request)
    schedule = bind(declaration())
    for start in (True, float("nan"), -0.0):
        with pytest.raises(CaseScheduleError): schedule.resolve(0, start)
    for edges in (None, {}, {"native_start": {"hex": "bad", "fraction": "0/1"}}):
        value = schedule.resolve(0, 0.0); value["resolved_edges"] = edges
        with pytest.raises(CaseScheduleError): schedule.check_resolved(value)
    with pytest.raises(CaseScheduleError): schedule.combined_digest("not a digest")
    with pytest.raises(CaseScheduleError): dataclasses.replace(schedule, events=schedule.events[:-1])
    with pytest.raises(CaseScheduleError): dataclasses.replace(schedule.events[0], stop_block=1)


def test_closed_json_rejects_duplicate_nonfinite_and_deep_input():
    from masy.case_cli import strict_json
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{} trailing', b'[' * 33 + b'0' + b']' * 33):
        with pytest.raises(ValueError): strict_json(raw, maximum=1024, entries=128, text=1024)
    with pytest.raises(ValueError): strict_json(b'{"long":"0123456789"}', maximum=1024, entries=128, text=3)


def test_case_policy_is_separate_and_no_inferred_larger_ceiling():
    from masy.case_runner import CaseCPUProcessPolicy
    from masy.dense_runner import DenseRunnerError
    from masy.case_cli import policy_from
    values = dict(address_space_bytes=4 << 30, cpu_soft_s=120, cpu_hard_s=135, wall_s=180.,
        max_output_bytes=1 << 20, max_run_bytes=64 << 20, max_request_bytes=1 << 20,
        max_input_entries=4096, max_path_chars=4096, max_water_input_file_bytes=8 << 20,
        max_water_input_bytes=16 << 20, max_report_bytes=1 << 20)
    policy = policy_from(values)
    assert type(policy) is CaseCPUProcessPolicy and policy.file_size_bytes == values['max_run_bytes']
    for field, value in [('cpu_soft_s', True), ('address_space_bytes', 8 << 30), ('cpu_hard_s', 119),
                         ('max_request_bytes', (1 << 20)+1), ('wall_s', float('nan'))]:
        changed = dict(values); changed[field] = value
        with pytest.raises((ValueError, DenseRunnerError)): policy_from(changed)
    changed = dict(values); changed['undeclared_headroom'] = 1
    with pytest.raises(CaseScheduleError): policy_from(changed)
