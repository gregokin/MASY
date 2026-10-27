"""Pure metadata refusals; no Native/CUDA import or physical success fixture."""
import dataclasses
import types

import pytest

from masy.dense_schedule import DenseSchedule
from masy.dense_runner import DenseRunnerError
from masy.dense_device_bed import DeviceBedError
from masy.windowed_device_bed import DeviceWindowedBedLimits, DeviceWindowedBedOwner
from masy.windowed_device_runner import _require_entry_api, _schedule_profile, build_windowed_device_runner


def _wind():
    return dict(kind="wind", duration_s=1.0, shear_velocity_m_s=0.5, direction_deg_cw_from_north=90.0)


def _rain():
    return dict(kind="rain", recession_guard_s=180.0,
        intervals=[dict(duration_s=60.0, intensity_mm_h=72.0)])


def _schedule(events):
    return DenseSchedule.from_input(dict(schema="masy.dense_schedule.v1", events=events))


def test_closed_four_event_schedule_rejects_other_valid_orders():
    accepted = _schedule([_wind(), _rain(), _wind(), _rain()])
    assert _schedule_profile(accepted) is None
    for events in ([_wind(), _rain()], [_rain(), _wind(), _rain(), _wind()],
                   [_wind(), _wind(), _rain(), _rain()]):
        with pytest.raises(DenseRunnerError, match="device_schedule"):
            _schedule_profile(_schedule(events))


def test_valid_dense_duration_does_not_relax_device_record_scope():
    wind = _wind()
    wind["duration_s"] = 2.0
    with pytest.raises(DenseRunnerError, match="device_schedule"):
        _schedule_profile(_schedule([wind, _rain(), _wind(), _rain()]))


def test_fixed_calendar_and_unknown_schedule_fields_stay_refused():
    for extra in ({"absolute_start_s": 0.0}, {"unknown": 1}):
        value = dict(schema="masy.dense_schedule.v1", events=[_wind(), _rain(), _wind(), _rain()], **extra)
        with pytest.raises(ValueError):
            DenseSchedule.from_input(value)


@pytest.mark.parametrize("value", [True, 0, -1, (2 << 30)+1])
def test_payload_limit_types_and_fixed_cap_remain_closed(value):
    with pytest.raises(DeviceBedError):
        DeviceWindowedBedLimits(value, 5 << 30, 2 << 30, 16 << 30)


def test_mutated_limit_is_revalidated_without_native_imports():
    limits = DeviceWindowedBedLimits(1 << 20, 5 << 30, 2 << 30, 16 << 30)
    object.__setattr__(limits, "extra", 1)
    with pytest.raises(DeviceBedError, match="invalid_limit"):
        limits.__post_init__()


def _api_module():
    """Synthetic Python signatures only, never a Native owner or result."""
    module = types.ModuleType("synthetic_signature_subject")
    exec("def commit_qualification_device_entry(state, context, graph, reference, current_time_s, routing_source_m, *, control):\n"
         "    raise AssertionError('signature-only subject must never execute')\n"
         "def complete_qualification_device_event(state, context, column, field, schedule, vegetation, sediment, *, max_end_s, control, policy, report_every_s):\n"
         "    raise AssertionError('signature-only subject must never execute')\n", vars(module))
    return module


def test_signature_guard_rejects_wrong_globals_and_extra_parameters():
    module = _api_module()
    assert _require_entry_api(types.SimpleNamespace(shared=module)) is None
    function = module.commit_qualification_device_entry
    namespace = dict(vars(module))
    module.commit_qualification_device_entry = types.FunctionType(function.__code__, namespace,
        function.__name__, function.__defaults__, function.__closure__)
    with pytest.raises(DenseRunnerError, match="unsupported_device_water_api"):
        _require_entry_api(types.SimpleNamespace(shared=module))
    module = _api_module()
    exec("def commit_qualification_device_entry(state, context, graph, reference, current_time_s, routing_source_m, *, control, fallback=None):\n"
         "    raise AssertionError('signature-only subject')\n", vars(module))
    with pytest.raises(DenseRunnerError, match="unsupported_device_water_api"):
        _require_entry_api(types.SimpleNamespace(shared=module))


def test_nonwindowed_owner_is_refused_before_any_native_receiver():
    with pytest.raises(DenseRunnerError, match="invalid_input"):
        build_windowed_device_runner(object(), None, wind=None, water=None, control=None,
            profile=None, record_limits=None, runner_limits=None, resource_limits=None, qualification_budget=None)
    with pytest.raises(TypeError, match="subclasses"):
        type("ForeignWindowedOwner", (DeviceWindowedBedOwner,), {})

