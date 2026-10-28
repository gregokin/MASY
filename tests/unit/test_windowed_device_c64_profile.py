"""Pure metadata refusals; no Native/CUDA import or physical success fixture."""
import types

import pytest

from masy.dense_schedule import DenseSchedule
from masy.dense_runner import DenseRunnerError
from masy.dense_device_bed import DeviceBedError
from masy.windowed_device_c64_bed import DeviceWindowedC64BedLimits, DeviceWindowedC64BedOwner
from masy.windowed_device_c64_runner import _require_entry_api, _schedule_profile, build_windowed_device_c64_runner
from masy import windowed_device_c64_bed as c64_bed
from masy import windowed_device_c64_runner as c64
from masy.windowed_device_bed import DeviceWindowedBedLimits


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
        DeviceWindowedC64BedLimits(value, 5 << 30, 2 << 30, 16 << 30)


def test_mutated_limit_is_revalidated_without_native_imports():
    limits = DeviceWindowedC64BedLimits(1 << 20, 5 << 30, 2 << 30, 16 << 30)
    object.__setattr__(limits, "extra", 1)
    with pytest.raises(DeviceBedError, match="invalid_limit"):
        limits.__post_init__()


def _api_module():
    """Synthetic Python signatures only, never a Native owner or result."""
    module = types.ModuleType("synthetic_signature_subject")
    exec("def commit_c64_qualification_device_entry(state, context, graph, reference, current_time_s, routing_source_m, *, control):\n"
         "    raise AssertionError('signature-only subject must never execute')\n"
         "def complete_c64_qualification_device_event(state, context, column, field, schedule, vegetation, sediment, *, max_end_s, control, policy, report_every_s):\n"
         "    raise AssertionError('signature-only subject must never execute')\n", vars(module))
    return module


def test_signature_guard_rejects_wrong_globals_and_extra_parameters():
    module = _api_module()
    assert _require_entry_api(types.SimpleNamespace(shared=module)) is None
    function = module.commit_c64_qualification_device_entry
    namespace = dict(vars(module))
    module.commit_c64_qualification_device_entry = types.FunctionType(function.__code__, namespace,
        function.__name__, function.__defaults__, function.__closure__)
    with pytest.raises(DenseRunnerError, match="unsupported_device_water_api"):
        _require_entry_api(types.SimpleNamespace(shared=module))
    module = _api_module()
    exec("def commit_c64_qualification_device_entry(state, context, graph, reference, current_time_s, routing_source_m, *, control, fallback=None):\n"
         "    raise AssertionError('signature-only subject')\n", vars(module))
    with pytest.raises(DenseRunnerError, match="unsupported_device_water_api"):
        _require_entry_api(types.SimpleNamespace(shared=module))


def test_nonwindowed_owner_is_refused_before_any_native_receiver():
    with pytest.raises(DenseRunnerError, match="invalid_input"):
        build_windowed_device_c64_runner(object(), None, wind=None, water=None, control=None,
            profile=None, record_limits=None, runner_limits=None, resource_limits=None, qualification_budget=None)
    with pytest.raises(TypeError, match="subclasses"):
        type("ForeignWindowedOwner", (DeviceWindowedC64BedOwner,), {})


def _dimension_metadata():
    token = object()
    return (token, token, token, token), dict(shape=(8, 8, 20, 3, 1, 8),
        impact_rows=512, creep_rows=256, solve_rows=256, validation_rows=256,
        block_steps=1, blocks=1, total_steps=1)


def test_profile_join_requires_actual_same_object_not_equal_metadata():
    identities, values = _dimension_metadata()
    assert c64._dimension_join(*identities, **values) is None
    for slot in range(4):
        changed = list(identities)
        changed[slot] = object()
        with pytest.raises(DenseRunnerError, match="incompatible_dimension_profile"):
            c64._dimension_join(*changed, **values)


@pytest.mark.parametrize("name,value", [("impact_rows", 256), ("creep_rows", 512),
    ("solve_rows", 512), ("validation_rows", 512), ("blocks", 2), ("total_steps", 2),
    ("block_steps", True)])
def test_wind_impact_policy_does_not_widen_nonimpact_or_multistep(name, value):
    identities, values = _dimension_metadata()
    values[name] = value
    with pytest.raises(DenseRunnerError, match="incompatible_dimension_policy"):
        c64._dimension_join(*identities, **values)


@pytest.mark.parametrize("shape", [(4, 4, 6, 3, 1, 8), (8, 8, 20, 3, True, 8), [8, 8, 20, 3, 1, 8]])
def test_dimension_shape_is_closed_plain_metadata(shape):
    identities, values = _dimension_metadata()
    with pytest.raises(DenseRunnerError, match="incompatible_dimension_profile"):
        c64._dimension_join(*identities, **dict(values, shape=shape))


def test_small_limit_kind_cannot_open_c64_native_receiver():
    old = DeviceWindowedBedLimits(1 << 20, 5 << 30, 2 << 30, 16 << 30)
    with pytest.raises(DeviceBedError, match="invalid_input"):
        DeviceWindowedC64BedOwner.from_device_bed(None, None, case_record={}, time_s=1234.0,
            limits=old, vegetation=None, threshold_modifier_state=None, trajectory_semantics=None)


def test_cover_requires_shared_owner_identity():
    owned = object()
    assert c64._cover_identity(owned, owned) is None
    with pytest.raises(DenseRunnerError, match="vegetation_identity"):
        c64._cover_identity(object(), owned)


def _counters(value=0):
    return {name: value for name in ("gate_count", "logical_work_debited", "logical_read_bytes_debited",
        "versions_reserved", "h2d_bytes_reserved", "d2h_bytes_reserved", "scalar_reads_reserved",
        "maximum_host_predicate", "maximum_device_predicate")}


def _counter_limits():
    # Only a plain metadata subject for the pure counter predicate; no Native provider/result.
    return types.SimpleNamespace(max_work=100, max_logical_read_bytes=100, max_new_versions=100,
                                 max_host_bytes=100, max_device_bytes=100)


def test_actual_counter_domains_sum_calls_and_keep_peak_maxima():
    result = c64._combine_water_counters(_counters(3), _counters(4), _counters(5), _counter_limits())
    assert result["gate_count"] == result["logical_work_debited"] == 12
    assert result["maximum_host_predicate"] == result["maximum_device_predicate"] == 5


@pytest.mark.parametrize("value", [True, -1, 1 << 63])
def test_bad_counter_does_not_become_an_accepted_delta(value):
    row = _counters()
    row["logical_read_bytes_debited"] = value
    with pytest.raises(DenseRunnerError, match="water_counter_domain"):
        c64._combine_water_counters(_counters(), row, _counters(), _counter_limits())


def test_counters_refuse_combined_ceiling_and_int63_overflow():
    with pytest.raises(DenseRunnerError, match="water_cumulative_ceiling"):
        c64._combine_water_counters(_counters(40), _counters(40), _counters(40), _counter_limits())
    row = _counters()
    row["h2d_bytes_reserved"] = (1 << 63)-1
    with pytest.raises(DenseRunnerError, match="water_cumulative_ceiling"):
        c64._combine_water_counters(row, row, _counters(), _counter_limits())


def test_unissued_native_source_operand_refuses_without_native_import():
    original = c64_bed._NATIVE_K1_SOURCE_PINS
    try:
        c64_bed._NATIVE_K1_SOURCE_PINS = None
        with pytest.raises(DeviceBedError, match="native_scale_source"):
            c64_bed._require_completed_scale_api(preflight=True)
        with pytest.raises(DeviceBedError, match="native_scale_source"):
            c64_bed._wrapper_source_frontier()
    finally:
        c64_bed._NATIVE_K1_SOURCE_PINS = original


def test_equal_rebound_source_pins_or_helper_roster_do_not_become_authority():
    for name in ("_NATIVE_K1_SOURCE_PINS", "_K1_SCALE_HELPERS"):
        original = getattr(c64_bed, name)
        equal_replacement = tuple(list(original))
        assert equal_replacement == original and equal_replacement is not original
        try:
            setattr(c64_bed, name, equal_replacement)
            with pytest.raises(DeviceBedError, match="native_scale_source"):
                c64_bed._check_data_policy()
        finally:
            setattr(c64_bed, name, original)


def test_rebound_runner_scope_or_report_cap_refuses():
    for name, value in (("SCOPE", "other"), ("REPORT_ROWS", 4097)):
        original = getattr(c64, name)
        try:
            setattr(c64, name, value)
            with pytest.raises(DenseRunnerError, match="source_binding_changed"):
                c64._check_data_policy()
        finally:
            setattr(c64, name, original)


def test_own_source_binding_refuses_changed_real_wrapper_callable():
    original = c64._cover_identity
    try:
        c64._cover_identity = lambda a, b: None
        with pytest.raises(DeviceBedError, match="wrapper_source"):
            c64_bed._wrapper_source_frontier()
    finally:
        c64._cover_identity = original


def test_common_wind_binding_replaces_only_loaded_globals_and_rejects_missing_owner(monkeypatch):
    common = c64._cpu_windowed
    replacement_names = {"_windowed_native", "_prepare_record", "_resident_fold_bound"}
    for function, expected in (
            (common.WindowedRunner._wind_event, {"_windowed_native", "_prepare_record", "_resident_fold_bound"}),
            (common.WindowedRunner._wind_accept_record, {"_resident_fold_bound"})):
        bound = c64._wind_binding(function)
        assert bound.__code__ is function.__code__
        assert bound.__defaults__ is function.__defaults__
        assert bound.__closure__ is function.__closure__
        assert bound.__globals__.keys() == vars(common).keys()
        assert {name for name, value in vars(common).items()
                if bound.__globals__[name] is not value} == expected
        assert replacement_names & set(function.__code__.co_names) == expected
    # Clone a genuine MASY enter body into the accept slot without calling it.
    # Its extra loaded globals must refuse, rather than use CPU originals.
    enter = common.WindowedRunner._wind_event
    accept = common.WindowedRunner._wind_accept_record
    changed = types.FunctionType(enter.__code__, vars(common), accept.__name__,
                                 enter.__defaults__, enter.__closure__)
    with monkeypatch.context() as patch:
        patch.setattr(common.WindowedRunner, "_wind_accept_record", changed)
        with pytest.raises(DenseRunnerError, match="source_binding_changed"):
            c64._wind_binding(changed)
    monkeypatch.delattr(common, "_resident_fold_bound")
    with pytest.raises(DenseRunnerError, match="source_binding_changed"):
        c64._wind_binding(common.WindowedRunner._wind_accept_record)
