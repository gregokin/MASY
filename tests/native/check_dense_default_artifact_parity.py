"""Executed twice by task harness, using original and candidate source packages."""
from __future__ import annotations

import resource
import sys

import fixture_dense_alternation as fixture

from masy import dense_bed as adapter
from masy import dense_runner as runner_module
from masy.dense_bed import SHARED_FIELDS


def artifact():
    if "masy.dense_production" in sys.modules:
        raise AssertionError("default runner imported the opt-in resource module")
    old = resource.getrlimit
    def forbidden(*args, **kwargs):
        raise AssertionError("default runner queried new process resource policy")
    resource.getrlimit = forbidden
    try:
        bed = fixture.build_bed()
        schedule = fixture.schedule(fixture.wind(), fixture.rain(), fixture.wind(), fixture.rain())
        runner = fixture.make_runner(bed, schedule, config=fixture.wind_config(bed, outer_timestep_s=0.025))
        report = runner.run()
        assert report["status"] == "complete", report
        # Also compare a transactional failure artifact and its recovery boundary.
        stopped = fixture.make_runner(fixture.build_bed(), fixture.reference_schedule(),
                                      limit=fixture.limits(max_receipt_bytes=256))
        stop = stopped.step()
        assert stop["code"] == "receipt_cap_exceeded", stop
        refusals = {}
        def keep(name, candidate, code):
            receipt = candidate.step()
            assert receipt["code"] == code, receipt
            refusals[name] = {"static_sha256": candidate.static_sha256, "receipt": receipt,
                              "report": candidate.report()}
        limited = fixture.make_runner(fixture.build_bed(), fixture.reference_schedule(),
                                      limit=fixture.limits(max_work_bytes=1))
        keep("work_cap", limited, "work_bytes_cap")
        mutated = fixture.make_runner(fixture.build_bed(), fixture.reference_schedule())
        mutated._water.vegetation_cover_fraction[0, 0] = 0.5
        keep("static_mutation", mutated, "static_input_mutated_before_event")
        advance = runner_module._advance_wind_block
        def refuse(*args, **kwargs):
            raise ValueError("authored default native refusal")
        try:
            runner_module._advance_wind_block = refuse
            failed = fixture.make_runner(fixture.build_bed(), fixture.reference_schedule())
            keep("native_refusal", failed, "wind_native_failure")
        finally:
            runner_module._advance_wind_block = advance
        complete = runner_module._complete_event
        try:
            runner_module._complete_event = lambda *args, **kwargs: object()
            wet = fixture.make_runner(fixture.build_bed(), fixture.schedule(fixture.rain()))
            keep("incomplete_water", wet, "rain_not_dry")
        finally:
            runner_module._complete_event = complete
        if "masy.dense_production" in sys.modules:
            raise AssertionError("default path loaded production resource helpers")
        return {"static": runner.static_record, "static_sha256": runner.static_sha256, "report": report,
                "fields": {name: adapter._canonical_static(getattr(bed.state, name), name) for name in SHARED_FIELDS},
                "vegetation": adapter._canonical_static(bed.vegetation, "vegetation"),
                "threshold": adapter._canonical_static(bed.threshold_modifier_state, "threshold"),
                "mobile": adapter._canonical_static(runner.mobile_state, "mobile"),
                "soil": adapter._array_digest(runner.soil_water_m), "time_s_hex": bed.time_s.hex(),
                "stop_static_sha256": stopped.static_sha256, "stop": stop, "stop_report": stopped.report(),
                "default_refusal_artifacts": refusals}
    finally:
        resource.getrlimit = old
