"""Dynamic original-input components: real native construction, no event runs."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from unittest import mock

import check_dense_dynamic_physical_checkpoint as f
import numpy as np
import pytest

from masy import dense_original_inputs as oi
from masy import landscape_vegetation as lv


@pytest.fixture(scope="module")
def original():
    runner = f.dynamic_genesis()
    return runner, oi.capture_original_inputs_v2(runner, source_identity=f.SOURCES)


def saved(tmp_path, pack):
    path = tmp_path / "original"
    seal = oi.save_original_inputs_v2(path, pack)
    return path, seal


def load(path, seal, **kwargs):
    return oi.load_original_inputs_v2(path, expected_manifest_sha256=seal,
        expected_source_identity=f.SOURCES, **kwargs)


def reseal(path, record):
    record["manifest_sha256"] = hashlib.sha256(oi._json(
        {key: value for key, value in record.items() if key != "manifest_sha256"})).hexdigest()
    (path / "manifest.json").chmod(0o600)
    (path / "manifest.json").write_bytes(oi._json(record))
    return record["manifest_sha256"]


def test_actual_typed_roundtrip_retains_one_full_future_plan_and_aliases(tmp_path, original):
    runner, pack = original
    path, seal = saved(tmp_path, pack)
    decoded = load(path, seal)
    out = decoded.originals
    assert type(decoded) is oi._DecodedOriginalInputsV2 and decoded._authority is oi._DECODED_V2_TOKEN
    assert decoded.plan.plan_record() == runner._vegetation_plan.plan_record()
    assert decoded.plan.plan_record()["deltas"][-1]["payload"]["after_event_index"] == 3
    assert out.vegetation_limits == runner._vegetation_plan.limits.as_record()
    assert out.static_parts["vegetation_profile"]["plan"] == decoded.plan.plan_record()
    assert out.wind.config.geometry is out.capsule.context.geometry
    assert out.wind.config.grain_classes is out.capsule.context.grain_classes
    assert out.wind.config.initial_available_fraction is out.capsule.context.initial_available_fraction
    assert out.content_sha256 == pack.content_sha256
    assert oi.validate_original_inputs_v2(out, admitted_plan=decoded.plan) == pack.content_sha256
    assert not out.initial_inventory_kg.flags.writeable
    assert not np.shares_memory(out.initial_inventory_kg, pack.initial_inventory_kg)


def test_original_roundtrip_never_initializes_bed_or_replays_update(tmp_path, original):
    runner, pack = original
    path, seal = saved(tmp_path, pack)
    forbidden = AssertionError("original decoder initialized/replayed state")
    with (mock.patch.object(f.db, "build_dense_bed", side_effect=forbidden),
          mock.patch.object(f.dr.DenseRunner, "__init__", side_effect=forbidden),
          mock.patch.object(lv, "prepare_vegetation", side_effect=forbidden),
          mock.patch.object(lv.LandscapeVegetation, "from_native", side_effect=forbidden),
          mock.patch.object(f.dr, "_advance_wind_block", side_effect=forbidden),
          mock.patch.object(f.dr, "_finalize_wind_event", side_effect=forbidden),
          mock.patch.object(f.dr, "_complete_event", side_effect=forbidden),
          mock.patch.object(lv, "vegetation_plan_from_record", wraps=lv.vegetation_plan_from_record) as decoder):
        result = load(path, seal)
    assert decoder.call_count == 1
    assert runner._bed._dynamic.version == 0 and result.plan.plan_sha256 == runner._vegetation_plan.plan_sha256


@pytest.mark.parametrize("bad", [None, 1, "ab", [["x"]], [[["x"], 1]]])
def test_raw_controls_report_closed_error_for_malformed_rows(bad):
    with pytest.raises(oi.OriginalInputsError):
        oi._raw_dynamic_controls_v2([], {"v": bad}, oi.OriginalPackLimits())


def test_limits_role_cannot_be_substituted_by_bool_or_above_ceiling(original):
    _, pack = original
    for value, code in ((True, "invalid_limit"), (lv._CEILINGS["max_cells"] + 1, "limit_above_ceiling")):
        forged = dataclasses.replace(pack, vegetation_limits={**pack.vegetation_limits, "max_cells": value})
        with pytest.raises((oi.OriginalInputsError, lv.VegetationError)) as error:
            oi.validate_original_inputs_v2(forged)
        assert error.value.code == code


def test_future_suffix_mutation_is_controlling_and_leaves_owner_unchanged(original):
    runner, pack = original
    before = f.signature(runner._bed.state), runner._bed._dynamic.as_record()
    parts = copy.deepcopy(pack.static_parts)
    parts["vegetation_profile"]["plan"]["deltas"][-1]["payload"]["reason"] = "prescribed_decrease"
    with pytest.raises((oi.OriginalInputsError, lv.VegetationError)) as error:
        oi.validate_original_inputs_v2(dataclasses.replace(pack, static_parts=parts))
    assert error.value.code == "plan_record_mismatch"
    assert before == (f.signature(runner._bed.state), runner._bed._dynamic.as_record())


def test_raw_byte_cap_refuses_before_npy_header_or_native_plan(tmp_path, original):
    _, pack = original
    path, seal = saved(tmp_path, pack)
    forbidden = AssertionError("expanded before cap")
    with (mock.patch.object(oi, "_read_npy_header", side_effect=forbidden),
          mock.patch.object(lv, "vegetation_plan_from_record", side_effect=forbidden),
          pytest.raises(oi.OriginalInputsError, match="serialized_cap")):
        load(path, seal, limits=oi.OriginalPackLimits(max_serialized_bytes=1))


def test_source_mismatch_refuses_before_payload_hash_or_native_factories(tmp_path, original):
    _, pack = original
    path, seal = saved(tmp_path, pack)
    with (mock.patch.object(oi, "_hash_fd", side_effect=AssertionError("payload hash")),
          mock.patch.object(oi, "_registry_v2", side_effect=AssertionError("native registry")),
          pytest.raises(oi.OriginalInputsError, match="source_identity")):
        oi.load_original_inputs_v2(path, expected_manifest_sha256=seal,
            expected_source_identity={**f.SOURCES, "masy": "a" * 64})


def test_unknown_or_v1_schema_cannot_be_loaded_by_v2(tmp_path, original):
    _, pack = original
    path, _ = saved(tmp_path, pack)
    record = json.loads((path / "manifest.json").read_bytes())
    record["schema"] = oi.SCHEMA
    seal = reseal(path, record)
    with pytest.raises(oi.OriginalInputsError, match="manifest_identity"):
        load(path, seal)


def test_v1_entrypoints_still_refuse_dynamic_capture_and_v2_capsule(tmp_path, original):
    runner, pack = original
    with pytest.raises(oi.OriginalInputsError):
        oi.capture_original_inputs(runner, source_identity=f.SOURCES)
    with pytest.raises(oi.OriginalInputsError):
        oi.save_original_inputs(tmp_path / "v1", pack)
    assert not (tmp_path / "v1").exists()
