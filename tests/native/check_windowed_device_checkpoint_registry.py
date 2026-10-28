"""Genuine Native CPU metadata only; no landscape, transport or GPU work.

Root runs this selector under its original CPU Native bootstrap. The values
are the unchanged existing Native avalanche fixture's delta-budget inputs.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from maple.surface.voxels.windowed_delta import WindowedDeltaBudgets
from masy import windowed_device_checkpoint as checkpoint
from masy.dense_runner_checkpoint import BoundaryImageLimits


TAG = "maple.surface.voxels.windowed_delta:WindowedDeltaBudgets"
FIELDS = ("max_gathered_cells", "max_gather_columns", "max_journal_records",
          "max_deep_read_bytes", "max_gather_temp_bytes", "max_resident_temp_bytes")
LIMITS = BoundaryImageLimits(max_cells=16, max_classes=1, max_voxels=6,
                             max_cohorts=4000, max_events=4)


def test_native_windowed_delta_budget_registry_roundtrip_and_strict_refusals(monkeypatch):
    budget = WindowedDeltaBudgets(max_gathered_cells=16, max_gather_columns=4,
        max_journal_records=320, max_deep_read_bytes=2*16*6*1*8,
        max_gather_temp_bytes=1 << 20, max_resident_temp_bytes=4 << 20)
    assert type(budget).__module__ + ":" + type(budget).__qualname__ == TAG
    assert tuple(field.name for field in dataclasses.fields(budget)) == FIELDS
    assert all(field.init for field in dataclasses.fields(budget))
    assert checkpoint._class_table()[TAG] == (FIELDS, False)
    assert checkpoint._class(TAG) == (WindowedDeltaBudgets, FIELDS, False)

    # Repeat the same real metadata object under nested input paths; the
    # existing wire must retain its backward alias and exact constructor.
    encoder = checkpoint._Encoder(LIMITS, np)
    root = encoder.node({"validation": budget, "step": {"delta_budgets": budget}})
    wire = {"root": root, "arrays": encoder.rows, "numeric_bytes": encoder.numeric}
    assert wire["arrays"] == [] and wire["numeric_bytes"] == 0
    assert checkpoint._wire_preflight(wire, LIMITS) == 0
    restored = checkpoint._decode(wire, np.empty(0, dtype=np.uint8), LIMITS, np, host=True)
    assert type(restored["validation"]) is WindowedDeltaBudgets
    assert dataclasses.asdict(restored["validation"]) == dataclasses.asdict(budget)
    assert restored["validation"] is restored["step"]["delta_budgets"]

    # Reproduce the exact old missing-row predicate, using the same genuine
    # Native object and unchanged encoder. No Native class/provider is faked.
    old_rows = tuple(row for row in checkpoint._EXTRA_TYPES
                     if row[:2] != ("maple.surface.voxels.windowed_delta", "WindowedDeltaBudgets"))
    assert len(old_rows) + 1 == len(checkpoint._EXTRA_TYPES)
    with monkeypatch.context() as omitted:
        omitted.setattr(checkpoint, "_EXTRA_TYPES", old_rows)
        with pytest.raises(checkpoint.WindowedDeviceCheckpointError,
                           match="class outside frozen Source roster"):
            checkpoint._Encoder(LIMITS, np).node(budget)

    for change in ("reordered", "missing", "extra"):
        bad = copy.deepcopy(wire)
        fields = bad["root"]["v"][0][1]["v"]
        if change == "reordered":
            fields.reverse()
        elif change == "missing":
            fields.pop()
        else:
            fields.append(["unknown", {"t": "scalar", "v": 0}])
        with pytest.raises(checkpoint.WindowedDeviceCheckpointError, match="class_fields"):
            checkpoint._wire_preflight(bad, LIMITS)

    unknown = copy.deepcopy(wire)
    unknown_tag = TAG + "Unregistered"
    unknown["root"]["v"][0][1]["c"] = unknown_tag
    with monkeypatch.context() as no_import:
        no_import.setattr(checkpoint.importlib, "import_module",
                         lambda *args: pytest.fail("unknown class must refuse before import"))
        with pytest.raises(checkpoint.WindowedDeviceCheckpointError, match="class_fields"):
            checkpoint._wire_preflight(unknown, LIMITS)
        with pytest.raises(checkpoint.WindowedDeviceCheckpointError, match="class outside frozen Source roster"):
            checkpoint._class(unknown_tag)

    # A closed field schema does not bypass the actual Native constructor's
    # policy checks on decode. Keep its original nonnegative integer bound.
    invalid = copy.deepcopy(wire)
    invalid["root"]["v"][0][1]["v"][-1][1]["v"] = -1
    assert checkpoint._wire_preflight(invalid, LIMITS) == 0
    with pytest.raises(ValueError, match="max_resident_temp_bytes must be >= 0"):
        checkpoint._decode(invalid, np.empty(0, dtype=np.uint8), LIMITS, np, host=True)
