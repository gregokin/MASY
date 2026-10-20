"""Tiny authored interface states; native physics and calibration unchanged.

Uses the existing qualified six-class fixture and actual one-event native
runner/finalizer to obtain a legitimate pending finalization ledger. This is
source-manifest CPU prototype evidence, not combined restart acceptance.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from fixture_dense_alternation import (
    build_bed,
    empty_pool,
    make_runner,
    schedule,
    wind,
    wind_config,
)


def pending_boundary():
    bed = build_bed()
    config = wind_config(bed)
    runner = make_runner(bed, schedule(wind()), config=config, pool=empty_pool())
    receipt = runner.step()
    assert receipt["status"] == "accepted", receipt
    assert np.any(bed.state.ledger.pending_bed_mass_change_kg != 0.0)
    assert bed.state.committed_topography.last_commit_time_s < bed.time_s
    assert np.any(bed.state.sediment_availability.bound_mass_kg > 0.0)
    return bed, runner.mobile_state, config, receipt


def initial_boundary():
    bed = build_bed()
    return bed, empty_pool(), wind_config(bed)


def leaves(value, path=""):
    if dataclasses.is_dataclass(value):
        result = {}
        for field in dataclasses.fields(value):
            result.update(leaves(getattr(value, field.name), path + "." + field.name))
        return result
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            result.update(leaves(item, path + "." + key))
        return result
    if type(value) in (list, tuple):
        result = {}
        for index, item in enumerate(value):
            result.update(leaves(item, path + f"[{index}]"))
        return result
    return {path: value}


def exact(value):
    result = {}
    for path, item in leaves(value).items():
        if type(item) is np.ndarray:
            result[path] = ("array", item.dtype.str, item.shape, item.tobytes())
        elif type(item) is float:
            result[path] = ("float", item.hex())
        else:
            result[path] = (type(item).__qualname__, item)
    return result


def bed_exact(bed, pool):
    return {"state": exact(bed.state), "pool": exact(pool), "vegetation": exact(bed.vegetation),
            "modifiers": exact(bed.threshold_modifier_state), "time": bed.time_s.hex(),
            "case": bed.case_record, "initialization": bed.initialization_receipt,
            "identity": bed.identity_sha256(), "context": exact(bed.context)}


if __name__ == "__main__":
    native_bed, native_pool, _, _ = pending_boundary()
    print({"fixture": "genuine native accepted pending finalization boundary",
           "time_kind": type(native_bed.time_s).__name__,
           "ledger_scalar_kind": type(native_bed.state.ledger.largest_residual_kg).__name__,
           "pool_genuinely_empty": not bool(np.any(native_pool.active_mask))})
