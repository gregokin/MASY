"""Native scalar tolerance wire contract; actual receipt payloads stay LOCAL.

The bounded controller selects baseline/native/checks. One-class and signed-zero
bodies are synthetic schema fixtures, not physical runs or calibration evidence.
"""

from __future__ import annotations

if not __debug__:
    raise SystemExit("Receipt checks require assertions")
import copy
import hashlib
import json
import math
import os
import stat
import struct
import unittest
from pathlib import Path
from unittest import mock

import fixture_dense_alternation as f
import numpy as np

from masy import dense_runner as dr
from masy import dense_runner_checkpoint as b

MODE = os.environ["MASY64_MODE"]
STAGE = Path(os.environ["MASY64_STAGE"])
DATA = Path(os.environ["MASY64_DATA"])
PAYLOAD = DATA / "native_rain_receipt.json"
COUNTS = {"wind_blocks": 0, "wind_finalization": 0, "water_completion": 0}
LIMITS = b.BoundaryImageLimits()


def write(path, value):
    raw = dr._canonical_json(value).encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    return hashlib.sha256(raw).hexdigest()


def read_actual():
    before = os.lstat(PAYLOAD)
    assert (
        stat.S_ISREG(before.st_mode)
        and before.st_nlink == 1
        and before.st_size <= 1 << 20
    )
    fd = os.open(PAYLOAD, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read((1 << 20) + 1)
        after = os.fstat(fd)
        assert (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
    finally:
        os.close(fd)
    assert (
        len(raw) == before.st_size
        and hashlib.sha256(raw).hexdigest() == os.environ["MASY64_RECEIPT_SHA256"]
    )
    value = json.loads(raw)
    assert dr._canonical_json(value).encode() == raw
    return value


def actual_receipt():
    native = {}

    def counted(name, fn):
        def call(*args, **kwargs):
            COUNTS[name] += 1
            result = fn(*args, **kwargs)
            if name == "water_completion":
                scalar = result.accounting["terminal_sediment_tolerance_kg"]
                native.update(
                    {
                        "terminal_type": type(scalar).__name__,
                        "terminal_hex": scalar.hex(),
                        "native_status": result.status,
                    }
                )
            return result

        return call

    with (
        mock.patch.object(
            dr,
            "_advance_wind_block",
            new=counted("wind_blocks", dr._advance_wind_block),
        ),
        mock.patch.object(
            dr,
            "_finalize_wind_event",
            new=counted("wind_finalization", dr._finalize_wind_event),
        ),
        mock.patch.object(
            dr, "_complete_event", new=counted("water_completion", dr._complete_event)
        ),
    ):
        runner = f.make_runner(f.build_bed(), f.schedule(f.wind(), f.rain()))
        wind = runner.step()
        assert wind["status"] == "accepted"
        rain = runner.step()
        assert rain["status"] == "accepted"
    assert COUNTS == {"wind_blocks": 1, "wind_finalization": 1, "water_completion": 1}
    assert native["terminal_type"] == "float" and native["native_status"] == "complete"
    tolerance = rain["classes"]["terminal_sediment_tolerance_kg"]
    assert (
        type(tolerance) is list
        and len(tolerance) == 1
        and type(tolerance[0]) is float
        and tolerance[0].hex() == native["terminal_hex"]
    )
    assert len(rain["classes"]["native_closure_tolerance_kg"]) == 6
    assert not np.any(runner.soil_water_m) and runner.report()["status"] == "complete"
    b._receipt_shape(wind, 6, LIMITS)
    evidence = {
        **native,
        "wire_terminal_length": len(tolerance),
        "closure_length": 6,
        "native_counts": COUNTS,
        "ledger": runner.ledger_record(),
        "unchanged_inputs": {
            "wind_s": f.WIND_S,
            "rain_s": f.RAIN_S,
            "intensity_mm_h": f.RAIN_MM_H,
            "recession_guard_s": f.GUARD_S,
        },
        "actual_rain_receipt_sha256": hashlib.sha256(
            dr._canonical_json(rain).encode()
        ).hexdigest(),
    }
    if MODE == "baseline":
        try:
            b._receipt_shape(rain, 6, LIMITS)
        except b.BoundaryCheckpointError as error:
            assert (
                error.code == "receipt_layout"
                and "terminal_sediment_tolerance_kg" in error.detail
            )
            evidence["baseline_refusal_reproduced"] = True
        else:
            raise AssertionError("baseline no longer reproduces confirmed defect")
        write(PAYLOAD, rain)
    else:
        before = dr._canonical_json(rain)
        b._receipt_shape(rain, 6, LIMITS)
        b._runner_admission(runner, LIMITS)
        assert (
            dr._canonical_json(rain) == before
            and dr._canonical_json(read_actual()) == before
        )
        evidence["fixed_full_runner_admission"] = True
    write(STAGE / "native_receipt_evidence.json", evidence)


def one_class_template(actual):
    """Synthetic receipt-SHAPE fixture; no one-class physical model is claimed."""
    body = copy.deepcopy(actual)
    for key, value in body["classes"].items():
        if type(value) is list and key != "terminal_sediment_tolerance_kg":
            body["classes"][key] = value[:1]
    for key in ("joint_residual_kg", "joint_bound_kg"):
        body["aggregate"][key] = body["aggregate"][key][:1]
    return body


class ReceiptSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actual = read_actual()

    def check(self, body, nc):
        b._receipt_shape(body, nc, LIMITS)

    def refuses_shape(self, value, nc=6, field="terminal_sediment_tolerance_kg"):
        body = copy.deepcopy(
            self.actual if nc == 6 else one_class_template(self.actual)
        )
        body["classes"][field] = value
        with self.assertRaises(b.BoundaryCheckpointError) as caught:
            self.check(body, nc)
        self.assertEqual(caught.exception.code, "receipt_layout")

    def test_actual_six_class_native_shape_is_passive(self):
        body = copy.deepcopy(self.actual)
        identity = id(body["classes"]["terminal_sediment_tolerance_kg"])
        before = dr._canonical_json(body)
        self.check(body, 6)
        self.assertEqual(dr._canonical_json(body), before)
        self.assertEqual(
            id(body["classes"]["terminal_sediment_tolerance_kg"]), identity
        )

    def test_one_class_shape_fixture_accepts_same_scalar_list(self):
        self.check(one_class_template(self.actual), 1)

    def test_terminal_tolerance_refuses_scalar_tuple_empty_and_two(self):
        for nc in (1, 6):
            for value in (0.0, (0.0,), [], [0.0, 0.0]):
                with self.subTest(nc=nc, value_type=type(value).__name__):
                    self.refuses_shape(value, nc)

    def test_six_entry_tolerance_is_not_silently_class_vector(self):
        self.refuses_shape([0.0] * 6)

    def test_terminal_scalar_refuses_other_kinds_and_nonfinite(self):
        for nc in (1, 6):
            for value in (
                [True],
                [0],
                [np.float64(0.0)],
                [float("nan")],
                [float("inf")],
                [-float("inf")],
            ):
                with self.subTest(nc=nc, value_type=type(value[0]).__name__):
                    body = copy.deepcopy(
                        self.actual if nc == 6 else one_class_template(self.actual)
                    )
                    body["classes"]["terminal_sediment_tolerance_kg"] = value
                    with self.assertRaises(b.BoundaryCheckpointError):
                        self.check(body, nc)

    def test_other_class_vectors_keep_exact_class_count(self):
        for nc in (1, 6):
            base = self.actual if nc == 6 else one_class_template(self.actual)
            for field, value in base["classes"].items():
                if type(value) is list and field != "terminal_sediment_tolerance_kg":
                    with self.subTest(nc=nc, field=field):
                        self.refuses_shape([] if nc == 1 else [0.0], nc, field)

    def test_native_closure_tolerance_stays_per_class(self):
        self.assertEqual(len(self.actual["classes"]["native_closure_tolerance_kg"]), 6)
        self.refuses_shape([0.0], 6, "native_closure_tolerance_kg")
        self.refuses_shape([0.0, 0.0], 1, "native_closure_tolerance_kg")

    def test_missing_and_unknown_class_fields_stay_closed(self):
        for action in ("remove", "add"):
            body = copy.deepcopy(self.actual)
            if action == "remove":
                body["classes"].pop("terminal_sediment_tolerance_kg")
            else:
                body["classes"]["terminal_sediment_tolerance_other_kg"] = [0.0]
            with self.assertRaises(b.BoundaryCheckpointError):
                self.check(body, 6)

    def test_signed_zero_shape_fixture_is_not_normalized(self):
        for nc in (1, 6):
            body = copy.deepcopy(
                self.actual if nc == 6 else one_class_template(self.actual)
            )
            body["classes"]["terminal_sediment_tolerance_kg"] = [-0.0]
            self.check(body, nc)
            self.assertEqual(
                struct.pack(">d", body["classes"]["terminal_sediment_tolerance_kg"][0]),
                struct.pack(">d", -0.0),
            )

    def test_actual_wire_scalar_is_finite_plain_float(self):
        value = self.actual["classes"]["terminal_sediment_tolerance_kg"]
        self.assertEqual(len(value), 1)
        self.assertIs(type(value[0]), float)
        self.assertTrue(math.isfinite(value[0]))


if __name__ == "__main__":
    if MODE in ("baseline", "native"):
        actual_receipt()
        print(json.dumps({"mode": MODE, "passed": True, "native_counts": COUNTS}))
    elif MODE == "checks":
        unittest.main(verbosity=2)
    else:
        raise SystemExit("unknown finite receipt check mode")
