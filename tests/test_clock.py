"""Exact clock ledger and forcing-edge resolution."""

from __future__ import annotations

import json
import unittest
from fractions import Fraction

import _support  # noqa: F401 - puts MASY src on sys.path

from masy.clock import ClockError, ClockLedger, decode_time, encode_time, resolve_forcing_edges

T0 = 1234.5678  # nonzero native epoch, not a short binary fraction


class ClockLedgerTests(unittest.TestCase):
    def assertClockError(self, code: str, call, *args, **kwargs) -> None:
        with self.assertRaises(ClockError) as ctx:
            call(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)

    def contiguous(self) -> ClockLedger:
        ledger = ClockLedger(T0)
        wind_end = T0 + 600.1
        ledger.append("event", native_start=T0, native_end=wind_end, commit_before=T0 - 10.0,
                      commit_after=wind_end, returned_elapsed=600.1)
        idle_end = wind_end + 0.3
        ledger.append("idle", native_start=wind_end, native_end=idle_end, commit_before=wind_end,
                      commit_after=wind_end)
        ledger.append("event", native_start=idle_end, native_end=idle_end + 7200.0, commit_before=wind_end,
                      commit_after=idle_end + 7000.0)
        return ledger

    def test_nonzero_epoch_has_zero_global_elapsed_and_exact_inverse(self) -> None:
        ledger = ClockLedger(T0)
        self.assertEqual(ledger.global_elapsed(T0), 0)
        t = T0 + 0.1
        g = ledger.global_elapsed(t)
        self.assertEqual(g, Fraction(t) - Fraction(T0))
        self.assertEqual(ledger.native_from_global(g).hex(), t.hex())

    def test_global_value_without_native_image_refuses(self) -> None:
        self.assertClockError("not_a_native_value", ClockLedger(T0).native_from_global, Fraction(1, 3))

    def test_contiguous_entries_pass_and_preserve_native_values(self) -> None:
        records = self.contiguous().records()
        self.assertEqual([r["kind"] for r in records], ["event", "idle", "event"])
        self.assertEqual(records[0]["global_start"], "0/1")
        self.assertEqual(records[1]["native_start"], records[0]["native_end"])
        self.assertEqual(decode_time(records[0]["native_end"]).hex(), (T0 + 600.1).hex())

    def test_returned_elapsed_discrepancy_is_recorded_not_corrected(self) -> None:
        first = self.contiguous().records()[0]
        endpoint = Fraction(T0 + 600.1) - Fraction(T0)
        self.assertEqual(first["endpoint_elapsed"], f"{endpoint.numerator}/{endpoint.denominator}")
        discrepancy = Fraction(600.1) - endpoint
        self.assertEqual(first["endpoint_discrepancy"], f"{discrepancy.numerator}/{discrepancy.denominator}")
        self.assertEqual(decode_time(first["native_end"]).hex(), (T0 + 600.1).hex())

        ledger = ClockLedger(T0)
        record = ledger.append("event", native_start=T0, native_end=T0 + 10.0, commit_before=T0,
                               commit_after=T0 + 10.0, returned_elapsed=9.5)
        self.assertEqual(record["endpoint_discrepancy"], "-1/2")
        self.assertEqual(record["endpoint_elapsed"], "10/1")
        self.assertEqual(decode_time(record["native_end"]).hex(), (T0 + 10.0).hex())

    def test_first_entry_must_start_at_epoch(self) -> None:
        self.assertClockError("non_contiguous_interval", ClockLedger(T0).append, "event", native_start=T0 + 1.0,
                              native_end=T0 + 2.0, commit_before=T0, commit_after=T0 + 2.0)

    def test_gap_without_idle_entry_refuses(self) -> None:
        ledger = ClockLedger(T0)
        ledger.append("event", native_start=T0, native_end=T0 + 10.0, commit_before=T0, commit_after=T0 + 10.0)
        self.assertClockError("non_contiguous_interval", ledger.append, "event", native_start=T0 + 11.0,
                              native_end=T0 + 20.0, commit_before=T0 + 10.0, commit_after=T0 + 20.0)

    def test_one_ulp_mismatch_is_not_contiguous(self) -> None:
        import math
        ledger = ClockLedger(T0)
        end = T0 + 0.1
        ledger.append("event", native_start=T0, native_end=end, commit_before=T0, commit_after=end)
        self.assertClockError("non_contiguous_interval", ledger.append, "idle", native_start=math.nextafter(end, 0),
                              native_end=end + 1.0, commit_before=end, commit_after=end)

    def test_reversed_interval_refuses(self) -> None:
        self.assertClockError("reversed_interval", ClockLedger(T0).append, "event", native_start=T0,
                              native_end=T0 - 1.0, commit_before=T0 - 2.0, commit_after=T0 - 2.0)

    def test_future_commit_refuses(self) -> None:
        self.assertClockError("future_commit", ClockLedger(T0).append, "event", native_start=T0,
                              native_end=T0 + 10.0, commit_before=T0, commit_after=T0 + 10.5)
        self.assertClockError("future_commit", ClockLedger(T0).append, "event", native_start=T0,
                              native_end=T0 + 10.0, commit_before=T0 + 1.0, commit_after=T0 + 5.0)

    def test_commit_stamps_must_be_monotone_and_chained(self) -> None:
        self.assertClockError("non_monotone_commit", ClockLedger(T0).append, "event", native_start=T0,
                              native_end=T0 + 10.0, commit_before=T0, commit_after=T0 - 1.0)
        ledger = ClockLedger(T0)
        ledger.append("event", native_start=T0, native_end=T0 + 10.0, commit_before=T0, commit_after=T0 + 9.0)
        self.assertClockError("non_monotone_commit", ledger.append, "event", native_start=T0 + 10.0,
                              native_end=T0 + 20.0, commit_before=T0 + 5.0, commit_after=T0 + 20.0)

    def test_idle_cannot_commit(self) -> None:
        ledger = ClockLedger(T0)
        self.assertClockError("idle_commit", ledger.append, "idle", native_start=T0, native_end=T0 + 5.0,
                              commit_before=T0, commit_after=T0 + 1.0)

    def test_invalid_values_refuse(self) -> None:
        for bad in (True, float("nan"), float("inf"), "1.0", 2 ** 60 + 1):
            with self.subTest(value=bad):
                self.assertClockError("invalid_time_value", ClockLedger(T0).append, "event", native_start=T0,
                                      native_end=bad, commit_before=T0, commit_after=T0)
        self.assertClockError("invalid_time_value", ClockLedger, False)
        self.assertClockError("invalid_entry_kind", ClockLedger(T0).append, "wind", native_start=T0,
                              native_end=T0, commit_before=T0, commit_after=T0)

    def test_serialized_ledger_round_trips_exactly(self) -> None:
        ledger = self.contiguous()
        record = json.loads(json.dumps(ledger.as_record()))
        rebuilt = ClockLedger.from_record(record)
        self.assertEqual(rebuilt.as_record(), ledger.as_record())

    def test_tampered_serialized_ledger_refuses(self) -> None:
        record = json.loads(json.dumps(self.contiguous().as_record()))
        record["entries"][1]["global_end"] = "0/1"
        self.assertClockError("inconsistent_ledger_record", ClockLedger.from_record, record)
        record = json.loads(json.dumps(self.contiguous().as_record()))
        record["entries"][0]["native_end"]["fraction"] = "1/1"
        self.assertClockError("inconsistent_time_record", ClockLedger.from_record, record)

    def test_negative_epoch_times_and_commits_refuse(self) -> None:
        self.assertClockError("negative_time_value", ClockLedger, -2.0)
        self.assertClockError("negative_time_value", ClockLedger, -1)
        self.assertClockError("negative_zero_time", ClockLedger, -0.0)
        self.assertClockError("negative_time_value", ClockLedger(0.0).append, "event", native_start=0.0,
                              native_end=1.0, commit_before=-1.0, commit_after=1.0)
        self.assertClockError("negative_zero_time", ClockLedger(0.0).append, "event", native_start=0.0,
                              native_end=1.0, commit_before=-0.0, commit_after=1.0)
        self.assertClockError("negative_time_value", ClockLedger(0.0).append, "event", native_start=0.0,
                              native_end=1.0, commit_before=0.0, commit_after=1.0, returned_elapsed=-1.0)

    def test_valid_values_are_preserved_bit_for_bit(self) -> None:
        ledger = ClockLedger(0.0)
        record = ledger.append("event", native_start=0.0, native_end=5e-324, commit_before=0.0,
                               commit_after=5e-324, returned_elapsed=0.1)
        self.assertEqual(record["native_end"], {"hex": (5e-324).hex(), "fraction": "1/" + str(2 ** 1074)})
        self.assertEqual(record["returned_elapsed"]["hex"], (0.1).hex())
        self.assertEqual(ledger.native_from_global(ledger.global_elapsed(0.0)).hex(), (0.0).hex())

    def test_conversions_refuse_negative_native_values(self) -> None:
        ledger = ClockLedger(100.0)
        self.assertClockError("negative_time_value", ledger.global_elapsed, -1.0)
        self.assertClockError("negative_zero_time", ledger.global_elapsed, -0.0)
        self.assertClockError("negative_time_value", ledger.native_from_global, Fraction(-101))
        self.assertClockError("negative_time_value", ClockLedger(0.0).native_from_global, Fraction(-1, 2 ** 1074))

    def test_negative_global_offset_maps_to_earlier_native_time(self) -> None:
        ledger = ClockLedger(100.0)
        self.assertEqual(ledger.native_from_global(Fraction(-75)).hex(), (25.0).hex())
        for t in (25.0, 0.0, 5e-324, 99.99999999999999):
            with self.subTest(t=t):
                g = ledger.global_elapsed(t)
                self.assertLess(g, 0)
                self.assertEqual(ledger.native_from_global(g).hex(), t.hex())

    def test_positive_zero_round_trips_with_nonzero_epoch(self) -> None:
        for epoch in (100.0, T0):
            with self.subTest(epoch=epoch):
                ledger = ClockLedger(epoch)
                g = ledger.global_elapsed(0.0)
                self.assertEqual(g, -Fraction(epoch))
                self.assertEqual(ledger.native_from_global(g).hex(), (0.0).hex())

    def test_codec_still_encodes_negative_values_losslessly(self) -> None:
        for value in (-0.0, -1.0, -5e-324):
            with self.subTest(value=value):
                self.assertEqual(decode_time(encode_time(value)).hex(), value.hex())

    def test_redefined_global_elapsed_definition_refuses(self) -> None:
        record = json.loads(json.dumps(self.contiguous().as_record()))
        record["global_elapsed_definition"] = "native_t * 1000"
        self.assertClockError("unsupported_global_elapsed_definition", ClockLedger.from_record, record)
        record = json.loads(json.dumps(self.contiguous().as_record()))
        record["schema"] = "masy.clock_ledger.v2"
        self.assertClockError("unsupported_ledger_schema", ClockLedger.from_record, record)

    def test_malformed_serialized_ledger_raises_clock_error(self) -> None:
        good = self.contiguous().as_record()

        def mutated(change):
            record = json.loads(json.dumps(good))
            change(record)
            return record

        cases = [
            ("malformed_ledger_record", None),
            ("malformed_ledger_record", []),
            ("malformed_ledger_record", mutated(lambda r: r.pop("schema"))),
            ("malformed_ledger_record", mutated(lambda r: r.__setitem__("note", "x"))),
            ("malformed_ledger_record", mutated(lambda r: r.__setitem__("entries", {}))),
            ("malformed_ledger_record", mutated(lambda r: r["entries"][0].pop("index"))),
            ("malformed_ledger_record", mutated(lambda r: r["entries"][0].__setitem__("extra", 1))),
            ("malformed_time_record", mutated(lambda r: r.__setitem__("native_epoch", "1234.5678"))),
            ("malformed_time_record", mutated(lambda r: r["native_epoch"].__setitem__("hex", "zz"))),
            ("malformed_time_record", mutated(lambda r: r["native_epoch"].__setitem__("hex", "inf"))),
            ("malformed_time_record", mutated(lambda r: r["native_epoch"].__setitem__("hex", "0x1.0p+0"))),
            ("invalid_entry_kind", mutated(lambda r: r["entries"][0].__setitem__("kind", ["event"]))),
            ("inconsistent_ledger_record", mutated(lambda r: r["entries"][1].__setitem__("index", True))),
            ("inconsistent_ledger_record", mutated(lambda r: r["entries"][0].__setitem__("endpoint_elapsed", "1/1"))),
        ]
        for code, record in cases:
            with self.subTest(code=code, record=str(record)[:80]):
                self.assertClockError(code, ClockLedger.from_record, record)

    def test_signed_zero_is_refused_not_reinterpreted(self) -> None:
        record = json.loads(json.dumps(ClockLedger(0.0).as_record()))
        record["native_epoch"] = {"hex": "-0x0.0p+0", "fraction": "0/1"}
        self.assertClockError("negative_zero_time", ClockLedger.from_record, record)

    def test_time_encoding_round_trip(self) -> None:
        for value in (0.0, -0.0, 0.1, T0, 1e300, 5e-324):
            with self.subTest(value=value):
                self.assertEqual(decode_time(encode_time(value)).hex(), value.hex())


class ForcingTests(unittest.TestCase):
    def test_ordinary_decimals_keep_intended_and_native_records(self) -> None:
        result = resolve_forcing_edges(0.2, ["0", "0.1", "0.3", "600"])
        edges = result["edges"]
        self.assertEqual([e["intended_decimal"] for e in edges], ["0", "0.1", "0.3", "600"])
        self.assertEqual(edges[1]["intended_fraction"], "1/10")
        self.assertEqual(decode_time(edges[1]["native"]).hex(), (0.2 + 0.1).hex())  # 0.30000000000000004
        offset = Fraction(0.2 + 0.1) - (Fraction(0.2) + Fraction(1, 10))
        self.assertNotEqual(offset, 0)
        self.assertEqual(edges[1]["native_minus_intended"], f"{offset.numerator}/{offset.denominator}")
        self.assertEqual(result["adapter_status"], "forcing_adapter_unqualified")

    def test_no_rescaling_of_large_edges(self) -> None:
        result = resolve_forcing_edges(T0, ["0", "86400", "172800.25"])
        self.assertEqual(decode_time(result["edges"][2]["native"]).hex(), (T0 + 172800.25).hex())

    def test_collapsed_reversed_and_invalid_edges_refuse(self) -> None:
        cases = [
            (["0", "1", "1"], "collapsed_forcing_edges"),
            (["0", "1", "1.0"], "collapsed_forcing_edges"),
            (["0", "2", "1"], "reversed_forcing_edges"),
            (["0", "nan"], "invalid_forcing_edges"),
            (["0", "inf"], "invalid_forcing_edges"),
            (["0", "-1"], "invalid_forcing_edges"),
            (["0", "1e3"], "invalid_forcing_edges"),
            (["0", 1.0], "invalid_forcing_edges"),
            (["0"], "invalid_forcing_edges"),
        ]
        for edges, code in cases:
            with self.subTest(edges=edges):
                with self.assertRaises(ClockError) as ctx:
                    resolve_forcing_edges(0.0, edges)
                self.assertEqual(ctx.exception.code, code)

    def test_edges_that_collapse_only_natively_refuse(self) -> None:
        with self.assertRaises(ClockError) as ctx:
            resolve_forcing_edges(1e17, ["0", "1"])
        self.assertEqual(ctx.exception.code, "collapsed_forcing_edges")

    def test_non_finite_native_start_refuses(self) -> None:
        with self.assertRaises(ClockError):
            resolve_forcing_edges(float("nan"), ["0", "1"])

    def test_first_positive_edge_collapsing_onto_origin_refuses(self) -> None:
        with self.assertRaises(ClockError) as ctx:
            resolve_forcing_edges(float(2 ** 53), ["1", "2"])
        self.assertEqual(ctx.exception.code, "collapsed_forcing_edges")
        self.assertIn("origin", ctx.exception.detail)
        result = resolve_forcing_edges(float(2 ** 53), ["2", "4"])  # representable: accepted unchanged
        self.assertEqual(decode_time(result["edges"][0]["native"]).hex(), float(2 ** 53 + 2).hex())
        result = resolve_forcing_edges(float(2 ** 53), ["0", "2"])  # edge 0 at the origin is not a collapse
        self.assertEqual(result["edges"][0]["native"], encode_time(float(2 ** 53)))

    def test_first_inexact_decimal_edge_is_still_accepted(self) -> None:
        result = resolve_forcing_edges(0.2, ["0.1", "0.3"])
        self.assertEqual(decode_time(result["edges"][0]["native"]).hex(), (0.2 + 0.1).hex())

    def test_negative_native_start_refuses(self) -> None:
        for start, code in ((-1.0, "negative_time_value"), (-0.0, "negative_zero_time")):
            with self.subTest(start=start):
                with self.assertRaises(ClockError) as ctx:
                    resolve_forcing_edges(start, ["0", "1"])
                self.assertEqual(ctx.exception.code, code)


if __name__ == "__main__":
    unittest.main()
