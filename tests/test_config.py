"""Strict closed-schema configuration."""

from __future__ import annotations

import copy
import tomllib
import unittest

import _support

from masy.config import UNSUPPORTED_CALENDAR_FIELDS, ConfigError, load_config, parse_config

BASE = {
    "schema": "masy.preflight.v1",
    "sources": {"maple": {"root": "/home/okin/MAPLE"}, "syrup": {"root": "/home/okin/SYRUP"}},
}


def parse(raw: dict):
    return parse_config(raw, masy_root="/home/okin/MASY")


class ConfigTests(unittest.TestCase):
    def assertRefuses(self, raw: dict, code: str) -> None:
        with self.assertRaises(ConfigError) as ctx:
            parse(raw)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)

    def test_example_config_parses(self) -> None:
        path = _support.IMPLEMENTATION / "examples" / "preflight.example.toml"
        config = load_config(str(path), masy_root="/home/okin/MASY")
        self.assertEqual(config.upstream("maple").package, "maple")
        self.assertEqual(config.upstream("syrup").project_name, "maple-syrup")
        self.assertEqual(config.forcing_edges_s, ("0", "600", "1800.5", "3600"))
        self.assertEqual(config.budgets["wind_settlement_max_s"], "3600")

    def test_minimal_config_defaults(self) -> None:
        config = parse(copy.deepcopy(BASE))
        self.assertIsNone(config.case_id)
        self.assertIsNone(config.budgets)
        self.assertEqual(config.native_epoch_s, 0.0)

    def test_each_calendar_field_refuses_at_every_level(self) -> None:
        for name in sorted(UNSUPPORTED_CALENDAR_FIELDS):
            for table in (None, "forcing", "clock", "case"):
                raw = copy.deepcopy(BASE)
                if table is None:
                    raw[name] = "2026-10-17T00:00:00Z"
                else:
                    raw[table] = {"id": "x", "edges_s": ["0", "1"]} if table == "forcing" else (
                        {"id": "x"} if table == "case" else {})
                    raw[table][name] = "2026-10-17T00:00:00Z"
                with self.subTest(field=name, table=table):
                    self.assertRefuses(raw, "unsupported_calendar_field")

    def test_calendar_field_in_case_input_refuses(self) -> None:
        raw = copy.deepcopy(BASE)
        raw["case"] = {"id": "c", "inputs": [{"name": "t", "path": "/d/t.npz", "fixed_calendar": True}]}
        self.assertRefuses(raw, "unsupported_calendar_field")

    def test_dated_ids_and_paths_are_accepted(self) -> None:
        raw = copy.deepcopy(BASE)
        raw["case"] = {"id": "candidate-2026-10-17T06:00", "inputs": [
            {"name": "terrain_2026-10-17", "path": "/data/2026-10-17/update/terrain_20261017.npz"}]}
        raw["forcing"] = {"id": "storm-2026-10-17", "edges_s": ["0", "3600"]}
        config = parse(raw)
        self.assertEqual(config.case_id, "candidate-2026-10-17T06:00")
        self.assertEqual(config.case_inputs[0].path, "/data/2026-10-17/update/terrain_20261017.npz")

    def test_near_miss_calendar_names_are_unknown_not_calendar(self) -> None:
        for name in ("start_datetime_note", "calendar_id", "update"):
            raw = copy.deepcopy(BASE)
            raw["forcing"] = {"id": "f", "edges_s": ["0", "1"], name: "x"}
            with self.subTest(field=name):
                self.assertRefuses(raw, "unknown_config_field")

    def test_unknown_keys_refuse(self) -> None:
        for mutate in (
            lambda r: r.__setitem__("update", True),
            lambda r: r["sources"]["maple"].__setitem__("branch", "main"),
            lambda r: r["sources"].__setitem__("mahleran", {"root": "/x"}),
            lambda r: r.__setitem__("limits", {"max_file": 1}),
        ):
            raw = copy.deepcopy(BASE)
            mutate(raw)
            self.assertRefuses(raw, "unknown_config_field")

    def test_invalid_values_refuse(self) -> None:
        cases = [
            (lambda r: r.__setitem__("schema", "v0"), "invalid_config_value"),
            (lambda r: r["sources"]["maple"].__setitem__("root", "relative/MAPLE"), "invalid_config_value"),
            (lambda r: r["sources"]["maple"].__setitem__("expected_revision", "abc"), "invalid_config_value"),
            (lambda r: r["sources"].pop("syrup"), "missing_config_field"),
            (lambda r: r.__setitem__("limits", {"max_files": True}), "invalid_config_value"),
            (lambda r: r.__setitem__("limits", {"git_timeout_s": float("inf")}), "invalid_config_value"),
            (lambda r: r.__setitem__("budgets", {"wind_settlement_max_s": 3600.0}), "invalid_config_value"),
            (lambda r: r.__setitem__("budgets", {"wind_settlement_max_s": "0"}), "invalid_config_value"),
            (lambda r: r.__setitem__("clock", {"native_epoch_s": float("nan")}), "invalid_config_value"),
            (lambda r: r.__setitem__("forcing", {"id": "f", "edges_s": [0, 1]}), "invalid_config_value"),
        ]
        for mutate, code in cases:
            raw = copy.deepcopy(BASE)
            mutate(raw)
            self.assertRefuses(raw, code)

    def test_negative_native_epoch_refuses_and_valid_values_are_kept(self) -> None:
        for epoch in (-2.0, -1, -0.0, -5e-324):
            raw = copy.deepcopy(BASE)
            raw["clock"] = {"native_epoch_s": epoch}
            with self.subTest(epoch=epoch):
                self.assertRefuses(raw, "invalid_config_value")
        for epoch in (0.0, 1234.5678, 5e-324):
            raw = copy.deepcopy(BASE)
            raw["clock"] = {"native_epoch_s": epoch}
            with self.subTest(epoch=epoch):
                self.assertEqual(parse(raw).native_epoch_s.hex(), float(epoch).hex())

    def test_toml_with_dated_strings_round_trips(self) -> None:
        raw = tomllib.loads(
            'schema = "masy.preflight.v1"\n'
            '[sources.maple]\nroot = "/home/okin/MAPLE"\n[sources.syrup]\nroot = "/home/okin/SYRUP"\n'
            '[case]\nid = "2026-10-17"\n'
        )
        self.assertEqual(parse(raw).case_id, "2026-10-17")


if __name__ == "__main__":
    unittest.main()
