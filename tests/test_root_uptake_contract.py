"""Reference arithmetic of the agreed plant-water contract (stdlib Fractions; not an adapter test)."""

from __future__ import annotations

import unittest
from fractions import Fraction

import _support  # noqa: F401 - puts MASY src on sys.path

from _root_uptake_reference import ContractError, allocate_uptake, depth_to_volume_m3, volume_to_depth_m


class RootUptakeContractTests(unittest.TestCase):
    def test_user_example_single_plant_two_layers(self) -> None:
        result = allocate_uptake({"p": 100}, {"p": {("c", 0): 0.5, ("c", 1): 0.5}}, {("c", 0): 25, ("c", 1): 200})
        self.assertEqual(result.requests["p"], {("c", 0): 50, ("c", 1): 50})
        self.assertEqual(result.fulfillment, {("c", 0): Fraction(1, 2), ("c", 1): 1})
        self.assertEqual(result.accepted["p"], {("c", 0): 25, ("c", 1): 50})
        self.assertEqual(result.remaining, {("c", 0): 0, ("c", 1): 150})
        self.assertEqual(result.shortfall, {("c", 0): 25, ("c", 1): 0})
        self.assertEqual(result.growth_fraction["p"], Fraction(3, 4))  # root-weighted, not min(0.5, 1.0)
        self.assertEqual(sum(result.debited.values()), 75)  # water closes independently of the growth factor

    def test_two_plants_two_cells_share_one_local_factor_without_reallocation(self) -> None:
        # Same case as the task's root_uptake_contract_probe.py (NumPy/Numba), here in exact rationals.
        F = Fraction  # horizontal x vertical root weights, exact (binary64 products such as 0.75*0.8 are inexact)
        weights = {"a": {("c0", 0): F(3, 4) * F(1, 2), ("c0", 1): F(3, 4) * F(1, 2), ("c1", 0): F(1, 4) * F(1, 2), ("c1", 1): F(1, 4) * F(1, 2)},
                   "b": {("c0", 0): F(1, 4) * F(4, 5), ("c0", 1): F(1, 4) * F(1, 5), ("c1", 0): F(3, 4) * F(4, 5), ("c1", 1): F(3, 4) * F(1, 5)}}
        available = {("c0", 0): 38.75, ("c0", 1): 95.0, ("c1", 0): 132.5, ("c1", 1): 0.0}
        result = allocate_uptake({"a": 100, "b": 200}, weights, available)
        self.assertEqual(result.local_demand, {("c0", 0): Fraction(155, 2), ("c0", 1): Fraction(95, 2), ("c1", 0): Fraction(265, 2), ("c1", 1): Fraction(85, 2)})
        self.assertEqual(result.fulfillment, {("c0", 0): Fraction(1, 2), ("c0", 1): 1, ("c1", 0): 1, ("c1", 1): 0})
        self.assertEqual(sum(result.accepted["a"].values()), Fraction(275, 4))  # 68.75
        self.assertEqual(sum(result.accepted["b"].values()), 150)
        self.assertEqual(result.growth_fraction, {"a": Fraction(11, 16), "b": Fraction(3, 4)})  # 0.6875, 0.75
        self.assertEqual(result.remaining, {("c0", 0): 0, ("c0", 1): Fraction(95, 2), ("c1", 0): 0, ("c1", 1): 0})
        floats = result.as_floats()
        self.assertEqual(floats["growth_fraction"], {"a": 0.6875, "b": 0.75})
        for loc, taken in result.debited.items():
            self.assertLessEqual(taken, Fraction(available[loc]))

    def test_zero_demand_and_contract_violations(self) -> None:
        result = allocate_uptake({"p": 0}, {"p": {("c", 0): 1}}, {("c", 0): 10})
        self.assertEqual(result.accepted["p"], {("c", 0): 0})
        self.assertIsNone(result.growth_fraction["p"])  # explicit: no invented growth, no division by zero
        self.assertEqual(result.remaining[("c", 0)], 10)
        for demands, weights, available, code in (
            ({"p": 1}, {"p": {("c", 0): 0.6}}, {("c", 0): 1}, "weights_not_normalized"),
            ({"p": 1}, {"p": {("c", 0): 1}}, {("d", 0): 1}, "unreported_location"),
            ({"p": 1}, {}, {("c", 0): 1}, "missing_weights"),
            ({"p": 1}, {"p": {}}, {("c", 0): 1}, "no_roots"),
            ({"p": -1}, {"p": {("c", 0): 1}}, {("c", 0): 1}, "invalid_value"),
            ({"p": 1}, {"p": {("c", 0): 1}}, {("c", 0): -1}, "invalid_value"),
            ({"p": True}, {"p": {("c", 0): 1}}, {("c", 0): 1}, "invalid_value"),
        ):
            with self.subTest(code=code):
                with self.assertRaises(ContractError) as ctx:
                    allocate_uptake(demands, weights, available)
                self.assertEqual(ctx.exception.code, code)
        # weights that sum to one only within a declared tolerance
        with self.assertRaises(ContractError):
            allocate_uptake({"p": 1}, {"p": {("c", 0): 0.1, ("c", 1): 0.2, ("c", 2): 0.7}}, {("c", i): 1 for i in range(3)})
        allocate_uptake({"p": 1}, {"p": {("c", 0): 0.1, ("c", 1): 0.2, ("c", 2): 0.7}}, {("c", i): 1 for i in range(3)},
                        weight_tolerance=Fraction(1, 10 ** 12))

    def test_explicit_depth_volume_conversion(self) -> None:
        self.assertEqual(depth_to_volume_m3(0.5, 2.0), 1)
        self.assertEqual(volume_to_depth_m(1, 2.0), Fraction(1, 2))
        with self.assertRaises(ContractError):
            depth_to_volume_m3(0.5, 0.0)


if __name__ == "__main__":
    unittest.main()
