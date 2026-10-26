"""Affected stdlib metadata properties; no Native/model imports or execution."""
import dataclasses
import inspect
import unittest

from masy import dense_production as original
from masy import windowed_process as policy


class TestWindowedProcess(unittest.TestCase):
    def test_closed_rows_and_original_dense_domain(self):
        expected = {"ownership": (120, 135, 180), "two_events": (720, 735, 1080),
                    "four_events": (1440, 1470, 2160)}
        for name, values in expected.items():
            limits = policy.WindowedProductionLimits.for_profile(name)
            self.assertEqual(len(dataclasses.fields(limits)), 8)
            self.assertEqual((limits.cpu_soft_s, limits.cpu_hard_s, limits.controller_wall_s), values)
            self.assertEqual(limits.address_space_bytes, 4 << 30)
            self.assertTrue(limits.as_record()["controller_wall_is_declared_not_observed"])
        self.assertEqual(original.DenseProductionLimits().cpu_soft_s, 120)
        with self.assertRaises(original.DenseProductionError):
            original.DenseProductionLimits(cpu_soft_s=720)

    def test_unknown_bool_and_live_frozen_mutation_refuse(self):
        for name in (None, True, "unknown"):
            with self.assertRaises(original.DenseProductionError):
                policy.WindowedProductionLimits.for_profile(name)
        with self.assertRaises(original.DenseProductionError):
            policy.WindowedProductionLimits(cpu_soft_s=True)
        limits = policy.WindowedProductionLimits()
        object.__setattr__(limits, "cpu_hard_s", 1470)
        with self.assertRaises(original.DenseProductionError):
            limits.__post_init__()

    def test_original_code_exact_two_global_substitutions_and_signature(self):
        policy._check_binding()
        self.assertIs(policy._BOUND_CHECK.__code__, original.check_process.__code__)
        self.assertIsNot(policy._BOUND_CHECK.__globals__, vars(original))
        different = {key for key in vars(original)
                     if policy._BOUND_CHECK.__globals__[key] is not vars(original)[key]}
        self.assertEqual(different, {"DenseProductionLimits", "SCOPE"})
        self.assertEqual(tuple(inspect.signature(policy.check_process).parameters),
                         tuple(inspect.signature(original.check_process).parameters))
        self.assertIs(policy.DenseProductionError, original.DenseProductionError)
        terms = policy.binding_metadata_terms()
        self.assertTrue(all(type(value) is int and value > 0 for _, value in terms))

    def test_owned_binding_mutation_refuses_without_original_repair(self):
        before = tuple(vars(original).items())
        scope = policy._BOUND_GLOBALS["SCOPE"]
        try:
            policy._BOUND_GLOBALS["SCOPE"] = "authored mutation"
            with self.assertRaisesRegex(original.DenseProductionError, "binding_changed"):
                policy._check_binding()
            self.assertEqual(tuple(vars(original).items()), before)
            self.assertEqual(policy._BOUND_GLOBALS["SCOPE"], "authored mutation")
        finally:
            policy._BOUND_GLOBALS["SCOPE"] = scope
        policy._check_binding()


if __name__ == "__main__":
    unittest.main()
