"""Actual scalar refusal and failure-cleanup controls; no hardware/admission is fabricated."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest
import weakref

from masy import dense_device_production as production
from masy import dense_device_runner as device
from masy import dense_production

GIB = 1 << 30
FLOOR_DETAIL = "driver free memory below min_device_free_bytes"


def limits():
    return production.DeviceProductionLimits(fft_workspace_reserve_bytes=1 << 20,
        jit_driver_reserve_bytes=1 << 20, native_transient_reserve_bytes=1 << 20,
        diagnostic_scratch_bytes=4096)


def observed(**changes):
    # Labelled synthetic scalar values only: not a process or device observation.
    values = dict(free=9 * GIB, total=11 * GIB, used=0, held=0, cap=8 * GIB,
                  upcoming=4 * GIB, checkpoint="constructing")
    values.update(changes)
    return values


def sanitized_floor_failure():
    # Return only primitive data; the original exception/traceback leaves scope.
    try:
        production._check_observed_device_memory(limits(), **observed())
    except production.DenseProductionError as exc:
        return exc.code, exc.detail, production._copy_device_memory_refusal(exc)
    raise AssertionError("the actual scalar floor predicate must refuse")


def error_with(data, *, code="device_free_floor"):
    error = production.DenseProductionError(code, FLOOR_DETAIL)
    error._device_memory_refusal = data
    return error


def module_tree(module):
    return ast.parse(Path(module.__file__).read_bytes())


def function(module, name):
    return next(node for node in module_tree(module).body if isinstance(node, ast.FunctionDef) and node.name == name)


class ScalarPredicates(unittest.TestCase):
    def refusal(self, code, detail, **changes):
        with self.assertRaises(production.DenseProductionError) as caught:
            production._check_observed_device_memory(limits(), **observed(**changes))
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.detail, detail)
        self.assertEqual(caught.exception.args, (f"{code}: {detail}",))
        return caught.exception

    def test_floor_contains_exact_same_scalar_data(self):
        code, detail, data = sanitized_floor_failure()
        self.assertEqual((code, detail), ("device_free_floor", FLOOR_DETAIL))
        self.assertIs(type(data), tuple)
        self.assertEqual(data, (9 * GIB, 11 * GIB, 0, 0, 8 * GIB, 4 * GIB, 10 * GIB, 8 * GIB))
        self.assertTrue(all(type(value) is int for value in data))

    def test_alias_is_actual_error_class(self):
        self.assertIs(production.DenseProductionError, dense_production.DenseProductionError)

    def test_exact_floor_pass_is_only_scalar_predicate_pass(self):
        self.assertIsNone(production._check_observed_device_memory(limits(), **observed(free=10 * GIB)))

    def test_floor_not_reapplied_during_work(self):
        self.assertIsNone(production._check_observed_device_memory(limits(), **observed(checkpoint="wrapper_work")))

    def test_counter_type_and_bounds_precede_budget_predicates(self):
        for value in (True, -1, (1 << 40) + 1, 1.0, None):
            with self.subTest(value_type=type(value).__name__):
                error = self.refusal("device_observation", "device memory counters are not bounded consistent ints",
                                     free=value, held=9 * GIB, cap=0)
                self.assertIsNone(production._copy_device_memory_refusal(error))

    def test_counter_relations_precede_pool_budget(self):
        self.refusal("device_observation", "device memory counters are not bounded consistent ints", total=8 * GIB)
        self.refusal("device_observation", "device memory counters are not bounded consistent ints", used=1, held=0)

    def test_pool_held_precedes_cap_and_prospective_budget(self):
        self.refusal("device_pool_limit", "pool held bytes exceed max_device_bytes",
                     used=9 * GIB, held=9 * GIB, cap=0)

    def test_pool_cap_precedes_prospective_budget(self):
        self.refusal("device_pool_limit", "default pool hard limit must equal max_device_bytes",
                     used=8 * GIB, held=8 * GIB, cap=7 * GIB)

    def test_pool_budget_precedes_current_free(self):
        self.refusal("device_budget", "pool used plus upcoming bytes exceeds max_device_bytes",
                     free=GIB, used=8 * GIB, held=8 * GIB)

    def test_current_free_precedes_floor(self):
        self.refusal("device_budget", "upcoming bytes exceed current driver free memory", free=3 * GIB)

    def test_nonfloor_refusals_have_no_optional_tuple(self):
        error = self.refusal("device_pool_limit", "default pool hard limit must equal max_device_bytes", cap=0)
        self.assertNotIn("_device_memory_refusal", error.__dict__)
        self.assertIsNone(production._copy_device_memory_refusal(error))


class OptionalData(unittest.TestCase):
    def test_genuine_scalar_error_returns_same_immutable_tuple(self):
        data = sanitized_floor_failure()[2]
        self.assertIs(production._copy_device_memory_refusal(error_with(data)), data)

    def test_wrong_error_class_code_and_missing_data_are_unavailable(self):
        class Foreign:
            @property
            def __dict__(self):
                raise AssertionError("foreign getter must not be read")
        class Derived(production.DenseProductionError):
            pass
        data = sanitized_floor_failure()[2]
        for error in (Foreign(), ValueError("device_free_floor"), Derived("device_free_floor", FLOOR_DETAIL),
                      error_with(data, code="device_budget"), production.DenseProductionError("device_free_floor", FLOOR_DETAIL)):
            with self.subTest(error_type=type(error).__name__):
                self.assertIsNone(production._copy_device_memory_refusal(error))

    def test_shape_type_and_range_refuse_without_coercion(self):
        class ForeignInt(int):
            def __le__(self, other):
                raise AssertionError("foreign integer comparison")
        class ForeignTuple(tuple):
            def __len__(self):
                raise AssertionError("foreign tuple length")
        data = sanitized_floor_failure()[2]
        cases = (None, list(data), data[:-1], data + (0,), ForeignTuple(data),
                 (True,) + data[1:], (ForeignInt(data[0]),) + data[1:],
                 (-1,) + data[1:], ((1 << 40) + 1,) + data[1:])
        for value in cases:
            with self.subTest(value_type=type(value).__name__):
                self.assertIsNone(production._copy_device_memory_refusal(error_with(value)))

    def test_foreign_error_dictionary_is_not_called(self):
        class ForeignDict(dict):
            def get(self, *args):
                raise AssertionError("foreign dictionary getter")
        error = error_with(sanitized_floor_failure()[2])
        error.__dict__ = ForeignDict(error.__dict__)
        self.assertIsNone(production._copy_device_memory_refusal(error))

    def test_all_data_relations_must_hold(self):
        data = sanitized_floor_failure()[2]
        changes = ({0: 12 * GIB}, {2: 1}, {3: 9 * GIB}, {4: 0},
                   {2: 8 * GIB, 3: 8 * GIB}, {0: 3 * GIB}, {0: 10 * GIB})
        for edits in changes:
            value = tuple(edits.get(index, item) for index, item in enumerate(data))
            with self.subTest(indices=tuple(edits)):
                self.assertIsNone(production._copy_device_memory_refusal(error_with(value)))

    def test_foreign_code_is_not_compared(self):
        class Foreign:
            def __eq__(self, other):
                raise AssertionError("foreign code comparison")
        error = error_with(sanitized_floor_failure()[2]); error.code = Foreign()
        self.assertIsNone(production._copy_device_memory_refusal(error))


class Cleanup(unittest.TestCase):
    def test_actual_returning_helper_releases_owned_cpu_sentinel(self):
        class Sentinel:
            pass
        sentinel = Sentinel(); reference = weakref.ref(sentinel); made = {"owned": sentinel}
        del sentinel
        failure = sanitized_floor_failure()
        error = device._production_device_failure(made, failure)
        self.assertEqual(made, {})
        self.assertIsNone(reference())
        self.assertIs(type(error), device.DenseRunnerError)
        self.assertEqual((error.code, error.detail), failure[:2])
        self.assertEqual(error.args, (f"{failure[0]}: {failure[1]}",))
        self.assertIs(error._device_memory_refusal, failure[2])
        self.assertEqual(set(error.__dict__), {"code", "detail", "_device_memory_refusal"})
        self.assertIsNone(error.__traceback__)
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)

    def test_unavailable_diagnostic_still_cleans_and_preserves_refusal(self):
        code, detail, _ = sanitized_floor_failure(); made = {"owned": object()}
        error = device._production_device_failure(made, (code, detail, None))
        self.assertEqual(made, {})
        self.assertEqual((error.code, error.detail), (code, detail))
        self.assertNotIn("_device_memory_refusal", error.__dict__)

    def test_generic_failure_keeps_its_original_code_detail(self):
        error = device._production_device_failure({}, ("device_construction_failure", "synthetic CPU detail", None))
        self.assertEqual(error.args, ("device_construction_failure: synthetic CPU detail",))
        self.assertEqual(error.__dict__, {"code": "device_construction_failure", "detail": "synthetic CPU detail"})

    def test_raise_after_return_has_no_original_context_or_helper_frame(self):
        error = device._production_device_failure({}, sanitized_floor_failure())
        try:
            raise error
        except device.DenseRunnerError as raised:
            self.assertIsNone(raised.__cause__)
            self.assertIsNone(raised.__context__)
            names = []
            trace = raised.__traceback__
            while trace is not None:
                names.append(trace.tb_frame.f_code.co_name); trace = trace.tb_next
            self.assertEqual(names, ["test_raise_after_return_has_no_original_context_or_helper_frame"])


class SourceWiring(unittest.TestCase):
    def test_primary_assignment_and_optional_copy_are_guarded_in_original_handler(self):
        factory = function(device, "_create_dense_production_device_runner")
        outer = next(node for node in factory.body if isinstance(node, ast.Try))
        handler = outer.handlers[0]
        self.assertEqual(ast.unparse(handler.body[0]), "failure = (exc.code, exc.detail, None)")
        optional = handler.body[1]
        self.assertIsInstance(optional, ast.Try)
        self.assertEqual(len(optional.body), 1)
        self.assertEqual(ast.unparse(optional.body[0]),
                         "failure = (failure[0], failure[1], production._copy_device_memory_refusal(exc))")
        self.assertEqual(ast.unparse(optional.handlers[0].type), "Exception")
        self.assertIsNone(optional.handlers[0].name)
        self.assertEqual(len(optional.handlers[0].body), 1)
        self.assertIsInstance(optional.handlers[0].body[0], ast.Pass)
        self.assertNotIn("_copy_device_memory_refusal", ast.unparse(outer.handlers[1]))

    def test_factory_raises_returned_error_outside_original_handlers(self):
        factory = function(device, "_create_dense_production_device_runner")
        final_if = factory.body[-2]
        self.assertIsInstance(final_if, ast.If)
        self.assertEqual(ast.unparse(final_if.test), "failure is not None")
        self.assertEqual(ast.unparse(final_if.body[0]), "raise _production_device_failure(made, failure)")
        self.assertEqual(len(final_if.body), 1)

    def test_cleanup_has_only_two_parameters_clears_first_and_never_raises(self):
        helper = function(device, "_production_device_failure")
        self.assertEqual([arg.arg for arg in helper.args.args], ["made", "failure"])
        self.assertEqual(ast.unparse(helper.body[0]), "made.clear()")
        self.assertEqual(ast.unparse(helper.body[1]), "error = DenseRunnerError(failure[0], failure[1])")
        self.assertEqual(ast.unparse(helper.body[-1]), "return error")
        self.assertFalse(any(isinstance(node, ast.Raise) for node in ast.walk(helper)))
        optional = helper.body[2].body[0]
        self.assertIsInstance(optional, ast.Try)
        self.assertEqual(ast.unparse(optional.body[0]), "error._device_memory_refusal = failure[2]")
        self.assertEqual(ast.unparse(optional.handlers[0].type), "Exception")
        self.assertIsNone(optional.handlers[0].name)
        self.assertEqual(len(optional.handlers[0].body), 1)
        self.assertIsInstance(optional.handlers[0].body[0], ast.Pass)

    def test_original_observation_sites_precede_one_shared_predicate_call(self):
        check = function(production, "check_process")
        calls = [(node.lineno, ast.unparse(node.func)) for node in ast.walk(check) if isinstance(node, ast.Call)]
        for name in ("cp.cuda.runtime.memGetInfo", "cp.get_default_memory_pool", "pool.used_bytes",
                     "pool.total_bytes", "pool.get_limit", "_check_observed_device_memory"):
            self.assertEqual(sum(callee == name for _, callee in calls), 1)
        predicate = next(line for line, callee in calls if callee == "_check_observed_device_memory")
        self.assertTrue(all(line < predicate for line, callee in calls if callee.startswith(("cp.", "pool."))))
        self.assertIsInstance(check.body[-1], ast.Return)
        self.assertEqual(len(check.body[-1].value.keys), 16)

    def test_only_floor_predicate_attaches_optional_primitive_data(self):
        helper = function(production, "_check_observed_device_memory")
        predicates = [node for node in helper.body if isinstance(node, ast.If)]
        self.assertEqual(len(predicates), 6)
        self.assertFalse(any("_device_memory_refusal" in ast.unparse(node) for node in predicates[:-1]))
        floor = predicates[-1]
        self.assertEqual(ast.unparse(floor.body[0]),
                         "error = DenseProductionError('device_free_floor', 'driver free memory below min_device_free_bytes')")
        optional = floor.body[1]
        self.assertIsInstance(optional, ast.Try)
        self.assertEqual(ast.unparse(optional.handlers[0].type), "Exception")
        self.assertIsNone(optional.handlers[0].name)
        self.assertEqual(ast.unparse(floor.body[-1]), "raise error")
        self.assertFalse(any(isinstance(node, ast.ExceptHandler) and ast.unparse(node.type) == "BaseException"
                             for node in ast.walk(helper)))


if __name__ == "__main__":
    unittest.main()
