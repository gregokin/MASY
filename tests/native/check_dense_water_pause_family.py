"""Negative metadata/order checks for one explicit water resource family.

Every replacement of _runtime_binding is an authored REFUSAL-path witness.
These tests never report a successful native/runtime binding, construct a bed,
or replace the genuine first/full water-pause qualification in the old check.
Execution requires a separate Root-owned candidate/source/CPU admission.
"""
from __future__ import annotations

import os
import threading
import unittest
import weakref
from unittest import mock

from masy import dense_boundary_store as store
from masy import dense_original_inputs as oi
from masy import dense_runner as dr
from masy import dense_water_pause as pause


NATIVE_BUNDLE = "55b564fbe547754e2fe1dc19745cee5c9f9fe9d1eeef005e65a582c2b1cf1f8a"
SYRUP_SOURCE_TABLE = "390ff7cc0d04b5e7d713156315a8a1607549b8a7f4b75fa5f04aa9b1d1bb2d03"
OLD_NATIVE_BUNDLE = "9eab1a52b23d59f674028062649eb25661824f82a1aaa1edada7f9a7641fe9bf"
OLD_SYRUP_TABLE = "f9778bf0af975d06da808514b08000c0c5d54bea6b1715f5d1001acc90ac03c9"


def authored_reference():
    return {"authored_negative_metadata_only": True}


def authored_source(**changed):
    return {"masy": "a" * 64, "maple": NATIVE_BUNDLE,
            "syrup": SYRUP_SOURCE_TABLE, **changed}


def nominal_paused_receiver():
    """Only the actual lease/refusal order is exercised; no native state exists."""
    runner = object.__new__(dr.DenseRunner)
    runner._stop_receipt, runner._next = None, 0
    lease = pause._WaterLease(weakref.ref(runner), weakref.ref(threading.current_thread()),
                              os.getpid(), 0, None, 0.0, phase="paused")
    lease.token = pause._WaterToken()
    lease.runtime_bytes = oi._json(authored_reference())
    lease.source, lease.environment = authored_source(), "authored_environment"
    lease.prepared, lease.progress = object(), object()
    runner._active_evidence = lease
    return runner, lease


class FamilyRefusalProtocols(unittest.TestCase):
    def test_owning_binding_failure_is_preserved_before_family_comparison(self):
        reference = authored_reference()
        failure = store.BoundaryStoreError("authored_owning_refusal", "no runtime success")
        with mock.patch.object(store, "_plain_metadata", wraps=store._plain_metadata) as plain, \
                mock.patch.object(store, "_runtime_binding", side_effect=failure) as binding:
            with self.assertRaises(store.BoundaryStoreError) as caught:
                pause._runtime(store, reference)
        self.assertIs(caught.exception, failure)
        plain.assert_called_once()
        binding.assert_called_once()
        self.assertEqual(binding.call_args.args[0], reference)
        self.assertIs(type(binding.call_args.args[1]), store.BoundaryStoreLimits)
        self.assertEqual(reference, authored_reference())

    def test_wrong_native_bundle_has_no_legacy_or_source_table_fallback(self):
        # The Native Source-table SHA is deliberately NOT a bundle identity.
        for wrong in (OLD_NATIVE_BUNDLE,
                      "aaf01a6c3edf0c1668af3d2405de646df0b568f42580868b4e3c7676f0f1102c"):
            with self.subTest(wrong=wrong), \
                    mock.patch.object(store, "_runtime_binding",
                                      return_value=(authored_source(maple=wrong), "unit_environment")) as binding:
                with self.assertRaises(dr.DenseRunnerError) as caught:
                    pause._runtime(store, authored_reference())
                self.assertEqual(caught.exception.code, "water_resource_source")
                binding.assert_called_once()

    def test_wrong_syrup_table_has_no_legacy_or_native_bundle_fallback(self):
        for wrong in (OLD_SYRUP_TABLE, NATIVE_BUNDLE):
            with self.subTest(wrong=wrong), \
                    mock.patch.object(store, "_runtime_binding",
                                      return_value=(authored_source(syrup=wrong), "unit_environment")) as binding:
                with self.assertRaises(dr.DenseRunnerError) as caught:
                    pause._runtime(store, authored_reference())
                self.assertEqual(caught.exception.code, "water_resource_source")
                binding.assert_called_once()

    def test_exact_family_still_refuses_retained_source_or_environment_drift(self):
        # Matching family literals only reach the existing drift refusal here;
        # neither case produces a successful _runtime return or native binding.
        for source, environment in ((authored_source(masy="b" * 64), "authored_environment"),
                                    (authored_source(), "changed_environment")):
            runner, lease = nominal_paused_receiver()
            token, prepared, progress = lease.token, lease.prepared, lease.progress
            saved_source, saved_environment = dict(lease.source), lease.environment
            with self.subTest(environment=environment), \
                    mock.patch.object(store, "_runtime_binding", return_value=(source, environment)):
                with self.assertRaises(dr.DenseRunnerError) as caught:
                    pause._runtime(store, None, lease)
            self.assertEqual(caught.exception.code, "water_runtime_changed")
            self.assertIs(runner._active_evidence, lease)
            self.assertIs(lease.token, token)
            self.assertIs(lease.prepared, prepared)
            self.assertIs(lease.progress, progress)
            self.assertEqual((lease.phase, lease.source, lease.environment),
                             ("paused", saved_source, saved_environment))

    def test_public_resume_family_refusal_precedes_consumption_and_native_work(self):
        for source in (authored_source(maple=OLD_NATIVE_BUNDLE),
                       authored_source(syrup=OLD_SYRUP_TABLE)):
            runner, lease = nominal_paused_receiver()
            token, prepared, progress = lease.token, lease.prepared, lease.progress
            with self.subTest(source=source), \
                    mock.patch.object(pause, "_declare", return_value=(None, None, {}, store)), \
                    mock.patch.object(pause, "_profile"), \
                    mock.patch.object(dr.DenseRunner, "_precheck"), \
                    mock.patch.object(store, "_runtime_binding", return_value=(source, lease.environment)), \
                    mock.patch.object(pause, "_reserve", side_effect=AssertionError("reserve after refused family")) as reserve, \
                    mock.patch.object(pause, "_entry_digest", side_effect=AssertionError("entry after refused family")) as entry, \
                    mock.patch.object(pause, "_wet", side_effect=AssertionError("wet after refused family")) as wet, \
                    mock.patch.object(pause, "_execute", side_effect=AssertionError("native after refused family")) as execute:
                with self.assertRaises(dr.DenseRunnerError) as caught:
                    pause.resume_water(runner, token)
                self.assertEqual(caught.exception.code, "water_resource_source")
                reserve.assert_not_called()
                entry.assert_not_called()
                wet.assert_not_called()
                execute.assert_not_called()
            self.assertIs(runner._active_evidence, lease)
            self.assertEqual(lease.phase, "paused")
            self.assertIs(lease.token, token)
            self.assertIs(lease.prepared, prepared)
            self.assertIs(lease.progress, progress)
            self.assertIsNone(runner._stop_receipt)
