"""Genuine Root-gated Source-census refusal checks for Dense wind pause.

The unpatched control proves Source-census admission only, with the real owning
binding; it runs no Native transport. These are NOT Native compatibility checks.
Each test then restores one
historical TABLE literal and requires the actual owning binding to refuse
with wind_resource_source. Requires the same Root reference env as the old
wind tests (MASY252_RUNTIME / MASY252_RUNTIME_SHA256), naming a NEW
source_manifest_prototype prepared against the locked committed pair.
"""
import unittest
from unittest import mock

from masy import dense_wind_pause as wp
from masy.dense_runner import DenseRunnerError
from check_dense_wind_pause import reference

_HISTORICAL_NATIVE = "aaf01a6c3edf0c1668af3d2405de646df0b568f42580868b4e3c7676f0f1102c"
_HISTORICAL_SYRUP = "f9778bf0af975d06da808514b08000c0c5d54bea6b1715f5d1001acc90ac03c9"


class DenseWindFamilyCensusRefusal(unittest.TestCase):
    def setUp(self):
        self.ref = reference()
        if self.ref.get("kind") != "source_manifest_prototype":
            self.fail("Root must supply a NEW source_manifest_prototype reference")
        self.api = wp._declare()

    def _unpatched_admits(self):
        # Real Source-census binding control only. No Native transport runs.
        # Its success makes each historical-literal refusal attributable.
        wp._runtime(self.api, self.ref)

    def _require_refusal(self):
        with self.assertRaises(DenseRunnerError) as cm:
            wp._runtime(self.api, self.ref)
        self.assertEqual(cm.exception.code, "wind_resource_source")

    def test_historical_native_table_refused(self):
        self._unpatched_admits()
        with mock.patch.object(wp, "_NATIVE_TABLE", _HISTORICAL_NATIVE):
            self._require_refusal()

    def test_historical_syrup_table_refused(self):
        self._unpatched_admits()
        with mock.patch.object(wp, "_SYRUP_TABLE", _HISTORICAL_SYRUP):
            self._require_refusal()


if __name__ == "__main__":
    unittest.main()
