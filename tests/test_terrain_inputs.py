"""Independent geometry and immutable-input controls for the bounded runner."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from masy.terrain_runner import terrain_maps, validate_inputs, _freeze_inputs
from masy import triplet_terrain_runtime as terrain_runtime
from masy import triplet_cuda_runtime as old_runtime
from masy.triplet_lock import TripletLockError


class TerrainInputTests(unittest.TestCase):
    def test_world_plane_derivatives_and_hole_do_not_create_a_cliff(self):
        row, col = np.mgrid[:7, :9]
        full = 1400.0 + 0.2*col + 0.1*row
        mask = np.ones((5, 7), dtype=bool)
        mask[2, 3] = False
        full[3, 4] = -9999
        aspect, slope = terrain_maps(full, mask, 1.0, -9999)
        np.testing.assert_allclose(slope.reshape(mask.shape)[mask], np.arctan(np.hypot(0.2,0.1)), atol=1e-12)
        self.assertTrue(np.all(aspect.reshape(mask.shape)[mask] == 6))  # downslope SW
        self.assertEqual(slope.reshape(mask.shape)[~mask][0], 0.0)

    def test_finite_stale_height_inside_nodata_cannot_enter_pet_gradient(self):
        row,col = np.mgrid[:7,:9]
        full = 1400.+.2*col+.1*row
        mask = np.ones((5,7),dtype=bool);mask[2,3]=False
        full[3,4] = 9000.  # finite non-sentinel stale/base value
        aspect,slope = terrain_maps(full,mask,1.,-9999.)
        np.testing.assert_allclose(slope.reshape(mask.shape)[mask],np.arctan(np.hypot(.2,.1)),atol=1e-12)
        self.assertTrue(np.all(aspect.reshape(mask.shape)[mask] == 6))

    def _inputs(self, root):
        (root / 'case').mkdir()
        (root / 'case/model.yaml').write_text('seed: 19\n')
        record = dict(schema='masy.hybrid_case_inputs.v1',
            files={'case/model.yaml': hashlib.sha256((root/'case/model.yaml').read_bytes()).hexdigest()})
        p = root / 'input_manifest.json'
        p.write_text(json.dumps(record))
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def test_verified_snapshot_is_independent_of_later_original_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha = self._inputs(root)
            copy = _freeze_inputs(validate_inputs(root, sha), root/'frozen')
            (root/'case/model.yaml').write_text('seed: 20\n')
            self.assertEqual((copy[0]/'case/model.yaml').read_text(), 'seed: 19\n')
            validate_inputs(copy[0], sha)
            with self.assertRaisesRegex(ValueError, 'changed'):
                validate_inputs(root, sha)

    def test_file_and_manifest_tamper_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha = self._inputs(root)
            (root/'case/model.yaml').unlink()
            (root/'case/model.yaml').symlink_to(root/'input_manifest.json')
            with self.assertRaises(ValueError):
                validate_inputs(root, sha)
            (root/'input_manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'external SHA256'):
                validate_inputs(root, sha)

    def test_terrain_and_historical_source_profiles_remain_distinct(self):
        q = dict(backend={'name':'cuda','provider':'cupy'},
            device={'physical_uuid':'GPU-b02858c1-29c3-9ea9-33b1-831c33f8cf75','logical_ordinal':0},
            native_profile=terrain_runtime.NATIVE_PROFILE)
        terrain_runtime._extension(q)
        with self.assertRaises(TripletLockError):
            old_runtime._extension(q)
        q['native_profile'] = old_runtime.NATIVE_PROFILE
        with self.assertRaises(TripletLockError):
            terrain_runtime._extension(q)


if __name__ == '__main__':
    unittest.main()
