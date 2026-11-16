"""New CUDA frontend metadata controls; physical cases require explicit runtime."""
import dataclasses
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from masy.gpu_contract import GpuPolicy, GpuContractError
from masy.daily_triplet_cuda import PROFILE, DailyCudaTripletSession
from masy.daily_triplet import PROFILE as CPU_PROFILE, TripletError

UUID = 'GPU-5c8bd3d4-2885-82c6-a0f0-ad2b7f87b871'


def policy(**kwargs):
    values = dict(device_id=0, physical_uuid=UUID, max_host_bytes=1 << 28, max_device_bytes=1 << 28,
                  host_reserve_bytes=1 << 20, device_reserve_bytes=1 << 20,
                  host_available_bytes_fn=lambda: 1 << 30, device_available_bytes_fn=lambda: 1 << 30)
    values.update(kwargs)
    return GpuPolicy(**values)


class Metadata(unittest.TestCase):
    def test_import_has_no_native_or_provider_side_effect(self):
        source = 'import sys;sys.path.insert(0,%r);import masy.daily_triplet_cuda;assert not any(n in sys.modules for n in ("numpy","cupy","maple","maple_syrup","leaf"))' % str(ROOT / 'src')
        subprocess.run([sys.executable, '-I', '-B', '-c', source], check=True)

    def test_cpu_and_cuda_profiles_remain_distinct(self):
        self.assertNotEqual(PROFILE, CPU_PROFILE)
        self.assertEqual(policy().record()['physical_uuid'], UUID)
        self.assertNotIn('host_available_bytes_fn', policy().record())

    def test_policy_refuses_ambiguous_or_invalid_resource_inputs(self):
        for name, values in {'device_id': (True, -1, 0.0), 'max_host_bytes': (True, 0, -1, 1.0),
                'max_device_bytes': (False, 0, -1, 1.0), 'host_reserve_bytes': (-1, False, 1.0),
                'device_reserve_bytes': (-1, True, 1.0), 'physical_uuid': ('GPU-' + 'x' * 36, UUID.upper(), '', None),
                'host_available_bytes_fn': (None, 1), 'device_available_bytes_fn': (None, 1)}.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(GpuContractError):
                    policy(**{name: value})

    def test_record_does_not_grant_backend_or_allocator_authority(self):
        p = policy()
        copied = p.record()
        copied['device_id'] = 3
        self.assertEqual(p.device_id, 0)
        # A metadata-only interpreter has no CuPy, so no reader is invoked.
        calls = []
        p = policy(host_available_bytes_fn=lambda: calls.append('host'), device_available_bytes_fn=lambda: calls.append('device'))
        with self.assertRaises(GpuContractError):
            p.admit(1, 1, 'negative preloaded-provider control')
        self.assertEqual(calls, [])

    def test_combined_checkpoint_is_explicitly_unqualified(self):
        s = object.__new__(DailyCudaTripletSession)
        with self.assertRaises(TripletError) as raised:
            s.save_checkpoint('unused')
        self.assertEqual(raised.exception.code, 'checkpoint_unqualified')


if __name__ == '__main__':
    unittest.main()
