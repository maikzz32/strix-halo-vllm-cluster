"""Communicator changes must reject unknown and already modified sources."""
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from patch_hip_rdma_communicator import patched


class PatchTests(unittest.TestCase):
    source = b'''class Communicator:
    def __init__(self):
        self.world_size = 4

    def _log_all_reduce_backend_selection(self) -> None:
        pass

    def all_reduce(self, input_):
        return input_
'''

    def test_exact_source_only_and_no_double_patch(self):
        digest = hashlib.sha256(self.source).hexdigest()
        output = patched(self.source, digest)
        self.assertIn(b'self._strix_hip_rdma = make_backend(self)', output)
        self.assertEqual(output.count(b'try_all_reduce(input_)'), 1)
        with self.assertRaises(ValueError):
            patched(output, digest)
        with self.assertRaises(ValueError):
            patched(self.source + b'\n', digest)

    def test_missing_anchor_rejected_even_with_matching_hash(self):
        source = self.source.replace(b'all_reduce(self, input_)', b'all_reduce(self, tensor)')
        with self.assertRaises(ValueError):
            patched(source, hashlib.sha256(source).hexdigest())


if __name__ == '__main__':
    unittest.main()
