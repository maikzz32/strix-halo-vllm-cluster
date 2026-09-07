"""Local arithmetic/layout checks, including independently archived RCCL bits."""
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from cpu_rdma_reference import input_bits, reduce_bits
from cpu_rdma_ring4_reference import PROFILE_SHA256, partition_orders, reduce_rccl_ring4_bits


class Ring4ReferenceTest(unittest.TestCase):
    def test_layout_and_orders(self):
        expected = ['2130','1302','3021','0213','2310','3102','1023','0231',
                    '3120','1203','2031','0312','1320','3201','2013','0132']
        chunks = list(partition_orders())
        self.assertEqual([''.join(map(str, c[2])) for c in chunks], expected)
        self.assertEqual([c[1]-c[0] for c in chunks], [768]*12+[256]*4)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], 10240)
        self.assertTrue(all(a[1] == b[0] for a,b in zip(chunks,chunks[1:])))
        self.assertEqual(len(PROFILE_SHA256), 64)

    def test_bad_shape_and_dtype(self):
        for shape,dtype in [((4,10239),np.uint16),((10240,4),np.uint16),((4,10240),np.float32)]:
            with self.assertRaises(ValueError):
                reduce_rccl_ring4_bits(np.zeros(shape,dtype=dtype))

    def test_fresh_cpu_scalar_oracle(self):
        # New finite, signed, broad exponent data, unrelated to archived banks.
        rng = np.random.default_rng(0x7c04a61)
        bits = rng.integers(0,65536,size=(4,10240),dtype=np.uint16)
        bits = np.where((bits & 0x7f80) == 0x7f80, bits ^ 0x0080, bits).astype(np.uint16)
        actual = reduce_rccl_ring4_bits(bits)
        for start,end,order in partition_orders():
            for index in list(range(start,start+9))+[end-1]:
                expected = reduce_bits([int(bits[r,index]) for r in order], 'bf16')
                self.assertEqual(int(actual[index]), expected, (index,order))

    def test_archived_32_banks(self):
        directory = ROOT / 'results/rccl-bf16-reference-20260907'
        paths = sorted(directory.glob('rank*-node*.bf16.bin'))
        if len(paths) != 4:
            self.skipTest('external archived four-rank evidence is not present')
        actual = paths[0].read_bytes()
        self.assertEqual(hashlib.sha256(actual).hexdigest(),
                         '17688c76319c552a6106d09015b5040f25e6d0aa48144340f3b034cce5590c24')
        self.assertTrue(all(path.read_bytes() == actual for path in paths))
        reconstructed = bytearray()
        for bank in range(32):
            inputs = np.array([[input_bits(rank,bank,i) for i in range(10240)]
                               for rank in range(4)],dtype=np.uint16)
            reconstructed.extend(reduce_rccl_ring4_bits(inputs).astype('<u2').tobytes())
        self.assertEqual(bytes(reconstructed), actual)


if __name__ == '__main__':
    unittest.main()
