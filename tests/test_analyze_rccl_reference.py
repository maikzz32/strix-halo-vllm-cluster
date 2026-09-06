"""Independent scalar checks for vectorized comparison of RCCL BF16 outputs."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from analyze_rccl_bf16_reference import decode, encode
from cpu_rdma_reference import bf16_to_float, float_to_bf16, input_bits, reduce_bits


class NumericsTests(unittest.TestCase):
    def test_all_bf16_patterns_match_independent_scalar_roundtrip(self):
        bits = np.arange(65536, dtype=np.uint16)
        expected = np.array([float_to_bf16(bf16_to_float(int(x))) for x in bits], dtype=np.uint16)
        np.testing.assert_array_equal(encode(decode(bits)), expected)

    def test_rotating_cancellation_banks_match_scalar_contracts(self):
        for phase in (0, 1, 17, 31):
            bits = np.array([[input_bits(rank, phase, i) for i in range(256)]
                             for rank in range(4)], dtype=np.uint16)
            values = decode(bits)
            for mode in ('fp32', 'bf16'):
                with self.subTest(phase=phase, mode=mode):
                    total = values[0].copy()
                    for rank in range(1, 4):
                        total = total + values[rank]
                        if mode == 'bf16':
                            total = decode(encode(total))
                    expected = np.array([reduce_bits([int(x) for x in bits[:, i]], mode)
                                         for i in range(256)], dtype=np.uint16)
                    np.testing.assert_array_equal(encode(total), expected)


if __name__ == '__main__':
    unittest.main()
