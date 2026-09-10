"""CPU-only numerical contract for the fixed-order four-rank BF16 reduction.

The reference oracle chooses nearest BF16 neighbors in value space, rather
than duplicating the implementation's bit-rounding expression. No networking,
GPU packages or third-party dependencies are imported.
"""
import bisect
import importlib.util
import math
from pathlib import Path
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('cpu_rdma_reference', ROOT/'tools/cpu_rdma_reference.py')
REF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REF)


def decode(bits):
    return struct.unpack('>f', struct.pack('>I', bits << 16))[0]


POSITIVE_FINITE = [decode(bits) for bits in range(0x7f80)]


def nearest_bf16(value):
    """Independent finite-input RNE oracle using ordered adjacent values."""
    negative = math.copysign(1.0,value)<0
    value = abs(struct.unpack('>f',struct.pack('>f',value))[0])
    upper = bisect.bisect_left(POSITIVE_FINITE,value)
    if upper >= len(POSITIVE_FINITE):
        raise ValueError('Oracle intentionally excludes overflow')
    if POSITIVE_FINITE[upper] == value:
        bits=upper
    else:
        lower=upper-1
        dl=value-POSITIVE_FINITE[lower]
        du=POSITIVE_FINITE[upper]-value
        bits=lower if dl<du or (dl==du and lower%2==0) else upper
    return bits | (0x8000 if negative else 0)


def oracle_reduce(bits,mode):
    accumulator=decode(bits[0])
    for item in bits[1:]:
        accumulator=struct.unpack('>f',struct.pack('>f',accumulator+decode(item)))[0]
        if mode=='bf16':
            accumulator=decode(nearest_bf16(accumulator))
    return nearest_bf16(accumulator)


class TestCpuRdmaReference(unittest.TestCase):
    def test_all_finite_bf16_round_trip(self):
        for bits in range(1 << 16):
            if bits & 0x7f80 == 0x7f80:
                continue
            value=REF.bf16_to_float(bits)
            self.assertEqual(value,decode(bits))
            self.assertEqual(REF.float_to_bf16(value),bits)

    def test_ties_to_even_and_subnormals(self):
        cases=[(1.00390625,0x3f80),(1.01171875,0x3f82),
               (-1.00390625,0xbf80),(-1.01171875,0xbf82),
               (2.0**-134,0),(3*2.0**-134,2),
               (-2.0**-134,0x8000),(-3*2.0**-134,0x8002),
               (0.0,0),(-0.0,0x8000)]
        for value,expected in cases:
            with self.subTest(value=value):
                self.assertEqual(REF.float_to_bf16(value),expected)

    def test_infinity_and_canonical_quiet_nan(self):
        self.assertEqual(REF.float_to_bf16(float('inf')),0x7f80)
        self.assertEqual(REF.float_to_bf16(float('-inf')),0xff80)
        for sign in (0,0x80000000):
            nan=struct.unpack('>f',struct.pack('>I',sign|0x7fc12345))[0]
            self.assertEqual(REF.float_to_bf16(nan),(sign>>16)|0x7fc0)

    def test_fixed_rank_order_and_rounding_goldens(self):
        cases=[([256,1,-256,1],0x4000,0x3f80),
               ([256,-256,1,1],0x4000,0x4000),
               ([2**24,1,-2**24,1],0x3f80,0x3f80),
               ([1,2**-8,2**-8,-1],0x3c00,0),
               ([0.5,-0.5,0.25,-0.25],0,0),
               ([-0.0,-0.0,-0.0,-0.0],0x8000,0x8000)]
        for values,fp32,bf16 in cases:
            bits=[nearest_bf16(value) for value in values]
            for mode,expected in (('fp32',fp32),('bf16',bf16)):
                with self.subTest(values=values,mode=mode):
                    self.assertEqual(REF.reduce_bits(bits,mode=mode),expected)
                    self.assertEqual(oracle_reduce(bits,mode),expected)

    def test_full_payload_is_finite_nonzero_deterministic_and_rotates(self):
        for rank in range(4):
            first=[REF.input_bits(rank,0,index) for index in range(10240)]
            again=[REF.input_bits(rank,0,index) for index in range(10240)]
            rotated=[REF.input_bits(rank,1,index) for index in range(10240)]
            self.assertEqual(first,again)
            self.assertNotEqual(first,rotated)
            self.assertGreater(len(set(first)),8)
            for bits in first+rotated:
                self.assertIsInstance(bits,int)
                self.assertGreaterEqual(bits,0)
                self.assertLess(bits,65536)
                self.assertNotEqual(bits & 0x7fff,0)
                self.assertNotEqual(bits & 0x7f80,0x7f80)

    def test_rotating_payload_matches_independent_reduction_oracle(self):
        differences=0
        mixed_signs=0
        for iteration in (0,1,7,23):
            # Includes both ends and samples the complete 20 KiB payload.
            for index in list(range(0,10240,17))+[10239]:
                bits=[REF.input_bits(rank,iteration,index) for rank in range(4)]
                mixed_signs+=len({item>>15 for item in bits})==2
                for mode in ('fp32','bf16'):
                    expected=oracle_reduce(bits,mode)
                    self.assertEqual(REF.reduce_bits(bits,mode=mode),expected,
                                     (iteration,index,mode,bits))
                    self.assertTrue(math.isfinite(decode(expected)))
                differences+=oracle_reduce(bits,'fp32')!=oracle_reduce(bits,'bf16')
        self.assertGreater(mixed_signs,0,'Input must exercise cancellation across ranks')
        self.assertGreater(differences,0,'Input must distinguish the two accumulation contracts')


if __name__=='__main__':
    unittest.main()
