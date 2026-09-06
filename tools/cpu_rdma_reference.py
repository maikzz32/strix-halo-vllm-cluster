"""CPU-only exact input/rounding specification for cpu_rdma_allreduce.c."""
import math
import struct

COUNT = 10240
PERIOD = 32


def bf16_to_float(bits):
    return struct.unpack('<f', struct.pack('<I', bits << 16))[0]


def _float32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def float_to_bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    if bits & 0x7f800000 == 0x7f800000:
        return (bits >> 16) if not bits & 0x7fffff else ((bits >> 16) & 0x8000) | 0x7fc0
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


def input_bits(rank, iteration, index):
    if not 0 <= rank < 4 or not 0 <= index < COUNT or iteration < 0:
        raise ValueError('invalid input coordinates')
    iteration %= PERIOD
    if index % 16 == 0:
        return [0x4380, 0x3f80, 0xc380, 0x3f80][rank]
    if index % 16 == 1:
        return [0x3f80, 0x3b80, 0xbf80, 0x3b80][rank]
    h = ((index * 0x9e3779b1) ^ ((iteration + 1) * 0x85ebca6b) ^ ((rank + 1) * 0xc2b2ae35)) & 0xffffffff
    h ^= h >> 16
    exponent = 123 + ((h >> 24) % 9)
    return ((h >> 15) & 0x8000) | (exponent << 7) | (h & 127)


def reduce_bits(values, mode='fp32'):
    if len(values) != 4 or mode not in ('fp32', 'bf16'):
        raise ValueError('four ranks and fp32 or bf16 required')
    acc = bf16_to_float(values[0])
    for bits in values[1:]:
        acc = _float32(acc + bf16_to_float(bits))
        if mode == 'bf16':
            acc = bf16_to_float(float_to_bf16(acc))
    return float_to_bf16(acc)


def fingerprints():
    """FNV-1a over little-endian uint16 values, matching C --self-test."""
    hashes = {'inputs': 14695981039346656037, 'fp32': 14695981039346656037, 'bf16': 14695981039346656037}
    def add(name, bits):
        for byte in (bits & 255, bits >> 8):
            hashes[name] = ((hashes[name] ^ byte) * 1099511628211) & 0xffffffffffffffff
    for iteration in range(PERIOD):
        for index in range(COUNT):
            values = [input_bits(rank, iteration, index) for rank in range(4)]
            for value in values:
                add('inputs', value)
            add('fp32', reduce_bits(values, 'fp32'))
            add('bf16', reduce_bits(values, 'bf16'))
    return {key: f'{value:016x}' for key, value in hashes.items()}


if __name__ == '__main__':
    import json
    print(json.dumps(fingerprints(), sort_keys=True))
