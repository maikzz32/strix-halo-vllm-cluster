#!/usr/bin/env python3
"""Shape/topology-specific arithmetic hypothesis; not a generic RCCL contract.

This profile reconstructs the archived 2026-09-07 finite RCCL reference exactly.
An actual communicator must pass independent eager/graph calibration before use.
No GPU, Torch, RDMA or network dependency.
"""
import hashlib
import json

import numpy as np

from analyze_rccl_bf16_reference import decode, encode

PROFILE_NAME = 'rccl-2.30.4-ring-ll-4ch-20k-0213-0231-0312-0132-v1'
PROFILE = {
    'name': PROFILE_NAME,
    'world_size': 4,
    'values': 10240,
    'dtype': 'bf16',
    'rounding': 'rne_after_each_rank_addition',
    'algorithm': 'Ring',
    'protocol': 'LL',
    'channels': [
        {'offset': 0, 'values': 3072, 'chunk_values': 768, 'ring': [0, 2, 1, 3]},
        {'offset': 3072, 'values': 3072, 'chunk_values': 768, 'ring': [0, 2, 3, 1]},
        {'offset': 6144, 'values': 3072, 'chunk_values': 768, 'ring': [0, 3, 1, 2]},
        {'offset': 9216, 'values': 1024, 'chunk_values': 256, 'ring': [0, 1, 3, 2]},
    ],
}
PROFILE_SHA256 = hashlib.sha256(
    json.dumps(PROFILE, sort_keys=True, separators=(',', ':')).encode('ascii')
).hexdigest()


def partition_orders():
    """Yield sixteen (start, end_exclusive, rank_order) tuples, from the profile."""
    for channel in PROFILE['channels']:
        ring = channel['ring']
        for chunk in range(4):
            start = channel['offset'] + chunk * channel['chunk_values']
            yield start, start + channel['chunk_values'], tuple(ring[chunk+1:] + ring[:chunk+1])


def reduce_rccl_ring4_bits(inputs):
    """(4,10240) uint16 BF16 encodings -> (10240,) uint16 encodings.

    Preserves float32 additions and canonical signed NaN/Inf conversion in the
    established NumPy encode/decode reference. Exceptional propagation is not
    claimed bit-identical to GPU compiler NaN choices. Finite bit parity is the
    calibration gate; nonfinite inputs require a separate explicit policy.
    """
    bits = np.asarray(inputs)
    if bits.shape != (4, 10240) or bits.dtype != np.dtype('uint16'):
        raise ValueError('expected exactly (4,10240) native uint16 BF16 encodings')
    values = decode(bits)
    result = np.empty(10240, dtype=np.uint16)
    with np.errstate(invalid='ignore', over='ignore', under='ignore'):
        for start, end, order in partition_orders():
            value = values[order[0], start:end].copy()
            for rank in order[1:]:
                value = decode(encode(value + values[rank, start:end]))
            result[start:end] = encode(value)
    return result
