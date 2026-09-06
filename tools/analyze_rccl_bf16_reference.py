#!/usr/bin/env python3
"""Compare archived RCCL BF16 results with explicit CPU accumulation contracts."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from cpu_rdma_reference import COUNT, PERIOD, input_bits


def decode(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def encode(values):
    values = np.asarray(values, dtype=np.float32)
    words = values.view(np.uint32)
    rounded = ((words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16).astype(np.uint16)
    exceptional = (words & 0x7f800000) == 0x7f800000
    special = np.where((words & 0x7fffff) != 0, ((words >> 16) & 0x8000) | 0x7fc0, words >> 16)
    return np.where(exceptional, special, rounded).astype(np.uint16)


def analyze(directory):
    files = sorted(directory.glob('rank*-node*.bf16.bin'))
    if len(files) != 4:
        raise ValueError('Exactly four archived rank outputs are required')
    arrays = [np.fromfile(p, dtype='<u2') for p in files]
    if any(a.size != COUNT * PERIOD for a in arrays):
        raise ValueError('Incorrect rank output size')
    if not all(np.array_equal(arrays[0], a) for a in arrays[1:]):
        raise ValueError('RCCL rank outputs differ')
    actual = arrays[0].reshape(PERIOD, COUNT)
    if not np.isfinite(decode(actual)).all():
        raise ValueError('Nonfinite RCCL output')
    inputs = np.array([[[input_bits(rank, bank, i) for i in range(COUNT)]
                        for bank in range(PERIOD)] for rank in range(4)], dtype=np.uint16)
    values = decode(inputs)
    truth = values.astype(np.float64).sum(axis=0)

    def accuracy(bits):
        error = decode(bits).astype(np.float64) - truth
        return {'max_abs_error': float(np.max(np.abs(error))),
                'rms_error': float(np.sqrt(np.mean(error * error)))}

    def candidate(name, bits):
        mismatch = bits != actual
        locations = np.argwhere(mismatch)
        first = None
        if len(locations):
            bank, index = (int(x) for x in locations[0])
            first = {'bank': bank, 'index': index, 'inputs_bits': inputs[:, bank, index].tolist(),
                     'rccl_bits': int(actual[bank, index]), 'candidate_bits': int(bits[bank, index])}
        return {'contract': name, 'mismatch_count': int(mismatch.sum()),
                'first_mismatch': first, **accuracy(bits)}

    results = []
    for each in (False, True):
        label = 'bf16_each_add' if each else 'fp32_then_bf16'
        for order in itertools.permutations(range(4)):
            total = values[order[0]].copy()
            for rank in order[1:]:
                total = total + values[rank]
                if each:
                    total = decode(encode(total))
            results.append(candidate(label + ':' + ''.join(map(str, order)), encode(total)))
        for a, b, c, d in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
            left, right = values[a] + values[b], values[c] + values[d]
            if each:
                left, right = decode(encode(left)), decode(encode(right))
            bits = encode(left + right)
            results.append(candidate(label + ':balanced:' + ''.join(map(str, (a, b, c, d))), bits))
    results.sort(key=lambda r: (r['mismatch_count'], r['contract']))
    return {'all_rank_outputs_equal': True, 'elements': COUNT * PERIOD,
            'rccl_accuracy': accuracy(actual), 'exact_contracts': [r['contract'] for r in results if not r['mismatch_count']],
            'contracts': results, 'output_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'note': 'Fixed synthetic finite banks. This does not establish model output or quality equivalence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = analyze(args.directory)
    target = args.directory / 'numeric-comparison.json'
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(target), 'exact_contracts': report['exact_contracts'],
                      'best': report['contracts'][0], 'rccl_accuracy': report['rccl_accuracy']}, indent=2))


if __name__ == '__main__':
    main()
