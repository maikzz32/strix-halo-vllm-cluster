#!/usr/bin/env python3
"""Read-only BF16 HC exponent survey and exact CPU packing roundtrip.

No Torch/HIP import, GPU access or checkpoint writes. Estimated byte counts
assume a four-byte descriptor per block; this is not a serving implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np


def analyze(model):
    started = time.monotonic()
    index = model / 'model.safetensors.index.json'
    mapping = json.loads(index.read_text())['weight_map']
    keys = sorted(k for k in mapping if 'hyper_connection' in k and
                  any(t in k for t in ('input_mix_weight_', 'block_inject_weight')))
    headers = {}
    rows = []
    for key in keys:
        path = model / mapping[key]
        if path not in headers:
            with path.open('rb') as f:
                length = struct.unpack('<Q', f.read(8))[0]
                headers[path] = (8 + length, json.loads(f.read(length)))
        start, header = headers[path]
        info = header[key]
        if info['dtype'] != 'BF16':
            raise ValueError((key, info['dtype']))
        lo, hi = info['data_offsets']
        with path.open('rb') as f:
            f.seek(start + lo)
            raw = f.read(hi - lo)
        if len(raw) != hi - lo:
            raise ValueError('short checkpoint read')
        words = np.frombuffer(raw, dtype='<u2')
        row = {'key': key, 'shape': info['shape'], 'bytes': len(raw),
               'sha256': hashlib.sha256(raw).hexdigest(), 'blocks': []}
        for block in (32, 64, 128, 256):
            if len(words) % block:
                raise ValueError('unaligned tensor')
            matrix = words.reshape(-1, block)
            exponent = (matrix >> 7) & 255
            base = exponent.min(axis=1)
            eligible = exponent.max(axis=1) - base <= 15
            selected = matrix[eligible]
            origin = base[eligible, None]
            delta = ((selected >> 7) & 255) - origin
            # Sign and mantissa retain all eight original bits, including -0.
            low = ((selected & 127) | ((selected >> 8) & 128)).astype(np.uint8)
            high = (delta[:, 0::2] | (delta[:, 1::2] << 4)).astype(np.uint8)
            unpacked = np.empty_like(selected)
            unpacked[:, 0::2] = high & 15
            unpacked[:, 1::2] = high >> 4
            restored = ((low.astype(np.uint16) & 127) |
                        ((low.astype(np.uint16) & 128) << 8) |
                        ((unpacked + origin) << 7))
            if not np.array_equal(restored, selected):
                raise ValueError('packing roundtrip mismatch')
            good, total = int(eligible.sum()), len(matrix)
            encoded = good * block * 3 // 2 + (total - good) * block * 2 + total * 4
            row['blocks'].append({'block_values': block, 'eligible_blocks': good,
                                  'total_blocks': total, 'estimated_bytes': encoded,
                                  'exact_roundtrip': True})
        rows.append(row)
    total_bytes = sum(r['bytes'] for r in rows)
    summary = []
    for j, block in enumerate((32, 64, 128, 256)):
        encoded = sum(r['blocks'][j]['estimated_bytes'] for r in rows)
        summary.append({'block_values': block, 'estimated_bytes': encoded,
                        'byte_saving_percent': 100 * (1 - encoded / total_bytes)})
    return {'schema': 1, 'model': str(model), 'index_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
            'tensor_count': len(rows), 'raw_bytes': total_bytes, 'summary': summary,
            'elapsed_s': time.monotonic() - started, 'tensors': rows,
            'notes': ['Read-only checkpoint survey; no runtime changes.',
                      'Eligible blocks use 8 sign/mantissa bits and 4 exponent-delta bits.',
                      'Other blocks retain all original 16 bits; descriptor budget is 4 bytes/block.',
                      'Physical GPU layout, decoder cost and model performance are unmeasured.',
                      'Every eligible block passed a byte-packed CPU bit-exact roundtrip.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.model)
    a.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'tensors'}))
