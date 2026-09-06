#!/usr/bin/env python3
"""Isolated native HIP graph host-callback overhead; GPU execution requires a lease.

No Torch import, model, network operation, custom device kernel, or runtime patch.
The C++ callback runs without Python/GIL and makes no HIP API calls.
"""
import argparse
import ctypes
import json
from pathlib import Path
import statistics
import time


class Result(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in (
        'mode', 'replay_count', 'sample_count', 'node_count', 'host_node_count', 'memcpy_node_count')]
    _fields_ += [(name, ctypes.c_uint64) for name in (
        'callback_count', 'expected_callback_count', 'bad_input_count', 'resident_bytes')]
    _fields_ += [('wall_us', ctypes.c_double * 5), ('hip_event_us', ctypes.c_double * 5),
                ('max_abs_error', ctypes.c_double)]
    _fields_ += [(name, ctypes.c_int) for name in ('finite', 'exact', 'capture_supported', 'runtime_version')]
    _fields_ += [('error', ctypes.c_char * 512)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--replays', type=int, default=50)
    p.add_argument('--samples', type=int, default=5)
    args = p.parse_args()
    assert 5 <= args.replays <= 100 and 1 <= args.samples <= 5
    started = time.monotonic()
    lib = ctypes.CDLL(str(args.library.resolve()))
    fn = lib.run_host_graph_case
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(Result)]
    fn.restype = ctypes.c_int
    report = {'status': 'running', 'cases': [], 'BF16_elements': 10240,
              'bytes_per_direction': 20480,
              'notes': ['Zero/one native CPU callback per graph replay; no Python callback.',
                        'Both HIP event duration and wall time include callback completion.',
                        'Roundtrip callback checks all input BF16 bits and negates exactly.',
                        'Copy-only control has exactly two copies and no host node; identity checked.',
                        'Input changes before each of four checked single replays and each timing batch.',
                        'Host-only callback writes/checks count+42; it does not scan 20 KiB.',
                        'Batch graph throughput, not a measured CPU-RDMA collective.',
                        'No transport, MPI, RDMA, inter-node barrier or Torch interference measured.']}
    for mode, label in [(0, 'host_callback_only'), (2, '20KiB_D2H_H2D_identity'),
                        (1, '20KiB_D2H_negate_H2D')]:
        result = Result()
        rc = fn(mode, args.replays, args.samples, ctypes.byref(result))
        row = {name: getattr(result, name) for name, _ in Result._fields_
               if name not in ('wall_us', 'hip_event_us', 'error')}
        row.update(method=label, returncode=rc, error=result.error.decode(errors='replace'))
        row['wall_samples_us'] = list(result.wall_us)[:args.samples]
        row['hip_event_samples_us'] = list(result.hip_event_us)[:args.samples]
        if rc == 0:
            row['wall_median_us'] = statistics.median(row['wall_samples_us'])
            row['hip_event_median_us'] = statistics.median(row['hip_event_samples_us'])
        report['cases'].append(row)
        if rc != 0:
            report['status'] = 'failed'
            report['wall_seconds'] = time.monotonic() - started
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
            raise SystemExit(1)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)
    report['status'] = 'passed'
    report['wall_seconds'] = time.monotonic() - started
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'], 'wall_seconds': report['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
