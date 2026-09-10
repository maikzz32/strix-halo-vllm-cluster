#!/usr/bin/env python3
"""One rank of the isolated four-GPU HIP-graph + CPU-verbs benchmark.

An exclusive four-rank GPU/network lease is required. No model/Torch import or
serving modification. The coordinator owns startup, idle checks, hard process
timeouts and cleanup. All stdout records are JSON; --output keeps full evidence.
"""
from __future__ import annotations
import argparse
import ctypes as C
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')


class Options(C.Structure):
    _fields_ = [(name, C.c_int) for name in (
        'rank', 'gid_index', 'port', 'deadline_seconds', 'mode', 'iterations', 'samples', 'warmup')]
    _fields_ += [(name, C.c_char_p) for name in ('device', 'master', 'run_id')]


class Result(C.Structure):
    _fields_ = [(name, C.c_int) for name in (
        'rank', 'mode', 'iterations', 'samples', 'warmup', 'runtime_version',
        'node_count', 'host_nodes', 'memcpy_nodes', 'main_selftest_rc', 'callback_selftest_rc',
        'main_rounding_before', 'main_rounding_after', 'callback_rounding_before', 'callback_rounding_after')]
    _fields_ += [(name, C.c_uint32) for name in (
        'main_mxcsr_before', 'main_mxcsr_after', 'callback_mxcsr_before', 'callback_mxcsr_after')]
    _fields_ += [(name, C.c_uint64) for name in (
        'callback_calls', 'successful_collectives', 'expected_collectives', 'arena_bytes')]
    _fields_ += [(name, C.c_int) for name in (
        'direct_pinned_checks', 'graph_numeric_checks', 'exact', 'finite', 'pinned_mr_ok',
        'negative_test_passed', 'expected_negative_code', 'cleanup_rc', 'arena_released')]
    _fields_ += [(name, C.c_double) for name in ('setup_seconds', 'total_seconds', 'max_abs_error')]
    _fields_ += [('wall_us', C.c_double * 5), ('hip_event_us', C.c_double * 5),
                ('error', C.c_char * 512), ('expected_negative_error', C.c_char * 512)]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--master', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--gid-index', type=int, default=1)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--deadline', type=int, default=30, help='Overall deadline per mode, seconds')
    parser.add_argument('--mode', choices=['fp32', 'bf16', 'both'], default='both')
    parser.add_argument('--iterations', type=int, default=32)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (0 <= args.rank < 4 and 0 <= args.gid_index <= 255 and 1024 <= args.port <= 65534
            and 10 <= args.deadline <= 120 and 1 <= args.iterations <= 128
            and 1 <= args.samples <= 5 and 1 <= args.warmup <= 16
            and len(args.run_id) == 32 and all(c in '0123456789abcdef' for c in args.run_id)):
        parser.error('Invalid bounded arguments or run ID')
    report = {'event': 'complete', 'status': 'running', 'run_id': args.run_id, 'rank': args.rank,
              'start_utc': utc(), 'cases': [], 'payload_BF16_values': 10240,
              'payload_bytes': 20480, 'rdma_tx_payload_bytes_per_rank': 61440,
              'borrowed_hip_host_arena_bytes': 204800,
              'library_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(),
              'library_path': str(args.library.resolve()),
              'notes': ['Three captured nodes: GPU input D2H -> native CPU verbs callback -> output H2D.',
                        'Separate GPU input/output; no Torch/model/runtime hook.',
                        'Pinned MR path checked directly on four changing inputs before capture.',
                        'Graph warmups individually checked; input changes outside graph each timing batch.',
                        'Fixed-rank scalar reference per FP32-final/BF16-each-add mode, not RCCL bit parity.',
                        'Main and callback-thread numerical selftests require RNE and FTZ/DAZ disabled.',
                        'MXCSR exception flags bits0..5 may change; FP control bits are checked separately.',
                        'Final invalid-count replay and a second replay must retain error and NaN output.',
                        'The per-mode overall timeout is supplemented by 5s backend-call deadlines.',
                        'Hard timeout/process ownership is the external coordinator responsibility.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report['end_utc'] = utc()
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    try:
        library = C.CDLL(str(args.library.resolve()))
        for symbol, expected in [('hip_rdma_options_size', C.sizeof(Options)),
                                 ('hip_rdma_result_size', C.sizeof(Result))]:
            size_fn = getattr(library, symbol)
            size_fn.argtypes = []
            size_fn.restype = C.c_size_t
            if size_fn() != expected:
                raise RuntimeError(f'C/Python ABI mismatch for {symbol}')
        fn = library.run_hip_rdma_graph_case
        fn.argtypes = [C.POINTER(Options), C.POINTER(Result)]
        fn.restype = C.c_int
        modes = [0, 1] if args.mode == 'both' else [int(args.mode == 'bf16')]
        for mode in modes:
            transport_id = hashlib.sha256(f'{args.run_id}:mode{mode}'.encode()).hexdigest()[:32]
            transport_port = args.port + mode
            options = Options(args.rank, args.gid_index, transport_port, args.deadline, mode,
                              args.iterations, args.samples, args.warmup,
                              args.device.encode(), args.master.encode(), transport_id.encode())
            result = Result()
            started_utc = utc()
            started = time.monotonic()
            rc = fn(C.byref(options), C.byref(result))
            row = {name: getattr(result, name) for name, _ in Result._fields_
                   if name not in ('wall_us', 'hip_event_us', 'error', 'expected_negative_error')}
            row.update(event='result', run_id=args.run_id, transport_run_id=transport_id,
                       transport_port=transport_port,
                       mode='fp32_then_bf16' if mode == 0 else 'bf16_each_add',
                       start_utc=started_utc, end_utc=utc(), process_call_seconds=time.monotonic() - started,
                       returncode=rc, error=result.error.decode(errors='replace'),
                       expected_negative_error=result.expected_negative_error.decode(errors='replace'),
                       wall_samples_us=list(result.wall_us)[:args.samples],
                       hip_event_samples_us=list(result.hip_event_us)[:args.samples])
            if rc == 0:
                row['wall_median_us'] = statistics.median(row['wall_samples_us'])
                row['hip_event_median_us'] = statistics.median(row['hip_event_samples_us'])
            report['cases'].append(row)
            save()
            print(json.dumps(row), flush=True)
            if rc != 0:
                raise RuntimeError(f'Native rank {args.rank} mode {mode} failed: {row["error"]}')
        report['status'] = 'passed'
        save()
        print(json.dumps(report), flush=True)
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        save()
        print(json.dumps(report), flush=True)
        # A native cleanup failure may intentionally retain registered HIP memory.
        # End this owned standalone process; never continue with another graph.
        sys.stdout.flush()
        os._exit(1)


if __name__ == '__main__':
    main()
