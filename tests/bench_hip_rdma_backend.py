#!/usr/bin/env python3
"""Owned, isolated four-rank backend test; never installs a serving hook."""
import argparse
import datetime
import json
from pathlib import Path
import signal
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--master', required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--gid-index', type=int, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--library', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['ring4'], required=True)
    p.add_argument('--deadline', type=int, required=True)
    p.add_argument('--iterations', type=int, required=True)
    p.add_argument('--samples', type=int, required=True)
    p.add_argument('--warmup', type=int, required=True)
    args = p.parse_args()
    if not (args.rank in range(4) and 1024 <= args.port <= 65534 and
            1 <= args.iterations <= 128 and 1 <= args.samples <= 5 and args.warmup == 4):
        p.error('invalid bounded rank settings')
    import os
    signal.signal(signal.SIGALRM, lambda *unused: os._exit(124))
    signal.alarm(args.deadline)
    report = {'status':'failed', 'rank':args.rank, 'run_id':args.run_id, 'mode':'ring4'}
    import numpy as np
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from hip_rdma_backend import create_if_enabled
    from validate_rccl_ring4 import validate, probe_bank
    from cpu_rdma_ring4_reference import reduce_rccl_ring4_bits
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    group = None
    reference = backend = graph = None
    try:
        dist.init_process_group('gloo', rank=args.rank, world_size=4,
                                init_method=f'tcp://{args.master}:{args.port}',
                                timeout=datetime.timedelta(seconds=20))
        group = dist.group.WORLD
        reference = PyNcclCommunicator(group, device=device)
        if not reference.available or reference.disabled:
            raise RuntimeError('PyNccl reference unavailable')
        environment = {'STRIX_HIP_RDMA_ENABLE':'1', 'STRIX_HIP_RDMA_LIBRARY':args.library,
                       'STRIX_HIP_RDMA_RUN_ID':args.run_id, 'STRIX_HIP_RDMA_DEVICE':args.device,
                       'STRIX_HIP_RDMA_MASTER':args.master, 'STRIX_HIP_RDMA_MODE':'ring4',
                       'STRIX_HIP_RDMA_PORT':str(args.port+1),
                       'STRIX_HIP_RDMA_GID_INDEX':str(args.gid_index)}
        backend = create_if_enabled(group, device, 'tp:isolated-rdma', [0,1,2,3],
            environment=environment, reference_validator=lambda b: validate(b, reference, group))
        report['parity_proof'] = backend.parity_proof
        # Unsupported shapes remain untouched before any native enqueue.
        unsupported = torch.zeros(16, dtype=torch.bfloat16, device=device)
        if backend.try_all_reduce(unsupported) is not None:
            raise RuntimeError('unsupported shape was accepted')
        report['unsupported_shape_falls_back'] = True
        source = torch.empty(10240, dtype=torch.bfloat16, device=device)
        outputs = []
        graph = torch.cuda.CUDAGraph()
        source.fill_(args.rank+1)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        operations = 32
        with torch.cuda.graph(graph):
            for _ in range(operations):
                outputs.append(backend.try_all_reduce(source))
        torch.cuda.synchronize()
        if len({t.data_ptr() for t in outputs}) != operations:
            raise RuntimeError('captured outputs alias')
        def prepare(bank):
            inputs = probe_bank(bank, seed=202609071029)
            source.copy_(torch.from_numpy(inputs[args.rank].copy()).view(torch.bfloat16))
            torch.cuda.synchronize()
            return inputs, reduce_rccl_ring4_bits(inputs)
        checks = 0
        def check(inputs, expected):
            nonlocal checks
            if not np.array_equal(source.view(torch.uint16).cpu().numpy(), inputs[args.rank]):
                raise RuntimeError('source changed during graph')
            for tensor in outputs:
                if not np.array_equal(tensor.view(torch.uint16).cpu().numpy(), expected):
                    raise RuntimeError('captured output differs from calibrated ring profile')
                checks += 1
        for bank in range(args.warmup):
            inputs, expected = prepare(bank)
            graph.replay()
            torch.cuda.synchronize()
            check(inputs, expected)
        wall, event = [], []
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for sample in range(args.samples):
            inputs, expected = prepare(100+sample)
            dist.barrier(group=group)
            start = time.perf_counter_ns()
            begin.record()
            for _ in range(args.iterations): graph.replay()
            end.record()
            end.synchronize()
            divisor = args.iterations * operations
            wall.append((time.perf_counter_ns()-start)/1000/divisor)
            event.append(begin.elapsed_time(end)*1000/divisor)
            check(inputs, expected)
        report.update(exact=True, finite=True, graph_output_checks=checks,
                      operations_per_graph=operations, iterations=args.iterations,
                      samples=args.samples, warmup=args.warmup,
                      wall_samples_us=wall, hip_event_samples_us=event)
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        try:
            if graph is not None:
                torch.cuda.synchronize()
                graph.reset()
                graph = None
            report['graphs_destroyed'] = True
            if backend is not None: backend.close_after_graphs_destroyed()
            report['backend_closed'] = backend is not None and not backend.handle.value
            if reference is not None and reference.available:
                reference.nccl.ncclCommDestroy(reference.comm)
                reference.available, reference.disabled = False, True
            report['pynccl_destroyed'] = reference is not None and not reference.available
            if dist.is_initialized(): dist.destroy_process_group()
            report['cpu_group_destroyed'] = not dist.is_initialized()
        except BaseException as error:
            report.update(status='failed', cleanup_error=repr(error))
        args.output.write_text(json.dumps(report,indent=2))
        print(json.dumps({'event':'backend_result', **report}), flush=True)
    signal.alarm(0)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
