#!/usr/bin/env python3
"""Bounded per-rank RCCL benchmark. Run through run_rccl_microbench.py.

Uses PyTorch's RCCL process group in the existing container. The event and
wall-clock block measurements include eager submission overhead; they are not
claimed to be GPU-only kernel latency or CUDA-graph replay latency.
"""
import argparse
import datetime
import json
import os
import signal
import socket
import statistics
import sys
import time


def profiler_smoke(torch, dist, rank, world, run_id):
    """Small collective graph with actual GPU-event and numeric assertions."""
    from pathlib import Path
    tensor = torch.full((10240,), float(rank + 1), dtype=torch.bfloat16, device="cuda")
    expected = float(world * (world + 1) // 2)
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    assert bool(torch.all(tensor == expected).item()), "eager all-reduce incorrect"
    for _ in range(3):
        tensor.fill_(rank + 1)
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            tensor.fill_(rank + 1)
            dist.all_reduce(tensor)
    torch.cuda.synchronize()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.all(tensor == expected).item()), "graph all-reduce incorrect"
    output = Path('/tmp') / ('strix-rccl-profile-' + run_id)
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / f'rank{rank}.json'
    exported = []
    def save(prof):
        prof.export_chrome_trace(str(trace_path))
        exported.append(str(trace_path))
    print(json.dumps({'kind':'profile_ready','rank':rank,'bytes':20480,
                      'collectives_per_graph':8,'trace':str(trace_path)}), flush=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, with_stack=False, profile_memory=False,
        schedule=torch.profiler.schedule(wait=0,warmup=2,active=4,repeat=1),
        on_trace_ready=save,
    ) as profiler:
        for _ in range(6):
            graph.replay()
            torch.cuda.synchronize()
            profiler.step()
    assert exported, 'profiler did not export an active cycle'
    assert bool(torch.all(tensor == expected).item()), 'profiled graph all-reduce incorrect'
    trace = json.loads(trace_path.read_text())
    kernels = [e for e in trace.get('traceEvents',[]) if e.get('ph')=='X'
               and any(c.strip().lower() in ('kernel','gpu_kernel') for c in str(e.get('cat','')).split(','))]
    collectives = [e for e in kernels if any(n in e.get('name','').lower() for n in ('nccl','rccl'))]
    launches = [e for e in trace.get('traceEvents',[]) if e.get('ph')=='X'
                and e.get('cat') in ('cuda_runtime','hip_runtime') and 'graphlaunch' in e.get('name','').lower()]
    assert len(collectives) >= 32, f'need32 real GPU collective events, got{len(collectives)}'
    assert len(launches) >= 4, f'need4 HIP graph launches, got{len(launches)}'
    print(json.dumps({'kind':'profile_result','rank':rank,'correct':True,'finite':True,
                      'expected_value':expected,'gpu_kernel_events':len(kernels),
                      'rccl_gpu_kernel_events':len(collectives),'hip_graph_launches':len(launches),
                      'gpu_collective_total_us':sum(e['dur'] for e in collectives),
                      'collective_kernel_names':sorted({e['name'] for e in collectives}),
                      'trace':str(trace_path)}), flush=True)
    graph.reset()
    del graph
    dist.barrier(device_ids=[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="4096,8192,16384")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--operations", type=int, default=32)
    parser.add_argument("--deadline", type=int, default=120)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--profile-smoke", action="store_true")
    parser.add_argument("--run-id", default="manual")
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    if any(value <= 0 or value % 2 for value in sizes):
        parser.error("sizes must be positive even byte counts (BF16)")
    if min(args.warmup, args.blocks, args.operations, args.deadline) <= 0:
        parser.error("counts and deadline must be positive")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    def expired(signum, frame):
        print(json.dumps({"rank": rank, "error": "hard deadline exceeded"}),
              file=sys.stderr, flush=True)
        os._exit(124)

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.deadline)
    # Install the hard deadline before loading GPU libraries or rendezvous.
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable in this container")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=min(60, args.deadline)))
    try:
        if args.profile_smoke:
            profiler_smoke(torch, dist, rank, world, args.run_id)
            print(json.dumps({'kind':'complete','rank':rank}),flush=True)
            return
        metadata = {"kind": "metadata", "rank": rank, "world_size": world,
                    "host": socket.gethostname(), "torch": torch.__version__,
                    "hip": torch.version.hip,
                    "gpu": torch.cuda.get_device_name(0),
                    "sizes_bytes": sizes, "dtype": "bfloat16",
                    "warmup": args.warmup, "blocks": args.blocks,
                    "operations_per_block": args.operations,
                    "cuda_graph": args.graph,
                    "environment": {key: value for key, value in os.environ.items()
                                    if key.startswith(("NCCL_", "RCCL_"))
                                    and "TOKEN" not in key and "KEY" not in key}}
        print(json.dumps(metadata), flush=True)
        for nbytes in sizes:
            tensor = torch.full((nbytes // 2,), float(rank + 1),
                                dtype=torch.bfloat16, device="cuda")
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            expected = world * (world + 1) / 2
            if not bool(torch.all(tensor == expected).item()):
                raise RuntimeError(f"incorrect all_reduce at {nbytes} bytes, rank {rank}")
            if args.graph:
                # Zero-only timings cannot detect an omitted captured collective.
                # Poison before each replay so both the captured fill and SUM
                # must execute correctly; exclude this proof from timing.
                proof = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(proof):
                        tensor.fill_(rank + 1)
                        dist.all_reduce(tensor)
                    torch.cuda.synchronize()
                    for _ in range(2):
                        tensor.fill_(-1)
                        proof.replay()
                        torch.cuda.synchronize()
                        if not bool(torch.all(tensor == expected).item()):
                            raise RuntimeError(
                                f"incorrect captured all_reduce at {nbytes} bytes, rank {rank}")
                finally:
                    proof.reset()
                    del proof
            # Repeated SUM of ones overflows in old tests. Zero remains finite.
            tensor.zero_()
            for _ in range(args.warmup):
                dist.all_reduce(tensor)
            torch.cuda.synchronize()
            dist.barrier(device_ids=[0])
            event_us, wall_us = [], []
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            graph = None
            if args.graph:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.operations):
                        dist.all_reduce(tensor)
                torch.cuda.synchronize()
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
            for _ in range(args.blocks):
                wall_begin = time.perf_counter_ns()
                begin.record()
                if graph is not None:
                    graph.replay()
                else:
                    for _ in range(args.operations):
                        dist.all_reduce(tensor)
                end.record()
                end.synchronize()
                wall_us.append((time.perf_counter_ns() - wall_begin) / 1000 / args.operations)
                event_us.append(begin.elapsed_time(end) * 1000 / args.operations)
            if not bool(torch.all(tensor == 0).item()):
                raise RuntimeError(f"zero-buffer corruption at {nbytes} bytes")

            def summary(values):
                ordered = sorted(values)
                return {"min": ordered[0], "median": statistics.median(ordered),
                        "mean": statistics.fmean(ordered),
                        "p95_block": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
                        "max": ordered[-1]}

            print(json.dumps({"kind": "result", "rank": rank, "world_size": world,
                              "bytes": nbytes, "correct": True,
                              "nonzero_graph_correctness": args.graph,
                              "event_us_per_op": summary(event_us),
                              "wall_us_per_op": summary(wall_us)}), flush=True)
            if graph is not None:
                # NCCL communicator teardown waits for its captured graphs.
                # Release each graph before destroying the process group.
                torch.cuda.synchronize()
                graph.reset()
                del graph
            dist.barrier(device_ids=[0])
        print(json.dumps({"kind": "complete", "rank": rank}), flush=True)
    finally:
        # Every healthy rank reaches teardown. The alarm also bounds teardown.
        dist.destroy_process_group()
    signal.alarm(0)


if __name__ == "__main__":
    main()
