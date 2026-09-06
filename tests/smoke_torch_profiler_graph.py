"""Profile the same five graph replays as smoke_rocprof_graph.py.

Initialization, three eager warmups, graph capture and validation are outside
the profiler. Exit 2 means the Chrome trace lacks five real GPU kernel events.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def kernel_events(trace):
    # CPU launch API names such as hipLaunchKernel are deliberately not enough.
    events = trace.get('traceEvents', []) if isinstance(trace, dict) else trace
    return [event for event in events
            if event.get('ph') == 'X'
            and any(category.strip().lower() in ('kernel', 'gpu_kernel')
                    for category in str(event.get('cat', '')).split(','))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(f'/tmp/torch-profiler-graph-smoke-{os.getpid()}.json'))
    args = parser.parse_args()
    output_path = args.output.resolve()
    assert output_path.is_absolute() and Path('/tmp') in output_path.parents, 'Output must be under /tmp'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    assert torch.version.hip and torch.cuda.is_available(), 'ROCm GPU required'
    assert torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities(), 'Torch CUDA/ROCm profiling unavailable'
    a = torch.randn((512, 512), device='cuda', dtype=torch.bfloat16)
    b = torch.randn_like(a)
    output = torch.empty_like(a)
    for _ in range(3):
        torch.mm(a, b, out=output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.mm(a, b, out=output)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_stack=False, record_shapes=False, profile_memory=False,
    ) as profiler:
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()

    profiler.export_chrome_trace(str(output_path))
    valid = bool(torch.isfinite(output).all().item())
    graph.reset()
    trace = json.loads(output_path.read_text(encoding='utf-8'))
    kernels = kernel_events(trace)
    device_events = [event for event in profiler.events()
                     if str(event.device_type).lower().endswith('.cuda')]
    complete = valid and len(kernels) >= 5
    print(json.dumps({'complete': complete, 'torch': torch.__version__, 'hip': torch.version.hip,
                      'profiled_graph_replays': 5, 'output_finite': valid,
                      'chrome_trace': str(output_path), 'kernel_event_count': len(kernels),
                      'cuda_device_event_count': len(device_events),
                      'kernel_names': sorted({event.get('name', '') for event in kernels}),
                      'kernel_total_us': sum(float(event.get('dur', 0)) for event in kernels)}), flush=True)
    raise SystemExit(0 if complete else 2)


if __name__ == '__main__':
    main()
