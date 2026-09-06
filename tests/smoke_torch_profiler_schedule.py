"""Isolated schedule diagnosis; each mode must run in a fresh process.

No vLLM runtime mutation, no model loading. Uses the existing tiny graph smoke
shape and tests actual vLLM wrapper semantics or the direct Torch equivalent.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['raw_scheduled', 'raw_unscheduled',
                                          'vllm_scheduled', 'vllm_unscheduled'], required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    assert torch.cuda.is_available() and torch.version.hip
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
    path = Path(f'/tmp/torch-profiler-{args.mode}-{time.time_ns()}.json.gz')
    scheduled = args.mode.endswith('_scheduled')
    emitted = []

    def export(profiler):
        raw = path.with_suffix('')
        profiler.export_chrome_trace(str(raw))
        path.write_bytes(gzip.compress(raw.read_bytes()))
        raw.unlink()
        emitted.append(str(path))

    if args.mode.startswith('vllm_'):
        from vllm.config import ProfilerConfig
        from vllm.profiler.wrapper import TorchProfilerWrapper
        config = ProfilerConfig(profiler='torch', torch_profiler_dir='/tmp',
                                torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
                                torch_profiler_with_memory=False, torch_profiler_with_flops=False,
                                torch_profiler_use_gzip=True, torch_profiler_dump_cuda_time_total=False,
                                ignore_frontend=True, warmup_iterations=2 if scheduled else 0,
                                wait_iterations=0, active_iterations=16, max_iterations=16)
        wrapper = TorchProfilerWrapper(config, worker_name=args.mode, local_rank=0,
                                       activities=['CPU', 'CUDA'], on_trace_ready=export)
        wrapper.start()
        states = []
        for index in range(1, 19):
            wrapper.step()  # Same placement as GPUWorker.annotate_profile.
            states.append({'worker_call': index, 'running': wrapper.is_running,
                           'action': str(wrapper.profiler.current_action)})
            graph.replay()
        wrapper.stop()
        profiler = wrapper.profiler
    else:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_stack=False, record_shapes=False, profile_memory=False, with_flops=False,
            schedule=torch.profiler.schedule(wait=0, warmup=2, active=16, repeat=1) if scheduled else None,
            on_trace_ready=export)
        profiler.start()
        states = []
        for index in range(1, 19 if scheduled else 17):
            profiler.step()
            states.append({'worker_call': index, 'action': str(profiler.current_action)})
            graph.replay()
        profiler.stop()
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(output).all().item())
    graph.reset()
    assert emitted, 'No trace emitted'
    trace = json.loads(gzip.decompress(path.read_bytes()))
    events = trace.get('traceEvents', [])
    kernels = [event for event in events if event.get('ph') == 'X' and event.get('cat') == 'kernel']
    print(json.dumps({'mode': args.mode, 'finite': finite, 'kernel_event_count': len(kernels),
                      'cuda_device_event_count': sum(str(event.device_type).lower().endswith('.cuda') for event in profiler.events()),
                      'categories': dict(Counter(event.get('cat', '') for event in events)),
                      'kernel_names': sorted({event.get('name', '') for event in kernels}),
                      'states': states, 'trace': str(path), 'torch': torch.__version__,
                      'hip': torch.version.hip, 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                      'gpu_complete': True}), flush=True)
    assert finite


if __name__ == '__main__':
    main()
