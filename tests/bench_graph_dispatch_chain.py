"""Synthetic dependent integer launches; not an estimate of model idle time."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch

torch.cuda.init()
paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                if 'libamdhip64.so' in line})
assert len(paths) == 1, paths
rows = []
for count in (1, 128, 1024, 2646):
    x = torch.zeros(256, device='cuda', dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x.add_(1)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(count):
            x.add_(1)
    for value in (7, 193):
        x.fill_(value)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert bool(torch.all(x == value + count))
    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize()
    wall, events = [], []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t = time.perf_counter_ns()
        start.record()
        for _ in range(16):
            graph.replay()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter_ns() - t) / 16000)
        events.append(start.elapsed_time(end) * 1000 / 16)
    rows.append(dict(launches=count, changed_input_checks=2,
                     wall_us=wall, event_us=events,
                     median_wall_us=statistics.median(wall),
                     median_event_us=statistics.median(events)))
    graph.reset()
print('RESULT=' + json.dumps(dict(status='passed', rows=rows,
    torch=torch.__version__, hip=torch.version.hip,
    library=paths[0], library_sha256=hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest(),
    environment={k: os.environ.get(k) for k in
                 ('GPU_MAX_HW_QUEUES', 'HIP_FORCE_DEV_KERNARG', 'AMD_DIRECT_DISPATCH',
                  'DEBUG_HIP_GRAPH_BATCH_SIZE', 'DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING')},
    note='Synthetic cached integer chain, no model, RDMA or mixed kernels.')))
