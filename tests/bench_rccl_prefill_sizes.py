"""Isolated RCCL latency at decode and statically derived prefill payload sizes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import statistics
import time

p=argparse.ArgumentParser()
p.add_argument('--rank',type=int,required=True)
p.add_argument('--master',required=True)
p.add_argument('--port',type=int,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert a.rank in range(4) and 1024<=a.port<=65535
signal.signal(signal.SIGALRM,lambda *unused:os._exit(124))
signal.alarm(150)
import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
torch.set_num_threads(1)
torch.cuda.set_device(0)
dist.init_process_group('gloo',rank=a.rank,world_size=4,
    init_method=f'tcp://{a.master}:{a.port}',timeout=datetime.timedelta(seconds=25))
comm=PyNcclCommunicator(dist.group.WORLD,device=torch.device('cuda:0'))
assert comm.available and not comm.disabled
rows=[]
try:
    for count in (10240,20480,1290240):
        x=torch.full((count,),a.rank+1,device='cuda',dtype=torch.bfloat16)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(4):y=comm.all_reduce(x)
        stream.synchronize()
        assert bool(torch.all(y==10))
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):y=comm.all_reduce(x)
        try:
            for shift in (0,1):
                x.fill_(a.rank+1+shift)
                torch.cuda.synchronize()
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize()
                assert bool(torch.all(y==10+4*shift))
            samples=[]
            wall=[]
            for _ in range(5):
                dist.barrier()
                start=torch.cuda.Event(enable_timing=True)
                end=torch.cuda.Event(enable_timing=True)
                t=time.perf_counter_ns()
                start.record()
                for _ in range(32):graph.replay()
                end.record()
                end.synchronize()
                wall.append((time.perf_counter_ns()-t)/32000)
                samples.append(start.elapsed_time(end)*1000/32)
            rows.append(dict(values=count,bytes=count*2,changed_input_checks=2,
                event_samples_us=samples,wall_samples_us=wall,median_us=statistics.median(samples)))
        finally:
            torch.cuda.synchronize()
            graph.reset()
    dist.barrier()
finally:
    comm.nccl.ncclCommDestroy(comm.comm)
    comm.available,comm.disabled=False,True
    dist.destroy_process_group()
record=dict(rank=a.rank,status='passed',cases=rows,
    environment={k:v for k,v in os.environ.items() if k.startswith(('NCCL_','GLOO_'))},
    note='Integer sum checks only, not cancellation-rich reduction-order calibration; isolated graph throughput excludes model arrival skew.')
a.output.write_text(json.dumps(record,indent=2))
print('RESULT='+json.dumps(record),flush=True)
signal.alarm(0)
