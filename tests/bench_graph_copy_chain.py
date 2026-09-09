"""Bounded synthetic D2D copy chain; not an additive estimate of model latency."""
import ctypes,json,statistics,sys
from pathlib import Path
import torch
sizes=json.loads(Path(sys.argv[1]).read_text());assert len(sizes)==88
torch.cuda.init()
paths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libamdhip64.so' in line}
assert len(paths)==1,paths
hip=ctypes.CDLL(next(iter(paths)))
copy=hip.hipMemcpyAsync;copy.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p];copy.restype=ctypes.c_int
x=torch.empty(sum(sizes),device='cuda',dtype=torch.uint8);y=torch.empty_like(x)
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
def copies():
    offset=0
    for size in sizes:
        assert copy(y.data_ptr()+offset,x.data_ptr()+offset,size,3,stream.cuda_stream)==0
        offset+=size
with torch.cuda.stream(stream):copies()
stream.synchronize();graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph,stream=stream):copies()
for value in (19,231):
    x.fill_(value);torch.cuda.synchronize();graph.replay();torch.cuda.synchronize()
    assert torch.equal(x,y)
samples=[]
for _ in range(5):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(32):graph.replay()
    end.record();end.synchronize();samples.append(start.elapsed_time(end)*1000/32)
print(json.dumps({'copies':len(sizes),'bytes':sum(sizes),'fresh_input_checks':2,'replays_per_sample':32,'samples_us':samples,'median_chain_us':statistics.median(samples),'note':'Isolated contiguous-buffer serial copy chain; excludes model kernels, interleaving, cache behavior and communication. Not subtractable from model time.'}))
graph.reset()
