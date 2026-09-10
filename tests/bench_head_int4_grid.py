"""Isolated launch-geometry test of the installed asymmetric INT4 HIP op."""
import json, statistics, os
os.environ['OMP_NUM_THREADS']='1'
import torch
from vllm import _custom_ops as ops
from vllm.utils.platform_utils import num_compute_units

torch.set_num_threads(1); torch.manual_seed(9566)
base=num_compute_units()
report={'reported_compute_units':base,'note':'Synthetic rotating weights; launch-grid argument, not physical CU count. No serving gain claim.','cases':[]}
for m in (1,4):
 for n,k in ((62080,2560),):
  x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
  pool=[(torch.randint(-128,128,(n,k//2),device='cuda',dtype=torch.int8),torch.rand((n,k//32),device='cuda',dtype=torch.bfloat16)*.02,torch.randint(0,16,(n,k//32),device='cuda').to(torch.bfloat16)) for _ in range(4)]
  def call(grid,w):return ops.wvSplitK_int4_g(w[0],x,w[1],grid,32,w[2],None)
  grids=list(dict.fromkeys([base,10,30,40,60]))
  expected=[call(base,w) for w in pool]
  for grid in grids:
   assert all(torch.equal(call(grid,w),e) for w,e in zip(pool,expected)),('eager parity',m,n,k,grid)
  graphs={};outputs={};stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for grid in grids:
    for w in pool:call(grid,w)
    stream.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):out=[call(grid,w) for w in pool]
    graphs[grid]=g;outputs[grid]=out
   x.mul_(.75);changed=[call(base,w) for w in pool]
   for grid,g in graphs.items():
    g.replay();stream.synchronize()
    assert all(torch.equal(y,e) for y,e in zip(outputs[grid],changed)),('changed graph parity',m,n,k,grid)
   samples={grid:[] for grid in grids}
   for r in range(6):
    for grid in (grids if r%2==0 else grids[::-1]):
     a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
     for _ in range(20):graphs[grid].replay()
     b.record();b.synchronize();samples[grid].append(a.elapsed_time(b)*1000/80)
  torch.cuda.current_stream().wait_stream(stream)
  row={'shape':[m,n,k],'parity':'exact eager and changed-input graph','samples_us':samples,'median_us':{g:statistics.median(v) for g,v in samples.items()}}
  report['cases'].append(row);print('CASE='+json.dumps(row),flush=True)
  del graphs,outputs,expected,changed,pool,x
report['status']='passed';print('RESULT='+json.dumps(report),flush=True)
