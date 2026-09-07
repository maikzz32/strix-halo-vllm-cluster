"""Isolated exact W2 reduction/expert-sum fusion. No serving edits."""
import argparse, hashlib, json, os, statistics
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'

def run(args):
 import torch
 import triton
 from vllm.model_executor.layers.fused_moe.fused_moe import _glm53_moe_int4_gemv_reduce as reduce
 from vllm import _custom_ops as ops
 from moe_w2_reduce_sum_kernel import w2_reduce_sum
 import inspect
 source=Path(inspect.getfile(reduce.fn))
 assert hashlib.sha256(source.read_bytes()).hexdigest()=='4f460bcc4c6c074dc8c08c3881bccdbdbd941490821bac458c9782c1404a0070'
 torch.set_num_threads(1);torch.manual_seed(9327)
 report={'status':'running','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':[]}
 def save():args.output.write_text(json.dumps(report,indent=2))
 for pattern in ('normal','cancellation','wide_range'):
  pool=[]
  for bank in range(24):
   p=torch.randn((5,40,2560),device='cuda',dtype=torch.float32)
   if pattern=='cancellation':p[1].copy_(-p[0]);p[3].copy_(-p[2])
   if pattern=='wide_range':p.mul_(torch.pow(2.0,torch.randint(-12,12,(5,40,1),device='cuda')))
   w=torch.softmax(torch.randn((4,10),device='cuda'),dim=-1)
   pool.append((p,w))
  case={'pattern':pattern,'pool_bytes':sum(p.numel()*4+w.numel()*4 for p,w in pool),'methods':[]};report['cases'].append(case);save()
  def call(method,item):
   p,w=item;out=torch.empty((4,2560),device='cuda',dtype=torch.bfloat16)
   if method=='installed':
    mid=torch.empty((4,10,2560),device='cuda',dtype=torch.bfloat16)
    reduce[(40*5,)](p,mid,w,40,N=2560,SPLITK=5,MUL_ROUTED=True,BLOCK=512)
    ops.moe_sum(mid,out)
   else:
    block=int(method.split('_')[1]);w2_reduce_sum[(4,triton.cdiv(2560,block))](p,w,out,ROUTES=40,N=2560,TOPK=10,SPLITK=5,BLOCK=block,enable_fp_fusion=False)
   return out
  for method in ('installed','fused_64','fused_128','fused_256'):
   mismatches=sum(int((call(method,item).view(torch.int16)!=call('installed',item).view(torch.int16)).sum()) for item in pool)
   row={'method':method,'bit_mismatches':mismatches};case['methods'].append(row);save()
   if mismatches:continue
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for item in pool:call(method,item)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):outputs=[call(method,item) for item in pool]
    graph.replay();stream.synchronize()
    assert all(torch.equal(y,call('installed',item)) for y,item in zip(outputs,pool))
    for p,w in pool:p.mul_(0.5)
    expected=[call('installed',item) for item in pool];graph.replay();stream.synchronize()
    assert all(torch.equal(y,e) for y,e in zip(outputs,expected))
    row['changed_input_graph_exact']=True
    for p,w in pool:p.mul_(2.0)
    samples=[]
    for sample in range(5):
     start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
     for repeat in range(20):graph.replay()
     end.record();end.synchronize();samples.append(start.elapsed_time(end)*1000/(20*24))
    row.update(samples_us=samples,median_us=statistics.median(samples));save()
   torch.cuda.current_stream().wait_stream(stream)
  torch.cuda.synchronize()
 report['status']='passed' if all(x['bit_mismatches']==0 for c in report['cases'] for x in c['methods']) else 'numerical_failure';save();print(json.dumps(report))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args())
