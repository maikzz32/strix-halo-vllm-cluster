"""Exercise the generated MoE function and observe actual kernel dispatch on GPU."""
import argparse, ast, hashlib, importlib, json, os, sys
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.path[:0]=['/opt/strix-halo-next/moe-w1-order-20260907-r1','/opt/strix-halo-next/moe-w2-serial-20260907-r1']
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=9564);args=p.parse_args()
import torch
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
fm=importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe')
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(args.seed)
base_sha=hashlib.sha256(Path(fm.__file__).read_bytes()).hexdigest()
assert base_sha=='8b0d48a769b03c62e786a880941531ebdfb1bc582d53b6e03c5e49e73b3b2493'
candidate=Path(__file__).with_name('candidate.py').read_bytes()
assert hashlib.sha256(candidate).hexdigest()=='ab22219d75f1add289f9418fdc2158f482418e45864dc5e2771157de456874e6'
fn=next(n for n in ast.parse(candidate).body if isinstance(n,ast.FunctionDef) and n.name=='_glm53_moe_int4_gemv')
scope=dict(fm.__dict__);counts=dict(partial=0,w1=0,w2=0)
class Observed:
 def __init__(self,k,name):self.k,self.name=k,name
 def __getitem__(self,grid):
  counts[self.name]+=1
  return self.k[grid]
for symbol,name in [('_glm53_moe_int4_gemv_partial','partial'),('w1_expert_kfast','w1'),('w2_serial','w2')]:scope[symbol]=Observed(scope[symbol],name)
exec(compile(ast.Module(body=[fn],type_ignores=[]),'candidate.py','exec'),scope)
os.environ['VLLM_GFX1X_MOE_INT4_GEMV']='1'
report=dict(status='running',seed=args.seed,source_sha256=base_sha,candidate_sha256=hashlib.sha256(candidate).hexdigest(),cases=[])
def save():args.output.write_text(json.dumps(report,indent=2))
for op,N,K in [('w1',320,2560),('w2',2560,160)]:
 weights=torch.empty((512,N,K//2),device='cuda',dtype=torch.uint8).random_(0,256)
 scales=torch.empty((512,N,K//32),device='cuda',dtype=torch.bfloat16).uniform_(0.002,0.01)
 zp=torch.empty((512,N//2,K//32),device='cuda',dtype=torch.uint8).random_(0,256)
 for M in (2,3,4,5):
  a=torch.randn((M if op=='w1' else M*10,K),device='cuda',dtype=torch.bfloat16)*0.25
  tw=torch.softmax(torch.randn((M,10),device='cuda'),dim=-1)
  for BM in (16,32):
   for pattern,stride in [('reuse',0),('overlap',5),('disjoint',10)]:
    ids=torch.tensor([[(r*stride+j+args.seed)%512 for j in range(10)] for r in range(M)],device='cuda',dtype=torch.int32)
    si,ei,nt=moe_align_block_size(ids,BM,512,None)
    kw=dict(A=a,B=weights,B_scale=scales,B_zp=zp,topk_weights=tw,sorted_token_ids=si,expert_ids=ei,num_tokens_post_padded=nt,mul_routed_weight=op=='w2',top_k=10 if op=='w1' else 1,compute_type=fm.tl.bfloat16,use_int4_w4a16=True,use_int8_w8a16=False,block_shape=[0,32],config={'BLOCK_SIZE_M':BM})
    def call(method):
     os.environ['STRIX_MOE_NEIGHBOR_DEPTHS']='1' if method=='on' else '0'
     os.environ['STRIX_MOE_W1_EXPERT_ORDER']='1';os.environ['STRIX_MOE_W2_SERIAL']='1'
     if method=='disabled':
      os.environ['STRIX_MOE_NEIGHBOR_DEPTHS']='1'
      os.environ['STRIX_MOE_W1_EXPERT_ORDER']='0';os.environ['STRIX_MOE_W2_SERIAL']='0'
     out=torch.empty((M,10,N),device='cuda',dtype=torch.bfloat16)
     before=dict(counts)
     assert (fm._glm53_moe_int4_gemv if method=='installed' else scope['_glm53_moe_int4_gemv'])(C=out,**kw)
     if method!='installed':
      expected=op if BM==16 and method!='disabled' and (M==4 or (method=='on' and M in (3,5))) else 'partial'
      assert {k:counts[k]-before[k] for k in counts}=={k:int(k==expected) for k in counts},(op,M,BM,method,counts,before)
     return out
    reference=call('installed')
    for method in ('off','on','disabled'):
     assert torch.equal(reference,call(method)),(op,M,BM,pattern,method,'eager')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
     graphs={};outputs={}
     for method in ('off','on','disabled'):
      call(method);stream.synchronize();g=torch.cuda.CUDAGraph()
      with torch.cuda.graph(g,stream=stream):out=call(method)
      graphs[method]=g;outputs[method]=out
     a.mul_(0.5);expected=call('installed')
     for g in graphs.values():g.replay()
     stream.synchronize()
     assert all(torch.equal(expected,out) for out in outputs.values()),(op,M,BM,pattern,'graph')
     a.mul_(2)
    torch.cuda.current_stream().wait_stream(stream)
    report['cases'].append(dict(operator=op,verification_tokens=M,block_size_m=BM,pattern=pattern,eager_exact=True,changed_input_graph_exact=True,dispatch_verified=True));save()
    del graphs,outputs
report.update(status='passed',dispatch_counts=counts);save();print(json.dumps(report))
