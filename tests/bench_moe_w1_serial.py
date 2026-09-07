"""Rotating-expert W1 geometry comparison; same split-K and arithmetic."""
import argparse, hashlib, importlib, json, os, statistics
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=9340);args=p.parse_args()
 import torch
 from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(args.seed)
 fm=importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe')
 source=Path(fm.__file__).read_text(encoding='utf-8');sha=hashlib.sha256(source.encode()).hexdigest()
 assert sha=='25ac1a380733b9f3ee3735cc58779e2d1215d5f87ed9e601cfc839d49c1403b7'
 os.environ['VLLM_GFX1X_MOE_INT4_GEMV']='1'
 from moe_w1_serial_kernel import w1_serial
 def serial(bn,warps,**kw):
  assert kw['A'].shape==(4,2560) and kw['top_k']==10 and not kw['mul_routed_weight']
  w1_serial[(40,320//bn)](kw['A'],kw['B'],kw['B_scale'],kw['B_zp'],kw['C'],kw['topk_weights'],kw['sorted_token_ids'],kw['expert_ids'],kw['num_tokens_post_padded'],40,2560,N=320,K=2560,GS=32,TOPK=10,BM=16,BN=bn,BK=128,SPLITK=4,num_warps=warps,num_stages=2,enable_fp_fusion=False)
  return True
 variants={'installed':fm._glm53_moe_int4_gemv,'serial64w4s2':lambda **kw:serial(64,4,**kw),'serial32w2s2':lambda **kw:serial(32,2,**kw)}

 E,N,K,GS,M,TOPK,BM=512,320,2560,32,4,10,16
 weights=torch.empty((E,N,K//2),device='cuda',dtype=torch.uint8).random_(0,256)
 scales=torch.empty((E,N,K//GS),device='cuda',dtype=torch.bfloat16).uniform_(0.002,0.01)
 zp=torch.empty((E,N//2,K//GS),device='cuda',dtype=torch.uint8).random_(0,256)
 a=torch.randn((M,K),device='cuda',dtype=torch.bfloat16)*0.25
 common=dict(A=a,B=weights,B_scale=scales,B_zp=zp,topk_weights=torch.softmax(torch.randn((M,TOPK),device="cuda"),dim=-1),mul_routed_weight=False,top_k=10,compute_type=fm.tl.bfloat16,use_int4_w4a16=True,use_int8_w8a16=False,block_shape=[0,GS],config={'BLOCK_SIZE_M':BM})
 report={'status':'running','source_sha256':sha,'seed':args.seed,'cases':[]}
 def save():args.output.write_text(json.dumps(report,indent=2))
 for pattern in ('reuse10','overlap25','disjoint40'):
  pool=[];selected=set()
  for bank in range(24):
   rows=[[(bank*19+(0 if pattern=='reuse10' else (r*5 if pattern=='overlap25' else r*10))+j)%512 for j in range(10)] for r in range(4)]
   selected.update(sum(rows,[]));ids=torch.tensor(rows,device='cuda',dtype=torch.int32)
   si,ei,nt=moe_align_block_size(ids,BM,E,None)
   pool.append(dict(sorted_token_ids=si,expert_ids=ei,num_tokens_post_padded=nt))
  case={'pattern':pattern,'distinct_experts':len(selected),'packed_working_set_bytes':len(selected)*(N*K//2+N*(K//GS)*2+(N//2)*(K//GS)),'methods':[]};report['cases'].append(case);save()
  def call(method,route):
   out=torch.empty((M,TOPK,N),device='cuda',dtype=torch.bfloat16)
   assert variants[method](C=out,**common,**route)
   return out
  mismatches={method:sum(int((call('installed',route).view(torch.int16)!=call(method,route).view(torch.int16)).sum()) for route in pool) for method in variants}
  case['bit_mismatches']=mismatches;save();assert all(v==0 for v in mismatches.values())
  graphs={};outputs={}
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for method in variants:
    for route in pool:call(method,route)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=[call(method,route) for route in pool]
    graphs[method]=graph;outputs[method]=out
   for graph in graphs.values():graph.replay()
   stream.synchronize();assert all(torch.equal(x,y) for out in outputs.values() for x,y in zip(outputs['installed'],out))
   a.mul_(0.5)
   expected=[call('installed',route) for route in pool]
   for graph in graphs.values():graph.replay()
   stream.synchronize();assert all(torch.equal(y,e) for out in outputs.values() for y,e in zip(out,expected))
   case['changed_input_graph_exact']=True;a.mul_(2.0)
   samples={method:[] for method in variants}
   for sample in range(5):
    for method in (list(variants) if sample%2==0 else list(reversed(variants))):
     start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
     for repeat in range(20):graphs[method].replay()
     end.record();end.synchronize();samples[method].append(start.elapsed_time(end)*1000/(20*24))
   case['methods']=[{'method':m,'samples_us':v,'median_us':statistics.median(v)} for m,v in samples.items()];save()
  torch.cuda.current_stream().wait_stream(stream)
  del graphs,outputs
 report['status']='passed';save();print(json.dumps(report))
if __name__=='__main__':main()
