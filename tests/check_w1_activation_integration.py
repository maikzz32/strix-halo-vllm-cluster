"""Candidate full GEMM parity in an isolated process; suppress duplicate op registration only."""
import ast,hashlib,importlib,importlib.util,json,os,sys
from pathlib import Path
os.environ.update(VLLM_GFX1X_MOE_INT4_GEMV='1',STRIX_MOE_W1_EXPERT_ORDER='1',STRIX_W1_REDUCE_SILU='1')
import torch
import vllm._custom_ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
fm=importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe')
path=Path(sys.argv[1]);source=path.read_text();tree=ast.parse(source)
tree.body=[x for x in tree.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Name) and x.value.func.id=='direct_register_custom_op')]
name='vllm.model_executor.layers.fused_moe.strix_candidate_test'
spec=importlib.util.spec_from_file_location(name,path);candidate=importlib.util.module_from_spec(spec);sys.modules[name]=candidate
exec(compile(tree,str(path),'exec'),candidate.__dict__)
torch.manual_seed(9602);torch.set_num_threads(1)
E,N,K,GS,M,TOPK,BM=512,320,2560,32,4,10,16
w=torch.empty((E,N,K//2),device='cuda',dtype=torch.uint8).random_(0,256)
sc=torch.empty((E,N,K//GS),device='cuda',dtype=torch.bfloat16).uniform_(.002,.01)
zp=torch.empty((E,N//2,K//GS),device='cuda',dtype=torch.uint8).random_(0,256)
a=torch.randn((M,K),device='cuda',dtype=torch.bfloat16)*.25
c=torch.empty((M,TOPK,N),device='cuda',dtype=torch.bfloat16);fused=torch.empty((40,160),device='cuda',dtype=torch.bfloat16);ref=torch.empty_like(fused)
common=dict(A=a,B=w,C=c,B_scale=sc,B_zp=zp,topk_weights=None,mul_routed_weight=False,top_k=10,compute_type=fm.tl.bfloat16,use_int4_w4a16=True,use_int8_w8a16=False,block_shape=[0,GS],config={'BLOCK_SIZE_M':BM})
rows=[]
for pattern in ('reuse10','overlap25','disjoint40'):
 for bank in range(24):
  ids=torch.tensor([[(bank*19+(0 if pattern=='reuse10' else r*(5 if pattern=='overlap25' else 10))+j)%512 for j in range(10)] for r in range(4)],device='cuda',dtype=torch.int32)
  si,ei,nt=moe_align_block_size(ids,BM,E,None)
  route=dict(sorted_token_ids=si,expert_ids=ei,num_tokens_post_padded=nt)
  assert fm._glm53_moe_int4_gemv(**common,**route) is True
  torch.ops._C.silu_and_mul(ref,c.view(40,320));old=c.clone()
  assert candidate._glm53_moe_int4_gemv(**common,**route,activation_output=fused)==2
  assert torch.equal(ref.view(torch.int16),fused.view(torch.int16)),(pattern,bank)
  os.environ['STRIX_W1_REDUCE_SILU']='0'
  assert candidate._glm53_moe_int4_gemv(**common,**route,activation_output=fused) is True
  assert torch.equal(c.view(torch.int16),old.view(torch.int16))
  os.environ['STRIX_W1_REDUCE_SILU']='1'
 assert candidate.invoke_fused_moe_wna16_triton_kernel(**common,**route,activation_output=fused) is True
 assert torch.equal(ref.view(torch.int16),fused.view(torch.int16))
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  candidate.invoke_fused_moe_wna16_triton_kernel(**common,**route,activation_output=fused)
 stream.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=stream):
  candidate.invoke_fused_moe_wna16_triton_kernel(**common,**route,activation_output=fused)
 a.normal_().mul_(.25);torch.cuda.synchronize();g.replay();torch.cuda.synchronize()
 assert fm._glm53_moe_int4_gemv(**common,**route) is True
 torch.ops._C.silu_and_mul(ref,c.view(40,320))
 assert torch.equal(ref.view(torch.int16),fused.view(torch.int16))
 g.reset()
 rows.append({'routing':pattern,'banks':24,'fused_exact':True,'disabled_fallback_exact':True,'wrapper_and_changed_graph_exact':True})
print(json.dumps({'candidate_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':rows,'note':'Full W1 GEMM helper tested. Expert caller and model serving remain untested.'}),flush=True)
