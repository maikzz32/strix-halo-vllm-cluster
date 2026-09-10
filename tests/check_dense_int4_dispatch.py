"""Validate actual candidate function dispatch and changed-input HIP graphs."""
import ast, hashlib, importlib.util, json
from pathlib import Path
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as installed
spec=importlib.util.spec_from_file_location('dense_generator',Path(__file__).resolve().parents[1]/'patches/dense_int4_grid.py')
generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
source=generator.generate(Path(installed.__file__).read_bytes())
tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_rdna_hybrid_w4a16_apply_impl')
ns=dict(vars(installed),_STRIX_DENSE_GRID=True)
exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-candidate-function>','exec'),ns)
candidate=ns[node.name]
original=ops.wvSplitK_int4_g
seen=[]
def observe(*args,**kw):
 seen.append(args[3]);return original(*args,**kw)
ops.wvSplitK_int4_g=observe
torch.manual_seed(9511);torch.set_num_threads(1)
rows=[]
try:
 for m,n,k in ((1,512,2560),(4,2560,160),(4,512,2560),(1,320,2560),(4,320,2560),(1,2560,160)):
  x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
  q=torch.randint(-128,128,(n,k//2),device='cuda',dtype=torch.int8)
  s=torch.rand((n,k//32),device='cuda',dtype=torch.bfloat16)*.02
  z=torch.randint(0,16,(n,k//32),device='cuda').to(torch.bfloat16)
  expect_grid=40 if (m,n,k) in ((1,512,2560),(4,2560,160)) else 20
  reference=installed._rdna_hybrid_w4a16_apply_impl(x,q,s,z,None,20,32)
  for enabled in (False,True):
   ns['_STRIX_DENSE_GRID']=enabled;seen.clear()
   y=candidate(x,q,s,z,None,20,32)
   assert seen==[expect_grid if enabled else 20],seen
   assert torch.equal(y,reference),(m,n,k,enabled)
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   candidate(x,q,s,z,None,20,32);stream.synchronize()
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g,stream=stream):out=candidate(x,q,s,z,None,20,32)
   for multiplier in (.75,-1.,0.):
    x.mul_(multiplier)
    reference=installed._rdna_hybrid_w4a16_apply_impl(x,q,s,z,None,20,32)
    g.replay();stream.synchronize();assert torch.equal(out,reference)
  torch.cuda.current_stream().wait_stream(stream)
  # Bias is deliberately outside the candidate contract.
  bias=torch.randn(n,device='cuda',dtype=torch.bfloat16);seen.clear()
  y=candidate(x,q,s,z,bias,20,32);assert seen==[20]
  assert torch.equal(y,installed._rdna_hybrid_w4a16_apply_impl(x,q,s,z,bias,20,32))
  rows.append({'shape':[m,n,k],'selected_grid':expect_grid,'flag_off_exact':True,'eager_exact':True,'changed_graph_exact':True,'bias_fallback_exact':True})
finally:ops.wvSplitK_int4_g=original
print('RESULT='+json.dumps({'status':'passed','candidate_sha256':hashlib.sha256(source.encode()).hexdigest(),'base_sha256':generator.BASE_SHA256,'cases':rows}),flush=True)
