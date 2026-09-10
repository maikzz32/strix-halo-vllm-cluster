"""Exhaustive finite BF16 gate values with several multipliers; exact output bits."""
import json
import torch,triton
from bench_w1_reduce_silu import base,fused

r,n,s=40,320,4
p=torch.zeros(s,r,n,device='cuda',dtype=torch.float32)
tmp=torch.empty(r,n,device='cuda',dtype=torch.bfloat16)
out=torch.empty(r,n//2,device='cuda',dtype=torch.bfloat16);trial=torch.empty_like(out)
raw=torch.arange(65536,dtype=torch.int32)
raw=raw[(raw&0x7f80)!=0x7f80]
values=raw.to(torch.int16).view(torch.bfloat16).float().cuda()
count=out.numel();checked=0;bad=0;examples=[]
for multiplier in (1.,-1.,0.125,3.25):
 for begin in range(0,values.numel(),count):
  v=values[begin:begin+count];k=v.numel();p.zero_()
  gate=torch.zeros(count,device='cuda');gate[:k]=v
  p[0,:,:n//2]=gate.view(r,n//2);p[0,:,n//2:]=multiplier
  base._glm53_moe_int4_gemv_reduce[(r*triton.cdiv(n,128),)](p,tmp,p,r,n,s,False,128)
  torch.ops._C.silu_and_mul(out,tmp)
  fused[(r,triton.cdiv(n//2,128))](p,trial,r,n,s,128,enable_fp_fusion=False)
  mismatch=out.view(torch.int16).flatten()[:k]!=trial.view(torch.int16).flatten()[:k]
  idx=torch.nonzero(mismatch).flatten();bad+=idx.numel();checked+=k
  for j in idx[:max(0,12-len(examples))].tolist():
   examples.append({'gate':v[j].item(),'multiplier':multiplier,'reference_bits':out.view(torch.int16).flatten()[j].item(),'candidate_bits':trial.view(torch.int16).flatten()[j].item()})
print(json.dumps({'checked':checked,'mismatches':bad,'examples':examples,'finite_gate_values':values.numel()}),flush=True)
