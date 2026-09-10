"""Check GDN output/state prefixes across adjacent verification lengths."""
import hashlib,json,os
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1'
import torch
import vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating as stock

def main():
 torch.set_num_threads(1);torch.manual_seed(9565)
 sha=hashlib.sha256(Path(stock.__file__).read_bytes()).hexdigest()
 assert sha=='000ab8996af9788fdb8843a6a3b91833e7a14c8acc0e1ea073a536330f64cb6f'
 H,HV,K,V=4,12,128,128
 rows=[]
 for short,long in [(3,4),(4,5)]:
  for bank in range(4):
   q=torch.randn(1,long,H,K,device='cuda',dtype=torch.bfloat16)*.25
   k=torch.randn_like(q);v=torch.randn(1,long,HV,V,device='cuda',dtype=torch.bfloat16)
   a=torch.randn(long,HV,device='cuda',dtype=torch.bfloat16);b=torch.randn_like(a)
   alog=torch.randn(HV,device='cuda');bias=torch.randn_like(alog)
   initial=torch.randn(16,HV,V,K,device='cuda',dtype=torch.float32)*.02
   storage={n:initial.clone() for n in (short,long)}
   states={n:s[::2] for n,s in storage.items()}
   index_storage=torch.zeros(1,8,device='cuda',dtype=torch.int32)
   index_storage[0,:long]=torch.tensor([1,3,2,4,5][:long],device='cuda')
   indices={n:index_storage[:,:n] for n in (short,long)}
   cu={n:torch.tensor([0,n],device='cuda',dtype=torch.int32) for n in (short,long)}
   accepted=torch.ones(1,device='cuda',dtype=torch.int32)
   def call(n):
    return stock.fused_sigmoid_gating_delta_rule_update(A_log=alog,a=a[:n],b=b[:n],dt_bias=bias,q=q[:,:n],k=k[:,:n],v=v[:,:n],initial_state=states[n],inplace_final_state=True,cu_seqlens=cu[n],ssm_state_indices=indices[n],num_accepted_tokens=accepted,use_qk_l2norm_in_kernel=True)[0]
   shared=[2*i for i in [1,3,2,4,5][:short]]
   untouched={n:[i for i in range(16) if i not in [2*j for j in [1,3,2,4,5][:n]]] for n in (short,long)}
   def reset():
    for s in storage.values():s.copy_(initial)
   def check(out):
    torch.cuda.synchronize()
    assert torch.equal(out[short],out[long][:,:short])
    assert torch.equal(storage[short][shared],storage[long][shared])
    assert all(torch.equal(storage[n][untouched[n]],initial[untouched[n]]) for n in (short,long))
   for count in range(1,short+1):
    accepted.fill_(count);reset();out={n:call(n) for n in (short,long)};check(out)
    expected={n:out[n].clone() for n in out};expected_states={n:s.clone() for n,s in storage.items()}
    reset();out={n:call(n) for n in (short,long)}
    assert all(torch.equal(out[n],expected[n]) and torch.equal(storage[n],expected_states[n]) for n in out)
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    graphs={};outputs={}
    for n in (short,long):
     reset();call(n);stream.synchronize();g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g,stream=stream):out=call(n)
     graphs[n]=g;outputs[n]=out
    q.mul_(.5);a.add_(.03125)
    for count in range(1,short+1):
     accepted.fill_(count);reset();expected={n:call(n).clone() for n in (short,long)}
     expected_states={n:s.clone() for n,s in storage.items()}
     reset()
     for g in graphs.values():g.replay()
     stream.synchronize();check(outputs)
     assert all(torch.equal(outputs[n],expected[n]) and torch.equal(storage[n],expected_states[n]) for n in outputs)
     rows.append(dict(short_rows=short,long_rows=long,bank=bank,accepted=count,eager_prefix_exact=True,state_prefix_exact=True,fixed_shape_repeatable=True,changed_input_graph_exact=True,untouched_state_exact=True))
   torch.cuda.current_stream().wait_stream(stream)
 print('RESULT='+json.dumps(dict(cases=rows,source_sha256={str(Path(stock.__file__)):sha},note='Isolated installed post-convolution GDN recurrence; valid state slots, C1, FP32 state. No convolution, norm, model metadata generation or full-model rollback test.')),flush=True)
if __name__=='__main__':main()
