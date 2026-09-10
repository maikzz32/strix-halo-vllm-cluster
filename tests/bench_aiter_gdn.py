"""Isolated packed GDN MTP recurrence parity and timing; no serving patch."""
import argparse,hashlib,json,statistics
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=9340);p.add_argument('--staged-wrapper',action='store_true');args=p.parse_args()
 import torch,importlib
 import vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating as stock
 from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import fused_rearrange_sigmoid_gated_delta_rule as candidate
 mod=importlib.import_module('aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr')
 from aiter_gdn_null_guard import fused_rearrange_sigmoid_gated_delta_rule_update_kernel
 mod.fused_rearrange_sigmoid_gated_delta_rule_update_kernel=fused_rearrange_sigmoid_gated_delta_rule_update_kernel
 if args.staged_wrapper:
  import importlib.util,sys
  root=Path('/opt/strix-halo-next/aiter-gdn-20260907-r1')
  manifest=json.loads(root.joinpath('manifest.json').read_text())
  for name in ('_strix_gdn_kernel','_strix_gdn'):
   path=root/(name+'.py')
   assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files'][path.name]
   qualified='vllm.model_executor.layers.mamba.gdn.'+name
   spec=importlib.util.spec_from_file_location(qualified,path)
   module=importlib.util.module_from_spec(spec);sys.modules[qualified]=module;spec.loader.exec_module(module)
  candidate=module.fused_rearrange_sigmoid_gated_delta_rule

 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(args.seed)
 sha=hashlib.sha256(Path(stock.__file__).read_bytes()).hexdigest();assert sha=='000ab8996af9788fdb8843a6a3b91833e7a14c8acc0e1ea073a536330f64cb6f'
 H,HV,K,V=4,12,128,128
 @torch.compile(fullgraph=True,dynamic=False)
 def split_reference(x):
  q,k,v=torch.split(x,[H*K,H*K,HV*V],dim=-1)
  flat=torch.cat([q.reshape(-1),k.reshape(-1),v.reshape(-1)])
  t=x.shape[0];n=t*H*K
  return flat[:n].view(1,t,H,K),flat[n:2*n].view(1,t,H,K),flat[2*n:].view(1,t,HV,V)
 report={'staged_wrapper':args.staged_wrapper,'status':'running','seed':args.seed,'source_sha256':sha,'scope':'Post-convolution MTP recurrence, packed QKV including torch.compile fused reference split; zero sentinel adapted; excludes conv/norm','cases':[]}
 def save():args.output.write_text(json.dumps(report,indent=2))
 for batch in (1,2,4,8):
  for dtype in (torch.float32,torch.bfloat16):
   T=batch*4;dim=2*H*K+HV*V
   packed=torch.randn(dim,T,device='cuda',dtype=torch.bfloat16).t()*.25
   a=torch.randn(T,HV,device='cuda',dtype=torch.bfloat16);b=torch.randn_like(a)
   alog=torch.randn(HV,device='cuda',dtype=torch.float32);bias=torch.randn_like(alog)
   initial_storage=torch.randn(batch*12,HV,V,K,device='cuda',dtype=dtype)*.02
   initial=initial_storage[::2]
   indices=torch.tensor([[i*6+j for j in (1,3,2,4)] for i in range(batch)],device='cuda',dtype=torch.int32)
   index_storage=torch.zeros(batch,8,device='cuda',dtype=torch.int32);index_storage[:,:4].copy_(indices);indices=index_storage[:,:4]
   cu=torch.arange(0,T+1,4,device='cuda',dtype=torch.int32);accepted=torch.ones(batch,device='cuda',dtype=torch.int32)
   state_storage={k:initial_storage.clone() for k in ('stock','aiter')}
   states={k:v[::2] for k,v in state_storage.items()}
   def call(name):
    kw=dict(A_log=alog,a=a,b=b,dt_bias=bias,initial_state=states[name],inplace_final_state=True,cu_seqlens=cu,ssm_state_indices=indices,num_accepted_tokens=accepted,use_qk_l2norm_in_kernel=True)
    if name=='stock':
     q,k,v=split_reference(packed)
     return stock.fused_sigmoid_gating_delta_rule_update(q=q,k=k,v=v,**kw)[0]
    return candidate(qkv=packed,key_dim=H*K,value_dim=HV*V,head_k_dim=K,head_v_dim=V,**kw)[0]
   case={'batch':batch,'state_dtype':str(dtype),'qkv_stride':list(packed.stride()),'index_stride':list(indices.stride()),'state_stride':list(initial.stride()),'checks':[]};report['cases'].append(case);save()
   for bank in range(8):
    packed.normal_();a.normal_();b.normal_()
    indices[:,2]=0 if bank%2 else torch.arange(batch,device='cuda',dtype=torch.int32)*6+2
    for count in (1,2,3,4):
     accepted.fill_(count)
     for s in states.values():s.copy_(initial)
     x=call('stock');y=call('aiter');torch.cuda.synchronize()
     check={'bank':bank,'accepted':count,'output_defined':not(bank%2 and count==3),'output_exact':True if bank%2 and count==3 else torch.equal(x.view(torch.int16),y.view(torch.int16)),'state_exact':torch.equal(state_storage['stock'].view(torch.uint8),state_storage['aiter'].view(torch.uint8))}
     case['checks'].append(check);save()
     assert check['output_exact'] and check['state_exact'],check
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());graphs={};outputs={}
   with torch.cuda.stream(stream):
    for name in states:
     call(name);stream.synchronize();g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g,stream=stream):out=call(name)
     graphs[name]=g;outputs[name]=out
    for count in (1,3,2,4):
     accepted.fill_(count);packed.mul_(.5);a.add_(.03125)
     for s in states.values():s.copy_(initial)
     for g in graphs.values():g.replay()
     stream.synchronize();assert (count==3 or torch.equal(outputs['stock'],outputs['aiter'])) and torch.equal(state_storage['stock'].view(torch.uint8),state_storage['aiter'].view(torch.uint8))
    case['changed_input_acceptance_graph_exact']=True
    indices[:,2]=torch.arange(batch,device='cuda',dtype=torch.int32)*6+2
    saved_indices=indices.clone()
    accepted.copy_(torch.arange(batch,device='cuda',dtype=torch.int32)%4+1)
    if batch>1:indices[-1].zero_()
    for st in states.values():st.copy_(initial)
    for g in graphs.values():g.replay()
    stream.synchronize()
    valid_tokens=T-4 if batch>1 else T
    assert torch.equal(outputs['stock'][:,:valid_tokens].view(torch.int16),outputs['aiter'][:,:valid_tokens].view(torch.int16))
    assert torch.equal(state_storage['stock'].view(torch.uint8),state_storage['aiter'].view(torch.uint8))
    case['mixed_acceptance_padded_graph_exact']=True
    case['padded_requests']=int(batch>1)
    indices.copy_(saved_indices);accepted.fill_(4)
    indices[:,2]=torch.arange(batch,device='cuda',dtype=torch.int32)*6+2
    case['timing_all_state_slots_valid']=True
    samples={name:[] for name in states}
    for repeat in range(5):
     for name in (list(states) if repeat%2==0 else list(reversed(states))):
      states[name].copy_(initial);start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
      for _ in range(100):graphs[name].replay()
      end.record();end.synchronize();samples[name].append(start.elapsed_time(end)*10)
    case['timing_us']={name:{'samples':v,'median':statistics.median(v)} for name,v in samples.items()};save()
   torch.cuda.current_stream().wait_stream(stream)
 report['status']='passed';save();print(json.dumps(report))
if __name__=='__main__':main()
