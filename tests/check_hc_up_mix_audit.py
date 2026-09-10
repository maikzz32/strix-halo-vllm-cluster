"""Isolated actual-HC-method dispatch/capture checks; no serving source mutation."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['VLLM_GFX1X_HC_QUANT']='0'
import importlib, json, pathlib, types, sys
import torch
import hc_up_mix_runtime as runtime
from hc_up_mix_source import generate
from vllm.model_executor.layers import utils
base=importlib.import_module('vllm.models.qwen4_exp.amd.hyperconnection')
p=pathlib.Path(base.__file__)
patched=types.ModuleType('vllm.models.qwen4_exp.amd._hc_candidate')
patched.__package__=base.__package__
sys.modules[patched.__name__]=patched
os.environ['STRIX_HC_UP_MIX']='1'
exec(compile(generate(p.read_bytes()), str(p)+'-candidate', 'exec'), patched.__dict__)
torch.set_num_threads(1);torch.manual_seed(9571);torch.set_grad_enabled(False)
# A private single-process group is sufficient for these replicated HC layers.
import tempfile
from vllm.config import VllmConfig, set_current_vllm_config
config_context=set_current_vllm_config(VllmConfig())
config_context.__enter__()
from vllm.distributed import init_distributed_environment, initialize_model_parallel
init_distributed_environment(world_size=1,rank=0,local_rank=0,
    distributed_init_method='file://'+tempfile.mkdtemp(prefix='hc-test-group-')+'/init',backend='gloo')
initialize_model_parallel(backend='gloo')

def make(prefix,combine):
    with torch.device('cuda'):
        m=patched.GatedResidual(base.HyperConnectionConfig(hidden_size=2560,
            hc_lowrank=320,hc_per_branch_norm=True),use_combine=combine,prefix=prefix)
    with torch.no_grad():
        for name,param in m.named_parameters():
            if 'hc_norm' in name:param.zero_()
            else:param.normal_(0,.02)
    assert not any('_strix' in key for key in m.state_dict())
    return m
report=[]
for prefix in ['model.layers.0.attn_hyper_connection','model.mtp.layers.0.attn_hyper_connection']:
 for combine in [True,False]:
    m=make(prefix if combine else prefix.replace('.0.','.1.'),combine)
    assert m._strix_up_mix_selected == ('mtp' not in prefix)
    for rows in [1,3,4,5]:
        hidden=torch.randn(rows,10240,device='cuda',dtype=torch.bfloat16)
        prev=torch.randn(rows,2560,device='cuda',dtype=torch.bfloat16)
        inj=torch.randn(rows,4,device='cuda',dtype=torch.bfloat16)
        for name,args in [('mix',(hidden,)),('combine_and_mix',(hidden,prev,inj))]:
            reference=getattr(base.GatedResidual,name)(m,*args)
            result=getattr(m,name)(*args)
            assert all(a is None and b is None or a is not None and b is not None and torch.equal(a,b) for a,b in zip(reference,result)),(prefix,combine,rows,name)
            if m._strix_up_mix_selected and rows == 4:
                full=torch.compile(getattr(m,name),backend='inductor',fullgraph=True)
                actual=full(*args)
                assert all(a is None and b is None or a is not None and b is not None and torch.equal(a,b) for a,b in zip(reference,actual)),('inductor-method',combine,name)
            report.append(dict(prefix=prefix,combine=combine,rows=rows,method=name,exact=True))
    if m._strix_up_mix_selected:
        x=torch.randn(4,320,device='cuda',dtype=torch.bfloat16);xn=torch.randn(4,10240,device='cuda',dtype=torch.bfloat16)
        def call(x,xn):return runtime.mix(m,x,xn,base.hc_gate_mix)
        captured=[]
        def backend(gm, inputs):
            captured.extend(str(n.target) for n in gm.graph.nodes if n.op=='call_function')
            return gm.forward
        compiled=torch.compile(call,backend=backend,fullgraph=True)
        assert torch.equal(call(x,xn),compiled(x,xn))
        assert any('strix_hc_up_mix_audit' in target for target in captured),captured
        optimized=torch.compile(call,backend='inductor',fullgraph=True)
        assert torch.equal(call(x,xn),optimized(x,xn))
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):call(x,xn)
            stream.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):out=optimized(x,xn)
            x.mul_(.5);xn.add_(.0625);g.replay();stream.synchronize()
            assert torch.equal(out,base.hc_gate_mix(xn,m.input_mix_weight_up(x),4))
        torch.cuda.current_stream().wait_stream(stream)
for layer in runtime.LAYERS.values():
    state=layer._strix_audit_state.tolist()
    assert state[0]>0 and state[1:]==[0,0,0],state
print('RESULT='+json.dumps(dict(status='passed',real_constructor=True,inductor_pair=True,inductor_full_methods=True,checks=report,compiled_fullgraph=True,changed_graph_exact=True)),flush=True)

from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
destroy_model_parallel();destroy_distributed_environment()

config_context.__exit__(None,None,None)
