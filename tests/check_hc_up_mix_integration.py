"""Isolated actual-HC-method dispatch/capture checks; no serving source mutation."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['VLLM_GFX1X_HC_QUANT']='0'
import importlib, json, pathlib, types
import torch
import hc_up_mix_runtime as runtime
from hc_up_mix_source import generate
from vllm.model_executor.layers import utils
base=importlib.import_module('vllm.models.qwen4_exp.amd.hyperconnection')
p=pathlib.Path(base.__file__)
patched=types.ModuleType('vllm.models.qwen4_exp.amd._hc_candidate')
patched.__package__=base.__package__
os.environ['STRIX_HC_UP_MIX']='1'
exec(compile(generate(p.read_bytes()), str(p)+'-candidate', 'exec'), patched.__dict__)
torch.set_num_threads(1);torch.manual_seed(9570)
# Actual original and candidate methods; isolated layers avoid a distributed engine.
class Linear(torch.nn.Module):
    def __init__(self,n,k,prefix):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02,requires_grad=False)
        self.prefix=prefix;self.bias=None
        self.quant_method=runtime.UnquantizedLinearMethod()
    def forward(self,x):return utils.rocm_unquantized_gemm_impl(x,self.weight)

def make(prefix,combine):
    m=patched.GatedResidual.__new__(patched.GatedResidual);torch.nn.Module.__init__(m)
    m.config=base.HyperConnectionConfig(hidden_size=2560,hc_lowrank=320,hc_per_branch_norm=True)
    m.hc_count=4;m.hidden_size=2560;m.lora_rank=320;m.use_combine=combine;m.pad_size=12 if combine else 0
    m.hc_norm=torch.nn.Module();m.hc_norm.weight=torch.nn.Parameter(torch.zeros(10240,device='cuda',dtype=torch.bfloat16),requires_grad=False)
    if combine:m.input_mix_weight_down_block_inject=Linear(336,10240,prefix+'.input_mix_weight_down_block_inject')
    else:m.input_mix_weight_down=Linear(320,10240,prefix+'.input_mix_weight_down')
    m.input_mix_weight_up=Linear(10240,320,prefix+'.input_mix_weight_up')
    runtime.prepare(m)
    return m
report=[]
for prefix in ['model.layers.0.attn_hyper_connection','model.mtp.layers.0.attn_hyper_connection']:
 for combine in [True,False]:
    m=make(prefix,combine)
    assert m._strix_up_mix_selected == ('mtp' not in prefix)
    for rows in [1,3,4,5]:
        hidden=torch.randn(rows,10240,device='cuda',dtype=torch.bfloat16)
        prev=torch.randn(rows,2560,device='cuda',dtype=torch.bfloat16)
        inj=torch.randn(rows,4,device='cuda',dtype=torch.bfloat16)
        for name,args in [('mix',(hidden,)),('combine_and_mix',(hidden,prev,inj))]:
            reference=getattr(base.GatedResidual,name)(m,*args)
            result=getattr(m,name)(*args)
            assert all(a is None and b is None or a is not None and b is not None and torch.equal(a,b) for a,b in zip(reference,result)),(prefix,combine,rows,name)
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
        assert any('strix_hc_up_mix' in target for target in captured),captured
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):call(x,xn)
            stream.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):out=call(x,xn)
            x.mul_(.5);xn.add_(.0625);g.replay();stream.synchronize()
            assert torch.equal(out,base.hc_gate_mix(xn,m.input_mix_weight_up(x),4))
        torch.cuda.current_stream().wait_stream(stream)
print('RESULT='+json.dumps(dict(status='passed',checks=report,compiled_fullgraph=True,changed_graph_exact=True)),flush=True)
