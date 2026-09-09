"""Compare small M tiles against the installed prefill WNA16 defaults."""
import argparse
import ast
import json
import os
from pathlib import Path
import statistics
import sys
import textwrap
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'patches'))
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.update(VLLM_GFX1X_MOE_INT4_GEMV='1', STRIX_MOE_W1_EXPERT_ORDER='1',
                  STRIX_MOE_W2_SERIAL='1', VLLM_ROCM_USE_AITER='0')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, int4_w4a16_moe_quant_config, RoutingMethodType
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonWNA16Experts
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEPModular
    torch.set_num_threads(1)
    torch.manual_seed(9403)
    from vllm.v1.worker.workspace import init_workspace_manager
    init_workspace_manager(torch.device('cuda:0'))
    import vllm.model_executor.layers.fused_moe.experts.triton_moe as tm
    import importlib
    from moe_prefill_tiles import transform
    fm = importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe')
    source = transform(Path(fm.__file__).read_text())
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'try_get_optimal_moe_config')
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<candidate>', 'exec'), fm.__dict__)
    tm.try_get_optimal_moe_config = fm.try_get_optimal_moe_config
    par = FusedMoEParallelConfig(tp_size=4, pcp_size=1, dp_size=1, ep_size=1,
        tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0, sp_size=1,
        use_ep=False, all2all_backend='naive', enable_eplb=False)
    cfg = FusedMoEConfig(num_experts=512, experts_per_token=10, hidden_dim=2560,
        intermediate_size=640, num_local_experts=512, num_logical_experts=512,
        activation=MoEActivation.SILU, device='cuda', routing_method=RoutingMethodType.Default,
        moe_parallel_config=par, in_dtype=torch.bfloat16)
    def random(shape, dtype):
        t = torch.empty(shape, device='cuda', dtype=dtype)
        return t.random_(0, 256) if dtype == torch.uint8 else t.uniform_(0.002, 0.01)
    w1 = random((512,320,1280), torch.uint8)
    w2 = random((512,2560,80), torch.uint8)
    qc = int4_w4a16_moe_quant_config(random((512,320,80),torch.bfloat16),
        random((512,2560,5),torch.bfloat16), random((512,160,80),torch.uint8),
        random((512,1280,5),torch.uint8), block_shape=[0,32])
    kernel = mk.FusedMoEKernelModularImpl(MoEPrepareAndFinalizeNoDPEPModular(), TritonWNA16Experts(cfg,qc))
    report = {'status':'running','cases':[]}
    for m in (4,32,63,64,65,128,512,1024,1025,2048):
        x = torch.randn((m,2560),device='cuda',dtype=torch.bfloat16)*0.25
        original_x = x.clone()
        ids = torch.stack([torch.randperm(512,device='cuda')[:10] for _ in range(m)]).int()
        weights = torch.softmax(torch.randn((m,10),device='cuda'),-1)
        def call(mode):
            os.environ['STRIX_MOE_PREFILL_TILES'] = str(mode)
            return kernel.apply(x,w1,w2,ids,weights)
        a=call(0)
        for mode in (1,):
            b=call(mode)
            assert torch.equal(a.view(torch.int16),b.view(torch.int16)), ('parity',m,mode,int((a!=b).sum()))
        assert torch.equal(x,original_x)
        graphs={}; outputs={}
        stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for mode in (0,1):
                for _ in range(3):call(mode)
                stream.synchronize()
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g,stream=stream):
                    out=call(mode)
                graphs[mode]=g;outputs[mode]=out
            x.mul_(0.5)
            expected=call(0)
            for g in graphs.values():g.replay()
            stream.synchronize()
            assert all(torch.equal(v.view(torch.int16),expected.view(torch.int16)) for v in outputs.values())
            samples={0:[],1:[]}
            for rep in range(8):
                for mode in ((0,1) if rep%2==0 else (1,0)):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(20):graphs[mode].replay()
                    end.record();end.synchronize()
                    samples[mode].append(start.elapsed_time(end)*50)
        case={'M':m,'eager_exact':True,'graph_changed_input_exact':True,
              'samples_us':samples,'median_us':{k:statistics.median(v) for k,v in samples.items()}}
        report['cases'].append(case)
        args.output.write_text(json.dumps(report,indent=2))
        print(json.dumps(case),flush=True)
    report['status']='passed';args.output.write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
