"""Isolated HC launch-grid comparison with unchanged native arithmetic."""
import argparse,ctypes,hashlib,json,os,statistics
from pathlib import Path
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert hashlib.sha256(a.library.read_bytes()).hexdigest()=='8ab4de1bd05abaf4f0188a1e82aa6a1f69e53acb8ed2cc63444e5f06b6b854bc'
    import torch
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
    torch.set_num_threads(1);torch.manual_seed(9503)
    import sys
    from types import SimpleNamespace
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'patches'))
    os.environ['STRIX_HC_NATIVE_GRID']='1'
    os.environ['STRIX_HC_NATIVE_LIBRARY']=str(a.library)
    from hc_native_grid import install
    install()
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    method=UnquantizedLinearMethod()
    report={'status':'running','cases':[],'note':'Synthetic BF16 weights, 24-matrix pools; Method1 exercises the gated UnquantizedLinearMethod adapter and registered custom op.'}
    for n,k in ((336,10240),(320,10240),(10240,320)):
        x=torch.randn((4,k),device='cuda',dtype=torch.bfloat16)
        pool=[torch.randn((n,k),device='cuda',dtype=torch.bfloat16)*.02 for _ in range(24)]
        if n==336:
            for w in pool:w[-12:]=0
        methods=[0,1]
        def call(grid,w):
            if grid==0:return rocm_unquantized_gemm_impl(x,w)
            name='input_mix_weight_up' if k==320 else ('input_mix_weight_down' if n==320 else 'input_mix_weight_down_block_inject')
            layer=SimpleNamespace(weight=w,prefix='model.layers.0.attn_hyper_connection.'+name)
            return method.apply(layer,x)
        expected=[call(0,w) for w in pool]
        for grid in methods[1:]:
            assert all(torch.equal(call(grid,w),e) for w,e in zip(pool,expected)),('eager parity',n,k,grid)
        graphs={};outputs={};stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for grid in methods:
                for w in pool:call(grid,w)
                stream.synchronize();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g,stream=stream):out=[call(grid,w) for w in pool]
                graphs[grid]=g;outputs[grid]=out
            x.mul_(.75);changed=[call(0,w) for w in pool]
            for grid,g in graphs.items():
                g.replay();stream.synchronize()
                assert all(torch.equal(y,e) for y,e in zip(outputs[grid],changed)),('graph parity',n,k,grid)
            samples={grid:[] for grid in methods}
            for r in range(6):
                for grid in (methods if r%2==0 else list(reversed(methods))):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(20):graphs[grid].replay()
                    end.record();end.synchronize();samples[grid].append(start.elapsed_time(end)*1000/480)
        torch.cuda.current_stream().wait_stream(stream)
        row={'shape':[4,n,k],'bitwise_eager_and_changed_graph':True,'samples_us':samples,'median_us':{g:statistics.median(v) for g,v in samples.items()}}
        report['cases'].append(row);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
        del graphs,outputs,expected,changed,pool,x
    report['status']='passed';a.output.write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
