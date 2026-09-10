"""Isolated HC-down prefetch parity and rotating-pool timing."""
import argparse,ctypes,hashlib,json,os,statistics
from pathlib import Path
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args()
    assert hashlib.sha256(a.library.read_bytes()).hexdigest()==a.sha256
    import torch
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
    torch.set_num_threads(1);torch.manual_seed(9603)
    lib=ctypes.CDLL(str(a.library))
    lib.hc_down_prefetch.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    lib.hc_down_prefetch.restype=ctypes.c_int
    report={'status':'running','cases':[],'note':'24 rotating synthetic BF16 matrices; methods: 0 installed, 1 isolated original, 2 prefetch. Same U4, wave32, YTILE1, grid20.'}
    for n,k in ((336,10240),(320,10240)):
        x=torch.randn((4,k),device='cuda',dtype=torch.bfloat16)
        pool=[torch.randn((n,k),device='cuda',dtype=torch.bfloat16)*.02 for _ in range(24)]
        if n==336:
            for w in pool:w[-12:]=0
        methods=[0,1,2]
        def call(grid,w):
            if grid==0:return rocm_unquantized_gemm_impl(x,w)
            y=torch.empty((4,n),device='cuda',dtype=torch.bfloat16)
            rc=lib.hc_down_prefetch(grid-1,w.data_ptr(),x.data_ptr(),y.data_ptr(),n,20,torch.cuda.current_stream().cuda_stream)
            assert rc==0,('launch',grid,rc)
            return y
        print('PHASE reference',flush=True)
        expected=[call(0,w) for w in pool]
        for grid in methods[1:]:
            print("PHASE eager",grid,flush=True)
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
if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise



