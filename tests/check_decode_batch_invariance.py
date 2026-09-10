"""Characterize row-prefix equality for two decode batch sizes, without modifying kernels."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1'
import torch
from vllm.model_executor.layers import utils
from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as dense

def stats(a,b):
    assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    delta=(a.float()-b.float()).abs()
    return {'elements':a.numel(),'unequal_elements':int((a!=b).sum()),'max_abs_difference':float(delta.max())}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--short-rows',type=int,default=3)
    parser.add_argument('--long-rows',type=int,default=4)
    args=parser.parse_args()
    assert 1 <= args.short_rows < args.long_rows <= 8
    short_rows,long_rows=args.short_rows,args.long_rows
    torch.set_num_threads(1);torch.manual_seed(9512)
    rows=[]
    for kind,n,k in [('bf16',336,10240),('bf16',320,10240),('bf16',10240,320),('bf16',1,2560),('int4',512,2560),('int4',320,2560),('int4',2560,160)]:
        for bank in range(4):
            x=torch.randn((long_rows,k),device='cuda',dtype=torch.bfloat16)
            if kind=='bf16':
                w=torch.randn((n,k),device='cuda',dtype=torch.bfloat16)*.02
                def call(v):return utils.rocm_unquantized_gemm_impl(v,w)
            else:
                q=torch.randint(-128,128,(n,k//2),device='cuda',dtype=torch.int8)
                s=torch.rand((n,k//32),device='cuda',dtype=torch.bfloat16)*.02
                z=torch.randint(0,16,(n,k//32),device='cuda').to(torch.bfloat16)
                def call(v):return dense._rdna_hybrid_w4a16_apply_impl(v,q,s,z,None,20,32)
            full=call(x);short=call(x[:short_rows]);first=stats(full[:short_rows],short)
            # Verify repeatability at each fixed shape independently.
            assert torch.equal(full,call(x)) and torch.equal(short,call(x[:short_rows]))
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                call(x);call(x[:short_rows]);stream.synchronize()
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g,stream=stream):a=call(x);b=call(x[:short_rows])
                x.mul_(.75);g.replay();stream.synchronize()
                changed=stats(a[:short_rows],b)
                assert torch.equal(a,call(x)) and torch.equal(b,call(x[:short_rows]))
            torch.cuda.current_stream().wait_stream(stream)
            rows.append({'kind':kind,'N':n,'K':k,'bank':bank,'eager_prefix':first,'changed_graph_prefix':changed,'fixed_shape_repeatability':True})
    paths=[Path(utils.__file__),Path(dense.__file__)]
    print('RESULT='+json.dumps({'short_rows':short_rows,'long_rows':long_rows,'torch_version':torch.__version__,'hip_version':torch.version.hip,'device':torch.cuda.get_device_name(),'cases':rows,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'note':'Synthetic stateless operators only. Cross-shape mismatches are observations, not automatically bugs or proof of full-model divergence cause. Serving source unchanged.'}),flush=True)
if __name__=='__main__':main()
