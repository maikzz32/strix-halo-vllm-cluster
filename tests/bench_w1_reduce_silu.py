"""Isolated W1 reduction/SiLU fusion with explicit intermediate BF16 rounding."""
import importlib,json,statistics
import torch,triton,triton.language as tl
import vllm._custom_ops
base=importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe')
@triton.jit
def fused(P,O,R:tl.constexpr,N:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);i=tl.program_id(1)*B+tl.arange(0,B);mask=i<N//2
    a=tl.full((B,),0,tl.float32);b=tl.full((B,),0,tl.float32)
    for s in tl.static_range(S):
        a+=tl.load(P+s*R*N+row*N+i,mask,0)
        b+=tl.load(P+s*R*N+row*N+i+N//2,mask,0)
    a=a.to(tl.bfloat16).to(tl.float32);b=b.to(tl.bfloat16).to(tl.float32)
    activated=(a/(1.0+tl.exp(-a))).to(tl.bfloat16).to(tl.float32)
    v=activated*b
    tl.store(O+row*(N//2)+i,v,mask)
def main():
    torch.manual_seed(9601);r,n,s=40,320,4
    p=torch.empty(s,r,n,device='cuda',dtype=torch.float32)
    tmp=torch.empty(r,n,device='cuda',dtype=torch.bfloat16)
    out=torch.empty(r,n//2,device='cuda',dtype=torch.bfloat16);candidate=torch.empty_like(out)
    def original():
        base._glm53_moe_int4_gemv_reduce[(r*triton.cdiv(n,128),)](p,tmp,p,r,n,s,False,128)
        torch.ops._C.silu_and_mul(out,tmp)
    def trial():fused[(r,triton.cdiv(n//2,128))](p,candidate,r,n,s,128,enable_fp_fusion=False)
    mismatches=0;max_abs=0.
    for bank in range(48):
        p.normal_().mul_((0.01,0.1,1.,10.)[bank%4]);original();trial()
        mismatches+=torch.count_nonzero(out!=candidate).item()
        max_abs=max(max_abs,(out.float()-candidate.float()).abs().max().item())
    result={'banks':48,'elements':48*out.numel(),'mismatches':mismatches,'max_abs':max_abs}
    graphs=[]
    for fn in (original,trial):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):fn()
        stream.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):fn()
        graphs.append(g)
    p.normal_();graphs[0].replay();graphs[1].replay();torch.cuda.synchronize()
    result['changed_graph_mismatches']=torch.count_nonzero(out!=candidate).item()
    samples=[[],[]]
    for rep in range(6):
        for k in ((0,1) if rep%2==0 else (1,0)):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(32):graphs[k].replay()
            end.record();end.synchronize();samples[k].append(start.elapsed_time(end)*1000/32)
    result.update(samples_us=samples,median_us=[statistics.median(x) for x in samples])
    print(json.dumps(result),flush=True)
    for g in graphs:g.reset()
if __name__=='__main__':main()
