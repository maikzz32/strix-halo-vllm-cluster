"""GPU-only first-mismatch recorder for an isolated HC diagnostic graph.

One state/snapshot pair belongs to one layer. Callers must retain all buffers,
serialize layer execution, and synchronize before reading snapshots on the CPU.
No model outputs are modified. This is diagnostic work, not a timing candidate.
"""
import triton
import triton.language as tl

@triton.jit
def _copy(src,dst, OFFSET:tl.constexpr,N:tl.constexpr,B:tl.constexpr):
    lane=tl.arange(0,B)
    for base in range(triton.cdiv(N,B)):
        i=base*B+lane
        tl.store(dst+OFFSET+i,tl.load(src+i,i<N,other=0),i<N)

@triton.jit
def record(x,xn,ref_gate,new_gate,ref_out,new_out,state,snapshot,B:tl.constexpr):
    lane=tl.arange(0,B)
    gd=tl.full((),0,tl.int32)
    od=tl.full((),0,tl.int32)
    for base in range(40):
        i=base*B+lane
        a=tl.load(ref_gate+i).to(tl.uint16,bitcast=True)
        b=tl.load(new_gate+i).to(tl.uint16,bitcast=True)
        gd+=tl.sum((a!=b).to(tl.int32),0)
    for base in range(10):
        i=base*B+lane
        a=tl.load(ref_out+i).to(tl.uint16,bitcast=True)
        b=tl.load(new_out+i).to(tl.uint16,bitcast=True)
        od+=tl.sum((a!=b).to(tl.int32),0)
    call=tl.atomic_add(state,1)+1
    tl.atomic_add(state+1,gd)
    tl.atomic_add(state+2,od)
    if gd+od>0:
        previous=tl.atomic_cas(state+3,0,call)
        if previous==0:
            _copy(x,snapshot,0,1280,B)
            _copy(xn,snapshot,1280,40960,B)
            _copy(ref_gate,snapshot,42240,40960,B)
            _copy(new_gate,snapshot,83200,40960,B)
            _copy(ref_out,snapshot,124160,10240,B)
            _copy(new_out,snapshot,134400,10240,B)

SNAPSHOT_ELEMENTS=144640

def inspect(x,xn,ref_gate,new_gate,ref_out,new_out,state,snapshot):
    import torch
    tensors=(x,xn,ref_gate,new_gate,ref_out,new_out)
    sizes=(1280,40960,40960,40960,10240,10240)
    assert all(t.is_cuda and t.is_contiguous() and t.dtype==torch.bfloat16
               and t.numel()==n and t.device==x.device for t,n in zip(tensors,sizes))
    assert state.device==snapshot.device==x.device and state.dtype==torch.int32
    assert state.shape==(4,) and state.is_contiguous()
    assert snapshot.dtype==torch.bfloat16 and snapshot.numel()==SNAPSHOT_ELEMENTS and snapshot.is_contiguous()
    record[(1,)](*tensors,state,snapshot,1024,num_warps=4)
