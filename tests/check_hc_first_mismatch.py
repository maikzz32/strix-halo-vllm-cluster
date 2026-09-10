"""Exercise first-error retention and counters across HIP graph replays."""
import json,torch
from hc_first_mismatch import inspect,SNAPSHOT_ELEMENTS

torch.set_num_threads(1);torch.manual_seed(9573)
x=torch.randn(4,320,device='cuda',dtype=torch.bfloat16)
xn=torch.randn(4,10240,device='cuda',dtype=torch.bfloat16)
g=torch.randn(4,10240,device='cuda',dtype=torch.bfloat16);ng=g.clone()
y=torch.randn(4,2560,device='cuda',dtype=torch.bfloat16);ny=y.clone()
state=torch.zeros(4,device='cuda',dtype=torch.int32)
snapshot=torch.zeros(SNAPSHOT_ELEMENTS,device='cuda',dtype=torch.bfloat16)
def call():inspect(x,xn,g,ng,y,ny,state,snapshot)
call();assert state.tolist()==[1,0,0,0]
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    call();stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):call()
    state.zero_();snapshot.zero_()
    graph.replay();stream.synchronize();assert state.tolist()==[1,0,0,0]
    # Fault injection is confined to test tensors.
    ng.view(-1)[17]=g.view(-1)[17]+1
    ny.view(-1)[29]=y.view(-1)[29]+1
    expected=torch.cat([t.flatten() for t in (x,xn,g,ng,y,ny)])
    graph.replay();stream.synchronize()
    assert state.tolist()==[2,1,1,2],state.tolist()
    assert torch.equal(snapshot,expected)
    x.add_(2);xn.mul_(.5);ng.copy_(g);ny.copy_(y)
    ny.view(-1)[7]=y.view(-1)[7]+1
    graph.replay();stream.synchronize()
    assert state.tolist()==[3,1,2,2],state.tolist()
    assert torch.equal(snapshot,expected),'first mismatch snapshot overwritten'
    # Re-arm explicitly, capture a different mismatch.
    state.zero_();graph.replay();stream.synchronize()
    expected2=torch.cat([t.flatten() for t in (x,xn,g,ng,y,ny)])
    assert state.tolist()==[1,0,1,1]
    assert torch.equal(snapshot,expected2)
print('RESULT='+json.dumps(dict(status='passed',graph_replay=True,first_error_retained=True,
    reset_rearms=True,counters_exact=True,snapshot_elements=SNAPSHOT_ELEMENTS)),flush=True)
