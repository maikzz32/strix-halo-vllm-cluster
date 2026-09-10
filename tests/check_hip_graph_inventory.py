"""Run under the inventory LD_PRELOAD in an idle, isolated GPU process."""
import json,os
from pathlib import Path
import torch

out=Path(os.environ['STRIX_GRAPH_INVENTORY_DIR'])
assert out.is_dir()
prefix=f'graph-{os.getpid()}-'
assert not list(out.glob(prefix+'*.json'))
x=torch.arange(32,device='cuda',dtype=torch.float32)
y=torch.empty_like(x)
s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    y.copy_(x);y.add_(1)
s.synchronize()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g,stream=s):
    y.copy_(x);y.add_(1)
g.replay();torch.cuda.synchronize()
assert torch.equal(y,x+1)
records=[json.loads(p.read_text()) for p in out.glob(prefix+'*.json')]
assert records,'Interposition did not observe graph instantiation'
for r in records:
    assert r['node_count']==len(r['nodes']) and r['node_count']>=2
    assert any(n['type']==0 for n in r['nodes']),r
    assert any(n['dependencies'] for n in r['nodes']),r
    assert all(all(0<=d<r['node_count'] for d in n['dependencies']) for n in r['nodes'])
print(json.dumps({'passed':True,'records':len(records),'node_counts':[r['node_count'] for r in records],
    'note':'Tensor copy may be implemented as a kernel. This smoke does not require a memcpy node.'}))
g.reset()
