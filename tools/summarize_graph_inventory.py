"""Validate and summarize direct HIP graph node inventories, without timing estimates."""
import argparse,hashlib,json
from collections import Counter,deque
from pathlib import Path
TYPES={0:'kernel',1:'memcpy',2:'memset',3:'host',4:'child_graph',5:'empty',6:'wait_event',7:'record_event',8:'semaphore_signal',9:'semaphore_wait',10:'alloc',11:'free',12:'copy_from_symbol',13:'copy_to_symbol',14:'batch_mem_op'}
def graph_summary(g):
    nodes=g['nodes'];n=g['node_count'];assert n==len(nodes)
    assert [x['index'] for x in nodes]==list(range(n))
    remaining=[];children=[[] for _ in nodes];depth=[1]*n
    for x in nodes:
        deps=x['dependencies'];assert len(deps)==len(set(deps))
        assert all(0<=d<n and d!=x['index'] for d in deps)
        remaining.append(len(deps))
        for d in deps:children[d].append(x['index'])
    roots=[i for i,d in enumerate(remaining) if not d];queue=deque(roots);seen=0
    while queue:
        i=queue.popleft();seen+=1
        for j in children[i]:
            depth[j]=max(depth[j],depth[i]+1);remaining[j]-=1
            if not remaining[j]:queue.append(j)
    assert seen==n,'Graph dependencies contain a cycle'
    return {'nodes':n,'types':dict(Counter(TYPES.get(x['type'],str(x['type'])) for x in nodes)),
        'root_count':len(roots),'edge_count':sum(len(x['dependencies']) for x in nodes),
        'longest_dependency_chain_nodes':max(depth,default=0),
        'copies':[x['copy'] for x in nodes if 'copy' in x],
        'child_graphs':[graph_summary(x['child']) for x in nodes if 'child' in x]}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records=[]
    for manifest in sorted(a.root.glob('*/manifest.json')):
        m=json.loads(manifest.read_text());rows=[]
        for name,sha in m['files'].items():
            raw=(manifest.parent/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
            rows.append({'name':name,'sha256':sha,**graph_summary(json.loads(raw))})
        records.append({'host':manifest.parent.name,'run_id':m['run_id'],'worker_pid':m['worker_pid'],'graphs':rows})
    assert len(records)==4
    a.output.write_text(json.dumps(records,indent=2)+'\n')
    for r in records:print(json.dumps({'host':r['host'],'graphs':[{'nodes':g['nodes'],'types':g['types'],'chain':g['longest_dependency_chain_nodes']} for g in r['graphs']]}))
if __name__=='__main__':main()
