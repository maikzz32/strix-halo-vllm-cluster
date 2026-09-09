"""Inventory graph kernels without assuming model-specific names or counts."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

def inspect(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
        data=json.load(f)
    events=[e for e in data['traceEvents'] if e.get('ph')=='X']
    kernels=[e for e in events if e.get('cat') in ('kernel','gpu_kernel')]
    if not kernels:raise ValueError('No GPU kernels in trace')
    scopes=[e for e in events if e.get('cat')=='user_annotation']
    bycid=defaultdict(list)
    for e in kernels:
        if 'correlation' in e.get('args',{}):bycid[e['args']['correlation']].append(e)
    launches=[e for e in events if e.get('name') in ('hipGraphLaunch','cudaGraphLaunch')]
    ids=[e['args']['correlation'] for e in launches]
    if len(ids)!=len(set(ids)):raise ValueError('Ambiguous launch correlation IDs')
    graphs=[]
    for launch,cid in zip(launches,ids):
        ks=sorted(bycid.get(cid,[]),key=lambda e:e['ts'])
        if not ks:continue
        names=Counter(); times=Counter()
        for e in ks:
            names[e['name']]+=1;times[e['name']]+=e['dur']
        parents=[s['name'] for s in scopes if s['pid']==launch['pid'] and s['tid']==launch['tid']
                 and s['ts']<=launch['ts']<s['ts']+s['dur']]
        graphs.append({'correlation':cid,'enclosing_scopes':parents,'kernel_count':len(ks),
            'span_ms':(max(e['ts']+e['dur'] for e in ks)-ks[0]['ts'])/1000,
            'summed_kernel_ms':sum(e['dur'] for e in ks)/1000,
            'kernels':[{'name':name,'count':names[name],'summed_ms':duration/1000}
                       for name,duration in times.most_common()]})
    return {'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'total_gpu_kernels':len(kernels),'graph_launches':len(launches),
        'graphs_with_kernels':len(graphs),'graphs':graphs,
        'note':'Correlation-based inventory, not semantic attribution. Summed kernel durations may overlap and profiling perturbs timings. Inspect completeness and source before assigning roles.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=inspect(a.trace)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='graphs'}))
