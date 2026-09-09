"""Count recorded GPU copies by graph-launch correlation, without inferring HIP node counts."""
import argparse
from collections import Counter,defaultdict
import gzip,hashlib,json
from pathlib import Path

def analyze(path):
    with gzip.open(path,'rt') as f:events=json.load(f)['traceEvents']
    es=[e for e in events if e.get('ph')=='X']
    launches={e['args']['correlation']:e for e in es if e.get('name')=='hipGraphLaunch'}
    counts=Counter(e.get('cat','') for e in es)
    memory=[e for e in es if e.get('cat') in ('gpu_memcpy','gpu_memset')]
    bygraph=defaultdict(list)
    for e in es:
        cid=e.get('args',{}).get('correlation')
        if cid in launches and e.get('cat') in ('kernel','gpu_kernel','gpu_memcpy','gpu_memset'):
            bygraph[cid].append(e)
    graphs=[]
    for cid,launch in launches.items():
        work=bygraph[cid]
        graphs.append({'launch_ts':launch['ts'],'categories':dict(Counter(e['cat'] for e in work)),
            'memory_names':dict(Counter(e['name'] for e in work if e['cat'] in ('gpu_memcpy','gpu_memset')))})
    return {'trace':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'all_categories':dict(counts),'graph_launches':len(launches),'graphs':graphs,
        'gpu_memory_events':len(memory),'gpu_memory_names':dict(Counter(e['name'] for e in memory)),
        'memory_correlated_to_graph':sum(e.get('args',{}).get('correlation') in launches for e in memory),
        'note':'Profiler event counts, not HIP graph node inventory. Missing events cannot establish absent nodes.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[analyze(next(d.glob('*.gz'))) for d in a.directories]
    a.output.write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:print(json.dumps({k:v for k,v in r.items() if k!='graphs'}))
