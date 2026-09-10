"""Attribute kernels by same-thread CPU scopes and runtime correlation IDs."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('root',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
records=[]
for path in sorted(a.root.glob('*/*.trace.json.gz')):
    data=path.read_bytes()
    events=json.loads(gzip.decompress(data))['traceEvents']
    scopes=[e for e in events if e.get('cat')=='user_annotation' and
            e.get('name','').startswith('execute_context_1(')]
    assert sorted(e['name'] for e in scopes)==[
        'execute_context_1(504)_generation_0(0)', 'execute_context_1(8)_generation_0(0)']
    assigned=set()
    phases=[]
    for scope in scopes:
        apis=[e for e in events if e.get('cat')=='cuda_runtime' and
              e.get('pid')==scope['pid'] and e.get('tid')==scope['tid'] and
              scope['ts']<=e['ts']<scope['ts']+scope['dur']]
        cids={e.get('args',{}).get('correlation') for e in apis}
        cids.discard(None)
        assert not (assigned & cids)
        assigned.update(cids)
        kernels=[e for e in events if e.get('cat')=='kernel' and
                 e.get('args',{}).get('correlation') in cids]
        assert len(kernels)==(3022 if '(504)' in scope['name'] else 3103)
        sums=collections.Counter()
        counts=collections.Counter()
        byname=collections.Counter()
        for e in kernels:
            name=e['name']
            category=('rccl_including_wait' if 'ncclDevKernel' in name else
                      'routed_moe_gemm' if 'fused_moe_kernel_gptq_awq' in name else
                      'qsa' if '_qsa_' in name else
                      'dense_int4' if '_triton_w4a16_skinny_fmt_kernel' in name else
                      'other')
            assert e['dur']>0
            sums[category]+=e['dur']/1000
            counts[category]+=1
            byname[name]+=e['dur']/1000
        assert counts['rccl_including_wait']==98
        intervals=sorted((e['ts'],e['ts']+e['dur']) for e in kernels)
        start=end=intervals[0][0]
        union=0
        for left,right in intervals:
            if left>end:
                union+=end-start
                start,end=left,right
            else:end=max(end,right)
        union+=end-start
        span=max(r for l,r in intervals)-intervals[0][0]
        phases.append(dict(scope=scope['name'],kernel_count=len(kernels),
            gpu_span_ms=span/1000,kernel_union_ms=union/1000,
            uncovered_ms=(span-union)/1000,category_summed_ms=dict(sums),
            category_counts=dict(counts),top_kernel_summed_ms=byname.most_common(15)))
    records.append(dict(host=path.parent.name,trace_sha256=hashlib.sha256(data).hexdigest(),phases=phases))
assert len(records)==4
a.output.write_text(json.dumps(dict(ranks=records,
    note='Profiled CPU-scope-correlated GPU intervals; sums include waits and are not end-to-end serving timings.'),indent=2))
for r in records:
    print(r['host'],[(s['scope'],s['category_summed_ms']['rccl_including_wait']) for s in r['phases']])
