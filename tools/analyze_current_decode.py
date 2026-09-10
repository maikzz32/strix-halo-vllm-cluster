"""Source-checked attribution for the September 9 W1/W2/UMA decode trace."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
from summarize_qwen_target_trace import initial_category, partition

def roles_for(ks):
    names=[k['name'] for k in ks];roles=[initial_category(n) for n in names]
    w1=[i for i,n in enumerate(names) if n=='w1_expert_kfast.kd']
    assert len(w1)==48 and names.count('w2_serial.kd')==48
    assert names.count('_glm53_moe_int4_gemv_reduce.kd')==48
    for i in w1:
        assert names[i+1]=='_glm53_moe_int4_gemv_reduce.kd'
        assert 'act_and_mul_kernel' in names[i+2] and 'silu' in names[i+2]
        assert names[i+3]=='w2_serial.kd' and 'moe_sum' in names[i+4]
        roles[i:i+5]=['routed_w1','routed_w1_reduce','routed_silu','routed_w2','routed_sum']
    hc=[i for i,n in enumerate(names) if n=='_hc_silu_kernel.kd']
    assert len(hc)==97
    for i in hc:
        assert all('wvSplitK_hf_' in names[j] and 'int4' not in names[j] for j in (i-1,i+1))
        assert names[i+2]=='_hc_gate_mix_kernel.kd'
        roles[i-1]='hc_down_bf16';roles[i+1]='hc_up_bf16'
    exchanges=[i for i,n in enumerate(names) if '(anonymous namespace)::exchange(' in n]
    assert len(exchanges)==98
    for i in exchanges:roles[i]='uma_exchange_includes_wait'
    return roles

def analyze(path):
    with gzip.open(path,'rt') as f:events=json.load(f)['traceEvents']
    es=[e for e in events if e.get('ph')=='X']
    scopes=[e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='execute_context_0(0)_generation_1(4)']
    bycid=defaultdict(list)
    for e in es:
        if e.get('cat') in ('kernel','gpu_kernel'):bycid[e.get('args',{}).get('correlation')].append(e)
    rows=[]
    for launch in es:
        if launch.get('name')!='hipGraphLaunch':continue
        parents=[s for s in scopes if s['pid']==launch['pid'] and s['tid']==launch['tid'] and s['ts']<=launch['ts']<s['ts']+s['dur']]
        if not parents:continue
        assert len(parents)==1
        ks=sorted(bycid[launch['args']['correlation']],key=lambda e:e['ts'])
        assert len(ks)==2646 and all(e['dur']>0 for e in ks)
        roles=roles_for(ks);dur=Counter()
        for k,r in zip(ks,roles):dur[r]+=k['dur']/1000
        rows.append({'span_ms':(max(k['ts']+k['dur'] for k in ks)-ks[0]['ts'])/1000,
            'counts':dict(Counter(roles)),'summed_ms':dict(dur),'disjoint_ms':partition(ks,roles)})
    assert len(rows)==16
    assert all(r['counts']==rows[0]['counts'] for r in rows)
    return {'trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'trace_name':path.name,
        'graphs':rows,'mean_span_ms':statistics.mean(r['span_ms'] for r in rows),
        'mean_disjoint_ms':{k:statistics.mean(r['disjoint_ms'].get(k,0) for r in rows) for k in set().union(*(r['disjoint_ms'] for r in rows))},
        'note':'Durations of 16 profiled M4 target graphs, not unprofiled response TPS. UMA time includes waits. No cross-host timestamp alignment assumed.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directories',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=[analyze(next(d.glob('*.gz'))) for d in a.directories]
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in result:print(json.dumps({k:v for k,v in r.items() if k!='graphs'}))
