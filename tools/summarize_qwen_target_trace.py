#!/usr/bin/env python3
"""CPU-only attribution of the captured Qwen TP4 M4 verification graphs.

Names alone never distinguish routed W1/W2 or HC linears. This report requires
the source-confirmed complete adjacent sequences and their exact per-graph
counts; a mismatch raises rather than silently assigning a role.
"""
import argparse
import collections
import gzip
import json
from pathlib import Path
import statistics

TARGET_SCOPE = 'execute_context_0(0)_generation_1(4)'
PARTIAL = '_glm53_moe_int4_gemv_partial.kd'
REDUCE = '_glm53_moe_int4_gemv_reduce.kd'


def initial_category(name):
    s = name.lower()
    if 'nccl' in s or 'rccl' in s:
        return 'rccl_includes_wait'
    if 'wvsplitk' in s and 'int4' in s:
        return 'dense_router_shared_int4'
    if 'wvsplitk' in s:
        return 'other_bf16_linear'
    if '_hc_' in s:
        return 'hc_elementwise'
    if 'qsa' in s:
        return 'qsa_attention_and_indexing'
    if 'topkgating' in s or 'moe_align' in s or 'count_and_sort_expert' in s:
        return 'routed_selection_alignment'
    if 'moe_sum' in s:
        return 'routed_expert_sum'
    if any(k in s for k in ('gated_delta', 'delta_rule', 'causal_conv')):
        return 'gdn_recurrent_conv'
    return 'other_kernels'


def assign_roles(kernels):
    names = [e['name'] for e in kernels]
    roles = [initial_category(n) for n in names]
    partials = [i for i,n in enumerate(names) if n == PARTIAL]
    assert len(partials) == 96, f'Expected 48 W1/W2 pairs, got {len(partials)} partials'
    for a,b in zip(partials[::2], partials[1::2]):
        assert b == a+3 and names[a+1] == REDUCE and names[b+1] == REDUCE
        assert 'act_and_mul_kernel' in names[a+2] and 'silu' in names[a+2]
        assert 'moe_sum' in names[b+2]
        roles[a:a+5] = ['routed_w1_partial', 'routed_w1_reduce',
                        'routed_silu', 'routed_w2_partial', 'routed_w2_reduce']
    assert names.count(REDUCE) == 96
    hc = [i for i,n in enumerate(names) if n == '_hc_silu_kernel.kd']
    assert len(hc) == 97, f'Expected 97 HC down/up pairs, got {len(hc)}'
    for i in hc:
        assert all('wvSplitK_hf_' in names[j] and 'int4' not in names[j] for j in (i-1,i+1))
        assert names[i+2] == '_hc_gate_mix_kernel.kd'
        roles[i-1] = 'hc_down_bf16'
        roles[i+1] = 'hc_up_bf16'
    assert sum('nccl' in n.lower() for n in names) == 98
    assert sum('_qsa_mqa_paged_kernel' in n for n in names) == 12
    return roles


def partition(kernels, roles):
    """Disjoint wall interval partition; multi-category overlap is explicit."""
    changes = collections.defaultdict(collections.Counter)
    for e,role in zip(kernels,roles):
        a = round(e['ts']*1000); b = a + round(e['dur']*1000)
        changes[a][role] += 1
        changes[b][role] -= 1
    active = collections.Counter(); budgets = collections.Counter(); previous = min(changes)
    for stamp,delta in sorted(changes.items()):
        labels = [role for role,count in active.items() if count]
        key = labels[0] if len(labels)==1 else ('multiple_categories_overlap' if labels else 'no_traced_kernel')
        budgets[key] += stamp-previous
        active.update(delta); previous=stamp
    return {k:v/1e6 for k,v in budgets.items()}


def kernel_gaps(kernels):
    previous=kernels[0]
    gaps=[]
    for e in kernels[1:]:
        gap=e['ts']-previous['ts']-previous['dur']
        if gap>0:
            gaps.append({'us':gap,'same_stream':e['tid']==previous['tid'],
                         'before':previous['name'],'after':e['name']})
        if e['ts']+e['dur']>previous['ts']+previous['dur']:
            previous=e
    return gaps


def summarize_gaps(gaps):
    values=sorted(g['us'] for g in gaps)
    total=sum(values)
    thresholds=[(0,1),(1,5),(5,10),(10,50),(50,100),(100,float('inf'))]
    histogram=[]
    for low,high in thresholds:
        ds=[v for v in values if low<v<=high]
        histogram.append({'range_us':f'({low},{high}]','count':len(ds),
                          'count_pct':100*len(ds)/len(values),'sum_ms':sum(ds)/1000,
                          'gap_time_pct':100*sum(ds)/total})
    return {'count':len(values),'sum_ms':total/1000,'median_us':statistics.median(values),
            'p90_us':values[int(.9*(len(values)-1))],'max_us':values[-1],
            'histogram':histogram,
            'same_stream_count':sum(g['same_stream'] for g in gaps),
            'same_stream_sum_ms':sum(g['us'] for g in gaps if g['same_stream'])/1000,
            'largest_gaps':sorted(gaps,key=lambda g:-g['us'])[:10]}


def inspect(path):
    data = json.load(gzip.open(path,'rt') if path.suffix=='.gz' else path.open())
    events = [e for e in data['traceEvents'] if e.get('ph')=='X']
    kernels = [e for e in events if e.get('cat') in ('kernel','gpu_kernel')]
    launches = [e for e in events if e.get('name')=='hipGraphLaunch']
    scopes = [e for e in events if e.get('name')==TARGET_SCOPE and e.get('cat')=='user_annotation']
    bycid = collections.defaultdict(list)
    for e in kernels:
        bycid[e.get('args',{}).get('correlation')].append(e)
    rows=[]
    gaps=[]
    for launch in launches:
        parents=[s for s in scopes if s['pid']==launch['pid'] and s['tid']==launch['tid']
                 and s['ts']<=launch['ts']<s['ts']+s['dur']]
        if not parents:
            continue
        assert len(parents)==1, 'Ambiguous target CPU scope'
        cid=launch['args']['correlation']
        ks=sorted(bycid[cid],key=lambda e:e['ts'])
        assert len(ks)==2694 and all(e['dur']>0 for e in ks)
        roles=assign_roles(ks)
        gaps.extend(kernel_gaps(ks))
        role_us=collections.Counter()
        for e,role in zip(ks,roles):
            role_us[role]+=e['dur']
        rows.append({'correlation_id':cid,'kernel_count':len(ks),
                     'cpu_scope_name':parents[0]['name'],
                     'gpu_span_ms':(max(e['ts']+e['dur'] for e in ks)-min(e['ts'] for e in ks))/1000,
                     'gpu_start_us':min(e['ts'] for e in ks),
                     'role_summed_ms':{k:v/1000 for k,v in role_us.items()},
                     'role_counts':dict(collections.Counter(roles)),
                     'disjoint_interval_ms':partition(ks,roles)})
    assert len(rows)==16, f'Expected 16 complete target graphs, got {len(rows)}'
    allcids={g['args']['correlation'] for g in launches}
    nongraph=[e for e in kernels if e.get('args',{}).get('correlation') not in allcids]
    graph_correlations=collections.Counter(g['args']['correlation'] for g in launches)
    assert all(v==1 for v in graph_correlations.values())
    apis={e.get('args',{}).get('correlation'):e for e in events if e.get('cat')=='cuda_runtime'}
    steps=[e for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep')
           and e['pid']==launches[0]['pid']]
    step_kernels=collections.defaultdict(list)
    target_cids={r['correlation_id'] for r in rows}
    for e in kernels:
        api=apis.get(e.get('args',{}).get('correlation'))
        if api is None:
            continue
        parents=[s for s in steps if s['pid']==api['pid'] and s['tid']==api['tid']
                 and s['ts']<=api['ts']<s['ts']+s['dur']]
        if len(parents)==1:
            step_kernels[parents[0]['name']].append(e)
    iterations=[]
    for step,ks in step_kernels.items():
        phase=collections.defaultdict(list)
        for e in ks:
            cid=e.get('args',{}).get('correlation')
            label='target_graph' if cid in target_cids else ('draft_step0_m4_model_m1_sample' if cid in allcids else 'eager_metadata_sampling_includes_draft_steps1_2_m1')
            phase[label].append(e)
        iterations.append({'step':step,'kernel_count':len(ks),
                           'gpu_span_ms':(max(e['ts']+e['dur'] for e in ks)-min(e['ts'] for e in ks))/1000,
                           'kernel_sum_ms':sum(e['dur'] for e in ks)/1000,
                           'phase_kernel_counts':{p:len(es) for p,es in phase.items()},
                           'phase_kernel_sum_ms':{p:sum(e['dur'] for e in es)/1000 for p,es in phase.items()}})
    assert len(iterations)==16 and all(r['kernel_count']==3151 for r in iterations)
    return {'source':str(path),'kernel_count':len(kernels),'graph_launch_count':len(launches),
            'target_graph_count':len(rows),'non_graph_or_boundary_kernel_count':len(nongraph),
            'positive_kernel_durations':all(e['dur']>0 for e in kernels),
            'target_mean_gpu_span_ms':statistics.mean(r['gpu_span_ms'] for r in rows),
            'target_mean_role_ms':{k:statistics.mean(r['role_summed_ms'][k] for r in rows) for k in rows[0]['role_summed_ms']},
            'target_mean_disjoint_interval_ms':{k:statistics.mean(r['disjoint_interval_ms'].get(k,0) for r in rows) for k in set().union(*(r['disjoint_interval_ms'] for r in rows))},
            'target_role_counts':rows[0]['role_counts'],'target_graphs':rows,
            'target_kernel_gaps_all_16_graphs':summarize_gaps(gaps),
            'full_iterations':iterations,
            'full_iteration_mean_gpu_span_ms':statistics.mean(r['gpu_span_ms'] for r in iterations),
            'full_iteration_mean_kernel_sum_ms':statistics.mean(r['kernel_sum_ms'] for r in iterations),
            'full_iteration_mean_phase_kernel_sum_ms':{p:statistics.mean(r['phase_kernel_sum_ms'][p] for r in iterations) for p in iterations[0]['phase_kernel_sum_ms']},
            'outside_correlated_cpu_steps_kernel_count':len(kernels)-sum(r['kernel_count'] for r in iterations),
            'notes':['Target identity comes from CPU launch containment in the explicit M4 target scope, then same-file correlation ID to asynchronous GPU kernels.',
                     'The 112-kernel graph is draft step0: M4 model forward, last-token gather, then M1 LM head and sampling. Active V2 source hashes match the audited snapshot.',
                     'Draft steps1 and2 run model M1 eagerly; the eager/metadata/sampling group also includes other work and is not a pure draft budget.',
                     'W1/W2 assignments require all 48 complete adjacent partial/reduce/SiLU/partial/reduce/sum sequences per target graph.',
                     'HC assignments require all 97 adjacent BF16 down/HC-SiLU/BF16 up/HC-gate sequences per target graph.',
                     'Raw timestamps cannot compare ranks without host clock synchronization; rank comparisons here use durations.',
                     'Profiler queue fallback may alter timings; validate any optimization with unprofiled workload A/B.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('traces',nargs='+',type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    paths=[f for p in args.traces for f in (sorted(p.glob('*.pt.trace.json.gz')) if p.is_dir() else [p])]
    reports=[inspect(path) for path in paths]
    args.output.write_text(json.dumps(reports,indent=2))
    for r in reports:
        print(json.dumps({k:v for k,v in r.items() if k not in ('target_graphs','full_iterations','target_kernel_gaps_all_16_graphs','notes')},indent=2))


if __name__=='__main__':
    main()
