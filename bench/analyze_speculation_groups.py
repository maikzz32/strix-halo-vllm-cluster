"""Check SSE group accounting and report descriptive acceptance transitions."""
import argparse, hashlib, json
from pathlib import Path

def analyze(path, depth=3):
    d=json.loads(path.read_text());assert all(d['checks'].values())
    g=[len(e['token_ids']) for e in d['events'] if e['token_ids']]
    def metric(name):return sum(v for k,v in d['metric_deltas'].items() if k.startswith('vllm:'+name+'{'))
    actual=[sum(v for k,v in d['metric_deltas'].items() if 'spec_decode_num_accepted_tokens_per_pos_total{' in k and f'position="{i}"' in k) for i in range(depth)]
    prefix=[sum(n>=i+2 for n in g[1:-1]) for i in range(depth)]
    terminal=[a for a in range(depth+1) if [prefix[i]+int(a>i) for i in range(depth)]==actual and a+1>=g[-1]]
    checks={'singleton_first':g[0]==1,'bounded_groups':all(1<=n<=depth+1 for n in g),'groups_match_drafts':len(g)-1==metric('spec_decode_num_drafts_total'),'positions_match_with_terminal_truncation':len(terminal)==1}
    assert all(checks.values()),(str(path),checks)
    interior=g[1:-1];rows=[]
    for streak in ((1,2,3,4) if depth == 3 else ()):
        nxt=[interior[i] for i in range(streak,len(interior)) if all(n<=2 for n in interior[i-streak:i])]
        rows.append({'prior_weak_groups':streak,'observations':len(nxt),'next_third_accepted':nxt.count(4),'next_total_output_tokens':sum(nxt),'next_third_acceptance_probability':nxt.count(4)/len(nxt) if nxt else None,'idealized_required_step_time_reduction_fraction':nxt.count(4)/sum(nxt) if nxt else None})
    return {'name':path.stem,'draft_depth':depth,'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'usage':d['usage'],'accounting_checks':checks,'terminal_accepted_count':terminal[0],'groups':g,'conditional_results':rows}

def main():
    a=argparse.ArgumentParser();a.add_argument('inputs',type=Path,nargs='+');a.add_argument('--output',type=Path,required=True);a.add_argument('--depth',type=int,choices=(2,3),default=3);args=a.parse_args()
    report={'cases':[analyze(p,args.depth) for p in args.inputs],'note':'Descriptive K3 trajectories only. Drop initial and final group from transitions. Required saving = third-token successes / total next-group tokens, under an idealized unchanged-state comparison E2=E3-p3. Switching depth changes future boundaries and trajectories; this is not a measured K2 or adaptive speedup. Aggregate accounting is consistent with iteration grouping, not an independent engine trace.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    for case in report['cases']:print(case['name'],case['accounting_checks'],case['conditional_results'][1:2])
if __name__=='__main__':main()
