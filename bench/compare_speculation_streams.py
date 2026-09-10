"""Compare observed SSE step intervals for identical deterministic token streams."""
import argparse, hashlib, json, statistics
from pathlib import Path
from analyze_speculation_groups import analyze

def load(path):
    data=json.loads(path.read_text());assert all(data['checks'].values()),path
    def total(name):return sum(v for k,v in data['metric_deltas'].items() if k.startswith('vllm:'+name+'{'))
    drafts=total('spec_decode_num_drafts_total');assert drafts>0
    ratio=total('spec_decode_num_draft_tokens_total')/drafts
    assert ratio in (2,3),ratio
    accounting=analyze(path,int(ratio))
    events=[e for e in data['events'] if e['token_ids']]
    ids=[token for e in events for token in e['token_ids']]
    # Exclude the initial token and final potentially truncated step.
    intervals=[events[i]['time_s']-events[i-1]['time_s'] for i in range(1,len(events)-1)]
    assert intervals and all(t>0 for t in intervals)
    return data,ids,{'draft_depth':int(ratio),'accounting_checks':accounting['accounting_checks'],'groups':len(events),'interior_intervals':len(intervals),'mean_interval_ms':statistics.mean(intervals)*1000,'median_interval_ms':statistics.median(intervals)*1000,'max_interval_ms':max(intervals)*1000,'intervals_over_twice_median':sum(t>2*statistics.median(intervals) for t in intervals),'p95_interval_ms':sorted(intervals)[int(.95*(len(intervals)-1))]*1000,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    a=argparse.ArgumentParser();a.add_argument('before',type=Path);a.add_argument('after',type=Path);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    b,bids,bstat=load(args.before);c,cids,cstat=load(args.after)
    checks={'same_request':b['request']==c['request'],'identical_token_ids':bids==cids,'same_usage':b['usage']==c['usage']}
    result={'checks':checks,'before':bstat,'after':cstat,'mean_interval_reduction_fraction':1-cstat['mean_interval_ms']/bstat['mean_interval_ms'],'median_interval_reduction_fraction':1-cstat['median_interval_ms']/bstat['median_interval_ms'],'note':'Client-observed SSE intervals, not isolated GPU timings. Requires separate token-group/engine-counter accounting. K changes verification tensor shapes and trajectory boundaries; interval savings are not an adaptive-policy speedup.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert all(checks.values()),checks
if __name__=='__main__':main()
