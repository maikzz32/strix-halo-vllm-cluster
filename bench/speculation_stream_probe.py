"""Capture SSE token-id groups and speculative counters without worker changes."""
import argparse, hashlib, json, time, urllib.request
from pathlib import Path

def metrics(url):
    with urllib.request.urlopen(url+'/metrics',timeout=10) as r: text=r.read().decode()
    return {line.split()[0]:float(line.split()[-1]) for line in text.splitlines()
            if line.startswith(('vllm:spec_decode','vllm:num_requests_running','vllm:num_requests_waiting','vllm:request_success_total'))}

def main():
    a=argparse.ArgumentParser();a.add_argument('--url',default='http://192.168.1.15:8000');a.add_argument('--output',type=Path,required=True);a.add_argument('--tokens',type=int,default=512);a.add_argument('--prompt-file',type=Path);args=a.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    before=metrics(args.url)
    assert not any(v for k,v in before.items() if 'num_requests_' in k),'service busy'
    with urllib.request.urlopen(args.url+'/v1/models',timeout=10) as r:model=json.load(r)['data'][0]['id']
    body={'model':model,'messages':[{'role':'user','content':args.prompt_file.read_text(encoding='utf-8') if args.prompt_file else 'Explain in detail how to measure communication and computation bottlenecks in a four-node inference cluster. Include a reproducible experimental design and discuss confounders.'}],'temperature':0,'seed':42,'max_tokens':args.tokens,'stream':True,'return_token_ids':True,'stream_options':{'include_usage':True,'continuous_usage_stats':True}}
    req=urllib.request.Request(args.url+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    start=time.perf_counter();events=[];usage={};done=False
    with urllib.request.urlopen(req,timeout=120) as response:
        for line in response:
            if not line.startswith(b'data:'):continue
            data=line[5:].strip()
            if data==b'[DONE]':done=True;break
            obj=json.loads(data)
            if obj.get('error'):raise RuntimeError(obj['error'])
            if obj.get('usage'):usage=obj['usage']
            for choice in obj.get('choices',[]):
                assert choice['index']==0
                ids=choice.get('token_ids')
                events.append({'time_s':time.perf_counter()-start,'token_ids':ids,'completion_tokens':usage.get('completion_tokens'),'finish_reason':choice.get('finish_reason')})
    after=metrics(args.url)
    delta={k:after.get(k,0)-before.get(k,0) for k in set(before)|set(after) if 'num_requests_' not in k}
    ids=[token for e in events for token in (e['token_ids'] or [])]
    checks={'done':done,'token_ids_present':bool(ids),'token_count_matches_usage':len(ids)==usage.get('completion_tokens'),'one_success':sum(v for k,v in delta.items() if 'request_success_total' in k)==1,'idle_after':not any(v for k,v in after.items() if 'num_requests_' in k)}
    result={'request':body,'events':events,'usage':usage,'metric_deltas':delta,'checks':checks,'nonempty_token_groups':sum(bool(e['token_ids']) for e in events),'token_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'note':'SSE grouping is not yet assumed to equal engine iterations. Validate counters and grouping before inferring acceptance transitions.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('usage','metric_deltas','checks','nonempty_token_groups')},indent=2))
    assert all(checks.values()),checks
if __name__=='__main__':main()
