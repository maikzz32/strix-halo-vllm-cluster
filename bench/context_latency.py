"""Measure exact-length cold/warm prefixes with timing and cache counters."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import uuid

def request(url, body=None):
    return urllib.request.urlopen(urllib.request.Request(url,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type':'application/json'}),timeout=600)

def counters(base, phases=False):
    with request(base+'/metrics') as r: lines=r.read().decode().splitlines()
    result = {line.split()[0]:float(line.split()[1]) for line in lines
        if line.startswith(('vllm:prefix_cache_queries_total','vllm:prefix_cache_hits_total',
                            'vllm:num_requests_running','vllm:num_requests_waiting'))}
    if phases:
        names = ('time_to_first_token_seconds', 'request_queue_time_seconds',
                 'request_inference_time_seconds', 'request_prefill_time_seconds',
                 'request_decode_time_seconds', 'request_prefill_kv_computed_tokens')
        wanted = {'vllm:' + name + suffix for name in names for suffix in ('_sum', '_count')}
        wanted.update(('vllm:iteration_tokens_total_sum', 'vllm:iteration_tokens_total_count'))
        for line in lines:
            if line and not line.startswith('#') and line.split('{')[0].split()[0] in wanted:
                result[line.split()[0]] = float(line.split()[1])
    return result

def run(args):
    phase_metrics = getattr(args, 'phase_metrics', False)
    with request(args.url+'/v1/models') as r:model=json.load(r)['data'][0]['id']
    initial=counters(args.url)
    if any(v for k,v in initial.items() if k.startswith('vllm:num_requests_')):
        raise RuntimeError('Server is busy')
    record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'model':model,'arguments':vars(args).copy(),'cases':[],
            'note':'First use of a unique prefix versus exact reuse; speed alone does not prove cache hits.'}
    record['arguments']['output']=str(args.output)
    salt=args.prefix_id or uuid.uuid4().hex
    record['prefix_id']=salt
    for length in args.lengths:
        # Unique tokens near the beginning prevent reuse of an earlier probe.
        text='Experiment '+salt+' context '+str(length)+'.\n'
        text+='\n'.join('Record %d: Warehouse inventory remains available on all four servers.'%i for i in range(length//5+100))
        text+='\nSummarize the inventory records in a short paragraph.'
        with request(args.url+'/tokenize',{'model':model,'messages':[{'role':'user','content':text}],
            'chat_template_kwargs':{'enable_thinking':False}}) as r:tok=json.load(r)
        ids=tok['tokens']
        if len(ids)<length:raise RuntimeError('Insufficient tokens')
        # Preserve the closing template and question; truncate only context.
        ids=ids[:length-128]+ids[-128:]
        assert len(ids)==length
        case={'prompt_tokens':length,'prompt_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'requests':[]}
        record['cases'].append(case)
        for repeat in range(args.repeats):
            before=counters(args.url, phase_metrics)
            body={'model':model,'prompt':ids,'max_tokens':args.tokens,'temperature':0,
                  'seed':42,'stream':True,'stream_options':{'include_usage':True}}
            started=time.perf_counter();first=None;parts=[];usage={};done=False
            with request(args.url+'/v1/completions',body) as response:
                for line in response:
                    if not line.startswith(b'data:'):continue
                    raw=line[5:].strip()
                    if raw==b'[DONE]':done=True;break
                    e=json.loads(raw)
                    if e.get('error'):raise RuntimeError(e['error'])
                    if e.get('usage'):usage=e['usage']
                    for choice in e.get('choices',[]):
                        part=choice.get('text') or ''
                        if part and first is None:first=time.perf_counter()
                        parts.append(part)
            elapsed=time.perf_counter()-started
            assert done and first is not None and usage['prompt_tokens']==length
            after=counters(args.url, phase_metrics)
            row={'repeat':repeat,'prefix_state':'first_use' if repeat==0 else 'reused',
                 'ttft_ms':(first-started)*1000,'wall_s':elapsed,'usage':usage,
                 'text_sha256':hashlib.sha256(''.join(parts).encode()).hexdigest(),
                 'counter_delta':{k:after[k]-before.get(k,0) for k in after if not k.startswith('vllm:num_requests_')}}
            case['requests'].append(row)
            if phase_metrics:
                counts = {k:v for k,v in row['counter_delta'].items() if '_count{' in k
                          and not k.startswith('vllm:iteration_tokens_total')}
                assert len(counts) == 6 and all(v == 1 for v in counts.values()), counts
                assert not any(v for k,v in after.items() if k.startswith('vllm:num_requests_'))
            args.output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
            print(json.dumps(row),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://192.168.1.15:8000')
    p.add_argument('--lengths',type=lambda s:[int(v) for v in s.split(',')],default=[2048,8192,32768])
    p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prefix-id',help='Reuse this identifier across restarted before/candidate/after runs for identical inputs.')
    p.add_argument('--phase-metrics',action='store_true',help='Collect per-request phase histogram deltas; requires an otherwise idle server.')
    a=p.parse_args()
    if min(a.lengths)<256 or a.repeats<2:p.error('Lengths >=256 and at least two repeats required')
    run(a)
