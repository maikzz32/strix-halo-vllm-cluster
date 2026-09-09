"""Compare matching tokenized contexts, outputs, cache coverage and latency."""
import argparse
import hashlib
import json
from pathlib import Path

def compare(before, after):
    a=json.loads(before.read_text()); b=json.loads(after.read_text())
    if a['model']!=b['model']:raise ValueError('Different model identifiers')
    for key in ('lengths','tokens','repeats'):
        if a['arguments'][key]!=b['arguments'][key]:raise ValueError('Different '+key)
    if len(a['cases'])!=len(a['arguments']['lengths']) or len(b['cases'])!=len(a['cases']):
        raise ValueError('Incomplete cases')
    rows=[]
    for x,y in zip(a['cases'],b['cases']):
        if x['prompt_ids_sha256']!=y['prompt_ids_sha256']:raise ValueError('Different prompt tokens')
        if len(x['requests'])!=a['arguments']['repeats'] or len(y['requests'])!=len(x['requests']):
            raise ValueError('Incomplete repetitions')
        for u,v in zip(x['requests'],y['requests']):
            if u['repeat']!=v['repeat']:raise ValueError('Different repetition')
            hits=lambda r:sum(value for key,value in r['counter_delta'].items() if key.startswith('vllm:prefix_cache_hits_total'))
            rows.append({'prompt_tokens':x['prompt_tokens'],'repeat':u['repeat'],
                'identical_output':u['text_sha256']==v['text_sha256'],
                'identical_usage':u['usage']==v['usage'],
                'cache_hits_before':hits(u),'cache_hits_after':hits(v),
                'ttft_ms_before':u['ttft_ms'],'ttft_ms_after':v['ttft_ms'],
                'ttft_change_percent':100*(v['ttft_ms']/u['ttft_ms']-1)})
    return {'before_sha256':hashlib.sha256(before.read_bytes()).hexdigest(),
        'after_sha256':hashlib.sha256(after.read_bytes()).hexdigest(),'cases':rows,
        'note':'Individual timing samples; matched input tokens and outputs do not prove a causal speedup. Check cache coverage and a post-trial control.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('before',type=Path);p.add_argument('after',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=compare(a.before,a.after)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
