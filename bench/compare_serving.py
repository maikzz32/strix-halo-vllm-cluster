"""Compare two matching raw vLLM serving results and retain parity evidence."""
import argparse
import hashlib
import json
from pathlib import Path

def compare(before_path, after_path):
    a=json.loads(before_path.read_text(encoding='utf-8'))
    b=json.loads(after_path.read_text(encoding='utf-8'))
    for key in ('model_id','backend','num_prompts','max_concurrency','input_lens'):
        if a.get(key)!=b.get(key):raise ValueError('Workload mismatch: '+key)
    if a.get('failed') or b.get('failed'):raise ValueError('Failed requests in comparison')
    if len(a.get('generated_texts',[]))!=a['completed'] or a['completed']!=b['completed']:
        raise ValueError('Incomplete response evidence')
    if len(b.get('generated_texts',[]))!=b['completed']:raise ValueError('Missing responses')
    stats=('spec_decode_num_drafts','spec_decode_draft_tokens','spec_decode_accepted_tokens',
           'spec_decode_per_position_acceptance_rates')
    output={
        'before_raw_sha256':hashlib.sha256(before_path.read_bytes()).hexdigest(),
        'after_raw_sha256':hashlib.sha256(after_path.read_bytes()).hexdigest(),
        'matched_requests':a['completed'],
        'identical_texts':sum(x==y for x,y in zip(a['generated_texts'],b['generated_texts'])),
        'identical_output_lengths':a['output_lens']==b['output_lens'],
        'identical_speculation':all(a.get(k)==b.get(k) for k in stats),
        'metrics':{},
        'note':'Request bodies, seed, temperature, config and dataset must also match; raw result files do not encode every parameter.'}
    for key in ('output_throughput','mean_ttft_ms','median_ttft_ms','p99_ttft_ms','mean_tpot_ms','p99_tpot_ms'):
        output['metrics'][key]={'before':a[key],'after':b[key],'change_percent':100*(b[key]/a[key]-1)}
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('before',type=Path);p.add_argument('after',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=compare(a.before,a.after)
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
