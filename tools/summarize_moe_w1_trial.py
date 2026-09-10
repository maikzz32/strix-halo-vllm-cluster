"""Summarize a bracketed W1 trial without publishing generated answer text."""
import argparse,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 raw={name:path.read_bytes() for name,path in [('before',a.before),('candidate',a.candidate),('after',a.after)]}
 data={name:json.loads(b) for name,b in raw.items()}
 keys=['completed','failed','total_input_tokens','total_output_tokens','output_throughput','mean_tpot_ms','mean_ttft_ms']
 report={'scope':'ShareGPT48 C1, original/candidate/original, same UMA transport','runs':{},'comparisons':{}}
 for name,r in data.items():
  assert r['completed']==48 and r['failed']==0
  report['runs'][name]={**{k:r[k] for k in keys},'spec':{k:v for k,v in r.items() if k.startswith('spec_decode')},'result_sha256':hashlib.sha256(raw[name]).hexdigest(),'answer_sha256':[hashlib.sha256(t.encode()).hexdigest() for t in r['generated_texts']],'input_lens':r['input_lens'],'output_lens':r['output_lens']}
 for name in ('before','after'):
  b=data[name];c=data['candidate']
  report['comparisons'][name]={'identical_answers':sum(x==y for x,y in zip(b['generated_texts'],c['generated_texts'])),'input_lens_equal':b['input_lens']==c['input_lens'],'output_lens_equal':b['output_lens']==c['output_lens'],'spec_equal':report['runs'][name]['spec']==report['runs']['candidate']['spec'],'throughput_gain_pct':100*(c['output_throughput']/b['output_throughput']-1),'decode_speed_gain_pct':100*(b['mean_tpot_ms']/c['mean_tpot_ms']-1)}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['comparisons'],indent=2))
if __name__=='__main__':main()
