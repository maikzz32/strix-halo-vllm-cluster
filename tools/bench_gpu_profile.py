"""Reversible all-node GPU auto/high/auto experiment; no manual clocks or power limits."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/gpu-profile-20260910'
REMOTE='''import pathlib,json
cards=[p for p in pathlib.Path('/sys/class/drm').glob('card[0-9]*') if (p/'device/vendor').exists() and (p/'device/vendor').read_text().strip()=='0x1002']
assert len(cards)==1
p=cards[0]/'device/power_dpm_force_performance_level'
def snapshot():return {'path':str(p.resolve()),'level':p.read_text().strip()}
before=snapshot()
if ACTION!='read':
 assert before['path']==ORIGINAL['path']
 assert before['level'] in (ORIGINAL['level'],'high')
 try:
  p.write_text('high' if ACTION=='set' else ORIGINAL['level'])
 except BaseException:
  p.write_text(ORIGINAL['level'])
  raise
after=snapshot()
if ACTION=='set':assert after['level']=='high'
if ACTION=='restore':assert after==ORIGINAL
print(json.dumps(after))
'''

def remote(host,action,original=None):
    source='ACTION='+repr(action)+'\nORIGINAL='+repr(original)+'\n'+REMOTE
    r=subprocess.run(['ssh','maik@'+host,'sudo -n python3 -S -'],input=source,text=True,capture_output=True,check=True,timeout=45)
    return json.loads(r.stdout)
def bench(name,tokens=256,repeats=2):
    subprocess.run(['python',str(ROOT/'bench/bench_stream.py'),'--url','http://192.168.1.15:8000','--tokens',str(tokens),'--repeats',str(repeats),'--concurrency','1','--output',str(OUT/(name+'.json'))],cwd=ROOT,check=True,timeout=180)
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    hosts=['192.168.1.'+str(i) for i in range(15,19)]
    original={h:remote(h,'read') for h in hosts}
    assert all(row['level']=='auto' for row in original.values())
    (OUT/'original.json').write_text(json.dumps(original,indent=2)+'\n')
    bench('warmup',32,1);bench('before')
    attempted=[];restored={}
    try:
        applied={}
        for h in hosts:
            attempted.append(h);applied[h]=remote(h,'set',original[h])
        (OUT/'applied.json').write_text(json.dumps(applied,indent=2)+'\n')
        print('GPU high profile verified on all four nodes',flush=True)
        bench('high')
    finally:
        errors=[]
        for h in attempted:
            try:restored[h]=remote(h,'restore',original[h])
            except BaseException as e:errors.append({'host':h,'error':str(e)})
        (OUT/'restore.json').write_text(json.dumps({'states':restored,'errors':errors},indent=2)+'\n')
        if errors:raise RuntimeError('GPU profile restoration needs attention: '+str(errors))
        print('Original GPU profile restored and verified on all attempted nodes',flush=True)
    bench('after')
    reports={n:json.loads((OUT/(n+'.json')).read_text(encoding='utf-8')) for n in ('before','high','after')}
    indexed={n:{(x['prompt'],x['repeat']):x for x in r['results']} for n,r in reports.items()}
    rows=[]
    for key,b in indexed['before'].items():
        row={'prompt':key[0],'repeat':key[1]}
        for n in ('high','after'):
            c=indexed[n][key]
            assert b['request']==c['request'] and b['usage']==c['usage']
            assert (b['content'],b['reasoning'])==(c['content'],c['reasoning'])
        for metric in ('decode_tps','ttft_s','end_to_end_tps'):
            row[metric]={n:indexed[n][key][metric] for n in indexed}
        rows.append(row)
    (OUT/'comparison.json').write_text(json.dumps({'identical_outputs':True,'rows':rows},indent=2)+'\n')
    print('GPU_PROFILE_EXPERIMENT_COMPLETE',flush=True)
if __name__=='__main__':main()
