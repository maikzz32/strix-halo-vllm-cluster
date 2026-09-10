"""Read-only five-second worker thread scheduling sample on all four nodes."""
import concurrent.futures,json,pathlib,subprocess
REMOTE=r'''
import pathlib,time,json,os
workers=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:cmd=p.joinpath('cmdline').read_bytes().split(b'\0')
 except FileNotFoundError:continue
 if any(a.startswith(b'VLLM::Worker_TP') for a in cmd):workers.append(p)
assert len(workers)==1,[(p.name,p.joinpath('cmdline').read_bytes().decode()) for p in workers]
p=workers[0]
def snap():
 r={}
 for t in p.joinpath('task').iterdir():
  try:
   s=t.joinpath('stat').read_text().rsplit(')',1)[1].split()
   r[t.name]={'ticks':int(s[11])+int(s[12]),'cpu':int(s[36]),'comm':t.joinpath('comm').read_text().strip(),'affinity':sorted(os.sched_getaffinity(int(t.name))),'migrations':[l.strip() for l in t.joinpath('sched').read_text().splitlines() if 'nr_migrations' in l]}
  except FileNotFoundError:pass
 return r
first=snap();time.sleep(5);second=snap()
r=[{'tid':int(k),'tick_delta':v['ticks']-first[k]['ticks'],'before':first[k],'after':v} for k,v in second.items() if k in first]
print(json.dumps({'worker':int(p.name),'threads':sorted(r,key=lambda r:-r['tick_delta'])}))
'''
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 def run(n):
  c='ray-head' if n==15 else 'ray-worker'
  r=subprocess.run(['ssh','-o','BatchMode=yes',f'cluster-user@192.168.1.{n}',f'podman exec -i {c} python3 -S -'],input=REMOTE.encode(),capture_output=True,timeout=20)
  (a.output/f'node{n}.log').write_bytes(r.stdout+r.stderr);assert r.returncode==0,r.stderr
  d=json.loads(r.stdout);(a.output/f'node{n}.json').write_text(json.dumps(d,indent=2))
  return {'node':n,'worker':d['worker'],'top_threads':[(t['tid'],t['tick_delta'],t['after']['cpu'],t['after']['migrations']) for t in d['threads'][:5]]}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(run,n) for n in (15,16,17,18)]
  errors=[]
  for f in futures:
   try:print(json.dumps(f.result()))
   except Exception as e:errors.append(str(e))
  if errors:raise RuntimeError(errors)
if __name__=='__main__':main()
