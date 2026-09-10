"""Run four isolated ranks with RCCL environment copied from verified workers."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import uuid

p=argparse.ArgumentParser()
p.add_argument('--service-run-id',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert len(a.service_run_id)==32 and all(c in '0123456789abcdef' for c in a.service_run_id)
source=(Path(__file__).resolve().parents[1]/'tests/bench_rccl_prefill_sizes.py').read_text()
run_id=uuid.uuid4().hex
def metrics():
    s=urllib.request.urlopen('http://192.168.1.15:8000/metrics',timeout=10).read().decode()
    d={l.split()[0]:float(l.split()[-1]) for l in s.splitlines()
       if l.startswith(('vllm:num_requests_running','vllm:num_requests_waiting','vllm:request_success_total'))}
    assert any('num_requests_running' in k for k in d)
    assert not any(v for k,v in d.items() if 'num_requests_' in k)
    return d
before=metrics()
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'manifest.json').write_text(json.dumps(dict(run_id=run_id,
    service_run_id=a.service_run_id,test_sha256=hashlib.sha256(source.encode()).hexdigest(),before=before)))
remote='''import pathlib,os,subprocess,json,socket
cfg=CONFIG
workers=[]
for p in pathlib.Path('/proc').iterdir():
 try:
  if not p.name.isdigit() or b'VLLM::Worker' not in (p/'cmdline').read_bytes():continue
  env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(bytes([0])) if b'=' in x)
  if env.get(b'STRIX_CLUSTER_RUN_ID')==cfg['service_run_id'].encode():workers.append((p.name,env))
 except FileNotFoundError:pass
assert len(workers)==1,workers
inherited={k.decode():v.decode() for k,v in workers[0][1].items() if k.startswith((b'NCCL_',b'GLOO_'))}
assert inherited['NCCL_MIN_NCHANNELS']=='4' and inherited['NCCL_MAX_NCHANNELS']=='4'
if cfg['rank']==0:
 with socket.socket() as check:check.bind(('192.168.100.1',29783))
root=pathlib.Path('/tmp/rccl-prefill-'+cfg['run_id']);root.mkdir()
(root/'test.py').write_text(cfg['source'])
env=dict(os.environ)
for k in list(env):
 if k.startswith(('NCCL_','GLOO_')):del env[k]
env.update(inherited)
env['GLOO_SOCKET_IFNAME']=cfg['iface']
env['STRIX_RCCL_PREFILL_PROBE']=cfg['run_id']
r=subprocess.run(['timeout','--signal=TERM','--kill-after=5s','155s','python3',str(root/'test.py'),
 '--rank',str(cfg['rank']),'--master','192.168.100.1','--port','29783','--output',str(root/'result.json')],
 env=env,capture_output=True,timeout=165)
remaining=[]
for p in pathlib.Path('/proc').iterdir():
 try:
  if p.name.isdigit() and ('STRIX_RCCL_PREFILL_PROBE='+cfg['run_id']).encode() in (p/'environ').read_bytes().split(bytes([0])):remaining.append(p.name)
 except OSError:pass
print(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode(),
 worker_pid=workers[0][0],inherited=inherited,remaining=remaining,
 result=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else None)))
'''
def one(rank):
    cfg=dict(rank=rank,run_id=run_id,service_run_id=a.service_run_id,source=source,
             iface='enp197s0f1np1' if rank==3 else 'enp197s0f3np3')
    container='qwen029-tp2' if rank<2 else 'qwen029-tp4'
    r=subprocess.run(['ssh','-o','BatchMode=yes','maik@192.168.1.'+str(rank+15),
        'podman exec -i '+container+' python3 -'],input=remote.replace('CONFIG',repr(cfg),1).encode(),
        capture_output=True,timeout=180)
    (a.output/f'rank{rank}.log').write_bytes(r.stdout+r.stderr)
    assert r.returncode==0,(rank,r.stderr.decode())
    result=json.loads(r.stdout)
    (a.output/f'rank{rank}.json').write_text(json.dumps(result,indent=2))
    assert result['exit_code']==0 and not result['remaining'],result
    assert result['result']['status']=='passed'
    return result
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(one,rank) for rank in range(4)]
    results=[]
    errors=[]
    for f in futures:
        try:results.append(f.result())
        except Exception as e:errors.append(str(e))
after=metrics()
(a.output/'after.json').write_text(json.dumps(after))
assert not errors,errors
assert before==after,'Serving activity changed; reject timings'
(a.output/'summary.json').write_text(json.dumps(dict(run_id=run_id,
    service_counters_unchanged=True,ranks=[r['result'] for r in results]),indent=2))
for r in results:print(r['result']['rank'],[(c['bytes'],c['median_us']) for c in r['result']['cases']])
