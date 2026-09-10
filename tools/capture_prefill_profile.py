"""Profile before request submission; keep raw text local for parity checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid

p=argparse.ArgumentParser()
p.add_argument('--run-id',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
base='http://192.168.1.15:8000'
def request(path,body=None):
    return urllib.request.urlopen(urllib.request.Request(base+path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type':'application/json'}),timeout=120)
deadline=time.monotonic()+900
while True:
    try:
        with request('/health') as r:
            if r.status==200:break
    except OSError:pass
    if time.monotonic()>deadline:raise TimeoutError('Inspect existing startup; do not restart')
    time.sleep(5)
proof={}
for host,container in [('192.168.1.15','qwen029-tp2'),('192.168.1.16','qwen029-tp2'),
                       ('192.168.1.17','qwen029-tp4'),('192.168.1.18','qwen029-tp4')]:
    code='''import pathlib,json
rows=[]
for p in pathlib.Path('/proc').iterdir():
 try:
  if not p.name.isdigit() or b'VLLM::Worker' not in (p/'cmdline').read_bytes():continue
  e=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\\0') if b'=' in x)
  if e.get(b'STRIX_CLUSTER_RUN_ID')==RUN.encode():rows.append(p.name)
 except FileNotFoundError:pass
assert len(rows)==1,rows
print(json.dumps(rows))
'''
    r=subprocess.run(['ssh','-o','BatchMode=yes','cluster-user@'+host,
        'podman exec -i '+container+' python3 -'],input=('RUN='+repr(a.run_id)+'\n'+code).encode(),
        capture_output=True,check=True,timeout=30)
    proof[host]=json.loads(r.stdout)
(a.output/'workers.json').write_text(json.dumps(proof))
with request('/v1/models') as r:model=json.load(r)['data'][0]['id']
text='Profile '+uuid.uuid4().hex+'\n'+'Inventory remains available on all four servers. '*160
with request('/tokenize',{'model':model,'prompt':text}) as r:ids=json.load(r)['tokens'][:512]
assert len(ids)==512
body={'model':model,'prompt':ids,'max_tokens':32,'ignore_eos':True,'temperature':0,'seed':42}
with request('/metrics') as r:before=r.read().decode()
assert not any(float(l.split()[-1]) for l in before.splitlines()
               if l.startswith(('vllm:num_requests_running','vllm:num_requests_waiting')))
with request('/v1/completions',body) as r:reference=json.load(r)
(a.output/'reference.json').write_text(json.dumps(reference))
started=False
try:
    with request('/start_profile',{}) as r:assert r.status==200
    started=True
    with request('/v1/completions',body) as r:candidate=json.load(r)
    (a.output/'candidate.json').write_text(json.dumps(candidate))
finally:
    if started:
        with request('/stop_profile',{}) as r:assert r.status==200
with request('/metrics') as r:after=r.read().decode()
(a.output/'before.metrics').write_text(before)
(a.output/'after.metrics').write_text(after)
equal=(reference['choices'][0]['text']==candidate['choices'][0]['text'] and
       reference['usage']==candidate['usage'])
record=dict(run_id=a.run_id,prompt_tokens=512,output_tokens=32,outputs_equal=equal,
    request_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),
    note='Profiler started before second request; not a performance benchmark.')
(a.output/'summary.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
assert equal
