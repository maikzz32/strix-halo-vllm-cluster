"""Run a hash-checked HC-down prefetch microbenchmark while serving is idle."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

p=argparse.ArgumentParser()
p.add_argument('--library',required=True,help='Absolute library path inside the Node 4 container')
p.add_argument('--sha256',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert a.library.startswith('/tmp/') and len(a.sha256)==64
root=Path(__file__).resolve().parents[1]
test=(root/'tests/bench_hc_down_prefetch.py').read_text()

def metrics():
    with urllib.request.urlopen('http://192.168.1.15:8000/metrics',timeout=10) as r:
        s=r.read().decode()
    return {line.split()[0]:float(line.split()[-1]) for line in s.splitlines()
            if line.startswith(('vllm:num_requests_running','vllm:num_requests_waiting','vllm:request_success_total'))}

before=metrics()
assert any('num_requests_running' in k for k in before)
assert not any(v for k,v in before.items() if 'num_requests_' in k)
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'before.json').write_text(json.dumps(before))
payload=('import pathlib,tempfile,sys,hashlib,json\n'
         f'lib=pathlib.Path({a.library!r})\n'
         f'assert hashlib.sha256(lib.read_bytes()).hexdigest()=={a.sha256!r}\n'
         'd=pathlib.Path(tempfile.mkdtemp(prefix="hc-down-prefetch-test-"))\n'
         f's={test!r}\n'
         '(d/"test.py").write_text(s)\n'
         f'sys.argv=[str(d/"test.py"),"--library",str(lib),"--sha256",{a.sha256!r},"--output",str(d/"result.json")]\n'
         'exec(compile(s,str(d/"test.py"),"exec"),{"__name__":"__main__","__file__":str(d/"test.py")})\n'
         'print("RESULT="+json.dumps(json.loads((d/"result.json").read_text())))\n')
r=subprocess.run(['ssh','-o','BatchMode=yes','cluster-user@192.168.1.18',
                  'podman exec -i qwen029-tp4 timeout --signal=TERM --kill-after=5s 150s python3 -'],
                 input=payload.encode(),capture_output=True,timeout=175)
(a.output/'run.log').write_bytes(r.stdout+r.stderr)
after=metrics();(a.output/'after.json').write_text(json.dumps(after))
assert r.returncode==0,(r.returncode,r.stderr.decode(errors='replace')[-1500:])
assert before==after,'Serving activity changed; timing must not be accepted'
rows=[json.loads(line[7:]) for line in r.stdout.decode().splitlines() if line.startswith('RESULT=')]
assert len(rows)==1 and rows[0]['status']=='passed'
result=rows[0]
result.update(library_sha256=a.sha256,test_sha256=hashlib.sha256(test.encode()).hexdigest(),
              host='192.168.1.18',service_counters_unchanged=True)
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
