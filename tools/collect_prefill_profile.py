"""Collect bounded profile files and inventory scopes without assuming attribution."""
import argparse
import collections
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
remote='/tmp/qwen029-prefill-profile-20260910-r1'
code='''import pathlib,tarfile,sys
root=pathlib.Path(ROOT)
files=list(root.rglob('*.trace.json.gz'))
assert len(files)==1,files
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for f in files:archive.add(f,arcname=f.name,recursive=False)
'''
summary=[]
for host,container in [('192.168.1.15','qwen029-tp2'),('192.168.1.16','qwen029-tp2'),
                       ('192.168.1.17','qwen029-tp4'),('192.168.1.18','qwen029-tp4')]:
    r=subprocess.run(['ssh','-o','BatchMode=yes','maik@'+host,
        'podman exec -i '+container+' python3 -'],input=('ROOT='+repr(remote)+'\n'+code).encode(),
        capture_output=True,check=True,timeout=120)
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as archive:
        files=[m for m in archive.getmembers() if m.isfile()]
        assert len(files)==1
        member=files[0]
        assert member.name==Path(member.name).name
        data=archive.extractfile(member).read()
    folder=a.output/host
    folder.mkdir()
    (folder/member.name).write_bytes(data)
    events=json.loads(gzip.decompress(data))['traceEvents']
    scopes=collections.Counter(e.get('name','') for e in events if e.get('cat')=='user_annotation')
    kernels=[e for e in events if e.get('cat')=='kernel']
    totals=collections.Counter()
    for e in kernels:totals[e['name']]+=e.get('dur',0)
    row=dict(host=host,trace_name=member.name,sha256=hashlib.sha256(data).hexdigest(),
             kernel_count=len(kernels),scopes=dict(scopes),
             top_kernel_summed_us=totals.most_common(30),
             note='Whole captured window; not yet separated into prefill and decode.')
    summary.append(row)
    (a.output/'inventory.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'host':host,'kernels':len(kernels),
                      'execute_scopes':{k:v for k,v in scopes.items() if 'execute_context' in k}}),flush=True)
