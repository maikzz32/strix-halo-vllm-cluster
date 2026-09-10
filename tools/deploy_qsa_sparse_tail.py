"""Stage, activate or restore the exact screened QSA source on all ranks."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '''import pathlib, hashlib, json
scope={};exec(GENERATOR,scope)
p=pathlib.Path('/usr/local/lib64/python3.12/site-packages/vllm/models/qwen4_exp/amd/ops/qsa.py')
root=pathlib.Path('/opt/strix-halo-next/qsa-tail-bound-20260910-r1')
sha=lambda b:hashlib.sha256(b).hexdigest()
if ACTION=='stage':
 root.mkdir(exist_ok=True)
 original=p.read_bytes()
 candidate=scope['generate'](original)
 for name,data in [('original.py',original),('candidate.py',candidate)]:
  target=root/name
  if target.exists():assert target.read_bytes()==data
  else:target.write_bytes(data)
else:
 original=(root/'original.py').read_bytes()
 candidate=scope['generate'](original)
 assert (root/'candidate.py').read_bytes()==candidate
 assert p.read_bytes() in (original,candidate)
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:cmd=proc.joinpath('cmdline').read_bytes()
  except FileNotFoundError:continue
  assert b'VLLM::Worker' not in cmd and b'VLLM::EngineCore' not in cmd, 'Stop service before mutation'
 if ACTION!='check':
  pending=p.with_name(p.name+'.qsa-pending')
  pending.write_bytes(candidate if ACTION=='activate' else original)
  pending.replace(p)
print(json.dumps(dict(original_sha=sha(original),candidate_sha=sha(candidate),active_sha=sha(p.read_bytes()))))
'''
parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','activate','restore']);args=parser.parse_args()
generator=(ROOT/'cluster-repo/patches/qsa_sparse_tail_bound.py').read_text(encoding='utf-8-sig')
nodes=json.loads((ROOT/'config/qwen029-tp4-graph-inventory.json').read_text())['nodes']
records=[]
for action in (['stage'] if args.action=='stage' else ['check',args.action]):
 for node in nodes:
  code='GENERATOR='+repr(generator)+'\nACTION='+repr(action)+'\n'+REMOTE
  r=subprocess.run(['ssh','-o','BatchMode=yes','maik@'+node['host'],'podman exec -i '+node['container']+' python3 -S -'],input=code.encode(),capture_output=True,timeout=30)
  assert r.returncode==0,(node['host'],r.stderr.decode())
  records.append(dict(host=node['host'],action=action,**json.loads(r.stdout)))
  print(records[-1],flush=True)
(ROOT/f'results/qsa-tail-{args.action}-20260910.json').write_text(json.dumps(records,indent=2))
