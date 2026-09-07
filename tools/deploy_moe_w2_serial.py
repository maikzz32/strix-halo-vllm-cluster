"""Stage/activate/restore the bounded W2 serial candidate; defaults to dry run."""
import argparse, hashlib, json, subprocess, uuid
from pathlib import Path
from patch_moe_w2_serial import BASE_SHA,patched
ROOT=Path(__file__).resolve().parents[1]
RELATIVE='usr/local/lib64/python3.12/site-packages/vllm/model_executor/layers/fused_moe/fused_moe.py'
ADDON='/opt/strix-halo-next/moe-w2-serial-20260907-r1'
REMOTE=r"""
import pathlib,hashlib,json
cfg=CONFIG
target=pathlib.Path('/'+cfg['relative']);root=pathlib.Path(cfg['addon'])
sha=lambda b:hashlib.sha256(b).hexdigest()
current=target.read_bytes()
if cfg['action']=='stage':
 assert sha(current)==cfg['base_sha']
 root.mkdir(parents=True,exist_ok=False)
 candidate=cfg['patched'].encode();compile(candidate,str(target),'exec')
 root.joinpath('original.py').write_bytes(current);root.joinpath('candidate.py').write_bytes(candidate)
 record={'base_sha':sha(current),'candidate_sha':sha(candidate)}
 root.joinpath('manifest.json').write_text(json.dumps(record))
else:
 for d in pathlib.Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:cmd=d.joinpath('cmdline').read_bytes().split(bytes([0]))
  except OSError:continue
  if any(b'VLLM::Worker' in a for a in cmd) or (b'serve' in cmd and any(a.endswith(b'/vllm') for a in cmd)):
   raise RuntimeError('Stop serving processes before changing source')
 record=json.loads(root.joinpath('manifest.json').read_text())
 assert sha(current) in (record['base_sha'],record['candidate_sha'])
 key='candidate' if cfg['action']=='activate' else 'original'
 data=root.joinpath(key+'.py').read_bytes()
 assert sha(data)==record['candidate_sha' if key=='candidate' else 'base_sha']
 tmp=target.with_suffix('.w2-serial-pending');tmp.write_bytes(data);tmp.replace(target)
print(json.dumps({'action':cfg['action'],'source_sha':sha(target.read_bytes()),**record}))
"""
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','activate','restore']);p.add_argument('--execute',action='store_true');p.add_argument('--base-source',type=Path);args=p.parse_args()
 cfg={'action':args.action,'relative':RELATIVE,'addon':ADDON,'base_sha':BASE_SHA}
 if args.action=='stage':
  path=args.base_source or ROOT/'snapshots/runtime-source'/RELATIVE
  cfg['patched']=patched(path.read_bytes()).decode()
 if not args.execute:print(json.dumps({k:v for k,v in cfg.items() if k!='patched'}));return
 dest=ROOT/'results'/('moe-w2-serial-deploy-'+args.action+'-'+uuid.uuid4().hex);dest.mkdir(parents=True)
 for node in (15,16,17,18):
  container='ray-head' if node==15 else 'ray-worker'
  r=subprocess.run(['ssh','-o','BatchMode=yes',f'maik@192.168.1.{node}',f'podman exec -i {container} python3 -S -'],input=REMOTE.replace('CONFIG',repr(cfg),1).encode(),capture_output=True,timeout=30)
  dest.joinpath(f'node{node}.log').write_bytes(r.stdout+r.stderr);print(node,r.returncode,r.stdout.decode(),r.stderr.decode(),flush=True)
  if r.returncode:raise RuntimeError('Deployment stopped; inspect partial state and restore if needed')
if __name__=='__main__':main()
