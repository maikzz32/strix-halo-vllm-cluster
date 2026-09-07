"""Stage, activate or restore the opt-in GDN candidate; dry run by default."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

REMOTE = r'''
import pathlib,hashlib,json,base64
cfg=CONFIG
root=pathlib.Path('/opt/strix-halo-next/aiter-gdn-20260907-r1')
target=pathlib.Path('/usr/local/lib64/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py')
sha=lambda b:hashlib.sha256(b).hexdigest()
if cfg['action']=='stage':
 data={n:base64.b64decode(v) for n,v in cfg['files'].items()}
 assert sha(target.read_bytes())==cfg['manifest']['base_sha']
 assert not root.exists(), 'Stage directory already exists; inspect before retry'
 for n,b in data.items():
  assert sha(b)==cfg['manifest']['files'][n]
  compile(b,n,'exec')
 root.mkdir(parents=True)
 for n,b in data.items():root.joinpath(n).write_bytes(b)
 root.joinpath('manifest.json').write_text(json.dumps(cfg['manifest']))
else:
 for d in pathlib.Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:cmd=d.joinpath('cmdline').read_bytes().split(bytes([0]))
  except OSError:continue
  if any(b'VLLM::Worker' in a for a in cmd) or (b'serve' in cmd and any(a.endswith(b'/vllm') for a in cmd)):
   raise RuntimeError('Stop serving before activation or restoration')
 m=json.loads(root.joinpath('manifest.json').read_text())
 assert sha(target.read_bytes()) in (m['files']['original.py'],m['files']['candidate.py'])
 if cfg['action']=='activate':
  for n in ('_strix_gdn_kernel.py','_strix_gdn.py'):
   data=root.joinpath(n).read_bytes();assert sha(data)==m['files'][n]
   dest=target.parent/n
   assert not dest.exists() or sha(dest.read_bytes())==sha(data)
   temp=dest.with_suffix('.gdn-pending');temp.write_bytes(data);temp.replace(dest)
 name='candidate.py' if cfg['action']=='activate' else 'original.py'
 data=root.joinpath(name).read_bytes();assert sha(data)==m['files'][name]
 temp=target.with_suffix('.gdn-pending');temp.write_bytes(data);temp.replace(target)
print(json.dumps({'action':cfg['action'],'root':str(root),'source_sha':sha(target.read_bytes())}))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['stage', 'activate', 'restore'])
    p.add_argument('--bundle', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    cfg = {'action': args.action}
    if args.action == 'stage':
        if args.bundle is None:
            p.error('--bundle required for stage')
        manifest = json.loads((args.bundle/'manifest.json').read_text())
        expected = {'original.py', 'candidate.py', '_strix_gdn.py', '_strix_gdn_kernel.py'}
        assert set(manifest['files']) == expected
        cfg['manifest'] = manifest
        cfg['files'] = {}
        for name in expected:
            data = (args.bundle/name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == manifest['files'][name]
            cfg['files'][name] = base64.b64encode(data).decode()
    if not args.execute:
        print(json.dumps({k:v for k,v in cfg.items() if k != 'files'}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    for node in (15,16,17,18):
        container = 'ray-head' if node == 15 else 'ray-worker'
        result = subprocess.run(['ssh','-o','BatchMode=yes',f'maik@192.168.1.{node}',
            f'podman exec -i {container} python3 -S -'],
            input=REMOTE.replace('CONFIG',repr(cfg),1).encode(),capture_output=True,timeout=30)
        (args.output/f'node{node}.log').write_bytes(result.stdout+result.stderr)
        print(node, result.returncode, result.stdout.decode(), result.stderr.decode(), flush=True)
        if result.returncode:
            raise RuntimeError('Stopped on failure; inspect partial state before retry')


if __name__ == '__main__':
    main()
