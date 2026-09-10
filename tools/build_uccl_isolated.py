"""Build a separate UCCL plugin on Node18; never load it or change serving files."""
import argparse,base64,hashlib,json,subprocess,uuid
from pathlib import Path
REMOTE=r'''
import base64,hashlib,io,json,pathlib,subprocess,tarfile
cfg=CONFIG
raw=base64.b64decode(cfg['archive']);assert hashlib.sha256(raw).hexdigest()==cfg['sha']
root=pathlib.Path('/tmp/strix-uccl-build-'+cfg['run_id']);root.mkdir()
with tarfile.open(fileobj=io.BytesIO(raw)) as ar:
 for m in ar:
  assert m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts
  target=root/m.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(ar.extractfile(m).read())
p=subprocess.run(['timeout','--signal=TERM','--kill-after=5s','150s','bash',str(root/'build-plugin.sh')],capture_output=True,timeout=160)
log=p.stdout+p.stderr;root.joinpath('build.log').write_bytes(log)
lib=root/'build/librccl-net-uccl.so'
print(json.dumps({'run_id':cfg['run_id'],'root':str(root),'exit_code':p.returncode,'log':log.decode(errors='replace'),'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest() if lib.exists() else None,'loaded':False}),flush=True)
raise SystemExit(p.returncode)
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 raw=(a.inputs/'source.tar.gz').read_bytes();m=json.loads((a.inputs/'manifest.json').read_text());sha=hashlib.sha256(raw).hexdigest();assert sha==m['archive_sha256']
 cfg={'run_id':uuid.uuid4().hex,'sha':sha,'archive':base64.b64encode(raw).decode()}
 if not a.execute:print(json.dumps({'run_id':cfg['run_id'],'archive_sha256':sha,'loaded':False}));return
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'manifest.json').write_text(json.dumps({'run_id':cfg['run_id'],'inputs':m},indent=2))
 r=subprocess.run(['ssh','-o','BatchMode=yes','cluster-user@192.168.1.18','podman exec -i ray-worker python3 -S -'],input=REMOTE.replace('CONFIG',repr(cfg),1).encode(),capture_output=True,timeout=180)
 (a.output/'run.log').write_bytes(r.stdout+r.stderr)
 result=json.loads(r.stdout);(a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 raise SystemExit(r.returncode)
if __name__=='__main__':main()
