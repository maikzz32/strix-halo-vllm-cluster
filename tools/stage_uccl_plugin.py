"""Copy a verified plugin to a new separate directory on all ranks; no activation."""
import argparse,base64,hashlib,json,re,subprocess
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--build-result',type=Path,required=True);p.add_argument('--addon',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 r=json.loads(a.build_result.read_text());assert r['exit_code']==0 and re.fullmatch('/tmp/strix-uccl-build-[0-9a-f]{32}',r['root']) and re.fullmatch('uccl-[a-z0-9-]+',a.addon)
 dest='/opt/strix-halo-next/'+a.addon
 if not a.execute:print(json.dumps({'path':dest,'sha':r['library_sha256']}));return
 b=subprocess.run(['ssh','-o','BatchMode=yes','maik@192.168.1.18','podman exec ray-worker cat '+r['root']+'/build/librccl-net-uccl.so'],capture_output=True,check=True,timeout=20).stdout
 assert hashlib.sha256(b).hexdigest()==r['library_sha256'];a.output.mkdir(parents=True,exist_ok=False);(a.output/'librccl-net-uccl.so').write_bytes(b)
 script="import pathlib,base64,hashlib,json\nc=CONFIG\np=pathlib.Path(c['dest']);p.mkdir(exist_ok=False)\nb=base64.b64decode(c['b']);assert hashlib.sha256(b).hexdigest()==c['sha']\np.joinpath('librccl-net-uccl.so').write_bytes(b)\nprint(json.dumps({'sha':c['sha'],'path':str(p/'librccl-net-uccl.so')}))\n"
 for node in (15,16,17,18):
  c={'dest':dest,'sha':r['library_sha256'],'b':base64.b64encode(b).decode()};container='ray-head' if node==15 else 'ray-worker'
  q=subprocess.run(['ssh','-o','BatchMode=yes',f'maik@192.168.1.{node}',f'podman exec -i {container} python3 -S -'],input=script.replace('CONFIG',repr(c),1).encode(),capture_output=True,timeout=30)
  (a.output/f'node{node}.log').write_bytes(q.stdout+q.stderr);assert q.returncode==0;print(node,q.stdout.decode())
if __name__=='__main__':main()
