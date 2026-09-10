"""Bounded isolated AITER import probe in Node18; no installation or operator launches."""
import argparse,hashlib,json,subprocess,urllib.request,uuid
from pathlib import Path
REMOTE=r'''
import pathlib,hashlib,tarfile,subprocess,json,os
c=CONFIG;root=pathlib.Path('/tmp/strix-aiter-probe-'+c['run_id']);root.mkdir()
archive=pathlib.Path(c['archive']);assert hashlib.sha256(archive.read_bytes()).hexdigest()==c['sha']
with tarfile.open(archive) as t:
 for m in t.getmembers():
  assert m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts
 t.extractall(root,filter='data')
env=dict(os.environ);env.update(STRIX_AITER_PROBE_ID=c['run_id'],AITER_TRITON_ONLY='1',PYTHONPATH=str(root),TRITON_CACHE_DIR=str(root/'cache'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
code="""
import importlib,json,aiter
EXPORT_DTYPES
names=['aiter.ops.triton.causal_conv1d_update_single_token','aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr','aiter.ops.triton.quant']
rows=[]
for name in names:
 try:
  m=importlib.import_module(name);rows.append({'module':name,'path':m.__file__,'ok':True})
 except Exception as e:rows.append({'module':name,'ok':False,'error':repr(e)})
print('IMPORTS='+json.dumps({'aiter_path':aiter.__file__,'rows':rows}),flush=True)
raise SystemExit(any(not r['ok'] for r in rows))
"""
code=code.replace('EXPORT_DTYPES','import sys,pathlib\nsys.path.insert(0,str(pathlib.Path(aiter.__file__).parent / "jit" / "utils"))\nfrom aiter.utility import dtypes\naiter.dtypes=dtypes' if c['export_dtypes'] else '')
p=subprocess.run(['timeout','--signal=TERM','--kill-after=5s','60s','python3','-c',code],env=env,capture_output=True,timeout=70)
remaining=[]
for d in pathlib.Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  if ('STRIX_AITER_PROBE_ID='+c['run_id']).encode() in d.joinpath('environ').read_bytes().split(bytes([0])):remaining.append(int(d.name))
 except OSError:pass
print(json.dumps({'run_id':c['run_id'],'root':str(root),'exit_code':p.returncode,'remaining_owned_processes':remaining,'log':(p.stdout+p.stderr).decode(errors='replace')}))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--export-dtypes',action='store_true');a=p.parse_args()
 raw=(a.inputs/'sources.tar.gz').read_bytes();m=json.loads((a.inputs/'manifest.json').read_text());sha=hashlib.sha256(raw).hexdigest();assert sha==m['archive_sha256']
 run=uuid.uuid4().hex;cfg={'run_id':run,'sha':sha,'archive':f'/home/cluster-user/strix-halo-next/aiter-probe-{run}.tar.gz','export_dtypes':a.export_dtypes}
 if not a.execute:print(json.dumps(cfg));return
 def metrics():return urllib.request.urlopen('http://192.168.1.15:8000/metrics',timeout=10).read().decode()
 import sys
 root=Path(__file__).resolve().parents[1]
 sys.path.insert(0,str(root/'tools' if (root/'tools/cluster.py').exists() else root/'scripts/native'));from cluster import request_counts
 before=metrics();assert not any(request_counts(before).values())
 a.output.mkdir(parents=True,exist_ok=False)
 subprocess.run(['scp','-q',str(a.inputs/'sources.tar.gz'),'cluster-user@192.168.1.18:'+cfg['archive']],check=True,timeout=20)
 r=subprocess.run(['ssh','-o','BatchMode=yes','cluster-user@192.168.1.18','podman exec -i ray-worker python3 -S -'],input=REMOTE.replace('CONFIG',repr(cfg),1).encode(),capture_output=True,timeout=90)
 (a.output/'run.log').write_bytes(r.stdout+r.stderr);assert r.returncode==0
 result=json.loads(r.stdout);result['configuration']=cfg;result['upstream_commit']=m['commit'];after=metrics()
 success=lambda s:sum(float(x.split()[-1]) for x in s.splitlines() if x.startswith('vllm:request_success_total{'))
 result['service']={'before':success(before),'after':success(after),'counts':request_counts(after)}
 (a.output/'result.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
 assert success(before)==success(after) and not any(request_counts(after).values()) and not result['remaining_owned_processes']
 print(json.dumps(result));raise SystemExit(result['exit_code']!=0)
if __name__=='__main__':main()
