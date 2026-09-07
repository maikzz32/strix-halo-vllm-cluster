import subprocess,json,concurrent.futures,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--run-id',required=True);args=p.parse_args()
code=r"""
import pathlib,json,hashlib
root=pathlib.Path('/usr/local/lib64/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn')
names=['qwen_gdn_linear_attn.py','_strix_gdn.py','_strix_gdn_kernel.py']
files={n:hashlib.sha256(root.joinpath(n).read_bytes()).hexdigest() for n in names}
workers=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=p.joinpath('cmdline').read_bytes()
  if not cmd.startswith(b'VLLM::Worker'):continue
  env=dict(x.split(b'=',1) for x in p.joinpath('environ').read_bytes().split(bytes([0])) if b'=' in x)
  workers.append({'pid':int(p.name),'environment':{k:env.get(k.encode(),b'').decode() for k in ('STRIX_PACKED_GDN','STRIX_HIP_RDMA_LIBRARY','STRIX_MOE_W2_SERIAL','STRIX_CLUSTER_RUN_ID')}})
 except OSError:pass
print(json.dumps({'files':files,'workers':workers}))
"""
def run(node):
 c='ray-head' if node==15 else 'ray-worker'
 r=subprocess.run(['ssh','-o','BatchMode=yes',f'maik@192.168.1.{node}',f'podman exec -i {c} python3 -S -'],input=code.encode(),capture_output=True,timeout=15)
 assert r.returncode==0,r.stderr
 return str(node),json.loads(r.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:r=dict(ex.map(run,(15,16,17,18)))
args.output.write_text(json.dumps(r,indent=2))
for n,v in r.items():
 assert v['files']=={'qwen_gdn_linear_attn.py':'409969ebb81cc4b5adec548271fff60ecf6ead91adc0b7328f5450685c6646f5','_strix_gdn.py':'a064bdbab5d4e0022af40d7c09b86e3a0c4fd258befe31948f5e685f2395d2b2','_strix_gdn_kernel.py':'5c7ed927e90a5ac7014bc894f35bb3ddbdf619eee5cf2839f152e9c266c293dc'}
 assert len(v['workers'])==1
 env=v['workers'][0]['environment']
 assert env['STRIX_PACKED_GDN']=='1' and env['STRIX_MOE_W2_SERIAL']=='1'
 assert env['STRIX_CLUSTER_RUN_ID']==args.run_id
 assert env['STRIX_HIP_RDMA_LIBRARY']=='/opt/strix-halo-next/hip-uma-ring4-20260907-r1/libhip_rdma_backend.so'
print('All four GDN source/helper hashes, W2/GDN flags, AVX2 library path and run IDs match.')
