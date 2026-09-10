"""Compile a separate exact AVX512 UMA library on all four nodes; never activate it."""
import argparse,hashlib,json,subprocess,uuid
from pathlib import Path
from build_cpu_rdma_wide import generate
from run_cpu_rdma_phases import metrics
ROOT=Path(__file__).resolve().parents[1]
ADDON='/opt/strix-halo-next/hip-uma-avx512-20260907-r1'
REMOTE=r"""
import pathlib,subprocess,hashlib,json
cfg=CONFIG
root=pathlib.Path(cfg['addon']);root.mkdir(parents=True,exist_ok=False)
for name,s in cfg['sources'].items():root.joinpath(name).write_bytes(s.encode())
commands=[['gcc','-std=c11','-O3','-fPIC','-I.','-c','cpu_rdma_transport.c','-o','transport.o'],['/opt/rocm/bin/hipcc','-std=c++17','-O3','--offload-arch=gfx1151','-DSTRIX_UMA_THREADS=1024','-fPIC','-I.','-c','hip_uma_backend.hip','-o','uma.o'],['g++','-shared','uma.o','transport.o','-L/opt/rocm/lib','-Wl,-rpath,/opt/rocm/lib','-lamdhip64','-libverbs','-lpthread','-o','libhip_rdma_backend.so']]
for command in commands:
 r=subprocess.run(command,cwd=root,capture_output=True,timeout=30)
 if r.returncode:raise RuntimeError((command,r.returncode,r.stderr.decode()))
record={'addon':str(root),'commands':commands,'binary_sha256':hashlib.sha256(root.joinpath('libhip_rdma_backend.so').read_bytes()).hexdigest(),'sources_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in cfg['sources'].items()}}
root.joinpath('manifest.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
"""
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 files={name:(ROOT/'tools'/name).read_text(encoding='utf-8') for name in ['cpu_rdma_transport.c','cpu_rdma_transport.h','cpu_rdma_reduce_avx2.h','hip_uma_backend.hip']}
 files['cpu_rdma_transport.c']=generate(files['cpu_rdma_transport.c']);files['cpu_rdma_reduce_avx512.h']=(ROOT/'tests/cpu_rdma_reduce_avx512.h').read_text(encoding='utf-8')
 cfg={'addon':ADDON,'sources':files}
 if not a.execute:print(json.dumps({'addon':ADDON,'sources_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()}},indent=2));return
 before=metrics();a.output.mkdir(parents=True,exist_ok=False);hashes=[]
 for rank,node in enumerate((15,16,17,18)):
  c='ray-head' if rank==0 else 'ray-worker'
  r=subprocess.run(['ssh','-o','BatchMode=yes',f'cluster-user@192.168.1.{node}',f'podman exec -i {c} timeout --signal=TERM --kill-after=3s 100s python3 -S -'],input=REMOTE.replace('CONFIG',repr(cfg),1).encode(),capture_output=True,timeout=110)
  a.output.joinpath(f'node{node}.log').write_bytes(r.stdout+r.stderr)
  if r.returncode:raise RuntimeError(f'Staging failed on {node}; inspect log, no activation performed')
  record=json.loads(r.stdout);hashes.append(record['binary_sha256']);print(node,record['binary_sha256'],flush=True)
 after=metrics();a.output.joinpath('service-check.json').write_text(json.dumps({'before':before,'after':after},indent=2));assert before['success']==after['success'] and len(set(hashes))==1
if __name__=='__main__':main()
