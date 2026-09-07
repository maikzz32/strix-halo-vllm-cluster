#!/usr/bin/env python3
"""Stage, activate or restore the isolated, opt-in communicator experiment."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid
from patch_hip_rdma_communicator import patched

ROOT = Path(__file__).resolve().parents[1]
ADDON = '/opt/strix-halo-next/hip-rdma-ring4-20260907-r1'
UMA_ADDON = '/opt/strix-halo-next/hip-uma-ring4-20260907-r1'
BASE_SHA = '03ece681b0bef349cb78dabc74585efea2cc160ae28bd711c81dfd807ca8dff5'
RELATIVE = 'usr/local/lib64/python3.12/site-packages/vllm/distributed/device_communicators/cuda_communicator.py'
FILES = ['tools/cpu_rdma_transport.c', 'tools/cpu_rdma_transport.h', 'tools/cpu_rdma_reduce_avx2.h',
         'tools/hip_rdma_backend.cpp', 'tools/hip_rdma_backend.py', 'tools/validate_rccl_ring4.py',
         'tools/cpu_rdma_ring4_reference.py', 'tools/analyze_rccl_bf16_reference.py',
         'tools/cpu_rdma_reference.py', 'patches/strix_hip_rdma_hook.py']
REMOTE = r'''
import hashlib,json,pathlib,subprocess
cfg=CONFIG
root=pathlib.Path(cfg['addon'])
target=pathlib.Path('/'+cfg['relative'])
current=hashlib.sha256(target.read_bytes()).hexdigest()
if cfg['action']=='stage':
 if current != cfg['base_sha']: raise RuntimeError('unexpected communicator source')
 root.mkdir(parents=True,exist_ok=False)
 for name,source in cfg['sources'].items():
  if pathlib.Path(name).name != name: raise ValueError('invalid source filename')
  root.joinpath(name).write_bytes(source.encode())
 root.joinpath('cuda_communicator.original.py').write_bytes(target.read_bytes())
 root.joinpath('cuda_communicator.patched.py').write_bytes(cfg['patched'].encode())
 for path in root.glob('*.py'): compile(path.read_bytes(),str(path),'exec')
 commands=[['gcc','-std=c11','-O3','-fPIC','-Wall','-Wextra','-Werror','-I.','-c','cpu_rdma_transport.c','-o','transport.o'],
 ['g++','-std=c++17','-O2','-fPIC','-shared','-Wall','-Wextra','-Werror','-D__HIP_PLATFORM_AMD__','-I.','-I/opt/rocm/include','hip_rdma_backend.cpp','transport.o','-L/opt/rocm/lib','-Wl,-rpath,/opt/rocm/lib','-lamdhip64','-libverbs','-lpthread','-lm','-o','libhip_rdma_backend.so']]
 if cfg['backend']=='uma':
  commands=commands[:1]+[
   ['/opt/rocm/bin/hipcc','-std=c++17','-O3','--offload-arch=gfx1151','-DSTRIX_UMA_THREADS=1024','-fPIC','-I.','-c','hip_uma_backend.hip','-o','uma.o'],
   ['g++','-shared','uma.o','transport.o','-L/opt/rocm/lib','-Wl,-rpath,/opt/rocm/lib','-lamdhip64','-libverbs','-lpthread','-o','libhip_rdma_backend.so']]
 for command in commands:
  p=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=25)
  print(json.dumps({'command':command,'exit_code':p.returncode,'stderr':p.stderr}),flush=True)
  if p.returncode: raise RuntimeError('build failed')
 record={'backend':cfg['backend'],'base_sha256':current,'patched_sha256':hashlib.sha256(cfg['patched'].encode()).hexdigest(),
         'binary_sha256':hashlib.sha256(root.joinpath('libhip_rdma_backend.so').read_bytes()).hexdigest(),
         'sources_sha256':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in cfg['sources'].items()}}
 root.joinpath('manifest.json').write_text(json.dumps(record,indent=2))
 print(json.dumps({'event':'staged',**record}),flush=True)
else:
 record=json.loads(root.joinpath('manifest.json').read_text())
 if current not in (record['base_sha256'],record['patched_sha256']): raise RuntimeError('foreign source modification')
 selected='cuda_communicator.patched.py' if cfg['action']=='activate' else 'cuda_communicator.original.py'
 data=root.joinpath(selected).read_bytes()
 expected=record['patched_sha256'] if cfg['action']=='activate' else record['base_sha256']
 if hashlib.sha256(data).hexdigest()!=expected: raise RuntimeError('staged source identity changed')
 if cfg['action']=='activate':
  hook=root.joinpath('strix_hip_rdma_hook.py').read_bytes()
  if hashlib.sha256(hook).hexdigest()!=record['sources_sha256']['strix_hip_rdma_hook.py']: raise RuntimeError('hook changed')
  helper=target.with_name('strix_hip_rdma_hook.py')
  temporary=helper.with_suffix('.pending');temporary.write_bytes(hook);temporary.replace(helper)
 temporary=target.with_suffix('.pending');temporary.write_bytes(data);temporary.replace(target)
 print(json.dumps({'event':cfg['action'],'source_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}),flush=True)
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['stage','activate','restore'])
    p.add_argument('--execute',action='store_true')
    p.add_argument('--backend',choices=['callback','uma'],default='callback')
    p.add_argument('--base-source',type=Path,
                   help='Reviewed original cuda_communicator.py; required for staging outside the research workspace')
    args=p.parse_args()
    addon=UMA_ADDON if args.backend=='uma' else ADDON
    cfg={'action':args.action,'addon':addon,'relative':RELATIVE,'base_sha':BASE_SHA,'backend':args.backend}
    if args.action=='stage':
        base_path=args.base_source or ROOT/'snapshots/runtime-source'/RELATIVE
        if not base_path.is_file():
            p.error('stage requires --base-source pointing to the reviewed original cuda_communicator.py')
        base=base_path.read_bytes()
        files=[('tools/hip_uma_backend.hip' if args.backend=='uma' and path=='tools/hip_rdma_backend.cpp' else path) for path in FILES]
        cfg.update(sources={Path(path).name:(ROOT/path).read_text() for path in files},
                   patched=patched(base,BASE_SHA).decode())
    if not args.execute:
        print(json.dumps({'action':args.action,'addon':addon,'base_sha':BASE_SHA,'source_files':list(cfg.get('sources',{}))}));return 0
    result_dir=ROOT/'results'/('hip-rdma-deploy-'+args.action+'-20260907-'+uuid.uuid4().hex[:8])
    result_dir.mkdir(parents=True,exist_ok=False)
    def run(node):
        container='ray-head' if node==15 else 'ray-worker'
        process=subprocess.run(['ssh','-o','BatchMode=yes',f'maik@192.168.1.{node}',
            f'podman exec -i {container} timeout --signal=TERM --kill-after=5s 65s python3 -S -'],
            input=REMOTE.replace('CONFIG',repr(cfg),1).encode(),capture_output=True,timeout=75)
        result_dir.joinpath(f'node{node}.log').write_bytes(process.stdout+process.stderr)
        print(node,process.returncode,process.stdout.decode()[-650:],process.stderr.decode(),flush=True)
        return {'node':node,'exit_code':process.returncode}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(run,[15,16,17,18]))
    result_dir.joinpath('status.json').write_text(json.dumps(results,indent=2))
    return int(any(r['exit_code'] for r in results))


if __name__=='__main__':
    sys.exit(main())
