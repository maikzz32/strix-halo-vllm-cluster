"""Bounded CPU-only four-rank AVX512 network comparison; serving must be idle."""
import argparse,concurrent.futures,hashlib,json,subprocess,urllib.request,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE_SHA='b44cc2ff7f1f6dfb72126499a985057446a586f5b4df6afd27e4619248f70fde'
from run_cpu_rdma_phases import sources as phase_sources
def sources():
 files=phase_sources();s=files['transport_profile.c']
 anchor='#include "cpu_rdma_reduce_avx2.h"'
 addition='\n#include "cpu_rdma_reduce_avx512.h"\nstatic int wide_enabled=0;\nstatic void reduce_variant(const uint16_t *a[4],uint16_t *out,size_t count,int mode){if(wide_enabled)reduce_avx512_impl(a,out,count,mode);else reduce_avx2(a,out,count,mode);}\n'
 assert s.count(anchor)==1;s=s.replace(anchor,anchor+addition)
 start=s.index('int cpu_rdma_reduce_host(');end=s.index('static int wait_fd(',start)
 s=s[:start]+s[start:end].replace('reduce_avx2(', 'reduce_variant(')+s[end:]
 files['transport_profile.c']=s
 files['cpu_rdma_reduce_avx512.h']=(ROOT/'tests/cpu_rdma_reduce_avx512.h').read_text()
 files['bench.c']=(ROOT/'tests/bench_cpu_rdma_widenet.c').read_text()
 return files
REMOTE=r"""
import pathlib,subprocess,json,os
cfg=CONFIG
root=pathlib.Path('/tmp/rdma-widenet-'+cfg['run_id'])
if cfg['action']=='build':
 root.mkdir()
 for name,s in cfg['sources'].items():root.joinpath(name).write_text(s)
 command=['gcc','-std=c11','-O3','bench.c','-libverbs','-lm','-o','bench']
else:command=['timeout','--signal=TERM','--kill-after=3s','20s',str(root/'bench'),str(cfg['rank']),cfg['run_id'],cfg['hca']]
env=dict(os.environ);env['STRIX_RDMA_WIDENET_RUN_ID']=cfg['run_id']
r=subprocess.run(command,cwd=root,env=env,capture_output=True,timeout=25)
remaining=[]
marker=('STRIX_RDMA_WIDENET_RUN_ID='+cfg['run_id']).encode()
for d in pathlib.Path('/proc').iterdir():
 if d.name.isdigit():
  try:
   if marker in d.joinpath('environ').read_bytes().split(bytes([0])):remaining.append(int(d.name))
  except OSError:pass
print(json.dumps({'exit_code':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode(),'remaining':remaining}))
"""
def metrics():
 text=urllib.request.urlopen('http://192.168.1.15:8000/metrics',timeout=5).read().decode()
 gauges=[x for x in text.splitlines() if x.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
 assert len(gauges)==2 and all(float(x.split()[-1])==0 for x in gauges)
 return {'gauges':gauges,'success':sum(float(x.split()[-1]) for x in text.splitlines() if x.startswith('vllm:request_success_total'))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 files=sources();run_id=uuid.uuid4().hex;manifest={'run_id':run_id,'base_sha256':BASE_SHA,'sources_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()}}
 if not a.execute:print(json.dumps(manifest,indent=2));return
 before=metrics();a.output.mkdir(parents=True,exist_ok=False);a.output.joinpath('manifest.json').write_text(json.dumps(manifest,indent=2))
 for action in ('build','run'):
  def one(rank):
   node=rank+15;container='ray-head' if rank==0 else 'ray-worker'
   cfg={'action':action,'run_id':run_id,'rank':rank,'hca':'rocep197s0f1' if rank==3 else 'rocep197s0f3','sources':files if action=='build' else {}}
   r=subprocess.run(['ssh','-o','BatchMode=yes',f'cluster-user@192.168.1.{node}',f'podman exec -i {container} python3 -S -'],input=REMOTE.replace('CONFIG',repr(cfg),1).encode(),capture_output=True,timeout=35)
   a.output.joinpath(f'{action}-rank{rank}.json').write_bytes(r.stdout);record=json.loads(r.stdout)
   assert r.returncode==0 and record['exit_code']==0 and not record['remaining'],record
   return rank
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   futures=[pool.submit(one,r) for r in range(4)];errors=[]
   for f in futures:
    try:print(action,f.result())
    except Exception as e:errors.append(str(e))
   if errors:raise RuntimeError(errors)
 after=metrics();assert before['success']==after['success']
 a.output.joinpath('service-check.json').write_text(json.dumps({'before':before,'after':after},indent=2))
 print('PASS',run_id)
if __name__=='__main__':main()
