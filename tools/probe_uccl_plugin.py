"""Bounded UCCL init/device probe on Node18; never alters the model process."""
import argparse,hashlib,importlib.util,json,subprocess,urllib.request,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REMOTE=r'''
import hashlib,json,os,pathlib,subprocess
cfg=CONFIG;root=pathlib.Path(cfg['root']);lib=root/'build/librccl-net-uccl.so'
assert hashlib.sha256(lib.read_bytes()).hexdigest()==cfg['sha']
work=root/('probe-'+cfg['run_id']);work.mkdir();src=work/'probe.cpp';src.write_text(cfg['source'])
c=subprocess.run(['g++','-std=c++17','-D__HIP_PLATFORM_AMD__','-I'+str(root/'rccl/src/include'),'-I/opt/rocm/include',str(src),'-o',str(work/'probe'),'-ldl'],capture_output=True,timeout=30)
env=dict(os.environ);env.update(cfg['env']);env['STRIX_UCCL_PROBE_RUN_ID']=cfg['run_id']
if c.returncode==0:
 p=subprocess.run(['timeout','--signal=TERM','--kill-after=3s','25s',str(work/'probe'),str(lib)],env=env,capture_output=True,timeout=32)
else:p=c
remaining=[];marker=('STRIX_UCCL_PROBE_RUN_ID='+cfg['run_id']).encode()
for d in pathlib.Path('/proc').iterdir():
 if d.name.isdigit():
  try:
   if marker in d.joinpath('environ').read_bytes().split(bytes([0])):remaining.append(int(d.name))
  except OSError:pass
record={'run_id':cfg['run_id'],'library_sha256':cfg['sha'],'compile_exit':c.returncode,'exit_code':p.returncode,'log':(c.stdout+c.stderr+(p.stdout+p.stderr if p is not c else b'')).decode(errors='replace'),'remaining':remaining,'probe_dir':str(work)}
work.joinpath('result.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 cfg={'run_id':uuid.uuid4().hex,'root':'/tmp/strix-uccl-build-86ea2a5eebc8454b8f58084daa6dc5ae','sha':'fbeaff859ba8322921b91de3d31ecada7852ec1280b264239f2eeb23a73b3c39','source':(ROOT/'tests/probe_uccl_plugin.cpp').read_text(),'env':{'UCCL_IB_HCA':'rocep197s0f1','NCCL_IB_HCA':'rocep197s0f1','NCCL_IB_GID_INDEX':'1','UCCL_SOCKET_IFNAME':'enp197s0f1np1','NCCL_SOCKET_IFNAME':'enp197s0f1np1','UCCL_NUM_ENGINES':'1','UCCL_PORT_ENTROPY':'1','UCCL_RCMODE':'1','NCCL_NET_GDR_LEVEL':'0'}}
 if not a.execute:print(json.dumps({k:v for k,v in cfg.items() if k!='source'}));return
 spec=importlib.util.spec_from_file_location('cluster_metrics',ROOT/'tools/cluster.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 def metrics():return urllib.request.urlopen('http://192.168.1.15:8000/metrics',timeout=5).read().decode()
 def count(s):return sum(float(x.split()[-1]) for x in s.splitlines() if x.startswith('vllm:request_success_total{'))
 before=metrics();assert not any(mod.request_counts(before).values())
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'manifest.json').write_text(json.dumps({k:v for k,v in cfg.items() if k!='source'},indent=2));(a.output/'probe.cpp').write_text(cfg['source'])
 r=subprocess.run(['ssh','-o','BatchMode=yes','maik@192.168.1.18','podman exec -i ray-worker python3 -S -'],input=REMOTE.replace('CONFIG',repr(cfg),1).encode(),capture_output=True,timeout=75)
 (a.output/'run.log').write_bytes(r.stdout+r.stderr);record=json.loads(r.stdout)
 after=metrics();service={'before_success':count(before),'after_success':count(after),'after_requests':mod.request_counts(after)};assert count(before)==count(after) and not any(service['after_requests'].values())
 record['service_check']=service;(a.output/'result.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
 assert not record['remaining']
if __name__=='__main__':main()
