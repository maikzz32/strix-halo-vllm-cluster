"""Read-only proof of a live UMA run: worker maps, library hash and startup parity."""
import argparse,concurrent.futures,json,re,subprocess
from pathlib import Path
REMOTE=r"""
import pathlib,json,hashlib
cfg=CONFIG;workers=[]
for d in pathlib.Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  cmd=d.joinpath('cmdline').read_bytes()
  if b'VLLM::Worker' not in cmd:continue
  env=dict(x.split(b'=',1) for x in d.joinpath('environ').read_bytes().split(bytes([0])) if b'=' in x)
  if env.get(b'STRIX_CLUSTER_RUN_ID',b'').decode()!=cfg['run_id']:continue
  maps=d.joinpath('maps').read_text()
  paths={x.split(maxsplit=5)[-1] for x in maps.splitlines() if len(x.split(maxsplit=5))==6}
  assert cfg['library'] in paths
  assert env.get(b'STRIX_HIP_RDMA_ENABLE')==b'1' and env.get(b'STRIX_HIP_RDMA_LIBRARY',b'').decode()==cfg['library']
  workers.append({'pid':int(d.name),'library_mapped':cfg['library'],'w1_flag':env.get(b'STRIX_MOE_W1_BN32',b'0').decode(),'w2_flag':env.get(b'STRIX_MOE_W2_SERIAL',b'0').decode()})
 except FileNotFoundError:continue
assert len(workers)==1 and workers[0]['w1_flag']=='0'
if cfg['w2_flag'] is not None:assert workers[0]['w2_flag']==cfg['w2_flag']
moe_sha=hashlib.sha256(pathlib.Path('/usr/local/lib64/python3.12/site-packages/vllm/model_executor/layers/fused_moe/fused_moe.py').read_bytes()).hexdigest()
if cfg['moe_sha'] is not None:assert moe_sha==cfg['moe_sha']
sha=hashlib.sha256(pathlib.Path(cfg['library']).read_bytes()).hexdigest();assert sha==cfg['sha']
logs=list(pathlib.Path('/home/maik/strix-halo-next/logs').glob('*'+cfg['run_id']+'-rank'+str(cfg['rank'])+'.log'));assert len(logs)==1
raw=logs[0].read_bytes();lines=raw.decode(errors='replace').splitlines()
proofs=[json.loads(x[x.index('{"event": "parity_proof"'):]) for x in lines if '{"event": "parity_proof"' in x];assert len(proofs)==1
proof=proofs[0]['proof'];assert proofs[0]['rank']==cfg['rank']
assert proof['banks']==32 and proof['exact_eager'] and proof['exact_graph'] and proof['validated_against_active_pynccl']
assert len(proof['rank_results'])==4 and {r['rank'] for r in proof['rank_results']}==set(range(4)) and all(not r['errors'] for r in proof['rank_results'])
assert any('STRIX_HIP_RDMA_ENABLED rank='+str(cfg['rank']) in x for x in lines)
print(json.dumps({'rank':cfg['rank'],'run_id':cfg['run_id'],'workers':workers,'installed_moe_source_sha256':moe_sha,'library_sha256':sha,'startup_log_snapshot_sha256':hashlib.sha256(raw).hexdigest(),'parity_proof':proof}))
"""
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--library',required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--moe-source-sha');p.add_argument('--w2-flag',choices=['0','1']);a=p.parse_args()
 assert re.fullmatch('[0-9a-f]{32}',a.run_id) and re.fullmatch('[0-9a-f]{64}',a.sha256)
 assert a.moe_source_sha is None or re.fullmatch('[0-9a-f]{64}',a.moe_source_sha)
 a.output.mkdir(parents=True,exist_ok=False)
 def one(rank):
  node=rank+15;c='ray-head' if rank==0 else 'ray-worker';cfg={'rank':rank,'run_id':a.run_id,'library':a.library,'sha':a.sha256,'moe_sha':a.moe_source_sha,'w2_flag':a.w2_flag}
  r=subprocess.run(['ssh','-o','BatchMode=yes',f'maik@192.168.1.{node}',f'podman exec -i {c} python3 -S -'],input=REMOTE.replace('CONFIG',repr(cfg),1).encode(),capture_output=True,timeout=20)
  a.output.joinpath(f'rank{rank}.log').write_bytes(r.stdout+r.stderr)
  if r.returncode:raise RuntimeError(f'Rank{rank} evidence failed; inspect log')
  row=json.loads(r.stdout);print(rank,row['workers'][0]['pid'],'mapped and parity verified',flush=True);return row
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(one,r) for r in range(4)];records=[];errors=[]
  for f in futures:
   try:records.append(f.result())
   except Exception as e:errors.append(str(e))
  a.output.joinpath('result.json').write_text(json.dumps({'ranks':records,'errors':errors},indent=2))
  if errors:raise RuntimeError(errors)
if __name__=='__main__':main()
