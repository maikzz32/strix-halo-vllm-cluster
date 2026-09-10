"""Back up/promote/restore the W1 expert scheduling configuration on the four hosts."""
import argparse,base64,hashlib,json,subprocess
from pathlib import Path
OLD='ce249995443ed3299c40c7dd6abdf06c8bf013593b16b662d9057b2a07b06366'
REMOTE=r'''
import pathlib,hashlib,base64,json
c=CONFIG;p=pathlib.Path('/home/cluster-user/strix-halo-next/config/cluster.json');backup=p.with_name('cluster.pre-w1-order-20260907.json')
sha=lambda b:hashlib.sha256(b).hexdigest()
current=p.read_bytes();new=base64.b64decode(c['new']);assert sha(new)==c['new_sha']
if c['action']=='backup':
 assert sha(current)==c['old_sha']
 if backup.exists():assert backup.read_bytes()==current
 else:backup.write_bytes(current)
else:
 old=backup.read_bytes();assert sha(old)==c['old_sha'];assert sha(current) in (c['old_sha'],c['new_sha'])
 x=json.loads(new);assert x['environment']['STRIX_HIP_RDMA_LIBRARY']=='/opt/strix-halo-next/hip-uma-avx512-20260907-r1/libhip_rdma_backend.so'
 assert x['environment'].pop('STRIX_MOE_W1_EXPERT_ORDER')=='1';assert x==json.loads(old)
 data=new if c['action']=='promote' else old
 tmp=p.with_name('cluster.w1-order-pending.json');tmp.write_bytes(data);tmp.replace(p)
print(json.dumps({'action':c['action'],'canonical_sha256':sha(p.read_bytes()),'backup_sha256':sha(backup.read_bytes())}))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['backup','promote','restore']);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 b=a.config.read_bytes().replace(b'\r\n',b'\n');c={'action':a.action,'old_sha':OLD,'new_sha':hashlib.sha256(b).hexdigest(),'new':base64.b64encode(b).decode()}
 assert c['new_sha']=='cfef32eb5d97b09f0ac8494a6484ba33e823cc84990de6e7480c2ffa190ee8e7'
 if not a.execute:print(json.dumps({k:v for k,v in c.items() if k!='new'}));return
 a.output.mkdir(parents=True,exist_ok=False)
 for node in (15,16,17,18):
  r=subprocess.run(['ssh','-o','BatchMode=yes',f'cluster-user@192.168.1.{node}','python3 -S -'],input=REMOTE.replace('CONFIG',repr(c),1).encode(),capture_output=True,timeout=20)
  (a.output/f'node{node}.log').write_bytes(r.stdout+r.stderr);print(node,r.stdout.decode(),r.stderr.decode());assert r.returncode==0
if __name__=='__main__':main()
