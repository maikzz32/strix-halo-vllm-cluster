"""Back up/promote/restore the exact W2-only configuration on the four hosts."""
import argparse,base64,hashlib,json,subprocess
from pathlib import Path
OLD='b7f78a76b557e81e86271027f4339a74c9759cb5b700faf3c2a35ee8df038546'
REMOTE=r'''
import pathlib,hashlib,base64,json
c=CONFIG;p=pathlib.Path('/home/maik/strix-halo-next/config/cluster.json');backup=p.with_name('cluster.pre-w2-serial-20260907.json')
sha=lambda b:hashlib.sha256(b).hexdigest()
current=p.read_bytes();new=base64.b64decode(c['new']);assert sha(new)==c['new_sha']
if c['action']=='backup':
 assert sha(current)==c['old_sha']
 if backup.exists():assert backup.read_bytes()==current
 else:backup.write_bytes(current)
else:
 old=backup.read_bytes();assert sha(old)==c['old_sha'];assert sha(current) in (c['old_sha'],c['new_sha'])
 x=json.loads(new);assert x['environment'].pop('STRIX_MOE_W2_SERIAL')=='1';assert x==json.loads(old)
 data=new if c['action']=='promote' else old
 tmp=p.with_name('cluster.w2-pending.json');tmp.write_bytes(data);tmp.replace(p)
print(json.dumps({'action':c['action'],'canonical_sha256':sha(p.read_bytes()),'backup_sha256':sha(backup.read_bytes())}))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['backup','promote','restore']);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 b=a.config.read_bytes().replace(b'\r\n',b'\n');c={'action':a.action,'old_sha':OLD,'new_sha':hashlib.sha256(b).hexdigest(),'new':base64.b64encode(b).decode()}
 if not a.execute:print(json.dumps({k:v for k,v in c.items() if k!='new'}));return
 a.output.mkdir(parents=True,exist_ok=False)
 for node in (15,16,17,18):
  r=subprocess.run(['ssh','-o','BatchMode=yes',f'maik@192.168.1.{node}','python3 -S -'],input=REMOTE.replace('CONFIG',repr(c),1).encode(),capture_output=True,timeout=20)
  (a.output/f'node{node}.log').write_bytes(r.stdout+r.stderr);print(node,r.stdout.decode(),r.stderr.decode());assert r.returncode==0
if __name__=='__main__':main()
