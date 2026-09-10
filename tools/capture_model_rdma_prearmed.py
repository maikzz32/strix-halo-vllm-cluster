"""Capture bounded per-peer timing during a verified worker's decode request."""
import argparse,re,base64,concurrent.futures,hashlib,json,subprocess,threading,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
RUN=args.run_id;assert re.fullmatch('[0-9a-f]{32}',RUN)
LIB='/opt/strix-halo-next/rdma-trace-20260910-r1/libhip_rdma_backend.so'
SHA='64c3190fc0392ae8b7d4fc04dd860ffedb962e3e20c1009fc22ac50ebc7a7b06'
OUT=args.output
NODES=json.loads(args.config.read_text())['nodes']
REMOTE=r'''import pathlib,json,hashlib,mmap,struct,base64
workers=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if b'VLLM::Worker' not in p.joinpath('cmdline').read_bytes():continue
  env=dict(x.split(b'=',1) for x in p.joinpath('environ').read_bytes().split(b'\0') if b'=' in x)
  if env.get(b'STRIX_CLUSTER_RUN_ID')!=RUN.encode():continue
  assert env.get(b'STRIX_HIP_RDMA_LIBRARY')==LIB.encode() and LIB in p.joinpath('maps').read_text()
  workers.append(p.name)
 except FileNotFoundError:pass
assert len(workers)==1,workers
assert hashlib.sha256(pathlib.Path(LIB).read_bytes()).hexdigest()==SHA
files=list((pathlib.Path(LIB).parent/'traces').glob('rdma-r*-p'+workers[0]+'.bin'))
assert len(files)==1,[str(p) for p in files]
p=files[0]
with p.open('r+b') as f,mmap.mmap(f.fileno(),0) as m:
 h=struct.unpack_from('<8Q',m)
 assert h[:2]==(0x31524354414d4452,1) and h[3]==2048 and len(m)==64+2048*128
 if ACTION=='arm':
  assert h[4]==h[5]==0
  struct.pack_into('<Q',m,32,2048)
 h=struct.unpack_from('<8Q',m)
 data=bytes(m) if ACTION=='collect' else None
out={'worker_pid':workers[0],'file':str(p),'header':h,'run_id':RUN,'library_sha256':SHA}
if data is not None:out.update(base64=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest())
print(json.dumps(out))
'''
def node(node,action):
 source='RUN='+repr(RUN)+'\nLIB='+repr(LIB)+'\nSHA='+repr(SHA)+'\nACTION='+repr(action)+'\n'+REMOTE
 r=subprocess.run(['ssh','cluster-user@'+node['host'],'podman exec -i '+node['container']+' python3 -S -'],input=source,text=True,capture_output=True,check=True,timeout=25)
 return node['host'],json.loads(r.stdout)
def all_nodes(action):
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:return dict(pool.map(lambda n:node(n,action),NODES))
def save(name,data):(OUT/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
BODY={'model':'/home/cluster-user/qwen38_rest','messages':[{'role':'user','content':'Write a detailed technical explanation of memory bandwidth, caches, and synchronization in distributed language model inference. Include several concrete examples.'}],'temperature':0,'max_tokens':2048,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
def request(first):
 req=urllib.request.Request('http://192.168.1.15:8000/v1/chat/completions',data=json.dumps(BODY).encode(),headers={'Content-Type':'application/json'})
 events=[];content='';reasoning='';usage=None
 with urllib.request.urlopen(req,timeout=180) as r:
  for raw in r:
   line=raw.decode().strip()
   if not line.startswith('data: ') or line=='data: [DONE]':continue
   e=json.loads(line[6:]);events.append(e)
   if e.get('usage'):usage=e['usage']
   for c in e.get('choices',[]):
    d=c.get('delta',{});content+=d.get('content') or '';reasoning+=d.get('reasoning') or ''
    if d.get('content') or d.get('reasoning'):first.set()
 assert usage and usage['completion_tokens']==2048
 return {'request':BODY,'content':content,'reasoning':reasoning,'usage':usage,'events':events}
def main():
 OUT.mkdir(exist_ok=False)
 deadline=time.monotonic()+900
 while True:
  try:
   with urllib.request.urlopen('http://192.168.1.15:8000/health',timeout=5) as r:
    if r.status==200:break
  except OSError:pass
  if time.monotonic()>deadline:raise TimeoutError('Inspect existing startup; do not restart')
  time.sleep(5)
 save('before',all_nodes('read'))
 print('Diagnostic workers verified; running untraced reference',flush=True)
 reference=request(threading.Event());save('reference',reference)
 armed=all_nodes('arm');save('armed',armed)
 verified=all_nodes('read');save('armed_verified',verified)
 assert sorted(row['header'][2] for row in verified.values())==[0,1,2,3]
 assert all(row['header'][4]==2048 and row['header'][5]==0 for row in verified.values()),'Activity occurred before all ranks were armed'
 print('All four maps armed and independently verified before request submission',flush=True)
 candidate=request(threading.Event())
 save('candidate',candidate)
 for k in ('request','content','reasoning','usage'):assert reference[k]==candidate[k],k
 collected=all_nodes('collect')
 import sys
 sys.path.insert(0,str(ROOT/'tools'))
 from read_rdma_trace import decode
 ranges=[]
 for host,r in collected.items():
  data=base64.b64decode(r.pop('base64'));assert hashlib.sha256(data).hexdigest()==r['sha256']
  (OUT/(host+'.bin')).write_bytes(data);decoded=decode(data);assert decoded['published']==2048
  save(host,decoded);ranges.append(set(x['sequence'] for x in decoded['rows']))
 common=sorted(set.intersection(*ranges));assert len(common)>=512,len(common)
 save('collection',collected);save('parity',{'identical_outputs':True,'common_sequences':len(common),'first_sequence':common[0],'last_sequence':common[-1]})
 print('MODEL_RDMA_TRACE_COMPLETE common_sequences='+str(len(common)),flush=True)
if __name__=='__main__':main()
