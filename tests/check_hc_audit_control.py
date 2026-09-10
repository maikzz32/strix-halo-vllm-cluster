"""Validate local capture transfer, first-error data and explicit reset."""
import hashlib,json,os,pathlib,tempfile,time,types,uuid
import torch
from hc_first_mismatch import inspect,SNAPSHOT_ELEMENTS
from hc_audit_control import start
root=pathlib.Path(tempfile.mkdtemp(prefix='hc-control-test-'))
os.environ['STRIX_HC_AUDIT_DIR']=str(root)
torch.set_num_threads(1);torch.manual_seed(9574)
x=torch.randn(4,320,device='cuda',dtype=torch.bfloat16)
xn=torch.randn(4,10240,device='cuda',dtype=torch.bfloat16)
g=torch.randn(4,10240,device='cuda',dtype=torch.bfloat16);ng=g.clone()
y=torch.randn(4,2560,device='cuda',dtype=torch.bfloat16);ny=y.clone();ny[0,0]+=1
m=types.SimpleNamespace(_strix_audit_state=torch.zeros(4,device='cuda',dtype=torch.int32),
 _strix_audit_snapshot=torch.empty(SNAPSHOT_ELEMENTS,device='cuda',dtype=torch.bfloat16))
inspect(x,xn,g,ng,y,ny,m._strix_audit_state,m._strix_audit_snapshot)
start({'test.layer':m})
def request(action):
    nonce=uuid.uuid4().hex
    p=root/('request-'+str(os.getpid())+'.json');tmp=p.with_suffix('.pending')
    tmp.write_text(json.dumps(dict(nonce=nonce,action=action)));tmp.replace(p)
    ack=root/('ack-'+str(os.getpid())+'-'+nonce+'.json')
    deadline=time.monotonic()+10
    while not ack.exists():
        assert time.monotonic()<deadline,'control timeout'
        time.sleep(.05)
    r=json.loads(ack.read_text());assert r['status']=='ok',r
    return r,root/(str(os.getpid())+'-'+nonce)
r,folder=request('collect');row=r['layers'][0]
assert row['state']==[1,0,1,1],row
p=folder/row['snapshot'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
actual=torch.load(p,weights_only=True)
expected=torch.cat([t.flatten() for t in (x,xn,g,ng,y,ny)]).cpu()
assert torch.equal(actual,expected)
request('reset');assert m._strix_audit_state.tolist()==[0,0,0,0]
r,_=request('collect');assert r['layers'][0]['state']==[0,0,0,0] and 'snapshot' not in r['layers'][0]
print('RESULT='+json.dumps(dict(status='passed',snapshot_exact=True,hash_exact=True,reset_exact=True,
    control_transport='local-files',network_listener=False)),flush=True)
