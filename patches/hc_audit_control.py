"""Local-file control for bounded HC diagnostic snapshots.

The caller must ensure the serving engine is idle before reset/collect and
validate unchanged serving counters afterwards. Synchronization alone does not
exclude a new serving request. No network endpoint or arbitrary output path.
"""
import hashlib,json,os,re,threading,time
from pathlib import Path
import torch
_THREAD=None

def start(layers):
    global _THREAD
    directory=os.environ.get('STRIX_HC_AUDIT_DIR')
    if not directory or _THREAD is not None:return
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
    def loop():
        request=root/('request-'+str(os.getpid())+'.json')
        while True:
            if not request.exists():time.sleep(.2);continue
            claim=request.with_suffix('.processing')
            try:request.replace(claim)
            except FileNotFoundError:continue
            nonce=None
            try:
                body=json.loads(claim.read_text())
                nonce=body['nonce']
                if not isinstance(nonce,str) or not re.fullmatch('[0-9a-f]{32}',nonce):
                    raise ValueError('invalid nonce')
                if set(body)!={'nonce','action'} or body['action'] not in ('reset','collect'):
                    raise ValueError('invalid action')
                out=root/(str(os.getpid())+'-'+nonce)
                out.mkdir(exist_ok=False)
                items=sorted(layers.items())
                if not 0<len(items)<=100:raise ValueError('invalid layer count')
                devices={m._strix_audit_state.device for _,m in items}
                for d in devices:torch.cuda.synchronize(d)
                records=[]
                for i,(prefix,m) in enumerate(items):
                    state=m._strix_audit_state
                    counts=state.cpu().tolist()
                    record={'prefix':prefix,'state':counts}
                    if body['action']=='reset':state.zero_()
                    elif counts[3]:
                        data=m._strix_audit_snapshot.detach().cpu()
                        path=out/('snapshot-'+str(i)+'.pt')
                        torch.save(data,path)
                        record.update(snapshot=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    records.append(record)
                for d in devices:torch.cuda.synchronize(d)
                report=dict(status='ok',pid=os.getpid(),nonce=nonce,action=body['action'],layers=records)
            except Exception as e:
                report=dict(status='error',pid=os.getpid(),error=type(e).__name__+': '+str(e))
            if nonce is not None and isinstance(nonce,str) and re.fullmatch('[0-9a-f]{32}',nonce):
                ack=root/('ack-'+str(os.getpid())+'-'+nonce+'.json')
                temp=ack.with_suffix('.pending');temp.write_text(json.dumps(report,indent=2));temp.replace(ack)
            claim.unlink(missing_ok=True)
    _THREAD=threading.Thread(target=loop,name='hc-audit-control',daemon=True)
    _THREAD.start()
