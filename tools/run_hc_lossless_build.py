"""Compile only, no GPU execution. Isolated Node18 container directory per run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid
from build_hc_lossless_source import generate

REMOTE = '''import pathlib,subprocess,json,hashlib
cfg=CONFIG
p=pathlib.Path(cfg['root']);p.mkdir();p.joinpath('native.hip').write_text(cfg['source'])
c=['/opt/rocm/bin/hipcc','-std=c++17','-O3','-save-temps','--offload-arch=gfx1151','-fPIC','-shared','native.hip','-o','native.so']
r=subprocess.run(c,cwd=p,capture_output=True,text=True,timeout=90)
print(json.dumps({'root':str(p),'command':c,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,
 'binary_sha256':hashlib.sha256(p.joinpath('native.so').read_bytes()).hexdigest() if r.returncode==0 else None}),flush=True)
raise SystemExit(r.returncode)
'''

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    source=generate(a.source)
    cfg={'root':'/tmp/hc-lossless-'+uuid.uuid4().hex,'source':source}
    if not a.execute:
        print(json.dumps({'root':cfg['root'],'source_sha256':hashlib.sha256(source.encode()).hexdigest()}))
        raise SystemExit(0)
    a.output.mkdir(parents=True,exist_ok=False)
    a.output.joinpath('native.hip').write_bytes(source.encode())
    r=subprocess.run(['ssh','-o','BatchMode=yes','maik@192.168.1.18',
        'podman exec -i ray-worker timeout --signal=TERM --kill-after=5s 110s python3 -S -'],
        input=REMOTE.replace('CONFIG',repr(cfg),1).encode(),capture_output=True,timeout=125)
    a.output.joinpath('build.log').write_bytes(r.stdout+r.stderr)
    print(r.stdout.decode(),r.stderr.decode())
    raise SystemExit(r.returncode)
