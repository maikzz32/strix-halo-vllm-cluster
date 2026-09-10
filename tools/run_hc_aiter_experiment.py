#!/usr/bin/env python3
"""Dry-run by default; bounded isolated AITER HC test in the existing Node18 container."""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import subprocess
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import pathlib,subprocess,json,os
cfg=CONFIG
def owned(run_id):
 result=[]
 marker=('STRIX_AITER_HC_RUN_ID='+run_id).encode()
 for d in pathlib.Path('/proc').iterdir():
  if d.name.isdigit():
   try:
    if marker in d.joinpath('environ').read_bytes().split(bytes([0])):result.append(int(d.name))
   except (OSError,ProcessLookupError):pass
 return result
old=owned(cfg['prior_run_id']) if cfg['prior_run_id'] else []
if old:raise RuntimeError('previous owned experiment remains: '+repr(old))
root=pathlib.Path('/tmp/strix-aiter-hc-'+cfg['run_id']);root.mkdir()
for name,source in cfg['files'].items():
 root.joinpath(name).write_text(source)
env=dict(os.environ);env['STRIX_AITER_HC_RUN_ID']=cfg['run_id'];env['TRITON_CACHE_DIR']=str(root/'triton-cache')
p=subprocess.run(['timeout','--signal=TERM','--kill-after=5s','150s','python3',str(root/'bench_hc_aiter_cold.py'),'--output',str(root/'result.json')],env=env,capture_output=True,timeout=160)
root.joinpath('benchmark.log').write_bytes(p.stdout+p.stderr)
remaining=owned(cfg['run_id'])
record={'run_id':cfg['run_id'],'root':str(root),'exit_code':p.returncode,'previous_owned_processes':old,
 'remaining_owned_processes':remaining,'log':(p.stdout+p.stderr).decode(errors='replace'),
 'report':json.loads(root.joinpath('result.json').read_text()) if root.joinpath('result.json').exists() else None}
print('STRIX_AITER_RESULT='+json.dumps(record),flush=True)
raise SystemExit(p.returncode or bool(remaining))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-run-id', help='Optionally verify cleanup of a previous owned test')
    args = parser.parse_args()
    if args.prior_run_id and not re.fullmatch('[0-9a-f]{32}',args.prior_run_id):
        parser.error('--prior-run-id must be 32 lowercase hexadecimal characters')
    paths = [ROOT/'tests'/name for name in ('bench_hc_aiter_cold.py','hc_aiter_pr4814.py')]
    files = {p.name:p.read_text(encoding='utf-8') for p in paths}
    cfg = {'run_id':uuid.uuid4().hex, 'files':files, 'prior_run_id':args.prior_run_id}
    manifest = {'run_id':cfg['run_id'], 'source_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()}}
    if not args.execute:
        print(json.dumps(manifest, indent=2));return 0
    controller=ROOT/'tools/cluster.py'
    if not controller.exists():controller=ROOT/'scripts/native/cluster.py'
    spec=importlib.util.spec_from_file_location('aiter_cluster_metrics',controller)
    metrics_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics_module)
    def metrics():
        return urllib.request.urlopen('http://192.168.1.15:8000/metrics',timeout=5).read().decode()
    def success(text):
        return sum(float(line.split()[-1]) for line in text.splitlines() if line.startswith('vllm:request_success_total'))
    before=metrics();counts=metrics_module.request_counts(before)
    if any(counts.values()):raise RuntimeError('model must be idle')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest.update(before_request_counts=counts, before_success_total=success(before))
    args.output.joinpath('manifest.json').write_text(json.dumps(manifest,indent=2))
    args.output.joinpath('metrics-before.txt').write_text(before)
    process=subprocess.run(['ssh','-o','BatchMode=yes','cluster-user@192.168.1.18',
        'podman exec -i ray-worker python3 -S -'],input=REMOTE.replace('CONFIG',repr(cfg),1).encode(),
        capture_output=True,timeout=180)
    args.output.joinpath('run.log').write_bytes(process.stdout+process.stderr)
    rows=[json.loads(line.partition('=')[2]) for line in process.stdout.decode().splitlines()
          if line.startswith('STRIX_AITER_RESULT=')]
    if len(rows)!=1:raise RuntimeError('missing unique result; inspect run.log')
    record=rows[0];args.output.joinpath('result.json').write_text(json.dumps(record,indent=2))
    after=metrics();args.output.joinpath('metrics-after.txt').write_text(after)
    service={'before_success_total':success(before),'after_success_total':success(after),
             'unchanged':success(before)==success(after),'after_request_counts':metrics_module.request_counts(after)}
    args.output.joinpath('service-check.json').write_text(json.dumps(service,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k not in ('report','log')}))
    for row in (record.get('report') or {}).get('cases',[]):
        print(row['shape'],row['method'],row['graph_us_per_matrix'])
    ok=(process.returncode==0 and record['exit_code']==0 and record['run_id']==cfg['run_id']
        and not record['remaining_owned_processes'] and service['unchanged']
        and not any(service['after_request_counts'].values())
        and (record.get('report') or {}).get('status')=='passed')
    if not ok:print(record.get('log','')[-4000:])
    return int(not ok)


if __name__=='__main__':
    raise SystemExit(main())
