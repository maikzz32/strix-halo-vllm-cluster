import subprocess,urllib.request,json,hashlib,argparse
from pathlib import Path
root=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--host',default='192.168.1.18',choices=['192.168.1.'+str(n) for n in range(15,19)]);a.add_argument('--short-rows',type=int,default=3);a.add_argument('--long-rows',type=int,default=4);a.add_argument('--head-only',action='store_true');args=a.parse_args()
assert 1 <= args.short_rows < args.long_rows <= 8
container='qwen029-tp2' if args.host.endswith(('.15','.16')) else 'qwen029-tp4'
def metrics():
 text=urllib.request.urlopen('http://192.168.1.15:8000/metrics',timeout=10).read().decode()
 return {line.split()[0]:float(line.split()[-1]) for line in text.splitlines() if line.startswith(('vllm:num_requests_running','vllm:num_requests_waiting','vllm:request_success_total'))}
before=metrics();assert not any(v for k,v in before.items() if 'num_requests_' in k),before
source=(root/'tests/check_decode_batch_invariance.py').read_bytes()
out=root/(f'results/decode-batch-invariance-{args.short_rows}-{args.long_rows}-'+('head-' if args.head_only else '')+args.host+'-20260910');out.mkdir(exist_ok=False)
(out/'before.json').write_text(json.dumps(before))
p=subprocess.run(['ssh','-o','BatchMode=yes','cluster-user@'+args.host,f'podman exec -i {container} timeout --signal=TERM --kill-after=5s 150s python3 - --short-rows {args.short_rows} --long-rows {args.long_rows}'+(' --head-only' if args.head_only else '')],input=source,capture_output=True,timeout=175)
(out/'run.log').write_bytes(p.stdout+p.stderr)
after=metrics();(out/'after.json').write_text(json.dumps(after))
print('host',args.host,'exit',p.returncode);print(p.stderr.decode(errors='replace')[-500:])
assert p.returncode==0,p.returncode
assert before==after,('service activity changed',before,after)
rows=[json.loads(s[7:]) for s in p.stdout.decode().splitlines() if s.startswith('RESULT=')];assert len(rows)==1
rows[0].update(host=args.host,container=container,benchmark_source_sha256=hashlib.sha256(source).hexdigest(),service_counters_unchanged=True)
(out/'result.json').write_text(json.dumps(rows[0],indent=2))
