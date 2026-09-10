import subprocess,urllib.request,json,hashlib,argparse,shlex
from pathlib import Path
root=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--host',default='192.168.1.18',choices=['192.168.1.'+str(n) for n in range(15,19)]);a.add_argument('--seed',type=int,default=9567);a.add_argument('--library',required=True);a.add_argument('--sha256',required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
container='qwen029-tp2' if args.host.endswith(('.15','.16')) else 'qwen029-tp4'
def metrics():
 text=urllib.request.urlopen('http://192.168.1.15:8000/metrics',timeout=10).read().decode()
 return {line.split()[0]:float(line.split()[-1]) for line in text.splitlines() if line.startswith(('vllm:num_requests_running','vllm:num_requests_waiting','vllm:request_success_total'))}
before=metrics();assert not any(v for k,v in before.items() if 'num_requests_' in k),before
source=(root/'tests/check_hc_up_mix_constructor.py').read_bytes()
out=args.output;out.mkdir(parents=True,exist_ok=False)
(out/'before.json').write_text(json.dumps(before))
modules={name:(root/'patches'/name).read_text() for name in ('hc_up_mix_runtime.py','hc_up_mix_source.py')}
p=subprocess.run(['ssh','-o','BatchMode=yes','maik@'+args.host,f'podman exec -i {container} timeout --signal=TERM --kill-after=5s 150s python3 - --seed {args.seed} --library {shlex.quote(args.library)} --sha256 {shlex.quote(args.sha256)}'],input=('import pathlib,tempfile,sys\np=pathlib.Path(tempfile.mkdtemp(prefix="hc-up-mix-table-"))/"test.py"\ns='+repr(source.decode('utf-8-sig'))+'\np.write_text(s)\nsys.path.insert(0,str(p.parent))\nmodules='+repr(modules)+'\nfor name,body in modules.items(): (p.parent/name).write_text(body)\nimport os\nos.environ["STRIX_HC_UP_MIX_LIBRARY"]='+repr(args.library)+'\nexec(compile(s,str(p),"exec"),{"__name__":"__main__","__file__":str(p)})\n').encode(),capture_output=True,timeout=175)
(out/'run.log').write_bytes(p.stdout+p.stderr)
after=metrics();(out/'after.json').write_text(json.dumps(after))
print('host',args.host,'exit',p.returncode);print(p.stderr.decode(errors='replace')[-500:])
assert p.returncode==0,p.returncode
assert before==after,('service activity changed',before,after)
rows=[json.loads(s[7:]) for s in p.stdout.decode().splitlines() if s.startswith('RESULT=')];assert len(rows)==1
rows[0]['module_sha256']={name:hashlib.sha256(body.encode()).hexdigest() for name,body in modules.items()}
rows[0].update(host=args.host,container=container,benchmark_source_sha256=hashlib.sha256(source).hexdigest(),service_counters_unchanged=True)
(out/'result.json').write_text(json.dumps(rows[0],indent=2))
