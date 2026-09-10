"""Run the launch-geometry diagnostic in an isolated temporary directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--host', default='192.168.1.18',
                    choices=['192.168.1.' + str(n) for n in range(15, 19)])
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--sha256', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
kernel = args.source.read_bytes()
assert hashlib.sha256(kernel).hexdigest() == args.sha256
test = (root / 'tests/check_norm_gate_launch_geometry.py').read_text()


def metrics():
    with urllib.request.urlopen('http://192.168.1.15:8000/metrics', timeout=10) as r:
        text = r.read().decode()
    return {line.split()[0]: float(line.split()[-1])
            for line in text.splitlines()
            if line.startswith(('vllm:num_requests_running',
                                'vllm:num_requests_waiting',
                                'vllm:request_success_total'))}


before = metrics()
assert before and not any(v for k, v in before.items() if 'num_requests_' in k)
args.output.mkdir(parents=True, exist_ok=False)
(args.output / 'before.json').write_text(json.dumps(before))
payload = (
    'import pathlib,tempfile,sys\n'
    'p=pathlib.Path(tempfile.mkdtemp(prefix="norm-gate-launch-test-"))\n'
    f'(p/"source.py").write_bytes({kernel!r})\n'
    f's={test!r}\n'
    '(p/"test.py").write_text(s)\n'
    f'sys.argv=[str(p/"test.py"),"--source",str(p/"source.py"),"--sha256",{args.sha256!r}]\n'
    'exec(compile(s,str(p/"test.py"),"exec"),'
    '{"__name__":"__main__","__file__":str(p/"test.py")})\n'
)
container = 'qwen029-tp2' if args.host.endswith(('.15', '.16')) else 'qwen029-tp4'
proc = subprocess.run(
    ['ssh', '-o', 'BatchMode=yes', 'cluster-user@' + args.host,
     f'podman exec -i {container} timeout --signal=TERM --kill-after=5s 150s python3 -'],
    input=payload.encode(), capture_output=True, timeout=175)
(args.output / 'run.log').write_bytes(proc.stdout + proc.stderr)
after = metrics()
(args.output / 'after.json').write_text(json.dumps(after))
assert proc.returncode == 0, (proc.returncode, proc.stderr.decode(errors='replace')[-1500:])
assert before == after, 'Serving activity changed during the diagnostic'
results = [json.loads(line[7:]) for line in proc.stdout.decode().splitlines()
           if line.startswith('RESULT=')]
assert len(results) == 1
result = results[0]
result.update(host=args.host, container=container,
              test_sha256=hashlib.sha256(test.encode()).hexdigest(),
              service_counters_unchanged=True)
(args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
