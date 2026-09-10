#!/usr/bin/env python3
"""Prepare or explicitly run a bounded HIP/CPU-RDMA graph test on four nodes.

Default is a local dry run. Execution requires idle serving and a separately
coordinated GPU/network window. All native compilation happens in the preserved
containers; no packages, model weights or serving settings are changed.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
_controller = ROOT / 'tools' / 'cluster.py'
if not _controller.exists():
    _controller = ROOT / 'scripts' / 'native' / 'cluster.py'
_spec = importlib.util.spec_from_file_location('hip_bench_cluster_metrics', _controller)
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
request_counts = _metrics.request_counts

SOURCES = {
    'cpu_rdma_transport.c': 'tools/cpu_rdma_transport.c',
    'cpu_rdma_transport.h': 'tools/cpu_rdma_transport.h',
    'cpu_rdma_reduce_avx2.h': 'tools/cpu_rdma_reduce_avx2.h',
    'cpu_rdma_reference.py': 'tools/cpu_rdma_reference.py',
    'hip_rdma_graph.cpp': 'tests/hip_rdma_graph.cpp',
    'bench_hip_rdma_graph.py': 'tests/bench_hip_rdma_graph.py',
}

BACKEND_SOURCES = {
    'cpu_rdma_transport.c': 'tools/cpu_rdma_transport.c',
    'cpu_rdma_transport.h': 'tools/cpu_rdma_transport.h',
    'cpu_rdma_reduce_avx2.h': 'tools/cpu_rdma_reduce_avx2.h',
    'cpu_rdma_reference.py': 'tools/cpu_rdma_reference.py',
    'cpu_rdma_ring4_reference.py': 'tools/cpu_rdma_ring4_reference.py',
    'analyze_rccl_bf16_reference.py': 'tools/analyze_rccl_bf16_reference.py',
    'hip_rdma_backend.cpp': 'tools/hip_rdma_backend.cpp',
    'hip_rdma_backend.py': 'tools/hip_rdma_backend.py',
    'validate_rccl_ring4.py': 'tools/validate_rccl_ring4.py',
    'bench_hip_rdma_backend.py': 'tests/bench_uma_affinity.py',
}

REMOTE = r'''
import hashlib, json, os, pathlib, signal, subprocess, sys, tempfile, time
cfg = CONFIG
proc = None
marker_name = 'STRIX_HIP_RDMA_RUN_ID'
marker = (marker_name + '=' + cfg['run_id']).encode()

def identity(pid):
    path = pathlib.Path('/proc') / str(pid)
    fields = path.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    return None if fields[0] == 'Z' else (pid, fields[19])

def has_marker(pid):
    return marker in (pathlib.Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0')

def owned_processes():
    found = []
    for path in pathlib.Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            before = identity(int(path.name))
            if before is not None and has_marker(before[0]) and identity(before[0]) == before:
                found.append(before)
        except (FileNotFoundError, ProcessLookupError):
            pass
    return found

def signal_owned(expected, signum):
    # Bind the signal to a kernel process handle, then recheck both ownership
    # and start time. A recycled numeric PID must never target another task.
    pidfd = None
    try:
        pidfd = os.pidfd_open(expected[0], 0)
        if identity(expected[0]) == expected and has_marker(expected[0]):
            signal.pidfd_send_signal(pidfd, signum, None, 0)
    except (FileNotFoundError, ProcessLookupError):
        pass
    finally:
        if pidfd is not None:
            os.close(pidfd)

def stop_owned():
    # Descendants retain the environment marker even after leader exit/setsid.
    # Rescan during each bounded phase to catch children born during shutdown.
    for signum in (signal.SIGTERM, signal.SIGKILL):
        deadline = time.monotonic() + 1.5
        signalled = set()
        while True:
            if proc is not None:
                proc.poll()  # Reap the direct child; this is not ownership proof.
            remaining = owned_processes()
            if not remaining:
                return []
            for item in remaining:
                if item not in signalled:
                    signal_owned(item, signum)
                    signalled.add(item)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    if proc is not None:
        proc.poll()
    return owned_processes()

def interrupted(signum, frame):
    raise RuntimeError('launcher interrupted: %d' % signum)
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)

def command(argv, seconds, cwd):
    global proc
    environment = os.environ.copy()
    environment.update(cfg.get('environment', {}))
    environment[marker_name] = cfg['run_id']
    proc = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True, env=environment)
    try:
        raw, _ = proc.communicate(timeout=seconds)
    except BaseException as error:
        remaining = stop_owned()
        incomplete_pipe = False
        try:
            raw, _ = proc.communicate(timeout=1)
        except subprocess.TimeoutExpired as drain_error:
            raw = drain_error.output or b''
            incomplete_pipe = True
            if proc.stdout is not None:
                proc.stdout.close()
        sys.stdout.buffer.write(raw); sys.stdout.buffer.flush()
        if remaining or incomplete_pipe:
            raise RuntimeError('owned child cleanup incomplete: remaining=%r pipe_open=%r' %
                               (remaining, incomplete_pipe)) from error
        raise
    sys.stdout.buffer.write(raw); sys.stdout.buffer.flush()
    if owned_processes():
        remaining = stop_owned()
        raise RuntimeError('child left marked descendants; remaining after cleanup=%r' % remaining)
    if proc.returncode:
        raise RuntimeError('child exit %d: %s' % (proc.returncode, argv[0]))

try:
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('pidfd ownership-safe cleanup requires Linux/Python pidfd support')
    with tempfile.TemporaryDirectory(prefix='strix-hip-rdma-' + cfg['run_id'] + '-') as directory:
        stage = pathlib.Path(directory)
        for name, text in cfg['sources'].items():
            data = text.encode('utf-8')
            if pathlib.Path(name).name != name or hashlib.sha256(data).hexdigest() != cfg['source_sha256'][name]:
                raise ValueError('source identity mismatch: ' + name)
            (stage / name).write_bytes(data)
        command(['gcc', '-std=c11', '-O3', '-fPIC', '-Wall', '-Wextra', '-Werror',
                 '-I.', '-c', 'cpu_rdma_transport.c', '-o', 'transport.o'], 25, stage)
        if cfg.get('uma_backend'):
            command(['/opt/rocm/bin/hipcc','-std=c++17','-O3','--offload-arch=gfx1151',
                     '-DSTRIX_UMA_THREADS='+str(cfg['uma_threads']),
                     '-fPIC','-I.','-c','hip_uma_backend.hip','-o','uma.o'],25,stage)
            command(['g++','-shared','uma.o','transport.o','-L/opt/rocm/lib',
                     '-Wl,-rpath,/opt/rocm/lib','-lamdhip64','-libverbs','-lpthread',
                     '-o','libhip_rdma_graph.so'],25,stage)
        else:
            command(['g++', '-std=c++17', '-O2', '-fPIC', '-shared', '-D__HIP_PLATFORM_AMD__',
                 '-I.', '-I/opt/rocm/include', cfg.get('cpp_source', 'hip_rdma_graph.cpp'), 'transport.o',
                 '-L/opt/rocm/lib', '-Wl,-rpath,/opt/rocm/lib', '-lamdhip64',
                 '-libverbs', '-lpthread', '-lm', '-o', 'libhip_rdma_graph.so'], 25, stage)
        library = stage / 'libhip_rdma_graph.so'
        print(json.dumps({'event':'build', 'rank':cfg['rank'], 'run_id':cfg['run_id'],
                          'source_sha256':cfg['source_sha256'],
                          'binary_sha256':hashlib.sha256(library.read_bytes()).hexdigest()}), flush=True)
        output = stage / 'result.json'
        try:
            command(['python3', *([] if cfg.get('use_site') else ['-S']),
                     str(stage / cfg.get('rank_script', 'bench_hip_rdma_graph.py')),
                     '--library', str(library), '--output', str(output), *cfg['arguments']],
                    cfg['deadline'], stage)
        finally:
            if output.exists():
                print(json.dumps({'event':'rank_artifact', 'rank':cfg['rank'],
                                  'run_id':cfg['run_id'], 'artifact':json.loads(output.read_text())}), flush=True)
    print(json.dumps({'event':'launcher_complete', 'rank':cfg['rank'], 'run_id':cfg['run_id'],
                      'child_exit_code':0, 'temporary_files_removed':True,
                      'owned_processes_gone':True}), flush=True)
except BaseException as error:
    try:
        remaining = stop_owned()
        cleanup_error = None
    except BaseException as cleanup_failure:
        remaining, cleanup_error = None, repr(cleanup_failure)
    print(json.dumps({'event':'launcher_error', 'rank':cfg['rank'], 'run_id':cfg['run_id'],
                      'error':repr(error), 'remaining':remaining,
                      'cleanup_error':cleanup_error}), flush=True)
    raise SystemExit(1)
'''


def serving_snapshot(url):
    with urllib.request.urlopen(url.rstrip('/') + '/metrics', timeout=10) as response:
        metrics = response.read().decode('utf-8')
    counts = request_counts(metrics)
    successes = [float(match.group(1)) for match in re.finditer(
        r'^vllm:request_success_total(?:\{.*\})?\s+(\S+)', metrics, re.M)]
    if not successes or any(not math.isfinite(v) or v < 0 for v in successes):
        raise RuntimeError('Usable request completion counters are required for the benchmark window')
    return {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'requests': counts, 'successful_requests': sum(successes)}


def ssh(node, container, *, timeout=False):
    remote = ['podman', 'exec', '-i', container]
    if timeout:
        remote += ['timeout', '--signal=TERM', '--kill-after=5s', '115s']
    remote += ['python3', '-S', '-']
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
            '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3',
            f'cluster-user@192.168.1.{node}', shlex.join(remote)]


def parse_records(raw):
    records = []
    for line in raw.decode('utf-8', errors='replace').splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            records.append(row)
    return records


def verify_artifact(artifact, config, modes, expected_collectives):
    if artifact.get('status') != 'passed' or artifact.get('rank') != config['rank'] or artifact.get('run_id') != config['run_id']:
        return False
    cases = artifact.get('cases', [])
    if len(cases) != len(modes) or {c.get('mode') for c in cases} != modes:
        return False
    ones = ('exact', 'finite', 'pinned_mr_ok', 'negative_test_passed', 'arena_released')
    zeros = ('returncode', 'main_selftest_rc', 'callback_selftest_rc', 'cleanup_rc', 'max_abs_error')
    for case in cases:
        if case.get('rank') != config['rank'] or case.get('run_id') != config['run_id']:
            return False
        if any(case.get(k) != 1 for k in ones) or any(case.get(k) != 0 for k in zeros) or case.get('error') != '':
            return False
        if (case.get('direct_pinned_checks') != 4 or case.get('node_count') != 3 or
                case.get('host_nodes') != 1 or case.get('memcpy_nodes') != 2 or
                case.get('successful_collectives') != expected_collectives or
                case.get('expected_collectives') != expected_collectives or
                case.get('callback_calls') != expected_collectives + 2):
            return False
        for key in ('wall_median_us', 'hip_event_median_us'):
            value = case.get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                return False
    return True


def verify_backend_artifact(artifact, config, iterations, samples):
    if (artifact.get('status') != 'passed' or artifact.get('mode') != 'ring4' or
            artifact.get('rank') != config['rank'] or artifact.get('run_id') != config['run_id']):
        return False
    flags = ('exact', 'finite', 'unsupported_shape_falls_back', 'graphs_destroyed',
             'backend_closed', 'pynccl_destroyed', 'cpu_group_destroyed')
    if any(artifact.get(k) is not True for k in flags) or artifact.get('cleanup_error'):
        return False
    if (artifact.get('operations_per_graph') != 32 or artifact.get('iterations') != iterations or
            artifact.get('samples') != samples or artifact.get('warmup') != 4 or
            artifact.get('graph_output_checks') != (4+6*samples)*32):
        return False
    for key in ('wall_samples_us', 'hip_event_samples_us'):
        values = artifact.get(key, [])
        if len(values) != 3*samples or any(not isinstance(v, (float,int)) or not math.isfinite(v) or v <= 0 for v in values):
            return False
    affinity=artifact.get('affinity',{})
    trials=affinity.get('trials',[])
    expected=[(i,v) for i in range(samples) for v in (['unbound','llc0','llc1'] if i%2==0 else ['llc1','llc0','unbound'])]
    if affinity.get('restored') is not True or [(t.get('sample'),t.get('variant')) for t in trials]!=expected:return False
    if len(affinity.get('selected_cpus',[]))!=2 or any(c not in affinity.get('original',[]) for c in affinity['selected_cpus']):return False
    if [t.get('wall_us') for t in trials]!=artifact['wall_samples_us'] or [t.get('event_us') for t in trials]!=artifact['hip_event_samples_us']:return False
    if any(t.get('migrations',-1)<0 for t in trials):return False
    proof = artifact.get('parity_proof', {})
    if proof.get('banks') != 32 or proof.get('elements_per_rank') != 10240:
        return False
    if any(proof.get(k) is not True for k in ('exact_eager','exact_graph','validated_against_active_pynccl','repeat_bank0','graph_destroyed')):
        return False
    ranks = proof.get('rank_results', [])
    return (len(ranks) == 4 and {r.get('rank') for r in ranks} == {0,1,2,3}
            and all(r.get('errors') == [] for r in ranks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--backend-test', action='store_true', help='Test isolated descriptor backend, including active PyNccl parity; no serving hook')
    parser.add_argument('--uma-backend',action='store_true',help='Use the isolated one-kernel UMA backend with --backend-test')
    parser.add_argument('--avx512-reduce',action='store_true',help='Isolated exact AVX512 CPU reduction; requires UMA backend')
    parser.add_argument('--uma-threads',type=int,choices=[256,512,1024],default=256)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--port', type=int, default=29881)
    parser.add_argument('--deadline', type=int, default=45)
    parser.add_argument('--replays', type=int, default=32)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--mode', choices=['fp32', 'bf16', 'both', 'ring4'], default='both')
    parser.add_argument('--url', default='http://192.168.1.15:8000')
    args = parser.parse_args()
    if not (args.backend_test and args.uma_backend and args.avx512_reduce and args.uma_threads==1024):
        parser.error("Affinity test requires AVX512 UMA1024 backend flags")
    if not 1024 <= args.port <= 65534 or not 10 <= args.deadline <= 50:
        parser.error('base port 1024–65534 and deadline 10–50 seconds required')
    if not 1 <= args.replays <= 128 or not 1 <= args.samples <= 5:
        parser.error('replays 1–128 and samples 1–5 required')
    if args.backend_test != (args.mode == 'ring4'):
        parser.error('--backend-test requires --mode ring4, and ring4 requires --backend-test')
    if args.uma_backend and not args.backend_test:
        parser.error('--uma-backend requires --backend-test --mode ring4')
    if args.avx512_reduce and not args.uma_backend:
        parser.error('--avx512-reduce requires --uma-backend')
    source_paths = dict(BACKEND_SOURCES if args.backend_test else SOURCES)
    if args.uma_backend:
        source_paths.pop('hip_rdma_backend.cpp')
        source_paths['hip_uma_backend.hip']='tests/hip_uma_affinity.hip'
    sources = {name: (ROOT / path).read_text(encoding='utf-8') for name, path in source_paths.items()}
    if args.avx512_reduce:
        from build_cpu_rdma_wide import generate
        sources['cpu_rdma_transport.c']=generate(sources['cpu_rdma_transport.c'])
        sources['cpu_rdma_reduce_avx512.h']=(ROOT/'tests/cpu_rdma_reduce_avx512.h').read_text(encoding='utf-8')
    hashes = {name: hashlib.sha256(text.encode('utf-8')).hexdigest() for name, text in sources.items()}
    run_id = uuid.uuid4().hex
    configs = []
    for rank, node in enumerate((15, 16, 17, 18)):
        container = 'ray-head' if node == 15 else 'ray-worker'
        arguments = ['--rank', str(rank), '--run-id', run_id,
                     '--device', 'rocep197s0f1' if node == 18 else 'rocep197s0f3',
                     '--master', '192.168.100.1', '--port', str(args.port), '--gid-index', '1',
                     '--deadline', '40' if args.backend_test else '10', '--iterations', str(args.replays), '--warmup', '4',
                     '--samples', str(args.samples), '--mode', args.mode]
        configs.append({'rank':rank, 'node':node, 'container':container, 'run_id':run_id,
                        'arguments':arguments, 'deadline':args.deadline, 'source_sha256':hashes})
        if args.backend_test:
            configs[-1].update(cpp_source='hip_rdma_backend.cpp',uma_backend=args.uma_backend,uma_threads=args.uma_threads,
                               rank_script='bench_hip_rdma_backend.py', use_site=True,
                               environment={'NCCL_IB_GID_INDEX':'1','NCCL_NET_GDR_LEVEL':'0',
                                            'NCCL_MIN_NCHANNELS':'4','NCCL_MAX_NCHANNELS':'4',
                                            'NCCL_LAUNCH_MODE':'GROUP','NCCL_GRAPH_MIXING_SUPPORT':'1',
                                            'NCCL_DEBUG':'INFO',
                                            'NCCL_SOCKET_IFNAME':'enp197s0f1np1' if node == 18 else 'enp197s0f3np3',
                                            'GLOO_SOCKET_IFNAME':'enp197s0f1np1' if node == 18 else 'enp197s0f3np3'})
    manifest = {'run_id':run_id, 'source_sha256':hashes, 'configs':configs,
                'payload_bytes':20480, 'gpu_dependency':True,
                'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'backend_test':args.backend_test,'uma_backend':args.uma_backend,'avx512_reduce':args.avx512_reduce}
    if not args.execute:
        print(json.dumps(manifest, indent=2)); return 0
    before = serving_snapshot(args.url)
    if any(before['requests'].values()):
        raise RuntimeError('Serving is busy; GPU/network test was not started')
    output = args.output or ROOT / 'results' / ('hip-rdma-' + run_id)
    output.mkdir(parents=True, exist_ok=False)
    manifest['serving_before'] = before
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'sources').mkdir()
    for name, source in sources.items():
        (output / 'sources' / name).write_text(source, encoding='utf-8')

    def rank(config):
        payload = REMOTE.replace('CONFIG', repr(dict(config, sources=sources)), 1).encode('utf-8')
        try:
            result = subprocess.run(ssh(config['node'], config['container'], timeout=True),
                                    input=payload, capture_output=True, timeout=130)
            raw, code = result.stdout + result.stderr, result.returncode
        except subprocess.TimeoutExpired as error:
            raw, code = (error.stdout or b'') + (error.stderr or b'') + b'\nLOCAL_TIMEOUT\n', 124
        except OSError as error:
            raw, code = str(error).encode(), 1
        (output / f"rank{config['rank']}-node{config['node']}.log").write_bytes(raw)
        rows = parse_records(raw)
        own = [r for r in rows if r.get('run_id') == run_id and r.get('rank') == config['rank']]
        joined = any(r.get('event') == 'launcher_complete' and r.get('child_exit_code') == 0 for r in own)
        artifacts = [r['artifact'] for r in own if r.get('event') == 'rank_artifact']
        modes = {'fp32_then_bf16', 'bf16_each_add'} if args.mode == 'both' else {
            'fp32_then_bf16' if args.mode == 'fp32' else 'bf16_each_add'}
        if args.backend_test:
            good = code == 0 and joined and len(artifacts) == 1 and verify_backend_artifact(
                artifacts[0], config, args.replays, args.samples)
        else:
            good = code == 0 and joined and len(artifacts) == 1 and verify_artifact(
                artifacts[0], config, modes, 4 + args.replays * args.samples)
        print(f"rank {config['rank']} / node {config['node']}: exit={code}, verified={good}", flush=True)
        return {'rank':config['rank'], 'node':config['node'], 'exit_code':code,
                'verified':good, 'records':rows}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        statuses = list(pool.map(rank, configs))
    (output / 'status.json').write_text(json.dumps(statuses, indent=2), encoding='utf-8')

    def scan(config):
        script = '''import json,pathlib
run_id=RUN_ID
marker=('STRIX_HIP_RDMA_RUN_ID='+run_id).encode()
found=[]
errors=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try:
  stat=p.joinpath('stat').read_text(); before=stat[stat.rfind(')')+2:].split()
  if before[0]=='Z': continue
  argv=p.joinpath('cmdline').read_bytes().split(b'\\0')
  marked=marker in p.joinpath('environ').read_bytes().split(b'\\0')
  named=any(a==b'--run-id' and argv[i+1]==run_id.encode() for i,a in enumerate(argv[:-1]))
  if marked or named:
   stat=p.joinpath('stat').read_text(); fields=stat[stat.rfind(')')+2:].split()
   if fields[0]!='Z' and before[19]==fields[19]:
    found.append({'pid':int(p.name),'start_ticks':fields[19],'marker':marked,'run_id_argument':named})
 except (FileNotFoundError,ProcessLookupError): pass
 except (OSError,IndexError,UnicodeError) as error: errors.append({'pid':int(p.name),'error':repr(error)})
print(json.dumps({'run_id':run_id,'remaining':found,'scan_errors':errors}))
raise SystemExit(1 if errors else 0)
'''.replace('RUN_ID', repr(run_id), 1)
        try:
            result = subprocess.run(ssh(config['node'], config['container']), input=script.encode(),
                                    capture_output=True, timeout=15)
            return {'node':config['node'], 'exit_code':result.returncode, **json.loads(result.stdout)}
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            return {'node':config['node'], 'exit_code':1, 'error':repr(error)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        release = list(pool.map(scan, configs))
    (output / 'release.json').write_text(json.dumps(release, indent=2), encoding='utf-8')
    try:
        after = serving_snapshot(args.url)
        quiet = not any(after['requests'].values()) and before['successful_requests'] == after['successful_requests']
    except (OSError, ValueError, RuntimeError) as error:
        after, quiet = {'error':repr(error)}, False
    (output / 'serving-after.json').write_text(json.dumps(after, indent=2), encoding='utf-8')
    clean = all(r['exit_code'] == 0 and r.get('remaining') == [] for r in release)
    ok = clean and quiet and all(r['verified'] for r in statuses)
    print(json.dumps({'output':str(output), 'passed':ok, 'owned_processes_gone':clean,
                      'serving_remained_idle':quiet}), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
