#!/usr/bin/env python3
"""Prepare or explicitly execute a bounded four-host CPU verbs benchmark.

Default is a local dry run. --execute requires a separately coordinated network
benchmark window. Never changes serving or installs dependencies; no GPU imports.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
import uuid

REMOTE = r'''
import hashlib, json, os, pathlib, signal, subprocess, sys, tempfile, time
cfg = CONFIG
proc = None
def stop_owned():
    global proc
    if proc is not None and proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait(timeout=3)
def interrupted(signum, frame):
    raise RuntimeError('launcher received signal %d' % signum)
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
def run_owned(command, deadline):
    global proc
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=deadline)
        sys.stdout.buffer.write(output); sys.stdout.buffer.flush()
        return proc.returncode, output
    except BaseException:
        stop_owned()
        output, _ = proc.communicate()
        sys.stdout.buffer.write(output); sys.stdout.buffer.flush()
        raise
try:
    source = cfg['source'].encode('utf-8')
    assert hashlib.sha256(source).hexdigest() == cfg['source_sha256']
    with tempfile.TemporaryDirectory(prefix='strix-cpu-rdma-' + cfg['run_id'] + '-') as directory:
        path = pathlib.Path(directory)
        src = path / 'allreduce.c'; binary = path / 'allreduce'
        src.write_bytes(source)
        code, _ = run_owned(['gcc', '-O3', '-std=c11', '-Wall', '-Wextra', '-Werror',
                             str(src), '-o', str(binary), '-libverbs', '-lm'], 20)
        if code: raise RuntimeError('compiler exit %d' % code)
        code, raw = run_owned([str(binary), '--self-test'], 5)
        if code: raise RuntimeError('CPU self-test exit %d' % code)
        result = json.loads(raw)
        for key, value in cfg['fingerprints'].items():
            if result[key] != value: raise RuntimeError('CPU fingerprint mismatch: ' + key)
        code, raw = run_owned([str(binary), '--sum-self-test'], 10)
        if code: raise RuntimeError('SIMD/scalar self-test exit %d' % code)
        checks = [json.loads(line) for line in raw.splitlines() if line.startswith(b'{')]
        if not any(r.get('event') == 'simd_self_test' and r.get('bitexact') is True and
                   (cfg['implementation'] != 'avx2' or r.get('avx2_available') is True) for r in checks):
            raise RuntimeError('SIMD exactness/availability proof absent')
        print(json.dumps({'event':'build', 'run_id':cfg['run_id'], 'rank':cfg['rank'],
                          'source_sha256':cfg['source_sha256'], 'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}), flush=True)
        code, _ = run_owned([str(binary)] + cfg['arguments'], cfg['deadline'] + 8)
        if code: raise RuntimeError('rank exit %d' % code)
    print(json.dumps({'event':'launcher_complete', 'run_id':cfg['run_id'], 'rank':cfg['rank'],
                      'child_exit_code':0, 'temporary_files_removed':True}), flush=True)
except BaseException as exc:
    stop_owned()
    print(json.dumps({'event':'launcher_error', 'run_id':cfg['run_id'], 'rank':cfg['rank'],
                      'error':str(exc)}), flush=True)
    sys.exit(1)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Use only in an authorized network-test window')
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--port', type=int, default=29871)
    parser.add_argument('--deadline', type=int, default=30)
    parser.add_argument('--iterations', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=16)
    parser.add_argument('--mode', choices=['fp32', 'bf16', 'both'], default='both')
    parser.add_argument('--implementation', choices=['scalar', 'avx2'], default='scalar')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not 5 <= args.deadline <= 120:
        parser.error('nonprivileged port and 5–120 second rank deadline required')
    if not 8 <= args.iterations <= 4096 or not 1 <= args.warmup <= 256:
        parser.error('iterations 8–4096 and warmup 1–256 required')
    source = pathlib.Path(__file__).with_name('cpu_rdma_allreduce.c').read_text(encoding='utf-8')
    header = pathlib.Path(__file__).with_name('cpu_rdma_reduce_avx2.h').read_text(encoding='utf-8')
    source = source.replace('#include "cpu_rdma_reduce_avx2.h"', header)
    sha = hashlib.sha256(source.encode('utf-8')).hexdigest()
    run_id = uuid.uuid4().hex
    fingerprints = {'inputs':'47a59eac269a4554', 'fp32':'2a50bdc15286f295', 'bf16':'7a1bd798ea622f59'}
    configs = []
    for rank, node in enumerate((15, 16, 17, 18)):
        arguments = ['--rank', str(rank), '--run-id', run_id, '--device', 'rocep197s0f1' if node == 18 else 'rocep197s0f3',
                     '--master', '192.168.100.1', '--port', str(args.port), '--gid-index', '1',
                     '--deadline', str(args.deadline), '--iterations', str(args.iterations),
                     '--warmup', str(args.warmup), '--mode', args.mode, '--implementation', args.implementation]
        configs.append({'rank':rank, 'node':node, 'source_sha256':sha, 'run_id':run_id,
                        'arguments':arguments, 'deadline':args.deadline, 'fingerprints':fingerprints,
                        'implementation':args.implementation})
    manifest = {'run_id':run_id, 'source_sha256':sha, 'configs':configs,
                'algorithm':'three parallel RC RDMA_WRITE_WITH_IMM sends then local rank-order sum',
                'payload_bytes':20480, 'tx_payload_bytes_per_rank':61440, 'gpu_dependency':False,
                'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if not args.execute:
        print(json.dumps(manifest, indent=2)); return 0
    output = args.output or pathlib.Path(__file__).resolve().parents[1] / 'results' / ('cpu-rdma-' + run_id)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'source.c').write_text(source, encoding='utf-8')

    def ssh(node):
        return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=5',
                '-o', 'ServerAliveCountMax=3', f'cluster-user@192.168.1.{node}', 'python3 -S -']

    def run_rank(config):
        cfg = dict(config, source=source)
        script = REMOTE.replace('CONFIG', repr(cfg), 1)
        try:
            completed = subprocess.run(ssh(cfg['node']), input=script.encode('utf-8'), capture_output=True,
                                       timeout=args.deadline + 45)
            raw = completed.stdout + completed.stderr
            code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            raw = (exc.stdout or b'') + (exc.stderr or b'') + b'\nLOCAL_SSH_TIMEOUT\n'; code = 124
        except OSError as exc:
            raw = str(exc).encode('utf-8'); code = 1
        (output / f"rank{cfg['rank']}-node{cfg['node']}.log").write_bytes(raw)
        records = []
        for line in raw.decode('utf-8', errors='replace').splitlines():
            try: record = json.loads(line)
            except ValueError: continue
            if isinstance(record, dict): records.append(record)
        complete = any(r.get('event') == 'complete' and r.get('run_id') == run_id and
                       r.get('rank') == cfg['rank'] and r.get('teardown_ok') is True for r in records)
        joined = any(r.get('event') == 'launcher_complete' and r.get('run_id') == run_id and
                     r.get('rank') == cfg['rank'] and r.get('child_exit_code') == 0 for r in records)
        modes = {r.get('mode') for r in records if r.get('event') == 'result' and r.get('correct') is True and
                 r.get('run_id') == run_id and r.get('rank') == cfg['rank']}
        expected_modes = {'fp32_then_bf16', 'bf16_each_add'} if args.mode == 'both' else {
            'fp32_then_bf16' if args.mode == 'fp32' else 'bf16_each_add'}
        good = code == 0 and complete and joined and modes == expected_modes
        print(f"rank {cfg['rank']} node {cfg['node']}: exit={code}, complete={complete}, joined={joined}", flush=True)
        return {'rank':cfg['rank'], 'node':cfg['node'], 'exit_code':code, 'verified':good, 'records':records}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        status = list(pool.map(run_rank, configs))
    (output / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')

    def release_scan(config):
        script = """import json,pathlib
run_id = RUN_ID
found=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try:
  argv=p.joinpath('cmdline').read_bytes().split(b'\\0')
  marker=b'--run-id'
  if marker in argv and argv[argv.index(marker)+1].decode()==run_id:
   stat=p.joinpath('stat').read_text(); tail=stat[stat.rfind(')')+2:].split()
   if tail[0]!='Z': found.append({'pid':int(p.name),'start_ticks':tail[19]})
 except (OSError,IndexError,UnicodeError): pass
print(json.dumps({'run_id':run_id,'remaining':found}))
""".replace('RUN_ID', repr(run_id), 1)
        try:
            result = subprocess.run(ssh(config['node']), input=script.encode(), capture_output=True, timeout=15)
            data = json.loads(result.stdout)
            return {'node':config['node'], 'exit_code':result.returncode, **data}
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            return {'node':config['node'], 'exit_code':1, 'error':str(exc)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        release = list(pool.map(release_scan, configs))
    (output / 'release.json').write_text(json.dumps(release, indent=2), encoding='utf-8')
    clean = all(r['exit_code'] == 0 and r.get('remaining') == [] for r in release)
    print(f'Results: {output}; owned-rank scan clean={clean}', flush=True)
    return 0 if clean and all(r['verified'] for r in status) else 1


if __name__ == '__main__':
    sys.exit(main())
