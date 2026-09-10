#!/usr/bin/env python3
"""Launch bounded, foreground RCCL ranks over SSH; no server restarts.

Validated on the four existing nodes; run only with idle GPUs and never
concurrently with inference performance comparisons.
Example: python tools/run_rccl_microbench.py --nodes 15,16,17,18
Optional override: --env NCCL_MIN_NCHANNELS=2 --env NCCL_MAX_NCHANNELS=2
Each subprocess has a remote timeout plus the Python rank's alarm; all SSH
processes are joined. Output from every rank is saved, including failures.
"""
import argparse
import array
import base64
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import sys
import uuid


def extract_bf16_reference(output, results, run_id, generator_sha):
    """CPU-only archival validation after all existing owned SSH ranks joined."""
    payloads, ranks, errors = [], [], []
    for status in results:
        rank, node = status['rank'], status['node']
        try:
            records = []
            for line in (output / f'rank{rank}-node{node}.log').read_text(encoding='utf-8', errors='replace').splitlines():
                try: record = json.loads(line)
                except ValueError: continue
                if isinstance(record, dict) and record.get('run_id') == run_id and record.get('rank') == rank:
                    records.append(record)
            found = [r for r in records if r.get('kind') == 'reference_payload']
            if len(found) != 1 or status['exit_code'] != 0:
                raise ValueError('missing unique reference payload or nonzero rank exit')
            record = dict(found[0])
            payload = base64.b64decode(record.pop('payload_b64'), validate=True)
            if len(payload) != 32 * 20480 or hashlib.sha256(payload).hexdigest() != record['output_sha256']:
                raise ValueError('output payload length/hash mismatch')
            words = array.array('H'); words.frombytes(payload)
            if sys.byteorder != 'little': words.byteswap()
            if any((word & 0x7f80) == 0x7f80 for word in words):
                raise ValueError('binary payload contains NaN/Infinity')
            if (record.get('generator_sha256') != generator_sha or record.get('banks') != 32 or
                record.get('values') != 10240 or record.get('byte_order') != 'little' or
                record.get('finite') is not True or record.get('all_gpu_inputs_checked') is not True or
                record.get('input_resets') != 33 or record.get('bank0_repeat_equal') is not True):
                raise ValueError('reference metadata/reset proof mismatch')
            if not any(r.get('kind') == 'complete' and r.get('graph_reset') is True and
                       r.get('communicator_destroyed') is True for r in records):
                raise ValueError('missing post-teardown completion proof')
            metadata = next((r for r in records if r.get('kind') == 'reference_metadata'), None)
            if (not metadata or metadata.get('world_size') != 4 or metadata.get('generator_sha256') != generator_sha or
                len(metadata.get('input_bank_sha256', [])) != 32 or
                record.get('output_bank_sha256') != [hashlib.sha256(payload[i*20480:(i+1)*20480]).hexdigest() for i in range(32)]):
                raise ValueError('missing rank metadata or per-bank hash mismatch')
            (output / f'rank{rank}-node{node}.bf16.bin').write_bytes(payload)
            (output / f'rank{rank}-node{node}.reference.json').write_text(
                json.dumps({'metadata':metadata, 'result':record, 'exit_code':status['exit_code']}, indent=2), encoding='utf-8')
            payloads.append(payload); ranks.append({'rank':rank, 'node':node, 'exit_code':status['exit_code'], **record})
        except (ValueError, KeyError, OSError) as exc:
            errors.append({'rank':rank, 'node':node, 'error':str(exc)})
    equal = len(payloads) == 4 and all(p == payloads[0] for p in payloads[1:])
    summary = {'run_id':run_id, 'generator_sha256':generator_sha, 'ranks':ranks, 'errors':errors,
               'all_rank_outputs_equal':equal, 'bytes_per_rank':32 * 20480,
               'performance_claim':False, 'model_quality_claim':False}
    (output / 'reference-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return not errors and equal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="15,16,17,18")
    parser.add_argument("--port", type=int, default=29761)
    parser.add_argument("--deadline", type=int, default=120)
    parser.add_argument("--sizes", help="Comma-separated positive even byte counts; default 4096,8192,16384 (profile smoke: 20480)")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--hca-by-node", action="store_true", help="Select the known connected Intel port on each node")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--profile-smoke", action="store_true", help="Four-rank20KiB graph profiler validation, hard kill at30s")
    parser.add_argument("--bf16-reference", action="store_true", help="Archive actual RCCL outputs for32 CPU-reference banks; no timing")
    args = parser.parse_args()
    if args.bf16_reference:
        if args.profile_smoke:
            parser.error('--bf16-reference cannot be combined with --profile-smoke')
        if args.sizes not in (None, '20480'):
            parser.error('BF16 reference is fixed at20480 bytes')
        args.sizes = '20480'; args.graph = True; args.deadline = min(args.deadline, 45)
    try:
        size_arg = args.sizes if args.sizes is not None else ('20480' if args.profile_smoke else '4096,8192,16384')
        sizes = [int(value) for value in size_arg.split(',')]
    except ValueError:
        parser.error('--sizes must be comma-separated byte counts')
    if any(value <= 0 or value % 2 for value in sizes):
        parser.error('--sizes must contain positive even byte counts (BF16)')
    if args.profile_smoke and sizes != [20480]:
        parser.error('profile smoke is fixed at --sizes 20480')
    nodes = [int(value) for value in args.nodes.split(",")]
    if len(nodes) not in (2, 4) or len(set(nodes)) != len(nodes) or not set(nodes) <= {15, 16, 17, 18}:
        parser.error("choose two or four distinct nodes from 15,16,17,18")
    if args.bf16_reference and len(nodes) != 4:
        parser.error('BF16 reference requires four ranks')
    if not 1024 <= args.port <= 65535 or not 30 <= args.deadline <= 600:
        parser.error("use a nonprivileged port and a deadline between 30 and 600 seconds")
    baseline = {"NCCL_IB_GID_INDEX": "1", "NCCL_NET_GDR_LEVEL": "0",
                "NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4",
                "NCCL_LAUNCH_MODE": "GROUP", "NCCL_GRAPH_MIXING_SUPPORT": "1"}
    if args.profile_smoke:
        if len(nodes) != 4:
            parser.error('profile smoke requires all four ranks')
        args.deadline = 30
        baseline.update({'ROCPROFILER_SET_ROCPROFILER_REGISTER_LIBRARY':'0',
                         'ROCPROFILER_QUEUE_INTERPOSITION':'0','OMP_NUM_THREADS':'1',
                         'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    for override in args.env:
        key, separator, value = override.partition("=")
        if not separator or not re.fullmatch(r"(?:NCCL|RCCL|UCCL)_[A-Z0-9_]+", key) or "\n" in value:
            parser.error("--env accepts NCCL_/RCCL_/UCCL_ NAME=VALUE entries only")
        baseline[key] = value
    source = pathlib.Path(__file__).with_name("rccl_bf16_reference.py" if args.bf16_reference else "rccl_microbench.py").read_text(encoding="utf-8")
    generator_sha = None
    if args.bf16_reference:
        reference = pathlib.Path(__file__).with_name('cpu_rdma_reference.py').read_bytes()
        generator_sha = hashlib.sha256(reference).hexdigest()
        source = source.replace('_REFERENCE_SOURCE_B64 = None', '_REFERENCE_SOURCE_B64 = ' + repr(base64.b64encode(reference).decode('ascii')), 1)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex
    output = args.output or pathlib.Path(__file__).resolve().parents[1] / "research" / f"rccl-{stamp}"
    commands = []
    for rank, node in enumerate(nodes):
        env = dict(baseline)
        env.update({"RANK": str(rank), "WORLD_SIZE": str(len(nodes)),
                    "MASTER_ADDR": f"192.168.100.{nodes[0] - 14}", "MASTER_PORT": str(args.port),
                    "NCCL_SOCKET_IFNAME": "enp197s0f1np1" if node == 18 else "enp197s0f3np3"})
        container = "ray-head" if node == 15 else "ray-worker"
        if args.hca_by_node:
            env['NCCL_IB_HCA'] = 'rocep197s0f1' if node == 18 else 'rocep197s0f3'
            env['UCCL_IB_HCA'] = env['NCCL_IB_HCA']
            env['UCCL_SOCKET_IFNAME'] = env['NCCL_SOCKET_IFNAME']
        remote = ["podman", "exec", "-i"]
        for key, value in env.items():
            remote += ["-e", f"{key}={value}"]
        if args.profile_smoke:
            remote += [container, 'timeout', '--signal=KILL', '30s', 'python3', '-',
                       '--deadline', '25', '--profile-smoke', '--run-id', run_id]
        else:
            remote += [container, "timeout", "--signal=TERM", "--kill-after=5s",
                       f"{args.deadline + 10}s", "python3", "-", "--deadline", str(args.deadline),
                       '--run-id', run_id]
        if args.graph:
            remote += ["--graph"]
        remote += ['--sizes', ','.join(str(value) for value in sizes)]
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                   f"cluster-user@192.168.1.{node}", shlex.join(remote)]
        commands.append((rank, node, command))
    if args.dry_run:
        print(json.dumps({"output": str(output), "run_id":run_id, "sizes_bytes":sizes,
                          "source_sha256":hashlib.sha256(source.encode('utf-8')).hexdigest(),
                          "bf16_reference":args.bf16_reference,"generator_sha256":generator_sha,
                          "commands": commands}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps({"nodes": nodes, "port": args.port,
        "baseline_environment": baseline, "deadline": args.deadline, "cuda_graph":args.graph,
        "sizes_bytes":sizes, "source_sha256":hashlib.sha256(source.encode('utf-8')).hexdigest(),
        "profile_smoke":args.profile_smoke,"bf16_reference":args.bf16_reference,"generator_sha256":generator_sha,
        "run_id":run_id,"commands": commands}, indent=2), encoding="utf-8")
    if args.bf16_reference:
        (output / 'rank-source.py').write_text(source, encoding='utf-8')

    def run_rank(item):
        rank, node, command = item
        try:
            result = subprocess.run(command, input=source.encode("utf-8"), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=args.deadline + 35)
            raw, code = result.stdout, result.returncode
        except subprocess.TimeoutExpired as exc:
            raw, code = (exc.stdout or b"") + b"\nLOCAL_SSH_TIMEOUT\n", 124
        (output / f"rank{rank}-node{node}.log").write_bytes(raw)
        print(f"rank {rank} / node {node}: exit {code}", flush=True)
        return {"rank": rank, "node": node, "exit_code": code}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        results = list(pool.map(run_rank, commands))
    (output / "status.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    reference_ok = True
    if args.bf16_reference:
        reference_ok = extract_bf16_reference(output, results, run_id, generator_sha)
    print(f"All rank processes joined; logs: {output}", flush=True)
    return 0 if reference_ok and all(result["exit_code"] == 0 for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
