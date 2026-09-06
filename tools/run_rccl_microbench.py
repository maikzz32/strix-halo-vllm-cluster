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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="15,16,17,18")
    parser.add_argument("--port", type=int, default=29761)
    parser.add_argument("--deadline", type=int, default=120)
    parser.add_argument("--sizes", help="Comma-separated positive even byte counts; default 4096,8192,16384 (profile smoke: 20480)")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--profile-smoke", action="store_true", help="Four-rank20KiB graph profiler validation, hard kill at30s")
    args = parser.parse_args()
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
        if not separator or not re.fullmatch(r"(?:NCCL|RCCL)_[A-Z0-9_]+", key) or "\n" in value:
            parser.error("--env accepts NCCL_/RCCL_ NAME=VALUE entries only")
        baseline[key] = value
    source = pathlib.Path(__file__).with_name("rccl_microbench.py").read_text(encoding="utf-8")
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
                   f"maik@192.168.1.{node}", shlex.join(remote)]
        commands.append((rank, node, command))
    if args.dry_run:
        print(json.dumps({"output": str(output), "run_id":run_id, "sizes_bytes":sizes,
                          "source_sha256":hashlib.sha256(source.encode('utf-8')).hexdigest(),
                          "commands": commands}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps({"nodes": nodes, "port": args.port,
        "baseline_environment": baseline, "deadline": args.deadline, "cuda_graph":args.graph,
        "sizes_bytes":sizes, "source_sha256":hashlib.sha256(source.encode('utf-8')).hexdigest(),
        "profile_smoke":args.profile_smoke,"run_id":run_id,"commands": commands}, indent=2), encoding="utf-8")

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
    print(f"All rank processes joined; logs: {output}", flush=True)
    return 0 if all(result["exit_code"] == 0 for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
