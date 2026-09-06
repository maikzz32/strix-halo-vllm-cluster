"""Run on node 1: python3 tools/cluster.py status|start|stop|restart."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import pathlib
import re
import shlex
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = pathlib.Path("/home/maik/strix-halo-next")


def remote(node, rank, action, args, run_id=None, quiet=False):
    argv = ["python3", str(ROOT / "tools/node_runtime.py"), action,
            "--rank", str(rank), "--config", args.config, "--tag", args.tag]
    if run_id:
        argv += ["--run-id", run_id]
    if action in ("validate", "preflight", "start"):
        argv += ["--expected-config-sha256", args.config_sha256]
    process = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
         f"maik@{node['host']}", shlex.join(argv)],
        capture_output=True, text=True, timeout=100 if action == "start" else 60)
    if not quiet or process.returncode:
        print(f"rank{rank} {action}: {process.stdout.strip()} {process.stderr.strip()}", flush=True)
    if process.returncode:
        raise RuntimeError(f"rank{rank} {action} failed ({process.returncode})")
    return json.loads(process.stdout)


def all_nodes(config, action, args, run_id=None, quiet=False):
    results, failures = {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(config["nodes"])) as pool:
        tasks = {pool.submit(remote, node, rank, action, args, run_id, quiet): rank
                 for rank, node in enumerate(config["nodes"])}
        for task in concurrent.futures.as_completed(tasks):
            rank = tasks[task]
            try:
                results[rank] = task.result()
            except Exception as error:
                failures.append(f"rank{rank}: {error}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return results


def request_counts(metrics):
    """Require usable running/waiting gauges before allowing a normal stop."""
    totals = {"running": 0.0, "waiting": 0.0}
    seen = set()
    pattern = re.compile(r'^vllm:num_requests_(running|waiting)(?:\{.*\})?\s+(\S+)(?:\s+\S+)?\s*$')
    for line in metrics.splitlines():
        match = pattern.fullmatch(line)
        if match:
            kind, value = match.groups()
            count = float(value)
            if not math.isfinite(count) or count < 0:
                raise RuntimeError(f"Invalid request gauge {kind}={value}; idle state unknown")
            totals[kind] += count
            seen.add(kind)
    if seen != set(totals):
        raise RuntimeError("Request gauges missing from metrics; idle state unknown")
    return totals


def ensure_idle(config, args):
    url = f"http://127.0.0.1:{config['api_port']}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            metrics = response.read().decode()
    except OSError as error:
        states = all_nodes(config, "status", args, quiet=True)
        if any(state["processes"] or state["launcher_alive"] for state in states.values()):
            raise RuntimeError("Cannot verify idle requests while inference is present; metrics unavailable") from error
        return
    if any(request_counts(metrics).values()):
        raise RuntimeError("Active requests present; retry after they finish.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "stop", "restart", "status", "validate", "preflight", "dry-run"])
    ap.add_argument("--config", default=str(ROOT / "config/cluster.json"))
    ap.add_argument("--tag", default="cluster")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--force-stop", action="store_true",
                    help="Explicitly stop even when requests are active or metrics are unavailable")
    args = ap.parse_args()
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    config = json.loads(pathlib.Path(args.config).read_text())
    args.config_sha256 = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if args.action in ("status", "validate", "preflight", "dry-run"):
        all_nodes(config, args.action, args)
        return 0
    if args.action == "restart":
        # Catch inaccessible nodes, config drift or missing model/runtime files
        # while the current service is still intact. Busy processes are allowed.
        all_nodes(config, "validate", args)
    if args.action in ("stop", "restart"):
        if not args.force_stop:
            ensure_idle(config, args)
        all_nodes(config, "stop", args)
    if args.action == "stop":
        return 0
    # Every rank must be ready before any rank is started. Node start rechecks
    # under a local lock to reject concurrent or stale coordinators.
    all_nodes(config, "preflight", args)
    run_id = uuid.uuid4().hex
    print(f"STARTING run_id={run_id}", flush=True)
    try:
        all_nodes(config, "start", args, run_id)
        deadline = time.monotonic() + args.timeout
        url = f"http://127.0.0.1:{config['api_port']}/v1/models"
        while time.monotonic() < deadline:
            states = all_nodes(config, "status", args, run_id, quiet=True)
            for rank, state in states.items():
                if state.get("run_id") != run_id or not state.get("launcher_alive"):
                    raise RuntimeError(f"Rank {rank} launcher died during startup; log={state.get('log')}")
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    models = json.load(response)
                if all(state["processes"] for state in states.values()) and any(
                        model["id"] == config["model"] for model in models["data"]):
                    print(f"READY {url} run_id={run_id}", flush=True)
                    return 0
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(5)
        raise RuntimeError("Startup timed out; rank logs retained")
    except BaseException:
        # Include uncertain SSH outcomes: cleanup matches the new token only,
        # never an inference process which existed before this attempted start.
        try:
            all_nodes(config, "cleanup", args, run_id)
        except Exception as cleanup_error:
            print(f"Run {run_id} rollback incomplete: {cleanup_error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
