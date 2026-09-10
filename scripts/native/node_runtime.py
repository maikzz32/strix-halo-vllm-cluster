"""Node-local lifecycle for the existing patched vLLM containers."""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path("/home/cluster-user/strix-halo-next")
RUN_ENV = "STRIX_CLUSTER_RUN_ID"


def verify_config(config, expected):
    actual = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if expected and actual != expected:
        raise RuntimeError(f"Configuration differs from coordinator: expected {expected}, found {actual}")
    return actual


def processes(proc_root=pathlib.Path("/proc")):
    result = {}
    for path in proc_root.glob("[0-9]*"):
        try:
            args = path.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
            stat = path.joinpath("stat").read_text().rsplit(")", 1)[1].split()
            # stat begins at field 3 (state); starttime is field 22.
            item = {"args": args, "state": stat[0], "ppid": int(stat[1]),
                    "start": stat[19], "run_id": None}
            try:
                for entry in path.joinpath("environ").read_bytes().split(b"\0"):
                    if entry.startswith((RUN_ENV + "=").encode()):
                        item["run_id"] = entry.split(b"=", 1)[1].decode()
                        break
            except (OSError, UnicodeError):
                pass
            if item["state"] not in ("Z", "X", "x"):
                result[int(path.name)] = item
        except (OSError, UnicodeError, ValueError, IndexError):
            pass
    return result


def inference_processes(run_id=None):
    ps = processes()
    if run_id:
        selected = {pid for pid, item in ps.items() if item["run_id"] == run_id}
    else:
        selected = set()
        for pid, item in ps.items():
            args = item["args"]
            titled = args and args[0].startswith(("VLLM::", "vllm::"))
            cli = any(pathlib.PurePosixPath(arg).name == "vllm"
                      and i + 1 < len(args) and args[i + 1] == "serve"
                      for i, arg in enumerate(args))
            module = "-m" in args and any(
                arg.startswith("vllm.entrypoints.") for arg in args)
            if titled or cli or module or item["run_id"]:
                selected.add(pid)
    # Include descendants of both original API roots and orphaned worker roots.
    while True:
        children = {pid for pid, item in ps.items() if item["ppid"] in selected}
        if children.issubset(selected):
            break
        selected |= children
    return {pid: ps[pid] for pid in selected}


def signal_process(pid, expected_start, sig):
    """Signal a verified process identity, never a persisted PID alone."""
    fd = None
    try:
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("Linux pidfd support is required for safe lifecycle signals")
        fd = os.pidfd_open(pid)
        current = processes().get(pid)
        if current and current["start"] == expected_start:
            signal.pidfd_send_signal(fd, sig)
    except ProcessLookupError:
        pass
    finally:
        if fd is not None:
            os.close(fd)


def internal_stop(run_id=None):
    signaled = set()
    initial = inference_processes(run_id)
    grace_end = time.monotonic() + 20
    kill_end = grace_end + 8
    quiet_since = None
    while True:
        current = inference_processes(run_id)
        now = time.monotonic()
        if not current:
            quiet_since = quiet_since or now
            # Catch processes that were in flight when the launcher was stopped.
            if now - quiet_since >= 1.5:
                print(json.dumps({"terminated_pids": sorted(initial), "remaining": {}}))
                return 0
        else:
            quiet_since = None
            sig = signal.SIGTERM if now < grace_end else signal.SIGKILL
            for pid, item in current.items():
                key = (pid, item["start"], sig)
                if key not in signaled:
                    signal_process(pid, item["start"], sig)
                    signaled.add(key)
        if now >= kill_end:
            remaining = inference_processes(run_id)
            print(json.dumps({"terminated_pids": sorted(initial), "remaining": remaining}))
            return 1 if remaining else 0
        time.sleep(0.25)


def command(config, rank, run_id=None):
    node = config["nodes"][rank]
    env = dict(config["environment"])
    env.update(VLLM_HOST_IP=node["rdma_ip"], NCCL_SOCKET_IFNAME=node["iface"],
               GLOO_SOCKET_IFNAME=node["iface"])
    if run_id:
        env[RUN_ENV] = run_id
    cmd = ["podman", "exec"]
    for key, value in env.items():
        cmd += ["-e", f"{key}={value}"]
    cmd += [node["container"], "vllm", "serve", config["model"],
            "--host", "0.0.0.0", "--port", str(config["api_port"]),
            "--tensor-parallel-size", str(len(config["nodes"])),
            "--nnodes", str(len(config["nodes"])), "--node-rank", str(rank),
            "--master-addr", config["master_addr"], "--master-port", str(config["master_port"]),
            "--distributed-executor-backend", "mp",
            "--compilation-config", json.dumps(config["compilation_config"]),
            "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0}),
            "--max-model-len", str(config["max_model_len"]),
            "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
            "--async-scheduling", "--max-num-batched-tokens", str(config["max_num_batched_tokens"]),
            "--max-num-seqs", str(config["max_num_seqs"])]
    if config.get("speculative_config"):
        cmd += ["--speculative-config", json.dumps(config["speculative_config"])]
    if config.get("enable_expert_parallel", False):
        cmd += ["--enable-expert-parallel"]
    if config.get("profiler_config"):
        cmd += ["--profiler-config", json.dumps(config["profiler_config"])]
    if rank == 0 and config.get("enable_auto_tool_choice", False):
        parser = config.get("tool_call_parser")
        if not parser:
            raise ValueError("Auto tool choice requires tool_call_parser")
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", parser]
    if rank:
        cmd += ["--headless"]
    return cmd


def read_state(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def write_state(path, state):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(state, indent=2))
    os.replace(temp, path)


def launcher_alive(state):
    if not state.get("host_start") or not state.get("host_pid"):
        return False
    current = processes().get(state["host_pid"])
    return bool(current and current["start"] == state["host_start"])


def stop_launcher(state):
    if not launcher_alive(state):
        return
    for sig, wait_seconds in ((signal.SIGTERM, 2), (signal.SIGKILL, 3)):
        if launcher_alive(state):
            signal_process(state["host_pid"], state["host_start"], sig)
        deadline = time.monotonic() + wait_seconds
        while launcher_alive(state) and time.monotonic() < deadline:
            time.sleep(0.1)
    if launcher_alive(state):
        raise RuntimeError("Launcher remains alive after stop")


@contextlib.contextmanager
def lifecycle_lock(logs, rank):
    with (logs / f"rank{rank}.lock").open("a") as lock:
        # A second coordinator must not queue a stale lifecycle action.
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another lifecycle action is in progress")
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def internal_call(node, action, args):
    argv = ["podman", "exec", node["container"], "python3",
            str(ROOT / "tools/node_runtime.py"), action,
            "--config", args.config, "--rank", str(args.rank)]
    if args.run_id:
        argv += ["--run-id", args.run_id]
    if getattr(args, "expected_config_sha256", None):
        argv += ["--expected-config-sha256", args.expected_config_sha256]
    p = subprocess.run(argv, capture_output=True, text=True, timeout=40)
    if p.returncode:
        raise RuntimeError(f"{action} failed ({p.returncode}): {p.stdout.strip()} {p.stderr.strip()}")
    return json.loads(p.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "stop", "cleanup", "status", "validate", "preflight",
                    "internal-stop", "internal-status", "internal-preflight", "dry-run"])
    ap.add_argument("--config", default=str(ROOT / "config/cluster.json"))
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--tag", default="cluster")
    ap.add_argument("--run-id")
    ap.add_argument("--expected-config-sha256")
    args = ap.parse_args()
    if args.run_id and (len(args.run_id) > 100 or not args.run_id.replace("-", "").isalnum()):
        raise ValueError("Invalid run ID")
    if args.action == "internal-stop":
        return internal_stop(args.run_id)
    if args.action == "internal-status":
        print(json.dumps(inference_processes(args.run_id)))
        return 0
    config = json.loads(pathlib.Path(args.config).read_text())
    config_sha256 = verify_config(config, args.expected_config_sha256)
    if not 0 <= args.rank < len(config["nodes"]):
        raise ValueError("Invalid rank")
    node = config["nodes"][args.rank]
    if args.action == "internal-preflight":
        # Always inspect every inference process, including previous runs.
        active = inference_processes()
        errors = []
        if not shutil.which("vllm"):
            errors.append("vllm executable missing")
        model = config["model"]
        if model.startswith("/") and not pathlib.Path(model).is_dir():
            errors.append("Local model directory missing")
        command(config, args.rank)  # Validate required launch fields before any launch.
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            errors.append("Linux pidfd support is required for safe stop")
        print(json.dumps({"processes": active, "errors": errors, "config_sha256": config_sha256}))
        return 0
    if args.action == "dry-run":
        print(json.dumps(command(config, args.rank, args.run_id)))
        return 0
    if args.action == "validate":
        check = internal_call(node, "internal-preflight", args)
        if check["errors"]:
            raise RuntimeError("; ".join(check["errors"]))
        print(json.dumps({"valid": True, "rank": args.rank, "config_sha256": config_sha256}))
        return 0
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    state_path = logs / f"rank{args.rank}.json"
    if args.action == "status":
        state = read_state(state_path)
        payload = {"processes": internal_call(node, "internal-status", args),
                   "launcher_alive": launcher_alive(state),
                   "run_id": state.get("run_id"), "host_pid": state.get("host_pid"),
                   "log": state.get("log")}
        print(json.dumps(payload))
        return 0
    with lifecycle_lock(logs, args.rank):
        state = read_state(state_path)
        if args.action in ("stop", "cleanup"):
            if args.action == "cleanup" and not args.run_id:
                raise ValueError("cleanup requires a run ID")
            if args.action == "stop" and args.run_id:
                raise ValueError("Use cleanup for a specific run")
            owned_launcher = (args.action == "stop" or state.get("run_id") == args.run_id)
            launcher_error = None
            if owned_launcher and launcher_alive(state):
                # Prevent a pending exec from creating new processes during cleanup.
                try:
                    stop_launcher(state)
                except Exception as error:
                    launcher_error = error
            result = internal_call(node, "internal-stop", args)
            if result.get("remaining"):
                raise RuntimeError("Inference processes remain after stop")
            if launcher_error and launcher_alive(state):
                raise launcher_error
            if owned_launcher:
                state["stopped_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                write_state(state_path, state)
            print(json.dumps(result))
            return 0
        check = internal_call(node, "internal-preflight", args)
        if check["processes"] or launcher_alive(state):
            raise RuntimeError("Inference or a pending launcher already running; use restart")
        if check["errors"]:
            raise RuntimeError("; ".join(check["errors"]))
        if args.action == "preflight":
            print(json.dumps({"ready": True, "rank": args.rank}))
            return 0
        if not args.run_id:
            raise ValueError("start requires a coordinator-generated run ID")
        if not args.tag.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid tag")
        cmd = command(config, args.rank, args.run_id)
        log = logs / f"{args.tag}-{args.run_id}-rank{args.rank}.log"
        child = None
        try:
            with log.open("ab", buffering=0) as output:
                output.write(("\nSTART " + datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n").encode())
                child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=output,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            current = processes().get(child.pid)
            state = {"host_pid": child.pid, "host_start": current["start"] if current else None,
                     "rank": args.rank, "run_id": args.run_id, "log": str(log),
                     "command": cmd, "config": config}
            # Keep each attempt's exact launch config after rankN.json advances
            # to a later run, including configurations from failed launches.
            write_state(log.with_suffix(".json"), state)
            write_state(state_path, state)
            if not state["host_start"]:
                raise RuntimeError(f"Cannot establish launcher identity; inspect {log}")
            time.sleep(0.3)
            if child.poll() is not None:
                raise RuntimeError(f"Launcher exited with code {child.returncode}; inspect {log}")
            print(json.dumps({"started_pid": child.pid, "run_id": args.run_id, "log": str(log)}))
            return 0
        except BaseException:
            # Only this launch carries the new token. Earlier services are excluded.
            try:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=3)
            except Exception as launcher_error:
                print(f"Local launcher cleanup failed: {launcher_error}", file=sys.stderr)
            try:
                internal_call(node, "internal-stop", args)
            except Exception as cleanup_error:
                print(f"Local launch cleanup failed: {cleanup_error}", file=sys.stderr)
            raise


if __name__ == "__main__":
    sys.exit(main())
