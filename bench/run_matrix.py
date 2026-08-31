#!/usr/bin/env python3
"""Benchmark matrix runner for the 4-node Strix Halo vLLM cluster.

Runs ON node1, either inside the cluster container or on the host against it
(the server is reachable at BENCH_BASE_URL either way, host network).

Dependencies: python3 stdlib + pyyaml + requests only.

For each (model, profile) cell it:
  1. starts the server via `scripts/serve.sh <model> <profile>` (subprocess),
  2. waits for /health (default timeout 20 min - first boot JITs Triton ~170 s),
  3. benchmarks each (prompt_len, concurrency) combination, preferring
     `vllm bench serve` and falling back to a raw OpenAI-compatible load
     generator (requests + threads) when the bench CLI is unavailable,
  4. tears the server down (pkill on all nodes - see TODO below),
  5. appends one JSON record per cell (JSONL) to bench/results/<timestamp>.json.

Resumable: pass --resume <file> to skip cells already present in that file.

Assumed contracts (owned by other parts of the repo):
  - scripts/serve.sh <model> <profile>  starts vLLM serving the registry model
    on the cluster and exports/forwards the current environment (NCCL_* etc.)
    to the server. It may run in the foreground or daemonize; both work here.
  - The OpenAI-compatible endpoint is reachable at BENCH_BASE_URL
    (default http://127.0.0.1:8000).

TODO: once scripts/cluster_down.sh exists, use it in teardown() instead of
the pkill fallback below (cluster_down.sh is owned by another agent).
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / "models" / "registry.yaml"
SERVE_SH = REPO_ROOT / "scripts" / "serve.sh"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

ALL_PROFILES = ["tp2", "tp4", "pp4", "tp2pp2", "ep", "solo"]
DEFAULT_CONCURRENCIES = [1, 4, 8, 16, 32]
DEFAULT_PROMPT_LENS = [512, 4096, 32768]
DEFAULT_OUTPUT_LEN = 128
HEALTH_TIMEOUT_S = 20 * 60  # first boot JITs Triton for ~170 s; be generous


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(value, cast=str):
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def worker_nodes():
    """Worker nodes for teardown. Override with BENCH_WORKER_NODES.

    Assumed hostnames node2..node4 (node1 is local); adjust via env if the
    inventory uses different names.
    """
    return os.environ.get("BENCH_WORKER_NODES", "node2 node3 node4").split()


def load_registry():
    with open(REGISTRY, encoding="utf-8") as f:
        return yaml.safe_load(f)


def start_server(model, profile, log_path):
    """Start scripts/serve.sh; returns (Popen, log file handle)."""
    if not SERVE_SH.exists():
        raise FileNotFoundError(f"{SERVE_SH} not found - serve.sh is provided elsewhere in the repo")
    logf = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", str(SERVE_SH), model, profile],
        cwd=REPO_ROOT,
        env=os.environ.copy(),  # NCCL_* / VLLM_* env propagates to serve.sh
        stdout=logf,
        stderr=subprocess.STDOUT,
    )
    return proc, logf


def wait_healthy(base_url, proc, timeout_s=HEALTH_TIMEOUT_S):
    """Poll /health until 200 or timeout. Aborts early if serve.sh died."""
    deadline = time.monotonic() + timeout_s
    url = f"{base_url}/health"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"serve.sh exited early with code {proc.returncode} (see serve log)")
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError(f"server not healthy after {timeout_s}s ({url})")


def teardown(proc=None, logf=None):
    """Kill every vllm process on all nodes. Never raise.

    TODO: replace with `scripts/cluster_down.sh` once it exists - pkill only
    matches 'vllm serve' and may miss Ray-spawned engine workers on multi-node
    profiles that live under a different cmdline.
    """
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    if logf is not None:
        logf.close()
    try:
        # [v] trick: a plain 'vllm serve' pattern matches the pkill command
        # line itself (and remote ssh shells carrying it), killing the session.
        subprocess.run(["pkill", "-f", "[v]llm serve"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        pass  # no procps on this host; serve.sh subprocess was already terminated
    for node in worker_nodes():
        try:
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                 node, "pkill -f '[v]llm serve' || true"],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass  # no ssh client on this host


# ---------------------------------------------------------------------------
# Benchmark backends
# ---------------------------------------------------------------------------

def _bench_metrics_from_vllm_stdout(text):
    def grab(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    return {
        "ttft_ms": grab(r"Mean TTFT \(ms\):\s*([\d.]+)"),
        "itl_ms": grab(r"Mean ITL \(ms\):\s*([\d.]+)"),
        "output_toks": grab(r"Output token throughput \(tok/s\):\s*([\d.]+)"),
    }


def run_vllm_bench(base_url, hf_repo, prompt_len, output_len, concurrency, num_prompts):
    """Run `vllm bench serve` with a random dataset; returns metrics or None."""
    if shutil.which("vllm") is None:
        return None
    parsed = urllib.parse.urlparse(base_url)
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--dataset-name", "random",
        "--random-input-len", str(prompt_len),
        "--random-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--ignore-eos",
        "--model", hf_repo,
        "--host", parsed.hostname or "127.0.0.1",
        "--port", str(parsed.port or 8000),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if res.returncode != 0:
        log(f"  vllm bench serve failed (rc={res.returncode}), falling back to raw generator")
        return None
    metrics = _bench_metrics_from_vllm_stdout(res.stdout)
    if metrics["output_toks"] is None:
        log("  could not parse vllm bench output, falling back to raw generator")
        return None
    return metrics


def run_fallback_bench(base_url, hf_repo, prompt_len, output_len, concurrency, num_prompts):
    """Raw OpenAI-compatible load generator (streaming /v1/completions).

    No tokenizer is available here, so the prompt length is approximated by
    repeating a fixed phrase (~5 chars/token is a rough English average).
    """
    phrase = "lorem ipsum dolor sit amet "
    prompt = phrase * max(1, (prompt_len * 5) // len(phrase))
    payload = {
        "model": hf_repo,
        "prompt": prompt,
        "max_tokens": output_len,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    def one_request():
        t0 = time.perf_counter()
        ttft, itls, n_chunks, completion_tokens = None, [], 0, None
        with requests.post(f"{base_url}/v1/completions", json=payload,
                           stream=True, timeout=3600) as r:
            r.raise_for_status()
            last = t0
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                now = time.perf_counter()
                obj = json.loads(data)
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    completion_tokens = usage["completion_tokens"]
                    continue
                choices = obj.get("choices") or []
                if not choices or choices[0].get("text") is None:
                    continue
                if ttft is None:
                    ttft = now - t0
                else:
                    itls.append(now - last)
                last = now
                n_chunks += 1
        total = time.perf_counter() - t0
        tokens = completion_tokens if completion_tokens is not None else n_chunks
        return (ttft if ttft is not None else total), itls, tokens, total

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda _: one_request(), range(num_prompts)))
    wall = time.perf_counter() - t_start

    ttfts = [r[0] for r in results]
    itls = [d for r in results for d in r[1]]
    total_tokens = sum(r[2] for r in results)
    return {
        "ttft_ms": (sum(ttfts) / len(ttfts)) * 1000 if ttfts else None,
        "itl_ms": (sum(itls) / len(itls)) * 1000 if itls else None,
        # aggregate output tok/s across the whole batch; for concurrency==1
        # (serial requests) this equals the single-stream decode rate
        "output_toks": total_tokens / wall if wall > 0 else None,
    }


def run_cell(base_url, hf_repo, prompt_len, output_len, concurrency):
    num_prompts = max(8, concurrency * 4)
    metrics = run_vllm_bench(base_url, hf_repo, prompt_len, output_len,
                             concurrency, num_prompts)
    tool = "vllm-bench"
    if metrics is None:
        metrics = run_fallback_bench(base_url, hf_repo, prompt_len, output_len,
                                     concurrency, num_prompts)
        tool = "fallback"
    return {
        "prompt_len": prompt_len,
        "output_len": output_len,
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "tool": tool,
        "ttft_ms": round(metrics["ttft_ms"], 1) if metrics["ttft_ms"] is not None else None,
        "itl_ms": round(metrics["itl_ms"], 1) if metrics["itl_ms"] is not None else None,
        "output_toks": round(metrics["output_toks"], 2) if metrics["output_toks"] is not None else None,
    }


# ---------------------------------------------------------------------------
# Resume handling
# ---------------------------------------------------------------------------

def cell_key(rec):
    return (rec.get("model"), rec.get("profile"),
            rec.get("prompt_len"), rec.get("concurrency"))


def load_resume_keys(path):
    keys = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in rec:
                keys.add(cell_key(rec))
    return keys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Run the model x profile x prompt-length x concurrency benchmark matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--models", default=None,
                    help="comma-separated registry model keys; default: all non-blocked models")
    ap.add_argument("--profiles", default=None,
                    help="comma-separated profiles; default: the model's allowed_profiles")
    ap.add_argument("--concurrencies", default=",".join(map(str, DEFAULT_CONCURRENCIES)),
                    help="comma-separated concurrency levels")
    ap.add_argument("--prompt-lengths", default=",".join(map(str, DEFAULT_PROMPT_LENS)),
                    help="comma-separated input lengths in tokens")
    ap.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN,
                    help="generated tokens per request")
    ap.add_argument("--base-url", default=os.environ.get("BENCH_BASE_URL", "http://127.0.0.1:8000"),
                    help="OpenAI-compatible endpoint (env BENCH_BASE_URL)")
    ap.add_argument("--health-timeout", type=int, default=HEALTH_TIMEOUT_S,
                    help="seconds to wait for /health per (model, profile)")
    ap.add_argument("--resume", default=None, metavar="RESULTS_FILE",
                    help="skip cells already present (without error) in this results file")
    ap.add_argument("--output", default=None, metavar="RESULTS_FILE",
                    help="results file; default bench/results/<utc-timestamp>.json")
    args = ap.parse_args()

    registry = load_registry()
    all_models = registry["models"]

    if args.models:
        models = parse_csv(args.models)
        unknown = [m for m in models if m not in all_models]
        if unknown:
            ap.error(f"unknown model(s) in registry.yaml: {', '.join(unknown)}")
        blocked = [m for m in models if all_models[m].get("status") == "blocked"]
        if blocked:
            log(f"WARNING: {', '.join(blocked)} marked 'blocked' in registry - expect failure")
    else:
        # blocked models cannot serve upstream; dev models are included
        # (the :dev image channel exists for them)
        models = [k for k, v in all_models.items() if v.get("status") != "blocked"]
    if not models:
        ap.error("no models selected")

    concurrencies = parse_csv(args.concurrencies, int)
    prompt_lens = parse_csv(args.prompt_lengths, int)

    out_path = Path(args.output) if args.output else (
        RESULTS_DIR / f"{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"results -> {out_path}")

    resume_keys = set()
    if args.resume:
        resume_keys = load_resume_keys(args.resume)
        log(f"resuming: {len(resume_keys)} completed cells loaded from {args.resume}")

    out_f = open(out_path, "a", encoding="utf-8")

    def emit(rec):
        rec["ts"] = f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()

    current_proc, current_logf = None, None
    try:
        for model in models:
            entry = all_models[model]
            profiles = (parse_csv(args.profiles) if args.profiles
                        else entry.get("allowed_profiles") or ALL_PROFILES)
            bad = [p for p in profiles if p not in ALL_PROFILES]
            if bad:
                log(f"WARNING: skipping unknown profile(s) for {model}: {', '.join(bad)}")
                profiles = [p for p in profiles if p in ALL_PROFILES]

            for profile in profiles:
                # skip fully-resumed (model, profile) pairs
                todo = [(pl, c) for pl in prompt_lens for c in concurrencies
                        if (model, profile, pl, c) not in resume_keys]
                if not todo:
                    log(f"SKIP {model}/{profile}: all cells already done")
                    continue

                log(f"=== {model} / {profile} ({len(todo)} cells) ===")
                serve_log = out_path.parent / f"serve_{out_path.stem}_{model}_{profile}.log"
                try:
                    current_proc, current_logf = start_server(model, profile, serve_log)
                    wait_healthy(args.base_url, current_proc, args.health_timeout)
                except (FileNotFoundError, RuntimeError, TimeoutError) as e:
                    log(f"ERROR starting {model}/{profile}: {e}")
                    emit({"model": model, "profile": profile,
                          "prompt_len": None, "concurrency": None, "error": str(e)})
                    teardown(current_proc, current_logf)
                    current_proc, current_logf = None, None
                    continue

                log(f"healthy, benchmarking (serve log: {serve_log})")
                for prompt_len in prompt_lens:
                    for concurrency in concurrencies:
                        if (model, profile, prompt_len, concurrency) in resume_keys:
                            continue
                        log(f"  cell prompt={prompt_len} concurrency={concurrency}")
                        rec = {"model": model, "hf_repo": entry["hf_repo"],
                               "profile": profile}
                        try:
                            rec.update(run_cell(args.base_url, entry["hf_repo"],
                                                prompt_len, args.output_len, concurrency))
                            log(f"    -> {rec['output_toks']} tok/s agg, "
                                f"TTFT {rec['ttft_ms']} ms, ITL {rec['itl_ms']} ms [{rec['tool']}]")
                        except Exception as e:  # record and continue the matrix
                            log(f"    ERROR: {e}")
                            rec.update({"prompt_len": prompt_len, "concurrency": concurrency,
                                        "error": str(e)})
                        emit(rec)

                teardown(current_proc, current_logf)
                current_proc, current_logf = None, None
                log(f"=== {model} / {profile} done, server stopped ===")
    finally:
        # guarantee no vllm process is left behind, even on Ctrl-C
        teardown(current_proc, current_logf)
        out_f.close()

    log("matrix complete")


if __name__ == "__main__":
    main()
