#!/usr/bin/env bash
# A/B benchmark: RCCL over RoCE (RDMA) vs. forced TCP sockets for one
# model+profile. Wraps bench/run_matrix.py with --concurrencies 1,16 and
# prints the per-concurrency delta at the end.
#
# Usage: bench/compare_eth_vs_rdma.sh <model> <profile> [prompt_lens]
# Example: bench/compare_eth_vs_rdma.sh qwen36-35b-a3b tp4 512,4096
#
# The NCCL_* env vars exported below are inherited by run_matrix.py and
# (assumed contract) forwarded by scripts/serve.sh to the vLLM server.
set -euo pipefail

MODEL=${1:?usage: compare_eth_vs_rdma.sh <model> <profile> [prompt_lens]}
PROFILE=${2:?usage: compare_eth_vs_rdma.sh <model> <profile> [prompt_lens]}
PROMPT_LENS=${3:-512,4096}

TS=$(date -u +%Y%m%dT%H%M%SZ)
RDMA_OUT="bench/results/${TS}_${MODEL}_${PROFILE}_rdma.json"
TCP_OUT="bench/results/${TS}_${MODEL}_${PROFILE}_tcp.json"

# Preflight: podman exec does NOT inherit the client environment, so these
# vars only reach vLLM if scripts/serve.sh forwards them via its -e list.
# serve.sh hardcodes NCCL_IB_GID_INDEX=1 (fine for run A) but must also pass
# NCCL_IB_DISABLE through for run B to actually switch to TCP sockets.
if ! grep -q 'NCCL_IB_DISABLE' scripts/serve.sh; then
    echo "ERROR: scripts/serve.sh does not forward NCCL_IB_DISABLE into the" >&2
    echo "container - run B would silently use RDMA again. Add NCCL_IB_DISABLE" >&2
    echo "to the forwarded 'for kv in ...' list in serve.sh, then re-run." >&2
    exit 1
fi

echo "=== Run A: RCCL over RoCE (RDMA) ==="
# NCCL_IB_GID_INDEX=1 is required for RoCEv2 on this setup.
NCCL_IB_GID_INDEX=1 NCCL_IB_DISABLE=0 \
python3 bench/run_matrix.py \
    --models "$MODEL" --profiles "$PROFILE" \
    --concurrencies 1,16 --prompt-lengths "$PROMPT_LENS" \
    --output "$RDMA_OUT"

echo "=== Run B: RCCL forced to TCP sockets ==="
# NCCL_IB_DISABLE=1 makes RCCL fall back to sockets. NCCL_SOCKET_IFNAME must
# select the 25GbE interface for a fair comparison; leave it to the cluster
# default env unless you know it must differ.
NCCL_IB_DISABLE=1 \
python3 bench/run_matrix.py \
    --models "$MODEL" --profiles "$PROFILE" \
    --concurrencies 1,16 --prompt-lengths "$PROMPT_LENS" \
    --output "$TCP_OUT"

echo
echo "=== Delta (RDMA vs TCP, aggregate output tok/s) ==="
python3 - "$RDMA_OUT" "$TCP_OUT" <<'PY'
import json, sys

def load(path):
    cells = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r or r.get("output_toks") is None:
                continue
            cells[(r["prompt_len"], r["concurrency"])] = r["output_toks"]
    return cells

rdma, tcp = load(sys.argv[1]), load(sys.argv[2])
print(f"{'prompt':>8} {'concurrency':>12} {'RDMA tok/s':>12} {'TCP tok/s':>12} {'delta':>8}")
for key in sorted(set(rdma) | set(tcp)):
    a, b = rdma.get(key), tcp.get(key)
    if a is not None and b is not None and b > 0:
        delta = f"{(a - b) / b * 100:+.1f}%"
    else:
        delta = "n/a"
    fmt = lambda v: f"{v:12.1f}" if v is not None else f"{'-':>12}"
    print(f"{key[0]:>8} {key[1]:>12} {fmt(a)} {fmt(b)} {delta:>8}")
PY
