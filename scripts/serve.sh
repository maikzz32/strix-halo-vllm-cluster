#!/usr/bin/env bash
# serve.sh <model> <profile> — serve a registry model on the cluster via vLLM.
#
#   model    key in models/registry.yaml (e.g. qwen36-35b-a3b)
#   profile  tp4 | pp4 | tp2pp2 | ep | solo (validated against the model's
#            allowed_profiles)
#
# Run this on the Ray head node (node1). The vllm serve process is started via
# `podman exec` inside the runtime container created by cluster_up.sh
# ($CONTAINER_NAME, default ray-head), so cluster_up.sh must have run first —
# also for profile `solo` (TP=1 simply does not use Ray).
#
# Env sources, in increasing precedence:
#   1. models/registry.yaml  defaults.env merged with the model's env
#      (carries VLLM_ROCM_USE_AITER=0, VLLM_GFX1X_MOE_TUNE=1, ...)
#   2. scripts/defaults.env  optional site-local overrides (VLLM_IMAGE,
#      ROCE_IFACE, ...) — sourced only if the file exists
#   3. distributed env computed here (NCCL_*, VLLM_HOST_IP, RAY_ADDRESS)
#
# Tunables (env): VLLM_IMAGE, CONTAINER_NAME, ROCE_IFACE, NCCL_SOCKET_IFNAME,
# VLLM_HOST_IP, PORT (default 8000).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REG="$SCRIPT_DIR/lib/registry.py"
PY="${PYTHON:-python3}"

usage() { sed -n '2,20p' "$0"; exit "${1:-1}"; }

[ $# -eq 2 ] || usage
MODEL="$1"
PROFILE="$2"

IMAGE="${VLLM_IMAGE:-ghcr.io/maikzz32/strix-vllm-gfx1151:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-ray-head}"
PORT="${PORT:-8000}"

# --- model + profile validation -------------------------------------------
STATUS="$("$PY" "$REG" status "$MODEL")"
ALLOWED="$("$PY" "$REG" allowed_profiles "$MODEL")"
case " $ALLOWED " in
  *" $PROFILE "*) ;;
  *) echo "serve.sh: profile '$PROFILE' not allowed for '$MODEL' (allowed: $ALLOWED)" >&2; exit 1 ;;
esac

if [ "$STATUS" = "blocked" ]; then
  echo "serve.sh: model '$MODEL' is blocked upstream. Tracking:" >&2
  "$PY" "$REG" tracking "$MODEL" | sed 's/^/  - /' >&2
  exit 1
fi
if [ "$STATUS" = "dev" ] && [[ "$IMAGE" != *:dev* ]]; then
  echo "serve.sh: model '$MODEL' has status 'dev' and needs a :dev-channel image" >&2
  echo "  (vLLM main). Set VLLM_IMAGE to a :dev image, current: $IMAGE" >&2
  exit 1
fi

# --- site-local overrides (optional) ---------------------------------------
if [ -f "$SCRIPT_DIR/defaults.env" ]; then
  # shellcheck disable=SC1091
  . "$SCRIPT_DIR/defaults.env"
fi

# --- registry env (defaults.env of the registry merged with model env) ------
ENV_ARGS=()
while IFS= read -r line; do
  [ -n "$line" ] || continue
  export "$line"
  ENV_ARGS+=(-e "$line")
done < <("$PY" "$REG" env "$MODEL")

# --- distributed env (RoCE/RCCL + Ray) --------------------------------------
detect_roce_iface() {  # first RDMA-capable netdev
  local d
  for d in /sys/class/infiniband/*/device/net/*; do
    [ -e "$d" ] && { basename "$d"; return 0; }
  done
  return 1
}

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${ROCE_IFACE:-$(detect_roce_iface || true)}}"
if [ -z "$NCCL_SOCKET_IFNAME" ]; then
  echo "serve.sh: no RDMA netdev found; set ROCE_IFACE or NCCL_SOCKET_IFNAME" >&2
  exit 1
fi
export NCCL_IB_GID_INDEX=1          # RoCEv2
export NCCL_NET_GDR_LEVEL=0         # no GPU-direct RDMA path on gfx1151 iGPU
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
# VLLM_HOST_IP: keep a pre-set value, else take the IPv4 of the RoCE iface.
export VLLM_HOST_IP="${VLLM_HOST_IP:-$(ip -4 -o addr show dev "$NCCL_SOCKET_IFNAME" | awk '{split($4,a,"/"); print a[1]; exit}')}"
if [ -z "$VLLM_HOST_IP" ]; then
  echo "serve.sh: could not determine VLLM_HOST_IP from $NCCL_SOCKET_IFNAME" >&2
  exit 1
fi

for kv in NCCL_SOCKET_IFNAME NCCL_IB_GID_INDEX NCCL_NET_GDR_LEVEL \
          RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES VLLM_HOST_IP; do
  ENV_ARGS+=(-e "$kv=${!kv}")
done
# Optional override, forwarded only when set: bench/compare_eth_vs_rdma.sh uses
# NCCL_IB_DISABLE=1 to force the RCCL TCP-socket leg of the A/B comparison.
if [ -n "${NCCL_IB_DISABLE:-}" ]; then
  ENV_ARGS+=(-e "NCCL_IB_DISABLE=$NCCL_IB_DISABLE")
fi

# --- build the vllm serve command -------------------------------------------
HF_REPO="$("$PY" "$REG" hf_repo "$MODEL")"
CMD=(vllm serve "$HF_REPO" --host 0.0.0.0 --port "$PORT")

# All multi-node profiles need the Ray executor backend (the mp backend cannot
# span nodes); solo stays local. RAY_ADDRESS=auto makes vLLM attach to the
# cluster started by cluster_up.sh instead of spawning a local Ray instance.
case "$PROFILE" in
  tp4)    CMD+=(--tensor-parallel-size 4 --distributed-executor-backend ray) ;;
  pp4)    CMD+=(--pipeline-parallel-size 4 --distributed-executor-backend ray) ;;
  tp2pp2) CMD+=(--tensor-parallel-size 2 --pipeline-parallel-size 2 --distributed-executor-backend ray) ;;
  ep)     CMD+=(--tensor-parallel-size 4 --enable-expert-parallel --distributed-executor-backend ray) ;;
  solo)   CMD+=(--tensor-parallel-size 1) ;;
esac
if [ "$PROFILE" != "solo" ]; then
  ENV_ARGS+=(-e "RAY_ADDRESS=auto")
fi

# Optional parsers from the registry (only when set).
val="$("$PY" "$REG" parser "$MODEL" tool_call_parser)";    [ -z "$val" ] || CMD+=(--tool-call-parser "$val")
val="$("$PY" "$REG" parser "$MODEL" reasoning_parser)";    [ -z "$val" ] || CMD+=(--reasoning-parser "$val")
val="$("$PY" "$REG" parser "$MODEL" tokenizer_mode)";      [ -z "$val" ] || CMD+=(--tokenizer-mode "$val")

# Model extra_args (registry defaults already merged in; includes --enforce-eager,
# mandatory on gfx1151 because HIP graph capture deadlocks, vllm#32180).
EXTRA_ARGS=()
while IFS= read -r line; do
  [ -n "$line" ] || continue
  EXTRA_ARGS+=("$line")
done < <("$PY" "$REG" extra_args "$MODEL")
[ ${#EXTRA_ARGS[@]} -eq 0 ] || CMD+=("${EXTRA_ARGS[@]}")

# --- launch ------------------------------------------------------------------
if ! podman container exists "$CONTAINER_NAME"; then
  echo "serve.sh: container '$CONTAINER_NAME' not found — run scripts/cluster_up.sh first" >&2
  exit 1
fi

echo "serve.sh: $MODEL ($HF_REPO) profile=$PROFILE image=$IMAGE"
echo "serve.sh: iface=$NCCL_SOCKET_IFNAME host_ip=$VLLM_HOST_IP port=$PORT"
echo "serve.sh: ${CMD[*]}"
exec podman exec "${ENV_ARGS[@]}" "$CONTAINER_NAME" "${CMD[@]}"
