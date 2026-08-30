#!/usr/bin/env bash
# cluster_up.sh — idempotent Ray head/worker bring-up across the 4 nodes.
#
# Node list comes from ansible/inventory.yaml (via scripts/lib/registry.py);
# the first host (node1) becomes the Ray head. Each node gets a detached
# runtime container (ray-head / ray-worker) running `ray start --block`.
# serve.sh later execs `vllm serve` into the head container.
#
# Pre-flight per node (SSH): memlock ulimit unlimited (else RCCL fails with
# ibv_reg_mr_iova2 ... Cannot allocate memory), /dev/infiniband present,
# ibv_devices lists at least one RDMA device.
#
# Idempotent: running containers are left alone, stopped ones are started,
# missing ones are created.
#
# Usage: cluster_up.sh [inventory.yaml]
# Env:   VLLM_IMAGE, SSH_OPTS (extra ssh options)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="${PYTHON:-python3}"
INVENTORY="${1:-$REPO_ROOT/ansible/inventory.yaml}"
IMAGE="${VLLM_IMAGE:-ghcr.io/maikzz32/strix-vllm-gfx1151:latest}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=5}"

NAMES=(); TARGETS=(); IPS=()
while read -r name host user; do
  NAMES+=("$name")
  IPS+=("$host")
  if [ -n "$user" ]; then TARGETS+=("$user@$host"); else TARGETS+=("$host"); fi
done < <("$PY" "$SCRIPT_DIR/lib/registry.py" nodes "$INVENTORY")

HEAD_IDX=0
HEAD_NAME="${NAMES[$HEAD_IDX]}"
HEAD_IP="${IPS[$HEAD_IDX]}"
EXPECTED=${#NAMES[@]}
echo "cluster_up: head=$HEAD_NAME ($HEAD_IP), nodes: ${NAMES[*]}"

ssh_node() { # ssh_node <idx> <cmd...>
  local idx="$1"; shift
  # shellcheck disable=SC2086
  ssh $SSH_OPTS "${TARGETS[$idx]}" "$@"
}

# --- pre-flight ---------------------------------------------------------------
for i in "${!NAMES[@]}"; do
  echo "cluster_up: pre-flight ${NAMES[$i]}"
  if ! ssh_node "$i" 'test "$(ulimit -l)" = unlimited' 2>/dev/null; then
    echo "cluster_up: ERROR ${NAMES[$i]}: memlock ulimit is not unlimited." >&2
    echo "  Fix: /etc/security/limits.d/99-memlock.conf ('* - memlock unlimited')" >&2
    echo "  Without it RCCL aborts with: ibv_reg_mr_iova2 ... Cannot allocate memory" >&2
    exit 1
  fi
  if ! ssh_node "$i" 'test -d /dev/infiniband' 2>/dev/null; then
    echo "cluster_up: ERROR ${NAMES[$i]}: /dev/infiniband missing (RDMA driver not loaded?)" >&2
    exit 1
  fi
  if ! ssh_node "$i" 'command -v ibv_devices >/dev/null && ibv_devices 2>/dev/null | tail -n +3 | grep -q .' 2>/dev/null; then
    echo "cluster_up: ERROR ${NAMES[$i]}: ibv_devices lists no RDMA device" >&2
    exit 1
  fi
done

# --- container start (idempotent) ----------------------------------------------
# Container contract (see docker/): --network host, --device /dev/infiniband,
# --ulimit memlock=-1. /dev/kfd + /dev/dri expose the iGPU, --ipc=host gives
# Ray/vLLM enough shared memory.
start_container() { # start_container <idx> <name> <ray-start-args...>
  local idx="$1" cname="$2"; shift 2
  if ssh_node "$idx" "podman ps --format '{{.Names}}' | grep -qx '$cname'" 2>/dev/null; then
    echo "cluster_up: $cname already running on ${NAMES[$idx]}"
    return 0
  fi
  if ssh_node "$idx" "podman container exists '$cname'" 2>/dev/null; then
    echo "cluster_up: starting stopped container $cname on ${NAMES[$idx]}"
    ssh_node "$idx" "podman start '$cname'"
    return 0
  fi
  echo "cluster_up: creating $cname on ${NAMES[$idx]} ($IMAGE)"
  ssh_node "$idx" "podman run -d --name '$cname' \
    --network host --ipc host \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband \
    --ulimit memlock=-1 \
    -e TRITON_CACHE_DIR=/triton-cache -v triton-cache:/triton-cache \
    '$IMAGE' ray start $* --block"
}

start_container "$HEAD_IDX" ray-head --head --port 6379
for i in "${!NAMES[@]}"; do
  [ "$i" -eq "$HEAD_IDX" ] && continue
  start_container "$i" ray-worker --address "$HEAD_IP:6379"
done

# --- verify ---------------------------------------------------------------------
echo "cluster_up: waiting for ray status..."
deadline=$((SECONDS + 120))
until ssh_node "$HEAD_IDX" 'podman exec ray-head ray status' >/tmp/ray-status.$$ 2>/dev/null; do
  [ $SECONDS -lt $deadline ] || { echo "cluster_up: ray status timed out" >&2; rm -f /tmp/ray-status.$$; exit 1; }
  sleep 3
done
ACTIVE=$(grep -c '^ [0-9]* node_' /tmp/ray-status.$$ || true)
cat /tmp/ray-status.$$
rm -f /tmp/ray-status.$$
if [ "$ACTIVE" -lt "$EXPECTED" ]; then
  echo "cluster_up: WARNING: only $ACTIVE/$EXPECTED ray nodes active — check 'podman logs ray-worker' on the missing nodes" >&2
else
  echo "cluster_up: OK, $ACTIVE/$EXPECTED nodes active"
fi
