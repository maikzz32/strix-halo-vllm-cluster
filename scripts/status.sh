#!/usr/bin/env bash
# status.sh — quick cluster health overview.
#
#   1. Ray status (from the head container, node1)
#   2. Per-node GPU memory usage (rocm-smi)
#   3. Per-node RDMA link state (/sys/class/infiniband)
#
# Read-only; every section degrades to a warning instead of failing the run.
#
# Usage: status.sh [inventory.yaml]
# Env:   SSH_OPTS (extra ssh options)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="${PYTHON:-python3}"
INVENTORY="${1:-$REPO_ROOT/ansible/inventory.yaml}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=5}"

NAMES=(); TARGETS=()
while read -r name host user; do
  NAMES+=("$name")
  if [ -n "$user" ]; then TARGETS+=("$user@$host"); else TARGETS+=("$host"); fi
done < <("$PY" "$SCRIPT_DIR/lib/registry.py" nodes "$INVENTORY" | tr -d '\r')

ssh_node() { # ssh_node <idx> <cmd...>
  local idx="$1"; shift
  # shellcheck disable=SC2086
  ssh $SSH_OPTS "${TARGETS[$idx]}" "$@"
}

echo "=== Ray (head: ${NAMES[0]}) ==="
ssh_node 0 'podman exec ray-head ray status' 2>/dev/null \
  || echo "  ray-head not reachable/running — cluster_up.sh?"

for i in "${!NAMES[@]}"; do
  echo
  echo "=== ${NAMES[$i]} — GPU memory ==="
  # rocm-smi on the host; falls back to the runtime container if absent.
  ssh_node "$i" 'rocm-smi --showmeminfo vram 2>/dev/null || \
    podman exec ray-head rocm-smi --showmeminfo vram 2>/dev/null || \
    podman exec ray-worker rocm-smi --showmeminfo vram 2>/dev/null' \
    || echo "  rocm-smi not available on ${NAMES[$i]}"

  echo "=== ${NAMES[$i]} — RDMA link ==="
  ssh_node "$i" 'for s in /sys/class/infiniband/*/ports/*/state; do
      [ -e "$s" ] || continue
      printf "  %s: %s\n" "$s" "$(cat "$s")"
    done' || echo "  RDMA state unreadable on ${NAMES[$i]}"
done
