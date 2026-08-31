#!/usr/bin/env bash
# cluster_down.sh — idempotent teardown of the Ray cluster on all nodes.
#
# Removes the ray-worker / ray-head containers created by cluster_up.sh
# (workers first, head last). Safe to run repeatedly; missing containers are
# skipped. Does not remove the triton-cache volume (persisted JIT cache).
#
# Usage: cluster_down.sh [inventory.yaml]
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

stop_container() { # stop_container <idx> <name>
  local idx="$1" cname="$2"
  if ssh_node "$idx" "podman container exists '$cname'" 2>/dev/null; then
    echo "cluster_down: removing $cname on ${NAMES[$idx]}"
    ssh_node "$idx" "podman rm -f '$cname'" || true
  else
    echo "cluster_down: $cname on ${NAMES[$idx]} already gone"
  fi
}

# Workers first, head (index 0) last.
for ((i=${#NAMES[@]}-1; i>=1; i--)); do
  stop_container "$i" ray-worker
done
stop_container 0 ray-head
echo "cluster_down: done"
